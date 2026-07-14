from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, List, Dict
from FunctionTools import GetParametersScale, SiteSpec, BaseOnlineWrapper, AGICoreModule, GrowableLoRALinear, RotaryEmbedding




class SelectiveSSM(AGICoreModule):
    def __init__(
        self,
        embedDim: int,
        stateDim: int = 4, 
        convKernel: int = 4, 
        useCausalConv: bool = True,
        slowDtScale: float = 0.25,):
        super().__init__()
        E = embedDim
        N = stateDim
        self.E = E
        self.N = N
        self.use_causal_conv = bool(useCausalConv)
        self.slow_dt_scale = float(slowDtScale)

        self.A_log = nn.Parameter(torch.randn(E, N))

        self.D = nn.Parameter(torch.zeros(E))

        self.in_proj = nn.Linear(E, 2 * E)

        self.param_proj = nn.Linear(E, E + 2 * E * N)

        self.dt_bias = nn.Parameter(torch.full((E,), -3.0))

        self.dw_conv = nn.Conv1d(E, E, convKernel, groups=E, bias=True)

        self.out_norm = nn.LayerNorm(E)
        self.time_mix_gate = nn.Linear(E, E)

        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.param_proj.weight, gain=0.5)
        nn.init.zeros_(self.param_proj.bias)
        nn.init.zeros_(self.time_mix_gate.weight)
        nn.init.zeros_(self.time_mix_gate.bias)

    def CausalDwconv(self, u: torch.Tensor) -> torch.Tensor:
        B, S, E = u.shape
        x = u.transpose(1, 2) 
        if self.use_causal_conv:
            k = self.dw_conv.kernel_size[0]
            x = F.pad(x, (k - 1, 0)) 
        y = self.dw_conv(x)
        y = y.transpose(1, 2)  
        return y

    def forward(
        self,
        x: torch.Tensor, # [B,S,E]
        tdError: torch.Tensor, # [B]
        uncertainty: torch.Tensor,# [B]
        keyPaddingMask: Optional[torch.Tensor] = None,  # [B,S] bool，True=padding
        ) -> torch.Tensor:
        B, S, E = x.shape
        N = self.N

        mod = torch.sigmoid(0.75 * tdError - 0.75 * uncertainty) 

        u_and_g = self.in_proj(x)
        u, g = torch.chunk(u_and_g, 2, dim=-1)
        u = F.silu(u)

        u = u + 0.5 * self.CausalDwconv(u)

        gate_bias = (0.5 * tdError - 0.5 * uncertainty)[:, None, None]
        g = torch.sigmoid(g + gate_bias) 

        p = self.param_proj(u) 
        dt_raw = p[..., :E]  
        bc = p[..., E:]  
        B_raw, C_raw = bc.split(E * N, dim=-1)

        B_t = torch.tanh(B_raw).view(B, S, E, N)  
        C_t = torch.tanh(C_raw).view(B, S, E, N)

        dt = F.softplus(dt_raw + self.dt_bias.view(1, 1, E))

        dt = dt * (0.75 + 0.50 * mod[:, None, None])

        A_pos = F.softplus(self.A_log) + 1e-4

        y = torch.empty((B, S, E), device=x.device, dtype=x.dtype)
        fast_state = torch.zeros((B, E, N), device=x.device, dtype=x.dtype)
        slow_state = torch.zeros((B, E, N), device=x.device, dtype=x.dtype)

        has_mask = (keyPaddingMask is not None)

        for t in range(S):
            if has_mask:
                keep = (~keyPaddingMask[:, t]).view(B, 1, 1) 
            else:
                keep = None

            u_t = u[:, t, :]
            dt_t = dt[:, t, :]

            decay = torch.exp(-dt_t.unsqueeze(-1) * A_pos.unsqueeze(0))
            inj = B_t[:, t, :, :] * u_t.unsqueeze(-1)

            new_fast = decay * fast_state + (1.0 - decay) * inj

            if keep is not None:
                fast_state = torch.where(keep, new_fast, fast_state)
            else:
                fast_state = new_fast

            slow_dt = dt_t * self.slow_dt_scale
            slow_decay = torch.exp(-slow_dt.unsqueeze(-1) * A_pos.unsqueeze(0))
            new_slow = slow_decay * slow_state + (1.0 - slow_decay) * inj
            if keep is not None:
                slow_state = torch.where(keep, new_slow, slow_state)
            else:
                slow_state = new_slow
            mix = torch.sigmoid(self.time_mix_gate(u_t)).unsqueeze(-1)
            state_for_read = mix * fast_state + (1.0 - mix) * slow_state

            out_t = (C_t[:, t, :, :] * state_for_read).sum(dim=-1) + self.D.unsqueeze(0) * u_t
            out_t = out_t * g[:, t, :]

            if keep is not None:
                out_t = out_t * keep.squeeze(-1)

            y[:, t, :] = out_t

        return self.out_norm(y)

class MultiHeadAttention(AGICoreModule):
    def __init__(
        self, 
        embedDim: int, 
        numHeads: int, 
        hebbianRate: float = 0.01,
        attnDropout: float = 0.1, 
        tdUncScale: float = 1.0, 
        lowRank: bool = True,
        rank: int = 8, 
        useHebbian: bool = True,):
        super().__init__()
        assert embedDim % numHeads == 0, "AttentionModule embed_dim must be divisible by num_heads"
        self.embed_dim = embedDim
        self.num_heads = numHeads
        self.head_dim = embedDim // numHeads
        self.base_hebbian_rate = hebbianRate
        self.attn_dropout_p = attnDropout
        self.td_unc_scale = tdUncScale
        self.rank = min(int(rank), self.head_dim)
        self.use_hebbian = useHebbian
        self.uv_decay = 1e-2 
        self.uv_max_norm = 1.5  
        self.hebb_eps = 1e-6     

        self.use_low_rank = bool(lowRank)

        self.temp_w_td = nn.Parameter(torch.tensor(0.8))
        self.temp_w_unc = nn.Parameter(torch.tensor(0.5))
        self.bias_w_td = nn.Parameter(torch.tensor(0.4))
        self.bias_w_unc = nn.Parameter(torch.tensor(0.6))

        self.q_proj = nn.Linear(embedDim, embedDim)
        self.k_proj = nn.Linear(embedDim, embedDim)
        self.v_proj = nn.Linear(embedDim, embedDim)
        self.out_proj = nn.Linear(embedDim, embedDim)

        self.q_adapter = GrowableLoRALinear(self.q_proj)
        self.k_adapter = GrowableLoRALinear(self.k_proj)
        self.v_adapter = GrowableLoRALinear(self.v_proj)
        self.o_adapter = GrowableLoRALinear(self.out_proj)
        self.rope = RotaryEmbedding(self.head_dim)

        self.register_buffer("hebbian_weights", torch.zeros(1, self.num_heads, self.head_dim, self.head_dim), persistent=True)  # (B,H,D,D)
        self.register_buffer("U", torch.zeros(1, self.num_heads, self.head_dim, self.rank), persistent=True) 
        self.register_buffer("V", torch.zeros(1, self.num_heads, self.head_dim, self.rank), persistent=True)

        self.ResetParameters()

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if self.U.size(0) != B:
            self.hebbian_weights = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
            
            self.U = torch.zeros(B, self.num_heads, self.head_dim, self.rank, device=device, dtype=dtype)
            self.V = torch.zeros(B, self.num_heads, self.head_dim, self.rank, device=device, dtype=dtype)
            
            if self.use_low_rank:
                 self.U.normal_(0.0, 1e-3)
                 self.V.normal_(0.0, 1e-3)

    def ModulateTau(self, tdError, uncertainty, B):
        tau = 1.0 + 0.5 * torch.tanh(self.temp_w_td * tdError + self.temp_w_unc * uncertainty) 
        return tau[:, None, None, None]


    def ResetParameters(self) -> None:
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_normal_(mod.weight, gain=1 / math.sqrt(2))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        self.ResetHebbianMemory()

        with torch.no_grad():
            if self.use_low_rank:
                self.U.normal_(0.0, 1e-3)
                self.V.normal_(0.0, 1e-3)

    def ResetHebbianMemory(self):
        if self.hebbian_weights.numel() > 0:
            self.hebbian_weights.zero_()
            
        if self.U.numel() > 0:
            self.U.zero_()
            self.V.zero_()


    def ScaledDotAttn(
        self,
        q: torch.Tensor, # [B, H, Lq, D]
        k: torch.Tensor, # [B, H, Lk, D]
        v: torch.Tensor, # [B, H, Lk, D]
        mask: Optional[torch.Tensor], # [B, 1, 1, Lk]
        dropoutP: float,) -> Tuple[torch.Tensor, torch.Tensor]:

        q_attn = self.rope.Apply(q)
        k_attn = self.rope.Apply(k)
        d = q_attn.size(-1)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            mask_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(mask, mask_value)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=dropoutP, training=self.training)
        context = torch.matmul(weights, v)
        return context, weights


    def ComputeNeuromodulation(self, tdError: torch.Tensor, B: int) -> torch.Tensor:
        neuromod = 1.0 + 0.5 * tdError
        return neuromod[:, None, None, None]
    
    def ComputeHebbMod(self, tdError: torch.Tensor, uncertainty: torch.Tensor, B: int) -> torch.Tensor:
        mod = ((tdError + 1.0) * 0.5) * (1.0 - uncertainty) # [B]
        return mod[:, None, None, None]


    @torch.no_grad()
    def UpdateHebbianWeights(self, v: torch.Tensor, q: torch.Tensor, alpha_tensor: torch.Tensor, keep4: Optional[torch.Tensor] = None):
        if (not self.use_hebbian):
            return

        v = v.detach()
        q = q.detach()

        if keep4 is not None:
            k = keep4.detach() 
            v = v * k
            q = q * k
            denom = k.sum(dim=-2, keepdim=True).clamp_min(1.0) 
        else:
            S_len = v.size(2)
            denom = S_len

        eps = getattr(self, "hebb_eps", 1e-6)
        v = F.normalize(v, dim=-1, eps=eps)
        q = F.normalize(q, dim=-1, eps=eps)

        hebb = torch.einsum("bhse,bhsd->bhde", v, q) / (denom + 1e-8)
        hebb = torch.tanh(hebb) 

        def clamp_fro(x: torch.Tensor, max_norm: float):
            n = torch.linalg.vector_norm(x, ord=2, dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(max_norm / n, max=1.0)
            return x * scale

        if self.use_low_rank:
            decay = float(getattr(self, "uv_decay", 1e-2))
            maxn = float(getattr(self, "uv_max_norm", 1.5))
            r = int(self.U.size(-1))

            M0 = torch.einsum("bhdr,bher->bhde", self.U, self.V) 

            M1 = (1.0 - decay) * M0 + alpha_tensor * hebb

            M1_f = M1
            U_s, S, Vh = torch.linalg.svd(M1_f, full_matrices=False)

            Sr = S[..., :r].clamp_min(0.0) 
            sqrtSr = Sr.sqrt().unsqueeze(-2) 

            Ur = U_s[..., :r] * sqrtSr 
            Vr = Vh.transpose(-2, -1)[..., :r] * sqrtSr

            Ur = clamp_fro(Ur, maxn)
            Vr = clamp_fro(Vr, maxn)

            self.U.copy_(Ur)
            self.V.copy_(Vr)
        else:
            W_new = (1.0 - alpha_tensor) * self.hebbian_weights + alpha_tensor * hebb
            self.hebbian_weights.copy_(W_new)


    def forward(
        self,
        query,
        key,
        value,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        headGate: Optional[torch.Tensor] = None,
        applyPlasticity: bool = True,):
        B, L, _ = query.shape
        self.EnsureB(B, device=self.device, dtype=self.dtype)

        td = tdError
        unc = uncertainty
        td_eff = td * self.td_unc_scale
        unc_eff = unc * self.td_unc_scale

        neuromod = self.ComputeNeuromodulation(td_eff, B)
        hebb_mod = self.ComputeHebbMod(td_eff, unc_eff, B)

        q_lin = self.q_adapter(query) 
        k_lin = self.k_adapter(key)
        v_lin = self.v_adapter(value) 

        q = q_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if applyPlasticity and self.use_hebbian and self.base_hebbian_rate > 0:
            alpha = self.base_hebbian_rate * hebb_mod
            if keyPaddingMask is not None:
                keep4 = (~keyPaddingMask).view(B, 1, L, 1)
                self.UpdateHebbianWeights(v, q, alpha, keep4=keep4)
            else:
                self.UpdateHebbianWeights(v, q, alpha, keep4=None)

        q = q * neuromod
        if headGate is not None:
            q = q * headGate

        if self.use_low_rank:
            vV = torch.matmul(v, self.V)
            delt = torch.matmul(vV, self.U.transpose(-2, -1))  
        else:
            delt = torch.einsum("bhse,bhde->bhsd", v, self.hebbian_weights)

        delt = delt * hebb_mod
        v_fast = v + delt

        tau = self.ModulateTau(td_eff, unc_eff, B)
        q = q / tau

        q_attn = self.rope.Apply(q)
        k_attn = self.rope.Apply(k)

        d = q_attn.size(-1)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(d)
        if keyPaddingMask is not None:
            mask = keyPaddingMask[:, None, None, :]  # bool [B,1,1,L]
            mask_val = -1e9 if q.dtype != torch.float16 else -1e4
            scores = scores.masked_fill(mask, mask_val)

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.attn_dropout_p if self.training else 0.0, training=self.training)
        context = torch.matmul(weights, v_fast)
        out = context.transpose(1, 2).reshape(B, L, self.embed_dim)

        out_lin = self.o_adapter(out)
        
        return out_lin


    @torch.no_grad()
    def LowrankToFullrank(self):
        W = torch.einsum('bhdr,bher->bhde', self.U, self.V) # [B,H,D,D]
        
        if self.hebbian_weights.shape != W.shape:
             self.hebbian_weights = torch.empty_like(W)
        self.hebbian_weights.copy_(W)
        self.use_low_rank = False
        
    @torch.no_grad()
    def FullrankToLowrank(self):
        M = self.hebbian_weights.clone() # [B,H,D,D]

        U_s, S, Vh = torch.linalg.svd(M, full_matrices=False) 
        r = self.U.size(-1)
        
        Sr = S[..., :r].clamp_min(0.0)
        sqrtSr = Sr.sqrt().unsqueeze(-2)

        Ur = U_s[..., :r] * sqrtSr
        Vr = Vh.transpose(-2, -1)[..., :r] * sqrtSr

        self.U.copy_(Ur)
        self.V.copy_(Vr)
        self.use_low_rank = True


class TemporalAttention(AGICoreModule):
    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        layerIdx: int = 0,
        useHebbian: bool = True,
        slowDtScale: float = 0.25,):
        super().__init__()

        td_unc_scale = 1.0 / (layerIdx + 1)
        self.mhsa = MultiHeadAttention(embedDim, numHeads, tdUncScale=td_unc_scale, useHebbian=useHebbian)
        self.ssm = SelectiveSSM(
            embedDim,
            stateDim=16,
            convKernel=4,
            useCausalConv=True,
            slowDtScale=slowDtScale)

        self.gamma = nn.Parameter(1e-1 * torch.ones(embedDim), requires_grad=True)

        self.mix_gate = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.SiLU(),
            nn.Linear(embedDim, 1))
        
        nn.init.constant_(self.mix_gate[-1].bias, -1.0)

        self.ffn = nn.Sequential(
            nn.Linear(embedDim, 4 * embedDim),
            nn.GELU(),
            nn.Linear(4 * embedDim, embedDim))
        
        self.gamma_ffn = nn.Parameter(1e-1 * torch.ones(embedDim), requires_grad=True)
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embedDim)
        self.norm_ffn = nn.LayerNorm(embedDim)

    def forward(
        self,
        x,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor]=None,
        headGate: Optional[torch.Tensor] = None,
        channelGate: Optional[torch.Tensor] = None,
        applyPlasticity: bool = True,):
        residual = x
        x_norm = self.norm(x)
        
        mhsa_out = self.mhsa(
            x_norm,
            x_norm,
            x_norm,
            keyPaddingMask=keyPaddingMask,
            tdError=tdError,
            uncertainty=uncertainty,
            headGate=headGate,
            applyPlasticity=applyPlasticity)

        ssm_out = self.ssm(x_norm, keyPaddingMask=keyPaddingMask, tdError=tdError, uncertainty=uncertainty)

        w = torch.sigmoid(self.mix_gate(x_norm)) # [B,S,1]

        y = w * mhsa_out + (1 - w) * ssm_out # [B,S,E]
        if channelGate is not None:
            y = y * channelGate

        x = residual + self.dropout(y) * self.gamma

        residual = x
        x_norm = self.norm_ffn(x)

        ffn_out = self.ffn(x_norm)

        out = residual + self.dropout(ffn_out) * self.gamma_ffn

        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1)  # [B,S,1]
            out = out * keep
        return out


class DynamicRouting(AGICoreModule):
    def __init__(self, inCaps: int, inDim: int, outCaps: int, outDim: int, iterations: int = 3):
        super().__init__()
        self.I = inCaps
        self.O = outCaps
        self.in_dim = inDim
        self.out_dim = outDim
        self.iterations = iterations

        self.transformation = nn.Parameter(torch.empty(inCaps, outCaps, inDim, outDim))
        self.routing_logits = nn.Parameter(torch.randn(1, inCaps, outCaps) * 0.01)
        self.ResetParameters()

    def ResetParameters(self):
        nn.init.kaiming_uniform_(self.transformation, a=math.sqrt(5))

    def Squash(self, vectors: torch.Tensor) -> torch.Tensor:
        squared_norm = vectors.pow(2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm) / (torch.sqrt(squared_norm + 1e-8))
        return scale * vectors

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None, ) -> torch.Tensor: 

        B, I, D = x.shape

        u_hat = torch.einsum("bid,iodc->bioc", x, self.transformation)

        logits = self.routing_logits.expand(B, -1, -1).clone()

        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1), -1e9)

        for r in range(self.iterations):
            weights = torch.softmax(logits, dim=-1)  # [B,I,O]

            if mask is not None:
                weights = weights.masked_fill(mask.unsqueeze(-1), 0.0)
                denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                weights = weights / denom

            s = torch.einsum("bioc,bio->boc", u_hat, weights)
            v = self.Squash(s)

            if r < self.iterations - 1:
                agreement = torch.einsum("bioc,boc->bio", u_hat, v).clamp(-5.0, 5.0)
                if mask is not None:
                    agreement = agreement.masked_fill(mask.unsqueeze(-1), 0.0)
                logits = logits + agreement
                if mask is not None:
                    logits = logits.masked_fill(mask.unsqueeze(-1), -1e9)

        return v


class HebbianFusion(AGICoreModule):
    def __init__(self, numModes: int, embedDim: int, hebbianRate: float = 0.01, useHebbian: bool = True, momentum: float = 0.9):
        super().__init__()
        self.num_modes = numModes
        self.embed_dim = embedDim
        self.hebbian_rate = hebbianRate
        self.use_hebbian = useHebbian
        self.momentum = momentum

        self.ctx_q = nn.Linear(self.embed_dim, 1, bias=False)

        self.base_weights = nn.Parameter(torch.empty(numModes, embedDim, embedDim))
        self.register_buffer("hebbian_memory", torch.zeros(1, numModes, embedDim, embedDim))

        self.ResetParameters()

        self.gate_head = nn.Sequential(
            nn.Linear(4 * embedDim, 2 * embedDim),
            nn.SiLU(),
            nn.Linear(2 * embedDim, 1))
        
        for m in self.gate_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   
                nn.init.zeros_(m.bias)

    def ResetParameters(self):
        eye = torch.eye(self.embed_dim, device=self.base_weights.device).unsqueeze(0).expand(self.num_modes, -1, -1)
        with torch.no_grad():
            self.base_weights.copy_(eye + 0.05 * torch.randn_like(eye))
        self.ResetHebbianMemory()

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.hebbian_memory.zero_()
            return
        mask = doneMask.detach().view(-1)
        if mask.numel() == self.hebbian_memory.size(0) and bool(mask.any().item()):
            self.hebbian_memory[mask] = 0

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if self.hebbian_memory.shape[0] != B:
            self.hebbian_memory = torch.zeros(B, self.num_modes, self.embed_dim, self.embed_dim, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor, applyPlasticity: bool = True) -> torch.Tensor: # inputs:[B,M,E]
        B, M, E = inputs.shape

        self.EnsureB(B, self.device, self.dtype)

        if self.use_hebbian:
            effW = self.momentum * self.base_weights.unsqueeze(0) + (1.0 - self.momentum) * self.hebbian_memory
            effW = torch.clamp(effW, -3.0, 3.0)
        else:
            effW = self.base_weights.unsqueeze(0)

        weighted = torch.einsum("bme,bmef->bmf", inputs, effW) # [B,M,E]

        alpha = torch.softmax(self.ctx_q(inputs).squeeze(-1) / math.sqrt(self.embed_dim), dim=1) # [B,M]
        context = torch.einsum("bme,bm->be", inputs, alpha).unsqueeze(1).expand(-1, M, -1)   

        gate_in = torch.cat([inputs, context, inputs - context, inputs * context], dim=-1) # [B,M,4E]
        gate_logits = self.gate_head(gate_in).squeeze(-1) # [B,M]
        gate_w = torch.softmax(gate_logits, dim=-1) # [B,M]

        fused = torch.einsum("bmf,bm->bf", weighted, gate_w) # [B,E]

        if applyPlasticity and self.use_hebbian and self.hebbian_rate > 0:
            with torch.no_grad():
                norm = math.sqrt(E)
                hebb_term = torch.einsum("bme,bf->bmef", inputs, fused) / (norm + 1e-8)
                mem_new = (1.0 - self.hebbian_rate) * self.hebbian_memory + self.hebbian_rate * hebb_term
                self.hebbian_memory.copy_(mem_new)

        return fused



class AttentionExtractor(AGICoreModule):
    def __init__(
        self,
        embedDim: int = 1024,
        sequenceLength: int = 32,
        numHeads: int = 16,
        temporalLayers: int = 12,
        capsDim: int = 256,
        routingIterations: int = 6,
        routingOutCaps: int = 8,
        hebbianRate: float = 0.01,
        useHebbian: bool = True,
        gradientClipVal: float = 1.0,
        structuredDim: Optional[int] = None,
        goalDim: Optional[int] = None,
        objectTokenCount: int = 16,
        useDistributedGating: bool = True,
        slowDtScale: float = 0.25,):
        super().__init__()

        self.num_caps = sequenceLength
        self.gradient_clip_val = gradientClipVal
        self.output_dim = embedDim
        self.use_hebbian = useHebbian
        self.num_heads = numHeads
        self.caps_dim = capsDim
        self.routing_out_caps = int(routingOutCaps)
        self.structured_dim = int(structuredDim if structuredDim is not None else max(1, embedDim // 2))
        self.goal_dim = int(goalDim if goalDim is not None else self.structured_dim)
        self.object_token_count = int(objectTokenCount)
        self.use_distributed_gating = bool(useDistributedGating)

        self.temporal_blocks: nn.ModuleList = nn.ModuleList([
            TemporalAttention(
                embedDim,
                numHeads,
                idx,
                useHebbian=useHebbian,
                slowDtScale=slowDtScale)
            for idx in range(temporalLayers)])

        self.caps_in_proj = nn.Sequential(
            nn.Linear(embedDim, self.caps_dim), 
            nn.LayerNorm(self.caps_dim), 
            nn.SiLU())
        self.routing = DynamicRouting(sequenceLength, self.caps_dim, self.routing_out_caps, self.caps_dim, iterations=routingIterations)
        self.caps_out_proj = nn.Linear(self.caps_dim, embedDim)

        self.fusion = HebbianFusion(numModes=3, embedDim=embedDim, hebbianRate=hebbianRate, useHebbian=useHebbian)

        self.context_proj = nn.Sequential(nn.Linear(embedDim * 2, embedDim), nn.GELU())

        self.static_mixer = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.Linear(embedDim, numHeads * 3))

        self.output_proj = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedDim * 2, embedDim),
            nn.LayerNorm(embedDim))

        self.object_pool_norm = nn.LayerNorm(self.structured_dim)
        self.object_pool_key = nn.Linear(self.structured_dim, self.structured_dim)
        self.object_pool_query = nn.Parameter(torch.randn(self.structured_dim) * 0.02)
        self.object_seq_proj = nn.Sequential(nn.LayerNorm(self.structured_dim), nn.Linear(self.structured_dim, embedDim), nn.GELU())
        self.motion_seq_proj = nn.Sequential(nn.LayerNorm(self.structured_dim), nn.Linear(self.structured_dim, embedDim), nn.GELU())
        self.quality_seq_proj = nn.Sequential(nn.LayerNorm(self.structured_dim), nn.Linear(self.structured_dim, embedDim), nn.GELU())
        self.pred_error_seq_proj = nn.Sequential(nn.LayerNorm(self.structured_dim), nn.Linear(self.structured_dim, embedDim), nn.GELU())
        self.goal_bias_proj = nn.Sequential(nn.LayerNorm(self.goal_dim), nn.Linear(self.goal_dim, embedDim), nn.GELU())
        self.precision_bias = nn.Linear(1, embedDim)
        gate_in_dim = self.goal_dim + 3
        gate_hidden = max(32, embedDim // 2)
        self.mod_gate = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, numHeads + embedDim))
        nn.init.zeros_(self.mod_gate[-1].weight)
        nn.init.zeros_(self.mod_gate[-1].bias)

        self.structured_gate = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 4))
        nn.init.zeros_(self.structured_gate[-1].weight)
        nn.init.zeros_(self.structured_gate[-1].bias)
            

    def ClipGrads(self):
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)

    def SanitizeModulators(
        self,
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        precision: torch.Tensor,
        B: int,
        x: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tdError is None:
            tdError = x.new_zeros(B)
        if uncertainty is None:
            uncertainty = x.new_zeros(B)
        if precision is None:
            precision = x.new_ones(B)

        return tdError, uncertainty, precision

    def ComputeDistributedGates(
        self,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_distributed_gating:
            return None, None
        B = goalBias.size(0)
        gate_in = torch.cat([
            goalBias,
            precision[:, None],
            tdError[:, None],
            uncertainty[:, None]], dim=-1)
        logits = self.mod_gate(gate_in)
        head_logits, channel_logits = logits.split([self.num_heads, self.output_dim], dim=-1)
        head_gate = 1.0 + 0.25 * torch.tanh(head_logits).view(B, self.num_heads, 1, 1)
        channel_gate = 1.0 + 0.25 * torch.tanh(channel_logits).view(B, 1, self.output_dim)
        return head_gate, channel_gate

    def ObjectAttentionPool(self, objectSeq: torch.Tensor) -> torch.Tensor:
        y = self.object_pool_norm(objectSeq)
        B, S, K, D = y.shape
        keys = self.object_pool_key(y.reshape(B * S * K, D)).reshape(B, S, K, D)
        scale = max(float(keys.size(-1)) ** 0.5, 1.0)
        scores = torch.einsum("bskd,d->bsk", keys, self.object_pool_query) / scale
        weights = F.softmax(scores, dim=2)
        return (objectSeq * weights.unsqueeze(-1)).sum(dim=2)

    def BuildStructuredFusion(
        self,
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        def project_seq(sig: torch.Tensor, proj: nn.Module) -> torch.Tensor:
            y = sig
            if y.dim() == 4:
                y = self.ObjectAttentionPool(y)
            return proj(y)

        terms = torch.stack([
            project_seq(objectSeq, self.object_seq_proj),
            project_seq(motionSeq, self.motion_seq_proj),
            project_seq(qualitySeq, self.quality_seq_proj),
            project_seq(predErrorSeq, self.pred_error_seq_proj)], dim=2) # [B,S,4,E]

        gate_in = torch.cat([
            goalBias,
            precision[:, None],
            tdError[:, None],
            uncertainty[:, None]], dim=-1)
        weights = torch.softmax(self.structured_gate(gate_in), dim=-1) # [B,4]
        fused = (terms * weights[:, None, :, None]).sum(dim=2) * float(terms.size(2))
        return fused, weights

    def forward(
        self,
        x: torch.Tensor, # [B,S,E]
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None, # [-1 ,1] [B]
        uncertainty: Optional[torch.Tensor]=None, # [0 ,1] [B]
        returnExtras: bool = False,
        applyPlasticity: bool = True,) -> torch.Tensor: # [B] or scalar
        B, S, E = x.shape

        tdError, uncertainty, precision = self.SanitizeModulators(tdError, uncertainty, precision, B, x)
        head_gate, channel_gate = self.ComputeDistributedGates(goalBias, precision, tdError, uncertainty)

        extras: Dict[str, Any] = {}

        structured_sum, structured_weights = self.BuildStructuredFusion(
            objectSeq,
            motionSeq,
            qualitySeq,
            predErrorSeq,
            goalBias,
            precision,
            tdError,
            uncertainty)
        x = x + 0.0625 * structured_sum
        extras["structured_terms"] = x.new_tensor(4.0)
        extras["structured_weights"] = structured_weights.detach()

        goal_term = self.goal_bias_proj(goalBias)
        x = x + 0.10 * goal_term.unsqueeze(1)
        extras["goal_bias_norm"] = goal_term.detach().norm(dim=-1)

        x = x * (0.75 + 0.50 * precision[:, None, None])
        x = x + 0.05 * self.precision_bias(precision[:, None]).unsqueeze(1)
        extras["precision"] = precision.detach()
        if head_gate is not None:
            extras["head_gate_mean"] = head_gate.detach().mean(dim=(1, 2, 3))
            extras["channel_gate_mean"] = channel_gate.detach().mean(dim=(1, 2))

        if S % self.num_caps != 0:
            pad_len = self.num_caps - (S % self.num_caps)
            x = F.pad(x, (0,0,0,pad_len))
            if keyPaddingMask is not None:
                keyPaddingMask = F.pad(keyPaddingMask, (0,pad_len), value=True)
            S = x.size(1)

        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1)
            x = x * keep
        
        for blk in self.temporal_blocks:
            x = blk(
                x,
                keyPaddingMask=keyPaddingMask,
                tdError=tdError,
                uncertainty=uncertainty,
                headGate=head_gate,
                channelGate=channel_gate,
                applyPlasticity=applyPlasticity)

        chunk = S // self.num_caps

        if keyPaddingMask is not None:
            seg_mask = keyPaddingMask.reshape(B, self.num_caps, chunk) # [B,I,chunk]
        else:
            seg_mask = x.new_zeros(B, self.num_caps, chunk, dtype=torch.bool)

        valid = (~seg_mask)                                  
        valid_cnt = valid.sum(dim=2, keepdim=True) # [B,I,1]
        valid_cnt_safe = valid_cnt.clamp_min(1.0)

        h_seg = x.reshape(B, self.num_caps, chunk, E) # [B,I,chunk,E]
        caps = (h_seg * valid.unsqueeze(-1)).sum(dim=2) / valid_cnt_safe # [B,I,E] masked mean
        caps_mask = (valid_cnt.squeeze(-1) == 0)                      

        cap_in = self.caps_in_proj(caps)
        routed = self.routing(cap_in, caps_mask) # [B,routingOutCaps,capsDim]
        routed = self.caps_out_proj(routed)
        routed = F.layer_norm(routed, (E,))

        routed_mean = routed.mean(dim=1) # [B,E]
        
        if keyPaddingMask is not None:
            denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0) # [B,1,1]
            temp_mean = (x * keep).sum(dim=1) / denom.squeeze(-1)  
        else:
            temp_mean = x.mean(dim=1)
        
        fusion_in = torch.stack([
            temp_mean, 
            routed_mean, 
            temp_mean + routed_mean], dim=1)  # [B,3,E]
        
        fused = self.fusion(fusion_in, applyPlasticity=applyPlasticity)  # [B,E]

        context = self.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))  # [B,E]

        logits = self.static_mixer(context).view(B, self.num_heads, 3)
        strat_w = torch.softmax(logits, dim=-1)  #[B,H,3]

        feats = torch.stack([temp_mean, routed_mean, fused], dim=1)  # [B,3,E]
        mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)  # [B,H,E]
        out = mixed_per_head.mean(dim=1)  # [B,E]

        out = self.output_proj(out)
        if returnExtras:
            return out, extras
        return out

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
        if doneMask is None:
            for blk in self.temporal_blocks:
                blk.mhsa.ResetHebbianMemory()
            self.fusion.ResetHebbianMemory()
            return

        mask = doneMask.detach().view(-1).bool()
        if not bool(mask.any().item()):
            return

        def zero_rows(t: torch.Tensor) -> None:
            if t.size(0) == mask.numel():
                t[mask] = 0

        for blk in self.temporal_blocks:
            mhsa = blk.mhsa
            zero_rows(mhsa.hebbian_weights)
            zero_rows(mhsa.U)
            zero_rows(mhsa.V)
        self.fusion.ResetHebbianMemory(doneMask=mask)

    def AttenLowrankToFullrank(self):
        for blk in self.temporal_blocks:
            blk.mhsa.LowrankToFullrank()
        
    def AttenFullrankToLowrank(self):
        for blk in self.temporal_blocks:
            blk.mhsa.FullrankToLowrank()
            

    @torch.no_grad()
    def ExportState(self) -> dict:
        st = {
            "fusion_hebb": self.fusion.hebbian_memory.detach().clone(),
            "mhsa": []}
        
        for blk in self.temporal_blocks:
            mhsa = blk.mhsa
            st["mhsa"].append({
                "U": mhsa.U.detach().clone(),
                "V": mhsa.V.detach().clone(),
                "hebbW": mhsa.hebbian_weights.detach().clone(),
                "use_low_rank": bool(mhsa.use_low_rank),})
            
        return st

    @torch.no_grad()
    def ImportState(self, st: dict):
        fusion_state = st.get("fusion_hebb")
        if fusion_state is not None:
            if self.fusion.hebbian_memory is None:
                self.fusion.hebbian_memory = fusion_state.clone()
            elif self.fusion.hebbian_memory.shape != fusion_state.shape:
                self.fusion.hebbian_memory = fusion_state.clone()
            else:
                self.fusion.hebbian_memory.copy_(fusion_state)
        
        if "mhsa" in st:
            for blk, s in zip(self.temporal_blocks, st["mhsa"]):
                mhsa = blk.mhsa
                
                target_B = s["U"].size(0)
                if mhsa.U.size(0) != target_B:
                    mhsa.EnsureB(target_B, mhsa.U.device, mhsa.U.dtype)

                mhsa.U.copy_(s["U"])
                mhsa.V.copy_(s["V"])
                mhsa.hebbian_weights.copy_(s["hebbW"])
                mhsa.use_low_rank = bool(s["use_low_rank"])


class AttentionOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: nn.Module,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankQ: int = 64,
        maxRankK: int = 64,
        maxRankV: int = 64,
        maxRankO: int = 64,):
        self.maxRankQ = int(maxRankQ)
        self.maxRankK = int(maxRankK)
        self.maxRankV = int(maxRankV)
        self.maxRankO = int(maxRankO)
        super().__init__(
            base,
            initRankEach=initRankEach,
            autoRank=autoRank,
            evThreshold=evThreshold,
            gradEma=gradEma,)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        L = len(self.base.temporal_blocks)
        assert L > 0, "AttentionExtractor.temporal_blocks is NULL"
        E = int(self.base.temporal_blocks[0].mhsa.embed_dim)

        def alloc_linear(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, E, device=device, dtype=dtype) * 1e-4) # [r, inDim]
            B = nn.Parameter(torch.zeros(E, addRank, device=device, dtype=dtype) * 1e-4) # [outDim, r]
            s = nn.Parameter(torch.tensor(1e-2, device=device, dtype=dtype))
            return A, B, s

        def compose_linear(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            s_eff = torch.tanh(s) * GetParametersScale(s) 
            return s_eff * (b @ a)

        return {
            "q": SiteSpec("q", L, E, E, self.maxRankQ, alloc_linear, compose_linear),
            "k": SiteSpec("k", L, E, E, self.maxRankK, alloc_linear, compose_linear),
            "v": SiteSpec("v", L, E, E, self.maxRankV, alloc_linear, compose_linear),
            "o": SiteSpec("o", L, E, E, self.maxRankO, alloc_linear, compose_linear),}

    def ForwardWithDeltas(
        self,
        x: torch.Tensor,  # [B,S,E]
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        *,
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        returnExtras: bool = False,
        applyPlasticity: bool = True,
        **kwargs,) -> torch.Tensor:
        B, S, E = x.shape

        tdError, uncertainty, precision = self.base.SanitizeModulators(tdError, uncertainty, precision, B, x)
        head_gate, channel_gate = self.base.ComputeDistributedGates(goalBias, precision, tdError, uncertainty)

        extras: Dict[str, Any] = {}

        structured_sum, structured_weights = self.base.BuildStructuredFusion(
            objectSeq,
            motionSeq,
            qualitySeq,
            predErrorSeq,
            goalBias,
            precision,
            tdError,
            uncertainty)
        x = x + 0.0625 * structured_sum
        extras["structured_terms"] = x.new_tensor(4.0)
        extras["structured_weights"] = structured_weights.detach()

        goal_term = self.base.goal_bias_proj(goalBias)
        x = x + 0.10 * goal_term.unsqueeze(1)
        extras["goal_bias_norm"] = goal_term.detach().norm(dim=-1)

        x = x * (0.75 + 0.50 * precision[:, None, None])
        x = x + 0.05 * self.base.precision_bias(precision[:, None]).unsqueeze(1)
        extras["precision"] = precision.detach()
        if head_gate is not None:
            extras["head_gate_mean"] = head_gate.detach().mean(dim=(1, 2, 3))
            extras["channel_gate_mean"] = channel_gate.detach().mean(dim=(1, 2))

        num_caps = int(self.base.num_caps)

        if S % num_caps != 0:
            pad_len = num_caps - (S % num_caps)
            x = F.pad(x, (0, 0, 0, pad_len))
            if keyPaddingMask is not None:
                keyPaddingMask = F.pad(keyPaddingMask, (0, pad_len), value=True)
            S = x.size(1)

        if keyPaddingMask is not None:
            keep0 = (~keyPaddingMask).unsqueeze(-1)
            h = x * keep0
        else:
            h = x

        for layerIdx, blk in enumerate(self.base.temporal_blocks):
            h = self.ForwardBlockWithDeltas(
                blk=blk,
                x=h,
                keyPaddingMask=keyPaddingMask,
                tdError=tdError,
                uncertainty=uncertainty,
                delta=deltasPerLayer[layerIdx],
                headGate=head_gate,
                channelGate=channel_gate,
                applyPlasticity=applyPlasticity)

        chunk = S // num_caps
        if keyPaddingMask is not None:
            seg_mask = keyPaddingMask.reshape(B, num_caps, chunk)
        else:
            seg_mask = h.new_zeros(B, num_caps, chunk, dtype=torch.bool)

        valid = (~seg_mask)
        valid_cnt = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        h_seg = h.reshape(B, num_caps, chunk, E)
        caps = (h_seg * valid.unsqueeze(-1)).sum(dim=2) / valid_cnt
        caps_mask = (valid_cnt.squeeze(-1) == 0)

        cap_in = self.base.caps_in_proj(caps) 
        routed = self.base.routing(cap_in, caps_mask)
        routed = self.base.caps_out_proj(routed) 
        routed = F.layer_norm(routed, (E,))

        routed_mean = routed.mean(dim=1)
        if keyPaddingMask is not None:
            denom = keep0.sum(dim=1, keepdim=True).clamp_min(1.0)
            temp_mean = (h * keep0).sum(dim=1) / denom.squeeze(-1)
        else:
            temp_mean = h.mean(dim=1)

        fusion_in = torch.stack([temp_mean, routed_mean, temp_mean + routed_mean], dim=1)  # [B,3,E]
        fused = self.base.fusion(fusion_in, applyPlasticity=applyPlasticity)

        context = self.base.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))  # [B,E]

        logits = self.base.static_mixer(context).view(B, self.base.num_heads, 3)
        strat_w = torch.softmax(logits, dim=-1)

        feats = torch.stack([temp_mean, routed_mean, fused], dim=1)  # [B,3,E]
        mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)
        out = mixed_per_head.mean(dim=1)

        out = self.base.output_proj(out)
        if returnExtras:
            return out, extras
        return out

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
        self.base.ResetHebbianMemory(doneMask=doneMask)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        r = int(a.size(0))
        if r <= 0 or a.numel() == 0 or b.numel() == 0 or abs(float(scale)) < 1e-12:
            return False

        blk = self.base.temporal_blocks[layerIdx]
        mhsa = blk.mhsa
        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "q":
            mhsa.q_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        elif site == "k":
            mhsa.k_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        elif site == "v":
            mhsa.v_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        elif site == "o":
            mhsa.o_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        else:
            raise ValueError(f"Unknown site: {site}")
        return True

    def ForwardBlockWithDeltas(
        self,
        blk,
        x: torch.Tensor,
        keyPaddingMask: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        delta: Dict[str, Optional[torch.Tensor]],
        headGate: Optional[torch.Tensor] = None,
        channelGate: Optional[torch.Tensor] = None,
        applyPlasticity: bool = True,) -> torch.Tensor:

        mhsa = blk.mhsa
        B, S, E = x.shape

        residual0 = x
        x_norm = blk.norm(x)

        mhsa.EnsureB(B, device=x_norm.device, dtype=x_norm.dtype)

        def eff_linear(weight: torch.Tensor, adapter, d2: Optional[torch.Tensor]):
            W = weight
            d_base = adapter.DeltaWeight()
            if d_base is not None:
                W = W + d_base
            if d2 is not None:
                W = W + d2
            return W

        Wq = eff_linear(mhsa.q_proj.weight, mhsa.q_adapter, delta.get("q"))
        Wk = eff_linear(mhsa.k_proj.weight, mhsa.k_adapter, delta.get("k"))
        Wv = eff_linear(mhsa.v_proj.weight, mhsa.v_adapter, delta.get("v"))
        Wo = eff_linear(mhsa.out_proj.weight, mhsa.o_adapter, delta.get("o"))

        td_eff  = tdError * mhsa.td_unc_scale
        unc_eff = uncertainty * mhsa.td_unc_scale

        neuromod = mhsa.ComputeNeuromodulation(td_eff, B) 
        tau = mhsa.ModulateTau(td_eff, unc_eff, B) 
        hebb_mod = mhsa.ComputeHebbMod(td_eff, unc_eff, B)

        q_lin = F.linear(x_norm, Wq, mhsa.q_proj.bias)
        k_lin = F.linear(x_norm, Wk, mhsa.k_proj.bias)
        v_lin = F.linear(x_norm, Wv, mhsa.v_proj.bias)

        q = q_lin.view(B, S, mhsa.num_heads, mhsa.head_dim).transpose(1, 2) 
        k = k_lin.view(B, S, mhsa.num_heads, mhsa.head_dim).transpose(1, 2)
        v = v_lin.view(B, S, mhsa.num_heads, mhsa.head_dim).transpose(1, 2)

        if applyPlasticity and mhsa.use_hebbian and mhsa.base_hebbian_rate > 0:
            alpha = mhsa.base_hebbian_rate * hebb_mod  
            if keyPaddingMask is not None:
                keep4 = (~keyPaddingMask).view(B, 1, S, 1)
                mhsa.UpdateHebbianWeights(v, q, alpha, keep4=keep4)
            else:
                mhsa.UpdateHebbianWeights(v, q, alpha, keep4=None)

        q = q * neuromod
        if headGate is not None:
            q = q * headGate
        q = q / tau

        if mhsa.use_low_rank:
            v_proj = torch.einsum("bhsd,bhdr->bhsr", v, mhsa.V) 
            delt = torch.einsum("bhsr,bhrd->bhsd", v_proj, mhsa.U.transpose(-2, -1)) 
        else:
            delt = torch.einsum("bhse,bhde->bhsd", v, mhsa.hebbian_weights) 

        delt = delt * hebb_mod
        v_fast = v + delt

        q_attn = mhsa.rope.Apply(q)
        k_attn = mhsa.rope.Apply(k)

        d = q_attn.size(-1)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(d)

        if keyPaddingMask is not None:
            mask = keyPaddingMask[:, None, None, :] 
            mask_val = -1e9 if q.dtype != torch.float16 else -1e4
            scores = scores.masked_fill(mask, mask_val)

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=mhsa.attn_dropout_p if mhsa.training else 0.0,training=mhsa.training)
        context = torch.matmul(weights, v_fast) 
        out = context.transpose(1, 2).reshape(B, S, mhsa.embed_dim) 

        mhsa_out = F.linear(out, Wo, mhsa.out_proj.bias)

        ssm_out = blk.ssm(x_norm, keyPaddingMask=keyPaddingMask,tdError=tdError, uncertainty=uncertainty)

        w = torch.sigmoid(blk.mix_gate(x_norm))
        y = w * mhsa_out + (1.0 - w) * ssm_out
        if channelGate is not None:
            y = y * channelGate

        x1 = residual0 + blk.dropout(y) * blk.gamma

        residual1 = x1
        x1_norm = blk.norm_ffn(x1)
        ffn_out = blk.ffn(x1_norm)
        out2 = residual1 + blk.dropout(ffn_out) * blk.gamma_ffn

        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1)
            out2 = out2 * keep

        return out2


class TestAttentionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        self.B = 2
        self.S = 16
        self.E = 32
        self.H = 4
        self.R = 4
        self.M = 3
        self.out_caps = 4

    def AttentionInputs(self, B: int, S: int, E: int, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        half = E // 2
        return {
            "objectSeq": torch.randn(B, S, 16, half, device=self.device, dtype=dtype),
            "motionSeq": torch.randn(B, S, half, device=self.device, dtype=dtype),
            "qualitySeq": torch.randn(B, S, half, device=self.device, dtype=dtype),
            "predErrorSeq": torch.randn(B, S, half, device=self.device, dtype=dtype),
            "goalBias": torch.randn(B, half, device=self.device, dtype=dtype),
            "precision": torch.ones(B, device=self.device, dtype=dtype),}

    def AttentionForward(self, model, x: torch.Tensor, **kwargs) -> torch.Tensor:
        args = self.AttentionInputs(int(x.size(0)), int(x.size(1)), int(x.size(2)), x.dtype)
        args.update(kwargs)
        return model(x, **args)

    def AdapterRankAndParams(self, adapter) -> Tuple[int, int]:
        rank_sum = 0
        param_cnt = 0
        if not hasattr(adapter, "A_list"):
            return 0, 0
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            rank_sum += int(A.shape[0])
            param_cnt += int(A.numel() + B.numel() + 1)
        return rank_sum, param_cnt

    def MhsaAllRanksAndParams(self, base) -> Tuple[List[Tuple[int,int,int,int]], Tuple[int,int,int,int]]:
        per_layer = []
        sum_q = sum_k = sum_v = sum_o = 0
        for blk in base.temporal_blocks:
            mhsa = blk.mhsa
            rq, _ = self.AdapterRankAndParams(mhsa.q_adapter)
            rk, _ = self.AdapterRankAndParams(mhsa.k_adapter)
            rv, _ = self.AdapterRankAndParams(mhsa.v_adapter)
            ro, _ = self.AdapterRankAndParams(mhsa.o_adapter)
            per_layer.append((rq, rk, rv, ro))
            sum_q += rq; sum_k += rk; sum_v += rv; sum_o += ro
        return per_layer, (sum_q, sum_k, sum_v, sum_o)

    def DeltaFromLinearAdapter(self, adapter) -> torch.Tensor:
        if (not hasattr(adapter, "A_list")) or len(adapter.A_list) == 0:
            out_f = getattr(adapter, "out_f", None)
            in_f = getattr(adapter, "in_f", None)
            if out_f is None or in_f is None:
                return torch.zeros(0, 0, device=self.device)
            return torch.zeros(out_f, in_f, device=self.device)
        out_f = adapter.out_f
        in_f = adapter.in_f
        delta = torch.zeros(out_f, in_f, device=adapter.A_list[0].device, dtype=adapter.A_list[0].dtype)
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            s_eff = torch.tanh(s.detach()) * GetParametersScale(s.detach())
            delta = delta + s_eff * (B @ A)
        return delta

    def AttentionStateMaxAbsDiff(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        vals = [(a["fusion_hebb"] - b["fusion_hebb"]).abs().max().item()]
        for x, y in zip(a["mhsa"], b["mhsa"]):
            vals.append((x["U"] - y["U"]).abs().max().item())
            vals.append((x["V"] - y["V"]).abs().max().item())
            vals.append((x["hebbW"] - y["hebbW"]).abs().max().item())
        return max(vals)

    def TestPlasticityGate(self):
        try:
            torch.manual_seed(444)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=1,capsDim=16,routingIterations=2,hebbianRate=0.05,useHebbian=True).to(self.device)
            model.eval()
            x = torch.randn(2, 16, 64, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x.dtype)
            td = torch.tensor([0.7, -0.2], device=self.device)
            unc = torch.tensor([0.1, 0.4], device=self.device)

            with torch.no_grad():
                _ = model(x, tdError=td, uncertainty=unc, applyPlasticity=False, **args)
                model.ResetHebbianMemory()
                st0 = model.ExportState()
                _ = model(x, tdError=td, uncertainty=unc, applyPlasticity=False, **args)
                st1 = model.ExportState()
                assert self.AttentionStateMaxAbsDiff(st0, st1) < 1e-12, "applyPlasticity=False changed Hebbian state"
                _ = model(x, tdError=td, uncertainty=unc, applyPlasticity=True, **args)
                st2 = model.ExportState()
                assert self.AttentionStateMaxAbsDiff(st1, st2) > 1e-8, "applyPlasticity=True did not update Hebbian state"
            print("PlasticityGate passed.")
            return True
        except AssertionError as e:
            print(f"PlasticityGate failed: {e}")
            return False
        except Exception as e:
            print(f"PlasticityGate error: {e}")
            return False

    def TestSelectiveDoneReset(self):
        try:
            torch.manual_seed(445)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=1,capsDim=16,routingIterations=2,hebbianRate=0.05,useHebbian=True).to(self.device)
            model.eval()
            x = torch.randn(2, 16, 64, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x.dtype)
            with torch.no_grad():
                _ = model(x, tdError=torch.ones(2, device=self.device), uncertainty=torch.zeros(2, device=self.device), applyPlasticity=True, **args)
                st0 = model.ExportState()
                model.ResetHebbianMemory(doneMask=torch.zeros(2, dtype=torch.bool, device=self.device))
                st_false = model.ExportState()
                assert self.AttentionStateMaxAbsDiff(st0, st_false) < 1e-12, "all-false doneMask changed state"

                model.ResetHebbianMemory(doneMask=torch.tensor([True, False], device=self.device))
                for blk_idx, s in enumerate(st0["mhsa"]):
                    mhsa_now = model.temporal_blocks[blk_idx].mhsa
                    assert mhsa_now.U[0].abs().max().item() < 1e-12 and mhsa_now.V[0].abs().max().item() < 1e-12, "done row U/V not cleared"
                    assert torch.allclose(mhsa_now.U[1], s["U"][1]) and torch.allclose(mhsa_now.V[1], s["V"][1]), "non-done row U/V changed"
                assert model.fusion.hebbian_memory[0].abs().max().item() < 1e-12, "done row fusion memory not cleared"
                assert torch.allclose(model.fusion.hebbian_memory[1], st0["fusion_hebb"][1]), "non-done fusion row changed"

                model.ResetHebbianMemory(doneMask=torch.ones(2, dtype=torch.bool, device=self.device))
                assert model.fusion.hebbian_memory.abs().max().item() < 1e-12, "all-true doneMask did not clear fusion"
                for blk in model.temporal_blocks:
                    assert blk.mhsa.U.abs().max().item() < 1e-12 and blk.mhsa.V.abs().max().item() < 1e-12, "all-true doneMask did not clear MHSA"
            print("SelectiveDoneReset passed.")
            return True
        except AssertionError as e:
            print(f"SelectiveDoneReset failed: {e}")
            return False
        except Exception as e:
            print(f"SelectiveDoneReset error: {e}")
            return False

    def TestModulatorsPassThrough(self):
        try:
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=1,capsDim=16,routingIterations=2,hebbianRate=0.0,useHebbian=False).to(self.device)
            model.eval()
            x = torch.randn(2, 16, 64, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x.dtype)
            td = torch.tensor([-0.5, 0.5], device=self.device)
            unc = torch.tensor([0.1, 0.7], device=self.device)
            precision = torch.tensor([0.2, 1.0], device=self.device)
            td_raw = torch.tensor([-2.0, 2.0], device=self.device)
            unc_raw = torch.tensor([-1.0, 2.0], device=self.device)
            precision_raw = torch.tensor([0.0, 2.0], device=self.device)
            td_s, unc_s, precision_s = model.SanitizeModulators(td_raw, unc_raw, precision_raw, 2, x)
            assert td_s is td_raw and unc_s is unc_raw and precision_s is precision_raw, "modulators should pass through unchanged"
            with torch.no_grad():
                y, extras = model(x, tdError=td, uncertainty=unc, precision=precision, returnExtras=True, **{k: v for k, v in args.items() if k != "precision"})
            assert torch.isfinite(y).all(), "valid modulators produced non-finite output"
            assert torch.allclose(extras["precision"], precision), "precision should pass through unchanged"

            print("ModulatorsPassThrough passed.")
            return True
        except AssertionError as e:
            print(f"ModulatorsPassThrough failed: {e}")
            return False
        except Exception as e:
            print(f"ModulatorsPassThrough error: {e}")
            return False

    def TestRoutingAndDistributedGates(self):
        try:
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=1,capsDim=16,routingIterations=2,hebbianRate=0.0,useHebbian=False).to(self.device)
            assert model.routing.O == 8, f"default routingOutCaps should be 8, got {model.routing.O}"
            assert model.routing.transformation.numel() == 16 * 8 * 16 * 16, "routing transformation parameter count mismatch"
            B = 3
            goal = torch.randn(B, 32, device=self.device)
            td, unc, precision = model.SanitizeModulators(
                torch.randn(B, device=self.device),
                torch.rand(B, device=self.device),
                torch.ones(B, device=self.device),
                B,
                goal)
            head_gate, channel_gate = model.ComputeDistributedGates(goal, precision, td, unc)
            assert tuple(head_gate.shape) == (B, 4, 1, 1), f"head gate shape mismatch: {head_gate.shape}"
            assert tuple(channel_gate.shape) == (B, 1, 64), f"channel gate shape mismatch: {channel_gate.shape}"
            assert torch.allclose(head_gate, torch.ones_like(head_gate)), "head gate should initialize neutral"
            assert torch.allclose(channel_gate, torch.ones_like(channel_gate)), "channel gate should initialize neutral"
            print("RoutingAndDistributedGates passed.")
            return True
        except AssertionError as e:
            print(f"RoutingAndDistributedGates failed: {e}")
            return False
        except Exception as e:
            print(f"RoutingAndDistributedGates error: {e}")
            return False

    def TestDualTimeConstantSSM(self):
        try:
            ssm = SelectiveSSM(self.E, stateDim=4, convKernel=4, useCausalConv=True, slowDtScale=0.25).to(self.device)
            ssm.train()
            x = torch.randn(self.B, self.S, self.E, device=self.device, requires_grad=True)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device)
            kpm[:, -2:] = True
            y = ssm(x, tdError=torch.randn(self.B, device=self.device), uncertainty=torch.rand(self.B, device=self.device), keyPaddingMask=kpm)
            assert y.shape == (self.B, self.S, self.E), f"dual SSM shape mismatch: {y.shape}"
            assert torch.isfinite(y).all(), "dual SSM output has non-finite values"
            assert y[:, -2:].abs().max().item() < 1e-6, "masked dual SSM positions should be zero"
            loss = y.square().mean()
            loss.backward()
            for n, p in ssm.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"dual SSM non-finite grad: {n}"
            print("DualTimeConstantSSM passed.")
            return True
        except AssertionError as e:
            print(f"DualTimeConstantSSM failed: {e}")
            return False
        except Exception as e:
            print(f"DualTimeConstantSSM error: {e}")
            return False

    def TestSimpleSSM(self):
        try:
            ssm = SelectiveSSM(self.E, stateDim=4, convKernel=4, useCausalConv=True).to(self.device)
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device)
            kpm[:, -3:] = True
            unc = torch.randn(self.B, device=self.device)
            y = ssm(x, keyPaddingMask=kpm, tdError=torch.randn(self.B, device=self.device), uncertainty = unc)
            assert y.shape == (self.B, self.S, self.E), f"Output shape mismatch: {y.shape}"
            print("SimpleSSM test passed.")
            return True
        except AssertionError as e:
            print(f"SimpleSSM test failed: {e}")
            return False
        except Exception as e:
            print(f"SimpleSSM test error: {e}")
            return False

    def TestMultiHeadAttention(self):
        try:
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device); kpm[:, -3:] = True
            td_unc = torch.randn(self.B, device=self.device)
            attn = MultiHeadAttention(embedDim=self.E, numHeads=self.H, lowRank=True, rank=self.R).to(self.device)
            y1 = attn(x, x, x, keyPaddingMask=kpm, tdError=td_unc, uncertainty=td_unc)
            assert y1.shape == (self.B, self.S, self.E)
            attn.LowrankToFullrank()
            y2 = attn(x, x, x, keyPaddingMask=kpm, tdError=td_unc, uncertainty=td_unc)
            assert y2.shape == (self.B, self.S, self.E)
            attn.FullrankToLowrank()
            y3 = attn(x, x, x, keyPaddingMask=kpm, tdError=td_unc, uncertainty=td_unc)
            assert y3.shape == (self.B, self.S, self.E)
            print("MultiHeadAttention test passed.")
            return True
        except AssertionError as e:
            print(f"MultiHeadAttention test failed: {e}")
            return False
        except Exception as e:
            print(f"MultiHeadAttention test error: {e}")
            return False

    def TestTemporalAttention(self):
        try:
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device)
            ta = TemporalAttention(self.E, self.H, layerIdx=0, useHebbian=True).to(self.device)
            y = ta(x, keyPaddingMask=kpm, tdError=torch.randn(self.B, device=self.device), uncertainty=torch.randn(self.B, device=self.device))
            assert y.shape == (self.B, self.S, self.E)
            print("TemporalAttention test passed.")
            return True
        except AssertionError as e:
            print(f"TemporalAttention test failed: {e}")
            return False
        except Exception as e:
            print(f"TemporalAttention test error: {e}")
            return False

    def TestDynamicRouting(self):
        try:
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            mask = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device); mask[:, -2:] = True
            router = DynamicRouting(inCaps=self.S, inDim=self.E, outCaps=self.out_caps, outDim=self.E, iterations=3).to(self.device)
            y = router(x, mask)
            assert y.shape == (self.B, self.out_caps, self.E), f"Output shape mismatch: {y.shape}"
            print("DynamicRouting test passed.")
            return True
        except AssertionError as e:
            print(f"DynamicRouting test failed: {e}")
            return False
        except Exception as e:
            print(f"DynamicRouting test error: {e}")
            return False

    def TestHebbianFusion(self):
        try:
            fusion = HebbianFusion(numModes=self.M, embedDim=self.E, hebbianRate=0.01, useHebbian=True).to(self.device)
            inputs = torch.randn(self.B, self.M, self.E, device=self.device)
            y = fusion(inputs)
            assert y.shape == (self.B, self.E), f"Output shape mismatch: {y.shape}"
            print("HebbianFusion test passed.")
            return True
        except AssertionError as e:
            print(f"HebbianFusion test failed: {e}")
            return False
        except Exception as e:
            print(f"HebbianFusion test error: {e}")
            return False

    def TestAttentionExtractor(self):
        try:
            model = AttentionExtractor(embedDim=self.E, sequenceLength=self.S, numHeads=self.H,temporalLayers=2, routingIterations=3, hebbianRate=0.01,useHebbian=True, gradientClipVal=0.5).to(self.device)
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device); kpm[:, -2:] = True
            td = torch.randn(self.B, device=self.device)
            y = self.AttentionForward(model, x, keyPaddingMask=kpm, tdError=td)
            assert y.shape == (self.B, self.E)

            loss = y.mean()
            loss.backward()
            model.ClipGrads()
            print("AttentionExtractor test passed.")
            return True
        except AssertionError as e:
            print(f"AttentionExtractor test failed: {e}")
            return False
        except Exception as e:
            print(f"AttentionExtractor test error: {e}")
            return False

    def TestAttentionExtractorIOShapes(self):
        try:
            batch_size = 2
            seq_len = 16
            embed_dim = 1024

            model = AttentionExtractor(
                embedDim=embed_dim,
                sequenceLength=seq_len,
                numHeads=16,
                temporalLayers=2,
                routingIterations=3,
                hebbianRate=0.0,
                useHebbian=False,
                gradientClipVal=0.5).to(self.device)
            model.eval()

            x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
            kpm = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
            kpm[:, -2:] = True
            td = torch.randn(batch_size, device=self.device)
            uncertainty = torch.rand(batch_size, device=self.device)

            def print_shape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            with torch.no_grad():
                print_shape("input.x", x)
                print_shape("input.keyPaddingMask", kpm)
                print_shape("input.tdError", td)
                print_shape("input.uncertainty", uncertainty)
                y = self.AttentionForward(model, x, keyPaddingMask=kpm, tdError=td, uncertainty=uncertainty)
                print_shape("output.y", y)

            expected_out_shape = (batch_size, embed_dim)
            assert tuple(y.shape) == expected_out_shape, f"Output shape mismatch: {y.shape}"
            return True
        except AssertionError as e:
            print(f"TestAttentionExtractorIOShapes failed: {e}")
            return False
        except Exception as e:
            print(f"TestAttentionExtractorIOShapes error: {e}")
            return False

    def TrainStepSmoke(self):
        try:
            torch.manual_seed(123)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.01, useHebbian=True, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            x = torch.randn(8, 16, 64, device=self.device)
            td = torch.randn(8, device=self.device)
            y = torch.randn(8, 12, device=self.device)

            out = self.AttentionForward(model, x, keyPaddingMask=None, tdError=td)
            pred = head(out)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.ClipGrads()

            grads_ok = True
            for _, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_ok = False; break
            assert grads_ok and head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Gradient invalid."
            opt.step()
            print("TrainStepSmoke passed.")
            return True
        except AssertionError as e:
            print(f"TrainStepSmoke failed: {e}")
            return False
        except Exception as e:
            print(f"TrainStepSmoke error: {e}")
            return False

    def NoNanAfterManySteps(self, steps: int = 30):
        try:
            torch.manual_seed(321)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.01, useHebbian=True, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            for t in range(steps):
                x = torch.randn(8, 16, 64, device=self.device)
                td = torch.randn(8, device=self.device)
                y = torch.randn(8, 12, device=self.device)

                pred = head(self.AttentionForward(model, x, tdError=td))
                loss = F.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                model.ClipGrads()
                for n, p in list(model.named_parameters()) + list(head.named_parameters()):
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"
                opt.step()
            print("NoNanAfterManySteps passed.")
            return True
        except AssertionError as e:
            print(f"NoNanAfterManySteps failed: {e}")
            return False
        except Exception as e:
            print(f"NoNanAfterManySteps error: {e}")
            return False

    def ParamsActuallyChange(self, steps: int = 10):
        try:
            torch.manual_seed(111)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.01, useHebbian=True, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            with torch.no_grad():
                key_params = {
                    "mhsa_q_proj": next(p for p in model.temporal_blocks[0].mhsa.q_proj.parameters() if p.dim() >= 2),
                    "fusion_base": model.fusion.base_weights,
                    "output_proj_w": next(p for p in model.output_proj[0].parameters() if p.dim() >= 2),
                    "head_w": head.weight}
                init_norms = {k: v.norm().item() for k, v in key_params.items()}

            for _ in range(steps):
                x = torch.randn(8, 16, 64, device=self.device)
                td = torch.randn(8, device=self.device)
                y = torch.randn(8, 12, device=self.device)
                pred = head(self.AttentionForward(model, x, tdError=td))
                loss = F.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                model.ClipGrads()
                opt.step()

            with torch.no_grad():
                new_norms = {k: v.norm().item() for k, v in key_params.items()}
            changed = any(abs(new_norms[k] - init_norms[k]) > 1e-6 for k in init_norms)
            assert changed, "Key parameters seem unchanged; suspected no updates."
            print("ParamsActuallyChange passed.")
            return True
        except AssertionError as e:
            print(f"ParamsActuallyChange failed: {e}")
            return False
        except Exception as e:
            print(f"ParamsActuallyChange error: {e}")
            return False

    def TestNormalTrainingConvergence(self, steps: int = 120, logEvery: int = 30):
        try:
            torch.manual_seed(222)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.01, useHebbian=True, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            B = 16
            data_x = torch.randn(B, 16, 64, device=self.device)
            data_y = torch.randn(B, 12, device=self.device)
            data_td = torch.randn(B, device=self.device)

            with torch.no_grad():
                start = F.mse_loss(head(self.AttentionForward(model, data_x, tdError=data_td)), data_y).item()

            for t in range(1, steps + 1):
                pred = head(self.AttentionForward(model, data_x, tdError=data_td))
                loss = F.mse_loss(pred, data_y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                model.ClipGrads()
                opt.step()
                if (t % logEvery) == 0 or t == 1:
                    print(f"[AttentionTrain] step {t}/{steps} | mse={loss.item():.6f}")

            with torch.no_grad():
                end = F.mse_loss(head(self.AttentionForward(model, data_x, tdError=data_td)), data_y).item()

            print(f"\n[AttentionTrain] loss start={start:.6f} -> end={end:.6f}")
            assert end <= 0.8 * start, "Training did not converge enough (<20% drop)."
            print("TestNormalTrainingConvergence passed.")
            return True
        except AssertionError as e:
            print(f"TestNormalTrainingConvergence failed: {e}")
            return False
        except Exception as e:
            print(f"TestNormalTrainingConvergence error: {e}")
            return False

    def WrapperForwardEqualWhenNoInitRank(self):
        try:
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.0, useHebbian=False).to(self.device)
            base.eval()
            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=False).to(self.device)
            wrapper.eval()

            x = torch.randn(3, 16, 64, device=self.device)
            kpm = torch.zeros(3, 16, dtype=torch.bool, device=self.device)
            args = self.AttentionInputs(3, 16, 64, x.dtype)
            with torch.no_grad():
                y_base = base(x, keyPaddingMask=kpm, tdError=None, **args)
                y_wrap = wrapper(x, keyPaddingMask=kpm, tdError=None, **args)

            max_abs = (y_base - y_wrap).abs().max().item()
            assert max_abs < 1e-6, f"Wrapper forward differs when ranks=0: max_abs={max_abs:.3e}"
            print("WrapperForwardEqualWhenNoInitRank passed.")
            return True
        except AssertionError as e:
            print(f"WrapperForwardEqualWhenNoInitRank failed: {e}")
            return False
        except Exception as e:
            print(f"WrapperForwardEqualWhenNoInitRank error: {e}")
            return False

    def WrapperAPIBasics(self):
        try:
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.0, useHebbian=False).to(self.device)
            base.eval()
            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=True).to(self.device)
            wrapper.train()

            r = wrapper.Update("ranks")["ranks"]
            assert all(row.get("q",0)==0 and row.get("k",0)==0 and row.get("v",0)==0 and row.get("o",0)==0 for row in r["perLayer"])

            wrapper.Update("grow", growFactor=2.0, addEach=1)
            r2 = wrapper.Update("ranks")["ranks"]
            assert all(row["q"] >= 1 and row["k"] >= 1 and row["v"] >= 1 and row["o"] >= 1 for row in r2["perLayer"])

            wrapper.Update("accumulategrads")

            st = wrapper.Update("set", evThreshold=0.85, gradEma=0.8, **{"maxRank:q":64, "maxRank:k":64, "maxRank:v":64, "maxRank:o":64})
            assert st["ok"]

            wrapper.Update("rollback")
            r3 = wrapper.Update("ranks")["ranks"]
            assert all(row["q"] == 0 and row["k"] == 0 and row["v"] == 0 and row["o"] == 0 for row in r3["perLayer"])

            print("WrapperAPIBasics passed.")
            return True
        except AssertionError as e:
            print("WrapperAPIBasics failed:\n", e)
            return False
        except Exception as e:
            print("WrapperAPIBasics error:\n", e)
            return False

    def WrapperManualGrowTrainAndCommit(self):
        try:
            torch.manual_seed(11)
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.0, useHebbian=False).to(self.device)
            base.eval()

            wrapper = AttentionOnlineWrapper(base=base, initRankEach=2, autoRank=False).to(self.device)
            wrapper.train()

            head = nn.Linear(64, 12).to(self.device).train()
            opt = torch.optim.Adam(list(wrapper.CandParameters()) + list(head.parameters()), lr=3e-3)

            _ = wrapper.Update("grow", growFactor=2.0, addEach=0)

            for _ in range(6):
                x = torch.randn(8, 16, 64, device=self.device)
                td = torch.randn(8, device=self.device)
                y = torch.randn(8, 12, device=self.device)

                pred = head(self.AttentionForward(wrapper, x, tdError=td))
                loss = F.mse_loss(pred, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            expected = []
            for li in range(wrapper.layerCount):
                per = {}
                for site in ("q", "k", "v", "o"):
                    per[site] = wrapper.ComposeOne(site, li).detach().clone()
                expected.append(per)

            res = wrapper.Update("commit")
            assert res["ok"] and res["commit_stats"]["committed_triples"] > 0, "Nothing committed."

            per_layer_rp, sums = self.MhsaAllRanksAndParams(base)
            print(f"[ManualCommit] committed_triples={int(res['commit_stats']['committed_triples'])}, "f"committed_rank={int(res['commit_stats']['committed_rank'])}")
            print(f"[Injected ranks per-layer] {per_layer_rp}; sums(q,k,v,o)={sums}")

            atol, rtol = 1e-6, 1e-4
            for li, blk in enumerate(base.temporal_blocks):
                mhsa = blk.mhsa
                got_q = self.DeltaFromLinearAdapter(mhsa.q_adapter); exp_q = expected[li]["q"]
                got_k = self.DeltaFromLinearAdapter(mhsa.k_adapter); exp_k = expected[li]["k"]
                got_v = self.DeltaFromLinearAdapter(mhsa.v_adapter); exp_v = expected[li]["v"]
                got_o = self.DeltaFromLinearAdapter(mhsa.o_adapter); exp_o = expected[li]["o"]

                if not torch.allclose(exp_q, torch.zeros_like(exp_q)): assert torch.allclose(got_q, exp_q, atol=atol, rtol=rtol), f"[L{li}] q delta mismatch"
                if not torch.allclose(exp_k, torch.zeros_like(exp_k)): assert torch.allclose(got_k, exp_k, atol=atol, rtol=rtol), f"[L{li}] k delta mismatch"
                if not torch.allclose(exp_v, torch.zeros_like(exp_v)): assert torch.allclose(got_v, exp_v, atol=atol, rtol=rtol), f"[L{li}] v delta mismatch"
                if not torch.allclose(exp_o, torch.zeros_like(exp_o)): assert torch.allclose(got_o, exp_o, atol=atol, rtol=rtol), f"[L{li}] o delta mismatch"

            r = wrapper.Update("ranks")["ranks"]
            assert all(row["q"] == 0 and row["k"] == 0 and row["v"] == 0 and row["o"] == 0 for row in r["perLayer"])

            base.eval(); wrapper.eval()
            x_chk = torch.randn(2, 16, 64, device=self.device)
            kpm_chk = torch.zeros(2, 16, dtype=torch.bool, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x_chk.dtype)
            with torch.no_grad():
                y0 = base(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
                y1 = wrapper(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
            assert torch.allclose(y0, y1, atol=1e-6, rtol=1e-4), "base vs wrapper mismatch after commit."

            print("WrapperManualGrowTrainAndCommit passed.")
            return True
        except AssertionError as e:
            print("WrapperManualGrowTrainAndCommit failed:\n", e)
            return False
        except Exception as e:
            print("WrapperManualGrowTrainAndCommit error:\n", e)
            return False

    def WrapperAutoInjectAndCommit(self):
        try:
            torch.manual_seed(2025)
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.0, useHebbian=False).to(self.device)

            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=True, evThreshold=0.9).to(self.device)

            _ = wrapper.Update("autogrow")
            r0 = wrapper.Update("ranks")["ranks"]
            seed_per_layer = [(row.get("q", 0), row.get("k", 0), row.get("v", 0), row.get("o", 0)) for row in r0["perLayer"]]
            seed_sums = (sum(r[0] for r in seed_per_layer),
                        sum(r[1] for r in seed_per_layer),
                        sum(r[2] for r in seed_per_layer),
                        sum(r[3] for r in seed_per_layer))
            print(f"[Seed ranks per-layer] {seed_per_layer}; sums(q,k,v,o)={seed_sums}")

            head = nn.Linear(64, 12).to(self.device)
            wrapper.train(); head.train()
            opt = torch.optim.Adam(list(wrapper.CandParameters()) + list(head.parameters()), lr=5e-3)

            for _ in range(8):
                x = torch.randn(16, 16, 64, device=self.device)
                td = torch.randn(16, device=self.device)
                y = torch.randn(16, 12, device=self.device)

                pred = head(self.AttentionForward(wrapper, x, tdError=td))
                loss = F.mse_loss(pred, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            res = wrapper.Update("commit")
            assert res["ok"], "Commit failed."
            stats = res.get("commit_stats", {})
            committed_rank = int(stats.get("committed_rank", 0))
            committed_triples = int(stats.get("committed_triples", 0))
            print(f"[AutoCommit] committed_triples={committed_triples}, committed_rank={committed_rank}")

            per_layer_rp, sums = self.MhsaAllRanksAndParams(base)
            print(f"[Injected ranks per-layer] {per_layer_rp}; sums(q,k,v,o)={sums}")
            assert (committed_rank > 0) == (sum(sums) > 0), "Committed rank/stat mismatch with adapters."

            base.eval(); wrapper.eval()
            x_chk = torch.randn(2, 16, 64, device=self.device)
            kpm_chk = torch.zeros(2, 16, dtype=torch.bool, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x_chk.dtype)
            with torch.no_grad():
                y0 = base(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
                y1 = wrapper(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
            max_abs = (y0 - y1).abs().max().item()
            assert max_abs < 1e-6, f"Wrapper vs base mismatch after auto-commit: {max_abs:.3e}"

            print("WrapperAutoInjectAndCommit passed.")
            return True
        except AssertionError as e:
            print("WrapperAutoInjectAndCommit failed:\n", e)
            return False
        except Exception as e:
            print("WrapperAutoInjectAndCommit error:\n", e)
            return False

    def WrapperPipelineCompatible(self):
        try:
            torch.manual_seed(13)
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.01, useHebbian=True).to(self.device)

            wrapper = AttentionOnlineWrapper(base=base, initRankEach=2, autoRank=False).to(self.device)

            for site in wrapper.sites:
                for li in range(wrapper.layerCount):
                    for s_param in wrapper.cand[site][li]["s"]:
                        s_param.requires_grad_(False)

            wrapper.train(); 
            head = nn.Linear(64, 10).to(self.device)
            head.train()

            opt = torch.optim.Adam(list(head.parameters()) + list(wrapper.CandParameters()), lr=1e-3)

            x  = torch.randn(6, 16, 64, device=self.device)
            td = torch.randn(6, device=self.device)
            y  = torch.randn(6, 10, device=self.device)

            pred = head(self.AttentionForward(wrapper, x, tdError=td))
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Head grad invalid with wrapper."

            opt.step()
            print("WrapperPipelineCompatible passed.")
            return True

        except AssertionError as e:
            print("WrapperPipelineCompatible failed:", e)
            return False
        except Exception as e:
            print("WrapperPipelineCompatible error:", e)
            return False


    def GradCoverageReportAttention(self, min_ratio: float = 0.65):
        try:
            torch.manual_seed(7)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3,hebbianRate=0.01, useHebbian=True, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            x  = torch.randn(8, 16, 64, device=self.device)
            td = torch.randn(8, device=self.device)
            y  = torch.randn(8, 12, device=self.device)

            pred = head(self.AttentionForward(model, x, tdError=td))
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.ClipGrads()

            named = dict(list(model.named_parameters()) + [('head.'+k, v) for k,v in head.named_parameters()])
            total_trainable = sum(1 for p in named.values() if p.requires_grad)
            total_with_grad = sum(1 for p in named.values() if (p.requires_grad and (p.grad is not None)))
            ratio = total_with_grad / max(1, total_trainable)

            must_have = [
                "temporal_blocks.0.mhsa.q_proj.weight",
                "temporal_blocks.0.mhsa.k_proj.weight",
                "temporal_blocks.0.mhsa.v_proj.weight",
                "temporal_blocks.0.mhsa.out_proj.weight",
                "temporal_blocks.0.ssm.in_proj.weight",
                "routing.transformation",
                "fusion.base_weights",
                "context_proj.0.weight",
                "static_mixer.0.weight",
                "output_proj.0.weight",
                "head.weight",]
            missing = [n for n in must_have if (n in named) and (named[n].grad is None)]
            assert len(missing) == 0, f"The key layer does not get the gradient: {missing}"
            assert ratio >= min_ratio, f"Gradient coverage is too low: {ratio:.2%} < {min_ratio:.2%}"

            print(f"GradCoverageReportAttention passed. grad_ratio={ratio:.2%}")
            return True
        except AssertionError as e:
            print(f"GradCoverageReportAttention failed: {e}")
            return False
        except Exception as e:
            print(f"GradCoverageReportAttention error: {e}")
            return False

    def MaskInvariance(self):
        try:
            torch.manual_seed(8)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, hebbianRate=0.01, useHebbian=False).to(self.device)
            model.eval()
            B, S, E = 3, 16, 64
            x1 = torch.randn(B, S, E, device=self.device)
            x2 = x1.clone()

            kpm = torch.zeros(B, S, dtype=torch.bool, device=self.device)
            kpm[:, -4:] = True
            x2[:, -4:, :] += torch.randn(B, 4, E, device=self.device) * 10.0

            args = self.AttentionInputs(B, S, E, x1.dtype)
            with torch.no_grad():
                y1 = model(x1, keyPaddingMask=kpm, **args)
                y2 = model(x2, keyPaddingMask=kpm, **args)

            max_abs = (y1 - y2).abs().max().item()
            assert max_abs < 5e-5, f"Mask invariance fails, max_abs={max_abs:.3e}"
            print("MaskInvariance passed.")
            return True
        except AssertionError as e:
            print(f"MaskInvariance failed: {e}")
            return False
        except Exception as e:
            print(f"MaskInvariance error: {e}")
            return False

    def LowrankFullrankConsistency(self):
        try:
            torch.manual_seed(9)

            attn = MultiHeadAttention(embedDim=64, numHeads=4, lowRank=True, rank=4, useHebbian=True).to(self.device)
            attn.eval()

            B, S, E = 2, 12, 64
            x = torch.randn(B, S, E, device=self.device)

            with torch.no_grad():
                for _ in range(2):
                    _ = attn(x, x, x, keyPaddingMask=None, tdError=torch.zeros(B, device=self.device), uncertainty=torch.zeros(B, device=self.device))

            with torch.no_grad():
                U0 = attn.U.detach().clone()
                V0 = attn.V.detach().clone()
                W0 = attn.hebbian_weights.detach().clone()
                use_lr0 = bool(attn.use_low_rank)
                use_hebb0 = bool(attn.use_hebbian)
                rate0 = float(attn.base_hebbian_rate)

            def restore_state(use_low_rank: bool):
                attn.EnsureB(B, device=x.device, dtype=x.dtype)
                attn.U.copy_(U0)
                attn.V.copy_(V0)
                attn.hebbian_weights.copy_(W0)
                attn.use_low_rank = bool(use_low_rank)

            def forward_freeze_hebbian_update():
                old_use = attn.use_hebbian
                old_rate = attn.base_hebbian_rate
                attn.use_hebbian = False 
                try:
                    return attn(x, x, x, keyPaddingMask=None, tdError=torch.zeros(B, device=self.device), uncertainty=torch.zeros(B, device=self.device))
                finally:
                    attn.use_hebbian = old_use
                    attn.base_hebbian_rate = old_rate

            with torch.no_grad():
                restore_state(use_low_rank=True)
                y_low = forward_freeze_hebbian_update()

                restore_state(use_low_rank=True)
                attn.LowrankToFullrank()
                y_full = forward_freeze_hebbian_update()

                max_abs = (y_low - y_full).abs().max().item()
                assert max_abs < 1e-5, (
                    f"[TestAttentionMTool.LowrankFullrankConsistency] "
                    f"Low rank->full rank values inconsistent, max_abs={max_abs:.3e}")

                restore_state(use_low_rank=True)
                attn.LowrankToFullrank()
                y_full2 = forward_freeze_hebbian_update()

                attn.FullrankToLowrank()
                y_low2 = forward_freeze_hebbian_update()

                max_abs2 = (y_full2 - y_low2).abs().max().item()
                assert max_abs2 < 1e-5, (
                    f"[TestAttentionMTool.LowrankFullrankConsistency] "
                    f"Full rank->low rank values inconsistent, max_abs={max_abs2:.3e}")

            print("LowrankFullrankConsistency passed.")
            return True

        except AssertionError as e:
            print(f"LowrankFullrankConsistency failed: {e}")
            return False
        except Exception as e:
            print(f"LowrankFullrankConsistency error: {e}")
            return False

    def HebbianMemoryLifecycleAttention(self):
        try:
            attn = MultiHeadAttention(embedDim=64, numHeads=4, lowRank=True, rank=4,useHebbian=True, hebbianRate=0.05).to(self.device)
            attn.train()
            B,S,E = 2, 10, 64
            x = torch.randn(B, S, E, device=self.device)

            U0 = attn.U.norm().item(); V0 = attn.V.norm().item()
            for _ in range(3):
                _ = attn(x, x, x, tdError=torch.randn(B, device=self.device), uncertainty=torch.randn(B, device=self.device))
            U1 = attn.U.norm().item(); V1 = attn.V.norm().item()
            assert U1 > U0 + 1e-8 and V1 > V0 + 1e-8, "MHSA Hebbian(U/V) not growing"

            attn.ResetHebbianMemory()
            assert attn.U.abs().max().item() < 1e-12 and attn.V.abs().max().item() < 1e-12, "MHSA Hebbian(U/V) not cleared"

            fusion = HebbianFusion(numModes=3, embedDim=64, hebbianRate=0.1, useHebbian=True).to(self.device)
            fusion.train()
            inputs = torch.randn(4, 3, 64, device=self.device)
            n0 = fusion.hebbian_memory.norm().item()
            _ = fusion(inputs)
            n1 = fusion.hebbian_memory.norm().item()
            assert n1 > n0 + 1e-8, "Fusion Hebbian memory not growing"
            fusion.ResetHebbianMemory()
            assert fusion.hebbian_memory.abs().max().item() < 1e-12, "Fusion Hebbian memory not cleared"

            print("HebbianMemoryLifecycleAttention passed.")
            return True
        except AssertionError as e:
            print(f"HebbianMemoryLifecycleAttention failed: {e}")
            return False
        except Exception as e:
            print(f"HebbianMemoryLifecycleAttention error: {e}")
            return False

    def WrapperKeepsBaseEval(self):
        try:
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, hebbianRate=0.0, useHebbian=False).to(self.device)
            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=True).to(self.device)
            wrapper.train()
            assert wrapper.training and (not base.training), "base should remain eval() when wrapper.train()"
            print("WrapperKeepsBaseEval passed.")
            return True
        except AssertionError as e:
            print(f"WrapperKeepsBaseEval failed: {e}")
            return False
        except Exception as e:
            print(f"WrapperKeepsBaseEval error: {e}")
            return False

    def SmallBatchSafety(self):
        try:
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, hebbianRate=0.01, useHebbian=True).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.eval(); head.train()
            x = torch.randn(1, 16, 64, device=self.device)
            td = torch.randn(1, device=self.device)
            y = torch.randn(1, 12, device=self.device)

            pred = head(self.AttentionForward(model, x, tdError=td))
            loss = F.mse_loss(pred, y)
            head.zero_grad(set_to_none=True)
            loss.backward()
            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Head gradient is abnormal when batch=1"
            print("SmallBatchSafety passed.")
            return True
        except AssertionError as e:
            print(f"SmallBatchSafety failed: {e}")
            return False
        except Exception as e:
            print(f"SmallBatchSafety error: {e}")
            return False

    def RunAll(self):
        results = {
            "SimpleSSM": self.TestSimpleSSM(),
            "MultiHeadAttention": self.TestMultiHeadAttention(),
            "TemporalAttention": self.TestTemporalAttention(),
            "DynamicRouting": self.TestDynamicRouting(),
            "HebbianFusion": self.TestHebbianFusion(),
            "AttentionExtractorForward": self.TestAttentionExtractor(),
            "AttentionExtractorIOShapes": self.TestAttentionExtractorIOShapes(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NoNanAfterManySteps": self.NoNanAfterManySteps(),
            "ParamsActuallyChange": self.ParamsActuallyChange(),
            "NormalTrainingConvergence": self.TestNormalTrainingConvergence(),
            "WrapperForwardEqualWhenNoInitRank": self.WrapperForwardEqualWhenNoInitRank(),
            "WrapperAPIBasics": self.WrapperAPIBasics(),
            "WrapperManualGrowTrainAndCommit": self.WrapperManualGrowTrainAndCommit(),
            "WrapperAutoInjectAndCommit": self.WrapperAutoInjectAndCommit(),
            "WrapperPipelineCompatible": self.WrapperPipelineCompatible(),
            "GradCoverageReportAttention": self.GradCoverageReportAttention(),
            "MaskInvariance": self.MaskInvariance(),
            "LowrankFullrankConsistency": self.LowrankFullrankConsistency(),
            "HebbianMemoryLifecycleAttention": self.HebbianMemoryLifecycleAttention(),
            "WrapperKeepsBaseEval": self.WrapperKeepsBaseEval(),
            "PlasticityGate": self.TestPlasticityGate(),
            "SelectiveDoneReset": self.TestSelectiveDoneReset(),
            "ModulatorsPassThrough": self.TestModulatorsPassThrough(),
            "RoutingAndDistributedGates": self.TestRoutingAndDistributedGates(),
            "DualTimeConstantSSM": self.TestDualTimeConstantSSM(),
            "SmallBatchSafety": self.SmallBatchSafety(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nAttention module tests (with wrapper): {passed}/{len(results)} passed.")
        return results
