from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from FunctionTools import GetParameterSScale, SiteSpec, BaseOnlineWrapper


class GrowableLoRALinear(nn.Module):
    def __init__(self, targetLinear: nn.Linear):
        super().__init__()
        self.target = targetLinear
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()

        self.out_f = targetLinear.out_features
        self.in_f = targetLinear.in_features

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if addRank <= 0:
            return
        if init is None: init = {}

        dev = self.target.weight.device
        dt = self.target.weight.dtype

        A = init.get("A", torch.randn(addRank, self.in_f, device=dev, dtype=dt) * 1e-4)
        B = init.get("B", torch.zeros(self.out_f, addRank, device=dev, dtype=dt) * 1e-4)
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=A.device, dtype=A.dtype))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def DeltaWeight(self) -> Optional[torch.Tensor]:
        if len(self.A_list) == 0:
            return None
        delta = self.target.weight.new_zeros(self.out_f, self.in_f)
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            s_eff = torch.tanh(s) * GetParameterSScale(s) 
            delta = delta + s_eff * (B @ A)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.target.weight
        delta = self.DeltaWeight()
        if delta is not None:
            W = W + delta
        return F.linear(x, W, self.target.bias)


class SimpleSSM(nn.Module):
    def __init__(self, embedDim: int, convKernel: int = 3):
        super().__init__()
        E = embedDim
        self.alpha_log = nn.Parameter(torch.zeros(E))
        self.beta = nn.Parameter(torch.randn(E)*0.05)
        self.gamma  = nn.Parameter(torch.randn(E)*0.05)
        self.delta = nn.Parameter(torch.zeros(E))
        self.in_proj = nn.Linear(E, 2*E)
        self.out_norm = nn.LayerNorm(E)
        self.dw_conv = nn.Conv1d(E, E, convKernel, groups=E, padding=convKernel//2)

    def forward(self, x, keyPaddingMask: Optional[torch.Tensor]=None, tdError: Optional[torch.Tensor]=None, uncertainty: Optional[torch.Tensor]=None):
        B, S, E = x.shape
        dev = x.device

        def z_(t):
            if t is None: return torch.zeros(B, device=dev)
            t = t.detach().to(dev).float().view(B)
            return torch.tanh((t - t.mean()) / (t.std(unbiased=False).clamp_min(1e-6)))
        
        td_z = z_(tdError)
        unc_z = z_(uncertainty)

        gate_bias = (0.5 * td_z + 0.5 * (-unc_z)).view(B,1,1)
        alpha = torch.sigmoid(self.alpha_log)
        alpha_scale = torch.sigmoid(0.5*td_z.unsqueeze(-1) - 0.5*unc_z.unsqueeze(-1))
        alpha_eff = (alpha.unsqueeze(0) * (0.75 + 0.25*alpha_scale))

        u_and_g = self.in_proj(x)
        u, g = torch.chunk(u_and_g, 2, dim=-1)
        u = F.silu(u)

        keep3 = None
        if keyPaddingMask is not None:
            keep3 = (~keyPaddingMask).to(u.dtype).unsqueeze(-1) 
            u = u * keep3  

        u_conv = self.dw_conv(u.transpose(1,2)).transpose(1,2)
        u = u + 0.5 * u_conv

        g = torch.sigmoid(g + gate_bias)

        u_conv = self.dw_conv(u.transpose(1,2)).transpose(1,2)
        u = u + 0.5 * u_conv

        y = torch.empty_like(x)
        state = torch.zeros(B, E, device=dev)

        has_mask = keyPaddingMask is not None
        for t in range(S):
            keep = (1.0 if not has_mask else (~keyPaddingMask[:, t]).float().unsqueeze(-1))  # (B,1)
            upd = torch.sigmoid(self.beta).unsqueeze(0) * u[:, t, :]
            new_state = alpha_eff * state + (1 - alpha_eff) * upd
            state = keep * new_state + (1 - keep) * state 
            out_t = self.gamma.unsqueeze(0) * state + self.delta.unsqueeze(0) * u[:, t, :]
            y[:, t, :] = (out_t * g[:, t, :]) * keep

        return self.out_norm(y)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedDim: int, numHeads: int, hebbianRate: float = 0.01,
                 attnDropout: float = 0.1, tdScale: float = 5.0, lowRank: bool = True,
                 rank: int = 8, hebbPeriod: int = 4, useHebbian: bool = True,):
        super().__init__()
        assert embedDim % numHeads == 0, "AttentionModule embed_dim must be divisible by num_heads"
        self.embed_dim = embedDim
        self.num_heads = numHeads
        self.head_dim = embedDim // numHeads
        self.base_hebbian_rate = hebbianRate
        self.attn_dropout_p = attnDropout
        self.td_scale = tdScale
        self.rank = rank
        self.hebb_period = max(1, hebbPeriod)
        self.update_hebbian_flag = useHebbian

        self.use_low_rank = bool(lowRank)

        self.register_buffer("hebb_step", torch.tensor(0, dtype=torch.long), persistent=False)

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

        eye = torch.eye(self.head_dim)
        hebb_shape = (numHeads, self.head_dim, self.head_dim)
        self.register_buffer("hebbian_weights", eye.unsqueeze(0).repeat(hebb_shape[0], 1, 1),persistent=False)

        self.register_buffer("U", torch.zeros(numHeads, self.head_dim, rank), persistent=False)
        self.register_buffer("V", torch.zeros(numHeads, self.head_dim, rank), persistent=False)

        self.ResetParameters()

    def ModulateTauBias(self, tdError, uncertainty, B):
        dev = self.hebbian_weights.device
        def z_(t):
            if t is None:
                return torch.zeros(B, device=dev)
            t = t.detach().to(dev).float().view(-1)  
            if t.numel() == 1:
                t = t.expand(B)                       
            elif t.numel() != B:
                raise ValueError(f"Expected size {B}, got {t.numel()}")
            return torch.tanh((t - t.mean()) / (t.std(unbiased=False).clamp_min(1e-6)))
        
        td_z = z_(tdError)
        unc_z = z_(uncertainty)

        tau = 1.0 + 0.5*torch.tanh(self.temp_w_td*td_z + self.temp_w_unc*unc_z)
        bias = 0.5*torch.tanh(self.bias_w_td*td_z + self.bias_w_unc*unc_z)
        return tau.view(B,1,1,1), bias.view(B,1,1,1)


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
        device = self.hebbian_weights.device
        eye = torch.eye(self.head_dim, device=device)
        self.hebbian_weights.copy_(eye.unsqueeze(0).repeat(self.num_heads, 1, 1))
        with torch.no_grad():
            self.U.zero_()
            self.V.zero_()


    def ScaledDotAttn(
        self,
        q: torch.Tensor, # (B, H, Lq, D)
        k: torch.Tensor, # (B, H, Lk, D)
        v: torch.Tensor, # (B, H, Lk, D)
        mask: Optional[torch.Tensor], # (B, 1, 1, Lk)
        dropoutP: float,) -> Tuple[torch.Tensor, torch.Tensor]:

        d = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            mask_value = -torch.finfo(scores.dtype).max
            scores = scores.masked_fill(mask, mask_value)
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=dropoutP, training=self.training)
        context = torch.matmul(weights, v)
        return context, weights


    def ComputeNeuromodulation(self, tdError: Optional[torch.Tensor], B: int) -> torch.Tensor:
        device = self.hebbian_weights.device
        if tdError is None:
            return torch.ones(B, 1, 1, 1, device=device)

        td = tdError.detach().to(device)
        if td.dim() == 0:
            td = td.view(1)
        if td.size(0) == 1 and B > 1:
            td = td.expand(B)
        if td.size(0) != B:
            raise ValueError(f"tdError size {td.size(0)} != batch size {B}")

        if B == 1:
            td_scaled = (td / self.td_scale).clamp(-10, 10)
            neuromod = 1.0 + 0.5 * torch.tanh(td_scaled)
        else:
            td_mean = td.mean()
            td_std = td.std(unbiased=False).clamp_min(1e-8)
            td_norm = (td - td_mean) / td_std
            neuromod = 1.0 + 0.5 * torch.tanh(td_norm / self.td_scale)

        return neuromod.view(B, 1, 1, 1)


    @torch.no_grad()
    def UpdateHebbianWeights(self, v: torch.Tensor, q: torch.Tensor, alpha: float):
        if not self.update_hebbian_flag or alpha <= 0:
            return

        v = v.detach()
        q = q.detach()

        norm_factor = v.size(0) * v.size(2)  # B * S
        hebb = torch.einsum("bhse,bhsd->hde", v, q) / (norm_factor + 1e-8)

        if self.use_low_rank:
            U_grad = torch.einsum('hde,her->hdr', hebb, self.V)
            V_grad = torch.einsum('hde,hdr->her', hebb, self.U)
            U_new = torch.clamp(self.U + alpha * U_grad, -1.5, 1.5)
            V_new = torch.clamp(self.V + alpha * V_grad, -1.5, 1.5)
            self.U.copy_(U_new)
            self.V.copy_(V_new)
        else:
            W_new = (1.0 - alpha) * self.hebbian_weights + alpha * torch.tanh(hebb)
            self.hebbian_weights.copy_(W_new)


    def forward(self, query, key, value, keyPaddingMask: Optional[torch.Tensor]=None, tdError: Optional[torch.Tensor]=None, uncertainty: Optional[torch.Tensor]=None):
        B, L, _ = query.shape
        neuromod = self.ComputeNeuromodulation(tdError, B)

        q_lin = self.q_adapter(query) 
        k_lin = self.k_adapter(key)
        v_lin = self.v_adapter(value) 

        q = q_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if self.training and self.update_hebbian_flag and self.base_hebbian_rate > 0:
            self.hebb_step.add_(1)
            if int(self.hebb_step.item()) % self.hebb_period == 0:
                alpha = float(self.base_hebbian_rate * neuromod.mean())
                if keyPaddingMask is not None:
                    keep4 = (~keyPaddingMask).to(v.dtype).view(B, 1, L, 1)  # (B,1,L,1)
                    v_upd, q_upd = v * keep4, q * keep4
                else:
                    v_upd, q_upd = v, q
                self.UpdateHebbianWeights(v_upd, q_upd, alpha)

        q = q * neuromod

        if self.use_low_rank:
            vU = torch.einsum("bhse,her->bhsr", v, self.U)
            delt = torch.einsum("bhsr,hdr->bhsd", vU, self.V)
            v_fast = v + delt
        else:
            v_fast = torch.einsum("bhse,hde->bhsd", v, self.hebbian_weights)

        tau, bias = self.ModulateTauBias(tdError, uncertainty, B)
        q = q / tau

        d = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores + bias
        if keyPaddingMask is not None:
            mask = keyPaddingMask[:, None, None, :]  # bool (B,1,1,L)
            mask_val = torch.tensor(-1e9 if q.dtype != torch.float16 else -1e4, dtype=q.dtype, device=q.device)
            scores = scores.masked_fill(mask, mask_val)

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.attn_dropout_p if self.training else 0.0, training=self.training)
        context = torch.matmul(weights, v_fast)
        out = context.transpose(1, 2).reshape(B, L, self.embed_dim)

        out_lin = self.o_adapter(out)
        
        return out_lin


    @torch.no_grad()
    def LowrankToFullrank(self, residual: bool = True):
        W = torch.einsum('hdr,her->hde', self.U, self.V) # (H,D,D)
        if residual:
            eye = torch.eye(W.size(1), device=W.device).unsqueeze(0).expand(W.size(0), -1, -1)
            W = W + eye
        self.hebbian_weights.copy_(W)
        self.use_low_rank = False 
        
    @torch.no_grad()
    def FullrankToLowrank(self, residual: bool = True):
        M = self.hebbian_weights.clone() # (H,D,D)
        if residual:
            eye = torch.eye(M.size(1), device=M.device).unsqueeze(0).expand(M.size(0), -1, -1)
            M = M - eye  

        U_s, S, Vh = torch.linalg.svd(M, full_matrices=False) # (H,D,D), (H,D), (H,D,D)
        r = self.U.size(-1)
        Ur = U_s[:, :, :r] * S[:, None, :r].clamp_min(0).sqrt()
        Vr = Vh.transpose(-2, -1)[:, :, :r] * S[:, None, :r].clamp_min(0).sqrt()

        self.U.copy_(Ur)
        self.V.copy_(Vr)
        self.use_low_rank = True


class TemporalAttention(nn.Module):
    def __init__(self, embedDim: int, numHeads: int, layerIdx: int = 0, useHebbian: bool = True):
        super().__init__()

        td_scale = 5.0 / (layerIdx + 1)
        self.mhsa = MultiHeadAttention(embedDim, numHeads, tdScale=td_scale, useHebbian=useHebbian)
        self.ssm = SimpleSSM(embedDim)

        self.mix_gate = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.SiLU(),
            nn.Linear(embedDim, 1))
        
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embedDim)

    def forward(self, x, keyPaddingMask: Optional[torch.Tensor]=None, tdError: Optional[torch.Tensor]=None, uncertainty: Optional[torch.Tensor]=None):
        
        mhsa_out = self.mhsa(x, x, x, keyPaddingMask=keyPaddingMask,tdError=tdError, uncertainty=uncertainty)

        ssm_out  = self.ssm(x, keyPaddingMask=keyPaddingMask, tdError=tdError, uncertainty=uncertainty)

        w = torch.sigmoid(self.mix_gate(x)) # (B,S,1)

        y = w * mhsa_out + (1 - w) * ssm_out # (B,S,E)

        out = self.norm(x + self.dropout(y))
        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1).to(out.dtype)  # (B,S,1)
            out = out * keep
        return out


class DynamicRouting(nn.Module):
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
        
        self.register_buffer("last_weights", torch.zeros(1, inCaps, outCaps), persistent=False)

    def ResetParameters(self):
        nn.init.kaiming_uniform_(self.transformation, a=math.sqrt(5))

    @staticmethod
    def Squash(vectors: torch.Tensor) -> torch.Tensor:
        squared_norm = vectors.pow(2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm) / (torch.sqrt(squared_norm + 1e-8))
        return scale * vectors

    def forward(
        self,
        x: torch.Tensor, # (B,I,D)
        mask: Optional[torch.Tensor] = None, # (B,I) bool
        ) -> torch.Tensor: # (B,O,out_dim)
        
        B, I, D = x.shape
        assert I == self.I and D == self.in_dim, "AttentionModule capsule input dim mismatch"

        u_hat = torch.einsum("bid,iodc->bioc", x, self.transformation) # (B,I,O,C)

        logits = self.routing_logits.expand(B, -1, -1) # (B,I,O)
        
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1), -1e4)

        cumulative_weights = torch.zeros_like(logits)
        
        for r in range(self.iterations):
            # Stable softmax calculation
            weights = F.log_softmax(logits + cumulative_weights, dim=-1).exp() # (B,I,O)
            
            if mask is not None:
                weights = weights.masked_fill(mask.unsqueeze(-1), 0.0)
            
            # Normalize weights
            weight_sum = weights.sum(dim=1, keepdim=True) + 1e-8
            weights = weights / weight_sum
            
            s = torch.einsum("bioc,bio->boc", u_hat, weights) # (B,O,C)
            v = self.Squash(s)
            
            if r < self.iterations - 1:
                agreement = torch.einsum("bioc,boc->bio", u_hat, v)
                # Clip agreement values for stability
                agreement = torch.clamp(agreement, -5.0, 5.0)
                cumulative_weights = cumulative_weights + agreement
        
        self.last_weights = weights.detach().mean(0, keepdim=True)
            
        return v # (B,O,C)


class HebbianFusion(nn.Module):
    def __init__(self, numModes: int, embedDim: int, hebbianRate: float = 0.01, useHebbian: bool = True, momentum: float = 0.9):
        super().__init__()
        self.num_modes = numModes
        self.embed_dim = embedDim
        self.hebbian_rate = hebbianRate
        self.use_hebbian = useHebbian
        self.momentum = momentum

        self.ctx_q = nn.Linear(self.embed_dim, 1, bias=False)

        self.base_weights = nn.Parameter(torch.empty(numModes, embedDim, embedDim))
        self.register_buffer("hebbian_memory", torch.zeros(numModes, embedDim, embedDim))

        self.ResetParameters()

        self.gate_head = nn.Sequential(
            nn.Linear(4 * embedDim, 2 * embedDim),
            nn.GELU(),
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

    def ResetHebbianMemory(self):
        self.hebbian_memory.zero_()

    def EffectiveWeights(self) -> torch.Tensor:
        if not self.use_hebbian:
            return self.base_weights
        eff = self.momentum * self.base_weights + (1.0 - self.momentum) * self.hebbian_memory
        return torch.clamp(eff, -3.0, 3.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor: # inputs:(B,M,E)
        B, M, E = inputs.shape

        assert M == self.num_modes and E == self.embed_dim, "AttentionModule HebbianFusion shape mismatch"

        effW = self.EffectiveWeights()  # (M,E,E)
        weighted = torch.einsum("bme,mef->bmf", inputs, effW) # (B,M,E)

        alpha = torch.softmax(self.ctx_q(inputs).squeeze(-1) / math.sqrt(self.embed_dim), dim=1) # (B,M)
        context = torch.einsum("bme,bm->be", inputs, alpha).unsqueeze(1).expand(-1, M, -1)   

        gate_in = torch.cat([inputs, context, inputs - context, inputs * context], dim=-1) # (B,M,4E)
        gate_logits = self.gate_head(gate_in).squeeze(-1) # (B,M)
        gate_w = torch.softmax(gate_logits, dim=-1) # (B,M)

        fused = torch.einsum("bmf,bm->bf", weighted, gate_w) # (B,E)

        if self.use_hebbian and self.hebbian_rate > 0:
            with torch.no_grad():
                norm = max(1, B) * math.sqrt(E)
                hebb_term = torch.einsum("bme,bf->mef", inputs.float(), fused.float()) / (norm + 1e-8)
                mem_new = (1.0 - self.hebbian_rate) * self.hebbian_memory + self.hebbian_rate * hebb_term
                self.hebbian_memory.copy_(mem_new)

        return fused



class AttentionExtractor(nn.Module):
    def __init__(self,
                 embedDim: int = 1024,
                 sequenceLength: int = 16,
                 numHeads: int = 8,
                 temporalLayers: int = 3,
                 routingIterations: int = 3,
                 hebbianRate: float = 0.01,
                 useHebbian: bool = True,
                 gradientClipVal: float = 1.0,):
        super().__init__()

        self.num_caps = sequenceLength
        self.gradient_clip_val = gradientClipVal
        self.output_dim = embedDim
        self.use_hebbian = useHebbian
        self.num_heads = numHeads

        self.temporal_blocks: nn.ModuleList = nn.ModuleList([
            TemporalAttention(embedDim, numHeads, idx, useHebbian=useHebbian)
            for idx in range(temporalLayers)])

        self.routing = DynamicRouting(sequenceLength, embedDim, 4, embedDim, iterations=routingIterations)

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
            

    def ClipGrads(self):
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)


    def forward(
        self,
        x: torch.Tensor, # (B,S,E)
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor]=None,) -> torch.Tensor: # (B,) or scalar
        
        B, S, E = x.shape

        # Handle sequence padding
        if S % self.num_caps != 0:
            pad_len = self.num_caps - (S % self.num_caps)
            x = F.pad(x, (0,0,0,pad_len))
            if keyPaddingMask is not None:
                keyPaddingMask = F.pad(keyPaddingMask, (0,pad_len), value=True)
            S = x.size(1)

        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1).to(x.dtype)
            x = x * keep
        
        # Use detached version of tdError for checkpointing
        #td_error_detached = tdError.detach() if tdError is not None else None
        for blk in self.temporal_blocks:
            x = blk(x, keyPaddingMask=keyPaddingMask, tdError=tdError, uncertainty= uncertainty)

        # Create capsule mask
        chunk = S // self.num_caps

        if keyPaddingMask is not None:
            seg_mask = keyPaddingMask.reshape(B, self.num_caps, chunk) # (B,I,chunk)
        else:
            seg_mask = x.new_zeros(B, self.num_caps, chunk, dtype=torch.bool)

        valid = (~seg_mask).float()                                  
        valid_cnt = valid.sum(dim=2, keepdim=True) # (B,I,1)
        valid_cnt_safe = valid_cnt.clamp_min(1.0)

        h_seg = x.reshape(B, self.num_caps, chunk, E) # (B,I,chunk,E)
        caps = (h_seg * valid.unsqueeze(-1)).sum(dim=2) / valid_cnt_safe # (B,I,E) masked mean
        caps_mask = (valid_cnt.squeeze(-1) == 0)                      

        routed = self.routing(caps, caps_mask) # (B,4,E)
        routed = F.layer_norm(routed, (E,))

        # Fusion of different representations
        routed_mean = routed.mean(dim=1) # (B,E)
        
        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).to(x.dtype).unsqueeze(-1)  # (B,S,1)
            denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0) # (B,1,1)
            temp_mean = (x * keep).sum(dim=1) / denom.squeeze(-1)  
        else:
            temp_mean = x.mean(dim=1)
        
        fusion_in = torch.stack([
            temp_mean, 
            routed_mean, 
            temp_mean + routed_mean], dim=1)  # (B,3,E)
        
        fused = self.fusion(fusion_in)  # (B,E)

        context = self.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))  # (B,E)

        logits = self.static_mixer(context).view(B, self.num_heads, 3)
        strat_w = torch.softmax(logits, dim=-1)  # (B,H,3)

        feats = torch.stack([temp_mean, routed_mean, fused], dim=1)  # (B,3,E)
        mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)  # (B,H,E)
        out = mixed_per_head.mean(dim=1)  # (B,E)

        return self.output_proj(out)

    def ResetFastWeights(self) -> None:
        for blk in self.temporal_blocks:
            blk.mhsa.ResetHebbianMemory()
        self.fusion.ResetHebbianMemory()

    def AttenLowrankToFullrank(self, residual: bool = True):
        for blk in self.temporal_blocks:
            blk.mhsa.LowrankToFullrank(residual)
        
    def AttenFullrankToLowrank(self, residual: bool = True):
        for blk in self.temporal_blocks:
            blk.mhsa.FullrankToLowrank(residual)


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
            A = nn.Parameter(torch.randn(addRank, E, device=device, dtype=dtype) * 1e-4) # (r, inDim)
            B = nn.Parameter(torch.zeros(E, addRank, device=device, dtype=dtype) * 1e-4) # (outDim, r)
            s = nn.Parameter(torch.tensor(1e-2, device=device, dtype=dtype))
            return A, B, s

        def compose_linear(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            s_eff = torch.tanh(s) * GetParameterSScale(s) 
            return s_eff * (b @ a)

        return {
            "q": SiteSpec("q", L, E, E, self.maxRankQ, alloc_linear, compose_linear),
            "k": SiteSpec("k", L, E, E, self.maxRankK, alloc_linear, compose_linear),
            "v": SiteSpec("v", L, E, E, self.maxRankV, alloc_linear, compose_linear),
            "o": SiteSpec("o", L, E, E, self.maxRankO, alloc_linear, compose_linear),}

    def ForwardWithDeltas(
        self,
        x: torch.Tensor,  # (B,S,E)
        keyPaddingMask: Optional[torch.Tensor],
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]],
        **kwargs,) -> torch.Tensor:
        B, S, E = x.shape
        num_caps = int(self.base.num_caps)

        if S % num_caps != 0:
            pad_len = num_caps - (S % num_caps)
            x = F.pad(x, (0, 0, 0, pad_len))
            if keyPaddingMask is not None:
                keyPaddingMask = F.pad(keyPaddingMask, (0, pad_len), value=True)
            S = x.size(1)

        if keyPaddingMask is not None:
            keep0 = (~keyPaddingMask).unsqueeze(-1).to(x.dtype)
            h = x * keep0
        else:
            h = x

        for layerIdx, blk in enumerate(self.base.temporal_blocks):
            h = self.ForwardBlockWithDeltas(blk=blk, x=h, keyPaddingMask=keyPaddingMask, tdError=tdError, uncertainty=uncertainty, delta=deltasPerLayer[layerIdx],)

        chunk = S // num_caps
        if keyPaddingMask is not None:
            seg_mask = keyPaddingMask.reshape(B, num_caps, chunk)
        else:
            seg_mask = h.new_zeros(B, num_caps, chunk, dtype=torch.bool)

        valid = (~seg_mask).float()
        valid_cnt = valid.sum(dim=2, keepdim=True).clamp_min(1.0)
        h_seg = h.reshape(B, num_caps, chunk, E)
        caps = (h_seg * valid.unsqueeze(-1)).sum(dim=2) / valid_cnt
        caps_mask = (valid_cnt.squeeze(-1) == 0)

        routed = self.base.routing(caps, caps_mask)  # (B,4,E)
        routed = F.layer_norm(routed, (E,))

        routed_mean = routed.mean(dim=1)
        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).to(h.dtype).unsqueeze(-1) 
            denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0) 
            temp_mean = (h * keep).sum(dim=1) / denom.squeeze(-1) 
        else:
            temp_mean = h.mean(dim=1)

        fusion_in = torch.stack([temp_mean, routed_mean, temp_mean + routed_mean], dim=1)  # (B,3,E)
        fused = self.base.fusion(fusion_in)

        context = self.base.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))  # (B,E)

        logits = self.base.static_mixer(context).view(B, self.base.num_heads, 3)
        strat_w = torch.softmax(logits, dim=-1)

        feats = torch.stack([temp_mean, routed_mean, fused], dim=1)  # (B,3,E)
        mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)
        out = mixed_per_head.mean(dim=1)

        return self.base.output_proj(out)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        r = int(a.size(0))
        if r <= 0 or a.numel() == 0 or b.numel() == 0 or abs(float(scale)) < 1e-12:
            return False

        blk = self.base.temporal_blocks[layerIdx]
        mhsa = blk.mhsa
        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "q":
            mhsa.q_adapter.Grow(addRank=r, init=init, freezeOld=False)
        elif site == "k":
            mhsa.k_adapter.Grow(addRank=r, init=init, freezeOld=False)
        elif site == "v":
            mhsa.v_adapter.Grow(addRank=r, init=init, freezeOld=False)
        elif site == "o":
            mhsa.o_adapter.Grow(addRank=r, init=init, freezeOld=False)
        else:
            raise ValueError(f"Unknown site: {site}")
        return True

    def ForwardBlockWithDeltas(
        self,
        blk,
        x: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        delta: Dict[str, Optional[torch.Tensor]],) -> torch.Tensor:
        mhsa = blk.mhsa
        B, S, _ = x.shape

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

        neuromod = mhsa.ComputeNeuromodulation(tdError, B)

        q_lin = F.linear(x, Wq, mhsa.q_proj.bias)
        k_lin = F.linear(x, Wk, mhsa.k_proj.bias)
        v_lin = F.linear(x, Wv, mhsa.v_proj.bias)

        q = q_lin.view(B, S, mhsa.num_heads, mhsa.head_dim).transpose(1, 2)
        k = k_lin.view(B, S, mhsa.num_heads, mhsa.head_dim).transpose(1, 2)
        v = v_lin.view(B, S, mhsa.num_heads, mhsa.head_dim).transpose(1, 2)

        if mhsa.training and mhsa.update_hebbian_flag and mhsa.base_hebbian_rate > 0:
            mhsa.hebb_step.add_(1)
            if int(mhsa.hebb_step.item()) % mhsa.hebb_period == 0:
                alpha = float(mhsa.base_hebbian_rate * neuromod.mean())
                if keyPaddingMask is not None:
                    keep4 = (~keyPaddingMask).to(v.dtype).view(B, 1, S, 1)  # (B,1,S,1)
                    v_upd, q_upd = v * keep4, q * keep4
                else:
                    v_upd, q_upd = v, q
                mhsa.UpdateHebbianWeights(v_upd, q_upd, alpha)

        q = q * neuromod
        if mhsa.use_low_rank:
            vU = torch.einsum("bhse,her->bhsr", v, mhsa.U)
            delt = torch.einsum("bhsr,hdr->bhsd", vU, mhsa.V)
            v_fast = v + delt
        else:
            v_fast = torch.einsum("bhse,hde->bhsd", v, mhsa.hebbian_weights)

        tau, bias = mhsa.ModulateTauBias(tdError, uncertainty, B)
        q = q / tau

        d = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        scores = scores + bias
        if keyPaddingMask is not None:
            mask = keyPaddingMask[:, None, None, :]
            mask_val = torch.tensor(-1e9 if q.dtype != torch.float16 else -1e4, dtype=q.dtype, device=q.device)
            scores = scores.masked_fill(mask, mask_val)

        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=mhsa.attn_dropout_p if mhsa.training else 0.0, training=mhsa.training)
        context = torch.matmul(weights, v_fast)
        out = context.transpose(1, 2).reshape(B, S, mhsa.embed_dim)

        mhsa_out = F.linear(out, Wo, mhsa.out_proj.bias)

        ssm_out = blk.ssm(x, keyPaddingMask=keyPaddingMask, tdError=tdError, uncertainty=uncertainty)
        w = torch.sigmoid(blk.mix_gate(x))
        y = w * mhsa_out + (1 - w) * ssm_out

        out = blk.norm(x + blk.dropout(y))
        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1).to(out.dtype) 
            out = out * keep
        return out



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
            s_eff = torch.tanh(s.detach()) * GetParameterSScale(s.detach())
            delta = delta + s_eff * (B @ A)
        return delta

    def TestSimpleSSM(self):
        try:
            ssm = SimpleSSM(self.E).to(self.device)
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device)
            kpm[:, -3:] = True
            y = ssm(x, keyPaddingMask=kpm, tdError=torch.randn(self.B, device=self.device))
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
            attn = MultiHeadAttention(embedDim=self.E, numHeads=self.H, lowRank=True, rank=self.R).to(self.device)
            y1 = attn(x, x, x, keyPaddingMask=kpm, tdError=None)
            assert y1.shape == (self.B, self.S, self.E)
            attn.LowrankToFullrank(residual=True)
            y2 = attn(x, x, x, keyPaddingMask=kpm, tdError=None)
            assert y2.shape == (self.B, self.S, self.E)
            attn.FullrankToLowrank(residual=True)
            y3 = attn(x, x, x, keyPaddingMask=kpm, tdError=None)
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
            y = ta(x, keyPaddingMask=kpm, tdError=torch.randn(self.B, device=self.device))
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
            y = model(x, keyPaddingMask=kpm, tdError=td)
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

            out = model(x, keyPaddingMask=None, tdError=td)
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

                pred = head(model(x, tdError=td))
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
                pred = head(model(x, tdError=td))
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
                start = F.mse_loss(head(model(data_x, tdError=data_td)), data_y).item()

            for t in range(1, steps + 1):
                pred = head(model(data_x, tdError=data_td))
                loss = F.mse_loss(pred, data_y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                model.ClipGrads()
                opt.step()
                if (t % logEvery) == 0 or t == 1:
                    print(f"[AttentionTrain] step {t}/{steps} | mse={loss.item():.6f}")

            with torch.no_grad():
                end = F.mse_loss(head(model(data_x, tdError=data_td)), data_y).item()

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
            with torch.no_grad():
                y_base = base(x, keyPaddingMask=kpm, tdError=None)
                y_wrap = wrapper(x, keyPaddingMask=kpm, tdError=None)

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

                pred = head(wrapper(x, tdError=td))
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
            with torch.no_grad():
                y0 = base(x_chk, keyPaddingMask=kpm_chk, tdError=None)
                y1 = wrapper(x_chk, keyPaddingMask=kpm_chk, tdError=None)
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

                pred = head(wrapper(x, tdError=td))
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
            with torch.no_grad():
                y0 = base(x_chk, keyPaddingMask=kpm_chk, tdError=None)
                y1 = wrapper(x_chk, keyPaddingMask=kpm_chk, tdError=None)
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

            pred = head(wrapper(x, tdError=td))
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

            pred = head(model(x, tdError=td))
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

            with torch.no_grad():
                y1 = model(x1, keyPaddingMask=kpm)
                y2 = model(x2, keyPaddingMask=kpm)

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
            B,S,E = 2, 12, 64
            x = torch.randn(B, S, E, device=self.device)
            with torch.no_grad():
                y_low = attn(x, x, x, keyPaddingMask=None, tdError=None)
                attn.LowrankToFullrank(residual=True)
                y_full = attn(x, x, x, keyPaddingMask=None, tdError=None)
            max_abs = (y_low - y_full).abs().max().item()
            assert max_abs < 1e-5, f"Low rank->full rank values are inconsistent, max_abs={max_abs:.3e}"

            with torch.no_grad():
                attn.FullrankToLowrank(residual=True)
                y_low2 = attn(x, x, x, keyPaddingMask=None, tdError=None)
            max_abs2 = (y_full - y_low2).abs().max().item()
            assert max_abs2 < 1e-5, f"Full rank-> low rank values are inconsistent, max_abs={max_abs2:.3e}"

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
            attn = MultiHeadAttention(embedDim=64, numHeads=4, lowRank=True, rank=4,useHebbian=True, hebbianRate=0.05, hebbPeriod=1).to(self.device)
            attn.train()
            B,S,E = 2, 10, 64
            x = torch.randn(B, S, E, device=self.device)

            U0 = attn.U.norm().item(); V0 = attn.V.norm().item()
            for _ in range(3):
                _ = attn(x, x, x, tdError=torch.randn(B, device=self.device))
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

            pred = head(model(x, tdError=td))
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
            "SmallBatchSafety": self.SmallBatchSafety(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nAttention module tests (with wrapper): {passed}/{len(results)} passed.")
        return results

