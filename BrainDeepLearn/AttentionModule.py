from __future__ import annotations
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.checkpoint

class MultiHeadAttention(nn.Module):
    """
    embedDim : Token embedding dimension E.
    numHeads : Number of attention heads H.
    hebbianRate : Base Hebbian learning-rate α₀. (default 0.01)
    attnDropout : Drop probability applied to attention weights. (default 0.1)
    tdScale : Scale factor dividing TD-error before tanh. (default 5.0)
    lowRank : Start with low-rank fast weights (can switch at runtime via use_low_rank).
    rank : Low-rank dimension R.
    hebbPeriod : Update Hebbian every N steps (N>=1).
    useHebbian : Master switch for Hebbian updates.
    """
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

        self.q_proj = nn.Linear(embedDim, embedDim)
        self.k_proj = nn.Linear(embedDim, embedDim)
        self.v_proj = nn.Linear(embedDim, embedDim)
        self.out_proj = nn.Linear(embedDim, embedDim)

        eye = torch.eye(self.head_dim)
        hebb_shape = (numHeads, self.head_dim, self.head_dim)
        self.register_buffer("hebbian_weights", eye.unsqueeze(0).repeat(hebb_shape[0], 1, 1),persistent=False)

        self.register_buffer("U", torch.zeros(numHeads, self.head_dim, rank), persistent=False)
        self.register_buffer("V", torch.zeros(numHeads, self.head_dim, rank), persistent=False)

        self.ResetParameters()


    def ResetParameters(self) -> None:
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_normal_(mod.weight, gain=1 / math.sqrt(2))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        self.ResetHebbianMemory()

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
            self.U.add_(alpha * (U_grad - self.U))
            self.V.add_(alpha * (V_grad - self.V))
            self.U.clamp_(-1.5, 1.5)
            self.V.clamp_(-1.5, 1.5)
        else:
            new_weights = (1 - alpha) * self.hebbian_weights + alpha * torch.tanh(hebb)
            self.hebbian_weights.copy_(new_weights)


    def ProJ(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        out: torch.Tensor = layer(x)
        return out.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,D)


    def forward(
        self,
        query: torch.Tensor, # (B, L, E)
        key: torch.Tensor, # (B, L, E)
        value: torch.Tensor, # (B, L, E)
        keyPaddingMask: Optional[torch.Tensor] = None, # (B, L) bool
        tdError: Optional[torch.Tensor] = None,) -> torch.Tensor:

        B, L, _ = query.shape

        neuromod = self.ComputeNeuromodulation(tdError, B)

        q: torch.Tensor = self.ProJ(self.q_proj, query) # (B,H,L,D)
        k: torch.Tensor = self.ProJ(self.k_proj, key) # (B,H,L,D)
        v: torch.Tensor = self.ProJ(self.v_proj, value) # (B,H,L,D)

        do_update = (self.update_hebbian_flag and self.base_hebbian_rate > 0)  
        if do_update:
            self.hebb_step.add_(1)
            if int(self.hebb_step.item()) % self.hebb_period == 0:
                alpha = float(self.base_hebbian_rate * neuromod.mean())
                self.UpdateHebbianWeights(v, q, alpha)

        q = q * neuromod

        if self.use_low_rank:
            vU = torch.einsum("bhse,her->bhsr", v, self.U) # (B,H,S,R)
            delt = torch.einsum("bhsr,hdr->bhsd", vU, self.V) # (B,H,S,D)
            v_fast: torch.Tensor = v + delt                    
        else:
            v_fast: torch.Tensor = torch.einsum("bhse,hde->bhsd", v, self.hebbian_weights)

        # Key padding mask → SDPA mask
        attn_mask = keyPaddingMask[:, None, None, :] if keyPaddingMask is not None else None

        context = F.scaled_dot_product_attention(
            q, k, v_fast,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False)

        out: torch.Tensor = context.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(out)


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
        self.attn: MultiHeadAttention = MultiHeadAttention(embedDim, numHeads, tdScale=td_scale, useHebbian= useHebbian)
        self.norm: nn.LayerNorm = nn.LayerNorm(embedDim)
        self.dropout: nn.Dropout = nn.Dropout(0.1)

    def forward(
        self,
        x: torch.Tensor, # (B,S,E)
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        attn_out = self.attn(x, x, x, keyPaddingMask, tdError)
        return self.norm(x + self.dropout(attn_out))


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
                cumulative_weights.add_(agreement)
        
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

        self.base_weights = nn.Parameter(torch.empty(numModes, embedDim, embedDim))
        self.register_buffer("hebbian_memory", torch.zeros(numModes, embedDim, embedDim))

        self.ResetParameters()

        self.gate_head = nn.Sequential(
            nn.Linear(4 * embedDim, 2 * embedDim),
            nn.GELU(),
            nn.Linear(2 * embedDim, 1))
        
        for m in self.gate_head:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="gelu")
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

        context = inputs.mean(dim=1, keepdim=True).expand(-1, M, -1) # (B,M,E)

        gate_in = torch.cat([inputs, context, inputs - context, inputs * context], dim=-1) # (B,M,4E)
        gate_logits = self.gate_head(gate_in).squeeze(-1) # (B,M)
        gate_w = torch.softmax(gate_logits, dim=-1) # (B,M)

        fused = torch.einsum("bmf,bm->bf", weighted, gate_w) # (B,E)

        if self.use_hebbian and self.hebbian_rate > 0:
            with torch.no_grad():
                norm = max(1, B) * math.sqrt(E)
                hebb_term = torch.einsum("bme,bf->mef", inputs.float(), fused.float()) / (norm + 1e-8)
                self.hebbian_memory.mul_(1.0 - self.hebbian_rate).add_(self.hebbian_rate * hebb_term)

        return fused


class MetaStrategySelector(nn.Module):
    def __init__(self, embedDim: int, numStrategies: int, numHeads: int = 4, *, useMeta: bool = False, ema: float = 0.05, offsetScale: float = 0.2, temperature: float = 1.0, confGate: Optional[float] = None):
        super().__init__()
        self.embed_dim = embedDim
        self.num_strategies = numStrategies
        self.num_heads = numHeads

        self.use_meta_update = useMeta
        self.ema = ema
        self.offset_scale = offsetScale
        self.temperature = temperature
        self.conf_gate = confGate  

        self.generator = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Linear(embedDim * 2, numHeads * numStrategies),)

        self.register_buffer("strategy_weights", torch.zeros(numHeads, numStrategies))
        self.ResetParameters()


    def ResetParameters(self):
        nn.init.uniform_(self.strategy_weights, -0.05, 0.05)
        for m in self.generator:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="gelu")
                nn.init.zeros_(m.bias)

    def ResetStrategyWeights(self):
        nn.init.normal_(self.strategy_weights, std=0.01)


    @staticmethod
    def MaskedSoftmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        x = logits / temperature

        if mask is not None:
            m = mask
            for _ in range(x.ndim - m.ndim):
                m = m.unsqueeze(0)
            m = m.expand_as(x)
            x = x.masked_fill(m, -1e4)

        x = x - x.max(dim=dim, keepdim=True).values
        ex = torch.exp(x)
        if mask is not None:
            ex = ex * (~m).float()
        denom = ex.sum(dim=dim, keepdim=True).clamp_min(1e-8)
        return ex / denom


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        assert x.shape[-1] == self.embed_dim, "embed dim mismatch"

        offsets = self.generator(x).view(B, self.num_heads, self.num_strategies) # (B,H,S)
        offsets = torch.tanh(offsets) * self.offset_scale

        params = self.strategy_weights + offsets  # (B,H,S)

        probs = self.MaskedSoftmax(params, mask, dim=-1, temperature=self.temperature) # (B,H,S)

        if self.use_meta_update:
            with torch.no_grad():
                if self.conf_gate is not None:
                    ent = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    max_ent = math.log(self.num_strategies)
                    conf = (1.0 - ent / max_ent).clamp(0.0, 1.0)  
                    gate = (conf.mean(dim=0) > float(self.conf_gate)).float() # (H,)
                    gate = gate.unsqueeze(-1)  # (H,1)
                else:
                    gate = None

                denom = probs.sum(dim=0) + 1e-8 # (H,S)
                batch_mean = (params * probs).sum(dim=0) / denom # (H,S)

                if gate is not None:
                    batch_mean = gate * batch_mean + (1 - gate) * self.strategy_weights

                self.strategy_weights.mul_(1.0 - self.ema).add_(self.ema * batch_mean)

        return probs  # (B,H,S)

    def SetUseMetaUpdate(self, flag: bool = True):
        self.use_meta_update = bool(flag)



class AttentionExtractor(nn.Module):
    def __init__(self,
                 embedDim: int = 512,
                 sequenceLength: int = 16,
                 numHeads: int = 8,
                 temporalLayers: int = 3,
                 routingIterations: int = 3,
                 hebbianRate: float = 0.01,
                 useHebbian: bool = True,
                 useMetaLearning: bool = True,
                 gradientClipVal: float = 1.0,):
        super().__init__()

        self.num_caps = sequenceLength
        self.use_meta_learning = useMetaLearning
        self.gradient_clip_val = gradientClipVal
        self.output_dim = embedDim
        self.use_hebbian = useHebbian

        self.temporal_blocks: nn.ModuleList = nn.ModuleList([
            TemporalAttention(embedDim, numHeads, idx, useHebbian=useHebbian)
            for idx in range(temporalLayers)])

        self.routing = DynamicRouting(sequenceLength, embedDim, 4, embedDim, iterations=routingIterations)

        self.fusion = HebbianFusion(numModes=3, embedDim=embedDim, hebbianRate=hebbianRate, useHebbian=useHebbian)

        self.meta_selector = MetaStrategySelector(embedDim, numStrategies=3, useMeta=useMetaLearning)
        self.context_proj = nn.Sequential(nn.Linear(embedDim * 2, embedDim), nn.GELU())

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
        tdError: Optional[torch.Tensor] = None,) -> torch.Tensor: # (B,) or scalar
        
        B, S, E = x.shape

        # Handle sequence padding
        if S % self.num_caps != 0:
            pad_len = self.num_caps - (S % self.num_caps)
            x = F.pad(x, (0,0,0,pad_len))
            if keyPaddingMask is not None:
                keyPaddingMask = F.pad(keyPaddingMask, (0,pad_len), value=True)
            S = x.size(1)

        h = x
        
        # Use detached version of tdError for checkpointing
        #td_error_detached = tdError.detach() if tdError is not None else None
        for blk in self.temporal_blocks:
            h = blk(h, keyPaddingMask=keyPaddingMask, tdError=tdError)

        # Create capsule mask
        chunk = S // self.num_caps

        if keyPaddingMask is not None:
            seg_mask = keyPaddingMask.reshape(B, self.num_caps, chunk) # (B,I,chunk)
        else:
            seg_mask = h.new_zeros(B, self.num_caps, chunk, dtype=torch.bool)

        valid = (~seg_mask).float()                                  
        valid_cnt = valid.sum(dim=2, keepdim=True) # (B,I,1)
        valid_cnt_safe = valid_cnt.clamp_min(1.0)

        h_seg = h.reshape(B, self.num_caps, chunk, E) # (B,I,chunk,E)
        caps = (h_seg * valid.unsqueeze(-1)).sum(dim=2) / valid_cnt_safe # (B,I,E) masked mean
        caps_mask = (valid_cnt.squeeze(-1) == 0)                      

        routed = self.routing(caps, caps_mask) # (B,4,E)
        routed = F.layer_norm(routed, (E,))

        # Fusion of different representations
        routed_mean = routed.mean(dim=1) # (B,E)
        temp_mean = h.mean(dim=1) # (B,E)
        
        fusion_in = torch.stack([
            temp_mean, 
            routed_mean, 
            temp_mean + routed_mean], dim=1)  # (B,3,E)
        
        fused = self.fusion(fusion_in)  # (B,E)

        context = self.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))
        strat_w = self.meta_selector(context)  # (B,H=4,S=3)
            
        feats = torch.stack([
            temp_mean, 
            routed_mean, 
            fused.detach()  # Detach fused to prevent double backprop
        ], dim=1)  # (B,3,E)
            
        mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)  # (B,H,E)
        out = mixed_per_head.mean(dim=1)

        return self.output_proj(out)

    def ResetFastWeights(self) -> None:
        """Reset all Hebbian weights and strategy weights in the module"""
        for blk in self.temporal_blocks:
            blk.attn.ResetHebbianMemory()
        self.fusion.ResetHebbianMemory()
        if self.use_meta_learning:
            self.meta_selector.ResetStrategyWeights()

    def AttenLowrankToFullrank(self, residual: bool = True):
        for blk in self.temporal_blocks:
            blk.attn.LowrankToFullrank(residual)
        
    def AttenFullrankToLowrank(self, residual: bool = True):
        for blk in self.temporal_blocks:
            blk.attn.FullrankToLowrank(residual)

    def SetUseMetaLearning(self, flag: bool = True):
        self.meta_selector.SetUseMetaUpdate(flag)



class TestAttentionMTool:
    def __init__(self):

        self.B = 2      
        self.S = 16      
        self.E = 32      
        self.H = 4        
        self.R = 4      
        self.M = 3       
        self.out_caps = 4  

    def TestMultiHeadAttention(self):
        try:
            x = torch.randn(self.B, self.S, self.E)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool)
            kpm[:, -3:] = True  

            attn = MultiHeadAttention(embedDim=self.E, numHeads=self.H, lowRank=True, rank=self.R)
            y1 = attn(x, x, x, keyPaddingMask=kpm, tdError=None)
            ok1 = (y1.shape == (self.B, self.S, self.E))

            y2 = attn(x, x, x, keyPaddingMask=None, tdError=torch.tensor(0.5))
            y3 = attn(x, x, x, keyPaddingMask=None, tdError=torch.randn(self.B))
            ok2 = (y2.shape == (self.B, self.S, self.E)) and (y3.shape == (self.B, self.S, self.E))

            attn.LowrankToFullrank(residual=True)
            y4 = attn(x, x, x, keyPaddingMask=kpm, tdError=None)
            attn.FullrankToLowrank(residual=True)
            y5 = attn(x, x, x, keyPaddingMask=kpm, tdError=None)
            ok3 = (y4.shape == (self.B, self.S, self.E)) and (y5.shape == (self.B, self.S, self.E))

            if ok1 and ok2 and ok3:
                print("MultiHeadAttention test passed.")
                return True
            else:
                print(f"MultiHeadAttention shape mismatch: y1={y1.shape}, y2={y2.shape}, y3={y3.shape}, y4={y4.shape}, y5={y5.shape}")
                return False
        except Exception as e:
            print("MultiHeadAttention test failed with exception:", e)
            return False

    def TestTemporalAttention(self):
        try:
            x = torch.randn(self.B, self.S, self.E)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool)
            ta = TemporalAttention(self.E, self.H, layerIdx=0, useHebbian=True)
            y = ta(x, keyPaddingMask=kpm, tdError=torch.randn(self.B))
            if y.shape == (self.B, self.S, self.E):
                print("TemporalAttention test passed.")
                return True
            else:
                print(f"TemporalAttention output shape mismatch: {y.shape}")
                return False
        except Exception as e:
            print("TemporalAttention test failed with exception:", e)
            return False

    def TestDynamicRouting(self):
        try:
            x = torch.randn(self.B, self.S, self.E)   
            mask = torch.zeros(self.B, self.S, dtype=torch.bool)
            mask[:, -2:] = True

            router = DynamicRouting(inCaps=self.S, inDim=self.E, outCaps=self.out_caps, outDim=self.E, iterations=3)
            y = router(x, mask)
            if y.shape == (self.B, self.out_caps, self.E):
                print("DynamicRouting test passed.")
                return True
            else:
                print(f"DynamicRouting output shape mismatch: {y.shape}")
                return False
        except Exception as e:
            print("DynamicRouting test failed with exception:", e)
            return False

    def TestHebbianFusion(self):
        try:
            fusion = HebbianFusion(numModes=self.M, embedDim=self.E, hebbianRate=0.01, useHebbian=True)
            inputs = torch.randn(self.B, self.M, self.E)
            y = fusion(inputs)
            if y.shape == (self.B, self.E):
                print("HebbianFusion test passed.")
                return True
            else:
                print(f"HebbianFusion output shape mismatch: {y.shape}")
                return False
        except AttributeError as e:
            print("HebbianFusion test failed (可能调用了不存在的 _effective_weights):", e)
            return False
        except Exception as e:
            print("HebbianFusion test failed with exception:", e)
            return False

    def TestMetaStrategySelector(self):
        try:
            selector = MetaStrategySelector(embedDim=self.E, numStrategies=self.M, numHeads=self.H, useMeta=True)
            x = torch.randn(self.B, self.E)
            probs = selector(x, mask=None)  # (B,H,M)
            ok_shape = (probs.shape == (self.B, self.H, self.M))
            ok_sum = torch.allclose(probs.sum(dim=-1), torch.ones(self.B, self.H), atol=1e-4)
            if ok_shape and ok_sum:
                print("MetaStrategySelector test passed.")
                return True
            else:
                print(f"MetaStrategySelector output invalid: shape={probs.shape}, sum={probs.sum(dim=-1)}")
                return False
        except Exception as e:
            print("MetaStrategySelector test failed with exception:", e)
            return False

    def TestAttentionExtractor(self):
        try:
            model = AttentionExtractor(
                embedDim=self.E,
                sequenceLength=self.S,  
                numHeads=self.H,
                temporalLayers=2,
                routingIterations=3,
                hebbianRate=0.01,
                useHebbian=True,
                useMetaLearning=True,
                gradientClipVal=0.5,)
            
            x = torch.randn(self.B, self.S, self.E)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool)
            kpm[:, -2:] = True
            td = torch.randn(self.B)

            y = model(x, keyPaddingMask=kpm, tdError=td)
            ok1 = (y.shape == (self.B, self.E))

            try:
                loss = y.mean()
                loss.backward()
                model.ClipGrads()
                ok2 = True
            except Exception as be:
                print("AttentionExtractor backward failed:", be)
                ok2 = False

            try:
                model.ResetFastWeights()
                _ = model(x, keyPaddingMask=kpm, tdError=None)
                model.AttenLowrankToFullrank(residual=True)
                _ = model(x, keyPaddingMask=kpm, tdError=None)
                model.AttenFullrankToLowrank(residual=True)
                _ = model(x, keyPaddingMask=kpm, tdError=None)
                ok3 = True
            except Exception as se:
                print("AttentionExtractor switch/reset failed:", se)
                ok3 = False

            if ok1 and ok2 and ok3:
                print("AttentionExtractor test passed. Output shape:", y.shape)
                return True
            else:
                if not ok1:
                    print(f"AttentionExtractor output shape mismatch: {y.shape}")
                return False
        except Exception as e:
            print("AttentionExtractor test failed with exception:", e)
            return False


    def RunAll(self):
        results = {
            "MultiHeadAttention": self.TestMultiHeadAttention(),
            "TemporalAttention": self.TestTemporalAttention(),
            "DynamicRouting": self.TestDynamicRouting(),
            "HebbianFusion": self.TestHebbianFusion(),
            "MetaStrategySelector": self.TestMetaStrategySelector(),
            "AttentionExtractor": self.TestAttentionExtractor(),}
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\nSummary: {passed}/{total} tests passed.")
        for k, v in results.items():
            print(f" - {k}: {'OK' if v else 'FAIL'}")
        return passed == total