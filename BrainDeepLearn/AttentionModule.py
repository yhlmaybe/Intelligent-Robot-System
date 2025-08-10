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
    """
    def __init__(self, embedDim: int, numHeads: int, hebbianRate: float = 0.01, 
                 attnDropout: float = 0.1, tdScale: float = 5.0, lowRank: bool = True, 
                 rank: int = 8, hebbPeriod: int = 4, updateHebbian: bool = True):
        super().__init__()
        assert embedDim % numHeads == 0, "AttentionModule embed_dim must be divisible by num_heads"
        self.embed_dim = embedDim
        self.num_heads = numHeads
        self.head_dim = embedDim // numHeads
        self.base_hebbian_rate = hebbianRate
        self.attn_dropout_p = attnDropout
        self.td_scale = tdScale
        self.low_rank = lowRank
        self.rank = rank
        self.hebb_period = max(1, hebbPeriod) 
        self.update_hebbian_flag = updateHebbian

        self.register_buffer("hebb_step", torch.tensor(0, dtype = torch.long), persistent=False)
            
        self.q_proj = nn.Linear(embedDim, embedDim)
        self.k_proj = nn.Linear(embedDim, embedDim)
        self.v_proj = nn.Linear(embedDim, embedDim)
        self.out_proj = nn.Linear(embedDim, embedDim)

        # Initialize Hebbian weights
        eye = torch.eye(self.head_dim)
        hebb_shape = (numHeads, self.head_dim, self.head_dim)
        self.register_buffer("hebbian_weights", eye.unsqueeze(0).repeat(hebb_shape[0], 1, 1), persistent=False)
        
        if lowRank:
            self.U = nn.Parameter(torch.randn(numHeads, self.head_dim, rank) * 0.02)
            self.V = nn.Parameter(torch.randn(numHeads, self.head_dim, rank) * 0.02)

        self.ResetParameters()

    def ResetParameters(self) -> None:
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_normal_(mod.weight, gain=1/math.sqrt(2))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        self.ResetHebbianMemory()

    def ResetHebbianMemory(self):
        device = self.hebbian_weights.device
        eye = torch.eye(self.head_dim, device=device)
        self.hebbian_weights.copy_(eye.unsqueeze(0).repeat(self.num_heads, 1, 1))
        
        if self.low_rank:
            nn.init.normal_(self.U, std=0.02)
            nn.init.normal_(self.V, std=0.02)

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
        
        # Detach tdError to prevent gradient flow into this module
        tdError = tdError.detach()
        
        # Handle scalar tdError
        if tdError.dim() == 0:
            tdError = tdError.view(1)
        
        # Ensure batch size matches
        if tdError.size(0) != B:
            if tdError.size(0) == 1:
                tdError = tdError.expand(B)
            else:
                raise ValueError(f"tdError size {tdError.size(0)} does not match batch size {B}")
        
        # Stable normalization
        td_mean = tdError.mean()
        td_std = tdError.std().clamp_min(1e-8)
        td_norm = (tdError - td_mean) / td_std
        
        neuromod = 1.0 + 0.5 * torch.tanh(td_norm / self.td_scale)
        return neuromod.view(B, 1, 1, 1)

    @torch.no_grad()
    def UpdateHebbianWeights(self, v: torch.Tensor, q: torch.Tensor, alpha: float):
        if not self.update_hebbian_flag or alpha <= 0:
            return
        
        # Detach inputs to prevent gradient flow
        v = v.detach()
        q = q.detach()
        
        # Normalization factor
        norm_factor = v.size(0) * v.size(2)  # batch_size * sequence_length
        hebb = torch.einsum("bhse,bhsd->hde", v, q) / (norm_factor + 1e-8)
        
        if self.low_rank:
            # Low-rank Hebbian update
            U_grad = torch.einsum('hde,her->hdr', hebb, self.V)
            V_grad = torch.einsum('hde,hdr->her', hebb, self.U)
            
            self.U.add_(alpha * (U_grad - self.U))
            self.V.add_(alpha * (V_grad - self.V))
        else:
            # Full-rank Hebbian update
            new_weights = (1 - alpha) * self.hebbian_weights + alpha * torch.tanh(hebb)
            self.hebbian_weights.copy_(new_weights)

    def ProJ(self, layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        out: torch.Tensor = layer(x)
        return out.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor, # (B, L, E)
        key: torch.Tensor, # (B, L, E)
        value: torch.Tensor, # (B, L, E)
        keyPaddingMask: Optional[torch.Tensor] = None, # (B, L)
        tdError: Optional[torch.Tensor] = None,) -> torch.Tensor: # (B,) or scalar

        B, L, _ = query.shape

        neuromod = self.ComputeNeuromodulation(tdError, B)

        q: torch.Tensor = self.ProJ(self.q_proj, query) # (B,H,L,D)
        k: torch.Tensor = self.ProJ(self.k_proj, key) # (B,H,L,D)
        v: torch.Tensor = self.ProJ(self.v_proj, value) # (B,H,L,D)

        if self.base_hebbian_rate > 0:
            self.hebb_step.add_(1)
            step_int = int(self.hebb_step.item())
            if (step_int % self.hebb_period) == 0:
                alpha = self.base_hebbian_rate * neuromod.mean()
                self.UpdateHebbianWeights(v, q, alpha)
        
        q = q * neuromod

        if self.low_rank:
            vU = torch.einsum("bhse,her->bhsr", v, self.U)
            v_fast: torch.Tensor = torch.einsum("bhsr,hdr->bhsd", vU, self.V)
        else:
            v_fast: torch.Tensor = torch.einsum("bhse,hde->bhsd", v, self.hebbian_weights)

        # Create attention mask
        attn_mask = None
        if keyPaddingMask is not None:
            attn_mask = keyPaddingMask[:, None, None, :]
        
        # Use PyTorch's efficient attention
        context = F.scaled_dot_product_attention(
            q, k, v_fast, 
            attn_mask=attn_mask, 
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False)

        out: torch.Tensor = context.transpose(1, 2).reshape(B, L, self.embed_dim)
        return self.out_proj(out)


class TemporalAttention(nn.Module):
    def __init__(self, embedDim: int, numHeads: int, layerIdx: int = 0):
        super().__init__()
        td_scale = 5.0 / (layerIdx + 1)
        self.attn: MultiHeadAttention = MultiHeadAttention(embedDim, numHeads, tdScale=td_scale)
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
    def __init__(self, numModes: int, embedDim: int, hebbianRate: float = 0.01):
        super().__init__()
        self.num_modes = numModes
        self.embed_dim = embedDim
        self.hebbian_rate = hebbianRate

        self.register_buffer("weights", torch.empty(numModes, embedDim, embedDim))
        self.register_buffer("hebbian_memory", torch.zeros_like(self.weights))
        self.register_buffer("momentum", torch.tensor(0.9, dtype=torch.float32))

        self.ResetParameters()

        self.gate = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Linear(embedDim * 2, numModes),
            nn.Softmax(dim=-1),)

    def ResetParameters(self):
        eye = torch.eye(self.embed_dim, device=self.weights.device).unsqueeze(0).repeat(self.num_modes, 1, 1)
        self.weights.copy_(eye + 0.05 * torch.randn_like(eye))

    def ResetHebbianMemory(self):
        self.hebbian_memory.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B,M,E)
        B, M, E = inputs.shape
        weighted = torch.einsum("bme,mef->bmf", inputs, self.weights) # (B,M,E)
        context = inputs.mean(dim=1) # (B,E)
        gate_w = self.gate(context) # (B,M)
        fused = torch.einsum("bmf,bm->bf", weighted, gate_w) # (B,E)

        if self.training:
            with torch.no_grad():
                # Stable Hebbian calculation
                norm_factor = max(1, B) * math.sqrt(E)
                hebb_term = torch.einsum("bme,bf->mef", inputs, fused) / (norm_factor + 1e-8)
                self.hebbian_memory.mul_(1 - self.hebbian_rate).add_(self.hebbian_rate * hebb_term)
                
                m = float(self.momentum)
                new_weights = m * self.weights + (1 - m) * self.hebbian_memory
                # Apply weight constraints
                new_weights = torch.clamp(new_weights, -3.0, 3.0)
                # Small weight decay
                decay_factor = 0.999
                self.weights.copy_(decay_factor * new_weights)

        return fused


class MetaStrategySelector(nn.Module):
    def __init__(self, embedDim: int, numStrategies: int, numHeads: int = 4):
        super().__init__()
        self.embed_dim = embedDim
        self.num_strategies = numStrategies
        self.num_heads = numHeads

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

    def ResetHebbianMemory(self):
        nn.init.normal_(self.strategy_weights, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B,E)
        offsets = self.generator(x).view(-1, self.num_heads, self.num_strategies)  # (B,H,S)
        params = self.strategy_weights + 0.2 * offsets
        
        if self.training:
            with torch.no_grad():
                w = F.softmax(params, dim=-1)
                batch_mean = (params * w).sum(dim=0) / w.sum(dim=0)
                self.strategy_weights.mul_(0.95).add_(0.05 * batch_mean)
        
        return F.softmax(params, dim=-1)  # (B,H,S)


class AttentionExtractor(nn.Module):
    def __init__(self,
                 embedDim: int = 512,
                 numHeads: int = 8,
                 temporalLayers: int = 3,
                 routingIterations: int = 3,
                 hebbianRate: float = 0.01,
                 useMetaLearning: bool = True,
                 gradientClipVal: float = 1.0,
                 checkPoint: bool = True):
        super().__init__()

        assert embedDim % 16 == 0, "AttentionModule embed_dim must be divisible by 16 (for 16 capsules)"

        self.use_meta_learning = useMetaLearning
        self.gradient_clip_val = gradientClipVal
        self.use_check_point = checkPoint
        self.output_dim = embedDim

        self.temporal_blocks: nn.ModuleList = nn.ModuleList([
            TemporalAttention(embedDim, numHeads, idx)
            for idx in range(temporalLayers)])

        in_dim = embedDim // 16
        self.routing = DynamicRouting(16, in_dim, 4, embedDim, iterations=routingIterations)

        self.fusion = HebbianFusion(numModes=3, embedDim=embedDim, hebbianRate=hebbianRate)

        if useMetaLearning:
            self.meta_selector = MetaStrategySelector(embedDim, numStrategies=3)
            self.context_proj = nn.Sequential(nn.Linear(embedDim * 2, embedDim), nn.GELU())
        else:
            self.register_parameter("meta_selector", None)

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
        if S % 16 != 0:
            pad_len = 16 - (S % 16)
            x = F.pad(x, (0, 0, 0, pad_len))
            if keyPaddingMask is not None:
                keyPaddingMask = F.pad(keyPaddingMask, (0, pad_len), value=True)
            S = x.size(1)

        h = x
        
        # Use detached version of tdError for checkpointing
        #td_error_detached = tdError.detach() if tdError is not None else None

        for blk in self.temporal_blocks:
            if self.use_check_point and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, keyPaddingMask, tdError)
            else:
                h = blk(h, keyPaddingMask, tdError)

        # Create capsule mask
        caps_mask: Optional[torch.Tensor] = None
        if keyPaddingMask is not None:
            # Ensure mask is properly padded
            padded_mask = keyPaddingMask
            if padded_mask.size(1) % 16 != 0:
                pad_len = 16 - (padded_mask.size(1) % 16)
                padded_mask = F.pad(padded_mask, (0, pad_len), value=True)
            
            caps_mask = padded_mask.reshape(B, 16, -1).any(dim=2)  # (B,16)

        # Extract capsules
        caps = h.view(B, 16, S//16, -1).mean(dim=2)  # [B,16,E]
        routed = self.routing(caps, caps_mask)  # (B,4,E)

        # Fusion of different representations
        routed_mean = routed.mean(dim=1) # (B,E)
        temp_mean = h.mean(dim=1) # (B,E)
        
        fusion_in = torch.stack([
            temp_mean, 
            routed_mean, 
            temp_mean + routed_mean
        ], dim=1)  # (B,3,E)
        
        fused = self.fusion(fusion_in)  # (B,E)

        if self.use_meta_learning:
            context = self.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))
            strat_w = self.meta_selector(context)  # (B,H=4,S=3)
            
            feats = torch.stack([
                temp_mean, 
                routed_mean, 
                fused.detach()  # Detach fused to prevent double backprop
            ], dim=1)  # (B,3,E)
            
            mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)  # (B,H,E)
            out = mixed_per_head.mean(dim=1)
        else:
            out = fused

        return self.output_proj(out)

    def ResetHebbianMemory(self) -> None:
        """Reset all Hebbian weights in the module"""
        for blk in self.temporal_blocks:
            blk.attn.ResetHebbianMemory()
        self.fusion.ResetHebbianMemory()
        if self.use_meta_learning:
            self.meta_selector.ResetHebbianMemory()