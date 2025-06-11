from __future__ import annotations
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, embedDim: int, numHeads: int, hebbianRate: float = 0.01, attnDropout: float = 0.1):
        super().__init__()
        assert embedDim % numHeads == 0, "AttentionModule embed_dim must be divisible by num_heads"
        self.embed_dim = embedDim
        self.num_heads = numHeads
        self.head_dim = embedDim // numHeads
        self.hebbian_rate = hebbianRate
        self.attn_dropout = attnDropout

        self.q_proj = nn.Linear(embedDim, embedDim)
        self.k_proj = nn.Linear(embedDim, embedDim)
        self.v_proj = nn.Linear(embedDim, embedDim)
        self.out_proj = nn.Linear(embedDim, embedDim)

        self.register_buffer("hebbian_weights", torch.zeros(numHeads, self.head_dim, self.head_dim))
        self.ResetParameters()

    def ResetParameters(self):
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        self.ResetHebbianMemory()

    def ResetHebbianMemory(self):
        eye = torch.eye(self.head_dim, device=self.hebbian_weights.device)
        self.hebbian_weights.copy_(eye.unsqueeze(0).repeat(self.num_heads, 1, 1))

    def ScaledDotProductAttention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor], p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_mask = mask

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p, is_causal=False)
            return out, None
        else:
            d = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
            if mask is not None:
                scores = scores.masked_fill(mask, float("-1e4"))
            weights = F.softmax(scores, dim=-1)
            weights = F.dropout(weights, p=p, training=self.training)
            out = torch.matmul(weights, v)
            return out, weights

    def forward(self,query: torch.Tensor,key: torch.Tensor,value: torch.Tensor, keyPaddingMask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, _ = query.shape

        def _proj(layer, x):
            return layer(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = _proj(self.q_proj, query)
        k = _proj(self.k_proj, key)
        v = _proj(self.v_proj, value)

        mask = keyPaddingMask.view(B, 1, 1, S) if keyPaddingMask is not None else None

        attn_out, attn_weights = self.ScaledDotProductAttention(q, k, v, mask, self.attn_dropout)

        if self.training:
            with torch.no_grad():
                hebb_term = torch.einsum("bhse,bhsd->hde", v, q) / self.head_dim
                hebb_term.clamp_(-1, 1)
                self.hebbian_weights.mul_(1 - self.hebbian_rate).add_(self.hebbian_rate * hebb_term)

        v_mod = torch.einsum("bhse,hde->bhsd", v, self.hebbian_weights)

        if attn_weights is None:
            attn_out = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                attn_out = attn_out.masked_fill(mask, float("-1e4"))
            attn_weights = F.softmax(attn_out, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        out = torch.matmul(attn_weights, v_mod)           # [B, H, S, D]
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)

class TemporalAttention(nn.Module):
    def __init__(self, embedDim: int, numHeads: int):
        super().__init__()
        self.attn = MultiHeadAttention(embedDim, numHeads)
        self.norm = nn.LayerNorm(embedDim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, keyPaddingMask: Optional[torch.Tensor] = None):
        return self.norm(x + self.dropout(self.attn(x, x, x, keyPaddingMask)))


class DynamicRouting(nn.Module):

    def __init__(self, inCaps: int, inDim: int, outCaps: int, outDim: int, iterations: int = 3):
        super().__init__()
        self.I = inCaps
        self.O = outCaps
        self.in_dim = inDim
        self.out_dim = outDim
        self.iterations = iterations

        self.transformation = nn.Parameter(torch.empty(inCaps, outCaps, inDim, outDim))
        self.routing_logits = nn.Parameter(torch.zeros(1, inCaps, outCaps))  
        self.ResetParameters()

    def ResetParameters(self):
        nn.init.kaiming_uniform_(self.transformation, a=math.sqrt(5))
        nn.init.normal_(self.routing_logits, std=0.01)

    @staticmethod
    def Squash(vectors: torch.Tensor) -> torch.Tensor:
        squared_norm = vectors.pow(2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1.0 + squared_norm) / (torch.sqrt(squared_norm + 1e-8))
        return scale * vectors

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B, I, in_dim]
        B, I, D = x.shape
        assert I == self.I and D == self.in_dim, "AttentionModule capsule input dim mismatch"

        u_hat = torch.einsum("bid,iocd->bioc", x, self.transformation)

        logits = self.routing_logits.expand(B, -1, -1)  # [B, I, O]
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(-1), -1e4)

        for r in range(self.iterations):
            weights = F.softmax(logits, dim=-1)  # [B, I, O]
            if mask is not None:
                weights = weights.masked_fill(mask.unsqueeze(-1), 0.0)

            s = torch.einsum("bioc,bio->boc", u_hat, weights)  # [B, O, out_dim]
            v = self.Squash(s)

            if r < self.iterations - 1:
                agreement = torch.einsum("bioc,boc->bio", u_hat, v)
                logits = logits + agreement
                if mask is not None:
                    logits = logits.masked_fill(mask.unsqueeze(-1), -1e4)
        return v  # [B, O, out_dim]


class HebbianFusion(nn.Module):

    def __init__(self, numModes: int, embedDim: int, hebbianRate: float = 0.01):
        super().__init__()
        self.num_modes = numModes
        self.embed_dim = embedDim
        self.hebbian_rate = hebbianRate

        self.register_buffer("weights", torch.empty(numModes, embedDim, embedDim))
        self.register_buffer("hebbian_memory", torch.zeros_like(self.weights))
        self.ResetParameters()

        self.gate = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Linear(embedDim * 2, numModes),
            nn.Softmax(dim=-1),
        )

    def ResetParameters(self):
        eye = torch.eye(self.embed_dim, device=self.weights.device).unsqueeze(0).repeat(self.num_modes, 1, 1)
        self.weights.copy_(eye + 0.05 * torch.randn_like(eye))

    def ResetHebbianMemory(self):
        self.hebbian_memory.zero_()

    def forward(self, inputs: torch.Tensor):
        # inputs: [B, M, E]
        B, M, E = inputs.shape
        weighted = torch.einsum("bme,mef->bmf", inputs, self.weights)  # [B, M, E]
        context = inputs.mean(dim=1)
        gate_w = self.gate(context)  # [B, M]
        fused = torch.einsum("bmf,bm->bf", weighted, gate_w)

        if self.training:
            with torch.no_grad():
                mode_act = inputs.mean(dim=0)          # [M, E]
                fused_act = fused.mean(dim=0)          # [E]
                hebb_term = torch.einsum("me,f->mef", mode_act, fused_act) / (E**0.5)
                self.hebbian_memory.mul_(1 - self.hebbian_rate).add_(self.hebbian_rate * hebb_term)
                self.weights.copy_(torch.clamp(self.weights * 0.95 + 0.05 * self.hebbian_memory, -3, 3))
        return fused  # [B, E]

class MetaStrategySelector(nn.Module):

    def __init__(self, embedDim: int, numStrategies: int, numHeads: int = 4):
        super().__init__()
        self.embed_dim = embedDim
        self.num_strategies = numStrategies
        self.num_heads = numHeads

        self.generator = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Linear(embedDim * 2, numHeads * numStrategies),
        )
        self.register_buffer("strategy_weights", torch.zeros(numHeads, numStrategies))
        self.ResetParameters()

    def ResetParameters(self):
        nn.init.uniform_(self.strategy_weights, -0.05, 0.05)
        for m in self.generator:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="gelu")
                nn.init.zeros_(m.bias)

    def ResetHebbianMemory(self):
        nn.init.normal_(self.strategy_weights, std = 0.02)

    def forward(self, x: torch.Tensor):  # x: [B, E]
        offsets = self.generator(x).view(-1, self.num_heads, self.num_strategies)
        params = self.strategy_weights + 0.2 * offsets
        if self.training:
            with torch.no_grad():
                w = F.softmax(params, dim=-1)
                batch_mean = (params * w).sum(dim=0) / w.sum(dim=0)
                self.strategy_weights.mul_(0.95).add_(0.05 * batch_mean)
        return F.softmax(params, dim=-1)  # [B, H, S]


class AttentionExtractor(nn.Module):
    def __init__(self,embedDim: int = 512,numHeads: int = 8,temporalLayers: int = 3,routingIterations: int = 3,hebbianRate: float = 0.01,useMetaLearning: bool = True,):
        super().__init__()
        assert embedDim % 16 == 0, "AttentionModule embed_dim must be divisible by 16 (for 16 capsules)"
        self.embed_dim = embedDim
        self.use_meta_learning = useMetaLearning

        self.temporal_blocks : List[TemporalAttention] = nn.ModuleList(
            [TemporalAttention(embedDim, numHeads) for _ in range(temporalLayers)]
        )

        in_dim = embedDim // 16
        self.routing = DynamicRouting(16, in_dim, 4, embedDim, iterations=routingIterations)

        self.fusion = HebbianFusion(numModes=3, embedDim=embedDim, hebbianRate=hebbianRate)

        if useMetaLearning:
            self.meta_selector = MetaStrategySelector(embedDim, numStrategies=3)
            self.context_proj = nn.Sequential(
                nn.Linear(embedDim * 2, embedDim),
                nn.GELU(),
            )
        else:
            self.register_parameter("meta_selector", None)

        self.output_proj = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedDim * 2, embedDim),
            nn.LayerNorm(embedDim),
        )

    def forward(self, x: torch.Tensor, keyPaddingMask: Optional[torch.Tensor] = None):
        # x: [B, S, E]
        B, S, E = x.shape

        h = x
        for block in self.temporal_blocks:
            h = block(h, keyPaddingMask)

        caps = rearrange(h, "b s (c d) -> b c s d", c=16)
        caps = caps.mean(dim=2)  
        caps_mask = None
        if keyPaddingMask is not None:
            caps_mask = keyPaddingMask.float().unsqueeze(1)          # [B, 1, S]
            caps_mask = F.adaptive_avg_pool1d(caps_mask, 16).squeeze(1) > 0.5  # [B, 16]

        routed = self.routing(caps, caps_mask)  # [B, 4, E]
        routed_mean = routed.mean(dim=1)        # [B, E]
        temp_mean = h.mean(dim=1)               # [B, E]

        fusion_in = torch.stack([temp_mean, routed_mean, temp_mean + routed_mean], dim=1)  # [B, 3, E]
        fused = self.fusion(fusion_in)  # [B, E]

        if self.use_meta_learning:
            context = self.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))  # [B, E]
            strat_w = self.meta_selector(context)  # [B, H, S(=3)]
            feats = torch.stack([temp_mean, routed_mean, fused.detach()], dim=1)  # [B, 3, E]
            mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)  # [B, H, E]
            out = mixed_per_head.mean(dim=1)
        else:
            out = fused

        return self.output_proj(out)  # [B, E]
    
    def ResetHebbianMemory(self):
        for blk in self.temporal_blocks:
            blk.attn.ResetHebbianMemory()
        self.fusion.ResetHebbianMemory()
        if self.use_meta_learning and self.meta_selector is not None:
            self.meta_selector.ResetHebbianMemory()

