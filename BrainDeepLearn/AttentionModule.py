from __future__ import annotations
import math
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
    Parameters
    ----------
    embedDim : int
        Token embedding dimension `E`.
    numHeads : int
        Number of attention heads `H`.
    hebbianRate : float, optional
        Base Hebbian learning-rate α₀.  (default 0.01)
    attnDropout : float, optional
        Drop probability applied to attention weights.  (default 0.1)
    tdScale : float, optional
        Scale factor dividing TD-error before tanh.  (default 5.0)
    """
    def __init__(self, embedDim: int, numHeads: int, hebbianRate: float = 0.01, attnDropout: float = 0.1, tdScale : float = 5.0):
        super().__init__()
        assert embedDim % numHeads == 0, "AttentionModule embed_dim must be divisible by num_heads"
        self.embed_dim = embedDim
        self.num_heads = numHeads
        self.head_dim = embedDim // numHeads
        self.base_hebbian_rate = hebbianRate
        self.attn_dropout_p = attnDropout
        self.td_scale = tdScale

        self.q_proj = nn.Linear(embedDim, embedDim)
        self.k_proj = nn.Linear(embedDim, embedDim)
        self.v_proj = nn.Linear(embedDim, embedDim)
        self.out_proj = nn.Linear(embedDim, embedDim)

        eye : torch.Tensor = torch.eye(self.head_dim)

        self.register_buffer("hebbian_weights", eye.unsqueeze(0).repeat(numHeads, 1, 1))
        self.ResetParameters()

    def ResetParameters(self) -> None:
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        self.ResetHebbianMemory()

    def ResetHebbianMemory(self):
        eye: torch.Tensor = torch.eye(self.head_dim, device=self.hebbian_weights.device)
        self.hebbian_weights.copy_(eye.unsqueeze(0).repeat(self.num_heads, 1, 1))

    def ScaledDotAttn(
        self,
        q: torch.Tensor,               # (B, H, Lq, D)
        k: torch.Tensor,               # (B, H, Lk, D)
        v: torch.Tensor,               # (B, H, Lk, D)
        mask: Optional[torch.Tensor],  # (B, 1, 1, Lk)
        dropoutP: float,) -> Tuple[torch.Tensor, torch.Tensor]:

        d = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-1e4"))
        weights = F.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=dropoutP, training=q.requires_grad)
        context = torch.matmul(weights, v)
        return context, weights

    def forward(
        self,
        query: torch.Tensor,                              # (B, Sq, E)
        key: torch.Tensor,                                # (B, Sk, E)
        value: torch.Tensor,                              # (B, Sk, E)
        keyPaddingMask: Optional[torch.Tensor] = None,    # (B, Sk)
        tdError: Optional[torch.Tensor] = None,           # (B,) or scalar
        ) -> torch.Tensor:
        B, Sq, _ = query.shape
        Sk: int = key.shape[1]

        neuromod: float = 1.0
        if tdError is not None:
            td_norm: torch.Tensor = (tdError - tdError.mean()) / (tdError.std() + 1e-8)
            neuromod = 1.0 + 0.5 * torch.tanh(td_norm.mean() / self.td_scale).item()

        def _proj(layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
            B_, L_, _ = x.shape
            return (layer(x).view(B_, L_, self.num_heads, self.head_dim).transpose(1, 2))  # (B,H,L,D)

        q: torch.Tensor = _proj(self.q_proj, query)   # (B,H,Sq,D)
        k: torch.Tensor = _proj(self.k_proj, key)     # (B,H,Sk,D)
        v: torch.Tensor = _proj(self.v_proj, value)   # (B,H,Sk,D)

        mask: Optional[torch.Tensor] = None
        if keyPaddingMask is not None:
            mask = keyPaddingMask.bool().view(B, 1, 1, Sk)

        if self.training:
            alpha: float = self.base_hebbian_rate * neuromod
            with torch.no_grad():
                hebb_term: torch.Tensor = torch.einsum(
                    "bhse,bhsd->hde", v, q
                ) / (B * Sq * self.head_dim)          # outer product avg
                hebb_term = torch.tanh(hebb_term)     # bound to [-1,1]
                self.hebbian_weights.mul_(1 - alpha).add_(alpha * hebb_term)
                # simple clamp for stability
                self.hebbian_weights.clamp_(-3.0, 3.0)

        v_fast: torch.Tensor = torch.einsum("bhse,hde->bhsd", v, self.hebbian_weights)

        q_mod: torch.Tensor = q * neuromod

        if hasattr(F, "scaled_dot_product_attention"):
            context: torch.Tensor = F.scaled_dot_product_attention(
                q_mod,
                k,
                v_fast,
                attn_mask=mask,
                dropout_p=self.attn_dropout_p,
                is_causal=False)
        else:
            context, _ = self.ScaledDotAttn(q_mod, k, v_fast, mask, self.attn_dropout_p)

        out: torch.Tensor = (context.transpose(1, 2).contiguous().view(B, Sq, self.embed_dim))

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
        x: torch.Tensor,                               # (B,S,E)
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

    def forward(
        self,
        x: torch.Tensor,                              # (B,I,D)
        mask: Optional[torch.Tensor] = None,          # (B,I) bool
        ) -> torch.Tensor:                            # (B,O,out_dim)
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
        self.register_buffer("momentum", torch.tensor(0.9, dtype=torch.float32))

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
                hebb_term: torch.Tensor = torch.einsum("bme,bf->bmef", inputs, fused).mean(0) / math.sqrt(E)            # (M,E,E)
                self.hebbian_memory.mul_(1 - self.hebbian_rate).add_(self.hebbian_rate * hebb_term)

                m = float(self.momentum)
                self.weights.copy_(torch.clamp(m * self.weights + (1 - m) * self.hebbian_memory, -3.0, 3.0))

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
        nn.init.normal_(self.strategy_weights, std = 0.01)

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

        self.E = embedDim
        self.use_meta_learning = useMetaLearning

        self.temporal_blocks : List[TemporalAttention] = nn.ModuleList([
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

    def forward(
        self,
        x: torch.Tensor,                                # (B,S,E)
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,        # (B,) or scalar
        ) -> torch.Tensor:

        B, S, E = x.shape
        h = x
        for blk in self.temporal_blocks:
            h = blk(h, keyPaddingMask, tdError)

        caps_mask: Optional[torch.Tensor] = None

        if keyPaddingMask is not None:          
            group_size = math.ceil(S / 16)
            pad_len = group_size * 16 - S
            if pad_len:
                h = F.pad(h, (0, 0, 0, pad_len))  
                keyPaddingMask = F.pad(keyPaddingMask, (0, pad_len), value=False)  

        caps = rearrange(h, "b s (c d) -> b c s d", c=16)
        caps = caps.mean(dim=2)          # (B,16,in_dim)

        caps_mask = keyPaddingMask.view(B, 16, group_size).any(dim=2)  # (B,16)

        routed = self.routing(caps, caps_mask)            # (B,4,E)

        routed_mean = routed.mean(dim=1)                  # (B,E)
        temp_mean = h.mean(dim=1)                         # (B,E)

        fusion_in = torch.stack([temp_mean, routed_mean, temp_mean + routed_mean], dim=1)  
                                                                     # (B,3,E)
        fused = self.fusion(fusion_in)                    # (B,E)

        if self.use_meta_learning:
            context = self.context_proj(torch.cat([temp_mean, routed_mean], dim=-1))

            strat_w = self.meta_selector(context)         # (B,H=4,S=3)
            feats = torch.stack([temp_mean, routed_mean, fused.detach()], dim=1)     # (B,3,E)
            mixed_per_head = torch.einsum("bhs,bse->bhe", strat_w, feats)  # (B,H,E)
            out = mixed_per_head.mean(dim=1)
        else:
            out = fused

        return self.output_proj(out)

    def ResetHebbianMemory(self) -> None:
        for blk in self.temporal_blocks:
            blk.attn.ResetHebbianMemory()
        self.fusion.ResetHebbianMemory()
        if self.use_meta_learning:
            self.meta_selector.ResetHebbianMemory()

