from __future__ import annotations 
from typing import Optional, Dict, Tuple, NamedTuple
from FunctionTools import AGICoreModule, RoPEMultiheadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


BankInput = Dict[str, torch.Tensor]

class FiLMBlock(AGICoreModule):
    def __init__(self, inDim: int, outDim: int):
        super().__init__()
        self.linear = nn.Linear(inDim, outDim)
        self.norm = nn.LayerNorm(outDim)

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        if (gamma is not None) or (beta is not None):
            if gamma is None:
                gamma = torch.zeros_like(h)
            if beta is None:
                beta = torch.zeros_like(h)
            h = h * (1.0 + gamma) + beta
        h = F.gelu(h)
        return h


class ConsciousnessHyperNet(AGICoreModule):
    def __init__(
        self,
        ctxDim: int,
        nSelfBlocks: int,
        nIntentBlocks: int,
        selfHiddenDim: int,
        intentHiddenDim: int,
        hiddenDim: int = 512,):
        super().__init__()
        self.ctx_dim = int(ctxDim)
        self.n_self_blocks = int(nSelfBlocks)
        self.n_intent_blocks = int(nIntentBlocks)
        self.self_hidden_dim = int(selfHiddenDim)
        self.intent_hidden_dim = int(intentHiddenDim)

        self.total_self_params = self.n_self_blocks * 2 * self.self_hidden_dim
        self.total_intent_params = self.n_intent_blocks * 2 * self.intent_hidden_dim

        out_dim = self.total_self_params + self.total_intent_params

        self.mlp = nn.Sequential(
            nn.Linear(self.ctx_dim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, out_dim),)

    def forward(self, ctxTensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        ctx = ctxTensor # [B, ctxDim]
        B, D = ctx.shape
        h = self.mlp(ctx) # [B, out_dim] 

        cur = 0
        hs = self.total_self_params
        hi = self.total_intent_params

        self_flat = h[:, cur:cur + hs]; cur += hs # [B, hs]
        intent_flat = h[:, cur:cur + hi]; cur += hi # [B, hi]

        self_flat = self_flat.view(B, self.n_self_blocks, 2, self.self_hidden_dim) # [B, nSelfBlocks, 2, selfHiddenDim]

        gamma_self = self_flat[:, :, 0, :] # [B, nSelfBlocks, selfHiddenDim]
        beta_self = self_flat[:, :, 1, :] # [B, nSelfBlocks, selfHiddenDim]

        intent_flat = intent_flat.view(B, self.n_intent_blocks, 2, self.intent_hidden_dim)

        gamma_intent = intent_flat[:, :, 0, :] # [B, nIntentBlocks, intentHiddenDim]
        beta_intent = intent_flat[:, :, 1, :] # [B, nIntentBlocks, intentHiddenDim]

        return {
            "gamma_self": gamma_self,
            "beta_self": beta_self,
            "gamma_intent": gamma_intent,
            "beta_intent": beta_intent,}



class ConsciousHebbianLinear(AGICoreModule):
    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,):
        super().__init__()
        self.in_f = int(inFeatures)
        self.out_f = int(outFeatures)

        self.weight = nn.Parameter(torch.randn(self.out_f, self.in_f) * 0.02) # [O,I]
        self.bias = nn.Parameter(torch.zeros(self.out_f)) # [O]

        self.register_buffer(
            "hebb",
            torch.zeros(1, self.out_f, self.in_f),
            persistent=False)

    @torch.no_grad()
    def EnsureB(self, B: int):
        if self.hebb.size(0) != B :
            self.hebb = self.hebb.new_zeros(B, self.out_f, self.in_f) # [B,O,I]

    @torch.no_grad()
    def ResetHebbianMemory(
        self,
        doneMask: Optional[torch.Tensor] = None,
        ) -> None:
        if doneMask is None:
            self.hebb.zero_()
            return
        mask = doneMask.view(-1)
        if mask.numel() != self.hebb.size(0):
            raise ValueError("Consciousness Hebbian reset mask must match its batch size")
        self.hebb[mask] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, I = x.shape

        W_eff = self.weight.unsqueeze(0) + self.hebb # [B,O,I]
        y_b = torch.einsum("bi,boi->bo", x, W_eff) + self.bias.unsqueeze(0) # [B,O]

        with torch.no_grad():
            x_n = F.normalize(x, dim=-1, eps=1e-6) # [B,I]
            y_n = F.normalize(y_b, dim=-1, eps=1e-6) # [B,O]

            d_hebb = y_n.unsqueeze(-1) * x_n.unsqueeze(1) # [B, O, I]

            self.hebb.mul_(0.95)
            self.hebb.add_(0.05 * d_hebb)

        return y_b # [B,O]


class NeuroGeoControlFusion(AGICoreModule):
    def __init__(
        self,
        devDim: int,
        selfDim: int,
        intentDim: int,
        *,
        nCharts: int = 4,
        hiddenMul: int = 2,
        dampMin: float = 0.05,
        dampMax: float = 0.35,):
        super().__init__()
        self.dev_dim = int(devDim)
        self.self_dim = int(selfDim)
        self.intent_dim = int(intentDim)
        self.n_charts = max(2, int(nCharts))
        self.damp_min = float(dampMin)
        self.damp_max = float(dampMax)

        hidden = max(self.dev_dim, int(hiddenMul) * self.dev_dim)

        self.dev_path = nn.Sequential(
            nn.LayerNorm(self.dev_dim),
            nn.Linear(self.dev_dim, self.dev_dim),
            nn.GELU(),)
        
        self.self_path = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, self.dev_dim),
            nn.GELU(),)
        
        self.intent_path = nn.Sequential(
            nn.LayerNorm(self.intent_dim),
            nn.Linear(self.intent_dim, self.dev_dim),
            nn.GELU(),)

        self.cross_ds = nn.Sequential(
            nn.Linear(2 * self.dev_dim, self.dev_dim),
            nn.LayerNorm(self.dev_dim),
            nn.GELU(),)
        
        self.cross_di = nn.Sequential(
            nn.Linear(2 * self.dev_dim, self.dev_dim),
            nn.LayerNorm(self.dev_dim),
            nn.GELU(),)
        
        self.cross_si = nn.Sequential(
            nn.Linear(2 * self.dev_dim, self.dev_dim),
            nn.LayerNorm(self.dev_dim),
            nn.GELU(),)

        self.feat_dim = 6 * self.dev_dim
        self.fuse_norm = nn.LayerNorm(self.feat_dim)
        self.fuse_proj = nn.Sequential(
            nn.Linear(self.feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.feat_dim),
            nn.GELU(),)

        self.chart_gate = nn.Linear(self.feat_dim, self.n_charts)
        self.chart_heads = nn.ModuleList([
            nn.Sequential(
                ConsciousHebbianLinear(self.feat_dim, self.dev_dim),
                nn.LayerNorm(self.dev_dim),
                nn.Tanh(),) for _ in range(self.n_charts)])

        self.base_head = nn.Sequential(
            ConsciousHebbianLinear(self.feat_dim, self.dev_dim),
            nn.LayerNorm(self.dev_dim),
            nn.Tanh(),)
        self.mix_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, self.dev_dim),
            nn.Sigmoid(),)
        self.damp_head = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, 1),
            nn.Sigmoid(),)

    def forward(
        self,
        devPrev: torch.Tensor,
        selfPrev: torch.Tensor,
        intentPrev: torch.Tensor,) -> torch.Tensor:
        h_d = self.dev_path(devPrev) # [B, Dd]
        h_s = self.self_path(selfPrev) # [B, Dd]
        h_i = self.intent_path(intentPrev) # [B, Dd]

        c_ds = self.cross_ds(torch.cat([h_d, h_s], dim=-1)) # [B, Dd]
        c_di = self.cross_di(torch.cat([h_d, h_i], dim=-1)) # [B, Dd]
        c_si = self.cross_si(torch.cat([h_s, h_i], dim=-1)) # [B, Dd]

        feat = torch.cat([h_d, h_s, h_i, c_ds, c_di, c_si], dim=-1)
        feat = self.fuse_norm(feat + self.fuse_proj(feat)) # [B, F]

        gate = F.softmax(self.chart_gate(feat), dim=-1) # [B, Nc]
        chart_delta = torch.stack([head(feat) for head in self.chart_heads], dim=1) # [B, Nc, Dd]
        delta_chart = (gate.unsqueeze(-1) * chart_delta).sum(dim=1) # [B, Dd]

        delta_base = self.base_head(feat) # [B, Dd]
        mix = self.mix_head(feat)
        damp = self.damp_min + (self.damp_max - self.damp_min) * self.damp_head(feat)

        delta_raw = mix * delta_chart + (1.0 - mix) * delta_base
        delta = torch.tanh((1.0 - damp) * delta_raw - damp * h_d)
        return delta # [B, Dd]


class ConsciousnessOutput(NamedTuple):
    self_sem: torch.Tensor 
    intent_sem: torch.Tensor  
    extras: Dict[str, torch.Tensor]

class ConsciousnessExtractor(AGICoreModule):
    def __init__(
        self,
        memItemDim: int = 1024,  
        worldItemDim: int = 512, 
        symItemDim: int = 256,
        hiddenDim: int = 512,
        devDim: int = 512, 
        selfDim: int = 1024, 
        intentDim: int = 1024, 
        nSelfBlocks: int = 4,
        nIntentBlocks: int = 4,
        hyperHiddenDim: int = 1024,
        topKMem: int = 128,
        randKMem: int = 64,
        topKWorld: int = 128,
        randKWorld: int = 64,):
        super().__init__()

        self.mem_item_dim = memItemDim
        self.world_item_dim = worldItemDim
        self.sym_item_dim = symItemDim
        self.dev_dim = devDim
        self.self_dim = selfDim
        self.intent_dim = intentDim
        self.n_self_blocks = nSelfBlocks
        self.n_intent_blocks = nIntentBlocks
        self.top_k_mem = topKMem
        self.rand_k_mem = randKMem
        self.top_k_world = topKWorld
        self.rand_k_world = randKWorld

        self.mem_score_net = nn.Sequential(
            nn.Linear(self.mem_item_dim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, 1),)
        
        self.world_score_net = nn.Sequential(
            nn.Linear(self.world_item_dim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, 1),)

        self.mem_agg_proj = nn.Sequential(
            nn.Linear(3 * self.mem_item_dim, self.mem_item_dim),
            nn.LayerNorm(self.mem_item_dim),
            nn.GELU(),)
        
        self.world_agg_proj = nn.Sequential(
            nn.Linear(3 * self.world_item_dim, self.world_item_dim),
            nn.LayerNorm(self.world_item_dim),
            nn.GELU(),)

        ctx_in_dim = self.mem_item_dim + self.world_item_dim + self.dev_dim

        self.ctx_norm = nn.LayerNorm(ctx_in_dim)

        self.ctx_proj = nn.Sequential(
            nn.Linear(ctx_in_dim, hyperHiddenDim),
            nn.LayerNorm(hyperHiddenDim),
            nn.GELU(),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)

        self.ctx_world_proj = nn.Sequential(
            nn.LayerNorm(self.world_item_dim),
            nn.Linear(self.world_item_dim, hyperHiddenDim),
            nn.GELU(),)
        
        self.ctx_mem_proj = nn.Sequential(
            nn.LayerNorm(self.mem_item_dim),
            nn.Linear(self.mem_item_dim, hyperHiddenDim),
            nn.GELU(),)
        
        self.ctx_dev_proj = nn.Sequential(
            nn.LayerNorm(self.dev_dim),
            nn.Linear(self.dev_dim, hyperHiddenDim),
            nn.GELU(),)

        self.ctx_world_logvar = nn.Sequential(
            nn.LayerNorm(self.world_item_dim),
            nn.Linear(self.world_item_dim, 1),)
        
        self.ctx_mem_logvar = nn.Sequential(
            nn.LayerNorm(self.mem_item_dim),
            nn.Linear(self.mem_item_dim, 1),)
        
        self.ctx_dev_logvar = nn.Sequential(
            nn.LayerNorm(self.dev_dim),
            nn.Linear(self.dev_dim, 1),)

        self.ctx_obs_post = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)
        
        self.ctx_prior = nn.Sequential(
            nn.Linear(self.self_dim + self.intent_dim + self.dev_dim, hyperHiddenDim),
            nn.LayerNorm(hyperHiddenDim),
            nn.GELU(),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),)
        
        self.ctx_kalman_gain = nn.Sequential(
            nn.LayerNorm(2 * hyperHiddenDim),
            nn.Linear(2 * hyperHiddenDim, hyperHiddenDim),
            nn.Sigmoid(),)
        
        self.ctx_post_norm = nn.LayerNorm(hyperHiddenDim)

        self.ctx_query_proj = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),)

        def pick_heads(embed_dim: int) -> int:
            for h in (8, 4, 2, 1):
                if (embed_dim % h) == 0:
                    return h
            return 1

        self.ctx_attn_heads = pick_heads(hyperHiddenDim)

        self.mem_token_proj = nn.Sequential(
            nn.LayerNorm(self.mem_item_dim),
            nn.Linear(self.mem_item_dim, hyperHiddenDim),)
        
        self.world_token_proj = nn.Sequential(
            nn.LayerNorm(self.world_item_dim),
            nn.Linear(self.world_item_dim, hyperHiddenDim),)
        
        self.mem_cross_attn = RoPEMultiheadAttention(
            embedDim=hyperHiddenDim,
            numHeads=self.ctx_attn_heads,)
        
        self.world_cross_attn = RoPEMultiheadAttention(
            embedDim=hyperHiddenDim,
            numHeads=self.ctx_attn_heads,)
        
        self.mem_obs_fuse = nn.Sequential(
            nn.LayerNorm(2 * hyperHiddenDim),
            nn.Linear(2 * hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)
        
        self.world_obs_fuse = nn.Sequential(
            nn.LayerNorm(2 * hyperHiddenDim),
            nn.Linear(2 * hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)

        self.z_world_head = nn.Linear(hyperHiddenDim, 2 * hyperHiddenDim)
        self.z_mem_head = nn.Linear(hyperHiddenDim, 2 * hyperHiddenDim)
        self.z_dev_head = nn.Linear(hyperHiddenDim, 2 * hyperHiddenDim)
        self.z_prior_head = nn.Linear(hyperHiddenDim, 2 * hyperHiddenDim)

        self.z_to_ctx = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)
        
        self.z_next_pred = nn.Sequential(
            nn.Linear(self.self_dim + self.intent_dim + self.dev_dim, hyperHiddenDim),
            nn.LayerNorm(hyperHiddenDim),
            nn.GELU(),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),)

        self.z_to_world = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)
        
        self.z_to_mem = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)
        
        self.z_to_dev = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, hyperHiddenDim),
            nn.GELU(),)

        self.nce_temp = 0.1
        self.nce_dim = max(64, hyperHiddenDim // 2)

        self.nce_proj_z = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, self.nce_dim),)
        
        self.nce_proj_world = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, self.nce_dim),)
        
        self.nce_proj_mem = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, self.nce_dim),)
        
        self.nce_proj_dev = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, self.nce_dim),)

        self.hyper = ConsciousnessHyperNet(
            ctxDim=hyperHiddenDim,
            nSelfBlocks=self.n_self_blocks,
            nIntentBlocks=self.n_intent_blocks,
            selfHiddenDim=self.self_dim,
            intentHiddenDim=self.intent_dim,
            hiddenDim=hyperHiddenDim * 2,)

        in_self = self.world_item_dim + self.mem_item_dim + self.dev_dim
        self.self_in = nn.Sequential(
            ConsciousHebbianLinear(in_self, self.self_dim),
            nn.LayerNorm(self.self_dim),
            nn.GELU(),)
        
        self.self_blocks = nn.ModuleList([FiLMBlock(self.self_dim, self.self_dim) for _ in range(self.n_self_blocks)])

        in_intent = self.self_dim + self.mem_item_dim + self.dev_dim
        self.intent_in = nn.Sequential(
            ConsciousHebbianLinear(in_intent, self.intent_dim),
            nn.LayerNorm(self.intent_dim),
            nn.GELU(),)
        
        self.intent_blocks = nn.ModuleList([FiLMBlock(self.intent_dim, self.intent_dim) for _ in range(self.n_intent_blocks)])

        self.ctx_to_self = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, self.self_dim),)
        
        self.ctx_to_self_gate = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, 1),
            nn.Sigmoid(),)
        
        self.ctx_to_intent = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, self.intent_dim),)
        
        self.ctx_to_intent_gate = nn.Sequential(
            nn.LayerNorm(hyperHiddenDim),
            nn.Linear(hyperHiddenDim, 1),
            nn.Sigmoid(),)

        self.dev_update = NeuroGeoControlFusion(
            devDim=self.dev_dim,
            selfDim=self.self_dim,
            intentDim=self.intent_dim,
            nCharts=4,
            hiddenMul=2,
            dampMin=0.05,
            dampMax=0.35,)
        
        self.mem_query_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, self.mem_item_dim),)

        self.world_query_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, self.world_item_dim),)

        self.mem_focus_to_self = nn.Sequential(
            nn.LayerNorm(self.mem_item_dim),
            nn.Linear(self.mem_item_dim, self.self_dim),)

        self.world_focus_to_self = nn.Sequential(
            nn.LayerNorm(self.world_item_dim),
            nn.Linear(self.world_item_dim, self.self_dim),)
        
        self.arousal_net = nn.Sequential(
            nn.LayerNorm(self.dev_dim),
            nn.Linear(self.dev_dim, 1),
            nn.Sigmoid(),)
        self.update_base = 0.05

        self.mem_gain_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, 1),
            nn.Sigmoid(),)
        
        self.world_gain_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, 1),
            nn.Sigmoid(),)
        
        self.gain_min = 0.05
        self.gain_max = 0.35

        self.dev_to_ctx = nn.Sequential(
            nn.LayerNorm(self.dev_dim),
            nn.Linear(self.dev_dim, hyperHiddenDim),)

        self.register_buffer("_dev_trace", torch.zeros(1, self.dev_dim), persistent=True)
        self.register_buffer("_last_self_intent", torch.zeros(1, self.intent_dim), persistent=True)
        self.register_buffer("_last_sem", torch.zeros(1, self.self_dim), persistent=True)
        self.register_buffer("_state_valid", torch.zeros(1, dtype=torch.bool), persistent=True)
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=True)

    @torch.no_grad()
    def EnsureB(self, B: int):
        if self._dev_trace.size(0) != B:
            self._dev_trace = self._dev_trace.new_zeros(B, self.dev_dim)
            self._last_self_intent = self._last_self_intent.new_zeros(
                B, self.intent_dim)
            self._last_sem = self._last_sem.new_zeros(B, self.self_dim)
            self._state_valid = self._state_valid.new_zeros(B)
            self._step = self._step.new_zeros(1)
        for module in self.modules():
            if isinstance(module, ConsciousHebbianLinear):
                module.EnsureB(B)

    @torch.no_grad()
    def ResetState(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self._dev_trace.zero_()
            self._last_self_intent.zero_()
            self._last_sem.zero_()
            self._state_valid.zero_()
            self._step.zero_()
            return
        done = doneMask.view(-1).to(
            device=self._dev_trace.device,
            dtype=torch.bool)
        if done.numel() != self._dev_trace.size(0):
            raise ValueError("Consciousness reset mask must match its batch size")
        self._dev_trace[done] = 0
        self._last_self_intent[done] = 0
        self._last_sem[done] = 0
        self._state_valid[done] = False
        if bool(done.all().item()):
            self._step.zero_()

    def GetState(self):
        step = int(self._step.item())
        if step <= 0:
            return None, None, None, 0
        return self._dev_trace.detach(), self._last_self_intent.detach(), self._last_sem.detach(), step

    def QueryTopK(
        self,
        bank: torch.Tensor, # [B, N, D]
        query: torch.Tensor, # [B, D]
        topK: int,
        validMask: Optional[torch.Tensor] = None,):

        B, N, D = bank.shape
        device = self.device

        if N == 0:
            focus = bank.new_zeros(B, D)
            top_idx = torch.zeros(B, 0, dtype=torch.long, device=device)
            top_w = torch.zeros(B, 0, device=device)
            return focus, top_idx, top_w

        q = F.normalize(query, dim=-1).unsqueeze(1) 
        x = F.normalize(bank, dim=-1)
        sim = (q * x).sum(-1) 
        valid = (
            torch.ones(B, N, device=device, dtype=torch.bool)
            if validMask is None
            else validMask)
        if valid.dtype != torch.bool:
            raise TypeError("query bank validity must use torch.bool")
        if valid.device != bank.device:
            raise ValueError("query bank validity must be on the bank device")
        if valid.shape != (B, N):
            raise ValueError(
                f"query bank validity must have shape {(B, N)}, got {tuple(valid.shape)}")
        sim = sim.masked_fill(~valid, torch.finfo(sim.dtype).min)

        k_top = min(topK, N)
        if k_top <= 0:
            focus = bank.new_zeros(B, D)
            top_idx = torch.zeros(B, 0, dtype=torch.long, device=device)
            top_w = torch.zeros(B, 0, device=device)
            return focus, top_idx, top_w

        top_scores, top_idx = torch.topk(sim, k=k_top, dim=1) # [B, k_top]
        top_valid = torch.gather(valid, 1, top_idx)
        top_w = F.softmax(top_scores, dim=-1) * top_valid.to(sim.dtype)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        top_items = torch.gather(bank, 1,top_idx.unsqueeze(-1).expand(-1, -1, D) )

        focus = torch.einsum('bk,bkd->bd', top_w, top_items) 

        return focus, top_idx, top_w # focus: [B, D], top_w: [B, k_top], top_idx: [B, k_top]


    def AggregateBank(
        self,
        bankTensor: torch.Tensor, # [B, N, D]
        scoreNet: nn.Module,
        topK: int,
        randK: int,
        validMask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        bank = bankTensor
        score_net = scoreNet
        top_k = int(topK)
        rand_k = int(randK)

        B, N, D = bank.shape
        device = self.device

        if N == 0:
            summary = torch.zeros(B, 3 * D, device=device, dtype=bank.dtype)
            stats = {
                "score_mean": torch.zeros(B, 1, device=device, dtype=bank.dtype),
                "n_items": torch.zeros(B, 1, device=device, dtype=bank.dtype),}
            
            return summary, stats

        valid = (
            torch.ones(B, N, device=device, dtype=torch.bool)
            if validMask is None
            else validMask)
        if valid.dtype != torch.bool:
            raise TypeError("bank validity must use torch.bool")
        if valid.device != bank.device:
            raise ValueError("bank validity must be on the bank device")
        if valid.shape != (B, N):
            raise ValueError(
                f"bank validity must have shape {(B, N)}, got {tuple(valid.shape)}")

        scores = score_net(bank).squeeze(-1) # [B, N]
        valid_f = valid.to(bank.dtype)
        n_valid = valid_f.sum(dim=1, keepdim=True)
        global_mean = (
            (bank * valid_f.unsqueeze(-1)).sum(dim=1)
            / n_valid.clamp_min(1.0)) # [B, D]

        k_top = min(top_k, N)
        if k_top > 0:
            masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            top_scores, top_idx = torch.topk(masked_scores, k=k_top, dim=1) # [B, k_top]

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D) 
            top_items = torch.gather(bank, 1, idx_exp) # [B, k_top, D]
            top_valid = torch.gather(valid, 1, top_idx)
            top_w = F.softmax(top_scores, dim=-1) * top_valid.to(scores.dtype)
            top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            top_mean = torch.einsum("bk,bkd->bd", top_w, top_items) # [B,D]
        else:
            top_mean = torch.zeros(B, D, device=device, dtype=bank.dtype)
            top_idx = None

        k_rand = min(rand_k, N)
        if k_rand > 0:
            avail_mask = valid.clone()
            if k_top > 0:
                avail_mask.scatter_(1, top_idx, False)

            rand_noise = torch.rand(B, N, device=device)
            rand_scores = rand_noise.masked_fill(~avail_mask, -1.0) # [B, N]

            _, rand_idx = torch.topk(rand_scores, k=k_rand, dim=1) # [B, k_rand]
            idx_exp2 = rand_idx.unsqueeze(-1).expand(-1, -1, D)
            rand_items = torch.gather(bank, 1, idx_exp2) # [B, k_rand]
            rand_valid = torch.gather(avail_mask, 1, rand_idx).to(bank.dtype)
            rand_mean = (
                (rand_items * rand_valid.unsqueeze(-1)).sum(dim=1)
                / rand_valid.sum(dim=1, keepdim=True).clamp_min(1.0)) # [B, D]
        else:
            rand_mean = torch.zeros(B, D, device=device, dtype=bank.dtype)

        summary = torch.cat([global_mean, top_mean, rand_mean], dim=-1)

        stats = {
            "score_mean": (
                (scores * valid_f).sum(dim=1, keepdim=True)
                / n_valid.clamp_min(1.0)),
            "n_items": n_valid,}

        return summary, stats # summary: [B, 3D]

    def NormalizeBankInput(
        self,
        bank: BankInput,
        *,
        expectedB: Optional[int],
        expectedDim: int,
        bankRole: str,
        device: torch.device,
        dtype: torch.dtype,) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(bank, dict):
            raise TypeError(f"{bankRole} bank must be a dict")
        if set(bank.keys()) != {"tokens", "valid"}:
            raise ValueError(
                f"{bankRole} bank must contain exactly 'tokens' and 'valid'")

        tokens = bank["tokens"]
        valid = bank["valid"]
        if not isinstance(tokens, torch.Tensor):
            raise TypeError(f"{bankRole}.tokens must be a tensor")
        if tokens.dim() != 3:
            raise ValueError(f"{bankRole}.tokens must have shape [B, N, D]")
        if expectedB is not None and int(tokens.size(0)) != expectedB:
            raise ValueError(
                f"{bankRole}.tokens batch size must be {expectedB}, "
                f"got {int(tokens.size(0))}")
        if int(tokens.size(2)) != expectedDim:
            raise ValueError(
                f"{bankRole}.tokens feature size must be {expectedDim}, "
                f"got {int(tokens.size(2))}")
        if tokens.device != device:
            raise ValueError(f"{bankRole}.tokens must be on the model device")
        if tokens.dtype != dtype:
            raise TypeError(f"{bankRole}.tokens must use the model dtype")
        if not isinstance(valid, torch.Tensor):
            raise TypeError(f"{bankRole}.valid must be a tensor")
        if valid.dtype != torch.bool:
            raise TypeError(f"{bankRole}.valid must use torch.bool")
        if valid.device != device:
            raise ValueError(f"{bankRole}.valid must be on the model device")
        if valid.shape != tokens.shape[:2]:
            raise ValueError(
                f"{bankRole}.valid must have shape {tuple(tokens.shape[:2])}, "
                f"got {tuple(valid.shape)}")
        return tokens.contiguous(), valid.contiguous()

    def forward(
        self,
        memoryBank: BankInput,
        worldBank: BankInput,) -> ConsciousnessOutput:

        # B：batch size
        # Dm = self.mem_item_dim
        # Dw = self.world_item_dim
        # Dd = self.dev_dim
        # Ds = self.self_dim
        # Di = self.intent_dim
        # H = hyperHiddenDim
        # Nm = memory_bank.size(1) memory token
        # Nw = world_bank.size(1) world token 

        device = self.device
        dtype = self.dtype

        memory_bank, memory_valid = self.NormalizeBankInput(
            memoryBank,
            expectedB=None,
            expectedDim=self.mem_item_dim,
            bankRole="mem",
            device=device,
            dtype=dtype)
        B = int(memory_bank.size(0))
        world_bank, world_valid = self.NormalizeBankInput(
            worldBank,
            expectedB=B,
            expectedDim=self.world_item_dim,
            bankRole="world",
            device=device,
            dtype=dtype)
        self.EnsureB(B)

        mem_available = memory_valid.any(dim=1, keepdim=True)
        world_available = world_valid.any(dim=1, keepdim=True)
        current_available = mem_available | world_available
        dev_available = self._state_valid.view(B, 1).clone() & current_available

        if not bool(current_available.any().item()):
            zeros_1 = torch.zeros(B, 1, device=device, dtype=dtype)
            extras: Dict[str, torch.Tensor] = {
                "loss": (torch.zeros((), device=device, dtype=dtype) if self.training else None),
                "dev_trace_norm": self._dev_trace.norm(dim=-1, keepdim=True).detach(),
                "mem_score_mean": zeros_1,
                "world_score_mean": zeros_1,
                "mem_n_items": zeros_1,
                "world_n_items": zeros_1,
                "mem_focus_norm": zeros_1,
                "world_focus_norm": zeros_1,
                "mem_available": zeros_1,
                "world_available": zeros_1,
                "dev_available": dev_available.to(dtype),
                "cold_start": (~self._state_valid).view(B, 1).to(dtype),
                "no_observation": torch.ones(B, 1, device=device, dtype=dtype),}
            return ConsciousnessOutput(
                self_sem=self._last_sem,
                intent_sem=self._last_self_intent,
                extras=extras,)

        inactive_rows = ~current_available.view(B)
        hebbian_snapshots = []
        if bool(inactive_rows.any().item()):
            for module in self.modules():
                if isinstance(module, ConsciousHebbianLinear):
                    hebbian_snapshots.append((module, module.hebb.detach().clone()))

        arousal = self.arousal_net(self._dev_trace)
        alpha = self.update_base * (0.5 + arousal)
        delta = self.dev_update(self._dev_trace, self._last_sem, self._last_self_intent)
        dev_candidate = self._dev_trace + alpha * delta
        dev_ctx = torch.where(
            dev_available,
            dev_candidate,
            torch.zeros_like(dev_candidate)) # [B, Dd]

        mem_summary_raw, mem_stats = self.AggregateBank(
            memory_bank,
            self.mem_score_net,
            topK=self.top_k_mem,
            randK=self.rand_k_mem,
            validMask=memory_valid,)
        mem_ctx = self.mem_agg_proj(mem_summary_raw)
        mem_ctx = mem_ctx * mem_available.to(dtype)

        world_summary_raw, world_stats = self.AggregateBank(
            world_bank,
            self.world_score_net,
            topK=self.top_k_world,
            randK=self.rand_k_world,
            validMask=world_valid,) # world_summary_raw" [B, 3*Dw]
        world_ctx = self.world_agg_proj(world_summary_raw)
        world_ctx = world_ctx * world_available.to(dtype)

        ctx_raw = torch.cat([world_ctx, mem_ctx, dev_ctx], dim=-1)
        ctx_norm = self.ctx_norm(ctx_raw)
        ctx_base = self.ctx_proj(ctx_norm) # [B, H]

        prior_seed = torch.cat([self._last_sem, self._last_self_intent, self._dev_trace], dim=-1) # [B, Ds+Di+Dd]
        prior_feat = self.ctx_prior(prior_seed) # [B,H]
        mu_prior, logvar_prior = self.z_prior_head(prior_feat).chunk(2, dim=-1) # [B,H]
        logvar_prior = logvar_prior.clamp(-8.0, 8.0)
        prec_prior = torch.exp(-logvar_prior) # [B,H]

        q_seed = torch.where(dev_available, prior_feat, ctx_base)
        q = self.ctx_query_proj(q_seed).unsqueeze(1)

        if memory_bank.size(1) > 0:
            mem_tokens = self.mem_token_proj(memory_bank)
            safe_memory_valid = memory_valid.clone()
            empty_memory_rows = ~mem_available.view(B)
            if bool(empty_memory_rows.any().item()):
                mem_tokens = mem_tokens.clone()
                mem_tokens[empty_memory_rows, 0] = 0.0
                safe_memory_valid[empty_memory_rows, 0] = True
            mem_attn_out, mem_attn_prob = self.mem_cross_attn(
                q,
                mem_tokens,
                mem_tokens,
                keyPaddingMask=~safe_memory_valid,
                needWeights=True)
            mem_attn_vec = mem_attn_out.squeeze(1) * mem_available.to(dtype) # [B,H]
            mem_attn_prob = mem_attn_prob.squeeze(1) # [B,Nm]
            mem_attn_prob = mem_attn_prob * memory_valid.to(dtype)
            mem_attn_entropy = -(mem_attn_prob * mem_attn_prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True) # [B,1]
        else:
            mem_attn_vec = torch.zeros(B, ctx_base.size(-1), device=device, dtype=dtype)
            mem_attn_prob = torch.zeros(B, 0, device=device, dtype=dtype)
            mem_attn_entropy = torch.zeros(B, 1, device=device, dtype=dtype)

        world_use_attn = world_bank.size(1) > 1
        if world_use_attn:
            world_tokens = self.world_token_proj(world_bank)
            safe_world_valid = world_valid.clone()
            empty_world_rows = ~world_available.view(B)
            if bool(empty_world_rows.any().item()):
                world_tokens = world_tokens.clone()
                world_tokens[empty_world_rows, 0] = 0.0
                safe_world_valid[empty_world_rows, 0] = True
            world_attn_out, world_attn_prob = self.world_cross_attn(
                q,
                world_tokens,
                world_tokens,
                keyPaddingMask=~safe_world_valid,
                needWeights=True)
            world_attn_vec = world_attn_out.squeeze(1) * world_available.to(dtype)
            world_attn_prob = world_attn_prob.squeeze(1) # [B,Nw]
            world_attn_prob = world_attn_prob * world_valid.to(dtype)
            world_attn_entropy = -(world_attn_prob * world_attn_prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True)
        else:
            world_attn_vec = torch.zeros(B, ctx_base.size(-1), device=device, dtype=dtype)
            world_attn_prob = torch.zeros(B, 0, device=device, dtype=dtype)
            world_attn_entropy = torch.zeros(B, 1, device=device, dtype=dtype)

        world_obs = self.ctx_world_proj(world_ctx) # [B,H]
        mem_obs = self.ctx_mem_proj(mem_ctx) # [B,H]
        dev_obs = self.ctx_dev_proj(dev_ctx) # [B,H]

        logvar_world_scalar = self.ctx_world_logvar(world_ctx).clamp(-6.0, 6.0) # [B,1]
        logvar_mem_scalar = self.ctx_mem_logvar(mem_ctx).clamp(-6.0, 6.0) # [B,1]
        logvar_dev_scalar = self.ctx_dev_logvar(dev_ctx).clamp(-6.0, 6.0) # [B,1]

        if world_use_attn:
            world_feat = self.world_obs_fuse(torch.cat([world_obs, world_attn_vec], dim=-1))
        else:
            world_feat = world_obs

        mem_feat = self.mem_obs_fuse(torch.cat([mem_obs, mem_attn_vec], dim=-1))
        dev_feat = dev_obs
        world_feat = world_feat * world_available.to(dtype)
        mem_feat = mem_feat * mem_available.to(dtype)
        dev_feat = dev_feat * dev_available.to(dtype)

        mu_world, logvar_world_head = self.z_world_head(world_feat).chunk(2, dim=-1)
        mu_mem, logvar_mem_head = self.z_mem_head(mem_feat).chunk(2, dim=-1)
        mu_dev, logvar_dev_head = self.z_dev_head(dev_feat).chunk(2, dim=-1)

        logvar_world_raw = logvar_world_head + logvar_world_scalar
        logvar_mem_raw = logvar_mem_head + logvar_mem_scalar
        logvar_dev_raw = logvar_dev_head + logvar_dev_scalar
        logvar_world = logvar_world_raw.clamp(-8.0, 8.0)
        logvar_mem = logvar_mem_raw.clamp(-8.0, 8.0)
        logvar_dev = logvar_dev_raw.clamp(-8.0, 8.0)

        prec_world = torch.exp(-logvar_world) # [B,H]
        prec_mem = torch.exp(-logvar_mem) # [B,H]
        prec_dev = torch.exp(-logvar_dev) # [B,H]
        prec_world_eff = prec_world * world_available.to(dtype)
        prec_mem_eff = prec_mem * mem_available.to(dtype)
        prec_dev_eff = prec_dev * dev_available.to(dtype)

        prec_sum = prec_world_eff + prec_mem_eff + prec_dev_eff + prec_prior

        mu_post = (
            mu_world * prec_world_eff +
            mu_mem * prec_mem_eff +
            mu_dev * prec_dev_eff +
            mu_prior * prec_prior) / prec_sum # [B,H]
        logvar_post = torch.log(1.0 / prec_sum) # [B,H]

        if self.training:
            eps = torch.zeros_like(mu_post)
            active_rows = current_available.view(B)
            eps[active_rows] = torch.randn_like(mu_post[active_rows])
            z_post = mu_post + torch.exp(0.5 * logvar_post) * eps
        else:
            z_post = mu_post

        obs_prec_scalar = torch.cat([
            prec_world_eff.mean(dim=-1, keepdim=True),
            prec_mem_eff.mean(dim=-1, keepdim=True),
            prec_dev_eff.mean(dim=-1, keepdim=True),], dim=-1)
        obs_prec_norm = obs_prec_scalar / obs_prec_scalar.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        ctx_obs = self.ctx_obs_post(
            obs_prec_norm[:, 0:1] * world_feat +
            obs_prec_norm[:, 1:2] * mem_feat +
            obs_prec_norm[:, 2:3] * dev_feat +
            0.5 * ctx_base)

        z_ctx = self.z_to_ctx(z_post) # [B,H]
        kalman_gain = self.ctx_kalman_gain(torch.cat([z_ctx, prior_feat], dim=-1)) # [B,H]
        ctx_vec = self.ctx_post_norm(prior_feat + kalman_gain * (z_ctx - prior_feat) + 0.3 * ctx_obs) # [B,H]

        modal_prec_scalar = torch.stack(
            [prec_world_eff.mean(dim=-1), prec_mem_eff.mean(dim=-1), prec_dev_eff.mean(dim=-1)],
            dim=-1) # [B,3]
        modal_prec_norm = modal_prec_scalar / modal_prec_scalar.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        w_world = modal_prec_norm[:, 0:1]
        w_mem = modal_prec_norm[:, 1:2]
        w_dev = modal_prec_norm[:, 2:3]
        hyper_out = self.hyper(ctx_vec)

        gamma_self = hyper_out["gamma_self"]
        beta_self = hyper_out["beta_self"]
        gamma_intent = hyper_out["gamma_intent"]
        beta_intent = hyper_out["beta_intent"]

        world_ctx_eff = world_ctx * (0.5 + w_world) # [B,Dw]
        mem_ctx_eff = mem_ctx * (0.5 + w_mem) # [B,Dm]
        dev_ctx_eff = dev_ctx * (0.5 + w_dev) # [B,Dd]

        self_in_vec = torch.cat([world_ctx_eff, mem_ctx_eff, dev_ctx_eff], dim=-1)
        h_self = self.self_in(self_in_vec)
        ctx_self_gain = 0.5 + self.ctx_to_self_gate(ctx_vec)
        ctx_self_delta = self.ctx_to_self(ctx_vec)  # [B,Ds]
        h_self = h_self + ctx_self_gain * ctx_self_delta # [B,Ds]

        for i, block in enumerate(self.self_blocks):
            g = gamma_self[:, i, :]
            b = beta_self[:, i, :]
            h_self = block(h_self, gamma=g, beta=b)
        self_sem = h_self # [B,Ds]

        q_mem = self.mem_query_net(self_sem) # [B,Dm]
        q_world = self.world_query_net(self_sem) # [B,Dw]

        mem_focus, mem_idx, mem_w = self.QueryTopK(
            memory_bank,
            q_mem,
            self.top_k_mem,
            validMask=memory_valid)

        world_focus, world_idx, world_w = self.QueryTopK(
            world_bank,
            q_world,
            self.top_k_world,
            validMask=world_valid)

        mem_delta = self.mem_focus_to_self(mem_focus) # [B,Ds]
        world_delta = self.world_focus_to_self(world_focus) # [B,Ds]
        mem_delta = mem_delta * mem_available.to(dtype)
        world_delta = world_delta * world_available.to(dtype)

        g_m = self.mem_gain_net(self_sem) 
        g_w = self.world_gain_net(self_sem)

        g_m = self.gain_min + (self.gain_max - self.gain_min) * g_m
        g_w = self.gain_min + (self.gain_max - self.gain_min) * g_w
        g_m = g_m * (0.5 + w_mem)
        g_w = g_w * (0.5 + w_world)

        self_sem = self_sem + g_m * mem_delta + g_w * world_delta # [B,Ds]

        intent_in_vec = torch.cat([self_sem, mem_ctx_eff, dev_ctx_eff], dim=-1)
        h_intent = self.intent_in(intent_in_vec)
        ctx_intent_gain = 0.5 + self.ctx_to_intent_gate(ctx_vec)
        ctx_intent_delta = self.ctx_to_intent(ctx_vec)
        h_intent = h_intent + ctx_intent_gain * ctx_intent_delta # [B,Di]
        
        for i, block in enumerate(self.intent_blocks):
            g = gamma_intent[:, i, :]
            b = beta_intent[:, i, :]
            h_intent = block(h_intent, gamma=g, beta=b)

        if self.training:
            def symmetric_info_nce(
                a: torch.Tensor,
                b: torch.Tensor,
                tau: float,
                valid: torch.Tensor,) -> torch.Tensor:
                keep = valid.view(-1)
                if int(keep.sum().item()) <= 1:
                    return (a.sum() + b.sum()) * 0.0
                a_n = F.normalize(a[keep], dim=-1)
                b_n = F.normalize(b[keep], dim=-1)
                logits = (a_n @ b_n.transpose(0, 1)) / tau
                labels = torch.arange(a_n.size(0), device=a_n.device)
                loss_ab = F.cross_entropy(logits, labels)
                loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
                return 0.5 * (loss_ab + loss_ba)

            world_mask = world_available.to(dtype)
            mem_mask = mem_available.to(dtype)
            dev_mask = dev_available.to(dtype)
            current_mask = current_available.to(dtype)
            modality_count = (world_mask + mem_mask + dev_mask).clamp_min(1.0)

            def active_mean(value: torch.Tensor) -> torch.Tensor:
                return (value * current_mask).sum() / current_mask.sum().clamp_min(1.0)

            ctx_hat = self.dev_to_ctx(dev_ctx) 
            ctx_align_loss = (
                F.mse_loss(ctx_hat, ctx_vec.detach(), reduction="none").mean(dim=1, keepdim=True)
                * dev_mask)
            slow_loss = (
                (dev_ctx - self._dev_trace).pow(2).mean(dim=1, keepdim=True)
                * dev_mask)

            kl = 0.5 * (
                logvar_prior - logvar_post +
                (torch.exp(logvar_post) + (mu_post - mu_prior).pow(2)) / torch.exp(logvar_prior) -1.0)
            loss_kl = kl.mean(dim=-1, keepdim=True) * current_mask

            world_recon = self.z_to_world(z_post)
            mem_recon = self.z_to_mem(z_post)
            dev_recon = self.z_to_dev(z_post)
            loss_recon = (
                F.smooth_l1_loss(world_recon, world_feat.detach(), reduction="none").mean(dim=-1, keepdim=True) * world_mask +
                F.smooth_l1_loss(mem_recon, mem_feat.detach(), reduction="none").mean(dim=-1, keepdim=True) * mem_mask +
                F.smooth_l1_loss(dev_recon, dev_feat.detach(), reduction="none").mean(dim=-1, keepdim=True) * dev_mask
            ) / modality_count

            z_n = self.nce_proj_z(z_post)
            w_n = self.nce_proj_world(world_feat)
            m_n = self.nce_proj_mem(mem_feat)
            d_n = self.nce_proj_dev(dev_feat)
            nce_world = symmetric_info_nce(z_n, w_n, self.nce_temp, world_available)
            nce_mem = symmetric_info_nce(z_n, m_n, self.nce_temp, mem_available)
            nce_dev = symmetric_info_nce(z_n, d_n, self.nce_temp, dev_available)
            nce_weights = torch.stack([
                world_available.sum() > 1,
                mem_available.sum() > 1,
                dev_available.sum() > 1,
            ])
            loss_nce = (
                nce_world * nce_weights[0]
                + nce_mem * nce_weights[1]
                + nce_dev * nce_weights[2]
            ) / nce_weights.sum().clamp_min(1.0)

            z_pred_prev = self.z_next_pred(prior_seed)
            loss_trans = (
                F.smooth_l1_loss(z_pred_prev, mu_post.detach(), reduction="none").mean(dim=-1, keepdim=True)
                * dev_mask)

            precision_target = mu_post.detach()
            variance_floor = 5e-2

            def calibrated_precision_loss(
                mean: torch.Tensor,
                logvar: torch.Tensor,
                mask: torch.Tensor,) -> torch.Tensor:
                mean_target_loss = F.smooth_l1_loss(
                    mean,
                    precision_target,
                    reduction="none")
                target_logvar = torch.log(
                    (mean.detach() - precision_target).pow(2) + variance_floor
                ).clamp(-4.0, 4.0)
                variance_loss = F.smooth_l1_loss(
                    logvar,
                    target_logvar,
                    reduction="none")
                return (mean_target_loss + 0.25 * variance_loss).mean(
                    dim=-1,
                    keepdim=True) * mask

            loss_precision = (
                calibrated_precision_loss(mu_world, logvar_world, world_mask)
                + calibrated_precision_loss(mu_mem, logvar_mem, mem_mask)
                + calibrated_precision_loss(mu_dev, logvar_dev, dev_mask)
            ) / modality_count
            loss_var_range = (
                (F.relu(logvar_world_raw - 4.0) + F.relu(-4.0 - logvar_world_raw)).mean(dim=-1, keepdim=True) * world_mask +
                (F.relu(logvar_mem_raw - 4.0) + F.relu(-4.0 - logvar_mem_raw)).mean(dim=-1, keepdim=True) * mem_mask +
                (F.relu(logvar_dev_raw - 4.0) + F.relu(-4.0 - logvar_dev_raw)).mean(dim=-1, keepdim=True) * dev_mask
            ) / modality_count

            world_mem_mask = world_mask * mem_mask
            world_dev_mask = world_mask * dev_mask
            mem_dev_mask = mem_mask * dev_mask
            pair_count = (world_mem_mask + world_dev_mask + mem_dev_mask).clamp_min(1.0)
            align_loss = (
                (1.0 - F.cosine_similarity(world_feat, mem_feat, dim=-1)).unsqueeze(-1) * world_mem_mask +
                (1.0 - F.cosine_similarity(world_feat, dev_feat, dim=-1)).unsqueeze(-1) * world_dev_mask +
                (1.0 - F.cosine_similarity(mem_feat, dev_feat, dim=-1)).unsqueeze(-1) * mem_dev_mask
            ) / pair_count

            prior_smooth = (
                F.smooth_l1_loss(ctx_vec, prior_feat.detach(), reduction="none").mean(dim=1, keepdim=True)
                * dev_mask)
            
            loss_ctx_inject = (
                (ctx_self_gain * ctx_self_delta).pow(2).mean(dim=-1, keepdim=True) +
                (ctx_intent_gain * ctx_intent_delta).pow(2).mean(dim=-1, keepdim=True)
            ) * 0.5 * current_mask

            loss = (
                0.05 * active_mean(ctx_align_loss) +
                1e-3 * active_mean(slow_loss) +
                0.02 * active_mean(loss_kl) +
                0.04 * active_mean(loss_recon) +
                0.02 * loss_nce +
                0.02 * active_mean(loss_trans) +
                0.01 * active_mean(loss_precision) +
                0.003 * active_mean(align_loss) +
                0.01 * active_mean(prior_smooth) +
                0.005 * active_mean(loss_var_range) +
                0.003 * active_mean(loss_ctx_inject))
        else:
            loss_kl = None
            loss_recon = None
            loss_nce = None
            loss_trans = None
            loss_precision = None
            loss_var_range = None
            align_loss = None
            prior_smooth = None
            loss_ctx_inject = None
            loss = None

        current_mask_bool = current_available
        dev_state = torch.where(current_mask_bool, dev_ctx, self._dev_trace)
        self_sem_out = torch.where(current_mask_bool, self_sem, self._last_sem)
        intent_sem_out = torch.where(current_mask_bool, h_intent, self._last_self_intent)
        if hebbian_snapshots:
            with torch.no_grad():
                for module, snapshot in hebbian_snapshots:
                    module.hebb[inactive_rows] = snapshot[inactive_rows]

        self._dev_trace = dev_state.detach()
        self._last_self_intent = intent_sem_out.detach()
        self._last_sem = self_sem_out.detach()
        self._state_valid.logical_or_(current_available.view(B))
        self._step.add_(1)

        extras: Dict[str, torch.Tensor] = {
            "loss": loss,
            "dev_trace_norm": dev_state.norm(dim=-1, keepdim=True).detach(),
            "mem_score_mean": mem_stats["score_mean"].detach(),
            "world_score_mean": world_stats["score_mean"].detach(),
            "modal_precision_world": w_world.detach(),
            "modal_precision_mem": w_mem.detach(),
            "modal_precision_dev": w_dev.detach(),
            "mem_available": mem_available.to(dtype).detach(),
            "world_available": world_available.to(dtype).detach(),
            "dev_available": dev_available.to(dtype).detach(),
            "no_observation": (~current_available).to(dtype).detach(),
            "ctx_self_gate": ctx_self_gain.detach(),
            "ctx_intent_gate": ctx_intent_gain.detach(),
            "ctx_prior_gap": (ctx_vec - prior_feat).pow(2).mean(dim=-1, keepdim=True).detach(),
            "mem_attn_entropy": mem_attn_entropy.detach(),
            "world_attn_entropy": world_attn_entropy.detach(),
            "post_var_mean": torch.exp(logvar_post).mean(dim=-1, keepdim=True).detach(),
            "mem_n_items": mem_stats["n_items"].detach(),
            "world_n_items": world_stats["n_items"].detach(),
            "mem_focus_norm": mem_focus.norm(dim=-1, keepdim=True).detach(),
            "world_focus_norm": world_focus.norm(dim=-1, keepdim=True).detach(),}

        if self.training:
            extras["loss_kl"] = active_mean(loss_kl).detach()
            extras["loss_recon"] = active_mean(loss_recon).detach()
            extras["loss_nce"] = loss_nce.detach()
            extras["nce_active_modalities"] = nce_weights.sum().detach()
            extras["loss_trans"] = active_mean(loss_trans).detach()
            extras["loss_precision"] = active_mean(loss_precision).detach()
            extras["loss_var_range"] = active_mean(loss_var_range).detach()
            extras["loss_align"] = active_mean(align_loss).detach()
            extras["loss_prior_smooth"] = active_mean(prior_smooth).detach()
            extras["loss_ctx_inject"] = active_mean(loss_ctx_inject).detach()

        return ConsciousnessOutput(
            self_sem=self_sem_out,
            intent_sem=intent_sem_out,
            extras=extras,)
    
    @torch.no_grad()
    def ResetHebbianMemory(
        self,
        doneMask: Optional[torch.Tensor] = None,
        ) -> None:
        for module in self.modules():
            if isinstance(module, ConsciousHebbianLinear):
                module.ResetHebbianMemory(doneMask=doneMask)



class TestConsciousMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        self.mem_dim = 128
        self.world_dim = 64
        self.sym_dim = max(4, self.mem_dim // 8)
        self.dev_dim = 64
        self.self_dim = 128
        self.intent_dim = 128
        self.n_self_blocks = 3
        self.n_intent_blocks = 3
        self.hyper_hidden = 256

    def BuildModel(self) -> "ConsciousnessExtractor":
        model = ConsciousnessExtractor(
            memItemDim=self.mem_dim,
            worldItemDim=self.world_dim,
            symItemDim=self.sym_dim,
            devDim=self.dev_dim,
            selfDim=self.self_dim,
            intentDim=self.intent_dim,
            nSelfBlocks=self.n_self_blocks,
            nIntentBlocks=self.n_intent_blocks,
            hyperHiddenDim=self.hyper_hidden,
            topKMem=4,
            randKMem=2,
            topKWorld=4,
            randKWorld=2,).to(self.device)
        model.train()
        return model

    def DummyBanks(self, B: int = 8, Nm: int = 10, Nw: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = torch.randn(B, Nm, self.mem_dim, device=self.device)
        world = torch.randn(B, Nw, self.world_dim, device=self.device)
        return mem, world

    def DummyBankDicts(self, B: int = 8, Nm: int = 10, Nw: int = 7) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mem, world = self.DummyBanks(B=B, Nm=Nm, Nw=Nw)
        mem_dict = {
            "tokens": mem.contiguous(),
            "valid": torch.ones(B, Nm, dtype=torch.bool, device=self.device),}
        world_dict = {
            "tokens": world.contiguous(),
            "valid": torch.ones(B, Nw, dtype=torch.bool, device=self.device),}
        return mem_dict, world_dict

    def TestDictBankInterface(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()
            mem_dict, world_dict = self.DummyBankDicts(B=4, Nm=9, Nw=6)

            out = model(mem_dict, world_dict)
            assert out.self_sem.shape == (4, self.self_dim), f"self_sem shape wrong: {out.self_sem.shape}"
            assert out.intent_sem.shape == (4, self.intent_dim), f"intent_sem shape wrong: {out.intent_sem.shape}"
            assert "mem_n_items" in out.extras and "world_n_items" in out.extras
            assert (out.extras["mem_n_items"] > 0).all()
            assert (out.extras["world_n_items"] > 0).all()
            assert set(mem_dict.keys()) == {"tokens", "valid"}
            assert set(world_dict.keys()) == {"tokens", "valid"}

            invalid_banks = [
                {"tokens": mem_dict["tokens"]},
                {
                    "tokens": mem_dict["tokens"],
                    "valid": mem_dict["valid"],
                    "payload": mem_dict["tokens"],},
                {"items": mem_dict["tokens"], "mask": mem_dict["valid"]},]
            for invalid in invalid_banks:
                rejected = False
                try:
                    model(invalid, world_dict)
                except ValueError:
                    rejected = True
                assert rejected

            invalid_valid = {
                "tokens": mem_dict["tokens"],
                "valid": mem_dict["valid"].to(dtype=mem_dict["tokens"].dtype),}
            rejected = False
            try:
                model(invalid_valid, world_dict)
            except TypeError:
                rejected = True
            assert rejected

            rejected = False
            try:
                model(None, world_dict)
            except TypeError:
                rejected = True
            assert rejected

            model.eval()
            empty_mem, first_world = self.DummyBankDicts(B=2, Nm=0, Nw=3)
            second_world = {
                "tokens": torch.randn_like(first_world["tokens"]),
                "valid": first_world["valid"].clone(),}
            model.ResetState()
            first = model(empty_mem, first_world)
            model.ResetState()
            second = model(empty_mem, second_world)
            assert not torch.allclose(first.self_sem, second.self_sem)

            print("TestDictBankInterface passed.")
            return True
        except AssertionError as e:
            print("TestDictBankInterface failed:", e)
            return False
        except Exception as e:
            print("TestDictBankInterface error:", e)
            return False

    def TestForwardShapes(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()
            mem, world = self.DummyBankDicts(B=4, Nm=9, Nw=6)

            out = model(mem, world)

            assert out.self_sem.shape == (4, self.self_dim), f"self_sem shape wrong: {out.self_sem.shape}"
            assert out.intent_sem.shape == (4, self.intent_dim), f"intent_sem shape wrong: {out.intent_sem.shape}"

            dev_trace, last_intent, last_sem, step = model.GetState()
            assert dev_trace is not None, "dev_trace should not be None after first forward."
            assert last_intent is not None and last_sem is not None, "cached last_intent/last_sem should not be None."

            assert dev_trace.shape == (4, self.dev_dim), f"dev_trace shape wrong: {dev_trace.shape}"
            assert last_intent.shape == (4, self.intent_dim), f"last_intent shape wrong: {last_intent.shape}"
            assert last_sem.shape == (4, self.self_dim), f"last_sem shape wrong: {last_sem.shape}"
            assert isinstance(step, int) and step == 1, f"step should be int==1 after first forward, got {step}"

            for k in [
                "loss",
                "dev_trace_norm", "mem_score_mean", "world_score_mean",
                "mem_n_items", "world_n_items",
                "mem_focus_norm", "world_focus_norm",]:
                assert k in out.extras, f"extras missing key: {k}"

            print("TestForwardShapes passed.")
            return True
        except AssertionError as e:
            print("TestForwardShapes failed:", e)
            return False
        except Exception as e:
            print("TestForwardShapes error:", e)
            return False

    def TestConsciousnessExtractorIOShapes(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()

            B = 2
            memory_bank, world_bank = self.DummyBankDicts(B=B, Nm=9, Nw=6)

            def print_shape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            def print_nested(prefix: str, obj):
                if isinstance(obj, torch.Tensor):
                    print_shape(prefix, obj)
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        next_prefix = f"{prefix}.{key}" if prefix else key
                        print_nested(next_prefix, value)
                elif hasattr(obj, "_asdict"):
                    for key, value in obj._asdict().items():
                        next_prefix = f"{prefix}.{key}" if prefix else key
                        print_nested(next_prefix, value)

            with torch.no_grad():
                print_nested("input.memoryBank", memory_bank)
                print_nested("input.worldBank", world_bank)

                out = model(memory_bank, world_bank)

                print_nested("output", out)

            assert out.self_sem.shape == (B, self.self_dim)
            assert out.intent_sem.shape == (B, self.intent_dim)
            return True
        except Exception as e:
            print("ConsciousnessExtractor IO shapes error:", type(e).__name__, e)
            return False

    def TestStateFlow(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()

            B = 3
            mem, world = self.DummyBankDicts(B=B, Nm=8, Nw=5)

            steps = []
            dev_list = []

            for _ in range(6):
                _ = model(mem, world)
                dev_trace, last_intent, last_sem, step = model.GetState()
                assert dev_trace is not None, "dev_trace became None unexpectedly."
                steps.append(step)
                dev_list.append(dev_trace.cpu())

            assert all(steps[i] < steps[i + 1] for i in range(len(steps) - 1)), f"step not strictly increasing: {steps}"

            diff = (dev_list[-1] - dev_list[0]).abs().mean().item()
            assert diff > 1e-7, f"dev_trace did not change across steps; diff={diff:.3e}"

            print("TestStateFlow passed.")
            return True
        except AssertionError as e:
            print("TestStateFlow failed:", e)
            return False
        except Exception as e:
            print("TestStateFlow error:", e)
            return False

    def TestEvalModeInferenceBranch(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()
            model.eval()

            mem, world = self.DummyBankDicts(B=3, Nm=8, Nw=5)
            with torch.no_grad():
                out = model(mem, world)

            assert out.self_sem.shape == (3, self.self_dim), f"self_sem shape wrong in eval: {out.self_sem.shape}"
            assert out.intent_sem.shape == (3, self.intent_dim), f"intent_sem shape wrong in eval: {out.intent_sem.shape}"
            assert out.extras["loss"] is None, "eval mode should return extras['loss']=None"

            train_only_keys = [
                "loss_kl",
                "loss_recon",
                "loss_nce",
                "loss_trans",
                "loss_precision",
                "loss_var_range",
                "loss_align",
                "loss_prior_smooth",
                "loss_ctx_inject",]
            for k in train_only_keys:
                assert k not in out.extras, f"eval extras should not contain training-only key: {k}"

            assert torch.isfinite(out.self_sem).all(), "self_sem contains non-finite values in eval mode"
            assert torch.isfinite(out.intent_sem).all(), "intent_sem contains non-finite values in eval mode"

            print("TestEvalModeInferenceBranch passed.")
            return True
        except AssertionError as e:
            print("TestEvalModeInferenceBranch failed:", e)
            return False
        except Exception as e:
            print("TestEvalModeInferenceBranch error:", e)
            return False

    def TestColdStartNoInput(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()
            model.train()

            empty_mem, empty_world = self.DummyBankDicts(B=2, Nm=0, Nw=0)
            out = model(empty_mem, empty_world)
            out_repeat = model(empty_mem, empty_world)

            assert out.self_sem.shape == (2, self.self_dim), f"cold-start self_sem shape wrong: {out.self_sem.shape}"
            assert out.intent_sem.shape == (2, self.intent_dim), f"cold-start intent_sem shape wrong: {out.intent_sem.shape}"
            assert "cold_start" in out.extras, "cold-start extras missing key: cold_start"
            assert torch.all(out.extras["cold_start"] == 1.0), "cold_start flag should be all ones"
            assert out.extras["loss"] is not None, "training cold-start should provide zero loss tensor"
            assert out.extras["loss"].dim() == 0, "cold-start loss should be scalar"
            assert abs(float(out.extras["loss"].item())) < 1e-12, f"cold-start loss should be zero, got {out.extras['loss'].item()}"
            assert torch.all(out.extras["mem_n_items"] == 0.0), "cold-start mem_n_items should be zero"
            assert torch.all(out.extras["world_n_items"] == 0.0), "cold-start world_n_items should be zero"
            assert torch.equal(out_repeat.self_sem, out.self_sem)
            assert torch.equal(out_repeat.intent_sem, out.intent_sem)
            assert float(out_repeat.extras["loss"].item()) == 0.0

            dev_trace, last_intent, last_sem, step = model.GetState()
            assert step == 0, f"empty observations must not advance state, got step={step}"
            assert dev_trace is None and last_intent is None and last_sem is None

            empty_mem, world = self.DummyBankDicts(B=2, Nm=0, Nw=1)
            observed = model(empty_mem, world)
            observed_step = model.GetState()[-1]
            empty_after_state = model(empty_mem, empty_world)
            assert observed_step == 1
            assert model.GetState()[-1] == observed_step
            assert torch.allclose(empty_after_state.self_sem, observed.self_sem.detach())
            assert torch.allclose(empty_after_state.intent_sem, observed.intent_sem.detach())
            assert float(empty_after_state.extras["loss"].item()) == 0.0
            assert torch.all(empty_after_state.extras["no_observation"] == 1.0)

            print("TestColdStartNoInput passed.")
            return True
        except AssertionError as e:
            print("TestColdStartNoInput failed:", e)
            return False
        except Exception as e:
            print("TestColdStartNoInput error:", e)
            return False

    def TestLowTokenFallbackBranches(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()
            model.train()

            B = 3
            mem_dict: Dict[str, torch.Tensor] = {
                "tokens": torch.zeros(B, 0, self.mem_dim, device=self.device),
                "valid": torch.zeros(B, 0, dtype=torch.bool, device=self.device),}
            world_dict: Dict[str, torch.Tensor] = {
                "tokens": torch.randn(B, 1, self.world_dim, device=self.device),
                "valid": torch.ones(B, 1, dtype=torch.bool, device=self.device),}

            out = model(mem_dict, world_dict)

            zeros = torch.zeros(B, 1, device=self.device)
            assert torch.allclose(out.extras["mem_attn_entropy"], zeros), "mem_attn_entropy should be zero when memory_bank is empty"
            assert torch.allclose(out.extras["world_attn_entropy"], zeros), "world_attn_entropy should be zero when world tokens <= 1"
            assert torch.all(out.extras["mem_n_items"] == 0.0), "mem_n_items should be zero for empty memory bank"
            assert torch.all(out.extras["world_n_items"] == 1.0), "world_n_items should equal 1 for single-token world bank"
            assert torch.all(out.extras["mem_available"] == 0.0)
            assert torch.all(out.extras["world_available"] == 1.0)
            assert torch.all(out.extras["dev_available"] == 0.0)
            assert torch.all(out.extras["modal_precision_mem"] == 0.0)
            assert torch.all(out.extras["modal_precision_dev"] == 0.0)

            print("TestLowTokenFallbackBranches passed.")
            return True
        except AssertionError as e:
            print("TestLowTokenFallbackBranches failed:", e)
            return False
        except Exception as e:
            print("TestLowTokenFallbackBranches error:", e)
            return False

    def TestAvailabilityAwareAuxiliaryLoss(self) -> bool:
        try:
            model = self.BuildModel()
            model.ResetState()
            model.train()
            hidden = int(model.z_world_head.out_features // 2)
            with torch.no_grad():
                model.z_world_head.bias[hidden:].fill_(-20.0)

            memory, world = self.DummyBankDicts(B=1, Nm=0, Nw=1)
            out = model(memory, world)
            assert torch.isfinite(out.extras["loss"])
            assert float(out.extras["loss_nce"].item()) == 0.0
            assert float(out.extras["loss_precision"].item()) >= 0.0
            assert float(out.extras["loss_var_range"].item()) > 0.0
            assert float(out.extras["modal_precision_mem"].item()) == 0.0
            assert float(out.extras["modal_precision_dev"].item()) == 0.0

            model.zero_grad(set_to_none=True)
            out.extras["loss"].backward()
            logvar_grad = model.z_world_head.bias.grad[hidden:]
            assert torch.isfinite(logvar_grad).all()
            assert float(logvar_grad.abs().sum().item()) > 0.0

            print("TestAvailabilityAwareAuxiliaryLoss passed.")
            return True
        except AssertionError as e:
            print("TestAvailabilityAwareAuxiliaryLoss failed:", e)
            return False
        except Exception as e:
            print("TestAvailabilityAwareAuxiliaryLoss error:", e)
            return False

    def TestBoundedPrecisionAndNceEligibility(self) -> bool:
        try:
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                logvar = torch.linspace(-8.0, 8.0, 33, dtype=dtype)
                precision = torch.exp(-logvar)
                assert torch.isfinite(precision).all()
                assert torch.equal(
                    precision,
                    precision.float().clamp(min=1e-6, max=1e6).to(dtype=dtype))
                precision_sum = precision * 4.0
                inverse = 1.0 / precision_sum
                assert torch.isfinite(inverse).all()
                assert torch.equal(
                    inverse,
                    inverse.float().clamp_min(1e-8).to(dtype=dtype))
                variance = torch.exp(logvar)
                assert torch.equal(
                    variance,
                    variance.float().clamp_min(1e-6).to(dtype=dtype))

            model = self.BuildModel()
            model.ResetState()
            model.train()
            memory, world = self.DummyBankDicts(B=2, Nm=2, Nw=1)
            world["valid"][1, 0] = False
            out = model(memory, world)
            assert torch.isfinite(out.extras["loss_nce"])
            assert float(out.extras["nce_active_modalities"].item()) == 1.0

            print("TestBoundedPrecisionAndNceEligibility passed.")
            return True
        except Exception as e:
            print("TestBoundedPrecisionAndNceEligibility failed:", e)
            return False

    def TestMixedRowTokenValidity(self) -> bool:
        try:
            B = 2
            model = self.BuildModel()
            model.ResetState()
            model.train()

            initial_memory, initial_world = self.DummyBankDicts(B=B, Nm=0, Nw=1)
            _ = model(initial_memory, initial_world)
            prev_dev, prev_intent, prev_self, _ = model.GetState()
            assert prev_dev is not None and prev_intent is not None and prev_self is not None
            hebb_before = {
                id(module): module.hebb[1].detach().clone()
                for module in model.modules()
                if isinstance(module, ConsciousHebbianLinear)}

            values = torch.randn(B, 2, self.mem_dim, device=self.device, requires_grad=True)
            with torch.no_grad():
                values[1].fill_(1e4)
            memory = {
                "tokens": values,
                "valid": torch.tensor(
                    [[True, True], [False, False]],
                    device=self.device)}
            empty_world = {
                "tokens": torch.zeros(B, 0, self.world_dim, device=self.device),
                "valid": torch.zeros(B, 0, dtype=torch.bool, device=self.device),}
            out = model(memory, empty_world)

            assert torch.isfinite(out.self_sem).all()
            assert torch.isfinite(out.intent_sem).all()
            assert torch.isfinite(out.extras["loss"])
            assert out.extras["mem_n_items"][:, 0].tolist() == [2.0, 0.0]
            assert out.extras["mem_available"][:, 0].tolist() == [1.0, 0.0]
            assert out.extras["no_observation"][:, 0].tolist() == [0.0, 1.0]
            assert float(out.extras["modal_precision_mem"][1].item()) == 0.0
            assert float(out.extras["mem_attn_entropy"][1].item()) == 0.0
            assert torch.allclose(out.self_sem[1], prev_self[1])
            assert torch.allclose(out.intent_sem[1], prev_intent[1])
            next_dev, next_intent, next_self, _ = model.GetState()
            assert next_dev is not None and next_intent is not None and next_self is not None
            assert torch.equal(next_dev[1], prev_dev[1])
            assert torch.equal(next_intent[1], prev_intent[1])
            assert torch.equal(next_self[1], prev_self[1])
            for module in model.modules():
                if isinstance(module, ConsciousHebbianLinear):
                    assert torch.equal(module.hebb[1], hebb_before[id(module)])

            model.zero_grad(set_to_none=True)
            out.extras["loss"].backward()
            assert values.grad is not None
            assert torch.isfinite(values.grad).all()
            assert float(values.grad[1].abs().sum().item()) == 0.0

            print("TestMixedRowTokenValidity passed.")
            return True
        except AssertionError as e:
            print("TestMixedRowTokenValidity failed:", e)
            return False
        except Exception as e:
            print("TestMixedRowTokenValidity error:", e)
            return False

    def TestConsciousHebbianLinearLifecycle(self) -> bool:
        try:
            lin = ConsciousHebbianLinear(
                inFeatures=16,
                outFeatures=12).to(self.device)

            x = torch.randn(5, 16, device=self.device)
            lin.EnsureB(int(x.size(0)))
            with torch.no_grad():
                n0 = lin.hebb.norm().item()
                expected = F.linear(x, lin.weight, lin.bias)
                first = lin(x)
                assert torch.allclose(first, expected, atol=1e-7, rtol=1e-6)
                for _ in range(2):
                    _ = lin(x)
                n1 = lin.hebb.norm().item()
                assert n1 > n0 + 1e-12, f"hebb did not grow: before={n0:.3e}, after={n1:.3e}"
                assert "hebb" not in lin.state_dict()

                lin.ResetHebbianMemory()
                n2 = lin.hebb.norm().item()
                assert n2 < 1e-12, f"hebb not cleared to zero: now={n2:.3e}"

            print("TestConsciousHebbianLinearLifecycle passed.")
            return True
        except AssertionError as e:
            print("TestConsciousHebbianLinearLifecycle failed:", e)
            return False
        except Exception as e:
            print("TestConsciousHebbianLinearLifecycle error:", e)
            return False

    def TestModuleResetHebbianMemory(self) -> bool:
        try:
            model = self.BuildModel()
            mem, world = self.DummyBankDicts(B=4, Nm=9, Nw=6)

            with torch.no_grad():
                model.ResetState()
                for _ in range(3):
                    _ = model(mem, world)

            hebb_norm_before = []
            for m in model.modules():
                if hasattr(m, "hebb"):
                    hebb_norm_before.append(m.hebb.norm().item())
            assert len(hebb_norm_before) > 0, "No hebb buffers found in model."
            assert any(n > 1e-9 for n in hebb_norm_before), "All hebb norms are zero before reset; nothing to test."

            before = [
                module.hebb.detach().clone()
                for module in model.modules()
                if isinstance(module, ConsciousHebbianLinear)]
            model.ResetHebbianMemory(doneMask=torch.tensor(
                [True, False, True, False],
                device=self.device))
            after = [
                module.hebb.detach().clone()
                for module in model.modules()
                if isinstance(module, ConsciousHebbianLinear)]
            for previous, current in zip(before, after):
                assert torch.count_nonzero(current[[0, 2]]).item() == 0
                assert torch.equal(current[[1, 3]], previous[[1, 3]])

            model.ResetHebbianMemory()
            hebb_norm_after = []
            for m in model.modules():
                if hasattr(m, "hebb"):
                    hebb_norm_after.append(m.hebb.norm().item())
            assert all(n < 1e-12 for n in hebb_norm_after), f"Some hebb buffers not cleared: {hebb_norm_after}"

            print("TestModuleResetHebbianMemory passed.")
            return True
        except AssertionError as e:
            print("TestModuleResetHebbianMemory failed:", e)
            return False
        except Exception as e:
            print("TestModuleResetHebbianMemory error:", e)
            return False

    def TrainStepSmoke(self) -> bool:
        try:
            model = self.BuildModel()
            head = nn.Linear(self.self_dim + self.intent_dim, 32).to(self.device)

            model.train()
            head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            B = 4
            mem, world = self.DummyBankDicts(B=B, Nm=7, Nw=5)
            target = torch.randn(B, 32, device=self.device)

            model.ResetState()
            model.ResetHebbianMemory()

            _ = model(mem, world) 
            out = model(mem, world) 

            rep = torch.cat([out.self_sem, out.intent_sem], dim=-1)
            pred = head(rep)
            main_loss = F.mse_loss(pred, target)
            aux_loss = out.extras["loss"] if ("loss" in out.extras) else torch.zeros((), device=self.device)
            total_loss = main_loss + aux_loss

            opt.zero_grad(set_to_none=True)
            total_loss.backward()

            grads_ok = True
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    print("Missing grad at:", n)
                    grads_ok = False
                    break
                if not torch.isfinite(p.grad).all():
                    print("Non-finite grad at:", n)
                    grads_ok = False
                    break

            if head.weight.grad is None or not torch.isfinite(head.weight.grad).all():
                grads_ok = False

            assert grads_ok, "Some parameters have None or non-finite grad."
            opt.step()

            print("TrainStepSmoke passed.")
            return True
        except AssertionError as e:
            print("TrainStepSmoke failed:", e)
            return False
        except Exception as e:
            print("TrainStepSmoke error:", e)
            return False

    def NormalTrainingConvergence(self, steps: int = 80, logEvery: int = 20) -> bool:
        try:
            model = self.BuildModel()
            head = nn.Linear(self.self_dim + self.intent_dim, 32).to(self.device)

            model.train()
            head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            B = 16
            mem, world = self.DummyBankDicts(B=B, Nm=10, Nw=7)
            target = torch.randn(B, 32, device=self.device)

            with torch.no_grad():
                model.ResetState()
                model.ResetHebbianMemory()
                _ = model(mem, world)
                out0 = model(mem, world)
                rep0 = torch.cat([out0.self_sem, out0.intent_sem], dim=-1)
                pred0 = head(rep0)
                start = F.mse_loss(pred0, target).item()

            last_loss = start
            for t in range(1, steps + 1):
                model.ResetState()
                model.ResetHebbianMemory()

                _ = model(mem, world)
                out = model(mem, world)

                rep = torch.cat([out.self_sem, out.intent_sem], dim=-1)
                pred = head(rep)
                main_loss = F.mse_loss(pred, target)
                aux_loss = out.extras["loss"] if ("loss" in out.extras) else 0.0
                total_loss = main_loss + aux_loss

                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                opt.step()

                last_loss = float(total_loss.item())
                if (t % logEvery) == 0 or t == 1:
                    print(f"[ConsciousTrain] step {t}/{steps} | total_loss={last_loss:.6f}")

            with torch.no_grad():
                model.ResetState()
                model.ResetHebbianMemory()
                _ = model(mem, world)
                out1 = model(mem, world)
                rep1 = torch.cat([out1.self_sem, out1.intent_sem], dim=-1)
                pred1 = head(rep1)
                end = F.mse_loss(pred1, target).item()

            print(f"\n[ConsciousTrain] loss start={start:.6f} -> end={end:.6f}")
            assert end <= 0.8 * start, "Training did not show sufficient convergence (<20% decline)."

            print("NormalTrainingConvergence passed.")
            return True
        except AssertionError as e:
            print("NormalTrainingConvergence failed:", e)
            return False
        except Exception as e:
            print("NormalTrainingConvergence error:", e)
            return False

    def GradCoverageReport(self) -> bool:
        try:
            model = self.BuildModel()
            head = nn.Linear(self.self_dim + self.intent_dim, 32).to(self.device)
            model.train()
            head.train()

            B = 8
            mem, world = self.DummyBankDicts(B=B, Nm=9, Nw=6)
            target = torch.randn(B, 32, device=self.device)

            model.ResetState()
            model.ResetHebbianMemory()

            _ = model(mem, world)
            out = model(mem, world)

            rep = torch.cat([out.self_sem, out.intent_sem], dim=-1)
            pred = head(rep)
            main_loss = F.mse_loss(pred, target)
            aux_loss = out.extras["loss"] if ("loss" in out.extras) else 0.0
            total_loss = main_loss + aux_loss

            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
            opt.zero_grad(set_to_none=True)
            total_loss.backward()

            missing_any = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    missing_any.append(n)

            assert len(missing_any) == 0, f"Some trainable params have no grad: {missing_any}"
            print("GradCoverageReport passed.")
            return True
        except AssertionError as e:
            print("GradCoverageReport failed:", e)
            return False
        except Exception as e:
            print("GradCoverageReport error:", e)
            return False

    def TestParameterUpdateAfterStep(self) -> bool:
        try:
            model = self.BuildModel()
            head = nn.Linear(self.self_dim + self.intent_dim, 32).to(self.device)
            model.train()
            head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            B = 8
            mem, world = self.DummyBankDicts(B=B, Nm=9, Nw=6)
            target = torch.randn(B, 32, device=self.device)

            model.ResetState()
            model.ResetHebbianMemory()

            _ = model(mem, world)
            out = model(mem, world)

            rep = torch.cat([out.self_sem, out.intent_sem], dim=-1)
            pred = head(rep)
            main_loss = F.mse_loss(pred, target)
            aux_loss = out.extras["loss"] if ("loss" in out.extras) else torch.zeros((), device=self.device)
            total_loss = main_loss + aux_loss

            before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            head_before = head.weight.detach().clone()

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            opt.step()

            changed = []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                delta = (p.detach() - before[n]).abs().max().item()
                if delta > 1e-12:
                    changed.append(n)

            assert len(changed) > 0, "No trainable model parameter changed after optimizer step."
            changed_ratio = len(changed) / max(1, len(before))
            assert changed_ratio >= 0.1, f"Too few parameters updated: {len(changed)}/{len(before)}"

            required_update_names = [
                "self_in.0.weight",
                "intent_in.0.weight",]
            for n in required_update_names:
                assert n in before, f"Missing expected parameter in model: {n}"
                delta = (model.state_dict()[n] - before[n]).abs().max().item()
                assert delta > 1e-12, f"Expected parameter not updated: {n}"

            head_delta = (head.weight.detach() - head_before).abs().max().item()
            assert head_delta > 1e-12, "Auxiliary head parameter was not updated."

            print("TestParameterUpdateAfterStep passed.")
            return True
        except AssertionError as e:
            print("TestParameterUpdateAfterStep failed:", e)
            return False
        except Exception as e:
            print("TestParameterUpdateAfterStep error:", e)
            return False

    def QueryTopKEdgeCases(self) -> bool:
        try:
            model = self.BuildModel()
            B = 2
            D = self.mem_dim

            bank_empty = torch.zeros(B, 0, D, device=self.device)
            q = torch.randn(B, D, device=self.device)
            focus, idx, w = model.QueryTopK(bank_empty, q, topK=4)
            assert focus.shape == (B, D)
            assert idx.shape == (B, 0)
            assert w.shape == (B, 0)

            bank = torch.randn(B, 5, D, device=self.device)
            focus2, idx2, w2 = model.QueryTopK(bank, q, topK=0)
            assert focus2.shape == (B, D)
            assert idx2.shape == (B, 0)
            assert w2.shape == (B, 0)

            print("QueryTopKEdgeCases passed.")
            return True
        except AssertionError as e:
            print("QueryTopKEdgeCases failed:", e)
            return False
        except Exception as e:
            print("QueryTopKEdgeCases error:", e)
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "ForwardShapes": self.TestForwardShapes(),
            "ConsciousnessExtractorIOShapes": self.TestConsciousnessExtractorIOShapes(),
            "DictBankInterface": self.TestDictBankInterface(),
            "StateFlow": self.TestStateFlow(),
            "EvalModeInferenceBranch": self.TestEvalModeInferenceBranch(),
            "ColdStartNoInput": self.TestColdStartNoInput(),
            "LowTokenFallbackBranches": self.TestLowTokenFallbackBranches(),
            "AvailabilityAwareAuxiliaryLoss": self.TestAvailabilityAwareAuxiliaryLoss(),
            "BoundedPrecisionAndNceEligibility": self.TestBoundedPrecisionAndNceEligibility(),
            "MixedRowTokenValidity": self.TestMixedRowTokenValidity(),
            "ConsciousHebbianLinearLifecycle": self.TestConsciousHebbianLinearLifecycle(),
            "ModuleResetHebbianMemory": self.TestModuleResetHebbianMemory(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NormalTrainingConvergence": self.NormalTrainingConvergence(),
            "GradCoverageReport": self.GradCoverageReport(),
            "ParameterUpdateAfterStep": self.TestParameterUpdateAfterStep(),
            "QueryTopKEdgeCases": self.QueryTopKEdgeCases(),}

        passed = sum(1 for v in results.values() if v)
        print(f"\nConsciousness module tests: {passed}/{len(results)} passed.")
        return results
