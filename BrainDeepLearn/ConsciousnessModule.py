from __future__ import annotations 
from typing import Optional, Dict, Tuple, NamedTuple
from FunctionTools import AGICoreModule, RoPEMultiheadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F


BankInput = Optional[Dict[str, torch.Tensor]] # world, memory

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
        outFeatures: int,
        *,
        hebbAlpha: float = 0.05,
        useHebb: bool = True,):
        super().__init__()
        self.in_f = int(inFeatures)
        self.out_f = int(outFeatures)

        self.weight = nn.Parameter(torch.randn(self.out_f, self.in_f) * 0.02) # [O,I]
        self.bias = nn.Parameter(torch.zeros(self.out_f)) # [O]

        self.register_buffer("hebb", torch.zeros(1, self.out_f, self.in_f)) # [1,O,I]

        self.hebb_alpha = float(hebbAlpha)
        self.use_hebbian: bool = bool(useHebb)

    @torch.no_grad()
    def EnsureB(self, B: int, device, dtype):
        if self.hebb.size(0) != B :
            self.hebb = torch.zeros(B, self.out_f, self.in_f, device=device, dtype=dtype) # [B,O,I]

    @torch.no_grad()
    def ResetHebbianMemory(self):
        self.hebb.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, I = x.shape

        if self.use_hebbian:
            self.EnsureB(B, device=self.device, dtype=self.dtype)

        if self.use_hebbian:
            W_eff = self.weight.unsqueeze(0) + self.hebb # [B,O,I]
        else:
            W_eff = self.weight.unsqueeze(0).expand(B, -1, -1) # [B,O,I]

        y_b = torch.einsum("bi,boi->bo", x, W_eff) + self.bias.unsqueeze(0) # [B,O]

        if self.use_hebbian:
            with torch.no_grad():
                x_n = F.normalize(x, dim=-1, eps=1e-6) # [B,I]
                y_n = F.normalize(y_b, dim=-1, eps=1e-6) # [B,O]

                d_hebb = y_n.unsqueeze(-1) * x_n.unsqueeze(1) # [B, O, I]

                a = self.hebb_alpha
                self.hebb.mul_(1.0 - a)
                self.hebb.add_(a * d_hebb)

        return y_b # [B,O]


class NeuroGeoControlFusion(AGICoreModule):
    def __init__(
        self,
        devDim: int,
        selfDim: int,
        intentDim: int,
        *,
        useHebb: bool = True,
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
                ConsciousHebbianLinear(self.feat_dim, self.dev_dim, useHebb=useHebb),
                nn.LayerNorm(self.dev_dim),
                nn.Tanh(),) for _ in range(self.n_charts)])

        self.base_head = nn.Sequential(
            ConsciousHebbianLinear(self.feat_dim, self.dev_dim, useHebb=useHebb),
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
    def TypeModuleName(self, typeKey: str) -> str:
        return f"type_{typeKey}"

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
        randKWorld: int = 64,
        useHebb: bool = True, ):
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
        self.use_hebbian = useHebb

        self.mem_type_keys: Tuple[str, ...] = ("gws", "kv", "ltm_sem", "ltm_epi", "sym")
        self.world_type_keys: Tuple[str, ...] = ("vals",)
        self.world_typed_min_tokens: int = 2

        def make_score_net(item_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(item_dim, hiddenDim),
                nn.LayerNorm(hiddenDim),
                nn.GELU(),
                nn.Linear(hiddenDim, 1),)

        def make_agg_proj(item_dim: int) -> nn.Sequential:
            hidden = 2 * item_dim
            return nn.Sequential(
                nn.Linear(3 * item_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, item_dim),
                nn.LayerNorm(item_dim),
                nn.GELU(),)

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

        self.mem_type_score_nets = nn.ModuleDict({
            self.TypeModuleName(k): make_score_net(self.mem_item_dim) for k in self.mem_type_keys})
        
        self.world_type_score_nets = nn.ModuleDict({
            self.TypeModuleName(k): make_score_net(self.world_item_dim) for k in self.world_type_keys})

        self.mem_agg_proj = nn.Sequential(
            nn.Linear(3 * self.mem_item_dim, self.mem_item_dim),
            nn.LayerNorm(self.mem_item_dim),
            nn.GELU(),)
        
        self.world_agg_proj = nn.Sequential(
            nn.Linear(3 * self.world_item_dim, self.world_item_dim),
            nn.LayerNorm(self.world_item_dim),
            nn.GELU(),)

        self.mem_type_agg_proj = nn.ModuleDict({
            self.TypeModuleName(k): make_agg_proj(self.mem_item_dim) for k in self.mem_type_keys})
        
        self.world_type_agg_proj = nn.ModuleDict({
            self.TypeModuleName(k): make_agg_proj(self.world_item_dim) for k in self.world_type_keys})

        self.mem_type_fuse_proj = nn.Sequential(
            nn.Linear(len(self.mem_type_keys) * self.mem_item_dim, self.mem_item_dim),
            nn.LayerNorm(self.mem_item_dim),
            nn.GELU(),)
        
        self.world_type_fuse_proj = nn.Sequential(
            nn.Linear(len(self.world_type_keys) * self.world_item_dim, self.world_item_dim),
            nn.LayerNorm(self.world_item_dim),
            nn.GELU(),)

        self.mem_type_gate = nn.Linear(len(self.mem_type_keys) * self.mem_item_dim, len(self.mem_type_keys))

        self.world_type_gate = nn.Linear(len(self.world_type_keys) * self.world_item_dim, len(self.world_type_keys))

        self.mem_type_film = nn.Linear(len(self.mem_type_keys) * self.mem_item_dim, 2 * self.mem_item_dim)
        self.world_type_film = nn.Linear(len(self.world_type_keys) * self.world_item_dim, 2 * self.world_item_dim)
        self.mem_type_alpha_net = nn.Linear(len(self.mem_type_keys) * self.mem_item_dim, 1)
        self.world_type_alpha_net = nn.Linear(len(self.world_type_keys) * self.world_item_dim, 1)
        self.mem_type_mix_norm = nn.LayerNorm(self.mem_item_dim)
        self.mem_type_film_norm = nn.LayerNorm(self.mem_item_dim)
        self.world_type_mix_norm = nn.LayerNorm(self.world_item_dim)
        self.world_type_film_norm = nn.LayerNorm(self.world_item_dim)

        self.mem_ctx_blend = nn.Sequential(
            nn.LayerNorm(2 * self.mem_item_dim),
            nn.Linear(2 * self.mem_item_dim, self.mem_item_dim),
            nn.GELU(),
            nn.Linear(self.mem_item_dim, 1),
            nn.Sigmoid(),)
        
        self.world_ctx_blend = nn.Sequential(
            nn.LayerNorm(2 * self.world_item_dim),
            nn.Linear(2 * self.world_item_dim, self.world_item_dim),
            nn.GELU(),
            nn.Linear(self.world_item_dim, 1),
            nn.Sigmoid(),)

        def make_dim_mapper(outDim: int, inDim: int) -> nn.Sequential:
            first = nn.Linear(inDim, outDim)
            return nn.Sequential(
                first,
                nn.LayerNorm(outDim),
                nn.GELU(),
                nn.Linear(outDim, outDim),)

        self.mem_type_dim_mapper = nn.ModuleDict({
            self.TypeModuleName("gws"): make_dim_mapper(self.mem_item_dim, self.mem_item_dim),
            self.TypeModuleName("kv"): make_dim_mapper(self.mem_item_dim, self.mem_item_dim),
            self.TypeModuleName("ltm_sem"): make_dim_mapper(self.mem_item_dim, self.mem_item_dim),
            self.TypeModuleName("ltm_epi"): make_dim_mapper(self.mem_item_dim, self.mem_item_dim),
            self.TypeModuleName("sym"): make_dim_mapper(self.mem_item_dim, self.sym_item_dim),})

        self.world_type_dim_mapper = nn.ModuleDict({
            self.TypeModuleName("vals"): make_dim_mapper(self.world_item_dim, self.world_item_dim),})

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
            ConsciousHebbianLinear(in_self,self.self_dim,useHebb=useHebb,),
            nn.LayerNorm(self.self_dim),
            nn.GELU(),)
        
        self.self_blocks = nn.ModuleList([FiLMBlock(self.self_dim, self.self_dim) for _ in range(self.n_self_blocks)])

        in_intent = self.self_dim + self.mem_item_dim + self.dev_dim
        self.intent_in = nn.Sequential(
            ConsciousHebbianLinear(in_intent,self.intent_dim,useHebb=useHebb,),
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
            useHebb=useHebb,
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
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=True)

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if self._dev_trace.size(0) != B:
            self._dev_trace = torch.zeros(B, self.dev_dim, device=device, dtype=dtype)
            self._last_self_intent = torch.zeros(B, self.intent_dim, device=device, dtype=dtype)
            self._last_sem = torch.zeros(B, self.self_dim, device=device, dtype=dtype)
            self._step = torch.zeros(1, device=device, dtype=torch.long)

    @torch.no_grad()
    def ResetState(self):
        self._dev_trace.zero_()
        self._last_self_intent.zero_()
        self._last_sem.zero_()
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
        topK: int,):

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

        k_top = min(topK, N)
        if k_top <= 0:
            focus = bank.new_zeros(B, D)
            top_idx = torch.zeros(B, 0, dtype=torch.long, device=device)
            top_w = torch.zeros(B, 0, device=device)
            return focus, top_idx, top_w

        top_scores, top_idx = torch.topk(sim, k=k_top, dim=1) # [B, k_top]
        top_w = F.softmax(top_scores, dim=-1) # [B, k_top]

        top_items = torch.gather(bank, 1,top_idx.unsqueeze(-1).expand(-1, -1, D) )

        focus = torch.einsum('bk,bkd->bd', top_w, top_items) 

        return focus, top_idx, top_w # focus: [B, D], top_w: [B, k_top], top_idx: [B, k_top]


    def AggregateBank(
        self,
        bankTensor: torch.Tensor, # [B, N, D]
        scoreNet: nn.Module,
        topK: int,
        randK: int,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        bank = bankTensor
        score_net = scoreNet
        top_k = int(topK)
        rand_k = int(randK)

        B, N, D = bank.shape
        device = self.device

        if N == 0:
            summary = torch.zeros(B, 3 * D, device=device, dtype=self.dtype)
            stats = {
                "score_mean": torch.zeros(B, 1, device=device),
                "n_items": torch.zeros(B, 1, device=device),}
            
            return summary, stats

        scores = score_net(bank).squeeze(-1) # [B, N]

        global_mean = bank.mean(dim=1) # [B, D]

        k_top = min(top_k, N)
        if k_top > 0:
            top_scores, top_idx = torch.topk(scores, k=k_top, dim=1) # [B, k_top]

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D) 
            top_items = torch.gather(bank, 1, idx_exp) # [B, k_top, D]

            top_w = F.softmax(top_scores, dim=-1)  
            top_mean = torch.einsum("bk,bkd->bd", top_w, top_items) # [B,D]
        else:
            top_mean = torch.zeros(B, D, device=device, dtype=bank.dtype)
            top_idx = None

        k_rand = min(rand_k, max(0, N - k_top))
        if k_rand > 0:
            avail_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            if k_top > 0:
                avail_mask.scatter_(1, top_idx, False)

            rand_noise = torch.rand(B, N, device=device)
            rand_scores = rand_noise + (~avail_mask) * (-1e6) # [B, N]

            _, rand_idx = torch.topk(rand_scores, k=k_rand, dim=1) # [B, k_rand]
            idx_exp2 = rand_idx.unsqueeze(-1).expand(-1, -1, D)
            rand_items = torch.gather(bank, 1, idx_exp2) # [B, k_rand]
            rand_mean = rand_items.mean(dim=1) # [B, D]
        else:
            rand_mean = torch.zeros(B, D, device=device, dtype=bank.dtype)

        summary = torch.cat([global_mean, top_mean, rand_mean], dim=-1)

        stats = {
            "score_mean": scores.mean(dim=1, keepdim=True),
            "n_items": torch.full((B,1), float(N), device=device),}

        return summary, stats # summary: [B, 3D]

    def FirstTensorFromBank(self, bank: BankInput) -> Optional[torch.Tensor]:
        if isinstance(bank, dict):
            for v in bank.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    return v
        return None

    def ResolveDimMapper(self, bankRole: str, typeKey: str) -> nn.Module:
        if bankRole == "mem":
            mapper_dict = self.mem_type_dim_mapper
        elif bankRole == "world":
            mapper_dict = self.world_type_dim_mapper
        else:
            raise ValueError(f"Unknown bankRole: {bankRole}")

        mk = self.TypeModuleName(typeKey)
        if mk in mapper_dict:
            return mapper_dict[mk]
        raise KeyError(f"Unknown typeKey for {bankRole} bank: {typeKey}")

    def MapBankItemDimWithNet(
        self,
        x: torch.Tensor,
        targetDim: int,
        bankRole: str,
        typeKey: str,) -> torch.Tensor:

        B, N, D = x.shape
        if N == 0:
            return torch.zeros(B, 0, int(targetDim), device=x.device, dtype=x.dtype)

        mapper = self.ResolveDimMapper(bankRole=bankRole, typeKey=typeKey)

        z = x.reshape(B * N, D)
        z = mapper(z)

        z = z.reshape(B, N, int(targetDim))
        return z # [B, N, D]


    def NormalizeBankDictInput(
        self,
        bank: BankInput,
        *,
        expectedB: int,
        targetDim: int,
        bankRole: str,
        preferredKeys: Tuple[str, ...],
        device: torch.device,
        dtype: torch.dtype,) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if (bank is not None) and (not isinstance(bank, dict)):
            raise TypeError("Bank input must be Dict[str, Tensor] or None")

        for k in preferredKeys:
            if (bank is None) or (k not in bank) or (not isinstance(bank[k], torch.Tensor)):
                out[k] = torch.zeros(expectedB, 0, targetDim, device=device, dtype=dtype)
                continue

            v = bank[k]
            out[k] = self.MapBankItemDimWithNet(
                v,
                targetDim=targetDim,
                bankRole=bankRole,
                typeKey=k)

        return out # [B, N, D]

    def AggregateTypedBank(
        self,
        bankByType: Dict[str, torch.Tensor],
        *,
        bankRole: str,
        topK: int,
        randK: int,
        expectedB: int,
        targetDim: int,
        device: torch.device,
        dtype: torch.dtype,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if bankRole == "mem":
            typeKeys = self.mem_type_keys
            scoreNets = self.mem_type_score_nets
            aggProjs = self.mem_type_agg_proj
            fuseProj = self.mem_type_fuse_proj
            gateNet = self.mem_type_gate
            filmNet = self.mem_type_film
            alphaNet = self.mem_type_alpha_net
            mixNorm = self.mem_type_mix_norm
            filmNorm = self.mem_type_film_norm
        elif bankRole == "world":
            typeKeys = self.world_type_keys
            scoreNets = self.world_type_score_nets
            aggProjs = self.world_type_agg_proj
            fuseProj = self.world_type_fuse_proj
            gateNet = self.world_type_gate
            filmNet = self.world_type_film
            alphaNet = self.world_type_alpha_net
            mixNorm = self.world_type_mix_norm
            filmNorm = self.world_type_film_norm
        else:
            raise ValueError(f"Unknown bankRole: {bankRole}")

        zero_bank = torch.zeros(expectedB, 0, targetDim, device=device, dtype=dtype)

        per_type_ctx = []
        per_type_score = []
        per_type_n = []
        per_type_stats: Dict[str, torch.Tensor] = {}

        for k in typeKeys:
            cur = bankByType.get(k, zero_bank)
            mk = self.TypeModuleName(k)
            summary, st = self.AggregateBank(cur, scoreNets[mk], topK=topK, randK=randK)
            ctx_k = aggProjs[mk](summary) + summary[:, :targetDim]

            per_type_ctx.append(ctx_k)
            per_type_score.append(st["score_mean"])
            per_type_n.append(st["n_items"])
            per_type_stats[f"{k}_score_mean"] = st["score_mean"]
            per_type_stats[f"{k}_n_items"] = st["n_items"]

        if len(per_type_ctx) == 0:
            ctx = torch.zeros(expectedB, targetDim, device=device, dtype=dtype)
            stats = {
                "score_mean": torch.zeros(expectedB, 1, device=device, dtype=dtype),
                "n_items": torch.zeros(expectedB, 1, device=device, dtype=dtype),
                "type_gate": torch.zeros(expectedB, 0, device=device, dtype=dtype),}
            return ctx, stats, per_type_stats

        ctx_cat = torch.cat(per_type_ctx, dim=-1)
        gate_logits = gateNet(ctx_cat)
        gate = F.softmax(gate_logits, dim=-1)

        ctx_stack = torch.stack(per_type_ctx, dim=1)
        ctx_mix = torch.einsum("bt,btd->bd", gate, ctx_stack)
        ctx_proj = fuseProj(ctx_cat)
        gamma_beta = filmNet(ctx_cat)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        ctx_film = ctx_proj * (1.0 + torch.tanh(gamma)) + beta
        alpha = torch.sigmoid(alphaNet(ctx_cat))
        ctx = mixNorm(ctx_mix) + alpha * filmNorm(ctx_film)

        score_mean = torch.stack(per_type_score, dim=1).mean(dim=1)
        n_items = torch.stack(per_type_n, dim=1).sum(dim=1)
        stats = {
            "score_mean": score_mean,
            "n_items": n_items,
            "type_gate": gate,}
        return ctx, stats, per_type_stats

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
        # Tmem = len(self.mem_type_keys)
        # Tworld = len(self.world_type_keys)
        # Nm = memory_bank.size(1) memory token
        # Nw = world_bank.size(1) world token 

        ref_mem = self.FirstTensorFromBank(memoryBank)
        ref_world = self.FirstTensorFromBank(worldBank)
        ref = ref_mem if ref_mem is not None else ref_world

        device = self.device
        dtype = self.dtype

        if ref is None:
            B = int(self._dev_trace.size(0))
        else:
            B = int(ref.size(0))

        self.EnsureB(B, device=device, dtype=dtype)

        if (ref is None) and (int(self._step.item()) <= 0):
            self_sem = self._last_sem
            intent_sem = self._last_self_intent
            zeros_1 = torch.zeros(B, 1, device=device, dtype=dtype)
            extras: Dict[str, torch.Tensor] = {
                "loss": (torch.zeros((), device=device, dtype=dtype) if self.training else None),
                "dev_trace_norm": zeros_1,
                "mem_score_mean": zeros_1,
                "world_score_mean": zeros_1,
                "mem_n_items": zeros_1,
                "world_n_items": zeros_1,
                "mem_focus_norm": zeros_1,
                "world_focus_norm": zeros_1,
                "mem_type_gate": torch.zeros(B, len(self.mem_type_keys), device=device, dtype=dtype),
                "world_type_gate": torch.zeros(B, len(self.world_type_keys), device=device, dtype=dtype),
                "cold_start": torch.ones(B, 1, device=device, dtype=dtype),}
            for k in self.mem_type_keys:
                extras[f"mem_type_{k}_score_mean"] = zeros_1
                extras[f"mem_type_{k}_n_items"] = zeros_1
            for k in self.world_type_keys:
                extras[f"world_type_{k}_score_mean"] = zeros_1
                extras[f"world_type_{k}_n_items"] = zeros_1

            self._step.add_(1)
            
            return ConsciousnessOutput(
                self_sem=self_sem,
                intent_sem=intent_sem,
                extras=extras,)

        mem_by_type = self.NormalizeBankDictInput(
            memoryBank,
            expectedB=B,
            targetDim=self.mem_item_dim,
            bankRole="mem",
            preferredKeys=self.mem_type_keys,
            device=device,
            dtype=dtype,)
        memory_bank = torch.cat([mem_by_type[k] for k in self.mem_type_keys], dim=1).contiguous() # [B, N_k, Dm]

        world_by_type = self.NormalizeBankDictInput(
            worldBank,
            expectedB=B,
            targetDim=self.world_item_dim,
            bankRole="world",
            preferredKeys=self.world_type_keys,
            device=device,
            dtype=dtype,)
        world_bank = torch.cat([world_by_type[k] for k in self.world_type_keys], dim=1).contiguous() # [B, N_k, Dw]

        if int(self._step.item()) <= 0:
            dev_ctx = torch.zeros(B, self.dev_dim, device=device, dtype=dtype)
        else:
            arousal = self.arousal_net(self._dev_trace)
            alpha = self.update_base * (0.5 + arousal)
            delta = self.dev_update(self._dev_trace, self._last_sem, self._last_self_intent)

            dev_ctx = self._dev_trace + alpha * delta # [B, Dd]

        mem_summary_raw, mem_stats_global = self.AggregateBank(
            memory_bank,
            self.mem_score_net,
            topK=self.top_k_mem,
            randK=self.rand_k_mem,)
        mem_ctx_global = self.mem_agg_proj(mem_summary_raw) # [B, Dm]

        mem_ctx_typed, mem_stats_typed, mem_type_stats = self.AggregateTypedBank(
            mem_by_type,
            bankRole="mem",
            topK=self.top_k_mem,
            randK=self.rand_k_mem,
            expectedB=B,
            targetDim=self.mem_item_dim,
            device=device,
            dtype=dtype,) # mem_ctx_typed: [B, Dm]
        
        mem_alpha = self.mem_ctx_blend(torch.cat([mem_ctx_global, mem_ctx_typed], dim=-1))

        mem_ctx = mem_ctx_global + mem_alpha * (mem_ctx_typed - mem_ctx_global) # [B, Dm]
        
        mem_stats = {
            "score_mean": mem_stats_global["score_mean"] + mem_alpha * (mem_stats_typed["score_mean"] - mem_stats_global["score_mean"]),
            "n_items": mem_stats_global["n_items"],
            "type_gate": mem_stats_typed["type_gate"],}

        world_summary_raw, world_stats_global = self.AggregateBank(
            world_bank,
            self.world_score_net,
            topK=self.top_k_world,
            randK=self.rand_k_world,) # world_summary_raw" [B, 3*Dw]
        
        world_ctx_global = self.world_agg_proj(world_summary_raw) #: [B, Dw]
        world_use_typed = (world_bank.size(1) >= int(self.world_typed_min_tokens))
        if world_use_typed:
            world_ctx_typed, world_stats_typed, world_type_stats = self.AggregateTypedBank(
                world_by_type,
                bankRole="world",
                topK=self.top_k_world,
                randK=self.rand_k_world,
                expectedB=B,
                targetDim=self.world_item_dim,
                device=device,
                dtype=dtype,) # world_ctx_typed: [B, Dw]
            
            world_alpha = self.world_ctx_blend(torch.cat([world_ctx_global, world_ctx_typed], dim=-1))

            world_ctx = world_ctx_global + world_alpha * (world_ctx_typed - world_ctx_global) #: [B,Dw]

            world_stats = {
                "score_mean": world_stats_global["score_mean"] + world_alpha * (world_stats_typed["score_mean"] - world_stats_global["score_mean"]),
                "n_items": world_stats_global["n_items"],
                "type_gate": world_stats_typed["type_gate"],}
        else:
            world_alpha = torch.zeros(B, 1, device=device, dtype=dtype)
            world_ctx = world_ctx_global
            world_type_stats = {}
            for k in self.world_type_keys:
                world_type_stats[f"{k}_score_mean"] = world_stats_global["score_mean"]
                world_type_stats[f"{k}_n_items"] = world_stats_global["n_items"]
            world_stats = {
                "score_mean": world_stats_global["score_mean"],
                "n_items": world_stats_global["n_items"],
                "type_gate": torch.ones(B, len(self.world_type_keys), device=device, dtype=dtype),}

        ctx_raw = torch.cat([world_ctx, mem_ctx, dev_ctx], dim=-1)
        ctx_norm = self.ctx_norm(ctx_raw)
        ctx_base = self.ctx_proj(ctx_norm) # [B, H]

        prior_seed = torch.cat([self._last_sem, self._last_self_intent, self._dev_trace], dim=-1) # [B, Ds+Di+Dd]
        prior_feat = self.ctx_prior(prior_seed) # [B,H]
        mu_prior, logvar_prior = self.z_prior_head(prior_feat).chunk(2, dim=-1) # [B,H]
        logvar_prior = logvar_prior.clamp(-8.0, 8.0)
        prec_prior = torch.exp(-logvar_prior).clamp(min=1e-6, max=1e6) # [B,H]

        q_seed = prior_feat if int(self._step.item()) > 0 else ctx_base
        q = self.ctx_query_proj(q_seed).unsqueeze(1)

        if memory_bank.size(1) > 0:
            mem_tokens = self.mem_token_proj(memory_bank)
            mem_attn_out, mem_attn_prob = self.mem_cross_attn(q, mem_tokens, mem_tokens, needWeights=True)
            mem_attn_vec = mem_attn_out.squeeze(1) # [B,H]
            mem_attn_prob = mem_attn_prob.squeeze(1) # [B,Nm]
            mem_attn_entropy = -(mem_attn_prob * mem_attn_prob.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True) # [B,1]
        else:
            mem_attn_vec = torch.zeros(B, ctx_base.size(-1), device=device, dtype=dtype)
            mem_attn_prob = torch.zeros(B, 0, device=device, dtype=dtype)
            mem_attn_entropy = torch.zeros(B, 1, device=device, dtype=dtype)

        world_use_attn = world_bank.size(1) > 1
        if world_use_attn:
            world_tokens = self.world_token_proj(world_bank)
            world_attn_out, world_attn_prob = self.world_cross_attn(q, world_tokens, world_tokens, needWeights=True)
            world_attn_vec = world_attn_out.squeeze(1) # [B,H]
            world_attn_prob = world_attn_prob.squeeze(1) # [B,Nw]
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

        mu_world, logvar_world = self.z_world_head(world_feat).chunk(2, dim=-1)
        mu_mem, logvar_mem = self.z_mem_head(mem_feat).chunk(2, dim=-1)
        mu_dev, logvar_dev = self.z_dev_head(dev_feat).chunk(2, dim=-1)

        logvar_world = (logvar_world + logvar_world_scalar).clamp(-8.0, 8.0)
        logvar_mem = (logvar_mem + logvar_mem_scalar).clamp(-8.0, 8.0)
        logvar_dev = (logvar_dev + logvar_dev_scalar).clamp(-8.0, 8.0)

        prec_world = torch.exp(-logvar_world).clamp(min=1e-6, max=1e6) # [B,H]
        prec_mem = torch.exp(-logvar_mem).clamp(min=1e-6, max=1e6) # [B,H]
        prec_dev = torch.exp(-logvar_dev).clamp(min=1e-6, max=1e6) # [B,H]

        prec_sum = (prec_world + prec_mem + prec_dev + prec_prior).clamp_min(1e-6)

        mu_post = (
            mu_world * prec_world +
            mu_mem * prec_mem +
            mu_dev * prec_dev +
            mu_prior * prec_prior) / prec_sum # [B,H]
        logvar_post = torch.log((1.0 / prec_sum).clamp_min(1e-8)) # [B,H]

        if self.training:
            eps = torch.randn_like(mu_post)
            z_post = mu_post + torch.exp(0.5 * logvar_post) * eps
        else:
            z_post = mu_post

        obs_prec_scalar = torch.exp(-torch.cat([logvar_world_scalar, logvar_mem_scalar, logvar_dev_scalar], dim=-1))
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
            [prec_world.mean(dim=-1), prec_mem.mean(dim=-1), prec_dev.mean(dim=-1)],
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

        mem_focus, mem_idx, mem_w = self.QueryTopK(memory_bank, q_mem, self.top_k_mem) 

        world_focus, world_idx, world_w = self.QueryTopK(world_bank, q_world, self.top_k_world) 

        mem_delta = self.mem_focus_to_self(mem_focus) # [B,Ds]
        world_delta = self.world_focus_to_self(world_focus) # [B,Ds]

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
            def symmetric_info_nce(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
                a_n = F.normalize(a, dim=-1)
                b_n = F.normalize(b, dim=-1)
                logits = (a_n @ b_n.transpose(0, 1)) / tau
                labels = torch.arange(a_n.size(0), device=a_n.device)
                loss_ab = F.cross_entropy(logits, labels)
                loss_ba = F.cross_entropy(logits.transpose(0, 1), labels)
                return 0.5 * (loss_ab + loss_ba)

            ctx_hat = self.dev_to_ctx(dev_ctx) 
            ctx_align_loss = F.mse_loss(ctx_hat, ctx_vec.detach(), reduction="none").mean(dim=1, keepdim=True)
            slow_loss = (dev_ctx - self._dev_trace).pow(2).mean(dim=1, keepdim=True)

            kl = 0.5 * (
                logvar_prior - logvar_post +
                (torch.exp(logvar_post) + (mu_post - mu_prior).pow(2)) / torch.exp(logvar_prior).clamp_min(1e-6) -1.0)
            loss_kl = kl.mean(dim=-1, keepdim=True)

            world_recon = self.z_to_world(z_post)
            mem_recon = self.z_to_mem(z_post)
            dev_recon = self.z_to_dev(z_post)
            loss_recon = (
                F.smooth_l1_loss(world_recon, world_feat, reduction="none").mean(dim=-1, keepdim=True) +
                F.smooth_l1_loss(mem_recon, mem_feat, reduction="none").mean(dim=-1, keepdim=True) +
                F.smooth_l1_loss(dev_recon, dev_feat, reduction="none").mean(dim=-1, keepdim=True)) / 3.0

            z_n = self.nce_proj_z(z_post)
            w_n = self.nce_proj_world(world_feat)
            m_n = self.nce_proj_mem(mem_feat)
            d_n = self.nce_proj_dev(dev_feat)
            loss_nce = (
                symmetric_info_nce(z_n, w_n, self.nce_temp) +
                symmetric_info_nce(z_n, m_n, self.nce_temp) +
                symmetric_info_nce(z_n, d_n, self.nce_temp)) / 3.0

            z_pred_prev = self.z_next_pred(prior_seed)
            loss_trans = F.smooth_l1_loss(z_pred_prev, mu_post.detach(), reduction="none").mean(dim=-1, keepdim=True)

            precision_nll = 0.5 * (
                torch.exp(-logvar_world) * (mu_post - mu_world).pow(2) + logvar_world +
                torch.exp(-logvar_mem) * (mu_post - mu_mem).pow(2) + logvar_mem +
                torch.exp(-logvar_dev) * (mu_post - mu_dev).pow(2) + logvar_dev)
            loss_precision = precision_nll.mean(dim=-1, keepdim=True)
            loss_var_range = (
                F.relu(logvar_world - 4.0).mean(dim=-1, keepdim=True) + F.relu(-4.0 - logvar_world).mean(dim=-1, keepdim=True) +
                F.relu(logvar_mem - 4.0).mean(dim=-1, keepdim=True) + F.relu(-4.0 - logvar_mem).mean(dim=-1, keepdim=True) +
                F.relu(logvar_dev - 4.0).mean(dim=-1, keepdim=True) + F.relu(-4.0 - logvar_dev).mean(dim=-1, keepdim=True)) / 3.0

            align_loss = (
                (1.0 - F.cosine_similarity(world_feat, mem_feat, dim=-1)) +
                (1.0 - F.cosine_similarity(world_feat, dev_feat, dim=-1)) +
                (1.0 - F.cosine_similarity(mem_feat, dev_feat, dim=-1))) / 3.0
            align_loss = align_loss.unsqueeze(-1)

            prior_smooth = F.smooth_l1_loss(ctx_vec, prior_feat.detach(), reduction="none").mean(dim=1, keepdim=True)
            
            loss_ctx_inject = (
                (ctx_self_gain * ctx_self_delta).pow(2).mean(dim=-1, keepdim=True) +
                (ctx_intent_gain * ctx_intent_delta).pow(2).mean(dim=-1, keepdim=True)) * 0.5

            loss = (
                0.05 * ctx_align_loss.mean() +
                1e-3 * slow_loss.mean() +
                0.02 * loss_kl.mean() +
                0.04 * loss_recon.mean() +
                0.02 * loss_nce +
                0.02 * loss_trans.mean() +
                0.01 * loss_precision.mean() +
                0.003 * align_loss.mean() +
                0.01 * prior_smooth.mean() +
                0.005 * loss_var_range.mean() +
                0.003 * loss_ctx_inject.mean())
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

        self._dev_trace = dev_ctx.detach()
        self._last_self_intent = h_intent.detach()
        self._last_sem = self_sem.detach()
        self._step.add_(1)

        extras: Dict[str, torch.Tensor] = {
            "loss": loss,
            "dev_trace_norm": dev_ctx.norm(dim=-1, keepdim=True).detach(),
            "mem_score_mean": mem_stats["score_mean"].detach(),
            "world_score_mean": world_stats["score_mean"].detach(),
            "mem_ctx_blend_alpha": mem_alpha.detach(),
            "world_ctx_blend_alpha": world_alpha.detach(),
            "modal_precision_world": w_world.detach(),
            "modal_precision_mem": w_mem.detach(),
            "modal_precision_dev": w_dev.detach(),
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
            extras["loss_kl"] = loss_kl.mean().detach()
            extras["loss_recon"] = loss_recon.mean().detach()
            extras["loss_nce"] = loss_nce.detach()
            extras["loss_trans"] = loss_trans.mean().detach()
            extras["loss_precision"] = loss_precision.mean().detach()
            extras["loss_var_range"] = loss_var_range.mean().detach()
            extras["loss_align"] = align_loss.mean().detach()
            extras["loss_prior_smooth"] = prior_smooth.mean().detach()
            extras["loss_ctx_inject"] = loss_ctx_inject.mean().detach()

        if "type_gate" in mem_stats:
            extras["mem_type_gate"] = mem_stats["type_gate"].detach()
        if "type_gate" in world_stats:
            extras["world_type_gate"] = world_stats["type_gate"].detach()
        for k, v in mem_type_stats.items():
            extras[f"mem_type_{k}"] = v.detach()
        for k, v in world_type_stats.items():
            extras[f"world_type_{k}"] = v.detach()

        return ConsciousnessOutput(
            self_sem=self_sem,
            intent_sem=h_intent,
            extras=extras,)
    
    @torch.no_grad()
    def ResetHebbianMemory(self):
        for m in self.modules():
            if hasattr(m, "ResetHebbianMemory") and m is not self:
                m.ResetHebbianMemory()



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

    def BuildModel(self, useHebb: bool = True) -> "ConsciousnessExtractor":
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
            randKWorld=2,
            useHebb=useHebb,).to(self.device)
        model.train()
        return model

    def DummyBanks(self, B: int = 8, Nm: int = 10, Nw: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
        mem = torch.randn(B, Nm, self.mem_dim, device=self.device)
        world = torch.randn(B, Nw, self.world_dim, device=self.device)
        return mem, world

    def DummyBankDicts(self, B: int = 8, Nm: int = 10, Nw: int = 7) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mem, world = self.DummyBanks(B=B, Nm=Nm, Nw=Nw)
        mem_dict = {
            "gws": mem[:, :max(1, Nm // 4), :].contiguous(),
            "kv": mem.contiguous(),
            "ltm_sem": mem[:, :max(1, Nm // 2), :].contiguous(),
            "ltm_epi": mem[:, :max(1, Nm // 3), :].contiguous(),
            "sym": torch.randn(B, max(1, Nm // 3), self.sym_dim, device=self.device),}
        world_dict = {
            "vals": world.contiguous(),
            "idx": torch.randint(0, max(1, Nw), (B, Nw), device=self.device),
            "size": torch.full((B,), Nw, device=self.device, dtype=torch.long),}
        return mem_dict, world_dict

    def TestDictBankInterface(self) -> bool:
        try:
            model = self.BuildModel(useHebb=True)
            model.ResetState()
            mem_dict, world_dict = self.DummyBankDicts(B=4, Nm=9, Nw=6)

            out = model(mem_dict, world_dict)
            assert out.self_sem.shape == (4, self.self_dim), f"self_sem shape wrong: {out.self_sem.shape}"
            assert out.intent_sem.shape == (4, self.intent_dim), f"intent_sem shape wrong: {out.intent_sem.shape}"
            assert "mem_n_items" in out.extras and "world_n_items" in out.extras
            assert (out.extras["mem_n_items"] > 0).all()
            assert (out.extras["world_n_items"] > 0).all()
            assert "mem_type_gate" in out.extras and "world_type_gate" in out.extras

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
            model = self.BuildModel(useHebb=True)
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

    def TestStateFlow(self) -> bool:
        try:
            model = self.BuildModel(useHebb=True)
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
            model = self.BuildModel(useHebb=True)
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
            model = self.BuildModel(useHebb=True)
            model.ResetState()
            model.train()

            out = model(None, None)

            assert out.self_sem.shape == (1, self.self_dim), f"cold-start self_sem shape wrong: {out.self_sem.shape}"
            assert out.intent_sem.shape == (1, self.intent_dim), f"cold-start intent_sem shape wrong: {out.intent_sem.shape}"
            assert "cold_start" in out.extras, "cold-start extras missing key: cold_start"
            assert torch.all(out.extras["cold_start"] == 1.0), "cold_start flag should be all ones"
            assert out.extras["loss"] is not None, "training cold-start should provide zero loss tensor"
            assert out.extras["loss"].dim() == 0, "cold-start loss should be scalar"
            assert abs(float(out.extras["loss"].item())) < 1e-12, f"cold-start loss should be zero, got {out.extras['loss'].item()}"
            assert torch.all(out.extras["mem_n_items"] == 0.0), "cold-start mem_n_items should be zero"
            assert torch.all(out.extras["world_n_items"] == 0.0), "cold-start world_n_items should be zero"

            dev_trace, last_intent, last_sem, step = model.GetState()
            assert step == 1, f"step should be 1 after cold start forward, got {step}"
            assert dev_trace is not None and dev_trace.shape == (1, self.dev_dim), "cold-start dev_trace shape mismatch"
            assert last_intent is not None and last_intent.shape == (1, self.intent_dim), "cold-start last_intent shape mismatch"
            assert last_sem is not None and last_sem.shape == (1, self.self_dim), "cold-start last_sem shape mismatch"

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
            model = self.BuildModel(useHebb=True)
            model.ResetState()
            model.train()

            B = 3
            mem_dict: Dict[str, torch.Tensor] = {
                "gws": torch.zeros(B, 0, self.mem_dim, device=self.device),
                "kv": torch.zeros(B, 0, self.mem_dim, device=self.device),
                "ltm_sem": torch.zeros(B, 0, self.mem_dim, device=self.device),
                "ltm_epi": torch.zeros(B, 0, self.mem_dim, device=self.device),
                "sym": torch.zeros(B, 0, self.sym_dim, device=self.device),}
            world_dict: Dict[str, torch.Tensor] = {
                "vals": torch.randn(B, 1, self.world_dim, device=self.device),}

            out = model(mem_dict, world_dict)

            zeros = torch.zeros(B, 1, device=self.device)
            ones_gate = torch.ones(B, len(model.world_type_keys), device=self.device)
            assert torch.allclose(out.extras["world_ctx_blend_alpha"], zeros), "world_alpha should be zero when world tokens < world_typed_min_tokens"
            assert torch.allclose(out.extras["mem_attn_entropy"], zeros), "mem_attn_entropy should be zero when memory_bank is empty"
            assert torch.allclose(out.extras["world_attn_entropy"], zeros), "world_attn_entropy should be zero when world tokens <= 1"
            assert torch.allclose(out.extras["world_type_gate"], ones_gate), "world_type_gate should be all ones in world typed fallback branch"
            assert torch.all(out.extras["mem_n_items"] == 0.0), "mem_n_items should be zero for empty memory bank"
            assert torch.all(out.extras["world_n_items"] == 1.0), "world_n_items should equal 1 for single-token world bank"

            print("TestLowTokenFallbackBranches passed.")
            return True
        except AssertionError as e:
            print("TestLowTokenFallbackBranches failed:", e)
            return False
        except Exception as e:
            print("TestLowTokenFallbackBranches error:", e)
            return False

    def TestConsciousHebbianLinearLifecycle(self) -> bool:
        try:
            lin = ConsciousHebbianLinear(inFeatures=16, outFeatures=12, hebbAlpha=0.1, useHebb=True).to(self.device)

            x = torch.randn(5, 16, device=self.device)
            with torch.no_grad():
                n0 = lin.hebb.norm().item()
                for _ in range(3):
                    _ = lin(x)
                n1 = lin.hebb.norm().item()
                assert n1 > n0 + 1e-12, f"hebb did not grow: before={n0:.3e}, after={n1:.3e}"

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
            model = self.BuildModel(useHebb=True)
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
            model = self.BuildModel(useHebb=True)
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
            model = self.BuildModel(useHebb=True)
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
            model = self.BuildModel(useHebb=True)
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
            model = self.BuildModel(useHebb=True)
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
            model = self.BuildModel(useHebb=False)
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
            "DictBankInterface": self.TestDictBankInterface(),
            "StateFlow": self.TestStateFlow(),
            "EvalModeInferenceBranch": self.TestEvalModeInferenceBranch(),
            "ColdStartNoInput": self.TestColdStartNoInput(),
            "LowTokenFallbackBranches": self.TestLowTokenFallbackBranches(),
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
