from __future__ import annotations 
from typing import Optional, Dict, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsciousnessState(NamedTuple):
    dev_trace: torch.Tensor 
    step: torch.Tensor  


class FiLMBlock(nn.Module):
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


class ConsciousnessHyperNet(nn.Module):
    def __init__(
        self,
        ctxDim: int,
        nSelfBlocks: int,
        nIntentBlocks: int,
        selfHiddenDim: int,
        intentHiddenDim: int,
        gateDim: int = 3,
        hiddenDim: int = 512,):
        super().__init__()
        self.ctx_dim = int(ctxDim)
        self.n_self_blocks = int(nSelfBlocks)
        self.n_intent_blocks = int(nIntentBlocks)
        self.self_hidden_dim = int(selfHiddenDim)
        self.intent_hidden_dim = int(intentHiddenDim)
        self.gate_dim = int(gateDim)

        self.total_self_params = self.n_self_blocks * 2 * self.self_hidden_dim
        self.total_intent_params = self.n_intent_blocks * 2 * self.intent_hidden_dim
        self.total_gate_params = self.gate_dim

        out_dim = self.total_self_params + self.total_intent_params + self.total_gate_params

        self.mlp = nn.Sequential(
            nn.Linear(self.ctx_dim, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, out_dim),)

    def forward(self, ctxTensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        ctx = ctxTensor
        B, D = ctx.shape
        h = self.mlp(ctx)  

        cur = 0
        hs = self.total_self_params
        hi = self.total_intent_params
        hg = self.total_gate_params

        self_flat = h[:, cur:cur + hs]; cur += hs
        intent_flat = h[:, cur:cur + hi]; cur += hi
        gates = h[:, cur:cur + hg]

        self_flat = self_flat.view(B, self.n_self_blocks, 2, self.self_hidden_dim)

        gamma_self = self_flat[:, :, 0, :]
        beta_self = self_flat[:, :, 1, :]

        intent_flat = intent_flat.view(B, self.n_intent_blocks, 2, self.intent_hidden_dim)

        gamma_intent = intent_flat[:, :, 0, :]
        beta_intent = intent_flat[:, :, 1, :]

        gates = torch.sigmoid(gates)

        return {
            "gamma_self": gamma_self,
            "beta_self": beta_self,
            "gamma_intent": gamma_intent,
            "beta_intent": beta_intent,
            "gates": gates, }


class ConsciousnessOutput(NamedTuple):
    self_sem: torch.Tensor 
    intent_sem: torch.Tensor  
    gate_lang: torch.Tensor  
    gate_world: torch.Tensor 
    gate_memory: torch.Tensor 
    new_state: ConsciousnessState
    extras: Dict[str, torch.Tensor]

class ConsciousHebbianLinear(nn.Module):
    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,
        *,
        hebbAlpha: float = 0.05,
        useHebb: bool = True,):
        super().__init__()
        self.in_f = inFeatures
        self.out_f = outFeatures

        self.weight = nn.Parameter(torch.randn(self.out_f, self.in_f) * 0.02)

        self.bias = nn.Parameter(torch.zeros(self.out_f))

        self.register_buffer("hebb",torch.zeros(self.out_f, self.in_f))

        self.hebb_alpha = hebbAlpha

        self.use_hebbian: bool = useHebb

    @torch.no_grad()
    def ResetHebbianMemory(self):
        self.hebb.zero_()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = self.weight
        if self.use_hebbian:
            W_eff = W_eff + self.hebb

        y = F.linear(x, W_eff, self.bias)

        if self.use_hebbian:
            with torch.no_grad():
                if x.dim() == 1:
                    x_b = x.unsqueeze(0)
                    y_b = y.unsqueeze(0) 
                else:
                    x_b = x
                    y_b = y 

                x_n = F.normalize(x_b, dim=-1, eps=1e-6)
                y_n = F.normalize(y_b, dim=-1, eps=1e-6)

                d_hebb = torch.einsum("bo,bi->oi", y_n, x_n)

                self.hebb.mul_(1.0 - self.hebb_alpha)
                self.hebb.add_(self.hebb_alpha * d_hebb)

        return y


class ConsciousnessExtractor(nn.Module):
    def __init__(
        self,
        memItemDim: int = 1024,  
        worldItemDim: int = 512, 
        devDim: int = 512, 
        selfDim: int = 1024, 
        intentDim: int = 1024, 
        nSelfBlocks: int = 4,
        nIntentBlocks: int = 4,
        hyperHiddenDim: int = 1024,
        topKMem: int = 8,
        randKMem: int = 4,
        topKWorld: int = 8,
        randKWorld: int = 4,
        useHebb: bool = True, ):
        super().__init__()

        self.mem_item_dim = memItemDim
        self.world_item_dim = worldItemDim
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

        self.mem_score_net = nn.Sequential(
            nn.Linear(self.mem_item_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),)
        
        self.world_score_net = nn.Sequential(
            nn.Linear(self.world_item_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1),)

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

        self.hyper = ConsciousnessHyperNet(
            ctxDim=hyperHiddenDim,
            nSelfBlocks=self.n_self_blocks,
            nIntentBlocks=self.n_intent_blocks,
            selfHiddenDim=self.self_dim,
            intentHiddenDim=self.intent_dim,
            gateDim=3,
            hiddenDim=hyperHiddenDim,)

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

        in_dev = self.dev_dim + self.self_dim + self.intent_dim
        self.dev_update = nn.Sequential(
            ConsciousHebbianLinear(in_dev,self.dev_dim,useHebb=useHebb,),
            nn.LayerNorm(self.dev_dim),
            nn.Tanh(),)

        self.monitor_head = nn.Sequential(
            nn.Linear(self.self_dim + self.intent_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 3),)
        
        self.mem_query_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, self.mem_item_dim),)

        self.world_query_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, self.world_item_dim),)

    def QueryTopK(
        self,
        bank: torch.Tensor, # [B, N, D]
        query: torch.Tensor, # [B, D]
        topK: int,):

        B, N, D = bank.shape
        device = bank.device

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

        top_scores, top_idx = torch.topk(sim, k=k_top, dim=1)
        top_w = F.softmax(top_scores, dim=-1)

        top_items = torch.gather(bank, 1,top_idx.unsqueeze(-1).expand(-1, -1, D) )

        focus = torch.einsum('bk,bkd->bd', top_w, top_items) 

        return focus, top_idx, top_w


    def InitialState(self, batchSize: int, device: torch.device) -> ConsciousnessState:
        dev_trace = torch.zeros(batchSize, self.dev_dim, device=device)
        step = torch.zeros(batchSize, device=device)
        return ConsciousnessState(dev_trace=dev_trace, step=step)

    def AggregateBank(
        self,
        bankTensor: torch.Tensor, 
        scoreNet: nn.Module,
        topK: int,
        randK: int,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        bank = bankTensor
        score_net = scoreNet
        top_k = int(topK)
        rand_k = int(randK)

        if bank.dim() != 3:
            raise ValueError("AggregateBank expects bank of shape [B, N, D]")

        B, N, D = bank.shape
        device = bank.device

        if N == 0:
            summary = torch.zeros(B, 3 * D, device=device, dtype=bank.dtype)
            stats = {"score_mean": torch.zeros(B, device=device),"n_items": torch.zeros(B, device=device),}
            return summary, stats

        scores = score_net(bank).squeeze(-1)

        global_mean = bank.mean(dim=1) 

        k_top = min(top_k, N)
        if k_top > 0:
            _, top_idx = torch.topk(scores, k=k_top, dim=1)
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)
            top_items = torch.gather(bank, 1, idx_exp)
            top_mean = top_items.mean(dim=1)
        else:
            top_mean = torch.zeros(B, D, device=device, dtype=bank.dtype)
            top_idx = None

        k_rand = min(rand_k, max(0, N - k_top))
        if k_rand > 0:
            avail_mask = torch.ones(B, N, dtype=torch.bool, device=device)
            if k_top > 0:
                avail_mask.scatter_(1, top_idx, False)

            rand_noise = torch.rand(B, N, device=device)
            rand_scores = rand_noise + (~avail_mask) * (-1e6)

            _, rand_idx = torch.topk(rand_scores, k=k_rand, dim=1)
            idx_exp2 = rand_idx.unsqueeze(-1).expand(-1, -1, D)
            rand_items = torch.gather(bank, 1, idx_exp2) 
            rand_mean = rand_items.mean(dim=1) 
        else:
            rand_mean = torch.zeros(B, D, device=device, dtype=bank.dtype)

        summary = torch.cat([global_mean, top_mean, rand_mean], dim=-1)

        stats = {"score_mean": scores.mean(dim=1),  "n_items": torch.full((B,), float(N), device=device), }

        return summary, stats

    def forward(
        self,
        memoryBank: torch.Tensor,
        worldBank: torch.Tensor,
        prevState: Optional[ConsciousnessState] = None,
        *,
        detachBase: bool = True,) -> ConsciousnessOutput:

        memory_bank = memoryBank
        world_bank = worldBank
        prev_state = prevState
        detach_base = detachBase

        B, Nm, Dm = memory_bank.shape
        Bw, Nw, Dw = world_bank.shape
        if B != Bw:
            raise ValueError("The batch dimension of memory_bank and world_bank must be the same")

        device = memory_bank.device

        if prev_state is None:
            prev_state = self.InitialState(B, device=device)
        dev_trace, step = prev_state.dev_trace, prev_state.step

        mem_summary_raw, mem_stats = self.AggregateBank(memory_bank,self.mem_score_net,topK=self.top_k_mem,randK=self.rand_k_mem,)

        world_summary_raw, world_stats = self.AggregateBank(world_bank,self.world_score_net,topK=self.top_k_world,randK=self.rand_k_world,)

        mem_summary = self.mem_agg_proj(mem_summary_raw)
        world_summary = self.world_agg_proj(world_summary_raw)

        if detach_base:
            mem_ctx = mem_summary.detach()
            world_ctx = world_summary.detach()
            dev_ctx = dev_trace.detach()
        else:
            mem_ctx = mem_summary
            world_ctx = world_summary
            dev_ctx = dev_trace

        ctx_raw = torch.cat([world_ctx, mem_ctx, dev_ctx], dim=-1)
        ctx_norm = self.ctx_norm(ctx_raw)
        ctx_vec = self.ctx_proj(ctx_norm) 
        hyper_out = self.hyper(ctx_vec)

        gamma_self = hyper_out["gamma_self"]
        beta_self = hyper_out["beta_self"]
        gamma_intent = hyper_out["gamma_intent"]
        beta_intent = hyper_out["beta_intent"]
        gates = hyper_out["gates"] 

        gate_lang = gates[:, 0]
        gate_world = gates[:, 1]
        gate_memory = gates[:, 2]

        self_in_vec = torch.cat([world_ctx, mem_ctx, dev_ctx], dim=-1) 
        h_self = self.self_in(self_in_vec) 

        for i, block in enumerate(self.self_blocks):
            g = gamma_self[:, i, :]
            b = beta_self[:, i, :]
            h_self = block(h_self, gamma=g, beta=b)
        self_sem = h_self

        q_mem = self.mem_query_net(self_sem) 
        q_world = self.world_query_net(self_sem) 

        mem_focus, mem_idx, mem_w = self.QueryTopK(memory_bank, q_mem, self.top_k_mem) 

        world_focus, world_idx, world_w = self.QueryTopK(world_bank, q_world, self.top_k_world) 

        mem_delta = F.pad(mem_focus,(0, max(0, self.self_dim - mem_focus.size(-1))),)

        world_delta = F.pad(world_focus,(0, max(0, self.self_dim - world_focus.size(-1))),)

        self_sem = self_sem + 0.2 * mem_delta + 0.2 * world_delta

        intent_in_vec = torch.cat([self_sem, mem_ctx, dev_ctx], dim=-1)
        h_intent = self.intent_in(intent_in_vec)
        for i, block in enumerate(self.intent_blocks):
            g = gamma_intent[:, i, :]
            b = beta_intent[:, i, :]
            h_intent = block(h_intent, gamma=g, beta=b)
        intent_sem = h_intent

        dev_in = torch.cat([dev_ctx, self_sem, intent_sem], dim=-1) 
        dev_update = self.dev_update(dev_in) 
        new_dev = dev_trace + 0.05 * dev_update
        new_step = step + 1.0

        new_state = ConsciousnessState(dev_trace=new_dev, step=new_step)

        monitor_vec = torch.cat([self_sem, intent_sem], dim=-1) 
        monitor = self.monitor_head(monitor_vec)
        self_energy = monitor[:, 0]
        intent_energy = monitor[:, 1]
        entropy_like = monitor[:, 2]

        extras: Dict[str, torch.Tensor] = {
            "self_energy": self_energy.detach(),
            "intent_energy": intent_energy.detach(),
            "entropy_like": entropy_like.detach(),
            "gate_lang_raw": gate_lang.detach(),
            "gate_world_raw": gate_world.detach(),
            "gate_memory_raw": gate_memory.detach(),
            "dev_trace_norm": new_dev.norm(dim=-1).detach(),
            "mem_score_mean": mem_stats["score_mean"].detach(),
            "world_score_mean": world_stats["score_mean"].detach(),
            "mem_n_items": mem_stats["n_items"].detach(),
            "world_n_items": world_stats["n_items"].detach(),
            "mem_focus_norm": mem_focus.norm(dim=-1).detach(),
            "world_focus_norm": world_focus.norm(dim=-1).detach(),}

        return ConsciousnessOutput(
            self_sem=self_sem,
            intent_sem=intent_sem,
            gate_lang=gate_lang,
            gate_world=gate_world,
            gate_memory=gate_memory,
            new_state=new_state,
            extras=extras,)
    
    @torch.no_grad()
    def ResetHebbianMemory(self):
        for _, m in self.named_children():
            if hasattr(m, "ResetHebbianMemory"):
                m.ResetHebbianMemory()
