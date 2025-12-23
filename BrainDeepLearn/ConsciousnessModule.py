from __future__ import annotations 
from typing import Optional, Dict, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        ctx = ctxTensor
        B, D = ctx.shape
        h = self.mlp(ctx)  

        cur = 0
        hs = self.total_self_params
        hi = self.total_intent_params

        self_flat = h[:, cur:cur + hs]; cur += hs
        intent_flat = h[:, cur:cur + hi]; cur += hi

        self_flat = self_flat.view(B, self.n_self_blocks, 2, self.self_hidden_dim)

        gamma_self = self_flat[:, :, 0, :]
        beta_self = self_flat[:, :, 1, :]

        intent_flat = intent_flat.view(B, self.n_intent_blocks, 2, self.intent_hidden_dim)

        gamma_intent = intent_flat[:, :, 0, :]
        beta_intent = intent_flat[:, :, 1, :]

        return {
            "gamma_self": gamma_self,
            "beta_self": beta_self,
            "gamma_intent": gamma_intent,
            "beta_intent": beta_intent,}


class ConsciousnessOutput(NamedTuple):
    self_sem: torch.Tensor 
    intent_sem: torch.Tensor  
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
        
        self.mem_query_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, self.mem_item_dim),)

        self.world_query_net = nn.Sequential(
            nn.LayerNorm(self.self_dim),
            nn.Linear(self.self_dim, self.world_item_dim),)
        
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
        
        self._dev_trace = None
        self._last_self_intent = None
        self._last_sem = None
        self._step: int = 0

    @torch.no_grad()
    def ResetState(self):
        self._dev_trace = None
        self._last_self_intent = None
        self._last_sem = None
        self._step = 0

    def GetState(self):
        if self._dev_trace is None or self._last_self_intent is None or self._last_sem is None:
            return None, None, None, self._step
        return self._dev_trace.detach(), self._last_self_intent.detach(), self._last_sem.detach(), self._step

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
            stats = {
                "score_mean": torch.zeros(B, 1, device=device),
                "n_items": torch.zeros(B, 1, device=device),}
            
            return summary, stats

        scores = score_net(bank).squeeze(-1)

        global_mean = bank.mean(dim=1)

        k_top = min(top_k, N)
        if k_top > 0:
            top_scores, top_idx = torch.topk(scores, k=k_top, dim=1)

            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, D) 
            top_items = torch.gather(bank, 1, idx_exp) 

            top_w = F.softmax(top_scores, dim=-1)  
            top_mean = torch.einsum("bk,bkd->bd", top_w, top_items) 
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

        stats = {
            "score_mean": scores.mean(dim=1, keepdim=True),
            "n_items": torch.full((B,1), float(N), device=device),}

        return summary, stats

    def forward(
        self,
        memoryBank: torch.Tensor,
        worldBank: torch.Tensor,) -> ConsciousnessOutput:

        memory_bank = memoryBank
        world_bank = worldBank

        B, Nm, Dm = memory_bank.shape
        Bw, Nw, Dw = world_bank.shape
        if B != Bw:
            raise ValueError("The batch dimension of memory_bank and world_bank must be the same")

        if self._dev_trace is None or self._last_self_intent is None or self._last_sem is None:
            device = memory_bank.device
            dtype = memory_bank.dtype

            self._dev_trace = torch.zeros(B, self.dev_dim, device=device, dtype=dtype)
            dev_ctx = torch.zeros(B, self.dev_dim, device=device, dtype=dtype)
        else:
            arousal = self.arousal_net(self._dev_trace)
            alpha = self.update_base * (0.5 + arousal)
            delta = self.dev_update(torch.cat([self._dev_trace, self._last_sem, self._last_self_intent], dim=-1))

            dev_ctx = self._dev_trace + alpha * delta

        mem_summary_raw, mem_stats = self.AggregateBank(memory_bank,self.mem_score_net,topK=self.top_k_mem,randK=self.rand_k_mem,)

        world_summary_raw, world_stats = self.AggregateBank(world_bank,self.world_score_net,topK=self.top_k_world,randK=self.rand_k_world,)

        mem_ctx = self.mem_agg_proj(mem_summary_raw)
        world_ctx = self.world_agg_proj(world_summary_raw)

        ctx_raw = torch.cat([world_ctx, mem_ctx, dev_ctx], dim=-1)
        ctx_norm = self.ctx_norm(ctx_raw)
        ctx_vec = self.ctx_proj(ctx_norm) 
        hyper_out = self.hyper(ctx_vec)

        gamma_self = hyper_out["gamma_self"]
        beta_self = hyper_out["beta_self"]
        gamma_intent = hyper_out["gamma_intent"]
        beta_intent = hyper_out["beta_intent"]

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

        g_m = self.mem_gain_net(self_sem) 
        g_w = self.world_gain_net(self_sem)

        g_m = self.gain_min + (self.gain_max - self.gain_min) * g_m
        g_w = self.gain_min + (self.gain_max - self.gain_min) * g_w

        self_sem = self_sem + g_m * mem_delta + g_w * world_delta

        intent_in_vec = torch.cat([self_sem, mem_ctx, dev_ctx], dim=-1)
        h_intent = self.intent_in(intent_in_vec)
        for i, block in enumerate(self.intent_blocks):
            g = gamma_intent[:, i, :]
            b = beta_intent[:, i, :]
            h_intent = block(h_intent, gamma=g, beta=b)

        if self.training:
            ctx_hat = self.dev_to_ctx(dev_ctx) 
            ctx_align_loss = F.mse_loss(ctx_hat, ctx_vec.detach(), reduction="none").mean(dim=1, keepdim=True)
            slow_loss = (dev_ctx - self._dev_trace).pow(2).mean(dim=1, keepdim=True)

            loss = 0.1 * ctx_align_loss.mean() + 1e-3 * slow_loss.mean()
        else:
            loss = None

        self._dev_trace = dev_ctx.detach()
        self._last_self_intent = h_intent.detach()
        self._last_sem = self_sem.detach()
        self._step += 1

        extras: Dict[str, torch.Tensor] = {
            "loss": loss,
            "dev_trace_norm": dev_ctx.norm(dim=-1, keepdim=True).detach(),
            "mem_score_mean": mem_stats["score_mean"].detach(),
            "world_score_mean": world_stats["score_mean"].detach(),
            "mem_n_items": mem_stats["n_items"].detach(),
            "world_n_items": world_stats["n_items"].detach(),
            "mem_focus_norm": mem_focus.norm(dim=-1, keepdim=True).detach(),
            "world_focus_norm": world_focus.norm(dim=-1, keepdim=True).detach(),}

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

    def TestForwardShapes(self) -> bool:
        try:
            model = self.BuildModel(useHebb=True)
            model.ResetState()
            mem, world = self.DummyBanks(B=4, Nm=9, Nw=6)

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
            mem, world = self.DummyBanks(B=B, Nm=8, Nw=5)

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
            mem, world = self.DummyBanks(B=4, Nm=9, Nw=6)

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
            mem, world = self.DummyBanks(B=B, Nm=7, Nw=5)
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
            mem, world = self.DummyBanks(B=B, Nm=10, Nw=7)
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
            mem, world = self.DummyBanks(B=B, Nm=9, Nw=6)
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
            "StateFlow": self.TestStateFlow(),
            "ConsciousHebbianLinearLifecycle": self.TestConsciousHebbianLinearLifecycle(),
            "ModuleResetHebbianMemory": self.TestModuleResetHebbianMemory(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NormalTrainingConvergence": self.NormalTrainingConvergence(),
            "GradCoverageReport": self.GradCoverageReport(),
            "QueryTopKEdgeCases": self.QueryTopKEdgeCases(),}

        passed = sum(1 for v in results.values() if v)
        print(f"\nConsciousness module tests: {passed}/{len(results)} passed.")
        return results


