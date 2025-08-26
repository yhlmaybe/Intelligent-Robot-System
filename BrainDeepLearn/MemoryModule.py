from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



def StableTopk(scores: torch.Tensor, k: int):
    N = scores.size(-1)
    eps = (torch.arange(N, device=scores.device, dtype=torch.float32) * 1e-7).view(1, -1)
    while eps.dim() < scores.dim():
        eps = eps.unsqueeze(0)
    biased = scores.float() + eps
    return torch.topk(biased, k, dim=-1)

class GlobalWorkspace(nn.Module):
    def __init__(self, dim: int, slots: int = 12, defaultTtl: int = 8, recencyTemp: float = 0.07, priorityTemp: float = 1.0):
        super().__init__()
        self.dim = dim
        self.slots = slots
        self.default_ttl = defaultTtl
        self.recency_temp = recencyTemp
        self.priority_temp = priorityTemp

        self.register_buffer("keys", torch.zeros(slots, dim))
        self.register_buffer("vals", torch.zeros(slots, dim))
        self.register_buffer("priority", torch.zeros(slots))
        self.register_buffer("ttl", torch.zeros(slots, dtype=torch.long))
        self.register_buffer("last_step", torch.zeros(slots, dtype=torch.long))
        self.register_buffer("tag_id", torch.zeros(slots, dtype=torch.long))
        self.register_buffer("owner_id", torch.zeros(slots, dtype=torch.long))
        self.global_step = 0

    @torch.no_grad()
    def Reset(self):
        self.keys.zero_(); self.vals.zero_()
        self.priority.zero_(); self.ttl.zero_(); self.last_step.zero_()
        self.tag_id.zero_(); self.owner_id.zero_()
        self.global_step = 0

    @torch.no_grad()
    def StepTick(self):
        self.global_step += 1
        alive_mask = self.ttl > 0
        self.ttl[alive_mask] -= 1
        expired = self.ttl <= 0
        if expired.any():
            idx = torch.nonzero(expired, as_tuple=False).flatten()
            self.keys[idx].zero_(); self.vals[idx].zero_()
            self.priority[idx].zero_(); self.last_step[idx].zero_()
            self.tag_id[idx].zero_(); self.owner_id[idx].zero_()

    @torch.no_grad()
    def Write(self,
              key: torch.Tensor,
              val: torch.Tensor,
              *,
              priority: Union[float, torch.Tensor] = 1.0,
              ttl: Optional[int] = None,
              tagId: int = 0,
              ownerId: int = 0,
              replacePolicy: str = "soft"):
        
        device = self.keys.device
        key = F.normalize(key.detach().to(device), dim=-1)
        val = F.normalize(val.detach().to(device), dim=-1)
        ttl = int(ttl) if ttl is not None else self.default_ttl
        pr_t = torch.as_tensor(priority, dtype=self.priority.dtype, device=self.priority.device)

        empty = (self.priority <= 0)
        if empty.any():
            idx = int(torch.nonzero(empty, as_tuple=False)[0].item())
        else:
            age = (self.global_step - self.last_step).clamp(min=0).float()
            freshness = torch.exp(-age * self.recency_temp)
            eff = self.priority * self.priority_temp * freshness
            idx = int(torch.argmin(eff).item())

        self.keys[idx] = key
        self.vals[idx] = val
        if replacePolicy == "hard":
            self.priority[idx] = pr_t
        else:
            self.priority[idx] = torch.maximum(pr_t, self.priority[idx])
        self.ttl[idx] = ttl
        self.last_step[idx] = self.global_step
        self.tag_id[idx] = tagId
        self.owner_id[idx] = ownerId
        return idx

    def Attend(self,
               query: torch.Tensor,
               *,
               topk: int = 4,
               tagMask: Optional[List[int]] = None,
               returnWeights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self.keys.numel() == 0:
            out = torch.zeros(query.size(0), self.dim, device=query.device, dtype=query.dtype)
            return (out, None)

        device = query.device
        keys = self.keys
        vals = self.vals
        pr = self.priority
        ttl = self.ttl
        last = self.last_step

        alive = (ttl > 0) & (pr > 0)
        if torch.count_nonzero(alive).item() == 0:
            out = torch.zeros(query.size(0), self.dim, device=device, dtype=query.dtype)
            return (out, None)

        idx_alive = torch.nonzero(alive, as_tuple=False).flatten()
        keys_a = keys[idx_alive].to(device)
        vals_a = vals[idx_alive].to(device)
        pr_a = pr[idx_alive].to(device)
        last_a = last[idx_alive].to(device)

        if tagMask is not None:
            tag_a = self.tag_id[idx_alive]
            allowed = torch.zeros_like(tag_a, dtype=torch.bool)
            for t in tagMask:
                allowed |= (tag_a == t)
            if allowed.sum() == 0:
                out = torch.zeros(query.size(0), self.dim, device=device, dtype=query.dtype)
                return (out, None)
            idx_alive = idx_alive[allowed]
            keys_a = keys[idx_alive].to(device)
            vals_a = vals[idx_alive].to(device)
            pr_a = pr[idx_alive].to(device)
            last_a = last[idx_alive].to(device)

        sim = F.normalize(query, dim=-1) @ F.normalize(keys_a, dim=-1).t()
        age = (self.global_step - last_a).clamp(min=0).float()
        freshness = torch.exp(-age * self.recency_temp)
        bias = (pr_a * freshness).unsqueeze(0)
        sim = sim * bias

        k = max(1, min(topk, sim.size(-1)))
        top_sim, top_idx = StableTopk(sim, k)
        w = F.softmax(top_sim, dim=-1)
        v = vals_a[top_idx]
        out = torch.einsum('bk,bkd->bd', w, v)

        with torch.no_grad():
            flat_idx = idx_alive[top_idx.reshape(-1)]
            self.last_step[flat_idx] = self.global_step

        return (out, w) if returnWeights else (out, None)

    @torch.no_grad()
    def Inspect(self) -> Dict[str, torch.Tensor]:
        return {
            'keys': self.keys.clone(),
            'vals': self.vals.clone(),
            'priority': self.priority.clone(),
            'ttl': self.ttl.clone(),
            'last_step': self.last_step.clone(),
            'tag_id': self.tag_id.clone(),
            'owner_id': self.owner_id.clone(),}


class SemanticLTM(nn.Module):
    def __init__(self, dim: int, capacity: int = 4096):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.register_buffer("emb", torch.zeros(capacity, dim, dtype=torch.float16))
        self.register_buffer("prio", torch.zeros(capacity))
        self.register_buffer("touch", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("step", torch.zeros(capacity, dtype=torch.long))
        self.filled = 0
        self.global_step = 0

    @torch.no_grad()
    def StepTick(self):
        self.global_step += 1

    @torch.no_grad()
    def Store(self, vec: torch.Tensor, score: float = 1.0):
        vec = F.normalize(vec.detach(), dim=-1)
        if self.filled < self.capacity:
            i = self.filled; self.filled += 1
        else:
            age = (self.global_step - self.step[:self.filled]).clamp(min=0).float()
            freshness = torch.exp(-age * 0.01)
            eff = self.prio[:self.filled] * freshness
            i = int(torch.argmin(eff).item())
        self.emb[i] = vec.to(self.emb.dtype)
        self.prio[i] = max(float(score), float(self.prio[i].item()))
        self.touch[i] += 1
        self.step[i] = self.global_step

    def Retrieve(self, query: torch.Tensor, topk: int = 8) -> torch.Tensor:
        if self.filled == 0:
            return torch.zeros(query.size(0), self.dim, device=query.device, dtype=query.dtype)
        E = self.emb[:self.filled].to(query.device).float()
        sim = F.normalize(query, dim=-1) @ F.normalize(E, dim=-1).t()
        k = max(1, min(topk, self.filled))
        top_sim, idx = StableTopk(sim, k)
        w = F.softmax(top_sim, dim=-1)
        v = E[idx]
        out = torch.einsum('bk,bkd->bd', w, v)
        with torch.no_grad():
            flat = idx.reshape(-1)
            self.touch[flat] += 1
            self.step[flat] = self.global_step
        return out


class EpisodicLTM(nn.Module):
    def __init__(self, dim: int, capacity: int = 4096):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.register_buffer("emb", torch.zeros(capacity, dim, dtype=torch.float16))
        self.register_buffer("rew", torch.zeros(capacity))
        self.register_buffer("prio", torch.zeros(capacity))
        self.register_buffer("step", torch.zeros(capacity, dtype=torch.long))
        self.filled = 0
        self.global_step = 0

    @torch.no_grad()
    def StepTick(self):
        self.global_step += 1

    @torch.no_grad()
    def Store(self, vec: torch.Tensor, reward: float = 0.0, score: float = 1.0):
        vec = F.normalize(vec.detach(), dim=-1)
        if self.filled < self.capacity:
            i = self.filled; self.filled += 1
        else:
            age = (self.global_step - self.step[:self.filled]).clamp(min=0).float()
            freshness = torch.exp(-age * 0.01)
            eff = (self.prio[:self.filled] + 0.5 * self.rew[:self.filled]) * freshness
            i = int(torch.argmin(eff).item())
        self.emb[i] = vec.to(self.emb.dtype)
        self.rew[i] = float(reward)
        self.prio[i] = max(float(score), float(self.prio[i].item()))
        self.step[i] = self.global_step

    def Retrieve(self, query: torch.Tensor, topk: int = 8, recentBias: float = 0.05) -> torch.Tensor:
        if self.filled == 0:
            return torch.zeros(query.size(0), self.dim, device=query.device, dtype=query.dtype)
        E = self.emb[:self.filled].to(query.device).float()
        sim = F.normalize(query, dim=-1) @ F.normalize(E, dim=-1).t()
        age = (self.global_step - self.step[:self.filled]).clamp(min=0).float().to(query.device)
        freshness = torch.exp(-age * recentBias)
        sim = sim * freshness.unsqueeze(0)
        k = max(1, min(topk, self.filled))
        top_sim, idx = StableTopk(sim, k)
        w = F.softmax(top_sim, dim=-1)
        v = E[idx]
        out = torch.einsum('bk,bkd->bd', w, v)
        return out


class LongTermMemory(nn.Module):
    def __init__(self, dim: int, semCap: int = 4096, epiCap: int = 4096):
        super().__init__()
        self.semantic = SemanticLTM(dim, semCap)
        self.episodic = EpisodicLTM(dim, epiCap)

    @torch.no_grad()
    def Reset(self):
        self.semantic.emb.zero_(); self.semantic.prio.zero_(); self.semantic.touch.zero_(); self.semantic.step.zero_()
        self.semantic.filled = 0; self.semantic.global_step = 0
        self.episodic.emb.zero_(); self.episodic.prio.zero_(); self.episodic.rew.zero_(); self.episodic.step.zero_()
        self.episodic.filled = 0; self.episodic.global_step = 0

    @torch.no_grad()
    def StepTick(self):
        self.semantic.StepTick(); self.episodic.StepTick()

    def Retrieve(self, query: torch.Tensor, topkSem: int = 6, topkEpi: int = 2) -> torch.Tensor:
        sem = self.semantic.Retrieve(query, topk=topkSem)
        epi = self.episodic.Retrieve(query, topk=topkEpi)
        return F.normalize(sem + epi, dim=-1)


class MetaPlasticityController(nn.Module):
    def __init__(self, metaInDim: int = 9, hiddenDim: int = 96):
        super().__init__()
        self.rnn = nn.GRUCell(metaInDim, hiddenDim)
        self.fc_out = nn.Linear(hiddenDim, 5)
        self.h_state: Optional[torch.Tensor] = None
        self.memory_utilization = 0.0

    def UpdateMemoryUtilization(self, util: float):
        self.memory_utilization = util

    def forward(self, metaFeat: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B = metaFeat.size(0); device = metaFeat.device
        if self.h_state is None or self.h_state.size(0) != B or self.h_state.device != device:
            self.h_state = torch.zeros(B, self.rnn.hidden_size, device=device)

        prev = None if self.h_state is None else self.h_state.detach() 

        self.h_state = self.rnn(metaFeat, prev)     
        a_hat, b_hat, bias_hat, fus_hat, imp_hat = self.fc_out(self.h_state).chunk(5, dim=1)
        a = 0.5 + 0.5 * torch.sigmoid(a_hat)
        b = 0.5 + 0.5 * torch.sigmoid(b_hat)
        bias = torch.tanh(bias_hat)
        fusion_gate = torch.sigmoid(fus_hat)
        importance = torch.sigmoid(imp_hat)
        return a.squeeze(-1), b.squeeze(-1), bias.squeeze(-1), fusion_gate.squeeze(-1), importance.squeeze(-1)

    def Reset(self):
        self.h_state = None
        self.memory_utilization = 0.0


class MemoryExtractor(nn.Module):
    def __init__(
        self,
        inputDim: int = 512,
        ssmStateDim: int = 512,
        memoryDim: int = 768,
        memorySize: int = 200,
        outputDim: int = 768,
        hebbAlpha: float = 0.15,
        decayFactor: float = 0.95,
        topk: int = 8,
        tdScale: float = 5.0,
        softBeta: float = 0.2,
        useMeta: bool = True,
        useAmp: bool = True,
        svdInterval: int = 10,
        svdMin: float = 0.1,
        svdMax: float = 1.5,
        gws: Optional[GlobalWorkspace] = None,
        ltm: Optional[LongTermMemory] = None,
        gwsSlots: int = 12,
        gwsTtl: int = 8,
        consolidateEvery: int = 200,
        rehearseEvery: int = 300,
        ownerId: int = 1,) -> None:
        super().__init__()

        self.ssm_state_dim = ssmStateDim
        self.memory_dim = memoryDim
        self.output_dim = outputDim
        self.memory_size = memorySize
        self.topk = min(topk, memorySize)
        self.hebb_alpha = hebbAlpha
        self.decay = decayFactor
        self.td_scale = tdScale
        self.soft_beta = softBeta
        self.use_meta = useMeta
        self.use_amp = useAmp
        self.svd_interval = max(1, svdInterval)
        self.svd_min = svdMin
        self.svd_max = svdMax

        self.consolidate_every = consolidateEvery
        self.rehearse_every = rehearseEvery
        self.owner_id = ownerId

        A_init = torch.empty(ssmStateDim, ssmStateDim)
        nn.init.orthogonal_(A_init, gain=0.8)
        self.A_full = nn.Parameter(A_init * 0.05)
        self.B_mat = nn.Linear(inputDim, ssmStateDim, bias=False)
        self.C_mat = nn.Linear(ssmStateDim, outputDim, bias=False)
        self.D_mat = nn.Linear(inputDim, outputDim, bias=False)
        for p in (self.B_mat, self.C_mat, self.D_mat):
            nn.init.xavier_uniform_(p.weight)

        self.register_buffer("h_state", torch.zeros(1, ssmStateDim))

        self.register_buffer("fast_weights", torch.zeros(memoryDim, memoryDim))
        self.register_buffer("memory_keys", torch.zeros(memorySize, memoryDim, dtype=torch.float16))
        self.register_buffer("memory_values", torch.zeros(memorySize, memoryDim, dtype=torch.float16))
        self.register_buffer("memory_importance", torch.zeros(memorySize))
        self.register_buffer("memory_steps", torch.zeros(memorySize, dtype=torch.long))
        self.register_buffer("memory_corr", torch.zeros(memorySize))

        self.mem_ptr = 0
        self.time_step = 0
        self.memory_filled = 0
        self.memory_usage = 0.0
        self.last_compress_step = 0
        self._steps_since_svd = 0
        self.fro_norm_history: List[float] = []
        self.svd_threshold = 5.0

        self.state2mem = nn.Linear(ssmStateDim, memoryDim)
        self.state2val = nn.Linear(ssmStateDim, memoryDim)
        self.importance_net = nn.Sequential(
            nn.Linear(ssmStateDim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),)

        self.local_gate = nn.Sequential(
            nn.Linear(ssmStateDim, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),)

        self.fusion_gate_net = nn.Sequential(
            nn.Linear(memoryDim * 3, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),)

        self.meta_ctrl = MetaPlasticityController(hiddenDim=96) if useMeta else None

        self.fusion = nn.Sequential(
            nn.Linear(outputDim + memoryDim, 1024), nn.GELU(),
            nn.Linear(1024, outputDim),)

        self.norm = nn.LayerNorm(outputDim)
        self.grad_bridge = nn.Parameter(torch.tensor(0.3))

        self.gws = gws if gws is not None else GlobalWorkspace(dim=memoryDim, slots=gwsSlots, defaultTtl=gwsTtl)
        self.ltm = ltm if ltm is not None else LongTermMemory(dim=memoryDim)
        self.gws_summary = nn.Linear(ssmStateDim + outputDim + memoryDim, memoryDim)
        self.gws_gate = nn.Sequential(nn.Linear(memoryDim * 2, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

        self.ns_enable: bool = True
        self.ns_lambda: float = 0.1
        self.ns_alpha_write: float = 0.2
        self.ns_retrieve_boost: float = 0.3
        self.ns_alpha_out: float = 0.2

        self.ns_K: int = 6
        self.ns_exclusives: List[List[int]] = [[0,1],[3,4]]
        self.ns_implications: List[Tuple[int, int]] = [(0,2)]
        self.ns_atleast_one: List[List[int]] = [[0,2]]

        self.ns_head_pre = nn.Linear(self.memory_dim, self.ns_K)
        self.ns_head_post = nn.Linear(self.memory_dim, self.ns_K)
        nn.init.xavier_uniform_(self.ns_head_pre.weight); nn.init.zeros_(self.ns_head_pre.bias)
        nn.init.xavier_uniform_(self.ns_head_post.weight); nn.init.zeros_(self.ns_head_post.bias)

        self._ns_prev_P_pre: Optional[torch.Tensor] = None
        self._ns_prev_P_post: Optional[torch.Tensor] = None
        self._ns_penalty_vec: Optional[torch.Tensor] = None
        self.ns_last: Dict[str, torch.Tensor] = {}

        self._extra_losses: List[torch.Tensor] = []

    def ResetInternalLoss(self):
        self._extra_losses = []

    def AddInternalLoss(self, loss: torch.Tensor):
        if isinstance(loss, torch.Tensor):
            self._extra_losses.append(loss)

    def GetInternalLoss(self) -> torch.Tensor:
        if len(self._extra_losses) == 0:
            dev = self.h_state.device
            return torch.zeros([], device=dev)
        return torch.stack([l for l in self._extra_losses]).sum()

    def attach_loss(self, mainLoss: torch.Tensor) -> torch.Tensor:
        return mainLoss + self.GetInternalLoss()

    def backward(self, mainLoss: torch.Tensor, **kwargs):
        total = self.attach_loss(mainLoss)
        total.backward(**kwargs)

    def NsEnsurePrev(self, B: int, device: torch.device):
        if (self._ns_prev_P_pre is None) or (self._ns_prev_P_pre.size(0) != B) or (self._ns_prev_P_pre.device != device):
            self._ns_prev_P_pre = torch.zeros(B, self.ns_K, device=device)
        if (self._ns_prev_P_post is None) or (self._ns_prev_P_post.size(0) != B) or (self._ns_prev_P_post.device != device):
            self._ns_prev_P_post = torch.zeros(B, self.ns_K, device=device)

    def NsRules(self, P: torch.Tensor, P_prev: Optional[torch.Tensor]) -> torch.Tensor:
        B = P.size(0); device = P.device
        per_sample = torch.zeros(B, device=device)

        for S in self.ns_exclusives:
            if len(S) >= 2:
                ps = P[:, S]
                s = ps.sum(dim=1)
                s2 = (ps * ps).sum(dim=1)
                per_sample = per_sample + 0.5 * (s * s - s2)

        for (a, b) in self.ns_implications:
            per_sample = per_sample + torch.relu(P[:, a] - P[:, b])

        for S in self.ns_atleast_one:
            if len(S) > 0:
                anyp = P[:, S].max(dim=1).values
                per_sample = per_sample + (1.0 - anyp)

        if (P_prev is not None) and (P_prev.shape == P.shape):
            per_sample = per_sample + torch.relu(P_prev - P).mean(dim=1)
        return per_sample

    def NsPreWrite(self, val: torch.Tensor, importance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.ns_enable:
            dev = val.device
            return torch.empty(0, device=dev), torch.empty(0, device=dev), torch.zeros([], device=dev)
        B, device = val.size(0), val.device
        self.NsEnsurePrev(B, device)

        P_pre = torch.sigmoid(self.ns_head_pre(val))
        per_sample_pre = self.NsRules(P_pre, self._ns_prev_P_pre)
        self._ns_prev_P_pre = P_pre.detach()

        updated_importance = importance
        if self.ns_alpha_write > 0.0:
            damp = torch.clamp(per_sample_pre, 0, 1).view(-1, 1)
            updated_importance = importance * (1.0 - self.ns_alpha_write * damp)

        rule_loss_pre = per_sample_pre.mean()
        self._ns_penalty_vec = torch.clamp(per_sample_pre, 0, 1).view(-1, 1).detach()

        return P_pre, per_sample_pre, rule_loss_pre, updated_importance  

    def NsPostRead(self, memRecall: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.ns_enable:
            dev = memRecall.device
            B = memRecall.size(0)
            return (torch.empty(0, device=dev),
                    torch.empty(0, device=dev),
                    torch.zeros([], device=dev),
                    torch.zeros(B, 1, device=dev),
                    memRecall)
        
        B, device = memRecall.size(0), memRecall.device
        self.NsEnsurePrev(B, device)

        P_post = torch.sigmoid(self.ns_head_post(memRecall))
        per_sample_post = self.NsRules(P_post, self._ns_prev_P_post)
        self._ns_prev_P_post = P_post.detach()

        damp = torch.clamp(per_sample_post, 0, 1).view(-1, 1)
        adj_mem = memRecall
        if self.ns_alpha_out > 0.0:
            adj_mem = memRecall * (1.0 - self.ns_alpha_out * damp)  

        self._ns_penalty_vec = damp.detach()
        rule_loss_post = per_sample_post.mean()
        return P_post, per_sample_post, rule_loss_post, damp, adj_mem 

    def forward(self,
                x: torch.Tensor,
                *,
                tdError: Optional[torch.Tensor] = None,
                entropy: Optional[torch.Tensor] = None,
                reward: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None,
                reset: bool = False,
                softReset: bool = False,) -> Tuple[torch.Tensor, torch.Tensor]:

        amp_enable = self.use_amp and x.is_cuda
        dtype = torch.float16 if x.is_cuda else torch.bfloat16

        self.ResetInternalLoss()

        with torch.autocast(device_type=x.device.type, dtype=dtype, enabled=amp_enable):
            B, device = x.size(0), x.device
            if reset:
                self.ResetAll()
            elif softReset:
                self.SoftReset()
            if self.h_state.size(0) != B or self.h_state.device != device:
                self.h_state = torch.zeros(B, self.ssm_state_dim, device=device)

            self.time_step += 1
            self.gws.StepTick(); self.ltm.StepTick()

            h_new = self.h_state @ self.A_full.t() + self.B_mat(x)
            y_ssm = self.C_mat(h_new) + self.D_mat(x)

            gb = self.grad_bridge.clamp(0.1, 0.9)
            new_state = gb * h_new + (1 - gb) * h_new.detach()
            self.h_state = new_state.detach()   

            key = F.normalize(self.state2mem(h_new), dim=-1)
            val = F.normalize(self.state2val(h_new), dim=-1)
            importance = self.importance_net(h_new)
            gate_local = self.local_gate(h_new)

            neuromod = self.GetNeuromod(tdError)
            self.UpdateMemoryUtilization()
            self.AutoCompress()

            a, b, gate_bias, fusion_gate, meta_imp = self.GetMetaSignals(tdError, entropy, reward, uncertainty, B, device)
            importance = 0.7 * importance + 0.3 * meta_imp.view(-1, 1)

            if self.ns_enable:
                P_pre, per_pre, rule_pre, importance = self.NsPreWrite(val, importance) 
            else:
                rule_pre = torch.zeros([], device=h_new.device)

            self.HebbianUpdate(key, gate_local, neuromod, a, b)
            self.KvWrite(key, val, importance)

            mem_recall = self.Retrieve(key, fusion_gate)

            msg = torch.cat([h_new, y_ssm, mem_recall], dim=-1)
            ws_val = F.normalize(self.gws_summary(msg), dim=-1)
            for i in range(B):
                self.gws.Write(key[i], ws_val[i], priority=float(importance[i].item()), ttl=6, tagId=1, ownerId=self.owner_id)

            gws_read, _ = self.gws.Attend(key, topk=4)
            fuse_gate = self.gws_gate(torch.cat([gws_read, mem_recall], dim=-1))
            mem_recall = fuse_gate * gws_read + (1 - fuse_gate) * mem_recall

            if self.ns_enable:
                P_post, per_post, rule_post, damp, mem_recall = self.NsPostRead(mem_recall) 
                self.ns_last = {
                    "P_pre": P_pre.detach(),
                    "P_post": P_post.detach(),
                    "per_sample_pre": per_pre.detach(),
                    "per_sample_post": per_post.detach(),
                    "rule_loss_pre": (self.ns_lambda * rule_pre).detach(),
                    "rule_loss_post": (self.ns_lambda * rule_post).detach(),}
                self.AddInternalLoss(self.ns_lambda * (rule_pre + rule_post))
            else:
                self.ns_last = {}

            if tdError is not None:
                mem_recall = self.ApplyOutputGate(mem_recall, tdError, gate_bias)

            fused = self.fusion(torch.cat([y_ssm, mem_recall], dim=-1))
            output = self.norm(fused)

            if (self.time_step % self.consolidate_every) == 0:
                self.ConsolidateFromGWS()
            if (self.time_step % self.rehearse_every) == 0:
                self.RehearseFromLTM(batch=min(8, self.memory_filled if self.memory_filled > 0 else 1))

        return output.float(), mem_recall.float()

    def GetNeuromod(self, tdError: Optional[torch.Tensor]) -> torch.Tensor:
        if tdError is None:
            return torch.ones(1, 1, 1, device=self.h_state.device)
        return torch.tanh(tdError / self.td_scale).view(-1, 1, 1)

    def GetMetaSignals(self, tdError, entropy, reward, uncertainty, B, device):
        if self.meta_ctrl is None or tdError is None:
            one = torch.ones(B, device=device)
            zero = torch.zeros(B, device=device)
            half = torch.full((B,), 0.5, device=device)
            return (one, one, zero, half, half)
        mem_fill_ratio = self.memory_filled / self.memory_size if self.memory_size > 0 else 0.0
        mem_fill_ratio_t = torch.full((B,), float(mem_fill_ratio), device=device)
        if self.memory_filled > 0:
            imp_mean_t = self.memory_importance[:self.memory_filled].mean().to(device)
            corr_mean_t = self.memory_corr[:self.memory_filled].mean().to(device)
        else:
            imp_mean_t = torch.tensor(0.0, device=device)
            corr_mean_t = torch.tensor(0.0, device=device)
        imp_mean_t = imp_mean_t.expand(B)
        corr_mean_t = corr_mean_t.expand(B)
        meta_feat = torch.stack([
            tdError,
            tdError.abs(),
            entropy if entropy is not None else torch.full_like(tdError, -1.0),
            reward  if reward  is not None else torch.full_like(tdError, -1.0),
            torch.full_like(tdError, self.memory_usage),
            uncertainty if uncertainty is not None else torch.full_like(tdError, -1.0),
            mem_fill_ratio_t,
            imp_mean_t,
            corr_mean_t,
        ], dim=-1)
        return self.meta_ctrl(meta_feat)

    def ApplyOutputGate(self, memRecall: torch.Tensor, tdError: torch.Tensor, gateBias: torch.Tensor) -> torch.Tensor:
        gate_out = (1.0 + torch.tanh(tdError / self.td_scale + gateBias)).view(-1, 1) / 2.0
        return gate_out * memRecall

    def SoftReset(self):
        self.h_state = self.h_state * self.soft_beta
        self.fast_weights.mul_(self.soft_beta)
        if self.memory_importance is not None and self.memory_filled > 0:
            self.memory_importance[:self.memory_filled] *= self.soft_beta

    @torch.no_grad()
    def HebbianUpdate(self, key: torch.Tensor, gateLocal: torch.Tensor, neuromod: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
        a = a.view(-1, 1, 1); b = b.view(-1, 1, 1)
        outer = torch.einsum('bi,bj->bij', key, key)
        update = (neuromod * self.hebb_alpha * a * gateLocal.view(-1, 1, 1) * outer).sum(0)
        self.fast_weights.mul_(self.decay * b.mean()).add_(update)
        self._steps_since_svd += 1
        current_fro = torch.norm(self.fast_weights, p='fro').item()
        if len(self.fro_norm_history) >= 100 and self.time_step % 100 == 0:
            mean_fro = np.mean(self.fro_norm_history)
            std_fro = np.std(self.fro_norm_history)
            self.svd_threshold = mean_fro + 2 * std_fro
            self.fro_norm_history.clear()
        self.fro_norm_history.append(current_fro)
        need_svd = (self._steps_since_svd >= self.svd_interval) or (current_fro > self.svd_threshold)
        if need_svd:
            self._steps_since_svd = 0
            fw = self.fast_weights.float()
            eye = torch.eye(fw.size(0), device=fw.device, dtype=fw.dtype)
            fw = fw + 1e-6 * eye
            U, S, Vh = torch.linalg.svd(fw, full_matrices=False)
            S = torch.clamp(S, self.svd_min, self.svd_max)
            fw_proj = U @ torch.diag(S) @ Vh
            self.fast_weights.copy_(fw_proj.to(self.fast_weights.dtype))

    @torch.no_grad()
    def KvWrite(self, key: torch.Tensor, val: torch.Tensor, importance: torch.Tensor) -> None:
        n = key.size(0); device = key.device
        if self.memory_filled < self.memory_size:
            n = min(n, self.memory_size - self.memory_filled)
            start = self.memory_filled
            idx = torch.arange(start, start + n, device=device)
            self.memory_filled += n
            self.mem_ptr = (start + n) % self.memory_size
        else:
            imp_slice = self.memory_importance[:self.memory_filled]
            ksel = min(n, self.memory_filled)
            Nsel = imp_slice.numel()
            eps = (torch.arange(Nsel, device=imp_slice.device, dtype=torch.float32) * 1e-7)
            _, idx = torch.topk((-imp_slice.float() + eps), k=ksel, largest=True)
        if self.memory_filled > 0:
            mask = torch.ones(self.memory_filled, dtype=torch.bool, device=device)
            mask[idx] = False
            valid_keys = self.memory_keys[:self.memory_filled][mask].float()
            if valid_keys.numel() > 0:
                corr = torch.mm(key[:n], valid_keys.t()).mean(dim=1)
                self.memory_corr[idx] = corr.detach()
            else:
                self.memory_corr[idx].fill_(1.0)
        else:
            self.memory_corr[idx].fill_(1.0)
        self.memory_keys[idx] = key[:n].detach().half()
        self.memory_values[idx] = val[:n].detach().half()
        self.memory_importance[idx] = importance[:n].squeeze().detach()
        self.memory_steps[idx] = self.time_step

    def Retrieve(self, query: torch.Tensor, fusionGate: torch.Tensor) -> torch.Tensor:
        fast_part = query @ self.fast_weights
        if self.memory_filled == 0:
            kv_part = torch.zeros_like(fast_part)
        else:
            keys = self.memory_keys[:self.memory_filled].float()
            values = self.memory_values[:self.memory_filled].float()
            importance = self.memory_importance[:self.memory_filled]
            corr = self.memory_corr[:self.memory_filled]
            steps = self.memory_steps[:self.memory_filled]
            sim = query @ keys.t()
            sim = sim * importance.unsqueeze(0) * corr.unsqueeze(0)
            age = (self.time_step - steps).clamp(min=0).float()
            sim = sim * torch.exp(-0.05 * age).unsqueeze(0)
            k = max(1, min(self.topk, self.memory_filled))
            top_sim, top_idx = StableTopk(sim, k)
            th = top_sim.mean(dim=-1, keepdim=True) - 0.5 * top_sim.std(dim=-1, keepdim=True)
            th_min = top_sim.min(dim=-1, keepdim=True).values
            th_max = top_sim.max(dim=-1, keepdim=True).values
            th = torch.clamp(th, min=th_min, max=th_max)

            if getattr(self, "ns_enable", False) and self.ns_retrieve_boost > 0.0 and (self._ns_penalty_vec is not None):
                th = th + self.ns_retrieve_boost * self._ns_penalty_vec.to(th.device)

            mask = top_sim > th
            masked_top = top_sim.masked_fill(~mask, -1e9)
            all_false = ~mask.any(dim=-1, keepdim=True)
            top_sim_eff = torch.where(all_false, top_sim, masked_top)
            attn_weights = F.softmax(top_sim_eff.float(), dim=-1)
            vals = values[top_idx]
            kv_part = torch.einsum('bk,bkd->bd', attn_weights, vals)
        fusion_input = torch.cat([query, fast_part, kv_part], dim=-1)
        gate = self.fusion_gate_net(fusion_input)
        gate = 0.5 * gate + 0.5 * fusionGate.view(-1, 1)
        return gate * fast_part + (1 - gate) * kv_part

    @torch.no_grad()
    def UpdateMemoryUtilization(self):
        window_size = max(50, min(200, self.memory_size // 5))
        min_step = max(1, self.time_step - window_size)
        accessed = ((self.memory_steps >= min_step) & (self.memory_steps > 0) &
                    (torch.arange(self.memory_size, device=self.memory_steps.device) < self.memory_filled))
        accessed_count = accessed.sum().item()
        self.memory_usage = (min(1.0, accessed_count / self.memory_filled) if self.memory_filled > 0 else 0.0)
        if self.meta_ctrl:
            self.meta_ctrl.UpdateMemoryUtilization(self.memory_usage)

    @torch.no_grad()
    def AutoCompress(self):
        current_thresh = max(0.6, min(0.9, 0.7 + self.memory_usage * 0.2))
        if self.memory_filled < self.memory_size * current_thresh:
            return
        if self.time_step - self.last_compress_step < 100:
            return
        if self.memory_filled > 0:
            time_diff = self.time_step - self.memory_steps[:self.memory_filled]
            decay_factor = torch.exp(-0.01 * time_diff.float())
            self.memory_importance[:self.memory_filled] *= decay_factor
        importances = self.memory_importance[:self.memory_filled]
        Nimp = importances.numel()
        eps = (torch.arange(Nimp, device=importances.device, dtype=torch.float32) * 1e-7)
        _, sorted_idx = torch.sort((importances.float() + eps), descending=True)
        keep_num = min(int(self.memory_size * 0.7), self.memory_filled)
        sorted_idx = sorted_idx[:keep_num]
        self.memory_keys[:keep_num] = self.memory_keys[sorted_idx]
        self.memory_values[:keep_num] = self.memory_values[sorted_idx]
        self.memory_importance[:keep_num] = importances[sorted_idx]
        self.memory_steps[:keep_num] = self.memory_steps[sorted_idx]
        self.memory_corr[:keep_num] = self.memory_corr[sorted_idx]
        self.memory_filled = keep_num
        self.mem_ptr = keep_num % self.memory_size
        self.last_compress_step = self.time_step

    @torch.no_grad()
    def ConsolidateFromGWS(self, minPriority: float = 0.25):
        snap = self.gws.Inspect()
        keys = snap['keys']; vals = snap['vals']; pr = snap['priority']; ttl = snap['ttl']
        alive = (ttl > 0) & (pr > minPriority)
        if alive.sum() == 0:
            return
        idx = torch.nonzero(alive, as_tuple=False).flatten()
        for i in idx.tolist():
            v = vals[i].to(self.ltm.semantic.emb.device).float()
            k = keys[i].to(v.device).float()
            score = float(pr[i].item())
            self.ltm.semantic.Store(F.normalize(v, dim=-1), score=score)
            self.ltm.episodic.Store(F.normalize(k, dim=-1), reward=0.0, score=score)

    @torch.no_grad()
    def RehearseFromLTM(self, batch: int = 8):
        if self.ltm.semantic.filled == 0 and self.ltm.episodic.filled == 0:
            return
        device = self.fast_weights.device
        h = self.h_state.mean(dim=0, keepdim=True).to(device)
        q = F.normalize(self.state2mem(h), dim=-1)
        sem = self.ltm.semantic.Retrieve(q, topk=min(batch, max(1, self.ltm.semantic.filled)))
        epi = self.ltm.episodic.Retrieve(q, topk=min(max(1, batch // 2), max(1, self.ltm.episodic.filled)))
        keys = F.normalize(torch.cat([sem, epi], dim=0), dim=-1)
        if keys.numel() == 0:
            return
        Bp = keys.size(0)
        a = torch.full((Bp,), 0.55, device=device)
        b = torch.full((Bp,), 0.98, device=device)
        gate = torch.ones(Bp, 1, device=device) * 0.5
        neu = torch.ones(Bp, 1, 1, device=device) * 0.2
        self.HebbianUpdate(keys, gate, neu, a, b)

    def ResetAll(self):
        self.fast_weights.zero_()
        self.memory_keys.zero_(); self.memory_values.zero_()
        self.memory_importance.zero_(); self.memory_steps.zero_(); self.memory_corr.zero_()
        self.mem_ptr = 0; self.time_step = 0; self.memory_filled = 0
        self.memory_usage = 0.0; self.last_compress_step = 0; self._steps_since_svd = 0
        self.h_state.zero_()
        if self.meta_ctrl: self.meta_ctrl.Reset()
        self.gws.Reset(); self.ltm.Reset()

        self._ns_prev_P_pre = None
        self._ns_prev_P_post = None
        self._ns_penalty_vec = None
        self.ns_last = {}
        self.ResetInternalLoss()

    @torch.no_grad()
    def GetState(self) -> dict:
        gws_snap = self.gws.Inspect()
        sem = self.ltm.semantic
        epi = self.ltm.episodic

        return {
            "h_state": self.h_state.clone(),
            "fast_weights": self.fast_weights.clone(),
            "memory_keys": self.memory_keys.clone(),
            "memory_values": self.memory_values.clone(),
            "memory_importance": self.memory_importance.clone(),
            "memory_corr": self.memory_corr.clone(),
            "memory_steps": self.memory_steps.clone(),
            "mem_ptr": torch.tensor(self.mem_ptr),
            "time_step": torch.tensor(self.time_step),
            "memory_filled": torch.tensor(self.memory_filled),

            "meta_ctrl": (self.meta_ctrl.h_state.clone()
                          if (self.meta_ctrl and self.meta_ctrl.h_state is not None) else None),

            "_steps_since_svd": torch.tensor(self._steps_since_svd),
            "last_compress_step": torch.tensor(self.last_compress_step),
            "memory_usage": torch.tensor(self.memory_usage, dtype=torch.float32),
            "svd_threshold": torch.tensor(self.svd_threshold, dtype=torch.float32),
            "fro_norm_history": torch.tensor(self.fro_norm_history, dtype=torch.float32),

            "gws_keys": gws_snap["keys"],
            "gws_vals": gws_snap["vals"],
            "gws_priority": gws_snap["priority"],
            "gws_ttl": gws_snap["ttl"],
            "gws_last_step": gws_snap["last_step"],
            "gws_tag_id": gws_snap["tag_id"],
            "gws_owner_id": gws_snap["owner_id"],
            "gws_global_step": torch.tensor(self.gws.global_step),

            "ltm_sem_emb": sem.emb.clone(),
            "ltm_sem_prio": sem.prio.clone(),
            "ltm_sem_touch": sem.touch.clone(),
            "ltm_sem_step": sem.step.clone(),
            "ltm_sem_filled": torch.tensor(sem.filled),
            "ltm_sem_global_step": torch.tensor(sem.global_step),

            "ltm_epi_emb": epi.emb.clone(),
            "ltm_epi_prio": epi.prio.clone(),
            "ltm_epi_rew": epi.rew.clone(),
            "ltm_epi_step": epi.step.clone(),
            "ltm_epi_filled": torch.tensor(epi.filled),
            "ltm_epi_global_step": torch.tensor(epi.global_step),

            "ns_prev_P_pre": (self._ns_prev_P_pre.clone() if (self._ns_prev_P_pre is not None) else None),
            "ns_prev_P_post": (self._ns_prev_P_post.clone() if (self._ns_prev_P_post is not None) else None),

            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,}

    @torch.no_grad()
    def SetState(self, state: dict):
        if "rng_cpu" in state:
            torch.set_rng_state(state["rng_cpu"].to("cpu"))
        if "rng_cuda" in state and state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])

        self.h_state.copy_(state["h_state"].to(self.h_state.device))
        self.fast_weights.copy_(state["fast_weights"].to(self.fast_weights.device))
        self.memory_keys.copy_(state["memory_keys"].to(self.memory_keys.device))
        self.memory_values.copy_(state["memory_values"].to(self.memory_values.device))
        self.memory_importance.copy_(state["memory_importance"].to(self.memory_importance.device))
        self.memory_corr.copy_(state["memory_corr"].to(self.memory_corr.device))
        self.memory_steps.copy_(state["memory_steps"].to(self.memory_steps.device))
        self.mem_ptr = int(state["mem_ptr"].item())
        self.time_step = int(state["time_step"].item())
        self.memory_filled = int(state["memory_filled"].item())

        if self.meta_ctrl and state["meta_ctrl"] is not None:
            hs = state["meta_ctrl"].to(self.h_state.device)
            if (self.meta_ctrl.h_state is None) or (self.meta_ctrl.h_state.shape != hs.shape):
                self.meta_ctrl.h_state = torch.zeros_like(hs)
            self.meta_ctrl.h_state.copy_(hs)

        if "_steps_since_svd" in state:
            self._steps_since_svd = int(state["_steps_since_svd"].item())
        if "last_compress_step" in state:
            self.last_compress_step = int(state["last_compress_step"].item())
        if "memory_usage" in state:
            self.memory_usage = float(state["memory_usage"].item())
        if "svd_threshold" in state:
            self.svd_threshold = float(state["svd_threshold"].item())
        if "fro_norm_history" in state:
            self.fro_norm_history = state["fro_norm_history"].flatten().tolist()

        if "gws_keys" in state:
            self.gws.keys.copy_(state["gws_keys"].to(self.gws.keys.device))
            self.gws.vals.copy_(state["gws_vals"].to(self.gws.vals.device))
            self.gws.priority.copy_(state["gws_priority"].to(self.gws.priority.device))
            self.gws.ttl.copy_(state["gws_ttl"].to(self.gws.ttl.device))
            self.gws.last_step.copy_(state["gws_last_step"].to(self.gws.last_step.device))
            self.gws.tag_id.copy_(state["gws_tag_id"].to(self.gws.tag_id.device))
            self.gws.owner_id.copy_(state["gws_owner_id"].to(self.gws.owner_id.device))
            self.gws.global_step = int(state["gws_global_step"].item())

        if "ltm_sem_emb" in state:
            sem = self.ltm.semantic
            sem.emb.copy_(state["ltm_sem_emb"].to(sem.emb.device))
            sem.prio.copy_(state["ltm_sem_prio"].to(sem.prio.device))
            sem.touch.copy_(state["ltm_sem_touch"].to(sem.touch.device))
            sem.step.copy_(state["ltm_sem_step"].to(sem.step.device))
            sem.filled = int(state["ltm_sem_filled"].item())
            sem.global_step = int(state["ltm_sem_global_step"].item())

        if "ltm_epi_emb" in state:
            epi = self.ltm.episodic
            epi.emb.copy_(state["ltm_epi_emb"].to(epi.emb.device))
            epi.prio.copy_(state["ltm_epi_prio"].to(epi.prio.device))
            epi.rew.copy_(state["ltm_epi_rew"].to(epi.rew.device))
            epi.step.copy_(state["ltm_epi_step"].to(epi.step.device))
            epi.filled = int(state["ltm_epi_filled"].item())
            epi.global_step = int(state["ltm_epi_global_step"].item())

        if "ns_prev_P_pre" in state:
            self._ns_prev_P_pre = (state["ns_prev_P_pre"].to(self.h_state.device)
                                    if state["ns_prev_P_pre"] is not None else None)
        if "ns_prev_P_post" in state:
            self._ns_prev_P_post = (state["ns_prev_P_post"].to(self.h_state.device)
                                    if state["ns_prev_P_post"] is not None else None)

    def Step(self, x: torch.Tensor, state: Optional[dict] = None, tdError=None, entropy=None, reward=None, uncertainty=None):
        if state is not None:
            orig = self.GetState()
            self.SetState(state)
        out, _ = self.forward(x, tdError=tdError, entropy=entropy, reward=reward, uncertainty=uncertainty, reset=False, softReset=False)
        new_state = self.GetState() if state is not None else None
        if state is not None:
            self.SetState(orig)
        return out, new_state

    @torch.no_grad()
    def Reason(self, goal: Optional[torch.Tensor] = None, steps: int = 3) -> torch.Tensor:
        device = self.h_state.device
        h = self.h_state.mean(dim=0, keepdim=True)
        q = F.normalize(self.state2mem(h), dim=-1)
        if goal is not None:
            goal = F.normalize(goal.to(device).view(1, -1), dim=-1)
            q = F.normalize(q + goal, dim=-1)
        hyp = torch.zeros(1, self.memory_dim, device=device)
        for _ in range(max(1, steps)):
            gws_r, _ = self.gws.Attend(q, topk=4)
            ltm_r = self.ltm.Retrieve(q, topkSem=6, topkEpi=2)
            hyp = F.normalize(gws_r + ltm_r + q, dim=-1)
            self.gws.Write(hyp[0], hyp[0], priority=0.5, ttl=4, tagId=2, ownerId=self.owner_id)
            q = hyp
        return hyp



class TestMemoryMTool:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def TestGlobalWorkspace(self):
        try:
            dim, slots = 16, 4
            gws = GlobalWorkspace(dim=dim, slots=slots, defaultTtl=2).to(self.device)
            gws.Reset()
            gws.StepTick()

            e0 = torch.zeros(dim, device=self.device); e0[0] = 1
            e1 = torch.zeros(dim, device=self.device); e1[1] = 1
            e2 = torch.zeros(dim, device=self.device); e2[2] = 1
            val = torch.randn(dim, device=self.device)

            gws.Write(e0, val, priority=0.9, ttl=2, tagId=1, ownerId=1)
            gws.Write(e1, val, priority=0.8, ttl=2, tagId=1, ownerId=1)
            gws.Write(e2, val, priority=0.7, ttl=1, tagId=2, ownerId=1)

            q = (e0 + e1).unsqueeze(0)
            out, w = gws.Attend(q, topk=2, returnWeights=True)
            assert out.shape == (1, dim), f"GWS attend shape mismatch: {out.shape}"
            assert w is not None and w.shape == (1, 2), f"GWS weights shape mismatch: {None if w is None else w.shape}"

            gws.StepTick()
            out2, _ = gws.Attend(q, topk=3)
            assert out2.shape == (1, dim), "GWS attend after TTL shape mismatch"

            print("GlobalWorkspace test passed.")
            return True
        except AssertionError as e:
            print(f"GlobalWorkspace test failed: {e}")
            return False
        except Exception as e:
            print(f"GlobalWorkspace test error: {e}")
            return False

    def TestLongTermMemory(self):
        try:
            dim = 32
            ltm = LongTermMemory(dim=dim, semCap=32, epiCap=32).to(self.device)
            ltm.Reset()

            vec = torch.randn(1, dim, device=self.device)
            ltm.semantic.Store(vec[0], score=0.9)
            ltm.episodic.Store(vec[0], reward=1.0, score=0.8)
            ltm.StepTick()

            q = vec
            fused = ltm.Retrieve(q, topkSem=4, topkEpi=2)
            assert fused.shape == (1, dim), f"LTM retrieve shape mismatch: {fused.shape}"
            assert torch.linalg.norm(fused).item() > 0, "LTM retrieve returned near-zero vector"

            print("LongTermMemory test passed.")
            return True
        except AssertionError as e:
            print(f"LongTermMemory test failed: {e}")
            return False
        except Exception as e:
            print(f"LongTermMemory test error: {e}")
            return False

    def TestMemoryExtractorForward(self):
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32, outputDim=96,useMeta=True, useAmp=False, gwsSlots=8, gwsTtl=6, consolidateEvery=50, rehearseEvery=60)
            mem = MemoryExtractor(**cfg).to(self.device)
            B = 4
            x = torch.randn(B, cfg["inputDim"], device=self.device)
            out, memrec = mem(x)

            assert out.shape == (B, cfg["outputDim"]), f"MemoryExtractor out shape mismatch: {out.shape}"
            assert memrec.shape == (B, cfg["memoryDim"]), f"MemoryExtractor mem_recall shape mismatch: {memrec.shape}"
            print("MemoryExtractor forward test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor forward test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor forward test error: {e}")
            return False

    def TestStateSaveRestore(self):
        try:
            cfg = dict(inputDim=48, ssmStateDim=48, memoryDim=64, memorySize=24, outputDim=64,useMeta=False, useAmp=False, gwsSlots=8, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)
            state0 = mem.GetState()
            torch.manual_seed(123)
            x = torch.randn(3, cfg["inputDim"], device=self.device)
            out1, _ = mem(x)
            _ = mem(x)
            mem.SetState(state0)
            out2, _ = mem(x)
            assert torch.allclose(out1, out2, atol=1e-5), "State restore did not reproduce identical output from the same pre-forward state."
            print("MemoryExtractor state save/restore test passed.")
            return True

        except AssertionError as e:
            print(f"MemoryExtractor state test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor state test error: {e}")
            return False

    def TestReason(self):
        try:
            cfg = dict(inputDim=32, ssmStateDim=32, memoryDim=48, memorySize=32, outputDim=48,
                       useMeta=False, useAmp=False, gwsSlots=10, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)
            for _ in range(5):
                x = torch.randn(2, cfg["inputDim"], device=self.device)
                mem(x)

            goal = torch.randn(cfg["memoryDim"], device=self.device)
            hyp = mem.Reason(goal=goal, steps=3)
            assert hyp.shape == (1, cfg["memoryDim"]), f"Reason output shape mismatch: {hyp.shape}"
            assert torch.linalg.norm(hyp).item() > 0, "Reason produced near-zero vector"

            snap = mem.gws.Inspect()
            assert torch.count_nonzero(snap["priority"] > 0).item() > 0, "GWS appears empty after Reason"

            print("MemoryExtractor Reason test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor Reason test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor Reason test error: {e}")
            return False

    def TestResetAndSoftReset(self):
        try:
            cfg = dict(inputDim=32, ssmStateDim=32, memoryDim=48, memorySize=32, outputDim=48,
                       useMeta=False, useAmp=False, gwsSlots=8, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)
            x = torch.randn(4, cfg["inputDim"], device=self.device)
            mem(x)

            fw_before = mem.fast_weights.detach().clone()
            imp_before = mem.memory_importance.detach().clone()
            mem.SoftReset()
            fw_after = mem.fast_weights.detach().clone()
            imp_after = mem.memory_importance.detach().clone()

            assert torch.linalg.norm(fw_after) <= torch.linalg.norm(fw_before) + 1e-6, "SoftReset did not reduce fast_weights norm"
            if mem.memory_filled > 0:
                assert torch.mean(imp_after[:mem.memory_filled]) <= torch.mean(imp_before[:mem.memory_filled]) + 1e-6, "SoftReset did not reduce memory importance"

            mem.ResetAll()
            assert mem.memory_filled == 0 and mem.time_step == 0, "ResetAll did not clear counters"
            gws_snap = mem.gws.Inspect()
            assert torch.count_nonzero(gws_snap["priority"]).item() == 0, "ResetAll did not clear GWS priorities"
            assert mem.ltm.semantic.filled == 0 and mem.ltm.episodic.filled == 0, "ResetAll did not clear LTM"

            print("MemoryExtractor Reset/SoftReset test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor Reset/SoftReset test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor Reset/SoftReset test error: {e}")
            return False

    def RunAll(self):
        results = {
            "GlobalWorkspace": self.TestGlobalWorkspace(),
            "LongTermMemory": self.TestLongTermMemory(),
            "MemoryExtractorForward": self.TestMemoryExtractorForward(),
            "StateSaveRestore": self.TestStateSaveRestore(),
            "Reason": self.TestReason(),
            "ResetAndSoftReset": self.TestResetAndSoftReset(),}
        
        passed = sum(1 for v in results.values() if v)
        print(f"\nMemory module tests: {passed}/{len(results)} passed.")
        return results
    

    @torch.no_grad()
    def NumericalStabilityProbe(
        self,
        mem, *,
        steps: int = 200,
        batch: int = 4,
        eps: float = 1e-6,
        perturb_each_step: bool = True,
        seed: int = 123,
        device: torch.device | None = None,
        print_every: int = 25,):

        device = device or (next(mem.parameters()).device if any(p.is_cuda for p in mem.parameters()) else torch.device("cpu"))
        rng = torch.Generator(device=device).manual_seed(seed)

        warmup = 10
        in_dim = mem.B_mat.in_features
        for _ in range(warmup):
            xw = torch.randn(batch, in_dim, generator=rng, device=device)
            mem(xw)

        mem2 = copy.deepcopy(mem)

        cos_hist = []
        for t in range(steps):
            x = torch.randn(batch, in_dim, generator=rng, device=device)
            if perturb_each_step:
                x2 = x + eps * self.RandnLikeGen(x, generator=rng)
            else:
                x2 = x if t > 0 else (x + eps * self.RandnLikeGen(x, generator=rng))

            y1, _ = mem(x)
            y2, _ = mem2(x2)

            c = F.cosine_similarity(y1, y2, dim=-1).mean().item()
            cos_hist.append(c)

            if (t + 1) % print_every == 0 or t == 0:
                print(f"[probe] step {t+1:4d}/{steps}, cosine={c:.6f}")

        cos_np = np.array(cos_hist, dtype=np.float64)

        print(f"steps={steps}, batch={batch}, eps={eps:g}, perturb_each_step={perturb_each_step}")
        print(f"cos(mean)={cos_np.mean():.6f}, cos(std)={cos_np.std():.6f}, " f"cos(min)={cos_np.min():.6f}, cos(max)={cos_np.max():.6f}")
        print(f"cos(start)={cos_np[0]:.6f}, cos(last)={cos_np[-1]:.6f}, drop={cos_np[0]-cos_np[-1]:.6f}")

        return list(cos_hist)
    
    def RandnLikeGen(self, x: torch.Tensor, generator: torch.Generator | None = None):
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)


    def TestNumericalStability(self):
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=48, outputDim=96,useMeta=False, useAmp=False, gwsSlots=8, gwsTtl=6, consolidateEvery=1000, rehearseEvery=1000)
            mem = MemoryExtractor(**cfg).to(self.device)

            cos_hist = self.NumericalStabilityProbe(
                mem,
                steps=200,
                batch=4,
                eps=1e-6,
                perturb_each_step=True,
                seed=123,
                device=self.device,
                print_every=25,)

            ok = (sum(c < 0.95 for c in cos_hist[:30]) == 0)  
            if ok:
                print("MemoryExtractor numerical stability test passed.")
                return True
            else:
                print("MemoryExtractor numerical stability test borderline (early divergence).")
                return False
        except Exception as e:
            print(f"MemoryExtractor numerical stability test error: {e}")
            return False


    def AttachAllInternalLosses(self, rootModule: torch.nn.Module, baseLoss: torch.Tensor):
        extras = []
        for _, m in rootModule.named_modules():
            getter = getattr(m, "GetInternalLoss", None) or getattr(m, "get_internal_loss", None)
            if callable(getter):
                v = getter()
                if isinstance(v, torch.Tensor):
                    extras.append(v.to(device=baseLoss.device, dtype=baseLoss.dtype))
        return baseLoss + (torch.stack(extras).sum() if extras else baseLoss.new_zeros([]))


    def TrainStepSmoke(self):
        cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32, outputDim=96, useMeta=True, useAmp=False, gwsSlots=8, gwsTtl=6, consolidateEvery=10_000, rehearseEvery=10_000)
        mem = MemoryExtractor(**cfg).to(self.device)
        mem.train()
        opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

        B = 8
        x = torch.randn(B, cfg["inputDim"], device=self.device)
        target = torch.randn(B, cfg["outputDim"], device=self.device)

        out, _ = mem(x)
        base = F.mse_loss(out, target)

        total = self.AttachAllInternalLosses(mem, base)
        opt.zero_grad()
        total.backward()

        nesy_grad_ok = any(
            (("ns_head_pre" in n or "ns_head_post" in n) and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
            for n, p in mem.named_parameters())
        assert nesy_grad_ok, "NeSy heads did not receive gradients."

        for n, p in mem.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

        opt.step()
        print("TrainStepSmoke passed.")
        return True
    
    def TrainNeSyOnlySanity(self, steps: int = 30):
        cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32, outputDim=96, useMeta=False, useAmp=False, gwsSlots=8, gwsTtl=6)
        mem = MemoryExtractor(**cfg).to(self.device)
        mem.ns_enable = True
        mem.ns_lambda = 0.5 
        mem.train()
        opt = torch.optim.Adam([p for p in mem.parameters() if p.requires_grad], lr=1e-3)

        in_dim = cfg["inputDim"]
        last = None
        for t in range(steps):
            x = torch.randn(8, in_dim, device=self.device)
            out, _ = mem(x) 
            base = torch.zeros([], device=self.device)  
            total = self.AttachAllInternalLosses(mem, base)

            opt.zero_grad()
            total.backward()
            opt.step()

            cur = float(total.detach().item())
            if last is not None:
                pass
            last = cur

        changed = False
        for n, p in mem.named_parameters():
            if ("ns_head_pre" in n or "ns_head_post" in n) and p.grad is not None:
                changed = True
                break
        assert changed, "NeSy heads were not updated under internal loss only."
        print("TrainNeSyOnlySanity passed.")
        return True
    


    @torch.no_grad()
    def PrimeMemoryWithConflict(self, mem: MemoryExtractor, rounds: int = 4):
        in_dim = mem.B_mat.in_features
        for _ in range(rounds):
            x = torch.randn(8, in_dim, device=self.device)
            mem(x)

    def TestNeSyRetrievalEffect(self):
        cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=64, outputDim=96, useMeta=False, useAmp=False, gwsSlots=8, gwsTtl=6)
        mem = MemoryExtractor(**cfg).to(self.device)
        mem.eval()
        mem.ns_enable = False
        self.PrimeMemoryWithConflict(mem, rounds=6)

        x = torch.randn(8, cfg["inputDim"], device=self.device)
        _, rec_off = mem(x) 

        mem.ns_enable = True
        mem.ns_retrieve_boost = 0.5
        mem.ns_alpha_out = 0.4

        _, rec_on = mem(x)  

        delta = (rec_on - rec_off).norm(dim=1).mean().item()
        cos = F.cosine_similarity(rec_on, rec_off, dim=1).mean().item()
        assert (delta > 1e-3) or (cos < 0.99), f"NeSy had negligible effect on retrieval (delta={delta:.3e}, cos={cos:.4f})"
        print("TestNeSyRetrievalEffect passed.")
        return True
    
    def CheckAttachCollector(self):
        mem = MemoryExtractor(inputDim=32, ssmStateDim=32, memoryDim=48, memorySize=16, outputDim=48, useMeta=False, useAmp=False).to(self.device)
        mem.train()
        x = torch.randn(4, 32, device=self.device)
        out, _ = mem(x)
        base = F.mse_loss(out, torch.zeros_like(out))

        manual = mem.GetInternalLoss()
        total = self.AttachAllInternalLosses(mem, base)
        auto_extra = total - base

        diff = float((manual - auto_extra).abs().item())
        assert diff < 1e-7, f"Collector mismatch: {diff}"
        print("CheckAttachCollector passed.")
        return True
    

    def NoNanAfterManySteps(self, steps: int = 50):
        cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=64, outputDim=96, useMeta=True, useAmp=False, gwsSlots=8, gwsTtl=6)
        mem = MemoryExtractor(**cfg).to(self.device)
        mem.train()
        opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

        for t in range(steps):
            x = torch.randn(8, cfg["inputDim"], device=self.device)
            y = torch.randn(8, cfg["outputDim"], device=self.device)
            out, _ = mem(x)
            base = F.mse_loss(out, y)
            total = self.AttachAllInternalLosses(mem, base)

            opt.zero_grad()
            total.backward()
            for n, p in mem.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"
            opt.step()
        print("NoNanAfterManySteps passed.")
        return True
    
    def TestTraining(self):
        ok = True
        ok &= self.TrainStepSmoke()
        ok &= self.TrainNeSyOnlySanity()
        ok &= self.TestNeSyRetrievalEffect()
        ok &= self.CheckAttachCollector()
        ok &= self.NoNanAfterManySteps()
        print(f"\nTraining-like tests: {'OK' if ok else 'FAIL'}")
        return ok