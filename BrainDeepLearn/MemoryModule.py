from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
from FunctionTools import AGICoreModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import inspect



def StableTopk(scores: torch.Tensor, k: int, epsMax: float = 1e-6, preferLowIndex: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    N = int(scores.size(-1))

    if scores.dtype in (torch.float32, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = scores.dtype if scores.dtype.is_floating_point else torch.float32

    scores_f = scores.to(dtype=compute_dtype)

    base = torch.arange(N, device=scores.device, dtype=compute_dtype)
    denom = max(1, N - 1)
    eps = (base / denom) * epsMax

    view_shape = [1] * (scores.dim() - 1) + [N]
    eps = eps.view(*view_shape)

    biased = scores_f - eps if preferLowIndex else scores_f + eps

    _, indices = torch.topk(biased, k, dim=-1)
    values = torch.gather(scores, -1, indices)
    return values, indices

class MemoryType:
    SRC_REAL: int = 0 
    SRC_IMAGINE: int = 1 
    SRC_MIXED: int = 2

class SoftSymbolicRules(AGICoreModule):
    def __init__(
        self,
        k: int,
        gExcl: int = 5,
        gOr: int = 5,
        impScale: float = 1.0,
        binarize: float = 1e-3,
        l1Imp: float = 1e-4,
        massExcl: float = 3.0,
        massOr: float = 3.0,
        massImpRow: float = 1.0,
        massWeight: float = 1e-2,
        initStd: float = 0.5,
        seedDisjoint: bool = True,
        temporalL2: float = 0.1,
        temporalMonotonic: bool = False,):
        super().__init__()
        self.K = int(k)
        self.G_excl = int(gExcl)
        self.G_or = int(gOr)

        self.imp_scale = float(impScale)

        self.binarize = float(binarize)
        self.l1_imp = float(l1Imp)

        self.mass_excl = float(massExcl)
        self.mass_or = float(massOr)
        self.mass_imp_row = float(massImpRow)
        self.mass_weight = float(massWeight)

        self.temporal_l2 = float(temporalL2)
        self.temporal_monotonic = bool(temporalMonotonic)

        self.excl_logits = nn.Parameter(torch.empty(self.G_excl, self.K)) if self.G_excl > 0 else None
        self.or_logits = nn.Parameter(torch.empty(self.G_or, self.K)) if self.G_or > 0 else None
        self.imp_logits = nn.Parameter(torch.full((self.K, self.K), -4.0))

        self.register_buffer("no_self_mask", ~torch.eye(self.K, dtype=torch.bool), persistent=True)

        nn.init.normal_(self.excl_logits, mean=0.0, std=float(initStd))
        nn.init.normal_(self.or_logits, mean=0.0, std=float(initStd))

        if seedDisjoint:
            self.SeedDisjointInit()

    @torch.no_grad()
    def SeedDisjointInit(self):
        if (self.K <= 0):
            return

        perm = torch.randperm(self.K, device=self.device)
        cursor = 0

        if self.excl_logits is not None and self.G_excl > 0:
            self.excl_logits.fill_(-2.0)
            m = max(1, int(round(self.mass_excl)))
            for g in range(self.G_excl):
                idx = perm[cursor:cursor + m]
                if idx.numel() < m:
                    idx = torch.cat([idx, perm[:(m - idx.numel())]], dim=0)
                self.excl_logits[g, idx] = 2.0
                cursor = (cursor + m) % self.K

        if self.or_logits is not None and self.G_or > 0:
            self.or_logits.fill_(-2.0)
            m = max(1, int(round(self.mass_or)))
            for g in range(self.G_or):
                idx = perm[cursor:cursor + m]
                if idx.numel() < m:
                    idx = torch.cat([idx, perm[:(m - idx.numel())]], dim=0)
                self.or_logits[g, idx] = 2.0
                cursor = (cursor + m) % self.K

    def Weights(self):
        W_excl = torch.sigmoid(self.excl_logits) if self.excl_logits is not None else None #[G_excl, K]
        W_or = torch.sigmoid(self.or_logits) if self.or_logits is not None else None #[G_excl, K]

        A_imp = torch.sigmoid(self.imp_logits) #[K, K]
        A_imp = A_imp.masked_fill(~self.no_self_mask, 0.0) 

        return W_excl, W_or, A_imp

    def forward(self, p: torch.Tensor, pPrev: Optional[torch.Tensor] = None):
        p_f = p.float() #[B, K]
        B, K = int(p_f.size(0)), int(p_f.size(1))

        total_penalty = p_f.new_zeros(B)
        aux_reg = p_f.new_zeros(())

        W_excl, W_or, A_imp = self.Weights()#[G_excl, K] , [G_or, K] , [K, K]

        if W_excl is not None and W_excl.numel() > 0:
            s = torch.matmul(p_f, W_excl.t()) #[B, G_excl]
            s2 = torch.matmul(p_f.pow(2), W_excl.pow(2).t())
            excl_pen = 0.5 * (s.pow(2) - s2) 
            total_penalty = total_penalty + excl_pen.mean(dim=1) #[B]

        if W_or is not None and W_or.numel() > 0:
            eps = p_f.new_tensor(1e-6)
            z = (p_f.unsqueeze(1) * W_or.unsqueeze(0)).clamp(0.0, 1.0 - eps) #[B, G_or, K]
            prob_not_sat = torch.exp(torch.log1p(-z).sum(dim=-1)) 
            total_penalty = total_penalty + prob_not_sat.mean(dim=1)

        if self.imp_scale > 0.0:
            violation = F.relu(p_f.unsqueeze(2) - p_f.unsqueeze(1)) #[B, K, K]
            weighted = violation * A_imp.unsqueeze(0) # [B, K, K]

            denom = max(1, K * (K - 1))
            imp_pen = weighted.sum(dim=(1, 2)) / float(denom)  #[B]
            total_penalty = total_penalty + self.imp_scale * imp_pen

        if pPrev is not None :
            pPrev_f = pPrev #[B, K]
            if self.temporal_monotonic:
                total_penalty = total_penalty + self.temporal_l2 * torch.relu(pPrev_f - p_f).mean(dim=1)
            else:
                total_penalty = total_penalty + self.temporal_l2 * (p_f - pPrev_f).pow(2).mean(dim=1)

        if self.training:
            def CrispPenalty(W: torch.Tensor):
                return (W * (1.0 - W)).mean()

            if self.binarize > 0:
                if W_excl is not None:
                    aux_reg = aux_reg + self.binarize * CrispPenalty(W_excl)
                if W_or is not None:
                    aux_reg = aux_reg + self.binarize * CrispPenalty(W_or)
                aux_reg = aux_reg + self.binarize * CrispPenalty(A_imp)

            if self.mass_weight > 0:
                if W_excl is not None:
                    aux_reg = aux_reg + self.mass_weight * (W_excl.sum(dim=1) - self.mass_excl).pow(2).mean()
                if W_or is not None:
                    aux_reg = aux_reg + self.mass_weight * (W_or.sum(dim=1) - self.mass_or).pow(2).mean()

                row_sum = A_imp.sum(dim=1) 
                aux_reg = aux_reg + self.mass_weight * (row_sum - self.mass_imp_row).pow(2).mean()

            if self.l1_imp > 0:
                aux_reg = aux_reg + self.l1_imp * A_imp.mean()

        return total_penalty, aux_reg



class SymbolicCoder(AGICoreModule):
    def __init__(self, inDim: int, k: int, hidden: int = 1024, experts: int = 8, dropout: float = 0.1):
        super().__init__()
        self.K = k
        self.dropout_val = dropout

        self.gate = nn.Linear(inDim, experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, k)) for _ in range(experts)])
        
        self.proto = nn.Parameter(torch.randn(k, inDim) * 0.02)

        self.proto_scale_log = nn.Parameter(torch.tensor(2.3)) 

        self.proto_mix_log = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):  # x: [B, inDim]
        gate_logits = self.gate(x) #[B, experts]
        weights = F.softmax(gate_logits, dim=-1) #[B, experts]

        expert_outputs = [e(x) for e in self.experts] #list experts * [B, K]
        
        stacked_experts = torch.stack(expert_outputs, dim=1) #[B, experts, k]
        
        moe_logits = (weights.unsqueeze(-1) * stacked_experts).sum(dim=1) #[B, k]

        x_norm = F.normalize(x, dim=-1) #[B, inDim]
        proto_norm = F.normalize(self.proto, dim=-1) #[k, inDim]
        
        cosine_sim = x_norm @ proto_norm.t() #[B, k]
        
        scale = torch.exp(self.proto_scale_log)
        proto_logits = cosine_sim * scale

        alpha = torch.sigmoid(self.proto_mix_log)
        total_logits = moe_logits + alpha * proto_logits

        P = torch.sigmoid(total_logits) #[B, k]

        return P #[B, k]


class QueryToSymbol(AGICoreModule):
    def __init__(self, inDim: int, k: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(inDim),
            nn.Linear(inDim, hidden), nn.SiLU(),
            nn.Linear(hidden, k))

    def forward(self, q: torch.Tensor): #q: [B, D]
        return torch.sigmoid(self.net(q)) #[B, K]


class SymbolicMemory(AGICoreModule):
    def __init__(self, k: int, capacity: int = 16384):
        super().__init__()
        self.K = int(k)
        self.capacity = int(capacity)

        self.register_buffer("P_keys", torch.zeros(self.capacity, self.K)) 
        self.register_buffer("P_vals", torch.zeros(self.capacity, self.K))
        self.register_buffer("prio", torch.zeros(self.capacity))
        self.register_buffer("step", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("touch", torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer("source", torch.zeros(self.capacity, dtype=torch.int8))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def StepTick(self):
        self.global_step.add_(1)

    @torch.no_grad()
    def Store(self, key: torch.Tensor, value: torch.Tensor, score: float = 1.0, source: int = MemoryType.SRC_REAL):
        filled = int(self.filled.item())

        if filled < self.capacity:
            i = filled
            self.filled.add_(1)
        else:
            age = (self.global_step - self.step[:filled]).clamp(min=0)
            eff = self.prio[:filled] * torch.exp(-0.01 * age) 
            src = self.source[:filled]
            is_imag = (src == MemoryType.SRC_IMAGINE)
            eff = eff * torch.where(is_imag, eff.new_tensor(0.6), eff.new_tensor(1.0))
            i = int(torch.argmin(eff).item())

        self.P_keys[i] = key
        self.P_vals[i] = value
        self.prio[i] = float(score)
        self.step[i] = self.global_step
        self.touch[i] = 1
        self.source[i] = int(source)

    def Retrieve(
        self,
        qSym: torch.Tensor, #[B, K]
        topK: int = 8,
        recentBias: float = 0.05,):
        filled = int(self.filled.item())
        if filled <= 0:
            out = torch.zeros(qSym.size(0), self.K, device=self.device, dtype=qSym.dtype)
            return out

        stored_keys = self.P_keys[:filled]  # [filled, K]
        stored_vals = self.P_vals[:filled]  # [filled, K]

        sim = qSym @ stored_keys.t()

        age = (self.global_step - self.step[:filled]).clamp(min=0)
        prio = self.prio[:filled]

        decay = torch.exp(-float(recentBias) * age).unsqueeze(0)
        sim = sim * decay * prio.unsqueeze(0)

        k = max(1, min(int(topK), filled))
        top_sim, idx = StableTopk(sim, k)
        w = F.softmax(top_sim, dim=-1) 

        selected_vals = stored_vals[idx]
        out = torch.einsum("bk,bkd->bd", w, selected_vals)  

        with torch.no_grad():
            flat = idx.reshape(-1)
            self.touch.index_add_(0, flat, torch.ones_like(flat, dtype=torch.long))
            self.step[flat] = self.global_step

        return out #[B, K]

    @torch.no_grad()
    def Reset(self):
        self.P_keys.zero_()
        self.P_vals.zero_()
        self.prio.zero_()
        self.step.zero_()
        self.touch.zero_()
        self.source.zero_()
        self.filled.fill_(0)
        self.global_step.fill_(0)


class SymbolicEmbed(AGICoreModule):
    def __init__(self, k: int, outDim: int, hidden_cap: int = 2048):
        super().__init__()
        self.K = int(k)
        self.outDim = int(outDim)

        in_feat = 2 * self.K + 4

        hidden = min(int(hidden_cap), max(512, in_feat // 2))

        self.mlp = nn.Sequential(
            nn.Linear(in_feat, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, self.outDim),)

    def forward(
        self, 
        pCur: torch.Tensor, #[B, K]
        symRecall: torch.Tensor):
        eps = pCur.new_tensor(1e-6)
        p_c = pCur.clamp(eps, 1.0 - eps) #[B, K]

        ent = -(p_c * p_c.log() + (1.0 - p_c) * (1.0 - p_c).log()).mean(dim=-1, keepdim=True) #[B, 1]
        pmax = p_c.max(dim=-1, keepdim=True).values
        pmean = p_c.mean(dim=-1, keepdim=True)

        pspa = (p_c < 0.1).float().mean(dim=-1, keepdim=True)

        feat = torch.cat([pCur, symRecall, ent, pmax, pmean, pspa], dim=-1) #[B, 2*K + 4]

        out = self.mlp(feat)
        return out #[B, outDim]


class GlobalWorkspace(AGICoreModule):
    def __init__(
        self,
        dim: int,
        slots: int = 12,
        defaultTtl: int = 8,
        recencyTemp: float = 0.07,
        priorityTemp: float = 1.0,):
        super().__init__()

        self.dim = int(dim)
        self.slots = int(slots)
        self.default_ttl = int(defaultTtl)
        self.recency_temp = float(recencyTemp)
        self.priority_temp = float(priorityTemp)

        B0 = 1
        self.register_buffer("keys", torch.zeros(B0, slots, dim))
        self.register_buffer("vals", torch.zeros(B0, slots, dim))
        self.register_buffer("priority", torch.zeros(B0, slots))
        self.register_buffer("ttl", torch.zeros(B0, slots, dtype=torch.long))
        self.register_buffer("last_step", torch.zeros(B0, slots, dtype=torch.long))
        self.register_buffer("source", torch.zeros(B0, slots, dtype=torch.int8))
        self.register_buffer("global_step", torch.zeros(B0, dtype=torch.long))

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        B0 = int(self.keys.size(0))
        if B0 == int(B):
            return

        self.keys = torch.zeros(B, self.slots, self.dim, device=device, dtype=dtype)
        self.vals = torch.zeros(B, self.slots, self.dim, device=device, dtype=dtype)
        self.priority = torch.zeros(B, self.slots, device=device, dtype=dtype)
        self.ttl = torch.zeros(B, self.slots, device=device, dtype=torch.long)
        self.last_step = torch.zeros(B, self.slots, device=device, dtype=torch.long)
        self.source = torch.zeros(B, self.slots, device=device, dtype=torch.int8)
        self.global_step = torch.zeros(B, device=device, dtype=torch.long)


    @torch.no_grad()
    def Reset(self):
        self.keys.zero_()
        self.vals.zero_()
        self.priority.zero_()
        self.ttl.zero_()
        self.last_step.zero_()
        self.source.zero_()
        self.global_step.zero_()

    @torch.no_grad()
    def StepTick(self):
        self.global_step.add_(1)

        alive_mask = self.ttl > 0
        self.ttl[alive_mask] -= 1

        expired = self.ttl <= 0
        if expired.any():
            self.keys[expired] = 0
            self.vals[expired] = 0
            self.priority[expired] = 0
            self.last_step[expired] = 0
            self.source[expired] = 0

    @torch.no_grad()
    def Write(
        self,
        key: torch.Tensor,  # [B, Dim]
        val: torch.Tensor,  # [B, Dim]
        *,
        priority: Optional[torch.Tensor] = None, # [B]
        ttl: Optional[torch.Tensor] = None, # [B]
        tagId: Optional[torch.Tensor] = None, # [B]
        ) -> torch.Tensor:
        B = int(key.size(0))

        self.EnsureB(B, device=key.device, dtype=key.dtype)

        if priority is None:
            pr = torch.ones(B, device=self.device, dtype=self.dtype)
        else:
            pr = priority

        if ttl is None:
            ttl_t = torch.full((B,), self.default_ttl, device=self.device, dtype=torch.long) # [B]
        else:
            ttl_t = ttl # [B]
            ttl_t = torch.clamp(ttl_t, min=0)

        if tagId is None:
            tag_t = torch.zeros(B, device=self.device, dtype=torch.int8) # [B]
        else:
            tag_t = tagId # [B]

        empty = (self.ttl <= 0) | (self.priority <= 0)
        has_empty = empty.any(dim=1) 
        idx_empty = empty.float().argmax(dim=1)

        age = (self.global_step.unsqueeze(1) - self.last_step).clamp(min=0).float() 
        freshness = torch.exp(-age * self.recency_temp) 
        eff = self.priority * self.priority_temp * freshness 
        idx_min = torch.argmin(eff, dim=1)

        idx = torch.where(has_empty, idx_empty, idx_min) 
        b_idx = torch.arange(B, device=self.device, dtype=torch.long)

        self.keys[b_idx, idx] = key
        self.vals[b_idx, idx] = val

        self.priority[b_idx, idx] = pr

        self.ttl[b_idx, idx] = ttl_t
        self.last_step[b_idx, idx] = self.global_step
        self.source[b_idx, idx] = tag_t

        return idx # [B]

    def Attend(
        self,
        query: torch.Tensor, # [B, Dim]
        *,
        topk: int = 4,
        tagMask: Optional[List[int]] = None,) -> torch.Tensor:

        B = int(query.size(0))

        self.EnsureB(B, device=self.device, dtype=self.dtype)

        alive = (self.ttl > 0) & (self.priority > 0) #[B, slots]

        if tagMask is not None:
            allowed = torch.zeros_like(alive, dtype=torch.bool)
            for t in tagMask:
                allowed |= (self.source == int(t))
            alive = alive & allowed

        any_alive = alive.any(dim=1)

        q = query #[B, dim]
        k = self.keys #[B, slots, dim]
        sim = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) #[B, slots]

        age = (self.global_step.unsqueeze(1) - self.last_step).clamp(min=0).float() #[B, slots]
        freshness = torch.exp(-age * self.recency_temp) #[B, slots]
        bias = (self.priority * freshness) #[B, slots]
        sim = sim * bias #[B, slots]

        sim = sim.masked_fill(~alive, float("-inf"))

        kk = max(1, min(int(topk), self.slots))

        if not bool(any_alive.all()):
            sim = sim.clone()
            sim[~any_alive] = 0.0

        top_sim, top_idx = StableTopk(sim, kk)  #[B, kk]
        w = F.softmax(top_sim, dim=-1)

        gather_idx = top_idx.unsqueeze(-1).expand(B, kk, self.dim)
        v_top = torch.gather(self.vals, dim=1, index=gather_idx) #[B, kk, dim]

        out = torch.einsum("bk,bkd->bd", w, v_top) #[B, dim]

        if not bool(any_alive.all()):
            out = out.clone()
            out[~any_alive] = 0.0

        with torch.no_grad():
            b_idx = torch.arange(B, device=self.device, dtype=torch.long).unsqueeze(1).expand(B, kk)
            self.last_step[b_idx, top_idx] = self.global_step.unsqueeze(1).expand(B, kk)

        return out #[B, dim]

    @torch.no_grad()
    def Inspect(self) -> Dict[str, torch.Tensor]:
        return {
            "keys": self.keys.clone(),
            "vals": self.vals.clone(),
            "priority": self.priority.clone(),
            "ttl": self.ttl.clone(),
            "last_step": self.last_step.clone(),
            "source": self.source.clone(),
            "global_step": self.global_step.clone(),}



class SemanticLTM(AGICoreModule):
    def __init__(self, dim: int, capacity: int = 16384):
        super().__init__()
        self.dim = int(dim)
        self.capacity = int(capacity)
        B0 = 1
        self.register_buffer("keys", torch.zeros(B0, capacity, dim))
        self.register_buffer("vals", torch.zeros(B0, capacity, dim))
        self.register_buffer("prio", torch.zeros(B0, capacity))
        self.register_buffer("touch", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("source", torch.zeros(B0, capacity, dtype=torch.int8))
        self.register_buffer("filled", torch.zeros(B0, dtype=torch.long))
        self.register_buffer("global_step", torch.zeros(B0, dtype=torch.long))

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        B0 = int(self.keys.size(0))
        if B0 == int(B):
            return

        self.keys = torch.zeros(B, self.capacity, self.dim, device=device, dtype=dtype)
        self.vals = torch.zeros(B, self.capacity, self.dim, device=device, dtype=dtype)
        self.prio = torch.zeros(B, self.capacity, device=device, dtype=dtype)
        self.touch = torch.zeros(B, self.capacity, device=device, dtype=torch.long)
        self.step = torch.zeros(B, self.capacity, device=device, dtype=torch.long)
        self.source = torch.zeros(B, self.capacity, device=device, dtype=torch.int8)
        self.filled = torch.zeros(B, device=device, dtype=torch.long)
        self.global_step = torch.zeros(B, device=device, dtype=torch.long)


    @torch.no_grad()
    def StepTick(self):
        self.global_step.add_(1)

    @torch.no_grad()
    def Store(
        self,
        key: torch.Tensor,  # [B, D] 
        value: torch.Tensor,  # [B, D] 
        score: torch.Tensor, # [B,]
        source: Optional[torch.Tensor] = None): #[int = MemoryType.SRC_REAL]

        B = int(key.size(0))
        self.EnsureB(B, self.device, self.dtype)

        if source is None:
            source = torch.full((B,), MemoryType.SRC_REAL, device=self.device, dtype=torch.int8)

        filled = self.filled #[B]
        gstep = self.global_step #[B]

        is_full = filled >= self.capacity  #[B]
        idx_append = filled.clamp(max=self.capacity - 1) #[B]

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0) #[1, capacity]
        valid = slots < filled.unsqueeze(1)  #[B, capacity]

        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float() #[B, capacity]
        freshness = torch.exp(-age * 0.001)
        eff = self.prio * freshness * torch.log1p(self.touch.float() + 1.0)
        eff = torch.where(valid, eff, torch.full_like(eff, float("inf"))) #[B, capacity]

        idx_evict = torch.argmin(eff, dim=1) #[B]
        idx = torch.where(is_full, idx_evict, idx_append) #[B]

        b_idx = torch.arange(B, device=self.device)

        self.keys[b_idx, idx] = key
        self.vals[b_idx, idx] = value
        self.prio[b_idx, idx] = score

        self.touch[b_idx, idx] = 1
        self.step[b_idx, idx] = gstep
        self.source[b_idx, idx] = source

        filled_new = torch.where(is_full, filled, filled + 1)
        self.filled = filled_new

    def Retrieve(
        self, 
        query: torch.Tensor, #[B, D]
        topk: int = 8):
        B = int(query.size(0))
        self.EnsureB(B, self.device, self.dtype)

        filled = self.filled
        gstep = self.global_step

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0) #[1, capacity] 
        valid_mask = slots < filled.unsqueeze(1) #[B, capacity]
        any_valid = valid_mask.any(dim=1)  

        Q = query #[B, D]
        M = self.keys #[B, capacity, D]
        sim = torch.bmm(Q.unsqueeze(1), M.transpose(1, 2)).squeeze(1) #[B, capacity]

        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float() #[B, capacity]
        is_imag = (self.source == 1).float()
        beta = 0.001 + 0.01 * is_imag
        freshness = torch.exp(-beta * age)
        sim = sim * freshness #[B, capacity]

        sim = sim.masked_fill(~valid_mask, float("-inf"))

        K = int(min(max(1, int(filled.max().item())), int(topk), self.capacity))
        top_sim, top_idx = StableTopk(sim, K) #[B, K]
        w = F.softmax(top_sim, dim=-1) #[B, K]

        idx_expanded = top_idx.unsqueeze(-1).expand(B, K, self.dim) 
        vecs = torch.gather(self.vals, 1, idx_expanded) #[B, K, D]
        out = torch.einsum("bk,bkd->bd", w, vecs)

        if not any_valid.all():
            out[~any_valid] = 0

        with torch.no_grad():
            ones = torch.ones(B, K, device=self.device, dtype=torch.long)
            self.touch.scatter_add_(dim=1, index=top_idx, src=ones)

            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand_as(top_idx)
            self.step[b_idx, top_idx] = gstep.unsqueeze(1).expand_as(top_idx)

        return out #[B, D]


class EpisodicLTM(AGICoreModule):
    def __init__(self, dim: int, capacity: int = 16384):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        B0 = 1
        self.register_buffer("keys", torch.zeros(B0, capacity, dim))
        self.register_buffer("vals", torch.zeros(B0, capacity, dim))
        self.register_buffer("rew", torch.zeros(B0, capacity))
        self.register_buffer("prio", torch.zeros(B0, capacity))
        self.register_buffer("step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("touch", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("source", torch.zeros(B0, capacity, dtype=torch.int8))
        self.register_buffer("filled", torch.zeros(B0, dtype=torch.long))
        self.register_buffer("global_step", torch.zeros(B0, dtype=torch.long))

    @torch.no_grad()
    def StepTick(self):
        self.global_step.add_(1)

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        B0 = int(self.keys.size(0))
        if B0 == int(B):
            return

        self.keys = torch.zeros(B, self.capacity, self.dim, device=device, dtype=dtype)
        self.vals = torch.zeros(B, self.capacity, self.dim, device=device, dtype=dtype)
        self.rew = torch.zeros(B, self.capacity, device=device, dtype=dtype)
        self.prio = torch.zeros(B, self.capacity, device=device, dtype=dtype)
        self.step = torch.zeros(B, self.capacity, device=device, dtype=torch.long)
        self.touch = torch.zeros(B, self.capacity, device=device, dtype=torch.long)
        self.source = torch.zeros(B, self.capacity, device=device, dtype=torch.int8)
        self.filled = torch.zeros(B, device=device, dtype=torch.long)
        self.global_step = torch.zeros(B, device=device, dtype=torch.long)

    @torch.no_grad()
    def Store(
        self,
        key: torch.Tensor, # [B, D],
        value: torch.Tensor, # [B, D]
        reward: torch.Tensor, # [B]
        score: torch.Tensor, # [B]
        source: Optional[torch.Tensor] = None): #  [int = MemoryType.SRC_REAL]
        B = int(key.size(0))
        self.EnsureB(B, self.device, self.dtype)

        if source is None:
            source = torch.full((B,), MemoryType.SRC_REAL, device=self.device, dtype=torch.int8)

        filled = self.filled 
        gstep = self.global_step  

        is_full = filled >= self.capacity
        idx_append = filled.clamp(max=self.capacity - 1)

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0)  
        valid = slots < filled.unsqueeze(1) 

        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float()  
        freshness = torch.exp(-age * 0.01)
        touch_w = torch.log1p(self.touch.float() + 1.0) 
        eff = (self.prio + 0.5 * self.rew) * freshness * touch_w
        eff = torch.where(valid, eff, torch.full_like(eff, float("inf")))

        idx_evict = torch.argmin(eff, dim=1)  
        idx = torch.where(is_full, idx_evict, idx_append)  

        b_idx = torch.arange(B, device=self.device)

        self.keys[b_idx, idx] = key
        self.vals[b_idx, idx] = value
        self.rew[b_idx, idx] = reward

        self.prio[b_idx, idx] = score

        self.step[b_idx, idx] = gstep
        self.touch[b_idx, idx] = 1
        self.source[b_idx, idx] = source
        self.filled = torch.where(is_full, filled, filled + 1)

    def Retrieve(self, query: torch.Tensor, topk: int = 8, recentBias: float = 0.05):
        B = int(query.size(0))
        self.EnsureB(B, self.device, self.dtype)

        filled = self.filled
        gstep = self.global_step

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0) 
        valid_mask = slots < filled.unsqueeze(1) 
        any_valid = valid_mask.any(dim=1)

        Q = query
        M = self.keys
        sim = torch.bmm(Q.unsqueeze(1), M.transpose(1, 2)).squeeze(1) 

        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float()
        src = self.source
        is_imag = (src == MemoryType.SRC_IMAGINE).float()

        freshness = torch.exp(-recentBias * age)
        extra_bias = 0.03
        freshness = freshness * torch.exp(-extra_bias * age * is_imag)
        sim = sim * freshness

        sim = sim.masked_fill(~valid_mask, float("-inf"))

        K = int(min(max(1, int(filled.max().item())), int(topk), self.capacity))
        top_sim, idx = StableTopk(sim, K)
        w = F.softmax(top_sim, dim=-1)

        idx_expanded = idx.unsqueeze(-1).expand(B, K, self.dim)
        vecs = torch.gather(self.vals, 1, idx_expanded)
        out = torch.einsum("bk,bkd->bd", w, vecs)

        if not any_valid.all():
            out[~any_valid] = 0.0

        with torch.no_grad():
            ones = torch.ones(B, K, device=self.device, dtype=torch.long)
            self.touch.scatter_add_(dim=1, index=idx, src=ones)
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand_as(idx)
            self.step[b_idx, idx] = gstep.unsqueeze(1).expand_as(idx)

        return out #[B, D]

class LTMFuser(AGICoreModule):
    def __init__(self, dim: int, hidden_scale: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)

        self.norm_sem = nn.LayerNorm(self.dim)
        self.norm_epi = nn.LayerNorm(self.dim)

        in_dim = 4 * self.dim + 4
        hidden = int(self.dim * float(hidden_scale))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.dim),)

    def forward(
        self, 
        semOut: torch.Tensor, #[B, dim] 
        epiOut: torch.Tensor #[B, dim] 
        ) -> torch.Tensor:
        sem = self.norm_sem(semOut)
        epi = self.norm_epi(epiOut)

        diff = torch.abs(sem - epi)
        prod = sem * epi

        cos = F.cosine_similarity(sem, epi, dim=-1, eps=1e-8).clamp(-1.0, 1.0) #[B] 
        n_sem = torch.linalg.vector_norm(sem, ord=2, dim=-1) #[B]
        n_epi = torch.linalg.vector_norm(epi, ord=2, dim=-1) #[B]
        n_diff = torch.linalg.vector_norm((sem - epi), ord=2, dim=-1) #[B]
        stats = torch.stack([cos, n_sem, n_epi, n_diff], dim=1) #[B, 4]

        x = torch.cat([sem, epi, diff, prod, stats], dim=1) # [B, 4*dim + 4]
        fused = self.net(x)
        return fused #[B, dim]


class LongTermMemory(AGICoreModule):
    def __init__(self, dim: int, semCap: int = 16384, epiCap: int = 16384):
        super().__init__()
        self.semantic = SemanticLTM(dim, semCap)
        self.episodic = EpisodicLTM(dim, epiCap)

        self.fuser = LTMFuser(dim)

    @torch.no_grad()
    def Reset(self):
        self.semantic.keys.zero_()
        self.semantic.vals.zero_()
        self.semantic.prio.zero_()
        self.semantic.touch.zero_()
        self.semantic.step.zero_()
        self.semantic.source.zero_()
        self.semantic.filled.zero_()
        self.semantic.global_step.zero_()

        self.episodic.keys.zero_()
        self.episodic.vals.zero_()
        self.episodic.rew.zero_()
        self.episodic.prio.zero_()
        self.episodic.step.zero_()
        self.episodic.touch.zero_()
        self.episodic.source.zero_()
        self.episodic.filled.zero_()
        self.episodic.global_step.zero_()

    @torch.no_grad()
    def StepTick(self):
        self.semantic.StepTick()
        self.episodic.StepTick()

    def Retrieve(self, query: torch.Tensor, topkSem: int = 6, topkEpi: int = 4):
        sem_out = self.semantic.Retrieve(query, topk=topkSem) #[B, D]
        epi_out = self.episodic.Retrieve(query, topk=topkEpi) #[B, D]

        return sem_out, epi_out


class FusionMoE(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        outDim: int,
        numExperts: int = 4,
        hidden: int = 1024,
        noisyGating: bool = True,
        noiseStd: Optional[float] = 0.1,  
        temperature: float = 0.9, 
        expertDropout: float = 0.1,):
        super().__init__()
        self.numExperts = int(numExperts)
        self.noisy_gating = bool(noisyGating)
        self.noise_std = float(1.0 / self.numExperts) if noiseStd is None else float(noiseStd)
        self.temperature = float(temperature)
        self.expert_dropout = float(expertDropout)

        self.gate = nn.Linear(inDim, self.numExperts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, outDim),)for _ in range(self.numExperts)])

        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

    def GetAuxLoss(self) -> torch.Tensor:
        return self.aux_loss

    def forward(
        self, 
        x: torch.Tensor #[B, inDim]
        ) -> torch.Tensor:

        logits = self.gate(x) # [B, numExperts]

        if self.training and self.noisy_gating and (self.noise_std > 0.0):
            logits = logits + torch.randn_like(logits) * self.noise_std

        if self.training and (self.expert_dropout > 0.0):
            keep = (torch.rand_like(logits) > self.expert_dropout) 

            all_drop = (~keep).all(dim=-1)
            if all_drop.any():
                k = int(all_drop.sum().item())
                rand_idx = torch.randint(0, self.numExperts, (k,), device=self.device)
                keep[all_drop] = False
                keep[all_drop, rand_idx] = True

            logits = logits.masked_fill(~keep, -1e9)

        t = max(self.temperature, 1e-6)
        a = F.softmax((logits / t).float(), dim=-1) # [B, numExperts]

        if self.training:
            importance = a.float().mean(dim=0) 
            self.aux_loss  = float(self.numExperts) * (importance.pow(2).sum())
        else:
            self.aux_loss = x.new_zeros(())

        ys = [expert(x) for expert in self.experts] 
        y = torch.stack(ys, dim=-1) # [B, outDim, numExperts]
        out = (y * a.unsqueeze(1)).sum(dim=-1) 
        return out # [B, outDim]
    

class MemoryExtractor(AGICoreModule):
    def __init__(
        self,
        inputDim: int = 1024,
        ssmStateDim: int = 1024,
        memoryDim: int = 1024,
        memorySize: int = 16384,
        symSize: int = 16384,
        ltmSize: int = 16384,
        nsK: int = 256,
        outputDim: int = 1024,
        hebbAlpha: float = 0.15,
        decayFactor: float = 0.95,
        topk: int = 8,
        useHebbian: bool = True,
        gwsSlots: int = 24,
        gwsTtl: int = 64,
        compressEvery: int = 8192,
        emotionDim: int = 512,) -> None:
        super().__init__()

        self.ssm_state_dim = ssmStateDim
        self.memory_dim = memoryDim
        self.output_dim = outputDim
        self.memory_size = memorySize
        self.topk = min(topk, memorySize)
        self.hebb_alpha = hebbAlpha
        self.decay = decayFactor
        self.use_hebbian = useHebbian

        self.ltm_topk_sem = 6
        self.ltm_topk_epi = 4
        self.ltm_align_lambda = 0.0 
        self.ltm_online_imp_thresh = 0.75
        self.ltm_online_td_thresh  = 0.60

        self.gws_align_weight = 0.05

        self.enable_rehearsal = True
        self.ltm_cache_ttl = 2

        self.compress_every = compressEvery

        A_init = torch.empty(ssmStateDim, ssmStateDim)
        nn.init.orthogonal_(A_init, gain=0.8)
        self.A_full = nn.Parameter(A_init * 0.05)
        self.B_mat = nn.Linear(inputDim, ssmStateDim, bias=False)
        self.C_mat = nn.Linear(ssmStateDim, outputDim, bias=False)
        self.D_mat = nn.Linear(inputDim, outputDim, bias=False)

        for p in (self.B_mat, self.C_mat, self.D_mat):
            nn.init.xavier_uniform_(p.weight)

        B0 = 1
        self.register_buffer("h_state", torch.zeros(B0, ssmStateDim))
        self.register_buffer("fast_weights", torch.zeros(B0, memoryDim, memoryDim))
        self.register_buffer("memory_keys", torch.zeros(B0, memorySize, memoryDim))
        self.register_buffer("memory_values", torch.zeros(B0, memorySize, memoryDim))
        self.register_buffer("memory_importance", torch.zeros(B0, memorySize))
        self.register_buffer("memory_steps", torch.zeros(B0, memorySize, dtype=torch.long))
        self.register_buffer("memory_source", torch.zeros(B0, memorySize, dtype=torch.int8))

        self.emotion_dim = int(emotionDim)

        self.register_buffer("memory_emotion", torch.zeros(B0, memorySize, self.emotion_dim),)

        self.emo_write_proj = nn.Sequential(
            nn.Linear(self.emotion_dim, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),)

        self.emo_content_gate = nn.Sequential(
            nn.Linear(self.memory_dim * 2, 1024),
            nn.SiLU(),
            nn.Linear(1024, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),)

        self.td_affect_gate = nn.Sequential(
            nn.Linear(1 + self.emotion_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),)
        
        self.emo_val_mod = nn.Sequential(
            nn.Linear(self.memory_dim, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, 2 * self.memory_dim),)

        self.emo_write_alpha = nn.Parameter(torch.tensor(0.3))

        self.sem_gamma_emo = nn.Linear(self.memory_dim, self.memory_dim)
        self.sem_beta_emo = nn.Linear(self.memory_dim, self.memory_dim)

        self.epi_gamma_emo = nn.Linear(self.memory_dim, self.memory_dim)
        self.epi_beta_emo = nn.Linear(self.memory_dim, self.memory_dim)

        self.register_buffer("time_step", torch.zeros(B0, dtype=torch.long)) 
        self.register_buffer("memory_filled", torch.zeros(B0, dtype=torch.long)) 
        self.register_buffer("last_compress_step", torch.zeros(B0, dtype=torch.long))

        self.ctrl_norm = nn.LayerNorm(ssmStateDim)

        ctrl_hidden = self.memory_dim * 4
        self.ctrl_head = nn.Sequential(
            nn.Linear(self.ctrl_norm.normalized_shape[0] + self.memory_dim * 2  + 3 + 1 + 2, ctrl_hidden), nn.SiLU(),
            nn.Linear(ctrl_hidden, ctrl_hidden // 4), nn.SiLU(),
            nn.Linear(ctrl_hidden // 4, ctrl_hidden // 8), nn.SiLU(),
            nn.Linear(ctrl_hidden // 8, 4))
        
        nn.init.zeros_(self.ctrl_head[-1].weight)
        nn.init.zeros_(self.ctrl_head[-1].bias)

        self.kv_mlp = nn.Sequential(
            nn.LayerNorm(ssmStateDim),
            nn.Linear(ssmStateDim, memoryDim // 2), nn.SiLU(),
            nn.Linear(memoryDim // 2, memoryDim // 2), nn.SiLU())

        self.kv_heads = 4
        assert memoryDim % self.kv_heads == 0, "memoryDim must be divisible by kv_heads"
        self.kv_head_dim = memoryDim // self.kv_heads

        self.kv_head_proj = nn.Parameter(torch.randn(self.kv_heads, memoryDim // 2, self.kv_head_dim * 2) * 0.02)
        self.k_bias = nn.Parameter(torch.zeros(self.kv_heads, self.kv_head_dim))
        self.v_bias = nn.Parameter(torch.zeros(self.kv_heads, self.kv_head_dim))
        
        self.importance_net = nn.Sequential(
            nn.Linear(ssmStateDim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 1), nn.Sigmoid(),)

        self.local_gate = nn.Sequential(
            nn.Linear(ssmStateDim, 512), nn.SiLU(),
            nn.Linear(512, 1), nn.Sigmoid(),)

        self.fusion_gate_net = nn.Sequential(
            nn.Linear(memoryDim * 3 + 4, 512), nn.SiLU(),
            nn.Linear(512, 1), nn.Sigmoid(),)
        
        self.output_refine = nn.Sequential(
            nn.Linear(memoryDim * 2, memoryDim),
            nn.LayerNorm(memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim))

        self.fusion = FusionMoE(inDim = memoryDim * 4, outDim = outputDim, numExperts = 4, hidden = 2048)

        self.grad_bridge = nn.Parameter(torch.tensor(0.3))

        self.ltm_cap = ltmSize

        self.gws = GlobalWorkspace(dim=memoryDim, slots=gwsSlots, defaultTtl=gwsTtl)
        self.ltm = LongTermMemory(dim=memoryDim, semCap=ltmSize, epiCap=ltmSize)
        self.gws_summary = nn.Linear(ssmStateDim + outputDim + memoryDim, memoryDim,)

        self.ns_lambda = 0.08
        self.ns_retrieve_boost = 0.3

        self.ns_K: int = nsK
        self.sym_capacity: int = symSize 
        self.ns_gExcl: int = 5  
        self.ns_gOr: int = 5 

        self.ns_coder_post = SymbolicCoder(self.memory_dim, self.ns_K, hidden=1024, experts=4)

        self.sym_rules = SoftSymbolicRules(
            k=nsK,
            gExcl=40,  
            gOr=40,    
            impScale=0.2, 
            binarize=1e-4, 
            l1Imp=1e-5,    
            massExcl=3.0,
            massOr=3.0,
            massImpRow=1.5,
            massWeight=1e-2, 
            initStd=0.5, 
            seedDisjoint=True,)

        self.sym_query = QueryToSymbol(inDim=self.memory_dim, k=self.ns_K, hidden=512) 
        self.sym_mem = SymbolicMemory(k=self.ns_K, capacity=self.sym_capacity)

        self.sym_embed = SymbolicEmbed(self.ns_K, outDim=self.memory_dim)

        self.register_buffer("ns_prev_P_post", torch.zeros(B0, self.ns_K)) 
        self.register_buffer("ns_penalty_vec", torch.zeros(B0, 1))

        def make_film(D: int) -> nn.Module:
            m = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, 2 * D),
                nn.SiLU(),
                nn.Linear(2 * D, 2 * D),)
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)
            return m

        self.mem_film_norm = nn.LayerNorm(self.memory_dim)
        self.gws_film_norm = nn.LayerNorm(self.memory_dim)
        self.sem_film_norm = nn.LayerNorm(self.memory_dim)
        self.epi_film_norm = nn.LayerNorm(self.memory_dim)

        self.film_mem = make_film(self.memory_dim)
        self.film_gws = make_film(self.memory_dim)
        self.film_sem = make_film(self.memory_dim)
        self.film_epi = make_film(self.memory_dim)

        self.film_clip = 0.5

        self.extra_losses: List[torch.Tensor] = []

        self.pending = []

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype) -> None:
        B0 = int(self.h_state.size(0))
        if B0 == int(B):
            return

        self.h_state = torch.zeros(B, self.ssm_state_dim, device=device, dtype=dtype)

        self.fast_weights = torch.zeros(B, self.memory_dim, self.memory_dim, device=device, dtype=dtype)

        self.memory_keys = torch.zeros(B, self.memory_size, self.memory_dim, device=device, dtype=dtype)
        self.memory_values = torch.zeros(B, self.memory_size, self.memory_dim, device=device, dtype=dtype)

        self.memory_importance = torch.zeros(B, self.memory_size, device=device, dtype=dtype)
        self.memory_steps = torch.zeros(B, self.memory_size, device=device, dtype=torch.long)
        self.memory_source = torch.zeros(B, self.memory_size, device=device, dtype=torch.int8)

        self.memory_emotion = torch.zeros(B, self.memory_size, self.emotion_dim, device=device, dtype=dtype)

        self.time_step = torch.zeros(B, device=device, dtype=torch.long)
        self.memory_filled = torch.zeros(B, device=device, dtype=torch.long)
        self.last_compress_step = torch.zeros(B, device=device, dtype=torch.long)

        self.ns_prev_P_post = torch.zeros(B, self.ns_K, device=device, dtype=dtype)
        self.ns_penalty_vec = torch.zeros(B, 1, device=device, dtype=dtype)

        self.ResetInternalLoss()

    def ResetInternalLoss(self):
        self.extra_losses = []

    def AddInternalLoss(self, loss: torch.Tensor):
        self.extra_losses.append(loss)

    def GetInternalLoss(self) -> torch.Tensor:
        if len(self.extra_losses) == 0:
            return torch.zeros([], device=self.device)
        return torch.stack([l for l in self.extra_losses]).sum()

    def EncodeKV(
        self, 
        h: torch.Tensor, #[B, ssm_state_dim]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.kv_mlp(h) #[B, memory_dim // 2]
        heads = torch.einsum('bd,kdf->bkf', x, self.kv_head_proj) #[B, 4, kv_head_dim * 2]

        k_raw, v_raw = heads.chunk(2, dim=-1) # [B, 4, kv_head_dim]

        k_heads = k_raw + self.k_bias.unsqueeze(0) # [B, 4, kv_head_dim]
        v_heads = v_raw + self.v_bias.unsqueeze(0) # [B, 4, kv_head_dim]

        key = k_heads.reshape(h.size(0), -1) 
        val = v_heads.reshape(h.size(0), -1)

        key = F.normalize(key, dim=-1)
        
        return key, val # [B, memory_dim]

    @torch.no_grad()
    def KvStats(
        self, 
        key: torch.Tensor, # [B, memory_dim]
        topK: int = 8) -> torch.Tensor:
        B, M = int(key.size(0)), int(self.memory_size)

        filled = self.memory_filled # [B]
        any_valid = filled > 0 # [B]
        out = torch.zeros(B, 3, device=self.device)

        if not bool(any_valid.any()):
            return out

        sim = torch.einsum("bd,bmd->bm", key, self.memory_keys) # [B, memory_size]

        ar = torch.arange(M, device=key.device).view(1, M)
        valid = ar < filled.view(B, 1) # [B, M]
        sim = sim.masked_fill(~valid, float("-inf"))

        k = int(min(max(1, int(self.memory_filled.max().item())), int(topK), self.memory_size))
        topk_vals, _ = StableTopk(sim, k) # [B, k]

        m = topk_vals.mean(dim=1) # [B]
        s = topk_vals.std(dim=1, unbiased=False) # [B]

        attn = torch.softmax(sim.float(), dim=-1) # [B, M]
        age = (self.time_step.view(B, 1) - self.memory_steps).clamp(min=0).float()
        age_w = (attn * age).sum(dim=1)
        age_w = torch.tanh(age_w / 1000.0)

        out[:, 0] = torch.where(any_valid, m, out[:, 0])
        out[:, 1] = torch.where(any_valid, s, out[:, 1])
        out[:, 2] = torch.where(any_valid, age_w, out[:, 2])
        return out # [B, 3]

    def NsRules(self, P: torch.Tensor, P_prev: Optional[torch.Tensor]) -> torch.Tensor: # P: [B, nsK]

        total_penalty, aux_reg = self.sym_rules(P, P_prev) # total_penalty: [B]

        if self.training:
            self.AddInternalLoss(self.ns_lambda * aux_reg)

        return total_penalty

    @torch.no_grad()
    def NsStore(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: torch.Tensor,
        sourceLabel: Optional[torch.Tensor] = None,):

        B = key.size(0)

        if sourceLabel is None:
            src_all = torch.full((B,), MemoryType.SRC_REAL, dtype=torch.int8, device=self.device)
        else:
            src_all = sourceLabel

        for i in range(B):
            self.sym_mem.Store(key=key[i].detach(), value=value[i].detach(),score=float(importance[i].item()),source=int(src_all[i].item()),)

    def NsPostRead(
        self, 
        memRecall: torch.Tensor #[B, memory_dim]
        ) -> torch.Tensor:
        P_post = self.ns_coder_post(memRecall)  # [B, nsK]
        per_sample_post = self.NsRules(P_post, self.ns_prev_P_post) # [B]
        self.ns_prev_P_post = P_post.detach()

        damp = torch.clamp(per_sample_post, 0, 1).view(-1, 1) # [B, 1]

        self.ns_penalty_vec = damp
        return P_post # [B, nsK], [B, memory_dim]

    def FilmParams(self, film: nn.Module, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gb = film(src) # [B, 2D]
        g, b = gb.chunk(2, dim=-1) # [B,D], [B,D]
        g = self.film_clip * torch.tanh(g)
        return g, b

    def forward(self,
        x: torch.Tensor, # [B, inputDim]
        tdError: torch.Tensor, # [B] [-1, 1]
        emotion: torch.Tensor, # [B, emotionDim]
        reward: torch.Tensor, # [B] 
        reset: bool = False,
        softReset: bool = False,
        sourceLabel: Optional[torch.Tensor] = None) -> torch.Tensor:

        self.ResetInternalLoss()

        B = x.size(0)

        self.EnsureB(B, self.device, self.dtype)

        src_all = sourceLabel
        if sourceLabel is None:
            src_all = torch.full((B,), MemoryType.SRC_REAL, dtype=torch.int8, device=self.device) # [B]

        emotion_eff = emotion
        tdError_eff = tdError
        reward_eff = reward

        if reset:
            self.ResetAll()
        elif softReset:
            self.SoftReset()

        self.FlushPendingWrites()

        self.time_step.add_(1)

        self.gws.StepTick()
        self.ltm.StepTick()
        self.sym_mem.StepTick()

        h_prev = self.h_state.detach() # [B, ssmStateDim]

        h_new = self.h_state @ self.A_full.t() + self.B_mat(x) # [B, ssmStateDim]

        gb = 0.1 + 0.8 * torch.sigmoid(self.grad_bridge) 
        h_mix = gb * h_new + (1.0 - gb) * h_prev # [B, ssmStateDim]

        y_ssm = self.C_mat(h_mix) + self.D_mat(x) # [B, outputDim]

        key, val = self.EncodeKV(h_mix) # [B, memoryDim]

        emo_emb = self.emo_write_proj(emotion_eff) # [B, memoryDim]

        mod = self.emo_val_mod(emo_emb) # [B, 2*memoryDim]
        gamma, beta = mod.chunk(2, dim=-1) # [B, memoryDim]
        gamma = torch.tanh(gamma)

        val_mod = (1 + gamma) * val + beta # [B, memoryDim]

        alpha = 0.25 * torch.tanh(self.emo_write_alpha)
        val = val_mod + alpha * emo_emb # [B, memoryDim]

        sem_scale = self.sem_gamma_emo(emo_emb) 
        sem_shift = self.sem_beta_emo(emo_emb)
    
        epi_scale = self.epi_gamma_emo(emo_emb)
        epi_shift = self.epi_beta_emo(emo_emb)

        sem_in = val * (1 + sem_scale) + sem_shift # [B, memoryDim]
        epi_in = val * (1 + epi_scale) + epi_shift # [B, memoryDim]

        self.h_state = h_mix.detach().clone()

        importance = self.importance_net(h_mix).squeeze(-1) # [B]
        gate_local = self.local_gate(h_mix).squeeze(-1) # [B]

        kv_feat = self.KvStats(key) # [B, 3]

        phi = torch.cat([self.ctrl_norm(h_mix), emo_emb, key, kv_feat, importance.unsqueeze(-1), gate_local.unsqueeze(-1), tdError_eff.unsqueeze(-1)], dim=-1)

        ctrl = self.ctrl_head(phi) # [B, 4]
        a_raw, b_raw, f_raw, bias_raw = ctrl.split(1, dim=-1) # [B, 1]

        a = (0.5 + 0.5 * torch.sigmoid(a_raw)).squeeze(-1) # [B]
        b = (0.9 + 0.1 * torch.sigmoid(b_raw)).squeeze(-1) # [B]
        fusion_gate = torch.sigmoid(f_raw).squeeze(-1) # [B]
        gate_bias = 0.5 * torch.tanh(bias_raw).squeeze(-1) # [B]

        self.HebbianUpdate(key, gate_local, tdError_eff, a, b)

        mem_recall = self.Retrieve(key, fusion_gate, importance=importance, localGate=gate_local, emotion=emotion_eff, tdError=tdError_eff,) # [B, memoryDim]
        
        g2, b2 = self.FilmParams(self.film_mem, val) 
        s2 = 1.0 + g2

        mem_state = self.mem_film_norm(mem_recall * s2 + b2)

        self.pending.append(("kv",
                             (key.detach(),val.detach(),importance.detach(),emotion_eff.detach(),src_all.detach())))

        msg = torch.cat([h_new, y_ssm, val], dim=-1) # [B, ssm+out+mem]

        gws_val = self.gws_summary(msg) # [B, memoryDim]

        gws_recall = self.gws.Attend(key, topk=1) # [B, memoryDim]

        affect_mag = tdError_eff.abs()
        prio = importance * (1.0 + 0.5 * affect_mag).clamp(0.5, 2.0)

        g1, b1 = self.FilmParams(self.film_gws, gws_val) 
        s1 = 1.0 + g1

        gws_state = self.gws_film_norm(gws_recall * s1 + b1) # [B, memoryDim]

        ttl = torch.full((B,), 6, device=self.device, dtype=torch.long)
        ttl = torch.where(src_all == MemoryType.SRC_REAL, torch.full_like(ttl, 10), ttl)
        ttl = torch.where(src_all == MemoryType.SRC_IMAGINE, torch.full_like(ttl, 4), ttl)

        self.pending.append(("gws",
                             (key.detach(),gws_val.detach(),prio.detach(),ttl.detach(),src_all.detach())))
        

        sem_recall, epi_recall= self.ltm.Retrieve(key, topkSem=self.ltm_topk_sem, topkEpi=self.ltm_topk_epi) # [B, memoryDim]

        g3, b3 = self.FilmParams(self.film_sem, sem_in)  
        g4, b4 = self.FilmParams(self.film_epi, epi_in) 

        s3 = 1.0 + g3
        s4 = 1.0 + g4

        sem_state = self.sem_film_norm(sem_recall * s3 + b3) # [B, memoryDim]
        epi_state = self.epi_film_norm(epi_recall * s4 + b4) # [B, memoryDim]

        self.pending.append(("ltm",
                              (key.detach(),sem_in.detach(),epi_in.detach(),importance.detach(),tdError_eff.detach(),reward_eff.detach(),src_all.detach())))     

        ltm_fused = self.ltm.fuser(sem_state, epi_state)

        P_post = self.NsPostRead(val)

        Qsym = self.sym_query(key) # [B, nsK]
        Qsym = F.normalize(Qsym, dim=-1)

        self.pending.append(("ns",
                             (Qsym.detach(), P_post.detach(), importance.detach(), src_all.detach())))

        sym_recall = self.sym_mem.Retrieve(Qsym, topK=8) 

        sym_vec = self.sym_embed(P_post, sym_recall) # [B, memoryDim]

        fused_state = self.fusion(torch.cat([mem_state, gws_state, ltm_fused, sym_vec], dim=-1)) # [B, outputDim]

        fused_state = self.ApplyOutputGate(fused_state, tdError_eff, gate_bias)

        if self.training:
            self.AddInternalLoss(self.fusion.GetAuxLoss())

        step0 = int(self.time_step[0].item())

        if (step0 % self.compress_every) == 0:
            self.pending.append(("compress", None))

        return fused_state

    
    def ApplyOutputGate(self, memRecall: torch.Tensor, tdError: torch.Tensor, gateBias: torch.Tensor) -> torch.Tensor:
        gate_out = (1.0 + torch.tanh(tdError + gateBias)) / 2.0
        return gate_out.view(-1, 1) * memRecall


    @torch.no_grad()
    def HebbianUpdate(
        self,
        key: torch.Tensor, # [B, memory_dim]
        gateLocal: torch.Tensor, # [B]
        neuromod: torch.Tensor, # [B]
        a: torch.Tensor, # [B]
        b: torch.Tensor # [B]
        ) -> None:

        if self.use_hebbian is False:
            return

        B = int(key.size(0))

        key_d = key.detach()

        a3 = a.view(B, 1, 1)
        b3 = b.view(B, 1, 1)
        g3 = gateLocal.view(B, 1, 1)
        n3 = neuromod.view(B, 1, 1)

        outer = torch.bmm(key_d.unsqueeze(2), key_d.unsqueeze(1)) # [B, memory_dim, memory_dim]

        update = (n3 * self.hebb_alpha) * a3 * g3 * outer # [B, memory_dim, memory_dim]

        new_weights = self.fast_weights * self.decay * b3 + update

        max_fro = 5.0
        flat = new_weights.reshape(B, -1)
        fro = torch.linalg.vector_norm(flat, ord=2, dim=1) 
        scale = (max_fro / (fro + 1e-12)).clamp(max=1.0) 

        self.fast_weights = new_weights * scale.view(B, 1, 1)

    @torch.no_grad()
    def KvWrite(
        self, 
        key: torch.Tensor, # [B, memory_dim]
        val: torch.Tensor, # [B, memory_dim]
        importance: torch.Tensor, # [B]
        emotion: torch.Tensor, # [B, emotion_dim]
        source: torch.Tensor # [B]
        ) -> None:
        B = int(key.size(0))
        M = int(self.memory_size)

        imp_w = importance
        src_w = source

        filled = self.memory_filled
        is_full = filled >= M

        idx_append = filled.clamp(max=M - 1)

        idx_evict = torch.argmin(self.memory_importance, dim=1)

        target_idx = torch.where(is_full, idx_evict, idx_append).long()
        b_indices = torch.arange(B, device=self.device)

        self.memory_keys[b_indices, target_idx] = key.detach()
        self.memory_values[b_indices, target_idx] = val.detach()
        self.memory_importance[b_indices, target_idx] = imp_w
        self.memory_steps[b_indices, target_idx] = self.time_step.view(-1)
        self.memory_emotion[b_indices, target_idx] = emotion.detach()
        self.memory_source[b_indices, target_idx] = src_w

        new_filled = (filled + 1).clamp(max=M)
        self.memory_filled.copy_(torch.where(is_full, filled, new_filled))

    @torch.no_grad()
    def LtmOnlineStore(
        self, 
        key: torch.Tensor, # [B, memory_dim]
        valSem: torch.Tensor, # [B, memory_dim]
        valEpi: torch.Tensor, # [B, memory_dim]
        importance: torch.Tensor, # [B]
        tdError: torch.Tensor, # [B]
        reward: torch.Tensor, # [B] [-10, 10]
        sourceLabel: torch.Tensor):
        mask_base = (importance > self.ltm_online_imp_thresh) | (tdError.abs() > self.ltm_online_td_thresh)
        is_imag = (sourceLabel == MemoryType.SRC_IMAGINE)
        mask_imag = is_imag & (importance > self.ltm_online_imp_thresh * 1.3)
        mask_real = (~is_imag) & mask_base
        mask = (mask_real | mask_imag)

        if not mask.any():
            return

        sem = self.ltm.semantic
        epi = self.ltm.episodic

        sem.Store(key=key, value=valSem, score=importance, source=sourceLabel)
        epi.Store(key=key, value=valEpi, reward=reward, score=importance, source=sourceLabel)

    def Retrieve(
        self, 
        query: torch.Tensor, # [B, memory_dim] 
        fusionGate: torch.Tensor, # [B]
        importance: torch.Tensor, # [B] 
        localGate: torch.Tensor, # [B] 
        emotion: torch.Tensor, # [B, emotion_dim]
        tdError: torch.Tensor # [B]
        ) -> torch.Tensor:

        B = int(query.size(0))
        M = int(self.memory_size)
        D = int(self.memory_dim)

        fw = self.fast_weights # [B, memory_dim, memory_dim]
        qf = query
        fast_part = torch.bmm(qf.unsqueeze(1), fw).squeeze(1) # [B, memory_dim]

        filled = self.memory_filled 
        any_valid = filled > 0

        feat_imp = importance.view(B, 1)
        feat_local = localGate.view(B, 1)
        feat_fuse = fusionGate.view(B, 1)
        feat_td = tdError.abs().view(B, 1)

        if not any_valid.any():
            kv_part = torch.zeros_like(fast_part) # [B, memory_dim]
        else:
            keys = self.memory_keys # [B, M, D]
            values = self.memory_values # [B, M, D]
            imp_kv = self.memory_importance # [B, M]
            steps = self.memory_steps # [B, M] 
            src = self.memory_source # [B, M]

            sim = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) # [B, M]
            sim = sim * imp_kv # [B, M]

            age = (self.time_step.view(B, 1) - steps).clamp(min=0).float() # [B, M]
            base_decay = torch.exp(-0.05 * age) # [B, M]
            is_imag = (src == MemoryType.SRC_IMAGINE).float() # [B, M]
            extra_decay = torch.exp(-0.05 * age * is_imag) # [B, M]
            sim = sim * (base_decay * extra_decay) # [B, M]

            ar = torch.arange(M, device=self.device).view(1, M)
            valid_mask = ar < filled.view(B, 1) # [B, M]

            sim = sim.masked_fill(~valid_mask, -1e9)
            sim = torch.where(any_valid.view(B,1), sim, torch.zeros_like(sim)) # [B, M]

            k = int(min(self.topk, M, int(filled.max().item())))
            top_sim, top_idx = StableTopk(sim, k) #[B, k]

            th = top_sim.mean(dim=-1, keepdim=True) - 0.5 * top_sim.std(dim=-1, keepdim=True, unbiased=False) # [B, 1]
            th = torch.clamp(th, min=top_sim.min(dim=-1, keepdim=True).values, max=top_sim.max(dim=-1, keepdim=True).values)

            if self.ns_retrieve_boost > 0.0:
                th = th + self.ns_retrieve_boost * self.ns_penalty_vec # [B, 1]

            mask = top_sim > th # [B, k]
            masked_top = top_sim.masked_fill(~mask, -1e9)
            all_false = ~mask.any(dim=-1, keepdim=True)
            top_sim_eff = torch.where(all_false, top_sim, masked_top)

            attn = F.softmax(top_sim_eff, dim=-1)
            attn = torch.where(any_valid.view(B,1), attn, torch.zeros_like(attn))

            idx3 = top_idx.unsqueeze(-1).expand(B, k, D)
            vals = torch.gather(values, 1, idx3) # [B, k, D]

            src_sel = torch.gather(src, 1, top_idx) # [B, k]
            mask_real = (src_sel == MemoryType.SRC_REAL)
            mask_imag = (src_sel == MemoryType.SRC_IMAGINE)

            w_real = attn * mask_real.float() # [B, k]
            w_imag = attn * mask_imag.float() # [B, k]
            eps = 1e-6
            sum_real = w_real.sum(dim=-1, keepdim=True) # [B, 1]
            sum_imag = w_imag.sum(dim=-1, keepdim=True) # [B, 1]

            zeros_attn = torch.zeros_like(attn) # [B, k]
            w_real = torch.where(sum_real > eps, w_real / (sum_real + eps), zeros_attn) # [B, k]
            w_imag = torch.where(sum_imag > eps, w_imag / (sum_imag + eps), zeros_attn) # [B, k]

            mem_real = torch.einsum("bk,bkd->bd", w_real, vals) # [B, D]
            mem_imag = torch.einsum("bk,bkd->bd", w_imag, vals) # [B, D]

            strength_real = w_real.sum(dim=-1, keepdim=True) # [B, 1]
            strength_imag = w_imag.sum(dim=-1, keepdim=True) # [B, 1]

            lambda_logit = (strength_imag - strength_real) + 0.5 * feat_td - 0.3
            lambda_imag = torch.sigmoid(lambda_logit)
            mem_task = (1.0 - lambda_imag) * mem_real + lambda_imag * mem_imag # [B, D]

            emo_vals = self.memory_emotion # [B, M, emotion_dim]
            emo_sel = torch.gather(emo_vals, 1, top_idx.unsqueeze(-1).expand(B, k, self.emotion_dim)) # [B, k, emotion_dim]
            emo_q = emotion.unsqueeze(1) # [B, 1, emotion_dim]
            emo_sim = F.cosine_similarity(emo_sel, emo_q.expand_as(emo_sel), dim=-1) # [B, k]
            emo_w = F.softmax(emo_sim.float(), dim=-1) # [B, k]
            mem_affect = torch.einsum("bk,bkd->bd", emo_w, vals) # [B, D]

            emo_vec = torch.einsum("bk,bke->be", emo_w, emo_sel) # [B, emotion_dim]
            emo_embed = self.emo_write_proj(emo_vec) # [B, memory_dim]
            emo_gate = self.emo_content_gate(torch.cat([mem_affect, emo_embed], dim=-1)) # [B, 1]
            mem_affect = (1.0 - emo_gate) * mem_affect + emo_gate * emo_embed # [B, D]

            gate_feat = torch.cat([feat_td, emotion], dim=-1) # [B, 1 + emotion_dim]
            gamma = self.td_affect_gate(gate_feat) # [B, 1]
            kv_part = gamma * mem_task + (1.0 - gamma) * mem_affect # [B, D]

        fusion_input = torch.cat([query, fast_part, kv_part, feat_imp, feat_local, feat_fuse, feat_td], dim=-1) # [B, 3*D + 4]
        gate = self.fusion_gate_net(fusion_input)

        base_out = gate * fast_part + (1.0 - gate) * kv_part

        concat_feat = torch.cat([fast_part, kv_part], dim=-1)
        refine_out = self.output_refine(concat_feat)

        return base_out + refine_out # [B, D]


    @torch.no_grad()
    def AutoCompress(self):
        B = int(self.memory_filled.size(0))
        M = int(self.memory_size)
        device = self.device

        cond_fill = self.memory_filled >= self.memory_size # [B] 

        if not cond_fill.any():
            return

        target_indices = torch.nonzero(cond_fill, as_tuple=True)[0]

        sim_matrix = torch.bmm(self.memory_keys, self.memory_keys.transpose(1, 2)) # [B, M, M]

        range_mat = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
        valid_mask = range_mat < self.memory_filled.unsqueeze(1) # [B, M]
        
        for b in target_indices:
            n = int(self.memory_filled[b].item())
            if n < 2: continue

            sim = sim_matrix[b, :n, :n]
            
            mask_tri = torch.triu(torch.ones_like(sim), diagonal=1).bool()
            
            src_idx, tgt_idx = torch.nonzero((sim > 0.95) & mask_tri, as_tuple=True)
            
            if src_idx.numel() == 0:
                continue

            processed = torch.zeros(n, device=device, dtype=torch.bool)
            
            for s, t in zip(src_idx.tolist(), tgt_idx.tolist()):
                if processed[s] or processed[t]:
                    continue
                
                w_s = self.memory_importance[b, s]
                w_t = self.memory_importance[b, t]
                w_sum = w_s + w_t + 1e-6 
                
                v_s = self.memory_values[b, s]
                v_t = self.memory_values[b, t]
                new_v = (w_s * v_s + w_t * v_t) / w_sum
                self.memory_values[b, t] = new_v

                k_s = self.memory_keys[b, s]
                k_t = self.memory_keys[b, t]
                new_k = (w_s * k_s + w_t * k_t) / w_sum
                self.memory_keys[b, t] = F.normalize(new_k, dim=0)

                self.memory_importance[b, t] = w_s + w_t

                self.memory_steps[b, t] = max(self.memory_steps[b, s], self.memory_steps[b, t])

                self.memory_importance[b, s] = -1e9
                
                processed[s] = True
                processed[t] = True

        age = (self.time_step.unsqueeze(1) - self.memory_steps).clamp(min=0).float() # [B, M]
        decay = torch.exp(-0.01 * age) # [B, M]

        temp_imp = self.memory_importance * decay # [B, M]

        is_imag = (self.memory_source == MemoryType.SRC_IMAGINE).float() # [B, M]
        scores = temp_imp * (1.0 - 0.3 * is_imag) # [B, M]

        scores = scores.masked_fill(~valid_mask, -1e9) # [B, M]

        _, sorted_idx = torch.sort(scores, dim=1, descending=True) # [B, M]

        keep_target = (float(M) * 0.7)
        keep_nums = torch.min(self.memory_filled, torch.tensor(keep_target, device=device).long())
        
        new_valid_mask = range_mat < keep_nums.unsqueeze(1)

        def gather_and_trim(tensor, idx_map, mask, dim_last=False):
            if dim_last: 
                idx_expanded = idx_map.unsqueeze(-1).expand(-1, -1, tensor.size(-1))
                gathered = torch.gather(tensor, 1, idx_expanded)
                return gathered * mask.unsqueeze(-1).float()
            else: 
                gathered = torch.gather(tensor, 1, idx_map)
                return gathered * mask.float()

        new_keys = gather_and_trim(self.memory_keys, sorted_idx, new_valid_mask, dim_last=True)
        new_vals = gather_and_trim(self.memory_values, sorted_idx, new_valid_mask, dim_last=True)
        new_emos = gather_and_trim(self.memory_emotion, sorted_idx, new_valid_mask, dim_last=True)
        
        new_imps = gather_and_trim(self.memory_importance, sorted_idx, new_valid_mask, dim_last=False)
        new_steps = gather_and_trim(self.memory_steps, sorted_idx, new_valid_mask, dim_last=False)
        new_srcs = gather_and_trim(self.memory_source, sorted_idx, new_valid_mask, dim_last=False)

        m_b11 = cond_fill.view(B, 1, 1)
        m_b1 = cond_fill.view(B, 1)

        self.memory_keys = torch.where(m_b11, new_keys, self.memory_keys)
        self.memory_values = torch.where(m_b11, new_vals, self.memory_values)
        self.memory_emotion = torch.where(m_b11, new_emos, self.memory_emotion)

        self.memory_importance = torch.where(m_b1, new_imps, self.memory_importance)
        self.memory_steps = torch.where(m_b1, new_steps.long(), self.memory_steps)
        self.memory_source = torch.where(m_b1, new_srcs.to(torch.int8), self.memory_source)

        self.memory_filled = torch.where(cond_fill, keep_nums, self.memory_filled)
        self.last_compress_step = torch.where(cond_fill, self.time_step, self.last_compress_step)


    def FlushPendingWrites(self):
        if not self.pending:
            return

        for kind, payload in self.pending:
            if kind == "gws":
                key, ws_val, prio, ttl, src = payload
                self.gws.Write(key, ws_val,priority=prio,ttl=ttl,tagId=src,)

            elif kind == "kv":
                key, val, imp, emo, src = payload
                self.KvWrite(key=key,val=val,importance=imp,emotion=emo,source=src,)

            elif kind == "ltm":
                key, sem, epi, imp, td, rwd, src = payload
                self.LtmOnlineStore(key=key,valSem=sem,valEpi=epi,importance=imp,tdError=td,reward=rwd,sourceLabel=src,)
                
            elif kind == "ns":
                key, P_post, importance, src = payload
                self.NsStore(key, P_post, importance, src)

            elif kind == "compress":
                self.AutoCompress()

        self.pending.clear()

    @torch.no_grad()
    def SoftReset(self):
        self.h_state.zero_()
        self.fast_weights.zero_()

        self.time_step.zero_()
        self.memory_filled.zero_()
        self.last_compress_step.zero_()

        self.ns_prev_P_post.zero_()
        self.ns_penalty_vec.zero_()
        self.ResetInternalLoss()
        self.pending.clear()


    def ResetAll(self):
        self.h_state.zero_()
        self.fast_weights.zero_()

        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_importance.zero_()
        self.memory_steps.zero_()
        self.memory_emotion.zero_()
        self.memory_source.zero_()

        self.time_step.zero_()
        self.memory_filled.zero_()
        self.last_compress_step.zero_()

        self.gws.Reset()
        self.ltm.Reset()
        self.sym_mem.Reset()

        self.ns_prev_P_post.zero_()
        self.ns_penalty_vec.zero_()
        self.ResetInternalLoss()
        self.pending.clear()

    def ResetHebbianMemory(self):
        self.fast_weights.zero_()

    @torch.no_grad()
    def InitMemoryDocument(self, path: str):
        dir_ = os.path.dirname(path)
        if dir_ and (not os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)

        dev = self.h_state.device

        gws_snap = self.gws.Inspect()
        sem = self.ltm.semantic
        epi = self.ltm.episodic

        state = {
            "h_state": torch.zeros_like(self.h_state, device=dev),
            "fast_weights": torch.zeros_like(self.fast_weights, device=dev),
            "memory_keys": torch.zeros_like(self.memory_keys, device=dev),
            "memory_emotion": torch.zeros_like(self.memory_emotion, device=dev),
            "memory_values": torch.zeros_like(self.memory_values, device=dev),
            "memory_importance": torch.zeros_like(self.memory_importance, device=dev),
            "memory_steps": torch.zeros_like(self.memory_steps, device=dev),
            "time_step": torch.zeros_like(self.time_step, device=dev),
            "memory_filled": torch.zeros_like(self.memory_filled, device=dev),
            "memory_source": torch.zeros_like(self.memory_source, device=dev),

            "last_compress_step": torch.zeros_like(self.last_compress_step, device=dev),

            "gws_keys": gws_snap["keys"].clone().zero_().to(dev),
            "gws_vals": gws_snap["vals"].clone().zero_().to(dev),
            "gws_priority": gws_snap["priority"].clone().zero_().to(dev),
            "gws_ttl": gws_snap["ttl"].clone().zero_().to(dev),
            "gws_last_step": gws_snap["last_step"].clone().zero_().to(dev),
            "gws_source": gws_snap["source"].clone().zero_().to(dev),
            "gws_global_step": gws_snap["global_step"].clone().zero_().to(dev),

            "ltm_sem_keys": sem.keys.clone().zero_().to(dev),
            "ltm_sem_vals": sem.vals.clone().zero_().to(dev),
            "ltm_sem_prio": sem.prio.clone().zero_().to(dev),
            "ltm_sem_touch": sem.touch.clone().zero_().to(dev),
            "ltm_sem_step": sem.step.clone().zero_().to(dev),
            "ltm_sem_filled": sem.filled.clone().zero_().to(dev),
            "ltm_sem_global_step": sem.global_step.clone().zero_(),
            "ltm_sem_source": sem.source.clone().zero_().to(dev),

            "ltm_epi_keys": epi.keys.clone().zero_().to(dev),
            "ltm_epi_vals": epi.vals.clone().zero_().to(dev),
            "ltm_epi_prio": epi.prio.clone().zero_().to(dev),
            "ltm_epi_rew": epi.rew.clone().zero_().to(dev),
            "ltm_epi_step": epi.step.clone().zero_().to(dev),
            "ltm_epi_filled": epi.filled.clone().zero_().to(dev),
            "ltm_epi_global_step": epi.global_step.clone().zero_(),
            "ltm_epi_source": epi.source.clone().zero_().to(dev),

            "sym_mem_P_keys": self.sym_mem.P_keys.clone().zero_().to(dev),
            "sym_mem_P_vals": self.sym_mem.P_vals.clone().zero_().to(dev),
            "sym_mem_prio": self.sym_mem.prio.clone().zero_().to(dev),
            "sym_mem_step": self.sym_mem.step.clone().zero_().to(dev),
            "sym_mem_touch": self.sym_mem.touch.clone().zero_().to(dev),
            "sym_mem_filled": self.sym_mem.filled.clone().zero_().to(dev),
            "sym_mem_global_step": self.sym_mem.global_step.clone().zero_(),
            "sym_mem_source": self.sym_mem.source.clone().zero_().to(dev),

            "ns_prev_P_post": self.ns_prev_P_post.clone().zero_(),

            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,}

        torch.save(state, path)

    @torch.no_grad()
    def SaveState(self, path: str):
        torch.save({"state_dict": self.state_dict()}, path)

    @torch.no_grad()
    def LoadState(self, path: str):
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            return
        
        obj = torch.load(path, weights_only=False)
        sd = obj["state_dict"]
        
        if "h_state" in sd:
            saved_B = sd["h_state"].size(0)
            current_B = self.h_state.size(0)
            if saved_B != current_B:
                self.EnsureB(saved_B, device=self.device, dtype=self.dtype)
                self.gws.EnsureB(saved_B, device=self.device, dtype=self.dtype)
                self.ltm.semantic.EnsureB(saved_B, device=self.device, dtype=self.dtype)
                self.ltm.episodic.EnsureB(saved_B, device=self.device, dtype=self.dtype)

        self.load_state_dict(sd, strict=True)
        self.pending.clear()


    @torch.no_grad()
    def ExportMemoryBank(self, topk: int = 1024,) -> Optional[Dict[str, torch.Tensor]]:
        device = self.device
        B = int(self.memory_filled.size(0)) # [B]
        topk = int(topk)

        if topk <= 0 or self.memory_filled.min().item() <=0:
            return None

        out: Dict[str, torch.Tensor] = {}

        def GatherTopkLatestFirst(
            values: torch.Tensor,
            indices: torch.Tensor,
            steps: torch.Tensor,) -> torch.Tensor:
            
            sel_steps = torch.gather(steps, 1, indices)
            time_order = torch.argsort(sel_steps, dim=1, descending=True)
            time_indices = torch.gather(indices, 1, time_order)
            idx_exp = time_indices.unsqueeze(-1).expand(B, time_indices.size(1), values.size(-1))
            return torch.gather(values, 1, idx_exp).contiguous()

        S_gws = int(self.gws.slots)
        gws_valid = (self.gws.ttl > 0) & (self.gws.priority > 0)  # [B, S_gws]
        K_gws = min(topk, S_gws, int(gws_valid.sum(dim=1).min().item()))
        if K_gws > 0:
            gws_age = (self.gws.global_step.unsqueeze(1) - self.gws.last_step).clamp(min=0).float()
            gws_fresh = torch.exp(-gws_age * float(self.gws.recency_temp))
            gws_scores = (self.gws.priority * gws_fresh).masked_fill(~gws_valid, -1e9)
            _, gws_idx = StableTopk(gws_scores, K_gws)  # [B, K_gws]
            out["gws"] = GatherTopkLatestFirst(self.gws.vals, gws_idx, self.gws.last_step)

        M_kv = int(self.memory_size)
        filled_kv = self.memory_filled  # [B]
        K_kv = min(topk, M_kv, int(filled_kv.min().item()))
        if K_kv > 0:
            ar = torch.arange(M_kv, device=device).view(1, M_kv) # [1, M_kv]
            valid = ar < filled_kv.view(B, 1) # [B, M_kv]
            scores = self.memory_importance.masked_fill(~valid, -1e9) # [B, M_kv]
            _, idx = StableTopk(scores, K_kv) # [B, K_kv]
            out["kv"] = GatherTopkLatestFirst(self.memory_values, idx, self.memory_steps) # [B, K_kv, D]


        sem = self.ltm.semantic
        M_sem = int(sem.capacity)
        filled_sem = sem.filled # [B]
        K_sem = min(topk, M_sem, int(filled_sem.min().item()))
        if K_sem > 0:
            ar = torch.arange(M_sem, device=sem.prio.device).view(1, M_sem) # [1, M_sem]
            valid = ar < filled_sem.view(B, 1) # [B, M_sem]
            scores = sem.prio.masked_fill(~valid, -1e9) # [B, M_sem]
            _, idx = StableTopk(scores, K_sem) # [B, K_sem]
            out["ltm_sem"] = GatherTopkLatestFirst(sem.vals, idx, sem.step) # [B, K_sem, D]


        epi = self.ltm.episodic
        M_epi = int(epi.capacity)
        filled_epi = epi.filled  # [B]
        K_epi = min(topk, M_epi, int(filled_epi.min().item()))
        if K_epi > 0:
            ar = torch.arange(M_epi, device=epi.prio.device).view(1, M_epi) # [1, M_epi]
            valid = ar < filled_epi.view(B, 1) # [B, M_epi]
            scores = epi.prio.masked_fill(~valid, -1e9) # [B, M_epi]
            _, idx = StableTopk(scores, K_epi) # [B, K_epi]
            out["ltm_epi"] = GatherTopkLatestFirst(epi.vals, idx, epi.step) # [B, K_epi, D]


        sym = self.sym_mem
        n_sym = int(sym.filled.item())
        nsK = int(sym.K)

        K_sym = min(topk, n_sym)
        if K_sym > 0:
            sym_scores = sym.prio[:n_sym] # [n_sym]
            _, sym_idx = StableTopk(sym_scores, K_sym) # [K_sym]
            sel_steps = sym.step[:n_sym].index_select(0, sym_idx)
            time_order = torch.argsort(sel_steps, dim=0, descending=True)
            sym_idx = sym_idx.index_select(0, time_order)
            top_vals = sym.P_vals[:n_sym].index_select(0, sym_idx) # [K_sym, nsK]
            out["sym"] = top_vals.unsqueeze(0).expand(B, K_sym, nsK).contiguous() # [B, K_sym, nsK]

        return out
    

    @torch.no_grad()
    def ExportState(self, step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        gws_snap = self.gws.Inspect()
        sem = self.ltm.semantic
        epi = self.ltm.episodic
        sym = self.sym_mem

        state: Dict[str, torch.Tensor] = {
            "time_step": self.time_step.clone(), # [B]
            "memory_filled": self.memory_filled.clone(), # [B]
            "last_compress_step": self.last_compress_step.clone(), # [B]

            "h_state": self.h_state.clone(), # [B, ssm]
            "fast_weights": self.fast_weights.clone(), # [B, D, D]

            "ns_prev_P_post": self.ns_prev_P_post.clone(), # [B, nsK]
            "ns_penalty_vec": self.ns_penalty_vec.clone(), # [B, 1]

            "gws_global_step": self.gws.global_step.clone(), # [B]
            "ltm_sem_global_step": sem.global_step.clone(), # [B]
            "ltm_epi_global_step": epi.global_step.clone(), # [B]
            "sym_mem_global_step": sym.global_step.clone(),}

        state.update({
            "memory_keys": self.memory_keys.clone(), # [B, M, D]
            "memory_values": self.memory_values.clone(), # [B, M, D]
            "memory_importance": self.memory_importance.clone(), # [B, M]
            "memory_steps": self.memory_steps.clone(), # [B, M]
            "memory_emotion": self.memory_emotion.clone(), # [B, M, E]
            "memory_source": self.memory_source.clone(),}) # [B, M]
        
        state.update({
            "gws_keys": gws_snap["keys"].clone(), # [B, S, D]
            "gws_vals": gws_snap["vals"].clone(), # [B, S, D]
            "gws_priority": gws_snap["priority"].clone(), # [B, S]
            "gws_ttl": gws_snap["ttl"].clone(), # [B, S]
            "gws_last_step": gws_snap["last_step"].clone(), # [B, S]
            "gws_source": gws_snap["source"].clone(),}) # [B, S]
        

        state.update({
            "ltm_sem_keys": sem.keys.clone(), # [B, Cap, D]
            "ltm_sem_vals": sem.vals.clone(), # [B, Cap, D]
            "ltm_sem_prio": sem.prio.clone(), # [B, Cap]
            "ltm_sem_touch": sem.touch.clone(), # [B, Cap]
            "ltm_sem_step": sem.step.clone(), # [B, Cap]
            "ltm_sem_filled": sem.filled.clone(), # [B]
            "ltm_sem_source": sem.source.clone(),})  # [B, Cap]

        state.update({
            "ltm_epi_keys": epi.keys.clone(), # [B, Cap, D]
            "ltm_epi_vals": epi.vals.clone(), # [B, Cap, D]
            "ltm_epi_prio": epi.prio.clone(), # [B, Cap]
            "ltm_epi_rew": epi.rew.clone(), # [B, Cap]
            "ltm_epi_step": epi.step.clone(), # [B, Cap]
            "ltm_epi_touch": epi.touch.clone(), # [B, Cap]
            "ltm_epi_filled": epi.filled.clone(), # [B]
            "ltm_epi_source": epi.source.clone(),}) # [B, Cap]
        
        state.update({
            "sym_mem_P_keys": sym.P_keys.clone(), # [Cap, nsK]
            "sym_mem_P_vals": sym.P_vals.clone(), # [Cap, nsK]
            "sym_mem_prio": sym.prio.clone(), # [Cap]
            "sym_mem_step": sym.step.clone(), # [Cap]
            "sym_mem_touch": sym.touch.clone(), # [Cap]
            "sym_mem_filled": sym.filled.clone(), 
            "sym_mem_source": sym.source.clone(),}) # [Cap]
        
        if step is None:
            return state

        s = int(step)
        device = self.device

        B = int(self.memory_filled.size(0))
        M = int(self.memory_size)
        D = int(self.memory_dim)
        ar = torch.arange(M, device=device).view(1, M) # [1,M]

        filled = state["memory_filled"] # [B]
        valid0 = ar < filled.view(B, 1) # [B,M]
        keep0 = valid0 & (state["memory_steps"] > s) # [B,M]
        keep_cnt = keep0.sum(dim=1) # [B]

        metric = torch.where(keep0,state["memory_steps"].float(),torch.full_like(state["memory_steps"].float(), -1e9),)  # [B,M]
        _, idx = torch.sort(metric, dim=1, descending=True) # [B,M]

        new_valid = ar < keep_cnt.view(B, 1) # [B,M]

        idx3 = idx.unsqueeze(-1).expand(B, M, D)
        idxE = idx.unsqueeze(-1).expand(B, M, int(self.emotion_dim))

        state["memory_keys"] = torch.gather(state["memory_keys"], 1, idx3) * new_valid.unsqueeze(-1).float()
        state["memory_values"] = torch.gather(state["memory_values"], 1, idx3) * new_valid.unsqueeze(-1).float()
        state["memory_emotion"] = torch.gather(state["memory_emotion"], 1, idxE) * new_valid.unsqueeze(-1).float()

        state["memory_importance"] = torch.gather(state["memory_importance"], 1, idx) * new_valid.float()
        state["memory_steps"] = (torch.gather(state["memory_steps"], 1, idx) * new_valid.long())
        state["memory_source"] = torch.where(new_valid,torch.gather(state["memory_source"], 1, idx),torch.zeros_like(state["memory_source"]),)

        state["memory_filled"] = keep_cnt # [B]

        CapS = int(sem.capacity)
        arS = torch.arange(CapS, device=device).view(1, CapS)

        filledS = state["ltm_sem_filled"] # [B]
        validS0 = arS < filledS.view(B, 1)
        keepS0 = validS0 & (state["ltm_sem_step"] > s)
        keepS = keepS0.sum(dim=1)

        metricS = torch.where(keepS0,state["ltm_sem_step"].float(),torch.full_like(state["ltm_sem_step"].float(), -1e9),)
        _, idxS = torch.sort(metricS, dim=1, descending=True)
        new_validS = arS < keepS.view(B, 1)

        idxS3 = idxS.unsqueeze(-1).expand(B, CapS, D)

        state["ltm_sem_keys"] = torch.gather(state["ltm_sem_keys"], 1, idxS3) * new_validS.unsqueeze(-1).float()
        state["ltm_sem_vals"] = torch.gather(state["ltm_sem_vals"], 1, idxS3) * new_validS.unsqueeze(-1).float()
        state["ltm_sem_prio"] = torch.gather(state["ltm_sem_prio"], 1, idxS) * new_validS.float()
        state["ltm_sem_touch"] = (torch.gather(state["ltm_sem_touch"], 1, idxS) * new_validS.long())
        state["ltm_sem_step"] = (torch.gather(state["ltm_sem_step"], 1, idxS) * new_validS.long())
        state["ltm_sem_source"] = torch.where(new_validS,torch.gather(state["ltm_sem_source"], 1, idxS),torch.zeros_like(state["ltm_sem_source"]),)
        state["ltm_sem_filled"] = keepS

        CapE = int(epi.capacity)
        arE = torch.arange(CapE, device=device).view(1, CapE)

        filledE = state["ltm_epi_filled"]
        validE0 = arE < filledE.view(B, 1)
        keepE0 = validE0 & (state["ltm_epi_step"] > s)
        keepE = keepE0.sum(dim=1)

        metricE = torch.where(keepE0,state["ltm_epi_step"].float(),torch.full_like(state["ltm_epi_step"].float(), -1e9),)
        _, idxEpi = torch.sort(metricE, dim=1, descending=True)
        new_validE = arE < keepE.view(B, 1)

        idxE3 = idxEpi.unsqueeze(-1).expand(B, CapE, D)

        state["ltm_epi_keys"] = torch.gather(state["ltm_epi_keys"], 1, idxE3) * new_validE.unsqueeze(-1).float()
        state["ltm_epi_vals"] = torch.gather(state["ltm_epi_vals"], 1, idxE3) * new_validE.unsqueeze(-1).float()
        state["ltm_epi_prio"] = torch.gather(state["ltm_epi_prio"], 1, idxEpi) * new_validE.float()
        state["ltm_epi_rew"] = torch.gather(state["ltm_epi_rew"], 1, idxEpi) * new_validE.float()
        state["ltm_epi_touch"] = (torch.gather(state["ltm_epi_touch"], 1, idxEpi) * new_validE.long())
        state["ltm_epi_step"] = (torch.gather(state["ltm_epi_step"], 1, idxEpi) * new_validE.long())
        state["ltm_epi_source"] = torch.where(new_validE,torch.gather(state["ltm_epi_source"], 1, idxEpi),torch.zeros_like(state["ltm_epi_source"]),)
        state["ltm_epi_filled"] = keepE

        cap_sym = int(sym.capacity)
        n_sym = int(state["sym_mem_filled"].item())

        if n_sym > 0:
            steps_sym = state["sym_mem_step"][:n_sym] # [n_sym]
            keep_sym0 = steps_sym > s # [n_sym]
            keep_sym = int(keep_sym0.sum().item())

            metric_sym = torch.full((cap_sym,), -1e9, device=state["sym_mem_step"].device, dtype=torch.float32)
            metric_sym[:n_sym] = torch.where(keep_sym0, steps_sym.float(), torch.full_like(steps_sym.float(), -1e9))

            idx_sym = torch.argsort(metric_sym, descending=True) # [cap_sym]
            new_valid_sym = torch.arange(cap_sym, device=idx_sym.device) < keep_sym

            state["sym_mem_P_keys"] = state["sym_mem_P_keys"].index_select(0, idx_sym)
            state["sym_mem_P_vals"] = state["sym_mem_P_vals"].index_select(0, idx_sym)
            state["sym_mem_prio"] = state["sym_mem_prio"].index_select(0, idx_sym)
            state["sym_mem_step"] = state["sym_mem_step"].index_select(0, idx_sym)
            state["sym_mem_touch"] = state["sym_mem_touch"].index_select(0, idx_sym)
            state["sym_mem_source"] = state["sym_mem_source"].index_select(0, idx_sym)

            state["sym_mem_P_keys"][~new_valid_sym] = 0
            state["sym_mem_P_vals"][~new_valid_sym] = 0
            state["sym_mem_prio"][~new_valid_sym] = 0
            state["sym_mem_step"][~new_valid_sym] = 0
            state["sym_mem_touch"][~new_valid_sym] = 0
            state["sym_mem_source"][~new_valid_sym] = 0

            state["sym_mem_filled"] = torch.tensor(keep_sym, device=state["sym_mem_filled"].device, dtype=state["sym_mem_filled"].dtype)
        else:
            state["sym_mem_filled"] = state["sym_mem_filled"].clone().zero_()

        return state

    @torch.no_grad()
    def MergeMemoryState(
        self,
        state: Dict[str, torch.Tensor],
        mergeGws: bool = False,) -> None:

        device = self.device
        dtype = self.dtype

        B_dst = int(self.memory_filled.size(0))

        if mergeGws and ("gws_keys" in state) and ("gws_vals" in state):
            gk = state["gws_keys"].to(device=self.gws.keys.device, dtype=self.gws.keys.dtype) # [B, S, D]
            gv = state["gws_vals"].to(device=self.gws.vals.device, dtype=self.gws.vals.dtype) # [B, S, D]
            gpr = state["gws_priority"].to(device=self.gws.priority.device, dtype=self.gws.priority.dtype) # [B, S]
            gttl = state["gws_ttl"].to(device=self.gws.ttl.device, dtype=torch.long) # [B, S]
            gls = state["gws_last_step"].to(device=self.gws.last_step.device, dtype=torch.long) # [B, S]
            gsrc = state["gws_source"].to(device=self.gws.source.device, dtype=torch.int8) # [B, S]
            ggs = state["gws_global_step"].to(device=self.gws.global_step.device, dtype=torch.long) # [B]

            B_src = int(gk.size(0))
            S_src = int(gk.size(1))
            S_dst = int(self.gws.slots)
            D_dst = int(self.memory_dim)

            B = min(B_dst, B_src)
            S = min(S_dst, S_src)

            if int(gk.size(-1)) == D_dst and int(gv.size(-1)) == D_dst:
                self.gws.keys[:B, :S].copy_(gk[:B, :S])
                self.gws.vals[:B, :S].copy_(gv[:B, :S])
                self.gws.priority[:B, :S].copy_(gpr[:B, :S])
                self.gws.ttl[:B, :S].copy_(gttl[:B, :S])
                self.gws.last_step[:B, :S].copy_(gls[:B, :S])
                self.gws.source[:B, :S].copy_(gsrc[:B, :S])
                self.gws.global_step[:B].copy_(ggs[:B])

        if ("memory_keys" in state) and ("memory_values" in state):
            k_src = state["memory_keys"].to(device=device, dtype=dtype) # [B, M, D]
            v_src = state["memory_values"].to(device=device, dtype=dtype) # [B, M, D]

            if k_src.dim() == 3 and v_src.dim() == 3:
                B_src, M_src, D_src = int(k_src.size(0)), int(k_src.size(1)), int(k_src.size(2))
                B = min(B_dst, B_src)

                if D_src == int(self.memory_dim) and int(v_src.size(2)) == int(self.memory_dim):
                    filled_src = state["memory_filled"].to(device=device, dtype=torch.long)
                    filled_src = filled_src.clamp(min=0, max=M_src)  # [B_src]

                    imp_src = state["memory_importance"].to(device=device, dtype=dtype)

                    emo_src = state["memory_emotion"].to(device=device, dtype=dtype)

                    src_src = state["memory_source"].to(device=device, dtype=torch.int8)

                    step_src = state["memory_steps"].to(device=device, dtype=torch.long)  # [B, M]

                    max_n = int(filled_src[:B].max().item())
                    M_dst = int(self.memory_size)

                    for t in range(max_n):
                        mask = (t < filled_src[:B])  # [B]
                        if not bool(mask.any().item()):
                            continue

                        filled_dst = self.memory_filled[:B] # [B]
                        is_full = filled_dst >= M_dst # [B]
                        idx_append = filled_dst.clamp(max=M_dst - 1) # [B]
                        idx_evict = torch.argmin(self.memory_importance[:B], dim=1) # [B]
                        tgt = torch.where(is_full, idx_evict, idx_append) # [B]

                        b_idx = torch.arange(B, device=device, dtype=torch.long)
                        b_sel = b_idx[mask]
                        tgt_sel = tgt[mask]

                        k_t = k_src[:B, t] # [B, D]
                        v_t = v_src[:B, t] # [B, D]
                        imp_t = imp_src[:B, t]
                        emo_t = emo_src[:B, t]
                        src_t = src_src[:B, t]

                        st_t = step_src[:B, t]

                        self.memory_keys[b_sel, tgt_sel].copy_(k_t[mask])
                        self.memory_values[b_sel, tgt_sel].copy_(v_t[mask])
                        self.memory_importance[b_sel, tgt_sel].copy_(imp_t[mask])
                        self.memory_emotion[b_sel, tgt_sel].copy_(emo_t[mask])
                        self.memory_source[b_sel, tgt_sel].copy_(src_t[mask])
                        self.memory_steps[b_sel, tgt_sel].copy_(st_t[mask])

                        filled_new = (filled_dst + (~is_full).long()).clamp(max=M_dst)
                        self.memory_filled[:B].copy_(torch.where(mask, filled_new, filled_dst))

        if ("ltm_sem_keys" in state) and ("ltm_sem_vals" in state):
            sem = self.ltm.semantic
            k_src = state["ltm_sem_keys"].to(device=sem.keys.device, dtype=sem.keys.dtype) # [B, C, D]
            v_src = state["ltm_sem_vals"].to(device=sem.vals.device, dtype=sem.vals.dtype) # [B, C, D]

            if k_src.dim() == 3 and v_src.dim() == 3:
                B_src, C_src, D_src = int(k_src.size(0)), int(k_src.size(1)), int(k_src.size(2))
                B = min(B_dst, B_src, int(sem.filled.size(0)))
                C_dst = int(sem.capacity)

                if D_src == int(self.memory_dim) and int(v_src.size(2)) == int(self.memory_dim):
                    filled_src = state["ltm_sem_filled"].to(device=sem.keys.device, dtype=torch.long)
                    filled_src = filled_src.clamp(min=0, max=C_src)

                    pr_src = state["ltm_sem_prio"].to(device=sem.prio.device, dtype=sem.prio.dtype)

                    src_src = state["ltm_sem_source"].to(device=sem.source.device, dtype=torch.int8)

                    max_n = int(filled_src[:B].max().item())
                    for t in range(max_n):
                        mask = (t < filled_src[:B])
                        if not bool(mask.any().item()):
                            continue

                        filled_dst = sem.filled[:B]
                        is_full = filled_dst >= C_dst
                        idx_append = filled_dst.clamp(max=C_dst - 1)

                        idx_evict = torch.argmin(sem.prio[:B], dim=1)
                        tgt = torch.where(is_full, idx_evict, idx_append)

                        b_idx = torch.arange(B, device=sem.keys.device, dtype=torch.long)
                        b_sel = b_idx[mask]
                        tgt_sel = tgt[mask]

                        k_t = k_src[:B, t]
                        v_t = v_src[:B, t]
                        p_t = pr_src[:B, t]
                        s_t = src_src[:B, t]

                        sem.keys[b_sel, tgt_sel].copy_(k_t[mask])
                        sem.vals[b_sel, tgt_sel].copy_(v_t[mask])
                        sem.prio[b_sel, tgt_sel].copy_(p_t[mask])
                        sem.source[b_sel, tgt_sel].copy_(s_t[mask])

                        sem.touch[b_sel, tgt_sel].fill_(1)
                        sem.step[b_sel, tgt_sel].copy_(sem.global_step[:B][mask])

                        filled_new = (filled_dst + (~is_full).long()).clamp(max=C_dst)
                        sem.filled[:B].copy_(torch.where(mask, filled_new, filled_dst))

        if ("ltm_epi_keys" in state) and ("ltm_epi_vals" in state):
            epi = self.ltm.episodic
            k_src = state["ltm_epi_keys"].to(device=epi.keys.device, dtype=epi.keys.dtype) # [B, C, D]
            v_src = state["ltm_epi_vals"].to(device=epi.vals.device, dtype=epi.vals.dtype) # [B, C, D]

            if k_src.dim() == 3 and v_src.dim() == 3:
                B_src, C_src, D_src = int(k_src.size(0)), int(k_src.size(1)), int(k_src.size(2))
                B = min(B_dst, B_src, int(epi.filled.size(0)))
                C_dst = int(epi.capacity)

                if D_src == int(self.memory_dim) and int(v_src.size(2)) == int(self.memory_dim):
                    filled_src = state["ltm_epi_filled"].to(device=epi.keys.device, dtype=torch.long)
                    filled_src = filled_src.clamp(min=0, max=C_src)

                    pr_src = state["ltm_epi_prio"].to(device=epi.prio.device, dtype=epi.prio.dtype)

                    rw_src = state["ltm_epi_rew"].to(device=epi.rew.device, dtype=epi.rew.dtype)

                    src_src = state["ltm_epi_source"].to(device=epi.source.device, dtype=torch.int8)

                    max_n = int(filled_src[:B].max().item())
                    for t in range(max_n):
                        mask = (t < filled_src[:B])
                        if not bool(mask.any().item()):
                            continue

                        filled_dst = epi.filled[:B]
                        is_full = filled_dst >= C_dst
                        idx_append = filled_dst.clamp(max=C_dst - 1)

                        idx_evict = torch.argmin(epi.prio[:B], dim=1)
                        tgt = torch.where(is_full, idx_evict, idx_append)

                        b_idx = torch.arange(B, device=epi.keys.device, dtype=torch.long)
                        b_sel = b_idx[mask]
                        tgt_sel = tgt[mask]

                        k_t = k_src[:B, t]
                        v_t = v_src[:B, t]
                        p_t = pr_src[:B, t]
                        r_t = rw_src[:B, t]
                        s_t = src_src[:B, t]

                        epi.keys[b_sel, tgt_sel].copy_(k_t[mask])
                        epi.vals[b_sel, tgt_sel].copy_(v_t[mask])
                        epi.prio[b_sel, tgt_sel].copy_(p_t[mask])
                        epi.rew[b_sel, tgt_sel].copy_(r_t[mask])
                        epi.source[b_sel, tgt_sel].copy_(s_t[mask])

                        epi.touch[b_sel, tgt_sel].fill_(1)
                        epi.step[b_sel, tgt_sel].copy_(epi.global_step[:B][mask])

                        filled_new = (filled_dst + (~is_full).long()).clamp(max=C_dst)
                        epi.filled[:B].copy_(torch.where(mask, filled_new, filled_dst))


        if ("sym_mem_P_keys" in state) and ("sym_mem_P_vals" in state):
            sym = self.sym_mem
            Pk = state["sym_mem_P_keys"].to(device=sym.P_keys.device, dtype=sym.P_keys.dtype) # [C, K]
            Pv = state["sym_mem_P_vals"].to(device=sym.P_vals.device, dtype=sym.P_vals.dtype) # [C, K]
            pr = state["sym_mem_prio"].to(device=sym.prio.device, dtype=sym.prio.dtype).view(-1)
            src = state["sym_mem_source"].to(device=sym.source.device, dtype=torch.int8).view(-1)
            filled_sym = state["sym_mem_filled"]
            n = int(filled_sym.item())
            n = max(0, min(n, Pk.size(0), Pv.size(0), pr.numel(), src.numel()))

            for i in range(n):
                sym.Store(
                    key=Pk[i],
                    value=Pv[i],
                    score=float(pr[i].item()),
                    source=int(src[i].item()),)

        self.pending.clear()

    @torch.no_grad()
    def ImportState(
        self,
        state: Dict[str, torch.Tensor],
        *,
        importGws: bool = True,
        importLtm: bool = True,
        importSym: bool = True,) -> None:
        dev = self.device
        dtype = self.dtype

        B = None
        for k in ("h_state", "memory_keys", "memory_values", "memory_importance", "time_step", "memory_filled"):
            if k in state and isinstance(state[k], torch.Tensor):
                t = state[k]
                if t.dim() >= 1:
                    B = int(t.size(0))
                    break
        if B is None:
            B = int(self.h_state.size(0))

        self.EnsureB(B, device=dev, dtype=dtype)

        if importGws:
            self.gws.EnsureB(B, device=dev, dtype=dtype)
        if importLtm:
            self.ltm.semantic.EnsureB(B, device=dev, dtype=dtype)
            self.ltm.episodic.EnsureB(B, device=dev, dtype=dtype)

        def as_B_vec(x: torch.Tensor, *, to_long: bool) -> torch.Tensor:
            x = x.to(device=dev)
            x = x.long() if to_long else x.to(dtype=dtype)
            if x.dim() == 0:
                x = x.view(1).expand(B)
            elif x.dim() == 1 and x.numel() == 1 and B > 1:
                x = x.expand(B)
            return x

        def copy_batch(dst: torch.Tensor, src: torch.Tensor, *, float_cast: bool = True):
            if not isinstance(src, torch.Tensor):
                return
            src = src.to(device=dst.device)
            if float_cast and dst.dtype.is_floating_point:
                src = src.to(dtype=dst.dtype)

            if src.dim() == dst.dim() - 1:
                src = src.unsqueeze(0).expand(dst.size(0), *src.shape)

            if src.dim() != dst.dim():
                return 

            dst.zero_()

            slices_dst, slices_src = [], []
            for d in range(dst.dim()):
                n = min(dst.size(d), src.size(d))
                slices_dst.append(slice(0, n))
                slices_src.append(slice(0, n))
            dst[tuple(slices_dst)].copy_(src[tuple(slices_src)])

        if "time_step" in state:
            self.time_step.copy_(as_B_vec(state["time_step"], to_long=True))

        if "memory_filled" in state:
            mf = as_B_vec(state["memory_filled"], to_long=True).clamp(min=0, max=int(self.memory_size))
            self.memory_filled.copy_(mf)

        if "last_compress_step" in state:
            self.last_compress_step.copy_(as_B_vec(state["last_compress_step"], to_long=True))

        if "h_state" in state:
            copy_batch(self.h_state, state["h_state"], float_cast=True)

        if "fast_weights" in state:
            copy_batch(self.fast_weights, state["fast_weights"], float_cast=True)

        if "memory_keys" in state:
            copy_batch(self.memory_keys, state["memory_keys"], float_cast=True)
        if "memory_values" in state:
            copy_batch(self.memory_values, state["memory_values"], float_cast=True)
        if "memory_importance" in state:
            copy_batch(self.memory_importance, state["memory_importance"], float_cast=True)
        if "memory_steps" in state:
            copy_batch(self.memory_steps, state["memory_steps"].long(), float_cast=False)
        if "memory_emotion" in state:
            copy_batch(self.memory_emotion, state["memory_emotion"], float_cast=True)
        if "memory_source" in state:
            copy_batch(self.memory_source, state["memory_source"].to(torch.int8), float_cast=False)

        if importGws:
            if "gws_global_step" in state:
                self.gws.global_step.copy_(as_B_vec(state["gws_global_step"], to_long=True))

            if "gws_keys" in state:
                copy_batch(self.gws.keys, state["gws_keys"], float_cast=True)
            if "gws_vals" in state:
                copy_batch(self.gws.vals, state["gws_vals"], float_cast=True)
            if "gws_priority" in state:
                copy_batch(self.gws.priority, state["gws_priority"], float_cast=True)
            if "gws_ttl" in state:
                copy_batch(self.gws.ttl, state["gws_ttl"].long(), float_cast=False)
            if "gws_last_step" in state:
                copy_batch(self.gws.last_step, state["gws_last_step"].long(), float_cast=False)

            if "gws_source" in state:
                copy_batch(self.gws.source, state["gws_source"].to(torch.int8), float_cast=False)

        if importLtm:
            sem = self.ltm.semantic
            epi = self.ltm.episodic

            if "ltm_sem_global_step" in state:
                sem.global_step.copy_(as_B_vec(state["ltm_sem_global_step"], to_long=True))
            if "ltm_epi_global_step" in state:
                epi.global_step.copy_(as_B_vec(state["ltm_epi_global_step"], to_long=True))

            if "ltm_sem_keys" in state:
                copy_batch(sem.keys, state["ltm_sem_keys"], float_cast=True)
            if "ltm_sem_vals" in state:
                copy_batch(sem.vals, state["ltm_sem_vals"], float_cast=True)
            if "ltm_sem_prio" in state:
                copy_batch(sem.prio, state["ltm_sem_prio"], float_cast=True)
            if "ltm_sem_touch" in state:
                copy_batch(sem.touch, state["ltm_sem_touch"].long(), float_cast=False)
            if "ltm_sem_step" in state:
                copy_batch(sem.step, state["ltm_sem_step"].long(), float_cast=False)
            if "ltm_sem_filled" in state:
                sem.filled.copy_(as_B_vec(state["ltm_sem_filled"], to_long=True).clamp(min=0, max=int(sem.capacity)))
            if "ltm_sem_source" in state:
                copy_batch(sem.source, state["ltm_sem_source"].to(torch.int8), float_cast=False)

            if "ltm_epi_keys" in state:
                copy_batch(epi.keys, state["ltm_epi_keys"], float_cast=True)
            if "ltm_epi_vals" in state:
                copy_batch(epi.vals, state["ltm_epi_vals"], float_cast=True)
            if "ltm_epi_prio" in state:
                copy_batch(epi.prio, state["ltm_epi_prio"], float_cast=True)
            if "ltm_epi_rew" in state:
                copy_batch(epi.rew, state["ltm_epi_rew"], float_cast=True)
            if "ltm_epi_step" in state:
                copy_batch(epi.step, state["ltm_epi_step"].long(), float_cast=False)
            if "ltm_epi_filled" in state:
                epi.filled.copy_(as_B_vec(state["ltm_epi_filled"], to_long=True).clamp(min=0, max=int(epi.capacity)))
            if "ltm_epi_source" in state:
                copy_batch(epi.source, state["ltm_epi_source"].to(torch.int8), float_cast=False)

        if importSym:
            if "sym_mem_global_step" in state:
                self.sym_mem.global_step.copy_(
                    state["sym_mem_global_step"].to(self.sym_mem.global_step.device).long().view(()))

            if "sym_mem_P_keys" in state:
                self.sym_mem.P_keys.zero_()
                src = state["sym_mem_P_keys"].to(self.sym_mem.P_keys.device).float()
                n = min(self.sym_mem.P_keys.size(0), src.size(0))
                self.sym_mem.P_keys[:n].copy_(src[:n])

            if "sym_mem_P_vals" in state:
                self.sym_mem.P_vals.zero_()
                src = state["sym_mem_P_vals"].to(self.sym_mem.P_vals.device).float()
                n = min(self.sym_mem.P_vals.size(0), src.size(0))
                self.sym_mem.P_vals[:n].copy_(src[:n])

            if "sym_mem_prio" in state:
                self.sym_mem.prio.zero_()
                src = state["sym_mem_prio"].to(self.sym_mem.prio.device).float()
                n = min(self.sym_mem.prio.size(0), src.size(0))
                self.sym_mem.prio[:n].copy_(src[:n])

            if "sym_mem_step" in state:
                self.sym_mem.step.zero_()
                src = state["sym_mem_step"].to(self.sym_mem.step.device).long()
                n = min(self.sym_mem.step.size(0), src.size(0))
                self.sym_mem.step[:n].copy_(src[:n])

            if "sym_mem_touch" in state:
                self.sym_mem.touch.zero_()
                src = state["sym_mem_touch"].to(self.sym_mem.touch.device).long()
                n = min(self.sym_mem.touch.size(0), src.size(0))
                self.sym_mem.touch[:n].copy_(src[:n])

            if "sym_mem_source" in state:
                self.sym_mem.source.zero_()
                src = state["sym_mem_source"].to(self.sym_mem.source.device).to(torch.int8)
                n = min(self.sym_mem.source.size(0), src.size(0))
                self.sym_mem.source[:n].copy_(src[:n])

            if "sym_mem_filled" in state:
                filled = state["sym_mem_filled"].to(self.sym_mem.filled.device).long()
                if filled.dim() != 0:
                    filled = filled.view(-1)[0]
                filled = filled.clamp(min=0, max=int(self.sym_mem.capacity))
                self.sym_mem.filled.copy_(filled.view(()))

        if "ns_prev_P_post" in state and isinstance(state["ns_prev_P_post"], torch.Tensor):
            copy_batch(self.ns_prev_P_post, state["ns_prev_P_post"], float_cast=True)
        if "ns_penalty_vec" in state and isinstance(state["ns_penalty_vec"], torch.Tensor):
            copy_batch(self.ns_penalty_vec, state["ns_penalty_vec"], float_cast=True)

        self.ResetInternalLoss()
        self.pending.clear()


    @torch.no_grad()
    def ResetSteps(self, resetGlobal: bool = True) -> None:
        if hasattr(self, "time_step") and isinstance(self.time_step, torch.Tensor):
            self.time_step.zero_() 
        if hasattr(self, "last_compress_step") and isinstance(self.last_compress_step, torch.Tensor):
            self.last_compress_step.zero_()  
        if hasattr(self, "memory_steps") and isinstance(self.memory_steps, torch.Tensor) and self.memory_steps.numel() > 0:
            self.memory_steps.zero_() 

        gws = getattr(self, "gws", None)
        if gws is not None:
            if hasattr(gws, "last_step") and isinstance(gws.last_step, torch.Tensor) and gws.last_step.numel() > 0:
                gws.last_step.zero_() 
            if resetGlobal and hasattr(gws, "global_step") and isinstance(gws.global_step, torch.Tensor):
                gws.global_step.zero_() 

        ltm = getattr(self, "ltm", None)
        if ltm is not None:
            sem = ltm.semantic
            epi = ltm.episodic

            if hasattr(sem, "step") and isinstance(sem.step, torch.Tensor) and sem.step.numel() > 0:
                sem.step.zero_() 
            if resetGlobal and hasattr(sem, "global_step") and isinstance(sem.global_step, torch.Tensor):
                sem.global_step.zero_() 

            if hasattr(epi, "step") and isinstance(epi.step, torch.Tensor) and epi.step.numel() > 0:
                epi.step.zero_() 
            if resetGlobal and hasattr(epi, "global_step") and isinstance(epi.global_step, torch.Tensor):
                epi.global_step.zero_()

        sym = getattr(self, "sym_mem", None)
        if sym is not None:
            if hasattr(sym, "step") and isinstance(sym.step, torch.Tensor) and sym.step.numel() > 0:
                sym.step.zero_() 
            if resetGlobal and hasattr(sym, "global_step") and isinstance(sym.global_step, torch.Tensor):
                sym.global_step.zero_()

    @torch.no_grad()
    def ReorderMemorySteps(self) -> None:
        def RebaseBatchedSteps(steps: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            B, N = steps.shape
            if (B <= 0) or (N <= 0):
                return torch.zeros_like(steps), torch.zeros(B, device=steps.device, dtype=torch.long)

            if not bool(valid.any().item()):
                return torch.zeros_like(steps), torch.zeros(B, device=steps.device, dtype=torch.long)

            max_step = torch.iinfo(steps.dtype).max
            metric = torch.where(valid, steps, torch.full_like(steps, max_step))
            order = torch.argsort(metric, dim=1, descending=False)

            counts = valid.sum(dim=1).to(dtype=torch.long)
            ranks = torch.arange(1, N + 1, device=steps.device, dtype=torch.long).view(1, N).expand(B, N)
            rank_valid = ranks <= counts.view(B, 1)
            assign = torch.where(rank_valid, ranks, torch.zeros_like(ranks))

            new_steps = torch.zeros_like(steps)
            new_steps.scatter_(1, order, assign)
            new_steps = torch.where(valid, new_steps, torch.zeros_like(new_steps))
            return new_steps, counts

        def RebaseFlatSteps(steps: torch.Tensor, filled: int) -> Tuple[torch.Tensor, int]:
            n = int(filled)
            new_steps = torch.zeros_like(steps)
            if n <= 0:
                return new_steps, 0

            order = torch.argsort(steps[:n], dim=0, descending=False)
            ranks = torch.arange(1, n + 1, device=steps.device, dtype=torch.long)
            new_steps[:n].scatter_(0, order, ranks)
            return new_steps, n

        if hasattr(self, "memory_steps") and isinstance(self.memory_steps, torch.Tensor) and self.memory_steps.numel() > 0:
            B, M = self.memory_steps.shape
            slots = torch.arange(M, device=self.memory_steps.device).view(1, M)
            valid = slots < self.memory_filled.view(B, 1)
            new_steps, counts = RebaseBatchedSteps(self.memory_steps, valid)
            self.memory_steps.copy_(new_steps)
            self.time_step.copy_(counts)
            if hasattr(self, "last_compress_step") and isinstance(self.last_compress_step, torch.Tensor):
                self.last_compress_step.zero_()

        gws = getattr(self, "gws", None)
        if gws is not None and hasattr(gws, "last_step") and isinstance(gws.last_step, torch.Tensor):
            valid = (gws.ttl > 0) & (gws.priority > 0)
            new_steps, counts = RebaseBatchedSteps(gws.last_step, valid)
            gws.last_step.copy_(new_steps)
            if hasattr(gws, "global_step") and isinstance(gws.global_step, torch.Tensor):
                gws.global_step.copy_(counts)

        ltm = getattr(self, "ltm", None)
        if ltm is not None:
            sem = ltm.semantic
            epi = ltm.episodic

            if hasattr(sem, "step") and isinstance(sem.step, torch.Tensor):
                B, C = sem.step.shape
                slots = torch.arange(C, device=sem.step.device).view(1, C)
                valid = slots < sem.filled.view(B, 1)
                new_steps, counts = RebaseBatchedSteps(sem.step, valid)
                sem.step.copy_(new_steps)
                if hasattr(sem, "global_step") and isinstance(sem.global_step, torch.Tensor):
                    sem.global_step.copy_(counts)

            if hasattr(epi, "step") and isinstance(epi.step, torch.Tensor):
                B, C = epi.step.shape
                slots = torch.arange(C, device=epi.step.device).view(1, C)
                valid = slots < epi.filled.view(B, 1)
                new_steps, counts = RebaseBatchedSteps(epi.step, valid)
                epi.step.copy_(new_steps)
                if hasattr(epi, "global_step") and isinstance(epi.global_step, torch.Tensor):
                    epi.global_step.copy_(counts)

        sym = getattr(self, "sym_mem", None)
        if sym is not None and hasattr(sym, "step") and isinstance(sym.step, torch.Tensor):
            new_steps, count = RebaseFlatSteps(sym.step, int(sym.filled.item()))
            sym.step.copy_(new_steps)
            if hasattr(sym, "global_step") and isinstance(sym.global_step, torch.Tensor):
                sym.global_step.fill_(int(count))




class TestMemoryMTool:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        self.root = Path("BrainDeepLearn/TestData").expanduser().absolute()
        self.root.mkdir(parents=True, exist_ok=True)

    def StatePath(self, name: str = "memory_state.pth") -> Path:
        return (self.root / name).absolute()

    def FilterKwargs(self, cls, cfg: Dict) -> Dict:
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        return {k: v for k, v in cfg.items() if k in allowed}

    def MakeEmotion(self, B: int, mem) -> torch.Tensor:
        return torch.randn(B, int(mem.emotion_dim), device=self.device)

    def MakeEmotionGen(self, B: int, mem, generator: torch.Generator) -> torch.Tensor:
        return torch.randn(B, int(mem.emotion_dim), device=self.device, generator=generator)

    def RandnLikeGen(self, x: torch.Tensor, generator: torch.Generator | None = None):
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    def CallMemForward(
        self,
        mem,
        x: torch.Tensor,
        *,
        tdError: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        emotion: Optional[torch.Tensor] = None,
        reset: bool = False,
        softReset: bool = False,
        sourceLabel: Optional[torch.Tensor] = None,) -> torch.Tensor:
        B = int(x.size(0))
        device = x.device
        tdError = torch.zeros(B, device=device) if tdError is None else tdError.view(B).to(device=device)
        reward = torch.zeros(B, device=device) if reward is None else reward.view(B).to(device=device)
        if emotion is None:
            emotion = self.MakeEmotion(B, mem)
        else:
            emotion = emotion.to(device=device)

        return mem(
            x,
            tdError=tdError,
            emotion=emotion,
            reward=reward,
            reset=reset,
            softReset=softReset,
            sourceLabel=sourceLabel,)

    def AssertClose(self, a: torch.Tensor, b: torch.Tensor, *, atol=1e-5, rtol=1e-4, msg=""):
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            diff = (a - b).abs().max().item()
            raise AssertionError(f"{msg} not close: max|diff|={diff:g}, atol={atol:g}, rtol={rtol:g}")

    def AttachAllInternalLosses(self, rootModule: torch.nn.Module, baseLoss: torch.Tensor):
        getter = getattr(rootModule, "GetInternalLoss", None) or getattr(rootModule, "get_internal_loss", None)
        if callable(getter):
            extra = getter()
            if isinstance(extra, torch.Tensor):
                extra = extra.to(device=baseLoss.device, dtype=baseLoss.dtype)
                return baseLoss + extra
        return baseLoss

    def TestStableTopkTieBreak(self):
        try:
            scores = torch.zeros(2, 10, device=self.device)
            vals, idx = StableTopk(scores, k=4, epsMax=1e-6, preferLowIndex=True)
            assert idx.shape == (2, 4)
            expect = torch.tensor([0, 1, 2, 3], device=self.device).view(1, 4).expand(2, 4)
            assert torch.equal(idx, expect), f"tie-break failed: got {idx[0].tolist()}"
            print("StableTopk tie-break test passed.")
            return True
        except Exception as e:
            print(f"StableTopk tie-break test failed: {e}")
            return False

    def TestGlobalWorkspace(self):
        try:
            dim, slots = 16, 4
            gws = GlobalWorkspace(dim=dim, slots=slots, defaultTtl=2).to(self.device)
            gws.Reset()

            B = 1
            e0 = torch.zeros(B, dim, device=self.device); e0[0, 0] = 1
            e1 = torch.zeros(B, dim, device=self.device); e1[0, 1] = 1
            e2 = torch.zeros(B, dim, device=self.device); e2[0, 2] = 1
            val = torch.randn(B, dim, device=self.device)

            gws.StepTick()
            gws.Write(e0, val, priority=torch.tensor([0.9], device=self.device),
                      ttl=torch.tensor([2], device=self.device), tagId=torch.tensor([1], device=self.device, dtype=torch.int8))
            gws.Write(e1, val, priority=torch.tensor([0.8], device=self.device),
                      ttl=torch.tensor([2], device=self.device), tagId=torch.tensor([1], device=self.device, dtype=torch.int8))
            gws.Write(e2, val, priority=torch.tensor([0.7], device=self.device),
                      ttl=torch.tensor([1], device=self.device), tagId=torch.tensor([2], device=self.device, dtype=torch.int8))

            q = (e0 + e1)
            out = gws.Attend(q, topk=2)
            assert out.shape == (B, dim)

            out_tag1 = gws.Attend(q, topk=3, tagMask=[1])
            assert out_tag1.shape == (B, dim)

            gws.StepTick()
            snap = gws.Inspect()
            assert torch.isfinite(snap["vals"]).all()

            print("GlobalWorkspace basic test passed.")
            return True
        except AssertionError as e:
            print(f"GlobalWorkspace test failed: {e}")
            return False
        except Exception as e:
            print(f"GlobalWorkspace test error: {e}")
            return False

    def TestGlobalWorkspaceEvictionBehavior(self):
        try:
            dim, slots = 8, 3
            gws = GlobalWorkspace(dim=dim, slots=slots, defaultTtl=10, recencyTemp=0.0).to(self.device)
            gws.Reset()
            B = 1

            def put(i, pr):
                k = torch.zeros(B, dim, device=self.device); k[0, i] = 1.0
                v = torch.randn(B, dim, device=self.device)
                gws.Write(
                    k, v,
                    priority=torch.tensor([pr], device=self.device),
                    ttl=torch.tensor([10], device=self.device),
                    tagId=torch.tensor([0], device=self.device, dtype=torch.int8),)

            gws.StepTick()
            put(0, 0.1)
            put(1, 0.2)
            put(2, 0.3)

            snap0 = gws.Inspect()
            keys0 = snap0["keys"].clone()

            put(3, 0.9)

            snap1 = gws.Inspect()
            keys1 = snap1["keys"].clone()

            delta = (keys1 - keys0).abs().sum(dim=-1).squeeze(0)
            changed = int(delta.argmax().item())
            old_onehot = keys0[0, :, :].argmax(dim=-1)
            assert int(old_onehot[changed].item()) == 0, f"evict wrong slot: expected key0, got key{int(old_onehot[changed].item())}"

            print("GlobalWorkspace eviction behavior test passed.")
            return True
        except AssertionError as e:
            print(f"GlobalWorkspace eviction behavior test failed: {e}")
            return False
        except Exception as e:
            print(f"GlobalWorkspace eviction behavior test error: {e}")
            return False

    def TestLongTermMemory(self):
        try:
            dim = 32
            ltm = LongTermMemory(dim=dim, semCap=16, epiCap=16).to(self.device)
            ltm.Reset()

            B = 1
            key = F.normalize(torch.randn(B, dim, device=self.device), dim=-1)
            val = torch.randn(B, dim, device=self.device)

            ltm.semantic.Store(key=key, value=val, score=torch.tensor([0.9], device=self.device),
                              source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8))
            ltm.episodic.Store(key=key, value=val, reward=torch.tensor([-1.0], device=self.device),
                               score=torch.tensor([0.8], device=self.device),
                               source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8))

            ltm.StepTick()
            sem_out, epi_out = ltm.Retrieve(key, topkSem=4, topkEpi=2)
            assert sem_out.shape == (B, dim) and epi_out.shape == (B, dim)
            assert torch.linalg.norm(sem_out).item() > 0 and torch.linalg.norm(epi_out).item() > 0

            print("LongTermMemory test passed.")
            return True
        except AssertionError as e:
            print(f"LongTermMemory test failed: {e}")
            return False
        except Exception as e:
            print(f"LongTermMemory test error: {e}")
            return False

    def TestEpisodicEvictionTouchPrefersKeepHighTouch(self):
        try:
            dim = 16
            cap = 3
            epi = EpisodicLTM(dim=dim, capacity=cap).to(self.device)
            epi.EnsureB(B=1, device=self.device, dtype=torch.float32)
            epi.StepTick()

            def store_with(i, prio, rew):
                k = torch.zeros(1, dim, device=self.device); k[0, i] = 1.0
                v = torch.randn(1, dim, device=self.device)
                epi.Store(
                    key=k,
                    value=v,
                    reward=torch.tensor([rew], device=self.device),
                    score=torch.tensor([prio], device=self.device),
                    source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),)

            store_with(0, prio=1.0, rew=0.0)
            store_with(1, prio=1.0, rew=0.0)
            store_with(2, prio=100.0, rew=0.0) 

            with torch.no_grad():
                epi.touch[0, 0] = 1
                epi.touch[0, 1] = 50
                epi.touch[0, 2] = 1
                epi.step[0, :3] = epi.global_step[0]

            keys_before = epi.keys.clone()

            store_with(3, prio=0.5, rew=0.0)
            keys_after = epi.keys.clone()

            changed = (keys_after - keys_before).abs().sum(dim=-1).squeeze(0).argmax().item()
            assert int(changed) == 0, f"expected evict slot0(low touch), but evicted slot{changed}"

            print("EpisodicLTM eviction touch-direction test passed.")
            return True
        except AssertionError as e:
            print(f"EpisodicLTM eviction touch-direction test failed: {e}")
            return False
        except Exception as e:
            print(f"EpisodicLTM eviction touch-direction test error: {e}")
            return False

    def TestEpisodicEvictionAbsRewardForNegative(self):
        try:
            dim = 16
            cap = 3
            epi = EpisodicLTM(dim=dim, capacity=cap).to(self.device)
            epi.EnsureB(B=1, device=self.device, dtype=torch.float32)
            epi.StepTick()

            def store_with(i, prio, rew):
                k = torch.zeros(1, dim, device=self.device); k[0, i] = 1.0
                v = torch.randn(1, dim, device=self.device)
                epi.Store(
                    key=k,
                    value=v,
                    reward=torch.tensor([rew], device=self.device),
                    score=torch.tensor([prio], device=self.device),
                    source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),)

            store_with(0, prio=1.0, rew=-10.0)
            store_with(1, prio=1.0, rew=0.0)
            store_with(2, prio=100.0, rew=0.0)

            with torch.no_grad():
                epi.touch[0, :3] = 1
                epi.step[0, :3] = epi.global_step[0]

            keys_before = epi.keys.clone()
            store_with(3, prio=0.5, rew=0.0) 
            keys_after = epi.keys.clone()

            changed = (keys_after - keys_before).abs().sum(dim=-1).squeeze(0).argmax().item()
            assert int(changed) == 0, f"expected evict slot0(rew=-10), but evicted slot{changed}"
            print("EpisodicLTM eviction negative-reward test passed.")
            return True
        except AssertionError as e:
            print(f"EpisodicLTM eviction abs(reward) test failed: {e}")
            return False
        except Exception as e:
            print(f"EpisodicLTM eviction abs(reward) test error: {e}")
            return False

    def TestExportMemoryBankLatestFirstOrder(self):
        try:
            cfg = dict(
                inputDim=32,
                ssmStateDim=32,
                memoryDim=8,
                memorySize=8,
                symSize=8,
                ltmSize=8,
                nsK=4,
                outputDim=8,
                gwsSlots=4,
                gwsTtl=8,
                compressEvery=100,
                emotionDim=4,)

            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device)
            mem.ResetAll()

            mark_old, mark_mid, mark_new = 10.0, 20.0, 30.0
            steps = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)
            scores = torch.tensor([3.0, 2.0, 1.0], device=self.device, dtype=mem.dtype)
            expected = torch.tensor([mark_new, mark_mid, mark_old], device=self.device, dtype=mem.dtype)

            with torch.no_grad():
                mem.time_step.fill_(30)
                mem.memory_filled.fill_(3)
                mem.memory_importance[0, :3] = scores
                mem.memory_steps[0, :3] = steps
                mem.memory_values.zero_()
                mem.memory_values[0, 0, 0] = mark_new
                mem.memory_values[0, 1, 0] = mark_old
                mem.memory_values[0, 2, 0] = mark_mid

                mem.gws.global_step.fill_(30)
                mem.gws.priority.zero_()
                mem.gws.ttl.zero_()
                mem.gws.last_step.zero_()
                mem.gws.vals.zero_()
                mem.gws.priority[0, :3] = scores
                mem.gws.ttl[0, :3] = 1
                mem.gws.last_step[0, :3] = steps
                mem.gws.vals[0, 0, 0] = mark_new
                mem.gws.vals[0, 1, 0] = mark_old
                mem.gws.vals[0, 2, 0] = mark_mid

                sem = mem.ltm.semantic
                sem.global_step.fill_(30)
                sem.filled.fill_(3)
                sem.prio.zero_()
                sem.step.zero_()
                sem.vals.zero_()
                sem.prio[0, :3] = scores
                sem.step[0, :3] = steps
                sem.vals[0, 0, 0] = mark_new
                sem.vals[0, 1, 0] = mark_old
                sem.vals[0, 2, 0] = mark_mid

                epi = mem.ltm.episodic
                epi.global_step.fill_(30)
                epi.filled.fill_(3)
                epi.prio.zero_()
                epi.step.zero_()
                epi.vals.zero_()
                epi.prio[0, :3] = scores
                epi.step[0, :3] = steps
                epi.vals[0, 0, 0] = mark_new
                epi.vals[0, 1, 0] = mark_old
                epi.vals[0, 2, 0] = mark_mid

                sym = mem.sym_mem
                sym.filled.fill_(3)
                sym.prio.zero_()
                sym.step.zero_()
                sym.P_vals.zero_()
                sym.prio[:3] = scores
                sym.step[:3] = steps
                sym.P_vals[0, 0] = mark_new
                sym.P_vals[1, 0] = mark_old
                sym.P_vals[2, 0] = mark_mid

            bank = mem.ExportMemoryBank(topk=3)
            assert bank is not None

            for k in ("gws", "kv", "ltm_sem", "ltm_epi", "sym"):
                got = bank[k][0, :, 0]
                assert torch.allclose(got, expected), f"{k} export not latest-first: got {got.tolist()}"

            print("ExportMemoryBank latest-first-order test passed.")
            return True
        except AssertionError as e:
            print(f"ExportMemoryBank latest-first-order test failed: {e}")
            return False
        except Exception as e:
            print(f"ExportMemoryBank latest-first-order test error: {e}")
            return False

    def TestReorderMemorySteps(self):
        try:
            cfg = dict(
                inputDim=32,
                ssmStateDim=32,
                memoryDim=8,
                memorySize=8,
                symSize=8,
                ltmSize=8,
                nsK=4,
                outputDim=8,
                gwsSlots=4,
                gwsTtl=8,
                compressEvery=100,
                emotionDim=4,)

            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device)
            mem.ResetAll()

            with torch.no_grad():
                mem.time_step.fill_(30)
                mem.memory_filled.fill_(3)
                mem.memory_steps.zero_()
                mem.memory_steps[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)

                mem.gws.priority.zero_()
                mem.gws.ttl.zero_()
                mem.gws.last_step.zero_()
                mem.gws.global_step.fill_(30)
                mem.gws.priority[0, :3] = torch.tensor([3.0, 2.0, 1.0], device=self.device, dtype=mem.dtype)
                mem.gws.ttl[0, :3] = 1
                mem.gws.last_step[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)

                sem = mem.ltm.semantic
                sem.filled.fill_(3)
                sem.step.zero_()
                sem.global_step.fill_(30)
                sem.step[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)

                epi = mem.ltm.episodic
                epi.filled.fill_(3)
                epi.step.zero_()
                epi.global_step.fill_(30)
                epi.step[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)

                sym = mem.sym_mem
                sym.filled.fill_(3)
                sym.step.zero_()
                sym.global_step.fill_(30)
                sym.step[:3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)

                mem.memory_values.zero_()
                mem.memory_values[0, 0, 0] = 30.0
                mem.memory_values[0, 1, 0] = 10.0
                mem.memory_values[0, 2, 0] = 20.0

                mem.gws.vals.zero_()
                mem.gws.vals[0, 0, 0] = 30.0
                mem.gws.vals[0, 1, 0] = 10.0
                mem.gws.vals[0, 2, 0] = 20.0

                sem.vals.zero_()
                sem.vals[0, 0, 0] = 30.0
                sem.vals[0, 1, 0] = 10.0
                sem.vals[0, 2, 0] = 20.0

                epi.vals.zero_()
                epi.vals[0, 0, 0] = 30.0
                epi.vals[0, 1, 0] = 10.0
                epi.vals[0, 2, 0] = 20.0

                sym.P_vals.zero_()
                sym.P_vals[0, 0] = 30.0
                sym.P_vals[1, 0] = 10.0
                sym.P_vals[2, 0] = 20.0

            mem.ReorderMemorySteps()

            expect_local = torch.tensor([3, 1, 2], device=self.device, dtype=torch.long)
            if not torch.equal(mem.memory_steps[0, :3], expect_local):
                raise AssertionError(f"kv steps mismatch: {mem.memory_steps[0, :3].tolist()}")
            if not torch.equal(mem.gws.last_step[0, :3], expect_local):
                raise AssertionError(f"gws steps mismatch: {mem.gws.last_step[0, :3].tolist()}")
            if not torch.equal(mem.ltm.semantic.step[0, :3], expect_local):
                raise AssertionError(f"ltm_sem steps mismatch: {mem.ltm.semantic.step[0, :3].tolist()}")
            if not torch.equal(mem.ltm.episodic.step[0, :3], expect_local):
                raise AssertionError(f"ltm_epi steps mismatch: {mem.ltm.episodic.step[0, :3].tolist()}")
            if not torch.equal(mem.sym_mem.step[:3], expect_local):
                raise AssertionError(f"sym steps mismatch: {mem.sym_mem.step[:3].tolist()}")

            if int(mem.time_step[0].item()) != 3:
                raise AssertionError(f"time_step={int(mem.time_step[0].item())}, expected 3")
            if int(mem.gws.global_step[0].item()) != 3:
                raise AssertionError(f"gws.global_step={int(mem.gws.global_step[0].item())}, expected 3")
            if int(mem.ltm.semantic.global_step[0].item()) != 3:
                raise AssertionError(f"ltm_sem.global_step={int(mem.ltm.semantic.global_step[0].item())}, expected 3")
            if int(mem.ltm.episodic.global_step[0].item()) != 3:
                raise AssertionError(f"ltm_epi.global_step={int(mem.ltm.episodic.global_step[0].item())}, expected 3")
            if int(mem.sym_mem.global_step.item()) != 3:
                raise AssertionError(f"sym.global_step={int(mem.sym_mem.global_step.item())}, expected 3")

            bank = mem.ExportMemoryBank(topk=3)
            assert bank is not None
            expected = torch.tensor([30.0, 20.0, 10.0], device=self.device, dtype=mem.dtype)
            for k in ("gws", "kv", "ltm_sem", "ltm_epi", "sym"):
                got = bank[k][0, :, 0]
                assert torch.allclose(got, expected), f"{k} export not latest-first after reset: got {got.tolist()}"

            print("ReorderMemorySteps test passed.")
            return True
        except AssertionError as e:
            print(f"ReorderMemorySteps test failed: {e}")
            return False
        except Exception as e:
            print(f"ReorderMemorySteps test error: {e}")
            return False

    def TestMemoryExtractorForward(self):
        try:
            cfg = dict(
                inputDim=64,
                ssmStateDim=64,
                memoryDim=96,
                memorySize=32,
                symSize=64,
                ltmSize=64,
                nsK=32,
                outputDim=96,
                gwsSlots=8,
                gwsTtl=6,
                compressEvery=50,
                emotionDim=32,)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device)
            mem.eval()

            B = 4
            x = torch.randn(B, cfg.get("inputDim", 64), device=self.device)
            td = torch.randn(B, device=self.device)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem)

            y = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)
            assert y.shape == (B, cfg.get("outputDim", int(mem.output_dim)))

            assert int(mem.time_step[0].item()) >= 1
            assert (mem.memory_filled >= 0).all()
            assert (mem.memory_filled <= int(mem.memory_size)).all()

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
            cfg = dict(
                inputDim=48,
                ssmStateDim=48,
                memoryDim=64,
                memorySize=24,
                symSize=48,
                ltmSize=48,
                nsK=24,
                outputDim=64,
                gwsSlots=8,
                gwsTtl=6,
                compressEvery=10_000,
                emotionDim=16,)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device)
            mem.eval()

            path = self.StatePath("memory_state_test.pth")
            mem.SaveState(str(path))

            B = 3
            torch.manual_seed(123)
            x = torch.randn(B, cfg.get("inputDim", 48), device=self.device)
            td = torch.randn(B, device=self.device)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem)

            y1 = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)

            _ = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)

            mem.LoadState(str(path))
            try:
                path.unlink()
            except OSError:
                pass

            y2 = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)

            self.AssertClose(y1, y2, atol=5e-5, rtol=5e-4, msg="save/restore output")
            print("MemoryExtractor state save/restore test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor state test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor state test error: {e}")
            return False

    def TestExportImportStateRoundTrip(self):
        try:
            cfg = dict(
                inputDim=32,
                ssmStateDim=32,
                memoryDim=48,
                memorySize=16,
                symSize=32,
                ltmSize=32,
                nsK=16,
                outputDim=48,
                gwsSlots=6,
                gwsTtl=4,
                compressEvery=10_000,
                emotionDim=16,)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)
            mem1 = MemoryExtractor(**cfg).to(self.device).eval()

            B = 1
            x = torch.randn(B, cfg.get("inputDim", 32), device=self.device)
            td = torch.randn(B, device=self.device)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem1)

            for _ in range(5):
                _ = self.CallMemForward(mem1, x, tdError=td, reward=rwd, emotion=emotion)

            mem1.pending.clear()
            state = mem1.ExportState()

            mem2 = MemoryExtractor(**cfg).to(self.device).eval()
            mem2.load_state_dict(mem1.state_dict(), strict=True)
            mem2.ImportState(state, importGws=True, importLtm=True, importSym=True)

            y1 = self.CallMemForward(mem1, x, tdError=td, reward=rwd, emotion=emotion)
            y2 = self.CallMemForward(mem2, x, tdError=td, reward=rwd, emotion=emotion)

            self.AssertClose(y1, y2, atol=1e-4, rtol=1e-3, msg="Export/Import output")
            print("ExportState/ImportState round-trip test passed.")
            return True
        except AssertionError as e:
            print(f"Export/Import round-trip test failed: {e}")
            return False
        except Exception as e:
            print(f"Export/Import round-trip test error: {e}")
            return False

    def TestAutoCompress(self):
        try:
            cfg = dict(
                inputDim=16,
                ssmStateDim=16,
                memoryDim=24,
                memorySize=10, 
                symSize=20,
                ltmSize=20,
                nsK=12,
                outputDim=24,
                gwsSlots=4,
                gwsTtl=4,
                compressEvery=5,
                emotionDim=8,)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)
            mem = MemoryExtractor(**cfg).to(self.device).eval()

            B = 1
            in_dim = cfg.get("inputDim", 16)
            td = torch.ones(B, device=self.device)
            rwd = torch.zeros(B, device=self.device)

            for t in range(30):
                x = torch.randn(B, in_dim, device=self.device)
                emotion = self.MakeEmotion(B, mem)
                _ = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)

            filled = int(mem.memory_filled[0].item())
            assert 0 <= filled <= int(mem.memory_size)
            assert torch.isfinite(mem.memory_keys).all()
            assert torch.isfinite(mem.memory_values).all()

            print("AutoCompress smoke test passed.")
            return True
        except AssertionError as e:
            print(f"AutoCompress test failed: {e}")
            return False
        except Exception as e:
            print(f"AutoCompress test error: {e}")
            return False

    def TestResetAndSoftReset(self):
        try:
            cfg = dict(
                inputDim=32, ssmStateDim=32, memoryDim=48, memorySize=16,
                symSize=32, ltmSize=32, nsK=16,
                outputDim=48, gwsSlots=6, gwsTtl=6, compressEvery=10_000, emotionDim=16)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device).train()

            B = 4
            x = torch.randn(B, cfg.get("inputDim", 32), device=self.device)
            td = torch.randn(B, device=self.device)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem)

            _ = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)

            fw_before = mem.fast_weights.detach().clone()
            h_before = mem.h_state.detach().clone()

            mem.SoftReset()

            assert torch.count_nonzero(mem.fast_weights).item() == 0
            assert torch.count_nonzero(mem.h_state).item() == 0
            assert int(mem.time_step[0].item()) == 0
            assert int(mem.memory_filled[0].item()) == 0

            mem.ResetAll()
            assert torch.count_nonzero(mem.memory_keys).item() == 0
            assert torch.count_nonzero(mem.memory_values).item() == 0
            assert int(mem.time_step[0].item()) == 0
            assert int(mem.memory_filled[0].item()) == 0

            gws_snap = mem.gws.Inspect()
            assert torch.count_nonzero(gws_snap["priority"]).item() == 0
            assert torch.count_nonzero(mem.ltm.semantic.filled).item() == 0
            assert torch.count_nonzero(mem.ltm.episodic.filled).item() == 0
            assert int(mem.sym_mem.filled.item()) == 0

            assert torch.linalg.norm(fw_before).item() >= 0 and torch.linalg.norm(h_before).item() >= 0

            print("MemoryExtractor Reset/SoftReset test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor Reset/SoftReset test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor Reset/SoftReset test error: {e}")
            return False

    def TrainStepSmoke(self):
        try:
            cfg = dict(
                inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32,
                symSize=64, ltmSize=64, nsK=32,
                outputDim=96, gwsSlots=8, gwsTtl=6, compressEvery=10_000, emotionDim=32)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device).train()
            opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

            B = 8
            x = torch.randn(B, cfg.get("inputDim", 64), device=self.device)
            td = torch.randn(B, device=self.device)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem)

            target = torch.randn(B, cfg.get("outputDim", int(mem.output_dim)), device=self.device)

            out = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)
            base = F.mse_loss(out, target)
            total = self.AttachAllInternalLosses(mem, base)

            opt.zero_grad()
            total.backward()

            for n, p in mem.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

            torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
            opt.step()

            print("TrainStepSmoke passed.")
            return True
        except AssertionError as e:
            print(f"TrainStepSmoke failed: {e}")
            return False
        except Exception as e:
            print(f"TrainStepSmoke error: {e}")
            return False

    @torch.no_grad()
    def NumericalStabilityProbe(
        self,
        mem,
        *,
        steps: int = 200,
        batch: int = 4,
        eps: float = 1e-6,
        seed: int = 123,
        print_every: int = 25,):
        device = self.device
        mem = mem.to(device).eval()

        rng = torch.Generator(device=device).manual_seed(seed)

        warmup = 5
        in_dim = int(mem.B_mat.in_features)
        for _ in range(warmup):
            xw = torch.randn(batch, in_dim, generator=rng, device=device)
            emotion = self.MakeEmotionGen(batch, mem, generator=rng)
            _ = self.CallMemForward(mem, xw, emotion=emotion)

        cfg = dict(
            inputDim=int(mem.B_mat.in_features),
            ssmStateDim=int(mem.ssm_state_dim),
            memoryDim=int(mem.memory_dim),
            memorySize=int(mem.memory_size),
            symSize=int(getattr(mem, "sym_capacity", 64)),
            ltmSize=int(getattr(mem, "ltm_cap", 64)),
            nsK=int(getattr(mem, "ns_K", 32)),
            outputDim=int(mem.output_dim),
            gwsSlots=int(mem.gws.slots),
            gwsTtl=int(mem.gws.default_ttl),
            compressEvery=int(getattr(mem, "compress_every", 10_000)),
            emotionDim=int(mem.emotion_dim),)
        
        cfg = self.FilterKwargs(MemoryExtractor, cfg)

        mem2 = MemoryExtractor(**cfg).to(device).eval()

        tmp_path = self.StatePath("memory_state_probe.pth")
        mem.SaveState(str(tmp_path))
        mem2.LoadState(str(tmp_path))
        try:
            tmp_path.unlink()
        except OSError:
            pass

        cos_hist = []
        for t in range(steps):
            x = torch.randn(batch, in_dim, generator=rng, device=device)
            x2 = x + eps * self.RandnLikeGen(x, generator=rng)

            emotion = self.MakeEmotionGen(batch, mem, generator=rng)

            y1 = self.CallMemForward(mem, x, emotion=emotion)
            y2 = self.CallMemForward(mem2, x2, emotion=emotion)

            c = F.cosine_similarity(y1, y2, dim=-1).mean().item()
            cos_hist.append(c)

            if (t + 1) % print_every == 0 or t == 0:
                print(f"[probe] step {t+1:4d}/{steps}, cosine={c:.6f}")

        cos_np = np.array(cos_hist, dtype=np.float64)
        assert np.isfinite(cos_np).all(), "cosine became non-finite"
        assert (cos_np >= -1.001).all() and (cos_np <= 1.001).all(), f"cosine left valid range: min={cos_np.min():.3f}, max={cos_np.max():.3f}"

        return list(cos_hist)

    def TestNumericalStability(self):
        try:
            cfg = dict(
                inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=48,
                symSize=64, ltmSize=64, nsK=32,
                outputDim=96, gwsSlots=8, gwsTtl=6, compressEvery=1000, emotionDim=32)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device)
            cos_hist = self.NumericalStabilityProbe(mem, steps=100, batch=4, eps=1e-6, seed=123, print_every=25)

            for c in cos_hist:
                assert math.isfinite(c), "cosine became non-finite"

            print("MemoryExtractor numerical stability test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor numerical stability test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor numerical stability test error: {e}")
            return False
        
    def TestAllTrainableParamsHaveGrad(self):
        try:
            cfg = dict(
                inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32,
                symSize=64, ltmSize=64, nsK=32,
                outputDim=96, gwsSlots=8, gwsTtl=6, compressEvery=10_000, emotionDim=32)

            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device).train()

            if hasattr(mem, "fusion") and hasattr(mem.fusion, "noisy_gating"):
                mem.fusion.noisy_gating = False
            if hasattr(mem, "fusion") and hasattr(mem.fusion, "expert_dropout"):
                mem.fusion.expert_dropout = 0.0

            opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

            B = 8
            x1 = torch.randn(B, cfg.get("inputDim", 64), device=self.device)
            td1 = torch.randn(B, device=self.device)
            rwd1 = torch.randn(B, device=self.device)
            emotion1 = self.MakeEmotion(B, mem)

            x2 = torch.randn(B, cfg.get("inputDim", 64), device=self.device)
            td2 = torch.randn(B, device=self.device)
            rwd2 = torch.randn(B, device=self.device)
            emotion2 = self.MakeEmotion(B, mem)

            target = torch.randn(B, cfg.get("outputDim", int(mem.output_dim)), device=self.device)

            with torch.no_grad():
                _ = self.CallMemForward(mem, x1, tdError=td1, reward=rwd1, emotion=emotion1)

            out = self.CallMemForward(mem, x2, tdError=td2, reward=rwd2, emotion=emotion2)
            base = F.mse_loss(out, target)
            total = self.AttachAllInternalLosses(mem, base)

            opt.zero_grad(set_to_none=True)
            total.backward()

            missing, nonfinite = [], []
            for n, p in mem.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    missing.append(n)
                    continue
                if not torch.isfinite(p.grad).all():
                    nonfinite.append(n)

            if missing:
                raise AssertionError("Missing grad params:\n" + "\n".join(missing))
            if nonfinite:
                raise AssertionError("Non-finite grad params:\n" + "\n".join(nonfinite))

            print("All-trainable-params grad test passed. (2-step warmup)")
            return True

        except AssertionError as e:
            print(f"All-trainable-params grad test failed: {e}")
            return False
        except Exception as e:
            print(f"All-trainable-params grad test error: {e}")
            return False
        
    def TestLossDecreases(self):
        try:
            cfg = dict(
                inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32,
                symSize=64, ltmSize=64, nsK=32,
                outputDim=96, gwsSlots=8, gwsTtl=6,
                compressEvery=10_000, emotionDim=32)
            
            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device).train()

            if hasattr(mem, "fusion"):
                if hasattr(mem.fusion, "noisy_gating"):
                    mem.fusion.noisy_gating = False
                if hasattr(mem.fusion, "expert_dropout"):
                    mem.fusion.expert_dropout = 0.0
                if hasattr(mem.fusion, "noise_std"):
                    mem.fusion.noise_std = 0.0

            g = torch.Generator(device=self.device).manual_seed(12345)

            B = 16
            in_dim = int(cfg.get("inputDim", 64))
            out_dim = int(cfg.get("outputDim", int(mem.output_dim)))

            x = torch.randn(B, in_dim, device=self.device, generator=g)
            td = torch.randn(B, device=self.device, generator=g)
            rwd = torch.randn(B, device=self.device, generator=g)
            emotion = torch.randn(B, int(mem.emotion_dim), device=self.device, generator=g)

            target = torch.randn(B, out_dim, device=self.device, generator=g)

            opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

            steps = 60
            base_hist, total_hist = [], []

            for _ in range(steps):
                out = self.CallMemForward(mem, x,tdError=td, reward=rwd, emotion=emotion,reset=True)

                base = F.mse_loss(out, target)
                total = self.AttachAllInternalLosses(mem, base)

                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
                opt.step()

                base_hist.append(float(base.detach().item()))
                total_hist.append(float(total.detach().item()))

            k = 10
            base_head = sum(base_hist[:k]) / k
            base_tail = sum(base_hist[-k:]) / k

            total_head = sum(total_hist[:k]) / k
            total_tail = sum(total_hist[-k:]) / k

            if not (base_tail < base_head * 0.85):
                raise AssertionError(
                    f"base loss did not decrease enough: "
                    f"head={base_head:.6f}, tail={base_tail:.6f}, ratio={base_tail/base_head:.3f}")

            print(
                f"LossDecrease test passed. "
                f"base head={base_head:.6f} -> tail={base_tail:.6f}; "
                f"total head={total_head:.6f} -> tail={total_tail:.6f}")
            
            return True

        except AssertionError as e:
            print(f"LossDecrease test failed: {e}")
            return False
        except Exception as e:
            print(f"LossDecrease test error: {e}")
            return False

    def RunAll(self):
        results = {
            "StableTopkTieBreak": self.TestStableTopkTieBreak(),
            "GlobalWorkspace": self.TestGlobalWorkspace(),
            "GlobalWorkspaceEviction": self.TestGlobalWorkspaceEvictionBehavior(),
            "LongTermMemory": self.TestLongTermMemory(),
            "EpisodicEvictTouch": self.TestEpisodicEvictionTouchPrefersKeepHighTouch(),
            "EpisodicEvictAbsReward": self.TestEpisodicEvictionAbsRewardForNegative(),
            "ExportMemoryBankLatestFirst": self.TestExportMemoryBankLatestFirstOrder(),
            "ReorderMemorySteps": self.TestReorderMemorySteps(),
            "MemoryExtractorForward": self.TestMemoryExtractorForward(),
            "StateSaveRestore": self.TestStateSaveRestore(),
            "ExportImportRoundTrip": self.TestExportImportStateRoundTrip(),
            "AutoCompress": self.TestAutoCompress(),
            "ResetAndSoftReset": self.TestResetAndSoftReset(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NumericalStability": self.TestNumericalStability(),
            "AllTrainableParamsHaveGrad": self.TestAllTrainableParamsHaveGrad(),
            "LossDecreases": self.TestLossDecreases()}

        passed = sum(1 for v in results.values() if v)
        print(f"\nMemory module tests: {passed}/{len(results)} passed.")
        return results
