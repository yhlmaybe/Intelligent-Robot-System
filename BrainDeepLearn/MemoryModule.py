from __future__ import annotations
from typing import Any, Optional, Tuple, Dict, List, Union
from pathlib import Path
from types import SimpleNamespace
from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import inspect
import tempfile



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


def SourceConfidence(
    source: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
    real: float = 1.0,
    mixed: float = 0.90,
    imagine: float = 0.72,) -> torch.Tensor:
    conf = torch.full(source.shape, float(real), device=source.device, dtype=dtype)
    conf = torch.where(source == MemoryType.SRC_MIXED, conf.new_tensor(float(mixed)), conf)
    conf = torch.where(source == MemoryType.SRC_IMAGINE, conf.new_tensor(float(imagine)), conf)
    return conf


def UnifiedMemoryScore(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    age: torch.Tensor,
    priority: Optional[torch.Tensor] = None,
    touch: Optional[torch.Tensor] = None,
    confidence: Optional[torch.Tensor] = None,
    source: Optional[torch.Tensor] = None,
    rewardAbs: Optional[torch.Tensor] = None,
    novelty: Optional[torch.Tensor] = None,
    validMask: Optional[torch.Tensor] = None,
    baseAgeBeta: float = 0.003,
    mixedExtraBeta: float = 0.006,
    imagineExtraBeta: float = 0.018,) -> torch.Tensor:
    if keys.dim() == 2:
        sim = torch.matmul(query, keys.t())
    elif keys.dim() == 3:
        sim = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
    else:
        raise ValueError(f"keys must be 2D or 3D, got {keys.dim()}D")

    age_f = age
    if source is None:
        source = torch.zeros_like(age, dtype=torch.int8)
    src = source

    beta = torch.full_like(age_f, float(baseAgeBeta))
    beta = torch.where(src == MemoryType.SRC_MIXED, beta + float(mixedExtraBeta), beta)
    beta = torch.where(src == MemoryType.SRC_IMAGINE, beta + float(imagineExtraBeta), beta)
    freshness = torch.exp(-beta * age_f)

    if priority is None:
        salience = torch.ones_like(age_f)
    else:
        salience = priority.float().clamp_min(0.0)
    if rewardAbs is not None:
        salience = salience * (1.0 + 0.5 * torch.tanh(rewardAbs.float().clamp_min(0.0)))
    if novelty is not None:
        salience = salience * (1.0 + 0.25 * torch.tanh(novelty.float().clamp_min(0.0)))

    if touch is None:
        rehearsal = torch.ones_like(age_f)
    else:
        rehearsal = 1.0 + 0.1 * torch.log1p(touch.float().clamp_min(0.0))

    if confidence is None:
        conf = SourceConfidence(src, dtype=sim.dtype)
    else:
        conf = confidence.float()

    score = sim * freshness * salience * rehearsal * conf
    if validMask is not None:
        score = score.masked_fill(~validMask, -1e9)
    return score

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

        self.excl_logits = nn.Parameter(torch.empty(self.G_excl, self.K))
        self.or_logits = nn.Parameter(torch.empty(self.G_or, self.K))
        self.imp_logits = nn.Parameter(torch.full((self.K, self.K), -4.0))

        self.register_buffer("no_self_mask", ~torch.eye(self.K, dtype=torch.bool), persistent=True)

        if self.excl_logits.numel() > 0:
            nn.init.normal_(self.excl_logits, mean=0.0, std=float(initStd))
        if self.or_logits.numel() > 0:
            nn.init.normal_(self.or_logits, mean=0.0, std=float(initStd))

        if seedDisjoint:
            self.SeedDisjointInit()

    @torch.no_grad()
    def SeedDisjointInit(self):
        if (self.K <= 0):
            return

        perm = torch.randperm(self.K, device=self.device)
        cursor = 0

        if self.excl_logits.numel() > 0 and self.G_excl > 0:
            self.excl_logits.fill_(-2.0)
            m = max(1, int(round(self.mass_excl)))
            for g in range(self.G_excl):
                idx = perm[cursor:cursor + m]
                if idx.numel() < m:
                    idx = torch.cat([idx, perm[:(m - idx.numel())]], dim=0)
                self.excl_logits[g, idx] = 2.0
                cursor = (cursor + m) % self.K

        if self.or_logits.numel() > 0 and self.G_or > 0:
            self.or_logits.fill_(-2.0)
            m = max(1, int(round(self.mass_or)))
            for g in range(self.G_or):
                idx = perm[cursor:cursor + m]
                if idx.numel() < m:
                    idx = torch.cat([idx, perm[:(m - idx.numel())]], dim=0)
                self.or_logits[g, idx] = 2.0
                cursor = (cursor + m) % self.K

    def Weights(self):
        W_excl = torch.sigmoid(self.excl_logits) #[G_excl, K]
        W_or = torch.sigmoid(self.or_logits) #[G_or, K]

        A_imp = torch.sigmoid(self.imp_logits) #[K, K]
        A_imp = A_imp.masked_fill(~self.no_self_mask, 0.0) 

        return W_excl, W_or, A_imp

    def forward(self, p: torch.Tensor, pPrev: Optional[torch.Tensor] = None):
        p_f = p.float() #[B, K]
        B, K = int(p_f.size(0)), int(p_f.size(1))

        total_penalty = p_f.new_zeros(B)
        aux_reg = p_f.new_zeros(())

        W_excl, W_or, A_imp = self.Weights()#[G_excl, K] , [G_or, K] , [K, K]

        if W_excl.numel() > 0:
            s = torch.matmul(p_f, W_excl.t()) #[B, G_excl]
            s2 = torch.matmul(p_f.pow(2), W_excl.pow(2).t())
            excl_pen = 0.5 * (s.pow(2) - s2) 
            total_penalty = total_penalty + excl_pen.mean(dim=1) #[B]

        if W_or.numel() > 0:
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
            src = self.source[:filled]
            eff = self.prio[:filled] * torch.exp(-0.01 * age.float())
            eff = eff * SourceConfidence(src, dtype=eff.dtype)
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

        age = (self.global_step - self.step[:filled]).clamp(min=0).float()
        prio = self.prio[:filled]
        touch = self.touch[:filled]
        src = self.source[:filled]
        sim = UnifiedMemoryScore(
            qSym,
            stored_keys,
            age=age.view(1, filled).expand(qSym.size(0), filled),
            priority=prio.view(1, filled).expand(qSym.size(0), filled),
            touch=touch.view(1, filled).expand(qSym.size(0), filled),
            source=src.view(1, filled).expand(qSym.size(0), filled),
            baseAgeBeta=float(recentBias),
            mixedExtraBeta=0.02,
            imagineExtraBeta=0.05,)

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
    def EnsureB(self, B: int):
        B0 = int(self.keys.size(0))
        if B0 == int(B):
            return

        self.keys = self.keys.new_zeros(B, self.slots, self.dim)
        self.vals = self.vals.new_zeros(B, self.slots, self.dim)
        self.priority = self.priority.new_zeros(B, self.slots)
        self.ttl = self.ttl.new_zeros(B, self.slots)
        self.last_step = self.last_step.new_zeros(B, self.slots)
        self.source = self.source.new_zeros(B, self.slots)
        self.global_step = self.global_step.new_zeros(B)


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

        self.EnsureB(B)

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

        out_idx = torch.zeros(B, device=self.device, dtype=torch.long)
        for i in range(B):
            empty = (self.ttl[0] <= 0) | (self.priority[0] <= 0)
            if bool(empty.any().item()):
                idx = int(empty.float().argmax().item())
            else:
                age = (self.global_step[0] - self.last_step[0]).clamp(min=0).float()
                eff = self.priority[0] * torch.exp(-age * self.recency_temp)
                eff = eff * SourceConfidence(self.source[0], dtype=eff.dtype)
                idx = int(torch.argmin(eff).item())

            self.keys[0, idx] = key[i]
            self.vals[0, idx] = val[i]
            self.priority[0, idx] = pr[i]
            self.ttl[0, idx] = ttl_t[i]
            self.last_step[0, idx] = self.global_step[0]
            self.source[0, idx] = tag_t[i]
            out_idx[i] = idx

        return out_idx # [B]

    def Attend(
        self,
        query: torch.Tensor, # [B, Dim]
        *,
        topk: int = 4,
        tagMask: Optional[List[int]] = None,) -> torch.Tensor:

        B = int(query.size(0))

        self.EnsureB(B)

        alive_one = (self.ttl[:1] > 0) & (self.priority[:1] > 0) #[1, slots]

        if tagMask is not None:
            allowed = torch.zeros_like(alive_one, dtype=torch.bool)
            for t in tagMask:
                allowed |= (self.source[:1] == int(t))
            alive_one = alive_one & allowed

        alive = alive_one.expand(B, self.slots)
        any_alive = alive.any(dim=1)

        q = query #[B, dim]
        k = self.keys[:1].expand(B, self.slots, self.dim) #[B, slots, dim]
        age = (self.global_step[:1].view(1, 1) - self.last_step[:1]).clamp(min=0).float().expand(B, self.slots)
        sim = UnifiedMemoryScore(
            q,
            k,
            age=age,
            priority=self.priority[:1].expand(B, self.slots),
            source=self.source[:1].expand(B, self.slots),
            validMask=alive,
            baseAgeBeta=float(self.recency_temp),
            mixedExtraBeta=float(self.recency_temp) * 0.5,
            imagineExtraBeta=float(self.recency_temp),)

        kk = max(1, min(int(topk), self.slots))

        if not bool(any_alive.all()):
            sim = sim.clone()
            sim[~any_alive] = 0.0

        top_sim, top_idx = StableTopk(sim, kk)  #[B, kk]
        w = F.softmax(top_sim, dim=-1)

        gather_idx = top_idx.unsqueeze(-1).expand(B, kk, self.dim)
        v_top = torch.gather(self.vals[:1].expand(B, self.slots, self.dim), dim=1, index=gather_idx) #[B, kk, dim]

        out = torch.einsum("bk,bkd->bd", w, v_top) #[B, dim]

        if not bool(any_alive.all()):
            out = out.clone()
            out[~any_alive] = 0.0

        with torch.no_grad():
            flat_idx = top_idx.reshape(-1)
            if flat_idx.numel() > 0:
                self.last_step[0, flat_idx] = self.global_step[0]

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
    def EnsureB(self, B: int):
        B0 = int(self.keys.size(0))
        if B0 == int(B):
            return

        self.keys = self.keys.new_zeros(B, self.capacity, self.dim)
        self.vals = self.vals.new_zeros(B, self.capacity, self.dim)
        self.prio = self.prio.new_zeros(B, self.capacity)
        self.touch = self.touch.new_zeros(B, self.capacity)
        self.step = self.step.new_zeros(B, self.capacity)
        self.source = self.source.new_zeros(B, self.capacity)
        self.filled = self.filled.new_zeros(B)
        self.global_step = self.global_step.new_zeros(B)


    @torch.no_grad()
    def StepTick(self):
        self.global_step.add_(1)

    @torch.no_grad()
    def Store(
        self,
        key: torch.Tensor,  # [B, D] 
        value: torch.Tensor,  # [B, D] 
        score: torch.Tensor, # [B,]
        source: Optional[torch.Tensor] = None,
        writeMask: Optional[torch.Tensor] = None): #[int = MemoryType.SRC_REAL]

        B = int(key.size(0))
        self.EnsureB(B)

        if source is None:
            source = torch.full((B,), MemoryType.SRC_REAL, device=self.device, dtype=torch.int8)
        if writeMask is None:
            writeMask = torch.ones(B, device=self.device, dtype=torch.bool)
        if not bool(writeMask.any().item()):
            return

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

        b_sel = b_idx[writeMask]
        idx_sel = idx[writeMask]
        self.keys[b_sel, idx_sel] = key[writeMask]
        self.vals[b_sel, idx_sel] = value[writeMask]
        self.prio[b_sel, idx_sel] = score[writeMask]

        self.touch[b_sel, idx_sel] = 1
        self.step[b_sel, idx_sel] = gstep[writeMask]
        self.source[b_sel, idx_sel] = source[writeMask]

        filled_new = torch.where(is_full, filled, filled + 1)
        self.filled = torch.where(writeMask, filled_new, filled)

    def Retrieve(
        self, 
        query: torch.Tensor, #[B, D]
        topk: int = 8):
        B = int(query.size(0))
        self.EnsureB(B)

        filled = self.filled
        gstep = self.global_step

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0) #[1, capacity] 
        valid_mask = slots < filled.unsqueeze(1) #[B, capacity]
        any_valid = valid_mask.any(dim=1)  
        if not bool(any_valid.any().item()):
            return torch.zeros(B, self.dim, device=self.device, dtype=query.dtype)

        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float() #[B, capacity]
        sim = UnifiedMemoryScore(
            query,
            self.keys,
            age=age,
            priority=self.prio,
            touch=self.touch,
            source=self.source,
            validMask=valid_mask,
            baseAgeBeta=0.001,
            mixedExtraBeta=0.004,
            imagineExtraBeta=0.012,)
        sim = torch.where(any_valid.view(B, 1), sim, torch.zeros_like(sim))

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
        self.register_buffer("state_keys", torch.zeros(B0, capacity, dim))
        self.register_buffer("vals", torch.zeros(B0, capacity, dim))
        self.register_buffer("rew", torch.zeros(B0, capacity))
        self.register_buffer("rew_abs", torch.zeros(B0, capacity))
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
    def EnsureB(self, B: int):
        B0 = int(self.keys.size(0))
        if B0 == int(B):
            return

        self.keys = self.keys.new_zeros(B, self.capacity, self.dim)
        self.state_keys = self.state_keys.new_zeros(B, self.capacity, self.dim)
        self.vals = self.vals.new_zeros(B, self.capacity, self.dim)
        self.rew = self.rew.new_zeros(B, self.capacity)
        self.rew_abs = self.rew_abs.new_zeros(B, self.capacity)
        self.prio = self.prio.new_zeros(B, self.capacity)
        self.step = self.step.new_zeros(B, self.capacity)
        self.touch = self.touch.new_zeros(B, self.capacity)
        self.source = self.source.new_zeros(B, self.capacity)
        self.filled = self.filled.new_zeros(B)
        self.global_step = self.global_step.new_zeros(B)

    @torch.no_grad()
    def Store(
        self,
        key: torch.Tensor, # [B, D],
        value: torch.Tensor, # [B, D]
        reward: torch.Tensor, # [B]
        score: torch.Tensor, # [B]
        source: Optional[torch.Tensor] = None,
        writeMask: Optional[torch.Tensor] = None,
        stateKey: Optional[torch.Tensor] = None): #  [int = MemoryType.SRC_REAL]
        B = int(key.size(0))
        self.EnsureB(B)

        if source is None:
            source = torch.full((B,), MemoryType.SRC_REAL, device=self.device, dtype=torch.int8)
        if writeMask is None:
            writeMask = torch.ones(B, device=self.device, dtype=torch.bool)
        if stateKey is None:
            stateKey = key
        if not bool(writeMask.any().item()):
            return

        filled = self.filled 
        gstep = self.global_step  

        is_full = filled >= self.capacity
        idx_append = filled.clamp(max=self.capacity - 1)

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0)  
        valid = slots < filled.unsqueeze(1) 

        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float()  
        freshness = torch.exp(-age * 0.01)
        touch_w = torch.log1p(self.touch.float() + 1.0) 
        eff = (self.prio + 0.5 * self.rew_abs) * freshness * touch_w
        eff = eff * SourceConfidence(self.source, dtype=eff.dtype)
        eff = torch.where(valid, eff, torch.full_like(eff, float("inf")))

        idx_evict = torch.argmin(eff, dim=1)  
        idx = torch.where(is_full, idx_evict, idx_append)  

        b_idx = torch.arange(B, device=self.device)

        b_sel = b_idx[writeMask]
        idx_sel = idx[writeMask]
        self.keys[b_sel, idx_sel] = key[writeMask]
        self.state_keys[b_sel, idx_sel] = stateKey[writeMask]
        self.vals[b_sel, idx_sel] = value[writeMask]
        self.rew[b_sel, idx_sel] = reward[writeMask]
        self.rew_abs[b_sel, idx_sel] = reward[writeMask].abs()

        self.prio[b_sel, idx_sel] = score[writeMask]

        self.step[b_sel, idx_sel] = gstep[writeMask]
        self.touch[b_sel, idx_sel] = 1
        self.source[b_sel, idx_sel] = source[writeMask]
        filled_new = torch.where(is_full, filled, filled + 1)
        self.filled = torch.where(writeMask, filled_new, filled)

    def Retrieve(self, query: torch.Tensor, topk: int = 8, recentBias: float = 0.05, useStateKey: bool = False):
        B = int(query.size(0))
        self.EnsureB(B)

        filled = self.filled
        gstep = self.global_step

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0) 
        valid_mask = slots < filled.unsqueeze(1) 
        any_valid = valid_mask.any(dim=1)
        if not bool(any_valid.any().item()):
            return torch.zeros(B, self.dim, device=self.device, dtype=query.dtype)

        keys = self.state_keys if bool(useStateKey) else self.keys
        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float()
        sim = UnifiedMemoryScore(
            query,
            keys,
            age=age,
            priority=self.prio,
            touch=self.touch,
            source=self.source,
            rewardAbs=self.rew_abs,
            validMask=valid_mask,
            baseAgeBeta=float(recentBias),
            mixedExtraBeta=0.02,
            imagineExtraBeta=0.05,)
        sim = torch.where(any_valid.view(B, 1), sim, torch.zeros_like(sim))

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
        self.episodic.state_keys.zero_()
        self.episodic.vals.zero_()
        self.episodic.rew.zero_()
        self.episodic.rew_abs.zero_()
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

    def Retrieve(self, query: torch.Tensor, topkSem: int = 6, topkEpi: int = 4, epiQuery: Optional[torch.Tensor] = None):
        sem_out = self.semantic.Retrieve(query, topk=topkSem) #[B, D]
        epi_out = self.episodic.Retrieve(query if epiQuery is None else epiQuery, topk=topkEpi) #[B, D]

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
            importance = a.mean(dim=0)
            self.aux_loss  = float(self.numExperts) * (importance.pow(2).sum())
        else:
            self.aux_loss = x.new_zeros(())

        ys = [expert(x) for expert in self.experts] 
        y = torch.stack(ys, dim=-1) # [B, outDim, numExperts]
        out = (y * a.unsqueeze(1)).sum(dim=-1) 
        return out # [B, outDim]
    

class ObjectUsageBank(AGICoreModule):
    """Per-instance tensorized affordance table (Part 6, Option C + Gaussian residual).

    Indexed by integer object/skill ids. Stores applicability, default parameters,
    expected slot-tensor deltas, Beta-distributed success priors, and a per-(skill,
    instance) Gaussian over the continuous parameter vector for online refinement.
    Novel objects are bootstrapped by Robo-ABC cosine retrieval over identity
    descriptors. No knowledge graph, no language tokens.
    """

    def __init__(
        self,
        numObjects: int = ModuleDim.UsageNumObjects,
        numSkills: int = ModuleDim.UsageNumSkills,
        paramDim: int = ModuleDim.UsageParamDim,
        idDim: int = ModuleDim.PstIdDim,
        slotDeltaDim: int = ModuleDim.PstSlotDim,
        usageDim: int = ModuleDim.PstUsageDim,
        attrDim: int = ModuleDim.PstAttrDim,):
        super().__init__()
        self.num_objects = int(numObjects)
        self.num_skills = int(numSkills)
        self.param_dim = int(paramDim)
        self.attr_dim = int(attrDim)
        self.usage_dim = int(usageDim)

        self.register_buffer("applicable", torch.zeros(self.num_objects, self.num_skills))
        self.register_buffer("default_params", torch.zeros(self.num_objects, self.num_skills, self.param_dim))
        self.register_buffer("expected_dx", torch.zeros(self.num_objects, self.num_skills, int(slotDeltaDim)))
        self.register_buffer("success_alpha", torch.ones(self.num_objects, self.num_skills))
        self.register_buffer("success_beta", torch.ones(self.num_objects, self.num_skills))
        self.register_buffer("param_mu", torch.zeros(self.num_objects, self.num_skills, self.param_dim))
        self.register_buffer("param_logvar", torch.zeros(self.num_objects, self.num_skills, self.param_dim))
        self.register_buffer("instance_descriptors", F.normalize(torch.randn(self.num_objects, int(idDim)), dim=-1))
        # Per-(object, skill) attribute centroid: lets attribute deltas refine the readout
        # so we don't reduce to identity-only lookup.
        self.register_buffer("attribute_centroid", torch.zeros(self.num_objects, self.num_skills, self.attr_dim))

        self.readout_proj = nn.Linear(
            self.param_dim + self.attr_dim + 4,
            self.usage_dim)
        self.needs_lookup = nn.Linear(4, 1)

    def SuccessRate(self) -> torch.Tensor:
        return self.success_alpha / (self.success_alpha + self.success_beta)

    def Confidence(self) -> torch.Tensor:
        total = self.success_alpha + self.success_beta
        return self.applicable * total / (total + 10.0)

    def NearestObject(self, descriptor: torch.Tensor) -> torch.Tensor:
        """Robo-ABC retrieval: cosine NN over identity descriptors. descriptor [..., D_c]."""
        sim = torch.matmul(F.normalize(descriptor, dim=-1), self.instance_descriptors.t())
        return sim.argmax(dim=-1)

    def BestObjectsForSkill(self, skillId: int) -> torch.Tensor:
        return self.applicable[:, int(skillId)].argmax()

    def SlotReadout(
        self,
        identity: torch.Tensor,
        attribute: torch.Tensor,
        slotAttention: torch.Tensor,) -> torch.Tensor:
        """Per-slot top-1 skill summary cached into PST.U. identity [B,K,D_c] -> [B,K,D_u].
        Attribute residuals against the bank's centroid let attribute deltas refine the readout."""
        B, K, _ = identity.shape
        obj_idx = self.NearestObject(identity)                       # [B,K]
        applicable_rows = self.applicable[obj_idx]                   # [B,K,N_skills]
        best_skill = applicable_rows.argmax(dim=-1)                  # [B,K]
        gather = lambda t: t[obj_idx.reshape(-1), best_skill.reshape(-1)].view(B, K, -1)
        params = gather(self.default_params)                        # [B,K,P]
        attribute_centroid = gather(self.attribute_centroid)        # [B,K,A]
        attribute_residual = attribute - attribute_centroid
        success = self.SuccessRate()[obj_idx, best_skill].unsqueeze(-1)
        confidence = self.Confidence()[obj_idx, best_skill].unsqueeze(-1)
        best_applicable = applicable_rows.gather(-1, best_skill.unsqueeze(-1))
        lookup_need = self.NeedsLookupScore(
            best_applicable.squeeze(-1),
            confidence.squeeze(-1),
            success.squeeze(-1),
            slotAttention).unsqueeze(-1)
        summary = torch.cat([
            params,
            attribute_residual,
            success,
            confidence,
            best_applicable,
            lookup_need], dim=-1)
        return self.readout_proj(summary)

    def NeedsLookupScore(
        self,
        applicability: torch.Tensor,
        confidence: torch.Tensor,
        successRate: torch.Tensor,
        slotAttention: torch.Tensor,) -> torch.Tensor:
        feats = torch.stack([applicability, confidence, successRate, slotAttention], dim=-1)
        return torch.sigmoid(self.needs_lookup(feats)).squeeze(-1)

    @torch.no_grad()
    def ExecutionOutcome(
        self,
        objId: torch.Tensor,
        skillId: torch.Tensor,
        success: torch.Tensor,
        observedParams: torch.Tensor,
        observedAttributes: Optional[torch.Tensor] = None,
        momentum: float = 0.9,) -> None:
        """Online update: Beta posterior on success and Bayesian moment matching on params.
        Also tracks attribute centroid per (object, skill) when attributes are supplied."""
        obj = objId.reshape(-1).long()
        skill = skillId.reshape(-1).long()
        succ = success.reshape(-1).to(self.success_alpha.dtype)
        self.success_alpha[obj, skill] += succ
        self.success_beta[obj, skill] += (1.0 - succ)
        mu = self.param_mu[obj, skill]
        new_mu = momentum * mu + (1.0 - momentum) * observedParams.view(mu.shape)
        var = self.param_logvar[obj, skill].exp()
        new_var = momentum * var + (1.0 - momentum) * (observedParams.view(mu.shape) - new_mu).square()
        self.param_mu[obj, skill] = new_mu
        self.param_logvar[obj, skill] = new_var.clamp_min(1e-6).log()
        if observedAttributes is not None:
            centroid = self.attribute_centroid[obj, skill]
            self.attribute_centroid[obj, skill] = (
                momentum * centroid + (1.0 - momentum) * observedAttributes.view(centroid.shape))


class MemoryExtractor(AGICoreModule):
    DURABLE_MEMORY_ARTIFACT_TYPE = "MemoryExtractorDurableMemory"
    DURABLE_MEMORY_SCHEMA_VERSION = 1
    DURABLE_MEMORY_STATE_FIELDS = (
        "time_step",
        "memory_filled",
        "memory_version",
        "memory_keys",
        "memory_values",
        "memory_importance",
        "memory_steps",
        "memory_emotion",
        "memory_source",
        "memory_reward_abs",
        "ltm_sem_global_step",
        "ltm_sem_keys",
        "ltm_sem_vals",
        "ltm_sem_prio",
        "ltm_sem_touch",
        "ltm_sem_step",
        "ltm_sem_filled",
        "ltm_sem_source",
        "ltm_epi_global_step",
        "ltm_epi_keys",
        "ltm_epi_state_keys",
        "ltm_epi_vals",
        "ltm_epi_prio",
        "ltm_epi_rew",
        "ltm_epi_rew_abs",
        "ltm_epi_step",
        "ltm_epi_touch",
        "ltm_epi_filled",
        "ltm_epi_source",
        "sym_mem_global_step",
        "sym_mem_P_keys",
        "sym_mem_P_vals",
        "sym_mem_prio",
        "sym_mem_step",
        "sym_mem_touch",
        "sym_mem_filled",
        "sym_mem_source",
        "usage_applicable",
        "usage_default_params",
        "usage_expected_dx",
        "usage_success_alpha",
        "usage_success_beta",
        "usage_param_mu",
        "usage_param_logvar",
        "usage_instance_descriptors",
        "usage_attribute_centroid",)
    TRANSIENT_MEMORY_STATE_FIELDS = (
        "last_compress_step",
        "h_state",
        "fast_weights",
        "ns_prev_P_post",
        "ns_penalty_vec",
        "gws_global_step",
        "gws_keys",
        "gws_vals",
        "gws_priority",
        "gws_ttl",
        "gws_last_step",
        "gws_source",)
    FULL_MEMORY_STATE_FIELDS = (
        "time_step",
        "memory_filled",
        "last_compress_step",
        "h_state",
        "fast_weights",
        "ns_prev_P_post",
        "ns_penalty_vec",
        "gws_global_step",
        "ltm_sem_global_step",
        "ltm_epi_global_step",
        "sym_mem_global_step",
        "memory_version",
        "memory_keys",
        "memory_values",
        "memory_importance",
        "memory_steps",
        "memory_emotion",
        "memory_source",
        "memory_reward_abs",
        "gws_keys",
        "gws_vals",
        "gws_priority",
        "gws_ttl",
        "gws_last_step",
        "gws_source",
        "ltm_sem_keys",
        "ltm_sem_vals",
        "ltm_sem_prio",
        "ltm_sem_touch",
        "ltm_sem_step",
        "ltm_sem_filled",
        "ltm_sem_source",
        "ltm_epi_keys",
        "ltm_epi_state_keys",
        "ltm_epi_vals",
        "ltm_epi_prio",
        "ltm_epi_rew",
        "ltm_epi_rew_abs",
        "ltm_epi_step",
        "ltm_epi_touch",
        "ltm_epi_filled",
        "ltm_epi_source",
        "sym_mem_P_keys",
        "sym_mem_P_vals",
        "sym_mem_prio",
        "sym_mem_step",
        "sym_mem_touch",
        "sym_mem_filled",
        "sym_mem_source",)

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
        topk: int = 8,
        gwsSlots: int = 24,
        gwsTtl: int = 64,
        compressEvery: int = 8192,
        emotionDim: int = 512,) -> None:
        super().__init__()

        self.ssm_state_dim = ssmStateDim
        self.input_dim = inputDim
        self.memory_dim = memoryDim
        self.output_dim = outputDim
        self.memory_size = memorySize
        self.topk = min(topk, memorySize)

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
        self.register_buffer(
            "fast_weights",
            torch.zeros(B0, memoryDim, memoryDim),
            persistent=False)
        self.register_buffer("memory_keys", torch.zeros(B0, memorySize, memoryDim))
        self.register_buffer("memory_values", torch.zeros(B0, memorySize, memoryDim))
        self.register_buffer("memory_importance", torch.zeros(B0, memorySize))
        self.register_buffer("memory_steps", torch.zeros(B0, memorySize, dtype=torch.long))
        self.register_buffer("memory_source", torch.zeros(B0, memorySize, dtype=torch.int8))
        self.register_buffer("memory_reward_abs", torch.zeros(B0, memorySize))
        self.register_buffer("memory_version", torch.zeros((), dtype=torch.long))

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

        self.sem_context_mod = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, 2 * self.memory_dim),)

        epi_mod_in_dim = self.memory_dim + 1024 + 5
        self.epi_event_mod = nn.Sequential(
            nn.LayerNorm(epi_mod_in_dim),
            nn.Linear(epi_mod_in_dim, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, 2 * self.memory_dim),)

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
        self.gws_h_proj = nn.Sequential(
            nn.LayerNorm(ssmStateDim),
            nn.Linear(ssmStateDim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.gws_y_proj = nn.Sequential(
            nn.LayerNorm(outputDim),
            nn.Linear(outputDim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.gws_val_proj = nn.Sequential(
            nn.LayerNorm(memoryDim),
            nn.Linear(memoryDim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.gws_source_gate = nn.Sequential(
            nn.LayerNorm(memoryDim * 3 + 3),
            nn.Linear(memoryDim * 3 + 3, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, 3),)
        nn.init.zeros_(self.gws_source_gate[-1].weight)
        nn.init.zeros_(self.gws_source_gate[-1].bias)
        self.gws_summary_norm = nn.LayerNorm(memoryDim)

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
        self.sym_query_fusion = nn.Sequential(
            nn.LayerNorm(self.ns_K * 2),
            nn.Linear(self.ns_K * 2, self.ns_K),
            nn.Sigmoid(),)
        nn.init.zeros_(self.sym_query_fusion[1].weight)
        nn.init.zeros_(self.sym_query_fusion[1].bias)
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
        self.film_context_proj = nn.Sequential(
            nn.LayerNorm(self.memory_dim * 4 + 4),
            nn.Linear(self.memory_dim * 4 + 4, self.memory_dim),
            nn.SiLU(),
            nn.Linear(self.memory_dim, self.memory_dim),)

        self.extra_losses: List[torch.Tensor] = []

        self.pending = []

        self.visual_context_proj = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, inputDim),
            nn.SiLU(),
            nn.Linear(inputDim, inputDim),)
        self.ocr_context_proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, inputDim),
            nn.SiLU(),
            nn.Linear(inputDim, inputDim),)
        self.intent_context_proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, inputDim),
            nn.SiLU(),
            nn.Linear(inputDim, inputDim),)
        self.context_fuse = nn.Sequential(
            nn.LayerNorm(inputDim * 4),
            nn.Linear(inputDim * 4, inputDim),
            nn.SiLU(),
            nn.Linear(inputDim, inputDim),)
        self.context_fuse_scale = nn.Parameter(torch.tensor(0.1))

        event_in_dim = inputDim + 2048 + 512 + 512 + self.emotion_dim + 4
        self.event_context_proj = nn.Sequential(
            nn.LayerNorm(event_in_dim),
            nn.Linear(event_in_dim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.event_completion_gate = nn.Sequential(
            nn.Linear(memoryDim * 3 + 1, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, 1),
            nn.Sigmoid(),)

        # Object usage knowledge bank, alongside working/episodic/semantic/spatial/skill banks.
        self.usage_bank = ObjectUsageBank()

    @torch.no_grad()
    def EnsureB(self, B: int) -> None:
        B0 = int(self.h_state.size(0))
        if B0 == int(B):
            return

        self.h_state = self.h_state.new_zeros(B, self.ssm_state_dim)

        self.fast_weights = self.fast_weights.new_zeros(
            B, self.memory_dim, self.memory_dim)

        self.memory_keys = self.memory_keys.new_zeros(
            B, self.memory_size, self.memory_dim)
        self.memory_values = self.memory_values.new_zeros(
            B, self.memory_size, self.memory_dim)

        self.memory_importance = self.memory_importance.new_zeros(
            B, self.memory_size)
        self.memory_steps = self.memory_steps.new_zeros(B, self.memory_size)
        self.memory_source = self.memory_source.new_zeros(B, self.memory_size)
        self.memory_reward_abs = self.memory_reward_abs.new_zeros(
            B, self.memory_size)

        self.memory_emotion = self.memory_emotion.new_zeros(
            B, self.memory_size, self.emotion_dim)

        self.time_step = self.time_step.new_zeros(B)
        self.memory_filled = self.memory_filled.new_zeros(B)
        self.last_compress_step = self.last_compress_step.new_zeros(B)

        self.ns_prev_P_post = self.ns_prev_P_post.new_zeros(B, self.ns_K)
        self.ns_penalty_vec = self.ns_penalty_vec.new_zeros(B, 1)

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
            self.AddInternalLoss(
                self.ns_lambda * (total_penalty.mean() + aux_reg))

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

    def BuildFilmContext(
        self,
        writeValue: torch.Tensor,
        recallValue: torch.Tensor,
        modSignal: torch.Tensor,
        tdMemory: torch.Tensor,
        risk: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,) -> torch.Tensor:
        ctx = torch.cat([
            writeValue,
            recallValue,
            writeValue - recallValue,
            modSignal,
            tdMemory.unsqueeze(-1),
            risk.unsqueeze(-1),
            uncertainty.unsqueeze(-1),
            confidence.unsqueeze(-1)], dim=-1)
        return self.film_context_proj(ctx)

    def BuildGwsValue(
        self,
        hNew: torch.Tensor,
        ySsm: torch.Tensor,
        val: torch.Tensor,
        tdMemory: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        h_part = self.gws_h_proj(hNew)
        y_part = self.gws_y_proj(ySsm)
        val_part = self.gws_val_proj(val)
        gate_in = torch.cat([
            h_part,
            y_part,
            val_part,
            tdMemory.unsqueeze(-1),
            risk.unsqueeze(-1),
            confidence.unsqueeze(-1)], dim=-1)
        weight = F.softmax(self.gws_source_gate(gate_in), dim=-1)
        out = (
            weight[:, 0:1] * h_part
            + weight[:, 1:2] * y_part
            + weight[:, 2:3] * val_part)
        return self.gws_summary_norm(out + 0.1 * val), self.gws_summary_norm(out)

    def FuseExternalContext(
        self,
        x: torch.Tensor,
        visualState: Any,
        ocrSemantic: torch.Tensor,
        intentHint: torch.Tensor,) -> torch.Tensor:
        vis = torch.cat([
            visualState.IntegratedFeat,
            visualState.MotionToken,
            visualState.PredErrorToken], dim=-1)
        visual_ctx = self.visual_context_proj(vis)
        ocr_ctx = self.ocr_context_proj(ocrSemantic)
        intent_ctx = self.intent_context_proj(intentHint)

        fused = self.context_fuse(torch.cat([x, visual_ctx, ocr_ctx, intent_ctx], dim=-1))
        return x + torch.tanh(self.context_fuse_scale) * fused

    def PatternSeparate(self, x: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.memory_dim) // 8)
        _, idx = torch.topk(x.abs(), k=k, dim=-1)
        sparse = torch.zeros_like(x)
        sparse.scatter_(1, idx, torch.gather(x, 1, idx))
        return F.normalize(sparse, dim=-1)

    def BuildEventCode(
        self,
        x: torch.Tensor,
        visualState: Any,
        ocrSemantic: torch.Tensor,
        intentHint: torch.Tensor,
        emotion: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        vis = torch.cat([
            visualState.IntegratedFeat,
            visualState.MotionToken,
            visualState.PredErrorToken], dim=-1)
        raw = self.event_context_proj(torch.cat([
            x,
            vis,
            ocrSemantic,
            intentHint,
            emotion,
            tdError.view(-1, 1),
            uncertainty.view(-1, 1),
            risk.view(-1, 1),
            confidence.view(-1, 1),], dim=-1))
        return raw, self.PatternSeparate(raw)

    def forward(self,
        x: torch.Tensor, # [B, inputDim]
        tdError: torch.Tensor, # [B] [-1, 1]
        emotion: torch.Tensor, # [B, emotionDim]
        reward: torch.Tensor, # [B] 
        visualState: Any,
        ocrSemantic: torch.Tensor,
        intentHint: torch.Tensor,
        uncertainty: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        *,
        reset: bool = False,
        softReset: bool = False,
        sourceLabel: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.training:
            self.ResetInternalLoss()

        B = x.size(0)

        self.EnsureB(B)

        src_all = sourceLabel
        if sourceLabel is None:
            src_all = torch.full((B,), MemoryType.SRC_REAL, dtype=torch.int8, device=self.device) # [B]

        emotion_eff = emotion
        tdError_eff = tdError
        reward_eff = reward
        uncertainty_eff = uncertainty
        risk_eff = risk
        confidence_eff = confidence

        write_strength = (
            (0.65 + 0.55 * confidence_eff)
            * (1.0 - 0.45 * uncertainty_eff).clamp(0.35, 1.0)
            + 0.15 * risk_eff).clamp(0.25, 1.50)
        td_memory = tdError_eff * (0.50 + 0.50 * confidence_eff)
        reward_abs_eff = reward_eff.detach().abs() * (1.0 + 0.25 * risk_eff)

        if reset:
            self.ResetAll()
        elif softReset:
            self.SoftReset()

        x = self.FuseExternalContext(
            x,
            visualState=visualState,
            ocrSemantic=ocrSemantic,
            intentHint=intentHint)

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
        event_dense, event_key = self.BuildEventCode(
            x,
            visualState=visualState,
            ocrSemantic=ocrSemantic,
            intentHint=intentHint,
            emotion=emotion_eff,
            tdError=td_memory,
            uncertainty=uncertainty_eff,
            risk=risk_eff,
            confidence=confidence_eff)

        emo_emb = self.emo_write_proj(emotion_eff) # [B, memoryDim]

        mod = self.emo_val_mod(emo_emb) # [B, 2*memoryDim]
        gamma, beta = mod.chunk(2, dim=-1) # [B, memoryDim]
        gamma = torch.tanh(gamma)

        val_mod = (1 + gamma) * val + beta # [B, memoryDim]

        alpha = 0.25 * torch.tanh(self.emo_write_alpha)
        val = val_mod + alpha * emo_emb # [B, memoryDim]

        sem_context = torch.cat([
            visualState.IntegratedFeat,
            ocrSemantic,
            intentHint], dim=-1)

        sem_mod = self.sem_context_mod(sem_context)
        sem_scale, sem_shift = sem_mod.chunk(2, dim=-1)
        sem_scale = torch.tanh(sem_scale)
        sem_mod_signal = sem_shift + 0.1 * sem_scale

        epi_mod_context = torch.cat([
            event_dense,
            visualState.MotionToken,
            visualState.PredErrorToken,
            td_memory.unsqueeze(-1),
            reward_eff.unsqueeze(-1),
            risk_eff.unsqueeze(-1),
            uncertainty_eff.unsqueeze(-1),
            confidence_eff.unsqueeze(-1)], dim=-1)
        epi_mod = self.epi_event_mod(epi_mod_context)
        epi_scale, epi_shift = epi_mod.chunk(2, dim=-1)
        epi_scale = torch.tanh(epi_scale)
        epi_mod_signal = epi_shift + 0.1 * epi_scale

        sem_in = val * (1 + sem_scale) + sem_shift # [B, memoryDim]
        epi_in = val * (1 + epi_scale) + epi_shift # [B, memoryDim]

        self.h_state = h_mix.detach().clone()

        importance = self.importance_net(h_mix).squeeze(-1) # [B]
        importance_eff = (importance * write_strength).clamp(0.0, 1.50) # [B]
        gate_local = self.local_gate(h_mix).squeeze(-1) # [B]

        kv_feat = self.KvStats(key) # [B, 3]

        phi = torch.cat([self.ctrl_norm(h_mix), emo_emb, key, kv_feat, importance_eff.unsqueeze(-1), gate_local.unsqueeze(-1), td_memory.unsqueeze(-1)], dim=-1)

        ctrl = self.ctrl_head(phi) # [B, 4]
        a_raw, b_raw, f_raw, bias_raw = ctrl.split(1, dim=-1) # [B, 1]

        a = (0.5 + 0.5 * torch.sigmoid(a_raw)).squeeze(-1) # [B]
        b = (0.9 + 0.1 * torch.sigmoid(b_raw)).squeeze(-1) # [B]
        fusion_gate = torch.sigmoid(f_raw).squeeze(-1) # [B]
        gate_bias = 0.5 * torch.tanh(bias_raw).squeeze(-1) # [B]

        mem_recall = self.Retrieve(key, fusion_gate, importance=importance_eff, localGate=gate_local, emotion=emotion_eff, tdError=td_memory,) # [B, memoryDim]
        self.HebbianUpdate(key, gate_local, td_memory, a, b)
        
        mem_film_ctx = self.BuildFilmContext(val, mem_recall, emo_emb, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        g2, b2 = self.FilmParams(self.film_mem, mem_film_ctx)
        s2 = 1.0 + g2

        mem_state = self.mem_film_norm(mem_recall * s2 + b2)

        self.pending.append(("kv",
                             (key.detach(),val.detach(),importance_eff.detach(),emotion_eff.detach(),reward_abs_eff.detach(),src_all.detach())))

        gws_val, gws_mod_signal = self.BuildGwsValue(h_new, y_ssm, val, td_memory, risk_eff, confidence_eff) # [B, memoryDim]

        gws_recall = self.gws.Attend(key, topk=1) # [B, memoryDim]

        affect_mag = td_memory.abs()
        prio = importance_eff * (1.0 + 0.5 * affect_mag + 0.25 * risk_eff).clamp(0.5, 2.0)

        gws_film_ctx = self.BuildFilmContext(gws_val, gws_recall, gws_mod_signal, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        g1, b1 = self.FilmParams(self.film_gws, gws_film_ctx)
        s1 = 1.0 + g1

        gws_state = self.gws_film_norm(gws_recall * s1 + b1) # [B, memoryDim]

        ttl = torch.full((B,), 6, device=self.device, dtype=torch.long)
        ttl = torch.where(src_all == MemoryType.SRC_REAL, torch.full_like(ttl, 10), ttl)
        ttl = torch.where(src_all == MemoryType.SRC_IMAGINE, torch.full_like(ttl, 4), ttl)

        self.pending.append(("gws",
                             (key.detach(),gws_val.detach(),prio.detach(),ttl.detach(),src_all.detach())))
        

        sem_recall, epi_event_recall = self.ltm.Retrieve(key, topkSem=self.ltm_topk_sem, topkEpi=self.ltm_topk_epi, epiQuery=event_key) # [B, memoryDim]
        epi_dense_recall = self.ltm.episodic.Retrieve(key, topk=self.ltm_topk_epi, useStateKey=True)
        completion_gate = self.event_completion_gate(torch.cat([epi_event_recall, epi_dense_recall, event_dense, td_memory.view(B, 1)], dim=-1))
        epi_recall = completion_gate * epi_event_recall + (1.0 - completion_gate) * epi_dense_recall + 0.05 * completion_gate * event_dense

        sem_film_ctx = self.BuildFilmContext(sem_in, sem_recall, sem_mod_signal, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        epi_film_ctx = self.BuildFilmContext(epi_in, epi_recall, epi_mod_signal, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        g3, b3 = self.FilmParams(self.film_sem, sem_film_ctx)
        g4, b4 = self.FilmParams(self.film_epi, epi_film_ctx)

        s3 = 1.0 + g3
        s4 = 1.0 + g4

        sem_state = self.sem_film_norm(sem_recall * s3 + b3) # [B, memoryDim]
        epi_state = self.epi_film_norm(epi_recall * s4 + b4) # [B, memoryDim]

        self.pending.append(("ltm",
                              (key.detach(),event_key.detach(),key.detach(),
                               sem_in.detach(),epi_in.detach(),
                               importance_eff.detach(),td_memory.detach(),reward_eff.detach(),src_all.detach(),
                               uncertainty_eff.detach(),risk_eff.detach(),confidence_eff.detach())))

        ltm_fused = self.ltm.fuser(sem_state, epi_state)

        P_post = self.NsPostRead(val)

        Qsym_key = self.sym_query(key) # [B, nsK]
        qsym_mix = self.sym_query_fusion(torch.cat([Qsym_key, P_post], dim=-1)) # [B, nsK]
        Qsym = qsym_mix * Qsym_key + (1.0 - qsym_mix) * P_post
        Qsym = F.normalize(Qsym, dim=-1)

        self.pending.append(("ns",
                             (Qsym.detach(), P_post.detach(), importance_eff.detach(), src_all.detach())))

        sym_recall = self.sym_mem.Retrieve(Qsym, topK=8) 

        sym_vec = self.sym_embed(P_post, sym_recall) # [B, memoryDim]

        fused_state = self.fusion(torch.cat([mem_state, gws_state, ltm_fused, sym_vec], dim=-1)) # [B, outputDim]

        fused_state = self.ApplyOutputGate(fused_state, td_memory, gate_bias)

        if self.training:
            self.AddInternalLoss(self.fusion.GetAuxLoss())

        step0 = int(self.time_step[0].item())

        if (step0 % self.compress_every) == 0:
            self.pending.append(("compress", None))
        if (step0 % max(1, self.compress_every // 4)) == 0:
            self.pending.append(("consolidate", None))

        return fused_state

    
    def ApplyOutputGate(self, memRecall: torch.Tensor, tdError: torch.Tensor, gateBias: torch.Tensor) -> torch.Tensor:
        gate_delta = torch.tanh(tdError + gateBias).view(-1, 1)
        return memRecall + 0.3 * gate_delta * memRecall


    @torch.no_grad()
    def HebbianUpdate(
        self,
        key: torch.Tensor, # [B, memory_dim]
        gateLocal: torch.Tensor, # [B]
        neuromod: torch.Tensor, # [B]
        a: torch.Tensor, # [B]
        b: torch.Tensor # [B]
        ) -> None:

        B = int(key.size(0))

        key_d = key.detach()

        a3 = a.view(B, 1, 1)
        b3 = b.view(B, 1, 1)
        g3 = gateLocal.view(B, 1, 1)
        n3 = neuromod.view(B, 1, 1)

        outer = torch.bmm(key_d.unsqueeze(2), key_d.unsqueeze(1)) # [B, memory_dim, memory_dim]

        update = (n3 * 0.15) * a3 * g3 * outer # [B, memory_dim, memory_dim]

        new_weights = self.fast_weights * 0.95 * b3 + update

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
        rewardAbs: torch.Tensor, # [B]
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
        self.memory_reward_abs[b_indices, target_idx] = rewardAbs.detach().clamp_min(0.0)
        self.memory_source[b_indices, target_idx] = src_w

        new_filled = (filled + 1).clamp(max=M)
        self.memory_filled.copy_(torch.where(is_full, filled, new_filled))
        self.memory_version.add_(1)

    @torch.no_grad()
    def LtmOnlineStore(
        self, 
        keySem: torch.Tensor, # [B, memory_dim]
        keyEpi: torch.Tensor, # [B, memory_dim]
        keyEpiState: torch.Tensor, # [B, memory_dim]
        valSem: torch.Tensor, # [B, memory_dim]
        valEpi: torch.Tensor, # [B, memory_dim]
        importance: torch.Tensor, # [B]
        tdError: torch.Tensor, # [B]
        reward: torch.Tensor, # [B] [-10, 10]
        sourceLabel: torch.Tensor,
        uncertainty: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor):
        source_conf = SourceConfidence(sourceLabel, dtype=importance.dtype)
        unc_eff = uncertainty
        risk_eff = risk
        conf_eff = confidence
        value_gain = ((0.65 + 0.55 * conf_eff) * (1.0 - 0.45 * unc_eff).clamp(0.35, 1.0) + 0.15 * risk_eff).clamp(0.25, 1.50)

        salience = (importance + 0.35 * tdError.abs() + 0.15 * torch.tanh(reward.abs()) + 0.15 * risk_eff) * source_conf * value_gain
        mask_base = (salience > self.ltm_online_imp_thresh) | (tdError.abs() > self.ltm_online_td_thresh)
        is_imag = (sourceLabel == MemoryType.SRC_IMAGINE)
        is_mixed = (sourceLabel == MemoryType.SRC_MIXED)
        mask_imag = is_imag & (salience > self.ltm_online_imp_thresh * 1.4)
        mask_mixed = is_mixed & (salience > self.ltm_online_imp_thresh * 1.1)
        mask_real = (sourceLabel == MemoryType.SRC_REAL) & mask_base
        mask = (mask_real | mask_mixed | mask_imag)

        if not mask.any():
            return

        sem = self.ltm.semantic
        epi = self.ltm.episodic

        sem.Store(key=keySem, value=valSem, score=salience, source=sourceLabel, writeMask=mask)
        epi.Store(key=keyEpi, value=valEpi, reward=reward, score=salience, source=sourceLabel, writeMask=mask, stateKey=keyEpiState)
        self.memory_version.add_(1)

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

        fw = self.fast_weights.detach().clone() # [B, memory_dim, memory_dim]
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

            age = (self.time_step.view(B, 1) - steps).clamp(min=0).float() # [B, M]
            ar = torch.arange(M, device=self.device).view(1, M)
            valid_mask = ar < filled.view(B, 1) # [B, M]
            sim = UnifiedMemoryScore(
                query,
                keys,
                age=age,
                priority=imp_kv,
                source=src,
                rewardAbs=self.memory_reward_abs,
                validMask=valid_mask,
                baseAgeBeta=0.05,
                mixedExtraBeta=0.03,
                imagineExtraBeta=0.05,)
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
            mask_mixed = (src_sel == MemoryType.SRC_MIXED)

            w_real = attn * mask_real.float() # [B, k]
            w_imag = attn * (mask_imag.float() + 0.5 * mask_mixed.float()) # [B, k]
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

        scores = temp_imp * SourceConfidence(self.memory_source, dtype=temp_imp.dtype)
        scores = scores * (1.0 + 0.5 * torch.tanh(self.memory_reward_abs.clamp_min(0.0))) # [B, M]

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
        new_rew_abs = gather_and_trim(self.memory_reward_abs, sorted_idx, new_valid_mask, dim_last=False)

        m_b11 = cond_fill.view(B, 1, 1)
        m_b1 = cond_fill.view(B, 1)

        self.memory_keys = torch.where(m_b11, new_keys, self.memory_keys)
        self.memory_values = torch.where(m_b11, new_vals, self.memory_values)
        self.memory_emotion = torch.where(m_b11, new_emos, self.memory_emotion)

        self.memory_importance = torch.where(m_b1, new_imps, self.memory_importance)
        self.memory_steps = torch.where(m_b1, new_steps.long(), self.memory_steps)
        self.memory_source = torch.where(m_b1, new_srcs.to(torch.int8), self.memory_source)
        self.memory_reward_abs = torch.where(m_b1, new_rew_abs, self.memory_reward_abs)

        self.memory_filled = torch.where(cond_fill, keep_nums, self.memory_filled)
        self.last_compress_step = torch.where(cond_fill, self.time_step, self.last_compress_step)

    @torch.no_grad()
    def ConsolidateMemory(self, topk: int = 8) -> None:
        B = int(self.memory_filled.size(0))
        device = self.device
        dtype = self.dtype

        epi = self.ltm.episodic
        sem = self.ltm.semantic
        if int(epi.filled.max().item()) <= 0:
            return

        C = int(epi.capacity)
        slots = torch.arange(C, device=device).view(1, C)
        valid = slots < epi.filled.view(B, 1)
        age = (epi.global_step.view(B, 1) - epi.step).clamp(min=0).float()
        source_conf = SourceConfidence(epi.source, dtype=dtype)
        freshness = torch.exp(-0.001 * age)
        salience = epi.prio * (1.0 + 0.5 * torch.tanh(epi.rew_abs.clamp_min(0.0)))
        salience = salience * (1.0 + 0.1 * torch.log1p(epi.touch.float().clamp_min(0.0)))
        salience = salience * freshness * source_conf
        salience = salience.masked_fill(~valid, -1e9)
        any_valid = valid.any(dim=1)
        if not bool(any_valid.any().item()):
            return
        salience = torch.where(any_valid.view(B, 1), salience, torch.zeros_like(salience))
        k = max(1, min(int(topk), C, int(epi.filled.max().item())))
        top_score, top_idx = StableTopk(salience, k)

        idx3 = top_idx.unsqueeze(-1).expand(B, k, int(self.memory_dim))
        keys = torch.gather(epi.state_keys, 1, idx3)
        vals = torch.gather(epi.vals, 1, idx3)
        src = torch.gather(epi.source, 1, top_idx)
        score = top_score.to(dtype=dtype)
        keep = torch.isfinite(score) & (score > 0)
        if not bool(keep.any().item()):
            return

        for j in range(k):
            sem.Store(
                key=keys[:, j],
                value=vals[:, j],
                score=score[:, j],
                source=src[:, j],
                writeMask=keep[:, j],)

        sym_keep = keep & (src != MemoryType.SRC_IMAGINE)
        if bool(sym_keep.any().item()):
            vals_flat = vals.reshape(B * k, int(self.memory_dim))
            src_flat = src.reshape(B * k)
            score_flat = score.reshape(B * k)
            sym_keep_flat = sym_keep.reshape(B * k)
            P = self.ns_coder_post(vals_flat[sym_keep_flat])
            for i in range(P.size(0)):
                self.sym_mem.Store(
                    key=F.normalize(P[i].detach(), dim=0),
                    value=P[i].detach(),
                    score=float(score_flat[sym_keep_flat][i].item()),
                    source=int(src_flat[sym_keep_flat][i].item()),)
        self.memory_version.add_(1)


    def FlushPendingWrites(self):
        if not self.pending:
            return

        for kind, payload in self.pending:
            if kind == "gws":
                key, ws_val, prio, ttl, src = payload
                self.gws.Write(key, ws_val,priority=prio,ttl=ttl,tagId=src,)

            elif kind == "kv":
                key, val, imp, emo, rew_abs, src = payload
                self.KvWrite(key=key,val=val,importance=imp,emotion=emo,rewardAbs=rew_abs,source=src,)

            elif kind == "ltm":
                key_sem, key_epi, key_epi_state, sem, epi, imp, td, rwd, src, unc, risk, conf = payload
                self.LtmOnlineStore(keySem=key_sem,keyEpi=key_epi,keyEpiState=key_epi_state,valSem=sem,valEpi=epi,importance=imp,tdError=td,reward=rwd,sourceLabel=src,uncertainty=unc,risk=risk,confidence=conf,)
                
            elif kind == "ns":
                key, P_post, importance, src = payload
                self.NsStore(key, P_post, importance, src)

            elif kind == "compress":
                self.AutoCompress()

            elif kind == "consolidate":
                self.ConsolidateMemory()

        self.pending.clear()

    @torch.no_grad()
    def SoftReset(self):
        self.h_state.zero_()
        self.fast_weights.zero_()

        self.time_step.zero_()
        self.memory_filled.zero_()
        self.last_compress_step.zero_()
        self.memory_version.add_(1)

        self.ns_prev_P_post.zero_()
        self.ns_penalty_vec.zero_()
        self.ResetInternalLoss()
        self.pending.clear()

    @torch.no_grad()
    def ResetEpisodeState(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            done = torch.ones(
                self.h_state.size(0),
                device=self.h_state.device,
                dtype=torch.bool)
        else:
            if not torch.is_tensor(doneMask) or doneMask.dtype != torch.bool:
                raise TypeError("Memory episode reset mask must be a bool tensor")
            if doneMask.device != self.h_state.device:
                raise ValueError("Memory episode reset mask must be on the memory device")
            if doneMask.shape != self.h_state.shape[:1]:
                raise ValueError("Memory episode reset mask must have shape [B]")
            done = doneMask
        self.FlushPendingWrites()
        self.h_state[done] = 0
        self.fast_weights[done] = 0
        self.ns_prev_P_post[done] = 0
        self.ns_penalty_vec[done] = 0
        if bool(done.all().item()):
            self.gws.Reset()


    def ResetAll(self):
        self.h_state.zero_()
        self.fast_weights.zero_()

        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_importance.zero_()
        self.memory_steps.zero_()
        self.memory_emotion.zero_()
        self.memory_source.zero_()
        self.memory_reward_abs.zero_()

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
        self.memory_version.add_(1)

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.fast_weights.zero_()
            return
        done = doneMask.view(-1)
        if done.numel() != self.fast_weights.size(0):
            raise ValueError("Memory Hebbian reset mask must match its batch size")
        self.fast_weights[done] = 0

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
            "memory_reward_abs": torch.zeros_like(self.memory_reward_abs, device=dev),
            "memory_version": torch.zeros_like(self.memory_version, device=dev),

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
            "ltm_epi_state_keys": epi.state_keys.clone().zero_().to(dev),
            "ltm_epi_vals": epi.vals.clone().zero_().to(dev),
            "ltm_epi_prio": epi.prio.clone().zero_().to(dev),
            "ltm_epi_rew": epi.rew.clone().zero_().to(dev),
            "ltm_epi_rew_abs": epi.rew_abs.clone().zero_().to(dev),
            "ltm_epi_step": epi.step.clone().zero_().to(dev),
            "ltm_epi_touch": epi.touch.clone().zero_().to(dev),
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
            "ns_penalty_vec": self.ns_penalty_vec.clone().zero_(),

            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,}

        torch.save(state, path)

    @torch.no_grad()
    def SaveState(self, path: str):
        state = self.ExportDurableState()
        payload = {
            "artifact_type": self.DURABLE_MEMORY_ARTIFACT_TYPE,
            "schema_version": self.DURABLE_MEMORY_SCHEMA_VERSION,
            "batch_size": int(state["memory_filled"].size(0)),
            "state": state,}
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=directory or ".")
        os.close(fd)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @torch.no_grad()
    def LoadState(self, path: str, *, expectedBatch: Optional[int] = None):
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            return

        obj = torch.load(path, map_location=self.device, weights_only=True)
        expected_payload_fields = {"artifact_type", "schema_version", "batch_size", "state"}
        if type(obj) is not dict or set(obj) != expected_payload_fields:
            raise TypeError("memory artifact fields do not match the durable-memory schema")
        if obj["artifact_type"] != self.DURABLE_MEMORY_ARTIFACT_TYPE:
            raise ValueError(f"unsupported memory artifact type: {obj['artifact_type']!r}")
        if (
            type(obj["schema_version"]) is not int
            or obj["schema_version"] != self.DURABLE_MEMORY_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported durable-memory schema: {obj['schema_version']!r}")
        if type(obj["batch_size"]) is not int or obj["batch_size"] < 1:
            raise TypeError("durable-memory batch_size must be a positive integer")
        if expectedBatch is not None and obj["batch_size"] != int(expectedBatch):
            raise ValueError(
                f"durable memory batch_size={obj['batch_size']} does not match "
                f"runtime batchSize={expectedBatch}")
        state = obj["state"]
        self.ValidateDurableState(state, expectedBatch=obj["batch_size"])
        self.ImportDurableState(state)

    def DurableStateTensors(self) -> Dict[str, torch.Tensor]:
        sem = self.ltm.semantic
        epi = self.ltm.episodic
        sym = self.sym_mem
        usage = self.usage_bank
        return {
            "time_step": self.time_step,
            "memory_filled": self.memory_filled,
            "memory_version": self.memory_version,
            "memory_keys": self.memory_keys,
            "memory_values": self.memory_values,
            "memory_importance": self.memory_importance,
            "memory_steps": self.memory_steps,
            "memory_emotion": self.memory_emotion,
            "memory_source": self.memory_source,
            "memory_reward_abs": self.memory_reward_abs,
            "ltm_sem_global_step": sem.global_step,
            "ltm_sem_keys": sem.keys,
            "ltm_sem_vals": sem.vals,
            "ltm_sem_prio": sem.prio,
            "ltm_sem_touch": sem.touch,
            "ltm_sem_step": sem.step,
            "ltm_sem_filled": sem.filled,
            "ltm_sem_source": sem.source,
            "ltm_epi_global_step": epi.global_step,
            "ltm_epi_keys": epi.keys,
            "ltm_epi_state_keys": epi.state_keys,
            "ltm_epi_vals": epi.vals,
            "ltm_epi_prio": epi.prio,
            "ltm_epi_rew": epi.rew,
            "ltm_epi_rew_abs": epi.rew_abs,
            "ltm_epi_step": epi.step,
            "ltm_epi_touch": epi.touch,
            "ltm_epi_filled": epi.filled,
            "ltm_epi_source": epi.source,
            "sym_mem_global_step": sym.global_step,
            "sym_mem_P_keys": sym.P_keys,
            "sym_mem_P_vals": sym.P_vals,
            "sym_mem_prio": sym.prio,
            "sym_mem_step": sym.step,
            "sym_mem_touch": sym.touch,
            "sym_mem_filled": sym.filled,
            "sym_mem_source": sym.source,
            "usage_applicable": usage.applicable,
            "usage_default_params": usage.default_params,
            "usage_expected_dx": usage.expected_dx,
            "usage_success_alpha": usage.success_alpha,
            "usage_success_beta": usage.success_beta,
            "usage_param_mu": usage.param_mu,
            "usage_param_logvar": usage.param_logvar,
            "usage_instance_descriptors": usage.instance_descriptors,
            "usage_attribute_centroid": usage.attribute_centroid,}

    @torch.no_grad()
    def ExportDurableState(self) -> Dict[str, torch.Tensor]:
        state = {
            name: value.detach().clone()
            for name, value in self.DurableStateTensors().items()}
        if tuple(state) != self.DURABLE_MEMORY_STATE_FIELDS:
            raise RuntimeError("durable-memory field declaration and export order disagree")
        return state

    def ValidateDurableState(
        self,
        state: Dict[str, torch.Tensor],
        *,
        expectedBatch: Optional[int] = None,) -> None:
        if type(state) is not dict or set(state) != set(self.DURABLE_MEMORY_STATE_FIELDS):
            raise TypeError("durable-memory state fields do not match the current schema")
        if not all(type(state[name]) is torch.Tensor for name in self.DURABLE_MEMORY_STATE_FIELDS):
            raise TypeError("every durable-memory state field must be a tensor")

        memory_filled = state["memory_filled"]
        if memory_filled.dim() != 1 or int(memory_filled.size(0)) < 1:
            raise ValueError("durable-memory memory_filled must have shape [B] with B >= 1")
        batch_size = int(memory_filled.size(0))
        if expectedBatch is not None and batch_size != int(expectedBatch):
            raise ValueError("durable-memory batch_size does not match its state tensors")

        expected_shapes = {
            "time_step": (batch_size,),
            "memory_filled": (batch_size,),
            "memory_version": (),
            "memory_keys": (batch_size, self.memory_size, self.memory_dim),
            "memory_values": (batch_size, self.memory_size, self.memory_dim),
            "memory_importance": (batch_size, self.memory_size),
            "memory_steps": (batch_size, self.memory_size),
            "memory_emotion": (batch_size, self.memory_size, self.emotion_dim),
            "memory_source": (batch_size, self.memory_size),
            "memory_reward_abs": (batch_size, self.memory_size),
            "ltm_sem_global_step": (batch_size,),
            "ltm_sem_keys": (batch_size, self.ltm.semantic.capacity, self.memory_dim),
            "ltm_sem_vals": (batch_size, self.ltm.semantic.capacity, self.memory_dim),
            "ltm_sem_prio": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_touch": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_step": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_filled": (batch_size,),
            "ltm_sem_source": (batch_size, self.ltm.semantic.capacity),
            "ltm_epi_global_step": (batch_size,),
            "ltm_epi_keys": (batch_size, self.ltm.episodic.capacity, self.memory_dim),
            "ltm_epi_state_keys": (batch_size, self.ltm.episodic.capacity, self.memory_dim),
            "ltm_epi_vals": (batch_size, self.ltm.episodic.capacity, self.memory_dim),
            "ltm_epi_prio": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_rew": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_rew_abs": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_step": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_touch": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_filled": (batch_size,),
            "ltm_epi_source": (batch_size, self.ltm.episodic.capacity),
            "sym_mem_global_step": (),
            "sym_mem_P_keys": tuple(self.sym_mem.P_keys.shape),
            "sym_mem_P_vals": tuple(self.sym_mem.P_vals.shape),
            "sym_mem_prio": tuple(self.sym_mem.prio.shape),
            "sym_mem_step": tuple(self.sym_mem.step.shape),
            "sym_mem_touch": tuple(self.sym_mem.touch.shape),
            "sym_mem_filled": (),
            "sym_mem_source": tuple(self.sym_mem.source.shape),
            "usage_applicable": tuple(self.usage_bank.applicable.shape),
            "usage_default_params": tuple(self.usage_bank.default_params.shape),
            "usage_expected_dx": tuple(self.usage_bank.expected_dx.shape),
            "usage_success_alpha": tuple(self.usage_bank.success_alpha.shape),
            "usage_success_beta": tuple(self.usage_bank.success_beta.shape),
            "usage_param_mu": tuple(self.usage_bank.param_mu.shape),
            "usage_param_logvar": tuple(self.usage_bank.param_logvar.shape),
            "usage_instance_descriptors": tuple(self.usage_bank.instance_descriptors.shape),
            "usage_attribute_centroid": tuple(self.usage_bank.attribute_centroid.shape),}
        current_dtypes = {
            name: tensor.dtype
            for name, tensor in self.DurableStateTensors().items()}
        for name in self.DURABLE_MEMORY_STATE_FIELDS:
            value = state[name]
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"durable-memory field {name} has shape {tuple(value.shape)}, "
                    f"expected {expected_shapes[name]}")
            if value.dtype != current_dtypes[name]:
                raise TypeError(
                    f"durable-memory field {name} has dtype {value.dtype}, "
                    f"expected {current_dtypes[name]}")

    @torch.no_grad()
    def ImportDurableState(self, state: Dict[str, torch.Tensor]) -> None:
        self.ValidateDurableState(state)
        batch_size = int(state["memory_filled"].size(0))
        self.EnsureB(batch_size)
        self.ltm.semantic.EnsureB(batch_size)
        self.ltm.episodic.EnsureB(batch_size)

        targets = self.DurableStateTensors()
        for name in self.DURABLE_MEMORY_STATE_FIELDS:
            targets[name].copy_(state[name])

        self.h_state.zero_()
        self.fast_weights.zero_()
        self.ns_prev_P_post.zero_()
        self.ns_penalty_vec.zero_()
        self.last_compress_step.copy_(self.time_step)
        self.gws.EnsureB(batch_size)
        self.gws.Reset()
        self.ResetInternalLoss()
        self.pending.clear()

    @torch.no_grad()
    def ExportMemoryBank(
        self,
        topk: int = 1024,
        *,
        totalBudget: Optional[int] = None,
        perTypeBudget: Optional[Dict[str, int]] = None,
        includeMeta: bool = True,) -> Optional[Dict[str, torch.Tensor]]:
        device = self.device
        B = int(self.memory_filled.size(0)) # [B]
        topk = int(topk)

        if topk <= 0:
            return None

        type_keys = ("gws", "kv", "ltm_sem", "ltm_epi", "sym")
        if perTypeBudget is None:
            default_budget = topk if totalBudget is None else max(1, int(totalBudget) // len(type_keys))
            perTypeBudget = {k: default_budget for k in type_keys}
        else:
            perTypeBudget = {k: int(perTypeBudget.get(k, topk)) for k in type_keys}

        out: Dict[str, torch.Tensor] = {}
        meta: Dict[str, Dict[str, torch.Tensor]] = {}

        def GatherTopkLatestFirst(
            values: torch.Tensor,
            indices: torch.Tensor,
            steps: torch.Tensor,) -> torch.Tensor:
            if values.size(0) == 1 and B > 1:
                values = values.expand(B, *values.shape[1:])
            if steps.size(0) == 1 and B > 1:
                steps = steps.expand(B, *steps.shape[1:])
            sel_steps = torch.gather(steps, 1, indices)
            time_order = torch.argsort(sel_steps, dim=1, descending=True)
            time_indices = torch.gather(indices, 1, time_order)
            idx_exp = time_indices.unsqueeze(-1).expand(B, time_indices.size(1), values.size(-1))
            return torch.gather(values, 1, idx_exp).contiguous()

        def GatherMeta(
            values: torch.Tensor,
            indices: torch.Tensor,
            steps: torch.Tensor,) -> torch.Tensor:
            if values.size(0) == 1 and B > 1:
                values = values.expand(B, *values.shape[1:])
            if steps.size(0) == 1 and B > 1:
                steps = steps.expand(B, *steps.shape[1:])
            sel_steps = torch.gather(steps, 1, indices)
            time_order = torch.argsort(sel_steps, dim=1, descending=True)
            time_indices = torch.gather(indices, 1, time_order)
            return torch.gather(values, 1, time_indices).contiguous()

        S_gws = int(self.gws.slots)
        gws_valid = ((self.gws.ttl[:1] > 0) & (self.gws.priority[:1] > 0)).expand(B, S_gws)  # [B, S_gws]
        K_gws = min(perTypeBudget["gws"], S_gws, int(gws_valid.sum(dim=1).max().item()))
        if K_gws > 0:
            gws_age = (self.gws.global_step[:1].view(1, 1) - self.gws.last_step[:1]).clamp(min=0).float().expand(B, S_gws)
            gws_source = self.gws.source[:1].expand(B, S_gws)
            gws_beta = torch.full_like(gws_age, float(self.gws.recency_temp))
            gws_beta = torch.where(gws_source == MemoryType.SRC_MIXED, gws_beta + float(self.gws.recency_temp) * 0.5, gws_beta)
            gws_beta = torch.where(gws_source == MemoryType.SRC_IMAGINE, gws_beta + float(self.gws.recency_temp), gws_beta)
            gws_scores = self.gws.priority[:1].expand(B, S_gws) * torch.exp(-gws_beta * gws_age)
            gws_scores = gws_scores * SourceConfidence(gws_source, dtype=gws_scores.dtype)
            gws_scores = gws_scores.masked_fill(~gws_valid, -1e9)
            _, gws_idx = StableTopk(gws_scores, K_gws)  # [B, K_gws]
            out["gws"] = GatherTopkLatestFirst(self.gws.vals[:1], gws_idx, self.gws.last_step[:1])
            out["gws_valid"] = GatherMeta(
                gws_valid,
                gws_idx,
                self.gws.last_step[:1])
            meta["gws"] = {
                "score": GatherMeta(gws_scores, gws_idx, self.gws.last_step[:1]),
                "source": GatherMeta(self.gws.source[:1], gws_idx, self.gws.last_step[:1]).float(),
                "confidence": GatherMeta(SourceConfidence(self.gws.source[:1], dtype=self.dtype), gws_idx, self.gws.last_step[:1]),
                "age": GatherMeta(gws_age, gws_idx, self.gws.last_step[:1]),
                "touch": torch.zeros(B, K_gws, device=device, dtype=self.dtype),
                "reward_abs": torch.zeros(B, K_gws, device=device, dtype=self.dtype),
                "step": GatherMeta(self.gws.last_step[:1].float(), gws_idx, self.gws.last_step[:1]),}

        M_kv = int(self.memory_size)
        filled_kv = self.memory_filled  # [B]
        K_kv = min(perTypeBudget["kv"], M_kv, int(filled_kv.max().item()))
        if K_kv > 0:
            ar = torch.arange(M_kv, device=device).view(1, M_kv) # [1, M_kv]
            valid = ar < filled_kv.view(B, 1) # [B, M_kv]
            age = (self.time_step.view(B, 1) - self.memory_steps).clamp(min=0).float()
            scores = (self.memory_importance * (1.0 + 0.5 * torch.tanh(self.memory_reward_abs))).masked_fill(~valid, -1e9) # [B, M_kv]
            _, idx = StableTopk(scores, K_kv) # [B, K_kv]
            out["kv"] = GatherTopkLatestFirst(self.memory_values, idx, self.memory_steps) # [B, K_kv, D]
            out["kv_valid"] = GatherMeta(valid, idx, self.memory_steps)
            meta["kv"] = {
                "score": GatherMeta(scores, idx, self.memory_steps),
                "source": GatherMeta(self.memory_source.float(), idx, self.memory_steps),
                "confidence": GatherMeta(SourceConfidence(self.memory_source, dtype=self.dtype), idx, self.memory_steps),
                "age": GatherMeta(age, idx, self.memory_steps),
                "touch": torch.zeros(B, K_kv, device=device, dtype=self.dtype),
                "reward_abs": GatherMeta(self.memory_reward_abs, idx, self.memory_steps),
                "step": GatherMeta(self.memory_steps.float(), idx, self.memory_steps),}


        sem = self.ltm.semantic
        M_sem = int(sem.capacity)
        filled_sem = sem.filled # [B]
        K_sem = min(perTypeBudget["ltm_sem"], M_sem, int(filled_sem.max().item()))
        if K_sem > 0:
            ar = torch.arange(M_sem, device=sem.prio.device).view(1, M_sem) # [1, M_sem]
            valid = ar < filled_sem.view(B, 1) # [B, M_sem]
            age = (sem.global_step.view(B, 1) - sem.step).clamp(min=0).float()
            scores = (sem.prio * SourceConfidence(sem.source, dtype=sem.prio.dtype)).masked_fill(~valid, -1e9) # [B, M_sem]
            _, idx = StableTopk(scores, K_sem) # [B, K_sem]
            out["ltm_sem"] = GatherTopkLatestFirst(sem.vals, idx, sem.step) # [B, K_sem, D]
            out["ltm_sem_valid"] = GatherMeta(valid, idx, sem.step)
            meta["ltm_sem"] = {
                "score": GatherMeta(scores, idx, sem.step),
                "source": GatherMeta(sem.source.float(), idx, sem.step),
                "confidence": GatherMeta(SourceConfidence(sem.source, dtype=sem.prio.dtype), idx, sem.step),
                "age": GatherMeta(age, idx, sem.step),
                "touch": GatherMeta(sem.touch.float(), idx, sem.step),
                "reward_abs": torch.zeros(B, K_sem, device=device, dtype=self.dtype),
                "step": GatherMeta(sem.step.float(), idx, sem.step),}


        epi = self.ltm.episodic
        M_epi = int(epi.capacity)
        filled_epi = epi.filled  # [B]
        K_epi = min(perTypeBudget["ltm_epi"], M_epi, int(filled_epi.max().item()))
        if K_epi > 0:
            ar = torch.arange(M_epi, device=epi.prio.device).view(1, M_epi) # [1, M_epi]
            valid = ar < filled_epi.view(B, 1) # [B, M_epi]
            age = (epi.global_step.view(B, 1) - epi.step).clamp(min=0).float()
            scores = (epi.prio * (1.0 + 0.5 * torch.tanh(epi.rew_abs)) * SourceConfidence(epi.source, dtype=epi.prio.dtype)).masked_fill(~valid, -1e9) # [B, M_epi]
            _, idx = StableTopk(scores, K_epi) # [B, K_epi]
            out["ltm_epi"] = GatherTopkLatestFirst(epi.vals, idx, epi.step) # [B, K_epi, D]
            out["ltm_epi_valid"] = GatherMeta(valid, idx, epi.step)
            meta["ltm_epi"] = {
                "score": GatherMeta(scores, idx, epi.step),
                "source": GatherMeta(epi.source.float(), idx, epi.step),
                "confidence": GatherMeta(SourceConfidence(epi.source, dtype=epi.prio.dtype), idx, epi.step),
                "age": GatherMeta(age, idx, epi.step),
                "touch": GatherMeta(epi.touch.float(), idx, epi.step),
                "reward_abs": GatherMeta(epi.rew_abs, idx, epi.step),
                "step": GatherMeta(epi.step.float(), idx, epi.step),}


        sym = self.sym_mem
        n_sym = int(sym.filled.item())
        nsK = int(sym.K)

        K_sym = min(perTypeBudget["sym"], n_sym)
        if K_sym > 0:
            sym_scores = sym.prio[:n_sym] # [n_sym]
            _, sym_idx = StableTopk(sym_scores, K_sym) # [K_sym]
            sel_steps = sym.step[:n_sym].index_select(0, sym_idx)
            time_order = torch.argsort(sel_steps, dim=0, descending=True)
            sym_idx = sym_idx.index_select(0, time_order)
            top_vals = sym.P_vals[:n_sym].index_select(0, sym_idx) # [K_sym, nsK]
            out["sym"] = top_vals.unsqueeze(0).expand(B, K_sym, nsK).contiguous() # [B, K_sym, nsK]
            out["sym_valid"] = torch.ones(B, K_sym, device=device, dtype=torch.bool)
            if includeMeta:
                age = (sym.global_step - sym.step[:n_sym]).clamp(min=0).float().index_select(0, sym_idx)
                src = sym.source[:n_sym].index_select(0, sym_idx)
                meta["sym"] = {
                    "score": sym.prio[:n_sym].index_select(0, sym_idx).view(1, K_sym).expand(B, K_sym).contiguous(),
                    "source": src.float().view(1, K_sym).expand(B, K_sym).contiguous(),
                    "confidence": SourceConfidence(src, dtype=self.dtype).view(1, K_sym).expand(B, K_sym).contiguous(),
                    "age": age.view(1, K_sym).expand(B, K_sym).contiguous(),
                    "touch": sym.touch[:n_sym].index_select(0, sym_idx).float().view(1, K_sym).expand(B, K_sym).contiguous(),
                    "reward_abs": torch.zeros(B, K_sym, device=device, dtype=self.dtype),
                    "step": sym.step[:n_sym].index_select(0, sym_idx).float().view(1, K_sym).expand(B, K_sym).contiguous(),}

        if includeMeta and meta:
            out["meta"] = meta
        return out
    

    @torch.no_grad()
    def ExportTransientState(self) -> Dict[str, torch.Tensor]:
        gws = self.gws.Inspect()
        state = {
            "last_compress_step": self.last_compress_step.detach().clone(),
            "h_state": self.h_state.detach().clone(),
            "fast_weights": self.fast_weights.detach().clone(),
            "ns_prev_P_post": self.ns_prev_P_post.detach().clone(),
            "ns_penalty_vec": self.ns_penalty_vec.detach().clone(),
            "gws_global_step": self.gws.global_step.detach().clone(),
            "gws_keys": gws["keys"].detach().clone(),
            "gws_vals": gws["vals"].detach().clone(),
            "gws_priority": gws["priority"].detach().clone(),
            "gws_ttl": gws["ttl"].detach().clone(),
            "gws_last_step": gws["last_step"].detach().clone(),
            "gws_source": gws["source"].detach().clone(),}
        if tuple(state) != self.TRANSIENT_MEMORY_STATE_FIELDS:
            raise RuntimeError("transient-memory field declaration and export order disagree")
        return state

    @torch.no_grad()
    def ImportTransientState(self, state: Dict[str, torch.Tensor]) -> None:
        self._ImportCurrentState(state, includeDurable=False)

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
            "sym_mem_global_step": sym.global_step.clone(),
            "memory_version": self.memory_version.clone(),}

        state.update({
            "memory_keys": self.memory_keys.clone(), # [B, M, D]
            "memory_values": self.memory_values.clone(), # [B, M, D]
            "memory_importance": self.memory_importance.clone(), # [B, M]
            "memory_steps": self.memory_steps.clone(), # [B, M]
            "memory_emotion": self.memory_emotion.clone(), # [B, M, E]
            "memory_source": self.memory_source.clone(),
            "memory_reward_abs": self.memory_reward_abs.clone(),}) # [B, M]
        
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
            "ltm_epi_state_keys": epi.state_keys.clone(), # [B, Cap, D]
            "ltm_epi_vals": epi.vals.clone(), # [B, Cap, D]
            "ltm_epi_prio": epi.prio.clone(), # [B, Cap]
            "ltm_epi_rew": epi.rew.clone(), # [B, Cap]
            "ltm_epi_rew_abs": epi.rew_abs.clone(), # [B, Cap]
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
            if tuple(state) != self.FULL_MEMORY_STATE_FIELDS:
                raise RuntimeError("memory-state export does not match its schema")
            return state

        s = int(step)
        device = self.device

        B = int(state["memory_filled"].size(0))
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
        state["memory_reward_abs"] = torch.gather(state["memory_reward_abs"], 1, idx) * new_valid.float()

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
        state["ltm_epi_state_keys"] = torch.gather(state["ltm_epi_state_keys"], 1, idxE3) * new_validE.unsqueeze(-1).float()
        state["ltm_epi_vals"] = torch.gather(state["ltm_epi_vals"], 1, idxE3) * new_validE.unsqueeze(-1).float()
        state["ltm_epi_prio"] = torch.gather(state["ltm_epi_prio"], 1, idxEpi) * new_validE.float()
        state["ltm_epi_rew"] = torch.gather(state["ltm_epi_rew"], 1, idxEpi) * new_validE.float()
        state["ltm_epi_rew_abs"] = torch.gather(state["ltm_epi_rew_abs"], 1, idxEpi) * new_validE.float()
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

        if tuple(state) != self.FULL_MEMORY_STATE_FIELDS:
            raise RuntimeError("memory-state export does not match its schema")
        return state

    @torch.no_grad()
    def MergeMemoryState(
        self,
        state: Dict[str, torch.Tensor],
        mergeGws: bool = False,) -> None:

        if type(state) is not dict or tuple(state) != self.FULL_MEMORY_STATE_FIELDS:
            raise TypeError("merged memory-state fields do not match the current schema")
        if not all(torch.is_tensor(value) for value in state.values()):
            raise TypeError("every merged memory-state field must be a tensor")

        def require_shape(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"invalid merged memory-state shape: {message}")

        memory_keys = state["memory_keys"]
        memory_values = state["memory_values"]
        require_shape(memory_keys.dim() == 3, "memory_keys must be [B,M,D]")
        require_shape(memory_values.shape == memory_keys.shape, "memory_values must match memory_keys")
        B_src, M_src, D_src = memory_keys.shape
        require_shape(D_src == self.memory_dim, "memory feature dimension differs from the destination")
        require_shape(state["memory_filled"].shape == (B_src,), "memory_filled must be [B]")
        for field in ("memory_importance", "memory_steps", "memory_source", "memory_reward_abs"):
            require_shape(state[field].shape == (B_src, M_src), f"{field} must be [B,M]")
        require_shape(
            state["memory_emotion"].shape == (B_src, M_src, self.emotion_dim),
            "memory_emotion must be [B,M,E]")

        sem_keys = state["ltm_sem_keys"]
        sem_vals = state["ltm_sem_vals"]
        require_shape(sem_keys.dim() == 3, "ltm_sem_keys must be [B,C,D]")
        require_shape(sem_vals.shape == sem_keys.shape, "ltm_sem_vals must match ltm_sem_keys")
        B_sem, C_sem, D_sem = sem_keys.shape
        require_shape(B_sem == B_src and D_sem == self.memory_dim, "semantic-memory batch/feature dimensions differ")
        require_shape(state["ltm_sem_filled"].shape == (B_src,), "ltm_sem_filled must be [B]")
        for field in ("ltm_sem_prio", "ltm_sem_touch", "ltm_sem_step", "ltm_sem_source"):
            require_shape(state[field].shape == (B_src, C_sem), f"{field} must be [B,C]")

        epi_keys = state["ltm_epi_keys"]
        epi_state_keys = state["ltm_epi_state_keys"]
        epi_vals = state["ltm_epi_vals"]
        require_shape(epi_keys.dim() == 3, "ltm_epi_keys must be [B,C,D]")
        require_shape(epi_state_keys.shape == epi_keys.shape, "ltm_epi_state_keys must match ltm_epi_keys")
        require_shape(epi_vals.shape == epi_keys.shape, "ltm_epi_vals must match ltm_epi_keys")
        B_epi, C_epi, D_epi = epi_keys.shape
        require_shape(B_epi == B_src and D_epi == self.memory_dim, "episodic-memory batch/feature dimensions differ")
        require_shape(state["ltm_epi_filled"].shape == (B_src,), "ltm_epi_filled must be [B]")
        for field in (
            "ltm_epi_prio", "ltm_epi_rew", "ltm_epi_rew_abs",
            "ltm_epi_step", "ltm_epi_touch", "ltm_epi_source",):
            require_shape(state[field].shape == (B_src, C_epi), f"{field} must be [B,C]")

        sym_keys = state["sym_mem_P_keys"]
        sym_vals = state["sym_mem_P_vals"]
        require_shape(sym_keys.dim() == 2, "sym_mem_P_keys must be [C,K]")
        require_shape(sym_vals.shape == sym_keys.shape, "sym_mem_P_vals must match sym_mem_P_keys")
        C_sym, K_sym = sym_keys.shape
        require_shape(K_sym == self.sym_mem.K, "symbolic-memory feature dimension differs")
        for field in ("sym_mem_prio", "sym_mem_step", "sym_mem_touch", "sym_mem_source"):
            require_shape(state[field].shape == (C_sym,), f"{field} must be [C]")
        require_shape(state["sym_mem_filled"].shape == (), "sym_mem_filled must be scalar")

        if mergeGws:
            gws_keys = state["gws_keys"]
            gws_vals = state["gws_vals"]
            require_shape(gws_keys.dim() == 3, "gws_keys must be [B,S,D]")
            require_shape(gws_vals.shape == gws_keys.shape, "gws_vals must match gws_keys")
            B_gws, S_gws, D_gws = gws_keys.shape
            require_shape(B_gws == B_src and D_gws == self.memory_dim, "workspace batch/feature dimensions differ")
            for field in ("gws_priority", "gws_ttl", "gws_last_step", "gws_source"):
                require_shape(state[field].shape == (B_src, S_gws), f"{field} must be [B,S]")
            require_shape(state["gws_global_step"].shape == (B_src,), "gws_global_step must be [B]")

        device = self.device
        dtype = self.dtype

        B_dst = int(self.memory_filled.size(0))

        if mergeGws:
            gk = state["gws_keys"].to(device=self.gws.keys.device, dtype=self.gws.keys.dtype) # [B, S, D]
            gv = state["gws_vals"].to(device=self.gws.vals.device, dtype=self.gws.vals.dtype) # [B, S, D]
            gpr = state["gws_priority"].to(device=self.gws.priority.device, dtype=self.gws.priority.dtype) # [B, S]
            gttl = state["gws_ttl"].to(device=self.gws.ttl.device, dtype=torch.long) # [B, S]
            gls = state["gws_last_step"].to(device=self.gws.last_step.device, dtype=torch.long) # [B, S]
            gsrc = state["gws_source"].to(device=self.gws.source.device, dtype=torch.int8) # [B, S]
            ggs = state["gws_global_step"].to(device=self.gws.global_step.device, dtype=torch.long) # [B]

            B_gws_src = int(gk.size(0))
            S_src = int(gk.size(1))
            S_dst = int(self.gws.slots)
            B = min(B_dst, B_gws_src)
            S = min(S_dst, S_src)

            self.gws.keys[:B, :S].copy_(gk[:B, :S])
            self.gws.vals[:B, :S].copy_(gv[:B, :S])
            self.gws.priority[:B, :S].copy_(gpr[:B, :S])
            self.gws.ttl[:B, :S].copy_(gttl[:B, :S])
            self.gws.last_step[:B, :S].copy_(gls[:B, :S])
            self.gws.source[:B, :S].copy_(gsrc[:B, :S])
            self.gws.global_step[:B].copy_(ggs[:B])

        k_src = state["memory_keys"].to(device=device, dtype=dtype) # [B, M, D]
        v_src = state["memory_values"].to(device=device, dtype=dtype) # [B, M, D]
        B = min(B_dst, B_src)
        filled_src = state["memory_filled"].to(device=device, dtype=torch.long)
        if bool(((filled_src < 0) | (filled_src > M_src)).any().item()):
            raise ValueError("memory_filled is outside the source memory capacity")
        imp_src = state["memory_importance"].to(device=device, dtype=dtype)
        emo_src = state["memory_emotion"].to(device=device, dtype=dtype)
        src_src = state["memory_source"].to(device=device, dtype=torch.int8)
        rew_abs_src = state["memory_reward_abs"].to(device=device, dtype=dtype)
        step_src = state["memory_steps"].to(device=device, dtype=torch.long)  # [B, M]

        max_n = int(filled_src[:B].max().item()) if B > 0 else 0
        M_dst = int(self.memory_size)
        for t in range(max_n):
            mask = (t < filled_src[:B])  # [B]
            filled_dst = self.memory_filled[:B] # [B]
            is_full = filled_dst >= M_dst # [B]
            idx_append = filled_dst.clamp(max=M_dst - 1) # [B]
            idx_evict = torch.argmin(self.memory_importance[:B], dim=1) # [B]
            tgt = torch.where(is_full, idx_evict, idx_append) # [B]

            b_idx = torch.arange(B, device=device, dtype=torch.long)
            b_sel = b_idx[mask]
            tgt_sel = tgt[mask]

            self.memory_keys[b_sel, tgt_sel] = k_src[:B, t][mask]
            self.memory_values[b_sel, tgt_sel] = v_src[:B, t][mask]
            self.memory_importance[b_sel, tgt_sel] = imp_src[:B, t][mask]
            self.memory_emotion[b_sel, tgt_sel] = emo_src[:B, t][mask]
            self.memory_source[b_sel, tgt_sel] = src_src[:B, t][mask]
            self.memory_reward_abs[b_sel, tgt_sel] = rew_abs_src[:B, t][mask]
            self.memory_steps[b_sel, tgt_sel] = step_src[:B, t][mask]

            filled_new = filled_dst + (~is_full).long()
            self.memory_filled[:B].copy_(torch.where(mask, filled_new, filled_dst))

        sem = self.ltm.semantic
        k_src = state["ltm_sem_keys"].to(device=sem.keys.device, dtype=sem.keys.dtype) # [B, C, D]
        v_src = state["ltm_sem_vals"].to(device=sem.vals.device, dtype=sem.vals.dtype) # [B, C, D]
        B = min(B_dst, B_sem, int(sem.filled.size(0)))
        C_dst = int(sem.capacity)
        filled_src = state["ltm_sem_filled"].to(device=sem.keys.device, dtype=torch.long)
        if bool(((filled_src < 0) | (filled_src > C_sem)).any().item()):
            raise ValueError("ltm_sem_filled is outside the source semantic-memory capacity")
        pr_src = state["ltm_sem_prio"].to(device=sem.prio.device, dtype=sem.prio.dtype)
        src_src = state["ltm_sem_source"].to(device=sem.source.device, dtype=torch.int8)

        max_n = int(filled_src[:B].max().item()) if B > 0 else 0
        for t in range(max_n):
            mask = (t < filled_src[:B])
            filled_dst = sem.filled[:B]
            is_full = filled_dst >= C_dst
            idx_append = filled_dst.clamp(max=C_dst - 1)
            idx_evict = torch.argmin(sem.prio[:B], dim=1)
            tgt = torch.where(is_full, idx_evict, idx_append)

            b_idx = torch.arange(B, device=sem.keys.device, dtype=torch.long)
            b_sel = b_idx[mask]
            tgt_sel = tgt[mask]
            sem.keys[b_sel, tgt_sel] = k_src[:B, t][mask]
            sem.vals[b_sel, tgt_sel] = v_src[:B, t][mask]
            sem.prio[b_sel, tgt_sel] = pr_src[:B, t][mask]
            sem.source[b_sel, tgt_sel] = src_src[:B, t][mask]
            sem.touch[b_sel, tgt_sel] = 1
            sem.step[b_sel, tgt_sel] = sem.global_step[:B][mask]

            filled_new = filled_dst + (~is_full).long()
            sem.filled[:B].copy_(torch.where(mask, filled_new, filled_dst))

        epi = self.ltm.episodic
        k_src = state["ltm_epi_keys"].to(device=epi.keys.device, dtype=epi.keys.dtype) # [B, C, D]
        ks_src = state["ltm_epi_state_keys"].to(device=epi.state_keys.device, dtype=epi.state_keys.dtype) # [B, C, D]
        v_src = state["ltm_epi_vals"].to(device=epi.vals.device, dtype=epi.vals.dtype) # [B, C, D]
        B = min(B_dst, B_epi, int(epi.filled.size(0)))
        C_dst = int(epi.capacity)
        filled_src = state["ltm_epi_filled"].to(device=epi.keys.device, dtype=torch.long)
        if bool(((filled_src < 0) | (filled_src > C_epi)).any().item()):
            raise ValueError("ltm_epi_filled is outside the source episodic-memory capacity")
        pr_src = state["ltm_epi_prio"].to(device=epi.prio.device, dtype=epi.prio.dtype)
        rw_src = state["ltm_epi_rew"].to(device=epi.rew.device, dtype=epi.rew.dtype)
        rw_abs_src = state["ltm_epi_rew_abs"].to(device=epi.rew_abs.device, dtype=epi.rew_abs.dtype)
        src_src = state["ltm_epi_source"].to(device=epi.source.device, dtype=torch.int8)

        max_n = int(filled_src[:B].max().item()) if B > 0 else 0
        for t in range(max_n):
            mask = (t < filled_src[:B])
            filled_dst = epi.filled[:B]
            is_full = filled_dst >= C_dst
            idx_append = filled_dst.clamp(max=C_dst - 1)
            idx_evict = torch.argmin(epi.prio[:B], dim=1)
            tgt = torch.where(is_full, idx_evict, idx_append)

            b_idx = torch.arange(B, device=epi.keys.device, dtype=torch.long)
            b_sel = b_idx[mask]
            tgt_sel = tgt[mask]
            epi.keys[b_sel, tgt_sel] = k_src[:B, t][mask]
            epi.state_keys[b_sel, tgt_sel] = ks_src[:B, t][mask]
            epi.vals[b_sel, tgt_sel] = v_src[:B, t][mask]
            epi.prio[b_sel, tgt_sel] = pr_src[:B, t][mask]
            epi.rew[b_sel, tgt_sel] = rw_src[:B, t][mask]
            epi.rew_abs[b_sel, tgt_sel] = rw_abs_src[:B, t][mask]
            epi.source[b_sel, tgt_sel] = src_src[:B, t][mask]
            epi.touch[b_sel, tgt_sel] = 1
            epi.step[b_sel, tgt_sel] = epi.global_step[:B][mask]

            filled_new = filled_dst + (~is_full).long()
            epi.filled[:B].copy_(torch.where(mask, filled_new, filled_dst))


        sym = self.sym_mem
        Pk = state["sym_mem_P_keys"].to(device=sym.P_keys.device, dtype=sym.P_keys.dtype) # [C, K]
        Pv = state["sym_mem_P_vals"].to(device=sym.P_vals.device, dtype=sym.P_vals.dtype) # [C, K]
        pr = state["sym_mem_prio"].to(device=sym.prio.device, dtype=sym.prio.dtype)
        src = state["sym_mem_source"].to(device=sym.source.device, dtype=torch.int8)
        n = int(state["sym_mem_filled"].item())
        if n < 0 or n > C_sym:
            raise ValueError("sym_mem_filled is outside the source symbolic-memory capacity")

        for i in range(n):
            sym.Store(
                key=Pk[i],
                value=Pv[i],
                score=float(pr[i].item()),
                source=int(src[i].item()),)

        self.pending.clear()
        self.memory_version.add_(1)

    @torch.no_grad()
    def _ImportCurrentState(
        self,
        state: Dict[str, torch.Tensor],
        *,
        includeDurable: bool,) -> None:
        expected_fields = (
            self.FULL_MEMORY_STATE_FIELDS
            if includeDurable
            else self.TRANSIENT_MEMORY_STATE_FIELDS)
        if type(state) is not dict or tuple(state) != expected_fields:
            raise TypeError("memory-state fields do not match the current schema")
        if not all(torch.is_tensor(value) for value in state.values()):
            raise TypeError("every memory-state field must be a tensor")

        if state["h_state"].dim() == 0:
            raise ValueError("memory-state h_state must have a batch dimension")
        B = int(state["h_state"].size(0))
        self.EnsureB(B)

        self.gws.EnsureB(B)
        if includeDurable:
            self.ltm.semantic.EnsureB(B)
            self.ltm.episodic.EnsureB(B)

        targets = [
            ("last_compress_step", self.last_compress_step),
            ("h_state", self.h_state),
            ("fast_weights", self.fast_weights),
            ("ns_prev_P_post", self.ns_prev_P_post),
            ("ns_penalty_vec", self.ns_penalty_vec),
            ("gws_global_step", self.gws.global_step),
            ("gws_keys", self.gws.keys),
            ("gws_vals", self.gws.vals),
            ("gws_priority", self.gws.priority),
            ("gws_ttl", self.gws.ttl),
            ("gws_last_step", self.gws.last_step),
            ("gws_source", self.gws.source),]

        if includeDurable:
            sem = self.ltm.semantic
            epi = self.ltm.episodic

            targets.extend([
                ("time_step", self.time_step),
                ("memory_filled", self.memory_filled),
                ("memory_version", self.memory_version),
                ("memory_keys", self.memory_keys),
                ("memory_values", self.memory_values),
                ("memory_importance", self.memory_importance),
                ("memory_steps", self.memory_steps),
                ("memory_emotion", self.memory_emotion),
                ("memory_source", self.memory_source),
                ("memory_reward_abs", self.memory_reward_abs),
                ("ltm_sem_global_step", sem.global_step),
                ("ltm_sem_keys", sem.keys),
                ("ltm_sem_vals", sem.vals),
                ("ltm_sem_prio", sem.prio),
                ("ltm_sem_touch", sem.touch),
                ("ltm_sem_step", sem.step),
                ("ltm_sem_filled", sem.filled),
                ("ltm_sem_source", sem.source),
                ("ltm_epi_global_step", epi.global_step),
                ("ltm_epi_keys", epi.keys),
                ("ltm_epi_state_keys", epi.state_keys),
                ("ltm_epi_vals", epi.vals),
                ("ltm_epi_prio", epi.prio),
                ("ltm_epi_rew", epi.rew),
                ("ltm_epi_rew_abs", epi.rew_abs),
                ("ltm_epi_step", epi.step),
                ("ltm_epi_touch", epi.touch),
                ("ltm_epi_filled", epi.filled),
                ("ltm_epi_source", epi.source),])

        if includeDurable:
            targets.extend([
                ("sym_mem_global_step", self.sym_mem.global_step),
                ("sym_mem_P_keys", self.sym_mem.P_keys),
                ("sym_mem_P_vals", self.sym_mem.P_vals),
                ("sym_mem_prio", self.sym_mem.prio),
                ("sym_mem_step", self.sym_mem.step),
                ("sym_mem_touch", self.sym_mem.touch),
                ("sym_mem_filled", self.sym_mem.filled),
                ("sym_mem_source", self.sym_mem.source),])

        for field, dst in targets:
            src = state[field]
            if src.shape != dst.shape:
                raise ValueError(
                    f"memory-state field {field!r} has shape {tuple(src.shape)}, "
                    f"expected {tuple(dst.shape)}")
        for field, dst in targets:
            dst.copy_(state[field].to(device=dst.device, dtype=dst.dtype))

        self.ResetInternalLoss()
        self.pending.clear()

    @torch.no_grad()
    def ImportState(self, state: Dict[str, torch.Tensor]) -> None:
        self._ImportCurrentState(state, includeDurable=True)


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

    def MakeVisualState(self, B: int, device: torch.device, dtype: torch.dtype = torch.float32):
        return SimpleNamespace(
            IntegratedFeat=torch.randn(B, 1024, device=device, dtype=dtype),
            MotionToken=torch.randn(B, 512, device=device, dtype=dtype),
            PredErrorToken=torch.randn(B, 512, device=device, dtype=dtype),)

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
        sourceLabel: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        risk: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,) -> torch.Tensor:
        B = int(x.size(0))
        device = x.device
        tdError = torch.zeros(B, device=device) if tdError is None else tdError.view(B).to(device=device)
        reward = torch.zeros(B, device=device) if reward is None else reward.view(B).to(device=device)
        if emotion is None:
            emotion = self.MakeEmotion(B, mem)
        else:
            emotion = emotion.to(device=device)
        if uncertainty is None:
            uncertainty = torch.zeros(B, device=device, dtype=x.dtype)
        if risk is None:
            risk = torch.zeros(B, device=device, dtype=x.dtype)
        if confidence is None:
            confidence = torch.ones(B, device=device, dtype=x.dtype)

        visualState = self.MakeVisualState(B, device, x.dtype)
        ocrSemantic = torch.randn(B, 512, device=device, dtype=x.dtype)
        intentHint = torch.randn(B, 512, device=device, dtype=x.dtype)
        return mem(
            x,
            tdError=tdError,
            emotion=emotion,
            reward=reward,
            visualState=visualState,
            ocrSemantic=ocrSemantic,
            intentHint=intentHint,
            reset=reset,
            softReset=softReset,
            sourceLabel=sourceLabel,
            uncertainty=uncertainty,
            risk=risk,
            confidence=confidence,)

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
            key = torch.zeros(B, dim, device=self.device)
            key[0, 0] = 1.0
            state_key = torch.zeros(B, dim, device=self.device)
            state_key[0, 1] = 1.0
            val = torch.randn(B, dim, device=self.device)
            key2 = torch.zeros(B, dim, device=self.device)
            key2[0, 2] = 1.0
            state_key2 = torch.zeros(B, dim, device=self.device)
            state_key2[0, 3] = 1.0
            val2 = torch.zeros(B, dim, device=self.device)
            val2[0, 4] = 3.0

            ltm.semantic.Store(key=key, value=val, score=torch.tensor([0.9], device=self.device),
                              source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8))
            ltm.episodic.Store(key=key, value=val, reward=torch.tensor([-1.0], device=self.device),
                               score=torch.tensor([0.8], device=self.device),
                               source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),
                               stateKey=state_key)
            ltm.episodic.Store(key=key2, value=val2, reward=torch.tensor([2.0], device=self.device),
                               score=torch.tensor([1.0], device=self.device),
                               source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),
                               stateKey=state_key2)

            ltm.StepTick()
            sem_out, epi_out = ltm.Retrieve(key, topkSem=4, topkEpi=2)
            epi_state_out = ltm.episodic.Retrieve(state_key, topk=2, useStateKey=True)
            epi_event2 = ltm.episodic.Retrieve(key2, topk=1)
            epi_state2 = ltm.episodic.Retrieve(state_key2, topk=1, useStateKey=True)
            assert sem_out.shape == (B, dim) and epi_out.shape == (B, dim)
            assert epi_state_out.shape == (B, dim)
            assert torch.linalg.norm(sem_out).item() > 0 and torch.linalg.norm(epi_out).item() > 0
            assert torch.linalg.norm(epi_state_out).item() > 0
            assert torch.allclose(epi_event2, val2, atol=1e-6, rtol=1e-6)
            assert torch.allclose(epi_state2, val2, atol=1e-6, rtol=1e-6)

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
            epi.EnsureB(1)
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
            epi.EnsureB(1)
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
            assert int(changed) == 1, f"expected keep slot0(rew=-10) and evict slot1, but evicted slot{changed}"
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
                valid = bank[f"{k}_valid"]
                assert valid.dtype == torch.bool
                assert valid.shape == bank[k].shape[:2]
                assert bool(valid.all().item())

            print("ExportMemoryBank latest-first-order test passed.")
            return True
        except AssertionError as e:
            print(f"ExportMemoryBank latest-first-order test failed: {e}")
            return False
        except Exception as e:
            print(f"ExportMemoryBank latest-first-order test error: {e}")
            return False

    def TestExportMemoryBankMixedRowValidity(self):
        try:
            cfg = self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16,
                ssmStateDim=16,
                memoryDim=8,
                memorySize=4,
                symSize=8,
                ltmSize=4,
                nsK=4,
                outputDim=8,
                gwsSlots=2,
                gwsTtl=4,
                compressEvery=100,
                emotionDim=4,))
            mem = MemoryExtractor(**cfg).to(self.device).eval()
            mem.EnsureB(2)
            mem.ResetAll()

            with torch.no_grad():
                mem.memory_filled.copy_(torch.tensor([2, 1], device=self.device))
                mem.memory_importance[:, :2] = torch.tensor(
                    [[2.0, 1.0], [2.0, 0.0]],
                    device=self.device,
                    dtype=mem.dtype)
                mem.memory_steps[:, :2] = torch.tensor(
                    [[2, 1], [2, 0]],
                    device=self.device)

            bank = mem.ExportMemoryBank(topk=2, includeMeta=False)
            assert bank is not None
            assert bank["kv_valid"].dtype == torch.bool
            assert bank["kv_valid"].tolist() == [[True, True], [True, False]]

            print("ExportMemoryBank mixed-row validity test passed.")
            return True
        except AssertionError as e:
            print(f"ExportMemoryBank mixed-row validity test failed: {e}")
            return False
        except Exception as e:
            print(f"ExportMemoryBank mixed-row validity test error: {e}")
            return False

    def TestDurableStatePreservesBatchLanes(self):
        path = self.StatePath("memory_durable_batch_test.pth")
        try:
            cfg = self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=6,
                symSize=8, ltmSize=6, nsK=4, outputDim=8,
                gwsSlots=3, gwsTtl=4, compressEvery=100, emotionDim=4))

            mem = MemoryExtractor(**cfg).to(self.device).eval()
            mem.EnsureB(3)
            mem.ltm.semantic.EnsureB(3)
            mem.ltm.episodic.EnsureB(3)
            mem.gws.EnsureB(3)

            with torch.no_grad():
                mem.time_step[:] = torch.tensor([101, 202, 303], device=self.device)
                mem.memory_filled[:] = torch.tensor([2, 1, 1], device=self.device)
                mem.memory_keys[:, 0, 0] = torch.tensor([11.0, 22.0, 33.0], device=self.device)
                mem.memory_values[:, 0, 1] = torch.tensor([12.0, 23.0, 34.0], device=self.device)
                mem.memory_importance[:, 0] = torch.tensor([0.9, 0.2, 0.1], device=self.device)
                mem.memory_reward_abs[:, 0] = torch.tensor([3.0, 0.0, 4.0], device=self.device)

                sem = mem.ltm.semantic
                sem.filled.fill_(1)
                sem.keys[:, 0, 0] = torch.tensor([41.0, 42.0, 43.0], device=self.device)
                epi = mem.ltm.episodic
                epi.filled.fill_(1)
                epi.rew_abs[:, 0] = torch.tensor([9.0, 1.0, 5.0], device=self.device)

                mem.sym_mem.filled.fill_(1)
                mem.sym_mem.P_vals[0, 0] = 77.0
                mem.usage_bank.success_alpha[0, 0] = 19.0
                mem.usage_bank.attribute_centroid[0, 0, 0] = 23.0

                mem.h_state.fill_(31.0)
                mem.fast_weights.fill_(32.0)
                mem.gws.keys.fill_(33.0)
                mem.ns_prev_P_post.fill_(34.0)
                mem.ns_penalty_vec.fill_(35.0)
                mem.last_compress_step.fill_(36)

            expected_state = mem.ExportDurableState()
            transient_state = mem.ExportTransientState()
            assert set(transient_state).isdisjoint(expected_state)
            mem.SaveState(str(path))
            payload = torch.load(path, map_location=self.device, weights_only=True)
            assert set(payload) == {"artifact_type", "schema_version", "batch_size", "state"}
            assert payload["batch_size"] == 3
            assert set(payload["state"]) == set(MemoryExtractor.DURABLE_MEMORY_STATE_FIELDS)
            assert "state_dict" not in payload
            assert "h_state" not in payload["state"]
            assert "gws_keys" not in payload["state"]

            mem2 = MemoryExtractor(**cfg).to(self.device).eval()
            with torch.no_grad():
                mem2.A_full.fill_(0.314159)
            mem2.LoadState(str(path))

            restored_state = mem2.ExportDurableState()
            for name in MemoryExtractor.DURABLE_MEMORY_STATE_FIELDS:
                assert torch.equal(restored_state[name], expected_state[name]), name
            assert mem2.memory_keys[:, 0, 0].tolist() == [11.0, 22.0, 33.0]
            assert mem2.ltm.semantic.keys[:, 0, 0].tolist() == [41.0, 42.0, 43.0]
            assert mem2.ltm.episodic.rew_abs[:, 0].tolist() == [9.0, 1.0, 5.0]
            assert torch.count_nonzero(mem2.h_state).item() == 0
            assert torch.count_nonzero(mem2.fast_weights).item() == 0
            assert torch.count_nonzero(mem2.gws.keys).item() == 0
            assert torch.count_nonzero(mem2.ns_prev_P_post).item() == 0
            assert torch.count_nonzero(mem2.ns_penalty_vec).item() == 0
            assert torch.equal(mem2.last_compress_step, mem2.time_step)
            assert torch.all(mem2.A_full == mem2.A_full.new_tensor(0.314159))

            mem2.ImportTransientState(transient_state)
            restored_transient = mem2.ExportTransientState()
            for name in MemoryExtractor.TRANSIENT_MEMORY_STATE_FIELDS:
                assert torch.equal(restored_transient[name], transient_state[name]), name

            print("Durable-memory full-batch save/load test passed.")
            return True
        except AssertionError as e:
            print(f"Durable-memory full-batch test failed: {e}")
            return False
        except Exception as e:
            print(f"Durable-memory full-batch test error: {e}")
            return False
        finally:
            try:
                path.unlink()
            except OSError:
                pass

    def TestDurableStateRejectsNonCurrentSchema(self):
        legacy_path = self.StatePath("memory_legacy_schema_test.pth")
        incomplete_path = self.StatePath("memory_incomplete_schema_test.pth")
        try:
            cfg = self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=6,
                symSize=8, ltmSize=6, nsK=4, outputDim=8,
                gwsSlots=3, gwsTtl=4, compressEvery=100, emotionDim=4))
            mem = MemoryExtractor(**cfg).to(self.device).eval()

            torch.save({"state_dict": mem.state_dict()}, legacy_path)
            legacy_rejected = False
            try:
                mem.LoadState(str(legacy_path))
            except (TypeError, ValueError):
                legacy_rejected = True
            assert legacy_rejected, "legacy state_dict artifact was accepted"

            state = mem.ExportDurableState()
            del state["usage_attribute_centroid"]
            torch.save({
                "artifact_type": MemoryExtractor.DURABLE_MEMORY_ARTIFACT_TYPE,
                "schema_version": MemoryExtractor.DURABLE_MEMORY_SCHEMA_VERSION,
                "batch_size": 1,
                "state": state,}, incomplete_path)
            incomplete_rejected = False
            try:
                mem.LoadState(str(incomplete_path))
            except (TypeError, ValueError):
                incomplete_rejected = True
            assert incomplete_rejected, "incomplete durable-memory state was accepted"

            print("Durable-memory strict-schema rejection test passed.")
            return True
        except AssertionError as e:
            print(f"Durable-memory strict-schema rejection test failed: {e}")
            return False
        except Exception as e:
            print(f"Durable-memory strict-schema rejection test error: {e}")
            return False
        finally:
            for path in (legacy_path, incomplete_path):
                try:
                    path.unlink()
                except OSError:
                    pass

    def TestGlobalWorkspaceSharedBatch(self):
        try:
            dim, slots = 8, 4
            gws = GlobalWorkspace(dim=dim, slots=slots, defaultTtl=5).to(self.device)
            gws.Reset()
            gws.StepTick()

            keys = torch.zeros(2, dim, device=self.device)
            vals = torch.zeros(2, dim, device=self.device)
            keys[0, 0] = 1.0
            keys[1, 1] = 1.0
            vals[0, 2] = 10.0
            vals[1, 3] = 20.0
            gws.Write(
                keys,
                vals,
                priority=torch.tensor([1.0, 1.0], device=self.device),
                ttl=torch.tensor([5, 5], device=self.device),
                tagId=torch.tensor([MemoryType.SRC_REAL, MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),)

            assert int(gws.keys.size(0)) == 2, f"GWS EnsureB should use caller B, got B={gws.keys.size(0)}"
            assert int((gws.priority[0] > 0).sum().item()) == 2

            out = gws.Attend(keys, topk=1)
            assert out.shape == (2, dim)
            assert float(out[0, 2].item()) > 9.0, f"row0 recall mismatch: {out[0].tolist()}"
            assert float(out[1, 3].item()) > 19.0, f"row1 recall mismatch: {out[1].tolist()}"

            print("GlobalWorkspace shared-batch test passed.")
            return True
        except AssertionError as e:
            print(f"GlobalWorkspace shared-batch test failed: {e}")
            return False
        except Exception as e:
            print(f"GlobalWorkspace shared-batch test error: {e}")
            return False

    def TestSourceConfidenceOrdering(self):
        try:
            dim = 8
            q = F.normalize(torch.ones(1, dim, device=self.device), dim=-1)
            keys = q.view(1, 1, dim).expand(1, 3, dim).contiguous()
            src = torch.tensor(
                [[MemoryType.SRC_REAL, MemoryType.SRC_MIXED, MemoryType.SRC_IMAGINE]],
                device=self.device,
                dtype=torch.int8,)
            scores = UnifiedMemoryScore(
                q,
                keys,
                age=torch.zeros(1, 3, device=self.device),
                priority=torch.ones(1, 3, device=self.device),
                source=src,)
            assert scores[0, 0] > scores[0, 1] > scores[0, 2], f"source discount order broken: {scores.tolist()}"
            conf = SourceConfidence(src)
            assert conf[0, 0] > conf[0, 1] > conf[0, 2], f"confidence order broken: {conf.tolist()}"

            print("Source confidence ordering test passed.")
            return True
        except AssertionError as e:
            print(f"Source confidence ordering test failed: {e}")
            return False
        except Exception as e:
            print(f"Source confidence ordering test error: {e}")
            return False

    def TestExportMemoryBankMetaBudget(self):
        try:
            cfg = dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=6,
                symSize=8, ltmSize=6, nsK=4, outputDim=8,
                gwsSlots=4, gwsTtl=6, compressEvery=100, emotionDim=4)
            cfg = self.FilterKwargs(MemoryExtractor, cfg)
            mem = MemoryExtractor(**cfg).to(self.device).eval()
            mem.ResetAll()

            with torch.no_grad():
                mem.time_step.fill_(3)
                mem.memory_filled.fill_(2)
                mem.memory_importance[0, :2] = torch.tensor([0.8, 0.6], device=self.device)
                mem.memory_steps[0, :2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                mem.memory_values[0, 0, 0] = 1.0
                mem.memory_values[0, 1, 0] = 2.0
                mem.memory_reward_abs[0, 1] = 5.0
                mem.memory_source[0, :2] = torch.tensor([MemoryType.SRC_REAL, MemoryType.SRC_MIXED], device=self.device, dtype=torch.int8)

                mem.gws.global_step.fill_(3)
                mem.gws.priority[0, :2] = torch.tensor([0.7, 0.9], device=self.device)
                mem.gws.ttl[0, :2] = 5
                mem.gws.last_step[0, :2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                mem.gws.vals[0, 0, 0] = 3.0
                mem.gws.vals[0, 1, 0] = 4.0

                sem = mem.ltm.semantic
                sem.filled.fill_(2)
                sem.global_step.fill_(3)
                sem.prio[0, :2] = torch.tensor([0.5, 0.9], device=self.device)
                sem.step[0, :2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                sem.touch[0, :2] = 1
                sem.vals[0, 0, 0] = 5.0
                sem.vals[0, 1, 0] = 6.0

                epi = mem.ltm.episodic
                epi.filled.fill_(2)
                epi.global_step.fill_(3)
                epi.prio[0, :2] = torch.tensor([0.5, 0.9], device=self.device)
                epi.rew_abs[0, 1] = 4.0
                epi.step[0, :2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                epi.touch[0, :2] = 1
                epi.vals[0, 0, 0] = 7.0
                epi.vals[0, 1, 0] = 8.0

                sym = mem.sym_mem
                sym.filled.fill_(2)
                sym.prio[:2] = torch.tensor([0.5, 0.9], device=self.device)
                sym.step[:2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                sym.touch[:2] = 1
                sym.P_vals[0, 0] = 9.0
                sym.P_vals[1, 0] = 10.0

            bank = mem.ExportMemoryBank(topk=3, totalBudget=5, includeMeta=True)
            assert bank is not None
            for key in ("gws", "kv", "ltm_sem", "ltm_epi", "sym"):
                assert key in bank, f"missing memory bank key: {key}"
                assert int(bank[key].size(1)) <= 1, f"{key} budget not applied: {bank[key].shape}"
            assert "meta" in bank
            for key in ("gws", "kv", "ltm_sem", "ltm_epi", "sym"):
                assert key in bank["meta"], f"missing meta for {key}"
                assert "score" in bank["meta"][key] and "confidence" in bank["meta"][key]
                assert int(bank["meta"][key]["score"].size(1)) <= 1

            print("ExportMemoryBank meta/budget test passed.")
            return True
        except AssertionError as e:
            print(f"ExportMemoryBank meta/budget test failed: {e}")
            return False
        except Exception as e:
            print(f"ExportMemoryBank meta/budget test error: {e}")
            return False

    def TestConsolidationWritesSemanticAndSymbolic(self):
        try:
            cfg = dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=6,
                symSize=8, ltmSize=6, nsK=4, outputDim=8,
                gwsSlots=4, gwsTtl=6, compressEvery=100, emotionDim=4)
            cfg = self.FilterKwargs(MemoryExtractor, cfg)
            mem = MemoryExtractor(**cfg).to(self.device).eval()
            mem.EnsureB(2)
            mem.ltm.semantic.EnsureB(2)
            mem.ltm.episodic.EnsureB(2)
            mem.ltm.semantic.filled.zero_()
            mem.sym_mem.filled.zero_()

            with torch.no_grad():
                epi = mem.ltm.episodic
                epi.filled.fill_(2)
                epi.global_step.fill_(5)
                epi.keys.zero_()
                epi.state_keys.zero_()
                epi.vals.zero_()
                epi.prio.zero_()
                epi.keys[:, 0, 0] = 1.0
                epi.keys[:, 1, 1] = 1.0
                epi.state_keys[:, 0, 0] = 1.0
                epi.state_keys[:, 1, 1] = 1.0
                epi.vals[:, 0] = torch.randn(2, int(mem.memory_dim), device=self.device)
                epi.vals[:, 1] = torch.randn(2, int(mem.memory_dim), device=self.device)
                epi.prio[:, :2] = torch.tensor([[1.0, 0.8], [0.7, 0.6]], device=self.device)
                epi.rew_abs[:, :2] = torch.tensor([[5.0, 0.0], [3.0, 1.0]], device=self.device)
                epi.step[:, :2] = torch.tensor([[4, 5], [4, 5]], device=self.device, dtype=torch.long)
                epi.touch[:, :2] = 1
                epi.source[:, :2] = MemoryType.SRC_REAL

            sem_before = int(mem.ltm.semantic.filled.max().item())
            sym_before = int(mem.sym_mem.filled.item())
            mem.ConsolidateMemory(topk=2)
            sem_after = int(mem.ltm.semantic.filled.max().item())
            sym_after = int(mem.sym_mem.filled.item())
            assert sem_after > sem_before, f"semantic not filled: {sem_before}->{sem_after}"
            assert sym_after > sym_before, f"symbolic not filled: {sym_before}->{sym_after}"

            print("Consolidation semantic/symbolic fill test passed.")
            return True
        except AssertionError as e:
            print(f"Consolidation semantic/symbolic fill test failed: {e}")
            return False
        except Exception as e:
            print(f"Consolidation semantic/symbolic fill test error: {e}")
            return False

    def TestEventCodeSeparationCompletion(self):
        try:
            cfg = dict(
                inputDim=16, ssmStateDim=16, memoryDim=16, memorySize=6,
                symSize=8, ltmSize=6, nsK=4, outputDim=16,
                gwsSlots=4, gwsTtl=6, compressEvery=100, emotionDim=4)
            cfg = self.FilterKwargs(MemoryExtractor, cfg)
            mem = MemoryExtractor(**cfg).to(self.device).eval()

            B = 2
            x0 = torch.randn(1, cfg["inputDim"], device=self.device)
            x = x0.expand(B, -1).clone()
            visual = SimpleNamespace(
                IntegratedFeat=torch.zeros(B, 1024, device=self.device),
                MotionToken=torch.zeros(B, 512, device=self.device),
                PredErrorToken=torch.zeros(B, 512, device=self.device),)
            visual.IntegratedFeat[1, 0] = 5.0
            visual.MotionToken[1, 0] = -3.0
            ocr = torch.zeros(B, 512, device=self.device)
            intent = torch.zeros(B, 512, device=self.device)
            ocr[1, 0] = 2.0
            intent[1, 0] = -2.0
            emotion = torch.zeros(B, cfg["emotionDim"], device=self.device)
            emotion[1, 0] = 1.0
            td = torch.tensor([0.0, 1.0], device=self.device)
            uncertainty = torch.tensor([0.0, 0.2], device=self.device)
            risk = torch.tensor([0.0, 0.4], device=self.device)
            confidence = torch.tensor([1.0, 0.8], device=self.device)

            dense, event_key = mem.BuildEventCode(
                x,
                visualState=visual,
                ocrSemantic=ocr,
                intentHint=intent,
                emotion=emotion,
                tdError=td,
                uncertainty=uncertainty,
                risk=risk,
                confidence=confidence,)
            assert dense.shape == (B, cfg["memoryDim"])
            assert event_key.shape == (B, cfg["memoryDim"])
            assert torch.isfinite(dense).all() and torch.isfinite(event_key).all()
            assert torch.linalg.norm(event_key[0] - event_key[1]).item() > 1e-4

            gate = mem.event_completion_gate(torch.cat([dense, dense, dense, td.view(B, 1)], dim=-1))
            assert torch.isfinite(gate).all()
            assert bool(((gate >= 0.0) & (gate <= 1.0)).all().item())

            print("Event-code separation/completion test passed.")
            return True
        except AssertionError as e:
            print(f"Event-code separation/completion test failed: {e}")
            return False
        except Exception as e:
            print(f"Event-code separation/completion test error: {e}")
            return False

    def TestStoredKeyNormalizationContract(self):
        try:
            cfg = dict(
                inputDim=16, ssmStateDim=16, memoryDim=16, memorySize=8,
                symSize=8, ltmSize=8, nsK=8, outputDim=16,
                gwsSlots=4, gwsTtl=6, compressEvery=100, emotionDim=4)
            cfg = self.FilterKwargs(MemoryExtractor, cfg)
            mem = MemoryExtractor(**cfg).to(self.device).eval()

            B = 2
            x = torch.randn(B, cfg["inputDim"], device=self.device)
            td = torch.full((B,), 2.0, device=self.device)
            reward = torch.full((B,), -3.0, device=self.device)
            emotion = self.MakeEmotion(B, mem)
            _ = self.CallMemForward(mem, x, tdError=td, reward=reward, emotion=emotion)
            mem.FlushPendingWrites()

            def assert_unit(name: str, keys: torch.Tensor, mask: torch.Tensor):
                if not bool(mask.any().item()):
                    raise AssertionError(f"{name} has no valid keys to check")
                norms = torch.linalg.vector_norm(keys[mask], ord=2, dim=-1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                    raise AssertionError(f"{name} key norms not unit: {norms.detach().cpu().tolist()}")

            kv_mask = torch.arange(int(mem.memory_size), device=self.device).view(1, -1) < mem.memory_filled.view(-1, 1)
            assert_unit("kv", mem.memory_keys, kv_mask)

            gws_mask = (mem.gws.ttl > 0) & (mem.gws.priority > 0)
            assert_unit("gws", mem.gws.keys, gws_mask)

            sem = mem.ltm.semantic
            sem_mask = torch.arange(int(sem.capacity), device=self.device).view(1, -1) < sem.filled.view(-1, 1)
            assert_unit("ltm_sem", sem.keys, sem_mask)

            epi = mem.ltm.episodic
            epi_mask = torch.arange(int(epi.capacity), device=self.device).view(1, -1) < epi.filled.view(-1, 1)
            assert_unit("ltm_epi", epi.keys, epi_mask)
            assert_unit("ltm_epi_state", epi.state_keys, epi_mask)

            sym_n = int(mem.sym_mem.filled.item())
            if sym_n <= 0:
                raise AssertionError("sym_mem has no valid keys to check")
            sym_norms = torch.linalg.vector_norm(mem.sym_mem.P_keys[:sym_n], ord=2, dim=-1)
            if not torch.allclose(sym_norms, torch.ones_like(sym_norms), atol=1e-4, rtol=1e-4):
                raise AssertionError(f"sym key norms not unit: {sym_norms.detach().cpu().tolist()}")

            print("Stored-key normalization contract test passed.")
            return True
        except AssertionError as e:
            print(f"Stored-key normalization contract test failed: {e}")
            return False
        except Exception as e:
            print(f"Stored-key normalization contract test error: {e}")
            return False

    def TestEnsureBClearsOnResize(self):
        try:
            cfg = dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=6,
                symSize=8, ltmSize=6, nsK=4, outputDim=8,
                gwsSlots=3, gwsTtl=4, compressEvery=100, emotionDim=4)
            cfg = self.FilterKwargs(MemoryExtractor, cfg)
            mem = MemoryExtractor(**cfg).to(self.device).eval()

            with torch.no_grad():
                mem.memory_keys[0, 0, 0] = 9.0
                mem.memory_values[0, 0, 0] = 8.0
                mem.memory_filled[0] = 1
                mem.time_step[0] = 7
                mem.ltm.semantic.keys[0, 0, 0] = 6.0
                mem.ltm.semantic.filled[0] = 1
                mem.ltm.episodic.keys[0, 0, 0] = 5.0
                mem.ltm.episodic.state_keys[0, 0, 0] = 4.0
                mem.ltm.episodic.filled[0] = 1

            mem.EnsureB(3)
            mem.ltm.semantic.EnsureB(3)
            mem.ltm.episodic.EnsureB(3)

            assert int(mem.memory_keys.size(0)) == 3
            assert int(mem.memory_filled.sum().item()) == 0
            assert int(mem.time_step.sum().item()) == 0
            assert torch.count_nonzero(mem.memory_keys).item() == 0
            assert torch.count_nonzero(mem.memory_values).item() == 0
            assert int(mem.ltm.semantic.filled.sum().item()) == 0
            assert torch.count_nonzero(mem.ltm.semantic.keys).item() == 0
            assert int(mem.ltm.episodic.filled.sum().item()) == 0
            assert torch.count_nonzero(mem.ltm.episodic.keys).item() == 0
            assert torch.count_nonzero(mem.ltm.episodic.state_keys).item() == 0

            gws = GlobalWorkspace(dim=8, slots=3, defaultTtl=4).to(self.device)
            with torch.no_grad():
                gws.keys[0, 0, 0] = 1.0
                gws.priority[0, 0] = 1.0
                gws.ttl[0, 0] = 4
            gws = gws.to(dtype=torch.float64)
            gws.EnsureB(2)
            assert int(gws.keys.size(0)) == 2
            assert gws.keys.dtype == torch.float64
            assert torch.count_nonzero(gws.keys).item() == 0
            assert torch.count_nonzero(gws.priority).item() == 0
            assert torch.count_nonzero(gws.ttl).item() == 0

            print("EnsureB clears-on-resize test passed.")
            return True
        except AssertionError as e:
            print(f"EnsureB clears-on-resize test failed: {e}")
            return False
        except Exception as e:
            print(f"EnsureB clears-on-resize test error: {e}")
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

    def TestMemoryValueModulatedForward(self):
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
            td = torch.randn(B, device=self.device).clamp(-1.0, 1.0)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem)
            uncertainty = torch.linspace(0.0, 0.8, B, device=self.device)
            risk = torch.linspace(0.1, 0.9, B, device=self.device)
            confidence = torch.linspace(0.95, 0.35, B, device=self.device)

            y = self.CallMemForward(
                mem,
                x,
                tdError=td,
                reward=rwd,
                emotion=emotion,
                uncertainty=uncertainty,
                risk=risk,
                confidence=confidence)
            assert y.shape == (B, cfg.get("outputDim", int(mem.output_dim)))
            assert torch.isfinite(y).all()
            mem.FlushPendingWrites()
            assert (mem.memory_filled >= 0).all()

            print("Memory value-modulated forward test passed.")
            return True
        except AssertionError as e:
            print(f"Memory value-modulated forward test failed: {e}")
            return False
        except Exception as e:
            print(f"Memory value-modulated forward test error: {e}")
            return False

    def TestMemoryExtractorIOShapes(self):
        try:
            cfg = dict(
                inputDim=1024,
                ssmStateDim=1024,
                memoryDim=1024,
                memorySize=32,
                symSize=64,
                ltmSize=64,
                nsK=32,
                outputDim=1024,
                gwsSlots=8,
                gwsTtl=6,
                compressEvery=50,
                emotionDim=64,)

            cfg = self.FilterKwargs(MemoryExtractor, cfg)

            mem = MemoryExtractor(**cfg).to(self.device)
            mem.eval()

            B = 2
            x = torch.randn(B, cfg["inputDim"], device=self.device)
            td = torch.randn(B, device=self.device)
            reward = torch.randn(B, device=self.device)
            emotion = torch.randn(B, cfg["emotionDim"], device=self.device)
            source_label = torch.zeros(B, dtype=torch.int8, device=self.device)

            def print_shape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            with torch.no_grad():
                print_shape("input.x", x)
                print_shape("input.tdError", td)
                print_shape("input.emotion", emotion)
                print_shape("input.reward", reward)
                print_shape("input.sourceLabel", source_label)

                y = self.CallMemForward(
                    mem,
                    x,
                    tdError=td,
                    emotion=emotion,
                    reward=reward,
                    sourceLabel=source_label,)
                print_shape("output.y", y)

            assert y.shape == (B, cfg["outputDim"]), f"Output shape mismatch: {y.shape}"
            print("MemoryExtractor IO shapes test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor IO shapes test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor IO shapes test error: {e}")
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

            torch.manual_seed(1234)
            y1 = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)

            _ = self.CallMemForward(mem, x, tdError=td, reward=rwd, emotion=emotion)

            mem.LoadState(str(path))
            try:
                path.unlink()
            except OSError:
                pass

            torch.manual_seed(1234)
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

            B = 3
            x = torch.randn(B, cfg.get("inputDim", 32), device=self.device)
            td = torch.randn(B, device=self.device)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem1)

            for _ in range(5):
                _ = self.CallMemForward(mem1, x, tdError=td, reward=rwd, emotion=emotion)

            mem1.pending.clear()
            state = mem1.ExportState()
            assert tuple(state["h_state"].shape[:1]) == (B,)
            assert tuple(state["memory_keys"].shape[:1]) == (B,)

            missing_state = dict(state)
            del missing_state["ltm_epi_state_keys"]
            strict_import_rejected = False
            try:
                mem1.ImportState(missing_state)
            except TypeError:
                strict_import_rejected = True
            strict_merge_rejected = False
            try:
                mem1.MergeMemoryState(missing_state)
            except TypeError:
                strict_merge_rejected = True

            invalid_shape_state = dict(state)
            invalid_shape_state["gws_keys"] = state["gws_keys"][..., :-1]
            invalid_shape_rejected = False
            try:
                mem1.ImportState(invalid_shape_state)
            except ValueError:
                invalid_shape_rejected = True

            mixed_dtype_state = {
                name: (
                    value.to(torch.float64)
                    if value.dtype.is_floating_point
                    else value.to(torch.int32)
                    if value.dtype in (torch.int64, torch.int8)
                    else value)
                for name, value in state.items()}

            mem2 = MemoryExtractor(**cfg).to(self.device).eval()
            mem2.EnsureB(B)
            mem2.gws.EnsureB(B)
            mem2.ltm.semantic.EnsureB(B)
            mem2.ltm.episodic.EnsureB(B)
            mem2.load_state_dict(mem1.state_dict(), strict=True)
            mem2.ltm.episodic.touch.zero_()
            mem2.ImportState(mixed_dtype_state)

            self.AssertClose(mem1.ltm.episodic.touch.float(), mem2.ltm.episodic.touch.float(), msg="Episodic touch ImportState")
            assert strict_import_rejected
            assert strict_merge_rejected
            assert invalid_shape_rejected
            assert mem2.memory_keys.dtype == mem1.memory_keys.dtype
            assert mem2.memory_steps.dtype == mem1.memory_steps.dtype
            assert mem2.memory_source.dtype == mem1.memory_source.dtype
            assert mem2.sym_mem.P_keys.dtype == mem1.sym_mem.P_keys.dtype
            assert mem2.sym_mem.step.dtype == mem1.sym_mem.step.dtype

            torch.manual_seed(5678)
            y1 = self.CallMemForward(mem1, x, tdError=td, reward=rwd, emotion=emotion)
            torch.manual_seed(5678)
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

    def TestMergeStateCrossCapacityAndMalformedReject(self):
        try:
            common = dict(
                inputDim=8,
                ssmStateDim=8,
                memoryDim=8,
                nsK=4,
                outputDim=8,
                gwsTtl=4,
                compressEvery=10_000,
                emotionDim=4,)
            src = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                **common,
                memorySize=4,
                symSize=4,
                ltmSize=4,
                gwsSlots=3,))).to(self.device).eval()
            dst = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                **common,
                memorySize=2,
                symSize=2,
                ltmSize=2,
                gwsSlots=2,))).to(self.device).eval()

            B_src, B_dst = 2, 3
            src.EnsureB(B_src)
            src.gws.EnsureB(B_src)
            src.ltm.semantic.EnsureB(B_src)
            src.ltm.episodic.EnsureB(B_src)
            dst.EnsureB(B_dst)
            dst.gws.EnsureB(B_dst)
            dst.ltm.semantic.EnsureB(B_dst)
            dst.ltm.episodic.EnsureB(B_dst)

            with torch.no_grad():
                src.memory_filled.copy_(torch.tensor([3, 1], device=self.device))
                src.ltm.semantic.filled.copy_(torch.tensor([3, 1], device=self.device))
                src.ltm.episodic.filled.copy_(torch.tensor([3, 1], device=self.device))
                for t in range(3):
                    value = float(t + 1)
                    src.memory_keys[0, t, 0] = value
                    src.memory_values[0, t, 0] = value
                    src.memory_importance[0, t] = value
                    src.memory_steps[0, t] = t + 1
                    src.ltm.semantic.keys[0, t, 0] = value
                    src.ltm.semantic.vals[0, t, 0] = value
                    src.ltm.semantic.prio[0, t] = value
                    src.ltm.episodic.keys[0, t, 0] = value
                    src.ltm.episodic.state_keys[0, t, 0] = value + 10.0
                    src.ltm.episodic.vals[0, t, 0] = value
                    src.ltm.episodic.prio[0, t] = value
                    src.ltm.episodic.rew[0, t] = value
                    src.ltm.episodic.rew_abs[0, t] = value
                    src.sym_mem.P_keys[t, t % src.sym_mem.K] = 1.0
                    src.sym_mem.P_vals[t, 0] = value
                    src.sym_mem.prio[t] = value
                src.sym_mem.filled.fill_(3)
                src.memory_keys[1, 0, 0] = 9.0
                src.memory_values[1, 0, 0] = 9.0
                src.memory_importance[1, 0] = 9.0
                src.ltm.semantic.keys[1, 0, 0] = 9.0
                src.ltm.semantic.vals[1, 0, 0] = 9.0
                src.ltm.semantic.prio[1, 0] = 9.0
                src.ltm.episodic.keys[1, 0, 0] = 9.0
                src.ltm.episodic.state_keys[1, 0, 0] = 19.0
                src.ltm.episodic.vals[1, 0, 0] = 9.0
                src.ltm.episodic.prio[1, 0] = 9.0
                src.gws.keys[:, :, 0] = 7.0
                src.gws.vals[:, :, 0] = 8.0
                src.gws.priority.fill_(1.0)
                src.gws.ttl.fill_(2)

            state = src.ExportState()
            dst.MergeMemoryState(state, mergeGws=True)
            assert dst.memory_filled.tolist() == [2, 1, 0], dst.memory_filled.tolist()
            assert set(dst.memory_values[0, :2, 0].tolist()) == {2.0, 3.0}, dst.memory_values[0, :2, 0].tolist()
            assert dst.memory_values[1, 0, 0].item() == 9.0, dst.memory_values[1, 0, 0].item()
            assert dst.ltm.semantic.filled.tolist() == [2, 1, 0], dst.ltm.semantic.filled.tolist()
            assert dst.ltm.episodic.filled.tolist() == [2, 1, 0], dst.ltm.episodic.filled.tolist()
            assert int(dst.sym_mem.filled.item()) == 2, int(dst.sym_mem.filled.item())
            assert bool((dst.gws.vals[:2, :2, 0] == 8.0).all().item()), dst.gws.vals[:2, :2, 0]

            malformed_feature = dict(state)
            malformed_feature["memory_values"] = state["memory_values"][..., :-1]
            feature_rejected = False
            try:
                dst.MergeMemoryState(malformed_feature)
            except ValueError:
                feature_rejected = True

            malformed_symbol = dict(state)
            malformed_symbol["sym_mem_P_keys"] = state["sym_mem_P_keys"].unsqueeze(0)
            symbol_rejected = False
            try:
                dst.MergeMemoryState(malformed_symbol)
            except ValueError:
                symbol_rejected = True
            assert feature_rejected and symbol_rejected

            print("MergeState cross-capacity and malformed-state test passed.")
            return True
        except AssertionError as e:
            print(f"MergeState cross-capacity test failed: {e}")
            return False
        except Exception as e:
            print(f"MergeState cross-capacity test error: {e}")
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
            mem_peer = MemoryExtractor(**cfg).to(self.device).train()
            mem_peer.load_state_dict(mem.state_dict(), strict=True)

            B = 4
            x = torch.randn(B, cfg.get("inputDim", 32), device=self.device)
            td = torch.randn(B, device=self.device)
            rwd = torch.randn(B, device=self.device)
            emotion = self.MakeEmotion(B, mem)

            torch.manual_seed(91)
            first = self.CallMemForward(
                mem, x, tdError=td, reward=rwd, emotion=emotion)
            torch.manual_seed(91)
            first_peer = self.CallMemForward(
                mem_peer,
                x,
                tdError=td,
                reward=rwd,
                emotion=emotion)
            assert torch.allclose(
                first,
                first_peer,
                atol=1e-7,
                rtol=1e-6)
            assert torch.allclose(
                mem.fast_weights,
                mem_peer.fast_weights,
                atol=1e-7,
                rtol=1e-6)
            assert "fast_weights" not in mem.state_dict()

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

    def TestPartialEpisodeResetPreservesSharedState(self):
        try:
            cfg = dict(
                inputDim=16, ssmStateDim=16, memoryDim=16, memorySize=8,
                symSize=8, ltmSize=8, nsK=4,
                outputDim=16, gwsSlots=4, gwsTtl=6,
                compressEvery=10_000, emotionDim=8)
            mem = MemoryExtractor(
                **self.FilterKwargs(MemoryExtractor, cfg)).to(self.device).eval()
            mem.EnsureB(2)
            mem.gws.EnsureB(2)
            with torch.no_grad():
                mem.h_state.fill_(1.0)
                mem.fast_weights.fill_(1.0)
                mem.ns_prev_P_post.fill_(1.0)
                mem.ns_penalty_vec.fill_(1.0)
                mem.gws.keys[0, 0, 0] = 5.0
                mem.gws.priority[0, 0] = 1.0
                mem.gws.ttl[0, 0] = 5
            key = torch.zeros(2, mem.memory_dim, device=self.device, dtype=mem.dtype)
            value = torch.zeros_like(key)
            key[:, 0] = torch.tensor([1.0, 2.0], device=self.device)
            value[:, 0] = torch.tensor([3.0, 4.0], device=self.device)
            pending_write = (
                "kv",
                (
                    key,
                    value,
                    torch.ones(2, device=self.device, dtype=mem.dtype),
                    torch.zeros(2, mem.emotion_dim, device=self.device, dtype=mem.dtype),
                    torch.zeros(2, device=self.device, dtype=mem.dtype),
                    torch.zeros(2, device=self.device, dtype=torch.int8),))
            mem.pending = [pending_write]

            mem.ResetEpisodeState(torch.tensor(
                [True, False], device=self.device))
            partial_ok = (
                torch.count_nonzero(mem.h_state[0]).item() == 0
                and torch.count_nonzero(mem.h_state[1]).item() > 0
                and torch.count_nonzero(mem.fast_weights[0]).item() == 0
                and torch.count_nonzero(mem.fast_weights[1]).item() > 0
                and float(mem.gws.keys[0, 0, 0].item()) == 5.0
                and float(mem.gws.priority[0, 0].item()) == 1.0
                and mem.memory_filled.tolist() == [1, 1]
                and not mem.pending)

            mem.pending = [pending_write]
            mem.ResetEpisodeState(torch.tensor(
                [True, True], device=self.device))
            full_ok = (
                torch.count_nonzero(mem.gws.keys).item() == 0
                and torch.count_nonzero(mem.gws.priority).item() == 0
                and mem.memory_filled.tolist() == [2, 2]
                and not mem.pending)
            dtype_rejected = False
            try:
                mem.ResetEpisodeState(torch.ones(
                    2, device=self.device, dtype=mem.dtype))
            except TypeError:
                dtype_rejected = True
            device_rejected = False
            try:
                mem.ResetEpisodeState(torch.ones(
                    2, device="meta", dtype=torch.bool))
            except ValueError:
                device_rejected = True
            assert partial_ok and full_ok and dtype_rejected and device_rejected
            print("MemoryExtractor partial episode reset test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor partial episode reset test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor partial episode reset test error: {e}")
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
        with torch.no_grad():
            source_parameters = dict(mem.named_parameters())
            for name, parameter in mem2.named_parameters():
                parameter.copy_(source_parameters[name])

        tmp_path = self.StatePath("memory_state_probe.pth")
        mem.SaveState(str(tmp_path))
        mem2.LoadState(str(tmp_path))
        mem.ImportDurableState(mem.ExportDurableState())
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
            usage = mem.usage_bank.SlotReadout(
                torch.randn(
                    B, 3, ModuleDim.PstIdDim,
                    device=self.device),
                torch.randn(
                    B, 3, ModuleDim.PstAttrDim,
                    device=self.device),
                torch.rand(B, 3, device=self.device))
            total = self.AttachAllInternalLosses(
                mem,
                base + 0.01 * usage.square().mean())

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
        
    def TestSymbolicViolationBackprop(self):
        try:
            cfg = self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16,
                ssmStateDim=16,
                memoryDim=8,
                memorySize=4,
                symSize=4,
                ltmSize=4,
                nsK=4,
                outputDim=8,
                gwsSlots=2,
                gwsTtl=2,
                compressEvery=100,
                emotionDim=4,))
            mem = MemoryExtractor(**cfg).to(self.device).train()
            proposition = torch.full(
                (3, mem.ns_K),
                0.8,
                device=self.device,
                requires_grad=True)
            previous = torch.zeros_like(proposition)

            mem.ResetInternalLoss()
            violation = mem.NsRules(proposition, previous)
            internal_loss = mem.GetInternalLoss()
            proposition_grad = torch.autograd.grad(
                internal_loss,
                proposition,
                retain_graph=True)[0]

            _, auxiliary = mem.sym_rules(proposition, previous)
            expected = mem.ns_lambda * (violation.mean() + auxiliary)
            ok = bool(
                float(violation.detach().mean().item()) > 0.0
                and torch.isfinite(internal_loss).item()
                and float(proposition_grad.detach().abs().sum().item()) > 0.0
                and torch.allclose(
                    internal_loss,
                    expected,
                    atol=1e-7,
                    rtol=1e-6))
            print(f"SymbolicViolationBackprop {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"SymbolicViolationBackprop error: {e}")
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
            "ExportMemoryBankMixedRowValidity": self.TestExportMemoryBankMixedRowValidity(),
            "DurableStatePreservesBatchLanes": self.TestDurableStatePreservesBatchLanes(),
            "DurableStateRejectsNonCurrentSchema": self.TestDurableStateRejectsNonCurrentSchema(),
            "GlobalWorkspaceSharedBatch": self.TestGlobalWorkspaceSharedBatch(),
            "SourceConfidenceOrdering": self.TestSourceConfidenceOrdering(),
            "ExportMemoryBankMetaBudget": self.TestExportMemoryBankMetaBudget(),
            "ConsolidationWritesSemanticAndSymbolic": self.TestConsolidationWritesSemanticAndSymbolic(),
            "EventCodeSeparationCompletion": self.TestEventCodeSeparationCompletion(),
            "StoredKeyNormalizationContract": self.TestStoredKeyNormalizationContract(),
            "EnsureBClearsOnResize": self.TestEnsureBClearsOnResize(),
            "ReorderMemorySteps": self.TestReorderMemorySteps(),
            "MemoryExtractorForward": self.TestMemoryExtractorForward(),
            "MemoryValueModulatedForward": self.TestMemoryValueModulatedForward(),
            "MemoryExtractorIOShapes": self.TestMemoryExtractorIOShapes(),
            "StateSaveRestore": self.TestStateSaveRestore(),
            "ExportImportRoundTrip": self.TestExportImportStateRoundTrip(),
            "MergeStateCrossCapacityAndMalformedReject": self.TestMergeStateCrossCapacityAndMalformedReject(),
            "AutoCompress": self.TestAutoCompress(),
            "ResetAndSoftReset": self.TestResetAndSoftReset(),
            "PartialEpisodeResetPreservesSharedState": self.TestPartialEpisodeResetPreservesSharedState(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NumericalStability": self.TestNumericalStability(),
            "AllTrainableParamsHaveGrad": self.TestAllTrainableParamsHaveGrad(),
            "SymbolicViolationBackprop": self.TestSymbolicViolationBackprop(),
            "LossDecreases": self.TestLossDecreases()}

        passed = sum(1 for v in results.values() if v)
        print(f"\nMemory module tests: {passed}/{len(results)} passed.")
        return results
