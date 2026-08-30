from __future__ import annotations
from typing import Any, Optional, Tuple, Dict, List
from pathlib import Path
from types import SimpleNamespace, MethodType
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
import hashlib


ONTOLOGY_REALM_SELF = 0
ONTOLOGY_REALM_EXTERNAL = 1
ONTOLOGY_REALM_VIRTUAL = 2
ONTOLOGY_REALM_EFFECT = 3
ONTOLOGY_REALM_UNKNOWN = 4
ONTOLOGY_AGENCY_SELF = 0
ONTOLOGY_AGENCY_EXTERNAL = 1
ONTOLOGY_AGENCY_AUTONOMOUS = 2
ONTOLOGY_AGENCY_UNKNOWN = 4



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


def UnknownEpisodicTypeMetadata(
    realm: torch.Tensor,
    *,
    dtype: torch.dtype,) -> Dict[str, torch.Tensor]:
    B = int(realm.size(0))
    return {
        "agency_id": torch.full(
            (B,), ONTOLOGY_AGENCY_UNKNOWN, device=realm.device, dtype=torch.int8),
        "motion_layer_prob": torch.zeros(
            B, 5, device=realm.device, dtype=dtype),
        "observed_surface_parent_index": torch.full(
            (B,), -1, device=realm.device, dtype=torch.long),
        "surface_uv": torch.zeros(
            B, 2, device=realm.device, dtype=dtype),
        "surface_uv_confidence": torch.zeros(
            B, device=realm.device, dtype=dtype),
        "verification": torch.zeros(
            B, device=realm.device, dtype=dtype),
        "content_change": torch.zeros(
            B, device=realm.device, dtype=dtype),}


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


def SourceProbabilityReal(
    source: torch.Tensor,
    reliability: torch.Tensor,) -> torch.Tensor:
    probability = torch.ones_like(reliability)
    probability = torch.where(
        source == MemoryType.SRC_MIXED,
        0.50 + 0.40 * reliability,
        probability)
    probability = torch.where(
        source == MemoryType.SRC_IMAGINE,
        0.05 + 0.20 * reliability,
        probability)
    return probability


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
    imagineExtraBeta: float = 0.018,
    contentTemperature: float = 0.25,
    priorityWeight: float = 0.50,
    rehearsalWeight: float = 0.10,
    confidenceWeight: float = 0.50,
    rewardWeight: float = 0.50,
    noveltyWeight: float = 0.25,) -> torch.Tensor:
    if keys.dim() == 2:
        sim = torch.matmul(query, keys.t())
    elif keys.dim() == 3:
        sim = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
    else:
        raise ValueError(f"keys must be 2D or 3D, got {keys.dim()}D")

    age_f = age.float()
    if source is None:
        source = torch.zeros_like(age, dtype=torch.int8)
    src = source

    beta = torch.full_like(age_f, float(baseAgeBeta))
    beta = torch.where(src == MemoryType.SRC_MIXED, beta + float(mixedExtraBeta), beta)
    beta = torch.where(src == MemoryType.SRC_IMAGINE, beta + float(imagineExtraBeta), beta)
    score = sim / float(contentTemperature) - beta * age_f
    if priority is not None:
        score = score + float(priorityWeight) * torch.log(priority.float() + 1e-6)
    if touch is not None:
        score = score + float(rehearsalWeight) * torch.log1p(touch.float())

    conf = SourceConfidence(src, dtype=sim.dtype) if confidence is None else confidence.float()
    score = score + float(confidenceWeight) * torch.log(conf + 1e-6)

    if rewardAbs is not None:
        score = score + float(rewardWeight) * torch.tanh(rewardAbs.float())
    if novelty is not None:
        score = score + float(noveltyWeight) * torch.tanh(novelty.float())
    if validMask is not None:
        score = score.masked_fill(~validMask, -torch.inf)
    return score


def SourceBalancedRecall(
    attention: torch.Tensor,
    values: torch.Tensor,
    probabilityReal: torch.Tensor,
    imaginedGain: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    real_weight = attention * probabilityReal
    imagined_weight = attention * (1.0 - probabilityReal)
    real_mass = real_weight.sum(dim=-1, keepdim=True)
    imagined_mass = imagined_weight.sum(dim=-1, keepdim=True)
    real_content = torch.einsum(
        "bk,bkd->bd",
        real_weight / (real_mass + 1e-6),
        values)
    imagined_content = torch.einsum(
        "bk,bkd->bd",
        imagined_weight / (imagined_mass + 1e-6),
        values)
    total_mass = real_mass + imaginedGain * imagined_mass
    recall = (
        real_mass * real_content
        + imaginedGain * imagined_mass * imagined_content
    ) / (total_mass + 1e-6)
    recall = torch.where(total_mass > 0, recall, torch.zeros_like(recall))
    return recall, real_mass, imagined_mass


def NullGatedTopKWeights(
    scores: torch.Tensor,
    nullLogit: torch.Tensor,
    candidateValid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    null = nullLogit.expand(scores.size(0), 1)
    valid_scores = scores.masked_fill(~candidateValid, -torch.inf)
    base_weights = F.softmax(
        torch.cat([valid_scores, null], dim=-1),
        dim=-1)

    margin = scores - null
    accepted = candidateValid & (margin > 0)
    soft_gate = torch.sigmoid(margin) * candidateValid.to(scores.dtype)
    hard_gate = accepted.to(scores.dtype)
    evidence_gate = hard_gate + soft_gate - soft_gate.detach()

    soft_any = 1.0 - torch.prod(1.0 - soft_gate, dim=-1, keepdim=True)
    hard_any = accepted.any(dim=-1, keepdim=True).to(scores.dtype)
    branch_evidence = hard_any + soft_any - soft_any.detach()

    memory_weights = base_weights[:, :scores.size(1)] * evidence_gate
    normalizer = memory_weights.sum(dim=-1, keepdim=True) + base_weights[:, -1:]
    return memory_weights / normalizer, accepted, branch_evidence

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

        W_excl, W_or, A_imp = self.Weights() #[G_excl, K] , [G_or, K] , [K, K]

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
            imp_pen = weighted.sum(dim=(1, 2)) / float(denom) #[B]
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

    def forward(self, x: torch.Tensor): # x: [B, inDim]
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
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        B0 = 1
        self.register_buffer("P_keys", torch.zeros(B0, self.capacity, self.K))
        self.register_buffer("P_vals", torch.zeros(B0, self.capacity, self.K))
        self.register_buffer("prio", torch.zeros(B0, self.capacity))
        self.register_buffer("created_step", torch.zeros(B0, self.capacity, dtype=torch.long))
        self.register_buffer("last_access_step", torch.zeros(B0, self.capacity, dtype=torch.long))
        self.register_buffer("last_rehearsal_step", torch.zeros(B0, self.capacity, dtype=torch.long))
        self.register_buffer("touch", torch.zeros(B0, self.capacity, dtype=torch.long))
        self.register_buffer("source", torch.zeros(B0, self.capacity, dtype=torch.int8))
        self.register_buffer("source_confidence", torch.zeros(B0, self.capacity))
        self.register_buffer("filled", torch.zeros(B0, dtype=torch.long))
        self.register_buffer("global_step", torch.zeros(B0, dtype=torch.long))

    @property
    def step(self) -> torch.Tensor:
        return self.created_step

    @torch.no_grad()
    def EnsureB(self, B: int):
        if int(self.P_keys.size(0)) == int(B):
            return
        self.P_keys = self.P_keys.new_zeros(B, self.capacity, self.K)
        self.P_vals = self.P_vals.new_zeros(B, self.capacity, self.K)
        self.prio = self.prio.new_zeros(B, self.capacity)
        self.created_step = self.created_step.new_zeros(B, self.capacity)
        self.last_access_step = self.last_access_step.new_zeros(B, self.capacity)
        self.last_rehearsal_step = self.last_rehearsal_step.new_zeros(B, self.capacity)
        self.touch = self.touch.new_zeros(B, self.capacity)
        self.source = self.source.new_zeros(B, self.capacity)
        self.source_confidence = self.source_confidence.new_zeros(B, self.capacity)
        self.filled = self.filled.new_zeros(B)
        self.global_step = self.global_step.new_zeros(B)

    @torch.no_grad()
    def StepTick(self):
        self.global_step.add_(1)

    @torch.no_grad()
    def Store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        score: torch.Tensor,
        source: torch.Tensor,
        sourceConfidence: torch.Tensor,
        writeMask: Optional[torch.Tensor] = None,):
        B = int(key.size(0))
        self.EnsureB(B)
        write_mask = torch.ones(B, device=key.device, dtype=torch.bool) if writeMask is None else writeMask
        for b in range(B):
            if not bool(write_mask[b].item()):
                continue
            filled = int(self.filled[b].item())
            if filled < self.capacity:
                i = filled
                self.filled[b] += 1
            else:
                age = (self.global_step[b] - self.created_step[b]).clamp(min=0).float()
                eff = self.prio[b] * torch.exp(-0.01 * age)
                eff = eff * self.source_confidence[b]
                i = int(torch.argmin(eff).item())

            self.P_keys[b, i] = key[b]
            self.P_vals[b, i] = value[b]
            self.prio[b, i] = score[b]
            self.created_step[b, i] = self.global_step[b]
            self.last_access_step[b, i] = self.global_step[b]
            self.last_rehearsal_step[b, i] = self.global_step[b]
            self.touch[b, i] = 1
            self.source[b, i] = source[b]
            self.source_confidence[b, i] = sourceConfidence[b]

    def Retrieve(
        self,
        qSym: torch.Tensor, #[B, K]
        topK: int = 8,
        recentBias: float = 0.05,
        returnEvidence: bool = False,):
        B = int(qSym.size(0))
        self.EnsureB(B)
        slots = torch.arange(self.capacity, device=qSym.device).view(1, self.capacity)
        valid = slots < self.filled.view(B, 1)
        any_valid = valid.any(dim=1)
        if not bool(any_valid.any().item()):
            out = qSym.new_zeros(B, self.K)
            evidence = qSym.new_zeros(B, 1)
            return (out, evidence) if returnEvidence else out

        age = (self.global_step.view(B, 1) - self.created_step).clamp(min=0).float()
        sim = UnifiedMemoryScore(
            qSym,
            self.P_keys,
            age=age,
            priority=self.prio,
            touch=self.touch,
            confidence=self.source_confidence,
            source=self.source,
            validMask=valid,
            baseAgeBeta=float(recentBias),
            mixedExtraBeta=0.02,
            imagineExtraBeta=0.05,)
        sim = torch.where(any_valid.view(B, 1), sim, torch.zeros_like(sim))
        k = max(1, min(int(topK), int(self.filled.max().item()), self.capacity))
        top_sim, idx = StableTopk(sim, k)
        candidate_valid = torch.gather(valid, 1, idx)
        w, accepted, evidence = NullGatedTopKWeights(
            top_sim,
            self.null_logit,
            candidate_valid)
        w = torch.where(any_valid.view(B, 1), w, torch.zeros_like(w))
        evidence = torch.where(
            any_valid.view(B, 1), evidence, torch.zeros_like(evidence))
        selected_vals = torch.gather(
            self.P_vals,
            1,
            idx.unsqueeze(-1).expand(B, k, self.K))
        out = torch.einsum("bk,bkd->bd", w, selected_vals)

        with torch.no_grad():
            b_idx = torch.arange(B, device=qSym.device).unsqueeze(1).expand_as(idx)
            self.touch[b_idx[accepted], idx[accepted]] += 1
            self.last_access_step[b_idx[accepted], idx[accepted]] = (
                self.global_step.unsqueeze(1).expand_as(idx)[accepted])

        return (out, evidence) if returnEvidence else out #[B, K]

    @torch.no_grad()
    def Reset(self):
        self.P_keys.zero_()
        self.P_vals.zero_()
        self.prio.zero_()
        self.created_step.zero_()
        self.last_access_step.zero_()
        self.last_rehearsal_step.zero_()
        self.touch.zero_()
        self.source.zero_()
        self.source_confidence.zero_()
        self.filled.fill_(0)
        self.global_step.fill_(0)

    @torch.no_grad()
    def ResetRows(self, done: torch.Tensor):
        self.P_keys[done] = 0
        self.P_vals[done] = 0
        self.prio[done] = 0
        self.created_step[done] = 0
        self.last_access_step[done] = 0
        self.last_rehearsal_step[done] = 0
        self.touch[done] = 0
        self.source[done] = 0
        self.source_confidence[done] = 0
        self.filled[done] = 0
        self.global_step[done] = 0


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
        self.null_logit = nn.Parameter(torch.tensor(0.0))

        B0 = 1
        self.register_buffer("keys", torch.zeros(B0, slots, dim))
        self.register_buffer("vals", torch.zeros(B0, slots, dim))
        self.register_buffer("priority", torch.zeros(B0, slots))
        self.register_buffer("ttl", torch.zeros(B0, slots, dtype=torch.long))
        self.register_buffer("created_step", torch.zeros(B0, slots, dtype=torch.long))
        self.register_buffer("last_step", torch.zeros(B0, slots, dtype=torch.long))
        self.register_buffer("last_rehearsal_step", torch.zeros(B0, slots, dtype=torch.long))
        self.register_buffer("touch", torch.zeros(B0, slots, dtype=torch.long))
        self.register_buffer("source", torch.zeros(B0, slots, dtype=torch.int8))
        self.register_buffer("source_confidence", torch.zeros(B0, slots))
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
        self.created_step = self.created_step.new_zeros(B, self.slots)
        self.last_step = self.last_step.new_zeros(B, self.slots)
        self.last_rehearsal_step = self.last_rehearsal_step.new_zeros(B, self.slots)
        self.touch = self.touch.new_zeros(B, self.slots)
        self.source = self.source.new_zeros(B, self.slots)
        self.source_confidence = self.source_confidence.new_zeros(B, self.slots)
        self.global_step = self.global_step.new_zeros(B)


    @torch.no_grad()
    def Reset(self):
        self.keys.zero_()
        self.vals.zero_()
        self.priority.zero_()
        self.ttl.zero_()
        self.created_step.zero_()
        self.last_step.zero_()
        self.last_rehearsal_step.zero_()
        self.touch.zero_()
        self.source.zero_()
        self.source_confidence.zero_()
        self.global_step.zero_()

    @torch.no_grad()
    def ResetRows(self, done: torch.Tensor):
        self.keys[done] = 0
        self.vals[done] = 0
        self.priority[done] = 0
        self.ttl[done] = 0
        self.created_step[done] = 0
        self.last_step[done] = 0
        self.last_rehearsal_step[done] = 0
        self.touch[done] = 0
        self.source[done] = 0
        self.source_confidence[done] = 0
        self.global_step[done] = 0

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
            self.created_step[expired] = 0
            self.last_step[expired] = 0
            self.last_rehearsal_step[expired] = 0
            self.touch[expired] = 0
            self.source[expired] = 0
            self.source_confidence[expired] = 0

    @torch.no_grad()
    def Write(
        self,
        key: torch.Tensor, # [B, Dim]
        val: torch.Tensor, # [B, Dim]
        *,
        priority: Optional[torch.Tensor] = None, # [B]
        ttl: Optional[torch.Tensor] = None, # [B]
        tagId: Optional[torch.Tensor] = None, # [B]
        sourceConfidence: Optional[torch.Tensor] = None, # [B]
        writeMask: Optional[torch.Tensor] = None, # [B]
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
        if sourceConfidence is None:
            source_conf = SourceConfidence(tag_t, dtype=key.dtype)
        else:
            source_conf = sourceConfidence
        if writeMask is None:
            write_mask = torch.ones(B, device=key.device, dtype=torch.bool)
        else:
            write_mask = writeMask

        out_idx = torch.full((B,), -1, device=key.device, dtype=torch.long)
        for i in range(B):
            if not bool(write_mask[i].item()):
                continue
            empty = (self.ttl[i] <= 0) | (self.priority[i] <= 0)
            if bool(empty.any().item()):
                idx = int(empty.float().argmax().item())
            else:
                age = (self.global_step[i] - self.last_step[i]).clamp(min=0).float()
                eff = self.priority[i] * torch.exp(-age * self.recency_temp)
                eff = eff * self.source_confidence[i]
                idx = int(torch.argmin(eff).item())

            self.keys[i, idx] = key[i]
            self.vals[i, idx] = val[i]
            self.priority[i, idx] = pr[i]
            self.ttl[i, idx] = ttl_t[i]
            self.created_step[i, idx] = self.global_step[i]
            self.last_step[i, idx] = self.global_step[i]
            self.last_rehearsal_step[i, idx] = self.global_step[i]
            self.touch[i, idx] = 1
            self.source[i, idx] = tag_t[i]
            self.source_confidence[i, idx] = source_conf[i]
            out_idx[i] = idx

        return out_idx # [B]

    def Attend(
        self,
        query: torch.Tensor, # [B, Dim]
        *,
        topk: int = 4,
        tagMask: Optional[List[int]] = None,
        returnEvidence: bool = False,):

        B = int(query.size(0))

        self.EnsureB(B)

        alive = (self.ttl > 0) & (self.priority > 0) #[B, slots]

        if tagMask is not None:
            allowed = torch.zeros_like(alive, dtype=torch.bool)
            for t in tagMask:
                allowed |= (self.source == int(t))
            alive = alive & allowed

        any_alive = alive.any(dim=1)

        q = query #[B, dim]
        k = self.keys #[B, slots, dim]
        age = (self.global_step.view(B, 1) - self.created_step).clamp(min=0).float()
        sim = UnifiedMemoryScore(
            q,
            k,
            age=age,
            priority=self.priority,
            touch=self.touch,
            confidence=self.source_confidence,
            source=self.source,
            validMask=alive,
            baseAgeBeta=float(self.recency_temp),
            mixedExtraBeta=float(self.recency_temp) * 0.5,
            imagineExtraBeta=float(self.recency_temp),)

        kk = max(1, min(int(topk), self.slots))

        if not bool(any_alive.all()):
            sim = sim.clone()
            sim[~any_alive] = 0.0

        top_sim, top_idx = StableTopk(sim, kk) #[B, kk]
        candidate_alive = torch.gather(alive, 1, top_idx)
        w, accepted, evidence = NullGatedTopKWeights(
            top_sim,
            self.null_logit,
            candidate_alive)

        gather_idx = top_idx.unsqueeze(-1).expand(B, kk, self.dim)
        v_top = torch.gather(self.vals, dim=1, index=gather_idx) #[B, kk, dim]

        out = torch.einsum("bk,bkd->bd", w, v_top) #[B, dim]

        if not bool(any_alive.all()):
            out = out.clone()
            out[~any_alive] = 0.0
            evidence = evidence.clone()
            evidence[~any_alive] = 0.0

        with torch.no_grad():
            b_idx = torch.arange(B, device=query.device).unsqueeze(1).expand_as(top_idx)
            self.touch[b_idx[accepted], top_idx[accepted]] += 1
            self.last_step[b_idx[accepted], top_idx[accepted]] = (
                self.global_step.unsqueeze(1).expand_as(top_idx)[accepted])

        return (out, evidence) if returnEvidence else out #[B, dim]

    @torch.no_grad()
    def Inspect(self) -> Dict[str, torch.Tensor]:
        return {
            "keys": self.keys.clone(),
            "vals": self.vals.clone(),
            "priority": self.priority.clone(),
            "ttl": self.ttl.clone(),
            "created_step": self.created_step.clone(),
            "last_step": self.last_step.clone(),
            "last_rehearsal_step": self.last_rehearsal_step.clone(),
            "touch": self.touch.clone(),
            "source": self.source.clone(),
            "source_confidence": self.source_confidence.clone(),
            "global_step": self.global_step.clone(),}



class SemanticLTM(AGICoreModule):
    def __init__(self, dim: int, capacity: int = 16384):
        super().__init__()
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        B0 = 1
        self.register_buffer("keys", torch.zeros(B0, capacity, dim))
        self.register_buffer("vals", torch.zeros(B0, capacity, dim))
        self.register_buffer("prio", torch.zeros(B0, capacity))
        self.register_buffer("touch", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("last_access_step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("last_rehearsal_step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("consolidation_count", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("prototype_count", torch.zeros(B0, capacity))
        self.register_buffer("prototype_variance", torch.zeros(B0, capacity, dim))
        self.register_buffer("source", torch.zeros(B0, capacity, dtype=torch.int8))
        self.register_buffer("source_confidence", torch.zeros(B0, capacity))
        self.register_buffer(
            "realm",
            torch.full(
                (B0, capacity),
                ONTOLOGY_REALM_UNKNOWN,
                dtype=torch.int8))
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
        self.last_access_step = self.last_access_step.new_zeros(B, self.capacity)
        self.last_rehearsal_step = self.last_rehearsal_step.new_zeros(B, self.capacity)
        self.consolidation_count = self.consolidation_count.new_zeros(B, self.capacity)
        self.prototype_count = self.prototype_count.new_zeros(B, self.capacity)
        self.prototype_variance = self.prototype_variance.new_zeros(B, self.capacity, self.dim)
        self.source = self.source.new_zeros(B, self.capacity)
        self.source_confidence = self.source_confidence.new_zeros(B, self.capacity)
        self.realm = self.realm.new_full(
            (B, self.capacity),
            ONTOLOGY_REALM_UNKNOWN)
        self.filled = self.filled.new_zeros(B)
        self.global_step = self.global_step.new_zeros(B)


    @torch.no_grad()
    def StepTick(self):
        self.global_step.add_(1)

    @torch.no_grad()
    def Store(
        self,
        key: torch.Tensor, # [B, D]
        value: torch.Tensor, # [B, D]
        score: torch.Tensor, # [B,]
        source: Optional[torch.Tensor] = None,
        writeMask: Optional[torch.Tensor] = None,
        sourceConfidence: Optional[torch.Tensor] = None,
        consolidated: bool = False,
        *,
        realm: torch.Tensor):

        B = int(key.size(0))
        self.EnsureB(B)

        if source is None:
            source = torch.full((B,), MemoryType.SRC_REAL, device=self.device, dtype=torch.int8)
        if writeMask is None:
            writeMask = torch.ones(B, device=self.device, dtype=torch.bool)
        if sourceConfidence is None:
            sourceConfidence = SourceConfidence(source, dtype=key.dtype)
        if not bool(writeMask.any().item()):
            return
        for b in range(B):
            if not bool(writeMask[b].item()):
                continue
            n = int(self.filled[b].item())
            match = -1
            if n > 0:
                similarity = torch.mv(self.keys[b, :n], key[b])
                similarity = similarity.masked_fill(
                    self.realm[b, :n] != realm[b],
                    -torch.inf)
                best_similarity, best_index = similarity.max(dim=0)
                if float(best_similarity.item()) >= 0.85:
                    match = int(best_index.item())

            if match >= 0:
                count = self.prototype_count[b, match]
                new_count = count + 1.0
                old_value = self.vals[b, match].clone()
                value_delta = value[b] - old_value
                self.keys[b, match] = F.normalize(
                    (count * self.keys[b, match] + key[b]) / new_count,
                    dim=0)
                new_value = old_value + value_delta / new_count
                self.vals[b, match] = new_value
                self.prototype_variance[b, match] = (
                    count * self.prototype_variance[b, match]
                    + value_delta * (value[b] - new_value)
                ) / new_count
                self.prototype_count[b, match] = new_count
                self.prio[b, match] = torch.maximum(self.prio[b, match], score[b])
                self.last_rehearsal_step[b, match] = self.global_step[b]
                self.touch[b, match] += 1
                if bool(consolidated):
                    self.consolidation_count[b, match] += 1
                old_source = self.source[b, match]
                if int(old_source.item()) != int(source[b].item()):
                    self.source[b, match] = MemoryType.SRC_MIXED
                old_conf = self.source_confidence[b, match]
                evidence = sourceConfidence[b]
                self.source_confidence[b, match] = 1.0 - (1.0 - old_conf) * (1.0 - evidence)
                continue

            if n < self.capacity:
                idx = n
                self.filled[b] += 1
            else:
                age = (self.global_step[b] - self.step[b]).clamp(min=0).float()
                retention = (
                    self.prio[b]
                    * torch.exp(-0.001 * age)
                    * torch.log1p(self.touch[b].float() + 1.0)
                    * self.source_confidence[b])
                idx = int(torch.argmin(retention).item())

            self.keys[b, idx] = key[b]
            self.vals[b, idx] = value[b]
            self.prio[b, idx] = score[b]
            self.touch[b, idx] = 1
            self.step[b, idx] = self.global_step[b]
            self.last_access_step[b, idx] = self.global_step[b]
            self.last_rehearsal_step[b, idx] = self.global_step[b]
            self.consolidation_count[b, idx] = int(bool(consolidated))
            self.prototype_count[b, idx] = 1.0
            self.prototype_variance[b, idx] = 0
            self.source[b, idx] = source[b]
            self.source_confidence[b, idx] = sourceConfidence[b]
            self.realm[b, idx] = realm[b]

    def Retrieve(
        self,
        query: torch.Tensor, #[B, D]
        topk: int = 8,
        returnEvidence: bool = False,):
        B = int(query.size(0))
        self.EnsureB(B)

        filled = self.filled
        gstep = self.global_step

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0) #[1, capacity]
        valid_mask = slots < filled.unsqueeze(1) #[B, capacity]
        any_valid = valid_mask.any(dim=1)
        if not bool(any_valid.any().item()):
            out = torch.zeros(B, self.dim, device=self.device, dtype=query.dtype)
            evidence = torch.zeros(B, 1, device=self.device, dtype=query.dtype)
            return (out, evidence) if returnEvidence else out

        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float() #[B, capacity]
        sim = UnifiedMemoryScore(
            query,
            self.keys,
            age=age,
            priority=self.prio,
            touch=self.touch,
            confidence=self.source_confidence,
            source=self.source,
            validMask=valid_mask,
            baseAgeBeta=0.001,
            mixedExtraBeta=0.004,
            imagineExtraBeta=0.012,)
        sim = torch.where(any_valid.view(B, 1), sim, torch.zeros_like(sim))

        K = int(min(max(1, int(filled.max().item())), int(topk), self.capacity))
        top_sim, top_idx = StableTopk(sim, K) #[B, K]
        candidate_valid = torch.gather(valid_mask, 1, top_idx)
        w, accepted, evidence = NullGatedTopKWeights(
            top_sim,
            self.null_logit,
            candidate_valid)

        idx_expanded = top_idx.unsqueeze(-1).expand(B, K, self.dim)
        vecs = torch.gather(self.vals, 1, idx_expanded) #[B, K, D]
        out = torch.einsum("bk,bkd->bd", w, vecs)

        if not any_valid.all():
            out[~any_valid] = 0
            evidence[~any_valid] = 0

        with torch.no_grad():
            b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand_as(top_idx)
            self.touch[b_idx[accepted], top_idx[accepted]] += 1
            self.last_access_step[b_idx[accepted], top_idx[accepted]] = (
                gstep.unsqueeze(1).expand_as(top_idx)[accepted])

        return (out, evidence) if returnEvidence else out #[B, D]


class EpisodicLTM(AGICoreModule):
    def __init__(self, dim: int, capacity: int = 16384):
        super().__init__()
        self.dim = dim
        self.capacity = capacity
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        B0 = 1
        self.register_buffer("keys", torch.zeros(B0, capacity, dim))
        self.register_buffer("state_keys", torch.zeros(B0, capacity, dim))
        self.register_buffer("vals", torch.zeros(B0, capacity, dim))
        self.register_buffer("rew", torch.zeros(B0, capacity))
        self.register_buffer("rew_abs", torch.zeros(B0, capacity))
        self.register_buffer("prio", torch.zeros(B0, capacity))
        self.register_buffer("step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("last_access_step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("last_rehearsal_step", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("consolidation_count", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("touch", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("source", torch.zeros(B0, capacity, dtype=torch.int8))
        self.register_buffer("source_confidence", torch.zeros(B0, capacity))
        self.register_buffer(
            "realm",
            torch.full(
                (B0, capacity),
                ONTOLOGY_REALM_UNKNOWN,
                dtype=torch.int8))
        self.register_buffer(
            "agency_id",
            torch.full(
                (B0, capacity),
                ONTOLOGY_AGENCY_UNKNOWN,
                dtype=torch.int8))
        self.register_buffer(
            "motion_layer_prob",
            torch.zeros(B0, capacity, 5))
        self.register_buffer(
            "observed_surface_parent_index",
            torch.full((B0, capacity), -1, dtype=torch.long))
        self.register_buffer(
            "surface_uv",
            torch.zeros(B0, capacity, 2))
        self.register_buffer(
            "surface_uv_confidence",
            torch.zeros(B0, capacity))
        self.register_buffer(
            "verification",
            torch.zeros(B0, capacity))
        self.register_buffer(
            "content_change",
            torch.zeros(B0, capacity))
        self.register_buffer("episode_id", torch.full((B0, capacity), -1, dtype=torch.long))
        self.register_buffer("event_id", torch.full((B0, capacity), -1, dtype=torch.long))
        self.register_buffer("prev_index", torch.full((B0, capacity), -1, dtype=torch.long))
        self.register_buffer("next_index", torch.full((B0, capacity), -1, dtype=torch.long))
        self.register_buffer("slot_generation", torch.zeros(B0, capacity, dtype=torch.long))
        self.register_buffer("prev_generation", torch.full((B0, capacity), -1, dtype=torch.long))
        self.register_buffer("next_generation", torch.full((B0, capacity), -1, dtype=torch.long))
        self.register_buffer("last_event_index", torch.full((B0,), -1, dtype=torch.long))
        self.register_buffer("current_episode_id", torch.zeros(B0, dtype=torch.long))
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
        self.last_access_step = self.last_access_step.new_zeros(B, self.capacity)
        self.last_rehearsal_step = self.last_rehearsal_step.new_zeros(B, self.capacity)
        self.consolidation_count = self.consolidation_count.new_zeros(B, self.capacity)
        self.touch = self.touch.new_zeros(B, self.capacity)
        self.source = self.source.new_zeros(B, self.capacity)
        self.source_confidence = self.source_confidence.new_zeros(B, self.capacity)
        self.realm = self.realm.new_full(
            (B, self.capacity),
            ONTOLOGY_REALM_UNKNOWN)
        self.agency_id = self.agency_id.new_full(
            (B, self.capacity),
            ONTOLOGY_AGENCY_UNKNOWN)
        self.motion_layer_prob = self.motion_layer_prob.new_zeros(
            B, self.capacity, 5)
        self.observed_surface_parent_index = (
            self.observed_surface_parent_index.new_full(
                (B, self.capacity), -1))
        self.surface_uv = self.surface_uv.new_zeros(B, self.capacity, 2)
        self.surface_uv_confidence = (
            self.surface_uv_confidence.new_zeros(B, self.capacity))
        self.verification = self.verification.new_zeros(B, self.capacity)
        self.content_change = self.content_change.new_zeros(B, self.capacity)
        self.episode_id = self.episode_id.new_full((B, self.capacity), -1)
        self.event_id = self.event_id.new_full((B, self.capacity), -1)
        self.prev_index = self.prev_index.new_full((B, self.capacity), -1)
        self.next_index = self.next_index.new_full((B, self.capacity), -1)
        self.slot_generation = self.slot_generation.new_zeros(B, self.capacity)
        self.prev_generation = self.prev_generation.new_full((B, self.capacity), -1)
        self.next_generation = self.next_generation.new_full((B, self.capacity), -1)
        self.last_event_index = self.last_event_index.new_full((B,), -1)
        self.current_episode_id = self.current_episode_id.new_zeros(B)
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
        stateKey: Optional[torch.Tensor] = None,
        sourceConfidence: Optional[torch.Tensor] = None,
        episodeId: Optional[torch.Tensor] = None,
        eventId: Optional[torch.Tensor] = None,
        *,
        realm: torch.Tensor,
        typedMetadata: Dict[str, torch.Tensor]):
        B = int(key.size(0))
        self.EnsureB(B)

        if source is None:
            source = torch.full((B,), MemoryType.SRC_REAL, device=self.device, dtype=torch.int8)
        if writeMask is None:
            writeMask = torch.ones(B, device=self.device, dtype=torch.bool)
        if stateKey is None:
            stateKey = key
        if sourceConfidence is None:
            sourceConfidence = SourceConfidence(source, dtype=key.dtype)
        if episodeId is None:
            episodeId = self.current_episode_id
        if eventId is None:
            eventId = self.global_step
        if not bool(writeMask.any().item()):
            return
        for b in range(B):
            if not bool(writeMask[b].item()):
                continue
            n = int(self.filled[b].item())
            if n < self.capacity:
                idx = n
                self.filled[b] += 1
            else:
                age = (self.global_step[b] - self.step[b]).clamp(min=0).float()
                retention = (
                    (self.prio[b] + 0.5 * self.rew_abs[b])
                    * torch.exp(-0.01 * age)
                    * torch.log1p(self.touch[b].float() + 1.0)
                    * self.source_confidence[b])
                idx = int(torch.argmin(retention).item())

                old_prev = int(self.prev_index[b, idx].item())
                old_next = int(self.next_index[b, idx].item())
                if old_prev >= 0 and int(self.next_index[b, old_prev].item()) == idx:
                    self.next_index[b, old_prev] = -1
                    self.next_generation[b, old_prev] = -1
                if old_next >= 0 and int(self.prev_index[b, old_next].item()) == idx:
                    self.prev_index[b, old_next] = -1
                    self.prev_generation[b, old_next] = -1

            self.slot_generation[b, idx] += 1
            generation = self.slot_generation[b, idx]
            previous = int(self.last_event_index[b].item())
            valid_previous = (
                previous >= 0
                and previous != idx
                and int(self.episode_id[b, previous].item()) == int(episodeId[b].item()))

            self.keys[b, idx] = key[b]
            self.state_keys[b, idx] = stateKey[b]
            self.vals[b, idx] = value[b]
            self.rew[b, idx] = reward[b]
            self.rew_abs[b, idx] = reward[b].abs()
            self.prio[b, idx] = score[b]
            self.step[b, idx] = self.global_step[b]
            self.last_access_step[b, idx] = self.global_step[b]
            self.last_rehearsal_step[b, idx] = self.global_step[b]
            self.consolidation_count[b, idx] = 0
            self.touch[b, idx] = 1
            self.source[b, idx] = source[b]
            self.source_confidence[b, idx] = sourceConfidence[b]
            self.realm[b, idx] = realm[b]
            self.agency_id[b, idx] = typedMetadata["agency_id"][b]
            self.motion_layer_prob[b, idx] = typedMetadata[
                "motion_layer_prob"][b]
            self.observed_surface_parent_index[b, idx] = typedMetadata[
                "observed_surface_parent_index"][b]
            self.surface_uv[b, idx] = typedMetadata["surface_uv"][b]
            self.surface_uv_confidence[b, idx] = typedMetadata[
                "surface_uv_confidence"][b]
            self.verification[b, idx] = typedMetadata["verification"][b]
            self.content_change[b, idx] = typedMetadata[
                "content_change"][b]
            self.episode_id[b, idx] = episodeId[b]
            self.event_id[b, idx] = eventId[b]
            self.prev_index[b, idx] = previous if valid_previous else -1
            self.prev_generation[b, idx] = (
                self.slot_generation[b, previous] if valid_previous else -1)
            self.next_index[b, idx] = -1
            self.next_generation[b, idx] = -1
            if valid_previous:
                self.next_index[b, previous] = idx
                self.next_generation[b, previous] = generation
            self.last_event_index[b] = idx

    def Retrieve(
        self,
        query: torch.Tensor,
        topk: int = 8,
        recentBias: float = 0.05,
        useStateKey: bool = False,
        recordAccess: bool = True,
        returnEvidence: bool = False,):
        B = int(query.size(0))
        self.EnsureB(B)

        filled = self.filled
        gstep = self.global_step

        slots = torch.arange(self.capacity, device=self.device).unsqueeze(0)
        valid_mask = slots < filled.unsqueeze(1)
        any_valid = valid_mask.any(dim=1)
        if not bool(any_valid.any().item()):
            out = torch.zeros(B, self.dim, device=self.device, dtype=query.dtype)
            evidence = torch.zeros(B, 1, device=self.device, dtype=query.dtype)
            return (out, evidence) if returnEvidence else out

        keys = self.state_keys if bool(useStateKey) else self.keys
        age = (gstep.unsqueeze(1) - self.step).clamp(min=0).float()
        sim = UnifiedMemoryScore(
            query,
            keys,
            age=age,
            priority=self.prio,
            touch=self.touch,
            confidence=self.source_confidence,
            source=self.source,
            rewardAbs=self.rew_abs,
            validMask=valid_mask,
            baseAgeBeta=float(recentBias),
            mixedExtraBeta=0.02,
            imagineExtraBeta=0.05,)
        sim = torch.where(any_valid.view(B, 1), sim, torch.zeros_like(sim))

        K = int(min(max(1, int(filled.max().item())), int(topk), self.capacity))
        top_sim, idx = StableTopk(sim, K)
        candidate_valid = torch.gather(valid_mask, 1, idx)
        w, accepted, evidence = NullGatedTopKWeights(
            top_sim,
            self.null_logit,
            candidate_valid)

        idx_expanded = idx.unsqueeze(-1).expand(B, K, self.dim)
        vecs = torch.gather(self.vals, 1, idx_expanded)
        out = torch.einsum("bk,bkd->bd", w, vecs)

        if not any_valid.all():
            out[~any_valid] = 0.0
            evidence[~any_valid] = 0.0

        if recordAccess:
            with torch.no_grad():
                b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand_as(idx)
                self.touch[b_idx[accepted], idx[accepted]] += 1
                self.last_access_step[b_idx[accepted], idx[accepted]] = (
                    gstep.unsqueeze(1).expand_as(idx)[accepted])

        return (out, evidence) if returnEvidence else out #[B, D]

    def RetrieveSequence(self, seedIndex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(seedIndex.size(0))
        sequence = self.vals.new_zeros(B, 3, self.dim)
        valid = torch.zeros(B, 3, device=seedIndex.device, dtype=torch.bool)
        for b in range(B):
            seed = int(seedIndex[b].item())
            if seed < 0 or seed >= int(self.filled[b].item()):
                continue
            sequence[b, 1] = self.vals[b, seed]
            valid[b, 1] = True
            previous = int(self.prev_index[b, seed].item())
            if (
                previous >= 0
                and int(self.slot_generation[b, previous].item())
                == int(self.prev_generation[b, seed].item())
                and int(self.episode_id[b, previous].item())
                == int(self.episode_id[b, seed].item())
            ):
                sequence[b, 0] = self.vals[b, previous]
                valid[b, 0] = True
            following = int(self.next_index[b, seed].item())
            if (
                following >= 0
                and int(self.slot_generation[b, following].item())
                == int(self.next_generation[b, seed].item())
                and int(self.episode_id[b, following].item())
                == int(self.episode_id[b, seed].item())
            ):
                sequence[b, 2] = self.vals[b, following]
                valid[b, 2] = True
        return sequence, valid

    @torch.no_grad()
    def RebuildSequenceLinks(self) -> None:

        self.prev_index.fill_(-1)
        self.next_index.fill_(-1)
        self.prev_generation.fill_(-1)
        self.next_generation.fill_(-1)
        self.last_event_index.fill_(-1)
        for b in range(int(self.filled.size(0))):
            n = int(self.filled[b].item())
            if n == 0:
                continue
            self.slot_generation[b, :n].clamp_min_(1)
            episodes = torch.unique(self.episode_id[b, :n])
            for episode in episodes.tolist():
                for imagined_branch in (False, True):
                    candidates = [
                        idx
                        for idx in range(n)
                        if int(self.episode_id[b, idx].item()) == int(episode)
                        and bool(int(self.source[b, idx].item()) == MemoryType.SRC_IMAGINE)
                        == imagined_branch]
                    candidates.sort(key=lambda idx: (
                        int(self.event_id[b, idx].item()),
                        int(self.step[b, idx].item()),
                        idx))
                    for previous, following in zip(candidates, candidates[1:]):
                        self.next_index[b, previous] = following
                        self.next_generation[b, previous] = self.slot_generation[b, following]
                        self.prev_index[b, following] = previous
                        self.prev_generation[b, following] = self.slot_generation[b, previous]

            current_episode = int(self.current_episode_id[b].item())
            current = [
                idx
                for idx in range(n)
                if int(self.episode_id[b, idx].item()) == current_episode
                and int(self.source[b, idx].item()) != MemoryType.SRC_IMAGINE]
            if not current:
                current = [
                    idx
                    for idx in range(n)
                    if int(self.episode_id[b, idx].item()) == current_episode]
            if current:
                self.last_event_index[b] = max(
                    current,
                    key=lambda idx: (
                        int(self.event_id[b, idx].item()),
                        int(self.step[b, idx].item()),
                        idx))

    def RetrieveSeedIndex(self, query: torch.Tensor, useStateKey: bool = False) -> torch.Tensor:
        B = int(query.size(0))
        slots = torch.arange(self.capacity, device=query.device).view(1, self.capacity)
        valid = slots < self.filled.view(B, 1)
        age = (self.global_step.view(B, 1) - self.step).clamp(min=0).float()
        score = UnifiedMemoryScore(
            query,
            self.state_keys if useStateKey else self.keys,
            age=age,
            priority=self.prio,
            touch=self.touch,
            confidence=self.source_confidence,
            source=self.source,
            rewardAbs=self.rew_abs,
            validMask=valid,
            baseAgeBeta=0.05,
            mixedExtraBeta=0.02,
            imagineExtraBeta=0.05,)
        best_score, seed = score.max(dim=1)
        has_evidence = valid.any(dim=1) & (best_score > self.null_logit)
        return torch.where(has_evidence, seed, torch.full_like(seed, -1))

    @torch.no_grad()
    def StartNewEpisode(self, done: torch.Tensor) -> torch.Tensor:
        slots = torch.arange(
            self.capacity,
            device=self.episode_id.device).unsqueeze(0)
        valid = slots < self.filled.unsqueeze(1)
        largest_stored = self.episode_id.masked_fill(~valid, -1).max(dim=1).values
        next_episode = torch.maximum(
            self.current_episode_id + 1,
            largest_stored + 1)
        self.current_episode_id[done] = next_episode[done]
        self.last_event_index[done] = -1
        return self.current_episode_id

    @torch.no_grad()
    def VerifyWithRealEvidence(
        self,
        key: torch.Tensor,
        realMask: torch.Tensor,
        realm: torch.Tensor,):
        B = int(key.size(0))
        source_branch_changed = False
        for b in range(B):
            if not bool(realMask[b].item()):
                continue
            n = int(self.filled[b].item())
            if n == 0:
                continue
            hypothesis = (
                (self.source[b, :n] == MemoryType.SRC_IMAGINE)
                | (self.source[b, :n] == MemoryType.SRC_MIXED))
            hypothesis = hypothesis & (self.realm[b, :n] == realm[b])
            if not bool(hypothesis.any().item()):
                continue
            similarity = torch.mv(self.keys[b, :n], key[b]).masked_fill(
                ~hypothesis,
                -torch.inf)
            best_similarity, best_index = similarity.max(dim=0)
            if float(best_similarity.item()) < 0.90:
                continue
            idx = int(best_index.item())
            was_imagined = int(self.source[b, idx].item()) == MemoryType.SRC_IMAGINE
            posterior = self.source_confidence[b, idx]
            self.source_confidence[b, idx] = posterior + 0.50 * (1.0 - posterior)
            self.source[b, idx] = MemoryType.SRC_MIXED
            self.last_rehearsal_step[b, idx] = self.global_step[b]
            source_branch_changed = source_branch_changed or was_imagined
        if source_branch_changed:
            self.RebuildSequenceLinks()

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


class EpisodicSequenceEncoder(AGICoreModule):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.position = nn.Parameter(torch.randn(3, self.dim) * 0.02)
        self.input_norm = nn.LayerNorm(self.dim)
        self.sequence = nn.GRU(self.dim, self.dim, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, 1),)
        self.output = nn.Sequential(
            nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.SiLU(),
            nn.Linear(self.dim * 2, self.dim),)

    def forward(self, sequence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        B, S, D = sequence.shape
        positioned = self.input_norm(sequence) + self.position.unsqueeze(0)
        compact = sequence.new_zeros(B, S, D)
        compact_valid = torch.zeros(B, S, device=sequence.device, dtype=torch.bool)
        rank = valid.long().cumsum(dim=1) - 1
        for source_index in range(S):
            source_valid = valid[:, source_index]
            batch_index = torch.arange(B, device=sequence.device)[source_valid]
            target_index = rank[source_valid, source_index]
            compact[batch_index, target_index] = positioned[source_valid, source_index]
            compact_valid[batch_index, target_index] = True
        encoded, _ = self.sequence(compact)
        logits = self.readout(encoded).squeeze(-1).masked_fill(~compact_valid, -torch.inf)
        any_valid = compact_valid.any(dim=1)
        logits = torch.where(any_valid.unsqueeze(1), logits, torch.zeros_like(logits))
        weights = F.softmax(logits, dim=1)
        weights = torch.where(any_valid.unsqueeze(1), weights, torch.zeros_like(weights))
        context = torch.einsum("bs,bsd->bd", weights, encoded)
        seed = sequence[:, 1]
        out = self.output(torch.cat([seed, context], dim=-1))
        return torch.where(any_valid.unsqueeze(1), out, torch.zeros_like(out))


class LongTermMemory(AGICoreModule):
    def __init__(self, dim: int, semCap: int = 16384, epiCap: int = 16384):
        super().__init__()
        self.semantic = SemanticLTM(dim, semCap)
        self.episodic = EpisodicLTM(dim, epiCap)

        self.fuser = LTMFuser(dim)
        self.sequence_encoder = EpisodicSequenceEncoder(dim)

    @torch.no_grad()
    def Reset(self):
        self.semantic.keys.zero_()
        self.semantic.vals.zero_()
        self.semantic.prio.zero_()
        self.semantic.touch.zero_()
        self.semantic.step.zero_()
        self.semantic.last_access_step.zero_()
        self.semantic.last_rehearsal_step.zero_()
        self.semantic.consolidation_count.zero_()
        self.semantic.prototype_count.zero_()
        self.semantic.prototype_variance.zero_()
        self.semantic.source.zero_()
        self.semantic.source_confidence.zero_()
        self.semantic.realm.fill_(ONTOLOGY_REALM_UNKNOWN)
        self.semantic.filled.zero_()
        self.semantic.global_step.zero_()

        self.episodic.keys.zero_()
        self.episodic.state_keys.zero_()
        self.episodic.vals.zero_()
        self.episodic.rew.zero_()
        self.episodic.rew_abs.zero_()
        self.episodic.prio.zero_()
        self.episodic.step.zero_()
        self.episodic.last_access_step.zero_()
        self.episodic.last_rehearsal_step.zero_()
        self.episodic.consolidation_count.zero_()
        self.episodic.touch.zero_()
        self.episodic.source.zero_()
        self.episodic.source_confidence.zero_()
        self.episodic.realm.fill_(ONTOLOGY_REALM_UNKNOWN)
        self.episodic.agency_id.fill_(ONTOLOGY_AGENCY_UNKNOWN)
        self.episodic.motion_layer_prob.zero_()
        self.episodic.observed_surface_parent_index.fill_(-1)
        self.episodic.surface_uv.zero_()
        self.episodic.surface_uv_confidence.zero_()
        self.episodic.verification.zero_()
        self.episodic.content_change.zero_()
        self.episodic.episode_id.fill_(-1)
        self.episodic.event_id.fill_(-1)
        self.episodic.prev_index.fill_(-1)
        self.episodic.next_index.fill_(-1)
        self.episodic.slot_generation.zero_()
        self.episodic.prev_generation.fill_(-1)
        self.episodic.next_generation.fill_(-1)
        self.episodic.last_event_index.fill_(-1)
        self.episodic.current_episode_id.zero_()
        self.episodic.filled.zero_()
        self.episodic.global_step.zero_()

    @torch.no_grad()
    def StepTick(self):
        self.semantic.StepTick()
        self.episodic.StepTick()

    def Retrieve(
        self,
        query: torch.Tensor,
        topkSem: int = 6,
        topkEpi: int = 4,
        epiQuery: Optional[torch.Tensor] = None,
        returnEvidence: bool = False,):
        sem_out, sem_evidence = self.semantic.Retrieve(
            query,
            topk=topkSem,
            returnEvidence=True) #[B, D]
        epi_out, epi_evidence = self.episodic.Retrieve(
            query if epiQuery is None else epiQuery,
            topk=topkEpi,
            returnEvidence=True) #[B, D]

        if returnEvidence:
            return sem_out, epi_out, sem_evidence, epi_evidence
        return sem_out, epi_out

    def RetrieveEpisodeSequence(
        self,
        query: torch.Tensor,
        returnEvidence: bool = False,):
        seed = self.episodic.RetrieveSeedIndex(query)
        sequence, valid = self.episodic.RetrieveSequence(seed)
        out = self.sequence_encoder(sequence, valid)
        evidence = valid.any(dim=1, keepdim=True).to(query.dtype)
        return (out, evidence) if returnEvidence else out


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
        x: torch.Tensor, #[B, inDim]
        sampleMask: Optional[torch.Tensor] = None,
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

            logits = logits.masked_fill(
                ~keep,
                torch.finfo(logits.dtype).min)

        t = max(self.temperature, 1e-6)
        a = F.softmax((logits / t).float(), dim=-1) # [B, numExperts]

        if self.training:
            if sampleMask is None:
                sample_mask = torch.ones(
                    x.size(0), device=x.device, dtype=torch.bool)
            else:
                if (
                    not torch.is_tensor(sampleMask)
                    or tuple(sampleMask.shape) != (x.size(0),)
                    or sampleMask.dtype != torch.bool
                    or sampleMask.device != x.device
                ):
                    raise ValueError("fusion sample mask must be a boolean batch vector")
                sample_mask = sampleMask
            if bool(sample_mask.any().item()):
                importance = a[sample_mask].mean(dim=0)
                self.aux_loss = float(self.numExperts) * (
                    importance.pow(2).sum())
            else:
                self.aux_loss = x.new_zeros(())
        else:
            self.aux_loss = x.new_zeros(())

        ys = [expert(x) for expert in self.experts]
        y = torch.stack(ys, dim=-1) # [B, outDim, numExperts]
        out = (y * a.unsqueeze(1)).sum(dim=-1)
        return out # [B, outDim]


class ObjectUsageBank(AGICoreModule):









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
        self.slot_delta_dim = int(slotDeltaDim)

        self.register_buffer("applicable", torch.zeros(self.num_objects, self.num_skills))
        self.register_buffer("default_params", torch.zeros(self.num_objects, self.num_skills, self.param_dim))
        self.register_buffer("expected_dx", torch.zeros(
            self.num_objects,
            self.num_skills,
            self.slot_delta_dim))
        self.register_buffer("success_alpha", torch.ones(self.num_objects, self.num_skills))
        self.register_buffer("success_beta", torch.ones(self.num_objects, self.num_skills))
        self.register_buffer("param_mu", torch.zeros(self.num_objects, self.num_skills, self.param_dim))
        self.register_buffer("param_logvar", torch.zeros(self.num_objects, self.num_skills, self.param_dim))
        self.register_buffer("parameter_observations", torch.zeros(self.num_objects, self.num_skills))
        self.register_buffer("instance_descriptors", F.normalize(torch.randn(self.num_objects, int(idDim)), dim=-1))


        self.register_buffer("attribute_centroid", torch.zeros(self.num_objects, self.num_skills, self.attr_dim))

        self.readout_proj = nn.Linear(
            self.param_dim + self.attr_dim + self.slot_delta_dim + 4,
            self.usage_dim)
        self.needs_lookup = nn.Linear(4, 1)
        self.unknown_similarity_logit = nn.Parameter(torch.tensor(0.84729786))

    def SuccessRate(self) -> torch.Tensor:
        return self.success_alpha / (self.success_alpha + self.success_beta)

    def Confidence(self) -> torch.Tensor:
        total = self.success_alpha + self.success_beta
        return self.applicable * total / (total + 10.0)

    def NearestObject(self, descriptor: torch.Tensor) -> torch.Tensor:

        similarity, index = self.NearestObjectMatch(descriptor)
        threshold = torch.sigmoid(self.unknown_similarity_logit)
        return torch.where(similarity >= threshold, index, torch.full_like(index, -1))

    def NearestObjectMatch(self, descriptor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        similarity = torch.matmul(
            F.normalize(descriptor, dim=-1),
            self.instance_descriptors.t())
        return similarity.max(dim=-1)

    def BestObjectsForSkill(self, skillId: int) -> torch.Tensor:
        return self.applicable[:, int(skillId)].argmax()

    def SlotReadout(
        self,
        identity: torch.Tensor,
        attribute: Optional[torch.Tensor],
        slotAttention: torch.Tensor,) -> torch.Tensor:


        B, K, _ = identity.shape
        match_similarity, obj_idx = self.NearestObjectMatch(identity) # [B,K]
        known_probability = torch.sigmoid(
            12.0 * (match_similarity - torch.sigmoid(self.unknown_similarity_logit)))
        applicable_rows = self.applicable[obj_idx] # [B,K,N_skills]
        best_skill = applicable_rows.argmax(dim=-1) # [B,K]
        gather = lambda t: t[obj_idx.reshape(-1), best_skill.reshape(-1)].view(B, K, -1)
        default_params = gather(self.default_params) # [B,K,P]
        posterior_params = gather(self.param_mu)
        observation_count = self.parameter_observations[obj_idx, best_skill].unsqueeze(-1)
        posterior_weight = observation_count / (observation_count + 10.0)
        params = (1.0 - posterior_weight) * default_params + posterior_weight * posterior_params
        expected_delta = gather(self.expected_dx) # [B,K,D_slot]
        attribute_centroid = gather(self.attribute_centroid) # [B,K,A]
        attribute_residual = (
            torch.zeros_like(attribute_centroid)
            if attribute is None
            else attribute - attribute_centroid)
        success = self.SuccessRate()[obj_idx, best_skill].unsqueeze(-1)
        confidence = self.Confidence()[obj_idx, best_skill].unsqueeze(-1)
        best_applicable = applicable_rows.gather(-1, best_skill.unsqueeze(-1))
        lookup_need = self.NeedsLookupScore(
            best_applicable.squeeze(-1) * known_probability,
            confidence.squeeze(-1),
            success.squeeze(-1),
            slotAttention)
        lookup_need = 1.0 - known_probability * (1.0 - lookup_need)
        lookup_need = lookup_need.unsqueeze(-1)
        known_payload = torch.cat([
            params,
            attribute_residual,
            expected_delta,
            success,
            confidence,
            best_applicable,], dim=-1) * known_probability.unsqueeze(-1)
        summary = torch.cat([
            known_payload,
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
        self.parameter_observations[obj, skill] += 1
        if observedAttributes is not None:
            centroid = self.attribute_centroid[obj, skill]
            self.attribute_centroid[obj, skill] = (
                momentum * centroid + (1.0 - momentum) * observedAttributes.view(centroid.shape))


class MemoryExtractor(AGICoreModule):
    DURABLE_MEMORY_ARTIFACT_TYPE = "MemoryExtractorDurableMemory"
    DURABLE_MEMORY_SCHEMA_VERSION = 6
    COUNTERFACTUAL_PREDICTED_REWARD = 0
    COUNTERFACTUAL_CORRECTED_REWARD = 1
    COUNTERFACTUAL_PREDICTED_DONE = 2
    COUNTERFACTUAL_CORRECTED_DONE = 3
    COUNTERFACTUAL_REWARD_VARIANCE = 4
    COUNTERFACTUAL_DONE_VARIANCE = 5
    COUNTERFACTUAL_INFORMATION = 6
    COUNTERFACTUAL_SOURCE = 7
    COUNTERFACTUAL_OUTCOME_DIM = 8
    DURABLE_MEMORY_STATE_FIELDS = (
        "time_step",
        "memory_filled",
        "memory_version",
        "merged_delta_signature",
        "memory_keys",
        "memory_values",
        "memory_importance",
        "memory_steps",
        "memory_last_access_steps",
        "memory_last_rehearsal_steps",
        "memory_touch",
        "memory_merge_count",
        "memory_emotion",
        "memory_source",
        "memory_source_confidence",
        "memory_realm",
        "memory_reward_abs",
        "episode_id",
        "event_id",
        "ltm_sem_global_step",
        "ltm_sem_keys",
        "ltm_sem_vals",
        "ltm_sem_prio",
        "ltm_sem_touch",
        "ltm_sem_step",
        "ltm_sem_last_access_step",
        "ltm_sem_last_rehearsal_step",
        "ltm_sem_consolidation_count",
        "ltm_sem_prototype_count",
        "ltm_sem_prototype_variance",
        "ltm_sem_filled",
        "ltm_sem_source",
        "ltm_sem_source_confidence",
        "ltm_sem_realm",
        "ltm_epi_global_step",
        "ltm_epi_keys",
        "ltm_epi_state_keys",
        "ltm_epi_vals",
        "ltm_epi_prio",
        "ltm_epi_rew",
        "ltm_epi_rew_abs",
        "ltm_epi_step",
        "ltm_epi_last_access_step",
        "ltm_epi_last_rehearsal_step",
        "ltm_epi_consolidation_count",
        "ltm_epi_touch",
        "ltm_epi_filled",
        "ltm_epi_source",
        "ltm_epi_source_confidence",
        "ltm_epi_realm",
        "ltm_epi_agency_id",
        "ltm_epi_motion_layer_prob",
        "ltm_epi_observed_surface_parent_index",
        "ltm_epi_surface_uv",
        "ltm_epi_surface_uv_confidence",
        "ltm_epi_verification",
        "ltm_epi_content_change",
        "ltm_epi_episode_id",
        "ltm_epi_event_id",
        "ltm_epi_prev_index",
        "ltm_epi_next_index",
        "ltm_epi_slot_generation",
        "ltm_epi_prev_generation",
        "ltm_epi_next_generation",
        "ltm_epi_last_event_index",
        "ltm_epi_current_episode_id",
        "sym_mem_global_step",
        "sym_mem_P_keys",
        "sym_mem_P_vals",
        "sym_mem_prio",
        "sym_mem_step",
        "sym_mem_last_access_step",
        "sym_mem_last_rehearsal_step",
        "sym_mem_touch",
        "sym_mem_filled",
        "sym_mem_source",
        "sym_mem_source_confidence",
        "usage_applicable",
        "usage_default_params",
        "usage_expected_dx",
        "usage_success_alpha",
        "usage_success_beta",
        "usage_param_mu",
        "usage_param_logvar",
        "usage_parameter_observations",
        "usage_instance_descriptors",
        "usage_attribute_centroid",)
    TRANSIENT_MEMORY_STATE_FIELDS = (
        "last_compress_step",
        "h_state",
        "fast_weights",
        "ns_prev_P_post",
        "ns_penalty_vec",
        "pattern_usage",
        "previous_attention",
        "previous_intent",
        "previous_object_summary",
        "previous_motion_token",
        "previous_ontology_context",
        "event_age",
        "has_previous_event",
        "gws_global_step",
        "gws_keys",
        "gws_vals",
        "gws_priority",
        "gws_ttl",
        "gws_created_step",
        "gws_last_step",
        "gws_last_rehearsal_step",
        "gws_touch",
        "gws_source",
        "gws_source_confidence",)
    FULL_MEMORY_STATE_FIELDS = DURABLE_MEMORY_STATE_FIELDS + TRANSIENT_MEMORY_STATE_FIELDS
    UNBATCHED_MEMORY_STATE_FIELDS = (
        "memory_version",
        "merged_delta_signature",
        "pattern_usage",
        "usage_applicable",
        "usage_default_params",
        "usage_expected_dx",
        "usage_success_alpha",
        "usage_success_beta",
        "usage_param_mu",
        "usage_param_logvar",
        "usage_parameter_observations",
        "usage_instance_descriptors",
        "usage_attribute_centroid",)
    FORWARD_MUTABLE_ROW_STATE_FIELDS = (
        "time_step",
        "memory_last_access_steps",
        "memory_touch",
        "episode_id",
        "event_id",
        "ltm_sem_global_step",
        "ltm_sem_touch",
        "ltm_sem_last_access_step",
        "ltm_epi_global_step",
        "ltm_epi_last_access_step",
        "ltm_epi_touch",
        "sym_mem_global_step",
        "sym_mem_last_access_step",
        "sym_mem_touch",
        "h_state",
        "fast_weights",
        "ns_prev_P_post",
        "ns_penalty_vec",
        "previous_attention",
        "previous_intent",
        "previous_object_summary",
        "previous_motion_token",
        "previous_ontology_context",
        "event_age",
        "has_previous_event",
        "gws_global_step",
        "gws_keys",
        "gws_vals",
        "gws_priority",
        "gws_ttl",
        "gws_created_step",
        "gws_last_step",
        "gws_last_rehearsal_step",
        "gws_touch",
        "gws_source",
        "gws_source_confidence",)

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
        emotionDim: int = 512,
        replayCapacity: int = 1024,) -> None:
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
        retention = torch.logspace(
            math.log10(0.80),
            math.log10(0.995),
            ssmStateDim)
        self.state_retention_logit = nn.Parameter(torch.logit(retention))
        self.state_coupling_logit = nn.Parameter(torch.tensor(0.0))
        self.state_candidate_norm = nn.LayerNorm(ssmStateDim)
        self.state_input_gate = nn.Sequential(
            nn.Linear(inputDim + ssmStateDim, ssmStateDim * 2),
            nn.SiLU(),
            nn.Linear(ssmStateDim * 2, ssmStateDim),)
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
        self.register_buffer("memory_last_access_steps", torch.zeros(B0, memorySize, dtype=torch.long))
        self.register_buffer("memory_last_rehearsal_steps", torch.zeros(B0, memorySize, dtype=torch.long))
        self.register_buffer("memory_touch", torch.zeros(B0, memorySize, dtype=torch.long))
        self.register_buffer("memory_merge_count", torch.zeros(B0, memorySize, dtype=torch.long))
        self.register_buffer("memory_source", torch.zeros(B0, memorySize, dtype=torch.int8))
        self.register_buffer("memory_source_confidence", torch.zeros(B0, memorySize))
        self.register_buffer(
            "memory_realm",
            torch.full(
                (B0, memorySize),
                ONTOLOGY_REALM_UNKNOWN,
                dtype=torch.int8))
        self.register_buffer("memory_reward_abs", torch.zeros(B0, memorySize))
        self.register_buffer("memory_version", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "merged_delta_signature",
            torch.full((3,), -1, dtype=torch.long))

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
        self.emotion_content_scale = nn.Parameter(torch.tensor(-2.944439))

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
        self.kv_null_logit = nn.Parameter(torch.tensor(0.0))

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

        def MakeFilm(D: int) -> nn.Module:
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

        self.film_mem = MakeFilm(self.memory_dim)
        self.film_gws = MakeFilm(self.memory_dim)
        self.film_sem = MakeFilm(self.memory_dim)
        self.film_epi = MakeFilm(self.memory_dim)

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
        self.context_fuse_scale = nn.Parameter(torch.tensor(-2.944439))
        self.context_modulation_gate = nn.Sequential(
            nn.LayerNorm(inputDim * 2),
            nn.Linear(inputDim * 2, inputDim * 2),
            nn.SiLU(),
            nn.Linear(inputDim * 2, inputDim),)

        event_in_dim = inputDim + 2048 + 512 + 512 + self.emotion_dim + 4
        self.event_context_proj = nn.Sequential(
            nn.LayerNorm(event_in_dim),
            nn.Linear(event_in_dim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.pattern_separation_proj = nn.Sequential(
            nn.LayerNorm(memoryDim),
            nn.Linear(memoryDim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.register_buffer("pattern_usage", torch.zeros(memoryDim))
        self.event_boundary_net = nn.Sequential(
            nn.Linear(8, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),)
        nn.init.zeros_(self.event_boundary_net[-1].weight)
        nn.init.constant_(self.event_boundary_net[-1].bias, -2.0)
        self.ontology_feature_dim = (
            69 + ModuleDim.PstSelfPartSemanticDim)
        self.ontology_context_encoder = nn.Sequential(
            nn.LayerNorm(self.ontology_feature_dim),
            nn.Linear(self.ontology_feature_dim, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),
            nn.LayerNorm(memoryDim),)
        self.event_ontology_residual = nn.Sequential(
            nn.LayerNorm(memoryDim * 2),
            nn.Linear(memoryDim * 2, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),)
        self.event_ontology_gain = nn.Parameter(torch.tensor(-2.944439))
        self.entity_text_feature_dim = 515
        self.entity_text_object_encoder = nn.Sequential(
            nn.LayerNorm(self.entity_text_feature_dim),
            nn.Linear(self.entity_text_feature_dim, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),
            nn.LayerNorm(memoryDim),)
        self.entity_text_event_encoder = nn.Sequential(
            nn.LayerNorm(self.entity_text_feature_dim),
            nn.Linear(self.entity_text_feature_dim, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),
            nn.LayerNorm(memoryDim),)
        self.event_text_residual = nn.Sequential(
            nn.LayerNorm(memoryDim * 2),
            nn.Linear(memoryDim * 2, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),)
        self.entity_text_object_gain = nn.Parameter(torch.tensor(-2.944439))
        self.entity_text_event_gain = nn.Parameter(torch.tensor(-2.944439))
        nn.init.zeros_(self.event_text_residual[-1].weight)
        nn.init.zeros_(self.event_text_residual[-1].bias)
        self.ontology_boundary_net = nn.Sequential(
            nn.Linear(6, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),)
        nn.init.zeros_(self.ontology_boundary_net[-1].weight)
        nn.init.constant_(self.ontology_boundary_net[-1].bias, -2.0)
        self.register_buffer("previous_attention", torch.zeros(B0, inputDim))
        self.register_buffer("previous_intent", torch.zeros(B0, 512))
        self.register_buffer("previous_object_summary", torch.zeros(B0, 512))
        self.register_buffer("previous_motion_token", torch.zeros(B0, 512))
        self.register_buffer(
            "previous_ontology_context",
            torch.zeros(B0, self.ontology_feature_dim))
        self.register_buffer("event_age", torch.zeros(B0, dtype=torch.long))
        self.register_buffer("event_id", torch.zeros(B0, dtype=torch.long))
        self.register_buffer("episode_id", torch.zeros(B0, dtype=torch.long))
        self.register_buffer("has_previous_event", torch.zeros(B0, dtype=torch.bool))
        self.event_completion_gate = nn.Sequential(
            nn.Linear(memoryDim * 4 + 1, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, 3),)

        self.usage_bank = ObjectUsageBank()
        object_input_dim = (
            512 + 7 + 4 + 1
            + ModuleDim.PstIdDim
            + ModuleDim.PstAttrDim)
        self.object_relational_proj = nn.Sequential(
            nn.LayerNorm(object_input_dim),
            nn.Linear(object_input_dim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.object_relation_attention = nn.MultiheadAttention(
            memoryDim,
            num_heads=8,
            batch_first=True)
        self.object_query_proj = nn.Sequential(
            nn.LayerNorm(inputDim),
            nn.Linear(inputDim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.ontology_focus_key = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.ontology_object_encoder = nn.Sequential(
            nn.LayerNorm(self.ontology_feature_dim),
            nn.Linear(self.ontology_feature_dim, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),)
        self.ontology_object_gain = nn.Parameter(torch.tensor(-2.944439))
        self.object_attention_anchor = nn.Linear(
            inputDim,
            memoryDim,
            bias=False)
        self.object_identity_proj = nn.Linear(512, ModuleDim.PstIdDim)
        self.object_attribute_proj = nn.Linear(512, ModuleDim.PstAttrDim)
        self.usage_memory_proj = nn.Sequential(
            nn.LayerNorm(ModuleDim.PstUsageDim),
            nn.Linear(ModuleDim.PstUsageDim, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, memoryDim),)
        self.embodied_memory_expert = nn.Sequential(
            nn.LayerNorm(memoryDim * 2),
            nn.Linear(memoryDim * 2, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),)
        self.embodied_output_proj = nn.Linear(memoryDim, outputDim)
        self.embodied_memory_gate = nn.Sequential(
            nn.LayerNorm(outputDim * 2),
            nn.Linear(outputDim * 2, memoryDim),
            nn.SiLU(),
            nn.Linear(memoryDim, 1),
            nn.Sigmoid(),)
        self.replay_capacity = max(1, int(replayCapacity))
        self._plan_cache: Dict[str, Dict[str, Any]] = {}
        self._skill_cache: Dict[str, Dict[str, Any]] = {}
        self._offline_replay: List[Dict[str, Any]] = []
        self._replay_signature: Optional[str] = None
        self._replay_sequence = 0
        self._replay_transaction_version = 0
        self._replay_timeline_version = 0
        self.counterfactual_replay_predictor = nn.Sequential(
            nn.LayerNorm(memoryDim),
            nn.Linear(memoryDim, memoryDim * 2),
            nn.SiLU(),
            nn.Linear(memoryDim * 2, memoryDim),)
        nn.init.zeros_(self.counterfactual_replay_predictor[-1].weight)
        nn.init.zeros_(self.counterfactual_replay_predictor[-1].bias)
        self._row_merge_contract_id: Optional[str] = None
        self._row_merge_model_signature: Optional[str] = None
        self._row_merge_transactions: Dict[str, str] = {}
        self._row_merge_versions: Dict[int, Tuple[int, int]] = {}

    def ValidateModelSignature(self, modelSignature: str) -> str:
        if type(modelSignature) is not str or not modelSignature.strip():
            raise ValueError("modelSignature must be a non-empty string")
        return modelSignature

    def ValidateWriteMask(
        self,
        writeMask: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
    ) -> torch.Tensor:
        if writeMask is None:
            return torch.ones(batchSize, device=device, dtype=torch.bool)
        if not torch.is_tensor(writeMask) or writeMask.dtype != torch.bool:
            raise TypeError("writeMask must be a bool tensor")
        if writeMask.device != device:
            raise ValueError("writeMask must be on the memory input device")
        if tuple(writeMask.shape) != (batchSize,):
            raise ValueError("writeMask must have shape [B]")
        return writeMask

    def ValidateLossSampleMask(
        self,
        lossSampleMask: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
    ) -> torch.Tensor:
        if lossSampleMask is None:
            return torch.ones(
                batchSize, device=device, dtype=torch.bool)
        if (
            not torch.is_tensor(lossSampleMask)
            or tuple(lossSampleMask.shape) != (batchSize,)
            or lossSampleMask.dtype != torch.bool
            or lossSampleMask.device != device
        ):
            raise ValueError("lossSampleMask must be a boolean batch vector")
        return lossSampleMask

    def MutableMemoryStateTensors(self) -> Dict[str, torch.Tensor]:
        state = dict(self.DurableStateTensors())
        state.update({
            "last_compress_step": self.last_compress_step,
            "h_state": self.h_state,
            "fast_weights": self.fast_weights,
            "ns_prev_P_post": self.ns_prev_P_post,
            "ns_penalty_vec": self.ns_penalty_vec,
            "pattern_usage": self.pattern_usage,
            "previous_attention": self.previous_attention,
            "previous_intent": self.previous_intent,
            "previous_object_summary": self.previous_object_summary,
            "previous_motion_token": self.previous_motion_token,
            "previous_ontology_context": self.previous_ontology_context,
            "event_age": self.event_age,
            "has_previous_event": self.has_previous_event,
            "gws_global_step": self.gws.global_step,
            "gws_keys": self.gws.keys,
            "gws_vals": self.gws.vals,
            "gws_priority": self.gws.priority,
            "gws_ttl": self.gws.ttl,
            "gws_created_step": self.gws.created_step,
            "gws_last_step": self.gws.last_step,
            "gws_last_rehearsal_step": self.gws.last_rehearsal_step,
            "gws_touch": self.gws.touch,
            "gws_source": self.gws.source,
            "gws_source_confidence": self.gws.source_confidence,})
        return state

    @torch.no_grad()
    def CaptureFrozenRows(
        self,
        writeMask: torch.Tensor,
        fieldNames: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        frozen = ~writeMask
        if not bool(frozen.any().item()):
            return {}
        state = self.MutableMemoryStateTensors()
        names = (
            tuple(state)
            if fieldNames is None
            else tuple(fieldNames))
        return {
            name: tensor[frozen].detach().clone()
            for name in names
            for tensor in (state[name],)
            if name not in self.UNBATCHED_MEMORY_STATE_FIELDS}

    @torch.no_grad()
    def RestoreFrozenRows(
        self,
        frozenState: Dict[str, torch.Tensor],
        writeMask: torch.Tensor,
    ) -> None:
        if not frozenState:
            return
        frozen = ~writeMask
        targets = self.MutableMemoryStateTensors()
        for name, value in frozenState.items():
            target = targets[name]
            restored = target.clone()
            restored[frozen] = value
            self.ReplaceMutableStateTensor(target, restored)

    def ReplaceMutableStateTensor(
        self,
        target: torch.Tensor,
        replacement: torch.Tensor,
    ) -> None:
        for module in self.modules():
            for name, value in module._buffers.items():
                if value is target:
                    setattr(module, name, replacement)
                    return
        raise RuntimeError("mutable memory state tensor is not a registered buffer")

    def ValidateCacheKey(self, key: str) -> str:
        if type(key) is not str or not key.strip():
            raise ValueError("cache key must be a non-empty string")
        return key

    def CloneCognitiveRecord(self, value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {
                key: self.CloneCognitiveRecord(item)
                for key, item in value.items()}
        if isinstance(value, list):
            return [self.CloneCognitiveRecord(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.CloneCognitiveRecord(item) for item in value)
        return value

    def CachePlan(
        self,
        planId: str,
        planFeature: torch.Tensor,
        modelSignature: str,
        validMask: Optional[torch.Tensor] = None,
    ) -> None:
        plan_id = self.ValidateCacheKey(planId)
        signature = self.ValidateModelSignature(modelSignature)
        if not torch.is_tensor(planFeature) or planFeature.dim() < 2:
            raise ValueError("planFeature must be a batched tensor")
        batch_size = int(planFeature.size(0))
        if validMask is None:
            valid = torch.ones(
                batch_size,
                dtype=torch.bool,
                device=planFeature.device)
        else:
            if (
                not torch.is_tensor(validMask)
                or tuple(validMask.shape) != (batch_size,)
                or validMask.dtype != torch.bool
                or validMask.device != planFeature.device
            ):
                raise ValueError(
                    "validMask must match the plan feature batch")
            valid = validMask
        previous = self._plan_cache.get(plan_id)
        if previous is not None and previous["modelSignature"] != signature:
            raise ValueError("plan cache model signature mismatch")
        if previous is not None and tuple(previous["feature"].shape) != tuple(
            planFeature.shape
        ):
            raise ValueError("plan cache feature shape mismatch")
        version = 1 if previous is None else int(previous["version"]) + 1
        if previous is None:
            feature = torch.zeros_like(planFeature)
            previous_valid = torch.zeros_like(valid)
            age = torch.zeros(
                batch_size,
                device=planFeature.device,
                dtype=torch.long)
        else:
            feature = previous["feature"].to(
                device=planFeature.device,
                dtype=planFeature.dtype).clone()
            previous_valid = previous["valid"].to(
                device=valid.device).clone()
            previous_age = previous["age"]
            if type(previous_age) is int:
                age = torch.full(
                    (batch_size,),
                    previous_age,
                    device=planFeature.device,
                    dtype=torch.long)
            else:
                age = previous_age.to(
                    device=planFeature.device,
                    dtype=torch.long).clone()
            if tuple(age.shape) != (batch_size,) or bool(
                (age < 0).any().item()
            ):
                raise ValueError("plan cache age is invalid")
        selection = valid.reshape(
            (batch_size,) + (1,) * (planFeature.dim() - 1))
        feature = torch.where(
            selection,
            planFeature.detach(),
            feature)
        age = torch.where(valid, torch.zeros_like(age), age)
        self._plan_cache[plan_id] = {
            "modelSignature": signature,
            "feature": feature,
            "valid": previous_valid | valid.detach(),
            "age": age,
            "version": version,}

    def RecallPlan(
        self,
        planId: str,
        modelSignature: str,) -> Optional[Dict[str, Any]]:
        plan_id = self.ValidateCacheKey(planId)
        signature = self.ValidateModelSignature(modelSignature)
        cached = self._plan_cache.get(plan_id)
        if cached is None:
            return None
        if cached["modelSignature"] != signature:
            raise ValueError("plan cache model signature mismatch")
        if not bool(cached["valid"].any().item()):
            return None
        return self.CloneCognitiveRecord(cached)

    def AgePlanCache(
        self,
        increment: int = 1,
        ageMask: Optional[torch.Tensor] = None,
        ) -> None:
        step = int(increment)
        if step < 0:
            raise ValueError("plan cache age increment must be non-negative")
        for cached in self._plan_cache.values():
            age = cached["age"]
            valid = cached["valid"]
            if type(age) is int:
                age = torch.full_like(valid, age, dtype=torch.long)
            if (
                not torch.is_tensor(age)
                or tuple(age.shape) != tuple(valid.shape)
                or age.dtype != torch.long
                or age.device != valid.device
                or bool((age < 0).any().item())
            ):
                raise ValueError("plan cache age is invalid")
            if ageMask is None:
                age_mask = torch.ones_like(valid)
            elif (
                not torch.is_tensor(ageMask)
                or tuple(ageMask.shape) != tuple(valid.shape)
                or ageMask.dtype != torch.bool
                or ageMask.device != valid.device
            ):
                raise ValueError("ageMask must match the plan cache batch")
            else:
                age_mask = ageMask
            bounded_step = min(step, torch.iinfo(age.dtype).max)
            incremented = age.clamp_max(
                torch.iinfo(age.dtype).max - bounded_step) + bounded_step
            cached["age"] = torch.where(
                valid & age_mask,
                incremented,
                age)

    def InvalidatePlan(
        self,
        planId: str,
        modelSignature: str,) -> None:
        plan_id = self.ValidateCacheKey(planId)
        signature = self.ValidateModelSignature(modelSignature)
        cached = self._plan_cache.get(plan_id)
        if cached is None:
            return
        if cached["modelSignature"] != signature:
            raise ValueError("plan cache model signature mismatch")
        self._plan_cache.pop(plan_id)

    def CacheSkill(
        self,
        skillId: str,
        skillFeature: torch.Tensor,
        modelSignature: str,) -> None:
        skill_id = self.ValidateCacheKey(skillId)
        signature = self.ValidateModelSignature(modelSignature)
        if not torch.is_tensor(skillFeature) or skillFeature.dim() < 1:
            raise ValueError("skillFeature must be a tensor")
        previous = self._skill_cache.get(skill_id)
        if previous is not None and previous["modelSignature"] != signature:
            raise ValueError("skill cache model signature mismatch")
        self._skill_cache[skill_id] = {
            "modelSignature": signature,
            "feature": skillFeature.detach().clone(),}

    def RecallSkill(
        self,
        skillId: str,
        modelSignature: str,) -> Optional[torch.Tensor]:
        skill_id = self.ValidateCacheKey(skillId)
        signature = self.ValidateModelSignature(modelSignature)
        cached = self._skill_cache.get(skill_id)
        if cached is None:
            return None
        if cached["modelSignature"] != signature:
            raise ValueError("skill cache model signature mismatch")
        return cached["feature"].detach().clone()

    def ValidateReplaySignature(self, modelSignature: str) -> str:
        signature = self.ValidateModelSignature(modelSignature)
        if self._replay_signature is None:
            self._replay_signature = signature
        elif self._replay_signature != signature:
            raise ValueError("offline replay model signature mismatch")
        return signature

    def ValidateReplayVersion(
        self,
        transactionVersion: int,
        timelineVersion: int,
    ) -> Tuple[int, int]:
        if (
            type(transactionVersion) is not int
            or transactionVersion < 0
            or type(timelineVersion) is not int
            or timelineVersion < 0
        ):
            raise ValueError("replay versions must be non-negative integers")
        return transactionVersion, timelineVersion

    def AdvanceReplayBoundary(
        self,
        transactionVersion: int,
        timelineVersion: int,
    ) -> None:
        transaction, timeline = self.ValidateReplayVersion(
            transactionVersion,
            timelineVersion)
        if timeline < self._replay_timeline_version:
            raise ValueError("offline replay timeline is stale")
        if (
            timeline == self._replay_timeline_version
            and transaction < self._replay_transaction_version
        ):
            raise ValueError("offline replay transaction is stale")
        if timeline > self._replay_timeline_version:
            self._replay_timeline_version = timeline
            self._replay_transaction_version = transaction
        else:
            self._replay_transaction_version = max(
                self._replay_transaction_version,
                transaction)

    def ValidateReplayConfidence(
        self,
        confidence: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if confidence is None:
            return torch.ones(batchSize, device=device, dtype=dtype)
        if not torch.is_tensor(confidence):
            raise TypeError("replay confidence must be a tensor")
        value = confidence.reshape(-1)
        if (
            tuple(value.shape) != (batchSize,)
            or value.device != device
            or not value.dtype.is_floating_point
            or not bool(torch.isfinite(value).all().item())
            or bool(((value < 0.0) | (value > 1.0)).any().item())
        ):
            raise ValueError("replay confidence must be finite within [0, 1]")
        return value.to(dtype=dtype)

    def RecordReplayEpisode(
        self,
        kind: str,
        context: torch.Tensor,
        outcome: torch.Tensor,
        modelSignature: str,
        *,
        confidence: Optional[torch.Tensor] = None,
        transactionVersion: int = 0,
        timelineVersion: int = 0,) -> None:
        episode_kind = self.ValidateCacheKey(kind)
        signature = self.ValidateModelSignature(modelSignature)
        if self._replay_signature not in (None, signature):
            raise ValueError("offline replay model signature mismatch")
        if not torch.is_tensor(context) or context.dim() < 2:
            raise ValueError("episode context must be a batched tensor")
        if not torch.is_tensor(outcome) or outcome.dim() < 2:
            raise ValueError("episode outcome must be a batched tensor")
        if context.size(0) != outcome.size(0):
            raise ValueError("episode context and outcome batch sizes must match")
        if (
            not context.dtype.is_floating_point
            or not outcome.dtype.is_floating_point
            or not bool(torch.isfinite(context).all().item())
            or not bool(torch.isfinite(outcome).all().item())
        ):
            raise ValueError("episode tensors must be finite floating point values")
        if context.device != outcome.device:
            raise ValueError("episode tensors must share a device")
        transaction, timeline = self.ValidateReplayVersion(
            transactionVersion,
            timelineVersion)
        replay_confidence = self.ValidateReplayConfidence(
            confidence,
            int(context.size(0)),
            context.device,
            context.dtype)
        self.AdvanceReplayBoundary(transaction, timeline)
        if self._replay_signature is None:
            self._replay_signature = signature
        self._offline_replay.append({
            "kind": episode_kind,
            "context": context.detach().clone(),
            "outcome": outcome.detach().clone(),
            "confidence": replay_confidence.detach().clone(),
            "modelSignature": signature,
            "sequence": self._replay_sequence,
            "transactionVersion": transaction,
            "timelineVersion": timeline,})
        self._replay_sequence += 1
        overflow = len(self._offline_replay) - self.replay_capacity
        if overflow > 0:
            del self._offline_replay[:overflow]

    def RecordHierarchyTransition(
        self,
        context: torch.Tensor,
        coarseProgress: torch.Tensor,
        detailProgress: torch.Tensor,
        modelSignature: str,
        *,
        confidence: Optional[torch.Tensor] = None,
        transactionVersion: int = 0,
        timelineVersion: int = 0,) -> None:
        coarse = coarseProgress.reshape(context.size(0), -1)
        detail = detailProgress.reshape(context.size(0), -1)
        self.RecordReplayEpisode(
            "hierarchyTransition",
            context,
            torch.cat([coarse, detail], dim=-1),
            modelSignature,
            confidence=confidence,
            transactionVersion=transactionVersion,
            timelineVersion=timelineVersion)

    def RecordFailureEpisode(
        self,
        context: torch.Tensor,
        outcome: torch.Tensor,
        modelSignature: str,
        *,
        confidence: Optional[torch.Tensor] = None,
        transactionVersion: int = 0,
        timelineVersion: int = 0,) -> None:
        self.RecordReplayEpisode(
            "failure",
            context,
            outcome,
            modelSignature,
            confidence=confidence,
            transactionVersion=transactionVersion,
            timelineVersion=timelineVersion)

    def RecordCounterfactualEpisode(
        self,
        context: torch.Tensor,
        outcome: torch.Tensor,
        modelSignature: str,
        *,
        confidence: Optional[torch.Tensor] = None,
        transactionVersion: int = 0,
        timelineVersion: int = 0,) -> None:
        replay_confidence = confidence
        if not torch.is_tensor(outcome) or outcome.dim() < 2:
            raise ValueError("episode outcome must be a batched tensor")
        flat_outcome = outcome.reshape(outcome.size(0), -1)
        if (
            replay_confidence is None
            and flat_outcome.size(1) == self.COUNTERFACTUAL_OUTCOME_DIM
        ):
            replay_confidence = flat_outcome[
                :,
                self.COUNTERFACTUAL_INFORMATION].clamp(0.0, 1.0)
        self.RecordReplayEpisode(
            "counterfactual",
            context,
            outcome,
            modelSignature,
            confidence=replay_confidence,
            transactionVersion=transactionVersion,
            timelineVersion=timelineVersion)

    def SampleOfflineReplay(
        self,
        batchSize: int,
        modelSignature: str,
        *,
        kind: Optional[str] = None,
        transactionVersion: Optional[int] = None,
        timelineVersion: Optional[int] = None,
        seed: Optional[int] = None,) -> List[Dict[str, Any]]:
        self.ValidateReplaySignature(modelSignature)
        count = int(batchSize)
        if count <= 0:
            raise ValueError("batchSize must be positive")
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError("replay seed must be a non-negative integer")
        transaction = (
            self._replay_transaction_version
            if transactionVersion is None
            else transactionVersion)
        timeline = (
            self._replay_timeline_version
            if timelineVersion is None
            else timelineVersion)
        transaction, timeline = self.ValidateReplayVersion(
            transaction,
            timeline)
        if (
            transaction != self._replay_transaction_version
            or timeline != self._replay_timeline_version
        ):
            raise ValueError("offline replay boundary is stale")
        selected_kind = None if kind is None else self.ValidateCacheKey(kind)
        candidates = [
            record
            for record in self._offline_replay
            if (
                record["timelineVersion"] < timeline
                or (
                    record["timelineVersion"] == timeline
                    and record["transactionVersion"] <= transaction))
            and (selected_kind is None or record["kind"] == selected_kind)]
        if seed is None or len(candidates) <= count:
            return self.CloneCognitiveRecord(candidates[-count:])
        weights = torch.tensor([
            float(record["confidence"].mean().item())
            for record in candidates], dtype=torch.float64)
        valid = weights > 0.0
        if not bool(valid.any().item()):
            return []
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        sample_count = min(count, int(valid_indices.numel()))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        sampled = torch.multinomial(
            weights.index_select(0, valid_indices),
            sample_count,
            replacement=False,
            generator=generator)
        indices = valid_indices.index_select(0, sampled).sort().values.tolist()
        return self.CloneCognitiveRecord([candidates[index] for index in indices])

    def BuildCounterfactualRevision(
        self,
        context: torch.Tensor,
        outcome: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if (
            not torch.is_tensor(context)
            or context.dim() != 2
            or int(context.size(1)) != self.memory_dim
            or not context.dtype.is_floating_point
            or not bool(torch.isfinite(context).all().item())
        ):
            raise ValueError("counterfactual context has invalid shape or values")
        if (
            not torch.is_tensor(outcome)
            or outcome.dim() != 2
            or tuple(outcome.shape) != (
                int(context.size(0)),
                self.COUNTERFACTUAL_OUTCOME_DIM)
            or outcome.device != context.device
            or not outcome.dtype.is_floating_point
            or not bool(torch.isfinite(outcome).all().item())
        ):
            raise ValueError("counterfactual outcome has invalid shape or values")
        replay_confidence = self.ValidateReplayConfidence(
            confidence,
            int(context.size(0)),
            context.device,
            context.dtype)
        predicted_reward = outcome[:, self.COUNTERFACTUAL_PREDICTED_REWARD]
        corrected_reward = outcome[:, self.COUNTERFACTUAL_CORRECTED_REWARD]
        predicted_done = outcome[:, self.COUNTERFACTUAL_PREDICTED_DONE]
        corrected_done = outcome[:, self.COUNTERFACTUAL_CORRECTED_DONE]
        reward_variance = outcome[:, self.COUNTERFACTUAL_REWARD_VARIANCE]
        done_variance = outcome[:, self.COUNTERFACTUAL_DONE_VARIANCE]
        information = outcome[:, self.COUNTERFACTUAL_INFORMATION]
        if (
            bool(((predicted_done < 0.0) | (predicted_done > 1.0)).any().item())
            or bool(((corrected_done < 0.0) | (corrected_done > 1.0)).any().item())
            or bool((reward_variance < 0.0).any().item())
            or bool((done_variance < 0.0).any().item())
            or bool(((information < 0.0) | (information > 1.0)).any().item())
        ):
            raise ValueError("counterfactual outcome violates its probability schema")
        epsilon = torch.finfo(context.dtype).eps
        reward_scale = torch.sqrt(reward_variance + epsilon)
        done_scale = torch.sqrt(
            done_variance
            + corrected_done * (1.0 - corrected_done)
            + epsilon)
        reward_innovation = torch.tanh(
            (corrected_reward - predicted_reward) / reward_scale)
        done_innovation = torch.tanh(
            (corrected_done - predicted_done) / done_scale)
        predicted_done_centered = predicted_done * 2.0 - 1.0
        corrected_done_centered = corrected_done * 2.0 - 1.0
        information_centered = information * 2.0 - 1.0
        descriptor = torch.stack([
            reward_innovation,
            done_innovation,
            predicted_done_centered,
            corrected_done_centered,
            information_centered,
            reward_innovation * done_innovation,
            reward_innovation * information_centered,
            done_innovation * information_centered,
        ], dim=-1)
        if self.memory_dim != descriptor.size(1):
            descriptor = F.interpolate(
                descriptor.unsqueeze(1),
                size=self.memory_dim,
                mode="linear",
                align_corners=False).squeeze(1)
        descriptor = F.normalize(descriptor, dim=-1)
        context_scale = context.square().mean(dim=-1).sqrt().clamp_min(1.0)
        revision_scale = (
            0.25 * (0.25 + information) * context_scale
        ).unsqueeze(-1)
        residual = (
            replay_confidence.unsqueeze(-1)
            * revision_scale
            * descriptor)
        return {
            "query": F.normalize(context, dim=-1),
            "revisedValue": context + residual,
            "residual": residual,
            "descriptor": descriptor,}

    def ConsumeCounterfactualReplay(
        self,
        batchSize: int,
        modelSignature: str,
        *,
        transactionVersion: Optional[int] = None,
        timelineVersion: Optional[int] = None,
        seed: int = 0,
        addInternalLoss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        count = int(batchSize)
        if count <= 0:
            raise ValueError("batchSize must be positive")
        if type(seed) is not int or seed < 0:
            raise ValueError("replay seed must be a non-negative integer")
        records = self.SampleOfflineReplay(
            self.replay_capacity,
            modelSignature,
            kind="counterfactual",
            transactionVersion=transactionVersion,
            timelineVersion=timelineVersion)
        contexts = []
        outcomes = []
        confidences = []
        sequences = []
        row_indices = []
        for record in records:
            flat_context = record["context"].reshape(
                record["context"].size(0), -1)
            flat_outcome = record["outcome"].reshape(
                record["outcome"].size(0), -1)
            if int(flat_context.size(1)) != self.memory_dim:
                raise ValueError(
                    "counterfactual replay context does not match memory dimension")
            if int(flat_outcome.size(1)) != self.COUNTERFACTUAL_OUTCOME_DIM:
                raise ValueError("counterfactual replay outcome schema does not match")
            contexts.append(flat_context)
            outcomes.append(flat_outcome)
            confidences.append(record["confidence"])
            sequences.append(torch.full(
                (flat_context.size(0),),
                int(record["sequence"]),
                device=flat_context.device,
                dtype=torch.long))
            row_indices.append(torch.arange(
                flat_context.size(0),
                device=flat_context.device,
                dtype=torch.long))
        if not contexts:
            zero = sum((
                parameter.sum() * 0.0
                for parameter in self.counterfactual_replay_predictor.parameters()),
                self.h_state.sum() * 0.0)
            empty_feature = self.h_state.new_zeros(0, self.memory_dim)
            empty_scalar = self.h_state.new_zeros(0)
            empty_index = torch.zeros(0, device=self.device, dtype=torch.long)
            return {
                "loss": zero,
                "query": empty_feature,
                "revisedValue": empty_feature.clone(),
                "confidence": empty_scalar,
                "sequence": empty_index,
                "rowIndex": empty_index.clone(),
                "available": torch.zeros(
                    (), device=self.device, dtype=torch.bool),
                "sampleCount": torch.zeros(
                    (), device=self.device, dtype=torch.long),}
        context = torch.cat(contexts, dim=0).to(
            device=self.device,
            dtype=self.dtype)
        outcome = torch.cat(outcomes, dim=0).to(
            device=self.device,
            dtype=self.dtype)
        confidence = torch.cat(confidences, dim=0).to(
            device=self.device,
            dtype=self.dtype)
        sequence = torch.cat(sequences, dim=0).to(device=self.device)
        row_index = torch.cat(row_indices, dim=0).to(device=self.device)
        valid = confidence > 0.0
        if not bool(valid.any().item()):
            zero = self.counterfactual_replay_predictor[1].weight.sum() * 0.0
            empty_feature = context.new_zeros(0, self.memory_dim)
            empty_scalar = context.new_zeros(0)
            empty_index = torch.zeros(0, device=self.device, dtype=torch.long)
            return {
                "loss": zero,
                "query": empty_feature,
                "revisedValue": empty_feature.clone(),
                "confidence": empty_scalar,
                "sequence": empty_index,
                "rowIndex": empty_index.clone(),
                "available": torch.zeros(
                    (), device=self.device, dtype=torch.bool),
                "sampleCount": torch.zeros(
                    (), device=self.device, dtype=torch.long),}
        candidate_index = torch.nonzero(valid, as_tuple=False).reshape(-1)
        weights = confidence.index_select(0, candidate_index).double().cpu()
        sample_count = min(count, int(candidate_index.numel()))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        sampled_local = torch.multinomial(
            weights,
            sample_count,
            replacement=False,
            generator=generator)
        sampled = candidate_index.index_select(
            0,
            sampled_local.to(device=candidate_index.device))
        context = context.index_select(0, sampled)
        outcome = outcome.index_select(0, sampled)
        confidence = confidence.index_select(0, sampled)
        sequence = sequence.index_select(0, sampled)
        row_index = row_index.index_select(0, sampled)
        revision = self.BuildCounterfactualRevision(
            context,
            outcome,
            confidence)
        prediction_scale = context.square().mean(
            dim=-1).sqrt().clamp_min(1.0).unsqueeze(-1)
        prediction = (
            context
            + 0.5
            * prediction_scale
            * torch.tanh(self.counterfactual_replay_predictor(context)))
        target = revision["revisedValue"].detach()
        regression = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none").mean(dim=-1)
        angular = 1.0 - F.cosine_similarity(
            prediction,
            target,
            dim=-1).clamp(-1.0, 1.0)
        per_sample = regression + 0.1 * angular
        loss = (
            per_sample * confidence
        ).sum() / confidence.sum().clamp_min(torch.finfo(confidence.dtype).eps)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("counterfactual replay loss is non-finite")
        if self.training and addInternalLoss:
            self.AddInternalLoss(0.05 * loss)
        return {
            "loss": loss,
            "query": revision["query"],
            "revisedValue": revision["revisedValue"],
            "confidence": confidence,
            "sequence": sequence,
            "rowIndex": row_index,
            "available": torch.ones(
                (), device=self.device, dtype=torch.bool),
            "sampleCount": torch.tensor(
                sample_count,
                device=self.device,
                dtype=torch.long),}

    def ExportCognitiveCacheState(self) -> Dict[str, Any]:
        return self.CloneCognitiveRecord({
            "planCache": self._plan_cache,
            "skillCache": self._skill_cache,
            "offlineReplay": self._offline_replay,
            "replaySignature": self._replay_signature,
            "replaySequence": self._replay_sequence,
            "replayTransactionVersion": self._replay_transaction_version,
            "replayTimelineVersion": self._replay_timeline_version,
        })

    def ImportCognitiveCacheState(
        self,
        state: Dict[str, Any],
        *,
        modelSignature: str,
        batchSize: int,
    ) -> None:
        signature = self.ValidateModelSignature(modelSignature)
        if type(batchSize) is not int or batchSize < 1:
            raise ValueError("batchSize must be positive")
        legacy_fields = {
            "planCache",
            "skillCache",
            "offlineReplay",
            "replaySignature",
            "replaySequence",
        }
        current_fields = legacy_fields | {
            "replayTransactionVersion",
            "replayTimelineVersion",
        }
        state_fields = set(state) if type(state) is dict else set()
        if (
            type(state) is not dict
            or (
                state_fields != legacy_fields
                and state_fields != current_fields)
        ):
            raise ValueError("cognitive cache state fields do not match")
        plan_cache = state["planCache"]
        skill_cache = state["skillCache"]
        offline_replay = state["offlineReplay"]
        replay_signature = state["replaySignature"]
        replay_sequence = state["replaySequence"]
        replay_transaction = state.get("replayTransactionVersion", 0)
        replay_timeline = state.get("replayTimelineVersion", 0)
        self.ValidateReplayVersion(replay_transaction, replay_timeline)
        if (
            type(plan_cache) is not dict
            or type(skill_cache) is not dict
            or type(offline_replay) is not list
            or replay_signature not in (None, signature)
            or self._replay_signature not in (None, signature)
            or type(replay_sequence) is not int
            or replay_sequence < 0
        ):
            raise ValueError("cognitive cache state is invalid")
        batch_size = int(self.h_state.size(0))
        if batchSize != batch_size:
            raise ValueError("cognitive cache batch does not match memory")
        for plan_id, cached in plan_cache.items():
            self.ValidateCacheKey(plan_id)
            if type(cached) is not dict or set(cached) != {
                "modelSignature",
                "feature",
                "valid",
                "age",
                "version",
            }:
                raise ValueError("plan cache record fields do not match")
            if cached["modelSignature"] != signature:
                raise ValueError("plan cache model signature mismatch")
            if (
                not torch.is_tensor(cached["feature"])
                or cached["feature"].dim() < 2
                or int(cached["feature"].size(0)) != batch_size
                or not torch.is_tensor(cached["valid"])
                or tuple(cached["valid"].shape) != (batch_size,)
                or cached["valid"].dtype != torch.bool
                or cached["valid"].device != cached["feature"].device
                or (
                    type(cached["age"]) is not int
                    and (
                        not torch.is_tensor(cached["age"])
                        or tuple(cached["age"].shape) != (batch_size,)
                        or cached["age"].dtype != torch.long
                        or cached["age"].device != cached["feature"].device
                        or bool((cached["age"] < 0).any().item())))
                or (
                    type(cached["age"]) is int
                    and cached["age"] < 0)
                or type(cached["version"]) is not int
                or cached["version"] < 1
            ):
                raise ValueError("plan cache record is invalid")
        for skill_id, cached in skill_cache.items():
            self.ValidateCacheKey(skill_id)
            if type(cached) is not dict or set(cached) != {
                "modelSignature",
                "feature",
            }:
                raise ValueError("skill cache record fields do not match")
            if (
                cached["modelSignature"] != signature
                or not torch.is_tensor(cached["feature"])
            ):
                raise ValueError("skill cache record is invalid")
        previous_sequence = -1
        for record in offline_replay:
            legacy_record_fields = {
                "kind",
                "context",
                "outcome",
                "modelSignature",
                "sequence",
            }
            current_record_fields = legacy_record_fields | {
                "confidence",
                "transactionVersion",
                "timelineVersion",
            }
            fields = set(record) if type(record) is dict else set()
            if (
                type(record) is not dict
                or (
                    fields != legacy_record_fields
                    and fields != current_record_fields)
                or record["modelSignature"] != signature
                or not torch.is_tensor(record["context"])
                or not torch.is_tensor(record["outcome"])
                or record["context"].dim() < 2
                or record["outcome"].dim() < 2
                or not record["context"].dtype.is_floating_point
                or not record["outcome"].dtype.is_floating_point
                or not bool(torch.isfinite(record["context"]).all().item())
                or not bool(torch.isfinite(record["outcome"]).all().item())
                or int(record["context"].size(0))
                != int(record["outcome"].size(0))
                or type(record["sequence"]) is not int
                or record["sequence"] <= previous_sequence
                or record["sequence"] >= replay_sequence
            ):
                raise ValueError("offline replay record is invalid")
            if fields == current_record_fields:
                self.ValidateReplayConfidence(
                    record["confidence"],
                    int(record["context"].size(0)),
                    record["context"].device,
                    record["context"].dtype)
                transaction, timeline = self.ValidateReplayVersion(
                    record["transactionVersion"],
                    record["timelineVersion"])
            else:
                transaction, timeline = 0, 0
            if (
                timeline > replay_timeline
                or (
                    timeline == replay_timeline
                    and transaction > replay_transaction)
            ):
                raise ValueError("offline replay record exceeds its boundary")
            previous_sequence = record["sequence"]
        migrated_plan = self.CloneCognitiveRecord(plan_cache)
        for cached in migrated_plan.values():
            cached["feature"] = cached["feature"].to(
                device=self.device,
                dtype=self.dtype)
            cached["valid"] = cached["valid"].to(
                device=self.device,
                dtype=torch.bool)
            if type(cached["age"]) is int:
                cached["age"] = torch.full(
                    (batch_size,),
                    cached["age"],
                    device=self.device,
                    dtype=torch.long)
            else:
                cached["age"] = cached["age"].to(
                    device=self.device,
                    dtype=torch.long)
        migrated_skill = self.CloneCognitiveRecord(skill_cache)
        for cached in migrated_skill.values():
            cached["feature"] = cached["feature"].to(
                device=self.device)
        migrated_replay = self.CloneCognitiveRecord(offline_replay)
        for record in migrated_replay:
            record["context"] = record["context"].to(
                device=self.device,
                dtype=self.dtype)
            record["outcome"] = record["outcome"].to(
                device=self.device,
                dtype=self.dtype)
            if "confidence" not in record:
                record["confidence"] = torch.ones(
                    record["context"].size(0),
                    device=self.device,
                    dtype=self.dtype)
                record["transactionVersion"] = 0
                record["timelineVersion"] = 0
            else:
                record["confidence"] = record["confidence"].to(
                    device=self.device,
                    dtype=self.dtype)
        self._plan_cache = migrated_plan
        self._skill_cache = migrated_skill
        self._offline_replay = migrated_replay
        self._replay_signature = replay_signature
        self._replay_sequence = replay_sequence
        self._replay_transaction_version = replay_transaction
        self._replay_timeline_version = replay_timeline

    @torch.no_grad()
    def ReconsolidateSemantic(
        self,
        query: torch.Tensor,
        revisedValue: torch.Tensor,
        confidence: torch.Tensor,
        similarityThreshold: float = -1.0,) -> torch.Tensor:
        if query.shape != revisedValue.shape:
            raise ValueError("query and revisedValue shapes must match")
        if query.dim() != 2 or query.size(1) != self.memory_dim:
            raise ValueError("semantic reconsolidation tensors have invalid shape")
        if confidence.reshape(-1).size(0) != query.size(0):
            raise ValueError("semantic reconsolidation confidence has invalid shape")
        if (
            isinstance(similarityThreshold, bool)
            or not isinstance(similarityThreshold, (int, float))
            or not math.isfinite(float(similarityThreshold))
            or float(similarityThreshold) < -1.0
            or float(similarityThreshold) > 1.0
        ):
            raise ValueError("similarityThreshold must be finite and within [-1, 1]")
        self.EnsureB(int(query.size(0)))
        semantic = self.ltm.semantic
        valid = torch.arange(
            semantic.capacity,
            device=query.device).view(1, -1) < semantic.filled.view(-1, 1)
        similarity = torch.einsum(
            "bd,bmd->bm",
            F.normalize(query, dim=-1),
            F.normalize(semantic.keys, dim=-1))
        similarity = similarity.masked_fill(~valid, -torch.inf)
        best_similarity, index = similarity.max(dim=-1)
        accepted = (
            valid.any(dim=-1)
            & (confidence.reshape(-1) > 0.0)
            & (best_similarity >= float(similarityThreshold)))
        rows = torch.arange(query.size(0), device=query.device)
        alpha = confidence.reshape(-1).clamp(0.0, 1.0)
        current = semantic.vals[rows, index]
        residual = revisedValue - current
        updated = current + alpha.unsqueeze(-1) * residual
        semantic.vals[rows[accepted], index[accepted]] = updated[accepted]
        current_variance = semantic.prototype_variance[rows, index]
        updated_variance = (
            (1.0 - alpha).unsqueeze(-1) * current_variance
            + (alpha * (1.0 - alpha)).unsqueeze(-1)
            * residual.square()).clamp_min(0.0)
        semantic.prototype_variance[rows[accepted], index[accepted]] = (
            updated_variance[accepted])
        semantic.prototype_count[rows[accepted], index[accepted]] += alpha[accepted]
        semantic.consolidation_count[rows[accepted], index[accepted]] += 1
        semantic.last_rehearsal_step[rows[accepted], index[accepted]] = (
            semantic.global_step[rows[accepted]])
        if bool(accepted.any().item()):
            self.memory_version.add_(1)
        return accepted

    @torch.no_grad()
    def SemanticConsolidation(self, topk: int = 8) -> None:
        self.ConsolidateMemory(topk=topk)

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
        self.memory_last_access_steps = self.memory_last_access_steps.new_zeros(B, self.memory_size)
        self.memory_last_rehearsal_steps = self.memory_last_rehearsal_steps.new_zeros(B, self.memory_size)
        self.memory_touch = self.memory_touch.new_zeros(B, self.memory_size)
        self.memory_merge_count = self.memory_merge_count.new_zeros(B, self.memory_size)
        self.memory_source = self.memory_source.new_zeros(B, self.memory_size)
        self.memory_source_confidence = self.memory_source_confidence.new_zeros(B, self.memory_size)
        self.memory_realm = self.memory_realm.new_full(
            (B, self.memory_size),
            ONTOLOGY_REALM_UNKNOWN)
        self.memory_reward_abs = self.memory_reward_abs.new_zeros(
            B, self.memory_size)

        self.memory_emotion = self.memory_emotion.new_zeros(
            B, self.memory_size, self.emotion_dim)

        self.time_step = self.time_step.new_zeros(B)
        self.memory_filled = self.memory_filled.new_zeros(B)
        self.last_compress_step = self.last_compress_step.new_zeros(B)

        self.ns_prev_P_post = self.ns_prev_P_post.new_zeros(B, self.ns_K)
        self.ns_penalty_vec = self.ns_penalty_vec.new_zeros(B, 1)
        self.previous_attention = self.previous_attention.new_zeros(B, self.input_dim)
        self.previous_intent = self.previous_intent.new_zeros(B, 512)
        self.previous_object_summary = self.previous_object_summary.new_zeros(B, 512)
        self.previous_motion_token = self.previous_motion_token.new_zeros(B, 512)
        self.previous_ontology_context = (
            self.previous_ontology_context.new_zeros(
                B,
                self.ontology_feature_dim))
        self.event_age = self.event_age.new_zeros(B)
        self.event_id = self.event_id.new_zeros(B)
        self.episode_id = self.episode_id.new_zeros(B)
        self.has_previous_event = self.has_previous_event.new_zeros(B)

        self.gws.EnsureB(B)
        self.ltm.semantic.EnsureB(B)
        self.ltm.episodic.EnsureB(B)
        self.sym_mem.EnsureB(B)
        self.merged_delta_signature.fill_(-1)
        self.pending.clear()

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
        topk_vals, topk_idx = StableTopk(sim, k) # [B, k]
        topk_valid = torch.gather(valid, 1, topk_idx)
        topk_finite = torch.where(
            topk_valid,
            topk_vals,
            torch.zeros_like(topk_vals))
        count = topk_valid.sum(dim=1).to(topk_vals.dtype)
        denominator = torch.where(
            any_valid,
            count,
            torch.ones_like(count))
        m = topk_finite.sum(dim=1) / denominator # [B]
        centered = (
            topk_finite - m.unsqueeze(1)) * topk_valid.to(topk_vals.dtype)
        s = torch.sqrt(centered.square().sum(dim=1) / denominator) # [B]

        safe_sim = torch.where(
            any_valid.unsqueeze(1),
            sim,
            torch.zeros_like(sim))
        attn = torch.softmax(safe_sim.float(), dim=-1) # [B, M]
        attn = attn * valid.to(attn.dtype)
        age = (self.time_step.view(B, 1) - self.memory_steps).clamp(min=0).float()
        age_w = (attn * age).sum(dim=1)
        age_w = torch.tanh(age_w / 1000.0)

        out[:, 0] = m
        out[:, 1] = s
        out[:, 2] = age_w
        return out # [B, 3]

    def NsRules(
        self,
        P: torch.Tensor, # P: [B, nsK]
        P_prev: Optional[torch.Tensor],
        lossSampleMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        total_penalty, aux_reg = self.sym_rules(P, P_prev) # total_penalty: [B]

        if self.training:
            loss_mask = self.ValidateLossSampleMask(
                lossSampleMask,
                int(P.size(0)),
                P.device)
            if bool(loss_mask.any().item()):
                masked_penalty = total_penalty[loss_mask].mean()
                self.AddInternalLoss(
                    self.ns_lambda * (masked_penalty + aux_reg))

        return total_penalty

    @torch.no_grad()
    def NsStore(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        importance: torch.Tensor,
        sourceLabel: torch.Tensor,
        sourceConfidence: torch.Tensor,
        writeMask: torch.Tensor,):

        self.sym_mem.Store(
            key=key.detach(),
            value=value.detach(),
            score=importance,
            source=sourceLabel,
            sourceConfidence=sourceConfidence,
            writeMask=writeMask)

    def NsPostRead(
        self,
        memRecall: torch.Tensor, #[B, memory_dim]
        lossSampleMask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        P_post = self.ns_coder_post(memRecall) # [B, nsK]
        per_sample_post = self.NsRules(
            P_post,
            self.ns_prev_P_post,
            lossSampleMask=lossSampleMask) # [B]
        self.ns_prev_P_post = P_post.detach().clone()

        damp = torch.clamp(per_sample_post, 0, 1).view(-1, 1) # [B, 1]

        self.ns_penalty_vec = damp.detach().clone()
        return P_post # [B, nsK], [B, memory_dim]

    def FilmParams(self, film: nn.Module, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gb = film(src) # [B, 2D]
        g, b = gb.chunk(2, dim=-1) # [B,D], [B,D]
        g = self.film_clip * torch.tanh(g)
        b = self.film_clip * torch.tanh(b)
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

    def StableTransition(self) -> torch.Tensor:
        raw = torch.tanh(self.A_full)
        diagonal_raw = torch.diagonal(raw)
        off_diagonal = raw - torch.diag_embed(diagonal_raw)
        row_mass = off_diagonal.abs().sum(dim=1, keepdim=True) + 1.0
        direction = off_diagonal / row_mass
        retention = torch.sigmoid(self.state_retention_logit + diagonal_raw)
        coupling = torch.sigmoid(self.state_coupling_logit)
        return (
            torch.diag_embed(retention)
            + direction * ((1.0 - retention) * coupling).unsqueeze(1))

    def UpdateWorkingState(self, x: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        transition = self.StableTransition()
        candidate = torch.tanh(
            self.state_candidate_norm(previous @ transition.t() + self.B_mat(x)))
        gate_logits = self.state_input_gate(torch.cat([x, previous], dim=-1))
        gate = torch.sigmoid(gate_logits + self.grad_bridge)
        return gate * candidate + (1.0 - gate) * previous

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
        modulation = torch.tanh(self.context_modulation_gate(torch.cat([x, fused], dim=-1)))
        scale = 0.25 * torch.sigmoid(self.context_fuse_scale)
        return x + scale * modulation * F.layer_norm(x, (self.input_dim,))

    def PatternSeparate(
        self,
        x: torch.Tensor,
        writeMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.pattern_separation_proj(x)
        k = max(1, int(self.memory_dim) // 8)
        homeostatic_score = x.abs() / (1.0 + self.pattern_usage.unsqueeze(0))
        _, idx = torch.topk(homeostatic_score, k=k, dim=-1)
        sparse = torch.zeros_like(x)
        sparse.scatter_(1, idx, torch.gather(x, 1, idx))
        with torch.no_grad():
            activityRows = torch.zeros_like(x).scatter_(1, idx, 1.0)
            if writeMask is None:
                activity = activityRows.mean(dim=0)
                self.pattern_usage.mul_(0.995).add_(activity, alpha=0.005)
            elif bool(writeMask.any().item()):
                activity = activityRows[writeMask].mean(dim=0)
                self.pattern_usage.mul_(0.995).add_(activity, alpha=0.005)
        return F.normalize(sparse, dim=-1)

    def BuildOntologyObjectContext(
        self,
        attended: torch.Tensor,
        visualState: Any,) -> Dict[str, torch.Tensor]:
        auxiliary = visualState.Auxiliary
        realm = auxiliary["RealmProb"]
        agency = auxiliary["AgencyProb"]
        motion_layer = auxiliary["MotionLayerProb"]
        layer_agency = auxiliary["LayerAgencyProb"].flatten(-2)
        body_membership = auxiliary["BodyMembershipProb"].unsqueeze(-1)
        self_part = auxiliary["SelfPartSemantic"]
        physical_interaction = auxiliary[
            "PhysicalInteractionProb"].unsqueeze(-1)
        surface_uv = auxiliary["SurfaceUV"]
        surface_uv_confidence = auxiliary[
            "SurfaceUVConfidence"].unsqueeze(-1)
        surface_uv_evidence = surface_uv * surface_uv_confidence
        verification = auxiliary["VerificationConfidence"].unsqueeze(-1)
        content_motion = auxiliary["ContentMotionUV"]
        content_change = auxiliary["ContentChangeProb"].unsqueeze(-1)

        surface_parent = auxiliary["SurfaceParentProb"]
        surface_parent_summary = torch.stack([
            surface_parent[..., :-1].amax(dim=-1),
            surface_parent[..., -1],
        ], dim=-1)
        relations = auxiliary["OntologyRelationProb"]
        outgoing_relations = relations.amax(dim=2)
        incoming_relations = relations.amax(dim=1)

        object_features = torch.cat([
            realm,
            agency,
            motion_layer,
            layer_agency,
            body_membership,
            self_part,
            physical_interaction,
            surface_uv_evidence,
            surface_uv_confidence,
            verification,
            content_motion,
            content_change,
            surface_parent_summary,
            outgoing_relations,
            incoming_relations,
        ], dim=-1)

        presence = visualState.Auxiliary["PerceptualPresence"]
        valid = presence > 0.05
        any_valid = valid.any(dim=1)
        query = self.object_query_proj(attended)
        object_key = self.ontology_focus_key(visualState.ObjectTokens)
        focus_logits = torch.einsum(
            "bd,bkd->bk",
            query,
            object_key) / math.sqrt(float(self.memory_dim))
        focus_logits = focus_logits + torch.log(presence + 1e-6)
        focus_logits = focus_logits.masked_fill(~valid, -torch.inf)
        focus_logits = torch.where(
            any_valid.unsqueeze(1),
            focus_logits,
            torch.zeros_like(focus_logits))
        weights = F.softmax(focus_logits, dim=1)
        weights = torch.where(
            any_valid.unsqueeze(1),
            weights,
            torch.zeros_like(weights))
        context = torch.einsum("bk,bkd->bd", weights, object_features)

        realm_posterior = torch.einsum("bk,bkr->br", weights, realm)
        unknown = F.one_hot(
            torch.full(
                (attended.size(0),),
                ONTOLOGY_REALM_UNKNOWN,
                device=attended.device),
            num_classes=5).to(realm_posterior.dtype)
        realm_posterior = torch.where(
            any_valid.unsqueeze(1),
            realm_posterior,
            unknown)
        focused_realm = realm_posterior.argmax(dim=-1).to(torch.int8)

        physical_realm = (
            realm[..., ONTOLOGY_REALM_SELF]
            + realm[..., ONTOLOGY_REALM_EXTERNAL])
        physical_layer = motion_layer[..., 1] + motion_layer[..., 2]
        physical_motion = torch.einsum(
            "bk,bk->b",
            weights,
            physical_realm * physical_layer)
        content_change_score = torch.einsum(
            "bk,bk->b",
            weights,
            content_change.squeeze(-1))
        verification_score = torch.einsum(
            "bk,bk->b",
            weights,
            verification.squeeze(-1))
        virtual_effect = (
            realm_posterior[:, ONTOLOGY_REALM_VIRTUAL]
            + realm_posterior[:, ONTOLOGY_REALM_EFFECT])
        realm_entropy = -(
            realm_posterior.clamp_min(1e-8)
            * realm_posterior.clamp_min(1e-8).log()).sum(dim=-1)
        realm_entropy = realm_entropy / math.log(5.0)
        focus_index = weights.argmax(dim=1)
        batch_index = torch.arange(
            attended.size(0),
            device=attended.device)
        selected_agency = agency[batch_index, focus_index].argmax(
            dim=-1).to(torch.int8)
        selected_motion_layer = motion_layer[batch_index, focus_index]
        selected_parent_class = surface_parent[
            batch_index,
            focus_index].argmax(dim=-1)
        selected_parent = torch.where(
            any_valid & (selected_parent_class < surface_parent.size(-1) - 1),
            selected_parent_class,
            torch.full_like(selected_parent_class, -1))
        selected_surface_uv = surface_uv[batch_index, focus_index]
        selected_surface_uv_confidence = surface_uv_confidence[
            batch_index,
            focus_index,
            0]
        selected_verification = verification[
            batch_index,
            focus_index,
            0]
        selected_content_change = content_change[
            batch_index,
            focus_index,
            0]
        selected_agency = torch.where(
            any_valid,
            selected_agency,
            torch.full_like(selected_agency, ONTOLOGY_AGENCY_UNKNOWN))
        selected_motion_layer = torch.where(
            any_valid.unsqueeze(-1),
            selected_motion_layer,
            torch.zeros_like(selected_motion_layer))
        selected_surface_uv = torch.where(
            any_valid.unsqueeze(-1),
            selected_surface_uv,
            torch.zeros_like(selected_surface_uv))
        selected_surface_uv_confidence = torch.where(
            any_valid,
            selected_surface_uv_confidence,
            torch.zeros_like(selected_surface_uv_confidence))
        selected_verification = torch.where(
            any_valid,
            selected_verification,
            torch.zeros_like(selected_verification))
        selected_content_change = torch.where(
            any_valid,
            selected_content_change,
            torch.zeros_like(selected_content_change))
        return {
            "object_features": object_features,
            "weights": weights,
            "context": context,
            "realm_posterior": realm_posterior,
            "realm_id": focused_realm,
            "physical_motion": physical_motion,
            "content_change": content_change_score,
            "verification": verification_score,
            "virtual_effect": virtual_effect,
            "realm_entropy": realm_entropy,
            "episodic_metadata": {
                "agency_id": selected_agency,
                "motion_layer_prob": selected_motion_layer,
                "observed_surface_parent_index": selected_parent,
                "surface_uv": selected_surface_uv,
                "surface_uv_confidence": selected_surface_uv_confidence,
                "verification": selected_verification,
                "content_change": selected_content_change,},}

    def DetectEventBoundary(
        self,
        attended: torch.Tensor,
        visualState: Any,
        intentHint: torch.Tensor,
        tdError: torch.Tensor,
        risk: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,
        ontology: Dict[str, torch.Tensor],
        lossSampleMask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        presence = visualState.Auxiliary["PerceptualPresence"]
        valid_object = presence > 0.05
        object_weight = presence * valid_object.to(presence.dtype)
        object_summary = torch.einsum(
            "bk,bkd->bd",
            object_weight / (object_weight.sum(dim=1, keepdim=True) + 1e-6),
            visualState.ObjectTokens)
        object_summary = torch.where(
            valid_object.any(dim=1, keepdim=True),
            object_summary,
            torch.zeros_like(object_summary))
        attention_change = torch.linalg.vector_norm(
            attended - self.previous_attention, dim=-1) / math.sqrt(self.input_dim)
        intent_change = torch.linalg.vector_norm(
            intentHint - self.previous_intent, dim=-1) / math.sqrt(512.0)
        object_change = torch.linalg.vector_norm(
            object_summary - self.previous_object_summary, dim=-1) / math.sqrt(512.0)
        motion = torch.linalg.vector_norm(
            visualState.MotionToken - self.previous_motion_token,
            dim=-1) / math.sqrt(float(visualState.MotionToken.size(-1)))
        prediction_error = torch.linalg.vector_norm(
            visualState.PredErrorToken, dim=-1) / math.sqrt(float(visualState.PredErrorToken.size(-1)))
        features = torch.stack([
            attention_change,
            intent_change,
            object_change,
            motion,
            prediction_error,
            tdError.abs(),
            risk,
            uncertainty + (1.0 - confidence),], dim=-1)
        learned_probability = torch.sigmoid(
            self.event_boundary_net(features).squeeze(-1))
        ontology_delta = torch.linalg.vector_norm(
            ontology["context"] - self.previous_ontology_context,
            dim=-1) / math.sqrt(float(self.ontology_feature_dim))
        ontology_features = torch.stack([
            ontology_delta,
            ontology["content_change"],
            ontology["physical_motion"],
            ontology["verification"],
            ontology["virtual_effect"],
            ontology["realm_entropy"],
        ], dim=-1)
        ontology_probability = torch.sigmoid(
            self.ontology_boundary_net(ontology_features).squeeze(-1))
        probability = 1.0 - (
            (1.0 - learned_probability)
            * (1.0 - ontology_probability))
        physical_realm = (
            ontology["realm_posterior"][:, ONTOLOGY_REALM_SELF]
            + ontology["realm_posterior"][:, ONTOLOGY_REALM_EXTERNAL])
        screen_dominant = ontology["virtual_effect"] > physical_realm
        text_change = ontology.get(
            "text_change",
            ontology["content_change"].new_zeros(
                ontology["content_change"].shape))
        stable_screen_content = (
            screen_dominant
            & (ontology["content_change"] < 0.50)
            & (text_change < 0.50)
            & (intent_change < 0.35)
            & (tdError.abs() < 0.60)
            & (risk < 0.70))
        probability = torch.where(
            stable_screen_content,
            probability * ontology["content_change"],
            probability)
        age_boundary = (
            (self.event_age >= 7)
            & (
                (ontology["content_change"] > 0.35)
                | (ontology["physical_motion"] > 0.50)
                | (intent_change > 0.35)
                | ((attention_change > 0.75) & ~screen_dominant)))
        object_boundary = (
            (object_change > 0.75)
            & (~screen_dominant | (ontology["content_change"] > 0.50)))
        heuristic = (
            ~self.has_previous_event
            | age_boundary
            | ((attention_change > 0.75) & ~stable_screen_content)
            | object_boundary
            | (ontology["content_change"] > 0.70)
            | (text_change > 0.70)
            | ((ontology_delta > 0.50) & ~stable_screen_content)
            | (intent_change > 0.75)
            | (tdError.abs() > 0.60)
            | (risk > 0.70))
        boundary = heuristic | (probability > 0.50)
        if self.training:
            loss_mask = self.ValidateLossSampleMask(
                lossSampleMask,
                int(probability.size(0)),
                probability.device)
            if bool(loss_mask.any().item()):
                boundary_loss = F.binary_cross_entropy(
                    probability,
                    heuristic.float(),
                    reduction="none")
                self.AddInternalLoss(
                    0.01 * boundary_loss[loss_mask].mean())
        with torch.no_grad():
            self.previous_attention.copy_(attended)
            self.previous_intent.copy_(intentHint)
            self.previous_object_summary.copy_(object_summary)
            self.previous_motion_token.copy_(visualState.MotionToken)
            self.previous_ontology_context.copy_(ontology["context"])
            self.event_age.copy_(torch.where(boundary, torch.zeros_like(self.event_age), self.event_age + 1))
            self.event_id.add_(boundary.long())
            self.has_previous_event.fill_(True)
        return probability, boundary

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
        confidence: torch.Tensor,
        ontologyContext: torch.Tensor,
        entityTextContext: Optional[torch.Tensor] = None,
        writeMask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        vis = torch.cat([
            visualState.IntegratedFeat,
            visualState.MotionToken,
            visualState.PredErrorToken], dim=-1)
        base = self.event_context_proj(torch.cat([
            x,
            vis,
            ocrSemantic,
            intentHint,
            emotion,
            tdError.view(-1, 1),
            uncertainty.view(-1, 1),
            risk.view(-1, 1),
            confidence.view(-1, 1),], dim=-1))
        ontology_code = self.ontology_context_encoder(ontologyContext)
        residual = self.event_ontology_residual(torch.cat([
            base,
            ontology_code,
        ], dim=-1))
        raw = base + 0.25 * torch.sigmoid(
            self.event_ontology_gain) * residual
        if entityTextContext is not None:
            text_code = self.entity_text_event_encoder(entityTextContext)
            text_residual = self.event_text_residual(torch.cat([
                raw,
                text_code,], dim=-1))
            raw = raw + 0.25 * torch.sigmoid(
                self.entity_text_event_gain) * text_residual
        return raw, self.PatternSeparate(raw, writeMask=writeMask)

    def BuildEntityTextContext(
        self,
        visualState: Any,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        auxiliary = visualState.Auxiliary
        required = (
            "EntityTextSemantic",
            "EntityTextConfidence",
            "EntityTextRevision",
            "EntityTextChanged",)
        if any(name not in auxiliary for name in required):
            B, K = weights.shape
            return {
                "context": weights.new_zeros(B, self.entity_text_feature_dim),
                "object_code": weights.new_zeros(B, K, self.memory_dim),
                "change": weights.new_zeros(B),}
        confidence = auxiliary["EntityTextConfidence"].unsqueeze(-1)
        revision = torch.tanh(
            auxiliary["EntityTextRevision"].to(weights.dtype).unsqueeze(-1)
            / 16.0)
        changed = auxiliary["EntityTextChanged"].to(
            weights.dtype).unsqueeze(-1)
        features = torch.cat([
            auxiliary["EntityTextSemantic"].to(weights.dtype),
            confidence,
            revision,
            changed,], dim=-1)
        context = torch.einsum("bk,bkd->bd", weights, features)
        object_code = self.entity_text_object_encoder(features)
        object_code = object_code * confidence
        change = torch.einsum(
            "bk,bk->b",
            weights,
            changed.squeeze(-1))
        return {
            "context": context,
            "object_code": object_code,
            "change": change,}

    def BuildEmbodiedMemory(
        self,
        attended: torch.Tensor,
        visualState: Any,
        ontologyObjectFeatures: torch.Tensor,
        entityTextObjectCode: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        objects = visualState.ObjectTokens
        presence = visualState.Auxiliary["PerceptualPresence"]
        pose = visualState.SemanticNodes["SpatialState"]
        bbox = visualState.SemanticNodes["bbox_2d"]
        semantic_identity = torch.cat([
            F.softmax(visualState.SemanticNodes["level_logits"], dim=-1),
            F.softmax(visualState.SemanticNodes["object_class_logits"], dim=-1),
            F.softmax(visualState.SemanticNodes["part_class_logits"], dim=-1),], dim=-1)
        identity = F.normalize(torch.cat([
            F.normalize(visualState.SemanticNodes["identity_embed"], dim=-1),
            0.25 * semantic_identity,], dim=-1), dim=-1)
        latent_identity = self.object_identity_proj(objects)
        latent_attribute = self.object_attribute_proj(objects)
        object_input = torch.cat([
            objects,
            pose,
            bbox,
            presence.unsqueeze(-1),
            latent_identity,
            latent_attribute,], dim=-1)
        valid = presence > 0.05
        any_valid = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[~any_valid, 0] = True
        base_object_code = self.object_relational_proj(object_input)
        ontology_object_code = self.ontology_object_encoder(
            ontologyObjectFeatures)
        object_code = (
            base_object_code
            + 0.25 * torch.sigmoid(self.ontology_object_gain)
            * ontology_object_code) * valid.unsqueeze(-1)
        if entityTextObjectCode is not None:
            object_code = object_code + (
                0.25 * torch.sigmoid(self.entity_text_object_gain)
                * entityTextObjectCode
                * valid.unsqueeze(-1))
        relational_code, _ = self.object_relation_attention(
            object_code,
            object_code,
            object_code,
            key_padding_mask=~safe_valid,
            need_weights=False)
        relational_code = relational_code * valid.unsqueeze(-1)
        query = self.object_query_proj(attended)
        logits = torch.einsum(
            "bd,bkd->bk",
            query,
            relational_code) / math.sqrt(float(self.memory_dim))
        logits = logits.masked_fill(~valid, -torch.inf)
        logits = torch.where(any_valid.unsqueeze(1), logits, torch.zeros_like(logits))
        weights = F.softmax(logits, dim=1)
        weights = torch.where(any_valid.unsqueeze(1), weights, torch.zeros_like(weights))
        object_summary = torch.einsum("bk,bkd->bd", weights, relational_code)

        usage_slots = self.usage_bank.SlotReadout(identity, None, presence)
        usage_summary = torch.einsum("bk,bkd->bd", weights, usage_slots)
        usage_summary = self.usage_memory_proj(usage_summary)
        attended_anchor = torch.tanh(self.object_attention_anchor(attended))
        embodied = self.embodied_memory_expert(
            torch.cat([object_summary, usage_summary], dim=-1))
        return torch.where(
            any_valid.unsqueeze(1),
            embodied * attended_anchor,
            torch.zeros_like(embodied))

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
        sourceLabel: Optional[torch.Tensor] = None,
        writeMask: Optional[torch.Tensor] = None,
        lossSampleMask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.training:
            self.ResetInternalLoss()

        B = x.size(0)

        self.EnsureB(B)
        write_mask = self.ValidateWriteMask(writeMask, B, x.device)
        loss_sample_mask = self.ValidateLossSampleMask(
            lossSampleMask, B, x.device)

        src_all = sourceLabel
        if sourceLabel is None:
            src_all = torch.full((B,), MemoryType.SRC_REAL, dtype=torch.int8, device=self.device) # [B]

        emotion_eff = emotion
        tdError_eff = tdError
        reward_eff = reward
        uncertainty_eff = uncertainty
        risk_eff = risk
        confidence_eff = confidence
        source_reliability = confidence_eff

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

        attended_x = x

        self.FlushPendingWrites()
        frozen_state = self.CaptureFrozenRows(
            write_mask,
            self.FORWARD_MUTABLE_ROW_STATE_FIELDS)
        next_pending = []

        ontology = self.BuildOntologyObjectContext(
            attended_x,
            visualState)
        entity_text = self.BuildEntityTextContext(
            visualState,
            ontology["weights"])
        ontology["text_change"] = entity_text["change"]
        focused_realm = ontology["realm_id"]

        x = self.FuseExternalContext(
            attended_x,
            visualState=visualState,
            ocrSemantic=ocrSemantic,
            intentHint=intentHint)

        self.time_step.add_(1)

        self.gws.StepTick()
        self.ltm.StepTick()
        self.sym_mem.StepTick()

        h_prev = self.h_state.detach() # [B, ssmStateDim]

        h_mix = self.UpdateWorkingState(x, h_prev) # [B, ssmStateDim]
        h_new = h_mix

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
            confidence=confidence_eff,
            ontologyContext=ontology["context"],
            entityTextContext=entity_text["context"],
            writeMask=write_mask)
        _, event_boundary = self.DetectEventBoundary(
            attended_x,
            visualState,
            intentHint,
            td_memory,
            risk_eff,
            uncertainty_eff,
            confidence_eff,
            ontology,
            lossSampleMask=loss_sample_mask)

        emo_emb = self.emo_write_proj(emotion_eff) # [B, memoryDim]

        mod = self.emo_val_mod(emo_emb) # [B, 2*memoryDim]
        gamma, beta = mod.chunk(2, dim=-1) # [B, memoryDim]
        content_scale = 0.10 * torch.sigmoid(self.emotion_content_scale)
        gamma = content_scale * torch.tanh(gamma)
        beta = content_scale * torch.tanh(beta)
        val = (1.0 + gamma) * val + beta
        alpha = content_scale * torch.tanh(self.emo_write_alpha)
        val = val + alpha * torch.tanh(emo_emb) # [B, memoryDim]

        sem_context = torch.cat([
            visualState.IntegratedFeat,
            ocrSemantic,
            intentHint], dim=-1)

        sem_mod = self.sem_context_mod(sem_context)
        sem_scale, sem_shift = sem_mod.chunk(2, dim=-1)
        sem_mod_signal = sem_shift + 0.1 * torch.tanh(sem_scale)

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
        epi_mod_signal = epi_shift + 0.1 * torch.tanh(epi_scale)

        value_anchor = F.layer_norm(val, (self.memory_dim,))
        sem_context_gain = 0.10 * (
            torch.tanh(sem_scale) + torch.tanh(sem_shift))
        epi_context_gain = 0.10 * (
            torch.tanh(epi_scale) + torch.tanh(epi_shift))
        sem_in = val + sem_context_gain * value_anchor # [B, memoryDim]
        epi_in = val + epi_context_gain * value_anchor # [B, memoryDim]

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

        mem_recall, mem_evidence = self.Retrieve(
            key,
            fusion_gate,
            importance=importance_eff,
            localGate=gate_local,
            emotion=emotion_eff,
            tdError=td_memory,
            returnEvidence=True,) # [B, memoryDim], [B, 1]
        self.HebbianUpdate(
            key,
            val,
            gate_local,
            td_memory.abs(),
            a,
            b,
            writeMask=write_mask)

        mem_film_ctx = self.BuildFilmContext(val, mem_recall, emo_emb, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        g2, b2 = self.FilmParams(self.film_mem, mem_film_ctx)
        s2 = 1.0 + g2

        mem_state = self.mem_film_norm(mem_recall * s2 + b2) * mem_evidence

        durable_write_mask = event_boundary & write_mask
        next_pending.append(("kv",
                             (key.detach(),val.detach(),importance_eff.detach(),emotion_eff.detach(),reward_abs_eff.detach(),src_all.detach(),source_reliability.detach(),focused_realm.detach(),durable_write_mask.detach(),write_mask.detach())))

        gws_val, gws_mod_signal = self.BuildGwsValue(h_new, y_ssm, val, td_memory, risk_eff, confidence_eff) # [B, memoryDim]

        gws_recall, gws_evidence = self.gws.Attend(
            key,
            topk=1,
            returnEvidence=True) # [B, memoryDim], [B, 1]

        affect_mag = td_memory.abs()
        prio = importance_eff * (1.0 + 0.5 * affect_mag + 0.25 * risk_eff).clamp(0.5, 2.0)

        gws_film_ctx = self.BuildFilmContext(gws_val, gws_recall, gws_mod_signal, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        g1, b1 = self.FilmParams(self.film_gws, gws_film_ctx)
        s1 = 1.0 + g1

        gws_state = self.gws_film_norm(
            gws_recall * s1 + b1) * gws_evidence # [B, memoryDim]

        ttl = torch.full((B,), 6, device=self.device, dtype=torch.long)
        ttl = torch.where(src_all == MemoryType.SRC_REAL, torch.full_like(ttl, 10), ttl)
        ttl = torch.where(src_all == MemoryType.SRC_IMAGINE, torch.full_like(ttl, 4), ttl)

        next_pending.append(("gws",
                             (key.detach(),gws_val.detach(),prio.detach(),ttl.detach(),src_all.detach(),source_reliability.detach(),durable_write_mask.detach(),write_mask.detach())))


        (
            sem_recall,
            epi_event_recall,
            sem_evidence,
            epi_event_evidence,
        ) = self.ltm.Retrieve(
            key,
            topkSem=self.ltm_topk_sem,
            topkEpi=self.ltm_topk_epi,
            epiQuery=event_key,
            returnEvidence=True) # [B, memoryDim], [B, 1]
        epi_dense_recall, epi_dense_evidence = self.ltm.episodic.Retrieve(
            key,
            topk=self.ltm_topk_epi,
            useStateKey=True,
            recordAccess=False,
            returnEvidence=True)
        epi_sequence_recall, epi_sequence_evidence = (
            self.ltm.RetrieveEpisodeSequence(
                event_key,
                returnEvidence=True))
        completion_weight = F.softmax(
            self.event_completion_gate(torch.cat([
                epi_event_recall,
                epi_dense_recall,
                epi_sequence_recall,
                event_dense,
                td_memory.view(B, 1)], dim=-1)),
            dim=-1)
        epi_recall = (
            completion_weight[:, 0:1] * epi_event_recall
            + completion_weight[:, 1:2] * epi_dense_recall
            + completion_weight[:, 2:3] * epi_sequence_recall)

        sem_film_ctx = self.BuildFilmContext(sem_in, sem_recall, sem_mod_signal, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        epi_film_ctx = self.BuildFilmContext(epi_in, epi_recall, epi_mod_signal, td_memory, risk_eff, uncertainty_eff, confidence_eff)
        g3, b3 = self.FilmParams(self.film_sem, sem_film_ctx)
        g4, b4 = self.FilmParams(self.film_epi, epi_film_ctx)

        s3 = 1.0 + g3
        s4 = 1.0 + g4

        epi_evidence = 1.0 - (
            (1.0 - epi_event_evidence)
            * (1.0 - epi_dense_evidence)
            * (1.0 - epi_sequence_evidence))
        sem_state = self.sem_film_norm(
            sem_recall * s3 + b3) * sem_evidence # [B, memoryDim]
        epi_state = self.epi_film_norm(
            epi_recall * s4 + b4) * epi_evidence # [B, memoryDim]

        next_pending.append(("ltm",
                              (key.detach(),event_key.detach(),key.detach(),
                               sem_in.detach(),epi_in.detach(),
                               importance_eff.detach(),td_memory.detach(),reward_eff.detach(),src_all.detach(),
                               uncertainty_eff.detach(),risk_eff.detach(),confidence_eff.detach(),
                               source_reliability.detach(),focused_realm.detach(),durable_write_mask.detach(),self.episode_id.detach().clone(),self.event_id.detach().clone(),
                               {
                                   name: value.detach()
                                   for name, value in ontology[
                                       "episodic_metadata"].items()},
                               write_mask.detach())))

        ltm_fused = self.ltm.fuser(sem_state, epi_state)
        ltm_evidence = 1.0 - (
            (1.0 - sem_evidence) * (1.0 - epi_evidence))
        ltm_fused = ltm_fused * ltm_evidence

        P_post = self.NsPostRead(
            val,
            lossSampleMask=loss_sample_mask)

        Qsym_key = self.sym_query(key) # [B, nsK]
        qsym_mix = self.sym_query_fusion(torch.cat([Qsym_key, P_post], dim=-1)) # [B, nsK]
        Qsym = qsym_mix * Qsym_key + (1.0 - qsym_mix) * P_post
        Qsym = F.normalize(Qsym, dim=-1)

        symbolic_verified = (
            SourceProbabilityReal(src_all, source_reliability) >= 0.55)
        next_pending.append(("ns",
                             (Qsym.detach(), P_post.detach(), importance_eff.detach(), src_all.detach(),source_reliability.detach(),(durable_write_mask & symbolic_verified).detach(),write_mask.detach())))

        sym_recall = self.sym_mem.Retrieve(Qsym, topK=8)

        sym_vec = self.sym_embed(P_post, sym_recall) # [B, memoryDim]
        embodied_memory = self.BuildEmbodiedMemory(
            attended_x,
            visualState,
            ontology["object_features"],
            entity_text["object_code"])

        fused_state = self.fusion(
            torch.cat([mem_state, gws_state, ltm_fused, sym_vec], dim=-1),
            sampleMask=loss_sample_mask) # [B, outputDim]
        embodied_output = self.embodied_output_proj(embodied_memory)
        embodied_gate = self.embodied_memory_gate(
            torch.cat([fused_state, embodied_output], dim=-1))
        fused_state = fused_state + embodied_gate * embodied_output

        fused_state = self.ApplyOutputGate(fused_state, td_memory, gate_bias)

        if self.training:
            self.AddInternalLoss(self.fusion.GetAuxLoss())

        step0 = int(self.time_step[0].item())

        if bool(write_mask.any().item()):
            if (step0 % self.compress_every) == 0:
                next_pending.append(("compress", write_mask.detach()))
            if (step0 % max(1, self.compress_every // 4)) == 0:
                next_pending.append(("consolidate", write_mask.detach()))

        self.pending = next_pending
        self.RestoreFrozenRows(frozen_state, write_mask)

        return fused_state


    def ApplyOutputGate(self, memRecall: torch.Tensor, tdError: torch.Tensor, gateBias: torch.Tensor) -> torch.Tensor:
        gate_delta = torch.tanh(tdError + gateBias).view(-1, 1)
        return memRecall + 0.3 * gate_delta * memRecall


    @torch.no_grad()
    def HebbianUpdate(
        self,
        key: torch.Tensor, # [B, memory_dim]
        value: torch.Tensor, # [B, memory_dim]
        gateLocal: torch.Tensor, # [B]
        surprise: torch.Tensor, # [B]
        a: torch.Tensor, # [B]
        b: torch.Tensor, # [B]
        writeMask: Optional[torch.Tensor] = None,
        ) -> None:

        B = int(key.size(0))

        key_d = key.detach()
        value_d = value.detach()

        a3 = a.view(B, 1, 1)
        b3 = b.view(B, 1, 1)
        g3 = gateLocal.view(B, 1, 1)
        learning_rate = (
            0.01 + 0.14 * torch.tanh(surprise.detach().abs())
        ).view(B, 1, 1)

        decayed = self.fast_weights * 0.95 * b3
        prediction = torch.bmm(key_d.unsqueeze(1), decayed).squeeze(1)
        error = value_d - prediction
        update = learning_rate * a3 * g3 * torch.bmm(
            key_d.unsqueeze(2), error.unsqueeze(1))
        new_weights = decayed + update

        max_fro = math.sqrt(float(self.memory_dim))
        flat = new_weights.reshape(B, -1)
        fro = torch.linalg.vector_norm(flat, ord=2, dim=1)
        scale = (max_fro / (fro + 1e-12)).clamp(max=1.0)

        updated_weights = new_weights * scale.view(B, 1, 1)
        if writeMask is None:
            self.fast_weights = updated_weights
        else:
            self.fast_weights = torch.where(
                writeMask.view(B, 1, 1),
                updated_weights,
                self.fast_weights)

    @torch.no_grad()
    def KvWrite(
        self,
        key: torch.Tensor, # [B, memory_dim]
        val: torch.Tensor, # [B, memory_dim]
        importance: torch.Tensor, # [B]
        emotion: torch.Tensor, # [B, emotion_dim]
        rewardAbs: torch.Tensor, # [B]
        source: torch.Tensor, # [B]
        sourceConfidence: torch.Tensor, # [B]
        writeMask: torch.Tensor, # [B]
        *,
        realm: torch.Tensor,
        ) -> None:
        B = int(key.size(0))
        M = int(self.memory_size)
        wrote = False
        for b in range(B):
            if not bool(writeMask[b].item()):
                continue
            n = int(self.memory_filled[b].item())
            merge_index = -1
            if n > 0:
                key_similarity = torch.mv(self.memory_keys[b, :n], key[b])
                key_similarity = key_similarity.masked_fill(
                    self.memory_realm[b, :n] != realm[b],
                    -torch.inf)
                best_similarity, best_index = key_similarity.max(dim=0)
                best_value_similarity = F.cosine_similarity(
                    self.memory_values[b, best_index].unsqueeze(0),
                    val[b].unsqueeze(0),
                    dim=-1).squeeze(0)
                if (
                    float(best_similarity.item()) >= 0.97
                    and float(best_value_similarity.item()) >= 0.90
                ):
                    merge_index = int(best_index.item())

            if merge_index >= 0:
                idx = merge_index
                old_weight = self.memory_merge_count[b, idx].float() + 1.0
                new_weight = old_weight + 1.0
                self.memory_keys[b, idx] = F.normalize(
                    (old_weight * self.memory_keys[b, idx] + key[b]) / new_weight,
                    dim=0)
                self.memory_values[b, idx] = (
                    old_weight * self.memory_values[b, idx] + val[b]) / new_weight
                self.memory_emotion[b, idx] = (
                    old_weight * self.memory_emotion[b, idx] + emotion[b]) / new_weight
                self.memory_importance[b, idx] = torch.maximum(
                    self.memory_importance[b, idx], importance[b])
                self.memory_reward_abs[b, idx] = torch.maximum(
                    self.memory_reward_abs[b, idx], rewardAbs[b])
                if int(self.memory_source[b, idx].item()) != int(source[b].item()):
                    self.memory_source[b, idx] = MemoryType.SRC_MIXED
                self.memory_source_confidence[b, idx] = (
                    old_weight * self.memory_source_confidence[b, idx]
                    + sourceConfidence[b]) / new_weight
                self.memory_last_rehearsal_steps[b, idx] = self.time_step[b]
                self.memory_touch[b, idx] += 1
                self.memory_merge_count[b, idx] += 1
                wrote = True
                continue

            if n < M:
                idx = n
                self.memory_filled[b] += 1
            else:
                age = (self.time_step[b] - self.memory_steps[b]).clamp(min=0).float()
                retention = (
                    -0.01 * age
                    + 0.50 * torch.log(self.memory_importance[b] + 1e-6)
                    + 0.10 * torch.log1p(self.memory_touch[b].float())
                    + 0.50 * torch.log(self.memory_source_confidence[b] + 1e-6)
                    + 0.50 * torch.tanh(self.memory_reward_abs[b]))
                idx = int(torch.argmin(retention).item())

            self.memory_keys[b, idx] = key[b]
            self.memory_values[b, idx] = val[b]
            self.memory_importance[b, idx] = importance[b]
            self.memory_steps[b, idx] = self.time_step[b]
            self.memory_last_access_steps[b, idx] = self.time_step[b]
            self.memory_last_rehearsal_steps[b, idx] = self.time_step[b]
            self.memory_touch[b, idx] = 1
            self.memory_merge_count[b, idx] = 0
            self.memory_emotion[b, idx] = emotion[b]
            self.memory_reward_abs[b, idx] = rewardAbs[b]
            self.memory_source[b, idx] = source[b]
            self.memory_source_confidence[b, idx] = sourceConfidence[b]
            self.memory_realm[b, idx] = realm[b]
            wrote = True
        if wrote:
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
        realm: torch.Tensor,
        episodicMetadata: Dict[str, torch.Tensor],
        uncertainty: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        sourceConfidence: torch.Tensor,
        eventMask: torch.Tensor,
        episodeId: torch.Tensor,
        eventId: torch.Tensor):
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
        mask = (mask_real | mask_mixed | mask_imag) & eventMask

        if not mask.any():
            return

        sem = self.ltm.semantic
        epi = self.ltm.episodic

        epi.VerifyWithRealEvidence(
            keyEpi,
            (sourceLabel == MemoryType.SRC_REAL) & eventMask,
            realm)

        probability_real = SourceProbabilityReal(sourceLabel, sourceConfidence)
        semantic_mask = (
            mask
            & (probability_real >= 0.55)
            & (realm != ONTOLOGY_REALM_EFFECT))
        sem.Store(
            key=keySem,
            value=valSem,
            score=salience,
            realm=realm,
            source=sourceLabel,
            writeMask=semantic_mask,
            sourceConfidence=sourceConfidence)
        epi.Store(
            key=keyEpi,
            value=valEpi,
            reward=reward,
            score=salience,
            realm=realm,
            source=sourceLabel,
            writeMask=mask,
            stateKey=keyEpiState,
            sourceConfidence=sourceConfidence,
            episodeId=episodeId,
            eventId=eventId,
            typedMetadata=episodicMetadata)
        self.memory_version.add_(1)

    def Retrieve(
        self,
        query: torch.Tensor, # [B, memory_dim]
        fusionGate: torch.Tensor, # [B]
        importance: torch.Tensor, # [B]
        localGate: torch.Tensor, # [B]
        emotion: torch.Tensor, # [B, emotion_dim]
        tdError: torch.Tensor, # [B]
        returnEvidence: bool = False,
        ):

        B = int(query.size(0))
        M = int(self.memory_size)
        D = int(self.memory_dim)

        fw = self.fast_weights.detach() # [B, memory_dim, memory_dim]
        qf = query
        fast_part = torch.bmm(qf.unsqueeze(1), fw).squeeze(1) # [B, memory_dim]

        filled = self.memory_filled
        any_valid = filled > 0

        feat_imp = importance.view(B, 1)
        feat_local = localGate.view(B, 1)
        feat_fuse = fusionGate.view(B, 1)
        feat_td = tdError.abs().view(B, 1)
        memory_evidence_gate = torch.zeros(
            B, 1, device=query.device, dtype=query.dtype)

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
                touch=self.memory_touch,
                confidence=self.memory_source_confidence,
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

            candidate_valid = torch.gather(valid_mask, 1, top_idx)
            attn, accepted, memory_evidence_gate = NullGatedTopKWeights(
                top_sim_eff,
                self.kv_null_logit,
                candidate_valid)
            attn = torch.where(any_valid.view(B,1), attn, torch.zeros_like(attn))
            memory_evidence_gate = torch.where(
                any_valid.view(B, 1),
                memory_evidence_gate,
                torch.zeros_like(memory_evidence_gate))
            evidence_mass = attn.sum(dim=-1, keepdim=True)
            idx3 = top_idx.unsqueeze(-1).expand(B, k, D)
            vals = torch.gather(values, 1, idx3) # [B, k, D]

            selected_source = torch.gather(src, 1, top_idx)
            selected_reliability = torch.gather(
                self.memory_source_confidence,
                1,
                top_idx)
            p_real = SourceProbabilityReal(
                selected_source,
                selected_reliability)
            lambda_imag = torch.sigmoid(0.5 * feat_td - 0.3)
            mem_task, _, _ = SourceBalancedRecall(
                attn,
                vals,
                p_real,
                lambda_imag)

            emo_vals = self.memory_emotion # [B, M, emotion_dim]
            emo_sel = torch.gather(emo_vals, 1, top_idx.unsqueeze(-1).expand(B, k, self.emotion_dim)) # [B, k, emotion_dim]
            emo_q = emotion.unsqueeze(1) # [B, 1, emotion_dim]
            emo_sim = F.cosine_similarity(emo_sel, emo_q.expand_as(emo_sel), dim=-1) # [B, k]
            emo_logits = emo_sim.float().masked_fill(~accepted, -torch.inf)
            emo_w = F.softmax(torch.cat([
                emo_logits,
                torch.zeros(B, 1, device=emo_logits.device, dtype=emo_logits.dtype)],
                dim=-1), dim=-1)[:, :k] # [B, k]
            mem_affect = torch.einsum("bk,bkd->bd", emo_w, vals) # [B, D]

            emo_vec = torch.einsum("bk,bke->be", emo_w, emo_sel) # [B, emotion_dim]
            emo_embed = self.emo_write_proj(emo_vec) # [B, memory_dim]
            emo_gate = self.emo_content_gate(torch.cat([mem_affect, emo_embed], dim=-1)) # [B, 1]
            mem_affect = (1.0 - emo_gate) * mem_affect + emo_gate * emo_embed # [B, D]

            gate_feat = torch.cat([feat_td, emotion], dim=-1) # [B, 1 + emotion_dim]
            gamma = self.td_affect_gate(gate_feat) # [B, 1]
            kv_part = (
                gamma * mem_task + (1.0 - gamma) * mem_affect
            ) * evidence_mass # [B, D]
            kv_part = torch.where(any_valid.view(B, 1), kv_part, torch.zeros_like(kv_part))

            with torch.no_grad():
                b_idx = torch.arange(B, device=query.device).unsqueeze(1).expand_as(top_idx)
                self.memory_touch[b_idx[accepted], top_idx[accepted]] += 1
                self.memory_last_access_steps[b_idx[accepted], top_idx[accepted]] = (
                    self.time_step.unsqueeze(1).expand_as(top_idx)[accepted])

        fusion_input = torch.cat([query, fast_part, kv_part, feat_imp, feat_local, feat_fuse, feat_td], dim=-1) # [B, 3*D + 4]
        gate = self.fusion_gate_net(fusion_input)

        base_out = gate * fast_part + (1.0 - gate) * kv_part

        concat_feat = torch.cat([fast_part, kv_part], dim=-1)
        refine_out = self.output_refine(concat_feat)

        out = base_out + refine_out
        has_fast = (fast_part.abs().sum(dim=1, keepdim=True) > 0).to(query.dtype)
        evidence = 1.0 - (1.0 - memory_evidence_gate) * (1.0 - has_fast)
        out = out * evidence
        return (out, evidence) if returnEvidence else out # [B, D]


    @torch.no_grad()
    def AutoCompress(self):
        B = int(self.memory_filled.size(0))
        M = int(self.memory_size)
        slots = torch.arange(M, device=self.memory_keys.device).view(1, M)
        valid = slots < self.memory_filled.view(B, 1)
        if not bool(valid.any().item()):
            return
        age = (self.time_step.view(B, 1) - self.memory_steps).clamp(min=0).float()
        retention = (
            -0.01 * age
            + 0.50 * torch.log(self.memory_importance + 1e-6)
            + 0.10 * torch.log1p(self.memory_touch.float())
            + 0.50 * torch.log(self.memory_source_confidence + 1e-6)
            + 0.50 * torch.tanh(self.memory_reward_abs))
        retention = retention.masked_fill(~valid, -torch.inf)
        order = torch.argsort(retention, dim=1, descending=True)

        def reorder(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.dim() == 3:
                index = order.unsqueeze(-1).expand(B, M, tensor.size(-1))
            else:
                index = order
            return torch.gather(tensor, 1, index)

        self.memory_keys.copy_(reorder(self.memory_keys))
        self.memory_values.copy_(reorder(self.memory_values))
        self.memory_emotion.copy_(reorder(self.memory_emotion))
        self.memory_importance.copy_(reorder(self.memory_importance))
        self.memory_steps.copy_(reorder(self.memory_steps))
        self.memory_last_access_steps.copy_(reorder(self.memory_last_access_steps))
        self.memory_last_rehearsal_steps.copy_(reorder(self.memory_last_rehearsal_steps))
        self.memory_touch.copy_(reorder(self.memory_touch))
        self.memory_merge_count.copy_(reorder(self.memory_merge_count))
        self.memory_source.copy_(reorder(self.memory_source))
        self.memory_source_confidence.copy_(reorder(self.memory_source_confidence))
        self.memory_realm.copy_(reorder(self.memory_realm))
        self.memory_reward_abs.copy_(reorder(self.memory_reward_abs))
        self.last_compress_step.copy_(self.time_step)
        self.memory_version.add_(1)

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
        source_conf = epi.source_confidence
        freshness = torch.exp(-0.001 * age)
        salience = epi.prio * (1.0 + 0.5 * torch.tanh(epi.rew_abs.clamp_min(0.0)))
        salience = salience * (1.0 + 0.1 * torch.log1p(epi.touch.float().clamp_min(0.0)))
        salience = salience * (1.0 + 1.0 / (1.0 + epi.consolidation_count.float()))
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
        src_conf = torch.gather(epi.source_confidence, 1, top_idx)
        realm = torch.gather(epi.realm, 1, top_idx)
        score = top_score.to(dtype=dtype)
        probability_real = SourceProbabilityReal(src, src_conf)
        keep = (
            torch.isfinite(score)
            & (score > 0)
            & (probability_real >= 0.55)
            & (realm != ONTOLOGY_REALM_EFFECT))
        if not bool(keep.any().item()):
            return

        for j in range(k):
            sem.Store(
                key=keys[:, j],
                value=vals[:, j],
                score=score[:, j],
                realm=realm[:, j],
                source=src[:, j],
                writeMask=keep[:, j],
                sourceConfidence=src_conf[:, j],
                consolidated=True)

        selected_valid = torch.gather(valid, 1, top_idx) & keep
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(top_idx)
        epi.consolidation_count[b_idx[selected_valid], top_idx[selected_valid]] += 1
        epi.last_rehearsal_step[b_idx[selected_valid], top_idx[selected_valid]] = (
            epi.global_step.unsqueeze(1).expand_as(top_idx)[selected_valid])

        sym_keep = keep & (src != MemoryType.SRC_IMAGINE)
        if bool(sym_keep.any().item()):
            for j in range(k):
                P = self.ns_coder_post(vals[:, j])
                self.sym_mem.Store(
                    key=F.normalize(P.detach(), dim=-1),
                    value=P.detach(),
                    score=score[:, j],
                    source=src[:, j],
                    sourceConfidence=src_conf[:, j],
                    writeMask=sym_keep[:, j])
        self.memory_version.add_(1)


    def FlushPendingWrites(self):
        while self.pending:
            kind, payload = self.pending[0]
            if kind == "gws":
                if len(payload) == 8:
                    key, ws_val, prio, ttl, src, src_conf, write_mask, hard_mask = payload
                    write_mask = write_mask & hard_mask
                else:
                    key, ws_val, prio, ttl, src, src_conf, write_mask = payload
                self.gws.Write(
                    key,
                    ws_val,
                    priority=prio,
                    ttl=ttl,
                    tagId=src,
                    sourceConfidence=src_conf,
                    writeMask=write_mask)

            elif kind == "kv":
                if len(payload) == 10:
                    key, val, imp, emo, rew_abs, src, src_conf, realm, write_mask, hard_mask = payload
                    write_mask = write_mask & hard_mask
                else:
                    key, val, imp, emo, rew_abs, src, src_conf, realm, write_mask = payload
                self.KvWrite(
                    key=key,
                    val=val,
                    importance=imp,
                    emotion=emo,
                    rewardAbs=rew_abs,
                    source=src,
                    sourceConfidence=src_conf,
                    realm=realm,
                    writeMask=write_mask)

            elif kind == "ltm":
                if len(payload) == 19:
                    key_sem, key_epi, key_epi_state, sem, epi, imp, td, rwd, src, unc, risk, conf, src_conf, realm, event_mask, episode_id, event_id, episodic_metadata, hard_mask = payload
                    event_mask = event_mask & hard_mask
                else:
                    key_sem, key_epi, key_epi_state, sem, epi, imp, td, rwd, src, unc, risk, conf, src_conf, realm, event_mask, episode_id, event_id, episodic_metadata = payload
                self.LtmOnlineStore(
                    keySem=key_sem,
                    keyEpi=key_epi,
                    keyEpiState=key_epi_state,
                    valSem=sem,
                    valEpi=epi,
                    importance=imp,
                    tdError=td,
                    reward=rwd,
                    sourceLabel=src,
                    realm=realm,
                    episodicMetadata=episodic_metadata,
                    uncertainty=unc,
                    risk=risk,
                    confidence=conf,
                    sourceConfidence=src_conf,
                    eventMask=event_mask,
                    episodeId=episode_id,
                    eventId=event_id)

            elif kind == "ns":
                if len(payload) == 7:
                    key, P_post, importance, src, src_conf, write_mask, hard_mask = payload
                    write_mask = write_mask & hard_mask
                else:
                    key, P_post, importance, src, src_conf, write_mask = payload
                self.NsStore(key, P_post, importance, src, src_conf, write_mask)

            elif kind == "compress":
                frozen_state = self.CaptureFrozenRows(payload)
                self.AutoCompress()
                self.RestoreFrozenRows(frozen_state, payload)

            elif kind == "consolidate":
                frozen_state = self.CaptureFrozenRows(payload)
                self.ConsolidateMemory()
                self.RestoreFrozenRows(frozen_state, payload)

            self.pending.pop(0)

    @torch.no_grad()
    def SealPendingRows(self, done: torch.Tensor) -> None:
        sealed = []
        for kind, payload in self.pending:
            if kind == "kv":
                fields = list(payload)
                hard_mask = (
                    fields[9]
                    if len(fields) == 10
                    else torch.ones_like(done))
                terminal = done & hard_mask
                fields[2] = torch.where(
                    terminal,
                    torch.maximum(fields[2], torch.ones_like(fields[2])),
                    fields[2])
                fields[8] = (fields[8] | terminal) & hard_mask
                payload = tuple(fields)
            elif kind == "gws":
                fields = list(payload)
                hard_mask = (
                    fields[7]
                    if len(fields) == 8
                    else torch.ones_like(done))
                terminal = done & hard_mask
                fields[2] = torch.where(
                    terminal,
                    torch.maximum(fields[2], torch.ones_like(fields[2])),
                    fields[2])
                fields[6] = (fields[6] | terminal) & hard_mask
                payload = tuple(fields)
            elif kind == "ltm":
                fields = list(payload)
                hard_mask = (
                    fields[18]
                    if len(fields) == 19
                    else torch.ones_like(done))
                terminal = done & hard_mask
                fields[5] = torch.where(
                    terminal,
                    torch.maximum(fields[5], torch.full_like(fields[5], 1.5)),
                    fields[5])
                already_an_event = fields[14]
                allocate_terminal_event = terminal & ~already_an_event
                next_event_id = torch.maximum(
                    fields[16],
                    self.event_id) + 1
                fields[16] = torch.where(
                    allocate_terminal_event,
                    next_event_id,
                    fields[16])
                updated_event_id = torch.where(
                    allocate_terminal_event,
                    next_event_id,
                    torch.maximum(self.event_id, fields[16]))
                self.event_id.copy_(torch.where(
                    hard_mask,
                    updated_event_id,
                    self.event_id))
                fields[14] = (fields[14] | terminal) & hard_mask
                payload = tuple(fields)
            elif kind == "ns":
                fields = list(payload)
                hard_mask = (
                    fields[6]
                    if len(fields) == 7
                    else torch.ones_like(done))
                terminal = done & hard_mask
                fields[2] = torch.where(
                    terminal,
                    torch.maximum(fields[2], torch.ones_like(fields[2])),
                    fields[2])
                fields[5] = fields[5] | (
                    terminal
                    & (SourceProbabilityReal(fields[3], fields[4]) >= 0.55))
                fields[5] = fields[5] & hard_mask
                payload = tuple(fields)
            sealed.append((kind, payload))
        self.pending = sealed

    @torch.no_grad()
    def SoftReset(self):
        done = torch.ones(
            self.h_state.size(0),
            device=self.h_state.device,
            dtype=torch.bool)
        self.h_state.zero_()
        self.fast_weights.zero_()

        self.memory_filled.zero_()


        self.last_compress_step.copy_(self.time_step)
        self.memory_version.add_(1)

        self.ns_prev_P_post.zero_()
        self.ns_penalty_vec.zero_()
        self.pattern_usage.zero_()
        self.previous_attention.zero_()
        self.previous_intent.zero_()
        self.previous_object_summary.zero_()
        self.previous_motion_token.zero_()
        self.previous_ontology_context.zero_()
        self.event_age.zero_()
        self.event_id.zero_()
        next_episode = self.ltm.episodic.StartNewEpisode(done)
        self.episode_id.copy_(next_episode)
        self.gws.ResetRows(done)
        self.has_previous_event.zero_()
        self.ResetInternalLoss()
        self.pending.clear()
        self._plan_cache.clear()

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
        self.SealPendingRows(done)
        self.FlushPendingWrites()
        self.h_state[done] = 0
        self.fast_weights[done] = 0
        self.ns_prev_P_post[done] = 0
        self.ns_penalty_vec[done] = 0
        self.previous_attention[done] = 0
        self.previous_intent[done] = 0
        self.previous_object_summary[done] = 0
        self.previous_motion_token[done] = 0
        self.previous_ontology_context[done] = 0
        self.event_age[done] = 0
        self.event_id[done] = 0
        self.has_previous_event[done] = False
        self.gws.ResetRows(done)
        next_episode = self.ltm.episodic.StartNewEpisode(done)
        self.episode_id[done] = next_episode[done]
        if bool(done.any().item()):
            invalidated = []
            for plan_id, cached in self._plan_cache.items():
                valid = cached.get("valid")
                feature = cached.get("feature")
                age = cached.get("age")
                if (
                    not torch.is_tensor(valid)
                    or not torch.is_tensor(feature)
                    or not torch.is_tensor(age)
                    or tuple(valid.shape) != tuple(done.shape)
                    or tuple(age.shape) != tuple(done.shape)
                    or valid.device != done.device
                    or age.device != done.device
                    or age.dtype != torch.long
                    or int(feature.size(0)) != int(done.numel())
                ):
                    invalidated.append(plan_id)
                    continue
                valid = valid.clone()
                feature = feature.clone()
                age = age.clone()
                valid[done] = False
                feature[done] = 0
                age[done] = 0
                cached["valid"] = valid
                cached["feature"] = feature
                cached["age"] = age
                if not bool(valid.any().item()):
                    invalidated.append(plan_id)
            for plan_id in invalidated:
                self._plan_cache.pop(plan_id, None)


    def ResetAll(self):
        self.h_state.zero_()
        self.fast_weights.zero_()

        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_importance.zero_()
        self.memory_steps.zero_()
        self.memory_last_access_steps.zero_()
        self.memory_last_rehearsal_steps.zero_()
        self.memory_touch.zero_()
        self.memory_merge_count.zero_()
        self.memory_emotion.zero_()
        self.memory_source.zero_()
        self.memory_source_confidence.zero_()
        self.memory_realm.fill_(ONTOLOGY_REALM_UNKNOWN)
        self.memory_reward_abs.zero_()
        self.merged_delta_signature.fill_(-1)

        self.time_step.zero_()
        self.memory_filled.zero_()
        self.last_compress_step.zero_()

        self.gws.Reset()
        self.ltm.Reset()
        self.sym_mem.Reset()

        self.ns_prev_P_post.zero_()
        self.ns_penalty_vec.zero_()
        self.pattern_usage.zero_()
        self.previous_attention.zero_()
        self.previous_intent.zero_()
        self.previous_object_summary.zero_()
        self.previous_motion_token.zero_()
        self.previous_ontology_context.zero_()
        self.event_age.zero_()
        self.event_id.zero_()
        self.episode_id.zero_()
        self.has_previous_event.zero_()
        self.ResetInternalLoss()
        self.pending.clear()
        self._plan_cache.clear()
        self._skill_cache.clear()
        self._offline_replay.clear()
        self._replay_signature = None
        self._replay_sequence = 0
        self._replay_transaction_version = 0
        self._replay_timeline_version = 0
        self._row_merge_contract_id = None
        self._row_merge_model_signature = None
        self._row_merge_transactions.clear()
        self._row_merge_versions.clear()
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
    def SaveState(self, path: str):
        self.FlushPendingWrites()
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
            "merged_delta_signature": self.merged_delta_signature,
            "memory_keys": self.memory_keys,
            "memory_values": self.memory_values,
            "memory_importance": self.memory_importance,
            "memory_steps": self.memory_steps,
            "memory_last_access_steps": self.memory_last_access_steps,
            "memory_last_rehearsal_steps": self.memory_last_rehearsal_steps,
            "memory_touch": self.memory_touch,
            "memory_merge_count": self.memory_merge_count,
            "memory_emotion": self.memory_emotion,
            "memory_source": self.memory_source,
            "memory_source_confidence": self.memory_source_confidence,
            "memory_realm": self.memory_realm,
            "memory_reward_abs": self.memory_reward_abs,
            "episode_id": self.episode_id,
            "event_id": self.event_id,
            "ltm_sem_global_step": sem.global_step,
            "ltm_sem_keys": sem.keys,
            "ltm_sem_vals": sem.vals,
            "ltm_sem_prio": sem.prio,
            "ltm_sem_touch": sem.touch,
            "ltm_sem_step": sem.step,
            "ltm_sem_last_access_step": sem.last_access_step,
            "ltm_sem_last_rehearsal_step": sem.last_rehearsal_step,
            "ltm_sem_consolidation_count": sem.consolidation_count,
            "ltm_sem_prototype_count": sem.prototype_count,
            "ltm_sem_prototype_variance": sem.prototype_variance,
            "ltm_sem_filled": sem.filled,
            "ltm_sem_source": sem.source,
            "ltm_sem_source_confidence": sem.source_confidence,
            "ltm_sem_realm": sem.realm,
            "ltm_epi_global_step": epi.global_step,
            "ltm_epi_keys": epi.keys,
            "ltm_epi_state_keys": epi.state_keys,
            "ltm_epi_vals": epi.vals,
            "ltm_epi_prio": epi.prio,
            "ltm_epi_rew": epi.rew,
            "ltm_epi_rew_abs": epi.rew_abs,
            "ltm_epi_step": epi.step,
            "ltm_epi_last_access_step": epi.last_access_step,
            "ltm_epi_last_rehearsal_step": epi.last_rehearsal_step,
            "ltm_epi_consolidation_count": epi.consolidation_count,
            "ltm_epi_touch": epi.touch,
            "ltm_epi_filled": epi.filled,
            "ltm_epi_source": epi.source,
            "ltm_epi_source_confidence": epi.source_confidence,
            "ltm_epi_realm": epi.realm,
            "ltm_epi_agency_id": epi.agency_id,
            "ltm_epi_motion_layer_prob": epi.motion_layer_prob,
            "ltm_epi_observed_surface_parent_index": (
                epi.observed_surface_parent_index),
            "ltm_epi_surface_uv": epi.surface_uv,
            "ltm_epi_surface_uv_confidence": epi.surface_uv_confidence,
            "ltm_epi_verification": epi.verification,
            "ltm_epi_content_change": epi.content_change,
            "ltm_epi_episode_id": epi.episode_id,
            "ltm_epi_event_id": epi.event_id,
            "ltm_epi_prev_index": epi.prev_index,
            "ltm_epi_next_index": epi.next_index,
            "ltm_epi_slot_generation": epi.slot_generation,
            "ltm_epi_prev_generation": epi.prev_generation,
            "ltm_epi_next_generation": epi.next_generation,
            "ltm_epi_last_event_index": epi.last_event_index,
            "ltm_epi_current_episode_id": epi.current_episode_id,
            "sym_mem_global_step": sym.global_step,
            "sym_mem_P_keys": sym.P_keys,
            "sym_mem_P_vals": sym.P_vals,
            "sym_mem_prio": sym.prio,
            "sym_mem_step": sym.step,
            "sym_mem_last_access_step": sym.last_access_step,
            "sym_mem_last_rehearsal_step": sym.last_rehearsal_step,
            "sym_mem_touch": sym.touch,
            "sym_mem_filled": sym.filled,
            "sym_mem_source": sym.source,
            "sym_mem_source_confidence": sym.source_confidence,
            "usage_applicable": usage.applicable,
            "usage_default_params": usage.default_params,
            "usage_expected_dx": usage.expected_dx,
            "usage_success_alpha": usage.success_alpha,
            "usage_success_beta": usage.success_beta,
            "usage_param_mu": usage.param_mu,
            "usage_param_logvar": usage.param_logvar,
            "usage_parameter_observations": usage.parameter_observations,
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
            "merged_delta_signature": (3,),
            "memory_keys": (batch_size, self.memory_size, self.memory_dim),
            "memory_values": (batch_size, self.memory_size, self.memory_dim),
            "memory_importance": (batch_size, self.memory_size),
            "memory_steps": (batch_size, self.memory_size),
            "memory_last_access_steps": (batch_size, self.memory_size),
            "memory_last_rehearsal_steps": (batch_size, self.memory_size),
            "memory_touch": (batch_size, self.memory_size),
            "memory_merge_count": (batch_size, self.memory_size),
            "memory_emotion": (batch_size, self.memory_size, self.emotion_dim),
            "memory_source": (batch_size, self.memory_size),
            "memory_source_confidence": (batch_size, self.memory_size),
            "memory_realm": (batch_size, self.memory_size),
            "memory_reward_abs": (batch_size, self.memory_size),
            "episode_id": (batch_size,),
            "event_id": (batch_size,),
            "ltm_sem_global_step": (batch_size,),
            "ltm_sem_keys": (batch_size, self.ltm.semantic.capacity, self.memory_dim),
            "ltm_sem_vals": (batch_size, self.ltm.semantic.capacity, self.memory_dim),
            "ltm_sem_prio": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_touch": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_step": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_last_access_step": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_last_rehearsal_step": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_consolidation_count": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_prototype_count": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_prototype_variance": (batch_size, self.ltm.semantic.capacity, self.memory_dim),
            "ltm_sem_filled": (batch_size,),
            "ltm_sem_source": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_source_confidence": (batch_size, self.ltm.semantic.capacity),
            "ltm_sem_realm": (batch_size, self.ltm.semantic.capacity),
            "ltm_epi_global_step": (batch_size,),
            "ltm_epi_keys": (batch_size, self.ltm.episodic.capacity, self.memory_dim),
            "ltm_epi_state_keys": (batch_size, self.ltm.episodic.capacity, self.memory_dim),
            "ltm_epi_vals": (batch_size, self.ltm.episodic.capacity, self.memory_dim),
            "ltm_epi_prio": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_rew": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_rew_abs": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_step": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_last_access_step": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_last_rehearsal_step": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_consolidation_count": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_touch": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_filled": (batch_size,),
            "ltm_epi_source": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_source_confidence": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_realm": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_agency_id": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_motion_layer_prob": (
                batch_size, self.ltm.episodic.capacity, 5),
            "ltm_epi_observed_surface_parent_index": (
                batch_size, self.ltm.episodic.capacity),
            "ltm_epi_surface_uv": (
                batch_size, self.ltm.episodic.capacity, 2),
            "ltm_epi_surface_uv_confidence": (
                batch_size, self.ltm.episodic.capacity),
            "ltm_epi_verification": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_content_change": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_episode_id": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_event_id": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_prev_index": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_next_index": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_slot_generation": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_prev_generation": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_next_generation": (batch_size, self.ltm.episodic.capacity),
            "ltm_epi_last_event_index": (batch_size,),
            "ltm_epi_current_episode_id": (batch_size,),
            "sym_mem_global_step": (batch_size,),
            "sym_mem_P_keys": (batch_size, self.sym_mem.capacity, self.sym_mem.K),
            "sym_mem_P_vals": (batch_size, self.sym_mem.capacity, self.sym_mem.K),
            "sym_mem_prio": (batch_size, self.sym_mem.capacity),
            "sym_mem_step": (batch_size, self.sym_mem.capacity),
            "sym_mem_last_access_step": (batch_size, self.sym_mem.capacity),
            "sym_mem_last_rehearsal_step": (batch_size, self.sym_mem.capacity),
            "sym_mem_touch": (batch_size, self.sym_mem.capacity),
            "sym_mem_filled": (batch_size,),
            "sym_mem_source": (batch_size, self.sym_mem.capacity),
            "sym_mem_source_confidence": (batch_size, self.sym_mem.capacity),
            "usage_applicable": tuple(self.usage_bank.applicable.shape),
            "usage_default_params": tuple(self.usage_bank.default_params.shape),
            "usage_expected_dx": tuple(self.usage_bank.expected_dx.shape),
            "usage_success_alpha": tuple(self.usage_bank.success_alpha.shape),
            "usage_success_beta": tuple(self.usage_bank.success_beta.shape),
            "usage_param_mu": tuple(self.usage_bank.param_mu.shape),
            "usage_param_logvar": tuple(self.usage_bank.param_logvar.shape),
            "usage_parameter_observations": tuple(self.usage_bank.parameter_observations.shape),
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

        filled_ranges = (
            ("memory_filled", self.memory_size),
            ("ltm_sem_filled", self.ltm.semantic.capacity),
            ("ltm_epi_filled", self.ltm.episodic.capacity),
            ("sym_mem_filled", self.sym_mem.capacity),)
        for field, capacity in filled_ranges:
            filled = state[field]
            if bool(((filled < 0) | (filled > capacity)).any().item()):
                raise ValueError(
                    f"durable-memory field {field} is outside its capacity")

        realm_fields = (
            ("memory_realm", "memory_filled"),
            ("ltm_sem_realm", "ltm_sem_filled"),
            ("ltm_epi_realm", "ltm_epi_filled"),)
        for realm_field, filled_field in realm_fields:
            values = state[realm_field]
            slots = torch.arange(
                values.size(1),
                device=values.device).view(1, -1)
            valid = slots < state[filled_field].view(-1, 1)
            invalid_realm = (
                (values < ONTOLOGY_REALM_SELF)
                | (values > ONTOLOGY_REALM_UNKNOWN))
            if bool((invalid_realm & valid).any().item()):
                raise ValueError(
                    f"durable-memory field {realm_field} contains an "
                    "invalid entity realm")

        epi_slots = torch.arange(
            self.ltm.episodic.capacity,
            device=state["ltm_epi_filled"].device).view(1, -1)
        epi_valid = epi_slots < state["ltm_epi_filled"].view(-1, 1)
        agency_id = state["ltm_epi_agency_id"]
        if bool((epi_valid & (
            (agency_id < ONTOLOGY_AGENCY_SELF)
            | (agency_id > ONTOLOGY_AGENCY_UNKNOWN)
        )).any().item()):
            raise ValueError("durable episodic agency_id is invalid")
        observed_parent = state[
            "ltm_epi_observed_surface_parent_index"]
        if bool((epi_valid & (observed_parent < -1)).any().item()):
            raise ValueError(
                "durable episodic observed surface-parent index is invalid")
        for field in (
            "ltm_epi_motion_layer_prob",
            "ltm_epi_surface_uv_confidence",
            "ltm_epi_verification",
            "ltm_epi_content_change",):
            value = state[field]
            valid = epi_valid.unsqueeze(-1) if value.dim() == 3 else epi_valid
            if bool((valid & (
                ~torch.isfinite(value)
                | (value < 0.0)
                | (value > 1.0)
            )).any().item()):
                raise ValueError(
                    f"durable episodic field {field} is not a probability")
        if bool((
            epi_valid.unsqueeze(-1)
            & ~torch.isfinite(state["ltm_epi_surface_uv"])
        ).any().item()):
            raise ValueError("durable episodic surface_uv is not finite")

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
        self.pattern_usage.zero_()
        self.previous_attention.zero_()
        self.previous_intent.zero_()
        self.previous_object_summary.zero_()
        self.previous_motion_token.zero_()
        self.previous_ontology_context.zero_()
        self.event_age.zero_()
        self.has_previous_event.zero_()
        self.last_compress_step.copy_(self.time_step)
        self.gws.EnsureB(batch_size)
        self.gws.Reset()
        self.ResetInternalLoss()
        self.pending.clear()
        self._row_merge_transactions.clear()
        self._row_merge_versions.clear()

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
        gws_valid = (self.gws.ttl > 0) & (self.gws.priority > 0) # [B, S_gws]
        K_gws = min(perTypeBudget["gws"], S_gws, int(gws_valid.sum(dim=1).max().item()))
        if K_gws > 0:
            gws_age = (self.gws.global_step.view(B, 1) - self.gws.created_step).clamp(min=0).float()
            gws_source = self.gws.source
            gws_beta = torch.full_like(gws_age, float(self.gws.recency_temp))
            gws_beta = torch.where(gws_source == MemoryType.SRC_MIXED, gws_beta + float(self.gws.recency_temp) * 0.5, gws_beta)
            gws_beta = torch.where(gws_source == MemoryType.SRC_IMAGINE, gws_beta + float(self.gws.recency_temp), gws_beta)
            gws_scores = self.gws.priority * torch.exp(-gws_beta * gws_age)
            gws_scores = gws_scores * self.gws.source_confidence
            gws_scores = gws_scores.masked_fill(~gws_valid, -1e9)
            _, gws_idx = StableTopk(gws_scores, K_gws) # [B, K_gws]
            out["gws"] = GatherTopkLatestFirst(self.gws.vals, gws_idx, self.gws.created_step)
            out["gws_valid"] = GatherMeta(
                gws_valid,
                gws_idx,
                self.gws.created_step)
            meta["gws"] = {
                "score": GatherMeta(gws_scores, gws_idx, self.gws.created_step),
                "source": GatherMeta(self.gws.source, gws_idx, self.gws.created_step).float(),
                "confidence": GatherMeta(self.gws.source_confidence, gws_idx, self.gws.created_step),
                "age": GatherMeta(gws_age, gws_idx, self.gws.created_step),
                "touch": GatherMeta(self.gws.touch.float(), gws_idx, self.gws.created_step),
                "reward_abs": torch.zeros(B, K_gws, device=device, dtype=self.dtype),
                "step": GatherMeta(self.gws.created_step.float(), gws_idx, self.gws.created_step),}

        M_kv = int(self.memory_size)
        filled_kv = self.memory_filled # [B]
        K_kv = min(perTypeBudget["kv"], M_kv, int(filled_kv.max().item()))
        if K_kv > 0:
            ar = torch.arange(M_kv, device=device).view(1, M_kv) # [1, M_kv]
            valid = ar < filled_kv.view(B, 1) # [B, M_kv]
            age = (self.time_step.view(B, 1) - self.memory_steps).clamp(min=0).float()
            if self.memory_importance.dtype == torch.float16:
                scores = (
                    self.memory_importance
                    * (1.0 + 0.5 * torch.tanh(self.memory_reward_abs))
                ).masked_fill(
                    ~valid,
                    torch.finfo(self.memory_importance.dtype).min)
            else:
                scores = (self.memory_importance * (1.0 + 0.5 * torch.tanh(self.memory_reward_abs))).masked_fill(~valid, -1e9) # [B, M_kv]
            _, idx = StableTopk(scores, K_kv) # [B, K_kv]
            out["kv"] = GatherTopkLatestFirst(self.memory_values, idx, self.memory_steps) # [B, K_kv, D]
            out["kv_valid"] = GatherMeta(valid, idx, self.memory_steps)
            meta["kv"] = {
                "score": GatherMeta(scores, idx, self.memory_steps),
                "source": GatherMeta(self.memory_source.float(), idx, self.memory_steps),
                "realm": GatherMeta(self.memory_realm.float(), idx, self.memory_steps),
                "confidence": GatherMeta(self.memory_source_confidence, idx, self.memory_steps),
                "age": GatherMeta(age, idx, self.memory_steps),
                "touch": GatherMeta(self.memory_touch.float(), idx, self.memory_steps),
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
            if sem.prio.dtype == torch.float16:
                scores = (
                    sem.prio
                    * SourceConfidence(sem.source, dtype=sem.prio.dtype)
                ).masked_fill(
                    ~valid,
                    torch.finfo(sem.prio.dtype).min)
            else:
                scores = (sem.prio * SourceConfidence(sem.source, dtype=sem.prio.dtype)).masked_fill(~valid, -1e9) # [B, M_sem]
            _, idx = StableTopk(scores, K_sem) # [B, K_sem]
            out["ltm_sem"] = GatherTopkLatestFirst(sem.vals, idx, sem.step) # [B, K_sem, D]
            out["ltm_sem_valid"] = GatherMeta(valid, idx, sem.step)
            meta["ltm_sem"] = {
                "score": GatherMeta(scores, idx, sem.step),
                "source": GatherMeta(sem.source.float(), idx, sem.step),
                "realm": GatherMeta(sem.realm.float(), idx, sem.step),
                "confidence": GatherMeta(sem.source_confidence, idx, sem.step),
                "age": GatherMeta(age, idx, sem.step),
                "touch": GatherMeta(sem.touch.float(), idx, sem.step),
                "reward_abs": torch.zeros(B, K_sem, device=device, dtype=self.dtype),
                "step": GatherMeta(sem.step.float(), idx, sem.step),}


        epi = self.ltm.episodic
        M_epi = int(epi.capacity)
        filled_epi = epi.filled # [B]
        K_epi = min(perTypeBudget["ltm_epi"], M_epi, int(filled_epi.max().item()))
        if K_epi > 0:
            ar = torch.arange(M_epi, device=epi.prio.device).view(1, M_epi) # [1, M_epi]
            valid = ar < filled_epi.view(B, 1) # [B, M_epi]
            age = (epi.global_step.view(B, 1) - epi.step).clamp(min=0).float()
            if epi.prio.dtype == torch.float16:
                scores = (
                    epi.prio
                    * (1.0 + 0.5 * torch.tanh(epi.rew_abs))
                    * SourceConfidence(epi.source, dtype=epi.prio.dtype)
                ).masked_fill(
                    ~valid,
                    torch.finfo(epi.prio.dtype).min)
            else:
                scores = (epi.prio * (1.0 + 0.5 * torch.tanh(epi.rew_abs)) * SourceConfidence(epi.source, dtype=epi.prio.dtype)).masked_fill(~valid, -1e9) # [B, M_epi]
            _, idx = StableTopk(scores, K_epi) # [B, K_epi]
            out["ltm_epi"] = GatherTopkLatestFirst(epi.vals, idx, epi.step) # [B, K_epi, D]
            out["ltm_epi_valid"] = GatherMeta(valid, idx, epi.step)
            meta["ltm_epi"] = {
                "score": GatherMeta(scores, idx, epi.step),
                "source": GatherMeta(epi.source.float(), idx, epi.step),
                "realm": GatherMeta(epi.realm.float(), idx, epi.step),
                "agency_id": GatherMeta(
                    epi.agency_id.float(), idx, epi.step),
                "motion_layer_prob": GatherTopkLatestFirst(
                    epi.motion_layer_prob, idx, epi.step),
                "observed_surface_parent_index": GatherMeta(
                    epi.observed_surface_parent_index.float(),
                    idx,
                    epi.step),
                "surface_uv": GatherTopkLatestFirst(
                    epi.surface_uv, idx, epi.step),
                "surface_uv_confidence": GatherMeta(
                    epi.surface_uv_confidence, idx, epi.step),
                "verification": GatherMeta(
                    epi.verification, idx, epi.step),
                "content_change": GatherMeta(
                    epi.content_change, idx, epi.step),
                "confidence": GatherMeta(epi.source_confidence, idx, epi.step),
                "age": GatherMeta(age, idx, epi.step),
                "touch": GatherMeta(epi.touch.float(), idx, epi.step),
                "reward_abs": GatherMeta(epi.rew_abs, idx, epi.step),
                "step": GatherMeta(epi.step.float(), idx, epi.step),}


        sym = self.sym_mem
        K_sym = min(perTypeBudget["sym"], int(sym.filled.max().item()))
        if K_sym > 0:
            sym_slots = torch.arange(sym.capacity, device=device).view(1, sym.capacity)
            sym_valid = sym_slots < sym.filled.view(B, 1)
            sym_scores = sym.prio.masked_fill(~sym_valid, -torch.inf)
            _, sym_idx = StableTopk(sym_scores, K_sym)
            out["sym"] = GatherTopkLatestFirst(sym.P_vals, sym_idx, sym.step)
            out["sym_valid"] = GatherMeta(sym_valid, sym_idx, sym.step)
            if includeMeta:
                age = (sym.global_step.view(B, 1) - sym.step).clamp(min=0).float()
                meta["sym"] = {
                    "score": GatherMeta(sym_scores, sym_idx, sym.step),
                    "source": GatherMeta(sym.source.float(), sym_idx, sym.step),
                    "confidence": GatherMeta(sym.source_confidence, sym_idx, sym.step),
                    "age": GatherMeta(age, sym_idx, sym.step),
                    "touch": GatherMeta(sym.touch.float(), sym_idx, sym.step),
                    "reward_abs": torch.zeros(B, K_sym, device=device, dtype=self.dtype),
                    "step": GatherMeta(sym.step.float(), sym_idx, sym.step),}

        if includeMeta and meta:
            out["meta"] = meta
        return out

    def ExportConsciousBank(
        self,
        topk: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        B = int(self.memory_filled.size(0))
        budget = max(0, int(topk))
        if budget == 0:
            return {
                "tokens": torch.zeros(
                    B, 0, self.memory_dim,
                    device=self.device, dtype=self.dtype),
                "valid": torch.zeros(
                    B, 0,
                    device=self.device, dtype=torch.bool),}
        bank = self.ExportMemoryBank(
            topk=budget,
            perTypeBudget={
                name: budget
                for name in ("gws", "kv", "ltm_sem", "ltm_epi", "sym")},
            includeMeta=True)
        if bank is None:
            return {
                "tokens": torch.zeros(
                    B, 0, self.memory_dim,
                    device=self.device, dtype=self.dtype),
                "valid": torch.zeros(
                    B, 0,
                    device=self.device, dtype=torch.bool),}
        tokens: List[torch.Tensor] = []
        valid: List[torch.Tensor] = []
        scores: List[torch.Tensor] = []
        sources: List[torch.Tensor] = []
        ages: List[torch.Tensor] = []
        confidences: List[torch.Tensor] = []
        staleness: List[torch.Tensor] = []
        for name in ("gws", "kv", "ltm_sem", "ltm_epi", "sym"):
            if name not in bank:
                continue
            value = bank[name]
            if name == "sym":
                value = self.sym_embed(value)
            if value.size(-1) != self.memory_dim:
                raise RuntimeError(
                    f"memory conscious token {name!r} has invalid width "
                    f"{value.size(-1)}")
            tokens.append(value)
            item_valid = bank[f"{name}_valid"]
            valid.append(item_valid)
            scores.append(
                bank["meta"][name]["score"].masked_fill(
                    ~item_valid, -torch.inf))
            metadata = bank["meta"][name]
            source = metadata["source"].to(dtype=torch.long).clamp_min(0)
            age = metadata["age"].to(dtype=value.dtype).clamp_min(0.0)
            confidence = metadata["confidence"].to(
                dtype=value.dtype).clamp(0.0, 1.0)
            sources.append(source)
            ages.append(age)
            confidences.append(confidence)
            staleness.append(age / (1.0 + age))
        if not tokens:
            return {
                "tokens": torch.zeros(
                    B, 0, self.memory_dim,
                    device=self.device, dtype=self.dtype),
                "valid": torch.zeros(
                    B, 0,
                    device=self.device, dtype=torch.bool),}
        merged_tokens = torch.cat(tokens, dim=1)
        merged_valid = torch.cat(valid, dim=1)
        merged_scores = torch.cat(scores, dim=1)
        merged_sources = torch.cat(sources, dim=1)
        merged_ages = torch.cat(ages, dim=1)
        merged_confidences = torch.cat(confidences, dim=1)
        merged_staleness = torch.cat(staleness, dim=1)
        count = min(budget, int(merged_tokens.size(1)))
        _, indices = torch.topk(merged_scores, k=count, dim=1)
        merged_tokens = torch.gather(
            merged_tokens,
            1,
            indices.unsqueeze(-1).expand(B, count, self.memory_dim))
        merged_valid = torch.gather(merged_valid, 1, indices)
        merged_sources = torch.gather(merged_sources, 1, indices)
        merged_ages = torch.gather(merged_ages, 1, indices)
        merged_confidences = torch.gather(
            merged_confidences, 1, indices)
        merged_staleness = torch.gather(merged_staleness, 1, indices)
        merged_tokens = merged_tokens * merged_valid.unsqueeze(-1).to(
            merged_tokens.dtype)
        return {
            "tokens": merged_tokens,
            "valid": merged_valid,
            "source": merged_sources,
            "age": merged_ages,
            "confidence": merged_confidences,
            "staleness": merged_staleness,}


    @torch.no_grad()
    def ExportTransientState(self) -> Dict[str, torch.Tensor]:
        gws = self.gws.Inspect()
        state = {
            "last_compress_step": self.last_compress_step.detach().clone(),
            "h_state": self.h_state.detach().clone(),
            "fast_weights": self.fast_weights.detach().clone(),
            "ns_prev_P_post": self.ns_prev_P_post.detach().clone(),
            "ns_penalty_vec": self.ns_penalty_vec.detach().clone(),
            "pattern_usage": self.pattern_usage.detach().clone(),
            "previous_attention": self.previous_attention.detach().clone(),
            "previous_intent": self.previous_intent.detach().clone(),
            "previous_object_summary": self.previous_object_summary.detach().clone(),
            "previous_motion_token": self.previous_motion_token.detach().clone(),
            "previous_ontology_context": self.previous_ontology_context.detach().clone(),
            "event_age": self.event_age.detach().clone(),
            "has_previous_event": self.has_previous_event.detach().clone(),
            "gws_global_step": self.gws.global_step.detach().clone(),
            "gws_keys": gws["keys"].detach().clone(),
            "gws_vals": gws["vals"].detach().clone(),
            "gws_priority": gws["priority"].detach().clone(),
            "gws_ttl": gws["ttl"].detach().clone(),
            "gws_created_step": gws["created_step"].detach().clone(),
            "gws_last_step": gws["last_step"].detach().clone(),
            "gws_last_rehearsal_step": gws["last_rehearsal_step"].detach().clone(),
            "gws_touch": gws["touch"].detach().clone(),
            "gws_source": gws["source"].detach().clone(),
            "gws_source_confidence": gws["source_confidence"].detach().clone(),}
        if tuple(state) != self.TRANSIENT_MEMORY_STATE_FIELDS:
            raise RuntimeError("transient-memory field declaration and export order disagree")
        return state

    @torch.no_grad()
    def ImportTransientState(self, state: Dict[str, torch.Tensor]) -> None:
        self.ImportCurrentState(state, includeDurable=False)

    @torch.no_grad()
    def ExportState(self, step: Optional[int] = None) -> Dict[str, torch.Tensor]:
        self.FlushPendingWrites()
        sem = self.ltm.semantic
        epi = self.ltm.episodic
        sym = self.sym_mem
        state: Dict[str, torch.Tensor] = {
            **self.ExportDurableState(),
            **self.ExportTransientState(),}

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
        modified_step = torch.maximum(
            state["memory_steps"],
            state["memory_last_rehearsal_steps"])
        keep0 = valid0 & (modified_step > s) # [B,M]
        keep_cnt = keep0.sum(dim=1) # [B]

        metric = torch.where(
            keep0,
            modified_step.float(),
            torch.full_like(modified_step.float(), -1e9)) # [B,M]
        _, idx = torch.sort(metric, dim=1, descending=True) # [B,M]

        new_valid = ar < keep_cnt.view(B, 1) # [B,M]

        idx3 = idx.unsqueeze(-1).expand(B, M, D)
        idxE = idx.unsqueeze(-1).expand(B, M, int(self.emotion_dim))

        state["memory_keys"] = torch.gather(state["memory_keys"], 1, idx3) * new_valid.unsqueeze(-1).float()
        state["memory_values"] = torch.gather(state["memory_values"], 1, idx3) * new_valid.unsqueeze(-1).float()
        state["memory_emotion"] = torch.gather(state["memory_emotion"], 1, idxE) * new_valid.unsqueeze(-1).float()

        state["memory_importance"] = torch.gather(state["memory_importance"], 1, idx) * new_valid.float()
        state["memory_steps"] = (torch.gather(state["memory_steps"], 1, idx) * new_valid.long())
        state["memory_last_access_steps"] = torch.gather(state["memory_last_access_steps"], 1, idx) * new_valid.long()
        state["memory_last_rehearsal_steps"] = torch.gather(state["memory_last_rehearsal_steps"], 1, idx) * new_valid.long()
        state["memory_touch"] = torch.gather(state["memory_touch"], 1, idx) * new_valid.long()
        state["memory_merge_count"] = torch.gather(state["memory_merge_count"], 1, idx) * new_valid.long()
        state["memory_source"] = torch.where(new_valid,torch.gather(state["memory_source"], 1, idx),torch.zeros_like(state["memory_source"]),)
        state["memory_source_confidence"] = torch.gather(state["memory_source_confidence"], 1, idx) * new_valid.float()
        state["memory_realm"] = torch.where(
            new_valid,
            torch.gather(state["memory_realm"], 1, idx),
            torch.full_like(
                state["memory_realm"],
                ONTOLOGY_REALM_UNKNOWN))
        state["memory_reward_abs"] = torch.gather(state["memory_reward_abs"], 1, idx) * new_valid.float()

        state["memory_filled"] = keep_cnt # [B]

        CapS = int(sem.capacity)
        arS = torch.arange(CapS, device=device).view(1, CapS)

        filledS = state["ltm_sem_filled"] # [B]
        validS0 = arS < filledS.view(B, 1)
        modified_sem = torch.maximum(
            state["ltm_sem_step"],
            state["ltm_sem_last_rehearsal_step"])
        keepS0 = validS0 & (modified_sem > s)
        keepS = keepS0.sum(dim=1)

        metricS = torch.where(
            keepS0,
            modified_sem.float(),
            torch.full_like(modified_sem.float(), -1e9))
        _, idxS = torch.sort(metricS, dim=1, descending=True)
        new_validS = arS < keepS.view(B, 1)

        idxS3 = idxS.unsqueeze(-1).expand(B, CapS, D)

        state["ltm_sem_keys"] = torch.gather(state["ltm_sem_keys"], 1, idxS3) * new_validS.unsqueeze(-1).float()
        state["ltm_sem_vals"] = torch.gather(state["ltm_sem_vals"], 1, idxS3) * new_validS.unsqueeze(-1).float()
        state["ltm_sem_prio"] = torch.gather(state["ltm_sem_prio"], 1, idxS) * new_validS.float()
        state["ltm_sem_touch"] = (torch.gather(state["ltm_sem_touch"], 1, idxS) * new_validS.long())
        state["ltm_sem_step"] = (torch.gather(state["ltm_sem_step"], 1, idxS) * new_validS.long())
        state["ltm_sem_last_access_step"] = torch.gather(state["ltm_sem_last_access_step"], 1, idxS) * new_validS.long()
        state["ltm_sem_last_rehearsal_step"] = torch.gather(state["ltm_sem_last_rehearsal_step"], 1, idxS) * new_validS.long()
        state["ltm_sem_consolidation_count"] = torch.gather(state["ltm_sem_consolidation_count"], 1, idxS) * new_validS.long()
        state["ltm_sem_prototype_count"] = torch.gather(state["ltm_sem_prototype_count"], 1, idxS) * new_validS.float()
        state["ltm_sem_prototype_variance"] = torch.gather(
            state["ltm_sem_prototype_variance"], 1, idxS3) * new_validS.unsqueeze(-1).float()
        state["ltm_sem_source"] = torch.where(new_validS,torch.gather(state["ltm_sem_source"], 1, idxS),torch.zeros_like(state["ltm_sem_source"]),)
        state["ltm_sem_source_confidence"] = torch.gather(state["ltm_sem_source_confidence"], 1, idxS) * new_validS.float()
        state["ltm_sem_realm"] = torch.where(
            new_validS,
            torch.gather(state["ltm_sem_realm"], 1, idxS),
            torch.full_like(
                state["ltm_sem_realm"],
                ONTOLOGY_REALM_UNKNOWN))
        state["ltm_sem_filled"] = keepS

        CapE = int(epi.capacity)
        arE = torch.arange(CapE, device=device).view(1, CapE)

        filledE = state["ltm_epi_filled"]
        validE0 = arE < filledE.view(B, 1)
        modified_epi = torch.maximum(
            state["ltm_epi_step"],
            state["ltm_epi_last_rehearsal_step"])
        keepE0 = validE0 & (modified_epi > s)
        keepE = keepE0.sum(dim=1)

        metricE = torch.where(
            keepE0,
            modified_epi.float(),
            torch.full_like(modified_epi.float(), -1e9))
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
        for field in (
            "ltm_epi_last_access_step",
            "ltm_epi_last_rehearsal_step",
            "ltm_epi_consolidation_count",
            "ltm_epi_episode_id",
            "ltm_epi_event_id",
            "ltm_epi_slot_generation",):
            state[field] = torch.gather(state[field], 1, idxEpi) * new_validE.long()
        state["ltm_epi_source_confidence"] = torch.gather(
            state["ltm_epi_source_confidence"], 1, idxEpi) * new_validE.float()
        state["ltm_epi_prev_index"].fill_(-1)
        state["ltm_epi_next_index"].fill_(-1)
        state["ltm_epi_prev_generation"].fill_(-1)
        state["ltm_epi_next_generation"].fill_(-1)
        state["ltm_epi_last_event_index"].fill_(-1)
        state["ltm_epi_source"] = torch.where(new_validE,torch.gather(state["ltm_epi_source"], 1, idxEpi),torch.zeros_like(state["ltm_epi_source"]),)
        state["ltm_epi_realm"] = torch.where(
            new_validE,
            torch.gather(state["ltm_epi_realm"], 1, idxEpi),
            torch.full_like(
                state["ltm_epi_realm"],
                ONTOLOGY_REALM_UNKNOWN))
        state["ltm_epi_agency_id"] = torch.where(
            new_validE,
            torch.gather(state["ltm_epi_agency_id"], 1, idxEpi),
            torch.full_like(
                state["ltm_epi_agency_id"],
                ONTOLOGY_AGENCY_UNKNOWN))
        state["ltm_epi_observed_surface_parent_index"] = torch.where(
            new_validE,
            torch.gather(
                state["ltm_epi_observed_surface_parent_index"],
                1,
                idxEpi),
            torch.full_like(
                state["ltm_epi_observed_surface_parent_index"],
                -1))
        idxE5 = idxEpi.unsqueeze(-1).expand(B, CapE, 5)
        idxE2 = idxEpi.unsqueeze(-1).expand(B, CapE, 2)
        state["ltm_epi_motion_layer_prob"] = torch.gather(
            state["ltm_epi_motion_layer_prob"],
            1,
            idxE5) * new_validE.unsqueeze(-1)
        state["ltm_epi_surface_uv"] = torch.gather(
            state["ltm_epi_surface_uv"],
            1,
            idxE2) * new_validE.unsqueeze(-1)
        for field in (
            "ltm_epi_surface_uv_confidence",
            "ltm_epi_verification",
            "ltm_epi_content_change",):
            state[field] = torch.gather(
                state[field],
                1,
                idxEpi) * new_validE
        state["ltm_epi_filled"] = keepE

        cap_sym = int(sym.capacity)
        sym_slots = torch.arange(cap_sym, device=device).view(1, cap_sym)
        sym_valid = sym_slots < state["sym_mem_filled"].view(B, 1)
        modified_sym = torch.maximum(
            state["sym_mem_step"],
            state["sym_mem_last_rehearsal_step"])
        sym_keep_mask = sym_valid & (modified_sym > s)
        sym_keep_count = sym_keep_mask.sum(dim=1)
        sym_metric = torch.where(
            sym_keep_mask,
            modified_sym.float(),
            torch.full_like(modified_sym.float(), -torch.inf))
        sym_order = torch.argsort(sym_metric, dim=1, descending=True)
        sym_new_valid = sym_slots < sym_keep_count.view(B, 1)
        sym_index3 = sym_order.unsqueeze(-1).expand(B, cap_sym, sym.K)
        for field in ("sym_mem_P_keys", "sym_mem_P_vals"):
            state[field] = torch.gather(state[field], 1, sym_index3) * sym_new_valid.unsqueeze(-1)
        for field in (
            "sym_mem_prio",
            "sym_mem_step",
            "sym_mem_last_access_step",
            "sym_mem_last_rehearsal_step",
            "sym_mem_touch",
            "sym_mem_source",
            "sym_mem_source_confidence",):
            gathered = torch.gather(state[field], 1, sym_order)
            state[field] = torch.where(sym_new_valid, gathered, torch.zeros_like(gathered))
        state["sym_mem_filled"] = sym_keep_count

        if tuple(state) != self.FULL_MEMORY_STATE_FIELDS:
            raise RuntimeError("memory-state export does not match its schema")
        return state

    @torch.no_grad()
    def MergeMemoryState(
        self,
        state: Dict[str, torch.Tensor],
        mergeGws: bool = False,
        sourceBaseStep: Optional[int] = None,
        sourceNewStep: Optional[int] = None,) -> None:

        if type(state) is not dict or tuple(state) != self.FULL_MEMORY_STATE_FIELDS:
            raise TypeError("merged memory-state fields do not match the current schema")
        if not all(torch.is_tensor(value) for value in state.values()):
            raise TypeError("every merged memory-state field must be a tensor")
        if (sourceBaseStep is None) != (sourceNewStep is None):
            raise ValueError("delta merge requires both source base and new steps")
        if sourceBaseStep is not None and int(sourceNewStep) < int(sourceBaseStep):
            raise ValueError("delta new step precedes its base step")

        def MapTimestamp(
            value: int,
            destinationNow: int,
            sourceNow: int,) -> int:
            if sourceBaseStep is None:
                source_age = max(0, int(sourceNow) - int(value))
                return int(destinationNow) - source_age
            if value <= int(sourceBaseStep):
                return int(value)
            relative_age = int(sourceNewStep) - int(value)
            return max(0, int(destinationNow) - relative_age)

        def RequireShape(condition: bool, message: str) -> None:
            if not condition:
                raise ValueError(f"invalid merged memory-state shape: {message}")

        usage = self.usage_bank
        usage_fields = (
            ("usage_applicable", usage.applicable),
            ("usage_default_params", usage.default_params),
            ("usage_expected_dx", usage.expected_dx),
            ("usage_success_alpha", usage.success_alpha),
            ("usage_success_beta", usage.success_beta),
            ("usage_param_mu", usage.param_mu),
            ("usage_param_logvar", usage.param_logvar),
            ("usage_parameter_observations", usage.parameter_observations),
            ("usage_instance_descriptors", usage.instance_descriptors),
            ("usage_attribute_centroid", usage.attribute_centroid),)
        for field, destination in usage_fields:
            RequireShape(
                state[field].shape == destination.shape,
                f"{field} differs from the destination")

        memory_keys = state["memory_keys"]
        memory_values = state["memory_values"]
        RequireShape(memory_keys.dim() == 3, "memory_keys must be [B,M,D]")
        RequireShape(memory_values.shape == memory_keys.shape, "memory_values must match memory_keys")
        B_src, M_src, D_src = memory_keys.shape
        RequireShape(D_src == self.memory_dim, "memory feature dimension differs from the destination")
        RequireShape(state["memory_filled"].shape == (B_src,), "memory_filled must be [B]")
        for field in (
            "memory_importance",
            "memory_steps",
            "memory_last_access_steps",
            "memory_last_rehearsal_steps",
            "memory_touch",
            "memory_merge_count",
            "memory_source",
            "memory_source_confidence",
            "memory_realm",
            "memory_reward_abs",):
            RequireShape(state[field].shape == (B_src, M_src), f"{field} must be [B,M]")
        RequireShape(
            state["memory_emotion"].shape == (B_src, M_src, self.emotion_dim),
            "memory_emotion must be [B,M,E]")
        RequireShape(state["time_step"].shape == (B_src,), "time_step must be [B]")

        sem_keys = state["ltm_sem_keys"]
        sem_vals = state["ltm_sem_vals"]
        RequireShape(sem_keys.dim() == 3, "ltm_sem_keys must be [B,C,D]")
        RequireShape(sem_vals.shape == sem_keys.shape, "ltm_sem_vals must match ltm_sem_keys")
        B_sem, C_sem, D_sem = sem_keys.shape
        RequireShape(B_sem == B_src and D_sem == self.memory_dim, "semantic-memory batch/feature dimensions differ")
        RequireShape(state["ltm_sem_filled"].shape == (B_src,), "ltm_sem_filled must be [B]")
        for field in (
            "ltm_sem_prio",
            "ltm_sem_touch",
            "ltm_sem_step",
            "ltm_sem_last_access_step",
            "ltm_sem_last_rehearsal_step",
            "ltm_sem_consolidation_count",
            "ltm_sem_prototype_count",
            "ltm_sem_source",
            "ltm_sem_source_confidence",
            "ltm_sem_realm",):
            RequireShape(state[field].shape == (B_src, C_sem), f"{field} must be [B,C]")
        RequireShape(
            state["ltm_sem_prototype_variance"].shape == sem_keys.shape,
            "ltm_sem_prototype_variance must be [B,C,D]")
        RequireShape(
            state["ltm_sem_global_step"].shape == (B_src,),
            "ltm_sem_global_step must be [B]")

        epi_keys = state["ltm_epi_keys"]
        epi_state_keys = state["ltm_epi_state_keys"]
        epi_vals = state["ltm_epi_vals"]
        RequireShape(epi_keys.dim() == 3, "ltm_epi_keys must be [B,C,D]")
        RequireShape(epi_state_keys.shape == epi_keys.shape, "ltm_epi_state_keys must match ltm_epi_keys")
        RequireShape(epi_vals.shape == epi_keys.shape, "ltm_epi_vals must match ltm_epi_keys")
        B_epi, C_epi, D_epi = epi_keys.shape
        RequireShape(B_epi == B_src and D_epi == self.memory_dim, "episodic-memory batch/feature dimensions differ")
        RequireShape(state["ltm_epi_filled"].shape == (B_src,), "ltm_epi_filled must be [B]")
        for field in (
            "ltm_epi_prio", "ltm_epi_rew", "ltm_epi_rew_abs",
            "ltm_epi_step", "ltm_epi_last_access_step",
            "ltm_epi_last_rehearsal_step", "ltm_epi_consolidation_count",
            "ltm_epi_touch", "ltm_epi_source", "ltm_epi_source_confidence",
            "ltm_epi_realm",
            "ltm_epi_agency_id",
            "ltm_epi_observed_surface_parent_index",
            "ltm_epi_surface_uv_confidence", "ltm_epi_verification",
            "ltm_epi_content_change",
            "ltm_epi_episode_id", "ltm_epi_event_id", "ltm_epi_prev_index",
            "ltm_epi_next_index", "ltm_epi_slot_generation",
            "ltm_epi_prev_generation", "ltm_epi_next_generation",):
            RequireShape(state[field].shape == (B_src, C_epi), f"{field} must be [B,C]")
        RequireShape(
            state["ltm_epi_motion_layer_prob"].shape == (B_src, C_epi, 5),
            "ltm_epi_motion_layer_prob must be [B,C,5]")
        RequireShape(
            state["ltm_epi_surface_uv"].shape == (B_src, C_epi, 2),
            "ltm_epi_surface_uv must be [B,C,2]")
        for field in ("ltm_epi_last_event_index", "ltm_epi_current_episode_id"):
            RequireShape(state[field].shape == (B_src,), f"{field} must be [B]")
        RequireShape(
            state["ltm_epi_global_step"].shape == (B_src,),
            "ltm_epi_global_step must be [B]")

        sym_keys = state["sym_mem_P_keys"]
        sym_vals = state["sym_mem_P_vals"]
        RequireShape(sym_keys.dim() == 3, "sym_mem_P_keys must be [B,C,K]")
        RequireShape(sym_vals.shape == sym_keys.shape, "sym_mem_P_vals must match sym_mem_P_keys")
        B_sym, C_sym, K_sym = sym_keys.shape
        RequireShape(B_sym == B_src and K_sym == self.sym_mem.K, "symbolic-memory batch/feature dimension differs")
        for field in (
            "sym_mem_prio", "sym_mem_step", "sym_mem_last_access_step",
            "sym_mem_last_rehearsal_step", "sym_mem_touch", "sym_mem_source",
            "sym_mem_source_confidence",):
            RequireShape(state[field].shape == (B_src, C_sym), f"{field} must be [B,C]")
        RequireShape(state["sym_mem_filled"].shape == (B_src,), "sym_mem_filled must be [B]")
        RequireShape(
            state["sym_mem_global_step"].shape == (B_src,),
            "sym_mem_global_step must be [B]")

        filled_ranges = (
            ("memory_filled", M_src),
            ("ltm_sem_filled", C_sem),
            ("ltm_epi_filled", C_epi),
            ("sym_mem_filled", C_sym),)
        for field, capacity in filled_ranges:
            filled = state[field]
            if bool(((filled < 0) | (filled > capacity)).any().item()):
                raise ValueError(
                    f"{field} is outside the source memory capacity")

        for realm_field, filled_field in (
            ("memory_realm", "memory_filled"),
            ("ltm_sem_realm", "ltm_sem_filled"),
            ("ltm_epi_realm", "ltm_epi_filled"),):
            values = state[realm_field]
            slots = torch.arange(
                values.size(1),
                device=values.device).view(1, -1)
            valid = slots < state[filled_field].view(-1, 1)
            invalid = (
                (values < ONTOLOGY_REALM_SELF)
                | (values > ONTOLOGY_REALM_UNKNOWN))
            if bool((invalid & valid).any().item()):
                raise ValueError(
                    f"{realm_field} contains an invalid entity realm")

        epi_slots = torch.arange(
            C_epi,
            device=state["ltm_epi_filled"].device).view(1, C_epi)
        epi_valid = epi_slots < state["ltm_epi_filled"].view(-1, 1)
        agency_id = state["ltm_epi_agency_id"]
        if bool((epi_valid & (
            (agency_id < ONTOLOGY_AGENCY_SELF)
            | (agency_id > ONTOLOGY_AGENCY_UNKNOWN)
        )).any().item()):
            raise ValueError("ltm_epi_agency_id contains an invalid agency")
        observed_parent = state[
            "ltm_epi_observed_surface_parent_index"]
        if bool((epi_valid & (observed_parent < -1)).any().item()):
            raise ValueError(
                "ltm_epi_observed_surface_parent_index is invalid")
        for field in (
            "ltm_epi_motion_layer_prob",
            "ltm_epi_surface_uv_confidence",
            "ltm_epi_verification",
            "ltm_epi_content_change",):
            value = state[field]
            valid = epi_valid.unsqueeze(-1) if value.dim() == 3 else epi_valid
            if bool((valid & (
                ~torch.isfinite(value)
                | (value < 0.0)
                | (value > 1.0)
            )).any().item()):
                raise ValueError(f"{field} is not a finite probability")
        if bool((
            epi_valid.unsqueeze(-1)
            & ~torch.isfinite(state["ltm_epi_surface_uv"])
        ).any().item()):
            raise ValueError("ltm_epi_surface_uv is not finite")

        if mergeGws:
            gws_keys = state["gws_keys"]
            gws_vals = state["gws_vals"]
            RequireShape(gws_keys.dim() == 3, "gws_keys must be [B,S,D]")
            RequireShape(gws_vals.shape == gws_keys.shape, "gws_vals must match gws_keys")
            B_gws, S_gws, D_gws = gws_keys.shape
            RequireShape(B_gws == B_src and D_gws == self.memory_dim, "workspace batch/feature dimensions differ")
            for field in (
                "gws_priority", "gws_ttl", "gws_created_step", "gws_last_step",
                "gws_last_rehearsal_step", "gws_touch", "gws_source",
                "gws_source_confidence",):
                RequireShape(state[field].shape == (B_src, S_gws), f"{field} must be [B,S]")
            RequireShape(state["gws_global_step"].shape == (B_src,), "gws_global_step must be [B]")

        self.FlushPendingWrites()

        device = self.device
        dtype = self.dtype

        B_dst = int(self.memory_filled.size(0))

        if sourceBaseStep is None:
            B_clock = min(B_dst, B_src)
            for b in range(B_clock):
                source_ages = []

                def AppendSourceAge(
                    globalField: str,
                    createdField: str,
                    count: int,) -> None:
                    if count == 0:
                        return
                    source_now = int(state[globalField][b].item())
                    earliest = int(state[createdField][b, :count].min().item())
                    source_ages.append(max(0, source_now - earliest))

                AppendSourceAge(
                    "time_step",
                    "memory_steps",
                    int(state["memory_filled"][b].item()))
                AppendSourceAge(
                    "ltm_sem_global_step",
                    "ltm_sem_step",
                    int(state["ltm_sem_filled"][b].item()))
                AppendSourceAge(
                    "ltm_epi_global_step",
                    "ltm_epi_step",
                    int(state["ltm_epi_filled"][b].item()))
                AppendSourceAge(
                    "sym_mem_global_step",
                    "sym_mem_step",
                    int(state["sym_mem_filled"][b].item()))
                if mergeGws:
                    source_gws_valid = (
                        (state["gws_ttl"][b] > 0)
                        & (state["gws_priority"][b] > 0))
                    if bool(source_gws_valid.any().item()):
                        source_gws_now = int(
                            state["gws_global_step"][b].item())
                        source_gws_earliest = int(
                            state["gws_created_step"][b][source_gws_valid]
                            .min().item())
                        source_ages.append(max(
                            0,
                            source_gws_now - source_gws_earliest))

                target_now = max(
                    int(self.time_step[b].item()),
                    int(self.gws.global_step[b].item()),
                    int(self.ltm.semantic.global_step[b].item()),
                    int(self.ltm.episodic.global_step[b].item()),
                    int(self.sym_mem.global_step[b].item()),
                    max(source_ages, default=0))

                def ShiftDestinationAxis(
                    created: torch.Tensor,
                    accessed: torch.Tensor,
                    rehearsed: torch.Tensor,
                    valid: torch.Tensor,
                    globalStep: torch.Tensor,) -> int:
                    delta = target_now - int(globalStep[b].item())
                    if delta != 0:
                        created[b, valid] += delta
                        accessed[b, valid] += delta
                        rehearsed[b, valid] += delta
                    globalStep[b] = target_now
                    return delta

                memory_valid = torch.arange(
                    self.memory_size,
                    device=self.memory_steps.device) < self.memory_filled[b]
                main_delta = ShiftDestinationAxis(
                    self.memory_steps,
                    self.memory_last_access_steps,
                    self.memory_last_rehearsal_steps,
                    memory_valid,
                    self.time_step)
                self.last_compress_step[b] += main_delta

                gws_valid = (
                    (self.gws.ttl[b] > 0)
                    & (self.gws.priority[b] > 0))
                ShiftDestinationAxis(
                    self.gws.created_step,
                    self.gws.last_step,
                    self.gws.last_rehearsal_step,
                    gws_valid,
                    self.gws.global_step)

                semantic = self.ltm.semantic
                semantic_valid = torch.arange(
                    semantic.capacity,
                    device=semantic.step.device) < semantic.filled[b]
                ShiftDestinationAxis(
                    semantic.step,
                    semantic.last_access_step,
                    semantic.last_rehearsal_step,
                    semantic_valid,
                    semantic.global_step)

                episodic = self.ltm.episodic
                episodic_valid = torch.arange(
                    episodic.capacity,
                    device=episodic.step.device) < episodic.filled[b]
                ShiftDestinationAxis(
                    episodic.step,
                    episodic.last_access_step,
                    episodic.last_rehearsal_step,
                    episodic_valid,
                    episodic.global_step)

                symbolic = self.sym_mem
                symbolic_valid = torch.arange(
                    symbolic.capacity,
                    device=symbolic.created_step.device) < symbolic.filled[b]
                ShiftDestinationAxis(
                    symbolic.created_step,
                    symbolic.last_access_step,
                    symbolic.last_rehearsal_step,
                    symbolic_valid,
                    symbolic.global_step)

        if mergeGws:
            gk = state["gws_keys"].to(device=self.gws.keys.device, dtype=self.gws.keys.dtype) # [B, S, D]
            gv = state["gws_vals"].to(device=self.gws.vals.device, dtype=self.gws.vals.dtype) # [B, S, D]
            gpr = state["gws_priority"].to(device=self.gws.priority.device, dtype=self.gws.priority.dtype) # [B, S]
            gttl = state["gws_ttl"].to(device=self.gws.ttl.device, dtype=torch.long) # [B, S]
            gcs = state["gws_created_step"].to(device=self.gws.created_step.device, dtype=torch.long)
            gls = state["gws_last_step"].to(device=self.gws.last_step.device, dtype=torch.long) # [B, S]
            grs = state["gws_last_rehearsal_step"].to(device=self.gws.last_rehearsal_step.device, dtype=torch.long)
            gt = state["gws_touch"].to(device=self.gws.touch.device, dtype=torch.long)
            gsrc = state["gws_source"].to(device=self.gws.source.device, dtype=torch.int8) # [B, S]
            gconf = state["gws_source_confidence"].to(device=self.gws.source_confidence.device, dtype=self.gws.source_confidence.dtype)
            ggs = state["gws_global_step"].to(device=self.gws.global_step.device, dtype=torch.long) # [B]

            B_gws_src = int(gk.size(0))
            S_src = int(gk.size(1))
            S_dst = int(self.gws.slots)
            B = min(B_dst, B_gws_src)
            S = min(S_dst, S_src)

            source_alive = (gttl[:B] > 0) & (gpr[:B] > 0)
            source_age = (
                ggs[:B].unsqueeze(1) - gcs[:B]
            ).clamp_min(0).to(gpr.dtype)
            source_retention = (
                torch.log(gpr[:B] + 1e-6)
                + 0.10 * torch.log1p(gt[:B].to(gpr.dtype))
                + 0.10 * torch.log(gconf[:B] + 1e-6)
                + 0.01 * gttl[:B].to(gpr.dtype)
                - 0.001 * source_age)
            source_retention = source_retention.masked_fill(
                ~source_alive,
                -torch.inf)
            _, selected = StableTopk(source_retention, S)
            selected3 = selected.unsqueeze(-1).expand(B, S, self.memory_dim)

            selected_keys = torch.gather(gk[:B], 1, selected3)
            selected_vals = torch.gather(gv[:B], 1, selected3)
            selected_priority = torch.gather(gpr[:B], 1, selected)
            selected_ttl = torch.gather(gttl[:B], 1, selected)
            selected_created = torch.gather(gcs[:B], 1, selected)
            selected_accessed = torch.gather(gls[:B], 1, selected)
            selected_rehearsed = torch.gather(grs[:B], 1, selected)
            selected_touch = torch.gather(gt[:B], 1, selected)
            selected_source = torch.gather(gsrc[:B], 1, selected)
            selected_confidence = torch.gather(gconf[:B], 1, selected)

            replaced_rows = torch.arange(
                B_dst,
                device=self.gws.keys.device) < B
            target_gws_global = self.gws.global_step[:B].clone()
            self.gws.ResetRows(replaced_rows)
            self.gws.global_step[:B].copy_(target_gws_global)
            incoming_alive = (
                (selected_ttl > 0)
                & (selected_priority > 0))
            alive3 = incoming_alive.unsqueeze(-1)
            target_gws_now = target_gws_global.unsqueeze(1)
            created_mapped = target_gws_now - (
                ggs[:B].unsqueeze(1) - selected_created).clamp_min(0)
            accessed_mapped = target_gws_now - (
                ggs[:B].unsqueeze(1) - selected_accessed).clamp_min(0)
            rehearsed_mapped = target_gws_now - (
                ggs[:B].unsqueeze(1) - selected_rehearsed).clamp_min(0)
            self.gws.keys[:B, :S].copy_(torch.where(
                alive3,
                selected_keys,
                torch.zeros_like(selected_keys)))
            self.gws.vals[:B, :S].copy_(torch.where(
                alive3,
                selected_vals,
                torch.zeros_like(selected_vals)))
            self.gws.priority[:B, :S].copy_(torch.where(
                incoming_alive,
                selected_priority,
                torch.zeros_like(selected_priority)))
            self.gws.ttl[:B, :S].copy_(torch.where(
                incoming_alive,
                selected_ttl,
                torch.zeros_like(selected_ttl)))
            self.gws.created_step[:B, :S].copy_(torch.where(
                incoming_alive,
                created_mapped,
                torch.zeros_like(created_mapped)))
            self.gws.last_step[:B, :S].copy_(torch.where(
                incoming_alive,
                accessed_mapped,
                torch.zeros_like(accessed_mapped)))
            self.gws.last_rehearsal_step[:B, :S].copy_(torch.where(
                incoming_alive,
                rehearsed_mapped,
                torch.zeros_like(rehearsed_mapped)))
            self.gws.touch[:B, :S].copy_(torch.where(
                incoming_alive,
                selected_touch,
                torch.zeros_like(selected_touch)))
            self.gws.source[:B, :S].copy_(torch.where(
                incoming_alive,
                selected_source,
                torch.zeros_like(selected_source)))
            self.gws.source_confidence[:B, :S].copy_(torch.where(
                incoming_alive,
                selected_confidence,
                torch.zeros_like(selected_confidence)))

        k_src = state["memory_keys"].to(device=device, dtype=dtype) # [B, M, D]
        v_src = state["memory_values"].to(device=device, dtype=dtype) # [B, M, D]
        B = min(B_dst, B_src)
        filled_src = state["memory_filled"].to(device=device, dtype=torch.long)
        imp_src = state["memory_importance"].to(device=device, dtype=dtype)
        emo_src = state["memory_emotion"].to(device=device, dtype=dtype)
        src_src = state["memory_source"].to(device=device, dtype=torch.int8)
        src_conf_src = state["memory_source_confidence"].to(device=device, dtype=dtype)
        realm_src = state["memory_realm"].to(
            device=self.memory_realm.device,
            dtype=torch.int8)
        rew_abs_src = state["memory_reward_abs"].to(device=device, dtype=dtype)
        step_src = state["memory_steps"].to(device=device, dtype=torch.long) # [B, M]
        access_src = state["memory_last_access_steps"].to(device=device, dtype=torch.long)
        rehearsal_src = state["memory_last_rehearsal_steps"].to(device=device, dtype=torch.long)
        touch_src = state["memory_touch"].to(device=device, dtype=torch.long)
        merge_src = state["memory_merge_count"].to(device=device, dtype=torch.long)
        global_src = state["time_step"].to(device=device, dtype=torch.long)

        M_dst = int(self.memory_size)
        for b in range(B):
            for t in range(int(filled_src[b].item())):
                created = int(step_src[b, t].item())
                mapped_created = MapTimestamp(
                    created,
                    int(self.time_step[b].item()),
                    int(global_src[b].item()))
                n = int(self.memory_filled[b].item())
                target = -1
                if n > 0 and sourceBaseStep is not None:
                    same_created = (
                        (self.memory_steps[b, :n] == mapped_created)
                        & (self.memory_realm[b, :n] == realm_src[b, t]))
                    candidates = same_created.nonzero(as_tuple=False).flatten()
                    if candidates.numel() > 0:
                        similarities = F.cosine_similarity(
                            self.memory_keys[b, candidates],
                            k_src[b, t].unsqueeze(0),
                            dim=-1)
                        similarity, candidate_index = similarities.max(dim=0)
                        is_existing_delta = (
                            sourceBaseStep is not None
                            and created <= int(sourceBaseStep))
                        threshold = 0.50 if is_existing_delta else 0.999
                        if float(similarity.item()) >= threshold:
                            target = int(candidates[candidate_index].item())

                target_existed = target >= 0
                if target < 0:
                    if n < M_dst:
                        target = n
                        self.memory_filled[b] += 1
                    else:
                        age = (
                            self.time_step[b] - self.memory_steps[b]
                        ).clamp(min=0).float()
                        retention = (
                            -0.01 * age
                            + 0.50 * torch.log(self.memory_importance[b] + 1e-6)
                            + 0.10 * torch.log1p(self.memory_touch[b].float())
                            + 0.50 * torch.log(
                                self.memory_source_confidence[b] + 1e-6)
                            + 0.50 * torch.tanh(self.memory_reward_abs[b]))
                        target = int(torch.argmin(retention).item())

                mapped_access = max(
                    mapped_created,
                    MapTimestamp(
                        int(access_src[b, t].item()),
                        int(self.time_step[b].item()),
                        int(global_src[b].item())))
                mapped_rehearsal = max(
                    mapped_created,
                    MapTimestamp(
                        int(rehearsal_src[b, t].item()),
                        int(self.time_step[b].item()),
                        int(global_src[b].item())))
                current_rehearsal = int(
                    self.memory_last_rehearsal_steps[b, target].item())
                incoming_merges = int(merge_src[b, t].item())
                current_merges = int(self.memory_merge_count[b, target].item())
                destination_changed_since_fork = (
                    target_existed
                    and sourceBaseStep is not None
                    and created <= int(sourceBaseStep)
                    and current_rehearsal > int(sourceBaseStep))
                use_incoming_content = (
                    not target_existed
                    or (
                        not destination_changed_since_fork
                        and (
                            mapped_rehearsal > current_rehearsal
                            or (
                                mapped_rehearsal == current_rehearsal
                                and incoming_merges > current_merges))))
                if use_incoming_content:
                    self.memory_keys[b, target] = k_src[b, t]
                    self.memory_values[b, target] = v_src[b, t]
                    self.memory_emotion[b, target] = emo_src[b, t]
                    self.memory_realm[b, target] = realm_src[b, t]
                self.memory_importance[b, target] = (
                    torch.maximum(self.memory_importance[b, target], imp_src[b, t])
                    if target_existed else imp_src[b, t])
                if use_incoming_content:
                    self.memory_source[b, target] = (
                        MemoryType.SRC_MIXED
                        if target_existed
                        and int(self.memory_source[b, target].item())
                        != int(src_src[b, t].item())
                        else src_src[b, t])
                    self.memory_source_confidence[b, target] = (
                        torch.maximum(
                            self.memory_source_confidence[b, target],
                            src_conf_src[b, t])
                        if target_existed else src_conf_src[b, t])
                self.memory_reward_abs[b, target] = (
                    torch.maximum(self.memory_reward_abs[b, target], rew_abs_src[b, t])
                    if target_existed else rew_abs_src[b, t])
                self.memory_steps[b, target] = mapped_created
                self.memory_last_access_steps[b, target] = (
                    max(int(self.memory_last_access_steps[b, target].item()), mapped_access)
                    if target_existed else mapped_access)
                self.memory_last_rehearsal_steps[b, target] = (
                    max(int(self.memory_last_rehearsal_steps[b, target].item()), mapped_rehearsal)
                    if target_existed else mapped_rehearsal)
                self.memory_touch[b, target] = (
                    torch.maximum(self.memory_touch[b, target], touch_src[b, t])
                    if target_existed else touch_src[b, t])
                self.memory_merge_count[b, target] = (
                    torch.maximum(self.memory_merge_count[b, target], merge_src[b, t])
                    if target_existed else merge_src[b, t])

        sem = self.ltm.semantic
        k_src = state["ltm_sem_keys"].to(device=sem.keys.device, dtype=sem.keys.dtype) # [B, C, D]
        v_src = state["ltm_sem_vals"].to(device=sem.vals.device, dtype=sem.vals.dtype) # [B, C, D]
        B = min(B_dst, B_sem, int(sem.filled.size(0)))
        C_dst = int(sem.capacity)
        filled_src = state["ltm_sem_filled"].to(device=sem.keys.device, dtype=torch.long)
        pr_src = state["ltm_sem_prio"].to(device=sem.prio.device, dtype=sem.prio.dtype)
        src_src = state["ltm_sem_source"].to(device=sem.source.device, dtype=torch.int8)
        conf_src = state["ltm_sem_source_confidence"].to(device=sem.source_confidence.device, dtype=sem.source_confidence.dtype)
        realm_sem_src = state["ltm_sem_realm"].to(
            device=sem.realm.device,
            dtype=torch.int8)
        touch_src = state["ltm_sem_touch"].to(device=sem.touch.device, dtype=torch.long)
        step_sem_src = state["ltm_sem_step"].to(device=sem.step.device, dtype=torch.long)
        access_sem_src = state["ltm_sem_last_access_step"].to(device=sem.last_access_step.device, dtype=torch.long)
        rehearsal_sem_src = state["ltm_sem_last_rehearsal_step"].to(device=sem.last_rehearsal_step.device, dtype=torch.long)
        consolidation_sem_src = state["ltm_sem_consolidation_count"].to(device=sem.consolidation_count.device, dtype=torch.long)
        prototype_count_src = state["ltm_sem_prototype_count"].to(device=sem.prototype_count.device, dtype=sem.prototype_count.dtype)
        prototype_variance_src = state["ltm_sem_prototype_variance"].to(device=sem.prototype_variance.device, dtype=sem.prototype_variance.dtype)
        sem_global_src = state["ltm_sem_global_step"].to(
            device=sem.global_step.device,
            dtype=torch.long)

        for b in range(B):
            for t in range(int(filled_src[b].item())):
                created = int(step_sem_src[b, t].item())
                mapped_created = MapTimestamp(
                    created,
                    int(sem.global_step[b].item()),
                    int(sem_global_src[b].item()))
                n = int(sem.filled[b].item())
                target = -1
                if n > 0 and sourceBaseStep is not None:
                    candidates = (
                        (sem.step[b, :n] == mapped_created)
                        & (sem.realm[b, :n] == realm_sem_src[b, t])
                    ).nonzero(as_tuple=False).flatten()
                    if candidates.numel() > 0:
                        similarities = F.cosine_similarity(
                            sem.keys[b, candidates],
                            k_src[b, t].unsqueeze(0),
                            dim=-1)
                        similarity, candidate_index = similarities.max(dim=0)
                        is_existing_delta = (
                            sourceBaseStep is not None
                            and created <= int(sourceBaseStep))
                        threshold = 0.50 if is_existing_delta else 0.999
                        if float(similarity.item()) >= threshold:
                            target = int(candidates[candidate_index].item())

                target_existed = target >= 0
                if target < 0:
                    if n < C_dst:
                        target = n
                        sem.filled[b] += 1
                    else:
                        age = (
                            sem.global_step[b] - sem.step[b]
                        ).clamp(min=0).float()
                        retention = (
                            sem.prio[b]
                            * torch.exp(-0.001 * age)
                            * torch.log1p(sem.touch[b].float() + 1.0)
                            * sem.source_confidence[b])
                        target = int(torch.argmin(retention).item())

                mapped_access = max(
                    mapped_created,
                    MapTimestamp(
                        int(access_sem_src[b, t].item()),
                        int(sem.global_step[b].item()),
                        int(sem_global_src[b].item())))
                mapped_rehearsal = max(
                    mapped_created,
                    MapTimestamp(
                        int(rehearsal_sem_src[b, t].item()),
                        int(sem.global_step[b].item()),
                        int(sem_global_src[b].item())))
                current_rehearsal = int(sem.last_rehearsal_step[b, target].item())
                incoming_count = float(prototype_count_src[b, t].item())
                current_count = float(sem.prototype_count[b, target].item())
                destination_changed_since_fork = (
                    target_existed
                    and sourceBaseStep is not None
                    and created <= int(sourceBaseStep)
                    and current_rehearsal > int(sourceBaseStep))
                use_incoming_statistics = (
                    not target_existed
                    or (
                        not destination_changed_since_fork
                        and (
                            mapped_rehearsal > current_rehearsal
                            or (
                                mapped_rehearsal == current_rehearsal
                                and incoming_count > current_count))))
                if use_incoming_statistics:
                    sem.keys[b, target] = k_src[b, t]
                    sem.vals[b, target] = v_src[b, t]
                    sem.prototype_count[b, target] = prototype_count_src[b, t]
                    sem.prototype_variance[b, target] = prototype_variance_src[b, t]
                    sem.realm[b, target] = realm_sem_src[b, t]
                sem.prio[b, target] = (
                    torch.maximum(sem.prio[b, target], pr_src[b, t])
                    if target_existed else pr_src[b, t])
                if use_incoming_statistics:
                    sem.source[b, target] = (
                        MemoryType.SRC_MIXED
                        if target_existed
                        and int(sem.source[b, target].item()) != int(src_src[b, t].item())
                        else src_src[b, t])
                    sem.source_confidence[b, target] = (
                        torch.maximum(sem.source_confidence[b, target], conf_src[b, t])
                        if target_existed else conf_src[b, t])
                sem.touch[b, target] = (
                    torch.maximum(sem.touch[b, target], touch_src[b, t])
                    if target_existed else touch_src[b, t])
                sem.step[b, target] = mapped_created
                sem.last_access_step[b, target] = (
                    max(int(sem.last_access_step[b, target].item()), mapped_access)
                    if target_existed else mapped_access)
                sem.last_rehearsal_step[b, target] = (
                    max(int(sem.last_rehearsal_step[b, target].item()), mapped_rehearsal)
                    if target_existed else mapped_rehearsal)
                sem.consolidation_count[b, target] = (
                    torch.maximum(
                        sem.consolidation_count[b, target],
                        consolidation_sem_src[b, t])
                    if target_existed else consolidation_sem_src[b, t])

        epi = self.ltm.episodic
        k_src = state["ltm_epi_keys"].to(device=epi.keys.device, dtype=epi.keys.dtype) # [B, C, D]
        ks_src = state["ltm_epi_state_keys"].to(device=epi.state_keys.device, dtype=epi.state_keys.dtype) # [B, C, D]
        v_src = state["ltm_epi_vals"].to(device=epi.vals.device, dtype=epi.vals.dtype) # [B, C, D]
        B = min(B_dst, B_epi, int(epi.filled.size(0)))
        C_dst = int(epi.capacity)
        filled_src = state["ltm_epi_filled"].to(device=epi.keys.device, dtype=torch.long)
        pr_src = state["ltm_epi_prio"].to(device=epi.prio.device, dtype=epi.prio.dtype)
        rw_src = state["ltm_epi_rew"].to(device=epi.rew.device, dtype=epi.rew.dtype)
        rw_abs_src = state["ltm_epi_rew_abs"].to(device=epi.rew_abs.device, dtype=epi.rew_abs.dtype)
        src_src = state["ltm_epi_source"].to(device=epi.source.device, dtype=torch.int8)
        conf_epi_src = state["ltm_epi_source_confidence"].to(device=epi.source_confidence.device, dtype=epi.source_confidence.dtype)
        realm_epi_src = state["ltm_epi_realm"].to(
            device=epi.realm.device,
            dtype=torch.int8)
        agency_epi_src = state["ltm_epi_agency_id"].to(
            device=epi.agency_id.device,
            dtype=torch.int8)
        motion_layer_epi_src = state["ltm_epi_motion_layer_prob"].to(
            device=epi.motion_layer_prob.device,
            dtype=epi.motion_layer_prob.dtype)
        surface_parent_epi_src = state[
            "ltm_epi_observed_surface_parent_index"].to(
                device=epi.observed_surface_parent_index.device,
                dtype=torch.long)
        surface_uv_epi_src = state["ltm_epi_surface_uv"].to(
            device=epi.surface_uv.device,
            dtype=epi.surface_uv.dtype)
        surface_uv_confidence_epi_src = state[
            "ltm_epi_surface_uv_confidence"].to(
                device=epi.surface_uv_confidence.device,
                dtype=epi.surface_uv_confidence.dtype)
        verification_epi_src = state["ltm_epi_verification"].to(
            device=epi.verification.device,
            dtype=epi.verification.dtype)
        content_change_epi_src = state["ltm_epi_content_change"].to(
            device=epi.content_change.device,
            dtype=epi.content_change.dtype)
        touch_epi_src = state["ltm_epi_touch"].to(device=epi.touch.device, dtype=torch.long)
        step_epi_src = state["ltm_epi_step"].to(device=epi.step.device, dtype=torch.long)
        access_epi_src = state["ltm_epi_last_access_step"].to(device=epi.last_access_step.device, dtype=torch.long)
        rehearsal_epi_src = state["ltm_epi_last_rehearsal_step"].to(device=epi.last_rehearsal_step.device, dtype=torch.long)
        consolidation_epi_src = state["ltm_epi_consolidation_count"].to(device=epi.consolidation_count.device, dtype=torch.long)
        episode_epi_src = state["ltm_epi_episode_id"].to(device=epi.episode_id.device, dtype=torch.long)
        event_epi_src = state["ltm_epi_event_id"].to(device=epi.event_id.device, dtype=torch.long)
        epi_global_src = state["ltm_epi_global_step"].to(
            device=epi.global_step.device,
            dtype=torch.long)

        for b in range(B):
            episode_remap: Dict[int, int] = {}
            destination_filled = int(epi.filled[b].item())
            largest_destination_episode = int(
                epi.current_episode_id[b].item())
            if destination_filled > 0:
                largest_destination_episode = max(
                    largest_destination_episode,
                    int(epi.episode_id[b, :destination_filled].max().item()))
            next_imported_episode = largest_destination_episode + 1
            for t in range(int(filled_src[b].item())):
                created = int(step_epi_src[b, t].item())
                mapped_created = MapTimestamp(
                    created,
                    int(epi.global_step[b].item()),
                    int(epi_global_src[b].item()))
                incoming_episode = int(episode_epi_src[b, t].item())
                if sourceBaseStep is None:
                    if incoming_episode not in episode_remap:
                        episode_remap[incoming_episode] = next_imported_episode
                        next_imported_episode += 1
                    incoming_episode = episode_remap[incoming_episode]
                incoming_event = int(event_epi_src[b, t].item())
                incoming_source = int(src_src[b, t].item())
                n = int(epi.filled[b].item())
                target = -1
                if n > 0 and sourceBaseStep is not None:
                    same_event = (
                        (epi.episode_id[b, :n] == incoming_episode)
                        & (epi.event_id[b, :n] == incoming_event)
                        & (epi.realm[b, :n] == realm_epi_src[b, t]))
                    is_existing_delta = (
                        sourceBaseStep is not None
                        and created <= int(sourceBaseStep))
                    if not is_existing_delta:
                        incoming_imagined = incoming_source == MemoryType.SRC_IMAGINE
                        same_event = same_event & (
                            (epi.source[b, :n] == MemoryType.SRC_IMAGINE)
                            == incoming_imagined)
                    candidates = same_event.nonzero(as_tuple=False).flatten()
                    if candidates.numel() > 0:
                        similarities = F.cosine_similarity(
                            epi.keys[b, candidates],
                            k_src[b, t].unsqueeze(0),
                            dim=-1)
                        _, candidate_index = similarities.max(dim=0)
                        target = int(candidates[candidate_index].item())

                target_existed = target >= 0
                if target < 0:
                    if n < C_dst:
                        target = n
                        epi.filled[b] += 1
                    else:
                        age = (
                            epi.global_step[b] - epi.step[b]
                        ).clamp(min=0).float()
                        retention = (
                            (epi.prio[b] + 0.5 * epi.rew_abs[b])
                            * torch.exp(-0.01 * age)
                            * torch.log1p(epi.touch[b].float() + 1.0)
                            * epi.source_confidence[b])
                        target = int(torch.argmin(retention).item())
                    epi.slot_generation[b, target] += 1

                mapped_access = max(
                    mapped_created,
                    MapTimestamp(
                        int(access_epi_src[b, t].item()),
                        int(epi.global_step[b].item()),
                        int(epi_global_src[b].item())))
                mapped_rehearsal = max(
                    mapped_created,
                    MapTimestamp(
                        int(rehearsal_epi_src[b, t].item()),
                        int(epi.global_step[b].item()),
                        int(epi_global_src[b].item())))
                current_rehearsal = int(
                    epi.last_rehearsal_step[b, target].item())
                destination_changed_since_fork = (
                    target_existed
                    and sourceBaseStep is not None
                    and created <= int(sourceBaseStep)
                    and current_rehearsal > int(sourceBaseStep))
                use_incoming_content = (
                    not target_existed
                    or not destination_changed_since_fork)
                if use_incoming_content:
                    epi.keys[b, target] = k_src[b, t]
                    epi.state_keys[b, target] = ks_src[b, t]
                    epi.vals[b, target] = v_src[b, t]
                    epi.rew[b, target] = rw_src[b, t]
                    epi.realm[b, target] = realm_epi_src[b, t]
                    epi.agency_id[b, target] = agency_epi_src[b, t]
                    epi.motion_layer_prob[b, target] = (
                        motion_layer_epi_src[b, t])
                    epi.observed_surface_parent_index[b, target] = (
                        surface_parent_epi_src[b, t])
                    epi.surface_uv[b, target] = surface_uv_epi_src[b, t]
                    epi.surface_uv_confidence[b, target] = (
                        surface_uv_confidence_epi_src[b, t])
                    epi.verification[b, target] = verification_epi_src[b, t]
                    epi.content_change[b, target] = (
                        content_change_epi_src[b, t])
                epi.prio[b, target] = (
                    torch.maximum(epi.prio[b, target], pr_src[b, t])
                    if target_existed else pr_src[b, t])
                epi.rew_abs[b, target] = (
                    torch.maximum(epi.rew_abs[b, target], rw_abs_src[b, t])
                    if target_existed else rw_abs_src[b, t])
                if use_incoming_content:
                    epi.source[b, target] = (
                        MemoryType.SRC_MIXED
                        if target_existed
                        and int(epi.source[b, target].item()) != int(src_src[b, t].item())
                        else src_src[b, t])
                    epi.source_confidence[b, target] = (
                        torch.maximum(
                            epi.source_confidence[b, target],
                            conf_epi_src[b, t])
                        if target_existed else conf_epi_src[b, t])
                epi.touch[b, target] = (
                    torch.maximum(epi.touch[b, target], touch_epi_src[b, t])
                    if target_existed else touch_epi_src[b, t])
                epi.step[b, target] = mapped_created
                epi.last_access_step[b, target] = (
                    max(int(epi.last_access_step[b, target].item()), mapped_access)
                    if target_existed else mapped_access)
                epi.last_rehearsal_step[b, target] = (
                    max(int(epi.last_rehearsal_step[b, target].item()), mapped_rehearsal)
                    if target_existed else mapped_rehearsal)
                epi.consolidation_count[b, target] = (
                    torch.maximum(
                        epi.consolidation_count[b, target],
                        consolidation_epi_src[b, t])
                    if target_existed else consolidation_epi_src[b, t])
                epi.episode_id[b, target] = incoming_episode
                epi.event_id[b, target] = event_epi_src[b, t]
                epi.slot_generation[b, target].clamp_min_(1)



        epi.RebuildSequenceLinks()


        sym = self.sym_mem
        Pk = state["sym_mem_P_keys"].to(device=sym.P_keys.device, dtype=sym.P_keys.dtype)
        Pv = state["sym_mem_P_vals"].to(device=sym.P_vals.device, dtype=sym.P_vals.dtype)
        pr = state["sym_mem_prio"].to(device=sym.prio.device, dtype=sym.prio.dtype)
        src = state["sym_mem_source"].to(device=sym.source.device, dtype=torch.int8)
        conf = state["sym_mem_source_confidence"].to(
            device=sym.source_confidence.device,
            dtype=sym.source_confidence.dtype)
        sym_created_src = state["sym_mem_step"].to(
            device=sym.step.device,
            dtype=torch.long)
        sym_access_src = state["sym_mem_last_access_step"].to(
            device=sym.last_access_step.device,
            dtype=torch.long)
        sym_rehearsal_src = state["sym_mem_last_rehearsal_step"].to(
            device=sym.last_rehearsal_step.device,
            dtype=torch.long)
        sym_touch_src = state["sym_mem_touch"].to(
            device=sym.touch.device,
            dtype=torch.long)
        sym_global_src = state["sym_mem_global_step"].to(
            device=sym.global_step.device,
            dtype=torch.long)
        sym_filled_src = state["sym_mem_filled"].to(device=sym.filled.device, dtype=torch.long)
        B_sym_merge = min(B_dst, B_src)
        for b in range(B_sym_merge):
            for i in range(int(sym_filled_src[b].item())):
                created = int(sym_created_src[b, i].item())
                mapped_created = MapTimestamp(
                    created,
                    int(sym.global_step[b].item()),
                    int(sym_global_src[b].item()))
                n = int(sym.filled[b].item())
                target = -1
                if n > 0 and sourceBaseStep is not None:
                    candidates = (
                        sym.created_step[b, :n] == mapped_created
                    ).nonzero(as_tuple=False).flatten()
                    if candidates.numel() > 0:
                        similarities = F.cosine_similarity(
                            sym.P_keys[b, candidates],
                            Pk[b, i].unsqueeze(0),
                            dim=-1)
                        similarity, candidate_index = similarities.max(dim=0)
                        is_existing_delta = (
                            sourceBaseStep is not None
                            and created <= int(sourceBaseStep))
                        threshold = 0.50 if is_existing_delta else 0.999
                        if float(similarity.item()) >= threshold:
                            target = int(candidates[candidate_index].item())

                target_existed = target >= 0
                if target < 0:
                    if n < sym.capacity:
                        target = n
                        sym.filled[b] += 1
                    else:
                        age = (
                            sym.global_step[b] - sym.created_step[b]
                        ).clamp(min=0).float()
                        retention = (
                            sym.prio[b]
                            * torch.exp(-0.01 * age)
                            * sym.source_confidence[b])
                        target = int(torch.argmin(retention).item())

                mapped_access = max(
                    mapped_created,
                    MapTimestamp(
                        int(sym_access_src[b, i].item()),
                        int(sym.global_step[b].item()),
                        int(sym_global_src[b].item())))
                mapped_rehearsal = max(
                    mapped_created,
                    MapTimestamp(
                        int(sym_rehearsal_src[b, i].item()),
                        int(sym.global_step[b].item()),
                        int(sym_global_src[b].item())))
                current_rehearsal = int(
                    sym.last_rehearsal_step[b, target].item())
                destination_changed_since_fork = (
                    target_existed
                    and sourceBaseStep is not None
                    and created <= int(sourceBaseStep)
                    and current_rehearsal > int(sourceBaseStep))
                use_incoming_content = (
                    not target_existed
                    or not destination_changed_since_fork)
                if use_incoming_content:
                    sym.P_keys[b, target] = Pk[b, i]
                    sym.P_vals[b, target] = Pv[b, i]
                sym.prio[b, target] = (
                    torch.maximum(sym.prio[b, target], pr[b, i])
                    if target_existed else pr[b, i])
                if use_incoming_content:
                    sym.source[b, target] = (
                        MemoryType.SRC_MIXED
                        if target_existed
                        and int(sym.source[b, target].item()) != int(src[b, i].item())
                        else src[b, i])
                    sym.source_confidence[b, target] = (
                        torch.maximum(sym.source_confidence[b, target], conf[b, i])
                        if target_existed else conf[b, i])
                sym.created_step[b, target] = mapped_created
                sym.last_access_step[b, target] = (
                    max(int(sym.last_access_step[b, target].item()), mapped_access)
                    if target_existed else mapped_access)
                sym.last_rehearsal_step[b, target] = (
                    max(int(sym.last_rehearsal_step[b, target].item()), mapped_rehearsal)
                    if target_existed else mapped_rehearsal)
                sym.touch[b, target] = (
                    torch.maximum(sym.touch[b, target], sym_touch_src[b, i])
                    if target_existed else sym_touch_src[b, i])




        if sourceBaseStep is None:
            src_usage = {
                field: state[field].to(
                    device=destination.device,
                    dtype=destination.dtype)
                for field, destination in usage_fields}

            dst_alpha = usage.success_alpha.clone()
            dst_beta = usage.success_beta.clone()
            src_alpha = src_usage["usage_success_alpha"]
            src_beta = src_usage["usage_success_beta"]
            dst_trials = (dst_alpha + dst_beta - 2.0).clamp_min(0.0)
            src_trials = (src_alpha + src_beta - 2.0).clamp_min(0.0)

            dst_count = usage.parameter_observations.clone()
            src_count = src_usage["usage_parameter_observations"]
            total_count = dst_count + src_count
            dst_mu = usage.param_mu.clone()
            src_mu = src_usage["usage_param_mu"]
            combined_mu = (
                dst_count.unsqueeze(-1) * dst_mu
                + src_count.unsqueeze(-1) * src_mu
            ) / total_count.clamp_min(1.0).unsqueeze(-1)
            combined_mu = torch.where(
                (total_count > 0).unsqueeze(-1),
                combined_mu,
                dst_mu)

            dst_var = usage.param_logvar.exp()
            src_var = src_usage["usage_param_logvar"].exp()
            combined_var = (
                dst_count.unsqueeze(-1) * (
                    dst_var + (dst_mu - combined_mu).square())
                + src_count.unsqueeze(-1) * (
                    src_var + (src_mu - combined_mu).square())
            ) / total_count.clamp_min(1.0).unsqueeze(-1)
            combined_var = torch.where(
                (total_count > 0).unsqueeze(-1),
                combined_var,
                dst_var)

            evidence_dst = 1.0 + dst_trials + dst_count
            evidence_src = 1.0 + src_trials + src_count
            evidence_total = evidence_dst + evidence_src
            weighted = lambda current, incoming: (
                evidence_dst.unsqueeze(-1) * current
                + evidence_src.unsqueeze(-1) * incoming
            ) / evidence_total.unsqueeze(-1)

            usage.applicable.copy_(torch.maximum(
                usage.applicable,
                src_usage["usage_applicable"]))
            usage.default_params.copy_(weighted(
                usage.default_params,
                src_usage["usage_default_params"]))
            usage.expected_dx.copy_(weighted(
                usage.expected_dx,
                src_usage["usage_expected_dx"]))
            usage.success_alpha.copy_(
                1.0 + (dst_alpha - 1.0).clamp_min(0.0)
                + (src_alpha - 1.0).clamp_min(0.0))
            usage.success_beta.copy_(
                1.0 + (dst_beta - 1.0).clamp_min(0.0)
                + (src_beta - 1.0).clamp_min(0.0))
            usage.param_mu.copy_(combined_mu)
            usage.param_logvar.copy_(combined_var.clamp_min(1e-6).log())
            usage.parameter_observations.copy_(total_count)
            usage.attribute_centroid.copy_(weighted(
                usage.attribute_centroid,
                src_usage["usage_attribute_centroid"]))

            descriptor_dst_weight = evidence_dst.sum(dim=1, keepdim=True)
            descriptor_src_weight = evidence_src.sum(dim=1, keepdim=True)
            usage.instance_descriptors.copy_(F.normalize(
                descriptor_dst_weight * usage.instance_descriptors
                + descriptor_src_weight * src_usage["usage_instance_descriptors"],
                dim=-1))

        self.memory_version.add_(1)

    @torch.no_grad()
    def MergeMemoryDelta(self, delta: Dict[str, Any]) -> None:
        expected = ("state", "base_step", "new_step", "kind")
        if type(delta) is not dict or tuple(delta) != expected:
            raise TypeError("memory delta does not match its transaction envelope")
        if not all(torch.is_tensor(delta[name]) for name in expected[1:]):
            raise TypeError("memory delta transaction metadata must be tensors")
        if delta["base_step"].numel() != 1 or delta["new_step"].numel() != 1:
            raise ValueError("memory delta transaction steps must be scalar tensors")
        kind = int(delta["kind"].item())
        if kind not in (1, 2):
            raise ValueError("memory delta transaction kind is invalid")
        signature = self.merged_delta_signature.new_tensor([
            int(delta["base_step"].item()),
            int(delta["new_step"].item()),
            kind,])
        if torch.equal(signature, self.merged_delta_signature):
            return
        self.MergeMemoryState(
            delta["state"],
            sourceBaseStep=int(delta["base_step"].item()),
            sourceNewStep=int(delta["new_step"].item()))
        self.merged_delta_signature.copy_(signature)

    def MemoryDeltaRowsFingerprint(
        self,
        delta: Dict[str, Any],
        destinationRows: torch.Tensor,
        contractId: str,
        modelSignature: str,
        transactionId: str,
        timelineVersion: int,
        episodeVersion: int,
    ) -> str:
        digest = hashlib.sha256()
        identity = (
            contractId,
            modelSignature,
            transactionId,
            int(timelineVersion),
            int(episodeVersion),
            tuple(int(row) for row in destinationRows.tolist()),
            int(delta["base_step"].item()),
            int(delta["new_step"].item()),
            int(delta["kind"].item()),)
        digest.update(repr(identity).encode("utf-8"))
        for name in self.FULL_MEMORY_STATE_FIELDS:
            tensor = delta["state"][name].detach().to(
                device=torch.device("cpu")).contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def MergeMemoryDeltaRows(
        self,
        delta: Dict[str, Any],
        destinationRows: torch.Tensor,
        contractId: str,
        modelSignature: str,
        transactionId: str,
        timelineVersion: int,
        episodeVersion: int,
    ) -> None:
        expected = ("state", "base_step", "new_step", "kind")
        if type(delta) is not dict or tuple(delta) != expected:
            raise TypeError("memory delta does not match its transaction envelope")
        if not all(torch.is_tensor(delta[name]) for name in expected[1:]):
            raise TypeError("memory delta transaction metadata must be tensors")
        if any(delta[name].numel() != 1 for name in expected[1:]):
            raise ValueError("memory delta transaction metadata must be scalar")
        base_step = int(delta["base_step"].item())
        new_step = int(delta["new_step"].item())
        kind = int(delta["kind"].item())
        if new_step < base_step:
            raise ValueError("delta new step precedes its base step")
        if kind not in (1, 2):
            raise ValueError("memory delta transaction kind is invalid")
        state = delta["state"]
        if type(state) is not dict or tuple(state) != self.FULL_MEMORY_STATE_FIELDS:
            raise TypeError("memory delta state fields do not match the current schema")
        if not all(torch.is_tensor(value) for value in state.values()):
            raise TypeError("every memory delta state field must be a tensor")
        if type(contractId) is not str or not contractId.strip():
            raise ValueError("contractId must be a non-empty string")
        signature = self.ValidateModelSignature(modelSignature)
        if type(transactionId) is not str or not transactionId.strip():
            raise ValueError("transactionId must be a non-empty string")
        if type(timelineVersion) is not int or timelineVersion < 0:
            raise ValueError("timelineVersion must be a non-negative integer")
        if type(episodeVersion) is not int or episodeVersion < 0:
            raise ValueError("episodeVersion must be a non-negative integer")
        if not torch.is_tensor(destinationRows):
            raise TypeError("destinationRows must be a tensor")
        if destinationRows.dtype != torch.long:
            raise TypeError("destinationRows must use torch.long")
        if destinationRows.device != self.device:
            raise ValueError("destinationRows must be on the memory device")
        if destinationRows.dim() != 1 or destinationRows.numel() < 1:
            raise ValueError("destinationRows must be a non-empty vector")
        if state["h_state"].dim() < 1:
            raise ValueError("memory delta state must have a batch dimension")
        source_batch = int(state["h_state"].size(0))
        destination_batch = int(self.h_state.size(0))
        if source_batch != int(destinationRows.numel()):
            raise ValueError("source batch must match destinationRows")
        if (
            bool((destinationRows < 0).any().item())
            or bool((destinationRows >= destination_batch).any().item())
        ):
            raise ValueError("destinationRows contains an out-of-range row")
        if int(torch.unique(destinationRows).numel()) != source_batch:
            raise ValueError("destinationRows must be unique")
        if (
            self._row_merge_contract_id is not None
            and self._row_merge_contract_id != contractId
        ):
            raise ValueError("row merge contract identity mismatch")
        if (
            self._row_merge_model_signature is not None
            and self._row_merge_model_signature != signature
        ):
            raise ValueError("row merge model signature mismatch")
        for row in destinationRows.tolist():
            previous = self._row_merge_versions.get(int(row))
            if previous is not None and (
                timelineVersion < previous[0]
                or (
                    timelineVersion == previous[0]
                    and episodeVersion < previous[1])
            ):
                raise ValueError("row merge version is stale")
        fingerprint = self.MemoryDeltaRowsFingerprint(
            delta,
            destinationRows,
            contractId,
            signature,
            transactionId,
            timelineVersion,
            episodeVersion)
        previous_fingerprint = self._row_merge_transactions.get(transactionId)
        if previous_fingerprint is not None:
            if previous_fingerprint != fingerprint:
                raise ValueError("row merge transaction identity mismatch")
            return
        current = self.ExportState()
        remapped = {}
        for name in self.FULL_MEMORY_STATE_FIELDS:
            destination = current[name]
            source = state[name]
            if name in self.UNBATCHED_MEMORY_STATE_FIELDS:
                if tuple(source.shape) != tuple(destination.shape):
                    raise ValueError(f"memory delta field {name} has an invalid shape")
                remapped[name] = destination.clone()
                continue
            expected_shape = (source_batch, *tuple(destination.shape[1:]))
            if tuple(source.shape) != expected_shape:
                raise ValueError(f"memory delta field {name} has an invalid shape")
            mapped = destination.clone()
            mapped.index_copy_(
                0,
                destinationRows,
                source.to(device=mapped.device, dtype=mapped.dtype))
            remapped[name] = mapped
        selected = torch.zeros(
            destination_batch,
            device=self.device,
            dtype=torch.bool)
        selected[destinationRows] = True
        frozen = self.CaptureFrozenRows(selected)
        try:
            self.MergeMemoryState(
                remapped,
                sourceBaseStep=base_step,
                sourceNewStep=new_step)
        except Exception:
            transactions = dict(self._row_merge_transactions)
            versions = dict(self._row_merge_versions)
            self.ImportState(current)
            self._row_merge_transactions = transactions
            self._row_merge_versions = versions
            raise
        self.RestoreFrozenRows(frozen, selected)
        self._row_merge_contract_id = contractId
        self._row_merge_model_signature = signature
        self._row_merge_transactions[transactionId] = fingerprint
        for row in destinationRows.tolist():
            self._row_merge_versions[int(row)] = (
                timelineVersion,
                episodeVersion)

    @torch.no_grad()
    def ImportCurrentState(
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
        if B < 1:
            raise ValueError("memory-state batch dimension must be positive")

        def CurrentTargets():
            targets = [
                ("last_compress_step", self.last_compress_step),
                ("h_state", self.h_state),
                ("fast_weights", self.fast_weights),
                ("ns_prev_P_post", self.ns_prev_P_post),
                ("ns_penalty_vec", self.ns_penalty_vec),
                ("pattern_usage", self.pattern_usage),
                ("previous_attention", self.previous_attention),
                ("previous_intent", self.previous_intent),
                ("previous_object_summary", self.previous_object_summary),
                ("previous_motion_token", self.previous_motion_token),
                ("previous_ontology_context", self.previous_ontology_context),
                ("event_age", self.event_age),
                ("has_previous_event", self.has_previous_event),
                ("gws_global_step", self.gws.global_step),
                ("gws_keys", self.gws.keys),
                ("gws_vals", self.gws.vals),
                ("gws_priority", self.gws.priority),
                ("gws_ttl", self.gws.ttl),
                ("gws_created_step", self.gws.created_step),
                ("gws_last_step", self.gws.last_step),
                ("gws_last_rehearsal_step", self.gws.last_rehearsal_step),
                ("gws_touch", self.gws.touch),
                ("gws_source", self.gws.source),
                ("gws_source_confidence", self.gws.source_confidence),]
            if includeDurable:
                targets.extend(self.DurableStateTensors().items())
            return targets

        unbatched_fields = {
            "pattern_usage",
            "memory_version",
            "merged_delta_signature",
            "usage_applicable",
            "usage_default_params",
            "usage_expected_dx",
            "usage_success_alpha",
            "usage_success_beta",
            "usage_param_mu",
            "usage_param_logvar",
            "usage_parameter_observations",
            "usage_instance_descriptors",
            "usage_attribute_centroid",}
        for field, dst in CurrentTargets():
            src = state[field]
            expected_shape = (
                tuple(dst.shape)
                if field in unbatched_fields
                else (B, *tuple(dst.shape[1:])))
            if tuple(src.shape) != expected_shape:
                raise ValueError(
                    f"memory-state field {field!r} has shape {tuple(src.shape)}, "
                    f"expected {expected_shape}")

        if includeDurable:
            filled_ranges = (
                ("memory_filled", self.memory_size),
                ("ltm_sem_filled", self.ltm.semantic.capacity),
                ("ltm_epi_filled", self.ltm.episodic.capacity),
                ("sym_mem_filled", self.sym_mem.capacity),)
            for field, capacity in filled_ranges:
                filled = state[field]
                if bool(((filled < 0) | (filled > capacity)).any().item()):
                    raise ValueError(
                        f"memory-state field {field!r} is outside its capacity")

        self.EnsureB(B)
        for field, dst in CurrentTargets():
            dst.copy_(state[field].to(device=dst.device, dtype=dst.dtype))

        self.ResetInternalLoss()
        self.pending.clear()
        if includeDurable:
            self._row_merge_transactions.clear()
            self._row_merge_versions.clear()

    @torch.no_grad()
    def ImportState(self, state: Dict[str, torch.Tensor]) -> None:
        self.ImportCurrentState(state, includeDurable=True)


    @torch.no_grad()
    def ResetSteps(self, resetGlobal: bool = True) -> None:
        self.merged_delta_signature.fill_(-1)
        self._row_merge_transactions.clear()
        self._row_merge_versions.clear()
        if hasattr(self, "time_step") and isinstance(self.time_step, torch.Tensor):
            self.time_step.zero_()
        if hasattr(self, "last_compress_step") and isinstance(self.last_compress_step, torch.Tensor):
            self.last_compress_step.zero_()
        if hasattr(self, "memory_steps") and isinstance(self.memory_steps, torch.Tensor) and self.memory_steps.numel() > 0:
            self.memory_steps.zero_()
            self.memory_last_access_steps.zero_()
            self.memory_last_rehearsal_steps.zero_()

        gws = getattr(self, "gws", None)
        if gws is not None:
            if hasattr(gws, "last_step") and isinstance(gws.last_step, torch.Tensor) and gws.last_step.numel() > 0:
                gws.created_step.zero_()
                gws.last_step.zero_()
                gws.last_rehearsal_step.zero_()
            if resetGlobal and hasattr(gws, "global_step") and isinstance(gws.global_step, torch.Tensor):
                gws.global_step.zero_()

        ltm = getattr(self, "ltm", None)
        if ltm is not None:
            sem = ltm.semantic
            epi = ltm.episodic

            if hasattr(sem, "step") and isinstance(sem.step, torch.Tensor) and sem.step.numel() > 0:
                sem.step.zero_()
                sem.last_access_step.zero_()
                sem.last_rehearsal_step.zero_()
            if resetGlobal and hasattr(sem, "global_step") and isinstance(sem.global_step, torch.Tensor):
                sem.global_step.zero_()

            if hasattr(epi, "step") and isinstance(epi.step, torch.Tensor) and epi.step.numel() > 0:
                epi.step.zero_()
                epi.last_access_step.zero_()
                epi.last_rehearsal_step.zero_()
            if resetGlobal and hasattr(epi, "global_step") and isinstance(epi.global_step, torch.Tensor):
                epi.global_step.zero_()

        sym = getattr(self, "sym_mem", None)
        if sym is not None:
            if hasattr(sym, "step") and isinstance(sym.step, torch.Tensor) and sym.step.numel() > 0:
                sym.step.zero_()
                sym.last_access_step.zero_()
                sym.last_rehearsal_step.zero_()
            if resetGlobal and hasattr(sym, "global_step") and isinstance(sym.global_step, torch.Tensor):
                sym.global_step.zero_()

    @torch.no_grad()
    def ReorderMemorySteps(self) -> None:
        self.merged_delta_signature.fill_(-1)
        self._row_merge_transactions.clear()
        self._row_merge_versions.clear()

        B, M = self.memory_steps.shape
        slots = torch.arange(M, device=self.memory_steps.device).view(1, M)
        kv_valid = slots < self.memory_filled.view(B, 1)
        gws_valid = ((self.gws.ttl > 0) & (self.gws.priority > 0))

        sem = self.ltm.semantic
        sem_slots = torch.arange(sem.capacity, device=sem.step.device).view(1, sem.capacity)
        sem_valid = sem_slots < sem.filled.view(B, 1)

        epi = self.ltm.episodic
        epi_slots = torch.arange(epi.capacity, device=epi.step.device).view(1, epi.capacity)
        epi_valid = epi_slots < epi.filled.view(B, 1)

        sym = self.sym_mem
        sym_slots = torch.arange(sym.capacity, device=sym.step.device).view(1, sym.capacity)
        sym_valid = sym_slots < sym.filled.view(B, 1)

        axes = (
            (self.memory_steps, self.memory_last_access_steps,
             self.memory_last_rehearsal_steps, kv_valid),
            (self.gws.created_step, self.gws.last_step,
             self.gws.last_rehearsal_step, gws_valid),
            (sem.step, sem.last_access_step, sem.last_rehearsal_step, sem_valid),
            (epi.step, epi.last_access_step, epi.last_rehearsal_step, epi_valid),
            (sym.step, sym.last_access_step, sym.last_rehearsal_step, sym_valid),)

        largest = torch.iinfo(self.memory_steps.dtype).max
        earliest_candidates = []
        validity = []
        for created, _, _, valid in axes:
            earliest_candidates.append(torch.where(
                valid,
                created,
                torch.full_like(created, largest)).min(dim=1).values)
            validity.append(valid.any(dim=1))
        any_valid = torch.stack(validity, dim=1).any(dim=1)
        earliest = torch.stack(earliest_candidates, dim=1).min(dim=1).values
        offset = torch.where(
            any_valid,
            (earliest - 1).clamp_min(0),
            torch.zeros_like(earliest))

        latest_rebased = []
        for created, accessed, rehearsed, valid in axes:
            created_new = (created - offset.unsqueeze(1)).clamp_min(0)
            accessed_new = (
                torch.maximum(accessed, created) - offset.unsqueeze(1)
            ).clamp_min(0)
            rehearsed_new = (
                torch.maximum(rehearsed, created) - offset.unsqueeze(1)
            ).clamp_min(0)
            created.copy_(torch.where(valid, created_new, torch.zeros_like(created_new)))
            accessed.copy_(torch.where(valid, accessed_new, torch.zeros_like(accessed_new)))
            rehearsed.copy_(torch.where(valid, rehearsed_new, torch.zeros_like(rehearsed_new)))
            latest_rebased.append(torch.maximum(
                accessed_new.masked_fill(~valid, 0).max(dim=1).values,
                rehearsed_new.masked_fill(~valid, 0).max(dim=1).values))

        global_steps = (
            self.time_step,
            self.gws.global_step,
            sem.global_step,
            epi.global_step,
            sym.global_step,)
        global_now = torch.stack(global_steps, dim=1).max(dim=1).values
        latest = torch.stack(latest_rebased, dim=1).max(dim=1).values
        unified_global = torch.where(
            any_valid,
            torch.maximum((global_now - offset).clamp_min(0), latest),
            torch.zeros_like(global_now))
        for global_step in global_steps:
            global_step.copy_(unified_global)
        self.last_compress_step.copy_(unified_global)




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
        K = 8
        realm = torch.zeros(B, K, 5, device=device, dtype=dtype)
        realm[..., ONTOLOGY_REALM_EXTERNAL] = 1.0
        agency = torch.zeros(B, K, 5, device=device, dtype=dtype)
        agency[..., 1] = 1.0
        layer_agency = torch.zeros(B, K, 5, 5, device=device, dtype=dtype)
        layer_agency[..., 4] = 1.0
        self_part = torch.zeros(
            B, K, ModuleDim.PstSelfPartSemanticDim,
            device=device, dtype=dtype)
        surface_parent = torch.zeros(B, K, K + 1, device=device, dtype=dtype)
        surface_parent[..., -1] = 1.0
        auxiliary = {
            "PerceptualPresence": torch.ones(
                B, K, device=device, dtype=dtype),
            "RealmProb": realm,
            "AgencyProb": agency,
            "MotionLayerProb": torch.zeros(B, K, 5, device=device, dtype=dtype),
            "LayerAgencyProb": layer_agency,
            "BodyMembershipProb": torch.zeros(B, K, device=device, dtype=dtype),
            "SelfPartSemantic": self_part,
            "PhysicalInteractionProb": torch.ones(B, K, device=device, dtype=dtype),
            "SurfaceParentProb": surface_parent,
            "SurfaceUV": torch.zeros(B, K, 2, device=device, dtype=dtype),
            "SurfaceUVConfidence": torch.ones(
                B, K, device=device, dtype=dtype),
            "VerificationConfidence": torch.ones(B, K, device=device, dtype=dtype),
            "OntologyRelationProb": torch.zeros(B, K, K, 9, device=device, dtype=dtype),
            "ContentMotionUV": torch.zeros(B, K, 2, device=device, dtype=dtype),
            "ContentChangeProb": torch.zeros(B, K, device=device, dtype=dtype),
            "EntityTextSemantic": torch.randn(
                B, K, 512, device=device, dtype=dtype),
            "EntityTextConfidence": torch.ones(
                B, K, device=device, dtype=dtype),
            "EntityTextRevision": torch.ones(
                B, K, device=device, dtype=torch.long),
            "EntityTextChanged": torch.ones(
                B, K, device=device, dtype=torch.bool),}
        return SimpleNamespace(
            IntegratedFeat=torch.randn(B, 1024, device=device, dtype=dtype),
            MotionToken=torch.randn(B, 512, device=device, dtype=dtype),
            PredErrorToken=torch.randn(B, 512, device=device, dtype=dtype),
            ObjectTokens=torch.randn(B, K, 512, device=device, dtype=dtype),
            SemanticNodes={
                "node_logits": torch.randn(B, K, 2, device=device, dtype=dtype),
                "level_logits": torch.randn(
                    B, K, 3, device=device, dtype=dtype),
                "object_class_logits": torch.randn(
                    B, K, ModuleDim.PstObjectClasses, device=device, dtype=dtype),
                "part_class_logits": torch.randn(
                    B, K, ModuleDim.PstPartClasses, device=device, dtype=dtype),
                "identity_embed": F.normalize(torch.randn(
                    B, K, ModuleDim.PstIdentityDim, device=device, dtype=dtype), dim=-1),
                "SpatialState": torch.randn(B, K, 7, device=device, dtype=dtype),
                "bbox_2d": torch.randn(B, K, 4, device=device, dtype=dtype),},
            Auxiliary=auxiliary,)

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

    def MakeTinyMemory(self, batch: int = 1) -> MemoryExtractor:
        memory = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
            inputDim=8,
            ssmStateDim=8,
            memoryDim=8,
            memorySize=8,
            symSize=8,
            ltmSize=8,
            nsK=4,
            outputDim=8,
            gwsSlots=2,
            emotionDim=4,))).to(self.device).eval()
        memory.EnsureB(batch)
        return memory

    def MemoryDeltaEnvelope(
        self,
        memory: MemoryExtractor,
        baseStep: int,
        newStep: int,
        kind: int = 1,) -> Dict[str, Any]:
        return {
            "state": memory.ExportState(step=baseStep),
            "base_step": torch.tensor(baseStep, device=self.device, dtype=torch.long),
            "new_step": torch.tensor(newStep, device=self.device, dtype=torch.long),
            "kind": torch.tensor(kind, device=self.device, dtype=torch.long),}

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
            realm = torch.full(
                (B,),
                ONTOLOGY_REALM_EXTERNAL,
                device=self.device,
                dtype=torch.int8)

            ltm.semantic.Store(key=key, value=val, score=torch.tensor([0.9], device=self.device),
                              source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),
                              realm=realm)
            ltm.episodic.Store(key=key, value=val, reward=torch.tensor([-1.0], device=self.device),
                               score=torch.tensor([0.8], device=self.device),
                               source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),
                               stateKey=state_key,
                               realm=realm,
                               typedMetadata=UnknownEpisodicTypeMetadata(
                                   realm, dtype=key.dtype))
            ltm.episodic.Store(key=key2, value=val2, reward=torch.tensor([2.0], device=self.device),
                               score=torch.tensor([1.0], device=self.device),
                               source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),
                               stateKey=state_key2,
                               realm=realm,
                               typedMetadata=UnknownEpisodicTypeMetadata(
                                   realm, dtype=key2.dtype))

            ltm.StepTick()
            sem_out, epi_out = ltm.Retrieve(key, topkSem=4, topkEpi=2)
            epi_state_out = ltm.episodic.Retrieve(state_key, topk=2, useStateKey=True)
            epi_event2 = ltm.episodic.Retrieve(key2, topk=1)
            epi_state2 = ltm.episodic.Retrieve(state_key2, topk=1, useStateKey=True)
            assert sem_out.shape == (B, dim) and epi_out.shape == (B, dim)
            assert epi_state_out.shape == (B, dim)
            assert torch.linalg.norm(sem_out).item() > 0 and torch.linalg.norm(epi_out).item() > 0
            assert torch.linalg.norm(epi_state_out).item() > 0
            assert F.cosine_similarity(epi_event2, val2).item() > 0.999
            assert F.cosine_similarity(epi_state2, val2).item() > 0.999
            assert 0.95 * val2.norm().item() < epi_event2.norm().item() <= val2.norm().item()
            assert 0.95 * val2.norm().item() < epi_state2.norm().item() <= val2.norm().item()

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

            def StoreWith(i, prio, rew):
                k = torch.zeros(1, dim, device=self.device); k[0, i] = 1.0
                v = torch.randn(1, dim, device=self.device)
                realm = torch.full(
                    (1,),
                    ONTOLOGY_REALM_EXTERNAL,
                    device=self.device,
                    dtype=torch.int8)
                epi.Store(
                    key=k,
                    value=v,
                    reward=torch.tensor([rew], device=self.device),
                    score=torch.tensor([prio], device=self.device),
                    source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),
                    realm=realm,
                    typedMetadata=UnknownEpisodicTypeMetadata(
                        realm, dtype=k.dtype),)

            StoreWith(0, prio=1.0, rew=0.0)
            StoreWith(1, prio=1.0, rew=0.0)
            StoreWith(2, prio=100.0, rew=0.0)

            with torch.no_grad():
                epi.touch[0, 0] = 1
                epi.touch[0, 1] = 50
                epi.touch[0, 2] = 1
                epi.step[0, :3] = epi.global_step[0]

            keys_before = epi.keys.clone()

            StoreWith(3, prio=0.5, rew=0.0)
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

            def StoreWith(i, prio, rew):
                k = torch.zeros(1, dim, device=self.device); k[0, i] = 1.0
                v = torch.randn(1, dim, device=self.device)
                realm = torch.full(
                    (1,),
                    ONTOLOGY_REALM_EXTERNAL,
                    device=self.device,
                    dtype=torch.int8)
                epi.Store(
                    key=k,
                    value=v,
                    reward=torch.tensor([rew], device=self.device),
                    score=torch.tensor([prio], device=self.device),
                    source=torch.tensor([MemoryType.SRC_REAL], device=self.device, dtype=torch.int8),
                    realm=realm,
                    typedMetadata=UnknownEpisodicTypeMetadata(
                        realm, dtype=k.dtype),)

            StoreWith(0, prio=1.0, rew=-10.0)
            StoreWith(1, prio=1.0, rew=0.0)
            StoreWith(2, prio=100.0, rew=0.0)

            with torch.no_grad():
                epi.touch[0, :3] = 1
                epi.step[0, :3] = epi.global_step[0]

            keys_before = epi.keys.clone()
            StoreWith(3, prio=0.5, rew=0.0)
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
                mem.gws.created_step.zero_()
                mem.gws.last_step.zero_()
                mem.gws.vals.zero_()
                mem.gws.priority[0, :3] = scores
                mem.gws.ttl[0, :3] = 1
                mem.gws.created_step[0, :3] = steps
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
                sym.prio[0, :3] = scores
                sym.step[0, :3] = steps
                sym.P_vals[0, 0, 0] = mark_new
                sym.P_vals[0, 1, 0] = mark_old
                sym.P_vals[0, 2, 0] = mark_mid

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

    def TestExportConsciousBank(self):
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
                mem.memory_filled.copy_(
                    torch.tensor([2, 1], device=self.device))
                mem.memory_importance[:, :2] = torch.tensor(
                    [[2.0, 1.0], [2.0, 0.0]],
                    device=self.device,
                    dtype=mem.dtype)
                mem.memory_steps[:, :2] = torch.tensor(
                    [[2, 1], [2, 0]],
                    device=self.device)
                mem.memory_values[0, 0, 0] = 3.0
                mem.memory_values[1, 0, 0] = 4.0
            bank = mem.ExportConsciousBank(topk=2)
            assert set(bank) == {"tokens", "valid"}
            assert bank["tokens"].shape == (2, 2, mem.memory_dim)
            assert bank["valid"].shape == (2, 2)
            assert bank["valid"].dtype == torch.bool
            assert bank["valid"].tolist() == [[True, True], [True, False]]
            assert bool(torch.isfinite(bank["tokens"]).all().item())
            assert int(torch.count_nonzero(
                bank["tokens"][1, 1]).item()) == 0
            print("ExportConsciousBank test passed.")
            return True
        except AssertionError as e:
            print(f"ExportConsciousBank test failed: {e}")
            return False
        except Exception as e:
            print(f"ExportConsciousBank test error: {e}")
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

    def TestGlobalWorkspaceLaneIsolation(self):
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
            assert int((gws.priority[0] > 0).sum().item()) == 1
            assert int((gws.priority[1] > 0).sum().item()) == 1
            assert torch.count_nonzero(gws.keys[0, :, 1]).item() == 0
            assert torch.count_nonzero(gws.keys[1, :, 0]).item() == 0

            out = gws.Attend(keys, topk=1)
            assert out.shape == (2, dim)
            assert float(out[0, 2].item()) > 9.0, f"row0 recall mismatch: {out[0].tolist()}"
            assert float(out[1, 3].item()) > 19.0, f"row1 recall mismatch: {out[1].tolist()}"

            print("GlobalWorkspace lane-isolation test passed.")
            return True
        except AssertionError as e:
            print(f"GlobalWorkspace lane-isolation test failed: {e}")
            return False
        except Exception as e:
            print(f"GlobalWorkspace lane-isolation test error: {e}")
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
            probability_real = SourceProbabilityReal(
                src,
                torch.tensor([[0.0, 0.5, 1.0]], device=self.device))
            assert float(probability_real[0, 0].item()) == 1.0
            assert torch.allclose(
                probability_real[0, 1],
                torch.tensor(0.7, device=self.device))
            assert torch.allclose(
                probability_real[0, 2],
                torch.tensor(0.25, device=self.device))

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
                mem.memory_source_confidence[0, :2] = torch.tensor([1.0, 0.9], device=self.device)

                mem.gws.global_step.fill_(3)
                mem.gws.priority[0, :2] = torch.tensor([0.7, 0.9], device=self.device)
                mem.gws.ttl[0, :2] = 5
                mem.gws.created_step[0, :2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                mem.gws.last_step[0, :2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                mem.gws.source_confidence[0, :2] = 1.0
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
                sym.prio[0, :2] = torch.tensor([0.5, 0.9], device=self.device)
                sym.step[0, :2] = torch.tensor([2, 3], device=self.device, dtype=torch.long)
                sym.touch[0, :2] = 1
                sym.source_confidence[0, :2] = 1.0
                sym.P_vals[0, 0, 0] = 9.0
                sym.P_vals[0, 1, 0] = 10.0

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
                epi.source_confidence[:, :2] = 1.0

            sem_before = int(mem.ltm.semantic.filled.max().item())
            sym_before = int(mem.sym_mem.filled.max().item())
            mem.ConsolidateMemory(topk=2)
            sem_after = int(mem.ltm.semantic.filled.max().item())
            sym_after = int(mem.sym_mem.filled.max().item())
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
            visual = self.MakeVisualState(B, self.device)
            visual.IntegratedFeat.zero_()
            visual.MotionToken.zero_()
            visual.PredErrorToken.zero_()
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

            ontology = mem.BuildOntologyObjectContext(x, visual)
            dense, event_key = mem.BuildEventCode(
                x,
                visualState=visual,
                ocrSemantic=ocr,
                intentHint=intent,
                emotion=emotion,
                tdError=td,
                uncertainty=uncertainty,
                risk=risk,
                confidence=confidence,
                ontologyContext=ontology["context"],)
            assert dense.shape == (B, cfg["memoryDim"])
            assert event_key.shape == (B, cfg["memoryDim"])
            assert torch.isfinite(dense).all() and torch.isfinite(event_key).all()
            assert torch.linalg.norm(event_key[0] - event_key[1]).item() > 1e-4

            gate = F.softmax(
                mem.event_completion_gate(torch.cat([dense, dense, dense, dense, td.view(B, 1)], dim=-1)),
                dim=-1)
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

            def AssertUnit(name: str, keys: torch.Tensor, mask: torch.Tensor):
                if not bool(mask.any().item()):
                    raise AssertionError(f"{name} has no valid keys to check")
                norms = torch.linalg.vector_norm(keys[mask], ord=2, dim=-1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
                    raise AssertionError(f"{name} key norms not unit: {norms.detach().cpu().tolist()}")

            kv_mask = torch.arange(int(mem.memory_size), device=self.device).view(1, -1) < mem.memory_filled.view(-1, 1)
            AssertUnit("kv", mem.memory_keys, kv_mask)

            gws_mask = (mem.gws.ttl > 0) & (mem.gws.priority > 0)
            AssertUnit("gws", mem.gws.keys, gws_mask)

            sem = mem.ltm.semantic
            sem_mask = torch.arange(int(sem.capacity), device=self.device).view(1, -1) < sem.filled.view(-1, 1)
            AssertUnit("ltm_sem", sem.keys, sem_mask)

            epi = mem.ltm.episodic
            epi_mask = torch.arange(int(epi.capacity), device=self.device).view(1, -1) < epi.filled.view(-1, 1)
            AssertUnit("ltm_epi", epi.keys, epi_mask)
            AssertUnit("ltm_epi_state", epi.state_keys, epi_mask)

            sym_n = int(mem.sym_mem.filled.max().item())
            if sym_n <= 0:
                raise AssertionError("sym_mem has no valid keys to check")
            sym_mask = torch.arange(mem.sym_mem.capacity, device=self.device).view(1, -1) < mem.sym_mem.filled.view(-1, 1)
            sym_norms = torch.linalg.vector_norm(mem.sym_mem.P_keys[sym_mask], ord=2, dim=-1)
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
                mem.memory_last_access_steps[0, :3] = torch.tensor(
                    [30, 12, 25], device=self.device, dtype=torch.long)
                mem.memory_last_rehearsal_steps[0, :3] = torch.tensor(
                    [30, 11, 24], device=self.device, dtype=torch.long)

                mem.gws.priority.zero_()
                mem.gws.ttl.zero_()
                mem.gws.created_step.zero_()
                mem.gws.last_step.zero_()
                mem.gws.global_step.fill_(30)
                mem.gws.priority[0, :3] = torch.tensor([3.0, 2.0, 1.0], device=self.device, dtype=mem.dtype)
                mem.gws.ttl[0, :3] = 1
                mem.gws.created_step[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)
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
                sym.step[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)

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
                sym.P_vals[0, 0, 0] = 30.0
                sym.P_vals[0, 1, 0] = 10.0
                sym.P_vals[0, 2, 0] = 20.0

            mem.ReorderMemorySteps()

            expect_local = torch.tensor([21, 1, 11], device=self.device, dtype=torch.long)
            if not torch.equal(mem.memory_steps[0, :3], expect_local):
                raise AssertionError(f"kv steps mismatch: {mem.memory_steps[0, :3].tolist()}")
            if not torch.equal(mem.gws.last_step[0, :3], expect_local):
                raise AssertionError(f"gws steps mismatch: {mem.gws.last_step[0, :3].tolist()}")
            if not torch.equal(mem.ltm.semantic.step[0, :3], expect_local):
                raise AssertionError(f"ltm_sem steps mismatch: {mem.ltm.semantic.step[0, :3].tolist()}")
            if not torch.equal(mem.ltm.episodic.step[0, :3], expect_local):
                raise AssertionError(f"ltm_epi steps mismatch: {mem.ltm.episodic.step[0, :3].tolist()}")
            if not torch.equal(mem.sym_mem.step[0, :3], expect_local):
                raise AssertionError(f"sym steps mismatch: {mem.sym_mem.step[0, :3].tolist()}")

            if int(mem.time_step[0].item()) != 21:
                raise AssertionError(f"time_step={int(mem.time_step[0].item())}, expected 21")
            if int(mem.gws.global_step[0].item()) != 21:
                raise AssertionError(f"gws.global_step={int(mem.gws.global_step[0].item())}, expected 21")
            if int(mem.ltm.semantic.global_step[0].item()) != 21:
                raise AssertionError(f"ltm_sem.global_step={int(mem.ltm.semantic.global_step[0].item())}, expected 21")
            if int(mem.ltm.episodic.global_step[0].item()) != 21:
                raise AssertionError(f"ltm_epi.global_step={int(mem.ltm.episodic.global_step[0].item())}, expected 21")
            if int(mem.sym_mem.global_step[0].item()) != 21:
                raise AssertionError(f"sym.global_step={int(mem.sym_mem.global_step[0].item())}, expected 21")
            assert torch.equal(
                mem.memory_last_access_steps[0, :3],
                torch.tensor([21, 3, 16], device=self.device, dtype=torch.long))
            assert torch.equal(
                mem.memory_last_rehearsal_steps[0, :3],
                torch.tensor([21, 2, 15], device=self.device, dtype=torch.long))

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

            def PrintShape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            with torch.no_grad():
                PrintShape("input.x", x)
                PrintShape("input.tdError", td)
                PrintShape("input.emotion", emotion)
                PrintShape("input.reward", reward)
                PrintShape("input.sourceLabel", source_label)

                y = self.CallMemForward(
                    mem,
                    x,
                    tdError=td,
                    emotion=emotion,
                    reward=reward,
                    sourceLabel=source_label,)
                PrintShape("output.y", y)

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

    def TestFullMergeRebasesIndependentClocks(self):
        try:
            common = dict(
                inputDim=8,
                ssmStateDim=8,
                memoryDim=8,
                memorySize=8,
                symSize=8,
                ltmSize=8,
                nsK=4,
                outputDim=8,
                emotionDim=4,)
            source = MemoryExtractor(**self.FilterKwargs(
                MemoryExtractor,
                dict(**common, gwsSlots=1))).to(self.device).eval()
            destination = MemoryExtractor(**self.FilterKwargs(
                MemoryExtractor,
                dict(**common, gwsSlots=3))).to(self.device).eval()
            with torch.no_grad():
                source.time_step.fill_(10_000)
                source.ltm.semantic.global_step.fill_(10_000)
                source.ltm.episodic.global_step.fill_(10_000)
                source.sym_mem.global_step.fill_(10_000)
                destination.time_step.fill_(100)
                destination.ltm.semantic.global_step.fill_(100)
                destination.ltm.episodic.global_step.fill_(100)
                destination.sym_mem.global_step.fill_(100)
                source.gws.global_step.fill_(10)
                source.gws.keys[0, 0, 0] = 1.0
                source.gws.vals[0, 0, 1] = 1.0
                source.gws.priority[0, 0] = 1.0
                source.gws.ttl[0, 0] = 5
                source.gws.created_step[0, 0] = 5
                source.gws.last_step[0, 0] = 8
                source.gws.last_rehearsal_step[0, 0] = 9
                destination.gws.global_step.fill_(100)
                destination.gws.keys.fill_(7.0)
                destination.gws.vals.fill_(7.0)
                destination.gws.priority.fill_(1.0)
                destination.gws.ttl.fill_(5)
                destination.gws.created_step.fill_(90)
                destination.gws.last_step.fill_(95)
                destination.gws.last_rehearsal_step.fill_(99)

                source.memory_filled.fill_(1)
                source.memory_keys[0, 0, 0] = 1.0
                source.memory_values[0, 0, 1] = 1.0
                source.memory_importance[0, 0] = 1.0
                source.memory_source_confidence[0, 0] = 1.0
                source.memory_steps[0, 0] = 9_990
                source.memory_last_access_steps[0, 0] = 9_995
                source.memory_last_rehearsal_steps[0, 0] = 9_999

                semantic = source.ltm.semantic
                semantic.filled.fill_(1)
                semantic.keys[0, 0, 0] = 1.0
                semantic.vals[0, 0, 1] = 1.0
                semantic.prio[0, 0] = 1.0
                semantic.source_confidence[0, 0] = 1.0
                semantic.step[0, 0] = 9_990
                semantic.last_access_step[0, 0] = 9_995
                semantic.last_rehearsal_step[0, 0] = 9_999

                episodic = source.ltm.episodic
                episodic.filled.fill_(1)
                episodic.keys[0, 0, 0] = 1.0
                episodic.state_keys[0, 0, 0] = 1.0
                episodic.vals[0, 0, 1] = 1.0
                episodic.prio[0, 0] = 1.0
                episodic.source_confidence[0, 0] = 1.0
                episodic.step[0, 0] = 9_990
                episodic.last_access_step[0, 0] = 9_995
                episodic.last_rehearsal_step[0, 0] = 9_999
                episodic.episode_id[0, 0] = 7
                episodic.event_id[0, 0] = 3
                episodic.slot_generation[0, 0] = 1

                symbolic = source.sym_mem
                symbolic.filled.fill_(1)
                symbolic.P_keys[0, 0, 0] = 1.0
                symbolic.P_vals[0, 0, 1] = 1.0
                symbolic.prio[0, 0] = 1.0
                symbolic.source_confidence[0, 0] = 1.0
                symbolic.created_step[0, 0] = 9_990
                symbolic.last_access_step[0, 0] = 9_995
                symbolic.last_rehearsal_step[0, 0] = 9_999

            destination.MergeMemoryState(
                source.ExportState(),
                mergeGws=True)
            expected = torch.tensor([90, 95, 99], device=self.device)
            assert torch.equal(torch.stack([
                destination.memory_steps[0, 0],
                destination.memory_last_access_steps[0, 0],
                destination.memory_last_rehearsal_steps[0, 0],]), expected)
            assert torch.equal(torch.stack([
                destination.ltm.semantic.step[0, 0],
                destination.ltm.semantic.last_access_step[0, 0],
                destination.ltm.semantic.last_rehearsal_step[0, 0],]), expected)
            assert torch.equal(torch.stack([
                destination.ltm.episodic.step[0, 0],
                destination.ltm.episodic.last_access_step[0, 0],
                destination.ltm.episodic.last_rehearsal_step[0, 0],]), expected)
            assert torch.equal(torch.stack([
                destination.sym_mem.created_step[0, 0],
                destination.sym_mem.last_access_step[0, 0],
                destination.sym_mem.last_rehearsal_step[0, 0],]), expected)
            assert int(destination.gws.global_step[0].item()) == 100
            assert destination.gws.created_step[0, 0].item() == 95
            assert destination.gws.last_step[0, 0].item() == 98
            assert destination.gws.last_rehearsal_step[0, 0].item() == 99
            assert torch.count_nonzero(destination.gws.keys[0, 1:]).item() == 0
            assert torch.count_nonzero(destination.gws.priority[0, 1:]).item() == 0

            hole_source = MemoryExtractor(**self.FilterKwargs(
                MemoryExtractor,
                dict(**common, gwsSlots=3))).to(self.device).eval()
            compact_destination = MemoryExtractor(**self.FilterKwargs(
                MemoryExtractor,
                dict(**common, gwsSlots=1))).to(self.device).eval()
            with torch.no_grad():
                hole_source.gws.global_step.fill_(10)
                hole_source.gws.keys[0, 1, 3] = 1.0
                hole_source.gws.vals[0, 1, 4] = 8.0
                hole_source.gws.priority[0, 1] = 2.0
                hole_source.gws.ttl[0, 1] = 4
                hole_source.gws.created_step[0, 1] = 6
                hole_source.gws.last_step[0, 1] = 8
                hole_source.gws.last_rehearsal_step[0, 1] = 9
                hole_source.gws.source_confidence[0, 1] = 1.0
            compact_destination.MergeMemoryState(
                hole_source.ExportState(),
                mergeGws=True)
            assert compact_destination.gws.keys[0, 0, 3].item() == 1.0
            assert compact_destination.gws.vals[0, 0, 4].item() == 8.0

            fresh = MemoryExtractor(**self.FilterKwargs(
                MemoryExtractor,
                dict(**common, gwsSlots=3))).to(self.device).eval()
            fresh.MergeMemoryState(source.ExportState())
            assert int(fresh.time_step[0].item()) == 10
            assert torch.equal(torch.stack([
                fresh.memory_steps[0, 0],
                fresh.memory_last_access_steps[0, 0],
                fresh.memory_last_rehearsal_steps[0, 0],]), torch.tensor(
                    [0, 5, 9],
                    device=self.device))
            print("Full-state merge independent-clock rebase test passed.")
            return True
        except Exception as e:
            print(f"Full-state merge independent-clock rebase test failed: {e}")
            return False

    def TestMalformedImportAndMergeAreAtomic(self):
        try:
            mem = self.MakeTinyMemory()
            one = torch.ones(1, device=self.device)
            mask = torch.ones(1, device=self.device, dtype=torch.bool)
            key = torch.zeros(1, 8, device=self.device); key[0, 0] = 1.0
            value = torch.zeros(1, 8, device=self.device); value[0, 1] = 1.0
            source = torch.zeros(1, device=self.device, dtype=torch.int8)
            emotion = torch.zeros(1, 4, device=self.device)
            mem.time_step.fill_(7)
            mem.KvWrite(
                key, value, one, emotion, one, source, one, mask,
                realm=torch.full_like(
                    source,
                    ONTOLOGY_REALM_EXTERNAL))
            snapshot = mem.ExportState()

            malformed = dict(snapshot)
            malformed["h_state"] = snapshot["h_state"].new_zeros(2, 8)
            rejected = False
            try:
                mem.ImportState(malformed)
            except ValueError:
                rejected = True
            assert rejected
            after_reject = mem.ExportState()
            for field in mem.FULL_MEMORY_STATE_FIELDS:
                assert torch.equal(after_reject[field], snapshot[field]), field

            invalid_range = dict(snapshot)
            invalid_range["memory_filled"] = snapshot["memory_filled"].clone()
            invalid_range["memory_filled"].fill_(mem.memory_size + 1)
            rejected = False
            try:
                mem.ImportState(invalid_range)
            except ValueError:
                rejected = True
            assert rejected
            assert int(mem.memory_filled[0].item()) == 1

            destination = self.MakeTinyMemory()
            self.CallMemForward(
                destination,
                torch.randn(1, 8, device=self.device))
            pending_count = len(destination.pending)
            h_before = destination.h_state.clone()
            assert pending_count > 0
            bad_merge = dict(snapshot)
            bad_merge["ltm_sem_filled"] = snapshot["ltm_sem_filled"].clone()
            bad_merge["ltm_sem_filled"].fill_(
                int(snapshot["ltm_sem_keys"].size(1)) + 1)
            rejected = False
            try:
                destination.MergeMemoryState(bad_merge)
            except ValueError:
                rejected = True
            assert rejected
            assert len(destination.pending) == pending_count
            assert int(destination.memory_filled[0].item()) == 0
            assert torch.equal(destination.h_state, h_before)
            print("Malformed import/merge atomic-preflight test passed.")
            return True
        except Exception as e:
            print(f"Malformed import/merge atomic-preflight test failed: {e}")
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

            src.episode_id.fill_(9)
            src.ltm.episodic.current_episode_id.fill_(9)
            dst.episode_id.fill_(2)
            dst.ltm.episodic.current_episode_id.fill_(2)
            src.usage_bank.success_alpha[0, 0] = 7.0
            src.usage_bank.parameter_observations[0, 0] = 2.0
            src.usage_bank.param_mu[0, 0].fill_(5.0)
            dst.usage_bank.parameter_observations[0, 0] = 1.0
            dst.usage_bank.param_mu[0, 0].fill_(1.0)

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
                    src.ltm.episodic.episode_id[0, t] = 0 if t < 2 else 1
                    src.ltm.episodic.event_id[0, t] = t if t < 2 else 0
                    src.sym_mem.P_keys[0, t, t % src.sym_mem.K] = 1.0
                    src.sym_mem.P_vals[0, t, 0] = value
                    src.sym_mem.prio[0, t] = value
                    src.sym_mem.source_confidence[0, t] = 1.0
                src.sym_mem.filled.copy_(torch.tensor([3, 0], device=self.device))
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
                src.ltm.episodic.episode_id[1, 0] = 0
                src.ltm.episodic.event_id[1, 0] = 0
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
            assert dst.episode_id.tolist() == [2, 2, 2]
            assert dst.ltm.episodic.current_episode_id.tolist() == [2, 2, 2]
            assert bool((dst.ltm.episodic.episode_id[0, :2] > 2).all().item())
            assert int(torch.unique(
                dst.ltm.episodic.episode_id[0, :2]).numel()) == 2
            assert float(dst.usage_bank.success_alpha[0, 0].item()) == 7.0
            assert float(dst.usage_bank.parameter_observations[0, 0].item()) == 3.0
            assert torch.allclose(
                dst.usage_bank.param_mu[0, 0],
                torch.full_like(dst.usage_bank.param_mu[0, 0], 11.0 / 3.0))
            assert dst.sym_mem.filled.tolist() == [2, 0, 0], dst.sym_mem.filled.tolist()
            assert bool((dst.gws.vals[:2, :2, 0] == 8.0).all().item()), dst.gws.vals[:2, :2, 0]
            imported_episode_ids = dst.ltm.episodic.episode_id[0, :2].clone()
            dst.ResetEpisodeState(torch.tensor(
                [True, False, False],
                device=self.device))
            assert int(dst.episode_id[0].item()) > int(imported_episode_ids.max().item())
            assert torch.equal(
                dst.episode_id,
                dst.ltm.episodic.current_episode_id)

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
            time_before_soft_reset = mem.time_step.detach().clone()
            with torch.no_grad():
                mem.gws.priority[:, 0] = 1.0
                mem.gws.ttl[:, 0] = 5

            mem.SoftReset()

            assert torch.count_nonzero(mem.fast_weights).item() == 0
            assert torch.count_nonzero(mem.h_state).item() == 0
            assert torch.equal(mem.time_step, time_before_soft_reset)
            assert torch.equal(mem.last_compress_step, time_before_soft_reset)
            assert int(mem.memory_filled[0].item()) == 0
            assert torch.count_nonzero(mem.gws.priority).item() == 0
            assert torch.count_nonzero(mem.gws.ttl).item() == 0

            mem.ResetAll()
            assert torch.count_nonzero(mem.memory_keys).item() == 0
            assert torch.count_nonzero(mem.memory_values).item() == 0
            assert int(mem.time_step[0].item()) == 0
            assert int(mem.memory_filled[0].item()) == 0

            gws_snap = mem.gws.Inspect()
            assert torch.count_nonzero(gws_snap["priority"]).item() == 0
            assert torch.count_nonzero(mem.ltm.semantic.filled).item() == 0
            assert torch.count_nonzero(mem.ltm.episodic.filled).item() == 0
            assert torch.count_nonzero(mem.sym_mem.filled).item() == 0

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
                mem.gws.keys[1, 0, 0] = 7.0
                mem.gws.priority[1, 0] = 1.0
                mem.gws.ttl[1, 0] = 5
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
                    torch.zeros(2, device=self.device, dtype=torch.int8),
                    torch.ones(2, device=self.device, dtype=mem.dtype),
                    torch.full(
                        (2,),
                        ONTOLOGY_REALM_EXTERNAL,
                        device=self.device,
                        dtype=torch.int8),
                    torch.ones(2, device=self.device, dtype=torch.bool),))
            mem.pending = [pending_write]
            plan = torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                device=self.device)
            mem.CachePlan("active", plan, "shape-a")

            mem.ResetEpisodeState(torch.tensor(
                [True, False], device=self.device))
            cached_plan = mem.RecallPlan("active", "shape-a")
            partial_ok = (
                torch.count_nonzero(mem.h_state[0]).item() == 0
                and torch.count_nonzero(mem.h_state[1]).item() > 0
                and torch.count_nonzero(mem.fast_weights[0]).item() == 0
                and torch.count_nonzero(mem.fast_weights[1]).item() > 0
                and torch.count_nonzero(mem.gws.keys[0]).item() == 0
                and float(mem.gws.keys[1, 0, 0].item()) == 7.0
                and float(mem.gws.priority[1, 0].item()) == 1.0
                and mem.memory_filled.tolist() == [1, 1]
                and cached_plan is not None
                and cached_plan["valid"].tolist() == [False, True]
                and torch.equal(
                    cached_plan["feature"],
                    torch.tensor(
                        [[0.0, 0.0], [3.0, 4.0]],
                        device=self.device))
                and not mem.pending)

            mem.pending = [pending_write]
            mem.ResetEpisodeState(torch.tensor(
                [True, True], device=self.device))
            full_ok = (
                torch.count_nonzero(mem.gws.keys).item() == 0
                and torch.count_nonzero(mem.gws.priority).item() == 0
                and mem.memory_filled.tolist() == [1, 1]
                and mem.RecallPlan("active", "shape-a") is None
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
            total = self.AttachAllInternalLosses(mem, base)

            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()

            out = self.CallMemForward(
                mem,
                x2,
                tdError=td2,
                reward=rwd2,
                emotion=emotion2)
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

    def TestUnifiedMemoryScoreNegativeSimilarityMonotonic(self):
        try:
            query = torch.tensor([[1.0, 0.0]], device=self.device)
            key = torch.tensor(
                [[[-0.5, math.sqrt(0.75)], [-0.5, math.sqrt(0.75)]]],
                device=self.device)
            zeros = torch.zeros(1, 2, device=self.device)
            ones = torch.ones(1, 2, device=self.device)

            age_score = UnifiedMemoryScore(
                query, key, age=torch.tensor([[0.0, 100.0]], device=self.device),
                priority=ones, touch=zeros, confidence=ones)
            priority_score = UnifiedMemoryScore(
                query, key, age=zeros,
                priority=torch.tensor([[2.0, 1.0]], device=self.device),
                touch=zeros, confidence=ones)
            touch_score = UnifiedMemoryScore(
                query, key, age=zeros, priority=ones,
                touch=torch.tensor([[10.0, 0.0]], device=self.device), confidence=ones)
            confidence_score = UnifiedMemoryScore(
                query, key, age=zeros, priority=ones, touch=zeros,
                confidence=torch.tensor([[1.0, 0.2]], device=self.device))
            assert age_score[0, 0] > age_score[0, 1]
            assert priority_score[0, 0] > priority_score[0, 1]
            assert touch_score[0, 0] > touch_score[0, 1]
            assert confidence_score[0, 0] > confidence_score[0, 1]
            print("UnifiedMemoryScore negative-similarity monotonicity test passed.")
            return True
        except Exception as e:
            print(f"UnifiedMemoryScore negative-similarity monotonicity test failed: {e}")
            return False

    def TestSourceEvidenceMassPreserved(self):
        try:
            attention = torch.tensor([[0.95, 0.05]], device=self.device)
            values = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=self.device)
            probability_real = torch.tensor([[1.0, 0.0]], device=self.device)
            recall, real_mass, imagined_mass = SourceBalancedRecall(
                attention,
                values,
                probability_real,
                torch.tensor([[0.5]], device=self.device))
            assert torch.allclose(real_mass, torch.tensor([[0.95]], device=self.device))
            assert torch.allclose(imagined_mass, torch.tensor([[0.05]], device=self.device))
            assert float(recall[0, 0].item()) > 0.97
            assert float(recall[0, 1].item()) < 0.03

            single, _, single_imagined = SourceBalancedRecall(
                torch.ones(1, 1, device=self.device),
                values[:, :1],
                torch.ones(1, 1, device=self.device),
                torch.ones(1, 1, device=self.device))
            assert torch.allclose(single, values[:, 0], atol=3e-6, rtol=1e-6)
            assert torch.count_nonzero(single_imagined).item() == 0
            print("Source evidence-mass preservation test passed.")
            return True
        except Exception as e:
            print(f"Source evidence-mass preservation test failed: {e}")
            return False

    def TestSymbolicMemoryLaneIsolation(self):
        try:
            sym = SymbolicMemory(k=4, capacity=4).to(self.device)
            keys = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0]],
                device=self.device)
            values = torch.tensor(
                [[0.0, 0.0, 2.0, 0.0],
                 [0.0, 0.0, 0.0, 3.0]],
                device=self.device)
            score = torch.ones(2, device=self.device)
            source = torch.zeros(2, device=self.device, dtype=torch.int8)
            confidence = torch.ones(2, device=self.device)
            sym.Store(keys, values, score, source, confidence)
            out = sym.Retrieve(keys, topK=1)
            assert bool((out[:, 2:].abs().sum(dim=1) > 0).all().item())
            assert float(out[0, 3].item()) == 0.0
            assert float(out[1, 2].item()) == 0.0
            assert torch.count_nonzero(sym.P_keys[0, :, 1]).item() == 0
            assert torch.count_nonzero(sym.P_keys[1, :, 0]).item() == 0

            sym.ResetRows(torch.tensor([True, False], device=self.device))
            assert int(sym.filled[0].item()) == 0
            assert int(sym.filled[1].item()) == 1
            assert torch.count_nonzero(sym.P_vals[0]).item() == 0
            assert torch.allclose(sym.P_vals[1, 0], values[1])
            print("Symbolic-memory lane-isolation test passed.")
            return True
        except Exception as e:
            print(f"Symbolic-memory lane-isolation test failed: {e}")
            return False

    def TestEmptyRecallRowsStrictZero(self):
        try:
            mem = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=4,
                symSize=4, ltmSize=4, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            mem.EnsureB(2)
            with torch.no_grad():
                mem.memory_filled.copy_(torch.tensor([1, 0], device=self.device))
                mem.memory_keys[0, 0, 0] = 1.0
                mem.memory_values[0, 0, 1] = 2.0
                mem.memory_importance[0, 0] = 1.0
                mem.memory_source_confidence[0, 0] = 1.0
            query = torch.zeros(2, 8, device=self.device)
            query[:, 0] = 1.0
            out = mem.Retrieve(
                query,
                fusionGate=torch.zeros(2, device=self.device),
                importance=torch.zeros(2, device=self.device),
                localGate=torch.zeros(2, device=self.device),
                emotion=torch.zeros(2, 4, device=self.device),
                tdError=torch.zeros(2, device=self.device))
            assert torch.count_nonzero(out[1]).item() == 0
            assert torch.isfinite(out).all()
            print("Mixed-batch empty recall test passed.")
            return True
        except Exception as e:
            print(f"Mixed-batch empty recall test failed: {e}")
            return False

    def TestKvStatsMixedLaneCounts(self):
        try:
            mem = self.MakeTinyMemory(batch=3)
            with torch.no_grad():
                mem.memory_filled.copy_(torch.tensor(
                    [1, 2, 0],
                    device=self.device))
                mem.memory_keys.zero_()
                mem.memory_keys[0, 0, 0] = 1.0
                mem.memory_keys[1, 0, 0] = 1.0
                mem.memory_keys[1, 1, 1] = 1.0
            query = torch.zeros(3, mem.memory_dim, device=self.device)
            query[:, 0] = 1.0
            stats = mem.KvStats(query, topK=8)
            assert torch.isfinite(stats).all()
            assert torch.allclose(
                stats[0],
                torch.tensor([1.0, 0.0, 0.0], device=self.device))
            assert torch.count_nonzero(stats[2]).item() == 0
            print("KV-statistics mixed-lane count test passed.")
            return True
        except Exception as e:
            print(f"KV-statistics mixed-lane count test failed: {e}")
            return False

    def TestCreationTimeIsImmutableOnRetrieve(self):
        try:
            D = 8
            key = torch.zeros(1, D, device=self.device); key[0, 0] = 1.0
            value = torch.zeros(1, D, device=self.device); value[0, 1] = 1.0
            score = torch.ones(1, device=self.device)
            source = torch.zeros(1, device=self.device, dtype=torch.int8)
            confidence = torch.ones(1, device=self.device)

            sem = SemanticLTM(D, 4).to(self.device)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)
            sem.Store(
                key,
                value,
                score,
                source,
                sourceConfidence=confidence,
                realm=realm)
            sem.StepTick(); created_sem = sem.step.clone()
            sem.Retrieve(key, topk=1)
            assert torch.equal(sem.step, created_sem)
            assert int(sem.last_access_step[0, 0].item()) == int(sem.global_step[0].item())

            epi = EpisodicLTM(D, 4).to(self.device)
            epi.Store(
                key,
                value,
                score,
                score,
                source,
                sourceConfidence=confidence,
                realm=realm,
                typedMetadata=UnknownEpisodicTypeMetadata(
                    realm, dtype=key.dtype))
            epi.StepTick(); created_epi = epi.step.clone()
            epi.Retrieve(key, topk=1)
            assert torch.equal(epi.step, created_epi)
            assert int(epi.last_access_step[0, 0].item()) == int(epi.global_step[0].item())

            sym = SymbolicMemory(4, 4).to(self.device)
            sym_key = F.normalize(torch.ones(1, 4, device=self.device), dim=-1)
            sym.Store(sym_key, sym_key, score, source, confidence)
            sym.StepTick(); created_sym = sym.step.clone()
            sym.Retrieve(sym_key, topK=1)
            assert torch.equal(sym.step, created_sym)
            assert int(sym.last_access_step[0, 0].item()) == int(sym.global_step[0].item())
            print("Creation/access/rehearsal timestamp separation test passed.")
            return True
        except Exception as e:
            print(f"Creation/access/rehearsal timestamp separation test failed: {e}")
            return False

    def TestFullRankHeteroAssociativeHebbian(self):
        try:
            cfg = self.FilterKwargs(MemoryExtractor, dict(
                inputDim=8, ssmStateDim=8, memoryDim=8, memorySize=2,
                symSize=2, ltmSize=2, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))
            positive = MemoryExtractor(**cfg).to(self.device).eval()
            negative = MemoryExtractor(**cfg).to(self.device).eval()
            key = torch.zeros(1, 8, device=self.device); key[0, 0] = 1.0
            value = torch.zeros(1, 8, device=self.device); value[0, 3] = 2.0
            one = torch.ones(1, device=self.device)
            positive.HebbianUpdate(key, value, one, one, one, one)
            negative.HebbianUpdate(key, value, one, -one, one, one)
            assert positive.fast_weights.shape == (1, 8, 8)
            assert float(positive.fast_weights[0, 0, 3].item()) > 0.0
            assert torch.count_nonzero(positive.fast_weights[0, 1:]).item() == 0
            assert torch.allclose(positive.fast_weights, negative.fast_weights)

            with torch.no_grad():
                positive.fast_weights.zero_()
                positive.fast_weights[0, 0, 0] = 1.0
            zero = torch.zeros(1, device=self.device)
            positive.HebbianUpdate(key, value, zero, zero, one, one)
            assert torch.allclose(
                positive.fast_weights[0, 0, 0],
                torch.tensor(0.95, device=self.device),
                atol=1e-6,
                rtol=1e-6)
            assert torch.count_nonzero(positive.fast_weights).item() == 1
            print("Full-rank hetero-associative Hebbian test passed.")
            return True
        except Exception as e:
            print(f"Full-rank hetero-associative Hebbian test failed: {e}")
            return False

    def TestStableFullMatrixWorkingState(self):
        try:
            mem = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=2,
                symSize=2, ltmSize=2, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            transition = mem.StableTransition()
            row_bound = transition.abs().sum(dim=1)
            assert bool((row_bound < 1.0).all().item())
            assert transition.shape == mem.A_full.shape == (16, 16)
            assert float(torch.sigmoid(mem.state_retention_logit).max().item()) > 0.99
            assert float(torch.sigmoid(mem.state_retention_logit).min().item()) < 0.81
            state = torch.randn(1, 16, device=self.device)
            zero_input = torch.zeros(1, 16, device=self.device)
            for _ in range(1000):
                state = mem.UpdateWorkingState(zero_input, state)
            assert torch.isfinite(state).all()
            assert float(state.abs().max().item()) <= 1.0
            print("Stable full-matrix working-state test passed.")
            return True
        except Exception as e:
            print(f"Stable full-matrix working-state test failed: {e}")
            return False

    def TestStreamingCompressionMetadata(self):
        try:
            mem = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=8, ssmStateDim=8, memoryDim=8, memorySize=4,
                symSize=2, ltmSize=2, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            key = torch.zeros(1, 8, device=self.device); key[0, 0] = 1.0
            value_a = torch.zeros(1, 8, device=self.device); value_a[0, 1] = 2.0
            value_b = torch.zeros(1, 8, device=self.device); value_b[0, 1] = 4.0
            emotion_a = torch.zeros(1, 4, device=self.device)
            emotion_b = torch.full((1, 4), 2.0, device=self.device)
            mask = torch.ones(1, device=self.device, dtype=torch.bool)
            realm = torch.full(
                (1,),
                ONTOLOGY_REALM_EXTERNAL,
                device=self.device,
                dtype=torch.int8)
            mem.time_step.fill_(1)
            mem.KvWrite(key, value_a, torch.ones(1, device=self.device), emotion_a,
                        torch.ones(1, device=self.device),
                        torch.zeros(1, device=self.device, dtype=torch.int8),
                        torch.full((1,), 0.9, device=self.device), mask,
                        realm=realm)
            mem.time_step.fill_(2)
            mem.KvWrite(key, value_b, torch.ones(1, device=self.device), emotion_b,
                        torch.full((1,), 5.0, device=self.device),
                        torch.ones(1, device=self.device, dtype=torch.int8),
                        torch.full((1,), 0.1, device=self.device), mask,
                        realm=realm)
            assert int(mem.memory_filled[0].item()) == 1
            assert int(mem.memory_source[0, 0].item()) == MemoryType.SRC_MIXED
            assert torch.allclose(mem.memory_emotion[0, 0], torch.ones(4, device=self.device))
            assert float(mem.memory_reward_abs[0, 0].item()) == 5.0
            assert int(mem.memory_merge_count[0, 0].item()) == 1

            contradictory = torch.zeros(1, 8, device=self.device)
            contradictory[0, 2] = 4.0
            mem.time_step.fill_(3)
            mem.KvWrite(
                key,
                contradictory,
                torch.ones(1, device=self.device),
                emotion_b,
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device, dtype=torch.int8),
                torch.ones(1, device=self.device),
                mask,
                realm=realm)
            assert int(mem.memory_filled[0].item()) == 2

            original_bmm = torch.bmm
            try:
                torch.bmm = lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("AutoCompress attempted quadratic pairwise bmm"))
                mem.AutoCompress()
            finally:
                torch.bmm = original_bmm
            assert int(mem.memory_filled[0].item()) == 2
            print("Streaming compression and metadata-fusion test passed.")
            return True
        except Exception as e:
            print(f"Streaming compression and metadata-fusion test failed: {e}")
            return False

    def TestEventBoundarySelectiveWriting(self):
        try:
            mem = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=4,
                symSize=2, ltmSize=2, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            visual = self.MakeVisualState(1, self.device)
            attended = torch.zeros(1, 16, device=self.device)
            intent = torch.zeros(1, 512, device=self.device)
            zero = torch.zeros(1, device=self.device)
            one = torch.ones(1, device=self.device)
            ontology = mem.BuildOntologyObjectContext(attended, visual)
            _, first = mem.DetectEventBoundary(
                attended, visual, intent, zero, zero, zero, one, ontology)
            ontology = mem.BuildOntologyObjectContext(attended, visual)
            _, repeated = mem.DetectEventBoundary(
                attended, visual, intent, zero, zero, zero, one, ontology)
            ontology = mem.BuildOntologyObjectContext(attended, visual)
            _, risk_boundary = mem.DetectEventBoundary(
                attended, visual, intent, zero, torch.ones_like(zero), zero, one,
                ontology)
            assert bool(first.item())
            assert not bool(repeated.item())
            assert bool(risk_boundary.item())
            print("Event-boundary selective-writing test passed.")
            return True
        except Exception as e:
            print(f"Event-boundary selective-writing test failed: {e}")
            return False

    def TestEpisodicSequenceLinks(self):
        try:
            epi = EpisodicLTM(dim=8, capacity=8).to(self.device)
            source = torch.zeros(1, device=self.device, dtype=torch.int8)
            confidence = torch.ones(1, device=self.device)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)
            for event in range(3):
                key = torch.zeros(1, 8, device=self.device); key[0, event] = 1.0
                value = torch.zeros(1, 8, device=self.device); value[0, event] = float(event + 1)
                epi.Store(
                    key, value,
                    reward=torch.zeros(1, device=self.device),
                    score=torch.ones(1, device=self.device),
                    source=source,
                    sourceConfidence=confidence,
                    episodeId=torch.zeros(1, device=self.device, dtype=torch.long),
                    eventId=torch.tensor([event], device=self.device),
                    realm=realm,
                    typedMetadata=UnknownEpisodicTypeMetadata(
                        realm, dtype=key.dtype))
                epi.StepTick()
            sequence, valid = epi.RetrieveSequence(torch.tensor([1], device=self.device))
            assert valid.tolist() == [[True, True, True]]
            assert sequence[0, 0, 0].item() == 1.0
            assert sequence[0, 1, 1].item() == 2.0
            assert sequence[0, 2, 2].item() == 3.0
            epi.StartNewEpisode(torch.ones(1, device=self.device, dtype=torch.bool))
            key = torch.zeros(1, 8, device=self.device); key[0, 4] = 1.0
            epi.Store(key, key, torch.zeros(1, device=self.device),
                      torch.ones(1, device=self.device), source,
                      sourceConfidence=confidence, eventId=torch.tensor([4], device=self.device),
                      realm=realm,
                      typedMetadata=UnknownEpisodicTypeMetadata(
                          realm, dtype=key.dtype))
            assert int(epi.prev_index[0, 3].item()) == -1
            print("Episodic sequence-link test passed.")
            return True
        except Exception as e:
            print(f"Episodic sequence-link test failed: {e}")
            return False

    def TestVerifiedEpisodicSourceRelinksSequence(self):
        try:
            epi = EpisodicLTM(dim=8, capacity=8).to(self.device)
            imagined = torch.full(
                (1,),
                MemoryType.SRC_IMAGINE,
                device=self.device,
                dtype=torch.int8)
            realm = torch.full_like(
                imagined,
                ONTOLOGY_REALM_EXTERNAL)
            for event in range(3):
                key = torch.zeros(1, 8, device=self.device)
                key[0, event] = 1.0
                epi.Store(
                    key,
                    key,
                    reward=torch.zeros(1, device=self.device),
                    score=torch.ones(1, device=self.device),
                    source=imagined,
                    sourceConfidence=torch.full((1,), 0.1, device=self.device),
                    episodeId=torch.zeros(1, device=self.device, dtype=torch.long),
                    eventId=torch.tensor([event], device=self.device),
                    realm=realm,
                    typedMetadata=UnknownEpisodicTypeMetadata(
                        realm, dtype=key.dtype))
                epi.StepTick()
            _, before = epi.RetrieveSequence(torch.tensor([1], device=self.device))
            assert before.tolist() == [[True, True, True]]
            middle_key = torch.zeros(1, 8, device=self.device)
            middle_key[0, 1] = 1.0
            epi.VerifyWithRealEvidence(
                middle_key,
                torch.ones(1, device=self.device, dtype=torch.bool),
                realm)
            _, after = epi.RetrieveSequence(torch.tensor([1], device=self.device))
            assert int(epi.source[0, 1].item()) == MemoryType.SRC_MIXED
            assert after.tolist() == [[False, True, False]]
            print("Verified episodic source-branch relink test passed.")
            return True
        except Exception as e:
            print(f"Verified episodic source-branch relink test failed: {e}")
            return False

    def TestSemanticPrototypeAbstraction(self):
        try:
            sem = SemanticLTM(dim=8, capacity=8).to(self.device)
            key_a = torch.zeros(1, 8, device=self.device); key_a[0, 0] = 1.0
            key_b = F.normalize(key_a + 0.05 * F.one_hot(
                torch.tensor([1], device=self.device), 8).float(), dim=-1)
            value_a = torch.zeros(1, 8, device=self.device); value_a[0, 2] = 1.0
            value_b = torch.zeros(1, 8, device=self.device); value_b[0, 2] = 3.0
            score = torch.ones(1, device=self.device)
            source = torch.zeros(1, device=self.device, dtype=torch.int8)
            confidence = torch.ones(1, device=self.device)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)
            sem.Store(
                key_a, value_a, score, source,
                sourceConfidence=confidence,
                realm=realm)
            sem.Store(
                key_b, value_b, score, source,
                sourceConfidence=confidence,
                realm=realm)
            assert int(sem.filled[0].item()) == 1
            assert float(sem.prototype_count[0, 0].item()) == 2.0
            assert torch.allclose(sem.vals[0, 0, 2], torch.tensor(2.0, device=self.device))
            assert torch.allclose(
                sem.prototype_variance[0, 0, 2],
                torch.tensor(1.0, device=self.device))
            print("Semantic prototype-abstraction test passed.")
            return True
        except Exception as e:
            print(f"Semantic prototype-abstraction test failed: {e}")
            return False

    def TestOntologyTypedMemoryIsolation(self):
        try:
            mem = self.MakeTinyMemory()
            key = torch.zeros(1, 8, device=self.device)
            key[0, 0] = 1.0
            value = torch.zeros(1, 8, device=self.device)
            value[0, 1] = 1.0
            one = torch.ones(1, device=self.device)
            mask = torch.ones(1, device=self.device, dtype=torch.bool)
            emotion = torch.zeros(1, 4, device=self.device)
            real = torch.full(
                (1,),
                MemoryType.SRC_REAL,
                device=self.device,
                dtype=torch.int8)
            imagined = torch.full(
                (1,),
                MemoryType.SRC_IMAGINE,
                device=self.device,
                dtype=torch.int8)
            physical = torch.full(
                (1,),
                ONTOLOGY_REALM_EXTERNAL,
                device=self.device,
                dtype=torch.int8)
            virtual = torch.full(
                (1,),
                ONTOLOGY_REALM_VIRTUAL,
                device=self.device,
                dtype=torch.int8)

            mem.KvWrite(
                key, value, one, emotion, one, real, one, mask,
                realm=physical)
            mem.KvWrite(
                key, value, one, emotion, one, imagined, one, mask,
                realm=virtual)
            assert int(mem.memory_filled[0].item()) == 2
            assert mem.memory_realm[0, :2].tolist() == [
                ONTOLOGY_REALM_EXTERNAL,
                ONTOLOGY_REALM_VIRTUAL]
            assert mem.memory_source[0, :2].tolist() == [
                MemoryType.SRC_REAL,
                MemoryType.SRC_IMAGINE]

            semantic = mem.ltm.semantic
            semantic.Store(
                key, value, one, real,
                sourceConfidence=one,
                realm=physical)
            semantic.Store(
                key, value, one, imagined,
                sourceConfidence=one,
                realm=virtual)
            assert int(semantic.filled[0].item()) == 2
            assert semantic.prototype_count[0, :2].tolist() == [1.0, 1.0]
            assert semantic.realm[0, :2].tolist() == [
                ONTOLOGY_REALM_EXTERNAL,
                ONTOLOGY_REALM_VIRTUAL]

            policy_memory = self.MakeTinyMemory()
            visual_effect = torch.full(
                (1,),
                ONTOLOGY_REALM_EFFECT,
                device=self.device,
                dtype=torch.int8)
            effect_metadata = UnknownEpisodicTypeMetadata(
                visual_effect,
                dtype=key.dtype)
            effect_metadata["agency_id"].fill_(ONTOLOGY_AGENCY_AUTONOMOUS)
            effect_metadata["motion_layer_prob"][0, 4] = 1.0
            effect_metadata["surface_uv"][0] = torch.tensor(
                [0.25, 0.75], device=self.device)
            effect_metadata["surface_uv_confidence"].fill_(0.8)
            effect_metadata["verification"].fill_(0.9)
            effect_metadata["content_change"].fill_(0.7)
            policy_memory.LtmOnlineStore(
                keySem=key,
                keyEpi=key,
                keyEpiState=key,
                valSem=value,
                valEpi=value,
                importance=torch.full(
                    (1,), 1.5, device=self.device),
                tdError=one,
                reward=torch.zeros(1, device=self.device),
                sourceLabel=real,
                realm=visual_effect,
                episodicMetadata=effect_metadata,
                uncertainty=torch.zeros(1, device=self.device),
                risk=torch.zeros(1, device=self.device),
                confidence=one,
                sourceConfidence=one,
                eventMask=mask,
                episodeId=torch.zeros(
                    1, device=self.device, dtype=torch.long),
                eventId=torch.ones(
                    1, device=self.device, dtype=torch.long))
            assert int(policy_memory.ltm.semantic.filled[0].item()) == 0
            assert int(policy_memory.ltm.episodic.filled[0].item()) == 1
            assert int(policy_memory.ltm.episodic.realm[0, 0].item()) == int(
                ONTOLOGY_REALM_EFFECT)
            assert int(policy_memory.ltm.episodic.agency_id[0, 0].item()) == int(
                ONTOLOGY_AGENCY_AUTONOMOUS)
            assert torch.equal(
                policy_memory.ltm.episodic.motion_layer_prob[0, 0],
                effect_metadata["motion_layer_prob"][0])
            assert torch.equal(
                policy_memory.ltm.episodic.surface_uv[0, 0],
                effect_metadata["surface_uv"][0])
            assert torch.equal(
                policy_memory.ltm.episodic.surface_uv_confidence[0, 0],
                effect_metadata["surface_uv_confidence"][0])
            policy_memory.ConsolidateMemory(topk=1)
            assert int(policy_memory.ltm.semantic.filled[0].item()) == 0

            state = mem.ExportDurableState()
            assert mem.DURABLE_MEMORY_SCHEMA_VERSION == 6
            restored = self.MakeTinyMemory()
            restored.ImportDurableState(state)
            assert torch.equal(restored.memory_realm, mem.memory_realm)
            assert torch.equal(restored.ltm.semantic.realm, semantic.realm)
            policy_state = policy_memory.ExportDurableState()
            restored.ImportDurableState(policy_state)
            assert torch.equal(
                restored.ltm.episodic.surface_uv_confidence,
                policy_memory.ltm.episodic.surface_uv_confidence)
            assert torch.equal(
                restored.ltm.episodic.motion_layer_prob,
                policy_memory.ltm.episodic.motion_layer_prob)
            print("Ontology-typed memory isolation test passed.")
            return True
        except Exception as e:
            print(f"Ontology-typed memory isolation test failed: {e}")
            return False

    def TestOntologyContextAndScreenBoundary(self):
        try:
            mem = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=16, ssmStateDim=16, memoryDim=8, memorySize=4,
                symSize=2, ltmSize=2, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            visual = self.MakeVisualState(1, self.device)
            visual.SemanticNodes["node_logits"][..., 0] = -10.0
            visual.SemanticNodes["node_logits"][..., 1] = 10.0
            realm = visual.Auxiliary["RealmProb"]
            realm.zero_()
            realm[..., ONTOLOGY_REALM_VIRTUAL] = 1.0
            visual.Auxiliary["PhysicalInteractionProb"].zero_()
            visual.Auxiliary["ContentChangeProb"].zero_()
            attended = torch.zeros(1, 16, device=self.device)
            intent = torch.zeros(1, 512, device=self.device)
            zero = torch.zeros(1, device=self.device)
            one = torch.ones(1, device=self.device)

            ontology = mem.BuildOntologyObjectContext(attended, visual)
            assert int(ontology["realm_id"][0].item()) == int(
                ONTOLOGY_REALM_VIRTUAL)
            _, first = mem.DetectEventBoundary(
                attended, visual, intent, zero, zero, zero, one, ontology)
            assert bool(first.item())

            for step in range(10):
                visual.ObjectTokens.mul_(-1.0)
                visual.MotionToken.fill_(float(step + 1))
                visual.Auxiliary["ContentMotionUV"].fill_(
                    float(step + 1))
                ontology = mem.BuildOntologyObjectContext(attended, visual)
                _, boundary = mem.DetectEventBoundary(
                    attended, visual, intent, zero, zero, zero, one, ontology)
                assert not bool(boundary.item()), step

            visual.Auxiliary["ContentChangeProb"].fill_(1.0)
            ontology = mem.BuildOntologyObjectContext(attended, visual)
            _, changed = mem.DetectEventBoundary(
                attended, visual, intent, zero, zero, zero, one, ontology)
            assert bool(changed.item())
            print("Ontology context/screen-boundary test passed.")
            return True
        except Exception as e:
            print(f"Ontology context/screen-boundary test failed: {e}")
            return False

    def TestForwardFlushPersistsFocusedEpisodicMetadata(self):
        try:
            mem = self.MakeTinyMemory().eval()
            visual = self.MakeVisualState(1, self.device)
            auxiliary = visual.Auxiliary
            auxiliary["PerceptualPresence"].zero_()
            auxiliary["PerceptualPresence"][0, 0] = 1.0
            auxiliary["RealmProb"].zero_()
            auxiliary["RealmProb"][
                0, 0, ONTOLOGY_REALM_VIRTUAL] = 1.0
            auxiliary["AgencyProb"].zero_()
            auxiliary["AgencyProb"][
                0, 0, ONTOLOGY_AGENCY_AUTONOMOUS] = 1.0
            auxiliary["MotionLayerProb"].zero_()
            auxiliary["MotionLayerProb"][0, 0, 3] = 1.0
            auxiliary["BodyMembershipProb"].zero_()
            auxiliary["SurfaceParentProb"].zero_()
            auxiliary["SurfaceParentProb"][0, :, -1] = 1.0
            auxiliary["SurfaceParentProb"][0, 0, -1] = 0.0
            auxiliary["SurfaceParentProb"][0, 0, 1] = 1.0
            auxiliary["SurfaceUV"][0, 0] = torch.tensor(
                [0.2, 0.8], device=self.device)
            auxiliary["SurfaceUVConfidence"][0, 0] = 0.63
            auxiliary["VerificationConfidence"][0, 0] = 0.91
            auxiliary["ContentChangeProb"][0, 0] = 0.82

            x = torch.randn(1, mem.input_dim, device=self.device)
            td_error = torch.ones(1, device=self.device)
            emotion = torch.zeros(1, mem.emotion_dim, device=self.device)
            ocr = torch.zeros(1, 512, device=self.device)
            intent = torch.zeros(1, 512, device=self.device)
            forward_args = dict(
                tdError=td_error,
                emotion=emotion,
                reward=torch.zeros(1, device=self.device),
                visualState=visual,
                ocrSemantic=ocr,
                intentHint=intent,
                uncertainty=torch.zeros(1, device=self.device),
                risk=torch.zeros(1, device=self.device),
                confidence=torch.ones(1, device=self.device))
            mem(x, **forward_args)
            ltm_pending = [
                payload for kind, payload in mem.pending if kind == "ltm"]
            assert len(ltm_pending) == 1
            pending_metadata = ltm_pending[0][17]
            assert torch.equal(
                pending_metadata["surface_uv_confidence"],
                torch.tensor([0.63], device=self.device))

            mem(x, **forward_args)
            episodic = mem.ltm.episodic
            assert int(episodic.filled[0].item()) == 1
            assert int(episodic.realm[0, 0].item()) == int(
                ONTOLOGY_REALM_VIRTUAL)
            assert int(episodic.source[0, 0].item()) == MemoryType.SRC_REAL
            assert int(episodic.agency_id[0, 0].item()) == int(
                ONTOLOGY_AGENCY_AUTONOMOUS)
            assert int(episodic.observed_surface_parent_index[0, 0].item()) == 1
            assert torch.allclose(
                episodic.surface_uv[0, 0],
                torch.tensor([0.2, 0.8], device=self.device))
            assert torch.allclose(
                episodic.surface_uv_confidence[0, 0],
                torch.tensor(0.63, device=self.device))
            assert torch.allclose(
                episodic.verification[0, 0],
                torch.tensor(0.91, device=self.device))
            assert torch.allclose(
                episodic.content_change[0, 0],
                torch.tensor(0.82, device=self.device))
            print("Forward/flush typed episodic-metadata test passed.")
            return True
        except Exception as e:
            print(f"Forward/flush typed episodic-metadata test failed: {e}")
            return False

    def TestObjectUsagePosteriorAndUnknown(self):
        try:
            bank = ObjectUsageBank(
                numObjects=2, numSkills=1, paramDim=2, idDim=3,
                slotDeltaDim=2, usageDim=4, attrDim=2).to(self.device)
            with torch.no_grad():
                bank.instance_descriptors.copy_(torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=self.device))
                bank.applicable.fill_(1.0)
                bank.default_params[0, 0] = torch.tensor([1.0, 0.0], device=self.device)
                bank.param_mu[0, 0] = torch.tensor([5.0, 0.0], device=self.device)
                bank.readout_proj.weight.zero_(); bank.readout_proj.bias.zero_()
                bank.readout_proj.weight[0, 0] = 1.0
                bank.readout_proj.weight[1, -1] = 1.0
                bank.readout_proj.weight[2, 4] = 1.0
            identity = torch.tensor([[[1.0, 0.0, 0.0]]], device=self.device)
            attribute = torch.zeros(1, 1, 2, device=self.device)
            attention = torch.ones(1, 1, device=self.device)
            before = bank.SlotReadout(identity, attribute, attention)
            with torch.no_grad():
                bank.parameter_observations[0, 0] = 100.0
                bank.expected_dx[0, 0, 0] = 3.0
            after = bank.SlotReadout(identity, attribute, attention)
            unknown = bank.SlotReadout(
                torch.tensor([[[0.0, 0.0, 1.0]]], device=self.device),
                attribute,
                attention)
            assert float(after[0, 0, 0].item()) > float(before[0, 0, 0].item()) + 3.0
            assert float(after[0, 0, 2].item()) > 2.5
            assert abs(float(unknown[0, 0, 0].item())) < 0.01
            assert abs(float(unknown[0, 0, 2].item())) < 0.01
            assert float(unknown[0, 0, 1].item()) > float(after[0, 0, 1].item())
            print("Object-usage posterior/open-set test passed.")
            return True
        except Exception as e:
            print(f"Object-usage posterior/open-set test failed: {e}")
            return False

    def TestEmptyObjectRelationalMemoryFinite(self):
        try:
            mem = self.MakeTinyMemory().train()
            visual = self.MakeVisualState(2, self.device)
            with torch.no_grad():
                visual.SemanticNodes["node_logits"][..., 0] = 100.0
                visual.SemanticNodes["node_logits"][..., 1] = -100.0
                visual.Auxiliary["PerceptualPresence"].zero_()
            attended = torch.randn(
                2, mem.input_dim, device=self.device, requires_grad=True)
            differentiable_inputs = [
                visual.ObjectTokens,
                visual.Auxiliary["PerceptualPresence"],
                visual.SemanticNodes["level_logits"],
                visual.SemanticNodes["object_class_logits"],
                visual.SemanticNodes["part_class_logits"],
                visual.SemanticNodes["identity_embed"],
                visual.SemanticNodes["SpatialState"],
                visual.SemanticNodes["bbox_2d"],]
            for value in differentiable_inputs:
                value.requires_grad_(True)
            ontology = mem.BuildOntologyObjectContext(attended, visual)
            embodied = mem.BuildEmbodiedMemory(
                attended,
                visual,
                ontology["object_features"])
            assert torch.isfinite(embodied).all()
            assert torch.count_nonzero(embodied).item() == 0
            embodied.square().sum().backward()
            for value in [attended] + differentiable_inputs:
                assert value.grad is not None
                assert torch.isfinite(value.grad).all()
            print("Empty-object relational-memory forward/backward test passed.")
            return True
        except Exception as e:
            print(f"Empty-object relational-memory test failed: {e}")
            return False

    def TestLowPresenceObjectsDoNotTriggerBoundary(self):
        try:
            mem = self.MakeTinyMemory().eval()
            visual = self.MakeVisualState(1, self.device)
            with torch.no_grad():
                visual.SemanticNodes["node_logits"][..., 0] = 5.0
                visual.SemanticNodes["node_logits"][..., 1] = 0.0
                visual.Auxiliary["PerceptualPresence"].zero_()
                visual.ObjectTokens.fill_(100.0)
                visual.MotionToken.zero_()
                visual.PredErrorToken.zero_()
                mem.previous_attention.zero_()
                mem.previous_intent.zero_()
                mem.previous_object_summary.zero_()
                mem.previous_motion_token.zero_()
                mem.event_age.zero_()
                mem.has_previous_event.fill_(True)
            zeros = torch.zeros(1, device=self.device)
            ontology = mem.BuildOntologyObjectContext(
                torch.zeros(1, mem.input_dim, device=self.device),
                visual)
            probability_a, boundary_a = mem.DetectEventBoundary(
                torch.zeros(1, mem.input_dim, device=self.device),
                visual,
                torch.zeros(1, 512, device=self.device),
                zeros,
                zeros,
                zeros,
                torch.ones(1, device=self.device),
                ontology)
            with torch.no_grad():
                visual.ObjectTokens.fill_(-100.0)
            ontology = mem.BuildOntologyObjectContext(
                torch.zeros(1, mem.input_dim, device=self.device),
                visual)
            probability_b, boundary_b = mem.DetectEventBoundary(
                torch.zeros(1, mem.input_dim, device=self.device),
                visual,
                torch.zeros(1, 512, device=self.device),
                zeros,
                zeros,
                zeros,
                torch.ones(1, device=self.device),
                ontology)
            assert torch.count_nonzero(mem.previous_object_summary).item() == 0
            assert torch.allclose(probability_a, probability_b)
            assert not bool(boundary_a.any().item())
            assert not bool(boundary_b.any().item())
            print("Low-presence object-boundary rejection test passed.")
            return True
        except Exception as e:
            print(f"Low-presence object-boundary rejection test failed: {e}")
            return False

    def TestImaginedMemoryVerificationPolicy(self):
        try:
            mem = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=8, ssmStateDim=8, memoryDim=8, memorySize=4,
                symSize=4, ltmSize=4, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            key = torch.zeros(1, 8, device=self.device); key[0, 0] = 1.0
            value = torch.zeros(1, 8, device=self.device); value[0, 1] = 1.0
            realm = torch.full(
                (1,),
                ONTOLOGY_REALM_EXTERNAL,
                device=self.device,
                dtype=torch.int8)
            common = dict(
                keySem=key, keyEpi=key, keyEpiState=key,
                valSem=value, valEpi=value,
                importance=torch.full((1,), 1.5, device=self.device),
                tdError=torch.ones(1, device=self.device),
                reward=torch.zeros(1, device=self.device),
                uncertainty=torch.zeros(1, device=self.device),
                risk=torch.zeros(1, device=self.device),
                confidence=torch.ones(1, device=self.device),
                realm=realm,
                episodicMetadata=UnknownEpisodicTypeMetadata(
                    realm, dtype=key.dtype),
                eventMask=torch.ones(1, device=self.device, dtype=torch.bool),
                episodeId=torch.zeros(1, device=self.device, dtype=torch.long),
                eventId=torch.ones(1, device=self.device, dtype=torch.long))
            mem.LtmOnlineStore(
                **common,
                sourceLabel=torch.ones(1, device=self.device, dtype=torch.int8),
                sourceConfidence=torch.full((1,), 0.1, device=self.device))
            assert int(mem.ltm.semantic.filled[0].item()) == 0
            assert int(mem.ltm.episodic.filled[0].item()) == 1
            mem.LtmOnlineStore(
                **common,
                sourceLabel=torch.zeros(1, device=self.device, dtype=torch.int8),
                sourceConfidence=torch.ones(1, device=self.device))
            assert int(mem.ltm.semantic.filled[0].item()) == 1
            assert int(mem.ltm.episodic.source[0, 0].item()) == MemoryType.SRC_MIXED
            assert float(mem.ltm.episodic.source_confidence[0, 0].item()) > 0.5
            print("Imagined-memory verification policy test passed.")
            return True
        except Exception as e:
            print(f"Imagined-memory verification policy test failed: {e}")
            return False

    def TestSaveStateFlushesPending(self):
        path = self.StatePath("memory_pending_transaction.pth")
        try:
            mem = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=8, ssmStateDim=8, memoryDim=8, memorySize=4,
                symSize=4, ltmSize=4, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            self.CallMemForward(mem, torch.randn(1, 8, device=self.device))
            assert len(mem.pending) > 0 and int(mem.memory_filled[0].item()) == 0
            mem.SaveState(str(path))
            assert not mem.pending and int(mem.memory_filled[0].item()) == 1
            restored = MemoryExtractor(**self.FilterKwargs(MemoryExtractor, dict(
                inputDim=8, ssmStateDim=8, memoryDim=8, memorySize=4,
                symSize=4, ltmSize=4, nsK=4, outputDim=8,
                gwsSlots=2, emotionDim=4))).to(self.device).eval()
            restored.LoadState(str(path), expectedBatch=1)
            assert int(restored.memory_filled[0].item()) == 1
            print("SaveState pending-transaction test passed.")
            return True
        except Exception as e:
            print(f"SaveState pending-transaction test failed: {e}")
            return False
        finally:
            try:
                path.unlink()
            except OSError:
                pass

    def TestCapacityNotReduced(self):
        try:
            mem = MemoryExtractor(memorySize=1, symSize=1, ltmSize=1, gwsSlots=1)
            count = sum(p.numel() for p in mem.parameters() if p.requires_grad)
            assert count >= 186_120_703, count
            assert len(mem.fusion.experts) == 4
            assert mem.A_full.numel() == 1024 * 1024
            mem.EnsureB(1)
            assert mem.fast_weights.shape == (1, 1024, 1024)
            print(f"Memory capacity guard passed: {count:,} trainable parameters.")
            return True
        except Exception as e:
            print(f"Memory capacity guard failed: {e}")
            return False

    def TestDeltaMergeKvIdentityTimeAndIdempotency(self):
        try:
            destination = self.MakeTinyMemory()
            key_old = torch.zeros(1, 8, device=self.device); key_old[0, 0] = 1.0
            key_new = torch.zeros(1, 8, device=self.device); key_new[0, 2] = 1.0
            value_old = torch.zeros(1, 8, device=self.device); value_old[0, 1] = 1.0
            value_new = torch.zeros(1, 8, device=self.device); value_new[0, 3] = 1.0
            one = torch.ones(1, device=self.device)
            mask = torch.ones(1, device=self.device, dtype=torch.bool)
            source = torch.zeros(1, device=self.device, dtype=torch.int8)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)
            emotion = torch.zeros(1, 4, device=self.device)

            destination.time_step.fill_(10)
            destination.KvWrite(
                key_old, value_old, one, emotion, one, source, one, mask,
                realm=realm)
            shadow = self.MakeTinyMemory()
            shadow.ImportState(destination.ExportState())
            shadow.time_step.fill_(11)
            shadow.KvWrite(
                key_new, value_new, one, emotion, one, source, one, mask,
                realm=realm)
            shadow.time_step.fill_(12)
            shadow.KvWrite(
                key_old, 2.0 * value_old, one, emotion, one, source, one, mask,
                realm=realm)
            envelope = self.MemoryDeltaEnvelope(shadow, 10, 12)

            destination.time_step.fill_(100)
            destination.MergeMemoryDelta(envelope)
            assert int(destination.memory_filled[0].item()) == 2
            old_index = int(torch.argmax(
                destination.memory_keys[0, :2, 0]).item())
            new_index = int(torch.argmax(
                destination.memory_keys[0, :2, 2]).item())
            assert torch.allclose(
                destination.memory_values[0, old_index],
                1.5 * value_old[0])
            assert int(destination.memory_steps[0, old_index].item()) == 10
            assert int(destination.memory_last_rehearsal_steps[0, old_index].item()) == 100
            assert int(destination.memory_steps[0, new_index].item()) == 99
            assert int(destination.memory_last_access_steps[0, new_index].item()) == 99
            assert int(destination.memory_merge_count[0, old_index].item()) == 1

            snapshot = destination.ExportDurableState()
            destination.time_step.fill_(105)
            destination.MergeMemoryDelta(envelope)
            for field in (
                "memory_filled", "memory_keys", "memory_values",
                "memory_steps", "memory_last_access_steps",
                "memory_last_rehearsal_steps", "memory_touch",
                "memory_merge_count",):
                assert torch.equal(destination.ExportDurableState()[field], snapshot[field])
            print("Delta KV identity/time/idempotency test passed.")
            return True
        except Exception as e:
            print(f"Delta KV identity/time/idempotency test failed: {e}")
            return False

    def TestDeltaSignatureTimelineBoundaries(self):
        try:
            mem = self.MakeTinyMemory()
            signature = torch.tensor(
                [3, 7, 2], device=self.device, dtype=torch.long)
            mem.merged_delta_signature.copy_(signature)
            mem.time_step.fill_(11)
            mem.SoftReset()
            assert torch.equal(mem.merged_delta_signature, signature)
            assert int(mem.time_step[0].item()) == 11

            mem.EnsureB(1)
            assert torch.equal(mem.merged_delta_signature, signature)
            mem.ResetEpisodeState(torch.zeros(
                1, device=self.device, dtype=torch.bool))
            assert torch.equal(mem.merged_delta_signature, signature)

            mem.EnsureB(2)
            assert bool((mem.merged_delta_signature == -1).all().item())
            mem.merged_delta_signature.copy_(signature)
            mem.ResetSteps()
            assert bool((mem.merged_delta_signature == -1).all().item())
            mem.merged_delta_signature.copy_(signature)
            mem.ReorderMemorySteps()
            assert bool((mem.merged_delta_signature == -1).all().item())
            mem.merged_delta_signature.copy_(signature)
            mem.ResetAll()
            assert bool((mem.merged_delta_signature == -1).all().item())
            print("Delta-signature timeline-boundary test passed.")
            return True
        except Exception as e:
            print(f"Delta-signature timeline-boundary test failed: {e}")
            return False

    def TestDeltaMergeSemanticStatistics(self):
        try:
            base = self.MakeTinyMemory()
            key_a = torch.zeros(1, 8, device=self.device); key_a[0, 0] = 1.0
            key_b = F.normalize(
                key_a + 0.05 * F.one_hot(
                    torch.tensor([1], device=self.device), 8).float(),
                dim=-1)
            value_a = torch.zeros(1, 8, device=self.device); value_a[0, 2] = 1.0
            value_b = torch.zeros(1, 8, device=self.device); value_b[0, 2] = 3.0
            one = torch.ones(1, device=self.device)
            source = torch.zeros(1, device=self.device, dtype=torch.int8)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)
            base.ltm.semantic.global_step.fill_(10)
            base.ltm.semantic.Store(
                key_a, value_a, one, source, sourceConfidence=one,
                realm=realm)

            shadow = self.MakeTinyMemory()
            shadow.ImportState(base.ExportState())
            shadow.time_step.fill_(12)
            shadow.ltm.semantic.global_step.fill_(12)
            shadow.ltm.semantic.Store(
                key_b, value_b, one, source, sourceConfidence=one,
                realm=realm)
            envelope = self.MemoryDeltaEnvelope(shadow, 10, 12)

            base.time_step.fill_(100)
            base.ltm.semantic.global_step.fill_(100)
            base.MergeMemoryDelta(envelope)
            semantic = base.ltm.semantic
            assert int(semantic.filled[0].item()) == 1
            assert float(semantic.prototype_count[0, 0].item()) == 2.0
            assert torch.allclose(
                semantic.vals[0, 0, 2],
                torch.tensor(2.0, device=self.device))
            assert torch.allclose(
                semantic.prototype_variance[0, 0, 2],
                torch.tensor(1.0, device=self.device))
            assert int(semantic.step[0, 0].item()) == 10
            assert int(semantic.last_rehearsal_step[0, 0].item()) == 100

            concurrent = self.MakeTinyMemory()
            concurrent.ImportState(base.ExportState())
            concurrent.merged_delta_signature.fill_(-1)
            concurrent.ltm.semantic.global_step.fill_(100)
            value_c = torch.zeros_like(value_a); value_c[0, 2] = 5.0
            concurrent.ltm.semantic.Store(
                key_b, value_c, one, source, sourceConfidence=one,
                realm=realm)
            before = {
                "key": concurrent.ltm.semantic.keys[0, 0].clone(),
                "value": concurrent.ltm.semantic.vals[0, 0].clone(),
                "count": concurrent.ltm.semantic.prototype_count[0, 0].clone(),
                "variance": concurrent.ltm.semantic.prototype_variance[0, 0].clone(),}
            concurrent.MergeMemoryDelta(envelope)
            assert torch.equal(concurrent.ltm.semantic.keys[0, 0], before["key"])
            assert torch.equal(concurrent.ltm.semantic.vals[0, 0], before["value"])
            assert torch.equal(concurrent.ltm.semantic.prototype_count[0, 0], before["count"])
            assert torch.equal(concurrent.ltm.semantic.prototype_variance[0, 0], before["variance"])
            print("Delta semantic-statistics consistency test passed.")
            return True
        except Exception as e:
            print(f"Delta semantic-statistics consistency test failed: {e}")
            return False

    def TestConcurrentDeltaPreservesMainUpdates(self):
        try:
            base = self.MakeTinyMemory()
            key = torch.zeros(1, 8, device=self.device); key[0, 0] = 1.0
            base_value = torch.zeros(1, 8, device=self.device); base_value[0, 1] = 1.0
            shadow_value = torch.zeros_like(base_value); shadow_value[0, 1] = 3.0
            main_value = torch.zeros_like(base_value); main_value[0, 1] = 5.0
            one = torch.ones(1, device=self.device)
            mask = torch.ones(1, device=self.device, dtype=torch.bool)
            real = torch.full(
                (1,), MemoryType.SRC_REAL,
                device=self.device,
                dtype=torch.int8)
            imagined = torch.full(
                (1,), MemoryType.SRC_IMAGINE,
                device=self.device,
                dtype=torch.int8)
            emotion = torch.zeros(1, 4, device=self.device)
            realm = torch.full(
                (1,),
                ONTOLOGY_REALM_EXTERNAL,
                device=self.device,
                dtype=torch.int8)

            base.time_step.fill_(10)
            base.ltm.semantic.global_step.fill_(10)
            base.KvWrite(
                key, base_value, one, emotion, one, real, one, mask,
                realm=realm)
            base.ltm.semantic.Store(
                key, base_value, one, real, sourceConfidence=one,
                realm=realm)
            fork_state = base.ExportState()

            shadow = self.MakeTinyMemory()
            shadow.ImportState(fork_state)
            shadow.time_step.fill_(12)
            shadow.ltm.semantic.global_step.fill_(12)
            shadow.KvWrite(
                key, shadow_value, one, emotion, one,
                imagined, one, mask,
                realm=realm)
            shadow.ltm.semantic.Store(
                key, shadow_value, one, imagined,
                sourceConfidence=one,
                realm=realm)
            envelope = self.MemoryDeltaEnvelope(shadow, 10, 12)

            main = self.MakeTinyMemory()
            main.ImportState(fork_state)
            main.time_step.fill_(100)
            main.ltm.semantic.global_step.fill_(100)
            main.KvWrite(
                key, main_value, one, emotion, one, real, one, mask,
                realm=realm)
            main.ltm.semantic.Store(
                key, main_value, one, real, sourceConfidence=one,
                realm=realm)
            main.MergeMemoryDelta(envelope)

            expected = torch.tensor(3.0, device=self.device)
            assert torch.equal(main.memory_values[0, 0, 1], expected)
            assert int(main.memory_merge_count[0, 0].item()) == 1
            assert int(main.memory_source[0, 0].item()) == MemoryType.SRC_REAL
            assert torch.equal(main.ltm.semantic.vals[0, 0, 1], expected)
            assert float(main.ltm.semantic.prototype_count[0, 0].item()) == 2.0
            assert int(main.ltm.semantic.source[0, 0].item()) == MemoryType.SRC_REAL
            print("Concurrent delta main-update protection test passed.")
            return True
        except Exception as e:
            print(f"Concurrent delta main-update protection test failed: {e}")
            return False

    def TestDeltaMergeEpisodicLinksAndSymbolicMetadata(self):
        try:
            destination = self.MakeTinyMemory()
            epi = destination.ltm.episodic
            one = torch.ones(1, device=self.device)
            for event in range(3):
                epi.global_step.fill_(10 + event)
                key = torch.zeros(1, 8, device=self.device); key[0, event] = 1.0
                value = key * float(event + 1)
                source = torch.tensor([
                    MemoryType.SRC_IMAGINE
                    if event == 1 else MemoryType.SRC_REAL],
                    device=self.device,
                    dtype=torch.int8)
                confidence = torch.tensor([
                    0.2 if event == 1 else 1.0],
                    device=self.device)
                realm = torch.full(
                    (1,),
                    ONTOLOGY_REALM_EXTERNAL,
                    device=self.device,
                    dtype=torch.int8)
                typed_metadata = UnknownEpisodicTypeMetadata(
                    realm,
                    dtype=key.dtype)
                typed_metadata["agency_id"].fill_(
                    ONTOLOGY_AGENCY_EXTERNAL)
                typed_metadata["motion_layer_prob"][0, event % 5] = 1.0
                typed_metadata["observed_surface_parent_index"].fill_(event)
                typed_metadata["surface_uv"][0] = torch.tensor(
                    [0.1 * event, 0.2 * event], device=self.device)
                typed_metadata["surface_uv_confidence"].fill_(
                    0.5 + 0.1 * event)
                typed_metadata["verification"].fill_(0.6 + 0.1 * event)
                typed_metadata["content_change"].fill_(0.2 * event)
                epi.Store(
                    key,
                    value,
                    torch.zeros(1, device=self.device),
                    one,
                    source,
                    sourceConfidence=confidence,
                    episodeId=torch.tensor([7], device=self.device),
                    eventId=torch.tensor([100 + event], device=self.device),
                    realm=realm,
                    typedMetadata=typed_metadata)

            sym = destination.sym_mem
            sym.global_step.fill_(10)
            sym_key = F.normalize(torch.ones(1, 4, device=self.device), dim=-1)
            sym_value = torch.arange(4, device=self.device, dtype=torch.float32).view(1, 4)
            real = torch.zeros(1, device=self.device, dtype=torch.int8)
            sym.Store(sym_key, sym_value, one, real, one)

            shadow = self.MakeTinyMemory()
            shadow.ImportState(destination.ExportState())
            shadow.time_step.fill_(20)
            shadow.ltm.episodic.global_step.fill_(20)
            middle_key = torch.zeros(1, 8, device=self.device); middle_key[0, 1] = 1.0
            shadow.ltm.episodic.VerifyWithRealEvidence(
                middle_key,
                torch.ones(1, device=self.device, dtype=torch.bool),
                realm)
            shadow.ltm.episodic.surface_uv[0, 1] = torch.tensor(
                [0.35, 0.65], device=self.device)
            shadow.ltm.episodic.surface_uv_confidence[0, 1] = 0.77
            shadow.ltm.episodic.verification[0, 1] = 0.88
            shadow.ltm.episodic.content_change[0, 1] = 0.93
            shadow.sym_mem.global_step.fill_(20)
            shadow.sym_mem.P_vals[0, 0].add_(1.0)
            shadow.sym_mem.prio[0, 0] = 0.8
            shadow.sym_mem.touch[0, 0] = 7
            shadow.sym_mem.last_access_step[0, 0] = 18
            shadow.sym_mem.last_rehearsal_step[0, 0] = 20
            shadow.sym_mem.source[0, 0] = MemoryType.SRC_MIXED
            shadow.sym_mem.source_confidence[0, 0] = 0.65
            envelope = self.MemoryDeltaEnvelope(shadow, 12, 20)

            destination.time_step.fill_(100)
            destination.ltm.episodic.global_step.fill_(100)
            destination.sym_mem.global_step.fill_(100)
            destination.MergeMemoryDelta(envelope)
            epi = destination.ltm.episodic
            matches = (
                (epi.episode_id[0, :epi.filled[0]] == 7)
                & (epi.event_id[0, :epi.filled[0]] == 101)
            ).nonzero(as_tuple=False).flatten()
            assert matches.numel() == 1
            middle = int(matches.item())
            assert int(epi.filled[0].item()) == 3
            assert int(epi.source[0, middle].item()) == MemoryType.SRC_MIXED
            assert torch.allclose(
                epi.source_confidence[0, middle],
                torch.tensor(0.6, device=self.device))
            assert int(epi.last_rehearsal_step[0, middle].item()) == 100
            assert torch.allclose(
                epi.surface_uv[0, middle],
                torch.tensor([0.35, 0.65], device=self.device))
            assert torch.allclose(
                epi.surface_uv_confidence[0, middle],
                torch.tensor(0.77, device=self.device))
            assert torch.allclose(
                epi.verification[0, middle],
                torch.tensor(0.88, device=self.device))
            assert torch.allclose(
                epi.content_change[0, middle],
                torch.tensor(0.93, device=self.device))
            sequence, valid = epi.RetrieveSequence(
                torch.tensor([middle], device=self.device))
            assert valid.tolist() == [[True, True, True]]
            assert torch.equal(sequence[0, 0], epi.vals[0, int(epi.prev_index[0, middle].item())])
            assert torch.equal(sequence[0, 2], epi.vals[0, int(epi.next_index[0, middle].item())])
            assert int(epi.prev_generation[0, middle].item()) == int(
                epi.slot_generation[0, epi.prev_index[0, middle]].item())
            assert int(epi.next_generation[0, middle].item()) == int(
                epi.slot_generation[0, epi.next_index[0, middle]].item())

            sym = destination.sym_mem
            assert int(sym.filled[0].item()) == 1
            assert int(sym.created_step[0, 0].item()) == 10
            assert int(sym.last_access_step[0, 0].item()) == 98
            assert int(sym.last_rehearsal_step[0, 0].item()) == 100
            assert int(sym.touch[0, 0].item()) == 7
            assert int(sym.source[0, 0].item()) == MemoryType.SRC_MIXED
            assert torch.equal(sym.P_vals[0, 0], sym_value[0] + 1.0)
            print("Delta episodic-link/symbolic-metadata test passed.")
            return True
        except Exception as e:
            print(f"Delta episodic-link/symbolic-metadata test failed: {e}")
            return False

    def TestTerminalSealLaneIsolationAndReset(self):
        try:
            memory = self.MakeTinyMemory(batch=2)
            B, D = 2, 8
            key = torch.zeros(B, D, device=self.device)
            value = torch.zeros(B, D, device=self.device)
            importance = torch.tensor([0.2, 0.3], device=self.device)
            emotion = torch.zeros(B, 4, device=self.device)
            reward = torch.zeros(B, device=self.device)
            source = torch.zeros(B, device=self.device, dtype=torch.int8)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)
            confidence = torch.ones(B, device=self.device)
            write_mask = torch.tensor([False, True], device=self.device)
            ttl = torch.full((B,), 4, device=self.device, dtype=torch.long)
            episode = torch.tensor([2, 3], device=self.device, dtype=torch.long)
            event = torch.tensor([7, 9], device=self.device, dtype=torch.long)
            memory.pending = [
                ("kv", (
                    key, value, importance, emotion, reward, source,
                    confidence, realm, write_mask)),
                ("gws", (
                    key, value, importance, ttl, source,
                    confidence, write_mask)),
                ("ltm", (
                    key, key, key, value, value, importance,
                    reward, reward, source, reward, reward, confidence,
                    confidence, realm, write_mask, episode, event,
                    UnknownEpisodicTypeMetadata(
                        realm, dtype=key.dtype))),
                ("ns", (
                    torch.zeros(B, 4, device=self.device),
                    torch.zeros(B, 4, device=self.device),
                    importance,
                    source,
                    confidence,
                    write_mask)),]
            done = torch.tensor([True, False], device=self.device)
            memory.SealPendingRows(done)
            by_kind = {kind: payload for kind, payload in memory.pending}
            for kind, priority_index, mask_index, floor in (
                ("kv", 2, 8, 1.0),
                ("gws", 2, 6, 1.0),
                ("ltm", 5, 14, 1.5),
                ("ns", 2, 5, 1.0),):
                payload = by_kind[kind]
                assert torch.allclose(
                    payload[priority_index][0],
                    torch.tensor(floor, device=self.device))
                assert torch.allclose(
                    payload[priority_index][1],
                    torch.tensor(0.3, device=self.device))
                assert payload[mask_index].tolist() == [True, True]
            assert by_kind["ltm"][16].tolist() == [8, 9]

            memory.FlushPendingWrites()
            second_mask = torch.zeros(B, device=self.device, dtype=torch.bool)
            memory.pending = [("ltm", (
                key, key, key, value, value, importance,
                reward, reward, source, reward, reward, confidence,
                confidence, realm, second_mask, episode,
                memory.event_id.detach().clone(),
                UnknownEpisodicTypeMetadata(
                    realm, dtype=key.dtype)))]
            memory.SealPendingRows(done)
            assert memory.pending[0][1][16].tolist() == [9, 9]
            memory.FlushPendingWrites()
            epi = memory.ltm.episodic
            lane_zero_events = epi.event_id[0, :epi.filled[0]]
            assert lane_zero_events.unique().numel() == lane_zero_events.numel()
            assert set(lane_zero_events.tolist()) == {8, 9}
            memory.ResetAll()
            assert not memory.pending
            assert torch.count_nonzero(memory.memory_keys).item() == 0
            print("Terminal-seal lane-isolation/reset test passed.")
            return True
        except Exception as e:
            print(f"Terminal-seal lane-isolation/reset test failed: {e}")
            return False

    def TestNullMemoryRejectsContradictoryCue(self):
        try:
            query = torch.zeros(1, 8, device=self.device); query[0, 0] = 1.0
            opposite = -query
            value = torch.zeros(1, 8, device=self.device); value[0, 3] = 1.0
            one = torch.ones(1, device=self.device)
            source = torch.zeros(1, device=self.device, dtype=torch.int8)
            realm = torch.full_like(
                source,
                ONTOLOGY_REALM_EXTERNAL)

            semantic = SemanticLTM(8, 4).to(self.device)
            semantic.Store(
                opposite,
                value,
                one,
                source,
                sourceConfidence=one,
                realm=realm)
            touch_before = semantic.touch.clone()
            rejected = semantic.Retrieve(query, topk=1)
            assert torch.equal(semantic.touch, touch_before)
            accepted = semantic.Retrieve(opposite, topk=1)
            assert int(semantic.touch[0, 0].item()) == int(touch_before[0, 0].item()) + 1
            assert float(rejected.norm().item()) < 0.05
            assert float(accepted.norm().item()) > 0.95

            episodic = EpisodicLTM(8, 4).to(self.device)
            episodic.Store(
                opposite,
                value,
                torch.zeros(1, device=self.device),
                one,
                source,
                sourceConfidence=one,
                realm=realm,
                typedMetadata=UnknownEpisodicTypeMetadata(
                    realm, dtype=opposite.dtype))
            assert int(episodic.RetrieveSeedIndex(query)[0].item()) == -1
            assert int(episodic.RetrieveSeedIndex(opposite)[0].item()) == 0
            print("Null-memory contradictory-cue rejection test passed.")
            return True
        except Exception as e:
            print(f"Null-memory contradictory-cue rejection test failed: {e}")
            return False

    def TestNullMemoryCandidateCountInvariant(self):
        try:
            dim, count = 8, 7
            query = torch.zeros(1, dim, device=self.device)
            query[0, 0] = 1.0
            weak_key = torch.zeros(count, dim, device=self.device)
            weak_key[:, 0] = -0.05
            weak_key[:, 1] = math.sqrt(1.0 - 0.05 ** 2)
            weak_value = torch.zeros(count, dim, device=self.device)
            weak_value[:, 3] = 1.0

            symbolic = SymbolicMemory(dim, capacity=8).to(self.device)
            workspace = GlobalWorkspace(dim, slots=8).to(self.device)
            semantic = SemanticLTM(dim, capacity=8).to(self.device)
            episodic = EpisodicLTM(dim, capacity=8).to(self.device)
            memory = self.MakeTinyMemory()
            with torch.no_grad():
                symbolic.filled.fill_(count)
                symbolic.P_keys[0, :count] = weak_key
                symbolic.P_vals[0, :count] = weak_value
                symbolic.prio[0, :count] = 1.0
                symbolic.source_confidence[0, :count] = 1.0

                workspace.keys[0, :count] = weak_key
                workspace.vals[0, :count] = weak_value
                workspace.priority[0, :count] = 1.0
                workspace.ttl[0, :count] = 5
                workspace.source_confidence[0, :count] = 1.0

                semantic.filled.fill_(count)
                semantic.keys[0, :count] = weak_key
                semantic.vals[0, :count] = weak_value
                semantic.prio[0, :count] = 1.0
                semantic.source_confidence[0, :count] = 1.0

                episodic.filled.fill_(count)
                episodic.keys[0, :count] = weak_key
                episodic.state_keys[0, :count] = weak_key
                episodic.vals[0, :count] = weak_value
                episodic.prio[0, :count] = 1.0
                episodic.source_confidence[0, :count] = 1.0

                memory.memory_filled.fill_(count)
                memory.memory_keys[0, :count] = weak_key
                memory.memory_values[0, :count] = weak_value
                memory.memory_importance[0, :count] = 1.0
                memory.memory_source_confidence[0, :count] = 1.0

            touch_before = (
                symbolic.touch.clone(),
                workspace.touch.clone(),
                semantic.touch.clone(),
                episodic.touch.clone(),
                memory.memory_touch.clone(),)
            outputs = (
                symbolic.Retrieve(query, topK=count),
                workspace.Attend(query, topk=count),
                semantic.Retrieve(query, topk=count),
                episodic.Retrieve(query, topk=count),
                memory.Retrieve(
                    query,
                    fusionGate=torch.zeros(1, device=self.device),
                    importance=torch.ones(1, device=self.device),
                    localGate=torch.zeros(1, device=self.device),
                    emotion=torch.zeros(1, memory.emotion_dim, device=self.device),
                    tdError=torch.zeros(1, device=self.device)),)
            for output in outputs:
                assert torch.count_nonzero(output).item() == 0
            assert torch.equal(symbolic.touch, touch_before[0])
            assert torch.equal(workspace.touch, touch_before[1])
            assert torch.equal(semantic.touch, touch_before[2])
            assert torch.equal(episodic.touch, touch_before[3])
            assert torch.equal(memory.memory_touch, touch_before[4])
            assert int(episodic.RetrieveSeedIndex(query)[0].item()) == -1

            scores = torch.tensor(
                [[1.0, -0.2, -0.3]],
                device=self.device,
                requires_grad=True)
            weights, accepted, evidence = NullGatedTopKWeights(
                scores,
                torch.tensor(0.0, device=self.device),
                torch.ones_like(scores, dtype=torch.bool))
            assert accepted.tolist() == [[True, False, False]]
            assert torch.count_nonzero(weights[:, 1:]).item() == 0
            assert float(weights[0, 0].item()) > 0.0
            assert float(evidence.item()) == 1.0
            (weights.sum() + evidence.sum()).backward()
            assert scores.grad is not None and torch.isfinite(scores.grad).all()
            print("Null-memory candidate-count invariance test passed.")
            return True
        except Exception as e:
            print(f"Null-memory candidate-count invariance test failed: {e}")
            return False

    def TestNullEvidenceBlocksFilmBias(self):
        try:
            memory = self.MakeTinyMemory().eval()
            D = memory.memory_dim
            with torch.no_grad():
                memory.memory_filled.fill_(1)
                memory.gws.priority[:, 0] = 1.0
                memory.gws.ttl[:, 0] = 5
                memory.ltm.semantic.filled.fill_(1)
                memory.ltm.episodic.filled.fill_(1)
                for film in (
                    memory.film_mem,
                    memory.film_gws,
                    memory.film_sem,
                    memory.film_epi,):
                    film[-1].weight.zero_()
                    film[-1].bias.zero_()
                    film[-1].bias[D:] = 1.0

            def ZeroRecall(module, query, *args, returnEvidence=False, **kwargs):
                out = query.new_zeros(query.size(0), D)
                evidence = query.new_zeros(query.size(0), 1)
                return (out, evidence) if returnEvidence else out

            def ZeroLtm(module, query, *args, returnEvidence=False, **kwargs):
                out = query.new_zeros(query.size(0), D)
                evidence = query.new_zeros(query.size(0), 1)
                if returnEvidence:
                    return out, out.clone(), evidence, evidence.clone()
                return out, out.clone()

            memory.Retrieve = MethodType(ZeroRecall, memory)
            memory.gws.Attend = MethodType(ZeroRecall, memory.gws)
            memory.ltm.Retrieve = MethodType(ZeroLtm, memory.ltm)
            memory.ltm.episodic.Retrieve = MethodType(
                ZeroRecall,
                memory.ltm.episodic)
            memory.ltm.RetrieveEpisodeSequence = MethodType(
                ZeroRecall,
                memory.ltm)

            captured = {}
            handle = memory.fusion.register_forward_pre_hook(
                lambda module, args: captured.update(fusion_input=args[0].detach().clone()))
            self.CallMemForward(
                memory,
                torch.randn(1, memory.input_dim, device=self.device))
            handle.remove()
            fusion_input = captured["fusion_input"]
            assert torch.count_nonzero(fusion_input[:, :3 * D]).item() == 0
            print("Null-evidence FiLM/fuser boundary test passed.")
            return True
        except Exception as e:
            print(f"Null-evidence FiLM/fuser boundary test failed: {e}")
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

    def TestSignatureBoundPlanAndSkillCache(self):
        try:
            memory = self.MakeTinyMemory()
            plan = torch.randn(2, 7, device=self.device)
            skill = torch.randn(2, 5, device=self.device)
            memory.CachePlan("active", plan, "shape-a")
            memory.CacheSkill("refine", skill, "shape-a")
            recalled_plan = memory.RecallPlan("active", "shape-a")
            recalled_skill = memory.RecallSkill("refine", "shape-a")
            recalled_plan["feature"].zero_()
            recalled_skill.zero_()
            if not torch.equal(
                memory.RecallPlan("active", "shape-a")["feature"],
                plan):
                return False
            if not torch.equal(memory.RecallSkill("refine", "shape-a"), skill):
                return False
            memory.AgePlanCache()
            if not torch.equal(
                memory.RecallPlan("active", "shape-a")["age"],
                torch.ones(2, device=self.device, dtype=torch.long)
            ):
                return False
            memory.RecordFailureEpisode(
                torch.randn(2, 8, device=self.device),
                torch.randn(2, 3, device=self.device),
                "shape-a")
            cache_state = memory.ExportCognitiveCacheState()
            restored = self.MakeTinyMemory(batch=2)
            restored.ImportCognitiveCacheState(
                cache_state,
                modelSignature="shape-a",
                batchSize=2)
            restored_plan = restored.RecallPlan("active", "shape-a")
            if (
                restored_plan is None
                or not torch.equal(
                    restored_plan["age"],
                    torch.ones(2, device=self.device, dtype=torch.long))
                or not torch.equal(restored_plan["feature"], plan)
                or len(restored.SampleOfflineReplay(1, "shape-a")) != 1
            ):
                return False
            rejected = 0
            for recall in (
                lambda: memory.RecallPlan("active", "shape-b"),
                lambda: memory.RecallSkill("refine", "shape-b"),
                lambda: memory.CachePlan("active", plan, "shape-b"),
                lambda: memory.CacheSkill("refine", skill, "shape-b")):
                try:
                    recall()
                except ValueError:
                    rejected += 1
            return rejected == 4
        except Exception as error:
            print(f"SignatureBoundPlanAndSkillCache error: {error}")
            return False

    def TestReplayAndTransitionEpisodes(self):
        try:
            memory = self.MakeTinyMemory()
            context = torch.randn(2, 8, device=self.device)
            outcome = torch.randn(2, 4, device=self.device)
            memory.RecordHierarchyTransition(
                context,
                torch.tensor([0.2, 0.8], device=self.device),
                torch.tensor([0.0, 1.0], device=self.device),
                "shape-a")
            memory.RecordFailureEpisode(
                context,
                outcome,
                "shape-a")
            memory.RecordCounterfactualEpisode(
                context,
                -outcome,
                "shape-a")
            replay = memory.SampleOfflineReplay(8, "shape-a")
            if len(replay) != 3:
                return False
            kinds = {item["kind"] for item in replay}
            if kinds != {"hierarchyTransition", "failure", "counterfactual"}:
                return False
            replay[0]["context"].zero_()
            replay_again = memory.SampleOfflineReplay(8, "shape-a")
            if not bool(any(item["context"].abs().sum().item() > 0 for item in replay_again)):
                return False
            try:
                memory.SampleOfflineReplay(1, "shape-b")
                return False
            except ValueError:
                return True
        except Exception as error:
            print(f"ReplayAndTransitionEpisodes error: {error}")
            return False

    def TestSemanticReconsolidation(self):
        try:
            memory = self.MakeTinyMemory()
            memory.EnsureB(1)
            query = F.normalize(
                torch.randn(1, memory.memory_dim, device=self.device),
                dim=-1)
            original = torch.randn(1, memory.memory_dim, device=self.device)
            revised = torch.randn(1, memory.memory_dim, device=self.device)
            memory.ltm.semantic.Store(
                query,
                original,
                torch.ones(1, device=self.device),
                realm=torch.full(
                    (1,),
                    ONTOLOGY_REALM_EXTERNAL,
                    device=self.device,
                    dtype=torch.int8),
                writeMask=torch.ones(1, device=self.device, dtype=torch.bool))
            before = memory.ltm.semantic.vals[0, 0].clone()
            updated = memory.ReconsolidateSemantic(
                query,
                revised,
                torch.ones(1, device=self.device))
            after = memory.ltm.semantic.vals[0, 0]
            if not bool(updated.all().item()):
                return False
            if torch.equal(before, after):
                return False
            memory.SemanticConsolidation(topk=1)
            return bool(torch.isfinite(after).all().item())
        except Exception as error:
            print(f"SemanticReconsolidation error: {error}")
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
            "ExportConsciousBank": self.TestExportConsciousBank(),
            "DurableStatePreservesBatchLanes": self.TestDurableStatePreservesBatchLanes(),
            "DurableStateRejectsNonCurrentSchema": self.TestDurableStateRejectsNonCurrentSchema(),
            "GlobalWorkspaceLaneIsolation": self.TestGlobalWorkspaceLaneIsolation(),
            "SymbolicMemoryLaneIsolation": self.TestSymbolicMemoryLaneIsolation(),
            "SourceConfidenceOrdering": self.TestSourceConfidenceOrdering(),
            "UnifiedMemoryScoreNegativeSimilarityMonotonic": self.TestUnifiedMemoryScoreNegativeSimilarityMonotonic(),
            "SourceEvidenceMassPreserved": self.TestSourceEvidenceMassPreserved(),
            "EmptyRecallRowsStrictZero": self.TestEmptyRecallRowsStrictZero(),
            "KvStatsMixedLaneCounts": self.TestKvStatsMixedLaneCounts(),
            "CreationTimeImmutableOnRetrieve": self.TestCreationTimeIsImmutableOnRetrieve(),
            "FullRankHeteroAssociativeHebbian": self.TestFullRankHeteroAssociativeHebbian(),
            "StableFullMatrixWorkingState": self.TestStableFullMatrixWorkingState(),
            "StreamingCompressionMetadata": self.TestStreamingCompressionMetadata(),
            "EventBoundarySelectiveWriting": self.TestEventBoundarySelectiveWriting(),
            "EpisodicSequenceLinks": self.TestEpisodicSequenceLinks(),
            "VerifiedEpisodicSourceRelinksSequence": self.TestVerifiedEpisodicSourceRelinksSequence(),
            "SemanticPrototypeAbstraction": self.TestSemanticPrototypeAbstraction(),
            "OntologyTypedMemoryIsolation": self.TestOntologyTypedMemoryIsolation(),
            "OntologyContextAndScreenBoundary": self.TestOntologyContextAndScreenBoundary(),
            "ForwardFlushPersistsFocusedEpisodicMetadata": (
                self.TestForwardFlushPersistsFocusedEpisodicMetadata()),
            "ObjectUsagePosteriorAndUnknown": self.TestObjectUsagePosteriorAndUnknown(),
            "EmptyObjectRelationalMemoryFinite": self.TestEmptyObjectRelationalMemoryFinite(),
            "LowPresenceObjectsDoNotTriggerBoundary": self.TestLowPresenceObjectsDoNotTriggerBoundary(),
            "ImaginedMemoryVerificationPolicy": self.TestImaginedMemoryVerificationPolicy(),
            "SaveStateFlushesPending": self.TestSaveStateFlushesPending(),
            "CapacityNotReduced": self.TestCapacityNotReduced(),
            "DeltaMergeKvIdentityTimeAndIdempotency": self.TestDeltaMergeKvIdentityTimeAndIdempotency(),
            "DeltaSignatureTimelineBoundaries": self.TestDeltaSignatureTimelineBoundaries(),
            "DeltaMergeSemanticStatistics": self.TestDeltaMergeSemanticStatistics(),
            "ConcurrentDeltaPreservesMainUpdates": self.TestConcurrentDeltaPreservesMainUpdates(),
            "DeltaMergeEpisodicLinksAndSymbolicMetadata": self.TestDeltaMergeEpisodicLinksAndSymbolicMetadata(),
            "TerminalSealLaneIsolationAndReset": self.TestTerminalSealLaneIsolationAndReset(),
            "NullMemoryRejectsContradictoryCue": self.TestNullMemoryRejectsContradictoryCue(),
            "NullMemoryCandidateCountInvariant": self.TestNullMemoryCandidateCountInvariant(),
            "NullEvidenceBlocksFilmBias": self.TestNullEvidenceBlocksFilmBias(),
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
            "FullMergeRebasesIndependentClocks": self.TestFullMergeRebasesIndependentClocks(),
            "MalformedImportAndMergeAreAtomic": self.TestMalformedImportAndMergeAreAtomic(),
            "MergeStateCrossCapacityAndMalformedReject": self.TestMergeStateCrossCapacityAndMalformedReject(),
            "AutoCompress": self.TestAutoCompress(),
            "ResetAndSoftReset": self.TestResetAndSoftReset(),
            "PartialEpisodeResetPreservesSharedState": self.TestPartialEpisodeResetPreservesSharedState(),
            "SignatureBoundPlanAndSkillCache": self.TestSignatureBoundPlanAndSkillCache(),
            "ReplayAndTransitionEpisodes": self.TestReplayAndTransitionEpisodes(),
            "SemanticReconsolidation": self.TestSemanticReconsolidation(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NumericalStability": self.TestNumericalStability(),
            "AllTrainableParamsHaveGrad": self.TestAllTrainableParamsHaveGrad(),
            "SymbolicViolationBackprop": self.TestSymbolicViolationBackprop(),
            "LossDecreases": self.TestLossDecreases()}

        passed = sum(1 for v in results.values() if v)
        print(f"\nMemory module tests: {passed}/{len(results)} passed.")
        return results
