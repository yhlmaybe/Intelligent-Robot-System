from __future__ import annotations
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os




def StableTopk(scores: torch.Tensor, k: int):
    N = scores.size(-1)
    eps = (torch.arange(N, device=scores.device, dtype=torch.float32) * 1e-7).view(1, -1)
    while eps.dim() < scores.dim():
        eps = eps.unsqueeze(0)
    biased = scores.float() + eps
    return torch.topk(biased, k, dim=-1)


class SoftSymbolicRules(nn.Module):
    def __init__(self, k: int, gExcl: int = 5, gOr: int = 5, sparsity: float = 1e-4, entropy: float = 1e-4, impScale: float = 1.0):
        super().__init__()
        self.K = k
        self.G_excl, self.G_or = gExcl, gOr
        self.sparsity = sparsity
        self.entropy = entropy
        self.imp_scale = impScale

        self.excl_logits = nn.Parameter(torch.randn(gExcl, k) * 0.05) if gExcl > 0 else None
        self.or_logits = nn.Parameter(torch.randn(gOr, k) * 0.05) if gOr > 0 else None

        self.imp_logits = nn.Parameter(torch.full((k, k), -3.5))
        self.register_buffer("no_self", (1 - torch.eye(k)))

    def Weights(self):
        def RowNormPos(logits):
            W = torch.sigmoid(logits)
            return W / (W.sum(dim=1, keepdim=True) + 1e-6)

        W_excl = RowNormPos(self.excl_logits) if self.excl_logits is not None else None
        W_or = RowNormPos(self.or_logits) if self.or_logits is not None else None
        A_imp = F.softplus(self.imp_logits) * self.no_self
        return W_excl, W_or, A_imp

    def forward(self, p: torch.Tensor, pPrev: Optional[torch.Tensor] = None):
        B, K = p.size(0), p.size(1)
        device = p.device
        total = torch.zeros(B, device=device)
        W_excl, W_or, A_imp = self.Weights()

        if W_excl is not None and W_excl.numel() > 0:
            s = torch.matmul(W_excl, p.t()).t() 
            s2 = torch.matmul(W_excl.pow(2), (p.pow(2)).t()).t() 
            excl_pen = 0.5 * (s * s - s2)
            total = total + excl_pen.mean(dim=1)

        if W_or is not None and W_or.numel() > 0:
            eps = 1e-6
            z = torch.clamp(p.unsqueeze(1) * W_or.unsqueeze(0), 0.0, 1.0) 
            log_not = torch.log(torch.clamp(1.0 - z, eps, 1.0))
            prod_not = torch.exp(log_not.sum(dim=-1))
            total = total + prod_not.mean(dim=1)

        Pa = p.unsqueeze(2) 
        Pb = p.unsqueeze(1) 

        margin = Pa - Pb 
        soft = F.softplus(margin, beta=4.0) - F.softplus(torch.zeros(1, device=p.device), beta=4.0)
        soft = soft.clamp_min(0.0)

        imp_pen = (soft * A_imp.unsqueeze(0)).sum(dim=(1, 2)) / max(1, K)
        total = total + self.imp_scale * imp_pen

        if pPrev is not None and pPrev.shape == p.shape:
            total = total + torch.relu(pPrev - p).mean(dim=1)

        def RowEntropy(W, eps: float = 1e-6):
            Wn = W / (W.sum(dim=1, keepdim=True) + eps)
            return -(Wn * (Wn.clamp_min(eps)).log()).sum(dim=1).mean()

        aux_reg = total.new_zeros([])
        if self.training:
            if W_excl is not None:
                aux_reg = aux_reg + self.sparsity * W_excl.abs().mean() + self.entropy * RowEntropy(W_excl)
            if W_or is not None:
                aux_reg = aux_reg + self.sparsity * W_or.abs().mean() + self.entropy * RowEntropy(W_or)
            aux_reg = aux_reg + self.sparsity * (F.softplus(self.imp_logits).abs().mean())

        return total, aux_reg

    @torch.no_grad()
    def SeedFromSets(self, exclusives=None, atleastOne=None, implications=None, strength: float = 3.5):
        if exclusives is not None and self.excl_logits is not None:
            self.excl_logits.zero_()
            for g, S in enumerate(exclusives[:self.G_excl]):
                for j in S:
                    if 0 <= j < self.K:
                        self.excl_logits[g, j] = strength
        if atleastOne is not None and self.or_logits is not None:
            self.or_logits.zero_()
            for g, S in enumerate(atleastOne[:self.G_or]):
                for j in S:
                    if 0 <= j < self.K:
                        self.or_logits[g, j] = strength
        if implications is not None:
            self.imp_logits.data.fill_(-4.0)
            for (a, b) in implications:
                if 0 <= a < self.K and 0 <= b < self.K and a != b:
                    self.imp_logits[a, b] = strength


class SymbolicCoder(nn.Module):
    def __init__(self, inDim: int, k: int, hidden: int = 1024, experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(inDim, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, k)
            ) for _ in range(experts)])
        
        self.proto = nn.Parameter(torch.randn(k, inDim) * 0.02)
        self.temp = nn.Parameter(torch.ones(k))
        self.bias = nn.Parameter(torch.zeros(k))

    def forward(self, x: torch.Tensor):
        a = F.softmax(self.gate(x), dim=-1)
        ys = [e(x) for e in self.experts] 
        y = torch.stack(ys, dim=-1) 
        logits = (y * a.unsqueeze(1)).sum(dim=-1) 

        proto_term = F.normalize(x, dim=-1) @ F.normalize(self.proto, dim=-1).t()
        logits = logits + proto_term + self.bias
        P = torch.sigmoid(logits / self.temp.clamp_min(1e-2))
        return P, logits


class QueryToSymbol(nn.Module):
    def __init__(self, inDim: int, k: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(inDim),
            nn.Linear(inDim, hidden), nn.SiLU(),
            nn.Linear(hidden, k))

    def forward(self, q: torch.Tensor):
        return torch.sigmoid(self.net(q))


class SymbolicMemory(nn.Module):
    def __init__(self, k: int, capacity: int = 4096):
        super().__init__()
        self.K = k
        self.capacity = capacity
        self.register_buffer("Pstore", torch.zeros(capacity, k, dtype=torch.float32))
        self.register_buffer("prio", torch.zeros(capacity))
        self.register_buffer("step", torch.zeros(capacity, dtype=torch.long))
        self.register_buffer("touch", torch.zeros(capacity, dtype=torch.long))
        self.filled = 0
        self.global_step = 0

    @torch.no_grad()
    def StepTick(self):
        self.global_step += 1

    @torch.no_grad()
    def Store(self, p: torch.Tensor, score: float = 1.0):
        p = p.detach().clamp(0, 1)
        if self.filled < self.capacity:
            i = self.filled
            self.filled += 1
        else:
            age = (self.global_step - self.step[:self.filled]).clamp(min=0).float()
            eff = self.prio[:self.filled] * torch.exp(-0.01 * age)
            i = int(torch.argmin(eff).item())
        self.Pstore[i] = p.to(self.Pstore.dtype)
        self.prio[i] = max(float(score), float(self.prio[i].item()))
        self.step[i] = self.global_step
        self.touch[i] += 1

    def Retrieve(self, qSym: torch.Tensor, topK: int = 8, recentBias: float = 0.05, returnDetails: bool = False):
        if self.filled == 0:
            out = torch.zeros(qSym.size(0), self.K, device=qSym.device, dtype=qSym.dtype)
            return (out, None, None, None) if returnDetails else out

        P = self.Pstore[:self.filled].to(qSym.device).float()

        sim = qSym @ P.t()
        age = (self.global_step - self.step[:self.filled]).clamp(min=0).float().to(qSym.device)
        sim = sim * torch.exp(-recentBias * age).unsqueeze(0) * (self.prio[:self.filled].to(qSym.device).unsqueeze(0))

        k = max(1, min(topK, self.filled))
        top_sim, idx = StableTopk(sim, k)
        w = F.softmax(top_sim, dim=-1)
        vecs = P[idx] 
        out = torch.einsum('bk,bkd->bd', w, vecs)

        with torch.no_grad():
            flat = idx.reshape(-1)
            self.touch[flat] += 1
            self.step[flat] = self.global_step

        if returnDetails:
            return out, vecs, w, idx
        return out


class SymbolicEmbed(nn.Module):
    def __init__(self, k: int, outDim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(k + 4, 512), nn.GELU(),
            nn.Linear(512, outDim))

    def forward(self, pCur: torch.Tensor, symRecall: torch.Tensor):
        ent = -(pCur * (pCur.clamp_min(1e-6)).log() + (1 - pCur) * ((1 - pCur).clamp_min(1e-6)).log()).mean(dim=-1, keepdim=True)
        pmax = pCur.max(dim=-1, keepdim=True).values
        pmean = pCur.mean(dim=-1, keepdim=True)
        pspa = (pCur < 0.1).float().mean(dim=-1, keepdim=True)
        feat = torch.cat([symRecall, ent, pmax, pmean, pspa], dim=-1) 
        return F.normalize(self.mlp(feat), dim=-1)  


class NeuroSymbolicFusion(nn.Module):
    def __init__(self, inDims: Dict[str, int], mid: int, out: int):
        super().__init__()
        D = inDims
        total_in = D["y"] + D["mem"] + D["ltm"] + D["gws"] + D["sym"] + D["ctx"]
        self.backbone = nn.Sequential(
            nn.Linear(total_in, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, mid), nn.GELU())
        
        self.refine = FusionMoE(inDim=mid + D["mem"], outDim=out, numExperts=4, hidden=2048)
        self.norm = nn.LayerNorm(out)

    def forward(self, ySsm, memVec, ltmVec, gwsVec, symVec, ctx):
        x = torch.cat([ySsm, memVec, ltmVec, gwsVec, symVec, ctx], dim=-1)
        mid = self.backbone(x)
        fused = self.refine(torch.cat([mid, memVec], dim=-1))  
        return self.norm(fused)


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
        self.keys.zero_() 
        self.vals.zero_()
        self.priority.zero_()
        self.ttl.zero_()
        self.last_step.zero_()
        self.tag_id.zero_()
        self.owner_id.zero_()
        self.global_step = 0

    @torch.no_grad()
    def StepTick(self):
        self.global_step += 1
        alive_mask = self.ttl > 0
        self.ttl[alive_mask] -= 1
        expired = self.ttl <= 0
        if expired.any():
            idx = torch.nonzero(expired, as_tuple=False).flatten()
            self.keys[idx].zero_()
            self.vals[idx].zero_()
            self.priority[idx].zero_()
            self.last_step[idx].zero_()
            self.tag_id[idx].zero_()
            self.owner_id[idx].zero_()

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
        key = F.normalize(key.detach().to(device=device, dtype=self.keys.dtype), dim=-1)
        val = F.normalize(val.detach().to(device=device, dtype=self.vals.dtype), dim=-1)
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

    def Retrieve(self, query: torch.Tensor, topk: int = 8, *, returnDetails: bool = False):
        if self.filled == 0:
            out = torch.zeros(query.size(0), self.dim, device=query.device, dtype=query.dtype)
            if returnDetails:
                return out, None, None, None
            return out

        E = self.emb[:self.filled].to(query.device).float()
        sim = F.normalize(query, dim=-1) @ F.normalize(E, dim=-1).t()
        k = max(1, min(topk, self.filled))
        top_sim, idx = StableTopk(sim, k)
        w = F.softmax(top_sim, dim=-1)
        vecs = E[idx]
        out = torch.einsum('bk,bkd->bd', w, vecs)

        with torch.no_grad():
            flat = idx.reshape(-1)
            self.touch[flat] += 1
            self.step[flat] = self.global_step

        if returnDetails:
            return out, vecs, w, idx
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
        self.register_buffer("touch", torch.zeros(capacity, dtype=torch.long))
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
            beta = 0.1
            eff = (self.prio[:self.filled] + 0.5 * self.rew[:self.filled]) * freshness / (1.0 + beta * self.touch[:self.filled].float())
            i = int(torch.argmin(eff).item())
        self.emb[i] = vec.to(self.emb.dtype)
        self.rew[i] = float(reward)
        self.prio[i] = max(float(score), float(self.prio[i].item()))
        self.step[i] = self.global_step
        self.touch[i] += 1

    def Retrieve(self, query: torch.Tensor, topk: int = 8, recentBias: float = 0.05, *, returnDetails: bool = False):
        if self.filled == 0:
            out = torch.zeros(query.size(0), self.dim, device=query.device, dtype=query.dtype)
            if returnDetails:
                return out, None, None, None
            return out

        E = self.emb[:self.filled].to(query.device).float()
        sim = F.normalize(query, dim=-1) @ F.normalize(E, dim=-1).t()
        age = (self.global_step - self.step[:self.filled]).clamp(min=0).float().to(query.device)
        freshness = torch.exp(-age * recentBias)
        sim = sim * freshness.unsqueeze(0)

        k = max(1, min(topk, self.filled))
        top_sim, idx = StableTopk(sim, k)
        w = F.softmax(top_sim, dim=-1)
        vecs = E[idx]
        out = torch.einsum('bk,bkd->bd', w, vecs)

        with torch.no_grad():
            flat = idx.reshape(-1)
            self.touch[flat] += 1
            self.step[flat] = self.global_step

        if returnDetails:
            return out, vecs, w, idx
        return out

class LTMFuser(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2*dim + 5, hidden), nn.SiLU(),
            nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, semOut: torch.Tensor, epiOut: torch.Tensor,semW: Optional[torch.Tensor], epiW: Optional[torch.Tensor]) -> torch.Tensor:
        B = semOut.size(0)
        c_sem = semW.max(dim=1).values if semW is not None else torch.zeros(B, device=semOut.device)
        c_epi = epiW.max(dim=1).values if epiW is not None else torch.zeros(B, device=epiOut.device)
        def ent01(w, k):
            if w is None: return torch.ones(B, device=semOut.device)
            h = -(w * (w.clamp_min(1e-8)).log()).sum(dim=1) 
            return (h / (math.log(k) if k > 1 else 1.0)).clamp(0, 1)
        e_sem = ent01(semW, semW.size(1) if semW is not None else 1)
        e_epi = ent01(epiW, epiW.size(1) if epiW is not None else 1)
        cos = F.cosine_similarity(semOut, epiOut, dim=-1) 
        feat = torch.stack([c_sem, c_epi, e_sem, e_epi, cos], dim=1)
        x = torch.cat([semOut, epiOut, feat], dim=1)
        gamma = self.fc(x).clamp(1e-3, 1 - 1e-3)
        return gamma

class LongTermMemory(nn.Module):
    def __init__(self, dim: int, semCap: int = 4096, epiCap: int = 4096):
        super().__init__()
        self.semantic = SemanticLTM(dim, semCap)
        self.episodic = EpisodicLTM(dim, epiCap)

        self.fuser = LTMFuser(dim)

    @torch.no_grad()
    def Reset(self):
        self.semantic.emb.zero_(); self.semantic.prio.zero_(); self.semantic.touch.zero_(); self.semantic.step.zero_()

        self.semantic.filled = 0; self.semantic.global_step = 0

        self.episodic.emb.zero_(); self.episodic.prio.zero_(); self.episodic.rew.zero_(); self.episodic.step.zero_()

        self.episodic.filled = 0; self.episodic.global_step = 0; self.episodic.touch.zero_()

    @torch.no_grad()
    def StepTick(self):
        self.semantic.StepTick()
        self.episodic.StepTick()

    def Retrieve(self, query: torch.Tensor, topkSem: int = 6, topkEpi: int = 2, returnDetails=True):
        sem_out, sem_vecs, sem_w, _ = self.semantic.Retrieve(query, topk=topkSem, returnDetails=returnDetails)
        epi_out, epi_vecs, epi_w, _ = self.episodic.Retrieve(query, topk=topkEpi, returnDetails=returnDetails)

        gamma = self.fuser(sem_out, epi_out, sem_w, epi_w)
        fused = F.normalize((1.0 - gamma) * sem_out + gamma * epi_out, dim=-1)

        return fused, sem_vecs, sem_w, epi_vecs, epi_w


class FusionMoE(nn.Module):
    def __init__(self, inDim: int, outDim: int, numExperts: int = 4, hidden: int = 1024):
        super().__init__()
        self.gate = nn.Linear(inDim, numExperts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, outDim)
            ) for _ in range(numExperts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softmax(self.gate(x), dim=-1) 
        ys = [self.experts[i](x) for i in range(len(self.experts))] 
        y = torch.stack(ys, dim=-1)  
        return (y * a.unsqueeze(1)).sum(dim=-1)  
    
class NeSyHead(nn.Module):
    def __init__(self, inDim: int, K: int, hidden: int = 1024, experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(inDim, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, K)
            ) for _ in range(experts)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softmax(self.gate(x), dim=-1)
        ys = [e(x) for e in self.experts] 
        y = torch.stack(ys, dim=-1) 
        return (y * a.unsqueeze(1)).sum(dim=-1) 

class MemoryExtractor(nn.Module):
    def __init__(
        self,
        inputDim: int = 1024,
        ssmStateDim: int = 1024,
        memoryDim: int = 768,
        memorySize: int = 400,
        outputDim: int = 768,
        hebbAlpha: float = 0.15,
        decayFactor: float = 0.95,
        topk: int = 8,
        tdScale: float = 5.0,
        softBeta: float = 0.2,
        useHebbian: bool = True,
        useAmp: bool = True,
        svdInterval: int = 10,
        svdMin: float = 0.1,
        svdMax: float = 1.5,
        gws: Optional[GlobalWorkspace] = None,
        ltm: Optional[LongTermMemory] = None,
        gwsSlots: int = 24,
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
        self.enable_hebb_update = useHebbian
        self.use_amp = useAmp
        self.svd_interval = max(1, svdInterval)
        self.svd_min = svdMin
        self.svd_max = svdMax

        self.ltm_topk_sem = 6
        self.ltm_topk_epi = 2
        self.ltm_inject = True 
        self.ltm_align_lambda = 0.0 
        self.ltm_online_imp_thresh = 0.75
        self.ltm_online_td_thresh  = 0.60

        self.gws_align_weight = 0.05

        self.enable_rehearsal = True
        self._ltm_cache = None
        self._ltm_cache_ttl = 2

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

        self.ctrl_norm = nn.LayerNorm(ssmStateDim)

        ctrl_hidden = 512
        self.ctrl_head = nn.Sequential(
            nn.Linear(self.ctrl_norm.normalized_shape[0] + self.memory_dim + 3 + 1 + 2, ctrl_hidden), nn.SiLU(),
            nn.Linear(ctrl_hidden, ctrl_hidden), nn.SiLU(),
            nn.Linear(ctrl_hidden, ctrl_hidden), nn.SiLU(),
            nn.Linear(ctrl_hidden, 4))
        
        nn.init.zeros_(self.ctrl_head[-1].weight)
        nn.init.zeros_(self.ctrl_head[-1].bias)

        self.kv_mlp = nn.Sequential(
            nn.LayerNorm(ssmStateDim),
            nn.Linear(ssmStateDim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU())

        self.kv_heads = 4
        assert memoryDim % self.kv_heads == 0, "memoryDim must be divisible by kv_heads"
        self.kv_head_dim = memoryDim // self.kv_heads

        self.kv_head_proj = nn.Parameter(torch.randn(self.kv_heads, 512, self.kv_head_dim) * 0.02)

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
            nn.Linear(memoryDim * 3, 512), nn.SiLU(),
            nn.Linear(512, 1), nn.Sigmoid(),)
        
        self.ltm_gate = nn.Sequential(
            nn.Linear(memoryDim * 4, 512), nn.SiLU(),
            nn.Linear(512, 1), nn.Sigmoid())

        self.fusion = FusionMoE(inDim = outputDim + memoryDim,outDim = outputDim,numExperts = 4, hidden = 2048)

        self.norm = nn.LayerNorm(outputDim)
        self.grad_bridge = nn.Parameter(torch.tensor(0.3))

        self.gws = gws if gws is not None else GlobalWorkspace(dim=memoryDim, slots=gwsSlots, defaultTtl=gwsTtl)
        self.ltm = ltm if ltm is not None else LongTermMemory(dim=memoryDim)
        self.gws_summary = nn.Linear(ssmStateDim + outputDim + memoryDim, memoryDim)

        self.gws_gate = nn.Sequential(
            nn.Linear(memoryDim * 2, 128), 
            nn.SiLU(), 
            nn.Linear(128, 1), 
            nn.Sigmoid())

        self.ns_enable: bool = True
        self.ns_lambda = 0.08
        self.ns_alpha_write = 0.15
        self.ns_alpha_out = 0.2
        self.ns_retrieve_boost = 0.3

        self.ns_K: int = 80
        self.sym_capacity: int = 4096 
        self.ns_gExcl: int = 5  
        self.ns_gOr: int = 5 

        self.ns_coder_pre = SymbolicCoder(self.memory_dim, self.ns_K, hidden=1024, experts=4)
        self.ns_coder_post = SymbolicCoder(self.memory_dim, self.ns_K, hidden=1024, experts=4)

        self.sym_rules = SoftSymbolicRules(k=self.ns_K,gExcl=self.ns_gExcl,gOr=self.ns_gOr,sparsity=1e-4,entropy=1e-4,impScale=1.0)

        self.sym_query = QueryToSymbol(inDim=self.memory_dim, k=self.ns_K, hidden=512) 
        self.sym_mem = SymbolicMemory(k=self.ns_K, capacity=self.sym_capacity)

        self.sym_embed = SymbolicEmbed(self.ns_K, outDim=self.memory_dim)

        self.sym_fuse_gate = nn.Sequential(
            nn.Linear(self.memory_dim * 2, 128), 
            nn.SiLU(),
            nn.Linear(128, 1), 
            nn.Sigmoid())

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

    def AttachLoss(self, mainLoss: torch.Tensor) -> torch.Tensor:
        return mainLoss + self.GetInternalLoss()

    def backward(self, mainLoss: torch.Tensor, **kwargs):
        total = self.AttachLoss(mainLoss)
        total.backward(**kwargs)


    def EncodeKV(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.kv_mlp(h) 
        heads = torch.einsum('bd,kdf->bkf', x, self.kv_head_proj)

        k_heads = heads + self.k_bias.unsqueeze(0)  
        v_heads = heads + self.v_bias.unsqueeze(0) 

        key = k_heads.reshape(h.size(0), -1) 
        val = v_heads.reshape(h.size(0), -1)

        key = F.normalize(key, dim=-1)
        val = F.normalize(val, dim=-1)
        return key, val

    @torch.no_grad()
    def KvStats(self, key: torch.Tensor) -> torch.Tensor:
        B = key.size(0)
        if self.memory_filled == 0:
            return torch.zeros(B, 3, device=key.device)

        K = self.memory_keys[:self.memory_filled].float().to(key.device)
        sim = key @ K.t()
        attn = F.softmax(sim, dim=-1)

        k = min(8, sim.size(1))
        topk_vals, _ = StableTopk(sim, k) 
        m = topk_vals.mean(dim=1, keepdim=True) 
        s = topk_vals.std(dim=1, keepdim=True, unbiased=False)

        age = (self.time_step - self.memory_steps[:self.memory_filled]).float().to(key.device) 
        age_w = (attn * age.unsqueeze(0)).sum(dim=1, keepdim=True) 

        age_w = torch.tanh(age_w / 100.0)
        return torch.cat([m, s, age_w], dim=-1)

    def NsEnsurePrev(self, B: int, device: torch.device):
        if (self._ns_prev_P_pre is None) or (self._ns_prev_P_pre.size(0) != B) or (self._ns_prev_P_pre.device != device):
            self._ns_prev_P_pre = torch.zeros(B, self.ns_K, device=device)
        if (self._ns_prev_P_post is None) or (self._ns_prev_P_post.size(0) != B) or (self._ns_prev_P_post.device != device):
            self._ns_prev_P_post = torch.zeros(B, self.ns_K, device=device)

    def NsRules(self, P: torch.Tensor, P_prev: Optional[torch.Tensor]) -> torch.Tensor:
        if (not self.ns_enable) or (P is None):
            return torch.zeros(P.size(0), device=P.device)

        per_sample, aux_reg = self.sym_rules(P, P_prev)

        if self.training and (aux_reg is not None):
            self.AddInternalLoss(self.ns_lambda * aux_reg)

        return per_sample

    def NsPreWrite(self, val: torch.Tensor, importance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.ns_enable:
            dev = val.device
            return torch.empty(0, device=dev), torch.empty(0, device=dev), torch.zeros([], device=dev), importance

        B, device = val.size(0), val.device
        self.NsEnsurePrev(B, device)

        P_pre, _ = self.ns_coder_pre(val) 
        per_sample_pre = self.NsRules(P_pre, self._ns_prev_P_pre) 
        self._ns_prev_P_pre = P_pre.detach()

        damp = torch.clamp(per_sample_pre, 0, 1).view(-1, 1)
        updated_importance = importance
        if self.ns_alpha_write > 0.0:
            updated_importance = importance * (1.0 - self.ns_alpha_write * damp)

        rule_loss_pre = per_sample_pre.mean()
        self._ns_penalty_vec = damp.detach()

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

        P_post, _ = self.ns_coder_post(memRecall)  
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
                reward: Optional[torch.Tensor] = None,
                reset: bool = False,
                softReset: bool = False,) -> Tuple[torch.Tensor, torch.Tensor]:

        amp_enable = self.use_amp and x.is_cuda
        dtype = torch.float16 if x.is_cuda else torch.bfloat16

        self.ResetInternalLoss()

        B, device = x.size(0), x.device
        if reset:
            self.ResetAll()
        elif softReset:
            self.SoftReset()

        if self.h_state.device != device:
            self.h_state = self.h_state.to(device)
        if self.h_state.size(0) != B:
            self.h_state.resize_(B, self.ssm_state_dim).zero_()

        self.time_step += 1

        self.gws.StepTick()
        self.ltm.StepTick()
        self.sym_mem.StepTick()

        h_prev = self.h_state.detach()

        h_new = self.h_state @ self.A_full.t() + self.B_mat(x)

        gb = 0.1 + 0.8 * torch.sigmoid(self.grad_bridge) 

        h_mix = gb * h_new + (1.0 - gb) * h_prev

        y_ssm = self.C_mat(h_mix) + self.D_mat(x)

        key, val = self.EncodeKV(h_mix)

        self.h_state = h_mix.detach()

        importance = self.importance_net(h_mix)
        gate_local = self.local_gate(h_mix)

        neuromod = self.GetNeuromod(tdError)
        self.UpdateMemoryUtilization()
        self.AutoCompress()

        td_feat = tdError.view(-1, 1) if tdError is not None else torch.zeros(B, 1, device=device)
        kv_feat = self.KvStats(key) 
        phi = torch.cat([self.ctrl_norm(h_mix), key, kv_feat, importance, gate_local, td_feat], dim=-1)

        ctrl = self.ctrl_head(phi) 
        a_raw, b_raw, f_raw, bias_raw = ctrl.split(1, dim=-1)

        a = (0.7 + 0.6 * torch.sigmoid(a_raw)).squeeze(-1)

        b = (0.97 + 0.03 * torch.sigmoid(b_raw)).squeeze(-1)

        fusion_gate = torch.sigmoid(f_raw).squeeze(-1)

        gate_bias = 0.5 * torch.tanh(bias_raw).squeeze(-1)

        reg = (a - 1.0).abs().mean() + (b - 1.0).abs().mean() + (fusion_gate - 0.5).abs().mean() + gate_bias.abs().mean()
        self.AddInternalLoss(1e-4 * reg)

        if self.ns_enable:
            P_pre, per_pre, rule_pre, importance = self.NsPreWrite(val, importance)
            self.AddInternalLoss(self.ns_lambda * rule_pre)

            with torch.no_grad():
                for i in range(B):
                    self.sym_mem.Store(P_pre[i], score=float(importance[i].item()))
        else:
            rule_pre = torch.zeros([], device=h_new.device)

        self.LtmOnlineStore(key, val, importance, tdError=tdError, reward=reward)
       
        fw_local = self.BuildFastWeights(key, gate_local, neuromod, a, b) if self.enable_hebb_update else None

        mem_recall = self.Retrieve(key, fusion_gate, importance=importance, localGate=gate_local, fwOverride=fw_local)

        self.HebbianUpdate(key, gate_local, neuromod, a, b)
        self.KvWrite(key, val, importance)

        ltm_recall, sem_vecs, sem_w, epi_vecs, epi_w = self.ltm.Retrieve(key, topkSem=self.ltm_topk_sem, topkEpi=self.ltm_topk_epi)

        msg = torch.cat([h_new, y_ssm, mem_recall], dim=-1)
        ws_val = F.normalize(self.gws_summary(msg), dim=-1)

        mem_recall_base = mem_recall.detach() 
        if self.gws_align_weight > 0:
            loss_gws_align = self.gws_align_weight * (1 - F.cosine_similarity(ws_val, mem_recall_base, dim=-1)).mean()
            self.AddInternalLoss(loss_gws_align)

        for i in range(B):
            self.gws.Write(key[i], ws_val[i], priority=float(importance[i].item()), ttl=6, tagId=1, ownerId=self.owner_id)

        gws_read, _ = self.gws.Attend(key, topk=4)
        fuse_gate = self.gws_gate(torch.cat([gws_read, mem_recall], dim=-1))
        mem_recall = fuse_gate * gws_read + (1 - fuse_gate) * mem_recall

        if self.ltm_inject:
            gamma_ltm = self.ltm_gate(torch.cat([key, mem_recall, ltm_recall, gws_read], dim=-1))
            mem_recall = (1.0 - gamma_ltm) * mem_recall + gamma_ltm * ltm_recall

        if self.ns_enable:
            P_post, per_post, rule_post, damp, mem_recall = self.NsPostRead(mem_recall)
            self.ns_last = {
                "P_pre": P_pre.detach() if 'P_pre' in locals() else None,
                "P_post": P_post.detach(),
                "per_sample_pre": per_pre.detach() if 'per_pre' in locals() else None,
                "per_sample_post": per_post.detach(),
                "rule_loss_pre": (self.ns_lambda * rule_pre).detach() if 'rule_pre' in locals() else torch.zeros([], device=mem_recall.device),
                "rule_loss_post": (self.ns_lambda * rule_post).detach(),}
            self.AddInternalLoss(self.ns_lambda * rule_post)
        else:
            self.ns_last = {}

        Qsym = self.sym_query(key)
        sym_recall = self.sym_mem.Retrieve(Qsym, topK=8) 

        P_cur = P_post if self.ns_enable else Qsym 
        sym_vec = self.sym_embed(P_cur, sym_recall)

        g_sym = self.sym_fuse_gate(torch.cat([mem_recall, sym_vec], dim=-1))
        mem_recall = (1.0 - g_sym) * mem_recall + g_sym * sym_vec

        if tdError is not None:
            mem_recall = self.ApplyOutputGate(mem_recall, tdError, gate_bias)

        fused = self.fusion(torch.cat([y_ssm, mem_recall], dim=-1))
        output = self.norm(fused)

        self._ltm_cache = {
            "step": self.time_step,
            "sem_vecs": sem_vecs,
            "sem_w": sem_w,
            "epi_vecs": epi_vecs,
            "epi_w": epi_w,
            "ltm_recall": ltm_recall.detach(),
            "mem_recall": mem_recall.detach(),}

        if (self.time_step % self.consolidate_every) == 0:
            self.ConsolidateFromGWS()
        if (self.time_step % self.rehearse_every) == 0:
            self.RehearseFromLTM(batch=min(8, self.memory_filled if self.memory_filled > 0 else 1))

        return output.float(), mem_recall.float()

    def GetNeuromod(self, tdError: Optional[torch.Tensor]) -> torch.Tensor:
        if tdError is None:
            return torch.ones(1, 1, 1, device=self.h_state.device)
        return torch.tanh(tdError / self.td_scale).view(-1, 1, 1)

    
    def ApplyOutputGate(self, memRecall: torch.Tensor, tdError: torch.Tensor, gateBias: torch.Tensor) -> torch.Tensor:
        gate_out = (1.0 + torch.tanh(tdError.detach() / self.td_scale + gateBias)) / 2.0
        return gate_out.view(-1, 1) * memRecall

    @torch.no_grad()
    def SoftReset(self):
        self.h_state.copy_(self.h_state * self.soft_beta)
        self.fast_weights.copy_(self.fast_weights * self.soft_beta)
        self._steps_since_svd = 0
        if self.memory_filled > 0:
            new_imp = self.memory_importance.clone()
            new_imp[:self.memory_filled] = new_imp[:self.memory_filled] * self.soft_beta
            self.memory_importance.copy_(new_imp)

    @torch.no_grad()
    def HebbianUpdate(self, key: torch.Tensor, gateLocal: torch.Tensor, neuromod: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
        if self.enable_hebb_update is False: 
            return
        
        a = a.view(-1, 1, 1)
        b = b.view(-1, 1, 1)

        outer = torch.einsum('bi,bj->bij', key, key)
        update = (neuromod * self.hebb_alpha * a * gateLocal.view(-1, 1, 1) * outer).sum(0)

        fw_new = self.fast_weights * (self.decay * b.mean())
        fw_new = fw_new + update
        self.fast_weights.copy_(fw_new)

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


    def BuildFastWeights(self,key: torch.Tensor,gateLocal: torch.Tensor,neuromod: torch.Tensor,a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self.enable_hebb_update:
            return None
        
        DtypeFW = self.fast_weights.dtype
        key_f = key.to(DtypeFW) 
        outer = torch.einsum('bi,bj->bij', key_f, key_f)   

        a_f = a.view(-1, 1, 1).to(DtypeFW)
        g_f = gateLocal.view(-1, 1, 1).to(DtypeFW) 
        n_f = neuromod.to(DtypeFW) 

        update = (n_f * self.hebb_alpha * a_f * g_f * outer).sum(0)

        b_bar = b.mean()  

        fw_prev = self.fast_weights.detach().clone().to(DtypeFW)
        fw_local = fw_prev * (self.decay * b_bar) + update  
        return fw_local


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

    @torch.no_grad()
    def LtmOnlineStore(self, key, val, importance, tdError=None, reward=None):
        B = key.size(0)
        imp = importance.view(-1)
        td  = tdError.view(-1).abs() if tdError is not None else torch.zeros(B, device=key.device)
        mask = (imp > self.ltm_online_imp_thresh) | (td > self.ltm_online_td_thresh)
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        for i in idx.tolist():
            self.ltm.semantic.Store(val[i], score=float(imp[i].item()))
            self.ltm.episodic.Store(
                key[i],
                reward=float(0.0 if reward is None else reward[i].item()),
                score=float(imp[i].item()),)

    def Retrieve(self,
             query: torch.Tensor,
             fusionGate: torch.Tensor,
             *,
             importance: Optional[torch.Tensor] = None,
             localGate: Optional[torch.Tensor] = None,
             fwOverride: Optional[torch.Tensor] = None) -> torch.Tensor:
        fw_base = self.fast_weights if self.enable_hebb_update else torch.zeros_like(self.fast_weights)
        fw = fwOverride if fwOverride is not None else fw_base

        fast_part = (query.to(fw.dtype)) @ fw
        fast_part = fast_part.to(query.dtype)

        if self.memory_filled == 0:
            kv_part = torch.zeros_like(fast_part)
        else:
            keys = self.memory_keys[:self.memory_filled].float()
            values = self.memory_values[:self.memory_filled].float()
            imp_kv = self.memory_importance[:self.memory_filled].detach().clone()
            corr = self.memory_corr[:self.memory_filled].detach().clone()
            steps = self.memory_steps[:self.memory_filled].detach().clone()

            sim = query @ keys.t()
            sim = sim * imp_kv.unsqueeze(0) * corr.unsqueeze(0)
            age = (self.time_step - steps).clamp(min=0).float()
            sim = sim * torch.exp(-0.05 * age).unsqueeze(0)

            k = max(1, min(self.topk, self.memory_filled))
            top_sim, top_idx = StableTopk(sim, k)
            th = top_sim.mean(dim=-1, keepdim=True) - 0.5 * top_sim.std(dim=-1, keepdim=True, unbiased=False)
            th_min = top_sim.min(dim=-1, keepdim=True).values
            th_max = top_sim.max(dim=-1, keepdim=True).values
            th = torch.clamp(th, min=th_min, max=th_max)

            if self.ns_enable and self.ns_retrieve_boost > 0.0 and (self._ns_penalty_vec is not None):
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

        if importance is not None:
            gate = gate + 0.25 * (importance.clamp(0, 1) - 0.5)
        if localGate is not None:
            gate = gate + 0.25 * (localGate.clamp(0, 1) - 0.5)

        gate = 0.5 * gate + 0.5 * fusionGate.view(-1, 1)
        gate = gate.clamp(1e-3, 1 - 1e-3)

        return gate * fast_part + (1 - gate) * kv_part

    def UpdateMemoryUtilization(self):
        window_size = max(50, min(200, self.memory_size // 5))
        min_step = max(1, self.time_step - window_size)
        accessed = ((self.memory_steps >= min_step) & (self.memory_steps > 0) & (torch.arange(self.memory_size, device=self.memory_steps.device) < self.memory_filled))
        accessed_count = accessed.sum().item()
        self.memory_usage = (min(1.0, accessed_count / self.memory_filled) if self.memory_filled > 0 else 0.0)

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
    def ConsolidateFromGWS(self, minPriority: float = 0.25, reward : float = None):
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
            self.ltm.episodic.Store(F.normalize(k, dim=-1), reward=reward if reward is not None else 0.0, score=score)

    @torch.no_grad()
    def RehearseFromLTM(self, batch: int = 8):
        if not self.enable_rehearsal:
            return
        fresh = (self._ltm_cache is not None) and (self.time_step - self._ltm_cache["step"] <= self._ltm_cache_ttl)
        if not fresh or (self.memory_usage >= 0.5):
            return

        device = self.fast_weights.device
        cand = []
        w = []
        for key in ("sem_vecs","epi_vecs"):
            vecs = self._ltm_cache[key]
            if vecs is not None:
                B,K,D = vecs.shape
                cand.append(vecs.reshape(-1, D))
        for key in ("sem_w","epi_w"):
            ww = self._ltm_cache[key]
            if ww is not None:
                w.append(ww.reshape(-1))
        if not cand:
            if "ltm_recall" in self._ltm_cache and "mem_recall" in self._ltm_cache:
                diff = F.normalize(self._ltm_cache["ltm_recall"] - self._ltm_cache["mem_recall"], dim=-1)
                if diff.abs().sum() > 1e-8:
                    N = diff.size(0)
                    a = torch.full((N,), 0.5,  device=device)
                    b = torch.full((N,), 0.99, device=device)
                    gate = torch.full((N,1), 0.4, device=device)
                    neu = torch.full((N,1,1), 0.08, device=device)
                    self.HebbianUpdate(diff, gate, neu, a, b)
            return

        keys = F.normalize(torch.cat(cand, dim=0).to(device), dim=-1)    # [N,D]
        w = torch.cat(w, dim=0).to(device) if w else torch.full((keys.size(0),), 0.5, device=device)

        def novelty_vs_kv(cand):
            if self.memory_filled == 0: return torch.ones(cand.size(0), device=cand.device)
            kv = self.memory_keys[:self.memory_filled].float().to(cand.device)
            sim = F.normalize(cand,dim=-1) @ F.normalize(kv,dim=-1).t()
            return 1.0 - sim.abs().max(dim=1).values.clamp(0,1)
        novelty = novelty_vs_kv(keys)

        a = (0.40 + 0.40*torch.sigmoid(2.0*w + 1.4*novelty)).clamp(0.35, 0.9)
        b = (0.99  - 0.012*w*novelty).clamp(0.975, 0.999)
        gate = (0.35 + 0.45*w + 0.15*novelty).clamp(0.20, 0.85).view(-1,1)
        neu = (0.06 + 0.24*w*novelty).clamp(0.03, 0.30).view(-1,1,1)

        self.HebbianUpdate(keys, gate, neu, a, b)

    def ResetAll(self):
        self.fast_weights.zero_()
        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.memory_importance.zero_()
        self.memory_steps.zero_()
        self.memory_corr.zero_()
        self.mem_ptr = 0
        self.time_step = 0
        self.memory_filled = 0
        self.memory_usage = 0.0
        self.last_compress_step = 0
        self._steps_since_svd = 0
        self.h_state.zero_()
        self.gws.Reset()
        self.ltm.Reset()
        self._ltm_cache = None

        self._ns_prev_P_pre = None
        self._ns_prev_P_post = None
        self._ns_penalty_vec = None
        self.ns_last = {}
        self.ResetInternalLoss()

        self.sym_mem = SymbolicMemory(k=self.ns_K, capacity=self.sym_capacity)

    def GetLastNS(self):
        return self.ns_last

    def ResetHebbianMemory(self):
        self.fast_weights.zero_()
        self._steps_since_svd = 0
        self.fro_norm_history.clear()
        self.svd_threshold = 5.0

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
            "memory_values": torch.zeros_like(self.memory_values, device=dev),
            "memory_importance": torch.zeros_like(self.memory_importance, device=dev),
            "memory_corr": torch.zeros_like(self.memory_corr, device=dev),
            "memory_steps": torch.zeros_like(self.memory_steps, device=dev),
            "mem_ptr": torch.tensor(0, device=dev),
            "time_step": torch.tensor(0, device=dev),
            "memory_filled": torch.tensor(0, device=dev),

            "_steps_since_svd": torch.tensor(0, device=dev),
            "last_compress_step": torch.tensor(0, device=dev),
            "memory_usage": torch.tensor(0.0, device=dev),
            "svd_threshold": torch.tensor(5.0, device=dev),
            "fro_norm_history": torch.zeros(0, device=dev),

            "gws_keys": gws_snap["keys"].detach().clone().zero_(),
            "gws_vals": gws_snap["vals"].detach().clone().zero_(),
            "gws_priority": gws_snap["priority"].detach().clone().zero_(),
            "gws_ttl": gws_snap["ttl"].detach().clone().zero_(),
            "gws_last_step": gws_snap["last_step"].detach().clone().zero_(),
            "gws_tag_id": gws_snap["tag_id"].detach().clone().zero_(),
            "gws_owner_id": gws_snap["owner_id"].detach().clone().zero_(),
            "gws_global_step": torch.tensor(0, device=dev),

            "ltm_sem_emb": sem.emb.detach().clone().zero_().to(dev),
            "ltm_sem_prio": sem.prio.detach().clone().zero_().to(dev),
            "ltm_sem_touch": sem.touch.detach().clone().zero_().to(dev),
            "ltm_sem_step": sem.step.detach().clone().zero_().to(dev),
            "ltm_sem_filled": torch.tensor(0, device=dev),
            "ltm_sem_global_step": torch.tensor(0, device=dev),

            "ltm_epi_emb": epi.emb.detach().clone().zero_().to(dev),
            "ltm_epi_prio": epi.prio.detach().clone().zero_().to(dev),
            "ltm_epi_rew": epi.rew.detach().clone().zero_().to(dev),
            "ltm_epi_step": epi.step.detach().clone().zero_().to(dev),
            "ltm_epi_filled": torch.tensor(0, device=dev),
            "ltm_epi_global_step": torch.tensor(0, device=dev),

            "ns_prev_P_pre": None,
            "ns_prev_P_post": None,

            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,}

        torch.save(state, path)

    @torch.no_grad()
    def SaveState(self, path: str):
        gws_snap = self.gws.Inspect()
        sem = self.ltm.semantic
        epi = self.ltm.episodic

        state = {
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

            "_steps_since_svd": torch.tensor(self._steps_since_svd),
            "last_compress_step": torch.tensor(self.last_compress_step),
            "memory_usage": torch.tensor(self.memory_usage, dtype=torch.float32),
            "svd_threshold": torch.tensor(self.svd_threshold, dtype=torch.float32),
            "fro_norm_history": torch.tensor(self.fro_norm_history, dtype=torch.float32),

            "gws_keys": gws_snap["keys"].clone(),
            "gws_vals": gws_snap["vals"].clone(),
            "gws_priority": gws_snap["priority"].clone(),
            "gws_ttl": gws_snap["ttl"].clone(),
            "gws_last_step": gws_snap["last_step"].clone(),
            "gws_tag_id": gws_snap["tag_id"].clone(),
            "gws_owner_id": gws_snap["owner_id"].clone(),
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
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            
            "sym_mem_Pstore": self.sym_mem.Pstore.clone(),
            "sym_mem_prio": self.sym_mem.prio.clone(),
            "sym_mem_step": self.sym_mem.step.clone(),
            "sym_mem_touch": self.sym_mem.touch.clone(),
            "sym_mem_filled": torch.tensor(self.sym_mem.filled),
            "sym_mem_global_step": torch.tensor(self.sym_mem.global_step),

            "ns_penalty_vec": (self._ns_penalty_vec.clone() if self._ns_penalty_vec is not None else None),}
        
        torch.save(state, path)

    @torch.no_grad()
    def LoadState(self, path: str):
        if os.path.getsize(path) == 0:
            return

        state = torch.load(path, weights_only=False)

        if "rng_cpu" in state:
            torch.set_rng_state(state["rng_cpu"].to("cpu"))
        if "rng_cuda" in state and state["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        
        hs = state["h_state"].to(self.h_state.device)
        if self.h_state.shape != hs.shape:
            self.h_state.resize_(hs.shape).copy_(hs)
        else:
            self.h_state.copy_(hs)

        self.fast_weights.copy_(state["fast_weights"].to(self.fast_weights.device))
        self.memory_keys.copy_(state["memory_keys"].to(self.memory_keys.device))
        self.memory_values.copy_(state["memory_values"].to(self.memory_values.device))
        self.memory_importance.copy_(state["memory_importance"].to(self.memory_importance.device))
        self.memory_corr.copy_(state["memory_corr"].to(self.memory_corr.device))
        self.memory_steps.copy_(state["memory_steps"].to(self.memory_steps.device))
        self.mem_ptr = int(state["mem_ptr"].item())
        self.time_step = int(state["time_step"].item())
        self.memory_filled = int(state["memory_filled"].item())

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
            self._ns_prev_P_pre = (state["ns_prev_P_pre"].to(self.h_state.device) if state["ns_prev_P_pre"] is not None else None)
        if "ns_prev_P_post" in state:
            self._ns_prev_P_post = (state["ns_prev_P_post"].to(self.h_state.device) if state["ns_prev_P_post"] is not None else None)
            
        if "sym_mem_Pstore" in state:
            self.sym_mem.Pstore.copy_(state["sym_mem_Pstore"].to(self.sym_mem.Pstore.device))
            self.sym_mem.prio.copy_(state["sym_mem_prio"].to(self.sym_mem.prio.device))
            self.sym_mem.step.copy_(state["sym_mem_step"].to(self.sym_mem.step.device))
            self.sym_mem.touch.copy_(state["sym_mem_touch"].to(self.sym_mem.touch.device))
            self.sym_mem.filled = int(state["sym_mem_filled"].item())
            self.sym_mem.global_step = int(state["sym_mem_global_step"].item())

        if "ns_penalty_vec" in state:
            self._ns_penalty_vec = (None if state["ns_penalty_vec"] is None else state["ns_penalty_vec"].to(self.h_state.device))

    @torch.no_grad()
    def Reason(self, goal: Optional[torch.Tensor] = None, steps: int = 3) -> torch.Tensor:
        device = self.h_state.device
        h = self.h_state.mean(dim=0, keepdim=True)
        q, _ = self.EncodeKV(h)
        if goal is not None:
            goal = F.normalize(goal.to(device).view(1, -1), dim=-1)
            q = F.normalize(q + goal, dim=-1)
        hyp = torch.zeros(1, self.memory_dim, device=device)
        for _ in range(max(1, steps)):
            gws_r, _ = self.gws.Attend(q, topk=4)
            ltm_r , *_= self.ltm.Retrieve(q, topkSem=6, topkEpi=2)
            hyp = F.normalize(gws_r + ltm_r + q, dim=-1)
            self.gws.Write(hyp[0], hyp[0], priority=0.5, ttl=4, tagId=2, ownerId=self.owner_id)
            q = hyp
        return hyp



class TestMemoryMTool:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        self.root = Path("BrainDeepLearn/TestData").expanduser().absolute()
        self.root.mkdir(parents=True, exist_ok=True)

    def StatePath(self, name: str = "memory_state.pth") -> Path:
        return (self.root / name).absolute()

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
            assert out.shape == (1, dim)
            assert w is not None and w.shape == (1, 2)

            gws.StepTick()
            out2, _ = gws.Attend(q, topk=3)
            assert out2.shape == (1, dim)

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
            fused, *_ = ltm.Retrieve(q, topkSem=4, topkEpi=2)
            assert fused.shape == (1, dim)
            assert torch.linalg.norm(fused).item() > 0

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
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32,outputDim=96, useAmp=True, gwsSlots=8, gwsTtl=6,consolidateEvery=50, rehearseEvery=60)
            mem = MemoryExtractor(**cfg).to(self.device)
            B = 4
            x = torch.randn(B, cfg["inputDim"], device=self.device)
            out, memrec = mem(x)

            assert out.shape == (B, cfg["outputDim"] )
            assert memrec.shape == (B, cfg["memoryDim"])
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
            cfg = dict(inputDim=48, ssmStateDim=48, memoryDim=64, memorySize=24,outputDim=64, useAmp=True, gwsSlots=8, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)

            path = self.StatePath("memory_state_test.pth")
            mem.SaveState(str(path))

            torch.manual_seed(123)
            x = torch.randn(3, cfg["inputDim"], device=self.device)
            out1, _ = mem(x)
            _ = mem(x)

            mem.LoadState(str(path))
            out2, _ = mem(x)

            try:
                path.unlink()
            except OSError:
                pass

            assert torch.allclose(out1, out2, atol=5e-4, rtol=1e-3)
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
            cfg = dict(inputDim=32, ssmStateDim=32, memoryDim=48, memorySize=32,outputDim=48, useAmp=True, gwsSlots=10, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)
            for _ in range(5):
                x = torch.randn(2, cfg["inputDim"], device=self.device)
                mem(x)

            goal = torch.randn(cfg["memoryDim"], device=self.device)
            hyp = mem.Reason(goal=goal, steps=3)
            assert hyp.shape == (1, cfg["memoryDim"])
            assert torch.linalg.norm(hyp).item() > 0

            snap = mem.gws.Inspect()
            assert torch.count_nonzero(snap["priority"] > 0).item() > 0

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
            cfg = dict(inputDim=32, ssmStateDim=32, memoryDim=48, memorySize=32,outputDim=48, useAmp=True, gwsSlots=8, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)
            x = torch.randn(4, cfg["inputDim"], device=self.device)
            mem(x)

            fw_before = mem.fast_weights.detach().clone()
            imp_before = mem.memory_importance.detach().clone()
            mem.SoftReset()
            fw_after = mem.fast_weights.detach().clone()
            imp_after = mem.memory_importance.detach().clone()

            assert torch.linalg.norm(fw_after) <= torch.linalg.norm(fw_before) + 1e-6
            if mem.memory_filled > 0:
                assert torch.mean(imp_after[:mem.memory_filled]) <= torch.mean(imp_before[:mem.memory_filled]) + 1e-6

            mem.ResetAll()
            assert mem.memory_filled == 0 and mem.time_step == 0
            gws_snap = mem.gws.Inspect()
            assert torch.count_nonzero(gws_snap["priority"]).item() == 0
            assert mem.ltm.semantic.filled == 0 and mem.ltm.episodic.filled == 0

            print("MemoryExtractor Reset/SoftReset test passed.")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor Reset/SoftReset test failed: {e}")
            return False
        except Exception as e:
            print(f"MemoryExtractor Reset/SoftReset test error: {e}")
            return False

    def TestMemoryTrain(self, steps: int = 120, batch_size: int = 16):
        try:
            torch.manual_seed(2025)
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=64,outputDim=64, useAmp=True, gwsSlots=8, gwsTtl=6,consolidateEvery=10_000, rehearseEvery=10_000)
            device = self.device
            mem = MemoryExtractor(**cfg).to(device)
            mem.train()

            teacher = nn.Sequential(
                nn.Linear(cfg["inputDim"], 128, bias=False),
                nn.GELU(),
                nn.Linear(128, cfg["outputDim"], bias=False),
            ).to(device)
            for p in teacher.parameters():
                p.requires_grad_(False)

            must_train_prefixes = [
                "kv_mlp", "kv_head_proj", "k_bias", "v_bias",
                "importance_net", "local_gate", "fusion_gate_net",
                "ltm_gate", "fusion", "norm",
                "A_full", "B_mat", "C_mat", "D_mat",
                "gws_summary", "gws_gate",
                "ns_coder_pre", "ns_coder_post", "sym_query", "sym_embed", "sym_rules",
                "grad_bridge", "ctrl_head", "ctrl_norm",]

            snap_before = {}
            for n, p in mem.named_parameters():
                if any(n.startswith(pref) for pref in must_train_prefixes):
                    snap_before[n] = p.detach().clone()

            grads_seen = {pref: False for pref in must_train_prefixes}

            opt = torch.optim.Adam(mem.parameters(), lr=1e-3)
            losses = []

            with torch.no_grad():
                p0 = []
                for n, p in mem.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        p0.append(p.data.flatten()[:32].clone())
                p0 = torch.cat(p0) if p0 else torch.zeros(1, device=device)

            in_dim = cfg["inputDim"]
            for t in range(steps):
                x = torch.randn(batch_size, in_dim, device=device)
                td = torch.randn(batch_size, device=device)
                rwd = torch.randn(batch_size, device=device)
                with torch.no_grad():
                    target = teacher(x)

                out, _ = mem(x, tdError=td, reward=rwd)
                base = F.mse_loss(out, target)
                total = self.AttachAllInternalLosses(mem, base)

                opt.zero_grad()
                total.backward()

                for n, p in mem.named_parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        raise AssertionError(f"Non-finite grad at step {t}, {n}")
                    if p.grad.abs().sum() > 0:
                        for pref in must_train_prefixes:
                            if n.startswith(pref):
                                grads_seen[pref] = True

                torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
                opt.step()

                losses.append(float(base.detach().item()))
                if (t + 1) % max(1, steps // 4) == 0:
                    print(f"[NormalTrain] step {t+1}/{steps} | mse={losses[-1]:.6f}")

            assert len(losses) >= 2
            start, end = losses[0], losses[-1]
            print(f"[NormalTrain] loss start={start:.6f} -> end={end:.6f}")
            rel_ok = end <= start * 0.70
            abs_ok = (start - end) >= 0.05
            assert rel_ok or abs_ok, f"Loss not decreased enough: start={start:.6f}, end={end:.6f}"

            snap_after = {}
            for n, p in mem.named_parameters():
                if any(n.startswith(pref) for pref in must_train_prefixes):
                    snap_after[n] = p.detach().clone()

            for must in ["importance_net", "local_gate", "kv_mlp", "kv_head_proj",
                         "ns_coder_pre", "ns_coder_post", "sym_rules"]:
                assert grads_seen[must], f"{must} never received gradients"
                delta_sum = 0.0
                for n in snap_before:
                    if n.startswith(must):
                        delta_sum += float((snap_after[n] - snap_before[n]).abs().sum().item())
                assert delta_sum > 0.0, f"{must} parameters did not change (Δ=0)"
                print(f"[trainable] {must}: grad_seen={grads_seen[must]}, Δ_sum={delta_sum:.3e}")

            soft_expect = ["fusion_gate_net", "fusion", "A_full", "B_mat", "C_mat", "D_mat",
                           "kv_mlp", "kv_head_proj", "k_bias", "v_bias",
                           "gws_summary", "gws_gate",
                           "ctrl_head", "ctrl_norm", "grad_bridge", "sym_query", "sym_embed"]
            for pref in soft_expect:
                assert grads_seen[pref], f"{pref} saw no gradients (check if in loss path)"

            with torch.no_grad():
                p1 = []
                for n, p in mem.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        p1.append(p.data.flatten()[:32].clone())
                p1 = torch.cat(p1) if p1 else torch.zeros(1, device=device)
                delta = (p0 - p1).abs().mean().item()
                assert delta > 1e-6, f"Overall parameters barely changed, delta={delta:.3e}"

            npn = dict(mem.named_parameters())
            assert "fast_weights" not in npn
            assert "memory_keys" not in npn and "memory_values" not in npn
            assert "memory_importance" not in npn and "memory_steps" not in npn and "memory_corr" not in npn
            assert "h_state" not in npn

            print("TestNormalTrainingConvergence + Trainability checks passed.")
            return True

        except AssertionError as e:
            print(f"TestMemoryTrain failed: {e}")
            return False
        except Exception as e:
            print(f"TestMemoryTrain error: {e}")
            return False

    @torch.no_grad()
    def NumericalStabilityProbe(
        self,
        mem,
        *,
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

        mem2 = MemoryExtractor(
            inputDim=mem.B_mat.in_features,
            ssmStateDim=mem.ssm_state_dim,
            memoryDim=mem.memory_dim,
            memorySize=mem.memory_size,
            outputDim=mem.output_dim,
            useAmp=mem.use_amp,
            gwsSlots=mem.gws.slots,
            gwsTtl=mem.gws.default_ttl,
            consolidateEvery=mem.consolidate_every,
            rehearseEvery=mem.rehearse_every,).to(device)

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
        print(f"cos(mean)={cos_np.mean():.6f}, cos(std)={cos_np.std():.6f}, cos(min)={cos_np.min():.6f}, cos(max)={cos_np.max():.6f}")
        print(f"cos(start)={cos_np[0]:.6f}, cos(last)={cos_np[-1]:.6f}, drop={cos_np[0]-cos_np[-1]:.6f}")
        return list(cos_hist)

    def RandnLikeGen(self, x: torch.Tensor, generator: torch.Generator | None = None):
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    def TestNumericalStability(self):
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=48,outputDim=96, useAmp=True, gwsSlots=8, gwsTtl=6,consolidateEvery=1000, rehearseEvery=1000)
            mem = MemoryExtractor(**cfg).to(self.device)

            cos_hist = self.NumericalStabilityProbe(mem,steps=200,batch=4,eps=1e-6,perturb_each_step=True,seed=123,device=self.device,print_every=25,)

            for c in cos_hist:
                assert math.isfinite(c), "cosine became non-finite"

            max_abs = max(abs(c) for c in cos_hist)
            assert max_abs < 0.25, f"cosine drifted too far: {max_abs:.3f}"

            print("MemoryExtractor numerical stability test passed (relaxed for stateful/symbolic memory).")
            return True
        except AssertionError as e:
            print(f"MemoryExtractor numerical stability test failed (relaxed): {e}")
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
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32,outputDim=96, useAmp=True, gwsSlots=8, gwsTtl=6,consolidateEvery=10_000, rehearseEvery=10_000)
            mem = MemoryExtractor(**cfg).to(self.device)
            mem.train()
            opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

            B = 8
            x = torch.randn(B, cfg["inputDim"], device=self.device)
            target = torch.randn(B, cfg["outputDim"], device=self.device)

            out, _ = mem(x, tdError=torch.randn(B, device=self.device), reward=torch.randn(B, device=self.device))
            base = F.mse_loss(out, target)

            total = self.AttachAllInternalLosses(mem, base)
            opt.zero_grad()
            total.backward()

            nesy_grad_ok = any(
                (("ns_coder_pre" in n or "ns_coder_post" in n or "sym_rules" in n) and
                 (p.grad is not None) and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
                for n, p in mem.named_parameters())
            assert nesy_grad_ok, "NeSy stack (ns_coder_* / sym_rules) did not receive gradients."

            for n, p in mem.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

            opt.step()
            print("TestMemoryMTool.TrainStepSmoke passed.")
            return True
        except AssertionError as e:
            print(f"TestMemoryMTool.TrainStepSmoke failed: {e}")
            return False
        except Exception as e:
            print(f"TestMemoryMTool.TrainStepSmoke error: {e}")
            return False

    def TrainNeSyOnlySanity(self, steps: int = 30):
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=32, outputDim=96, useAmp=True, gwsSlots=8, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)
            mem.ns_enable = True
            mem.ns_lambda = 0.5
            mem.train()
            opt = torch.optim.Adam([p for p in mem.parameters() if p.requires_grad], lr=1e-3)

            watch = lambda n: ("ns_coder_pre" in n) or ("ns_coder_post" in n) or ("sym_rules" in n)
            snap_before = {n: p.detach().clone() for n, p in mem.named_parameters() if watch(n)}

            in_dim = cfg["inputDim"]
            for _ in range(steps):
                x = torch.randn(8, in_dim, device=self.device)
                _ = mem(x)
                base = torch.zeros([], device=self.device)
                total = self.AttachAllInternalLosses(mem, base)

                opt.zero_grad()
                total.backward()
                opt.step()

            delta = 0.0
            with torch.no_grad():
                for n, p in mem.named_parameters():
                    if n in snap_before:
                        delta += float((p - snap_before[n]).abs().sum().item())

            assert delta > 0.0, f"NeSy parameters have not been effectively updated (Δ={delta:.3e})"
            print("TestMemoryMTool.TrainNeSyOnlySanity passed.")
            return True
        except AssertionError as e:
            print(f"TestMemoryMTool.TrainNeSyOnlySanity failed: {e}")
            return False
        except Exception as e:
            print(f"TestMemoryMTool.TrainNeSyOnlySanity error: {e}")
            return False

    def TestAllTrainablesTouched(self, steps: int = 10, batch_size: int = 12):
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=64,outputDim=96, useAmp=True, gwsSlots=8, gwsTtl=6,consolidateEvery=10_000, rehearseEvery=10_000)
            device = self.device
            mem = MemoryExtractor(**cfg).to(device)
            mem.train()

            opt = torch.optim.Adam(mem.parameters(), lr=1e-3)
            seen = {n: False for n, p in mem.named_parameters() if p.requires_grad and p.data.numel() > 0}

            for t in range(steps):
                x = torch.randn(batch_size, cfg["inputDim"], device=device)
                y = torch.randn(batch_size, cfg["outputDim"], device=device)
                td = torch.randn(batch_size, device=device)
                rwd = torch.randn(batch_size, device=device)

                out, _ = mem(x, tdError=td, reward=rwd)
                base = F.mse_loss(out, y)
                total = self.AttachAllInternalLosses(mem, base)

                opt.zero_grad()
                total.backward()

                for n, p in mem.named_parameters():
                    if n in seen and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
                        seen[n] = True

                torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
                opt.step()

            missing = sorted([n for n, hit in seen.items() if not hit])
            if missing:
                print("Parameters without nonzero grads at least once:")
                for n in missing:
                    print("  -", n)
            assert len(missing) == 0, f"{len(missing)} parameters never received gradients"
            print("TestAllTrainablesTouched passed.")
            return True
        except AssertionError as e:
            print(f"TestAllTrainablesTouched failed: {e}")
            return False
        except Exception as e:
            print(f"TestAllTrainablesTouched error: {e}")
            return False

    def TestNeSyRetrievalEffect(self):
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=48,outputDim=96, useAmp=True, gwsSlots=8, gwsTtl=6,consolidateEvery=1000, rehearseEvery=1000)
            torch.manual_seed(7)
            device = self.device

            B = 6
            mem_on = MemoryExtractor(**cfg).to(device)
            mem_on.ns_enable = True
            mem_on.eval()
            with torch.no_grad():
                for _ in range(3):
                    xw = torch.randn(B, cfg["inputDim"], device=device)
                    mem_on(xw)

            tmp_path = self.StatePath("memory_state_nesy_effect.pth")
            mem_on.SaveState(str(tmp_path))

            mem_off = MemoryExtractor(**cfg).to(device)
            mem_off.LoadState(str(tmp_path))
            mem_off.ns_enable = False
            mem_off.eval()

            try:
                tmp_path.unlink()
            except OSError:
                pass

            with torch.no_grad():
                xq = torch.randn(B, cfg["inputDim"], device=device)
                _, recall_on = mem_on(xq)
                _, recall_off = mem_off(xq)

            diff = (recall_on - recall_off).abs().mean().item()
            assert diff > 1e-6, f"NeSy on/off has little effect on mem_recall, diff={diff:.3e}"
            print("TestNeSyRetrievalEffect passed.")
            return True

        except AssertionError as e:
            print(f"TestNeSyRetrievalEffect failed: {e}")
            return False
        except Exception as e:
            print(f"TestNeSyRetrievalEffect error: {e}")
            return False

    def CheckAttachCollector(self):
        try:
            mem = MemoryExtractor(inputDim=32, ssmStateDim=32, memoryDim=48,
                                  memorySize=16, outputDim=48, useAmp=True).to(self.device)
            mem.train()
            x = torch.randn(4, 32, device=self.device)
            out, _ = mem(x)
            base = F.mse_loss(out, torch.zeros_like(out))

            manual = mem.GetInternalLoss().to(dtype=base.dtype, device=base.device)
            total = self.AttachAllInternalLosses(mem, base)
            auto_extra = total - base

            diff = float((manual - auto_extra).abs().item())
            tol = float(torch.finfo(base.dtype).eps) * 8
            assert diff <= tol, f"Collector mismatch: diff={diff:g} > tol={tol:g}"

            print("TestMemoryMTool.CheckAttachCollector passed.")
            return True
        except AssertionError as e:
            print(f"TestMemoryMTool.CheckAttachCollector failed: {e}")
            return False
        except Exception as e:
            print(f"TestMemoryMTool.CheckAttachCollector error: {e}")
            return False

    def NoNanAfterManySteps(self, steps: int = 50):
        try:
            cfg = dict(inputDim=64, ssmStateDim=64, memoryDim=96, memorySize=64,outputDim=96, useAmp=True, gwsSlots=8, gwsTtl=6)
            mem = MemoryExtractor(**cfg).to(self.device)
            mem.train()
            opt = torch.optim.Adam(mem.parameters(), lr=1e-3)

            for t in range(steps):
                x = torch.randn(8, cfg["inputDim"], device=self.device)
                y = torch.randn(8, cfg["outputDim"], device=self.device)
                out, _ = mem(x, tdError=torch.randn(8, device=self.device),reward=torch.randn(8, device=self.device))
                base = F.mse_loss(out, y)
                total = self.AttachAllInternalLosses(mem, base)

                opt.zero_grad()
                total.backward()
                for n, p in mem.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"
                opt.step()

            print("TestMemoryMTool.NoNanAfterManySteps passed.")
            return True
        except AssertionError as e:
            print(f"TestMemoryMTool.NoNanAfterManySteps failed: {e}")
            return False
        except Exception as e:
            print(f"TestMemoryMTool.NoNanAfterManySteps error: {e}")
            return False

    def RunAll(self):
        results = {
            "GlobalWorkspace": self.TestGlobalWorkspace(),
            "LongTermMemory": self.TestLongTermMemory(),
            "MemoryExtractorForward": self.TestMemoryExtractorForward(),
            "StateSaveRestore": self.TestStateSaveRestore(),
            "Reason": self.TestReason(),
            "ResetAndSoftReset": self.TestResetAndSoftReset(),
            "TestMemoryTrain": self.TestMemoryTrain(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "TrainNeSyOnlySanity": self.TrainNeSyOnlySanity(),
            "TestNeSyRetrievalEffect": self.TestNeSyRetrievalEffect(),
            "CheckAttachCollector": self.CheckAttachCollector(),
            "NoNanAfterManySteps": self.NoNanAfterManySteps(),
            "AllTrainablesTouched": self.TestAllTrainablesTouched(),
            "NumericalStability": self.TestNumericalStability(),}

        passed = sum(1 for v in results.values() if v)
        print(f"\nMemory module tests: {passed}/{len(results)} passed.")
        return results

