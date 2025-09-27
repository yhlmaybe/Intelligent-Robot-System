from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from DecisionModule import KEYBOARD_LAYOUT


def ClampLogStd(logstd: torch.Tensor, low: float = -6.0, high: float = 2.0) -> torch.Tensor:
    return torch.clamp(logstd, low, high)


def KLDiagNormal(muQ: torch.Tensor, logstdQ: torch.Tensor, muP: torch.Tensor, logstdP: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(2 * logstdQ)
    var_p = torch.exp(2 * logstdP)
    kl = 0.5 * (((var_q + (muQ - muP) ** 2) / var_p).sum(-1) + 2 * (logstdP - logstdQ).sum(-1) - muQ.size(-1))
    return kl


def BalancedKL(muQ: torch.Tensor,logstdQ: torch.Tensor,muP: torch.Tensor,logstdP: torch.Tensor,alpha: float = 0.8,freeNats: float = 1.0,) -> torch.Tensor:
    mu_p_sg, logstd_p_sg = muP.detach(), logstdP.detach()
    mu_q_sg, logstd_q_sg = muQ.detach(), logstdQ.detach()
    kl_qp = KLDiagNormal(muQ, logstdQ, mu_p_sg, logstd_p_sg)
    kl_pq = KLDiagNormal(mu_q_sg, logstd_q_sg, muP, logstdP)
    kl = alpha * kl_qp + (1.0 - alpha) * kl_pq
    if freeNats and freeNats > 0:
        kl = torch.relu(kl - freeNats)
    return kl


class ActionEncoder(nn.Module):
    def __init__(self, numDiscrete: int = 106, contDim: int = 2, outDim: int = 128):
        super().__init__()
        self.disc_proj = nn.Linear(numDiscrete, outDim, bias=False)
        self.cont_net = nn.Sequential(nn.Linear(contDim, 64), nn.ReLU(), nn.Linear(64, outDim))
        self.fuse = nn.Sequential(nn.Linear(outDim * 2, outDim), nn.Tanh())
        nn.init.normal_(self.disc_proj.weight, mean=0.0, std=0.02)

    def forward(self, keysOnehot: torch.Tensor, mouseDelta: Optional[torch.Tensor] = None) -> torch.Tensor:
        disc_vec = self.disc_proj(keysOnehot.float())
        if mouseDelta is None:
            return disc_vec
        cont_vec = self.cont_net(mouseDelta.float())
        return self.fuse(torch.cat([disc_vec, cont_vec], dim=-1))


class RSSMWorldModel(nn.Module):
    def __init__(
        self,
        visionDim: int = 512,
        actionDim: int = 128,
        deterDim: int = 256,
        stochDim: int = 32,
        stateDim: int = 256,
        useDecoder: bool = True,
        useMemory: bool = False,
        memoryCapacity: int = 4096,
        memoryPath: Optional[str] = None,
        memoryAutosaveEvery: int = 0,
        nsEnabled: bool = True,
        nsLambdaExclusive: float = 1e-2,
        nsLambdaAtLeastOne: float = 1e-2,
        nsLambdaImplication: float = 1e-2,
        nsBiasPrior: bool = True,
        metaDim: int = 32,):
        super().__init__()

        self.vision_dim = visionDim
        self.action_dim = actionDim
        self.deter_dim = deterDim
        self.stoch_dim = stochDim
        self.state_dim = stateDim
        self.use_decoder = useDecoder

        self.obs_enc = nn.Sequential(
            nn.LayerNorm(visionDim),
            nn.Linear(visionDim, stateDim),
            nn.GELU(),
            nn.LayerNorm(stateDim),
            nn.Linear(stateDim, stochDim),)

        self.action_encoder = ActionEncoder(numDiscrete=106, contDim=2, outDim=actionDim)
        self.act_proj = nn.Sequential(nn.Linear(actionDim, stochDim), nn.LayerNorm(stochDim), nn.Tanh())

        self.gru = nn.GRUCell(input_size=stochDim + stochDim, hidden_size=deterDim)

        self.prior_net = nn.Sequential(nn.Linear(deterDim, 2 * stochDim))
        self.post_net = nn.Sequential(nn.Linear(deterDim + stochDim, 2 * stochDim))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + stochDim),
            nn.Linear(deterDim + stochDim, stateDim),
            nn.LayerNorm(stateDim),)

        self.rew_head = nn.Sequential(nn.Linear(stateDim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.done_head = nn.Sequential(nn.Linear(stateDim, 256), nn.ReLU(), nn.Linear(256, 1))
        nn.init.zeros_(self.rew_head[-1].bias)
        nn.init.zeros_(self.done_head[-1].bias)

        if useDecoder:
            self.obs_dec = nn.Sequential(nn.Linear(stateDim, stateDim), nn.GELU(), nn.Linear(stateDim, visionDim))

        self._use_memory = bool(useMemory)
        self._mem_capacity = int(memoryCapacity)
        self._mem_path = memoryPath
        self._mem_autosave_every = int(memoryAutosaveEvery)
        self._mem_add_count = 0

        if self._use_memory:
            self.register_buffer("_mem_keys", torch.zeros(self._mem_capacity, stochDim))
            self.register_buffer("_mem_vals", torch.zeros(self._mem_capacity, deterDim))
            self._mem_size: int = 0
            self._mem_ptr: int = 0
        else:
            self._mem_keys = self._mem_vals = None
            self._mem_size = self._mem_ptr = 0

        self._ns_enabled = bool(nsEnabled)
        self._ns_bias_prior = bool(nsBiasPrior)
        self._ns_K: int = 24

        self.ns_exclusives = [
            [0, 1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15, 16, 17],]
        
        self.ns_atleast_one = [
            [0, 1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15, 16, 17],]
        
        self.ns_implications = [
            (12, 1),
            (11, 0),
            (10, 0),
            (1, 13),
            (2, 14),
            (3, 15),
            (6, 16),
            (6, 14),
            (9, 15),
            (7, 13),
            (20, 17),
            (21, 17),
            (22, 13),]

        if self._ns_enabled:
            hid = max(128, stochDim)
            self.ns_head_post = nn.Sequential(nn.Linear(deterDim + stochDim, hid), nn.GELU(), nn.Linear(hid, self._ns_K))
            self.ns_head_prior = nn.Sequential(nn.Linear(deterDim, hid), nn.GELU(), nn.Linear(hid, self._ns_K))
            self.ns_to_delta_e = nn.Linear(self._ns_K, stochDim)
            self.ns_to_delta_mu = nn.Linear(self._ns_K, stochDim)
            self.ns_gate_e = nn.Linear(deterDim + stochDim, stochDim)
            self.ns_gate_mu = nn.Linear(deterDim + stochDim, stochDim)

            self.ns_lambda_excl = float(nsLambdaExclusive)
            self.ns_lambda_alo = float(nsLambdaAtLeastOne)
            self.ns_lambda_impl = float(nsLambdaImplication)
        else:
            self.ns_head_post = self.ns_head_prior = None
            self.ns_to_delta_e = self.ns_to_delta_mu = None
            self.ns_gate_e = self.ns_gate_mu = None
            self.ns_exclusives = self.ns_atleast_one = self.ns_implications = []
            self.ns_lambda_excl = self.ns_lambda_alo = self.ns_lambda_impl = 0.0

        self._meta_dim = int(metaDim)
        self.meta_to_e = nn.Linear(self._meta_dim, stochDim, bias=False)
        self.meta_to_mu = nn.Linear(self._meta_dim, stochDim, bias=False)
        self.meta_gate_e = nn.Linear(deterDim + stochDim, stochDim)
        self.meta_gate_mu = nn.Linear(deterDim + stochDim, stochDim)
        self.meta_ctx = nn.Parameter(torch.randn(self._meta_dim) * 1e-2)

        nn.init.xavier_uniform_(self.meta_to_e.weight)
        nn.init.xavier_uniform_(self.meta_to_mu.weight)
        nn.init.constant_(self.meta_gate_e.bias, -1.0)
        nn.init.constant_(self.meta_gate_mu.bias, -1.0)

        self.ResetHidden()

        if self._use_memory and self._mem_path:
            self.LoadMemory(self._mem_path, map_location=None, strict=False)
        if self._use_memory:
            self.mem_val_to_e = nn.Sequential(nn.Linear(deterDim, stochDim), nn.LayerNorm(stochDim))
        else:
            self.mem_val_to_e = None

    def SaveMemory(self, path: Optional[str] = None):
        if not self._use_memory:
            return
        p = path or self._mem_path
        if not p:
            return
        dirpath = os.path.dirname(p)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        payload = {
            "mem_keys": self._mem_keys[: self._mem_size].detach().cpu(),
            "mem_vals": self._mem_vals[: self._mem_size].detach().cpu(),
            "mem_size": self._mem_size,
            "mem_ptr": self._mem_ptr,}
        
        torch.save(payload, p)

    def LoadMemory(self, path: str, map_location: Optional[str] = None, strict: bool = False):
        if not self._use_memory:
            return
        if not os.path.exists(path):
            if strict:
                raise FileNotFoundError(path)
            return
        payload = torch.load(path, map_location=map_location or "cpu")
        keys = payload.get("mem_keys", None)
        vals = payload.get("mem_vals", None)
        size = int(payload.get("mem_size", 0))
        ptr = int(payload.get("mem_ptr", 0))
        if keys is None or vals is None:
            if strict:
                raise ValueError("Invalid memory file.")
            return
        device = self._mem_keys.device
        N = min(size, self._mem_capacity, keys.size(0), vals.size(0))
        self._mem_keys[:N] = keys[:N].to(device)
        self._mem_vals[:N] = vals[:N].to(device)
        self._mem_size = N
        self._mem_ptr = min(max(ptr, 0), self._mem_capacity - 1)
        self._mem_path = path

    def ResetMemory(self):
        if not self._use_memory:
            return
        self._mem_keys.zero_()
        self._mem_vals.zero_()
        self._mem_size = 0
        self._mem_ptr = 0

    @torch.no_grad()
    def MemAdd(self, keyE: torch.Tensor, valH: torch.Tensor):
        if not self._use_memory:
            return
        if keyE.dim() == 1:
            keyE = keyE.unsqueeze(0)
        if valH.dim() == 1:
            valH = valH.unsqueeze(0)
        B = keyE.size(0)
        if B >= self._mem_capacity:
            keyE = keyE[-self._mem_capacity :]
            valH = valH[-self._mem_capacity :]
            B = keyE.size(0)
        for i in range(B):
            self._mem_keys[self._mem_ptr] = keyE[i]
            self._mem_vals[self._mem_ptr] = valH[i]
            self._mem_ptr = (self._mem_ptr + 1) % self._mem_capacity
            self._mem_size = min(self._mem_size + 1, self._mem_capacity)

        if self._mem_path and self._mem_autosave_every > 0:
            self._mem_add_count += 1
            if self._mem_add_count % self._mem_autosave_every == 0:
                self.SaveMemory(self._mem_path)

    @torch.no_grad()
    def MemRetrieve(self, queryE: torch.Tensor) -> Optional[torch.Tensor]:
        if (not self._use_memory) or (self._mem_size == 0):
            return None
        single = queryE.dim() == 1
        if single:
            queryE = queryE.unsqueeze(0)
        keys = self._mem_keys[: self._mem_size] # [N, stochDim]
        q = F.normalize(queryE, dim=-1, eps=1e-6)
        k = F.normalize(keys, dim=-1, eps=1e-6)
        sims = torch.matmul(q, k.t()) # [B, N]
        idx = sims.argmax(dim=-1)
        vals = self._mem_vals[idx] # [B, deterDim]
        return vals[0] if single else vals

    def ResetHidden(self, batchSize: int = 1, device: torch.device | str = "cpu"):
        device = torch.device(device)
        self._h = torch.zeros(batchSize, self.deter_dim, device=device)
        self._z = torch.zeros(batchSize, self.stoch_dim, device=device)

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._h, self._z

    def ImportState(self, h: torch.Tensor, z: torch.Tensor):
        self._h = h.detach().clone()
        self._z = z.detach().clone()

    def NsProjectProbs(self, logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        sig = logits / max(1e-6, temp)
        probs = sig.sigmoid()
        for grp in self.ns_exclusives:
            if len(grp) <= 1:
                continue
            g = sig[:, grp]
            g_sm = F.softmax(g, dim=-1)
            probs = probs.clone()
            probs[:, grp] = g_sm
        return probs

    @torch.no_grad()
    def NsProjectRuntime(
        self,
        P: torch.Tensor,
        *,
        aloTau: float = 0.60,
        implAlpha: float = 1.0,
        temp: float = 1.0,) -> Tuple[torch.Tensor, torch.Tensor]:

        eps = 1e-6
        Q = P.detach().clamp(eps, 1 - eps).clone()

        for grp in self.ns_exclusives:
            if len(grp) <= 1:
                continue
            g = torch.log(Q[:, grp]) / max(1e-6, temp)
            g_sm = F.softmax(g, dim=-1)
            Q[:, grp] = g_sm.clamp(eps, 1 - eps)

        for grp in self.ns_atleast_one:
            if len(grp) == 0:
                continue
            m = Q[:, grp].max(dim=-1, keepdim=True).values
            scale = torch.where(m < aloTau, aloTau / (m + eps), torch.ones_like(m))
            Q[:, grp] = (Q[:, grp] * scale).clamp(eps, 1 - eps)

        for a, b in self.ns_implications:
            Q[:, b] = torch.maximum(Q[:, b], implAlpha * Q[:, a]).clamp(eps, 1 - eps)

        excl_v = Q.new_zeros(Q.size(0))
        for grp in self.ns_exclusives:
            if len(grp) <= 1:
                continue
            Psum = Q[:, grp].sum(dim=-1)
            P2 = (Q[:, grp] * Q[:, grp]).sum(dim=-1)
            excl_v = excl_v + 0.5 * (Psum * Psum - P2)

        alo_v = Q.new_zeros(Q.size(0))
        for grp in self.ns_atleast_one:
            if len(grp) == 0:
                continue
            alo_v = alo_v + F.relu(aloTau - Q[:, grp].max(dim=-1).values)

        impl_v = Q.new_zeros(Q.size(0))
        for a, b in self.ns_implications:
            impl_v = impl_v + F.relu(Q[:, a] - Q[:, b])

        pen = excl_v + alo_v + impl_v
        pen = (pen / (pen.detach().quantile(0.9) + 1e-6)).clamp(0.0, 1.0)
        return Q, pen

    def NsConfidence(self, P: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        H = -( P.clamp(eps, 1 - eps) * torch.log(P.clamp(eps, 1 - eps)) + (1 - P).clamp(eps, 1 - eps) * torch.log((1 - P).clamp(eps, 1 - eps)))
        H = H.mean(dim=-1, keepdim=True)
        Hmax = torch.tensor(0.6931, device=P.device)
        conf = (1.0 - H / Hmax).clamp(0.0, 1.0)
        return conf

    def NsLogicLosses(self, probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        excl = probs.new_zeros([])
        for grp in self.ns_exclusives:
            if len(grp) <= 1:
                continue
            P = probs[:, grp]
            s1 = P.sum(dim=-1)
            s2 = (P * P).sum(dim=-1)
            excl = excl + 0.5 * (s1 * s1 - s2).mean()

        alo = probs.new_zeros([])
        tau = 0.60
        for grp in self.ns_atleast_one:
            if len(grp) == 0:
                continue
            top1 = probs[:, grp].max(dim=-1).values
            alo = alo + (F.relu(tau - top1) ** 2).mean()

        impl = probs.new_zeros([])
        for a, b in self.ns_implications:
            impl = impl + F.relu(probs[:, a] - probs[:, b]).mean()

        loss = self.ns_lambda_excl * excl + self.ns_lambda_alo * alo + self.ns_lambda_impl * impl
        stats = {"excl": excl.detach(), "alo": alo.detach(), "impl": impl.detach()}
        return loss, stats

    @torch.no_grad()
    def StepPriorOnly(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        a_t = self.act_proj(actionEnc)
        h_next = self.gru(torch.cat([zPrev, a_t], dim=-1), hPrev)

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        if self._ns_enabled and self._ns_bias_prior:
            ns_logits = self.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.NsConfidence(P_raw)

            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)

            dmu = self.ns_to_delta_mu(Q)
            mu_p = mu_p + gate * dmu

        if sample:
            eps = torch.randn_like(mu_p)
            z_next = mu_p + torch.exp(logstd_p) * eps
        else:
            z_next = mu_p

        s_next = self.state_proj(torch.cat([h_next, z_next], dim=-1))
        r_pred = self.rew_head(s_next).squeeze(-1)
        d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)
        return h_next, z_next, s_next, r_pred, d_prob

    def StepPosterior(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        visionIn: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:

        raw_e = self.obs_enc(visionIn)
        a_t = self.act_proj(actionEnc)
        h_next = self.gru(torch.cat([zPrev, a_t], dim=-1), hPrev)

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        e_in = raw_e
        if self._use_memory:
            with torch.no_grad():
                mem_h = self.MemRetrieve(raw_e)
            if mem_h is not None:
                mem_e = self.mem_val_to_e(mem_h)
                gate_m = torch.sigmoid(self.meta_gate_e(torch.cat([h_next, raw_e], dim=-1)))
                e_in = raw_e + gate_m * mem_e

        if self._meta_dim > 0:
            ctx = self.meta_ctx.view(1, -1).expand_as(h_next[:, : self._meta_dim])
            de_meta = self.meta_to_e(ctx)
            dmu_meta = self.meta_to_mu(ctx)
            gate_e = torch.sigmoid(self.meta_gate_e(torch.cat([h_next, e_in], dim=-1)))
            gate_mu = torch.sigmoid(self.meta_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            e_in = e_in + gate_e * de_meta
            mu_p = mu_p + gate_mu * dmu_meta

        if self._ns_enabled:
            ns_logits = self.ns_head_post(torch.cat([h_next, e_in], dim=-1))
            P_raw = torch.sigmoid(ns_logits)
            P_train = self.NsProjectProbs(ns_logits)
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.NsConfidence(P_raw)

            base_gate = torch.sigmoid(self.ns_gate_e(torch.cat([h_next, e_in], dim=-1)))
            gate_scale = (1.0 - 0.25 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)

            de = self.ns_to_delta_e(Q)
            e_t = e_in + gate * de
        else:
            ns_logits = None
            P_train = None
            e_t = e_in

        mu_q, logstd_q = self.post_net(torch.cat([h_next, e_t], dim=-1)).chunk(2, dim=-1)
        logstd_q = ClampLogStd(logstd_q)
        z_next = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q) if sample else mu_q

        s_next = self.state_proj(torch.cat([h_next, z_next], dim=-1))
        r_pred = self.rew_head(s_next).squeeze(-1)
        d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)

        out = {
            "h_next": h_next,
            "z_next": z_next,
            "s_next": s_next,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "mu_p": mu_p,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "logstd_q": logstd_q,}
        
        if self._ns_enabled:
            out["ns_logits"] = ns_logits
            out["ns_probs"] = P_train
        if self.use_decoder:
            out["recon"] = self.obs_dec(s_next)
            out["recon_target"] = visionIn

        self._h = h_next.detach()
        self._z = z_next.detach()

        if self._use_memory:
            with torch.no_grad():
                self.MemAdd(raw_e.detach(), h_next.detach())
        return out

    def ForwardTrainSeq(
        self,
        visionSeq: torch.Tensor,  # [B, visionDim]
        keysVec: torch.Tensor,  # [B,106]
        mouseSeq: torch.Tensor,  # [B,2]
        h0: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        rewardSeq: Optional[torch.Tensor] = None,  # [B]
        doneSeq: Optional[torch.Tensor] = None,  # [B]
        alphaKl: float = 0.8,
        freeNats: float = 1.0,
        reconCoef: float = 1.0,
        rewardCoef: float = 1.0,
        doneCoef: float = 1.0,
        nsCoef: float = 1.0,
        nsDistillCoef: float = 1e-2, 
        nsPriorLogicCoef: float = 1e-3,) -> Dict[str, torch.Tensor]:

        B = visionSeq.size(0)
        device = visionSeq.device
        if h0 is None:
            h0 = torch.zeros(B, self.deter_dim, device=device)
        if z0 is None:
            z0 = torch.zeros(B, self.stoch_dim, device=device)

        a_enc = self.action_encoder(keysVec, mouseSeq)
        a_enc = self.act_proj(a_enc)

        h_next = self.gru(torch.cat([z0, a_enc], dim=-1), h0)

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        raw_e = self.obs_enc(visionSeq)
        e_in = raw_e
        if self._use_memory:
            with torch.no_grad():
                mem_h = self.MemRetrieve(raw_e)
            if mem_h is not None:
                mem_e = self.mem_val_to_e(mem_h)
                gate_m = torch.sigmoid(self.meta_gate_e(torch.cat([h_next, raw_e], dim=-1)))
                e_in = raw_e + gate_m * mem_e

        if self._meta_dim > 0:
            ctx = self.meta_ctx.view(1, -1).expand(B, -1)
            de_meta = self.meta_to_e(ctx)
            dmu_meta = self.meta_to_mu(ctx)
            gate_e_meta = torch.sigmoid(self.meta_gate_e(torch.cat([h_next, e_in], dim=-1)))
            gate_mu_meta = torch.sigmoid(self.meta_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            e_in = e_in + gate_e_meta * de_meta
            mu_p = mu_p + gate_mu_meta * dmu_meta

        ns_prior_loss = visionSeq.new_tensor(0.0)
        ns_prior_logic = visionSeq.new_tensor(0.0)
        if self._ns_enabled:
            logits_pr = self.ns_head_prior(h_next)  # [B, K]
            P_pr_raw = torch.sigmoid(logits_pr)
            P_pr_train = self.NsProjectProbs(logits_pr) 

            de_mu = self.ns_to_delta_mu(P_pr_train)
            base_gate_mu = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            with torch.no_grad():
                _, pen_pr = self.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf_pr = self.NsConfidence(P_pr_raw)  # grad flows
            gate_scale_mu = (1.0 - 0.25 * pen_pr.view(-1, 1)) * (0.6 + 0.4 * conf_pr)
            gate_mu = (base_gate_mu * gate_scale_mu).clamp(0.0, 1.0)
            mu_p = mu_p + gate_mu * de_mu
        else:
            logits_pr = None
            P_pr_train = None

        ns_loss = visionSeq.new_tensor(0.0)
        if self._ns_enabled:
            ns_logits = self.ns_head_post(torch.cat([h_next, e_in], dim=-1))
            sym_probs = self.NsProjectProbs(ns_logits)
            de = self.ns_to_delta_e(sym_probs)
            gate_e = torch.sigmoid(self.ns_gate_e(torch.cat([h_next, e_in], dim=-1)))
            e_t = e_in + gate_e * de
            ns_loss, _ = self.NsLogicLosses(sym_probs)
        else:
            ns_logits = None
            e_t = e_in

        ns_distill = visionSeq.new_tensor(0.0)
        if self._ns_enabled and (logits_pr is not None) and (ns_logits is not None) and (nsDistillCoef > 0):
            with torch.no_grad():
                P_teacher = torch.sigmoid(ns_logits) 
            ns_distill = F.binary_cross_entropy_with_logits(logits_pr, P_teacher, reduction="mean")
            if nsPriorLogicCoef > 0 and (P_pr_train is not None):
                ns_prior_logic, _ = self.NsLogicLosses(P_pr_train)

        mu_q, logstd_q = self.post_net(torch.cat([h_next, e_t], dim=-1)).chunk(2, dim=-1)
        logstd_q = ClampLogStd(logstd_q)
        z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
        s1 = self.state_proj(torch.cat([h_next, z1], dim=-1))
        r1 = self.rew_head(s1).squeeze(-1)
        d_logit = self.done_head(s1).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)

        loss_recon = visionSeq.new_tensor(0.0)
        if self.use_decoder:
            recon = self.obs_dec(s1)
            loss_recon = F.mse_loss(recon, visionSeq, reduction="mean")

        loss_reward = (
            F.mse_loss(r1, torch.zeros_like(r1), reduction="mean")
            if rewardSeq is None
            else F.mse_loss(r1, rewardSeq, reduction="mean"))
        
        loss_done = F.binary_cross_entropy_with_logits(
            d_logit,
            torch.zeros_like(d_logit) if doneSeq is None else doneSeq.float(),
            reduction="mean",)
        
        loss_kl = BalancedKL(mu_q, logstd_q, mu_p, logstd_p, alpha=alphaKl, freeNats=freeNats).mean()

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic)

        if self._use_memory:
            with torch.no_grad():
                self.MemAdd(raw_e.detach(), h_next.detach())

        return {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "s_last": s1,
            "h_last": h_next,
            "z_last": z1,
            "r_pred": r1,
            "d_prob": d_prob,}

    def MetaReset(self):
        with torch.no_grad():
            self.meta_ctx.zero_()

    def MetaInnerStep(
        self,
        vision: torch.Tensor,
        keys: torch.Tensor,
        mouse: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        lr: float = 0.1,):

        self.train()
        for p in self.parameters():
            p.requires_grad_(False)
        self.meta_ctx.requires_grad_(True)
        self.meta_to_e.weight.requires_grad_(True)
        self.meta_to_mu.weight.requires_grad_(True)

        out = self.ForwardTrainSeq(vision, keys, mouse, rewardSeq=reward, doneSeq=done)
        loss = out["loss"]
        loss.backward()
        with torch.no_grad():
            if self.meta_ctx.grad is not None:
                self.meta_ctx -= lr * self.meta_ctx.grad
                self.meta_ctx.grad.zero_()
            if self.meta_to_e.weight.grad is not None:
                self.meta_to_e.weight -= lr * self.meta_to_e.weight.grad
                self.meta_to_e.weight.grad.zero_()
            if self.meta_to_mu.weight.grad is not None:
                self.meta_to_mu.weight -= lr * self.meta_to_mu.weight.grad
                self.meta_to_mu.weight.grad.zero_()

        for p in self.parameters():
            p.requires_grad_(True)
        return float(loss.detach().item())


class WMAdapterForPlanner(nn.Module):
    def __init__(
        self,
        wm: RSSMWorldModel,
        maxCode: int,
        baseCodes: List[int],
        skillCodes: List[int],
        extraCodes: List[int],
        noSkillId: Optional[int],
        deterministicZ: bool = True,):
        super().__init__()
        self.wm = wm
        self.max_code = maxCode
        self.register_buffer("base_codes_buf", torch.tensor(baseCodes, dtype=torch.long))
        self.register_buffer("skill_codes_buf", torch.tensor(skillCodes, dtype=torch.long))
        self.register_buffer("extra_codes_buf", torch.tensor(extraCodes, dtype=torch.long))
        self.no_skill_id = noSkillId
        self.deterministic_z = deterministicZ

    @staticmethod
    def MakeKeys(
        baseAct: torch.Tensor,
        skillIdx: torch.Tensor,
        extraAct: torch.Tensor,
        baseCodes: torch.Tensor,
        skillCodes: torch.Tensor,
        extraCodes: torch.Tensor,
        noSkillId: Optional[int],
        numDiscrete: int,) -> torch.Tensor:
        B, device = baseAct.size(0), baseAct.device
        base_M = F.one_hot(baseCodes.to(device), num_classes=numDiscrete).float()
        extra_M = F.one_hot(extraCodes.to(device), num_classes=numDiscrete).float()
        base_part = baseAct @ base_M
        extra_part = extraAct @ extra_M
        if noSkillId is None:
            chosen = skillIdx
            valid = torch.ones_like(chosen, dtype=torch.bool)
        else:
            valid = skillIdx != noSkillId
            chosen = skillIdx.clamp_max(skillCodes.numel() - 1)
        sel_codes = skillCodes.to(device)[chosen.clamp_min(0)]
        skill_onehot = F.one_hot(sel_codes, num_classes=numDiscrete).float() * valid.view(-1, 1).float()
        key_vec = base_part + extra_part + skill_onehot
        return key_vec

    @torch.no_grad()
    def Step(
        self,
        aMouse: torch.Tensor,  # [B,2]
        aSkill: torch.Tensor,  # [B]
        aBase: Optional[torch.Tensor] = None,  # [B, n_base]
        aExtra: Optional[torch.Tensor] = None,  # [B, n_extra]
        h0: Optional[torch.Tensor] = None,  # [B, deterDim]
        z0: Optional[torch.Tensor] = None,  # [B, stochDim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = aMouse.size(0)
        device = aMouse.device
        if aBase is None:
            aBase = torch.zeros(B, self.base_codes_buf.numel(), device=device)
        if aExtra is None:
            aExtra = torch.zeros(B, self.extra_codes_buf.numel(), device=device)

        numDiscrete = int(self.wm.action_encoder.disc_proj.in_features)
        keysSeq = self.MakeKeys(
            aBase,
            aSkill,
            aExtra,
            self.base_codes_buf.to(device),
            self.skill_codes_buf.to(device),
            self.extra_codes_buf.to(device),
            self.no_skill_id,
            numDiscrete,)
        
        a_enc = self.wm.action_encoder(keysSeq, aMouse)

        if h0 is None or z0 is None:
            h_prev, z_prev = self.wm.ExportState()
            if h_prev is None or h_prev.size(0) != B or h_prev.device != device:
                h_prev = torch.zeros(B, self.wm.deter_dim, device=device)
                z_prev = torch.zeros(B, self.wm.stoch_dim, device=device)
        else:
            h_prev, z_prev = h0, z0

        h1, z1, s1, r, d = self.wm.StepPriorOnly(h_prev, z_prev, a_enc, sample=not self.deterministic_z)
        return s1, r, d


class TestWorldMTool:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(0)

        self.base_codes = [KEYBOARD_LAYOUT["base_keys"][k] for k in KEYBOARD_LAYOUT["base_keys"]]
        self.skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in KEYBOARD_LAYOUT["skill_keys"]]
        self.extra_codes = []
        for grp in ["menu_keys", "system_keys", "alpha_keys"]:
            self.extra_codes += [KEYBOARD_LAYOUT[grp][k] for k in KEYBOARD_LAYOUT[grp]]

        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        self.max_code = max(all_codes)

        self.wm = RSSMWorldModel(
            visionDim=512,
            actionDim=128,
            deterDim=256,
            stochDim=32,
            stateDim=256,
            useDecoder=True,
            useMemory=False,
            nsEnabled=True,
        ).to(self.device)
        self.wm.ResetHidden(batchSize=4, device=self.device)

    def TestActionEncoder(self):
        enc = ActionEncoder(numDiscrete=106, contDim=2, outDim=128).to(self.device)
        B = 3
        keys = torch.zeros(B, 106, device=self.device)
        keys[:, 17] = 1.0
        keys[:, 57] = 1.0
        mouse = torch.randn(B, 2, device=self.device)
        y1 = enc(keys, mouse)
        y2 = enc(keys, None)
        ok = (y1.shape == (B, 128)) and (y2.shape == (B, 128))
        print("ActionEncoder test " + ("passed." if ok else "failed."))
        return ok

    def TestRSSMStepPosterior(self):
        B = 4
        vision = torch.randn(B, 512, device=self.device)
        keys = torch.zeros(B, 106, device=self.device)
        keys[:, 17] = 1.0
        keys[:, 57] = 1.0
        mouse = torch.randn(B, 2, device=self.device)
        a_enc = self.wm.action_encoder(keys, mouse)
        h0, z0 = self.wm.ExportState()
        out = self.wm.StepPosterior(h0, z0, vision, a_enc, sample=False)
        ok_shapes = (
            out["h_next"].shape == (B, self.wm.deter_dim)
            and out["z_next"].shape == (B, self.wm.stoch_dim)
            and out["s_next"].shape == (B, self.wm.state_dim)
            and out["r_pred"].shape == (B,)
            and out["d_prob"].shape == (B,))
        
        h1, z1 = self.wm.ExportState()
        changed = (not torch.allclose(h0, h1)) or (not torch.allclose(z0, z1))
        in_range = (out["d_prob"].min() >= 0.0) and (out["d_prob"].max() <= 1.0)
        ok = ok_shapes and changed and in_range
        print("RSSM StepPosterior test " + ("passed." if ok else "failed."))
        return ok

    def TestRSSMStepPriorOnly(self):
        B = 4
        keys = torch.zeros(B, 106, device=self.device)
        keys[:, 30] = 1.0
        mouse = torch.randn(B, 2, device=self.device)
        a_enc = self.wm.action_encoder(keys, mouse)
        h0, z0 = self.wm.ExportState()
        h_before, z_before = h0.clone(), z0.clone()
        h1, z1, s1, r, d = self.wm.StepPriorOnly(h0, z0, a_enc, sample=False)
        ok_shapes = (
            h1.shape == (B, self.wm.deter_dim)
            and z1.shape == (B, self.wm.stoch_dim)
            and s1.shape == (B, self.wm.state_dim)
            and r.shape == (B,)
            and d.shape == (B,))
        hin, zin = self.wm.ExportState()
        not_written = torch.allclose(hin, h_before) and torch.allclose(zin, z_before)
        ok = ok_shapes and not_written
        print("RSSM StepPriorOnly test " + ("passed." if ok else "failed."))
        return ok

    def TestForwardTrainSeq(self):
        B = 2
        vision = torch.randn(B, 512, device=self.device)
        keys = torch.zeros(B, 106, device=self.device)
        keys[:, 17] = 1.0
        mouse = torch.randn(B, 2, device=self.device)
        out = self.wm.ForwardTrainSeq(
            visionSeq=vision,
            keysVec=keys,
            mouseSeq=mouse,
            rewardSeq=None,
            doneSeq=None,
            alphaKl=0.8,
            freeNats=1.0,
            reconCoef=1.0,
            rewardCoef=1.0,
            doneCoef=1.0,)
        
        loss = out["loss"]
        if not torch.isfinite(loss):
            print("ForwardTrainSeq loss is not finite.")
            return False
        loss.backward()
        print(
            "RSSM ForwardTrainSeq test passed. loss =",
            float(loss.item()),
            " | distill=",
            float(out["loss_ns_distill"].item()),)
        
        return True

    def TestWMAdapterForPlanner(self):
        adapter = WMAdapterForPlanner(
            wm=self.wm,
            maxCode=self.max_code,
            baseCodes=self.base_codes,
            skillCodes=self.skill_codes,
            extraCodes=self.extra_codes,
            noSkillId=None,
            deterministicZ=True,
        ).to(self.device)
        B = 3
        a_mouse = torch.randn(B, 2, device=self.device)
        a_base = (torch.rand(B, len(self.base_codes), device=self.device) > 0.5).float()
        a_extra = (torch.rand(B, len(self.extra_codes), device=self.device) > 0.5).float()
        a_skill = torch.randint(low=0, high=len(self.skill_codes), size=(B,), device=self.device)
        s1, r, d = adapter.Step(aMouse=a_mouse, aSkill=a_skill, aBase=a_base, aExtra=a_extra)
        ok_shapes = s1.shape == (B, self.wm.state_dim) and r.shape == (B,) and d.shape == (B,)
        print("WMAdapterForPlanner.Step test " + ("passed." if ok_shapes else "failed."))
        return ok_shapes

    def RunAll(self):
        results = []
        results.append(self.TestActionEncoder())
        results.append(self.TestRSSMStepPosterior())
        results.append(self.TestRSSMStepPriorOnly())
        results.append(self.TestForwardTrainSeq())
        results.append(self.TestWMAdapterForPlanner())
        passed = sum(1 for x in results if x)
        total = len(results)
        print(f"\nWorldModel test summary: {passed}/{total} passed.")
        return all(results)
