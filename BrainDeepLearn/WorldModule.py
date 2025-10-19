from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from DecisionModule import KEYBOARD_LAYOUT
from FunctionTools import SiteSpec, BaseOnlineWrapper


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


class GrowableLoRALinear(nn.Module):
    def __init__(self, targetLinear: nn.Linear):
        super().__init__()
        assert isinstance(targetLinear, nn.Linear)
        self.target = targetLinear
        self.in_f = targetLinear.in_features
        self.out_f = targetLinear.out_features

        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if addRank <= 0: 
            return
        if init is None: 
            init = {}
        dev, dt = self.target.weight.device, self.target.weight.dtype
        A = init.get("A", torch.randn(addRank, self.in_f, device=dev, dtype=dt) * 1e-4)
        B = init.get("B", torch.zeros(self.out_f, addRank, device=dev, dtype=dt))
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def DeltaWeight(self) -> Optional[torch.Tensor]:
        if len(self.A_list) == 0:
            return None
        dW = self.target.weight.new_zeros(self.out_f, self.in_f)
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            dW = dW + s * (B @ A)
        return dW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.target.weight
        delta = self.DeltaWeight()
        if delta is not None:
            W = W + delta
        return F.linear(x, W, self.target.bias)


class GrowableLoRAGRUCell(nn.Module):
    def __init__(self, targetGRU: nn.GRUCell, gradMomentum: float = 0.9, winnerRatio: float = 1.2):
        super().__init__()
        assert isinstance(targetGRU, nn.GRUCell)
        self.target = targetGRU
        self.input_size  = int(targetGRU.input_size)
        self.hidden_size = int(targetGRU.hidden_size)

        self.A_ih = nn.ParameterList(); self.B_ih = nn.ParameterList(); self.s_ih = nn.ParameterList()

        self.A_hh = nn.ParameterList(); self.B_hh = nn.ParameterList(); self.s_hh = nn.ParameterList()

        self.register_buffer("ema_g2_ih", torch.tensor(0.0))
        self.register_buffer("ema_g2_hh", torch.tensor(0.0))
        self.grad_momentum = float(gradMomentum)
        self.winner_ratio = float(winnerRatio) 

        def hook_ih(grad):
            with torch.no_grad():
                g2 = grad.detach().pow(2).mean()
                self.ema_g2_ih.mul_(self.grad_momentum).add_(g2, alpha=1 - self.grad_momentum)
            return grad
        def hook_hh(grad):
            with torch.no_grad():
                g2 = grad.detach().pow(2).mean()
                self.ema_g2_hh.mul_(self.grad_momentum).add_(g2, alpha=1 - self.grad_momentum)
            return grad

        self.target.weight_ih.register_hook(hook_ih)
        self.target.weight_hh.register_hook(hook_hh)

    @torch.no_grad()
    def DecideSplit(self, addRank: int) -> tuple[int, int]:
        g_ih = float(self.ema_g2_ih)
        g_hh = float(self.ema_g2_hh)

        if g_ih <= 0 and g_hh <= 0:
            r_ih = addRank // 2
            r_hh = addRank - r_ih
            return r_ih, r_hh
        if g_ih > self.winner_ratio * g_hh:
            return addRank, 0
        if g_hh > self.winner_ratio * g_ih:
            return 0, addRank
        
        p_ih = g_ih / (g_ih + g_hh + 1e-12)
        r_ih = int(round(addRank * p_ih))
        r_ih = max(0, min(addRank, r_ih))
        r_hh = addRank - r_ih
        return r_ih, r_hh

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True, mode: str = "auto"):
        if addRank <= 0:
            return
        dev = self.target.weight_ih.device
        dt = self.target.weight_ih.dtype
        I, H = self.input_size, self.hidden_size
        out_rows = 3 * H

        if freezeOld:
            for p in list(self.A_ih)+list(self.B_ih)+list(self.s_ih)+list(self.A_hh)+list(self.B_hh)+list(self.s_hh):
                p.requires_grad_(False)

        if mode == "ih":
            r_ih, r_hh = addRank, 0
        elif mode == "hh":
            r_ih, r_hh = 0, addRank
        elif mode == "both_even":
            r_ih = addRank // 2
            r_hh = addRank - r_ih
        else:
            r_ih, r_hh = self.DecideSplit(addRank)

        if r_ih > 0:
            A = (init.get("A_ih", None) if init else None) or (torch.randn(r_ih, I, device=dev, dtype=dt) * 1e-4)
            B = (init.get("B_ih", None) if init else None) or (torch.zeros(out_rows, r_ih, device=dev, dtype=dt))
            s = (init.get("scale_ih", None) if init else None) or 1e-3
            self.A_ih.append(nn.Parameter(A.contiguous().to(device=dev, dtype=dt)))
            self.B_ih.append(nn.Parameter(B.contiguous().to(device=dev, dtype=dt)))
            self.s_ih.append(nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt)))

        if r_hh > 0:
            A = (init.get("A_hh", None) if init else None) or (torch.randn(r_hh, H, device=dev, dtype=dt) * 1e-4)
            B = (init.get("B_hh", None) if init else None) or (torch.zeros(out_rows, r_hh, device=dev, dtype=dt))
            s = (init.get("scale_hh", None) if init else None) or 1e-3
            self.A_hh.append(nn.Parameter(A.contiguous().to(device=dev, dtype=dt)))
            self.B_hh.append(nn.Parameter(B.contiguous().to(device=dev, dtype=dt)))
            self.s_hh.append(nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt)))

    def DeltaIH(self) -> Optional[torch.Tensor]:
        if len(self.A_ih) == 0:
            return None
        dW = self.target.weight_ih.new_zeros(3*self.hidden_size, self.input_size)
        for A, B, s in zip(self.A_ih, self.B_ih, self.s_ih):
            dW = dW + s * (B @ A)
        return dW

    def DeltaHH(self) -> Optional[torch.Tensor]:
        if len(self.A_hh) == 0:
            return None
        dW = self.target.weight_hh.new_zeros(3*self.hidden_size, self.hidden_size)
        for A, B, s in zip(self.A_hh, self.B_hh, self.s_hh):
            dW = dW + s * (B @ A)
        return dW

    def forward(self, x: torch.Tensor, hPrev: torch.Tensor) -> torch.Tensor:
        W_ih = self.target.weight_ih
        W_hh = self.target.weight_hh
        b_ih = self.target.bias_ih
        b_hh = self.target.bias_hh

        d_ih = self.DeltaIH()
        d_hh = self.DeltaHH()
        if d_ih is not None: W_ih = W_ih + d_ih
        if d_hh is not None: W_hh = W_hh + d_hh

        gi = F.linear(x, W_ih, b_ih) 
        gh = F.linear(hPrev, W_hh, b_hh)  
        i_r, i_i, i_n = gi.chunk(3, dim=1)
        h_r, h_i, h_n = gh.chunk(3, dim=1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        h_next = newgate + inputgate * (hPrev - newgate)
        return h_next



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
            GrowableLoRALinear(nn.Linear(visionDim, stateDim, bias=True)),
            nn.GELU(),
            nn.LayerNorm(stateDim),
            GrowableLoRALinear(nn.Linear(stateDim, stochDim, bias=True)),)

        self.action_encoder = ActionEncoder(numDiscrete=106, contDim=2, outDim=actionDim)

        self.act_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(actionDim, stochDim, bias=True)),
            nn.LayerNorm(stochDim),
            nn.Tanh(),)

        self.gru = GrowableLoRAGRUCell(nn.GRUCell(input_size=stochDim + stochDim, hidden_size=deterDim))

        self.prior_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim, 2 * stochDim, bias=True)))
        
        self.post_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim + stochDim, 2 * stochDim, bias=True)))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + stochDim),
            GrowableLoRALinear(nn.Linear(deterDim + stochDim, stateDim, bias=True)),
            nn.LayerNorm(stateDim),)

        self.rew_head = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, 256, bias=True)),
            nn.ReLU(),
            GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)
        
        self.done_head = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, 256, bias=True)),
            nn.ReLU(),
            GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)
        
        nn.init.zeros_(self.rew_head[-1].target.bias)
        nn.init.zeros_(self.done_head[-1].target.bias)

        self.obs_dec = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, stateDim, bias=True)),
            nn.GELU(),
            GrowableLoRALinear(nn.Linear(stateDim, visionDim, bias=True)),)

        self._use_memory = bool(useMemory)
        self._mem_capacity = int(memoryCapacity)
        self._mem_path = memoryPath
        self._mem_autosave_every = int(memoryAutosaveEvery)
        self._mem_add_count = 0

        self.register_buffer("_mem_keys", torch.zeros(self._mem_capacity, stochDim))
        self.register_buffer("_mem_vals", torch.zeros(self._mem_capacity, deterDim))
        self._mem_size: int = 0
        self._mem_ptr: int = 0

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

        self.mem_val_to_e = nn.Sequential(nn.Linear(deterDim, stochDim), nn.LayerNorm(stochDim))

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


class WorldModelOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: "RSSMWorldModel",
        initRankEach: int = 4,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankObs1: int = 64,
        maxRankObs2: int = 64,
        maxRankAct: int = 64,
        maxRankPrior:int = 64,
        maxRankPost: int = 64,
        maxRankState:int = 64,
        maxRankRew1: int = 32,
        maxRankRew2: int = 8,
        maxRankDone1:int = 32,
        maxRankDone2:int = 8,
        maxRankDec1: int = 64,
        maxRankDec2: int = 64,
        maxRankGruIH: int = 64,
        maxRankGruHH: int = 64,):
        self._maxRank = dict(
            obs1 = int(maxRankObs1),
            obs2 = int(maxRankObs2),
            act = int(maxRankAct),
            prior = int(maxRankPrior),
            post = int(maxRankPost),
            state = int(maxRankState),
            rew1 = int(maxRankRew1),
            rew2 = int(maxRankRew2),
            done1 = int(maxRankDone1),
            done2 = int(maxRankDone2),
            dec1 = int(maxRankDec1),
            dec2 = int(maxRankDec2),
            gru_ih = int(maxRankGruIH),
            gru_hh = int(maxRankGruHH),)
        super().__init__(base, initRankEach=initRankEach, autoRank=autoRank, evThreshold=evThreshold, gradEma=gradEma)

    def BuildSiteSpecs(self) -> Dict[str, "SiteSpec"]:
        wm = self.base
        V, S, Z, D = wm.vision_dim, wm.state_dim, wm.stoch_dim, wm.deter_dim
        nL = 1

        def alloc_linear(addRank: int, device: torch.device, dtype: torch.dtype, inDim: int, outDim: int):
            A = nn.Parameter(torch.randn(addRank, inDim, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype))
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_linear(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return s * (b @ a)  # [out,in]

        specs: Dict[str, SiteSpec] = {}
        add = lambda name, inD, outD, maxRank: specs.setdefault(
            name,
            SiteSpec(
                name=name,
                nLayers=nL,
                inDim=inD,
                outDim=outD,
                maxRank=int(maxRank),
                allocFn=lambda r, dev, dt, _in=inD, _out=outD: alloc_linear(r, dev, dt, _in, _out),
                composeFn=compose_linear,))

        add("obs1", V, S, self._maxRank["obs1"]) # obs_enc[1]: vision -> state
        add("obs2", S, Z, self._maxRank["obs2"]) # obs_enc[4]: state  -> stoch
        add("act", wm.action_dim, Z, self._maxRank["act"]) # act_proj[0]: action -> stoch
        add("prior", D, 2*Z, self._maxRank["prior"]) # prior_net[0]: deter -> (mu,std)
        add("post", D+Z, 2*Z, self._maxRank["post"]) # post_net[0] : [deter, e] -> (mu,std)
        add("state", D+Z, S, self._maxRank["state"]) # state_proj[1]: [deter, z] -> state
        add("rew1", S, 256, self._maxRank["rew1"]) # rew_head[0]
        add("rew2", 256, 1, self._maxRank["rew2"]) # rew_head[2]
        add("done1", S, 256, self._maxRank["done1"]) # done_head[0]
        add("done2", 256, 1, self._maxRank["done2"]) # done_head[2]
        add("gru_ih", 2*Z, 3*D, self._maxRank["gru_ih"])
        add("gru_hh", D, 3*D, self._maxRank["gru_hh"])

        if wm.use_decoder:
            add("dec1", S, S, self._maxRank["dec1"])  # obs_dec[0]
            add("dec2", S, V, self._maxRank["dec2"]) 
        return specs

    def ForwardWithDeltas(
        self,
        x: Dict[str, torch.Tensor],
        keyPaddingMask: Optional[torch.Tensor],
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]],) -> Dict[str, torch.Tensor]:
        wm = self.base

        B = x["vision"].size(0)
        device = x["vision"].device

        mode = str(x.get("mode", "posterior")).lower() 
        sample = bool(x.get("sample", False))

        hPrev = x.get("hPrev", None)
        zPrev = x.get("zPrev", None)
        if (hPrev is None) or (zPrev is None):
            h0, z0 = wm.ExportState()
            if hPrev is None: hPrev = h0
            if zPrev is None: zPrev = z0
            if hPrev.size(0) != B: hPrev = torch.zeros(B, wm.deter_dim, device=device)
            if zPrev.size(0) != B: zPrev = torch.zeros(B, wm.stoch_dim, device=device)

        deltas = deltasPerLayer[0] if len(deltasPerLayer) > 0 else {}

        def eff_linear(lo: "GrowableLoRALinear", deltaExtra: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            W = lo.target.weight
            base_delta = lo.DeltaWeight()
            if base_delta is not None: W = W + base_delta
            if deltaExtra is not None: W = W + deltaExtra
            return W, lo.target.bias

        v = x["vision"]
        v1 = wm.obs_enc[0](v)
        W_obs1, b_obs1 = eff_linear(wm.obs_enc[1], deltas.get("obs1", None))
        h_obs = F.linear(v1, W_obs1, b_obs1)
        h_obs = wm.obs_enc[2](h_obs)
        h_obs = wm.obs_enc[3](h_obs)
        W_obs2, b_obs2 = eff_linear(wm.obs_enc[4], deltas.get("obs2", None))
        raw_e = F.linear(h_obs, W_obs2, b_obs2)

        a_enc = wm.action_encoder(x["keys"], x["mouse"])
        W_act, b_act = eff_linear(wm.act_proj[0], deltas.get("act", None))
        a_t = F.linear(a_enc, W_act, b_act)
        a_t = wm.act_proj[1](a_t)
        a_t = wm.act_proj[2](a_t)

        x_in = torch.cat([zPrev, a_t], dim=-1)
        W_ih = wm.gru.target.weight_ih
        W_hh = wm.gru.target.weight_hh
        if (d := wm.gru.DeltaIH()) is not None: W_ih = W_ih + d
        if (d := wm.gru.DeltaHH()) is not None: W_hh = W_hh + d
        if "gru_ih" in deltas and deltas["gru_ih"] is not None: W_ih = W_ih + deltas["gru_ih"]
        if "gru_hh" in deltas and deltas["gru_hh"] is not None: W_hh = W_hh + deltas["gru_hh"]
        b_ih = wm.gru.target.bias_ih
        b_hh = wm.gru.target.bias_hh

        gi = F.linear(x_in, W_ih, b_ih)
        gh = F.linear(hPrev, W_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, dim=1)
        h_r, h_i, h_n = gh.chunk(3, dim=1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        h_next = newgate + inputgate * (hPrev - newgate)

        W_prior, b_prior = eff_linear(wm.prior_net[0], deltas.get("prior", None))
        prior_out = F.linear(h_next, W_prior, b_prior)
        mu_p, logstd_p = prior_out.chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        e_in = raw_e
        if wm._use_memory:
            with torch.no_grad():
                mem_h = wm.MemRetrieve(raw_e)
            if mem_h is not None:
                mem_e = wm.mem_val_to_e(mem_h)
                gate_m = torch.sigmoid(wm.meta_gate_e(torch.cat([h_next, raw_e], dim=-1)))
                e_in = raw_e + gate_m * mem_e

        ns_logits = None
        ns_probs = None

        if mode == "posterior":
            if wm._meta_dim > 0:
                ctx = wm.meta_ctx.view(1, -1).expand(B, -1)
                de_meta = wm.meta_to_e(ctx)
                dmu_meta = wm.meta_to_mu(ctx)
                gate_e = torch.sigmoid(wm.meta_gate_e(torch.cat([h_next, e_in], dim=-1)))
                gate_mu = torch.sigmoid(wm.meta_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
                e_in = e_in + gate_e * de_meta
                mu_p = mu_p + gate_mu * dmu_meta

            if wm._ns_enabled:
                ns_logits = wm.ns_head_post(torch.cat([h_next, e_in], dim=-1)) 
                P_raw = torch.sigmoid(ns_logits)
                P_train = wm.NsProjectProbs(ns_logits) 
                Q, pen = wm.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
                conf = wm.NsConfidence(P_raw)

                base_gate = torch.sigmoid(wm.ns_gate_e(torch.cat([h_next, e_in], dim=-1)))
                gate_scale = (1.0 - 0.25 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
                gate = (base_gate * gate_scale).clamp(0.0, 1.0)

                de = wm.ns_to_delta_e(Q)
                e_t = e_in + gate * de

                ns_probs = P_train
            else:
                e_t = e_in

            W_post, b_post = eff_linear(wm.post_net[0], deltas.get("post", None))
            post_out = F.linear(torch.cat([h_next, e_t], dim=-1), W_post, b_post)
            mu_q, logstd_q = post_out.chunk(2, dim=-1)
            logstd_q = ClampLogStd(logstd_q)
            z_next = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q) if sample else mu_q

        else:
            if wm._ns_enabled and wm._ns_bias_prior:
                ns_logits_pr = wm.ns_head_prior(h_next)
                P_raw = torch.sigmoid(ns_logits_pr)
                Q, pen = wm.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
                conf = wm.NsConfidence(P_raw)

                base_gate = torch.sigmoid(wm.ns_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
                gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
                gate = (base_gate * gate_scale).clamp(0.0, 1.0)

                dmu = wm.ns_to_delta_mu(Q)
                mu_p = mu_p + gate * dmu

            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p) if sample else mu_p
            mu_q = None
            logstd_q = None

        hz = torch.cat([h_next, z_next], dim=-1)
        s_pre = wm.state_proj[0](hz)
        W_state, b_state = eff_linear(wm.state_proj[1], deltas.get("state", None))
        s_mid = F.linear(s_pre, W_state, b_state)
        s_next = wm.state_proj[2](s_mid)

        W_rw1, b_rw1 = eff_linear(wm.rew_head[0], deltas.get("rew1", None))
        r_mid = F.linear(s_next, W_rw1, b_rw1)
        r_mid = wm.rew_head[1](r_mid)
        W_rw2, b_rw2 = eff_linear(wm.rew_head[2], deltas.get("rew2", None))
        r_pred = F.linear(r_mid, W_rw2, b_rw2).squeeze(-1)

        W_dn1, b_dn1 = eff_linear(wm.done_head[0], deltas.get("done1", None))
        d_mid = F.linear(s_next, W_dn1, b_dn1)
        d_mid = wm.done_head[1](d_mid)
        W_dn2, b_dn2 = eff_linear(wm.done_head[2], deltas.get("done2", None))
        d_logit = F.linear(d_mid, W_dn2, b_dn2).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)

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

        if (mode == "posterior") and wm._ns_enabled:
            out["ns_logits"] = ns_logits
            out["ns_probs"]  = ns_probs

        if (mode == "posterior") and wm.use_decoder:
            W_dec1, b_dec1 = eff_linear(wm.obs_dec[0], deltas.get("dec1", None))
            dec_mid = F.linear(s_next, W_dec1, b_dec1)
            dec_mid = wm.obs_dec[1](dec_mid)
            W_dec2, b_dec2 = eff_linear(wm.obs_dec[2], deltas.get("dec2", None))
            recon = F.linear(dec_mid, W_dec2, b_dec2)
            out["recon"] = recon
            out["recon_target"] = v

        if mode == "posterior":
            wm._h = h_next.detach()
            wm._z = z_next.detach()
            if wm._use_memory:
                with torch.no_grad():
                    wm.MemAdd(raw_e.detach(), h_next.detach())

        return out
    
    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        wm = self.base
        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "obs1":
            wm.obs_enc[1].Grow(addRank=a.size(0), init=init, freezeOld=False); return True
        if site == "obs2":
            wm.obs_enc[4].Grow(addRank=a.size(0), init=init, freezeOld=False); return True
        if site == "act":
            wm.act_proj[0].Grow(addRank=a.size(0), init=init, freezeOld=False);   return True
        if site == "prior":
            wm.prior_net[0].Grow(addRank=a.size(0), init=init, freezeOld=False);  return True
        if site == "post":
            wm.post_net[0].Grow(addRank=a.size(0), init=init, freezeOld=False);   return True
        if site == "state":
            wm.state_proj[1].Grow(addRank=a.size(0), init=init, freezeOld=False); return True
        if site == "rew1":
            wm.rew_head[0].Grow(addRank=a.size(0), init=init, freezeOld=False);   return True
        if site == "rew2":
            wm.rew_head[2].Grow(addRank=a.size(0), init=init, freezeOld=False);   return True
        if site == "done1":
            wm.done_head[0].Grow(addRank=a.size(0), init=init, freezeOld=False);  return True
        if site == "done2":
            wm.done_head[2].Grow(addRank=a.size(0), init=init, freezeOld=False);  return True
        if site == "dec1":
            wm.obs_dec[0].Grow(addRank=a.size(0), init=init, freezeOld=False);  return True
        if site == "dec2":
            wm.obs_dec[2].Grow(addRank=a.size(0), init=init, freezeOld=False);  return True
        if site == "gru_ih":
            init_gru = {"A_ih": a.detach().clone(), "B_ih": b.detach().clone(), "scale_ih": float(scale)}
            wm.gru.Grow(addRank=a.size(0), init=init_gru, freezeOld=False, mode="ih");  return True
        if site == "gru_hh":
            init_gru = {"A_hh": a.detach().clone(), "B_hh": b.detach().clone(), "scale_hh": float(scale)}
            wm.gru.Grow(addRank=a.size(0), init=init_gru, freezeOld=False, mode="hh");  return True
        return False


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

        self.wm = RSSMWorldModel(visionDim=512, actionDim=128, deterDim=256, stochDim=32, stateDim=256, useDecoder=True, useMemory=False, nsEnabled=True, ).to(self.device)
        self.wm.ResetHidden(batchSize=4, device=self.device)

    def Batch(self, B, keyIdx=(17,)):
        vision = torch.randn(B, 512, device=self.device)
        keys = torch.zeros(B, 106, device=self.device)
        for k in keyIdx:
            keys[:, k] = 1.0
        mouse = torch.randn(B, 2, device=self.device)
        return vision, keys, mouse

    def MKX(self, vision, keys, mouse, mode="posterior", sample=False, hPrev=None, zPrev=None):
        x = {"vision": vision, "keys": keys, "mouse": mouse, "mode": mode, "sample": sample}
        if hPrev is not None: x["hPrev"] = hPrev
        if zPrev is not None: x["zPrev"] = zPrev
        return x

    def SeedNonzeroCandidates(self, wrapper, scale=1e-3):
        with torch.no_grad():
            for name, layer_list in wrapper.cand.items():
                for slot in layer_list: 
                    for Bp in slot["B"]:
                        if Bp.numel() > 0:
                            Bp.data.normal_(0.0, scale)
                    for sp in slot["s"]:
                        sp.data.fill_(1.0)

    def TestActionEncoder(self):
        try:
            enc = ActionEncoder(numDiscrete=106, contDim=2, outDim=128).to(self.device)
            B = 3
            keys = torch.zeros(B, 106, device=self.device)
            keys[:, 17] = 1.0; keys[:, 57] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            y1 = enc(keys, mouse)
            y2 = enc(keys, None)
            ok = (y1.shape == (B, 128)) and (y2.shape == (B, 128))
            print("ActionEncoder test " + ("passed." if ok else "failed."))
            return ok
        except Exception as e:
            print(f"ActionEncoder test FAILED: {type(e).__name__}: {e}")
            return False

    def TestRSSMStepPosterior(self):
        try:
            B = 4
            vision, keys, mouse = self.Batch(B, keyIdx=(17,57))
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
        except Exception as e:
            print(f"RSSM StepPosterior test FAILED: {type(e).__name__}: {e}")
            return False

    def TestRSSMStepPriorOnly(self):
        try:
            B = 4
            _, keys, mouse = self.Batch(B, keyIdx=(30,))
            a_enc = self.wm.action_encoder(keys, mouse)
            h0, z0 = self.wm.ExportState()
            h_before, z_before = h0.clone(), z0.clone()
            h1, z1, s1, r, d = self.wm.StepPriorOnly(h0, z0, a_enc, sample=False)
            ok_shapes = ( h1.shape == (B, self.wm.deter_dim) and z1.shape == (B, self.wm.stoch_dim) and s1.shape == (B, self.wm.state_dim) and r.shape == (B,) and d.shape == (B,))
            hin, zin = self.wm.ExportState()
            not_written = torch.allclose(hin, h_before) and torch.allclose(zin, z_before)
            ok = ok_shapes and not_written
            print("RSSM StepPriorOnly test " + ("passed." if ok else "failed."))
            return ok
        except Exception as e:
            print(f"RSSM StepPriorOnly test FAILED: {type(e).__name__}: {e}")
            return False

    def TestForwardTrainSeq(self):
        try:
            B = 2
            vision, keys, mouse = self.Batch(B, keyIdx=(17,))
            out = self.wm.ForwardTrainSeq(visionSeq=vision, keysVec=keys, mouseSeq=mouse, rewardSeq=None, doneSeq=None, alphaKl=0.8, freeNats=1.0, reconCoef=1.0, rewardCoef=1.0, doneCoef=1.0, )
            loss = out["loss"]
            if not torch.isfinite(loss):
                print("ForwardTrainSeq loss is not finite.")
                return False
            self.wm.zero_grad(set_to_none=True)
            loss.backward()
            print("RSSM ForwardTrainSeq test passed. loss =", float(loss.item()), " | distill=", float(out["loss_ns_distill"].item()))
            return True
        except Exception as e:
            print(f"ForwardTrainSeq test FAILED: {type(e).__name__}: {e}")
            return False

    def TestWMAdapterForPlanner(self):
        try:
            adapter = WMAdapterForPlanner(
                wm=self.wm, maxCode=self.max_code,
                baseCodes=self.base_codes, skillCodes=self.skill_codes, extraCodes=self.extra_codes,
                noSkillId=None, deterministicZ=True,).to(self.device)
            B = 3
            a_mouse = torch.randn(B, 2, device=self.device)
            a_base = (torch.rand(B, len(self.base_codes), device=self.device) > 0.5).float()
            a_extra = (torch.rand(B, len(self.extra_codes), device=self.device) > 0.5).float()
            a_skill = torch.randint(low=0, high=len(self.skill_codes), size=(B,), device=self.device)
            s1, r, d = adapter.Step(aMouse=a_mouse, aSkill=a_skill, aBase=a_base, aExtra=a_extra)
            ok_shapes = s1.shape == (B, self.wm.state_dim) and r.shape == (B,) and d.shape == (B,)
            print("WMAdapterForPlanner.Step test " + ("passed." if ok_shapes else "failed."))
            return ok_shapes
        except Exception as e:
            print(f"WMAdapterForPlanner.Step test FAILED: {type(e).__name__}: {e}")
            return False

    def TestWrapperAPIBasics(self):
        try:
            wrapper = WorldModelOnlineWrapper(self.wm, initRankEach=0, autoRank=False).to(self.device)
            wrapper.train()
            B = 4
            vision, keys, mouse = self.Batch(B, keyIdx=(17,57))
            x = self.MKX(vision, keys, mouse, mode="posterior", sample=False)
            out = wrapper(x)
            ok = ( ("h_next" in out) and ("z_next" in out) and ("s_next" in out) and ("r_pred" in out) and ("d_prob" in out) and out["s_next"].shape == (B, self.wm.state_dim))
            
            print("Wrapper API basics " + ("passed." if ok else "failed."))
            return ok
        except Exception as e:
            print(f"Wrapper API basics FAILED: {type(e).__name__}: {e}")
            return False

    def TestForwardWithDeltasInjection(self):
        try:
            wrapper = WorldModelOnlineWrapper(self.wm, initRankEach=0, autoRank=False).to(self.device)
            wrapper.eval()
            B = 3
            vision, keys, mouse = self.Batch(B, keyIdx=(31,))
            x = self.MKX(vision, keys, mouse, mode="posterior", sample=False)

            out0 = wrapper.ForwardWithDeltas(x, None, None, None, [{}])
            Z = self.wm.stoch_dim
            A = torch.randn(Z, self.wm.action_dim, device=self.device) * 1e-3
            out1 = wrapper.ForwardWithDeltas(x, None, None, None, [{"act": A}])

            diff = (out0["s_next"] - out1["s_next"]).abs().mean().item()
            ok = diff > 1e-7
            print(f"ForwardWithDeltas injection {'passed' if ok else 'failed'} (|Δ|={diff:.3e})")
            return ok
        except Exception as e:
            print(f"ForwardWithDeltas injection FAILED: {type(e).__name__}: {e}")
            return False

    def TestCommitOneGrowAndValueChange(self):
        try:
            wrapper = WorldModelOnlineWrapper(self.wm, initRankEach=0, autoRank=False).to(self.device)
            wrapper.train()
            lo = self.wm.act_proj[0]
            n0 = len(lo.A_list)

            r = 2
            A = torch.randn(r, self.wm.action_dim, device=self.device) * 1e-2
            Bm = torch.randn(self.wm.stoch_dim, r, device=self.device) * 1e-2
            s = 1.0
            ok_commit = wrapper.CommitOne("act", 0, A, Bm, s)

            n1 = len(lo.A_list)
            grew = ok_commit and (n1 == n0 + 1)

            Bsz = 4
            vision, keys, mouse = self.Batch(Bsz, keyIdx=(33,))
            x = self.MKX(vision, keys, mouse, mode="posterior", sample=False)

            with torch.no_grad():
                last_alpha = lo.alpha[-1].clone()
                lo.alpha[-1].zero_()
                out_before = wrapper(x)
                lo.alpha[-1].copy_(last_alpha)

            out_after = wrapper(x)
            change = (out_after["s_next"] - out_before["s_next"]).abs().mean().item()

            ok = grew and (change > 1e-7)
            print(f"CommitOne grow & effect {'passed' if ok else 'failed'} (rank {n0}->{n1}, |Δ|={change:.3e})")
            return ok
        except Exception as e:
            print(f"CommitOne grow & effect FAILED: {type(e).__name__}: {e}")
            return False

    def TestGradFlowCandidates(self):
        try:
            wrapper = WorldModelOnlineWrapper(self.wm, initRankEach=2, autoRank=False).to(self.device)
            wrapper.train()
            self.SeedNonzeroCandidates(wrapper, scale=1e-3)

            B = 6
            vision, keys, mouse = self.Batch(B, keyIdx=(45,))
            x = self.MKX(vision, keys, mouse, mode="posterior", sample=False)

            out = wrapper(x)
            loss = ( F.mse_loss(out["r_pred"], torch.zeros_like(out["r_pred"])) +
                0.5 * F.binary_cross_entropy(out["d_prob"], torch.zeros_like(out["d_prob"])) +
                0.1 * F.mse_loss(out["s_next"], torch.zeros_like(out["s_next"])))

            params = list(wrapper.CandParameters())
            if len(params) == 0:
                print("Grad flow (wrapper candidates): failed | no candidate params")
                return False
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()

            def has_grad(plist):
                for p in plist:
                    if (p.grad is not None) and torch.isfinite(p.grad).all() and (p.grad.abs().sum() > 0):
                        return True
                return False

            cand = wrapper.cand["act"][0]
            ok = has_grad(cand["A"]) and has_grad(cand["B"]) and has_grad(cand["s"])
            print(f"Grad flow (wrapper candidates): {'passed' if ok else 'failed'} | loss={float(loss.item()):.6f}")
            return ok
        except Exception as e:
            print(f"Grad flow (wrapper candidates) FAILED: {type(e).__name__}: {e}")
            return False

    def TestWrapperTrainLossDecreases(self):
        try:
            wrapper = WorldModelOnlineWrapper(self.wm, initRankEach=2, autoRank=False).to(self.device)
            wrapper.train()
            self.SeedNonzeroCandidates(wrapper, scale=1e-3)

            opt = torch.optim.Adam(list(wrapper.CandParameters()), lr=2e-3)

            B = 8
            vision, keys, mouse = self.Batch(B, keyIdx=(31,49))
            x = self.MKX(vision, keys, mouse, mode="posterior", sample=False)

            watch = None
            for p in wrapper.CandParameters():
                watch = p
                break
            before = watch.detach().clone() if watch is not None else None

            steps = 40
            losses = []
            for t in range(1, steps + 1):
                out = wrapper(x)
                loss = (
                    F.mse_loss(out["r_pred"], torch.zeros_like(out["r_pred"])) +
                    0.5 * F.binary_cross_entropy(out["d_prob"], torch.zeros_like(out["d_prob"])) +
                    0.1 * F.mse_loss(out["s_next"], torch.zeros_like(out["s_next"])))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(wrapper.CandParameters()), 3.0)
                opt.step()
                losses.append(float(loss.item()))
                if t % 10 == 0:
                    print(f"[WrapperTrain] step {t}/{steps} | loss={losses[-1]:.6f}")

            start, end = losses[0], losses[-1]
            decreased = end <= start * 0.75 or (start - end) >= 0.3
            changed = True
            if before is not None:
                delta = (watch.detach() - before).abs().mean().item()
                changed = delta > 1e-7

            ok = decreased and changed
            print(f"Wrapper multi-step training {'passed' if ok else 'failed'} (loss {start:.4f} -> {end:.4f}; params changed={changed})")
            return ok
        except Exception as e:
            print(f"Wrapper multi-step training FAILED: {type(e).__name__}: {e}")
            return False

    def TestBaseGradFlow(self):
        try:
            base = RSSMWorldModel(visionDim=512, actionDim=128, deterDim=256, stochDim=32, stateDim=256, useDecoder=True, useMemory=False, nsEnabled=True).to(self.device)
            base.train()
            B = 6
            vision, keys, mouse = self.Batch(B, keyIdx=(32,))
            out = base.ForwardTrainSeq(vision, keys, mouse)
            loss = out["loss"]
            base.zero_grad(set_to_none=True)
            loss.backward()

            sample_params = [ base.obs_enc[1].target.weight, base.gru.target.weight_ih, base.rew_head[0].target.weight, base.done_head[2].target.weight, ]
            grads_ok = True
            for p in sample_params:
                g = p.grad
                if (g is None) or (not torch.isfinite(g).all()) or (g.abs().sum() == 0):
                    grads_ok = False
                    break
            print(f"Base grad flow: {'passed' if grads_ok else 'failed'} | loss={float(loss.item()):.6f}")
            return grads_ok
        except Exception as e:
            print(f"Base grad flow FAILED: {type(e).__name__}: {e}")
            return False

    def TestBaseTrainingLossDecreases(self):
        try:
            base = RSSMWorldModel(visionDim=512, actionDim=128, deterDim=256, stochDim=32, stateDim=256, useDecoder=True, useMemory=False, nsEnabled=True).to(self.device)
            base.train()
            opt = torch.optim.Adam(base.parameters(), lr=1e-3)

            B = 8
            vision, keys, mouse = self.Batch(B, keyIdx=(32,))

            watch_params = [base.obs_enc[1].target.weight, base.gru.target.weight_ih]
            before_vals = [p.detach().clone() for p in watch_params]

            losses = []
            steps = 30
            for t in range(1, steps + 1):
                out = base.ForwardTrainSeq(vision, keys, mouse)
                loss = out["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(base.parameters(), 3.0)
                opt.step()
                losses.append(float(loss.item()))
                if t % 10 == 0:
                    print(f"[BaseTrain] step {t}/{steps} | loss={losses[-1]:.6f}")

            start, end = losses[0], losses[-1]
            decreased = end < start
            changed = True
            for p, b in zip(watch_params, before_vals):
                if (p.detach() - b).abs().mean().item() <= 1e-7:
                    changed = False
                    break

            ok = decreased and changed
            print(f"Base ForwardTrainSeq training {'passed' if ok else 'failed'} (loss {start:.4f} -> {end:.4f}; params changed={changed})")
            return ok
        except Exception as e:
            print(f"Base ForwardTrainSeq training FAILED: {type(e).__name__}: {e}")
            return False

    def TestGradCoverageCandidateSites(self):
        try:
            wrapper = WorldModelOnlineWrapper(self.wm, initRankEach=2, autoRank=False).to(self.device)
            wrapper.train()
            self.SeedNonzeroCandidates(wrapper, scale=1e-3)

            B = 5
            vision, keys, mouse = self.Batch(B, keyIdx=(31,))

            x1 = self.MKX(vision, keys, mouse, mode="posterior", sample=False)
            out1 = wrapper(x1)
            mu_q_star = out1["mu_q"].detach()
            logstd_q_star = out1["logstd_q"].detach()

            x2 = self.MKX(vision, keys, mouse, mode="posterior", sample=False)
            out2 = wrapper(x2)

            base_loss = (
                F.mse_loss(out2["r_pred"], torch.zeros_like(out2["r_pred"])) +
                0.5 * F.binary_cross_entropy(out2["d_prob"], torch.zeros_like(out2["d_prob"])) +
                0.1 * F.mse_loss(out2["s_next"], torch.zeros_like(out2["s_next"])))   

            kl_prior = KLDiagNormal( mu_q_star, logstd_q_star, out2["mu_p"], out2["logstd_p"]).mean()

            loss = base_loss + 0.1 * kl_prior 

            for p in wrapper.CandParameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()

            missing = []
            def site_ok(name: str) -> bool:
                slot = wrapper.cand[name][0]
                def hasg(lst):
                    return any( (p.grad is not None) and torch.isfinite(p.grad).all() and (p.grad.abs().sum() > 0) for p in lst)
                return hasg(slot["A"]) and hasg(slot["B"]) and hasg(slot["s"])

            for name in ["obs1","obs2","act","prior","post","state","rew1","rew2","done1","done2"]:
                if not site_ok(name):
                    missing.append(name)

            ok_all = (len(missing) == 0)
            print(f"Grad coverage (candidates) {'passed' if ok_all else 'failed'}; missing/no-grad: {missing}")
            return ok_all

        except Exception as e:
            print(f"Grad coverage (candidates) FAILED: {type(e).__name__}: {e}")
            return False
        
    def TestParityPosterior(self):
        try:
            torch.manual_seed(0)
            wm = self.wm
            wrapper = WorldModelOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device)
            wrapper.eval(); wm.eval()

            B = 3
            vision, keys, mouse = self.Batch(B, keyIdx=(31,57))

            h0 = torch.zeros(B, wm.deter_dim, device=self.device)
            z0 = torch.zeros(B, wm.stoch_dim, device=self.device)

            a_enc = wm.action_encoder(keys, mouse)
            out_base = wm.StepPosterior(h0, z0, vision, a_enc, sample=False)

            x = self.MKX(vision, keys, mouse, mode="posterior", sample=False, hPrev=h0, zPrev=z0)
            out_wrap = wrapper.ForwardWithDeltas(x, None, None, None, [{}])

            def close(a,b): return torch.allclose(a, b, atol=1e-6, rtol=1e-5)

            ok = (close(out_base["h_next"], out_wrap["h_next"])
                and close(out_base["z_next"], out_wrap["z_next"])
                and close(out_base["s_next"], out_wrap["s_next"])
                and close(out_base["r_pred"], out_wrap["r_pred"])
                and close(out_base["d_prob"], out_wrap["d_prob"]) )
            print("Parity posterior " + ("passed." if ok else "failed."))
            return ok
        except Exception as e:
            print(f"Parity posterior FAILED: {type(e).__name__}: {e}")
            return False

    def TestParityPrior(self):
        try:
            torch.manual_seed(0)
            wm = self.wm
            wrapper = WorldModelOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device)
            wrapper.eval(); wm.eval()

            B = 3
            _, keys, mouse = self.Batch(B, keyIdx=(33,))
            
            h0 = torch.zeros(B, wm.deter_dim, device=self.device)
            z0 = torch.zeros(B, wm.stoch_dim, device=self.device)

            a_enc = wm.action_encoder(keys, mouse)
            h1, z1, s1, r1, d1 = wm.StepPriorOnly(h0, z0, a_enc, sample=False)

            x = self.MKX(torch.randn(B, 512, device=self.device), keys, mouse, mode="prior", sample=False, hPrev=h0, zPrev=z0)
            out_wrap = wrapper.ForwardWithDeltas(x, None, None, None, [{}])

            def close(a,b): return torch.allclose(a, b, atol=1e-6, rtol=1e-5)

            ok = (close(h1, out_wrap["h_next"])
                and close(z1, out_wrap["z_next"])
                and close(s1, out_wrap["s_next"])
                and close(r1, out_wrap["r_pred"])
                and close(d1, out_wrap["d_prob"]) )
            print("Parity prior " + ("passed." if ok else "failed."))
            return ok
        except Exception as e:
            print(f"Parity prior FAILED: {type(e).__name__}: {e}")
            return False

    def RunAll(self):
        results = {
            "ActionEncoder": self.TestActionEncoder(),
            "RSSMStepPosterior": self.TestRSSMStepPosterior(),
            "RSSMStepPriorOnly": self.TestRSSMStepPriorOnly(),
            "ForwardTrainSeq": self.TestForwardTrainSeq(),
            "WMAdapterForPlanner": self.TestWMAdapterForPlanner(),
            "WrapperAPIBasics": self.TestWrapperAPIBasics(),
            "ForwardWithDeltasInjection": self.TestForwardWithDeltasInjection(),
            "WrapperManualGrowTrainAndCommit": self.TestCommitOneGrowAndValueChange(),
            "GradFlowCandidates": self.TestGradFlowCandidates(),
            "WrapperTrainLossDecreases": self.TestWrapperTrainLossDecreases(),
            "BaseGradFlow": self.TestBaseGradFlow(),  
            "BaseTrainLossDecreases": self.TestBaseTrainingLossDecreases(), 
            "GradCoverageCandidateSites": self.TestGradCoverageCandidateSites(),
            "ParityPosterior": self.TestParityPosterior(),
            "ParityPrior": self.TestParityPrior()}

        passed = sum(1 for ok in results.values() if ok)
        total = len(results)
        print(f"\nWorldModel test summary: {passed}/{total} passed.")
        return all(results.values())

