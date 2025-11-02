from __future__ import annotations
from typing import Optional, Dict, NamedTuple, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics as stats
from FunctionTools import SiteSpec, BaseOnlineWrapper


class GrowableLoRALinear(nn.Module):
    def __init__(self, targetLinear: nn.Linear):
        super().__init__()
        assert isinstance(targetLinear, nn.Linear)
        self.target = targetLinear
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()
        self.out_f = int(targetLinear.out_features)
        self.in_f = int(targetLinear.in_features)

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if (addRank is None) or (addRank <= 0): return
        if init is None: init = {}
        dev = self.target.weight.device; dt = self.target.weight.dtype
        A = init.get("A", torch.randn(addRank, self.in_f,  device=dev, dtype=dt) * 1e-4)
        B = init.get("B", torch.randn(self.out_f, addRank, device=dev, dtype=dt) * 1e-4)
        s = init.get("scale", 1e-3)
        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)
        self.A_list.append(A); self.B_list.append(B); self.alpha.append(s)

    def DeltaWeight(self) -> Optional[torch.Tensor]:
        if len(self.A_list) == 0: return None
        delta = self.target.weight.new_zeros(self.out_f, self.in_f)
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            delta = delta + s * (B @ A)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.target.weight
        delta = self.DeltaWeight()
        if delta is not None: w = w + delta
        return F.linear(x, w, self.target.bias)


class HebbianLinearFW(nn.Module):
    def __init__(self, inFeatures: int, outFeatures: int, bias: bool = True,*,initEta: float = 1e-3, initLambda: float = 0.1, cap: float = 1.0, useOja: bool = True, detachHebb: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outFeatures, inFeatures))
        self.bias = nn.Parameter(torch.zeros(outFeatures)) if bias else None
        nn.init.orthogonal_(self.weight); 
        if self.bias is not None: nn.init.zeros_(self.bias)
        self.register_buffer("H", torch.zeros(outFeatures, inFeatures))
        self.init_eta = initEta 
        self.init_lambda = initLambda
        self.cap = cap
        self.use_oja = useOja
        self.detach_hebb = detachHebb

    @torch.no_grad() 
    def ResetHebbianMemory(self): self.H.zero_()

    @torch.no_grad()
    def ProjectCap(self):
        if self.cap is None: return
        n = self.H.norm(dim=1, keepdim=True)
        scale = (self.cap / (n + 1e-12)).clamp_max(1.0)
        self.H.mul_(scale)

    def forward(self, x: torch.Tensor, *,
                eta: Optional[torch.Tensor] = None,
                lam: Optional[torch.Tensor] = None,
                betaMix: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        scale = 0.0 if betaMix is None else betaMix.detach().mean()
        W_eff = self.weight + scale * self.H
        y = F.linear(x, W_eff, self.bias)
        with torch.no_grad():
            pre, post = x, y
            if self.detach_hebb: pre = pre.detach(); post = post.detach()
            pre_n = pre / (pre.norm(dim=-1, keepdim=True) + 1e-6)
            post_n = post / (post.norm(dim=-1, keepdim=True) + 1e-6)
            dH = torch.einsum('bo,bi->oi', post_n, pre_n) / max(1, x.size(0))
            if self.use_oja:
                post_sq = (post_n**2).mean(dim=0)
                dH = dH - torch.einsum('oi,o->oi', self.H, post_sq)
            _eta = float(self.init_eta) if eta is None else float(eta.mean().item())
            _lam = float(self.init_lambda) if lam is None else float(lam.mean().item())
            self.H.mul_(1.0 - _lam).add_(_eta * dH)
            self.ProjectCap()
        extras = {"H_norm": self.H.norm().detach()}
        return y, extras


class RunningEMA(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum; self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))

    @torch.no_grad()
    def Update(self, x: torch.Tensor):
        if not self.training: return
        if x.dim() == 1: x = x.unsqueeze(-1)
        mask = torch.isfinite(x).all(dim=-1)
        if mask.any():
            x = x[mask]; m = x.mean(0); v = x.var(0, unbiased=False)
            self.mean.copy_(self.mean * self.momentum + (1 - self.momentum) * m)
            self.var.copy_( self.var * self.momentum + (1 - self.momentum) * v)

    def Norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std = (self.var.to(x.device) + self.eps).sqrt()
        return (x - mean) / std.clamp_min(self.eps)


class IntrinsicRewardOut(NamedTuple):
    rInt: torch.Tensor
    components: Dict[str, torch.Tensor]
    eT: torch.Tensor

class IntrinsicRewardGenerator(nn.Module):
    def __init__(self, memoryDim: int = 768, attnDim: int = 1024, stateDim: int = 256, *,
                 hidden: int = 256, alphaNovelty: float = 1.0, alphaEntropy: float = 0.2,
                 alphaProgress: float = 0.5, alphaUncertPenalty: float = 0.5,
                 rClip: float = 5.0, tau0: float = 1.0, beta: float = 1.0,
                 lr0: float = 1.0,  kappa: float = 0.5, gamma0: float = 0.99, delta: float = 0.02,
                 tauMin: Optional[float] = None, tauMax: Optional[float] = 10.0,
                 lrMin: Optional[float]  = 0.25, lrMax: Optional[float]  = 3.0,
                 gammaMin: float = 0.90, gammaMax: float = 0.9999,
                 emaMomentum: float = 0.99, teacherDropoutProb: float = 0.1,
                 gateReg: float = 1e-3, eTAnchor: float = 1e-3):
        super().__init__()
        self.alpha_novelty = alphaNovelty
        self.alpha_entropy = alphaEntropy
        self.alpha_progress = alphaProgress
        self.alpha_uncert_penalty = alphaUncertPenalty
        self.r_clip = rClip
        self.tau0, self.beta = tau0, beta
        self.lr0, self.kappa = lr0, kappa
        self.gamma0, self.delta = gamma0, delta
        self.tau_min, self.tau_max = tauMin, tauMax
        self.lr_min, self.lr_max = lrMin, lrMax
        self.gamma_min, self.gamma_max = gammaMin, gammaMax

        self.nov_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.unc_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.prog_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.ent_ema = RunningEMA(dim=1, momentum=emaMomentum)

        in_dim = memoryDim + attnDim + stateDim

        self.affect_net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.ReLU(),
            nn.Linear(hidden, hidden), 
            nn.ReLU(),)
        
        self.progress_head = nn.Linear(hidden, 1)

        mid = max(32, hidden // 2)

        self.entropy_from_h = nn.Sequential(
            nn.Linear(hidden, mid), 
            nn.ReLU(), 
            nn.Linear(mid, 1), 
            nn.Softplus())
        
        self.uncert_from_h = nn.Sequential(
            nn.Linear(hidden, mid), 
            nn.ReLU(), 
            nn.Linear(mid, 1), 
            nn.Softplus())

        self.entropy_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

        self.uncert_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

        self.register_buffer("state_ema", torch.zeros(stateDim))
        self.state_momentum = emaMomentum
        self.teacher_dropout_prob = teacherDropoutProb
        self.gate_reg = gateReg
        self.eT_anchor= eTAnchor
        self._eps = 1e-6

        nn.init.zeros_(self.progress_head.bias)
        for m in self.affect_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

    @torch.no_grad()
    def UpdateStateEma(self, s: torch.Tensor):
        if not self.training: return
        mean_s = s.mean(0)
        self.state_ema.copy_(self.state_ema * self.state_momentum + (1 - self.state_momentum) * mean_s)

    def MaybeDropoutTeacher(self, t: Optional[torch.Tensor], B: int, device) -> Optional[torch.Tensor]:
        if (t is None) or (not self.training) or (self.teacher_dropout_prob <= 0): return t
        keep = torch.rand(B, device=device) > self.teacher_dropout_prob
        if keep.all(): return t
        t = t.clone(); t[~keep] = float('nan')
        return t

    def forward(self,
                memoryPrev: torch.Tensor, 
                attnPrev: torch.Tensor, 
                stateCurr: torch.Tensor, 
                *,
                policyEntropyPrev: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None,
                tdErrorPrev: Optional[torch.Tensor] = None) -> IntrinsicRewardOut:
        
        B, device = stateCurr.size(0), stateCurr.device
        self.UpdateStateEma(stateCurr)
        novelty = (stateCurr - self.state_ema.to(device)).pow(2).mean(-1).sqrt()
        h = self.affect_net(torch.cat([memoryPrev, attnPrev, stateCurr], dim=-1))

        if tdErrorPrev is not None: progress = -tdErrorPrev.abs()
        else: progress = torch.tanh(self.progress_head(h).squeeze(-1))

        policyEntropyPrev = self.MaybeDropoutTeacher(policyEntropyPrev, B, device)
        uncertainty = self.MaybeDropoutTeacher(uncertainty, B, device)

        entropy_pred = self.entropy_from_h(h).squeeze(-1)
        uncert_pred = self.uncert_from_h(h).squeeze(-1)
        g_e = self.entropy_gate(h).squeeze(-1)
        g_u = self.uncert_gate(h).squeeze(-1)

        if policyEntropyPrev is not None:
            pe = torch.where(torch.isfinite(policyEntropyPrev), policyEntropyPrev, entropy_pred.detach())
            fused_entropy = (1.0 - g_e) * pe + g_e * entropy_pred
        else:
            fused_entropy = entropy_pred

        if uncertainty is not None:
            uu = torch.where(torch.isfinite(uncertainty), uncertainty, uncert_pred.detach())
            fused_uncert = (1.0 - g_u) * uu + g_u * uncert_pred
        else:
            fused_uncert = uncert_pred

        self.nov_ema.Update(novelty.unsqueeze(-1))
        self.prog_ema.Update(progress.unsqueeze(-1))
        self.ent_ema.Update(fused_entropy.unsqueeze(-1))
        self.unc_ema.Update(fused_uncert.unsqueeze(-1))

        novelty_n = self.nov_ema.Norm(novelty).clamp_(-8.0, 8.0)
        progress_n = self.prog_ema.Norm(progress).clamp_(-8.0, 8.0)
        entropy_n = self.ent_ema.Norm(fused_entropy).clamp_(-8.0, 8.0)
        uncert_n = self.unc_ema.Norm(fused_uncert).clamp_(-8.0, 8.0)

        r_int = ( self.alpha_novelty * novelty_n
                + self.alpha_progress * progress_n
                + self.alpha_entropy * entropy_n
                - self.alpha_uncert_penalty * uncert_n ).clamp(-self.r_clip, self.r_clip)

        exp_arg = (self.beta * uncert_n).clamp(-15.0, 15.0)

        temp_scale= self.tau0 * torch.exp(exp_arg)

        if (self.tau_min is not None) or (self.tau_max is not None):
            lo = self.tau_min if self.tau_min is not None else -float('inf')
            hi = self.tau_max if self.tau_max is not None else float('inf')
            temp_scale = temp_scale.clamp(lo, hi)

        lr_scale = self.lr0 * (1.0 + self.kappa * novelty_n.clamp_min(0.0))
        if (self.lr_min is not None) or (self.lr_max is not None):
            lo = self.lr_min if self.lr_min is not None else -float("inf")
            hi = self.lr_max if self.lr_max is not None else  float("inf")
            lr_scale = lr_scale.clamp(lo, hi)

        valence = torch.tanh(progress_n)
        gamma_mod = (self.gamma0 + self.delta * valence).clamp(self.gamma_min, self.gamma_max)
        e_t = torch.stack([temp_scale, lr_scale, gamma_mod], dim=-1)

        comps: Dict[str, torch.Tensor] = {
            "novelty": novelty, "progress": progress,
            "entropy": fused_entropy, "uncertainty": fused_uncert,
            "entropy_pred": entropy_pred, "uncert_pred": uncert_pred,
            "entropy_gate": g_e, "uncert_gate": g_u,
            "novelty_n": novelty_n, "progress_n": progress_n,
            "entropy_n": entropy_n, "uncertainty_n": uncert_n,
            "valence": valence,}
        
        if self.training:
            comps["reg_gate"] = self.gate_reg * ((g_e - 0.5).pow(2) + (g_u - 0.5).pow(2))
            comps["reg_eT"] = self.eT_anchor * ((temp_scale - self.tau0).pow(2) + (lr_scale - self.lr0).pow(2) + (gamma_mod - self.gamma0).pow(2))

        return IntrinsicRewardOut(rInt=r_int, components=comps, eT=e_t)



class MaxPlusLinear(nn.Module):
    def __init__(self, inFeatures: int, outFeatures: int, useSoft: bool = True, temperature: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(outFeatures, inFeatures))
        self.b = nn.Parameter(torch.zeros(outFeatures))
        self.use_soft = useSoft; self.temperature = temperature
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5)); nn.init.zeros_(self.b)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.W.unsqueeze(0) + x.unsqueeze(1)
        if self.use_soft:
            t = max(self.temperature, 1e-6)
            z = score / t
            m = z.amax(dim=-1, keepdim=True)
            y = t * (m.squeeze(-1) + torch.logsumexp(z - m, dim=-1))
        else:
            y,_ = score.max(dim=-1)
        return y + self.b


class TropicalAffineTransport(nn.Module):
    def __init__(self, hDim: int, useSoftTrop: bool = True, temp: float = 0.2, epsA: float = 1e-3):
        super().__init__()
        self.trop = MaxPlusLinear(inFeatures=hDim+1, outFeatures=1, useSoft=useSoftTrop, temperature=temp)

        self.a_net = nn.Sequential(nn.Linear(hDim, hDim//2), nn.ReLU(), nn.Linear(hDim//2, 1))
        self.b_net = nn.Sequential(nn.Linear(hDim, hDim//2), nn.ReLU(), nn.Linear(hDim//2, 1))
        self.g_net = nn.Sequential(nn.Linear(hDim, hDim//2), nn.ReLU(), nn.Linear(hDim//2, 1), nn.Sigmoid())

        self.epsA = epsA
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v_in = v.unsqueeze(-1)
        a = F.softplus(self.a_net(h)) + self.epsA
        b = self.b_net(h)
        g = self.g_net(h)
        trop_in  = torch.cat([h, v_in], dim=-1)
        trop_out = self.trop(trop_in).squeeze(-1)
        aff_out  = (a.squeeze(-1) * v) + b.squeeze(-1)
        v_next_hat = g.squeeze(-1) * trop_out + (1.0 - g.squeeze(-1)) * aff_out
        extras = {"gate_trop": g.squeeze(-1), "a": a.squeeze(-1), "b": b.squeeze(-1),"trop_out": trop_out, "aff_out": aff_out}

        return v_next_hat, extras



class GITGaugeRegularizer(nn.Module):
    def __init__(self, wScale: float = 1e-3, wShift: float = 1e-3, wSign: float = 1e-3):
        super().__init__()
        self.w_scale = wScale; self.w_shift = wShift; self.w_sign = wSign

    def forward(self, valueHead: nn.Linear, transpExtras: Dict[str, torch.Tensor], adapter: Optional[nn.Module] = None) -> torch.Tensor:
        W = valueHead.weight
        if (adapter is not None) and hasattr(adapter, "DeltaWeight"):
            dW = adapter.DeltaWeight()
            if dW is not None:
                W = W + dW
        reg = W.new_zeros(())
        fro = torch.linalg.matrix_norm(W, ord='fro')
        reg = reg + self.w_scale * (fro - 1.0).pow(2)
        if "b" in transpExtras:
            reg = reg + self.w_shift * (transpExtras["b"].mean()).pow(2)
        with torch.no_grad():
            idx = torch.argmax(W.abs()).item() if W.numel() > 0 else 0
        row = idx // W.size(1) if W.numel() > 0 else 0
        if W.numel() > 0:
            wrow = W[row]
            reg = reg + self.w_sign * F.relu(-wrow.sum())
        return reg



class TemporalMicroGraph(nn.Module):
    def __init__(self, embDim: int, maxAnchors: int = 256, topk: int = 4, distTau: float = 0.5, lenPower: float = 0.5, eps: float = 1e-8):
        super().__init__()
        self.emb_dim = embDim
        self.max_anchors = maxAnchors
        self.topk = topk
        self.dist_tau = distTau
        self.len_power = lenPower
        self.eps = eps
        self.register_buffer("prefix_G", torch.ones(1))
        self.register_buffer("prefix_C", torch.zeros(1))
        self.anchors: List[Dict[str, torch.Tensor]] = []
        self._step = 0

    @torch.no_grad()
    def Reset(self, device=None):
            dev = device if device is not None else self.prefix_G.device
            dt = self.prefix_G.dtype
            self.prefix_G.data = torch.ones(1, device=dev, dtype=dt)
            self.prefix_C.data = torch.zeros(1, device=dev, dtype=dt)
            self.anchors.clear()
            self._step = 0

    def Stack(self, key: str) -> Optional[torch.Tensor]:
        if len(self.anchors) == 0: return None
        return torch.stack([a[key] for a in self.anchors], dim=0)

    @torch.no_grad()
    def PreviewEdges(self, zNow: torch.Tensor, rNow: torch.Tensor, gNow: torch.Tensor, topk: Optional[int] = None) -> Dict[str, torch.Tensor]:

        device = zNow.device

        G_new = self.prefix_G.to(device) * gNow
        C_new = self.prefix_C.to(device) + self.prefix_G.to(device) * rNow

        Z = self.Stack("z")
        if Z is None or Z.numel() == 0:
            return {"idx": torch.empty(0, dtype=torch.long, device=device),
                    "R": torch.empty(0, device=device),
                    "Gamma": torch.empty(0, device=device),
                    "v_hist": torch.empty(0, device=device),
                    "w": torch.empty(0, device=device),
                    "dist": torch.empty(0, device=device)}
        
        Z = Z.to(device)

        d = torch.cdist(zNow.unsqueeze(0), Z)[0] 
        k = min(int(topk or self.topk), int(d.numel()))

        if k <= 0:
            return {"idx": torch.empty(0, dtype=torch.long, device=device),
                    "R": torch.empty(0, device=device),
                    "Gamma": torch.empty(0, device=device),
                    "v_hist": torch.empty(0, device=device),
                    "w": torch.empty(0, device=device),
                    "dist": torch.empty(0, device=device)}

        vals, idx = torch.topk(-d, k=k) 
        idx = idx.to(device)
        d_sel = d[idx]

        G_i = self.Stack("G").to(device)[idx]
        C_i = self.Stack("C").to(device)[idx]
        v_i = self.Stack("v").to(device)[idx]

        Gamma_seg = (G_new / (G_i + self.eps)).squeeze(-1)
        R_seg = ((C_new - C_i) / (G_i + self.eps)).squeeze(-1)

        w = torch.exp(-d_sel / max(self.dist_tau, 1e-6)) * (Gamma_seg.clamp_min(1e-6) ** self.len_power)

        return {"idx": idx, "R": R_seg, "Gamma": Gamma_seg, "v_hist": v_i.squeeze(-1), "w": w, "dist": d_sel}

    @torch.no_grad()
    def CommitStep(self, zNow: torch.Tensor, vNow: torch.Tensor, rNow: torch.Tensor, gNow: torch.Tensor):

        dev, dt = zNow.device, zNow.dtype
        self.prefix_C.add_(self.prefix_G.to(dev, dt) * rNow.to(dev, dt))
        self.prefix_G.mul_(gNow.to(dev, dt))
        anchor = {
            "z": zNow.detach().to(dev, dt),
            "v": vNow.detach().to(dev, dt).unsqueeze(-1),
            "G": self.prefix_G.detach().clone(),
            "C": self.prefix_C.detach().clone(),}
        
        self.anchors.append(anchor)
        if len(self.anchors) > int(self.max_anchors):
            self.anchors.pop(0)
        self._step += 1



class GeoTropicalOut(NamedTuple):
    value: torch.Tensor
    tdError: torch.Tensor
    loss: torch.Tensor
    eT: torch.Tensor
    rInt: torch.Tensor
    rComps: Dict[str, torch.Tensor]
    uncertainty: torch.Tensor
    extras: Dict[str, torch.Tensor]



class ValueEstimationExtractor(nn.Module):
    def __init__(self,
                 memoryDim: int = 768, attnDim: int = 1024, stateDim: int = 256, *,
                 hidden: int = 2048, useLayerNorm: bool = True, irgKwargs: Optional[dict] = None,
                 wExt: float = 1.0, wInt: float = 1.0, stopGradRGamma: bool = True,
                 useSoftTrop: bool = True, tropTemp: float = 0.2, epsA: float = 1e-3,
                 wTD: float = 1.0, wCycle: float = 1e-2,  wGlue1: float = 1e-2, 
                 microMaxAnchors: int = 256, microTopK: int = 4, microDistTau: float = 0.5, microLenPower: float = 0.5,
                 wUncertTeacher: float = 1e-2, wEntropyTeacher: float = 1e-3,
                 wGITScale: float = 1e-3, wGITShift: float = 1e-3, wGITSign: float = 1e-3,
                 useHebb: bool = True, hebbCap: float = 1.0, hebbOja: bool = True, detachHebbGrad: bool = True,):
        super().__init__()

        if irgKwargs is None:
            irgKwargs = {"hidden": 1024}

        self.in_dim = memoryDim + attnDim + stateDim
        H = hidden

        self.use_hebb = useHebb
        self.wEntropyTeacher = wEntropyTeacher

        self.fc1 = nn.Linear(self.in_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.norm1 = nn.LayerNorm(H) if useLayerNorm else None
        self.norm2 = nn.LayerNorm(H) if useLayerNorm else None

        self.hebb_value = HebbianLinearFW(H, 1, bias=True, initEta=1e-3, initLambda=0.1, cap=hebbCap, useOja=hebbOja, detachHebb=detachHebbGrad)
        self.value_head = nn.Linear(H, 1)
        self.uncert_head = nn.Linear(H, 1)

        self.fc1_adapter = GrowableLoRALinear(self.fc1)
        self.fc2_adapter = GrowableLoRALinear(self.fc2)
        self.value_adapter = GrowableLoRALinear(self.value_head)
        self.uncert_adapter= GrowableLoRALinear(self.uncert_head)

        self.mix_gate = nn.Linear(H, 1)

        self.wMixGateReg = 1e-3

        self.transport = TropicalAffineTransport(H, useSoftTrop, tropTemp, epsA)
        self.rgen = IntrinsicRewardGenerator(memoryDim, attnDim, stateDim, **(irgKwargs or {}))
        self.git = GITGaugeRegularizer(wScale=wGITScale, wShift=wGITShift, wSign=wGITSign)

        self.micro = TemporalMicroGraph(embDim=H, maxAnchors=microMaxAnchors, topk=microTopK, distTau=microDistTau, lenPower=microLenPower)

        self.wExt, self.wInt = wExt, wInt
        self.wTD = wTD
        self.wCycle = wCycle
        self.wGlue1 = wGlue1
        self.wUncertTeacher = wUncertTeacher
        self.stopGrad_r_gamma = stopGradRGamma

        self._prev_vhat: Optional[torch.Tensor] = None  

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

        nn.init.zeros_(self.mix_gate.weight)
        nn.init.constant_(self.mix_gate.bias, -2.0)  

    def Trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1_adapter(x))
        h = self.norm1(h) if self.norm1 is not None else h
        h = F.gelu(self.fc2_adapter(h))
        h = self.norm2(h) if self.norm2 is not None else h
        return h

    def forward(self,
                memory: torch.Tensor, 
                attn: torch.Tensor, 
                state: torch.Tensor, 
                *,
                rewardExt: Optional[torch.Tensor] = None,
                policyEntropyPrev: Optional[torch.Tensor] = None,
                uncertaintyTeacher: Optional[torch.Tensor] = None,
                tdErrorPrev: Optional[torch.Tensor] = None,
                done: Optional[torch.Tensor] = None) -> GeoTropicalOut:

        B, device = state.size(0), state.device
        x = torch.cat([memory, attn, state], dim=-1)
        h = self.Trunk(x)

        uncert_raw = self.uncert_adapter(h).squeeze(-1)
        uncert_pred_fallback = F.softplus(uncert_raw).detach()

        irg_out = self.rgen(memoryPrev=memory, attnPrev=attn, stateCurr=state,
                            policyEntropyPrev=policyEntropyPrev,
                            uncertainty=(uncertaintyTeacher if uncertaintyTeacher is not None else uncert_pred_fallback),
                            tdErrorPrev=tdErrorPrev)
        
        r_int, eT, comps = irg_out.rInt.detach(), irg_out.eT, irg_out.components

        uncert_pred = F.softplus(uncert_raw)

        if self.use_hebb:
            hebb_eta = eT[..., 1].clamp_min(0).tanh() * 0.01
            hebb_lam = torch.full((B,), 0.1, device=device)

            mix = torch.sigmoid(self.mix_gate(h)).squeeze(-1) 
            mix = mix.clamp(1e-3, 1 - 1e-3)

            beta_mix = mix.detach() 
            v_hebb, hebb_extras = self.hebb_value(h, eta=hebb_eta, lam=hebb_lam, betaMix=beta_mix) 

            v_param = self.value_adapter(h).squeeze(-1) 

            value = (1.0 - mix) * v_param + mix * v_hebb.squeeze(-1)  
        else:
            value = self.value_adapter(h).squeeze(-1)
            hebb_extras = {"H_norm": torch.tensor(0.0, device=device)}
            mix = torch.zeros(B, device=device) 

        if rewardExt is None: r_used = self.wInt * r_int
        else: r_used = self.wExt * rewardExt.to(device) + self.wInt * r_int

        gamma = eT[..., 2]
        if self.stopGrad_r_gamma:
            r_used = r_used.detach()
            gamma = gamma.detach()
        if done is not None:
            gamma = gamma * (1.0 - done.float())

        v_next_hat, transp_extras = self.transport(h, value)
        delta = r_used + gamma * v_next_hat - value
        loss_td = (delta ** 2).mean() * self.wTD

        edges = self.micro.PreviewEdges(zNow=h.detach().mean(dim=0) if B>1 else h.detach().squeeze(0),
                                         rNow=r_used.mean().detach(),
                                         gNow=gamma.mean().detach())
        if edges["w"].numel() > 0:
            e_cycle = edges["R"] - (edges["v_hist"].detach() - edges["Gamma"] * value.mean())
            loss_cycle = ((edges["w"] * (e_cycle**2)).sum() / edges["w"].sum().clamp_min(1.0) ) * self.wCycle
        else:
            loss_cycle = value.new_zeros(())

        if self._prev_vhat is not None:
            loss_glue1 = F.mse_loss(self._prev_vhat.to(device), value.mean()) * self.wGlue1
        else:
            loss_glue1 = value.new_zeros(())

        loss_unc = value.new_zeros(())
        if (uncertaintyTeacher is not None) and (self.wUncertTeacher > 0):
            m = torch.isfinite(uncertaintyTeacher)
            loss_unc = self.wUncertTeacher * F.mse_loss(uncert_pred[m], uncertaintyTeacher[m]) if m.any() else 1e-4 * uncert_pred.mean()
        elif self.wUncertTeacher > 0:
            loss_unc = 1e-4 * uncert_pred.mean()

        loss_ent = torch.tensor(0.0, device=device)
        if (policyEntropyPrev is not None) and (self.wEntropyTeacher > 0):
            e_pred = irg_out.components.get("entropy_pred", None)
            if e_pred is not None:
                m = torch.isfinite(policyEntropyPrev)
                if m.any(): loss_ent = F.mse_loss(e_pred[m], policyEntropyPrev[m]) * self.wEntropyTeacher

        loss_git = self.git(self.value_head, transp_extras, adapter=self.value_adapter)

        loss_mixgate = torch.tensor(0.0, device=device)
        if hasattr(self, "wMixGateReg") and self.wMixGateReg > 0:
            loss_mixgate = self.wMixGateReg * ((mix - 0.5) ** 2).mean()

        total_loss = loss_td + loss_cycle + loss_glue1 + loss_unc + loss_git + comps.get("reg_gate", value.new_zeros(())).mean() + comps.get("reg_eT", value.new_zeros(())).mean() + loss_ent + loss_mixgate

        alive = (done is None) or (done.float().mean() < 0.5)

        if alive:
            self.micro.CommitStep(zNow=h.detach().mean(dim=0) if B>1 else h.detach().squeeze(0),vNow=value.mean().detach(),rNow=r_used.mean().detach(),gNow=gamma.mean().detach())
            
        self._prev_vhat = v_next_hat.mean().detach()

        extras: Dict[str, torch.Tensor] = {
            "loss_td": loss_td.detach(),
            "loss_cycle": loss_cycle.detach(),
            "loss_glue1": loss_glue1.detach(),
            "loss_unc": loss_unc.detach(),
            "loss_git": loss_git.detach(),
            "v_next_hat": v_next_hat.detach(),
            "gate_trop": transp_extras.get("gate_trop", torch.zeros_like(value)).detach(),
            "trop_out": transp_extras.get("trop_out", torch.zeros_like(value)).detach(),
            "aff_out": transp_extras.get("aff_out", torch.zeros_like(value)).detach(),
            "a": transp_extras.get("a", torch.zeros_like(value)).detach(),
            "b": transp_extras.get("b", torch.zeros_like(value)).detach(),
            "hebb_H_norm": hebb_extras.get("H_norm", torch.tensor(0.0, device=device)).detach(),
            "micro_edges": torch.tensor(float(edges["w"].numel()), device=device),
            "mix_mean": mix.mean().detach(),
            "mix_gt_half": (mix > 0.5).float().mean().detach(),}

        return GeoTropicalOut(
            value=value, tdError=delta, loss=total_loss, eT=eT, rInt=r_int,
            rComps={k: v.detach() for k, v in comps.items()},
            uncertainty=uncert_pred, extras=extras)

    @torch.no_grad()
    def ResetHebbianMemory(self): 
        self.hebb_value.ResetHebbianMemory()
        self._prev_vhat=None

    @torch.no_grad()
    def ResetMicroGraph(self): 
        self.micro.Reset(device=None)
        self._prev_vhat = None


class ValueEstimationOnlineWrapper(BaseOnlineWrapper):
    def __init__(self, 
                 base: nn.Module, 
                 initRankEach: int = 0, 
                 autoRank: bool = True,
                 evThreshold: float = 0.90, 
                 gradEma: float = 0.9,
                 maxRankFc1: int = 128, 
                 maxRankFc2: int = 128,
                 maxRankVHead: int = 64,  
                 maxRankUHead: int = 64,):
        self.maxRankFc1 = int(maxRankFc1)
        self.maxRankFc2 = int(maxRankFc2)
        self.maxRankVHead= int(maxRankVHead)
        self.maxRankUHead= int(maxRankUHead)
        super().__init__(base, initRankEach=initRankEach, autoRank=autoRank, evThreshold=evThreshold, gradEma=gradEma)

    @staticmethod
    def LinearWithDelta(layer: nn.Linear, 
                        x: torch.Tensor,
                        delta_mat: Optional[torch.Tensor] = None,
                        base_adapter: Optional[nn.Module] = None) -> torch.Tensor:
        W_eff = layer.weight
        if (base_adapter is not None) and hasattr(base_adapter, "DeltaWeight"):
            base_delta = base_adapter.DeltaWeight()
            if base_delta is not None:
                W_eff = W_eff + base_delta
        if delta_mat is not None:
            W_eff = W_eff + delta_mat
        return F.linear(x, W_eff, layer.bias)

    @staticmethod
    def EnsureInputs(x):
        if isinstance(x, (tuple, list)) and len(x) == 3:
            return x[0], x[1], x[2]
        if isinstance(x, dict) and all(k in x for k in ("memory","attn","state")):
            return x["memory"], x["attn"], x["state"]
        raise TypeError("ValueEstimationOnlineWrapper expects x as (memory, attn, state) or dict with those keys.")

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        H = int(self.base.value_head.in_features)
        Din = int(self.base.fc1.in_features)
        L = 2

        def alloc(addRank: int, inDim: int, outDim: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, inDim, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype))
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return s * (b @ a)

        return {
            "fc1": SiteSpec("fc1", L, Din, H, self.maxRankFc1, lambda r,dv,dt: alloc(r, Din, H, dv, dt), compose),
            "fc2": SiteSpec("fc2", L, H, H, self.maxRankFc2, lambda r,dv,dt: alloc(r, H, H, dv, dt), compose),
            "vhead": SiteSpec("vhead", L, H, 1, self.maxRankVHead, lambda r,dv,dt: alloc(r, H, 1, dv, dt), compose),
            "uhead": SiteSpec("uhead", L, H, 1, self.maxRankUHead, lambda r,dv,dt: alloc(r, H, 1, dv, dt), compose),}

    def ForwardWithDeltas(self, 
                          x, 
                          keyPaddingMask: Optional[torch.Tensor],
                          tdError: Optional[torch.Tensor],
                          uncertainty: Optional[torch.Tensor],
                          deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]], 
                          **kwargs):
        
        allowed = {"rewardExt", "policyEntropyPrev", "done"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f"Unknown kwargs in ForwardWithDeltas: {unknown}")

        rewardExt = kwargs.get("rewardExt", None)
        policyEntropyPrev = kwargs.get("policyEntropyPrev", None)
        done = kwargs.get("done", None)

        memory, attn, state = self.EnsureInputs(x)
        base = self.base
        B, device = state.size(0), state.device

        d_fc1 = deltasPerLayer[0].get("fc1", None)
        h = self.LinearWithDelta(base.fc1, torch.cat([memory, attn, state], dim=-1), d_fc1, getattr(base, "fc1_adapter", None))
        h = F.gelu(h)
        h = base.norm1(h) if base.norm1 is not None else h

        d_fc2 = deltasPerLayer[1].get("fc2", None)
        h = self.LinearWithDelta(base.fc2, h, d_fc2, getattr(base, "fc2_adapter", None))
        h = F.gelu(h)
        h = base.norm2(h) if base.norm2 is not None else h

        d_uh = deltasPerLayer[1].get("uhead", None)
        uncert_raw = self.LinearWithDelta(base.uncert_head, h, d_uh, getattr(base, "uncert_adapter", None)).squeeze(-1)
        uncert_pred = F.softplus(uncert_raw)
        uncert_pred_fallback = uncert_pred.detach()

        irg_out = base.rgen(memoryPrev=memory, attnPrev=attn, stateCurr=state,
                            policyEntropyPrev=policyEntropyPrev,
                            uncertainty=(uncertainty if uncertainty is not None else uncert_pred_fallback),
                            tdErrorPrev=tdError)
        
        r_int, eT, comps = irg_out.rInt.detach(), irg_out.eT, irg_out.components

        if base.use_hebb:
            hebb_eta = eT[..., 1].clamp_min(0).tanh() * 0.01
            hebb_lam = torch.full((B,), 0.1, device=device)

            mix = torch.sigmoid(base.mix_gate(h)).squeeze(-1).clamp(1e-3, 1 - 1e-3)

            beta_mix = mix.detach()
            v_hebb, hebb_extras = base.hebb_value(h, eta=hebb_eta, lam=hebb_lam, betaMix=beta_mix)

            d_vh = deltasPerLayer[1].get("vhead", None)
            v_param= self.LinearWithDelta(base.value_head, h, d_vh, getattr(base, "value_adapter", None)).squeeze(-1)

            value = (1.0 - mix) * v_param + mix * v_hebb.squeeze(-1)
        else:
            d_vh = deltasPerLayer[1].get("vhead", None)
            value = self.LinearWithDelta(base.value_head, h, d_vh, getattr(base, "value_adapter", None)).squeeze(-1)
            hebb_extras = {"H_norm": torch.tensor(0.0, device=device)}
            mix = torch.zeros(B, device=device)

        if rewardExt is None: r_used = base.wInt * r_int
        else: r_used = base.wExt * rewardExt.to(device) + base.wInt * r_int

        gamma = eT[..., 2]
        if base.stopGrad_r_gamma:
            r_used = r_used.detach()
            gamma  = gamma.detach()
        if done is not None:
            gamma = gamma * (1.0 - done.float())

        v_next_hat, transp_extras = base.transport(h, value)
        delta = r_used + gamma * v_next_hat - value
        loss_td = (delta ** 2).mean() * base.wTD

        edges = base.micro.PreviewEdges(
            zNow=h.detach().mean(dim=0) if B > 1 else h.detach().squeeze(0),
            rNow=r_used.mean().detach(),
            gNow=gamma.mean().detach())
        if edges["w"].numel() > 0:
            e_cycle = edges["R"] - (edges["v_hist"].detach() - edges["Gamma"] * value.mean())
            loss_cycle = ((edges["w"] * (e_cycle ** 2)).sum() / edges["w"].sum().clamp_min(1.0)) * base.wCycle
        else:
            loss_cycle = value.new_zeros(())

        if base._prev_vhat is not None:
            loss_glue1 = F.mse_loss(base._prev_vhat.to(device), value.mean()) * base.wGlue1
        else:
            loss_glue1 = value.new_zeros(())

        if (uncertainty is not None) and (base.wUncertTeacher > 0):
            m = torch.isfinite(uncertainty)
            loss_unc = base.wUncertTeacher * F.mse_loss(uncert_pred[m], uncertainty[m]) if m.any() else 1e-4 * uncert_pred.mean()
        elif base.wUncertTeacher > 0:
            loss_unc = 1e-4 * uncert_pred.mean()
        else:
            loss_unc = value.new_zeros(())

        loss_ent = torch.tensor(0.0, device=device)
        if (policyEntropyPrev is not None) and (base.wEntropyTeacher > 0):
            e_pred = irg_out.components.get("entropy_pred", None)
            if e_pred is not None:
                m = torch.isfinite(policyEntropyPrev)
                if m.any():
                    loss_ent = F.mse_loss(e_pred[m], policyEntropyPrev[m]) * base.wEntropyTeacher

        loss_git = base.git(base.value_head, transp_extras, adapter=getattr(base, "value_adapter", None))

        loss_mixgate = torch.tensor(0.0, device=device)
        if hasattr(base, "wMixGateReg") and base.wMixGateReg > 0:
            loss_mixgate = base.wMixGateReg * ((mix - 0.5) ** 2).mean()

        total_loss = (
            loss_td + loss_cycle + loss_glue1 + loss_unc + loss_git +
            comps.get("reg_gate", value.new_zeros(())).mean() +
            comps.get("reg_eT",   value.new_zeros(())).mean() +
            loss_ent + loss_mixgate)

        alive = (done is None) or (done.float().mean() < 0.5)
        if alive:
            base.micro.CommitStep(
                zNow=h.detach().mean(dim=0) if B > 1 else h.detach().squeeze(0),
                vNow=value.mean().detach(),
                rNow=r_used.mean().detach(),
                gNow=gamma.mean().detach())
            
        base._prev_vhat = v_next_hat.mean().detach()

        extras: Dict[str, torch.Tensor] = {
            "loss_td": loss_td.detach(),
            "loss_cycle": loss_cycle.detach(),
            "loss_glue1": loss_glue1.detach(),
            "loss_unc": loss_unc.detach(),
            "loss_git": loss_git.detach(),
            "v_next_hat": v_next_hat.detach(),
            "gate_trop": transp_extras.get("gate_trop", torch.zeros_like(value)).detach(),
            "trop_out": transp_extras.get("trop_out", torch.zeros_like(value)).detach(),
            "aff_out": transp_extras.get("aff_out", torch.zeros_like(value)).detach(),
            "a": transp_extras.get("a", torch.zeros_like(value)).detach(),
            "b": transp_extras.get("b", torch.zeros_like(value)).detach(),
            "hebb_H_norm": hebb_extras.get("H_norm", torch.tensor(0.0, device=device)).detach(),
            "micro_edges": torch.tensor(float(edges["w"].numel()), device=device),
            "mix_mean": mix.mean().detach(),
            "mix_gt_half": (mix > 0.5).float().mean().detach(),}

        return GeoTropicalOut(
            value=value, tdError=delta, loss=total_loss, eT=eT, rInt=r_int,
            rComps={k: v.detach() for k, v in comps.items()},
            uncertainty=uncert_pred, extras=extras)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        mapping = {
            "fc1": ("fc1_adapter", "fc1", [0]),
            "fc2": ("fc2_adapter", "fc2", [1]),
            "vhead": ("value_adapter", "value_head", [1]),
            "uhead": ("uncert_adapter","uncert_head", [1]),}
        
        if site not in mapping:
            return False
        attr_name, tgt_name, allow_layers = mapping[site]
        if layerIdx not in allow_layers:
            return False

        target: nn.Linear = getattr(self.base, tgt_name)
        if not hasattr(self.base, attr_name) or not isinstance(getattr(self.base, attr_name), nn.Module):
            adapter = GrowableLoRALinear(target)
            setattr(self.base, attr_name, adapter.to(target.weight.device, dtype=target.weight.dtype))

        adapter: GrowableLoRALinear = getattr(self.base, attr_name)
        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}
        adapter.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
        return True




class TestValueEstimationMTool:
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mem_dim = 768
        self.attn_dim = 512
        self.state_dim = 256

    def MakeEstimatorHebb(self, **overrides):
        est = ValueEstimationExtractor(
            memoryDim=self.mem_dim,
            attnDim=self.attn_dim,
            stateDim=self.state_dim,
            useLayerNorm=True,
            useHebb=True,
            irgKwargs={"teacherDropoutProb": 0.0},
            **overrides).to(self.device)
        est.train()
        return est

    def RandBatch(self, B: int = 3):
        mem = torch.randn(B, self.mem_dim,  device=self.device)
        attn = torch.randn(B, self.attn_dim, device=self.device)
        state = torch.randn(B, self.state_dim,device=self.device)
        return mem, attn, state

    def MakeChainEdges(self, B: int, closed: bool = False):
        if B <= 1:
            idx = torch.zeros((2,0), dtype=torch.long, device=self.device)
            return idx
        src = torch.arange(0, B-1, device=self.device, dtype=torch.long)
        dst = torch.arange(1, B, device=self.device, dtype=torch.long)
        if closed:
            src = torch.cat([src, torch.tensor([B-1], device=self.device)])
            dst = torch.cat([dst, torch.tensor([0], device=self.device)])
        edgeIndex = torch.stack([src, dst], dim=0)
        return edgeIndex

    def ParamIds(self, module: nn.Module):
        return {id(p) for p in module.parameters() if p.requires_grad}


    def TestLoRAForwardEquivalence(self) -> bool:
        try:
            torch.manual_seed(42)
            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useLayerNorm=True, useHebb=False).to(self.device)
            est.eval()

            ad = est.fc1_adapter
            in_f = int(est.fc1.in_features)
            out_f = int(est.fc1.out_features)

            rank = 3
            A = torch.randn(rank, in_f, device=self.device) * 1e-3
            Bm = torch.randn(out_f, rank, device=self.device) * 1e-3
            s = 0.7

            ad.Grow(rank, init={"A": A, "B": Bm, "scale": s}, freezeOld=True)

            x = torch.randn(5, in_f, device=self.device)
            y_ad = ad(x)
            y_ref = F.linear(x, est.fc1.weight + s * (Bm @ A), est.fc1.bias)

            ok = torch.allclose(y_ad, y_ref, atol=1e-6, rtol=1e-5)
            print(f"LoRAForwardEquivalence {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"LoRAForwardEquivalence error: {e}")
            return False

    def TestLoRAParamsGrad(self) -> bool:
        try:
            torch.manual_seed(7)
            B = 8
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useLayerNorm=True, useHebb=True,irgKwargs={"teacherDropoutProb": 0.0}).to(self.device)
            est.train()

            for ad in [est.fc1_adapter, est.fc2_adapter, est.value_adapter, est.uncert_adapter]:
                ad.Grow(2, init=None, freezeOld=True)

            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))

            out = est(memory=mem, attn=attn, state=state,
                      rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                      uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                      done=done)

            opt.zero_grad(set_to_none=True)
            out.loss.backward()

            ok = True
            msgs = []

            def finite(t):
                return torch.is_tensor(t) and torch.isfinite(t).all().item()

            def check_ad(name, ad, target_layer):
                nonlocal ok
                if len(ad.A_list) == 0:
                    ok = False; msgs.append(f"{name} has no ranks")
                    return
                A = ad.A_list[-1]; Bm = ad.B_list[-1]; s = ad.alpha[-1]
                for tag, p in [("A", A), ("B", Bm), ("alpha", s)]:
                    if (p.grad is None) or (not finite(p.grad)):
                        ok = False; msgs.append(f"{name}.{tag} grad missing/non-finite")
                if (target_layer.weight.grad is None) or (not finite(target_layer.weight.grad)):
                    ok = False; msgs.append(f"{name}.target.weight grad missing/non-finite")

            check_ad("fc1_adapter", est.fc1_adapter, est.fc1)
            check_ad("fc2_adapter", est.fc2_adapter, est.fc2)
            check_ad("value_adapter", est.value_adapter, est.value_head)
            check_ad("uncert_adapter", est.uncert_adapter, est.uncert_head)

            if not ok:
                print("LoRAParamsGrad fail:\n  " + "\n  ".join(msgs))
                return False

            torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
            opt.step()
            print("LoRAParamsGrad pass")
            return True
        except Exception as e:
            print(f"LoRAParamsGrad error: {e}")
            return False

    def TestLoRAFreezeOld(self) -> bool:
        try:
            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useLayerNorm=True, useHebb=False).to(self.device)
            est.train()

            ad = est.fc1_adapter
            ad.Grow(1, init=None, freezeOld=True)
            ad.Grow(1, init=None, freezeOld=True)

            ok_old = (not ad.A_list[0].requires_grad) and (not ad.B_list[0].requires_grad) and (not ad.alpha[0].requires_grad)
            ok_new = (ad.A_list[1].requires_grad) and (ad.B_list[1].requires_grad) and (ad.alpha[1].requires_grad)

            ok = ok_old and ok_new
            print(f"LoRAFreezeOld {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"LoRAFreezeOld error: {e}")
            return False

    def TestAllTrainableParamsHaveGradAndStep(self) -> bool:
        try:
            torch.manual_seed(101)
            B = 10
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useLayerNorm=True, useHebb=True,irgKwargs={"teacherDropoutProb": 0.0}).to(self.device)
            est.train()

            for ad in [est.fc1_adapter, est.fc2_adapter, est.value_adapter, est.uncert_adapter]:
                ad.Grow(2, init=None, freezeOld=True)

            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            with torch.no_grad():
                before = {n: p.detach().clone()
                          for n, p in est.named_parameters()
                          if p.requires_grad and p.data.numel() > 0}

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))

            out = est(memory=mem, attn=attn, state=state,
                      rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                      uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                      done=done)

            opt.zero_grad(set_to_none=True)
            out.loss.backward()

            missing = []
            for n, p in est.named_parameters():
                if p.requires_grad and p.data.numel() > 0:
                    if (p.grad is None) or (not torch.isfinite(p.grad).all().item()):
                        missing.append(n)

            if missing:
                print("[AllTrainable] missing/non-finite grads:")
                for n in missing: print("  -", n)
                return False

            torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
            opt.step()

            unchanged = []
            with torch.no_grad():
                for n, p in est.named_parameters():
                    if p.requires_grad and p.data.numel() > 0 and n in before and p.data.shape == before[n].shape:
                        if torch.allclose(p.data, before[n], atol=0, rtol=0):
                            unchanged.append(n)

            if unchanged:
                print("[AllTrainable] unchanged after step:")
                for n in unchanged: print("  -", n)
                return False

            print("AllTrainableParamsHaveGradAndStep pass")
            return True
        except Exception as e:
            print(f"AllTrainableParamsHaveGradAndStep error: {e}")
            return False

    def TestGlue1TriggersOnSecondStep(self) -> bool:
        try:
            torch.manual_seed(303)
            B = 6
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            est.train()

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))

            out1 = est(memory=mem, attn=attn, state=state,
                       rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                       uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                       done=done)
            
            g1 = float(out1.extras["loss_glue1"].item())

            out2 = est(memory=mem, attn=attn, state=state,
                       rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                       uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                       done=done)
            
            g2 = float(out2.extras["loss_glue1"].item())

            ok = (abs(g1) < 1e-12) and (g2 >= 0.0)
            print(f"Glue1TriggersOnSecondStep {'pass' if ok else 'fail'} (g1={g1:.3e}, g2={g2:.3e})")
            return ok
        except Exception as e:
            print(f"Glue1TriggersOnSecondStep error: {e}")
            return False

    def TestIntrinsicRewardGenerator(self) -> bool:
        try:
            B = 4
            mem, attn, state = self.RandBatch(B)
            irg = IntrinsicRewardGenerator(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim).to(self.device)
            irg.train()

            entropy_prev = torch.rand(B, device=self.device)
            uncert = F.softplus(torch.randn(B, device=self.device))
            td_prev = torch.randn(B, device=self.device) * 0.1

            out = irg(mem, attn, state,
                      policyEntropyPrev=entropy_prev,
                      uncertainty=uncert,
                      tdErrorPrev=td_prev)

            ok = True
            ok &= (out.rInt.shape == (B,))
            ok &= (out.eT.shape == (B,3))
            needed = ["novelty","progress","entropy","uncertainty",
                      "novelty_n","progress_n","entropy_n","uncertainty_n","valence"]
            ok &= all(k in out.components for k in needed)
            print(f"IntrinsicRewardGenerator test {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"IntrinsicRewardGenerator test error: {e}")
            return False

    def TestForwardNoReward(self) -> bool:
        try:
            B = 6
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim,stateDim=self.state_dim, useLayerNorm=True).to(self.device)
            est.train()
            est.rgen.teacher_dropout_prob = 0.0

            entropy_prev = torch.rand(B, device=self.device)
            out = est(memory=mem, attn=attn, state=state,
                      rewardExt=None, policyEntropyPrev=entropy_prev,
                      uncertaintyTeacher=None, tdErrorPrev=None,
                      done=done)

            ok = True
            ok &= (out.value.shape == (B,))
            ok &= (out.tdError.shape == (B,))
            ok &= (out.eT.shape == (B,3))
            ok &= (out.rInt.shape == (B,))
            ok &= torch.isfinite(out.loss).all()

            with torch.no_grad():
                r_used = est.wInt * out.rInt
                gamma = out.eT[...,2] * (1.0 - done)
                vhat = out.extras["v_next_hat"]
                delta_expected = r_used + gamma * vhat - out.value
                ok &= torch.allclose(out.tdError, delta_expected, atol=1e-6, rtol=1e-5)

            print(f"GeoTropical Forward (no external reward) test {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"GeoTropical Forward (no external reward) test error: {e}")
            return False

    def TestForwardWithReward(self) -> bool:
        try:
            B = 7
            mem, attn, state = self.RandBatch(B)
            done = torch.randint(0, 2, (B,), device=self.device).float()

            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim, useLayerNorm=False, wExt=1.0, wInt=1.0).to(self.device)
            est.eval()
            est.rgen.teacher_dropout_prob = 0.0

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))

            out = est(memory=mem, attn=attn, state=state,
                      rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                      uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                      done=done)

            ok = True
            ok &= (out.value.shape == (B,))
            ok &= (out.uncertainty.shape == (B,))
            ok &= (out.eT.shape == (B,3))

            with torch.no_grad():
                r_used = est.wExt * reward_ext + est.wInt * out.rInt
                gamma = out.eT[...,2] * (1.0 - done)
                vhat = out.extras["v_next_hat"]
                td_expected = r_used + gamma * vhat - out.value
                ok &= torch.allclose(out.tdError, td_expected, atol=1e-6, rtol=1e-5)

            print(f"GeoTropical Forward (with external reward) test {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"GeoTropical Forward (with external reward) test error: {e}")
            return False

    def TestBackwardOneStep(self) -> bool:
        try:
            B = 8
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim,attnDim=self.attn_dim,stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))
            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)

            out = est(memory=mem, attn=attn, state=state,
                rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                done=done)

            loss = out.loss
            assert loss.dim() == 0 and torch.isfinite(loss), "loss not scalar/finite"

            opt.zero_grad(set_to_none=True)
            loss.backward()

            has_grad = True
            bad = []
            for n, p in est.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith("rgen."):
                    if (p.grad is not None) and (not torch.isfinite(p.grad).all()):
                        has_grad = False
                        bad.append(n)
                    continue
                if (p.grad is None) or (not torch.isfinite(p.grad).all()):
                    has_grad = False
                    bad.append(n)

            if not has_grad:
                print("Bad/None grad at:\n", bad)
                return False

            torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
            opt.step()
            print("GeoTropical Backward one step passed.")
            return True
        except Exception as e:
            print(f"GeoTropical Backward one step error: {e}")
            return False

    def NoNanAfterManySteps(self, steps: int = 50) -> bool:
        try:
            est = ValueEstimationExtractor(memoryDim=self.mem_dim,attnDim=self.attn_dim,stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            for t in range(steps):
                B = 8
                mem, attn, state = self.RandBatch(B)
                done = torch.randint(0, 2, (B,), device=self.device).float() * 0

                entropy_prev = torch.rand(B, device=self.device)
                uncert_teacher = F.softplus(torch.randn(B, device=self.device))
                reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)

                out = est(memory=mem, attn=attn, state=state,
                          rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                          uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                          done=done)

                total = out.loss
                opt.zero_grad(set_to_none=True)
                total.backward()

                for n, p in est.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"
                opt.step()
            print("GeoTropical NoNanAfterManySteps passed.")
            return True
        except AssertionError as e:
            print(f"GeoTropical NoNanAfterManySteps failed: {e}")
            return False
        except Exception as e:
            print(f"GeoTropical NoNanAfterManySteps error: {e}")
            return False

    def ParamsActuallyChange(self, steps: int = 30) -> bool:
        try:
            est = ValueEstimationExtractor(memoryDim=self.mem_dim,attnDim=self.attn_dim,stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            with torch.no_grad():
                p0 = []
                for n, p in est.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        p0.append(p.data.flatten()[:64].clone())
                p0 = torch.cat(p0) if p0 else torch.zeros(1, device=self.device)

            for _ in range(steps):
                B = 8
                mem, attn, state = self.RandBatch(B)
                done = torch.zeros(B, device=self.device)
                entropy_prev = torch.rand(B, device=self.device)
                uncert_teacher = F.softplus(torch.randn(B, device=self.device))
                reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)

                out = est(memory=mem, attn=attn, state=state,
                          rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                          uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                          done=done)

                opt.zero_grad(set_to_none=True)
                out.loss.backward()
                opt.step()

            with torch.no_grad():
                p1 = []
                for n, p in est.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        p1.append(p.data.flatten()[:64].clone())
                p1 = torch.cat(p1) if p1 else torch.zeros(1, device=self.device)
                delta = (p0 - p1).abs().mean().item()

            ok = delta > 1e-6
            if ok:
                print(f"GeoTropical ParamsActuallyChange passed (delta={delta:.3e}).")
            else:
                print(f"GeoTropical ParamsActuallyChange failed (delta={delta:.3e}).")
            return ok
        except Exception as e:
            print(f"GeoTropical ParamsActuallyChange error: {e}")
            return False

    def TestLossDecreases(self, steps: int = 120, batch_size: int = 16) -> bool:
        try:
            torch.manual_seed(2025)
            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useLayerNorm=True, wExt=1.0, wInt=0.1).to(self.device)
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            losses = []
            for t in range(steps):
                mem = torch.randn(batch_size, self.mem_dim,  device=self.device)
                attn= torch.randn(batch_size, self.attn_dim, device=self.device)
                state=torch.randn(batch_size, self.state_dim,device=self.device)
                done = torch.zeros(batch_size, device=self.device)
                reward_ext = torch.randn(batch_size, device=self.device).clamp(-1, 1)
                entropy_prev= torch.rand(batch_size, device=self.device)
                uncert_teacher=F.softplus(torch.randn(batch_size, device=self.device))

                out = est(memory=mem, attn=attn, state=state,
                        rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                        uncertaintyTeacher=uncert_teacher, tdErrorPrev=None, done=done)

                total = out.loss
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
                opt.step()

                losses.append(float(total.detach().item()))
                if (t + 1) % max(1, steps // 4) == 0:
                    print(f"[GeoTropTrain] step {t+1}/{steps} | loss={losses[-1]:.6f}")

            assert len(losses) >= 8, "No valid loss trajectory is generated"

            half = len(losses)//2
            q4 = losses[-(len(losses)//4):]
            med_q4 = stats.median(q4)
            max_first_half = max(losses[:half])
            ok1 = med_q4 <= 0.3 * max_first_half

            mid = losses[half//2: half + half//2]
            tail= losses[-(len(losses)//3):]
            ok2 = (sum(tail)/len(tail)) < (sum(mid)/len(mid))

            ok = ok1 or ok2
            print(f"GeoTropical TestLossDecreases {'passed' if ok else 'failed'} " f"(median_last_quarter={med_q4:.4f}, max_first_half={max_first_half:.4f})")
            return ok
        except AssertionError as e:
            print(f"GeoTropical TestLossDecreases failed: {e}")
            return False
        except Exception as e:
            print(f"GeoTropical TestLossDecreases error: {e}")
            return False

    def TestHebbMemoryUpdates(self) -> bool:
        try:
            torch.manual_seed(7)
            B = 6
            mem, attn, state = self.RandBatch(B)
            est = self.MakeEstimatorHebb()
            H0 = est.hebb_value.H.detach().clone()

            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))
            _ = est(memory=mem, attn=attn, state=state,
                    rewardExt=None, policyEntropyPrev=entropy_prev,
                    uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                    done=torch.zeros(B, device=self.device))

            H1 = est.hebb_value.H.detach().clone()
            changed = (H1 - H0).abs().sum().item()
            ok = changed > 1e-9

            print(f"Hebbian memory update {'passed' if ok else 'failed'} (|ΔH|={changed:.3e}).")
            return ok
        except Exception as e:
            print(f"Hebbian memory update error: {e}")
            return False

    def TestWrapperAlignmentNoDelta(self) -> bool:
        try:
            torch.manual_seed(123)
            B = 6
            mem, attn, state = self.RandBatch(B)
            done = torch.ones(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            est.eval()
            est.rgen.teacher_dropout_prob = 0.0

            est.ResetMicroGraph()
            est.ResetHebbianMemory()
            H0 = est.hebb_value.H.detach().clone()

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))
            td_prev = torch.randn(B, device=self.device) * 0.1

            out_ref = est(
                memory=mem, attn=attn, state=state,
                rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                uncertaintyTeacher=uncert_teacher, tdErrorPrev=td_prev,
                done=done)

            est.hebb_value.H.copy_(H0) 
            est.ResetMicroGraph()

            rgen_state = {k: v.clone() for k, v in est.rgen.state_dict().items()}
            est.rgen.load_state_dict(rgen_state, strict=True)

            wrapper = ValueEstimationOnlineWrapper(est, initRankEach=0, autoRank=False)
            deltas = [{"fc1": None}, {"fc2": None, "vhead": None, "uhead": None}]
            out_wr = wrapper.ForwardWithDeltas(
                x=(mem, attn, state),
                keyPaddingMask=None,
                tdError=td_prev,
                uncertainty=uncert_teacher,
                deltasPerLayer=deltas,
                rewardExt=reward_ext,
                policyEntropyPrev=entropy_prev,
                done=done)

            atol, rtol = 1e-6, 1e-5
            ok = (
                torch.allclose(out_wr.value, out_ref.value, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.tdError, out_ref.tdError, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.loss, out_ref.loss, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.eT, out_ref.eT, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.uncertainty, out_ref.uncertainty, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.extras["v_next_hat"], out_ref.extras["v_next_hat"], atol=atol, rtol=rtol))
            
            print(f"WrapperAlignment_NoDelta {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"WrapperAlignment_NoDelta error: {e}")
            return False
    
    def TestSimThenCommitVHead(self) -> bool:
        try:
            torch.manual_seed(7)
            B = 5
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim, useLayerNorm=True, useHebb=False,).to(self.device)
            est.eval()
            est.rgen.teacher_dropout_prob = 0.0
            wrapper = ValueEstimationOnlineWrapper(est, initRankEach=0, autoRank=False)

            reward_ext = None
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))
            td_prev = torch.randn(B, device=self.device) * 0.05

            H = int(est.value_head.in_features)
            delta_v = (torch.randn(1, H, device=self.device) * 1e-3)

            deltas_sim = [ {"fc1": None}, {"fc2": None, "vhead": delta_v, "uhead": None} ]
            out_sim = wrapper.ForwardWithDeltas(
                x=(mem, attn, state),
                keyPaddingMask=None,
                tdError=td_prev,
                uncertainty=uncert_teacher,
                deltasPerLayer=deltas_sim,
                rewardExt=reward_ext,
                policyEntropyPrev=entropy_prev,
                done=done)

            A = delta_v.clone()
            norm = A.norm() + 1e-12
            A = A / norm
            Bm = torch.tensor([[norm.item()]], device=self.device)
            scale = 1.0

            before_ids = self.ParamIds(est)
            ok_commit = wrapper.CommitOne("vhead", layerIdx=1, a=A, b=Bm, scale=float(scale))
            assert ok_commit, "CommitOne failed for vhead"
            after_ids = self.ParamIds(est)
            new_param_count = len(after_ids - before_ids)
            assert new_param_count > 0, "No new trainable parameters after CommitOne"

            deltas_none = [ {"fc1": None}, {"fc2": None, "vhead": None, "uhead": None} ]

            est.ResetMicroGraph()

            out_after = wrapper.ForwardWithDeltas(
                x=(mem, attn, state),
                keyPaddingMask=None,
                tdError=td_prev,
                uncertainty=uncert_teacher,
                deltasPerLayer=deltas_none,
                rewardExt=reward_ext,
                policyEntropyPrev=entropy_prev,
                done=done)

            atol, rtol = 5e-6, 1e-4
            ok = (
                torch.allclose(out_after.value, out_sim.value, atol=atol, rtol=rtol) and
                torch.allclose(out_after.tdError, out_sim.tdError, atol=atol, rtol=rtol) and
                torch.allclose(out_after.loss, out_sim.loss, atol=atol, rtol=rtol) and
                torch.allclose(out_after.uncertainty,out_sim.uncertainty,atol=atol, rtol=rtol) and
                torch.allclose(out_after.extras["v_next_hat"], out_sim.extras["v_next_hat"], atol=atol, rtol=rtol))
            
            print(f"SimThenCommit_VHead {'pass' if ok else 'fail'} (new_params={new_param_count})")
            return ok
        except Exception as e:
            print(f"SimThenCommit_VHead error: {e}")
            return False

    def TestWrapperTempDeltasTrainable(self) -> bool:
        try:
            torch.manual_seed(11)
            B = 6
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim,attnDim=self.attn_dim,stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            wrapper = ValueEstimationOnlineWrapper(est, initRankEach=0, autoRank=False)

            with torch.no_grad():
                W_fc1_0 = est.fc1.weight.clone()
                W_fc2_0 = est.fc2.weight.clone()
                W_vh_0 = est.value_head.weight.clone()
                W_uh_0 = est.uncert_head.weight.clone()

            d_fc1 = nn.Parameter(torch.zeros_like(est.fc1.weight))
            d_fc2 = nn.Parameter(torch.zeros_like(est.fc2.weight))
            d_vh = nn.Parameter(torch.zeros_like(est.value_head.weight))
            d_uh = nn.Parameter(torch.zeros_like(est.uncert_head.weight))

            opt = torch.optim.Adam([d_fc1, d_fc2, d_vh, d_uh], lr=1e-1)

            entropy_prev = torch.rand(B, device=self.device)
            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))
            td_prev = None

            for _ in range(5):
                deltas = [ {"fc1": d_fc1}, {"fc2": d_fc2, "vhead": d_vh, "uhead": d_uh} ]
                out = wrapper.ForwardWithDeltas(
                    x=(mem, attn, state),
                    keyPaddingMask=None,
                    tdError=td_prev,
                    uncertainty=uncert_teacher,
                    deltasPerLayer=deltas,
                    rewardExt=reward_ext,
                    policyEntropyPrev=entropy_prev,
                    done=done)
                loss = out.loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                ok_base_unchanged = (
                    torch.allclose(est.fc1.weight, W_fc1_0, atol=0, rtol=0) and
                    torch.allclose(est.fc2.weight, W_fc2_0, atol=0, rtol=0) and
                    torch.allclose(est.value_head.weight, W_vh_0, atol=0, rtol=0) and
                    torch.allclose(est.uncert_head.weight, W_uh_0, atol=0, rtol=0))
            delta_change = (d_fc1.detach().abs().mean() + d_fc2.detach().abs().mean() + d_vh.detach().abs().mean() + d_uh.detach().abs().mean()).item()
            ok_delta_changed = delta_change > 0

            ok = ok_base_unchanged and ok_delta_changed
            print(f"WrapperTempDeltasTrainable {'pass' if ok else 'fail'} "
                  f"(Δ_abs_mean_sum={delta_change:.3e})")
            return ok
        except Exception as e:
            print(f"WrapperTempDeltasTrainable error: {e}")
            return False

    def TestGradFlowCoverage(self) -> bool:
        try:
            torch.manual_seed(17)
            B = 8
            mem, attn, state = self.RandBatch(B)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim,attnDim=self.attn_dim,stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            est.train()
            est.rgen.teacher_dropout_prob = 0.0

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))

            out = est(memory=mem, attn=attn, state=state,rewardExt=reward_ext, policyEntropyPrev=entropy_prev,uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,done=done)

            loss = out.loss
            loss.backward()

            keys = [
                "fc1.weight", "fc2.weight",
                "value_head.weight", "uncert_head.weight",
                "transport.trop.W", "transport.a_net.0.weight",
                "transport.b_net.0.weight", "transport.g_net.0.weight",]
            ok = True
            bad = []
            for name, p in est.named_parameters():
                if any(name.endswith(k) for k in keys):
                    if (p.grad is None) or (not torch.isfinite(p.grad).all()):
                        ok = False; bad.append(name)
            if not ok:
                print("GradFlowCoverage failed:", bad)
            else:
                print("GradFlowCoverage pass")
            return ok
        except Exception as e:
            print(f"GradFlowCoverage error: {e}")
            return False

    def TestWrapperKwargsValidation(self) -> bool:
        try:
            B = 3
            mem, attn, state = self.RandBatch(B)
            est = ValueEstimationExtractor(memoryDim=self.mem_dim,attnDim=self.attn_dim,stateDim=self.state_dim,useLayerNorm=True).to(self.device)
            wrapper = ValueEstimationOnlineWrapper(est, initRankEach=0, autoRank=False)

            try:
                _ = wrapper.ForwardWithDeltas(
                    x=(mem, attn, state),
                    keyPaddingMask=None,
                    tdError=None,
                    uncertainty=None,
                    deltasPerLayer=[{"fc1": None}, {"fc2": None, "vhead": None, "uhead": None}],
                    mysterious_key=torch.tensor(1.0, device=self.device), )
            except TypeError:
                print("WrapperKwargsValidation pass")
                return True
            except Exception as e:
                print(f"WrapperKwargsValidation wrong exception: {e}")
                return False
            print("WrapperKwargsValidation fail (no exception)")
            return False
        except Exception as e:
            print(f"WrapperKwargsValidation error: {e}")
            return False

    def RunAll(self):
        results = {
            "LoRAForwardEquivalence": self.TestLoRAForwardEquivalence(),
            "LoRAParamsGrad": self.TestLoRAParamsGrad(),
            "LoRAFreezeOld": self.TestLoRAFreezeOld(),
            "AllTrainableParamsHaveGradAndStep": self.TestAllTrainableParamsHaveGradAndStep(),
            "Glue1TriggersOnSecondStep": self.TestGlue1TriggersOnSecondStep(),
            "WrapperAlignmentNoDelta": self.TestWrapperAlignmentNoDelta(),
            "SimThenCommitVHead": self.TestSimThenCommitVHead(),
            "WrapperTempDeltasTrainable": self.TestWrapperTempDeltasTrainable(),
            "GradFlowCoverage": self.TestGradFlowCoverage(),
            "WrapperKwargsValidation": self.TestWrapperKwargsValidation(),
            "IntrinsicRewardGenerator": self.TestIntrinsicRewardGenerator(),
            "ForwardNoReward": self.TestForwardNoReward(),
            "ForwardWithReward": self.TestForwardWithReward(),
            "BackwardOneStep": self.TestBackwardOneStep(),
            "NoNanAfterManySteps": self.NoNanAfterManySteps(),
            "ParamsActuallyChange": self.ParamsActuallyChange(),
            "LossDecreases": self.TestLossDecreases(),
            "HebbMemoryUpdates": self.TestHebbMemoryUpdates(),}

        passed = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\n[ValueEstimationExtractor Tests] {passed}/{total} passed.")
        return results
