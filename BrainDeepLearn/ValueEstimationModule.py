from __future__ import annotations
from typing import Optional, Dict, NamedTuple, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HebbianLinearFW(nn.Module):
    def __init__(self, inFeatures: int, outFeatures: int, bias: bool = True, *,initEta: float = 1e-3, initLambda: float = 0.1, cap: float = 1.0,useOja: bool = True, detachHebb: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outFeatures, inFeatures))
        self.bias = nn.Parameter(torch.zeros(outFeatures)) if bias else None
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        self.register_buffer("H", torch.zeros(outFeatures, inFeatures))
        self.init_eta = initEta
        self.init_lambda = initLambda
        self.cap = cap
        self.use_oja = useOja
        self.detach_hebb = detachHebb

    @torch.no_grad()
    def ResetHebbianMemory(self):
        self.H.zero_()

    @torch.no_grad()
    def ProjectCap(self):
        if self.cap is None:
            return
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
            pre  = x
            post = y
            if self.detach_hebb:
                pre = pre.detach()
                post = post.detach()

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
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var",  torch.ones(dim))

    @torch.no_grad()
    def Update(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        mask = torch.isfinite(x).all(dim=-1)
        if mask.any():
            x = x[mask]
            m = x.mean(0)
            v = x.var(0, unbiased=False)
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
    def __init__(self,
                 memoryDim: int = 768,
                 attnDim: int = 1024,
                 stateDim: int = 256,
                 *,
                 hidden: int = 256,
                 alphaNovelty: float = 1.0,
                 alphaEntropy: float = 0.2,
                 alphaProgress: float = 0.5,
                 alphaUncertPenalty: float = 0.5,
                 rClip: float = 5.0,
                 tau0: float = 1.0, beta: float = 1.0,
                 lr0: float = 1.0,  kappa: float = 0.5,
                 gamma0: float = 0.99, delta: float = 0.02,
                 tauMin: Optional[float] = None, tauMax: Optional[float] = 10.0,
                 lrMin: Optional[float]  = 0.25, lrMax: Optional[float]  = 3.0,
                 gammaMin: float = 0.90, gammaMax: float = 0.9999,
                 emaMomentum: float = 0.99,
                 teacherDropoutProb: float = 0.1,
                 gateReg: float = 1e-3,
                 eTAnchor: float = 1e-3):
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
        self.lr_min, self.lr_max  = lrMin,  lrMax
        self.gamma_min, self.gamma_max = gammaMin, gammaMax

        self.nov_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.unc_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.prog_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.ent_ema = RunningEMA(dim=1, momentum=emaMomentum)

        in_dim = memoryDim + attnDim + stateDim
        self.affect_net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),)
        
        self.progress_head = nn.Linear(hidden, 1)

        mid = max(32, hidden // 2)

        self.entropy_from_h = nn.Sequential(
            nn.Linear(hidden, mid), nn.ReLU(),
            nn.Linear(mid, 1), nn.Softplus())
        
        self.uncert_from_h = nn.Sequential(
            nn.Linear(hidden, mid), nn.ReLU(),
            nn.Linear(mid, 1), nn.Softplus())
        
        self.entropy_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.uncert_gate  = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

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
        mean_s = s.mean(0)
        self.state_ema.copy_(self.state_ema * self.state_momentum + (1 - self.state_momentum) * mean_s)

    def MaybeDropoutTeacher(self, t: Optional[torch.Tensor], B: int, device) -> Optional[torch.Tensor]:
        if (t is None) or (not self.training) or (self.teacher_dropout_prob <= 0):
            return t
        keep = torch.rand(B, device=device) > self.teacher_dropout_prob
        if keep.all():
            return t
        t = t.clone()
        t[~keep] = float('nan')
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

        if tdErrorPrev is not None:
            progress = -tdErrorPrev.abs()
        else:
            progress = torch.tanh(self.progress_head(h).squeeze(-1))

        policyEntropyPrev = self.MaybeDropoutTeacher(policyEntropyPrev, B, device)
        uncertainty = self.MaybeDropoutTeacher(uncertainty, B, device)

        entropy_pred = self.entropy_from_h(h).squeeze(-1)
        uncert_pred = self.uncert_from_h(h).squeeze(-1)
        g_e = self.entropy_gate(h).squeeze(-1)
        g_u = self.uncert_gate(h).squeeze(-1)

        if policyEntropyPrev is not None:
            pe = policyEntropyPrev
            pe = torch.where(torch.isfinite(pe), pe, entropy_pred.detach())
            fused_entropy = (1.0 - g_e) * pe + g_e * entropy_pred
        else:
            fused_entropy = entropy_pred

        if uncertainty is not None:
            uu = uncertainty
            uu = torch.where(torch.isfinite(uu), uu, uncert_pred.detach())
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
        temp_scale = self.tau0 * torch.exp(exp_arg)

        if (self.tau_min is not None) or (self.tau_max is not None):
            lo = self.tau_min if self.tau_min is not None else -float('inf')
            hi = self.tau_max if self.tau_max is not None else float('inf')
            temp_scale = temp_scale.clamp(lo, hi)

        lr_scale = self.lr0 * (1.0 + self.kappa * novelty_n.clamp_min(0.0))
        if (self.lr_min is not None) or (self.lr_max is not None):
            lo = self.lr_min if self.lr_min is not None else -float("inf")
            hi = self.lr_max if self.lr_max is not None else +float("inf")
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
            comps["reg_eT"] = self.eT_anchor * ((temp_scale - self.tau0).pow(2) + (lr_scale - self.lr0 ).pow(2) + (gamma_mod - self.gamma0).pow(2))

        return IntrinsicRewardOut(rInt=r_int, components=comps, eT=e_t)

class MaxPlusLinear(nn.Module):
    def __init__(self, inFeatures: int, outFeatures: int, useSoft: bool = True, temperature: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(outFeatures, inFeatures))
        self.b = nn.Parameter(torch.zeros(outFeatures))
        self.use_soft = useSoft
        self.temperature = temperature
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.W.unsqueeze(0) + x.unsqueeze(1)
        if self.use_soft:
            t = max(self.temperature, 1e-6) 
            z = score / t
            m = z.amax(dim=-1, keepdim=True) 
            y = t * (m.squeeze(-1) + torch.logsumexp(z - m, dim=-1))
        else:
            y, _ = score.max(dim=-1)
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
        trop_in = torch.cat([h, v_in], dim=-1)
        trop_out = self.trop(trop_in).squeeze(-1)
        aff_out  = (a.squeeze(-1) * v) + b.squeeze(-1)
        v_next_hat = g.squeeze(-1) * trop_out + (1.0 - g.squeeze(-1)) * aff_out
        extras = {
            "gate_trop": g.squeeze(-1),
            "a": a.squeeze(-1),
            "b": b.squeeze(-1),
            "trop_out": trop_out,
            "aff_out": aff_out,}
        return v_next_hat, extras

class GITGaugeRegularizer(nn.Module):
    def __init__(self, wScale: float = 1e-3, wShift: float = 1e-3, wSign: float = 1e-3):
        super().__init__()
        self.w_scale = wScale
        self.w_shift = wShift
        self.w_sign = wSign

    def forward(self,
                value_head: nn.Linear,
                transp_extras: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        reg = value_head.weight.new_zeros(())
        W = value_head.weight
        fro = torch.linalg.matrix_norm(W, ord='fro')
        reg = reg + self.w_scale * (fro - 1.0).pow(2)

        if "b" in transp_extras:
            b = transp_extras["b"]
            reg = reg + self.w_shift * (b.mean()).pow(2)

        with torch.no_grad():
            idx = torch.argmax(W.abs()).item() if W.numel() > 0 else 0
        row = idx // W.size(1) if W.numel() > 0 else 0
        if W.numel() > 0:
            wrow = W[row]
            reg = reg + self.w_sign * F.relu(-wrow.sum())

        return reg

def BtVMinusF(value: torch.Tensor,gammaEdge: torch.Tensor,rEdge: torch.Tensor,edgeIndex: torch.Tensor) -> torch.Tensor:
    u, v = edgeIndex[0], edgeIndex[1]
    pred = value[u] - gammaEdge * value[v]
    e = rEdge - pred
    return e

def WeightedLeastSquaresCycleEnergy(valueInit: torch.Tensor, rEdge: torch.Tensor,gammaEdge: torch.Tensor,edgeIndex: torch.Tensor, wEdge: torch.Tensor, cgSteps: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    N = int(valueInit.numel())
    device = valueInit.device

    u, v = edgeIndex[0], edgeIndex[1]

    def BWBTMv(x: torch.Tensor) -> torch.Tensor:
        # x: [N] → y = B W B^T x
        # y_e = (B^T x)_e = x_u - gamma_e * x_v
        y_e = x[u] - gammaEdge * x[v] 
        y_e = wEdge * y_e 
        #  y = B y_e
        y = torch.zeros_like(x)
        y.index_add_(0, u, y_e)
        y.index_add_(0, v, -gammaEdge * y_e)
        return y

    with torch.no_grad():
        rhs_e = wEdge * rEdge
        rhs = torch.zeros(N, device=device)
        rhs.index_add_(0, u, rhs_e)
        rhs.index_add_(0, v, - gammaEdge * rhs_e)

    if cgSteps <= 0:
        A_diag = torch.zeros(N, device=device)
        A_diag.index_add_(0, u, wEdge)
        A_diag.index_add_(0, v, wEdge * (gammaEdge**2))
        reg = 1e-5
        x = rhs / (A_diag + reg) 
    else:
        x = torch.zeros(N, device=device)
        r = rhs - BWBTMv(x)
        p = r.clone()
        rs_old = (r*r).sum()
        for _ in range(cgSteps):
            Ap = BWBTMv(p)
            alpha = rs_old / (p*Ap + 1e-12).sum()
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = (r*r).sum()
            if rs_new.sqrt() < 1e-6:
                break
            p = r + (rs_new/rs_old) * p
            rs_old = rs_new

    with torch.enable_grad():
        x_det = x.detach().requires_grad_(True)
        # B^T x_det - r
        pred_e = x_det[u] - gammaEdge * x_det[v]
        res_e  = (pred_e - rEdge)
        res_e_w = (wEdge.sqrt() * res_e)
        energy = (res_e_w**2).sum() / (wEdge.sum().clamp_min(1.0))
    return energy, x.detach()

class PersistentCohomologyRegularizer(nn.Module):
    def __init__(self,
                 numLevels: int = 5,
                 temperature: float = 0.5,
                 cgSteps: int = 0,
                 weight: float = 1e-3,
                 blend: float = 0.5,
                 alignWeight: float = 1e-3,
                 alignUseDegree: bool = True):
        super().__init__()
        self.num_levels = numLevels
        self.temperature = temperature
        self.cg_steps = cgSteps
        self.weight = weight
        self.blend = blend
        self.align_weight = alignWeight
        self.align_use_degree = alignUseDegree

    def forward(self, valueNodes, rEdge, gammaEdge, edgeIndex):
        if edgeIndex is None or rEdge.numel() == 0:
            return torch.tensor(0.0, device=valueNodes.device)

        u, v = edgeIndex[0], edgeIndex[1]

        inst_res = rEdge - (valueNodes[u] - gammaEdge * valueNodes[v])
        inst_energy = (inst_res**2).mean()

        with torch.no_grad():
            e0 = (rEdge - (valueNodes[u] - gammaEdge * valueNodes[v])).abs()
            K = self.num_levels
            qs = torch.linspace(0.2, 0.9, K, device=valueNodes.device)
            Ts = torch.quantile(e0, qs).detach()

        proj_total = 0.0
        align_total = 0.0

        for T in Ts:
            w = torch.sigmoid((e0 - T) / self.temperature)
            energy_proj, x_star = WeightedLeastSquaresCycleEnergy(
                valueInit=valueNodes, rEdge=rEdge, gammaEdge=gammaEdge,
                edgeIndex=edgeIndex, wEdge=w, cgSteps=self.cg_steps)

            proj_total = proj_total + energy_proj

            if self.align_use_degree:
                deg = torch.zeros_like(valueNodes)
                deg.index_add_(0, u, w)
                deg.index_add_(0, v, w * (gammaEdge**2))
                align = (deg * (valueNodes - x_star).pow(2)).sum() / deg.sum().clamp_min(1.0)
            else:
                align = (valueNodes - x_star).pow(2).mean()

            align_total = align_total + align

        proj_total  = proj_total  / K
        align_total = align_total / K

        total = (self.weight * ( self.blend * proj_total + (1.0 - self.blend) * inst_energy ) + self.align_weight * align_total)
        return total

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
                 memoryDim: int = 768,
                 attnDim: int = 1024,
                 stateDim: int = 256,
                 *,
                 hidden: int = 512,
                 useLayerNorm: bool = False,
                 irgKwargs: Optional[dict] = None,
                 wExt: float = 1.0, wInt: float = 1.0,
                 stopGradRGamma: bool = True,
                 useSoftTrop: bool = True, tropTemp: float = 0.2, epsA: float = 1e-3,
                 wTD: float = 1.0, wGlue: float = 1e-2, wCurv: float = 1e-3,
                 wUncertTeacher: float = 1e-2,
                 wEntropyTeacher: float = 1e-3,
                 persLevels: int = 5, persTemp: float = 0.5, persWeight: float = 1e-3,
                 wGITScale: float = 1e-3, wGITShift: float = 1e-3, wGITSign: float = 1e-3,
                 useHebb: bool = False, hebbCap: float = 1.0, hebbOja: bool = True, detachHebbGrad: bool = True,):
        super().__init__()
        self.in_dim = memoryDim + attnDim + stateDim
        H = hidden

        self.use_hebb = useHebb

        self.wEntropyTeacher = wEntropyTeacher

        self.fc1 = nn.Linear(self.in_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.norm1 = nn.LayerNorm(H) if useLayerNorm else None
        self.norm2 = nn.LayerNorm(H) if useLayerNorm else None

        if self.use_hebb:
            self.hebb_value = HebbianLinearFW(H, 1, bias=True,initEta=1e-3, initLambda=0.1,cap=hebbCap, useOja=hebbOja, detachHebb=detachHebbGrad)

        self.value_head  = nn.Linear(H, 1)
        self.uncert_head = nn.Linear(H, 1)

        self.transport = TropicalAffineTransport(H, useSoftTrop, tropTemp, epsA)
        self.rgen = IntrinsicRewardGenerator(memoryDim, attnDim, stateDim, **(irgKwargs or {}))
        self.git = GITGaugeRegularizer(wScale=wGITScale, wShift=wGITShift, wSign=wGITSign)
        self.pers = PersistentCohomologyRegularizer(numLevels=persLevels, temperature=persTemp, weight=persWeight)

        self.wExt, self.wInt = wExt, wInt
        self.wTD, self.wGlue, self.wCurv, self.wUncertTeacher = wTD, wGlue, wCurv, wUncertTeacher
        self.stopGrad_r_gamma = stopGradRGamma

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

    def Trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x)); h = self.norm1(h) if self.norm1 is not None else h
        h = F.relu(self.fc2(h)); h = self.norm2(h) if self.norm2 is not None else h
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
                done: Optional[torch.Tensor] = None, 
                edgeIndex: Optional[torch.Tensor] = None,
                edgeWeight: Optional[torch.Tensor] = None
                ) -> GeoTropicalOut:

        B, device = state.size(0), state.device
        x = torch.cat([memory, attn, state], dim=-1)
        h = self.Trunk(x)

        uncert_pred_fallback = F.softplus(self.uncert_head(h).squeeze(-1)).detach()

        irg_out = self.rgen(memoryPrev=memory, attnPrev=attn, stateCurr=state,
                            policyEntropyPrev=policyEntropyPrev,
                            uncertainty=(uncertaintyTeacher if uncertaintyTeacher is not None else uncert_pred_fallback),
                            tdErrorPrev=tdErrorPrev)
        r_int, eT, comps = irg_out.rInt.detach(), irg_out.eT, irg_out.components

        hebb_eta = hebb_lam = beta_mix = None

        uncert_pred = F.softplus(self.uncert_head(h).squeeze(-1))

        if self.use_hebb:
            hebb_eta = eT[..., 1].clamp_min(0).tanh() * 0.01 
            hebb_lam = torch.full((B,), 0.1, device=device)
            beta_mix = torch.zeros(B, device=device)
            v_hebb, hebb_extras = self.hebb_value(h, eta=hebb_eta, lam=hebb_lam, betaMix=beta_mix)
            v_param = self.value_head(h).squeeze(-1)
            mix = torch.sigmoid(beta_mix) if beta_mix is not None else 0.5
            value = (1.0 - mix) * v_param + mix * v_hebb.squeeze(-1)
        else:
            value = self.value_head(h).squeeze(-1)
            hebb_extras = {"H_norm": torch.tensor(0.0, device=device)}

        if rewardExt is None:
            r_used = self.wInt * r_int
        else:
            r_used = self.wExt * rewardExt.to(device) + self.wInt * r_int
        gamma = eT[..., 2]
        if self.stopGrad_r_gamma:
            r_used = r_used.detach()
            gamma  = gamma.detach()
        if done is not None:
            gamma = gamma * (1.0 - done.float()) 

        v_next_hat, transp_extras = self.transport(h, value) 

        delta = r_used + gamma * v_next_hat - value
        loss_td = (delta ** 2).mean() * self.wTD

        if edgeIndex is not None:
            assert edgeIndex.dtype in (torch.int64, torch.long), "edgeIndex must be LongTensor [2,E]"
            assert edgeIndex.dim() == 2 and edgeIndex.size(0) == 2, "edgeIndex shape must be [2,E]"
            assert (edgeIndex[0] >= 0).all() and (edgeIndex[1] >= 0).all() and (edgeIndex[0] < B).all() and (edgeIndex[1] < B).all(), "edgeIndex out of batch range"

        loss_glue = torch.tensor(0.0, device=device)
        loss_curv = torch.tensor(0.0, device=device)
        loss_pers = torch.tensor(0.0, device=device)

        if (edgeIndex is not None) and (edgeIndex.numel() > 0):
            u_all, v_all = edgeIndex[0].to(device), edgeIndex[1].to(device)
            if done is not None:
                mask_edge = (done[u_all] < 0.5)
                u_all, v_all = u_all[mask_edge], v_all[mask_edge]

            if u_all.numel() > 0:
                w = edgeWeight.to(device) if edgeWeight is not None else torch.ones_like(u_all, dtype=value.dtype, device=device)

                glue_resid = (v_next_hat[u_all] - value[v_all])
                loss_glue = ((w * glue_resid.pow(2)).sum() / (w.sum().clamp_min(1.0))) * self.wGlue

                r_edge = r_used[u_all]
                gamma_edge = gamma[u_all]
                eidx_masked = torch.stack([u_all, v_all], dim=0)
                curv_e = BtVMinusF(value, gamma_edge, r_edge, eidx_masked)
                loss_curv = ((w * curv_e.pow(2)).sum() / (w.sum().clamp_min(1.0))) * self.wCurv

                loss_pers = self.pers(valueNodes=value,
                                      rEdge=r_edge,
                                      gammaEdge=gamma_edge,
                                      edgeIndex=eidx_masked)

        loss_unc = value.new_zeros(()) 
        if (uncertaintyTeacher is not None) and (self.wUncertTeacher > 0):
            m = torch.isfinite(uncertaintyTeacher) 
            if m.any():
                loss_unc = self.wUncertTeacher * F.mse_loss(uncert_pred[m], uncertaintyTeacher[m])
            else:
                loss_unc = 1e-4 * uncert_pred.mean()
        elif self.wUncertTeacher > 0:
            loss_unc = 1e-4 * uncert_pred.mean()

        loss_git = self.git(self.value_head, transp_extras)

        loss_gate_trop = torch.tensor(0.0, device=device)
        if "gate_trop" in transp_extras:
            gt = transp_extras["gate_trop"]
            loss_gate_trop = 1e-3 * ((gt - 0.5)**2).mean()

        loss_irg = 0.0
        if "reg_gate" in comps: loss_irg = loss_irg + comps["reg_gate"].mean()
        if "reg_eT" in comps: loss_irg = loss_irg + comps["reg_eT"].mean()

        loss_ent = torch.tensor(0.0, device=device)
        if (policyEntropyPrev is not None) and (self.wEntropyTeacher > 0):
            e_pred = irg_out.components.get("entropy_pred", None)
            if e_pred is not None:
                m = torch.isfinite(policyEntropyPrev)
                if m.any():
                    loss_ent = F.mse_loss(e_pred[m], policyEntropyPrev[m]) * self.wEntropyTeacher


        total_loss = loss_td + loss_glue + loss_curv + loss_pers + loss_unc + loss_git + loss_gate_trop + loss_irg +loss_ent

        extras: Dict[str, torch.Tensor] = {
            "loss_td": loss_td.detach(),
            "loss_glue": loss_glue.detach(),
            "loss_curv": loss_curv.detach(),
            "loss_pers": loss_pers.detach(),
            "loss_unc": loss_unc.detach(),
            "loss_git": loss_git.detach(),
            "loss_gate_trop": loss_gate_trop.detach(),
            "v_next_hat": v_next_hat.detach(),
            "gate_trop": transp_extras.get("gate_trop", torch.zeros_like(value)).detach(),
            "trop_out": transp_extras.get("trop_out", torch.zeros_like(value)).detach(),
            "aff_out": transp_extras.get("aff_out", torch.zeros_like(value)).detach(),
            "a": transp_extras.get("a", torch.zeros_like(value)).detach(),
            "b": transp_extras.get("b", torch.zeros_like(value)).detach(),
            "hebb_H_norm": hebb_extras.get("H_norm", torch.tensor(0.0, device=device)).detach()}

        return GeoTropicalOut(
            value=value,
            tdError=delta,
            loss=total_loss,
            eT=eT,
            rInt=r_int,
            rComps={k: v.detach() for k, v in comps.items()},
            uncertainty=uncert_pred,
            extras=extras)




class TestValueEstimationMTool:
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mem_dim = 768
        self.attn_dim = 512
        self.state_dim= 256

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
            edgeIndex = self.MakeChainEdges(B, closed=False)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(memoryDim=self.mem_dim,
                                            attnDim=self.attn_dim,
                                            stateDim=self.state_dim,
                                            useLayerNorm=True).to(self.device)
            est.train()

            entropy_prev = torch.rand(B, device=self.device)
            out = est(memory=mem, attn=attn, state=state,
                      rewardExt=None, policyEntropyPrev=entropy_prev,
                      uncertaintyTeacher=None, tdErrorPrev=None,
                      done=done, edgeIndex=edgeIndex, edgeWeight=None)

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
            edgeIndex = self.MakeChainEdges(B, closed=True)
            done = torch.randint(0, 2, (B,), device=self.device).float()

            est = ValueEstimationExtractor(memoryDim=self.mem_dim,
                                            attnDim=self.attn_dim,
                                            stateDim=self.state_dim,
                                            useLayerNorm=False,
                                            wExt=1.0, wInt=1.0).to(self.device)
            est.eval()

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))

            out = est(memory=mem, attn=attn, state=state,
                      rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                      uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                      done=done, edgeIndex=edgeIndex, edgeWeight=None)

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
            edgeIndex = self.MakeChainEdges(B, closed=True)
            done = torch.zeros(B, device=self.device)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim,
                attnDim=self.attn_dim,
                stateDim=self.state_dim,
                useLayerNorm=True).to(self.device)
            est.train()
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            entropy_prev = torch.rand(B, device=self.device)
            uncert_teacher = F.softplus(torch.randn(B, device=self.device))
            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)

            out = est(
                memory=mem, attn=attn, state=state,
                rewardExt=reward_ext,
                policyEntropyPrev=entropy_prev,
                uncertaintyTeacher=uncert_teacher,
                tdErrorPrev=None,
                done=done,
                edgeIndex=edgeIndex,
                edgeWeight=None)

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
            est = ValueEstimationExtractor(memoryDim=self.mem_dim,
                                            attnDim=self.attn_dim,
                                            stateDim=self.state_dim,
                                            useLayerNorm=True).to(self.device)
            est.train()
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            for t in range(steps):
                B = 8
                mem, attn, state = self.RandBatch(B)
                edgeIndex = self.MakeChainEdges(B, closed=(t%3==0))
                done = torch.randint(0, 2, (B,), device=self.device).float() * 0 

                entropy_prev = torch.rand(B, device=self.device)
                uncert_teacher = F.softplus(torch.randn(B, device=self.device))
                reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)

                out = est(memory=mem, attn=attn, state=state,
                          rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                          uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                          done=done, edgeIndex=edgeIndex, edgeWeight=None)

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
            est = ValueEstimationExtractor(memoryDim=self.mem_dim,
                                            attnDim=self.attn_dim,
                                            stateDim=self.state_dim,
                                            useLayerNorm=True).to(self.device)
            est.train()
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
                edgeIndex = self.MakeChainEdges(B, closed=True)
                done = torch.zeros(B, device=self.device)
                entropy_prev = torch.rand(B, device=self.device)
                uncert_teacher = F.softplus(torch.randn(B, device=self.device))
                reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)

                out = est(memory=mem, attn=attn, state=state,
                          rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                          uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                          done=done, edgeIndex=edgeIndex, edgeWeight=None)

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
            est = ValueEstimationExtractor(memoryDim=self.mem_dim,
                                            attnDim=self.attn_dim,
                                            stateDim=self.state_dim,
                                            useLayerNorm=True,
                                            wExt=1.0, wInt=0.1).to(self.device)
            est.train()
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            losses = []
            for t in range(steps):
                mem = torch.randn(batch_size, self.mem_dim,  device=self.device)
                attn = torch.randn(batch_size, self.attn_dim, device=self.device)
                state = torch.randn(batch_size, self.state_dim,device=self.device)

                edgeIndex = self.MakeChainEdges(batch_size, closed=(t%4==0))
                done = torch.zeros(batch_size, device=self.device)

                reward_ext = torch.randn(batch_size, device=self.device).clamp(-1, 1)
                entropy_prev = torch.rand(batch_size, device=self.device)
                uncert_teacher = F.softplus(torch.randn(batch_size, device=self.device))

                out = est(memory=mem, attn=attn, state=state,
                          rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                          uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                          done=done, edgeIndex=edgeIndex, edgeWeight=None)

                total = out.loss
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
                opt.step()

                losses.append(float(total.detach().item()))
                if (t + 1) % max(1, steps // 4) == 0:
                    print(f"[GeoTropTrain] step {t+1}/{steps} | loss={losses[-1]:.6f}")

            assert len(losses) >= 2, "No valid loss trajectory is generated"
            start, end = losses[0], min(losses[-1], sum(losses[-10:])/max(1,len(losses[-10:])))
            print(f"\n[GeoTropTrain] loss start={start:.6f} -> end={end:.6f}\n")

            rel_ok = end <= start * 0.80
            abs_ok = (start - end) >= 0.05
            ok = rel_ok or abs_ok
            print(f"GeoTropical TestLossDecreases {'passed' if ok else 'failed'}.")
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
            out = est(memory=mem, attn=attn, state=state,
                      rewardExt=None, policyEntropyPrev=entropy_prev,
                      uncertaintyTeacher=uncert_teacher, tdErrorPrev=None,
                      done=torch.zeros(B, device=self.device),
                      edgeIndex=self.MakeChainEdges(B, closed=False), edgeWeight=None)

            H1 = est.hebb_value.H.detach().clone()
            changed = (H1 - H0).abs().sum().item()
            ok = changed > 1e-9

            print(f"Hebbian memory update {'passed' if ok else 'failed'} (|ΔH|={changed:.3e}).")
            return ok
        except Exception as e:
            print(f"Hebbian memory update error: {e}")
            return False


    def RunAll(self):
        results = []
        results.append(self.TestIntrinsicRewardGenerator())
        results.append(self.TestForwardNoReward())
        results.append(self.TestForwardWithReward())
        results.append(self.TestBackwardOneStep())
        results.append(self.NoNanAfterManySteps())
        results.append(self.ParamsActuallyChange())
        results.append(self.TestLossDecreases())
        results.append(self.TestHebbMemoryUpdates())

        passed = sum(1 for x in results if x)
        print(f"\n[ValueEstimationExtractor Tests] {passed}/{len(results)} passed.")
        return all(results)
