from __future__ import annotations
from typing import Optional, Dict, NamedTuple, Tuple, List, Any
import math
import copy
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

        scale = 0.0 if betaMix is None else betaMix.mean()
        H_eff = self.H.detach().clone()
        W_eff = self.weight + scale * H_eff
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
        std = (self.var + self.eps).sqrt()
        return (x - self.mean) / std.clamp_min(self.eps)



class UncertaintyCore(nn.Module):
    def __init__(
        self, hDim: int, *, ensK: int = 4, emaMomentum: float = 0.99,
        w_td: float = 1.0, w_r: float = 0.25, w_ent: float = 0.25,
        w_nll: float = 1.0, w_calib: float = 0.25, w_smooth: float = 0.05,
        w_ens: float = 0.10, bootstrap_keep: float = 0.67, eps: float = 1e-6):
        super().__init__()
        self.ensK = int(ensK)
        self.logvar_head = nn.Linear(hDim, 1) 
        self.ens = nn.ModuleList([nn.Linear(hDim, 1) for _ in range(self.ensK)])

        self.td_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.r_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.ent_ema = RunningEMA(dim=1, momentum=emaMomentum)

        self.w_td, self.w_r, self.w_ent = float(w_td), float(w_r), float(w_ent)
        self.w_nll, self.w_calib, self.w_smooth = float(w_nll), float(w_calib), float(w_smooth)
        self.w_ens = float(w_ens)
        self.bootstrap_keep = float(bootstrap_keep)
        self.eps = float(eps)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def PriorFromPrev(
        self,
        *,
        tdErrorPrev: Optional[torch.Tensor],
        rewardPrev: Optional[torch.Tensor],
        policyEntropyPrev: Optional[torch.Tensor],
        like: torch.Tensor) -> torch.Tensor:
        B = like.size(0)
        z = like.new_zeros((B, 1))

        td = tdErrorPrev if tdErrorPrev is not None else z
        rew = rewardPrev if rewardPrev is not None else z
        ent = policyEntropyPrev if policyEntropyPrev is not None else z

        td_abs = td.abs()
        r_abs = rew.abs()
        ent_v = ent

        self.td_ema.Update(td_abs)
        self.r_ema.Update(r_abs)
        self.ent_ema.Update(ent_v)

        td_n = self.td_ema.Norm(td_abs).clamp_(-8.0, 8.0)
        r_n = self.r_ema.Norm(r_abs).clamp_(-8.0, 8.0)
        ent_n = self.ent_ema.Norm(ent_v).clamp_(-8.0, 8.0)

        u_lin = self.w_td * td_n + self.w_r * r_n + self.w_ent * ent_n
        u_prior = F.softplus(u_lin) + self.eps
        return u_prior

    def forward(self, *,
        h: torch.Tensor,
        valueParam: torch.Tensor, 
        valueHebb: Optional[torch.Tensor],
        tdErrorCurr: Optional[torch.Tensor], 
        tdErrorPrev: Optional[torch.Tensor],
        rewardPrev: Optional[torch.Tensor], 
        policyEntropyPrev: Optional[torch.Tensor],
        donePrev: Optional[torch.Tensor] = None, ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        B = h.size(0)

        u_prior = self.PriorFromPrev(tdErrorPrev=tdErrorPrev, rewardPrev=rewardPrev,
                                     policyEntropyPrev=policyEntropyPrev, like=valueParam)

        if donePrev is not None:
            alive = (1.0 - donePrev.float()).clamp(0.0, 1.0)
            u_prior = u_prior * alive

        log_sigma2 = self.logvar_head(h).clamp(-10.0, 5.0)
        sigma2_ale = torch.exp(log_sigma2) + self.eps

        ens_vals = torch.stack([m(h) for m in self.ens], dim=0) 
        var_ens = ens_vals.var(dim=0, unbiased=False)

        dis_ph = valueParam.new_zeros((B, 1))
        if valueHebb is not None:
            dis_ph = (valueParam.detach() - valueHebb.detach()).pow(2)

        sigma2_epi = (var_ens + dis_ph).clamp_min(0.0)

        unc2 = sigma2_ale + sigma2_epi
        unc_total = torch.sqrt(unc2 + self.eps) 

        loss_unc = h.new_zeros(())
        if self.training:
            loss_ens = h.new_zeros(())
            if self.w_ens > 0:
                keep = (torch.rand(self.ensK, B, 1, device=h.device) < self.bootstrap_keep).float()
                tgt = valueParam.detach()
                denom = keep.sum().clamp_min(1.0)
                loss_ens = ((keep * (ens_vals - tgt).pow(2)).sum() / denom)

            loss_nll = h.new_zeros(())
            loss_calib = h.new_zeros(())
            loss_smooth = h.new_zeros(())

            if tdErrorCurr is not None:
                err = tdErrorCurr.detach()
                if donePrev is not None:
                    err = err * (1.0 - donePrev.float()).clamp(0.0, 1.0)

                loss_nll = 0.5 * (err.pow(2) / sigma2_ale + log_sigma2).mean()

                calib_tgt = torch.log(err.pow(2) + self.eps)
                calib_pred = torch.log(unc2 + self.eps)
                loss_calib = F.mse_loss(calib_pred, calib_tgt)

                loss_smooth = F.mse_loss(unc_total, u_prior.detach())

            loss_unc = (self.w_ens * loss_ens
                        + self.w_nll * loss_nll
                        + self.w_calib * loss_calib
                        + self.w_smooth * loss_smooth)

        comps = {
            "u_prior": u_prior,
            "log_sigma2": log_sigma2,
            "sigma2_ale": sigma2_ale,
            "sigma2_epi": sigma2_epi,
            "unc2": unc2,
            "ens_var": var_ens,
            "dis_ph": dis_ph,
            "unc_total": unc_total,}
        
        return unc_total, comps, loss_unc



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
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.SiLU(),)
        
        self.progress_head = nn.Linear(hidden, 1)

        mid = max(32, hidden // 2)

        self.entropy_from_h = nn.Sequential(
            nn.Linear(hidden, mid), 
            nn.SiLU(), 
            nn.Linear(mid, 1), 
            nn.Softplus())
        
        self.uncert_from_h = nn.Sequential(
            nn.Linear(hidden, mid), 
            nn.SiLU(), 
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

        B = stateCurr.size(0)
        self.UpdateStateEma(stateCurr)

        novelty = (stateCurr - self.state_ema).pow(2).mean(-1, keepdim=True).sqrt()

        h = self.affect_net(torch.cat([memoryPrev, attnPrev, stateCurr], dim=-1))

        if tdErrorPrev is not None:
            td_ref = tdErrorPrev.detach()
            prog_from_td = -td_ref.abs()
            prog_pred = torch.tanh(self.progress_head(h))
            alpha = 0.5
            progress = alpha * prog_from_td + (1.0 - alpha) * prog_pred
        else:
            progress = torch.tanh(self.progress_head(h))

        policyEntropyPrev = self.MaybeDropoutTeacher(policyEntropyPrev, B, stateCurr.device)
        uncertainty = self.MaybeDropoutTeacher(uncertainty, B, stateCurr.device)

        entropy_pred = self.entropy_from_h(h)
        uncert_pred = self.uncert_from_h(h)
        g_e = self.entropy_gate(h)
        g_u = self.uncert_gate(h)

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

        self.nov_ema.Update(novelty)
        self.prog_ema.Update(progress)
        self.ent_ema.Update(fused_entropy)
        self.unc_ema.Update(fused_uncert)

        novelty_n = self.nov_ema.Norm(novelty).clamp_(-8.0, 8.0)
        progress_n = self.prog_ema.Norm(progress).clamp_(-8.0, 8.0)
        entropy_n = self.ent_ema.Norm(fused_entropy).clamp_(-8.0, 8.0)
        uncert_n = self.unc_ema.Norm(fused_uncert).clamp_(-8.0, 8.0)

        r_int = ( self.alpha_novelty * novelty_n
                + self.alpha_progress * progress_n
                + self.alpha_entropy * entropy_n
                - self.alpha_uncert_penalty * uncert_n ).clamp(-self.r_clip, self.r_clip)

        exp_arg = (self.beta * uncert_n).clamp(-15.0, 15.0)
        temp_scale = self.tau0 * torch.exp(exp_arg)   # [B,1]

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

        e_t = torch.cat([temp_scale, lr_scale, gamma_mod], dim=-1)

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
            comps["reg_eT"] = self.eT_anchor * (
                (temp_scale - self.tau0).pow(2)
                + (lr_scale - self.lr0).pow(2)
                + (gamma_mod - self.gamma0).pow(2))

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

        self.a_net = nn.Sequential(nn.Linear(hDim, hDim//2), nn.SiLU(), nn.Linear(hDim//2, 1))
        self.b_net = nn.Sequential(nn.Linear(hDim, hDim//2), nn.SiLU(), nn.Linear(hDim//2, 1))
        self.g_net = nn.Sequential(nn.Linear(hDim, hDim//2), nn.SiLU(), nn.Linear(hDim//2, 1), nn.Sigmoid())

        self.epsA = epsA
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v_in = v
        a = F.softplus(self.a_net(h)) + self.epsA 
        b = self.b_net(h) 
        g = self.g_net(h)

        trop_in  = torch.cat([h, v_in], dim=-1)
        trop_out = self.trop(trop_in) 
        aff_out  = a * v + b 
        v_next_hat = g * trop_out + (1.0 - g) * aff_out

        extras = {
            "gate_trop": g, 
            "a": a, 
            "b": b,
            "trop_out": trop_out, 
            "aff_out": aff_out}
        
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
    def Reset(self, batchSize=1, device=None):
            dev = device if device is not None else self.prefix_G.device
            dt = self.prefix_G.dtype
            self.prefix_G.data = torch.ones(batchSize, device=dev, dtype=dt)
            self.prefix_C.data = torch.zeros(batchSize, device=dev, dtype=dt)
            self.anchors.clear()
            self._step = 0

    def Stack(self, key: str) -> Optional[torch.Tensor]:
        if len(self.anchors) == 0: return None
        return torch.stack([a[key] for a in self.anchors], dim=0)

    @torch.no_grad()
    def PreviewEdges(self, zNow: torch.Tensor, rNow: torch.Tensor, gNow: torch.Tensor, topk: Optional[int] = None) -> Dict[str, torch.Tensor]:

        device = zNow.device

        G_new = self.prefix_G * gNow
        C_new = self.prefix_C + self.prefix_G * rNow

        Z = self.Stack("z")
        if Z is None or Z.numel() == 0:
            return {"idx": torch.empty(0, dtype=torch.long, device=device),
                    "R": torch.empty(0, device=device),
                    "Gamma": torch.empty(0, device=device),
                    "v_hist": torch.empty(0, device=device),
                    "w": torch.empty(0, device=device),
                    "dist": torch.empty(0, device=device)}

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
        d_sel = d[idx]

        G_i = self.Stack("G")[idx]
        C_i = self.Stack("C")[idx]
        v_i = self.Stack("v")[idx]

        Gamma_seg = (G_new / (G_i + self.eps)).squeeze(-1)
        R_seg = ((C_new - C_i) / (G_i + self.eps)).squeeze(-1)

        w = torch.exp(-d_sel / max(self.dist_tau, 1e-6)) * (Gamma_seg.clamp_min(1e-6) ** self.len_power)

        return {"idx": idx, "R": R_seg, "Gamma": Gamma_seg, "v_hist": v_i.squeeze(-1), "w": w, "dist": d_sel}

    @torch.no_grad()
    def CommitStep(self, zNow: torch.Tensor, vNow: torch.Tensor, rNow: torch.Tensor, gNow: torch.Tensor):
        self.prefix_C.add_(self.prefix_G * rNow)
        self.prefix_G.mul_(gNow)
        anchor = {
            "z": zNow.detach(),
            "v": vNow.detach().unsqueeze(-1),
            "G": self.prefix_G.detach().clone(),
            "C": self.prefix_C.detach().clone(),}
        
        self.anchors.append(anchor)
        if len(self.anchors) > int(self.max_anchors):
            self.anchors.pop(0)
        self._step += 1



class EmotionCore(nn.Module):
    def __init__(
        self,
        *,
        stateDim: int,
        memoryDim: int,
        attnDim: int,
        emotionDim: int = 64,
        fastHidden: int = 128,
        slowHidden: int = 128,
        moodDecay: float = 0.95,
        useInternalGate: bool = True, 
        useHebbHead: bool = True):
        super().__init__()

        self.stateDim = stateDim
        self.memoryDim = memoryDim
        self.attnDim = attnDim
        self.emotionDim = emotionDim
        self.fastHidden = fastHidden
        self.slowHidden = slowHidden
        self.moodDecay = float(moodDecay)
        self.useInternalGate = useInternalGate
        self.useHebbHead = useHebbHead

        fast_in_dim = stateDim + attnDim
        
        self.fast_net = nn.Sequential(
            nn.Linear(fast_in_dim, fastHidden),
            nn.SiLU(),
            nn.Linear(fastHidden, fastHidden),
            nn.SiLU(),)
        
        if useHebbHead:
            self.fast_head = HebbianLinearFW(inFeatures=fastHidden,outFeatures=emotionDim, bias=True,initEta=1e-3, initLambda=0.05,cap=1.0,useOja=True,detachHebb=True,)
        else:
            self.fast_head = nn.Linear(fastHidden, emotionDim)

        slow_in_dim = stateDim + memoryDim + attnDim
        self.slow_cell = nn.LSTMCell(input_size=slow_in_dim, hidden_size=slowHidden)

        if useHebbHead:
            self.slow_head = HebbianLinearFW(inFeatures=slowHidden,outFeatures=emotionDim, bias=True,initEta=1e-3,initLambda=0.05, cap=1.0,useOja=True, detachHebb=True,)
        else:
            self.slow_head = nn.Linear(slowHidden, emotionDim)

        self.register_buffer("h", torch.zeros(1, slowHidden), persistent=False)
        self.register_buffer("c", torch.zeros(1, slowHidden), persistent=False)
        self.register_buffer("mood", torch.zeros(1, emotionDim), persistent=False)

        self.w_fast = nn.Parameter(torch.tensor(0.5))
        self.w_slow = nn.Parameter(torch.tensor(0.5))
        self.w_mood = nn.Parameter(torch.tensor(0.1))

        self.beta_fast_param = nn.Parameter(torch.tensor(0.0)) 
        self.beta_slow_param = nn.Parameter(torch.tensor(0.0))

        if useInternalGate:
            gate_in_dim = memoryDim + attnDim + stateDim
            hidden_gate = max(32, gate_in_dim // 2)
            self.gate_net = nn.Sequential(
                nn.Linear(gate_in_dim, hidden_gate),
                nn.SiLU(),
                nn.Linear(hidden_gate, emotionDim),
                nn.Sigmoid(),)
        else:
            self.gate_net = None


    def ResetParams(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.gate_net is not None:
            last = self.gate_net[-2]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.constant_(last.bias, 2.0) 

        with torch.no_grad():
            self.w_fast.fill_(0.5)
            self.w_slow.fill_(0.5)
            self.w_mood.fill_(0.1)

    @torch.no_grad()
    def ResetState(self, batchSize: int = 1, device: Optional[torch.device] = None):
        if device is None:
            device = self.h.device
        self.h = torch.zeros(batchSize, self.slowHidden, device=device)
        self.c = torch.zeros(batchSize, self.slowHidden, device=device)
        self.mood = torch.zeros(batchSize, self.emotionDim, device=device)


    def EnsureStateShape(self, B: int, device: torch.device):
        if self.h is None or self.h.size(0) != B or self.h.device != device:
            self.ResetState(B, device)

    def forward(
        self,
        memoryPrev: torch.Tensor, 
        attnPrev: torch.Tensor, 
        stateCurr: torch.Tensor,) -> torch.Tensor:

        B, device = stateCurr.size(0), stateCurr.device
        self.EnsureStateShape(B, device)

        h_prev = self.h
        c_prev = self.c
        mood_prev = self.mood

        fast_in = torch.cat([stateCurr, attnPrev], dim=-1)
        fast_h = self.fast_net(fast_in)

        if self.useHebbHead:
            beta_fast = torch.sigmoid(self.beta_fast_param)
            fast_raw, fast_extras = self.fast_head(fast_h, betaMix=beta_fast)
        else:
            fast_raw = self.fast_head(fast_h)

        emotion_fast = torch.tanh(fast_raw)

        slow_in = torch.cat([stateCurr, memoryPrev, attnPrev], dim=-1) 
        h_t, c_t = self.slow_cell(slow_in, (h_prev, c_prev)) 

        if self.useHebbHead:
            beta_slow = torch.sigmoid(self.beta_slow_param)
            slow_raw, slow_extras = self.slow_head(h_t, betaMix=beta_slow)
        else:
            slow_raw = self.slow_head(h_t)

        emotion_slow = torch.tanh(slow_raw)  

        decay = self.moodDecay
        mood_t = decay * mood_prev + (1.0 - decay) * emotion_slow  

        w_fast = F.softplus(self.w_fast)
        w_slow = F.softplus(self.w_slow)
        w_mood = F.softplus(self.w_mood)

        w_sum = (w_fast + w_slow + w_mood).clamp_min(1e-6)
        wf = w_fast / w_sum
        ws = w_slow / w_sum
        wm = w_mood / w_sum

        wf_b = wf.view(1, 1)
        ws_b = ws.view(1, 1)
        wm_b = wm.view(1, 1)

        emotion_raw = (wf_b * emotion_fast + ws_b * emotion_slow + wm_b * mood_t) 

        if self.gate_net is not None:
            gate_in = torch.cat([memoryPrev, attnPrev, stateCurr], dim=-1)  
            gate = self.gate_net(gate_in) 
        else:
            gate = torch.ones_like(emotion_raw)

        gate = gate.clamp(0.0, 1.0)
        emotion = torch.tanh(emotion_raw * (1.0 + gate))  

        self.h = h_t.detach()
        self.c = c_t.detach()
        self.mood = mood_t.detach()

        return emotion

    @torch.no_grad()
    def ResetHebbianMemory(self):
        if self.useHebbHead:
            self.slow_head.ResetHebbianMemory()
            self.fast_head.ResetHebbianMemory()


class GeoTropicalOut(NamedTuple):
    value: torch.Tensor
    tdError: torch.Tensor
    loss: torch.Tensor
    eT: torch.Tensor
    rInt: torch.Tensor
    emotion: torch.Tensor 
    rComps: Dict[str, torch.Tensor]
    uncertainty: torch.Tensor
    extras: Dict[str, torch.Tensor]



class ValueEstimationExtractor(nn.Module):
    def __init__(self,
        memoryDim: int = 768, attnDim: int = 1024, stateDim: int = 256, *,
        emotionDim: int = 64, 
        hidden: int = 2048, useLayerNorm: bool = True, irgKwargs: Optional[dict] = None,
        wExt: float = 1.0, wInt: float = 1.0, stopGradRGamma: bool = True,
        useSoftTrop: bool = True, tropTemp: float = 0.2, epsA: float = 1e-3,
        wTD: float = 1.0, wCycle: float = 1e-2,  wGlue1: float = 1e-2, 
        microMaxAnchors: int = 256, microTopK: int = 4, microDistTau: float = 0.5, microLenPower: float = 0.5,
        wEntropyTeacher: float = 1e-3,wGITScale: float = 1e-3, wGITShift: float = 1e-3, wGITSign: float = 1e-3,
        useHebb: bool = True, hebbCap: float = 1.0, hebbOja: bool = True, detachHebbGrad: bool = True,):
        super().__init__()

        if irgKwargs is None:
            irgKwargs = {"hidden": 1024}

        self.in_dim = memoryDim + attnDim + stateDim
        H = hidden

        self.use_hebb = useHebb
        self.wEntropyTeacher = wEntropyTeacher

        self.w_unc = 0.1

        self.fc1 = nn.Linear(self.in_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.norm1 = nn.LayerNorm(H) if useLayerNorm else None
        self.norm2 = nn.LayerNorm(H) if useLayerNorm else None

        self.hebb_value = HebbianLinearFW(H, 1, bias=True, initEta=1e-3, initLambda=0.1, cap=hebbCap, useOja=hebbOja, detachHebb=detachHebbGrad)
        self.value_head = nn.Linear(H, 1)

        self.fc1_adapter = GrowableLoRALinear(self.fc1)
        self.fc2_adapter = GrowableLoRALinear(self.fc2)
        self.value_adapter = GrowableLoRALinear(self.value_head)

        self.mix_gate = nn.Linear(H, 1)

        self.emotion_dim = emotionDim
        self.emotion_core = EmotionCore(stateDim=stateDim, memoryDim=memoryDim,attnDim=attnDim, emotionDim=emotionDim)

        self.wMixGateReg = 1e-3

        self.transport = TropicalAffineTransport(H, useSoftTrop, tropTemp, epsA)
        self.rgen = IntrinsicRewardGenerator(memoryDim, attnDim, stateDim, **(irgKwargs or {}))
        self.git = GITGaugeRegularizer(wScale=wGITScale, wShift=wGITShift, wSign=wGITSign)

        self.micro = TemporalMicroGraph(embDim=H, maxAnchors=microMaxAnchors, topk=microTopK, distTau=microDistTau, lenPower=microLenPower)

        self.wExt, self.wInt = wExt, wInt
        self.wTD = wTD
        self.wCycle = wCycle
        self.wGlue1 = wGlue1
        self.stopGrad_r_gamma = stopGradRGamma

        self._prev_vhat: Optional[torch.Tensor] = None
        self._prev_td: Optional[torch.Tensor] = None
        self._prev_r: Optional[torch.Tensor] = None
        self._prev_done: Optional[torch.Tensor] = None 
        self._prev_unc: Optional[torch.Tensor] = None

        self.unc_core = UncertaintyCore(hDim=H)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

        nn.init.zeros_(self.mix_gate.weight)
        nn.init.constant_(self.mix_gate.bias, -2.0)  

        self.emotion_core.ResetParams()

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
        done: Optional[torch.Tensor] = None) -> GeoTropicalOut:

        B = state.size(0)

        x = torch.cat([memory, attn, state], dim=-1)
        h = self.Trunk(x)

        emotion = self.emotion_core(memoryPrev=memory, attnPrev=attn, stateCurr=state)

        td_prev = self._prev_td
        r_prev = self._prev_r
        done_prev = self._prev_done
        unc_prev = self._prev_unc
        prev_vhat = self._prev_vhat

        v_param = self.value_adapter(h)

        irg_out = self.rgen(
            memoryPrev=memory,
            attnPrev=attn,
            stateCurr=state,
            policyEntropyPrev=policyEntropyPrev,
            uncertainty=(unc_prev.detach() if unc_prev is not None else None),
            tdErrorPrev=td_prev)
        
        r_int, eT, comps = irg_out.rInt.detach(), irg_out.eT, irg_out.components

        if self.use_hebb:
            hebb_eta = eT[..., 1:2].clamp_min(0).tanh() * 0.01
            hebb_lam = h.new_full((B, 1), 0.1)

            mix = torch.sigmoid(self.mix_gate(h)).clamp(1e-3, 1.0 - 1e-3)
            beta_mix = mix.detach()

            v_hebb, hebb_extras = self.hebb_value(h, eta=hebb_eta, lam=hebb_lam, betaMix=beta_mix)
            value = (1.0 - mix) * v_param + mix * v_hebb
        else:
            v_hebb = None
            hebb_extras = {"H_norm": h.new_zeros(())}
            mix = h.new_zeros((B, 1))
            value = v_param

        if rewardExt is None:
            r_used = self.wInt * r_int
        else:
            r_used = self.wExt * rewardExt + self.wInt * r_int

        gamma = eT[..., 2:3]
        if self.stopGrad_r_gamma:
            r_used = r_used.detach()
            gamma = gamma.detach()
        if done is not None:
            gamma = gamma * (1.0 - done.float())

        v_next_hat, transp_extras = self.transport(h, value)
        delta = r_used + gamma * v_next_hat - value

        unc_total, unc_comps, loss_unc = self.unc_core(
            h=h,
            valueParam=v_param,
            valueHebb=v_hebb,
            tdErrorCurr=delta,
            tdErrorPrev=td_prev,
            rewardPrev=r_prev,
            policyEntropyPrev=policyEntropyPrev,
            donePrev=done_prev)

        self._prev_td = delta.detach()
        self._prev_r = r_used.detach()
        self._prev_done = done.detach()
        self._prev_unc = unc_total.detach()

        if not self.training:
            return GeoTropicalOut(
                value=value,
                tdError=delta,
                loss=None,
                eT=eT,
                rInt=r_int,
                emotion=emotion,
                rComps=None,
                uncertainty=unc_total,
                extras=None,)
        
        self._prev_vhat = v_next_hat.mean().detach()
        
        loss_td = (delta ** 2).mean() * self.wTD

        edges = self.micro.PreviewEdges(
            zNow=h.detach().mean(dim=0) if B > 1 else h.detach().squeeze(0),
            rNow=r_used.mean().detach(),
            gNow=gamma.mean().detach())
        
        if edges["w"].numel() > 0:
            e_cycle = edges["R"] - (edges["v_hist"].detach() - edges["Gamma"] * value.mean())
            loss_cycle = ((edges["w"] * (e_cycle ** 2)).sum() / edges["w"].sum().clamp_min(1.0)) * self.wCycle
        else:
            loss_cycle = value.new_zeros(())

        if prev_vhat is not None:
            loss_glue1 = F.mse_loss(prev_vhat, value.mean()) * self.wGlue1
        else:
            loss_glue1 = value.new_zeros(())

        w_unc = float(getattr(self, "w_unc", 1.0))
        loss_unc = loss_unc * w_unc

        loss_ent = value.new_zeros(())
        if (policyEntropyPrev is not None) and (self.wEntropyTeacher > 0):
            e_pred = irg_out.components.get("entropy_pred", None)
            if e_pred is not None:
                m = torch.isfinite(policyEntropyPrev)
                if m.any():
                    loss_ent = F.mse_loss(e_pred[m], policyEntropyPrev[m]) * self.wEntropyTeacher

        loss_git = self.git(self.value_head, transp_extras, adapter=self.value_adapter)

        loss_mixgate = value.new_zeros(())
        if hasattr(self, "wMixGateReg") and self.wMixGateReg > 0:
            loss_mixgate = self.wMixGateReg * ((mix - 0.5) ** 2).mean()

        total_loss = (
            loss_td + loss_cycle + loss_glue1 + loss_unc + loss_git
            + comps.get("reg_gate", value.new_zeros(())).mean()
            + comps.get("reg_eT", value.new_zeros(())).mean()
            + loss_ent + loss_mixgate)

        alive = (done is None) or (done.float().mean() < 0.5)
        if alive:
            self.micro.CommitStep(
                zNow=h.detach().mean(dim=0) if B > 1 else h.detach().squeeze(0),
                vNow=value.mean().detach(),
                rNow=r_used.mean().detach(),
                gNow=gamma.mean().detach())

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

            "hebb_H_norm": hebb_extras.get("H_norm", value.new_zeros(())).detach(),

            "unc_total": unc_total.detach(),
            "unc_u_prior": unc_comps["u_prior"].detach(),
            "unc_sigma2_ale": unc_comps["sigma2_ale"].detach(),
            "unc_var_ens": unc_comps["ens_var"].detach(),
            "unc_dis_ph": unc_comps["dis_ph"].detach(),

            "micro_edges": torch.tensor(float(edges["w"].numel()), device=value.device),
            "mix_mean": mix.mean().detach(),
            "mix_gt_half": (mix > 0.5).float().mean().detach(),}

        return GeoTropicalOut(
            value=value,
            tdError=delta,
            loss=total_loss,
            eT=eT,
            rInt=r_int,
            emotion=emotion,
            rComps={k: v.detach() for k, v in comps.items()},
            uncertainty=unc_total,
            extras=extras,)

    @torch.no_grad()
    def ResetHebbianMemory(self):
        self.hebb_value.ResetHebbianMemory()
        self.emotion_core.ResetHebbianMemory()

    @torch.no_grad()
    def ResetState(self, batchSize):
        self.micro.Reset(batchSize=batchSize, device=None)
        self.emotion_core.ResetState(batchSize=batchSize)
        self._prev_vhat = None
        self._prev_td = None
        self._prev_r = None
        self._prev_done = None
        self._prev_unc = None

    @torch.no_grad()
    def ExportState(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}

        state["ve_is_training"] = bool(self.training)

        state["ve_prev_vhat"] = (None if self._prev_vhat is None else self._prev_vhat.detach().clone())
        state["ve_prev_td"] = (None if self._prev_td is None else self._prev_td.detach().clone())
        state["ve_prev_r"] = (None if self._prev_r is None else self._prev_r.detach().clone())
        state["ve_prev_done"] = (None if self._prev_done is None else self._prev_done.detach().clone())
        state["ve_prev_unc"] = (None if self._prev_unc is None else self._prev_unc.detach().clone())

        if hasattr(self, "hebb_value") and hasattr(self.hebb_value, "H"):
            state["hebb_H"] = self.hebb_value.H.detach().clone()

        if hasattr(self, "emotion_core"):
            ec = self.emotion_core
            if hasattr(ec, "h") and ec.h is not None:
                state["emo_h"] = ec.h.detach().clone()
            if hasattr(ec, "c") and ec.c is not None:
                state["emo_c"] = ec.c.detach().clone()
            if hasattr(ec, "mood") and ec.mood is not None:
                state["emo_mood"] = ec.mood.detach().clone()

        if hasattr(self, "unc_core"):
            uc = self.unc_core
            for name in ["td_ema", "r_ema", "ent_ema"]:
                if hasattr(uc, name):
                    ema = getattr(uc, name)
                    if hasattr(ema, "mean"): state[f"unc_{name}_mean"] = ema.mean.detach().clone()
                    if hasattr(ema, "var"): state[f"unc_{name}_var"] = ema.var.detach().clone()

        if hasattr(self, "rgen"):
            rg = self.rgen

            if hasattr(rg, "state_ema"):
                state["rgen_state_ema"] = rg.state_ema.detach().clone()

            for name in ["nov_ema", "prog_ema", "ent_ema", "unc_ema"]:
                if hasattr(rg, name):
                    ema = getattr(rg, name)
                    if hasattr(ema, "mean"): state[f"rgen_{name}_mean"] = ema.mean.detach().clone()
                    if hasattr(ema, "var"): state[f"rgen_{name}_var"] = ema.var.detach().clone()

        if hasattr(self, "micro"):
            mg = self.micro
            if hasattr(mg, "prefix_G"): state["micro_prefix_G"] = mg.prefix_G.detach().clone()
            if hasattr(mg, "prefix_C"): state["micro_prefix_C"] = mg.prefix_C.detach().clone()

            state["micro_step"] = int(getattr(mg, "_step", 0))

            anchors: List[Dict[str, torch.Tensor]] = getattr(mg, "anchors", [])
            n = len(anchors)

            state["micro_anchors_n"] = int(n)
            if n > 0:
                z_list = [a["z"].detach() for a in anchors]
                v_list = [a["v"].detach() for a in anchors] 
                G_list = [a["G"].detach() for a in anchors]
                C_list = [a["C"].detach() for a in anchors]

                state["micro_anchors_z"] = torch.stack(z_list, dim=0).clone()
                state["micro_anchors_v"] = torch.stack(v_list, dim=0).clone()
                state["micro_anchors_G"] = torch.stack(G_list, dim=0).clone()
                state["micro_anchors_C"] = torch.stack(C_list, dim=0).clone()

        return state

    @torch.no_grad()
    def ImportState(self, state: Dict[str, Any],):
        def need_(key: str) -> Any:
            if key in state:
                return state[key]
            return None

        if "ve_prev_vhat" in state: self._prev_vhat = state["ve_prev_vhat"]
        if "ve_prev_td" in state: self._prev_td = state["ve_prev_td"]
        if "ve_prev_r" in state: self._prev_r = state["ve_prev_r"]
        if "ve_prev_done" in state: self._prev_done = state["ve_prev_done"]
        if "ve_prev_unc" in state: self._prev_unc = state["ve_prev_unc"]

        if hasattr(self, "hebb_value") and hasattr(self.hebb_value, "H"):
            H = need_("hebb_H")
            if H is not None:
                H = H.to(device=self.hebb_value.H.device, dtype=self.hebb_value.H.dtype)
                if self.hebb_value.H.shape != H.shape:
                    self.hebb_value.H.resize_(H.shape).copy_(H)
                else:
                    self.hebb_value.H.copy_(H)

        if hasattr(self, "emotion_core"):
            ec = self.emotion_core
            if "emo_h" in state and state["emo_h"] is not None:
                h = state["emo_h"].to(device=ec.h.device, dtype=ec.h.dtype)
                ec.h = h.clone()
            if "emo_c" in state and state["emo_c"] is not None:
                c = state["emo_c"].to(device=ec.c.device, dtype=ec.c.dtype)
                ec.c = c.clone()
            if "emo_mood" in state and state["emo_mood"] is not None:
                m = state["emo_mood"].to(device=ec.mood.device, dtype=ec.mood.dtype)
                ec.mood = m.clone()

        if hasattr(self, "unc_core"):
            uc = self.unc_core
            for name in ["td_ema", "r_ema", "ent_ema"]:
                if hasattr(uc, name):
                    ema = getattr(uc, name)
                    k_mean = f"unc_{name}_mean"
                    k_var  = f"unc_{name}_var"
                    if k_mean in state and state[k_mean] is not None:
                        mean = state[k_mean].to(device=ema.mean.device, dtype=ema.mean.dtype)
                        if ema.mean.shape != mean.shape:
                            ema.mean.resize_(mean.shape).copy_(mean)
                        else:
                            ema.mean.copy_(mean)
                    if k_var in state and state[k_var] is not None:
                        var = state[k_var].to(device=ema.var.device, dtype=ema.var.dtype)
                        if ema.var.shape != var.shape:
                            ema.var.resize_(var.shape).copy_(var)
                        else:
                            ema.var.copy_(var)

        if hasattr(self, "rgen"):
            rg = self.rgen
            if "rgen_state_ema" in state and state["rgen_state_ema"] is not None and hasattr(rg, "state_ema"):
                s = state["rgen_state_ema"].to(device=rg.state_ema.device, dtype=rg.state_ema.dtype)
                if rg.state_ema.shape != s.shape:
                    rg.state_ema.resize_(s.shape).copy_(s)
                else:
                    rg.state_ema.copy_(s)

            for name in ["nov_ema", "prog_ema", "ent_ema", "unc_ema"]:
                if hasattr(rg, name):
                    ema = getattr(rg, name)
                    k_mean = f"rgen_{name}_mean"
                    k_var  = f"rgen_{name}_var"
                    if k_mean in state and state[k_mean] is not None:
                        mean = state[k_mean].to(device=ema.mean.device, dtype=ema.mean.dtype)
                        if ema.mean.shape != mean.shape:
                            ema.mean.resize_(mean.shape).copy_(mean)
                        else:
                            ema.mean.copy_(mean)
                    if k_var in state and state[k_var] is not None:
                        var = state[k_var].to(device=ema.var.device, dtype=ema.var.dtype)
                        if ema.var.shape != var.shape:
                            ema.var.resize_(var.shape).copy_(var)
                        else:
                            ema.var.copy_(var)

        if hasattr(self, "micro"):
            mg = self.micro

            if "micro_prefix_G" in state and state["micro_prefix_G"] is not None and hasattr(mg, "prefix_G"):
                G = state["micro_prefix_G"].to(device=mg.prefix_G.device, dtype=mg.prefix_G.dtype)
                if mg.prefix_G.shape != G.shape:
                    mg.prefix_G.resize_(G.shape).copy_(G)
                else:
                    mg.prefix_G.copy_(G)

            if "micro_prefix_C" in state and state["micro_prefix_C"] is not None and hasattr(mg, "prefix_C"):
                C = state["micro_prefix_C"].to(device=mg.prefix_C.device, dtype=mg.prefix_C.dtype)
                if mg.prefix_C.shape != C.shape:
                    mg.prefix_C.resize_(C.shape).copy_(C)
                else:
                    mg.prefix_C.copy_(C)

            if "micro_step" in state:
                mg._step = int(state["micro_step"])

            n = int(state.get("micro_anchors_n", 0))
            mg.anchors.clear()
            if n > 0:
                zA = need_("micro_anchors_z")
                vA = need_("micro_anchors_v")
                GA = need_("micro_anchors_G")
                CA = need_("micro_anchors_C")

                dev = mg.prefix_G.device if hasattr(mg, "prefix_G") else zA.device
                zA = zA.to(dev).detach()
                vA = vA.to(dev).detach()
                GA = GA.to(dev).detach()
                CA = CA.to(dev).detach()

                for i in range(n):
                    mg.anchors.append({
                        "z": zA[i].clone(),
                        "v": vA[i].clone(),
                        "G": GA[i].clone(),
                        "C": CA[i].clone(),})


class ValueEstimationOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: nn.Module,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankFc1: int = 128,
        maxRankFc2: int = 128,
        maxRankVHead: int = 64,):
        self.maxRankFc1 = int(maxRankFc1)
        self.maxRankFc2 = int(maxRankFc2)
        self.maxRankVHead = int(maxRankVHead)
        super().__init__(base, initRankEach=initRankEach, autoRank=autoRank, evThreshold=evThreshold, gradEma=gradEma)

    @staticmethod
    def LinearWithDelta(
        layer: nn.Linear,
        x: torch.Tensor,
        delta_mat: Optional[torch.Tensor] = None,
        base_adapter: Optional[nn.Module] = None,) -> torch.Tensor:
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
        if isinstance(x, dict) and all(k in x for k in ("memory", "attn", "state")):
            return x["memory"], x["attn"], x["state"]
        raise TypeError("ValueEstimationOnlineWrapper expects x as (memory, attn, state) or dict with those keys.")

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        base = self.base
        assert hasattr(base, "fc1") and hasattr(base, "fc2") and hasattr(base, "value_head")

        H = int(base.value_head.in_features)
        Din = int(base.fc1.in_features)
        L = 2 

        def alloc(addRank: int, inDim: int, outDim: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, inDim, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype))
            s = nn.Parameter(torch.as_tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return s * (b @ a)

        return {
            "fc1": SiteSpec("fc1", L, Din, H, self.maxRankFc1, lambda r, dv, dt: alloc(r, Din, H, dv, dt), compose),
            "fc2": SiteSpec("fc2", L, H, H, self.maxRankFc2, lambda r, dv, dt: alloc(r, H, H, dv, dt), compose),
            "vhead": SiteSpec("vhead", L, H, 1, self.maxRankVHead, lambda r, dv, dt: alloc(r, H, 1, dv, dt), compose),}

    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None, 
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        **kwargs,):
        base: ValueEstimationExtractor = self.base 

        rewardExt = kwargs.get("rewardExt", None)
        policyEntropyPrev = kwargs.get("policyEntropyPrev", None)
        done = kwargs.get("done", None)

        memory, attn, state = self.EnsureInputs(x)
        B = state.size(0)

        if deltasPerLayer is None:
            deltasPerLayer = [{}, {}]
        else:
            if len(deltasPerLayer) < 2:
                deltasPerLayer = list(deltasPerLayer) + [{} for _ in range(2 - len(deltasPerLayer))]

        d0 = deltasPerLayer[0] or {}
        d1 = deltasPerLayer[1] or {}

        x_cat = torch.cat([memory, attn, state], dim=-1)

        h = self.LinearWithDelta(
            base.fc1,
            x_cat,
            delta_mat=d0.get("fc1", None),
            base_adapter=getattr(base, "fc1_adapter", None),)
        
        h = F.gelu(h)
        if base.norm1 is not None:
            h = base.norm1(h)

        h = self.LinearWithDelta(
            base.fc2,
            h,
            delta_mat=d1.get("fc2", None),
            base_adapter=getattr(base, "fc2_adapter", None),)
        
        h = F.gelu(h)
        if base.norm2 is not None:
            h = base.norm2(h)

        emotion = base.emotion_core(memoryPrev=memory, attnPrev=attn, stateCurr=state)

        td_prev = tdError if tdError is not None else base._prev_td
        r_prev = base._prev_r
        done_prev = base._prev_done
        unc_prev = uncertainty if uncertainty is not None else base._prev_unc
        prev_vhat = base._prev_vhat

        v_param = self.LinearWithDelta(
            base.value_head,
            h,
            delta_mat=d1.get("vhead", None),
            base_adapter=getattr(base, "value_adapter", None),) 

        irg_out = base.rgen(
            memoryPrev=memory,
            attnPrev=attn,
            stateCurr=state,
            policyEntropyPrev=policyEntropyPrev,
            uncertainty=(unc_prev.detach() if (unc_prev is not None) else None),
            tdErrorPrev=(td_prev.detach() if (td_prev is not None) else None),)
        
        r_int, eT, comps = irg_out.rInt.detach(), irg_out.eT, irg_out.components 

        if base.use_hebb:
            hebb_eta = eT[..., 1:2].clamp_min(0).tanh() * 0.01 
            hebb_lam = h.new_full((B, 1), 0.1) 

            mix = torch.sigmoid(base.mix_gate(h)).clamp(1e-3, 1.0 - 1e-3) 
            beta_mix = mix.detach()

            v_hebb, hebb_extras = base.hebb_value(h, eta=hebb_eta, lam=hebb_lam, betaMix=beta_mix) 
            value = (1.0 - mix) * v_param + mix * v_hebb
        else:
            v_hebb = None
            hebb_extras = {"H_norm": h.new_zeros(())}
            mix = h.new_zeros((B, 1))
            value = v_param

        if rewardExt is None:
            r_used = base.wInt * r_int
        else:
            r_used = base.wExt * rewardExt + base.wInt * r_int

        gamma = eT[..., 2:3]
        if base.stopGrad_r_gamma:
            r_used = r_used.detach()
            gamma = gamma.detach()
        if done is not None:
            gamma = gamma * (1.0 - done.float())

        v_next_hat, transp_extras = base.transport(h, value) 
        delta = r_used + gamma * v_next_hat - value  

        unc_total, unc_comps, loss_unc = base.unc_core(
            h=h,
            valueParam=v_param,
            valueHebb=v_hebb,
            tdErrorCurr=delta,
            tdErrorPrev=td_prev,
            rewardPrev=r_prev,
            policyEntropyPrev=policyEntropyPrev,
            donePrev=done_prev,)

        base._prev_td = delta.detach()
        base._prev_r = r_used.detach()
        base._prev_done = done.detach()
        base._prev_unc = unc_total.detach()

        if not base.training:
            return GeoTropicalOut(
                value=value,
                tdError=delta,
                loss=None,
                eT=eT,
                rInt=r_int,
                emotion=emotion,
                rComps=None,
                uncertainty=unc_total,
                extras=None,)
        
        base._prev_vhat = v_next_hat.mean().detach()

        loss_td = (delta ** 2).mean() * base.wTD

        edges = base.micro.PreviewEdges(
            zNow=h.detach().mean(dim=0) if B > 1 else h.detach().squeeze(0),
            rNow=r_used.mean().detach(),
            gNow=gamma.mean().detach(),)
        
        if edges["w"].numel() > 0:
            e_cycle = edges["R"] - (edges["v_hist"].detach() - edges["Gamma"] * value.mean())
            loss_cycle = ((edges["w"] * (e_cycle ** 2)).sum() / edges["w"].sum().clamp_min(1.0)) * base.wCycle
        else:
            loss_cycle = value.new_zeros(())

        if prev_vhat is not None:
            loss_glue1 = F.mse_loss(prev_vhat, value.mean()) * base.wGlue1
        else:
            loss_glue1 = value.new_zeros(())

        loss_unc = loss_unc * float(getattr(base, "w_unc", 1.0))

        loss_ent = value.new_zeros(())
        if (policyEntropyPrev is not None) and (base.wEntropyTeacher > 0):
            e_pred = comps.get("entropy_pred", None)
            if e_pred is not None:
                m = torch.isfinite(policyEntropyPrev)
                if m.any():
                    loss_ent = F.mse_loss(e_pred[m], policyEntropyPrev[m]) * base.wEntropyTeacher

        loss_git = base.git(base.value_head, transp_extras, adapter=getattr(base, "value_adapter", None))

        loss_mixgate = value.new_zeros(())
        if getattr(base, "wMixGateReg", 0.0) > 0:
            loss_mixgate = base.wMixGateReg * ((mix - 0.5) ** 2).mean()

        total_loss = (
            loss_td + loss_cycle + loss_glue1 + loss_unc + loss_git
            + comps.get("reg_gate", value.new_zeros(())).mean()
            + comps.get("reg_eT", value.new_zeros(())).mean()
            + loss_ent + loss_mixgate)

        alive = (done is None) or (done.float().mean() < 0.5)
        if alive:
            base.micro.CommitStep(
                zNow=h.detach().mean(dim=0) if B > 1 else h.detach().squeeze(0),
                vNow=value.mean().detach(),
                rNow=r_used.mean().detach(),
                gNow=gamma.mean().detach(),)

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

            "hebb_H_norm": hebb_extras.get("H_norm", value.new_zeros(())).detach(),

            "unc_total": unc_total.detach(),
            "unc_u_prior": unc_comps["u_prior"].detach(),
            "unc_sigma2_ale": unc_comps["sigma2_ale"].detach(),
            "unc_var_ens": unc_comps["ens_var"].detach(),
            "unc_dis_ph": unc_comps["dis_ph"].detach(),

            "micro_edges": torch.tensor(float(edges["w"].numel()), device=value.device),
            "mix_mean": mix.mean().detach(),
            "mix_gt_half": (mix > 0.5).float().mean().detach(),}

        return GeoTropicalOut(
            value=value,
            tdError=delta,
            loss=total_loss,
            eT=eT,
            rInt=r_int,
            emotion=emotion,
            rComps={k: v.detach() for k, v in comps.items()},
            uncertainty=unc_total,
            extras=extras,)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        mapping = {
            "fc1": ("fc1_adapter", "fc1", [0]),
            "fc2": ("fc2_adapter", "fc2", [1]),
            "vhead": ("value_adapter", "value_head", [1]),}

        if site not in mapping:
            return False
        attr_name, tgt_name, allow_layers = mapping[site]
        if layerIdx not in allow_layers:
            return False

        target: nn.Linear = getattr(self.base, tgt_name)

        if (not hasattr(self.base, attr_name)) or (not isinstance(getattr(self.base, attr_name), nn.Module)):
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
        self.hidden = 512

    def MakeEstimatorHebb(self, **overrides):
        est = ValueEstimationExtractor(
            memoryDim=self.mem_dim,
            attnDim=self.attn_dim,
            stateDim=self.state_dim,
            hidden=self.hidden,
            useLayerNorm=True,
            useHebb=True,
            irgKwargs={"teacherDropoutProb": 0.0},
            **overrides).to(self.device)
        
        est.train()
        return est

    def RandBatch(self, B: int = 3):
        mem = torch.randn(B, self.mem_dim, device=self.device)
        attn = torch.randn(B, self.attn_dim, device=self.device)
        state = torch.randn(B, self.state_dim, device=self.device)
        return mem, attn, state

    def Done(self, B: int, ones: bool = False):
        if ones:
            return torch.ones(B, 1, device=self.device)
        return torch.zeros(B, 1, device=self.device)

    def Reward(self, B: int):
        return torch.randn(B, 1, device=self.device).clamp(-1, 1)

    def Entropy(self, B: int):
        return torch.rand(B, 1, device=self.device)

    def ParamIds(self, module: nn.Module):
        return {id(p) for p in module.parameters() if p.requires_grad}


    def TestLoRAForwardEquivalence(self) -> bool:
        try:
            torch.manual_seed(42)
            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True, useHebb=False).to(self.device)
            
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
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True, useHebb=True,
                irgKwargs={"teacherDropoutProb": 0.0},).to(self.device)
            
            est.train()

            for ad in [est.fc1_adapter, est.fc2_adapter, est.value_adapter]:
                ad.Grow(2, init=None, freezeOld=True)

            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            reward_ext = self.Reward(B)
            entropy_prev = self.Entropy(B)

            out = est(
                memory=mem, attn=attn, state=state,
                rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                done=done,)

            opt.zero_grad(set_to_none=True)
            out.loss.backward()

            ok = True
            msgs = []

            def finite(t):
                return torch.is_tensor(t) and torch.isfinite(t).all().item()

            def check_ad(name, ad, target_layer):
                nonlocal ok
                if len(ad.A_list) == 0:
                    ok = False
                    msgs.append(f"{name} has no ranks")
                    return
                A = ad.A_list[-1]
                Bm = ad.B_list[-1]
                s = ad.alpha[-1]
                for tag, p in [("A", A), ("B", Bm), ("alpha", s)]:
                    if (p.grad is None) or (not finite(p.grad)):
                        ok = False
                        msgs.append(f"{name}.{tag} grad missing/non-finite")
                if (target_layer.weight.grad is None) or (not finite(target_layer.weight.grad)):
                    ok = False
                    msgs.append(f"{name}.target.weight grad missing/non-finite")

            check_ad("fc1_adapter", est.fc1_adapter, est.fc1)
            check_ad("fc2_adapter", est.fc2_adapter, est.fc2)
            check_ad("value_adapter", est.value_adapter, est.value_head)

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
            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True, useHebb=False).to(self.device)
            
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
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True, useHebb=True,
                irgKwargs={"teacherDropoutProb": 0.0},).to(self.device)

            est.train()

            for ad in [est.fc1_adapter, est.fc2_adapter, est.value_adapter]:
                ad.Grow(2, init=None, freezeOld=True)

            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            with torch.no_grad():
                before = {n: p.detach().clone()
                        for n, p in est.named_parameters()
                        if p.requires_grad and p.data.numel() > 0}

            reward_ext = self.Reward(B)
            entropy_prev = self.Entropy(B)

            for step in range(2):
                mem, attn, state = self.RandBatch(B)

                out = est(
                    memory=mem, attn=attn, state=state,
                    rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                    done=done,)

                loss = out.loss
                if out.emotion is not None:
                    loss = loss + 0.01 * (out.emotion ** 2).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()

                missing = []
                for n, p in est.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        if (p.grad is None) or (not torch.isfinite(p.grad).all().item()):
                            missing.append(n)
                if missing:
                    print("[AllTrainable] missing/non-finite grads:")
                    for n in missing:
                        print("  -", n)
                    return False

                torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
                opt.step()

            IGNORE_UNCHANGED = {"emotion_core.beta_fast_param", "emotion_core.beta_slow_param",}

            unchanged_non_ign = []
            with torch.no_grad():
                for n, p in est.named_parameters():
                    if (p.requires_grad and p.data.numel() > 0 and n in before and p.data.shape == before[n].shape):
                        if torch.allclose(p.data, before[n], atol=0.0, rtol=0.0):
                            if n not in IGNORE_UNCHANGED:
                                unchanged_non_ign.append(n)

            if unchanged_non_ign:
                print("[AllTrainable] unchanged after 2 steps (NOT whitelisted):")
                for n in unchanged_non_ign:
                    print("  -", n)
                return False

            print("AllTrainableParamsHaveGradAndStep pass")
            return True

        except Exception as e:
            print(f"AllTrainableParamsHaveGradAndStep error: {e}")
            return False


    def TestEmotionSecondStepGrad(self) -> bool:
        try:
            torch.manual_seed(999)
            B = 8
            done = self.Done(B)

            est = self.MakeEstimatorHebb()
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            mem, attn, state = self.RandBatch(B)
            out1 = est(
                memory=mem, attn=attn, state=state,
                rewardExt=self.Reward(B),
                policyEntropyPrev=self.Entropy(B),
                done=done,)
            
            loss1 = out1.loss + 0.01 * (out1.emotion ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss1.backward()
            opt.step()

            mem2, attn2, state2 = self.RandBatch(B)
            out2 = est(
                memory=mem2, attn=attn2, state=state2,
                rewardExt=self.Reward(B),
                policyEntropyPrev=self.Entropy(B),
                done=done,)
            
            loss2 = out2.loss + 0.01 * (out2.emotion ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss2.backward()

            target_names = {
                "emotion_core.beta_fast_param",
                "emotion_core.beta_slow_param",
                "emotion_core.slow_cell.weight_hh",}

            bad = []
            for n, p in est.named_parameters():
                if n in target_names:
                    if (p.grad is None) or (not torch.isfinite(p.grad).all()):
                        bad.append(f"{n}: grad None/NaN")
                    else:
                        gmax = p.grad.abs().max().item()
                        if gmax < 1e-12:
                            bad.append(f"{n}: grad too small ({gmax:.3e})")

            if bad:
                print("EmotionSecondStepGrad failed:")
                for msg in bad:
                    print("  -", msg)
                return False

            print("EmotionSecondStepGrad pass")
            return True
        except Exception as e:
            print(f"EmotionSecondStepGrad error: {e}")
            return False


    def TestGlue1TriggersOnSecondStep(self) -> bool:
        try:
            torch.manual_seed(303)
            B = 6
            mem, attn, state = self.RandBatch(B)
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device)
            
            est.train()
            est.rgen.teacher_dropout_prob = 0.0

            reward_ext = self.Reward(B)
            entropy_prev = self.Entropy(B)

            out1 = est(
                memory=mem, attn=attn, state=state,
                rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                done=done,)
            
            g1 = float(out1.extras["loss_glue1"].item())

            out2 = est(
                memory=mem, attn=attn, state=state,
                rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                done=done,)
            
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

            entropy_prev = self.Entropy(B)
            uncert = F.softplus(torch.randn(B, 1, device=self.device))
            td_prev = torch.randn(B, 1, device=self.device) * 0.1

            out = irg(
                mem, attn, state,
                policyEntropyPrev=entropy_prev,
                uncertainty=uncert,
                tdErrorPrev=td_prev,)

            ok = True
            ok &= (out.rInt.shape == (B, 1))
            ok &= (out.eT.shape == (B, 3))
            needed = ["novelty", "progress", "entropy", "uncertainty",
                      "novelty_n", "progress_n", "entropy_n", "uncertainty_n", "valence"]
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
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device)
            
            est.train()
            est.rgen.teacher_dropout_prob = 0.0

            entropy_prev = self.Entropy(B)
            out = est(
                memory=mem, attn=attn, state=state,
                rewardExt=None, policyEntropyPrev=entropy_prev,
                done=done,)

            ok = True
            ok &= (out.value.shape == (B, 1))
            ok &= (out.tdError.shape == (B, 1))
            ok &= (out.eT.shape == (B, 3))
            ok &= (out.rInt.shape == (B, 1))
            ok &= torch.isfinite(out.loss).all()

            with torch.no_grad():
                r_used = est.wInt * out.rInt
                gamma = out.eT[..., 2:3] * (1.0 - done)
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
            done = (torch.randint(0, 2, (B, 1), device=self.device).float())

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=False, wExt=1.0, wInt=1.0).to(self.device)

            est.eval()
            est.rgen.teacher_dropout_prob = 0.0

            reward_ext = self.Reward(B)
            entropy_prev = self.Entropy(B)

            out = est(
                memory=mem, attn=attn, state=state,
                rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                done=done,)

            ok = True
            ok &= (out.value.shape == (B, 1))
            ok &= (out.uncertainty.shape == (B, 1))
            ok &= (out.eT.shape == (B, 3))

            with torch.no_grad():
                r_used = est.wExt * reward_ext + est.wInt * out.rInt
                gamma = out.eT[..., 2:3] * (1.0 - done)
                x = torch.cat([mem, attn, state], dim=-1)
                h = est.Trunk(x)
                vhat, _ = est.transport(h, out.value)
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
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device)
            
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            out = est(
                memory=mem, attn=attn, state=state,
                rewardExt=self.Reward(B),
                policyEntropyPrev=self.Entropy(B),
                done=done,)

            loss = out.loss + 0.01 * (out.emotion ** 2).mean()
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
            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device)
            
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            for t in range(steps):
                B = 8
                mem, attn, state = self.RandBatch(B)
                done = self.Done(B)

                out = est(
                    memory=mem, attn=attn, state=state,
                    rewardExt=self.Reward(B),
                    policyEntropyPrev=self.Entropy(B),
                    done=done,)

                opt.zero_grad(set_to_none=True)
                out.loss.backward()

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
            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device)
            
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
                done = self.Done(B)

                out = est(
                    memory=mem, attn=attn, state=state,
                    rewardExt=self.Reward(B),
                    policyEntropyPrev=self.Entropy(B),
                    done=done,)

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
            print(f"GeoTropical ParamsActuallyChange {'passed' if ok else 'failed'} (delta={delta:.3e}).")
            return ok
        except Exception as e:
            print(f"GeoTropical ParamsActuallyChange error: {e}")
            return False

    def TestLossDecreases(self, steps: int = 120, batch_size: int = 16) -> bool:
        try:
            torch.manual_seed(2025)
            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True, wExt=1.0, wInt=0.1).to(self.device)
            
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            opt = torch.optim.Adam(est.parameters(), lr=1e-3)

            losses = []
            for t in range(steps):
                mem = torch.randn(batch_size, self.mem_dim, device=self.device)
                attn = torch.randn(batch_size, self.attn_dim, device=self.device)
                state = torch.randn(batch_size, self.state_dim, device=self.device)
                done = torch.zeros(batch_size, 1, device=self.device)

                out = est(
                    memory=mem, attn=attn, state=state,
                    rewardExt=self.Reward(batch_size),
                    policyEntropyPrev=self.Entropy(batch_size),
                    done=done,)

                opt.zero_grad(set_to_none=True)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
                opt.step()

                losses.append(float(out.loss.detach().item()))
                if (t + 1) % max(1, steps // 4) == 0:
                    print(f"[GeoTropTrain] step {t+1}/{steps} | loss={losses[-1]:.6f}")

            half = len(losses) // 2
            q4 = losses[-(len(losses) // 4):]
            med_q4 = stats.median(q4)
            max_first_half = max(losses[:half])
            ok1 = med_q4 <= 0.3 * max_first_half

            mid = losses[half // 2: half + half // 2]
            tail = losses[-(len(losses) // 3):]
            ok2 = (sum(tail) / len(tail)) < (sum(mid) / len(mid))

            ok = ok1 or ok2
            print(f"GeoTropical TestLossDecreases {'passed' if ok else 'failed'} "
                  f"(median_last_quarter={med_q4:.4f}, max_first_half={max_first_half:.4f})")
            return ok
        except Exception as e:
            print(f"GeoTropical TestLossDecreases error: {e}")
            return False

    def TestHebbMemoryUpdates(self) -> bool:
        try:
            torch.manual_seed(7)
            B = 6
            mem, attn, state = self.RandBatch(B)
            done = self.Done(B)

            est = self.MakeEstimatorHebb()
            H0 = est.hebb_value.H.detach().clone()

            _ = est(
                memory=mem, attn=attn, state=state,
                rewardExt=None,
                policyEntropyPrev=self.Entropy(B),
                done=done,)

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
            done = self.Done(B, ones=True)

            est_ref = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device).eval()
            
            est_ref.rgen.teacher_dropout_prob = 0.0
            est_ref.ResetState(batchSize=B)
            est_ref.ResetHebbianMemory()

            est_wrapped = copy.deepcopy(est_ref).to(self.device).eval()
            wrapper = ValueEstimationOnlineWrapper(est_wrapped, initRankEach=0, autoRank=False)

            reward_ext = self.Reward(B)
            entropy_prev = self.Entropy(B)

            out_ref = est_ref(
                memory=mem, attn=attn, state=state,
                rewardExt=reward_ext, policyEntropyPrev=entropy_prev,
                done=done,)

            out_wr = wrapper.ForwardWithDeltas(
                x=(mem, attn, state),
                keyPaddingMask=None,
                tdError=None,
                uncertainty=None,
                deltasPerLayer=[{}, {}],
                rewardExt=reward_ext,
                policyEntropyPrev=entropy_prev,
                done=done,)

            atol, rtol = 1e-6, 1e-5
            ok = (
                torch.allclose(out_wr.value, out_ref.value, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.tdError, out_ref.tdError, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.eT, out_ref.eT, atol=atol, rtol=rtol) and
                torch.allclose(out_wr.uncertainty, out_ref.uncertainty, atol=atol, rtol=rtol))

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
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True, useHebb=False).to(self.device).eval()
            
            est.rgen.teacher_dropout_prob = 0.0
            wrapper = ValueEstimationOnlineWrapper(est, initRankEach=0, autoRank=False)

            reward_ext = None
            entropy_prev = self.Entropy(B)

            H = int(est.value_head.in_features)
            delta_v = (torch.randn(1, H, device=self.device) * 1e-3)

            out_sim = wrapper.ForwardWithDeltas(
                x=(mem, attn, state),
                keyPaddingMask=None,
                tdError=None,
                uncertainty=None,
                deltasPerLayer=[{"fc1": None}, {"fc2": None, "vhead": delta_v}],
                rewardExt=reward_ext,
                policyEntropyPrev=entropy_prev,
                done=done,)

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

            est.ResetState(batchSize=B)
            out_after = wrapper.ForwardWithDeltas(
                x=(mem, attn, state),
                keyPaddingMask=None,
                tdError=None,
                uncertainty=None,
                deltasPerLayer=[{"fc1": None}, {"fc2": None, "vhead": None}],
                rewardExt=reward_ext,
                policyEntropyPrev=entropy_prev,
                done=done,)

            atol, rtol = 5e-6, 1e-4
            ok = (
                torch.allclose(out_after.value, out_sim.value, atol=atol, rtol=rtol) and
                torch.allclose(out_after.tdError, out_sim.tdError, atol=atol, rtol=rtol) and
                torch.allclose(out_after.eT, out_sim.eT, atol=atol, rtol=rtol) and
                torch.allclose(out_after.uncertainty, out_sim.uncertainty, atol=atol, rtol=rtol))

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
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device)
            
            est.train()
            est.rgen.teacher_dropout_prob = 0.0
            wrapper = ValueEstimationOnlineWrapper(est, initRankEach=0, autoRank=False)

            with torch.no_grad():
                W_fc1_0 = est.fc1.weight.clone()
                W_fc2_0 = est.fc2.weight.clone()
                W_vh_0 = est.value_head.weight.clone()

            d_fc1 = nn.Parameter(torch.zeros_like(est.fc1.weight))
            d_fc2 = nn.Parameter(torch.zeros_like(est.fc2.weight))
            d_vh = nn.Parameter(torch.zeros_like(est.value_head.weight))

            opt = torch.optim.Adam([d_fc1, d_fc2, d_vh], lr=1e-1)

            entropy_prev = self.Entropy(B)
            reward_ext = self.Reward(B)

            for _ in range(5):
                out = wrapper.ForwardWithDeltas(
                    x=(mem, attn, state),
                    keyPaddingMask=None,
                    tdError=None,
                    uncertainty=None,
                    deltasPerLayer=[{"fc1": d_fc1}, {"fc2": d_fc2, "vhead": d_vh}],
                    rewardExt=reward_ext,
                    policyEntropyPrev=entropy_prev,
                    done=done,)
                
                loss = out.loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                ok_base_unchanged = (
                    torch.allclose(est.fc1.weight, W_fc1_0, atol=0, rtol=0) and
                    torch.allclose(est.fc2.weight, W_fc2_0, atol=0, rtol=0) and
                    torch.allclose(est.value_head.weight, W_vh_0, atol=0, rtol=0))

            delta_change = (d_fc1.detach().abs().mean() + d_fc2.detach().abs().mean() + d_vh.detach().abs().mean()).item()
            ok_delta_changed = delta_change > 0

            ok = ok_base_unchanged and ok_delta_changed
            print(f"WrapperTempDeltasTrainable {'pass' if ok else 'fail'} (Δ_abs_mean_sum={delta_change:.3e})")
            return ok
        except Exception as e:
            print(f"WrapperTempDeltasTrainable error: {e}")
            return False

    def TestGradFlowCoverage(self) -> bool:
        try:
            torch.manual_seed(17)
            B = 8
            mem, attn, state = self.RandBatch(B)
            done = self.Done(B)

            est = ValueEstimationExtractor(
                memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,
                hidden=self.hidden, useLayerNorm=True).to(self.device)
            
            est.train()
            est.rgen.teacher_dropout_prob = 0.0

            out = est(
                memory=mem, attn=attn, state=state,
                rewardExt=self.Reward(B),
                policyEntropyPrev=self.Entropy(B),
                done=done,)

            out.loss.backward()

            keys = [
                "fc1.weight", "fc2.weight",
                "value_head.weight",
                "transport.trop.W",
                "transport.a_net.0.weight",
                "transport.b_net.0.weight",
                "transport.g_net.0.weight",
                "unc_core.logvar_head.weight",]
            
            ok = True
            bad = []
            for name, p in est.named_parameters():
                if any(name.endswith(k) for k in keys):
                    if (p.grad is None) or (not torch.isfinite(p.grad).all()):
                        ok = False
                        bad.append(name)
            if not ok:
                print("GradFlowCoverage failed:", bad)
            else:
                print("GradFlowCoverage pass")
            return ok
        except Exception as e:
            print(f"GradFlowCoverage error: {e}")
            return False


    def RunAll(self):
        results = {
            "LoRAForwardEquivalence": self.TestLoRAForwardEquivalence(),
            "LoRAParamsGrad": self.TestLoRAParamsGrad(),
            "LoRAFreezeOld": self.TestLoRAFreezeOld(),
            "AllTrainableParamsHaveGradAndStep": self.TestAllTrainableParamsHaveGradAndStep(),
            "EmotionSecondStepGrad": self.TestEmotionSecondStepGrad(),
            "Glue1TriggersOnSecondStep": self.TestGlue1TriggersOnSecondStep(),
            "WrapperAlignmentNoDelta": self.TestWrapperAlignmentNoDelta(),
            "SimThenCommitVHead": self.TestSimThenCommitVHead(),
            "WrapperTempDeltasTrainable": self.TestWrapperTempDeltasTrainable(),
            "GradFlowCoverage": self.TestGradFlowCoverage(),
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
