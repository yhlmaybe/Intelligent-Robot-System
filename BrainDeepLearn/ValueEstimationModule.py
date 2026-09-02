from __future__ import annotations
from collections import deque
from typing import Optional, Dict, NamedTuple, Tuple, List, Any
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics as stats
from torch.func import functional_call as torch_functional_call
from FunctionTools import SiteSpec, BaseOnlineWrapper, AGICoreModule, GrowableLoRALinear, GetParametersScale

COGNITIVE_COMPUTE_REASON_COUNT = 7


ONTOLOGY_REALM_SELF = 0
ONTOLOGY_REALM_EXTERNAL = 1
ONTOLOGY_REALM_VIRTUAL = 2
ONTOLOGY_REALM_EFFECT = 3
ONTOLOGY_MOTION_CARRIER = 1
ONTOLOGY_MOTION_ARTICULATION = 2
ONTOLOGY_MOTION_SURFACE = 3
ONTOLOGY_AGENCY_EXTERNAL = 1
ONTOLOGY_AGENCY_AUTONOMOUS = 2
ONTOLOGY_AGENCY_MIXED = 3


class HebbianLinearFW(AGICoreModule):
    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,):
        super().__init__()
        self.in_f = int(inFeatures)
        self.out_f = int(outFeatures)

        self.weight = nn.Parameter(torch.empty(self.out_f, self.in_f))
        self.bias = nn.Parameter(torch.zeros(self.out_f))

        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)

        self.register_buffer(
            "H",
            torch.zeros(1, self.out_f, self.in_f),
            persistent=False)

    @torch.no_grad()
    def EnsureB(self, B: int):
        if self.H.size(0) != B:
            self.H = self.H.new_zeros(B, self.out_f, self.in_f)

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.H.zero_()
            return
        mask = doneMask.view(-1)
        if mask.numel() != self.H.size(0):
            raise ValueError("Value Hebbian reset mask must match its batch size")
        self.H[mask] = 0

    @torch.no_grad()
    def ProjectCap(self):
        n = self.H.norm(dim=-1, keepdim=True) # [B,O,1]
        scale = (1.0 / (n + 1e-12)).clamp_max(1.0)
        self.H.mul_(scale)

    def forward(
        self,
        x: torch.Tensor,
        commitMask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        y_base = F.linear(x, self.weight, self.bias) # [B,O]

        y_hebb = torch.bmm(self.H.detach().clone(), x.unsqueeze(-1)).squeeze(-1) # [B,O]
        y = y_base + 0.1 * y_hebb

        with torch.no_grad():
            pre = x
            post = y

            pre_n = pre / (pre.norm(dim=-1, keepdim=True) + 1e-6) # [B,I]
            post_n = post / (post.norm(dim=-1, keepdim=True) + 1e-6) # [B,O]

            dH = post_n.unsqueeze(-1) * pre_n.unsqueeze(1) # [B,O,I]

            post_sq = (post_n ** 2).unsqueeze(-1) # [B,O,1]
            dH = dH - self.H * post_sq

            next_h = self.H * 0.9 + 0.001 * dH
            next_norm = next_h.norm(dim=-1, keepdim=True)
            next_h = next_h * (1.0 / (next_norm + 1e-12)).clamp_max(1.0)
            if commitMask is None:
                self.H.copy_(next_h)
            else:
                if (
                    not torch.is_tensor(commitMask)
                    or tuple(commitMask.shape) != (x.size(0),)
                    or commitMask.device != x.device
                    or commitMask.dtype != torch.bool
                ):
                    raise ValueError("commitMask must be a batched boolean mask")
                self.H.copy_(torch.where(
                    commitMask.view(-1, 1, 1),
                    next_h,
                    self.H))

        extras = {"H_norm": self.H.norm(dim=(1, 2)).detach(),} # [B]

        return y, extras


class RunningEMA(AGICoreModule):
    def __init__(self, momentum: float = 0.99, eps: float = 1e-6, varFloor: float = 0.0):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.var_floor = float(varFloor)

        self.register_buffer("mean", torch.empty(0, dtype=torch.float32))
        self.register_buffer("var", torch.empty(0, dtype=torch.float32))

    @torch.no_grad()
    def EnsureB(self, B: int):
        B = int(B)
        if self.mean.numel() != B:
            self.mean = self.mean.new_zeros(B)
            self.var = self.var.new_ones(B)

    @torch.no_grad()
    def ResetAll(self, doneMask: Optional[torch.Tensor] = None):
        if self.mean.numel() > 0:
            if doneMask is None:
                self.mean.zero_()
                self.var.fill_(1.0)
            else:
                mask = doneMask.bool().view(-1)
                n = min(int(mask.numel()), int(self.mean.numel()))
                if n > 0:
                    rows = mask[:n].nonzero(as_tuple=False).view(-1)
                    if rows.numel() > 0:
                        self.mean[rows] = 0
                        self.var[rows] = 1.0

    @torch.no_grad()
    def Update(
        self,
        x: torch.Tensor,
        updateMask: Optional[torch.Tensor] = None,
    ):
        mask = torch.isfinite(x) # [B]
        if updateMask is not None:
            if (
                not torch.is_tensor(updateMask)
                or tuple(updateMask.shape) != tuple(mask.shape)
                or updateMask.device != x.device
                or updateMask.dtype != torch.bool
            ):
                raise ValueError("updateMask must be a batched boolean mask")
            mask = mask & updateMask
        if not mask.any():
            return

        mom = self.momentum
        mean_old = self.mean # [B]

        mean_new = mean_old * mom + (1.0 - mom) * x
        resid2 = (x - mean_old) ** 2 # [B]
        var_new = self.var * mom + (1.0 - mom) * resid2

        if self.var_floor > 0.0:
            var_new = var_new.clamp_min(self.var_floor)

        self.mean[mask] = mean_new[mask]
        self.var[mask] = var_new[mask]

    def ZScore(self, x: torch.Tensor) -> torch.Tensor:
        std = (self.var + self.eps).sqrt()
        y = (x - self.mean) / std
        return y



class UncertaintyCore(AGICoreModule):
    def __init__(
        self,
        *,
        stateDim: int,
        memDim: int,
        attnDim: int,
        hidden: int = 1024,
        ensK: int = 4,
        emaMomentum: float = 0.99,
        bootstrapKeep: float = 0.67,
        wTd: float = 1.00,
        wEnt: float = 0.30,
        wState: float = 0.45,
        wTr: float = 0.65,
        wPh: float = 0.55,
        wCtx: float = 0.25,
        tdScale: float = 1.0,
        entScale: float = 1.0,
        reconScale: float = 1.0,
        epsPrior: float = 1e-4,
        eps: float = 1e-6,):
        super().__init__()

        self.state_dim = int(stateDim)
        self.mem_dim = int(memDim)
        self.attn_dim = int(attnDim)
        self.hidden = int(hidden)
        self.ens_k = int(ensK)
        self.bootstrap_keep = float(bootstrapKeep)
        self.w_td = float(wTd)
        self.w_ent = float(wEnt)
        self.w_state = float(wState)
        self.w_tr = float(wTr)
        self.w_ph = float(wPh)
        self.w_ctx = float(wCtx)
        self.td_scale = float(tdScale)
        self.ent_scale = float(entScale)
        self.recon_scale = float(reconScale)
        self.eps_prior = float(epsPrior)
        self.eps = float(eps)

        self.td_ema = RunningEMA(momentum=emaMomentum)
        self.ent_ema = RunningEMA(momentum=emaMomentum)
        self.state_ema = RunningEMA(momentum=emaMomentum)
        self.tr_ema = RunningEMA(momentum=emaMomentum)
        self.ph_ema = RunningEMA(momentum=emaMomentum)
        self.ctx_ema = RunningEMA(momentum=emaMomentum)


    @torch.no_grad()
    def EnsureB(self, B: int):
        self.td_ema.EnsureB(B)
        self.ent_ema.EnsureB(B)
        self.state_ema.EnsureB(B)
        self.tr_ema.EnsureB(B)
        self.ph_ema.EnsureB(B)
        self.ctx_ema.EnsureB(B)

    @torch.no_grad()
    def ResetState(self, doneMask: Optional[torch.Tensor] = None):
        for ema in (self.td_ema, self.ent_ema, self.state_ema, self.tr_ema, self.ph_ema, self.ctx_ema):
            ema.ResetAll(doneMask=doneMask)

    def forward(
        self,
        memoryPrev: torch.Tensor, # [B,memDim]
        attnPrev: torch.Tensor, # [B,attnDim]
        stateCurr: torch.Tensor, # [B,stateDim]
        entropyPrev: torch.Tensor, # [B]
        tdCurr:torch.Tensor, # [B]
        worldDeltaTransport: torch.Tensor, # [B,stateDim]
        worldDeltaPhysics: torch.Tensor, # [B,stateDim]
        doneCurr: torch.Tensor, # [B]
        commitMask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


        B = int(stateCurr.size(0))
        self.EnsureB(B)

        H_prev = entropyPrev
        td_curr = tdCurr

        alive = 1.0 - doneCurr.float()
        H_prev = H_prev * alive
        td_curr = td_curr * alive

        td_curr_abs = (td_curr / max(self.td_scale, self.eps)).abs()
        ent_scaled = H_prev / max(self.ent_scale, self.eps)

        tr_mag = worldDeltaTransport.pow(2).mean(dim=-1).sqrt()
        ph_mag = worldDeltaPhysics.pow(2).mean(dim=-1).sqrt()

        pred_scaled = (0.5 * (tr_mag + ph_mag)) / max(self.recon_scale, self.eps)

        memory_energy = memoryPrev.pow(2).mean(dim=-1).sqrt()
        attn_energy = attnPrev.pow(2).mean(dim=-1).sqrt()
        ctx_mag = 0.5 * (memory_energy + attn_energy)
        state_mag = stateCurr.pow(2).mean(dim=-1).sqrt()

        td_curr_n = self.td_ema.ZScore(td_curr_abs).clamp(-8.0, 8.0)
        ent_n = self.ent_ema.ZScore(ent_scaled).clamp(-8.0, 8.0)
        state_n = self.state_ema.ZScore(state_mag).abs().clamp(0.0, 8.0)
        tr_n = self.tr_ema.ZScore(tr_mag).clamp(-8.0, 8.0)
        ph_n = self.ph_ema.ZScore(ph_mag).clamp(-8.0, 8.0)
        ctx_n = self.ctx_ema.ZScore(ctx_mag).abs().clamp(0.0, 8.0)

        self.td_ema.Update(td_curr_abs.detach(), updateMask=commitMask)
        self.ent_ema.Update(ent_scaled.detach(), updateMask=commitMask)
        self.state_ema.Update(state_mag.detach(), updateMask=commitMask)
        self.tr_ema.Update(tr_mag.detach(), updateMask=commitMask)
        self.ph_ema.Update(ph_mag.detach(), updateMask=commitMask)
        self.ctx_ema.Update(ctx_mag.detach(), updateMask=commitMask)

        e_td = F.relu(td_curr_n)
        e_ent = F.relu(ent_n)
        e_state = state_n
        e_tr = F.relu(tr_n)
        e_ph = F.relu(ph_n)
        e_ctx = ctx_n

        evidence = (
            self.w_td * e_td
            + self.w_ent * e_ent
            + self.w_state * e_state
            + self.w_tr * e_tr
            + self.w_ph * e_ph
            + self.w_ctx * e_ctx)

        unc_total = F.softplus(evidence) - math.log(2.0) + self.eps_prior

        comps: Dict[str, torch.Tensor] = {
            "sigma2_ale": pred_scaled.pow(2),
            "ens_var": tr_mag.pow(2),
            "dis_ph": ph_mag.pow(2),
            "unc_total": unc_total,
            "td_n": td_curr_n,
            "td_curr_n": td_curr_n,
            "ent_n": ent_n,
            "state_n": state_n,
            "tr_n": tr_n,
            "ph_n": ph_n,
            "ctx_n": ctx_n,
            "e_td": e_td,
            "e_ent": e_ent,
            "e_state": e_state,
            "e_tr": e_tr,
            "e_ph": e_ph,
            "e_ctx": e_ctx,
            "td_curr_abs": td_curr_abs,
            "tr_mag": tr_mag,
            "ph_mag": ph_mag,
            "pred_err": pred_scaled,
            "recon_err": pred_scaled,
            "ctx_mag": ctx_mag,
            "state_mag": state_mag,}

        return unc_total.detach(), comps # unc_total:[B]


class KalmanLocalLevelNext(AGICoreModule):
    def __init__(
        self,
        tau: float = 2.0,
        eps: float = 1e-6):
        super().__init__()
        self.tau = tau
        self.eps = eps

    @torch.no_grad()
    def PredictNext(self, y: torch.Tensor) -> torch.Tensor:
        _, n = y.shape

        t = torch.arange(n, device=self.device, dtype=self.dtype)

        if self.tau and self.tau > 0:
            ww = torch.exp((t - (n - 1)) / float(self.tau))
        else:
            ww = torch.ones_like(t)

        wsum = ww.sum().clamp_min(self.eps)
        t_mean = (ww * t).sum() / wsum
        y_mean = (y * ww).sum(dim=1) / wsum

        dt = t - t_mean
        denom = (ww * dt * dt).sum().clamp_min(self.eps)
        slope = ((y - y_mean[:, None]) * (ww * dt)[None, :]).sum(dim=1) / denom
        intercept = y_mean - slope * t_mean

        y_next = intercept + slope * float(n)
        return y_next


class NaturalCubicSplineNext(AGICoreModule):
    def __init__(
        self,
        eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    @torch.no_grad()
    def PredictNext(self, y: torch.Tensor) -> torch.Tensor:
        B, n = y.shape
        if n == 0:
            return y.new_full((B,), float("nan"))
        if n == 1:
            return y[:, 0]
        if n == 2:
            return (y[:, 1] + (y[:, 1] - y[:, 0]))
        if n == 3:
            d0 = y[:, 1] - y[:, 0]
            d1 = y[:, 2] - y[:, 1]
            return (y[:, 2] + d1 + (d1 - d0))

        d0 = (-3.0 * y[:, 0] + 4.0 * y[:, 1] - y[:, 2]) * 0.5
        dn = (3.0 * y[:, -1] - 4.0 * y[:, -2] + y[:, -3]) * 0.5

        dvec = torch.empty((B, n), device=self.device, dtype=self.dtype)
        dvec[:, 0] = 6.0 * ((y[:, 1] - y[:, 0]) - d0)
        dvec[:, 1:-1] = 6.0 * (y[:, 2:] - 2.0 * y[:, 1:-1] + y[:, :-2])
        dvec[:, -1] = 6.0 * (dn - (y[:, -1] - y[:, -2]))

        a = torch.ones((n,), device=self.device, dtype=self.dtype)
        bdiag = torch.full((n,), 4.0, device=self.device, dtype=self.dtype)
        c = torch.ones((n,), device=self.device, dtype=self.dtype)

        a[0] = 0.0
        bdiag[0] = 2.0
        c[0] = 1.0
        a[-1] = 1.0
        bdiag[-1] = 2.0
        c[-1] = 0.0

        cp = torch.empty((n,), device=self.device, dtype=self.dtype)
        dp = torch.empty((B, n), device=self.device, dtype=self.dtype)

        denom = bdiag[0].clamp_min(self.eps)
        cp[0] = c[0] / denom
        dp[:, 0] = dvec[:, 0] / denom

        for i in range(1, n):
            denom = (bdiag[i] - a[i] * cp[i - 1]).clamp_min(self.eps)
            cp[i] = (c[i] / denom) if i < n - 1 else torch.tensor(0.0, device=self.device, dtype=self.dtype)
            dp[:, i] = (dvec[:, i] - a[i] * dp[:, i - 1]) / denom

        M = torch.empty((B, n), device=self.device, dtype=self.dtype)
        M[:, -1] = dp[:, -1]
        for i in range(n - 2, -1, -1):
            M[:, i] = dp[:, i] - cp[i] * M[:, i + 1]

        y_next = 2.0 * y[:, -1] - y[:, -2] + M[:, -1]
        return y_next


class HarmonicRegressionNext(AGICoreModule):
    def __init__(
        self,
        maxHarmonics: int = 3,
        includeTrend: bool = True,
        ridge: float = 1e-6,
        eps: float = 1e-6):
        super().__init__()

        self.max_harmonics = maxHarmonics
        self.include_trend = includeTrend
        self.ridge = ridge
        self.eps = eps

    @torch.no_grad()
    def PredictNext(self, y: torch.Tensor) -> torch.Tensor:
        B, n = y.shape
        if n == 0:
            return y.new_full((B,), float("nan"))
        if n == 1:
            return y[:, 0]
        if n == 2:
            return (y[:, 1] + (y[:, 1] - y[:, 0]))

        t = torch.arange(n, device=self.device, dtype=self.dtype)

        if self.include_trend:
            t_mean = t.mean()
            dt = t - t_mean
            var_t = (dt * dt).sum().clamp_min(self.eps)
            y_mean = y.mean(dim=1)
            slope = ((y - y_mean[:, None]) * dt[None, :]).sum(dim=1) / var_t
            intercept = y_mean - slope * t_mean
            resid = y - (intercept[:, None] + slope[:, None] * t[None, :])
        else:
            resid = y - y.mean(dim=1, keepdim=True)

        fft = torch.fft.rfft(resid, dim=1)
        mag = torch.abs(fft)
        if mag.size(1) <= 1:
            return (y[:, -1] + (y[:, -1] - y[:, -2]))

        mag[:, 0] = 0.0
        mag_tail = mag[:, 1:]

        K = int(min(self.max_harmonics, mag_tail.size(1)))
        if K <= 0:
            return (y[:, -1] + (y[:, -1] - y[:, -2]))

        idx = torch.topk(mag_tail, k=K, dim=1, largest=True, sorted=True).indices
        k_bin = (idx + 1).float()
        w = 2.0 * torch.pi * k_bin / float(n)

        cols = [torch.ones((B, n, 1), device=self.device, dtype=self.dtype)]
        if self.include_trend:
            cols.append(t[None, :, None].expand(B, n, 1))

        phase = t[None, :, None] * w[:, None, :]
        cols.append(torch.cos(phase))
        cols.append(torch.sin(phase))

        X = torch.cat(cols, dim=2)
        p = X.size(2)

        Xt = X.transpose(1, 2)
        XtX = Xt @ X
        Xty = Xt @ y[:, :, None]

        I = torch.eye(p, device=self.device, dtype=self.dtype)[None, :, :]
        beta = torch.linalg.solve(XtX + float(self.ridge) * I, Xty).squeeze(-1)

        t_next = torch.tensor(float(n), device=self.device, dtype=self.dtype)
        x_next_cols = [torch.ones((B, 1), device=self.device, dtype=self.dtype)]
        if self.include_trend:
            x_next_cols.append(t_next.expand(B, 1))

        phase_next = t_next * w
        x_next_cols.append(torch.cos(phase_next))
        x_next_cols.append(torch.sin(phase_next))
        x_next = torch.cat(x_next_cols, dim=1)

        y_next = (x_next * beta).sum(dim=1)
        return y_next


class KalmanFilteredEnsembleNext(AGICoreModule):
    def __init__(
        self,
        *,
        processNoise: float = 1e-3,
        measureNoise: float = 1e-2,
        initVar: float = 1.0,
        historyLen: int = 128,
        fitLastN: int = 32,
        predictMode: str = "auto",
        autoPolicy: str = "blend",
        autoTemperature: float = 1.0,
        fuseWeights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        eps: float = 1e-6,):
        super().__init__()

        self.process_noise = float(processNoise)
        self.measure_noise = float(measureNoise)
        self.init_var = float(initVar)
        self.history_len = max(1, int(historyLen))
        self.fit_last_n = int(fitLastN)
        self.predict_mode = self.NormMode(predictMode)
        self.auto_policy = self.NormAutoPolicy(autoPolicy)
        self.auto_temperature = float(autoTemperature)
        self.eps = float(eps)

        for name, value in (
            ("processNoise", self.process_noise),
            ("measureNoise", self.measure_noise),
            ("initVar", self.init_var),
            ("eps", self.eps),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(name + " must be finite and positive")

        w = torch.tensor(list(fuseWeights), dtype=torch.float32)
        if w.numel() != 3:
            raise ValueError("fuseWeights must contain exactly 3 values.")
        w = w.clamp_min(0.0)
        if float(w.sum().item()) <= 0.0:
            w = torch.ones_like(w)
        self.register_buffer("fuse_weights", w / w.sum().clamp_min(self.eps))

        self.register_buffer("kf_mean", torch.empty(0, dtype=torch.float32))
        self.register_buffer("kf_var", torch.empty(0, dtype=torch.float32))
        self.register_buffer("smooth_hist", torch.empty(0, 0, dtype=torch.float32))
        self.register_buffer("hist_len", torch.empty(0, dtype=torch.long))

        self.kalman_next = KalmanLocalLevelNext(eps=eps)
        self.spline_next = NaturalCubicSplineNext(eps=eps)
        self.harmonic_next = HarmonicRegressionNext(eps=eps)

    @torch.no_grad()
    def NormMode(self, mode: str) -> str:
        m = str(mode).strip().lower()
        mode_map = {
            "kalman": "kalman",
            "spline": "spline",
            "harmonic": "harmonic",
            "fuse": "fuse",
            "auto": "auto",}
        if m not in mode_map:
            raise ValueError("predictMode must be one of: kalman/spline/harmonic/fuse/auto")
        return mode_map[m]

    @torch.no_grad()
    def NormAutoPolicy(self, policy: str) -> str:
        p = str(policy).strip().lower()
        policy_map = {
            "blend": "blend",
            "best": "best",}
        if p not in policy_map:
            raise ValueError("autoPolicy must be one of: blend/best")
        return policy_map[p]

    @torch.no_grad()
    def SetMode(self, predictMode: str, autoPolicy: str):
        self.predict_mode = self.NormMode(predictMode)
        self.auto_policy = self.NormAutoPolicy(autoPolicy)

    @torch.no_grad()
    def SetFitLastN(self, fitLastN: int):
        n = int(fitLastN)
        if n < 0:
            raise ValueError("fitLastN must be >= 0.")
        self.fit_last_n = n

    @torch.no_grad()
    def Reset(self,  doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.kf_mean = self.kf_mean.new_empty(0)
            self.kf_var = self.kf_var.new_empty(0)
            self.smooth_hist = self.smooth_hist.new_empty(0, 0)
            self.hist_len = self.hist_len.new_empty(0)
            return
        if self.kf_mean.numel() <= 0:
            return
        mask = doneMask.bool().view(-1)
        n = min(int(mask.numel()), int(self.kf_mean.numel()))
        if n <= 0:
            return
        rows = mask[:n].nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        self.kf_mean[rows] = 0
        self.kf_var[rows] = float(self.init_var)
        self.hist_len[rows] = 0
        if self.smooth_hist.size(0) >= n:
            self.smooth_hist[rows] = 0

    @torch.no_grad()
    def AppendHistory(self, value: torch.Tensor):
        B = int(value.size(0))
        previous = self.smooth_hist
        previous_len = self.hist_len
        next_len = (previous_len + 1).clamp_max(self.history_len)
        width = int(next_len.max().item())
        history = value.new_zeros(B, width)
        for row in range(B):
            old_len = int(previous_len[row].item())
            keep_old = min(old_len, max(0, width - 1))
            if keep_old > 0:
                history[row, -(keep_old + 1):-1] = previous[row, -keep_old:]
            history[row, -1] = value[row]
        self.smooth_hist = history.detach()
        self.hist_len = next_len.detach()

    @torch.no_grad()
    def PredictValidHistories(self, mode: str, autoPolicy: str) -> torch.Tensor:
        B = int(self.smooth_hist.size(0))
        predicted = self.kf_mean.new_empty(B)
        for valid_len in torch.unique(self.hist_len, sorted=True).tolist():
            length = int(valid_len)
            rows = (self.hist_len == length).nonzero(as_tuple=False).view(-1)
            fit_len = length if self.fit_last_n <= 0 else min(length, self.fit_last_n)
            y_fit = self.smooth_hist.index_select(0, rows)[:, -fit_len:]
            if fit_len < 2:
                predicted[rows] = y_fit[:, -1]
                continue
            y_k, y_s, y_h = self.PredictAll(y_fit)
            predicted[rows] = self.SelectPrediction(y_k, y_s, y_h, y_fit, mode, autoPolicy)
        return predicted

    @torch.no_grad()
    def KalmanStep(
        self,
        z: torch.Tensor, # [B]
        meanPrev: torch.Tensor, # [B]
        varPrev: torch.Tensor # [B]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        p_prior = varPrev + self.process_noise
        gain = p_prior / (p_prior + self.measure_noise + self.eps)
        mean_post = meanPrev + gain * (z - meanPrev)
        var_post = ((1.0 - gain) * p_prior).clamp_min(self.eps)
        return mean_post, var_post

    @torch.no_grad()
    def PredictAll(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_k = self.kalman_next.PredictNext(y)
        y_s = self.spline_next.PredictNext(y)
        y_h = self.harmonic_next.PredictNext(y)
        return y_k, y_s, y_h

    @torch.no_grad()
    def AutoWeights(self, yFit: torch.Tensor, autoPolicy: str) -> torch.Tensor:
        if yFit.size(1) <= 1:
            B = int(yFit.size(0))
            return self.fuse_weights.view(1, 3).expand(B, 3)

        y_ctx = yFit[:, :-1]
        target = yFit[:, -1]
        p_k, p_s, p_h = self.PredictAll(y_ctx)
        err = torch.stack(
            [(p_k - target).abs(), (p_s - target).abs(), (p_h - target).abs()],
            dim=1).clamp_min(self.eps)

        if autoPolicy == "best":
            idx = torch.argmin(err, dim=1)
            w = torch.zeros_like(err)
            w.scatter_(1, idx.unsqueeze(1), 1.0)
            return w

        temp = max(self.auto_temperature, self.eps)
        return F.softmax(-err / temp, dim=1)

    @torch.no_grad()
    def SelectPrediction(
        self,
        y_k: torch.Tensor,
        y_s: torch.Tensor,
        y_h: torch.Tensor,
        yFit: torch.Tensor,
        mode: str,
        autoPolicy: str,) -> torch.Tensor:
        pred_stack = torch.stack([y_k, y_s, y_h], dim=1)
        mode_now = self.NormMode(mode)
        auto_now = self.NormAutoPolicy(autoPolicy)

        if mode_now == "kalman":
            return y_k
        if mode_now == "spline":
            return y_s
        if mode_now == "harmonic":
            return y_h
        if mode_now == "fuse":
            w = self.fuse_weights.view(1, 3).expand(y_k.size(0), 3)
            return (pred_stack * w).sum(dim=1)

        w = self.AutoWeights(yFit, auto_now)
        return (pred_stack * w).sum(dim=1) # [B]

    @torch.no_grad()
    def PredictNext(
        self,
        obsPrev: torch.Tensor, # [B]
        mode: Optional[str] = None,
        autoPolicy: Optional[str] = None,
        commitMask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        z = obsPrev
        B = int(z.size(0))
        if commitMask is None:
            commit_mask = torch.ones(B, device=z.device, dtype=torch.bool)
        elif (
            not torch.is_tensor(commitMask)
            or tuple(commitMask.shape) != (B,)
            or commitMask.device != z.device
            or commitMask.dtype != torch.bool
        ):
            raise ValueError("commitMask must be a batched boolean mask")
        else:
            commit_mask = commitMask
        previous_mean = self.kf_mean.detach().clone()
        previous_var = self.kf_var.detach().clone()
        previous_history = self.smooth_hist.detach().clone()
        previous_length = self.hist_len.detach().clone()

        mode_now = self.NormMode(mode if mode is not None else self.predict_mode)
        auto_now = self.NormAutoPolicy(autoPolicy if autoPolicy is not None else self.auto_policy)

        if self.kf_mean.numel() != B:
            self.kf_mean = z.detach()
            self.kf_var = torch.full_like(z, float(self.init_var))
            self.smooth_hist = z.unsqueeze(1).detach()
            self.hist_len = torch.ones_like(z, dtype=torch.long)
        else:
            initialized = self.hist_len > 0
            x_post = z.detach().clone()
            p_post = torch.full_like(z, float(self.init_var))
            x_live, p_live = self.KalmanStep(
                z[initialized],
                self.kf_mean[initialized],
                self.kf_var[initialized])
            x_post[initialized] = x_live
            p_post[initialized] = p_live
            self.kf_mean = x_post.detach()
            self.kf_var = p_post.detach()
            self.AppendHistory(x_post)

        prediction = self.PredictValidHistories(mode_now, auto_now)
        if not bool(commit_mask.all().item()):
            if previous_mean.numel() != B:
                previous_mean = torch.zeros_like(self.kf_mean)
                previous_var = torch.full_like(self.kf_var, float(self.init_var))
                previous_length = torch.zeros_like(self.hist_len)
                previous_history = self.smooth_hist.new_zeros(B, 0)
            self.kf_mean = torch.where(
                commit_mask,
                self.kf_mean,
                previous_mean)
            self.kf_var = torch.where(
                commit_mask,
                self.kf_var,
                previous_var)
            self.hist_len = torch.where(
                commit_mask,
                self.hist_len,
                previous_length)
            restored_history = self.smooth_hist.clone()
            inactive = ~commit_mask
            restored_history[inactive] = 0
            if previous_history.size(1) > 0:
                restored_history[
                    inactive,
                    -previous_history.size(1):] = previous_history[inactive]
            self.smooth_hist = restored_history
        return prediction

    @torch.no_grad()
    def PosteriorSmooth(
        self,
        observations: torch.Tensor,
        terminalObservation: torch.Tensor,
        *,
        bounded: bool = False,
        returnVariance: bool = False,
        timestamps: Optional[torch.Tensor] = None,
    ):
        if (
            not torch.is_tensor(observations)
            or observations.dim() != 2
            or not observations.is_floating_point()
            or not bool(torch.isfinite(observations).all().item())
        ):
            raise ValueError("posterior observations must be finite [B,T]")
        batch_size, length = observations.shape
        if (
            not torch.is_tensor(terminalObservation)
            or tuple(terminalObservation.shape) != (batch_size,)
            or terminalObservation.device != observations.device
            or not terminalObservation.is_floating_point()
            or not bool(torch.isfinite(terminalObservation).all().item())
        ):
            raise ValueError("terminal observation must be finite [B]")
        eps = max(float(self.eps), torch.finfo(observations.dtype).eps)
        process_noise = max(float(self.process_noise), eps)
        measure_noise = max(float(self.measure_noise), eps)
        if timestamps is None:
            process_variance = observations.new_full(
                (batch_size, length),
                process_noise)
        else:
            if (
                not torch.is_tensor(timestamps)
                or tuple(timestamps.shape) != (batch_size, length + 1)
                or timestamps.device != observations.device
                or not timestamps.is_floating_point()
                or not bool(torch.isfinite(timestamps).all().item())
            ):
                raise ValueError(
                    "posterior timestamps must be finite [B,T+1]")
            timestamp64 = timestamps.to(dtype=torch.float64)
            time_delta64 = timestamp64[:, 1:] - timestamp64[:, :-1]
            if bool((time_delta64 < 0.0).any().item()):
                raise ValueError("posterior timestamps must not decrease")
            time_delta = time_delta64.to(dtype=observations.dtype)
            variance_limit = torch.finfo(observations.dtype).max ** 0.5
            process_variance = (
                time_delta * process_noise
            ).clamp(min=0.0, max=variance_limit)
        if bounded:
            if bool(((observations < 0.0) | (observations > 1.0)).any().item()):
                raise ValueError("bounded posterior observations must be probabilities")
            if bool(((terminalObservation < 0.0) | (terminalObservation > 1.0)).any().item()):
                raise ValueError("bounded terminal observation must be a probability")
            sequence = torch.logit(observations.clamp(eps, 1.0 - eps))
            terminal = torch.logit(
                terminalObservation.clamp(eps, 1.0 - eps))
        else:
            sequence = observations
            terminal = terminalObservation
        if length == 0:
            empty = observations.clone()
            return (empty, empty) if returnVariance else empty
        filtered_mean = torch.empty_like(sequence)
        filtered_var = torch.empty_like(sequence)
        mean = sequence[:, 0]
        prior_variance = torch.full_like(mean, float(self.init_var))
        initial_gain = prior_variance / (
            prior_variance + measure_noise)
        variance = (
            (1.0 - initial_gain).square() * prior_variance
            + initial_gain.square() * measure_noise
        ).clamp_min(eps)
        filtered_mean[:, 0] = mean
        filtered_var[:, 0] = variance
        for index in range(1, length):
            prior_variance = variance + process_variance[:, index - 1]
            gain = prior_variance / (
                prior_variance + measure_noise)
            mean = mean + gain * (sequence[:, index] - mean)
            variance = (
                (1.0 - gain).square() * prior_variance
                + gain.square() * measure_noise
            ).clamp_min(eps)
            filtered_mean[:, index] = mean
            filtered_var[:, index] = variance
        terminal_prior_mean = filtered_mean[:, -1]
        terminal_prior_variance = (
            filtered_var[:, -1] + process_variance[:, -1])
        terminal_gain = terminal_prior_variance / (
            terminal_prior_variance + measure_noise)
        next_mean = (
            terminal_prior_mean
            + terminal_gain * (terminal - terminal_prior_mean))
        next_variance = (
            (1.0 - terminal_gain).square() * terminal_prior_variance
            + terminal_gain.square() * measure_noise
        ).clamp_min(eps)
        smooth_mean = filtered_mean.clone()
        smooth_var = filtered_var.clone()
        for index in range(length - 1, -1, -1):
            predicted_variance = (
                filtered_var[:, index] + process_variance[:, index])
            smoothing_gain = filtered_var[:, index] / predicted_variance
            smooth_mean[:, index] = (
                filtered_mean[:, index]
                + smoothing_gain
                * (next_mean - filtered_mean[:, index]))
            smooth_var[:, index] = (
                filtered_var[:, index]
                + smoothing_gain.square()
                * (next_variance - predicted_variance)
            ).clamp_min(eps)
            next_mean = smooth_mean[:, index]
            next_variance = smooth_var[:, index]
        if bounded:
            result = torch.sigmoid(smooth_mean)
            result_gradient = result * (1.0 - result)
            result_variance = (
                result_gradient.square() * smooth_var
            ).clamp_min(torch.finfo(result.dtype).tiny)
            result_variance = torch.minimum(
                result_variance,
                result_gradient)
        else:
            result = smooth_mean
            result_variance = smooth_var
        return (result, result_variance) if returnVariance else result



class MaxPlusLinear(AGICoreModule):
    def __init__(self, inFeatures: int, outFeatures: int, useSoft: bool = True, temperature: float = 0.2):
        super().__init__()
        self.W = nn.Parameter(torch.empty(outFeatures, inFeatures))
        self.b = nn.Parameter(torch.zeros(outFeatures))
        self.use_soft = useSoft
        self.temperature = temperature
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5)); nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.W.unsqueeze(0) + x.unsqueeze(1) # [B, O, I]
        if self.use_soft:
            t = max(self.temperature, 1e-6)
            z = score / t # [B, O, I]
            m = z.amax(dim=-1, keepdim=True) # [B, O, I]
            y = t * (m.squeeze(-1) + torch.logsumexp(z - m, dim=-1)) # [B, O]
        else:
            y,_ = score.max(dim=-1)
        return y + self.b # [B, O] + [O] -> [B, O]


class TropicalAffineTransport(AGICoreModule):
    def __init__(
        self,
        hDim: int,
        useSoftTrop: bool = True,
        temp: float = 0.2,
        epsA: float = 1e-3,
        numExperts: int = 4,
        numCounterfactuals: int = 4,
        expertTemp: float = 1.0,
        aDeltaLimit: float = 0.25,
        bLimit: float = 0.5,
        driftScale: float = 0.10,
        manifoldZDim: Optional[int] = None,
        manifoldRank: int = 8,
        manifoldStepScale: float = 0.10,
        manifoldCorrectionScale: float = 0.05,
        branchTensorScale: float = 0.10,
        manifoldFieldEma: float = 0.99,):
        super().__init__()
        self.k = max(1, int(numExperts))
        self.c = max(1, int(numCounterfactuals))
        self.epsA = float(epsA)
        self.expert_temp = float(expertTemp)
        self.a_delta_limit = float(aDeltaLimit)
        self.b_limit = float(bLimit)
        self.drift_scale = float(driftScale)
        self.manifold_z_dim = int(manifoldZDim or max(32, min(128, hDim // 8)))
        self.manifold_rank = max(1, int(manifoldRank))
        self.manifold_step_scale = float(manifoldStepScale)
        self.manifold_correction_scale = float(manifoldCorrectionScale)
        self.branch_tensor_scale = float(branchTensorScale)
        self.manifold_field_ema = float(manifoldFieldEma)
        self.field_ctx = nn.Sequential(
            nn.Linear(hDim, hDim),
            nn.LayerNorm(hDim),
            nn.GELU(),
            nn.Linear(hDim, hDim),
            nn.GELU(),)

        self.expert_gate = nn.Linear(hDim, self.k)
        self.trop = MaxPlusLinear(inFeatures=hDim + 1, outFeatures=self.k, useSoft=useSoftTrop, temperature=temp)

        self.affine_core = nn.Sequential(
            nn.Linear(hDim, hDim),
            nn.LayerNorm(hDim),
            nn.GELU(),
            nn.Linear(hDim, hDim),
            nn.GELU(),)

        self.a_head = nn.Linear(hDim, self.k)
        self.b_head = nn.Linear(hDim, self.k)
        self.g_head = nn.Linear(hDim, self.k)
        self.value_tensor_head = nn.Linear(hDim, self.k * self.manifold_z_dim)

        self.cf_core = nn.Sequential(
            nn.Linear(hDim, hDim),
            nn.LayerNorm(hDim),
            nn.GELU(),
            nn.Linear(hDim, hDim),
            nn.GELU(),)
        self.cf_gate = nn.Linear(hDim, self.c)
        self.cf_trop = MaxPlusLinear(inFeatures=hDim + 1, outFeatures=self.c, useSoft=useSoftTrop, temperature=temp)
        self.cf_a_head = nn.Linear(hDim, self.c)
        self.cf_b_head = nn.Linear(hDim, self.c)
        self.cf_g_head = nn.Linear(hDim, self.c)
        self.cf_value_tensor_head = nn.Linear(hDim, self.c * self.manifold_z_dim)
        self.branch_signal_tensor_basis = nn.Parameter(torch.empty(self.k + self.c, self.manifold_z_dim))

        self.manifold_encoder = nn.Sequential(
            nn.Linear(hDim, self.manifold_z_dim),
            nn.LayerNorm(self.manifold_z_dim),
            nn.GELU(),
            nn.Linear(self.manifold_z_dim, self.manifold_z_dim),
            nn.LayerNorm(self.manifold_z_dim),
            nn.GELU(),)
        self.manifold_encoder_blocks = nn.ModuleList([
            ResidualMLPBlock(self.manifold_z_dim, hiddenMul=2.0, scaleInit=0.20)
            for _ in range(2)])
        self.manifold_encoder_out = nn.Linear(self.manifold_z_dim, self.manifold_z_dim)

        self.manifold_aux_dim = hDim + 2 * self.manifold_z_dim + 3
        self.manifold_aux_norm = nn.LayerNorm(self.manifold_aux_dim)
        self.manifold_aux_blocks = nn.ModuleList([
            ResidualMLPBlock(self.manifold_aux_dim, hiddenMul=1.5, scaleInit=0.15)
            for _ in range(2)])
        manifold_hidden = max(self.manifold_z_dim * 2, min(hDim, 256))
        self.manifold_drift_head = nn.Sequential(
            nn.Linear(self.manifold_aux_dim, manifold_hidden),
            nn.LayerNorm(manifold_hidden),
            nn.GELU(),
            nn.Linear(manifold_hidden, manifold_hidden),
            nn.GELU(),
            nn.Linear(manifold_hidden, self.manifold_z_dim),)
        self.manifold_connection_gate = nn.Sequential(
            nn.Linear(self.manifold_aux_dim, manifold_hidden),
            nn.LayerNorm(manifold_hidden),
            nn.GELU(),
            nn.Linear(manifold_hidden, self.manifold_rank),)
        self.manifold_metric_head = nn.Sequential(
            nn.Linear(self.manifold_aux_dim, manifold_hidden),
            nn.LayerNorm(manifold_hidden),
            nn.GELU(),
            nn.Linear(manifold_hidden, self.manifold_z_dim),)
        correction_dim = 5 * self.manifold_z_dim + 1
        self.manifold_value_correction = nn.Sequential(
            nn.Linear(correction_dim, manifold_hidden),
            nn.LayerNorm(manifold_hidden),
            nn.GELU(),
            nn.Linear(manifold_hidden, manifold_hidden),
            nn.GELU(),
            nn.Linear(manifold_hidden, self.manifold_z_dim),)
        self.manifold_value_readout = nn.Linear(self.manifold_z_dim, 1, bias=False)
        self.flow_fusion_dim = hDim + 7
        self.flow_fusion_norm = nn.LayerNorm(self.flow_fusion_dim)
        self.flow_fusion_gate = nn.Sequential(
            nn.Linear(self.flow_fusion_dim, manifold_hidden),
            nn.LayerNorm(manifold_hidden),
            nn.GELU(),
            nn.Linear(manifold_hidden, 4),)
        self.manifold_connection_basis = nn.Parameter(torch.empty(self.manifold_rank, self.manifold_z_dim, self.manifold_z_dim))
        self.manifold_tensor_basis = nn.Parameter(torch.empty(self.manifold_rank, self.manifold_z_dim))
        self.register_buffer("manifold_tensor_field_ema", torch.zeros(0, self.manifold_z_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.expert_gate.weight)
        nn.init.zeros_(self.expert_gate.bias)
        nn.init.zeros_(self.a_head.weight)
        nn.init.zeros_(self.a_head.bias)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)
        nn.init.zeros_(self.g_head.weight)
        nn.init.constant_(self.g_head.bias, -2.0)
        nn.init.zeros_(self.value_tensor_head.weight)
        nn.init.zeros_(self.value_tensor_head.bias)
        nn.init.zeros_(self.cf_gate.weight)
        nn.init.constant_(self.cf_gate.bias, -4.0)
        nn.init.zeros_(self.cf_a_head.weight)
        nn.init.zeros_(self.cf_a_head.bias)
        nn.init.zeros_(self.cf_b_head.weight)
        nn.init.zeros_(self.cf_b_head.bias)
        nn.init.zeros_(self.cf_g_head.weight)
        nn.init.constant_(self.cf_g_head.bias, -2.0)
        nn.init.zeros_(self.cf_value_tensor_head.weight)
        nn.init.zeros_(self.cf_value_tensor_head.bias)
        nn.init.normal_(self.manifold_drift_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.manifold_drift_head[-1].bias)
        nn.init.normal_(self.manifold_connection_gate[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.manifold_connection_gate[-1].bias)
        nn.init.normal_(self.manifold_metric_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.manifold_metric_head[-1].bias)
        nn.init.normal_(self.manifold_value_correction[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.manifold_value_correction[-1].bias)
        nn.init.normal_(self.flow_fusion_gate[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow_fusion_gate[-1].bias)
        nn.init.normal_(self.manifold_value_readout.weight, mean=0.0, std=1.0 / math.sqrt(max(1, self.manifold_z_dim)))
        nn.init.normal_(self.manifold_connection_basis, mean=0.0, std=1.0 / math.sqrt(max(1, self.manifold_z_dim)))
        nn.init.normal_(self.manifold_tensor_basis, mean=0.0, std=1.0 / math.sqrt(max(1, self.manifold_z_dim)))
        nn.init.normal_(self.branch_signal_tensor_basis, mean=0.0, std=1.0 / math.sqrt(max(1, self.manifold_z_dim)))

    @torch.no_grad()
    def EnsureB(self, B: int):
        if self.manifold_tensor_field_ema.size(0) != B:
            self.manifold_tensor_field_ema = self.manifold_tensor_field_ema.new_zeros(
                B, self.manifold_z_dim)

    @torch.no_grad()
    def ResetState(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.manifold_tensor_field_ema.zero_()
        else:
            self.manifold_tensor_field_ema[doneMask.bool().view(-1)] = 0

    def BuildManifoldConnection(
        self,
        ctx: torch.Tensor,
        vIn: torch.Tensor,
        rawDelta: torch.Tensor,
        branchValueTensorAll: torch.Tensor,
        branchValueTensorMix: torch.Tensor,) -> Dict[str, torch.Tensor]:
        B = int(ctx.size(0))
        self.EnsureB(B)
        z_state = self.manifold_encoder(ctx) # [B,Z]
        for blk in self.manifold_encoder_blocks:
            z_state = blk(z_state)
        z = torch.tanh(self.manifold_encoder_out(z_state)) # [B,Z]

        v_in = vIn
        raw_delta = rawDelta
        branch_tensor = branchValueTensorAll.view(B, -1, self.manifold_z_dim)
        branch_tensor_mix = branchValueTensorMix.view(B, self.manifold_z_dim)
        branch_mean = branch_tensor.mean(dim=(1, 2), keepdim=False).view(B, 1)
        branch_std = branch_tensor.std(dim=(1, 2), unbiased=False, keepdim=False).view(B, 1)

        aux = torch.cat([ctx, z, v_in, raw_delta, branch_mean, branch_std], dim=-1)
        aux = self.manifold_aux_norm(aux)
        for blk in self.manifold_aux_blocks:
            aux = blk(aux)

        gate = torch.softmax(self.manifold_connection_gate(aux), dim=-1) # [B,R]
        connection = torch.einsum("br,rij->bij", gate, self.manifold_connection_basis) # [B,Z,Z]
        field = torch.einsum("br,rz->bz", gate, self.manifold_tensor_basis) # [B,Z]
        metric_diag = F.softplus(self.manifold_metric_head(aux)).clamp_max(10.0) + self.epsA # [B,Z]

        parallel = torch.bmm(connection, z.unsqueeze(-1)).squeeze(-1) / math.sqrt(max(1, self.manifold_z_dim))
        drift = self.manifold_drift_head(aux)
        u = torch.tanh(drift + field + 0.10 * self.manifold_tensor_field_ema + parallel / metric_diag)
        z_next = torch.tanh(z + self.manifold_step_scale * u)
        corr_in = torch.cat([z, z_next, u, branch_tensor_mix, raw_delta, v_in], dim=-1)
        value_tensor_delta = torch.tanh(self.manifold_value_correction(corr_in))
        value_tensor = branch_tensor_mix + self.manifold_correction_scale * value_tensor_delta
        value_correction = value_tensor - branch_tensor_mix

        metric_reg = (metric_diag - 1.0).pow(2).mean(dim=-1)
        connection_norm = connection.pow(2).mean(dim=(1, 2)).sqrt()
        field_norm = field.pow(2).mean(dim=-1).sqrt()
        u_norm = u.pow(2).mean(dim=-1).sqrt()
        value_tensor_norm = value_tensor.pow(2).mean(dim=-1).sqrt()
        parameter_reg = (
            self.manifold_connection_basis.pow(2).mean()
            + self.manifold_tensor_basis.pow(2).mean())
        reg_per_row = ctx.new_tensor(1e-4) * (
            parameter_reg
            + connection_norm.pow(2)
            + field_norm.pow(2)
            + u_norm.pow(2)
            + value_tensor_norm.pow(2)
            + metric_reg)
        reg = reg_per_row.mean()

        return {
            "z": z,
            "z_next": z_next,
            "u": u,
            "field": field,
            "value_tensor": value_tensor,
            "value_tensor_delta": value_tensor_delta,
            "value_correction": value_correction,
            "connection_norm": connection_norm,
            "field_norm": field_norm,
            "u_norm": u_norm,
            "value_tensor_norm": value_tensor_norm,
            "metric_diag": metric_diag,
            "metric_mean": metric_diag.mean(dim=-1),
            "flow_residual": value_correction.pow(2).mean(dim=-1).sqrt(),
            "reg": reg,
            "reg_per_row": reg_per_row,}

    @torch.no_grad()
    def CommitManifoldField(
        self,
        field: torch.Tensor,
        commitMask: Optional[torch.Tensor] = None,
    ):
        decay = self.manifold_field_ema
        update = (
            self.manifold_tensor_field_ema * decay
            + field.detach() * (1.0 - decay))
        if commitMask is None:
            self.manifold_tensor_field_ema.copy_(update)
        else:
            if (
                not torch.is_tensor(commitMask)
                or tuple(commitMask.shape) != (field.size(0),)
                or commitMask.device != field.device
                or commitMask.dtype != torch.bool
            ):
                raise ValueError("commitMask must be a batched boolean mask")
            self.manifold_tensor_field_ema.copy_(torch.where(
                commitMask.unsqueeze(-1),
                update,
                self.manifold_tensor_field_ema))

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        returnExtras: Optional[bool] = None,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v_in = v # [B,D]
        v_energy = v_in.pow(2).mean(dim=-1, keepdim=True).sqrt() # [B,1], physical tensor energy
        ctx = self.field_ctx(h) # [B,H]

        tau = max(self.expert_temp, 1e-6)
        w_logits = self.expert_gate(ctx) # [B,K]

        z_aff = self.affine_core(ctx) # [B,H]
        a_raw = self.a_head(z_aff) # [B,K]
        b_raw = self.b_head(z_aff) # [B,K]
        g_raw = self.g_head(z_aff) # [B,K]

        a = (1.0 + self.a_delta_limit * torch.tanh(a_raw)).clamp_min(self.epsA) # [B,K]
        b = self.b_limit * torch.tanh(b_raw) # [B,K]
        g = torch.sigmoid(g_raw) # [B,K]

        trop_in = torch.cat([ctx, v_energy], dim=-1)
        trop_all = self.trop(trop_in) # [B,K]
        aff_all = a * v_energy + b # [B,K]
        tensor_signal_all = g * trop_all + (1.0 - g) * aff_all # [B,K]
        value_tensor_all = self.value_tensor_head(z_aff).view(-1, self.k, self.manifold_z_dim)

        cf_ctx = self.cf_core(ctx) # [B,H]
        cf_logits = self.cf_gate(cf_ctx) # [B,C]
        cf_a = (1.0 + self.a_delta_limit * torch.tanh(self.cf_a_head(cf_ctx))).clamp_min(self.epsA) # [B,C]
        cf_b = self.b_limit * torch.tanh(self.cf_b_head(cf_ctx)) # [B,C]
        cf_g = torch.sigmoid(self.cf_g_head(cf_ctx)) # [B,C]
        cf_trop_in = torch.cat([cf_ctx, v_energy], dim=-1)
        cf_trop_all = self.cf_trop(cf_trop_in) # [B,C]
        cf_aff_all = cf_a * v_energy + cf_b # [B,C]
        cf_tensor_signal_all = cf_g * cf_trop_all + (1.0 - cf_g) * cf_aff_all # [B,C]
        cf_value_tensor_all = self.cf_value_tensor_head(cf_ctx).view(-1, self.c, self.manifold_z_dim)

        branch_logits = torch.cat([w_logits, cf_logits], dim=-1) # [B,K+C]
        branch_w = torch.softmax(branch_logits / tau, dim=-1) # [B,K+C]
        branch_tensor_signal_all = torch.cat([tensor_signal_all, cf_tensor_signal_all], dim=-1) # [B,K+C]
        branch_value_tensor_base = torch.cat([value_tensor_all, cf_value_tensor_all], dim=1) # [B,K+C,Z]
        branch_tensor_delta = branch_tensor_signal_all - v_energy
        branch_value_tensor_all = (
            v_in.unsqueeze(1)
            + self.branch_tensor_scale * torch.tanh(
                branch_value_tensor_base
                + branch_tensor_delta.unsqueeze(-1) * self.branch_signal_tensor_basis.unsqueeze(0)))

        branch_value_tensor_mix = (branch_w.unsqueeze(-1) * branch_value_tensor_all).sum(dim=1) # [B,Z]
        raw_delta = self.manifold_value_readout(branch_value_tensor_mix - v_in) # [B,1]

        manifold = self.BuildManifoldConnection(
            ctx,
            vIn=v_in,
            rawDelta=raw_delta,
            branchValueTensorAll=branch_value_tensor_all,
            branchValueTensorMix=branch_value_tensor_mix)

        manifold_flow = self.manifold_value_readout(manifold["value_correction"]) # [B,1]

        tensor_signal_mix = (branch_w * branch_tensor_signal_all).sum(dim=-1, keepdim=True)

        tensor_delta_mix = tensor_signal_mix - v_energy

        branch_value_tensor_connected = (
            branch_value_tensor_all
            + self.manifold_correction_scale * manifold["value_tensor_delta"].unsqueeze(1))

        branch_flow_all = self.manifold_value_readout(
            branch_value_tensor_all.reshape(-1, self.manifold_z_dim)).view(v_in.size(0), self.k + self.c)

        branch_flow_connected = self.manifold_value_readout(
            branch_value_tensor_connected.reshape(-1, self.manifold_z_dim)).view(v_in.size(0), self.k + self.c)

        branch_flow_mix = (branch_w * branch_flow_all).sum(dim=-1, keepdim=True)
        branch_flow_connected_mix = (branch_w * branch_flow_connected).sum(dim=-1, keepdim=True)
        branch_value_tensor_connected_mix = (branch_w.unsqueeze(-1) * branch_value_tensor_connected).sum(dim=1)
        branch_flow_spread = branch_flow_connected.std(dim=-1, unbiased=False, keepdim=True)

        flow_fusion_ctx = torch.cat([
            ctx,
            tensor_delta_mix,
            branch_flow_mix,
            branch_flow_connected_mix,
            manifold_flow,
            branch_flow_spread,
            manifold["u_norm"].view(-1, 1),
            manifold["metric_mean"].view(-1, 1)], dim=-1)

        flow_fusion_logits = self.flow_fusion_gate(self.flow_fusion_norm(flow_fusion_ctx))
        flow_fusion_w = torch.softmax(flow_fusion_logits, dim=-1)

        flow_candidates = torch.cat([
            tensor_delta_mix,
            branch_flow_mix,
            branch_flow_connected_mix,
            manifold_flow], dim=-1)

        flow_candidate_mix = (flow_fusion_w * flow_candidates).sum(dim=-1, keepdim=True)
        flow_tensor_candidates = torch.stack([
            branch_value_tensor_mix,
            branch_value_tensor_connected_mix,
            manifold["value_tensor"],
            v_in + tensor_delta_mix.expand_as(v_in),], dim=1) # [B,4,D]
        flow_tensor_mix = (flow_fusion_w.unsqueeze(-1) * flow_tensor_candidates).sum(dim=1)
        flow_mix = self.manifold_value_readout(flow_tensor_mix - v_in)

        dv = self.drift_scale * torch.tanh(flow_tensor_mix - v_in)

        v_next_hat = v_in + dv # [B,D]

        if returnExtras is None:
            returnExtras = bool(self.training)
        if not returnExtras:
            return v_next_hat, {}

        w = branch_w[:, :self.k] # [B,K]
        cf_w = branch_w[:, self.k:] # [B,C]
        branch_next_all = (
            v_in.unsqueeze(1)
            + self.drift_scale * torch.tanh(branch_value_tensor_connected - v_in.unsqueeze(1))) # [B,K+C,D]

        g_all = torch.cat([g, cf_g], dim=-1)
        a_all = torch.cat([a, cf_a], dim=-1)
        b_all = torch.cat([b, cf_b], dim=-1)
        trop_all_full = torch.cat([trop_all, cf_trop_all], dim=-1)
        aff_all_full = torch.cat([aff_all, cf_aff_all], dim=-1)

        gate_trop = (branch_w * g_all).sum(dim=-1, keepdim=True)
        a_mix = (branch_w * a_all).sum(dim=-1, keepdim=True)
        b_mix = (branch_w * b_all).sum(dim=-1, keepdim=True)
        trop_mix = (branch_w * trop_all_full).sum(dim=-1, keepdim=True)
        aff_mix = (branch_w * aff_all_full).sum(dim=-1, keepdim=True)

        extras = {
            "gate_trop": gate_trop,
            "a": a_mix,
            "b": b_mix,
            "trop_out": trop_mix,
            "aff_out": aff_mix,
            "flow_dv": dv,
            "manifold_flow_residual": manifold_flow,
            "manifold_raw_delta": raw_delta,
            "tensor_signal_mix": tensor_signal_mix,
            "tensor_delta_mix": tensor_delta_mix,
            "flow_mix": flow_mix,
            "flow_candidate_mix": flow_candidate_mix,
            "flow_fusion_w": flow_fusion_w,
            "flow_candidates": flow_candidates,
            "branch_flow_mix": branch_flow_mix,
            "branch_flow_connected_mix": branch_flow_connected_mix,
            "branch_flow_spread": branch_flow_spread,
            "expert_w": w,
            "expert_trop": trop_all,
            "expert_aff": aff_all,
            "counterfactual_w": cf_w,
            "counterfactual_values": branch_next_all[:, self.k:],
            "counterfactual_trop": cf_trop_all,
            "counterfactual_aff": cf_aff_all,
            "branch_w": branch_w,
            "branch_flow": branch_flow_all,
            "branch_flow_connected": branch_flow_connected,
            "branch_tensor_signal": branch_tensor_signal_all,
            "branch_value_tensor": branch_value_tensor_all,
            "branch_value_tensor_connected": branch_value_tensor_connected,
            "branch_next": branch_next_all,
            "manifold": manifold,
            "manifold_z": manifold["z"],
            "manifold_z_next": manifold["z_next"],
            "manifold_u": manifold["u"],
            "manifold_value_tensor": manifold["value_tensor"],
            "manifold_value_correction": manifold["value_correction"],
            "manifold_connection_norm": manifold["connection_norm"],
            "manifold_field_norm": manifold["field_norm"],
            "manifold_u_norm": manifold["u_norm"],
            "manifold_value_tensor_norm": manifold["value_tensor_norm"],
            "manifold_metric_mean": manifold["metric_mean"],
            "manifold_reg": manifold["reg"],}

        return v_next_hat, extras


class TemporalMicroGraph(AGICoreModule):
    def __init__(
        self,
        valueDim: int,
        zDim: int,
        maxAnchors: int = 128,
        topK: int = 4,
        distTau: float = 1.0,
        valueDistScale: float = 0.25,
        ageScale: float = 1e-3,
        eps: float = 1e-6,):
        super().__init__()
        self.value_dim = int(valueDim)
        self.z_dim = int(zDim)
        self.max_anchors = int(maxAnchors)
        self.top_k = int(topK)
        self.dist_tau = float(distTau)
        self.value_dist_scale = float(valueDistScale)
        self.age_scale = float(ageScale)
        self.eps = float(eps)

        self.register_buffer("anchor_value", torch.zeros(1, self.max_anchors, self.value_dim))
        self.register_buffer("anchor_value_next", torch.zeros(1, self.max_anchors, self.value_dim))
        self.register_buffer("anchor_z", torch.zeros(1, self.max_anchors, self.z_dim))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self._step = 0

    @torch.no_grad()
    def EnsureB(self, B: int):
        if self.anchor_value.size(0) == int(B):
            return
        self.anchor_value = self.anchor_value.new_zeros(
            B, self.max_anchors, self.value_dim)
        self.anchor_value_next = self.anchor_value_next.new_zeros(
            B, self.max_anchors, self.value_dim)
        self.anchor_z = self.anchor_z.new_zeros(
            B, self.max_anchors, self.z_dim)
        self.filled = self.filled.new_zeros(B)
        self.ptr = self.ptr.new_zeros(B)
        self._step = 0

    @torch.no_grad()
    def Reset(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.anchor_value.zero_()
            self.anchor_value_next.zero_()
            self.anchor_z.zero_()
            self.filled.zero_()
            self.ptr.zero_()
            self._step = 0
            return
        mask = doneMask.bool().view(-1)
        rows = mask[:self.filled.size(0)].nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        self.anchor_value[rows] = 0
        self.anchor_value_next[rows] = 0
        self.anchor_z[rows] = 0
        self.filled[rows] = 0
        self.ptr[rows] = 0

    def Preview(
        self,
        value: torch.Tensor,
        z: torch.Tensor,) -> Dict[str, torch.Tensor]:
        B = int(value.size(0))
        self.EnsureB(B)

        ar = torch.arange(self.max_anchors, device=value.device).view(1, self.max_anchors)
        valid = ar < self.filled.view(B, 1)
        age = ((self.ptr.view(B, 1) - 1 - ar) % self.max_anchors).float()

        z_dist = (self.anchor_z - z.detach().unsqueeze(1)).pow(2).mean(dim=-1)
        value_dist = (self.anchor_value - value.detach().unsqueeze(1)).pow(2).mean(dim=-1)
        dist = z_dist + self.value_dist_scale * value_dist + self.age_scale * age
        dist = dist.masked_fill(~valid, float("inf"))

        K = min(self.top_k, self.max_anchors)
        idx = torch.topk(-dist, k=K, dim=1).indices
        d_sel = dist.gather(1, idx)
        sel_valid = torch.isfinite(d_sel)
        logits = -d_sel / max(self.dist_tau, self.eps)
        logits = logits.masked_fill(~sel_valid, -1e9)
        w = torch.softmax(logits, dim=-1) * sel_valid.float()
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        idx_v = idx.unsqueeze(-1).expand(B, K, self.value_dim)
        value_hist = self.anchor_value.gather(1, idx_v)
        value_next_hist = self.anchor_value_next.gather(1, idx_v)
        delta_hist = value_next_hist - value_hist
        graph_delta = (w.unsqueeze(-1) * delta_hist).sum(dim=1)
        graph_next = value.detach() + graph_delta

        row_valid = sel_valid.any(dim=-1).float()
        energy = graph_delta.pow(2).mean(dim=-1).sqrt() * row_valid

        return {
            "graph_next": graph_next,
            "graph_delta": graph_delta,
            "correction": graph_delta,
            "energy": energy,
            "weight_sum": w.sum(dim=-1),
            "valid": row_valid,
            "dist": d_sel.masked_fill(~sel_valid, 0.0),}

    @torch.no_grad()
    def CommitStep(
        self,
        value: torch.Tensor,
        valueNext: torch.Tensor,
        z: torch.Tensor,
        alive: torch.Tensor,
        commitMask: Optional[torch.Tensor] = None,):
        B = int(value.size(0))
        self.EnsureB(B)
        if commitMask is None:
            commit_mask = torch.ones(B, device=value.device, dtype=torch.bool)
        elif (
            not torch.is_tensor(commitMask)
            or tuple(commitMask.shape) != (B,)
            or commitMask.device != value.device
            or commitMask.dtype != torch.bool
        ):
            raise ValueError("commitMask must be a batched boolean mask")
        else:
            commit_mask = commitMask
        live = (alive.view(B) > 0.5) & commit_mask
        dead = (alive.view(B) <= 0.5) & commit_mask
        if dead.any():
            self.Reset(doneMask=dead)
        rows = live.nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        pos = self.ptr[rows].clamp_min(0).clamp_max(self.max_anchors - 1)
        self.anchor_value[rows, pos] = value[rows].detach()
        self.anchor_value_next[rows, pos] = valueNext[rows].detach()
        self.anchor_z[rows, pos] = z[rows].detach()
        self.ptr[rows] = (self.ptr[rows] + 1) % self.max_anchors
        self.filled[rows] = torch.minimum(self.filled[rows] + 1, torch.full_like(self.filled[rows], self.max_anchors))
        self._step += 1


class EmotionCore(AGICoreModule):
    def __init__(
        self,
        *,
        stateDim: int,
        memoryDim: int,
        attnDim: int,
        emotionDim: int = 64,
        fastHidden: int = 128,
        slowHidden: int = 128,
        moodDecay: float = 0.95,):
        super().__init__()

        self.stateDim = stateDim
        self.memoryDim = memoryDim
        self.attnDim = attnDim
        self.emotionDim = emotionDim
        self.fastHidden = fastHidden
        self.slowHidden = slowHidden
        self.moodDecay = float(moodDecay)

        fast_in_dim = stateDim + attnDim

        self.fast_net = nn.Sequential(
            nn.Linear(fast_in_dim, fastHidden),
            nn.SiLU(),
            nn.Linear(fastHidden, fastHidden),
            nn.SiLU(),)

        self.fast_head = HebbianLinearFW(
            inFeatures=fastHidden,
            outFeatures=emotionDim)

        slow_in_dim = stateDim + memoryDim + attnDim
        self.slow_cell = nn.LSTMCell(input_size=slow_in_dim, hidden_size=slowHidden)

        self.slow_head = HebbianLinearFW(
            inFeatures=slowHidden,
            outFeatures=emotionDim)

        self.register_buffer("h", torch.zeros(1, slowHidden))
        self.register_buffer("c", torch.zeros(1, slowHidden))
        self.register_buffer("mood", torch.zeros(1, emotionDim))

        self.w_fast = nn.Parameter(torch.tensor(0.5))
        self.w_slow = nn.Parameter(torch.tensor(0.5))
        self.w_mood = nn.Parameter(torch.tensor(0.1))

        gate_in_dim = memoryDim + attnDim + stateDim
        hidden_gate = max(32, gate_in_dim // 2)
        self.gate_net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_gate),
            nn.SiLU(),
            nn.Linear(hidden_gate, emotionDim),
            nn.Sigmoid(),)


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
    def ResetState(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.h.zero_()
            self.c.zero_()
            self.mood.zero_()
            return
        if self.h.size(0) <= 0:
            return
        mask = doneMask.bool().view(-1)
        n = min(int(mask.numel()), int(self.h.size(0)))
        if n <= 0:
            return
        rows = mask[:n].nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        self.h[rows] = 0
        self.c[rows] = 0
        self.mood[rows] = 0


    def EnsureB(self, B: int):
        if self.h.size(0) != B:
            self.h = self.h.new_zeros(B, self.slowHidden)
            self.c = self.c.new_zeros(B, self.slowHidden)
            self.mood = self.mood.new_zeros(B, self.emotionDim)
        self.fast_head.EnsureB(B)
        self.slow_head.EnsureB(B)

    def forward(
        self,
        memoryPrev: torch.Tensor,
        attnPrev: torch.Tensor,
        stateCurr: torch.Tensor,
        commitMask: Optional[torch.Tensor] = None,) -> torch.Tensor:

        h_prev = self.h # [B,H]
        c_prev = self.c # [B,H]
        mood_prev = self.mood # [B,E]

        fast_in = torch.cat([stateCurr, attnPrev], dim=-1)
        fast_h = self.fast_net(fast_in) # [B, F]

        fast_raw, _ = self.fast_head(fast_h, commitMask=commitMask)

        emotion_fast = torch.tanh(fast_raw)

        slow_in = torch.cat([stateCurr, memoryPrev, attnPrev], dim=-1)
        h_t, c_t = self.slow_cell(slow_in, (h_prev, c_prev)) # h_t: [B,H], c_t: [B,H]

        slow_raw, _ = self.slow_head(h_t, commitMask=commitMask) # [B,E]

        emotion_slow = torch.tanh(slow_raw) # [B,E]

        decay = self.moodDecay
        mood_t = decay * mood_prev + (1.0 - decay) * emotion_slow # [B,E]

        w_fast = F.softplus(self.w_fast)
        w_slow = F.softplus(self.w_slow)
        w_mood = F.softplus(self.w_mood)

        w_sum = w_fast + w_slow + w_mood
        wf = w_fast / w_sum
        ws = w_slow / w_sum
        wm = w_mood / w_sum

        wf_b = wf.view(1, 1)
        ws_b = ws.view(1, 1)
        wm_b = wm.view(1, 1)

        emotion_raw = (wf_b * emotion_fast + ws_b * emotion_slow + wm_b * mood_t)

        gate_in = torch.cat([memoryPrev, attnPrev, stateCurr], dim=-1)
        gate = self.gate_net(gate_in)
        emotion = torch.tanh(emotion_raw * (1.0 + gate)) # [B,E]

        if commitMask is None:
            self.h = h_t.detach()
            self.c = c_t.detach()
            self.mood = mood_t.detach()
        else:
            if (
                not torch.is_tensor(commitMask)
                or tuple(commitMask.shape) != (stateCurr.size(0),)
                or commitMask.device != stateCurr.device
                or commitMask.dtype != torch.bool
            ):
                raise ValueError("commitMask must be a batched boolean mask")
            mask = commitMask.unsqueeze(-1)
            self.h = torch.where(mask, h_t.detach(), self.h)
            self.c = torch.where(mask, c_t.detach(), self.c)
            self.mood = torch.where(mask, mood_t.detach(), self.mood)

        return emotion

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        self.slow_head.ResetHebbianMemory(doneMask=doneMask)
        self.fast_head.ResetHebbianMemory(doneMask=doneMask)


class GeoTropicalOut(NamedTuple):






    value: torch.Tensor
    valueNext: torch.Tensor
    tdError: torch.Tensor
    returnValue: torch.Tensor
    returnAdvantage: torch.Tensor
    loss: torch.Tensor
    emotion: torch.Tensor
    rComps: Dict[str, torch.Tensor]
    uncertainty: torch.Tensor
    precision: torch.Tensor
    valueHidden: torch.Tensor
    cognitiveValue: "CognitiveValueOut"
    extras: Dict[str, torch.Tensor]


class CognitiveValueOut(NamedTuple):
    feature: torch.Tensor
    coarseProgress: torch.Tensor
    detailProgress: torch.Tensor
    planStaleness: torch.Tensor
    replanBenefit: torch.Tensor
    computeCost: torch.Tensor
    feasibility: torch.Tensor
    safetyConstraint: torch.Tensor
    eventReasonLogits: torch.Tensor


class PotentialShapingOut(NamedTuple):
    shapedReward: torch.Tensor
    shapingReward: torch.Tensor
    previousPotential: torch.Tensor
    nextPotential: torch.Tensor
    previousKnown: torch.Tensor
    nextKnown: torch.Tensor


class SmdpAdvantageOut(NamedTuple):
    optionReward: torch.Tensor
    temporalDifference: torch.Tensor
    advantage: torch.Tensor
    returnTarget: torch.Tensor
    bootstrapDiscount: torch.Tensor
    traceDiscount: torch.Tensor


class SensorimotorPotentialShaper(AGICoreModule):
    def __init__(
        self,
        discount: float = 0.99,
        scale: float = 1.0,
    ):
        super().__init__()
        self.discount = float(discount)
        self.scale = float(scale)
        self.register_buffer(
            "last_potential",
            torch.empty(0, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "potential_known",
            torch.empty(0, dtype=torch.bool),
            persistent=False)

    @torch.no_grad()
    def EnsureBatch(self, reference: torch.Tensor):
        batch_size = int(reference.shape[0])
        if self.last_potential.shape != (batch_size,):
            self.last_potential = reference.new_zeros(batch_size)
            self.potential_known = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=reference.device)

    @torch.no_grad()
    def ResetState(self, resetMask: Optional[torch.Tensor] = None):
        if resetMask is None:
            self.last_potential.zero_()
            self.potential_known.zero_()
            return
        reset_mask = resetMask.bool().view(-1)
        self.last_potential.masked_fill_(reset_mask, 0.0)
        self.potential_known.logical_and_(~reset_mask)

    @torch.no_grad()
    def forward(
        self,
        baseReward: torch.Tensor,
        inconsistency: torch.Tensor,
        observationValid: torch.Tensor,
        duration: torch.Tensor,
        terminated: torch.Tensor,
        truncated: Optional[torch.Tensor] = None,
        commitMask: Optional[torch.Tensor] = None,
    ) -> PotentialShapingOut:
        reward = baseReward.view(-1)
        self.EnsureBatch(reward)
        valid = observationValid.bool().view(-1)
        terminal = terminated.bool().view(-1)
        cutoff = (
            torch.zeros_like(terminal)
            if truncated is None
            else truncated.bool().view(-1))
        commit = (
            torch.ones_like(terminal)
            if commitMask is None
            else commitMask.bool().view(-1))
        duration_value = duration.to(
            device=reward.device,
            dtype=reward.dtype).view(-1).clamp_min(0.0)
        energy = inconsistency.to(
            device=reward.device,
            dtype=reward.dtype).reshape(reward.shape[0], -1).mean(dim=-1)
        observed_potential = -energy
        previous_known = self.potential_known.clone()
        held_potential = torch.where(
            valid,
            observed_potential,
            self.last_potential)
        held_known = valid | previous_known
        previous_potential = torch.where(
            previous_known,
            self.last_potential,
            held_potential)
        next_potential = torch.where(
            terminal,
            torch.zeros_like(held_potential),
            held_potential)
        bootstrap_discount = torch.pow(
            reward.new_full((), self.discount),
            duration_value)
        shaping_reward = self.scale * (
            bootstrap_discount * next_potential - previous_potential)
        shaping_reward = torch.where(
            previous_known & held_known,
            shaping_reward,
            torch.zeros_like(shaping_reward))
        ended = terminal | cutoff
        stored_potential = torch.where(
            ended,
            torch.zeros_like(held_potential),
            held_potential)
        stored_known = held_known & ~ended
        self.last_potential.copy_(torch.where(
            commit,
            stored_potential,
            self.last_potential))
        self.potential_known.copy_(torch.where(
            commit,
            stored_known,
            self.potential_known))
        return PotentialShapingOut(
            shapedReward=reward + shaping_reward,
            shapingReward=shaping_reward,
            previousPotential=previous_potential,
            nextPotential=next_potential,
            previousKnown=previous_known,
            nextKnown=held_known & ~terminal)


class SmdpReturnEstimator:
    @staticmethod
    def DiscountOptionRewards(
        primitiveRewards: torch.Tensor,
        duration: torch.Tensor,
        discount: float,
        rewardValid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        horizon = int(primitiveRewards.shape[-1])
        duration_value = duration.to(
            device=primitiveRewards.device).long().clamp(0, horizon)
        step = torch.arange(
            horizon,
            device=primitiveRewards.device,
            dtype=primitiveRewards.dtype)
        active = step.view(*([1] * duration_value.dim()), horizon) < duration_value.unsqueeze(-1)
        if rewardValid is not None:
            active = active & rewardValid.bool()
        weights = torch.pow(
            primitiveRewards.new_full((), float(discount)),
            step)
        return (
            primitiveRewards
            * active.to(dtype=primitiveRewards.dtype)
            * weights.view(*([1] * duration_value.dim()), horizon)
        ).sum(dim=-1)

    @staticmethod
    def EstimateAdvantages(
        optionReward: torch.Tensor,
        values: torch.Tensor,
        duration: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        discount: float = 0.99,
        traceDecay: float = 0.95,
    ) -> SmdpAdvantageOut:
        transition_valid = (
            torch.ones_like(optionReward, dtype=torch.bool)
            if valid is None
            else valid.bool())
        terminal = terminated.bool()
        cutoff = truncated.bool()
        duration_value = duration.to(
            device=optionReward.device,
            dtype=optionReward.dtype).clamp_min(0.0)
        bootstrap_discount = torch.pow(
            optionReward.new_full((), float(discount)),
            duration_value)
        trace_discount = torch.pow(
            optionReward.new_full((), float(discount) * float(traceDecay)),
            duration_value)
        current_value = values[:-1]
        next_value = values[1:]
        temporal_difference = (
            optionReward
            + bootstrap_discount * (~terminal).to(optionReward.dtype) * next_value
            - current_value)
        temporal_difference = torch.where(
            transition_valid,
            temporal_difference,
            torch.zeros_like(temporal_difference))
        advantage = torch.zeros_like(optionReward)
        continuation = transition_valid & ~terminal & ~cutoff
        carried = values.new_zeros(values.shape[1:])
        for time_index in range(int(optionReward.shape[0]) - 1, -1, -1):
            current_advantage = (
                temporal_difference[time_index]
                + trace_discount[time_index]
                * continuation[time_index].to(optionReward.dtype)
                * carried)
            current_advantage = torch.where(
                transition_valid[time_index],
                current_advantage,
                torch.zeros_like(current_advantage))
            advantage[time_index] = current_advantage
            carried = current_advantage
        return SmdpAdvantageOut(
            optionReward=optionReward,
            temporalDifference=temporal_difference,
            advantage=advantage,
            returnTarget=advantage + current_value,
            bootstrapDiscount=bootstrap_discount,
            traceDiscount=trace_discount)


class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hiddenMul: float = 2.0,
        scaleInit: float = 0.25,):
        super().__init__()
        inner = max(int(dim), int(round(float(hiddenMul) * float(dim))))
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.res_scale = nn.Parameter(torch.tensor(float(scaleInit)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.norm(x)
        r = F.gelu(self.fc1(r))
        r = self.fc2(r)
        return x + torch.tanh(self.res_scale) * r


class ValueEstimationExtractor(AGICoreModule):
    RUNTIME_STATE_FIELDS = (
        "ve_is_training",
        "ve_last_batch_size",
        "pending_transitions",
        "active_stream_ids",
        "transport_prev_grad",
        "transport_curr_grad",
        "transport_delayed_ready",
        "transport_grad_accum_steps",
        "transport_prev_grad_hook_seen",
        "transport_curr_grad_hook_seen",
        "return_value_prev",
        "return_value_valid",
        "td_out_ema_mean",
        "td_out_ema_var",
        "emo_h",
        "emo_c",
        "emo_mood",
        "emo_fast_H",
        "emo_slow_H",
    ) + tuple(
        f"unc_{name}_{stat}"
        for name in ("td_ema", "ent_ema", "state_ema", "tr_ema", "ph_ema", "ctx_ema")
        for stat in ("mean", "var")
    ) + tuple(
        f"{prefix}_{name}"
        for prefix in ("reward_pred", "done_pred")
        for name in (
            "kf_mean",
            "kf_var",
            "smooth_hist",
            "hist_len",
            "predict_mode",
            "auto_policy",
            "auto_temperature",
            "fit_last_n",)
    ) + (
        "micro_anchor_value",
        "micro_anchor_value_next",
        "micro_anchor_z",
        "micro_filled",
        "micro_ptr",
        "micro_step",
        "transport_manifold_tensor_field_ema",)

    def __init__(self,
        memoryDim: int = 768, attnDim: int = 1024, stateDim: int = 256, *,
        emotionDim: int = 64,
        hidden: int = 2048,
        trunkResBlocks: int = 4,
        trunkResHiddenMul: float = 2.0,
        trunkResScaleInit: float = 0.25,
        useSoftTrop: bool = True, tropTemp: float = 0.2, epsA: float = 1e-3,
        valueTensorDim: int = 512,
        microMaxAnchors: int = 64,
        microTopK: int = 4,
        microDistTau: float = 1.0,
        microValueDistScale: float = 0.25,
        microAgeScale: float = 1e-3,
        tdGeomRank: int = 16,
        tdHeatRank: int = 32,
        tdBuresSlots: int = 16,
        tdOtBins: int = 64,
        tdOtCostDim: int = 8,
        tdOtIters: int = 8,
        tdHuberKappa: float = 1.0,
        tdGeomSignTau: float = 1.0,
        tdGeomSobAlpha: float = 0.10,
        tdGeomEps: float = 1e-6,
        returnDiscount: float = 0.99,
        cognitiveValueDim: int = 64,):
        super().__init__()

        self.in_dim = memoryDim + attnDim + stateDim
        H = hidden
        self.num_quantiles = 32
        self.value_tensor_dim = int(valueTensorDim)
        self.cvar_alpha = 0.20
        tau = (torch.arange(self.num_quantiles, dtype=torch.float32) + 0.5) / float(self.num_quantiles)
        self.register_buffer("quantile_tau", tau)

        self.td_geom_rank = max(1, int(tdGeomRank))
        self.td_heat_rank = max(1, int(tdHeatRank))
        self.td_bures_slots = int(tdBuresSlots)
        self.td_bures_dim = self.value_tensor_dim // self.td_bures_slots
        self.td_ot_bins = int(tdOtBins)
        self.td_ot_bin_width = self.value_tensor_dim // self.td_ot_bins
        self.td_ot_cost_dim = int(tdOtCostDim)
        self.td_ot_iters = int(tdOtIters)
        self.td_huber_kappa = float(tdHuberKappa)
        self.td_geom_sign_tau = max(float(tdGeomSignTau), 1e-6)
        self.td_geom_sob_alpha = float(tdGeomSobAlpha)
        self.td_geom_eps = float(tdGeomEps)
        self.return_discount = float(returnDiscount)
        self.cognitive_value_dim = int(cognitiveValueDim)
        if self.cognitive_value_dim <= 0:
            raise ValueError("cognitiveValueDim must be positive")

        self.td_out_ema = RunningEMA(momentum=0.99)
        self.td_scale_min = 1e-3
        self.unc_tau = 4.0
        self.reward_next_mod_scale = 0.10
        self.micro_graph_mix = 0.05

        self.wDiff = 1.0
        self.wDiffBranch = 0.25
        self.wBranchStructure = 0.01
        self.wManifoldLatent = 0.05
        self.wPhysicalTDParamReg = 1.0
        self.entity_ontology_risk_loss_weight = 0.02

        self.entity_ontology_physical_risk_head = nn.Sequential(
            nn.LayerNorm(20),
            nn.Linear(20, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),)
        nn.init.zeros_(self.entity_ontology_physical_risk_head[-1].weight)
        nn.init.zeros_(self.entity_ontology_physical_risk_head[-1].bias)

        self.reward_predictor = KalmanFilteredEnsembleNext()
        self.done_predictor = KalmanFilteredEnsembleNext()

        self.fc1 = nn.Linear(self.in_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)
        self.trunk_res_blocks = nn.ModuleList([
            ResidualMLPBlock(dim=H, hiddenMul=trunkResHiddenMul, scaleInit=trunkResScaleInit)
            for _ in range(max(0, int(trunkResBlocks)))])

        self.cognitive_value_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, self.cognitive_value_dim),
            nn.GELU(),
            nn.Linear(self.cognitive_value_dim, self.cognitive_value_dim),)
        self.prospective_context_head = nn.Sequential(
            nn.Linear(4, self.cognitive_value_dim),
            nn.GELU(),
            nn.Linear(self.cognitive_value_dim, self.cognitive_value_dim),)
        self.cognitive_value_norm = nn.LayerNorm(self.cognitive_value_dim)
        self.cognitive_metric_head = nn.Linear(self.cognitive_value_dim, 7)
        self.cognitive_proxy_head = nn.Linear(self.cognitive_value_dim, 5)
        self.cognitive_event_reason_head = nn.Linear(
            self.cognitive_value_dim,
            COGNITIVE_COMPUTE_REASON_COUNT)

        self.quantile_head = nn.Linear(H, self.num_quantiles)
        self.value_ensemble_heads = nn.ModuleList([nn.Linear(H, 1) for _ in range(4)])

        self.value_tensor_tail = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, self.value_tensor_dim),
            nn.GELU(),
            nn.Linear(self.value_tensor_dim, self.value_tensor_dim),)

        self.value_tensor_out_norm = nn.LayerNorm(self.value_tensor_dim)
        self.value_tensor_log_scale = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))
        self.return_value_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 1),)
        self.register_buffer("return_value_prev", torch.zeros(1), persistent=False)
        self.register_buffer("return_value_valid", torch.zeros(1, dtype=torch.bool), persistent=False)

        scale_init = math.log(math.e - 1.0)
        self.latent_mahalanobis_logvar = nn.Parameter(torch.full((self.value_tensor_dim,), scale_init))
        self.latent_L_factor = nn.Parameter(torch.zeros(self.td_geom_rank, self.value_tensor_dim))
        nn.init.normal_(self.latent_L_factor, mean=0.0, std=1e-3)
        self.latent_L_diag = nn.Parameter(torch.zeros(self.value_tensor_dim))
        self.physical_metric_log_scale = nn.Parameter(torch.full((6,), scale_init))
        self.physical_td_logits = nn.Parameter(torch.zeros(6))
        context_hidden = max(256, H // 4)
        self.td_context_head = nn.Sequential(
            nn.LayerNorm(H + 6, elementwise_affine=False),
            nn.Linear(H + 6, context_hidden),
            nn.GELU(),
            nn.Linear(context_hidden, self.value_tensor_dim),)
        self.td_heat_basis = nn.Parameter(torch.empty(self.td_heat_rank, self.value_tensor_dim))
        self.td_heat_log_eigs = nn.Parameter(torch.zeros(self.td_heat_rank))
        self.td_ot_cost_embed = nn.Parameter(torch.empty(self.td_ot_bins, self.td_ot_cost_dim))

        self.fc1_adapter = GrowableLoRALinear(self.fc1)
        self.fc2_adapter = GrowableLoRALinear(self.fc2)
        self.quantile_adapter = GrowableLoRALinear(self.quantile_head)

        self.emotion_dim = emotionDim
        self.emotion_core = EmotionCore(
            stateDim=stateDim,
            memoryDim=memoryDim,
            attnDim=attnDim,
            emotionDim=emotionDim)
        self.emotion_to_hidden = nn.Sequential(
            nn.Linear(emotionDim, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Linear(H, H),)
        self.emotion_fusion_gate = nn.Linear(H + emotionDim, 1)
        self.emotion_fusion_scale = nn.Parameter(torch.tensor(0.05))

        self.transport = TropicalAffineTransport(H, useSoftTrop, tropTemp, epsA, manifoldZDim=self.value_tensor_dim)
        self.micro = TemporalMicroGraph(
            valueDim=self.value_tensor_dim,
            zDim=self.value_tensor_dim,
            maxAnchors=microMaxAnchors,
            topK=microTopK,
            distTau=microDistTau,
            valueDistScale=microValueDistScale,
            ageScale=microAgeScale)
        self._pending_transitions: Dict[int, deque] = {}
        self._last_batch_size: Optional[int] = None
        self._active_stream_ids: Optional[List[int]] = None
        self._max_pending_per_stream: int = 8
        self._transport_prev_grad: Dict[str, torch.Tensor] = {}
        self._transport_curr_grad: Dict[str, torch.Tensor] = {}
        self._transport_delayed_ready: bool = False
        self._transport_grad_accum_steps: int = 0
        self._transport_prev_grad_hook_seen: set = set()
        self._transport_curr_grad_hook_seen: set = set()
        self._transport_grad_hooks = []
        self.InstallTransportGradHooks()

        self.unc_core = UncertaintyCore(
            stateDim=stateDim,
            memDim=memoryDim,
            attnDim=attnDim,
            hidden=max(256, H // 2))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.emotion_to_hidden[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.emotion_to_hidden[-1].bias)
        nn.init.zeros_(self.emotion_fusion_gate.weight)
        nn.init.constant_(self.emotion_fusion_gate.bias, -2.0)
        nn.init.zeros_(self.quantile_head.weight)
        nn.init.zeros_(self.quantile_head.bias)
        nn.init.zeros_(self.return_value_head[-1].weight)
        nn.init.zeros_(self.return_value_head[-1].bias)
        nn.init.normal_(self.td_context_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.td_context_head[-1].bias)
        nn.init.normal_(self.td_heat_basis, mean=0.0, std=1.0 / math.sqrt(float(self.value_tensor_dim)))
        nn.init.normal_(self.td_ot_cost_embed, mean=0.0, std=1.0 / math.sqrt(float(self.td_ot_cost_dim)))
        nn.init.zeros_(self.entity_ontology_physical_risk_head[-1].weight)
        nn.init.zeros_(self.entity_ontology_physical_risk_head[-1].bias)
        self.emotion_core.ResetParams()

    def EnsureB(self, B: int):
        self.td_out_ema.EnsureB(B)
        self.micro.EnsureB(B)
        self.emotion_core.EnsureB(B)
        if self.return_value_prev.shape != (B,):
            self.return_value_prev = self.return_value_prev.new_zeros(B)
            self.return_value_valid = self.return_value_valid.new_zeros(B)

        if self._last_batch_size is None:
            self._last_batch_size = int(B)
        elif int(self._last_batch_size) != int(B):
            self._pending_transitions.clear()
            self.ClearTransportGradAccumulator()
            self._last_batch_size = int(B)
            self._active_stream_ids = None

    def Trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1_adapter(x))
        h = self.norm1(h)
        h = F.gelu(self.fc2_adapter(h))
        h = self.norm2(h)
        for blk in self.trunk_res_blocks:
            h = blk(h)
        return h

    def DecisionSummary(
        self,
        decisionFeature: torch.Tensor,) -> torch.Tensor:
        if not torch.is_tensor(decisionFeature):
            raise TypeError("decisionFeature must be a tensor")
        if decisionFeature.dim() < 2:
            raise ValueError("decisionFeature must have a batch and feature dimension")
        flattened = decisionFeature.reshape(decisionFeature.size(0), -1)
        if flattened.size(1) <= 0:
            raise ValueError("decisionFeature must not be empty")
        return torch.stack([
            flattened.mean(dim=-1),
            flattened.std(dim=-1, unbiased=False),
            flattened.abs().amax(dim=-1),
            flattened.square().mean(dim=-1).sqrt(),], dim=-1)

    def BuildCognitiveValue(
        self,
        hidden: torch.Tensor,
        decisionFeature: Optional[torch.Tensor] = None,) -> CognitiveValueOut:
        source = hidden if decisionFeature is None else decisionFeature
        if source.size(0) != hidden.size(0):
            raise ValueError("decisionFeature batch size must match hidden state")
        context = self.prospective_context_head(
            self.DecisionSummary(source))
        feature = torch.tanh(self.cognitive_value_norm(
            self.cognitive_value_head(hidden) + context))
        metrics = self.cognitive_metric_head(feature)
        return CognitiveValueOut(
            feature=feature,
            coarseProgress=torch.sigmoid(metrics[:, 0]),
            detailProgress=torch.sigmoid(metrics[:, 1]),
            planStaleness=torch.sigmoid(metrics[:, 2]),
            replanBenefit=F.softplus(metrics[:, 3]),
            computeCost=F.softplus(metrics[:, 4]),
            feasibility=torch.sigmoid(metrics[:, 5]),
            safetyConstraint=torch.sigmoid(metrics[:, 6]),
            eventReasonLogits=self.cognitive_event_reason_head(feature))

    def StoreValueConditioning(
        self,
        valueOutput: GeoTropicalOut,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(valueOutput, GeoTropicalOut):
            raise TypeError("valueOutput must be GeoTropicalOut")
        return {
            "valueHidden": valueOutput.valueHidden.detach().clone(),
            "valueBaseline": valueOutput.returnValue.detach().clone(),
        }

    def RecomputeReturnValue(
        self,
        conditioning: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if set(conditioning) != {"valueHidden", "valueBaseline"}:
            raise ValueError("value conditioning fields do not match")
        hidden = conditioning["valueHidden"]
        baseline = conditioning["valueBaseline"]
        if (
            not torch.is_tensor(hidden)
            or hidden.dim() != 2
            or int(hidden.size(1)) != self.return_value_head.in_features
            or not hidden.is_floating_point()
            or not bool(torch.isfinite(hidden).all().item())
            or not torch.is_tensor(baseline)
            or tuple(baseline.shape) != (int(hidden.size(0)),)
            or baseline.device != hidden.device
            or baseline.dtype != hidden.dtype
            or not bool(torch.isfinite(baseline).all().item())
        ):
            raise ValueError("value conditioning tensors do not match")
        return self.return_value_head(hidden).squeeze(-1)

    def BuildCognitiveProxy(
        self,
        feature: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        metrics = torch.sigmoid(self.cognitive_proxy_head(feature))
        return {
            "executionContinuity": metrics[:, 0],
            "sensoryReliability": metrics[:, 1],
            "replanUrgency": metrics[:, 2],
            "operationalConfidence": metrics[:, 3],
            "riskAbsence": metrics[:, 4],}

    def PreDecisionValue(
        self,
        memoryPrev: torch.Tensor,
        attnPrev: torch.Tensor,
        state: torch.Tensor,) -> CognitiveValueOut:
        self.EnsureB(int(state.size(0)))
        hidden = self.Trunk(torch.cat([memoryPrev, attnPrev, state], dim=-1))
        emotion = self.emotion_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state)
        hidden = self.FuseEmotionIntoHidden(hidden, emotion)
        return self.BuildCognitiveValue(hidden)

    def ProspectiveValue(
        self,
        memoryPrev: torch.Tensor,
        attnPrev: torch.Tensor,
        state: torch.Tensor,
        decisionFeature: torch.Tensor,) -> CognitiveValueOut:
        self.EnsureB(int(state.size(0)))
        hidden = self.Trunk(torch.cat([memoryPrev, attnPrev, state], dim=-1))
        emotion = self.emotion_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state)
        hidden = self.FuseEmotionIntoHidden(hidden, emotion)
        return self.BuildCognitiveValue(hidden, decisionFeature)

    def FuseEmotionIntoHidden(self, h: torch.Tensor, emotion: torch.Tensor) -> torch.Tensor:
        e = torch.tanh(self.emotion_to_hidden(emotion))
        gate = torch.sigmoid(self.emotion_fusion_gate(torch.cat([h, emotion], dim=-1)))
        return h + torch.tanh(self.emotion_fusion_scale) * gate * e

    def ApplyRewardNextModulation(
        self,
        value: torch.Tensor,
        valueNext: torch.Tensor,
        transpExtras: Dict[str, torch.Tensor],
        rewardNext: torch.Tensor,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        gain = 1.0 + self.reward_next_mod_scale * torch.tanh(rewardNext).view(-1, 1)
        value_next = value + gain * (valueNext - value)
        extras = dict(transpExtras)
        extras["branch_next"] = value.unsqueeze(1) + gain.unsqueeze(1) * (transpExtras["branch_next"] - value.unsqueeze(1))
        return value_next, extras

    def ApplyMicroGraphPrior(
        self,
        valueNext: torch.Tensor,
        transpExtras: Dict[str, torch.Tensor],
        microGraph: Dict[str, torch.Tensor],) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mix = valueNext.new_tensor(self.micro_graph_mix) * microGraph["valid"].view(-1, 1)
        graph_next = microGraph["graph_next"]
        value_next = valueNext + mix * (graph_next - valueNext)
        extras = dict(transpExtras)
        extras["branch_next"] = transpExtras["branch_next"] + (value_next - valueNext).unsqueeze(1)
        return value_next, extras

    def DistributionStats(self, quantiles: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_sorted = torch.sort(quantiles, dim=-1).values # [B,N]
        q_mean = quantiles.mean(dim=-1) # [B]
        q_var = quantiles.var(dim=-1, unbiased=False) # [B]
        q_std = (q_var + 1e-6).sqrt() # [B]
        n_cvar = max(1, int(math.ceil(float(self.num_quantiles) * float(self.cvar_alpha))))
        q_cvar = q_sorted[:, :n_cvar].mean(dim=-1) # [B]
        downside = (q_mean - q_cvar).clamp_min(0.0) # [B]
        scale = q_mean.detach().abs() + 1.0 # [B]
        dist_risk = 1.0 - torch.exp(-(q_std + downside) / scale) # [B]
        return {
            "sorted": q_sorted,
            "mean": q_mean,
            "std": q_std,
            "cvar": q_cvar,
            "downside": downside,
            "risk": dist_risk,}

    def QuantileHuberLoss(
        self,
        quantiles: torch.Tensor,
        target: torch.Tensor,
        kappa: float = 1.0,
        sampleWeight: Optional[torch.Tensor] = None,) -> torch.Tensor:
        err = target - quantiles # [B,N]
        abs_err = err.abs()
        k = float(kappa)
        huber = torch.where(abs_err <= k, 0.5 * err.pow(2), k * (abs_err - 0.5 * k))
        tau = self.quantile_tau.view(1, -1)
        weight = (tau - (err.detach() < 0).float()).abs()
        loss_per_row = (weight * huber).mean(dim=-1) # [B]
        if sampleWeight is None:
            return loss_per_row.mean()
        sw = sampleWeight.view(-1)
        return (loss_per_row * sw).sum() / sw.sum().clamp_min(1.0)

    @staticmethod
    def QuantileCrossingLoss(quantiles: torch.Tensor, sampleWeight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if quantiles.size(-1) <= 1:
            return quantiles.new_zeros(())
        loss_per_row = F.relu(quantiles[:, :-1] - quantiles[:, 1:]).mean(dim=-1) # [B]
        if sampleWeight is None:
            return loss_per_row.mean()
        sw = sampleWeight.view(-1)
        return (loss_per_row * sw).sum() / sw.sum().clamp_min(1.0)

    def ManifoldLocalLog(
        self,
        zBase: torch.Tensor,
        zTarget: torch.Tensor,
        stepScale: float = 1.0,
        eps: float = 1e-4,) -> torch.Tensor:
        z0 = zBase.clamp(-1.0 + float(eps), 1.0 - float(eps))
        z1 = zTarget.clamp(-1.0 + float(eps), 1.0 - float(eps))
        pre0 = 0.5 * (torch.log1p(z0) - torch.log1p(-z0))
        pre1 = 0.5 * (torch.log1p(z1) - torch.log1p(-z1))
        scale = max(float(stepScale), float(eps))
        return ((pre1 - pre0) / scale).clamp(-5.0, 5.0)

    def BuildValueTensor(self, h: torch.Tensor) -> torch.Tensor:
        value_raw = self.value_tensor_out_norm(self.value_tensor_tail(h))
        scale = F.softplus(self.value_tensor_log_scale).clamp(0.05, 2.0)
        return torch.tanh(value_raw) * scale

    def BuildValueGraph(
        self,
        h: torch.Tensor,) -> Dict[str, torch.Tensor]:
        value_quantiles = self.quantile_adapter(h) # [B,Nq]
        dist_stats = self.DistributionStats(value_quantiles)
        value_ensemble = torch.cat([head(h) for head in self.value_ensemble_heads], dim=-1) # [B,K]
        value_epistemic = value_ensemble.var(dim=-1, unbiased=False) # [B]
        value = self.BuildValueTensor(h)
        return {
            "value": value,
            "value_quantiles": value_quantiles,
            "value_ensemble": value_ensemble,
            "value_epistemic": value_epistemic,
            "dist_stats": dist_stats,
            "h": h,}

    def BuildTDGraph(
        self,
        tdCurrent: torch.Tensor,
        commitMask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        td_abs = tdCurrent.detach().abs().squeeze(-1)
        self.td_out_ema.Update(td_abs, updateMask=commitMask)
        td_mean = self.td_out_ema.mean
        td_std = (self.td_out_ema.var + 1e-6).sqrt()
        td_scale = (td_mean + 2.0 * td_std).clamp_min(self.td_scale_min)
        td_bounded = torch.tanh(tdCurrent.squeeze(-1) / td_scale)
        return {
            "td_raw": tdCurrent,
            "td_bounded": td_bounded,
            "td_mean": td_mean,
            "td_std": td_std,
            "td_scale": td_scale,}

    def BuresWassersteinEnergy(self, valueTensor: torch.Tensor, valueNextTensor: torch.Tensor) -> torch.Tensor:
        B = valueTensor.size(0)
        eps = self.td_geom_eps
        x = valueTensor.view(B, self.td_bures_slots, self.td_bures_dim)
        y = valueNextTensor.view(B, self.td_bures_slots, self.td_bures_dim)
        mean_x = x.mean(dim=1)
        mean_y = y.mean(dim=1)
        x = x - mean_x.unsqueeze(1)
        y = y - mean_y.unsqueeze(1)
        denom = float(self.td_bures_slots - 1)
        cov_x = x.transpose(1, 2).matmul(x) / denom
        cov_y = y.transpose(1, 2).matmul(y) / denom
        eye = torch.eye(self.td_bures_dim, device=valueTensor.device, dtype=valueTensor.dtype).unsqueeze(0)
        cov_x = cov_x + eps * eye
        cov_y = cov_y + eps * eye
        chol_x = torch.linalg.cholesky(cov_x)
        chol_y = torch.linalg.cholesky(cov_y)

        def Directional(
            mean_a: torch.Tensor,
            chol_a: torch.Tensor,
            mean_b: torch.Tensor,
            chol_b: torch.Tensor,) -> torch.Tensor:
            with torch.no_grad():
                u, _, vh = torch.linalg.svd(
                    chol_a.transpose(-1, -2).matmul(chol_b),
                    full_matrices=False)
                alignment = vh.transpose(-1, -2).matmul(u.transpose(-1, -2))
            covariance_residual = chol_a - chol_b.matmul(alignment)
            return (
                (mean_a - mean_b).pow(2).sum(dim=-1)
                + covariance_residual.pow(2).sum(dim=(-2, -1)))

        return 0.5 * (
            Directional(mean_x, chol_x, mean_y, chol_y)
            + Directional(mean_y, chol_y, mean_x, chol_x))

    def HeatKernelEnergy(self, delta: torch.Tensor) -> torch.Tensor:
        basis = F.normalize(self.td_heat_basis, dim=-1)
        coeff = F.linear(delta, basis)
        eig = F.softplus(self.td_heat_log_eigs).clamp_min(self.td_geom_eps)
        heat = (1.0 - torch.exp(-eig)).pow(2)
        return (coeff.pow(2) * heat.unsqueeze(0)).sum(dim=-1)

    def SinkhornOTEnergy(self, valueTensor: torch.Tensor, valueNextTensor: torch.Tensor) -> torch.Tensor:
        B = valueTensor.size(0)
        tau = 0.10
        x = valueTensor.view(B, self.td_ot_bins, self.td_ot_bin_width).mean(dim=-1)
        y = valueNextTensor.view(B, self.td_ot_bins, self.td_ot_bin_width).mean(dim=-1)

        def LogProbability(z: torch.Tensor) -> torch.Tensor:
            log_mass = F.softplus(z).log()
            return log_mass - torch.logsumexp(log_mass, dim=-1, keepdim=True)

        log_p = LogProbability(x)
        log_q = LogProbability(y)
        emb = F.normalize(self.td_ot_cost_embed, dim=-1)
        cost = torch.cdist(emb, emb, p=2).pow(2)
        cost = cost / cost.mean()

        def RegularizedOT(log_source: torch.Tensor, log_target: torch.Tensor) -> torch.Tensor:
            log_kernel = -cost / tau
            log_u = torch.zeros_like(log_source)
            log_v = torch.zeros_like(log_target)
            for _ in range(self.td_ot_iters):
                log_u = log_source - torch.logsumexp(
                    log_kernel.unsqueeze(0) + log_v.unsqueeze(1),
                    dim=2)
                log_v = log_target - torch.logsumexp(
                    log_kernel.unsqueeze(0) + log_u.unsqueeze(2),
                    dim=1)
            log_plan = (
                log_u.unsqueeze(2)
                + log_kernel.unsqueeze(0)
                + log_v.unsqueeze(1))
            plan = log_plan.exp()
            transport = (plan * cost.unsqueeze(0)).sum(dim=(1, 2))
            log_reference = log_source.unsqueeze(2) + log_target.unsqueeze(1)
            reference = log_reference.exp()
            kl = plan * (log_plan - log_reference) - plan + reference
            return transport + tau * kl.sum(dim=(1, 2))

        cross = 0.5 * (
            RegularizedOT(log_p, log_q)
            + RegularizedOT(log_q, log_p))
        return (
            cross
            - 0.5 * RegularizedOT(log_p, log_p)
            - 0.5 * RegularizedOT(log_q, log_q))

    def BuildPhysicalTD(
        self,
        valueTensor: torch.Tensor,
        valueNextTensor: torch.Tensor,
        h: torch.Tensor,) -> Dict[str, torch.Tensor]:
        delta = valueNextTensor - valueTensor
        eps = self.td_geom_eps
        kappa = self.td_huber_kappa
        abs_delta = delta.abs()
        huber_per = torch.where(
            abs_delta <= kappa,
            0.5 * delta.pow(2),
            kappa * (abs_delta - 0.5 * kappa))

        sigma2 = F.softplus(self.latent_mahalanobis_logvar).clamp_min(eps)
        diag_energy = (delta.pow(2) / sigma2.unsqueeze(0)).sum(dim=-1)
        low_rank_energy = F.linear(delta, self.latent_L_factor).pow(2).sum(dim=-1)
        td_dirichlet = (diag_energy + low_rank_energy).clamp_min(eps).sqrt()

        l_diag = F.softplus(self.latent_L_diag)
        channel_grad = delta[:, 1:] - delta[:, :-1]
        sob_local = delta.pow(2).mean(dim=-1)
        sob_field = channel_grad.pow(2).mean(dim=-1)
        sob_metric = (delta.pow(2) * l_diag.unsqueeze(0)).mean(dim=-1)
        td_sobolev = (sob_local + self.td_geom_sob_alpha * (sob_field + sob_metric)).clamp_min(eps).sqrt()

        energy_now = valueTensor.pow(2).mean(dim=-1).clamp_min(eps).sqrt()
        energy_next = valueNextTensor.pow(2).mean(dim=-1).clamp_min(eps).sqrt()
        stat = torch.stack([
            delta.mean(dim=-1),
            delta.std(dim=-1, unbiased=False),
            delta.abs().mean(dim=-1),
            delta.pow(2).mean(dim=-1).clamp_min(eps).sqrt(),
            delta.abs().amax(dim=-1),
            energy_next - energy_now], dim=-1)
        context_logits = self.td_context_head(torch.cat([h, stat], dim=-1))
        context_w = F.softmax(context_logits, dim=-1)
        td_context = (context_w * huber_per).sum(dim=-1).clamp_min(eps).sqrt()

        td_bures = self.BuresWassersteinEnergy(valueTensor, valueNextTensor)
        td_heat = self.HeatKernelEnergy(delta).clamp_min(eps).sqrt()
        td_ot = self.SinkhornOTEnergy(valueTensor, valueNextTensor)

        metric_scale = F.softplus(self.physical_metric_log_scale).clamp_min(eps)
        component_stack = torch.stack([
            td_dirichlet / metric_scale[0],
            td_sobolev / metric_scale[1],
            td_context / metric_scale[2],
            td_bures / metric_scale[3],
            td_heat / metric_scale[4],
            td_ot / metric_scale[5]], dim=-1)
        component_w = F.softmax(self.physical_td_logits, dim=-1)
        td_mag = (component_w.unsqueeze(0) * component_stack).sum(dim=-1)
        td_sign = torch.tanh((energy_next.detach() - energy_now.detach()) / self.td_geom_sign_tau)
        td_scalar_train = td_sign * td_mag

        return {
            "td_tensor": delta,
            "td_scalar_train": td_scalar_train,
            "td_mag": td_mag,
            "td_sign": td_sign,
            "td_dirichlet": td_dirichlet,
            "td_sobolev": td_sobolev,
            "td_context": td_context,
            "td_bures": td_bures,
            "td_heat": td_heat,
            "td_ot": td_ot,
            "td_component_w": component_w,
            "td_context_w": context_w,}

    def RowDirectionDiversityLoss(self, x: torch.Tensor) -> torch.Tensor:
        rows = F.normalize(x, dim=-1)
        gram = rows.matmul(rows.transpose(0, 1))
        eye = torch.eye(rows.size(0), device=x.device, dtype=x.dtype)
        return (gram * (1.0 - eye)).pow(2).mean()

    def BranchStructureLoss(
        self,
        branchNext: torch.Tensor,
        branchWeight: torch.Tensor,
        sampleWeight: Optional[torch.Tensor] = None,) -> torch.Tensor:
        centered = branchNext - branchNext.mean(dim=1, keepdim=True)
        branch_vec = F.normalize(centered, dim=-1)
        gram = branch_vec.matmul(branch_vec.transpose(1, 2))
        eye = torch.eye(branchNext.size(1), device=branchNext.device, dtype=branchNext.dtype).unsqueeze(0)
        pair_weight = branchWeight.unsqueeze(2) * branchWeight.unsqueeze(1)
        offdiag = 1.0 - eye
        certainty = branchWeight.max(dim=-1).values.detach()
        uncertainty = (1.0 - certainty).detach()
        diversity = (gram.pow(2) * pair_weight.detach() * offdiag).sum(dim=(1, 2)) / (pair_weight.detach() * offdiag).sum(dim=(1, 2)).clamp_min(1e-6)
        branch_mean = (branchWeight.detach().unsqueeze(-1) * branchNext).sum(dim=1, keepdim=True)
        consistency = (branchWeight.detach().unsqueeze(-1) * (branchNext - branch_mean).pow(2)).mean(dim=(1, 2))
        loss_per_row = uncertainty * diversity + certainty * consistency
        if sampleWeight is None:
            return loss_per_row.mean()
        weight = sampleWeight.view(-1)
        return (loss_per_row * weight).sum() / weight.sum().clamp_min(1.0)

    def BuildPhysicalTDParameterRegularizer(self) -> Dict[str, torch.Tensor]:
        eps = self.td_geom_eps

        sigma2 = F.softplus(self.latent_mahalanobis_logvar).clamp_min(eps)
        l_diag = F.softplus(self.latent_L_diag).clamp_min(eps)
        metric_scale = F.softplus(self.physical_metric_log_scale).clamp_min(eps)
        heat_eig = F.softplus(self.td_heat_log_eigs).clamp_min(eps)
        component_w = F.softmax(self.physical_td_logits, dim=-1)

        scale_reg = (
            sigma2.log().pow(2).mean()
            + l_diag.log().pow(2).mean()
            + metric_scale.log().pow(2).mean()
            + heat_eig.log().pow(2).mean())

        low_rank_reg = (
            self.latent_L_factor.pow(2).mean()
            + self.RowDirectionDiversityLoss(self.latent_L_factor))

        heat_basis_reg = (
            self.RowDirectionDiversityLoss(self.td_heat_basis)
            + self.td_heat_basis.pow(2).mean() * self.td_geom_eps)

        emb = F.normalize(self.td_ot_cost_embed, dim=-1)
        cost = torch.cdist(emb, emb, p=2).pow(2)
        eye = torch.eye(self.td_ot_bins, device=emb.device, dtype=torch.bool)
        cost_off = cost[~eye]
        ot_cost_reg = (cost_off.mean() - 2.0).pow(2) + torch.exp(-cost_off).mean()

        uniform = component_w.new_full(component_w.shape, 1.0 / float(component_w.numel()))
        component_reg = (component_w - uniform).pow(2).mean()

        context_reg = self.td_context_head[-1].weight.pow(2).mean() + self.td_context_head[-1].bias.pow(2).mean()

        total = (
            self.latent_mahalanobis_logvar.new_tensor(1e-4) * scale_reg
            + self.latent_mahalanobis_logvar.new_tensor(1e-5) * low_rank_reg
            + self.latent_mahalanobis_logvar.new_tensor(1e-5) * heat_basis_reg
            + self.latent_mahalanobis_logvar.new_tensor(1e-5) * ot_cost_reg
            + self.latent_mahalanobis_logvar.new_tensor(1e-4) * component_reg
            + self.latent_mahalanobis_logvar.new_tensor(1e-6) * context_reg)

        return {
            "loss": total,
            "scale": scale_reg,
            "low_rank": low_rank_reg,
            "heat_basis": heat_basis_reg,
            "ot_cost": ot_cost_reg,
            "component": component_reg,
            "context": context_reg,}

    def TransportGradBucketNorm(self, bucket: Dict[str, torch.Tensor]) -> float:
        total_sq = 0.0
        for g in bucket.values():
            total_sq += float(g.pow(2).sum().item())
        return math.sqrt(max(total_sq, 0.0))

    def AddTransportGrad(self, bucket: Dict[str, torch.Tensor], name: str, grad: torch.Tensor):
        g = grad.detach()
        if name in bucket:
            bucket[name].add_(g)
        else:
            bucket[name] = g

    def BuildTransportSnapshotGraph(
        self,
        h: torch.Tensor,
        value: torch.Tensor,
        gradBucket: Dict[str, torch.Tensor],
        hookSeen: set,
        transportState: Optional[Dict[str, torch.Tensor]] = None,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        params = {}
        for name, p in self.transport.named_parameters():
            source = (
                p
                if transportState is None
                else transportState[name])
            snap = source.detach().clone().requires_grad_(p.requires_grad)
            if snap.requires_grad:
                def SnapshotHook(
                    grad: torch.Tensor,
                    *,
                    bucket: Dict[str, torch.Tensor] = gradBucket,
                    seen: set = hookSeen,
                    param_name: str = name,) -> torch.Tensor:
                    g = grad.detach()
                    if param_name in bucket:
                        bucket[param_name].add_(g)
                    else:
                        bucket[param_name] = g
                    seen.add(param_name)
                    return grad
                snap.register_hook(SnapshotHook)
            params[name] = snap
        buffers = {
            name: (
                b
                if transportState is None
                else transportState[name])
            for name, b in self.transport.named_buffers()}
        transport_state = {}
        transport_state.update(params)
        transport_state.update(buffers)
        return torch_functional_call(
            self.transport,
            transport_state,
            (h, value),
            {"returnExtras": True})

    @torch.no_grad()
    def ExportTransportSnapshotState(self) -> Dict[str, torch.Tensor]:
        state = {
            name: parameter.detach().clone()
            for name, parameter in self.transport.named_parameters()}
        state.update({
            name: buffer.detach().clone()
            for name, buffer in self.transport.named_buffers()})
        return state

    def InstallTransportGradHooks(self):
        for handle in self._transport_grad_hooks:
            handle.remove()
        self._transport_grad_hooks = []
        for name, p in self.transport.named_parameters():
            if p.requires_grad:
                def MainHook(
                    grad: torch.Tensor,
                    *,
                    param_name: str = name,) -> torch.Tensor:
                    self.AddTransportGrad(self._transport_curr_grad, param_name, grad)
                    self._transport_curr_grad_hook_seen.add(param_name)
                    return torch.zeros_like(grad)
                self._transport_grad_hooks.append(p.register_hook(MainHook))

    @torch.no_grad()
    def ClearTransportGradAccumulator(self):
        self._transport_prev_grad.clear()
        self._transport_curr_grad.clear()
        self._transport_delayed_ready = False
        self._transport_grad_accum_steps = 0
        self._transport_prev_grad_hook_seen.clear()
        self._transport_curr_grad_hook_seen.clear()

    @torch.no_grad()
    def CaptureTransportGrad(
        self,
        clearParamGrad: bool = True,) -> Dict[str, float]:
        prev_count = len(self._transport_prev_grad_hook_seen)
        curr_count = len(self._transport_curr_grad_hook_seen)
        captured = prev_count + curr_count
        if clearParamGrad:
            for p in self.transport.parameters():
                p.grad = None
        if captured <= 0:
            return {
                "captured": 0.0,
                "grad_norm": 0.0,
                "accum_steps": float(self._transport_grad_accum_steps),}

        if prev_count > 0:
            self._transport_delayed_ready = True
            grad_norm = self.TransportGradBucketNorm(self._transport_prev_grad)
        else:
            grad_norm = self.TransportGradBucketNorm(self._transport_curr_grad)
        self._transport_prev_grad_hook_seen.clear()
        self._transport_curr_grad_hook_seen.clear()
        return {
            "captured": float(captured),
            "grad_norm": grad_norm,
            "accum_steps": float(self._transport_grad_accum_steps),}

    @torch.no_grad()
    def RotateTransportOnlineGrad(self):
        self._transport_prev_grad.clear()
        for name, grad in self._transport_curr_grad.items():
            self._transport_prev_grad[name] = grad
        self._transport_curr_grad.clear()

    @torch.no_grad()
    def ApplyTransportManualGrad(
        self,
        lr: float,
        maxNorm: Optional[float] = None,
        weightDecay: float = 0.0,
        clear: bool = True,) -> Dict[str, float]:
        if not self._transport_delayed_ready:
            waiting_norm = self.TransportGradBucketNorm(self._transport_curr_grad)
            self.RotateTransportOnlineGrad()
            return {
                "updated": 0.0,
                "grad_norm": waiting_norm,
                "scale": 0.0,}

        grad_by_name: Dict[str, torch.Tensor] = self._transport_prev_grad

        if len(grad_by_name) <= 0:
            if clear:
                self._transport_delayed_ready = False
                self.RotateTransportOnlineGrad()
            return {"updated": 0.0, "grad_norm": 0.0, "scale": 1.0}

        total_sq = 0.0
        for g in grad_by_name.values():
            total_sq += float(g.pow(2).sum().item())
        grad_norm = math.sqrt(max(total_sq, 0.0))
        scale = 1.0
        if maxNorm is not None and float(maxNorm) > 0.0:
            scale = min(1.0, float(maxNorm) / (grad_norm + 1e-12))
        updated = 0
        lr_f = float(lr)
        wd_f = float(weightDecay)
        for name, p in self.transport.named_parameters():
            g = grad_by_name.get(name)
            if g is None:
                continue
            if wd_f != 0.0:
                p.mul_(1.0 - lr_f * wd_f)
            p.add_(g, alpha=-lr_f * scale)
            updated += 1
        if updated > 0:
            self._transport_grad_accum_steps += 1
        if clear:
            self._transport_delayed_ready = False
            self.RotateTransportOnlineGrad()
        return {
            "updated": float(updated),
            "grad_norm": grad_norm,
            "scale": float(scale),
            "accum_steps": float(self._transport_grad_accum_steps),}

    def NormalizeStreamIds(self, B: int, streamIds: Optional[torch.Tensor]) -> List[int]:
        if streamIds is None:
            normalized = list(range(B))
        elif torch.is_tensor(streamIds):
            normalized = [int(v) for v in streamIds.view(-1).tolist()]
        else:
            normalized = [int(v) for v in streamIds]
        self._active_stream_ids = normalized
        return normalized

    def StackLiveRows(self, items: List[Any]) -> Any:
        first = items[0]
        if torch.is_tensor(first):
            if first.dim() == 0:
                return torch.stack(items, dim=0).mean()
            return torch.cat(items, dim=0)
        if isinstance(first, dict):
            return {k: self.StackLiveRows([it[k] for it in items if k in it]) for k in first.keys()}
        return first

    def SelectBatchRow(self, value: Any, row: int, batchSize: int) -> Any:
        if torch.is_tensor(value):
            if value.dim() > 0 and int(value.size(0)) == int(batchSize):
                return value[row:row + 1]
            return value
        if isinstance(value, dict):
            return {
                key: self.SelectBatchRow(item, row, batchSize)
                for key, item in value.items()}
        return value

    def CacheDelayedTransitionInputs(
        self,
        transportHidden: torch.Tensor,
        transportValue: torch.Tensor,
        rewardNext: torch.Tensor,
        transportState: Dict[str, torch.Tensor],
        microGraph: Dict[str, torch.Tensor],
        returnHidden: torch.Tensor,
        alive: torch.Tensor,
        streamIds: Optional[torch.Tensor] = None,
        commitMask: Optional[torch.Tensor] = None):
        B = int(transportHidden.size(0))
        sids = self.NormalizeStreamIds(B, streamIds)
        alive_f = alive.detach().view(B)
        if commitMask is None:
            commit_mask = torch.ones(
                B, device=transportHidden.device, dtype=torch.bool)
        elif (
            not torch.is_tensor(commitMask)
            or tuple(commitMask.shape) != (B,)
            or commitMask.device != transportHidden.device
            or commitMask.dtype != torch.bool
        ):
            raise ValueError("commitMask must be a batched boolean mask")
        else:
            commit_mask = commitMask

        for i, sid in enumerate(sids):
            if not bool(commit_mask[i].item()):
                continue
            item = {
                "transport_hidden": transportHidden[i:i + 1].detach(),
                "transport_value": transportValue[i:i + 1].detach(),
                "reward_next": rewardNext[i:i + 1].detach(),
                "transport_state": transportState,
                "micro_graph": self.SelectBatchRow(microGraph, i, B),
                "return_hidden": returnHidden[i:i + 1].detach(),
                "alive": alive_f[i],}
            q = self._pending_transitions.setdefault(int(sid), deque())
            q.append(item)
            while len(q) > int(self._max_pending_per_stream):
                q.popleft()

    def AfterOptimizerStep(self):
        pass

    def BuildUncertaintyGraph(
        self,
        distStats: Dict[str, torch.Tensor],
        valueEpistemic: torch.Tensor,
        tdBounded: torch.Tensor,
        uncTotal: torch.Tensor,
        physicalTd: Dict[str, torch.Tensor],
        branchSpread: torch.Tensor,
        terminationRisk: torch.Tensor,) -> Dict[str, torch.Tensor]:
        base = float(self.unc_core.eps_prior)
        unc_adj = uncTotal.view(-1) - base
        unc_prior01 = 1.0 - torch.exp(-unc_adj / self.unc_tau)
        unc_prior_evidence = unc_prior01.detach()

        unc_epistemic01 = 1.0 - torch.exp(-valueEpistemic.detach())
        risk_dist = distStats["risk"].detach()
        td_abs = tdBounded.detach().abs()
        risk_base = 1.0 - torch.exp(-(td_abs + unc_prior_evidence))
        ambiguity = unc_prior_evidence
        surprise = 1.0 - torch.exp(-td_abs)
        confidence_base = torch.exp(-(
            0.50 * risk_base
            + 0.35 * ambiguity
            + 0.15 * surprise))
        physical_raw = (
            0.28 * physicalTd["td_dirichlet"]
            + 0.24 * physicalTd["td_sobolev"]
            + 0.18 * physicalTd["td_context"]
            + 0.12 * physicalTd["td_bures"]
            + 0.10 * physicalTd["td_heat"]
            + 0.08 * physicalTd["td_ot"])
        physical01 = (1.0 - torch.exp(-physical_raw)).detach()
        branch01 = 1.0 - torch.exp(-branchSpread.detach())
        termination01 = terminationRisk.detach().view(-1).clamp(0.0, 1.0)
        risk = (
            0.24 * risk_base
            + 0.18 * physical01
            + 0.16 * risk_dist
            + 0.14 * unc_prior_evidence
            + 0.12 * unc_epistemic01
            + 0.08 * branch01
            + 0.08 * termination01)
        confidence_dist = torch.exp(-(unc_prior_evidence + unc_epistemic01 + risk))
        confidence = 0.5 * confidence_base + 0.5 * confidence_dist
        learned_unc = (
            0.30 * risk
            + 0.25 * ambiguity
            + 0.20 * surprise
            + 0.25 * (1.0 - confidence))
        unc01 = (
            0.35 * unc_prior_evidence
            + 0.25 * learned_unc
            + 0.20 * unc_epistemic01
            + 0.10 * risk)
        precision = (confidence * (1.0 - unc01)).clamp_min(0.05)
        unc_pred = 0.25 * (risk + ambiguity + surprise + (1.0 - confidence))
        return {
            "unc_pred": unc_pred,
            "risk": risk,
            "ambiguity": ambiguity,
            "surprise": surprise,
            "confidence": confidence,
            "learned_unc": learned_unc,
            "unc01": unc01,
            "precision": precision,
            "risk_physical": physical01,
            "risk_branch": branch01,
            "risk_termination": termination01,
            "unc_prior01": unc_prior_evidence,
            "unc_epistemic01": unc_epistemic01,}

    def RefineEntityOntologyRisk(
        self,
        output: GeoTropicalOut,
        physicalState: Dict[str, torch.Tensor],
        sampleMask: Optional[torch.Tensor] = None,) -> GeoTropicalOut:
        realm = physicalState["RealmProb"]
        motion_layer = physicalState["MotionLayerProb"]
        layer_agency = physicalState["LayerAgencyProb"]
        presence = physicalState["PerceptualPresence"]
        physical_entity = physicalState["PhysicalEntityProb"]
        physical_interaction = physicalState["PhysicalInteractionProb"]
        body_membership = physicalState["BodyMembershipProb"]
        verification = physicalState["VerificationConfidence"]
        contact = physicalState["ContactProbRaw"]

        physical_realm = torch.stack([
            realm[..., ONTOLOGY_REALM_SELF],
            realm[..., ONTOLOGY_REALM_EXTERNAL],], dim=-1)
        physical_layers = torch.stack([
            motion_layer[..., ONTOLOGY_MOTION_CARRIER],
            motion_layer[..., ONTOLOGY_MOTION_ARTICULATION],], dim=-1)
        physical_layer_agency = torch.stack([
            layer_agency[..., ONTOLOGY_MOTION_CARRIER, :],
            layer_agency[..., ONTOLOGY_MOTION_ARTICULATION, :],], dim=-2)
        physical_token = torch.cat([
            physical_realm,
            physical_layers,
            physical_layer_agency.flatten(start_dim=-2),
            body_membership.unsqueeze(-1),
            physical_entity.unsqueeze(-1),
            physical_interaction.unsqueeze(-1),
            presence.unsqueeze(-1),
            verification.unsqueeze(-1),
            contact.unsqueeze(-1),], dim=-1)

        physical_support = (
            presence
            * physical_entity
            * physical_realm.sum(dim=-1))
        pooled_physical = (
            physical_token * physical_support.unsqueeze(-1)
        ).sum(dim=1) / physical_support.sum(dim=1, keepdim=True).clamp_min(1e-6)
        physical_risk_raw = self.entity_ontology_physical_risk_head(
            pooled_physical).squeeze(-1)

        non_self_agency = (
            physical_layer_agency[..., ONTOLOGY_AGENCY_EXTERNAL]
            + physical_layer_agency[..., ONTOLOGY_AGENCY_AUTONOMOUS]
            + physical_layer_agency[..., ONTOLOGY_AGENCY_MIXED])
        non_self_motion = 1.0 - torch.prod(
            1.0 - physical_layers * non_self_agency,
            dim=-1)
        interaction_event = 1.0 - (
            (1.0 - non_self_motion) * (1.0 - contact))
        physical_hazard_target = (
            physical_support
            * (0.5 + 0.5 * physical_interaction)
            * interaction_event).amax(dim=1)

        positive_risk_raw = torch.where(
            physical_risk_raw >= 0.0,
            physical_risk_raw,
            torch.zeros_like(physical_risk_raw))
        physical_risk_residual = (
            1.0 - torch.exp(-positive_risk_raw)
        ) * physical_support.amax(dim=1)

        virtual_support = (
            presence * realm[..., ONTOLOGY_REALM_VIRTUAL])
        virtual_animation = (
            virtual_support
            * (1.0 - (
                1.0 - motion_layer[..., ONTOLOGY_MOTION_SURFACE])
                * (1.0 - physicalState["ContentChangeProb"])))
        virtual_animation_salience = virtual_animation.amax(dim=1)
        visual_effect_salience = (
            presence
            * realm[..., ONTOLOGY_REALM_EFFECT]
            * motion_layer[..., 4]
        ).amax(dim=1)

        risk_base = output.rComps["risk"]
        risk_with_physical_residual = 1.0 - (
            (1.0 - risk_base) * (1.0 - physical_risk_residual))
        refined_risk = torch.where(
            physical_risk_residual > 0.0,
            risk_with_physical_residual,
            risk_base)
        risk_loss_rows = F.binary_cross_entropy_with_logits(
            physical_risk_raw,
            physical_hazard_target.detach(),
            reduction="none")
        if sampleMask is None:
            risk_loss = risk_loss_rows.mean()
        else:
            if (
                not torch.is_tensor(sampleMask)
                or tuple(sampleMask.shape) != (physical_risk_raw.size(0),)
                or sampleMask.device != physical_risk_raw.device
                or sampleMask.dtype != torch.bool
            ):
                raise ValueError("sampleMask must be a batched boolean mask")
            weight = sampleMask.to(dtype=risk_loss_rows.dtype)
            risk_loss = (
                risk_loss_rows * weight
            ).sum() / weight.sum().clamp_min(1.0)
        loss = output.loss
        if loss is not None:
            loss = loss + self.entity_ontology_risk_loss_weight * risk_loss

        r_comps = dict(output.rComps)
        r_comps.update({
            "risk": refined_risk.detach(),
            "riskBase": risk_base.detach(),
            "physicalEntityHazard": physical_hazard_target.detach(),
            "physicalRiskResidual": physical_risk_residual.detach(),
            "virtualAnimationSalience": virtual_animation_salience.detach(),
            "visualEffectSalience": visual_effect_salience.detach(),
            "lossEntityOntologyRisk": risk_loss.detach(),})
        return output._replace(loss=loss, rComps=r_comps)

    def ConsumePendingTransitions(
        self,
        B: int,
        valueLabel: torch.Tensor,
        zLabel: torch.Tensor,
        streamIds: Optional[torch.Tensor],
        commitMask: Optional[torch.Tensor] = None,) -> Dict[str, Any]:
        target_empty = valueLabel.new_zeros((B,) + tuple(valueLabel.shape[1:]))
        prev = {
            "ready": False,
            "pred": target_empty,
            "transp_extras": {},
            "loss_mask": valueLabel.new_zeros((0,)),
            "rows": torch.empty(0, device=valueLabel.device, dtype=torch.long),}

        sids = self.NormalizeStreamIds(B, streamIds)
        if commitMask is None:
            commit_mask = torch.ones(
                B, device=valueLabel.device, dtype=torch.bool)
        elif (
            not torch.is_tensor(commitMask)
            or tuple(commitMask.shape) != (B,)
            or commitMask.device != valueLabel.device
            or commitMask.dtype != torch.bool
        ):
            raise ValueError("commitMask must be a batched boolean mask")
        else:
            commit_mask = commitMask
        rows: List[int] = []
        items: List[Dict[str, Any]] = []
        for row, sid in enumerate(sids):
            if not bool(commit_mask[row].item()):
                continue
            q = self._pending_transitions.get(int(sid))
            if q is None or len(q) <= 0:
                continue
            rows.append(int(row))
            items.append(q.popleft())
            if len(q) <= 0:
                self._pending_transitions.pop(int(sid), None)

        if len(items) <= 0:
            return prev

        rows_t = torch.as_tensor(rows, device=valueLabel.device, dtype=torch.long)
        alive = torch.stack([it["alive"].view(()) for it in items], dim=0)

        target = valueLabel.index_select(0, rows_t)
        z_target = zLabel.index_select(0, rows_t)
        grouped: Dict[int, Dict[str, Any]] = {}
        for position, item in enumerate(items):
            group = grouped.setdefault(id(item["transport_state"]), {
                "state": item["transport_state"],
                "positions": [],
                "items": []})
            group["positions"].append(position)
            group["items"].append(item)

        rebuilt_items: List[Optional[Dict[str, Any]]] = [None] * len(items)
        for group in grouped.values():
            group_items = group["items"]
            group_hidden = torch.cat([
                item["transport_hidden"]
                for item in group_items], dim=0)
            group_value = torch.cat([
                item["transport_value"]
                for item in group_items], dim=0)
            group_reward = torch.cat([
                item["reward_next"]
                for item in group_items], dim=0)
            group_micro_graph = self.StackLiveRows([
                item["micro_graph"]
                for item in group_items])
            pred_item, extras_item = self.BuildTransportSnapshotGraph(
                group_hidden,
                group_value,
                self._transport_prev_grad,
                self._transport_prev_grad_hook_seen,
                transportState=group["state"])
            pred_item, extras_item = self.ApplyRewardNextModulation(
                group_value,
                pred_item,
                extras_item,
                group_reward)
            pred_item, extras_item = self.ApplyMicroGraphPrior(
                pred_item,
                extras_item,
                group_micro_graph)
            group_size = len(group_items)
            for local_row, position in enumerate(group["positions"]):
                rebuilt_items[position] = {
                    "pred": pred_item[local_row:local_row + 1],
                    "transp_extras": self.SelectBatchRow(
                        extras_item,
                        local_row,
                        group_size)}
        if any(item is None for item in rebuilt_items):
            raise RuntimeError("delayed transport reconstruction lost a stream row")
        pred = torch.cat([item["pred"] for item in rebuilt_items], dim=0)
        return_hidden = torch.cat([it["return_hidden"] for it in items], dim=0)
        transp_extras = self.StackLiveRows([
            item["transp_extras"]
            for item in rebuilt_items])
        manifold = transp_extras["manifold"]

        prev.update({
            "ready": True,
            "pred": pred,
            "return_hidden": return_hidden,
            "transp_extras": transp_extras,
            "target_m": target,
            "loss_mask": alive,
            "z": manifold["z"],
            "z_next": manifold["z_next"],
            "u": manifold["u"],
            "z_target": z_target,
            "manifold_reg_per_row": manifold["reg_per_row"],})
        prev["rows"] = rows_t
        return prev

    def forward(self,
        memoryPrev: torch.Tensor, # [B,memDim]
        attnPrev: torch.Tensor, # [B,attnDim]
        state: torch.Tensor, # [B,stateDim]
        *,
        rewardModel: torch.Tensor,
        doneModel: torch.Tensor,
        policyEntropyPrev: Optional[torch.Tensor] = None, # [B]
        worldDeltaTransport: Optional[torch.Tensor] = None, # [B,stateDim]
        worldDeltaPhysics: Optional[torch.Tensor] = None, # [B,stateDim]
        streamIds: Optional[torch.Tensor] = None,
        computeLoss: Optional[bool] = None,
        commitMask: Optional[torch.Tensor] = None,
        )-> GeoTropicalOut:

        B = state.size(0)
        self.EnsureB(B)
        if commitMask is None:
            commit_mask = torch.ones(B, device=state.device, dtype=torch.bool)
        elif (
            not torch.is_tensor(commitMask)
            or tuple(commitMask.shape) != (B,)
            or commitMask.device != state.device
            or commitMask.dtype != torch.bool
        ):
            raise ValueError("commitMask must be a batched boolean mask")
        else:
            commit_mask = commitMask

        def MaskedMean(value: torch.Tensor) -> torch.Tensor:
            per_row = value.reshape(B, -1).mean(dim=-1)
            weight = commit_mask.to(dtype=per_row.dtype)
            return (per_row * weight).sum() / weight.sum().clamp_min(1.0)

        if policyEntropyPrev is None:
            policyEntropyPrev = state.new_zeros(B)
        if worldDeltaTransport is None:
            worldDeltaTransport = state.new_zeros(B, self.unc_core.state_dim)
        if worldDeltaPhysics is None:
            worldDeltaPhysics = state.new_zeros(B, self.unc_core.state_dim)
        reward_value = rewardModel.detach().view(B)
        done_value = doneModel.detach().view(B)
        with torch.no_grad():
            r_next_hat = self.reward_predictor.PredictNext(
                reward_value.detach(),
                commitMask=commit_mask).view(B)
            done_next_hat = self.done_predictor.PredictNext(
                done_value.detach(),
                commitMask=commit_mask).view(B)

        x = torch.cat([memoryPrev, attnPrev, state], dim=-1)
        h = self.Trunk(x) # [B,H]

        emotion = self.emotion_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state,
            commitMask=commit_mask) # [B,emotionDim]
        h = self.FuseEmotionIntoHidden(h, emotion)
        cognitive_value = self.BuildCognitiveValue(h)
        cognitive_proxy = self.BuildCognitiveProxy(cognitive_value.feature)

        value_parts = self.BuildValueGraph(h=h)
        value = value_parts["value"]
        return_value = self.return_value_head(h).squeeze(-1)
        return_target_now = (
            reward_value
            + self.return_discount * (1.0 - done_value) * return_value.detach())
        return_advantage = torch.where(
            self.return_value_valid,
            return_target_now - self.return_value_prev,
            torch.zeros_like(return_target_now))
        self.return_value_prev = torch.where(
            commit_mask,
            return_value.detach(),
            self.return_value_prev)
        self.return_value_valid = self.return_value_valid | commit_mask
        value_epistemic = value_parts["value_epistemic"]
        dist_stats = value_parts["dist_stats"]

        transport_h = h.detach()
        transport_value = value.detach()

        value_next, transp_extras = self.transport(
            transport_h,
            transport_value,
            returnExtras=True)
        value_next, transp_extras = self.ApplyRewardNextModulation(
            transport_value,
            value_next,
            transp_extras,
            r_next_hat)
        manifold_out = transp_extras["manifold"]

        micro_graph = self.micro.Preview(
            value=transport_value,
            z=manifold_out["z"].detach())
        value_next, transp_extras = self.ApplyMicroGraphPrior(
            value_next,
            transp_extras,
            micro_graph)
        physical_td = self.BuildPhysicalTD(transport_value, value_next, transport_h)
        td_current = physical_td["td_scalar_train"].view(B, 1)
        td_graph = self.BuildTDGraph(
            td_current,
            commitMask=commit_mask)
        td_bounded = td_graph["td_bounded"] # [-1,1], [B]

        unc_total, _ = self.unc_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state,
            entropyPrev=policyEntropyPrev,
            tdCurr=td_bounded.detach(),
            worldDeltaTransport=worldDeltaTransport,
            worldDeltaPhysics=worldDeltaPhysics,
            doneCurr=done_value,
            commitMask=commit_mask) # unc_total:[B]

        branch_next = transp_extras.get("branch_next")
        transport_branch_std = (
            value.new_zeros((B,))
            if branch_next is None
            else branch_next.detach().view(B, -1).std(dim=-1, unbiased=False))

        uncertainty_graph = self.BuildUncertaintyGraph(
            distStats=dist_stats,
            valueEpistemic=value_epistemic,
            tdBounded=td_bounded,
            uncTotal=unc_total,
            physicalTd=physical_td,
            branchSpread=transport_branch_std,
            terminationRisk=done_next_hat)
        risk = uncertainty_graph["risk"]
        ambiguity = uncertainty_graph["ambiguity"]
        surprise = uncertainty_graph["surprise"]
        confidence = uncertainty_graph["confidence"]
        unc01 = uncertainty_graph["unc01"]
        precision = uncertainty_graph["precision"]
        rComps = {
            "risk": risk.detach(),
            "ambiguity": ambiguity.detach(),
            "surprise": surprise.detach(),
            "confidence": confidence.detach(),
            "terminationRisk": uncertainty_graph["risk_termination"].detach(),
            "coarseProgress": cognitive_value.coarseProgress.detach(),
            "detailProgress": cognitive_value.detailProgress.detach(),
            "planStaleness": cognitive_value.planStaleness.detach(),
            "replanBenefit": cognitive_value.replanBenefit.detach(),
            "computeCost": cognitive_value.computeCost.detach(),
            "feasibility": cognitive_value.feasibility.detach(),
            "safetyConstraint": cognitive_value.safetyConstraint.detach(),
            "executionContinuity": cognitive_proxy[
                "executionContinuity"].detach(),
            "sensoryReliability": cognitive_proxy[
                "sensoryReliability"].detach(),
            "replanUrgency": cognitive_proxy["replanUrgency"].detach(),
            "operationalConfidence": cognitive_proxy[
                "operationalConfidence"].detach(),
            "riskAbsence": cognitive_proxy["riskAbsence"].detach(),}

        self.micro.CommitStep(
            value=transport_value,
            valueNext=value_next,
            z=manifold_out["z"].detach(),
            alive=1.0 - done_value,
            commitMask=commit_mask)

        should_compute_loss = (
            bool(self.training)
            if computeLoss is None
            else bool(computeLoss))
        if not should_compute_loss:
            self.transport.CommitManifoldField(
                manifold_out["field"],
                commitMask=commit_mask)
            return GeoTropicalOut(
                value=value,
                valueNext=value_next,
                tdError=td_bounded.detach(),
                returnValue=return_value,
                returnAdvantage=return_advantage.detach(),
                loss=None,
                emotion=emotion,
                rComps=rComps,
                uncertainty=unc01,
                precision=precision,
                valueHidden=h,
                cognitiveValue=cognitive_value,
                extras=None,)

        prev = self.ConsumePendingTransitions(
            B=B,
            valueLabel=value.detach(),
            zLabel=manifold_out["z"].detach(),
            streamIds=streamIds,
            commitMask=commit_mask)

        has_prev_pred = bool(prev["ready"])

        loss_diff = value.new_zeros(())
        loss_diff_branch = value.new_zeros(())
        loss_branch_structure = value.new_zeros(())
        loss_manifold_geo = value.new_zeros(())
        loss_manifold_tangent = value.new_zeros(())
        loss_manifold_latent = value.new_zeros(())
        loss_manifold_reg = value.new_zeros(())
        loss_mask = prev["loss_mask"]
        valid_denom = loss_mask.sum().clamp_min(1.0)

        def ValidMeanM(vec: torch.Tensor) -> torch.Tensor:
            return (vec.view(-1) * loss_mask).sum() / valid_denom

        if has_prev_pred:
            target_m = prev["target_m"].detach()
            prev_pred = prev["pred"]
            loss_diff_vec = F.smooth_l1_loss(prev_pred, target_m, reduction="none").view(prev_pred.size(0), -1).mean(dim=-1)
            loss_diff = ValidMeanM(loss_diff_vec)

            prev_transp_extras = prev["transp_extras"]
            branch_next = prev_transp_extras["branch_next"]
            branch_w = prev_transp_extras["branch_w"]
            branch_mix = (branch_w.unsqueeze(-1) * branch_next).sum(dim=1)
            branch_loss_vec = F.smooth_l1_loss(
                branch_mix,
                target_m,
                reduction="none").view(branch_mix.size(0), -1).mean(dim=-1)
            loss_diff_branch = ValidMeanM(branch_loss_vec)
            loss_branch_structure = self.BranchStructureLoss(
                branch_next,
                branch_w,
                sampleWeight=loss_mask)

            loss_manifold_latent_vec = F.smooth_l1_loss(
                prev["z_next"],
                prev["z_target"].detach(),
                reduction="none").mean(dim=-1)
            loss_manifold_geo = ValidMeanM(loss_manifold_latent_vec)

            u_target = self.ManifoldLocalLog(
                prev["z"].detach(),
                prev["z_target"].detach(),
                stepScale=self.transport.manifold_step_scale)
            loss_manifold_tangent_vec = F.smooth_l1_loss(
                prev["u"],
                u_target,
                reduction="none").mean(dim=-1)
            loss_manifold_tangent = ValidMeanM(loss_manifold_tangent_vec)

            loss_manifold_latent = loss_manifold_geo + value.new_tensor(0.5) * loss_manifold_tangent

            loss_manifold_reg = ValidMeanM(prev["manifold_reg_per_row"])

        loss_transport = (
            value.new_tensor(self.wDiff) * loss_diff
            + value.new_tensor(self.wDiffBranch) * loss_diff_branch
            + value.new_tensor(self.wBranchStructure) * loss_branch_structure
            + value.new_tensor(self.wManifoldLatent) * loss_manifold_latent
            + loss_manifold_reg)
        loss_return = value.new_zeros(())
        if has_prev_pred:
            rows = prev["rows"]
            previous_return_value = self.return_value_head(
                prev["return_hidden"]).squeeze(-1)
            bellman_target = (
                reward_value.index_select(0, rows)
                + self.return_discount
                * (1.0 - done_value.index_select(0, rows))
                * return_value.detach().index_select(0, rows))
            return_loss_vec = F.smooth_l1_loss(
                previous_return_value,
                bellman_target,
                reduction="none")
            loss_return = ValidMeanM(return_loss_vec)
        loss_transport = loss_transport + loss_return
        loss_transport_delayed = loss_transport

        alive_current = (
            (1.0 - done_value)
            * commit_mask.to(dtype=done_value.dtype))
        current_denom = alive_current.sum().clamp_min(1.0)

        def CurrentMean(vec: torch.Tensor) -> torch.Tensor:
            return (vec.view(B, -1).mean(dim=-1) * alive_current).sum() / current_denom

        loss_physical_td = CurrentMean(F.smooth_l1_loss(
            td_bounded,
            td_bounded.new_zeros(td_bounded.shape),
            reduction="none"))

        loss_physical_aux = (
            value.new_tensor(0.02) * CurrentMean(physical_td["td_bures"])
            + value.new_tensor(0.01) * CurrentMean(physical_td["td_heat"])
            + value.new_tensor(0.005) * CurrentMean(physical_td["td_ot"]))

        physical_param_reg = self.BuildPhysicalTDParameterRegularizer()
        loss_physical_param_reg = (
            value.new_tensor(self.wPhysicalTDParamReg)
            * physical_param_reg["loss"]
            * commit_mask.any().to(dtype=value.dtype))

        loss_value_tensor_energy = value.new_tensor(1e-6) * CurrentMean(value.pow(2).mean(dim=-1))








        value_return_target = (
            (1.0 - done_value) * torch.tanh(return_value.detach()))
        loss_value_return_anchor = MaskedMean(F.smooth_l1_loss(
            value[:, 0],
            value_return_target,
            reduction="none"))

        quantile_target = (
            td_bounded.detach().view(-1, 1)
            + (self.quantile_tau.view(1, -1) - 0.5)
            * 2.0
            * physical_td["td_mag"].detach().view(-1, 1)).clamp(-3.0, 3.0)
        loss_quantile_fit = self.QuantileHuberLoss(
            value_parts["value_quantiles"],
            quantile_target,
            sampleWeight=alive_current)
        loss_quantile_order = self.QuantileCrossingLoss(
            value_parts["value_quantiles"],
            sampleWeight=alive_current)
        loss_quantile = (
            value.new_tensor(0.02) * loss_quantile_fit
            + value.new_tensor(0.001) * loss_quantile_order)

        ensemble_value = value_parts["value_ensemble"]
        ensemble_mean = ensemble_value.mean(dim=-1)
        ensemble_var = ensemble_value.var(dim=-1, unbiased=False)
        ensemble_mean_target = physical_td["td_mag"].detach()
        ensemble_var_target = (
            0.50 * physical_td["td_context"].detach()
            + 0.20 * physical_td["td_bures"].detach()
            + 0.20 * physical_td["td_heat"].detach()
            + 0.10 * physical_td["td_ot"].detach()).clamp_max(3.0)
        loss_ensemble = (
            value.new_tensor(0.01) * CurrentMean(F.smooth_l1_loss(
                ensemble_mean,
                ensemble_mean_target,
                reduction="none"))
            + value.new_tensor(0.005) * CurrentMean(F.smooth_l1_loss(
                ensemble_var,
                ensemble_var_target,
                reduction="none")))

        continuity_target = ((1.0 - done_value) * (1.0 - risk)).detach()
        reliability_target = (
            (1.0 - done_value)
            * confidence
            * (1.0 - surprise)).detach()
        urgency_target = (
            (
                surprise.detach()
                + risk.detach()
                + unc01.detach()
                + ambiguity.detach()
                + torch.sigmoid(policyEntropyPrev.detach())
            ) / 5.0)
        operational_target = (
            (1.0 - risk) * (1.0 - done_value)).detach()
        risk_absence_target = (1.0 - risk).detach()
        loss_cognitive_value = (
            MaskedMean(F.binary_cross_entropy(
                cognitive_proxy["executionContinuity"],
                continuity_target,
                reduction="none"))
            + MaskedMean(F.binary_cross_entropy(
                cognitive_proxy["sensoryReliability"],
                reliability_target,
                reduction="none"))
            + MaskedMean(F.smooth_l1_loss(
                cognitive_proxy["replanUrgency"],
                urgency_target,
                reduction="none"))
            + MaskedMean(F.binary_cross_entropy(
                cognitive_proxy["operationalConfidence"],
                operational_target,
                reduction="none"))
            + MaskedMean(F.binary_cross_entropy(
                cognitive_proxy["riskAbsence"],
                risk_absence_target,
                reduction="none"))) / 5.0

        self.CacheDelayedTransitionInputs(
            transportHidden=transport_h,
            transportValue=transport_value,
            rewardNext=r_next_hat,
            transportState=self.ExportTransportSnapshotState(),
            microGraph=micro_graph,
            returnHidden=h,
            alive=1.0 - done_value,
            streamIds=streamIds,
            commitMask=commit_mask)

        self.transport.CommitManifoldField(
            manifold_out["field"],
            commitMask=commit_mask)

        loss_current = (
            loss_physical_td
            + loss_physical_aux
            + loss_physical_param_reg
            + loss_value_tensor_energy
            + loss_value_return_anchor
            + loss_quantile
            + loss_ensemble
            + value.new_tensor(0.01) * loss_cognitive_value)

        extras = {
            "loss_transport": loss_transport_delayed.detach(),
            "loss_return": loss_return.detach(),
            "loss_physical_td": loss_physical_td.detach(),
            "loss_physical_aux": loss_physical_aux.detach(),
            "loss_physical_param_reg": loss_physical_param_reg.detach(),
            "loss_value_tensor_energy": loss_value_tensor_energy.detach(),
            "loss_value_return_anchor": loss_value_return_anchor.detach(),
            "loss_quantile": loss_quantile.detach(),
            "loss_ensemble": loss_ensemble.detach(),
            "loss_cognitive_value": loss_cognitive_value.detach(),
            "loss_cognitive_auxiliary": loss_cognitive_value.detach(),
            "loss_diff": loss_diff.detach(),
            "loss_diff_branch": loss_diff_branch.detach(),
            "loss_branch_structure": loss_branch_structure.detach(),
            "loss_manifold_geo": loss_manifold_geo.detach(),
            "loss_manifold_tangent": loss_manifold_tangent.detach(),
            "loss_manifold_latent": loss_manifold_latent.detach(),
            "td_mag": physical_td["td_mag"].detach(),
            "loss_current_graph": loss_current,
            "loss_cognitive_auxiliary_graph": loss_cognitive_value,
            "loss_transport_delayed_graph": loss_transport_delayed,}

        return GeoTropicalOut(
            value=value,
            valueNext=value_next,
            tdError=td_bounded.detach(),
            returnValue=return_value,
            returnAdvantage=return_advantage.detach(),
            loss=loss_current,
            emotion=emotion,
            rComps=rComps,
            uncertainty=unc01,
            precision=precision,
            valueHidden=h,
            cognitiveValue=cognitive_value,
            extras=extras,)


    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        self.emotion_core.ResetHebbianMemory(doneMask=doneMask)

    @torch.no_grad()
    def ResetState(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.return_value_prev.zero_()
            self.return_value_valid.zero_()
            self.micro.Reset()
            self.emotion_core.ResetState()
            self.td_out_ema.ResetAll()
            self.unc_core.ResetState()
            self.reward_predictor.Reset()
            self.done_predictor.Reset()
            self.transport.ResetState()
            self._pending_transitions.clear()
            self._last_batch_size = None
            self._active_stream_ids = None
            self.ClearTransportGradAccumulator()
            return
        self.return_value_prev.masked_fill_(doneMask.view(-1).bool(), 0.0)
        self.return_value_valid.logical_and_(~doneMask.view(-1).bool())
        self.micro.Reset(doneMask=doneMask)
        self.emotion_core.ResetState(doneMask=doneMask)
        self.td_out_ema.ResetAll(doneMask=doneMask)
        self.unc_core.ResetState(doneMask=doneMask)
        self.reward_predictor.Reset(doneMask=doneMask)
        self.done_predictor.Reset(doneMask=doneMask)
        self.transport.ResetState(doneMask=doneMask)
        self.ResetPrevTransition(doneMask)

    @torch.no_grad()
    def ResetPrevTransition(self, doneMask: torch.Tensor):
        mask = doneMask.bool().view(-1)
        rows = mask.nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        stream_ids = self._active_stream_ids
        for r in rows.detach().tolist():
            sid = int(r) if stream_ids is None else int(stream_ids[int(r)])
            self._pending_transitions.pop(sid, None)

    def SuspendTransientTrainingGraph(self) -> Dict[str, Any]:
        state = {
            "pending_transitions": self._pending_transitions,
            "active_stream_ids": self._active_stream_ids,
            "transport_prev_grad": self._transport_prev_grad,
            "transport_curr_grad": self._transport_curr_grad,
            "transport_delayed_ready": self._transport_delayed_ready,
            "transport_grad_accum_steps": self._transport_grad_accum_steps,
            "transport_prev_grad_hook_seen": self._transport_prev_grad_hook_seen,
            "transport_curr_grad_hook_seen": self._transport_curr_grad_hook_seen,}
        self._pending_transitions = {}
        self._active_stream_ids = None
        self._transport_prev_grad = {}
        self._transport_curr_grad = {}
        self._transport_delayed_ready = False
        self._transport_grad_accum_steps = 0
        self._transport_prev_grad_hook_seen = set()
        self._transport_curr_grad_hook_seen = set()
        return state

    def RestoreTransientTrainingGraph(self, state: Dict[str, Any]) -> None:
        self._pending_transitions = state["pending_transitions"]
        self._active_stream_ids = state["active_stream_ids"]
        self._transport_prev_grad = state["transport_prev_grad"]
        self._transport_curr_grad = state["transport_curr_grad"]
        self._transport_delayed_ready = state["transport_delayed_ready"]
        self._transport_grad_accum_steps = state["transport_grad_accum_steps"]
        self._transport_prev_grad_hook_seen = state["transport_prev_grad_hook_seen"]
        self._transport_curr_grad_hook_seen = state["transport_curr_grad_hook_seen"]

    @torch.no_grad()
    def ExportState(self) -> Dict[str, Any]:
        clone_memo: Dict[int, Any] = {}

        def CloneRuntime(value: Any) -> Any:
            value_id = id(value)
            if value_id in clone_memo:
                return clone_memo[value_id]
            if torch.is_tensor(value):
                cloned = value.detach().clone()
                clone_memo[value_id] = cloned
                return cloned
            if isinstance(value, dict):
                cloned: Dict[Any, Any] = {}
                clone_memo[value_id] = cloned
                cloned.update({
                    key: CloneRuntime(item)
                    for key, item in value.items()})
                return cloned
            if isinstance(value, (list, tuple, deque)):
                cloned_list: List[Any] = []
                clone_memo[value_id] = cloned_list
                cloned_list.extend(CloneRuntime(item) for item in value)
                return cloned_list
            return value

        state: Dict[str, Any] = {"ve_is_training": bool(self.training)}
        state["ve_last_batch_size"] = self._last_batch_size
        state["pending_transitions"] = {
            int(stream_id): [
                CloneRuntime(item)
                for item in queue]
            for stream_id, queue in self._pending_transitions.items()}
        state["active_stream_ids"] = (
            None
            if self._active_stream_ids is None
            else list(self._active_stream_ids))
        state["transport_prev_grad"] = CloneRuntime(
            self._transport_prev_grad)
        state["transport_curr_grad"] = CloneRuntime(
            self._transport_curr_grad)
        state["transport_delayed_ready"] = bool(
            self._transport_delayed_ready)
        state["transport_grad_accum_steps"] = int(
            self._transport_grad_accum_steps)
        state["transport_prev_grad_hook_seen"] = sorted(
            self._transport_prev_grad_hook_seen)
        state["transport_curr_grad_hook_seen"] = sorted(
            self._transport_curr_grad_hook_seen)
        state["return_value_prev"] = self.return_value_prev.detach().clone()
        state["return_value_valid"] = self.return_value_valid.detach().clone()

        state["td_out_ema_mean"] = self.td_out_ema.mean.detach().clone()
        state["td_out_ema_var"] = self.td_out_ema.var.detach().clone()

        ec = self.emotion_core
        state["emo_h"] = None if ec.h is None else ec.h.detach().clone()
        state["emo_c"] = None if ec.c is None else ec.c.detach().clone()
        state["emo_mood"] = None if ec.mood is None else ec.mood.detach().clone()
        state["emo_fast_H"] = ec.fast_head.H.detach().clone()
        state["emo_slow_H"] = ec.slow_head.H.detach().clone()

        uc = self.unc_core
        for name in ["td_ema", "ent_ema", "state_ema", "tr_ema", "ph_ema", "ctx_ema"]:
            ema = getattr(uc, name)
            state[f"unc_{name}_mean"] = ema.mean.detach().clone()
            state[f"unc_{name}_var"] = ema.var.detach().clone()

        for prefix, pred in [("reward_pred", self.reward_predictor),
                             ("done_pred", self.done_predictor)]:
            for n in ["kf_mean", "kf_var", "smooth_hist", "hist_len"]:
                t = getattr(pred, n)
                state[f"{prefix}_{n}"] = t.detach().clone()
            for n in ["predict_mode", "auto_policy", "auto_temperature", "fit_last_n"]:
                state[f"{prefix}_{n}"] = getattr(pred, n)

        mg = self.micro
        for n in ["anchor_value", "anchor_value_next", "anchor_z", "filled", "ptr"]:
            t = getattr(mg, n)
            state[f"micro_{n}"] = t.detach().clone()
        state["micro_step"] = int(mg._step)
        state["transport_manifold_tensor_field_ema"] = self.transport.manifold_tensor_field_ema.detach().clone()

        if tuple(state) != self.RUNTIME_STATE_FIELDS:
            raise RuntimeError("value runtime-state export does not match its schema")
        return state

    @torch.no_grad()
    def ImportState(self, state: Dict[str, Any],):
        if type(state) is not dict or tuple(state) != self.RUNTIME_STATE_FIELDS:
            raise TypeError("value runtime-state fields do not match the current schema")

        self.train(bool(state["ve_is_training"]))

        def CopyTensorAttr(obj: Any, name: str, t: torch.Tensor):
            if not torch.is_tensor(t):
                raise TypeError(f"value runtime-state field {name!r} must be a tensor")
            t = MoveRuntime(t)
            cur = getattr(obj, name)
            if not torch.is_tensor(cur):
                setattr(obj, name, t)
                return
            if cur.shape != t.shape:
                cur.resize_(t.shape).copy_(t)
            else:
                cur.copy_(t)

        move_memo: Dict[int, Any] = {}

        def MoveRuntime(value: Any) -> Any:
            value_id = id(value)
            if value_id in move_memo:
                return move_memo[value_id]
            if torch.is_tensor(value):
                if value.dtype.is_floating_point:
                    moved = value.to(device=self.device, dtype=self.dtype)
                else:
                    moved = value.to(device=self.device)
                move_memo[value_id] = moved
                return moved
            if isinstance(value, dict):
                moved: Dict[Any, Any] = {}
                move_memo[value_id] = moved
                moved.update({
                    key: MoveRuntime(item)
                    for key, item in value.items()})
                return moved
            if isinstance(value, list):
                moved_list: List[Any] = []
                move_memo[value_id] = moved_list
                moved_list.extend(MoveRuntime(item) for item in value)
                return moved_list
            return value

        self._last_batch_size = state["ve_last_batch_size"]
        self._active_stream_ids = None
        self._pending_transitions.clear()
        self.ClearTransportGradAccumulator()
        pending = state["pending_transitions"]
        self._pending_transitions = {
            int(stream_id): deque(
                MoveRuntime(item)
                for item in items)
            for stream_id, items in pending.items()}
        active_stream_ids = state["active_stream_ids"]
        self._active_stream_ids = (
            None
            if active_stream_ids is None
            else [int(item) for item in active_stream_ids])
        self._transport_prev_grad = MoveRuntime(
            state["transport_prev_grad"])
        self._transport_curr_grad = MoveRuntime(
            state["transport_curr_grad"])
        self._transport_delayed_ready = bool(
            state["transport_delayed_ready"])
        self._transport_grad_accum_steps = int(
            state["transport_grad_accum_steps"])
        self._transport_prev_grad_hook_seen = set(
            state["transport_prev_grad_hook_seen"])
        self._transport_curr_grad_hook_seen = set(
            state["transport_curr_grad_hook_seen"])
        CopyTensorAttr(self, "return_value_prev", state["return_value_prev"])
        CopyTensorAttr(self, "return_value_valid", state["return_value_valid"])

        CopyTensorAttr(self.td_out_ema, "mean", state["td_out_ema_mean"])
        CopyTensorAttr(self.td_out_ema, "var", state["td_out_ema_var"])

        ec = self.emotion_core
        ec.h = MoveRuntime(state["emo_h"])
        ec.c = MoveRuntime(state["emo_c"])
        ec.mood = MoveRuntime(state["emo_mood"])
        CopyTensorAttr(ec.fast_head, "H", state["emo_fast_H"])
        CopyTensorAttr(ec.slow_head, "H", state["emo_slow_H"])

        uc = self.unc_core
        for name in ["td_ema", "ent_ema", "state_ema", "tr_ema", "ph_ema", "ctx_ema"]:
            ema = getattr(uc, name)
            CopyTensorAttr(ema, "mean", state[f"unc_{name}_mean"])
            CopyTensorAttr(ema, "var", state[f"unc_{name}_var"])

        for prefix, pred in [("reward_pred", self.reward_predictor),
                             ("done_pred", self.done_predictor)]:
            for n in ["kf_mean", "kf_var", "smooth_hist", "hist_len"]:
                CopyTensorAttr(pred, n, state[f"{prefix}_{n}"])
            for n in ["predict_mode", "auto_policy", "auto_temperature", "fit_last_n"]:
                setattr(pred, n, state[f"{prefix}_{n}"])

        mg = self.micro
        for n in ["anchor_value", "anchor_value_next", "anchor_z", "filled", "ptr"]:
            CopyTensorAttr(mg, n, state[f"micro_{n}"])
        mg._step = int(state["micro_step"])
        CopyTensorAttr(
            self.transport,
            "manifold_tensor_field_ema",
            state["transport_manifold_tensor_field_ema"])


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
        maxRankQuantile: int = 64,
        ):
        self.maxRankFc1 = int(maxRankFc1)
        self.maxRankFc2 = int(maxRankFc2)
        self.maxRankQuantile = int(maxRankQuantile)
        super().__init__(base, initRankEach=initRankEach, autoRank=autoRank, evThreshold=evThreshold, gradEma=gradEma)
        self.RestoreBaseTrainabilityAfterCommit()

    def DirectOnlineHeads(self) -> Tuple[nn.Module, ...]:
        return (
            self.base.transport,
            self.base.return_value_head,
            self.base.cognitive_value_head,
            self.base.prospective_context_head,
            self.base.cognitive_value_norm,
            self.base.cognitive_metric_head,
            self.base.cognitive_proxy_head,
            self.base.cognitive_event_reason_head,)

    def RestoreBaseTrainabilityAfterCommit(self) -> None:
        super().RestoreBaseTrainabilityAfterCommit()
        for head in self.DirectOnlineHeads():
            for parameter in head.parameters():
                parameter.requires_grad_(True)

    @staticmethod
    def EnsureInputs(x):
        if isinstance(x, (tuple, list)) and len(x) == 3:
            return x[0], x[1], x[2]
        if isinstance(x, dict) and all(k in x for k in ("memoryPrev", "attnPrev", "state")):
            return x["memoryPrev"], x["attnPrev"], x["state"]
        raise TypeError("ValueEstimationOnlineWrapper expects x as (memoryPrev, attnPrev, state) or dict with those keys.")

    def AfterOptimizerStep(self):
        return self.base.AfterOptimizerStep()

    def CaptureTransportGrad(
        self,
        clearParamGrad: bool = True) -> Dict[str, float]:
        return self.base.CaptureTransportGrad(clearParamGrad=clearParamGrad)

    @torch.no_grad()
    def ApplyTransportManualGrad(
        self,
        lr: float,
        maxNorm: Optional[float] = None,
        weightDecay: float = 0.0,
        clear: bool = True,) -> Dict[str, float]:
        return self.base.ApplyTransportManualGrad(lr=lr, maxNorm=maxNorm, weightDecay=weightDecay, clear=clear)

    @torch.no_grad()
    def ClearTransportGradAccumulator(self):
        self.base.ClearTransportGradAccumulator()

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        base = self.base
        H = int(base.fc2.out_features)
        Din = int(base.fc1.in_features)
        L = 1

        def alloc(addRank: int, inDim: int, outDim: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, inDim, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype) * 1e-4)
            s = nn.Parameter(torch.as_tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            s_eff = torch.tanh(s) * GetParametersScale(s)
            return s_eff * (b @ a)

        return {
            "fc1": SiteSpec("fc1", L, Din, H, self.maxRankFc1, lambda r, dv, dt: alloc(r, Din, H, dv, dt), compose),
            "fc2": SiteSpec("fc2", L, H, H, self.maxRankFc2, lambda r, dv, dt: alloc(r, H, H, dv, dt), compose),
            "qhead": SiteSpec("qhead", L, H, int(base.quantile_head.out_features), self.maxRankQuantile, lambda r, dv, dt: alloc(r, H, int(base.quantile_head.out_features), dv, dt), compose),}

    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        **kwargs,):
        base: ValueEstimationExtractor = self.base
        memoryPrev, attnPrev, state = self.EnsureInputs(x)

        deltas = deltasPerLayer[0]
        parameter_overrides: Dict[str, torch.Tensor] = {}
        for name, layer, delta in [
            ("fc1_adapter.target.weight", base.fc1, deltas["fc1"]),
            ("fc2_adapter.target.weight", base.fc2, deltas["fc2"]),
            ("quantile_adapter.target.weight", base.quantile_head, deltas["qhead"])]:
            if delta is not None:
                parameter_overrides[name] = layer.weight + delta

        forward_kwargs = {
            "rewardModel": kwargs["rewardModel"],
            "doneModel": kwargs["doneModel"],
            "streamIds": kwargs.get("streamIds", None),
            "computeLoss": kwargs.get("computeLoss", None),
            "commitMask": kwargs.get("commitMask", None),
            "policyEntropyPrev": kwargs.get("policyEntropyPrev", None),
            "worldDeltaTransport": kwargs.get("worldDeltaTransport", None),
            "worldDeltaPhysics": kwargs.get("worldDeltaPhysics", None),}
        base_training = bool(base.training)
        base.training = bool(self.training)
        try:
            return torch_functional_call(
                base,
                parameter_overrides,
                (memoryPrev, attnPrev, state),
                forward_kwargs,
                tie_weights=False)
        finally:
            base.training = base_training

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        mapping = {
            "fc1": "fc1_adapter",
            "fc2": "fc2_adapter",
            "qhead": "quantile_adapter",}

        adapter: GrowableLoRALinear = getattr(self.base, mapping[site])
        init = {"A": a.detach(), "B": b.detach(), "scale": float(scale)}
        adapter.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
        return True





class TestValueEstimationMTool:
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mem_dim = 768
        self.attn_dim = 1024
        self.state_dim = 256
        self.hidden = 512
        self.last_loss_report: Optional[Dict[str, float]] = None

    def NewEstimator(self):
        return ValueEstimationExtractor(
            memoryDim=self.mem_dim,
            attnDim=self.attn_dim,
            stateDim=self.state_dim,
            hidden=self.hidden).to(self.device)

    def NewSmallEstimator(self):
        return ValueEstimationExtractor(
            memoryDim=4,
            attnDim=4,
            stateDim=4,
            hidden=16,
            valueTensorDim=16,
            tdHeatRank=2,
            tdBuresSlots=4,
            tdOtBins=4,
            tdOtCostDim=2,
            tdOtIters=1,
            microMaxAnchors=4,
            microTopK=2).to(self.device)

    def RandSmallBatch(self, B: int):
        return (
            torch.randn(B, 4, device=self.device),
            torch.randn(B, 4, device=self.device),
            torch.randn(B, 4, device=self.device))

    def RandSmallSignals(self, B: int):
        return (
            torch.randn(B, device=self.device).clamp(-1.0, 1.0),
            torch.rand(B, device=self.device),
            torch.zeros(B, device=self.device),
            torch.randn(B, 4, device=self.device) * 0.2,
            torch.randn(B, 4, device=self.device) * 0.2)

    def RandBatch(self, B: int):
        mem = torch.randn(B, self.mem_dim, device=self.device)
        attn = torch.randn(B, self.attn_dim, device=self.device)
        state = torch.randn(B, self.state_dim, device=self.device)
        return mem, attn, state

    def RandSignals(self, B: int, doneProb: float = 0.0):
        reward = torch.randn(B, device=self.device).clamp(-1.0, 1.0)
        entropy = torch.rand(B, device=self.device)
        if doneProb > 0.0:
            done = (torch.rand(B, device=self.device) < float(doneProb)).float()
        else:
            done = torch.zeros(B, device=self.device)
        d_tr = torch.randn(B, self.state_dim, device=self.device) * 0.2
        d_ph = torch.randn(B, self.state_dim, device=self.device) * 0.2
        return reward, entropy, done, d_tr, d_ph

    def OntologyPhysicalState(self, B: int, K: int = 4) -> Dict[str, torch.Tensor]:
        realm = torch.zeros(B, K, 5, device=self.device)
        realm[..., ONTOLOGY_REALM_EXTERNAL] = 1.0
        motion = torch.zeros(B, K, 5, device=self.device)
        motion[..., ONTOLOGY_MOTION_CARRIER] = 1.0
        agency = torch.zeros(
            B,
            K,
            5,
            5,
            device=self.device)
        agency[..., 4] = 1.0
        agency[..., ONTOLOGY_MOTION_CARRIER, :] = 0.0
        agency[..., ONTOLOGY_MOTION_CARRIER, ONTOLOGY_AGENCY_EXTERNAL] = 1.0
        ones = torch.ones(B, K, device=self.device)
        zeros = torch.zeros(B, K, device=self.device)
        return {
            "RealmProb": realm,
            "MotionLayerProb": motion,
            "LayerAgencyProb": agency,
            "PerceptualPresence": ones.clone(),
            "PhysicalEntityProb": ones.clone(),
            "PhysicalInteractionProb": ones.clone(),
            "BodyMembershipProb": zeros.clone(),
            "VerificationConfidence": ones.clone(),
            "ContactProbRaw": zeros.clone(),
            "ContentChangeProb": zeros.clone(),}

    def TestEntityOntologyRiskSeparation(self) -> bool:
        try:
            torch.manual_seed(113)
            B, K = 2, 4
            est = self.NewEstimator().eval()
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(
                B, doneProb=0.0)
            out = self.ForwardOnce(
                est,
                mem,
                attn,
                state,
                reward,
                entropy,
                done,
                d_tr,
                d_ph)
            physical_state = self.OntologyPhysicalState(B, K)
            physical_state["RealmProb"].zero_()
            physical_state["RealmProb"][
                ..., ONTOLOGY_REALM_VIRTUAL] = 1.0
            physical_state["MotionLayerProb"].zero_()
            physical_state["MotionLayerProb"][
                ..., ONTOLOGY_MOTION_SURFACE] = 1.0
            physical_state["LayerAgencyProb"].zero_()
            physical_state["LayerAgencyProb"][
                ..., 4] = 1.0
            physical_state["PhysicalEntityProb"].zero_()
            physical_state["PhysicalInteractionProb"].zero_()
            physical_state["ContentChangeProb"].fill_(1.0)

            refined = est.RefineEntityOntologyRisk(out, physical_state)
            ok = bool(
                torch.equal(
                    refined.rComps["risk"],
                    out.rComps["risk"].detach())
                and int(torch.count_nonzero(
                    refined.rComps["physicalRiskResidual"]).item()) == 0
                and bool((
                    refined.rComps["virtualAnimationSalience"] > 0.0
                ).all().item()))
            print(
                "EntityOntologyRiskSeparation "
                f"{'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"EntityOntologyRiskSeparation error: {e}")
            return False

    def ParamIds(self, module: nn.Module):
        return {id(p) for p in module.parameters() if p.requires_grad}

    def MonitorKeys(self):
        return {
            "risk",
            "ambiguity",
            "surprise",
            "confidence",}

    def TrainingMonitorKeys(self):
        return self.MonitorKeys()

    def ForwardOnce(
        self,
        est: ValueEstimationExtractor,
        mem: torch.Tensor,
        attn: torch.Tensor,
        state: torch.Tensor,
        reward: torch.Tensor,
        entropy: torch.Tensor,
        done: torch.Tensor,
        d_tr: torch.Tensor,
        d_ph: torch.Tensor) -> GeoTropicalOut:
        return est(
            memoryPrev=mem,
            attnPrev=attn,
            state=state,
            rewardModel=reward,
            policyEntropyPrev=entropy,
            doneModel=done,
            worldDeltaTransport=d_tr,
            worldDeltaPhysics=d_ph,)

    def RebuildSharedPendingPrediction(
        self,
        est: ValueEstimationExtractor,
        items: List[Dict[str, Any]],) -> torch.Tensor:
        transport_state = items[0]["transport_state"]
        if any(item["transport_state"] is not transport_state for item in items):
            raise RuntimeError("synchronous batch does not share one transport snapshot")
        with torch.no_grad():
            transport_value = torch.cat([
                item["transport_value"]
                for item in items], dim=0)
            prediction, extras = est.BuildTransportSnapshotGraph(
                torch.cat([
                    item["transport_hidden"]
                    for item in items], dim=0),
                transport_value,
                {},
                set(),
                transportState=transport_state)
            prediction, extras = est.ApplyRewardNextModulation(
                transport_value,
                prediction,
                extras,
                torch.cat([
                    item["reward_next"]
                    for item in items], dim=0))
            prediction, _ = est.ApplyMicroGraphPrior(
                prediction,
                extras,
                est.StackLiveRows([
                    item["micro_graph"]
                    for item in items]))
        return prediction

    def CloneTransportParams(self, est: ValueEstimationExtractor) -> Dict[str, torch.Tensor]:
        return {name: p.detach().clone() for name, p in est.transport.named_parameters() if p.requires_grad}

    def MaxTransportParamDelta(self, before: Dict[str, torch.Tensor], est: ValueEstimationExtractor) -> float:
        max_delta = 0.0
        for name, p in est.transport.named_parameters():
            if name in before:
                max_delta = max(max_delta, float((p.detach() - before[name]).abs().max().item()))
        return max_delta

    def TransportGradBucketNorm(self, bucket: Dict[str, torch.Tensor]) -> float:
        total = 0.0
        for grad in bucket.values():
            total += float(grad.detach().pow(2).sum().item())
        return total ** 0.5

    def PendingTransitionCount(self, est: ValueEstimationExtractor) -> int:
        return sum(len(q) for q in est._pending_transitions.values())

    def TestExtractorFunctional(self) -> bool:
        try:
            torch.manual_seed(42)
            B = 6
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.2)

            est = self.NewEstimator()
            est.eval()
            out_eval = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            ok = True
            ok &= (out_eval.value.shape == (B, est.value_tensor_dim))
            ok &= (out_eval.tdError.shape == (B,))
            ok &= (out_eval.uncertainty.shape == (B,))
            ok &= (out_eval.precision.shape == (B,))
            ok &= (out_eval.emotion.shape[0] == B)
            ok &= (out_eval.loss is None and out_eval.extras is None and isinstance(out_eval.rComps, dict))
            ok &= torch.isfinite(out_eval.value).all().item()
            ok &= torch.isfinite(out_eval.tdError).all().item()
            ok &= torch.isfinite(out_eval.uncertainty).all().item()
            ok &= torch.isfinite(out_eval.precision).all().item()

            est.train()
            out_t1 = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out_t2 = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            ok &= torch.is_tensor(out_t1.loss) and out_t1.loss.dim() == 0 and torch.isfinite(out_t1.loss).item()
            ok &= torch.is_tensor(out_t2.loss) and out_t2.loss.dim() == 0 and torch.isfinite(out_t2.loss).item()
            ok &= isinstance(out_t2.extras, dict) and isinstance(out_t2.rComps, dict)
            ok &= "loss_transport" in out_t2.extras
            ok &= "loss_current_graph" in out_t2.extras
            ok &= "loss_transport_delayed_graph" in out_t2.extras
            for name in self.TrainingMonitorKeys():
                ok &= name in out_t2.rComps
            ok &= "value_quantiles" not in out_t2.rComps
            ok &= "transport_counterfactual_values" not in out_t2.rComps

            print(f"ExtractorFunctional {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ExtractorFunctional error: {e}")
            return False

    def TestValueEstimatorIOShapes(self) -> bool:
        try:
            torch.manual_seed(142)
            B = 2
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            est = self.NewEstimator().train()

            def PrintShape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            PrintShape("input.memoryPrev", mem)
            PrintShape("input.attnPrev", attn)
            PrintShape("input.state", state)
            PrintShape("input.rewardModel", reward)
            PrintShape("input.policyEntropyPrev", entropy)
            PrintShape("input.done", done)
            PrintShape("input.worldDeltaTransport", d_tr)
            PrintShape("input.worldDeltaPhysics", d_ph)

            out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            out_dict = out._asdict()
            for key, value in out_dict.items():
                if isinstance(value, torch.Tensor):
                    PrintShape(f"output.{key}", value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            PrintShape(f"output.{key}.{sub_key}", sub_value)

            ok = True
            ok &= (out.value.shape == (B, est.value_tensor_dim))
            ok &= (out.valueNext.shape == (B, est.value_tensor_dim))
            ok &= (out.tdError.shape == (B,))
            ok &= (out.loss.shape == ())
            ok &= (out.emotion.shape[0] == B)
            ok &= (out.uncertainty.shape == (B,))
            ok &= (out.precision.shape == (B,))
            for name in self.MonitorKeys():
                ok &= (out.rComps is not None and out.rComps[name].shape == (B,))
            ok &= (out.extras is not None and torch.is_tensor(out.extras["loss_current_graph"]))

            print(f"ValueEstimator IO shapes {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ValueEstimator IO shapes error: {e}")
            return False

    def TestTDUncertaintyBounds(self) -> bool:
        try:
            torch.manual_seed(43)
            B = 9
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.3)

            est = self.NewEstimator().eval()
            out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            ok = True
            ok &= torch.isfinite(out.tdError).all().item()
            ok &= torch.isfinite(out.uncertainty).all().item()
            ok &= torch.isfinite(out.precision).all().item()
            ok &= (float(out.tdError.abs().max().item()) <= 1.0 + 1e-6)
            ok &= (float(out.uncertainty.min().item()) >= -1e-6)
            ok &= (float(out.uncertainty.max().item()) <= 1.0 + 1e-6)
            ok &= (float(out.precision.min().item()) >= 0.05 - 1e-6)
            ok &= (float(out.precision.max().item()) <= 1.0 + 1e-6)
            for name in self.MonitorKeys():
                comp = out.rComps[name]
                ok &= torch.isfinite(comp).all().item()
                ok &= (float(comp.min().item()) >= -1e-6)
                ok &= (float(comp.max().item()) <= 1.0 + 1e-6)

            print(f"TDUncertaintyBounds {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"TDUncertaintyBounds error: {e}")
            return False

    def TestPhysicalTDNoCrossBatch(self) -> bool:
        try:
            torch.manual_seed(431)
            B = 2
            est = self.NewEstimator().eval()
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            mem_b = mem.clone()
            attn_b = attn.clone()
            state_b = state.clone()
            reward_b = reward.clone()
            entropy_b = entropy.clone()
            done_b = done.clone()
            d_tr_b = d_tr.clone()
            d_ph_b = d_ph.clone()
            mem_b[1].add_(7.0)
            attn_b[1].mul_(-3.0)
            state_b[1].add_(5.0)
            reward_b[1].add_(4.0)
            entropy_b[1].fill_(0.95)
            d_tr_b[1].mul_(4.0)
            d_ph_b[1].mul_(-4.0)

            est.ResetState()
            est.ResetHebbianMemory()
            with torch.no_grad():
                out_a = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            est.ResetState()
            est.ResetHebbianMemory()
            with torch.no_grad():
                out_b = self.ForwardOnce(est, mem_b, attn_b, state_b, reward_b, entropy_b, done_b, d_tr_b, d_ph_b)

            ok = True
            ok &= torch.allclose(out_a.value[0], out_b.value[0], atol=1e-6, rtol=1e-5)
            ok &= torch.allclose(out_a.valueNext[0], out_b.valueNext[0], atol=1e-6, rtol=1e-5)
            ok &= torch.allclose(out_a.tdError[0], out_b.tdError[0], atol=1e-6, rtol=1e-5)
            for key in self.MonitorKeys():
                ok &= torch.allclose(out_a.rComps[key][0], out_b.rComps[key][0], atol=1e-6, rtol=1e-5)

            print(f"PhysicalTDNoCrossBatch {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"PhysicalTDNoCrossBatch error: {e}")
            return False

    def TestModelTargetOnly(self) -> bool:
        try:
            torch.manual_seed(432)
            B = 5
            mem, attn, state = self.RandBatch(B)
            entropy = torch.rand(B, device=self.device)
            d_tr = torch.randn(B, self.state_dim, device=self.device) * 0.2
            d_ph = torch.randn(B, self.state_dim, device=self.device) * 0.2
            reward_model = torch.full((B,), -0.5, device=self.device)
            done_model = torch.full((B,), 0.25, device=self.device)

            est = self.NewEstimator().eval()
            out_model = est(
                memoryPrev=mem,
                attnPrev=attn,
                state=state,
                rewardModel=reward_model,
                policyEntropyPrev=entropy,
                doneModel=done_model,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph)

            ok = True
            ok &= torch.isfinite(out_model.value).all().item()
            ok &= self.MonitorKeys().issubset(set(out_model.rComps.keys()))
            print(f"ModelTargetOnly {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ModelTargetOnly error: {e}")
            return False

    def TestDifferentialTDSemantics(self) -> bool:
        try:
            torch.manual_seed(434)
            B = 4
            est = self.NewEstimator().train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)
            done2 = torch.tensor([0.0, 1.0, 0.0, 1.0], device=self.device)

            out1 = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            pending_after_t1 = sum(len(q) for q in est._pending_transitions.values()) == B
            out2 = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            ok = True
            ok &= pending_after_t1
            ok &= torch.isfinite(out1.tdError).all().item()
            ok &= torch.isfinite(out2.tdError).all().item()
            ok &= (float(out2.tdError.abs().max().item()) <= 1.0 + 1e-6)
            ok &= torch.isfinite(out2.extras["loss_diff"]).item()

            print(f"DifferentialTDSemantics {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"DifferentialTDSemantics error: {e}")
            return False

    def TestCurrentValueUsedAsDelayedLabel(self) -> bool:
        try:
            torch.manual_seed(439)
            B = 5
            est = self.NewEstimator().train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            pending_items = [est._pending_transitions[i][0] for i in range(B)]
            prev_pred = self.RebuildSharedPendingPrediction(est, pending_items)
            out2 = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            ok = True
            expected_loss = F.smooth_l1_loss(prev_pred, out2.value.detach(), reduction="none").view(B, -1).mean(dim=-1).mean()
            ok &= torch.allclose(out2.extras["loss_diff"], expected_loss.detach(), atol=1e-6, rtol=1e-5)
            ok &= torch.isfinite(out2.extras["loss_diff"]).item()
            ok &= torch.isfinite(out2.extras["loss_manifold_geo"]).item()
            ok &= torch.isfinite(out2.extras["loss_manifold_tangent"]).item()

            print(f"CurrentValueUsedAsDelayedLabel {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"CurrentValueUsedAsDelayedLabel error: {e}")
            return False

    def TestPendingQueueStableStreamPairing(self) -> bool:
        try:
            torch.manual_seed(440)
            B = 3
            est = self.NewEstimator().train()
            sids_a = torch.tensor([10, 20, 30], device=self.device)
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            _ = est(mem, attn, state, rewardModel=reward, policyEntropyPrev=entropy, doneModel=done,
                    worldDeltaTransport=d_tr, worldDeltaPhysics=d_ph, streamIds=sids_a)
            pending_after_one = sum(len(q) for q in est._pending_transitions.values())
            out = est(mem, attn, state, rewardModel=reward, policyEntropyPrev=entropy, doneModel=done,
                      worldDeltaTransport=d_tr, worldDeltaPhysics=d_ph, streamIds=sids_a)
            pending_after_consume = sum(len(q) for q in est._pending_transitions.values())

            ok = True
            ok &= pending_after_one == B
            ok &= pending_after_consume == B
            ok &= torch.isfinite(out.extras["loss_diff"]).item()

            print(f"PendingQueueStableStreamPairing {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"PendingQueueStableStreamPairing error: {e}")
            return False

    def TestTemporalPairingOfUncertainty(self) -> bool:
        try:
            torch.manual_seed(441)
            B = 4
            est = self.NewEstimator().train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            pending_items = [est._pending_transitions[i][0] for i in range(B)]
            out2 = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            ok = True
            ok &= all("unc" not in it for it in pending_items)
            expected_keys = {
                "transport_hidden",
                "transport_value",
                "reward_next",
                "transport_state",
                "micro_graph",
                "return_hidden",
                "alive",}
            ok &= all(set(it.keys()) == expected_keys for it in pending_items)
            ok &= len({id(it["transport_state"]) for it in pending_items}) == 1
            cached_tensors = []
            for item in pending_items:
                for key in expected_keys:
                    value = item[key]
                    cached_tensors.extend(
                        value.values()
                        if isinstance(value, dict)
                        else (value,))
            ok &= all(
                tensor.grad_fn is None and not tensor.requires_grad
                for tensor in cached_tensors)
            ok &= sum(len(q) for q in est._pending_transitions.values()) == B
            ok &= torch.isfinite(out2.extras["loss_current_graph"]).item()
            ok &= torch.isfinite(out2.extras["loss_transport_delayed_graph"]).item()

            print(f"TemporalPairingOfUncertainty {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"TemporalPairingOfUncertainty error: {e}")
            return False

    def TestTransportManifoldFieldGrad(self) -> bool:
        try:
            torch.manual_seed(442)
            B = 5
            est = self.NewEstimator().train()
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
            _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            est.zero_grad(set_to_none=True)
            out.extras["loss_transport_delayed_graph"].backward(retain_graph=True)
            est.CaptureTransportGrad(clearParamGrad=True)
            groups = {
                "manifold_encoder": [],
                "manifold_drift_head": [],
                "manifold_connection_gate": [],
                "manifold_metric_head": [],
                "manifold_value_correction": [],
                "manifold_value_readout": [],}
            for n, p in est.transport.named_parameters():
                g_src = p.grad
                if (g_src is None or float(g_src.detach().abs().max().item()) <= 0.0) and n in est._transport_prev_grad:
                    g_src = est._transport_prev_grad[n]
                if g_src is None:
                    continue
                g = g_src.detach().abs().max().item()
                for prefix in groups:
                    if n.startswith(prefix):
                        groups[prefix].append(g)
            ok = all(bool(vals) and max(vals) > 0.0 for vals in groups.values())
            all_manifold_grads = [g for vals in groups.values() for g in vals]
            ok &= bool(all_manifold_grads) and max(all_manifold_grads) > 0.0
            ok &= torch.isfinite(out.extras["loss_manifold_geo"]).item()
            ok &= torch.isfinite(out.extras["loss_manifold_tangent"]).item()
            ok &= torch.isfinite(out.extras["loss_manifold_latent"]).item()
            print(f"TransportManifoldFieldGrad {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"TransportManifoldFieldGrad error: {e}")
            return False

    def TestManualTransportGradWorkflow(self) -> bool:
        try:
            torch.manual_seed(443)
            B = 4
            est = self.NewEstimator().train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            out1 = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            est.zero_grad(set_to_none=True)
            out1.valueNext.pow(2).mean().backward(retain_graph=True)
            capture_current_0 = est.CaptureTransportGrad(clearParamGrad=True)
            apply_wait = est.ApplyTransportManualGrad(lr=1e-3, maxNorm=1.0)

            out = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)
            before = {n: p.detach().clone() for n, p in est.transport.named_parameters() if p.requires_grad}
            est.zero_grad(set_to_none=True)
            out.extras["loss_transport_delayed_graph"].backward(retain_graph=True)
            capture_delayed = est.CaptureTransportGrad(clearParamGrad=True)
            transport_grads_cleared = all(p.grad is None for p in est.transport.parameters())

            est.zero_grad(set_to_none=True)
            out.valueNext.pow(2).mean().backward(retain_graph=True)
            capture_current_1 = est.CaptureTransportGrad(clearParamGrad=True)
            apply = est.ApplyTransportManualGrad(lr=1e-3, maxNorm=1.0)

            changed = []
            for n, p in est.transport.named_parameters():
                if n in before:
                    changed.append(float((p.detach() - before[n]).abs().max().item()) > 0.0)

            ok = True
            ok &= apply_wait["updated"] == 0.0
            ok &= capture_delayed["captured"] > 0.0
            ok &= capture_delayed["grad_norm"] > 0.0
            ok &= capture_current_0["captured"] >= 0.0
            ok &= capture_current_1["captured"] >= 0.0
            ok &= transport_grads_cleared
            ok &= apply["updated"] > 0.0
            ok &= apply["grad_norm"] > 0.0
            ok &= any(changed)
            ok &= len(est._transport_curr_grad) == 0

            print(f"ManualTransportGradWorkflow {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ManualTransportGradWorkflow error: {e}")
            return False

    def TestTransportDelayedGradientPipeline(self) -> bool:
        try:
            torch.manual_seed(20260514)
            B = 3
            lr = 1e-3
            est = self.NewEstimator().train()

            mem0, attn0, state0 = self.RandBatch(B)
            reward0, entropy0, done0, d_tr0, d_ph0 = self.RandSignals(B, doneProb=0.0)
            mem1, attn1, state1 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            mem2, attn2, state2 = self.RandBatch(B)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            out0 = self.ForwardOnce(est, mem0, attn0, state0, reward0, entropy0, done0, d_tr0, d_ph0)
            p_t0_forward = self.CloneTransportParams(est)
            pending_after_t0 = self.PendingTransitionCount(est)

            est.zero_grad(set_to_none=True)
            out0.valueNext.pow(2).mean().backward(retain_graph=True)
            capture_current0 = est.CaptureTransportGrad(clearParamGrad=True)
            curr0_norm = self.TransportGradBucketNorm(est._transport_curr_grad)
            delta_after_current0_backward = self.MaxTransportParamDelta(p_t0_forward, est)
            apply0 = est.ApplyTransportManualGrad(lr=lr, maxNorm=1.0)
            delta_after_apply0 = self.MaxTransportParamDelta(p_t0_forward, est)
            prev_after_t0_norm = self.TransportGradBucketNorm(est._transport_prev_grad)
            curr_after_t0_norm = self.TransportGradBucketNorm(est._transport_curr_grad)

            out1 = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            p_before_delayed_update1 = self.CloneTransportParams(est)

            est.zero_grad(set_to_none=True)
            out1.extras["loss_transport_delayed_graph"].backward(retain_graph=True)
            capture_delayed1 = est.CaptureTransportGrad(clearParamGrad=True)
            prev_before_update1_norm = self.TransportGradBucketNorm(est._transport_prev_grad)
            curr_before_update1_norm = self.TransportGradBucketNorm(est._transport_curr_grad)

            est.zero_grad(set_to_none=True)
            out1.valueNext.pow(2).mean().backward(retain_graph=True)
            capture_current1 = est.CaptureTransportGrad(clearParamGrad=True)
            curr1_norm = self.TransportGradBucketNorm(est._transport_curr_grad)
            apply1_update = est.ApplyTransportManualGrad(lr=lr, maxNorm=1.0)
            delta_after_delayed_update1 = self.MaxTransportParamDelta(p_before_delayed_update1, est)
            prev_after_t1_norm = self.TransportGradBucketNorm(est._transport_prev_grad)
            curr_after_t1_norm = self.TransportGradBucketNorm(est._transport_curr_grad)

            out2 = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)
            pending_after_t2 = self.PendingTransitionCount(est)
            p_before_delayed_update2 = self.CloneTransportParams(est)

            est.zero_grad(set_to_none=True)
            out2.extras["loss_transport_delayed_graph"].backward(retain_graph=True)
            capture_delayed2 = est.CaptureTransportGrad(clearParamGrad=True)
            prev_before_update2_norm = self.TransportGradBucketNorm(est._transport_prev_grad)
            curr_before_update2_norm = self.TransportGradBucketNorm(est._transport_curr_grad)

            est.zero_grad(set_to_none=True)
            out2.valueNext.pow(2).mean().backward(retain_graph=True)
            capture_current2 = est.CaptureTransportGrad(clearParamGrad=True)
            curr2_norm = self.TransportGradBucketNorm(est._transport_curr_grad)
            apply2_update = est.ApplyTransportManualGrad(lr=lr, maxNorm=1.0)
            delta_after_delayed_update2 = self.MaxTransportParamDelta(p_before_delayed_update2, est)
            prev_after_t2_norm = self.TransportGradBucketNorm(est._transport_prev_grad)
            curr_after_t2_norm = self.TransportGradBucketNorm(est._transport_curr_grad)

            metrics = {
                "pending_after_t0": float(pending_after_t0),
                "pending_after_t2": float(pending_after_t2),
                "t0_current_captured": float(capture_current0["captured"]),
                "t0_current_grad_norm": float(curr0_norm),
                "t0_param_delta_after_current_backward": float(delta_after_current0_backward),
                "t0_apply_updated": float(apply0["updated"]),
                "t0_param_delta_after_apply": float(delta_after_apply0),
                "t0_prev_grad_norm_after_rotate": float(prev_after_t0_norm),
                "t0_curr_grad_norm_after_rotate": float(curr_after_t0_norm),
                "t1_delayed_captured": float(capture_delayed1["captured"]),
                "t1_prev_grad_norm_before_update": float(prev_before_update1_norm),
                "t1_curr_grad_norm_before_update": float(curr_before_update1_norm),
                "t1_delayed_apply_updated": float(apply1_update["updated"]),
                "t1_param_delta_after_delayed_update": float(delta_after_delayed_update1),
                "t1_current_captured": float(capture_current1["captured"]),
                "t1_current_grad_norm": float(curr1_norm),
                "t1_prev_grad_norm_after_rotate": float(prev_after_t1_norm),
                "t1_curr_grad_norm_after_rotate": float(curr_after_t1_norm),
                "t2_delayed_captured": float(capture_delayed2["captured"]),
                "t2_prev_grad_norm_before_update": float(prev_before_update2_norm),
                "t2_curr_grad_norm_before_update": float(curr_before_update2_norm),
                "t2_delayed_apply_updated": float(apply2_update["updated"]),
                "t2_param_delta_after_delayed_update": float(delta_after_delayed_update2),
                "t2_current_captured": float(capture_current2["captured"]),
                "t2_current_grad_norm": float(curr2_norm),
                "t2_prev_grad_norm_after_rotate": float(prev_after_t2_norm),
                "t2_curr_grad_norm_after_rotate": float(curr_after_t2_norm),
                }

            ok = True
            ok &= metrics["pending_after_t0"] == float(B)
            ok &= metrics["t0_current_grad_norm"] > 0.0
            ok &= metrics["t0_apply_updated"] == 0.0
            ok &= metrics["t0_param_delta_after_apply"] == 0.0
            ok &= metrics["t0_prev_grad_norm_after_rotate"] > 0.0
            ok &= metrics["t1_delayed_captured"] > 0.0
            ok &= metrics["t1_prev_grad_norm_before_update"] > metrics["t0_prev_grad_norm_after_rotate"]
            ok &= metrics["t1_curr_grad_norm_before_update"] == 0.0
            ok &= metrics["t1_delayed_apply_updated"] > 0.0
            ok &= metrics["t1_param_delta_after_delayed_update"] > 0.0
            ok &= metrics["t1_current_grad_norm"] > 0.0

            ok &= metrics["t1_prev_grad_norm_after_rotate"] > 0.0
            ok &= metrics["t1_curr_grad_norm_after_rotate"] == 0.0
            ok &= metrics["pending_after_t2"] == float(B)
            ok &= metrics["t2_delayed_captured"] > 0.0

            ok &= metrics["t2_prev_grad_norm_before_update"] > 0.0
            ok &= metrics["t2_curr_grad_norm_before_update"] == 0.0
            ok &= metrics["t2_delayed_apply_updated"] > 0.0
            ok &= metrics["t2_param_delta_after_delayed_update"] > 0.0
            ok &= metrics["t2_current_grad_norm"] > 0.0
            ok &= metrics["t2_prev_grad_norm_after_rotate"] > 0.0
            ok &= metrics["t2_curr_grad_norm_after_rotate"] == 0.0

            print("\n[Transport delayed-gradient pipeline trace]")
            for key in sorted(metrics):
                print(f"{key}: {metrics[key]}")
            print(f"TransportDelayedGradientPipeline {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"TransportDelayedGradientPipeline error: {e}")
            return False

    def TestHebbianSnapshotUsedForDelayedRebuild(self) -> bool:
        try:
            torch.manual_seed(444)
            B = 4
            est = self.NewEstimator().train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            items = [est._pending_transitions[i][0] for i in range(B)]
            out2 = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            ok = True
            ok &= all("hebb_state" not in it for it in items)
            ok &= all("memory" not in it and "attn" not in it and "state" not in it for it in items)
            ok &= len({id(it["transport_state"]) for it in items}) == 1
            ok &= all(torch.isfinite(it["transport_hidden"]).all().item() for it in items)
            ok &= all(not it["transport_hidden"].requires_grad for it in items)
            ok &= torch.isfinite(out2.extras["loss_transport_delayed_graph"]).item()

            print(f"HebbianSnapshotUsedForDelayedRebuild {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"HebbianSnapshotUsedForDelayedRebuild error: {e}")
            return False

    def TestTerminalMaskDelayedLosses(self) -> bool:
        try:
            torch.manual_seed(438)
            B = 4
            est_a = self.NewEstimator().train()
            est_b = copy.deepcopy(est_a).train()

            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1 = torch.zeros(B, device=self.device)
            entropy1 = torch.rand(B, device=self.device)
            done1 = torch.tensor([1.0, 0.0, 1.0, 0.0], device=self.device)
            d_tr1 = torch.randn(B, self.state_dim, device=self.device) * 0.2
            d_ph1 = torch.randn(B, self.state_dim, device=self.device) * 0.2
            reward2_a = torch.zeros(B, device=self.device)
            reward2_b = reward2_a.clone()
            reward2_b[done1 > 0.5] = 9.0
            entropy2 = torch.rand(B, device=self.device)
            done2 = torch.zeros(B, device=self.device)
            d_tr2 = torch.randn(B, self.state_dim, device=self.device) * 0.2
            d_ph2 = torch.randn(B, self.state_dim, device=self.device) * 0.2

            _ = self.ForwardOnce(est_a, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            _ = self.ForwardOnce(est_b, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            prev_valid = torch.stack([
                est_a._pending_transitions[i][0]["alive"]
                for i in range(B)], dim=0).detach().clone()
            out_a = self.ForwardOnce(est_a, mem2, attn2, state2, reward2_a, entropy2, done2, d_tr2, d_ph2)
            out_b = self.ForwardOnce(est_b, mem2, attn2, state2, reward2_b, entropy2, done2, d_tr2, d_ph2)

            masked_rows = done1 > 0.5
            keys = [
                "loss_diff",
                "loss_diff_branch",
                "loss_transport",
                "loss_manifold_geo",
                "loss_manifold_tangent",
                "loss_manifold_latent"]
            ok = True
            ok &= torch.allclose(prev_valid[masked_rows], torch.zeros_like(prev_valid[masked_rows]), atol=1e-6, rtol=1e-5)
            for key in keys:
                ok &= torch.allclose(out_a.extras[key], out_b.extras[key], atol=2e-5, rtol=1e-5)

            print(f"TerminalMaskDelayedLosses {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"TerminalMaskDelayedLosses error: {e}")
            return False

    def TestAllTerminalDelayedLossFullyMasked(self) -> bool:
        try:
            torch.manual_seed(20260713)
            B = 4
            est = self.NewSmallEstimator().train()
            mem1, attn1, state1 = self.RandSmallBatch(B)
            mem2, attn2, state2 = self.RandSmallBatch(B)
            reward1, entropy1, _, d_tr1, d_ph1 = self.RandSmallSignals(B)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSmallSignals(B)
            done1 = torch.ones(B, device=self.device)

            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            out = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            keys = [
                "loss_diff",
                "loss_diff_branch",
                "loss_branch_structure",
                "loss_manifold_geo",
                "loss_manifold_tangent",
                "loss_manifold_latent",
                "loss_transport"]
            ok = all(
                torch.allclose(out.extras[key], out.extras[key].new_zeros(()), atol=1e-8, rtol=0.0)
                for key in keys)
            print(f"AllTerminalDelayedLossFullyMasked {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"AllTerminalDelayedLossFullyMasked error: {e}")
            return False

    def TestPerRowPredictorResetUsesOnlyNewEpisodeHistory(self) -> bool:
        try:
            pred = KalmanFilteredEnsembleNext(
                historyLen=16,
                fitLastN=16,
                predictMode="kalman").to(self.device)
            previous = torch.tensor([1.0, 2.0], device=self.device)
            for _ in range(6):
                _ = pred.PredictNext(previous)

            pred.Reset(doneMask=torch.tensor([True, False], device=self.device))
            predicted = pred.PredictNext(previous)

            ok = True
            ok &= torch.allclose(predicted[0], previous[0], atol=1e-6, rtol=1e-6)
            ok &= int(pred.hist_len[0].item()) == 1
            ok &= int(pred.hist_len[1].item()) == 7
            print(f"PerRowPredictorResetUsesOnlyNewEpisodeHistory {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"PerRowPredictorResetUsesOnlyNewEpisodeHistory error: {e}")
            return False

    def TestCurrentDoneMasksTemporalUncertaintyEvidence(self) -> bool:
        try:
            B = 3
            core = UncertaintyCore(
                stateDim=self.state_dim,
                memDim=self.mem_dim,
                attnDim=self.attn_dim).to(self.device)
            zeros_mem = torch.zeros(B, self.mem_dim, device=self.device)
            zeros_attn = torch.zeros(B, self.attn_dim, device=self.device)
            zeros_state = torch.zeros(B, self.state_dim, device=self.device)
            high = torch.full((B,), 10.0, device=self.device)
            done = torch.ones(B, device=self.device)
            unc, _ = core(
                memoryPrev=zeros_mem,
                attnPrev=zeros_attn,
                stateCurr=zeros_state,
                entropyPrev=high,
                tdCurr=high,
                worldDeltaTransport=zeros_state,
                worldDeltaPhysics=zeros_state,
                doneCurr=done)
            ok = torch.allclose(
                unc,
                torch.full_like(unc, float(core.eps_prior)),
                atol=1e-6,
                rtol=1e-4)
            print(f"CurrentDoneMasksTemporalUncertaintyEvidence {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"CurrentDoneMasksTemporalUncertaintyEvidence error: {e}")
            return False

    def TestManifoldFieldEmaIsRuntimeState(self) -> bool:
        try:
            torch.manual_seed(20260714)
            B = 4
            est = self.NewSmallEstimator().train()
            mem, attn, state = self.RandSmallBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSmallSignals(B)
            _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            parameter_names = {name for name, _ in est.transport.named_parameters()}
            after = est.transport.manifold_tensor_field_ema.detach()
            ok = True
            ok &= "manifold_tensor_field_ema" not in parameter_names
            ok &= after.shape == (B, est.value_tensor_dim)
            ok &= float(after.abs().max().item()) > 0.0
            print(f"ManifoldFieldEmaIsRuntimeState {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"ManifoldFieldEmaIsRuntimeState error: {e}")
            return False

    def TestTransportSnapshotUsesSameRuntimeState(self) -> bool:
        try:
            torch.manual_seed(20260717)
            B = 3
            est = self.NewSmallEstimator().train()
            est.micro_graph_mix = 0.0
            est.transport.manifold_field_ema = 0.0
            mem, attn, state = self.RandSmallBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSmallSignals(B)
            out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            pending_items = [est._pending_transitions[row][0] for row in range(B)]
            rebuilt = self.RebuildSharedPendingPrediction(est, pending_items)
            ok = torch.allclose(out.valueNext.detach(), rebuilt, atol=1e-7, rtol=1e-6)
            print(f"TransportSnapshotUsesSameRuntimeState {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"TransportSnapshotUsesSameRuntimeState error: {e}")
            return False

    def TestCurrentTransitionLossMasksTerminalGeometry(self) -> bool:
        try:
            torch.manual_seed(20260718)
            B = 2
            est_a = self.NewSmallEstimator().train()
            est_b = copy.deepcopy(est_a).train()
            mem, attn, state = self.RandSmallBatch(B)
            reward, entropy, _, d_tr, d_ph = self.RandSmallSignals(B)
            done = torch.tensor([0.0, 1.0], device=self.device)

            mem_b = mem.clone()
            attn_b = attn.clone()
            state_b = state.clone()
            reward_b = reward.clone()
            entropy_b = entropy.clone()
            d_tr_b = d_tr.clone()
            d_ph_b = d_ph.clone()
            mem_b[1].add_(100.0)
            attn_b[1].sub_(80.0)
            state_b[1].mul_(50.0)
            reward_b[1].add_(20.0)
            entropy_b[1] = 0.99
            d_tr_b[1].add_(30.0)
            d_ph_b[1].sub_(40.0)

            out_a = self.ForwardOnce(est_a, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out_b = self.ForwardOnce(est_b, mem_b, attn_b, state_b, reward_b, entropy_b, done, d_tr_b, d_ph_b)
            loss_without_anchor_a = (
                out_a.loss - out_a.extras["loss_value_return_anchor"])
            loss_without_anchor_b = (
                out_b.loss - out_b.extras["loss_value_return_anchor"])
            target_a = (
                (1.0 - done) * torch.tanh(out_a.returnValue.detach()))
            target_b = (
                (1.0 - done) * torch.tanh(out_b.returnValue.detach()))
            ok = bool(
                torch.allclose(
                    loss_without_anchor_a,
                    loss_without_anchor_b,
                    atol=1e-7,
                    rtol=1e-6)
                and float(target_a[1].item()) == 0.0
                and float(target_b[1].item()) == 0.0)
            print(f"CurrentTransitionLossMasksTerminalGeometry {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"CurrentTransitionLossMasksTerminalGeometry error: {e}")
            return False

    def TestHebbianLinearFWLifecycle(self) -> bool:
        try:
            torch.manual_seed(20260719)
            B = 3
            layer = HebbianLinearFW(4, 3).to(self.device)
            layer.EnsureB(B)
            value = torch.randn(B, 4, device=self.device)

            with torch.no_grad():
                base = F.linear(value, layer.weight, layer.bias)
                first, _ = layer(value)
                first_memory = layer.H.detach().clone()
                expected_second = (
                    base
                    + 0.1
                    * torch.bmm(
                        first_memory,
                        value.unsqueeze(-1)).squeeze(-1))
                second, _ = layer(value)

                layer.ResetHebbianMemory()
                reset_cleared = torch.count_nonzero(layer.H).item() == 0
                first_after_reset, _ = layer(value)

            ok = torch.allclose(first, base, atol=1e-7, rtol=1e-6)
            ok &= torch.count_nonzero(first_memory).item() > 0
            ok &= torch.allclose(
                second,
                expected_second,
                atol=1e-7,
                rtol=1e-6)
            ok &= not torch.allclose(second, base, atol=1e-9, rtol=1e-9)
            ok &= reset_cleared
            ok &= torch.allclose(
                first_after_reset,
                base,
                atol=1e-7,
                rtol=1e-6)
            ok &= "H" not in layer.state_dict()
            print(f"HebbianLinearFWLifecycle {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"HebbianLinearFWLifecycle error: {e}")
            return False

    def TestStreamIdEpisodeReplacement(self) -> bool:
        try:
            torch.manual_seed(20260715)
            B = 2
            est = self.NewSmallEstimator().train()
            mem, attn, state = self.RandSmallBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSmallSignals(B)
            stream_ids = torch.tensor([10, 20], device=self.device)
            _ = est(
                mem, attn, state,
                rewardModel=reward,
                policyEntropyPrev=entropy,
                doneModel=done,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph,
                streamIds=stream_ids)

            est.ResetState(doneMask=torch.tensor([True, False], device=self.device))
            replacement_ids = torch.tensor([11, 20], device=self.device)
            _ = est(
                mem, attn, state,
                rewardModel=reward,
                policyEntropyPrev=entropy,
                doneModel=done,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph,
                streamIds=replacement_ids)

            ok = 10 not in est._pending_transitions
            ok &= 11 in est._pending_transitions and 20 in est._pending_transitions
            print(f"StreamIdEpisodeReplacement {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"StreamIdEpisodeReplacement error: {e}")
            return False

    def TestGeometricDistances(self) -> bool:
        try:
            torch.manual_seed(20260716)
            est = self.NewSmallEstimator().eval()
            x = torch.randn(3, est.value_tensor_dim, device=self.device)
            translated = est.BuresWassersteinEnergy(x, x + 0.5)
            identical_ot = est.SinkhornOTEnergy(x, x)

            same_x = x.detach().clone().requires_grad_(True)
            same_y = x.detach().clone().requires_grad_(True)
            same_bures = est.BuresWassersteinEnergy(same_x, same_y)
            same_ot = est.SinkhornOTEnergy(same_x, same_y)
            same_grad_x, same_grad_y = torch.autograd.grad(
                (same_bures + same_ot).sum(),
                (same_x, same_y))

            zero_x = torch.zeros(2, est.value_tensor_dim, device=self.device, requires_grad=True)
            zero_y = torch.zeros(2, est.value_tensor_dim, device=self.device, requires_grad=True)
            zero_bures = est.BuresWassersteinEnergy(zero_x, zero_y)
            zero_bures.sum().backward()

            ok = bool((translated > 0.5).all().item())
            ok &= torch.allclose(identical_ot, torch.zeros_like(identical_ot), atol=1e-7, rtol=0.0)
            ok &= torch.allclose(same_bures, torch.zeros_like(same_bures), atol=1e-7, rtol=0.0)
            ok &= torch.allclose(same_ot, torch.zeros_like(same_ot), atol=1e-7, rtol=0.0)
            ok &= float(same_grad_x.abs().sum().item()) < 1e-4
            ok &= float(same_grad_y.abs().sum().item()) < 1e-4
            ok &= torch.isfinite(zero_x.grad).all().item()
            ok &= torch.isfinite(zero_y.grad).all().item()
            print(f"GeometricDistances {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"GeometricDistances error: {e}")
            return False

    def TestUncertaintyFloorNearEps(self) -> bool:
        try:
            torch.manual_seed(435)
            B = 3
            core = UncertaintyCore(
                stateDim=self.state_dim,
                memDim=self.mem_dim,
                attnDim=self.attn_dim).to(self.device)
            zeros_mem = torch.zeros(B, self.mem_dim, device=self.device)
            zeros_attn = torch.zeros(B, self.attn_dim, device=self.device)
            zeros_state = torch.zeros(B, self.state_dim, device=self.device)
            zeros = torch.zeros(B, device=self.device)
            unc, _ = core(
                memoryPrev=zeros_mem,
                attnPrev=zeros_attn,
                stateCurr=zeros_state,
                entropyPrev=zeros,
                tdCurr=zeros,
                worldDeltaTransport=zeros_state,
                worldDeltaPhysics=zeros_state,
                doneCurr=zeros)
            ok = bool(torch.allclose(
                unc,
                torch.full_like(unc, float(core.eps_prior)),
                atol=1e-6,
                rtol=1e-4))
            print(f"UncertaintyFloorNearEps {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"UncertaintyFloorNearEps error: {e}")
            return False

    def TestDoneMaskResetPerSample(self) -> bool:
        try:
            torch.manual_seed(436)
            B = 4
            est = self.NewEstimator().train()
            for _ in range(3):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
                _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            done_mask = torch.tensor([True, False, True, False], device=self.device)
            est.ResetState(doneMask=done_mask)
            est.ResetHebbianMemory(doneMask=done_mask)

            ok = True
            ok &= torch.count_nonzero(
                est.emotion_core.fast_head.H[done_mask]).item() == 0
            ok &= torch.count_nonzero(
                est.emotion_core.fast_head.H[~done_mask]).item() > 0
            ok &= torch.count_nonzero(
                est.emotion_core.slow_head.H[done_mask]).item() == 0
            ok &= torch.count_nonzero(
                est.emotion_core.slow_head.H[~done_mask]).item() > 0
            ok &= int(est.micro.filled[0].item()) == 0
            ok &= int(est.micro.filled[2].item()) == 0
            ok &= int(est.micro.filled[1].item()) > 0
            ok &= int(est.micro.filled[3].item()) > 0
            ok &= 0 not in est._pending_transitions
            ok &= 2 not in est._pending_transitions
            ok &= 1 in est._pending_transitions
            ok &= 3 in est._pending_transitions
            ok &= torch.count_nonzero(est.transport.manifold_tensor_field_ema[done_mask]).item() == 0
            ok &= torch.count_nonzero(est.transport.manifold_tensor_field_ema[~done_mask]).item() > 0
            print(f"DoneMaskResetPerSample {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"DoneMaskResetPerSample error: {e}")
            return False

    def TestDistributionalValueAndTransport(self) -> bool:
        try:
            torch.manual_seed(437)
            B = 5
            est = self.NewEstimator().train()
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
            _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            ok = True
            ok &= bool(torch.isfinite(out.extras["loss_current_graph"]).item())
            ok &= self.TrainingMonitorKeys().issubset(set(out.rComps.keys()))
            ok &= "value_quantiles" not in out.rComps
            ok &= "transport_counterfactual_values" not in out.rComps

            est.zero_grad(set_to_none=True)
            out.extras["loss_transport_delayed_graph"].backward(retain_graph=True)
            est.CaptureTransportGrad(clearParamGrad=True)
            out.loss.backward(retain_graph=True)
            est.CaptureTransportGrad(clearParamGrad=True)
            q_grad = est.quantile_head.weight.grad
            cf_grad = est.transport.cf_trop.W.grad
            if (cf_grad is None or float(cf_grad.detach().abs().max().item()) <= 0.0):
                cf_grad = est._transport_curr_grad.get("cf_trop.W")
            if (cf_grad is None or float(cf_grad.detach().abs().max().item()) <= 0.0):
                cf_grad = est._transport_prev_grad.get("cf_trop.W")
            ok &= q_grad is not None and float(q_grad.abs().max().item()) > 0.0
            ok &= cf_grad is not None and float(cf_grad.abs().max().item()) > 0.0

            print(f"DistributionalValueAndTransport {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"DistributionalValueAndTransport error: {e}")
            return False

    def TestStateMachineAndMicroGraph(self) -> bool:
        try:
            torch.manual_seed(44)
            B = 7
            est = self.NewEstimator().train()

            mem, attn, state = self.RandBatch(B)
            reward, entropy, done0, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
            out1 = self.ForwardOnce(est, mem, attn, state, reward, entropy, done0, d_tr, d_ph)
            pending_alive_1 = torch.stack([est._pending_transitions[i][0]["alive"] for i in range(B)], dim=0)

            ok = True
            ok &= (pending_alive_1.shape == (B,))
            ok &= (sum(len(q) for q in est._pending_transitions.values()) == B)
            ok &= (out1.rComps is not None and out1.rComps["risk"].shape == (B,))
            ok &= torch.equal(est.micro.filled, torch.ones_like(est.micro.filled))

            done1 = torch.ones(B, device=self.device)
            out2 = self.ForwardOnce(est, mem, attn, state, reward, entropy, done1, d_tr, d_ph)
            pending_alive_2 = torch.stack([est._pending_transitions[i][0]["alive"] for i in range(B)], dim=0)
            ok &= (float(pending_alive_2.abs().max().item()) <= 1e-6)
            ok &= (sum(len(q) for q in est._pending_transitions.values()) == B)
            ok &= (out2.rComps is not None and out2.rComps["confidence"].shape == (B,))
            ok &= torch.equal(est.micro.filled, torch.zeros_like(est.micro.filled))

            print(f"StateMachineAndMicroGraph {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"StateMachineAndMicroGraph error: {e}")
            return False

    def TestBatchResizeAndPredictorShapes(self) -> bool:
        try:
            torch.manual_seed(45)
            est = self.NewEstimator().train()

            B1 = 4
            mem1, attn1, state1 = self.RandBatch(B1)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B1, doneProb=0.0)
            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)

            B2 = 11
            mem2, attn2, state2 = self.RandBatch(B2)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B2, doneProb=0.0)
            _ = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            ok = True
            ok &= (sum(len(q) for q in est._pending_transitions.values()) == B2)
            ok &= (est.td_out_ema.mean.numel() == B2 and est.td_out_ema.var.numel() == B2)
            ok &= (est.reward_predictor.kf_mean.numel() == B2)
            ok &= (est.done_predictor.kf_mean.numel() == B2)
            ok &= (est.unc_core.td_ema.mean.numel() == B2)
            ok &= (est.unc_core.ent_ema.mean.numel() == B2)
            ok &= (est.micro.anchor_value.size(0) == B2)
            ok &= (est.micro.anchor_value.size(-1) == est.value_tensor_dim)

            print(f"BatchResizeAndPredictorShapes {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"BatchResizeAndPredictorShapes error: {e}")
            return False

    def TestResetFunctions(self) -> bool:
        try:
            torch.manual_seed(46)
            B = 6
            est = self.NewEstimator().train()

            for _ in range(3):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
                _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            est.ResetState()
            est.ResetHebbianMemory()

            ok = True
            ok &= len(est._pending_transitions) == 0
            ok &= (int(est.micro._step) == 0)
            ok &= torch.equal(est.micro.filled, torch.zeros_like(est.micro.filled))
            ok &= torch.equal(est.micro.ptr, torch.zeros_like(est.micro.ptr))
            ok &= (float(est.emotion_core.fast_head.H.abs().sum().item()) <= 1e-12)
            ok &= (float(est.emotion_core.slow_head.H.abs().sum().item()) <= 1e-12)
            ok &= (float(est.transport.manifold_tensor_field_ema.abs().sum().item()) <= 1e-12)

            print(f"ResetFunctions {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ResetFunctions error: {e}")
            return False

    def TestAllTrainableParamsHaveGradAndStep(self) -> bool:
        try:
            torch.manual_seed(101)
            B = 10
            est = self.NewEstimator()
            est.train()
            for ad in [est.fc1_adapter, est.fc2_adapter]:
                ad.Grow(2, init=None, freezeOld=True)

            opt = torch.optim.Adam(est.parameters(), lr=1e-3)
            trainable = {
                n: p for n, p in est.named_parameters()
                if p.requires_grad and p.numel() > 0 and not n.startswith("transport.")}
            grad_seen = {n: False for n in trainable}

            with torch.no_grad():
                before = {n: p.detach().clone() for n, p in trainable.items()}

            for _ in range(4):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B)

                out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
                out = est.RefineEntityOntologyRisk(
                    out,
                    self.OntologyPhysicalState(B))
                loss = (
                    out.loss
                    + out.extras["loss_transport_delayed_graph"]
                    + 1e-2 * (out.emotion ** 2).mean())
                if not torch.isfinite(loss).item():
                    print("AllTrainableParamsHaveGradAndStep fail: loss is not finite")
                    return False

                opt.zero_grad(set_to_none=True)
                loss.backward()

                for n, p in trainable.items():
                    g = p.grad
                    if g is None:
                        continue
                    if not torch.isfinite(g).all().item():
                        print(f"AllTrainableParamsHaveGradAndStep fail: non-finite grad at {n}")
                        return False
                    if float(g.abs().max().item()) > 0.0:
                        grad_seen[n] = True

                torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
                opt.step()
                est.AfterOptimizerStep()

            no_grad_names = [n for n, seen in grad_seen.items() if not seen]
            if no_grad_names:
                print("AllTrainableParamsHaveGradAndStep fail: no effective grad seen in")
                for n in no_grad_names:
                    print("  -", n)
                return False

            unchanged = []
            with torch.no_grad():
                for n, p in trainable.items():
                    delta = (p.detach() - before[n]).abs().max().item()
                    if delta <= 0.0:
                        unchanged.append(n)

            if unchanged:
                print("AllTrainableParamsHaveGradAndStep fail: unchanged after training")
                for n in unchanged:
                    print("  -", n)
                return False

            print("AllTrainableParamsHaveGradAndStep pass")
            return True
        except Exception as e:
            print(f"AllTrainableParamsHaveGradAndStep error: {e}")
            return False

    def TestWrapperAlignmentNoDelta(self) -> bool:
        try:
            torch.manual_seed(123)
            B = 6
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            est_ref_eval = self.NewEstimator().eval()
            est_wrap_eval = copy.deepcopy(est_ref_eval).eval()
            wrapper_eval = ValueEstimationOnlineWrapper(est_wrap_eval, initRankEach=0, autoRank=False).eval()

            out_ref_eval = self.ForwardOnce(est_ref_eval, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out_wr_eval = wrapper_eval(
                x={"memoryPrev": mem, "attnPrev": attn, "state": state},
                rewardModel=reward,
                policyEntropyPrev=entropy,
                doneModel=done,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph,)

            atol, rtol = 1e-6, 1e-5
            loss_atol = 2e-5
            ok = True
            ok &= torch.allclose(out_ref_eval.value, out_wr_eval.value, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_eval.tdError, out_wr_eval.tdError, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_eval.uncertainty, out_wr_eval.uncertainty, atol=atol, rtol=rtol)

            est_ref_train = self.NewEstimator().train()
            est_wrap_train = copy.deepcopy(est_ref_train).train()
            wrapper_train = ValueEstimationOnlineWrapper(est_wrap_train, initRankEach=0, autoRank=False).train()

            out_ref_t1 = self.ForwardOnce(est_ref_train, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out_wr_t1 = wrapper_train(
                x={"memoryPrev": mem, "attnPrev": attn, "state": state},
                rewardModel=reward,
                policyEntropyPrev=entropy,
                doneModel=done,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph,)

            out_ref_t2 = self.ForwardOnce(est_ref_train, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out_wr_t2 = wrapper_train(
                x={"memoryPrev": mem, "attnPrev": attn, "state": state},
                rewardModel=reward,
                policyEntropyPrev=entropy,
                doneModel=done,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph,)

            ok &= torch.allclose(out_ref_t1.value, out_wr_t1.value, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_t1.tdError, out_wr_t1.tdError, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_t1.uncertainty, out_wr_t1.uncertainty, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_t1.loss, out_wr_t1.loss, atol=loss_atol, rtol=rtol)

            ok &= torch.allclose(out_ref_t2.value, out_wr_t2.value, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_t2.tdError, out_wr_t2.tdError, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_t2.uncertainty, out_wr_t2.uncertainty, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_ref_t2.loss, out_wr_t2.loss, atol=loss_atol, rtol=rtol)
            ok &= sorted(out_ref_t2.extras.keys()) == sorted(out_wr_t2.extras.keys())
            ok &= sorted(out_ref_t2.rComps.keys()) == sorted(out_wr_t2.rComps.keys())

            print(f"WrapperAlignmentNoDelta {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"WrapperAlignmentNoDelta error: {e}")
            return False

    def TestWrapperCandidateParamsTrainable(self) -> bool:
        try:
            torch.manual_seed(77)
            B = 8
            base = self.NewEstimator()
            wrapper = ValueEstimationOnlineWrapper(base, initRankEach=0, autoRank=False)
            wrapper.train()

            active_slots = [("fc1", 0), ("fc2", 0), ("qhead", 0)]
            for site, layer_idx in active_slots:
                spec = wrapper.sites[site]
                a, b, s = spec.allocFn(2, wrapper.deviceRef, wrapper.dtypeRef)
                wrapper.cand[site][layer_idx]["A"].append(a)
                wrapper.cand[site][layer_idx]["B"].append(b)
                wrapper.cand[site][layer_idx]["s"].append(s)

            cand_params = list(wrapper.CandParameters())
            if len(cand_params) == 0:
                print("WrapperCandidateParamsTrainable fail: no candidate parameters")
                return False

            with torch.no_grad():
                before = [p.detach().clone() for p in cand_params]

            grad_seen = [False for _ in cand_params]
            opt = torch.optim.Adam(cand_params, lr=5e-3)

            for _ in range(8):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

                out = wrapper(
                    x={"memoryPrev": mem, "attnPrev": attn, "state": state},
                    rewardModel=reward,
                    policyEntropyPrev=entropy,
                    doneModel=done,
                    worldDeltaTransport=d_tr,
                    worldDeltaPhysics=d_ph,)

                loss = out.loss
                if not torch.isfinite(loss).item():
                    print("WrapperCandidateParamsTrainable fail: loss is not finite")
                    return False

                opt.zero_grad(set_to_none=True)
                loss.backward()

                for i, p in enumerate(cand_params):
                    g = p.grad
                    if g is None:
                        continue
                    if not torch.isfinite(g).all().item():
                        print("WrapperCandidateParamsTrainable fail: non-finite candidate grad")
                        return False
                    if float(g.abs().max().item()) > 0.0:
                        grad_seen[i] = True

                torch.nn.utils.clip_grad_norm_(cand_params, 1.0)
                opt.step()
                wrapper.AfterOptimizerStep()

            if any(not seen for seen in grad_seen):
                print("WrapperCandidateParamsTrainable fail: some candidate params never got effective grad")
                return False

            with torch.no_grad():
                changed = []
                for b, p in zip(before, cand_params):
                    changed.append(float((p.detach() - b).abs().max().item()) > 0.0)
                trainable_base_ids = {
                    id(p)
                    for p in wrapper.base.transport.parameters()}
                trainable_base_ids.update(
                    id(p)
                    for p in wrapper.base.return_value_head.parameters())
                base_frozen_ok = not any(
                    p.requires_grad
                    for p in wrapper.base.parameters()
                    if id(p) not in trainable_base_ids)
                transport_trainable = all(
                    p.requires_grad for p in wrapper.base.transport.parameters())
                return_head_trainable = all(
                    p.requires_grad
                    for p in wrapper.base.return_value_head.parameters())

            ok = (
                all(changed)
                and base_frozen_ok
                and transport_trainable
                and return_head_trainable)
            print(f"WrapperCandidateParamsTrainable {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"WrapperCandidateParamsTrainable error: {e}")
            return False

    def TestWrapperTransportManualGrad(self) -> bool:
        try:
            torch.manual_seed(20260720)
            B = 2
            base = self.NewSmallEstimator().train()
            wrapper = ValueEstimationOnlineWrapper(
                base,
                initRankEach=1,
                autoRank=False).train()

            for _ in range(2):
                mem, attn, state = self.RandSmallBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSmallSignals(B)
                out = wrapper(
                    x=(mem, attn, state),
                    rewardModel=reward,
                    doneModel=done,
                    policyEntropyPrev=entropy,
                    worldDeltaTransport=d_tr,
                    worldDeltaPhysics=d_ph)

            before = self.CloneTransportParams(base)
            out.extras["loss_transport_delayed_graph"].backward()
            captured = wrapper.CaptureTransportGrad()
            applied = wrapper.ApplyTransportManualGrad(lr=1e-3)
            delta = self.MaxTransportParamDelta(before, base)
            ok = out.extras["loss_transport_delayed_graph"].requires_grad
            ok &= captured["captured"] > 0.0
            ok &= applied["updated"] > 0.0
            ok &= delta > 0.0
            print(f"WrapperTransportManualGrad {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"WrapperTransportManualGrad error: {e}")
            return False

    def TestWrapperSimCommitEquivalence(self) -> bool:
        try:
            torch.manual_seed(78)
            B = 6
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            base = self.NewEstimator().eval()
            wrapper = ValueEstimationOnlineWrapper(base, initRankEach=0, autoRank=False).eval()

            spec = wrapper.sites["qhead"]
            a, b, s = spec.allocFn(2, wrapper.deviceRef, wrapper.dtypeRef)
            with torch.no_grad():
                a.mul_(0.1)
                b.mul_(0.1)
                s.fill_(0.7)
            wrapper.cand["qhead"][0]["A"].append(a)
            wrapper.cand["qhead"][0]["B"].append(b)
            wrapper.cand["qhead"][0]["s"].append(s)

            snap = wrapper.base.ExportState()
            out_sim = wrapper(
                x={"memoryPrev": mem, "attnPrev": attn, "state": state},
                rewardModel=reward,
                policyEntropyPrev=entropy,
                doneModel=done,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph,)

            wrapper.base.ImportState(snap)
            commit_info = wrapper.Update("commit")
            out_after = wrapper(
                x={"memoryPrev": mem, "attnPrev": attn, "state": state},
                rewardModel=reward,
                policyEntropyPrev=entropy,
                doneModel=done,
                worldDeltaTransport=d_tr,
                worldDeltaPhysics=d_ph,)

            atol, rtol = 1e-6, 1e-5
            ok = True
            ok &= torch.allclose(out_sim.value, out_after.value, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_sim.tdError, out_after.tdError, atol=atol, rtol=rtol)
            ok &= torch.allclose(out_sim.uncertainty, out_after.uncertainty, atol=atol, rtol=rtol)
            ok &= (commit_info.get("ok", False) is True)
            ok &= (commit_info.get("commit_stats", {}).get("committed_triples", 0.0) > 0.0)

            print(f"WrapperSimCommitEquivalence {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"WrapperSimCommitEquivalence error: {e}")
            return False

    def TestWrapperUpdateWorkflow(self) -> bool:
        try:
            torch.manual_seed(79)
            B = 8
            base = self.NewEstimator()
            wrapper = ValueEstimationOnlineWrapper(base, initRankEach=1, autoRank=True).train()

            cand_params = list(wrapper.CandParameters())
            if len(cand_params) == 0:
                print("WrapperUpdateWorkflow fail: no initial candidates")
                return False

            opt = torch.optim.Adam(cand_params, lr=5e-3)
            for _ in range(3):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
                out = wrapper(
                    x={"memoryPrev": mem, "attnPrev": attn, "state": state},
                    rewardModel=reward,
                    policyEntropyPrev=entropy,
                    doneModel=done,
                    worldDeltaTransport=d_tr,
                    worldDeltaPhysics=d_ph,)
                opt.zero_grad(set_to_none=True)
                out.loss.backward()
                opt.step()
                wrapper.AfterOptimizerStep()
                wrapper.Update("accumulategrads")

            r0 = wrapper.Update("ranks")
            g1 = wrapper.Update("grow", addEach=1)
            a1 = wrapper.Update("autogrow")
            c1 = wrapper.Update("commit")
            r1 = wrapper.Update("ranks")
            rb = wrapper.Update("rollback")
            rs = wrapper.Update("reset", initRankEach=0)

            ok = True
            ok &= bool(r0.get("ok", False))
            ok &= bool(g1.get("ok", False))
            ok &= bool(a1.get("ok", False))
            ok &= bool(c1.get("ok", False))
            ok &= bool(r1.get("ok", False))
            ok &= bool(rb.get("ok", False))
            ok &= bool(rs.get("ok", False))
            ok &= (c1.get("commit_stats", {}).get("committed_triples", 0.0) >= 0.0)
            ok &= all(v == 0 for v in r1["ranks"]["sum"].values())
            ok &= all(v == 0 for v in rs["ranks"]["sum"].values())

            print(f"WrapperUpdateWorkflow {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"WrapperUpdateWorkflow error: {e}")
            return False

    def TestExportImportStateRoundTrip(self) -> bool:
        try:
            torch.manual_seed(88)
            B = 5
            est1 = self.NewEstimator().train()

            for _ in range(3):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
                _ = self.ForwardOnce(est1, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            dyn_state = est1.ExportState()
            missing_state = dict(dyn_state)
            del missing_state["return_value_prev"]
            missing_rejected = False
            try:
                est1.ImportState(missing_state)
            except TypeError:
                missing_rejected = True

            unexpected_state = dict(dyn_state)
            unexpected_state["legacy_runtime_field"] = None
            unexpected_rejected = False
            try:
                est1.ImportState(unexpected_state)
            except TypeError:
                unexpected_rejected = True

            exported_pending = [
                items[0]
                for items in dyn_state["pending_transitions"].values()
                if len(items) > 0]

            est2 = self.NewEstimator().train()
            src = est1.state_dict()
            dst = est2.state_dict()
            loadable = {k: v for k, v in src.items() if (k in dst and dst[k].shape == v.shape)}
            est2.load_state_dict(loadable, strict=False)
            est2.ImportState(dyn_state)
            imported_pending = [
                queue[0]
                for queue in est2._pending_transitions.values()
                if len(queue) > 0]

            ok = True
            ok &= missing_rejected
            ok &= unexpected_rejected
            ok &= len(exported_pending) == B
            ok &= len({id(item["transport_state"]) for item in exported_pending}) == 1
            ok &= len(imported_pending) == B
            ok &= len({id(item["transport_state"]) for item in imported_pending}) == 1
            ok &= len(est2._pending_transitions) == len(est1._pending_transitions)
            ok &= torch.equal(est1.micro.filled, est2.micro.filled)
            ok &= torch.equal(est1.micro.ptr, est2.micro.ptr)
            ok &= torch.allclose(est1.micro.anchor_value, est2.micro.anchor_value)
            ok &= torch.allclose(est1.micro.anchor_value_next, est2.micro.anchor_value_next)
            ok &= torch.allclose(est1.reward_predictor.kf_mean, est2.reward_predictor.kf_mean)
            ok &= torch.allclose(est1.done_predictor.kf_mean, est2.done_predictor.kf_mean)
            ok &= torch.equal(est1.reward_predictor.hist_len, est2.reward_predictor.hist_len)
            ok &= torch.equal(est1.done_predictor.hist_len, est2.done_predictor.hist_len)
            ok &= torch.allclose(est1.emotion_core.fast_head.H, est2.emotion_core.fast_head.H)
            ok &= torch.allclose(est1.emotion_core.slow_head.H, est2.emotion_core.slow_head.H)

            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None)
            out1 = self.ForwardOnce(est1, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, self.device)
            out2 = self.ForwardOnce(est2, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            atol, rtol = 1e-6, 1e-5
            ok &= torch.allclose(out1.value, out2.value, atol=atol, rtol=rtol)
            ok &= torch.allclose(
                out1.extras["loss_transport_delayed_graph"],
                out2.extras["loss_transport_delayed_graph"],
                atol=atol,
                rtol=rtol)

            print(f"ExportImportStateRoundTrip {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ExportImportStateRoundTrip error: {e}")
            return False

    def TestTransientTrainingGraphRoundTrip(self) -> bool:
        try:
            torch.manual_seed(20260717)
            source = self.NewSmallEstimator().train()
            _ = self.ForwardOnce(
                source,
                *self.RandSmallBatch(2),
                *self.RandSmallSignals(2))

            runtime_state = source.ExportState()
            restored = self.NewSmallEstimator().train()
            source_parameters = source.state_dict()
            restored_parameters = restored.state_dict()
            loadable = {
                name: tensor
                for name, tensor in source_parameters.items()
                if (
                    name in restored_parameters
                    and restored_parameters[name].shape == tensor.shape)}
            restored.load_state_dict(loadable, strict=False)
            restored.ImportState(runtime_state)

            out = self.ForwardOnce(
                restored,
                *self.RandSmallBatch(2),
                *self.RandSmallSignals(2))
            delayed = out.extras["loss_transport_delayed_graph"]
            delayed.backward()
            captured = restored.CaptureTransportGrad()

            ok = delayed.requires_grad
            ok &= captured["captured"] > 0.0
            print(f"TransientTrainingGraphRoundTrip {'pass' if ok else 'fail'}")
            return bool(ok)
        except Exception as e:
            print(f"TransientTrainingGraphRoundTrip error: {e}")
            return False

    def TestLossDecreases(self, steps: int = 48, batchSize: int = 8) -> bool:
        try:
            torch.manual_seed(2026)
            est = self.NewEstimator().train()
            opt = torch.optim.Adam(est.parameters(), lr=5e-4)

            mem, attn, state = self.RandBatch(batchSize)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(batchSize, doneProb=0.0)

            with torch.no_grad():
                for _ in range(8):
                    _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            snap = est.ExportState()

            losses: List[float] = []
            for t in range(int(steps)):
                est.ImportState(snap)

                out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
                loss = out.loss
                if not torch.isfinite(loss).item():
                    print(f"LossDecreases fail: non-finite loss at step {t}")
                    return False

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
                opt.step()
                est.AfterOptimizerStep()

                losses.append(float(loss.detach().item()))

            if len(losses) < 20:
                print("LossDecreases fail: too few steps")
                return False

            cut = max(8, len(losses) // 6)
            first_win = losses[cut: cut + max(8, len(losses) // 4)]
            last_win = losses[-max(8, len(losses) // 4):]
            first_mean = sum(first_win) / max(1, len(first_win))
            last_mean = sum(last_win) / max(1, len(last_win))
            first_med = stats.median(first_win)
            last_med = stats.median(last_win)
            loss_start = float(losses[0])
            loss_end = float(losses[-1])
            trend_from = float(first_mean)
            trend_to = float(last_mean)
            trend_drop = float(trend_from - trend_to)
            trend_drop_ratio = float(trend_drop / (abs(trend_from) + 1e-12))

            self.last_loss_report = {
                "loss_start": loss_start,
                "loss_end": loss_end,
                "trend_from": trend_from,
                "trend_to": trend_to,
                "trend_drop": trend_drop,
                "trend_drop_ratio": trend_drop_ratio,
                "first_mean": float(first_mean),
                "last_mean": float(last_mean),
                "first_median": float(first_med),
                "last_median": float(last_med),}

            ok = (last_mean < first_mean) and (last_med < first_med)
            print(
                f"LossDecreases {'pass' if ok else 'fail'} "
                f"(trend loss: {trend_from:.6f} -> {trend_to:.6f}, "
                f"drop={trend_drop:.6f}, drop_ratio={trend_drop_ratio:.2%}, "
                f"raw start/end: {loss_start:.6f} -> {loss_end:.6f}, "
                f"first_mean={first_mean:.6f}, last_mean={last_mean:.6f}, "
                f"first_med={first_med:.6f}, last_med={last_med:.6f})")

            return ok

        except Exception as e:
            print(f"LossDecreases error: {e}")
            return False

    def TestNoNanStress(self, steps: int = 20) -> bool:
        try:
            torch.manual_seed(2027)
            est = self.NewEstimator().train()
            opt = torch.optim.Adam(est.parameters(), lr=8e-4)

            for t in range(int(steps)):
                B = int(torch.randint(low=4, high=9, size=(1,)).item())
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.2)
                out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
                if not torch.isfinite(out.loss).item():
                    print(f"NoNanStress fail: non-finite loss at step {t}")
                    return False

                opt.zero_grad(set_to_none=True)
                out.loss.backward()
                for n, p in est.named_parameters():
                    if p.grad is not None and (not torch.isfinite(p.grad).all().item()):
                        print(f"NoNanStress fail: non-finite grad at step {t} ({n})")
                        return False
                torch.nn.utils.clip_grad_norm_(est.parameters(), 1.0)
                opt.step()
                est.AfterOptimizerStep()

            print("NoNanStress pass")
            return True
        except Exception as e:
            print(f"NoNanStress error: {e}")
            return False

    def TestBellmanReturnSemantics(self) -> bool:
        try:
            torch.manual_seed(20260718)
            B = 2
            batch = self.RandSmallBatch(B)
            _, entropy, _, d_tr, d_ph = self.RandSmallSignals(B)

            sign_est = self.NewSmallEstimator().eval()
            with torch.no_grad():
                sign_est.return_value_head[-1].weight.zero_()
                sign_est.return_value_head[-1].bias.zero_()
                _ = self.ForwardOnce(
                    sign_est, *batch, torch.zeros(B, device=self.device),
                    entropy, torch.zeros(B, device=self.device), d_tr, d_ph)
                positive = self.ForwardOnce(
                    sign_est, *batch, torch.ones(B, device=self.device),
                    entropy, torch.zeros(B, device=self.device), d_tr, d_ph)
                sign_est.ResetState()
                sign_est.ResetHebbianMemory()
                _ = self.ForwardOnce(
                    sign_est, *batch, torch.zeros(B, device=self.device),
                    entropy, torch.zeros(B, device=self.device), d_tr, d_ph)
                negative = self.ForwardOnce(
                    sign_est, *batch, -torch.ones(B, device=self.device),
                    entropy, torch.zeros(B, device=self.device), d_tr, d_ph)

            terminal_est = self.NewSmallEstimator().eval()
            with torch.no_grad():
                terminal_est.return_value_head[-1].weight.zero_()
                terminal_est.return_value_head[-1].bias.fill_(2.0)
                _ = self.ForwardOnce(
                    terminal_est, *batch, torch.zeros(B, device=self.device),
                    entropy, torch.zeros(B, device=self.device), d_tr, d_ph)
                terminal = self.ForwardOnce(
                    terminal_est, *batch, torch.zeros(B, device=self.device),
                    entropy, torch.ones(B, device=self.device), d_tr, d_ph)
                terminal_est.ResetState()
                terminal_est.ResetHebbianMemory()
                _ = self.ForwardOnce(
                    terminal_est, *batch, torch.zeros(B, device=self.device),
                    entropy, torch.zeros(B, device=self.device), d_tr, d_ph)
                nonterminal = self.ForwardOnce(
                    terminal_est, *batch, torch.zeros(B, device=self.device),
                    entropy, torch.zeros(B, device=self.device), d_tr, d_ph)

            train_est = self.NewSmallEstimator().train()
            with torch.no_grad():
                train_est.return_value_head[-1].weight.zero_()
                train_est.return_value_head[-1].bias.zero_()
            first_train = self.ForwardOnce(
                train_est, *batch, torch.zeros(B, device=self.device),
                entropy, torch.zeros(B, device=self.device), d_tr, d_ph)
            return_parameters = list(train_est.return_value_head.parameters())
            return_optimizer = torch.optim.SGD(return_parameters, lr=1e-2)
            return_optimizer.zero_grad(set_to_none=True)
            first_train.returnValue.mean().backward()
            return_optimizer.step()
            trained = self.ForwardOnce(
                train_est, *batch, torch.ones(B, device=self.device),
                entropy, torch.zeros(B, device=self.device), d_tr, d_ph)
            return_optimizer.zero_grad(set_to_none=True)
            trained.extras["loss_transport_delayed_graph"].backward(
                inputs=return_parameters)
            return_grad = sum(
                float(parameter.grad.detach().abs().sum().item())
                for parameter in return_parameters
                if parameter.grad is not None)

            ok = bool(
                torch.all(positive.returnAdvantage > 0.0).item()
                and torch.all(negative.returnAdvantage < 0.0).item()
                and torch.allclose(
                    terminal.returnAdvantage,
                    terminal.returnAdvantage.new_full((B,), -2.0),
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    nonterminal.returnAdvantage,
                    nonterminal.returnAdvantage.new_full(
                        (B,), 2.0 * (terminal_est.return_discount - 1.0)),
                    atol=1e-6,
                    rtol=1e-6)
                and float(trained.extras["loss_return"].item()) > 0.0
                and return_grad > 0.0)
            print(f"BellmanReturnSemantics {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"BellmanReturnSemantics error: {e}")
            return False

    def TestValueTensorBellmanAnchor(self) -> bool:
        try:
            torch.manual_seed(20260719)
            B = 3
            est = self.NewSmallEstimator().train()
            with torch.no_grad():
                est.value_tensor_tail[-1].weight.zero_()
                est.value_tensor_tail[-1].bias.zero_()
                est.value_tensor_out_norm.weight.fill_(1.0)
                est.value_tensor_out_norm.bias.zero_()
                est.return_value_head[-1].weight.zero_()
                est.return_value_head[-1].bias.fill_(1.5)

            batch = self.RandSmallBatch(B)
            _, entropy, _, d_tr, d_ph = self.RandSmallSignals(B)
            done = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            out = self.ForwardOnce(
                est,
                *batch,
                torch.zeros(B, device=self.device),
                entropy,
                done,
                d_tr,
                d_ph)

            return_target = (
                (1.0 - done) * torch.tanh(out.returnValue.detach()))
            anchor_graph = F.smooth_l1_loss(
                out.value[:, 0],
                return_target,
                reduction="mean")
            value_parameters = (
                list(est.value_tensor_tail.parameters())
                + list(est.value_tensor_out_norm.parameters())
                + [est.value_tensor_log_scale])
            return_parameters = list(est.return_value_head.parameters())
            gradients = torch.autograd.grad(
                anchor_graph,
                value_parameters + return_parameters,
                allow_unused=True)
            value_grad = sum(
                float(grad.detach().abs().sum().item())
                for grad in gradients[:len(value_parameters)]
                if grad is not None)
            return_grad = sum(
                float(grad.detach().abs().sum().item())
                for grad in gradients[len(value_parameters):]
                if grad is not None)

            current_part_names = (
                "loss_physical_td",
                "loss_physical_aux",
                "loss_physical_param_reg",
                "loss_value_tensor_energy",
                "loss_value_return_anchor",
                "loss_quantile",
                "loss_ensemble",)
            expected_current = sum(
                (out.extras[name] for name in current_part_names),
                out.loss.new_zeros(()))
            ok = bool(
                torch.all(return_target[[0, 2]].abs() > 0.0).item()
                and float(return_target[1].item()) == 0.0
                and float(anchor_graph.detach().item()) > 0.0
                and value_grad > 0.0
                and return_grad == 0.0
                and torch.allclose(
                    out.extras["loss_value_return_anchor"],
                    anchor_graph.detach(),
                    atol=1e-7,
                    rtol=1e-6)
                and torch.allclose(
                    out.extras["loss_current_graph"].detach(),
                    expected_current,
                    atol=1e-7,
                    rtol=1e-6))
            print(f"ValueTensorBellmanAnchor {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ValueTensorBellmanAnchor error: {e}")
            return False

    def TestCognitiveValueInterfaces(self) -> bool:
        try:
            torch.manual_seed(131)
            estimator = self.NewSmallEstimator().train()
            memory, attention, state = self.RandSmallBatch(3)
            proposal = torch.randn(3, 11, device=self.device, requires_grad=True)
            before = estimator.PreDecisionValue(memory, attention, state)
            after = estimator.ProspectiveValue(
                memory,
                attention,
                state,
                proposal)
            if before.feature.shape != (3, estimator.cognitive_value_dim):
                return False
            if after.feature.shape != before.feature.shape:
                return False
            for estimate in (before, after):
                values = (
                    estimate.feature,
                    estimate.coarseProgress,
                    estimate.detailProgress,
                    estimate.planStaleness,
                    estimate.replanBenefit,
                    estimate.computeCost,
                    estimate.feasibility,
                    estimate.safetyConstraint)
                if not all(bool(torch.isfinite(value).all().item()) for value in values):
                    return False
                for probability in (
                    estimate.coarseProgress,
                    estimate.detailProgress,
                    estimate.planStaleness,
                    estimate.feasibility,
                    estimate.safetyConstraint):
                    if not bool(((probability >= 0.0) & (probability <= 1.0)).all().item()):
                        return False
                if not bool((estimate.replanBenefit >= 0.0).all().item()):
                    return False
                if not bool((estimate.computeCost >= 0.0).all().item()):
                    return False
            objective = (
                before.feature.square().mean()
                + after.feature.square().mean()
                + after.coarseProgress.mean()
                + after.detailProgress.mean()
                + after.planStaleness.mean()
                + after.replanBenefit.mean()
                + after.computeCost.mean()
                + after.feasibility.mean()
                + after.safetyConstraint.mean())
            objective.backward()
            if proposal.grad is None or not bool(torch.isfinite(proposal.grad).all().item()):
                return False
            watched = (
                estimator.cognitive_value_head[1].weight,
                estimator.prospective_context_head[0].weight,
                estimator.cognitive_metric_head.weight)
            return all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                for parameter in watched)
        except Exception as error:
            print(f"CognitiveValueInterfaces error: {error}")
            return False

    def TestForwardCognitiveValueSignals(self) -> bool:
        try:
            torch.manual_seed(137)
            estimator = self.NewSmallEstimator().train()
            memory, attention, state = self.RandSmallBatch(2)
            reward, entropy, done, transport, dynamics = self.RandSmallSignals(2)
            output = self.ForwardOnce(
                estimator,
                memory,
                attention,
                state,
                reward,
                entropy,
                done,
                transport,
                dynamics)
            expected = {
                "coarseProgress",
                "detailProgress",
                "planStaleness",
                "replanBenefit",
                "computeCost",
                "feasibility",
                "safetyConstraint"}
            if not expected.issubset(output.rComps):
                return False
            if "loss_cognitive_value" not in output.extras:
                return False
            output.loss.backward()
            return bool(
                estimator.cognitive_metric_head.weight.grad is not None
                and torch.isfinite(
                    estimator.cognitive_metric_head.weight.grad).all().item())
        except Exception as error:
            print(f"ForwardCognitiveValueSignals error: {error}")
            return False

    def TestSensorimotorPotentialShaping(self) -> bool:
        try:
            shaper = SensorimotorPotentialShaper(
                discount=0.5,
                scale=1.0).to(self.device)
            first = shaper(
                baseReward=torch.zeros(1, device=self.device),
                inconsistency=torch.tensor([2.0], device=self.device),
                observationValid=torch.tensor([True], device=self.device),
                duration=torch.ones(1, device=self.device),
                terminated=torch.tensor([False], device=self.device))
            missing = shaper(
                baseReward=torch.zeros(1, device=self.device),
                inconsistency=torch.tensor([99.0], device=self.device),
                observationValid=torch.tensor([False], device=self.device),
                duration=torch.ones(1, device=self.device),
                terminated=torch.tensor([False], device=self.device))
            terminal = shaper(
                baseReward=torch.zeros(1, device=self.device),
                inconsistency=torch.tensor([1.0], device=self.device),
                observationValid=torch.tensor([True], device=self.device),
                duration=torch.ones(1, device=self.device),
                terminated=torch.tensor([True], device=self.device))
            restart = shaper(
                baseReward=torch.zeros(1, device=self.device),
                inconsistency=torch.tensor([4.0], device=self.device),
                observationValid=torch.tensor([True], device=self.device),
                duration=torch.ones(1, device=self.device),
                terminated=torch.tensor([False], device=self.device))
            cutoff = shaper(
                baseReward=torch.zeros(1, device=self.device),
                inconsistency=torch.tensor([2.0], device=self.device),
                observationValid=torch.tensor([True], device=self.device),
                duration=torch.ones(1, device=self.device),
                terminated=torch.tensor([False], device=self.device),
                truncated=torch.tensor([True], device=self.device))
            return bool(
                torch.allclose(first.shapingReward, torch.tensor([0.0], device=self.device))
                and torch.allclose(missing.previousPotential, torch.tensor([-2.0], device=self.device))
                and torch.allclose(missing.nextPotential, torch.tensor([-2.0], device=self.device))
                and torch.allclose(missing.shapingReward, torch.tensor([1.0], device=self.device))
                and torch.allclose(terminal.nextPotential, torch.tensor([0.0], device=self.device))
                and torch.allclose(terminal.shapingReward, torch.tensor([2.0], device=self.device))
                and not bool(shaper.potential_known.item())
                and torch.allclose(restart.shapingReward, torch.tensor([0.0], device=self.device))
                and torch.allclose(cutoff.previousPotential, torch.tensor([-4.0], device=self.device))
                and torch.allclose(cutoff.nextPotential, torch.tensor([-2.0], device=self.device))
                and torch.allclose(cutoff.shapingReward, torch.tensor([3.0], device=self.device)))
        except Exception as error:
            print(f"SensorimotorPotentialShaping error: {error}")
            return False

    def TestSmdpDiscountedOptionReward(self) -> bool:
        try:
            primitive_rewards = torch.tensor(
                [
                    [[1.0, 2.0, 100.0], [2.0, 4.0, 8.0]],
                    [[3.0, 99.0, 99.0], [4.0, 8.0, 16.0]],
                ],
                device=self.device)
            duration = torch.tensor(
                [[2, 3], [1, 2]],
                device=self.device)
            reward_valid = torch.tensor(
                [
                    [[True, True, True], [True, True, True]],
                    [[True, True, True], [True, False, True]],
                ],
                device=self.device)
            option_reward = SmdpReturnEstimator.DiscountOptionRewards(
                primitive_rewards,
                duration,
                discount=0.5,
                rewardValid=reward_valid)
            expected = torch.tensor(
                [[2.0, 6.0], [3.0, 4.0]],
                device=self.device)
            return bool(torch.allclose(option_reward, expected))
        except Exception as error:
            print(f"SmdpDiscountedOptionReward error: {error}")
            return False

    def TestSmdpTerminationAndTruncation(self) -> bool:
        try:
            option_reward = torch.tensor(
                [[1.0, 1.0], [2.0, 2.0]],
                device=self.device)
            values = torch.tensor(
                [[0.5, 0.5], [1.0, 1.0], [3.0, 3.0]],
                device=self.device)
            duration = torch.tensor(
                [[2.0, 2.0], [1.0, 1.0]],
                device=self.device)
            terminated = torch.tensor(
                [[False, False], [True, False]],
                device=self.device)
            truncated = torch.tensor(
                [[False, False], [False, True]],
                device=self.device)
            output = SmdpReturnEstimator.EstimateAdvantages(
                optionReward=option_reward,
                values=values,
                duration=duration,
                terminated=terminated,
                truncated=truncated,
                discount=0.5,
                traceDecay=0.8)
            early_cutoff = SmdpReturnEstimator.EstimateAdvantages(
                optionReward=option_reward[:, :1],
                values=values[:, :1],
                duration=duration[:, :1],
                terminated=torch.zeros_like(terminated[:, :1]),
                truncated=torch.tensor(
                    [[True], [False]],
                    device=self.device),
                discount=0.5,
                traceDecay=0.8)
            expected_delta = torch.tensor(
                [[0.75, 0.75], [1.0, 2.5]],
                device=self.device)
            expected_advantage = torch.tensor(
                [[0.91, 1.15], [1.0, 2.5]],
                device=self.device)
            expected_return = torch.tensor(
                [[1.41, 1.65], [2.0, 3.5]],
                device=self.device)
            return bool(
                torch.allclose(output.temporalDifference, expected_delta)
                and torch.allclose(output.advantage, expected_advantage)
                and torch.allclose(output.returnTarget, expected_return)
                and torch.allclose(
                    output.bootstrapDiscount,
                    torch.tensor([[0.25, 0.25], [0.5, 0.5]], device=self.device))
                and torch.allclose(
                    output.traceDiscount,
                    torch.tensor([[0.16, 0.16], [0.4, 0.4]], device=self.device))
                and torch.allclose(
                    early_cutoff.advantage[0],
                    early_cutoff.temporalDifference[0]))
        except Exception as error:
            print(f"SmdpTerminationAndTruncation error: {error}")
            return False

    def RunAll(self):
        import gc as _gc
        tests = [
            ("ExtractorFunctional", self.TestExtractorFunctional),
            ("ValueEstimatorIOShapes", self.TestValueEstimatorIOShapes),
            ("TDUncertaintyBounds", self.TestTDUncertaintyBounds),
            ("PhysicalTDNoCrossBatch", self.TestPhysicalTDNoCrossBatch),
            ("ModelTargetOnly", self.TestModelTargetOnly),
            ("DifferentialTDSemantics", self.TestDifferentialTDSemantics),
            ("CurrentValueUsedAsDelayedLabel", self.TestCurrentValueUsedAsDelayedLabel),
            ("PendingQueueStableStreamPairing", self.TestPendingQueueStableStreamPairing),
            ("TemporalPairingOfUncertainty", self.TestTemporalPairingOfUncertainty),
            ("TransportManifoldFieldGrad", self.TestTransportManifoldFieldGrad),
            ("ManualTransportGradWorkflow", self.TestManualTransportGradWorkflow),
            ("TransportDelayedGradientPipeline", self.TestTransportDelayedGradientPipeline),
            ("HebbianSnapshotUsedForDelayedRebuild", self.TestHebbianSnapshotUsedForDelayedRebuild),
            ("TerminalMaskDelayedLosses", self.TestTerminalMaskDelayedLosses),
            ("AllTerminalDelayedLossFullyMasked", self.TestAllTerminalDelayedLossFullyMasked),
            ("PerRowPredictorResetUsesOnlyNewEpisodeHistory", self.TestPerRowPredictorResetUsesOnlyNewEpisodeHistory),
            ("CurrentDoneMasksTemporalUncertaintyEvidence", self.TestCurrentDoneMasksTemporalUncertaintyEvidence),
            ("ManifoldFieldEmaIsRuntimeState", self.TestManifoldFieldEmaIsRuntimeState),
            ("TransportSnapshotUsesSameRuntimeState", self.TestTransportSnapshotUsesSameRuntimeState),
            ("CurrentTransitionLossMasksTerminalGeometry", self.TestCurrentTransitionLossMasksTerminalGeometry),
            ("HebbianLinearFWLifecycle", self.TestHebbianLinearFWLifecycle),
            ("StreamIdEpisodeReplacement", self.TestStreamIdEpisodeReplacement),
            ("GeometricDistances", self.TestGeometricDistances),
            ("UncertaintyFloorNearEps", self.TestUncertaintyFloorNearEps),
            ("DoneMaskResetPerSample", self.TestDoneMaskResetPerSample),
            ("DistributionalValueAndTransport", self.TestDistributionalValueAndTransport),
            ("StateMachineAndMicroGraph", self.TestStateMachineAndMicroGraph),
            ("BatchResizeAndPredictorShapes", self.TestBatchResizeAndPredictorShapes),
            ("ResetFunctions", self.TestResetFunctions),
            ("EntityOntologyRiskSeparation", self.TestEntityOntologyRiskSeparation),
            ("AllTrainableParamsHaveGradAndStep", self.TestAllTrainableParamsHaveGradAndStep),
            ("WrapperAlignmentNoDelta", self.TestWrapperAlignmentNoDelta),
            ("WrapperCandidateParamsTrainable", self.TestWrapperCandidateParamsTrainable),
            ("WrapperTransportManualGrad", self.TestWrapperTransportManualGrad),
            ("WrapperSimCommitEquivalence", self.TestWrapperSimCommitEquivalence),
            ("WrapperUpdateWorkflow", self.TestWrapperUpdateWorkflow),
            ("ExportImportStateRoundTrip", self.TestExportImportStateRoundTrip),
            ("TransientTrainingGraphRoundTrip", self.TestTransientTrainingGraphRoundTrip),
            ("BellmanReturnSemantics", self.TestBellmanReturnSemantics),
            ("ValueTensorBellmanAnchor", self.TestValueTensorBellmanAnchor),
            ("CognitiveValueInterfaces", self.TestCognitiveValueInterfaces),
            ("ForwardCognitiveValueSignals", self.TestForwardCognitiveValueSignals),
            ("SensorimotorPotentialShaping", self.TestSensorimotorPotentialShaping),
            ("SmdpDiscountedOptionReward", self.TestSmdpDiscountedOptionReward),
            ("SmdpTerminationAndTruncation", self.TestSmdpTerminationAndTruncation),
            ("LossDecreases", self.TestLossDecreases),
            ("NoNanStress", self.TestNoNanStress),]
        results = {}
        for name, fn in tests:
            results[name] = fn()
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        passed = sum(1 for v in results.values() if v)
        total = len(results)
        print(f"\n[ValueEstimationModule Tests] {passed}/{total} passed.")
        if self.last_loss_report is not None:
            r = self.last_loss_report
            print(
                f"[Loss Report] trend {r['trend_from']:.6f} -> {r['trend_to']:.6f}, "
                f"drop={r['trend_drop']:.6f}, drop_ratio={r['trend_drop_ratio']:.2%}, "
                f"raw {r['loss_start']:.6f} -> {r['loss_end']:.6f}")
        return results
