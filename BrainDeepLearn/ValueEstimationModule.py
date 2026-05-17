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


class HebbianLinearFW(AGICoreModule):
    def __init__(
        self,
        inFeatures: int,
        outFeatures: int,
        bias: bool = True,
        useHebbian = True,
        *,
        eta: float = 1e-3,
        lam: float = 0.1,
        betaMix: float = 0.1,
        cap: float = 1.0,
        useOja: bool = True,):
        super().__init__()
        self.in_f = int(inFeatures)
        self.out_f = int(outFeatures)
        self.use_hebbian = useHebbian
        self.use_bias = bool(bias)

        self.weight = nn.Parameter(torch.empty(self.out_f, self.in_f))
        self.bias = nn.Parameter(torch.zeros(self.out_f))

        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)

        self.register_buffer("H", torch.zeros(1, self.out_f, self.in_f))

        self.eta = float(eta)
        self.lam = float(lam)
        self.beta_mix = float(betaMix)

        self.cap = cap
        self.use_oja = bool(useOja)

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if self.H.size(0) != B:
            self.H = torch.zeros(B, self.out_f, self.in_f, device=device, dtype=dtype)

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.H.zero_()
            return
        if self.H.size(0) <= 0:
            return
        mask = doneMask.bool().view(-1)
        n = min(int(mask.numel()), int(self.H.size(0)))
        if n > 0:
            rows = mask[:n].nonzero(as_tuple=False).view(-1)
            if rows.numel() > 0:
                self.H[rows] = 0

    @torch.no_grad()
    def ProjectCap(self):
        if self.cap is None:
            return
        
        n = self.H.norm(dim=-1, keepdim=True) # [B,O,1]
        scale = (float(self.cap) / (n + 1e-12)).clamp_max(1.0)
        self.H.mul_(scale)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = int(x.size(0))

        device, dtype = self.device, self.dtype
        self.EnsureB(B, device=device, dtype=dtype)

        W = self.weight
        b = self.bias if self.use_bias else None
        y_base = F.linear(x, W, b) # [B,O]

        if not self.use_hebbian:
            extras = {"use_hebb": x.new_tensor(0.0)} 
            return y_base, extras

        y_hebb = torch.bmm(self.H.detach().clone(), x.unsqueeze(-1)).squeeze(-1) # [B,O]

        beta = x.new_tensor(self.beta_mix)  
        y = y_base + beta * y_hebb

        with torch.no_grad():
            pre = x
            post = y

            pre_n = pre / (pre.norm(dim=-1, keepdim=True) + 1e-6) # [B,I]
            post_n = post / (post.norm(dim=-1, keepdim=True) + 1e-6) # [B,O]

            dH = post_n.unsqueeze(-1) * pre_n.unsqueeze(1) # [B,O,I]

            if self.use_oja:
                post_sq = (post_n ** 2).unsqueeze(-1) # [B,O,1]
                dH = dH - self.H * post_sq

            eta_t = x.new_tensor(self.eta).view(1, 1, 1)
            lam_t = x.new_tensor(self.lam).view(1, 1, 1)

            self.H.mul_(1.0 - lam_t).add_(eta_t * dH)
            self.ProjectCap()

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
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        B = int(B)
        if self.mean.numel() != B:
            self.mean = torch.zeros(B, device=device, dtype=dtype)
            self.var = torch.ones(B, device=device, dtype=dtype)

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
    def Update(self, x: torch.Tensor):
        mask = torch.isfinite(x) # [B]
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
        std = (self.var + self.eps).sqrt().clamp_min(self.eps)
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
    def EnsureB(self, B: int, device, dtype):
        self.td_ema.EnsureB(B, device, dtype)
        self.ent_ema.EnsureB(B, device, dtype)
        self.state_ema.EnsureB(B, device, dtype)
        self.tr_ema.EnsureB(B, device, dtype)
        self.ph_ema.EnsureB(B, device, dtype)
        self.ctx_ema.EnsureB(B, device, dtype)

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
        donePrev: torch.Tensor, # [B]
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:


        B = int(stateCurr.size(0))
        self.EnsureB(B, self.device, self.dtype)

        H_prev = entropyPrev
        td_curr = tdCurr

        alive = (1.0 - donePrev.float()).clamp(0.0, 1.0)
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

        self.td_ema.Update(td_curr_abs.detach())
        self.ent_ema.Update(ent_scaled.detach())
        self.state_ema.Update(state_mag.detach())
        self.tr_ema.Update(tr_mag.detach())
        self.ph_ema.Update(ph_mag.detach())
        self.ctx_ema.Update(ctx_mag.detach())

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
        
        unc_total = (F.softplus(evidence) - math.log(2.0)).clamp_min(0.0) + self.eps_prior

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
            self.kf_mean = torch.empty(0, device=self.device, dtype=self.dtype)
            self.kf_var = torch.empty(0, device=self.device, dtype=self.dtype)
            self.smooth_hist = torch.empty(0, 0, device=self.device, dtype=self.dtype)
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
        if self.smooth_hist.size(0) >= n:
            self.smooth_hist[rows] = 0

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
    def SelectFitWindow(self, hist: torch.Tensor) -> torch.Tensor:
        if self.fit_last_n > 0 and hist.size(1) > self.fit_last_n:
            return hist[:, -self.fit_last_n:]
        return hist

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
        autoPolicy: Optional[str] = None,) -> torch.Tensor:
        z = obsPrev
        B = int(z.size(0))

        mode_now = self.NormMode(mode if mode is not None else self.predict_mode)
        auto_now = self.NormAutoPolicy(autoPolicy if autoPolicy is not None else self.auto_policy)

        if self.kf_mean.numel() != B:
            self.kf_mean = z.detach()
            self.kf_var = torch.full((B,), float(self.init_var), device=self.device, dtype=self.dtype)
            self.smooth_hist = z.unsqueeze(1).detach()
        else:
            x_post, p_post = self.KalmanStep(z, self.kf_mean, self.kf_var)
            self.kf_mean = x_post.detach()
            self.kf_var = p_post.detach()

            hist = torch.cat([self.smooth_hist, x_post.unsqueeze(1)], dim=1)
            if hist.size(1) > self.history_len:
                hist = hist[:, -self.history_len:]
            self.smooth_hist = hist.detach()

        y_fit = self.SelectFitWindow(self.smooth_hist)
        if y_fit.size(1) < 2:
            return y_fit[:, -1]

        y_k, y_s, y_h = self.PredictAll(y_fit)
        return self.SelectPrediction(y_k, y_s, y_h, y_fit, mode_now, auto_now)



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
        self.flow_fusion_residual_scale = 0.05

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
        self.manifold_tensor_field_ema = nn.Parameter(torch.zeros(1, self.manifold_z_dim))

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

    def BuildManifoldConnection(
        self,
        ctx: torch.Tensor,
        vIn: torch.Tensor,
        rawDelta: torch.Tensor,
        branchValueTensorAll: torch.Tensor,
        branchValueTensorMix: torch.Tensor,) -> Dict[str, torch.Tensor]:
        B = int(ctx.size(0))
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
        u = torch.tanh(drift + field + 0.10 * self.manifold_tensor_field_ema + parallel / metric_diag.clamp_min(self.epsA))
        z_next = torch.tanh(z + self.manifold_step_scale * u)
        corr_in = torch.cat([z, z_next, u, branch_tensor_mix, raw_delta, v_in], dim=-1)
        value_tensor_delta = torch.tanh(self.manifold_value_correction(corr_in))
        value_tensor = branch_tensor_mix + self.manifold_correction_scale * value_tensor_delta
        value_correction = value_tensor - branch_tensor_mix

        metric_reg = (metric_diag - 1.0).pow(2).mean()
        connection_norm = connection.pow(2).mean(dim=(1, 2)).sqrt()
        field_norm = field.pow(2).mean(dim=-1).sqrt()
        u_norm = u.pow(2).mean(dim=-1).sqrt()
        value_tensor_norm = value_tensor.pow(2).mean(dim=-1).sqrt()
        reg = ctx.new_tensor(1e-4) * (
            self.manifold_connection_basis.pow(2).mean()
            + self.manifold_tensor_basis.pow(2).mean()
            + connection_norm.pow(2).mean()
            + field_norm.pow(2).mean()
            + u_norm.pow(2).mean()
            + value_tensor_norm.pow(2).mean()
            + metric_reg)

        return {
            "z": z,
            "z_next": z_next,
            "u": u,
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
            "reg": reg,}

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
    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if self.anchor_value.size(0) == int(B):
            return
        self.anchor_value = torch.zeros(B, self.max_anchors, self.value_dim, device=device, dtype=dtype)
        self.anchor_value_next = torch.zeros(B, self.max_anchors, self.value_dim, device=device, dtype=dtype)
        self.anchor_z = torch.zeros(B, self.max_anchors, self.z_dim, device=device, dtype=dtype)
        self.filled = torch.zeros(B, device=device, dtype=torch.long)
        self.ptr = torch.zeros(B, device=device, dtype=torch.long)
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
        self.EnsureB(B, value.device, value.dtype)

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
        alive: torch.Tensor,):
        B = int(value.size(0))
        self.EnsureB(B, value.device, value.dtype)
        live = alive.view(B) > 0.5
        dead = ~live
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
        moodDecay: float = 0.95,
        useHebbHead: bool = True):
        super().__init__()

        self.stateDim = stateDim
        self.memoryDim = memoryDim
        self.attnDim = attnDim
        self.emotionDim = emotionDim
        self.fastHidden = fastHidden
        self.slowHidden = slowHidden
        self.moodDecay = float(moodDecay)
        self.useHebbHead = useHebbHead

        fast_in_dim = stateDim + attnDim
        
        self.fast_net = nn.Sequential(
            nn.Linear(fast_in_dim, fastHidden),
            nn.SiLU(),
            nn.Linear(fastHidden, fastHidden),
            nn.SiLU(),)
        
        self.fast_head = HebbianLinearFW(inFeatures=fastHidden,outFeatures=emotionDim, bias=True,useHebbian=useHebbHead, useOja=True,)

        slow_in_dim = stateDim + memoryDim + attnDim
        self.slow_cell = nn.LSTMCell(input_size=slow_in_dim, hidden_size=slowHidden)

        self.slow_head = HebbianLinearFW(inFeatures=slowHidden,outFeatures=emotionDim, bias=True,useHebbian=useHebbHead, useOja=True,)

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


    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if self.h.size(0) != B:
            self.h = torch.zeros(B, self.slowHidden, device=device, dtype=dtype)
            self.c = torch.zeros(B, self.slowHidden, device=device, dtype=dtype)
            self.mood = torch.zeros(B, self.emotionDim, device=device, dtype=dtype)

    def forward(
        self,
        memoryPrev: torch.Tensor, 
        attnPrev: torch.Tensor, 
        stateCurr: torch.Tensor,) -> torch.Tensor:

        B = stateCurr.size(0)
        self.EnsureB(B, self.device, self.dtype)

        h_prev = self.h # [B,H]
        c_prev = self.c # [B,H]
        mood_prev = self.mood # [B,E]

        fast_in = torch.cat([stateCurr, attnPrev], dim=-1)
        fast_h = self.fast_net(fast_in) # [B, F]

        fast_raw, _ = self.fast_head(fast_h)

        emotion_fast = torch.tanh(fast_raw)

        slow_in = torch.cat([stateCurr, memoryPrev, attnPrev], dim=-1) 
        h_t, c_t = self.slow_cell(slow_in, (h_prev, c_prev)) # h_t: [B,H], c_t: [B,H]

        slow_raw, _ = self.slow_head(h_t) # [B,E]

        emotion_slow = torch.tanh(slow_raw) # [B,E]

        decay = self.moodDecay
        mood_t = decay * mood_prev + (1.0 - decay) * emotion_slow # [B,E]

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

        gate_in = torch.cat([memoryPrev, attnPrev, stateCurr], dim=-1)  
        gate = self.gate_net(gate_in) 

        gate = gate.clamp(0.0, 1.0) # [B,E] 
        emotion = torch.tanh(emotion_raw * (1.0 + gate)) # [B,E]

        self.h = h_t.detach()
        self.c = c_t.detach()
        self.mood = mood_t.detach()

        return emotion

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        if self.useHebbHead:
            self.slow_head.ResetHebbianMemory(doneMask=doneMask)
            self.fast_head.ResetHebbianMemory(doneMask=doneMask)


class GeoTropicalOut(NamedTuple):
    value: torch.Tensor
    valueNext: torch.Tensor
    tdError: torch.Tensor
    loss: torch.Tensor
    emotion: torch.Tensor
    rComps: Dict[str, torch.Tensor]
    uncertainty: torch.Tensor
    precision: torch.Tensor
    extras: Dict[str, torch.Tensor]


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
    def __init__(self,
        memoryDim: int = 768, attnDim: int = 1024, stateDim: int = 256, *,
        emotionDim: int = 64,
        hidden: int = 2048,
        trunkResBlocks: int = 4,
        trunkResHiddenMul: float = 2.0,
        trunkResScaleInit: float = 0.25,
        useSoftTrop: bool = True, tropTemp: float = 0.2, epsA: float = 1e-3,
        useHebb: bool = True,
        valueTensorDim: int = 512,
        microMaxAnchors: int = 64,
        microTopK: int = 4,
        microDistTau: float = 1.0,
        microValueDistScale: float = 0.25,
        microAgeScale: float = 1e-3,
        tdGeomRank: int = 16, # rank of low-rank Laplacian factor (Sobolev term)
        tdHeatRank: int = 32,
        tdBuresSlots: int = 16,
        tdOtBins: int = 64,
        tdOtCostDim: int = 8,
        tdOtIters: int = 8,
        tdHuberKappa: float = 1.0, # Huber threshold for elementwise residual
        tdGeomSignTau: float = 1.0, # τ in sign σ_b = tanh(td_anchor / τ)
        tdGeomSobAlpha: float = 0.10, # α coupling for Sobolev term  ||Δ||² + α(Δᵀ L Δ)
        tdGeomEps: float = 1e-6,):
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

        self.use_hebb = useHebb

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

        self.reward_perdetic = KalmanFilteredEnsembleNext()
        self.done_perdetic = KalmanFilteredEnsembleNext()

        self.fc1 = nn.Linear(self.in_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)
        self.trunk_res_blocks = nn.ModuleList([
            ResidualMLPBlock(dim=H, hiddenMul=trunkResHiddenMul, scaleInit=trunkResScaleInit)
            for _ in range(max(0, int(trunkResBlocks)))])

        self.quantile_head = nn.Linear(H, self.num_quantiles)
        self.value_ensemble_heads = nn.ModuleList([nn.Linear(H, 1) for _ in range(4)])

        self.value_tensor_tail = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, self.value_tensor_dim),
            nn.GELU(),
            nn.Linear(self.value_tensor_dim, self.value_tensor_dim),)

        self.value_tensor_out_norm = nn.LayerNorm(self.value_tensor_dim)
        self.value_tensor_log_scale = nn.Parameter(torch.tensor(math.log(math.exp(0.5) - 1.0)))

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
        self.emotion_core = EmotionCore(stateDim=stateDim, memoryDim=memoryDim,attnDim=attnDim, emotionDim=emotionDim)
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
        nn.init.zeros_(self.td_context_head[-1].weight)
        nn.init.zeros_(self.td_context_head[-1].bias)
        nn.init.normal_(self.td_heat_basis, mean=0.0, std=1.0 / math.sqrt(float(self.value_tensor_dim)))
        nn.init.normal_(self.td_ot_cost_embed, mean=0.0, std=1.0 / math.sqrt(float(self.td_ot_cost_dim)))
        self.emotion_core.ResetParams()

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        self.td_out_ema.EnsureB(B, device, dtype)
        self.micro.EnsureB(B, device, dtype)

        if self._last_batch_size is None:
            self._last_batch_size = int(B)
        elif int(self._last_batch_size) != int(B):
            self._pending_transitions.clear()
            self.ClearTransportGradAccumulator()
            self._last_batch_size = int(B)

    def Trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1_adapter(x))
        h = self.norm1(h)
        h = F.gelu(self.fc2_adapter(h))
        h = self.norm2(h)
        for blk in self.trunk_res_blocks:
            h = blk(h)
        return h

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
        scale = (q_mean.detach().abs() + 1.0).clamp_min(1e-6) # [B]
        dist_risk = (1.0 - torch.exp(-(q_std + downside) / scale)).clamp(0.0, 1.0) # [B]
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

    def BuildTDGraph(self, tdCurrent: torch.Tensor) -> Dict[str, torch.Tensor]:
        td_abs = tdCurrent.detach().abs().squeeze(-1)
        self.td_out_ema.Update(td_abs)
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

    def SymmetricMatrixSqrt(self, x: torch.Tensor) -> torch.Tensor:
        x_sym = 0.5 * (x + x.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(x_sym)
        eigvals = eigvals.clamp_min(self.td_geom_eps).sqrt()
        return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)

    def BuresWassersteinEnergy(self, valueTensor: torch.Tensor, valueNextTensor: torch.Tensor) -> torch.Tensor:
        B = valueTensor.size(0)
        eps = self.td_geom_eps
        x = valueTensor.view(B, self.td_bures_slots, self.td_bures_dim)
        y = valueNextTensor.view(B, self.td_bures_slots, self.td_bures_dim)
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        cov_x = x.transpose(1, 2).matmul(x) / float(max(1, self.td_bures_slots - 1))
        cov_y = y.transpose(1, 2).matmul(y) / float(max(1, self.td_bures_slots - 1))
        eye = torch.eye(self.td_bures_dim, device=valueTensor.device, dtype=valueTensor.dtype).unsqueeze(0)
        cov_x = cov_x + eps * eye
        cov_y = cov_y + eps * eye
        sqrt_x = self.SymmetricMatrixSqrt(cov_x)
        middle = sqrt_x.matmul(cov_y).matmul(sqrt_x)
        sqrt_middle = self.SymmetricMatrixSqrt(middle)
        return (cov_x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                + cov_y.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
                - 2.0 * sqrt_middle.diagonal(dim1=-2, dim2=-1).sum(dim=-1)).clamp_min(0.0)

    def HeatKernelEnergy(self, delta: torch.Tensor) -> torch.Tensor:
        basis = F.normalize(self.td_heat_basis, dim=-1)
        coeff = F.linear(delta, basis)
        eig = F.softplus(self.td_heat_log_eigs).clamp_min(self.td_geom_eps)
        heat = (1.0 - torch.exp(-eig)).pow(2)
        return (coeff.pow(2) * heat.unsqueeze(0)).sum(dim=-1)

    def SinkhornOTEnergy(self, valueTensor: torch.Tensor, valueNextTensor: torch.Tensor) -> torch.Tensor:
        B = valueTensor.size(0)
        eps = self.td_geom_eps
        x = valueTensor.view(B, self.td_ot_bins, self.td_ot_bin_width).mean(dim=-1)
        y = valueNextTensor.view(B, self.td_ot_bins, self.td_ot_bin_width).mean(dim=-1)
        p = F.softplus(x).clamp_min(eps)
        q = F.softplus(y).clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
        emb = F.normalize(self.td_ot_cost_embed, dim=-1)
        cost = torch.cdist(emb, emb, p=2).pow(2)
        cost = cost / cost.mean().clamp_min(eps)
        kernel = torch.exp(-cost / 0.10).clamp_min(eps)
        u = torch.ones_like(p)
        v = torch.ones_like(q)
        for _ in range(self.td_ot_iters):
            u = p / torch.matmul(v, kernel.transpose(0, 1)).clamp_min(eps)
            v = q / torch.matmul(u, kernel).clamp_min(eps)
        plan_cost = u.unsqueeze(2) * kernel.unsqueeze(0) * v.unsqueeze(1) * cost.unsqueeze(0)
        return plan_cost.sum(dim=(1, 2))

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

        td_bures = self.BuresWassersteinEnergy(valueTensor, valueNextTensor).clamp_min(eps).sqrt()
        td_heat = self.HeatKernelEnergy(delta).clamp_min(eps).sqrt()
        td_ot = self.SinkhornOTEnergy(valueTensor, valueNextTensor).clamp_min(eps).sqrt()

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

    def BranchStructureLoss(self, branchNext: torch.Tensor, branchWeight: torch.Tensor) -> torch.Tensor:
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
        return (uncertainty * diversity + certainty * consistency).mean()

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
        hookSeen: set) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        params = {}
        for name, p in self.transport.named_parameters():
            snap = p.detach().clone().requires_grad_(p.requires_grad)
            if snap.requires_grad:
                def snapshot_hook(
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
                snap.register_hook(snapshot_hook)
            params[name] = snap
        buffers = {name: b for name, b in self.transport.named_buffers()}
        transport_state = {}
        transport_state.update(params)
        transport_state.update(buffers)
        return torch_functional_call(
            self.transport,
            transport_state,
            (h, value),
            {"returnExtras": True})

    def InstallTransportGradHooks(self):
        for handle in self._transport_grad_hooks:
            handle.remove()
        self._transport_grad_hooks = []
        for name, p in self.transport.named_parameters():
            if p.requires_grad:
                def main_hook(
                    grad: torch.Tensor,
                    *,
                    param_name: str = name,) -> torch.Tensor:
                    self.AddTransportGrad(self._transport_curr_grad, param_name, grad)
                    self._transport_curr_grad_hook_seen.add(param_name)
                    return torch.zeros_like(grad)
                self._transport_grad_hooks.append(p.register_hook(main_hook))

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
        shadow_params: Dict[str, torch.Tensor] = {}
        for name, p in self.transport.named_parameters():
            g = grad_by_name.get(name)
            if g is None:
                continue
            step = g.mul(scale)
            if wd_f != 0.0:
                step = step.add(p.detach(), alpha=wd_f)
            shadow_params[name] = p.detach().clone().add(step, alpha=-lr_f)

        for name, p in self.transport.named_parameters():
            shadow = shadow_params.get(name)
            if shadow is None:
                continue
            p.copy_(shadow)
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
            return [int(i) for i in range(int(B))]
        if torch.is_tensor(streamIds):
            vals = streamIds.view(-1).tolist()
        else:
            vals = list(streamIds)
        return [int(v) for v in vals]

    def SelectLiveRow(self, x: Any, row: int, B: int) -> Any:
        if torch.is_tensor(x):
            if x.dim() > 0 and int(x.size(0)) == int(B):
                return x[row:row + 1]
            return x
        if isinstance(x, dict):
            return {k: self.SelectLiveRow(v, row, B) for k, v in x.items()}
        return x

    def StackLiveRows(self, items: List[Any]) -> Any:
        first = items[0]
        if torch.is_tensor(first):
            if first.dim() == 0:
                return torch.stack(items, dim=0).mean()
            return torch.cat(items, dim=0)
        if isinstance(first, dict):
            return {k: self.StackLiveRows([it[k] for it in items if k in it]) for k in first.keys()}
        return first

    def CacheDelayedTransitionInputs(
        self,
        vNextHat: torch.Tensor,
        alive: torch.Tensor,
        transpExtras: Dict[str, Any],
        streamIds: Optional[torch.Tensor] = None):
        B = int(vNextHat.size(0))
        sids = self.NormalizeStreamIds(B, streamIds)
        alive_f = alive.detach().view(B).clamp(0.0, 1.0)

        for i, sid in enumerate(sids):
            item = {
                "pred_live": vNextHat[i:i + 1],
                "alive": alive_f[i],
                "transp_extras_live": self.SelectLiveRow(transpExtras, i, B),}
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
        branchSpread: torch.Tensor,) -> Dict[str, torch.Tensor]:
        base = float(self.unc_core.eps_prior)
        unc_adj = (uncTotal.view(-1) - base).clamp_min(0.0)
        unc_prior01 = (1.0 - torch.exp(-unc_adj / max(self.unc_tau, 1e-6))).clamp(0.0, 1.0)
        unc_prior_evidence = unc_prior01.detach()

        unc_epistemic01 = (1.0 - torch.exp(-valueEpistemic.detach())).clamp(0.0, 1.0)
        risk_dist = distStats["risk"].detach()
        td_abs_train = tdBounded.abs()
        td_abs = td_abs_train.detach()
        risk_base = (1.0 - torch.exp(-(td_abs + unc_prior_evidence))).clamp(0.0, 1.0)
        ambiguity = unc_prior_evidence.clamp(0.0, 1.0)
        surprise = (1.0 - torch.exp(-td_abs)).clamp(0.0, 1.0)
        confidence_base = torch.exp(-(
            0.50 * risk_base
            + 0.35 * ambiguity
            + 0.15 * surprise)).clamp(0.0, 1.0)
        physical_raw = (
            0.28 * physicalTd["td_dirichlet"]
            + 0.24 * physicalTd["td_sobolev"]
            + 0.18 * physicalTd["td_context"]
            + 0.12 * physicalTd["td_bures"]
            + 0.10 * physicalTd["td_heat"]
            + 0.08 * physicalTd["td_ot"])
        physical01_train = (1.0 - torch.exp(-physical_raw)).clamp(0.0, 1.0)
        physical01 = physical01_train.detach()
        branch01 = (1.0 - torch.exp(-branchSpread.detach())).clamp(0.0, 1.0)
        risk = (
            0.24 * risk_base
            + 0.18 * physical01
            + 0.16 * risk_dist
            + 0.14 * unc_prior_evidence
            + 0.12 * unc_epistemic01
            + 0.08 * branch01).clamp(0.0, 1.0)
        confidence_dist = torch.exp(-(unc_prior_evidence + unc_epistemic01 + risk)).clamp(0.0, 1.0)
        confidence = (0.5 * confidence_base + 0.5 * confidence_dist).clamp(0.0, 1.0)
        learned_unc = (
            0.30 * risk
            + 0.25 * ambiguity
            + 0.20 * surprise
            + 0.25 * (1.0 - confidence)).clamp(0.0, 1.0)
        unc01 = (
            0.35 * unc_prior_evidence
            + 0.25 * learned_unc
            + 0.20 * unc_epistemic01
            + 0.10 * risk).clamp(0.0, 1.0)
        precision = (confidence * (1.0 - unc01)).clamp(0.05, 1.0)
        unc_pred = (0.25 * (risk + ambiguity + surprise + (1.0 - confidence))).clamp(0.0, 1.0)
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
            "unc_prior01": unc_prior_evidence,
            "unc_epistemic01": unc_epistemic01,}

    def ConsumePendingTransitions(
        self,
        B: int,
        valueLabel: torch.Tensor,
        zLabel: torch.Tensor,
        streamIds: Optional[torch.Tensor]) -> Dict[str, Any]:
        target_empty = valueLabel.new_zeros((B,) + tuple(valueLabel.shape[1:]))
        prev = {
            "ready": False,
            "pred": target_empty,
            "transp_extras": {},
            "loss_mask": valueLabel.new_zeros((0,)),}

        sids = self.NormalizeStreamIds(B, streamIds)
        rows: List[int] = []
        items: List[Dict[str, Any]] = []
        for row, sid in enumerate(sids):
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
        alive = torch.stack([it["alive"].view(()) for it in items], dim=0).clamp(0.0, 1.0)

        target = valueLabel.index_select(0, rows_t)
        z_target = zLabel.index_select(0, rows_t)
        pred = torch.cat([it["pred_live"] for it in items], dim=0)
        transp_extras = self.StackLiveRows([it["transp_extras_live"] for it in items])
        manifold = transp_extras["manifold"]

        prev.update({
            "ready": True,
            "pred": pred,
            "transp_extras": transp_extras,
            "target_m": target,
            "loss_mask": alive,
            "z": manifold["z"],
            "z_next": manifold["z_next"],
            "u": manifold["u"],
            "z_target": z_target,
            "manifold_reg": manifold["reg"],})
        return prev

    def forward(self,
        memoryPrev: torch.Tensor, # [B,memDim]
        attnPrev: torch.Tensor, # [B,attnDim]
        state: torch.Tensor, # [B,stateDim]
        *,
        policyEntropyPrev: Optional[torch.Tensor] = None, # [B]
        worldDeltaTransport: Optional[torch.Tensor] = None, # [B,stateDim]
        worldDeltaPhysics: Optional[torch.Tensor] = None, # [B,stateDim]
        rewardModel: Optional[torch.Tensor] = None,
        doneModel: Optional[torch.Tensor] = None,
        streamIds: Optional[torch.Tensor] = None,
        )-> GeoTropicalOut: 

        B = state.size(0)
        self.EnsureB(B, self.device, self.dtype)

        if policyEntropyPrev is None:
            policyEntropyPrev = state.new_zeros(B)
        if worldDeltaTransport is None:
            worldDeltaTransport = state.new_zeros(B, self.unc_core.state_dim)
        if worldDeltaPhysics is None:
            worldDeltaPhysics = state.new_zeros(B, self.unc_core.state_dim)

        reward_value = rewardModel
        done_value = doneModel
        with torch.no_grad():
            r_next_hat = self.reward_perdetic.PredictNext(reward_value.detach()).view(B)
            done_next_hat = self.done_perdetic.PredictNext(done_value.detach()).view(B).clamp(0.0, 1.0)

        x = torch.cat([memoryPrev, attnPrev, state], dim=-1)
        h = self.Trunk(x) # [B,H]

        emotion = self.emotion_core(memoryPrev=memoryPrev, attnPrev=attnPrev, stateCurr=state) # [B,emotionDim]
        h = self.FuseEmotionIntoHidden(h, emotion)

        value_parts = self.BuildValueGraph(h=h)
        value = value_parts["value"]
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
        td_graph = self.BuildTDGraph(td_current)
        td_bounded = td_graph["td_bounded"]  # [-1,1], [B]

        unc_total, _ = self.unc_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state,
            entropyPrev=policyEntropyPrev,
            tdCurr=td_bounded.detach(),
            worldDeltaTransport=worldDeltaTransport,
            worldDeltaPhysics=worldDeltaPhysics,
            donePrev=done_next_hat.detach()) # unc_total:[B]
        
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
            branchSpread=transport_branch_std)
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
            "confidence": confidence.detach(),}

        self.micro.CommitStep(
            value=transport_value,
            valueNext=value_next,
            z=manifold_out["z"].detach(),
            alive=(1.0 - done_value).clamp(0.0, 1.0))

        if not self.training:
            return GeoTropicalOut(
                value=value,
                valueNext=value_next,
                tdError=td_bounded.detach(),
                loss=None,
                emotion=emotion,
                rComps=rComps,
                uncertainty=unc01,
                precision=precision,
                extras=None,)

        prev = self.ConsumePendingTransitions(
            B=B,
            valueLabel=value.detach(),
            zLabel=manifold_out["z"].detach(),
            streamIds=streamIds)
        
        has_prev_pred = bool(prev["ready"])

        loss_diff = value.new_zeros(())
        loss_diff_branch = value.new_zeros(())
        loss_branch_structure = value.new_zeros(())
        loss_manifold_geo = value.new_zeros(())
        loss_manifold_tangent = value.new_zeros(())
        loss_manifold_latent = value.new_zeros(())
        loss_manifold_reg = value.new_zeros(())
        loss_mask = prev["loss_mask"]
        valid_denom = loss_mask.sum().clamp_min(1.0) if loss_mask.numel() > 0 else value.new_tensor(1.0)

        def ValidMeanM(vec: torch.Tensor) -> torch.Tensor:
            if loss_mask.numel() <= 0:
                return value.new_zeros(())
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
            loss_branch_structure = self.BranchStructureLoss(branch_next, branch_w)
                
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

            loss_manifold_reg = prev["manifold_reg"]

        loss_transport = (
            value.new_tensor(self.wDiff) * loss_diff
            + value.new_tensor(self.wDiffBranch) * loss_diff_branch
            + value.new_tensor(self.wBranchStructure) * loss_branch_structure
            + value.new_tensor(self.wManifoldLatent) * loss_manifold_latent
            + loss_manifold_reg)
        loss_transport_delayed = loss_transport

        loss_physical_td = F.smooth_l1_loss(td_bounded, td_bounded.new_zeros(td_bounded.shape))

        loss_physical_aux = (
            value.new_tensor(0.02) * physical_td["td_bures"].mean()
            + value.new_tensor(0.01) * physical_td["td_heat"].mean()
            + value.new_tensor(0.005) * physical_td["td_ot"].mean())
        
        physical_param_reg = self.BuildPhysicalTDParameterRegularizer()
        loss_physical_param_reg = value.new_tensor(self.wPhysicalTDParamReg) * physical_param_reg["loss"]
        
        loss_value_tensor_energy = value.new_tensor(1e-6) * value.pow(2).mean()
        
        quantile_target = (
            td_bounded.detach().view(-1, 1)
            + (self.quantile_tau.view(1, -1) - 0.5)
            * 2.0
            * physical_td["td_mag"].detach().view(-1, 1)).clamp(-3.0, 3.0)
        loss_quantile_fit = self.QuantileHuberLoss(
            value_parts["value_quantiles"],
            quantile_target)
        loss_quantile_order = self.QuantileCrossingLoss(value_parts["value_quantiles"])
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
            + 0.10 * physical_td["td_ot"].detach()).clamp(0.0, 3.0)
        loss_ensemble = (
            value.new_tensor(0.01) * F.smooth_l1_loss(ensemble_mean, ensemble_mean_target)
            + value.new_tensor(0.005) * F.smooth_l1_loss(ensemble_var, ensemble_var_target))
        
        pending_v_next_hat, pending_transp_extras = self.BuildTransportSnapshotGraph(
            transport_h,
            transport_value,
            self._transport_prev_grad,
            self._transport_prev_grad_hook_seen)

        pending_v_next_hat, pending_transp_extras = self.ApplyRewardNextModulation(
            transport_value,
            pending_v_next_hat,
            pending_transp_extras,
            r_next_hat)
        
        self.CacheDelayedTransitionInputs(
            vNextHat=pending_v_next_hat,
            alive=(1.0 - done_value),
            transpExtras=pending_transp_extras,
            streamIds=streamIds)

        loss_current = loss_physical_td + loss_physical_aux + loss_physical_param_reg + loss_value_tensor_energy + loss_quantile + loss_ensemble

        extras = {
            "loss_transport": loss_transport_delayed.detach(),
            "loss_physical_param_reg": loss_physical_param_reg.detach(),
            "loss_quantile": loss_quantile.detach(),
            "loss_ensemble": loss_ensemble.detach(),
            "loss_diff": loss_diff.detach(),
            "loss_diff_branch": loss_diff_branch.detach(),
            "loss_branch_structure": loss_branch_structure.detach(),
            "loss_manifold_geo": loss_manifold_geo.detach(),
            "loss_manifold_tangent": loss_manifold_tangent.detach(),
            "loss_manifold_latent": loss_manifold_latent.detach(),
            "td_mag": physical_td["td_mag"].detach(),
            "loss_current_graph": loss_current,
            "loss_transport_delayed_graph": loss_transport_delayed,}

        return GeoTropicalOut(
            value=value,
            valueNext=value_next,
            tdError=td_bounded.detach(),
            loss=loss_current,
            emotion=emotion,
            rComps=rComps,
            uncertainty=unc01,
            precision=precision,
            extras=extras,)


    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        self.emotion_core.ResetHebbianMemory(doneMask=doneMask)
        if doneMask is not None:
            self.ResetState(doneMask=doneMask)

    @torch.no_grad()
    def ResetState(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.micro.Reset()
            self.emotion_core.ResetState()
            self.td_out_ema.ResetAll()
            self.unc_core.ResetState()
            self.reward_perdetic.Reset()
            self.done_perdetic.Reset()
            self._pending_transitions.clear()
            self._last_batch_size = None
            self.ClearTransportGradAccumulator()
            return
        self.micro.Reset(doneMask=doneMask)
        self.emotion_core.ResetState(doneMask=doneMask)
        self.td_out_ema.ResetAll(doneMask=doneMask)
        self.unc_core.ResetState(doneMask=doneMask)
        self.reward_perdetic.Reset(doneMask=doneMask)
        self.done_perdetic.Reset(doneMask=doneMask)
        self.ResetPrevTransition(doneMask)

    @torch.no_grad()
    def ResetPrevTransition(self, doneMask: torch.Tensor):
        mask = doneMask.bool().view(-1)
        rows = mask.nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        for r in rows.detach().tolist():
            self._pending_transitions.pop(int(r), None)

    @torch.no_grad()
    def ExportState(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"ve_is_training": bool(self.training)}
        state["ve_last_batch_size"] = self._last_batch_size

        state["td_out_ema_mean"] = self.td_out_ema.mean.detach().clone()
        state["td_out_ema_var"] = self.td_out_ema.var.detach().clone()

        ec = self.emotion_core
        if ec.h is not None:
            state["emo_h"] = ec.h.detach().clone()
        if ec.c is not None:
            state["emo_c"] = ec.c.detach().clone()
        if ec.mood is not None:
            state["emo_mood"] = ec.mood.detach().clone()
        state["emo_fast_H"] = ec.fast_head.H.detach().clone()
        state["emo_slow_H"] = ec.slow_head.H.detach().clone()

        uc = self.unc_core
        for name in ["td_ema", "ent_ema", "state_ema", "tr_ema", "ph_ema", "ctx_ema"]:
            ema = getattr(uc, name)
            state[f"unc_{name}_mean"] = ema.mean.detach().clone()
            state[f"unc_{name}_var"] = ema.var.detach().clone()

        for prefix, pred in [("reward_pred", self.reward_perdetic),
                             ("done_pred", self.done_perdetic)]:
            for n in ["kf_mean", "kf_var", "smooth_hist"]:
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

        return state

    @torch.no_grad()
    def ImportState(self, state: Dict[str, Any],):
        def need_(key: str) -> Any:
            return state[key] if key in state else None

        if "ve_is_training" in state:
            self.train(bool(state["ve_is_training"]))

        def copy_tensor_attr(obj: Any, name: str, t: Optional[torch.Tensor]):
            if t is None:
                return
            cur = getattr(obj, name)
            if not torch.is_tensor(cur):
                setattr(obj, name, t)
                return
            if cur.shape != t.shape:
                cur.resize_(t.shape).copy_(t)
            else:
                cur.copy_(t)

        self._last_batch_size = state.get("ve_last_batch_size", getattr(self, "_last_batch_size", None))
        self._pending_transitions.clear()
        self.ClearTransportGradAccumulator()

        copy_tensor_attr(self.td_out_ema, "mean", need_("td_out_ema_mean"))
        copy_tensor_attr(self.td_out_ema, "var", need_("td_out_ema_var"))

        ec = self.emotion_core
        if need_("emo_h") is not None:
            ec.h = need_("emo_h")
        if need_("emo_c") is not None:
            ec.c = need_("emo_c")
        if need_("emo_mood") is not None:
            ec.mood = need_("emo_mood")
        copy_tensor_attr(ec.fast_head, "H", need_("emo_fast_H"))
        copy_tensor_attr(ec.slow_head, "H", need_("emo_slow_H"))

        uc = self.unc_core
        for name in ["td_ema", "ent_ema", "state_ema", "tr_ema", "ph_ema", "ctx_ema"]:
            ema = getattr(uc, name)
            copy_tensor_attr(ema, "mean", need_(f"unc_{name}_mean"))
            copy_tensor_attr(ema, "var", need_(f"unc_{name}_var"))

        for prefix, pred in [("reward_pred", self.reward_perdetic),
                             ("done_pred", self.done_perdetic)]:
            for n in ["kf_mean", "kf_var", "smooth_hist"]:
                copy_tensor_attr(pred, n, need_(f"{prefix}_{n}"))
            for n in ["predict_mode", "auto_policy", "auto_temperature", "fit_last_n"]:
                k = f"{prefix}_{n}"
                if k in state:
                    setattr(pred, n, state[k])

        mg = self.micro
        for n in ["anchor_value", "anchor_value_next", "anchor_z", "filled", "ptr"]:
            copy_tensor_attr(mg, n, need_(f"micro_{n}"))
        if "micro_step" in state:
            mg._step = int(state["micro_step"])
        transport_field_ema = need_("transport_manifold_tensor_field_ema")
        copy_tensor_attr(self.transport, "manifold_tensor_field_ema", transport_field_ema)


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

    @staticmethod
    def LinearWithDelta(
        layer: nn.Linear,
        x: torch.Tensor,
        delta_mat: Optional[torch.Tensor] = None,
        base_adapter: Optional[nn.Module] = None,) -> torch.Tensor:
        W_eff = layer.weight
        if base_adapter is not None:
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
        L = 2 

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

        rewardModel = kwargs.get("rewardModel", None)
        doneModel = kwargs.get("doneModel", None)
        gamma = kwargs.get("gamma", 0.99)
        streamIds = kwargs.get("streamIds", None)
        policyEntropyPrev = kwargs.get("policyEntropyPrev", None)
        worldDeltaTransport = kwargs.get("worldDeltaTransport", None)
        worldDeltaPhysics = kwargs.get("worldDeltaPhysics", None)

        memoryPrev, attnPrev, state = self.EnsureInputs(x)
        B = state.size(0)
        base.EnsureB(B, base.device, base.dtype)

        if deltasPerLayer is None:
            deltasPerLayer = [{}, {}]
        else:
            if len(deltasPerLayer) < 2:
                deltasPerLayer = list(deltasPerLayer) + [{} for _ in range(2 - len(deltasPerLayer))]

        d0 = deltasPerLayer[0] or {}
        d1 = deltasPerLayer[1] or {}

        def BuildOnlineParametricValueBranches(h_in: torch.Tensor) -> Dict[str, torch.Tensor]:
            value_quantiles_in = self.LinearWithDelta(
                base.quantile_head,
                h_in,
                delta_mat=d1.get("qhead", None),
                base_adapter=getattr(base, "quantile_adapter", None),)
            dist_stats_in = base.DistributionStats(value_quantiles_in)
            value_ensemble_in = torch.cat([head(h_in) for head in base.value_ensemble_heads], dim=-1)
            value_epistemic_in = value_ensemble_in.var(dim=-1, unbiased=False)
            return {
                "value_quantiles": value_quantiles_in,
                "dist_stats": dist_stats_in,
                "value_ensemble": value_ensemble_in,
                "value_epistemic": value_epistemic_in,}

        x_cat = torch.cat([memoryPrev, attnPrev, state], dim=-1)

        h = self.LinearWithDelta(
            base.fc1,
            x_cat,
            delta_mat=d0.get("fc1", None),
            base_adapter=getattr(base, "fc1_adapter", None),)
        
        h = base.norm1(F.gelu(h))

        h = self.LinearWithDelta(
            base.fc2,
            h,
            delta_mat=d1.get("fc2", None),
            base_adapter=getattr(base, "fc2_adapter", None),)
        
        h = base.norm2(F.gelu(h))
        for blk in base.trunk_res_blocks:
            h = blk(h)

        emotion = base.emotion_core(memoryPrev=memoryPrev, attnPrev=attnPrev, stateCurr=state)
        h = base.FuseEmotionIntoHidden(h, emotion)

        value_parts = BuildOnlineParametricValueBranches(h)
        value_quantiles = value_parts["value_quantiles"]
        dist_stats = value_parts["dist_stats"]
        value_ensemble = value_parts["value_ensemble"]
        value_epistemic = value_parts["value_epistemic"]

        reward_value = rewardModel.detach().view(B)
        done_value = doneModel.detach().view(B).clamp(0.0, 1.0)
        with torch.no_grad():
            r_next_hat = base.reward_perdetic.PredictNext(reward_value.detach()).view(B)
            done_next_hat = base.done_perdetic.PredictNext(done_value.detach()).view(B).clamp(0.0, 1.0)

        value = base.BuildValueTensor(h)

        transport_h = h.detach()
        transport_value = value.detach()

        value_next, transp_extras = base.transport(
            transport_h,
            transport_value,
            returnExtras=True)
        value_next, transp_extras = base.ApplyRewardNextModulation(
            transport_value,
            value_next,
            transp_extras,
            r_next_hat)
        manifold_out = transp_extras["manifold"]
        micro_graph = base.micro.Preview(
            value=transport_value,
            z=manifold_out["z"].detach())
        value_next, transp_extras = base.ApplyMicroGraphPrior(
            value_next,
            transp_extras,
            micro_graph)
        physical_td = base.BuildPhysicalTD(transport_value, value_next, transport_h)
        td_current = physical_td["td_scalar_train"].view(B, 1)
        td_graph = base.BuildTDGraph(td_current)
        td_bounded = td_graph["td_bounded"]

        unc_total, _ = base.unc_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state,
            entropyPrev=policyEntropyPrev,
            tdCurr=td_bounded.detach(),
            worldDeltaTransport=worldDeltaTransport,
            worldDeltaPhysics=worldDeltaPhysics,
            donePrev=done_next_hat.detach())
        
        branch_next = transp_extras.get("branch_next")
        transport_branch_std = (
            value.new_zeros((B,))
            if branch_next is None
            else branch_next.detach().view(B, -1).std(dim=-1, unbiased=False))

        uncertainty_graph = base.BuildUncertaintyGraph(
            distStats=dist_stats,
            valueEpistemic=value_epistemic,
            tdBounded=td_bounded,
            uncTotal=unc_total,
            physicalTd=physical_td,
            branchSpread=transport_branch_std)
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
            "confidence": confidence.detach(),}

        base.micro.CommitStep(
            value=transport_value,
            valueNext=value_next,
            z=manifold_out["z"].detach(),
            alive=(1.0 - done_value).clamp(0.0, 1.0))

        if not self.training:
            return GeoTropicalOut(
                value=value,
                valueNext=value_next,
                tdError=td_bounded.detach(),
                loss=None,
                emotion=emotion,
                rComps=rComps,
                uncertainty=unc01,
                precision=precision,
                extras=None,)
        
        prev = base.ConsumePendingTransitions(
            B=B,
            valueLabel=value.detach(),
            zLabel=manifold_out["z"].detach(),
            streamIds=streamIds)
        has_prev_pred = bool(prev["ready"])

        loss_diff = value.new_zeros(())
        loss_diff_branch = value.new_zeros(())
        loss_branch_structure = value.new_zeros(())
        loss_manifold_geo = value.new_zeros(())
        loss_manifold_tangent = value.new_zeros(())
        loss_manifold_latent = value.new_zeros(())
        loss_manifold_reg = value.new_zeros(())
        loss_mask = prev["loss_mask"]
        valid_denom = loss_mask.sum().clamp_min(1.0) if loss_mask.numel() > 0 else value.new_tensor(1.0)

        def ValidMeanM(vec: torch.Tensor) -> torch.Tensor:
            if loss_mask.numel() <= 0:
                return value.new_zeros(())
            return (vec.view(-1) * loss_mask).sum() / valid_denom

        if has_prev_pred:
            target_m = prev["target_m"].detach()

            prev_pred = prev["pred"]
            prev_transp_extras = prev["transp_extras"]
            loss_diff_vec = F.smooth_l1_loss(prev_pred, target_m, reduction="none").view(prev_pred.size(0), -1).mean(dim=-1)
            loss_diff = ValidMeanM(loss_diff_vec)
            branch_next = prev_transp_extras["branch_next"]
            branch_w = prev_transp_extras["branch_w"]
            branch_mix = (branch_w.unsqueeze(-1) * branch_next).sum(dim=1)
            branch_loss_vec = F.smooth_l1_loss(
                branch_mix,
                target_m,
                reduction="none").view(branch_mix.size(0), -1).mean(dim=-1)
            loss_diff_branch = ValidMeanM(branch_loss_vec)
            loss_branch_structure = base.BranchStructureLoss(branch_next, branch_w)
            loss_manifold_latent_vec = F.smooth_l1_loss(
                prev["z_next"],
                prev["z_target"].detach(),
                reduction="none").mean(dim=-1)
            loss_manifold_geo = ValidMeanM(loss_manifold_latent_vec)
            u_target = base.ManifoldLocalLog(
                prev["z"].detach(),
                prev["z_target"].detach(),
                stepScale=base.transport.manifold_step_scale)
            loss_manifold_tangent_vec = F.smooth_l1_loss(
                prev["u"],
                u_target,
                reduction="none").mean(dim=-1)
            loss_manifold_tangent = ValidMeanM(loss_manifold_tangent_vec)
            loss_manifold_latent = loss_manifold_geo + value.new_tensor(0.5) * loss_manifold_tangent
            loss_manifold_reg = prev["manifold_reg"]
        loss_transport = (
            value.new_tensor(base.wDiff) * loss_diff
            + value.new_tensor(base.wDiffBranch) * loss_diff_branch
            + value.new_tensor(base.wBranchStructure) * loss_branch_structure
            + value.new_tensor(base.wManifoldLatent) * loss_manifold_latent
            + loss_manifold_reg)

        loss_transport_delayed = loss_transport

        loss_physical_td = F.smooth_l1_loss(td_bounded, td_bounded.new_zeros(td_bounded.shape))
        loss_physical_aux = (
            value.new_tensor(0.02) * physical_td["td_bures"].mean()
            + value.new_tensor(0.01) * physical_td["td_heat"].mean()
            + value.new_tensor(0.005) * physical_td["td_ot"].mean())
        physical_param_reg = base.BuildPhysicalTDParameterRegularizer()
        loss_physical_param_reg = value.new_tensor(base.wPhysicalTDParamReg) * physical_param_reg["loss"]
        loss_value_tensor_energy = value.new_tensor(1e-6) * value.pow(2).mean()
        quantile_target = (
            td_bounded.detach().view(-1, 1)
            + (base.quantile_tau.view(1, -1) - 0.5)
            * 2.0
            * physical_td["td_mag"].detach().view(-1, 1)).clamp(-3.0, 3.0)
        loss_quantile_fit = base.QuantileHuberLoss(value_quantiles, quantile_target)
        loss_quantile_order = base.QuantileCrossingLoss(value_quantiles)
        loss_quantile = (
            value.new_tensor(0.02) * loss_quantile_fit
            + value.new_tensor(0.001) * loss_quantile_order)
        ensemble_mean = value_ensemble.mean(dim=-1)
        ensemble_var = value_ensemble.var(dim=-1, unbiased=False)
        ensemble_mean_target = physical_td["td_mag"].detach()
        ensemble_var_target = (
            0.50 * physical_td["td_context"].detach()
            + 0.20 * physical_td["td_bures"].detach()
            + 0.20 * physical_td["td_heat"].detach()
            + 0.10 * physical_td["td_ot"].detach()).clamp(0.0, 3.0)
        loss_ensemble = (
            value.new_tensor(0.01) * F.smooth_l1_loss(ensemble_mean, ensemble_mean_target)
            + value.new_tensor(0.005) * F.smooth_l1_loss(ensemble_var, ensemble_var_target))
        pending_v_next_hat, pending_transp_extras = base.BuildTransportSnapshotGraph(
            transport_h,
            transport_value,
            base._transport_prev_grad,
            base._transport_prev_grad_hook_seen)
        pending_v_next_hat, pending_transp_extras = base.ApplyRewardNextModulation(
            transport_value,
            pending_v_next_hat,
            pending_transp_extras,
            r_next_hat)
        base.CacheDelayedTransitionInputs(
            vNextHat=pending_v_next_hat,
            alive=(1.0 - done_value),
            transpExtras=pending_transp_extras,
            streamIds=streamIds)

        loss_current = loss_physical_td + loss_physical_aux + loss_physical_param_reg + loss_value_tensor_energy + loss_quantile + loss_ensemble

        extras: Dict[str, torch.Tensor] = {
            "loss_transport": loss_transport_delayed.detach(),
            "loss_physical_param_reg": loss_physical_param_reg.detach(),
            "loss_quantile": loss_quantile.detach(),
            "loss_ensemble": loss_ensemble.detach(),
            "loss_diff": loss_diff.detach(),
            "loss_diff_branch": loss_diff_branch.detach(),
            "loss_branch_structure": loss_branch_structure.detach(),
            "loss_manifold_geo": loss_manifold_geo.detach(),
            "loss_manifold_tangent": loss_manifold_tangent.detach(),
            "loss_manifold_latent": loss_manifold_latent.detach(),
            "td_mag": physical_td["td_mag"].detach(),
            "loss_current_graph": loss_current,
            "loss_transport_delayed_graph": loss_transport_delayed,}

        return GeoTropicalOut(
            value=value,
            valueNext=value_next,
            tdError=td_bounded.detach(),
            loss=loss_current,
            emotion=emotion,
            rComps=rComps,
            uncertainty=unc01,
            precision=precision,
            extras=extras,)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        mapping = {
            "fc1": ("fc1_adapter", "fc1", [0]),
            "fc2": ("fc2_adapter", "fc2", [1]),
            "qhead": ("quantile_adapter", "quantile_head", [1]),}

        if site not in mapping:
            return False
        attr_name, tgt_name, allow_layers = mapping[site]
        if layerIdx not in allow_layers:
            return False

        target: nn.Linear = getattr(self.base, tgt_name)

        adapter: GrowableLoRALinear = getattr(self.base, attr_name)
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

    def NewEstimator(self, *, useHebb: bool = True):
        return ValueEstimationExtractor(
            memoryDim=self.mem_dim,
            attnDim=self.attn_dim,
            stateDim=self.state_dim,
            hidden=self.hidden,
            useHebb=useHebb).to(self.device)

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

            est = self.NewEstimator(useHebb=True)
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

            est = self.NewEstimator(useHebb=False).train()

            def print_shape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            print_shape("input.memoryPrev", mem)
            print_shape("input.attnPrev", attn)
            print_shape("input.state", state)
            print_shape("input.rewardModel", reward)
            print_shape("input.policyEntropyPrev", entropy)
            print_shape("input.done", done)
            print_shape("input.worldDeltaTransport", d_tr)
            print_shape("input.worldDeltaPhysics", d_ph)

            out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            out_dict = out._asdict()
            for key, value in out_dict.items():
                if isinstance(value, torch.Tensor):
                    print_shape(f"output.{key}", value)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            print_shape(f"output.{key}.{sub_key}", sub_value)

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

            est = self.NewEstimator(useHebb=False).eval()
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
            est = self.NewEstimator(useHebb=False).eval()
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
            with torch.no_grad():
                out_a = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            est.ResetState()
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

            est = self.NewEstimator(useHebb=False).eval()
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
            est = self.NewEstimator(useHebb=False).train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)
            done2 = torch.tensor([0.0, 1.0, 0.0, 1.0], device=self.device)

            out1 = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            pending_after_t1 = sum(len(q) for q in est._pending_transitions.values()) == B
            pending_items = [est._pending_transitions[i][0] for i in range(B)]
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
            est = self.NewEstimator(useHebb=False).train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            prev_pred = torch.cat([est._pending_transitions[i][0]["pred_live"].detach() for i in range(B)], dim=0)
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

    def TestPendingQueueMultipleOutstandingItems(self) -> bool:
        try:
            torch.manual_seed(440)
            B = 3
            est = self.NewEstimator(useHebb=False).train()
            sids_a = torch.tensor([10, 20, 30], device=self.device)
            sids_b = torch.tensor([40, 50, 60], device=self.device)
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            _ = est(mem, attn, state, rewardModel=reward, policyEntropyPrev=entropy, doneModel=done,
                    worldDeltaTransport=d_tr, worldDeltaPhysics=d_ph, streamIds=sids_a)
            _ = est(mem, attn, state, rewardModel=reward, policyEntropyPrev=entropy, doneModel=done,
                    worldDeltaTransport=d_tr, worldDeltaPhysics=d_ph, streamIds=sids_b)
            pending_after_two = sum(len(q) for q in est._pending_transitions.values())
            out = est(mem, attn, state, rewardModel=reward, policyEntropyPrev=entropy, doneModel=done,
                      worldDeltaTransport=d_tr, worldDeltaPhysics=d_ph, streamIds=sids_a)
            pending_after_consume = sum(len(q) for q in est._pending_transitions.values())

            ok = True
            ok &= pending_after_two == 2 * B
            ok &= pending_after_consume == 2 * B
            ok &= torch.isfinite(out.extras["loss_diff"]).item()

            print(f"PendingQueueMultipleOutstandingItems {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"PendingQueueMultipleOutstandingItems error: {e}")
            return False

    def TestTemporalPairingOfUncertainty(self) -> bool:
        try:
            torch.manual_seed(441)
            B = 4
            est = self.NewEstimator(useHebb=False).train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            pending_items = [est._pending_transitions[i][0] for i in range(B)]
            out2 = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            ok = True
            ok &= all("unc" not in it for it in pending_items)
            ok &= all("pred_live" in it and "transp_extras_live" in it for it in pending_items)
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
            est = self.NewEstimator(useHebb=False).train()
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
            est = self.NewEstimator(useHebb=False).train()
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
            est = self.NewEstimator(useHebb=False).train()

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
            # beat: after Apply, curr (t1 main) is rotated into prev for next beat
            ok &= metrics["t1_prev_grad_norm_after_rotate"] > 0.0
            ok &= metrics["t1_curr_grad_norm_after_rotate"] == 0.0
            ok &= metrics["pending_after_t2"] == float(B)
            ok &= metrics["t2_delayed_captured"] > 0.0
            # prev holds {t1 main (rotated) + t2 diff (snapshot hook)} before Apply
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
            est = self.NewEstimator(useHebb=True).train()
            mem1, attn1, state1 = self.RandBatch(B)
            mem2, attn2, state2 = self.RandBatch(B)
            reward1, entropy1, done1, d_tr1, d_ph1 = self.RandSignals(B, doneProb=0.0)
            reward2, entropy2, done2, d_tr2, d_ph2 = self.RandSignals(B, doneProb=0.0)

            _ = self.ForwardOnce(est, mem1, attn1, state1, reward1, entropy1, done1, d_tr1, d_ph1)
            items = [est._pending_transitions[i][0] for i in range(B)]
            _ = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)

            ok = True
            ok &= all("hebb_state" not in it for it in items)
            ok &= all("memory" not in it and "attn" not in it and "state" not in it for it in items)
            ok &= torch.isfinite(est._pending_transitions[0][0]["pred_live"]).all().item()

            print(f"HebbianSnapshotUsedForDelayedRebuild {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"HebbianSnapshotUsedForDelayedRebuild error: {e}")
            return False

    def TestTerminalMaskDelayedLosses(self) -> bool:
        try:
            torch.manual_seed(438)
            B = 4
            est_a = self.NewEstimator(useHebb=False).train()
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
            prev_valid = torch.stack([est_a._pending_transitions[i][0]["alive"] for i in range(B)], dim=0).detach().clone().clamp(0.0, 1.0)
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
                donePrev=zeros)
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
            est = self.NewEstimator(useHebb=True).train()
            for _ in range(3):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
                _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            done_mask = torch.tensor([True, False, True, False], device=self.device)
            est.ResetHebbianMemory(doneMask=done_mask)

            ok = True
            ok &= int(est.micro.filled[0].item()) == 0
            ok &= int(est.micro.filled[2].item()) == 0
            ok &= int(est.micro.filled[1].item()) > 0
            ok &= int(est.micro.filled[3].item()) > 0
            ok &= 0 not in est._pending_transitions
            ok &= 2 not in est._pending_transitions
            ok &= 1 in est._pending_transitions
            ok &= 3 in est._pending_transitions
            print(f"DoneMaskResetPerSample {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"DoneMaskResetPerSample error: {e}")
            return False

    def TestDistributionalValueAndTransport(self) -> bool:
        try:
            torch.manual_seed(437)
            B = 5
            est = self.NewEstimator(useHebb=False).train()
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
            est = self.NewEstimator(useHebb=False).train()

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
            est = self.NewEstimator(useHebb=False).train()

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
            ok &= (est.reward_perdetic.kf_mean.numel() == B2)
            ok &= (est.done_perdetic.kf_mean.numel() == B2)
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
            est = self.NewEstimator(useHebb=True).train()

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

            print(f"ResetFunctions {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ResetFunctions error: {e}")
            return False

    def TestAllTrainableParamsHaveGradAndStep(self) -> bool:
        try:
            torch.manual_seed(101)
            B = 10
            est = self.NewEstimator(useHebb=False)
            est.train()
            est.emotion_core.fast_head.use_hebbian = False
            est.emotion_core.slow_head.use_hebbian = False
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
                loss = out.loss + 1e-2 * (out.emotion ** 2).mean()
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

            est_ref_eval = self.NewEstimator(useHebb=True).eval()
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

            est_ref_train = self.NewEstimator(useHebb=True).train()
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
            base = self.NewEstimator(useHebb=False)
            wrapper = ValueEstimationOnlineWrapper(base, initRankEach=0, autoRank=False)
            wrapper.train()

            active_slots = [("fc1", 0), ("fc2", 1)]
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
                base_frozen_ok = not any(p.requires_grad for p in wrapper.base.parameters())

            ok = all(changed) and base_frozen_ok
            print(f"WrapperCandidateParamsTrainable {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"WrapperCandidateParamsTrainable error: {e}")
            return False

    def TestWrapperSimCommitEquivalence(self) -> bool:
        try:
            torch.manual_seed(78)
            B = 6
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            base = self.NewEstimator(useHebb=False).eval()
            wrapper = ValueEstimationOnlineWrapper(base, initRankEach=0, autoRank=False).eval()

            spec = wrapper.sites["qhead"]
            a, b, s = spec.allocFn(2, wrapper.deviceRef, wrapper.dtypeRef)
            with torch.no_grad():
                a.mul_(0.1)
                b.mul_(0.1)
                s.fill_(0.7)
            wrapper.cand["qhead"][1]["A"].append(a)
            wrapper.cand["qhead"][1]["B"].append(b)
            wrapper.cand["qhead"][1]["s"].append(s)

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
            base = self.NewEstimator(useHebb=False)
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
            est1 = self.NewEstimator(useHebb=True).train()

            for _ in range(3):
                mem, attn, state = self.RandBatch(B)
                reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
                _ = self.ForwardOnce(est1, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            dyn_state = est1.ExportState()

            est2 = self.NewEstimator(useHebb=True).train()
            src = est1.state_dict()
            dst = est2.state_dict()
            loadable = {k: v for k, v in src.items() if (k in dst and dst[k].shape == v.shape)}
            est2.load_state_dict(loadable, strict=False)
            est2.ImportState(dyn_state)

            ok = True
            ok &= len(est2._pending_transitions) == 0
            ok &= torch.equal(est1.micro.filled, est2.micro.filled)
            ok &= torch.equal(est1.micro.ptr, est2.micro.ptr)
            ok &= torch.allclose(est1.micro.anchor_value, est2.micro.anchor_value)
            ok &= torch.allclose(est1.micro.anchor_value_next, est2.micro.anchor_value_next)
            ok &= torch.allclose(est1.reward_perdetic.kf_mean, est2.reward_perdetic.kf_mean)
            ok &= torch.allclose(est1.done_perdetic.kf_mean, est2.done_perdetic.kf_mean)
            ok &= torch.allclose(est1.emotion_core.fast_head.H, est2.emotion_core.fast_head.H)
            ok &= torch.allclose(est1.emotion_core.slow_head.H, est2.emotion_core.slow_head.H)

            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)

            out1 = self.ForwardOnce(est1, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out2 = self.ForwardOnce(est2, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            atol, rtol = 1e-6, 1e-5
            ok &= torch.allclose(out1.value, out2.value, atol=atol, rtol=rtol)

            print(f"ExportImportStateRoundTrip {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"ExportImportStateRoundTrip error: {e}")
            return False

    def TestLossDecreases(self, steps: int = 48, batchSize: int = 8) -> bool:
        try:
            torch.manual_seed(2026)
            est = self.NewEstimator(useHebb=False).train()
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
            est = self.NewEstimator(useHebb=False).train()
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
            ("PendingQueueMultipleOutstandingItems", self.TestPendingQueueMultipleOutstandingItems),
            ("TemporalPairingOfUncertainty", self.TestTemporalPairingOfUncertainty),
            ("TransportManifoldFieldGrad", self.TestTransportManifoldFieldGrad),
            ("ManualTransportGradWorkflow", self.TestManualTransportGradWorkflow),
            ("TransportDelayedGradientPipeline", self.TestTransportDelayedGradientPipeline),
            ("HebbianSnapshotUsedForDelayedRebuild", self.TestHebbianSnapshotUsedForDelayedRebuild),
            ("TerminalMaskDelayedLosses", self.TestTerminalMaskDelayedLosses),
            ("UncertaintyFloorNearEps", self.TestUncertaintyFloorNearEps),
            ("DoneMaskResetPerSample", self.TestDoneMaskResetPerSample),
            ("DistributionalValueAndTransport", self.TestDistributionalValueAndTransport),
            ("StateMachineAndMicroGraph", self.TestStateMachineAndMicroGraph),
            ("BatchResizeAndPredictorShapes", self.TestBatchResizeAndPredictorShapes),
            ("ResetFunctions", self.TestResetFunctions),
            ("AllTrainableParamsHaveGradAndStep", self.TestAllTrainableParamsHaveGradAndStep),
            ("WrapperAlignmentNoDelta", self.TestWrapperAlignmentNoDelta),
            ("WrapperCandidateParamsTrainable", self.TestWrapperCandidateParamsTrainable),
            ("WrapperSimCommitEquivalence", self.TestWrapperSimCommitEquivalence),
            ("WrapperUpdateWorkflow", self.TestWrapperUpdateWorkflow),
            ("ExportImportStateRoundTrip", self.TestExportImportStateRoundTrip),
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
