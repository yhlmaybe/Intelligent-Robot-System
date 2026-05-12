from __future__ import annotations
from typing import Optional, Dict, NamedTuple, Tuple, List, Any
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics as stats
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
        mask = doneMask.to(device=self.H.device, dtype=torch.bool).view(-1)
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

        y_hebb = torch.bmm(self.H, x.unsqueeze(-1)).squeeze(-1) # [B,O]

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
                mask = doneMask.to(device=self.mean.device, dtype=torch.bool).view(-1)
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
        mag = torch.abs(fft).clone()
        if mag.size(1) <= 1:
            return (y[:, -1] + (y[:, -1] - y[:, -2]))

        mag[:, 0] = 0.0
        mag_tail = mag[:, 1:]

        K = int(min(self.max_harmonics, mag_tail.size(1)))
        if K <= 0:
            return (y[:, -1] + (y[:, -1] - y[:, -2]))

        idx = torch.topk(mag_tail, k=K, dim=1, largest=True, sorted=True).indices
        k_bin = (idx + 1).to(self.dtype)
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
        mask = doneMask.to(device=self.kf_mean.device, dtype=torch.bool).view(-1)
        n = min(int(mask.numel()), int(self.kf_mean.numel()))
        if n <= 0:
            return
        rows = mask[:n].nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        self.kf_mean[rows] = 0
        self.kf_var[rows] = float(self.init_var)
        if self.smooth_hist.dim() == 2 and self.smooth_hist.size(0) >= n:
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
            self.kf_mean = z.detach().clone()
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
        driftScale: float = 0.10,):
        super().__init__()
        self.k = max(1, int(numExperts))
        self.c = max(1, int(numCounterfactuals))
        self.epsA = float(epsA)
        self.expert_temp = float(expertTemp)
        self.a_delta_limit = float(aDeltaLimit)
        self.b_limit = float(bLimit)
        self.drift_scale = float(driftScale)

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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

        nn.init.zeros_(self.expert_gate.weight)
        nn.init.zeros_(self.expert_gate.bias)
        nn.init.zeros_(self.a_head.weight)
        nn.init.zeros_(self.a_head.bias)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)
        nn.init.zeros_(self.g_head.weight)
        nn.init.constant_(self.g_head.bias, -2.0)
        nn.init.zeros_(self.cf_gate.weight)
        nn.init.constant_(self.cf_gate.bias, -4.0)
        nn.init.zeros_(self.cf_a_head.weight)
        nn.init.zeros_(self.cf_a_head.bias)
        nn.init.zeros_(self.cf_b_head.weight)
        nn.init.zeros_(self.cf_b_head.bias)
        nn.init.zeros_(self.cf_g_head.weight)
        nn.init.constant_(self.cf_g_head.bias, -2.0)

    def forward(
        self,
        h: torch.Tensor,
        v: torch.Tensor,
        returnExtras: Optional[bool] = None,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        v_in = v # [B,1]
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

        trop_in = torch.cat([ctx, v_in], dim=-1)
        trop_all = self.trop(trop_in) # [B,K]
        aff_all = a * v_in + b # [B,K]
        flow_all = g * trop_all + (1.0 - g) * aff_all # [B,K]

        cf_ctx = self.cf_core(ctx) # [B,H]
        cf_logits = self.cf_gate(cf_ctx) # [B,C]
        cf_a = (1.0 + self.a_delta_limit * torch.tanh(self.cf_a_head(cf_ctx))).clamp_min(self.epsA) # [B,C]
        cf_b = self.b_limit * torch.tanh(self.cf_b_head(cf_ctx)) # [B,C]
        cf_g = torch.sigmoid(self.cf_g_head(cf_ctx)) # [B,C]
        cf_trop_in = torch.cat([cf_ctx, v_in], dim=-1)
        cf_trop_all = self.cf_trop(cf_trop_in) # [B,C]
        cf_aff_all = cf_a * v_in + cf_b # [B,C]
        cf_flow_all = cf_g * cf_trop_all + (1.0 - cf_g) * cf_aff_all # [B,C]

        branch_logits = torch.cat([w_logits, cf_logits], dim=-1) # [B,K+C]
        branch_w = torch.softmax(branch_logits / tau, dim=-1) # [B,K+C]
        branch_flow_all = torch.cat([flow_all, cf_flow_all], dim=-1) # [B,K+C]

        flow_mix = (branch_w * branch_flow_all).sum(dim=-1, keepdim=True) # [B,1]
        dv = self.drift_scale * torch.tanh(flow_mix - v_in)
        v_next_hat = v_in + dv # [B,1]
        if returnExtras is None:
            returnExtras = bool(self.training)
        if not returnExtras:
            return v_next_hat, {}

        w = branch_w[:, :self.k] # [B,K]
        cf_w = branch_w[:, self.k:] # [B,C]
        branch_next_all = v_in + self.drift_scale * torch.tanh(branch_flow_all - v_in) # [B,K+C]

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
            "expert_w": w,
            "expert_trop": trop_all,
            "expert_aff": aff_all,
            "counterfactual_w": cf_w,
            "counterfactual_values": branch_next_all[:, self.k:],
            "counterfactual_trop": cf_trop_all,
            "counterfactual_aff": cf_aff_all,
            "branch_w": branch_w,
            "branch_flow": branch_flow_all,
            "branch_next": branch_next_all,}
        
        return v_next_hat, extras



class GITGaugeRegularizer(AGICoreModule):
    def __init__(self, wScale: float = 1e-3, wShift: float = 1e-3, wSign: float = 1e-3, tauSign: float = 0.2, eps: float = 1e-8):
        super().__init__()
        self.w_scale = float(wScale)
        self.w_shift = float(wShift)
        self.w_sign  = float(wSign)
        self.tau_sign = float(tauSign)
        self.eps = float(eps)

    def forward(self, valueHead: nn.Linear, transpExtras: Dict[str, torch.Tensor], adapter: Optional[nn.Module] = None) -> torch.Tensor:
        W = valueHead.weight # [O, I]

        if (adapter is not None) and hasattr(adapter, "DeltaWeight"):
            dW = adapter.DeltaWeight()
            if dW is not None:
                W = W + dW

        reg = W.new_zeros(())

        fro = torch.linalg.matrix_norm(W, ord="fro")
        fro_n = fro / (W.numel() ** 0.5 + self.eps)
        reg = reg + self.w_scale * (fro_n - 1.0).pow(2)

        if "b" in transpExtras and transpExtras["b"] is not None:
            b = transpExtras["b"]
            reg = reg + self.w_shift * b.pow(2).mean()

        if W.numel() > 0:
            row_score = W.abs().amax(dim=1) # [O]
            tau = max(self.tau_sign, 1e-6)
            row_w = torch.softmax(row_score / tau, dim=0)  
            row_mean = W.mean(dim=1)  
            soft_row_mean = (row_w * row_mean).sum() 
            reg = reg + self.w_sign * F.relu(-soft_row_mean)

        return reg



class TemporalMicroGraph(AGICoreModule):
    def __init__(
        self,
        embDim: int,
        maxAnchors: int = 512,
        topk: int = 4,
        distTau: float = 0.5,
        lenPower: float = 0.5,
        eps: float = 1e-8,):
        super().__init__()
        self.emb_dim = int(embDim)
        self.max_anchors = int(maxAnchors)
        self.topk = int(topk)
        self.dist_tau = float(distTau)
        self.len_power = float(lenPower)
        self.eps = float(eps)

        self.register_buffer("prefix_G", torch.ones(1, 1))
        self.register_buffer("prefix_C", torch.zeros(1, 1))

        self.register_buffer("anchor_z", torch.zeros(1, self.max_anchors, self.emb_dim))
        self.register_buffer("anchor_v", torch.zeros(1, self.max_anchors, 1))
        self.register_buffer("anchor_G", torch.ones(1, self.max_anchors, 1))
        self.register_buffer("anchor_C", torch.zeros(1, self.max_anchors, 1))

        self.register_buffer("filled", torch.zeros(1, dtype=torch.long)) # [B]
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long)) # [B]
        self._step = 0

    @torch.no_grad()
    def EnsureB(self, B: int, device, dtype):
        B = int(B)

        if B != self.prefix_G.size(0):

            self.prefix_G = torch.ones(B, 1, device=device, dtype=dtype)
            self.prefix_C = torch.zeros(B, 1, device=device, dtype=dtype)

            M = int(self.max_anchors)
            D = int(self.emb_dim)

            self.anchor_z = torch.zeros(B, M, D, device=device, dtype=dtype)
            self.anchor_v = torch.zeros(B, M, 1, device=device, dtype=dtype)
            self.anchor_G = torch.ones(B, M, 1, device=device, dtype=dtype)
            self.anchor_C = torch.zeros(B, M, 1, device=device, dtype=dtype)

            self.filled = torch.zeros(B, device=device, dtype=torch.long)
            self.ptr = torch.zeros(B, device=device, dtype=torch.long)

            self._step = 0

    @torch.no_grad()
    def Reset(self, doneMask: Optional[torch.Tensor] = None):
        if doneMask is None:
            self.prefix_G.fill_(1.0)
            self.prefix_C.zero_()
            self.anchor_z.zero_()
            self.anchor_v.zero_()
            self.anchor_G.fill_(1.0)
            self.anchor_C.zero_()
            self.filled.zero_()
            self.ptr.zero_()
            self._step = 0
            return
        if self.prefix_G.size(0) <= 0:
            return
        mask = doneMask.to(device=self.prefix_G.device, dtype=torch.bool).view(-1)
        n = min(int(mask.numel()), int(self.prefix_G.size(0)))
        if n <= 0:
            return
        rows = mask[:n].nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        self.prefix_G[rows] = 1.0
        self.prefix_C[rows] = 0
        self.anchor_z[rows] = 0
        self.anchor_v[rows] = 0
        self.anchor_G[rows] = 1.0
        self.anchor_C[rows] = 0
        self.filled[rows] = 0
        self.ptr[rows] = 0


    def PreviewEdges(
        self,
        zNow: torch.Tensor,
        rNow: torch.Tensor,
        gNow: torch.Tensor,
        topk: Optional[int] = None) -> Dict[str, torch.Tensor]:
        B = zNow.size(0)
        device, dtype = self.device, self.dtype

        self.EnsureB(B, device, dtype)

        rNow = rNow.view(B, 1) # [B,1]
        gNow = gNow.view(B, 1).clamp(0.0, 1.0) # [B,1]

        G_new = self.prefix_G * gNow # [B,1]
        C_new = self.prefix_C + self.prefix_G * rNow  # [B,1]

        K = min(int(topk or self.topk), self.max_anchors)
        if K <= 0:
            empty_idx = torch.empty((B, 0), dtype=torch.long, device=device)
            empty = torch.empty((B, 0), device=device, dtype=dtype)
            return {"idx": empty_idx, "R": empty, "Gamma": empty, "v_hist": empty, "w": empty, "dist": empty, "valid": empty.bool()}

        M = self.max_anchors
        ar = torch.arange(M, device=device).view(1, M)
        valid_mask = ar < self.filled.view(B, 1) # [B,M]

        diff = self.anchor_z - zNow.unsqueeze(1) # [B,M,D]
        dist = (diff * diff).sum(dim=-1).sqrt() # [B,M]

        M = self.max_anchors
        pos = torch.arange(M, device=device).view(1, M).expand(B, M)  # [B,M]

        age = (self.ptr.view(B, 1) - 1 - pos) % M
        age = age.to(dist.dtype)

        dist = dist + age * 1e-6

        dist = dist.masked_fill(~valid_mask, float("inf"))

        _, idx = torch.topk(-dist, k=K, dim=1) # idx: [B,K]
        d_sel = dist.gather(1, idx) # [B,K]
        valid = torch.isfinite(d_sel) # [B,K]

        idx3 = idx.unsqueeze(-1) # [B,K,1]
        G_i = self.anchor_G.gather(1, idx3) # [B,K,1]
        C_i = self.anchor_C.gather(1, idx3) # [B,K,1]
        v_i = self.anchor_v.gather(1, idx3) # [B,K,1]

        Gamma_seg = (G_new.unsqueeze(1) / (G_i + self.eps)).squeeze(-1) # [B,K]
        R_seg = ((C_new.unsqueeze(1) - C_i) / (G_i + self.eps)).squeeze(-1) # [B,K]

        tau = max(self.dist_tau, 1e-6)
        w = torch.exp(-d_sel / tau) * (Gamma_seg.clamp_min(1e-6) ** self.len_power) # [B,K]

        w = w.masked_fill(~valid, 0.0) # [B,K]
        Gamma_seg = Gamma_seg.masked_fill(~valid, 0.0)
        R_seg = R_seg.masked_fill(~valid, 0.0)
        v_hist = v_i.squeeze(-1).masked_fill(~valid, 0.0) # [B,K]

        return {"idx": idx, "R": R_seg, "Gamma": Gamma_seg, "v_hist": v_hist, "w": w, "dist": d_sel, "valid": valid}

    @torch.no_grad()
    def CommitStep(self, zNow: torch.Tensor, vNow: torch.Tensor, rNow: torch.Tensor, gNow: torch.Tensor):
        B, D = zNow.size(0), zNow.size(1)
        device, dtype = self.device, self.dtype
        self.EnsureB(B, device, dtype)

        vNow = vNow.view(B, 1) # [B,1]
        rNow = rNow.view(B, 1) # [B,1]
        gNow = gNow.view(B, 1).clamp(0.0, 1.0) # [B,1]

        self.prefix_C.add_(self.prefix_G * rNow) # [B,1]
        self.prefix_G.mul_(gNow) # [B,1]

        b_idx = torch.arange(B, device=device)
        pos = self.ptr.clamp_min(0).clamp_max(self.max_anchors - 1) # [B]

        self.anchor_z[b_idx, pos] = zNow.detach()
        self.anchor_v[b_idx, pos, 0] = vNow.detach().view(B)
        self.anchor_G[b_idx, pos, 0] = self.prefix_G.detach().view(B)
        self.anchor_C[b_idx, pos, 0] = self.prefix_C.detach().view(B)

        self.ptr = (self.ptr + 1) % self.max_anchors
        self.filled = torch.minimum(self.filled + 1, torch.full_like(self.filled, self.max_anchors))

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
        mask = doneMask.to(device=self.h.device, dtype=torch.bool).view(-1)
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
        microMaxAnchors: int = 256, microTopK: int = 4, microDistTau: float = 0.5, microLenPower: float = 0.5,
        wGITScale: float = 1e-3, wGITShift: float = 1e-3, wGITSign: float = 1e-3,
        useHebb: bool = True,):
        super().__init__()

        self.in_dim = memoryDim + attnDim + stateDim
        H = hidden
        self.num_quantiles = 32
        self.cvar_alpha = 0.20
        tau = (torch.arange(self.num_quantiles, dtype=torch.float32) + 0.5) / float(self.num_quantiles)
        self.register_buffer("quantile_tau", tau)

        self.use_hebb = useHebb

        self.td_out_ema = RunningEMA(momentum=0.99) 
        self.td_scale_min = 1e-3
        self.unc_tau = 4.0 

        self.w_unc = 0.1

        self.reward_perdetic = KalmanFilteredEnsembleNext()
        self.done_perdetic = KalmanFilteredEnsembleNext()

        self.fc1 = nn.Linear(self.in_dim, H)
        self.fc2 = nn.Linear(H, H)
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)
        self.trunk_res_blocks = nn.ModuleList([
            ResidualMLPBlock(dim=H, hiddenMul=trunkResHiddenMul, scaleInit=trunkResScaleInit)
            for _ in range(max(0, int(trunkResBlocks)))])

        self.hebb_value = HebbianLinearFW(H, 1, bias=True,useHebbian=useHebb)
        self.value_head = nn.Linear(H, 1)
        self.model_value_head = nn.Linear(H, 1)
        self.calibration_head = nn.Linear(H, 4)
        self.quantile_head = nn.Linear(H, self.num_quantiles)
        self.value_ensemble_heads = nn.ModuleList([nn.Linear(H, 1) for _ in range(4)])
        self.aleatoric_head = nn.Linear(H, 1)

        self.fc1_adapter = GrowableLoRALinear(self.fc1)
        self.fc2_adapter = GrowableLoRALinear(self.fc2)
        self.value_adapter = GrowableLoRALinear(self.value_head)
        self.model_value_adapter = GrowableLoRALinear(self.model_value_head)
        self.calibration_adapter = GrowableLoRALinear(self.calibration_head)
        self.quantile_adapter = GrowableLoRALinear(self.quantile_head)

        self.mix_gate = nn.Linear(H, 1)
        self.model_fusion_gate = nn.Linear(H, 1)
        self.graph_fusion_gate = nn.Linear(H, 1)
        self.risk_fusion_gate = nn.Linear(H + 6, 1)

        self.emotion_dim = emotionDim
        self.emotion_core = EmotionCore(stateDim=stateDim, memoryDim=memoryDim,attnDim=attnDim, emotionDim=emotionDim)

        self.wMixGateReg = 1e-3

        self.transport = TropicalAffineTransport(H, useSoftTrop, tropTemp, epsA)
        self.git = GITGaugeRegularizer(wScale=wGITScale, wShift=wGITShift, wSign=wGITSign)

        self.micro = TemporalMicroGraph(embDim=H, maxAnchors=microMaxAnchors, topk=microTopK, distTau=microDistTau, lenPower=microLenPower)

        self._prev_h: torch.Tensor = None # [B,H]
        self._prev_value: torch.Tensor = None # [B,1]
        self._prev_v_next_pred: torch.Tensor = None # [B,1]
        self._prev_transp_extras: Dict[str, torch.Tensor] = {}
        self._prev_value_td: torch.Tensor = None # [B,1]
        self._prev_value_model: torch.Tensor = None # [B,1]
        self._prev_value_quantiles: torch.Tensor = None # [B,Nq]
        self._prev_value_ensemble: torch.Tensor = None # [B,K]
        self._prev_reward_next: torch.Tensor = None # [B]
        self._prev_discount_next: torch.Tensor = None # [B]
        self._prev_alive: torch.Tensor = None # [B]
        self._prev_unc: torch.Tensor = None # [B]
        self._delay_graph_cache: Optional[Dict[str, torch.Tensor]] = None
        self.delayed_graph_after_optimizer_step = False

        self.unc_core = UncertaintyCore(
            stateDim=stateDim,
            memDim=memoryDim,
            attnDim=attnDim,
            hidden=max(256, H // 2))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

        nn.init.zeros_(self.mix_gate.weight)
        nn.init.constant_(self.mix_gate.bias, -2.0)  
        nn.init.zeros_(self.model_fusion_gate.weight)
        nn.init.constant_(self.model_fusion_gate.bias, -2.0)
        nn.init.zeros_(self.graph_fusion_gate.weight)
        nn.init.constant_(self.graph_fusion_gate.bias, -2.0)
        nn.init.zeros_(self.risk_fusion_gate.weight)
        nn.init.zeros_(self.risk_fusion_gate.bias)
        nn.init.zeros_(self.calibration_head.weight)
        nn.init.zeros_(self.quantile_head.weight)
        nn.init.zeros_(self.quantile_head.bias)
        with torch.no_grad():
            self.calibration_head.bias.copy_(torch.tensor([-2.0, -2.0, -2.0, 2.0], dtype=self.calibration_head.bias.dtype))
            self.aleatoric_head.weight.zero_()
            self.aleatoric_head.bias.fill_(-4.0)

        self.emotion_core.ResetParams()

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        self.td_out_ema.EnsureB(B, device, dtype)

        if self._prev_h is not None and self._prev_h.size(0) != B:
            self._prev_h: torch.Tensor = None # [B,H]
            self._prev_value: torch.Tensor = None # [B,1]
            self._prev_v_next_pred: torch.Tensor = None # [B,1]
            self._prev_transp_extras = {}
            self._prev_value_td = None
            self._prev_value_model = None
            self._prev_value_quantiles = None
            self._prev_value_ensemble = None
            self._prev_reward_next = None
            self._prev_discount_next = None
            self._prev_alive: torch.Tensor = None # [B]
            self._prev_unc: torch.Tensor = None # [B]
            self._delay_graph_cache = None

    def Trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1_adapter(x))
        h = self.norm1(h)
        h = F.gelu(self.fc2_adapter(h))
        h = self.norm2(h)
        for blk in self.trunk_res_blocks:
            h = blk(h)
        return h

    def BuildCalibration(
        self,
        calibRaw: torch.Tensor,
        tdBounded: torch.Tensor,
        uncPrior01: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        risk_logit, amb_logit, surprise_logit, conf_logit = calibRaw.split(1, dim=-1)
        td_abs = tdBounded.detach().abs().view(-1, 1)
        unc_prior = uncPrior01.detach().view(-1, 1)

        risk = torch.sigmoid(risk_logit).squeeze(-1)
        ambiguity = torch.sigmoid(amb_logit + unc_prior).squeeze(-1)
        surprise = torch.sigmoid(surprise_logit + td_abs).squeeze(-1)
        confidence = torch.sigmoid(conf_logit - 0.75 * ambiguity.view(-1, 1) - 0.50 * unc_prior).squeeze(-1)

        learned_unc = (
            0.35 * risk
            + 0.30 * ambiguity
            + 0.20 * surprise
            + 0.15 * (1.0 - confidence)).clamp(0.0, 1.0)
        return risk, ambiguity, surprise, confidence, learned_unc

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
        if target.dim() == 1:
            target = target[:, None]
        target = target.detach()
        err = target - quantiles # [B,N]
        abs_err = err.abs()
        k = float(kappa)
        huber = torch.where(abs_err <= k, 0.5 * err.pow(2), k * (abs_err - 0.5 * k))
        tau = self.quantile_tau.to(device=quantiles.device, dtype=quantiles.dtype).view(1, -1)
        weight = (tau - (err.detach() < 0).to(dtype=quantiles.dtype)).abs()
        loss_per_row = (weight * huber).mean(dim=-1) # [B]
        if sampleWeight is None:
            return loss_per_row.mean()
        sw = sampleWeight.to(device=quantiles.device, dtype=quantiles.dtype).view(-1)
        return (loss_per_row * sw).sum() / sw.sum().clamp_min(1.0)

    @staticmethod
    def QuantileCrossingLoss(quantiles: torch.Tensor, sampleWeight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if quantiles.size(-1) <= 1:
            return quantiles.new_zeros(())
        loss_per_row = F.relu(quantiles[:, :-1] - quantiles[:, 1:]).mean(dim=-1) # [B]
        if sampleWeight is None:
            return loss_per_row.mean()
        sw = sampleWeight.to(device=quantiles.device, dtype=quantiles.dtype).view(-1)
        return (loss_per_row * sw).sum() / sw.sum().clamp_min(1.0)

    def ReadHebbianValue(self, h: torch.Tensor) -> torch.Tensor:
        hv = self.hebb_value
        y_base = F.linear(h, hv.weight, hv.bias if hv.use_bias else None)
        if not hv.use_hebbian:
            return y_base
        hv.EnsureB(int(h.size(0)), device=h.device, dtype=h.dtype)
        y_hebb = torch.bmm(hv.H, h.unsqueeze(-1)).squeeze(-1)
        return y_base + h.new_tensor(hv.beta_mix) * y_hebb

    def BuildDelayedValueGraph(self, h: torch.Tensor, vMicro: torch.Tensor, hasEdge: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_param = self.value_adapter(h) # [B,1]
        value_model = self.model_value_adapter(h) # [B,1]
        value_quantiles = self.quantile_adapter(h) # [B,Nq]
        dist_stats = self.DistributionStats(value_quantiles)
        value_dist = dist_stats["mean"][:, None] # [B,1]
        value_ensemble = torch.cat([head(h) for head in self.value_ensemble_heads], dim=-1) # [B,K]
        value_td = 0.5 * (v_param + value_dist) # [B,1]
        mix = torch.sigmoid(self.mix_gate(h)).clamp(1e-3, 1.0 - 1e-3) # [B,1]
        v_hebb = self.ReadHebbianValue(h) # [B,1]
        value_base = (1.0 - mix) * value_td + mix * v_hebb # [B,1]
        model_gate = torch.sigmoid(self.model_fusion_gate(h)).clamp(1e-3, 1.0 - 1e-3)
        graph_gate = torch.sigmoid(self.graph_fusion_gate(h)).clamp(1e-3, 1.0 - 1e-3) * hasEdge
        value = value_base + model_gate * (value_model - value_base) + graph_gate * (vMicro - value_base)
        return {
            "value": value,
            "value_td": value_td,
            "value_model": value_model,
            "value_quantiles": value_quantiles,
            "value_ensemble": value_ensemble,}

    def SetDelayedGraphAfterOptimizerStep(self, enabled: bool = True):
        self.delayed_graph_after_optimizer_step = bool(enabled)

    @torch.no_grad()
    def CacheDelayedTransitionInputs(
        self,
        h: torch.Tensor,
        value: torch.Tensor,
        vMicro: torch.Tensor,
        hasEdge: torch.Tensor,
        rewardNext: torch.Tensor,
        discountNext: torch.Tensor,
        alive: torch.Tensor,
        unc: torch.Tensor,
        enabled: Optional[bool] = None):
        if enabled is None:
            enabled = bool(self.training)
        if not enabled:
            return
        self._delay_graph_cache = {
            "h": h.detach(),
            "value": value.detach(),
            "v_micro": vMicro.detach(),
            "has_edge": hasEdge.detach(),
            "reward_next": rewardNext.detach(),
            "discount_next": discountNext.detach(),
            "alive": alive.detach(),
            "unc": unc.detach(),}

    def BuildDelayedTransitionGraph(self, returnExtras: Optional[bool] = None, enabled: Optional[bool] = None):
        if enabled is None:
            enabled = bool(self.training)
        if not enabled:
            return
        cache = self._delay_graph_cache
        if cache is None:
            return
        h = cache["h"]
        delayed_parts = self.BuildDelayedValueGraph(
            h=h,
            vMicro=cache["v_micro"],
            hasEdge=cache["has_edge"])
        value = cache["value"]
        v_next_hat, transp_extras = self.transport(h, value, returnExtras=returnExtras)
        self._prev_h = h
        self._prev_value = value
        self._prev_v_next_pred = v_next_hat
        self._prev_transp_extras = transp_extras
        self._prev_value_td = delayed_parts["value_td"]
        self._prev_value_model = delayed_parts["value_model"]
        self._prev_value_quantiles = delayed_parts["value_quantiles"]
        self._prev_value_ensemble = delayed_parts["value_ensemble"]
        self._prev_reward_next = cache["reward_next"]
        self._prev_discount_next = cache["discount_next"]
        self._prev_alive = cache["alive"]
        self._prev_unc = cache["unc"]
        self._delay_graph_cache = None

    def ReadPrevTransition(
        self,
        B: int,
        value: torch.Tensor,
        includeValueHeads: bool = False,
        enabled: Optional[bool] = None) -> Dict[str, Any]:
        prev = {
            "ready": False,
            "value_heads_ready": False,
            "pred": None,
            "transp_extras": {},
            "valid": value.new_zeros((B,)),
            "target": value.new_zeros((B, 1)),
            "td_err": value.new_zeros((B, 1)),}
        if enabled is None:
            enabled = bool(self.training)
        if not enabled:
            return prev

        pred = self._prev_v_next_pred
        reward_next = self._prev_reward_next
        discount_next = self._prev_discount_next
        alive = self._prev_alive
        if (
            pred is None
            or reward_next is None
            or discount_next is None
            or alive is None
            or pred.size(0) != B):
            return prev

        reward_next = reward_next.to(device=value.device, dtype=value.dtype).view(B)
        discount_next = discount_next.to(device=value.device, dtype=value.dtype).view(B).clamp(0.0, 1.0)
        valid = alive.to(device=value.device, dtype=value.dtype).view(B).clamp(0.0, 1.0)
        target = reward_next[:, None] + discount_next[:, None] * value.detach()
        prev.update({
            "ready": True,
            "pred": pred,
            "transp_extras": self._prev_transp_extras or {},
            "valid": valid,
            "target": target,
            "td_err": target.detach() - pred,})

        if includeValueHeads:
            heads = {
                "value_td": self._prev_value_td,
                "value_model": self._prev_value_model,
                "value_quantiles": self._prev_value_quantiles,
                "value_ensemble": self._prev_value_ensemble,}
            if all(torch.is_tensor(t) and t.size(0) == B for t in heads.values()):
                prev.update(heads)
                prev["value_heads_ready"] = True
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
        gamma: float = 0.99,
        )-> GeoTropicalOut: 

        B = state.size(0)
        self.EnsureB(B, self.device, self.dtype)

        if policyEntropyPrev is None:
            policyEntropyPrev = state.new_zeros(B)
        if worldDeltaTransport is None:
            worldDeltaTransport = state.new_zeros(B, self.unc_core.state_dim)
        if worldDeltaPhysics is None:
            worldDeltaPhysics = state.new_zeros(B, self.unc_core.state_dim)

        if rewardModel is None:
            raise ValueError("ValueEstimationExtractor requires rewardModel")
        if doneModel is None:
            raise ValueError("ValueEstimationExtractor requires doneModel")
        reward_value = rewardModel.to(device=state.device, dtype=state.dtype).view(B)
        done_value = doneModel.to(device=state.device, dtype=state.dtype).view(B).clamp(0.0, 1.0)
        with torch.no_grad():
            r_next_hat = self.reward_perdetic.PredictNext(reward_value.detach()).to(device=state.device, dtype=state.dtype).view(B)
            done_next_hat = self.done_perdetic.PredictNext(done_value.detach()).to(device=state.device, dtype=state.dtype).view(B).clamp(0.0, 1.0)
        gamma_t = state.new_tensor(float(gamma)).clamp(0.0, 1.0)

        x = torch.cat([memoryPrev, attnPrev, state], dim=-1)
        h = self.Trunk(x) # [B,H]

        emotion = self.emotion_core(memoryPrev=memoryPrev, attnPrev=attnPrev, stateCurr=state) # [B,emotionDim]

        v_param = self.value_adapter(h) # [B,1]
        value_model = self.model_value_adapter(h) # [B,1]
        calib_raw = self.calibration_adapter(h) # [B,4]
        value_quantiles = self.quantile_adapter(h) # [B,Nq]
        dist_stats = self.DistributionStats(value_quantiles)
        value_dist = dist_stats["mean"][:, None] # [B,1]
        value_ensemble = torch.cat([head(h) for head in self.value_ensemble_heads], dim=-1) # [B,K]
        value_epistemic = value_ensemble.var(dim=-1, unbiased=False) # [B]
        value_aleatoric = F.softplus(self.aleatoric_head(h)).squeeze(-1).clamp_min(1e-6) # [B]

        mix = torch.sigmoid(self.mix_gate(h)).clamp(1e-3, 1.0 - 1e-3) # [B,1]

        v_hebb, hebb_extras = self.hebb_value(h) # v_hebb:[B,1]

        value_td = 0.5 * (v_param + value_dist) # [B,1]
        value_base = (1.0 - mix) * value_td + mix * v_hebb # [B,1]

        g_next_hat = (gamma_t * (1.0 - done_next_hat)).clamp(0.0, float(gamma_t.item())) # [B]
        with torch.no_grad():
            edges = self.micro.PreviewEdges(zNow=h, rNow=r_next_hat, gNow=g_next_hat)
            w = edges["w"] # [B,K]
            denom = w.sum(dim=1) # [B]
            v_bar = (w * (edges["R"] + edges["Gamma"] * edges["v_hist"])).sum(dim=1) / denom.clamp_min(1e-6) # [B]
            v_bar_B1 = v_bar[:, None] # [B,1]
            has_edge = edges["valid"].any(dim=1).float()[:, None] # [B,1]

        model_gate = torch.sigmoid(self.model_fusion_gate(h)).clamp(1e-3, 1.0 - 1e-3)
        graph_gate = torch.sigmoid(self.graph_fusion_gate(h)).clamp(1e-3, 1.0 - 1e-3) * has_edge
        value = value_base + model_gate * (value_model - value_base) + graph_gate * (v_bar_B1.detach() - value_base)

        v_next_hat, transp_extras = self.transport(h, value) # v_next_hat:[B,1]

        td_target = r_next_hat[:, None] + g_next_hat[:, None] * v_next_hat # [B,1]
        td_current = td_target - value # [B,1]
        if self.training and (not self.delayed_graph_after_optimizer_step):
            self.BuildDelayedTransitionGraph(returnExtras=self.training)
        prev = self.ReadPrevTransition(B, value, includeValueHeads=self.training)
        has_prev_pred = bool(prev["ready"])
        has_prev_value = bool(prev["value_heads_ready"])
        prev_valid = prev["valid"]
        delayed_target = prev["target"]
        td_align_err = prev["td_err"]
        self.td_out_ema.Update(td_current.detach().abs().squeeze(-1))
        td_scale = (self.td_out_ema.mean + 2.0 * (self.td_out_ema.var + 1e-6).sqrt()).clamp_min(self.td_scale_min)  # [B]

        td_bounded = torch.tanh(td_current.squeeze(-1) / td_scale)  # [-1,1], [B]

        unc_total, unc_comps = self.unc_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state,
            entropyPrev=policyEntropyPrev,
            tdCurr=td_bounded.detach(),
            worldDeltaTransport=worldDeltaTransport,
            worldDeltaPhysics=worldDeltaPhysics,
            donePrev=done_next_hat.detach()) # unc_total:[B]
        
        base = float(self.unc_core.eps_prior)
        unc_adj = (unc_total - base).clamp_min(0.0) # [B]
        unc_prior01 = (1.0 - torch.exp(-unc_adj / max(self.unc_tau, 1e-6))).clamp(0.0, 1.0) # [B]
        risk_calib, ambiguity, surprise, confidence_calib, _ = self.BuildCalibration(calib_raw, td_bounded, unc_prior01)
        unc_epistemic01 = (1.0 - torch.exp(-value_epistemic)).clamp(0.0, 1.0) # [B]
        unc_aleatoric01 = (1.0 - torch.exp(-value_aleatoric)).clamp(0.0, 1.0) # [B]
        risk_dist = dist_stats["risk"].detach() # [B]
        risk_gate_in = torch.cat([
            h,
            risk_calib[:, None],
            risk_dist[:, None],
            unc_prior01[:, None],
            unc_epistemic01[:, None],
            unc_aleatoric01[:, None],
            td_bounded.detach().abs()[:, None]], dim=-1)
        risk_gate_in = torch.nan_to_num(risk_gate_in.detach(), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0).contiguous()
        risk_gate_logit = (risk_gate_in * self.risk_fusion_gate.weight).sum(dim=-1) + self.risk_fusion_gate.bias.view(1)
        risk_gate = torch.sigmoid(risk_gate_logit) # [B]
        risk_gate_mix = risk_gate.detach()
        risk = (risk_gate_mix * risk_calib + (1.0 - risk_gate_mix) * risk_dist).clamp(0.0, 1.0) # [B]
        confidence_dist = torch.exp(-(unc_prior01 + unc_epistemic01 + unc_aleatoric01 + risk)).clamp(0.0, 1.0) # [B]
        confidence = (0.5 * confidence_calib + 0.5 * confidence_dist).clamp(0.0, 1.0) # [B]
        learned_unc = (
            0.30 * risk
            + 0.25 * ambiguity
            + 0.20 * surprise
            + 0.25 * (1.0 - confidence)).clamp(0.0, 1.0)
        unc01 = (
            0.35 * unc_prior01
            + 0.25 * learned_unc
            + 0.20 * unc_epistemic01
            + 0.10 * unc_aleatoric01
            + 0.10 * risk).clamp(0.0, 1.0) # [B]
        precision = (confidence * (1.0 - unc01)).clamp(0.05, 1.0) # [B]
        branch_next = transp_extras.get("branch_next")
        transport_branch_std = (
            value.new_zeros((B,))
            if branch_next is None
            else branch_next.detach().std(dim=-1, unbiased=False))

        rComps = {
            "value_base": value_base.detach(),
            "value_td": value_td.detach(),
            "value_model": value_model.detach(),
            "value_dist": value_dist.detach(),
            "v_micro": v_bar_B1.detach(),
            "model_gate": model_gate.detach().squeeze(-1),
            "graph_gate": graph_gate.detach().squeeze(-1),
            "risk": risk.detach(),
            "ambiguity": ambiguity.detach(),
            "surprise": surprise.detach(),
            "confidence": confidence.detach(),
            "unc_total": unc_total.detach(),
            "unc_epistemic": value_epistemic.detach(),
            "unc_aleatoric": value_aleatoric.detach(),
            "value_dist_std": dist_stats["std"].detach(),
            "value_dist_cvar": dist_stats["cvar"].detach(),
            "transport_branch_std": transport_branch_std,
            "reward_basis": r_next_hat.detach(),
            "done_basis": done_next_hat.detach(),
            "td_target": td_target.detach(),
            "delayed_td_target": delayed_target.detach(),}

        with torch.no_grad():
            self.micro.CommitStep(zNow=h, vNow=value, rNow=r_next_hat, gNow=g_next_hat)

        if not self.training:
            return GeoTropicalOut(
                value=value,
                tdError=td_bounded.detach(),
                loss=None,
                emotion=emotion,
                rComps=rComps,
                uncertainty=unc01,
                precision=precision,
                extras=None,)

        loss_td = value.new_zeros(())
        loss_model = value.new_zeros(())
        loss_ensemble = value.new_zeros(())
        loss_dist = value.new_zeros(())
        valid_denom = prev_valid.sum().clamp_min(1.0)

        def ValidMean(vec: torch.Tensor) -> torch.Tensor:
            return (vec.view(B) * prev_valid).sum() / valid_denom

        if has_prev_value:
            loss_td_vec = F.smooth_l1_loss(prev["value_td"], delayed_target.detach(), reduction="none").squeeze(-1) # [B]
            w_td_raw = (1.0 / (unc_total.detach().sqrt() + 1e-3)).clamp(0.25, 4.0) # [B]
            w_td_mean = (w_td_raw * prev_valid).sum() / valid_denom
            w_td = w_td_raw / w_td_mean.clamp_min(1e-6)
            loss_td = ValidMean(loss_td_vec * w_td)
            loss_model_vec = F.smooth_l1_loss(prev["value_model"], delayed_target.detach(), reduction="none").squeeze(-1)
            loss_model = ValidMean(loss_model_vec)
            loss_ensemble_vec = F.smooth_l1_loss(
                prev["value_ensemble"],
                delayed_target.detach().expand_as(prev["value_ensemble"]),
                reduction="none").mean(dim=-1)
            loss_ensemble = value.new_tensor(0.01) * ValidMean(loss_ensemble_vec)
            loss_dist = (
                value.new_tensor(0.05) * self.QuantileHuberLoss(prev["value_quantiles"], delayed_target.detach(), sampleWeight=prev_valid)
                + value.new_tensor(1e-3) * self.QuantileCrossingLoss(prev["value_quantiles"], sampleWeight=prev_valid))

        loss_trans = value.new_zeros(())
        if has_prev_pred:
            prev_pred = prev["pred"]
            prev_transp_extras = prev["transp_extras"]
            loss_trans_vec = F.smooth_l1_loss(prev_pred, delayed_target.detach(), reduction="none").squeeze(-1)
            loss_trans = ValidMean(loss_trans_vec)
            if "branch_next" in prev_transp_extras:
                branch_target = delayed_target.detach().expand_as(prev_transp_extras["branch_next"])
                branch_loss_vec = F.smooth_l1_loss(
                    prev_transp_extras["branch_next"],
                    branch_target,
                    reduction="none").mean(dim=-1)
                loss_trans = loss_trans + 0.25 * ValidMean(branch_loss_vec)

        micro_err = F.smooth_l1_loss(value, v_bar_B1.detach(), reduction="none") * has_edge # [B,1]
        loss_micro = micro_err.sum() / has_edge.sum().clamp_min(1.0)

        loss_git = self.git(self.value_head, transp_extras, adapter=self.value_adapter)  

        loss_mix = value.new_tensor(self.wMixGateReg) * (mix - 0.11920292202211755).pow(2).mean()
        loss_gate = value.new_tensor(self.wMixGateReg) * (
            model_gate.pow(2).mean()
            + graph_gate.pow(2).mean()
            + risk_gate.pow(2).mean())

        loss_hebb_wd = value.new_tensor(1e-6) * (self.hebb_value.weight.pow(2).mean())

        td_sq_det = td_align_err.detach().squeeze(-1).pow(2) # [B]
        ale_safe = value_aleatoric.clamp_min(1e-6) # [B]
        if has_prev_pred:
            loss_aleatoric_vec = 0.5 * ((td_sq_det / ale_safe) + torch.log(ale_safe)) + 2.0
            loss_aleatoric = value.new_tensor(0.01 * self.w_unc) * ValidMean(loss_aleatoric_vec)
        else:
            loss_aleatoric = value.new_zeros(())
        if has_prev_pred:
            unc_pred = (0.25 * (risk + ambiguity + surprise + (1.0 - confidence))).clamp(0.0, 1.0)
            unc_target = (1.0 - torch.exp(-td_sq_det.detach())).clamp(0.0, 1.0)
            loss_unc_vec = F.smooth_l1_loss(unc_pred, unc_target, reduction="none")
            loss_unc = value.new_tensor(self.w_unc) * (
                (loss_unc_vec * prev_valid).sum() / valid_denom)
        else:
            loss_unc = value.new_zeros(())

        self.CacheDelayedTransitionInputs(
            h=h,
            value=value,
            vMicro=v_bar_B1,
            hasEdge=has_edge,
            rewardNext=r_next_hat,
            discountNext=g_next_hat,
            alive=(1.0 - done_next_hat),
            unc=unc_total)

        loss = (
            loss_td
            + 0.01 * loss_trans
            + 0.01 * loss_micro
            + 0.05 * loss_model
            + loss_ensemble
            + loss_dist
            + loss_git
            + loss_mix
            + loss_gate
            + loss_hebb_wd
            + loss_aleatoric
            + loss_unc)

        extras = {
            "loss_td": loss_td.detach(),
            "loss_trans": loss_trans.detach(),
            "loss_micro": loss_micro.detach(),
            "loss_model": loss_model.detach(),
            "loss_ensemble": loss_ensemble.detach(),
            "loss_dist": loss_dist.detach(),
            "loss_git": loss_git.detach(),
            "loss_mix": loss_mix.detach(),
            "loss_gate": loss_gate.detach(),
            "loss_hebb_wd": loss_hebb_wd.detach(),
            "loss_aleatoric": loss_aleatoric.detach(),
            "loss_unc": loss_unc.detach(),
            "mix_mean": mix.detach().mean(),
            "td_err_abs_mean": ((td_align_err.detach().abs().squeeze(-1) * prev_valid).sum() / valid_denom),}

        return GeoTropicalOut(
            value=value,
            tdError=td_bounded.detach(),
            loss=loss,
            emotion=emotion,
            rComps=rComps,
            uncertainty=unc01,
            precision=precision,
            extras=extras,)


    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None):
        self.hebb_value.ResetHebbianMemory(doneMask=doneMask)
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
            self._prev_h = None
            self._prev_value = None
            self._prev_v_next_pred = None
            self._prev_transp_extras = {}
            self._prev_value_td = None
            self._prev_value_model = None
            self._prev_value_quantiles = None
            self._prev_value_ensemble = None
            self._prev_reward_next = None
            self._prev_discount_next = None
            self._prev_alive = None
            self._prev_unc = None
            self._delay_graph_cache = None
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
        refs = [
            self._prev_h,
            self._prev_value,
            self._prev_v_next_pred,
            self._prev_value_td,
            self._prev_value_model,
            self._prev_value_quantiles,
            self._prev_value_ensemble,
            self._prev_reward_next,
            self._prev_discount_next,
            self._prev_alive,
            self._prev_unc]
        ref = next((t for t in refs if torch.is_tensor(t)), None)
        if ref is None:
            return
        mask = doneMask.to(device=ref.device, dtype=torch.bool).view(-1)
        n = min(int(mask.numel()), int(ref.size(0)))
        if n <= 0:
            return
        rows = mask[:n].nonzero(as_tuple=False).view(-1)
        if rows.numel() <= 0:
            return
        for name in [
            "_prev_h",
            "_prev_value",
            "_prev_v_next_pred",
            "_prev_value_td",
            "_prev_value_model",
            "_prev_value_quantiles",
            "_prev_value_ensemble",
            "_prev_reward_next",
            "_prev_discount_next",
            "_prev_alive",
            "_prev_unc"]:
            t = getattr(self, name, None)
            if torch.is_tensor(t) and t.size(0) >= n:
                t[rows] = 0

    @torch.no_grad()
    def ExportState(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"ve_is_training": bool(self.training)}

        prev_state_keys = [
            "_prev_h",
            "_prev_value",
            "_prev_v_next_pred",
            "_prev_value_td",
            "_prev_value_model",
            "_prev_value_quantiles",
            "_prev_value_ensemble",
            "_prev_reward_next",
            "_prev_discount_next",
            "_prev_alive",
            "_prev_unc"]
        for k in prev_state_keys:
            v = getattr(self, k, None)
            state[f"ve{k}"] = (None if v is None else v.detach().clone())

        if hasattr(self, "td_out_ema"):
            state["td_out_ema_mean"] = self.td_out_ema.mean.detach().clone()
            state["td_out_ema_var"] = self.td_out_ema.var.detach().clone()

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
            if hasattr(ec, "fast_head") and hasattr(ec.fast_head, "H"):
                state["emo_fast_H"] = ec.fast_head.H.detach().clone()
            if hasattr(ec, "slow_head") and hasattr(ec.slow_head, "H"):
                state["emo_slow_H"] = ec.slow_head.H.detach().clone()

        if hasattr(self, "unc_core"):
            uc = self.unc_core
            for name in ["td_ema", "ent_ema", "state_ema", "tr_ema", "ph_ema", "ctx_ema"]:
                if hasattr(uc, name):
                    ema = getattr(uc, name)
                    if hasattr(ema, "mean"):
                        state[f"unc_{name}_mean"] = ema.mean.detach().clone()
                    if hasattr(ema, "var"):
                        state[f"unc_{name}_var"] = ema.var.detach().clone()

        for prefix, pred in [("reward_pred", getattr(self, "reward_perdetic", None)),
                             ("done_pred", getattr(self, "done_perdetic", None))]:
            if pred is None:
                continue
            for n in ["kf_mean", "kf_var", "smooth_hist"]:
                if hasattr(pred, n):
                    t = getattr(pred, n)
                    if torch.is_tensor(t):
                        state[f"{prefix}_{n}"] = t.detach().clone()
            for n in ["predict_mode", "auto_policy", "auto_temperature", "fit_last_n"]:
                if hasattr(pred, n):
                    state[f"{prefix}_{n}"] = getattr(pred, n)

        if hasattr(self, "micro"):
            mg = self.micro
            for n in ["prefix_G", "prefix_C", "anchor_z", "anchor_v", "anchor_G", "anchor_C", "filled", "ptr"]:
                if hasattr(mg, n):
                    t = getattr(mg, n)
                    if torch.is_tensor(t):
                        state[f"micro_{n}"] = t.detach().clone()
            state["micro_step"] = int(getattr(mg, "_step", 0))

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
            if not hasattr(obj, name):
                return
            cur = getattr(obj, name)
            if not torch.is_tensor(cur):
                setattr(obj, name, t.clone())
                return
            v = t.to(device=cur.device, dtype=cur.dtype)
            if cur.shape != v.shape:
                cur.resize_(v.shape).copy_(v)
            else:
                cur.copy_(v)

        ref_dev = self.value_head.weight.device
        ref_dt = self.value_head.weight.dtype
        for k in [
            "_prev_h",
            "_prev_value",
            "_prev_v_next_pred",
            "_prev_value_td",
            "_prev_value_model",
            "_prev_value_quantiles",
            "_prev_value_ensemble",
            "_prev_reward_next",
            "_prev_discount_next",
            "_prev_alive",
            "_prev_unc"]:
            kk = f"ve{k}"
            if kk in state:
                v = state[kk]
                setattr(self, k, None if v is None else v.to(device=ref_dev, dtype=ref_dt).clone())
        if ("_prev_unc" not in self.__dict__ or self._prev_unc is None) and ("ve_prev_unc" in state):
            v = state["ve_prev_unc"]
            self._prev_unc = None if v is None else v.to(device=ref_dev, dtype=ref_dt).clone()

        if hasattr(self, "hebb_value") and hasattr(self.hebb_value, "H"):
            copy_tensor_attr(self.hebb_value, "H", need_("hebb_H"))

        if hasattr(self, "td_out_ema"):
            copy_tensor_attr(self.td_out_ema, "mean", need_("td_out_ema_mean"))
            copy_tensor_attr(self.td_out_ema, "var", need_("td_out_ema_var"))

        if hasattr(self, "emotion_core"):
            ec = self.emotion_core
            if need_("emo_h") is not None:
                ec.h = need_("emo_h").to(device=ec.h.device, dtype=ec.h.dtype).clone()
            if need_("emo_c") is not None:
                ec.c = need_("emo_c").to(device=ec.c.device, dtype=ec.c.dtype).clone()
            if need_("emo_mood") is not None:
                ec.mood = need_("emo_mood").to(device=ec.mood.device, dtype=ec.mood.dtype).clone()
            if hasattr(ec, "fast_head") and hasattr(ec.fast_head, "H"):
                copy_tensor_attr(ec.fast_head, "H", need_("emo_fast_H"))
            if hasattr(ec, "slow_head") and hasattr(ec.slow_head, "H"):
                copy_tensor_attr(ec.slow_head, "H", need_("emo_slow_H"))

        if hasattr(self, "unc_core"):
            uc = self.unc_core
            for name in ["td_ema", "ent_ema", "state_ema", "tr_ema", "ph_ema", "ctx_ema"]:
                if hasattr(uc, name):
                    ema = getattr(uc, name)
                    copy_tensor_attr(ema, "mean", need_(f"unc_{name}_mean"))
                    copy_tensor_attr(ema, "var", need_(f"unc_{name}_var"))

        for prefix, pred in [("reward_pred", getattr(self, "reward_perdetic", None)),
                             ("done_pred", getattr(self, "done_perdetic", None))]:
            if pred is None:
                continue
            for n in ["kf_mean", "kf_var", "smooth_hist"]:
                copy_tensor_attr(pred, n, need_(f"{prefix}_{n}"))
            for n in ["predict_mode", "auto_policy", "auto_temperature", "fit_last_n"]:
                k = f"{prefix}_{n}"
                if k in state and hasattr(pred, n):
                    setattr(pred, n, state[k])

        if hasattr(self, "micro"):
            mg = self.micro
            for n in ["prefix_G", "prefix_C", "anchor_z", "anchor_v", "anchor_G", "anchor_C", "filled", "ptr"]:
                copy_tensor_attr(mg, n, need_(f"micro_{n}"))
            if "micro_step" in state:
                mg._step = int(state["micro_step"])


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
        maxRankVHead: int = 64,
        maxRankModelVHead: int = 64,
        maxRankQuantile: int = 64,
        maxRankCalib: int = 32,):
        self.maxRankFc1 = int(maxRankFc1)
        self.maxRankFc2 = int(maxRankFc2)
        self.maxRankVHead = int(maxRankVHead)
        self.maxRankModelVHead = int(maxRankModelVHead)
        self.maxRankQuantile = int(maxRankQuantile)
        self.maxRankCalib = int(maxRankCalib)
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
        if isinstance(x, dict) and all(k in x for k in ("memoryPrev", "attnPrev", "state")):
            return x["memoryPrev"], x["attnPrev"], x["state"]
        raise TypeError("ValueEstimationOnlineWrapper expects x as (memoryPrev, attnPrev, state) or dict with those keys.")

    def SetDelayedGraphAfterOptimizerStep(self, enabled: bool = True):
        if hasattr(self.base, "SetDelayedGraphAfterOptimizerStep"):
            self.base.SetDelayedGraphAfterOptimizerStep(enabled)

    def BuildDelayedTransitionGraph(self):
        if hasattr(self.base, "BuildDelayedTransitionGraph"):
            return self.base.BuildDelayedTransitionGraph(returnExtras=self.training, enabled=self.training)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        base = self.base
        assert hasattr(base, "fc1") and hasattr(base, "fc2") and hasattr(base, "value_head")
        assert hasattr(base, "model_value_head") and hasattr(base, "calibration_head") and hasattr(base, "quantile_head")

        H = int(base.value_head.in_features)
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
            "vhead": SiteSpec("vhead", L, H, 1, self.maxRankVHead, lambda r, dv, dt: alloc(r, H, 1, dv, dt), compose),
            "model_vhead": SiteSpec("model_vhead", L, H, 1, self.maxRankModelVHead, lambda r, dv, dt: alloc(r, H, 1, dv, dt), compose),
            "qhead": SiteSpec("qhead", L, H, int(base.quantile_head.out_features), self.maxRankQuantile, lambda r, dv, dt: alloc(r, H, int(base.quantile_head.out_features), dv, dt), compose),
            "calib": SiteSpec("calib", L, H, 4, self.maxRankCalib, lambda r, dv, dt: alloc(r, H, 4, dv, dt), compose),}

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
        policyEntropyPrev = kwargs.get("policyEntropyPrev", None)
        worldDeltaTransport = kwargs.get("worldDeltaTransport", None)
        worldDeltaPhysics = kwargs.get("worldDeltaPhysics", None)

        miss = []
        if policyEntropyPrev is None:
            miss.append("policyEntropyPrev")
        if worldDeltaTransport is None:
            miss.append("worldDeltaTransport")
        if worldDeltaPhysics is None:
            miss.append("worldDeltaPhysics")
        if rewardModel is None:
            miss.append("rewardModel")
        if doneModel is None:
            miss.append("doneModel")
        if len(miss) > 0:
            raise ValueError(f"ValueEstimationOnlineWrapper missing required kwargs: {', '.join(miss)}")

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
            v_param_in = self.LinearWithDelta(
                base.value_head,
                h_in,
                delta_mat=d1.get("vhead", None),
                base_adapter=getattr(base, "value_adapter", None),)
            value_model_in = self.LinearWithDelta(
                base.model_value_head,
                h_in,
                delta_mat=d1.get("model_vhead", None),
                base_adapter=getattr(base, "model_value_adapter", None),)
            calib_raw_in = self.LinearWithDelta(
                base.calibration_head,
                h_in,
                delta_mat=d1.get("calib", None),
                base_adapter=getattr(base, "calibration_adapter", None),)
            value_quantiles_in = self.LinearWithDelta(
                base.quantile_head,
                h_in,
                delta_mat=d1.get("qhead", None),
                base_adapter=getattr(base, "quantile_adapter", None),)
            dist_stats_in = base.DistributionStats(value_quantiles_in)
            value_dist_in = dist_stats_in["mean"][:, None]
            value_ensemble_in = torch.cat([head(h_in) for head in base.value_ensemble_heads], dim=-1)
            value_epistemic_in = value_ensemble_in.var(dim=-1, unbiased=False)
            value_aleatoric_in = F.softplus(base.aleatoric_head(h_in)).squeeze(-1).clamp_min(1e-6)
            value_td_in = 0.5 * (v_param_in + value_dist_in)
            return {
                "v_param": v_param_in,
                "value_model": value_model_in,
                "calib_raw": calib_raw_in,
                "value_quantiles": value_quantiles_in,
                "dist_stats": dist_stats_in,
                "value_dist": value_dist_in,
                "value_ensemble": value_ensemble_in,
                "value_epistemic": value_epistemic_in,
                "value_aleatoric": value_aleatoric_in,
                "value_td": value_td_in,}

        x_cat = torch.cat([memoryPrev, attnPrev, state], dim=-1)

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
        if hasattr(base, "trunk_res_blocks"):
            for blk in base.trunk_res_blocks:
                h = blk(h)

        emotion = base.emotion_core(memoryPrev=memoryPrev, attnPrev=attnPrev, stateCurr=state)

        value_parts = BuildOnlineParametricValueBranches(h)
        value_model = value_parts["value_model"]
        calib_raw = value_parts["calib_raw"]
        value_quantiles = value_parts["value_quantiles"]
        dist_stats = value_parts["dist_stats"]
        value_dist = value_parts["value_dist"]
        value_ensemble = value_parts["value_ensemble"]
        value_epistemic = value_parts["value_epistemic"]
        value_aleatoric = value_parts["value_aleatoric"]

        if rewardModel is None:
            raise ValueError("ValueEstimationOnlineWrapper requires rewardModel")
        if doneModel is None:
            raise ValueError("ValueEstimationOnlineWrapper requires doneModel")
        reward_value = rewardModel.detach().to(device=state.device, dtype=state.dtype).view(B)
        done_value = doneModel.detach().to(device=state.device, dtype=state.dtype).view(B).clamp(0.0, 1.0)
        with torch.no_grad():
            r_next_hat = base.reward_perdetic.PredictNext(reward_value.detach()).to(device=state.device, dtype=state.dtype).view(B)
            done_next_hat = base.done_perdetic.PredictNext(done_value.detach()).to(device=state.device, dtype=state.dtype).view(B).clamp(0.0, 1.0)
        gamma_t = state.new_tensor(float(gamma)).clamp(0.0, 1.0)

        mix = torch.sigmoid(base.mix_gate(h)).clamp(1e-3, 1.0 - 1e-3) 
        v_hebb, _ = base.hebb_value(h)
        value_td = value_parts["value_td"]
        value_base = (1.0 - mix) * value_td + mix * v_hebb

        g_next_hat = (gamma_t * (1.0 - done_next_hat)).clamp(0.0, float(gamma_t.item()))
        with torch.no_grad():
            edges = base.micro.PreviewEdges(zNow=h, rNow=r_next_hat, gNow=g_next_hat)
            w = edges["w"]
            denom = w.sum(dim=1)
            v_bar = (w * (edges["R"] + edges["Gamma"] * edges["v_hist"])).sum(dim=1) / denom.clamp_min(1e-6)
            v_bar_B1 = v_bar[:, None]
            has_edge = edges["valid"].any(dim=1).float()[:, None]

        model_gate = torch.sigmoid(base.model_fusion_gate(h)).clamp(1e-3, 1.0 - 1e-3)
        graph_gate = torch.sigmoid(base.graph_fusion_gate(h)).clamp(1e-3, 1.0 - 1e-3) * has_edge
        value = value_base + model_gate * (value_model - value_base) + graph_gate * (v_bar_B1.detach() - value_base)

        v_next_hat, transp_extras = base.transport(h, value, returnExtras=self.training)
        td_target = r_next_hat[:, None] + g_next_hat[:, None] * v_next_hat
        td_current = td_target - value
        if self.training and (not base.delayed_graph_after_optimizer_step):
            base.BuildDelayedTransitionGraph(returnExtras=self.training, enabled=self.training)
        prev = base.ReadPrevTransition(B, value, includeValueHeads=self.training, enabled=self.training)
        has_prev_pred = bool(prev["ready"])
        has_prev_value = bool(prev["value_heads_ready"])
        prev_valid = prev["valid"]
        delayed_target = prev["target"]
        td_align_err = prev["td_err"]
        base.td_out_ema.Update(td_current.detach().abs().squeeze(-1))
        td_scale = (base.td_out_ema.mean + 2.0 * (base.td_out_ema.var + 1e-6).sqrt()).clamp_min(base.td_scale_min)
        td_bounded = torch.tanh(td_current.squeeze(-1) / td_scale)

        unc_total, unc_comps = base.unc_core(
            memoryPrev=memoryPrev,
            attnPrev=attnPrev,
            stateCurr=state,
            entropyPrev=policyEntropyPrev,
            tdCurr=td_bounded.detach(),
            worldDeltaTransport=worldDeltaTransport,
            worldDeltaPhysics=worldDeltaPhysics,
            donePrev=done_next_hat.detach())
        
        unc_base = float(base.unc_core.eps_prior)
        unc_adj = (unc_total - unc_base).clamp_min(0.0)
        unc_prior01 = (1.0 - torch.exp(-unc_adj / max(base.unc_tau, 1e-6))).clamp(0.0, 1.0)
        risk_calib, ambiguity, surprise, confidence_calib, _ = base.BuildCalibration(calib_raw, td_bounded, unc_prior01)
        unc_epistemic01 = (1.0 - torch.exp(-value_epistemic)).clamp(0.0, 1.0)
        unc_aleatoric01 = (1.0 - torch.exp(-value_aleatoric)).clamp(0.0, 1.0)
        risk_dist = dist_stats["risk"].detach()
        risk_gate_in = torch.cat([
            h,
            risk_calib[:, None],
            risk_dist[:, None],
            unc_prior01[:, None],
            unc_epistemic01[:, None],
            unc_aleatoric01[:, None],
            td_bounded.detach().abs()[:, None]], dim=-1)
        risk_gate_in = torch.nan_to_num(risk_gate_in.detach(), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0).contiguous()
        risk_gate_logit = (risk_gate_in * base.risk_fusion_gate.weight).sum(dim=-1) + base.risk_fusion_gate.bias.view(1)
        risk_gate = torch.sigmoid(risk_gate_logit)
        risk_gate_mix = risk_gate.detach()
        risk = (risk_gate_mix * risk_calib + (1.0 - risk_gate_mix) * risk_dist).clamp(0.0, 1.0)
        confidence_dist = torch.exp(-(unc_prior01 + unc_epistemic01 + unc_aleatoric01 + risk)).clamp(0.0, 1.0)
        confidence = (0.5 * confidence_calib + 0.5 * confidence_dist).clamp(0.0, 1.0)
        learned_unc = (
            0.30 * risk
            + 0.25 * ambiguity
            + 0.20 * surprise
            + 0.25 * (1.0 - confidence)).clamp(0.0, 1.0)
        unc01 = (
            0.35 * unc_prior01
            + 0.25 * learned_unc
            + 0.20 * unc_epistemic01
            + 0.10 * unc_aleatoric01
            + 0.10 * risk).clamp(0.0, 1.0)
        precision = (confidence * (1.0 - unc01)).clamp(0.05, 1.0)
        branch_next = transp_extras.get("branch_next")
        transport_branch_std = (
            value.new_zeros((B,))
            if branch_next is None
            else branch_next.detach().std(dim=-1, unbiased=False))

        rComps = {
            "value_base": value_base.detach(),
            "value_td": value_td.detach(),
            "value_model": value_model.detach(),
            "value_dist": value_dist.detach(),
            "v_micro": v_bar_B1.detach(),
            "model_gate": model_gate.detach().squeeze(-1),
            "graph_gate": graph_gate.detach().squeeze(-1),
            "risk": risk.detach(),
            "ambiguity": ambiguity.detach(),
            "surprise": surprise.detach(),
            "confidence": confidence.detach(),
            "unc_total": unc_total.detach(),
            "unc_epistemic": value_epistemic.detach(),
            "unc_aleatoric": value_aleatoric.detach(),
            "value_dist_std": dist_stats["std"].detach(),
            "value_dist_cvar": dist_stats["cvar"].detach(),
            "transport_branch_std": transport_branch_std,
            "reward_basis": r_next_hat.detach(),
            "done_basis": done_next_hat.detach(),
            "td_target": td_target.detach(),
            "delayed_td_target": delayed_target.detach(),}

        with torch.no_grad():
            base.micro.CommitStep(zNow=h, vNow=value, rNow=r_next_hat, gNow=g_next_hat)

        if not self.training:
            return GeoTropicalOut(
                value=value,
                tdError=td_bounded.detach(),
                loss=None,
                emotion=emotion,
                rComps=rComps,
                uncertainty=unc01,
                precision=precision,
                extras=None,)
        
        loss_td = value.new_zeros(())
        loss_model = value.new_zeros(())
        loss_ensemble = value.new_zeros(())
        loss_dist = value.new_zeros(())
        valid_denom = prev_valid.sum().clamp_min(1.0)

        def ValidMean(vec: torch.Tensor) -> torch.Tensor:
            return (vec.view(B) * prev_valid).sum() / valid_denom

        if has_prev_value:
            loss_td_vec = F.smooth_l1_loss(prev["value_td"], delayed_target.detach(), reduction="none").squeeze(-1)
            w_td_raw = (1.0 / (unc_total.detach().sqrt() + 1e-3)).clamp(0.25, 4.0)
            w_td_mean = (w_td_raw * prev_valid).sum() / valid_denom
            w_td = w_td_raw / w_td_mean.clamp_min(1e-6)
            loss_td = ValidMean(loss_td_vec * w_td)
            loss_model_vec = F.smooth_l1_loss(prev["value_model"], delayed_target.detach(), reduction="none").squeeze(-1)
            loss_model = ValidMean(loss_model_vec)
            loss_ensemble_vec = F.smooth_l1_loss(
                prev["value_ensemble"],
                delayed_target.detach().expand_as(prev["value_ensemble"]),
                reduction="none").mean(dim=-1)
            loss_ensemble = value.new_tensor(0.01) * ValidMean(loss_ensemble_vec)
            loss_dist = (
                value.new_tensor(0.05) * base.QuantileHuberLoss(prev["value_quantiles"], delayed_target.detach(), sampleWeight=prev_valid)
                + value.new_tensor(1e-3) * base.QuantileCrossingLoss(prev["value_quantiles"], sampleWeight=prev_valid))

        loss_trans = value.new_zeros(())
        if has_prev_pred:
            prev_pred = prev["pred"]
            prev_transp_extras = prev["transp_extras"]
            loss_trans_vec = F.smooth_l1_loss(prev_pred, delayed_target.detach(), reduction="none").squeeze(-1)
            loss_trans = ValidMean(loss_trans_vec)
            if "branch_next" in prev_transp_extras:
                branch_target = delayed_target.detach().expand_as(prev_transp_extras["branch_next"])
                branch_loss_vec = F.smooth_l1_loss(
                    prev_transp_extras["branch_next"],
                    branch_target,
                    reduction="none").mean(dim=-1)
                loss_trans = loss_trans + 0.25 * ValidMean(branch_loss_vec)

        micro_err = F.smooth_l1_loss(value, v_bar_B1.detach(), reduction="none") * has_edge
        loss_micro = micro_err.sum() / has_edge.sum().clamp_min(1.0)

        def GitLossWithDelta(deltaMat: Optional[torch.Tensor]) -> torch.Tensor:
            W = base.value_head.weight
            adapter = getattr(base, "value_adapter", None)
            if (adapter is not None) and hasattr(adapter, "DeltaWeight"):
                base_delta = adapter.DeltaWeight()
                if base_delta is not None:
                    W = W + base_delta
            if deltaMat is not None:
                W = W + deltaMat

            reg = W.new_zeros(())
            fro = torch.linalg.matrix_norm(W, ord="fro")
            fro_n = fro / (W.numel() ** 0.5 + base.git.eps)
            reg = reg + base.git.w_scale * (fro_n - 1.0).pow(2)

            if "b" in transp_extras and transp_extras["b"] is not None:
                reg = reg + base.git.w_shift * transp_extras["b"].pow(2).mean()

            if W.numel() > 0:
                row_score = W.abs().amax(dim=1)
                tau = max(base.git.tau_sign, 1e-6)
                row_w = torch.softmax(row_score / tau, dim=0)
                row_mean = W.mean(dim=1)
                soft_row_mean = (row_w * row_mean).sum()
                reg = reg + base.git.w_sign * F.relu(-soft_row_mean)
            return reg

        loss_git = GitLossWithDelta(d1.get("vhead", None))

        loss_mix = value.new_tensor(base.wMixGateReg) * (mix - 0.11920292202211755).pow(2).mean()
        loss_gate = value.new_tensor(base.wMixGateReg) * (
            model_gate.pow(2).mean()
            + graph_gate.pow(2).mean()
            + risk_gate.pow(2).mean())
        loss_hebb_wd = value.new_tensor(1e-6) * (base.hebb_value.weight.pow(2).mean())

        td_sq_det = td_align_err.detach().squeeze(-1).pow(2)
        ale_safe = value_aleatoric.clamp_min(1e-6)
        if has_prev_pred:
            loss_aleatoric_vec = 0.5 * ((td_sq_det / ale_safe) + torch.log(ale_safe)) + 2.0
            loss_aleatoric = value.new_tensor(0.01 * base.w_unc) * ValidMean(loss_aleatoric_vec)
        else:
            loss_aleatoric = value.new_zeros(())
        if has_prev_pred:
            unc_pred = (0.25 * (risk + ambiguity + surprise + (1.0 - confidence))).clamp(0.0, 1.0)
            unc_target = (1.0 - torch.exp(-td_sq_det.detach())).clamp(0.0, 1.0)
            loss_unc_vec = F.smooth_l1_loss(unc_pred, unc_target, reduction="none")
            loss_unc = value.new_tensor(base.w_unc) * (
                (loss_unc_vec * prev_valid).sum() / valid_denom)
        else:
            loss_unc = value.new_zeros(())

        base.CacheDelayedTransitionInputs(
            h=h,
            value=value,
            vMicro=v_bar_B1,
            hasEdge=has_edge,
            rewardNext=r_next_hat,
            discountNext=g_next_hat,
            alive=(1.0 - done_next_hat),
            unc=unc_total,
            enabled=self.training)

        total_loss = (
            loss_td
            + 0.01 * loss_trans
            + 0.01 * loss_micro
            + 0.05 * loss_model
            + loss_ensemble
            + loss_dist
            + loss_git
            + loss_mix
            + loss_gate
            + loss_hebb_wd
            + loss_aleatoric
            + loss_unc)

        extras: Dict[str, torch.Tensor] = {
            "loss_td": loss_td.detach(),
            "loss_trans": loss_trans.detach(),
            "loss_micro": loss_micro.detach(),
            "loss_model": loss_model.detach(),
            "loss_ensemble": loss_ensemble.detach(),
            "loss_dist": loss_dist.detach(),
            "loss_git": loss_git.detach(),
            "loss_mix": loss_mix.detach(),
            "loss_gate": loss_gate.detach(),
            "loss_hebb_wd": loss_hebb_wd.detach(),
            "loss_aleatoric": loss_aleatoric.detach(),
            "loss_unc": loss_unc.detach(),
            "mix_mean": mix.detach().mean(),
            "td_err_abs_mean": ((td_align_err.detach().abs().squeeze(-1) * prev_valid).sum() / valid_denom),}

        return GeoTropicalOut(
            value=value,
            tdError=td_bounded.detach(),
            loss=total_loss,
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
            "vhead": ("value_adapter", "value_head", [1]),
            "model_vhead": ("model_value_adapter", "model_value_head", [1]),
            "qhead": ("quantile_adapter", "quantile_head", [1]),
            "calib": ("calibration_adapter", "calibration_head", [1]),}

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
            "value_base",
            "value_td",
            "value_model",
            "value_dist",
            "v_micro",
            "model_gate",
            "graph_gate",
            "risk",
            "ambiguity",
            "surprise",
            "confidence",
            "unc_total",
            "unc_epistemic",
            "unc_aleatoric",
            "value_dist_std",
            "value_dist_cvar",
            "transport_branch_std",
            "reward_basis",
            "done_basis",
            "td_target",}

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
            ok &= (out_eval.value.shape == (B, 1))
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
            ok &= "loss_td" in out_t2.extras
            ok &= "loss_trans" in out_t2.extras
            ok &= "loss_micro" in out_t2.extras
            ok &= "loss_unc" in out_t2.extras
            for name in self.MonitorKeys():
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
            ok &= (out.value.shape == (B, 1))
            ok &= (out.tdError.shape == (B,))
            ok &= (out.loss.shape == ())
            ok &= (out.emotion.shape[0] == B)
            ok &= (out.uncertainty.shape == (B,))
            ok &= (out.precision.shape == (B,))
            for name in ["risk", "ambiguity", "surprise", "confidence", "unc_total",
                         "unc_epistemic", "unc_aleatoric", "value_dist_std",
                         "value_dist_cvar", "transport_branch_std", "reward_basis", "done_basis"]:
                ok &= (out.rComps is not None and out.rComps[name].shape == (B,))
            for name in ["value_base", "value_td", "value_model", "value_dist", "v_micro", "td_target"]:
                ok &= (out.rComps is not None and out.rComps[name].shape == (B, 1))
            for name in ["model_gate", "graph_gate"]:
                ok &= (out.rComps is not None and out.rComps[name].shape == (B,))
            ok &= (out.extras is not None and torch.is_tensor(out.extras["loss_td"]))

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
            for name in ["risk", "ambiguity", "surprise", "confidence"]:
                comp = out.rComps[name]
                ok &= torch.isfinite(comp).all().item()
                ok &= (float(comp.min().item()) >= -1e-6)
                ok &= (float(comp.max().item()) <= 1.0 + 1e-6)

            print(f"TDUncertaintyBounds {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"TDUncertaintyBounds error: {e}")
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
            ok &= torch.allclose(out_model.rComps["reward_basis"], reward_model)
            ok &= torch.allclose(out_model.rComps["done_basis"], done_model)
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
            pending_after_t1 = est._delay_graph_cache is not None
            out2 = self.ForwardOnce(est, mem2, attn2, state2, reward2, entropy2, done2, d_tr2, d_ph2)
            prev_pred = est._prev_v_next_pred.detach().clone()
            prev_reward_next = est._prev_reward_next.detach().clone()
            prev_discount_next = est._prev_discount_next.detach().clone()
            prev_valid = est._prev_alive.detach().clone().clamp(0.0, 1.0)

            td_scale = (est.td_out_ema.mean + 2.0 * (est.td_out_ema.var + 1e-6).sqrt()).clamp_min(est.td_scale_min)
            expected_td = torch.tanh((out2.rComps["td_target"] - out2.value.detach()).squeeze(-1) / td_scale)
            expected_delayed_target = prev_reward_next[:, None] + prev_discount_next[:, None] * out2.value.detach()
            delayed_raw = (expected_delayed_target - prev_pred).squeeze(-1)
            expected_delayed_abs = (delayed_raw.abs() * prev_valid).sum() / prev_valid.sum().clamp_min(1.0)

            ok = True
            ok &= pending_after_t1
            ok &= torch.isfinite(out1.tdError).all().item()
            ok &= torch.allclose(out2.tdError, expected_td.detach(), atol=1e-6, rtol=1e-5)
            ok &= torch.allclose(out2.rComps["delayed_td_target"], expected_delayed_target, atol=1e-6, rtol=1e-5)
            ok &= torch.allclose(out2.extras["td_err_abs_mean"], expected_delayed_abs.detach(), atol=1e-6, rtol=1e-5)
            ok &= (out2.rComps["reward_basis"].shape == (B,) and torch.isfinite(out2.rComps["reward_basis"]).all().item())
            ok &= torch.allclose(out2.rComps["done_basis"], out2.rComps["done_basis"].clamp(0.0, 1.0))

            print(f"DifferentialTDSemantics {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"DifferentialTDSemantics error: {e}")
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
            prev_valid = est_a._delay_graph_cache["alive"].detach().clone().clamp(0.0, 1.0)
            out_a = self.ForwardOnce(est_a, mem2, attn2, state2, reward2_a, entropy2, done2, d_tr2, d_ph2)
            out_b = self.ForwardOnce(est_b, mem2, attn2, state2, reward2_b, entropy2, done2, d_tr2, d_ph2)

            masked_rows = done1 > 0.5
            keys = ["loss_td", "loss_trans", "loss_model", "loss_ensemble", "loss_dist", "loss_unc", "loss_aleatoric"]
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
            ok &= torch.allclose(est.hebb_value.H[0], torch.zeros_like(est.hebb_value.H[0]))
            ok &= torch.allclose(est.hebb_value.H[2], torch.zeros_like(est.hebb_value.H[2]))
            ok &= float(est.hebb_value.H[1].abs().sum().item()) > 0.0
            ok &= int(est.micro.filled[0].item()) == 0
            ok &= int(est.micro.filled[2].item()) == 0
            ok &= int(est.micro.filled[1].item()) > 0
            ok &= int(est.micro.filled[3].item()) > 0
            ok &= est._prev_alive is not None
            ok &= float(est._prev_alive[0].abs().item()) <= 1e-12
            ok &= float(est._prev_alive[2].abs().item()) <= 1e-12
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
            ok &= bool(torch.isfinite(out.extras["loss_dist"]).item())
            ok &= self.MonitorKeys().issubset(set(out.rComps.keys()))
            ok &= out.rComps["value_dist_std"].shape == (B,)
            ok &= out.rComps["value_dist_cvar"].shape == (B,)
            ok &= out.rComps["transport_branch_std"].shape == (B,)
            ok &= "value_quantiles" not in out.rComps
            ok &= "transport_counterfactual_values" not in out.rComps

            est.zero_grad(set_to_none=True)
            out.loss.backward()
            q_grad = est.quantile_head.weight.grad
            cf_grad = est.transport.cf_trop.W.grad
            ok &= q_grad is not None and float(q_grad.abs().max().item()) > 0.0
            ok &= cf_grad is not None and float(cf_grad.abs().max().item()) > 0.0

            print(f"DistributionalValueAndTransport {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"DistributionalValueAndTransport error: {e}")
            return False

    def TestCalibrationHeadGrad(self) -> bool:
        try:
            torch.manual_seed(433)
            B = 6
            est = self.NewEstimator(useHebb=False).train()
            mem, attn, state = self.RandBatch(B)
            reward, entropy, done, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
            _ = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)
            out = self.ForwardOnce(est, mem, attn, state, reward, entropy, done, d_tr, d_ph)

            est.zero_grad(set_to_none=True)
            out.loss.backward()
            grads = [
                p.grad.detach().abs().max().item()
                for p in est.calibration_head.parameters()
                if p.grad is not None]
            ok = bool(grads) and max(grads) > 0.0
            print(f"CalibrationHeadGrad {'pass' if ok else 'fail'}")
            return ok
        except Exception as e:
            print(f"CalibrationHeadGrad error: {e}")
            return False

    def TestStateMachineAndMicroGraph(self) -> bool:
        try:
            torch.manual_seed(44)
            B = 7
            est = self.NewEstimator(useHebb=False).train()

            mem, attn, state = self.RandBatch(B)
            reward, entropy, done0, d_tr, d_ph = self.RandSignals(B, doneProb=0.0)
            out1 = self.ForwardOnce(est, mem, attn, state, reward, entropy, done0, d_tr, d_ph)

            ok = True
            ok &= (est._delay_graph_cache is not None)
            ok &= (est._delay_graph_cache["h"].shape[0] == B)
            ok &= (est._delay_graph_cache["value"].shape == (B, 1))
            ok &= (est._delay_graph_cache["alive"].shape == (B,))
            ok &= (int(est.micro._step) == 1)
            ok &= torch.equal(est.micro.filled, torch.ones_like(est.micro.filled))
            ok &= (out1.rComps is not None and out1.rComps["risk"].shape == (B,))

            done1 = torch.ones(B, device=self.device)
            out2 = self.ForwardOnce(est, mem, attn, state, reward, entropy, done1, d_tr, d_ph)
            ok &= (int(est.micro._step) == 2)
            ok &= (est._prev_alive is not None and est._prev_alive.shape == (B,))
            ok &= (est._delay_graph_cache is not None and float(est._delay_graph_cache["alive"].abs().max().item()) <= 1e-6)
            ok &= (out2.rComps is not None and out2.rComps["confidence"].shape == (B,))

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
            ok &= (est._delay_graph_cache is not None and est._delay_graph_cache["h"].shape[0] == B2)
            ok &= (est.td_out_ema.mean.numel() == B2 and est.td_out_ema.var.numel() == B2)
            ok &= (est.reward_perdetic.kf_mean.numel() == B2)
            ok &= (est.done_perdetic.kf_mean.numel() == B2)
            ok &= (est.unc_core.td_ema.mean.numel() == B2)
            ok &= (est.unc_core.ent_ema.mean.numel() == B2)

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

            pre_hebb = float(est.hebb_value.H.abs().sum().item())
            est.ResetState()
            est.ResetHebbianMemory()

            ok = True
            ok &= (est._prev_h is None and est._prev_value is None and est._prev_v_next_pred is None)
            ok &= (est._prev_alive is None and est._prev_unc is None)
            ok &= (int(est.micro._step) == 0)
            ok &= torch.equal(est.micro.filled, torch.zeros_like(est.micro.filled))
            ok &= torch.equal(est.micro.ptr, torch.zeros_like(est.micro.ptr))
            ok &= (float(est.hebb_value.H.abs().sum().item()) <= 1e-12)
            ok &= (float(est.emotion_core.fast_head.H.abs().sum().item()) <= 1e-12)
            ok &= (float(est.emotion_core.slow_head.H.abs().sum().item()) <= 1e-12)
            ok &= (pre_hebb >= 0.0)

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
            est.SetDelayedGraphAfterOptimizerStep(True)
            est.emotion_core.fast_head.use_hebbian = False
            est.emotion_core.slow_head.use_hebbian = False
            for ad in [est.fc1_adapter, est.fc2_adapter, est.value_adapter]:
                ad.Grow(2, init=None, freezeOld=True)

            opt = torch.optim.Adam(est.parameters(), lr=1e-3)
            trainable = {n: p for n, p in est.named_parameters() if p.requires_grad and p.numel() > 0}
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
                est.BuildDelayedTransitionGraph()

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
            wrapper.SetDelayedGraphAfterOptimizerStep(True)

            active_slots = [("fc1", 0), ("fc2", 1), ("vhead", 1), ("qhead", 1)]
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
                wrapper.BuildDelayedTransitionGraph()

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

            spec = wrapper.sites["vhead"]
            a, b, s = spec.allocFn(2, wrapper.deviceRef, wrapper.dtypeRef)
            with torch.no_grad():
                a.mul_(0.1)
                b.mul_(0.1)
                s.fill_(0.7)
            wrapper.cand["vhead"][1]["A"].append(a)
            wrapper.cand["vhead"][1]["B"].append(b)
            wrapper.cand["vhead"][1]["s"].append(s)

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
            wrapper.SetDelayedGraphAfterOptimizerStep(True)

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
                wrapper.BuildDelayedTransitionGraph()
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
            ok &= (est1._prev_h is not None) and (est2._prev_h is not None) and torch.allclose(est1._prev_h, est2._prev_h)
            ok &= (est1._prev_v_next_pred is not None) and (est2._prev_v_next_pred is not None) and torch.allclose(est1._prev_v_next_pred, est2._prev_v_next_pred)
            ok &= torch.equal(est1.micro.filled, est2.micro.filled)
            ok &= torch.equal(est1.micro.ptr, est2.micro.ptr)
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

    def TestLossDecreases(self, steps: int = 120, batchSize: int = 24) -> bool:
        try:
            torch.manual_seed(2026)
            est = self.NewEstimator(useHebb=False).train()
            est.SetDelayedGraphAfterOptimizerStep(True)
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
                est.BuildDelayedTransitionGraph()

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

    def TestNoNanStress(self, steps: int = 40) -> bool:
        try:
            torch.manual_seed(2027)
            est = self.NewEstimator(useHebb=False).train()
            est.SetDelayedGraphAfterOptimizerStep(True)
            opt = torch.optim.Adam(est.parameters(), lr=8e-4)

            for t in range(int(steps)):
                B = int(torch.randint(low=4, high=14, size=(1,)).item())
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
                est.BuildDelayedTransitionGraph()

            print("NoNanStress pass")
            return True
        except Exception as e:
            print(f"NoNanStress error: {e}")
            return False

    def RunAll(self):
        results = {
            "ExtractorFunctional": self.TestExtractorFunctional(),
            "ValueEstimatorIOShapes": self.TestValueEstimatorIOShapes(),
            "TDUncertaintyBounds": self.TestTDUncertaintyBounds(),
            "ModelTargetOnly": self.TestModelTargetOnly(),
            "DifferentialTDSemantics": self.TestDifferentialTDSemantics(),
            "TerminalMaskDelayedLosses": self.TestTerminalMaskDelayedLosses(),
            "UncertaintyFloorNearEps": self.TestUncertaintyFloorNearEps(),
            "DoneMaskResetPerSample": self.TestDoneMaskResetPerSample(),
            "DistributionalValueAndTransport": self.TestDistributionalValueAndTransport(),
            "CalibrationHeadGrad": self.TestCalibrationHeadGrad(),
            "StateMachineAndMicroGraph": self.TestStateMachineAndMicroGraph(),
            "BatchResizeAndPredictorShapes": self.TestBatchResizeAndPredictorShapes(),
            "ResetFunctions": self.TestResetFunctions(),
            "AllTrainableParamsHaveGradAndStep": self.TestAllTrainableParamsHaveGradAndStep(),
            "WrapperAlignmentNoDelta": self.TestWrapperAlignmentNoDelta(),
            "WrapperCandidateParamsTrainable": self.TestWrapperCandidateParamsTrainable(),
            "WrapperSimCommitEquivalence": self.TestWrapperSimCommitEquivalence(),
            "WrapperUpdateWorkflow": self.TestWrapperUpdateWorkflow(),
            "ExportImportStateRoundTrip": self.TestExportImportStateRoundTrip(),
            "LossDecreases": self.TestLossDecreases(),
            "NoNanStress": self.TestNoNanStress(),}

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
