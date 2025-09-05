from __future__ import annotations
from typing import Optional, Dict, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningEMA(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.99, eps: float = 1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var",  torch.ones(dim))

    @torch.no_grad()
    def Update(self, x: torch.Tensor):
        m = x.mean(0)
        v = x.var(0, unbiased=False)
        self.mean.copy_(self.mean * self.momentum + (1 - self.momentum) * m)
        self.var.copy_( self.var  * self.momentum + (1 - self.momentum) * v)

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
                 tau0: float = 1.0, beta: float = 1.0, # temp = tau0 * exp(+beta * uncert_n)
                 lr0: float = 1.0, kappa: float = 0.5,  # lr = lr0 * (1 + kappa * relu(novelty_n))
                 gamma0: float = 0.99, delta: float = 0.02, # gamma = clip(gamma0 + delta * valence)
                 tauMin: Optional[float] = None, tauMax: Optional[float] = None,
                 lrMin: Optional[float]  = None, lrMax: Optional[float]  = None,
                 gammaMin: float = 0.90, gammaMax: float = 0.9999,
                 emaMomentum: float = 0.99):
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
        self.lr_min, self.lr_max = lrMin,  lrMax
        self.gamma_min, self.gamma_max = gammaMin, gammaMax

        self.affect_net = nn.Sequential(
            nn.Linear(memoryDim + attnDim + stateDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),)
        
        self.progress_head = nn.Linear(hidden, 1)

        self.nov_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.unc_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.prog_ema = RunningEMA(dim=1, momentum=emaMomentum)
        self.ent_ema = RunningEMA(dim=1, momentum=emaMomentum)

        self.register_buffer("state_ema", torch.zeros(stateDim))
        self.state_momentum = emaMomentum
        self.eps = 1e-6

        nn.init.zeros_(self.progress_head.bias)
        for m in self.affect_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight); nn.init.zeros_(m.bias)

    @torch.no_grad()
    def UpdateStateEma(self, s: torch.Tensor):
        mean_s = s.mean(0)
        self.state_ema.copy_(self.state_ema * self.state_momentum + (1 - self.state_momentum) * mean_s)

    def forward(self,
                memoryPrev: torch.Tensor,
                attnPrev: torch.Tensor,
                stateCurr: torch.Tensor,
                *,
                policyEntropyPrev: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None, 
                tdErrorPrev: Optional[torch.Tensor] = None ) -> IntrinsicRewardOut:

        B = stateCurr.size(0)
        device = stateCurr.device

        with torch.no_grad():
            self.UpdateStateEma(stateCurr)
        novelty = (stateCurr - self.state_ema.to(device)).pow(2).mean(dim=-1).sqrt()

        h = self.affect_net(torch.cat([memoryPrev, attnPrev, stateCurr], dim=-1)) 
        if tdErrorPrev is not None:
            progress = -tdErrorPrev.abs()
        else:
            progress = torch.tanh(self.progress_head(h).squeeze(-1))

        entropy = policyEntropyPrev if policyEntropyPrev is not None else torch.zeros(B, device=device)

        uncert = uncertainty if uncertainty is not None else torch.zeros(B, device=device)

        with torch.no_grad():
            self.nov_ema.Update(novelty)
            self.prog_ema.Update(progress)
            self.ent_ema.Update(entropy)
            self.unc_ema.Update(uncert)

        novelty_n = self.nov_ema.Norm(novelty)
        progress_n = self.prog_ema.Norm(progress)
        entropy_n = self.ent_ema.Norm(entropy)
        uncert_n = self.unc_ema.Norm(uncert)

        r_int = (self.alpha_novelty * novelty_n + self.alpha_progress * progress_n + self.alpha_entropy  * entropy_n - self.alpha_uncert_penalty * uncert_n).clamp(-self.r_clip, self.r_clip) 

        temp_scale = self.tau0 * torch.exp(+ self.beta * uncert_n)
        if self.tau_min is not None or self.tau_max is not None:
            lo = self.tau_min if self.tau_min is not None else -float("inf")
            hi = self.tau_max if self.tau_max is not None else +float("inf")
            temp_scale = temp_scale.clamp(lo, hi)

        lr_scale = self.lr0 * (1.0 + self.kappa * novelty_n.clamp_min(0.0))
        if self.lr_min is not None or self.lr_max is not None:
            lo = self.lr_min if self.lr_min is not None else -float("inf")
            hi = self.lr_max if self.lr_max is not None else +float("inf")
            lr_scale = lr_scale.clamp(lo, hi)

        valence = torch.tanh(progress_n)
        gamma_mod = (self.gamma0 + self.delta * valence).clamp(self.gamma_min, self.gamma_max)

        e_t = torch.stack([temp_scale, lr_scale, gamma_mod], dim=-1)

        comps: Dict[str, torch.Tensor] = {
            "novelty": novelty,
            "progress": progress,
            "entropy": entropy,
            "uncertainty": uncert,
            "novelty_n": novelty_n,
            "progress_n": progress_n,
            "entropy_n": entropy_n,
            "uncertainty_n": uncert_n,
            "valence": valence,}

        return IntrinsicRewardOut(rInt=r_int, components=comps, eT=e_t)


class CriticOut(NamedTuple):
    value: torch.Tensor
    tdError: torch.Tensor
    tdErrorDe: torch.Tensor
    entropy: torch.Tensor
    uncertainty: torch.Tensor
    eT: torch.Tensor
    rewardUsed: torch.Tensor
    rInt: torch.Tensor
    rComps: Dict[str, torch.Tensor]


class ValueEstimationExtractor(nn.Module):
    def __init__(self,
                 memoryDim: int = 768,
                 attnDim: int = 1024,
                 stateDim: int  = 256,
                 *,
                 hidden: int = 512,
                 gammaDefault: float = 0.99,
                 useLayerNorm: bool = False,
                 valueLossType: str = "mse",
                 useUncertHead: bool = True,
                 wExt: float = 1.0,
                 wInt: float = 1.0,
                 bootstrapIfMissing: bool = True,
                 irgKwargs: Optional[dict] = None):
        super().__init__()

        self.gamma_default = gammaDefault
        self.value_loss_type = valueLossType.lower()
        self.use_uncert = useUncertHead
        self.w_ext = wExt
        self.w_int = wInt
        self.bootstrap_if_missing = bootstrapIfMissing

        in_dim = memoryDim + attnDim + stateDim
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        if useLayerNorm:
            self.norm1 = nn.LayerNorm(hidden)
            self.norm2 = nn.LayerNorm(hidden)
        else:
            self.norm1 = self.norm2 = None

        self.value_head = nn.Linear(hidden, 1)
        self.uncert_head = nn.Linear(hidden, 1) if useUncertHead else None

        self.rgen = IntrinsicRewardGenerator(memoryDim, attnDim, stateDim, **(irgKwargs or {}))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self,
                memoryPrev: torch.Tensor,
                attnPrev: torch.Tensor,
                stateCurr: torch.Tensor,
                *,
                rewardExt: Optional[torch.Tensor] = None,
                nextValue: Optional[torch.Tensor] = None,
                done: Optional[torch.Tensor] = None,
                policyEntropyPrev: Optional[torch.Tensor] = None,
                uncertainty: Optional[torch.Tensor] = None,
                tdErrorPrev: Optional[torch.Tensor] = None) -> CriticOut:

        B = stateCurr.size(0)
        device = stateCurr.device

        x = torch.cat([memoryPrev, attnPrev, stateCurr], dim=-1)
        x = F.relu(self.fc1(x));  x = self.norm1(x) if self.norm1 is not None else x
        x = F.relu(self.fc2(x));  x = self.norm2(x) if self.norm2 is not None else x

        value = self.value_head(x).squeeze(-1) 
        uncert_val = torch.zeros(B, device=device)
        if self.uncert_head is not None:
            uncert_val = F.softplus(self.uncert_head(x).squeeze(-1)) 

        r_int_out = self.rgen(
            memoryPrev, attnPrev, stateCurr,
            policyEntropyPrev=policyEntropyPrev,
            uncertainty=(uncertainty if uncertainty is not None else uncert_val.detach()),
            tdErrorPrev=tdErrorPrev)
        
        r_int, comps, e_t = r_int_out.rInt, r_int_out.components, r_int_out.eT

        if rewardExt is None:
            reward_used = self.w_int * r_int
        else:
            reward_used = self.w_ext * rewardExt.to(device) + self.w_int * r_int
        reward_used_det = reward_used.detach()

        gamma = e_t[..., 2].detach() if e_t is not None else torch.full((B,), self.gamma_default, device=device)

        if nextValue is None:
            nextValue = value.detach() if self.bootstrap_if_missing else torch.zeros(B, device=device)
        if done is None:
            done = torch.zeros(B, device=device)

        td_target = reward_used_det + gamma * nextValue.detach() * (1 - done.float())
        td_error = td_target - value
        td_error_de = td_error.detach()

        entropy_out = policyEntropyPrev if policyEntropyPrev is not None else torch.zeros(B, device=device)

        return CriticOut(
            value=value,
            tdError=td_error,
            tdErrorDe=td_error_de,
            entropy=entropy_out,
            uncertainty=uncert_val,
            eT=e_t,
            rewardUsed=reward_used_det,
            rInt=r_int.detach(),
            rComps={k: v.detach() for k, v in comps.items()},)

    def ValueLoss(self,
                  vPred: torch.Tensor,
                  target: torch.Tensor,
                  *,
                  clipDelta: Optional[float] = None) -> torch.Tensor:
        
        if self.value_loss_type == "huber":
            loss_elem = F.smooth_l1_loss(vPred, target, reduction="none")
        else:
            loss_elem = F.mse_loss(vPred, target, reduction="none")

        if clipDelta is not None:
            v_clip = vPred + (vPred - vPred.detach()).clamp(-clipDelta, clipDelta)
            loss_alt = F.mse_loss(v_clip, target, reduction="none")
            loss_elem = torch.max(loss_elem, loss_alt)

        return loss_elem.mean()



class TestValueEstimationMTool:
    def __init__(self, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mem_dim = 768
        self.attn_dim = 512
        self.state_dim= 256

    def RandBatch(self, B: int = 3):
        mem = torch.randn(B, self.mem_dim,  device=self.device)
        attn = torch.randn(B, self.attn_dim, device=self.device)
        state = torch.randn(B, self.state_dim,device=self.device)
        return mem, attn, state

    def AuxLossForIRG(self, critic, mem, attn, state, entropy_prev, uncert_opt, td_prev):

        rgen_out = critic.rgen(mem, attn, state,policyEntropyPrev=entropy_prev,uncertainty=uncert_opt,tdErrorPrev=td_prev)
        
        w_r, w_e = 1e-2, 1e-2
        loss_r = (rgen_out.rInt ** 2).mean() 

        tau0 = critic.rgen.tau0
        lr0 = critic.rgen.lr0
        g0 = critic.rgen.gamma0
        eT = rgen_out.eT
        loss_e = ((eT[...,0] - tau0)**2 + (eT[...,1] - lr0)**2 + (eT[...,2] - g0)**2).mean()
        return w_r * loss_r + w_e * loss_e

    def TestIntrinsicRewardGenerator(self) -> bool:
        try:
            B = 4
            mem, attn, state = self.RandBatch(B)

            irg = IntrinsicRewardGenerator(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim).to(self.device)

            irg.train()

            entropy_prev = torch.rand(B, device=self.device)
            uncert = F.softplus(torch.randn(B, device=self.device))
            td_prev = torch.randn(B, device=self.device) * 0.1

            out = irg(mem, attn, state,policyEntropyPrev=entropy_prev,uncertainty=uncert,tdErrorPrev=td_prev)

            ok = True
            ok &= (out.rInt.shape == (B,))
            ok &= (out.eT.shape == (B,3))
            needed = ["novelty","progress","entropy","uncertainty","novelty_n","progress_n","entropy_n","uncertainty_n","valence"]
            ok &= all(k in out.components for k in needed)

            print(f"IntrinsicRewardGenerator test {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"IntrinsicRewardGenerator test error: {e}")
            return False

    def TestValueEstimationNoReward(self) -> bool:
        try:
            B = 3
            mem, attn, state = self.RandBatch(B)

            critic = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useUncertHead=True, bootstrapIfMissing=True).to(self.device)

            critic.train()

            entropy_prev = torch.rand(B, device=self.device)

            out = critic(memoryPrev=mem, attnPrev=attn, stateCurr=state,rewardExt=None,nextValue=None,done=None,    policyEntropyPrev=entropy_prev,uncertainty=None,tdErrorPrev=None)

            ok = True
            ok &= (out.value.shape == (B,))
            ok &= (out.tdError.shape == (B,))
            ok &= (out.tdErrorDe.shape == (B,))
            ok &= (out.entropy.shape == (B,))
            ok &= (out.uncertainty.shape == (B,))
            ok &= (out.eT.shape == (B,3))
            ok &= (out.rewardUsed.shape == (B,))
            ok &= (out.rInt.shape == (B,))

            ok &= torch.allclose(out.rewardUsed, out.rInt, atol=1e-5, rtol=1e-5)

            print(f"ValueEstimation (no external reward) test {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"ValueEstimation (no external reward) test error: {e}")
            return False

    def TestValueEstimationWithReward(self) -> bool:
        try:
            B = 5
            mem, attn, state = self.RandBatch(B)

            critic = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useUncertHead=True, wExt=1.0, wInt=1.0).to(self.device)

            critic.eval()

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            next_value = torch.randn(B, device=self.device)
            done = torch.randint(0, 2, (B,), device=self.device).float()
            entropy_prev = torch.rand(B, device=self.device)
            uncert_opt = F.softplus(torch.randn(B, device=self.device))

            out = critic(memoryPrev=mem, attnPrev=attn, stateCurr=state,rewardExt=reward_ext, nextValue=next_value, done=done,policyEntropyPrev=entropy_prev, uncertainty=uncert_opt, tdErrorPrev=None)

            gamma = out.eT[..., 2]
            td_target_expected = out.rewardUsed + gamma * next_value * (1.0 - done)
            td_error_expected = td_target_expected - out.value

            ok = True
            ok &= torch.allclose(out.tdError, td_error_expected, atol=1e-5, rtol=1e-5)
            ok &= (out.value.shape == (B,))
            ok &= (out.entropy.shape == (B,))
            ok &= (out.uncertainty.shape == (B,))
            ok &= (out.eT.shape == (B,3))

            print(f"ValueEstimation (with external reward) test {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"ValueEstimation (with external reward) test error: {e}")
            return False

    def TestValueLossAndBackward(self) -> bool:
        try:
            B = 6
            mem, attn, state = self.RandBatch(B)

            critic = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useUncertHead=False).to(self.device)

            critic.train()

            out = critic(memoryPrev=mem, attnPrev=attn, stateCurr=state)
            gamma = out.eT[..., 2].detach()
            target = (out.rewardUsed + gamma * out.value.detach()) 
            loss = critic.ValueLoss(out.value, target)

            ok = True
            ok &= (isinstance(loss, torch.Tensor) and loss.dim() == 0)
            loss.backward()
            has_grad = any((p.grad is not None and torch.isfinite(p.grad).all()) for p in critic.parameters() if p.requires_grad)
            ok &= has_grad

            print(f"ValueLoss & backward test {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"ValueLoss & backward test error: {e}")
            return False

    def TrainStepSmoke(self) -> bool:
        try:
            B = 8
            mem, attn, state = self.RandBatch(B)

            critic = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useUncertHead=True, useLayerNorm=True, wExt=1.0, wInt=1.0).to(self.device)

            critic.train()
            opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

            reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
            next_value = torch.randn(B, device=self.device)
            done = torch.randint(0, 2, (B,), device=self.device).float()
            entropy_prev = torch.rand(B, device=self.device)
            uncert_opt = F.softplus(torch.randn(B, device=self.device))

            out = critic(memoryPrev=mem, attnPrev=attn, stateCurr=state,rewardExt=reward_ext, nextValue=next_value, done=done,policyEntropyPrev=entropy_prev, uncertainty=uncert_opt, tdErrorPrev=None)

            main_loss = (out.tdError ** 2).mean()

            aux_rgen = self.AuxLossForIRG(critic, mem, attn, state, entropy_prev, uncert_opt, None)

            aux_uncert_w = 0.01
            aux_uncert = aux_uncert_w * F.mse_loss(out.uncertainty, uncert_opt)

            total = main_loss + aux_rgen + aux_uncert

            opt.zero_grad(set_to_none=True)
            total.backward()

            bad = []
            got_rgen_grad = False
            got_uncert_grad = False
            for n, p in critic.named_parameters():
                if not p.requires_grad:
                    continue
                if (p.grad is None) or (not torch.isfinite(p.grad).all()):
                    bad.append(n)
                if n.startswith("rgen.") and (p.grad is not None) and (p.grad.abs().sum() > 0):
                    got_rgen_grad = True
                if n.startswith("uncert_head") and (p.grad is not None) and (p.grad.abs().sum() > 0):
                    got_uncert_grad = True

            if bad or (not got_rgen_grad) or (not got_uncert_grad):
                print("\nValue TrainStepSmoke failed:\n")
                if bad:
                    print("\n Bad grad at:\n", bad)
                if not got_rgen_grad:
                    print("\n rgen.* did not receive gradients.")
                if not got_uncert_grad:
                    print("\n uncert_head.* did not receive gradients.")
                return False

            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt.step()
            print("Value TrainStepSmoke passed.")
            return True
        except Exception as e:
            print(f"Value TrainStepSmoke error: {e}")
            return False

    def NoNanAfterManySteps(self, steps: int = 50) -> bool:
        try:
            critic = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useUncertHead=True).to(self.device)
            critic.train()
            opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

            for t in range(steps):
                B = 8
                mem, attn, state = self.RandBatch(B)
                reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
                next_value = torch.randn(B, device=self.device)
                done = torch.randint(0, 2, (B,), device=self.device).float()
                entropy_prev = torch.rand(B, device=self.device)
                uncert_opt = F.softplus(torch.randn(B, device=self.device))

                out = critic(memoryPrev=mem, attnPrev=attn, stateCurr=state,rewardExt=reward_ext, nextValue=next_value, done=done,policyEntropyPrev=entropy_prev, uncertainty=uncert_opt, tdErrorPrev=None)

                base = (out.tdError ** 2).mean()
                aux = self.AuxLossForIRG(critic, mem, attn, state, entropy_prev, uncert_opt, None)
                aux += 0.01 * F.mse_loss(out.uncertainty, uncert_opt)
                total = base + aux

                opt.zero_grad(set_to_none=True)
                total.backward()
                for n, p in critic.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"
                opt.step()
            print("Value NoNanAfterManySteps passed.")
            return True
        except AssertionError as e:
            print(f"Value NoNanAfterManySteps failed: {e}")
            return False
        except Exception as e:
            print(f"Value NoNanAfterManySteps error: {e}")
            return False

    def ParamsActuallyChange(self, steps: int = 30) -> bool:
        try:
            critic = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useUncertHead=True).to(self.device)
            
            critic.train()
            opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

            with torch.no_grad():
                p0 = []
                for n, p in critic.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        p0.append(p.data.flatten()[:64].clone())
                p0 = torch.cat(p0) if p0 else torch.zeros(1, device=self.device)

            for _ in range(steps):
                B = 8
                mem, attn, state = self.RandBatch(B)
                reward_ext = torch.randn(B, device=self.device).clamp(-1, 1)
                next_value = torch.randn(B, device=self.device)
                done = torch.randint(0, 2, (B,), device=self.device).float()
                entropy_prev = torch.rand(B, device=self.device)
                uncert_opt = F.softplus(torch.randn(B, device=self.device))

                out = critic(memoryPrev=mem, attnPrev=attn, stateCurr=state,rewardExt=reward_ext, nextValue=next_value, done=done,policyEntropyPrev=entropy_prev, uncertainty=uncert_opt, tdErrorPrev=None)

                base = (out.tdError ** 2).mean()
                aux = self.AuxLossForIRG(critic, mem, attn, state, entropy_prev, uncert_opt, None)
                aux += 0.01 * F.mse_loss(out.uncertainty, uncert_opt)
                total = base + aux

                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()

            with torch.no_grad():
                p1 = []
                for n, p in critic.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        p1.append(p.data.flatten()[:64].clone())
                p1 = torch.cat(p1) if p1 else torch.zeros(1, device=self.device)
                delta = (p0 - p1).abs().mean().item()

            ok = delta > 1e-6
            if ok:
                print(f"Value ParamsActuallyChange passed (delta={delta:.3e}).")
            else:
                print(f"Value ParamsActuallyChange failed (delta={delta:.3e}).")
            return ok
        except Exception as e:
            print(f"Value ParamsActuallyChange error: {e}")
            return False

    def TestNormalTrainingConvergence(self, steps: int = 120, batch_size: int = 16) -> bool:
        try:
            torch.manual_seed(2025)

            critic = ValueEstimationExtractor(memoryDim=self.mem_dim, attnDim=self.attn_dim, stateDim=self.state_dim,useUncertHead=True, useLayerNorm=True).to(self.device)

            critic.train()

            in_dim = self.mem_dim + self.attn_dim + self.state_dim
            teacher = nn.Sequential(
                nn.Linear(in_dim, 256, bias=False),
                nn.GELU(),
                nn.Linear(256, 1, bias=False),
            ).to(self.device)

            for p in teacher.parameters():
                p.requires_grad_(False)

            opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
            losses = []

            for t in range(steps):
                mem = torch.randn(batch_size, self.mem_dim,  device=self.device)
                attn = torch.randn(batch_size, self.attn_dim, device=self.device)
                state = torch.randn(batch_size, self.state_dim,device=self.device)

                with torch.no_grad():
                    target = teacher(torch.cat([mem, attn, state], dim=-1)).squeeze(-1)

                out = critic(memoryPrev=mem, attnPrev=attn, stateCurr=state)

                base = F.mse_loss(out.value, target)

                aux = self.AuxLossForIRG(critic, mem, attn, state, None, F.softplus(torch.randn(batch_size, device=self.device)), None)
                aux += 0.01 * out.uncertainty.mean()

                total = base + aux

                opt.zero_grad(set_to_none=True)
                total.backward()

                for n, p in critic.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"

                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt.step()

                losses.append(float(base.detach().item()))
                if (t + 1) % max(1, steps // 4) == 0:
                    print(f"[ValueTrain] step {t+1}/{steps} | loss={losses[-1]:.6f}")

            assert len(losses) >= 2, "No valid loss trajectory is generated"
            start, end = losses[0], losses[-1]
            print(f"\n[ValueTrain] loss start={start:.6f} -> end={end:.6f}\n")

            rel_ok = end <= start * 0.70
            abs_ok = (start - end) >= 0.05
            ok = rel_ok or abs_ok
            print(f"Value TestNormalTrainingConvergence {'passed' if ok else 'failed'}.")
            return ok
        except AssertionError as e:
            print(f"Value TestNormalTrainingConvergence failed: {e}")
            return False
        except Exception as e:
            print(f"Value TestNormalTrainingConvergence error: {e}")
            return False

    def RunAll(self):
        results = []
        results.append(self.TestIntrinsicRewardGenerator())
        results.append(self.TestValueEstimationNoReward())
        results.append(self.TestValueEstimationWithReward())
        results.append(self.TestValueLossAndBackward())
        results.append(self.TrainStepSmoke())
        results.append(self.NoNanAfterManySteps())
        results.append(self.ParamsActuallyChange())
        results.append(self.TestNormalTrainingConvergence())
        passed = sum(1 for x in results if x)
        print(f"\n[ValueModule Tests] {passed}/{len(results)} passed.")
        return all(results)
