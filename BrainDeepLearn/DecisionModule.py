from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DecisionDecoupler import DecisionDecouplerV2, EndpointPoseEncoding
from FunctionTools import AGICoreModule, BaseOnlineWrapper, GetParametersScale, RoPEMultiheadAttention, SiteSpec
from ModuleMessagerManager import ModuleDim
from NeuroSymbolicModule import FAILURE_CAUSES, OPERATORS, NeuroSymbolicOutput


class LoRALinearAdapter(nn.Module):
    def __init__(self, targetLinear: nn.Linear):
        super().__init__()
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
        factory = {"device": self.target.weight.device, "dtype": self.target.weight.dtype}
        A = nn.Parameter(init.get("A", torch.randn(addRank, self.in_f, **factory) * 1e-4).contiguous())
        B = nn.Parameter(init.get("B", torch.randn(self.out_f, addRank, **factory) * 1e-4).contiguous())
        s = nn.Parameter(torch.tensor(init.get("scale", 1e-2), **factory))
        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)
        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.target.weight
        if len(self.A_list) > 0:
            dW = W.new_zeros(self.out_f, self.in_f)
            for A, B, s in zip(self.A_list, self.B_list, self.alpha):
                dW = dW + torch.tanh(s) * GetParametersScale(s) * (B @ A)
            W = W + dW
        return F.linear(x, W, self.target.bias)


class MatLoRAAdapter(AGICoreModule):
    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.M, self.N = int(rows), int(cols)
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList()

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if addRank <= 0:
            return
        if init is None:
            init = {}
        factory = {"device": self.device, "dtype": self.dtype}
        A = nn.Parameter(init.get("A", torch.randn(addRank, self.N, **factory) * 1e-4).contiguous())
        B = nn.Parameter(init.get("B", torch.randn(self.M, addRank, **factory) * 1e-4).contiguous())
        s = nn.Parameter(torch.as_tensor(init.get("scale", 1e-3), **factory))
        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)
        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def forward(self, baseMatrix: torch.Tensor) -> torch.Tensor:
        out = baseMatrix
        if len(self.A_list) > 0:
            delta = baseMatrix.new_zeros(self.M, self.N)
            for A, B, s in zip(self.A_list, self.B_list, self.alpha):
                delta = delta + torch.tanh(s) * GetParametersScale(s) * (B @ A)
            out = out + delta
        return out


class HebbianPlasticityLayer(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        outDim: int,
        rate: float = 1e-3,
        decay: float = 0.995,
        maxRowNorm: float = 2.0,
        useHebbian: bool = True,
        applyScale: float = 0.25,
    ):
        super().__init__()
        self.in_dim = int(inDim)
        self.out_dim = int(outDim)
        self.hebb_rate = float(rate)
        self.ema_alpha = float(decay)
        self.mem_norm_cap = float(maxRowNorm)
        self.apply_scale = float(applyScale)
        self.use_hebbian = bool(useHebbian)
        self.base = nn.Parameter(torch.randn(self.out_dim, self.in_dim, device=self.device, dtype=self.dtype) * 0.02)
        self.register_buffer("hebb", torch.empty(0), persistent=True)

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if int(self.hebb.size(0)) != B:
            self.hebb = torch.zeros(B, self.out_dim, self.in_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_hebbian:
            return F.linear(x, self.base)

        B = int(x.size(0))
        self.EnsureB(B, self.device, self.dtype)
        w_eff = self.base.unsqueeze(0) + self.apply_scale * self.hebb.detach()
        out = torch.einsum("bi,boi->bo", x, w_eff)
        with torch.no_grad():
            pre = x.detach()
            post = out.detach()
            hebb_term = torch.einsum("bo,bi->boi", post, pre)
            decay_term = post.square().unsqueeze(-1) * self.hebb
            delta = self.hebb_rate * (hebb_term - decay_term)
            self.hebb = self.ema_alpha * self.hebb + (1 - self.ema_alpha) * delta
            if self.mem_norm_cap > 0.0:
                flat = self.hebb.reshape(B, -1)
                nrm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                scale = (self.mem_norm_cap / nrm).clamp_max(1.0)
                self.hebb = self.hebb * scale.view(B, 1, 1)
        return out


class OptionPolicy(AGICoreModule):
    def __init__(self, zDim: int = 512, numOptions: int = 16, psiDim: int = 128, hidden: int = 256):
        super().__init__()
        self.K = int(numOptions)
        self.psiDim = int(psiDim)
        self.enc = nn.Sequential(nn.Linear(zDim, hidden), nn.SiLU())
        self.pi_o = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.K),
        )
        self.trans = nn.Parameter(torch.zeros(self.K, self.K))
        self.trans_adapter = MatLoRAAdapter(self.K, self.K)
        self.psi_head = nn.Linear(hidden, self.K * self.psiDim)
        self.psi_amp_global = nn.Parameter(torch.tensor(1.0))
        self.psi_amp_per_option = nn.Parameter(torch.ones(self.K))

    def forward(self, z: torch.Tensor, prevLogitsOpt: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(z)
        logits_base = self.pi_o(h)
        if prevLogitsOpt is not None:
            trans_eff = self.trans_adapter(self.trans)
            logits_o = logits_base + prevLogitsOpt.detach() @ torch.nan_to_num(trans_eff, nan=0.0).clamp(-10.0, 10.0)
        else:
            logits_o = logits_base
        psi_all = self.psi_head(h).view(-1, self.K, self.psiDim)
        psi_all = psi_all * self.psi_amp_global * self.psi_amp_per_option.view(1, self.K, 1)
        return logits_o, psi_all


class SwiGLUBlock(AGICoreModule):
    def __init__(self, dim: int = 768, drop: float = 0.1, layerscale: float = 1e-2):
        super().__init__()
        self.hidden = 3 * int(dim)
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, self.hidden * 2)
        self.fc2 = nn.Linear(self.hidden, dim)
        self.drop = nn.Dropout(drop)
        self.gamma = nn.Parameter(torch.ones(dim) * layerscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        a, b = self.fc1(h).chunk(2, dim=-1)
        h = self.fc2(F.silu(a) * b)
        return x + self.drop(h * self.gamma)


class ValueDynamicsRoPEBlock(AGICoreModule):
    def __init__(self, dim: int, numHeads: int, ffDim: int, drop: float = 0.05):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = RoPEMultiheadAttention(embedDim=dim, numHeads=numHeads, dropout=drop)
        self.drop_attn = nn.Dropout(drop)
        self.ln_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ffDim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(ffDim, dim),
        )
        self.drop_ff = nn.Dropout(drop)
        self.gamma_attn = nn.Parameter(torch.ones(dim) * 0.1)
        self.gamma_ff = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.ln_attn(tokens)
        attn, _ = self.attn(x, x, x, needWeights=False)
        tokens = tokens + self.drop_attn(attn * self.gamma_attn)
        ff = self.ff(self.ln_ff(tokens))
        return tokens + self.drop_ff(ff * self.gamma_ff)


class IntentFusion(AGICoreModule):
    def __init__(
        self,
        stateDim: int,
        intentDim: int,
        hidden: int = 1024,
        numHeads: int = 8,
        numIntentTokens: int = 4,
        bilinearRank: int = 64,
        drop: float = 0.1,
        layerscaleInit: float = 1e-2,
    ):
        super().__init__()
        self.state_dim = int(stateDim)
        self.intent_dim = int(intentDim)
        self.num_intent_tokens = int(numIntentTokens)
        self.rank = int(bilinearRank)
        self.ln_s = nn.LayerNorm(self.state_dim)
        self.ln_i = nn.LayerNorm(self.intent_dim)
        self.i_to_s = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),
        )
        self.film_gain = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),
        )
        self.film_bias = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),
        )
        self.bilin_s = nn.Linear(self.state_dim, self.rank, bias=False)
        self.bilin_i = nn.Linear(self.intent_dim, self.rank, bias=False)
        self.bilin_o = nn.Sequential(nn.Linear(self.rank, self.state_dim), nn.Dropout(drop))
        self.intent_to_tokens = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.num_intent_tokens * self.state_dim),
        )
        self.cross_attn = RoPEMultiheadAttention(embedDim=self.state_dim, numHeads=numHeads, dropout=drop)
        self.attn_out = nn.Sequential(nn.Linear(self.state_dim, self.state_dim), nn.Dropout(drop))
        self.fuse_mlp = nn.Sequential(
            nn.Linear(self.state_dim * 6, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),
        )
        self.gate = nn.Sequential(nn.Linear(self.state_dim * 3, self.state_dim), nn.Sigmoid())
        self.layerscale = nn.Parameter(torch.ones(self.state_dim) * layerscaleInit)
        nn.init.zeros_(self.fuse_mlp[-1].weight)
        nn.init.zeros_(self.fuse_mlp[-1].bias)

    def forward(self, state: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:
        s = self.ln_s(state)
        i = self.ln_i(intent)
        i_proj = self.i_to_s(i)
        film = state * (1.0 + torch.tanh(self.film_gain(i))) + self.film_bias(i)
        film_n = self.ln_s(film)
        inter_mul = s * i_proj
        bilin = self.bilin_o(self.bilin_s(s) * self.bilin_i(i))
        tokens = self.intent_to_tokens(i).view(-1, self.num_intent_tokens, self.state_dim)
        ctx, _ = self.cross_attn(s.unsqueeze(1), tokens, tokens, needWeights=False)
        ctx = self.attn_out(ctx.squeeze(1))
        delta = self.fuse_mlp(torch.cat([s, i_proj, film_n, inter_mul, bilin, ctx], dim=-1))
        gate = self.gate(torch.cat([s, i_proj, ctx], dim=-1))
        return state + (gate * self.layerscale) * delta


class BeliefAssembler(AGICoreModule):
    def __init__(
        self,
        memDim: int,
        intentDim: int,
        valueDim: int,
        vNextDim: int,
        worldHzxDim: int,
        beliefDim: int = 1024,
        hidden: int = 1024,
        drop: float = 0.05,):
        super().__init__()
        self.belief_dim = int(beliefDim)
        self.world_hzx_dim = int(worldHzxDim)

        self.ln_mem = nn.LayerNorm(memDim)
        self.ln_intent = nn.LayerNorm(intentDim)
        self.ln_value = nn.LayerNorm(valueDim)
        self.ln_vnext = nn.LayerNorm(vNextDim)
        self.ln_world = nn.LayerNorm(worldHzxDim)

        self.p_mem = nn.Linear(memDim, beliefDim)
        self.p_intent = nn.Linear(intentDim, beliefDim)
        self.p_value = nn.Linear(valueDim, beliefDim)
        self.p_vnext = nn.Linear(vNextDim, beliefDim)
        self.p_world = nn.Linear(worldHzxDim, beliefDim)
        self.p_scalar = nn.Linear(2, beliefDim)

        fuse_in = beliefDim * 6
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, beliefDim),)
        self.gate = nn.Sequential(
            nn.Linear(fuse_in, beliefDim),
            nn.Sigmoid(),)
        self.ln_out = nn.LayerNorm(beliefDim)
        self.layerscale = nn.Parameter(torch.ones(beliefDim) * 1e-2)

        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(
        self,
        memFeat: torch.Tensor,
        intentFeat: torch.Tensor,
        valueTensor: torch.Tensor,
        vNextTensor: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,
        worldHzx: torch.Tensor,) -> torch.Tensor:
        scalars = torch.cat([uncertainty.view(-1, 1), confidence.view(-1, 1)], dim=-1)

        m = self.p_mem(self.ln_mem(memFeat))
        i = self.p_intent(self.ln_intent(intentFeat))
        v = self.p_value(self.ln_value(valueTensor))
        vn = self.p_vnext(self.ln_vnext(vNextTensor))
        w = self.p_world(self.ln_world(worldHzx))
        s = self.p_scalar(scalars)

        fused_in = torch.cat([m, i, v, vn, w, s], dim=-1)
        delta = self.fuse(fused_in)
        gate = self.gate(fused_in)
        return self.ln_out(m + i + v + vn + w + s + gate * self.layerscale * delta)


class LatentControlInferer(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        dynDim: int,
        uDim: int,
        hidden: int = 256,
        drop: float = 0.05,):
        super().__init__()
        self.u_dim = int(uDim)
        in_dim = int(beliefDim) + int(dynDim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),)
        self.mu_head = nn.Linear(hidden, uDim)
        self.logvar_head = nn.Linear(hidden, uDim)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        with torch.no_grad():
            self.logvar_head.bias.fill_(-1.0)

    def forward(self, belief: torch.Tensor, decisionState: torch.Tensor):
        h = self.trunk(torch.cat([belief, decisionState], dim=-1))
        return self.mu_head(h), self.logvar_head(h).clamp(-8.0, 4.0)


class PredictiveDecisionCore(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        uDim: int,
        dynDim: int,
        nSteps: int = 2,
        hidden: int = 256,
        drop: float = 0.05,):
        super().__init__()
        self.n_steps = int(nSteps)
        self.dyn_dim = int(dynDim)
        in_dim = int(dynDim) + int(beliefDim) + int(uDim) + 1

        self.f = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dynDim),
            nn.Tanh(),)
        self.belief_to_dyn = nn.Linear(beliefDim, dynDim, bias=False)
        self.dt = nn.Parameter(torch.tensor(0.5))
        self.pull_gain = nn.Parameter(torch.tensor(0.1))
        self.jacobian_norm = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.f[-2].weight)
        nn.init.zeros_(self.f[-2].bias)

    def Field(
        self,
        h: torch.Tensor,
        belief: torch.Tensor,
        u: torch.Tensor,
        precision: torch.Tensor,) -> torch.Tensor:
        x = torch.cat([h, belief, u, precision], dim=-1)
        drift = self.f(x)
        belief_pull = (self.belief_to_dyn(belief) - h) * torch.sigmoid(self.pull_gain) * precision
        return torch.tanh(self.jacobian_norm) * (drift + belief_pull)

    def forward(
        self,
        prevState: torch.Tensor,
        belief: torch.Tensor,
        u: torch.Tensor,
        precision: torch.Tensor,) -> torch.Tensor:
        h = prevState
        dt = torch.sigmoid(self.dt) / float(self.n_steps)
        for _ in range(self.n_steps):
            k1 = self.Field(h, belief, u, precision)
            k2 = self.Field(h + dt * k1, belief, u, precision)
            h = h + 0.5 * dt * (k1 + k2)
        return h


class PredictionErrorHead(AGICoreModule):
    def __init__(self, dynDim: int, beliefDim: int, hidden: int = 256, drop: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dynDim),
            nn.Linear(dynDim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, beliefDim),)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decisionState: torch.Tensor) -> torch.Tensor:
        return self.net(decisionState)


class ExpectedFreeEnergyHead(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        uDim: int,
        vDim: int,
        hidden: int = 128,
        controlCost: float = 1e-3,):
        super().__init__()
        self.control_cost = float(controlCost)
        self.risk_head = nn.Sequential(
            nn.LayerNorm(beliefDim + uDim + vDim),
            nn.Linear(beliefDim + uDim + vDim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),)
        self.amb_head = nn.Sequential(
            nn.LayerNorm(beliefDim + 2),
            nn.Linear(beliefDim + 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),)
        nn.init.zeros_(self.risk_head[-1].weight)
        nn.init.zeros_(self.risk_head[-1].bias)
        nn.init.zeros_(self.amb_head[-1].weight)
        nn.init.zeros_(self.amb_head[-1].bias)

    def forward(
        self,
        belief: torch.Tensor,
        u: torch.Tensor,
        uMu: torch.Tensor,
        uLogvar: torch.Tensor,
        vNext: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,) -> Dict[str, torch.Tensor]:
        risk = F.softplus(self.risk_head(torch.cat([belief, u, vNext], dim=-1)).squeeze(-1))
        ambiguity = F.softplus(
            self.amb_head(torch.cat([belief, uncertainty.view(-1, 1), confidence.view(-1, 1)], dim=-1)).squeeze(-1)
        ) + uncertainty.view(-1)
        epistemic = 0.5 * uLogvar.sum(dim=-1)
        control_cost = self.control_cost * u.square().sum(dim=-1)
        efe = risk + ambiguity - epistemic + control_cost
        return {
            "efe": efe,
            "risk": risk,
            "ambiguity": ambiguity,
            "epistemic": epistemic,
            "control_cost": control_cost,}


class EligibilityTracePlasticityLayer(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        outDim: int,
        lam: float = 0.9,
        eta: float = 1e-3,
        gamma: float = 1e-2,
        applyScale: float = 0.25,
        maxRowNorm: float = 2.0,):
        super().__init__()
        self.in_dim = int(inDim)
        self.out_dim = int(outDim)
        self.lam = float(lam)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.apply_scale = float(applyScale)
        self.max_row_norm = float(maxRowNorm)

        self.base = nn.Parameter(torch.randn(outDim, inDim, device=self.device, dtype=self.dtype) * 0.02)
        self.register_buffer("trace", torch.empty(0), persistent=True)
        self.register_buffer("fast", torch.empty(0), persistent=True)

    def EnsureBatch(self, B: int, device: torch.device, dtype: torch.dtype):
        if int(self.trace.size(0)) != B:
            self.trace = torch.zeros(B, self.out_dim, self.in_dim, device=device, dtype=dtype)
            self.fast = torch.zeros(B, self.out_dim, self.in_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, neuromod: torch.Tensor) -> torch.Tensor:
        B = int(x.size(0))
        self.EnsureBatch(B, self.device, self.dtype)

        w_eff = self.base.unsqueeze(0) + self.apply_scale * self.fast.detach()
        out = torch.einsum("bi,boi->bo", x, w_eff)

        with torch.no_grad():
            pre = x.detach()
            post = out.detach()
            outer = torch.einsum("bo,bi->boi", post, pre)
            self.trace = self.lam * self.trace + (1.0 - self.lam) * outer

            mod = neuromod.detach().view(B, 1, 1)
            decay = post.square().unsqueeze(-1) * self.fast
            self.fast = self.fast + self.eta * mod * self.trace - self.gamma * decay

            if self.max_row_norm > 0.0:
                flat = self.fast.reshape(B, -1)
                nrm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                scale = (self.max_row_norm / nrm).clamp_max(1.0)
                self.fast = self.fast * scale.view(B, 1, 1)

        return out

    def Reset(self, doneMask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            if doneMask is None:
                self.trace.zero_()
                self.fast.zero_()
                return
            keep = (1.0 - doneMask.view(-1).to(self.trace.dtype)).view(-1, 1, 1)
            self.trace.mul_(keep)
            self.fast.mul_(keep)


def ActiveInferenceLoss(
    decisionOut: Dict[str, Any],
    *,
    wEfe: float = 1.0,
    wPredErr: float = 1.0,
    wKl: float = 1e-2,
    wDyn: float = 1e-3,) -> Dict[str, torch.Tensor]:
    efe = decisionOut["efe"]["efe"].mean()
    pred_err = decisionOut["prediction_error"].square().mean()
    mu = decisionOut["latent_control"]["mu"]
    logvar = decisionOut["latent_control"]["logvar"]
    kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=-1).mean()
    dyn = decisionOut["decision_state"].square().mean()
    total = float(wEfe) * efe + float(wPredErr) * pred_err + float(wKl) * kl + float(wDyn) * dyn
    return {
        "total": total,
        "efe": efe,
        "prediction_error": pred_err,
        "kl_latent": kl,
        "dyn_reg": dyn,}


class NeuroSymbolicConditioner(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        planDim: int = 256,
        subgoalFeatureDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        constraintTokens: int = 8,
        constraintTokenDim: int = 128,
        operatorDim: int = len(OPERATORS),
        failureDim: int = len(FAILURE_CAUSES),
        hiddenDim: int = 1024,):
        super().__init__()
        self.belief_dim = int(beliefDim)
        self.plan_dim = int(planDim)
        self.subgoal_feature_dim = int(subgoalFeatureDim)
        self.constraint_tokens = int(constraintTokens)
        self.constraint_token_dim = int(constraintTokenDim)
        self.operator_dim = int(operatorDim)
        self.failure_dim = int(failureDim)

        raw_dim = (
            self.plan_dim
            + self.subgoal_feature_dim
            + self.constraint_tokens * self.constraint_token_dim
            + self.operator_dim
            + self.failure_dim
            + 1)
        self.symbol_projector = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, hiddenDim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hiddenDim, self.belief_dim),
            nn.LayerNorm(self.belief_dim),)
        self.plan_refiner = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.plan_dim),
            nn.Linear(self.belief_dim + self.plan_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.plan_dim),
            nn.LayerNorm(self.plan_dim),)
        self.subgoal_refiner = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.subgoal_feature_dim),
            nn.Linear(self.belief_dim + self.subgoal_feature_dim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, self.subgoal_feature_dim),
            nn.LayerNorm(self.subgoal_feature_dim),)
        self.constraint_context = nn.Linear(self.belief_dim, self.constraint_token_dim)
        self.constraint_refiner = nn.Sequential(
            nn.LayerNorm(2 * self.constraint_token_dim),
            nn.Linear(2 * self.constraint_token_dim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, self.constraint_token_dim),
            nn.LayerNorm(self.constraint_token_dim),)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.operator_dim + self.failure_dim + 1),
            nn.Linear(self.belief_dim + self.operator_dim + self.failure_dim + 1, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, 1),)

    def forward(self, neuroSymbolic: NeuroSymbolicOutput) -> Dict[str, torch.Tensor]:
        B = neuroSymbolic.plan_latent.size(0)
        constraint_flat = neuroSymbolic.constraint_tokens.reshape(B, -1)
        invoke = neuroSymbolic.invoke_mask.view(B, 1)
        risk_cause_logits = getattr(
            neuroSymbolic,
            "risk_cause_logits",
            neuroSymbolic.failure_cause_logits)
        raw = torch.cat([
            neuroSymbolic.plan_latent,
            neuroSymbolic.subgoal_feature,
            constraint_flat,
            neuroSymbolic.operator_logits,
            risk_cause_logits,
            invoke,
        ], dim=-1)
        context = self.symbol_projector(raw)
        plan_latent = self.plan_refiner(torch.cat([context, neuroSymbolic.plan_latent], dim=-1))
        subgoal_feature = self.subgoal_refiner(torch.cat([context, neuroSymbolic.subgoal_feature], dim=-1))
        token_context = self.constraint_context(context).unsqueeze(1).expand(
            B,
            self.constraint_tokens,
            self.constraint_token_dim,)
        constraint_tokens = self.constraint_refiner(torch.cat([
            neuroSymbolic.constraint_tokens,
            token_context,
        ], dim=-1))
        energy = self.energy_head(torch.cat([
            context,
            neuroSymbolic.operator_logits,
            risk_cause_logits,
            invoke,
        ], dim=-1)).squeeze(-1)
        return {
            "context": context,
            "plan_latent": plan_latent,
            "subgoal_feature": subgoal_feature,
            "constraint_tokens": constraint_tokens,
            "energy": energy,}


class TemporalDecisionHead(AGICoreModule):
    def __init__(
        self,
        beliefDim: int = ModuleDim.DecisionBeliefDim,
        temporalContextDim: int = ModuleDim.TemporalContextDim,
        primitiveCount: int = ModuleDim.TemporalPrimitiveCount,
        reasonDim: int = ModuleDim.TemporalReasonDim,
        hiddenDim: int = 512,):
        super().__init__()
        self.primitive_count = int(primitiveCount)
        in_dim = (
            int(beliefDim)
            + int(temporalContextDim)
            + 2 * self.primitive_count
            + int(reasonDim)
            + 9)
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.SiLU(),)
        self.kind_head = nn.Linear(hiddenDim, self.primitive_count)
        self.duration_head = nn.Linear(hiddenDim, 3)
        self.interrupt_head = nn.Linear(hiddenDim, 1)
        self.redispatch_head = nn.Linear(hiddenDim, 1)

    def forward(
        self,
        decisionFeature: torch.Tensor,
        neuroSymbolic: NeuroSymbolicOutput,
        temporalContextFeat: torch.Tensor,
        temporalGoal: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        scalar = torch.stack([
            neuroSymbolic.continue_guard_score,
            neuroSymbolic.interrupt_guard_score,
            neuroSymbolic.redispatch_guard_score,
            temporalGoal["goal_hold_score"],
            temporalGoal["goal_replan_score"],
            neuroSymbolic.same_operator,
            neuroSymbolic.operator_changed,
            neuroSymbolic.invoke_delta,
            neuroSymbolic.reference_drift,], dim=-1)
        h = self.trunk(torch.cat([
            decisionFeature,
            temporalContextFeat,
            neuroSymbolic.temporal_logits,
            neuroSymbolic.temporal_reason_logits,
            temporalGoal["goal_mode_logits"],
            scalar,], dim=-1))
        duration_raw = F.softplus(self.duration_head(h)) * 1000.0
        kind_logits = self.kind_head(h) + neuroSymbolic.temporal_logits + temporalGoal["goal_mode_logits"]
        return {
            "kind_logits": kind_logits,
            "duration_ms": duration_raw[:, 0],
            "soft_timeout_ms": duration_raw[:, 1] + temporalGoal["goal_timeout_soft_ms"],
            "hard_timeout_ms": duration_raw[:, 1] + duration_raw[:, 2] + temporalGoal["goal_timeout_hard_ms"],
            "p_interrupt": torch.sigmoid(self.interrupt_head(h)).squeeze(-1),
            "redispatch_score": torch.sigmoid(self.redispatch_head(h)).squeeze(-1),
            "same_operator": neuroSymbolic.same_operator,
            "operator_changed": neuroSymbolic.operator_changed,
            "invoke_delta": neuroSymbolic.invoke_delta,
            "reference_drift": neuroSymbolic.reference_drift,}




class DecisionExtractor(AGICoreModule):
    def __init__(
        self,
        stateDim: int = 1024,
        useHebb: bool = True,
        optionNum: int = 80,
        hiddenDim: int = 1024,
        psiDim: int = 1024,
        intentDim: int = 1024,
        includeNoSkill: bool = True,
        *,
        valueTensorDim: int = 512,
        vNextTensorDim: int = 512,
        worldHDim: int = 512,
        worldZDim: int = 64,
        worldXDim: int = 64,
        beliefDim: int = 1024,
        decisionDynDim: int = 256,
        latentControlDim: int = 64,
        mapperEmbedDim: int = 256,
        actionEmbedDim: int = ModuleDim.DecisionFeedbackEmbedDim,):
        super().__init__()
        self.stateDim = int(stateDim)
        self.intentDim = int(intentDim)
        self.includeNoSkill = bool(includeNoSkill)
        self.value_tensor_dim = int(valueTensorDim)
        self.v_next_tensor_dim = int(vNextTensorDim)
        self.world_h_dim = int(worldHDim)
        self.world_z_dim = int(worldZDim)
        self.world_x_dim = int(worldXDim)
        self.world_hzx_dim = self.world_h_dim + self.world_z_dim + self.world_x_dim
        self.belief_dim = int(beliefDim)
        self.dyn_dim = int(decisionDynDim)
        self.u_dim = int(latentControlDim)
        self.action_embed_dim = int(actionEmbedDim)
        self.mapper_hidden_dim = int(mapperEmbedDim)
        self.num_options = int(optionNum)
        self.psi_dim = int(psiDim)

        self.belief_assembler = BeliefAssembler(
            memDim=self.stateDim,
            intentDim=self.intentDim,
            valueDim=self.value_tensor_dim,
            vNextDim=self.v_next_tensor_dim,
            worldHzxDim=self.world_hzx_dim,
            beliefDim=self.belief_dim,)
        self.latent_inferer = LatentControlInferer(
            beliefDim=self.belief_dim,
            dynDim=self.dyn_dim,
            uDim=self.u_dim,)
        self.predictive_core = PredictiveDecisionCore(
            beliefDim=self.belief_dim,
            uDim=self.u_dim,
            dynDim=self.dyn_dim,
            nSteps=2,)
        self.belief_predictor = PredictionErrorHead(
            dynDim=self.dyn_dim,
            beliefDim=self.belief_dim,)
        self.efe_head = ExpectedFreeEnergyHead(
            beliefDim=self.belief_dim,
            uDim=self.u_dim,
            vDim=self.v_next_tensor_dim,)
        self.elig_plasticity = EligibilityTracePlasticityLayer(
            inDim=self.u_dim,
            outDim=self.u_dim,)
        action_ctx_dim = self.u_dim + self.dyn_dim + 2 + self.action_embed_dim
        self.action_context = nn.Sequential(
            nn.LayerNorm(action_ctx_dim),
            nn.Linear(action_ctx_dim, self.mapper_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(self.mapper_hidden_dim, self.mapper_hidden_dim),
            nn.SiLU(),)
        self.decision_feedback_prior_head = nn.Sequential(
            nn.LayerNorm(self.mapper_hidden_dim),
            nn.Linear(self.mapper_hidden_dim, self.mapper_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.mapper_hidden_dim, self.action_embed_dim),
            nn.LayerNorm(self.action_embed_dim),)

        option_in_dim = self.dyn_dim + self.u_dim + self.mapper_hidden_dim
        self.option_head = nn.Sequential(
            nn.LayerNorm(option_in_dim),
            nn.Linear(option_in_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.num_options),)
        self.option_psi_head = nn.Sequential(
            nn.LayerNorm(option_in_dim),
            nn.Linear(option_in_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.num_options * self.psi_dim),)

        self.nesy_conditioner = NeuroSymbolicConditioner(
            beliefDim=self.belief_dim,
            planDim=256,
            subgoalFeatureDim=ModuleDim.DecisionEndpointPoseFeatDim,
            constraintTokens=8,
            constraintTokenDim=128,)
        refine_in_dim = (
            self.belief_dim
            + self.dyn_dim
            + self.u_dim
            + self.world_hzx_dim
            + ModuleDim.PstSlotDim
            + ModuleDim.DecisionEndpointPoseFeatDim
            + self.belief_dim)
        self.nesy_decision_refiner = nn.Sequential(
            nn.LayerNorm(refine_in_dim),
            nn.Linear(refine_in_dim, hiddenDim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hiddenDim, self.belief_dim),)
        self.nesy_decision_gate = nn.Sequential(
            nn.LayerNorm(refine_in_dim),
            nn.Linear(refine_in_dim, self.belief_dim),
            nn.Sigmoid(),)
        self.final_decision_norm = nn.LayerNorm(self.belief_dim)
        self.decoder_plan_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + 256),
            nn.Linear(self.belief_dim + 256, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, 256),
            nn.LayerNorm(256),)
        self.decoder_subgoal_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + 2 * ModuleDim.DecisionEndpointPoseFeatDim),
            nn.Linear(self.belief_dim + 2 * ModuleDim.DecisionEndpointPoseFeatDim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, ModuleDim.DecisionEndpointPoseFeatDim),
            nn.LayerNorm(ModuleDim.DecisionEndpointPoseFeatDim),)
        self.decoder_constraint_seed = nn.Linear(self.belief_dim, 8 * 128)
        self.decoder_constraint_head = nn.Sequential(
            nn.LayerNorm(2 * 128),
            nn.Linear(2 * 128, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, 128),
            nn.LayerNorm(128),)
        self.decision_energy_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.world_hzx_dim + ModuleDim.DecisionEndpointPoseFeatDim),
            nn.Linear(self.belief_dim + self.world_hzx_dim + ModuleDim.DecisionEndpointPoseFeatDim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, 1),)
        self.temporal_decision_head = TemporalDecisionHead(
            beliefDim=self.belief_dim,
            temporalContextDim=ModuleDim.TemporalContextDim)

    @staticmethod
    def FormatValueTensor(valueTensor: torch.Tensor, dim: int, B: int) -> torch.Tensor:
        x = valueTensor.view(B, -1)
        if x.size(-1) >= dim:
            return x[..., :dim]
        return torch.cat([x, x.new_zeros(B, dim - x.size(-1))], dim=-1)

    @staticmethod
    def Safe(x: torch.Tensor, clip: float = 60.0) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)

    @staticmethod
    def SafeSoftmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.softmax(DecisionExtractor.Safe(logits, 60.0), dim=dim)

    def forward(
        self,
        stateFeat: torch.Tensor,
        intentFeat: torch.Tensor,
        *,
        valueTensor: torch.Tensor,
        vNextTensor: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,
        worldHzx: torch.Tensor,
        prevDecisionState: torch.Tensor,
        prevLatentControl: torch.Tensor,
        prevActionEmbed: torch.Tensor,
        prevMapperHidden: torch.Tensor,
        prevTdError: torch.Tensor,
        sample: bool = True,
        deterministic: bool = False,
        prevOptionLogit: Optional[torch.Tensor] = None,
        prior: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        mixW: float = 0.3,) -> Dict[str, Any]:

        B = stateFeat.size(0)
        value_in = self.FormatValueTensor(valueTensor, self.value_tensor_dim, B)
        v_next_in = self.FormatValueTensor(vNextTensor, self.v_next_tensor_dim, B)
        belief = self.belief_assembler(
            memFeat=stateFeat,
            intentFeat=intentFeat,
            valueTensor=value_in,
            vNextTensor=v_next_in,
            uncertainty=uncertainty,
            confidence=confidence,
            worldHzx=worldHzx,)

        u_mu, u_logvar = self.latent_inferer(belief, prevDecisionState)
        if sample and not deterministic:
            u_t = u_mu + torch.exp(0.5 * u_logvar) * torch.randn_like(u_mu)
        else:
            u_t = u_mu
        u_t = 0.8 * u_t + 0.2 * prevLatentControl

        neuromod = prevTdError + 0.5 * (confidence - uncertainty)
        u_t = u_t + 0.1 * self.elig_plasticity(u_t, neuromod)

        precision = (1.0 / (uncertainty.view(-1, 1) + 1e-3)).clamp_max(20.0)
        decision_state = self.predictive_core(prevDecisionState, belief, u_t, precision)
        belief_pred = self.belief_predictor(decision_state)
        prediction_error = belief - belief_pred
        efe = self.efe_head(
            belief=belief,
            u=u_t,
            uMu=u_mu,
            uLogvar=u_logvar,
            vNext=v_next_in,
            uncertainty=uncertainty,
            confidence=confidence,)

        scalars = torch.cat([uncertainty.view(-1, 1), confidence.view(-1, 1)], dim=-1)
        mapper_hidden = self.action_context(torch.cat([
            u_t,
            decision_state,
            scalars,
            prevActionEmbed,], dim=-1))
        mapper_hidden_next = 0.5 * prevMapperHidden + 0.5 * mapper_hidden
        decision_feedback_prior = self.decision_feedback_prior_head(mapper_hidden_next)

        option_in = torch.cat([decision_state, u_t, mapper_hidden_next], dim=-1)
        option_logits = self.option_head(option_in)
        if prevOptionLogit is not None:
            option_logits = option_logits + 0.05 * prevOptionLogit.detach()
        w_t = self.SafeSoftmax(option_logits, dim=-1)
        psi_all = self.option_psi_head(option_in).view(B, self.num_options, self.psi_dim)
        psi_mix = (w_t.unsqueeze(-1) * psi_all).sum(dim=1)

        entropy_scalar = (0.5 * (1.0 + math.log(2.0 * math.pi)) + 0.5 * u_logvar).sum(dim=-1)
        decision_uncertainty = torch.sigmoid(
            efe["ambiguity"] + prediction_error.square().mean(dim=-1))

        out: Dict[str, Any] = {
            "z": decision_state,
            "entropy": entropy_scalar,
            "option": {
                "logits": option_logits,
                "psi_all": psi_all,
                "w_t": w_t,
                "psi_mix": psi_mix,},
            "prevOptionLogit_next": option_logits.detach(),
            "belief": belief,
            "decision_state": decision_state,
            "decision_state_next": decision_state.detach(),
            "decision_uncertainty": decision_uncertainty,
            "prediction_error": prediction_error,
            "latent_control": {
                "u": u_t,
                "mu": u_mu,
                "logvar": u_logvar,},
            "latent_control_next": u_t.detach(),
            "efe": efe,
            "mapper": {
                "hidden": mapper_hidden,
                "hidden_next": mapper_hidden_next.detach(),},
            "decision_feedback_prior": decision_feedback_prior,
            "decision_feedback_prior_next": decision_feedback_prior.detach(),}

        if sample:
            if deterministic:
                opt_idx = torch.argmax(option_logits, dim=-1)
            else:
                opt_idx = torch.distributions.Categorical(probs=w_t).sample()
            logp_option = F.log_softmax(option_logits, dim=-1).gather(1, opt_idx.view(-1, 1)).squeeze(1)

            out["option"].update({"opt_idx": opt_idx, "logp_option": logp_option})

        return out

    def RefineWithNeuroSymbolic(
        self,
        baseActOut: Dict[str, Any],
        neuroSymbolic: NeuroSymbolicOutput,
        endpointPoseFeat: torch.Tensor,
        worldHzx: torch.Tensor,
        pstSummary: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        temporalGoal: Dict[str, torch.Tensor],) -> Dict[str, Any]:
        nesy = self.nesy_conditioner(neuroSymbolic)
        latent_u = baseActOut["latent_control"]["u"]
        refine_in = torch.cat([
            baseActOut["belief"],
            baseActOut["decision_state"],
            latent_u,
            worldHzx,
            pstSummary,
            endpointPoseFeat,
            nesy["context"],
        ], dim=-1)
        delta = self.nesy_decision_refiner(refine_in)
        gate = self.nesy_decision_gate(refine_in)
        decision_feature = self.final_decision_norm(baseActOut["belief"] + gate * delta)
        decoder_plan_latent = self.decoder_plan_head(torch.cat([
            decision_feature,
            nesy["plan_latent"],
        ], dim=-1))
        decoder_subgoal_feature = self.decoder_subgoal_head(torch.cat([
            decision_feature,
            nesy["subgoal_feature"],
            endpointPoseFeat,
        ], dim=-1))
        B = decision_feature.size(0)
        constraint_seed = self.decoder_constraint_seed(decision_feature).view(B, 8, 128)
        decoder_constraint_tokens = self.decoder_constraint_head(torch.cat([
            nesy["constraint_tokens"],
            constraint_seed,
        ], dim=-1))
        decision_energy = self.decision_energy_head(torch.cat([
            decision_feature,
            worldHzx,
            endpointPoseFeat,
        ], dim=-1)).squeeze(-1) + nesy["energy"]
        temporal_decision = self.temporal_decision_head(
            decision_feature,
            neuroSymbolic,
            temporalContextFeat,
            temporalGoal)

        baseActOut["base_belief"] = baseActOut["belief"]
        baseActOut["belief_final"] = decision_feature
        baseActOut["decision_feature"] = decision_feature
        baseActOut["decoder_plan_latent"] = decoder_plan_latent
        baseActOut["decoder_subgoal_feature"] = decoder_subgoal_feature
        baseActOut["decoder_constraint_tokens"] = decoder_constraint_tokens
        baseActOut["neuro_symbolic_condition"] = {
            "context": nesy["context"],
            "gate": gate,
            "symbolic_energy": nesy["energy"],}
        baseActOut["decision_energy"] = decision_energy
        baseActOut["temporal_decision"] = temporal_decision
        return baseActOut

    def ResetHebbianMemory(self, value: float = 0.0, doneMask: Optional[torch.Tensor] = None):
        for m in self.modules():
            if isinstance(m, EligibilityTracePlasticityLayer):
                m.Reset(doneMask=doneMask)




class DecisionOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: DecisionExtractor,
        *,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankContext: int = 64,
        maxRankOption: int = 64,
        maxRankFeedback: int = 64,):
        self.maxRankContext = int(maxRankContext)
        self.maxRankOption = int(maxRankOption)
        self.maxRankFeedback = int(maxRankFeedback)
        super().__init__(
            base=base,
            initRankEach=initRankEach,
            autoRank=autoRank,
            evThreshold=evThreshold,
            gradEma=gradEma,)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        def alloc_lin(addRank, device, dtype, inDim, outDim):
            A = nn.Parameter(torch.randn(addRank, inDim, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype) * 1e-4)
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_lin(a, b, s):
            return torch.tanh(s) * GetParametersScale(s) * (b @ a)

        specs: Dict[str, SiteSpec] = {}

        def add_linear_site(name: str, layer: nn.Linear, maxRank: int):
            in_dim = int(layer.in_features)
            out_dim = int(layer.out_features)
            specs[name] = SiteSpec(
                name, 1, in_dim, out_dim, maxRank,
                lambda r, d, dt, _in=in_dim, _out=out_dim: alloc_lin(r, d, dt, _in, _out),
                compose_lin,)

        add_linear_site("ctx0", self.base.action_context[1], self.maxRankContext)
        add_linear_site("ctx1", self.base.action_context[4], self.maxRankContext)
        add_linear_site("feedback0", self.base.decision_feedback_prior_head[1], self.maxRankFeedback)
        add_linear_site("feedback1", self.base.decision_feedback_prior_head[3], self.maxRankFeedback)
        add_linear_site("option0", self.base.option_head[1], self.maxRankOption)
        add_linear_site("option1", self.base.option_head[3], self.maxRankOption)
        add_linear_site("option_psi0", self.base.option_psi_head[1], self.maxRankOption)
        add_linear_site("option_psi1", self.base.option_psi_head[3], self.maxRankOption)
        return specs

    @torch.no_grad()
    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,) -> Dict[str, Any]:
        out = self.base(x, **kwargs)
        D = deltasPerLayer[0] if (deltasPerLayer and len(deltasPerLayer) > 0) else {}
        mapper_hidden = out["mapper"]["hidden"]
        if D.get("feedback1") is not None:
            delta_feedback = F.linear(mapper_hidden, D["feedback1"], bias=None)
            out["decision_feedback_prior"] = out["decision_feedback_prior"] + delta_feedback
            out["decision_feedback_prior_next"] = out["decision_feedback_prior"].detach()
        if D.get("option1") is not None:
            option_delta = F.linear(torch.cat([
                out["decision_state"],
                out["latent_control"]["u"],
                out["mapper"]["hidden_next"],], dim=-1), D["option1"], bias=None)
            out["option"]["logits"] = out["option"]["logits"] + option_delta
            out["option"]["w_t"] = self.base.SafeSoftmax(out["option"]["logits"], dim=-1)
            out["prevOptionLogit_next"] = out["option"]["logits"].detach()
        return out

    def RefineWithNeuroSymbolic(
        self,
        baseActOut: Dict[str, Any],
        neuroSymbolic: NeuroSymbolicOutput,
        endpointPoseFeat: torch.Tensor,
        worldHzx: torch.Tensor,
        pstSummary: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        temporalGoal: Dict[str, torch.Tensor],) -> Dict[str, Any]:
        return self.base.RefineWithNeuroSymbolic(
            baseActOut,
            neuroSymbolic,
            endpointPoseFeat,
            worldHzx,
            pstSummary,
            temporalContextFeat,
            temporalGoal,)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        return False

class CEMPlanner(AGICoreModule):
    def __init__(
        self,
        worldModel: nn.Module,
        wmIsOnlineWrapper: bool,
        decisionDecoupler: DecisionDecouplerV2,
        N: int = 64,
        elite: int = 8,
        iters: int = 3,
        temperature: float = 1.0,
        momentum: float = 0.15,
        minVar: float = 1e-4,
    ):
        super().__init__()
        self.wm = worldModel
        self.wm_is_online_wrapper = bool(wmIsOnlineWrapper)
        self.decision_decoupler = decisionDecoupler
        self.N = int(N)
        self.elite = int(elite)
        self.iters = int(iters)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.min_var = float(minVar)

    @torch.no_grad()
    def Plan(
        self,
        decisionTensor: torch.Tensor,
        endpointPose: torch.Tensor,
        endpointPoseEncoding: EndpointPoseEncoding,
        h0: torch.Tensor,
        z0: torch.Tensor,
        x0: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        returnDiagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = int(decisionTensor.size(0))
        N = self.N
        E = min(self.elite, N)
        mu = self.decision_decoupler.MaskDecisionTensor(decisionTensor)
        std = decisionTensor.new_ones(B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim)
        physical_state_candidates = {
            k: v.unsqueeze(1).expand(B, N, *v.shape[1:]).reshape(B * N, *v.shape[1:]).contiguous()
            for k, v in physicalState.items()
        }

        for _ in range(self.iters):
            noise = torch.randn(B, N, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim, device=decisionTensor.device, dtype=decisionTensor.dtype)
            samples = self.decision_decoupler.MaskDecisionTensor(mu.unsqueeze(1) + noise * std.unsqueeze(1))
            h = h0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
            z = z0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
            x = x0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
            pose = endpointPose.unsqueeze(1).expand(B, N, -1, -1).reshape(B * N, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim).contiguous()

            action = samples.reshape(B * N, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim)
            zero_tracking = action.new_zeros(B * N, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim)
            encoding = self.decision_decoupler.EncodeEndpointPose(pose, zero_tracking, zero_tracking)
            target_pose = self.decision_decoupler.DecodeEndpointPose(pose, action)
            action_enc = self.decision_decoupler.EncodeDecisionFeedback(action, target_pose, encoding)
            if self.wm_is_online_wrapper:
                prior = self.wm.StepPriorWithDeltas(h, z, x, action_enc, physicalState=physical_state_candidates, sample=False)
            else:
                prior = self.wm.StepPriorOnly(h, z, x, action_enc, physicalState=physical_state_candidates, sample=False)
            score = prior["r_pred"].view(B, N)
            score = score * (1.0 - prior["d_prob"].view(B, N))

            topk = torch.topk(score, k=E, dim=1).indices
            elite_scores = score.gather(1, topk)
            if self.temperature <= 0.0:
                weights = torch.full_like(elite_scores, 1.0 / float(E))
            else:
                weights = F.softmax(elite_scores / float(self.temperature), dim=1)
            b_idx = torch.arange(B, device=decisionTensor.device).unsqueeze(1).expand(B, E)
            w = weights.unsqueeze(-1).unsqueeze(-1)
            elite_action = samples[b_idx, topk]
            mu_new = (w * elite_action).sum(dim=1)
            var_new = (w * (elite_action - mu_new.unsqueeze(1)).square()).sum(dim=1).clamp_min(self.min_var)
            std_new = var_new.sqrt()
            mu = self.decision_decoupler.MaskDecisionTensor(self.momentum * mu + (1.0 - self.momentum) * mu_new)
            std = self.momentum * std + (1.0 - self.momentum) * std_new

        out = {"decision_tensor": self.decision_decoupler.MaskDecisionTensor(mu)}
        if returnDiagnostics:
            out["diagnostics"] = {"std": std}
        return out


class DecisionPlannerExtractor:
    def BuildPlanner(
        self,
        worldModel: nn.Module,
        wmIsOnlineWrapper: bool,
        decisionDecoupler: DecisionDecouplerV2,
        **cemKwargs: Any,
    ) -> CEMPlanner:
        return CEMPlanner(
            worldModel=worldModel,
            wmIsOnlineWrapper=wmIsOnlineWrapper,
            decisionDecoupler=decisionDecoupler,
            **cemKwargs,
        )


class TestDecisionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def TestDecisionExtractorIOShapes(self) -> bool:
        B = 2
        model = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat,
            intentDim=ModuleDim.IntentionFeat,
            hiddenDim=256,
            psiDim=256,
            optionNum=8,
            useHebb=False,
        ).to(self.device)
        out = model(
            torch.randn(B, ModuleDim.MemoryFeat, device=self.device),
            torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
            valueTensor=torch.randn(B, model.value_tensor_dim, device=self.device),
            vNextTensor=torch.randn(B, model.v_next_tensor_dim, device=self.device),
            prevOptionLogit=torch.zeros(B, 8, device=self.device),
            uncertainty=torch.zeros(B, device=self.device),
            confidence=torch.ones(B, device=self.device),
            worldHzx=torch.randn(B, ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState, device=self.device),
            prevDecisionState=torch.zeros(B, ModuleDim.DecisionDynDim, device=self.device),
            prevLatentControl=torch.zeros(B, ModuleDim.LatentControlDim, device=self.device),
            prevActionEmbed=torch.zeros(B, ModuleDim.DecisionFeedbackEmbedDim, device=self.device),
            prevMapperHidden=torch.zeros(B, ModuleDim.MapperHiddenDim, device=self.device),
            prevTdError=torch.zeros(B, device=self.device),
        )
        return (
            out["belief"].shape == (B, ModuleDim.DecisionBeliefDim)
            and out["decision_state_next"].shape == (B, ModuleDim.DecisionDynDim)
            and out["latent_control_next"].shape == (B, ModuleDim.LatentControlDim)
            and out["decision_feedback_prior"].shape == (B, ModuleDim.DecisionFeedbackEmbedDim)
            and out["mapper"]["hidden_next"].shape == (B, ModuleDim.MapperHiddenDim)
            and out["prediction_error"].shape == (B, ModuleDim.DecisionBeliefDim)
            and out["efe"]["efe"].shape == (B,)
        )

    def TestNeuroSymbolicRefineShapes(self) -> bool:
        B = 2
        model = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat,
            intentDim=ModuleDim.IntentionFeat,
            hiddenDim=256,
            psiDim=256,
            optionNum=8,
            useHebb=False,
        ).to(self.device)
        base = model(
            torch.randn(B, ModuleDim.MemoryFeat, device=self.device),
            torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
            valueTensor=torch.randn(B, model.value_tensor_dim, device=self.device),
            vNextTensor=torch.randn(B, model.v_next_tensor_dim, device=self.device),
            prevOptionLogit=torch.zeros(B, 8, device=self.device),
            uncertainty=torch.zeros(B, device=self.device),
            confidence=torch.ones(B, device=self.device),
            worldHzx=torch.randn(B, ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState, device=self.device),
            prevDecisionState=torch.zeros(B, ModuleDim.DecisionDynDim, device=self.device),
            prevLatentControl=torch.zeros(B, ModuleDim.LatentControlDim, device=self.device),
            prevActionEmbed=torch.zeros(B, ModuleDim.DecisionFeedbackEmbedDim, device=self.device),
            prevMapperHidden=torch.zeros(B, ModuleDim.MapperHiddenDim, device=self.device),
            prevTdError=torch.zeros(B, device=self.device),
        )
        nesy = NeuroSymbolicOutput(
            facts=[],
            operator_logits=torch.randn(B, len(OPERATORS), device=self.device),
            plan_steps=[],
            operator_rationales=[],
            plan_latent=torch.randn(B, 256, device=self.device),
            subgoal_feature=torch.randn(B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
            constraint_tokens=torch.randn(B, 8, 128, device=self.device),
            risk_cause_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            risk_cause_raw_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            failure_cause_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            failure_cause_raw_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            failure_gate_logits=torch.randn(B, device=self.device),
            failure_gate=torch.sigmoid(torch.randn(B, device=self.device)),
            invoke_mask=torch.sigmoid(torch.randn(B, device=self.device)),
            same_operator=torch.zeros(B, device=self.device),
            operator_changed=torch.zeros(B, device=self.device),
            invoke_delta=torch.zeros(B, device=self.device),
            reference_drift=torch.zeros(B, device=self.device),
            temporal_logits=torch.randn(B, ModuleDim.TemporalPrimitiveCount, device=self.device),
            temporal_reason_logits=torch.randn(B, ModuleDim.TemporalReasonDim, device=self.device),
            continue_guard_score=torch.sigmoid(torch.randn(B, device=self.device)),
            interrupt_guard_score=torch.sigmoid(torch.randn(B, device=self.device)),
            redispatch_guard_score=torch.sigmoid(torch.randn(B, device=self.device)),)
        temporal_goal = {
            "goal_mode_logits": torch.randn(B, ModuleDim.TemporalPrimitiveCount, device=self.device),
            "goal_hold_score": torch.sigmoid(torch.randn(B, device=self.device)),
            "goal_replan_score": torch.sigmoid(torch.randn(B, device=self.device)),
            "goal_timeout_soft_ms": torch.rand(B, device=self.device) * 1000.0,
            "goal_timeout_hard_ms": torch.rand(B, device=self.device) * 2000.0,}
        out = model.RefineWithNeuroSymbolic(
            base,
            nesy,
            torch.randn(B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
            torch.randn(B, ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState, device=self.device),
            torch.randn(B, ModuleDim.PstSlotDim, device=self.device),
            torch.randn(B, ModuleDim.TemporalContextDim, device=self.device),
            temporal_goal,)
        return (
            out["decision_feature"].shape == (B, ModuleDim.DecisionBeliefDim)
            and out["decoder_plan_latent"].shape == (B, 256)
            and out["decoder_subgoal_feature"].shape == (B, ModuleDim.DecisionEndpointPoseFeatDim)
            and out["decoder_constraint_tokens"].shape == (B, 8, 128)
            and out["decision_energy"].shape == (B,)
            and out["temporal_decision"]["kind_logits"].shape == (B, ModuleDim.TemporalPrimitiveCount)
        )

    def RunAllTests(self) -> Dict[str, bool]:
        return {
            "DecisionExtractorIOShapes": self.TestDecisionExtractorIOShapes(),
            "NeuroSymbolicRefineShapes": self.TestNeuroSymbolicRefineShapes(),}
