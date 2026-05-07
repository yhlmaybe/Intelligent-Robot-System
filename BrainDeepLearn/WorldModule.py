from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from DecisionModule import RAW_KEYBOARD_LAYOUT
from FunctionTools import SiteSpec, BaseOnlineWrapper, AGICoreModule, GrowableLoRALinear, GetParametersScale

NUM_DISCRETE_KEYS = len(RAW_KEYBOARD_LAYOUT)


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


@dataclass
class PredictedVisualPack:
    GlobalFeat: torch.Tensor
    ObjectTokens: torch.Tensor
    MotionPred: torch.Tensor
    LegacyFeat: torch.Tensor


class PredictedVisualHead(nn.Module):
    def __init__(
        self,
        stateDim: int,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        legacyFeatDim: int = 1024,):
        super().__init__()
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.legacy_feat_dim = int(legacyFeatDim)

        hidden = max(int(stateDim), self.global_feat_dim, self.object_token_dim * 2)

        def head(outDim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(int(stateDim)),
                nn.Linear(int(stateDim), hidden),
                nn.GELU(),
                nn.Linear(hidden, int(outDim)),)

        self.global_head = head(self.global_feat_dim)
        self.object_head = head(self.num_object_tokens * self.object_token_dim)
        self.motion_head = head(self.motion_pred_dim)
        self.legacy_head = head(self.legacy_feat_dim)

    def forward(self, state: torch.Tensor) -> PredictedVisualPack:
        B = int(state.size(0))
        objects = self.object_head(state).view(B, self.num_object_tokens, self.object_token_dim)
        return PredictedVisualPack(
            GlobalFeat=self.global_head(state),
            ObjectTokens=objects,
            MotionPred=self.motion_head(state),
            LegacyFeat=self.legacy_head(state),)

class VisualReconstructor(nn.Module):
    def __init__(
        self,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        legacyFeatDim: int = 1024,):
        super().__init__()
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.legacy_feat_dim = int(legacyFeatDim)

        self.global_adapter = nn.Sequential(
            nn.LayerNorm(self.global_feat_dim),
            nn.Linear(self.global_feat_dim, self.global_feat_dim),)
        self.object_adapter = nn.Sequential(
            nn.LayerNorm(self.object_token_dim),
            nn.Linear(self.object_token_dim, self.object_token_dim),)
        self.legacy_adapter = nn.Sequential(
            nn.LayerNorm(self.legacy_feat_dim),
            nn.Linear(self.legacy_feat_dim, self.legacy_feat_dim),)
        self.pred_error_basis = nn.Sequential(
            nn.LayerNorm(self.global_feat_dim + self.object_token_dim),
            nn.Linear(self.global_feat_dim + self.object_token_dim, self.global_feat_dim),
            nn.GELU(),
            nn.Linear(self.global_feat_dim, self.global_feat_dim),)

    def forward(self, predictedVisual: PredictedVisualPack) -> Dict[str, torch.Tensor]:
        global_feat = self.global_adapter(predictedVisual.GlobalFeat)
        object_tokens = self.object_adapter(predictedVisual.ObjectTokens)
        object_summary = object_tokens.mean(dim=1)
        legacy_feat = self.legacy_adapter(predictedVisual.LegacyFeat)
        return {
            "LegacyFeat": legacy_feat,
            "GlobalFeat": global_feat,
            "ObjectTokens": object_tokens,
            "MotionPred": predictedVisual.MotionPred,
            "PredErrorBasis": self.pred_error_basis(torch.cat([global_feat, object_summary], dim=-1)),}

class ActionEncoder(AGICoreModule):
    def __init__(
        self,
        numDiscrete: int,
        mouseDim: int = 2,
        clickDim: int = 2,
        outDim: int = 256,
        hidden: int = 512,
        dropout: float = 0.1,):
        super().__init__()

        self.K, self.M, self.C = int(numDiscrete), int(mouseDim), int(clickDim)
        self.outDim, self.hidden = int(outDim), int(hidden)

        self.gain_keys = nn.Parameter(torch.full((1, self.K), 2.0))
        self.gain_mouse = nn.Parameter(torch.full((1, self.M), 1.0))
        self.gain_click = nn.Parameter(torch.full((1, self.C), 2.0))

        self.mouse_tau = nn.Parameter(torch.full((1, self.M), 200.0))

        inDim = self.K + self.M + self.C

        self.ln_in = nn.LayerNorm(inDim)

        self.fc1 = nn.Linear(inDim, 2 * self.hidden, bias=True) 
        self.fc2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.fc3 = nn.Linear(self.hidden, self.outDim, bias=True)

        self.drop  = nn.Dropout(float(dropout))
        self.ln_mid = nn.LayerNorm(self.hidden)
        self.ln_out = nn.LayerNorm(self.outDim)

        self.out_gate = nn.Linear(self.outDim, self.outDim, bias=True)

    def forward(self, keysOnehot: torch.Tensor, mouseDelta: torch.Tensor, mouseClick: torch.Tensor) -> torch.Tensor:
        m = torch.tanh(mouseDelta / self.mouse_tau) # [B, M]

        x = torch.cat([keysOnehot.float() * self.gain_keys, m * self.gain_mouse, mouseClick.float() * self.gain_click], dim=-1) # [B, K+M+C]
        x = self.ln_in(x)

        a, b = self.fc1(x).chunk(2, dim=-1) # [B, H], [B, H]
        h = F.gelu(a) * b # [B, H]

        h2 = self.drop(F.gelu(self.fc2(h))) # [B, H]
        h = self.ln_mid(h + h2) # [B, H]

        out = self.ln_out(self.fc3(h)) # [B, outDim]

        g = torch.sigmoid(self.out_gate(out)) # [B, outDim]

        return out * (0.5 + 0.5 * g) # [B, outDim]


class S4DCell(AGICoreModule):
    def __init__(self, inDim: int, deterDim: int, ssmDim: int = 512, dt: float = 1.0, dropout: float = 0.0, ffnMult: int = 4):
        super().__init__()
        self.in_dim = int(inDim)
        self.deter_dim = int(deterDim)
        self.ssm_dim = int(ssmDim)
        self.dt = float(dt)

        self.theta = nn.Parameter(torch.randn(self.ssm_dim) * 0.1)

        self.in_to_ssm = GrowableLoRALinear(nn.Linear(self.in_dim, self.ssm_dim, bias=True))
        self.ssm_to_deter = GrowableLoRALinear(nn.Linear(self.ssm_dim, self.deter_dim, bias=True))
        self.in_to_deter = GrowableLoRALinear(nn.Linear(self.in_dim, self.deter_dim, bias=True))
        self.gate = GrowableLoRALinear(nn.Linear(self.in_dim, self.ssm_dim, bias=True))
        self.out_gate = GrowableLoRALinear(nn.Linear(self.ssm_dim, self.deter_dim, bias=True))

        self.ln_y = nn.LayerNorm(self.deter_dim)
        self.ln_ffn = nn.LayerNorm(self.deter_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.deter_dim, ffnMult * self.deter_dim, bias=True),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffnMult * self.deter_dim, self.deter_dim, bias=True),)

        self.register_buffer("x", torch.zeros(1, self.ssm_dim), persistent=True)

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if self.x.size(0) != B:
            self.x = torch.zeros(int(B), self.ssm_dim, device=device, dtype=dtype)

    def CayleyStep(self, aDiag: torch.Tensor, x: torch.Tensor, Bu: torch.Tensor, dt: float):
        A = -F.softplus(aDiag)
        k = 0.5 * dt * A
        num = (1 + k) * x + dt * Bu
        denom = (1 - k).clamp_min(1e-6)
        return num / denom

    def ResetState(self, batch):
        self.x = torch.zeros(batch, self.ssm_dim, device=self.device, dtype=self.dtype)

    def Step(self, zPrev: torch.Tensor, action: torch.Tensor, *, updateState: bool = True) -> torch.Tensor:
        u = torch.cat([zPrev, action], dim=-1)
        g = torch.sigmoid(self.gate(u))
        Bu = self.in_to_ssm(u) * g

        x_next = self.CayleyStep(self.theta, self.x, Bu, self.dt)
        y_lin = self.ssm_to_deter(x_next) + self.in_to_deter(u)
        y_glu = y_lin * torch.sigmoid(self.out_gate(x_next))
        y = self.ln_y(y_glu)
        y = y + self.ffn(self.ln_ffn(y))

        if updateState:
            self.x = x_next.detach()
        return y # [B, D] deterministic state

    def StepWithX(self, zPrev: torch.Tensor, action: torch.Tensor, x: torch.Tensor): # zPrev: stochastic state
        u = torch.cat([zPrev, action], dim=-1)
        g = torch.sigmoid(self.gate(u))
        Bu = self.in_to_ssm(u) * g

        x_next = self.CayleyStep(self.theta, x, Bu, self.dt)
        y_lin = self.ssm_to_deter(x_next) + self.in_to_deter(u)
        y_glu = y_lin * torch.sigmoid(self.out_gate(x_next))
        y = self.ln_y(y_glu)
        y = y + self.ffn(self.ln_ffn(y))

        return y, x_next.detach() # y: [B, D] deterministic state
    


class PhysRefinerHead(AGICoreModule):
    def __init__(
        self,
        deterDim: int,
        actDim: int,
        projDim: int = 256,
        hidden: int = 512,
        dt: float = 1.0,
        substeps: int = 2,
        lambdaWorkCons: float = 0.10, 
        lambdaForceSmooth: float = 0.05, 
        lambdaDelta: float = 0.01, 
        clampResidualRatio: float = 0.50,  
        dampP: float = 0.00, ):
        super().__init__()
        self.D = int(deterDim)
        self.A = int(actDim)
        self.P = int(projDim)
        assert self.P % 2 == 0, f"projDim must be even, got {self.P}"
        self.Q = self.P // 2

        self.dt = float(dt)
        self.substeps = int(max(1, substeps))
        self.clamp_ratio = float(clampResidualRatio)
        self.dampP = float(dampP)

        self.l_work = float(lambdaWorkCons)
        self.l_smooth = float(lambdaForceSmooth)
        self.l_delta = float(lambdaDelta)

        self.to_qp = GrowableLoRALinear(nn.Linear(self.D, self.P, bias=True))
        self.from_qp = GrowableLoRALinear(nn.Linear(self.P, self.D, bias=True))

        self.H_net = nn.Sequential(
            GrowableLoRALinear(nn.Linear(self.P, hidden, bias=True)),
            nn.Softplus(),
            GrowableLoRALinear(nn.Linear(hidden, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, 1, bias=True)),)

        self.force_net = nn.Sequential(
            GrowableLoRALinear(nn.Linear(self.D + self.A, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.Q, bias=True)),)

        self.g_force = GrowableLoRALinear(nn.Linear(self.D + self.A, self.Q, bias=True)) 
        self.g_phys  = GrowableLoRALinear(nn.Linear(self.D + self.A, self.D, bias=True)) 

        self.g_fuse = GrowableLoRALinear(nn.Linear(self.D + self.A + self.D, self.D, bias=True))

    def HAndGrad(self, qp: torch.Tensor, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.H_net(qp) # [B,1]
        g = torch.autograd.grad(
            H.sum(), qp,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,)[0] # [B,P]
        return H, g

    def SymplecticLeapfrog(self, q: torch.Tensor, p: torch.Tensor, dt: float, create_graph: bool
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qp0 = torch.cat([q, p], dim=-1)
        H0, g0 = self.HAndGrad(qp0, create_graph=create_graph)
        dH_dq0, _ = g0.chunk(2, dim=-1)

        p_half = p - 0.5 * dt * dH_dq0

        qp_mid = torch.cat([q, p_half], dim=-1)
        _, gm = self.HAndGrad(qp_mid, create_graph=create_graph)
        _, dH_dp_mid = gm.chunk(2, dim=-1)

        q1 = q + dt * dH_dp_mid

        qp_for_p = torch.cat([q1, p_half], dim=-1)
        H1, g2 = self.HAndGrad(qp_for_p, create_graph=create_graph)
        dH_dq2, _ = g2.chunk(2, dim=-1)

        p1 = p_half - 0.5 * dt * dH_dq2
        return q1, p1, H0, H1, dH_dp_mid 

    def ClampResidual(self, delta: torch.Tensor, base: torch.Tensor, ratio: float) -> torch.Tensor:
        eps = 1e-8
        dnorm = delta.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        bnorm = base.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-3
        maxn = ratio * bnorm + eps
        scale = (maxn / dnorm).clamp(max=1.0)
        return delta * scale

    def forward(
        self,
        hPrev: torch.Tensor,
        action: torch.Tensor,
        hS4: torch.Tensor,):

        training_mode = bool(self.training)

        create_graph = bool(training_mode)

        dt_sub = self.dt / float(self.substeps)

        if training_mode:
            H_start = None
            H_end = None
            work_acc = hPrev.new_zeros(hPrev.size(0), 1) # [B,1]
            smooth_acc = hPrev.new_tensor(0.0)

        with torch.enable_grad():
            qp = self.to_qp(hPrev)
            q, p = qp.chunk(2, dim=-1)

            for i in range(self.substeps):
                h_cur = self.from_qp(torch.cat([q, p], dim=-1))

                fa0_inp = torch.cat([h_cur, action], dim=-1)
                F0 = self.force_net(fa0_inp) * torch.sigmoid(self.g_force(fa0_inp)) # [B,Q]

                if self.dampP > 0.0:
                    p = p * torch.exp(-self.dampP * dt_sub)

                p = p + 0.5 * dt_sub * F0

                q, p, H0, H1, dH_dp_mid = self.SymplecticLeapfrog(q, p, dt_sub, create_graph=create_graph)

                if training_mode:
                    if i == 0:
                        H_start = H0
                    H_end = H1

                h_mid = self.from_qp(torch.cat([q, p], dim=-1))
                fa1_inp = torch.cat([h_mid, action], dim=-1)
                F1 = self.force_net(fa1_inp) * torch.sigmoid(self.g_force(fa1_inp)) # [B,Q]

                p = p + 0.5 * dt_sub * F1

                if training_mode:
                    smooth_acc = smooth_acc + (F1 - F0).pow(2).mean()
                    F_avg = 0.5 * (F0 + F1)
                    work_acc = work_acc + (dH_dp_mid * F_avg).sum(dim=-1, keepdim=True) * dt_sub

        h_phys_raw = self.from_qp(torch.cat([q, p], dim=-1))

        d_corr = h_phys_raw - hS4 # [B,D]

        gph = torch.sigmoid(self.g_phys(torch.cat([hPrev, action], dim=-1))) # [B,D]
        d_corr = d_corr * gph

        base = hS4 - hPrev # [B,D]  
        d_corr = self.ClampResidual(d_corr, base, ratio=self.clamp_ratio)

        alpha = torch.sigmoid(self.g_fuse(torch.cat([hPrev, action, hS4], dim=-1))) # [B,D]
        h_fused = hS4 + alpha * d_corr

        if not training_mode:
            return h_fused, None, None

        if (H_start is None) or (H_end is None):
            e_work = hPrev.new_tensor(0.0)
        else:
            denom = H_start.detach().abs().mean().clamp_min(1e-6)
            dH = (H_end - H_start)
            e_work = ((dH - work_acc) / denom).pow(2).mean()

        e_smooth = smooth_acc / float(self.substeps)
        e_delta = d_corr.pow(2).mean()

        loss = (
            self.l_work * e_work
            + self.l_smooth * e_smooth
            + self.l_delta * e_delta)

        aux: Dict[str, torch.Tensor] = {}
        aux = {
            "L_work": e_work.detach(),
            "L_smooth": e_smooth.detach(),
            "L_delta": e_delta.detach(),}

        return h_fused, loss, aux


class NeSyHead(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        K: int,
        hidden: int = 1024,
        experts: int = 8,
        dropout: float = 0.1,
        *,
        temperature: float = 1.0,
        noisyGating: bool = True,
        noiseStd: float = 0.1,
        expertDropout: float = 0.0,):
        super().__init__()
        self.K = int(K)
        self.E = int(experts)

        self.temperature = float(max(1e-6, temperature))
        self.noisyGating = bool(noisyGating)
        self.noiseStd = float(noiseStd)
        self.expertDropout = float(expertDropout)

        self.input_ln = nn.LayerNorm(inDim)

        self.gate = nn.Linear(inDim, self.E)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.K),)for _ in range(self.E)])

        self.out_scale_log = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

    def GetAuxLoss(self) -> torch.Tensor:
        return self.aux_loss

    def GateWeights(self, x_aligned: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x_aligned) # [B,E]

        if self.training and self.noisyGating and (self.noiseStd > 0.0):
            logits = logits + torch.randn_like(logits) * self.noiseStd

        if self.training and (self.expertDropout > 0.0):
            keep = (torch.rand_like(logits) > self.expertDropout)
            all_drop = (~keep).all(dim=-1)
            if all_drop.any():
                rand_idx = torch.randint(0, self.E, (int(all_drop.sum().item()),), device=self.device)
                keep[all_drop] = False
                keep[all_drop, rand_idx] = True
            logits = logits.masked_fill(~keep, -1e9)

        w = F.softmax((logits / self.temperature).float(), dim=-1) # [B,E]

        if self.training:
            importance = w.float().mean(dim=0) # [E]
            self.aux_loss = float(self.E) * (importance.pow(2).sum())
        else:
            self.aux_loss = x_aligned.new_zeros(())

        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x_aligned = self.input_ln(x) # [B,inDim]
        w = self.GateWeights(x_aligned) # [B,E]

        expert_logits = torch.stack([e(x_aligned) for e in self.experts], dim=1)

        out_logits = (w.unsqueeze(-1) * expert_logits).sum(dim=1) # [B,K]

        scale = torch.exp(self.out_scale_log).clamp(1e-3, 100.0)
        out_logits = out_logits * scale

        return out_logits # [B,K]



class GeometricLinear(AGICoreModule):
    def __init__(self, inFeatures, outFeatures, wrapLinear=None, gain=0.1):
        super().__init__()
        lin = nn.Linear(inFeatures, outFeatures, bias=True)
        nn.init.orthogonal_(lin.weight, gain=gain)
        nn.init.zeros_(lin.bias)
        self.linear = wrapLinear(lin) if wrapLinear is not None else lin

    def forward(self, x): return self.linear(x)

class FilmResidual(AGICoreModule):
    def __init__(self, hidden, alpha=0.1, wrapLinear=None):
        super().__init__()
        self.alpha = float(alpha)
        self.ln = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            GeometricLinear(hidden, hidden, wrapLinear),
            nn.GELU(),
            nn.LayerNorm(hidden),
            GeometricLinear(hidden, hidden, wrapLinear),)
        
    def forward(self, h, gx, bx):
        y = (1.0 + gx) * h + bx
        y = self.ln(y)
        y = self.ff(y)
        return h + self.alpha * y

class ConnNet(AGICoreModule):
    def __init__(self,
        stateDim: int,
        actDim: int,
        *,
        hidden: int = 512,
        numBlocks: int = 3,
        rank: int = 8,
        useFull: bool = True,
        useLowrank: bool = True,
        dt: float = 1.0,
        lambdaFro: float = 1e-4,
        lambdaL1: float = 1e-5,
        lambdaSmooth: float = 3e-5,
        normClip: float = 0.8,
        wrapLinear=None):
        super().__init__()
        self.S = int(stateDim)
        self.A = int(actDim)
        self.H = int(hidden)
        self.r = int(rank)
        self.use_full = bool(useFull)
        self.use_lowrank = bool(useLowrank)
        self.dt = float(dt)
        self.lambda_fro = float(lambdaFro)
        self.lambda_l1 = float(lambdaL1)
        self.lambda_smooth = float(lambdaSmooth)
        self.norm_clip = float(normClip)

        self.enc_s = nn.Sequential(
            nn.LayerNorm(self.S),
            GeometricLinear(self.S, self.H, wrapLinear),
            nn.GELU(),)
        
        self.enc_a = nn.Sequential(
            nn.LayerNorm(self.A),
            GeometricLinear(self.A, self.H, wrapLinear),
            nn.GELU(),)

        self.film_gamma_a = GeometricLinear(self.H, self.H, wrapLinear)
        self.film_beta_a = GeometricLinear(self.H, self.H, wrapLinear)

        self.blocks = nn.ModuleList([FilmResidual(self.H, alpha=0.1, wrapLinear=wrapLinear) for _ in range(numBlocks)])

        self.head_uv = GeometricLinear(self.H, 2 * self.S * self.r, wrapLinear)
        self.head_full = GeometricLinear(self.H, self.S * self.S, wrapLinear)
        nBranches = int(self.use_lowrank) + int(self.use_full)
        self.mix = GeometricLinear(self.H, max(1, nBranches), wrapLinear)

    def BuildLowrank(self, h):
        uv = self.head_uv(h)
        U, V = uv.split(self.S * self.r, dim=-1)
        U = U.view(-1, self.S, self.r)
        V = V.view(-1, self.S, self.r)
        return U @ V.transpose(1, 2) - V @ U.transpose(1, 2)

    def BuildFull(self, h):
        M = self.head_full(h).view(-1, self.S, self.S)
        return 0.5 * (M - M.transpose(1, 2))

    def TransportApply(self, A: torch.Tensor, sBase: torch.Tensor) -> torch.Tensor:
        B, S = A.size(0), A.size(1)
        dt = self.dt

        I = torch.eye(S, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, S, S)
        lhs = I - 0.5 * dt * A
        rhs_vec = torch.einsum("bij,bj->bi", I + 0.5 * dt * A, sBase)

        lhs = lhs.contiguous()
        rhs = rhs_vec.unsqueeze(-1).contiguous()

        cayley = torch.linalg.solve(lhs, rhs).squeeze(-1)

        return cayley # [B, S]


    def ComputeGeomReg(self, A, prevA=None):
        reg = self.lambda_fro * A.pow(2).mean()
        if self.use_full and self.lambda_l1 > 0:
            reg = reg + self.lambda_l1 * A.abs().mean()
        if (prevA is not None) and (self.lambda_smooth > 0):
            reg = reg + self.lambda_smooth * (A - prevA).pow(2).mean()
        return reg

    def forward(self, sBase: torch.Tensor, actPrev: torch.Tensor) -> torch.Tensor:
        B = sBase.size(0)
        hs = self.enc_s(sBase) 
        ha = self.enc_a(actPrev) 

        g = torch.tanh(self.film_gamma_a(ha))
        b = self.film_beta_a(ha) 

        h = hs
        for blk in self.blocks:
            h = blk(h, g, b) 

        A_list = []
        if self.use_lowrank:
            A_list.append(self.BuildLowrank(h))
        if self.use_full:
            A_list.append(self.BuildFull(h))

        if not A_list:
            A = torch.zeros(B, self.S, self.S, device=sBase.device, dtype=sBase.dtype)
        elif len(A_list) == 1:
            A = A_list[0]
        else:
            w = F.softmax(self.mix(h), dim=-1) 
            A = w[:, :1].view(B, 1, 1) * A_list[0] + w[:, 1:2].view(B, 1, 1) * A_list[1]

        if self.norm_clip and self.norm_clip > 0:
            fro = A.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), self.norm_clip / fro).view(B, 1, 1)
            A = A * scale
        return A



class SoftNeSyStructure(AGICoreModule):
    def __init__(self, k: int, gExcl: int = 8, gAlo: int = 8, tauInit: float = 1.0, lambdaDag: float = 1e-3):
        super().__init__()
        self.K = int(k)
        self.Ge = int(gExcl)
        self.Ga = int(gAlo)
        self.lambda_dag = float(lambdaDag)
        self.tau = nn.Parameter(torch.tensor(float(tauInit)))
        self.M_excl = nn.Parameter(torch.randn(self.Ge, self.K) * 0.01)
        self.M_alo = nn.Parameter(torch.randn(self.Ga, self.K) * 0.01)
        self.E = nn.Parameter(torch.zeros(self.K, self.K))
        self.register_buffer("_eye", torch.eye(self.K))

    def MixExclusive(self, P: torch.Tensor, temp: float) -> torch.Tensor:
        eps = 1e-6
        Wg = F.softmax(self.M_excl, dim=-1) # [Ge, K]
        logP = torch.log(P.clamp(eps, 1 - eps)) / max(1e-6, temp) # [B, K]
        g = logP.unsqueeze(1) + torch.log(Wg.unsqueeze(0).clamp(eps)) # [B, Ge, K]
        g_sm = F.softmax(g, dim=-1) # [B, Ge, K]
        Wk = F.softmax(self.M_excl.t(), dim=-1)
        P_new = torch.einsum("bgk,kg->bk", g_sm, Wk)
        return P_new # [B, K]

    def EnforceAlo(self, P: torch.Tensor, tau: float) -> torch.Tensor:
        eps = 1e-6
        Wa = F.softmax(self.M_alo, dim=-1)
        group_vals = (P.unsqueeze(1) * Wa.unsqueeze(0)).max(dim=-1).values
        scale = torch.where(group_vals < tau, tau / (group_vals + eps), torch.ones_like(group_vals)) # [B, Ga]
        P_scaled = P.clone()
        Wk = F.softmax(self.M_alo.t(), dim=-1)
        s = torch.einsum("bg,kg->bk", scale, Wk)
        P_scaled = P_scaled * s
        return P_scaled # [B, K]

    def ApplyImplications(self, P: torch.Tensor, alpha: float) -> torch.Tensor:
        eps = 1e-6
        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        contrib = P.unsqueeze(2) * W.unsqueeze(0) # [B, K, K]
        implied = contrib.max(dim=1).values # [B, K]
        Q = torch.maximum(P, alpha * implied).clamp(eps, 1 - eps)
        return Q # [B, K]

    def ProjectTrain(self, P: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        eps = 1e-6
        t = float(max(1e-3, temp))

        soft = max(1e-3, 0.25 * t)

        P1 = self.MixExclusive(P, temp=t).clamp(eps, 1.0 - eps) # [B,K]

        aloTau = P1.new_tensor(0.60)   
        Wa = F.softmax(self.M_alo, dim=-1) # [Ga,K]

        v = (P1.unsqueeze(1) * Wa.unsqueeze(0)) # [B,Ga,K]
        attn = F.softmax(v / soft, dim=-1) # [B,Ga,K]
        group_vals = (attn * v).sum(dim=-1).clamp_min(eps) # [B,Ga]  

        deficiency = F.softplus((aloTau - group_vals) / soft) * soft # [B,Ga] 

        scale = 1.0 + deficiency / group_vals # [B,Ga]
        scale = scale.clamp(1.0, 10.0)     

        Wk = F.softmax(self.M_alo.t(), dim=-1) # [K,Ga]
        s = torch.einsum("bg,kg->bk", scale, Wk) # [B,K]
        P2 = (P1 * s).clamp(eps, 1.0 - eps) # [B,K]

        implAlpha = P2.new_tensor(1.0)   
        W = torch.sigmoid(self.E) * (1.0 - self._eye) # [K,K]

        contrib = P2.unsqueeze(2) * W.unsqueeze(0) # [B,K,K]  
        w_imp = F.softmax(contrib / soft, dim=1)  
        implied = (w_imp * contrib).sum(dim=1) # [B,K]  

        b = (implAlpha * implied).clamp(eps, 1.0 - eps) # [B,K]

        smax = torch.sigmoid((b - P2) / soft) # [B,K]
        Q = (1.0 - smax) * P2 + smax * b
        Q = Q.clamp(eps, 1.0 - eps)

        return Q # [B,K]

    @torch.no_grad()
    def ProjectRuntime(self, P: torch.Tensor, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        Q = self.MixExclusive(P, temp)
        Q = self.EnforceAlo(Q, aloTau)
        Q = self.ApplyImplications(Q, implAlpha)

        Ge = F.softmax(self.M_excl, dim=-1) # [Ge, K]
        gprob = F.softmax((torch.log(Q)).unsqueeze(1) + torch.log(Ge.unsqueeze(0)), dim=-1) # [B, Ge, K]
        excl_pen = 0.5 * ((gprob.sum(-1) ** 2) - (gprob ** 2).sum(-1)).mean(dim=-1)

        Ga = F.softmax(self.M_alo, dim=-1) # [Ga, K]
        alo_val = (Q.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values # [B, Ga]
        alo_pen = F.relu(aloTau - alo_val).mean(dim=-1) # [B]

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl_pen = (W.unsqueeze(0) * F.relu(Q.unsqueeze(2) - Q.unsqueeze(1))).mean(dim=(1,2))

        pen = excl_pen + alo_pen + impl_pen
        pen = (1.0 - torch.exp(-pen)).clamp(0.0, 1.0)
        return Q.clamp(1e-6, 1.0 - 1e-6), pen

    def LogicLosses(self, P: torch.Tensor, lambdaExcl: float, lambdaAlo: float, lambdaImpl: float, aloTau: float = 0.60):
        Ge = F.softmax(self.M_excl, dim=-1)
        g = (torch.log(P.clamp(1e-6, 1-1e-6))).unsqueeze(1) + torch.log(Ge.unsqueeze(0).clamp(1e-6)) # [B, Ge, K]
        g_sm = F.softmax(g, dim=-1) # [B, Ge, K]
        excl = 0.5 * ((g_sm.sum(-1)**2) - (g_sm**2).sum(-1)) # [B, Ge]
        excl = excl.mean()

        Ga = F.softmax(self.M_alo, dim=-1) # [Ga, K]
        top1 = (P.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values # [B, Ga]
        aloTau_t = top1.new_tensor(float(aloTau))
        alo = (F.relu(aloTau_t - top1) ** 2).mean()

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl = (W.unsqueeze(0) * F.relu(P.unsqueeze(2) - P.unsqueeze(1))).mean()

        loss = lambdaExcl * excl + lambdaAlo * alo + lambdaImpl * impl

        reg = 1e-4 * W.mean()

        Ge_sm = F.softmax(self.M_excl, dim=-1).clamp_min(1e-6)
        Ga_sm = F.softmax(self.M_alo,  dim=-1).clamp_min(1e-6)
        reg = reg + 1e-3 * (
            (Ge_sm * torch.log(Ge_sm)).sum() / float(self.Ge) +
            (Ga_sm * torch.log(Ga_sm)).sum() / float(self.Ga))

        A = (W * W) / float(self.K)  # [K,K]
        dag = torch.trace(torch.matrix_exp(A.float())) - float(self.K)
        dag = dag.to(dtype=P.dtype, device=P.device)
        reg = reg + self.lambda_dag * dag

        loss = loss + reg
        stats = {"excl": excl.detach(), "alo": alo.detach(), "impl": impl.detach()}
        return loss, stats


class FiLMHResidual(AGICoreModule):
    def __init__(
        self,
        baseDim: int,
        rediusDim: int,  
        hidden: int = 512,
        dropout: float = 0.1,
        filmScale: float = 0.10,   
        outLayerNorm: bool = True,):
        super().__init__()
        self.D = int(baseDim)
        self.Z = int(rediusDim)
        self.H = int(hidden)
        self.film_scale = float(filmScale)
        self.use_out_ln = bool(outLayerNorm)

        self.ln_h = nn.LayerNorm(self.D)
        self.ln_e = nn.LayerNorm(self.Z)

        self.e_to_gb = GrowableLoRALinear(nn.Linear(self.Z, 2 * self.D, bias=True))

        self.e_to_h = GrowableLoRALinear(nn.Linear(self.Z, self.D, bias=True))

        self.delta_ln = nn.LayerNorm(4 * self.D)
        self.delta_mlp = nn.Sequential(
            GrowableLoRALinear(nn.Linear(4 * self.D, self.H, bias=True)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            GrowableLoRALinear(nn.Linear(self.H, self.D, bias=True)),)

        self.to_gate = GrowableLoRALinear(nn.Linear(2 * self.D, self.D, bias=True))

        self.out_ln = nn.LayerNorm(self.D)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h0 = self.ln_h(h)  # [B,D]
        e0 = self.ln_e(e)  # [B,Z]

        gamma, beta = self.e_to_gb(e0).chunk(2, dim=-1) # [B,D],[B,D]
        gamma = self.film_scale * torch.tanh(gamma) # [B,D]
        beta  = self.film_scale * torch.tanh(beta) # [B,D]

        h_film = (1.0 + gamma) * h0 + beta # [B,D]

        e_h = self.e_to_h(e0) # [B,D]
        e_h = self.film_scale * torch.tanh(e_h) # [B,D] 

        feat = torch.cat([h_film, e_h, h_film * e_h, h_film - e_h], dim=-1) # [B,4D]
        feat = self.delta_ln(feat) # [B,4D]
        delta = self.delta_mlp(feat) # [B,D]

        gate_in = torch.cat([h_film, e_h], dim=-1) # [B,2D]
        gate = torch.sigmoid(self.to_gate(gate_in)) # [B,D]

        h_out = h + gate * delta # [B,D]

        if self.use_out_ln:
            h_out = self.out_ln(h_out) # [B,D]
        return h_out

class KeyEmbed(AGICoreModule):
    def __init__(self, Z: int, keyDim: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.Z = int(Z)
        self.keyDim = int(keyDim)
        H = int(hidden)

        self.ln_e = nn.LayerNorm(self.Z)
        self.ln_a = nn.LayerNorm(self.Z)

        self.to_gb = GrowableLoRALinear(nn.Linear(self.Z, 2 * self.Z, bias=True))

        self.ln_feat = nn.LayerNorm(4 * self.Z)
        self.mlp1 = GrowableLoRALinear(nn.Linear(4 * self.Z, H, bias=True))
        self.mlp2 = GrowableLoRALinear(nn.Linear(H, self.keyDim, bias=True))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, base: torch.Tensor, redius: torch.Tensor) -> torch.Tensor:
        e = self.ln_e(base) # [B,Z]
        a = self.ln_a(redius) # [B,Z]

        gamma, beta = self.to_gb(a).chunk(2, dim=-1) # [B,Z],[B,Z]
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)

        e_film = (1.0 + gamma) * e + beta # [B,Z]

        feat = torch.cat([e_film, a, e_film * a, e_film - a], dim=-1) # [B,4Z]
        feat = self.ln_feat(feat)

        h = F.silu(self.mlp1(feat)) # [B,H]
        h = self.drop(h)
        k = self.mlp2(h) # [B,keyDim]

        k = F.normalize(k, dim=-1, eps=1e-6) # [B,keyDim]
        return k


class RSSMWorldModel(AGICoreModule):
    def __init__(
        self,
        visionDim: int = 1024,
        actionDim: int = 256,
        deterDim: int = 512,
        stochDim: int = 64,
        stateDim: int = 512,
        ssmDim: int = 512,
        useDecoder: bool = True,
        useMemory: bool = True,
        memoryCapacity: int = 16384,
        memoryPath: Optional[str] = None,
        memoryAutosaveEvery: int = 0,
        nsEnabled: bool = True,
        nsLambdaExclusive: float = 1e-2,
        nsLambdaAtLeastOne: float = 1e-2,
        nsLambdaImplication: float = 1e-2,
        memTopK = 4,
        memTemp: float = 1.0,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        legacyFeatDim: int = 1024,):
        super().__init__()

        self.vision_dim = visionDim
        self.action_dim = actionDim
        self.deter_dim = deterDim
        self.stoch_dim = stochDim
        self.state_dim = stateDim
        self.use_decoder = useDecoder
        self.ssm_dim = ssmDim

        self._mem_topk: int = int(memTopK)
        self._mem_temp: float = float(memTemp)
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.legacy_feat_dim = int(legacyFeatDim)

        self._A_prev = None

        self.obs_enc = nn.Sequential(
            nn.LayerNorm(visionDim),
            GrowableLoRALinear(nn.Linear(visionDim, stateDim, bias=True)),
            nn.GELU(),
            nn.LayerNorm(stateDim),
            GrowableLoRALinear(nn.Linear(stateDim, stochDim, bias=True)),)

        self.action_encoder = ActionEncoder(numDiscrete=NUM_DISCRETE_KEYS, mouseDim=2, clickDim=2, outDim=actionDim)

        self.act_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(actionDim, stochDim, bias=True)),
            nn.LayerNorm(stochDim),
            nn.Tanh(),)
        
        self.s4 = S4DCell(inDim=stochDim + stochDim, deterDim=deterDim, ssmDim=self.ssm_dim, dt=1.0)

        self.prior_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim, 2 * stochDim, bias=True)))

        self.post_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim + stochDim, 2 * stochDim, bias=True)))
        
        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + stochDim),
            GrowableLoRALinear(nn.Linear(deterDim + stochDim, stateDim, bias=True)),
            nn.LayerNorm(stateDim),)

        self.rdone_ln = nn.LayerNorm(2 * stateDim + stochDim)

        self.rdone_trunk = nn.Sequential(
            GrowableLoRALinear(nn.Linear(2 * stateDim + stochDim, 512, bias=True)),
            nn.SiLU(),
            nn.Dropout(0.1),
            GrowableLoRALinear(nn.Linear(512, 256, bias=True)),
            nn.SiLU(),)

        self.rew_head = nn.Sequential(GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)

        self.done_head = nn.Sequential(GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)
        
        self.obs_dec = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, stateDim, bias=True)),
            nn.GELU(),
            GrowableLoRALinear(nn.Linear(stateDim, visionDim, bias=True)),)

        self._use_memory = bool(useMemory)
        self._mem_capacity = int(memoryCapacity)
        self._mem_path = memoryPath
        self._mem_autosave_every = int(memoryAutosaveEvery)
        self._mem_add_count = 0

        self.register_buffer("_mem_keys", torch.zeros(1, self._mem_capacity, stochDim))
        self.register_buffer("_mem_vals", torch.zeros(1, self._mem_capacity, stateDim))
        self.register_buffer("_mem_size", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_mem_imp", torch.zeros(1, self._mem_capacity))
        self.register_buffer("_mem_steps", torch.zeros(1, self._mem_capacity, dtype=torch.long))
        self.register_buffer("_mem_global_step", torch.zeros(1, dtype=torch.long))

        self._mem_imp_lr = 0.10

        self._ns_enabled = bool(nsEnabled)
 
        self._ns_K: int = 128
        self.ns_struct = SoftNeSyStructure(k=self._ns_K, gExcl=30, gAlo=30, tauInit=1.0)

        self.ns_head_prior = NeSyHead(deterDim, self._ns_K, hidden=1024, experts=4)
        self.ns_head_post = NeSyHead(deterDim + stochDim, self._ns_K, hidden=1024, experts=4)

        self.ns_to_delta_mu = nn.Linear(self._ns_K, stochDim)
        self.ns_gate_mu = nn.Linear(deterDim + stochDim, stochDim)
        self.ns_gate_mu_post = nn.Linear(deterDim + 2 * stochDim, stochDim)

        self.key_emb = KeyEmbed(Z=stochDim, keyDim=stochDim)

        self.state_state_film = FiLMHResidual(baseDim=self.state_dim, rediusDim=self.state_dim, hidden=512)

        self.ns_lambda_excl = float(nsLambdaExclusive)
        self.ns_lambda_alo = float(nsLambdaAtLeastOne)
        self.ns_lambda_impl = float(nsLambdaImplication)

        self.ResetState(batchSize=1)

        if self._use_memory and self._mem_path:
            self.LoadMemory(self._mem_path, mapLocation=None, strict=False)

        self.conn = ConnNet(stateDim=stateDim,actDim=stochDim,wrapLinear=GrowableLoRALinear)

        self.phys_refiner = PhysRefinerHead(deterDim=self.deter_dim,actDim=self.stoch_dim)

        self.mix_gate = nn.Sequential(GrowableLoRALinear(nn.Linear(3 * self.state_dim, 3)))

        self.future_action_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, max(self.state_dim, self.action_dim)),
            nn.GELU(),
            nn.Linear(max(self.state_dim, self.action_dim), self.action_dim),
            nn.LayerNorm(self.action_dim),)
        self.predicted_visual_head = PredictedVisualHead(
            stateDim=self.state_dim,
            globalFeatDim=self.global_feat_dim,
            objectTokenDim=self.object_token_dim,
            numObjectTokens=self.num_object_tokens,
            motionPredDim=self.motion_pred_dim,
            legacyFeatDim=self.legacy_feat_dim,)
        self.visual_reconstructor = VisualReconstructor(
            globalFeatDim=self.global_feat_dim,
            objectTokenDim=self.object_token_dim,
            numObjectTokens=self.num_object_tokens,
            legacyFeatDim=self.legacy_feat_dim,)

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        B = int(B)
        cap = int(self._mem_capacity)

        if self._mem_keys.size(0) == B:
            return

        self._mem_keys = torch.zeros(B, cap, self.stoch_dim, device=device, dtype=dtype)
        self._mem_vals = torch.zeros(B, cap, self.state_dim, device=device, dtype=dtype)
        self._mem_imp = torch.zeros(B, cap, device=device, dtype=dtype)
        self._mem_steps = torch.zeros(B, cap, device=device, dtype=torch.long)
        self._mem_size = torch.zeros(B, device=device, dtype=torch.long)
        self._mem_global_step = torch.zeros(B, device=device, dtype=torch.long)

        self._h = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
        self._z = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)

        self.s4.EnsureB(B, device, dtype)


    def SaveMemory(self, path: Optional[str] = None):
        if not self._use_memory:
            return
        p = path or self._mem_path
        if not p:
            return

        dirpath = os.path.dirname(p)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        B = int(self._mem_keys.size(0))
        cap = int(self._mem_capacity)

        maxN = int(self._mem_size.max().item()) if B > 0 else 0
        maxN = max(0, min(maxN, cap))

        payload = {
            "mem_keys": self._mem_keys[:, :maxN].detach().cpu(), # [B,maxN,Z]
            "mem_vals": self._mem_vals[:, :maxN].detach().cpu(), # [B,maxN,S]
            "mem_imp": self._mem_imp[:,  :maxN].detach().cpu(), # [B,maxN]
            "mem_steps": self._mem_steps[:, :maxN].detach().cpu(), # [B,maxN]
            "mem_size": self._mem_size.detach().cpu(), # [B]
            "mem_global_step": self._mem_global_step.detach().cpu(),} # [B]

        torch.save(payload, p)

    def LoadMemory(self, path: str, mapLocation: Optional[str] = None, strict: bool = False):
        if (not self._use_memory):
            return
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            if strict and (not os.path.exists(path)):
                raise FileNotFoundError(path)
            return

        payload = torch.load(path, map_location=mapLocation, weights_only=False)

        keys = payload["mem_keys"] # [B, C, stochDim]
        vals = payload["mem_vals"] # [B, C, stateDim]
        size = payload["mem_size"] # [B]
        imp = payload["mem_imp"] # [B, C]
        steps = payload["mem_steps"] # [B, C]
        global_step = payload["mem_global_step"] # [B]

        Bf = int(keys.size(0))
        Cf = int(keys.size(1))

        new_cap = int(max(self._mem_capacity, Cf))
        self._mem_capacity = new_cap

        dev = self.device
        dtyp = self.dtype

        new_keys = torch.zeros(Bf, new_cap, self.stoch_dim, device=dev, dtype=dtyp)
        new_vals = torch.zeros(Bf, new_cap, self.state_dim, device=dev, dtype=dtyp)
        new_imp = torch.zeros(Bf, new_cap, device=dev, dtype=dtyp)
        new_steps = torch.zeros(Bf, new_cap, device=dev, dtype=torch.long)
        new_size = torch.zeros(Bf, device=dev, dtype=torch.long)
        new_global_step = torch.zeros(Bf, device=dev, dtype=torch.long)

        new_keys[:, :Cf] = keys.to(device=dev, dtype=dtyp).contiguous()
        new_vals[:, :Cf] = vals.to(device=dev, dtype=dtyp).contiguous()
        new_imp[:,  :Cf] = imp.to(device=dev, dtype=dtyp).contiguous()
        new_steps[:, :Cf] = steps.to(device=dev, dtype=torch.long).contiguous()
        new_size[:] = size.to(device=dev, dtype=torch.long).clamp_(0, Cf)
        new_global_step[:] = global_step.to(device=dev, dtype=torch.long).view(-1)[:Bf]

        self._mem_keys = new_keys
        self._mem_vals = new_vals
        self._mem_imp = new_imp
        self._mem_steps = new_steps
        self._mem_size = new_size
        self._mem_global_step = new_global_step

    def ResetMemory(self):
        if not self._use_memory:
            return
        self._mem_keys.zero_()
        self._mem_vals.zero_()
        self._mem_imp.zero_()
        self._mem_steps.zero_()
        self._mem_size.zero_()
        self._mem_global_step.zero_()

    @torch.no_grad()
    def ReorderMemorySteps(self):
        if not self._use_memory:
            return

        B = int(self._mem_size.size(0))
        cap = int(self._mem_capacity)
        device = self.device

        if cap <= 0 or B <= 0:
            self._mem_steps.zero_()
            self._mem_global_step.zero_()
            return

        slots = torch.arange(cap, device=device).view(1, cap)
        valid = slots < self._mem_size.view(B, 1) # [B, cap]

        if not bool(valid.any().item()):
            self._mem_steps.zero_()
            self._mem_global_step.zero_()
            return

        max_step = torch.iinfo(self._mem_steps.dtype).max
        metric = torch.where(valid, self._mem_steps, torch.full_like(self._mem_steps, max_step))
        order = torch.argsort(metric, dim=1, descending=False)

        new_steps = torch.zeros_like(self._mem_steps)
        ranks = torch.arange(1, cap + 1, device=device, dtype=torch.long).view(1, cap).expand(B, cap)
        rank_valid = ranks <= self._mem_size.view(B, 1)
        assign = torch.where(rank_valid, ranks, torch.zeros_like(ranks))
        new_steps.scatter_(1, order, assign)
        new_steps = torch.where(valid, new_steps, torch.zeros_like(new_steps))

        self._mem_steps.copy_(new_steps)
        self._mem_global_step.copy_(self._mem_size)

    @torch.no_grad()
    def MemAdd(
        self,
        keyE: torch.Tensor, # [B, Z]
        valH: torch.Tensor, # [B, D]
        imp: torch.Tensor,): # [B]
    
        if not self._use_memory:
            return

        B = int(keyE.size(0))

        cap = int(self._mem_capacity)

        size = self._mem_size # [B]  
        has_space = size < cap # [B]  
        idx_replace = torch.argmin(self._mem_imp, dim=1) # [B]
        idx = torch.where(has_space, size, idx_replace).long() # [B]

        self._mem_global_step.add_(1)
        self._mem_size = torch.where(has_space, size + 1, size) # [B]

        bidx = torch.arange(B, device=self.device)

        self._mem_keys[bidx, idx] = keyE # [B,Z]
        self._mem_vals[bidx, idx] = valH # [B,S]
        self._mem_imp[bidx, idx] = imp # [B]
        self._mem_steps[bidx, idx] = self._mem_global_step

        if self._mem_path and self._mem_autosave_every > 0:
            self._mem_add_count += 1
            if self._mem_add_count % self._mem_autosave_every == 0:
                self.SaveMemory(self._mem_path)

    @torch.no_grad()
    def MemRetrieve(self, queryE: torch.Tensor) -> Optional[torch.Tensor]:
        if not self._use_memory:
            return None

        B = int(queryE.size(0))

        filled = self._mem_size # [B]
        filled_min = int(filled.min().item())

        if filled_min <= 0:
            return None

        cap = int(self._mem_capacity)
        K_req = max(int(self._mem_topk), 1)
        K = min(K_req, filled_min, cap)
        temp = max(float(self._mem_temp), 1e-6)

        keys = self._mem_keys # [B,C,Z]
        vals = self._mem_vals # [B,C,S]

        q = queryE # [B,Z]
        k = keys

        sims = torch.einsum("bd,bnd->bn", q, k) # [B,C]

        idx = torch.arange(cap, device=sims.device).view(1, cap)
        valid = idx < self._mem_size.view(B, 1)
        sims = sims.masked_fill(~valid, -1e9)

        top_vals, top_idx = torch.topk(sims, k=K, dim=-1) # [B,K]

        logits = top_vals / temp
        weights = F.softmax(logits, dim=-1) # [B,K]

        empty = (self._mem_size <= 0)
        if empty.any():
            weights = torch.where(empty.view(B, 1), torch.zeros_like(weights), weights)

        gathered = vals.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, vals.size(-1))) # [B,K,D]
        mem_h = (weights.unsqueeze(-1) * gathered).sum(dim=1) # [B,D]

        inc = weights.detach() # [B,K] in [0,1], sum=1
        if empty.any():
            inc = inc * (~empty).float().view(B, 1)

        cur = self._mem_imp.gather(1, top_idx) # [B,K]

        lr = float(getattr(self, "_mem_imp_lr", 0.10))

        new = cur + lr * (1.0 - cur) * inc
        self._mem_imp.scatter_(1, top_idx, new.clamp_(0.0, 1.0))

        return mem_h # [B,D]

    def ResetState(self, batchSize: int = 1):
        device, dtype = self.device, self.dtype
        B = int(batchSize)

        self._h = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
        self._z = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)
        self.s4.ResetState(B)
        self._A_prev = None

        if self._use_memory:
            self.EnsureB(B, device, self.dtype)

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._h, self._z, self.s4.x

    def ImportState(self, h: torch.Tensor, z: torch.Tensor, s4x: torch.Tensor):
        self._h = h.detach().clone()
        self._z = z.detach().clone()
        self.s4.x = s4x.detach().clone()

    def BuildPredictedVisual(self, state: torch.Tensor) -> Dict[str, Any]:
        predicted_visual = self.predicted_visual_head(state)
        reconstructed = self.visual_reconstructor(predicted_visual)
        return {"predicted_visual": predicted_visual,"reconstructed_visual_state": reconstructed,}

    def ObjectAttentionError(self, predictedObjects: torch.Tensor, targetObjects: torch.Tensor) -> torch.Tensor:
        D = int(targetObjects.size(-1))
        scores = torch.matmul(targetObjects, predictedObjects.transpose(1, 2)) / max(float(D) ** 0.5, 1.0)
        weights = F.softmax(scores, dim=-1)
        aligned_pred = torch.matmul(weights, predictedObjects)
        return (aligned_pred - targetObjects).pow(2).mean(dim=(1, 2))

    def ComputePredictionLoss(
        self,
        predictedVisual: PredictedVisualPack,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        precision: torch.Tensor,) -> Dict[str, torch.Tensor]:
        target = {
            "GlobalFeat": targetVisualState.GlobalFeat.detach(),
            "ObjectTokens": targetVisualState.ObjectTokens.detach(),
            "LegacyFeat": targetVisualState.LegacyFeat.detach(),
            "MotionPred": targetVisualState.MotionToken.detach(),}

        global_err = (predictedVisual.GlobalFeat - target["GlobalFeat"]).pow(2).mean(dim=-1)
        object_err = self.ObjectAttentionError(predictedVisual.ObjectTokens, target["ObjectTokens"])
        legacy_err = (predictedVisual.LegacyFeat - target["LegacyFeat"]).pow(2).mean(dim=-1)
        motion_err = (predictedVisual.MotionPred - target["MotionPred"]).pow(2).mean(dim=-1)

        recon_err = (
            (reconstructedVisualState["GlobalFeat"] - target["GlobalFeat"]).pow(2).mean(dim=-1)
            + 0.5 * self.ObjectAttentionError(reconstructedVisualState["ObjectTokens"], target["ObjectTokens"])
            + 0.25 * (reconstructedVisualState["MotionPred"] - target["MotionPred"]).pow(2).mean(dim=-1)
            + 0.5 * (reconstructedVisualState["LegacyFeat"] - target["LegacyFeat"]).pow(2).mean(dim=-1))
        basis_err = (reconstructedVisualState["PredErrorBasis"] - target["GlobalFeat"]).pow(2).mean(dim=-1)

        per_sample = global_err + object_err + 0.5 * legacy_err + 0.25 * motion_err + 0.5 * recon_err + 0.1 * basis_err
        p = precision.detach().view(-1).clamp(0.05, 1.0)
        precision_loss = (p * per_sample).mean()

        return {
            "loss_pred_global": global_err.mean(),
            "loss_pred_object": object_err.mean(),
            "loss_pred_legacy": legacy_err.mean(),
            "loss_pred_motion": motion_err.mean(),
            "loss_pred_recon": recon_err.mean(),
            "loss_pred_basis": basis_err.mean(),
            "loss_pred_precision": precision_loss,
            "loss_pred_total": precision_loss,}

    def PriorRolloutFromStateAction(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: Optional[torch.Tensor] = None,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1))
        if actionEnc is None:
            actionEnc = self.future_action_head(s_prev_base)

        a_t = self.act_proj(actionEnc)

        h_next, x_next = self.s4.StepWithX(zPrev, a_t, s4xPrev)
        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True)
            dmu = self.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            mu_p = mu_p + (base_gate * gate_scale).clamp(0.0, 1.0) * dmu

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1))

        A_t = self.conn(s_prev_base, a_t)
        s_transport = self.conn.TransportApply(A_t, s_prev_base)
        h_phys, _, _ = self.phys_refiner(hPrev, a_t, h_next)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        w = F.softmax(self.mix_gate(torch.cat([s_base, d_tr, d_ph], dim=-1)), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        trunk = self.rdone_trunk(self.rdone_ln(torch.cat([s_base, s_next, a_t], dim=-1)))
        return {
            "h_next": h_next,
            "z_next": z_next,
            "x_next": x_next,
            "s_next": s_next,
            "action_enc": actionEnc,
            "r_pred": self.rew_head(trunk).squeeze(-1),
            "d_prob": torch.sigmoid(self.done_head(trunk).squeeze(-1)),
            "d_tr": d_tr,
            "d_ph": d_ph,}

    def PredictNextVisualFromPosterior(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        actionEnc: Optional[torch.Tensor] = None,
        sample: bool = False,) -> Dict[str, Any]:
        rollout = self.PriorRolloutFromStateAction(
            hPrev=h,
            zPrev=z,
            s4xPrev=s4x,
            actionEnc=actionEnc,
            sample=sample,)
        pred = self.BuildPredictedVisual(rollout["s_next"])
        pred["prior_rollout"] = rollout
        return pred

    def NsProjectProbs(self, P: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        return self.ns_struct.ProjectTrain(P, temp=temp)

    @torch.no_grad()
    def NsProjectRuntime(self, P: torch.Tensor, *, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        return self.ns_struct.ProjectRuntime(P, aloTau=aloTau, implAlpha=implAlpha, temp=temp)

    def NsConfidence(self, P: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        P = P.clamp(eps, 1 - eps) # [B,K]
        H = -(P * torch.log(P) + (1 - P) * torch.log(1 - P)) # [B,K]
        Hmax = P.new_tensor(0.6931471805599453) # ln(2)
        conf = (1.0 - H / Hmax).clamp(0.0, 1.0) # [B,K]
        return conf

    def NsLogicLosses(self, probs: torch.Tensor):
        loss, stats = self.ns_struct.LogicLosses(
            probs,
            lambdaExcl=self.ns_lambda_excl,
            lambdaAlo=self.ns_lambda_alo,
            lambdaImpl=self.ns_lambda_impl,
            aloTau=0.6,)
        
        return loss, stats

    @torch.no_grad()
    def StepPriorOnly(
        self,
        hPrev: torch.Tensor, # deterministic state
        zPrev: torch.Tensor, # stochastic state
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        B = actionEnc.size(0)
        device, dtype = self.device, self.dtype
        self.EnsureB(B, device, dtype)

        if hPrev is None or zPrev is None or s4xPrev is None:
            hPrev = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
            zPrev = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)
            s4xPrev = torch.zeros(B, self.ssm_dim, device=device, dtype=dtype)

        a_t = self.act_proj(actionEnc)
        h_next, s4x_next = self.s4.StepWithX(zPrev, a_t, s4xPrev) # h_next: [B, deterDim], s4x_next: [B, ssmDim]

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1) # [B, stochDim]
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_prior(h_next) # [B,K]
            P_raw = torch.sigmoid(ns_logits) # [B,K]
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # Q:[B,K], pen:[B]
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True) # [B,1]

            dmu = self.ns_to_delta_mu(Q) # [B, stochDim]

            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, dmu], dim=-1))) # [B, stochDim]
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf) # [B,1]

            gate = (base_gate * gate_scale).clamp(0.0, 1.0) # [B, stochDim]

            mu_p = mu_p + gate * dmu # [B, stochDim]

        if sample:
            logstd_p = logstd_p.clamp(-7.0, 2.0) 
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p # [B, stochDim]

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1)) # [B, stateDim]
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1)) # [B, stateDim]

        A_t = self.conn(s_prev_base, a_t) # [B, stateDim, stateDim]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B, stateDim]

        h_phys, _, _ = self.phys_refiner(hPrev, a_t, h_next)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))

        d_tr = s_transport - s_base # [B,stateDim]
        d_ph = s_phys - s_base # [B,stateDim]
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3 * stateDim]

        logits = self.mix_gate(g_in) # [B,3]
        w = F.softmax(logits, dim=-1) # [B,3]
        s_next = w[:,0:1]*s_base + w[:,1:2]*s_transport + w[:,2:3]*s_phys # [B,stateDim]

        inp = torch.cat([s_base, s_next, a_t], dim=-1) # [B, 2 * stateDim + stochDim]
        h = self.rdone_trunk(self.rdone_ln(inp))

        r_pred = self.rew_head(h).squeeze(-1) # [B]
        d_logit = self.done_head(h).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]

        return h_next, z_next, s_next, s4x_next, r_pred, d_prob # s_next is world state


    def StepPosterior(
        self,
        visionIn: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,  # False: Deterministic Forward, True: Reparameterized sampling with noise(More exploratory)
        ) -> Dict[str, torch.Tensor]:

        B = int(visionIn.size(0))
        device, dtype = self.device, self.dtype
        self.EnsureB(B, device, dtype)

        raw_e = self.obs_enc(visionIn) # [B, stochDim]
        a_t = self.act_proj(actionEnc) # [B, stochDim]
        key = self.key_emb(raw_e, a_t) # [B, stochDim]

        h_pred = self.s4.Step(self._z, a_t, updateState=True) # [B, deterDim]
        x_next = self.s4.x # [B, ssmDim]

        mu_q, logstd_q = self.post_net(torch.cat([h_pred, raw_e], dim=-1)).chunk(2, dim=-1) # [B,stochDim]
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_post(torch.cat([h_pred, raw_e], dim=-1)) # [B,K]
            P_raw = torch.sigmoid(ns_logits) # [B,K]
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # Q:[B,K], pen:[B]
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True) # [B,1]

            dmu = self.ns_to_delta_mu(Q) # [B, stochDim]

            base_gate = torch.sigmoid(self.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu], dim=-1))) # [B, stochDim]
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf) # [B,1]

            gate = (base_gate * gate_scale).clamp(0.0, 1.0) # [B, stochDim]

            mu_q = mu_q + gate * dmu # [B, stochDim]

        if sample:
            z_next = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
        else:
            z_next = mu_q

        s_base = self.state_proj(torch.cat([h_pred, z_next], dim=-1)) # [B, stateDim]
        s_prev_base = self.state_proj(torch.cat([self._h, self._z], dim=-1)) # [B, stateDim]

        A_t = self.conn(s_prev_base, a_t) # [B,S,S]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B,S]

        h_phys, _, _ = self.phys_refiner(self._h, a_t, h_pred) # [B,D]
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1)) # [B,S]

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3S]
        logits = self.mix_gate(g_in) # [B,3]
        w = F.softmax(logits, dim=-1) # [B,3]
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        if self._use_memory:
            with torch.no_grad():
                mem_s = self.MemRetrieve(key) # [B,S] 

                inp_rd = torch.cat([s_base, s_next, a_t], dim=-1)
                h_rd = self.rdone_trunk(self.rdone_ln(inp_rd))
                r_pred_tmp = self.rew_head(h_rd).squeeze(-1)
                d_logit_tmp = self.done_head(h_rd).squeeze(-1)
                d_prob_tmp = torch.sigmoid(d_logit_tmp)

                r_score = torch.tanh(r_pred_tmp.detach().abs()).clamp(0.0, 1.0)
                d_score = d_prob_tmp.detach().clamp(0.0, 1.0)

                if self._ns_enabled:
                    conf_scalar = self.NsConfidence(Q).mean(dim=-1) # [B]
                    imp_ns = ((1.0 - pen).clamp(0.0, 1.0) * (0.5 + 0.5 * conf_scalar)).clamp(0.0, 1.0)
                else:
                    imp_ns = torch.full((B,), 0.5, device=device, dtype=dtype)

                imp = (0.60 * imp_ns + 0.25 * r_score + 0.15 * d_score).clamp(0.0, 1.0)
                self.MemAdd(key.detach(), s_next.detach(), imp.detach())

                if mem_s is not None:
                    s_next = self.state_state_film(s_next, mem_s)

        inp = torch.cat([s_base, s_next, a_t], dim=-1)
        h = self.rdone_trunk(self.rdone_ln(inp))
        r_pred = self.rew_head(h).squeeze(-1) # [B]
        d_logit = self.done_head(h).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]

        out: Dict[str, torch.Tensor] = {
            "h_next": h_pred,
            "z_next": z_next,
            "x_next": x_next,
            "s_next": s_next,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "mu_q": mu_q,
            "logstd_q": logstd_q,}

        if self._ns_enabled:
            out["ns_logits"] = ns_logits
            out["ns_Q"] = Q
            out["ns_pen"] = pen

        if self.use_decoder:
            out["recon"] = self.obs_dec(s_next)
            out["recon_target"] = visionIn

        out.update(self.PredictNextVisualFromPosterior(
            h_pred,
            z_next,
            x_next,
            actionEnc=None,
            sample=False,))

        self._h = h_pred.detach()
        self._z = z_next.detach()

        return out


    def ForwardTrain(
        self,
        visionIn: torch.Tensor, # [B, visionDim]
        keysVec: torch.Tensor, # [B, K]
        mouseClick: torch.Tensor, # [B, 2]
        mouseSeq: torch.Tensor, # [B, 2]
        reward: torch.Tensor, # [B]  
        done: torch.Tensor, # [B]   
        *,
        sample: bool = True,
        alphaKl: float = 0.8,
        freeNats: float = 1.0,
        reconCoef: float = 1.0,
        rewardCoef: float = 1.0,
        doneCoef: float = 1.0,
        nsCoef: float = 1.0,
        nsDistillCoef: float = 1e-2,
        nsPriorLogicCoef: float = 1e-3,
        physCoef: float = 1e-4,) -> Dict[str, torch.Tensor]:

        B = visionIn.size(0)
        device, dtype = self.device, self.dtype
        self.EnsureB(B, device, dtype)

        h0 = self._h
        z0 = self._z

        a_enc = self.action_encoder(keysVec, mouseSeq, mouseClick) # [B, actionDim]
        a_t = self.act_proj(a_enc) # [B, stochDim]

        h_pred = self.s4.Step(z0, a_t) # [B,D]

        mu_p, logstd_p = self.prior_net(h_pred).chunk(2, dim=-1) # [B,stochDim]
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        logits_pr = None
        P_pr_train = None
        ns_prior_logic = visionIn.new_tensor(0.0)

        if self._ns_enabled:
            logits_pr = self.ns_head_prior(h_pred) # [B,K]
            P_pr_raw = torch.sigmoid(logits_pr) # [B,K]
            P_pr_train = self.NsProjectProbs(P_pr_raw) # [B,K] 

            dmu_p = self.ns_to_delta_mu(P_pr_train) # [B,stochDim]
            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_pred, dmu_p], dim=-1)))  # [B,stochDim]

            _, pen_pr = self.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)  # [B]

            conf = self.NsConfidence(P_pr_train).mean(dim=-1, keepdim=True)  # [B,1]
            gate_scale = (1.0 - 0.40 * pen_pr.view(-1, 1)) * (0.6 + 0.4 * conf) # [B,1]
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)  # [B,stochDim]

            mu_p = mu_p + gate * dmu_p

            if nsPriorLogicCoef > 0.0:
                ns_prior_logic, _ = self.NsLogicLosses(P_pr_train)

        raw_e = self.obs_enc(visionIn) # [B,stochDim]

        mu_q, logstd_q = self.post_net(torch.cat([h_pred, raw_e], dim=-1)).chunk(2, dim=-1) # [B,stochDim]
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        ns_loss = visionIn.new_tensor(0.0)
        ns_distill = visionIn.new_tensor(0.0)
        logits_q = None
        Q_train = None
        pen_q = None

        if self._ns_enabled:
            logits_q = self.ns_head_post(torch.cat([h_pred, raw_e], dim=-1)) # [B,K]
            P_q_raw = torch.sigmoid(logits_q) # [B,K]
            Q_train = self.NsProjectProbs(P_q_raw) # [B,K]

            dmu_q = self.ns_to_delta_mu(Q_train) # [B,stochDim]
            base_gate_q = torch.sigmoid(self.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu_q], dim=-1))) # [B,stochDim]

            _, pen_q = self.NsProjectRuntime(P_q_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # [B]

            conf_q = self.NsConfidence(Q_train).mean(dim=-1, keepdim=True) # [B,1]
            gate_scale_q = (1.0 - 0.40 * pen_q.view(-1, 1)) * (0.6 + 0.4 * conf_q)
            gate_q = (base_gate_q * gate_scale_q).clamp(0.0, 1.0)

            mu_q = mu_q + gate_q * dmu_q

            ns_loss, _ = self.NsLogicLosses(Q_train)

            if (logits_pr is not None) and (nsDistillCoef > 0.0):
                with torch.no_grad():
                    P_teacher = torch.sigmoid(logits_q) # [B,K]
                ns_distill = F.binary_cross_entropy_with_logits(logits_pr, P_teacher, reduction="mean")

        if sample:
            z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q) # [B,stochDim]
        else:
            z1 = mu_q

        s_base = self.state_proj(torch.cat([h_pred, z1], dim=-1)) # [B,S]
        s_prev_base = self.state_proj(torch.cat([h0, z0], dim=-1)) # [B,S]

        A_t = self.conn(s_prev_base, a_t) # [B,S,S]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B,S]

        prevA = self._A_prev if (self._A_prev is not None and self._A_prev.shape == A_t.shape) else None
        reg_A = self.conn.ComputeGeomReg(A_t, prevA)
        self._A_prev = A_t.detach()

        h_phys, phys_loss, _ = self.phys_refiner(h0, a_t, h_pred) # h_phys:[B,D]
        s_phys = self.state_proj(torch.cat([h_phys, z1], dim=-1)) # [B,S]

        d_tr = s_transport - s_base # [B,S]
        d_ph = s_phys - s_base # [B,S] 
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3S]
        w = F.softmax(self.mix_gate(g_in), dim=-1) # [B,3]
        s1 = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys # [B,S]

        if self._use_memory:
            with torch.no_grad():
                key = self.key_emb(raw_e, a_t) # [B,stochDim]
                mem_s = self.MemRetrieve(key) # [B,S]

                inp_rd = torch.cat([s_base, s1, a_t], dim=-1)
                h_rd = self.rdone_trunk(self.rdone_ln(inp_rd))
                r_pred_tmp = self.rew_head(h_rd).squeeze(-1)
                d_logit_tmp = self.done_head(h_rd).squeeze(-1)
                d_prob_tmp = torch.sigmoid(d_logit_tmp)

                r_score = torch.tanh(r_pred_tmp.detach().abs()).clamp(0.0, 1.0) # [B]
                d_score = d_prob_tmp.detach().clamp(0.0, 1.0) # [B]

                if self._ns_enabled and (Q_train is not None) and (pen_q is not None):
                    conf_scalar = self.NsConfidence(Q_train).mean(dim=-1) # [B]
                    imp_ns = ((1.0 - pen_q).clamp(0.0, 1.0) * (0.5 + 0.5 * conf_scalar)).clamp(0.0, 1.0)
                else:
                    imp_ns = visionIn.new_full((B,), 0.5)

                imp = (0.60 * imp_ns + 0.25 * r_score + 0.15 * d_score).clamp(0.0, 1.0)
                self.MemAdd(key.detach(), s1.detach(), imp.detach())

            if mem_s is not None:
                s1 = self.state_state_film(s1, mem_s)

        inp = torch.cat([s_base, s1, a_t], dim=-1) # [B,2S+stochDim]
        trunk = self.rdone_trunk(self.rdone_ln(inp)) # [B,256]
        r_pred = self.rew_head(trunk).squeeze(-1) # [B]
        d_logit = self.done_head(trunk).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]

        loss_recon = visionIn.new_tensor(0.0)
        recon_error = visionIn.new_zeros(B)
        recon = None
        if self.use_decoder:
            recon = self.obs_dec(s1)  # [B, visionDim]

            target = self.obs_enc[0](visionIn)  # nn.LayerNorm(visionDim)

            recon_n = F.layer_norm(
                recon,
                normalized_shape=(int(recon.size(-1)),),
                weight=self.obs_enc[0].weight,
                bias=self.obs_enc[0].bias,
                eps=self.obs_enc[0].eps,)

            recon_error = (recon_n - target).pow(2).mean(dim=-1)
            loss_recon = recon_error.mean()

        aux_moe = visionIn.new_tensor(0.0)
        if self._ns_enabled:
            aux_moe = self.ns_head_prior.GetAuxLoss() + self.ns_head_post.GetAuxLoss()

        loss_reward = F.mse_loss(r_pred, reward.to(device=device, dtype=dtype), reduction="mean")
        loss_done = F.binary_cross_entropy_with_logits(
            d_logit, done.to(device=device, dtype=dtype), reduction="mean")

        loss_kl = BalancedKL(mu_q, logstd_q, mu_p, logstd_p, alpha=alphaKl, freeNats=freeNats).mean()

        self._h = h_pred.detach()
        self._z = z1.detach()

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic
            + physCoef * phys_loss
            + reg_A
            + 1e-1 * aux_moe)

        out: Dict[str, Any] = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "loss_phys": phys_loss,
            "loss_conn_reg": reg_A,

            "h_next": h_pred,
            "z_next": z1,
            "s_next": s1,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "recon_error": recon_error,
            "mu_p": mu_p,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "logstd_q": logstd_q,}

        if self.use_decoder and recon is not None:
            out["recon"] = recon
            out["recon_target"] = visionIn

        out.update(self.PredictNextVisualFromPosterior(
            h_pred,
            z1,
            self.s4.x,
            actionEnc=None,
            sample=False,))

        return out


    def ExportWorldMemoryBank(self, topk: int = 1024, onlyVals: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        if (not getattr(self, "_use_memory", False)):
            return None

        K = int(topk)
        if K <= 0:
            return None

        B = int(self._mem_vals.size(0))
        cap = int(self._mem_vals.size(1))

        filled = self._mem_size # [B] 
        if (filled <= 0).all():
            return None

        K = min(K, cap)
        K = min(K, int(filled.min().item()))
        if K <= 0:
            return None

        ar = torch.arange(cap, device=self._mem_vals.device).view(1, cap) # [1,cap]
        valid = ar < filled.view(B, 1) # [B,cap]
        scores = self._mem_imp.masked_fill(~valid, -1e9) # [B,cap]

        _, idx = torch.topk(scores, k=K, dim=-1) # [B,K]
        sel_steps = torch.gather(self._mem_steps, 1, idx) # [B,K]
        time_order = torch.argsort(sel_steps, dim=-1, descending=True)
        idx = torch.gather(idx, 1, time_order)

        out: Dict[str, torch.Tensor] = {} 

        Dv = int(self._mem_vals.size(-1))
        out["vals"] = torch.gather(self._mem_vals, 1, idx.unsqueeze(-1).expand(B, K, Dv)).contiguous()

        if onlyVals:
            return out

        out["size"] = filled.detach().clone() # [B]
        out["idx"]  = idx.contiguous() # [B,K]
        out["steps"] = torch.gather(self._mem_steps, 1, idx).contiguous() # [B,K]

        Dk = int(self._mem_keys.size(-1))
        out["keys"] = torch.gather(self._mem_keys, 1, idx.unsqueeze(-1).expand(B, K, Dk)).contiguous()

        out["imp"]  = torch.gather(self._mem_imp, 1, idx).contiguous() # [B,K]
        return out
    


class WorldOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: nn.Module,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        *,
        maxRank: int = 64,
        maxRankSmall: int = 16,
        maxRankHuge: int = 8,
        maxRankConnHeadUV: int = 16,
        maxRankConnHeadFull: int = 8,):
        self.maxRank = int(maxRank)
        self.maxRankSmall = int(maxRankSmall)
        self.maxRankHuge = int(maxRankHuge)
        self.maxRankConnHeadUV = int(maxRankConnHeadUV)
        self.maxRankConnHeadFull = int(maxRankConnHeadFull)
        super().__init__(
            base,
            initRankEach=initRankEach,
            autoRank=autoRank,
            evThreshold=evThreshold,
            gradEma=gradEma,)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        V = int(self.base.vision_dim)
        A = int(self.base.action_dim)
        D = int(self.base.deter_dim)
        Z = int(self.base.stoch_dim)
        S = int(self.base.state_dim)
        X = int(self.base.ssm_dim)

        conn = self.base.conn
        phys = self.base.phys_refiner
        key = self.base.key_emb
        film = self.base.state_state_film

        def mk(name: str, inDim: int, outDim: int, maxRank: int) -> SiteSpec:
            inDim_i, outDim_i, maxRank_i = int(inDim), int(outDim), int(maxRank)

            def alloc(addRank: int, device: torch.device, dtype: torch.dtype):
                A_ = nn.Parameter(torch.randn(addRank, inDim_i, device=device, dtype=dtype) * 1e-4) # [r,in]
                B_ = nn.Parameter(torch.zeros(outDim_i, addRank, device=device, dtype=dtype) * 1e-4) # [out,r]
                s_ = nn.Parameter(torch.tensor(1e-2, device=device, dtype=dtype))
                return A_, B_, s_

            def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                s_eff = torch.tanh(s) * GetParametersScale(s)
                return s_eff * (b @ a) # [out,in]

            return SiteSpec(name, 1, inDim_i, outDim_i, maxRank_i, alloc, compose)

        specs: Dict[str, SiteSpec] = {}

        specs["obs_enc0"] = mk("obs_enc0", V, S, self.maxRank)
        specs["obs_enc1"] = mk("obs_enc1", S, Z, self.maxRank)

        specs["act_proj"] = mk("act_proj", A, Z, self.maxRank)

        specs["s4_in_to_ssm"] = mk("s4_in_to_ssm", 2 * Z, X, self.maxRank)
        specs["s4_ssm_to_deter"] = mk("s4_ssm_to_deter", X, D, self.maxRank)
        specs["s4_in_to_deter"] = mk("s4_in_to_deter", 2 * Z, D, self.maxRank)
        specs["s4_gate"] = mk("s4_gate", 2 * Z, X, self.maxRank)
        specs["s4_out_gate"] = mk("s4_out_gate", X, D, self.maxRank)

        specs["prior"] = mk("prior", D, 2 * Z, self.maxRank)
        specs["post"] = mk("post", D + Z, 2 * Z, self.maxRank)
        specs["state_proj"] = mk("state_proj", D + Z, S, self.maxRank)

        specs["rdone0"] = mk("rdone0", 2 * S + Z, 512, self.maxRank)
        specs["rdone1"] = mk("rdone1", 512, 256, self.maxRank)
        specs["rew"] = mk("rew", 256, 1, self.maxRankSmall)
        specs["done"] = mk("done", 256, 1, self.maxRankSmall)

        specs["obs_dec0"] = mk("obs_dec0", S, S, self.maxRank)
        specs["obs_dec1"] = mk("obs_dec1", S, V, self.maxRank)

        specs["mix_gate"] = mk("mix_gate", 3 * S, 3, self.maxRankSmall)

        specs["key_to_gb"] = mk("key_to_gb", Z, 2 * Z, self.maxRank)
        specs["key_mlp1"] = mk("key_mlp1", 4 * Z, int(key.mlp1.out_f), self.maxRank)
        specs["key_mlp2"] = mk("key_mlp2", int(key.mlp2.in_f), int(key.mlp2.out_f), self.maxRank)

        specs["ssfilm_e_to_gb"] = mk("ssfilm_e_to_gb", int(film.e_to_gb.in_f), int(film.e_to_gb.out_f), self.maxRank)
        specs["ssfilm_e_to_h"] = mk("ssfilm_e_to_h", int(film.e_to_h.in_f), int(film.e_to_h.out_f), self.maxRank)
        specs["ssfilm_delta0"] = mk("ssfilm_delta0", int(film.delta_mlp[0].in_f), int(film.delta_mlp[0].out_f), self.maxRank)
        specs["ssfilm_delta1"] = mk("ssfilm_delta1", int(film.delta_mlp[3].in_f), int(film.delta_mlp[3].out_f), self.maxRank)
        specs["ssfilm_to_gate"] = mk("ssfilm_to_gate", int(film.to_gate.in_f), int(film.to_gate.out_f), self.maxRank)

        specs["conn_enc_s"] = mk("conn_enc_s", int(conn.enc_s[1].linear.in_f), int(conn.enc_s[1].linear.out_f), self.maxRank)
        specs["conn_enc_a"] = mk("conn_enc_a", int(conn.enc_a[1].linear.in_f), int(conn.enc_a[1].linear.out_f), self.maxRank)

        specs["conn_film_gamma_a"] = mk("conn_film_gamma_a", int(conn.film_gamma_a.linear.in_f), int(conn.film_gamma_a.linear.out_f), self.maxRank)
        specs["conn_film_beta_a"] = mk("conn_film_beta_a", int(conn.film_beta_a.linear.in_f), int(conn.film_beta_a.linear.out_f), self.maxRank)

        for i, blk in enumerate(conn.blocks):
            specs[f"conn_blk{i}_ff0"] = mk(f"conn_blk{i}_ff0", int(blk.ff[0].linear.in_f), int(blk.ff[0].linear.out_f), self.maxRank)
            specs[f"conn_blk{i}_ff1"] = mk(f"conn_blk{i}_ff1", int(blk.ff[3].linear.in_f), int(blk.ff[3].linear.out_f), self.maxRank)

        if conn.use_lowrank:
            specs["conn_head_uv"] = mk("conn_head_uv", int(conn.head_uv.linear.in_f), int(conn.head_uv.linear.out_f), self.maxRankConnHeadUV)
        if conn.use_full:
            specs["conn_head_full"] = mk("conn_head_full", int(conn.head_full.linear.in_f), int(conn.head_full.linear.out_f), self.maxRankConnHeadFull)

        specs["conn_mix"] = mk("conn_mix", int(conn.mix.linear.in_f), int(conn.mix.linear.out_f), self.maxRankSmall)

        specs["phys_to_qp"] = mk("phys_to_qp", int(phys.to_qp.in_f), int(phys.to_qp.out_f), self.maxRank)
        specs["phys_from_qp"] = mk("phys_from_qp", int(phys.from_qp.in_f), int(phys.from_qp.out_f), self.maxRank)

        specs["phys_H0"] = mk("phys_H0", int(phys.H_net[0].in_f), int(phys.H_net[0].out_f), self.maxRank)
        specs["phys_H1"] = mk("phys_H1", int(phys.H_net[2].in_f), int(phys.H_net[2].out_f), self.maxRank)
        specs["phys_H2"] = mk("phys_H2", int(phys.H_net[4].in_f), int(phys.H_net[4].out_f), self.maxRankSmall)

        specs["phys_force0"] = mk("phys_force0", int(phys.force_net[0].in_f), int(phys.force_net[0].out_f), self.maxRank)
        specs["phys_force1"] = mk("phys_force1", int(phys.force_net[2].in_f), int(phys.force_net[2].out_f), self.maxRank)

        specs["phys_g_force"] = mk("phys_g_force", int(phys.g_force.in_f), int(phys.g_force.out_f), self.maxRank)
        specs["phys_g_phys"] = mk("phys_g_phys", int(phys.g_phys.in_f), int(phys.g_phys.out_f), self.maxRank)
        specs["phys_g_fuse"] = mk("phys_g_fuse", int(phys.g_fuse.in_f), int(phys.g_fuse.out_f), self.maxRank)

        return specs

    def EffW(self, gll, d2: Optional[torch.Tensor]) -> torch.Tensor:
        W = gll.target.weight
        d0 = gll.DeltaWeight()
        if d0 is not None:
            W = W + d0
        if d2 is not None:
            W = W + d2
        return W

    def Lin(self, x: torch.Tensor, gll, d2: Optional[torch.Tensor]) -> torch.Tensor:
        return F.linear(x, self.EffW(gll, d2), gll.target.bias)


    def ObsEnc(self, visionIn: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.base.obs_enc[0](visionIn)
        x = self.Lin(x, self.base.obs_enc[1], d.get("obs_enc0"))
        x = self.base.obs_enc[2](x)
        x = self.base.obs_enc[3](x)
        x = self.Lin(x, self.base.obs_enc[4], d.get("obs_enc1"))
        return x # [B,Z]

    def ActProj(self, actionEnc: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.Lin(actionEnc, self.base.act_proj[0], d.get("act_proj"))
        x = self.base.act_proj[1](x)
        x = self.base.act_proj[2](x)
        return x # [B,Z]

    def Prior(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(h, self.base.prior_net[0], d.get("prior"))

    def Post(self, hz: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(hz, self.base.post_net[0], d.get("post"))

    def StateProj(self, hz: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.base.state_proj[0](hz)
        x = self.Lin(x, self.base.state_proj[1], d.get("state_proj"))
        x = self.base.state_proj[2](x)
        return x  # [B,S]

    def RdoneTrunk(self, inp: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.Lin(inp, self.base.rdone_trunk[0], d.get("rdone0"))
        x = self.base.rdone_trunk[1](x)
        x = self.base.rdone_trunk[2](x)
        x = self.Lin(x, self.base.rdone_trunk[3], d.get("rdone1"))
        x = self.base.rdone_trunk[4](x)
        return x  # [B,256]

    def Rew(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(h, self.base.rew_head[0], d.get("rew"))

    def Done(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(h, self.base.done_head[0], d.get("done"))

    def ObsDec(self, s: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.Lin(s, self.base.obs_dec[0], d.get("obs_dec0"))
        x = self.base.obs_dec[1](x)
        x = self.Lin(x, self.base.obs_dec[2], d.get("obs_dec1"))
        return x

    def MixGate(self, g_in: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(g_in, self.base.mix_gate[0], d.get("mix_gate"))


    def S4_Step(self, zPrev: torch.Tensor, a_t: torch.Tensor, *, updateState: bool, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        s4 = self.base.s4
        u = torch.cat([zPrev, a_t], dim=-1)  # [B,2Z]

        g = torch.sigmoid(self.Lin(u, s4.gate, d.get("s4_gate"))) # [B,X]
        Bu = self.Lin(u, s4.in_to_ssm, d.get("s4_in_to_ssm")) * g # [B,X]

        x_next = s4.CayleyStep(s4.theta, s4.x, Bu, s4.dt)
        y_lin = self.Lin(x_next, s4.ssm_to_deter, d.get("s4_ssm_to_deter")) + self.Lin(u, s4.in_to_deter, d.get("s4_in_to_deter"))
        y_glu = y_lin * torch.sigmoid(self.Lin(x_next, s4.out_gate, d.get("s4_out_gate")))
        y = s4.ln_y(y_glu)
        y = y + s4.ffn(s4.ln_ffn(y))

        if updateState:
            s4.x = x_next.detach()
        return y # [B,D]

    def S4StepWithX(self, zPrev: torch.Tensor, a_t: torch.Tensor, x: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        s4 = self.base.s4
        u = torch.cat([zPrev, a_t], dim=-1)

        g = torch.sigmoid(self.Lin(u, s4.gate, d.get("s4_gate")))
        Bu = self.Lin(u, s4.in_to_ssm, d.get("s4_in_to_ssm")) * g

        x_next = s4.CayleyStep(s4.theta, x, Bu, s4.dt)
        y_lin = self.Lin(x_next, s4.ssm_to_deter, d.get("s4_ssm_to_deter")) + self.Lin(u, s4.in_to_deter, d.get("s4_in_to_deter"))
        y_glu = y_lin * torch.sigmoid(self.Lin(x_next, s4.out_gate, d.get("s4_out_gate")))
        y = s4.ln_y(y_glu)
        y = y + s4.ffn(s4.ln_ffn(y))
        return y, x_next.detach()


    def KeyEmbed(self, base_e: torch.Tensor, redius: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        ke = self.base.key_emb
        e = ke.ln_e(base_e)
        a = ke.ln_a(redius)

        gb = self.Lin(a, ke.to_gb, d.get("key_to_gb"))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)

        e_film = (1.0 + gamma) * e + beta

        feat = torch.cat([e_film, a, e_film * a, e_film - a], dim=-1)
        feat = ke.ln_feat(feat)

        h = F.silu(self.Lin(feat, ke.mlp1, d.get("key_mlp1")))
        h = ke.drop(h)
        k = self.Lin(h, ke.mlp2, d.get("key_mlp2"))

        k = F.normalize(k, dim=-1, eps=1e-6)
        return k


    def FiLMHResidual(self, h: torch.Tensor, e: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        fr = self.base.state_state_film
        h0 = fr.ln_h(h)
        e0 = fr.ln_e(e)

        gb = self.Lin(e0, fr.e_to_gb, d.get("ssfilm_e_to_gb"))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = fr.film_scale * torch.tanh(gamma)
        beta = fr.film_scale * torch.tanh(beta)

        h_film = (1.0 + gamma) * h0 + beta

        e_h = self.Lin(e0, fr.e_to_h, d.get("ssfilm_e_to_h"))
        e_h = fr.film_scale * torch.tanh(e_h)

        feat = torch.cat([h_film, e_h, h_film * e_h, h_film - e_h], dim=-1)
        feat = fr.delta_ln(feat)

        y = self.Lin(feat, fr.delta_mlp[0], d.get("ssfilm_delta0"))
        y = fr.delta_mlp[1](y) # SiLU
        y = fr.delta_mlp[2](y) # Dropout
        delta = self.Lin(y, fr.delta_mlp[3], d.get("ssfilm_delta1"))

        gate_in = torch.cat([h_film, e_h], dim=-1)
        gate = torch.sigmoid(self.Lin(gate_in, fr.to_gate, d.get("ssfilm_to_gate")))

        h_out = h + gate * delta
        if fr.use_out_ln:
            h_out = fr.out_ln(h_out)
        return h_out

    def ConnNet(self, sBase: torch.Tensor, actPrev: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        conn = self.base.conn
        B = int(sBase.size(0))

        hs = conn.enc_s[0](sBase)
        hs = self.Lin(hs, conn.enc_s[1].linear, d.get("conn_enc_s"))
        hs = conn.enc_s[2](hs)

        ha = conn.enc_a[0](actPrev)
        ha = self.Lin(ha, conn.enc_a[1].linear, d.get("conn_enc_a"))
        ha = conn.enc_a[2](ha)

        g = torch.tanh(self.Lin(ha, conn.film_gamma_a.linear, d.get("conn_film_gamma_a")))
        b = self.Lin(ha, conn.film_beta_a.linear, d.get("conn_film_beta_a"))

        h = hs
        for i, blk in enumerate(conn.blocks):
            y = (1.0 + g) * h + b
            y = blk.ln(y)

            y = self.Lin(y, blk.ff[0].linear, d.get(f"conn_blk{i}_ff0"))
            y = blk.ff[1](y) 
            y = blk.ff[2](y) 
            y = self.Lin(y, blk.ff[3].linear, d.get(f"conn_blk{i}_ff1"))

            h = h + blk.alpha * y

        A_list: List[torch.Tensor] = []

        if conn.use_lowrank:
            uv = self.Lin(h, conn.head_uv.linear, d.get("conn_head_uv"))
            U, Vv = uv.split(conn.S * conn.r, dim=-1)
            U = U.view(B, conn.S, conn.r)
            Vv = Vv.view(B, conn.S, conn.r)
            A_list.append(U @ Vv.transpose(1, 2) - Vv @ U.transpose(1, 2))

        if conn.use_full:
            M = self.Lin(h, conn.head_full.linear, d.get("conn_head_full")).view(B, conn.S, conn.S)
            A_list.append(0.5 * (M - M.transpose(1, 2)))

        if not A_list:
            A = torch.zeros(B, conn.S, conn.S, device=sBase.device, dtype=sBase.dtype)
        elif len(A_list) == 1:
            A = A_list[0]
        else:
            w = F.softmax(self.Lin(h, conn.mix.linear, d.get("conn_mix")), dim=-1)
            A = w[:, :1].view(B, 1, 1) * A_list[0] + w[:, 1:2].view(B, 1, 1) * A_list[1]

        if conn.norm_clip and conn.norm_clip > 0:
            fro = A.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), conn.norm_clip / fro).view(B, 1, 1)
            A = A * scale

        return A # [B,S,S]


    def PhysRefiner(self, hPrev: torch.Tensor, action: torch.Tensor, hS4: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]):
        pr = self.base.phys_refiner
        training_mode = bool(self.training)
        create_graph = bool(training_mode)
        dt_sub = pr.dt / float(pr.substeps)

        if training_mode:
            H_start = None
            H_end = None
            work_acc = hPrev.new_zeros(hPrev.size(0), 1)
            smooth_acc = hPrev.new_tensor(0.0)

        def H_net(qp: torch.Tensor) -> torch.Tensor:
            x = self.Lin(qp, pr.H_net[0], d.get("phys_H0"))
            x = pr.H_net[1](x)  
            x = self.Lin(x, pr.H_net[2], d.get("phys_H1"))
            x = pr.H_net[3](x)  
            x = self.Lin(x, pr.H_net[4], d.get("phys_H2"))
            return x # [B,1]

        def Force_net(fa: torch.Tensor) -> torch.Tensor:
            x = self.Lin(fa, pr.force_net[0], d.get("phys_force0"))
            x = pr.force_net[1](x) 
            x = self.Lin(x, pr.force_net[2], d.get("phys_force1"))
            return x # [B,Q]

        def HAndGrad(qp: torch.Tensor, create_graph_: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            H = H_net(qp) # [B,1]
            g = torch.autograd.grad(
                H.sum(), qp,
                create_graph=create_graph_,
                retain_graph=create_graph_,
                allow_unused=False,
            )[0]  # [B,P]
            return H, g

        def SymplecticLeapfrog(q: torch.Tensor, p: torch.Tensor, dt: float, create_graph_: bool):
            qp0 = torch.cat([q, p], dim=-1)
            H0, g0 = HAndGrad(qp0, create_graph_=create_graph_)
            dH_dq0, _ = g0.chunk(2, dim=-1)

            p_half = p - 0.5 * dt * dH_dq0

            qp_mid = torch.cat([q, p_half], dim=-1)
            _, gm = HAndGrad(qp_mid, create_graph_=create_graph_)
            _, dH_dp_mid = gm.chunk(2, dim=-1)

            q1 = q + dt * dH_dp_mid

            qp_for_p = torch.cat([q1, p_half], dim=-1)
            H1, g2 = HAndGrad(qp_for_p, create_graph_=create_graph_)
            dH_dq2, _ = g2.chunk(2, dim=-1)

            p1 = p_half - 0.5 * dt * dH_dq2
            return q1, p1, H0, H1, dH_dp_mid

        def ClampResidual(delta_: torch.Tensor, base_: torch.Tensor, ratio: float) -> torch.Tensor:
            eps = 1e-8
            dnorm = delta_.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
            bnorm = base_.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-3
            maxn = ratio * bnorm + eps
            scale = (maxn / dnorm).clamp(max=1.0)
            return delta_ * scale

        with torch.enable_grad():
            qp = self.Lin(hPrev, pr.to_qp, d.get("phys_to_qp"))

            if not qp.requires_grad:
                qp = qp.detach().requires_grad_(True)
            
            q, p = qp.chunk(2, dim=-1)

            for i in range(pr.substeps):
                h_cur = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))

                fa0_inp = torch.cat([h_cur, action], dim=-1)
                F0 = Force_net(fa0_inp) * torch.sigmoid(self.Lin(fa0_inp, pr.g_force, d.get("phys_g_force")))

                if pr.dampP > 0.0:
                    p = p * torch.exp(-pr.dampP * dt_sub)

                p = p + 0.5 * dt_sub * F0

                q, p, H0, H1, dH_dp_mid = SymplecticLeapfrog(q, p, dt_sub, create_graph_=create_graph)

                if training_mode:
                    if i == 0:
                        H_start = H0
                    H_end = H1

                h_mid = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))
                fa1_inp = torch.cat([h_mid, action], dim=-1)
                F1 = Force_net(fa1_inp) * torch.sigmoid(self.Lin(fa1_inp, pr.g_force, d.get("phys_g_force")))

                p = p + 0.5 * dt_sub * F1

                if training_mode:
                    smooth_acc = smooth_acc + (F1 - F0).pow(2).mean()
                    F_avg = 0.5 * (F0 + F1)
                    work_acc = work_acc + (dH_dp_mid * F_avg).sum(dim=-1, keepdim=True) * dt_sub

        h_phys_raw = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))

        d_corr = h_phys_raw - hS4
        gph = torch.sigmoid(self.Lin(torch.cat([hPrev, action], dim=-1), pr.g_phys, d.get("phys_g_phys")))
        d_corr = d_corr * gph

        base_ = hS4 - hPrev
        d_corr = ClampResidual(d_corr, base_, ratio=pr.clamp_ratio)

        alpha = torch.sigmoid(self.Lin(torch.cat([hPrev, action, hS4], dim=-1), pr.g_fuse, d.get("phys_g_fuse")))
        h_fused = hS4 + alpha * d_corr

        if not training_mode:
            return h_fused, None, None

        if (H_start is None) or (H_end is None):
            e_work = hPrev.new_tensor(0.0)
        else:
            denom = H_start.detach().abs().mean().clamp_min(1e-6)
            dH = (H_end - H_start)
            e_work = ((dH - work_acc) / denom).pow(2).mean()

        e_smooth = smooth_acc / float(pr.substeps)
        e_delta = d_corr.pow(2).mean()

        loss = pr.l_work * e_work + pr.l_smooth * e_smooth + pr.l_delta * e_delta
        aux = {"L_work": e_work.detach(), "L_smooth": e_smooth.detach(), "L_delta": e_delta.detach()}
        return h_fused, loss, aux


    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,) -> Dict[str, torch.Tensor]:

        visionIn = x
        keysVec = kwargs["keysVec"]
        mouseClick = kwargs["mouseClick"]
        mouseSeq = kwargs["mouseSeq"]
        reward = kwargs["reward"]
        done = kwargs["done"]

        sample = kwargs.get("sample", True)
        alphaKl = kwargs.get("alphaKl", 0.8)
        freeNats = kwargs.get("freeNats", 1.0)
        reconCoef = kwargs.get("reconCoef", 1.0)
        rewardCoef = kwargs.get("rewardCoef", 1.0)
        doneCoef = kwargs.get("doneCoef", 1.0)
        nsCoef = kwargs.get("nsCoef", 1.0)
        nsDistillCoef = kwargs.get("nsDistillCoef", 1e-2)
        nsPriorLogicCoef = kwargs.get("nsPriorLogicCoef", 1e-3)
        physCoef = kwargs.get("physCoef", 1e-4)

        B = int(visionIn.size(0))
        device, dtype = self.base.device, self.base.dtype
        self.base.EnsureB(B, device, dtype)

        d = deltasPerLayer[0] if (deltasPerLayer is not None) else {}

        h0 = self.base._h
        z0 = self.base._z

        a_enc = self.base.action_encoder(keysVec, mouseSeq, mouseClick) # [B, actionDim]
        a_t = self.ActProj(a_enc, d) # [B, stochDim]

        h_pred = self.S4_Step(z0, a_t, updateState=True, d=d) # [B, deterDim]

        mu_p, logstd_p = self.Prior(h_pred, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        logits_pr = None
        P_pr_train = None
        ns_prior_logic = visionIn.new_tensor(0.0)

        if self.base._ns_enabled:
            logits_pr = self.base.ns_head_prior(h_pred)
            P_pr_raw = torch.sigmoid(logits_pr)
            P_pr_train = self.base.NsProjectProbs(P_pr_raw)

            dmu_p = self.base.ns_to_delta_mu(P_pr_train)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_pred, dmu_p], dim=-1)))

            _, pen_pr = self.base.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)

            conf = self.base.NsConfidence(P_pr_train).mean(dim=-1, keepdim=True)
            gate_scale = (1.0 - 0.40 * pen_pr.view(-1, 1)) * (0.6 + 0.4 * conf)
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)

            mu_p = mu_p + gate * dmu_p

            if nsPriorLogicCoef > 0.0:
                ns_prior_logic, _ = self.base.NsLogicLosses(P_pr_train)

        raw_e = self.ObsEnc(visionIn, d) # [B, stochDim]

        mu_q, logstd_q = self.Post(torch.cat([h_pred, raw_e], dim=-1), d).chunk(2, dim=-1)
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        ns_loss = visionIn.new_tensor(0.0)
        ns_distill = visionIn.new_tensor(0.0)
        logits_q = None
        Q_train = None
        pen_q = None

        if self.base._ns_enabled:
            logits_q = self.base.ns_head_post(torch.cat([h_pred, raw_e], dim=-1))
            P_q_raw = torch.sigmoid(logits_q)
            Q_train = self.base.NsProjectProbs(P_q_raw)

            dmu_q = self.base.ns_to_delta_mu(Q_train)
            base_gate_q = torch.sigmoid(self.base.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu_q], dim=-1)))

            _, pen_q = self.base.NsProjectRuntime(P_q_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)

            conf_q = self.base.NsConfidence(Q_train).mean(dim=-1, keepdim=True)
            gate_scale_q = (1.0 - 0.40 * pen_q.view(-1, 1)) * (0.6 + 0.4 * conf_q)
            gate_q = (base_gate_q * gate_scale_q).clamp(0.0, 1.0)

            mu_q = mu_q + gate_q * dmu_q

            ns_loss, _ = self.base.NsLogicLosses(Q_train)

            if (logits_pr is not None) and (nsDistillCoef > 0.0):
                with torch.no_grad():
                    P_teacher = torch.sigmoid(logits_q)
                ns_distill = F.binary_cross_entropy_with_logits(logits_pr, P_teacher, reduction="mean")

        if sample:
            z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
        else:
            z1 = mu_q

        s_base = self.StateProj(torch.cat([h_pred, z1], dim=-1), d)
        s_prev_base = self.StateProj(torch.cat([h0, z0], dim=-1), d)

        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)

        prevA = self.base._A_prev if (self.base._A_prev is not None and self.base._A_prev.shape == A_t.shape) else None
        reg_A = self.base.conn.ComputeGeomReg(A_t, prevA)
        self.base._A_prev = A_t.detach()

        h_phys, phys_loss, _ = self.PhysRefiner(h0, a_t, h_pred, d)
        if phys_loss is None:
            phys_loss = visionIn.new_zeros(())
        s_phys = self.StateProj(torch.cat([h_phys, z1], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1)
        w = F.softmax(self.MixGate(g_in, d), dim=-1)
        s1 = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        if self.base._use_memory:
            with torch.no_grad():
                key = self.KeyEmbed(raw_e, a_t, d)
                mem_s = self.base.MemRetrieve(key)

                inp_rd = torch.cat([s_base, s1, a_t], dim=-1)
                h_rd = self.RdoneTrunk(self.base.rdone_ln(inp_rd), d)
                r_pred_tmp = self.Rew(h_rd, d).squeeze(-1)
                d_logit_tmp = self.Done(h_rd, d).squeeze(-1)
                d_prob_tmp = torch.sigmoid(d_logit_tmp)

                r_score = torch.tanh(r_pred_tmp.detach().abs()).clamp(0.0, 1.0)
                d_score = d_prob_tmp.detach().clamp(0.0, 1.0)

                if self.base._ns_enabled and (Q_train is not None) and (pen_q is not None):
                    conf_scalar = self.base.NsConfidence(Q_train).mean(dim=-1)
                    imp_ns = ((1.0 - pen_q).clamp(0.0, 1.0) * (0.5 + 0.5 * conf_scalar)).clamp(0.0, 1.0)
                else:
                    imp_ns = visionIn.new_full((B,), 0.5)

                imp = (0.60 * imp_ns + 0.25 * r_score + 0.15 * d_score).clamp(0.0, 1.0)
                self.base.MemAdd(key.detach(), s1.detach(), imp.detach())

            if mem_s is not None:
                s1 = self.FiLMHResidual(s1, mem_s, d)

        inp = torch.cat([s_base, s1, a_t], dim=-1)
        trunk = self.RdoneTrunk(self.base.rdone_ln(inp), d)
        r_pred = self.Rew(trunk, d).squeeze(-1)
        d_logit = self.Done(trunk, d).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)

        loss_recon = visionIn.new_tensor(0.0)
        recon_error = visionIn.new_zeros(B)
        recon = None
        if self.base.use_decoder:
            recon = self.ObsDec(s1, d)

            target = self.base.obs_enc[0](visionIn)

            recon_n = F.layer_norm(
                recon,
                normalized_shape=(int(recon.size(-1)),),
                weight=self.base.obs_enc[0].weight,
                bias=self.base.obs_enc[0].bias,
                eps=self.base.obs_enc[0].eps,)

            recon_error = (recon_n - target).pow(2).mean(dim=-1)
            loss_recon = recon_error.mean()

        aux_moe = visionIn.new_tensor(0.0)
        if self.base._ns_enabled:
            aux_moe = self.base.ns_head_prior.GetAuxLoss() + self.base.ns_head_post.GetAuxLoss()

        loss_reward = F.mse_loss(r_pred, reward.to(device=device, dtype=dtype), reduction="mean")
        loss_done = F.binary_cross_entropy_with_logits(d_logit, done.to(device=device, dtype=dtype), reduction="mean")
        loss_kl = BalancedKL(mu_q, logstd_q, mu_p, logstd_p, alpha=alphaKl, freeNats=freeNats).mean()

        self.base._h = h_pred.detach()
        self.base._z = z1.detach()

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic
            + physCoef * phys_loss
            + reg_A
            + 1e-1 * aux_moe)

        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "loss_phys": phys_loss,
            "loss_conn_reg": reg_A,
            "h_next": h_pred,
            "z_next": z1,
            "s_next": s1,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "recon_error": recon_error,
            "mu_p": mu_p,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "logstd_q": logstd_q,}
        
        if self.base.use_decoder and recon is not None:
            out["recon"] = recon
            out["recon_target"] = visionIn

        out.update(self.PredictNextVisualFromPosteriorWithDeltas(
            h_pred,
            z1,
            self.base.s4.x,
            actionEnc=None,
            sample=False,
            d=d,))
        return out

    @torch.no_grad()
    def StepPriorWithDeltas(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,
        deltasPerLayer: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        **kwargs,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = int(actionEnc.size(0))
        device, dtype = self.base.device, self.base.dtype
        self.base.EnsureB(B, device, dtype)

        d = deltasPerLayer[0] if (deltasPerLayer is not None) else {}

        if hPrev is None or zPrev is None or s4xPrev is None:
            hPrev = torch.zeros(B, self.base.deter_dim, device=device, dtype=dtype)
            zPrev = torch.zeros(B, self.base.stoch_dim, device=device, dtype=dtype)
            s4xPrev = torch.zeros(B, self.base.ssm_dim, device=device, dtype=dtype)

        a_t = self.ActProj(actionEnc, d)
        h_next, s4x_next = self.S4StepWithX(zPrev, a_t, s4xPrev, d)

        mu_p, logstd_p = self.Prior(h_next, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self.base._ns_enabled:
            ns_logits = self.base.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.base.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.base.NsConfidence(Q).mean(dim=-1, keepdim=True)

            dmu = self.base.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)

            mu_p = mu_p + gate * dmu

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.StateProj(torch.cat([h_next, z_next], dim=-1), d)
        s_prev_base = self.StateProj(torch.cat([hPrev, zPrev], dim=-1), d)

        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)

        h_phys, _, _ = self.PhysRefiner(hPrev, a_t, h_next, d)
        s_phys = self.StateProj(torch.cat([h_phys, z_next], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1)

        w = F.softmax(self.MixGate(g_in, d), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        inp = torch.cat([s_base, s_next, a_t], dim=-1)
        h = self.RdoneTrunk(self.base.rdone_ln(inp), d)

        r_pred = self.Rew(h, d).squeeze(-1)
        d_logit = self.Done(h, d).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)

        return h_next, z_next, s_next, s4x_next, r_pred, d_prob

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.base.ExportState()

    def PriorRolloutFromStateAction(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: Optional[torch.Tensor] = None,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        d = self.ComposeLayerDelta(0)
        return self.PriorRolloutFromStateActionWithDeltas(
            hPrev,
            zPrev,
            s4xPrev,
            actionEnc=actionEnc,
            sample=sample,
            d=d)

    def PriorRolloutFromStateActionWithDeltas(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: Optional[torch.Tensor],
        sample: bool,
        d: Dict[str, Optional[torch.Tensor]],) -> Dict[str, torch.Tensor]:
        s_prev_base = self.StateProj(torch.cat([hPrev, zPrev], dim=-1), d)
        if actionEnc is None:
            actionEnc = self.base.future_action_head(s_prev_base)

        a_t = self.ActProj(actionEnc, d)
        h_next, x_next = self.S4StepWithX(zPrev, a_t, s4xPrev, d)
        mu_p, logstd_p = self.Prior(h_next, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self.base._ns_enabled:
            ns_logits = self.base.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.base.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.base.NsConfidence(Q).mean(dim=-1, keepdim=True)
            dmu = self.base.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            mu_p = mu_p + (base_gate * gate_scale).clamp(0.0, 1.0) * dmu

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.StateProj(torch.cat([h_next, z_next], dim=-1), d)
        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)
        h_phys, _, _ = self.PhysRefiner(hPrev, a_t, h_next, d)
        s_phys = self.StateProj(torch.cat([h_phys, z_next], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        w = F.softmax(self.MixGate(torch.cat([s_base, d_tr, d_ph], dim=-1), d), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        trunk = self.RdoneTrunk(self.base.rdone_ln(torch.cat([s_base, s_next, a_t], dim=-1)), d)
        return {
            "h_next": h_next,
            "z_next": z_next,
            "x_next": x_next,
            "s_next": s_next,
            "action_enc": actionEnc,
            "r_pred": self.Rew(trunk, d).squeeze(-1),
            "d_prob": torch.sigmoid(self.Done(trunk, d).squeeze(-1)),
            "d_tr": d_tr,
            "d_ph": d_ph,}

    def PredictNextVisualFromPosterior(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        actionEnc: Optional[torch.Tensor] = None,
        sample: bool = False,) -> Dict[str, Any]:
        d = self.ComposeLayerDelta(0)
        return self.PredictNextVisualFromPosteriorWithDeltas(
            h,
            z,
            s4x,
            actionEnc=actionEnc,
            sample=sample,
            d=d)

    def PredictNextVisualFromPosteriorWithDeltas(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        actionEnc: Optional[torch.Tensor],
        sample: bool,
        d: Dict[str, Optional[torch.Tensor]],) -> Dict[str, Any]:
        rollout = self.PriorRolloutFromStateActionWithDeltas(
            h,
            z,
            s4x,
            actionEnc=actionEnc,
            sample=sample,
            d=d)
        pred = self.base.BuildPredictedVisual(rollout["s_next"])
        pred["prior_rollout"] = rollout
        return pred

    def ComputePredictionLoss(
        self,
        predictedVisual: PredictedVisualPack,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        precision: torch.Tensor,) -> Dict[str, torch.Tensor]:
        return self.base.ComputePredictionLoss(
            predictedVisual=predictedVisual,
            reconstructedVisualState=reconstructedVisualState,
            targetVisualState=targetVisualState,
            precision=precision,)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        r = int(a.size(0))
        if r <= 0 or a.numel() == 0 or b.numel() == 0 or abs(float(scale)) < 1e-12:
            return False

        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "obs_enc0":
            self.base.obs_enc[1].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "obs_enc1":
            self.base.obs_enc[4].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "act_proj":
            self.base.act_proj[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "s4_in_to_ssm":
            self.base.s4.in_to_ssm.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_ssm_to_deter":
            self.base.s4.ssm_to_deter.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_in_to_deter":
            self.base.s4.in_to_deter.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_gate":
            self.base.s4.gate.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_out_gate":
            self.base.s4.out_gate.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "prior":
            self.base.prior_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "post":
            self.base.post_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "state_proj":
            self.base.state_proj[1].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "rdone0":
            self.base.rdone_trunk[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "rdone1":
            self.base.rdone_trunk[3].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "rew":
            self.base.rew_head[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "done":
            self.base.done_head[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "obs_dec0":
            self.base.obs_dec[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "obs_dec1":
            self.base.obs_dec[2].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "mix_gate":
            self.base.mix_gate[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "key_to_gb":
            self.base.key_emb.to_gb.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "key_mlp1":
            self.base.key_emb.mlp1.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "key_mlp2":
            self.base.key_emb.mlp2.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "ssfilm_e_to_gb":
            self.base.state_state_film.e_to_gb.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_e_to_h":
            self.base.state_state_film.e_to_h.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_delta0":
            self.base.state_state_film.delta_mlp[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_delta1":
            self.base.state_state_film.delta_mlp[3].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_to_gate":
            self.base.state_state_film.to_gate.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "conn_enc_s":
            self.base.conn.enc_s[1].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_enc_a":
            self.base.conn.enc_a[1].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_film_gamma_a":
            self.base.conn.film_gamma_a.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_film_beta_a":
            self.base.conn.film_beta_a.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site.startswith("conn_blk") and (("_ff0" in site) or ("_ff1" in site)):
            s0 = site.replace("conn_blk", "")
            i_str, which = s0.split("_", 1)
            i = int(i_str)
            if which == "ff0":
                self.base.conn.blocks[i].ff[0].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
            elif which == "ff1":
                self.base.conn.blocks[i].ff[3].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
            else:
                raise ValueError(site)

        elif site == "conn_head_uv":
            self.base.conn.head_uv.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_head_full":
            self.base.conn.head_full.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_mix":
            self.base.conn.mix.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_to_qp":
            self.base.phys_refiner.to_qp.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_from_qp":
            self.base.phys_refiner.from_qp.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_H0":
            self.base.phys_refiner.H_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_H1":
            self.base.phys_refiner.H_net[2].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_H2":
            self.base.phys_refiner.H_net[4].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_force0":
            self.base.phys_refiner.force_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_force1":
            self.base.phys_refiner.force_net[2].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_g_force":
            self.base.phys_refiner.g_force.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_g_phys":
            self.base.phys_refiner.g_phys.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_g_fuse":
            self.base.phys_refiner.g_fuse.Grow(r, init=init, freezeOld=self.freezeOldPar)

        else:
            raise ValueError(f"Unknown site: {site}")

        return True





class TestWorldMTool:
    def __init__(self, device: Optional[str] = None, seed: int = 0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(int(seed))

        self.key_dim = int(NUM_DISCRETE_KEYS)

        self.wm = RSSMWorldModel(
            visionDim=256,
            actionDim=128,
            deterDim=128,
            stochDim=32,
            stateDim=128,
            ssmDim=128,
            useDecoder=True,
            useMemory=False,
            nsEnabled=True,).to(self.device)

        self.wm.ResetState(batchSize=4)

    def TestActionEncoder(self) -> bool:
        try:
            torch.manual_seed(0)
            B = 4
            K = self.key_dim

            enc = ActionEncoder(
                numDiscrete=K, mouseDim=2, clickDim=2,
                outDim=64, hidden=128, dropout=0.0).to(self.device).train()

            keys = torch.zeros(B, K, device=self.device)
            keys[:, min(5, K - 1)] = 1.0
            keys[:, min(17, K - 1)] = 1.0

            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)
            click[:, 0] = 1.0

            y = enc(keys, mouse, click)
            if y.dim() != 2 or y.shape[0] != B or y.shape[1] != 64:
                print(f"ActionEncoder FAILED: y shape={tuple(y.shape)} expected={(B,64)}")
                return False

            loss = (y ** 2).mean()
            enc.zero_grad(set_to_none=True)
            loss.backward()

            any_grad = False
            for p in enc.parameters():
                if p.grad is not None and torch.isfinite(p.grad).all().item() and (p.grad.abs().sum().item() > 0):
                    any_grad = True
                    break

            ok = bool(any_grad)
            print(f"ActionEncoder {'passed' if ok else 'failed'} | loss={float(loss.item()):.6f}")
            return ok

        except Exception as e:
            print(f"ActionEncoder FAILED: {type(e).__name__}: {e}")
            return False


    def TestRSSMStepPosterior(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.eval()

            B = 4
            wm.ResetState(batchSize=B)

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(9, self.key_dim - 1)] = 1.0
            keys[:, min(33, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)

            action_enc = wm.action_encoder(keys, mouse, click)

            h_before, z_before, x_before = wm.ExportState()
            out = wm.StepPosterior(vision, action_enc, sample=False)
            h_after, z_after, x_after = wm.ExportState()

            ok_shapes = (
                out["h_next"].shape == (B, wm.deter_dim)
                and out["z_next"].shape == (B, wm.stoch_dim)
                and out["x_next"].shape == (B, wm.ssm_dim)
                and out["s_next"].shape == (B, wm.state_dim)
                and out["r_pred"].shape == (B,)
                and out["d_prob"].shape == (B,)
                and out["mu_q"].shape == (B, wm.stoch_dim)
                and out["logstd_q"].shape == (B, wm.stoch_dim))

            changed = (
                (not torch.allclose(h_before, h_after))
                or (not torch.allclose(z_before, z_after))
                or (not torch.allclose(x_before, x_after)))

            dmin = float(out["d_prob"].min().item())
            dmax = float(out["d_prob"].max().item())
            in_range = (dmin >= 0.0) and (dmax <= 1.0)

            ns_ok = True
            if getattr(wm, "_ns_enabled", False):
                ns_ok = (
                    ("ns_logits" in out) and ("ns_Q" in out) and ("ns_pen" in out)
                    and out["ns_logits"].shape == (B, wm._ns_K)
                    and out["ns_Q"].shape == (B, wm._ns_K)
                    and out["ns_pen"].shape == (B,)
                    and (out["ns_Q"].min().item() >= 0.0) and (out["ns_Q"].max().item() <= 1.0)
                    and (out["ns_pen"].min().item() >= 0.0) and (out["ns_pen"].max().item() <= 1.0))

            recon_ok = True
            if getattr(wm, "use_decoder", False):
                recon_ok = ("recon" in out) and (out["recon"].shape == (B, wm.vision_dim))

            ok = bool(ok_shapes and changed and in_range and ns_ok and recon_ok)
            print(f"RSSM StepPosterior {'passed' if ok else 'failed'} | d_prob=[{dmin:.3f},{dmax:.3f}]")
            return ok

        except Exception as e:
            print(f"RSSM StepPosterior FAILED: {type(e).__name__}: {e}")
            return False


    def TestRSSMStepPriorOnly(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.eval()

            B = 4
            wm.ResetState(batchSize=B)

            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(12, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)
            action_enc = wm.action_encoder(keys, mouse, click)

            hPrev = torch.randn(B, wm.deter_dim, device=self.device)
            zPrev = torch.randn(B, wm.stoch_dim, device=self.device)
            s4xPrev = torch.randn(B, wm.ssm_dim, device=self.device)

            h_before, z_before, x_before = wm.ExportState()
            h1, z1, s1, x1, r, d = wm.StepPriorOnly(hPrev, zPrev, s4xPrev, action_enc, sample=False)
            h_after, z_after, x_after = wm.ExportState()

            ok_shapes = (
                h1.shape == (B, wm.deter_dim)
                and z1.shape == (B, wm.stoch_dim)
                and s1.shape == (B, wm.state_dim)
                and x1.shape == (B, wm.ssm_dim)
                and r.shape == (B,)
                and d.shape == (B,))

            not_written = (
                torch.allclose(h_before, h_after)
                and torch.allclose(z_before, z_after)
                and torch.allclose(x_before, x_after))

            dmin = float(d.min().item())
            dmax = float(d.max().item())
            in_range = (dmin >= 0.0) and (dmax <= 1.0)

            ok = bool(ok_shapes and not_written and in_range)
            print(f"RSSM StepPriorOnly {'passed' if ok else 'failed'} | d=[{dmin:.3f},{dmax:.3f}]")
            return ok

        except Exception as e:
            print(f"RSSM StepPriorOnly FAILED: {type(e).__name__}: {e}")
            return False


    def TestForwardTrainFiniteGrad(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.train()

            B = 3
            wm.ResetState(batchSize=B)

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(7, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)

            reward = torch.zeros(B, device=self.device)
            done = torch.zeros(B, device=self.device)

            sample = False
            alphaKl = 0.8
            freeNats = 1.0
            reconCoef = 1.0
            rewardCoef = 1.0
            doneCoef = 1.0
            nsCoef = 1.0
            nsDistillCoef = 1e-2
            nsPriorLogicCoef = 1e-3
            physCoef = 1e-4

            out = wm.ForwardTrain(
                vision, keys, click, mouse, reward, done,
                sample=sample,
                alphaKl=alphaKl, freeNats=freeNats,
                reconCoef=reconCoef, rewardCoef=rewardCoef, doneCoef=doneCoef,
                nsCoef=nsCoef, nsDistillCoef=nsDistillCoef, nsPriorLogicCoef=nsPriorLogicCoef,
                physCoef=physCoef,)

            loss = out["loss"]
            if not torch.isfinite(loss).item():
                print("ForwardTrain FAILED: loss not finite")
                return False

            aux_moe = vision.new_zeros(())
            if getattr(wm, "_ns_enabled", False):
                aux_moe = wm.ns_head_prior.GetAuxLoss() + wm.ns_head_post.GetAuxLoss()

            terms = [
                ("recon", reconCoef, out["loss_recon"]),
                ("reward", rewardCoef, out["loss_reward"]),
                ("done", doneCoef, out["loss_done"]),
                ("kl", 1.0, out["loss_kl"]),
                ("ns", nsCoef, out["loss_ns"]),
                ("ns_distill", nsDistillCoef, out["loss_ns_distill"]),
                ("ns_prior_logic", nsPriorLogicCoef, out["loss_ns_prior_logic"]),
                ("phys", physCoef, out["loss_phys"]),
                ("conn_reg", 1.0, out["loss_conn_reg"]),
                ("aux_moe", 1e-1, aux_moe),]

            rows = []
            for name, coef, v in terms:
                v_f = float(v.detach().item()) if torch.is_tensor(v) else float(v)
                contrib = float(coef) * v_f
                rows.append((abs(contrib), name, float(coef), v_f, contrib))

            rows.sort(reverse=True, key=lambda x: x[0])

            total_calc = sum(r[4] for r in rows)
            total_out = float(loss.detach().item())

            print("\n[ForwardTrain loss breakdown]")
            print(f"total(out) = {total_out:.6f} | total(calc) = {total_calc:.6f} | diff = {total_out - total_calc:.6f}")
            for _, name, coef, raw, contrib in rows:
                print(f"  {name:<14} raw={raw:>12.6f}  coef={coef:<10g}  contrib={contrib:>12.6f}")

            top_name = rows[0][1]
            if top_name == "kl":
                mu_p = out["mu_p"]; mu_q = out["mu_q"]
                ls_p = out["logstd_p"]; ls_q = out["logstd_q"]
                print("\n[KL debug stats]")
                print(f" mu_p: mean|x|={mu_p.abs().mean().item():.4f} max|x|={mu_p.abs().max().item():.4f}")
                print(f" mu_q: mean|x|={mu_q.abs().mean().item():.4f} max|x|={mu_q.abs().max().item():.4f}")
                print(f" logstd_p: min={ls_p.min().item():.4f} max={ls_p.max().item():.4f}")
                print(f" logstd_q: min={ls_q.min().item():.4f} max={ls_q.max().item():.4f}")

            elif top_name == "recon" and ("recon" in out):
                recon = out["recon"]
                target = wm.obs_enc[0](vision) 
                recon_n = F.layer_norm(
                    recon,
                    normalized_shape=(int(recon.size(-1)),),
                    weight=wm.obs_enc[0].weight,
                    bias=wm.obs_enc[0].bias,
                    eps=wm.obs_enc[0].eps,)
                
                print("\n[Recon debug stats]")
                print(f" recon: mean={recon.mean().item():.4f} std={recon.std().item():.4f} max|x|={recon.abs().max().item():.4f}")
                print(f" recon_n: mean={recon_n.mean().item():.4f} std={recon_n.std().item():.4f} max|x|={recon_n.abs().max().item():.4f}")
                print(f" target: mean={target.mean().item():.4f} std={target.std().item():.4f} max|x|={target.abs().max().item():.4f}")

            wm.zero_grad(set_to_none=True)
            loss.backward()

            any_grad = False
            for p in wm.parameters():
                if p.requires_grad and (p.grad is not None) and torch.isfinite(p.grad).all().item() and (p.grad.abs().sum().item() > 0):
                    any_grad = True
                    break

            ok = bool(any_grad)
            print(f"\nForwardTrain finite&grad {'passed' if ok else 'failed'} | loss={total_out:.6f}")
            return ok

        except Exception as e:
            print(f"ForwardTrain finite&grad FAILED: {type(e).__name__}: {e}")
            return False

    def TestWorldForwardIOShapes(self) -> bool:
        try:
            torch.manual_seed(0)

            wm = RSSMWorldModel(
                visionDim=1024,
                actionDim=256,
                deterDim=512,
                stochDim=64,
                stateDim=512,
                ssmDim=64,
                useDecoder=True,
                useMemory=False,
                nsEnabled=True,).to(self.device)
            wm.train()

            B = 2
            wm.ResetState(batchSize=B)

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(7, self.key_dim - 1)] = 1.0
            keys[:, min(33, self.key_dim - 1)] = 1.0
            click = torch.zeros(B, 2, device=self.device)
            mouse = torch.randn(B, 2, device=self.device)
            reward = torch.randn(B, device=self.device)
            done = torch.zeros(B, device=self.device)

            def print_shape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            with torch.no_grad():
                print_shape("input.visionIn", vision)
                print_shape("input.keysVec", keys)
                print_shape("input.mouseClick", click)
                print_shape("input.mouseSeq", mouse)
                print_shape("input.reward", reward)
                print_shape("input.done", done)

            out = wm.ForwardTrain(
                vision,
                keys,
                click,
                mouse,
                reward,
                done,
                sample=False)

            for key, value in out.items():
                if isinstance(value, torch.Tensor):
                    print_shape(f"output.{key}", value)

            expected = {
                "h_next": (B, wm.deter_dim),
                "z_next": (B, wm.stoch_dim),
                "s_next": (B, wm.state_dim),
                "r_pred": (B,),
                "d_prob": (B,),
                "mu_p": (B, wm.stoch_dim),
                "logstd_p": (B, wm.stoch_dim),
                "mu_q": (B, wm.stoch_dim),
                "logstd_q": (B, wm.stoch_dim),}

            for name, shape in expected.items():
                if tuple(out[name].shape) != shape:
                    print(f"World forward IO shape mismatch: {name}={tuple(out[name].shape)} expected={shape}")
                    return False

            print("World forward IO shapes test passed.")
            return True

        except Exception as e:
            print(f"World forward IO shapes FAILED: {type(e).__name__}: {e}")
            return False


    def TestLossDecrease(self, steps: int = 80, lr: float = 1e-3) -> bool:
        try:
            torch.manual_seed(0)

            wm = RSSMWorldModel(
                visionDim=128,
                actionDim=64,
                deterDim=64,
                stochDim=16,
                stateDim=64,
                ssmDim=64,
                useDecoder=True,
                useMemory=False,
                nsEnabled=False,).to(self.device).train()

            B = 6
            wm.ResetState(batchSize=B)

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(3, self.key_dim - 1)] = 1.0
            keys[:, min(25, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)

            reward = torch.zeros(B, device=self.device)
            done = torch.zeros(B, device=self.device)

            opt = torch.optim.Adam(wm.parameters(), lr=float(lr))

            losses: List[float] = []

            with torch.no_grad():
                out0 = wm.ForwardTrain(
                    vision, keys, click, mouse, reward, done,
                    sample=False,
                    alphaKl=0.8, freeNats=1.0,
                    reconCoef=1.0, rewardCoef=0.0, doneCoef=0.0,
                    nsCoef=0.0, nsDistillCoef=0.0, nsPriorLogicCoef=0.0,
                    physCoef=0.0,)
                init_loss = float(out0["loss"].item())

            for t in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                out = wm.ForwardTrain(
                    vision, keys, click, mouse, reward, done,
                    sample=False,
                    alphaKl=0.8, freeNats=1.0,
                    reconCoef=1.0, rewardCoef=0.0, doneCoef=0.0,
                    nsCoef=0.0, nsDistillCoef=0.0, nsPriorLogicCoef=0.0,
                    physCoef=0.0,)
                loss = out["loss"]
                if not torch.isfinite(loss).item():
                    print(f"LossDecrease FAILED: non-finite at step {t}, loss={loss}")
                    return False
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))

            head_n = min(5, len(losses))
            tail_n = min(5, len(losses))
            head = sum(losses[:head_n]) / max(1, head_n)
            tail = sum(losses[-tail_n:]) / max(1, tail_n)

            margin = max(1e-4, 0.02 * abs(head))
            ok = bool(tail <= head - margin)

            with torch.no_grad():
                out1 = wm.ForwardTrain(
                    vision, keys, click, mouse, reward, done,
                    sample=False,
                    alphaKl=0.8, freeNats=1.0,
                    reconCoef=1.0, rewardCoef=0.0, doneCoef=0.0,
                    nsCoef=0.0, nsDistillCoef=0.0, nsPriorLogicCoef=0.0,
                    physCoef=0.0,)
                final_loss = float(out1["loss"].item())

            print(
                f"LossDecrease {'passed' if ok else 'failed'} | "
                f"init={init_loss:.6f} -> final={final_loss:.6f}; "
                f"head={head:.6f} -> tail={tail:.6f}")
            return ok

        except Exception as e:
            print(f"LossDecrease FAILED: {type(e).__name__}: {e}")
            return False


    def TestConnRegReset(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.train()

            B = 3
            wm.ResetState(batchSize=B)

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(11, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)

            reward = torch.zeros(B, device=self.device)
            done = torch.zeros(B, device=self.device)

            _ = wm.ForwardTrain(vision, keys, click, mouse, reward, done, sample=False)
            has_prev = getattr(wm, "_A_prev", None) is not None

            wm.ResetState(batchSize=B)
            cleared = getattr(wm, "_A_prev", None) is None

            ok = bool(has_prev and cleared)
            print(f"Conn regularization cache reset {'passed' if ok else 'failed'}")
            return ok

        except Exception as e:
            print(f"Conn regularization cache reset FAILED: {type(e).__name__}: {e}")
            return False


    def TestExportWorldMemoryBank(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = RSSMWorldModel(
                visionDim=64,
                actionDim=32,
                deterDim=64,
                stochDim=16,
                stateDim=32,
                ssmDim=32,
                useDecoder=False,
                useMemory=True,
                memoryCapacity=32,
                nsEnabled=False,
                memTopK=4,
                memTemp=1.0,).to(self.device).eval()

            B = 2
            wm.ResetState(batchSize=B)

            out0 = wm.ExportWorldMemoryBank(topk=4)
            if out0 is not None:
                print("ExportWorldMemoryBank FAILED: expected None when not filled enough.")
                return False

            for _ in range(6):
                keyE = F.normalize(torch.randn(B, wm.stoch_dim, device=self.device), dim=-1)
                valH = torch.randn(B, wm.state_dim, device=self.device)
                imp = torch.rand(B, device=self.device)
                wm.MemAdd(keyE, valH, imp)

            out = wm.ExportWorldMemoryBank(topk=4)
            if out is None:
                print("ExportWorldMemoryBank FAILED: expected dict after filling.")
                return False

            ok_shapes = (
                ("size" in out) and ("idx" in out) and ("vals" in out) and ("keys" in out) and ("imp" in out) and ("steps" in out)
                and out["size"].shape == (B,)
                and out["idx"].shape == (B, 4)
                and out["vals"].shape == (B, 4, wm.state_dim)
                and out["keys"].shape == (B, 4, wm.stoch_dim)
                and out["imp"].shape == (B, 4)
                and out["steps"].shape == (B, 4))

            idx_ok = bool((out["idx"] >= 0).all().item() and (out["idx"] < wm._mem_capacity).all().item())
            imp_ok = bool(torch.isfinite(out["imp"]).all().item())
            step_ok = bool((out["steps"][:, :-1] >= out["steps"][:, 1:]).all().item())

            ok = bool(ok_shapes and idx_ok and imp_ok and step_ok)
            print(f"ExportWorldMemoryBank {'passed' if ok else 'failed'}")
            return ok

        except Exception as e:
            print(f"ExportWorldMemoryBank FAILED: {type(e).__name__}: {e}")
            return False

    def TestExportWorldMemoryBankLatestFirst(self) -> bool:
        try:
            wm = RSSMWorldModel(
                visionDim=32,
                actionDim=16,
                deterDim=32,
                stochDim=8,
                stateDim=12,
                ssmDim=16,
                useDecoder=False,
                useMemory=True,
                memoryCapacity=8,
                nsEnabled=False,
                memTopK=4,
                memTemp=1.0,).to(self.device).eval()

            B = 1
            wm.ResetState(batchSize=B)
            wm.ResetMemory()

            with torch.no_grad():
                wm._mem_size.fill_(3)
                wm._mem_imp.zero_()
                wm._mem_steps.zero_()
                wm._mem_vals.zero_()
                wm._mem_keys.zero_()
                wm._mem_global_step.fill_(30)

                wm._mem_imp[0, :3] = torch.tensor([0.9, 0.8, 0.7], device=self.device, dtype=wm.dtype)
                wm._mem_steps[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)

                wm._mem_vals[0, 0, 0] = 30.0
                wm._mem_vals[0, 1, 0] = 10.0
                wm._mem_vals[0, 2, 0] = 20.0

                wm._mem_keys[0, 0, 0] = 3.0
                wm._mem_keys[0, 1, 0] = 1.0
                wm._mem_keys[0, 2, 0] = 2.0

            out = wm.ExportWorldMemoryBank(topk=3)
            if out is None:
                print("ExportWorldMemoryBank latest-first FAILED: expected dict.")
                return False

            expected_steps = torch.tensor([30, 20, 10], device=self.device, dtype=torch.long)
            expected_vals = torch.tensor([30.0, 20.0, 10.0], device=self.device, dtype=wm.dtype)
            expected_keys = torch.tensor([3.0, 2.0, 1.0], device=self.device, dtype=wm.dtype)

            ok = bool(
                torch.equal(out["steps"][0], expected_steps)
                and torch.allclose(out["vals"][0, :, 0], expected_vals)
                and torch.allclose(out["keys"][0, :, 0], expected_keys))

            print(f"ExportWorldMemoryBank latest-first {'passed' if ok else 'failed'}")
            return ok
        except Exception as e:
            print(f"ExportWorldMemoryBank latest-first FAILED: {type(e).__name__}: {e}")
            return False

    def TestReorderMemorySteps(self) -> bool:
        try:
            wm = RSSMWorldModel(
                visionDim=32,
                actionDim=16,
                deterDim=32,
                stochDim=8,
                stateDim=12,
                ssmDim=16,
                useDecoder=False,
                useMemory=True,
                memoryCapacity=8,
                nsEnabled=False,
                memTopK=4,
                memTemp=1.0,).to(self.device).eval()

            B = 1
            wm.ResetState(batchSize=B)
            wm.ResetMemory()

            with torch.no_grad():
                wm._mem_size.fill_(3)
                wm._mem_imp.zero_()
                wm._mem_steps.zero_()
                wm._mem_vals.zero_()
                wm._mem_global_step.fill_(30)

                wm._mem_imp[0, :3] = torch.tensor([0.9, 0.8, 0.7], device=self.device, dtype=wm.dtype)
                wm._mem_steps[0, :3] = torch.tensor([30, 10, 20], device=self.device, dtype=torch.long)
                wm._mem_vals[0, 0, 0] = 30.0
                wm._mem_vals[0, 1, 0] = 10.0
                wm._mem_vals[0, 2, 0] = 20.0

            wm.ReorderMemorySteps()

            expect_steps = torch.tensor([3, 1, 2], device=self.device, dtype=torch.long)
            if not torch.equal(wm._mem_steps[0, :3], expect_steps):
                print(f"ReorderMemorySteps FAILED: got local steps {wm._mem_steps[0, :3].tolist()}")
                return False

            if int(wm._mem_global_step[0].item()) != 3:
                print(f"ReorderMemorySteps FAILED: global step={int(wm._mem_global_step[0].item())}, expected 3")
                return False

            out = wm.ExportWorldMemoryBank(topk=3)
            if out is None:
                print("ReorderMemorySteps FAILED: expected export dict after reset.")
                return False

            expect_export_steps = torch.tensor([3, 2, 1], device=self.device, dtype=torch.long)
            expect_export_vals = torch.tensor([30.0, 20.0, 10.0], device=self.device, dtype=wm.dtype)

            ok = bool(
                torch.equal(out["steps"][0], expect_export_steps)
                and torch.allclose(out["vals"][0, :, 0], expect_export_vals))

            print(f"ReorderMemorySteps {'passed' if ok else 'failed'}")
            return ok
        except Exception as e:
            print(f"ReorderMemorySteps FAILED: {type(e).__name__}: {e}")
            return False


    def TestWrapperAPIBasics(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.eval()

            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()

            B = 4
            wm.ResetState(batchSize=B)

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(6, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)
            reward = torch.zeros(B, device=self.device)
            done = torch.zeros(B, device=self.device)

            out = wrapper(
                vision,
                keysVec=keys,
                mouseClick=click,
                mouseSeq=mouse,
                reward=reward,
                done=done,
                sample=False,)

            must = ["loss", "h_next", "z_next", "s_next", "r_pred", "d_prob"]
            ok = all(k in out for k in must)

            ok = ok and (out["h_next"].shape == (B, wm.deter_dim))
            ok = ok and (out["z_next"].shape == (B, wm.stoch_dim))
            ok = ok and (out["s_next"].shape == (B, wm.state_dim))
            ok = ok and (out["r_pred"].shape == (B,))
            ok = ok and (out["d_prob"].shape == (B,))

            dmin = float(out["d_prob"].min().item())
            dmax = float(out["d_prob"].max().item())
            ok = ok and (0.0 <= dmin <= dmax <= 1.0)

            print(f"Wrapper API basics {'passed' if ok else 'failed'} | d_prob=[{dmin:.3f},{dmax:.3f}]")
            return bool(ok)

        except Exception as e:
            print(f"Wrapper API basics FAILED: {type(e).__name__}: {e}")
            return False


    def TestForwardWithDeltasInjection(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.eval()

            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()

            B = 3
            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(10, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)
            reward = torch.zeros(B, device=self.device)
            done = torch.zeros(B, device=self.device)

            site = "act_proj"
            Wshape = wm.act_proj[0].target.weight.shape  
            deltaW = torch.randn(*Wshape, device=self.device) * 1e-3

            torch.manual_seed(123)
            wm.ResetState(batchSize=B)
            out0 = wrapper.ForwardWithDeltas(
                vision, None, None, None, [{}],
                keysVec=keys, mouseClick=click, mouseSeq=mouse, reward=reward, done=done, sample=False,)

            torch.manual_seed(123)
            wm.ResetState(batchSize=B)
            out1 = wrapper.ForwardWithDeltas(
                vision, None, None, None, [{site: deltaW}],
                keysVec=keys, mouseClick=click, mouseSeq=mouse, reward=reward, done=done, sample=False,)

            diff = float((out0["s_next"] - out1["s_next"]).abs().mean().item())
            ok = diff > 1e-7
            print(f"ForwardWithDeltas injection {'passed' if ok else 'failed'} | site='{site}', |Δ|={diff:.3e}")
            return bool(ok)

        except Exception as e:
            print(f"ForwardWithDeltas injection FAILED: {type(e).__name__}: {e}")
            return False


    def TestCommitOneGrowAndValueChange(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.eval()

            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).train()

            lo = wm.act_proj[0]
            n0 = len(lo.A_list)

            out_f, in_f = lo.target.weight.shape 
            r = 2

            A = torch.randn(r, in_f, device=self.device) * 0.10
            Bm = torch.randn(out_f, r, device=self.device) * 0.10
            ok_commit = wrapper.CommitOne("act_proj", 0, A, Bm, 10.0)

            n1 = len(lo.A_list)
            grew = bool(ok_commit and (n1 == n0 + 1))

            Bsz = 4
            vision = torch.randn(Bsz, wm.vision_dim, device=self.device)
            keys = torch.zeros(Bsz, self.key_dim, device=self.device)
            keys[:, min(8, self.key_dim - 1)] = 1.0
            mouse = torch.randn(Bsz, 2, device=self.device)
            click = torch.zeros(Bsz, 2, device=self.device)
            reward = torch.zeros(Bsz, device=self.device)
            done = torch.zeros(Bsz, device=self.device)

            wrapper.eval()
            with torch.no_grad():
                last_s = lo.alpha[-1].clone()

                lo.alpha[-1].zero_()
                torch.manual_seed(456)
                wm.ResetState(batchSize=Bsz)
                out_before = wrapper(
                    vision,
                    keysVec=keys, mouseClick=click, mouseSeq=mouse, reward=reward, done=done,
                    sample=False,)

                lo.alpha[-1].copy_(last_s)
                torch.manual_seed(456)
                wm.ResetState(batchSize=Bsz)
                out_after = wrapper(
                    vision,
                    keysVec=keys, mouseClick=click, mouseSeq=mouse, reward=reward, done=done,
                    sample=False,)

            change = float((out_after["s_next"] - out_before["s_next"]).abs().mean().item())
            ok = bool(grew and (change > 1e-7))
            print(f"CommitOne grow & effect {'passed' if ok else 'failed'} | rank {n0}->{n1}, |Δ|={change:.3e}")
            return ok

        except Exception as e:
            print(f"CommitOne grow & effect FAILED: {type(e).__name__}: {e}")
            return False


    def TestGradFlowCandidates(self) -> bool:
        try:
            torch.manual_seed(0)
            wm = self.wm
            wm.eval()

            wrapper = WorldOnlineWrapper(wm, initRankEach=2, autoRank=False).to(self.device).train()

            B = 5
            wm.ResetState(batchSize=B)

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            keys = torch.zeros(B, self.key_dim, device=self.device)
            keys[:, min(14, self.key_dim - 1)] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            click = torch.zeros(B, 2, device=self.device)
            reward = torch.zeros(B, device=self.device)
            done = torch.zeros(B, device=self.device)

            out = wrapper(
                vision,
                keysVec=keys, mouseClick=click, mouseSeq=mouse, reward=reward, done=done,
                sample=False,)

            loss = out["loss"]
            if not torch.isfinite(loss).item():
                print("GradFlowCandidates FAILED: loss not finite")
                return False

            wrapper.zero_grad(set_to_none=True)
            loss.backward()

            params = list(wrapper.CandParameters())
            if len(params) == 0:
                print("GradFlowCandidates FAILED: no candidate params")
                return False

            any_grad = False
            for p in params:
                if p.grad is not None and torch.isfinite(p.grad).all().item() and (p.grad.abs().sum().item() > 0):
                    any_grad = True
                    break

            ok = bool(any_grad)
            print(f"Grad flow (wrapper candidates) {'passed' if ok else 'failed'} | loss={float(loss.item()):.6f}")
            return ok

        except Exception as e:
            print(f"Grad flow (wrapper candidates) FAILED: {type(e).__name__}: {e}")
            return False

    def TestWrapperUpdateInjectLoRA(self) -> bool:
        try:
            torch.manual_seed(0)

            wm = self.wm.to(self.device).eval()
            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()
            base = wrapper.base

            B = 4
            V = int(base.vision_dim)
            K = int(getattr(base.action_encoder, "K", 64))

            visionIn = torch.randn(B, V, device=self.device, dtype=base.dtype)
            keysVec = torch.zeros(B, K, device=self.device, dtype=base.dtype)
            keysVec[:, min(3, K - 1)] = 1.0

            mouseSeq = torch.zeros(B, 2, device=self.device, dtype=base.dtype)
            mouseClick = torch.zeros(B, 2, device=self.device, dtype=base.dtype)

            reward = torch.zeros(B, device=self.device, dtype=base.dtype)
            done = torch.zeros(B, device=self.device, dtype=base.dtype)

            base_params0 = {n: p.detach().clone() for n, p in base.named_parameters()}

            use_mem0 = bool(getattr(base, "_use_memory", False))
            if hasattr(base, "_use_memory"):
                base._use_memory = False

            base.EnsureB(B, base.device, base.dtype)

            h0 = base._h.detach().clone()
            z0 = base._z.detach().clone()
            s4x0 = base.s4.x.detach().clone()
            Aprev0 = None
            if hasattr(base, "_A_prev") and (base._A_prev is not None):
                Aprev0 = base._A_prev.detach().clone()

            def _restore_state():
                base._h = h0.detach().clone()
                base._z = z0.detach().clone()
                base.s4.x = s4x0.detach().clone()
                if hasattr(base, "_A_prev"):
                    base._A_prev = None if Aprev0 is None else Aprev0.detach().clone()

            wrapper.Update("reset", initRankEach=0)
            _restore_state()

            with torch.no_grad():
                out0 = wrapper(
                    visionIn,
                    keysVec=keysVec,
                    mouseClick=mouseClick,
                    mouseSeq=mouseSeq,
                    reward=reward,
                    done=done,
                    sample=False,)
            dprob0 = out0["d_prob"].detach().float().mean().item()

            wrapper.Update("reset", initRankEach=0)
            wrapper.Update("grow", growFactor=1.0, addEach=1)   

            assert "act_proj" in wrapper.sites, "Site 'act_proj' not found in wrapper.sites"
            assert len(wrapper.cand["act_proj"][0]["A"]) > 0, "act_proj layer0 has no candidate after grow()"

            with torch.no_grad():
                A = wrapper.cand["act_proj"][0]["A"][0]
                Bm = wrapper.cand["act_proj"][0]["B"][0]
                s = wrapper.cand["act_proj"][0]["s"][0]

                A.normal_(0.0, 5e-2)
                Bm.normal_(0.0, 5e-2)
                s.fill_(3.0)  

            dmat = wrapper.ComposeOne("act_proj", 0)
            dnorm = float(dmat.detach().float().abs().mean().item())
            assert dnorm > 0.0, "Injected delta is still zero; injection failed"

            _restore_state()
            with torch.no_grad():
                out1 = wrapper(
                    visionIn,
                    keysVec=keysVec,
                    mouseClick=mouseClick,
                    mouseSeq=mouseSeq,
                    reward=reward,
                    done=done,
                    sample=False,)
            dprob1 = out1["d_prob"].detach().float().mean().item()

            same_base = True
            for n, p in base.named_parameters():
                if not torch.equal(base_params0[n], p.detach()):
                    same_base = False
                    break

            diff = abs(dprob1 - dprob0)
            ok = (diff > 1e-9) and same_base

            print(
                f"Wrapper Update-inject LoRA {'passed' if ok else 'FAILED'} | "
                f"site='act_proj' | mean|Δ|={dnorm:.3e} | "
                f"d_prob_mean: {dprob0:.6f} -> {dprob1:.6f} |Δ|={diff:.3e} | "
                f"base_unchanged={same_base}")

            if hasattr(base, "_use_memory"):
                base._use_memory = use_mem0

            return ok

        except Exception as e:
            try:
                if "base" in locals() and hasattr(base, "_use_memory") and "use_mem0" in locals():
                    base._use_memory = use_mem0
            except Exception:
                pass

            print(f"Wrapper Update-inject LoRA FAILED: {type(e).__name__}: {e}")
            return False



    def RunAll(self) -> bool:
        results = {
            "ActionEncoder": self.TestActionEncoder(),
            "RSSMStepPosterior": self.TestRSSMStepPosterior(),
            "RSSMStepPriorOnly": self.TestRSSMStepPriorOnly(),
            "ForwardTrainFiniteGrad": self.TestForwardTrainFiniteGrad(),
            "WorldForwardIOShapes": self.TestWorldForwardIOShapes(),
            "LossDecrease": self.TestLossDecrease(),
            "ConnRegReset": self.TestConnRegReset(),
            "ExportWorldMemoryBank": self.TestExportWorldMemoryBank(),
            "ExportWorldMemoryBankLatestFirst": self.TestExportWorldMemoryBankLatestFirst(),
            "ReorderMemorySteps": self.TestReorderMemorySteps(),
            "WrapperAPIBasics": self.TestWrapperAPIBasics(),
            "ForwardWithDeltasInjection": self.TestForwardWithDeltasInjection(),
            "CommitOneGrowAndValueChange": self.TestCommitOneGrowAndValueChange(),
            "GradFlowCandidates": self.TestGradFlowCandidates(),
            "WrapperUpdateInjectLoRA": self.TestWrapperUpdateInjectLoRA(),}

        passed = sum(1 for v in results.values() if v)
        print(f"\n[WorldModule Tests] {passed}/{len(results)} passed.")
        return passed == len(results)
