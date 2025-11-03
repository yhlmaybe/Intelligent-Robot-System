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
    def __init__(self, numDiscrete=106, contDim=2, outDim=256, hidden=512):
        super().__init__()

        self.disc_net = nn.Sequential(
            nn.LayerNorm(numDiscrete),
            nn.Linear(numDiscrete, hidden), 
            nn.GELU(),
            nn.Linear(hidden, outDim),)

        self.cont_net = nn.Sequential(
            nn.LayerNorm(contDim),
            nn.Linear(contDim, 128), 
            nn.SiLU(),
            nn.Linear(128, outDim),)
        
        self.to_gamma = nn.Linear(outDim, outDim)

        self.to_beta  = nn.Linear(outDim, outDim)
        
        self.fuse = nn.Sequential(
            nn.LayerNorm(outDim * 2),
            nn.Linear(outDim * 2, outDim * 2), 
            nn.GELU(),
            nn.Linear(outDim * 2, outDim),)

    def forward(self, keysOnehot, mouseDelta=None):
        d = self.disc_net(keysOnehot.float())
        if mouseDelta is None:
            return d
        c = self.cont_net(mouseDelta.float())
        d = (1.0 + torch.tanh(self.to_gamma(c))) * d + self.to_beta(c)
        return self.fuse(torch.cat([d, c], dim=-1))


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
        B = init.get("B", torch.randn(self.out_f, addRank, device=dev, dtype=dt) * 1e-4)
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


class S4DCell(nn.Module):
    def __init__(self, inDim: int, deterDim: int, ssmDim: int = 512, dt: float = 1.0, dropout: float = 0.0, ffnMult: int = 4):
        super().__init__()
        self.U = int(inDim)
        self.D = int(deterDim)
        self.N = int(ssmDim)
        self.dt = float(dt)

        self.theta = nn.Parameter(torch.randn(self.N) * 0.1)

        self.B = GrowableLoRALinear(nn.Linear(self.U, self.N, bias=True))
        self.C = GrowableLoRALinear(nn.Linear(self.N, self.D, bias=True))
        self.D0 = GrowableLoRALinear(nn.Linear(self.U, self.D, bias=True))
        self.gate = GrowableLoRALinear(nn.Linear(self.U, self.N, bias=True))
        self.out_gate = GrowableLoRALinear(nn.Linear(self.N, self.D, bias=True))

        self.ln_y = nn.LayerNorm(self.D)
        self.ln_ffn = nn.LayerNorm(self.D)
        self.ffn = nn.Sequential(
            nn.Linear(self.D, ffnMult * self.D, bias=True),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffnMult * self.D, self.D, bias=True),)

        self.register_buffer("x", torch.zeros(1, self.N), persistent=False)

    @staticmethod
    def CayleyStep(aDiag: torch.Tensor, x: torch.Tensor, Bu: torch.Tensor, dt: float):
        A = -F.softplus(aDiag)
        k = 0.5 * dt * A
        num = (1 + k) * x + dt * Bu
        denom = (1 - k).clamp_min(1e-6)
        return num / denom

    def ResetState(self, batch: int):
        self.x = torch.zeros(batch, self.N)

    def Step(self, zPrev: torch.Tensor, aT: torch.Tensor, *, updateState: bool = True) -> torch.Tensor:
        u = torch.cat([zPrev, aT], dim=-1)
        g = torch.sigmoid(self.gate(u))
        Bu = self.B(u) * g

        B = u.size(0)
        if (self.x.dim() != 2
            or self.x.size(0) != B
            or self.x.size(1) != self.N
            or self.x.device != u.device
            or self.x.dtype != u.dtype):
            self.x = torch.zeros(B, self.N, device=u.device, dtype=u.dtype)
        x_use = self.x 

        x_next = self.CayleyStep(self.theta, x_use, Bu, self.dt)
        y_lin = self.C(x_next) + self.D0(u)
        y_glu = y_lin * torch.sigmoid(self.out_gate(x_next))
        y = self.ln_y(y_glu)
        y = y + self.ffn(self.ln_ffn(y))

        if updateState:
            self.x = x_next.detach()
        return y
    
class HNNPhysHead(nn.Module):
    def __init__(self, deterDim: int, projDim: int = 256):
        super().__init__()
        self.D = deterDim
        self.P = projDim
        self.to_qp = GrowableLoRALinear(nn.Linear(self.D, self.P, bias=True))

        self.H = nn.Sequential(
            GrowableLoRALinear(nn.Linear(self.P, self.P)),
            nn.GELU(),
            GrowableLoRALinear(nn.Linear(self.P, 1)))
        
        self.from_qp = GrowableLoRALinear(nn.Linear(self.P, self.D, bias=True))
        self.dt = 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qp = self.to_qp(x)
        q, p = qp.chunk(2, dim=-1)

        with torch.enable_grad():
            qp_in = torch.cat([q, p], dim=-1).detach().requires_grad_(True)
            H_val = self.H(qp_in)
            grad = torch.autograd.grad(H_val.sum(), qp_in, create_graph=self.training, retain_graph=True)[0]
            dH_dq, dH_dp = grad.chunk(2, dim=-1)

            p_half = p - 0.5 * self.dt * dH_dq

            qp_mid = torch.cat([q, p_half], dim=-1).detach().requires_grad_(True)
            H_mid = self.H(qp_mid)
            grad_mid = torch.autograd.grad(H_mid.sum(), qp_mid, create_graph=self.training, retain_graph=True)[0]
            dH_dq_mid, dH_dp_mid = grad_mid.chunk(2, dim=-1)

            q_new = q + self.dt * dH_dp_mid

            qp_new = torch.cat([q_new, p_half], dim=-1).detach().requires_grad_(True)
            H_new = self.H(qp_new)
            grad2 = torch.autograd.grad(H_new.sum(), qp_new, create_graph=self.training, retain_graph=True)[0]
            dH_dq2, dH_dp2 = grad2.chunk(2, dim=-1)

            p_new = p_half - 0.5 * self.dt * dH_dq2

        h_phys = self.from_qp(torch.cat([q_new, p_new], dim=-1))
        e_cons = (H_val.detach() - self.H(torch.cat([q_new, p_new], dim=-1))).pow(2).mean()
        return h_phys, e_cons


class ODEPhysHead(nn.Module):
    def __init__(self, deterDim: int, actDim: int, hidden: int = 256):
        super().__init__()
        self.f = nn.Sequential(
            GrowableLoRALinear(nn.Linear(deterDim + actDim, hidden)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, deterDim)))
        
        self.dt = 1.0

    def forward(self, h: torch.Tensor, aT: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a_t = aT
        inp = torch.cat([h, a_t], dim=-1)
        k1 = self.f(inp)
        mid = h + 0.5*self.dt*k1
        k2 = self.f(torch.cat([mid, a_t], dim=-1))
        h_ode = h + self.dt * k2
        smooth = (k2 - k1).pow(2).mean()
        return h_ode, smooth

class NeSyHead(nn.Module):
    def __init__(self, inDim: int, K: int, hidden: int = 1024, experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(inDim, experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, K)
            ) for _ in range(experts)])

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = F.softmax(self.gate(x), dim=-1) 
        ys = [e(x) for e in self.experts]
        y = torch.stack(ys, dim=-1)  
        return (y * a.unsqueeze(1)).sum(dim=-1)



class GeometricLinear(nn.Module):
    def __init__(self, inFeatures, outFeatures, wrapLinear=None, gain=0.1):
        super().__init__()
        lin = nn.Linear(inFeatures, outFeatures, bias=True)
        nn.init.orthogonal_(lin.weight, gain=gain)
        nn.init.zeros_(lin.bias)
        self.linear = wrapLinear(lin) if wrapLinear is not None else lin

    def forward(self, x): return self.linear(x)

class FilmResidual(nn.Module):
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

class ConnNet(nn.Module):
    def __init__(self,
                 stateDim: int,
                 actDim: int,
                 *,
                 hidden: int = 512,
                 numBlocks: int = 3,
                 rank: int = 8,
                 useFull: bool = True,
                 useLowrank: bool = True,
                 transport: str = "auto",
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
        self.transport = str(transport).lower()
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

        if self.use_lowrank:
            self.head_uv = GeometricLinear(self.H, 2 * self.S * self.r, wrapLinear)
        if self.use_full:
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

        euler = sBase + dt * torch.einsum("bij,bj->bi", A, sBase)

        I = torch.eye(S, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, S, S)
        lhs = I - 0.5 * dt * A 
        rhs_vec = torch.einsum("bij,bj->bi", I + 0.5 * dt * A, sBase) 
        cayley = torch.linalg.solve(lhs, rhs_vec.unsqueeze(-1)).squeeze(-1)  

        mode = self.transport
        if mode == "euler":
            return euler
        if mode == "cayley":
            return cayley

        fro = A.pow(2).mean(dim=(1, 2)).sqrt() 
        mask = (fro > 0.75).to(A.dtype).view(B, 1)  
        return mask * cayley + (1.0 - mask) * euler

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



class SoftNeSyStructure(nn.Module):
    def __init__(self, k: int, gExcl: int = 8, gAlo: int = 8, tauInit: float = 1.0):
        super().__init__()
        self.K = int(k)
        self.Ge = int(gExcl)
        self.Ga = int(gAlo)
        self.tau = nn.Parameter(torch.tensor(float(tauInit)))
        self.M_excl = nn.Parameter(torch.randn(self.Ge, self.K) * 0.01)
        self.M_alo  = nn.Parameter(torch.randn(self.Ga, self.K) * 0.01)
        E = torch.zeros(self.K, self.K)
        E.fill_(0.0)
        self.E = nn.Parameter(E)
        self.register_buffer("_eye", torch.eye(self.K))

    def MixExclusive(self, P: torch.Tensor, temp: float) -> torch.Tensor:
        B, K = P.shape
        eps = 1e-6
        Wg = F.softmax(self.M_excl, dim=-1) 
        logP = torch.log(P.clamp(eps, 1 - eps)) / max(1e-6, temp)
        g = logP.unsqueeze(1) + torch.log(Wg.unsqueeze(0).clamp(eps))
        g_sm = F.softmax(g, dim=-1)
        Wk = F.softmax(self.M_excl.t(), dim=-1)
        P_new = torch.einsum("bgk,kg->bk", g_sm, Wk)
        return P_new.clamp(eps, 1 - eps)

    def EnforceAlo(self, P: torch.Tensor, tau: float) -> torch.Tensor:
        B, K = P.shape
        eps = 1e-6
        Wa = F.softmax(self.M_alo, dim=-1)
        group_vals = (P.unsqueeze(1) * Wa.unsqueeze(0)).max(dim=-1).values
        scale = torch.where(group_vals < tau, tau / (group_vals + eps), torch.ones_like(group_vals))
        P_scaled = P.clone()
        Wk = F.softmax(self.M_alo.t(), dim=-1)
        s = torch.einsum("bg,kg->bk", scale, Wk)
        P_scaled = (P_scaled * s).clamp(eps, 1 - eps)
        return P_scaled

    def ApplyImplications(self, P: torch.Tensor, alpha: float) -> torch.Tensor:
        B, K = P.shape
        eps = 1e-6
        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        contrib = P.unsqueeze(2) * W.unsqueeze(0)
        implied = contrib.max(dim=1).values
        Q = torch.maximum(P, alpha * implied).clamp(eps, 1 - eps)
        return Q

    def ProjectTrain(self, logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        P0 = torch.sigmoid(logits / max(1e-6, temp))
        P1 = self.MixExclusive(P0, temp)
        return P1

    @torch.no_grad()
    def ProjectRuntime(self, P: torch.Tensor, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        Q = self.MixExclusive(P, temp)
        Q = self.EnforceAlo(Q, aloTau)
        Q = self.ApplyImplications(Q, implAlpha)

        Ge = F.softmax(self.M_excl, dim=-1)
        gprob = F.softmax((torch.log(Q.clamp(1e-6, 1-1e-6))).unsqueeze(1) + torch.log(Ge.unsqueeze(0).clamp(1e-6)), dim=-1)
        excl_pen = 0.5 * ((gprob.sum(-1) ** 2) - (gprob ** 2).sum(-1)).mean(dim=-1)

        Ga = F.softmax(self.M_alo, dim=-1)
        alo_val = (Q.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values
        alo_pen = F.relu(aloTau - alo_val).mean(dim=-1)

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl_pen = (W.unsqueeze(0) * F.relu(Q.unsqueeze(2) - Q.unsqueeze(1))).mean(dim=(1,2))

        pen = excl_pen + alo_pen + impl_pen
        pen = (pen / (pen.detach().quantile(0.9) + 1e-6)).clamp(0.0, 1.0)
        return Q, pen

    def LogicLosses(self, P: torch.Tensor, lambdaExcl: float, lambdaAlo: float, lambdaImpl: float, aloTau: float = 0.60):
        Ge = F.softmax(self.M_excl, dim=-1)
        g = (torch.log(P.clamp(1e-6, 1-1e-6))).unsqueeze(1) + torch.log(Ge.unsqueeze(0).clamp(1e-6))
        g_sm = F.softmax(g, dim=-1)
        excl = 0.5 * ((g_sm.sum(-1)**2) - (g_sm**2).sum(-1))
        excl = excl.mean()

        Ga = F.softmax(self.M_alo, dim=-1)
        top1 = (P.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values

        if aloTau is None:
            aloTau_t = self.tau 
        else:
            aloTau_t = top1.new_tensor(float(aloTau))

        alo = (F.relu(aloTau_t - top1) ** 2).mean()

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl = (W.unsqueeze(0) * F.relu(P.unsqueeze(2) - P.unsqueeze(1))).mean()

        loss = lambdaExcl * excl + lambdaAlo * alo + lambdaImpl * impl

        reg = 1e-4 * W.mean() + 1e-3 * (
            (F.softmax(self.M_excl, dim=-1)*torch.log(F.softmax(self.M_excl, dim=-1)+1e-6)).sum()/self.Ge +
            (F.softmax(self.M_alo , dim=-1)*torch.log(F.softmax(self.M_alo , dim=-1)+1e-6)).sum()/self.Ga)
        
        reg = reg + 1e-6 * (torch.trace(torch.matrix_exp(W * W)) - self.K)

        loss = loss + reg
        stats = {"excl": excl.detach(), "alo": alo.detach(), "impl": impl.detach()}
        return loss, stats


class RSSMWorldModel(nn.Module):
    def __init__(
        self,
        visionDim: int = 1024,
        batchSize: int = 1,
        actionDim: int = 256,
        deterDim: int = 512,
        stochDim: int = 64,
        stateDim: int = 512,
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

        self._A_prev = None

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
        
        self.s4 = S4DCell(inDim=stochDim + stochDim, deterDim=deterDim, ssmDim=512, dt=1.0)

        self.prior_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim, 2 * stochDim, bias=True)))
        
        self.post_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim + stochDim, 2 * stochDim, bias=True)))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + stochDim),
            GrowableLoRALinear(nn.Linear(deterDim + stochDim, stateDim, bias=True)),
            nn.LayerNorm(stateDim),)

        self.rew_head = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, 256, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)
        
        self.done_head = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, 256, bias=True)),
            nn.SiLU(),
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
 
        self._ns_K: int = 60
        self.ns_struct = SoftNeSyStructure(k=self._ns_K, gExcl=8, gAlo=8, tauInit=1.0)

        self.ns_head_prior = NeSyHead(deterDim, self._ns_K, hidden=1024, experts=4)
        self.ns_head_post = NeSyHead(deterDim + stochDim, self._ns_K, hidden=1024, experts=4)

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

        self.ResetHidden(batchSize=batchSize)

        if self._use_memory and self._mem_path:
            self.LoadMemory(self._mem_path, mapLocation=None, strict=False)

        self.mem_val_to_e = nn.Sequential(nn.Linear(deterDim, stochDim), nn.LayerNorm(stochDim))


        self.conn = ConnNet(stateDim=stateDim,actDim=stochDim,wrapLinear=GrowableLoRALinear)

        self.phys_hnn = HNNPhysHead(deterDim=self.deter_dim, projDim=128)
        self.phys_ode = ODEPhysHead(deterDim=self.deter_dim, actDim=self.stoch_dim)
        self.mix_gate = nn.Sequential(GrowableLoRALinear(nn.Linear(self.deter_dim + 2*self.stoch_dim, 4)))

    @torch.no_grad()
    def InitWorldMemoryDocument(self, path: str):
        dir_ = os.path.dirname(path)
        if dir_ and (not os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)

        dev = self._mem_keys.device

        payload = {
            "mem_keys": torch.zeros_like(self._mem_keys, device=dev),
            "mem_vals": torch.zeros_like(self._mem_vals, device=dev),
            "mem_size": 0,
            "mem_ptr": 0,

            "h": torch.zeros(1, self.deter_dim, device=dev),
            "z": torch.zeros(1, self.stoch_dim, device=dev),

            "s4_x": torch.zeros(1, self.s4.N, device=dev),

            "_A_prev": None,

            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,}

        torch.save(payload, path)


    def SetMemoryOption(self, useMem: bool, path: str):
        self._use_memory = useMem
        if useMem:
            self._mem_path = path
            self.LoadMemory(path, mapLocation=None, strict=False)
        else:
            self._mem_path = None

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

    def LoadMemory(self, path: str, mapLocation: Optional[str] = None, strict: bool = False):
        if not self._use_memory:
            return
        if not os.path.exists(path):
            if strict:
                raise FileNotFoundError(path)
            return
        if os.path.getsize(path) == 0:
            return
        payload = torch.load(path, map_location=mapLocation, weights_only=False)
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

    def ResetHidden(self, batchSize: int = 1):
        self._h = torch.zeros(batchSize, self.deter_dim)
        self._z = torch.zeros(batchSize, self.stoch_dim)
        self.s4.ResetState(batchSize)
        self._A_prev = None

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._h, self._z

    def ImportState(self, h: torch.Tensor, z: torch.Tensor):
        self._h = h.detach().clone()
        self._z = z.detach().clone()

    def NsProjectProbs(self, logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        return self.ns_struct.ProjectTrain(logits, temp=temp)

    @torch.no_grad()
    def NsProjectRuntime(self, P: torch.Tensor, *, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        return self.ns_struct.ProjectRuntime(P, aloTau=aloTau, implAlpha=implAlpha, temp=temp)

    def NsConfidence(self, P: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        H = -( P.clamp(eps, 1 - eps) * torch.log(P.clamp(eps, 1 - eps)) + (1 - P).clamp(eps, 1 - eps) * torch.log((1 - P).clamp(eps, 1 - eps)))
        H = H.mean(dim=-1, keepdim=True)
        Hmax = torch.tensor(0.6931, device=P.device)
        conf = (1.0 - H / Hmax).clamp(0.0, 1.0)
        return conf

    def NsLogicLosses(self, probs: torch.Tensor):
        loss, stats = self.ns_struct.LogicLosses(
            probs,
            lambdaExcl=self.ns_lambda_excl,
            lambdaAlo=self.ns_lambda_alo,
            lambdaImpl=self.ns_lambda_impl,
            aloTau=None,)
        
        return loss, stats

    @torch.no_grad()
    def StepPriorOnly(
        self,
        hPrev: torch.Tensor, # deterministic state
        zPrev: torch.Tensor, # stochastic state
        actionEnc: torch.Tensor,
        sample: bool = False,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        a_t = self.act_proj(actionEnc)
        h_next = self.s4.Step(zPrev, a_t, updateState=False)

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

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1))
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1))
        A_t = self.conn(s_prev_base, a_t)
        s_transport = self.conn.TransportApply(A_t, s_prev_base)
        h_phys, _ = self.phys_hnn(h_next)
        h_ode, _ = self.phys_ode(h_next, a_t)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))
        s_ode = self.state_proj(torch.cat([h_ode, z_next], dim=-1))
        logits = self.mix_gate(torch.cat([h_next, z_next, a_t], dim=-1))
        w = F.softmax(logits, dim=-1)
        s_next = (w[:,0:1]*s_base + w[:,1:2]*s_transport + w[:,2:3]*s_phys + w[:,3:4]*s_ode)

        r_pred = self.rew_head(s_next).squeeze(-1)
        d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)
        return h_next, z_next, s_next, r_pred, d_prob # s_next is world state

    def StepPosterior(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        visionIn: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False, # False: Deterministic Forward, True: Reparameterized sampling with noise(More exploratory)
        ) -> Dict[str, torch.Tensor]:

        raw_e = self.obs_enc(visionIn)

        a_t = self.act_proj(actionEnc)
        h_next = self.s4.Step(zPrev, a_t, updateState=True)

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

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1))
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1))
        A_t = self.conn(s_prev_base, a_t)
        s_transport = self.conn.TransportApply(A_t, s_prev_base)
        h_phys, _ = self.phys_hnn(h_next)
        h_ode, _ = self.phys_ode(h_next, a_t)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))
        s_ode = self.state_proj(torch.cat([h_ode, z_next], dim=-1))
        logits = self.mix_gate(torch.cat([h_next, z_next, a_t], dim=-1))
        w = F.softmax(logits, dim=-1)
        s_next = (w[:,0:1]*s_base + w[:,1:2]*s_transport + w[:,2:3]*s_phys + w[:,3:4]*s_ode)

        r_pred = self.rew_head(s_next).squeeze(-1)
        d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)

        out = {
            "h_next": h_next,
            "z_next": z_next,
            "s_next": s_next,
            "r_pred": r_pred, # Prediction Rewards
            "d_prob": d_prob, # Predicted termination probability
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
        visionSeq: torch.Tensor,
        keysVec: torch.Tensor,
        mouseSeq: torch.Tensor,  
        h0: Optional[torch.Tensor] = None,
        z0: Optional[torch.Tensor] = None,
        rewardSeq: Optional[torch.Tensor] = None, # [B]
        doneSeq: Optional[torch.Tensor] = None, # [B]
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
        h_next = self.s4.Step(z0, a_enc, updateState=False)

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
            logits_pr = self.ns_head_prior(h_next) 
            P_pr_raw = torch.sigmoid(logits_pr)
            P_pr_train = self.NsProjectProbs(logits_pr)

            de_mu = self.ns_to_delta_mu(P_pr_train)
            base_gate_mu = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            with torch.no_grad():
                _, pen_pr = self.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf_pr = self.NsConfidence(P_pr_raw)
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

        s_base = self.state_proj(torch.cat([h_next, z1], dim=-1))
        s_prev_base = self.state_proj(torch.cat([h0, z0], dim=-1))

        A_t = self.conn(s_prev_base, a_enc)
        s_transport = self.conn.TransportApply(A_t, s_prev_base)

        prevA = (self._A_prev if (self._A_prev is not None  and self._A_prev.shape == A_t.shape) else None)
        reg_A = self.conn.ComputeGeomReg(A_t, prevA)
        self._A_prev = A_t.detach()
        
        h_phys, e_cons = self.phys_hnn(h_next)
        h_ode, e_smooth = self.phys_ode(h_next, a_enc)
        s_phys = self.state_proj(torch.cat([h_phys, z1], dim=-1))
        s_ode = self.state_proj(torch.cat([h_ode,  z1], dim=-1))
        logits = self.mix_gate(torch.cat([h_next, z1, a_enc], dim=-1))
        w = F.softmax(logits, dim=-1)
        s1 = (w[:,0:1]*s_base + w[:,1:2]*s_transport + w[:,2:3]*s_phys + w[:,3:4]*s_ode)

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
            + nsPriorLogicCoef * ns_prior_logic
            + 1e-5 * e_smooth
            + 1e-4 * e_cons
            + reg_A)

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
            "s_next": s1,
            "h_next": h_next,
            "z_next": z1,
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
        maxRankPrior: int = 64,
        maxRankPost: int = 64,
        maxRankState: int = 64,
        maxRankRew1: int = 32,
        maxRankRew2: int = 8,
        maxRankDone1: int = 32,
        maxRankDone2: int = 8,
        maxRankDec1: int = 64,
        maxRankDec2: int = 64,
        maxRankGruIH: int = 64,
        maxRankGruHH: int = 64,):

        self._maxRank = dict(
            obs1=int(maxRankObs1),
            obs2=int(maxRankObs2),
            act=int(maxRankAct),
            prior=int(maxRankPrior),
            post=int(maxRankPost),
            state=int(maxRankState),
            rew1=int(maxRankRew1),
            rew2=int(maxRankRew2),
            done1=int(maxRankDone1),
            done2=int(maxRankDone2),
            dec1=int(maxRankDec1),
            dec2=int(maxRankDec2),
            s4_B=int(maxRankGruIH),
            s4_C=int(maxRankGruHH),
            s4_D0=int(maxRankAct),
            s4_gate=int(maxRankAct),
            s4_out_gate=int(maxRankAct),
            hnn_to=32,
            hnn_H1=32,
            hnn_H2=8,
            hnn_from=32,
            ode_f1=32,
            ode_f2=32,
            mix=8,
            conn_enc_s=64,
            conn_enc_a=64,
            conn_gamma=64,
            conn_beta=64,
            conn_blk_ff1=64,
            conn_blk_ff2=64,
            conn_head_uv=64,
            conn_head_full=64,
            conn_mix=8,)
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
            return s * (b @ a)

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
                composeFn=compose_linear,),)

        add("obs1", V, S, self._maxRank["obs1"])
        add("obs2", S, Z, self._maxRank["obs2"])
        add("act", wm.action_dim, Z, self._maxRank["act"])
        add("prior", D, 2 * Z, self._maxRank["prior"])
        add("post", D + Z, 2 * Z, self._maxRank["post"])
        add("state", D + Z, S, self._maxRank["state"])
        add("rew1", S, 256, self._maxRank["rew1"])
        add("rew2", 256, 1, self._maxRank["rew2"])
        add("done1", S, 256, self._maxRank["done1"])
        add("done2", 256, 1, self._maxRank["done2"])

        add("s4_B", wm.s4.B.target.in_features, wm.s4.B.target.out_features, self._maxRank["s4_B"])
        add("s4_C", wm.s4.C.target.in_features, wm.s4.C.target.out_features, self._maxRank["s4_C"])
        add("s4_D0", wm.s4.D0.target.in_features, wm.s4.D0.target.out_features, self._maxRank["s4_D0"])
        add("s4_gate", wm.s4.gate.target.in_features, wm.s4.gate.target.out_features, self._maxRank["s4_gate"])
        add("s4_out_gate",wm.s4.out_gate.target.in_features,wm.s4.out_gate.target.out_features,self._maxRank["s4_out_gate"])

        add("hnn_to", wm.phys_hnn.to_qp.target.in_features, wm.phys_hnn.to_qp.target.out_features, self._maxRank["hnn_to"])
        add("hnn_H1", wm.phys_hnn.H[0].target.in_features, wm.phys_hnn.H[0].target.out_features, self._maxRank["hnn_H1"])
        add("hnn_H2", wm.phys_hnn.H[2].target.in_features, wm.phys_hnn.H[2].target.out_features, self._maxRank["hnn_H2"])
        add("hnn_from", wm.phys_hnn.from_qp.target.in_features, wm.phys_hnn.from_qp.target.out_features, self._maxRank["hnn_from"])

        add("ode_f1", wm.phys_ode.f[0].target.in_features, wm.phys_ode.f[0].target.out_features, self._maxRank["ode_f1"])
        add("ode_f2", wm.phys_ode.f[2].target.in_features, wm.phys_ode.f[2].target.out_features, self._maxRank["ode_f2"])

        add("mix", wm.mix_gate[0].target.in_features, wm.mix_gate[0].target.out_features, self._maxRank["mix"])

        add("conn_enc_s",
            wm.conn.enc_s[1].linear.target.in_features,
            wm.conn.enc_s[1].linear.target.out_features,
            self._maxRank["conn_enc_s"])
        add("conn_enc_a",
            wm.conn.enc_a[1].linear.target.in_features,
            wm.conn.enc_a[1].linear.target.out_features,
            self._maxRank["conn_enc_a"])
        add("conn_gamma",
            wm.conn.film_gamma_a.linear.target.in_features,
            wm.conn.film_gamma_a.linear.target.out_features,
            self._maxRank["conn_gamma"])
        add("conn_beta",
            wm.conn.film_beta_a.linear.target.in_features,
            wm.conn.film_beta_a.linear.target.out_features,
            self._maxRank["conn_beta"])
        for i, blk in enumerate(wm.conn.blocks):
            add(f"conn_blk{i}_ff1",
                blk.ff[0].linear.target.in_features,
                blk.ff[0].linear.target.out_features,
                self._maxRank["conn_blk_ff1"])
            add(f"conn_blk{i}_ff2",
                blk.ff[3].linear.target.in_features,
                blk.ff[3].linear.target.out_features,
                self._maxRank["conn_blk_ff2"])
        if wm.conn.use_lowrank:
            add("conn_head_uv",
                wm.conn.head_uv.linear.target.in_features,
                wm.conn.head_uv.linear.target.out_features,
                self._maxRank["conn_head_uv"])
        if wm.conn.use_full:
            add("conn_head_full",
                wm.conn.head_full.linear.target.in_features,
                wm.conn.head_full.linear.target.out_features,
                self._maxRank["conn_head_full"])
        add("conn_mix",
            wm.conn.mix.linear.target.in_features,
            wm.conn.mix.linear.target.out_features,
            self._maxRank["conn_mix"])

        if wm.use_decoder:
            add("dec1", S, S, self._maxRank["dec1"])
            add("dec2", S, V, self._maxRank["dec2"])
        return specs

    def ExportState(self):
        return self.base.ExportState()

    def ForwardWithDeltas(
        self,
        x: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]],
        **kwargs,) -> Dict[str, torch.Tensor]:

        wm = self.base  

        if isinstance(x, dict):
            vision = x["visionSeq"] if "visionSeq" in x else x["vision"]
            keys = x["keysVec"] if "keysVec" in x else x["keys"]
            mouse = x["mouseSeq"]  if "mouseSeq" in x else x["mouse"]
            h0 = x.get("h0") or x.get("hPrev")
            z0 = x.get("z0") or x.get("zPrev")
            rewardSeq = x.get("rewardSeq")
            doneSeq  = x.get("doneSeq")
        else:
            vision = x
            keys = kwargs["keysVec"] if "keysVec" in kwargs else kwargs["keys"]
            mouse = kwargs["mouseSeq"] if "mouseSeq" in kwargs else kwargs["mouse"]
            h0 = kwargs.get("h0") or kwargs.get("hPrev")
            z0 = kwargs.get("z0") or kwargs.get("zPrev")
            rewardSeq = kwargs.get("rewardSeq")
            doneSeq = kwargs.get("doneSeq")

        B = vision.size(0)
        device = vision.device
        dtype = vision.dtype

        if h0 is None:
            h0 = torch.zeros(B, wm.deter_dim, device=device, dtype=dtype)
        if z0 is None:
            z0 = torch.zeros(B, wm.stoch_dim, device=device, dtype=dtype)

        deltas: Dict[str, Optional[torch.Tensor]] = (deltasPerLayer[0] if (deltasPerLayer and len(deltasPerLayer) > 0) else {})

        def eff_linear(lo: "GrowableLoRALinear", deltaExtra: Optional[torch.Tensor]):
            W = lo.target.weight
            base_delta = lo.DeltaWeight()
            if base_delta is not None:
                W = W + base_delta
            if deltaExtra is not None:
                W = W + deltaExtra
            return W, lo.target.bias

        a_enc = wm.action_encoder(keys, mouse)
        W_act, b_act = eff_linear(wm.act_proj[0], deltas.get("act", None))
        a_t = F.linear(a_enc, W_act, b_act)
        a_t = wm.act_proj[1](a_t)
        a_t = wm.act_proj[2](a_t)

        has_s4_delta = any(deltas.get(k, None) is not None for k in ("s4_B", "s4_C", "s4_D0", "s4_gate", "s4_out_gate"))
        if not has_s4_delta:
            h_next = wm.s4.Step(z0, a_t, updateState=False)
        else:
            u = torch.cat([z0, a_t], dim=-1)

            W_gate, b_gate = eff_linear(wm.s4.gate, deltas.get("s4_gate", None))
            W_B, b_B = eff_linear(wm.s4.B, deltas.get("s4_B", None))
            W_C, b_C = eff_linear(wm.s4.C, deltas.get("s4_C", None))
            W_D0, b_D0 = eff_linear(wm.s4.D0, deltas.get("s4_D0", None))

            g = torch.sigmoid(F.linear(u, W_gate, b_gate))
            Bu = F.linear(u, W_B, b_B) * g

            x_prev = wm.s4.x
            if (
                x_prev.dim() != 2
                or x_prev.size(0) != B
                or x_prev.size(1) != wm.s4.N
                or x_prev.device != device
                or x_prev.dtype != dtype):
                x_prev = torch.zeros(B, wm.s4.N, device=device, dtype=dtype)

            x_next = wm.s4.CayleyStep(wm.s4.theta, x_prev, Bu, wm.s4.dt)

            W_outg, b_outg = eff_linear(wm.s4.out_gate, deltas.get("s4_out_gate", None))
            y_lin = F.linear(x_next, W_C, b_C) + F.linear(u, W_D0, b_D0)
            y_glu = y_lin * torch.sigmoid(F.linear(x_next, W_outg, b_outg))
            y = wm.s4.ln_y(y_glu)
            y = y + wm.s4.ffn(wm.s4.ln_ffn(y))
            h_next = y  

        W_prior, b_prior = eff_linear(wm.prior_net[0], deltas.get("prior", None))
        prior_out = F.linear(h_next, W_prior, b_prior)
        mu_p, logstd_p = prior_out.chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        v1 = wm.obs_enc[0](vision)
        W_obs1, b_obs1 = eff_linear(wm.obs_enc[1], deltas.get("obs1", None))
        h_obs = F.linear(v1, W_obs1, b_obs1)
        h_obs = wm.obs_enc[2](h_obs)
        h_obs = wm.obs_enc[3](h_obs)
        W_obs2, b_obs2 = eff_linear(wm.obs_enc[4], deltas.get("obs2", None))
        raw_e = F.linear(h_obs, W_obs2, b_obs2)

        e_in = raw_e
        if wm._use_memory:
            with torch.no_grad():
                mem_h = wm.MemRetrieve(raw_e)
            if mem_h is not None:
                mem_e = wm.mem_val_to_e(mem_h)
                gate_m = torch.sigmoid(wm.meta_gate_e(torch.cat([h_next, raw_e], dim=-1)))
                e_in = raw_e + gate_m * mem_e

        if wm._meta_dim > 0:
            ctx = wm.meta_ctx.view(1, -1).expand(B, -1)
            de_meta = wm.meta_to_e(ctx)
            dmu_meta = wm.meta_to_mu(ctx)
            gate_e = torch.sigmoid(wm.meta_gate_e(torch.cat([h_next, e_in], dim=-1)))
            gate_mu = torch.sigmoid(wm.meta_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            e_in = e_in + gate_e * de_meta
            mu_p = mu_p + gate_mu * dmu_meta

        ns_prior_logic = vision.new_tensor(0.0)
        logits_pr = None
        P_pr_train = None
        if wm._ns_enabled:
            logits_pr = wm.ns_head_prior(h_next)
            P_pr_raw = torch.sigmoid(logits_pr)
            P_pr_train = wm.NsProjectProbs(logits_pr)

            de_mu = wm.ns_to_delta_mu(P_pr_train)
            base_gate_mu = torch.sigmoid(wm.ns_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            with torch.no_grad():
                _, pen_pr = wm.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf_pr = wm.NsConfidence(P_pr_raw)
            gate_scale_mu = (1.0 - 0.25 * pen_pr.view(-1, 1)) * (0.6 + 0.4 * conf_pr)
            gate_mu = (base_gate_mu * gate_scale_mu).clamp(0.0, 1.0)
            mu_p = mu_p + gate_mu * de_mu

        ns_loss  = vision.new_tensor(0.0)
        ns_distill = vision.new_tensor(0.0)
        ns_logits = None
        sym_probs = None
        if wm._ns_enabled:
            ns_logits = wm.ns_head_post(torch.cat([h_next, e_in], dim=-1))
            sym_probs = wm.NsProjectProbs(ns_logits)

            de = wm.ns_to_delta_e(sym_probs)
            gate_e = torch.sigmoid(wm.ns_gate_e(torch.cat([h_next, e_in], dim=-1)))
            e_t = e_in + gate_e * de

            ns_loss, _ = wm.NsLogicLosses(sym_probs)

            if (logits_pr is not None) and (sym_probs is not None):
                with torch.no_grad():
                    P_teacher = torch.sigmoid(ns_logits)
                ns_distill = F.binary_cross_entropy_with_logits(logits_pr, P_teacher, reduction="mean")
                if P_pr_train is not None:
                    ns_prior_logic, _ = wm.NsLogicLosses(P_pr_train)
        else:
            e_t = e_in

        W_post, b_post = eff_linear(wm.post_net[0], deltas.get("post", None))
        post_out = F.linear(torch.cat([h_next, e_t], dim=-1), W_post, b_post)
        mu_q, logstd_q = post_out.chunk(2, dim=-1)
        logstd_q = ClampLogStd(logstd_q)
        z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)

        hz = torch.cat([h_next, z1], dim=-1)
        s_pre = wm.state_proj[0](hz)
        W_state, b_state = eff_linear(wm.state_proj[1], deltas.get("state", None))
        s_mid  = F.linear(s_pre, W_state, b_state)
        s_base = wm.state_proj[2](s_mid)

        hz_prev = torch.cat([h0, z0], dim=-1)
        s_prev_pre = wm.state_proj[0](hz_prev)
        s_prev_mid = F.linear(s_prev_pre, W_state, b_state)
        s_prev_base = wm.state_proj[2](s_prev_mid)

        hs = wm.conn.enc_s[0](s_prev_base)
        W_ces, b_ces = eff_linear(wm.conn.enc_s[1].linear, deltas.get("conn_enc_s", None))
        hs = F.linear(hs, W_ces, b_ces)
        hs = wm.conn.enc_s[2](hs)

        ha = wm.conn.enc_a[0](a_t)
        W_cea, b_cea = eff_linear(wm.conn.enc_a[1].linear, deltas.get("conn_enc_a", None))
        ha = F.linear(ha, W_cea, b_cea)
        ha = wm.conn.enc_a[2](ha)

        W_gam, b_gam = eff_linear(wm.conn.film_gamma_a.linear, deltas.get("conn_gamma", None))
        W_bet, b_bet = eff_linear(wm.conn.film_beta_a.linear, deltas.get("conn_beta", None))
        g = torch.tanh(F.linear(ha, W_gam, b_gam))
        b = F.linear(ha, W_bet, b_bet)

        h_conn = hs
        for i, blk in enumerate(wm.conn.blocks):
            y = (1.0 + g) * h_conn + b
            y = blk.ln(y)
            W_ff1, b_ff1 = eff_linear(blk.ff[0].linear, deltas.get(f"conn_blk{i}_ff1", None))
            t = F.linear(y, W_ff1, b_ff1)
            t = blk.ff[1](t); t = blk.ff[2](t)
            W_ff2, b_ff2 = eff_linear(blk.ff[3].linear, deltas.get(f"conn_blk{i}_ff2", None))
            t = F.linear(t, W_ff2, b_ff2)
            h_conn = h_conn + blk.alpha * t

        A_list = []
        if wm.conn.use_lowrank:
            W_uv, b_uv = eff_linear(wm.conn.head_uv.linear, deltas.get("conn_head_uv", None))
            uv = F.linear(h_conn, W_uv, b_uv)
            U, V = uv.split(wm.conn.S * wm.conn.r, dim=-1)
            U = U.view(-1, wm.conn.S, wm.conn.r)
            V = V.view(-1, wm.conn.S, wm.conn.r)
            A_lr = torch.bmm(U, V.transpose(1, 2)) - torch.bmm(V, U.transpose(1, 2))
            A_list.append(A_lr)
        if wm.conn.use_full:
            W_f, b_f = eff_linear(wm.conn.head_full.linear, deltas.get("conn_head_full", None))
            M = F.linear(h_conn, W_f, b_f).view(-1, wm.conn.S, wm.conn.S)
            A_full = 0.5 * (M - M.transpose(1, 2))
            A_list.append(A_full)

        if len(A_list) == 0:
            A_t_conn = torch.zeros(B, wm.conn.S, wm.conn.S, device=device, dtype=dtype)
        elif len(A_list) == 1:
            A_t_conn = A_list[0]
        else:
            W_mx, b_mx = eff_linear(wm.conn.mix.linear, deltas.get("conn_mix", None))
            w_mx = F.softmax(F.linear(h_conn, W_mx, b_mx), dim=-1)
            A_t_conn = (w_mx[:, :1].view(B, 1, 1) * A_list[0] + w_mx[:, 1:2].view(B, 1, 1) * A_list[1])

        if wm.conn.norm_clip and wm.conn.norm_clip > 0:
            fro = A_t_conn.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), wm.conn.norm_clip / fro).view(B, 1, 1)
            A_t_conn = A_t_conn * scale

        s_transport = wm.conn.TransportApply(A_t_conn, s_prev_base)

        prevA = wm._A_prev if (wm._A_prev is not None and wm._A_prev.shape == A_t_conn.shape) else None
        reg_A = wm.conn.ComputeGeomReg(A_t_conn, prevA)
        wm._A_prev = A_t_conn.detach()

        W_hnn_to, b_hnn_to = eff_linear(wm.phys_hnn.to_qp, deltas.get("hnn_to", None))
        W_hnn_H1, b_hnn_H1 = eff_linear(wm.phys_hnn.H[0], deltas.get("hnn_H1", None))
        W_hnn_H2, b_hnn_H2 = eff_linear(wm.phys_hnn.H[2], deltas.get("hnn_H2", None))
        W_hnn_from,b_hnn_from= eff_linear(wm.phys_hnn.from_qp, deltas.get("hnn_from", None))

        qp = F.linear(h_next, W_hnn_to, b_hnn_to)
        q, p_ = qp.chunk(2, dim=-1)

        need_graph = bool(wm.training and torch.is_grad_enabled())
        with torch.enable_grad():
            qp_in = torch.cat([q, p_], dim=-1).detach().requires_grad_(True)
            H_val = F.linear(F.gelu(F.linear(qp_in, W_hnn_H1, b_hnn_H1)), W_hnn_H2, b_hnn_H2)
            grad = torch.autograd.grad(H_val.sum(), qp_in, create_graph=need_graph, retain_graph=False)[0]
            dH_dq, dH_dp = grad.chunk(2, dim=-1)

            p_half = p_ - 0.5 * wm.phys_hnn.dt * dH_dq

            qp_mid = torch.cat([q, p_half], dim=-1).detach().requires_grad_(True)
            H_mid = F.linear(F.gelu(F.linear(qp_mid, W_hnn_H1, b_hnn_H1)), W_hnn_H2, b_hnn_H2)
            grad_mid = torch.autograd.grad(H_mid.sum(), qp_mid, create_graph=need_graph, retain_graph=False)[0]
            dH_dq_mid, dH_dp_mid = grad_mid.chunk(2, dim=-1)
            q_new = q + wm.phys_hnn.dt * dH_dp_mid

            qp_new = torch.cat([q_new, p_half], dim=-1).detach().requires_grad_(True)
            H_new = F.linear(F.gelu(F.linear(qp_new, W_hnn_H1, b_hnn_H1)), W_hnn_H2, b_hnn_H2)
            grad2 = torch.autograd.grad(H_new.sum(), qp_new, create_graph=need_graph, retain_graph=False)[0]
            dH_dq2, dH_dp2 = grad2.chunk(2, dim=-1)
            p_new = p_half - 0.5 * wm.phys_hnn.dt * dH_dq2

        h_phys = F.linear(torch.cat([q_new, p_new], dim=-1), W_hnn_from, b_hnn_from)

        h_new_eval = F.linear(F.gelu(F.linear(torch.cat([q_new, p_new], dim=-1), W_hnn_H1, b_hnn_H1)),W_hnn_H2,b_hnn_H2,)
        e_cons = (H_val.detach() - h_new_eval).pow(2).mean()

        W_ode_f1, b_ode_f1 = eff_linear(wm.phys_ode.f[0], deltas.get("ode_f1", None))
        W_ode_f2, b_ode_f2 = eff_linear(wm.phys_ode.f[2], deltas.get("ode_f2", None))

        def ode_f(inp):
            t = F.linear(inp, W_ode_f1, b_ode_f1)
            t = F.silu(t)
            t = F.linear(t, W_ode_f2, b_ode_f2)
            return t

        k1 = ode_f(torch.cat([h_next, a_t], dim=-1))
        mid = h_next + 0.5 * wm.phys_ode.dt * k1
        k2 = ode_f(torch.cat([mid, a_t], dim=-1))
        h_ode = h_next + wm.phys_ode.dt * k2
        e_smooth = (k2 - k1).pow(2).mean()

        sp0_phys = wm.state_proj[0](torch.cat([h_phys, z1], dim=-1))
        sp1_phys = F.linear(sp0_phys, W_state, b_state)
        s_phys = wm.state_proj[2](sp1_phys)

        sp0_ode = wm.state_proj[0](torch.cat([h_ode, z1], dim=-1))
        sp1_ode = F.linear(sp0_ode, W_state, b_state)
        s_ode = wm.state_proj[2](sp1_ode)

        W_mix, b_mix = eff_linear(wm.mix_gate[0], deltas.get("mix", None))
        logits_mix = F.linear(torch.cat([h_next, z1, a_t], dim=-1), W_mix, b_mix)
        w_mix = F.softmax(logits_mix, dim=-1)
        s_next = (w_mix[:, 0:1] * s_base + w_mix[:, 1:2] * s_transport + w_mix[:, 2:3] * s_phys + w_mix[:, 3:4] * s_ode)

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

        loss_recon = vision.new_tensor(0.0)
        recon = None
        if wm.use_decoder:
            W_dec1, b_dec1 = eff_linear(wm.obs_dec[0], deltas.get("dec1", None))
            dec_mid = F.linear(s_next, W_dec1, b_dec1)
            dec_mid = wm.obs_dec[1](dec_mid)
            W_dec2, b_dec2 = eff_linear(wm.obs_dec[2], deltas.get("dec2", None))
            recon = F.linear(dec_mid, W_dec2, b_dec2)
            loss_recon = F.mse_loss(recon, vision, reduction="mean")

        alphaKl = float(kwargs.get("alphaKl", 0.8))
        freeNats = float(kwargs.get("freeNats", 1.0))
        reconCoef = float(kwargs.get("reconCoef", 1.0))
        rewardCoef = float(kwargs.get("rewardCoef", 1.0))
        doneCoef = float(kwargs.get("doneCoef", 1.0))
        nsCoef = float(kwargs.get("nsCoef", 1.0))
        nsDistillCoef = float(kwargs.get("nsDistillCoef", 1e-2))
        nsPriorLogicCoef = float(kwargs.get("nsPriorLogicCoef", 1e-3))

        if rewardSeq is None:
            loss_reward = F.mse_loss(r_pred, torch.zeros_like(r_pred), reduction="mean")
        else:
            loss_reward = F.mse_loss(r_pred, rewardSeq, reduction="mean")

        loss_done = F.binary_cross_entropy_with_logits(
            d_logit,
            torch.zeros_like(d_logit) if doneSeq is None else doneSeq.float(),
            reduction="mean",)

        loss_kl = BalancedKL(mu_q, logstd_q, mu_p, logstd_p, alpha=alphaKl, freeNats=freeNats).mean()

        if not wm._ns_enabled:
            ns_prior_logic = vision.new_tensor(0.0)

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic
            + 1e-5 * e_smooth
            + 1e-4 * e_cons
            + reg_A)

        if wm._use_memory:
            with torch.no_grad():
                wm.MemAdd(raw_e.detach(), h_next.detach())

        out = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "s_next": s_next,
            "h_next": h_next,
            "z_next": z1,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "mu_p": mu_p,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "logstd_q": logstd_q,}
        
        if wm.use_decoder:
            out["recon"] = recon
            out["recon_target"] = vision
        if wm._ns_enabled:
            out["ns_logits"] = ns_logits
            out["ns_probs"]  = sym_probs

        return out


    @torch.no_grad()
    def StepPriorWithDeltas(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wm = self.base

        deltas_list = self.GetCurrentSimDeltas(detach=True, clone=True, skipZeros=True)
        deltas = deltas_list[0] if (deltas_list and len(deltas_list) > 0) else {}

        B = hPrev.size(0)
        device = hPrev.device
        dtype = hPrev.dtype

        def eff_linear(lo: "GrowableLoRALinear", delta_extra: torch.Tensor | None):
            W = lo.target.weight
            base_delta = lo.DeltaWeight()
            if base_delta is not None:
                W = W + base_delta
            if delta_extra is not None:
                W = W + delta_extra
            return W, lo.target.bias

        W_act, b_act = eff_linear(wm.act_proj[0], deltas.get("act", None))
        a_t = F.linear(actionEnc, W_act, b_act)
        a_t = wm.act_proj[1](a_t)
        a_t = wm.act_proj[2](a_t)

        s4_has_delta = any(deltas.get(k) is not None for k in ("s4_B", "s4_C", "s4_D0", "s4_gate", "s4_out_gate"))
        if not s4_has_delta:
            h_next = wm.s4.Step(zPrev, a_t, updateState=False)
        else:
            u = torch.cat([zPrev, a_t], dim=-1)

            W_gate, b_gate = eff_linear(wm.s4.gate, deltas.get("s4_gate", None))
            W_B, b_B = eff_linear(wm.s4.B, deltas.get("s4_B", None))
            W_C, b_C = eff_linear(wm.s4.C, deltas.get("s4_C", None))
            W_D0, b_D0 = eff_linear(wm.s4.D0, deltas.get("s4_D0", None))
            W_outg, b_outg = eff_linear(wm.s4.out_gate, deltas.get("s4_out_gate", None))

            g  = torch.sigmoid(F.linear(u, W_gate, b_gate))
            Bu = F.linear(u, W_B, b_B) * g

            x_prev = wm.s4.x
            if (
                x_prev.dim() != 2
                or x_prev.size(0) != B
                or x_prev.size(1) != wm.s4.N
                or x_prev.device != device
                or x_prev.dtype != dtype):
                x_prev = torch.zeros(B, wm.s4.N, device=device, dtype=dtype)

            x_next = wm.s4.CayleyStep(wm.s4.theta, x_prev, Bu, wm.s4.dt)
            y_lin = F.linear(x_next, W_C, b_C) + F.linear(u, W_D0, b_D0)
            y_glu = y_lin * torch.sigmoid(F.linear(x_next, W_outg, b_outg))
            y = wm.s4.ln_y(y_glu)
            y = y + wm.s4.ffn(wm.s4.ln_ffn(y))
            h_next = y

        W_prior, b_prior = eff_linear(wm.prior_net[0], deltas.get("prior", None))
        prior_out = F.linear(h_next, W_prior, b_prior)
        mu_p, logstd_p = prior_out.chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        if wm._ns_enabled and wm._ns_bias_prior:
            ns_logits = wm.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = wm.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = wm.NsConfidence(P_raw)

            base_gate = torch.sigmoid(wm.ns_gate_mu(torch.cat([h_next, mu_p], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)

            dmu = wm.ns_to_delta_mu(Q)
            mu_p = mu_p + gate * dmu

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        hz = torch.cat([h_next, z_next], dim=-1)
        sp0 = wm.state_proj[0](hz)
        W_state, b_state = eff_linear(wm.state_proj[1], deltas.get("state", None))
        sp1 = F.linear(sp0, W_state, b_state)
        s_base = wm.state_proj[2](sp1)

        hz_prev = torch.cat([hPrev, zPrev], dim=-1)
        sp0_prev = wm.state_proj[0](hz_prev)
        sp1_prev = F.linear(sp0_prev, W_state, b_state)
        s_prev_base = wm.state_proj[2](sp1_prev)

        hs = wm.conn.enc_s[0](s_prev_base)
        W_ces, b_ces = eff_linear(wm.conn.enc_s[1].linear, deltas.get("conn_enc_s", None))
        hs = F.linear(hs, W_ces, b_ces)
        hs = wm.conn.enc_s[2](hs)

        ha = wm.conn.enc_a[0](a_t)
        W_cea, b_cea = eff_linear(wm.conn.enc_a[1].linear, deltas.get("conn_enc_a", None))
        ha = F.linear(ha, W_cea, b_cea)
        ha = wm.conn.enc_a[2](ha)

        W_gam, b_gam = eff_linear(wm.conn.film_gamma_a.linear, deltas.get("conn_gamma", None))
        W_bet, b_bet = eff_linear(wm.conn.film_beta_a.linear, deltas.get("conn_beta", None))
        g = torch.tanh(F.linear(ha, W_gam, b_gam))
        b = F.linear(ha, W_bet, b_bet)

        h_conn = hs
        for i, blk in enumerate(wm.conn.blocks):
            y = (1.0 + g) * h_conn + b
            y = blk.ln(y)
            W_ff1, b_ff1 = eff_linear(blk.ff[0].linear, deltas.get(f"conn_blk{i}_ff1", None))
            t = F.linear(y, W_ff1, b_ff1)
            t = blk.ff[1](t); t = blk.ff[2](t)
            W_ff2, b_ff2 = eff_linear(blk.ff[3].linear, deltas.get(f"conn_blk{i}_ff2", None))
            t = F.linear(t, W_ff2, b_ff2)
            h_conn = h_conn + blk.alpha * t

        A_list = []
        if wm.conn.use_lowrank:
            W_uv, b_uv = eff_linear(wm.conn.head_uv.linear, deltas.get("conn_head_uv", None))
            uv = F.linear(h_conn, W_uv, b_uv)
            U, V = uv.split(wm.conn.S * wm.conn.r, dim=-1)
            U = U.view(-1, wm.conn.S, wm.conn.r)
            V = V.view(-1, wm.conn.S, wm.conn.r)
            A_lr = torch.bmm(U, V.transpose(1, 2)) - torch.bmm(V, U.transpose(1, 2))
            A_list.append(A_lr)

        if wm.conn.use_full:
            W_f, b_f = eff_linear(wm.conn.head_full.linear, deltas.get("conn_head_full", None))
            M = F.linear(h_conn, W_f, b_f).view(-1, wm.conn.S, wm.conn.S)
            A_full = 0.5 * (M - M.transpose(1, 2))
            A_list.append(A_full)

        if len(A_list) == 0:
            A_t_conn = torch.zeros(B, wm.conn.S, wm.conn.S, device=device, dtype=dtype)
        elif len(A_list) == 1:
            A_t_conn = A_list[0]
        else:
            W_mx, b_mx = eff_linear(wm.conn.mix.linear, deltas.get("conn_mix", None))
            w_mx = F.softmax(F.linear(h_conn, W_mx, b_mx), dim=-1)
            A_t_conn = (w_mx[:, :1].view(B, 1, 1) * A_list[0] + w_mx[:, 1:2].view(B, 1, 1) * A_list[1])

        if wm.conn.norm_clip and wm.conn.norm_clip > 0:
            fro = A_t_conn.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), wm.conn.norm_clip / fro).view(B, 1, 1)
            A_t_conn = A_t_conn * scale

        s_transport = wm.conn.TransportApply(A_t_conn, s_prev_base)

        W_hnn_to, b_hnn_to = eff_linear(wm.phys_hnn.to_qp, deltas.get("hnn_to", None))
        W_hnn_H1, b_hnn_H1 = eff_linear(wm.phys_hnn.H[0], deltas.get("hnn_H1", None))
        W_hnn_H2, b_hnn_H2 = eff_linear(wm.phys_hnn.H[2], deltas.get("hnn_H2", None))
        W_hnn_from, b_hnn_from = eff_linear(wm.phys_hnn.from_qp, deltas.get("hnn_from", None))

        qp = F.linear(h_next, W_hnn_to, b_hnn_to)
        q, p_ = qp.chunk(2, dim=-1)

        need_graph = bool(wm.training and torch.is_grad_enabled())
        with torch.enable_grad():
            qp_in = torch.cat([q, p_], dim=-1).detach().requires_grad_(True)
            H_val = F.linear(F.gelu(F.linear(qp_in, W_hnn_H1, b_hnn_H1)), W_hnn_H2, b_hnn_H2)
            grad = torch.autograd.grad(H_val.sum(), qp_in, create_graph=need_graph, retain_graph=False)[0]
            dH_dq, dH_dp = grad.chunk(2, dim=-1)

            p_half = p_ - 0.5 * wm.phys_hnn.dt * dH_dq

            qp_mid = torch.cat([q, p_half], dim=-1).detach().requires_grad_(True)
            H_mid = F.linear(F.gelu(F.linear(qp_mid, W_hnn_H1, b_hnn_H1)), W_hnn_H2, b_hnn_H2)
            grad_mid = torch.autograd.grad(H_mid.sum(), qp_mid, create_graph=need_graph, retain_graph=False)[0]
            dH_dq_mid, dH_dp_mid = grad_mid.chunk(2, dim=-1)
            q_new = q + wm.phys_hnn.dt * dH_dp_mid

            qp_new = torch.cat([q_new, p_half], dim=-1).detach().requires_grad_(True)
            H_new = F.linear(F.gelu(F.linear(qp_new, W_hnn_H1, b_hnn_H1)), W_hnn_H2, b_hnn_H2)
            grad2 = torch.autograd.grad(H_new.sum(), qp_new, create_graph=need_graph, retain_graph=False)[0]
            dH_dq2, dH_dp2 = grad2.chunk(2, dim=-1)
            p_new = p_half - 0.5 * wm.phys_hnn.dt * dH_dq2

        h_phys = F.linear(torch.cat([q_new, p_new], dim=-1), W_hnn_from, b_hnn_from)
        sp0_phys = wm.state_proj[0](torch.cat([h_phys, z_next], dim=-1))
        sp1_phys = F.linear(sp0_phys, W_state, b_state)
        s_phys = wm.state_proj[2](sp1_phys)

        W_ode_f1, b_ode_f1 = eff_linear(wm.phys_ode.f[0], deltas.get("ode_f1", None))
        W_ode_f2, b_ode_f2 = eff_linear(wm.phys_ode.f[2], deltas.get("ode_f2", None))

        def ode_f(inp: torch.Tensor) -> torch.Tensor:
            t = F.linear(inp, W_ode_f1, b_ode_f1)
            t = F.silu(t)
            t = F.linear(t, W_ode_f2, b_ode_f2)
            return t

        k1 = ode_f(torch.cat([h_next, a_t], dim=-1))
        mid = h_next + 0.5 * wm.phys_ode.dt * k1
        k2 = ode_f(torch.cat([mid, a_t], dim=-1))
        h_ode = h_next + wm.phys_ode.dt * k2

        sp0_ode = wm.state_proj[0](torch.cat([h_ode, z_next], dim=-1))
        sp1_ode = F.linear(sp0_ode, W_state, b_state)
        s_ode = wm.state_proj[2](sp1_ode)

        W_mix, b_mix = eff_linear(wm.mix_gate[0], deltas.get("mix", None))
        logits_mix = F.linear(torch.cat([h_next, z_next, a_t], dim=-1), W_mix, b_mix)
        w_mix = F.softmax(logits_mix, dim=-1)
        s_next = (w_mix[:, 0:1] * s_base + w_mix[:, 1:2] * s_transport + w_mix[:, 2:3] * s_phys + w_mix[:, 3:4] * s_ode)

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

        return h_next, z_next, s_next, r_pred, d_prob
    

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        wm = self.base
        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "obs1":
            wm.obs_enc[1].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "obs2":
            wm.obs_enc[4].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "act":
            wm.act_proj[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "prior":
            wm.prior_net[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "post":
            wm.post_net[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "state":
            wm.state_proj[1].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "rew1":
            wm.rew_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "rew2":
            wm.rew_head[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "done1":
            wm.done_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "done2":
            wm.done_head[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "dec1":
            wm.obs_dec[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "dec2":
            wm.obs_dec[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        if site == "s4_B":
            wm.s4.B.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "s4_C":
            wm.s4.C.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "s4_D0":
            wm.s4.D0.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "s4_gate":
            wm.s4.gate.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "s4_out_gate":
            wm.s4.out_gate.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        if site == "hnn_to":
            wm.phys_hnn.to_qp.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "hnn_H1":
            wm.phys_hnn.H[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "hnn_H2":
            wm.phys_hnn.H[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "hnn_from":
            wm.phys_hnn.from_qp.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        if site == "ode_f1":
            wm.phys_ode.f[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "ode_f2":
            wm.phys_ode.f[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        if site == "mix":
            wm.mix_gate[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        if site == "conn_enc_s":
            wm.conn.enc_s[1].linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "conn_enc_a":
            wm.conn.enc_a[1].linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "conn_gamma":
            wm.conn.film_gamma_a.linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "conn_beta":
            wm.conn.film_beta_a.linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        if site.startswith("conn_blk"):
            tag = site[len("conn_blk"):] 
            idx_str, which = tag.split("_ff")
            i = int(idx_str)
            if which == "1":
                wm.conn.blocks[i].ff[0].linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
            if which == "2":
                wm.conn.blocks[i].ff[3].linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        if site == "conn_head_uv" and wm.conn.use_lowrank:
            wm.conn.head_uv.linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "conn_head_full" and wm.conn.use_full:
            wm.conn.head_full.linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        if site == "conn_mix":
            wm.conn.mix.linear.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

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

        self.wm = RSSMWorldModel(
            visionDim=1024, actionDim=256, deterDim=256, stochDim=32, stateDim=256,
            useDecoder=True, useMemory=False, nsEnabled=True).to(self.device)
        self.wm.ResetHidden(batchSize=4)

    def Batch(self, B, keyIdx=(17,)):
        vision = torch.randn(B, 1024, device=self.device)
        keys = torch.zeros(B, 106, device=self.device)
        for k in keyIdx:
            keys[:, k] = 1.0
        mouse = torch.randn(B, 2, device=self.device)
        return vision, keys, mouse

    def MKX(self, vision, keys, mouse, sample=False, hPrev=None, zPrev=None):
        x = {"vision": vision, "keys": keys, "mouse": mouse, "sample": sample}
        if hPrev is not None: x["hPrev"] = hPrev
        if zPrev is not None: x["zPrev"] = zPrev
        return x

    def SeedNonzeroCandidates(self, wrapper, scale=1e-3):
        with torch.no_grad():
            for _, layer_list in wrapper.cand.items():
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
            self.wm.ResetHidden(batchSize=B)

            a_enc = self.wm.action_encoder(keys, mouse)
            h0 = torch.zeros(B, self.wm.deter_dim, device=self.device)
            z0 = torch.zeros(B, self.wm.stoch_dim, device=self.device)

            h_before, z_before = self.wm.ExportState()
            out = self.wm.StepPosterior(h0, z0, vision, a_enc, sample=False)
            h_after, z_after = self.wm.ExportState()

            ok_shapes = (
                out["h_next"].shape == (B, self.wm.deter_dim)
                and out["z_next"].shape == (B, self.wm.stoch_dim)
                and out["s_next"].shape == (B, self.wm.state_dim)
                and out["r_pred"].shape == (B,)
                and out["d_prob"].shape == (B,))
            
            changed = (not torch.allclose(h_before, h_after)) or (not torch.allclose(z_before, z_after))
            in_range = (out["d_prob"].min().item() >= 0.0) and (out["d_prob"].max().item() <= 1.0)

            with torch.no_grad():
                a_t = self.wm.act_proj(a_enc)
                logits = self.wm.mix_gate(torch.cat([out["h_next"], out["z_next"], a_t], dim=-1))
                w = torch.softmax(logits, dim=-1)
                mix_ok = torch.allclose(w.sum(dim=-1), torch.ones(B, device=w.device), atol=1e-6)

            ok = ok_shapes and changed and in_range and mix_ok
            print("RSSM StepPosterior test " + ("passed." if ok else "failed."))
            return ok
        except Exception as e:
            print(f"RSSM StepPosterior test FAILED: {type(e).__name__}: {e}")
            return False

    def TestRSSMStepPriorOnly(self):
        try:
            B = 4
            _, keys, mouse = self.Batch(B, keyIdx=(30,))
            self.wm.ResetHidden(batchSize=B)

            a_enc = self.wm.action_encoder(keys, mouse)
            h0 = torch.zeros(B, self.wm.deter_dim, device=self.device)
            z0 = torch.zeros(B, self.wm.stoch_dim, device=self.device)

            h_prev, z_prev = self.wm.ExportState()
            h1, z1, s1, r, d = self.wm.StepPriorOnly(h0, z0, a_enc, sample=False)

            ok_shapes = (
                h1.shape == (B, self.wm.deter_dim)
                and z1.shape == (B, self.wm.stoch_dim)
                and s1.shape == (B, self.wm.state_dim)
                and r.shape == (B,)
                and d.shape == (B,))
            
            h_after, z_after = self.wm.ExportState()
            not_written = torch.allclose(h_prev, h_after) and torch.allclose(z_prev, z_after)
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
            self.wm.train()
            for p in self.wm.parameters():
                p.requires_grad_(True)
            self.wm.ResetHidden(batchSize=B)

            out = self.wm.ForwardTrainSeq(
                visionSeq=vision, keysVec=keys, mouseSeq=mouse,
                rewardSeq=None, doneSeq=None,
                alphaKl=0.8, freeNats=1.0,
                reconCoef=1.0, rewardCoef=1.0, doneCoef=1.0)

            loss = out["loss"]
            if not torch.isfinite(loss).item():
                print("ForwardTrainSeq loss is not finite.")
                return False

            self.wm.zero_grad(set_to_none=True)
            loss.backward()
            print("RSSM ForwardTrainSeq test passed. loss =", float(loss.item()), " | distill=", float(out["loss_ns_distill"].item()))
            return True
        except Exception as e:
            print(f"ForwardTrainSeq test FAILED: {type(e).__name__}: {e}")
            return False

    def TestConnRegReset(self):
        try:
            B = 2
            vision, keys, mouse = self.Batch(B, keyIdx=(18,))
            self.wm.ResetHidden(batchSize=B)
            _ = self.wm.ForwardTrainSeq(vision, keys, mouse)
            has_prev = (self.wm._A_prev is not None) and (self.wm._A_prev.shape == (B, self.wm.state_dim, self.wm.state_dim))
            self.wm.ResetHidden(batchSize=B)
            cleared = (self.wm._A_prev is None)
            ok = has_prev and cleared
            print("Conn regularization cache reset " + ("passed." if ok else "failed."))
            return ok
        except Exception as e:
            print(f"Conn regularization cache reset FAILED: {type(e).__name__}: {e}")
            return False

    def TestWrapperAPIBasics(self):
        try:
            wrapper = WorldModelOnlineWrapper(self.wm, initRankEach=0, autoRank=False).to(self.device)
            wrapper.train()
            B = 4
            vision, keys, mouse = self.Batch(B, keyIdx=(17,57))
            self.wm.ResetHidden(batchSize=B)
            out = wrapper(self.MKX(vision, keys, mouse, sample=False))
            ok = ( ("h_next" in out) and ("z_next" in out) and ("s_next" in out)
                   and ("r_pred" in out) and ("d_prob" in out)
                   and out["s_next"].shape == (B, self.wm.state_dim))
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
            self.wm.ResetHidden(batchSize=B)

            out0 = wrapper.ForwardWithDeltas(self.MKX(vision, keys, mouse, sample=False), None, None, None, [{}])
            Z = self.wm.stoch_dim
            A = torch.randn(Z, self.wm.action_dim, device=self.device) * 1e-3
            out1 = wrapper.ForwardWithDeltas(self.MKX(vision, keys, mouse, sample=False), None, None, None, [{"act": A}])

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
            ok_commit = wrapper.CommitOne("act", 0, A, Bm, 1.0)
            n1 = len(lo.A_list)
            grew = ok_commit and (n1 == n0 + 1)

            Bsz = 4
            vision, keys, mouse = self.Batch(Bsz, keyIdx=(33,))
            self.wm.ResetHidden(batchSize=Bsz)

            with torch.no_grad():
                last_alpha = lo.alpha[-1].clone()
                lo.alpha[-1].zero_()
                out_before = wrapper(self.MKX(vision, keys, mouse, sample=False))
                lo.alpha[-1].copy_(last_alpha)

            out_after = wrapper(self.MKX(vision, keys, mouse, sample=False))
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
            self.wm.ResetHidden(batchSize=B)

            out = wrapper(self.MKX(vision, keys, mouse, sample=False))
            loss = ( F.mse_loss(out["r_pred"], torch.zeros_like(out["r_pred"])) +
                     0.5 * F.binary_cross_entropy(out["d_prob"], torch.zeros_like(out["d_prob"])) +
                     0.1 * F.mse_loss(out["s_next"], torch.zeros_like(out["s_next"])) )

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
                    if (p.grad is not None) and torch.isfinite(p.grad).all().item() and (p.grad.abs().sum().item() > 0):
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
            self.wm.ResetHidden(batchSize=B)

            watch = None
            for p in wrapper.CandParameters():
                watch = p; break
            before = watch.detach().clone() if watch is not None else None

            steps = 40
            losses = []
            for t in range(1, steps + 1):
                out = wrapper(self.MKX(vision, keys, mouse, sample=False))
                loss = ( F.mse_loss(out["r_pred"], torch.zeros_like(out["r_pred"])) +
                         0.5 * F.binary_cross_entropy(out["d_prob"], torch.zeros_like(out["d_prob"])) +
                         0.1 * F.mse_loss(out["s_next"], torch.zeros_like(out["s_next"])) )
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
            base = RSSMWorldModel(visionDim=1024, actionDim=256, deterDim=256, stochDim=32, stateDim=256, useDecoder=True, useMemory=False, nsEnabled=True).to(self.device)
            base.train()
            B = 6
            vision, keys, mouse = self.Batch(B, keyIdx=(32,))
            out = base.ForwardTrainSeq(vision, keys, mouse)
            loss = out["loss"]
            base.zero_grad(set_to_none=True)
            loss.backward()

            sample_params = [
                base.obs_enc[1].target.weight,
                base.s4.B.target.weight,
                base.rew_head[0].target.weight,
                base.done_head[2].target.weight,]
            
            grads_ok = True
            for p in sample_params:
                g = p.grad
                if (g is None) or (not torch.isfinite(g).all().item()) or (g.abs().sum().item() == 0.0):
                    grads_ok = False
                    break
            print(f"Base grad flow: {'passed' if grads_ok else 'failed'} | loss={float(loss.item()):.6f}")
            return grads_ok
        except Exception as e:
            print(f"Base grad flow FAILED: {type(e).__name__}: {e}")
            return False

    def TestBaseTrainingLossDecreases(self):
        try:
            base = RSSMWorldModel(visionDim=1024, actionDim=256, deterDim=256, stochDim=32, stateDim=256, useDecoder=True, useMemory=False, nsEnabled=True).to(self.device)
            base.train()
            opt = torch.optim.Adam(base.parameters(), lr=1e-3)

            B = 8
            vision, keys, mouse = self.Batch(B, keyIdx=(32,))
            watch_params = [base.obs_enc[1].target.weight, base.s4.B.target.weight]
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
            self.wm.ResetHidden(batchSize=B)

            x1 = self.MKX(vision, keys, mouse, sample=False)
            out1 = wrapper(x1)
            mu_q_star = out1["mu_q"].detach(); logstd_q_star = out1["logstd_q"].detach()

            x2 = self.MKX(vision, keys, mouse, sample=False)
            out2 = wrapper(x2)

            base_loss = ( F.mse_loss(out2["r_pred"], torch.zeros_like(out2["r_pred"])) +
                          0.5 * F.binary_cross_entropy(out2["d_prob"], torch.zeros_like(out2["d_prob"])) +
                          0.1 * F.mse_loss(out2["s_next"], torch.zeros_like(out2["s_next"])) )
            kl_prior = KLDiagNormal(mu_q_star, logstd_q_star, out2["mu_p"], out2["logstd_p"]).mean()
            loss = base_loss + 0.1 * kl_prior

            for p in wrapper.CandParameters():
                if p.grad is not None:
                    p.grad.zero_()
            loss.backward()

            def site_ok(name: str) -> bool:
                slot = wrapper.cand[name][0]
                def hasg(lst):
                    return any((p.grad is not None) and torch.isfinite(p.grad).all().item() and (p.grad.abs().sum().item() > 0) for p in lst)
                return hasg(slot["A"]) and hasg(slot["B"]) and hasg(slot["s"])

            missing = [name for name in ["obs1","obs2","act","prior","post","state","rew1","rew2","done1","done2"] if not site_ok(name)]
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
            wrapper.eval()
            wm.eval()

            B = 3
            vision, keys, mouse = self.Batch(B, keyIdx=(31, 57))
            wm.ResetHidden(batchSize=B)

            h0 = torch.zeros(B, wm.deter_dim, device=self.device)
            z0 = torch.zeros(B, wm.stoch_dim, device=self.device)

            a_enc = wm.action_encoder(keys, mouse)
            out_base = wm.StepPosterior(h0, z0, vision, a_enc, sample=False)

            x = self.MKX(vision, keys, mouse, sample=False, hPrev=h0, zPrev=z0)
            out_wrap = wrapper.ForwardWithDeltas(x, None, None, None, [{}])

            must_keys = [
                "h_next", "z_next", "s_next",
                "r_pred", "d_prob",
                "mu_p", "logstd_p", "mu_q", "logstd_q",]

            ok = True
            for k in must_keys:
                ok = ok and (k in out_base) and (k in out_wrap)
                if k in out_base and k in out_wrap:
                    ok = ok and (out_base[k].shape == out_wrap[k].shape)

            if wm._ns_enabled:
                ok = ok and ("ns_logits" in out_wrap) and ("ns_probs" in out_wrap)

            print("TestWorldMTool.Parity posterior (API/shape) " + ("passed." if ok else "failed."))
            return ok

        except Exception as e:
            print(f"TestWorldMTool.Parity posterior FAILED: {type(e).__name__}: {e}")
            return False

    def TestParityPrior(self):
        try:
            torch.manual_seed(0)
            wm = self.wm
            wrapper = WorldModelOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device)
            wrapper.eval()
            wm.eval()

            B = 3
            _, keys, mouse = self.Batch(B, keyIdx=(33,))
            wm.ResetHidden(batchSize=B)

            h0 = torch.zeros(B, wm.deter_dim, device=self.device)
            z0 = torch.zeros(B, wm.stoch_dim, device=self.device)
            a_enc = wm.action_encoder(keys, mouse)

            h1, z1, s1, r1, d1 = wm.StepPriorOnly(h0, z0, a_enc, sample=False)

            ok_shapes = (
                h1.shape == (B, wm.deter_dim)
                and z1.shape == (B, wm.stoch_dim)
                and s1.shape == (B, wm.state_dim)
                and r1.shape == (B,)
                and d1.shape == (B,))

            vision = torch.randn(B, wm.vision_dim, device=self.device)
            x = self.MKX(vision, keys, mouse, sample=False, hPrev=h0, zPrev=z0)
            _ = wrapper.ForwardWithDeltas(x, None, None, None, [{}])

            print("Parity prior (base StepPriorOnly shape) " + ("passed." if ok_shapes else "failed."))
            return ok_shapes

        except Exception as e:
            print(f"Parity prior FAILED: {type(e).__name__}: {e}")
            return False

    def RunAll(self):
        results = {
            "ActionEncoder": self.TestActionEncoder(),
            "RSSMStepPosterior": self.TestRSSMStepPosterior(),
            "RSSMStepPriorOnly": self.TestRSSMStepPriorOnly(),
            "ForwardTrainSeq": self.TestForwardTrainSeq(),
            "ConnRegReset": self.TestConnRegReset(),
            "WrapperAPIBasics": self.TestWrapperAPIBasics(),
            "ForwardWithDeltasInjection": self.TestForwardWithDeltasInjection(),
            "WrapperManualGrowTrainAndCommit": self.TestCommitOneGrowAndValueChange(),
            "GradFlowCandidates": self.TestGradFlowCandidates(),
            "WrapperTrainLossDecreases": self.TestWrapperTrainLossDecreases(),
            "BaseGradFlow": self.TestBaseGradFlow(),
            "BaseTrainLossDecreases": self.TestBaseTrainingLossDecreases(),
            "GradCoverageCandidateSites": self.TestGradCoverageCandidateSites(),
            "ParityPosterior": self.TestParityPosterior(),
            "ParityPrior": self.TestParityPrior(),}
        
        passed = sum(1 for ok in results.values() if ok)
        total = len(results)
        print(f"\nWorldModel test summary: {passed}/{total} passed.")
        return all(results.values())