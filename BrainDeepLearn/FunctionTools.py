from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def GetParametersScale(like: Optional[torch.Tensor] = None):
    val = 1e-1
    if like is None:
        return val
    return torch.as_tensor(val, device=like.device, dtype=like.dtype)


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
        factory = {"device": self.target.weight.device, "dtype": self.target.weight.dtype}
        A = init.get("A", torch.randn(addRank, self.in_f, **factory) * 1e-4)
        B = init.get("B", torch.randn(self.out_f, addRank, **factory) * 1e-4)
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(**factory))
        B = nn.Parameter(B.contiguous().to(**factory))
        s = nn.Parameter(torch.as_tensor(s, **factory))

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
            s_eff = torch.tanh(s) * GetParametersScale(s) 
            dW = dW + s_eff * (B @ A)
        return dW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.target.weight
        delta = self.DeltaWeight()
        if delta is not None:
            W = W + delta
        return F.linear(x, W, self.target.bias)


@dataclass
class SiteSpec:
    name: str
    nLayers: int
    inDim: int
    outDim: int
    maxRank: int
    allocFn: Callable[[int, torch.device, torch.dtype], Tuple[nn.Parameter, nn.Parameter, nn.Parameter]]
    composeFn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]



class AGICoreModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.register_buffer("anchor_", torch.empty(0), persistent=True)

    @property
    def device(self):
        return self.anchor_.device

    @property
    def dtype(self):
        return self.anchor_.dtype

    def NewZeros(self, *shape, dtype=None):
        return torch.zeros(*shape, device=self.device, dtype=(dtype or self.dtype))
    
    def NewOnes(self, *shape, dtype=None):
        return torch.ones(*shape, device=self.device, dtype=(dtype or self.dtype))

    def NewTensor(self, data, *, dtype=None):
        return torch.as_tensor(data, device=self.device, dtype=(dtype or self.dtype))


class RotaryEmbedding(AGICoreModule):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        rotary_dim = max(0, int(dim))
        if (rotary_dim % 2) != 0:
            rotary_dim -= 1
        self.dim = rotary_dim
        self.base = float(base)

        if self.dim > 0:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / float(self.dim)))
        else:
            inv_freq = torch.empty(0, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def BuildCosSin(
        self,
        seqLen: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        offset: int = 0,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.dim <= 0:
            return None, None

        pos = torch.arange(offset, offset + int(seqLen), device=device, dtype=self.dtype)
        freq = torch.outer(pos, self.inv_freq)
        freq = torch.repeat_interleave(freq, repeats=2, dim=-1)
        cos = freq.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = freq.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin

    @staticmethod
    def RotateHalf(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack([-x_odd, x_even], dim=-1)
        return x_rot.flatten(start_dim=-2)

    def Apply(
        self,
        x: torch.Tensor,
        *,
        offset: int = 0,) -> torch.Tensor:
        if self.dim <= 0:
            return x

        rotary = x[..., :self.dim]
        passthrough = x[..., self.dim:]
        cos, sin = self.BuildCosSin(
            seqLen=int(x.size(-2)),
            device=x.device,
            dtype=x.dtype,
            offset=offset,)
        rotary = rotary * cos + self.RotateHalf(rotary) * sin
        if passthrough.numel() == 0:
            return rotary
        return torch.cat([rotary, passthrough], dim=-1)


class RoPEMultiheadAttention(AGICoreModule):
    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        dropout: float = 0.0,
        ropeBase: float = 10000.0,):
        super().__init__()
        if embedDim % numHeads != 0:
            raise ValueError(f"RoPEMultiheadAttention: embedDim={embedDim} must be divisible by numHeads={numHeads}.")

        self.embed_dim = int(embedDim)
        self.num_heads = int(numHeads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim, base=ropeBase)

    def ReshapeHeads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def MergeHeads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return x

    @staticmethod
    def PrepareMask(mask: torch.Tensor, B: int, Tq: int, Tk: int) -> torch.Tensor:
        if mask.dim() == 2:
            return mask.reshape(1, 1, Tq, Tk)
        if mask.dim() == 3:
            return mask.reshape(B, 1, Tq, Tk)
        if mask.dim() == 4:
            return mask
        raise ValueError(f"RoPEMultiheadAttention: unsupported attn_mask dim={int(mask.dim())}.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        needWeights: bool = True,
        attnMask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, Tq, _ = query.shape
        Tk = int(key.size(1))

        q = self.ReshapeHeads(self.q_proj(query))
        k = self.ReshapeHeads(self.k_proj(key))
        v = self.ReshapeHeads(self.v_proj(value))

        q = self.rope.Apply(q)
        k = self.rope.Apply(k)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        neg_large = torch.finfo(scores.dtype).min

        if attnMask is not None:
            attnMaskView = self.PrepareMask(attnMask, B, Tq, Tk)
            if attnMaskView.dtype == torch.bool:
                scores = scores.masked_fill(attnMaskView, neg_large)
            else:
                scores = scores + attnMaskView

        if keyPaddingMask is not None:
            padMask = keyPaddingMask.reshape(B, 1, 1, Tk)
            scores = scores.masked_fill(padMask, neg_large)

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = self.MergeHeads(out)
        out = self.out_proj(out)

        weights = None
        if needWeights:
            weights = attn.mean(dim=1)
        return out, weights


class BaseOnlineWrapper(nn.Module):
    def __init__(
        self,
        base: AGICoreModule,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,):
        super().__init__()
        self.base = base

        self.autoRank = bool(autoRank)
        self.evThreshold = float(evThreshold)
        self.gradEma = float(gradEma)

        self.sites: Dict[str, SiteSpec] = self.BuildSiteSpecs()
        assert len(self.sites) > 0, "No site specs provided."

        self.layerCount = max(spec.nLayers for spec in self.sites.values())

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.cand: Dict[str, List[Dict[str, nn.ParameterList]]] = {}
        self.gradEmaBuf: Optional[Dict[str, List[Dict[str, List[Optional[torch.Tensor]]]]]] = None

        self.InitCandidates(initRankEach)

        self.freezeOldPar = True

    @property
    def deviceRef(self):
        return self.base.device

    @property
    def dtypeRef(self):
        return self.base.dtype

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        raise NotImplementedError


    def GetCurrentSimDeltas(self, *, detach: bool = True, clone: bool = True, skipZeros: bool = True) -> List[Dict[str, Optional[torch.Tensor]]]:
        deltas: List[Dict[str, Optional[torch.Tensor]]] = []
        Z0_cache: Dict[Tuple[int,int,torch.device,torch.dtype], torch.Tensor] = {}

        for layerIdx in range(self.layerCount):
            row: Dict[str, Optional[torch.Tensor]] = {}
            for name, spec in self.sites.items():
                slot = self.cand[name][layerIdx]
                if len(slot["A"]) == 0:
                    if skipZeros:
                        row[name] = None
                    else:
                        k = (spec.outDim, spec.inDim, self.deviceRef, self.dtypeRef)
                        if k not in Z0_cache:
                            Z0_cache[k] = torch.zeros(spec.outDim, spec.inDim,device=self.deviceRef, dtype=self.dtypeRef)
                        row[name] = Z0_cache[k].clone()
                    continue

                delta = torch.zeros(spec.outDim, spec.inDim,device=self.deviceRef, dtype=self.dtypeRef)

                for a, b, s in zip(slot["A"], slot["B"], slot["s"]):
                    delta = delta + self.sites[name].composeFn(a, b, s)

                if skipZeros and torch.allclose(delta, torch.zeros_like(delta)):
                    row[name] = None
                else:
                    if detach: delta = delta.detach()
                    if clone: delta = delta.clone()
                    row[name] = delta
            deltas.append(row)
        return deltas


    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,):
        raise NotImplementedError

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        raise NotImplementedError

    def forward(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        **kwargs,) -> torch.Tensor:
        deltas = [self.ComposeLayerDelta(layerIdx) for layerIdx in range(self.layerCount)]
        return self.ForwardWithDeltas(x, keyPaddingMask, tdError, uncertainty, deltas, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def Update(self, action: str, **kwargs):
        act = str(action).lower()
        if act == "reset":
            self.gradEmaBuf = None
            self.InitCandidates(int(kwargs.get("initRankEach", 0)))
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "ranks":
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "accumulategrads":
            self.AccumulateGrads()
            return {"ok": True}

        elif act == "grow":
            growFactor = float(kwargs.get("growFactor", 2.0))
            addEach = int(kwargs.get("addEach", 0))
            self.GrowCandidates(growFactor=growFactor, addEach=addEach)
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "autogrow":
            self.CandidatesAuto()
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "commit":
            stats = self.WritebackToBase()
            self.InitCandidates(0)
            return {"ok": True, "commit_stats": stats, "ranks": self.CurrentRanks()}

        elif act == "rollback":
            self.InitCandidates(0)
            return {"ok": True, "ranks": self.CurrentRanks()}

        elif act == "set":
            if "evThreshold" in kwargs:
                self.evThreshold = float(kwargs["evThreshold"])
            if "gradEma" in kwargs:
                self.gradEma = float(kwargs["gradEma"])
            if "autoRank" in kwargs:
                self.autoRank = bool(kwargs["autoRank"])
            for k, v in kwargs.items():
                if k.startswith("maxRank:"):
                    site = k.split("maxRank:", 1)[1]
                    if site in self.sites:
                        self.sites[site].maxRank = int(v)
            return {"ok": True, "settings": {
                "evThreshold": self.evThreshold,
                "gradEma": self.gradEma,
                "autoRank": self.autoRank,
                "maxRank": {name: spec.maxRank for name, spec in self.sites.items()},}}

        else:
            raise ValueError(f"Unknown action {action}")
    
    def SetFreezeOldPar(self, isfreezeOld: bool):
        self.freezeOldPar = isfreezeOld

    def EmptyLayerSlot(self):
        return {"A": nn.ParameterList(), "B": nn.ParameterList(), "s": nn.ParameterList()}

    def InitCandidates(self, initRankEach: int):
        self.cand = {} 
        self.gradEmaBuf = None

        for name in self.sites:  
            layer_slots = [] 

            for _ in range(self.layerCount):
                slot = self.EmptyLayerSlot()  
                layer_slots.append(slot)

            self.cand[name] = layer_slots

        if initRankEach > 0:
            for name, spec in self.sites.items():
                for layerIdx in range(spec.nLayers):
                    a, b, s = spec.allocFn(initRankEach, self.deviceRef, self.dtypeRef)
                    self.cand[name][layerIdx]["A"].append(a)
                    self.cand[name][layerIdx]["B"].append(b)
                    self.cand[name][layerIdx]["s"].append(s)

    def CurrentRanks(self):
        eps = 1e-12
        out = {"perLayer": []}

        for layerIdx in range(self.layerCount):
            row = {}
            for name, spec in self.sites.items():
                r = 0  
                slotA = self.cand[name][layerIdx]["A"]
                slotS = self.cand[name][layerIdx]["s"]

                for aParam, sParam in zip(slotA, slotS):
                    sEff = float(torch.tanh(sParam.detach()).item()) * float(GetParametersScale(sParam))
                    if abs(sEff) > eps:
                        r += int(aParam.size(0))

                row[name] = r
            out["perLayer"].append(row)

        out["sum"] = {name: sum(row[name] for row in out["perLayer"]) for name in self.sites}
        return out

    def CandParameters(self):
        for name in self.sites:
            for layerIdx in range(self.layerCount):
                for p in self.cand[name][layerIdx]["A"]:
                    if p.requires_grad:
                        yield p
                for p in self.cand[name][layerIdx]["B"]:
                    if p.requires_grad:
                        yield p
                for p in self.cand[name][layerIdx]["s"]:
                    if p.requires_grad:
                        yield p

    @torch.no_grad()
    def AccumulateGrads(self):
        if self.gradEmaBuf is None:
            self.gradEmaBuf = {name: [{"A": [], "B": []} for _ in range(self.layerCount)] for name in self.sites}

        def ema_update(dstList, srcList):
            if len(dstList) < len(srcList):
                dstList.clear()
            out = []
            for i, t in enumerate(srcList):
                if t is None:
                    out.append(dstList[i] if i < len(dstList) else None)
                else:
                    val = t.detach()
                    if i < len(dstList) and dstList[i] is not None:
                        val = self.gradEma * dstList[i] + (1 - self.gradEma) * val
                    out.append(val)
            return out

        for name in self.sites:
            for layerIdx in range(self.layerCount):
                gA = [p.grad.clone() if (p.grad is not None) else None for p in self.cand[name][layerIdx]["A"]]
                gB = [p.grad.clone() if (p.grad is not None) else None for p in self.cand[name][layerIdx]["B"]]
                self.gradEmaBuf[name][layerIdx]["A"] = ema_update(self.gradEmaBuf[name][layerIdx]["A"], gA)
                self.gradEmaBuf[name][layerIdx]["B"] = ema_update(self.gradEmaBuf[name][layerIdx]["B"], gB)

    @torch.no_grad()
    def GrowCandidates(self, growFactor: float = 2.0, addEach: int = 0):
        dev, dt = self.deviceRef, self.dtypeRef
        for name, spec in self.sites.items():
            for layerIdx in range(spec.nLayers):
                cur = sum(p.size(0) for p in self.cand[name][layerIdx]["A"])
                target = max(cur, cur if growFactor <= 1.0 else int(round(cur * growFactor)))
                target += max(0, addEach)
                target = min(target, spec.maxRank)
                add = max(0, target - cur)
                if add <= 0:
                    continue
                a, b, s = spec.allocFn(add, dev, dt)
                self.cand[name][layerIdx]["A"].append(a)
                self.cand[name][layerIdx]["B"].append(b)
                self.cand[name][layerIdx]["s"].append(s)

    @torch.no_grad()
    def CandidatesAuto(self):
        if not self.autoRank:
            return

        if self.gradEmaBuf is None:
            for name, spec in self.sites.items():
                for layerIdx in range(spec.nLayers):
                    if len(self.cand[name][layerIdx]["A"]) == 0:
                        a, b, s = spec.allocFn(1, self.deviceRef, self.dtypeRef)
                        self.cand[name][layerIdx]["A"].append(a)
                        self.cand[name][layerIdx]["B"].append(b)
                        self.cand[name][layerIdx]["s"].append(s)
            return

        def sugges_rank(gradAList, gradBList, aList, bList, sList, inDim, outDim, maxRank):
            gAcc = None
            cnt = 0
            for gA, gB, aParam, bParam, sParam in zip(gradAList, gradBList, aList, bList, sList):
                if gA is None and gB is None:
                    continue
                sEff = float(torch.tanh(sParam.detach()).item()) * float(GetParametersScale(sParam))
                sEff = sEff if abs(sEff) > 1e-12 else 1.0
                parts = []
                if gA is not None:
                    bt = bParam.t()
                    g_a = (torch.linalg.pinv(bt) @ gA) / sEff
                    parts.append(g_a)
                if gB is not None:
                    at = aParam.t()
                    g_b = (gB @ torch.linalg.pinv(at)) / sEff
                    parts.append(g_b)
                if not parts:
                    continue
                gest = sum(parts) / len(parts)
                gAcc = gest if gAcc is None else (gAcc + gest)
                cnt += 1
            if cnt == 0 or gAcc is None:
                return 0
            gMat = (gAcc / cnt).detach()
            svals = torch.linalg.svdvals(gMat)
            if svals.numel() == 0:
                return 0
            ev = (svals ** 2).cumsum(0) / (svals ** 2).sum()
            rNeed = int((ev < self.evThreshold).sum().item() + 1)
            return max(0, min(rNeed, maxRank, inDim, outDim))

        for name, spec in self.sites.items():
            for layerIdx in range(spec.nLayers):
                cur = sum(p.size(0) for p in self.cand[name][layerIdx]["A"])
                rTarget = sugges_rank(
                    self.gradEmaBuf[name][layerIdx]["A"],
                    self.gradEmaBuf[name][layerIdx]["B"],
                    list(self.cand[name][layerIdx]["A"]),
                    list(self.cand[name][layerIdx]["B"]),
                    list(self.cand[name][layerIdx]["s"]),
                    spec.inDim, spec.outDim, spec.maxRank,)
                if rTarget > cur:
                    a, b, s = spec.allocFn(rTarget - cur, self.deviceRef, self.dtypeRef)
                    self.cand[name][layerIdx]["A"].append(a)
                    self.cand[name][layerIdx]["B"].append(b)
                    self.cand[name][layerIdx]["s"].append(s)
                elif 0 <= rTarget < cur:
                    delta = self.ComposeOne(name, layerIdx)
                    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
                    r = int(rTarget)
                    if r <= 0:
                        for sParam in self.cand[name][layerIdx]["s"]:
                            if isinstance(sParam, nn.Parameter):
                                sParam.data.zero_()
                    else:
                        s_set = torch.tensor(1.0, device=self.deviceRef, dtype=self.dtypeRef)
                        c = torch.tanh(s_set) * float(GetParametersScale(s_set))
                        sqrtS = torch.sqrt(S[:r] / c)
                        bNew = (U[:, :r] * sqrtS.unsqueeze(0)).contiguous() 
                        aNew = (Vh[:r, :] * sqrtS.unsqueeze(1)).contiguous() 
                        slot = self.cand[name][layerIdx]
                        slot["A"] = nn.ParameterList([nn.Parameter(aNew)])
                        slot["B"] = nn.ParameterList([nn.Parameter(bNew)])
                        slot["s"] = nn.ParameterList([nn.Parameter(s_set)])
                        if self.gradEmaBuf is not None:
                            self.gradEmaBuf[name][layerIdx]["A"] = []
                            self.gradEmaBuf[name][layerIdx]["B"] = []

    def ComposeOne(self, site: str, layerIdx: int) -> torch.Tensor:
        spec = self.sites[site]
        delta = torch.zeros(spec.outDim, spec.inDim, device=self.deviceRef, dtype=self.dtypeRef)

        if len(self.cand[site][layerIdx]["A"]) == 0:
            return delta
            
        for aParam, bParam, sParam in zip(
            self.cand[site][layerIdx]["A"],
            self.cand[site][layerIdx]["B"],
            self.cand[site][layerIdx]["s"],):
            delta = delta + self.sites[site].composeFn(aParam, bParam, sParam)
        return delta

    def ComposeLayerDelta(self, layerIdx: int) -> Dict[str, Optional[torch.Tensor]]:
        out = {}
        for name, spec in self.sites.items():
            if layerIdx >= spec.nLayers:
                out[name] = None
                continue
            slot = self.cand[name][layerIdx]
            if len(slot["A"]) == 0:
                out[name] = None
                continue

            dMat = self.ComposeOne(name, layerIdx)
            out[name] = dMat
        return out

    @torch.no_grad()
    def WritebackToBase(self,) -> Dict[str, float]:
        committed_rank = 0
        committed_triples = 0

        self.eval()
        self.base.eval()

        for name, spec in self.sites.items():
            for layerIdx in range(spec.nLayers):
                slot = self.cand[name][layerIdx]
                for aParam, bParam, sParam in zip(slot["A"], slot["B"], slot["s"]):
                    s_val = float(sParam.detach().item()) if torch.is_tensor(sParam) else float(sParam)
                    if aParam.numel() == 0 or bParam.numel() == 0 or abs(s_val) < 1e-12:
                        continue
                    did_commit = self.CommitOne(name, layerIdx, aParam.detach().clone(), bParam.detach().clone(), s_val,)
                    if did_commit:
                        committed_rank += int(aParam.size(0))
                        committed_triples += 1

        return {
            "committed_rank": float(committed_rank),
            "committed_triples": float(committed_triples),}
