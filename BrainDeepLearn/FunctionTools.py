from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import torch
import torch.nn as nn


def GetParameterSScale(like: Optional[torch.Tensor] = None):
    val = 1e-1
    if like is None:
        return val
    return torch.as_tensor(val, device=like.device, dtype=like.dtype)


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
        self.register_buffer("anchor_", torch.empty(0), persistent=False)

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
            return {"ranks": self.CurrentRanks()}

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
            for name in self.sites:
                r = 0
                for aParam, sParam in zip(self.cand[name][layerIdx]["A"], self.cand[name][layerIdx]["s"]):
                    sVal = float(torch.tanh(sParam.detach()).item()) * 1e-1
                    if abs(sVal) > eps: r += int(aParam.size(0))
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
                sEff = float(torch.tanh(sParam.detach()).item()) * 1e-1
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
                        c = torch.tanh(s_set) * 1e-1
                        sqrtS = torch.sqrt(S[:r] / c).unsqueeze(0) 
                        bNew = (U[:, :r] * sqrtS).contiguous()
                        aNew = (sqrtS.t() @ Vh[:r, :]).contiguous()
                        self.cand[name][layerIdx]["A"].clear()
                        self.cand[name][layerIdx]["B"].clear()
                        self.cand[name][layerIdx]["s"].clear()
                        self.cand[name][layerIdx]["A"].append(nn.Parameter(aNew))
                        self.cand[name][layerIdx]["B"].append(nn.Parameter(bNew))
                        self.cand[name][layerIdx]["s"].append(nn.Parameter(s_set))

    def ComposeOne(self, site: str, layerIdx: int) -> torch.Tensor:
        spec = self.sites[site]
        if len(self.cand[site][layerIdx]["A"]) == 0:
            return torch.zeros(spec.outDim, spec.inDim, device=self.deviceRef, dtype=self.dtypeRef)
        delta = torch.zeros(spec.outDim, spec.inDim, device=self.deviceRef, dtype=self.dtypeRef)
        for aParam, bParam, sParam in zip(
            self.cand[site][layerIdx]["A"],
            self.cand[site][layerIdx]["B"],
            self.cand[site][layerIdx]["s"],):
            delta = delta + self.sites[site].composeFn(aParam, bParam, sParam)
        return delta

    def ComposeLayerDelta(self, layerIdx: int) -> Dict[str, Optional[torch.Tensor]]:
        out = {}
        for name, spec in self.sites.items():
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
