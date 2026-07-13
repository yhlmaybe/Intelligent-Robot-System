from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Callable, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def GetParametersScale(like: Optional[torch.Tensor] = None):
    val = 1e-1
    if like is None:
        return val
    return torch.as_tensor(val, device=like.device, dtype=like.dtype)


@dataclass
class ReferenceWeights:
    memory_recency: torch.Tensor
    observed_weight: torch.Tensor
    memory_weight: torch.Tensor
    slot_weight: torch.Tensor


def BuildReferenceWeights(
    physicalState: Dict[str, torch.Tensor],
    currentStep: torch.Tensor,
    *,
    memoryScale,
    memoryDecayHorizon: float,) -> ReferenceWeights:
    m_phys = physicalState["MphysRaw"]
    observed = physicalState["Observed"].float() # [B, K_world]
    memory_age = (currentStep - physicalState["LastSeen"].float()).clamp_min(0.0)
    memory_recency = torch.exp(-memory_age / float(memoryDecayHorizon))
    observed_weight = m_phys * observed

    memory_weight = (
        memoryScale
        * m_phys
        * physicalState["SlotPresence"]
        * (1.0 - observed)
        * memory_recency)

    return ReferenceWeights(
        memory_recency=memory_recency,
        observed_weight=observed_weight,
        memory_weight=memory_weight,
        slot_weight=observed_weight + memory_weight)


def BuildReferenceScaleContext(
    observedPst: Dict[str, torch.Tensor],
    demandQuery: torch.Tensor) -> torch.Tensor:
    observed_strength = observedPst["ObservedSlotMask"] * observedPst["MphysRaw"]

    demand = F.normalize(demandQuery, dim=-1, eps=1e-6) # [B, D]
    slot = F.normalize(observedPst["SlotState"], dim=-1, eps=1e-6) # [B, K_obs, D]

    demand_match = torch.einsum("bkd,bd->bk", slot, demand).add(1.0).mul(0.5)
    matched_strength = observed_strength * demand_match
    unmatched_strength = observed_strength * (1.0 - demand_match)

    top_match = torch.topk(matched_strength, k=2, dim=1).values

    observed_total = observed_strength.sum(dim=1, keepdim=True).clamp_min(1e-6)

    best_match = top_match[:, :1]
    second_match = top_match[:, 1:2]

    mean_match = matched_strength.sum(dim=1, keepdim=True) / observed_total
    ambiguity = second_match / best_match.clamp_min(1e-6)
    unresolved = unmatched_strength.sum(dim=1, keepdim=True) / observed_total

    return torch.cat([
        observed_strength.mean(dim=1, keepdim=True),
        observed_strength.amax(dim=1, keepdim=True),
        best_match,
        mean_match,
        1.0 - best_match,
        1.0 - observed_strength.amax(dim=1, keepdim=True),
        ambiguity,
        unresolved], dim=-1)


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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,):
        def saved_indices(name: str):
            key_prefix = f"{prefix}{name}."
            return {
                int(key[len(key_prefix):])
                for key in state_dict
                if key.startswith(key_prefix) and key[len(key_prefix):].isdigit()}

        a_indices = saved_indices("A_list")
        b_indices = saved_indices("B_list")
        s_indices = saved_indices("alpha")
        if a_indices or b_indices or s_indices:
            expected = set(range(len(a_indices)))
            if a_indices != expected or b_indices != expected or s_indices != expected:
                error_msgs.append(
                    f"{prefix[:-1]} has inconsistent dynamic LoRA topology: "
                    f"A={sorted(a_indices)}, B={sorted(b_indices)}, alpha={sorted(s_indices)}")
            else:
                saved_shapes = []
                for index in sorted(expected):
                    a_value = state_dict[f"{prefix}A_list.{index}"]
                    b_value = state_dict[f"{prefix}B_list.{index}"]
                    s_value = state_dict[f"{prefix}alpha.{index}"]
                    rank = int(a_value.size(0)) if a_value.dim() == 2 else -1
                    valid = (
                        tuple(a_value.shape) == (rank, self.in_f)
                        and tuple(b_value.shape) == (self.out_f, rank)
                        and s_value.numel() == 1)
                    if not valid:
                        error_msgs.append(
                            f"{prefix[:-1]} dynamic LoRA entry {index} has invalid shapes: "
                            f"A={tuple(a_value.shape)}, B={tuple(b_value.shape)}, "
                            f"alpha={tuple(s_value.shape)}")
                        saved_shapes = []
                        break
                    saved_shapes.append((rank, tuple(s_value.shape)))

                current_shapes = [
                    (int(a_value.size(0)), tuple(s_value.shape))
                    for a_value, s_value in zip(self.A_list, self.alpha)]
                if saved_shapes and current_shapes != saved_shapes:
                    self.A_list = nn.ParameterList()
                    self.B_list = nn.ParameterList()
                    self.alpha = nn.ParameterList()
                    for rank, scale_shape in saved_shapes:
                        scale = torch.zeros(
                            scale_shape,
                            device=self.target.weight.device,
                            dtype=self.target.weight.dtype)
                        self.Grow(rank, init={"scale": scale}, freezeOld=True)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs)

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


@torch.no_grad()
def SynchronizeGrowableLoRATopologyForFullLoad(
    root: nn.Module,
    stateDict: Dict[str, Any],
    ) -> int:
    """Make an explicit full-state load authoritative over empty LoRA topology.

    Generic ``load_state_dict(strict=False)`` remains suitable for partial updates and does
    not remove omitted adapters. Full checkpoint/parameter loaders call this helper first;
    when a layer's durable target weight is present but no committed LoRA entries are, the
    incoming state explicitly represents an empty committed topology.
    """
    cleared = 0
    for module_name, module in root.named_modules():
        if not isinstance(module, GrowableLoRALinear):
            continue
        prefix = f"{module_name}." if module_name else ""
        if f"{prefix}target.weight" not in stateDict:
            continue
        dynamic_prefixes = (
            f"{prefix}A_list.",
            f"{prefix}B_list.",
            f"{prefix}alpha.")
        if any(str(key).startswith(dynamic_prefixes) for key in stateDict):
            continue
        if not (module.A_list or module.B_list or module.alpha):
            continue
        cleared += len(module.A_list)
        module.A_list = nn.ParameterList()
        module.B_list = nn.ParameterList()
        module.alpha = nn.ParameterList()
    return cleared


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


def _RestoreOnlineWrapperTrainabilityAfterLoad(
    module: nn.Module,
    incompatibleKeys: Any,) -> None:
    del incompatibleKeys
    module.RestoreBaseTrainabilityAfterCommit()


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
        # Dynamic committed LoRA parameters are materialized while state_dict is loading.
        # They therefore do not inherit the base's pre-load requires_grad flags. Re-apply the
        # wrapper contract after all descendants have loaded, including when this wrapper is a
        # child of a larger BrainCore load rather than the direct load_state_dict target.
        self.register_load_state_dict_post_hook(
            _RestoreOnlineWrapperTrainabilityAfterLoad)

    def _apply(self, fn, recurse: bool = True):
        # Candidates are deliberately unregistered/ephemeral, so nn.Module._apply cannot move
        # them. Keep their identity (and therefore optimizer references) while moving data.
        super()._apply(fn, recurse=recurse)
        for parameter in self.CandParameters():
            with torch.no_grad():
                parameter.data = fn(parameter.data)
                if parameter.grad is not None:
                    parameter.grad.data = fn(parameter.grad.data)
        return self

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)
        for name in self.sites:
            for layerIdx in range(self.layerCount):
                slot = self.cand[name][layerIdx]
                for parameter_list in (slot["A"], slot["B"], slot["s"]):
                    for parameter in parameter_list:
                        if parameter.grad is None:
                            continue
                        if set_to_none:
                            parameter.grad = None
                        else:
                            parameter.grad.detach_()
                            parameter.grad.zero_()

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

    def RestoreBaseTrainabilityAfterCommit(self) -> None:
        if not self.freezeOldPar:
            return
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

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
                        slot = self.cand[name][layerIdx]
                        slot["A"] = nn.ParameterList()
                        slot["B"] = nn.ParameterList()
                        slot["s"] = nn.ParameterList()
                        if self.gradEmaBuf is not None:
                            self.gradEmaBuf[name][layerIdx]["A"] = []
                            self.gradEmaBuf[name][layerIdx]["B"] = []
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

        self.RestoreBaseTrainabilityAfterCommit()

        return {
            "committed_rank": float(committed_rank),
            "committed_triples": float(committed_triples),}


def HungarianRowsToCols(costRows: List[List[float]]) -> List[int]:
    if any(not math.isfinite(float(value)) for row in costRows for value in row):
        raise ValueError("HungarianRowsToCols requires finite costs")
    n = len(costRows)
    m = len(costRows[0])
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = costRows[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def HungarianAssignment(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if not bool(torch.isfinite(cost).all().item()):
        raise ValueError("HungarianAssignment requires finite costs")
    cost_cpu = cost.detach().float().cpu()
    rows, cols = int(cost_cpu.size(0)), int(cost_cpu.size(1))
    if rows <= cols:
        assignment = HungarianRowsToCols(cost_cpu.tolist())
        return (
            torch.arange(rows, device=cost.device, dtype=torch.long),
            torch.tensor(assignment, device=cost.device, dtype=torch.long))
    assignment = HungarianRowsToCols(cost_cpu.t().tolist())
    return (
        torch.tensor(assignment, device=cost.device, dtype=torch.long),
        torch.arange(cols, device=cost.device, dtype=torch.long))


class _TestOnlineBase(AGICoreModule):
    def __init__(self):
        super().__init__()
        self.adapter = GrowableLoRALinear(nn.Linear(3, 2, bias=True))


class _TestOnlineWrapper(BaseOnlineWrapper):
    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        def alloc(rank: int, device: torch.device, dtype: torch.dtype):
            return (
                nn.Parameter(torch.randn(rank, 3, device=device, dtype=dtype) * 0.1),
                nn.Parameter(torch.randn(2, rank, device=device, dtype=dtype) * 0.1),
                nn.Parameter(torch.tensor(0.3, device=device, dtype=dtype)))

        def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor):
            return torch.tanh(s) * GetParametersScale(s) * (b @ a)

        return {"adapter": SiteSpec("adapter", 1, 3, 2, 4, alloc, compose)}

    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask=None,
        tdError=None,
        uncertainty=None,
        deltasPerLayer=None,
        **kwargs,):
        adapter = self.base.adapter
        weight = adapter.target.weight
        committed = adapter.DeltaWeight()
        if committed is not None:
            weight = weight + committed
        candidate = deltasPerLayer[0]["adapter"]
        if candidate is not None:
            weight = weight + candidate
        return F.linear(x, weight, adapter.target.bias)

    @torch.no_grad()
    def CommitOne(self, site, layerIdx, a, b, scale):
        if site != "adapter" or layerIdx != 0:
            return False
        self.base.adapter.Grow(
            int(a.size(0)),
            init={"A": a, "B": b, "scale": scale},
            freezeOld=self.freezeOldPar)
        return True


class TestFunctionToolsMTool:
    def TestFullStateLoadClearsAbsentCommittedTopology(self) -> bool:
        try:
            torch.manual_seed(19)
            source = _TestOnlineBase()
            saved = source.state_dict()

            restored = _TestOnlineBase()
            restored.adapter.Grow(1)
            assert len(restored.adapter.A_list) == 1
            SynchronizeGrowableLoRATopologyForFullLoad(restored, saved)
            restored.load_state_dict(saved, strict=True)

            x = torch.randn(4, 3)
            assert len(restored.adapter.A_list) == 0
            assert len(restored.adapter.B_list) == 0
            assert len(restored.adapter.alpha) == 0
            assert torch.equal(restored.adapter(x), source.adapter(x))
            print("Full state load clears absent committed LoRA topology test passed.")
            return True
        except Exception as e:
            print(
                "Full state load clears absent committed LoRA topology test failed: "
                f"{type(e).__name__}: {e}")
            return False

    def TestOnlineWrapperStrictLoadRestoresBaseTrainability(self) -> bool:
        try:
            torch.manual_seed(13)
            source = _TestOnlineWrapper(_TestOnlineBase(), initRankEach=1)
            commit = source.Update("commit")
            assert commit["commit_stats"]["committed_triples"] == 1.0
            source_parent = nn.Module()
            source_parent.wrapper = source
            saved = source_parent.state_dict()

            restored = _TestOnlineWrapper(_TestOnlineBase(), initRankEach=0)
            restored_parent = nn.Module()
            restored_parent.wrapper = restored
            restored_parent.load_state_dict(saved, strict=True)
            committed = list(restored.base.adapter.A_list)
            assert committed, "committed LoRA topology was not restored"
            assert not any(
                parameter.requires_grad
                for parameter in restored.base.parameters()), (
                "strict load reactivated durable parameters in a frozen online base")
            print("Online wrapper strict-load trainability test passed.")
            return True
        except Exception as e:
            print(
                "Online wrapper strict-load trainability test failed: "
                f"{type(e).__name__}: {e}")
            return False

    def TestOnlineWrapperZeroGradIncludesCandidates(self) -> bool:
        try:
            wrapper = _TestOnlineWrapper(_TestOnlineBase(), initRankEach=1)
            candidate = list(wrapper.CandParameters())
            for parameter in candidate:
                parameter.grad = torch.ones_like(parameter)
            wrapper.zero_grad(set_to_none=True)
            assert all(parameter.grad is None for parameter in candidate)

            for parameter in candidate:
                parameter.grad = torch.ones_like(parameter)
            wrapper.zero_grad(set_to_none=False)
            assert all(
                parameter.grad is not None and int(torch.count_nonzero(parameter.grad).item()) == 0
                for parameter in candidate)
            print("Online wrapper candidate zero-grad test passed.")
            return True
        except Exception as e:
            print(f"Online wrapper candidate zero-grad test failed: {type(e).__name__}: {e}")
            return False

    def TestOnlineWrapperRankZeroSurvivesNextOptimizerStep(self) -> bool:
        try:
            torch.manual_seed(11)
            wrapper = _TestOnlineWrapper(_TestOnlineBase(), initRankEach=1)
            optimizer = torch.optim.Adam(list(wrapper.CandParameters()), lr=0.05)

            first_loss = wrapper(torch.randn(4, 3)).square().mean()
            first_loss.backward()
            optimizer.step()

            wrapper.zero_grad(set_to_none=True)
            slot = wrapper.cand["adapter"][0]
            slot["s"][0].grad = torch.ones_like(slot["s"][0])
            wrapper.Update("accumulategrads")
            wrapper.Update("autogrow")

            assert len(slot["A"]) == 0 and len(slot["B"]) == 0 and len(slot["s"]) == 0
            assert wrapper.gradEmaBuf["adapter"][0]["A"] == []
            assert wrapper.gradEmaBuf["adapter"][0]["B"] == []

            optimizer.step()
            assert wrapper.CurrentRanks()["sum"]["adapter"] == 0
            assert wrapper.ComposeLayerDelta(0)["adapter"] is None
            print("Online wrapper rank-zero optimizer-step test passed.")
            return True
        except Exception as e:
            print(f"Online wrapper rank-zero optimizer-step test failed: {type(e).__name__}: {e}")
            return False

    def TestGrowableLoRACommitSaveStrictLoad(self) -> bool:
        try:
            torch.manual_seed(7)
            source_base = _TestOnlineBase()
            source_wrapper = _TestOnlineWrapper(source_base, initRankEach=2)
            first_commit = source_wrapper.Update("commit")["commit_stats"]
            source_wrapper.Update("grow", addEach=1)
            second_commit = source_wrapper.Update("commit")["commit_stats"]
            assert first_commit["committed_rank"] == 2.0
            assert second_commit["committed_rank"] == 1.0

            sample = torch.randn(5, 3)
            expected = source_base.adapter(sample).detach()
            saved = source_base.state_dict()

            restored_base = _TestOnlineBase()
            restored_base.load_state_dict(saved, strict=True)
            actual = restored_base.adapter(sample).detach()

            assert len(restored_base.adapter.A_list) == 2
            assert [int(value.size(0)) for value in restored_base.adapter.A_list] == [2, 1]
            assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
            print("GrowableLoRA commit/save/strict-load test passed.")
            return True
        except Exception as e:
            print(f"GrowableLoRA commit/save/strict-load test failed: {type(e).__name__}: {e}")
            return False

    def TestHungarianRejectsNonFiniteCost(self) -> bool:
        try:
            for value in (float("nan"), float("inf"), -float("inf")):
                cost = torch.tensor([[value, value], [0.0, 1.0]])
                try:
                    HungarianAssignment(cost)
                except ValueError:
                    continue
                raise AssertionError(f"HungarianAssignment accepted non-finite cost {value}")
            print("Hungarian non-finite validation test passed.")
            return True
        except Exception as e:
            print(f"Hungarian non-finite validation test failed: {type(e).__name__}: {e}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "FullStateLoadClearsAbsentCommittedTopology": self.TestFullStateLoadClearsAbsentCommittedTopology(),
            "GrowableLoRACommitSaveStrictLoad": self.TestGrowableLoRACommitSaveStrictLoad(),
            "OnlineWrapperStrictLoadRestoresBaseTrainability": self.TestOnlineWrapperStrictLoadRestoresBaseTrainability(),
            "OnlineWrapperZeroGradIncludesCandidates": self.TestOnlineWrapperZeroGradIncludesCandidates(),
            "OnlineWrapperRankZeroSurvivesNextOptimizerStep": self.TestOnlineWrapperRankZeroSurvivesNextOptimizerStep(),
            "HungarianRejectsNonFiniteCost": self.TestHungarianRejectsNonFiniteCost(),}
        passed = sum(1 for value in results.values() if value)
        print(f"\n[FunctionTools Tests] {passed}/{len(results)} passed.")
        return results
