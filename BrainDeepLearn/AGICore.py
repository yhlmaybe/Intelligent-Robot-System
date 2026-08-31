from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import IntEnum
from collections import deque
import copy
import hashlib
import math
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from Config import BasicParameters
from CoreTypes import (
    BrainBuildSpec,
    BrainStepOutput,
    ContractAgentActInput,
    ContractAgentActOutput,
    ContractBrainStepInput,
    TEXT_TRUST_OPERATOR_COMMAND,
)
from PerceptionModule import (
    PerceiveExtractor,
    PerceptionOnlineWrapper,
    PerceptionRecallLoss,
    TopDownContext,
    VisualState,
)
from AttentionModule import AttentionExtractor, AttentionOnlineWrapper
from MemoryModule import MemoryExtractor, MemoryType
from DecisionModule import DecisionExtractor, DecisionPlannerExtractor
from DecisionDecoupler import DecisionDecouplerV2, PackedDecisionContext
from WorldModule import (
    PredictedVisualPack,
    RSSMWorldModel,
    WORLD_MEMORY_TENSOR_FIELDS,
    WorldOnlineWrapper,
)
from ValueEstimationModule import (
    COGNITIVE_COMPUTE_REASON_COUNT,
    ValueEstimationExtractor,
    ValueEstimationOnlineWrapper,
)
from ConsciousnessModule import ConsciousnessExtractor, ConsciousnessOutput
from IntentionModule import IntentionExtractor, IntentionOnlineWrapper
from OCRModule import OCREngineExtractor, OcrLineObs, OcrTrack
from PhysicalStateModule import (
    ContractPhysicalStateAdapter,
    PhysicalStateExtractor,
    PhysicalStateLoss,
)
from GoalModule import GoalGrounding, FourLevelGoalManager
from NeuroSymbolicModule import (
    CONTRACT_EVIDENCE_FIELDS,
    CONTRACT_EXECUTION_PREDICATES,
    CONTRACT_SLOT_PREDICATES,
    ContractGroundingOutput,
    ContractNeuroSymbolicGrounder,
    NeuroSymbolicOutput,
    NeuroSymbolicExtractor,
    OperatorRationale,
    OperatorStep,
    SymbolicFact,
)
from TemporalExecutionModule import (
    CANCEL,
    CONTINUE,
    DISPATCH,
    FAILSAFE_STOP,
    PACKED_TEMPORAL_KIND_NAMES,
    REDISPATCH,
    PackedTemporalEvent,
    PackedTemporalProposal,
    TemporalExecutionGateExtractor,
)
from ModuleMessagerManager import ModuleDim, ModuleMessagerManager
from FunctionTools import SynchronizeDynamicAdapterTopologiesForFullLoad
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    PackedEndEffectorTarget,
    RobotEmbodimentContractView,
)

BRAIN_RUNTIME_SCHEMA_VERSION = 33
WORLD_MEMORY_ARTIFACT_SCHEMA_VERSION = 2
AGENT_MEMORY_SCHEMA_VERSION = 2
BRAIN_RUNTIME_BUFFER_FIELDS = frozenset({
    "schema_version",
    "contract_id",
    "model_signature",
    "batch_size",
    "cognitive_state",
})
ONLINE_WRAPPER_ROOTS = ("perc", "attn", "world", "critic", "intention")
WORLD_RUNTIME_STATE_NAMES = frozenset(
    {f"_{name}" for name in WORLD_MEMORY_TENSOR_FIELDS}
    | {"s4.x"})
COGNITIVE_BACKBONE_SCHEMA_VERSION = 1
COGNITIVE_BACKBONE_ARTIFACT_FIELDS = frozenset({
    "schema_version",
    "cognitive_profile",
    "parameters",
})
COGNITIVE_BACKBONE_PARAMETER_PREFIXES = (
    "perc.",
    "attn.",
    "mem.",
    "actor.",
    "world.",
    "critic.",
    "conscious.",
    "intention.",
    "goal_manager.",
    "goal_grounding.",
    "goal_relation_adapter.",
    "goal_object_adapter.",
    "goal_requirement_head.",
    "attention_memory_top_down_adapter.",
    "attention_goal_top_down_adapter.",
    "attention_top_down_gain",
    "option_skill_gain",
    "world_abstract_decision_adapter.",
    "world_abstract_decision_gain",
    "OCR.",
)
COGNITIVE_BACKBONE_BLOCKED_PREFIXES = (
    "contract_physical_adapter.",
    "contract_joint_motion_action_adapter.",
    "contract_action_agency_encoder.",
    "contract_action_agency_gain",
    "contract_layer_agency_fuser.",
    "contract_layer_agency_gain",
    "contract_entity_summary_fuser.",
    "contract_entity_summary_gain",
    "packed_decision_decoupler.",
    "decision_decoupler.",
    "contract_pst_builder.",
    "pst_builder.",
    "contract_neuro_symbolic_grounder.",
    "contract_neuro_symbolic.",
    "neuro_symbolic.",
    "packed_temporal_gate.",
    "temporal_gate.",
    "cognitive_compute_gate.",
    "mem.usage_bank.",
    "mem.embodied_memory_expert.",
    "mem.embodied_output_proj.",
    "mem.embodied_memory_gate.",
    "actor.embodied_state_encoder.",
    "world.contract_embodiment_adapter.",
    "world.embodiment_adapter.",
    "world.robot_world_adapter.",
    "world.embodiment_context_proj.",
    "world.embodied_action_proj.",
)


def NormalizeRotation(value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value) or value.size(-1) != 4:
        raise ValueError("rotation must end in four coordinates")
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if bool((norm <= 1e-8).any().item()):
        raise ValueError("rotation norm must be positive")
    normalized = value / norm
    return torch.where(
        normalized[..., 3:4] < 0.0,
        -normalized,
        normalized)


def ComposeRotation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return NormalizeRotation(torch.stack((
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ), dim=-1))


def IsWorldRuntimeStateKey(name: str) -> bool:
    return (
        name.startswith("_pst_")
        or "._pst_" in name
        or any(
        name == runtime_name or name.endswith(f".{runtime_name}")
        for runtime_name in WORLD_RUNTIME_STATE_NAMES))


def BrainNonModelStateKeys(brain: nn.Module) -> frozenset[str]:
    parameter_names = frozenset(name for name, _ in brain.named_parameters())
    excluded = {
        name
        for name in brain.state_dict()
        if (
            IsWorldRuntimeStateKey(name)
            or name.startswith(("mem_copy.", "attn_copy.", "critic_copy."))
            or (
                name.startswith(("mem.", "critic."))
                and name not in parameter_names))}
    adaptive_modules_fn = getattr(brain, "AdaptiveRuntimeModules", None)
    if callable(adaptive_modules_fn):
        adaptive_module_ids = {
            id(module)
            for module in adaptive_modules_fn().values()}
        for module_path, module in brain.named_modules():
            if id(module) not in adaptive_module_ids:
                continue
            prefix = f"{module_path}." if module_path else ""
            excluded.update(
                f"{prefix}{buffer_name}"
                for buffer_name, _ in module.named_buffers())
    return frozenset(excluded)


def ExportBrainModelState(brain: nn.Module) -> Dict[str, torch.Tensor]:
    excluded = BrainNonModelStateKeys(brain)
    return {
        name: value
        for name, value in brain.state_dict().items()
        if name not in excluded}


def CanonicalCognitiveBackboneParameterName(name: str) -> str:
    if type(name) is not str or not name:
        raise ValueError("cognitive backbone parameter name must be non-empty")
    for root in ONLINE_WRAPPER_ROOTS:
        prefix = f"{root}.base."
        if name.startswith(prefix):
            return f"{root}.{name[len(prefix):]}"
    return name


def IsCognitiveBackboneParameter(name: str) -> bool:
    canonical = CanonicalCognitiveBackboneParameterName(name)
    return (
        canonical.startswith(COGNITIVE_BACKBONE_PARAMETER_PREFIXES)
        and not canonical.startswith(COGNITIVE_BACKBONE_BLOCKED_PREFIXES))


def CognitiveBackboneParameters(
    brain: nn.Module,
) -> Dict[str, nn.Parameter]:
    if not isinstance(brain, nn.Module):
        raise TypeError("cognitive backbone requires a torch module")
    if bool(getattr(brain, "is_online_learning", False)):
        for root in ONLINE_WRAPPER_ROOTS:
            wrapper = getattr(brain, root, None)
            if hasattr(wrapper, "CandParameters") and any(
                True for _ in wrapper.CandParameters()
            ):
                raise RuntimeError(
                    "online candidates must be materialized before cognitive export")
    parameters: Dict[str, nn.Parameter] = {}
    for name, value in brain.named_parameters():
        canonical = CanonicalCognitiveBackboneParameterName(name)
        if not IsCognitiveBackboneParameter(canonical):
            continue
        if canonical in parameters:
            raise RuntimeError(
                f"duplicate cognitive backbone parameter: {canonical}")
        parameters[canonical] = value
    if not parameters:
        raise ValueError("cognitive backbone parameter set is empty")
    return parameters


def ExportCognitiveBackboneState(brain: nn.Module) -> Dict[str, Any]:
    build_spec = getattr(brain, "brain_build_spec", None)
    if type(build_spec) is not BrainBuildSpec:
        raise TypeError("cognitive backbone source requires BrainBuildSpec")
    entries = []
    for name, value in sorted(CognitiveBackboneParameters(brain).items()):
        if not value.is_floating_point():
            raise TypeError(
                f"cognitive backbone parameter {name} must be floating point")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(
                f"cognitive backbone parameter {name} must be finite")
        entries.append((name, value.detach().cpu().clone()))
    return {
        "schema_version": COGNITIVE_BACKBONE_SCHEMA_VERSION,
        "cognitive_profile": build_spec.CognitiveProfilePayload(),
        "parameters": tuple(entries),
    }


def LoadCognitiveBackboneState(
    brain: nn.Module,
    artifact: Any,
) -> None:
    build_spec = getattr(brain, "brain_build_spec", None)
    if type(build_spec) is not BrainBuildSpec:
        raise TypeError("cognitive backbone target requires BrainBuildSpec")
    if type(artifact) is not dict or set(artifact) != COGNITIVE_BACKBONE_ARTIFACT_FIELDS:
        raise ValueError("cognitive backbone artifact fields do not match")
    if (
        type(artifact["schema_version"]) is not int
        or artifact["schema_version"] != COGNITIVE_BACKBONE_SCHEMA_VERSION
    ):
        raise ValueError("cognitive backbone schema is unsupported")
    build_spec.ValidateCognitiveProfileCompatibility(
        artifact["cognitive_profile"])
    entries = artifact["parameters"]
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError("cognitive backbone parameter list must be non-empty")
    incoming: Dict[str, torch.Tensor] = {}
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError("cognitive backbone parameter entry is invalid")
        name, value = entry
        if type(name) is not str or not name:
            raise ValueError("cognitive backbone parameter name must be non-empty")
        if name != CanonicalCognitiveBackboneParameterName(name):
            raise ValueError("cognitive backbone parameter name is not canonical")
        if not IsCognitiveBackboneParameter(name):
            raise ValueError(
                f"cognitive backbone parameter is outside the whitelist: {name}")
        if name in incoming:
            raise ValueError(
                f"duplicate cognitive backbone parameter: {name}")
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError(
                f"cognitive backbone parameter {name} must be a floating tensor")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(
                f"cognitive backbone parameter {name} must be finite")
        incoming[name] = value
    expected = CognitiveBackboneParameters(brain)
    missing = sorted(set(expected).difference(incoming))
    unknown = sorted(set(incoming).difference(expected))
    if missing or unknown:
        raise ValueError(
            f"cognitive backbone fields do not match: missing={missing}, "
            f"unknown={unknown}")
    for name, value in incoming.items():
        target = expected[name]
        if tuple(value.shape) != tuple(target.shape) or value.dtype != target.dtype:
            raise ValueError(
                f"cognitive backbone parameter {name} shape or dtype does not match")
    with torch.no_grad():
        for name, value in incoming.items():
            target = expected[name]
            target.copy_(value.to(device=target.device))


def ExportDeploymentModelState(brain: nn.Module) -> Dict[str, torch.Tensor]:
    state = ExportBrainModelState(brain)
    if not bool(getattr(brain, "is_online_learning", False)):
        return state
    for root in ONLINE_WRAPPER_ROOTS:
        wrapper = getattr(brain, root)
        if hasattr(wrapper, "CandParameters") and any(
            True for _ in wrapper.CandParameters()
        ):
            raise RuntimeError(
                "online candidates must be materialized before deployment export")
    canonical: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        mapped = name
        for root in ONLINE_WRAPPER_ROOTS:
            prefix = f"{root}.base."
            if name.startswith(prefix):
                mapped = f"{root}.{name[len(prefix):]}"
                break
        if mapped in canonical:
            raise RuntimeError(f"duplicate canonical deployment parameter: {mapped}")
        canonical[mapped] = value
    return canonical


def LoadDeploymentModelState(
    brain: nn.Module,
    state: Dict[str, torch.Tensor],) -> None:
    if not bool(getattr(brain, "is_online_learning", False)):
        LoadBrainModelState(brain, state)
        return
    wrapped_roots = tuple(
        root
        for root in ONLINE_WRAPPER_ROOTS
        if hasattr(getattr(brain, root), "base"))
    wrapped: Dict[str, torch.Tensor] = {}
    for name, value in state.items():
        mapped = name
        for root in wrapped_roots:
            prefix = f"{root}."
            if name.startswith(prefix):
                mapped = f"{root}.base.{name[len(prefix):]}"
                break
        if mapped in wrapped:
            raise RuntimeError(f"duplicate wrapped training parameter: {mapped}")
        wrapped[mapped] = value
    LoadBrainModelState(brain, wrapped)


def LoadBrainModelState(
    brain: nn.Module,
    state: Dict[str, torch.Tensor],) -> None:
    if type(state) is not dict:
        raise TypeError("brain model state must be a dictionary")
    non_model_keys = BrainNonModelStateKeys(brain)
    runtime_keys = sorted(name for name in state if name in non_model_keys)
    if runtime_keys:
        raise ValueError(
            f"brain parameter artifact contains non-model state: {runtime_keys}")
    SynchronizeDynamicAdapterTopologiesForFullLoad(brain, state)
    expected = ExportBrainModelState(brain)
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"brain model state fields do not match: missing={missing}, "
            f"unexpected={unexpected}")
    result = brain.load_state_dict(state, strict=False)
    expected_runtime = sorted(non_model_keys)
    if sorted(result.missing_keys) != expected_runtime or result.unexpected_keys:
        raise RuntimeError(
            "brain model load crossed the runtime-state boundary")


class CognitiveComputeMode(IntEnum):
    FAST_EXECUTE = 0
    DETAIL_EXECUTE = 1
    FULL_REPLAN = 2
    FAILSAFE = 3


COGNITIVE_COMPUTE_REASON_NAMES = (
    "PLAN_CACHE",
    "GOAL_INTENT",
    "MODEL_NOVELTY",
    "EXECUTION_DEVIATION",
    "SAFETY_FEEDBACK",
    "HIERARCHY_TRANSITION",
    "COMPUTE_VALUE",
)


@dataclass(frozen=True)
class CognitiveComputeDecision:
    mode: torch.Tensor
    hard_trigger: torch.Tensor
    activated_child_mask: torch.Tensor
    evc_trigger: torch.Tensor
    reason_target: torch.Tensor


@dataclass(frozen=True)
class ContractReplayTrace:
    context: torch.Tensor
    predicted_reward: torch.Tensor
    predicted_done: torch.Tensor
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    timestamp: torch.Tensor
    episode_version: int
    timeline_version: int


class CognitiveComputeGate(nn.Module):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        *,
        maxCacheAge: float = 32.0,
        worldSurpriseThreshold: float = 0.5,
        riskThreshold: float = 0.5,
        noveltyThreshold: float = 0.5,
        trackingErrorThreshold: float = 0.5,
        contactAnomalyThreshold: float = 0.5,
        evcThreshold: float = 0.0,
    ) -> None:
        super().__init__()
        if (
            len(COGNITIVE_COMPUTE_REASON_NAMES)
            != COGNITIVE_COMPUTE_REASON_COUNT
        ):
            raise RuntimeError("compute reason schema does not match Value")
        positive_thresholds = {
            "maxCacheAge": maxCacheAge,
            "worldSurpriseThreshold": worldSurpriseThreshold,
            "riskThreshold": riskThreshold,
            "noveltyThreshold": noveltyThreshold,
            "trackingErrorThreshold": trackingErrorThreshold,
            "contactAnomalyThreshold": contactAnomalyThreshold,
        }
        for name, value in positive_thresholds.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(maxCacheAge) <= 0.0:
            raise ValueError("maxCacheAge must be positive")
        if not math.isfinite(float(evcThreshold)):
            raise ValueError("evcThreshold must be finite")
        self.ContractView = contractView
        self.MaxCacheAge = float(maxCacheAge)
        self.WorldSurpriseThreshold = float(worldSurpriseThreshold)
        self.RiskThreshold = float(riskThreshold)
        self.NoveltyThreshold = float(noveltyThreshold)
        self.TrackingErrorThreshold = float(trackingErrorThreshold)
        self.ContactAnomalyThreshold = float(contactAnomalyThreshold)
        self.EvcThreshold = float(evcThreshold)
        self.register_buffer(
            "PreviousChildEnabled",
            torch.empty(0, dtype=torch.bool),
            persistent=False)

    def Reset(self) -> None:
        self.PreviousChildEnabled = self.PreviousChildEnabled.new_empty((0,))

    def ResetRows(self, doneMask: torch.Tensor) -> None:
        if (
            not torch.is_tensor(doneMask)
            or doneMask.dim() != 1
            or doneMask.dtype != torch.bool
        ):
            raise ValueError("compute gate reset mask must be batched boolean")
        if tuple(self.PreviousChildEnabled.shape[:1]) != tuple(doneMask.shape):
            self.Reset()
            return
        previous = self.PreviousChildEnabled.clone()
        previous[doneMask] = False
        self.PreviousChildEnabled = previous

    @staticmethod
    def BooleanEvent(
        value: torch.Tensor,
        name: str,
        batchSize: int,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.dtype != torch.bool
        ):
            raise ValueError(f"{name} must be a batched boolean event")
        return value.to(device=device)

    @staticmethod
    def RealEvent(
        value: torch.Tensor,
        name: str,
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.dtype == torch.bool
            or not (
                value.is_floating_point()
                or value.dtype in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64)
            )
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(f"{name} must be a finite batched real event")
        return value.to(device=device, dtype=dtype)

    def BuildFailsafe(
        self,
        batchSize: int,
        device: torch.device,
    ) -> CognitiveComputeDecision:
        reason_target = torch.zeros(
            batchSize,
            COGNITIVE_COMPUTE_REASON_COUNT,
            dtype=torch.bool,
            device=device)
        reason_target[:, 4] = True
        return CognitiveComputeDecision(
            mode=torch.full(
                (batchSize,),
                int(CognitiveComputeMode.FAILSAFE),
                dtype=torch.long,
                device=device),
            hard_trigger=torch.ones(
                batchSize,
                dtype=torch.bool,
                device=device),
            activated_child_mask=torch.zeros(
                batchSize,
                self.ContractView.end_effector_count,
                dtype=torch.bool,
                device=device),
            evc_trigger=torch.zeros(
                batchSize,
                dtype=torch.bool,
                device=device),
            reason_target=reason_target)

    def ChildActivation(
        self,
        childEnabled: torch.Tensor,
    ) -> torch.Tensor:
        previous = self.PreviousChildEnabled
        if (
            tuple(previous.shape) == tuple(childEnabled.shape)
            and previous.device == childEnabled.device
        ):
            activated = childEnabled & ~previous
        else:
            activated = torch.zeros_like(childEnabled)
        child_mask = self.ChildSlotMask(childEnabled.device)
        self.PreviousChildEnabled = childEnabled.detach().clone()
        return activated & child_mask

    def ChildSlotMask(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            tuple(index >= 0 for index in self.ContractView.parent_index),
            dtype=torch.bool,
            device=device).unsqueeze(0)

    def forward(
        self,
        feedbackPacket: BrainFeedbackPacket,
        *,
        planValid: torch.Tensor,
        cacheAge: torch.Tensor,
        goalChanged: torch.Tensor,
        intentChanged: torch.Tensor,
        targetChanged: torch.Tensor,
        worldSurprise: torch.Tensor,
        risk: torch.Tensor,
        novelty: torch.Tensor,
        noveltyRelevant: torch.Tensor,
        trackingError: torch.Tensor,
        contactAnomaly: torch.Tensor,
        evc: torch.Tensor,
        safetyViolation: Optional[torch.Tensor] = None,
        criticalInfeasible: Optional[torch.Tensor] = None,
    ) -> CognitiveComputeDecision:
        batch_size = int(feedbackPacket.joint_features.size(0))
        device = feedbackPacket.joint_features.device
        dtype = feedbackPacket.joint_features.dtype
        try:
            plan_valid = self.BooleanEvent(
                planValid, "planValid", batch_size, device)
            cache_age = self.RealEvent(
                cacheAge, "cacheAge", batch_size, device, dtype)
            goal_changed = self.BooleanEvent(
                goalChanged, "goalChanged", batch_size, device)
            intent_changed = self.BooleanEvent(
                intentChanged, "intentChanged", batch_size, device)
            target_changed = self.BooleanEvent(
                targetChanged, "targetChanged", batch_size, device)
            world_surprise = self.RealEvent(
                worldSurprise, "worldSurprise", batch_size, device, dtype)
            risk_value = self.RealEvent(
                risk, "risk", batch_size, device, dtype)
            novelty_value = self.RealEvent(
                novelty, "novelty", batch_size, device, dtype)
            novelty_relevant = self.BooleanEvent(
                noveltyRelevant, "noveltyRelevant", batch_size, device)
            tracking_error = self.RealEvent(
                trackingError, "trackingError", batch_size, device, dtype)
            contact_anomaly = self.RealEvent(
                contactAnomaly, "contactAnomaly", batch_size, device, dtype)
            evc_value = self.RealEvent(
                evc, "evc", batch_size, device, dtype)
            safety_violation = (
                torch.zeros(batch_size, dtype=torch.bool, device=device)
                if safetyViolation is None
                else self.BooleanEvent(
                    safetyViolation,
                    "safetyViolation",
                    batch_size,
                    device))
            critical_infeasible = (
                torch.zeros(batch_size, dtype=torch.bool, device=device)
                if criticalInfeasible is None
                else self.BooleanEvent(
                    criticalInfeasible,
                    "criticalInfeasible",
                    batch_size,
                    device))
        except (TypeError, ValueError, RuntimeError):
            self.Reset()
            return self.BuildFailsafe(batch_size, device)

        child_enabled = feedbackPacket.child_enabled.to(device=device)
        activated_child_mask = self.ChildActivation(child_enabled)
        feedback_failure = (
            feedbackPacket.target_active
            & ~feedbackPacket.endpoint_present).any(dim=-1)
        failsafe = (
            safety_violation
            | critical_infeasible
            | feedback_failure)

        hard_full = (
            ~plan_valid
            | goal_changed
            | intent_changed
            | target_changed
            | (cache_age >= self.MaxCacheAge)
            | (world_surprise >= self.WorldSurpriseThreshold)
            | (risk_value >= self.RiskThreshold)
            | (
                novelty_relevant
                & (novelty_value >= self.NoveltyThreshold))
            | (tracking_error >= self.TrackingErrorThreshold)
            | (contact_anomaly >= self.ContactAnomalyThreshold))
        evc_trigger = (
            ~failsafe
            & ~hard_full
            & (evc_value > self.EvcThreshold))
        full_replan = ~failsafe & (hard_full | evc_trigger)
        detail_execute = (
            ~failsafe
            & ~full_replan
            & (child_enabled & self.ChildSlotMask(device)).any(dim=-1))
        reason_target = torch.stack([
            ~plan_valid | (cache_age >= self.MaxCacheAge),
            goal_changed | intent_changed | target_changed,
            (world_surprise >= self.WorldSurpriseThreshold) | (
                novelty_relevant
                & (novelty_value >= self.NoveltyThreshold)),
            (tracking_error >= self.TrackingErrorThreshold) | (
                contact_anomaly >= self.ContactAnomalyThreshold),
            failsafe | (risk_value >= self.RiskThreshold),
            activated_child_mask.any(dim=-1),
            evc_trigger,
        ], dim=-1)
        mode = torch.full(
            (batch_size,),
            int(CognitiveComputeMode.FAST_EXECUTE),
            dtype=torch.long,
            device=device)
        mode = torch.where(
            detail_execute,
            mode.new_full(mode.shape, int(CognitiveComputeMode.DETAIL_EXECUTE)),
            mode)
        mode = torch.where(
            full_replan,
            mode.new_full(mode.shape, int(CognitiveComputeMode.FULL_REPLAN)),
            mode)
        mode = torch.where(
            failsafe,
            mode.new_full(mode.shape, int(CognitiveComputeMode.FAILSAFE)),
            mode)
        return CognitiveComputeDecision(
            mode=mode,
            hard_trigger=failsafe | hard_full,
            activated_child_mask=activated_child_mask,
            evc_trigger=evc_trigger,
            reason_target=reason_target)


class BrainCore(nn.Module):
    def __init__(
        self,
        brainBuildSpec: BrainBuildSpec,
        device: Optional[torch.device] = None,
        *,
        seqLen: int = BasicParameters.IMAGE_SEQ_LEN,
        prioritizeExtStr: bool = True,
        plasticOnlineLearning: bool = False,
        usePlanner: bool = True,
        plannerTeacherMode: bool = True,
        enablePerceptionSupervision: bool = False,
        saveModuleMessagerOutput: bool = True,
        needTrace: bool = True,
        slowPeriod: int = 4,
        ) -> None:
        super().__init__()
        if type(brainBuildSpec) is not BrainBuildSpec:
            raise TypeError("brainBuildSpec must be BrainBuildSpec")
        if brainBuildSpec.cognitive != ModuleDim.CognitiveProfile():
            raise ValueError(
                "BrainBuildSpec cognitive profile does not match the constructed architecture")
        self.contract_only = True
        self.brain_build_spec = brainBuildSpec
        self.robot_contract_view = brainBuildSpec.contract_view
        self.model_signature = brainBuildSpec.model_signature
        self.model_contract_id = self.model_signature
        self.description_id = self.robot_contract_view.description_id
        self.adapter_id = self.robot_contract_view.adapter_id
        self.register_buffer(
            "ContractStaticSlotTokens",
            torch.tensor(
                self.robot_contract_view.static_end_effector_tokens,
                dtype=torch.float32),
            persistent=True)
        self.SEQ_LEN = int(seqLen)
        if type(slowPeriod) is not int or slowPeriod <= 0:
            raise ValueError("slowPeriod must be a positive integer")
        self.slow_period = int(slowPeriod)
        self.is_online_learning = bool(plasticOnlineLearning)
        self.prioritize_ext_str = bool(prioritizeExtStr)
        self.need_trace = bool(needTrace)
        self.use_planner = bool(usePlanner)
        self.planner_teacher_mode = bool(plannerTeacherMode)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        perception_projection = self.robot_contract_view.perception_projection
        if perception_projection is None:
            raise ValueError(
                "brain contract requires a primary perception projection")
        expected_reference_size = (
            BasicParameters.IMAGE_SIZE,
            BasicParameters.IMAGE_SIZE)
        if perception_projection.reference_size != expected_reference_size:
            raise ValueError(
                "brain perception projection does not match the cognitive image size")
        self.calibration_id = perception_projection.calibration_id

        self.perc = PerceiveExtractor(
            projectionMatrix=torch.tensor(
                perception_projection.projection_matrix,
                dtype=torch.float32),
            imgSize=BasicParameters.IMAGE_SIZE,
            embedDim=ModuleDim.PerceptionEmbed,
            objectTokenCount=ModuleDim.PstObservedSlots,
            enableRecallAuxiliary=enablePerceptionSupervision)
        self.perception_recall_loss = (
            PerceptionRecallLoss()
            if enablePerceptionSupervision
            else None)
        self.attn = AttentionExtractor(
            embedDim=ModuleDim.AttentionFeat,
            sequenceLength=self.SEQ_LEN,
            structuredDim=ModuleDim.PerceptionEmbed,
            goalDim=ModuleDim.IntentionFeat,
            objectTokenCount=self.perc.object_token_count)
        self.mem = MemoryExtractor(
            inputDim=ModuleDim.AttentionFeat,
            ssmStateDim=ModuleDim.MemoryItem,
            memoryDim=ModuleDim.MemoryItem,
            outputDim=ModuleDim.MemoryFeat,
            emotionDim=ModuleDim.ValueEstimationOutEmotion)
        self.value_tensor_dim = int(ModuleDim.ValueTensorDim)
        self.actor = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat,
            intentDim=ModuleDim.IntentionFeat,
            includeNoSkill=True,
            valueTensorDim=self.value_tensor_dim,
            vNextTensorDim=self.value_tensor_dim,
            beliefDim=brainBuildSpec.cognitive.decision_dim,
            decisionDynDim=ModuleDim.DecisionDynDim,
            latentControlDim=ModuleDim.LatentControlDim,
            mapperEmbedDim=ModuleDim.MapperHiddenDim)
        self.contract_physical_adapter = ContractPhysicalStateAdapter(
            self.robot_contract_view,
            bodyDim=int(self.actor.body_state_feature_dim),
            controlFeedbackDim=int(
                self.actor.control_feedback_feature_dim),
            embodimentContextDim=int(
                self.actor.embodiment_context_feature_dim))
        action_agency_dim = int(self.actor.action_embed_dim)
        self.contract_joint_motion_action_adapter = nn.Sequential(
            nn.LayerNorm(int(self.actor.control_feedback_feature_dim)),
            nn.Linear(
                int(self.actor.control_feedback_feature_dim),
                action_agency_dim),
            nn.SiLU(),
            nn.Linear(action_agency_dim, action_agency_dim),
            nn.LayerNorm(action_agency_dim))
        self.contract_action_agency_encoder = nn.Sequential(
            nn.LayerNorm(action_agency_dim * 3 + 4),
            nn.Linear(action_agency_dim * 3 + 4, action_agency_dim * 2),
            nn.SiLU(),
            nn.Linear(action_agency_dim * 2, action_agency_dim),
            nn.LayerNorm(action_agency_dim))
        self.contract_action_agency_gain = nn.Parameter(
            torch.tensor(-2.944439))
        self.contract_layer_agency_fuser = nn.Sequential(
            nn.LayerNorm(
                ModuleDim.PstSlotDim
                + action_agency_dim
                + ModuleDim.PstLayerAgencyDim),
            nn.Linear(
                ModuleDim.PstSlotDim
                + action_agency_dim
                + ModuleDim.PstLayerAgencyDim,
                ModuleDim.PstSlotDim * 2),
            nn.SiLU(),
            nn.Linear(
                ModuleDim.PstSlotDim * 2,
                ModuleDim.PstLayerAgencyDim))
        self.contract_layer_agency_gain = nn.Parameter(
            torch.tensor(-2.944439))
        self.contract_entity_summary_fuser = nn.Sequential(
            nn.LayerNorm(ModuleDim.PstSlotDim * 2),
            nn.Linear(ModuleDim.PstSlotDim * 2, ModuleDim.PstSlotDim * 2),
            nn.SiLU(),
            nn.Linear(ModuleDim.PstSlotDim * 2, ModuleDim.PstSlotDim),
            nn.LayerNorm(ModuleDim.PstSlotDim))
        self.contract_entity_summary_gain = nn.Parameter(
            torch.tensor(-2.944439))
        self.packed_decision_decoupler = DecisionDecouplerV2(
            contractView=self.robot_contract_view,
            decisionDim=int(self.actor.belief_dim),
            worldActionDim=int(self.actor.action_embed_dim),
            planDim=int(self.actor.plan_feature_dim),
            subgoalDim=int(self.actor.subgoal_feature_dim),
            contextDim=int(self.actor.embodiment_state_feature_dim),
            constraintTokenDim=int(self.actor.constraint_token_dim))
        self.decision_decoupler = self.packed_decision_decoupler
        self.world = RSSMWorldModel(
            self.robot_contract_view,
            visionDim=ModuleDim.AttentionFeat,
            actionDim=int(self.actor.action_embed_dim),
            deterDim=ModuleDim.WorldOutHState,
            stochDim=ModuleDim.WorldOutZState,
            stateDim=ModuleDim.WorldFeat,
            ssmDim=ModuleDim.WorldOutXState,
            useMemory=True,
            globalFeatDim=ModuleDim.PerceptionFeat,
            objectTokenDim=ModuleDim.PerceptionEmbed,
            numObjectTokens=self.perc.object_token_count,
            motionPredDim=ModuleDim.PerceptionEmbed,
            integratedFeatDim=ModuleDim.PerceptionFeat,
            physicalSlots=ModuleDim.PstSlots,
            physicalSlotDim=ModuleDim.PstSlotDim,
            spatialFrameDim=ModuleDim.PstPoseDim,
            physicalAttrDim=ModuleDim.PstAttrDim,
            physicalIdDim=ModuleDim.PstIdDim,
            physicalRelDim=ModuleDim.PstRelDim,
            physicalRelationClasses=ModuleDim.PstRelationClasses,
            physicalSemanticDim=ModuleDim.PstSemanticDim,
            physicalStateDim=ModuleDim.PstStateDim,
            physicalAffordanceDim=ModuleDim.PstAffordanceDim,
            physicalTextDim=ModuleDim.PstTextDim,
            physicalSymbolDim=ModuleDim.PstSymbolClasses)
        self.world_abstract_decision_adapter = nn.Sequential(
            nn.LayerNorm(ModuleDim.WorldFeat),
            nn.Linear(ModuleDim.WorldFeat, ModuleDim.WorldFeat),
            nn.SiLU(),
            nn.Linear(
                ModuleDim.WorldFeat,
                int(self.actor.world_hzx_dim)))
        nn.init.zeros_(self.world_abstract_decision_adapter[-1].weight)
        nn.init.zeros_(self.world_abstract_decision_adapter[-1].bias)
        self.world_abstract_decision_gain = nn.Parameter(
            torch.tensor(-2.944439))
        self.critic = ValueEstimationExtractor(
            memoryDim=ModuleDim.MemoryFeat,
            attnDim=ModuleDim.AttentionFeat,
            stateDim=ModuleDim.WorldFeat,
            emotionDim=ModuleDim.ValueEstimationOutEmotion,
            valueTensorDim=self.value_tensor_dim)
        self.conscious = ConsciousnessExtractor(
            memItemDim=ModuleDim.MemoryItem,
            worldItemDim=ModuleDim.WorldMemoryItem,
            intentDim=ModuleDim.ConsciousnessState)
        self.intention = IntentionExtractor(
            dimSem=ModuleDim.IntentionFeat,
            consSelfDim=int(self.conscious.self_dim),
            consIntentDim=int(self.conscious.intent_dim),
            ocrDictPath=BasicParameters.OCR_DICT_PATH)
        self.contract_pst_builder = PhysicalStateExtractor(
            contractView=self.robot_contract_view,
            inObjectDim=ModuleDim.PerceptionEmbed)
        self.pst_builder = self.contract_pst_builder
        self.pst_loss = PhysicalStateLoss()
        world_latent_dim = (
            ModuleDim.WorldOutHState
            + ModuleDim.WorldOutZState
            + ModuleDim.WorldOutXState)
        self.goal_manager = FourLevelGoalManager(
            contractView=self.robot_contract_view,
            worldLatentDim=world_latent_dim,
            pstSummaryDim=ModuleDim.PstSlotDim,
            intentDim=ModuleDim.IntentionFeat)
        self.goal_grounding = GoalGrounding()
        self.goal_relation_adapter = nn.Linear(
            ModuleDim.IntentionFeat,
            self.goal_manager.task_relation_dim)
        self.goal_object_adapter = nn.Linear(
            ModuleDim.PstSlotDim,
            self.goal_manager.task_object_dim)
        self.goal_requirement_head = nn.Sequential(
            nn.LayerNorm(
                ModuleDim.IntentionFeat + ModuleDim.PstSlotDim),
            nn.Linear(
                ModuleDim.IntentionFeat + ModuleDim.PstSlotDim,
                4),
            nn.Sigmoid())
        self.attention_memory_top_down_adapter = nn.Sequential(
            nn.LayerNorm(ModuleDim.MemoryFeat),
            nn.Linear(ModuleDim.MemoryFeat, ModuleDim.IntentionFeat),
            nn.SiLU(),
            nn.Linear(ModuleDim.IntentionFeat, ModuleDim.IntentionFeat))
        self.attention_goal_top_down_adapter = nn.Sequential(
            nn.LayerNorm(ModuleDim.GoalShortDim),
            nn.Linear(ModuleDim.GoalShortDim, ModuleDim.IntentionFeat),
            nn.SiLU(),
            nn.Linear(ModuleDim.IntentionFeat, ModuleDim.IntentionFeat))
        nn.init.zeros_(self.attention_memory_top_down_adapter[-1].weight)
        nn.init.zeros_(self.attention_memory_top_down_adapter[-1].bias)
        nn.init.zeros_(self.attention_goal_top_down_adapter[-1].weight)
        nn.init.zeros_(self.attention_goal_top_down_adapter[-1].bias)
        self.attention_top_down_gain = nn.Parameter(
            torch.full((2,), -2.944439))
        self.option_skill_gain = nn.Parameter(torch.zeros(()))
        contract_slot_feature_dim = (
            brainBuildSpec.embodiment.end_effector_static_descriptor_dim
            + len(CONTRACT_EVIDENCE_FIELDS)
            + 2 * len(CONTRACT_SLOT_PREDICATES))
        self.contract_neuro_symbolic_grounder = (
            ContractNeuroSymbolicGrounder(
                self.robot_contract_view))
        self.contract_neuro_symbolic = NeuroSymbolicExtractor(
            contractSlotFeatureDim=contract_slot_feature_dim,
            slotDim=ModuleDim.PstSlotDim,
            goalDim=ModuleDim.GoalShortDim,
            worldDim=world_latent_dim,
            decisionDim=int(self.actor.belief_dim),
            embodimentStateDim=int(
                self.actor.body_state_feature_dim),
            controlFeedbackDim=int(
                self.actor.control_feedback_feature_dim),
            planDim=int(self.actor.plan_feature_dim),
            constraintTokenDim=int(
                self.actor.constraint_token_dim),
            constraintTokens=int(
                self.actor.constraint_token_count))
        self.neuro_symbolic = self.contract_neuro_symbolic
        self.packed_temporal_gate = TemporalExecutionGateExtractor(
            self.robot_contract_view,
            stepDurationMs=1.0)
        self.temporal_gate = self.packed_temporal_gate
        self.cognitive_compute_gate = CognitiveComputeGate(
            self.robot_contract_view,
            maxCacheAge=float(self.slow_period))
        self.OCR = OCREngineExtractor()
        self.history_len = int(BasicParameters.MEMORY_CALLBACK_LEN)
        self.ContractReplayHistory: List[deque] = []
        self.ContractReplayEpisodeVersion = torch.empty(
            0, dtype=torch.long)
        self.ContractReplayTimelineVersion = 0
        self.ContractReplayTransactionVersion = 0

        if self.is_online_learning:
            self.perc = PerceptionOnlineWrapper(self.perc)
            self.attn = AttentionOnlineWrapper(self.attn)
            self.world = WorldOnlineWrapper(self.world)
            self.critic = ValueEstimationOnlineWrapper(self.critic)
            self.intention = IntentionOnlineWrapper(self.intention)

        self.planner = None
        if self.use_planner or self.planner_teacher_mode:
            self.planner = DecisionPlannerExtractor().BuildPlanner(
                decisionDim=brainBuildSpec.cognitive.decision_dim,
                N=64,
                elite=8,
                iters=3,
                temperature=1.0,
                momentum=0.15,
                minVar=1e-4)

        self.moduleMessager = ModuleMessagerManager(maxSteps=256)
        self.save_module_messager_output = bool(
            saveModuleMessagerOutput)
        self.ContractCachedTarget: Optional[
            PackedEndEffectorTarget] = None
        self.ContractIntentionCommitmentState: Dict[
            str, torch.Tensor] = {}
        self.ContractRuntimeBatch = 0
        self.prev_visual_state = None
        self.visual_state_buffer: List[VisualState] = []
        self.visual_state_valid_buffer: List[torch.Tensor] = []
        self.perc_buffer: List[torch.Tensor] = []
        self.ResetContractCognitiveState(
            1,
            self.device,
            torch.float32)

    def SetModuleMessagerEnabled(self, enabled: bool):
            self.save_module_messager_output = bool(enabled)

    def SetTraceEnabled(self, enabled: bool):
            self.need_trace = bool(enabled)
            if not self.need_trace:
                for history in self.ContractReplayHistory:
                    history.clear()

    def RuntimeModule(self, mod: nn.Module) -> nn.Module:
            return mod.base if hasattr(mod, "base") else mod

    def ContractWorld(self) -> RSSMWorldModel:
            if bool(getattr(self, "contract_only", False)):
                return self.RuntimeModule(self.world)
            if self.contract_world_student is None:
                raise RuntimeError(
                    "brain instance is not bound to the contract World student")
            return self.contract_world_student

    def EncodeEmbodimentFeedback(
            self,
            feedbackPacket: BrainFeedbackPacket,
            *,
            batchSize: Optional[int] = None,
            device: Optional[torch.device] = None,
        ) -> Dict[str, Any]:
            if (
                self.robot_contract_view is None
                or self.contract_physical_adapter is None
            ):
                raise RuntimeError("brain instance is not bound to an embodiment contract")
            self.ValidateFeedbackPacket(
                feedbackPacket,
                batchSize=batchSize,
                device=device)
            world = self.ContractWorld()
            world_transition = world.EncodeContractEmbodiment(feedbackPacket)
            return {
                "Physical": self.contract_physical_adapter(feedbackPacket),
                "World": world_transition,
                "WorldPhysical": world.EncodeContractTransition(
                    feedbackPacket),
                "Perception": {
                    "Rotation": feedbackPacket.perception_rotation,
                    "RotationDelta": feedbackPacket.perception_rotation_delta,
                    "AngularVelocity": feedbackPacket.perception_angular_velocity,
                    "MotionPresent": feedbackPacket.perception_motion_present,
                },
            }

    def EncodeEmbodimentTransition(
            self,
            feedbackPacket: BrainFeedbackPacket,
        ) -> Dict[str, torch.Tensor]:
            self.ValidateFeedbackPacket(feedbackPacket)
            world = self.ContractWorld()
            return world.EncodeContractEmbodiment(feedbackPacket)

    def EncodeWorldContractTransition(
            self,
            feedbackPacket: BrainFeedbackPacket,
        ) -> torch.Tensor:
            self.ValidateFeedbackPacket(feedbackPacket)
            world = self.ContractWorld()
            return world.EncodeContractTransition(feedbackPacket)

    def ValidateFeedbackPacket(
            self,
            feedbackPacket: BrainFeedbackPacket,
            *,
            batchSize: Optional[int] = None,
            device: Optional[torch.device] = None,
        ) -> None:
            if self.brain_build_spec is None:
                raise RuntimeError("brain instance is not bound to BrainBuildSpec")
            self.brain_build_spec.ValidateFeedbackPacket(
                feedbackPacket,
                batchSize=batchSize,
                device=device)

    def SelectCognitiveComputeMode(
            self,
            feedbackPacket: BrainFeedbackPacket,
            **events: torch.Tensor,
        ) -> CognitiveComputeDecision:
            if self.cognitive_compute_gate is None:
                raise RuntimeError("brain instance is not bound to a compute gate")
            return self.cognitive_compute_gate(feedbackPacket, **events)

    def DecodePackedTarget(
            self,
            decisionBackbone: torch.Tensor,
            feedbackPacket: BrainFeedbackPacket,
        ) -> PackedEndEffectorTarget:
            if self.packed_decision_decoupler is None:
                raise RuntimeError("brain instance is not bound to a packed decoder")
            return self.packed_decision_decoupler(
                decisionBackbone,
                feedbackPacket=feedbackPacket)

    def ValidateContractStepInput(
            self,
            step: ContractBrainStepInput,
        ) -> None:
            if type(step) is not ContractBrainStepInput:
                raise TypeError("contract execution requires ContractBrainStepInput")
            if type(step.is_train) is not bool:
                raise TypeError("contract execution mode must be boolean")
            if (
                not torch.is_tensor(step.frame)
                or step.frame.dim() != 4
                or int(step.frame.size(0)) < 1
                or not step.frame.is_floating_point()
                or not bool(torch.isfinite(step.frame).all().item())
            ):
                raise ValueError("contract inference frame must be finite BCHW")
            batch_size = int(step.frame.size(0))
            if (
                not torch.is_tensor(step.depth)
                or tuple(step.depth.shape[:2]) != (batch_size, 1)
                or step.depth.device != step.frame.device
                or step.depth.dtype != step.frame.dtype
                or not step.depth.is_floating_point()
                or not bool(torch.isfinite(step.depth).all().item())
            ):
                raise ValueError(
                    "contract inference depth must be finite B1HW on the frame device")
            if (
                not torch.is_tensor(step.depth_valid)
                or tuple(step.depth_valid.shape) != tuple(step.depth.shape)
                or step.depth_valid.dtype != torch.bool
                or step.depth_valid.device != step.frame.device
            ):
                raise ValueError(
                    "contract inference depth validity must be a boolean depth mask")
            if bool((step.depth_valid & step.depth.le(0.0)).any().item()):
                raise ValueError("valid contract depth samples must be positive")
            if step.text_ext is not None and len(step.text_ext) != batch_size:
                raise ValueError("contract external text must match the batch")
            if step.text_trust is not None and len(step.text_trust) != batch_size:
                raise ValueError("contract text trust must match the batch")
            if type(step.perception_targets) is not dict:
                raise TypeError("contract perception targets must be a dictionary")
            for value in step.perception_targets.values():
                if (
                    not torch.is_tensor(value)
                    or value.dim() < 1
                    or int(value.size(0)) != batch_size
                    or value.device != step.frame.device
                ):
                    raise ValueError(
                        "contract perception targets must match the sensory batch")
            for name, value in (
                ("reward", step.reward_ext),
                ("done", step.done_flag),
            ):
                if value is None:
                    continue
                if (
                    not torch.is_tensor(value)
                    or value.numel() != batch_size
                    or value.device != step.frame.device
                    or not value.is_floating_point()
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise ValueError(
                        "contract " + name + " feedback must match the batch")

    def ResetContractCognitiveState(
            self,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            batch_size = int(batchSize)
            actor = self.RuntimeModule(self.actor)

            def Zeros(width: int = 0) -> torch.Tensor:
                shape = (batch_size,) if width == 0 else (batch_size, int(width))
                return torch.zeros(shape, device=device, dtype=dtype)

            self.prev_mem = Zeros(ModuleDim.MemoryFeat)
            self.prev_attn = Zeros(ModuleDim.AttentionFeat)
            self.prev_world_h = Zeros(ModuleDim.WorldOutHState)
            self.prev_world_z = Zeros(ModuleDim.WorldOutZState)
            self.prev_world_x = Zeros(ModuleDim.WorldOutXState)
            self.prev_world_s = Zeros(ModuleDim.WorldFeat)
            self.prev_world_embodiment = Zeros(
                self.ContractWorld().embodiment_state_dim)
            self.prev_done_flag = torch.ones(
                batch_size, device=device, dtype=torch.bool)
            self.prev_option_logit = Zeros(actor.num_options)
            self.prev_fast_option_logit = Zeros(actor.num_options)
            self.prev_detail_option_logit = Zeros(actor.num_options)
            option_policy_width = (
                actor.dyn_dim + actor.u_dim + actor.mapper_hidden_dim)
            self.active_option_policy_input = Zeros(option_policy_width)
            self.active_option_prior_logit = Zeros(actor.num_options)
            self.active_option_goal_mid = Zeros(ModuleDim.GoalMidDim)
            self.active_option_index = torch.zeros(
                batch_size, device=device, dtype=torch.long)
            self.active_option_valid = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            self.prev_decision_state = Zeros(actor.dyn_dim)
            self.prev_fast_decision_state = Zeros(actor.dyn_dim)
            self.prev_detail_decision_state = Zeros(actor.dyn_dim)
            self.prev_belief_prediction_state = Zeros(actor.dyn_dim)
            self.prev_belief_prediction_valid = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            self.prev_latent_control = Zeros(actor.u_dim)
            self.prev_fast_latent_control = Zeros(actor.u_dim)
            self.prev_detail_latent_control = Zeros(actor.u_dim)
            self.prev_mapper_hidden = Zeros(actor.mapper_hidden_dim)
            self.prev_fast_mapper_hidden = Zeros(actor.mapper_hidden_dim)
            self.prev_detail_mapper_hidden = Zeros(actor.mapper_hidden_dim)
            self.prev_td_error = Zeros()
            self.prev_entropy = Zeros()
            self.prev_precision = torch.ones(
                batch_size, device=device, dtype=dtype)
            self.prev_risk = Zeros()
            self.prev_world_surprise = Zeros()
            self.prev_novelty = Zeros()
            self.prev_information_gain = Zeros()
            self.prev_evc = Zeros()
            self.prev_intent_changed = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            self.prev_goal_changed = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            self.prev_goal_bias = Zeros(ModuleDim.IntentionFeat)
            self.prev_attention_goal = Zeros(ModuleDim.GoalShortDim)
            self.prev_self_sem = None
            self.prev_intent_sem = Zeros(ModuleDim.IntentionFeat)
            self.prev_failure_count = Zeros()
            self.prev_visual_state = None
            self.prev_visual_valid = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            self.prospective_visual_prediction = None
            self.perc_buffer = []
            self.visual_state_buffer = []
            self.visual_state_valid_buffer = []
            self.ContractCachedTarget = None
            self.ContractSlowCognitiveCache = None
            self.ContractSlowCacheValid = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            self.ContractReplayHistory = [
                deque(maxlen=self.history_len)
                for _ in range(batch_size)]
            self.ContractReplayEpisodeVersion = torch.zeros(
                batch_size, device=device, dtype=torch.long)
            self.ContractReplayTimelineVersion += 1
            self.ContractIntentionCommitmentState = {}
            self.ContractPreviousTextFingerprint = torch.zeros(
                batch_size, device=device, dtype=torch.long)
            self.ContractPreviousCommandFingerprint = torch.zeros(
                batch_size, device=device, dtype=torch.long)
            self.ContractCommandVersion = torch.zeros(
                batch_size, device=device, dtype=torch.long)
            goal_code_width = sum(
                int(head.groups)
                for head in (
                    self.goal_manager.ultimate_head,
                    self.goal_manager.long_head,
                    self.goal_manager.mid_head))
            self.ContractPreviousGoalCode = torch.full(
                (batch_size, goal_code_width),
                -1,
                device=device,
                dtype=torch.long)
            self.ContractPreviousReferenceIndex = torch.full(
                (batch_size,), -1, device=device, dtype=torch.long)
            self.ContractCachedActionEpoch = torch.zeros(
                batch_size, device=device, dtype=torch.long)
            self.ContractCacheAge = Zeros()
            self.ContractSlowCacheAge = Zeros()
            self.ContractPreviousTargetActive = torch.zeros(
                batch_size,
                self.robot_contract_view.end_effector_count,
                device=device,
                dtype=torch.bool)
            self.ContractPreviousProgress = torch.zeros(
                batch_size,
                self.robot_contract_view.end_effector_count,
                device=device,
                dtype=dtype)
            self.ContractObserverRotationGauge = torch.zeros(
                batch_size, 4, device=device, dtype=dtype)
            self.ContractObserverRotationGauge[:, -1] = 1.0
            self.ContractRuntimeBatch = batch_size
            self.OCR.ResetTemporal()
            if self.cognitive_compute_gate is not None:
                self.cognitive_compute_gate.Reset()
            actor.ResetHebbianMemory()
            self.ContractWorld().ResetState(batchSize=batch_size)

    def ResetContractStateRows(
            self,
            doneMask: torch.Tensor,
        ) -> None:
            state_names = (
                "prev_mem",
                "prev_attn",
                "prev_world_h",
                "prev_world_z",
                "prev_world_x",
                "prev_world_s",
                "prev_world_embodiment",
                "prev_option_logit",
                "prev_fast_option_logit",
                "prev_detail_option_logit",
                "active_option_policy_input",
                "active_option_prior_logit",
                "active_option_goal_mid",
                "prev_decision_state",
                "prev_fast_decision_state",
                "prev_detail_decision_state",
                "prev_belief_prediction_state",
                "prev_latent_control",
                "prev_fast_latent_control",
                "prev_detail_latent_control",
                "prev_mapper_hidden",
                "prev_fast_mapper_hidden",
                "prev_detail_mapper_hidden",
                "prev_td_error",
                "prev_entropy",
                "prev_goal_bias",
                "prev_attention_goal",
                "prev_intent_sem",
                "prev_failure_count",
                "prev_risk",
                "prev_world_surprise",
                "prev_novelty",
                "prev_information_gain",
                "prev_evc",
                "ContractCacheAge",
                "ContractSlowCacheAge")
            for name in state_names:
                value = getattr(self, name)
                value = value.clone()
                value[doneMask] = 0
                setattr(self, name, value)
            self.prev_precision = self.prev_precision.clone()
            self.prev_precision[doneMask] = 1.0
            self.prev_done_flag = self.prev_done_flag.clone()
            self.prev_done_flag[doneMask] = True
            self.prev_intent_changed = self.prev_intent_changed.clone()
            self.prev_intent_changed[doneMask] = False
            self.prev_goal_changed = self.prev_goal_changed.clone()
            self.prev_goal_changed[doneMask] = False
            self.prev_belief_prediction_valid = (
                self.prev_belief_prediction_valid.clone())
            self.prev_belief_prediction_valid[doneMask] = False
            self.active_option_index = self.active_option_index.clone()
            self.active_option_index[doneMask] = 0
            self.active_option_valid = self.active_option_valid.clone()
            self.active_option_valid[doneMask] = False
            self.ContractPreviousTargetActive = (
                self.ContractPreviousTargetActive.clone())
            self.ContractPreviousTargetActive[doneMask] = False
            self.ContractPreviousProgress = (
                self.ContractPreviousProgress.clone())
            self.ContractPreviousProgress[doneMask] = 0
            self.ContractPreviousTextFingerprint = (
                self.ContractPreviousTextFingerprint.clone())
            self.ContractPreviousTextFingerprint[doneMask] = 0
            self.ContractPreviousCommandFingerprint = (
                self.ContractPreviousCommandFingerprint.clone())
            self.ContractPreviousCommandFingerprint[doneMask] = 0
            self.ContractCommandVersion = self.ContractCommandVersion.clone()
            self.ContractCommandVersion[doneMask] = 0
            self.ContractPreviousGoalCode = (
                self.ContractPreviousGoalCode.clone())
            self.ContractPreviousGoalCode[doneMask] = -1
            self.ContractPreviousReferenceIndex = (
                self.ContractPreviousReferenceIndex.clone())
            self.ContractPreviousReferenceIndex[doneMask] = -1
            if self.prev_self_sem is not None:
                self.prev_self_sem = self.prev_self_sem.clone()
                self.prev_self_sem[doneMask] = 0
            reset_commitment: Dict[str, torch.Tensor] = {}
            for name, value in self.ContractIntentionCommitmentState.items():
                if torch.is_tensor(value) and value.size(0) == doneMask.size(0):
                    value = value.clone()
                    value[doneMask] = False if value.dtype == torch.bool else 0
                reset_commitment[name] = value
            self.ContractIntentionCommitmentState = reset_commitment
            self.prev_visual_state = self.ResetRuntimeRows(
                self.prev_visual_state,
                doneMask)
            self.visual_state_buffer = [
                self.ResetRuntimeRows(value, doneMask)
                for value in self.visual_state_buffer]
            self.visual_state_valid_buffer = [
                self.ResetRuntimeRows(value, doneMask)
                for value in self.visual_state_valid_buffer]
            self.perc_buffer = [
                self.ResetRuntimeRows(value, doneMask)
                for value in self.perc_buffer]
            self.prospective_visual_prediction = self.ResetRuntimeRows(
                self.prospective_visual_prediction,
                doneMask)
            self.ContractSlowCacheValid = self.ContractSlowCacheValid.clone()
            self.ContractSlowCacheValid[doneMask] = False
            for row_index in doneMask.nonzero(
                as_tuple=False).flatten().tolist():
                self.ContractReplayHistory[int(row_index)].clear()
            self.ContractReplayEpisodeVersion = (
                self.ContractReplayEpisodeVersion
                + doneMask.to(dtype=torch.long))
            if self.cognitive_compute_gate is not None:
                self.cognitive_compute_gate.ResetRows(doneMask)

    def EnsureContractCognitiveState(
            self,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            gauge = self.ContractObserverRotationGauge
            if (
                self.ContractRuntimeBatch != int(batchSize)
                or tuple(gauge.shape) != (int(batchSize), 4)
                or gauge.device != device
                or gauge.dtype != dtype
            ):
                self.ResetContractCognitiveState(
                    batchSize,
                    device,
                    dtype)

    def SelectContractPerceptionRotation(
            self,
            feedbackPacket: BrainFeedbackPacket,
            rotationDelta: Optional[torch.Tensor] = None,
            rotationPresent: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if self.robot_contract_view is None:
                raise RuntimeError("contract rotation selection requires a contract view")
            candidates = (
                feedbackPacket.perception_rotation_delta
                if rotationDelta is None
                else rotationDelta)
            present = (
                feedbackPacket.perception_motion_present
                if rotationPresent is None
                else rotationPresent)
            batch_size = int(feedbackPacket.joint_features.size(0))
            perception_count = len(
                self.robot_contract_view.perception_view_indices)
            if (
                tuple(candidates.shape) != (batch_size, perception_count, 4)
                or candidates.device != feedbackPacket.joint_features.device
                or candidates.dtype != feedbackPacket.joint_features.dtype
                or not bool(torch.isfinite(candidates).all().item())
            ):
                raise ValueError("contract perception rotations are invalid")
            if (
                tuple(present.shape) != (batch_size, perception_count)
                or present.dtype != torch.bool
                or present.device != feedbackPacket.joint_features.device
            ):
                raise ValueError("contract perception rotation presence is invalid")
            identity = feedbackPacket.joint_features.new_zeros(batch_size, 4)
            identity[:, -1] = 1.0
            if perception_count == 0:
                return identity, torch.zeros(
                    batch_size,
                    dtype=torch.bool,
                    device=feedbackPacket.joint_features.device)
            selected_index = int(
                self.robot_contract_view.primary_perception_view_index)
            selectable = present
            selected = candidates[:, selected_index]
            selected_present = selectable[:, selected_index]
            selected = torch.where(
                selected_present.unsqueeze(-1),
                NormalizeRotation(selected),
                identity)
            return selected, selected_present

    def SelectContractPerceptionMotion(
            self,
            feedbackPacket: BrainFeedbackPacket,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            selected_rotation, selected_present = (
                self.SelectContractPerceptionRotation(feedbackPacket))
            angular_velocity = feedbackPacket.perception_angular_velocity
            batch_size = int(feedbackPacket.joint_features.size(0))
            perception_count = len(
                self.robot_contract_view.perception_view_indices)
            if perception_count == 0:
                return (
                    selected_rotation,
                    feedbackPacket.joint_features.new_zeros(batch_size, 3),
                    selected_present)
            selected_index = int(
                self.robot_contract_view.primary_perception_view_index)
            selectable = feedbackPacket.perception_motion_present
            selected_velocity = angular_velocity[:, selected_index]
            selected_velocity = torch.where(
                selectable[:, selected_index].unsqueeze(-1),
                selected_velocity,
                torch.zeros_like(selected_velocity))
            return selected_rotation, selected_velocity, selected_present

    def BuildContractObserverGauge(
            self,
            rotationDelta: torch.Tensor,
            rotationPresent: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            previous = NormalizeRotation(
                self.ContractObserverRotationGauge)
            advanced = NormalizeRotation(
                ComposeRotation(previous, rotationDelta))
            current = torch.where(
                rotationPresent.unsqueeze(-1),
                advanced,
                previous)
            return previous, current

    def BuildContractActionAgencyEvidence(
            self,
            feedbackPacket: BrainFeedbackPacket,
            physicalFeedback: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            measured = self.contract_joint_motion_action_adapter(
                physicalFeedback["ControlFeedbackFeature"])
            batch_size = int(measured.size(0))
            measured_available = torch.ones(
                batch_size,
                device=measured.device,
                dtype=torch.bool)
            target_matches = torch.zeros(
                batch_size,
                device=measured.device,
                dtype=torch.bool)
            if self.ContractCachedTarget is None:
                commanded = torch.zeros_like(measured)
                commanded_valid = torch.zeros_like(measured_available)
            else:
                comparison_active = (
                    self.ContractCachedTarget.active
                    & feedbackPacket.endpoint_present)
                commanded_valid = comparison_active.any(dim=-1)
                target_matches = feedbackPacket.target_version.eq(
                    self.ContractCachedTarget.target_version)
                commanded_target = PackedEndEffectorTarget(
                    values=self.ContractCachedTarget.values,
                    active=comparison_active,
                    contract_id=self.ContractCachedTarget.contract_id,
                    model_signature=self.ContractCachedTarget.model_signature,
                    target_version=self.ContractCachedTarget.target_version,
                    timestamp=self.ContractCachedTarget.timestamp)
                commanded = self.packed_decision_decoupler.EncodeWorldAction(
                    commanded_target)
            evidence_valid = (
                measured_available
                & commanded_valid
                & target_matches)
            mismatch = (measured - commanded) * evidence_valid.to(
                dtype=measured.dtype).unsqueeze(-1)
            representation_mismatch = torch.linalg.vector_norm(
                mismatch,
                dim=-1) / math.sqrt(max(1, int(mismatch.size(-1))))
            provenance = torch.stack([
                feedbackPacket.target_active.any(dim=-1).to(
                    dtype=measured.dtype),
                measured_available.to(dtype=measured.dtype),
                target_matches.to(dtype=measured.dtype),
                representation_mismatch,
            ], dim=-1)
            evidence = self.contract_action_agency_encoder(torch.cat([
                measured,
                commanded,
                mismatch,
                provenance,
            ], dim=-1))
            evidence = evidence * evidence_valid.to(
                dtype=evidence.dtype).unsqueeze(-1)
            realized = (
                measured
                + torch.sigmoid(self.contract_action_agency_gain)
                * evidence) * measured_available.to(
                    dtype=measured.dtype).unsqueeze(-1)
            if tuple(realized.shape) != (
                batch_size,
                int(self.RuntimeModule(self.actor).action_embed_dim),
            ):
                raise RuntimeError(
                    "contract agency evidence does not match Decision action width")
            return realized, evidence, evidence_valid

    def RefineContractLayerAgency(
            self,
            physicalState: Dict[str, torch.Tensor],
            actionAgencyEvidence: torch.Tensor,
            evidenceValid: torch.Tensor,
        ) -> None:
            slot_state = physicalState["SlotState"]
            current = physicalState["LayerAgencyProb"]
            batch_size, slot_count = slot_state.shape[:2]
            if tuple(evidenceValid.shape) != (batch_size,):
                raise ValueError("contract agency validity must match the batch")
            evidence = actionAgencyEvidence.unsqueeze(1).expand(
                -1, slot_count, -1)
            delta = self.contract_layer_agency_fuser(torch.cat([
                slot_state,
                evidence,
                current.flatten(-2),
            ], dim=-1)).view_as(current)
            refined = F.softmax(
                torch.log(current.clamp_min(1e-8))
                + torch.sigmoid(self.contract_layer_agency_gain)
                * torch.tanh(delta),
                dim=-1)
            valid = (
                evidenceValid.view(batch_size, 1, 1, 1)
                & physicalState["ObservationMask"].gt(0.0).view(
                    batch_size, slot_count, 1, 1))
            physicalState["LayerAgencyProb"] = torch.where(
                valid,
                refined,
                current)
            physicalState["CausalLayerAgencyProb"] = physicalState[
                "LayerAgencyProb"]
            physicalState["CausalAgencyEvidenceValid"] = valid[
                ..., 0, 0]
            layer_weight = physicalState["MotionLayerProb"]
            layer_mass = layer_weight.sum(dim=-1, keepdim=True)
            marginal = (
                layer_weight.unsqueeze(-1)
                * physicalState["LayerAgencyProb"]
            ).sum(dim=-2) / layer_mass.clamp_min(1e-8)
            unknown = torch.zeros_like(marginal)
            unknown[..., -1] = 1.0
            marginal = torch.where(
                layer_mass > 1e-6,
                marginal,
                unknown)
            physicalState["AgencyProb"] = torch.where(
                valid.squeeze(-1),
                marginal,
                physicalState["AgencyProb"])

    def BuildTextFingerprint(
            self,
            textExt: Optional[List[Optional[str]]],
            textTrust: Optional[List[str]],
            batchSize: int,
            device: torch.device,
        ) -> torch.Tensor:
            texts = [None] * int(batchSize) if textExt is None else textExt
            trusts = [""] * int(batchSize) if textTrust is None else textTrust
            values: List[int] = []
            for text_value, trust_value in zip(texts, trusts):
                if text_value is None or not text_value.strip():
                    values.append(0)
                    continue
                payload = (
                    trust_value.strip()
                    + "\x1f"
                    + text_value.strip()).encode("utf-8")
                digest = hashlib.sha256(payload).digest()
                values.append(
                    int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
            return torch.tensor(values, device=device, dtype=torch.long)

    def ShouldRefreshSlowCognition(
            self,
            stepIsTrain: bool,
            mode: torch.Tensor,
            cacheAvailable: bool,
        ) -> bool:
            if mode.numel() != 1:
                raise ValueError("scalar compatibility requires one row")
            cache_valid = torch.full_like(
                mode, bool(cacheAvailable), dtype=torch.bool)
            return bool(self.BuildSlowRefreshMask(
                stepIsTrain,
                mode,
                cache_valid,
                torch.zeros(
                    int(mode.size(0)), 0,
                    device=mode.device,
                    dtype=torch.bool),
                torch.zeros_like(mode, dtype=torch.bool))[0].item())

    def ShouldRunPlanner(
            self,
            stepIsTrain: bool,
            mode: torch.Tensor,
        ) -> bool:
            if mode.numel() != 1:
                raise ValueError("scalar compatibility requires one row")
            return bool(self.BuildPlannerMask(
                stepIsTrain,
                mode,
                torch.zeros_like(mode, dtype=torch.bool))[0].item())

    def BuildSlowRefreshMask(
            self,
            stepIsTrain: bool,
            mode: torch.Tensor,
            cacheValid: torch.Tensor,
            activatedChildMask: torch.Tensor,
            stoppedMask: torch.Tensor,
        ) -> torch.Tensor:
            if (
                mode.dim() != 1
                or cacheValid.shape != mode.shape
                or stoppedMask.shape != mode.shape
                or activatedChildMask.dim() != 2
                or int(activatedChildMask.size(0)) != int(mode.size(0))
            ):
                raise ValueError("slow refresh masks must share the batch")
            if bool(stepIsTrain):
                return ~mode.eq(int(CognitiveComputeMode.FAILSAFE))
            hierarchy_refresh = activatedChildMask.any(dim=-1)
            return (
                mode.eq(int(CognitiveComputeMode.FULL_REPLAN))
                | hierarchy_refresh
                | ~cacheValid.to(dtype=torch.bool)
            ) & ~stoppedMask.to(dtype=torch.bool)

    def BuildPlannerMask(
            self,
            stepIsTrain: bool,
            mode: torch.Tensor,
            stoppedMask: torch.Tensor,
        ) -> torch.Tensor:
            if mode.dim() != 1 or stoppedMask.shape != mode.shape:
                raise ValueError("planner masks must share the batch")
            if bool(stepIsTrain):
                return ~mode.eq(int(CognitiveComputeMode.FAILSAFE))
            return (
                mode.eq(int(CognitiveComputeMode.FULL_REPLAN))
                & ~stoppedMask.to(dtype=torch.bool))

    def EscalateContractComputeDecision(
            self,
            decision: CognitiveComputeDecision,
            feedbackPacket: BrainFeedbackPacket,
            currentNovelty: torch.Tensor,
            worldSurprise: torch.Tensor,
            risk: torch.Tensor,
        ) -> CognitiveComputeDecision:
            batch_size = int(decision.mode.size(0))
            for name, value in (
                ("currentNovelty", currentNovelty),
                ("worldSurprise", worldSurprise),
                ("risk", risk),
            ):
                if (
                    not torch.is_tensor(value)
                    or value.reshape(-1).numel() != batch_size
                    or value.device != decision.mode.device
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise ValueError(name + " must match the compute batch")
            novelty_event = (
                currentNovelty.reshape(batch_size)
                >= float(self.cognitive_compute_gate.NoveltyThreshold))
            surprise_event = (
                worldSurprise.reshape(batch_size)
                >= float(self.cognitive_compute_gate.WorldSurpriseThreshold))
            risk_event = (
                risk.reshape(batch_size)
                >= float(self.cognitive_compute_gate.RiskThreshold))
            escalation = novelty_event | surprise_event | risk_event
            failsafe = decision.mode.eq(int(CognitiveComputeMode.FAILSAFE))
            mode = torch.where(
                escalation & ~failsafe,
                torch.full_like(
                    decision.mode,
                    int(CognitiveComputeMode.FULL_REPLAN)),
                decision.mode)
            reasons = decision.reason_target.clone()
            reasons[:, 2] = reasons[:, 2] | novelty_event | surprise_event
            reasons[:, 4] = reasons[:, 4] | risk_event
            return CognitiveComputeDecision(
                mode=mode,
                hard_trigger=decision.hard_trigger | escalation,
                activated_child_mask=decision.activated_child_mask,
                evc_trigger=decision.evc_trigger,
                reason_target=reasons)

    def BuildAttentionRefinementMask(
            self,
            stepIsTrain: bool,
            scheduledMode: torch.Tensor,
            updatedMode: torch.Tensor,
            stoppedMask: torch.Tensor,
        ) -> torch.Tensor:
            if type(stepIsTrain) is not bool:
                raise TypeError("attention refinement training flag must be boolean")
            if (
                not torch.is_tensor(scheduledMode)
                or scheduledMode.dim() != 1
                or updatedMode.shape != scheduledMode.shape
                or stoppedMask.shape != scheduledMode.shape
                or updatedMode.device != scheduledMode.device
                or stoppedMask.device != scheduledMode.device
                or stoppedMask.dtype != torch.bool
            ):
                raise ValueError("attention refinement masks must share the batch")
            if stepIsTrain:
                return torch.zeros_like(stoppedMask)
            return (
                updatedMode.eq(int(CognitiveComputeMode.FULL_REPLAN))
                & ~scheduledMode.eq(int(CognitiveComputeMode.FULL_REPLAN))
                & ~stoppedMask)

    def FuseWorldAbstractDecision(
            self,
            worldHzx: torch.Tensor,
            abstractFeature: torch.Tensor,
        ) -> torch.Tensor:
            if (
                not torch.is_tensor(worldHzx)
                or worldHzx.dim() != 2
                or not torch.is_tensor(abstractFeature)
                or abstractFeature.dim() != 2
                or abstractFeature.size(0) != worldHzx.size(0)
                or abstractFeature.device != worldHzx.device
                or abstractFeature.dtype != worldHzx.dtype
                or not bool(torch.isfinite(worldHzx).all().item())
                or not bool(torch.isfinite(abstractFeature).all().item())
            ):
                raise ValueError("world abstract decision inputs must be finite matrices")
            residual = self.world_abstract_decision_adapter(
                abstractFeature)
            if residual.shape != worldHzx.shape:
                raise RuntimeError("world abstract decision residual shape mismatch")
            return (
                worldHzx
                + torch.sigmoid(self.world_abstract_decision_gain)
                * residual)

    def RunWorldTrainingStep(
            self,
            *,
            visionIn: torch.Tensor,
            physicalState: Dict[str, torch.Tensor],
            transitionPhysicalState: Dict[str, torch.Tensor],
            actionEnc: torch.Tensor,
            embodimentState: torch.Tensor,
            transitionEmbodimentState: torch.Tensor,
            observerMotion: torch.Tensor,
            observerMotionValid: torch.Tensor,
            reward: Optional[torch.Tensor],
            done: Optional[torch.Tensor],
            sample: bool,
            commitMask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            kwargs = {
                "actionEnc": actionEnc,
                "reward": reward,
                "done": done,
                "physicalState": physicalState,
                "transitionPhysicalState": transitionPhysicalState,
                "embodimentState": embodimentState,
                "transitionEmbodimentState": transitionEmbodimentState,
                "observerMotion": observerMotion,
                "observerMotionValid": observerMotionValid,
                "sample": bool(sample),
                "commitMask": commitMask,
                "updateMemory": bool(self.training)}
            if self.is_online_learning:
                return self.world(visionIn, **kwargs)
            return self.ContractWorld().ForwardTrain(
                visionIn=visionIn,
                **kwargs)

    @staticmethod
    def ComputeTemporalKindSupervisionLoss(
            executionKindScores: torch.Tensor,
            targetKind: torch.Tensor,
            targetValid: torch.Tensor,
            activeMask: torch.Tensor,
            overrideApplied: torch.Tensor,
        ) -> torch.Tensor:
            inactive = activeMask <= 0.5
            illegal_while_inactive = (
                targetKind.eq(CONTINUE)
                | targetKind.eq(CANCEL)
                | targetKind.eq(REDISPATCH))
            learnable = (
                targetValid.to(dtype=torch.bool)
                & ~(inactive & illegal_while_inactive)
                & ~overrideApplied)
            if not bool(learnable.any().item()):
                return executionKindScores[:, DISPATCH].sum() * 0.0
            return F.cross_entropy(
                executionKindScores[learnable],
                targetKind[learnable].to(dtype=torch.long))

    def ShouldReuseOcr(
            self,
            currentNovelty: torch.Tensor,
            batchSize: int,
            stepIsTrain: bool,
        ) -> bool:
            if int(batchSize) != 1:
                raise ValueError("scalar OCR compatibility requires one row")
            return not bool(self.BuildOcrRefreshMask(
                currentNovelty,
                batchSize,
                stepIsTrain)[0].item())

    def BuildOcrRefreshMask(
            self,
            currentNovelty: torch.Tensor,
            batchSize: int,
            stepIsTrain: bool,
        ) -> torch.Tensor:
            cache = self.ContractSlowCognitiveCache
            if (
                bool(stepIsTrain)
                or cache is None
                or "OcrItems" not in cache
                or "OcrTexts" not in cache
                or currentNovelty.numel() != int(batchSize)
            ):
                return torch.ones(
                    int(batchSize),
                    device=currentNovelty.device,
                    dtype=torch.bool)
            cache_valid = getattr(
                self,
                "ContractSlowCacheValid",
                torch.ones(
                    int(batchSize),
                    device=currentNovelty.device,
                    dtype=torch.bool))
            if tuple(cache_valid.shape) != (int(batchSize),):
                raise ValueError("OCR cache validity must match the batch")
            threshold = 0.25 * float(
                self.cognitive_compute_gate.NoveltyThreshold)
            return (
                currentNovelty.reshape(int(batchSize)) > threshold
            ) | ~cache_valid.to(
                device=currentNovelty.device,
                dtype=torch.bool)

    def BuildFailsafeTemporalDecision(
            self,
            feedbackPacket: BrainFeedbackPacket,
        ) -> Any:
            if self.packed_temporal_gate is None:
                raise RuntimeError("failsafe requires a temporal execution gate")
            batch_size = int(feedbackPacket.joint_features.size(0))
            device = feedbackPacket.joint_features.device
            dtype = feedbackPacket.joint_features.dtype
            template = PackedEndEffectorTarget(
                values=torch.zeros(
                    batch_size,
                    self.robot_contract_view.end_effector_target_layout.PackedDim,
                    device=device,
                    dtype=dtype),
                active=torch.zeros(
                    batch_size,
                    self.robot_contract_view.end_effector_count,
                    device=device,
                    dtype=torch.bool),
                contract_id=self.robot_contract_view.contract_id,
                model_signature=self.robot_contract_view.model_signature,
                target_version=feedbackPacket.target_version + 1,
                timestamp=feedbackPacket.timestamp)
            execution_gate = self.packed_temporal_gate.execution_gate
            neutral = execution_gate.NeutralTarget(template)
            cached = (
                neutral
                if self.ContractCachedTarget is None
                else self.ContractCachedTarget)
            scores = torch.zeros(
                batch_size,
                len(PACKED_TEMPORAL_KIND_NAMES),
                device=device,
                dtype=dtype)
            scores[:, FAILSAFE_STOP] = 1.0
            zeros = torch.zeros(batch_size, device=device, dtype=dtype)
            proposal = PackedTemporalProposal(
                kind_scores=scores,
                same_operator=zeros,
                operator_changed=zeros,
                invoke_delta=zeros,
                reference_drift=zeros,
                redispatch_score=zeros,
                interrupt_score=torch.ones_like(zeros),
                duration_ms=zeros,
                soft_timeout_ms=zeros,
                hard_timeout_ms=zeros,
                action_epoch=self.ContractCachedActionEpoch)
            events = PackedTemporalEvent(
                cache_executing=torch.zeros(
                    batch_size, device=device, dtype=torch.bool),
                candidate_ready=torch.zeros(
                    batch_size, device=device, dtype=torch.bool),
                redispatch_requested=torch.zeros(
                    batch_size, device=device, dtype=torch.bool),
                cancel_requested=torch.ones(
                    batch_size, device=device, dtype=torch.bool),
                planner_failed=torch.ones(
                    batch_size, device=device, dtype=torch.bool),
                plan_reached=torch.zeros(
                    batch_size, device=device, dtype=torch.bool),
                hard_stop=torch.ones(
                    batch_size, device=device, dtype=torch.bool),
                active_risk=torch.ones_like(zeros),
                candidate_risk=torch.ones_like(zeros))
            return execution_gate.Step(
                feedback=feedbackPacket,
                candidateTarget=neutral,
                cachedTarget=cached,
                proposal=proposal,
                events=events,
                actionAgeSteps=self.ContractCacheAge)

    @staticmethod
    def CognitiveUtility(value: Any) -> torch.Tensor:
            return (
                value.coarseProgress
                + value.detailProgress
                + value.replanBenefit
                + value.feasibility
                + value.safetyConstraint
                - value.planStaleness
                - value.computeCost)

    @staticmethod
    def CounterfactualReplanBenefitTarget(
            newUtility: torch.Tensor,
            cachedUtility: torch.Tensor,
        ) -> torch.Tensor:
            if (
                not torch.is_tensor(newUtility)
                or not torch.is_tensor(cachedUtility)
                or newUtility.dim() != 1
                or cachedUtility.shape != newUtility.shape
                or newUtility.device != cachedUtility.device
                or newUtility.dtype != cachedUtility.dtype
                or not newUtility.is_floating_point()
                or not bool(torch.isfinite(newUtility).all().item())
                or not bool(torch.isfinite(cachedUtility).all().item())
            ):
                raise ValueError(
                    "counterfactual utilities must be finite compatible [B]")
            return (newUtility - cachedUtility).clamp_min(0.0).detach()

    @staticmethod
    def CounterfactualReplanBenefitSupervision(
            candidateUtility: torch.Tensor,
            candidateValid: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if (
                not torch.is_tensor(candidateUtility)
                or candidateUtility.dim() != 2
                or int(candidateUtility.size(1)) != 2
                or not candidateUtility.is_floating_point()
                or not bool(torch.isfinite(candidateUtility).all().item())
                or not torch.is_tensor(candidateValid)
                or candidateValid.shape != candidateUtility.shape
                or candidateValid.device != candidateUtility.device
                or candidateValid.dtype != torch.bool
            ):
                raise ValueError(
                    "counterfactual candidates must be finite [B,2] pairs")
            valid = candidateValid.all(dim=-1)
            benefit = BrainCore.CounterfactualReplanBenefitTarget(
                candidateUtility[:, 0],
                candidateUtility[:, 1])
            return torch.where(
                valid,
                benefit,
                torch.zeros_like(benefit)), valid

    @staticmethod
    def CounterfactualEvcLoss(
            replanBenefit: torch.Tensor,
            computeCost: torch.Tensor,
            benefitTarget: torch.Tensor,
            validMask: torch.Tensor,
        ) -> torch.Tensor:
            if (
                not torch.is_tensor(replanBenefit)
                or replanBenefit.dim() != 1
                or computeCost.shape != replanBenefit.shape
                or benefitTarget.shape != replanBenefit.shape
                or validMask.shape != replanBenefit.shape
                or validMask.dtype != torch.bool
                or any(
                    not torch.is_tensor(value)
                    or value.device != replanBenefit.device
                    or value.dtype != replanBenefit.dtype
                    or not value.is_floating_point()
                    or not bool(torch.isfinite(value).all().item())
                    for value in (computeCost, benefitTarget))
            ):
                raise ValueError("counterfactual EVC inputs must share [B]")
            detached_cost = computeCost.detach()
            predicted_evc = replanBenefit - detached_cost
            target_evc = benefitTarget.detach() - detached_cost
            per_row = F.smooth_l1_loss(
                predicted_evc,
                target_evc,
                reduction="none")
            weight = validMask.to(dtype=replanBenefit.dtype)
            return (per_row * weight).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def CognitiveComputeLoadTarget(
            attentionFraction: torch.Tensor,
            worldValuePasses: torch.Tensor,
            slowRefreshMask: torch.Tensor,
            fastDecisionMask: torch.Tensor,
            detailDecisionMask: torch.Tensor,
            fullDecisionMask: torch.Tensor,
            plannerMask: torch.Tensor,
            stoppedMask: torch.Tensor,
        ) -> torch.Tensor:
            if not torch.is_tensor(attentionFraction) or attentionFraction.dim() != 1:
                raise ValueError("attentionFraction must be a batched tensor")
            batch_size = int(attentionFraction.size(0))
            if (
                not attentionFraction.is_floating_point()
                or not bool(torch.isfinite(attentionFraction).all().item())
            ):
                raise ValueError("attentionFraction must be finite and floating point")
            if (
                not torch.is_tensor(worldValuePasses)
                or worldValuePasses.shape != attentionFraction.shape
                or worldValuePasses.device != attentionFraction.device
                or not worldValuePasses.is_floating_point()
                or not bool(torch.isfinite(worldValuePasses).all().item())
            ):
                raise ValueError("worldValuePasses must match attentionFraction")
            if bool((
                (attentionFraction < 0.0)
                | (attentionFraction > 2.0 + 1e-6)
                | (worldValuePasses < 0.0)
                | (worldValuePasses > 2.0 + 1e-6)
            ).any().item()):
                raise ValueError("cognitive compute load exceeds its execution budget")
            masks = (
                slowRefreshMask,
                fastDecisionMask,
                detailDecisionMask,
                fullDecisionMask,
                plannerMask,
                stoppedMask)
            if any(
                not torch.is_tensor(mask)
                or tuple(mask.shape) != (batch_size,)
                or mask.dtype != torch.bool
                or mask.device != attentionFraction.device
                for mask in masks
            ):
                raise ValueError("cognitive compute masks must be compatible boolean batches")
            route_count = (
                fastDecisionMask.to(dtype=torch.int8)
                + detailDecisionMask.to(dtype=torch.int8)
                + fullDecisionMask.to(dtype=torch.int8))
            if bool((route_count > 1).any().item()):
                raise ValueError("cognitive decision routes must be exclusive")
            dtype = attentionFraction.dtype
            decision_fraction = (
                0.25 * fastDecisionMask.to(dtype=dtype)
                + 0.5 * detailDecisionMask.to(dtype=dtype)
                + fullDecisionMask.to(dtype=dtype))
            target = (
                attentionFraction / 2.0
                + worldValuePasses / 2.0
                + slowRefreshMask.to(dtype=dtype)
                + decision_fraction
                + plannerMask.to(dtype=dtype)) / 5.0
            return torch.where(
                stoppedMask,
                torch.zeros_like(target),
                target).detach()

    @staticmethod
    def ComputeRealizedInformationGain(
            posteriorMean: torch.Tensor,
            posteriorLogStd: torch.Tensor,
            priorMean: torch.Tensor,
            priorLogStd: torch.Tensor,
        ) -> torch.Tensor:
            values = (
                posteriorMean,
                posteriorLogStd,
                priorMean,
                priorLogStd)
            if any(not torch.is_tensor(value) for value in values):
                raise TypeError("information gain inputs must be tensors")
            if posteriorMean.dim() != 2 or any(
                value.shape != posteriorMean.shape for value in values
            ):
                raise ValueError(
                    "information gain inputs must share shape [B, D]")
            if any(
                not value.is_floating_point()
                or value.device != posteriorMean.device
                or value.dtype != posteriorMean.dtype
                or not bool(torch.isfinite(value).all().item())
                for value in values
            ):
                raise ValueError(
                    "information gain inputs must be finite and compatible")
            return RSSMWorldModel.RealizedInformationGain(
                posteriorMean,
                posteriorLogStd,
                priorMean,
                priorLogStd)

    def BuildContractComputeDecision(
            self,
            feedbackPacket: BrainFeedbackPacket,
        ) -> CognitiveComputeDecision:
            if self.cognitive_compute_gate is None:
                raise RuntimeError("contract inference requires a compute gate")
            batch_size = int(feedbackPacket.joint_features.size(0))
            device = feedbackPacket.joint_features.device
            dtype = feedbackPacket.joint_features.dtype
            cache_present = (
                self.ContractCachedTarget is not None
                and self.ContractSlowCognitiveCache is not None
                and tuple(self.ContractSlowCacheValid.shape) == (batch_size,)
                and tuple(self.ContractCachedActionEpoch.shape) == (batch_size,))
            if cache_present:
                cached_active = self.ContractCachedTarget.active.any(dim=-1)
                target_matches = feedbackPacket.target_version.eq(
                    self.ContractCachedTarget.target_version)
                plan_valid = (
                    cached_active
                    & target_matches
                    & self.ContractSlowCacheValid)
            else:
                plan_valid = torch.zeros(
                    batch_size, device=device, dtype=torch.bool)
            active_plan = self.mem.RecallPlan(
                "activePlan",
                self.model_signature)
            active_plan_valid = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool)
            active_plan_age = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.long)
            if active_plan is not None:
                active_plan_feature = active_plan["feature"]
                active_plan_valid = active_plan["valid"].to(
                    device=device,
                    dtype=torch.bool)
                active_plan_age = active_plan["age"].to(
                    device=device,
                    dtype=torch.long)
                if (
                    not torch.is_tensor(active_plan_feature)
                    or active_plan_feature.dim() != 2
                    or int(active_plan_feature.size(0)) != batch_size
                    or tuple(active_plan_valid.shape) != (batch_size,)
                    or tuple(active_plan_age.shape) != (batch_size,)
                    or bool((active_plan_age < 0).any().item())
                    or not bool(torch.isfinite(
                        active_plan_feature).all().item())
                ):
                    raise RuntimeError("active plan cache is invalid")
            active_plan_fresh = (
                active_plan_valid
                & active_plan_age.lt(
                    self.cognitive_compute_gate.MaxCacheAge))
            plan_valid = plan_valid & active_plan_fresh
            compute_cache_age = torch.maximum(
                self.ContractSlowCacheAge,
                active_plan_age.to(
                    device=device,
                    dtype=self.ContractSlowCacheAge.dtype))
            target_changed = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool)
            enabled_weight = (
                feedbackPacket.child_enabled
                & feedbackPacket.endpoint_present).to(dtype=dtype)
            progress_regression = (
                self.ContractPreviousProgress
                - feedbackPacket.progress).clamp_min(0.0)
            progress_anomaly = torch.log1p((
                progress_regression * enabled_weight
            ).sum(dim=-1) / enabled_weight.sum(dim=-1).clamp_min(1.0))
            critical_invalid = (
                feedbackPacket.target_active
                & ~feedbackPacket.endpoint_present).any(dim=-1)
            reached = (
                feedbackPacket.reached
                & feedbackPacket.target_active).any(dim=-1)
            tracking_error = (
                (1.0 - feedbackPacket.progress)
                * enabled_weight).sum(dim=-1) / enabled_weight.sum(
                    dim=-1).clamp_min(1.0)
            risk = torch.maximum(
                critical_invalid.to(dtype=dtype),
                self.prev_risk)
            return self.cognitive_compute_gate(
                feedbackPacket,
                planValid=plan_valid,
                cacheAge=compute_cache_age,
                goalChanged=self.prev_goal_changed,
                intentChanged=self.prev_intent_changed,
                targetChanged=target_changed,
                worldSurprise=self.prev_world_surprise,
                risk=risk,
                novelty=self.prev_novelty,
                noveltyRelevant=self.prev_novelty.gt(0.0),
                trackingError=tracking_error,
                contactAnomaly=progress_anomaly,
                evc=self.prev_evc,
                safetyViolation=critical_invalid,
                criticalInfeasible=(
                    feedbackPacket.target_active
                    & ~feedbackPacket.child_enabled
                    & ~feedbackPacket.reached).any(dim=-1))

    def BuildContractDecisionContext(
            self,
            actOut: Dict[str, Any],
            feedbackPacket: BrainFeedbackPacket,
            risk: torch.Tensor,
            confidence: torch.Tensor,
            precision: torch.Tensor,
            slotRelevance: Optional[torch.Tensor] = None,
            slotSelectionMask: Optional[torch.Tensor] = None,
            preserveReachedTargets: Optional[torch.Tensor] = None,
        ) -> PackedDecisionContext:
            constraint_tokens = actOut["decoder_constraint_tokens"]
            slot_legal = feedbackPacket.endpoint_present
            previous_target_active = None
            if preserveReachedTargets is not None and (
                not torch.is_tensor(preserveReachedTargets)
                or tuple(preserveReachedTargets.shape)
                != (int(slot_legal.size(0)),)
                or preserveReachedTargets.dtype != torch.bool
                or preserveReachedTargets.device != slot_legal.device
            ):
                raise ValueError(
                    "preserveReachedTargets must match the decision batch")
            if self.ContractCachedTarget is not None:
                previous_target_active = self.ContractCachedTarget.active
                if preserveReachedTargets is not None:
                    previous_target_active = (
                        previous_target_active
                        & preserveReachedTargets.unsqueeze(-1))
            if preserveReachedTargets is not None:
                child_mask = torch.tensor(
                    self.robot_contract_view.child_mask,
                    device=slot_legal.device,
                    dtype=torch.bool).unsqueeze(0)
                slot_legal = slot_legal & ~(
                    ~preserveReachedTargets.unsqueeze(-1)
                    & child_mask)
            return PackedDecisionContext(
                plan_latent=actOut["decoder_plan_latent"],
                subgoal_feature=actOut["decoder_subgoal_feature"],
                context_feature=actOut["embodied_state_feature"],
                constraint_tokens=constraint_tokens,
                constraint_valid=torch.ones(
                    constraint_tokens.shape[:2],
                    device=constraint_tokens.device,
                    dtype=torch.bool),
                slot_legal=slot_legal,
                risk=risk,
                confidence=confidence,
                precision=precision,
                slot_relevance=slotRelevance,
                slot_selection_mask=slotSelectionMask,
                previous_target_values=(
                    None
                    if self.ContractCachedTarget is None
                    else self.ContractCachedTarget.values),
                previous_target_active=(
                    previous_target_active))

    def BuildOptionSkillPrior(
            self,
            currentGoal: torch.Tensor,
            cachedSkill: Optional[torch.Tensor],
            optionCount: int,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if (
                not torch.is_tensor(currentGoal)
                or currentGoal.dim() != 2
                or not currentGoal.is_floating_point()
                or not bool(torch.isfinite(currentGoal).all().item())
                or type(optionCount) is not int
                or optionCount < 1
            ):
                raise ValueError("option skill context is invalid")
            batch_size, goal_dim = currentGoal.shape
            if cachedSkill is None:
                return (
                    currentGoal.new_zeros(batch_size, optionCount),
                    currentGoal.new_zeros(batch_size))
            expected_width = goal_dim + optionCount + 1
            if (
                not torch.is_tensor(cachedSkill)
                or tuple(cachedSkill.shape)
                != (batch_size, expected_width)
                or cachedSkill.device != currentGoal.device
                or cachedSkill.dtype != currentGoal.dtype
                or not bool(torch.isfinite(cachedSkill).all().item())
            ):
                raise RuntimeError("option skill cache is invalid")
            cached_goal = cachedSkill[:, :goal_dim]
            option_evidence = cachedSkill[
                :, goal_dim:goal_dim + optionCount]
            valid = cachedSkill[:, -1].gt(0.5)
            if bool((
                (option_evidence < 0.0)
                | (option_evidence > 1.0)
            ).any().item()):
                raise RuntimeError("option skill evidence is invalid")
            similarity = F.cosine_similarity(
                currentGoal,
                cached_goal,
                dim=-1,
                eps=1e-6)
            relevance = (
                (similarity - 0.9) / 0.1
            ).clamp(0.0, 1.0) * valid.to(dtype=currentGoal.dtype)
            bias = (
                torch.tanh(self.option_skill_gain)
                * relevance.unsqueeze(-1)
                * option_evidence)
            return bias, relevance

    def ApplyFailsafeDecision(
            self,
            decision: Any,
            failsafeMask: torch.Tensor,
        ) -> Any:
            if not bool(failsafeMask.any().item()):
                return decision
            target = PackedEndEffectorTarget(
                values=torch.where(
                    failsafeMask.unsqueeze(-1),
                    torch.zeros_like(decision.target.values),
                    decision.target.values),
                active=torch.where(
                    failsafeMask.unsqueeze(-1),
                    torch.zeros_like(decision.target.active),
                    decision.target.active),
                contract_id=decision.target.contract_id,
                model_signature=decision.target.model_signature,
                target_version=decision.target.target_version,
                timestamp=decision.target.timestamp)
            values = {
                name: getattr(decision, name)
                for name in decision.__dataclass_fields__}
            values["target"] = target
            values["world_action_feature"] = torch.where(
                failsafeMask.unsqueeze(-1),
                torch.zeros_like(decision.world_action_feature),
                decision.world_action_feature)
            return type(decision)(**values)

    def ApplyStoppedTemporalDecision(
            self,
            decision: Any,
            stopMask: torch.Tensor,
        ) -> Any:
            if not bool(stopMask.any().item()):
                return decision
            selected_target = PackedEndEffectorTarget(
                values=torch.where(
                    stopMask.unsqueeze(-1),
                    torch.zeros_like(decision.selected_target.values),
                    decision.selected_target.values),
                active=torch.where(
                    stopMask.unsqueeze(-1),
                    torch.zeros_like(decision.selected_target.active),
                    decision.selected_target.active),
                contract_id=decision.selected_target.contract_id,
                model_signature=decision.selected_target.model_signature,
                target_version=decision.selected_target.target_version,
                timestamp=decision.selected_target.timestamp)
            values = {
                name: getattr(decision, name)
                for name in decision.__dataclass_fields__}
            values["selected_target"] = selected_target
            values["candidate_selected"] = (
                decision.candidate_selected & ~stopMask)
            values["cache_selected"] = decision.cache_selected & ~stopMask
            return type(decision)(**values)

    @staticmethod
    def ScheduleExecutedState(
            candidateState: torch.Tensor,
            previousState: torch.Tensor,
            candidateSelected: torch.Tensor,
            cacheSelected: torch.Tensor,
        ) -> torch.Tensor:
            if (
                not torch.is_tensor(candidateState)
                or not torch.is_tensor(previousState)
                or candidateState.shape != previousState.shape
                or candidateState.dim() < 1
                or candidateState.device != previousState.device
                or candidateState.dtype != previousState.dtype
            ):
                raise ValueError(
                    "candidate and previous policy states must share shape, device, and dtype")
            batch_size = int(candidateState.size(0))
            for name, mask in (
                ("candidateSelected", candidateSelected),
                ("cacheSelected", cacheSelected),
            ):
                if (
                    not torch.is_tensor(mask)
                    or tuple(mask.shape) != (batch_size,)
                    or mask.dtype != torch.bool
                    or mask.device != candidateState.device
                ):
                    raise ValueError(f"{name} must be a batched boolean mask")
            if bool((candidateSelected & cacheSelected).any().item()):
                raise ValueError("candidate and cached policy states are exclusive")
            candidate_mask = candidateSelected
            cache_mask = cacheSelected
            while candidate_mask.dim() < candidateState.dim():
                candidate_mask = candidate_mask.unsqueeze(-1)
                cache_mask = cache_mask.unsqueeze(-1)
            return torch.where(
                candidate_mask,
                candidateState,
                torch.where(
                    cache_mask,
                    previousState,
                    torch.zeros_like(previousState)))

    @staticmethod
    def PreserveRuntimeStateRows(
            update: torch.Tensor,
            previous: torch.Tensor,
            preserveMask: torch.Tensor,
        ) -> torch.Tensor:
            if (
                not torch.is_tensor(update)
                or not torch.is_tensor(previous)
                or update.shape != previous.shape
                or update.dim() < 1
                or update.device != previous.device
                or update.dtype != previous.dtype
                or not torch.is_tensor(preserveMask)
                or tuple(preserveMask.shape) != (update.size(0),)
                or preserveMask.dtype != torch.bool
                or preserveMask.device != update.device
            ):
                raise ValueError("runtime state preservation inputs are invalid")
            mask = preserveMask
            while mask.dim() < update.dim():
                mask = mask.unsqueeze(-1)
            return torch.where(mask, previous, update)

    @staticmethod
    def PredictCausalContractFeedback(
            world: Any,
            worldOutput: Dict[str, Any],
        ) -> Dict[str, torch.Tensor]:
            prior_binding = worldOutput.get("pst_binding_prior")
            if not isinstance(prior_binding, dict):
                raise ValueError("world output is missing its causal prior binding")
            prior_state = prior_binding.get("pst_summary_pred")
            if not torch.is_tensor(prior_state):
                raise ValueError("world causal prior binding is invalid")
            return world.PredictContractFeedback(prior_state)

    @staticmethod
    def EvaluateContractFeedbackCandidates(
            baseUtility: torch.Tensor,
            prediction: Dict[str, torch.Tensor],
            endpointActive: torch.Tensor,
            endpointAvailable: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if (
                not torch.is_tensor(baseUtility)
                or baseUtility.dim() != 1
                or not baseUtility.is_floating_point()
                or not bool(torch.isfinite(baseUtility).all().item())
            ):
                raise ValueError("candidate base utility must be finite [C]")
            candidate_count = int(baseUtility.size(0))
            if (
                not torch.is_tensor(endpointActive)
                or endpointActive.dim() != 2
                or int(endpointActive.size(0)) != candidate_count
                or endpointActive.dtype != torch.bool
                or endpointActive.device != baseUtility.device
                or not torch.is_tensor(endpointAvailable)
                or tuple(endpointAvailable.shape) != tuple(endpointActive.shape)
                or endpointAvailable.dtype != torch.bool
                or endpointAvailable.device != baseUtility.device
            ):
                raise ValueError("candidate endpoint masks must be compatible")
            required = (
                "Progress",
                "ReachedLogits",
                "LatentRisk",
                "LatentFeasibility")
            for name in required:
                value = prediction.get(name)
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != tuple(endpointActive.shape)
                    or value.device != baseUtility.device
                    or value.dtype != baseUtility.dtype
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise ValueError("candidate feedback prediction is invalid")
            for name in ("Progress", "LatentRisk", "LatentFeasibility"):
                if bool((
                    (prediction[name] < 0.0)
                    | (prediction[name] > 1.0)
                ).any().item()):
                    raise ValueError("candidate feedback prediction is invalid")
            weight = endpointActive.to(dtype=baseUtility.dtype)
            normalizer = weight.sum(dim=-1).clamp_min(1.0)
            operational_utility = (
                (
                    prediction["Progress"]
                    + torch.sigmoid(prediction["ReachedLogits"])
                    + prediction["LatentFeasibility"]
                    - prediction["LatentRisk"])
                * weight).sum(dim=-1) / normalizer
            score = baseUtility + 0.25 * operational_utility
            contract_violation = (
                endpointActive
                & ~endpointAvailable
            ).any(dim=-1)
            score = torch.where(
                contract_violation,
                torch.full_like(score, -10000.0),
                score)
            return score, ~contract_violation

    @staticmethod
    def ScoreContractFeedbackCandidates(
            baseUtility: torch.Tensor,
            prediction: Dict[str, torch.Tensor],
            endpointActive: torch.Tensor,
            endpointAvailable: torch.Tensor,
        ) -> torch.Tensor:
            score, _ = BrainCore.EvaluateContractFeedbackCandidates(
                baseUtility,
                prediction,
                endpointActive,
                endpointAvailable)
            return score

    def BuildContractDecisionFeatureEvaluator(
            self,
            *,
            h0: torch.Tensor,
            z0: torch.Tensor,
            x0: torch.Tensor,
            physicalState: Dict[str, torch.Tensor],
            embodimentState: torch.Tensor,
            feedbackPacket: BrainFeedbackPacket,
            decisionContext: PackedDecisionContext,
            activePerceptionRequirement: torch.Tensor,
        ) -> Callable[
            [torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor]]:
            if self.packed_decision_decoupler is None:
                raise RuntimeError("contract decision evaluator requires a packed decoder")
            batch_size = int(h0.size(0))
            if (
                not torch.is_tensor(activePerceptionRequirement)
                or tuple(activePerceptionRequirement.shape) != (batch_size,)
                or not activePerceptionRequirement.is_floating_point()
                or activePerceptionRequirement.device != h0.device
                or not bool(torch.isfinite(
                    activePerceptionRequirement).all().item())
            ):
                raise ValueError(
                    "active perception requirement must be finite [B]")
            world = self.ContractWorld()

            def Evaluate(
                candidateDecisionFeature: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                if (
                    not torch.is_tensor(candidateDecisionFeature)
                    or candidateDecisionFeature.dim() != 2
                    or int(candidateDecisionFeature.size(0)) % batch_size != 0
                ):
                    raise ValueError(
                        "contract decision candidates do not match the rollout batch")
                candidate_count = (
                    int(candidateDecisionFeature.size(0)) // batch_size)

                def ExpandCandidates(value: torch.Tensor) -> torch.Tensor:
                    return value.unsqueeze(1).expand(
                        batch_size,
                        candidate_count,
                        *value.shape[1:]).reshape(
                            batch_size * candidate_count,
                            *value.shape[1:]).contiguous()

                candidate_packet = feedbackPacket.RepeatCandidates(
                    candidate_count)
                candidate_context = PackedDecisionContext(
                    plan_latent=ExpandCandidates(
                        decisionContext.plan_latent),
                    subgoal_feature=ExpandCandidates(
                        decisionContext.subgoal_feature),
                    context_feature=ExpandCandidates(
                        decisionContext.context_feature),
                    constraint_tokens=ExpandCandidates(
                        decisionContext.constraint_tokens),
                    constraint_valid=ExpandCandidates(
                        decisionContext.constraint_valid),
                    slot_legal=ExpandCandidates(
                        decisionContext.slot_legal),
                    risk=ExpandCandidates(decisionContext.risk),
                    confidence=ExpandCandidates(decisionContext.confidence),
                    precision=ExpandCandidates(decisionContext.precision),
                    slot_relevance=(
                        None
                        if decisionContext.slot_relevance is None
                        else ExpandCandidates(
                            decisionContext.slot_relevance)),
                    slot_selection_mask=(
                        None
                        if decisionContext.slot_selection_mask is None
                        else ExpandCandidates(
                            decisionContext.slot_selection_mask)),
                    previous_target_values=(
                        None
                        if decisionContext.previous_target_values is None
                        else ExpandCandidates(
                            decisionContext.previous_target_values)),
                    previous_target_active=(
                        None
                        if decisionContext.previous_target_active is None
                        else ExpandCandidates(
                            decisionContext.previous_target_active)))
                decoded = self.packed_decision_decoupler.DecodeContract(
                    candidateDecisionFeature,
                    feedbackPacket=candidate_packet,
                    decisionContext=candidate_context)
                action = decoded.world_action_feature
                efference = (
                    self.packed_decision_decoupler
                    .DecodePerceptionRotationEfference(
                        decoded.target,
                        candidate_packet))
                observer_motion, observer_valid = (
                    self.SelectContractPerceptionRotation(
                        candidate_packet,
                        rotationDelta=efference.rotation_delta,
                        rotationPresent=efference.present))
                expanded_physical_state = {
                    name: ExpandCandidates(value)
                    for name, value in physicalState.items()
                }
                prior = world.StepPriorOnly(
                    ExpandCandidates(h0),
                    ExpandCandidates(z0),
                    ExpandCandidates(x0),
                    action,
                    physicalState=expanded_physical_state,
                    embodimentState=ExpandCandidates(
                        embodimentState),
                    observerMotion=observer_motion,
                    observerMotionValid=observer_valid,
                    sample=False)
                active_perception = ExpandCandidates(
                    activePerceptionRequirement).clamp(0.0, 1.0)
                epistemic_value = torch.log1p(
                    prior["information_gain_pred"])
                base_utility = (
                    prior["r_pred"]
                    - prior["d_prob"]
                    + 0.1 * active_perception * epistemic_value)
                predicted_feedback = world.PredictContractFeedback(
                    prior["pst_binding"]["pst_summary_pred"])
                candidate_score, candidate_valid = (
                    self.EvaluateContractFeedbackCandidates(
                        base_utility,
                        predicted_feedback,
                        decoded.target.active,
                        candidate_packet.endpoint_present
                        & candidate_packet.child_enabled))
                return candidate_score, candidate_valid

            return Evaluate

    def PredictPackedFeedback(
            self,
            priorWorldState: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            if self.robot_contract_view is None:
                raise RuntimeError("brain instance is not bound to a world adapter")
            world = self.ContractWorld()
            return world.PredictContractFeedback(priorWorldState)

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
            self.RuntimeModule(self.perc).ResetHebbianMemory(doneMask=doneMask)
            self.RuntimeModule(self.attn).ResetHebbianMemory(doneMask=doneMask)
            self.RuntimeModule(self.actor).ResetHebbianMemory(doneMask=doneMask)
            self.RuntimeModule(self.critic).ResetHebbianMemory(doneMask=doneMask)
            self.RuntimeModule(self.mem).ResetHebbianMemory(doneMask=doneMask)
            self.RuntimeModule(self.conscious).ResetHebbianMemory(doneMask=doneMask)

    def BuildTaskRelevanceQuery(self) -> torch.Tensor:
            intention = self.prev_goal_bias
            goal = self.prev_attention_goal
            if (
                intention.dim() != 2
                or goal.dim() != 2
                or goal.size(0) != intention.size(0)
                or goal.device != intention.device
                or goal.dtype != intention.dtype
                or not bool(torch.isfinite(intention).all().item())
                or not bool(torch.isfinite(goal).all().item())
            ):
                raise ValueError("task relevance state is invalid")
            gains = torch.sigmoid(self.attention_top_down_gain).to(
                device=intention.device,
                dtype=intention.dtype)
            return (
                intention
                + gains[1] * self.attention_goal_top_down_adapter(goal))

    def BuildAttentionTopDownBias(self) -> torch.Tensor:
            task_relevance = self.BuildTaskRelevanceQuery()
            memory = self.prev_mem
            if (
                memory.dim() != 2
                or memory.size(0) != task_relevance.size(0)
                or memory.device != task_relevance.device
                or memory.dtype != task_relevance.dtype
                or not bool(torch.isfinite(memory).all().item())
            ):
                raise ValueError("attention top-down state is invalid")
            gain = torch.sigmoid(self.attention_top_down_gain[0]).to(
                device=task_relevance.device,
                dtype=task_relevance.dtype)
            return (
                task_relevance
                + gain * self.attention_memory_top_down_adapter(memory))

    @staticmethod
    def AggregateTaskRelevantObjectNovelty(
            currentObjects: torch.Tensor,
            predictedObjects: torch.Tensor,
            currentPresence: torch.Tensor,
            predictedPresence: torch.Tensor,
            taskScores: torch.Tensor,
            taskCueValid: torch.Tensor,
        ) -> torch.Tensor:
            if (
                not torch.is_tensor(currentObjects)
                or currentObjects.dim() != 3
                or not currentObjects.is_floating_point()
                or not torch.is_tensor(predictedObjects)
                or predictedObjects.dim() != 3
                or int(predictedObjects.size(0))
                != int(currentObjects.size(0))
                or int(predictedObjects.size(2))
                != int(currentObjects.size(2))
                or predictedObjects.device != currentObjects.device
                or predictedObjects.dtype != currentObjects.dtype
                or int(currentObjects.size(1)) < 1
                or int(predictedObjects.size(1)) < 1
                or not bool(torch.isfinite(currentObjects).all().item())
                or not bool(torch.isfinite(predictedObjects).all().item())
            ):
                raise ValueError("object novelty tensors are invalid")
            batch_size = int(currentObjects.size(0))
            current_count = int(currentObjects.size(1))
            predicted_count = int(predictedObjects.size(1))
            expected = (
                (currentPresence, (batch_size, current_count)),
                (predictedPresence, (batch_size, predicted_count)),
                (taskScores, (batch_size, current_count)),
            )
            if any(
                not torch.is_tensor(value)
                or tuple(value.shape) != shape
                or value.device != currentObjects.device
                or value.dtype != currentObjects.dtype
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
                for value, shape in expected
            ) or (
                not torch.is_tensor(taskCueValid)
                or tuple(taskCueValid.shape) != (batch_size,)
                or taskCueValid.device != currentObjects.device
                or taskCueValid.dtype != torch.bool
            ):
                raise ValueError("object novelty evidence is invalid")
            current_normalized = F.normalize(
                currentObjects, dim=-1, eps=1e-6)
            predicted_normalized = F.normalize(
                predictedObjects, dim=-1, eps=1e-6)
            similarity = torch.einsum(
                "bkd,bjd->bkj",
                current_normalized,
                predicted_normalized).clamp(-1.0, 1.0)
            relative_prediction_presence = (
                predictedPresence.clamp(0.0, 1.0)
                / predictedPresence.clamp(0.0, 1.0).amax(
                    dim=-1,
                    keepdim=True).clamp_min(1e-6))
            prediction_support = (
                0.5 * (similarity + 1.0)
                * relative_prediction_presence.unsqueeze(1))
            object_error = 1.0 - prediction_support.max(dim=-1).values
            presence = currentPresence.clamp(0.0, 1.0)
            relevance = torch.softmax(taskScores, dim=-1) * presence
            normalizer = relevance.sum(dim=-1)
            novelty = (
                object_error * relevance
            ).sum(dim=-1) / normalizer.clamp_min(1e-6)
            valid = taskCueValid & normalizer.gt(1e-6)
            return torch.where(valid, novelty, torch.zeros_like(novelty))

    def BuildTaskRelevantNovelty(
            self,
            visualState: VisualState,
            taskQuery: torch.Tensor,
        ) -> torch.Tensor:
            current_objects = visualState.ObjectTokens
            batch_size = int(current_objects.size(0))
            if (
                not torch.is_tensor(taskQuery)
                or taskQuery.dim() != 2
                or int(taskQuery.size(0)) != batch_size
                or taskQuery.device != current_objects.device
                or taskQuery.dtype != current_objects.dtype
                or not bool(torch.isfinite(taskQuery).all().item())
            ):
                raise ValueError("task relevance query is invalid")
            prior = self.prospective_visual_prediction
            if prior is None:
                return current_objects.new_zeros(batch_size)
            if (
                type(prior) is not dict
                or type(prior.get("reconstructed_visual_state")) is not dict
            ):
                raise ValueError("task relevance visual prior is invalid")
            predicted = prior["reconstructed_visual_state"]
            predicted_objects = predicted.get("ObjectTokens")
            predicted_presence_logits = predicted.get("SlotPresenceLogits")
            node_logits = visualState.SemanticNodes.get("node_logits")
            if (
                not torch.is_tensor(node_logits)
                or tuple(node_logits.shape[:2])
                != tuple(current_objects.shape[:2])
                or int(node_logits.size(-1)) != 2
                or node_logits.device != current_objects.device
                or node_logits.dtype != current_objects.dtype
                or not torch.is_tensor(predicted_objects)
                or not torch.is_tensor(predicted_presence_logits)
            ):
                raise ValueError("task relevance object evidence is invalid")
            attention = self.RuntimeModule(self.attn)
            current_keys = attention.object_pool_key(
                attention.object_pool_norm(current_objects))
            query = attention.goal_object_query(taskQuery)
            task_scores = torch.einsum(
                "bd,bkd->bk",
                query,
                current_keys) / math.sqrt(float(attention.structured_dim))
            return self.AggregateTaskRelevantObjectNovelty(
                current_objects,
                predicted_objects.detach(),
                F.softmax(node_logits, dim=-1)[..., 1],
                torch.sigmoid(predicted_presence_logits.detach()),
                task_scores,
                (
                    self.prev_goal_bias.square().mean(dim=-1).gt(1e-12)
                    | self.prev_attention_goal.square().mean(
                        dim=-1).gt(1e-12)))

    def ResetRuntimeRows(self, obj: Any, doneMask: torch.Tensor) -> Any:
            if torch.is_tensor(obj):
                if obj.dim() == 0 or int(obj.size(0)) != int(doneMask.size(0)):
                    return obj
                value = obj.clone()
                value[doneMask] = False if value.dtype == torch.bool else 0
                return value
            if isinstance(obj, dict):
                return {
                    name: (
                        value
                        if name == "PatchGridShape"
                        else self.ResetRuntimeRows(value, doneMask))
                    for name, value in obj.items()}
            if isinstance(obj, list):
                return [
                    self.ResetRuntimeRows(value, doneMask)
                    for value in obj]
            if isinstance(obj, tuple) and hasattr(obj, "_fields"):
                return type(obj)(*(
                    self.ResetRuntimeRows(value, doneMask)
                    for value in obj))
            if isinstance(obj, tuple):
                return tuple(
                    self.ResetRuntimeRows(value, doneMask)
                    for value in obj)
            if isinstance(obj, deque):
                return deque((
                    self.ResetRuntimeRows(value, doneMask)
                    for value in obj), maxlen=obj.maxlen)
            if hasattr(obj, "__dataclass_fields__"):
                return type(obj)(**{
                    name: self.ResetRuntimeRows(
                        getattr(obj, name),
                        doneMask)
                    for name in obj.__dataclass_fields__})
            return obj

    def DetachVisualState(self, state: Optional[VisualState], *, clone: bool = False) -> Optional[VisualState]:
            if state is None:
                return None

            def DetachTensor(t: torch.Tensor) -> torch.Tensor:
                out = t.detach()
                return out.clone() if clone else out

            return VisualState(
                IntegratedFeat=DetachTensor(state.IntegratedFeat),
                GlobalFeat=DetachTensor(state.GlobalFeat),
                VentralFeat=DetachTensor(state.VentralFeat),
                DorsalFeat=DetachTensor(state.DorsalFeat),
                MotionToken=DetachTensor(state.MotionToken),
                QualityToken=DetachTensor(state.QualityToken),
                PredErrorToken=DetachTensor(state.PredErrorToken),
                ObjectTokens=DetachTensor(state.ObjectTokens),
                PatchTokens=DetachTensor(state.PatchTokens),
                SemanticNodes={k: DetachTensor(v) for k, v in state.SemanticNodes.items()},
                Auxiliary={k: DetachTensor(v) for k, v in state.Auxiliary.items() if isinstance(v, torch.Tensor)},)

    def DetachRuntimeObject(self, obj: Any, *, clone: bool = False) -> Any:
            if isinstance(obj, torch.Tensor):
                out = obj.detach()
                return out.clone() if clone else out
            if isinstance(obj, dict):
                return {k: self.DetachRuntimeObject(v, clone=clone) for k, v in obj.items()}
            if isinstance(obj, list):
                return [self.DetachRuntimeObject(v, clone=clone) for v in obj]
            if isinstance(obj, tuple) and hasattr(obj, "_fields"):
                return type(obj)(*(
                    self.DetachRuntimeObject(v, clone=clone)
                    for v in obj))
            if isinstance(obj, tuple):
                return tuple(self.DetachRuntimeObject(v, clone=clone) for v in obj)
            if isinstance(obj, deque):
                return deque((self.DetachRuntimeObject(v, clone=clone) for v in obj), maxlen=obj.maxlen)
            if hasattr(obj, "__dataclass_fields__"):
                vals = {
                    name: self.DetachRuntimeObject(getattr(obj, name), clone=clone)
                    for name in obj.__dataclass_fields__.keys()}
                return type(obj)(**vals)
            return obj

    def IndexRuntimeRows(
            self,
            obj: Any,
            rowIndices: torch.Tensor,
            batchSize: int,
        ) -> Any:
            if rowIndices.dim() != 1 or rowIndices.dtype != torch.long:
                raise ValueError("row indices must be a one-dimensional long tensor")
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and int(obj.size(0)) == int(batchSize):
                    return obj.index_select(0, rowIndices.to(device=obj.device))
                return obj
            if isinstance(obj, dict):
                return {
                    name: (
                        copy.deepcopy(value)
                        if name == "PatchGridShape"
                        else self.IndexRuntimeRows(
                            value, rowIndices, batchSize))
                    for name, value in obj.items()}
            if isinstance(obj, list):
                if len(obj) == int(batchSize):
                    return [
                        copy.deepcopy(obj[int(index)])
                        for index in rowIndices.tolist()]
                return [
                    self.IndexRuntimeRows(value, rowIndices, batchSize)
                    for value in obj]
            if isinstance(obj, tuple) and hasattr(obj, "_fields"):
                return type(obj)(*(
                    self.IndexRuntimeRows(value, rowIndices, batchSize)
                    for value in obj))
            if isinstance(obj, tuple):
                return tuple(
                    self.IndexRuntimeRows(value, rowIndices, batchSize)
                    for value in obj)
            if hasattr(obj, "__dataclass_fields__"):
                return type(obj)(**{
                    name: self.IndexRuntimeRows(
                        getattr(obj, name), rowIndices, batchSize)
                    for name in obj.__dataclass_fields__})
            return copy.deepcopy(obj)

    def ExpandRuntimeRows(
            self,
            obj: Any,
            rowIndices: torch.Tensor,
            batchSize: int,
        ) -> Any:
            selected_size = int(rowIndices.numel())
            if isinstance(obj, torch.Tensor):
                if obj.dim() > 0 and int(obj.size(0)) == selected_size:
                    expanded = obj.new_zeros((int(batchSize),) + tuple(obj.shape[1:]))
                    expanded.index_copy_(
                        0,
                        rowIndices.to(device=expanded.device),
                        obj)
                    return expanded
                return obj
            if isinstance(obj, dict):
                return {
                    name: (
                        copy.deepcopy(value)
                        if name == "PatchGridShape"
                        else self.ExpandRuntimeRows(
                            value, rowIndices, batchSize))
                    for name, value in obj.items()}
            if isinstance(obj, list):
                if (
                    len(obj) == selected_size
                    and not any(
                        hasattr(value, "__dataclass_fields__")
                        for value in obj)
                ):
                    expanded_list: List[Any] = [None] * int(batchSize)
                    for source, target in enumerate(rowIndices.tolist()):
                        expanded_list[int(target)] = copy.deepcopy(obj[source])
                    return expanded_list
                return [
                    self.ExpandRuntimeRows(value, rowIndices, batchSize)
                    for value in obj]
            if isinstance(obj, tuple) and hasattr(obj, "_fields"):
                return type(obj)(*(
                    self.ExpandRuntimeRows(value, rowIndices, batchSize)
                    for value in obj))
            if isinstance(obj, tuple):
                return tuple(
                    self.ExpandRuntimeRows(value, rowIndices, batchSize)
                    for value in obj)
            if hasattr(obj, "__dataclass_fields__"):
                return type(obj)(**{
                    name: self.ExpandRuntimeRows(
                        getattr(obj, name), rowIndices, batchSize)
                    for name in obj.__dataclass_fields__})
            return copy.deepcopy(obj)

    def ScatterRuntimeRows(
            self,
            base: Any,
            update: Any,
            rowIndices: torch.Tensor,
            batchSize: int,
        ) -> Any:
            if base is None:
                return self.ExpandRuntimeRows(update, rowIndices, batchSize)
            selected_size = int(rowIndices.numel())
            if isinstance(update, torch.Tensor):
                if (
                    isinstance(base, torch.Tensor)
                    and base.dim() > 0
                    and update.dim() > 0
                    and int(base.size(0)) == int(batchSize)
                    and int(update.size(0)) == selected_size
                    and tuple(base.shape[1:]) == tuple(update.shape[1:])
                ):
                    result = base.clone()
                    result.index_copy_(
                        0,
                        rowIndices.to(device=result.device),
                        update.to(device=result.device, dtype=result.dtype))
                    return result
                return update
            if isinstance(update, dict):
                base_dict = base if isinstance(base, dict) else {}
                result_dict = copy.deepcopy(base_dict)
                for name, value in update.items():
                    result_dict[name] = (
                        copy.deepcopy(value)
                        if name == "PatchGridShape"
                        else self.ScatterRuntimeRows(
                            base_dict.get(name),
                            value,
                            rowIndices,
                            batchSize))
                return result_dict
            if isinstance(update, list):
                if (
                    len(update) == selected_size
                    and isinstance(base, list)
                    and len(base) == int(batchSize)
                    and not any(
                        hasattr(value, "__dataclass_fields__")
                        for value in update)
                ):
                    result_list = copy.deepcopy(base)
                    for source, target in enumerate(rowIndices.tolist()):
                        result_list[int(target)] = copy.deepcopy(update[source])
                    return result_list
                if isinstance(base, list) and len(base) == len(update):
                    return [
                        self.ScatterRuntimeRows(
                            base_value,
                            update_value,
                            rowIndices,
                            batchSize)
                        for base_value, update_value in zip(base, update)]
                return copy.deepcopy(update)
            if isinstance(update, tuple) and hasattr(update, "_fields"):
                base_values = (
                    tuple(base)
                    if isinstance(base, tuple) and len(base) == len(update)
                    else (None,) * len(update))
                return type(update)(*(
                    self.ScatterRuntimeRows(
                        base_value,
                        update_value,
                        rowIndices,
                        batchSize)
                    for base_value, update_value in zip(
                        base_values, update)))
            if isinstance(update, tuple):
                base_values = (
                    tuple(base)
                    if isinstance(base, tuple) and len(base) == len(update)
                    else (None,) * len(update))
                return tuple(
                    self.ScatterRuntimeRows(
                        base_value,
                        update_value,
                        rowIndices,
                        batchSize)
                    for base_value, update_value in zip(
                        base_values, update))
            if hasattr(update, "__dataclass_fields__"):
                return type(update)(**{
                    name: self.ScatterRuntimeRows(
                        None if base is None else getattr(base, name),
                        getattr(update, name),
                        rowIndices,
                        batchSize)
                    for name in update.__dataclass_fields__})
            return copy.deepcopy(update)

    def MergeVisualHistory(
            self,
            previousStates: List[VisualState],
            previousValid: List[torch.Tensor],
            candidateStates: List[VisualState],
            candidateValid: List[torch.Tensor],
            preserveMask: torch.Tensor,
        ) -> Tuple[List[VisualState], List[torch.Tensor]]:
            if (
                len(previousStates) != len(previousValid)
                or len(candidateStates) != len(candidateValid)
                or len(candidateStates) < len(previousStates)
                or preserveMask.dim() != 1
                or preserveMask.dtype != torch.bool
            ):
                raise ValueError("visual history state is invalid")
            batch_size = int(preserveMask.numel())
            active_rows = (~preserveMask).nonzero(
                as_tuple=False).flatten()
            merged_states: List[VisualState] = []
            merged_valid: List[torch.Tensor] = []
            for index, candidate_state in enumerate(candidateStates):
                candidate_valid = candidateValid[index]
                if (
                    tuple(candidate_valid.shape) != (batch_size,)
                    or candidate_valid.dtype != torch.bool
                    or candidate_valid.device != preserveMask.device
                ):
                    raise ValueError("visual history validity is invalid")
                if index < len(previousStates):
                    base_state = previousStates[index]
                    base_valid = previousValid[index]
                else:
                    base_state = candidate_state
                    base_valid = torch.zeros_like(candidate_valid)
                active_state = self.IndexRuntimeRows(
                    candidate_state,
                    active_rows,
                    batch_size)
                merged_state = self.ScatterRuntimeRows(
                    base_state,
                    active_state,
                    active_rows,
                    batch_size)
                valid = base_valid.detach().clone()
                valid.index_copy_(
                    0,
                    active_rows,
                    candidate_valid.index_select(0, active_rows))
                merged_states.append(
                    self.DetachVisualState(merged_state))
                merged_valid.append(valid)
            return merged_states, merged_valid

    @torch.no_grad()
    def CorrectContractReplayFeedback(
            self,
            rewardObservation: Optional[torch.Tensor],
            doneObservation: Optional[torch.Tensor],
            *,
            isTrain: bool,
            feedbackTimestamp: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            batch_size = int(self.ContractRuntimeBatch)
            device = self.prev_mem.device
            dtype = self.prev_mem.dtype
            corrected_rows = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            correction_energy = torch.zeros(
                batch_size, device=device, dtype=dtype)
            reconsolidated = torch.zeros(
                batch_size, device=device, dtype=torch.bool)
            if (
                not self.need_trace
                or (rewardObservation is None and doneObservation is None)
            ):
                return {
                    "corrected_rows": corrected_rows,
                    "correction_energy": correction_energy,
                    "reconsolidated": reconsolidated}
            reward_observation = None
            if rewardObservation is not None:
                if (
                    not torch.is_tensor(rewardObservation)
                    or not rewardObservation.is_floating_point()
                    or rewardObservation.reshape(-1).numel() != batch_size
                    or not bool(torch.isfinite(rewardObservation).all().item())
                ):
                    raise ValueError(
                        "replay reward observation must be finite [B]")
                reward_observation = rewardObservation.reshape(
                    batch_size).to(device=device, dtype=dtype)
            done_observation = None
            if doneObservation is not None:
                if (
                    not torch.is_tensor(doneObservation)
                    or not doneObservation.is_floating_point()
                    or doneObservation.reshape(-1).numel() != batch_size
                    or not bool(torch.isfinite(doneObservation).all().item())
                    or bool(((doneObservation < 0.0)
                             | (doneObservation > 1.0)).any().item())
                ):
                    raise ValueError(
                        "replay done observation must be finite probabilities [B]")
                done_observation = doneObservation.reshape(
                    batch_size).to(device=device, dtype=dtype)
            feedback_timestamp = None
            if feedbackTimestamp is not None:
                if (
                    not torch.is_tensor(feedbackTimestamp)
                    or not feedbackTimestamp.is_floating_point()
                    or feedbackTimestamp.reshape(-1).numel() != batch_size
                    or not bool(torch.isfinite(feedbackTimestamp).all().item())
                ):
                    raise ValueError(
                        "replay feedback timestamp must be finite [B]")
                feedback_timestamp = feedbackTimestamp.reshape(
                    batch_size).to(device=device, dtype=torch.float64)
            value = self.RuntimeModule(self.critic)
            recon_query = torch.zeros_like(self.prev_mem)
            recon_outcome = torch.zeros(
                batch_size,
                self.mem.COUNTERFACTUAL_OUTCOME_DIM,
                device=device,
                dtype=dtype)
            recon_confidence = torch.zeros(
                batch_size, device=device, dtype=dtype)
            machine_epsilon = torch.finfo(dtype).eps
            scale_epsilon = machine_epsilon ** 0.5
            saturation_energy = -2.0 * math.log(torch.finfo(dtype).tiny)
            saturation_innovation = saturation_energy ** 0.5
            for row_index, history in enumerate(self.ContractReplayHistory):
                if not history:
                    continue
                traces = list(history)
                predicted_reward = torch.stack([
                    trace.predicted_reward
                    for trace in traces]).to(device=device, dtype=dtype)
                predicted_done_raw = torch.stack([
                    trace.predicted_done
                    for trace in traces]).to(device=device, dtype=dtype)
                trace_confidence = torch.stack([
                    trace.confidence
                    for trace in traces]).to(device=device, dtype=dtype)
                trace_uncertainty = torch.stack([
                    trace.uncertainty
                    for trace in traces]).to(device=device, dtype=dtype)
                if (
                    not bool(torch.isfinite(predicted_reward).all().item())
                    or not bool(torch.isfinite(predicted_done_raw).all().item())
                    or not bool(torch.isfinite(trace_confidence).all().item())
                    or not bool(torch.isfinite(trace_uncertainty).all().item())
                    or bool(((predicted_done_raw < 0.0)
                             | (predicted_done_raw > 1.0)).any().item())
                ):
                    raise ValueError("replay trace statistics are invalid")
                predicted_done = predicted_done_raw.clamp(0.0, 1.0)
                corrected_reward = predicted_reward
                corrected_done = predicted_done
                reward_variance = torch.zeros_like(predicted_reward)
                done_variance = torch.zeros_like(predicted_done)
                reward_innovation = torch.zeros_like(predicted_reward)
                done_innovation = torch.zeros_like(predicted_done)
                reward_noise = torch.zeros_like(predicted_reward)
                done_noise = torch.zeros_like(predicted_done)
                posterior_timestamps = None
                if feedback_timestamp is not None:
                    posterior_timestamps = torch.cat([
                        torch.stack([
                            trace.timestamp
                            for trace in traces]).to(
                                device=device,
                                dtype=torch.float64),
                        feedback_timestamp[row_index].reshape(1),
                    ]).unsqueeze(0)
                if reward_observation is not None:
                    terminal_reward = reward_observation[row_index]
                    reward_reference = torch.cat([
                        predicted_reward,
                        terminal_reward.reshape(1)])
                    reward_magnitude = reward_reference.abs().amax().clamp_min(
                        torch.finfo(dtype).tiny)
                    reward_reference_unit = (
                        reward_reference / reward_magnitude)
                    reward_center_unit = reward_reference_unit.mean()
                    reward_scale_unit = torch.sqrt(
                        (reward_reference_unit - reward_center_unit)
                        .square().mean()).clamp_min(scale_epsilon)
                    normalized_reward = (
                        predicted_reward / reward_magnitude
                        - reward_center_unit) / reward_scale_unit
                    normalized_terminal_reward = (
                        terminal_reward / reward_magnitude
                        - reward_center_unit) / reward_scale_unit
                    corrected, normalized_variance = (
                        value.reward_predictor.PosteriorSmooth(
                            normalized_reward.unsqueeze(0),
                            normalized_terminal_reward.reshape(1),
                            returnVariance=True,
                            timestamps=posterior_timestamps))
                    corrected_normalized_reward = corrected.squeeze(0)
                    normalized_reward_variance = normalized_variance.squeeze(0)
                    reward_scale = reward_scale_unit * reward_magnitude
                    corrected_reward = torch.nan_to_num(
                        (corrected_normalized_reward * reward_scale_unit
                         + reward_center_unit) * reward_magnitude,
                        nan=0.0,
                        posinf=torch.finfo(dtype).max,
                        neginf=-torch.finfo(dtype).max)
                    reward_variance = torch.nan_to_num(
                        normalized_reward_variance * reward_scale.square(),
                        nan=0.0,
                        posinf=torch.finfo(dtype).max,
                        neginf=0.0)
                    reward_innovation = (
                        corrected_normalized_reward - normalized_reward)
                    reward_noise = normalized_reward_variance
                if done_observation is not None:
                    terminal_done = done_observation[row_index]
                    corrected, variance = (
                        value.done_predictor.PosteriorSmooth(
                            predicted_done.unsqueeze(0),
                            terminal_done.reshape(1),
                            bounded=True,
                            returnVariance=True,
                            timestamps=posterior_timestamps))
                    corrected_done = corrected.squeeze(0)
                    done_variance = variance.squeeze(0)
                    done_reference = torch.cat([
                        predicted_done,
                        terminal_done.reshape(1)])
                    done_center = done_reference.mean()
                    done_scale_variance = (
                        (done_reference - done_center).square().mean()
                        + (done_reference
                           * (1.0 - done_reference)).mean())
                    done_scale = torch.sqrt(
                        done_scale_variance.clamp_min(machine_epsilon))
                    done_innovation = (
                        corrected_done - predicted_done) / done_scale
                    done_noise = done_variance / done_scale.square()
                active_channel_count = int(
                    reward_observation is not None) + int(
                        done_observation is not None)
                normalized_energy = torch.zeros_like(predicted_reward)
                normalized_noise = torch.zeros_like(predicted_reward)
                if reward_observation is not None:
                    normalized_energy = (
                        normalized_energy
                        + reward_innovation.clamp(
                            -saturation_innovation,
                            saturation_innovation).square())
                    normalized_noise = normalized_noise + reward_noise
                if done_observation is not None:
                    normalized_energy = (
                        normalized_energy
                        + done_innovation.clamp(
                            -saturation_innovation,
                            saturation_innovation).square())
                    normalized_noise = normalized_noise + done_noise
                normalized_energy = (
                    normalized_energy / float(active_channel_count)
                ).clamp(0.0, saturation_energy)
                information = -torch.expm1(-0.5 * normalized_energy)
                posterior_noise = torch.sqrt(
                    (normalized_noise / float(active_channel_count))
                    .clamp(0.0, saturation_energy))
                write_confidence = (
                    information / (1.0 + posterior_noise)
                    * trace_confidence.clamp(0.0, 1.0)
                    * (1.0 - trace_uncertainty).clamp(0.0, 1.0)
                ).clamp(0.0, 1.0)
                write_mask = write_confidence > 1e-4
                if bool(write_mask.any().item()):
                    contexts = torch.stack([
                        trace.context
                        for trace in traces]).to(device=device, dtype=dtype)
                    outcomes = torch.stack([
                        predicted_reward,
                        corrected_reward,
                        predicted_done,
                        corrected_done,
                        reward_variance,
                        done_variance,
                        information,
                        torch.full_like(
                            information,
                            float(MemoryType.SRC_MIXED)),
                    ], dim=-1)
                    self.mem.RecordCounterfactualEpisode(
                        contexts[write_mask],
                        outcomes[write_mask],
                        self.model_signature,
                        confidence=write_confidence[write_mask],
                        transactionVersion=(
                            self.ContractReplayTransactionVersion),
                        timelineVersion=self.ContractReplayTimelineVersion)
                    best_index = int(
                        write_confidence.argmax().item())
                    recon_query[row_index] = contexts[best_index]
                    recon_outcome[row_index] = outcomes[best_index]
                    recon_confidence[row_index] = write_confidence[best_index]
                    corrected_rows[row_index] = True
                    correction_energy[row_index] = information.mean()
                    self.ContractReplayTransactionVersion += 1
                history.clear()
            if bool(corrected_rows.any().item()) and not bool(isTrain):
                revision = self.mem.BuildCounterfactualRevision(
                    recon_query,
                    recon_outcome,
                    recon_confidence)
                reconsolidated = self.mem.ReconsolidateSemantic(
                    revision["query"],
                    revision["revisedValue"],
                    recon_confidence,
                    similarityThreshold=0.25)
            return {
                "corrected_rows": corrected_rows,
                "correction_energy": correction_energy,
                "reconsolidated": reconsolidated}

    @torch.no_grad()
    def RecordContractReplayTrace(
            self,
            context: torch.Tensor,
            predictedReward: torch.Tensor,
            predictedDone: torch.Tensor,
            confidence: torch.Tensor,
            uncertainty: torch.Tensor,
            timestamp: torch.Tensor,
            validMask: torch.Tensor,
            *,
            isTrain: bool,
        ) -> None:
            if not self.need_trace:
                return
            batch_size = int(self.ContractRuntimeBatch)
            expected_vectors = (
                predictedReward,
                predictedDone,
                confidence,
                uncertainty,
                timestamp,
                validMask)
            if (
                context.dim() != 2
                or int(context.size(0)) != batch_size
                or any(value.reshape(-1).numel() != batch_size
                       for value in expected_vectors)
            ):
                raise ValueError("replay trace fields must share the runtime batch")
            timeline_version = int(self.ContractReplayTimelineVersion)
            for row_index in validMask.to(
                device=context.device,
                dtype=torch.bool).nonzero(
                    as_tuple=False).flatten().tolist():
                row = int(row_index)
                self.ContractReplayHistory[row].append(ContractReplayTrace(
                    context=context[row].detach().clone(),
                    predicted_reward=(
                        predictedReward.reshape(-1)[row].detach().clone()),
                    predicted_done=(
                        predictedDone.reshape(-1)[row].detach().clone()),
                    confidence=confidence.reshape(-1)[row].detach().clone(),
                    uncertainty=(
                        uncertainty.reshape(-1)[row].detach().clone()),
                    timestamp=timestamp.reshape(-1)[row].detach().clone(),
                    episode_version=int(
                        self.ContractReplayEpisodeVersion[row].item()),
                    timeline_version=timeline_version))
            self.ContractReplayTimelineVersion += 1

    def BuildTopDownContext(
            self,
            realizedVisualPrior: Optional[Dict[str, torch.Tensor]],) -> TopDownContext:
            predicted_visual = None
            if realizedVisualPrior is not None:
                if (
                    type(realizedVisualPrior) is not dict
                    or set(realizedVisualPrior) != {
                        "predicted_visual",
                        "reconstructed_visual_state",
                        "prior_rollout"}
                    or type(realizedVisualPrior[
                        "reconstructed_visual_state"]) is not dict
                ):
                    raise ValueError("realized visual prior is invalid")
                predicted_visual = realizedVisualPrior[
                    "reconstructed_visual_state"]
            return TopDownContext(
                PredictedVisual=predicted_visual,
                Precision=self.prev_precision,
                MemoryCue=self.prev_mem,)

    def BuildVisualSequenceTensors(
            self,
            visualStates: List[VisualState],
            *,
            validMasks: Optional[List[torch.Tensor]] = None,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            recent = visualStates[-self.SEQ_LEN:]
            recent_valid = (
                validMasks[-self.SEQ_LEN:]
                if validMasks is not None
                else [torch.ones(batchSize, device=device, dtype=torch.bool) for _ in recent])
            if len(recent_valid) != len(recent):
                raise ValueError("validMasks must have one batch mask per visual state")
            T = len(recent)
            start = self.SEQ_LEN - T

            integrated = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionFeat, device=device, dtype=dtype)
            perc_mod = self.RuntimeModule(self.perc)
            object_token_count = int(getattr(perc_mod, "object_token_count", 16))
            object_seq = torch.zeros(batchSize, self.SEQ_LEN, object_token_count, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
            motion_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
            quality_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
            pred_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
            key_padding_mask = torch.ones(batchSize, self.SEQ_LEN, device=device, dtype=torch.bool)

            for idx, (vs, valid) in enumerate(zip(recent, recent_valid), start=start):
                integrated[:, idx] = vs.IntegratedFeat
                object_seq[:, idx] = vs.ObjectTokens
                motion_seq[:, idx] = vs.MotionToken
                quality_seq[:, idx] = vs.QualityToken
                pred_seq[:, idx] = vs.PredErrorToken
                key_padding_mask[:, idx] = ~valid

            return integrated.contiguous(), object_seq.contiguous(), motion_seq.contiguous(), quality_seq.contiguous(), pred_seq.contiguous(), key_padding_mask

    @staticmethod
    def BuildLocalDetailMask(
            objectSequence: torch.Tensor,
            motionSequence: torch.Tensor,
            predictionErrorSequence: torch.Tensor,
            activeMask: torch.Tensor,
            detailRows: torch.Tensor,
        ) -> torch.Tensor:
            if objectSequence.dim() != 4:
                raise ValueError("objectSequence must have shape [B, S, N, D]")
            batch_size, sequence_length = objectSequence.shape[:2]
            if (
                motionSequence.dim() != 3
                or predictionErrorSequence.dim() != 3
                or motionSequence.shape[:2] != (batch_size, sequence_length)
                or predictionErrorSequence.shape[:2]
                != (batch_size, sequence_length)
                or activeMask.shape != (batch_size, sequence_length)
                or detailRows.shape != (batch_size,)
                or activeMask.dtype != torch.bool
                or detailRows.dtype != torch.bool
            ):
                raise ValueError("detail attention inputs are incompatible")
            salience = (
                torch.log1p(objectSequence.square().mean(dim=(2, 3)))
                + torch.log1p(motionSequence.square().mean(dim=-1))
                + torch.log1p(
                    predictionErrorSequence.square().mean(dim=-1)))
            salience = salience.masked_fill(~activeMask, -torch.inf)
            selection_count = min(
                int(sequence_length),
                max(1, int(math.sqrt(int(sequence_length)))))
            selected_index = torch.topk(
                salience,
                k=selection_count,
                dim=-1).indices
            selected = torch.zeros_like(activeMask)
            selected.scatter_(1, selected_index, True)
            return selected & activeMask & detailRows.unsqueeze(-1)

    def EncodeOcrSemantic(self, ocrTexts: Optional[List[List[str]]], *, batchSize: int, device: torch.device) -> torch.Tensor:
            intention_mod = self.RuntimeModule(self.intention)
            if ocrTexts is None:
                merged = [""] * batchSize
            else:
                merged = intention_mod.MergeOcrTexts(ocrTexts)
            if len(merged) != batchSize:
                merged = (merged + [""] * batchSize)[:batchSize]
            with torch.no_grad():
                sem, _, _ = intention_mod.EncodeStringsWithSlots(merged, device=device)
            return sem.detach()

    def AttachEntityOntology(
            visualState: VisualState,
            observedPst: Dict[str, torch.Tensor],
        ) -> None:
            auxiliary = visualState.Auxiliary
            auxiliary["PerceptualObjectAgencyProb"] = auxiliary["ObjectAgencyProb"]
            auxiliary["PerceptualMotionLayerProb"] = auxiliary["ObjectMotionLayerProb"]
            auxiliary["PerceptualLayerAgencyProb"] = auxiliary["LayerAgencyProb"]
            auxiliary["PerceptualDisplaySurfaceProb"] = auxiliary["DisplaySurfaceProb"]
            auxiliary["PerceptualSurfaceUV"] = auxiliary["SurfaceUV"]
            auxiliary["PerceptualSurfaceUVConfidence"] = auxiliary["SurfaceUVConfidence"]
            auxiliary["PerceptualContentMotionUV"] = auxiliary["ContentMotionUV"]
            auxiliary["PerceptualContentChangeProb"] = auxiliary["ContentChangeProb"]
            for name in (
                "PerceptualPresence",
                "GeometryValidMask",
                "MphysRaw",
                "PhysicalEntityProb",
                "PhysicalInteractionProb",
                "RealmProb",
                "MotionLayerProb",
                "LayerAgencyProb",
                "AgencyProb",
                "BodyMembershipProb",
                "SelfPartProb",
                "SelfPartSemantic",
                "CarrierMotionObserverRaw",
                "ArticulationMotionObserverRaw",
                "ContentMotionUV",
                "ContentChangeProb",
                "DisplaySurfaceProb",
                "SurfaceParentProb",
                "SurfaceUV",
                "SurfaceUVConfidence",
                "VerificationConfidence",
                "OntologyRelationProb",):
                auxiliary[name] = observedPst[name]
            auxiliary["EntityRealmProb"] = observedPst["RealmProb"]
            auxiliary["ObjectAgencyProb"] = observedPst["AgencyProb"]
            auxiliary["ObjectMotionLayerProb"] = observedPst["MotionLayerProb"]

    def BuildOntologyAuxSequence(
            self,
            visualStates: List[VisualState],
            *,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> Dict[str, torch.Tensor]:
            recent = visualStates[-self.SEQ_LEN:]
            T = len(recent)
            start = self.SEQ_LEN - T
            K = self.RuntimeModule(self.perc).object_token_count
            shapes = {
                "PerceptualPresence": (),
                "EntityRealmProb": (ModuleDim.PstRealmClasses,),
                "ObjectAgencyProb": (ModuleDim.PstAgencyClasses,),
                "ObjectMotionLayerProb": (ModuleDim.PstMotionLayerClasses,),
                "LayerAgencyProb": (
                    ModuleDim.PstMotionLayerClasses,
                    ModuleDim.PstAgencyClasses),
                "BodyMembershipProb": (),
                "PhysicalInteractionProb": (),
                "SelfPartSemantic": (ModuleDim.PstSelfPartSemanticDim,),
                "ContentMotionUV": (2,),
                "ContentChangeProb": (),}
            sequence = {
                name: torch.zeros(
                    batchSize,
                    self.SEQ_LEN,
                    K,
                    *tail,
                    device=device,
                    dtype=dtype)
                for name, tail in shapes.items()}
            for index, state in enumerate(recent, start=start):
                auxiliary = state.Auxiliary
                sequence["PerceptualPresence"][:, index] = auxiliary[
                    "PerceptualPresence"]
                sequence["EntityRealmProb"][:, index] = auxiliary["RealmProb"]
                sequence["ObjectAgencyProb"][:, index] = auxiliary["AgencyProb"]
                sequence["ObjectMotionLayerProb"][:, index] = auxiliary["MotionLayerProb"]
                sequence["LayerAgencyProb"][:, index] = auxiliary["LayerAgencyProb"]
                sequence["BodyMembershipProb"][:, index] = auxiliary["BodyMembershipProb"]
                sequence["PhysicalInteractionProb"][:, index] = auxiliary["PhysicalInteractionProb"]
                sequence["SelfPartSemantic"][:, index] = auxiliary[
                    "SelfPartSemantic"]
                sequence["ContentMotionUV"][:, index] = auxiliary["PerceptualContentMotionUV"]
                sequence["ContentChangeProb"][:, index] = auxiliary["PerceptualContentChangeProb"]
            return sequence

    def StepContract(
            self,
            step: ContractBrainStepInput,
        ) -> BrainStepOutput:
            self.ValidateContractStepInput(step)
            self.ValidateFeedbackPacket(
                step.feedback_packet,
                batchSize=int(step.frame.size(0)),
                device=step.frame.device)
            feedback_packet = step.feedback_packet
            frame = step.frame
            batch_size = int(frame.size(0))
            device = frame.device
            dtype = frame.dtype
            self.EnsureContractCognitiveState(
                batch_size,
                device,
                dtype)
            is_begin_step = True

            def SaveModuleOutput(moduleName: str, output: Any) -> None:
                nonlocal is_begin_step
                if not self.save_module_messager_output:
                    return
                self.moduleMessager.SaveModuleOutput(
                    moduleName,
                    output,
                    isBeginStep=is_begin_step)
                is_begin_step = False

            done_event = (
                torch.zeros(batch_size, device=device, dtype=torch.bool)
                if step.done_flag is None
                else step.done_flag.reshape(batch_size).gt(0.5))
            replay_correction = self.CorrectContractReplayFeedback(
                step.reward_ext,
                step.done_flag,
                isTrain=step.is_train,
                feedbackTimestamp=feedback_packet.timestamp)
            SaveModuleOutput(
                "MemoryPosteriorCorrection",
                replay_correction)

            text_fingerprint = self.BuildTextFingerprint(
                step.text_ext,
                step.text_trust,
                batch_size,
                device)
            text_evidence_changed = text_fingerprint.ne(
                self.ContractPreviousTextFingerprint)
            command_text = [
                text_value
                if trust_value == TEXT_TRUST_OPERATOR_COMMAND
                else None
                for text_value, trust_value in zip(
                    [None] * batch_size
                    if step.text_ext is None
                    else step.text_ext,
                    [None] * batch_size
                    if step.text_trust is None
                    else step.text_trust)
            ]
            command_fingerprint = self.BuildTextFingerprint(
                command_text,
                step.text_trust,
                batch_size,
                device)
            command_changed = command_fingerprint.ne(
                self.ContractPreviousCommandFingerprint)
            command_version = (
                self.ContractCommandVersion
                + command_changed.to(dtype=torch.long))
            pending_intent_evidence = (
                self.ContractIntentionCommitmentState.get("pending_valid"))
            if pending_intent_evidence is None:
                pending_intent_evidence = torch.zeros(
                    batch_size,
                    device=device,
                    dtype=torch.bool)
            elif (
                not torch.is_tensor(pending_intent_evidence)
                or tuple(pending_intent_evidence.shape) != (batch_size,)
                or pending_intent_evidence.dtype != torch.bool
                or pending_intent_evidence.device != device
            ):
                raise ValueError("pending intention evidence is invalid")
            pre_attention_intent_event = command_changed
            compute_decision = self.BuildContractComputeDecision(
                feedback_packet)
            if bool(pre_attention_intent_event.any().item()):
                command_reason = compute_decision.reason_target.clone()
                command_reason[:, 1] = (
                    command_reason[:, 1] | pre_attention_intent_event)
                command_full_mode = torch.full_like(
                    compute_decision.mode,
                    int(CognitiveComputeMode.FULL_REPLAN))
                command_failsafe = compute_decision.mode.eq(
                    int(CognitiveComputeMode.FAILSAFE))
                compute_decision = CognitiveComputeDecision(
                    mode=torch.where(
                        pre_attention_intent_event & ~command_failsafe,
                        command_full_mode,
                        compute_decision.mode),
                    hard_trigger=(
                        compute_decision.hard_trigger
                        | pre_attention_intent_event),
                    activated_child_mask=(
                        compute_decision.activated_child_mask),
                    evc_trigger=compute_decision.evc_trigger,
                    reason_target=command_reason)
            scheduled_compute_mode = compute_decision.mode
            failsafe_event = scheduled_compute_mode.eq(
                int(CognitiveComputeMode.FAILSAFE))
            stopped_event = failsafe_event | done_event
            SaveModuleOutput("CognitiveCompute", compute_decision)
            if bool(stopped_event.all().item()) and (
                not step.is_train
                or bool(failsafe_event.all().item())
            ):
                temporal_decision = self.BuildFailsafeTemporalDecision(
                    feedback_packet)
                actor = self.RuntimeModule(self.actor)
                actor.ResetHebbianMemory()
                self.prev_option_logit = torch.zeros_like(
                    self.prev_option_logit)
                self.prev_fast_option_logit = torch.zeros_like(
                    self.prev_fast_option_logit)
                self.prev_detail_option_logit = torch.zeros_like(
                    self.prev_detail_option_logit)
                self.prev_decision_state = torch.zeros_like(
                    self.prev_decision_state)
                self.prev_fast_decision_state = torch.zeros_like(
                    self.prev_fast_decision_state)
                self.prev_detail_decision_state = torch.zeros_like(
                    self.prev_detail_decision_state)
                self.prev_latent_control = torch.zeros_like(
                    self.prev_latent_control)
                self.prev_fast_latent_control = torch.zeros_like(
                    self.prev_fast_latent_control)
                self.prev_detail_latent_control = torch.zeros_like(
                    self.prev_detail_latent_control)
                self.prev_mapper_hidden = torch.zeros_like(
                    self.prev_mapper_hidden)
                self.prev_fast_mapper_hidden = torch.zeros_like(
                    self.prev_fast_mapper_hidden)
                self.prev_detail_mapper_hidden = torch.zeros_like(
                    self.prev_detail_mapper_hidden)
                self.prev_entropy = torch.zeros_like(
                    self.prev_entropy)
                self.prev_belief_prediction_state = torch.zeros_like(
                    self.prev_belief_prediction_state)
                self.prev_belief_prediction_valid = torch.zeros_like(
                    self.prev_belief_prediction_valid)
                self.active_option_policy_input = torch.zeros_like(
                    self.active_option_policy_input)
                self.active_option_prior_logit = torch.zeros_like(
                    self.active_option_prior_logit)
                self.active_option_goal_mid = torch.zeros_like(
                    self.active_option_goal_mid)
                self.active_option_index = torch.zeros_like(
                    self.active_option_index)
                self.active_option_valid = torch.zeros_like(
                    self.active_option_valid)
                self.ContractCachedTarget = temporal_decision.selected_target
                self.ContractCachedActionEpoch = (
                    temporal_decision.action_epoch.detach().clone())
                self.ContractCacheAge = self.ContractCacheAge + 1.0
                self.ContractSlowCacheAge = torch.where(
                    failsafe_event,
                    self.ContractSlowCacheAge,
                    self.ContractSlowCacheAge + 1.0)
                self.ContractPreviousTextFingerprint = torch.where(
                    failsafe_event,
                    self.ContractPreviousTextFingerprint,
                    text_fingerprint).detach().clone()
                self.ContractPreviousCommandFingerprint = torch.where(
                    failsafe_event,
                    self.ContractPreviousCommandFingerprint,
                    command_fingerprint).detach().clone()
                self.ContractCommandVersion = torch.where(
                    failsafe_event,
                    self.ContractCommandVersion,
                    command_version).detach().clone()
                self.prev_intent_changed = (
                    self.prev_intent_changed.detach().clone())
                self.ContractPreviousTargetActive = (
                    feedback_packet.target_active.detach().clone())
                self.ContractPreviousProgress = (
                    feedback_packet.progress.detach().clone())
                self.prev_done_flag = torch.where(
                    failsafe_event,
                    self.prev_done_flag | done_event,
                    done_event).detach().clone()
                cached = self.ContractSlowCognitiveCache
                ocr_items = (
                    [[] for _ in range(batch_size)]
                    if cached is None
                    else copy.deepcopy(cached.get(
                        "OcrItems",
                        [[] for _ in range(batch_size)])))
                intention_texts = (
                    []
                    if cached is None
                    else copy.deepcopy(cached.get("IntentionTexts", [])))
                if bool(done_event.any().item()):
                    self.ResetContractStateRows(done_event)
                    self.ResetHebbianMemory(doneMask=done_event)
                    self.mem.ResetEpisodeState(done_event)
                    self.RuntimeModule(self.critic).ResetState(
                        doneMask=done_event)
                    self.conscious.ResetState(doneMask=done_event)
                    self.OCR.ResetTemporal(doneMask=done_event)
                    self.contract_neuro_symbolic.ResetPlan(done_event)
                    self.ContractWorld().ResetEpisodeState(done_event)
                SaveModuleOutput("TemporalExecution", temporal_decision)
                return BrainStepOutput(
                    decision={
                        "packed_target": temporal_decision.selected_target,
                        "packed_temporal": temporal_decision,
                        "planner_prior": None},
                    world={},
                    critic=None,
                    features={
                        "CognitiveCompute": compute_decision,
                        "MemoryPosteriorCorrection": replay_correction},
                    ocr=ocr_items,
                    intention_texts=intention_texts,
                    losses={},
                    stages={
                        "CognitiveCompute": compute_decision,
                        "ScheduledComputeMode": scheduled_compute_mode,
                        "SlowCognitionReused": True,
                        "PlannerExecuted": False,
                        "PlanCacheReused": torch.zeros(
                            batch_size,
                            device=device,
                            dtype=torch.bool),
                        "StoppedRows": stopped_event})
            if (
                self.contract_physical_adapter is None
                or self.contract_pst_builder is None
                or self.contract_neuro_symbolic_grounder is None
                or self.contract_neuro_symbolic is None
                or self.packed_decision_decoupler is None
                or self.packed_temporal_gate is None
            ):
                raise RuntimeError(
                    "StepContract requires a fully contract-bound BrainBuildSpec")

            rotation_delta, angular_velocity, rotation_valid = (
                self.SelectContractPerceptionMotion(feedback_packet))
            previous_rotation, current_rotation = (
                self.BuildContractObserverGauge(
                    rotation_delta,
                    rotation_valid))

            top_down = self.BuildTopDownContext(
                self.prospective_visual_prediction)
            previous_visual_state = self.prev_visual_state
            previous_visual_valid = self.prev_visual_valid
            visual_state = self.perc(
                frame,
                prevVisualState=previous_visual_state,
                prevVisualValid=previous_visual_valid,
                topDownContext=top_down,
                depth=step.depth,
                depthValid=step.depth_valid,
                observerRotation=rotation_delta,
                observerRotationValid=rotation_valid,
                observerAngularVelocity=angular_velocity)
            visual_sequence = self.visual_state_buffer + [visual_state]
            visual_valid_sequence = self.visual_state_valid_buffer + [
                torch.ones(batch_size, device=device, dtype=torch.bool)]
            if len(visual_sequence) > self.SEQ_LEN:
                visual_sequence = visual_sequence[-self.SEQ_LEN:]
                visual_valid_sequence = visual_valid_sequence[-self.SEQ_LEN:]
            (
                percs_seq,
                object_seq,
                motion_seq,
                quality_seq,
                pred_error_seq,
                key_padding_mask,
            ) = self.BuildVisualSequenceTensors(
                visual_sequence,
                validMasks=visual_valid_sequence,
                batchSize=batch_size,
                device=device,
                dtype=dtype)
            SaveModuleOutput("Perception", visual_state)
            current_novelty = torch.log1p(
                pred_error_seq[:, -1].square().mean(dim=-1))
            novelty_signal = torch.log1p(
                pred_error_seq.square().mean(dim=(1, 2)))
            task_relevance_query = self.BuildTaskRelevanceQuery()
            attention_top_down_bias = self.BuildAttentionTopDownBias()
            task_relevant_novelty = self.BuildTaskRelevantNovelty(
                visual_state,
                task_relevance_query)
            uncertainty_weighted_surprise = (
                novelty_signal
                * (1.0 - self.prev_precision).clamp(0.0, 1.0))
            ocr_refresh_mask = self.BuildOcrRefreshMask(
                current_novelty,
                batch_size,
                step.is_train)
            ocr_refresh_mask = ocr_refresh_mask & ~stopped_event
            ocr_row_index = ocr_refresh_mask.nonzero(
                as_tuple=False).flatten()
            ocr_cache = self.ContractSlowCognitiveCache
            ocr_items = (
                [[] for _ in range(batch_size)]
                if ocr_cache is None or "OcrItems" not in ocr_cache
                else copy.deepcopy(ocr_cache["OcrItems"]))
            fuse_ocr = (
                [[] for _ in range(batch_size)]
                if ocr_cache is None or "OcrTexts" not in ocr_cache
                else copy.deepcopy(ocr_cache["OcrTexts"]))
            if ocr_row_index.numel() > 0:
                ocr_update = self.OCR(
                    frame.index_select(0, ocr_row_index),
                    batchIndices=ocr_row_index,
                    fullBatchSize=batch_size)
                fused_update = [
                    self.OCR.ExportFusedTexts(int(row_index))
                    for row_index in ocr_row_index.tolist()]
                ocr_items = self.ScatterRuntimeRows(
                    ocr_items,
                    ocr_update,
                    ocr_row_index,
                    batch_size)
                fuse_ocr = self.ScatterRuntimeRows(
                    fuse_ocr,
                    fused_update,
                    ocr_row_index,
                    batch_size)
            SaveModuleOutput("OCR", ocr_items)
            ocr_semantic = self.EncodeOcrSemantic(
                fuse_ocr,
                batchSize=batch_size,
                device=device)

            contract_body = self.contract_physical_adapter(
                feedback_packet)
            semantic_view = self.contract_pst_builder.SemanticWorldView(
                visual_state.SemanticNodes)
            physical_out = self.contract_pst_builder(
                visual_state.ObjectTokens,
                visual_state.Auxiliary["ObjectMotion"],
                visual_state.Auxiliary["ObjectGeometry"],
                semantic_view["NodePresence"],
                visual_state.Auxiliary["ObjectGeometryValid"],
                slotBodyTokens=contract_body["SlotBodyTokens"],
                slotWeight=contract_body["SlotWeight"])
            realized_action, action_agency_evidence, agency_evidence_valid = (
                self.BuildContractActionAgencyEvidence(
                    feedback_packet,
                    contract_body))
            self.RefineContractLayerAgency(
                physical_out,
                action_agency_evidence,
                agency_evidence_valid)
            observed_pst = {
                **physical_out,
                **semantic_view,
                "ObservedSlotMask": physical_out["ObservationMask"],
            }
            self.AttachEntityOntology(visual_state, observed_pst)
            ontology_aux_sequence = self.BuildOntologyAuxSequence(
                visual_sequence,
                batchSize=batch_size,
                device=device,
                dtype=dtype)
            object_seq = self.RuntimeModule(
                self.attn).EncodeOntologyObjectSequence(
                    object_seq,
                    ontology_aux_sequence)

            (
                self.visual_state_buffer,
                self.visual_state_valid_buffer,
            ) = self.MergeVisualHistory(
                self.visual_state_buffer,
                self.visual_state_valid_buffer,
                visual_sequence,
                visual_valid_sequence,
                failsafe_event)
            self.perc_buffer = [
                value.IntegratedFeat
                for value in self.visual_state_buffer
                if value is not None]
            active_visual_rows = (~failsafe_event).nonzero(
                as_tuple=False).flatten()
            previous_visual_base = (
                visual_state
                if previous_visual_state is None
                else previous_visual_state)
            self.prev_visual_state = self.DetachVisualState(
                self.ScatterRuntimeRows(
                    previous_visual_base,
                    self.IndexRuntimeRows(
                        visual_state,
                        active_visual_rows,
                        batch_size),
                    active_visual_rows,
                    batch_size))
            self.prev_visual_valid = torch.where(
                failsafe_event,
                previous_visual_valid,
                torch.ones(
                    batch_size,
                    device=device,
                    dtype=torch.bool)).detach().clone()

            attention_stage = ~key_padding_mask
            detail_rows = (
                compute_decision.mode.eq(
                    int(CognitiveComputeMode.DETAIL_EXECUTE))
                | compute_decision.mode.eq(
                    int(CognitiveComputeMode.FULL_REPLAN))) & ~stopped_event
            local_detail = self.BuildLocalDetailMask(
                object_seq,
                motion_seq,
                pred_error_seq,
                attention_stage,
                detail_rows)
            attention_active = ~stopped_event
            attention_full_mask = (
                compute_decision.mode.eq(
                    int(CognitiveComputeMode.FULL_REPLAN))
                & attention_active)
            attention_fast_mask = (
                compute_decision.mode.eq(
                    int(CognitiveComputeMode.FAST_EXECUTE))
                & attention_active)
            attention_detail_mask = (
                compute_decision.mode.eq(
                    int(CognitiveComputeMode.DETAIL_EXECUTE))
                & attention_active)
            atten_out, attention_extras = self.attn.ForwardConditional(
                percs_seq,
                objectSeq=object_seq,
                motionSeq=motion_seq,
                qualitySeq=quality_seq,
                predErrorSeq=pred_error_seq,
                goalBias=attention_top_down_bias,
                precision=self.prev_precision,
                fullMask=attention_full_mask,
                fastMask=attention_fast_mask,
                detailMask=attention_detail_mask,
                keyPaddingMask=key_padding_mask,
                tdError=self.prev_td_error,
                uncertainty=(1.0 - self.prev_precision).clamp(0.0, 1.0),
                novelty=novelty_signal,
                risk=self.prev_risk,
                informationGain=self.prev_information_gain,
                stageMask=attention_stage,
                localDetailMask=local_detail,
                trainStudents=step.is_train,
                returnExtras=True)

            world = self.ContractWorld()
            world.EnsureB(batch_size)
            previous_physical_state = world.BuildContractModelPhysicalState(
                world.ExportPhysicalState(),
                previous_rotation)
            previous_embodiment_state = self.prev_world_embodiment
            world_embodiment_state = world.EncodeContractTransition(
                feedback_packet)
            physical_runtime_snapshot = (
                world.CapturePhysicalRuntimeRows(failsafe_event)
                if bool(failsafe_event.any().item())
                else None)
            pst = world.UpdateContractPhysicalState(
                observed_pst,
                observerRotationWorld=current_rotation,
                embodimentState=world_embodiment_state,
                observerValid=rotation_valid)
            if physical_runtime_snapshot is not None:
                world.RestorePhysicalRuntimeRows(
                    physical_runtime_snapshot,
                    failsafe_event)
            pst["U"] = self.mem.usage_bank.SlotReadout(
                pst["IdentityKey"],
                pst["ARaw"],
                pst["SlotPresence"] * pst["MphysRaw"]
            ) * pst["SlotPresence"].unsqueeze(-1)
            physical_mask = pst["SlotPresence"] * pst["MphysRaw"]
            physical_summary = self.contract_pst_builder.SlotSummary(
                pst["SlotState"],
                physical_mask)
            entity_mask = pst["SlotPresence"] * pst["PerceptualPresence"]
            entity_summary = self.contract_pst_builder.SlotSummary(
                pst["SlotState"],
                entity_mask)
            entity_valid = entity_mask.sum(dim=-1, keepdim=True).gt(0.0)
            pst_summary = (
                physical_summary
                + entity_valid.to(dtype=physical_summary.dtype)
                * torch.sigmoid(self.contract_entity_summary_gain)
                * self.contract_entity_summary_fuser(torch.cat([
                    physical_summary,
                    entity_summary,
                ], dim=-1)))
            SaveModuleOutput("PhysicalState", pst)

            preview_world = None
            preview_value = None
            preview_critic_state = None
            attention_refinement_mask = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool)
            world_value_passes = torch.ones(
                batch_size,
                device=device,
                dtype=dtype)
            if not step.is_train:
                with torch.no_grad():
                    preview_world = world.StepPosterior(
                        visionIn=atten_out,
                        actionEnc=realized_action,
                        physicalState=pst,
                        transitionPhysicalState=previous_physical_state,
                        embodimentState=world_embodiment_state,
                        transitionEmbodimentState=previous_embodiment_state,
                        observerMotion=rotation_delta,
                        observerMotionValid=rotation_valid,
                        sample=False,
                        commitState=False,
                        updateMemory=False)
                    preview_information_gain = (
                        self.ComputeRealizedInformationGain(
                            preview_world["mu_q"],
                            preview_world["logstd_q"],
                            preview_world["mu_p"],
                            preview_world["logstd_p"]))
                    preview_surprise = torch.log1p(
                        preview_information_gain)
                    preview_surprise = torch.where(
                        self.prev_done_flag,
                        torch.zeros_like(preview_surprise),
                        preview_surprise)
                    preview_reward = (
                        preview_world["r_pred"].detach()
                        if step.reward_ext is None
                        else step.reward_ext.detach().reshape(batch_size))
                    preview_done = (
                        preview_world["d_prob"].detach()
                        if step.done_flag is None
                        else step.done_flag.detach().reshape(batch_size))
                    critic_module = self.RuntimeModule(self.critic)
                    preview_critic_state = critic_module.ExportState()
                    try:
                        preview_value = critic_module(
                            memoryPrev=self.prev_mem,
                            attnPrev=self.prev_attn,
                            state=preview_world["s_next"],
                            rewardModel=preview_reward,
                            doneModel=preview_done,
                            computeLoss=False,
                            commitMask=~failsafe_event,
                            policyEntropyPrev=self.prev_entropy,
                            worldDeltaTransport=preview_world["d_tr"],
                            worldDeltaPhysics=preview_world["d_ph"])
                        preview_value = (
                            critic_module.RefineEntityOntologyRisk(
                                preview_value,
                                pst,
                                sampleMask=~failsafe_event))
                        compute_decision = (
                            self.EscalateContractComputeDecision(
                                compute_decision,
                                feedback_packet,
                                task_relevant_novelty,
                                preview_surprise,
                                preview_value.rComps["risk"].detach()))
                        attention_refinement_mask = (
                            self.BuildAttentionRefinementMask(
                                False,
                                scheduled_compute_mode,
                                compute_decision.mode,
                                stopped_event))
                    except BaseException:
                        critic_module.ImportState(preview_critic_state)
                        raise
                if bool(attention_refinement_mask.any().item()):
                    world_value_passes = torch.full_like(
                        world_value_passes,
                        2.0)
                    critic_module.ImportState(preview_critic_state)
                    refinement_local_detail = self.BuildLocalDetailMask(
                        object_seq,
                        motion_seq,
                        pred_error_seq,
                        attention_stage,
                        attention_refinement_mask)
                    preliminary_actual_units = attention_extras[
                        "actual_compute_units"]
                    refined_attention, refined_extras = (
                        self.attn.ForwardConditional(
                            percs_seq,
                            objectSeq=object_seq,
                            motionSeq=motion_seq,
                            qualitySeq=quality_seq,
                            predErrorSeq=pred_error_seq,
                            goalBias=attention_top_down_bias,
                            precision=preview_value.precision.detach(),
                            fullMask=attention_refinement_mask,
                            fastMask=torch.zeros_like(
                                attention_refinement_mask),
                            detailMask=torch.zeros_like(
                                attention_refinement_mask),
                            keyPaddingMask=key_padding_mask,
                            tdError=preview_value.tdError.detach(),
                            uncertainty=preview_value.uncertainty.detach(),
                            novelty=novelty_signal,
                            risk=preview_value.rComps["risk"].detach(),
                            informationGain=preview_information_gain.detach(),
                            stageMask=attention_stage,
                            localDetailMask=refinement_local_detail,
                            trainStudents=False,
                            returnExtras=True))
                    atten_out = torch.where(
                        attention_refinement_mask.unsqueeze(-1),
                        refined_attention,
                        atten_out)
                    refinement_index = attention_refinement_mask.nonzero(
                        as_tuple=False).flatten()
                    attention_module = self.RuntimeModule(self.attn)
                    selected_refinement_extras = (
                        attention_module.IndexAttentionExtras(
                            refined_extras,
                            refinement_index,
                            batch_size))
                    attention_extras = (
                        attention_module.ScatterAttentionExtras(
                            attention_extras,
                            selected_refinement_extras,
                            refinement_index,
                            batch_size))
                    actual_units = (
                        preliminary_actual_units
                        + refined_extras["actual_compute_units"])
                    full_units = attention_extras["full_compute_units"]
                    attention_extras["actual_compute_units"] = actual_units
                    attention_extras[
                        "selected_normalized_compute_fraction"] = (
                            attention_extras["selected_compute_units"]
                            / full_units)
                    attention_extras[
                        "actual_normalized_compute_fraction"] = (
                            actual_units / full_units)
            attention_extras["world_value_posterior_passes"] = (
                world_value_passes)
            attention_extras[
                "world_value_normalized_compute_fraction"] = (
                    world_value_passes / 2.0)
            SaveModuleOutput("Attention", atten_out)
            SaveModuleOutput("AttentionCompute", {
                name: attention_extras[name]
                for name in (
                    "selected_compute_units",
                    "actual_compute_units",
                    "full_compute_units",
                    "selected_normalized_compute_fraction",
                    "actual_normalized_compute_fraction",
                    "world_value_posterior_passes",
                    "world_value_normalized_compute_fraction")})

            if step.is_train:
                w_out = self.RunWorldTrainingStep(
                    visionIn=atten_out,
                    actionEnc=realized_action,
                    physicalState=pst,
                    transitionPhysicalState=previous_physical_state,
                    embodimentState=world_embodiment_state,
                    transitionEmbodimentState=previous_embodiment_state,
                    observerMotion=rotation_delta,
                    observerMotionValid=rotation_valid,
                    reward=step.reward_ext,
                    done=step.done_flag,
                    sample=(step.is_train and not step.deterministic_actor),
                    commitMask=~failsafe_event)
            elif not bool(attention_refinement_mask.any().item()):
                if (
                    preview_world is None
                    or preview_value is None
                    or preview_critic_state is None
                ):
                    raise RuntimeError("current appraisal preview is unavailable")
                try:
                    w_out = world.CommitPosteriorPreview(
                        preview_world,
                        commitMask=~failsafe_event)
                except BaseException:
                    self.RuntimeModule(self.critic).ImportState(
                        preview_critic_state)
                    raise
            else:
                w_out = world.StepPosterior(
                    visionIn=atten_out,
                    actionEnc=realized_action,
                    physicalState=pst,
                    transitionPhysicalState=previous_physical_state,
                    embodimentState=world_embodiment_state,
                    transitionEmbodimentState=previous_embodiment_state,
                    observerMotion=rotation_delta,
                    observerMotionValid=rotation_valid,
                    sample=False,
                    commitMask=~failsafe_event)
            if step.is_train:
                contract_feedback_prediction = (
                    self.PredictCausalContractFeedback(world, w_out))
                contract_feedback_losses = world.ComputeContractFeedbackLoss(
                    contract_feedback_prediction,
                    feedback_packet,
                    sampleMask=~failsafe_event)
                w_out["contract_feedback_prediction"] = (
                    contract_feedback_prediction)
                for name, value in contract_feedback_losses.items():
                    telemetry_name = (
                        "loss_contract_feedback"
                        if name == "loss"
                        else "loss_contract_feedback_" + name[5:])
                    w_out[telemetry_name] = value
                w_out["loss"] = (
                    w_out["loss"]
                    + 0.1 * contract_feedback_losses["loss"])
            information_gain = self.ComputeRealizedInformationGain(
                w_out["mu_q"],
                w_out["logstd_q"],
                w_out["mu_p"],
                w_out["logstd_p"])
            w_out["realized_information_gain"] = information_gain
            w_out["uncertainty_weighted_surprise"] = (
                uncertainty_weighted_surprise)
            SaveModuleOutput("World", w_out)
            world_hzx = torch.cat([
                w_out["h_next"],
                w_out["z_next"],
                w_out["x_next"],
            ], dim=-1)
            world_surprise = torch.log1p(
                information_gain)
            world_surprise = torch.where(
                self.prev_done_flag,
                torch.zeros_like(world_surprise),
                world_surprise)

            value_reward = (
                w_out["r_pred"].detach()
                if step.reward_ext is None
                else step.reward_ext.detach().reshape(batch_size))
            value_done = (
                w_out["d_prob"].detach()
                if step.done_flag is None
                else step.done_flag.detach().reshape(batch_size))
            if (
                not step.is_train
                and not bool(attention_refinement_mask.any().item())
            ):
                if preview_value is None:
                    raise RuntimeError("current value appraisal is unavailable")
                critic_out = preview_value
            else:
                critic_out = self.critic(
                    memoryPrev=self.prev_mem,
                    attnPrev=self.prev_attn,
                    state=w_out["s_next"],
                    rewardModel=value_reward,
                    doneModel=value_done,
                    computeLoss=step.is_train,
                    commitMask=~failsafe_event,
                    policyEntropyPrev=self.prev_entropy,
                    worldDeltaTransport=w_out["d_tr"],
                    worldDeltaPhysics=w_out["d_ph"])
                critic_out = self.RuntimeModule(
                    self.critic).RefineEntityOntologyRisk(
                        critic_out,
                        pst,
                        sampleMask=~failsafe_event)
            SaveModuleOutput("Value", critic_out)
            td_signal = critic_out.tdError.detach()
            return_advantage = critic_out.returnAdvantage.detach()
            uncertainty = critic_out.uncertainty.detach()
            precision = critic_out.precision.detach()
            emotion = critic_out.emotion.detach()
            risk = critic_out.rComps["risk"].detach()
            confidence = critic_out.rComps["confidence"].detach()
            compute_decision = self.EscalateContractComputeDecision(
                compute_decision,
                feedback_packet,
                task_relevant_novelty,
                world_surprise,
                risk)
            SaveModuleOutput(
                "CognitiveComputeEscalated",
                compute_decision)

            mem_feat = self.mem(
                atten_out,
                tdError=td_signal,
                emotion=emotion,
                reward=value_reward.detach(),
                visualState=visual_state,
                ocrSemantic=ocr_semantic,
                intentHint=self.prev_intent_sem,
                uncertainty=uncertainty,
                risk=risk,
                confidence=confidence,
                writeMask=~failsafe_event,
                lossSampleMask=~failsafe_event)
            SaveModuleOutput("Memory", mem_feat)
            if step.is_train:
                replay_training = self.mem.ConsumeCounterfactualReplay(
                    batchSize=batch_size,
                    modelSignature=self.model_signature,
                    seed=int(self.mem.time_step.max().item()),
                    addInternalLoss=True)
                SaveModuleOutput("MemoryCounterfactualReplay", replay_training)
            stopped_rows = done_event | failsafe_event
            self.RecordContractReplayTrace(
                mem_feat,
                w_out["r_pred"],
                w_out["d_prob"],
                confidence,
                uncertainty,
                feedback_packet.timestamp,
                ~stopped_rows,
                isTrain=step.is_train)
            actor = self.RuntimeModule(self.actor)
            recalled_plan = self.mem.RecallPlan(
                "activePlan",
                self.model_signature)
            recalled_feature = mem_feat.new_zeros(
                batch_size,
                actor.belief_dim)
            recalled_valid = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool)
            recalled_plan_age = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.long)
            if recalled_plan is not None:
                recalled_feature = recalled_plan["feature"].to(
                    device=device,
                    dtype=mem_feat.dtype)
                recalled_valid = recalled_plan["valid"].to(
                    device=device,
                    dtype=torch.bool)
                recalled_plan_age = recalled_plan["age"].to(
                    device=device,
                    dtype=torch.long)
                if (
                    tuple(recalled_feature.shape)
                    != (batch_size, actor.belief_dim)
                    or tuple(recalled_valid.shape) != (batch_size,)
                    or tuple(recalled_plan_age.shape) != (batch_size,)
                    or bool((recalled_plan_age < 0).any().item())
                    or not bool(torch.isfinite(recalled_feature).all().item())
                ):
                    raise RuntimeError("recalled plan shape does not match")
            cache_dependent_mode = (
                compute_decision.mode.eq(
                    int(CognitiveComputeMode.FAST_EXECUTE))
                | compute_decision.mode.eq(
                    int(CognitiveComputeMode.DETAIL_EXECUTE)))
            missing_decision_cache = (
                cache_dependent_mode
                & ~recalled_valid
                & ~stopped_rows)
            stale_decision_cache = (
                cache_dependent_mode
                & recalled_valid
                & recalled_plan_age.ge(
                    self.cognitive_compute_gate.MaxCacheAge)
                & ~stopped_rows)
            invalid_decision_cache = (
                missing_decision_cache | stale_decision_cache)
            if bool(invalid_decision_cache.any().item()):
                raise RuntimeError(
                    "active plan cache changed after cognitive scheduling")
            selected_slow_refresh_mask = self.BuildSlowRefreshMask(
                False,
                compute_decision.mode,
                self.ContractSlowCacheValid,
                compute_decision.activated_child_mask,
                stopped_rows)
            slow_refresh_mask = self.BuildSlowRefreshMask(
                step.is_train,
                compute_decision.mode,
                self.ContractSlowCacheValid,
                compute_decision.activated_child_mask,
                stopped_rows)
            slow_refresh_mask = (
                slow_refresh_mask
                | ((text_evidence_changed | pending_intent_evidence)
                   & ~stopped_rows))
            selected_slow_refresh_mask = (
                selected_slow_refresh_mask
                | ((text_evidence_changed | pending_intent_evidence)
                   & ~stopped_rows))
            slow_row_index = slow_refresh_mask.nonzero(
                as_tuple=False).flatten()
            slow_refresh = bool(slow_row_index.numel() > 0)
            live_slow_phase = None
            if slow_refresh:
                memory_bank = self.mem.ExportConsciousBank(
                    topk=BasicParameters.CONSCIOUSNESSTEM)
                world_bank = world.ExportConsciousBank(
                    topk=BasicParameters.CONSCIOUSNESSTEM)
                conscious_update = self.conscious.ForwardRows(
                    memoryBank=memory_bank,
                    worldBank=world_bank,
                    rowIndex=slow_row_index)
                selected_ocr = self.IndexRuntimeRows(
                    fuse_ocr,
                    slow_row_index,
                    batch_size)
                selected_external_text = self.IndexRuntimeRows(
                    step.text_ext,
                    slow_row_index,
                    batch_size)
                selected_text_trust = self.IndexRuntimeRows(
                    step.text_trust,
                    slow_row_index,
                    batch_size)
                selected_commitment = self.IndexRuntimeRows(
                    self.ContractIntentionCommitmentState,
                    slow_row_index,
                    batch_size)
                intent_update, sym_update, intention_update = self.intention(
                    conscious_update.self_sem,
                    conscious_update.intent_sem,
                    ocrTexts=selected_ocr,
                    extTexts=selected_external_text,
                    prioritizeExt=self.prioritize_ext_str,
                    textTrust=selected_text_trust,
                    commandVersion=command_version.index_select(
                        0, slow_row_index),
                    commitmentState=selected_commitment)
                commitment_update = (
                    {}
                    if intention_update is None
                    else intention_update.get("commitment_state", {}))
                self.ContractIntentionCommitmentState = (
                    self.ScatterRuntimeRows(
                        self.ContractIntentionCommitmentState,
                        commitment_update,
                        slow_row_index,
                        batch_size))
                intention_text_update = (
                    []
                    if intention_update is None
                    else intention_update.get("recall_texts", []))
                selected_pst_summary = pst_summary.index_select(
                    0, slow_row_index)
                task_requirements = self.goal_requirement_head(torch.cat([
                    intent_update,
                    selected_pst_summary], dim=-1))
                goal_update = self.goal_manager(
                    worldLatent=world_hzx.index_select(
                        0, slow_row_index),
                    pstSummary=selected_pst_summary,
                    intentEmbed=intent_update,
                    taskRelation=self.goal_relation_adapter(intent_update),
                    taskObject=self.goal_object_adapter(selected_pst_summary),
                    precisionRequirement=task_requirements[:, 0],
                    timeRequirement=task_requirements[:, 1],
                    terminationRequirement=task_requirements[:, 2],
                    activePerceptionRequirement=task_requirements[:, 3],
                    endpointAvailable=feedback_packet.endpoint_present.index_select(
                        0, slow_row_index),
                    hierarchyEnabled=feedback_packet.child_enabled.index_select(
                        0, slow_row_index))
                selected_pst = self.IndexRuntimeRows(
                    pst,
                    slow_row_index,
                    batch_size)
                selected_observed_pst = self.IndexRuntimeRows(
                    observed_pst,
                    slow_row_index,
                    batch_size)
                grounding_update = self.goal_grounding(
                    goal_update["goal_symbolic"],
                    intent_update,
                    selected_pst,
                    selected_observed_pst)
                phase_update = {
                    "Consciousness": conscious_update,
                    "Intention": intent_update,
                    "SymbolProbabilities": sym_update,
                    "IntentionState": intention_update,
                    "IntentionTexts": intention_text_update,
                    "Goals": goal_update,
                    "GoalGrounding": grounding_update,
                    "OcrItems": self.IndexRuntimeRows(
                        ocr_items,
                        slow_row_index,
                        batch_size),
                    "OcrTexts": selected_ocr}
                live_slow_phase = self.ScatterRuntimeRows(
                    self.ContractSlowCognitiveCache,
                    phase_update,
                    slow_row_index,
                    batch_size)
                self.ContractSlowCognitiveCache = self.DetachRuntimeObject(
                    live_slow_phase)
            slow_cache = self.ContractSlowCognitiveCache
            if slow_cache is None:
                raise RuntimeError("slow cognitive cache is unavailable")
            if step.is_train:
                if live_slow_phase is None:
                    raise RuntimeError("training slow cognition has no active rows")
                conscious_out = live_slow_phase["Consciousness"]
                intent_sem = live_slow_phase["Intention"]
                sym_probs = live_slow_phase["SymbolProbabilities"]
                intention_extras = live_slow_phase["IntentionState"]
                intention_texts = live_slow_phase["IntentionTexts"]
                goals = dict(live_slow_phase["Goals"])
                grounding = live_slow_phase["GoalGrounding"]
            else:
                conscious_out = slow_cache["Consciousness"]
                intent_sem = slow_cache["Intention"]
                intention_extras = slow_cache["IntentionState"]
                intention_texts = slow_cache["IntentionTexts"]
                goals = dict(slow_cache["Goals"])
                grounding = slow_cache["GoalGrounding"]
            if not step.is_train:
                live_short_goal = self.goal_manager.ShortGoal(
                    goals["g_ultimate"],
                    goals["g_long"],
                    goals["g_mid"],
                    pst_summary)
                live_fused_goals = self.goal_manager.FuseGoals(
                    goals["g_ultimate"],
                    goals["g_long"],
                    goals["g_mid"],
                    live_short_goal)
                goals["g_short"] = live_short_goal
                goals.update(live_fused_goals)
                grounding = self.goal_grounding(
                    goals["goal_symbolic"],
                    intent_sem,
                    pst,
                    observed_pst)
            goals["endpoint_active"] = (
                self.goal_manager.ResolveEndpointActivity(
                    endpointAvailable=feedback_packet.endpoint_present,
                    hierarchyEnabled=feedback_packet.child_enabled,
                    batchSize=batch_size,
                    device=device))
            goals["subtree_relevance"], goals["subtree_active"] = (
                self.goal_manager.BuildSubtreeState(
                    goals["endpoint_relevance"],
                    goals["endpoint_active"]))
            SaveModuleOutput("Consciousness", conscious_out)
            SaveModuleOutput("Intention", {
                "Embedding": intent_sem,
                "State": intention_extras})
            SaveModuleOutput("Goal", goals)
            intent_changed = (
                torch.zeros(batch_size, device=device, dtype=torch.bool)
                if intention_extras is None
                else intention_extras.get(
                    "intent_changed",
                    torch.zeros(
                        batch_size,
                        device=device,
                        dtype=torch.bool)) & slow_refresh_mask)
            intent_conflict_confirmed = (
                torch.zeros(batch_size, device=device, dtype=torch.bool)
                if intention_extras is None
                else intention_extras.get(
                    "intent_conflict_confirmed",
                    torch.zeros(
                        batch_size,
                        device=device,
                        dtype=torch.bool)) & slow_refresh_mask)
            confirmed_intent_event = (
                intent_changed | intent_conflict_confirmed)
            if bool(confirmed_intent_event.any().item()):
                confirmed_reason = compute_decision.reason_target.clone()
                confirmed_reason[:, 1] = (
                    confirmed_reason[:, 1] | confirmed_intent_event)
                full_mode = torch.full_like(
                    compute_decision.mode,
                    int(CognitiveComputeMode.FULL_REPLAN))
                compute_decision = CognitiveComputeDecision(
                    mode=torch.where(
                        confirmed_intent_event,
                        full_mode,
                        compute_decision.mode),
                    hard_trigger=(
                        compute_decision.hard_trigger
                        | confirmed_intent_event),
                    activated_child_mask=(
                        compute_decision.activated_child_mask),
                    evc_trigger=compute_decision.evc_trigger,
                    reason_target=confirmed_reason)
            goal_code = torch.cat([
                goals[name].argmax(dim=-1).to(dtype=torch.long)
                for name in (
                "ultimate_logits",
                "long_logits",
                "mid_logits",
                )], dim=-1)
            goal_known = self.ContractPreviousGoalCode.ge(0).all(dim=-1)
            goal_changed = goal_known & goal_code.ne(
                self.ContractPreviousGoalCode).any(dim=-1)
            reference_distribution = grounding["reference_distribution"]
            reference_index = reference_distribution.argmax(dim=-1)
            reference_known = self.ContractPreviousReferenceIndex.ge(0)
            reference_changed = (
                reference_known
                & grounding["reference_confidence"].gt(0.5)
                & reference_index.ne(self.ContractPreviousReferenceIndex))
            goal_changed = goal_changed | reference_changed
            if bool(goal_changed.any().item()):
                goal_reason = compute_decision.reason_target.clone()
                goal_reason[:, 1] = goal_reason[:, 1] | goal_changed
                full_mode = torch.full_like(
                    compute_decision.mode,
                    int(CognitiveComputeMode.FULL_REPLAN))
                compute_decision = CognitiveComputeDecision(
                    mode=torch.where(
                        goal_changed,
                        full_mode,
                        compute_decision.mode),
                    hard_trigger=(
                        compute_decision.hard_trigger | goal_changed),
                    activated_child_mask=(
                        compute_decision.activated_child_mask),
                    evc_trigger=compute_decision.evc_trigger,
                    reason_target=goal_reason)
            reference_uncertainty = -(
                reference_distribution
                * reference_distribution.clamp_min(1e-8).log()
            ).sum(dim=-1) / math.log(
                max(int(reference_distribution.size(-1)), 2))
            plan_stale = (
                compute_decision.reason_target[:, 0]
                | compute_decision.reason_target[:, 1]
                | confirmed_intent_event
                | goal_changed
                | reference_changed)
            plan_stale_known = torch.ones_like(plan_stale)

            cache_active = (
                torch.zeros(batch_size, device=device, dtype=torch.bool)
                if self.ContractCachedTarget is None
                else self.ContractCachedTarget.active.any(dim=-1))
            target_matches = (
                torch.zeros(batch_size, device=device, dtype=torch.bool)
                if self.ContractCachedTarget is None
                else feedback_packet.target_version.eq(
                    self.ContractCachedTarget.target_version))
            cache_executing = cache_active & target_matches
            progress_weight = (
                feedback_packet.target_active
                & feedback_packet.endpoint_present).to(dtype=dtype)
            planner_progress = (
                feedback_packet.progress * progress_weight
            ).sum(dim=-1) / progress_weight.sum(dim=-1).clamp_min(1.0)
            planner_reached = (
                feedback_packet.target_active.any(dim=-1)
                & (
                    feedback_packet.reached
                    | ~feedback_packet.target_active).all(dim=-1))
            planner_failed = (
                feedback_packet.target_active
                & ~feedback_packet.endpoint_present).any(dim=-1)
            planner_tracking_error = (
                (1.0 - feedback_packet.progress) * progress_weight
            ).sum(dim=-1) / progress_weight.sum(dim=-1).clamp_min(1.0)
            temporal_context = self.packed_temporal_gate.BuildContext(
                activeMask=cache_executing.to(dtype=dtype),
                actionAgeSteps=self.ContractCacheAge.to(dtype=dtype),
                noReferenceProb=grounding["no_reference_prob"],
                referenceConfidence=grounding["reference_confidence"],
                satisfactionProb=planner_progress,
                safetyRisk=risk,
                candidateSafetyRisk=risk,
                interruptRisk=torch.maximum(risk, uncertainty),
                observationFreshness=(~stopped_rows).to(dtype=dtype),
                canInterrupt=torch.ones(batch_size, device=device, dtype=dtype),
                hardStop=(
                    planner_failed
                    | done_event
                    | failsafe_event).to(dtype=dtype),
                plannerProgress=planner_progress,
                plannerTrackingError=planner_tracking_error,
                plannerExecuting=cache_executing.to(dtype=dtype),
                plannerReached=planner_reached.to(dtype=dtype),
                plannerFailed=planner_failed.to(dtype=dtype))
            temporal_goal = self.goal_manager.TemporalGoal(
                goals["goal_temporal"],
                temporal_context.feat)

            if int(realized_action.size(-1)) != int(actor.action_embed_dim):
                raise RuntimeError(
                    "contract realized action does not match Decision input width")
            decision_mask = (
                ~failsafe_event
                if step.is_train
                else ~stopped_rows)
            decision_row_index = decision_mask.nonzero(
                as_tuple=False).flatten()
            if decision_row_index.numel() < 1:
                raise RuntimeError("contract decision rows are unavailable")
            belief_prediction_state_prev = (
                self.prev_belief_prediction_state.detach())
            belief_prediction_valid_prev = (
                self.prev_belief_prediction_valid.detach())
            previous_world_hzx = torch.cat([
                self.prev_world_h,
                self.prev_world_z,
                self.prev_world_x,
            ], dim=-1).detach()
            credit_option_policy_input = (
                self.active_option_policy_input.detach())
            credit_option_prior_logit = (
                self.active_option_prior_logit.detach())
            credit_option_goal_mid = self.active_option_goal_mid.detach()
            credit_option_index = self.active_option_index.detach()
            credit_option_valid = (
                self.active_option_valid
                & target_matches
                & ~self.prev_done_flag
                & ~stopped_rows).detach()
            cached_option_skill = self.mem.RecallSkill(
                "executedOption",
                self.model_signature)
            if cached_option_skill is not None:
                cached_option_skill = cached_option_skill.to(
                    device=device,
                    dtype=dtype)
            skill_option_prior_bias, skill_relevance = (
                self.BuildOptionSkillPrior(
                    goals["g_mid"],
                    cached_option_skill,
                    actor.num_options))
            eligibility_invalid = (
                ~self.active_option_valid
                | ~target_matches
                | self.prev_done_flag
                | stopped_rows)
            actor.ClearInvalidEligibility(eligibility_invalid)
            full_decision_mask = (
                decision_mask
                & (
                    torch.ones_like(decision_mask)
                    if step.is_train
                    else compute_decision.mode.eq(
                        int(CognitiveComputeMode.FULL_REPLAN))))
            fast_decision_mask = (
                decision_mask
                & ~full_decision_mask
                & compute_decision.mode.eq(
                    int(CognitiveComputeMode.FAST_EXECUTE)))
            detail_decision_mask = (
                decision_mask
                & ~full_decision_mask
                & compute_decision.mode.eq(
                    int(CognitiveComputeMode.DETAIL_EXECUTE)))
            if not torch.equal(
                full_decision_mask
                | fast_decision_mask
                | detail_decision_mask,
                decision_mask
            ):
                raise RuntimeError("Decision compute modes do not partition rows")
            full_decision_rows = full_decision_mask.nonzero(
                as_tuple=False).flatten()
            fast_decision_rows = fast_decision_mask.nonzero(
                as_tuple=False).flatten()
            detail_decision_rows = detail_decision_mask.nonzero(
                as_tuple=False).flatten()
            feedback_td_error = (
                return_advantage
                * credit_option_valid.to(
                    dtype=return_advantage.dtype))
            decision_belief_update = actor.BuildBeliefContextRows(
                decision_row_index,
                mem_feat,
                intent_sem,
                valueTensor=critic_out.value,
                vNextTensor=critic_out.valueNext,
                uncertainty=uncertainty,
                confidence=confidence,
                precision=precision,
                risk=risk,
                worldHzx=world_hzx)
            decision_belief_context = self.ExpandRuntimeRows(
                decision_belief_update,
                decision_row_index,
                batch_size)
            live_contract_grounding = None
            live_neuro_symbolic = None
            if slow_refresh:
                contract_grounding_update = (
                    self.contract_neuro_symbolic_grounder.Ground(
                        feedback_packet.IndexSelectRows(slow_row_index),
                        returnExplain=self.save_module_messager_output,
                        planStale=plan_stale.index_select(
                            0, slow_row_index),
                        planStaleKnown=plan_stale_known.index_select(
                            0, slow_row_index)))
                cached_contract_grounding = (
                    None
                    if self.ContractSlowCognitiveCache is None
                    else self.ContractSlowCognitiveCache.get(
                        "ContractGrounding"))
                live_contract_grounding = self.ScatterRuntimeRows(
                    cached_contract_grounding,
                    contract_grounding_update,
                    slow_row_index,
                    batch_size)
                self.ContractSlowCognitiveCache = self.DetachRuntimeObject(
                    self.ScatterRuntimeRows(
                        self.ContractSlowCognitiveCache,
                        {"ContractGrounding": live_contract_grounding},
                        slow_row_index,
                        batch_size))
                neuro_symbolic_update = (
                    self.contract_neuro_symbolic.ForwardContractRows(
                        rowIndex=slow_row_index,
                        fullBatchSize=batch_size,
                        contractGrounding=live_contract_grounding,
                        pst=pst,
                        observedPst=observed_pst,
                        goalEmbed=goals["goal_symbolic"],
                        worldBelief=world_hzx,
                        decisionBelief=decision_belief_context,
                        embodimentState=contract_body["BodySummary"],
                        controlState=contract_body["ControlFeedbackFeature"],
                        uncertainty=uncertainty,
                        novelty=novelty_signal,
                        recentFailure=planner_failed.to(dtype=dtype),
                        referenceUncertainty=reference_uncertainty,
                        satisfactionProb=planner_progress,
                        referenced=grounding["referenced_object_probs"],
                        referenceConfidence=grounding[
                            "reference_confidence"],
                        noSlotProb=grounding["no_reference_prob"],
                        temporalContextFeat=temporal_context.feat,
                        returnExplain=self.save_module_messager_output))
                cached_neuro_symbolic = (
                    None
                    if self.ContractSlowCognitiveCache is None
                    else self.ContractSlowCognitiveCache.get(
                        "NeuroSymbolic"))
                live_neuro_symbolic = self.ScatterRuntimeRows(
                    cached_neuro_symbolic,
                    neuro_symbolic_update,
                    slow_row_index,
                    batch_size)
                self.ContractSlowCognitiveCache = self.DetachRuntimeObject(
                    self.ScatterRuntimeRows(
                        self.ContractSlowCognitiveCache,
                        {"NeuroSymbolic": live_neuro_symbolic},
                        slow_row_index,
                        batch_size))
                self.ContractSlowCacheValid = (
                    self.ContractSlowCacheValid.clone())
                self.ContractSlowCacheValid[slow_row_index] = True
            else:
                contract_grounding = slow_cache["ContractGrounding"]
            slow_cache = self.ContractSlowCognitiveCache
            if slow_cache is None:
                raise RuntimeError("slow cognitive cache is unavailable")
            if step.is_train:
                if (
                    live_contract_grounding is None
                    or live_neuro_symbolic is None
                ):
                    raise RuntimeError(
                        "training symbolic cognition has no active rows")
                contract_grounding = live_contract_grounding
                neuro_symbolic_out = live_neuro_symbolic
            else:
                contract_grounding = slow_cache["ContractGrounding"]
                neuro_symbolic_out = slow_cache["NeuroSymbolic"]
            SaveModuleOutput("NeuroSymbolic", neuro_symbolic_out)
            base_act_out = None
            if full_decision_rows.numel() > 0:
                full_base_update = actor.ForwardContractRows(
                    full_decision_rows,
                    stateFeat=mem_feat,
                    intentFeat=intent_sem,
                    sample=step.sample_actions,
                    deterministic=step.deterministic_actor,
                    prevOptionLogit=(
                        self.prev_option_logit
                        + skill_option_prior_bias),
                    valueTensor=critic_out.value,
                    vNextTensor=critic_out.valueNext,
                    uncertainty=uncertainty,
                    confidence=confidence,
                    precision=precision,
                    risk=risk,
                    worldHzx=world_hzx,
                    prevDecisionState=self.prev_decision_state,
                    prevLatentControl=self.prev_latent_control,
                    prevActionEmbed=realized_action,
                    prevMapperHidden=self.prev_mapper_hidden,
                    feedbackTdError=feedback_td_error)
                base_act_out = self.ScatterRuntimeRows(
                    base_act_out,
                    full_base_update,
                    full_decision_rows,
                    batch_size)
            if fast_decision_rows.numel() > 0:
                fast_base_update = actor.ForwardFastRows(
                    fast_decision_rows,
                    recalled_feature,
                    mem_feat,
                    intent_sem,
                    valueTensor=critic_out.value,
                    vNextTensor=critic_out.valueNext,
                    uncertainty=uncertainty,
                    confidence=confidence,
                    precision=precision,
                    risk=risk,
                    worldHzx=world_hzx,
                    prevDecisionState=self.prev_fast_decision_state,
                    prevLatentControl=self.prev_fast_latent_control,
                    prevActionEmbed=realized_action,
                    prevMapperHidden=self.prev_fast_mapper_hidden,
                    feedbackTdError=feedback_td_error,
                    prevOptionLogit=(
                        self.prev_fast_option_logit
                        + skill_option_prior_bias),
                    sample=False,
                    deterministic=True)
                base_act_out = self.ScatterRuntimeRows(
                    base_act_out,
                    fast_base_update,
                    fast_decision_rows,
                    batch_size)
            if detail_decision_rows.numel() > 0:
                detail_base_update = actor.ForwardDetailRows(
                    detail_decision_rows,
                    recalled_feature,
                    goals["goal_decision"],
                    mem_feat,
                    intent_sem,
                    valueTensor=critic_out.value,
                    vNextTensor=critic_out.valueNext,
                    uncertainty=uncertainty,
                    confidence=confidence,
                    precision=precision,
                    risk=risk,
                    worldHzx=world_hzx,
                    prevDecisionState=self.prev_detail_decision_state,
                    prevLatentControl=self.prev_detail_latent_control,
                    prevActionEmbed=realized_action,
                    prevMapperHidden=self.prev_detail_mapper_hidden,
                    feedbackTdError=feedback_td_error,
                    prevOptionLogit=(
                        self.prev_detail_option_logit
                        + skill_option_prior_bias),
                    sample=False,
                    deterministic=True)
                base_act_out = self.ScatterRuntimeRows(
                    base_act_out,
                    detail_base_update,
                    detail_decision_rows,
                    batch_size)
            if base_act_out is None:
                raise RuntimeError("Decision produced no active rows")
            fast_student_base = None
            detail_student_base = None
            if step.is_train:
                student_cache = torch.where(
                    recalled_valid.unsqueeze(-1),
                    recalled_feature,
                    base_act_out["belief"].detach())
                fast_student_base = actor.ForwardFastRows(
                    decision_row_index,
                    student_cache,
                    mem_feat,
                    intent_sem,
                    valueTensor=critic_out.value,
                    vNextTensor=critic_out.valueNext,
                    uncertainty=uncertainty,
                    confidence=confidence,
                    precision=precision,
                    risk=risk,
                    worldHzx=world_hzx,
                    prevDecisionState=self.prev_fast_decision_state,
                    prevLatentControl=self.prev_fast_latent_control,
                    prevActionEmbed=realized_action,
                    prevMapperHidden=self.prev_fast_mapper_hidden,
                    feedbackTdError=feedback_td_error,
                    prevOptionLogit=(
                        self.prev_fast_option_logit
                        + skill_option_prior_bias),
                    sample=False,
                    deterministic=True)
                detail_student_base = actor.ForwardDetailRows(
                    decision_row_index,
                    student_cache,
                    goals["goal_decision"],
                    mem_feat,
                    intent_sem,
                    valueTensor=critic_out.value,
                    vNextTensor=critic_out.valueNext,
                    uncertainty=uncertainty,
                    confidence=confidence,
                    precision=precision,
                    risk=risk,
                    worldHzx=world_hzx,
                    prevDecisionState=self.prev_detail_decision_state,
                    prevLatentControl=self.prev_detail_latent_control,
                    prevActionEmbed=realized_action,
                    prevMapperHidden=self.prev_detail_mapper_hidden,
                    feedbackTdError=feedback_td_error,
                    prevOptionLogit=(
                        self.prev_detail_option_logit
                        + skill_option_prior_bias),
                    sample=False,
                    deterministic=True)
            belief_prediction_delayed = actor.PredictBelief(
                belief_prediction_state_prev.index_select(
                    0, decision_row_index))
            prediction_error_update = (
                base_act_out["belief"].index_select(
                    0, decision_row_index)
                - belief_prediction_delayed)
            base_act_out["prediction_error"] = self.ExpandRuntimeRows(
                prediction_error_update,
                decision_row_index,
                batch_size)
            world_abstract = world.BuildWorldAbstract(
                w_out,
                pst,
                pst_summary,
                uncertainty,
                confidence)
            selected_neuro_symbolic = self.IndexRuntimeRows(
                neuro_symbolic_out,
                decision_row_index,
                batch_size)
            selected_world_hzx = world_abstract[
                "world_hzx"].index_select(0, decision_row_index)
            selected_abstract_feature = world_abstract[
                "abstract_feat"].index_select(0, decision_row_index)
            selected_world_abstract = self.FuseWorldAbstractDecision(
                selected_world_hzx,
                selected_abstract_feature)
            selected_scene_abstract = world_abstract[
                "pst_summary"].index_select(0, decision_row_index)
            selected_goal_decision = goals[
                "goal_decision"].index_select(0, decision_row_index)
            selected_temporal_goal = temporal_goal.index_select(
                0, decision_row_index)
            selected_body_state = contract_body[
                "BodySummary"].index_select(0, decision_row_index)
            selected_control_feedback = contract_body[
                "ControlFeedbackFeature"].index_select(
                    0, decision_row_index)
            selected_embodiment_context = contract_body[
                "EmbodimentContextFeature"].index_select(
                    0, decision_row_index)
            base_act_update = self.IndexRuntimeRows(
                base_act_out,
                decision_row_index,
                batch_size)
            act_update = actor.RefineWithNeuroSymbolic(
                base_act_update,
                selected_neuro_symbolic,
                selected_world_abstract,
                selected_scene_abstract,
                selected_goal_decision,
                selected_temporal_goal,
                selected_body_state,
                selected_control_feedback,
                selected_embodiment_context)
            act_out = self.ExpandRuntimeRows(
                act_update,
                decision_row_index,
                batch_size)
            fast_student_out = None
            detail_student_out = None
            if step.is_train:
                fast_student_update = actor.RefineWithNeuroSymbolic(
                    fast_student_base,
                    selected_neuro_symbolic,
                    selected_world_abstract,
                    selected_scene_abstract,
                    selected_goal_decision,
                    selected_temporal_goal,
                    selected_body_state,
                    selected_control_feedback,
                    selected_embodiment_context)
                detail_student_update = actor.RefineWithNeuroSymbolic(
                    detail_student_base,
                    selected_neuro_symbolic,
                    selected_world_abstract,
                    selected_scene_abstract,
                    selected_goal_decision,
                    selected_temporal_goal,
                    selected_body_state,
                    selected_control_feedback,
                    selected_embodiment_context)
                fast_student_out = self.ExpandRuntimeRows(
                    fast_student_update,
                    decision_row_index,
                    batch_size)
                detail_student_out = self.ExpandRuntimeRows(
                    detail_student_update,
                    decision_row_index,
                    batch_size)
            SaveModuleOutput("Decision", act_out)

            decision_context = self.BuildContractDecisionContext(
                act_out,
                feedback_packet,
                risk,
                confidence,
                precision,
                slotRelevance=goals["subtree_relevance"],
                slotSelectionMask=None,
                preserveReachedTargets=(
                    compute_decision.mode.ne(
                        int(CognitiveComputeMode.FULL_REPLAN))
                    & compute_decision.mode.ne(
                        int(CognitiveComputeMode.FAILSAFE))))
            network_decision_feature = act_out["decision_feature"]
            selected_decision_feature = network_decision_feature
            cached_decision_feature = torch.where(
                recalled_valid.unsqueeze(-1),
                recalled_feature,
                network_decision_feature.detach())
            cached_decision_valid = recalled_valid
            plan_cache_reused = (
                recalled_valid
                & decision_mask
                & (
                    compute_decision.mode.eq(
                        int(CognitiveComputeMode.FAST_EXECUTE))
                    | compute_decision.mode.eq(
                        int(CognitiveComputeMode.DETAIL_EXECUTE))))
            planner_prior = None
            selected_planner_mask = self.BuildPlannerMask(
                False,
                compute_decision.mode,
                stopped_rows)
            if self.planner is None:
                selected_planner_mask = torch.zeros_like(
                    selected_planner_mask)
            planner_mask = self.BuildPlannerMask(
                step.is_train,
                compute_decision.mode,
                stopped_rows)
            planner_row_index = planner_mask.nonzero(
                as_tuple=False).flatten()
            planner_required = (
                self.planner is not None
                and planner_row_index.numel() > 0)
            planner_valid_mask = torch.zeros(
                batch_size,
                device=device,
                dtype=torch.bool)
            if planner_required:
                with torch.no_grad():
                    candidate_evaluator = (
                        self.BuildContractDecisionFeatureEvaluator(
                            h0=w_out["h_next"].detach().index_select(
                                0, planner_row_index),
                            z0=w_out["z_next"].detach().index_select(
                                0, planner_row_index),
                            x0=w_out["x_next"].detach().index_select(
                                0, planner_row_index),
                            physicalState=self.IndexRuntimeRows(
                                pst,
                                planner_row_index,
                                batch_size),
                            embodimentState=(
                                world_embodiment_state.index_select(
                                    0, planner_row_index)),
                            feedbackPacket=(
                                feedback_packet.IndexSelectRows(
                                    planner_row_index)),
                            decisionContext=(
                                decision_context.IndexSelectRows(
                                    planner_row_index)),
                            activePerceptionRequirement=goals[
                                "active_perception_requirement"].index_select(
                                    0, planner_row_index)))
                    planner_update = self.planner.Plan(
                        decisionFeature=(
                            network_decision_feature.detach().index_select(
                                0, planner_row_index)),
                        candidateEvaluator=candidate_evaluator,
                        returnDiagnostics=self.planner_teacher_mode)
                    planner_prior = self.ExpandRuntimeRows(
                        planner_update,
                        planner_row_index,
                        batch_size)
                    planner_valid_mask = (
                        planner_mask
                        & planner_prior["valid"].to(dtype=torch.bool))
            if planner_prior is not None and self.use_planner:
                selected_decision_feature = torch.where(
                    planner_valid_mask.unsqueeze(-1),
                    planner_prior["decision_feature"].detach(),
                    selected_decision_feature)
            prospective_value = self.RuntimeModule(
                self.critic).BuildCognitiveValue(
                    critic_out.valueHidden,
                    selected_decision_feature)
            packed_update = self.packed_decision_decoupler.DecodeContract(
                selected_decision_feature.index_select(
                    0, decision_row_index),
                feedbackPacket=feedback_packet.IndexSelectRows(
                    decision_row_index),
                decisionContext=decision_context.IndexSelectRows(
                    decision_row_index))
            packed_decision = self.ExpandRuntimeRows(
                packed_update,
                decision_row_index,
                batch_size)
            packed_decision = self.ApplyFailsafeDecision(
                packed_decision,
                stopped_rows)
            SaveModuleOutput("DecisionDecoupler", packed_decision)

            neutral_target = PackedEndEffectorTarget(
                values=torch.zeros(
                    batch_size,
                    self.robot_contract_view.end_effector_target_layout.PackedDim,
                    device=device,
                    dtype=dtype),
                active=torch.zeros(
                    batch_size,
                    self.robot_contract_view.end_effector_count,
                    device=device,
                    dtype=torch.bool),
                contract_id=self.robot_contract_view.contract_id,
                model_signature=self.robot_contract_view.model_signature,
                target_version=feedback_packet.target_version + 1,
                timestamp=feedback_packet.timestamp)
            cached_target = (
                neutral_target
                if self.ContractCachedTarget is None
                else self.ContractCachedTarget)
            temporal_readout = act_out["temporal_decision"]
            temporal_proposal = PackedTemporalProposal(
                kind_scores=temporal_readout["kind_logits"],
                same_operator=temporal_readout["same_operator"],
                operator_changed=temporal_readout["operator_changed"],
                invoke_delta=temporal_readout["invoke_delta"],
                reference_drift=temporal_readout["reference_drift"],
                redispatch_score=temporal_readout["redispatch_score"],
                interrupt_score=temporal_readout["p_interrupt"],
                duration_ms=temporal_readout["duration_ms"],
                soft_timeout_ms=temporal_readout["soft_timeout_ms"],
                hard_timeout_ms=temporal_readout["hard_timeout_ms"],
                action_epoch=self.ContractCachedActionEpoch)
            temporal_events = PackedTemporalEvent(
                cache_executing=cache_executing,
                candidate_ready=packed_decision.target.active.any(dim=-1),
                redispatch_requested=(
                    temporal_readout["redispatch_score"] > 0.5),
                cancel_requested=(
                    temporal_readout["p_interrupt"] > 0.5),
                planner_failed=planner_failed,
                plan_reached=planner_reached,
                hard_stop=(
                    planner_failed
                    | done_event
                    | failsafe_event),
                active_risk=risk.clamp(0.0, 1.0),
                candidate_risk=risk.clamp(0.0, 1.0))
            temporal_decision = self.packed_temporal_gate.Step(
                feedback_packet,
                packed_decision.target,
                cached_target,
                temporal_context,
                temporal_proposal,
                temporal_events,
                temporal_readout["invoke_delta"])
            temporal_decision = self.ApplyStoppedTemporalDecision(
                temporal_decision,
                stopped_rows)
            SaveModuleOutput("TemporalExecution", temporal_decision)
            option_start = temporal_decision.candidate_selected
            option_continue = temporal_decision.cache_selected
            option_inactive = ~(option_start | option_continue)
            actor.CommitEligibilityRows(
                act_out["eligibility"]["pre"],
                act_out["eligibility"]["post"],
                option_start,
                full_decision_mask)
            self.active_option_policy_input = torch.where(
                option_start.unsqueeze(-1),
                base_act_out["option"]["policy_input"].detach(),
                self.active_option_policy_input)
            self.active_option_prior_logit = torch.where(
                option_start.unsqueeze(-1),
                base_act_out["option"]["prior_logits"].detach(),
                self.active_option_prior_logit)
            self.active_option_goal_mid = torch.where(
                option_start.unsqueeze(-1),
                goals["g_mid"].detach(),
                self.active_option_goal_mid)
            self.active_option_index = torch.where(
                option_start,
                base_act_out["option"]["opt_idx"].detach(),
                self.active_option_index)
            self.active_option_index = torch.where(
                option_inactive,
                torch.zeros_like(self.active_option_index),
                self.active_option_index)
            self.active_option_policy_input = torch.where(
                option_inactive.unsqueeze(-1),
                torch.zeros_like(self.active_option_policy_input),
                self.active_option_policy_input)
            self.active_option_prior_logit = torch.where(
                option_inactive.unsqueeze(-1),
                torch.zeros_like(self.active_option_prior_logit),
                self.active_option_prior_logit)
            self.active_option_goal_mid = torch.where(
                option_inactive.unsqueeze(-1),
                torch.zeros_like(self.active_option_goal_mid),
                self.active_option_goal_mid)
            self.active_option_valid = (
                option_start
                | (option_continue & self.active_option_valid))
            fast_state_out = (
                fast_student_out
                if step.is_train
                else act_out)
            detail_state_out = (
                detail_student_out
                if step.is_train
                else act_out)
            if fast_state_out is None or detail_state_out is None:
                raise RuntimeError("Decision student state is unavailable")
            full_candidate_selected = option_start & full_decision_mask
            fast_candidate_selected = option_start & (
                decision_mask if step.is_train else fast_decision_mask)
            detail_candidate_selected = option_start & (
                decision_mask if step.is_train else detail_decision_mask)
            scheduled_prev_option_logit = self.ScheduleExecutedState(
                act_out["prevOptionLogit_next"].detach(),
                self.prev_option_logit,
                full_candidate_selected,
                option_continue)
            scheduled_prev_fast_option_logit = self.ScheduleExecutedState(
                fast_state_out["prevOptionLogit_next"].detach(),
                self.prev_fast_option_logit,
                fast_candidate_selected,
                option_continue)
            scheduled_prev_detail_option_logit = self.ScheduleExecutedState(
                detail_state_out["prevOptionLogit_next"].detach(),
                self.prev_detail_option_logit,
                detail_candidate_selected,
                option_continue)
            act_out["candidate_option_index"] = base_act_out[
                "option"]["opt_idx"]
            act_out["scheduled_option_index"] = self.active_option_index
            act_out["scheduled_option_valid"] = self.active_option_valid
            act_out["credited_option_index"] = credit_option_index
            act_out["credited_option_valid"] = credit_option_valid
            act_out["skill_prior_bias"] = skill_option_prior_bias
            act_out["skill_relevance"] = skill_relevance
            act_out["option_assignment"] = {
                "candidate_index": base_act_out["option"]["opt_idx"],
                "candidate_selected": option_start,
                "scheduled_index": self.active_option_index,
                "scheduled_valid": self.active_option_valid,
                "continued": option_continue,
                "credited_index": credit_option_index,
                "credited_valid": credit_option_valid}
            SaveModuleOutput("Decision", act_out)

            selected_efference = (
                self.packed_decision_decoupler
                .DecodePerceptionRotationEfference(
                    temporal_decision.selected_target,
                    feedback_packet))
            prospective_rotation, prospective_rotation_valid = (
                self.SelectContractPerceptionRotation(
                    feedback_packet,
                    rotationDelta=selected_efference.rotation_delta,
                    rotationPresent=selected_efference.present))
            cached_world_action = self.packed_decision_decoupler.EncodeWorldAction(
                cached_target)
            prospective_action = torch.where(
                temporal_decision.candidate_selected.unsqueeze(-1),
                packed_decision.world_action_feature,
                torch.where(
                    temporal_decision.cache_selected.unsqueeze(-1),
                    cached_world_action,
                    torch.zeros_like(packed_decision.world_action_feature)))
            prospective_visual_prediction = (
                world.PredictNextVisualFromPosterior(
                    w_out["h_next"].detach(),
                    w_out["z_next"].detach(),
                    w_out["x_next"].detach(),
                    physicalState=pst,
                    actionEnc=prospective_action,
                    embodimentState=world_embodiment_state,
                    observerMotion=prospective_rotation,
                    observerMotionValid=prospective_rotation_valid,
                    sample=False))

            losses: Dict[str, torch.Tensor] = {}
            if step.is_train:
                loss_sample_mask = ~failsafe_event
                loss_row_index = loss_sample_mask.nonzero(
                    as_tuple=False).flatten()
                world_loss = w_out["loss"]
                memory_loss = self.mem.GetInternalLoss()
                critic_loss = (
                    world_loss.new_zeros(())
                    if critic_out.loss is None
                    else critic_out.loss)
                critic_current_loss = critic_loss
                critic_transport_delayed_loss = world_loss.new_zeros(())
                if critic_out.extras is not None:
                    critic_current_loss = critic_out.extras.get(
                        "loss_current_graph",
                        critic_current_loss)
                    critic_transport_delayed_loss = critic_out.extras.get(
                        "loss_transport_delayed_graph",
                        critic_transport_delayed_loss)
                conscious_loss = world_loss.new_zeros(())
                if conscious_out.extras is not None:
                    conscious_loss = conscious_out.extras.get(
                        "loss",
                        conscious_loss)
                intention_loss = world_loss.new_zeros(())
                loss_sym_probs = sym_probs.index_select(
                    0,
                    slow_row_index)
                if loss_sym_probs is not None:
                    intention_loss, _ = self.intention.GetInternalLoss(
                        loss_sym_probs)
                perception_loss = self.perc.ComputePerceptionLoss(
                    visual_state,
                    depthTarget=step.depth,
                    depthTargetValid=step.depth_valid,
                    observerRotation=rotation_delta,
                    prevVisualValid=previous_visual_valid,
                    prevVisualState=previous_visual_state,
                    observerRotationValid=rotation_valid,
                    sampleMask=loss_sample_mask)
                perception_recall_losses: Dict[str, torch.Tensor] = {}
                if (
                    self.perception_recall_loss is not None
                    and "node_valid" in step.perception_targets
                ):
                    recall_output = self.RuntimeModule(
                        self.perc).recall_heads(visual_state)
                    perception_recall_losses = self.perception_recall_loss(
                        self.IndexRuntimeRows(
                            recall_output,
                            loss_row_index,
                            batch_size),
                        self.IndexRuntimeRows(
                            step.perception_targets,
                            loss_row_index,
                            batch_size))
                    perception_loss = (
                        perception_loss
                        + perception_recall_losses["loss"])
                physical_loss = world_loss.new_zeros(())
                physical_losses: Dict[str, torch.Tensor] = {}
                if "node_valid" in step.perception_targets:
                    physical_losses = self.pst_loss(
                        self.IndexRuntimeRows(
                            observed_pst,
                            loss_row_index,
                            batch_size),
                        self.IndexRuntimeRows(
                            step.perception_targets,
                            loss_row_index,
                            batch_size))
                    physical_loss = physical_losses["loss"]
                prediction_error_per_row = (
                    visual_state.PredErrorToken.square().flatten(1).mean(
                        dim=-1))
                prediction_error_weight = loss_sample_mask.to(dtype=dtype)
                prediction_error_loss = (
                    prediction_error_per_row * prediction_error_weight
                ).sum() / prediction_error_weight.sum().clamp_min(1.0)
                detail_distill_per_row = F.smooth_l1_loss(
                    attention_extras["local_detail_readout"],
                    attention_extras["full_temporal_readout"].detach(),
                    reduction="none").mean(dim=-1)
                detail_distill_weight = (
                    local_detail.any(dim=-1)
                    & ~stopped_rows).to(dtype=dtype)
                attention_detail_distill_loss = (
                    detail_distill_per_row * detail_distill_weight
                ).sum() / detail_distill_weight.sum().clamp_min(1.0)
                attention_module = self.RuntimeModule(self.attn)
                fast_attention_distill_loss = (
                    attention_module.StudentDistillationLoss(
                        attention_extras["fast_student_attention"],
                        atten_out,
                        compute_decision.mode.eq(
                            int(CognitiveComputeMode.FAST_EXECUTE))
                        & ~stopped_rows))
                detail_attention_distill_loss = (
                    attention_module.StudentDistillationLoss(
                        attention_extras["detail_student_attention"],
                        atten_out,
                        compute_decision.mode.eq(
                            int(CognitiveComputeMode.DETAIL_EXECUTE))
                        & ~stopped_rows))
                world_prediction_loss = world_loss.new_zeros(())
                world_prediction_losses: Dict[str, torch.Tensor] = {}
                alive_prediction_mask = (
                    ~self.prev_done_flag
                    & previous_visual_valid
                    & loss_sample_mask)
                if bool(alive_prediction_mask.any().item()):
                    prediction_model = (
                        self.world
                        if self.is_online_learning
                        else world)
                    previous_prediction = (
                        prediction_model.PredictNextVisualFromPosterior(
                            self.prev_world_h,
                            self.prev_world_z,
                            self.prev_world_x,
                            physicalState=previous_physical_state,
                            actionEnc=realized_action,
                            embodimentState=previous_embodiment_state,
                            observerMotion=rotation_delta,
                            observerMotionValid=rotation_valid,
                            sample=False))
                    world_prediction_losses = (
                        prediction_model.ComputePredictionLoss(
                        predictedVisual=previous_prediction[
                            "predicted_visual"],
                        reconstructedVisualState=previous_prediction[
                            "reconstructed_visual_state"],
                        targetVisualState=visual_state,
                        precision=precision,
                        sampleMask=alive_prediction_mask))
                    world_prediction_loss = world_prediction_losses[
                        "loss_pred_total"]
                world_delta = (
                    world_hzx.detach() - previous_world_hzx
                ) * (~self.prev_done_flag).to(dtype=dtype).unsqueeze(-1)
                credited_goal_progress = self.goal_manager.ProjectedProgress(
                    world_delta,
                    credit_option_goal_mid)
                realized_information_gain = information_gain.detach()
                credited_advantage = (
                    return_advantage
                    + 0.1 * credited_goal_progress
                    + 0.05 * realized_information_gain).detach()
                credit_option_weight = credit_option_valid.to(dtype=dtype)
                actor_learning_mask = ~stopped_rows
                actor_learning_weight = actor_learning_mask.to(dtype=dtype)
                actor_loss = -1e-3 * (
                    base_act_out["entropy"] * actor_learning_weight
                ).sum() / actor_learning_weight.sum().clamp_min(1.0)
                if bool(credit_option_valid.any().item()):
                    credited_log_probability = actor.OptionLogProb(
                        credit_option_policy_input,
                        credit_option_prior_logit,
                        credit_option_index)
                    actor_loss = actor_loss - (
                        credited_advantage
                        * credited_log_probability
                        * credit_option_weight
                    ).sum() / credit_option_weight.sum()
                goal_progress_weight = (
                    credit_option_valid
                    & ~self.prev_done_flag).to(dtype=dtype)
                goal_progress_loss = world_loss.new_zeros(())
                if bool((goal_progress_weight > 0.0).any().item()):
                    goal_progress_loss = -(
                        credited_goal_progress * goal_progress_weight
                    ).sum() / goal_progress_weight.sum()
                decision_prediction_loss = world_loss.new_zeros(())
                prediction_weight = (
                    belief_prediction_valid_prev
                    & ~self.prev_done_flag
                    & ~stopped_rows).to(dtype=dtype) * precision.detach()
                if bool((prediction_weight > 0.0).any().item()):
                    prediction_per_row = F.smooth_l1_loss(
                        belief_prediction_delayed,
                        base_act_out["belief"].detach(),
                        reduction="none").mean(dim=-1)
                    decision_prediction_loss = (
                        prediction_per_row * prediction_weight
                    ).sum() / prediction_weight.sum()
                fast_distillation = actor.FastDistillationLoss(
                    fast_student_out,
                    act_out,
                    compute_decision.mode.eq(
                        int(CognitiveComputeMode.FAST_EXECUTE))
                    & ~stopped_rows)
                detail_distillation = actor.FastDistillationLoss(
                    detail_student_out,
                    act_out,
                    compute_decision.mode.eq(
                        int(CognitiveComputeMode.DETAIL_EXECUTE))
                    & ~stopped_rows)
                fast_decision_distill_loss = fast_distillation["total"]
                detail_decision_distill_loss = detail_distillation["total"]
                planner_distill_loss = world_loss.new_zeros(())
                decision_energy_loss = world_loss.new_zeros(())
                if planner_prior is not None and self.planner_teacher_mode:
                    stable = (
                        compute_decision.mode.eq(
                            int(CognitiveComputeMode.FAST_EXECUTE))
                        & ~stopped_rows
                        & planner_valid_mask).to(dtype=dtype)
                    convergence = torch.exp(
                        -planner_prior["diagnostics"]["std"].detach().mean(
                            dim=-1))
                    weight = (
                        stable
                        * precision.detach()
                        * (1.0 - risk.detach())
                        * convergence)
                    if bool((weight > 0.0).any().item()):
                        distill = F.smooth_l1_loss(
                            network_decision_feature,
                            planner_prior["decision_feature"].detach(),
                            reduction="none").mean(dim=-1)
                        planner_distill_loss = (
                            distill * weight).sum() / weight.sum()
                        energy = F.smooth_l1_loss(
                            act_out["decision_energy"],
                            -planner_prior["expected_return"].detach(),
                            reduction="none")
                        decision_energy_loss = (
                            energy * weight).sum() / weight.sum()
                symbolic_target = self.RuntimeModule(
                    self.contract_neuro_symbolic).InvocationNeedTarget(
                        torch.maximum(
                            uncertainty,
                            base_act_out["decision_uncertainty"].detach()),
                        novelty_signal.detach(),
                        planner_failed.to(dtype=dtype),
                        reference_uncertainty.detach(),
                        planner_progress.detach(),
                        grounding["no_reference_prob"].detach())
                symbolic_invocation_per_row = F.binary_cross_entropy(
                    neuro_symbolic_out.invoke_mask,
                    symbolic_target.detach(),
                    reduction="none").reshape(batch_size, -1).mean(dim=-1)
                symbolic_invocation_weight = loss_sample_mask.to(
                    dtype=dtype)
                symbolic_invocation_loss = (
                    symbolic_invocation_per_row
                    * symbolic_invocation_weight
                ).sum() / symbolic_invocation_weight.sum().clamp_min(1.0)
                symbolic_sparsity_weight = (
                    (1.0 - symbolic_target.detach())
                    * symbolic_invocation_weight)
                symbolic_sparsity_loss = (
                    neuro_symbolic_out.invoke_mask
                    * symbolic_sparsity_weight
                ).sum() / symbolic_sparsity_weight.sum().clamp_min(1.0)
                grounding_loss = grounding["grounding_consistency_loss"]
                codebook_util_loss = (
                    self.goal_manager.ultimate_head.UtilizationLoss(
                        goals["ultimate_logits"].index_select(
                            0,
                            loss_row_index))
                    + self.goal_manager.long_head.UtilizationLoss(
                        goals["long_logits"].index_select(
                            0,
                            loss_row_index))
                    + self.goal_manager.mid_head.UtilizationLoss(
                        goals["mid_logits"].index_select(
                            0,
                            loss_row_index)))
                root_mask = torch.tensor(
                    tuple(
                        root and not independent
                        for root, independent in zip(
                            self.robot_contract_view.root_mask,
                            self.robot_contract_view.independent_mask)),
                    device=device,
                    dtype=dtype).unsqueeze(0)
                detail_mask = torch.tensor(
                    self.robot_contract_view.child_mask,
                    device=device,
                    dtype=dtype).unsqueeze(0)
                valid_progress = (
                    feedback_packet.endpoint_present
                    & feedback_packet.target_active
                    & feedback_packet.child_enabled).to(dtype=dtype)
                coarse_weight = valid_progress * root_mask
                detail_weight = valid_progress * detail_mask
                coarse_progress = (
                    feedback_packet.progress * coarse_weight
                ).sum(dim=-1) / coarse_weight.sum(dim=-1).clamp_min(1.0)
                detail_progress = (
                    feedback_packet.progress * detail_weight
                ).sum(dim=-1) / detail_weight.sum(dim=-1).clamp_min(1.0)
                enabled_weight = (
                    feedback_packet.child_enabled
                    & feedback_packet.endpoint_present).to(dtype=dtype)
                coarse_supervision = coarse_weight.sum(dim=-1).gt(0.0).to(
                    dtype=dtype) * loss_sample_mask.to(dtype=dtype)
                detail_supervision = detail_weight.sum(dim=-1).gt(0.0).to(
                    dtype=dtype) * loss_sample_mask.to(dtype=dtype)
                plan_staleness_target = torch.maximum(
                    torch.maximum(
                        self.ContractSlowCacheAge
                        / float(self.cognitive_compute_gate.MaxCacheAge),
                        recalled_plan_age.to(dtype=dtype)
                        / float(self.cognitive_compute_gate.MaxCacheAge)
                    ).clamp(0.0, 1.0),
                    plan_stale.to(dtype=dtype))
                coarse_progress_loss = F.smooth_l1_loss(
                    critic_out.cognitiveValue.coarseProgress,
                    coarse_progress.detach(),
                    reduction="none")
                coarse_progress_loss = (
                    coarse_progress_loss * coarse_supervision
                ).sum() / coarse_supervision.sum().clamp_min(1.0)
                detail_progress_loss = F.smooth_l1_loss(
                    critic_out.cognitiveValue.detailProgress,
                    detail_progress.detach(),
                    reduction="none")
                detail_progress_loss = (
                    detail_progress_loss * detail_supervision
                ).sum() / detail_supervision.sum().clamp_min(1.0)
                plan_staleness_loss = F.smooth_l1_loss(
                    critic_out.cognitiveValue.planStaleness,
                    plan_staleness_target.detach(),
                    reduction="none")
                plan_staleness_loss = (
                    plan_staleness_loss * (~stopped_rows).to(dtype=dtype)
                ).sum() / (~stopped_rows).to(dtype=dtype).sum().clamp_min(1.0)
                feasibility_loss = world_loss.new_zeros(())
                safety_constraint_loss = world_loss.new_zeros(())
                cognitive_value_loss = (
                    coarse_progress_loss
                    + detail_progress_loss
                    + plan_staleness_loss
                    + feasibility_loss
                    + safety_constraint_loss)
                replan_target_valid = (
                    cached_decision_valid
                    & decision_mask
                    & ~stopped_rows)
                replan_benefit_target = torch.zeros_like(
                    prospective_value.replanBenefit)
                replan_target_rows = replan_target_valid.nonzero(
                    as_tuple=False).flatten()
                if replan_target_rows.numel() > 0:
                    with torch.no_grad():
                        replan_evaluator = (
                            self.BuildContractDecisionFeatureEvaluator(
                                h0=w_out["h_next"].detach().index_select(
                                    0, replan_target_rows),
                                z0=w_out["z_next"].detach().index_select(
                                    0, replan_target_rows),
                                x0=w_out["x_next"].detach().index_select(
                                    0, replan_target_rows),
                                physicalState=self.IndexRuntimeRows(
                                    pst,
                                    replan_target_rows,
                                    batch_size),
                                embodimentState=(
                                    world_embodiment_state.index_select(
                                        0, replan_target_rows)),
                                feedbackPacket=(
                                    feedback_packet.IndexSelectRows(
                                        replan_target_rows)),
                                decisionContext=(
                                    decision_context.IndexSelectRows(
                                        replan_target_rows)),
                                activePerceptionRequirement=goals[
                                    "active_perception_requirement"
                                ].index_select(0, replan_target_rows)))
                        replan_candidates = torch.stack([
                            selected_decision_feature.index_select(
                                0, replan_target_rows),
                            cached_decision_feature.index_select(
                                0, replan_target_rows),
                        ], dim=1).reshape(
                            2 * int(replan_target_rows.numel()),
                            -1)
                        (
                            replan_utilities,
                            replan_candidate_validity,
                        ) = replan_evaluator(replan_candidates)
                        replan_utilities = replan_utilities.view(-1, 2)
                        replan_candidate_validity = (
                            replan_candidate_validity.view(-1, 2))
                        (
                            replan_target_update,
                            replan_valid_update,
                        ) = self.CounterfactualReplanBenefitSupervision(
                            replan_utilities,
                            replan_candidate_validity)
                        replan_benefit_target.index_copy_(
                            0,
                            replan_target_rows,
                            replan_target_update)
                        replan_target_valid.index_copy_(
                            0,
                            replan_target_rows,
                            replan_valid_update)
                counterfactual_evc_loss = self.CounterfactualEvcLoss(
                    prospective_value.replanBenefit,
                    prospective_value.computeCost,
                    replan_benefit_target,
                    replan_target_valid)
                compute_cost_target = self.CognitiveComputeLoadTarget(
                    attention_extras[
                        (
                            "selected_normalized_compute_fraction"
                            if step.is_train
                            else "actual_normalized_compute_fraction"
                        )].to(
                            device=device,
                            dtype=dtype),
                    attention_extras[
                        "world_value_posterior_passes"].to(
                            device=device,
                            dtype=dtype),
                    selected_slow_refresh_mask,
                    compute_decision.mode.eq(
                        int(CognitiveComputeMode.FAST_EXECUTE))
                    & ~stopped_rows,
                    compute_decision.mode.eq(
                        int(CognitiveComputeMode.DETAIL_EXECUTE))
                    & ~stopped_rows,
                    compute_decision.mode.eq(
                        int(CognitiveComputeMode.FULL_REPLAN))
                    & ~stopped_rows,
                    selected_planner_mask,
                    stopped_rows)
                compute_cost_per_row = F.smooth_l1_loss(
                    prospective_value.computeCost,
                    compute_cost_target,
                    reduction="none")
                compute_cost_weight = (~stopped_rows).to(dtype=dtype)
                compute_cost_calibration_loss = (
                    compute_cost_per_row * compute_cost_weight
                ).sum() / compute_cost_weight.sum().clamp_min(1.0)
                compute_value_loss = (
                    counterfactual_evc_loss
                    + compute_cost_calibration_loss)
                event_reason_target = compute_decision.reason_target.to(
                    dtype=dtype)
                event_reason_logits = prospective_value.eventReasonLogits
                if tuple(event_reason_logits.shape) != tuple(
                    event_reason_target.shape
                ):
                    raise RuntimeError(
                        "compute reason prediction does not match its schema")
                event_reason_per_row = F.binary_cross_entropy_with_logits(
                    event_reason_logits,
                    event_reason_target,
                    reduction="none").mean(dim=-1)
                event_loss_weight = loss_sample_mask.to(dtype=dtype)
                event_reason_loss = (
                    event_reason_per_row * event_loss_weight
                ).sum() / event_loss_weight.sum().clamp_min(1.0)
                event_probability = (
                    1.0
                    - (1.0 - torch.sigmoid(event_reason_logits)).prod(
                        dim=-1)).clamp(1e-6, 1.0 - 1e-6)
                event_per_row = F.binary_cross_entropy(
                    event_probability,
                    event_reason_target.any(dim=-1).to(dtype=dtype),
                    reduction="none")
                event_loss = (
                    event_per_row * event_loss_weight
                ).sum() / event_loss_weight.sum().clamp_min(1.0)
                temporal_kind_loss = world_loss.new_zeros(())
                if "temporal_kind" in step.perception_targets:
                    temporal_kind_loss = (
                        self.ComputeTemporalKindSupervisionLoss(
                            temporal_decision.execution_kind_scores,
                            step.perception_targets["temporal_kind"],
                            step.perception_targets[
                                "temporal_kind_valid"]
                            & actor_learning_mask,
                            temporal_context.active_mask,
                            temporal_decision.override_applied))
                temporal_duration_loss = world_loss.new_zeros(())
                if "temporal_duration_ms" in step.perception_targets:
                    duration_valid = step.perception_targets[
                        "temporal_duration_valid"].to(
                            dtype=torch.bool) & actor_learning_mask
                    if bool(duration_valid.any().item()):
                        duration_error = F.smooth_l1_loss(
                            temporal_decision.duration_ms / 1000.0,
                            step.perception_targets[
                                "temporal_duration_ms"] / 1000.0,
                            reduction="none")
                        duration_weight = duration_valid.to(dtype=dtype)
                        temporal_duration_loss = (
                            duration_error * duration_weight
                        ).sum() / duration_weight.sum()
                action_learning_rows = actor_learning_mask.nonzero(
                    as_tuple=False).flatten()
                hierarchy_loss = world_loss.new_zeros(())
                safety_loss = world_loss.new_zeros(())
                legality_loss = world_loss.new_zeros(())
                safety_prediction_loss = world_loss.new_zeros(())
                selection_loss = world_loss.new_zeros(())
                target_continuity_loss = world_loss.new_zeros(())
                if action_learning_rows.numel() > 0:
                    continuation_supervision = (
                        option_continue
                        & target_matches
                        & cache_executing)
                    constraint_losses = (
                        self.packed_decision_decoupler.TrainingConstraintLoss(
                            self.IndexRuntimeRows(
                                packed_decision,
                                action_learning_rows,
                                batch_size),
                            feedback_packet.IndexSelectRows(
                                action_learning_rows),
                            decision_context.IndexSelectRows(
                                action_learning_rows),
                            continuationMask=(
                                continuation_supervision.index_select(
                                    0,
                                    action_learning_rows))))
                    hierarchy_loss = constraint_losses["hierarchy"]
                    safety_loss = constraint_losses["safety"]
                    legality_loss = constraint_losses["legality"]
                    safety_prediction_loss = constraint_losses[
                        "safety_prediction"]
                    selection_loss = constraint_losses["selection"]
                    target_continuity_loss = constraint_losses[
                        "target_continuity"]
                policy_auxiliary_loss = (
                    0.05 * symbolic_invocation_loss
                    + 0.01 * symbolic_sparsity_loss
                    + 0.05 * grounding_loss
                    + 0.01 * codebook_util_loss
                    + 0.05 * cognitive_value_loss
                    + 0.05 * compute_value_loss
                    + 0.05 * event_loss
                    + 0.05 * event_reason_loss
                    + 0.05 * hierarchy_loss
                    + 0.05 * safety_loss
                    + 0.05 * legality_loss
                    + 0.05 * safety_prediction_loss
                    + 0.05 * selection_loss
                    + 0.05 * target_continuity_loss
                    + 0.05 * goal_progress_loss
                    + 0.05 * decision_prediction_loss
                    + 0.05 * fast_decision_distill_loss
                    + 0.05 * detail_decision_distill_loss
                    + 0.5 * temporal_kind_loss
                    + 0.05 * temporal_duration_loss)
                world_optimization_loss = (
                    world_loss
                    + 0.05 * world_prediction_loss
                    + 0.05 * prediction_error_loss)
                critic_optimization_loss = (
                    critic_current_loss
                    + 0.05 * cognitive_value_loss
                    + 0.05 * compute_value_loss
                    + 0.05 * event_loss
                    + 0.05 * event_reason_loss)
                policy_optimization_loss = (
                    memory_loss
                    + conscious_loss
                    + intention_loss
                    + 0.05 * perception_loss
                    + 0.05 * prediction_error_loss
                    + 0.05 * attention_detail_distill_loss
                    + 0.05 * fast_attention_distill_loss
                    + 0.05 * detail_attention_distill_loss
                    + 0.1 * actor_loss
                    + planner_distill_loss
                    + 0.05 * decision_energy_loss
                    + policy_auxiliary_loss
                    + physical_loss)
                total_loss = (
                    world_optimization_loss
                    + critic_optimization_loss
                    + policy_optimization_loss)
                losses = {
                    "world_loss": world_loss,
                    "memory_loss": memory_loss,
                    "critic_loss": critic_loss,
                    "critic_current_loss": critic_current_loss,
                    "critic_transport_delayed_loss": (
                        critic_transport_delayed_loss),
                    "conscious_loss": conscious_loss,
                    "intention_loss": intention_loss,
                    "perception_loss": perception_loss,
                    "attention_detail_distill_loss": (
                        attention_detail_distill_loss),
                    "fast_attention_distill_loss": (
                        fast_attention_distill_loss),
                    "detail_attention_distill_loss": (
                        detail_attention_distill_loss),
                    "world_prediction_loss": world_prediction_loss,
                    "prediction_error_loss": prediction_error_loss,
                    "actor_loss": actor_loss,
                    "goal_progress_loss": goal_progress_loss,
                    "decision_prediction_loss": decision_prediction_loss,
                    "fast_decision_distill_loss": (
                        fast_decision_distill_loss),
                    "detail_decision_distill_loss": (
                        detail_decision_distill_loss),
                    "planner_distill_loss": planner_distill_loss,
                    "decision_energy_loss": decision_energy_loss,
                    "symbolic_invocation_loss": symbolic_invocation_loss,
                    "symbolic_sparsity_loss": symbolic_sparsity_loss,
                    "grounding_loss": grounding_loss,
                    "codebook_util_loss": codebook_util_loss,
                    "cognitive_value_loss": cognitive_value_loss,
                    "plan_staleness_loss": plan_staleness_loss,
                    "feasibility_value_loss": feasibility_loss,
                    "safety_constraint_value_loss": (
                        safety_constraint_loss),
                    "compute_value_loss": compute_value_loss,
                    "counterfactual_evc_loss": counterfactual_evc_loss,
                    "compute_cost_calibration_loss": (
                        compute_cost_calibration_loss),
                    "compute_cost_target": compute_cost_target.mean(),
                    "event_loss": event_loss,
                    "event_reason_loss": event_reason_loss,
                    "temporal_kind_loss": temporal_kind_loss,
                    "temporal_duration_loss": temporal_duration_loss,
                    "hierarchy_loss": hierarchy_loss,
                    "safety_loss": safety_loss,
                    "legality_loss": legality_loss,
                    "safety_prediction_loss": safety_prediction_loss,
                    "selection_loss": selection_loss,
                    "target_continuity_loss": target_continuity_loss,
                    "physical_loss": physical_loss,
                    "world_optimization_loss": world_optimization_loss,
                    "critic_optimization_loss": critic_optimization_loss,
                    "policy_optimization_loss": policy_optimization_loss,
                    "total_current_loss": total_loss,
                    "total_loss": total_loss}
                losses.update({
                    f"perception_recall_{name}": value
                    for name, value in perception_recall_losses.items()})
                losses.update({
                    f"physical_{name}": value
                    for name, value in physical_losses.items()
                    if name != "loss"})
                losses.update({
                    f"world_{name}": value
                    for name, value in w_out.items()
                    if name.startswith("loss_")})
                losses.update({
                    f"world_{name}": value
                    for name, value in world_prediction_losses.items()})
                SaveModuleOutput("Losses", losses)

            self.mem.AgePlanCache(ageMask=~failsafe_event)
            if bool(temporal_decision.candidate_selected.any().item()):
                self.mem.CachePlan(
                    "activePlan",
                    selected_decision_feature,
                    self.model_signature,
                    validMask=temporal_decision.candidate_selected)
            successful_skill = (
                credit_option_valid
                & planner_reached
                & ~planner_failed
                & ~stopped_rows)
            if bool(successful_skill.any().item()):
                option_evidence = F.one_hot(
                    credit_option_index,
                    num_classes=actor.num_options).to(dtype=dtype)
                skill_update = torch.cat([
                    credit_option_goal_mid.to(dtype=dtype),
                    option_evidence,
                    torch.ones(
                        batch_size,
                        1,
                        device=device,
                        dtype=dtype),
                ], dim=-1)
                previous_skill = self.mem.RecallSkill(
                    "executedOption",
                    self.model_signature)
                if previous_skill is None:
                    previous_skill = torch.zeros_like(skill_update)
                else:
                    previous_skill = previous_skill.to(
                        device=device,
                        dtype=dtype)
                    if previous_skill.shape != skill_update.shape:
                        raise RuntimeError("option skill cache is invalid")
                stored_skill = torch.where(
                    successful_skill.unsqueeze(-1),
                    skill_update,
                    previous_skill)
                self.mem.CacheSkill(
                    "executedOption",
                    stored_skill,
                    self.model_signature)
            hierarchy_rows = compute_decision.activated_child_mask.any(dim=-1)
            if bool(hierarchy_rows.any().item()):
                hierarchy_root = torch.tensor(
                    tuple(
                        root and not independent
                        for root, independent in zip(
                            self.robot_contract_view.root_mask,
                            self.robot_contract_view.independent_mask)),
                    device=device,
                    dtype=dtype).unsqueeze(0)
                hierarchy_child = torch.tensor(
                    self.robot_contract_view.child_mask,
                    device=device,
                    dtype=dtype).unsqueeze(0)
                hierarchy_valid = (
                    feedback_packet.endpoint_present
                    & feedback_packet.target_active
                    & feedback_packet.child_enabled).to(dtype=dtype)
                hierarchy_coarse_weight = hierarchy_valid * hierarchy_root
                hierarchy_detail_weight = hierarchy_valid * hierarchy_child
                hierarchy_coarse_progress = (
                    feedback_packet.progress * hierarchy_coarse_weight
                ).sum(dim=-1) / hierarchy_coarse_weight.sum(
                    dim=-1).clamp_min(1.0)
                hierarchy_detail_progress = (
                    feedback_packet.progress * hierarchy_detail_weight
                ).sum(dim=-1) / hierarchy_detail_weight.sum(
                    dim=-1).clamp_min(1.0)
                self.mem.RecordHierarchyTransition(
                    mem_feat[hierarchy_rows],
                    hierarchy_coarse_progress[hierarchy_rows],
                    hierarchy_detail_progress[hierarchy_rows],
                    self.model_signature,
                    transactionVersion=(
                        self.ContractReplayTransactionVersion),
                    timelineVersion=self.ContractReplayTimelineVersion)
            if bool(planner_failed.any().item()):
                self.mem.RecordFailureEpisode(
                    mem_feat[planner_failed],
                    torch.stack([
                        risk,
                        uncertainty,
                        planner_tracking_error], dim=-1)[planner_failed],
                    self.model_signature,
                    transactionVersion=(
                        self.ContractReplayTransactionVersion),
                    timelineVersion=self.ContractReplayTimelineVersion)

            self.prev_mem = self.PreserveRuntimeStateRows(
                mem_feat.detach(),
                self.prev_mem,
                failsafe_event).clone() # [B, D_mem]
            self.prev_attn = self.PreserveRuntimeStateRows(
                atten_out.detach(),
                self.prev_attn,
                failsafe_event).clone() # [B, D_attn]
            self.prev_world_h = self.PreserveRuntimeStateRows(
                w_out["h_next"].detach(),
                self.prev_world_h,
                failsafe_event).clone() # [B, D_world_h]
            self.prev_world_z = self.PreserveRuntimeStateRows(
                w_out["z_next"].detach(),
                self.prev_world_z,
                failsafe_event).clone() # [B, D_world_z] (PST-conditioned stochastic latent)
            self.prev_world_x = self.PreserveRuntimeStateRows(
                w_out["x_next"].detach(),
                self.prev_world_x,
                failsafe_event).clone() # [B, D_world_x]
            self.prev_world_s = self.PreserveRuntimeStateRows(
                w_out["s_next"].detach(),
                self.prev_world_s,
                failsafe_event).clone()
            self.prev_world_embodiment = self.PreserveRuntimeStateRows(
                world_embodiment_state.detach(),
                self.prev_world_embodiment,
                failsafe_event).clone()
            self.prev_entropy = self.ScheduleExecutedState(
                act_out["entropy"].detach(),
                self.prev_entropy,
                option_start,
                option_continue).clone()
            self.prev_option_logit = (
                scheduled_prev_option_logit.detach().clone())
            self.prev_fast_option_logit = (
                scheduled_prev_fast_option_logit.detach().clone())
            self.prev_detail_option_logit = (
                scheduled_prev_detail_option_logit.detach().clone())
            scheduled_decision_state = self.ScheduleExecutedState(
                act_out["decision_state_next"].detach(),
                self.prev_decision_state,
                full_candidate_selected,
                option_continue)
            self.prev_decision_state = scheduled_decision_state.clone()
            self.prev_fast_decision_state = self.ScheduleExecutedState(
                fast_state_out["decision_state_next"].detach(),
                self.prev_fast_decision_state,
                fast_candidate_selected,
                option_continue).clone()
            self.prev_detail_decision_state = self.ScheduleExecutedState(
                detail_state_out["decision_state_next"].detach(),
                self.prev_detail_decision_state,
                detail_candidate_selected,
                option_continue).clone()
            self.prev_belief_prediction_state = self.ScheduleExecutedState(
                act_out["decision_state_next"].detach(),
                self.prev_belief_prediction_state,
                option_start,
                option_continue).clone()
            self.prev_belief_prediction_valid = self.ScheduleExecutedState(
                torch.ones_like(self.prev_belief_prediction_valid),
                self.prev_belief_prediction_valid,
                option_start,
                option_continue).detach().clone()
            self.prev_latent_control = self.ScheduleExecutedState(
                act_out["latent_control_next"].detach(),
                self.prev_latent_control,
                full_candidate_selected,
                option_continue).clone()
            self.prev_fast_latent_control = self.ScheduleExecutedState(
                fast_state_out["latent_control_next"].detach(),
                self.prev_fast_latent_control,
                fast_candidate_selected,
                option_continue).clone()
            self.prev_detail_latent_control = self.ScheduleExecutedState(
                detail_state_out["latent_control_next"].detach(),
                self.prev_detail_latent_control,
                detail_candidate_selected,
                option_continue).clone()
            self.prev_mapper_hidden = self.ScheduleExecutedState(
                act_out["mapper"]["hidden_next"].detach(),
                self.prev_mapper_hidden,
                full_candidate_selected,
                option_continue).clone()
            self.prev_fast_mapper_hidden = self.ScheduleExecutedState(
                fast_state_out["mapper"]["hidden_next"].detach(),
                self.prev_fast_mapper_hidden,
                fast_candidate_selected,
                option_continue).clone()
            self.prev_detail_mapper_hidden = self.ScheduleExecutedState(
                detail_state_out["mapper"]["hidden_next"].detach(),
                self.prev_detail_mapper_hidden,
                detail_candidate_selected,
                option_continue).clone()
            self.prev_td_error = td_signal.detach().clone()
            self.prev_precision = precision.detach().clone()
            self.prev_risk = risk.detach().clone()
            self.prev_world_surprise = self.PreserveRuntimeStateRows(
                world_surprise.detach(),
                self.prev_world_surprise,
                failsafe_event).clone()
            self.prev_novelty = task_relevant_novelty.detach().clone()
            self.prev_information_gain = self.PreserveRuntimeStateRows(
                information_gain.detach(),
                self.prev_information_gain,
                failsafe_event).clone()
            self.prev_evc = self.PreserveRuntimeStateRows(
                (prospective_value.replanBenefit
                 - prospective_value.computeCost).detach(),
                self.prev_evc,
                failsafe_event).clone()
            self.prev_intent_changed = (
                confirmed_intent_event.detach().clone())
            self.prev_goal_changed = goal_changed.detach().clone()
            self.prev_goal_bias = self.PreserveRuntimeStateRows(
                intent_sem.detach(),
                self.prev_goal_bias,
                failsafe_event).clone()
            self.prev_attention_goal = self.PreserveRuntimeStateRows(
                goals["g_short"].detach(),
                self.prev_attention_goal,
                failsafe_event).clone()
            self.prev_self_sem = self.PreserveRuntimeStateRows(
                conscious_out.self_sem.detach(),
                self.prev_self_sem,
                failsafe_event).clone()
            self.prev_intent_sem = self.PreserveRuntimeStateRows(
                intent_sem.detach(),
                self.prev_intent_sem,
                failsafe_event).clone()
            self.prev_failure_count = (
                self.prev_failure_count
                + planner_failed.to(dtype=dtype)) * (
                    ~planner_reached).to(dtype=dtype)
            active_world_rows = torch.nonzero(
                ~failsafe_event,
                as_tuple=False).flatten()
            active_prospective_visual = self.IndexRuntimeRows(
                prospective_visual_prediction,
                active_world_rows,
                batch_size)
            self.prospective_visual_prediction = self.DetachRuntimeObject(
                self.ScatterRuntimeRows(
                    self.prospective_visual_prediction,
                    active_prospective_visual,
                    active_world_rows,
                    batch_size))
            self.ContractObserverRotationGauge = self.PreserveRuntimeStateRows(
                current_rotation.detach(),
                self.ContractObserverRotationGauge,
                failsafe_event).clone()
            self.ContractPreviousTargetActive = (
                feedback_packet.target_active.detach().clone())
            self.ContractPreviousProgress = (
                feedback_packet.progress.detach().clone())
            self.ContractPreviousTextFingerprint = (
                torch.where(
                    failsafe_event,
                    self.ContractPreviousTextFingerprint,
                    text_fingerprint).detach().clone())
            self.ContractPreviousCommandFingerprint = torch.where(
                failsafe_event,
                self.ContractPreviousCommandFingerprint,
                command_fingerprint).detach().clone()
            self.ContractCommandVersion = torch.where(
                failsafe_event,
                self.ContractCommandVersion,
                command_version).detach().clone()
            self.ContractPreviousGoalCode = self.PreserveRuntimeStateRows(
                goal_code.detach(),
                self.ContractPreviousGoalCode,
                failsafe_event).clone()
            self.ContractPreviousReferenceIndex = (
                self.PreserveRuntimeStateRows(
                    reference_index.detach(),
                    self.ContractPreviousReferenceIndex,
                    failsafe_event).clone())
            self.prev_done_flag = torch.where(
                failsafe_event,
                self.prev_done_flag | done_event,
                done_event).detach().clone()

            done_now = done_event
            cached_values = temporal_decision.selected_target.values.detach().clone()
            cached_active = temporal_decision.selected_target.active.detach().clone()
            cached_values[done_now] = 0.0
            cached_active[done_now] = False
            cached_version = temporal_decision.selected_target.target_version.detach().clone()
            cached_timestamp = temporal_decision.selected_target.timestamp.detach().clone()
            cached_version[done_now] = 0
            cached_timestamp[done_now] = 0.0
            self.ContractCachedTarget = PackedEndEffectorTarget(
                values=cached_values,
                active=cached_active,
                contract_id=self.robot_contract_view.contract_id,
                model_signature=self.robot_contract_view.model_signature,
                target_version=cached_version,
                timestamp=cached_timestamp)
            self.ContractCachedActionEpoch = torch.where(
                done_now,
                torch.zeros_like(temporal_decision.action_epoch),
                temporal_decision.action_epoch.detach()).clone()
            self.ContractCacheAge = torch.where(
                done_now | temporal_decision.candidate_selected,
                torch.zeros_like(self.ContractCacheAge),
                self.ContractCacheAge + 1.0)
            advanced_slow_cache_age = torch.where(
                slow_refresh_mask,
                torch.zeros_like(self.ContractSlowCacheAge),
                self.ContractSlowCacheAge + 1.0)
            self.ContractSlowCacheAge = torch.where(
                done_now,
                torch.zeros_like(self.ContractSlowCacheAge),
                torch.where(
                    failsafe_event,
                    self.ContractSlowCacheAge,
                    advanced_slow_cache_age))
            if bool(done_now.any().item()):
                self.ResetContractStateRows(done_now)
                identity = self.ContractObserverRotationGauge.new_zeros(
                    batch_size, 4)
                identity[:, -1] = 1.0
                self.ContractObserverRotationGauge = torch.where(
                    done_now.unsqueeze(-1),
                    identity,
                    self.ContractObserverRotationGauge)
                self.prev_visual_valid.logical_and_(~done_now)
                self.ResetHebbianMemory(doneMask=done_now)
                self.mem.ResetEpisodeState(done_now)
                self.RuntimeModule(self.critic).ResetState(doneMask=done_now)
                self.conscious.ResetState(doneMask=done_now)
                self.OCR.ResetTemporal(doneMask=done_now)
                self.contract_neuro_symbolic.ResetPlan(done_now)
                world.ResetEpisodeState(done_now)

            return BrainStepOutput(
                decision={
                    "decision_feature": selected_decision_feature,
                    "candidate_packed_decision": packed_decision,
                    "packed_target": temporal_decision.selected_target,
                    "packed_temporal": temporal_decision,
                    "planner_prior": planner_prior,
                },
                world=w_out,
                critic=critic_out,
                features={
                    "Perception": visual_state,
                    "PhysicalState": pst,
                    "ContractBody": contract_body,
                    "Attention": atten_out,
                    "AttentionCompute": {
                        name: attention_extras[name]
                        for name in (
                            "selected_compute_units",
                            "actual_compute_units",
                            "full_compute_units",
                            "selected_normalized_compute_fraction",
                            "actual_normalized_compute_fraction",
                            "world_value_posterior_passes",
                            "world_value_normalized_compute_fraction")},
                    "Memory": mem_feat,
                    "MemoryPosteriorCorrection": replay_correction,
                    "Consciousness": conscious_out,
                    "Intention": intent_sem,
                    "IntentionState": intention_extras,
                    "Goals": goals,
                    "GoalGrounding": grounding,
                    "ContractGrounding": contract_grounding,
                    "NeuroSymbolic": neuro_symbolic_out,
                    "CognitiveCompute": compute_decision,
                    "CognitiveValue": critic_out.cognitiveValue,
                    "ProspectiveValue": prospective_value,
                    "ProspectiveVisual": prospective_visual_prediction,
                },
                ocr=ocr_items,
                intention_texts=intention_texts,
                losses=losses,
                stages={
                    "CognitiveCompute": compute_decision,
                    "ScheduledComputeMode": scheduled_compute_mode,
                    "SlowCognitionReused": ~slow_refresh_mask,
                    "PlannerExecuted": (
                        planner_mask
                        if self.planner is not None
                        else torch.zeros_like(planner_mask)),
                    "PlanCacheReused": plan_cache_reused,
                    "StoppedRows": stopped_rows,
                })


    def AdaptiveRuntimeModules(self) -> Dict[str, nn.Module]:
            return {
                name: getattr(self, name)
                for name in ONLINE_WRAPPER_ROOTS
                if hasattr(getattr(self, name), "ExportCandidateState")
            }

    def ContractRuntimeTensorNames(self) -> Tuple[str, ...]:
            return (
                "prev_mem",
                "prev_attn",
                "prev_world_h",
                "prev_world_z",
                "prev_world_x",
                "prev_world_s",
                "prev_world_embodiment",
                "prev_done_flag",
                "prev_option_logit",
                "prev_fast_option_logit",
                "prev_detail_option_logit",
                "active_option_policy_input",
                "active_option_prior_logit",
                "active_option_goal_mid",
                "active_option_index",
                "active_option_valid",
                "prev_decision_state",
                "prev_fast_decision_state",
                "prev_detail_decision_state",
                "prev_belief_prediction_state",
                "prev_belief_prediction_valid",
                "prev_latent_control",
                "prev_fast_latent_control",
                "prev_detail_latent_control",
                "prev_mapper_hidden",
                "prev_fast_mapper_hidden",
                "prev_detail_mapper_hidden",
                "prev_td_error",
                "prev_entropy",
                "prev_precision",
                "prev_goal_bias",
                "prev_attention_goal",
                "prev_intent_sem",
                "prev_failure_count",
                "prev_visual_valid",
                "prev_risk",
                "prev_world_surprise",
                "prev_novelty",
                "prev_information_gain",
                "prev_evc",
                "prev_intent_changed",
                "prev_goal_changed",
                "ContractCachedActionEpoch",
                "ContractObserverRotationGauge",
                "ContractCacheAge",
                "ContractSlowCacheAge",
                "ContractSlowCacheValid",
                "ContractReplayEpisodeVersion",
                "ContractPreviousTargetActive",
                "ContractPreviousProgress",
                "ContractPreviousTextFingerprint",
                "ContractPreviousCommandFingerprint",
                "ContractCommandVersion",
                "ContractPreviousGoalCode",
                "ContractPreviousReferenceIndex")

    def SuspendTransientTrainingGraph(self) -> Dict[str, Any]:
            return self.RuntimeModule(
                self.critic).SuspendTransientTrainingGraph()

    def RestoreTransientTrainingGraph(
            self,
            state: Dict[str, Any],
        ) -> None:
            self.RuntimeModule(
                self.critic).RestoreTransientTrainingGraph(state)

    def MoveRuntimeStateToModel(self, value: Any) -> Any:
            device = next(self.parameters()).device
            if torch.is_tensor(value):
                return value.to(device=device)
            if isinstance(value, dict):
                return {
                    name: (
                        copy.deepcopy(item)
                        if name == "PatchGridShape"
                        else self.MoveRuntimeStateToModel(item))
                    for name, item in value.items()}
            if isinstance(value, list):
                return [
                    self.MoveRuntimeStateToModel(item)
                    for item in value]
            if isinstance(value, deque):
                return deque((
                    self.MoveRuntimeStateToModel(item)
                    for item in value), maxlen=value.maxlen)
            if isinstance(value, tuple) and hasattr(value, "_fields"):
                return type(value)(*(
                    self.MoveRuntimeStateToModel(item)
                    for item in value))
            if isinstance(value, tuple):
                return tuple(
                    self.MoveRuntimeStateToModel(item)
                    for item in value)
            if hasattr(value, "__dataclass_fields__"):
                return type(value)(**{
                    name: self.MoveRuntimeStateToModel(
                        getattr(value, name))
                    for name in value.__dataclass_fields__})
            return value

    @torch.no_grad()
    def ExportBuffers(self) -> Dict[str, Any]:
            world = self.ContractWorld()
            world_h, world_z, world_x = world.ExportState()
            tensors = {
                name: getattr(self, name).detach().clone()
                for name in self.ContractRuntimeTensorNames()}
            tensors["prev_self_sem"] = (
                None
                if self.prev_self_sem is None
                else self.prev_self_sem.detach().clone())
            conscious = {
                name: getattr(self.conscious, name).detach().clone()
                for name in (
                    "_dev_trace",
                    "_last_self_intent",
                    "_last_sem",
                    "_state_valid",
                    "_step")}
            cognitive_state = {
                "tensors": tensors,
                "world": {
                    "h": world_h.detach().clone(),
                    "z": world_z.detach().clone(),
                    "x": world_x.detach().clone()},
                "memory": self.mem.ExportTransientState(),
                "memory_cognitive_cache": (
                    self.mem.ExportCognitiveCacheState()),
                "attention": self.RuntimeModule(self.attn).ExportState(),
                "value": self.RuntimeModule(self.critic).ExportState(),
                "decision_eligibility": self.RuntimeModule(
                    self.actor).ExportEligibilityState(),
                "consciousness": conscious,
                "plan": self.contract_neuro_symbolic.ExportPlanState(),
                "cached_target": self.DetachRuntimeObject(
                    self.ContractCachedTarget,
                    clone=True),
                "commitment": self.DetachRuntimeObject(
                    self.ContractIntentionCommitmentState,
                    clone=True),
                "slow_cache": self.DetachRuntimeObject(
                    self.ContractSlowCognitiveCache,
                    clone=True),
                "replay_history": self.DetachRuntimeObject(
                    self.ContractReplayHistory,
                    clone=True),
                "replay_timeline_version": int(
                    self.ContractReplayTimelineVersion),
                "replay_transaction_version": int(
                    self.ContractReplayTransactionVersion),
                "previous_visual": self.DetachVisualState(
                    self.prev_visual_state,
                    clone=True),
                "visual_buffer": [
                    self.DetachVisualState(value, clone=True)
                    for value in self.visual_state_buffer],
                "visual_valid_buffer": [
                    value.detach().clone()
                    for value in self.visual_state_valid_buffer],
                "perception_buffer": [
                    value.detach().clone()
                    for value in self.perc_buffer],
                "prospective_visual": self.DetachRuntimeObject(
                    self.prospective_visual_prediction,
                    clone=True),
                "compute_previous": (
                    self.cognitive_compute_gate.PreviousChildEnabled
                    .detach().clone()),
                "ocr": {
                    "temporal_step": int(self.OCR._temporal_step),
                    "last_batch_size": int(self.OCR._last_batch_size),
                    "texts": copy.deepcopy(
                        self.OCR._last_ocr_texts_batch),
                    "tracks": copy.deepcopy(self.OCR._tracks_by_bi)}}
            return {
                "schema_version": BRAIN_RUNTIME_SCHEMA_VERSION,
                "contract_id": self.robot_contract_view.contract_id,
                "model_signature": self.model_signature,
                "batch_size": int(self.ContractRuntimeBatch),
                "cognitive_state": cognitive_state}

    @staticmethod
    def ValidateRuntimeTree(
            value: Any,
            device: torch.device,
            dtype: torch.dtype,
            path: str,
        ) -> None:
            if torch.is_tensor(value):
                if value.device != device or value.is_complex():
                    raise ValueError(path + " tensor type does not match")
                if value.is_floating_point():
                    if (
                        value.dtype != dtype
                        or not bool(torch.isfinite(value).all().item())
                    ):
                        raise ValueError(path + " tensor is invalid")
                elif value.dtype not in (torch.bool, torch.long):
                    raise ValueError(path + " tensor dtype is invalid")
                return
            if value is None or type(value) in (str, bool, int):
                return
            if type(value) is float:
                if not math.isfinite(value):
                    raise ValueError(path + " numeric value is invalid")
                return
            if type(value) is dict:
                if any(type(name) is not str or not name for name in value):
                    raise ValueError(path + " mapping keys are invalid")
                for name, item in value.items():
                    BrainCore.ValidateRuntimeTree(
                        item,
                        device,
                        dtype,
                        path + "." + name)
                return
            if type(value) in (list, tuple, deque):
                for index, item in enumerate(value):
                    BrainCore.ValidateRuntimeTree(
                        item,
                        device,
                        dtype,
                        path + "[" + str(index) + "]")
                return
            allowed_records = (
                ConsciousnessOutput,
                ContractGroundingOutput,
                NeuroSymbolicOutput,
                OperatorRationale,
                OperatorStep,
                SymbolicFact,
            )
            if type(value) in allowed_records:
                for name in value.__dataclass_fields__ if hasattr(
                    value, "__dataclass_fields__") else value._fields:
                    BrainCore.ValidateRuntimeTree(
                        getattr(value, name),
                        device,
                        dtype,
                        path + "." + name)
                return
            raise TypeError(path + " contains an unsupported runtime object")

    @staticmethod
    def ValidateBatchedTensor(
            value: Any,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
            path: str,
            trailingShape: Optional[Tuple[int, ...]] = None,
            expectedDtype: Optional[torch.dtype] = None,
        ) -> None:
            expected_shape = (
                None
                if trailingShape is None
                else (int(batchSize),) + tuple(trailingShape))
            if (
                not torch.is_tensor(value)
                or value.dim() < 1
                or int(value.size(0)) != int(batchSize)
                or (
                    expected_shape is not None
                    and tuple(value.shape) != expected_shape)
                or value.device != device
                or (
                    expectedDtype is not None
                    and value.dtype != expectedDtype)
                or (
                    expectedDtype is None
                    and (
                        not value.is_floating_point()
                        or value.dtype != dtype))
                or (
                    value.is_floating_point()
                    and not bool(torch.isfinite(value).all().item()))
            ):
                raise ValueError(path + " does not match the runtime batch")

    def ValidateVisualRuntimeState(
            self,
            state: Any,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
            path: str,
        ) -> None:
            if type(state) is not VisualState:
                raise TypeError(path + " must be a VisualState")
            perception = self.RuntimeModule(self.perc)
            integrated_dim = int(perception.integrated_dim)
            embed_dim = int(perception.embed_dim)
            object_count = int(perception.object_token_count)
            tensor_shapes = {
                "IntegratedFeat": (integrated_dim,),
                "GlobalFeat": (integrated_dim,),
                "VentralFeat": (embed_dim,),
                "DorsalFeat": (embed_dim,),
                "MotionToken": (embed_dim,),
                "QualityToken": (embed_dim,),
                "PredErrorToken": (embed_dim,),
                "ObjectTokens": (object_count, embed_dim),
            }
            for name, trailing_shape in tensor_shapes.items():
                self.ValidateBatchedTensor(
                    getattr(state, name),
                    batchSize,
                    device,
                    dtype,
                    path + "." + name,
                    trailingShape=trailing_shape)
            patch_tokens = state.PatchTokens
            if (
                not torch.is_tensor(patch_tokens)
                or patch_tokens.dim() != 3
                or tuple(patch_tokens.shape[:1]) != (int(batchSize),)
                or int(patch_tokens.size(1)) < 1
                or int(patch_tokens.size(2)) != embed_dim
                or patch_tokens.device != device
                or patch_tokens.dtype != dtype
                or not bool(torch.isfinite(patch_tokens).all().item())
            ):
                raise ValueError(path + ".PatchTokens is invalid")
            semantic_fields = {
                "node_logits",
                "level_logits",
                "object_class_logits",
                "part_class_logits",
                "parent_logits",
                "orientation_observer",
                "size_3d",
                "bbox_2d",
                "visible_ratio",
                "occlusion_ratio",
                "has_text_logits",
                "text_embed",
                "symbol_logits",
                "identity_embed",
                "scene_logits",
                "global_label_logits",
                "position_observer",
            }
            if (
                type(state.SemanticNodes) is not dict
                or set(state.SemanticNodes) != semantic_fields
            ):
                raise ValueError(path + ".SemanticNodes fields do not match")
            recall_heads = perception.recall_heads
            semantic_shapes = {
                "node_logits": (object_count, 2),
                "level_logits": (object_count, 3),
                "object_class_logits": (
                    object_count, recall_heads.num_object_classes),
                "part_class_logits": (
                    object_count, recall_heads.num_part_classes),
                "parent_logits": (object_count, object_count),
                "orientation_observer": (object_count, 4),
                "size_3d": (object_count, 3),
                "bbox_2d": (object_count, 4),
                "visible_ratio": (object_count,),
                "occlusion_ratio": (object_count,),
                "has_text_logits": (object_count, 2),
                "text_embed": (object_count, recall_heads.text_dim),
                "symbol_logits": (object_count, recall_heads.num_symbols),
                "identity_embed": (object_count, recall_heads.identity_dim),
                "scene_logits": (recall_heads.num_scene_classes,),
                "global_label_logits": (recall_heads.num_global_labels,),
                "position_observer": (object_count, 3),
            }
            for name, value in state.SemanticNodes.items():
                self.ValidateBatchedTensor(
                    value,
                    batchSize,
                    device,
                    dtype,
                    path + ".SemanticNodes." + name,
                    trailingShape=semantic_shapes[name])
            auxiliary_fields = {
                "TemporalState",
                "PredErrorTarget",
                "MonocularDepth",
                "MonocularDepthLogVariance",
                "MetricDepth",
                "MetricDepthLogVariance",
                "MetricInverseDepth",
                "SensorDepthReliability",
                "SensorDepthValid",
                "SensorDepthValidMask",
                "SensorDepthUsed",
                "ContentDepth",
                "VirtualMask",
                "VirtualMaskLogits",
                "PlanarityConfidence",
                "SensorLogVarianceSpatial",
                "VirtualTarget",
                "EdgeAwareSmoothness",
                "EdgeAwareSmoothnessPerRow",
                "MetricDepthFullRes",
                "MetricDepthFullResLogVariance",
                "ObjectMotion",
                "ObjectMotionBase",
                "ObjectTokensBase",
                "ObjectGeometry",
                "ObjectGeometryValid",
                "PatchSurfaceEvidence",
                "MotionLayer5",
                "LayerAgency",
                "StaticTemporalDepthWeightMap",
                "PatchMotionLayerProb",
                "PatchLayerAgencyProb",
                "ObjectMotionLayerProb",
                "ObjectLayerAgencyProb",
                "LayerAgencyProb",
                "ObjectAgencyProb",
                "ObjectMotionLayerVisualEvidence",
                "LayerAgencyVisualEvidence",
                "ObjectFactorTokens",
                "MotionFactorSummary",
                "MotionDynamicFactorSummary",
                "PatchDisplaySurfaceProb",
                "DisplaySurfaceProb",
                "DisplaySurfaceVisualEvidence",
                "SurfaceUV",
                "SurfaceUVVisualEvidence",
                "SurfaceUVConfidence",
                "ContentMotionUV",
                "ContentMotionUVVisualEvidence",
                "ContentChangeProb",
                "ContentChangeVisualEvidence",
                "PatchLocalFlowUV",
                "PatchFlowConfidence",
                "PatchFeatureChange",
                "PatchDynamicProb",
                "StaticTemporalDepthWeight",
                "PatchMotionTokens",
                "PatchMotionReliability",
                "PatchMotionWeights",
                "PatchMotionDepthResidual",
                "WarpedPrevPatchTokens",
                "WarpPrevPatchValid",
                "ObserverRotationFromPrev",
                "ObserverRotationValid",
                "ObserverAngularVelocity",
                "PatchGridShape",
                "DorsalReliabilityGate",
                "RigidPatchFlow",
                "WarpJacobianDet",
                "WarpJacobianSigmaMin",
                "WarpJacobianSigmaMax",
                "WarpTopologyValid",
                "WarpFoldPenalty",
                "WarpFoldPenaltyPerRow",
                "CorticalFastState",
                "CorticalSlowState",
                "CorticalStabilizedFastState",
                "CorticalStabilizedSlowState",
                "CorticalEnergy",
                "CorticalContextResponse",
                "CorticalTemporalResponse",
                "CorticalRetinalTemporalResponse",
                "CorticalStabilizedTemporalResponse",
                "CorticalRotationWarpValid",
                "PerceptualObjectAgencyProb",
                "PerceptualMotionLayerProb",
                "PerceptualLayerAgencyProb",
                "PerceptualDisplaySurfaceProb",
                "PerceptualSurfaceUV",
                "PerceptualSurfaceUVConfidence",
                "PerceptualContentMotionUV",
                "PerceptualContentChangeProb",
                "PerceptualPresence",
                "GeometryValidMask",
                "MphysRaw",
                "PhysicalEntityProb",
                "PhysicalInteractionProb",
                "RealmProb",
                "MotionLayerProb",
                "AgencyProb",
                "BodyMembershipProb",
                "SelfPartProb",
                "SelfPartSemantic",
                "CarrierMotionObserverRaw",
                "ArticulationMotionObserverRaw",
                "SurfaceParentProb",
                "VerificationConfidence",
                "OntologyRelationProb",
                "EntityRealmProb",
            }
            optional_fields = {
                "MetricDepthFullRes",
                "MetricDepthFullResLogVariance",
                "CorticalFastState",
                "CorticalSlowState",
                "CorticalStabilizedFastState",
                "CorticalStabilizedSlowState",
                "CorticalEnergy",
                "CorticalContextResponse",
                "CorticalTemporalResponse",
                "CorticalRetinalTemporalResponse",
                "CorticalStabilizedTemporalResponse",
                "CorticalRotationWarpValid",
                "PerceptualObjectAgencyProb",
                "PerceptualMotionLayerProb",
                "PerceptualLayerAgencyProb",
                "PerceptualDisplaySurfaceProb",
                "PerceptualSurfaceUV",
                "PerceptualSurfaceUVConfidence",
                "PerceptualContentMotionUV",
                "PerceptualContentChangeProb",
                "PerceptualPresence",
                "GeometryValidMask",
                "MphysRaw",
                "PhysicalEntityProb",
                "PhysicalInteractionProb",
                "RealmProb",
                "MotionLayerProb",
                "AgencyProb",
                "BodyMembershipProb",
                "SelfPartProb",
                "SelfPartSemantic",
                "CarrierMotionObserverRaw",
                "ArticulationMotionObserverRaw",
                "SurfaceParentProb",
                "VerificationConfidence",
                "OntologyRelationProb",
                "EntityRealmProb",
            }
            present_fields = set(state.Auxiliary)
            if (
                type(state.Auxiliary) is not dict
                or not auxiliary_fields.difference(optional_fields).issubset(
                    present_fields)
                or not present_fields.issubset(auxiliary_fields)
            ):
                raise ValueError(path + ".Auxiliary fields do not match")
            scalar_fields = {"EdgeAwareSmoothness", "WarpFoldPenalty"}
            boolean_fields = {
                "SensorDepthValidMask",
                "ObserverRotationValid",
                "CorticalRotationWarpValid",
            }
            for name, value in state.Auxiliary.items():
                if name == "PatchGridShape":
                    if (
                        not torch.is_tensor(value)
                        or tuple(value.shape) != (2,)
                        or value.dtype != torch.long
                        or value.device.type != "cpu"
                        or bool((value <= 0).any().item())
                        or int(value.prod().item())
                        != int(patch_tokens.size(1))
                    ):
                        raise ValueError(path + ".Auxiliary.PatchGridShape is invalid")
                    continue
                if not torch.is_tensor(value) or value.device != device:
                    raise ValueError(path + ".Auxiliary." + name + " is invalid")
                if name in scalar_fields:
                    if value.dim() != 0:
                        raise ValueError(path + ".Auxiliary." + name + " is invalid")
                elif value.dim() < 1 or int(value.size(0)) != int(batchSize):
                    raise ValueError(path + ".Auxiliary." + name + " is invalid")
                if name in boolean_fields:
                    if value.dtype != torch.bool:
                        raise ValueError(path + ".Auxiliary." + name + " is invalid")
                elif (
                    not value.is_floating_point()
                    or value.dtype not in (dtype, torch.float32)
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise ValueError(path + ".Auxiliary." + name + " is invalid")

    def ValidateCachedTargetRuntimeState(
            self,
            target: Any,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            if target is None:
                return
            if type(target) is not PackedEndEffectorTarget:
                raise TypeError(
                    "brain cached target must be PackedEndEffectorTarget")
            self.ValidateBatchedTensor(
                target.values,
                batchSize,
                device,
                dtype,
                "brain cached target values",
                trailingShape=(
                    self.robot_contract_view.end_effector_target_layout.PackedDim,))
            self.ValidateBatchedTensor(
                target.active,
                batchSize,
                device,
                dtype,
                "brain cached target activity",
                trailingShape=(self.robot_contract_view.end_effector_count,),
                expectedDtype=torch.bool)
            target.Validate(self.robot_contract_view)

    def ValidateCommitmentRuntimeState(
            self,
            state: Any,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            if type(state) is not dict:
                raise TypeError("brain commitment state must be a dictionary")
            if not state:
                return
            expected = {
                "committed_intent",
                "commitment_strength",
                "command_version",
                "commitment_valid",
                "pending_intent",
                "pending_evidence",
                "pending_dwell",
                "pending_valid",
                "pending_exit_dwell",
                "weak_dwell",
            }
            if set(state) != expected:
                raise ValueError("brain commitment fields do not match")
            intention_dim = int(self.RuntimeModule(self.intention).dimSem)
            for name in ("committed_intent", "pending_intent"):
                self.ValidateBatchedTensor(
                    state[name],
                    batchSize,
                    device,
                    dtype,
                    "brain commitment " + name,
                    trailingShape=(intention_dim,))
            for name in ("commitment_strength", "pending_evidence"):
                self.ValidateBatchedTensor(
                    state[name],
                    batchSize,
                    device,
                    dtype,
                    "brain commitment " + name,
                    trailingShape=())
                if bool(((state[name] < 0.0) | (state[name] > 1.0)).any().item()):
                    raise ValueError("brain commitment probability is invalid")
            for name in ("commitment_valid", "pending_valid"):
                self.ValidateBatchedTensor(
                    state[name],
                    batchSize,
                    device,
                    dtype,
                    "brain commitment " + name,
                    trailingShape=(),
                    expectedDtype=torch.bool)
            for name in (
                "command_version",
                "pending_dwell",
                "pending_exit_dwell",
                "weak_dwell",
            ):
                self.ValidateBatchedTensor(
                    state[name],
                    batchSize,
                    device,
                    dtype,
                    "brain commitment " + name,
                    trailingShape=(),
                    expectedDtype=torch.long)
                if bool((state[name] < 0).any().item()):
                    raise ValueError("brain commitment counter is invalid")

    @staticmethod
    def RuntimeTreesEqual(left: Any, right: Any) -> bool:
            if torch.is_tensor(left) or torch.is_tensor(right):
                return (
                    torch.is_tensor(left)
                    and torch.is_tensor(right)
                    and torch.equal(left, right))
            if type(left) is not type(right):
                return False
            if type(left) is dict:
                return set(left) == set(right) and all(
                    BrainCore.RuntimeTreesEqual(left[name], right[name])
                    for name in left)
            if type(left) in (list, tuple, deque):
                return len(left) == len(right) and all(
                    BrainCore.RuntimeTreesEqual(a, b)
                    for a, b in zip(left, right))
            if hasattr(left, "__dataclass_fields__"):
                return all(
                    BrainCore.RuntimeTreesEqual(
                        getattr(left, name),
                        getattr(right, name))
                    for name in left.__dataclass_fields__)
            return left == right

    def ValidateSlowCacheRuntimeState(
            self,
            cache: Any,
            commitment: Dict[str, torch.Tensor],
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            if cache is None:
                return
            expected = {
                "Consciousness",
                "Intention",
                "SymbolProbabilities",
                "IntentionState",
                "IntentionTexts",
                "Goals",
                "GoalGrounding",
                "OcrItems",
                "OcrTexts",
                "ContractGrounding",
                "NeuroSymbolic",
            }
            if type(cache) is not dict or set(cache) != expected:
                raise ValueError("brain slow cache fields do not match")
            consciousness = cache["Consciousness"]
            if type(consciousness) is not ConsciousnessOutput:
                raise TypeError("brain slow consciousness cache is invalid")
            self.ValidateBatchedTensor(
                consciousness.self_sem,
                batchSize,
                device,
                dtype,
                "brain slow consciousness self",
                trailingShape=(int(self.conscious.self_dim),))
            self.ValidateBatchedTensor(
                consciousness.intent_sem,
                batchSize,
                device,
                dtype,
                "brain slow consciousness intent",
                trailingShape=(int(self.conscious.intent_dim),))
            if type(consciousness.extras) is not dict:
                raise TypeError("brain slow consciousness extras are invalid")
            self.ValidateRuntimeTree(
                consciousness.extras,
                device,
                dtype,
                "brain slow consciousness extras")
            self.ValidateBatchedTensor(
                cache["Intention"],
                batchSize,
                device,
                dtype,
                "brain slow intention",
                trailingShape=(int(self.RuntimeModule(self.intention).dimSem),))
            self.ValidateBatchedTensor(
                cache["SymbolProbabilities"],
                batchSize,
                device,
                dtype,
                "brain slow symbolic probabilities",
                trailingShape=(
                    int(self.RuntimeModule(
                        self.intention).conceptEmb.size(0)),))
            intention_state = cache["IntentionState"]
            allowed_intention_fields = {
                "cons_self_sem",
                "cons_intent_sem",
                "cons_pair_sem",
                "cons_token_weights",
                "cons_sem",
                "text_trust",
                "ext_control_mask",
                "ocr_control_weight",
                "sem_ocr_raw",
                "sem_ocr_fused",
                "gate_ocr",
                "has_ocr_mask",
                "sem_ocr_slots",
                "sem_ocr_slot_mask",
                "gamma_ext",
                "sem_ext_fused",
                "sem_ext_observed",
                "sem_ext_controlled",
                "gate_ext",
                "has_ext_mask",
                "sem_ext_slots",
                "sem_ext_slot_mask",
                "intent_trans_norm",
                "intent_trans_mask_sum",
                "sym_probs_loop",
                "sym_ctrl_gains",
                "sym_tok_w",
                "sym_reason_alpha",
                "commitment_state",
                "commitment_strength",
                "command_version",
                "intent_changed",
                "intent_conflict",
                "intent_conflict_confirmed",
                "intent_change_score",
                "source_reliability",
                "source_reliability_components",
                "source_attribution",
                "pending_evidence",
                "pending_dwell",
                "pending_valid",
                "hard_command",
                "hard_accepted",
                "version_advanced",
                "version_regressed",
                "reason_alpha_final",
                "recall_texts",
                "recall_logits",
                "recall_targets",
                "recall_valid",
                "recall_pred_ids",
            }
            if (
                type(intention_state) is not dict
                or type(intention_state.get("commitment_state")) is not dict
                or not set(intention_state).issubset(
                    allowed_intention_fields)
            ):
                raise ValueError("brain slow intention state is invalid")
            self.ValidateRuntimeTree(
                intention_state,
                device,
                dtype,
                "brain slow intention state")
            for name, value in intention_state.items():
                if name == "commitment_state" or not torch.is_tensor(value):
                    continue
                if (
                    value.dim() < 1
                    or int(value.size(0)) != int(batchSize)
                    or value.device != device
                    or value.dtype not in (dtype, torch.bool, torch.long)
                    or (
                        value.is_floating_point()
                        and not bool(torch.isfinite(value).all().item()))
                ):
                    raise ValueError(
                        "brain slow intention tensor is invalid")
            self.ValidateCommitmentRuntimeState(
                intention_state["commitment_state"],
                batchSize,
                device,
                dtype)
            if not self.RuntimeTreesEqual(
                commitment,
                intention_state["commitment_state"],
            ):
                raise ValueError("brain commitment cache is inconsistent")
            for name in ("IntentionTexts", "OcrItems", "OcrTexts"):
                values = cache[name]
                if type(values) is not list or len(values) != int(batchSize):
                    raise ValueError("brain slow " + name + " batch is invalid")
                self.ValidateRuntimeTree(
                    values,
                    device,
                    dtype,
                    "brain slow " + name)
            if any(
                value is not None and type(value) is not str
                for value in cache["IntentionTexts"]
            ):
                raise ValueError("brain slow intention text is invalid")
            for row in cache["OcrTexts"]:
                if (
                    row is not None
                    and (
                        type(row) is not list
                        or any(type(value) is not str for value in row))
                ):
                    raise ValueError("brain slow OCR text is invalid")
            ocr_item_fields = {
                "box",
                "text",
                "det_score",
                "rec_conf",
                "score",
            }
            for row in cache["OcrItems"]:
                if row is None:
                    continue
                if type(row) is not list:
                    raise ValueError("brain slow OCR items are invalid")
                for item in row:
                    if (
                        type(item) is not dict
                        or set(item) != ocr_item_fields
                        or type(item["box"]) is not tuple
                        or len(item["box"]) != 4
                        or any(type(value) is not int for value in item["box"])
                        or type(item["text"]) is not str
                        or any(
                            type(item[name]) is not float
                            or not math.isfinite(item[name])
                            for name in ("det_score", "rec_conf", "score"))
                    ):
                        raise ValueError("brain slow OCR item is invalid")
            goals = cache["Goals"]
            goal_fields = {
                "g_ultimate",
                "g_long",
                "g_mid",
                "g_short",
                "goal_symbolic",
                "goal_temporal",
                "goal_decision",
                "ultimate_logits",
                "long_logits",
                "mid_logits",
                "task_context",
                "task_relation",
                "task_object",
                "precision_requirement",
                "time_requirement",
                "termination_requirement",
                "active_perception_requirement",
                "capability_relevance",
                "capability_summary",
                "endpoint_relevance",
                "subtree_relevance",
                "endpoint_active",
                "subtree_active",
            }
            if type(goals) is not dict or set(goals) != goal_fields:
                raise ValueError("brain slow goal fields do not match")
            for name, value in goals.items():
                self.ValidateBatchedTensor(
                    value,
                    batchSize,
                    device,
                    dtype,
                    "brain slow goal " + name,
                    expectedDtype=(
                        torch.bool
                        if name in ("endpoint_active", "subtree_active")
                        else None))
            grounding = cache["GoalGrounding"]
            grounding_fields = {
                "referenced_object_probs",
                "reference_distribution",
                "query_reference_distribution",
                "subgoal_reference_distribution",
                "grounding_consistency_loss",
                "referenced_entity_summary",
                "reference_confidence",
                "no_reference_prob",
                "semantic_reference_probs",
                "semantic_reference_distribution",
                "semantic_reference_confidence",
            }
            grounding_with_identity = grounding_fields | {
                "referenced_entity_id",
                "referenced_entity_generation",
            }
            if (
                type(grounding) is not dict
                or set(grounding) not in (
                    grounding_fields,
                    grounding_with_identity)
            ):
                raise ValueError("brain slow grounding fields do not match")
            for name, value in grounding.items():
                if name == "grounding_consistency_loss":
                    if (
                        not torch.is_tensor(value)
                        or value.dim() != 0
                        or value.device != device
                        or value.dtype != dtype
                        or not bool(torch.isfinite(value).item())
                    ):
                        raise ValueError("brain grounding loss is invalid")
                else:
                    expected_dtype = (
                        torch.long
                        if name in (
                            "referenced_entity_id",
                            "referenced_entity_generation")
                        else None)
                    self.ValidateBatchedTensor(
                        value,
                        batchSize,
                        device,
                        dtype,
                        "brain slow grounding " + name,
                        expectedDtype=expected_dtype)
            contract_grounding = cache["ContractGrounding"]
            if (
                type(contract_grounding) is not ContractGroundingOutput
                or contract_grounding.slot_predicate_names
                != CONTRACT_SLOT_PREDICATES
                or contract_grounding.execution_predicate_names
                != CONTRACT_EXECUTION_PREDICATES
                or contract_grounding.evidence_names
                != CONTRACT_EVIDENCE_FIELDS
            ):
                raise ValueError("brain contract grounding identity is invalid")
            slot_count = self.robot_contract_view.end_effector_count
            for name, trailing_shape, expected_dtype in (
                (
                    "slot_predicate_prob",
                    (slot_count, len(CONTRACT_SLOT_PREDICATES)),
                    None),
                (
                    "slot_predicate_known",
                    (slot_count, len(CONTRACT_SLOT_PREDICATES)),
                    torch.bool),
                (
                    "execution_predicate_prob",
                    (len(CONTRACT_EXECUTION_PREDICATES),),
                    None),
                (
                    "execution_predicate_known",
                    (len(CONTRACT_EXECUTION_PREDICATES),),
                    torch.bool),
                (
                    "evidence",
                    (slot_count, len(CONTRACT_EVIDENCE_FIELDS)),
                    None),
            ):
                self.ValidateBatchedTensor(
                    getattr(contract_grounding, name),
                    batchSize,
                    device,
                    dtype,
                    "brain contract grounding " + name,
                    trailingShape=trailing_shape,
                    expectedDtype=expected_dtype)
            self.ValidateBatchedTensor(
                contract_grounding.slot_features,
                batchSize,
                device,
                dtype,
                "brain contract grounding slot features")
            if type(contract_grounding.facts) is not list:
                raise TypeError("brain contract grounding facts are invalid")
            self.ValidateRuntimeTree(
                contract_grounding.facts,
                device,
                dtype,
                "brain contract grounding facts")
            neuro_symbolic = cache["NeuroSymbolic"]
            if type(neuro_symbolic) is not NeuroSymbolicOutput:
                raise TypeError("brain neuro-symbolic cache is invalid")
            for name in neuro_symbolic.__dataclass_fields__:
                value = getattr(neuro_symbolic, name)
                if torch.is_tensor(value):
                    self.ValidateBatchedTensor(
                        value,
                        batchSize,
                        device,
                        dtype,
                        "brain neuro-symbolic " + name,
                        expectedDtype=(
                            value.dtype
                            if value.dtype in (torch.bool, torch.long)
                            else None))
                else:
                    self.ValidateRuntimeTree(
                        value,
                        device,
                        dtype,
                        "brain neuro-symbolic " + name)

    def ValidateProspectiveVisualRuntimeState(
            self,
            state: Any,
            batchSize: int,
            device: torch.device,
            dtype: torch.dtype,
        ) -> None:
            if state is None:
                return
            if (
                type(state) is not dict
                or set(state) != {
                    "predicted_visual",
                    "reconstructed_visual_state",
                    "prior_rollout"}
            ):
                raise ValueError("brain prospective visual fields do not match")
            predicted = state["predicted_visual"]
            if type(predicted) is not PredictedVisualPack:
                raise TypeError("brain predicted visual pack is invalid")
            world = self.ContractWorld()
            predicted_shapes = {
                "GlobalFeat": (int(world.global_feat_dim),),
                "ObjectTokens": (
                    int(world.num_object_tokens),
                    int(world.object_token_dim)),
                "MotionPred": (int(world.motion_pred_dim),),
                "IntegratedFeat": (int(world.integrated_feat_dim),),
            }
            for name in predicted.__dataclass_fields__:
                self.ValidateBatchedTensor(
                    getattr(predicted, name),
                    batchSize,
                    device,
                    dtype,
                    "brain predicted visual " + name,
                    trailingShape=predicted_shapes[name])
            reconstructed = state["reconstructed_visual_state"]
            reconstructed_fields = {
                "IntegratedFeat",
                "GlobalFeat",
                "ObjectTokens",
                "MotionPred",
                "SlotState",
                "SlotPresenceLogits",
                "SceneSummary",
                "ObjectSummary",
                "PriorConfidence",
                "RealmProb",
                "MotionLayerProb",
                "LayerAgencyProb",
                "ObjectAgencyProb",
                "DisplaySurfaceProb",
                "SurfaceUV",
                "ContentMotionUV",
                "ContentChangeProb",
                "FactorPriorConfidence",
                "PredErrorBasis",
            }
            if (
                type(reconstructed) is not dict
                or set(reconstructed) != reconstructed_fields
            ):
                raise ValueError("brain reconstructed visual fields do not match")
            reconstructed_shapes = {
                "IntegratedFeat": (int(world.integrated_feat_dim),),
                "GlobalFeat": (int(world.global_feat_dim),),
                "ObjectTokens": (
                    int(world.num_object_tokens),
                    int(world.object_token_dim)),
                "MotionPred": (int(world.motion_pred_dim),),
                "SlotState": (
                    int(world.num_object_tokens),
                    int(world.object_token_dim)),
                "SlotPresenceLogits": (int(world.num_object_tokens),),
                "SceneSummary": (int(world.object_token_dim),),
                "ObjectSummary": (int(world.object_token_dim),),
                "PriorConfidence": (),
                "RealmProb": (
                    int(world.num_object_tokens),
                    ModuleDim.PstRealmClasses),
                "MotionLayerProb": (
                    int(world.num_object_tokens),
                    ModuleDim.PstMotionLayerClasses),
                "LayerAgencyProb": (
                    int(world.num_object_tokens),
                    ModuleDim.PstMotionLayerClasses,
                    ModuleDim.PstAgencyClasses),
                "ObjectAgencyProb": (
                    int(world.num_object_tokens),
                    ModuleDim.PstAgencyClasses),
                "DisplaySurfaceProb": (int(world.num_object_tokens),),
                "SurfaceUV": (int(world.num_object_tokens), 2),
                "ContentMotionUV": (int(world.num_object_tokens), 2),
                "ContentChangeProb": (int(world.num_object_tokens),),
                "FactorPriorConfidence": (
                    int(world.num_object_tokens),),
                "PredErrorBasis": (int(world.global_feat_dim),),
            }
            for name, value in reconstructed.items():
                self.ValidateBatchedTensor(
                    value,
                    batchSize,
                    device,
                    dtype,
                    "brain reconstructed visual " + name,
                    trailingShape=reconstructed_shapes[name])
            rollout = state["prior_rollout"]
            rollout_fields = {
                "h_next",
                "z_next",
                "z_next_raw",
                "x_next",
                "s_next",
                "action_enc",
                "embodied_action",
                "embodiment_context",
                "r_pred",
                "d_prob",
                "d_tr",
                "d_ph",
                "pst_binding",
                "loss_pst_bind",
            }
            if type(rollout) is not dict or set(rollout) != rollout_fields:
                raise ValueError("brain prospective rollout fields do not match")
            rollout_shapes = {
                "h_next": (int(world.deter_dim),),
                "z_next": (int(world.stoch_dim),),
                "z_next_raw": (int(world.stoch_dim),),
                "x_next": (int(world.ssm_dim),),
                "s_next": (int(world.state_dim),),
                "action_enc": (int(world.action_dim),),
                "embodied_action": (int(world.action_dim),),
                "embodiment_context": (
                    int(world.embodiment_context_dim),),
                "r_pred": (),
                "d_prob": (),
                "d_tr": (int(world.state_dim),),
                "d_ph": (int(world.state_dim),),
            }
            for name, value in rollout.items():
                if name in ("pst_binding", "loss_pst_bind"):
                    continue
                self.ValidateBatchedTensor(
                    value,
                    batchSize,
                    device,
                    dtype,
                    "brain prospective rollout " + name,
                    trailingShape=rollout_shapes[name])
            binding = rollout["pst_binding"]
            binding_fields = {
                "bound_mu",
                "delta_mu",
                "bind_gate",
                "pst_context",
                "slot_binding_weight",
                "observed_weight",
                "memory_weight",
                "query_pool_weight",
                "pst_summary_pred",
                "loss_pst_bind",
                "embodiment_context",
            }
            if type(binding) is not dict or set(binding) != binding_fields:
                raise ValueError("brain prospective binding fields do not match")
            binder = world.pst_binder
            binding_shapes = {
                "bound_mu": (int(world.stoch_dim),),
                "delta_mu": (int(world.stoch_dim),),
                "bind_gate": (int(world.stoch_dim),),
                "pst_context": (int(binder.slot_dim),),
                "slot_binding_weight": (int(world.physical_slots),),
                "observed_weight": (int(world.physical_slots),),
                "memory_weight": (int(world.physical_slots),),
                "query_pool_weight": (int(binder.query_count),),
                "pst_summary_pred": (int(binder.slot_dim),),
                "embodiment_context": (
                    int(world.embodiment_context_dim),),
            }
            for name, value in binding.items():
                if name == "loss_pst_bind":
                    if (
                        not torch.is_tensor(value)
                        or value.dim() != 0
                        or value.device != device
                        or value.dtype != dtype
                        or not bool(torch.isfinite(value).item())
                    ):
                        raise ValueError("brain prospective binding loss is invalid")
                else:
                    self.ValidateBatchedTensor(
                        value,
                        batchSize,
                        device,
                        dtype,
                        "brain prospective binding " + name,
                        trailingShape=binding_shapes[name])
            loss = rollout["loss_pst_bind"]
            if (
                not torch.is_tensor(loss)
                or loss.dim() != 0
                or loss.device != device
                or loss.dtype != dtype
                or not bool(torch.isfinite(loss).item())
                or not torch.equal(loss, binding["loss_pst_bind"])
            ):
                raise ValueError("brain prospective rollout loss is invalid")

    def ValidateOcrRuntimeState(
            self,
            state: Any,
            batchSize: int,
        ) -> None:
            if (
                type(state) is not dict
                or set(state) != {
                    "temporal_step",
                    "last_batch_size",
                    "texts",
                    "tracks"}
                or type(state["temporal_step"]) is not int
                or state["temporal_step"] < 0
                or type(state["last_batch_size"]) is not int
                or state["last_batch_size"] not in (0, int(batchSize))
            ):
                raise ValueError("brain OCR runtime state is invalid")
            texts = state["texts"]
            expected_text_rows = int(state["last_batch_size"])
            if (
                type(texts) is not list
                or len(texts) != expected_text_rows
                or any(
                    type(row) is not list
                    or any(type(value) is not str for value in row)
                    for row in texts)
            ):
                raise ValueError("brain OCR text state is invalid")
            tracks = state["tracks"]
            if type(tracks) is not dict:
                raise ValueError("brain OCR track state is invalid")
            temporal_steps = int(self.OCR.temporalSteps)
            if (
                state["last_batch_size"] == 0
                and (texts or tracks)
            ):
                raise ValueError("brain empty OCR state is inconsistent")
            if (
                temporal_steps <= 0
                and (state["temporal_step"] != 0 or tracks)
            ):
                raise ValueError("brain OCR tracks require temporal fusion")
            for row_index, row_tracks in tracks.items():
                if (
                    type(row_index) is not int
                    or row_index < 0
                    or row_index >= int(batchSize)
                    or type(row_tracks) is not list
                ):
                    raise ValueError("brain OCR track row is invalid")
                for track in row_tracks:
                    if (
                        type(track) is not OcrTrack
                        or type(track.obs) is not deque
                        or len(track.obs) < 1
                        or track.obs.maxlen != temporal_steps
                        or type(track.age) is not int
                        or track.age < 0
                        or track.age > temporal_steps
                    ):
                        raise ValueError("brain OCR track is invalid")
                    for observation in track.obs:
                        if (
                            type(observation) is not OcrLineObs
                            or type(observation.box) is not tuple
                            or len(observation.box) != 4
                            or any(
                                type(coordinate) is not int
                                for coordinate in observation.box)
                            or type(observation.text) is not str
                            or type(observation.det_score) is not float
                            or not math.isfinite(observation.det_score)
                            or type(observation.rec_conf) is not float
                            or not math.isfinite(observation.rec_conf)
                            or type(observation.step) is not int
                            or observation.step < 0
                            or observation.step > state["temporal_step"]
                        ):
                            raise ValueError("brain OCR observation is invalid")

    @torch.no_grad()
    def ValidateBufferState(
            self,
            state: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], int]:
            if type(state) is not dict or set(state) != BRAIN_RUNTIME_BUFFER_FIELDS:
                raise ValueError("brain runtime buffer fields do not match")
            if (
                type(state["schema_version"]) is not int
                or state["schema_version"] != BRAIN_RUNTIME_SCHEMA_VERSION
            ):
                raise ValueError("brain runtime schema does not match")
            if (
                state["contract_id"]
                != self.robot_contract_view.contract_id
                or state["model_signature"] != self.model_signature
            ):
                raise ValueError("brain runtime identity does not match")
            batch_size = state["batch_size"]
            if type(batch_size) is not int or batch_size < 1:
                raise ValueError("brain runtime batch size is invalid")
            cognitive = self.MoveRuntimeStateToModel(
                copy.deepcopy(state["cognitive_state"]))
            self.ValidateCognitiveRuntimeSchema(cognitive, batch_size)
            return cognitive, batch_size

    @torch.no_grad()
    def ValidateCognitiveRuntimeSchema(
            self,
            cognitive: Dict[str, Any],
            batchSize: int,
        ) -> None:
            expected_cognitive = {
                "tensors",
                "world",
                "memory",
                "memory_cognitive_cache",
                "attention",
                "value",
                "decision_eligibility",
                "consciousness",
                "plan",
                "cached_target",
                "commitment",
                "slow_cache",
                "replay_history",
                "replay_timeline_version",
                "replay_transaction_version",
                "previous_visual",
                "visual_buffer",
                "visual_valid_buffer",
                "perception_buffer",
                "prospective_visual",
                "compute_previous",
                "ocr",
            }
            if type(cognitive) is not dict or set(cognitive) != expected_cognitive:
                raise ValueError("brain cognitive runtime fields do not match")
            device = self.prev_mem.device
            dtype = self.prev_mem.dtype
            tensors = cognitive["tensors"]
            expected_tensors = set(
                self.ContractRuntimeTensorNames()) | {"prev_self_sem"}
            if type(tensors) is not dict or set(tensors) != expected_tensors:
                raise ValueError("brain cognitive tensor fields do not match")
            for name in self.ContractRuntimeTensorNames():
                value = tensors[name]
                template = getattr(self, name)
                if (
                    not torch.is_tensor(value)
                    or value.dim() < 1
                    or int(value.size(0)) != int(batchSize)
                    or tuple(value.shape[1:]) != tuple(template.shape[1:])
                    or value.dtype != template.dtype
                    or value.device != template.device
                    or (
                        value.is_floating_point()
                        and not bool(torch.isfinite(value).all().item()))
                ):
                    raise ValueError(
                        "brain cognitive tensor shape or type does not match")
            previous_self = tensors["prev_self_sem"]
            if previous_self is not None and (
                not torch.is_tensor(previous_self)
                or previous_self.dim() != 2
                or int(previous_self.size(0)) != int(batchSize)
                or tuple(previous_self.shape[1:])
                != tuple(self.conscious._last_sem.shape[1:])
                or previous_self.dtype != self.conscious._last_sem.dtype
                or previous_self.device != self.conscious._last_sem.device
                or not bool(torch.isfinite(previous_self).all().item())
            ):
                raise ValueError("brain previous self state is invalid")
            world_state = cognitive["world"]
            if type(world_state) is not dict or set(world_state) != {"h", "z", "x"}:
                raise ValueError("brain world runtime fields do not match")
            world_template = dict(zip(
                ("h", "z", "x"),
                self.ContractWorld().ExportState()))
            for name, template in world_template.items():
                value = world_state[name]
                if (
                    not torch.is_tensor(value)
                    or value.dim() < 1
                    or int(value.size(0)) != int(batchSize)
                    or tuple(value.shape[1:]) != tuple(template.shape[1:])
                    or value.dtype != template.dtype
                    or value.device != template.device
                    or not bool(torch.isfinite(value).all().item())
                ):
                    raise ValueError("brain world runtime state is invalid")
            consciousness = cognitive["consciousness"]
            consciousness_fields = {
                "_dev_trace",
                "_last_self_intent",
                "_last_sem",
                "_state_valid",
                "_step",
            }
            if (
                type(consciousness) is not dict
                or set(consciousness) != consciousness_fields
            ):
                raise ValueError("brain consciousness runtime fields do not match")
            for name in consciousness_fields:
                value = consciousness[name]
                template = getattr(self.conscious, name)
                expected_shape = (
                    tuple(template.shape)
                    if name == "_step"
                    else (int(batchSize),) + tuple(template.shape[1:]))
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != expected_shape
                    or value.dtype != template.dtype
                    or value.device != template.device
                    or (
                        value.is_floating_point()
                        and not bool(torch.isfinite(value).all().item()))
                ):
                    raise ValueError(
                        "brain consciousness runtime state is invalid")
            attention = cognitive["attention"]
            if (
                type(attention) is not dict
                or set(attention) != {"fusion", "mhsa"}
                or not isinstance(attention["mhsa"], list)
                or len(attention["mhsa"])
                != len(self.RuntimeModule(self.attn).temporal_blocks)
                or any(
                    type(item) is not dict
                    or set(item) != {"U", "V"}
                    for item in attention["mhsa"])
            ):
                raise ValueError("brain attention runtime fields do not match")
            plan = cognitive["plan"]
            plan_template = self.contract_neuro_symbolic.ExportPlanState()
            if type(plan) is not dict or set(plan) != set(plan_template):
                raise ValueError("brain plan runtime fields do not match")
            for name, template in plan_template.items():
                value = plan[name]
                if (
                    not torch.is_tensor(value)
                    or value.dim() < 1
                    or int(value.size(0)) != int(batchSize)
                    or tuple(value.shape[1:]) != tuple(template.shape[1:])
                    or value.dtype != template.dtype
                    or value.device != template.device
                    or (
                        value.is_floating_point()
                        and not bool(torch.isfinite(value).all().item()))
                ):
                    raise ValueError("brain plan runtime state is invalid")
            compute_previous = cognitive["compute_previous"]
            if (
                not torch.is_tensor(compute_previous)
                or tuple(compute_previous.shape) != (
                    int(batchSize),
                    self.robot_contract_view.end_effector_count)
                or compute_previous.dtype != torch.bool
                or compute_previous.device
                != self.cognitive_compute_gate.PreviousChildEnabled.device
            ):
                raise ValueError("brain compute-gate runtime state is invalid")
            replay_history = cognitive["replay_history"]
            replay_timeline = cognitive["replay_timeline_version"]
            replay_transaction = cognitive["replay_transaction_version"]
            if (
                not isinstance(replay_history, list)
                or len(replay_history) != int(batchSize)
                or any(not isinstance(history, deque)
                       for history in replay_history)
                or type(replay_timeline) is not int
                or replay_timeline < 0
                or type(replay_transaction) is not int
                or replay_transaction < 0
            ):
                raise ValueError("brain replay runtime state is invalid")
            episode_version = tensors["ContractReplayEpisodeVersion"]
            for row_index, history in enumerate(replay_history):
                previous_timeline = -1
                for trace in history:
                    if (
                        type(trace) is not ContractReplayTrace
                        or trace.episode_version != int(
                            episode_version[row_index].item())
                        or trace.timeline_version <= previous_timeline
                        or trace.timeline_version >= replay_timeline
                    ):
                        raise ValueError("brain replay trace identity is invalid")
                    previous_timeline = trace.timeline_version
            visual_buffer = cognitive["visual_buffer"]
            visual_valid_buffer = cognitive["visual_valid_buffer"]
            perception_buffer = cognitive["perception_buffer"]
            if (
                type(visual_buffer) is not list
                or type(visual_valid_buffer) is not list
                or type(perception_buffer) is not list
                or len(visual_buffer) != len(visual_valid_buffer)
                or len(visual_buffer) != len(perception_buffer)
                or len(visual_buffer) > self.SEQ_LEN
            ):
                raise ValueError("brain visual runtime state is invalid")
            previous_visual = cognitive["previous_visual"]
            if previous_visual is not None:
                self.ValidateVisualRuntimeState(
                    previous_visual,
                    batchSize,
                    device,
                    dtype,
                    "brain previous visual")
            for index, visual_state in enumerate(visual_buffer):
                self.ValidateVisualRuntimeState(
                    visual_state,
                    batchSize,
                    device,
                    dtype,
                    "brain visual buffer " + str(index))
                valid = visual_valid_buffer[index]
                perception = perception_buffer[index]
                if (
                    not torch.is_tensor(valid)
                    or tuple(valid.shape) != (int(batchSize),)
                    or valid.dtype != torch.bool
                    or valid.device != device
                    or not torch.is_tensor(perception)
                    or tuple(perception.shape)
                    != tuple(visual_state.IntegratedFeat.shape)
                    or perception.dtype != dtype
                    or perception.device != device
                    or not bool(torch.isfinite(perception).all().item())
                    or not torch.equal(
                        perception,
                        visual_state.IntegratedFeat)
                ):
                    raise ValueError("brain visual runtime state is invalid")
            self.ValidateCachedTargetRuntimeState(
                cognitive["cached_target"],
                batchSize,
                device,
                dtype)
            commitment = cognitive["commitment"]
            self.ValidateCommitmentRuntimeState(
                commitment,
                batchSize,
                device,
                dtype)
            self.ValidateSlowCacheRuntimeState(
                cognitive["slow_cache"],
                commitment,
                batchSize,
                device,
                dtype)
            self.ValidateProspectiveVisualRuntimeState(
                cognitive["prospective_visual"],
                batchSize,
                device,
                dtype)
            self.ValidateOcrRuntimeState(
                cognitive["ocr"],
                batchSize)

    @torch.no_grad()
    def ApplyCognitiveRuntimeState(
            self,
            cognitive: Dict[str, Any],
            batchSize: int,
        ) -> None:
            tensors = cognitive["tensors"]
            for name in self.ContractRuntimeTensorNames():
                setattr(self, name, tensors[name])
            self.prev_self_sem = tensors["prev_self_sem"]
            world_state = cognitive["world"]
            self.ContractWorld().ImportState(
                world_state["h"],
                world_state["z"],
                world_state["x"])
            self.mem.ImportTransientState(cognitive["memory"])
            self.mem.ImportCognitiveCacheState(
                cognitive["memory_cognitive_cache"],
                modelSignature=self.model_signature,
                batchSize=batchSize)
            self.RuntimeModule(self.attn).ImportState(
                cognitive["attention"])
            self.RuntimeModule(self.critic).ImportState(
                cognitive["value"])
            self.RuntimeModule(self.actor).ImportEligibilityState(
                cognitive["decision_eligibility"],
                batchSize)
            for name, value in cognitive["consciousness"].items():
                setattr(self.conscious, name, value)
            self.contract_neuro_symbolic.ImportPlanState(
                cognitive["plan"])
            self.ContractCachedTarget = cognitive["cached_target"]
            self.ContractIntentionCommitmentState = cognitive["commitment"]
            self.ContractSlowCognitiveCache = cognitive["slow_cache"]
            replay_history = cognitive["replay_history"]
            replay_timeline = cognitive["replay_timeline_version"]
            replay_transaction = cognitive["replay_transaction_version"]
            self.ContractReplayHistory = replay_history
            self.ContractReplayTimelineVersion = replay_timeline
            self.ContractReplayTransactionVersion = replay_transaction
            self.prev_visual_state = cognitive["previous_visual"]
            self.visual_state_buffer = cognitive["visual_buffer"]
            self.visual_state_valid_buffer = cognitive[
                "visual_valid_buffer"]
            self.perc_buffer = cognitive["perception_buffer"]
            self.prospective_visual_prediction = cognitive[
                "prospective_visual"]
            self.cognitive_compute_gate.PreviousChildEnabled = cognitive[
                "compute_previous"]
            ocr = cognitive["ocr"]
            self.OCR._temporal_step = int(ocr["temporal_step"])
            self.OCR._last_batch_size = int(ocr["last_batch_size"])
            self.OCR._last_ocr_texts_batch = copy.deepcopy(ocr["texts"])
            self.OCR._tracks_by_bi = copy.deepcopy(ocr["tracks"])
            self.ContractRuntimeBatch = int(batchSize)

    @torch.no_grad()
    def ImportBuffers(self, state: Dict[str, Any]) -> None:
            cognitive, batch_size = self.ValidateBufferState(state)
            previous_state = self.ExportBuffers()
            try:
                self.ApplyCognitiveRuntimeState(cognitive, batch_size)
            except BaseException as error:
                try:
                    previous_cognitive = self.MoveRuntimeStateToModel(
                        copy.deepcopy(previous_state["cognitive_state"]))
                    self.ApplyCognitiveRuntimeState(
                        previous_cognitive,
                        int(previous_state["batch_size"]))
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "brain runtime import failed and rollback failed") from rollback_error
                raise error


class Agent:
    def __init__(
        self,
        brain: BrainCore,
        isTrain: bool = False,
        device: Union[str, torch.device] = "cpu",
        worldMemoryPath: Optional[str] = None,
        memMemoryPath: Optional[str] = None,
    ) -> None:
        if not isinstance(brain, BrainCore):
            raise TypeError("Agent requires BrainCore")
        self.brain = brain
        self.is_train = bool(isTrain)
        self.device = torch.device(device)
        self.brain.to(self.device)
        self.world_memory_path = worldMemoryPath
        self.memory_path = memMemoryPath
        self.world_frame_id: Optional[str] = None
        self.world_memory_batch_size = 0
        self.transport_manual_lr = 2e-4
        self.transport_manual_max_norm = 1.0
        self.transport_manual_weight_decay = 0.0
        if self.is_train:
            actor_parameters = self.ActorOptimizerParameters()
            critic_parameters = self.CollectTrainableParams(
                self.brain.critic)
            world_parameters = self.WorldOptimizerParameters()
            if {
                id(parameter)
                for parameter in actor_parameters
            }.intersection({
                id(parameter)
                for parameter in world_parameters
            }):
                raise RuntimeError(
                    "policy and world optimizers cannot share parameters")
            if not actor_parameters or not world_parameters:
                raise RuntimeError(
                    "contract training requires trainable policy and world parameters")
            if not critic_parameters:
                critic_parameters = [next(self.brain.critic.parameters())]
            self.opt_actor = torch.optim.Adam(
                actor_parameters,
                lr=3e-4)
            self.opt_critic = torch.optim.Adam(
                critic_parameters,
                lr=2e-4)
            self.opt_world = torch.optim.Adam(
                world_parameters,
                lr=2e-4)

    def ActorOptimizerModules(self) -> Tuple[nn.Module, ...]:
        names = (
            "perc",
            "attn",
            "mem",
            "actor",
            "conscious",
            "intention",
            "contract_pst_builder",
            "goal_manager",
            "goal_grounding",
            "contract_neuro_symbolic_grounder",
            "contract_neuro_symbolic",
            "packed_decision_decoupler",
            "packed_temporal_gate",
            "contract_physical_adapter",
            "contract_joint_motion_action_adapter",
            "contract_action_agency_encoder",
            "contract_layer_agency_fuser",
            "contract_entity_summary_fuser",
            "world_abstract_decision_adapter",
            "goal_relation_adapter",
            "goal_object_adapter",
            "goal_requirement_head",
            "attention_memory_top_down_adapter",
            "attention_goal_top_down_adapter")
        modules = list(
            module
            for name in names
            for module in (getattr(self.brain, name, None),)
            if isinstance(module, nn.Module))
        world = getattr(self.brain, "world", None)
        world = world.base if hasattr(world, "base") else world
        if isinstance(world, nn.Module):
            for name in (
                "world_abstract_projector",
                "entity_conscious_encoder",
            ):
                module = getattr(world, name, None)
                if isinstance(module, nn.Module):
                    modules.append(module)
        return tuple(modules)

    def ActorOptimizerParameters(self) -> List[nn.Parameter]:
        parameters = self.CollectTrainableParams(
            *self.ActorOptimizerModules())
        seen = {id(parameter) for parameter in parameters}
        for name in (
            "contract_action_agency_gain",
            "contract_layer_agency_gain",
            "contract_entity_summary_gain",
            "world_abstract_decision_gain",
            "attention_top_down_gain",
            "option_skill_gain",
        ):
            parameter = getattr(self.brain, name, None)
            if not isinstance(parameter, nn.Parameter):
                continue
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
        return parameters

    def WorldOptimizerModules(self) -> Tuple[nn.Module, ...]:
        return (self.brain.world,)

    def WorldOptimizerParameters(self) -> List[nn.Parameter]:
        actor_parameter_ids = {
            id(parameter)
            for parameter in self.ActorOptimizerParameters()
        }
        return [
            parameter
            for parameter in self.CollectTrainableParams(
                *self.WorldOptimizerModules())
            if id(parameter) not in actor_parameter_ids]

    def CollectTrainableParams(
        self,
        *modules: nn.Module,
    ) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        seen = set()
        for module in modules:
            for parameter in module.parameters():
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameters.append(parameter)
            if hasattr(module, "CandParameters"):
                for parameter in module.CandParameters():
                    if not parameter.requires_grad or id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    def SyncOptimizerParameters(
        self,
        optimizer: torch.optim.Optimizer,
        parameters: List[nn.Parameter],
    ) -> None:
        if len(optimizer.param_groups) != 1:
            raise RuntimeError(
                "dynamic optimizer synchronization requires one parameter group")
        desired: List[nn.Parameter] = []
        desired_ids = set()
        for parameter in parameters:
            if id(parameter) in desired_ids:
                continue
            desired_ids.add(id(parameter))
            desired.append(parameter)
        optimizer.param_groups[0]["params"] = desired
        for parameter in tuple(optimizer.state):
            if id(parameter) not in desired_ids:
                del optimizer.state[parameter]

    def SyncTrainableOptimizers(self) -> None:
        if not self.is_train:
            return
        self.SyncOptimizerParameters(
            self.opt_actor,
            self.ActorOptimizerParameters())
        self.SyncOptimizerParameters(
            self.opt_critic,
            self.CollectTrainableParams(self.brain.critic))
        self.SyncOptimizerParameters(
            self.opt_world,
            self.WorldOptimizerParameters())

    def ClearTrainableOptimizerState(self) -> int:
        if not self.is_train:
            return 0
        cleared = 0
        for optimizer in (
            self.opt_actor,
            self.opt_critic,
            self.opt_world,
        ):
            cleared += len(optimizer.state)
            optimizer.state.clear()
        return cleared

    def OptimizerParameters(
        self,
        optimizers: Optional[Tuple[torch.optim.Optimizer, ...]] = None,
    ) -> List[nn.Parameter]:
        if not self.is_train:
            return []
        selected = (
            (self.opt_actor, self.opt_critic, self.opt_world)
            if optimizers is None
            else tuple(optimizers))
        parameters: List[nn.Parameter] = []
        seen = set()
        for optimizer in selected:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if not parameter.requires_grad or id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return parameters

    def NamedOnlineWrappers(self) -> Tuple[Tuple[str, nn.Module], ...]:
        if not self.brain.is_online_learning:
            return ()
        return tuple(
            (name, getattr(self.brain, name))
            for name in ONLINE_WRAPPER_ROOTS
            if hasattr(getattr(self.brain, name), "ExportCandidateState"))

    @torch.no_grad()
    def ExportOnlineCandidateState(self) -> Dict[str, Any]:
        model_signature = getattr(self.brain, "model_signature", None)
        if type(model_signature) is not str or not model_signature:
            raise ValueError("brain model signature is invalid")
        return {
            "model_signature": model_signature,
            "wrappers": {
                name: wrapper.ExportCandidateState()
                for name, wrapper in self.NamedOnlineWrappers()},
        }

    def ValidateOnlineCandidateState(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if type(state) is not dict:
            raise TypeError("online candidate state must be a dictionary")
        if set(state) != {"model_signature", "wrappers"}:
            raise ValueError("online candidate state fields do not match")
        model_signature = state["model_signature"]
        expected_signature = getattr(self.brain, "model_signature", None)
        if type(model_signature) is not str or not model_signature:
            raise ValueError("online candidate model signature is invalid")
        if type(expected_signature) is not str or not expected_signature:
            raise ValueError("brain model signature is invalid")
        if model_signature != expected_signature:
            raise ValueError("online candidate model signature does not match")
        wrapper_state = state["wrappers"]
        if type(wrapper_state) is not dict:
            raise TypeError("online candidate wrappers must be a dictionary")
        wrappers = dict(self.NamedOnlineWrappers())
        if set(wrapper_state) != set(wrappers):
            raise ValueError("online candidate wrappers do not match the brain")
        for name, wrapper in wrappers.items():
            validator = getattr(wrapper, "ValidateCandidateState", None)
            if not callable(validator):
                raise TypeError(
                    "online candidate wrapper does not expose validation")
            validator(wrapper_state[name])
        return wrapper_state

    @torch.no_grad()
    def ImportOnlineCandidateState(self, state: Dict[str, Any]) -> None:
        wrapper_state = self.ValidateOnlineCandidateState(state)
        wrappers = dict(self.NamedOnlineWrappers())
        for name, wrapper in wrappers.items():
            wrapper.ImportCandidateState(wrapper_state[name])

    @torch.no_grad()
    def ResetOnlineCandidateState(self) -> None:
        for _, wrapper in self.NamedOnlineWrappers():
            wrapper.Update("rollback")
        self.SyncTrainableOptimizers()

    def GetRuntimeWorld(self) -> RSSMWorldModel:
        return self.brain.ContractWorld()

    def BindWorldMemoryContext(
        self,
        worldFrameId: str,
        *,
        batchSize: int,
        loadPersistent: bool = True,
    ) -> None:
        if type(worldFrameId) is not str or not worldFrameId:
            raise ValueError("worldFrameId must be non-empty")
        if type(batchSize) is not int or batchSize < 1:
            raise ValueError("batchSize must be positive")
        if self.world_frame_id is not None and (
            self.world_frame_id != worldFrameId
            or self.world_memory_batch_size != batchSize
        ):
            raise RuntimeError("world memory context cannot change")
        world = self.GetRuntimeWorld()
        if self.world_frame_id is None:
            world.BindMemoryContext(
                self.brain.calibration_id,
                worldFrameId)
            self.world_frame_id = worldFrameId
            self.world_memory_batch_size = batchSize
        parameter = next(self.brain.parameters())
        if self.brain.ContractRuntimeBatch != batchSize:
            self.brain.ResetContractCognitiveState(
                batchSize,
                self.device,
                parameter.dtype)
        if loadPersistent:
            if self.world_memory_path is not None:
                self.LoadWorldMemory(self.world_memory_path)
            if self.memory_path is not None:
                self.LoadMemory(self.memory_path)

    def LoadWorldMemory(self, path: str) -> None:
        world = self.GetRuntimeWorld()
        if os.path.exists(path) and os.path.getsize(path) > 0:
            payload = torch.load(
                path,
                map_location=self.device,
                weights_only=True)
            expected = {
                "schema_version",
                "contract_id",
                "model_signature",
                "batch_size",
                "world",
            }
            if type(payload) is not dict or set(payload) != expected:
                raise ValueError("world memory fields do not match")
            if (
                type(payload["schema_version"]) is not int
                or payload["schema_version"]
                != WORLD_MEMORY_ARTIFACT_SCHEMA_VERSION
                or type(payload["batch_size"]) is not int
                or payload["contract_id"]
                != self.brain.robot_contract_view.contract_id
                or payload["model_signature"]
                != self.brain.model_signature
                or payload["batch_size"]
                != self.world_memory_batch_size
            ):
                raise ValueError("world memory identity does not match")
            world.ImportMemoryPayload(
                payload["world"],
                batchSize=self.world_memory_batch_size)
        else:
            world.EnsureB(self.world_memory_batch_size)

    def SaveWorldMemory(self, path: str) -> None:
        self.AtomicSave({
            "schema_version": WORLD_MEMORY_ARTIFACT_SCHEMA_VERSION,
            "contract_id": self.brain.robot_contract_view.contract_id,
            "model_signature": self.brain.model_signature,
            "batch_size": self.world_memory_batch_size,
            "world": self.GetRuntimeWorld().ExportMemoryPayload(),
        }, path)

    def LoadMemory(self, path: str) -> None:
        memory = self.brain.mem
        if os.path.exists(path) and os.path.getsize(path) > 0:
            payload = torch.load(
                path,
                map_location=self.device,
                weights_only=True)
            expected = {
                "schema_version",
                "contract_id",
                "model_signature",
                "batch_size",
                "memory",
            }
            if type(payload) is not dict or set(payload) != expected:
                raise ValueError("memory fields do not match")
            if (
                type(payload["schema_version"]) is not int
                or payload["schema_version"] != AGENT_MEMORY_SCHEMA_VERSION
                or type(payload["batch_size"]) is not int
                or payload["contract_id"]
                != self.brain.robot_contract_view.contract_id
                or payload["model_signature"]
                != self.brain.model_signature
                or payload["batch_size"]
                != self.world_memory_batch_size
            ):
                raise ValueError("memory identity does not match")
            memory.ImportDurableState(payload["memory"])
        else:
            memory.EnsureB(self.world_memory_batch_size)

    def SaveMemory(self, path: str) -> None:
        memory = self.brain.mem
        memory.FlushPendingWrites()
        self.AtomicSave({
            "schema_version": AGENT_MEMORY_SCHEMA_VERSION,
            "contract_id": self.brain.robot_contract_view.contract_id,
            "model_signature": self.brain.model_signature,
            "batch_size": self.world_memory_batch_size,
            "memory": memory.ExportDurableState(),
        }, path)

    @staticmethod
    def AtomicSave(payload: Dict[str, Any], path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix="brain.",
            suffix=".tmp",
            dir=directory or ".")
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def CommitPendingWorldAutosave(self) -> None:
        if self.world_memory_path is None:
            return
        world = self.GetRuntimeWorld()
        if world.HasMemoryAutosaveRequest():
            self.SaveWorldMemory(self.world_memory_path)
            world.AcknowledgeMemoryAutosaveRequest()

    def LoadBrainWeights(self, path: str) -> None:
        payload = torch.load(
            path,
            map_location=self.device,
            weights_only=True)
        expected = {
            "schema_version",
            "calibration_id",
            "model_contract_id",
            "brain",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise ValueError("brain parameter fields do not match")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != BRAIN_RUNTIME_SCHEMA_VERSION
            or payload["calibration_id"] != self.brain.calibration_id
            or payload["model_contract_id"]
            != self.brain.model_signature
        ):
            raise ValueError("brain parameter identity does not match")
        LoadDeploymentModelState(self.brain, payload["brain"])

    def EncodeEmbodimentFeedback(
        self,
        feedbackPacket: BrainFeedbackPacket,
        *,
        batchSize: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.brain.EncodeEmbodimentFeedback(
            feedbackPacket,
            batchSize=batchSize,
            device=self.device)

    def ExportModuleMessagerData(
        self,
        nSteps: int = 0,
    ) -> Dict[str, Any]:
        return self.brain.moduleMessager.ExportDict(nSteps=nSteps)

    def MovePerceptionTargets(
        self,
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if type(targets) is not dict:
            raise TypeError("perception targets must be a dictionary")
        return {
            name: (
                value.to(self.device)
                if torch.is_tensor(value)
                else value)
            for name, value in targets.items()}

    def RunStep(
        self,
        actInput: ContractAgentActInput,
        *,
        enableGrad: Optional[bool] = None,
        modelTraining: Optional[bool] = None,
    ) -> BrainStepOutput:
        if type(actInput) is not ContractAgentActInput:
            raise TypeError("Agent.RunStep requires ContractAgentActInput")
        model_training = (
            self.is_train
            if modelTraining is None
            else bool(modelTraining))
        self.brain.train(model_training)
        gradient_enabled = (
            self.is_train
            if enableGrad is None
            else bool(enableGrad))
        context = (
            torch.enable_grad()
            if gradient_enabled
            else torch.no_grad())
        with context:
            return self.brain.StepContract(ContractBrainStepInput(
                frame=actInput.frame.to(self.device),
                text_ext=actInput.text_ext,
                reward_ext=(
                    None
                    if actInput.reward is None
                    else actInput.reward.to(self.device)),
                done_flag=(
                    None
                    if actInput.done is None
                    else actInput.done.to(self.device)),
                is_train=self.is_train,
                sample_actions=actInput.sample_actions,
                deterministic_actor=actInput.deterministic_actor,
                depth=actInput.depth.to(self.device),
                depth_valid=actInput.depth_valid.to(self.device),
                feedback_packet=actInput.feedback_packet,
                text_trust=actInput.text_trust,
                perception_targets=self.MovePerceptionTargets(
                    actInput.perception_targets)))

    def Act(
        self,
        actInput: ContractAgentActInput,
    ) -> ContractAgentActOutput:
        output = self.RunStep(actInput)
        return ContractAgentActOutput(
            packed_target=output.decision["packed_target"],
            packed_temporal=output.decision["packed_temporal"],
            decision=output.decision,
            ocr=output.ocr,
            intention_texts=output.intention_texts)

    def ResetHebbianMemory(self) -> None:
        self.brain.ResetHebbianMemory()

    def ResetBrainState(self, batchSize: int = 1) -> None:
        world = self.GetRuntimeWorld()
        world.ResetState(batchSize=batchSize)
        world.ResetPhysicalState()
        self.brain.RuntimeModule(self.brain.critic).ResetState()
        self.brain.mem.ResetEpisodeState()
        self.brain.conscious.ResetState()
        self.brain.RuntimeModule(
            self.brain.intention).ResetTransientLossCache()
        self.brain.OCR.ResetTemporal()
        self.brain.ResetHebbianMemory()
        parameter = next(self.brain.parameters())
        self.brain.ResetContractCognitiveState(
            batchSize=batchSize,
            device=self.device,
            dtype=parameter.dtype)

    def CaptureCriticTransportGrad(self) -> Dict[str, float]:
        critic = self.brain.critic
        if hasattr(critic, "CaptureTransportGrad"):
            return critic.CaptureTransportGrad(clearParamGrad=True)
        return {
            "captured": 0.0,
            "grad_norm": 0.0,
            "accum_steps": 0.0,
        }

    def ApplyCriticTransportManualGrad(self) -> Dict[str, float]:
        critic = self.brain.critic
        if hasattr(critic, "ApplyTransportManualGrad"):
            return critic.ApplyTransportManualGrad(
                lr=self.transport_manual_lr,
                maxNorm=self.transport_manual_max_norm,
                weightDecay=self.transport_manual_weight_decay,
                clear=True)
        return {
            "updated": 0.0,
            "grad_norm": 0.0,
            "scale": 1.0,
        }

    def ClearCriticTransportGradAccumulator(self) -> None:
        critic = self.brain.critic
        if hasattr(critic, "ClearTransportGradAccumulator"):
            critic.ClearTransportGradAccumulator()

    def AfterOptimizerStep(self) -> None:
        if hasattr(self.brain.critic, "AfterOptimizerStep"):
            self.brain.critic.AfterOptimizerStep()

    def UpdateWrappers(
        self,
        wrappers: Tuple[nn.Module, ...],
        action: str,
        **kwargs: Any,
    ) -> List[Any]:
        results = []
        for wrapper in wrappers:
            if hasattr(wrapper, "Update"):
                results.append(wrapper.Update(action, **kwargs))
        self.SyncTrainableOptimizers()
        return results

    def UpdateAllWrappers(
        self,
        action: str,
        **kwargs: Any,
    ) -> List[Any]:
        return self.UpdateWrappers(
            tuple(
                getattr(self.brain, name)
                for name in ONLINE_WRAPPER_ROOTS),
            action,
            **kwargs)

    def GetModuleParamsCount(
        self,
        onlyTrainable: bool = True,
    ) -> Dict[str, int]:
        def CountModuleParams(module: Optional[nn.Module]) -> int:
            if module is None:
                return 0
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if not onlyTrainable or parameter.requires_grad)

        names = (
            "perc",
            "attn",
            "mem",
            "actor",
            "world",
            "critic",
            "conscious",
            "intention",
            "goal_manager",
            "contract_neuro_symbolic",
            "packed_decision_decoupler")
        counts = {
            name: CountModuleParams(getattr(self.brain, name, None))
            for name in names}
        counts["total"] = sum(counts.values())
        return counts


class TestAGICoreMTool:
    def RunAll(self) -> Dict[str, bool]:
        return {
            "ModelStateBoundary": callable(ExportBrainModelState),
            "ComputeModes": len(CognitiveComputeMode) == 4,
        }
