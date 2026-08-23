from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import Realm


PREDICATES: Tuple[str, ...] = (
    "observed",
    "localized",
    "reachable",
    "contactable",
    "graspable",
    "attached",
    "supported",
    "movable",
    "articulated",
    "container",
    "open",
    "closed",
    "aligned",
    "collision_free",
    "goal_satisfied",
    "recovery_needed",
    "in_execution",
    "feedback_fresh",
    "feedback_stale",
    "timeout_risk",
    "observation_needed",
    "safe_to_continue",
    "interruptible",
    "redispatch_needed",)

OPERATORS: Tuple[str, ...] = (
    "observe",
    "approach",
    "align",
    "reach",
    "contact",
    "grasp",
    "release",
    "lift",
    "lower",
    "transport",
    "place",
    "push",
    "pull",
    "slide",
    "rotate",
    "press",
    "open",
    "close",
    "insert",
    "extract",
    "pour",
    "wipe",
    "cut",
    "operate_tool",
    "handover",
    "hold",
    "retreat",
    "recover",
    "reobserve",
    "wait",
    "continue_execute",
    "cancel_execute",
    "failsafe_stop",
    "redispatch",)

TEMPORAL_PREDICATES: Tuple[str, ...] = (
    "in_execution",
    "feedback_fresh",
    "feedback_stale",
    "timeout_risk",
    "observation_needed",
    "safe_to_continue",
    "interruptible",
    "redispatch_needed",)

FAILURE_CAUSES: Tuple[str, ...] = (
    "occlusion",
    "unreachable",
    "slip",
    "collision",
    "misalignment",
    "low_confidence",
    "unstable_support",
    "articulation_blocked",
    "containment_error",
    "tool_error",
    "lost_attachment",
    "handover_failed",)


REALM_SELF_BODY = int(Realm.SELF_BODY)
REALM_EXTERNAL_PHYSICAL = int(Realm.EXTERNAL_PHYSICAL)
REALM_VIRTUAL_CONTENT = int(Realm.VIRTUAL_CONTENT)


def ResolveActuationReferenceWeights(
    pst: Dict[str, torch.Tensor],
    semanticReference: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    K = int(semanticReference.size(1))
    realm = pst["RealmProb"]
    verification = pst["VerificationConfidence"]
    interaction = pst["PhysicalInteractionProb"]
    presence = pst["PerceptualPresence"]

    direct_reference = (
        semanticReference
        * realm[..., REALM_EXTERNAL_PHYSICAL]
        * presence
        * interaction
        * verification)

    uv_confidence = pst["SurfaceUVConfidence"]
    virtual_child_reference = (
        semanticReference
        * realm[..., REALM_VIRTUAL_CONTENT]
        * presence
        * verification
        * uv_confidence)
    physical_parent_mass = (
        realm[..., REALM_SELF_BODY]
        + realm[..., REALM_EXTERNAL_PHYSICAL])
    parent_eligibility = (
        presence
        * physical_parent_mass
        * interaction
        * verification
        * pst["DisplaySurfaceProb"])
    verified_parent_path = (
        pst["SurfaceParentProb"][..., :K]
        * parent_eligibility.unsqueeze(1))
    surface_reference = torch.einsum(
        "bk,bkj->bj",
        virtual_child_reference,
        verified_parent_path)
    return {
        "direct_reference": direct_reference,
        "surface_reference": surface_reference,
        "actuation_reference": direct_reference,
        "virtual_child_reference": virtual_child_reference,
        "verified_parent_path": verified_parent_path,}


@dataclass
class SymbolicFact:
    name: str
    args: Tuple[str, ...]
    prob: torch.Tensor
    support: Optional[torch.Tensor]


@dataclass
class OperatorStep:
    op_name: str
    args: Tuple[str, ...]
    precond_score: torch.Tensor
    effect_score: torch.Tensor
    sampler_latent: torch.Tensor
    explanation: List[str]


@dataclass
class OperatorRationale:
    op_name: str
    args: Tuple[str, ...]
    selected_logit: torch.Tensor
    precond_score: torch.Tensor
    effect_score: torch.Tensor
    symbolic_score: torch.Tensor
    satisfied_preconditions: List[str]
    weak_preconditions: List[str]
    missing_goal_predicates: List[str]
    expected_effects: List[str]
    risk_causes: List[str]
    temporal_reasons: List[str]


@dataclass
class NeuroSymbolicOutput:
    facts: List[SymbolicFact]
    operator_logits: torch.Tensor
    plan_steps: List[OperatorStep]
    operator_rationales: List[OperatorRationale]
    plan_latent: torch.Tensor
    subgoal_feature: torch.Tensor
    constraint_tokens: torch.Tensor
    risk_cause_logits: torch.Tensor
    risk_cause_raw_logits: torch.Tensor
    failure_cause_logits: torch.Tensor
    failure_cause_raw_logits: torch.Tensor
    failure_gate_logits: torch.Tensor
    failure_gate: torch.Tensor
    invoke_mask: torch.Tensor
    same_operator: torch.Tensor
    operator_changed: torch.Tensor
    invoke_delta: torch.Tensor
    reference_drift: torch.Tensor
    temporal_logits: torch.Tensor
    temporal_reason_logits: torch.Tensor
    continue_guard_score: torch.Tensor
    interrupt_guard_score: torch.Tensor
    redispatch_guard_score: torch.Tensor


def _NeuroEndpointContract(
    robotMorphology: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if robotMorphology is None:
        raise TypeError("robot morphology is required")
    endpoint_count = int(robotMorphology.endpoint_count)
    node_count = int(robotMorphology.node_count)
    if endpoint_count < 0 or node_count < 1:
        raise ValueError("robot morphology counts are invalid")
    if not hasattr(robotMorphology, "EndpointSemanticDescriptor"):
        raise TypeError("robot morphology endpoint descriptor is required")
    action_dim = int(ModuleDim.RobotControlAxisDim)
    role_classes = int(ModuleDim.RobotBodyRoleClasses)
    side_classes = int(ModuleDim.RobotBodySideClasses)
    capability_dim = int(ModuleDim.RobotBodyCapabilityDim)
    endpoint_task_mask = torch.as_tensor(
        robotMorphology.endpoint_task_mask, dtype=torch.bool).detach().cpu()
    if tuple(endpoint_task_mask.shape) != (endpoint_count, action_dim):
        raise ValueError("robot morphology task mask does not match endpoint count")
    semantic = robotMorphology.EndpointSemanticDescriptor()
    required = (
        "controllable",
        "parent_node_index",
        "topology_depth",
        "task_mask",
        "role",
        "side",
        "capability",
        "node_role",
        "node_side",
        "node_capability",
        "parent_role",
        "parent_side",
        "parent_capability",
        "group_role_membership",
        "group_side_membership",
        "group_capability",
    )
    missing = tuple(name for name in required if name not in semantic)
    if missing:
        raise TypeError(
            "robot morphology endpoint descriptor is incomplete: "
            + ", ".join(missing))

    def vector(name: str, dtype: torch.dtype) -> torch.Tensor:
        value = torch.as_tensor(
            semantic[name], dtype=dtype).detach().cpu()
        if tuple(value.shape) != (endpoint_count,):
            raise ValueError(
                f"endpoint descriptor {name} shape is invalid")
        return value

    def matrix(
        name: str,
        width: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            semantic[name], dtype=dtype).detach().cpu()
        if tuple(value.shape) != (endpoint_count, int(width)):
            raise ValueError(
                f"endpoint descriptor {name} shape is invalid")
        return value

    role = {
        name: vector(name, torch.long)
        for name in ("role", "node_role", "parent_role")}
    side = {
        name: vector(name, torch.long)
        for name in ("side", "node_side", "parent_side")}
    parent_index = vector("parent_node_index", torch.long)
    if bool(((parent_index < -1) | (parent_index >= node_count)).any().item()):
        raise ValueError("endpoint parent node index is invalid")
    parent_valid = parent_index.ge(0)
    for name in ("role", "node_role"):
        if bool(((role[name] < 0) | (role[name] >= role_classes)).any().item()):
            raise ValueError("endpoint role semantic is invalid")
    for name in ("side", "node_side"):
        if bool(((side[name] < 0) | (side[name] >= side_classes)).any().item()):
            raise ValueError("endpoint side semantic is invalid")
    if bool((
        ((role["parent_role"] < 0) | (role["parent_role"] >= role_classes))
        & parent_valid
    ).any().item()):
        raise ValueError("endpoint parent role semantic is invalid")
    if bool((
        ((side["parent_side"] < 0) | (side["parent_side"] >= side_classes))
        & parent_valid
    ).any().item()):
        raise ValueError("endpoint parent side semantic is invalid")
    semantic_task_mask = matrix(
        "task_mask", action_dim, torch.bool)
    if not torch.equal(semantic_task_mask, endpoint_task_mask):
        raise ValueError("endpoint task mask semantics are inconsistent")
    controllable = vector("controllable", torch.bool)
    if not torch.equal(controllable, endpoint_task_mask.any(dim=-1)):
        raise ValueError("endpoint controllability semantics are inconsistent")
    parent_valid_f = parent_valid.to(torch.float32).unsqueeze(-1)
    topology_depth = vector("topology_depth", torch.float32)
    descriptor = torch.cat([
        F.one_hot(role["role"], num_classes=role_classes).to(torch.float32),
        F.one_hot(side["side"], num_classes=side_classes).to(torch.float32),
        matrix("capability", capability_dim),
        F.one_hot(
            role["node_role"], num_classes=role_classes).to(torch.float32),
        F.one_hot(
            side["node_side"], num_classes=side_classes).to(torch.float32),
        matrix("node_capability", capability_dim),
        F.one_hot(
            role["parent_role"].clamp(0, role_classes - 1),
            num_classes=role_classes).to(torch.float32) * parent_valid_f,
        F.one_hot(
            side["parent_side"].clamp(0, side_classes - 1),
            num_classes=side_classes).to(torch.float32) * parent_valid_f,
        matrix("parent_capability", capability_dim) * parent_valid_f,
        matrix("group_role_membership", role_classes),
        matrix("group_side_membership", side_classes),
        matrix("group_capability", capability_dim),
        (topology_depth / float(node_count)).unsqueeze(-1),
        endpoint_task_mask.to(torch.float32),
        controllable.to(torch.float32).unsqueeze(-1),
    ], dim=-1)
    if not bool(torch.isfinite(descriptor).all().item()):
        raise ValueError("endpoint descriptor is non-finite")
    return endpoint_task_mask, descriptor


class NeuroSymbolicRobotStateEncoder(AGICoreModule):
    """Neuro-symbolic private latent; never a shared RobotState input field."""

    def __init__(
        self,
        poseDim: int = ModuleDim.DecisionEndpointPoseDim,
        physicalReferenceDim: int = ModuleDim.RobotPhysicalReferenceDim,
        outDim: int = ModuleDim.PstSlotDim,
        hidden: int = 256,
        *,
        robotMorphology: Any,):
        super().__init__()
        if robotMorphology is None:
            raise TypeError("robot morphology is required")
        self.endpoint_count = int(robotMorphology.endpoint_count)
        self.pose_dim = int(poseDim)
        self.physical_reference_dim = int(physicalReferenceDim)
        _, descriptor = _NeuroEndpointContract(
            robotMorphology)
        self.register_buffer(
            "endpoint_descriptor",
            descriptor.unsqueeze(0),
            persistent=False)
        token_dim = int(outDim)
        self.token_net = nn.Sequential(
            nn.LayerNorm(self.pose_dim + int(descriptor.size(1))),
            nn.Linear(self.pose_dim + int(descriptor.size(1)), hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(2 * token_dim + self.physical_reference_dim),
            nn.Linear(2 * token_dim + self.physical_reference_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),)

    def forward(
        self,
        bodyProprioception: torch.Tensor,
        robotPhysicalReference: torch.Tensor,
        endpointStateValid: Optional[torch.Tensor] = None,) -> torch.Tensor:
        if bodyProprioception.dim() != 3 or tuple(
            bodyProprioception.shape[1:]
        ) != (self.endpoint_count, self.pose_dim):
            raise ValueError("body proprioception does not match endpoint count")
        if robotPhysicalReference.dim() != 2 or int(
            robotPhysicalReference.size(1)
        ) != self.physical_reference_dim:
            raise ValueError("robot physical reference has invalid shape")
        batch_size = int(bodyProprioception.size(0))
        if int(robotPhysicalReference.size(0)) != batch_size:
            raise ValueError("robot physical reference batch does not match")
        expected_mask_shape = (batch_size, self.endpoint_count)
        if self.endpoint_count and endpointStateValid is None:
            raise ValueError("endpoint state validity is required")
        if endpointStateValid is not None and tuple(
            endpointStateValid.shape
        ) != expected_mask_shape:
            raise ValueError("endpoint state validity does not match count")
        runtime_valid = (
            torch.zeros(
                expected_mask_shape,
                device=bodyProprioception.device,
                dtype=torch.bool)
            if endpointStateValid is None
            else endpointStateValid.to(
                device=bodyProprioception.device, dtype=torch.bool))
        safe_pose = torch.where(
            runtime_valid.unsqueeze(-1),
            torch.nan_to_num(bodyProprioception),
            torch.zeros_like(bodyProprioception))
        descriptor = self.endpoint_descriptor.to(
            device=bodyProprioception.device,
            dtype=bodyProprioception.dtype).expand(batch_size, -1, -1)
        valid_f = runtime_valid.to(
            dtype=bodyProprioception.dtype).unsqueeze(-1)
        tokens = self.token_net(torch.cat([
            safe_pose,
            descriptor,], dim=-1)) * valid_f
        count = valid_f.sum(dim=1).clamp_min(1.0)
        mean = tokens.sum(dim=1) / count
        variance = (
            (tokens - mean.unsqueeze(1)).square()
            * valid_f).sum(dim=1) / count
        reference_valid = torch.nan_to_num(
            robotPhysicalReference[:, -1:],
            nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        effective_reference = torch.cat([
            torch.nan_to_num(robotPhysicalReference[:, :-1])
            * reference_valid,
            reference_valid], dim=-1)
        summary = self.summary_net(torch.cat([
            mean,
            variance,
            effective_reference,], dim=-1))
        return summary * runtime_valid.any(
            dim=1, keepdim=True).to(dtype=summary.dtype)


class NeuroSymbolicControlFeedbackEncoder(AGICoreModule):
    """Private diagnostic state for control failure, correction, and replanning."""

    def __init__(
        self,
        outDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        hidden: int = 256,
        *,
        robotMorphology: Any,):
        super().__init__()
        if robotMorphology is None:
            raise TypeError("robot morphology is required")
        self.endpoint_count = int(robotMorphology.endpoint_count)
        self.action_dim = int(ModuleDim.RobotControlAxisDim)
        action_mask, descriptor = _NeuroEndpointContract(
            robotMorphology)
        self.register_buffer(
            "action_mask",
            action_mask.view(
                1,
                self.endpoint_count,
                self.action_dim),
            persistent=False)
        self.register_buffer(
            "endpoint_descriptor",
            descriptor.unsqueeze(0),
            persistent=False)
        token_dim = int(outDim)
        self.token_net = nn.Sequential(
            nn.LayerNorm(2 * self.action_dim + int(descriptor.size(1))),
            nn.Linear(2 * self.action_dim + int(descriptor.size(1)), hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(2 * token_dim),
            nn.Linear(2 * token_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, token_dim),
            nn.LayerNorm(token_dim),)

    def RuntimeMask(
        self,
        batchSize: int,
        device: torch.device,
        endpointStateValid: Optional[torch.Tensor],
        endpointControllable: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (int(batchSize), self.endpoint_count)
        if self.endpoint_count and (
            endpointStateValid is None or endpointControllable is None
        ):
            raise ValueError("endpoint runtime masks are required")
        if endpointStateValid is not None and tuple(
            endpointStateValid.shape
        ) != expected_shape:
            raise ValueError("endpoint state validity does not match count")
        if endpointControllable is not None and tuple(
            endpointControllable.shape
        ) != expected_shape:
            raise ValueError("endpoint controllability does not match count")
        state_valid = (
            torch.zeros(
                expected_shape,
                device=device,
                dtype=torch.bool)
            if endpointStateValid is None
            else endpointStateValid.to(device=device, dtype=torch.bool))
        static_controllable = self.action_mask.any(dim=-1).to(
            device=device).expand(int(batchSize), -1)
        controllable = (
            static_controllable
            if endpointControllable is None
            else endpointControllable.to(device=device, dtype=torch.bool)
                & static_controllable)
        runtime_valid = state_valid & controllable
        action_mask = (
            self.action_mask.to(device=device)
            & runtime_valid.unsqueeze(-1))
        return runtime_valid, action_mask

    def forward(
        self,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,
        endpointStateValid: Optional[torch.Tensor] = None,
        endpointControllable: Optional[torch.Tensor] = None,) -> torch.Tensor:
        expected_shape = (
            int(targetTrackingError.size(0)),
            self.endpoint_count,
            self.action_dim)
        if tuple(targetTrackingError.shape) != expected_shape:
            raise ValueError("target tracking error does not match endpoint count")
        if tuple(plannerTrackingError.shape) != expected_shape:
            raise ValueError("planner tracking error does not match endpoint count")
        runtime_valid, action_mask = self.RuntimeMask(
            targetTrackingError.size(0),
            targetTrackingError.device,
            endpointStateValid,
            endpointControllable)
        safe_target = torch.where(
            action_mask,
            torch.nan_to_num(targetTrackingError),
            torch.zeros_like(targetTrackingError))
        safe_planner = torch.where(
            action_mask,
            torch.nan_to_num(plannerTrackingError),
            torch.zeros_like(plannerTrackingError))
        descriptor = self.endpoint_descriptor.to(
            device=targetTrackingError.device,
            dtype=targetTrackingError.dtype).expand(
                targetTrackingError.size(0), -1, -1)
        valid_f = runtime_valid.to(
            dtype=targetTrackingError.dtype).unsqueeze(-1)
        tokens = self.token_net(torch.cat([
            safe_target,
            safe_planner,
            descriptor,], dim=-1)) * valid_f
        count = valid_f.sum(dim=1).clamp_min(1.0)
        mean = tokens.sum(dim=1) / count
        variance = (
            (tokens - mean.unsqueeze(1)).square()
            * valid_f).sum(dim=1) / count
        summary = self.summary_net(torch.cat([
            mean,
            variance,], dim=-1))
        return summary * runtime_valid.any(
            dim=1, keepdim=True).to(dtype=summary.dtype)


class PredicateGrounder(AGICoreModule):
    def __init__(
        self,
        slotDim: int = ModuleDim.PstSlotDim,
        goalDim: int = ModuleDim.GoalShortDim,
        worldDim: int = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        robotPhysicalStateDim: int = ModuleDim.PstSlotDim,
        poseDim: int = ModuleDim.PstPoseDim,
        hidden: int = 512,):
        super().__init__()
        self.slot_dim = int(slotDim)
        self.robot_physical_state_dim = int(robotPhysicalStateDim)
        self.pose_dim = int(poseDim)
        self.summary_context_dim = (
            int(goalDim)
            + int(worldDim)
            + int(decisionDim)
            + self.robot_physical_state_dim
            + self.pose_dim
            + 6)

        self.feature_dim = (
            self.slot_dim
            + int(goalDim)
            + int(worldDim)
            + int(decisionDim)
            + self.robot_physical_state_dim
            + self.pose_dim
            + 6)

        slot_input_dim = self.slot_dim + self.pose_dim + 6

        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(slot_input_dim),
            nn.Linear(slot_input_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.slot_dim),
            nn.LayerNorm(self.slot_dim),)

        self.summary_query = nn.Sequential(
            nn.LayerNorm(self.summary_context_dim),
            nn.Linear(self.summary_context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.slot_dim),)

        self.reference_memory_scale_head = nn.Sequential(
            nn.LayerNorm(self.summary_context_dim + 8),
            nn.Linear(self.summary_context_dim + 8, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),)

        self.summary_key = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.summary_value = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.summary_refiner = nn.Sequential(
            nn.LayerNorm(3 * self.slot_dim),
            nn.Linear(3 * self.slot_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.slot_dim),
            nn.LayerNorm(self.slot_dim),)

        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(PREDICATES)),)

    def SlotSummary(
        self,
        pst: Dict[str, torch.Tensor],
        summaryContext: torch.Tensor,
        referenced: torch.Tensor,
        referenceSlotIndex: torch.Tensor,
        referenceConfidence: torch.Tensor,
        memoryScale: torch.Tensor,) -> torch.Tensor:
        m = pst["PerceptualPresence"]
        current_step = pst["Step"].view(-1, 1).float()
        observed = pst["Observed"].float()
        memory_age = (
            current_step - pst["LastSeen"].float()
        ).clamp_min(0.0)
        memory_recency = torch.exp(-memory_age / 32.0)
        observed_weight = m * observed
        memory_weight = (
            memoryScale
            * m
            * pst["SlotPresence"]
            * (1.0 - observed)
            * memory_recency)
        slot_context_weight = observed_weight + memory_weight
        target_weight = m * referenced

        slot_input = torch.cat([
            pst["SlotState"],
            pst["PoseCamera"],
            m.unsqueeze(-1),
            observed_weight.unsqueeze(-1),
            memory_weight.unsqueeze(-1),
            memory_recency.unsqueeze(-1),
            referenced.unsqueeze(-1),
            target_weight.unsqueeze(-1),], dim=-1)

        slot_embed = self.slot_encoder(slot_input)

        query = self.summary_query(summaryContext)
        key = self.summary_key(slot_embed)
        value = self.summary_value(slot_embed)

        attn_logits = torch.einsum("bd,bkd->bk", query, key) / (float(self.slot_dim) ** 0.5)
        attn_logits = attn_logits + slot_context_weight.clamp_min(1e-6).log() + (0.05 + referenced).log()
        attn_logits = attn_logits.masked_fill(slot_context_weight <= 0.0, -1e9)

        attn = F.softmax(attn_logits, dim=-1) * slot_context_weight
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        attended_summary = (value * attn.unsqueeze(-1)).sum(dim=1)
        batch_idx = torch.arange(slot_embed.size(0), device=slot_embed.device)
        target_summary = slot_embed[batch_idx, referenceSlotIndex] * referenceConfidence.unsqueeze(-1)
        valid_summary = (slot_embed * slot_context_weight.unsqueeze(-1)).sum(dim=1) / slot_context_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return self.summary_refiner(torch.cat([
            attended_summary,
            target_summary,
            valid_summary,], dim=-1))

    def ReferenceSlotIndex(self, pst: Dict[str, torch.Tensor], referenced: torch.Tensor) -> torch.Tensor:
        resolved = ResolveActuationReferenceWeights(pst, referenced)
        return resolved["actuation_reference"].argmax(dim=1)

    def ReferencedPose(
        self,
        pst: Dict[str, torch.Tensor],
        referenceSlotIndex: torch.Tensor,
        referenceConfidence: torch.Tensor,) -> torch.Tensor:
        pose = pst["PoseCamera"]
        B = pose.size(0)
        batch_idx = torch.arange(B, device=pose.device)
        return pose[batch_idx, referenceSlotIndex] * referenceConfidence.unsqueeze(-1)

    def forward(
        self,
        pst: Dict[str, torch.Tensor],
        observedPst: Dict[str, torch.Tensor],
        goalEmbed: torch.Tensor,
        worldBelief: torch.Tensor,
        decisionBelief: torch.Tensor,
        robotPhysicalState: torch.Tensor,
        referenced: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        referenceUncertainty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,) -> Dict[str, torch.Tensor]:
        resolved_reference = ResolveActuationReferenceWeights(pst, referenced)
        actuation_reference = resolved_reference["actuation_reference"]
        actuation_confidence = actuation_reference.sum(dim=-1)
        reference_slot_idx = actuation_reference.argmax(dim=1)
        semantic_slot_idx = (
            pst["PerceptualPresence"] * referenced).argmax(dim=1)
        ref_pose = self.ReferencedPose(
            pst,
            reference_slot_idx,
            actuation_confidence)

        scalar = torch.stack([
            uncertainty,
            novelty,
            recentFailure,
            referenceUncertainty,
            satisfactionProb.view(-1),
            noSlotProb,], dim=-1)

        summary_context = torch.cat([
            goalEmbed,
            worldBelief,
            decisionBelief,
            robotPhysicalState,
            ref_pose,
            scalar,], dim=-1)
        observed_strength = (
            observedPst["ObservedSlotMask"]
            * observedPst["PerceptualPresence"])
        demand = F.normalize(
            self.summary_query(summary_context), dim=-1, eps=1e-6)
        observed_slot = F.normalize(
            observedPst["SlotState"], dim=-1, eps=1e-6)
        demand_match = torch.einsum(
            "bkd,bd->bk", observed_slot, demand).add(1.0).mul(0.5)
        matched_strength = observed_strength * demand_match
        unmatched_strength = observed_strength * (1.0 - demand_match)
        top_match = torch.topk(matched_strength, k=2, dim=1).values
        observed_total = observed_strength.sum(dim=1, keepdim=True).clamp_min(1e-6)
        observed_max = observed_strength.amax(dim=1, keepdim=True)
        best_match = top_match[:, :1]
        second_match = top_match[:, 1:2]
        reference_context = torch.cat([
            observed_strength.mean(dim=1, keepdim=True),
            observed_max,
            best_match,
            matched_strength.sum(dim=1, keepdim=True) / observed_total,
            1.0 - best_match,
            1.0 - observed_max,
            second_match / best_match.clamp_min(1e-6),
            unmatched_strength.sum(dim=1, keepdim=True) / observed_total,], dim=-1)
        memory_scale = torch.sigmoid(self.reference_memory_scale_head(torch.cat([summary_context, reference_context], dim=-1)))

        slot_summary = self.SlotSummary(
            pst,
            summary_context,
            referenced,
            semantic_slot_idx,
            referenceConfidence,
            memory_scale)

        x = torch.cat([
            slot_summary,
            goalEmbed,
            worldBelief,
            decisionBelief,
            robotPhysicalState,
            ref_pose,
            scalar,], dim=-1)

        logits = self.net(x)

        return {
            "features": x,
            "predicate_logits": logits,
            "predicate_prob": torch.sigmoid(logits),
            "referenced_pose": ref_pose,
            "reference_slot_idx": reference_slot_idx,
            "reference_confidence": referenceConfidence,
            "no_slot_prob": noSlotProb,
            "memory_reference_scale": memory_scale.squeeze(-1),
            "slot_summary": slot_summary,}


class OperatorLibrary(AGICoreModule):
    def __init__(self):
        super().__init__()
        precond = torch.zeros(len(OPERATORS), len(PREDICATES))
        effect = torch.zeros(len(OPERATORS), len(PREDICATES))

        p = {name: i for i, name in enumerate(PREDICATES)}
        o = {name: i for i, name in enumerate(OPERATORS)}

        def requires(op: str, *predicates: str) -> None:
            for pred in predicates:
                precond[o[op], p[pred]] = 1.0

        def causes(op: str, *predicates: str) -> None:
            for pred in predicates:
                effect[o[op], p[pred]] = 1.0

        causes("observe", "observed", "localized")
        requires("observe", "observation_needed")
        causes("observe", "feedback_fresh")
        requires("approach", "localized")
        causes("approach", "reachable")
        requires("align", "localized", "reachable")
        causes("align", "aligned")
        requires("reach", "reachable", "collision_free")
        causes("reach", "contactable")
        requires("contact", "reachable", "contactable", "collision_free")
        causes("contact", "localized")
        requires("grasp", "reachable", "contactable", "graspable")
        causes("grasp", "attached")
        requires("release", "attached")
        causes("release", "supported")
        requires("lift", "attached")
        causes("lift", "movable", "collision_free")
        requires("lower", "attached", "collision_free")
        causes("lower", "supported")
        requires("transport", "attached", "collision_free")
        causes("transport", "localized")
        requires("place", "attached", "collision_free")
        causes("place", "supported", "goal_satisfied")
        requires("push", "reachable", "contactable", "movable", "collision_free")
        causes("push", "localized")
        requires("pull", "reachable", "contactable", "movable", "collision_free")
        causes("pull", "localized", "reachable")
        requires("slide", "reachable", "contactable", "movable", "collision_free")
        causes("slide", "localized")
        requires("rotate", "reachable", "contactable", "movable", "collision_free")
        causes("rotate", "aligned")
        requires("press", "reachable", "contactable", "aligned", "collision_free")
        causes("press", "goal_satisfied")
        requires("open", "reachable", "contactable", "articulated", "closed", "aligned", "collision_free")
        causes("open", "open")
        requires("close", "reachable", "contactable", "articulated", "open", "aligned", "collision_free")
        causes("close", "closed")
        requires("insert", "attached", "aligned", "collision_free")
        causes("insert", "goal_satisfied")
        requires("extract", "reachable", "contactable", "aligned", "collision_free")
        causes("extract", "attached", "movable")
        requires("pour", "attached", "container", "aligned", "collision_free")
        causes("pour", "goal_satisfied")
        requires("wipe", "reachable", "contactable", "aligned", "collision_free")
        causes("wipe", "goal_satisfied")
        requires("cut", "attached", "contactable", "aligned", "collision_free")
        causes("cut", "goal_satisfied")
        requires("operate_tool", "attached", "contactable", "aligned", "collision_free")
        causes("operate_tool", "goal_satisfied")
        requires("handover", "attached", "reachable", "aligned", "collision_free")
        causes("handover", "goal_satisfied")
        requires("hold", "attached")
        causes("hold", "attached")
        requires("retreat", "localized")
        causes("retreat", "collision_free")
        requires("recover", "recovery_needed")
        causes("recover", "observed", "localized", "collision_free")
        requires("reobserve", "observation_needed")
        causes("reobserve", "observed", "localized", "feedback_fresh")
        requires("wait", "observation_needed")
        causes("wait", "feedback_fresh")
        requires("continue_execute", "in_execution", "feedback_fresh", "safe_to_continue", "interruptible")
        causes("continue_execute", "localized")
        requires("cancel_execute", "in_execution", "interruptible")
        causes("cancel_execute", "recovery_needed")
        requires("failsafe_stop", "timeout_risk")
        causes("failsafe_stop", "collision_free", "recovery_needed")
        requires("redispatch", "in_execution", "redispatch_needed", "safe_to_continue")
        causes("redispatch", "localized")

        self.register_buffer("precondition_mask", precond, persistent=False)
        self.register_buffer("effect_mask", effect, persistent=False)

        operator_index = {name: i for i, name in enumerate(OPERATORS)}

        def OperatorMask(*names: str) -> torch.Tensor:
            mask = torch.zeros(len(OPERATORS), dtype=torch.bool)
            for name in names:
                mask[operator_index[name]] = True
            return mask

        always_legal = OperatorMask(
            "observe",
            "reobserve",
            "wait",
            "cancel_execute",
            "failsafe_stop",)
        surface_legal = always_legal | OperatorMask(
            "approach",
            "align",
            "reach",
            "contact",
            "press",
            "retreat",
            "recover",
            "continue_execute",
            "redispatch",)
        self_body_legal = always_legal | OperatorMask(
            "retreat",
            "recover",
            "continue_execute",
            "redispatch",)
        self.register_buffer(
            "always_legal_operator_mask", always_legal, persistent=False)
        self.register_buffer(
            "surface_legal_operator_mask", surface_legal, persistent=False)
        self.register_buffer(
            "self_body_legal_operator_mask", self_body_legal, persistent=False)
        self.register_buffer(
            "direct_physical_operator_mask",
            torch.ones(len(OPERATORS), dtype=torch.bool),
            persistent=False)

    def RealmAwareLegality(
        self,
        pst: Dict[str, torch.Tensor],
        referenced: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, K = referenced.shape
        reference_mass = referenced.sum(dim=-1)
        semantic_weight = referenced / (reference_mass.unsqueeze(-1) + 1e-8)
        target_realm_prob = torch.einsum(
            "bk,bkr->br",
            semantic_weight,
            pst["RealmProb"])
        target_realm = target_realm_prob.argmax(dim=-1)
        reference_present = reference_mass > 0.5

        resolved = ResolveActuationReferenceWeights(pst, referenced)
        direct_confidence = resolved["direct_reference"].sum(dim=-1)
        direct_physical = (
            reference_present
            & (target_realm == REALM_EXTERNAL_PHYSICAL)
            & (direct_confidence > 0.5))

        virtual_slot_score = (
            semantic_weight
            * pst["RealmProb"][..., REALM_VIRTUAL_CONTENT])
        virtual_slot_index = virtual_slot_score.argmax(dim=-1)
        batch_index = torch.arange(B, device=referenced.device)
        child_parent_prob = pst["SurfaceParentProb"][
            batch_index, virtual_slot_index]
        physical_parent_prob = child_parent_prob[..., :K]
        parent_probability, parent_index = physical_parent_prob.max(dim=-1)
        no_parent_probability = child_parent_prob[..., K]

        child_verification = pst["VerificationConfidence"][
            batch_index, virtual_slot_index]
        child_presence = pst["PerceptualPresence"][
            batch_index, virtual_slot_index]
        uv_valid = (
            pst["SurfaceUVConfidence"][batch_index, virtual_slot_index]
            > 0.5)

        parent_realm = pst["RealmProb"][
            batch_index, parent_index].argmax(dim=-1)
        parent_is_physical = (
            (parent_realm == REALM_SELF_BODY)
            | (parent_realm == REALM_EXTERNAL_PHYSICAL))
        parent_valid = (
            (pst["PerceptualPresence"][batch_index, parent_index] > 0.5)
            & (pst["PhysicalInteractionProb"][batch_index, parent_index] > 0.5)
            & (pst["VerificationConfidence"][batch_index, parent_index] > 0.5)
            & (pst["DisplaySurfaceProb"][batch_index, parent_index] > 0.5))
        verified_surface = (
            reference_present
            & (target_realm == REALM_VIRTUAL_CONTENT)
            & (child_presence > 0.5)
            & (child_verification > 0.5)
            & uv_valid
            & (parent_probability > 0.5)
            & (parent_probability > no_parent_probability)
            & parent_is_physical
            & parent_valid)

        self_body_target = (
            reference_present
            & (target_realm == REALM_SELF_BODY))
        legality = self.always_legal_operator_mask.view(1, -1).expand(B, -1).clone()
        legality = legality | (
            direct_physical.unsqueeze(-1)
            & self.direct_physical_operator_mask.view(1, -1))
        legality = legality | (
            verified_surface.unsqueeze(-1)
            & self.surface_legal_operator_mask.view(1, -1))
        legality = legality | (
            self_body_target.unsqueeze(-1)
            & self.self_body_legal_operator_mask.view(1, -1))
        return {
            "operator_legality": legality,
            "target_realm_prob": target_realm_prob,
            "target_realm": target_realm,
            "direct_physical": direct_physical,
            "verified_surface": verified_surface,
            "reference_present": reference_present,
            "surface_parent_index": parent_index,}

    def ApplyRealmAwareLegality(
        self,
        operatorLogits: torch.Tensor,
        pst: Dict[str, torch.Tensor],
        referenced: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        legality = self.RealmAwareLegality(pst, referenced)
        row_min = operatorLogits.amin(dim=-1, keepdim=True)
        row_span = (
            operatorLogits.amax(dim=-1, keepdim=True) - row_min)
        illegal_floor = (row_min - row_span - 1.0).detach()
        return (
            torch.where(
                legality["operator_legality"],
                operatorLogits,
                illegal_floor),
            legality,)

    def Scores(
        self,
        predicateProb: torch.Tensor,
        goalPredicateNeed: torch.Tensor,) -> Dict[str, torch.Tensor]:
        precondition_mask = self.precondition_mask.unsqueeze(0)
        effect_mask = self.effect_mask.unsqueeze(0)
        predicate = predicateProb.unsqueeze(1).expand(-1, len(OPERATORS), -1)
        precond_count = self.precondition_mask.sum(dim=-1).clamp_min(1.0)
        precond_mean = (predicate * precondition_mask).sum(dim=-1) / precond_count.view(1, -1)
        precond_floor = predicate.masked_fill(precondition_mask == 0.0, 1.0).amin(dim=-1)
        precond_score = 0.70 * precond_floor + 0.30 * precond_mean

        goal_gap = goalPredicateNeed * (1.0 - predicateProb)
        effect_count = self.effect_mask.sum(dim=-1).clamp_min(1.0)
        effect_score = (goal_gap.unsqueeze(1) * effect_mask).sum(dim=-1) / effect_count.view(1, -1)

        redundant_effect = predicateProb.unsqueeze(1) * (1.0 - goalPredicateNeed).unsqueeze(1) * effect_mask
        redundancy_penalty = redundant_effect.sum(dim=-1) / effect_count.view(1, -1)

        symbolic_score = 1.50 * precond_score + 2.00 * effect_score - 0.50 * redundancy_penalty

        return {
            "precond_score": precond_score,
            "effect_score": effect_score,
            "redundancy_penalty": redundancy_penalty,
            "goal_gap": goal_gap,
            "symbolic_score": symbolic_score,}


class PlanRanker(AGICoreModule):
    def __init__(
        self,
        inputDim: int,
        temporalContextDim: int = ModuleDim.TemporalContextDim,
        hidden: int = 768,
        planDim: int = 256,):
        super().__init__()
        self.hidden = int(hidden)
        self.plan_dim = int(planDim)

        self.input_norm = nn.LayerNorm(inputDim + temporalContextDim)
        self.input_proj = nn.Linear(inputDim + temporalContextDim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 2 * hidden),
                nn.SiLU(),
                nn.Linear(2 * hidden, hidden),)
            for _ in range(3)])

        self.state_norm = nn.LayerNorm(hidden)

        self.operator_embedding = nn.Embedding(len(OPERATORS), hidden)

        self.operator_query = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),)

        self.operator_direct_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, len(OPERATORS)),)

        self.operator_bias = nn.Parameter(torch.zeros(len(OPERATORS)))

        self.plan_seed_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, planDim),
            nn.LayerNorm(planDim),)

        self.plan_refiner = nn.Sequential(
            nn.LayerNorm(planDim + hidden),
            nn.Linear(planDim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, planDim),
            nn.LayerNorm(planDim),)

    def RefinePlan(self, planSeed: torch.Tensor, operatorLogits: torch.Tensor) -> torch.Tensor:
        operator_prob = F.softmax(operatorLogits, dim=-1)
        operator_context = operator_prob @ self.operator_embedding.weight
        return self.plan_refiner(torch.cat([planSeed, operator_context], dim=-1))

    def forward(self, x: torch.Tensor, temporalContextFeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.input_proj(self.input_norm(torch.cat([x, temporalContextFeat], dim=-1)))
        for block in self.blocks:
            h = h + block(h)
        h = self.state_norm(h)

        operator_query = self.operator_query(h)
        match_logits = operator_query @ self.operator_embedding.weight.t() / (float(self.hidden) ** 0.5)
        operator_logits = self.operator_direct_head(h) + match_logits + self.operator_bias.view(1, -1)
        plan_seed = self.plan_seed_head(h)

        return {
            "plan_seed": plan_seed,
            "operator_logits": operator_logits,
            "ranker_state": h,}


class FailureExplainer(AGICoreModule):
    def __init__(
        self,
        inputDim: int,
        temporalContextDim: int = ModuleDim.TemporalContextDim,
        hidden: int = 512,):
        super().__init__()
        self.hidden = int(hidden)
        self.input_norm = nn.LayerNorm(inputDim + temporalContextDim + len(PREDICATES) + len(OPERATORS) + len(PREDICATES))
        self.input_proj = nn.Linear(inputDim + temporalContextDim + len(PREDICATES) + len(OPERATORS) + len(PREDICATES), hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 2 * hidden),
                nn.SiLU(),
                nn.Linear(2 * hidden, hidden),)
            for _ in range(2)])

        self.state_norm = nn.LayerNorm(hidden)

        self.failure_embedding = nn.Embedding(len(FAILURE_CAUSES), hidden)

        self.failure_query = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),)

        self.direct_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, len(FAILURE_CAUSES)),)

        self.failure_bias = nn.Parameter(torch.zeros(len(FAILURE_CAUSES)))

        p = {name: i for i, name in enumerate(PREDICATES)}
        f = {name: i for i, name in enumerate(FAILURE_CAUSES)}
        evidence = torch.zeros(len(FAILURE_CAUSES), len(PREDICATES))
        inhibition = torch.zeros(len(FAILURE_CAUSES), len(PREDICATES))

        def evidence_of(cause: str, *predicates: str) -> None:
            for pred in predicates:
                evidence[f[cause], p[pred]] = 1.0

        def inhibited_by(cause: str, *predicates: str) -> None:
            for pred in predicates:
                inhibition[f[cause], p[pred]] = 1.0

        evidence_of("occlusion", "observation_needed", "feedback_stale")
        inhibited_by("occlusion", "observed", "localized", "feedback_fresh")
        evidence_of("unreachable", "localized")
        inhibited_by("unreachable", "reachable")
        evidence_of("slip", "in_execution", "feedback_stale")
        inhibited_by("slip", "attached", "feedback_fresh")
        evidence_of("collision", "reachable", "contactable")
        inhibited_by("collision", "collision_free")
        evidence_of("misalignment", "reachable", "contactable")
        inhibited_by("misalignment", "aligned")
        evidence_of("low_confidence", "observation_needed", "feedback_stale")
        inhibited_by("low_confidence", "observed", "localized", "feedback_fresh")
        evidence_of("unstable_support", "supported", "movable")
        inhibited_by("unstable_support", "attached", "collision_free")
        evidence_of("articulation_blocked", "articulated", "timeout_risk")
        inhibited_by("articulation_blocked", "open", "closed", "safe_to_continue")
        evidence_of("containment_error", "container", "aligned")
        inhibited_by("containment_error", "goal_satisfied")
        evidence_of("tool_error", "attached", "contactable", "timeout_risk")
        inhibited_by("tool_error", "goal_satisfied", "safe_to_continue")
        evidence_of("lost_attachment", "in_execution", "feedback_stale")
        inhibited_by("lost_attachment", "attached", "feedback_fresh")
        evidence_of("handover_failed", "attached", "reachable", "timeout_risk")
        inhibited_by("handover_failed", "goal_satisfied", "safe_to_continue")

        self.register_buffer("failure_evidence_mask", evidence, persistent=False)
        self.register_buffer("failure_inhibition_mask", inhibition, persistent=False)

    def PredicateEvidence(self, predicateProb: torch.Tensor) -> torch.Tensor:
        evidence_count = self.failure_evidence_mask.sum(dim=-1).clamp_min(1.0)
        inhibition_count = self.failure_inhibition_mask.sum(dim=-1).clamp_min(1.0)
        evidence_score = (predicateProb.unsqueeze(1) * self.failure_evidence_mask.unsqueeze(0)).sum(dim=-1) / evidence_count.view(1, -1)
        inhibition_score = (predicateProb.unsqueeze(1) * self.failure_inhibition_mask.unsqueeze(0)).sum(dim=-1) / inhibition_count.view(1, -1)
        return evidence_score - inhibition_score

    def forward(
        self,
        x: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        predicateProb: torch.Tensor,
        operatorLogits: torch.Tensor,
        goalGap: torch.Tensor,) -> torch.Tensor:
        operator_prob = F.softmax(operatorLogits, dim=-1)
        h = self.input_proj(self.input_norm(torch.cat([
            x,
            temporalContextFeat,
            predicateProb,
            operator_prob,
            goalGap,], dim=-1)))

        for block in self.blocks:
            h = h + block(h)
        h = self.state_norm(h)

        failure_query = self.failure_query(h)
        match_logits = failure_query @ self.failure_embedding.weight.t() / (float(self.hidden) ** 0.5)
        evidence_logits = self.PredicateEvidence(predicateProb)

        return self.direct_head(h) + match_logits + evidence_logits + self.failure_bias.view(1, -1)


class TemporalSymbolicHead(AGICoreModule):
    def __init__(self, inputDim: int, hidden: int = 512):
        super().__init__()
        self.hidden = int(hidden)
        self.observe_idx = ModuleDim.TemporalPrimitiveNames.index("OBSERVE")
        self.dispatch_idx = ModuleDim.TemporalPrimitiveNames.index("DISPATCH")
        self.continue_idx = ModuleDim.TemporalPrimitiveNames.index("CONTINUE")
        self.cancel_idx = ModuleDim.TemporalPrimitiveNames.index("CANCEL")
        self.failsafe_idx = ModuleDim.TemporalPrimitiveNames.index("FAILSAFE_STOP")
        self.redispatch_idx = ModuleDim.TemporalPrimitiveNames.index("REDISPATCH")

        condition_dim = len(PREDICATES) + len(OPERATORS) + len(FAILURE_CAUSES) + 1 + len(PREDICATES)
        self.input_norm = nn.LayerNorm(inputDim + condition_dim)
        self.input_proj = nn.Linear(inputDim + condition_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 2 * hidden),
                nn.SiLU(),
                nn.Linear(2 * hidden, hidden),)
            for _ in range(2)])

        self.state_norm = nn.LayerNorm(hidden)

        self.primitive_embedding = nn.Embedding(ModuleDim.TemporalPrimitiveCount, hidden)

        self.primitive_query = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),)

        self.temporal_direct_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, ModuleDim.TemporalPrimitiveCount),)

        self.temporal_bias = nn.Parameter(torch.zeros(ModuleDim.TemporalPrimitiveCount))
        nn.init.zeros_(self.primitive_query[-2].weight)
        nn.init.zeros_(self.primitive_query[-2].bias)
        nn.init.zeros_(self.temporal_direct_head[-1].weight)
        nn.init.zeros_(self.temporal_direct_head[-1].bias)
        self.reason_predicate_indices = tuple(PREDICATES.index(name) for name in (
            "observation_needed",
            "feedback_stale",
            "goal_satisfied",
            "timeout_risk",
            "recovery_needed",
            "redispatch_needed",
            "safe_to_continue",
            "collision_free",
        ))

        p = {name: i for i, name in enumerate(PREDICATES)}
        primitive = {name: i for i, name in enumerate(ModuleDim.TemporalPrimitiveNames)}
        evidence = torch.zeros(ModuleDim.TemporalPrimitiveCount, len(PREDICATES))
        inhibition = torch.zeros(ModuleDim.TemporalPrimitiveCount, len(PREDICATES))

        def evidence_of(kind: str, *predicates: str) -> None:
            for pred in predicates:
                evidence[primitive[kind], p[pred]] = 1.0

        def inhibited_by(kind: str, *predicates: str) -> None:
            for pred in predicates:
                inhibition[primitive[kind], p[pred]] = 1.0

        evidence_of("OBSERVE", "observation_needed", "feedback_stale")
        inhibited_by("OBSERVE", "observed", "localized", "feedback_fresh")
        evidence_of("DISPATCH", "localized", "reachable", "collision_free")
        inhibited_by("DISPATCH", "in_execution", "goal_satisfied", "observation_needed")
        evidence_of("CONTINUE", "in_execution", "feedback_fresh", "safe_to_continue")
        inhibited_by("CONTINUE", "feedback_stale", "timeout_risk", "recovery_needed", "redispatch_needed")
        evidence_of("CANCEL", "in_execution", "interruptible", "feedback_stale", "timeout_risk", "recovery_needed")
        inhibited_by("CANCEL", "safe_to_continue", "feedback_fresh")
        evidence_of("FAILSAFE_STOP", "timeout_risk", "recovery_needed")
        inhibited_by("FAILSAFE_STOP", "safe_to_continue", "collision_free")
        evidence_of("REDISPATCH", "in_execution", "redispatch_needed", "safe_to_continue")
        inhibited_by("REDISPATCH", "timeout_risk", "feedback_stale")

        self.register_buffer("primitive_evidence_mask", evidence, persistent=False)
        self.register_buffer("primitive_inhibition_mask", inhibition, persistent=False)

    def PrimitiveRulePrior(self, predicateProb: torch.Tensor) -> torch.Tensor:
        evidence_count = self.primitive_evidence_mask.sum(dim=-1).clamp_min(1.0)
        inhibition_count = self.primitive_inhibition_mask.sum(dim=-1).clamp_min(1.0)
        evidence_score = (predicateProb.unsqueeze(1) * self.primitive_evidence_mask.unsqueeze(0)).sum(dim=-1) / evidence_count.view(1, -1)
        inhibition_score = (predicateProb.unsqueeze(1) * self.primitive_inhibition_mask.unsqueeze(0)).sum(dim=-1) / inhibition_count.view(1, -1)
        return evidence_score - inhibition_score

    def forward(
        self,
        x: torch.Tensor,
        predicateProb: torch.Tensor,
        operatorLogits: torch.Tensor,
        failureCauseLogits: torch.Tensor,
        failureGate: torch.Tensor,
        goalGap: torch.Tensor,) -> Dict[str, torch.Tensor]:
        operator_prob = F.softmax(operatorLogits, dim=-1)
        failure_prob = torch.sigmoid(failureCauseLogits)
        h = self.input_proj(self.input_norm(torch.cat([
            x,
            predicateProb,
            operator_prob,
            failure_prob,
            failureGate.unsqueeze(-1),
            goalGap,], dim=-1)))

        for block in self.blocks:
            h = h + block(h)
        h = self.state_norm(h)

        primitive_query = self.primitive_query(h)
        match_logits = primitive_query @ self.primitive_embedding.weight.t() / (float(self.hidden) ** 0.5)
        rule_logits = self.PrimitiveRulePrior(predicateProb)
        temporal_logits = self.temporal_direct_head(h) + match_logits + rule_logits + self.temporal_bias.view(1, -1)

        continue_support = (
            temporal_logits[:, self.continue_idx]
            - 0.5 * temporal_logits[:, self.cancel_idx]
            - 0.5 * temporal_logits[:, self.failsafe_idx]
            - 0.25 * temporal_logits[:, self.redispatch_idx])

        interrupt_support = (
            0.5 * temporal_logits[:, self.cancel_idx]
            + 0.5 * temporal_logits[:, self.failsafe_idx]
            - 0.5 * temporal_logits[:, self.continue_idx])

        redispatch_support = (
            temporal_logits[:, self.redispatch_idx]
            - 0.5 * temporal_logits[:, self.cancel_idx]
            - 0.5 * temporal_logits[:, self.failsafe_idx])

        return {
            "temporal_logits": temporal_logits,
            "temporal_reason_logits": predicateProb[:, self.reason_predicate_indices],
            "continue_guard_score": torch.sigmoid(continue_support),
            "interrupt_guard_score": torch.sigmoid(interrupt_support),
            "redispatch_guard_score": torch.sigmoid(redispatch_support),}


class SymbolicFeatureMixer(AGICoreModule):
    def __init__(
        self,
        planDim: int = 256,
        subgoalFeatureDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        constraintTokens: int = 8,
        constraintTokenDim: int = 128,
        predicateDim: int = len(PREDICATES),
        operatorDim: int = len(OPERATORS),
        failureDim: int = len(FAILURE_CAUSES),
        temporalPrimitiveDim: int = ModuleDim.TemporalPrimitiveCount,
        temporalReasonDim: int = ModuleDim.TemporalReasonDim,
        poseDim: int = ModuleDim.PstPoseDim,
        hidden: int = 768,):
        super().__init__()
        self.plan_dim = int(planDim)
        self.subgoal_feature_dim = int(subgoalFeatureDim)
        self.constraint_tokens = int(constraintTokens)
        self.constraint_token_dim = int(constraintTokenDim)
        self.predicate_dim = int(predicateDim)
        self.operator_dim = int(operatorDim)
        self.failure_dim = int(failureDim)
        self.temporal_primitive_dim = int(temporalPrimitiveDim)
        self.temporal_reason_dim = int(temporalReasonDim)
        self.pose_dim = int(poseDim)

        context_dim = (
            self.plan_dim
            + self.predicate_dim
            + self.operator_dim
            + self.failure_dim
            + self.temporal_primitive_dim
            + self.temporal_reason_dim
            + 3
            + self.predicate_dim
            + 1
            + self.pose_dim
            + self.constraint_token_dim)

        self.constraint_summary_query = nn.Parameter(torch.zeros(self.constraint_token_dim))

        self.context_input_norm = nn.LayerNorm(context_dim)

        self.context_input_proj = nn.Linear(context_dim, hidden)

        self.context_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 2 * hidden),
                nn.SiLU(),
                nn.Linear(2 * hidden, hidden),)
            for _ in range(2)])

        self.context_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.plan_dim),
            nn.LayerNorm(self.plan_dim),)

        self.context_gate = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.plan_dim),
            nn.Sigmoid(),)

        self.plan_refiner = nn.Sequential(
            nn.LayerNorm(2 * self.plan_dim),
            nn.Linear(2 * self.plan_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.plan_dim),
            nn.LayerNorm(self.plan_dim),)

        self.plan_out_norm = nn.LayerNorm(self.plan_dim)

        self.subgoal_refiner = nn.Sequential(
            nn.LayerNorm(self.plan_dim + self.subgoal_feature_dim),
            nn.Linear(self.plan_dim + self.subgoal_feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.subgoal_feature_dim),
            nn.LayerNorm(self.subgoal_feature_dim),)

        self.constraint_context = nn.Linear(2 * self.plan_dim, self.constraint_token_dim)

        self.constraint_refiner = nn.Sequential(
            nn.LayerNorm(2 * self.constraint_token_dim),
            nn.Linear(2 * self.constraint_token_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.constraint_token_dim),
            nn.LayerNorm(self.constraint_token_dim),)

    def forward(
        self,
        planLatent: torch.Tensor,
        predicateProb: torch.Tensor,
        operatorLogits: torch.Tensor,
        failureCauseLogits: torch.Tensor,
        failureGate: torch.Tensor,
        temporalLogits: torch.Tensor,
        temporalReasonLogits: torch.Tensor,
        continueGuardScore: torch.Tensor,
        interruptGuardScore: torch.Tensor,
        redispatchGuardScore: torch.Tensor,
        goalGap: torch.Tensor,
        subgoalFeature: torch.Tensor,
        constraintTokens: torch.Tensor,
        referencedPose: torch.Tensor,) -> Dict[str, torch.Tensor]:
        B = planLatent.size(0)
        operator_prob = F.softmax(operatorLogits, dim=-1)
        failure_prob = torch.sigmoid(failureCauseLogits)
        temporal_prob = F.softmax(temporalLogits, dim=-1)
        temporal_reason_prob = torch.sigmoid(temporalReasonLogits)

        temporal_guard = torch.stack([
            continueGuardScore,
            interruptGuardScore,
            redispatchGuardScore,], dim=-1)

        constraint_summary_logits = torch.einsum("btd,d->bt", constraintTokens, self.constraint_summary_query)
        constraint_summary_weight = F.softmax(constraint_summary_logits, dim=-1)
        constraint_summary = (constraintTokens * constraint_summary_weight.unsqueeze(-1)).sum(dim=1)

        ref_pose = referencedPose

        context_input = torch.cat([
            planLatent,
            predicateProb,
            operator_prob,
            failure_prob,
            temporal_prob,
            temporal_reason_prob,
            temporal_guard,
            goalGap,
            failureGate.unsqueeze(-1),
            ref_pose,
            constraint_summary,], dim=-1)

        h = self.context_input_proj(self.context_input_norm(context_input))

        for block in self.context_blocks:
            h = h + block(h)

        context = self.context_head(h)

        context_gate = self.context_gate(context_input)

        plan_delta = self.plan_refiner(torch.cat([planLatent, context], dim=-1))
        plan_latent = self.plan_out_norm(planLatent + context_gate * plan_delta)

        subgoal_feature = self.subgoal_refiner(torch.cat([plan_latent, subgoalFeature], dim=-1))

        token_context = self.constraint_context(torch.cat([plan_latent, context], dim=-1)).unsqueeze(1).expand(
            B,
            self.constraint_tokens,
            self.constraint_token_dim,)

        constraint_tokens = self.constraint_refiner(torch.cat([
            constraintTokens,
            token_context,], dim=-1))

        return {
            "plan_latent": plan_latent,
            "subgoal_feature": subgoal_feature,
            "constraint_tokens": constraint_tokens,}


class NeuroSymbolicExtractor(AGICoreModule):
    def __init__(
        self,
        *,
        robotMorphology: Any,
        slotDim: int = ModuleDim.PstSlotDim,
        goalDim: int = ModuleDim.GoalShortDim,
        worldDim: int = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        poseDim: int = ModuleDim.PstPoseDim,
        robotPhysicalStateDim: int = ModuleDim.PstSlotDim,
        planDim: int = 256,
        constraintTokenDim: int = 128,
        constraintTokens: int = 8,):
        super().__init__()
        self.pose_dim = int(poseDim)
        self.robot_physical_state_dim = int(robotPhysicalStateDim)
        self.plan_dim = int(planDim)
        self.constraint_token_dim = int(constraintTokenDim)
        self.constraint_tokens = int(constraintTokens)
        self.control_feedback_dim = int(
            ModuleDim.DecisionEndpointPoseFeatDim)
        self.robot_state_encoder = NeuroSymbolicRobotStateEncoder(
            outDim=self.robot_physical_state_dim,
            robotMorphology=robotMorphology)
        self.control_feedback_encoder = NeuroSymbolicControlFeedbackEncoder(
            outDim=self.control_feedback_dim,
            robotMorphology=robotMorphology)

        self.predicate_grounder = PredicateGrounder(
            slotDim=slotDim,
            goalDim=goalDim,
            worldDim=worldDim,
            decisionDim=decisionDim,
            robotPhysicalStateDim=self.robot_physical_state_dim,
            poseDim=self.pose_dim,)

        base_feature_dim = (
            self.predicate_grounder.feature_dim
            + len(PREDICATES)
            + self.control_feedback_dim)

        self.operator_library = OperatorLibrary()

        self.plan_ranker = PlanRanker(base_feature_dim, planDim=self.plan_dim)

        self.goal_predicate_head = nn.Sequential(
            nn.LayerNorm(base_feature_dim),
            nn.Linear(base_feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, len(PREDICATES)),)

        self.failure_explainer = FailureExplainer(base_feature_dim)

        self.failure_gate_head = nn.Sequential(
            nn.LayerNorm(base_feature_dim + ModuleDim.TemporalContextDim + len(FAILURE_CAUSES)),
            nn.Linear(base_feature_dim + ModuleDim.TemporalContextDim + len(FAILURE_CAUSES), 256),
            nn.SiLU(),
            nn.Linear(256, 1),)

        self.temporal_predicate_head = nn.Sequential(
            nn.LayerNorm(ModuleDim.TemporalContextDim),
            nn.Linear(ModuleDim.TemporalContextDim, 128),
            nn.SiLU(),
            nn.Linear(128, len(TEMPORAL_PREDICATES)),)

        self.temporal_symbolic_head = TemporalSymbolicHead(base_feature_dim + ModuleDim.TemporalContextDim)

        self.invoke_head = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, 1),)

        sampler_in = (
            self.plan_dim
            + self.pose_dim
            + self.robot_physical_state_dim
            + self.control_feedback_dim)

        self.subgoal_feature_head = nn.Sequential(
            nn.LayerNorm(sampler_in),
            nn.Linear(sampler_in, 256),
            nn.SiLU(),
            nn.Linear(256, self.robot_physical_state_dim),
            nn.LayerNorm(self.robot_physical_state_dim),)

        self.constraint_head = nn.Sequential(
            nn.LayerNorm(
                self.plan_dim
                + len(PREDICATES)
                + len(OPERATORS)
                + self.control_feedback_dim),
            nn.Linear(
                self.plan_dim
                + len(PREDICATES)
                + len(OPERATORS)
                + self.control_feedback_dim,
                512),
            nn.SiLU(),
            nn.Linear(512, self.constraint_tokens * self.constraint_token_dim),)

        self.symbolic_mixer = SymbolicFeatureMixer(
            planDim=self.plan_dim,
            subgoalFeatureDim=self.robot_physical_state_dim,
            constraintTokens=self.constraint_tokens,
            constraintTokenDim=self.constraint_token_dim,
            poseDim=self.pose_dim,)

        self.register_buffer("last_operator", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("last_operator_prob", torch.empty(0), persistent=False)
        self.register_buffer("last_invoke", torch.empty(0), persistent=False)
        self.register_buffer("last_reference_summary", torch.empty(0), persistent=False)

    @torch.no_grad()
    def ResetPlan(self, doneMask: torch.Tensor):
        done = doneMask.view(-1)
        if self.last_operator.numel() != done.numel():
            return
        self.last_operator[done] = -1
        self.last_operator_prob[done] = 0.0
        self.last_invoke[done] = 0.0
        self.last_reference_summary[done] = 0.0

    @torch.no_grad()
    def ExportPlanState(self) -> Dict[str, torch.Tensor]:
        return {
            "last_operator": self.last_operator.detach().clone(),
            "last_operator_prob": self.last_operator_prob.detach().clone(),
            "last_invoke": self.last_invoke.detach().clone(),
            "last_reference_summary": self.last_reference_summary.detach().clone(),}

    @staticmethod
    def InvocationNeedTarget(
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        referenceUncertainty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        noSlotProb: torch.Tensor,) -> torch.Tensor:
        """Semantic teacher for the learned symbolic-invocation intensity.

        Invocation is warranted by epistemic/risk/failure/binding evidence.  A
        satisfied goal attenuates that need but cannot erase a hard failure or
        missing reference.  The target is detached by the loss owner.
        """
        need = torch.stack([
            uncertainty,
            novelty,
            recentFailure,
            referenceUncertainty,
            noSlotProb,
        ], dim=-1).amax(dim=-1)
        return need * (1.0 - 0.5 * satisfactionProb.view(-1))

    @torch.no_grad()
    def ImportPlanState(self, state: Dict[str, torch.Tensor]):
        self.last_operator = state["last_operator"].clone()
        self.last_operator_prob = state["last_operator_prob"].clone()
        self.last_invoke = state["last_invoke"].clone()
        self.last_reference_summary = state["last_reference_summary"].clone()

    def BuildFacts(
        self,
        predicateProb: torch.Tensor,
        referenced: torch.Tensor,) -> List[SymbolicFact]:
        return [
            SymbolicFact(
                name=name,
                args=("task_object",),
                prob=predicateProb[:, i],
                support=referenced,)
            for i, name in enumerate(PREDICATES)]

    def NamesAbove(
        self,
        values: torch.Tensor,
        names: Tuple[str, ...],
        *,
        threshold: float,
        maxItems: int,) -> List[str]:
        if values.numel() == 0:
            return []
        values_detached = values.detach()
        idx = (values_detached >= float(threshold)).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return []
        scores = values_detached[idx]
        order = torch.argsort(scores, descending=True)
        idx = idx[order[:max(0, int(maxItems))]]
        return [names[int(i.item())] for i in idx]

    def TopNames(
        self,
        values: torch.Tensor,
        names: Tuple[str, ...],
        *,
        maxItems: int,) -> List[str]:
        if values.numel() == 0 or maxItems <= 0:
            return []
        kk = min(int(maxItems), int(values.numel()))
        _, idx = torch.topk(values.detach(), kk, dim=-1)
        return [names[int(i.item())] for i in idx]

    def BuildPlanSteps(
        self,
        operatorLogits: torch.Tensor,
        precondScore: torch.Tensor,
        effectScore: torch.Tensor,
        planLatent: torch.Tensor,) -> List[OperatorStep]:
        op_ids = operatorLogits.argmax(dim=-1)
        steps: List[OperatorStep] = []
        for b in range(operatorLogits.size(0)):
            op_id = int(op_ids[b].item())
            op_name = OPERATORS[op_id]
            steps.append(OperatorStep(
                op_name=op_name,
                args=("task_object",),
                precond_score=precondScore[b, op_id],
                effect_score=effectScore[b, op_id],
                sampler_latent=planLatent[b],
                explanation=[op_name],))

        return steps

    def BuildOperatorRationales(
        self,
        operatorLogits: torch.Tensor,
        predicateProb: torch.Tensor,
        goalGap: torch.Tensor,
        riskCauseLogits: torch.Tensor,
        temporalLogits: torch.Tensor,
        precondScore: torch.Tensor,
        effectScore: torch.Tensor,
        symbolicScore: torch.Tensor,) -> List[OperatorRationale]:
        op_ids = operatorLogits.argmax(dim=-1)
        risk_prob = torch.sigmoid(riskCauseLogits)
        temporal_prob = F.softmax(temporalLogits, dim=-1)
        precond_mask = self.operator_library.precondition_mask.bool()
        effect_mask = self.operator_library.effect_mask.bool()
        rationales: List[OperatorRationale] = []

        for b in range(operatorLogits.size(0)):
            op_id = int(op_ids[b].item())
            op_name = OPERATORS[op_id]
            pred_prob_b = predicateProb[b]
            precond_idx = precond_mask[op_id].nonzero(as_tuple=False).flatten()
            effect_idx = effect_mask[op_id].nonzero(as_tuple=False).flatten()

            satisfied_preconditions = [
                PREDICATES[int(i.item())]
                for i in precond_idx
                if float(pred_prob_b[i].detach().item()) >= 0.5]

            weak_preconditions = [
                PREDICATES[int(i.item())]
                for i in precond_idx
                if float(pred_prob_b[i].detach().item()) < 0.5]
            expected_effects = [PREDICATES[int(i.item())] for i in effect_idx]

            missing_goal_predicates = self.NamesAbove(
                goalGap[b],
                PREDICATES,
                threshold=0.20,
                maxItems=4)

            risk_causes = self.NamesAbove(
                risk_prob[b],
                FAILURE_CAUSES,
                threshold=0.50,
                maxItems=3)

            temporal_reasons = self.TopNames(
                temporal_prob[b],
                ModuleDim.TemporalPrimitiveNames,
                maxItems=2)

            rationales.append(OperatorRationale(
                op_name=op_name,
                args=("task_object",),
                selected_logit=operatorLogits[b, op_id],
                precond_score=precondScore[b, op_id],
                effect_score=effectScore[b, op_id],
                symbolic_score=symbolicScore[b, op_id],
                satisfied_preconditions=satisfied_preconditions,
                weak_preconditions=weak_preconditions,
                missing_goal_predicates=missing_goal_predicates,
                expected_effects=expected_effects,
                risk_causes=risk_causes,
                temporal_reasons=temporal_reasons,))

        return rationales

    def BuildSubgoalFeature(
        self,
        planLatent: torch.Tensor,
        referencedPose: torch.Tensor,
        robotPhysicalState: torch.Tensor,
        controlFeedbackState: torch.Tensor,) -> torch.Tensor:
        ref_pose = referencedPose
        return self.subgoal_feature_head(torch.cat([
            planLatent,
            ref_pose,
            robotPhysicalState,
            controlFeedbackState,], dim=-1))

    def forward(
        self,
        pst: Dict[str, torch.Tensor],
        observedPst: Dict[str, torch.Tensor],
        goalEmbed: torch.Tensor,
        worldBelief: torch.Tensor,
        decisionBelief: torch.Tensor,
        bodyProprioception: torch.Tensor,
        robotPhysicalReference: torch.Tensor,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        referenceUncertainty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenced: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        endpointStateValid: Optional[torch.Tensor] = None,
        endpointControllable: Optional[torch.Tensor] = None,
        returnExplain: bool = False,) -> NeuroSymbolicOutput:
        robot_physical_state = self.robot_state_encoder(
            bodyProprioception,
            robotPhysicalReference,
            endpointStateValid=endpointStateValid)
        control_feedback_state = self.control_feedback_encoder(
            targetTrackingError,
            plannerTrackingError,
            endpointStateValid=endpointStateValid,
            endpointControllable=endpointControllable)
        scalar = torch.stack([
            uncertainty,
            novelty,
            recentFailure,
            referenceUncertainty,
            satisfactionProb.view(-1),
            noSlotProb,], dim=-1)

        grounded = self.predicate_grounder(
            pst=pst,
            observedPst=observedPst,
            goalEmbed=goalEmbed,
            worldBelief=worldBelief,
            decisionBelief=decisionBelief,
            robotPhysicalState=robot_physical_state,
            referenced=referenced,
            uncertainty=uncertainty,
            novelty=novelty,
            recentFailure=recentFailure,
            referenceUncertainty=referenceUncertainty,
            satisfactionProb=satisfactionProb,
            referenceConfidence=referenceConfidence,
            noSlotProb=noSlotProb,)

        predicate_logits = grounded["predicate_logits"]

        temporal_predicate_logits = self.temporal_predicate_head(temporalContextFeat)

        temporal_start = len(PREDICATES) - len(TEMPORAL_PREDICATES)

        predicate_logits = torch.cat([
            predicate_logits[:, :temporal_start],
            predicate_logits[:, temporal_start:] + temporal_predicate_logits,], dim=-1)

        predicate_prob = torch.sigmoid(predicate_logits)

        ranker_in = torch.cat([
            grounded["features"],
            predicate_prob,
            control_feedback_state], dim=-1)
        ranked = self.plan_ranker(ranker_in, temporalContextFeat)

        goal_predicate_need = torch.sigmoid(self.goal_predicate_head(ranker_in))
        operator_scores = self.operator_library.Scores(predicate_prob, goal_predicate_need)
        operator_logits, _ = self.operator_library.ApplyRealmAwareLegality(
            ranked["operator_logits"] + operator_scores["symbolic_score"],
            pst,
            referenced)
        plan_latent = self.plan_ranker.RefinePlan(ranked["plan_seed"], operator_logits)
        subgoal_feature = self.BuildSubgoalFeature(
            plan_latent,
            grounded["referenced_pose"],
            robot_physical_state,
            control_feedback_state)
        operator_prob = F.softmax(operator_logits, dim=-1)

        constraint_tokens = self.constraint_head(torch.cat([
            plan_latent,
            predicate_prob,
            operator_prob,
            control_feedback_state,], dim=-1)).view(plan_latent.size(0), self.constraint_tokens, self.constraint_token_dim)

        risk_cause_raw_logits = self.failure_explainer(
            ranker_in,
            temporalContextFeat,
            predicate_prob,
            operator_logits,
            operator_scores["goal_gap"],)

        failure_gate_logits = self.failure_gate_head(torch.cat([
            ranker_in,
            temporalContextFeat,
            torch.sigmoid(risk_cause_raw_logits),], dim=-1)).squeeze(-1)

        failure_gate = torch.sigmoid(failure_gate_logits)

        risk_cause_logits = risk_cause_raw_logits + failure_gate_logits.unsqueeze(-1)

        _, runtime_action_mask = self.control_feedback_encoder.RuntimeMask(
            targetTrackingError.size(0),
            targetTrackingError.device,
            endpointStateValid,
            endpointControllable)
        safe_target_error = torch.where(
            runtime_action_mask,
            torch.nan_to_num(targetTrackingError),
            torch.zeros_like(targetTrackingError))
        safe_planner_error = torch.where(
            runtime_action_mask,
            torch.nan_to_num(plannerTrackingError),
            torch.zeros_like(plannerTrackingError))
        control_coordinate_count = runtime_action_mask.sum(
            dim=(1, 2)).clamp_min(1).to(dtype=targetTrackingError.dtype)
        control_error_evidence = torch.stack([
            torch.linalg.vector_norm(
                safe_target_error, dim=(1, 2))
            / control_coordinate_count.sqrt(),
            torch.linalg.vector_norm(
                safe_planner_error, dim=(1, 2))
            / control_coordinate_count.sqrt(),], dim=-1)
        invoke_mask = torch.sigmoid(self.invoke_head(torch.cat([
            scalar,
            control_error_evidence], dim=-1))).squeeze(-1)

        temporal_out = self.temporal_symbolic_head(
            torch.cat([ranker_in, temporalContextFeat], dim=-1),
            predicate_prob,
            operator_logits,
            risk_cause_logits,
            failure_gate,
            operator_scores["goal_gap"],)

        mixed = self.symbolic_mixer(
            plan_latent,
            predicate_prob,
            operator_logits,
            risk_cause_logits,
            failure_gate,
            temporal_out["temporal_logits"],
            temporal_out["temporal_reason_logits"],
            temporal_out["continue_guard_score"],
            temporal_out["interrupt_guard_score"],
            temporal_out["redispatch_guard_score"],
            operator_scores["goal_gap"],
            subgoal_feature,
            constraint_tokens,
            grounded["referenced_pose"],)

        plan_latent = mixed["plan_latent"]
        subgoal_feature = mixed["subgoal_feature"]
        constraint_tokens = mixed["constraint_tokens"]

        current_operator = operator_prob.argmax(dim=-1)
        if self.last_operator_prob.shape == operator_prob.shape:
            previous_operator = self.last_operator
            previous_valid = (previous_operator >= 0).float()
            same_operator = F.cosine_similarity(operator_prob, self.last_operator_prob, dim=-1) * previous_valid
            operator_changed = (1.0 - same_operator) * previous_valid
        else:
            same_operator = operator_logits.new_zeros(current_operator.shape)
            operator_changed = operator_logits.new_zeros(current_operator.shape)

        if self.last_invoke.numel() == invoke_mask.numel():
            previous_invoke = self.last_invoke
            invoke_delta = (invoke_mask - previous_invoke).abs()
        else:
            invoke_delta = invoke_mask.new_zeros(invoke_mask.shape)

        # Target-binding drift: the operator identity may stay the same while
        # the grounded target entity slowly drifts; gating with same_operator
        # makes this exactly the "same operator, different binding" detector.
        slot_summary = grounded["slot_summary"]
        if self.last_reference_summary.numel() == slot_summary.numel():
            reference_drift = (
                0.5
                * (1.0 - F.cosine_similarity(slot_summary, self.last_reference_summary, dim=-1))
                * same_operator)
        else:
            reference_drift = invoke_mask.new_zeros(invoke_mask.shape)

        self.last_operator = current_operator.detach()
        self.last_operator_prob = operator_prob.detach()
        self.last_invoke = invoke_mask.detach()
        self.last_reference_summary = slot_summary.detach()

        if returnExplain:
            facts = self.BuildFacts(predicate_prob, referenced)
            plan_steps = self.BuildPlanSteps(
                operator_logits,
                operator_scores["precond_score"],
                operator_scores["effect_score"],
                plan_latent,)
            operator_rationales = self.BuildOperatorRationales(
                operator_logits,
                predicate_prob,
                operator_scores["goal_gap"],
                risk_cause_logits,
                temporal_out["temporal_logits"],
                operator_scores["precond_score"],
                operator_scores["effect_score"],
                operator_scores["symbolic_score"],)
        else:
            facts = []
            plan_steps = []
            operator_rationales = []

        return NeuroSymbolicOutput(
            facts=facts,
            operator_logits=operator_logits,
            plan_steps=plan_steps,
            operator_rationales=operator_rationales,
            plan_latent=plan_latent,
            subgoal_feature=subgoal_feature,
            constraint_tokens=constraint_tokens,
            risk_cause_logits=risk_cause_logits,
            risk_cause_raw_logits=risk_cause_raw_logits,
            failure_cause_logits=risk_cause_logits,
            failure_cause_raw_logits=risk_cause_raw_logits,
            failure_gate_logits=failure_gate_logits,
            failure_gate=failure_gate,
            invoke_mask=invoke_mask,
            same_operator=same_operator,
            operator_changed=operator_changed,
            invoke_delta=invoke_delta,
            reference_drift=reference_drift,
            temporal_logits=temporal_out["temporal_logits"],
            temporal_reason_logits=temporal_out["temporal_reason_logits"],
            continue_guard_score=temporal_out["continue_guard_score"],
            interrupt_guard_score=temporal_out["interrupt_guard_score"],
            redispatch_guard_score=temporal_out["redispatch_guard_score"],)


class TestNeuroSymbolicMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        self.robot_morphology = self.MakeRobotMorphology(3, 5)

    def EndpointDescriptor(
        self,
        morphology: SimpleNamespace,
    ) -> Dict[str, torch.Tensor]:
        node_index = morphology.endpoint_to_node
        parent_index = morphology.parent_index.index_select(0, node_index)
        parent_valid = parent_index.ge(0)
        parent_role = torch.full_like(morphology.endpoint_role, -1)
        parent_side = torch.full_like(morphology.endpoint_side, -1)
        parent_capability = torch.zeros_like(morphology.endpoint_capability)
        parent_role[parent_valid] = morphology.node_role[
            parent_index[parent_valid]]
        parent_side[parent_valid] = morphology.node_side[
            parent_index[parent_valid]]
        parent_capability[parent_valid] = morphology.node_capability[
            parent_index[parent_valid]]
        return {
            "controllable": morphology.endpoint_task_mask.any(dim=-1),
            "parent_node_index": parent_index,
            "topology_depth": torch.ones(morphology.endpoint_count),
            "task_mask": morphology.endpoint_task_mask.clone(),
            "role": morphology.endpoint_role.clone(),
            "side": morphology.endpoint_side.clone(),
            "capability": morphology.endpoint_capability.clone(),
            "node_role": morphology.node_role.index_select(0, node_index),
            "node_side": morphology.node_side.index_select(0, node_index),
            "node_capability": morphology.node_capability.index_select(
                0, node_index),
            "parent_role": parent_role,
            "parent_side": parent_side,
            "parent_capability": parent_capability,
            "group_role_membership": torch.zeros(
                morphology.endpoint_count,
                ModuleDim.RobotBodyRoleClasses,
                dtype=torch.bool),
            "group_side_membership": torch.zeros(
                morphology.endpoint_count,
                ModuleDim.RobotBodySideClasses,
                dtype=torch.bool),
            "group_capability": torch.zeros(
                morphology.endpoint_count,
                ModuleDim.RobotBodyCapabilityDim,
                dtype=torch.bool),}

    def MakeRobotMorphology(
        self,
        endpointCount: int,
        nodeCount: int,
    ) -> SimpleNamespace:
        endpoint_count = int(endpointCount)
        node_count = int(nodeCount)
        if endpoint_count < 1 or node_count <= endpoint_count:
            raise ValueError("synthetic morphology counts are invalid")
        observer_index = endpoint_count - 1
        endpoint_task_mask = torch.zeros(
            endpoint_count,
            ModuleDim.RobotControlAxisDim,
            dtype=torch.bool)
        endpoint_task_mask[:observer_index] = True
        endpoint_task_mask[observer_index, 3:6] = True
        endpoint_to_node = torch.arange(1, endpoint_count + 1)
        endpoint_role = torch.empty(endpoint_count, dtype=torch.long)
        endpoint_side = torch.empty(endpoint_count, dtype=torch.long)
        endpoint_capability = torch.zeros(
            endpoint_count,
            ModuleDim.RobotBodyCapabilityDim)
        node_role = torch.full(
            (node_count,),
            ModuleDim.RobotBodyRoleNames.index("other"),
            dtype=torch.long)
        node_side = torch.full(
            (node_count,),
            ModuleDim.RobotBodySideNames.index("none"),
            dtype=torch.long)
        node_capability = torch.zeros(
            node_count,
            ModuleDim.RobotBodyCapabilityDim)
        role_cycle = ("arm", "hand", "sensor", "leg", "foot", "head")
        side_cycle = ("left", "right", "center")
        for index in range(endpoint_count):
            endpoint_role[index] = ModuleDim.RobotBodyRoleNames.index(
                role_cycle[index % len(role_cycle)])
            endpoint_side[index] = ModuleDim.RobotBodySideNames.index(
                side_cycle[index % len(side_cycle)])
            endpoint_capability[
                index,
                index % ModuleDim.RobotBodyCapabilityDim] = 1.0
        node_role[0] = ModuleDim.RobotBodyRoleNames.index("root")
        node_side[0] = ModuleDim.RobotBodySideNames.index("center")
        node_role[endpoint_to_node] = endpoint_role
        node_side[endpoint_to_node] = endpoint_side
        node_capability[endpoint_to_node] = endpoint_capability
        parent_index = torch.full((node_count,), -1, dtype=torch.long)
        parent_index[1:] = 0
        morphology = SimpleNamespace(
            endpoint_to_node=endpoint_to_node,
            endpoint_task_mask=endpoint_task_mask,
            endpoint_role=endpoint_role,
            endpoint_side=endpoint_side,
            endpoint_capability=endpoint_capability,
            parent_index=parent_index,
            node_role=node_role,
            node_side=node_side,
            node_capability=node_capability,
            observer_valid=True,
            observer_endpoint_index=observer_index,
            endpoint_names=tuple(
                f"endpoint_{index}" for index in range(endpoint_count)),
            endpoint_count=endpoint_count,
            node_count=node_count)
        morphology.EndpointSemanticDescriptor = (
            lambda morphology=morphology: self.EndpointDescriptor(morphology))
        return morphology

    def MakeExtractor(
        self,
        morphology: Optional[SimpleNamespace] = None,
    ) -> NeuroSymbolicExtractor:
        return NeuroSymbolicExtractor(
            robotMorphology=(
                self.robot_morphology if morphology is None else morphology)
        ).to(self.device)

    def PermuteEndpointMorphology(
        self,
        permutation: torch.Tensor,
    ) -> SimpleNamespace:
        values = dict(vars(self.robot_morphology))
        values.pop("EndpointSemanticDescriptor", None)
        for name in (
                "endpoint_to_node",
                "endpoint_task_mask",
                "endpoint_role",
                "endpoint_side",
                "endpoint_capability"):
            values[name] = values[name].index_select(0, permutation).clone()
        names = tuple(values["endpoint_names"])
        values["endpoint_names"] = tuple(
            names[int(index)]
            for index in permutation.tolist())
        observer_index = int(values["observer_endpoint_index"])
        values["observer_endpoint_index"] = int(
            torch.nonzero(
                permutation == observer_index,
                as_tuple=False).flatten()[0].item())
        morphology = SimpleNamespace(**values)
        morphology.EndpointSemanticDescriptor = (
            lambda morphology=morphology: self.EndpointDescriptor(morphology))
        return morphology

    def AssertFinite(self, value: torch.Tensor, name: str) -> None:
        assert torch.isfinite(value).all(), f"{name} contains non-finite values"

    def MakePst(
        self,
        B: int = 2,
        K: int = 4,
        morphology: Optional[SimpleNamespace] = None,
    ) -> Dict[str, torch.Tensor]:
        morphology = (
            self.robot_morphology if morphology is None else morphology)
        pose = torch.randn(B, K, ModuleDim.PstPoseDim, device=self.device) * 0.1
        pose[..., 6] = 1.0
        mask = torch.ones(B, K, device=self.device)
        realm = torch.zeros(B, K, 5, device=self.device)
        realm[..., REALM_EXTERNAL_PHYSICAL] = 1.0
        agency = torch.zeros(B, K, 5, device=self.device)
        agency[..., 4] = 1.0
        surface_parent = torch.zeros(B, K, K + 1, device=self.device)
        surface_parent[..., K] = 1.0
        return {
            "MphysRaw": mask.clone(),
            "Observed": torch.ones(B, K, device=self.device, dtype=torch.bool),
            "LastSeen": torch.arange(K, device=self.device, dtype=torch.float32).unsqueeze(0).expand(B, -1),
            "Step": torch.full((B,), K, device=self.device, dtype=torch.long),
            "SlotPresence": mask.clone(),
            "ObservedSlotMask": mask.clone(),
            "SlotState": torch.randn(B, K, ModuleDim.PstSlotDim, device=self.device),
            "PoseCamera": pose,
            "PerceptualPresence": mask.clone(),
            "PhysicalInteractionProb": mask.clone(),
            "RealmProb": realm,
            "MotionLayerProb": torch.zeros(B, K, 5, device=self.device),
            "LayerAgencyProb": torch.zeros(B, K, 5, 5, device=self.device),
            "AgencyProb": agency,
            "BodyMembershipProb": torch.zeros(B, K, device=self.device),
            "SelfPartProb": torch.zeros(
                B, K, morphology.node_count, device=self.device),
            "SurfaceParentProb": surface_parent,
            "SurfaceUV": torch.full((B, K, 2), 0.5, device=self.device),
            "SurfaceUVConfidence": mask.clone(),
            "DisplaySurfaceProb": mask.clone(),
            "VerificationConfidence": mask.clone(),}

    def MakeExtractorInputs(
        self,
        B: int = 2,
        K: int = 4,
        morphology: Optional[SimpleNamespace] = None,
    ) -> Dict[str, torch.Tensor]:
        morphology = (
            self.robot_morphology if morphology is None else morphology)
        endpoint_count = int(morphology.endpoint_count)
        referenced = F.softmax(torch.randn(B, K, device=self.device), dim=-1)
        body_proprioception = torch.zeros(
            B,
            endpoint_count,
            ModuleDim.DecisionEndpointPoseDim,
            device=self.device)
        body_proprioception[..., 6] = 1.0
        robot_physical_reference = torch.zeros(
            B, ModuleDim.RobotPhysicalReferenceDim, device=self.device)
        robot_physical_reference[:, 3] = 1.0
        robot_physical_reference[:, 6] = -1.0
        robot_physical_reference[:, 7] = 1.0
        return {
            "pst": self.MakePst(B=B, K=K, morphology=morphology),
            "observedPst": self.MakePst(
                B=B, K=K, morphology=morphology),
            "goalEmbed": torch.randn(B, ModuleDim.GoalShortDim, device=self.device),
            "worldBelief": torch.randn(
                B,
                ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
                device=self.device),
            "decisionBelief": torch.randn(B, ModuleDim.DecisionBeliefDim, device=self.device),
            "bodyProprioception": body_proprioception,
            "robotPhysicalReference": robot_physical_reference,
            "targetTrackingError": torch.zeros(
                B,
                endpoint_count,
                ModuleDim.RobotControlAxisDim,
                device=self.device),
            "plannerTrackingError": torch.zeros(
                B,
                endpoint_count,
                ModuleDim.RobotControlAxisDim,
                device=self.device),
            "endpointStateValid": torch.ones(
                B, endpoint_count, device=self.device, dtype=torch.bool),
            "endpointControllable": torch.ones(
                B, endpoint_count, device=self.device, dtype=torch.bool),
            "uncertainty": torch.rand(B, device=self.device),
            "novelty": torch.rand(B, device=self.device),
            "recentFailure": torch.rand(B, device=self.device),
            "referenceUncertainty": torch.rand(B, device=self.device),
            "satisfactionProb": torch.rand(B, device=self.device),
            "referenced": referenced,
            "referenceConfidence": torch.rand(B, device=self.device),
            "noSlotProb": torch.rand(B, device=self.device),
            "temporalContextFeat": torch.randn(B, ModuleDim.TemporalContextDim, device=self.device),}

    def TestPredicateGrounderShapes(self) -> bool:
        try:
            B, K = 2, 4
            model = PredicateGrounder().to(self.device)
            inputs = self.MakeExtractorInputs(B=B, K=K)
            out = model(
                pst=inputs["pst"],
                observedPst=inputs["observedPst"],
                goalEmbed=inputs["goalEmbed"],
                worldBelief=inputs["worldBelief"],
                decisionBelief=inputs["decisionBelief"],
                robotPhysicalState=torch.randn(
                    B, ModuleDim.PstSlotDim, device=self.device),
                referenced=inputs["referenced"],
                uncertainty=inputs["uncertainty"],
                novelty=inputs["novelty"],
                recentFailure=inputs["recentFailure"],
                referenceUncertainty=inputs["referenceUncertainty"],
                satisfactionProb=inputs["satisfactionProb"],
                referenceConfidence=inputs["referenceConfidence"],
                noSlotProb=inputs["noSlotProb"],)
            assert tuple(out["features"].shape) == (B, model.feature_dim)
            assert tuple(out["predicate_logits"].shape) == (B, len(PREDICATES))
            assert tuple(out["predicate_prob"].shape) == (B, len(PREDICATES))
            assert tuple(out["referenced_pose"].shape) == (B, ModuleDim.PstPoseDim)
            assert tuple(out["memory_reference_scale"].shape) == (B,)
            assert tuple(out["reference_slot_idx"].shape) == (B,)
            assert tuple(out["slot_summary"].shape) == (B, ModuleDim.PstSlotDim)
            for name, value in out.items():
                self.AssertFinite(value, f"PredicateGrounder {name}")
            print("PredicateGrounder shape test passed.")
            return True
        except Exception as e:
            print(f"PredicateGrounder shape test failed: {type(e).__name__}: {e}")
            return False

    def TestOperatorLibraryScores(self) -> bool:
        try:
            B = 2
            library = OperatorLibrary().to(self.device)
            predicate_prob = torch.zeros(B, len(PREDICATES), device=self.device)
            goal_need = torch.zeros(B, len(PREDICATES), device=self.device)
            pred = {name: i for i, name in enumerate(PREDICATES)}
            op = {name: i for i, name in enumerate(OPERATORS)}
            predicate_prob[:, pred["reachable"]] = 1.0
            predicate_prob[:, pred["contactable"]] = 1.0
            predicate_prob[:, pred["graspable"]] = 1.0
            goal_need[:, pred["attached"]] = 1.0
            out = library.Scores(predicate_prob, goal_need)
            assert tuple(library.precondition_mask.shape) == (len(OPERATORS), len(PREDICATES))
            assert tuple(library.effect_mask.shape) == (len(OPERATORS), len(PREDICATES))
            assert tuple(out["precond_score"].shape) == (B, len(OPERATORS))
            assert tuple(out["effect_score"].shape) == (B, len(OPERATORS))
            assert tuple(out["symbolic_score"].shape) == (B, len(OPERATORS))
            assert bool((out["precond_score"][:, op["grasp"]] > 0.99).all().item())
            assert bool((out["effect_score"][:, op["grasp"]] > 0.99).all().item())
            for name, value in out.items():
                self.AssertFinite(value, f"OperatorLibrary {name}")
            print("OperatorLibrary score test passed.")
            return True
        except Exception as e:
            print(f"OperatorLibrary score test failed: {type(e).__name__}: {e}")
            return False

    def TestRealmAwareOperatorLegality(self) -> bool:
        try:
            B, K = 6, 4
            library = OperatorLibrary().to(self.device)
            pst = self.MakePst(B=B, K=K)
            referenced = torch.zeros(B, K, device=self.device)
            referenced[0, 0] = 1.0
            referenced[1, 1] = 1.0
            referenced[2, 1] = 1.0
            referenced[3, 2] = 1.0
            referenced[4, 1] = 1.0
            referenced[5, 1] = 1.0

            for row in (1, 2, 4, 5):
                pst["RealmProb"][row, 1].zero_()
                pst["RealmProb"][row, 1, REALM_VIRTUAL_CONTENT] = 1.0
                pst["PhysicalInteractionProb"][row, 1] = 0.0
            pst["SurfaceParentProb"][1, 1].zero_()
            pst["SurfaceParentProb"][1, 1, 0] = 1.0
            pst["SurfaceParentProb"][4, 1].zero_()
            pst["SurfaceParentProb"][4, 1, 0] = 1.0
            pst["SurfaceUVConfidence"][4, 1] = 0.0
            pst["SurfaceParentProb"][5, 1].zero_()
            pst["SurfaceParentProb"][5, 1, 0] = 1.0
            pst["DisplaySurfaceProb"][5, 0] = 0.0
            pst["RealmProb"][3, 2].zero_()
            pst["RealmProb"][3, 2, int(Realm.VISUAL_EFFECT)] = 1.0
            pst["PhysicalInteractionProb"][3, 2] = 0.0

            result = library.RealmAwareLegality(pst, referenced)
            legality = result["operator_legality"]
            op = {name: i for i, name in enumerate(OPERATORS)}
            for name in (
                    "observe", "reobserve", "wait",
                    "cancel_execute", "failsafe_stop"):
                assert bool(legality[:, op[name]].all().item())
            assert bool(legality[0, op["grasp"]].item())
            assert not bool(legality[1:, op["grasp"]].any().item())
            assert bool(legality[1, op["press"]].item())
            assert not bool(legality[2, op["press"]].item())
            assert not bool(legality[3, op["press"]].item())
            assert torch.equal(
                result["verified_surface"],
                torch.tensor(
                    [False, True, False, False, False, False],
                    device=self.device))

            logits = torch.randn(B, len(OPERATORS), device=self.device)
            masked, _ = library.ApplyRealmAwareLegality(
                logits, pst, referenced)
            legal_min = torch.stack([
                masked[row, legality[row]].amin()
                for row in range(B)])
            assert bool((masked[1:, op["grasp"]] < legal_min[1:]).all().item())
            assert torch.isfinite(masked[:, op["observe"]]).all()
            assert torch.isfinite(masked).all()
            print("RealmAwareOperatorLegality passed.")
            return True
        except Exception as e:
            print(f"RealmAwareOperatorLegality failed: {type(e).__name__}: {e}")
            return False

    def TestPlanRankerShapes(self) -> bool:
        try:
            B, input_dim, hidden, plan_dim = 2, 64, 128, 32
            model = PlanRanker(inputDim=input_dim, hidden=hidden, planDim=plan_dim).to(self.device)
            out = model(
                torch.randn(B, input_dim, device=self.device),
                torch.randn(B, ModuleDim.TemporalContextDim, device=self.device),)
            refined = model.RefinePlan(out["plan_seed"], out["operator_logits"])
            assert tuple(out["plan_seed"].shape) == (B, plan_dim)
            assert tuple(out["operator_logits"].shape) == (B, len(OPERATORS))
            assert tuple(out["ranker_state"].shape) == (B, hidden)
            assert tuple(refined.shape) == (B, plan_dim)
            for name, value in out.items():
                self.AssertFinite(value, f"PlanRanker {name}")
            self.AssertFinite(refined, "PlanRanker refined plan")
            print("PlanRanker shape test passed.")
            return True
        except Exception as e:
            print(f"PlanRanker shape test failed: {type(e).__name__}: {e}")
            return False

    def TestFailureExplainerShapes(self) -> bool:
        try:
            B, input_dim = 2, 64
            model = FailureExplainer(inputDim=input_dim, hidden=128).to(self.device)
            predicate_prob = torch.rand(B, len(PREDICATES), device=self.device)
            operator_logits = torch.randn(B, len(OPERATORS), device=self.device)
            goal_gap = torch.rand(B, len(PREDICATES), device=self.device)
            evidence = model.PredicateEvidence(predicate_prob)
            out = model(
                torch.randn(B, input_dim, device=self.device),
                torch.randn(B, ModuleDim.TemporalContextDim, device=self.device),
                predicate_prob,
                operator_logits,
                goal_gap,)
            assert tuple(evidence.shape) == (B, len(FAILURE_CAUSES))
            assert tuple(out.shape) == (B, len(FAILURE_CAUSES))
            self.AssertFinite(evidence, "FailureExplainer evidence")
            self.AssertFinite(out, "FailureExplainer logits")
            print("FailureExplainer shape test passed.")
            return True
        except Exception as e:
            print(f"FailureExplainer shape test failed: {type(e).__name__}: {e}")
            return False

    def TestTemporalSymbolicHeadShapes(self) -> bool:
        try:
            B, input_dim = 2, 64
            model = TemporalSymbolicHead(inputDim=input_dim, hidden=128).to(self.device)
            predicate_prob = torch.rand(B, len(PREDICATES), device=self.device)
            operator_logits = torch.randn(B, len(OPERATORS), device=self.device)
            risk_logits = torch.randn(B, len(FAILURE_CAUSES), device=self.device)
            failure_gate = torch.rand(B, device=self.device)
            goal_gap = torch.rand(B, len(PREDICATES), device=self.device)
            out = model(
                torch.randn(B, input_dim, device=self.device),
                predicate_prob,
                operator_logits,
                risk_logits,
                failure_gate,
                goal_gap,)
            assert tuple(model.PrimitiveRulePrior(predicate_prob).shape) == (B, ModuleDim.TemporalPrimitiveCount)
            assert tuple(out["temporal_logits"].shape) == (B, ModuleDim.TemporalPrimitiveCount)
            assert tuple(out["temporal_reason_logits"].shape) == (B, ModuleDim.TemporalReasonDim)
            assert tuple(out["continue_guard_score"].shape) == (B,)
            assert tuple(out["interrupt_guard_score"].shape) == (B,)
            assert tuple(out["redispatch_guard_score"].shape) == (B,)
            for name, value in out.items():
                self.AssertFinite(value, f"TemporalSymbolicHead {name}")
            print("TemporalSymbolicHead shape test passed.")
            return True
        except Exception as e:
            print(f"TemporalSymbolicHead shape test failed: {type(e).__name__}: {e}")
            return False

    def TestSymbolicFeatureMixerShapes(self) -> bool:
        try:
            B, plan_dim, subgoal_dim, token_count, token_dim = 2, 32, 16, 4, 16
            model = SymbolicFeatureMixer(
                planDim=plan_dim,
                subgoalFeatureDim=subgoal_dim,
                constraintTokens=token_count,
                constraintTokenDim=token_dim,
                hidden=128,).to(self.device)
            out = model(
                torch.randn(B, plan_dim, device=self.device),
                torch.rand(B, len(PREDICATES), device=self.device),
                torch.randn(B, len(OPERATORS), device=self.device),
                torch.randn(B, len(FAILURE_CAUSES), device=self.device),
                torch.rand(B, device=self.device),
                torch.randn(B, ModuleDim.TemporalPrimitiveCount, device=self.device),
                torch.randn(B, ModuleDim.TemporalReasonDim, device=self.device),
                torch.rand(B, device=self.device),
                torch.rand(B, device=self.device),
                torch.rand(B, device=self.device),
                torch.rand(B, len(PREDICATES), device=self.device),
                torch.randn(B, subgoal_dim, device=self.device),
                torch.randn(B, token_count, token_dim, device=self.device),
                torch.randn(B, ModuleDim.PstPoseDim, device=self.device),)
            assert tuple(out["plan_latent"].shape) == (B, plan_dim)
            assert tuple(out["subgoal_feature"].shape) == (B, subgoal_dim)
            assert tuple(out["constraint_tokens"].shape) == (B, token_count, token_dim)
            for name, value in out.items():
                self.AssertFinite(value, f"SymbolicFeatureMixer {name}")
            print("SymbolicFeatureMixer shape test passed.")
            return True
        except Exception as e:
            print(f"SymbolicFeatureMixer shape test failed: {type(e).__name__}: {e}")
            return False

    def TestNeuroSymbolicExtractorForwardAndExplainSwitch(self) -> bool:
        try:
            B, K = 2, 4
            model = self.MakeExtractor()
            model.eval()
            inputs = self.MakeExtractorInputs(B=B, K=K)
            with torch.no_grad():
                out_no_explain = model(**inputs, returnExplain=False)
                out_explain = model(**inputs, returnExplain=True)
            assert tuple(out_no_explain.operator_logits.shape) == (B, len(OPERATORS))
            assert tuple(out_no_explain.plan_latent.shape) == (B, 256)
            assert tuple(out_no_explain.subgoal_feature.shape) == (B, ModuleDim.DecisionEndpointPoseFeatDim)
            assert tuple(out_no_explain.constraint_tokens.shape) == (B, 8, 128)
            assert tuple(out_no_explain.risk_cause_logits.shape) == (B, len(FAILURE_CAUSES))
            assert tuple(out_no_explain.temporal_logits.shape) == (B, ModuleDim.TemporalPrimitiveCount)
            assert tuple(out_no_explain.temporal_reason_logits.shape) == (B, ModuleDim.TemporalReasonDim)
            assert tuple(out_no_explain.reference_drift.shape) == (B,)
            assert not any(
                name.endswith((
                    "endpoint_valid",
                    "action_mask",
                    "decision_action_mask",
                ))
                for name in model.state_dict())
            assert len(out_no_explain.facts) == 0
            assert len(out_no_explain.plan_steps) == 0
            assert len(out_no_explain.operator_rationales) == 0
            assert len(out_explain.facts) == len(PREDICATES)
            assert len(out_explain.plan_steps) == B
            assert len(out_explain.operator_rationales) == B
            self.AssertFinite(out_no_explain.operator_logits, "NeuroSymbolicExtractor operator_logits")
            self.AssertFinite(out_no_explain.plan_latent, "NeuroSymbolicExtractor plan_latent")
            self.AssertFinite(out_no_explain.subgoal_feature, "NeuroSymbolicExtractor subgoal_feature")
            self.AssertFinite(out_no_explain.constraint_tokens, "NeuroSymbolicExtractor constraint_tokens")
            self.AssertFinite(out_no_explain.risk_cause_logits, "NeuroSymbolicExtractor risk_cause_logits")
            self.AssertFinite(out_no_explain.invoke_mask, "NeuroSymbolicExtractor invoke_mask")
            self.AssertFinite(out_no_explain.reference_drift, "NeuroSymbolicExtractor reference_drift")
            self.AssertFinite(out_no_explain.temporal_logits, "NeuroSymbolicExtractor temporal_logits")
            print("NeuroSymbolicExtractor forward/explain switch test passed.")
            return True
        except Exception as e:
            print(f"NeuroSymbolicExtractor forward/explain switch test failed: {type(e).__name__}: {e}")
            return False

    def TestNeuroSymbolicPhysicalAndControlInputSeparation(self) -> bool:
        try:
            from inspect import signature

            B, K = 2, 4
            model = self.MakeExtractor().eval()
            inputs = self.MakeExtractorInputs(B=B, K=K)
            forward_parameters = set(signature(model.forward).parameters)
            assert {
                "bodyProprioception",
                "robotPhysicalReference",
                "targetTrackingError",
                "plannerTrackingError",
                "endpointStateValid",
                "endpointControllable",} <= forward_parameters
            assert forward_parameters.isdisjoint({
                "robotPhysicalState",
                "robotSelfState",
                "endpointControlEncoding",})

            physical_captures = []
            control_captures = []

            def CaptureRobotPhysicalInput(module, args, output):
                del module
                assert len(args) == 2
                physical_captures.append(
                    (args[0], args[1], output.detach().clone()))

            def CaptureControlFeedbackInput(module, args, output):
                del module
                assert len(args) == 2
                control_captures.append(
                    (args[0], args[1], output.detach().clone()))

            physical_handle = model.robot_state_encoder.register_forward_hook(
                CaptureRobotPhysicalInput)
            control_handle = model.control_feedback_encoder.register_forward_hook(
                CaptureControlFeedbackInput)

            def EncodeThroughExtractor(**overrides):
                run_inputs = {**inputs, **overrides}
                physical_captures.clear()
                control_captures.clear()
                model.ResetPlan(torch.ones(
                    B, device=self.device, dtype=torch.bool))
                with torch.no_grad():
                    output = model(**run_inputs, returnExplain=False)
                assert len(physical_captures) == 1
                assert len(control_captures) == 1
                body_input, reference_input, physical_latent = physical_captures[0]
                target_input, planner_input, control_latent = control_captures[0]
                assert body_input is run_inputs["bodyProprioception"]
                assert reference_input is run_inputs["robotPhysicalReference"]
                assert target_input is run_inputs["targetTrackingError"]
                assert planner_input is run_inputs["plannerTrackingError"]
                return physical_latent, control_latent, output

            try:
                baseline_physical, baseline_control, baseline_output = (
                    EncodeThroughExtractor())
                external_physical, external_control, _ = EncodeThroughExtractor(
                    worldBelief=torch.full_like(inputs["worldBelief"], -53.0),
                    decisionBelief=torch.full_like(inputs["decisionBelief"], 37.0))

                changed_body = inputs["bodyProprioception"].clone()
                changed_body[:, 0, 0] = 0.25
                body_physical, _, _ = EncodeThroughExtractor(
                    bodyProprioception=changed_body)

                changed_rotation = inputs["robotPhysicalReference"].clone()
                sqrt_half = 2.0 ** -0.5
                changed_rotation[:, 0:4] = changed_rotation.new_tensor([
                    0.0, 0.0, sqrt_half, sqrt_half])
                rotation_physical, _, _ = EncodeThroughExtractor(
                    robotPhysicalReference=changed_rotation)

                changed_gravity = inputs["robotPhysicalReference"].clone()
                changed_gravity[:, 4:7] = changed_gravity.new_tensor([
                    0.0, -1.0, 0.0])
                gravity_physical, _, _ = EncodeThroughExtractor(
                    robotPhysicalReference=changed_gravity)

                changed_target = inputs["targetTrackingError"].clone()
                changed_target[:, 0, 0] = 2.0
                target_physical, target_control, target_output = (
                    EncodeThroughExtractor(targetTrackingError=changed_target))

                changed_planner = inputs["plannerTrackingError"].clone()
                changed_planner[:, 1, 3] = -3.0
                planner_physical, planner_control, planner_output = (
                    EncodeThroughExtractor(plannerTrackingError=changed_planner))

                observer_index = (
                    self.robot_morphology.observer_endpoint_index)
                changed_masked_observer = inputs["targetTrackingError"].clone()
                changed_masked_observer[:, observer_index, 0] = 23.0
                masked_observer_physical, masked_observer_control, _ = (
                    EncodeThroughExtractor(
                        targetTrackingError=changed_masked_observer))

                changed_observer_rotation = (
                    inputs["targetTrackingError"].clone())
                changed_observer_rotation[:, observer_index, 5] = 5.0
                observer_physical, observer_control, observer_output = (
                    EncodeThroughExtractor(
                        targetTrackingError=changed_observer_rotation))

            finally:
                physical_handle.remove()
                control_handle.remove()

            assert tuple(baseline_physical.shape) == (B, ModuleDim.PstSlotDim)
            assert tuple(baseline_control.shape) == (
                B, ModuleDim.DecisionEndpointPoseFeatDim)
            self.AssertFinite(
                baseline_physical,
                "NeuroSymbolic private robot physical latent")
            self.AssertFinite(
                baseline_control,
                "NeuroSymbolic private control feedback latent")
            assert torch.equal(baseline_physical, external_physical)
            assert torch.equal(baseline_control, external_control)
            for latent in (
                    body_physical,
                    rotation_physical,
                    gravity_physical):
                per_sample_delta = (
                    latent - baseline_physical).abs().amax(dim=-1)
                assert bool((per_sample_delta > 1e-7).all().item())
            assert torch.equal(baseline_physical, target_physical)
            assert torch.equal(baseline_physical, planner_physical)
            assert torch.equal(baseline_physical, masked_observer_physical)
            assert torch.equal(baseline_physical, observer_physical)
            assert torch.equal(baseline_control, masked_observer_control)
            for latent in (
                    target_control,
                    planner_control,
                    observer_control):
                per_sample_delta = (
                    latent - baseline_control).abs().amax(dim=-1)
                assert bool((per_sample_delta > 1e-7).all().item())

            for changed_output in (
                    target_output,
                    planner_output,
                    observer_output):
                output_delta = (
                    (changed_output.subgoal_feature - baseline_output.subgoal_feature)
                    .abs().sum()
                    + (changed_output.constraint_tokens - baseline_output.constraint_tokens)
                    .abs().sum()
                    + (changed_output.invoke_mask - baseline_output.invoke_mask)
                    .abs().sum())
                assert bool((output_delta > 1e-7).item())

            no_endpoint = torch.zeros(
                B,
                self.robot_morphology.endpoint_count,
                device=self.device,
                dtype=torch.bool)
            with torch.no_grad():
                zero_physical = model.robot_state_encoder(
                    inputs["bodyProprioception"],
                    inputs["robotPhysicalReference"],
                    endpointStateValid=no_endpoint)
                zero_control = model.control_feedback_encoder(
                    inputs["targetTrackingError"],
                    inputs["plannerTrackingError"],
                    endpointStateValid=no_endpoint,
                    endpointControllable=no_endpoint)
            assert torch.equal(
                zero_physical,
                torch.zeros_like(zero_physical))
            assert torch.equal(
                zero_control,
                torch.zeros_like(zero_control))

            physical_input = torch.randn_like(
                inputs["bodyProprioception"], requires_grad=True)
            one_endpoint = no_endpoint.clone()
            one_endpoint[:, 0] = True
            physical_loss = model.robot_state_encoder(
                physical_input,
                inputs["robotPhysicalReference"],
                endpointStateValid=one_endpoint).square().sum()
            physical_loss.backward()
            assert physical_input.grad is not None
            assert int(torch.count_nonzero(
                physical_input.grad[:, 1:]).item()) == 0

            target_input = torch.randn_like(
                inputs["targetTrackingError"], requires_grad=True)
            planner_input = torch.randn_like(
                inputs["plannerTrackingError"], requires_grad=True)
            runtime_endpoint = torch.ones(
                B,
                self.robot_morphology.endpoint_count,
                device=self.device,
                dtype=torch.bool)
            control_loss = model.control_feedback_encoder(
                target_input,
                planner_input,
                endpointStateValid=runtime_endpoint,
                endpointControllable=runtime_endpoint).square().sum()
            control_loss.backward()
            _, runtime_action_mask = (
                model.control_feedback_encoder.RuntimeMask(
                    B,
                    self.device,
                    runtime_endpoint,
                    runtime_endpoint))
            assert target_input.grad is not None
            assert planner_input.grad is not None
            assert int(torch.count_nonzero(
                target_input.grad.masked_select(
                    ~runtime_action_mask)).item()) == 0
            assert int(torch.count_nonzero(
                planner_input.grad.masked_select(
                    ~runtime_action_mask)).item()) == 0

            permutation = torch.arange(
                self.robot_morphology.endpoint_count,
                dtype=torch.long).roll(1)
            permuted_morphology = self.PermuteEndpointMorphology(permutation)
            permuted_model = NeuroSymbolicExtractor(
                robotMorphology=permuted_morphology).to(self.device).eval()
            permuted_model.load_state_dict(model.state_dict())
            permutation_device = permutation.to(self.device)
            permutation_body = torch.randn_like(
                inputs["bodyProprioception"])
            permutation_target = torch.randn_like(
                inputs["targetTrackingError"])
            permutation_planner = torch.randn_like(
                inputs["plannerTrackingError"])
            with torch.no_grad():
                original_physical = model.robot_state_encoder(
                    permutation_body,
                    inputs["robotPhysicalReference"],
                    endpointStateValid=inputs["endpointStateValid"])
                original_control = model.control_feedback_encoder(
                    permutation_target,
                    permutation_planner,
                    endpointStateValid=inputs["endpointStateValid"],
                    endpointControllable=inputs["endpointControllable"])
                synchronized_physical = (
                    permuted_model.robot_state_encoder(
                        permutation_body.index_select(
                            1, permutation_device),
                        inputs["robotPhysicalReference"],
                        endpointStateValid=inputs[
                            "endpointStateValid"].index_select(
                                1, permutation_device)))
                synchronized_control = (
                    permuted_model.control_feedback_encoder(
                        permutation_target.index_select(
                            1, permutation_device),
                        permutation_planner.index_select(
                            1, permutation_device),
                        endpointStateValid=inputs[
                            "endpointStateValid"].index_select(
                                1, permutation_device),
                        endpointControllable=inputs[
                            "endpointControllable"].index_select(
                                1, permutation_device)))
                unsynchronized_physical = model.robot_state_encoder(
                    permutation_body.index_select(
                        1, permutation_device),
                    inputs["robotPhysicalReference"],
                    endpointStateValid=inputs[
                        "endpointStateValid"].index_select(
                            1, permutation_device))
                unsynchronized_control = model.control_feedback_encoder(
                    permutation_target.index_select(
                        1, permutation_device),
                    permutation_planner.index_select(
                        1, permutation_device),
                    endpointStateValid=inputs[
                        "endpointStateValid"].index_select(
                            1, permutation_device),
                    endpointControllable=inputs[
                        "endpointControllable"].index_select(
                            1, permutation_device))
            assert torch.allclose(
                original_physical,
                synchronized_physical,
                atol=1e-6,
                rtol=1e-5)
            assert torch.allclose(
                original_control,
                synchronized_control,
                atol=1e-6,
                rtol=1e-5)
            assert bool((
                original_physical - unsynchronized_physical
            ).abs().amax().item() > 1e-7)
            assert bool((
                original_control - unsynchronized_control
            ).abs().amax().item() > 1e-7)

            print(
                "NeuroSymbolicExtractor physical/control input separation test passed.")
            return True
        except Exception as e:
            print(
                "NeuroSymbolicExtractor physical/control input separation "
                f"test failed: {type(e).__name__}: {e}")
            return False

    def TestNeuroSymbolicActualTopologyShapes(self) -> bool:
        try:
            morphologies = (
                self.MakeRobotMorphology(2, 4),
                self.MakeRobotMorphology(5, 8),)
            models = tuple(
                self.MakeExtractor(morphology).eval()
                for morphology in morphologies)
            models[1].load_state_dict(models[0].state_dict())
            for morphology, model in zip(morphologies, models):
                B, K = 2, 3
                inputs = self.MakeExtractorInputs(
                    B=B,
                    K=K,
                    morphology=morphology)
                with torch.no_grad():
                    output = model(**inputs, returnExplain=False)
                endpoint_count = int(morphology.endpoint_count)
                assert model.robot_state_encoder.endpoint_count == endpoint_count
                assert model.control_feedback_encoder.endpoint_count == endpoint_count
                assert tuple(inputs["bodyProprioception"].shape) == (
                    B, endpoint_count, ModuleDim.DecisionEndpointPoseDim)
                assert tuple(inputs["targetTrackingError"].shape) == (
                    B, endpoint_count, ModuleDim.RobotControlAxisDim)
                self.AssertFinite(
                    output.subgoal_feature,
                    "NeuroSymbolic actual topology subgoal feature")
                self.AssertFinite(
                    output.constraint_tokens,
                    "NeuroSymbolic actual topology constraint tokens")
            print("NeuroSymbolicExtractor actual topology shape test passed.")
            return True
        except Exception as e:
            print(
                "NeuroSymbolicExtractor actual topology shape test failed: "
                f"{type(e).__name__}: {e}")
            return False

    def TestNeuroSymbolicContinuityAndReset(self) -> bool:
        try:
            B, K = 2, 4
            model = self.MakeExtractor()
            model.eval()
            inputs = self.MakeExtractorInputs(B=B, K=K)
            with torch.no_grad():
                first = model(**inputs, returnExplain=False)
                second = model(**inputs, returnExplain=False)
                model.ResetPlan(torch.ones(B, device=self.device, dtype=torch.bool))
                third = model(**inputs, returnExplain=False)
            assert bool((first.same_operator == 0.0).all().item())
            assert bool((first.operator_changed == 0.0).all().item())
            assert bool((second.same_operator > 0.999).all().item())
            assert bool((second.operator_changed < 1e-4).all().item())
            assert bool((second.invoke_delta < 1e-6).all().item())
            assert bool((second.reference_drift < 1e-6).all().item())
            assert bool((third.same_operator == 0.0).all().item())
            assert bool((third.operator_changed == 0.0).all().item())
            assert bool((third.reference_drift == 0.0).all().item())
            print("NeuroSymbolicExtractor continuity/reset test passed.")
            return True
        except Exception as e:
            print(f"NeuroSymbolicExtractor continuity/reset test failed: {type(e).__name__}: {e}")
            return False

    def TestReferenceDriftRespondsToBindingChange(self) -> bool:
        try:
            B, K = 2, 4
            model = self.MakeExtractor()
            model.eval()
            inputs = self.MakeExtractorInputs(B=B, K=K)
            with torch.no_grad():
                _ = model(**inputs, returnExplain=False)
                model.last_reference_summary = -model.last_reference_summary
                drifted = model(**inputs, returnExplain=False)
            assert bool((drifted.same_operator > 0.999).all().item())
            assert bool((drifted.reference_drift > 0.90).all().item())
            self.AssertFinite(drifted.reference_drift, "NeuroSymbolicExtractor drifted reference_drift")
            print("NeuroSymbolicExtractor reference drift response test passed.")
            return True
        except Exception as e:
            print(f"NeuroSymbolicExtractor reference drift response test failed: {type(e).__name__}: {e}")
            return False

    def TestPlanStateExportImportRoundTrip(self) -> bool:
        try:
            B, K = 2, 4
            model = self.MakeExtractor()
            model.eval()
            inputs = self.MakeExtractorInputs(B=B, K=K)
            with torch.no_grad():
                _ = model(**inputs, returnExplain=False)
                state = model.ExportPlanState()
            restored = self.MakeExtractor()
            restored.ImportPlanState(state)
            assert torch.equal(restored.last_operator.cpu(), state["last_operator"].cpu())
            assert torch.allclose(restored.last_operator_prob.cpu(), state["last_operator_prob"].cpu())
            assert torch.allclose(restored.last_invoke.cpu(), state["last_invoke"].cpu())
            assert torch.allclose(restored.last_reference_summary.cpu(), state["last_reference_summary"].cpu())
            print("NeuroSymbolicExtractor plan state round-trip test passed.")
            return True
        except Exception as e:
            print(f"NeuroSymbolicExtractor plan state round-trip test failed: {type(e).__name__}: {e}")
            return False

    def TestPlanStateStrictSchema(self) -> bool:
        try:
            model = self.MakeExtractor()
            try:
                model.ImportPlanState({
                    "last_operator": torch.zeros(2, dtype=torch.long, device=self.device),
                    "last_invoke": torch.zeros(2, device=self.device),})
            except KeyError as e:
                assert "last_operator_prob" in str(e)
                print("NeuroSymbolicExtractor strict plan state schema test passed.")
                return True
            raise AssertionError("ImportPlanState accepted a state without last_operator_prob")
        except Exception as e:
            print(f"NeuroSymbolicExtractor strict plan state schema test failed: {type(e).__name__}: {e}")
            return False

    def TestInvocationNeedCalibration(self) -> bool:
        try:
            zeros = torch.zeros(2, device=self.device)
            low = NeuroSymbolicExtractor.InvocationNeedTarget(
                zeros, zeros, zeros, zeros, torch.ones_like(zeros), zeros)
            failure = NeuroSymbolicExtractor.InvocationNeedTarget(
                zeros, zeros, torch.ones_like(zeros), zeros,
                torch.ones_like(zeros), zeros)
            missing_reference = NeuroSymbolicExtractor.InvocationNeedTarget(
                zeros, zeros, zeros, zeros, zeros, torch.ones_like(zeros))

            invoke = torch.tensor(
                [0.2, 0.8], device=self.device, requires_grad=True)
            target = torch.tensor([1.0, 0.0], device=self.device)
            F.binary_cross_entropy(invoke, target).backward()
            ok = bool(
                torch.count_nonzero(low).item() == 0
                and torch.allclose(
                    failure,
                    torch.full_like(failure, 0.5))
                and torch.allclose(
                    missing_reference,
                    torch.ones_like(missing_reference))
                and invoke.grad is not None
                and invoke.grad[0] < 0.0
                and invoke.grad[1] > 0.0)
            print(f"InvocationNeedCalibration {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"InvocationNeedCalibration failed: {type(e).__name__}: {e}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "PredicateGrounderShapes": self.TestPredicateGrounderShapes(),
            "OperatorLibraryScores": self.TestOperatorLibraryScores(),
            "RealmAwareOperatorLegality": self.TestRealmAwareOperatorLegality(),
            "PlanRankerShapes": self.TestPlanRankerShapes(),
            "FailureExplainerShapes": self.TestFailureExplainerShapes(),
            "TemporalSymbolicHeadShapes": self.TestTemporalSymbolicHeadShapes(),
            "SymbolicFeatureMixerShapes": self.TestSymbolicFeatureMixerShapes(),
            "NeuroSymbolicExtractorForwardAndExplainSwitch": self.TestNeuroSymbolicExtractorForwardAndExplainSwitch(),
            "NeuroSymbolicPhysicalAndControlInputSeparation": self.TestNeuroSymbolicPhysicalAndControlInputSeparation(),
            "NeuroSymbolicActualTopologyShapes": self.TestNeuroSymbolicActualTopologyShapes(),
            "NeuroSymbolicContinuityAndReset": self.TestNeuroSymbolicContinuityAndReset(),
            "ReferenceDriftRespondsToBindingChange": self.TestReferenceDriftRespondsToBindingChange(),
            "PlanStateExportImportRoundTrip": self.TestPlanStateExportImportRoundTrip(),
            "PlanStateStrictSchema": self.TestPlanStateStrictSchema(),}
        results["InvocationNeedCalibration"] = self.TestInvocationNeedCalibration()
        passed = sum(1 for value in results.values() if value)
        print(f"\n[NeuroSymbolicModule Tests] {passed}/{len(results)} passed.")
        return results
