from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    RobotEmbodimentContractView,
)


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


CONTRACT_SLOT_PREDICATES: Tuple[str, ...] = (
    "HAS_PARENT",
    "PARENT_READY",
    "SLOT_ENABLED",
    "PRESENT",
    "COMMAND_IN_ALLOWED_SUBSPACE",
)


CONTRACT_EXECUTION_PREDICATES: Tuple[str, ...] = (
    "SAFE",
    "PLAN_STALE",
)


CONTRACT_PREDICATES: Tuple[str, ...] = (
    CONTRACT_SLOT_PREDICATES + CONTRACT_EXECUTION_PREDICATES)


CONTRACT_EVIDENCE_FIELDS: Tuple[str, ...] = (
    "progress",
    "reached",
    "child_enabled",
    "endpoint_present",
    "target_active",
    "target_known",
    "endpoint_state_present",
    "parent_ready",
    "execution_progress",
    "execution_active",
    "execution_reached",
    "execution_failed",
    "safe_known",
    "plan_stale",
)


SelfRealmIndex = 0
ExternalRealmIndex = 1
VirtualRealmIndex = 2


def ResolveGroundedReferenceWeights(
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
        * realm[..., ExternalRealmIndex]
        * presence
        * interaction
        * verification)

    uv_confidence = pst["SurfaceUVConfidence"]
    virtual_child_reference = (
        semanticReference
        * realm[..., VirtualRealmIndex]
        * presence
        * verification
        * uv_confidence)
    physical_parent_mass = (
        realm[..., SelfRealmIndex]
        + realm[..., ExternalRealmIndex])
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
        "grounded_reference": direct_reference,
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


@dataclass(frozen=True)
class ContractGroundingOutput:
    slot_predicate_names: Tuple[str, ...]
    slot_predicate_prob: torch.Tensor
    slot_predicate_known: torch.Tensor
    execution_predicate_names: Tuple[str, ...]
    execution_predicate_prob: torch.Tensor
    execution_predicate_known: torch.Tensor
    evidence_names: Tuple[str, ...]
    evidence: torch.Tensor
    slot_features: torch.Tensor
    facts: List[SymbolicFact]


class ContractNeuroSymbolicGrounder(AGICoreModule):
    def __init__(self, contractView: RobotEmbodimentContractView):
        super().__init__()
        self.contract_view = contractView

    def CommandPredicate(
        self,
        packet: BrainFeedbackPacket,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.ones_like(packet.target_active),
            packet.target_active)

    @staticmethod
    def PlanPredicate(
        packet: BrainFeedbackPacket,
        planStale: Optional[torch.Tensor],
        planStaleKnown: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(packet.joint_features.size(0))
        device = packet.joint_features.device
        if planStale is None and planStaleKnown is None:
            return (
                torch.zeros(batch_size, dtype=torch.bool, device=device),
                torch.zeros(batch_size, dtype=torch.bool, device=device))
        if planStale is None or planStaleKnown is None:
            raise ValueError(
                "plan staleness value and evidence mask must be supplied together")
        if (
            not torch.is_tensor(planStale)
            or tuple(planStale.shape) != (batch_size,)
            or planStale.dtype != torch.bool
            or planStale.device != device
            or not torch.is_tensor(planStaleKnown)
            or tuple(planStaleKnown.shape) != (batch_size,)
            or planStaleKnown.dtype != torch.bool
            or planStaleKnown.device != device
        ):
            raise ValueError(
                "plan staleness evidence must be boolean per batch on the feedback device")
        return planStale, planStaleKnown

    def BuildFacts(
        self,
        slotPredicateProb: torch.Tensor,
        slotPredicateKnown: torch.Tensor,
        executionPredicateProb: torch.Tensor,
        executionPredicateKnown: torch.Tensor,
    ) -> List[SymbolicFact]:
        facts: List[SymbolicFact] = []
        for slot_index in range(self.contract_view.end_effector_count):
            for predicate_index, predicate_name in enumerate(
                CONTRACT_SLOT_PREDICATES
            ):
                facts.append(SymbolicFact(
                    name=predicate_name,
                    args=("slot:" + str(slot_index),),
                    prob=slotPredicateProb[:, slot_index, predicate_index],
                    support=slotPredicateKnown[:, slot_index, predicate_index],
                ))
        for predicate_index, predicate_name in enumerate(
            CONTRACT_EXECUTION_PREDICATES
        ):
            argument = "execution" if predicate_name == "SAFE" else "plan"
            facts.append(SymbolicFact(
                name=predicate_name,
                args=(argument,),
                prob=executionPredicateProb[:, predicate_index],
                support=executionPredicateKnown[:, predicate_index],
            ))
        return facts

    def Ground(
        self,
        packet: BrainFeedbackPacket,
        returnExplain: bool = False,
        planStale: Optional[torch.Tensor] = None,
        planStaleKnown: Optional[torch.Tensor] = None,
    ) -> ContractGroundingOutput:
        if type(packet) is not BrainFeedbackPacket:
            raise TypeError("contract grounding requires a BrainFeedbackPacket")
        batch_size = int(packet.joint_features.size(0))
        slot_count = self.contract_view.end_effector_count
        device = packet.joint_features.device
        dtype = packet.joint_features.dtype

        parent_index = torch.tensor(
            self.contract_view.parent_index,
            dtype=torch.long,
            device=device)
        has_parent = parent_index.ge(0).unsqueeze(0).expand(
            batch_size, -1)
        parent_ready = torch.ones(
            batch_size, slot_count, dtype=torch.bool, device=device)
        parent_ready_known = ~has_parent
        child_index = has_parent[0].nonzero(as_tuple=False).flatten()
        if child_index.numel() > 0:
            selected_parent_index = parent_index.index_select(
                0, child_index)
            parent_ready[:, child_index] = packet.reached.index_select(
                1, selected_parent_index)
            parent_ready_known[:, child_index] = (
                packet.endpoint_present.index_select(
                    1, selected_parent_index))

        command_allowed, command_known = self.CommandPredicate(packet)
        plan_stale, plan_stale_known = self.PlanPredicate(
            packet,
            planStale,
            planStaleKnown)
        execution_active = packet.target_active
        active_present = execution_active.any(dim=-1)
        active_present = execution_active & packet.endpoint_present
        active_count = active_present.to(dtype=dtype).sum(
            dim=-1).clamp_min(1.0)
        execution_progress = (
            packet.progress * active_present.to(dtype=dtype)
        ).sum(dim=-1) / active_count
        execution_reached = (
            active_present
            & (packet.reached | ~execution_active).all(dim=-1))
        execution_failed = (
            execution_active & ~packet.endpoint_present).any(dim=-1)
        aggregate_safe = ~execution_failed
        aggregate_safe_known = execution_failed

        slot_predicate_value = torch.stack([
            has_parent,
            parent_ready,
            packet.child_enabled,
            packet.endpoint_present,
            command_allowed,
        ], dim=-1)
        slot_predicate_known = torch.stack([
            torch.ones_like(has_parent),
            parent_ready_known,
            torch.ones_like(packet.child_enabled),
            torch.ones_like(packet.endpoint_present),
            command_known,
        ], dim=-1)
        slot_predicate_prob = torch.where(
            slot_predicate_known,
            slot_predicate_value.to(dtype=dtype),
            torch.full_like(
                slot_predicate_value,
                0.5,
                dtype=dtype))
        execution_predicate_value = torch.stack([
            aggregate_safe,
            plan_stale,
        ], dim=-1)
        execution_predicate_known = torch.stack([
            aggregate_safe_known,
            plan_stale_known,
        ], dim=-1)
        execution_predicate_prob = torch.where(
            execution_predicate_known,
            execution_predicate_value.to(dtype=dtype),
            torch.full_like(
                execution_predicate_value,
                0.5,
                dtype=dtype))

        def Broadcast(value: torch.Tensor) -> torch.Tensor:
            return value.to(dtype=dtype).unsqueeze(-1).expand(-1, slot_count)

        endpoint_state_present = packet.endpoint_present.to(dtype=dtype)

        evidence = torch.stack([
            packet.progress,
            packet.reached.to(dtype=dtype),
            packet.child_enabled.to(dtype=dtype),
            packet.endpoint_present.to(dtype=dtype),
            packet.target_active.to(dtype=dtype),
            Broadcast(packet.target_version.ge(0)),
            endpoint_state_present,
            parent_ready.to(dtype=dtype),
            Broadcast(execution_progress),
            Broadcast(active_present),
            Broadcast(execution_reached),
            Broadcast(execution_failed),
            Broadcast(aggregate_safe_known),
            Broadcast(plan_stale),
        ], dim=-1)
        static_tokens = torch.tensor(
            self.contract_view.static_end_effector_tokens,
            dtype=dtype,
            device=device).unsqueeze(0).expand(batch_size, -1, -1)
        slot_features = torch.cat([
            static_tokens,
            evidence,
            slot_predicate_prob,
            slot_predicate_known.to(dtype=dtype),
        ], dim=-1)
        facts = (
            self.BuildFacts(
                slot_predicate_prob,
                slot_predicate_known,
                execution_predicate_prob,
                execution_predicate_known)
            if returnExplain
            else [])
        return ContractGroundingOutput(
            slot_predicate_names=CONTRACT_SLOT_PREDICATES,
            slot_predicate_prob=slot_predicate_prob,
            slot_predicate_known=slot_predicate_known,
            execution_predicate_names=CONTRACT_EXECUTION_PREDICATES,
            execution_predicate_prob=execution_predicate_prob,
            execution_predicate_known=execution_predicate_known,
            evidence_names=CONTRACT_EVIDENCE_FIELDS,
            evidence=evidence,
            slot_features=slot_features,
            facts=facts)

    def forward(
        self,
        packet: BrainFeedbackPacket,
        returnExplain: bool = False,
        planStale: Optional[torch.Tensor] = None,
        planStaleKnown: Optional[torch.Tensor] = None,
    ) -> ContractGroundingOutput:
        return self.Ground(
            packet,
            returnExplain,
            planStale,
            planStaleKnown)




class PredicateGrounder(AGICoreModule):
    def __init__(
        self,
        slotDim: int = ModuleDim.PstSlotDim,
        goalDim: int = ModuleDim.GoalShortDim,
        worldDim: int = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        embodimentStateDim: int = ModuleDim.PstSlotDim,
        poseDim: int = ModuleDim.PstPoseDim,
        hidden: int = 512,):
        super().__init__()
        self.slot_dim = int(slotDim)
        self.embodiment_state_dim = int(embodimentStateDim)
        self.pose_dim = int(poseDim)
        self.summary_context_dim = (
            int(goalDim)
            + int(worldDim)
            + int(decisionDim)
            + self.embodiment_state_dim
            + self.pose_dim
            + 6)

        self.feature_dim = (
            self.slot_dim
            + int(goalDim)
            + int(worldDim)
            + int(decisionDim)
            + self.embodiment_state_dim
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
            pst["SpatialFrame"],
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
        resolved = ResolveGroundedReferenceWeights(pst, referenced)
        return resolved["grounded_reference"].argmax(dim=1)

    def ReferencedPose(
        self,
        pst: Dict[str, torch.Tensor],
        referenceSlotIndex: torch.Tensor,
        referenceConfidence: torch.Tensor,) -> torch.Tensor:
        pose = pst["SpatialFrame"]
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
        embodimentState: torch.Tensor,
        referenced: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        referenceUncertainty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,) -> Dict[str, torch.Tensor]:
        resolved_reference = ResolveGroundedReferenceWeights(pst, referenced)
        grounded_reference = resolved_reference["grounded_reference"]
        grounded_confidence = grounded_reference.sum(dim=-1)
        reference_slot_idx = grounded_reference.argmax(dim=1)
        semantic_slot_idx = (
            pst["PerceptualPresence"] * referenced).argmax(dim=1)
        ref_pose = self.ReferencedPose(
            pst,
            reference_slot_idx,
            grounded_confidence)

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
            embodimentState,
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
            embodimentState,
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

        resolved = ResolveGroundedReferenceWeights(pst, referenced)
        direct_confidence = resolved["direct_reference"].sum(dim=-1)
        direct_physical = (
            reference_present
            & (target_realm == ExternalRealmIndex)
            & (direct_confidence > 0.5))

        virtual_slot_score = (
            semantic_weight
            * pst["RealmProb"][..., VirtualRealmIndex])
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
            (parent_realm == SelfRealmIndex)
            | (parent_realm == ExternalRealmIndex))
        parent_valid = (
            (pst["PerceptualPresence"][batch_index, parent_index] > 0.5)
            & (pst["PhysicalInteractionProb"][batch_index, parent_index] > 0.5)
            & (pst["VerificationConfidence"][batch_index, parent_index] > 0.5)
            & (pst["DisplaySurfaceProb"][batch_index, parent_index] > 0.5))
        verified_surface = (
            reference_present
            & (target_realm == VirtualRealmIndex)
            & (child_presence > 0.5)
            & (child_verification > 0.5)
            & uv_valid
            & (parent_probability > 0.5)
            & (parent_probability > no_parent_probability)
            & parent_is_physical
            & parent_valid)

        self_body_target = (
            reference_present
            & (target_realm == SelfRealmIndex))
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

        def EvidenceOf(cause: str, *predicates: str) -> None:
            for pred in predicates:
                evidence[f[cause], p[pred]] = 1.0

        def InhibitedBy(cause: str, *predicates: str) -> None:
            for pred in predicates:
                inhibition[f[cause], p[pred]] = 1.0

        EvidenceOf("occlusion", "observation_needed", "feedback_stale")
        InhibitedBy("occlusion", "observed", "localized", "feedback_fresh")
        EvidenceOf("unreachable", "localized")
        InhibitedBy("unreachable", "reachable")
        EvidenceOf("slip", "in_execution", "feedback_stale")
        InhibitedBy("slip", "attached", "feedback_fresh")
        EvidenceOf("collision", "reachable", "contactable")
        InhibitedBy("collision", "collision_free")
        EvidenceOf("misalignment", "reachable", "contactable")
        InhibitedBy("misalignment", "aligned")
        EvidenceOf("low_confidence", "observation_needed", "feedback_stale")
        InhibitedBy("low_confidence", "observed", "localized", "feedback_fresh")
        EvidenceOf("unstable_support", "supported", "movable")
        InhibitedBy("unstable_support", "attached", "collision_free")
        EvidenceOf("articulation_blocked", "articulated", "timeout_risk")
        InhibitedBy("articulation_blocked", "open", "closed", "safe_to_continue")
        EvidenceOf("containment_error", "container", "aligned")
        InhibitedBy("containment_error", "goal_satisfied")
        EvidenceOf("tool_error", "attached", "contactable", "timeout_risk")
        InhibitedBy("tool_error", "goal_satisfied", "safe_to_continue")
        EvidenceOf("lost_attachment", "in_execution", "feedback_stale")
        InhibitedBy("lost_attachment", "attached", "feedback_fresh")
        EvidenceOf("handover_failed", "attached", "reachable", "timeout_risk")
        InhibitedBy("handover_failed", "goal_satisfied", "safe_to_continue")

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

        def EvidenceOf(kind: str, *predicates: str) -> None:
            for pred in predicates:
                evidence[primitive[kind], p[pred]] = 1.0

        def InhibitedBy(kind: str, *predicates: str) -> None:
            for pred in predicates:
                inhibition[primitive[kind], p[pred]] = 1.0

        EvidenceOf("OBSERVE", "observation_needed", "feedback_stale")
        InhibitedBy("OBSERVE", "observed", "localized", "feedback_fresh")
        EvidenceOf("DISPATCH", "localized", "reachable", "collision_free")
        InhibitedBy("DISPATCH", "in_execution", "goal_satisfied", "observation_needed")
        EvidenceOf("CONTINUE", "in_execution", "feedback_fresh", "safe_to_continue")
        InhibitedBy("CONTINUE", "feedback_stale", "timeout_risk", "recovery_needed", "redispatch_needed")
        EvidenceOf("CANCEL", "in_execution", "interruptible", "feedback_stale", "timeout_risk", "recovery_needed")
        InhibitedBy("CANCEL", "safe_to_continue", "feedback_fresh")
        EvidenceOf("FAILSAFE_STOP", "timeout_risk", "recovery_needed")
        InhibitedBy("FAILSAFE_STOP", "safe_to_continue", "collision_free")
        EvidenceOf("REDISPATCH", "in_execution", "redispatch_needed", "safe_to_continue")
        InhibitedBy("REDISPATCH", "timeout_risk", "feedback_stale")

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
        subgoalFeatureDim: int = ModuleDim.DecisionLocalFeatureDim,
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


@dataclass(frozen=True)
class ContractSymbolicEncoding:
    predicate_delta: torch.Tensor
    ranker_delta: torch.Tensor
    invocation_evidence: torch.Tensor
    context: torch.Tensor


class ContractSymbolicEvidenceEncoder(AGICoreModule):
    def __init__(
        self,
        slotFeatureDim: int,
        outputFeatureDim: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.slot_feature_dim = int(slotFeatureDim)
        self.output_feature_dim = int(outputFeatureDim)
        self.hidden = int(hidden)
        if self.slot_feature_dim < 1 or self.output_feature_dim < 1:
            raise ValueError("contract symbolic feature dimensions must be positive")

        slot_input_dim = (
            self.slot_feature_dim
            + 2 * len(CONTRACT_SLOT_PREDICATES)
            + len(CONTRACT_EVIDENCE_FIELDS))
        execution_input_dim = 2 * len(CONTRACT_EXECUTION_PREDICATES)

        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(slot_input_dim),
            nn.Linear(slot_input_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),)
        self.slot_attention = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.SiLU(),
            nn.Linear(self.hidden // 2, 1),)
        self.execution_encoder = nn.Sequential(
            nn.LayerNorm(execution_input_dim),
            nn.Linear(execution_input_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),)

        summary_dim = 4 * self.hidden
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, 2 * self.hidden),
            nn.SiLU(),
            nn.Linear(2 * self.hidden, self.hidden),
            nn.LayerNorm(self.hidden),)
        self.predicate_head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, len(PREDICATES)),)
        self.ranker_head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.output_feature_dim),)
        self.invocation_head = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.SiLU(),
            nn.Linear(self.hidden // 2, 2),
            nn.Sigmoid(),)

    def Validate(
        self,
        grounding: ContractGroundingOutput,
    ) -> Tuple[int, int, torch.device, torch.dtype]:
        if type(grounding) is not ContractGroundingOutput:
            raise TypeError(
                "contract symbolic extraction requires ContractGroundingOutput")
        if grounding.slot_predicate_names != CONTRACT_SLOT_PREDICATES:
            raise ValueError("contract slot predicate semantics do not match")
        if grounding.execution_predicate_names != CONTRACT_EXECUTION_PREDICATES:
            raise ValueError("contract execution predicate semantics do not match")
        if grounding.evidence_names != CONTRACT_EVIDENCE_FIELDS:
            raise ValueError("contract continuous evidence semantics do not match")

        slot_prob = grounding.slot_predicate_prob
        slot_known = grounding.slot_predicate_known
        execution_prob = grounding.execution_predicate_prob
        execution_known = grounding.execution_predicate_known
        evidence = grounding.evidence
        slot_features = grounding.slot_features
        if not torch.is_tensor(slot_prob) or slot_prob.dim() != 3:
            raise ValueError("contract slot predicate probabilities must be rank three")
        batch_size, slot_count, predicate_count = slot_prob.shape
        if batch_size < 1 or slot_count < 1:
            raise ValueError("contract grounding must contain a non-empty batch and slots")
        if predicate_count != len(CONTRACT_SLOT_PREDICATES):
            raise ValueError("contract slot predicate width does not match")
        if tuple(slot_known.shape) != tuple(slot_prob.shape) or slot_known.dtype != torch.bool:
            raise ValueError("contract slot predicate known mask does not match")
        if tuple(evidence.shape) != (
            batch_size,
            slot_count,
            len(CONTRACT_EVIDENCE_FIELDS),
        ):
            raise ValueError("contract continuous evidence shape does not match")
        if tuple(slot_features.shape) != (
            batch_size,
            slot_count,
            self.slot_feature_dim,
        ):
            raise ValueError("contract slot feature shape does not match the model")
        if tuple(execution_prob.shape) != (
            batch_size,
            len(CONTRACT_EXECUTION_PREDICATES),
        ):
            raise ValueError("contract execution predicate shape does not match")
        if (
            tuple(execution_known.shape) != tuple(execution_prob.shape)
            or execution_known.dtype != torch.bool
        ):
            raise ValueError("contract execution predicate known mask does not match")

        floating = (slot_prob, execution_prob, evidence, slot_features)
        if any(not value.is_floating_point() for value in floating):
            raise ValueError("contract symbolic values must be floating point")
        device = slot_prob.device
        dtype = slot_prob.dtype
        if any(value.device != device or value.dtype != dtype for value in floating):
            raise ValueError("contract symbolic values must share device and dtype")
        if slot_known.device != device or execution_known.device != device:
            raise ValueError("contract symbolic masks must share the value device")
        if any(not bool(torch.isfinite(value).all().item()) for value in floating):
            raise ValueError("contract symbolic values must be finite")
        if bool(((slot_prob < 0.0) | (slot_prob > 1.0)).any().item()):
            raise ValueError("contract slot predicate probabilities must be normalized")
        if bool(((execution_prob < 0.0) | (execution_prob > 1.0)).any().item()):
            raise ValueError(
                "contract execution predicate probabilities must be normalized")
        return int(batch_size), int(slot_count), device, dtype

    def Encode(
        self,
        grounding: ContractGroundingOutput,
    ) -> ContractSymbolicEncoding:
        self.Validate(grounding)
        dtype = grounding.slot_predicate_prob.dtype
        slot_input = torch.cat([
            grounding.slot_features,
            grounding.slot_predicate_prob,
            grounding.slot_predicate_known.to(dtype=dtype),
            grounding.evidence,
        ], dim=-1)
        slot_embedding = self.slot_encoder(slot_input)
        attention = F.softmax(
            self.slot_attention(slot_embedding).squeeze(-1),
            dim=-1)
        attended = (
            slot_embedding * attention.unsqueeze(-1)).sum(dim=1)
        mean_summary = slot_embedding.mean(dim=1)
        max_summary = slot_embedding.amax(dim=1)
        execution_input = torch.cat([
            grounding.execution_predicate_prob,
            grounding.execution_predicate_known.to(dtype=dtype),
        ], dim=-1)
        execution_summary = self.execution_encoder(execution_input)
        context = self.context_encoder(torch.cat([
            attended,
            mean_summary,
            max_summary,
            execution_summary,
        ], dim=-1))
        return ContractSymbolicEncoding(
            predicate_delta=self.predicate_head(context),
            ranker_delta=self.ranker_head(context),
            invocation_evidence=self.invocation_head(context),
            context=context)

    def forward(
        self,
        grounding: ContractGroundingOutput,
    ) -> ContractSymbolicEncoding:
        return self.Encode(grounding)


class NeuroSymbolicExtractor(AGICoreModule):
    def __init__(
        self,
        *,
        contractSlotFeatureDim: int,
        slotDim: int = ModuleDim.PstSlotDim,
        goalDim: int = ModuleDim.GoalShortDim,
        worldDim: int = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        poseDim: int = ModuleDim.PstPoseDim,
        embodimentStateDim: int = ModuleDim.PstSlotDim,
        controlFeedbackDim: int = ModuleDim.DecisionLocalFeatureDim,
        planDim: int = 256,
        constraintTokenDim: int = 128,
        constraintTokens: int = 8,):
        super().__init__()
        if type(contractSlotFeatureDim) is not int or contractSlotFeatureDim < 1:
            raise ValueError("contract slot feature dimension must be positive")
        self.pose_dim = int(poseDim)
        self.embodiment_state_dim = int(embodimentStateDim)
        self.plan_dim = int(planDim)
        self.constraint_token_dim = int(constraintTokenDim)
        self.constraint_tokens = int(constraintTokens)
        self.control_feedback_dim = int(controlFeedbackDim)
        if self.control_feedback_dim < 1:
            raise ValueError("control feedback feature dimension must be positive")
        self.predicate_grounder = PredicateGrounder(
            slotDim=slotDim,
            goalDim=goalDim,
            worldDim=worldDim,
            decisionDim=decisionDim,
            embodimentStateDim=self.embodiment_state_dim,
            poseDim=self.pose_dim,)

        base_feature_dim = (
            self.predicate_grounder.feature_dim
            + len(PREDICATES)
            + self.control_feedback_dim)
        self.base_feature_dim = int(base_feature_dim)

        self.contract_symbolic_encoder = ContractSymbolicEvidenceEncoder(
            slotFeatureDim=int(contractSlotFeatureDim),
            outputFeatureDim=self.base_feature_dim)

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
            + self.embodiment_state_dim
            + self.control_feedback_dim)

        self.subgoal_feature_head = nn.Sequential(
            nn.LayerNorm(sampler_in),
            nn.Linear(sampler_in, 256),
            nn.SiLU(),
            nn.Linear(256, self.embodiment_state_dim),
            nn.LayerNorm(self.embodiment_state_dim),)

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
            subgoalFeatureDim=self.embodiment_state_dim,
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

    def BuildContractFacts(
        self,
        grounding: ContractGroundingOutput,
    ) -> List[SymbolicFact]:
        _, slot_count, _, _ = self.contract_symbolic_encoder.Validate(grounding)
        facts: List[SymbolicFact] = []
        for slot_index in range(slot_count):
            for predicate_index, predicate_name in enumerate(
                grounding.slot_predicate_names
            ):
                facts.append(SymbolicFact(
                    name=predicate_name,
                    args=("slot:" + str(slot_index),),
                    prob=grounding.slot_predicate_prob[
                        :, slot_index, predicate_index],
                    support=grounding.slot_predicate_known[
                        :, slot_index, predicate_index],
                ))
        for predicate_index, predicate_name in enumerate(
            grounding.execution_predicate_names
        ):
            facts.append(SymbolicFact(
                name=predicate_name,
                args=("execution",),
                prob=grounding.execution_predicate_prob[:, predicate_index],
                support=grounding.execution_predicate_known[:, predicate_index],
            ))
        return facts

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
        embodimentState: torch.Tensor,
        controlFeedbackState: torch.Tensor,) -> torch.Tensor:
        ref_pose = referencedPose
        return self.subgoal_feature_head(torch.cat([
            planLatent,
            ref_pose,
            embodimentState,
            controlFeedbackState,], dim=-1))

    def ForwardEncoded(
        self,
        pst: Dict[str, torch.Tensor],
        observedPst: Dict[str, torch.Tensor],
        goalEmbed: torch.Tensor,
        worldBelief: torch.Tensor,
        decisionBelief: torch.Tensor,
        embodimentState: torch.Tensor,
        controlFeedbackState: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        referenceUncertainty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenced: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        contractGrounding: ContractGroundingOutput,
        returnExplain: bool = False,) -> NeuroSymbolicOutput:
        batch_size = int(goalEmbed.size(0))
        expected_physical = (batch_size, self.embodiment_state_dim)
        expected_control = (batch_size, self.control_feedback_dim)
        if (
            not torch.is_tensor(embodimentState)
            or not embodimentState.is_floating_point()
            or tuple(embodimentState.shape) != expected_physical
        ):
            raise ValueError("encoded embodiment state does not match the cognitive model")
        if (
            not torch.is_tensor(controlFeedbackState)
            or not controlFeedbackState.is_floating_point()
            or tuple(controlFeedbackState.shape) != expected_control
        ):
            raise ValueError("encoded control feedback does not match the cognitive model")
        if (
            embodimentState.device != goalEmbed.device
            or controlFeedbackState.device != goalEmbed.device
            or embodimentState.dtype != goalEmbed.dtype
            or controlFeedbackState.dtype != goalEmbed.dtype
        ):
            raise ValueError(
                "encoded embodiment, control, and cognitive features must share device and dtype")
        if (
            not bool(torch.isfinite(embodimentState).all().item())
            or not bool(torch.isfinite(controlFeedbackState).all().item())
        ):
            raise ValueError("encoded embodiment and control features must be finite")

        contract_encoding = self.contract_symbolic_encoder.Encode(
            contractGrounding)
        if contract_encoding.context.size(0) != batch_size:
            raise ValueError("contract grounding batch does not match cognitive inputs")
        if (
            contract_encoding.context.device != goalEmbed.device
            or contract_encoding.context.dtype != goalEmbed.dtype
        ):
            raise ValueError(
                "contract grounding and cognitive features must share device and dtype")

        embodiment_state = embodimentState
        control_feedback_state = controlFeedbackState
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
            embodimentState=embodiment_state,
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

        predicate_logits = predicate_logits + contract_encoding.predicate_delta

        predicate_prob = torch.sigmoid(predicate_logits)

        ranker_in = torch.cat([
            grounded["features"],
            predicate_prob,
            control_feedback_state], dim=-1)
        ranker_in = ranker_in + contract_encoding.ranker_delta
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
            embodiment_state,
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

        control_error_evidence = contract_encoding.invocation_evidence
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
            facts.extend(self.BuildContractFacts(contractGrounding))
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

    def ForwardContract(
        self,
        contractGrounding: ContractGroundingOutput,
        pst: Dict[str, torch.Tensor],
        observedPst: Dict[str, torch.Tensor],
        goalEmbed: torch.Tensor,
        worldBelief: torch.Tensor,
        decisionBelief: torch.Tensor,
        embodimentState: torch.Tensor,
        controlState: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        referenceUncertainty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenced: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        returnExplain: bool = False,
    ) -> NeuroSymbolicOutput:
        return self.ForwardEncoded(
            pst=pst,
            observedPst=observedPst,
            goalEmbed=goalEmbed,
            worldBelief=worldBelief,
            decisionBelief=decisionBelief,
            embodimentState=embodimentState,
            controlFeedbackState=controlState,
            uncertainty=uncertainty,
            novelty=novelty,
            recentFailure=recentFailure,
            referenceUncertainty=referenceUncertainty,
            satisfactionProb=satisfactionProb,
            referenced=referenced,
            referenceConfidence=referenceConfidence,
            noSlotProb=noSlotProb,
            temporalContextFeat=temporalContextFeat,
            contractGrounding=contractGrounding,
            returnExplain=returnExplain)

    def SelectContractGroundingRows(
        self,
        grounding: ContractGroundingOutput,
        rowIndex: torch.Tensor,
        fullBatchSize: int,
    ) -> ContractGroundingOutput:
        if type(grounding) is not ContractGroundingOutput:
            raise TypeError("contract grounding rows require ContractGroundingOutput")

        def Select(value: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(value) or value.dim() < 1:
                raise ValueError("contract grounding tensors must have a batch dimension")
            if int(value.size(0)) != fullBatchSize:
                raise ValueError("contract grounding tensors must match fullBatchSize")
            if value.device != rowIndex.device:
                raise ValueError("contract grounding tensors must share the rowIndex device")
            return value.index_select(0, rowIndex)

        facts = [
            SymbolicFact(
                name=fact.name,
                args=fact.args,
                prob=Select(fact.prob),
                support=(
                    None
                    if fact.support is None
                    else Select(fact.support)))
            for fact in grounding.facts]
        return ContractGroundingOutput(
            slot_predicate_names=grounding.slot_predicate_names,
            slot_predicate_prob=Select(grounding.slot_predicate_prob),
            slot_predicate_known=Select(grounding.slot_predicate_known),
            execution_predicate_names=grounding.execution_predicate_names,
            execution_predicate_prob=Select(
                grounding.execution_predicate_prob),
            execution_predicate_known=Select(
                grounding.execution_predicate_known),
            evidence_names=grounding.evidence_names,
            evidence=Select(grounding.evidence),
            slot_features=Select(grounding.slot_features),
            facts=facts)

    def SelectFullBatchRows(
        self,
        value: Any,
        rowIndex: torch.Tensor,
        fullBatchSize: int,
        valueName: str,
    ) -> Any:
        if torch.is_tensor(value):
            if value.dim() < 1 or int(value.size(0)) != fullBatchSize:
                raise ValueError(f"{valueName} must match fullBatchSize")
            if value.device != rowIndex.device:
                raise ValueError(f"{valueName} must share the rowIndex device")
            return value.index_select(0, rowIndex)
        if type(value) is ContractGroundingOutput:
            return self.SelectContractGroundingRows(
                value,
                rowIndex,
                fullBatchSize)
        if isinstance(value, dict):
            return {
                name: self.SelectFullBatchRows(
                    item,
                    rowIndex,
                    fullBatchSize,
                    f"{valueName}.{name}")
                for name, item in value.items()}
        if isinstance(value, bool) or value is None:
            return value
        raise TypeError(f"{valueName} contains an unsupported full-batch value")

    def EnsurePlanBatch(
        self,
        fullBatchSize: int,
        reference: torch.Tensor,
    ) -> None:
        state = (
            self.last_operator,
            self.last_operator_prob,
            self.last_invoke,
            self.last_reference_summary)
        if all(value.numel() == 0 for value in state):
            self.last_operator = torch.full(
                (fullBatchSize,),
                -1,
                dtype=torch.long,
                device=reference.device)
            self.last_operator_prob = reference.new_zeros(
                fullBatchSize,
                len(OPERATORS))
            self.last_invoke = reference.new_zeros(fullBatchSize)
            self.last_reference_summary = reference.new_zeros(
                fullBatchSize,
                self.predicate_grounder.slot_dim)
            return
        expectedShapes = (
            (fullBatchSize,),
            (fullBatchSize, len(OPERATORS)),
            (fullBatchSize,),
            (fullBatchSize, self.predicate_grounder.slot_dim))
        if any(
            tuple(value.shape) != shape
            for value, shape in zip(state, expectedShapes)
        ):
            raise ValueError("neuro-symbolic plan state does not match fullBatchSize")
        if self.last_operator.device != reference.device:
            raise ValueError("neuro-symbolic plan state device does not match inputs")
        for value in state[1:]:
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("neuro-symbolic plan state type does not match inputs")

    def ForwardContractRows(
        self,
        rowIndex: torch.Tensor,
        fullBatchSize: int,
        **kwargs: Any,
    ) -> NeuroSymbolicOutput:
        if type(fullBatchSize) is not int or fullBatchSize < 1:
            raise ValueError("fullBatchSize must be a positive integer")
        if "goalEmbed" not in kwargs:
            raise ValueError("goalEmbed is required for row execution")
        reference = kwargs["goalEmbed"]
        if (
            not torch.is_tensor(rowIndex)
            or rowIndex.dim() != 1
            or rowIndex.dtype != torch.long
            or not torch.is_tensor(reference)
            or rowIndex.device != reference.device
        ):
            raise ValueError("rowIndex must be a one-dimensional long tensor on the input device")
        if rowIndex.numel() < 1:
            raise ValueError("rowIndex must select at least one row")
        if bool(((rowIndex < 0) | (rowIndex >= fullBatchSize)).any().item()):
            raise IndexError("rowIndex contains an out-of-range neuro-symbolic row")
        if int(torch.unique(rowIndex).numel()) != int(rowIndex.numel()):
            raise ValueError("rowIndex must not contain duplicate rows")
        selected = {
            name: self.SelectFullBatchRows(
                value,
                rowIndex,
                fullBatchSize,
                name)
            for name, value in kwargs.items()}
        fullRows = torch.arange(
            fullBatchSize,
            dtype=torch.long,
            device=rowIndex.device)
        if torch.equal(rowIndex, fullRows):
            return self.ForwardContract(**kwargs)

        originalState = self.ExportPlanState()
        try:
            self.EnsurePlanBatch(fullBatchSize, reference)
            fullState = self.ExportPlanState()
            self.last_operator = fullState["last_operator"].index_select(
                0,
                rowIndex)
            self.last_operator_prob = fullState[
                "last_operator_prob"].index_select(
                    0,
                    rowIndex)
            self.last_invoke = fullState["last_invoke"].index_select(
                0,
                rowIndex)
            self.last_reference_summary = fullState[
                "last_reference_summary"].index_select(0, rowIndex)
            output = self.ForwardContract(**selected)
            selectedState = self.ExportPlanState()
            self.last_operator = fullState["last_operator"].index_copy(
                0,
                rowIndex,
                selectedState["last_operator"])
            self.last_operator_prob = fullState[
                "last_operator_prob"].index_copy(
                    0,
                    rowIndex,
                    selectedState["last_operator_prob"])
            self.last_invoke = fullState["last_invoke"].index_copy(
                0,
                rowIndex,
                selectedState["last_invoke"])
            self.last_reference_summary = fullState[
                "last_reference_summary"].index_copy(
                    0,
                    rowIndex,
                    selectedState["last_reference_summary"])
        except BaseException:
            self.ImportPlanState(originalState)
            raise
        return output

    def forward(
        self,
        contractGrounding: ContractGroundingOutput,
        pst: Dict[str, torch.Tensor],
        observedPst: Dict[str, torch.Tensor],
        goalEmbed: torch.Tensor,
        worldBelief: torch.Tensor,
        decisionBelief: torch.Tensor,
        embodimentState: torch.Tensor,
        controlState: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        referenceUncertainty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenced: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        returnExplain: bool = False,
    ) -> NeuroSymbolicOutput:
        return self.ForwardContract(
            contractGrounding=contractGrounding,
            pst=pst,
            observedPst=observedPst,
            goalEmbed=goalEmbed,
            worldBelief=worldBelief,
            decisionBelief=decisionBelief,
            embodimentState=embodimentState,
            controlState=controlState,
            uncertainty=uncertainty,
            novelty=novelty,
            recentFailure=recentFailure,
            referenceUncertainty=referenceUncertainty,
            satisfactionProb=satisfactionProb,
            referenced=referenced,
            referenceConfidence=referenceConfidence,
            noSlotProb=noSlotProb,
            temporalContextFeat=temporalContextFeat,
            returnExplain=returnExplain)
