from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunctionTools import AGICoreModule, BuildReferenceWeights, BuildReferenceScaleContext
from ModuleMessagerManager import ModuleDim


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


class PredicateGrounder(AGICoreModule):
    def __init__(
        self,
        slotDim: int = ModuleDim.PstSlotDim,
        goalDim: int = ModuleDim.GoalShortDim,
        worldDim: int = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        endpointPoseFeatDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        poseDim: int = ModuleDim.PstPoseDim,
        hidden: int = 512,):
        super().__init__()
        self.slot_dim = int(slotDim)
        self.endpoint_pose_feat_dim = int(endpointPoseFeatDim)
        self.pose_dim = int(poseDim)
        self.summary_context_dim = (
            int(goalDim)
            + int(worldDim)
            + int(decisionDim)
            + self.endpoint_pose_feat_dim
            + self.pose_dim
            + 6)

        self.feature_dim = (
            self.slot_dim
            + int(goalDim)
            + int(worldDim)
            + int(decisionDim)
            + self.endpoint_pose_feat_dim
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
        m = pst["MphysRaw"]
        current_step = pst["Step"].view(-1, 1).float()
        reference_weights = BuildReferenceWeights(
            pst,
            current_step,
            memoryScale=memoryScale,
            memoryDecayHorizon=32.0)
        observed_weight = reference_weights.observed_weight
        memory_weight = reference_weights.memory_weight
        memory_recency = reference_weights.memory_recency
        slot_context_weight = reference_weights.slot_weight
        target_weight = m * referenced

        slot_input = torch.cat([
            pst["SlotState"],
            pst["PoseWorld"],
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
        score = pst["MphysRaw"] * referenced
        return score.argmax(dim=1)

    def ReferencedPose(
        self,
        pst: Dict[str, torch.Tensor],
        referenceSlotIndex: torch.Tensor,
        referenceConfidence: torch.Tensor,) -> torch.Tensor:
        pose = pst["PoseWorld"]
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
        endpointPoseFeat: torch.Tensor,
        referenced: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        intentNovelty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,) -> Dict[str, torch.Tensor]:
        reference_slot_idx = self.ReferenceSlotIndex(pst, referenced)
        ref_pose = self.ReferencedPose(pst, reference_slot_idx, referenceConfidence)

        scalar = torch.stack([
            uncertainty,
            novelty,
            recentFailure,
            intentNovelty,
            satisfactionProb.view(-1),
            noSlotProb,], dim=-1)

        summary_context = torch.cat([
            goalEmbed,
            worldBelief,
            decisionBelief,
            endpointPoseFeat,
            ref_pose,
            scalar,], dim=-1)
        reference_context = BuildReferenceScaleContext(
            observedPst,
            self.summary_query(summary_context))
        memory_scale = torch.sigmoid(self.reference_memory_scale_head(torch.cat([summary_context, reference_context], dim=-1)))

        slot_summary = self.SlotSummary(
            pst,
            summary_context,
            referenced,
            reference_slot_idx,
            referenceConfidence,
            memory_scale)

        x = torch.cat([
            slot_summary,
            goalEmbed,
            worldBelief,
            decisionBelief,
            endpointPoseFeat,
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
        slotDim: int = ModuleDim.PstSlotDim,
        goalDim: int = ModuleDim.GoalShortDim,
        worldDim: int = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        poseDim: int = ModuleDim.PstPoseDim,
        endpointPoseFeatDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        planDim: int = 256,
        constraintTokenDim: int = 128,
        constraintTokens: int = 8,):
        super().__init__()
        self.pose_dim = int(poseDim)
        self.endpoint_pose_feat_dim = int(endpointPoseFeatDim)
        self.plan_dim = int(planDim)
        self.constraint_token_dim = int(constraintTokenDim)
        self.constraint_tokens = int(constraintTokens)

        self.predicate_grounder = PredicateGrounder(
            slotDim=slotDim,
            goalDim=goalDim,
            worldDim=worldDim,
            decisionDim=decisionDim,
            endpointPoseFeatDim=self.endpoint_pose_feat_dim,
            poseDim=self.pose_dim,)

        base_feature_dim = self.predicate_grounder.feature_dim + len(PREDICATES)

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
            nn.LayerNorm(6),
            nn.Linear(6, 32),
            nn.SiLU(),
            nn.Linear(32, 1),)

        sampler_in = self.plan_dim + self.pose_dim + self.endpoint_pose_feat_dim

        self.subgoal_feature_head = nn.Sequential(
            nn.LayerNorm(sampler_in),
            nn.Linear(sampler_in, 256),
            nn.SiLU(),
            nn.Linear(256, self.endpoint_pose_feat_dim),
            nn.LayerNorm(self.endpoint_pose_feat_dim),)

        self.constraint_head = nn.Sequential(
            nn.LayerNorm(self.plan_dim + len(PREDICATES) + len(OPERATORS)),
            nn.Linear(self.plan_dim + len(PREDICATES) + len(OPERATORS), 512),
            nn.SiLU(),
            nn.Linear(512, self.constraint_tokens * self.constraint_token_dim),)

        self.symbolic_mixer = SymbolicFeatureMixer(
            planDim=self.plan_dim,
            subgoalFeatureDim=self.endpoint_pose_feat_dim,
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
        endpointPoseFeat: torch.Tensor,) -> torch.Tensor:
        ref_pose = referencedPose
        return self.subgoal_feature_head(torch.cat([
            planLatent,
            ref_pose,
            endpointPoseFeat,], dim=-1))

    def forward(
        self,
        pst: Dict[str, torch.Tensor],
        observedPst: Dict[str, torch.Tensor],
        goalEmbed: torch.Tensor,
        worldBelief: torch.Tensor,
        decisionBelief: torch.Tensor,
        endpointPoseFeat: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        recentFailure: torch.Tensor,
        intentNovelty: torch.Tensor,
        satisfactionProb: torch.Tensor,
        referenced: torch.Tensor,
        referenceConfidence: torch.Tensor,
        noSlotProb: torch.Tensor,
        temporalContextFeat: torch.Tensor,
        returnExplain: bool = False,) -> NeuroSymbolicOutput:
        scalar = torch.stack([
            uncertainty,
            novelty,
            recentFailure,
            intentNovelty,
            satisfactionProb.view(-1),
            noSlotProb,], dim=-1)

        grounded = self.predicate_grounder(
            pst=pst,
            observedPst=observedPst,
            goalEmbed=goalEmbed,
            worldBelief=worldBelief,
            decisionBelief=decisionBelief,
            endpointPoseFeat=endpointPoseFeat,
            referenced=referenced,
            uncertainty=uncertainty,
            novelty=novelty,
            recentFailure=recentFailure,
            intentNovelty=intentNovelty,
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

        ranker_in = torch.cat([grounded["features"], predicate_prob], dim=-1)
        ranked = self.plan_ranker(ranker_in, temporalContextFeat)

        goal_predicate_need = torch.sigmoid(self.goal_predicate_head(ranker_in))
        operator_scores = self.operator_library.Scores(predicate_prob, goal_predicate_need)
        operator_logits = ranked["operator_logits"] + operator_scores["symbolic_score"]
        plan_latent = self.plan_ranker.RefinePlan(ranked["plan_seed"], operator_logits)
        subgoal_feature = self.BuildSubgoalFeature(plan_latent, grounded["referenced_pose"], endpointPoseFeat)
        operator_prob = F.softmax(operator_logits, dim=-1)

        constraint_tokens = self.constraint_head(torch.cat([
            plan_latent,
            predicate_prob,
            operator_prob,], dim=-1)).view(plan_latent.size(0), self.constraint_tokens, self.constraint_token_dim)

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

        invoke_mask = torch.sigmoid(self.invoke_head(scalar)).squeeze(-1)

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

    def AssertFinite(self, value: torch.Tensor, name: str) -> None:
        assert torch.isfinite(value).all(), f"{name} contains non-finite values"

    def MakePst(self, B: int = 2, K: int = 4) -> Dict[str, torch.Tensor]:
        pose = torch.randn(B, K, ModuleDim.PstPoseDim, device=self.device) * 0.1
        pose[..., 6] = 1.0
        mask = torch.ones(B, K, device=self.device)
        return {
            "MphysRaw": mask.clone(),
            "Observed": torch.ones(B, K, device=self.device, dtype=torch.bool),
            "LastSeen": torch.arange(K, device=self.device, dtype=torch.float32).unsqueeze(0).expand(B, -1),
            "Step": torch.full((B,), K, device=self.device, dtype=torch.long),
            "SlotPresence": mask.clone(),
            "ObservedSlotMask": mask.clone(),
            "SlotState": torch.randn(B, K, ModuleDim.PstSlotDim, device=self.device),
            "PoseCamera": pose,
            "PoseWorld": pose,}

    def MakeExtractorInputs(self, B: int = 2, K: int = 4) -> Dict[str, torch.Tensor]:
        referenced = F.softmax(torch.randn(B, K, device=self.device), dim=-1)
        return {
            "pst": self.MakePst(B=B, K=K),
            "observedPst": self.MakePst(B=B, K=K),
            "goalEmbed": torch.randn(B, ModuleDim.GoalShortDim, device=self.device),
            "worldBelief": torch.randn(
                B,
                ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState,
                device=self.device),
            "decisionBelief": torch.randn(B, ModuleDim.DecisionBeliefDim, device=self.device),
            "endpointPoseFeat": torch.randn(B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
            "uncertainty": torch.rand(B, device=self.device),
            "novelty": torch.rand(B, device=self.device),
            "recentFailure": torch.rand(B, device=self.device),
            "intentNovelty": torch.rand(B, device=self.device),
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
                endpointPoseFeat=inputs["endpointPoseFeat"],
                referenced=inputs["referenced"],
                uncertainty=inputs["uncertainty"],
                novelty=inputs["novelty"],
                recentFailure=inputs["recentFailure"],
                intentNovelty=inputs["intentNovelty"],
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
            model = NeuroSymbolicExtractor().to(self.device)
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

    def TestNeuroSymbolicContinuityAndReset(self) -> bool:
        try:
            B, K = 2, 4
            model = NeuroSymbolicExtractor().to(self.device)
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
            model = NeuroSymbolicExtractor().to(self.device)
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
            model = NeuroSymbolicExtractor().to(self.device)
            model.eval()
            inputs = self.MakeExtractorInputs(B=B, K=K)
            with torch.no_grad():
                _ = model(**inputs, returnExplain=False)
                state = model.ExportPlanState()
            restored = NeuroSymbolicExtractor().to(self.device)
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
            model = NeuroSymbolicExtractor().to(self.device)
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

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "PredicateGrounderShapes": self.TestPredicateGrounderShapes(),
            "OperatorLibraryScores": self.TestOperatorLibraryScores(),
            "PlanRankerShapes": self.TestPlanRankerShapes(),
            "FailureExplainerShapes": self.TestFailureExplainerShapes(),
            "TemporalSymbolicHeadShapes": self.TestTemporalSymbolicHeadShapes(),
            "SymbolicFeatureMixerShapes": self.TestSymbolicFeatureMixerShapes(),
            "NeuroSymbolicExtractorForwardAndExplainSwitch": self.TestNeuroSymbolicExtractorForwardAndExplainSwitch(),
            "NeuroSymbolicContinuityAndReset": self.TestNeuroSymbolicContinuityAndReset(),
            "ReferenceDriftRespondsToBindingChange": self.TestReferenceDriftRespondsToBindingChange(),
            "PlanStateExportImportRoundTrip": self.TestPlanStateExportImportRoundTrip(),
            "PlanStateStrictSchema": self.TestPlanStateStrictSchema(),}
        passed = sum(1 for value in results.values() if value)
        print(f"\n[NeuroSymbolicModule Tests] {passed}/{len(results)} passed.")
        return results
