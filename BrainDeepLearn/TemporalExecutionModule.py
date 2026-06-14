from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DecisionDecoupler import MotionCommand
from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim


OBSERVE = 0
DISPATCH = 1
CONTINUE = 2
CANCEL = 3
FAILSAFE_STOP = 4
REDISPATCH = 5


@dataclass
class TemporalContext:
    feat: torch.Tensor
    active_mask: torch.Tensor
    action_age: torch.Tensor
    feedback_age: torch.Tensor
    no_slot_prob: torch.Tensor
    reference_confidence: torch.Tensor
    satisfaction_prob: torch.Tensor
    safety_risk: torch.Tensor
    interrupt_risk: torch.Tensor
    observation_freshness: torch.Tensor
    can_interrupt: torch.Tensor
    hard_stop: torch.Tensor
    planner_progress: torch.Tensor
    planner_tracking_error: torch.Tensor
    planner_executing: torch.Tensor
    planner_reached: torch.Tensor
    planner_failed: torch.Tensor
    planner_canceled: torch.Tensor


@dataclass
class TemporalDecisionEnvelope:
    kind_logits: torch.Tensor
    kind_id: torch.Tensor
    kind_names: Tuple[str, ...]
    action_id: torch.Tensor
    action_epoch: torch.Tensor
    reason_logits: torch.Tensor
    duration_ms: torch.Tensor
    soft_timeout_ms: torch.Tensor
    hard_timeout_ms: torch.Tensor
    publish_motion_command: torch.Tensor
    reuse_active_motion_command: torch.Tensor
    publish_stop_command: torch.Tensor
    publish_hold_command: torch.Tensor
    same_operator: torch.Tensor
    operator_changed: torch.Tensor
    invoke_delta: torch.Tensor
    reference_drift: torch.Tensor
    invoke_drift: torch.Tensor
    motion_command: MotionCommand


@dataclass
class TemporalExecutionState:
    active_mask: torch.Tensor
    action_age: torch.Tensor
    feedback_age: torch.Tensor
    action_epoch: torch.Tensor
    active_kind: torch.Tensor
    active_feedback_embed: torch.Tensor


class TemporalExecutionGateExtractor(AGICoreModule):
    def __init__(
        self,
        primitiveCount: int = ModuleDim.TemporalPrimitiveCount,
        contextDim: int = ModuleDim.TemporalContextDim,
        reasonDim: int = ModuleDim.TemporalReasonDim,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        actionDim: int = ModuleDim.DecisionActionDim,
        poseDim: int = ModuleDim.DecisionEndpointPoseDim,
        ageNorm: float = 128.0,):
        super().__init__()
        self.primitive_count = int(primitiveCount)
        self.context_dim = int(contextDim)
        self.reason_dim = int(reasonDim)
        self.endpoint_count = int(endpointCount)
        self.action_dim = int(actionDim)
        self.pose_dim = int(poseDim)
        self.age_norm = float(ageNorm)
 
        self.rule_gain_max = 16.0
        rule_gain_init = torch.tensor([2.0, 2.0, 2.0, 3.0, 8.0, 3.0])
        self.rule_gain_raw = nn.Parameter((rule_gain_init / (self.rule_gain_max - rule_gain_init)).log())
 
        self.inactive_penalty_raw = nn.Parameter(torch.tensor([4.0, 4.0, 4.0]).expm1().log())
        self.continue_base = nn.Parameter(torch.tensor(3.0))
        continue_penalty_init = torch.tensor([
            1.4,  # invoke_delta
            1.6,  # reference_drift
            2.0,  # planner_tracking_error
            2.0,  # safety_risk
            1.6,  # satisfaction_prob
            4.0,  # planner_failed
            4.0,  # planner_canceled
            2.0,  # planner_reached
            1.2,  # stale observation
            1.5,  # interrupt_risk
        ])
        self.continue_penalty_raw = nn.Parameter(continue_penalty_init.expm1().log())
 
        self.register_buffer("drift_threshold", torch.tensor(4.0), persistent=True)
        
        self.context_refiner = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.primitive_count),)
        
        self.reason_head = nn.Sequential(
            nn.LayerNorm(self.context_dim + self.primitive_count),
            nn.Linear(self.context_dim + self.primitive_count, 64),
            nn.SiLU(),
            nn.Linear(64, self.reason_dim),)

    @torch.no_grad()
    def CalibrateDriftThreshold(self, driftSamples: torch.Tensor, quantile: float = 0.95):
        self.drift_threshold.copy_(torch.quantile(driftSamples.view(-1), quantile))

    def BuildContext(
        self,
        activeMask: torch.Tensor,
        actionAge: torch.Tensor,
        feedbackAge: torch.Tensor,
        noSlotProb: torch.Tensor,
        referenceConfidence: torch.Tensor,
        satisfactionProb: torch.Tensor,
        safetyRisk: torch.Tensor,
        interruptRisk: torch.Tensor,
        observationFreshness: torch.Tensor,
        canInterrupt: torch.Tensor,
        hardStop: torch.Tensor,
        plannerProgress: torch.Tensor,
        plannerTrackingError: torch.Tensor,
        plannerExecuting: torch.Tensor,
        plannerReached: torch.Tensor,
        plannerFailed: torch.Tensor,
        plannerCanceled: torch.Tensor,) -> TemporalContext:
        active = activeMask.view(-1)
        age = actionAge.view(-1)
        feedback_age = feedbackAge.view(-1)
        no_slot = noSlotProb.view(-1)
        ref_conf = referenceConfidence.view(-1)
        satisfied = satisfactionProb.view(-1)
        safety = safetyRisk.view(-1)
        interrupt = interruptRisk.view(-1)
        freshness = observationFreshness.view(-1)
        can_interrupt = canInterrupt.view(-1)
        hard_stop = hardStop.view(-1)
        planner_progress = plannerProgress.view(-1)
        planner_tracking_error = plannerTrackingError.view(-1)
        planner_executing = plannerExecuting.view(-1)
        planner_reached = plannerReached.view(-1)
        planner_failed = plannerFailed.view(-1)
        planner_canceled = plannerCanceled.view(-1)

        feat = torch.stack([
            active,
            age / self.age_norm,
            feedback_age / self.age_norm,
            no_slot,
            ref_conf,
            satisfied,
            safety,
            interrupt,
            freshness,
            can_interrupt,
            hard_stop,
            active * (1.0 - satisfied),
            no_slot * (1.0 - active),
            interrupt * active,
            safety * can_interrupt,
            ref_conf * freshness,
            planner_progress,
            planner_tracking_error,
            planner_executing,
            planner_reached,
            planner_failed,
            planner_canceled,], dim=-1)
        
        return TemporalContext(
            feat=feat,
            active_mask=active,
            action_age=age,
            feedback_age=feedback_age,
            no_slot_prob=no_slot,
            reference_confidence=ref_conf,
            satisfaction_prob=satisfied,
            safety_risk=safety,
            interrupt_risk=interrupt,
            observation_freshness=freshness,
            can_interrupt=can_interrupt,
            hard_stop=hard_stop,
            planner_progress=planner_progress,
            planner_tracking_error=planner_tracking_error,
            planner_executing=planner_executing,
            planner_reached=planner_reached,
            planner_failed=planner_failed,
            planner_canceled=planner_canceled,)

    def HoldCommand(self, endpointPose: torch.Tensor, template: MotionCommand) -> MotionCommand:
        B = endpointPose.size(0)
        decision_tensor = endpointPose.new_zeros(B, self.endpoint_count, self.action_dim)
        return MotionCommand(
            decision_tensor=decision_tensor,
            target_endpoint_pose=endpointPose,
            endpoint_names=template.endpoint_names,
            gripper_cmd=template.gripper_cmd,
            mode_logits=template.mode_logits,
            safety_scores=template.safety_scores,)

    def SelectCommand(
        self,
        candidate: MotionCommand,
        active: MotionCommand,
        hold: MotionCommand,
        kindWeight: torch.Tensor,) -> MotionCommand:
        w_candidate = kindWeight[:, DISPATCH] + kindWeight[:, REDISPATCH]
        w_active = kindWeight[:, CONTINUE]
        w_hold = kindWeight[:, OBSERVE] + kindWeight[:, CANCEL] + kindWeight[:, FAILSAFE_STOP]
        use_candidate = w_candidate.view(-1, 1, 1)
        use_active = w_active.view(-1, 1, 1)
        use_hold = w_hold.view(-1, 1, 1)
        
        decision_tensor = (
            candidate.decision_tensor * use_candidate
            + active.decision_tensor * use_active
            + hold.decision_tensor * use_hold)
        
        pose_candidate = w_candidate.view(-1, 1, 1)
        pose_active = w_active.view(-1, 1, 1)
        pose_hold = w_hold.view(-1, 1, 1)
        
        target_pose = (
            candidate.target_endpoint_pose * pose_candidate
            + active.target_endpoint_pose * pose_active
            + hold.target_endpoint_pose * pose_hold)
        
        flat_candidate = w_candidate.view(-1, 1)
        flat_active = w_active.view(-1, 1)
        flat_hold = w_hold.view(-1, 1)
        gripper_candidate = flat_candidate.unsqueeze(-1)
        gripper_active = flat_active.unsqueeze(-1)
        gripper_hold = flat_hold.unsqueeze(-1)
        
        return MotionCommand(
            decision_tensor=decision_tensor,
            target_endpoint_pose=target_pose,
            endpoint_names=candidate.endpoint_names,
            gripper_cmd=candidate.gripper_cmd * gripper_candidate + active.gripper_cmd * gripper_active + hold.gripper_cmd * gripper_hold,
            mode_logits=candidate.mode_logits * flat_candidate + active.mode_logits * flat_active + hold.mode_logits * flat_hold,
            safety_scores=candidate.safety_scores * flat_candidate + active.safety_scores * flat_active + hold.safety_scores * flat_hold,)

    def forward(
        self,
        temporalContext: TemporalContext,
        decisionTemporal: dict,
        candidateCommand: MotionCommand,
        activeCommand: MotionCommand,
        endpointPose: torch.Tensor,
        actionEpoch: torch.Tensor,
        invokeDrift: torch.Tensor,) -> TemporalDecisionEnvelope:
        logits = decisionTemporal["kind_logits"] + self.context_refiner(temporalContext.feat)
        active = temporalContext.active_mask
        same_operator = decisionTemporal["same_operator"].view(-1)
        operator_changed = decisionTemporal["operator_changed"].view(-1)
        invoke_delta = decisionTemporal["invoke_delta"].view(-1)
        reference_drift = decisionTemporal["reference_drift"].view(-1)
 
        drift_signal = (invokeDrift.view(-1) / self.drift_threshold).clamp(0.0, 1.0)
        observe_needed = temporalContext.no_slot_prob * (1.0 - temporalContext.reference_confidence)
        continue_gate = (
            active
            * same_operator
            * temporalContext.planner_executing
            * (1.0 - temporalContext.planner_failed)
            * (1.0 - temporalContext.planner_canceled)
            * (1.0 - temporalContext.hard_stop))
        
        continue_penalty_terms = torch.stack([
            invoke_delta,
            reference_drift,
            temporalContext.planner_tracking_error,
            temporalContext.safety_risk,
            temporalContext.satisfaction_prob,
            temporalContext.planner_failed,
            temporalContext.planner_canceled,
            temporalContext.planner_reached,
            1.0 - temporalContext.observation_freshness,
            temporalContext.interrupt_risk,], dim=-1)
        
        continue_penalty = (
            continue_penalty_terms
            * F.softplus(self.continue_penalty_raw).view(1, -1)).sum(dim=-1)
        
        continue_ok = continue_gate * torch.sigmoid(self.continue_base - continue_penalty)
        
        cancel_need = active * temporalContext.can_interrupt * torch.maximum(
            torch.maximum(temporalContext.interrupt_risk, temporalContext.safety_risk),
            temporalContext.planner_canceled)
        
        redispatch_signal = torch.maximum(
            decisionTemporal["redispatch_score"],
            torch.maximum(
                torch.maximum(operator_changed, invoke_delta),
                torch.maximum(
                    torch.maximum(drift_signal, reference_drift),
                    torch.maximum(
                        torch.maximum(temporalContext.planner_tracking_error, temporalContext.planner_failed),
                        temporalContext.planner_reached * (1.0 - temporalContext.satisfaction_prob)))))
        
        redispatch_need = active * redispatch_signal * (1.0 - temporalContext.safety_risk)
        dispatch_need = (1.0 - active) * temporalContext.reference_confidence + active * temporalContext.satisfaction_prob
        gain = self.rule_gain_max * torch.sigmoid(self.rule_gain_raw)
        inactive_penalty = F.softplus(self.inactive_penalty_raw) * (1.0 - active).unsqueeze(-1)
        
        rule_bias = torch.stack([
            gain[0] * observe_needed,
            gain[1] * dispatch_need,
            gain[2] * continue_ok - inactive_penalty[:, 0],
            gain[3] * cancel_need - inactive_penalty[:, 1],
            gain[4] * temporalContext.hard_stop,
            gain[5] * redispatch_need - inactive_penalty[:, 2],], dim=-1)
        
        kind_logits = logits + rule_bias
        kind_id = kind_logits.argmax(dim=-1)
        hard_id = kind_id.new_full(kind_id.shape, FAILSAFE_STOP)
        kind_id = torch.where(temporalContext.hard_stop > 0.5, hard_id, kind_id)
 
        if self.training:
            kind_soft = F.softmax(kind_logits, dim=-1)
            kind_hard = torch.zeros_like(kind_soft).scatter_(-1, kind_id.unsqueeze(-1), 1.0)
            kind_weight = kind_hard + kind_soft - kind_soft.detach()
        else:
            kind_weight = torch.zeros_like(kind_logits).scatter_(-1, kind_id.unsqueeze(-1), 1.0)
        
        hold_command = self.HoldCommand(endpointPose, candidateCommand)
        motion_command = self.SelectCommand(candidateCommand, activeCommand, hold_command, kind_weight)
        publish_motion_command = kind_weight[:, DISPATCH] + kind_weight[:, REDISPATCH]
        reuse_active_motion_command = kind_weight[:, CONTINUE]
        publish_stop_command = kind_weight[:, CANCEL] + kind_weight[:, FAILSAFE_STOP]
 
        publish_hold_command = kind_weight[:, OBSERVE]
        reason_logits = self.reason_head(torch.cat([temporalContext.feat, kind_logits], dim=-1))
        
        return TemporalDecisionEnvelope(
            kind_logits=kind_logits,
            kind_id=kind_id,
            kind_names=ModuleDim.TemporalPrimitiveNames,
            action_id=actionEpoch,
            action_epoch=actionEpoch,
            reason_logits=reason_logits,
            duration_ms=decisionTemporal["duration_ms"],
            soft_timeout_ms=decisionTemporal["soft_timeout_ms"],
            hard_timeout_ms=decisionTemporal["hard_timeout_ms"],
            publish_motion_command=publish_motion_command,
            reuse_active_motion_command=reuse_active_motion_command,
            publish_stop_command=publish_stop_command,
            publish_hold_command=publish_hold_command,
            same_operator=same_operator,
            operator_changed=operator_changed,
            invoke_delta=invoke_delta,
            reference_drift=reference_drift,
            invoke_drift=invokeDrift.view(-1),
            motion_command=motion_command,)


class TestTemporalExecutionGateExtractorMTool:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def MakeGate(self) -> TemporalExecutionGateExtractor:
        return TemporalExecutionGateExtractor().to(self.device)

    def MakeEndpointPose(self, B: int) -> torch.Tensor:
        endpoint_pose = torch.zeros(
            B,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseDim,
            device=self.device)
        endpoint_pose[..., 6] = 1.0
        return endpoint_pose

    def MakeMotionCommand(self, B: int, value: float) -> MotionCommand:
        endpoint_pose = self.MakeEndpointPose(B)
        return MotionCommand(
            decision_tensor=torch.full(
                (B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim),
                float(value),
                device=self.device),
            target_endpoint_pose=endpoint_pose + float(value) * 0.01,
            endpoint_names=ModuleDim.DecisionEndpointNames,
            gripper_cmd=torch.full((B, ModuleDim.ArmCount, 1), float(value), device=self.device),
            mode_logits=torch.full((B, ModuleDim.ActTypeDim), float(value), device=self.device),
            safety_scores=torch.full((B, 5), float(value), device=self.device),)

    def MakeContext(
        self,
        gate: TemporalExecutionGateExtractor,
        B: int,
        *,
        active: float = 1.0,
        satisfied: float = 0.0,
        trackingError: float = 0.0,
        safetyRisk: float = 0.0,
        hardStop: float = 0.0,) -> TemporalContext:
        return gate.BuildContext(
            activeMask=torch.full((B,), float(active), device=self.device),
            actionAge=torch.ones(B, device=self.device),
            feedbackAge=torch.zeros(B, device=self.device),
            noSlotProb=torch.zeros(B, device=self.device),
            referenceConfidence=torch.ones(B, device=self.device),
            satisfactionProb=torch.full((B,), float(satisfied), device=self.device),
            safetyRisk=torch.full((B,), float(safetyRisk), device=self.device),
            interruptRisk=torch.zeros(B, device=self.device),
            observationFreshness=torch.ones(B, device=self.device),
            canInterrupt=torch.ones(B, device=self.device),
            hardStop=torch.full((B,), float(hardStop), device=self.device),
            plannerProgress=torch.zeros(B, device=self.device),
            plannerTrackingError=torch.full((B,), float(trackingError), device=self.device),
            plannerExecuting=torch.ones(B, device=self.device),
            plannerReached=torch.zeros(B, device=self.device),
            plannerFailed=torch.zeros(B, device=self.device),
            plannerCanceled=torch.zeros(B, device=self.device),)

    def MakeDecisionTemporal(
        self,
        B: int,
        *,
        sameOperator: float = 1.0,
        redispatchScore: float = 0.0,
        requiresGrad: bool = False,) -> Dict[str, torch.Tensor]:
        kind_logits = torch.zeros(
            B,
            ModuleDim.TemporalPrimitiveCount,
            device=self.device,
            requires_grad=requiresGrad)
        return {
            "kind_logits": kind_logits,
            "same_operator": torch.full((B,), float(sameOperator), device=self.device),
            "operator_changed": torch.full((B,), 1.0 - float(sameOperator), device=self.device),
            "invoke_delta": torch.zeros(B, device=self.device),
            "reference_drift": torch.zeros(B, device=self.device),
            "redispatch_score": torch.full((B,), float(redispatchScore), device=self.device),
            "duration_ms": torch.zeros(B, device=self.device),
            "soft_timeout_ms": torch.zeros(B, device=self.device),
            "hard_timeout_ms": torch.zeros(B, device=self.device),}

    def TestBuildContextShapes(self) -> bool:
        try:
            B = 3
            gate = self.MakeGate()
            ctx = self.MakeContext(gate, B)
            assert tuple(ctx.feat.shape) == (B, ModuleDim.TemporalContextDim)
            assert tuple(ctx.active_mask.shape) == (B,)
            assert tuple(ctx.planner_tracking_error.shape) == (B,)
            assert torch.isfinite(ctx.feat).all()
            print("TemporalExecutionGateExtractor BuildContext shape test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor BuildContext shape test failed: {type(e).__name__}: {e}")
            return False

    def TestHoldCommandShapes(self) -> bool:
        try:
            B = 2
            gate = self.MakeGate()
            endpoint_pose = self.MakeEndpointPose(B)
            template = self.MakeMotionCommand(B, 1.0)
            hold = gate.HoldCommand(endpoint_pose, template)
            assert tuple(hold.decision_tensor.shape) == (B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim)
            assert tuple(hold.target_endpoint_pose.shape) == (B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim)
            assert torch.allclose(hold.decision_tensor, torch.zeros_like(hold.decision_tensor))
            assert torch.allclose(hold.target_endpoint_pose, endpoint_pose)
            print("TemporalExecutionGateExtractor HoldCommand shape test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor HoldCommand shape test failed: {type(e).__name__}: {e}")
            return False

    def TestSelectCommandRoutes(self) -> bool:
        try:
            B = 1
            gate = self.MakeGate()
            candidate = self.MakeMotionCommand(B, 1.0)
            active = self.MakeMotionCommand(B, 2.0)
            hold = self.MakeMotionCommand(B, 3.0)
            weights = [
                (DISPATCH, candidate),
                (REDISPATCH, candidate),
                (CONTINUE, active),
                (OBSERVE, hold),
                (CANCEL, hold),
                (FAILSAFE_STOP, hold),]
            for kind, expected in weights:
                kind_weight = torch.zeros(B, ModuleDim.TemporalPrimitiveCount, device=self.device)
                kind_weight[:, kind] = 1.0
                out = gate.SelectCommand(candidate, active, hold, kind_weight)
                assert torch.allclose(out.decision_tensor, expected.decision_tensor)
                assert torch.allclose(out.target_endpoint_pose, expected.target_endpoint_pose)
            print("TemporalExecutionGateExtractor SelectCommand route test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor SelectCommand route test failed: {type(e).__name__}: {e}")
            return False

    def TestForwardShapesAndPublishFlags(self) -> bool:
        try:
            B = 2
            gate = self.MakeGate().eval()
            endpoint_pose = self.MakeEndpointPose(B)
            candidate = self.MakeMotionCommand(B, 1.0)
            active = self.MakeMotionCommand(B, 2.0)
            ctx = self.MakeContext(gate, B)
            temporal = self.MakeDecisionTemporal(B)
            with torch.no_grad():
                out = gate(
                    ctx,
                    temporal,
                    candidate,
                    active,
                    endpoint_pose,
                    torch.zeros(B, dtype=torch.long, device=self.device),
                    torch.zeros(B, device=self.device),)
            assert tuple(out.kind_logits.shape) == (B, ModuleDim.TemporalPrimitiveCount)
            assert tuple(out.kind_id.shape) == (B,)
            assert tuple(out.reason_logits.shape) == (B, ModuleDim.TemporalReasonDim)
            assert tuple(out.motion_command.decision_tensor.shape) == (B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim)
            flag_sum = (
                out.publish_motion_command
                + out.reuse_active_motion_command
                + out.publish_stop_command
                + out.publish_hold_command)
            assert torch.allclose(flag_sum, torch.ones_like(flag_sum))
            print("TemporalExecutionGateExtractor forward shape/publish flag test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor forward shape/publish flag test failed: {type(e).__name__}: {e}")
            return False

    def TestHardStopOverride(self) -> bool:
        try:
            B = 2
            gate = self.MakeGate().eval()
            endpoint_pose = self.MakeEndpointPose(B)
            cmd = self.MakeMotionCommand(B, 1.0)
            ctx = self.MakeContext(gate, B, hardStop=1.0)
            temporal = self.MakeDecisionTemporal(B)
            with torch.no_grad():
                out = gate(
                    ctx,
                    temporal,
                    cmd,
                    cmd,
                    endpoint_pose,
                    torch.zeros(B, dtype=torch.long, device=self.device),
                    torch.zeros(B, device=self.device),)
            assert bool((out.kind_id == FAILSAFE_STOP).all().item())
            assert bool((out.publish_stop_command == 1.0).all().item())
            print("TemporalExecutionGateExtractor hard stop override test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor hard stop override test failed: {type(e).__name__}: {e}")
            return False

    def TestContinuePenaltyResponse(self) -> bool:
        try:
            B = 1
            gate = self.MakeGate().eval()
            endpoint_pose = self.MakeEndpointPose(B)
            cmd = self.MakeMotionCommand(B, 1.0)
            temporal = self.MakeDecisionTemporal(B)
            ctx_good = self.MakeContext(gate, B, trackingError=0.0)
            ctx_bad = self.MakeContext(gate, B, trackingError=1.0)
            with torch.no_grad():
                out_good = gate(ctx_good, temporal, cmd, cmd, endpoint_pose, torch.zeros(B, dtype=torch.long, device=self.device), torch.zeros(B, device=self.device))
                out_bad = gate(ctx_bad, temporal, cmd, cmd, endpoint_pose, torch.zeros(B, dtype=torch.long, device=self.device), torch.zeros(B, device=self.device))
            assert out_good.kind_logits[0, CONTINUE] > out_bad.kind_logits[0, CONTINUE]
            print("TemporalExecutionGateExtractor continue penalty response test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor continue penalty response test failed: {type(e).__name__}: {e}")
            return False

    def TestTrainEvalSelectionConsistency(self) -> bool:
        try:
            B = 1
            gate = self.MakeGate()
            endpoint_pose = self.MakeEndpointPose(B)
            cmd = self.MakeMotionCommand(B, 1.0)
            ctx = self.MakeContext(gate, B)
            temporal = self.MakeDecisionTemporal(B)
            gate.train()
            out_train = gate(ctx, temporal, cmd, cmd, endpoint_pose, torch.zeros(B, dtype=torch.long, device=self.device), torch.zeros(B, device=self.device))
            gate.eval()
            with torch.no_grad():
                out_eval = gate(ctx, temporal, cmd, cmd, endpoint_pose, torch.zeros(B, dtype=torch.long, device=self.device), torch.zeros(B, device=self.device))
            assert torch.equal(out_train.kind_id, out_eval.kind_id)
            assert torch.allclose(out_train.motion_command.decision_tensor, out_eval.motion_command.decision_tensor)
            print("TemporalExecutionGateExtractor train/eval selection consistency test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor train/eval selection consistency test failed: {type(e).__name__}: {e}")
            return False

    def TestTrainingGradient(self) -> bool:
        try:
            B = 1
            gate = self.MakeGate().train()
            endpoint_pose = self.MakeEndpointPose(B)
            candidate = self.MakeMotionCommand(B, 1.0)
            active = self.MakeMotionCommand(B, 0.0)
            ctx = self.MakeContext(gate, B, active=0.0)
            temporal = self.MakeDecisionTemporal(B, requiresGrad=True)
            out = gate(
                ctx,
                temporal,
                candidate,
                active,
                endpoint_pose,
                torch.zeros(B, dtype=torch.long, device=self.device),
                torch.zeros(B, device=self.device),)
            loss = out.motion_command.decision_tensor.sum()
            loss.backward()
            grad = temporal["kind_logits"].grad
            assert grad is not None
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0
            print("TemporalExecutionGateExtractor training gradient test passed.")
            return True
        except Exception as e:
            print(f"TemporalExecutionGateExtractor training gradient test failed: {type(e).__name__}: {e}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "BuildContextShapes": self.TestBuildContextShapes(),
            "HoldCommandShapes": self.TestHoldCommandShapes(),
            "SelectCommandRoutes": self.TestSelectCommandRoutes(),
            "ForwardShapesAndPublishFlags": self.TestForwardShapesAndPublishFlags(),
            "HardStopOverride": self.TestHardStopOverride(),
            "ContinuePenaltyResponse": self.TestContinuePenaltyResponse(),
            "TrainEvalSelectionConsistency": self.TestTrainEvalSelectionConsistency(),
            "TrainingGradient": self.TestTrainingGradient(),}
        passed = sum(1 for ok in results.values() if ok)
        print(f"\n[TemporalExecutionGateExtractor Tests] {passed}/{len(results)} passed.")
        return results
