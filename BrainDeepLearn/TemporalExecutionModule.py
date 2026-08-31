from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from FunctionTools import AGICoreModule
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    PackedEndEffectorTarget,
    RobotEmbodimentContractView,
)


OBSERVE = 0
DISPATCH = 1
CONTINUE = 2
CANCEL = 3
FAILSAFE_STOP = 4
REDISPATCH = 5

PACKED_TEMPORAL_KIND_NAMES = (
    "OBSERVE",
    "DISPATCH",
    "CONTINUE",
    "CANCEL",
    "FAILSAFE_STOP",
    "REDISPATCH",
)


@dataclass(frozen=True)
class PackedTemporalProposal:
    kind_scores: torch.Tensor
    same_operator: torch.Tensor
    operator_changed: torch.Tensor
    invoke_delta: torch.Tensor
    reference_drift: torch.Tensor
    redispatch_score: torch.Tensor
    interrupt_score: torch.Tensor
    duration_ms: torch.Tensor
    soft_timeout_ms: torch.Tensor
    hard_timeout_ms: torch.Tensor
    action_epoch: torch.Tensor

    def Validate(self, batchSize: int, device: torch.device) -> None:
        batch_size = int(batchSize)
        if (
            not torch.is_tensor(self.kind_scores)
            or tuple(self.kind_scores.shape) != (
                batch_size,
                len(PACKED_TEMPORAL_KIND_NAMES),
            )
            or not self.kind_scores.is_floating_point()
            or self.kind_scores.device != device
            or bool(torch.isnan(self.kind_scores).any().item())
            or bool(torch.isposinf(self.kind_scores).any().item())
            or not bool(
                torch.isfinite(self.kind_scores).any(dim=-1).all().item())
        ):
            raise ValueError(
                "packed temporal proposal scores must cover every state")
        scalar_fields = (
            self.same_operator,
            self.operator_changed,
            self.invoke_delta,
            self.reference_drift,
            self.redispatch_score,
            self.interrupt_score,
            self.duration_ms,
            self.soft_timeout_ms,
            self.hard_timeout_ms,
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (batch_size,)
            or not value.is_floating_point()
            or value.device != device
            or not bool(torch.isfinite(value).all().item())
            for value in scalar_fields
        ):
            raise ValueError(
                "packed temporal proposal metadata must match the batch")
        if any(
            bool((value < 0.0).any().item())
            for value in (
                self.duration_ms,
                self.soft_timeout_ms,
                self.hard_timeout_ms,
            )
        ):
            raise ValueError("packed temporal timeouts cannot be negative")
        if bool((self.soft_timeout_ms > self.hard_timeout_ms).any().item()):
            raise ValueError(
                "packed temporal soft timeout cannot exceed hard timeout")
        if (
            not torch.is_tensor(self.action_epoch)
            or tuple(self.action_epoch.shape) != (batch_size,)
            or self.action_epoch.dtype != torch.long
            or self.action_epoch.device != device
            or bool((self.action_epoch < 0).any().item())
        ):
            raise ValueError(
                "packed temporal action epoch must be non-negative")


@dataclass(frozen=True)
class PackedTemporalEvent:
    cache_executing: torch.Tensor
    candidate_ready: torch.Tensor
    redispatch_requested: torch.Tensor
    cancel_requested: torch.Tensor
    planner_failed: torch.Tensor
    plan_reached: torch.Tensor
    hard_stop: torch.Tensor
    active_risk: torch.Tensor
    candidate_risk: torch.Tensor

    def Validate(self, batchSize: int, device: torch.device) -> None:
        expected_shape = (int(batchSize),)
        boolean_fields = (
            self.cache_executing,
            self.candidate_ready,
            self.redispatch_requested,
            self.cancel_requested,
            self.planner_failed,
            self.plan_reached,
            self.hard_stop,
        )
        risk_fields = (
            self.active_risk,
            self.candidate_risk,
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != expected_shape
            or value.dtype != torch.bool
            or value.device != device
            for value in boolean_fields
        ):
            raise ValueError(
                "packed temporal boolean events must match the feedback batch")
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != expected_shape
            or not value.is_floating_point()
            or value.device != device
            or not bool(torch.isfinite(value).all().item())
            or bool((value < 0.0).any().item())
            or bool((value > 1.0).any().item())
            for value in risk_fields
        ):
            raise ValueError(
                "packed temporal risks must be finite normalized batch tensors")


@dataclass(frozen=True)
class PackedTemporalDecision:
    proposal_scores: torch.Tensor
    execution_kind_scores: torch.Tensor
    proposal_kind_id: torch.Tensor
    kind_id: torch.Tensor
    kind_names: Tuple[str, ...]
    override_applied: torch.Tensor
    proposal_action_epoch: torch.Tensor
    action_epoch: torch.Tensor
    duration_ms: torch.Tensor
    soft_timeout_ms: torch.Tensor
    hard_timeout_ms: torch.Tensor
    same_operator: torch.Tensor
    operator_changed: torch.Tensor
    invoke_delta: torch.Tensor
    reference_drift: torch.Tensor
    redispatch_score: torch.Tensor
    interrupt_score: torch.Tensor
    selected_target: PackedEndEffectorTarget
    candidate_selected: torch.Tensor
    cache_selected: torch.Tensor
    hold_requested: torch.Tensor
    stop_requested: torch.Tensor
    failsafe: torch.Tensor


class PackedTemporalExecutionGate:
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        *,
        dispatchRiskThreshold: float = 0.5,
        failsafeRiskThreshold: float = 0.98,
        ageNormSteps: float = 100.0,
        stepDurationMs: float = 1.0,
    ) -> None:
        dispatch_threshold = float(dispatchRiskThreshold)
        failsafe_threshold = float(failsafeRiskThreshold)
        if not 0.0 <= dispatch_threshold <= 1.0:
            raise ValueError("dispatch risk threshold must be normalized")
        if not 0.0 <= failsafe_threshold <= 1.0:
            raise ValueError("failsafe risk threshold must be normalized")
        if failsafe_threshold < dispatch_threshold:
            raise ValueError(
                "failsafe risk threshold cannot be below dispatch threshold")
        age_norm_steps = float(ageNormSteps)
        if not torch.isfinite(torch.tensor(age_norm_steps)) or age_norm_steps <= 0.0:
            raise ValueError("packed temporal age normalization must be positive")
        step_duration_ms = float(stepDurationMs)
        if (
            not torch.isfinite(torch.tensor(step_duration_ms))
            or step_duration_ms <= 0.0
        ):
            raise ValueError("packed temporal step duration must be positive")
        self.contract_view = contractView
        self.dispatch_risk_threshold = dispatch_threshold
        self.failsafe_risk_threshold = failsafe_threshold
        self.age_norm_steps = age_norm_steps
        self.step_duration_ms = step_duration_ms

    def BuildContext(
        self,
        activeMask: torch.Tensor,
        actionAgeSteps: torch.Tensor,
        noReferenceProb: torch.Tensor,
        referenceConfidence: torch.Tensor,
        satisfactionProb: torch.Tensor,
        safetyRisk: torch.Tensor,
        candidateSafetyRisk: torch.Tensor,
        interruptRisk: torch.Tensor,
        observationFreshness: torch.Tensor,
        canInterrupt: torch.Tensor,
        hardStop: torch.Tensor,
        plannerProgress: torch.Tensor,
        plannerTrackingError: torch.Tensor,
        plannerExecuting: torch.Tensor,
        plannerReached: torch.Tensor,
        plannerFailed: torch.Tensor,
    ) -> "TemporalContext":
        active = activeMask.view(-1)
        action_age_steps = actionAgeSteps.view(-1)
        no_reference = noReferenceProb.view(-1)
        reference_confidence = referenceConfidence.view(-1)
        satisfaction = satisfactionProb.view(-1)
        safety = safetyRisk.view(-1)
        candidate_safety = candidateSafetyRisk.view(-1)
        interrupt = interruptRisk.view(-1)
        freshness = observationFreshness.view(-1)
        can_interrupt = canInterrupt.view(-1)
        hard_stop = hardStop.view(-1)
        planner_progress = plannerProgress.view(-1)
        planner_tracking_error = plannerTrackingError.view(-1)
        planner_executing = plannerExecuting.view(-1)
        planner_reached = plannerReached.view(-1)
        planner_failed = plannerFailed.view(-1)
        feature = torch.stack((
            active,
            action_age_steps / self.age_norm_steps,
            no_reference,
            reference_confidence,
            satisfaction,
            safety,
            interrupt,
            freshness,
            can_interrupt,
            hard_stop,
            active * (1.0 - satisfaction),
            no_reference * (1.0 - active),
            interrupt * active,
            safety * can_interrupt,
            reference_confidence * freshness,
            planner_progress,
            planner_tracking_error,
            planner_executing,
            planner_reached,
            planner_failed,
        ), dim=-1)
        return TemporalContext(
            feat=feature,
            active_mask=active,
            action_age_steps=action_age_steps,
            no_reference_prob=no_reference,
            reference_confidence=reference_confidence,
            satisfaction_prob=satisfaction,
            safety_risk=safety,
            candidate_safety_risk=candidate_safety,
            interrupt_risk=interrupt,
            observation_freshness=freshness,
            can_interrupt=can_interrupt,
            hard_stop=hard_stop,
            planner_progress=planner_progress,
            planner_tracking_error=planner_tracking_error,
            planner_executing=planner_executing,
            planner_reached=planner_reached,
            planner_failed=planner_failed)

    def ValidateTarget(
        self,
        target: PackedEndEffectorTarget,
        feedback: BrainFeedbackPacket,
        fieldName: str,
    ) -> None:
        if target.values.size(0) != feedback.joint_features.size(0):
            raise ValueError(fieldName + " batch does not match feedback")
        if target.values.device != feedback.joint_features.device:
            raise ValueError(fieldName + " device does not match feedback")
        if target.values.dtype != feedback.joint_features.dtype:
            raise ValueError(fieldName + " dtype does not match feedback")

    def TargetAvailable(
        self,
        target: PackedEndEffectorTarget,
        feedback: BrainFeedbackPacket,
    ) -> torch.Tensor:
        available = (
            feedback.endpoint_present
            & feedback.child_enabled
        )
        return ((~target.active) | available).all(dim=-1)

    def TargetCriticalFailure(
        self,
        target: PackedEndEffectorTarget,
        feedback: BrainFeedbackPacket,
    ) -> torch.Tensor:
        critical = (
            ~feedback.endpoint_present
            | ~feedback.child_enabled
        )
        return (target.active & critical).any(dim=-1)

    def NeutralTarget(
        self,
        template: PackedEndEffectorTarget,
    ) -> PackedEndEffectorTarget:
        return PackedEndEffectorTarget(
            values=torch.zeros_like(template.values),
            active=torch.zeros_like(template.active),
            contract_id=self.contract_view.contract_id,
            model_signature=self.contract_view.model_signature,
            target_version=template.target_version,
            timestamp=template.timestamp,
        )

    def Step(
        self,
        feedback: BrainFeedbackPacket,
        candidateTarget: PackedEndEffectorTarget,
        cachedTarget: PackedEndEffectorTarget,
        proposal: PackedTemporalProposal,
        events: PackedTemporalEvent,
        actionAgeSteps: torch.Tensor,
    ) -> PackedTemporalDecision:
        self.ValidateTarget(candidateTarget, feedback, "candidate target")
        self.ValidateTarget(cachedTarget, feedback, "cached target")
        if type(proposal) is not PackedTemporalProposal:
            raise TypeError(
                "packed temporal execution requires a learned proposal")
        if type(events) is not PackedTemporalEvent:
            raise TypeError("packed temporal execution requires packed events")
        batch_size = int(feedback.joint_features.size(0))
        proposal.Validate(batch_size, feedback.joint_features.device)
        events.Validate(batch_size, feedback.joint_features.device)
        action_age_steps = actionAgeSteps.reshape(-1)
        if (
            tuple(action_age_steps.shape) != (batch_size,)
            or not action_age_steps.is_floating_point()
            or action_age_steps.device != feedback.joint_features.device
            or not bool(torch.isfinite(action_age_steps).all().item())
            or bool((action_age_steps < 0.0).any().item())
        ):
            raise ValueError(
                "packed temporal action age must match the feedback batch")
        elapsed_ms = action_age_steps * self.step_duration_ms

        candidate_present = candidateTarget.active.any(dim=-1)
        cache_present = (
            events.cache_executing
            & cachedTarget.active.any(dim=-1)
        )
        candidate_available = (
            events.candidate_ready
            & candidate_present
            & self.TargetAvailable(candidateTarget, feedback)
            & events.candidate_risk.lt(self.dispatch_risk_threshold)
        )
        cache_available = self.TargetAvailable(cachedTarget, feedback)
        active_failure = self.TargetCriticalFailure(cachedTarget, feedback)
        hard_timeout = (
            cache_present
            & proposal.hard_timeout_ms.gt(0.0)
            & elapsed_ms.ge(proposal.hard_timeout_ms)
        )
        soft_timeout = (
            cache_present
            & ~hard_timeout
            & proposal.soft_timeout_ms.gt(0.0)
            & elapsed_ms.ge(proposal.soft_timeout_ms)
        )

        legal_states = torch.stack((
            torch.ones_like(cache_present),
            ~cache_present & candidate_available,
            cache_present & cache_available,
            cache_present,
            torch.ones_like(cache_present),
            cache_present & candidate_available,
        ), dim=-1)
        execution_kind_scores = proposal.kind_scores.masked_fill(
            ~legal_states,
            -torch.inf,
        )
        proposal_kind_id = proposal.kind_scores.argmax(dim=-1)
        kind_id = execution_kind_scores.argmax(dim=-1)

        rule_failsafe = (
            events.hard_stop
            | events.active_risk.ge(self.failsafe_risk_threshold)
            | (cache_present & active_failure)
            | hard_timeout
        )
        rule_cancel = (
            ~rule_failsafe
            & cache_present
            & (
                events.cancel_requested
                | events.planner_failed
                | events.plan_reached
                | ~cache_available
                | (soft_timeout & ~candidate_available)
            )
        )
        rule_redispatch = (
            ~rule_failsafe
            & ~rule_cancel
            & cache_present
            & (events.redispatch_requested | soft_timeout)
            & candidate_available
        )
        kind_id = torch.where(
            rule_redispatch,
            torch.full_like(kind_id, REDISPATCH),
            kind_id,
        )
        kind_id = torch.where(
            rule_cancel,
            torch.full_like(kind_id, CANCEL),
            kind_id,
        )
        kind_id = torch.where(
            rule_failsafe,
            torch.full_like(kind_id, FAILSAFE_STOP),
            kind_id,
        )

        dispatch = kind_id.eq(DISPATCH)
        keep_cache = kind_id.eq(CONTINUE)
        cancel = kind_id.eq(CANCEL)
        failsafe = kind_id.eq(FAILSAFE_STOP)
        redispatch = kind_id.eq(REDISPATCH)
        observe = kind_id.eq(OBSERVE)
        neutral = self.NeutralTarget(candidateTarget)
        candidate_rows = (dispatch | redispatch).unsqueeze(-1)
        cache_rows = keep_cache.unsqueeze(-1)
        selected_target = PackedEndEffectorTarget(
            values=torch.where(
                candidate_rows,
                candidateTarget.values,
                torch.where(
                    cache_rows,
                    cachedTarget.values,
                    neutral.values,
                ),
            ),
            active=torch.where(
                candidate_rows,
                candidateTarget.active,
                torch.where(
                    cache_rows,
                    cachedTarget.active,
                    neutral.active,
                ),
            ),
            contract_id=self.contract_view.contract_id,
            model_signature=self.contract_view.model_signature,
            target_version=torch.where(
                dispatch | redispatch,
                candidateTarget.target_version,
                torch.where(
                    keep_cache,
                    cachedTarget.target_version,
                    neutral.target_version)),
            timestamp=torch.where(
                dispatch | redispatch,
                candidateTarget.timestamp,
                torch.where(
                    keep_cache,
                    cachedTarget.timestamp,
                    neutral.timestamp)),
        )
        candidate_selected = dispatch | redispatch
        next_epoch = (
            proposal.action_epoch
            + candidate_selected.to(proposal.action_epoch.dtype)
        )
        return PackedTemporalDecision(
            proposal_scores=proposal.kind_scores,
            execution_kind_scores=execution_kind_scores,
            proposal_kind_id=proposal_kind_id,
            kind_id=kind_id,
            kind_names=PACKED_TEMPORAL_KIND_NAMES,
            override_applied=kind_id.ne(proposal_kind_id),
            proposal_action_epoch=proposal.action_epoch,
            action_epoch=next_epoch,
            duration_ms=proposal.duration_ms,
            soft_timeout_ms=proposal.soft_timeout_ms,
            hard_timeout_ms=proposal.hard_timeout_ms,
            same_operator=proposal.same_operator,
            operator_changed=proposal.operator_changed,
            invoke_delta=proposal.invoke_delta,
            reference_drift=proposal.reference_drift,
            redispatch_score=proposal.redispatch_score,
            interrupt_score=proposal.interrupt_score,
            selected_target=selected_target,
            candidate_selected=candidate_selected,
            cache_selected=keep_cache,
            hold_requested=observe,
            stop_requested=cancel | failsafe,
            failsafe=failsafe,
        )


@dataclass(frozen=True)
class TemporalContext:
    feat: torch.Tensor
    active_mask: torch.Tensor
    action_age_steps: torch.Tensor
    no_reference_prob: torch.Tensor
    reference_confidence: torch.Tensor
    satisfaction_prob: torch.Tensor
    safety_risk: torch.Tensor
    candidate_safety_risk: torch.Tensor
    interrupt_risk: torch.Tensor
    observation_freshness: torch.Tensor
    can_interrupt: torch.Tensor
    hard_stop: torch.Tensor
    planner_progress: torch.Tensor
    planner_tracking_error: torch.Tensor
    planner_executing: torch.Tensor
    planner_reached: torch.Tensor
    planner_failed: torch.Tensor

    def Validate(
        self,
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        B = int(batchSize)
        if (
            not torch.is_tensor(self.feat)
            or tuple(self.feat.shape) != (B, 20)
            or self.feat.device != device
            or self.feat.dtype != dtype
            or not bool(torch.isfinite(self.feat).all().item())
        ):
            raise ValueError("temporal context feature is invalid")
        values = (
            self.active_mask,
            self.action_age_steps,
            self.no_reference_prob,
            self.reference_confidence,
            self.satisfaction_prob,
            self.safety_risk,
            self.candidate_safety_risk,
            self.interrupt_risk,
            self.observation_freshness,
            self.can_interrupt,
            self.hard_stop,
            self.planner_progress,
            self.planner_tracking_error,
            self.planner_executing,
            self.planner_reached,
            self.planner_failed,
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (B,)
            or value.device != device
            or value.dtype != dtype
            or not bool(torch.isfinite(value).all().item())
            for value in values
        ):
            raise ValueError("temporal context values are invalid")
        normalized = (
            self.active_mask,
            self.no_reference_prob,
            self.reference_confidence,
            self.satisfaction_prob,
            self.safety_risk,
            self.candidate_safety_risk,
            self.interrupt_risk,
            self.observation_freshness,
            self.can_interrupt,
            self.hard_stop,
            self.planner_progress,
            self.planner_executing,
            self.planner_reached,
            self.planner_failed,
        )
        if any(
            bool((value < 0.0).any().item())
            or bool((value > 1.0).any().item())
            for value in normalized
        ):
            raise ValueError("temporal context values must be normalized")
        if (
            bool((self.action_age_steps < 0.0).any().item())
            or bool((self.planner_tracking_error < 0.0).any().item())
        ):
            raise ValueError("temporal magnitudes cannot be negative")


class TemporalExecutionGateExtractor(AGICoreModule):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        *,
        dispatchRiskThreshold: float = 0.5,
        failsafeRiskThreshold: float = 0.98,
        ageNormSteps: float = 128.0,
        stepDurationMs: float = 1.0,
    ) -> None:
        super().__init__()
        self.execution_gate = PackedTemporalExecutionGate(
            contractView=contractView,
            dispatchRiskThreshold=dispatchRiskThreshold,
            failsafeRiskThreshold=failsafeRiskThreshold,
            ageNormSteps=ageNormSteps,
            stepDurationMs=stepDurationMs)
        self.register_buffer(
            "rule_gain",
            torch.tensor([2.0, 2.0, 2.0, 3.0, 8.0, 3.0]),
            persistent=True)
        self.register_buffer(
            "inactive_penalty",
            torch.tensor([4.0, 4.0, 4.0]),
            persistent=True)
        self.register_buffer(
            "continue_base",
            torch.tensor(3.0),
            persistent=True)
        self.register_buffer(
            "continue_penalty",
            torch.tensor([
                1.4,
                1.6,
                2.0,
                2.0,
                1.6,
                4.0,
                2.0,
                1.2,
                1.5,
            ]),
            persistent=True)
        self.register_buffer(
            "drift_threshold",
            torch.tensor(4.0),
            persistent=True)

    @property
    def ContractView(self) -> RobotEmbodimentContractView:
        return self.execution_gate.contract_view

    @torch.no_grad()
    def CalibrateDriftThreshold(
        self,
        driftSamples: torch.Tensor,
        quantile: float = 0.95,
    ) -> None:
        samples = driftSamples.reshape(-1)
        if (
            samples.numel() < 1
            or not samples.is_floating_point()
            or not bool(torch.isfinite(samples).all().item())
            or bool((samples < 0.0).any().item())
        ):
            raise ValueError("drift samples must be finite and non-negative")
        q = float(quantile)
        if not 0.0 <= q <= 1.0:
            raise ValueError("drift quantile must be normalized")
        calibrated = torch.quantile(samples, q).clamp_min(1e-6)
        self.drift_threshold.copy_(calibrated.to(
            device=self.drift_threshold.device,
            dtype=self.drift_threshold.dtype))

    def BuildContext(
        self,
        activeMask: torch.Tensor,
        actionAgeSteps: torch.Tensor,
        noReferenceProb: torch.Tensor,
        referenceConfidence: torch.Tensor,
        satisfactionProb: torch.Tensor,
        safetyRisk: torch.Tensor,
        candidateSafetyRisk: torch.Tensor,
        interruptRisk: torch.Tensor,
        observationFreshness: torch.Tensor,
        canInterrupt: torch.Tensor,
        hardStop: torch.Tensor,
        plannerProgress: torch.Tensor,
        plannerTrackingError: torch.Tensor,
        plannerExecuting: torch.Tensor,
        plannerReached: torch.Tensor,
        plannerFailed: torch.Tensor,
    ) -> TemporalContext:
        return self.execution_gate.BuildContext(
            activeMask=activeMask,
            actionAgeSteps=actionAgeSteps,
            noReferenceProb=noReferenceProb,
            referenceConfidence=referenceConfidence,
            satisfactionProb=satisfactionProb,
            safetyRisk=safetyRisk,
            candidateSafetyRisk=candidateSafetyRisk,
            interruptRisk=interruptRisk,
            observationFreshness=observationFreshness,
            canInterrupt=canInterrupt,
            hardStop=hardStop,
            plannerProgress=plannerProgress,
            plannerTrackingError=plannerTrackingError,
            plannerExecuting=plannerExecuting,
            plannerReached=plannerReached,
            plannerFailed=plannerFailed)

    def RefineProposal(
        self,
        temporalContext: TemporalContext,
        proposal: PackedTemporalProposal,
        invokeDrift: torch.Tensor,
    ) -> PackedTemporalProposal:
        B = int(proposal.kind_scores.size(0))
        proposal.Validate(B, proposal.kind_scores.device)
        temporalContext.Validate(
            B,
            proposal.kind_scores.device,
            proposal.kind_scores.dtype)
        invoke_drift = invokeDrift.reshape(-1).to(
            device=proposal.kind_scores.device,
            dtype=proposal.kind_scores.dtype)
        if (
            tuple(invoke_drift.shape) != (B,)
            or not bool(torch.isfinite(invoke_drift).all().item())
            or bool((invoke_drift < 0.0).any().item())
        ):
            raise ValueError("invoke drift must be finite and non-negative")

        active = temporalContext.active_mask
        drift_signal = (
            invoke_drift
            / self.drift_threshold.to(
                device=invoke_drift.device,
                dtype=invoke_drift.dtype)
        ).clamp(0.0, 1.0)
        observe_needed = (
            temporalContext.no_reference_prob
            * (1.0 - temporalContext.reference_confidence))
        continue_gate = (
            active
            * proposal.same_operator
            * temporalContext.planner_executing
            * (1.0 - temporalContext.planner_failed)
            * (1.0 - temporalContext.hard_stop))
        continue_penalty_terms = torch.stack([
            proposal.invoke_delta,
            proposal.reference_drift,
            temporalContext.planner_tracking_error,
            temporalContext.safety_risk,
            temporalContext.satisfaction_prob,
            temporalContext.planner_failed,
            temporalContext.planner_reached,
            1.0 - temporalContext.observation_freshness,
            temporalContext.interrupt_risk,
        ], dim=-1)
        continue_penalty = (
            continue_penalty_terms
            * self.continue_penalty.to(
                device=continue_penalty_terms.device,
                dtype=continue_penalty_terms.dtype).view(1, -1)
        ).sum(dim=-1)
        continue_ok = continue_gate * torch.sigmoid(
            self.continue_base.to(
                device=continue_penalty.device,
                dtype=continue_penalty.dtype)
            - continue_penalty)
        cancel_need = (
            active
            * temporalContext.can_interrupt
            * torch.maximum(
                temporalContext.interrupt_risk,
                temporalContext.safety_risk))
        redispatch_signal = torch.maximum(
            proposal.redispatch_score,
            torch.maximum(
                torch.maximum(proposal.operator_changed, proposal.invoke_delta),
                torch.maximum(
                    torch.maximum(drift_signal, proposal.reference_drift),
                    torch.maximum(
                        torch.maximum(
                            temporalContext.planner_tracking_error,
                            temporalContext.planner_failed),
                        temporalContext.planner_reached
                        * (1.0 - temporalContext.satisfaction_prob)))))
        redispatch_need = (
            active
            * redispatch_signal
            * (1.0 - temporalContext.candidate_safety_risk))
        dispatch_need = (
            (1.0 - active) * temporalContext.reference_confidence
            + active * temporalContext.satisfaction_prob
        ) * (1.0 - temporalContext.candidate_safety_risk)
        gain = self.rule_gain.to(
            device=proposal.kind_scores.device,
            dtype=proposal.kind_scores.dtype)
        inactive_penalty = self.inactive_penalty.to(
            device=proposal.kind_scores.device,
            dtype=proposal.kind_scores.dtype) * (1.0 - active).unsqueeze(-1)
        rule_bias = torch.stack([
            gain[0] * observe_needed,
            gain[1] * dispatch_need,
            gain[2] * continue_ok - inactive_penalty[:, 0],
            gain[3] * cancel_need - inactive_penalty[:, 1],
            gain[4] * temporalContext.hard_stop,
            gain[5] * redispatch_need - inactive_penalty[:, 2],
        ], dim=-1)
        return PackedTemporalProposal(
            kind_scores=proposal.kind_scores + rule_bias,
            same_operator=proposal.same_operator,
            operator_changed=proposal.operator_changed,
            invoke_delta=proposal.invoke_delta,
            reference_drift=proposal.reference_drift,
            redispatch_score=redispatch_signal,
            interrupt_score=torch.maximum(
                proposal.interrupt_score,
                cancel_need),
            duration_ms=proposal.duration_ms,
            soft_timeout_ms=proposal.soft_timeout_ms,
            hard_timeout_ms=proposal.hard_timeout_ms,
            action_epoch=proposal.action_epoch)

    def Step(
        self,
        feedbackPacket: BrainFeedbackPacket,
        candidateTarget: PackedEndEffectorTarget,
        cachedTarget: PackedEndEffectorTarget,
        temporalContext: TemporalContext,
        proposal: PackedTemporalProposal,
        events: PackedTemporalEvent,
        invokeDrift: torch.Tensor,
    ) -> PackedTemporalDecision:
        refined = self.RefineProposal(
            temporalContext=temporalContext,
            proposal=proposal,
            invokeDrift=invokeDrift)
        return self.execution_gate.Step(
            feedback=feedbackPacket,
            candidateTarget=candidateTarget,
            cachedTarget=cachedTarget,
            proposal=refined,
            events=events,
            actionAgeSteps=temporalContext.action_age_steps)

    def forward(
        self,
        feedbackPacket: BrainFeedbackPacket,
        candidateTarget: PackedEndEffectorTarget,
        cachedTarget: PackedEndEffectorTarget,
        temporalContext: TemporalContext,
        proposal: PackedTemporalProposal,
        events: PackedTemporalEvent,
        invokeDrift: torch.Tensor,
    ) -> PackedTemporalDecision:
        return self.Step(
            feedbackPacket=feedbackPacket,
            candidateTarget=candidateTarget,
            cachedTarget=cachedTarget,
            temporalContext=temporalContext,
            proposal=proposal,
            events=events,
            invokeDrift=invokeDrift)
