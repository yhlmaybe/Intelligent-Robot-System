from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Optional, Sequence, Union, Protocol
from pathlib import Path
from dataclasses import dataclass
import threading
import random
import time
import json
import math
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
import shutil
import traceback
import os
import tempfile

from torch.utils.data import Dataset, DataLoader

from PerceptionModule import TestPerceptionMTool
from AttentionModule import  TestAttentionMTool
from MemoryModule import  TestMemoryMTool
from DecisionModule import DecisionExtractor, TestDecisionMTool
from WorldModule import ContractWorldEmbodimentAdapter
from ValueEstimationModule import (
    SensorimotorPotentialShaper,
    SmdpReturnEstimator,
    TestValueEstimationMTool)
from ConsciousnessModule import TestConsciousMTool
from IntentionModule import TestIntentionMTool
from OCRModule import TestOCRMTool, OCREngineExtractor, IouXyxy
from DataPreprocess import (
    DataPreprocessor,
    DataResizeMeta,
    OfflineGameDataset,
    OfflineOCRDataset,
    OfflineOCRRecognitionDataset,
    ValidateContractOfflineSensorManifest)
from AGICore import (
    Agent,
    BRAIN_RUNTIME_BUFFER_FIELDS,
    BRAIN_RUNTIME_SCHEMA_VERSION,
    BrainCore,
    ExportCognitiveBackboneState,
    ExportBrainModelState,
    ExportDeploymentModelState,
    IsWorldRuntimeStateKey,
    LoadBrainModelState,
    LoadCognitiveBackboneState,
    LoadDeploymentModelState,
    TestAGICoreMTool)
from Config import BasicParameters
from CoreTypes import (
    BrainBuildSpec,
    COGNITIVE_READOUT_SCHEMA_VERSION,
    CognitiveReadout,
    ContractAgentActInput,
    ContractAgentActOutput,
    ContractBrainStepInput,
    ContractOfflineBatch,
    ContractOfflineSample,
    DECISION_WIRE_SCHEMA_VERSION,
    DECISION_REQUEST_PROVENANCE_FIELDS,
    POLICY_PATH_DETAIL,
    POLICY_PATH_FAST,
    POLICY_PATH_FULL,
    SENSOR_PACKET_WIRE_FIELDS,
    SENSOR_PACKET_WIRE_SCHEMA_VERSION,
    TEXT_TRUST_OCR_OBSERVED,
    TEXT_TRUST_OPERATOR_COMMAND,
    TEXT_TRUST_UNSAFE_EXTERNAL)
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import (
    ActionExecutionResult,
    ActionRequest,
    BrainFeedbackPacket,
    PackedEndEffectorTarget,
    Robot,
    SlotExecutionStatus)


TRAIN_CHECKPOINT_SCHEMA_VERSION = BRAIN_RUNTIME_SCHEMA_VERSION
OCR_CHECKPOINT_SCHEMA_VERSION = 16

CONTRACT_TRAINING_TEST_CONFIG_FIELDS = (
    "DATA_ROOT_PATH_TEST",
    "DATA_SENSOR_MANIFEST_PATH_TEST",
    "DATA_DEPTH_PATH_TEST",
    "DATA_DEPTH_VALID_PATH_TEST",
    "DATA_FEEDBACK_PATH_TEST",
    "DATA_NORMAL_PATH_TEST",
    "DATA_SEMANTIC_SEGMENTATION_PATH_TEST",
    "DATA_INSTANCE_SEGMENTATION_PATH_TEST",
    "DATA_SYNTHETIC_SUPERVISION_PATH_TEST",
)

CONTRACT_TRAINING_TEST_LOCK = threading.Lock()

MODULE_PARAMETER_FIELDS = frozenset({
    "schema_version",
    "calibration_id",
    "model_contract_id",
    "brain",
})

TRAIN_CHECKPOINT_FIELDS = frozenset({
    "schema_version",
    "calibration_id",
    "description_id",
    "model_contract_id",
    "adapter_id",
    "world_frame_id",
    "epoch",
    "next_batch_index",
    "epoch_loss_sum",
    "best_val",
    "no_improve",
    "train_stage",
    "batch_size",
    "online_learning",
    "brain",
    "online_candidates",
    "opt_actor",
    "opt_critic",
    "opt_world",
    "train_indices",
    "val_indices",
    "test_indices",
    "processed_sample_count_total",
    "rng",
    "buffers",
    "world_memory",
    "memory_durable",
})

TRAIN_RNG_FIELDS = frozenset({
    "python",
    "torch",
    "numpy",
    "cuda_all",
})

DEPLOYMENT_MANIFEST_FIELDS = frozenset({
    "schema_version",
    "calibration_id",
    "description_id",
    "model_contract_id",
    "adapter_id",
    "generation",
    "model_path",
    "world_memory_path",
    "memory_path",
})

OCR_METADATA_FIELDS = frozenset({
    "vocab",
    "blank_index",
    "addon_cfg",
})

OCR_MODULE_PARAMETER_FIELDS = frozenset({
    "schema_version",
    "ocr",
    "ocr_meta",
})

OCR_RECOGNIZER_PARAMETER_FIELDS = frozenset({
    "schema_version",
    "recognizer",
    "ocr_meta",
})

OCR_TRAIN_CHECKPOINT_FIELDS = frozenset({
    "schema_version",
    "epoch",
    "best_val",
    "ocr",
    "ocr_meta",
    "optimizer",
    "train_indices",
    "val_indices",
    "test_indices",
    "processed_sample_count_total",
    "rng",
    "train_detection",
    "train_recognition",
})

OCR_RECOGNIZER_TRAIN_CHECKPOINT_FIELDS = frozenset({
    "schema_version",
    "epoch",
    "best_val",
    "recognizer",
    "ocr_meta",
    "optimizer",
    "train_indices",
    "val_indices",
    "test_indices",
    "processed_sample_count_total",
    "rng",
})


@dataclass(frozen=True)
class TrainingResumeState:
    epoch: int
    next_batch_index: int
    epoch_loss_sum: float
    best_val: float
    no_improve: int
    processed_sample_count_total: int
    train_dataset: Dataset
    validation_dataset: Dataset
    test_dataset: Dataset


@dataclass(frozen=True)
class EmbodiedEnvironmentTransition:
    observation: Any
    execution_result: ActionExecutionResult
    reward: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor


class EmbodiedEnvironmentProtocol(Protocol):
    def Reset(self) -> Any:
        ...

    def MaterializeObservation(
        self,
        observation: Any,
        robot: Robot,
    ) -> Any:
        ...

    def Step(
        self,
        actionRequest: ActionRequest,
    ) -> EmbodiedEnvironmentTransition:
        ...


@dataclass(frozen=True)
class AutonomousDecisionSnapshot:
    conditioning: Dict[str, torch.Tensor]
    policy: Dict[str, Any]
    value_conditioning: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class AutonomousPolicyOutput:
    action_request: ActionRequest
    value_baseline: torch.Tensor
    behavior_log_probability: torch.Tensor
    actor_credit_mask: torch.Tensor
    candidate_selected: torch.Tensor
    cache_selected: torch.Tensor
    neutral_selected: torch.Tensor
    controller_override: torch.Tensor
    sensorimotor_inconsistency: torch.Tensor
    sensorimotor_valid: torch.Tensor
    policy_snapshot: AutonomousDecisionSnapshot


class AutonomousPolicyProtocol(Protocol):
    def Reset(self, observation: Any) -> None:
        ...

    def PlannerEnabled(self) -> bool:
        ...

    def SetPlannerEnabled(self, enabled: bool) -> None:
        ...

    def Step(self, observation: Any) -> AutonomousPolicyOutput:
        ...

    def Value(self, observation: Any) -> torch.Tensor:
        ...

    def ValueAndReadout(
        self,
        observation: Any,
    ) -> Tuple[torch.Tensor, CognitiveReadout]:
        ...

    def ReevaluateValue(
        self,
        valueConditioning: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        ...

    def AfterOptimizerStep(self) -> None:
        ...


class AutonomousBrainPolicy:
    def __init__(
        self,
        agent: Agent,
    ) -> None:
        if not isinstance(agent, Agent) or not agent.is_train:
            raise TypeError("autonomous policy requires a training Agent")
        actor = agent.brain.RuntimeModule(agent.brain.actor)
        critic = agent.brain.RuntimeModule(agent.brain.critic)
        if not isinstance(actor, DecisionExtractor):
            raise TypeError("autonomous policy requires DecisionExtractor")
        if not callable(getattr(critic, "RecomputeReturnValue", None)):
            raise TypeError("autonomous policy critic cannot recompute value")
        self.agent = agent
        self.actor = actor
        self.critic = critic
        self.pending_reset = True

    def Reset(self, observation: Any) -> None:
        if type(observation) is not ContractAgentActInput:
            raise TypeError(
                "autonomous brain observation must be ContractAgentActInput")
        self.agent.ResetBrainState(int(observation.frame.size(0)))
        self.pending_reset = False

    def PlannerEnabled(self) -> bool:
        return bool(self.agent.brain.use_planner)

    def SetPlannerEnabled(self, enabled: bool) -> None:
        self.agent.brain.use_planner = bool(enabled)

    def CaptureTrainingModes(self) -> Dict[nn.Module, bool]:
        return {
            module: bool(module.training)
            for module in self.agent.brain.modules()}

    def RestoreTrainingModes(
        self,
        modes: Dict[nn.Module, bool],
    ) -> None:
        for module, training in modes.items():
            module.training = training

    def BuildOutput(self, output: Any) -> AutonomousPolicyOutput:
        if type(output.decision) is not dict:
            raise TypeError("autonomous brain decision must be a dictionary")
        decision = output.decision
        required = {
            "actor_credit_mask",
            "behavior_log_probability",
            "evaluated_policy_path",
            "packed_temporal",
            "policy_conditioning",
            "policy_path",
            "policy_snapshot",
            "value_baseline",
            "value_conditioning",
        }
        if not required.issubset(decision):
            raise ValueError("autonomous brain decision fields do not match")
        request = output.action_request
        if type(request) is not ActionRequest:
            raise TypeError("autonomous brain output requires ActionRequest")
        policy_path = decision["policy_path"]
        if not torch.equal(request.policy_path, policy_path):
            raise ValueError("autonomous request policy path does not match")
        raw_snapshot = decision["policy_snapshot"]
        conditioning = decision["policy_conditioning"]
        value_conditioning = decision["value_conditioning"]
        if type(raw_snapshot) is not dict:
            raise TypeError("autonomous policy snapshot must be a dictionary")
        if type(conditioning) is not dict or type(value_conditioning) is not dict:
            raise TypeError("autonomous conditioning must be dictionaries")
        if (
            not torch.equal(
                raw_snapshot["policyPath"],
                decision["evaluated_policy_path"])
            or not torch.equal(raw_snapshot["uRaw"], conditioning["uRaw"])
            or not torch.equal(
                raw_snapshot["optionIndex"],
                conditioning["optionIndex"])
        ):
            raise ValueError("autonomous snapshot does not match behavior policy")
        temporal = decision["packed_temporal"]
        candidate = temporal.candidate_selected | request.help_requested
        cache = temporal.cache_selected & ~request.help_requested
        neutral = ~(candidate | cache)
        readout = output.cognitive_readout
        if not isinstance(readout, CognitiveReadout):
            raise TypeError("autonomous brain output requires CognitiveReadout")
        return AutonomousPolicyOutput(
            action_request=request,
            value_baseline=decision["value_baseline"],
            behavior_log_probability=decision["behavior_log_probability"],
            actor_credit_mask=decision["actor_credit_mask"],
            candidate_selected=candidate,
            cache_selected=cache,
            neutral_selected=neutral,
            controller_override=(
                temporal.override_applied & ~request.help_requested),
            sensorimotor_inconsistency=(
                readout.sensorimotor_evidence[:, 0]),
            sensorimotor_valid=readout.sensorimotor_valid,
            policy_snapshot=AutonomousDecisionSnapshot(
                conditioning=conditioning,
                policy=raw_snapshot,
                value_conditioning=value_conditioning))

    def Step(self, observation: Any) -> AutonomousPolicyOutput:
        if type(observation) is not ContractAgentActInput:
            raise TypeError(
                "autonomous brain observation must be ContractAgentActInput")
        if not observation.sample_actions or observation.deterministic_actor:
            raise ValueError("autonomous rollout requires sampled actor actions")
        if self.pending_reset:
            raise RuntimeError("autonomous policy must be reset before rollout")
        modes = self.CaptureTrainingModes()
        try:
            output = self.agent.RunStep(
                observation,
                enableGrad=False,
                modelTraining=False)
        finally:
            self.RestoreTrainingModes(modes)
        return self.BuildOutput(output)

    def Value(self, observation: Any) -> torch.Tensor:
        modes = self.CaptureTrainingModes()
        try:
            return self.agent.EvaluateValue(observation)
        finally:
            self.RestoreTrainingModes(modes)

    def ValueAndReadout(
        self,
        observation: Any,
    ) -> Tuple[torch.Tensor, CognitiveReadout]:
        modes = self.CaptureTrainingModes()
        try:
            return self.agent.EvaluateValueAndReadout(observation)
        finally:
            self.RestoreTrainingModes(modes)

    def ReevaluateValue(
        self,
        valueConditioning: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.critic.RecomputeReturnValue(valueConditioning)

    def AfterOptimizerStep(self) -> None:
        self.agent.AfterOptimizerStep()


@dataclass(frozen=True)
class AutonomousPpoConfig:
    full_policy_path: int = POLICY_PATH_FULL
    fast_policy_path: int = POLICY_PATH_FAST
    detail_policy_path: int = POLICY_PATH_DETAIL
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatch_size: int = 32
    discount: float = 0.99
    trace_decay: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_log_ratio: float = 20.0
    potential_scale: float = 0.0
    applied_action_cost: float = 0.0
    help_request_cost: float = 0.0
    open_loop_cost: float = 0.0
    gradient_norm: float = 1.0


@dataclass
class AutonomousRolloutRecord:
    environment_index: int
    policy_path: int
    snapshot: AutonomousDecisionSnapshot
    old_log_probability: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    duration: int
    terminated: bool
    truncated: bool
    next_value: torch.Tensor
    trace_continues: bool
    help_requested: bool
    help_accepted: bool
    help_pending: bool
    applied_target_values: torch.Tensor
    applied_target_active: torch.Tensor
    applied_action_epoch: int
    advantage: torch.Tensor
    return_target: torch.Tensor
    predecessor: Optional["AutonomousRolloutRecord"] = None


class AutonomousInteractionTrainer:
    @classmethod
    def FromAgent(
        cls,
        environment: EmbodiedEnvironmentProtocol,
        agent: Agent,
        robot: Robot,
        config: Optional[AutonomousPpoConfig] = None,
    ) -> "AutonomousInteractionTrainer":
        policy = AutonomousBrainPolicy(agent)
        return cls(
            environment=environment,
            policy=policy,
            actor=policy.actor,
            robot=robot,
            actorOptimizer=agent.opt_actor,
            criticOptimizer=agent.opt_critic,
            config=(
                AutonomousPpoConfig()
                if config is None
                else config))

    def __init__(
        self,
        environment: EmbodiedEnvironmentProtocol,
        policy: AutonomousPolicyProtocol,
        actor: DecisionExtractor,
        robot: Robot,
        actorOptimizer: torch.optim.Optimizer,
        criticOptimizer: torch.optim.Optimizer,
        config: AutonomousPpoConfig,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.actor = actor
        self.robot = robot
        self.actor_optimizer = actorOptimizer
        self.critic_optimizer = criticOptimizer
        self.config = config
        self.ValidateConfiguration()
        self.potential_shaper = SensorimotorPotentialShaper(
            discount=self.config.discount,
            scale=self.config.potential_scale)

    def ValidateConfiguration(self) -> None:
        path_ids = (
            self.config.full_policy_path,
            self.config.fast_policy_path,
            self.config.detail_policy_path)
        if any(type(value) is not int or value < 0 for value in path_ids):
            raise ValueError("autonomous policy path ids must be non-negative integers")
        if len(set(path_ids)) != 3:
            raise ValueError("autonomous policy path ids must be distinct")
        if (
            self.config.rollout_steps < 1
            or self.config.ppo_epochs < 1
            or self.config.minibatch_size < 1
            or not 0.0 < self.config.discount <= 1.0
            or not 0.0 <= self.config.trace_decay <= 1.0
            or not 0.0 < self.config.clip_ratio < 1.0
            or not 0.0 < self.config.value_clip < 1.0
            or self.config.value_coefficient < 0.0
            or self.config.entropy_coefficient < 0.0
            or not math.isfinite(self.config.max_log_ratio)
            or self.config.max_log_ratio <= 0.0
            or self.config.gradient_norm <= 0.0
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (
                    self.config.potential_scale,
                    self.config.applied_action_cost,
                    self.config.help_request_cost,
                    self.config.open_loop_cost))
        ):
            raise ValueError("autonomous PPO configuration is invalid")

    def ValidateMask(
        self,
        value: torch.Tensor,
        batchSize: int,
        device: torch.device,
        name: str,
    ) -> None:
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.dtype != torch.bool
            or value.device != device
        ):
            raise ValueError(f"autonomous {name} must match the policy batch")

    def ValidatePolicyOutput(
        self,
        output: AutonomousPolicyOutput,
    ) -> int:
        if type(output) is not AutonomousPolicyOutput:
            raise TypeError("autonomous policy must return AutonomousPolicyOutput")
        request = output.action_request
        if type(request) is not ActionRequest:
            raise TypeError("autonomous policy output requires ActionRequest")
        request.Validate(self.robot.ContractView)
        batch_size = int(request.target.values.size(0))
        device = request.target.values.device
        if (
            not torch.is_tensor(output.value_baseline)
            or tuple(output.value_baseline.shape) != (batch_size,)
            or output.value_baseline.device != device
            or not output.value_baseline.is_floating_point()
            or not bool(torch.isfinite(output.value_baseline).all().item())
        ):
            raise ValueError("autonomous value baseline must match the policy batch")
        if (
            not torch.is_tensor(output.behavior_log_probability)
            or tuple(output.behavior_log_probability.shape) != (batch_size,)
            or output.behavior_log_probability.device != device
            or not output.behavior_log_probability.is_floating_point()
            or not bool(torch.isfinite(
                output.behavior_log_probability).all().item())
        ):
            raise ValueError("autonomous behavior log probability is invalid")
        if (
            not torch.is_tensor(output.sensorimotor_inconsistency)
            or tuple(output.sensorimotor_inconsistency.shape) != (batch_size,)
            or output.sensorimotor_inconsistency.device != device
            or not output.sensorimotor_inconsistency.is_floating_point()
            or not bool(torch.isfinite(
                output.sensorimotor_inconsistency).all().item())
        ):
            raise ValueError("autonomous sensorimotor inconsistency is invalid")
        masks = {
            "actor credit": output.actor_credit_mask,
            "candidate selection": output.candidate_selected,
            "cache selection": output.cache_selected,
            "neutral selection": output.neutral_selected,
            "controller override": output.controller_override,
            "sensorimotor validity": output.sensorimotor_valid}
        for name, value in masks.items():
            self.ValidateMask(value, batch_size, device, name)
        selection_count = (
            output.candidate_selected.to(dtype=torch.int8)
            + output.cache_selected.to(dtype=torch.int8)
            + output.neutral_selected.to(dtype=torch.int8))
        if bool(selection_count.ne(1).any().item()):
            raise ValueError("autonomous temporal selections must be exclusive and complete")
        if bool((output.actor_credit_mask & ~output.candidate_selected).any().item()):
            raise ValueError("an autonomous actor request must be a new candidate")
        if bool(request.planner_override.any().item()):
            raise RuntimeError("CEM must be disabled during autonomous rollout")
        valid_paths = torch.zeros_like(request.policy_path, dtype=torch.bool)
        for path_id in (
            self.config.full_policy_path,
            self.config.fast_policy_path,
            self.config.detail_policy_path,
        ):
            valid_paths |= request.policy_path.eq(path_id)
        if bool((output.actor_credit_mask & ~valid_paths).any().item()):
            raise ValueError("autonomous actor request has an unknown policy path")
        snapshot = output.policy_snapshot
        if type(snapshot) is not AutonomousDecisionSnapshot:
            raise TypeError("autonomous output requires a decision snapshot")
        if set(snapshot.conditioning) != {
            "uRaw",
            "mu",
            "logvar",
            "optionLogits",
            "optionIndex",
        }:
            raise ValueError("autonomous policy conditioning fields do not match")
        if set(snapshot.value_conditioning) != {
            "valueHidden",
            "valueBaseline",
        }:
            raise ValueError("autonomous value conditioning fields do not match")
        batched_values = tuple(snapshot.conditioning.values()) + tuple(
            snapshot.value_conditioning.values())
        if any(
            not torch.is_tensor(value)
            or value.dim() < 1
            or int(value.size(0)) != batch_size
            or value.device != device
            for value in batched_values
        ):
            raise ValueError("autonomous decision snapshot must match the policy batch")
        conditioned = self.actor.RecomputeActionLogProbability(
            snapshot.conditioning)["combinedActionLogProbability"]
        replayed = self.actor.RecomputePolicySnapshot(snapshot.policy)
        snapshot_path_matches = snapshot.policy["policyPath"].eq(
            request.policy_path)
        if (
            bool((output.actor_credit_mask & ~snapshot_path_matches).any().item())
            or bool((output.actor_credit_mask & ~replayed["valid"]).any().item())
        ):
            raise ValueError("autonomous policy snapshot path does not match")
        if not torch.allclose(
            output.behavior_log_probability,
            conditioned,
            atol=1e-6,
            rtol=1e-6,
        ) or not torch.allclose(
            output.behavior_log_probability[output.actor_credit_mask],
            replayed["combinedActionLogProbability"][
                output.actor_credit_mask],
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("autonomous behavior probability does not match conditioning")
        return batch_size

    def ValidateEnvironmentTransition(
        self,
        transition: EmbodiedEnvironmentTransition,
        request: ActionRequest,
        batchSize: int,
    ) -> None:
        if type(transition) is not EmbodiedEnvironmentTransition:
            raise TypeError("environment must return EmbodiedEnvironmentTransition")
        result = transition.execution_result
        if type(result) is not ActionExecutionResult:
            raise TypeError("environment transition requires ActionExecutionResult")
        result.Validate(request, self.robot.ContractView)
        device = request.target.values.device
        if (
            not torch.is_tensor(transition.reward)
            or tuple(transition.reward.shape) != (batchSize,)
            or transition.reward.device != device
            or not transition.reward.is_floating_point()
            or not bool(torch.isfinite(transition.reward).all().item())
        ):
            raise ValueError("environment reward must be a finite policy-batch vector")
        self.ValidateMask(
            transition.terminated,
            batchSize,
            device,
            "terminated")
        self.ValidateMask(
            transition.truncated,
            batchSize,
            device,
            "truncated")
        if bool((transition.terminated & transition.truncated).any().item()):
            raise ValueError("one environment row cannot terminate and truncate together")

    def IndexSnapshot(
        self,
        snapshot: AutonomousDecisionSnapshot,
        rowIndex: int,
    ) -> AutonomousDecisionSnapshot:
        index = torch.tensor(
            [rowIndex],
            device=snapshot.conditioning["uRaw"].device,
            dtype=torch.long)
        batch_size = int(snapshot.conditioning["uRaw"].size(0))
        return AutonomousDecisionSnapshot(
            conditioning={
                name: value.index_select(0, index).detach().clone()
                for name, value in snapshot.conditioning.items()},
            policy=self.IndexPolicyValue(
                snapshot.policy,
                index,
                batch_size),
            value_conditioning={
                name: value.index_select(0, index).detach().clone()
                for name, value in snapshot.value_conditioning.items()})

    def IndexPolicyValue(
        self,
        value: Any,
        rowIndex: torch.Tensor,
        batchSize: int,
    ) -> Any:
        if torch.is_tensor(value):
            selected = (
                value.index_select(0, rowIndex)
                if value.dim() > 0 and int(value.size(0)) == batchSize
                else value)
            return selected.detach().clone()
        if type(value) is dict:
            return {
                name: self.IndexPolicyValue(item, rowIndex, batchSize)
                for name, item in value.items()}
        return value

    def CreateRecord(
        self,
        output: AutonomousPolicyOutput,
        result: ActionExecutionResult,
        rowIndex: int,
        predecessor: Optional[AutonomousRolloutRecord],
    ) -> AutonomousRolloutRecord:
        snapshot = self.IndexSnapshot(output.policy_snapshot, rowIndex)
        old_probability = output.behavior_log_probability[rowIndex]
        request = output.action_request
        return AutonomousRolloutRecord(
            environment_index=rowIndex,
            policy_path=int(request.policy_path[rowIndex].item()),
            snapshot=snapshot,
            old_log_probability=old_probability.detach().clone(),
            value=output.value_baseline[rowIndex].detach().clone(),
            reward=output.value_baseline.new_zeros(()),
            duration=0,
            terminated=False,
            truncated=False,
            next_value=output.value_baseline.new_zeros(()),
            trace_continues=False,
            help_requested=bool(request.help_requested[rowIndex].item()),
            help_accepted=bool(result.help_accepted[rowIndex].item()),
            help_pending=bool(
                request.help_requested[rowIndex].item()
                and not result.help_accepted[rowIndex].item()),
            applied_target_values=self.robot.CanonicalizeTarget(
                result.applied_target)[rowIndex].detach().clone(),
            applied_target_active=result.applied_target.active[
                rowIndex].detach().clone(),
            applied_action_epoch=int(result.action_epoch[rowIndex].item()),
            advantage=output.value_baseline.new_zeros(()),
            return_target=output.value_baseline.new_zeros(()),
            predecessor=predecessor)

    def AddReward(
        self,
        record: AutonomousRolloutRecord,
        reward: torch.Tensor,
    ) -> None:
        discount = self.config.discount ** record.duration
        record.reward = record.reward + discount * reward.detach()
        record.duration += 1

    def FinalizeRecord(
        self,
        record: AutonomousRolloutRecord,
        nextValue: torch.Tensor,
        terminated: bool,
        truncated: bool,
        records: List[AutonomousRolloutRecord],
    ) -> None:
        if record.duration < 1:
            raise RuntimeError("an autonomous option record has no executed step")
        record.terminated = bool(terminated)
        record.truncated = bool(truncated)
        record.next_value = (
            nextValue.new_zeros(())
            if record.terminated
            else nextValue.detach().clone())
        records.append(record)

    def ComputeAdvantages(
        self,
        records: List[AutonomousRolloutRecord],
    ) -> None:
        for environment_index in sorted({
            record.environment_index for record in records
        }):
            sequence = [
                record for record in records
                if record.environment_index == environment_index]
            start = 0
            while start < len(sequence):
                end = start + 1
                while (
                    end < len(sequence)
                    and sequence[end - 1].trace_continues
                ):
                    end += 1
                chain = sequence[start:end]
                option_reward = torch.stack([
                    record.reward for record in chain]).unsqueeze(-1)
                values = torch.stack(
                    [record.value for record in chain]
                    + [chain[-1].next_value]).unsqueeze(-1)
                duration = torch.tensor(
                    [record.duration for record in chain],
                    device=option_reward.device,
                    dtype=option_reward.dtype).unsqueeze(-1)
                terminated = torch.tensor(
                    [record.terminated for record in chain],
                    device=option_reward.device,
                    dtype=torch.bool).unsqueeze(-1)
                truncated = torch.tensor(
                    [record.truncated for record in chain],
                    device=option_reward.device,
                    dtype=torch.bool).unsqueeze(-1)
                estimate = SmdpReturnEstimator.EstimateAdvantages(
                    option_reward,
                    values,
                    duration,
                    terminated,
                    truncated,
                    discount=self.config.discount,
                    traceDecay=self.config.trace_decay)
                for index, record in enumerate(chain):
                    record.advantage = estimate.advantage[index, 0]
                    record.return_target = estimate.returnTarget[index, 0]
                start = end

    def NormalizeAppliedTarget(
        self,
        values: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        lower = values.new_tensor(
            self.robot.ContractView.end_effector_target_lower).unsqueeze(0)
        upper = values.new_tensor(
            self.robot.ContractView.end_effector_target_upper).unsqueeze(0)
        normalized = 2.0 * (values - lower) / (upper - lower) - 1.0
        masked = torch.zeros_like(normalized)
        for slot_index in range(self.robot.ContractView.end_effector_count):
            target_slice = self.robot.ContractView.end_effector_target_layout.Slice(
                slot_index)
            masked[:, target_slice] = torch.where(
                active[:, slot_index].unsqueeze(-1),
                normalized[:, target_slice],
                torch.zeros_like(normalized[:, target_slice]))
        return masked

    def CausalExecutionMasks(
        self,
        request: ActionRequest,
        result: ActionExecutionResult,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        relevant = request.target.active | result.applied_target.active
        any_relevant = relevant.any(dim=-1)
        known = torch.where(
            any_relevant,
            (result.execution_known | ~relevant).all(dim=-1),
            result.execution_known.all(dim=-1))
        modified = result.execution_status.eq(
            int(SlotExecutionStatus.MODIFIED))
        overridden = torch.where(
            any_relevant,
            (modified & relevant).any(dim=-1),
            modified.any(dim=-1))
        return known, overridden

    def TargetsMatch(
        self,
        request: ActionRequest,
        result: ActionExecutionResult,
    ) -> torch.Tensor:
        requested_values = self.robot.CanonicalizeTarget(request.target)
        applied_values = self.robot.CanonicalizeTarget(
            result.applied_target)
        return self.robot.ContractView.TargetRowsMatch(
            requested_values,
            request.target.active,
            applied_values,
            result.applied_target.active)

    def ExecutionMatchesRequest(
        self,
        request: ActionRequest,
        result: ActionExecutionResult,
    ) -> torch.Tensor:
        stop_confirmed = (
            request.stop_requested
            & ~result.applied_target.active.any(dim=-1)
            & result.execution_status.eq(
                int(SlotExecutionStatus.STOPPED)).all(dim=-1))
        active_applied = (
            ~request.target.active
            | result.execution_status.eq(
                int(SlotExecutionStatus.APPLIED))).all(dim=-1)
        execution_semantics = torch.where(
            request.stop_requested,
            stop_confirmed,
            active_applied)
        return self.TargetsMatch(request, result) & execution_semantics

    def ExecutionContinuesRequest(
        self,
        request: ActionRequest,
        result: ActionExecutionResult,
    ) -> torch.Tensor:
        stop_confirmed = (
            request.stop_requested
            & ~result.applied_target.active.any(dim=-1)
            & result.execution_status.eq(
                int(SlotExecutionStatus.STOPPED)).all(dim=-1))
        active_continued = (
            ~request.target.active
            | result.execution_status.eq(
                int(SlotExecutionStatus.APPLIED))
            | result.execution_status.eq(
                int(SlotExecutionStatus.HELD))).all(dim=-1)
        execution_semantics = torch.where(
            request.stop_requested,
            stop_confirmed,
            active_continued)
        return self.TargetsMatch(request, result) & execution_semantics

    def RequestStartsExecution(
        self,
        output: AutonomousPolicyOutput,
        result: ActionExecutionResult,
    ) -> torch.Tensor:
        request = output.action_request
        known, modified = self.CausalExecutionMasks(request, result)
        candidate = output.candidate_selected & ~result.hard_stop
        standard = (
            candidate
            & ~request.help_requested
            & request.command_active
            & request.target.active.any(dim=-1)
            & known
            & ~modified
            & self.ExecutionMatchesRequest(request, result))
        help_execution = (
            candidate
            & request.help_requested
            & output.actor_credit_mask)
        return standard | help_execution

    def ActorOwnsExecution(
        self,
        output: AutonomousPolicyOutput,
        startsExecution: torch.Tensor,
    ) -> torch.Tensor:
        return (
            startsExecution
            & output.actor_credit_mask
            & ~output.neutral_selected
            & ~output.controller_override
            & ~output.action_request.planner_override)

    def RecordTargetMatches(
        self,
        record: AutonomousRolloutRecord,
        result: ActionExecutionResult,
        rowIndex: int,
    ) -> bool:
        applied_values = self.robot.CanonicalizeTarget(
            result.applied_target)[rowIndex]
        return bool(self.robot.ContractView.TargetRowsMatch(
            record.applied_target_values.unsqueeze(0),
            record.applied_target_active.unsqueeze(0),
            applied_values.unsqueeze(0),
            result.applied_target.active[rowIndex].unsqueeze(0)).item())

    def RecordExecutionContinues(
        self,
        record: AutonomousRolloutRecord,
        request: ActionRequest,
        result: ActionExecutionResult,
        rowIndex: int,
    ) -> bool:
        if bool(result.hard_stop[rowIndex].item()):
            return False
        if record.help_requested:
            if not bool(request.help_requested[rowIndex].item()):
                return False
            record.help_accepted = (
                record.help_accepted
                or bool(result.help_accepted[rowIndex].item()))
            record.help_pending = not record.help_accepted
            return True
        if bool(request.help_requested[rowIndex].item()):
            return False
        if not self.RecordTargetMatches(record, result, rowIndex):
            return False
        relevant = (
            record.applied_target_active
            | request.target.active[rowIndex]
            | result.applied_target.active[rowIndex])
        status = result.execution_status[rowIndex]
        allowed = (
            status.eq(int(SlotExecutionStatus.APPLIED))
            | status.eq(int(SlotExecutionStatus.REJECTED))
            | status.eq(int(SlotExecutionStatus.HELD)))
        return bool((result.execution_known[rowIndex] & allowed | ~relevant).all().item())

    def OpenLoopCost(
        self,
        normalizedApplied: torch.Tensor,
        active: torch.Tensor,
        observation: Any,
    ) -> torch.Tensor:
        if self.config.open_loop_cost == 0.0:
            return normalizedApplied.new_zeros(normalizedApplied.size(0))
        if type(observation) is not ContractAgentActInput:
            raise TypeError("open-loop cost requires ContractAgentActInput")
        feedback = observation.feedback_packet
        slot_magnitude = normalizedApplied.new_zeros(active.shape)
        for slot_index in range(self.robot.ContractView.end_effector_count):
            target_slice = self.robot.ContractView.end_effector_target_layout.Slice(
                slot_index)
            slot_magnitude[:, slot_index] = torch.linalg.vector_norm(
                normalizedApplied[:, target_slice],
                dim=-1) / math.sqrt(target_slice.stop - target_slice.start)
        dependency = active.to(dtype=normalizedApplied.dtype)
        missing = (~feedback.endpoint_present).to(
            dtype=normalizedApplied.dtype)
        weighted = (
            dependency
            * missing
            * feedback.observation_age
            * slot_magnitude)
        return weighted.sum(dim=-1) / dependency.sum(
            dim=-1).clamp_min(1.0)

    def PrimePotential(self, output: AutonomousPolicyOutput) -> None:
        reference = output.sensorimotor_inconsistency
        self.potential_shaper(
            reference.new_zeros(reference.shape),
            reference.unsqueeze(-1),
            output.sensorimotor_valid,
            reference.new_zeros(reference.shape),
            torch.zeros_like(output.sensorimotor_valid),
            torch.zeros_like(output.sensorimotor_valid))

    def BuildInteractionReward(
        self,
        output: AutonomousPolicyOutput,
        transition: EmbodiedEnvironmentTransition,
        nextReadout: Optional[CognitiveReadout],
        previousApplied: torch.Tensor,
        nextObservation: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        result = transition.execution_result
        execution_known, _ = self.CausalExecutionMasks(
            output.action_request,
            result)
        current_applied = self.NormalizeAppliedTarget(
            result.applied_target.values,
            result.applied_target.active)
        action_cost = (current_applied - previousApplied).square().mean(dim=-1)
        action_cost = torch.where(
            execution_known,
            action_cost,
            torch.zeros_like(action_cost))
        open_loop_cost = self.OpenLoopCost(
            current_applied,
            result.applied_target.active,
            nextObservation)
        open_loop_cost = torch.where(
            execution_known,
            open_loop_cost,
            torch.zeros_like(open_loop_cost))
        reward = transition.reward
        if self.config.potential_scale > 0.0:
            if nextReadout is None and bool(transition.terminated.all().item()):
                next_inconsistency = reward.new_zeros(
                    reward.size(0),
                    1)
                next_valid = torch.zeros_like(transition.terminated)
            elif isinstance(nextReadout, CognitiveReadout):
                next_inconsistency = torch.where(
                    transition.terminated.unsqueeze(-1),
                    torch.zeros_like(
                        nextReadout.sensorimotor_evidence[:, :1]),
                    nextReadout.sensorimotor_evidence[:, :1])
                next_valid = (
                    nextReadout.sensorimotor_valid
                    & ~transition.terminated)
            else:
                raise TypeError("potential shaping requires CognitiveReadout")
            reward = self.potential_shaper(
                reward,
                next_inconsistency,
                next_valid,
                reward.new_ones(reward.shape),
                transition.terminated,
                transition.truncated).shapedReward
        help_cost = output.action_request.help_requested.to(
            dtype=reward.dtype)
        reward = (
            reward
            - self.config.applied_action_cost * action_cost
            - self.config.help_request_cost * help_cost
            - self.config.open_loop_cost * open_loop_cost)
        next_applied = torch.where(
            execution_known.unsqueeze(-1),
            current_applied,
            previousApplied)
        return reward, next_applied

    def CollectRollout(
        self,
        stepCount: Optional[int] = None,
    ) -> List[AutonomousRolloutRecord]:
        steps = self.config.rollout_steps if stepCount is None else int(stepCount)
        if steps < 1:
            raise ValueError("autonomous rollout step count must be positive")
        records: List[AutonomousRolloutRecord] = []
        active: List[Optional[AutonomousRolloutRecord]] = []
        planner_enabled = self.policy.PlannerEnabled()
        actor_training = bool(self.actor.training)
        eligibility_frozen = self.actor.EligibilityFrozen()
        eligibility_state = self.actor.ExportEligibilityState()
        eligibility_batch_size = int(eligibility_state["trace"].size(0))
        try:
            self.policy.SetPlannerEnabled(False)
            self.robot.Reset()
            observation = self.environment.MaterializeObservation(
                self.environment.Reset(),
                self.robot)
            self.policy.Reset(observation)
            self.potential_shaper.ResetState()
            previous_applied = (
                self.NormalizeAppliedTarget(
                    observation.feedback_packet.applied_target_values,
                    observation.feedback_packet.applied_target_active)
                if type(observation) is ContractAgentActInput
                else None)
            potential_primed = False
            self.actor.eval()
            self.actor.SetEligibilityFrozen(True)
            for step_index in range(steps):
                output = self.policy.Step(observation)
                batch_size = self.ValidatePolicyOutput(output)
                if self.config.potential_scale > 0.0 and not potential_primed:
                    self.PrimePotential(output)
                    potential_primed = True
                if not active:
                    active = [None for _ in range(batch_size)]
                elif len(active) != batch_size:
                    raise ValueError("autonomous environment batch size changed")
                transition = self.environment.Step(output.action_request)
                self.ValidateEnvironmentTransition(
                    transition,
                    output.action_request,
                    batch_size)
                self.robot.CommitAppliedTarget(
                    output.action_request,
                    transition.execution_result)
                next_observation = self.environment.MaterializeObservation(
                    transition.observation,
                    self.robot)
                result = transition.execution_result
                evaluated_next_value: Optional[torch.Tensor] = None
                next_readout: Optional[CognitiveReadout] = None
                if (
                    self.config.potential_scale > 0.0
                    and not bool(transition.terminated.all().item())
                ):
                    evaluated_next_value, next_readout = (
                        self.policy.ValueAndReadout(next_observation))
                if previous_applied is None:
                    previous_applied = result.applied_target.values.new_zeros(
                        result.applied_target.values.shape)
                effective_reward, previous_applied = self.BuildInteractionReward(
                    output,
                    transition,
                    next_readout,
                    previous_applied,
                    next_observation)
                starts_execution = self.RequestStartsExecution(
                    output,
                    result)
                actor_owns_execution = self.ActorOwnsExecution(
                    output,
                    starts_execution)
                for row_index in range(batch_size):
                    current_record = active[row_index]
                    if (
                        current_record is not None
                        and current_record.help_requested
                        and bool(output.action_request.help_requested[
                            row_index].item())
                        and self.RecordExecutionContinues(
                            current_record,
                            output.action_request,
                            result,
                            row_index)
                    ):
                        self.AddReward(
                            current_record,
                            effective_reward[row_index])
                        continue
                    if bool(starts_execution[row_index].item()):
                        predecessor = current_record
                        if predecessor is not None:
                            self.FinalizeRecord(
                                predecessor,
                                output.value_baseline[row_index],
                                False,
                                False,
                                records)
                        active[row_index] = None
                        if not bool(actor_owns_execution[row_index].item()):
                            continue
                        record = self.CreateRecord(
                            output,
                            result,
                            row_index,
                            predecessor)
                        if predecessor is not None:
                            predecessor.trace_continues = True
                        self.AddReward(
                            record,
                            effective_reward[row_index])
                        active[row_index] = record
                        continue
                    if current_record is None:
                        continue
                    if self.RecordExecutionContinues(
                        current_record,
                        output.action_request,
                        result,
                        row_index,
                    ):
                        self.AddReward(
                            current_record,
                            effective_reward[row_index])
                    else:
                        self.FinalizeRecord(
                            current_record,
                            output.value_baseline[row_index],
                            False,
                            False,
                            records)
                        active[row_index] = None
                observation = next_observation
                environment_end = transition.terminated | transition.truncated
                rollout_end = bool(environment_end.any().item()) or step_index + 1 == steps
                if rollout_end:
                    next_value = (
                        evaluated_next_value
                        if evaluated_next_value is not None
                        else (
                            output.value_baseline.new_zeros(batch_size)
                            if bool(transition.terminated.all().item())
                            else self.policy.Value(observation)))
                    next_value = torch.where(
                        transition.terminated,
                        output.value_baseline.new_zeros(batch_size),
                        next_value)
                    if (
                        not torch.is_tensor(next_value)
                        or tuple(next_value.shape) != (batch_size,)
                        or next_value.device != output.value_baseline.device
                        or not bool(torch.isfinite(next_value).all().item())
                    ):
                        raise ValueError("autonomous bootstrap value is invalid")
                    for row_index in range(batch_size):
                        if active[row_index] is None:
                            continue
                        terminated = bool(transition.terminated[row_index].item())
                        truncated = bool(transition.truncated[row_index].item())
                        if not terminated and not truncated:
                            truncated = True
                        self.FinalizeRecord(
                            active[row_index],
                            next_value[row_index],
                            terminated,
                            truncated,
                            records)
                        active[row_index] = None
                    break
        finally:
            self.actor.ImportEligibilityState(
                eligibility_state,
                eligibility_batch_size)
            self.actor.SetEligibilityFrozen(eligibility_frozen)
            self.actor.train(actor_training)
            self.policy.SetPlannerEnabled(planner_enabled)
        self.ComputeAdvantages(records)
        return records

    def ConcatenatePolicyValues(self, values: Sequence[Any]) -> Any:
        first = values[0]
        if torch.is_tensor(first):
            return torch.cat(list(values), dim=0)
        if type(first) is dict:
            if any(type(value) is not dict or set(value) != set(first) for value in values):
                raise ValueError("autonomous policy snapshot structures do not match")
            return {
                name: self.ConcatenatePolicyValues([
                    value[name] for value in values])
                for name in first}
        if any(value != first for value in values[1:]):
            raise ValueError("autonomous policy snapshot constants do not match")
        return first

    def ReevaluatePolicy(
        self,
        records: Sequence[AutonomousRolloutRecord],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int]]:
        if len(records) < 1:
            raise ValueError("autonomous policy reevaluation requires records")
        path_counts = {
            path_id: sum(
                int(record.policy_path == path_id)
                for record in records)
            for path_id in (
                self.config.full_policy_path,
                self.config.fast_policy_path,
                self.config.detail_policy_path)}
        snapshot = self.ConcatenatePolicyValues([
            record.snapshot.policy for record in records])
        replayed = self.actor.RecomputePolicySnapshot(snapshot)
        if not bool(replayed["valid"].all().item()):
            raise RuntimeError("autonomous policy path reevaluation is incomplete")
        return (
            replayed["combinedActionLogProbability"],
            replayed["entropy"],
            path_counts)

    def OptimizerParameters(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> List[nn.Parameter]:
        return [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad]

    def StateTensorsFinite(self, value: Any) -> bool:
        if torch.is_tensor(value):
            return bool(
                not (value.is_floating_point() or value.is_complex())
                or torch.isfinite(value).all().item())
        if type(value) is dict:
            return all(self.StateTensorsFinite(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return all(self.StateTensorsFinite(item) for item in value)
        return True

    def OptimizePpo(
        self,
        records: List[AutonomousRolloutRecord],
    ) -> Dict[str, float]:
        if len(records) < 1:
            return {
                "sample_count": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "clip_fraction": 0.0,
                "full_count": 0.0,
                "fast_count": 0.0,
                "detail_count": 0.0,
                "help_acceptance": 0.0}
        advantages = torch.stack([
            record.advantage for record in records])
        if not bool(torch.isfinite(advantages).all().item()):
            raise ValueError("autonomous PPO advantages must be finite")
        if len(records) > 1:
            advantages = (
                advantages - advantages.mean()
            ) / advantages.std(unbiased=False).clamp_min(1e-6)
        normalized_advantages = {
            id(record): advantages[index].detach()
            for index, record in enumerate(records)}
        eligibility_state = self.actor.ExportEligibilityState()
        eligibility_batch_size = int(eligibility_state["trace"].size(0))
        eligibility_frozen = self.actor.EligibilityFrozen()
        actor_training = bool(self.actor.training)
        actor_parameters = self.OptimizerParameters(self.actor_optimizer)
        critic_parameters = self.OptimizerParameters(self.critic_optimizer)
        parameter_state = []
        parameter_ids = set()
        for parameter in actor_parameters + critic_parameters:
            if id(parameter) not in parameter_ids:
                parameter_ids.add(id(parameter))
                parameter_state.append((parameter, parameter.detach().clone()))
        actor_optimizer_state = copy.deepcopy(
            self.actor_optimizer.state_dict())
        critic_optimizer_state = copy.deepcopy(
            self.critic_optimizer.state_dict())
        self.actor.SetEligibilityFrozen(True)
        self.actor.eval()
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "clip_fraction": 0.0}
        path_totals = {
            self.config.full_policy_path: 0,
            self.config.fast_policy_path: 0,
            self.config.detail_policy_path: 0}
        update_count = 0
        try:
            for _ in range(self.config.ppo_epochs):
                order = torch.randperm(len(records)).tolist()
                for start in range(0, len(records), self.config.minibatch_size):
                    selected = [
                        records[index]
                        for index in order[
                            start:start + self.config.minibatch_size]]
                    new_log_probability, entropy, path_counts = (
                        self.ReevaluatePolicy(selected))
                    old_log_probability = torch.stack([
                        record.old_log_probability
                        for record in selected]).to(new_log_probability.device)
                    advantage = torch.stack([
                        normalized_advantages[id(record)]
                        for record in selected]).to(new_log_probability.device)
                    return_target = torch.stack([
                        record.return_target
                        for record in selected]).to(new_log_probability.device)
                    old_value = torch.stack([
                        record.value
                        for record in selected]).to(new_log_probability.device)
                    value_conditioning = {
                        name: torch.cat([
                            record.snapshot.value_conditioning[name]
                            for record in selected], dim=0)
                        for name in ("valueHidden", "valueBaseline")}
                    current_value = self.policy.ReevaluateValue(
                        value_conditioning).reshape(-1)
                    if tuple(current_value.shape) != tuple(old_value.shape):
                        raise ValueError(
                            "autonomous reevaluated value shape does not match")
                    if not all(bool(torch.isfinite(value).all().item()) for value in (
                            new_log_probability,
                            old_log_probability,
                            entropy,
                            advantage,
                            return_target,
                            old_value,
                            current_value)):
                        raise FloatingPointError(
                            "autonomous PPO minibatch is non-finite")
                    log_ratio = new_log_probability - old_log_probability
                    ratio = torch.exp(log_ratio.clamp(
                        -self.config.max_log_ratio,
                        self.config.max_log_ratio))
                    unclipped = ratio * advantage
                    clipped = ratio.clamp(
                        1.0 - self.config.clip_ratio,
                        1.0 + self.config.clip_ratio) * advantage
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_clipped = old_value + (
                        current_value - old_value).clamp(
                            -self.config.value_clip,
                            self.config.value_clip)
                    value_loss = 0.5 * torch.maximum(
                        (current_value - return_target).square(),
                        (value_clipped - return_target).square()).mean()
                    entropy_mean = entropy.mean()
                    loss = (
                        policy_loss
                        + self.config.value_coefficient * value_loss
                        - self.config.entropy_coefficient * entropy_mean)
                    if not all(bool(torch.isfinite(value).item()) for value in (
                        policy_loss,
                        value_loss,
                        entropy_mean,
                        loss,
                    )):
                        raise FloatingPointError(
                            "autonomous PPO objective is non-finite")
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor_parameters,
                        self.config.gradient_norm,
                        error_if_nonfinite=True)
                    torch.nn.utils.clip_grad_norm_(
                        critic_parameters,
                        self.config.gradient_norm,
                        error_if_nonfinite=True)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()
                    self.policy.AfterOptimizerStep()
                    if (
                        not all(bool(torch.isfinite(
                            parameter).all().item()) for parameter, _ in parameter_state)
                        or not self.StateTensorsFinite(
                            self.actor_optimizer.state_dict())
                        or not self.StateTensorsFinite(
                            self.critic_optimizer.state_dict())
                    ):
                        raise FloatingPointError(
                            "autonomous PPO update is non-finite")
                    totals["policy_loss"] += float(policy_loss.detach().item())
                    totals["value_loss"] += float(value_loss.detach().item())
                    totals["entropy"] += float(entropy_mean.detach().item())
                    totals["clip_fraction"] += float(
                        ratio.detach().sub(1.0).abs().gt(
                            self.config.clip_ratio).to(
                                dtype=ratio.dtype).mean().item())
                    for path_id, count in path_counts.items():
                        path_totals[path_id] += count
                    update_count += 1
        except Exception:
            with torch.no_grad():
                for parameter, value in parameter_state:
                    parameter.copy_(value)
            self.actor_optimizer.load_state_dict(actor_optimizer_state)
            self.critic_optimizer.load_state_dict(critic_optimizer_state)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            self.actor.ImportEligibilityState(
                eligibility_state,
                eligibility_batch_size)
            self.actor.SetEligibilityFrozen(eligibility_frozen)
            self.actor.train(actor_training)
        help_requested = sum(
            1 for record in records if record.help_requested)
        help_accepted = sum(
            1 for record in records
            if record.help_requested and record.help_accepted)
        denominator = float(max(1, update_count))
        return {
            "sample_count": float(len(records)),
            "policy_loss": totals["policy_loss"] / denominator,
            "value_loss": totals["value_loss"] / denominator,
            "entropy": totals["entropy"] / denominator,
            "clip_fraction": totals["clip_fraction"] / denominator,
            "full_count": float(path_totals[self.config.full_policy_path]),
            "fast_count": float(path_totals[self.config.fast_policy_path]),
            "detail_count": float(path_totals[self.config.detail_policy_path]),
            "help_acceptance": (
                float(help_accepted) / float(max(1, help_requested)))}

    def TrainIteration(self) -> Dict[str, float]:
        records = self.CollectRollout()
        return self.OptimizePpo(records)

try:
    import imageio.v3 as iio
except Exception:
    iio = None


class SequentialTrajectoryLoader:
    def __init__(
        self,
        dataset: Dataset,
        *,
        batchSize: int,
        collateFn: Optional[Callable[[Sequence[Any]], Any]] = None,
    ):
        self.dataset = dataset
        self.batch_size = int(batchSize)
        self.collate_fn = collateFn
        if self.batch_size <= 0:
            raise ValueError("batchSize must be positive")
        dataset_size = len(dataset)
        if dataset_size < self.batch_size:
            raise ValueError(
                f"sequential split has {dataset_size} frames, fewer than "
                f"batchSize={self.batch_size}; it cannot form one recurrent batch")
        if dataset_size % self.batch_size != 0:
            raise ValueError(
                f"sequential split has {dataset_size} frames, which is not divisible "
                f"by batchSize={self.batch_size}; equal-length recurrent streams "
                "cannot consume a partial batch")
        self.steps = dataset_size // self.batch_size

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        return self.IterFrom(0)

    def IterFrom(self, startStep: int):
        if type(startStep) is not int or not (0 <= startStep <= self.steps):
            raise ValueError("startStep must be within the sequential loader")
        for t in range(startStep, self.steps):
            items = [self.dataset[j * self.steps + t] for j in range(self.batch_size)]
            if self.collate_fn is None:
                yield torch.utils.data.default_collate(items)
            else:
                yield self.collate_fn(items)


class ModuleController:
    def __init__(self):
        self._lock = threading.Lock()
        self.status: Dict[str, Any] = {
            "state": "idle",
            "epoch": 0, "total_epochs": 0,
            "batch": 0, "total_batches": 0,
            "train_loss": 0.0, "val_loss": 0.0,
            "message": "Waiting to start",
            "trace": "",
            "visual": self.EmptyVisualStatus(),}
        self.parameter_receiver: Dict[str, Any] = self.EmptyParameterReceiver()
        self.stop_requested = False
        self.pause_requested = False
        self.reset_hebbian = False
        self.visual_state_enabled = True

    def EmptyVisualStatus(self, *, touch: bool = False) -> Dict[str, Any]:
        return {
            "bitmap": [],
            "text": "",
            "ocr_texts": [],
            "items": [],
            "updated_at": (time.time() if touch else 0.0),}

    def EmptyParameterReceiver(self) -> Dict[str, Any]:
        return {
            "reward": None,
            "done": None,
            "textExt": None,}

    def SetStatus(self, state: str, message: str, **kwargs):
        with self._lock:
            self.status["state"] = state
            self.status["message"] = message
            for k, v in kwargs.items():
                if k in self.status:
                    self.status[k] = v

    def GetStatus(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.status)
        
    def SetParameterReceiver(self, reward = None, done = None, textExt = None):
        with self._lock:
            self.parameter_receiver["reward"] = reward
            self.parameter_receiver["done"] = done
            self.parameter_receiver["textExt"] = textExt

    def GetParameterReceiver(self) -> Dict[str, Any]:
        with self._lock:
            current = dict(self.parameter_receiver)
            self.parameter_receiver = self.EmptyParameterReceiver()
            return current
        
    def ResetStatus(self):
        self.status: Dict[str, Any] = {
            "state": "idle",
            "epoch": 0, "total_epochs": 0,
            "batch": 0, "total_batches": 0,
            "train_loss": 0.0, "val_loss": 0.0,
            "message": "Waiting to start",
            "trace": "",
            "visual": self.EmptyVisualStatus(),}

    def RequestStop(self):
        with self._lock:
            self.stop_requested = True
            self.pause_requested = False

    def RequestPause(self):
        with self._lock:
            self.pause_requested = True

    def RequestResetHebbian(self):
        with self._lock:
            self.reset_hebbian = True

    def RequestCancelResetHebbian(self):
        with self._lock:
            self.reset_hebbian = False

    def RequestResume(self):
        with self._lock:
            self.pause_requested = False

    def ShouldStop(self) -> bool:
        with self._lock:
            return self.stop_requested

    def ShouldPause(self) -> bool:
        with self._lock:
            return self.pause_requested
        
    def ShouldResetHebbian(self) -> bool:
        with self._lock:
            return self.reset_hebbian

    def SetVisualStateEnabled(self, enabled: bool):
        with self._lock:
            self.visual_state_enabled = bool(enabled)
            if not self.visual_state_enabled:
                self.status["visual"] = self.EmptyVisualStatus(touch=True)

    def IsVisualStateEnabled(self) -> bool:
        with self._lock:
            return self.visual_state_enabled



class ManagerFunction:
    DEFAULT_OVERRIDE_CHECKPOINT_WITH_MODULE_PARAMS = False

    def __init__(
        self,
        device: Optional[str] = None,
        *,
        robot: Optional[Robot] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.robot = robot if robot is not None else Robot.CreateDefault()
        self.brain_build_spec = BrainBuildSpec.Compile(
            ModuleDim.CognitiveProfile(),
            self.robot.ContractView)
        self.controller = ModuleController()

        self.br_thread: Optional[threading.Thread] = None
        self.message_thread: Optional[threading.Thread] = None
        self.is_begin = False
        self.overrideCheckpointWithModuleParams = (
            self.DEFAULT_OVERRIDE_CHECKPOINT_WITH_MODULE_PARAMS)
        self.agent_handle: Optional[AgentHandle] = None
        self.perception_calibration_id: Optional[str] = None
        self.active_sensor_stream_id: Optional[str] = None
        self.active_world_frame_id: Optional[str] = None
        self.last_sensor_sequence_index: Optional[int] = None
        self.pending_action_request: Optional[ActionRequest] = None
        self.stream_terminated = False
        self.json_queue = None

        self.test = {
            "perception": TestPerceptionMTool(),
            "attention": TestAttentionMTool(),
            "memory": TestMemoryMTool(),
            "decision": TestDecisionMTool(),
            "value": TestValueEstimationMTool(),
            "consciousness": TestConsciousMTool(),
            "OCR": TestOCRMTool(),
            "intention": TestIntentionMTool(),
            "AGICore": TestAGICoreMTool(),
            "manager": TestManagerMTool(),}

    def EncodeBrainFeedback(
        self,
        rawPayload: Any,
        *,
        batchSize: Optional[int] = None,
    ) -> BrainFeedbackPacket:
        return self.robot.EncodeFeedback(
            rawPayload,
            self.device,
            batchSize=batchSize)

    def BuildNeutralRobotFeedbackPayload(
        self,
        *,
        timestamp: float,
    ) -> Dict[str, Any]:
        return dict(self.robot.BuildNeutralFeedbackPayload(
            timestamp,
            self.device,
            torch.float32))

    def PrepareContractOfflineBatch(
        self,
        samples: Sequence[ContractOfflineSample],
    ) -> ContractOfflineBatch:
        if (
            not isinstance(samples, (list, tuple))
            or len(samples) < 1
            or any(type(sample) is not ContractOfflineSample for sample in samples)
        ):
            raise TypeError(
                "contract offline samples must be a non-empty sample sequence")
        converted = DataPreprocessor.ConvertSensoryInputs(
            imgs=torch.utils.data.default_collate([
                sample.image for sample in samples]),
            reward=torch.utils.data.default_collate([
                sample.reward for sample in samples]),
            done=torch.utils.data.default_collate([
                sample.done for sample in samples]),
            depths=torch.utils.data.default_collate([
                sample.depth for sample in samples]),
            depthValids=torch.utils.data.default_collate([
                sample.depth_valid for sample in samples]),
            device=self.device,
            needVisualState=False)
        batch_size = int(converted["frames"].size(0))
        feedback_packet = self.EncodeBrainFeedback(
            tuple(sample.feedback_payload for sample in samples),
            batchSize=batch_size)
        target_payloads = tuple(
            sample.perception_targets for sample in samples)
        perception_targets = (
            {}
            if all(len(payload) == 0 for payload in target_payloads)
            else torch.utils.data.default_collate(target_payloads)
        )
        return ContractOfflineBatch(
            frames=converted["frames"],
            rewards=converted["rewards"],
            dones=converted["dones"],
            depths=converted["depths"],
            depth_valid=converted["depth_valid"],
            text_ext=[sample.text_ext for sample in samples],
            feedback_packet=feedback_packet,
            perception_targets=perception_targets)

    def CompleteContractOfflineTransition(
        self,
        batch: ContractOfflineBatch,
    ) -> None:
        if type(batch) is not ContractOfflineBatch:
            raise TypeError("batch must be a ContractOfflineBatch")
        batch_size = int(batch.frames.size(0))
        if batch.dones is None:
            return
        done_mask = batch.dones.to(dtype=torch.bool)
        if tuple(done_mask.shape) != (batch_size,):
            raise ValueError("offline done mask must match the sensory batch")
        if bool(done_mask.any().item()):
            self.robot.Reset()

    def RunContractOfflineTransition(
        self,
        samples: Sequence[ContractOfflineSample],
        transitionFn: Callable[[ContractOfflineBatch], Any],
    ) -> Any:
        if not callable(transitionFn):
            raise TypeError("transitionFn must be callable")
        batch = self.PrepareContractOfflineBatch(samples)
        result = transitionFn(batch)
        self.CompleteContractOfflineTransition(batch)
        return result

    def CreateContractTrajectoryLoader(
        self,
        dataset: Dataset,
        *,
        batchSize: int,
    ) -> SequentialTrajectoryLoader:
        return SequentialTrajectoryLoader(
            dataset,
            batchSize=batchSize,
            collateFn=self.PrepareContractOfflineBatch)

    def EncodeBrainEmbodiment(
        self,
        rawPayload: Any,
        *,
        batchSize: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")
        feedback = self.EncodeBrainFeedback(
            rawPayload,
            batchSize=batchSize)
        return self.agent_handle.EncodeEmbodimentFeedback(
            feedback,
            batchSize=batchSize)

    @staticmethod
    def NormalizeTrainStage(trainStage: str) -> str:
        if type(trainStage) is not str:
            raise TypeError("trainStage must be a string")
        stage = trainStage.strip().lower()
        if stage not in ("full", "world", "policy"):
            raise ValueError("trainStage must be one of: full/world/policy")
        return stage

    @staticmethod
    def TrainStageLossNames(trainStage: str) -> Tuple[str, ...]:
        stage = ManagerFunction.NormalizeTrainStage(trainStage)
        if stage == "world":
            return ("world",)
        if stage == "policy":
            return ("critic", "policy")
        return ("world", "critic", "policy")

    @staticmethod
    def TrainStageOnlineWrappers(brain: BrainCore, trainStage: str) -> List[nn.Module]:
        stage = ManagerFunction.NormalizeTrainStage(trainStage)
        if stage == "world":
            return [brain.world]
        policy_wrappers = [brain.perc, brain.attn, brain.critic, brain.intention]
        if stage == "policy":
            return policy_wrappers
        return [brain.perc, brain.attn, brain.world, brain.critic, brain.intention]

    def EvaluateWithRestoredBrainBuffers(
        self,
        brain: BrainCore,
        evaluate: Callable[[], Any],) -> Any:
        training_runtime = brain.ExportBuffers()
        world = brain.RuntimeModule(brain.world)
        training_world_physical = world.ExportPhysicalState()
        training_memory_durable = brain.mem.ExportDurableState()
        training_rng = self.CaptureRngState()
        training_modes = [
            (module, bool(module.training))
            for module in brain.modules()]
        training_graph = brain.SuspendTransientTrainingGraph()
        try:
            return evaluate()
        finally:
            try:
                world.ImportPhysicalState(training_world_physical)
                brain.mem.ImportDurableState(training_memory_durable)
                brain.ImportBuffers(training_runtime)
            finally:
                try:
                    brain.RestoreTransientTrainingGraph(training_graph)
                finally:
                    try:
                        for module, was_training in training_modes:
                            module.training = was_training
                    finally:
                        self.RestoreRngState(training_rng)

    def EvaluateValidationAndTestWithRestoredBrainBuffers(
        self,
        brain: BrainCore,
        evaluateValidation: Callable[[], Any],
        evaluateTest: Callable[[], Any],) -> Tuple[Any, Any]:
        validation_result = self.EvaluateWithRestoredBrainBuffers(
            brain, evaluateValidation)
        test_result = self.EvaluateWithRestoredBrainBuffers(
            brain, evaluateTest)
        return validation_result, test_result
        
    def InitAgentHandle(
        self,
        usePlanner: bool = True,
    ):
        projection = self.robot.ContractView.perception_projection
        if projection is None:
            raise RuntimeError("robot contract has no perception calibration")
        self.perception_calibration_id = projection.calibration_id
        self.agent_handle = AgentHandle(
            brainBuildSpec=self.brain_build_spec,
            robot=self.robot,
            usePlanner=usePlanner,
            device=self.device)
        self.active_sensor_stream_id = None
        self.active_world_frame_id = None
        self.last_sensor_sequence_index = None
        self.pending_action_request = None
        self.stream_terminated = False
        return True

    def TerminateSensorStream(self) -> None:
        self.robot.Reset()
        self.pending_action_request = None
        self.active_sensor_stream_id = None
        self.active_world_frame_id = None
        self.last_sensor_sequence_index = None
        self.stream_terminated = True

    def SetJsonQueue(self, queue):
        self.json_queue = queue
        return True

    def SetParameterReceiver(self, reward = None, done = None, textExt = None):
        self.controller.SetParameterReceiver(reward=reward, done=done, textExt=textExt)
        return True

    def ValidateContractRequestProvenance(
        self,
        requestProvenance: Dict[str, Any],
    ) -> None:
        if (
            type(requestProvenance) is not dict
            or set(requestProvenance)
            != set(DECISION_REQUEST_PROVENANCE_FIELDS)
        ):
            raise ValueError(
                "contract request provenance fields do not match the wire schema")
        for name in (
            "stream_id",
            "frame_id",
            "calibration_id",
            "world_frame_id",
            "description_id",
            "model_contract_id",
            "adapter_id",
        ):
            if (
                type(requestProvenance[name]) is not str
                or not requestProvenance[name]
            ):
                raise ValueError(
                    "contract request {} must be non-empty".format(name))
        if (
            type(requestProvenance["sequence_index"]) is not int
            or requestProvenance["sequence_index"] < 0
        ):
            raise ValueError(
                "contract request sequence_index must be non-negative")
        contract_view = self.brain_build_spec.contract_view
        projection = contract_view.perception_projection
        expected_identity = {
            "calibration_id": (
                projection.calibration_id if projection is not None else ""),
            "description_id": contract_view.description_id,
            "model_contract_id": self.brain_build_spec.model_signature,
            "adapter_id": contract_view.adapter_id,
        }
        for name, expected in expected_identity.items():
            if requestProvenance[name] != expected:
                raise ValueError(
                    "contract request {} does not match BrainBuildSpec".format(
                        name))

    def EncodeCognitiveReadout(
        self,
        readout: CognitiveReadout,
    ) -> Dict[str, Any]:
        readout.Validate(self.brain_build_spec)
        return {
            name: (
                value.detach().cpu().tolist()
                if torch.is_tensor(value)
                else value)
            for name, value in (
                (fieldName, getattr(readout, fieldName))
                for fieldName in readout.__dataclass_fields__)}

    def ForwardContractBatch(
        self,
        bitmap: Union[List[Any], np.ndarray, torch.Tensor],
        reward: Optional[float] = None,
        done: Optional[float] = None,
        textExt: Optional[List[Optional[str]]] = None,
        textTrust: Optional[List[str]] = None,
        sampleActions: bool = True,
        deterministicActor: bool = False,
        *,
        depthBitmap: Union[List[Any], np.ndarray, torch.Tensor],
        depthValid: Union[List[Any], np.ndarray, torch.Tensor],
        feedbackPayload: Any,
        requestProvenance: Dict[str, Any],
    ) -> ContractAgentActOutput:
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")
        self.ValidateContractRequestProvenance(requestProvenance)
        if type(sampleActions) is not bool:
            raise TypeError("sampleActions must be a boolean")
        if type(deterministicActor) is not bool:
            raise TypeError("deterministicActor must be a boolean")

        converted = DataPreprocessor.ConvertCppPerceptionFrame(
            bitmap=bitmap,
            reward=reward,
            done=done,
            depthBitmap=depthBitmap,
            depthValid=depthValid,
            device=self.device,
            needVisualState=False)
        batch_size = int(converted["frames"].size(0))
        if textExt is not None and (
            type(textExt) is not list
            or len(textExt) != batch_size
            or any(item is not None and type(item) is not str for item in textExt)
        ):
            raise ValueError("textExt must match the sensory batch")
        if textTrust is not None and (
            type(textTrust) is not list
            or len(textTrust) != batch_size
            or any(item not in (
                TEXT_TRUST_OCR_OBSERVED,
                TEXT_TRUST_OPERATOR_COMMAND,
                TEXT_TRUST_UNSAFE_EXTERNAL,
            ) for item in textTrust)
        ):
            raise ValueError("textTrust must match the sensory batch")
        feedback_packet = self.EncodeBrainFeedback(
            feedbackPayload,
            batchSize=batch_size)
        return self.ForwardMaterializedContractBatch(
            ContractAgentActInput(
                frame=converted["frames"],
                text_ext=textExt,
                reward=converted["rewards"],
                done=converted["dones"],
                sample_actions=sampleActions,
                deterministic_actor=deterministicActor,
                depth=converted["depths"],
                depth_valid=converted["depth_valid"],
                feedback_packet=feedback_packet,
                text_trust=textTrust),
            requestProvenance)

    def ForwardMaterializedContractBatch(
        self,
        request: ContractAgentActInput,
        requestProvenance: Dict[str, Any],
    ) -> ContractAgentActOutput:
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")
        if type(request) is not ContractAgentActInput:
            raise TypeError(
                "materialized contract input must be ContractAgentActInput")
        self.agent_handle.agent.BindWorldMemoryContext(
            requestProvenance["world_frame_id"],
            batchSize=int(request.frame.size(0)))
        act_output = self.agent_handle.ForwardStep(
            request.frame,
            textExt=request.text_ext,
            textTrust=request.text_trust,
            reward=request.reward,
            done=request.done,
            sampleActions=request.sample_actions,
            deterministicActor=request.deterministic_actor,
            depth=request.depth,
            depthValid=request.depth_valid,
            feedbackPacket=request.feedback_packet)
        return act_output

    def AgentHandleForwardJson(
        self,
        reward: Optional[float],
        done: Optional[float],
        sensorPacketJson: str,
        feedbackPayloadJson: str,
        executionResultJson: str,
    ) -> str:
        if self.stream_terminated:
            raise RuntimeError(
                "a terminated sensor stream requires a new AgentHandle")
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")
        sensor_packet = json.loads(sensorPacketJson)
        feedback_payload = json.loads(feedbackPayloadJson)
        execution_payload = json.loads(executionResultJson)
        if type(sensor_packet) is not dict or set(sensor_packet) != set(SENSOR_PACKET_WIRE_FIELDS):
            raise ValueError("sensor packet fields do not match the current schema")
        if (
            type(sensor_packet["schema_version"]) is not int
            or sensor_packet["schema_version"] != SENSOR_PACKET_WIRE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported sensor packet schema")
        if self.perception_calibration_id is None:
            raise RuntimeError("agent_handle has not been initialized")
        if sensor_packet["calibration_id"] != self.perception_calibration_id:
            raise ValueError("sensor packet calibration_id does not match configured K")
        if type(sensor_packet["stream_id"]) is not str or not sensor_packet["stream_id"]:
            raise ValueError("sensor stream_id must be a non-empty string")
        if (
            type(sensor_packet["sequence_index"]) is not int
            or sensor_packet["sequence_index"] < 0
        ):
            raise ValueError(
                "sensor sequence_index must be a non-negative integer")
        if not isinstance(sensor_packet["frame_id"], str) or not sensor_packet["frame_id"]:
            raise ValueError("frame_id must be a non-empty string")
        if sensor_packet["rgb_encoding"] != "rgb8" or sensor_packet["depth_unit"] != "meter":
            raise ValueError("sensor packet must contain rgb8 and metre depth")
        sequence_index = sensor_packet["sequence_index"]
        if type(feedback_payload) is not dict:
            raise TypeError("external robot feedback must be an object")
        execution_result = None
        if self.active_sensor_stream_id is None:
            if sequence_index != 0:
                raise ValueError("a sensor stream must begin at sequence_index 0")
            if execution_payload is not None or self.pending_action_request is not None:
                raise ValueError("the first sensor frame cannot carry an execution result")
        else:
            if sensor_packet["stream_id"] != self.active_sensor_stream_id:
                raise ValueError(
                    "sensor stream_id changed; initialize a new AgentHandle")
            if sequence_index != self.last_sensor_sequence_index + 1:
                raise ValueError(
                    "sensor sequence_index must increase by exactly one")
            if self.pending_action_request is None:
                raise RuntimeError("the sensor stream has no pending action request")
            execution_result = self.robot.DecodeActionExecutionResult(
                execution_payload,
                self.pending_action_request,
                self.device)
        text_ext = sensor_packet["text_ext"]
        text_trust = sensor_packet["text_trust"]
        if type(text_ext) is not list or type(text_trust) is not list:
            raise TypeError("sensor text_ext and text_trust must be arrays")
        if len(text_ext) != 1 or len(text_trust) != 1:
            raise ValueError("single-frame sensor text_ext/text_trust must have length 1")
        if text_ext[0] is not None and type(text_ext[0]) is not str:
            raise TypeError("sensor text_ext item must be a string or null")
        if text_trust[0] not in (
            TEXT_TRUST_OCR_OBSERVED,
            TEXT_TRUST_OPERATOR_COMMAND,
            TEXT_TRUST_UNSAFE_EXTERNAL,
        ):
            raise ValueError("unsupported text_trust value")
        if type(sensor_packet["sample_actions"]) is not bool:
            raise TypeError("sensor sample_actions must be a boolean")
        if type(sensor_packet["deterministic_actor"]) is not bool:
            raise TypeError("sensor deterministic_actor must be a boolean")
        world_context_id = "sensor_stream:{}".format(
            sensor_packet["stream_id"])
        request_provenance = {
            "stream_id": sensor_packet["stream_id"],
            "sequence_index": sequence_index,
            "frame_id": sensor_packet["frame_id"],
            "calibration_id": sensor_packet["calibration_id"],
            "world_frame_id": world_context_id,
            "description_id": (
                self.brain_build_spec.contract_view.description_id),
            "model_contract_id": self.brain_build_spec.model_signature,
            "adapter_id": self.brain_build_spec.contract_view.adapter_id,
        }
        self.ValidateContractRequestProvenance(request_provenance)
        converted = DataPreprocessor.ConvertCppPerceptionFrame(
            bitmap=sensor_packet["rgb"],
            reward=reward,
            done=done,
            depthBitmap=sensor_packet["depth"],
            depthValid=sensor_packet["depth_valid"],
            device=self.device,
            needVisualState=False)
        batch_size = int(converted["frames"].size(0))
        robot_runtime_fields = (
            "CachedTargetValues",
            "CachedTargetActive",
            "CachedTargetVersion",
            "CachedActionEpoch",
            "CachedRequestId",
            "CachedExecutionStatus",
            "CachedExecutionKnown",
            "CachedExecutionRelevant",
            "CachedExecutionResultKnown",
            "CachedExecutionTimestamp",
            "CachedHardStop",
            "CachedHelpAccepted",
            "DwellState",
            "ReachedState",
            "ProgressState",
            "ObservationAgeState",
            "ObservationKnownState",
            "LastTimestamp",
            "LastPerceptionRotation",
            "LastPerceptionStatePresent",
        )
        robot_runtime = {
            name: (
                None
                if getattr(self.robot, name) is None
                else getattr(self.robot, name).detach().clone())
            for name in robot_runtime_fields}
        try:
            if execution_result is not None:
                self.robot.CommitAppliedTarget(
                    self.pending_action_request,
                    execution_result)
            feedback_packet = self.EncodeBrainFeedback(
                feedback_payload,
                batchSize=batch_size)
            if (
                execution_result is not None
                and bool(feedback_packet.timestamp.lt(
                    execution_result.timestamp).any().item())
            ):
                raise ValueError(
                    "feedback time precedes the applied action result")
        except Exception:
            for name, value in robot_runtime.items():
                setattr(self.robot, name, value)
            raise
        try:
            act_output = self.ForwardMaterializedContractBatch(
                ContractAgentActInput(
                    frame=converted["frames"],
                    text_ext=text_ext,
                    reward=converted["rewards"],
                    done=converted["dones"],
                    sample_actions=sensor_packet["sample_actions"],
                    deterministic_actor=sensor_packet[
                        "deterministic_actor"],
                    depth=converted["depths"],
                    depth_valid=converted["depth_valid"],
                    feedback_packet=feedback_packet,
                    text_trust=text_trust),
                request_provenance)
            terminal = done is not None and float(done) > 0.5
            response = json.dumps({
                "schema_version": DECISION_WIRE_SCHEMA_VERSION,
                "request_provenance": request_provenance,
                "action_request": (
                    None
                    if terminal
                    else self.robot.EncodeActionRequest(
                        act_output.action_request)),
                "cognitive_readout": self.EncodeCognitiveReadout(
                    act_output.cognitive_readout),
                "intention_texts": list(act_output.intention_texts),
            }, ensure_ascii=False, allow_nan=False)
            if terminal:
                self.TerminateSensorStream()
            else:
                self.pending_action_request = act_output.action_request
                self.active_sensor_stream_id = sensor_packet["stream_id"]
                self.active_world_frame_id = world_context_id
                self.last_sensor_sequence_index = sequence_index
            return response
        except Exception:
            self.TerminateSensorStream()
            raise

    def ResetAgentHandleHebbian(self):
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")
        
        self.agent_handle.agent.ResetHebbianMemory()
        return True


    def SetBasicParameters(self, name: str, value: str):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        return BasicParameters.Set(name=name, value=value)
    
    def GetBasicParameters(self, name: str):
        return BasicParameters.Get(name=name)
    
    def GetBasicParametersDict(self):
        return BasicParameters.GetStringDict()

    def SetOverrideCheckpointWithModuleParams(self, enabled: bool):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.overrideCheckpointWithModuleParams = bool(enabled)
        return True

    def RunBackgroundTask(self, target: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        try:
            target(*args, **kwargs)
        except Exception as e:
            self.controller.SetStatus("error", f"Background task error: {e}", trace=traceback.format_exc())
        finally:
            self.is_begin = False

    def StartBackgroundTask(
        self,
        target: Callable[..., Any],
        *,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,) -> bool:
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False

        self.is_begin = True
        self.br_thread = threading.Thread(
            target=self.RunBackgroundTask,
            args=(target, args, kwargs or {}),
            daemon=False)
        self.br_thread.start()
        return True

    def StartTraining(
        self,
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        onlineLearning: bool = False,
        isTest: bool = False,
        overrideCheckpointWithModuleParams: Optional[bool] = None,
        saveEverySampleCount: int = 2000,
        trainStage: str = "full",
    ) -> bool:
        train_stage = self.NormalizeTrainStage(trainStage)
        ckpt_path = (
            BasicParameters.CKPT_PATH_TEST
            if isTest
            else BasicParameters.CKPT_PATH_TRAIN)
        out_path = (
            BasicParameters.MODULEPARAMETER_PATH_TEST
            if isTest
            else BasicParameters.MODULEPARAMETER_PATH)
        world_memory_path = (
            BasicParameters.WORLD_MEMORY_PATH_TEST_TRAIN
            if isTest
            else BasicParameters.WORLD_MEMORY_PATH_TRAIN)
        memory_path = (
            BasicParameters.MEMORY_MEMORY_PATH_TEST_TRAIN
            if isTest
            else BasicParameters.MEMORY_MEMORY_PATH_TRAIN)
        override_enabled = (
            bool(self.overrideCheckpointWithModuleParams)
            if overrideCheckpointWithModuleParams is None
            else bool(overrideCheckpointWithModuleParams))
        return self.StartBackgroundTask(
            self.TrainLoop,
            args=(epochs, batchSize, valSplit, resume, onlineLearning),
            kwargs={
                "worldMemPath": world_memory_path,
                "memMemPath": memory_path,
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "overrideCheckpointWithModuleParams": override_enabled,
                "saveEverySampleCount": saveEverySampleCount,
                "trainStage": train_stage,
                "isTest": isTest,
            })

    def StartOCRTraining(
        self,
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        isTest: bool = False,
        saveEverySampleCount = 2000,
        *,
        trainDetection: bool = True,
        trainRecognition: bool = True,
        overrideCheckpointWithModuleParams: Optional[bool] = None,):
        ckpt_path = BasicParameters.OCR_CKPT_PATH_TEST if isTest else BasicParameters.OCR_CKPT_PATH_TRAIN
        out_path = BasicParameters.OCR_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_MODULEPARAMETER_PATH
        recognizer_init_path = BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH
        override_enabled = bool(self.overrideCheckpointWithModuleParams) if overrideCheckpointWithModuleParams is None else bool(overrideCheckpointWithModuleParams)

        return self.StartBackgroundTask(
            self.OCRTrainLoop,
            args=(epochs, batchSize, valSplit, resume),
            kwargs={
                "isTest": isTest,
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "trainDetection": trainDetection,
                "trainRecognition": trainRecognition,
                "overrideCheckpointWithModuleParams": override_enabled,
                "saveEverySampleCount": saveEverySampleCount,
                "recognizerInitPath": recognizer_init_path,})

    def StartOCRRecognitionTraining(
        self,
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        isTest: bool = False,
        overrideCheckpointWithModuleParams: Optional[bool] = None,
        saveEverySampleCount = 2000):
        ckpt_path = BasicParameters.OCR_RECOGNIZER_CKPT_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_CKPT_PATH_TRAIN
        out_path = BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH
        override_enabled = bool(self.overrideCheckpointWithModuleParams) if overrideCheckpointWithModuleParams is None else bool(overrideCheckpointWithModuleParams)

        return self.StartBackgroundTask(
            self.OCRRecognitionTrainLoop,
            args=(epochs, batchSize, valSplit, resume),
            kwargs={
                "isTest": isTest,
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "overrideCheckpointWithModuleParams": override_enabled,
                "saveEverySampleCount": saveEverySampleCount,})

    def Stop(self):
        if self.is_begin:
            self.controller.RequestStop()
            if self.br_thread is not None:
                self.br_thread.join()
                self.br_thread = None

            if self.message_thread is not None:
                self.message_thread.join()
                self.message_thread = None

            self.is_begin = False
            return True
        return False

    def Pause(self):
        if self.is_begin:
            self.controller.RequestPause()
            return True
        return False

    def Resume(self):
        if self.is_begin:
            self.controller.RequestResume()
            return True
        return False
    
    def ResetHebbianMemory(self):
        if self.is_begin:
            self.controller.RequestResetHebbian()
            return True
        return False

    def GetCurrentStatus(self):
        return self.controller.GetStatus()

    def SetVisualStateEnabled(self, enabled: bool):
        self.controller.SetVisualStateEnabled(enabled)
        return True

    def GetVisualStateEnabled(self):
        return self.controller.IsVisualStateEnabled()

    def ResetControllerFlags(self) -> None:
        with self.controller._lock:
            self.controller.stop_requested = False
            self.controller.pause_requested = False
            self.controller.reset_hebbian = False

    def WaitWhilePaused(self, pausedMessage: str) -> bool:
        while self.controller.ShouldPause():
            if self.controller.ShouldStop():
                return False
            self.controller.SetStatus("paused", pausedMessage)
            time.sleep(0.2)
        return not self.controller.ShouldStop()

    def StartMessageMonitor(self, monitorFn: Callable[[], None]) -> None:
        self.message_thread = threading.Thread(target=monitorFn, args=(), daemon=False)
        self.message_thread.start()

    def RunNamedTest(self, testKey: str):
        return self.test[str(testKey)].RunAll()

    def CreateDataLoader(
        self,
        dataset: Dataset,
        *,
        batchSize: int,
        shuffle: bool,
        collateFn: Optional[Callable[..., Any]] = None,
        pinMemory: Optional[bool] = None,) -> DataLoader:
        loader_kwargs: Dict[str, Any] = {
            "batch_size": batchSize,
            "shuffle": shuffle,
            "num_workers": 0,
            "pin_memory": (
                bool(getattr(self.device, "type", "") == "cuda")
                if pinMemory is None else
                bool(pinMemory)),}
        if collateFn is not None:
            loader_kwargs["collate_fn"] = collateFn
        return DataLoader(dataset, **loader_kwargs)

    def SplitDataset(
        self,
        dataset: Dataset,
        *,
        valSplit: float,
        testSplit: float = 0.1,
        trainDataset: Optional[Dataset] = None,
        valDataset: Optional[Dataset] = None,
        testDataset: Optional[Dataset] = None,):
        if trainDataset is None:
            n_total = len(dataset)
            n_test = int(n_total * testSplit)
            n_val = int(n_total * valSplit)
            n_train = n_total - n_val - n_test
            trainDataset, valDataset, testDataset = torch.utils.data.random_split(
                dataset,
                [n_train, n_val, n_test])
        elif testDataset is None:
            train_indices = list(trainDataset.indices) if hasattr(trainDataset, "indices") else list(range(len(trainDataset)))
            val_indices = list(valDataset.indices) if hasattr(valDataset, "indices") else []
            used = set(train_indices) | set(val_indices)
            test_indices = [idx for idx in range(len(dataset)) if idx not in used]
            testDataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else valDataset

        return trainDataset, valDataset, testDataset

    def SplitDatasetSequential(
        self,
        dataset: Dataset,
        *,
        valSplit: float,
        testSplit: float = 0.1,
        trainDataset: Optional[Dataset] = None,
        valDataset: Optional[Dataset] = None,
        testDataset: Optional[Dataset] = None,):
        if trainDataset is None:
            n_total = len(dataset)
            n_test = int(n_total * testSplit)
            n_val = int(n_total * valSplit)
            n_train = n_total - n_val - n_test
            trainDataset = torch.utils.data.Subset(dataset, list(range(0, n_train)))
            valDataset = torch.utils.data.Subset(dataset, list(range(n_train, n_train + n_val)))
            testDataset = torch.utils.data.Subset(dataset, list(range(n_train + n_val, n_total)))
        return trainDataset, valDataset, testDataset

    def HasGameDataset(self, dataRoot: Optional[Union[str, Path]] = None) -> bool:
        if dataRoot is None:
            frames_dir = Path(BasicParameters.DATA_FRAMES_PATH)
            reward_dir = Path(BasicParameters.DATA_REWARD_PATH)
            done_dir = Path(BasicParameters.DATA_DONE_PATH)
            depth_dir = Path(BasicParameters.DATA_DEPTH_PATH)
            depth_valid_dir = Path(BasicParameters.DATA_DEPTH_VALID_PATH)
            feedback_path = getattr(BasicParameters, "DATA_FEEDBACK_PATH", None)
            if type(feedback_path) is not str or not feedback_path:
                return False
            feedback_dir = Path(feedback_path)
            sensor_manifest_path = Path(BasicParameters.DATA_SENSOR_MANIFEST_PATH)
            texts_dir = Path(BasicParameters.DATA_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            reward_dir = root / "reward"
            done_dir = root / "done"
            depth_dir = root / "depth"
            depth_valid_dir = root / "depth_valid"
            feedback_dir = root / "feedback"
            sensor_manifest_path = root / "sensor_manifest.json"
            texts_dir = root / "texts"

        required_dirs = (
            frames_dir,
            reward_dir,
            done_dir,
            depth_dir,
            depth_valid_dir,
            feedback_dir,
        )
        if (
            not all(path.exists() for path in required_dirs)
            or not sensor_manifest_path.is_file()
        ):
            return False
        try:
            projection = self.robot.ContractView.perception_projection
            if projection is None:
                return False
            manifest = json.loads(
                sensor_manifest_path.read_text(encoding="utf-8"))
            ValidateContractOfflineSensorManifest(
                manifest,
                projection.calibration_id,
                projection.reference_frame_id)
        except (OSError, TypeError, ValueError):
            return False

        counts = (
            len(sorted(frames_dir.glob("*.png"))),
            len(sorted(reward_dir.glob("*.npy"))),
            len(sorted(done_dir.glob("*.npy"))),
            len(DataPreprocessor.ListDepthFiles(depth_dir)),
            len(DataPreprocessor.ListDepthFiles(depth_valid_dir)),
            len(DataPreprocessor.ListJsonFiles(feedback_dir)),
        )
        if counts[0] == 0 or len(set(counts)) != 1:
            return False
        if texts_dir.exists():
            text_count = len(sorted(texts_dir.glob("*.txt")))
            if text_count not in (0, counts[0]):
                return False
        return True

    def SummarizeImageDirectory(self, path: Union[str, Path]) -> str:
        img_dir = Path(path)
        if not img_dir.exists():
            return "missing"

        img_files = DataPreprocessor.ListImageFiles(img_dir)
        if not img_files:
            return "0 supported image files"

        suffix_counts: Dict[str, int] = {}
        for img_path in img_files:
            suffix = img_path.suffix.lower() or "<no_ext>"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

        suffix_summary = ", ".join(
            f"{suffix}:{count}"
            for suffix, count in sorted(suffix_counts.items()))
        return f"{len(img_files)} image files ({suffix_summary})"

    def HasOcrDataset(self, dataRoot: Optional[Union[str, Path]] = None) -> bool:
        if dataRoot is None:
            root = Path(BasicParameters.OCR_DATA_ROOT_PATH)
            frames_dir = Path(BasicParameters.OCR_FRAMES_PATH)
            texts_dir = Path(BasicParameters.OCR_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

        if not (frames_dir.exists() and texts_dir.exists()):
            return False

        frame_files = DataPreprocessor.ListImageFiles(frames_dir)
        txt_files = sorted(texts_dir.glob("*.txt"))
        if not frame_files or not txt_files or len(frame_files) != len(txt_files):
            return False

        boxes_dir = root / "boxes"
        return any(boxes_dir.glob("*.npy")) or any(
            DataPreprocessor.TextFileLooksLikeOCRAnnotations(txt_path)
            for txt_path in txt_files)

    def HasOcrRecognitionDataset(self, dataRoot: Optional[Union[str, Path]] = None) -> bool:
        if dataRoot is None:
            frames_dir = Path(BasicParameters.OCR_RECOGNIZER_FRAMES_PATH)
            texts_dir = Path(BasicParameters.OCR_RECOGNIZER_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

        if not (frames_dir.exists() and texts_dir.exists()):
            return False

        frame_files = DataPreprocessor.ListImageFiles(frames_dir)
        text_files = sorted(texts_dir.glob("*.txt"))
        return bool(frame_files) and bool(text_files) and len(frame_files) == len(text_files)

    def MonitorStatus(
        self,
        *,
        prefix: str,
        monitorName: str,
        renderStatus: Callable[[Dict[str, Any]], str],
        terminalStates: Tuple[str, ...],
        sleepSeconds: float,) -> None:
        try:
            while True:
                st = self.GetCurrentStatus()
                print(f"[{prefix}] {renderStatus(st)}")

                if st["state"] == "error":
                    trace = st.get("trace")
                    if trace:
                        print(f"\n====== {prefix} ERROR TRACEBACK ======\n")
                        print(trace)
                        print("====================================\n")

                if st["state"] in terminalStates:
                    self.controller.ResetStatus()
                    break

                time.sleep(sleepSeconds)

        except Exception as e:
            print(f"[{monitorName}] monitor raised: {e}")
            print(traceback.format_exc())

    def RestoreOcrItemsToOriginal(
        self,
        ocrItems: Optional[List[Dict[str, Any]]],
        resizeMeta: Optional[DataResizeMeta] = None,) -> List[Dict[str, Any]]:
        restored_items = [dict(item) for item in (ocrItems or [])]
        if not restored_items:
            return restored_items

        boxes: List[List[float]] = []
        valid_indices: List[int] = []
        for idx, item in enumerate(restored_items):
            box = item.get("box", None)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                item["box"] = (0, 0, 0, 0)
                continue
            boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            valid_indices.append(idx)

        if not boxes:
            return restored_items

        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if resizeMeta is not None:
            boxes_np = DataPreprocessor.RestoreBoxesXYXY(boxes_np, resizeMeta, clamp=True)

        for idx, box in zip(valid_indices, boxes_np):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
            restored_items[idx]["box"] = (x1, y1, x2, y2)

        return restored_items

    def DrawBoxesOnImage(
        self,
        image: Union[np.ndarray, torch.Tensor],
        ocrItems: Optional[List[Dict[str, Any]]],
        *,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            canvas = image.detach().cpu().numpy()
        else:
            canvas = np.asarray(image)

        if canvas.ndim == 4:
            if canvas.shape[-1] in (1, 3, 4):
                canvas = canvas[0]
            elif canvas.shape[0] in (1, 3, 4):
                canvas = canvas[..., 0]
            else:
                raise ValueError(f"visual image must have 2 or 3 dims, but got {canvas.shape}")

        if canvas.ndim == 2:
            canvas = canvas[..., None]
        if canvas.ndim != 3:
            raise ValueError(f"visual image must have 2 or 3 dims, but got {canvas.shape}")

        if canvas.shape[0] in (1, 3, 4) and canvas.shape[-1] not in (1, 3, 4):
            canvas = np.moveaxis(canvas, 0, -1)

        if canvas.shape[-1] == 1:
            canvas = np.repeat(canvas, 3, axis=-1)
        elif canvas.shape[-1] >= 3:
            canvas = canvas[..., :3]
        else:
            raise ValueError(f"visual image channel layout is invalid: {canvas.shape}")

        canvas = np.ascontiguousarray(np.array(canvas, copy=True))
        if canvas.size == 0:
            return canvas

        h_img, w_img = canvas.shape[:2]
        thick = max(1, int(thickness))

        for item in ocrItems or []:
            box = item.get("box", None)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(x1 + 1, min(w_img, x2))
            y2 = max(y1 + 1, min(h_img, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            canvas[y1:min(h_img, y1 + thick), x1:x2] = color
            canvas[max(0, y2 - thick):y2, x1:x2] = color
            canvas[y1:y2, x1:min(w_img, x1 + thick)] = color
            canvas[y1:y2, max(0, x2 - thick):x2] = color

        return canvas

    def FormatVisualText(
        self,
        title: str,
        ocrTexts: List[str],
        extraLines: Optional[List[str]] = None,) -> str:
        lines: List[str] = []
        title_text = str(title).strip()
        if title_text:
            lines.append(title_text)

        if extraLines:
            for line in extraLines:
                line_text = str(line).strip()
                if line_text:
                    lines.append(line_text)

        lines.append("OCR Text:")
        if ocrTexts:
            lines.extend(ocrTexts)
        else:
            lines.append("<empty>")

        return "\n".join(lines)

    def BuildVisualPayload(
        self,
        image: Optional[Union[np.ndarray, torch.Tensor]],
        *,
        ocrTexts: Optional[List[str]] = None,
        ocrItems: Optional[List[Dict[str, Any]]] = None,
        resizeMeta: Optional[DataResizeMeta] = None,
        title: str = "",
        extraLines: Optional[List[str]] = None,
        drawBoxes: bool = True,) -> Dict[str, Any]:
        updated_at = time.time()
        clean_texts = [str(text).strip() for text in (ocrTexts or []) if str(text).strip() != ""]
        restored_items = self.RestoreOcrItemsToOriginal(ocrItems, resizeMeta)

        if image is None:
            payload = self.controller.EmptyVisualStatus()
            payload["text"] = self.FormatVisualText(title, clean_texts, extraLines)
            payload["ocr_texts"] = clean_texts
            payload["items"] = restored_items
            payload["updated_at"] = updated_at
            return payload

        bitmap_rgb = self.DrawBoxesOnImage(
            image,
            restored_items if drawBoxes else None)

        return {
            "bitmap": bitmap_rgb.tolist(),
            "text": self.FormatVisualText(title, clean_texts, extraLines),
            "ocr_texts": clean_texts,
            "items": restored_items,
            "updated_at": updated_at,}

    @staticmethod
    def AtomicTorchSave(payload: Any, path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent))
        os.close(fd)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, target)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @staticmethod
    def DeploymentManifestPath(modelPath: Union[str, Path]) -> Path:
        return Path(f"{modelPath}.manifest.json")

    @staticmethod
    def AtomicJsonSave(payload: Dict[str, Any], path: Union[str, Path]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, target)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    @classmethod
    def ResolveDeploymentArtifactPaths(
        cls,
        modelPath: Union[str, Path],
        *,
        calibrationId: str,
        brainBuildSpec: BrainBuildSpec,) -> Tuple[str, str, str]:
        if type(brainBuildSpec) is not BrainBuildSpec:
            raise TypeError("deployment resolution requires BrainBuildSpec")
        contract_view = brainBuildSpec.contract_view
        manifest_path = cls.DeploymentManifestPath(modelPath)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"deployment manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if type(manifest) is not dict or set(manifest) != DEPLOYMENT_MANIFEST_FIELDS:
            raise ValueError("deployment manifest fields do not match the current schema")
        if (
            type(manifest["schema_version"]) is not int
            or manifest["schema_version"] != TRAIN_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("deployment manifest schema is unsupported")
        if manifest["calibration_id"] != calibrationId:
            raise ValueError("deployment manifest calibration_id does not match configured K")
        if manifest["description_id"] != contract_view.description_id:
            raise ValueError(
                "deployment manifest description_id does not match the robot")
        if manifest["model_contract_id"] != brainBuildSpec.model_signature:
            raise ValueError(
                "deployment manifest model_contract_id does not match the foundation")
        if manifest["adapter_id"] != contract_view.adapter_id:
            raise ValueError(
                "deployment manifest adapter_id does not match the robot")
        for field in ("generation", "model_path", "world_memory_path", "memory_path"):
            if type(manifest[field]) is not str or not manifest[field]:
                raise TypeError(f"deployment manifest {field} must be a non-empty string")
        artifact_paths = tuple(Path(manifest[field]) for field in (
            "model_path", "world_memory_path", "memory_path"))
        generation_directories = {path.parent for path in artifact_paths}
        if (
            any(not path.is_absolute() for path in artifact_paths)
            or len(generation_directories) != 1
            or next(iter(generation_directories)).name != manifest["generation"]
        ):
            raise ValueError(
                "deployment artifacts must belong to the declared generation")
        for artifact_path in artifact_paths:
            if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
                raise FileNotFoundError(
                    f"deployment generation artifact is missing or empty: {artifact_path}")
        return tuple(str(path) for path in artifact_paths)

    def SaveModuleParameters(self, brain: BrainCore, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        brain_state = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in ExportDeploymentModelState(brain).items()}

        self.AtomicTorchSave({
            "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
            "calibration_id": brain.calibration_id,
            "model_contract_id": brain.model_contract_id,
            "brain": brain_state,}, out_path)

    def SaveCognitiveBackboneParameters(
        self,
        brain: BrainCore,
        path: str,
    ) -> None:
        if type(path) is not str or not path.strip():
            raise ValueError("cognitive backbone path must be non-empty")
        self.AtomicTorchSave(
            ExportCognitiveBackboneState(brain),
            Path(path))

    def SaveOCRParameters(self, engine: OCREngineExtractor, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ocr_state = {k: v.detach().cpu() for k, v in engine.state_dict().items()}
        self.AtomicTorchSave({
            "schema_version": OCR_CHECKPOINT_SCHEMA_VERSION,
            "ocr": ocr_state,
            "ocr_meta": self.CurrentOcrMetadata(engine),}, out_path)

    def SaveOCRRecognizerParameters(self, engine: OCREngineExtractor, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rec_state = {k: v.detach().cpu() for k, v in engine.recognizer.state_dict().items()}
        self.AtomicTorchSave({
            "schema_version": OCR_CHECKPOINT_SCHEMA_VERSION,
            "recognizer": rec_state,
            "ocr_meta": self.CurrentOcrMetadata(engine),}, out_path)

    @staticmethod
    def CurrentOcrMetadata(engine: OCREngineExtractor) -> Dict[str, Any]:
        metadata = engine.OcrMetadata()
        if type(metadata) is not dict:
            raise TypeError("OCR metadata must be a dictionary")
        if set(metadata) != OCR_METADATA_FIELDS:
            raise RuntimeError("OCR metadata does not match the current schema")
        return metadata

    @classmethod
    def ValidateOcrMetadata(
        cls,
        engine: OCREngineExtractor,
        metadata: Any,) -> None:
        if type(metadata) is not dict or set(metadata) != OCR_METADATA_FIELDS:
            raise ValueError("OCR metadata fields do not match the current schema")
        if metadata != cls.CurrentOcrMetadata(engine):
            raise ValueError("OCR metadata does not match the configured model")

    @staticmethod
    def ValidateExactStateDict(
        module: nn.Module,
        state: Any,
        *,
        name: str,) -> None:
        if not isinstance(state, dict):
            raise TypeError(f"{name} state must be a dictionary")
        expected = module.state_dict()
        if set(state) != set(expected):
            raise ValueError(f"{name} state fields do not match the current model")
        for key, value in state.items():
            current = expected[key]
            if not torch.is_tensor(value):
                raise TypeError(f"{name} state {key} must be a tensor")
            if tuple(value.shape) != tuple(current.shape) or value.dtype != current.dtype:
                raise ValueError(f"{name} state {key} shape or dtype does not match")

    @staticmethod
    def ValidateOcrCheckpointCursor(
        checkpoint: Dict[str, Any],
        dataset: Dataset,) -> None:
        if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 0:
            raise ValueError("OCR checkpoint epoch must be a non-negative integer")
        if type(checkpoint["processed_sample_count_total"]) is not int or (
            checkpoint["processed_sample_count_total"] < 0
        ):
            raise ValueError("OCR checkpoint processed sample count is invalid")
        if type(checkpoint["best_val"]) not in (int, float):
            raise TypeError("OCR checkpoint best_val must be numeric")
        best_val = float(checkpoint["best_val"])
        if math.isnan(best_val) or best_val == float("-inf"):
            raise ValueError("OCR checkpoint best_val is invalid")
        split_indices: List[int] = []
        for field in ("train_indices", "val_indices", "test_indices"):
            indices = checkpoint[field]
            if type(indices) is not list or any(type(index) is not int for index in indices):
                raise TypeError(f"OCR checkpoint {field} must be a list of integers")
            if len(indices) == 0 or len(set(indices)) != len(indices):
                raise ValueError(f"OCR checkpoint {field} must be non-empty and unique")
            if any(index < 0 or index >= len(dataset) for index in indices):
                raise ValueError(f"OCR checkpoint {field} contains an invalid index")
            split_indices.extend(indices)
        if len(set(split_indices)) != len(split_indices):
            raise ValueError("OCR checkpoint dataset splits must be disjoint")
        if set(split_indices) != set(range(len(dataset))):
            raise ValueError("OCR checkpoint dataset splits must cover the dataset")

    def CaptureRngState(self) -> Dict[str, Any]:
        return {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None),}

    def RestoreRngState(self, state: Dict[str, Any]) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        np.random.set_state(state["numpy"])
        if torch.cuda.is_available() and state["cuda_all"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_all"])

    @staticmethod
    def ValidateTrainingRngState(state: Dict[str, Any]) -> None:
        if type(state) is not dict or set(state) != TRAIN_RNG_FIELDS:
            raise ValueError("training RNG fields do not match the current schema")

    def ShouldTriggerPeriodicSave(
        self,
        previousCount: int,
        currentCount: int,
        saveEverySampleCount: int,) -> bool:
        return (
            int(currentCount) > int(previousCount)
            and (int(currentCount) // int(saveEverySampleCount)) > (int(previousCount) // int(saveEverySampleCount)))

    def ComputeOcrBoxMatchStats(
        self,
        predBoxes: List[Union[np.ndarray, torch.Tensor, Tuple[int, int, int, int]]],
        gtBoxes: List[Union[np.ndarray, torch.Tensor, Tuple[int, int, int, int]]],
        *,
        iouThresh: float = 0.7,) -> Tuple[int, int, int]:
        pred_list = [
            tuple(float(v) for v in np.asarray(box, dtype=np.float32).reshape(-1)[:4].tolist())
            for box in predBoxes]
        gt_list = [
            tuple(float(v) for v in np.asarray(box, dtype=np.float32).reshape(-1)[:4].tolist())
            for box in gtBoxes]

        candidates: List[Tuple[float, int, int]] = []
        for pred_idx, pred_box in enumerate(pred_list):
            for gt_idx, gt_box in enumerate(gt_list):
                iou = float(IouXyxy(pred_box, gt_box))
                if iou >= float(iouThresh):
                    candidates.append((iou, pred_idx, gt_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_pred = set()
        matched_gt = set()
        tp = 0
        for _, pred_idx, gt_idx in candidates:
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
            tp += 1

        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(gt_list) - tp)
        return tp, fp, fn

    def ComputeOcrCharMatchStats(
        self,
        predText: str,
        targetText: str,) -> Tuple[int, int]:
        pred = str(predText)
        target = str(targetText)
        aligned_len = min(len(pred), len(target))
        correct = sum(1 for idx in range(aligned_len) if pred[idx] == target[idx])
        total = max(len(pred), len(target))
        return correct, total

    def ComputeOcrHmean(
        self,
        tp: int,
        fp: int,
        fn: int,) -> float:
        precision = float(tp) / max(1.0, float(tp + fp))
        recall = float(tp) / max(1.0, float(tp + fn))
        if precision + recall <= 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)


    def LoadTorchPayload(self, path: str):
        return torch.load(path, map_location=self.device, weights_only=True)

    def LoadBrainWeights(
        self,
        brain: BrainCore,
        path: str,
        *,
        agent: Optional[Agent] = None,) -> None:
        payload = self.LoadTorchPayload(path)
        if type(payload) is not dict or set(payload) != MODULE_PARAMETER_FIELDS:
            raise TypeError(
                f"checkpoint {path} brain-weight fields do not match the current schema")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != TRAIN_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported brain parameter schema {payload['schema_version']!r}")
        if payload["calibration_id"] != brain.calibration_id:
            raise ValueError(
                "brain parameter calibration_id does not match configured K")
        if payload["model_contract_id"] != brain.model_contract_id:
            raise ValueError(
                "brain parameter model_contract_id does not match the foundation")
        brain_state = payload["brain"]
        if type(brain_state) is not dict:
            raise TypeError("brain model state must be a dictionary")

        LoadDeploymentModelState(brain, brain_state)
        if agent is not None:
            agent.ResetOnlineCandidateState()
            agent.ClearTrainableOptimizerState()

    def LoadCognitiveBackboneParameters(
        self,
        brain: BrainCore,
        path: str,
        *,
        agent: Optional[Agent] = None,
    ) -> None:
        if type(path) is not str or not path.strip():
            raise ValueError("cognitive backbone path must be non-empty")
        artifact = self.LoadTorchPayload(path)
        LoadCognitiveBackboneState(brain, artifact)
        if agent is not None:
            agent.ResetOnlineCandidateState()
            agent.ClearTrainableOptimizerState()

    def ApplyParameterOverrideAfterResume(
        self,
        *,
        enabled: bool,
        parameterPath: Optional[str],
        loadFn: Callable[[str], None],
        logPrefix: str,) -> bool:
        if not enabled:
            return False
        if not parameterPath:
            raise ValueError(
                f"[{logPrefix}] parameter override path must not be empty")

        override_path = Path(parameterPath)
        if not override_path.is_file():
            raise FileNotFoundError(
                f"[{logPrefix}] parameter override file not found: {override_path}")

        loadFn(str(override_path))
        print(f"[{logPrefix}] checkpoint weights overridden from parameter file: {override_path}")
        return True

    def TryLoadOCRTrainingArtifact(
        self,
        path: Optional[str],
        loadFn: Callable[[str], Any],
        *,
        logPrefix: str,
        artifactName: str,
    ) -> Tuple[bool, Any]:
        if not path:
            return False, None
        artifactPath = Path(path)
        if not artifactPath.is_file():
            return False, None
        try:
            state = loadFn(str(artifactPath))
        except (
            EOFError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            print(
                f"[{logPrefix}] {artifactName} is incompatible and will not be loaded: "
                f"{artifactPath} ({error})")
            return False, None
        print(f"[{logPrefix}] loaded {artifactName}: {artifactPath}")
        return True, state

    def ConfigureOCRTrainingTargets(
        self,
        engine: OCREngineExtractor,
        *,
        trainDetection: bool,
        trainRecognition: bool,) -> None:
        for p in engine.backbone.parameters():
            p.requires_grad = bool(trainDetection)
        for p in engine.dbHead.parameters():
            p.requires_grad = bool(trainDetection)
        for p in engine.recognizer.parameters():
            p.requires_grad = bool(trainRecognition)

    def LoadOCRWeightsIntoEngine(self, engine: OCREngineExtractor, path: str) -> None:
        payload = torch.load(
            path,
            map_location=self.device,
            weights_only=True)
        if type(payload) is not dict or set(payload) != OCR_MODULE_PARAMETER_FIELDS:
            raise ValueError("OCR parameter fields do not match the current schema")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != OCR_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("OCR parameter schema is unsupported")
        self.ValidateOcrMetadata(engine, payload["ocr_meta"])
        self.ValidateExactStateDict(engine, payload["ocr"], name="OCR")
        engine.load_state_dict(payload["ocr"], strict=True)

    def LoadRecognizerWeightsIntoEngine(self, engine: OCREngineExtractor, path: str) -> None:
        payload = torch.load(
            path,
            map_location=self.device,
            weights_only=True)
        if type(payload) is not dict or set(payload) != OCR_RECOGNIZER_PARAMETER_FIELDS:
            raise ValueError(
                "OCR recognizer parameter fields do not match the current schema")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != OCR_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("OCR recognizer parameter schema is unsupported")
        self.ValidateOcrMetadata(engine, payload["ocr_meta"])
        self.ValidateExactStateDict(
            engine.recognizer,
            payload["recognizer"],
            name="OCR recognizer")
        engine.recognizer.load_state_dict(payload["recognizer"], strict=True)

    def LoadOCRCheckpoint(
        self,
        engine: OCREngineExtractor,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        path: str,
        *,
        trainDetection: bool,
        trainRecognition: bool,):
        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False)
        if type(ckpt) is not dict or set(ckpt) != OCR_TRAIN_CHECKPOINT_FIELDS:
            raise ValueError("OCR checkpoint fields do not match the current schema")
        if (
            type(ckpt["schema_version"]) is not int
            or ckpt["schema_version"] != OCR_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("OCR checkpoint schema is unsupported")
        if (
            type(ckpt["train_detection"]) is not bool
            or ckpt["train_detection"] != trainDetection
            or type(ckpt["train_recognition"]) is not bool
            or ckpt["train_recognition"] != trainRecognition
        ):
            raise ValueError("OCR checkpoint training mode does not match")
        self.ValidateTrainingRngState(ckpt["rng"])
        self.ValidateOcrMetadata(engine, ckpt["ocr_meta"])
        self.ValidateExactStateDict(engine, ckpt["ocr"], name="OCR")
        self.ValidateOcrCheckpointCursor(ckpt, dataset)

        engine.load_state_dict(ckpt["ocr"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        self.RestoreRngState(ckpt["rng"])

        train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
        val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
        test_ds = torch.utils.data.Subset(dataset, ckpt["test_indices"])

        start_epoch = ckpt["epoch"]
        best_val = float(ckpt["best_val"])
        processed_sample_count_total = ckpt["processed_sample_count_total"]
        return start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds

    def LoadOCRRecognizerCheckpoint(
        self,
        engine: OCREngineExtractor,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        path: str,):
        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False)
        if (
            type(ckpt) is not dict
            or set(ckpt) != OCR_RECOGNIZER_TRAIN_CHECKPOINT_FIELDS
        ):
            raise ValueError(
                "OCR recognizer checkpoint fields do not match the current schema")
        if (
            type(ckpt["schema_version"]) is not int
            or ckpt["schema_version"] != OCR_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError("OCR recognizer checkpoint schema is unsupported")
        self.ValidateTrainingRngState(ckpt["rng"])
        self.ValidateOcrMetadata(engine, ckpt["ocr_meta"])
        self.ValidateExactStateDict(
            engine.recognizer,
            ckpt["recognizer"],
            name="OCR recognizer")
        self.ValidateOcrCheckpointCursor(ckpt, dataset)

        engine.recognizer.load_state_dict(ckpt["recognizer"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        self.RestoreRngState(ckpt["rng"])

        train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
        val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
        test_ds = torch.utils.data.Subset(dataset, ckpt["test_indices"])

        start_epoch = ckpt["epoch"]
        best_val = float(ckpt["best_val"])
        processed_sample_count_total = ckpt["processed_sample_count_total"]
        return start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds

    def OCRTrainLoop(
        self,
        epochs: int,
        batchSize: int,
        valSplit: float,
        resume: bool,
        *,
        saveEverySampleCount = 2000,
        isTest: bool,
        ckptPath: str,
        outPath: str,
        trainDetection: bool = True,
        trainRecognition: bool = True,
        recognizerInitPath: Optional[str] = None,
        overrideCheckpointWithModuleParams: bool = False,):
        try:

            self.ResetControllerFlags()
    
            if not trainDetection and not trainRecognition:
                raise ValueError("OCRTrainLoop requires trainDetection or trainRecognition to be True")

            ds = OfflineOCRDataset(isTest=isTest)

            def BuildOCRTrainingState():
                engineValue = OCREngineExtractor().to(self.device)
                self.ConfigureOCRTrainingTargets(
                    engineValue,
                    trainDetection=trainDetection,
                    trainRecognition=trainRecognition,)
                trainableParameters = [
                    parameter
                    for parameter in engineValue.parameters()
                    if parameter.requires_grad]
                if len(trainableParameters) == 0:
                    raise RuntimeError("no trainable OCR parameters selected")
                optimizerValue = torch.optim.AdamW(
                    trainableParameters,
                    lr=3e-4,
                    weight_decay=1e-2)
                return engineValue, optimizerValue

            engine, optimizer = BuildOCRTrainingState()

            start_epoch = 0
            best_val = float("inf")
            processed_sample_count_total = 0
            train_ds = val_ds = test_ds = None

            testSplit = 0.1
            checkpointLoaded = False
            parameterLoaded = False
            recognizerLoaded = False
            shouldLoadArtifacts = bool(resume or overrideCheckpointWithModuleParams)
            checkpointExists = bool(resume and Path(ckptPath).is_file())
            if checkpointExists and not overrideCheckpointWithModuleParams:
                checkpointLoaded, checkpointState = self.TryLoadOCRTrainingArtifact(
                    ckptPath,
                    lambda path: self.LoadOCRCheckpoint(
                        engine,
                        optimizer,
                        ds,
                        path,
                        trainDetection=trainDetection,
                        trainRecognition=trainRecognition),
                    logPrefix="TrainOCR",
                    artifactName="training checkpoint")
                if checkpointLoaded:
                    start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds = checkpointState
                else:
                    engine, optimizer = BuildOCRTrainingState()
            if shouldLoadArtifacts and not checkpointLoaded:
                parameterLoaded, _ = self.TryLoadOCRTrainingArtifact(
                    outPath,
                    lambda path: self.LoadOCRWeightsIntoEngine(engine, path),
                    logPrefix="TrainOCR",
                    artifactName="model parameters")
                if not parameterLoaded:
                    recognizerLoaded, _ = self.TryLoadOCRTrainingArtifact(
                        recognizerInitPath,
                        lambda path: self.LoadRecognizerWeightsIntoEngine(engine, path),
                        logPrefix="TrainOCR",
                        artifactName="recognizer parameters")
                if not parameterLoaded and not recognizerLoaded:
                    print(
                        "[TrainOCR] no compatible checkpoint or parameter file was found; "
                        "training a new model")
            self.ConfigureOCRTrainingTargets(
                engine,
                trainDetection=trainDetection,
                trainRecognition=trainRecognition,)

            train_ds, val_ds, test_ds = self.SplitDataset(
                ds,
                valSplit=valSplit,
                testSplit=testSplit,
                trainDataset=train_ds,
                valDataset=val_ds,
                testDataset=test_ds)

            def CollateOcrBatch(batch):
                imgs, boxes, texts, ignore_flags = zip(*batch)
                return list(imgs), list(boxes), list(texts), list(ignore_flags)

            train_dl = self.CreateDataLoader(
                train_ds,
                batchSize=batchSize,
                shuffle=True,
                collateFn=CollateOcrBatch)

            val_dl = self.CreateDataLoader(
                val_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=CollateOcrBatch)

            test_dl = self.CreateDataLoader(
                test_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=CollateOcrBatch)

            patience = 10
            min_delta = 1e-4
            no_improve = 0
            box_metric_threshold = 0.95
            text_metric_threshold = 0.95
            box_iou_threshold = 0.7
            validation_interval = max(1, int(epochs) // 10)

            def EvaluateSplit(dl):
                engine.eval()
                split_loss = 0.0
                split_det_loss = 0.0
                split_rec_loss = 0.0
                split_samples = 0
                box_tp = 0
                box_fp = 0
                box_fn = 0
                total_char_correct = 0
                total_char_count = 0

                with torch.no_grad():
                    for imgs_b, boxes_b, texts_b, ignore_b in dl:
                        for img, boxes, texts, ignore_flags in zip(imgs_b, boxes_b, texts_b, ignore_b):
                            sample: Dict[str, Any] = DataPreprocessor.PrepareOCRSample(
                                img, 
                                boxes,
                                texts,
                                ignoreFlags=ignore_flags,
                                char2Idx=engine.char2Idx,
                                imageSize=768,
                                targetH=32,
                                maxW=512,
                                device=self.device,)

                            zero = sample["detect_img"].new_zeros(())
                            det_loss = zero
                            det_boxes_pred: List[np.ndarray] = []
                            if trainDetection:
                                detect_img = sample["detect_img"].unsqueeze(0) # [1, 3, H, W]
                                gt_boxes = sample["gt_boxes"].unsqueeze(0) # [1, 1, H, W]
                                gt_mask = sample["gt_mask"].unsqueeze(0) # [1, 1, H, W]

                                det_out = engine.ForwardDetect(
                                    detect_img,
                                    gtBoxes=gt_boxes,
                                    gtMask=gt_mask,)
                                det_loss = det_out["loss"] # []
                                det_boxes_pred = engine.BitmapToBoxes(
                                    det_out["prob_map"][0],
                                    threshValue=0.3,
                                    minArea=10)
                                tp, fp, fn = self.ComputeOcrBoxMatchStats(
                                    det_boxes_pred,
                                    sample["det_boxes"],
                                    iouThresh=box_iou_threshold)
                                box_tp += tp
                                box_fp += fp
                                box_fn += fn

                            rec_loss = zero # []
                            recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                            recog_widths = sample["recog_widths"] # [N_line]
                            targets = sample["targets"] # [sum(target_lengths)]
                            target_lengths = sample["target_lengths"] # [N_line]
                            norm_texts = sample["norm_texts"]

                            if trainRecognition and recog_imgs.size(0) > 0 and targets.numel() > 0:
                                rec_out = engine.ForwardRecognize(
                                    recog_imgs,
                                    targetsTensor=targets,
                                    targetLengths=target_lengths,
                                    validWidths=recog_widths,)
                                rec_loss = rec_out["loss"] # []

                                pairs = engine.CtcGreedyDecodeWithConf(
                                    rec_out["log_probs"], # [T, N_line, C_vocab]
                                    idx2Char=engine.idx2Char,
                                    blankIndex=engine.blankIndex,)
                                pred_texts = [txt for txt, _ in pairs]
                                for pred_text, target_text in zip(pred_texts, norm_texts):
                                    correct_chars, total_chars = self.ComputeOcrCharMatchStats(
                                        pred_text,
                                        target_text)
                                    total_char_correct += correct_chars
                                    total_char_count += total_chars

                            split_loss += float((det_loss + rec_loss).item())
                            split_det_loss += float(det_loss.item())
                            split_rec_loss += float(rec_loss.item())
                            split_samples += 1

                avg_split_loss = split_loss / max(1, split_samples)
                avg_split_det_loss = split_det_loss / max(1, split_samples)
                avg_split_rec_loss = split_rec_loss / max(1, split_samples)
                box_hmean = self.ComputeOcrHmean(box_tp, box_fp, box_fn) if trainDetection else 0.0
                text_char_acc = (total_char_correct / max(1, total_char_count)) if trainRecognition else 0.0
                return avg_split_loss, avg_split_det_loss, avg_split_rec_loss, box_hmean, text_char_acc

            def BuildOCRCheckpointPayload(epochValue: int) -> Dict[str, Any]:
                return {
                    "schema_version": OCR_CHECKPOINT_SCHEMA_VERSION,
                    "epoch": int(epochValue),
                    "best_val": best_val,
                    "ocr": engine.state_dict(),
                    "ocr_meta": self.CurrentOcrMetadata(engine),
                    "optimizer": optimizer.state_dict(),
                    "train_indices": list(train_ds.indices),
                    "val_indices": list(val_ds.indices),
                    "test_indices": list(test_ds.indices),
                    "processed_sample_count_total": processed_sample_count_total,
                    "rng": self.CaptureRngState(),
                    "train_detection": bool(trainDetection),
                    "train_recognition": bool(trainRecognition),}

            def SaveOCRTrainingArtifacts(epochValue: int, *, logPeriodic: bool = False) -> None:
                self.SaveOCRParameters(engine, outPath)
                ckpt_dir = Path(ckptPath).parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.AtomicTorchSave(
                    BuildOCRCheckpointPayload(epochValue),
                    ckptPath)
                if logPeriodic:
                    print(
                        f"[TrainOCR] periodic save at processed_sample_count_total={processed_sample_count_total} "
                        f"(epoch {ep + 1}, batch {bi})")

            self.controller.SetStatus(
                "training",(
                    "OCR training started"
                    if trainDetection and trainRecognition else
                    "OCR training started (detect only)"
                    if trainDetection else
                    "OCR training started (recognize only)"),
                epoch=start_epoch,
                total_epochs=epochs,
                batch=0,
                total_batches=len(train_dl),
                visual=self.controller.EmptyVisualStatus(touch=True),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR training stopped")
                    break

                if not self.WaitWhilePaused("OCR training paused"):
                    break

                engine.train()
                epoch_loss = 0.0
                epoch_det_loss = 0.0
                epoch_rec_loss = 0.0
                nb = 0

                for bi, batch in enumerate(train_dl, start=1):
                    imgs_b, boxes_b, texts_b, ignore_b = batch
                    sample_losses: List[torch.Tensor] = []
                    sample_det_losses: List[float] = []
                    sample_rec_losses: List[float] = []
                    batch_char_correct = 0
                    batch_char_total = 0
                    latest_visual = None

                    for img, boxes, texts, ignore_flags in zip(imgs_b, boxes_b, texts_b, ignore_b):
                        sample: Dict[str, Any] = DataPreprocessor.PrepareOCRSample(
                            img,  
                            boxes, 
                            texts,
                            ignoreFlags=ignore_flags,
                            char2Idx=engine.char2Idx,
                            imageSize=768,
                            targetH=32,
                            maxW=512,
                            device=self.device,)

                        zero = sample["detect_img"].new_zeros(())
                        detect_img = sample["detect_img"].unsqueeze(0) # [1, 3, H, W]
                        det_loss = zero
                        det_forward_out: Optional[Dict[str, torch.Tensor]] = None
                        if trainDetection:
                            gt_boxes = sample["gt_boxes"].unsqueeze(0) # [1, 1, H, W]
                            gt_mask = sample["gt_mask"].unsqueeze(0) # [1, 1, H, W]

                            det_forward_out = engine.ForwardDetect(
                                detect_img,
                                gtBoxes=gt_boxes,
                                gtMask=gt_mask,)
                            det_loss = det_forward_out["loss"] # []

                        rec_loss = zero # []
                        recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                        recog_widths = sample["recog_widths"] # [N_line]
                        targets = sample["targets"] # [sum(target_lengths)]
                        target_lengths = sample["target_lengths"] # [N_line]
                        norm_texts = sample["norm_texts"]
                        rec_forward_texts: List[str] = []

                        if trainRecognition and recog_imgs.size(0) > 0 and targets.numel() > 0:
                            rec_out = engine.ForwardRecognize(
                                recog_imgs,
                                targetsTensor=targets,
                                targetLengths=target_lengths,
                                validWidths=recog_widths,)
                            rec_loss = rec_out["loss"] # []

                            with torch.no_grad():
                                pairs = engine.CtcGreedyDecodeWithConf(
                                    rec_out["log_probs"].detach(), # [T, N_line, C_vocab]
                                    idx2Char=engine.idx2Char,
                                    blankIndex=engine.blankIndex,)
                                rec_forward_texts = [txt for txt, _ in pairs]
                                for pred_text, target_text in zip(rec_forward_texts, norm_texts):
                                    correct_chars, total_chars = self.ComputeOcrCharMatchStats(
                                        pred_text,
                                        target_text)
                                    batch_char_correct += correct_chars
                                    batch_char_total += total_chars

                        pred_ocr_items: List[Dict[str, Any]] = []
                        pred_ocr_texts: List[str] = []
                        if self.controller.IsVisualStateEnabled():
                            with torch.no_grad():
                                if det_forward_out is None:
                                    det_forward_out = engine.ForwardDetect(detect_img.detach())

                                prob_map_vis = det_forward_out["prob_map"].detach()
                                pm_np = prob_map_vis[0].cpu().squeeze(0).numpy()
                                h_map, w_map = pm_np.shape
                                forward_boxes = engine.BitmapToBoxes(
                                    prob_map_vis[0],
                                    threshValue=0.3,
                                    minArea=10)

                                for box in forward_boxes:
                                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                                    x1 = max(0, x1)
                                    y1 = max(0, y1)
                                    x2 = min(w_map, x2)
                                    y2 = min(h_map, y2)
                                    region = pm_np[y1:y2, x1:x2]
                                    det_score = float(region.mean()) if region.size != 0 else 0.0
                                    pred_ocr_items.append({
                                        "box": (x1, y1, x2, y2),
                                        "text": "",
                                        "det_score": float(det_score),
                                        "score": float(det_score),})

                                restored_pred_box_items = self.RestoreOcrItemsToOriginal(
                                    pred_ocr_items,
                                    sample["resize_meta"])
                                pred_ocr_texts.append("Forward boxes:")
                                if restored_pred_box_items:
                                    for item in restored_pred_box_items:
                                        box_value = item.get("box", None)
                                        det_score = float(item.get("det_score", 0.0))
                                        if isinstance(box_value, (list, tuple)) and len(box_value) == 4:
                                            x1, y1, x2, y2 = [int(v) for v in box_value]
                                            pred_ocr_texts.append(
                                                f"[{x1}, {y1}, {x2}, {y2}] det={det_score:.3f}")
                                else:
                                    pred_ocr_texts.append("<empty>")

                                pred_ocr_texts.append("Forward texts:")
                                rec_forward_items: List[Dict[str, Any]] = []
                                for box, text_value in zip(sample["rec_boxes"], rec_forward_texts):
                                    x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                                    rec_forward_items.append({
                                        "box": (x1, y1, x2, y2),
                                        "text": str(text_value).strip(),})

                                restored_rec_forward_items = self.RestoreOcrItemsToOriginal(
                                    rec_forward_items,
                                    sample["resize_meta"])
                                if restored_rec_forward_items:
                                    for item, target_text in zip(restored_rec_forward_items, norm_texts):
                                        box_value = item.get("box", None)
                                        pred_text = str(item.get("text", "")).strip()
                                        display_text = pred_text if pred_text != "" else "<blank>"
                                        if isinstance(box_value, (list, tuple)) and len(box_value) == 4:
                                            x1, y1, x2, y2 = [int(v) for v in box_value]
                                            pred_ocr_texts.append(
                                                f"[{x1}, {y1}, {x2}, {y2}] pred={display_text} | gt={target_text}")
                                        else:
                                            pred_ocr_texts.append(
                                                f"pred={display_text} | gt={target_text}")
                                else:
                                    pred_ocr_texts.append("<empty>")

                            latest_visual = self.BuildVisualPayload(
                                img,
                                ocrTexts=pred_ocr_texts,
                                ocrItems=pred_ocr_items,
                                resizeMeta=sample["resize_meta"],
                                title="OCR Train",
                                extraLines=[
                                    f"epoch {ep + 1}/{epochs}",
                                    f"batch {bi}/{len(train_dl)}"],)

                        sample_losses.append(det_loss + rec_loss)
                        sample_det_losses.append(float(det_loss.item()))
                        sample_rec_losses.append(float(rec_loss.item()))

                    if not sample_losses:
                        continue

                    loss = torch.stack(sample_losses).mean() # []
                    batch_det_loss = sum(sample_det_losses) / max(1, len(sample_det_losses))
                    batch_rec_loss = sum(sample_rec_losses) / max(1, len(sample_rec_losses))
                    train_text_char_acc = (batch_char_correct / max(1, batch_char_total)) if trainRecognition else 0.0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(engine.parameters(), 3.0)
                    optimizer.step()

                    previous_processed_sample_count_total = processed_sample_count_total
                    processed_sample_count_total += len(sample_losses)
                    if self.ShouldTriggerPeriodicSave(
                        previous_processed_sample_count_total,
                        processed_sample_count_total,
                        saveEverySampleCount):
                        SaveOCRTrainingArtifacts(ep, logPeriodic=True)

                    epoch_loss += float(loss.item())
                    epoch_det_loss += batch_det_loss
                    epoch_rec_loss += batch_rec_loss
                    nb += 1

                    status_kwargs = {
                        "epoch": ep + 1,
                        "total_epochs": epochs,
                        "batch": bi,
                        "total_batches": len(train_dl),
                        "train_loss": float(loss.item()),}
                    if latest_visual is not None:
                        status_kwargs["visual"] = latest_visual

                    self.controller.SetStatus(
                        "training",(
                            f"OCR training... total_loss={float(loss.item()):.4f}, box_loss={batch_det_loss:.4f}, rec_loss={batch_rec_loss:.4f}, text_char_acc={train_text_char_acc:.3f}"
                            if trainRecognition else
                            f"OCR training... total_loss={float(loss.item()):.4f}, box_loss={batch_det_loss:.4f}, rec_loss={batch_rec_loss:.4f}"),
                        **status_kwargs)

                    if self.controller.ShouldStop():
                        break

                    if not self.WaitWhilePaused("OCR training paused"):
                        break

                avg_train = epoch_loss / max(1, nb)
                avg_train_box_loss = epoch_det_loss / max(1, nb)
                avg_train_rec_loss = epoch_rec_loss / max(1, nb)
                should_validate = (((ep + 1) % validation_interval) == 0) or ((ep + 1) == epochs)
                if should_validate:
                    avg_val, avg_val_box_loss, avg_val_rec_loss, val_box_hmean, val_text_char_acc = EvaluateSplit(val_dl)
                    test_loss, test_box_loss, test_rec_loss, test_box_hmean, test_text_char_acc = EvaluateSplit(test_dl)

                    improved = (best_val - avg_val) > min_delta
                    if improved:
                        best_val = avg_val
                        no_improve = 0
                        SaveOCRTrainingArtifacts(ep + 1)
                    else:
                        no_improve += 1

                    self.controller.SetStatus(
                        "training",(
                            f"OCR epoch {ep+1}/{epochs} done | "
                            f"train_loss {avg_train:.4f} (box={avg_train_box_loss:.4f}, rec={avg_train_rec_loss:.4f}) | "
                            f"val_loss {avg_val:.4f} (box={avg_val_box_loss:.4f}, rec={avg_val_rec_loss:.4f}), "
                            f"val_box_hmean={(f'{float(val_box_hmean):.3f}' if trainDetection else 'n/a')}, "
                            f"val_text_char_acc={(f'{float(val_text_char_acc):.3f}' if trainRecognition else 'n/a')} | "
                            f"test_loss {test_loss:.4f} (box={test_box_loss:.4f}, rec={test_rec_loss:.4f}), "
                            f"test_box_hmean={(f'{float(test_box_hmean):.3f}' if trainDetection else 'n/a')}, "
                            f"test_text_char_acc={(f'{float(test_text_char_acc):.3f}' if trainRecognition else 'n/a')}"),
                        epoch=ep + 1,
                        total_epochs=epochs,
                        val_loss=avg_val,)

                    loss_stabilized = (no_improve >= patience)
                    box_metric_ready = (
                        (not trainDetection)
                        or (val_box_hmean >= box_metric_threshold and test_box_hmean >= box_metric_threshold))
                    text_metric_ready = (
                        (not trainRecognition)
                        or (val_text_char_acc >= text_metric_threshold and test_text_char_acc >= text_metric_threshold))

                    if loss_stabilized and box_metric_ready and text_metric_ready:
                        if trainDetection and trainRecognition:
                            completion_msg = (
                                "OCR validation loss stabilized and both box_hmean/text_char_acc "
                                "reached threshold on val/test, early stop.")
                        elif trainDetection:
                            completion_msg = (
                                "OCR validation loss stabilized and box_hmean reached threshold "
                                "on val/test, early stop.")
                        else:
                            completion_msg = (
                                "OCR validation loss stabilized and text_char_acc reached threshold "
                                "on val/test, early stop.")
                        self.controller.SetStatus("completed", completion_msg)
                        break
                else:
                    next_eval_epoch = min(epochs, (((ep + 1) // validation_interval) + 1) * validation_interval)
                    self.controller.SetStatus(
                        "training",
                        (
                            f"OCR epoch {ep+1}/{epochs} done | "
                            f"train_loss {avg_train:.4f} (box={avg_train_box_loss:.4f}, rec={avg_train_rec_loss:.4f}) | "
                            f"validation skipped (interval={validation_interval}, next={next_eval_epoch})"
                        ),
                        epoch=ep + 1,
                        total_epochs=epochs,)

                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR training stopped")
                    break

            else:
                self.controller.SetStatus("completed", "OCR training completed")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"OCR training error: {e}", trace=tb)
        finally:
            self.is_begin = False

    def OCRRecognitionTrainLoop(
        self,
        epochs: int,
        batchSize: int,
        valSplit: float,
        resume: bool,
        *,
        saveEverySampleCount = 2000,
        isTest: bool,
        ckptPath: str,
        outPath: str,
        overrideCheckpointWithModuleParams: bool = False,):
        try:

            self.ResetControllerFlags()
    
            ds = OfflineOCRRecognitionDataset(isTest=isTest)

            def BuildOCRRecognizerTrainingState():
                engineValue = OCREngineExtractor().to(self.device)
                self.ConfigureOCRTrainingTargets(
                    engineValue,
                    trainDetection=False,
                    trainRecognition=True,)
                trainableParameters = [
                    parameter
                    for parameter in engineValue.parameters()
                    if parameter.requires_grad]
                if len(trainableParameters) == 0:
                    raise RuntimeError("no trainable OCR recognizer parameters selected")
                optimizerValue = torch.optim.AdamW(
                    trainableParameters,
                    lr=3e-4,
                    weight_decay=1e-2)
                return engineValue, optimizerValue

            engine, optimizer = BuildOCRRecognizerTrainingState()

            start_epoch = 0
            best_val = float("inf")
            processed_sample_count_total = 0
            train_ds = val_ds = test_ds = None

            testSplit = 0.1
            checkpointLoaded = False
            shouldLoadArtifacts = bool(resume or overrideCheckpointWithModuleParams)
            checkpointExists = bool(resume and Path(ckptPath).is_file())
            if checkpointExists and not overrideCheckpointWithModuleParams:
                checkpointLoaded, checkpointState = self.TryLoadOCRTrainingArtifact(
                    ckptPath,
                    lambda path: self.LoadOCRRecognizerCheckpoint(
                        engine,
                        optimizer,
                        ds,
                        path),
                    logPrefix="TrainOCRRec",
                    artifactName="training checkpoint")
                if checkpointLoaded:
                    start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds = checkpointState
                else:
                    engine, optimizer = BuildOCRRecognizerTrainingState()
            if shouldLoadArtifacts and not checkpointLoaded:
                parameterLoaded, _ = self.TryLoadOCRTrainingArtifact(
                    outPath,
                    lambda path: self.LoadRecognizerWeightsIntoEngine(engine, path),
                    logPrefix="TrainOCRRec",
                    artifactName="recognizer parameters")
                if not parameterLoaded:
                    print(
                        "[TrainOCRRec] no compatible checkpoint or parameter file was found; "
                        "training a new recognizer")
            self.ConfigureOCRTrainingTargets(
                engine,
                trainDetection=False,
                trainRecognition=True,)

            train_ds, val_ds, test_ds = self.SplitDataset(
                ds,
                valSplit=valSplit,
                testSplit=testSplit,
                trainDataset=train_ds,
                valDataset=val_ds,
                testDataset=test_ds)

            def CollateRecognitionBatch(batch):
                imgs, texts, ignore_flags = zip(*batch)
                return list(imgs), list(texts), list(ignore_flags)

            train_dl = self.CreateDataLoader(
                train_ds,
                batchSize=batchSize,
                shuffle=True,
                collateFn=CollateRecognitionBatch)

            val_dl = self.CreateDataLoader(
                val_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=CollateRecognitionBatch)

            test_dl = self.CreateDataLoader(
                test_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=CollateRecognitionBatch)

            patience = 5
            min_delta = 1e-4
            no_improve = 0

            def EvaluateSplit(dl):
                engine.eval()
                split_loss = 0.0
                split_samples = 0
                total_correct = 0
                total_elems = 0

                with torch.no_grad():
                    for imgs_b, texts_b, ignore_b in dl:
                        for img, text, ignore_flag in zip(imgs_b, texts_b, ignore_b):
                            sample = DataPreprocessor.PrepareOCRRecognitionSample(
                                img,
                                text,
                                ignoreFlag=ignore_flag,
                                char2Idx=engine.char2Idx,
                                targetH=32,
                                maxW=512,
                                device=self.device,)

                            recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                            recog_widths = sample["recog_widths"] # [N_line]
                            targets = sample["targets"] # [sum(target_lengths)]
                            target_lengths = sample["target_lengths"] # [N_line]
                            norm_text = sample["norm_text"]
                            if recog_imgs.size(0) == 0 or targets.numel() == 0:
                                continue

                            rec_out = engine.ForwardRecognize(
                                recog_imgs,
                                targetsTensor=targets,
                                targetLengths=target_lengths,
                                validWidths=recog_widths,)
                            rec_loss = rec_out["loss"] # []

                            pairs = engine.CtcGreedyDecodeWithConf(
                                rec_out["log_probs"], # [T, N_line, C_vocab]
                                idx2Char=engine.idx2Char,
                                blankIndex=engine.blankIndex,)
                            pred_text = pairs[0][0] if len(pairs) > 0 else ""
                            total_correct += int(pred_text == norm_text)
                            total_elems += 1
                            split_loss += float(rec_loss.item())
                            split_samples += 1

                avg_split_loss = split_loss / max(1, split_samples)
                split_acc = (total_correct / total_elems if total_elems > 0 else 0.0)
                return avg_split_loss, split_acc

            def BuildOCRRecognizerCheckpointPayload(epochValue: int) -> Dict[str, Any]:
                return {
                    "schema_version": OCR_CHECKPOINT_SCHEMA_VERSION,
                    "epoch": int(epochValue),
                    "best_val": best_val,
                    "recognizer": engine.recognizer.state_dict(),
                    "ocr_meta": self.CurrentOcrMetadata(engine),
                    "optimizer": optimizer.state_dict(),
                    "train_indices": list(train_ds.indices),
                    "val_indices": list(val_ds.indices),
                    "test_indices": list(test_ds.indices),
                    "processed_sample_count_total": processed_sample_count_total,
                    "rng": self.CaptureRngState(),}

            def SaveOCRRecognizerTrainingArtifacts(epochValue: int, *, logPeriodic: bool = False) -> None:
                self.SaveOCRRecognizerParameters(engine, outPath)
                ckpt_dir = Path(ckptPath).parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                self.AtomicTorchSave(
                    BuildOCRRecognizerCheckpointPayload(epochValue),
                    ckptPath)
                if logPeriodic:
                    print(
                        f"[TrainOCRRec] periodic save at processed_sample_count_total={processed_sample_count_total} "
                        f"(epoch {ep + 1}, batch {bi})")

            self.controller.SetStatus(
                "training",
                "OCR recognizer training started",
                epoch=start_epoch,
                total_epochs=epochs,
                batch=0,
                total_batches=len(train_dl),
                visual=self.controller.EmptyVisualStatus(touch=True),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR recognizer training stopped")
                    break

                if not self.WaitWhilePaused("OCR recognizer training paused"):
                    break

                engine.train()
                epoch_loss = 0.0
                nb = 0

                for bi, batch in enumerate(train_dl, start=1):
                    imgs_b, texts_b, ignore_b = batch
                    sample_losses: List[torch.Tensor] = []
                    batch_correct = 0
                    batch_elems = 0
                    latest_visual = None

                    for img, text, ignore_flag in zip(imgs_b, texts_b, ignore_b):
                        sample = DataPreprocessor.PrepareOCRRecognitionSample(
                            img,
                            text,
                            ignoreFlag=ignore_flag,
                            char2Idx=engine.char2Idx,
                            targetH=32,
                            maxW=512,
                            device=self.device,)

                        recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                        recog_widths = sample["recog_widths"] # [N_line]
                        targets = sample["targets"] # [sum(target_lengths)]
                        target_lengths = sample["target_lengths"] # [N_line]
                        norm_text = sample["norm_text"]
                        if recog_imgs.size(0) == 0 or targets.numel() == 0:
                            continue

                        rec_out = engine.ForwardRecognize(
                            recog_imgs,
                            targetsTensor=targets,
                            targetLengths=target_lengths,
                            validWidths=recog_widths,)
                        rec_loss = rec_out["loss"] # []

                        with torch.no_grad():
                            pairs = engine.CtcGreedyDecodeWithConf(
                                rec_out["log_probs"].detach(), # [T, N_line, C_vocab]
                                idx2Char=engine.idx2Char,
                                blankIndex=engine.blankIndex,)
                            pred_text = pairs[0][0] if len(pairs) > 0 else ""
                            batch_correct += int(pred_text == norm_text)
                            batch_elems += 1

                        if self.controller.IsVisualStateEnabled():
                            latest_visual = self.BuildVisualPayload(
                                img,
                                ocrTexts=([pred_text] if pred_text != "" else []),
                                title="OCR Recognition Train",
                                extraLines=[
                                    f"epoch {ep + 1}/{epochs}",
                                    f"batch {bi}/{len(train_dl)}"],
                                drawBoxes=False,)

                        sample_losses.append(rec_loss)

                    if not sample_losses:
                        continue

                    loss = torch.stack(sample_losses).mean() # []
                    rec_acc = (batch_correct / batch_elems) if batch_elems > 0 else 0.0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(engine.recognizer.parameters(), 3.0)
                    optimizer.step()

                    previous_processed_sample_count_total = processed_sample_count_total
                    processed_sample_count_total += len(sample_losses)
                    if self.ShouldTriggerPeriodicSave(
                        previous_processed_sample_count_total,
                        processed_sample_count_total,
                        saveEverySampleCount):
                        SaveOCRRecognizerTrainingArtifacts(ep, logPeriodic=True)

                    epoch_loss += float(loss.item())
                    nb += 1

                    status_kwargs = {
                        "epoch": ep + 1,
                        "total_epochs": epochs,
                        "batch": bi,
                        "total_batches": len(train_dl),
                        "train_loss": float(loss.item()),}
                    if latest_visual is not None:
                        status_kwargs["visual"] = latest_visual

                    self.controller.SetStatus(
                        "training",
                        f"OCR recognizer training... acc={rec_acc:.3f}",
                        **status_kwargs)

                    if self.controller.ShouldStop():
                        break

                    if not self.WaitWhilePaused("OCR recognizer training paused"):
                        break

                avg_train = epoch_loss / max(1, nb)
                avg_val, val_acc = EvaluateSplit(val_dl)
                test_loss, test_acc = EvaluateSplit(test_dl)

                improved = (best_val - avg_val) > min_delta
                if improved:
                    best_val = avg_val
                    no_improve = 0
                    SaveOCRRecognizerTrainingArtifacts(ep + 1)
                else:
                    no_improve += 1

                self.controller.SetStatus(
                    "training",(
                        f"OCR recognizer epoch {ep+1}/{epochs} done | "
                        f"train {avg_train:.4f} | "
                        f"val {avg_val:.4f}, acc={val_acc:.3f} | "
                        f"test {test_loss:.4f}, acc={test_acc:.3f}"),
                    epoch=ep + 1,
                    total_epochs=epochs,
                    val_loss=avg_val,)

                if no_improve >= patience:
                    self.controller.SetStatus("completed", "OCR recognizer validation stabilized, early stop.")
                    break

                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR recognizer training stopped")
                    break

            else:
                self.controller.SetStatus("completed", "OCR recognizer training completed")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"OCR recognizer training error: {e}", trace=tb)
        finally:
            self.is_begin = False
    def TrainLoop(
        self,
        epochs: int,
        batchSize: int,
        valSplit: float,
        resume: bool,
        onlineLearning: bool = False,
        *,
        saveEverySampleCount: int = 2000,
        isTest: bool = False,
        worldMemPath: Optional[str] = None,
        memMemPath: Optional[str] = None,
        ckptPath: Optional[str] = None,
        outPath: Optional[str] = None,
        overrideCheckpointWithModuleParams: bool = False,
        trainStage: str = "full",
    ) -> None:
        try:
            if type(epochs) is not int or epochs < 1:
                raise ValueError("epochs must be positive")
            if type(batchSize) is not int or batchSize < 1:
                raise ValueError("batchSize must be positive")
            if not 0.0 < float(valSplit) < 1.0:
                raise ValueError("valSplit must be between zero and one")
            if type(saveEverySampleCount) is not int or saveEverySampleCount < 1:
                raise ValueError("saveEverySampleCount must be positive")
            trainStage = self.NormalizeTrainStage(trainStage)
            critic_enabled = trainStage in ("full", "policy")
            checkpoint_path = str(
                ckptPath or BasicParameters.CKPT_PATH_TRAIN)
            output_path = str(
                outPath or (
                    BasicParameters.MODULEPARAMETER_PATH_TEST
                    if isTest
                    else BasicParameters.MODULEPARAMETER_PATH))
            self.ResetControllerFlags()
            projection = self.robot.ContractView.perception_projection
            if projection is None:
                raise RuntimeError("robot contract has no perception calibration")
            dataset = OfflineGameDataset(
                calibrationId=projection.calibration_id,
                sensorFrameName=projection.reference_frame_id,
                contractView=self.robot.ContractView,
                isTest=isTest)
            brain = BrainCore(
                brainBuildSpec=self.brain_build_spec,
                device=self.device,
                plasticOnlineLearning=onlineLearning,
                enablePerceptionSupervision=True)
            checkpoint_exists = resume and Path(checkpoint_path).is_file()
            if overrideCheckpointWithModuleParams:
                (
                    deployment_model_path,
                    initial_world_path,
                    initial_memory_path,
                ) = self.ResolveDeploymentArtifactPaths(
                    output_path,
                    calibrationId=projection.calibration_id,
                    brainBuildSpec=self.brain_build_spec)
            else:
                deployment_model_path = output_path
                initial_world_path = worldMemPath
                initial_memory_path = memMemPath
            agent = Agent(
                brain,
                isTrain=True,
                device=self.device,
                worldMemoryPath=initial_world_path,
                memMemoryPath=initial_memory_path)
            agent.BindWorldMemoryContext(
                dataset.world_frame_id,
                batchSize=batchSize,
                loadPersistent=(
                    not checkpoint_exists
                    and overrideCheckpointWithModuleParams))
            if onlineLearning and not checkpoint_exists:
                agent.ResetOnlineCandidateState()

            start_epoch = 0
            resume_batch_index = 0
            resume_epoch_loss = 0.0
            best_validation = float("inf")
            no_improve = 0
            processed_samples = 0
            train_dataset = None
            validation_dataset = None
            test_dataset = None

            if checkpoint_exists:
                resume_state = self.LoadCheckpoint(
                    brain,
                    agent,
                    dataset,
                    checkpoint_path,
                    batchSize=batchSize,
                    trainStage=trainStage,
                    onlineLearning=onlineLearning)
                start_epoch = resume_state.epoch
                resume_batch_index = resume_state.next_batch_index
                resume_epoch_loss = resume_state.epoch_loss_sum
                best_validation = resume_state.best_val
                no_improve = resume_state.no_improve
                processed_samples = (
                    resume_state.processed_sample_count_total)
                train_dataset = resume_state.train_dataset
                validation_dataset = resume_state.validation_dataset
                test_dataset = resume_state.test_dataset

            parameters_overridden = self.ApplyParameterOverrideAfterResume(
                enabled=overrideCheckpointWithModuleParams,
                parameterPath=deployment_model_path,
                loadFn=lambda path: self.LoadBrainWeights(
                    brain,
                    path,
                    agent=agent),
                logPrefix="Train")
            if parameters_overridden:
                resume_batch_index = 0
                resume_epoch_loss = 0.0
                best_validation = float("inf")
                no_improve = 0
            if onlineLearning and (
                not checkpoint_exists or parameters_overridden
            ):
                agent.UpdateWrappers(
                    tuple(self.TrainStageOnlineWrappers(brain, trainStage)),
                    "autogrow")

            train_dataset, validation_dataset, test_dataset = (
                self.SplitDatasetSequential(
                    dataset,
                    valSplit=valSplit,
                    testSplit=0.1,
                    trainDataset=train_dataset,
                    valDataset=validation_dataset,
                    testDataset=test_dataset))
            train_loader = self.CreateContractTrajectoryLoader(
                train_dataset,
                batchSize=batchSize)
            validation_loader = self.CreateContractTrajectoryLoader(
                validation_dataset,
                batchSize=batchSize)
            test_loader = self.CreateContractTrajectoryLoader(
                test_dataset,
                batchSize=batchSize)
            patience = 5
            min_delta = 1e-4

            def BuildActInput(
                batch: ContractOfflineBatch,
                deterministic: bool,
            ) -> ContractAgentActInput:
                perception_targets = dict(batch.perception_targets)
                perception_targets.update({
                    "rgb": batch.frames,
                    "depth": batch.depths,
                    "depth_valid": batch.depth_valid,
                })
                return ContractAgentActInput(
                    frame=batch.frames,
                    text_ext=batch.text_ext,
                    reward=batch.rewards,
                    done=batch.dones,
                    sample_actions=not deterministic,
                    deterministic_actor=deterministic,
                    depth=batch.depths,
                    depth_valid=batch.depth_valid,
                    feedback_packet=batch.feedback_packet,
                    text_trust=[
                        TEXT_TRUST_OPERATOR_COMMAND
                        for _ in batch.text_ext],
                    perception_targets=perception_targets)

            def BuildCheckpointPayload(
                epoch_value: int,
                next_batch_index: int,
                epoch_loss_sum: float,
            ) -> Dict[str, Any]:
                brain.mem.FlushPendingWrites()
                return {
                    "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                    "calibration_id": projection.calibration_id,
                    "description_id": brain.description_id,
                    "model_contract_id": brain.model_contract_id,
                    "adapter_id": brain.adapter_id,
                    "world_frame_id": dataset.world_frame_id,
                    "epoch": int(epoch_value),
                    "next_batch_index": int(next_batch_index),
                    "epoch_loss_sum": float(epoch_loss_sum),
                    "best_val": float(best_validation),
                    "no_improve": int(no_improve),
                    "train_stage": trainStage,
                    "batch_size": int(batchSize),
                    "online_learning": bool(onlineLearning),
                    "brain": ExportBrainModelState(brain),
                    "online_candidates": agent.ExportOnlineCandidateState(),
                    "opt_actor": agent.opt_actor.state_dict(),
                    "opt_critic": agent.opt_critic.state_dict(),
                    "opt_world": agent.opt_world.state_dict(),
                    "train_indices": list(train_dataset.indices),
                    "val_indices": list(validation_dataset.indices),
                    "test_indices": list(test_dataset.indices),
                    "processed_sample_count_total": int(processed_samples),
                    "rng": self.CaptureRngState(),
                    "buffers": brain.ExportBuffers(),
                    "world_memory": (
                        agent.GetRuntimeWorld().ExportMemoryPayload()),
                    "memory_durable": brain.mem.ExportDurableState(),
                }

            def SaveArtifacts(
                epoch_value: int,
                *,
                next_batch_index: int = 0,
                epoch_loss_sum: float = 0.0,
                publish: bool = False,
            ) -> None:
                checkpoint = BuildCheckpointPayload(
                    epoch_value,
                    next_batch_index,
                    epoch_loss_sum)
                if publish:
                    module_modes = [
                        (module, bool(module.training))
                        for module in brain.modules()]
                    try:
                        if onlineLearning:
                            agent.UpdateAllWrappers("commit")
                        generation = (
                            f"epoch-{int(epoch_value):08d}-"
                            f"samples-{int(processed_samples):012d}-"
                            f"{time.time_ns()}")
                        model_path = Path(output_path)
                        generation_directory = (
                            model_path.parent
                            / f".{model_path.stem}_deployments"
                            / generation)
                        generation_model = (
                            generation_directory / model_path.name)
                        generation_world = generation_directory / (
                            Path(worldMemPath).name
                            if worldMemPath
                            else "world_memory.pth")
                        generation_memory = generation_directory / (
                            Path(memMemPath).name
                            if memMemPath
                            else "memory.pth")
                        generation_directory.mkdir(
                            parents=True,
                            exist_ok=False)
                        agent.SaveWorldMemory(str(generation_world))
                        agent.SaveMemory(str(generation_memory))
                        self.SaveModuleParameters(
                            brain,
                            str(generation_model))
                        self.AtomicJsonSave({
                            "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                            "calibration_id": projection.calibration_id,
                            "description_id": brain.description_id,
                            "model_contract_id": brain.model_contract_id,
                            "adapter_id": brain.adapter_id,
                            "generation": generation,
                            "model_path": str(generation_model.resolve()),
                            "world_memory_path": str(
                                generation_world.resolve()),
                            "memory_path": str(
                                generation_memory.resolve()),
                        }, self.DeploymentManifestPath(output_path))
                    finally:
                        if onlineLearning:
                            self.ImportTrainingCheckpointState(
                                brain,
                                agent,
                                checkpoint,
                                batchSize=batchSize)
                        for module, was_training in module_modes:
                            module.training = was_training
                self.AtomicTorchSave(checkpoint, checkpoint_path)

            def SelectOptimizationLosses(
                output: Any,
            ) -> Dict[str, torch.Tensor]:
                required = {
                    "world": "world_optimization_loss",
                    "critic": "critic_optimization_loss",
                    "policy": "policy_optimization_loss",
                }
                if any(name not in output.losses for name in required.values()):
                    raise RuntimeError(
                        "BrainCore training losses are incomplete")
                return {
                    name: output.losses[source]
                    for name, source in required.items()}

            def Evaluate(loader: SequentialTrajectoryLoader) -> float:
                self.robot.Reset()
                agent.ResetBrainState(batchSize=batchSize)
                total = 0.0
                count = 0
                for batch in loader:
                    output = agent.RunStep(
                        BuildActInput(batch, True),
                        enableGrad=False,
                        modelTraining=False)
                    optimization = SelectOptimizationLosses(output)
                    selected = self.TrainStageLossNames(trainStage)
                    loss = sum(
                        (optimization[name] for name in selected),
                        batch.frames.new_zeros(()))
                    if critic_enabled:
                        loss = loss + output.losses[
                            "critic_transport_delayed_loss"]
                    total += float(loss.detach().item())
                    count += 1
                    self.CompleteContractOfflineTransition(batch)
                return total / max(1, count)

            self.controller.SetStatus(
                "training",
                "Training started",
                epoch=start_epoch,
                total_epochs=epochs,
                batch=0,
                total_batches=len(train_loader),
                visual=self.controller.EmptyVisualStatus(touch=True))

            for epoch_index in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break
                if not self.WaitWhilePaused("Training paused"):
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

                resume_current_epoch = (
                    epoch_index == start_epoch
                    and resume_batch_index > 0)
                if resume_current_epoch:
                    if resume_batch_index > len(train_loader):
                        raise ValueError(
                            "checkpoint batch cursor exceeds the train split")
                    epoch_loss = resume_epoch_loss
                    batch_count = resume_batch_index
                    iterator = train_loader.IterFrom(resume_batch_index)
                    enumeration_start = resume_batch_index + 1
                else:
                    epoch_loss = 0.0
                    batch_count = 0
                    agent.ResetBrainState(batchSize=batchSize)
                    iterator = iter(train_loader)
                    enumeration_start = 1

                for batch_index, batch in enumerate(
                    iterator,
                    start=enumeration_start,
                ):
                    if self.controller.ShouldStop():
                        break
                    if self.controller.ShouldResetHebbian():
                        agent.ResetHebbianMemory()
                        self.controller.RequestCancelResetHebbian()

                    output = agent.RunStep(BuildActInput(batch, False))
                    optimization = SelectOptimizationLosses(output)
                    delayed = output.losses[
                        "critic_transport_delayed_loss"]
                    selected = self.TrainStageLossNames(trainStage)
                    report_loss = sum(
                        (optimization[name] for name in selected),
                        batch.frames.new_zeros(()))
                    if critic_enabled:
                        report_loss = report_loss + delayed.detach()

                    agent.opt_world.zero_grad(set_to_none=True)
                    agent.opt_critic.zero_grad(set_to_none=True)
                    agent.opt_actor.zero_grad(set_to_none=True)
                    world_parameters = agent.OptimizerParameters((
                        agent.opt_world,))
                    critic_parameters = agent.OptimizerParameters((
                        agent.opt_critic,))
                    policy_parameters = agent.OptimizerParameters((
                        agent.opt_actor,))

                    if critic_enabled and delayed.requires_grad:
                        delayed.backward()
                        agent.CaptureCriticTransportGrad()
                    jobs: List[Tuple[torch.Tensor, List[nn.Parameter]]] = []
                    if trainStage in ("full", "world"):
                        jobs.append((optimization["world"], world_parameters))
                    if critic_enabled:
                        jobs.append((optimization["critic"], critic_parameters))
                    if trainStage in ("full", "policy"):
                        jobs.append((optimization["policy"], policy_parameters))
                    jobs = [
                        (objective, parameters)
                        for objective, parameters in jobs
                        if objective.requires_grad and parameters]
                    for job_index, (objective, parameters) in enumerate(jobs):
                        objective.backward(
                            inputs=parameters,
                            retain_graph=(job_index + 1 < len(jobs)))
                    if critic_enabled:
                        agent.CaptureCriticTransportGrad()
                        agent.ApplyCriticTransportManualGrad()
                    else:
                        agent.ClearCriticTransportGradAccumulator()

                    if trainStage == "world":
                        optimizers = (agent.opt_world,)
                    elif trainStage == "policy":
                        optimizers = (agent.opt_critic, agent.opt_actor)
                    else:
                        optimizers = (
                            agent.opt_world,
                            agent.opt_critic,
                            agent.opt_actor)
                    torch.nn.utils.clip_grad_norm_(
                        agent.OptimizerParameters(optimizers),
                        1.0)
                    if trainStage in ("full", "world"):
                        agent.opt_world.step()
                    if trainStage in ("full", "policy"):
                        agent.opt_critic.step()
                        agent.opt_actor.step()
                    if critic_enabled:
                        agent.AfterOptimizerStep()
                    if onlineLearning:
                        wrappers = tuple(
                            self.TrainStageOnlineWrappers(
                                brain,
                                trainStage))
                        agent.UpdateWrappers(
                            wrappers,
                            "accumulategrads")
                        agent.UpdateWrappers(wrappers, "autogrow")

                    previous_samples = processed_samples
                    processed_samples += int(batch.frames.size(0))
                    epoch_loss += float(report_loss.detach().item())
                    batch_count += 1
                    self.CompleteContractOfflineTransition(batch)
                    if self.ShouldTriggerPeriodicSave(
                        previous_samples,
                        processed_samples,
                        saveEverySampleCount,
                    ):
                        SaveArtifacts(
                            epoch_index,
                            next_batch_index=batch_index,
                            epoch_loss_sum=epoch_loss)
                    self.controller.SetStatus(
                        "training",
                        "Training",
                        epoch=epoch_index + 1,
                        total_epochs=epochs,
                        batch=batch_index,
                        total_batches=len(train_loader),
                        train_loss=float(report_loss.detach().item()))
                    if not self.WaitWhilePaused("Training paused"):
                        break

                resume_batch_index = 0
                resume_epoch_loss = 0.0
                average_train = epoch_loss / max(1, batch_count)
                SaveArtifacts(
                    epoch_index,
                    next_batch_index=batch_count,
                    epoch_loss_sum=epoch_loss)
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

                try:
                    average_validation, test_loss = (
                        self.EvaluateValidationAndTestWithRestoredBrainBuffers(
                            brain,
                            lambda: Evaluate(validation_loader),
                            lambda: Evaluate(test_loader)))
                finally:
                    self.robot.Reset()
                improved = (
                    best_validation - average_validation > min_delta)
                if improved:
                    best_validation = average_validation
                    no_improve = 0
                else:
                    no_improve += 1
                SaveArtifacts(epoch_index + 1, publish=improved)
                self.controller.SetStatus(
                    "training",
                    (
                        f"Epoch {epoch_index + 1}/{epochs} "
                        f"train {average_train:.4f} "
                        f"validation {average_validation:.4f} "
                        f"test {test_loss:.4f}"),
                    epoch=epoch_index + 1,
                    total_epochs=epochs,
                    train_loss=average_train,
                    val_loss=average_validation)
                if no_improve >= patience:
                    self.controller.SetStatus(
                        "completed",
                        "Validation stabilized")
                    break
            else:
                self.controller.SetStatus("completed", "Training completed")
        except Exception as exception:
            self.controller.SetStatus(
                "error",
                f"Training error: {exception}",
                trace=traceback.format_exc())
        finally:
            self.is_begin = False

    def ExportParamsFromCheckpoint(
        self,
        overwrite: bool = False,
        *,
        ckptPath: str = BasicParameters.CKPT_PATH_TRAIN,
        outPath: str = BasicParameters.MODULEPARAMETER_PATH,) -> None:

        ckpt_path = Path(ckptPath)
        out_path = Path(outPath)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        raw = torch.load(str(ckpt_path), map_location="cpu")
        if type(raw) is not dict or set(raw) != TRAIN_CHECKPOINT_FIELDS:
            raise ValueError(
                f"checkpoint {ckpt_path} fields do not match the current schema")

        if (
            type(raw["schema_version"]) is not int
            or raw["schema_version"] != TRAIN_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported training checkpoint schema {raw['schema_version']!r}")
        if type(raw["brain"]) is not dict:
            raise TypeError("training checkpoint brain state must be a dictionary")
        if raw["online_learning"]:
            raise ValueError(
                "online training checkpoints require the published deployment artifact; "
                "their candidate adapters cannot be exported without materialization")
        params = {
            "schema_version": raw["schema_version"],
            "calibration_id": raw["calibration_id"],
            "model_contract_id": raw["model_contract_id"],
            "brain": raw["brain"],}
        runtime_keys = sorted(
            name for name in params["brain"]
            if IsWorldRuntimeStateKey(name))
        if runtime_keys:
            raise ValueError(
                f"training checkpoint brain contains runtime world state: {runtime_keys}")

        if out_path.exists() and not overwrite:
            stem = out_path.stem
            suffix = out_path.suffix
            parent = out_path.parent

            idx = 1
            while True:
                cand = parent / f"{stem}_{idx}{suffix}"
                if not cand.exists():
                    out_path = cand
                    print(f"[ExportParamsOnly] target exists, save to {out_path} instead")
                    break
                idx += 1

        self.AtomicTorchSave(params, out_path)
        print(f"[ExportParamsOnly] saved params to {out_path}")


    def LoadCheckpoint(
        self,
        brain: BrainCore,
        agent: Agent,
        dataset: Dataset,
        path: str = None,
        *,
        batchSize: int,
        trainStage: str,
        onlineLearning: bool,) -> TrainingResumeState:
        ckpt = torch.load(
            path,
            map_location=self.device,
            weights_only=False)
        if type(ckpt) is not dict or set(ckpt) != TRAIN_CHECKPOINT_FIELDS:
            raise ValueError(
                "training checkpoint fields do not match the current schema")
        self.ValidateTrainingRngState(ckpt["rng"])
        if (
            type(ckpt["schema_version"]) is not int
            or ckpt["schema_version"] != TRAIN_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported training checkpoint schema {ckpt['schema_version']!r}")
        if ckpt["calibration_id"] != brain.calibration_id:
            raise ValueError(
                "training checkpoint calibration_id does not match configured K")
        if ckpt["description_id"] != brain.description_id:
            raise ValueError(
                "training checkpoint description_id does not match the robot")
        if ckpt["model_contract_id"] != brain.model_contract_id:
            raise ValueError(
                "training checkpoint model_contract_id does not match the foundation")
        if ckpt["adapter_id"] != brain.adapter_id:
            raise ValueError(
                "training checkpoint adapter_id does not match the robot")
        if ckpt["world_frame_id"] != dataset.world_frame_id:
            raise ValueError(
                "training checkpoint world_frame_id does not match the dataset")
        if type(ckpt["batch_size"]) is not int or ckpt["batch_size"] != batchSize:
            raise ValueError(
                "training checkpoint batch_size does not match the trajectory loader")
        if type(ckpt["online_learning"]) is not bool or ckpt["online_learning"] != onlineLearning:
            raise ValueError(
                "training checkpoint online_learning mode does not match the brain")

        for field in ("epoch", "next_batch_index"):
            if type(ckpt[field]) is not int:
                raise TypeError(f"training checkpoint {field} must be an integer")
        if type(ckpt["epoch_loss_sum"]) not in (int, float):
            raise TypeError("training checkpoint epoch_loss_sum must be numeric")
        start_epoch = ckpt["epoch"]
        next_batch_index = ckpt["next_batch_index"]
        epoch_loss_sum = float(ckpt["epoch_loss_sum"])
        if (
            start_epoch < 0
            or next_batch_index < 0
            or not math.isfinite(epoch_loss_sum)
        ):
            raise ValueError("training checkpoint cursor is invalid")
        checkpoint_stage = self.NormalizeTrainStage(ckpt["train_stage"])
        requested_stage = self.NormalizeTrainStage(trainStage)
        if checkpoint_stage != requested_stage:
            raise ValueError(
                "training checkpoint train_stage does not match the requested stage")
        split_indices: List[int] = []
        for field in ("train_indices", "val_indices", "test_indices"):
            indices = ckpt[field]
            if type(indices) is not list or any(type(index) is not int for index in indices):
                raise TypeError(f"training checkpoint {field} must be a list of integers")
            if len(set(indices)) != len(indices):
                raise ValueError(f"training checkpoint {field} contains duplicate indices")
            if any(index < 0 or index >= len(dataset) for index in indices):
                raise ValueError(f"training checkpoint {field} contains an invalid index")
            split_indices.extend(indices)
        if len(set(split_indices)) != len(split_indices):
            raise ValueError("training checkpoint dataset splits must be disjoint")
        if set(split_indices) != set(range(len(dataset))):
            raise ValueError("training checkpoint dataset splits must cover the dataset")
        maximum_batch_cursor = len(ckpt["train_indices"]) // batchSize
        if next_batch_index > maximum_batch_cursor:
            raise ValueError("training checkpoint cursor exceeds the train split")
        if type(ckpt["no_improve"]) is not int or ckpt["no_improve"] < 0:
            raise ValueError("training checkpoint no_improve is invalid")
        if type(ckpt["processed_sample_count_total"]) is not int or ckpt["processed_sample_count_total"] < 0:
            raise ValueError("training checkpoint processed sample count is invalid")
        if type(ckpt["best_val"]) not in (int, float):
            raise TypeError("training checkpoint best_val must be numeric")
        best_val = float(ckpt["best_val"])
        if math.isnan(best_val) or best_val == float("-inf"):
            raise ValueError("training checkpoint best_val is invalid")
        no_improve = ckpt["no_improve"]
        processed_sample_count_total = ckpt["processed_sample_count_total"]

        brain_state = ckpt["brain"]
        if type(brain_state) is not dict:
            raise TypeError("brain model state must be a dictionary")
        agent.ValidateOnlineCandidateState(ckpt["online_candidates"])
        world = agent.GetRuntimeWorld()
        world_batch_size, _ = world.ValidateMemoryPayload(ckpt["world_memory"])
        if world_batch_size != batchSize:
            raise ValueError("training checkpoint World memory batch size is invalid")
        brain.mem.ValidateDurableState(
            ckpt["memory_durable"],
            expectedBatch=batchSize)
        buffer_state = ckpt["buffers"]
        if type(buffer_state) is not dict or set(buffer_state) != BRAIN_RUNTIME_BUFFER_FIELDS:
            raise ValueError(
                "training checkpoint brain runtime buffer fields are invalid")
        if (
            type(buffer_state["schema_version"]) is not int
            or buffer_state["schema_version"] != BRAIN_RUNTIME_SCHEMA_VERSION
        ):
            raise ValueError(
                "training checkpoint brain runtime schema is invalid")
        brain.ValidateBufferState(buffer_state)
        self.ImportTrainingCheckpointState(
            brain,
            agent,
            ckpt,
            batchSize=batchSize)

        train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
        val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
        test_ds = torch.utils.data.Subset(dataset, ckpt["test_indices"])

        return TrainingResumeState(
            epoch=start_epoch,
            next_batch_index=next_batch_index,
            epoch_loss_sum=epoch_loss_sum,
            best_val=best_val,
            no_improve=no_improve,
            processed_sample_count_total=processed_sample_count_total,
            train_dataset=train_ds,
            validation_dataset=val_ds,
            test_dataset=test_ds)

    def ImportTrainingCheckpointState(
        self,
        brain: BrainCore,
        agent: Agent,
        checkpoint: Dict[str, Any],
        *,
        batchSize: int,) -> None:
        LoadBrainModelState(brain, checkpoint["brain"])
        agent.ImportOnlineCandidateState(checkpoint["online_candidates"])
        agent.SyncTrainableOptimizers()
        agent.opt_actor.load_state_dict(checkpoint["opt_actor"])
        agent.opt_critic.load_state_dict(checkpoint["opt_critic"])
        agent.opt_world.load_state_dict(checkpoint["opt_world"])
        agent.GetRuntimeWorld().ImportMemoryPayload(
            checkpoint["world_memory"],
            batchSize=batchSize)
        brain.mem.ImportDurableState(checkpoint["memory_durable"])
        brain.ImportBuffers(checkpoint["buffers"])
        self.robot.Reset()
        self.RestoreRngState(checkpoint["rng"])

    def BuildContractTestFeedback(
        self,
        batchSize: int = 1,
    ) -> BrainFeedbackPacket:
        if type(batchSize) is not int or batchSize < 1:
            raise ValueError("batchSize must be a positive integer")
        self.robot.Reset()
        payloads = tuple(
            self.BuildNeutralRobotFeedbackPayload(
                timestamp=float(index + 1))
            for index in range(batchSize))
        feedback = self.EncodeBrainFeedback(
            payloads,
            batchSize=batchSize)
        return feedback

    def RunContractWorldSmoke(self) -> bool:
        rng_state = self.CaptureRngState()
        try:
            contract_view = self.brain_build_spec.contract_view
            cognitive_dim = 16
            feedback = self.BuildContractTestFeedback(batchSize=2)
            adapter = ContractWorldEmbodimentAdapter(
                contract_view,
                cognitive_dim,
                cognitive_dim).to(self.device)
            adapter.train()
            transition = adapter.EncodeTransition(feedback)
            prior_world_state = torch.zeros(
                2,
                cognitive_dim,
                device=self.device,
                dtype=feedback.joint_features.dtype,
                requires_grad=True)
            prediction = adapter.PredictFeedback(prior_world_state)
            losses = adapter.ComputeFeedbackLoss(prediction, feedback)
            loss = losses["loss"]
            if (
                tuple(transition["EncodedTransition"].shape)
                != (2, cognitive_dim)
                or not bool(torch.isfinite(
                    transition["EncodedTransition"]).all().item())
                or loss.dim() != 0
                or not bool(torch.isfinite(loss).item())
            ):
                raise RuntimeError("world contract smoke produced invalid tensors")
            for name in ("LatentRisk", "LatentFeasibility"):
                value = prediction[name]
                if bool(((value < 0.0) | (value > 1.0)).any().item()):
                    raise RuntimeError(
                        "world contract prediction left its normalized domain")
            loss.backward()
            predictor_gradients = [
                parameter.grad
                for parameter in adapter.FeedbackPredictor.parameters()
                if parameter.requires_grad
                and parameter.grad is not None]
            if (
                prior_world_state.grad is None
                or not bool(torch.isfinite(
                    prior_world_state.grad).all().item())
                or not predictor_gradients
                or any(
                    not bool(torch.isfinite(gradient).all().item())
                    for gradient in predictor_gradients)
            ):
                raise RuntimeError("world contract smoke gradients are invalid")
            return True
        finally:
            self.RestoreRngState(rng_state)

    def ValidateContractTrainingTestData(
        self,
        dataRoot: Union[str, Path],
        *,
        batchSize: int,
        valSplit: float,
    ) -> int:
        if type(batchSize) is not int or batchSize < 1:
            raise ValueError("batchSize must be a positive integer")
        try:
            validation_split = float(valSplit)
        except (TypeError, ValueError) as exception:
            raise ValueError("valSplit must be finite") from exception
        if (
            not math.isfinite(validation_split)
            or not 0.0 < validation_split < 0.9
        ):
            raise ValueError("valSplit must leave non-empty train and test splits")
        root = Path(dataRoot).resolve()
        production_root = Path(BasicParameters.DATA_ROOT_PATH).resolve()
        if root == production_root:
            raise ValueError("training smoke cannot read the production data root")
        if not self.HasGameDataset(root):
            raise RuntimeError(
                "contract test dataset is missing or does not match the active contract")
        sample_count = len(tuple((root / "frames").glob("*.png")))
        test_count = int(sample_count * 0.1)
        validation_count = int(sample_count * validation_split)
        train_count = sample_count - validation_count - test_count
        split_counts = (train_count, validation_count, test_count)
        if any(count < batchSize for count in split_counts):
            raise RuntimeError(
                "contract test dataset cannot form all recurrent splits")
        if any(count % batchSize != 0 for count in split_counts):
            raise RuntimeError(
                "contract test split sizes must be divisible by batchSize")
        return sample_count

    @staticmethod
    def ContractTrainingTestSampleCount(
        *,
        batchSize: int,
        valSplit: float,
    ) -> int:
        if type(batchSize) is not int or batchSize < 1:
            raise ValueError("batchSize must be a positive integer")
        try:
            validation_split = float(valSplit)
        except (TypeError, ValueError) as exception:
            raise ValueError("valSplit must be finite") from exception
        if (
            not math.isfinite(validation_split)
            or not 0.0 < validation_split < 0.9
        ):
            raise ValueError("valSplit must leave non-empty train and test splits")
        for multiplier in range(3, 10001):
            sample_count = batchSize * multiplier
            test_count = int(sample_count * 0.1)
            validation_count = int(sample_count * validation_split)
            train_count = sample_count - validation_count - test_count
            split_counts = (train_count, validation_count, test_count)
            if (
                all(count >= batchSize for count in split_counts)
                and all(count % batchSize == 0 for count in split_counts)
            ):
                return sample_count
        raise RuntimeError("cannot construct divisible contract test splits")

    @staticmethod
    def ContractTrainingTestConfiguration(
        dataRoot: Union[str, Path],
    ) -> Dict[str, str]:
        root = Path(dataRoot).resolve()
        return {
            "DATA_ROOT_PATH_TEST": str(root),
            "DATA_SENSOR_MANIFEST_PATH_TEST": str(
                root / "sensor_manifest.json"),
            "DATA_DEPTH_PATH_TEST": str(root / "depth"),
            "DATA_DEPTH_VALID_PATH_TEST": str(root / "depth_valid"),
            "DATA_FEEDBACK_PATH_TEST": str(root / "feedback"),
            "DATA_NORMAL_PATH_TEST": str(root / "normal"),
            "DATA_SEMANTIC_SEGMENTATION_PATH_TEST": str(
                root / "semantic_segmentation"),
            "DATA_INSTANCE_SEGMENTATION_PATH_TEST": str(
                root / "instance_segmentation"),
            "DATA_SYNTHETIC_SUPERVISION_PATH_TEST": str(
                root / "synthetic_supervision"),
        }

    def BuildContractTrainingTestData(
        self,
        dataRoot: Union[str, Path],
        *,
        sampleCount: int,
        seed: int,
    ) -> int:
        if type(sampleCount) is not int or sampleCount < 1:
            raise ValueError("sampleCount must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if iio is None:
            raise RuntimeError("imageio.v3 is required for contract test data")
        root = Path(dataRoot).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError("contract test data root must be empty")
        directories = tuple(
            root / name
            for name in (
                "frames",
                "reward",
                "done",
                "depth",
                "depth_valid",
                "feedback",
                "texts",
                "normal",
                "semantic_segmentation",
                "instance_segmentation",
                "synthetic_supervision",
            ))
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        projection = self.robot.ContractView.perception_projection
        if projection is None:
            raise RuntimeError("robot contract has no perception calibration")
        manifest = {
            "calibration_id": projection.calibration_id,
            "rgb_encoding": "rgb8",
            "depth_unit": "meter",
            "depth_representation": "optical_axis_z",
            "rgb_depth_alignment": "registered_to_rgb",
            "rectification": "rectified",
            "synchronization": "synchronized_exposure",
            "object_motion_frame": projection.reference_frame_id,
        }
        (root / "sensor_manifest.json").write_text(
            json.dumps(manifest, allow_nan=False, sort_keys=True),
            encoding="utf-8")
        rng = np.random.default_rng(seed)
        image_size = int(BasicParameters.IMAGE_SIZE)
        depth = np.ones(
            (image_size, image_size),
            dtype=np.float32)
        depth_valid = np.ones_like(depth, dtype=np.bool_)
        for index in range(sampleCount):
            frame_id = f"{index:05d}"
            image = rng.integers(
                0,
                256,
                size=(image_size, image_size, 3),
                dtype=np.uint8)
            iio.imwrite(root / "frames" / f"{frame_id}.png", image)
            np.save(
                root / "reward" / f"{frame_id}.npy",
                np.zeros((1,), dtype=np.float32))
            np.save(
                root / "done" / f"{frame_id}.npy",
                np.zeros((1,), dtype=np.float32))
            np.save(root / "depth" / f"{frame_id}.npy", depth)
            np.save(
                root / "depth_valid" / f"{frame_id}.npy",
                depth_valid)
            payload = self.BuildNeutralRobotFeedbackPayload(
                timestamp=float(index + 1))
            (root / "feedback" / f"{frame_id}.json").write_text(
                json.dumps(payload, allow_nan=False, sort_keys=True),
                encoding="utf-8")
        if not self.HasGameDataset(root):
            raise RuntimeError("generated contract test data is invalid")
        return sampleCount

    def TestPerceptionModule(self):
        return self.RunNamedTest("perception")

    def TestAttentionModule(self):
        return self.RunNamedTest("attention")

    def TestMemoryModule(self):
        return self.RunNamedTest("memory")

    def TestDecisionModule(self):
        return self.RunNamedTest("decision")

    def TestWorldModule(self):
        if self.is_begin:
            self.controller.SetStatus(
                "recur",
                "Training or Deploy is already running")
            return False
        try:
            result = bool(self.RunContractWorldSmoke())
            print(
                "World contract smoke "
                f"{'passed' if result else 'failed'}")
            return result
        except Exception as exception:
            self.controller.SetStatus(
                "error",
                f"World contract smoke error: {exception}",
                trace=traceback.format_exc())
            print(f"World contract smoke error: {exception}")
            return False

    def TestValueEstimationModule(self):
        return self.RunNamedTest("value")
    
    def TestConsciousnessModule(self):
        return self.RunNamedTest("consciousness")
    
    def TestIntentionModule(self):
        return self.RunNamedTest("intention")

    def TestAGICoreModule(self):
        return self.RunNamedTest("AGICore")

    def TestManagerModule(self):
        return self.RunNamedTest("manager")
    
    def TestOCRModule(self):
        return self.RunNamedTest("OCR")
    

    def MonitorTraining(self):
        self.MonitorStatus(
            prefix="TRAIN",
            monitorName="MonitorTraining",
            renderStatus=lambda st: (
                f"{st['state']} | epoch {st['epoch']}/{st['total_epochs']} "
                f"| batch {st['batch']}/{st['total_batches']} "
                f"| train_loss={st['train_loss']:.4f} | msg={st['message']}"),
            terminalStates=("completed", "stopped", "error"),
            sleepSeconds=1.0,)


    def MonitorDeployment(self):
        self.MonitorStatus(
            prefix="DEPLOY",
            monitorName="MonitorDeployment",
            renderStatus=lambda st: f"{st['state']} | msg={st['message']}",
            terminalStates=("stopped", "error"),
            sleepSeconds=0.5,)


    def TestModuleTrain(
        self,
        onlineLearning: bool,
        *,
        dataRoot: Optional[str] = None,
        nSamples: Optional[int] = None,
        epochs: int = 1,
        batchSize: int = 1,
        valSplit: float = 0.2,
        seed: int = 42,
        trainStage: str = "full",
    ) -> bool:
        rng_state = None
        configuration_state = None
        owns_begin_state = False
        try:
            if type(onlineLearning) is bool:
                online_learning = onlineLearning
            elif type(onlineLearning) is int and onlineLearning in (0, 1):
                online_learning = bool(onlineLearning)
            else:
                raise TypeError("onlineLearning must be a boolean")
            if self.is_begin:
                self.controller.SetStatus(
                    "recur",
                    "Training or Deploy is already running")
                return False
            if type(epochs) is not int or epochs < 1:
                raise ValueError("epochs must be a positive integer")
            if type(seed) is not int or seed < 0:
                raise ValueError("seed must be a non-negative integer")
            if nSamples is not None and (
                type(nSamples) is not int or nSamples < 1
            ):
                raise ValueError("nSamples must be a positive integer")
            train_stage = self.NormalizeTrainStage(trainStage)
            with CONTRACT_TRAINING_TEST_LOCK:
                if self.is_begin:
                    self.controller.SetStatus(
                        "recur",
                        "Training or Deploy is already running")
                    return False
                self.robot.Reset()
                rng_state = self.CaptureRngState()
                with tempfile.TemporaryDirectory(
                    prefix="brain-contract-training-",
                ) as directory:
                    workspace = Path(directory)
                    requested_root = Path(
                        BasicParameters.DATA_ROOT_PATH_TEST
                        if dataRoot is None
                        else dataRoot).resolve()
                    production_root = Path(
                        BasicParameters.DATA_ROOT_PATH).resolve()
                    existing_valid = (
                        requested_root != production_root
                        and self.HasGameDataset(requested_root))
                    if existing_valid:
                        dataset_root = requested_root
                        sample_count = self.ValidateContractTrainingTestData(
                            dataset_root,
                            batchSize=batchSize,
                            valSplit=valSplit)
                    elif dataRoot is not None:
                        raise RuntimeError(
                            "explicit contract test dataset is invalid")
                    else:
                        sample_count = (
                            self.ContractTrainingTestSampleCount(
                                batchSize=batchSize,
                                valSplit=valSplit)
                            if nSamples is None
                            else nSamples)
                        dataset_root = workspace / "dataset"
                        self.BuildContractTrainingTestData(
                            dataset_root,
                            sampleCount=sample_count,
                            seed=seed)
                        sample_count = self.ValidateContractTrainingTestData(
                            dataset_root,
                            batchSize=batchSize,
                            valSplit=valSplit)
                    configuration_state = {
                        name: getattr(BasicParameters, name)
                        for name in CONTRACT_TRAINING_TEST_CONFIG_FIELDS}
                    configuration = self.ContractTrainingTestConfiguration(
                        dataset_root)
                    try:
                        for name, value in configuration.items():
                            setattr(BasicParameters, name, value)
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        artifact_root = workspace / "artifacts"
                        artifact_root.mkdir()
                        self.is_begin = True
                        owns_begin_state = True
                        self.TrainLoop(
                            epochs,
                            batchSize,
                            float(valSplit),
                            False,
                            online_learning,
                            saveEverySampleCount=sample_count + 1,
                            isTest=True,
                            worldMemPath=str(
                                artifact_root / "world_memory.pt"),
                            memMemPath=str(
                                artifact_root / "memory.pt"),
                            ckptPath=str(
                                artifact_root / "checkpoint.pth"),
                            outPath=str(artifact_root / "model.pth"),
                            overrideCheckpointWithModuleParams=False,
                            trainStage=train_stage)
                    finally:
                        for name, value in configuration_state.items():
                            setattr(BasicParameters, name, value)
                        configuration_state = None
                    result = (
                        self.controller.GetStatus()["state"]
                        == "completed")
            print(
                "Contract training smoke "
                f"{'passed' if result else 'failed'}")
            return bool(result)
        except Exception as exception:
            self.controller.SetStatus(
                "error",
                f"Contract training smoke error: {exception}",
                trace=traceback.format_exc())
            print(f"Contract training smoke error: {exception}")
            return False
        finally:
            try:
                if configuration_state is not None:
                    for name, value in configuration_state.items():
                        setattr(BasicParameters, name, value)
            finally:
                try:
                    self.robot.Reset()
                finally:
                    if rng_state is not None:
                        self.RestoreRngState(rng_state)
                    if owns_begin_state:
                        self.is_begin = False


    def TestOCRModuleTrain(
        self,
        *,
        dataRoot: str = BasicParameters.OCR_DATA_ROOT_PATH_TEST,
        nSamples: int = 32,
        epochs: int = 1,
        batchSize: int = 4,
        valSplit: float = 0.2,
        seed: int = 42,) -> Dict[str, Any]:
        if self.is_begin:
            return {"ok": False, "msg": "StartOCRTraining returns False (training may already be running)"}

        try:
            root = Path(dataRoot)
            BasicParameters.Set("OCR_DATA_ROOT_PATH_TEST", str(root))
            frames_dir = root / "frames"
            ocr_texts_dir = root / "OCRTexts"

            if not self.HasOcrDataset(root):
                if iio is None:
                    raise RuntimeError("imageio.v3 error")

                rng = np.random.default_rng(seed)
                if root.exists():
                    shutil.rmtree(root)
                frames_dir.mkdir(parents=True, exist_ok=True)
                ocr_texts_dir.mkdir(parents=True, exist_ok=True)

                vocab_path = Path(BasicParameters.OCR_DICT_PATH)
                if not vocab_path.exists():
                    raise FileNotFoundError(f"OCR vocab file not found: {vocab_path}")

                vocab_chars = [
                    line.strip()
                    for line in vocab_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
                if len(vocab_chars) == 0:
                    raise RuntimeError(f"OCR vocab file is empty: {vocab_path}")
                safe_vocab_chars = [
                    ch for ch in vocab_chars
                    if ch not in {"\"", ",", "\n", "\r", "#"}]
                if len(safe_vocab_chars) == 0:
                    safe_vocab_chars = list("ABC123")

                h_img = BasicParameters.IMAGE_SIZE
                w_img = BasicParameters.IMAGE_SIZE

                for i in range(nSamples):
                    img = rng.integers(0, 32, size=(h_img, w_img, 3), dtype=np.uint8)
                    n_lines = int(rng.integers(1, 5))
                    anno_lines: List[str] = []

                    top_margin = 24
                    bottom_margin = 24
                    usable_h = max(80, h_img - top_margin - bottom_margin)
                    lane_h = max(28, usable_h // max(1, n_lines))

                    for li in range(n_lines):
                        line_h = int(rng.integers(24, min(56, lane_h + 1)))
                        y1_base = top_margin + li * lane_h
                        y1_jitter = int(rng.integers(0, max(1, lane_h - line_h + 1)))
                        y1 = min(h_img - line_h - 1, y1_base + y1_jitter)
                        y2 = min(h_img, y1 + line_h)

                        x1 = int(rng.integers(16, 64))
                        line_w = int(rng.integers(96, min(360, w_img - x1)))
                        x2 = min(w_img, x1 + line_w)

                        bg_val = int(rng.integers(120, 220))
                        img[y1:y2, x1:x2, :] = bg_val

                        text_len = int(rng.integers(2, 9))
                        text = "".join(rng.choice(safe_vocab_chars, size=text_len, replace=True).tolist())
                        ignore_flag = 1 if rng.random() < 0.25 else 0
                        if ignore_flag:
                            chars = list(text)
                            n_mask = max(1, min(len(chars), int(rng.integers(1, len(chars) + 1))))
                            for pos in rng.choice(len(chars), size=n_mask, replace=False).tolist():
                                chars[pos] = "#"
                            text = "".join(chars)
                        anno_lines.append(
                            f'{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{ignore_flag},"{text}"')

                        inner_y1 = min(y2 - 1, y1 + 4)
                        inner_y2 = max(inner_y1 + 1, y2 - 4)
                        cursor_x = x1 + 6
                        char_w = max(6, (max(1, x2 - x1 - 12)) // max(1, text_len))
                        visible_text = text.replace("#", safe_vocab_chars[0])
                        for ch in visible_text:
                            stripe_w = max(2, char_w // 3)
                            glyph_h = max(4, inner_y2 - inner_y1)
                            tone = int((hash(ch) % 120) + 40)
                            gx1 = min(x2 - 1, cursor_x + int(rng.integers(0, max(1, char_w - stripe_w + 1))))
                            gx2 = min(x2, gx1 + stripe_w)
                            gy1 = inner_y1
                            gy2 = min(inner_y2, gy1 + glyph_h)
                            img[gy1:gy2, gx1:gx2, :] = tone
                            cursor_x += char_w
                            if cursor_x >= x2 - 4:
                                break

                    iio.imwrite(str(frames_dir / f"{i:05d}.png"), img)
                    ocr_texts_dir.joinpath(f"{i:05d}.txt").write_text("\n".join(anno_lines), encoding="utf-8")

                print(f"[TestOCR] created random OCR dataset at: {root}")
            else:
                print(f"[TestOCR] use existing OCR dataset at: {root}")

            ok = self.StartOCRTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=False,
                isTest=True,
                trainDetection=True,
                trainRecognition=True,)

            if not ok:
                print("StartOCRTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

            return {"ok": True}

        except Exception as e:
            print(f"TestOCRModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def TestOCRRecognitionTrain(
        self,
        *,
        dataRoot: str = BasicParameters.OCR_RECOGNIZER_DATA_ROOT_PATH_TEST,
        nSamples: int = 64,
        epochs: int = 1,
        batchSize: int = 8,
        valSplit: float = 0.2,
        seed: int = 42,) -> Dict[str, Any]:
        if self.is_begin:
            return {"ok": False, "msg": "StartOCRRecognitionTraining returns False (training may already be running)"}

        try:
            root = Path(dataRoot)
            BasicParameters.Set("OCR_RECOGNIZER_DATA_ROOT_PATH_TEST", str(root))
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

            if not self.HasOcrRecognitionDataset(root):
                if iio is None:
                    raise RuntimeError("imageio.v3 error")

                rng = np.random.default_rng(seed)
                if root.exists():
                    shutil.rmtree(root)
                frames_dir.mkdir(parents=True, exist_ok=True)
                texts_dir.mkdir(parents=True, exist_ok=True)

                vocab_path = Path(BasicParameters.OCR_DICT_PATH)
                if not vocab_path.exists():
                    raise FileNotFoundError(f"OCR vocab file not found: {vocab_path}")

                vocab_chars = [
                    line.strip()
                    for line in vocab_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
                if len(vocab_chars) == 0:
                    raise RuntimeError(f"OCR vocab file is empty: {vocab_path}")

                safe_vocab_chars = [
                    ch for ch in vocab_chars
                    if ch not in {"\"", ",", "\n", "\r", "#"}]
                if len(safe_vocab_chars) == 0:
                    safe_vocab_chars = list("ABC123")

                target_h = 32

                for i in range(nSamples):
                    text_len = int(rng.integers(2, 10))
                    text = "".join(rng.choice(safe_vocab_chars, size=text_len, replace=True).tolist())
                    ignore_flag = 1 if rng.random() < 0.2 else 0
                    stored_text = text
                    if ignore_flag:
                        chars = list(stored_text)
                        n_mask = max(1, min(len(chars), int(rng.integers(1, len(chars) + 1))))
                        for pos in rng.choice(len(chars), size=n_mask, replace=False).tolist():
                            chars[pos] = "#"
                        stored_text = "".join(chars)

                    img_w = int(rng.integers(72, 220))
                    img = np.full((target_h, img_w, 3), int(rng.integers(185, 235)), dtype=np.uint8)

                    cursor_x = 4
                    char_w = max(6, (img_w - 8) // max(1, text_len))
                    visible_text = stored_text.replace("#", safe_vocab_chars[0])
                    for ch in visible_text:
                        stripe_w = max(2, char_w // 3)
                        tone = int((hash(ch) % 120) + 40)
                        gx1 = min(img_w - 1, cursor_x + int(rng.integers(0, max(1, char_w - stripe_w + 1))))
                        gx2 = min(img_w, gx1 + stripe_w)
                        gy1 = int(rng.integers(4, 10))
                        gy2 = min(target_h, gy1 + int(rng.integers(16, 24)))
                        img[gy1:gy2, gx1:gx2, :] = tone
                        cursor_x += char_w
                        if cursor_x >= img_w - 4:
                            break

                    label_line = (
                        f'0,0,{img_w},0,{img_w},{target_h},0,{target_h},{ignore_flag},"{stored_text}"')

                    iio.imwrite(str(frames_dir / f"{i:05d}.png"), img)
                    texts_dir.joinpath(f"{i:05d}.txt").write_text(label_line, encoding="utf-8")

                print(f"[TestOCRRec] created random OCR recognition dataset at: {root}")
            else:
                print(f"[TestOCRRec] use existing OCR recognition dataset at: {root}")

            ok = self.StartOCRRecognitionTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=False,
                isTest=True,)

            if not ok:
                print("StartOCRRecognitionTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

            return {"ok": True}

        except Exception as e:
            print(f"TestOCRRecognitionTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def TrainOCRModule(
        self,
        epochs: int = 6,
        batchSize: int = 4,
        valSplit: float = 0.2,
        isResume: bool = False,
        saveEverySampleCount: Optional[int] = None,
        *,
        trainDetection: bool = True,
        trainRecognition: bool = True,
        overrideCheckpointWithModuleParams: Optional[bool] = None,) -> Dict[str, Any]:
        try:
            if saveEverySampleCount is None:
                saveEverySampleCount = BasicParameters.SAVE_EVERY_SAMPLE_COUNT
            if not (trainDetection or trainRecognition):
                print("[TrainOCR] trainDetection and trainRecognition cannot both be False.")
                return {"ok": False, "msg": "no_train_target"}

            frames_dir = Path(BasicParameters.OCR_FRAMES_PATH)
            ocr_texts_dir = Path(BasicParameters.OCR_TEXTS_PATH)
            txt_files = sorted(ocr_texts_dir.glob("*.txt")) if ocr_texts_dir.exists() else []

            if not self.HasOcrDataset():
                pairing_hint = ""
                if "test_images" in str(frames_dir).lower() and "train_gts" in str(ocr_texts_dir).lower():
                    pairing_hint = " It looks like OCR_FRAMES_PATH points to test_images while OCR_TEXTS_PATH points to train_gts; use matching train_images/train_gts folders for training."
                print(
                    "[TrainOCR] no valid OCR dataset found. "
                    f"frames={BasicParameters.OCR_FRAMES_PATH} ({self.SummarizeImageDirectory(frames_dir)}), "
                    f"texts={BasicParameters.OCR_TEXTS_PATH} ({len(txt_files)} txt). "
                    "Expected matching image/txt counts with parseable OCR annotations."
                    f"{pairing_hint}")
                return {"ok": False, "msg": "invalid ocr dataset"}

            for txt_path in txt_files:
                try:
                    boxes, texts, ignore_flags = DataPreprocessor.LoadOCRAnnotations(txt_path)
                except Exception as e:
                    print(f"[TrainOCR] failed to parse OCR annotation file {txt_path}: {e}")
                    return {"ok": False, "msg": "invalid ocr labels"}

                if len(texts) == 0 or len(boxes) != len(texts) or len(ignore_flags) != len(texts):
                    print(f"[TrainOCR] OCR annotation file is empty or invalid: {txt_path}")
                    return {"ok": False, "msg": "invalid ocr labels"}

            print(
                "[TrainOCR] use configured OCR folders: "
                f"{BasicParameters.OCR_FRAMES_PATH}, "
                f"{BasicParameters.OCR_TEXTS_PATH}.")

            ok = self.StartOCRTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                isTest=False,
                saveEverySampleCount=saveEverySampleCount,
                trainDetection=trainDetection,
                trainRecognition=trainRecognition,
                overrideCheckpointWithModuleParams=overrideCheckpointWithModuleParams,)

            if not ok:
                print("StartOCRTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

            return {"ok": True}

        except Exception as e:
            print(f"TrainOCRModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def TrainOCRRecognitionModule(
        self,
        epochs: int = 6,
        batchSize: int = 8,
        valSplit: float = 0.2,
        isResume: bool = False,
        overrideCheckpointWithModuleParams: Optional[bool] = None,
        saveEverySampleCount: Optional[int] = None,) -> Dict[str, Any]:
        try:
            if saveEverySampleCount is None:
                saveEverySampleCount = BasicParameters.SAVE_EVERY_SAMPLE_COUNT

            frames_dir = Path(BasicParameters.OCR_RECOGNIZER_FRAMES_PATH)
            texts_dir = Path(BasicParameters.OCR_RECOGNIZER_TEXTS_PATH)
            text_files = sorted(texts_dir.glob("*.txt")) if texts_dir.exists() else []

            if not self.HasOcrRecognitionDataset():
                print(
                    "[TrainOCRRec] no valid OCR recognition dataset found. "
                    f"frames={BasicParameters.OCR_RECOGNIZER_FRAMES_PATH} ({self.SummarizeImageDirectory(frames_dir)}), "
                    f"texts={BasicParameters.OCR_RECOGNIZER_TEXTS_PATH} ({len(text_files)} txt). "
                    "Expected matching image/txt counts.")
                return {"ok": False, "msg": "invalid ocr recognition dataset"}

            print(
                "[TrainOCRRec] use configured OCR recognition folders: "
                f"{BasicParameters.OCR_RECOGNIZER_FRAMES_PATH}, "
                f"{BasicParameters.OCR_RECOGNIZER_TEXTS_PATH}.")

            ok = self.StartOCRRecognitionTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                isTest=False,
                overrideCheckpointWithModuleParams=overrideCheckpointWithModuleParams,
                saveEverySampleCount=saveEverySampleCount,)

            if not ok:
                print("StartOCRRecognitionTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

            return {"ok": True}

        except Exception as e:
            print(f"TrainOCRRecognitionModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def TrainModule(
        self,
        onlineLearning: bool,
        epochs: int = 6,
        batchSize: int = 1,
        valSplit: float = 0.2,
        isResume: bool = False,
        overrideCheckpointWithModuleParams: Optional[bool] = None,
        saveEverySampleCount: Optional[int] = None,
        trainStage: str = "full",
    ) -> Dict[str, Any]:
        try:
            if saveEverySampleCount is None:
                saveEverySampleCount = (
                    BasicParameters.SAVE_EVERY_SAMPLE_COUNT)
            if not self.HasGameDataset():
                return {"ok": False, "msg": "no dataset"}
            started = self.StartTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                onlineLearning=onlineLearning,
                isTest=False,
                overrideCheckpointWithModuleParams=(
                    overrideCheckpointWithModuleParams),
                saveEverySampleCount=saveEverySampleCount,
                trainStage=trainStage)
            if not started:
                return {"ok": False, "msg": "already_running"}
            self.StartMessageMonitor(self.MonitorTraining)
            return {"ok": True}
        except Exception as exception:
            print(f"ModuleTrain failed with error: {exception}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def DeployModule(
        self,
        usePlanner: bool = True,) -> Dict[str, Any]:
        try:
            self.InitAgentHandle(
                usePlanner=usePlanner)
            self.controller.SetStatus(
                "is_begin",
                "Agent streaming initialized",
                visual=self.controller.EmptyVisualStatus(touch=True))
            return {"ok": True}

        except Exception as e:
            print(f"DeployModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

class TestManagerMTool:
    def TestDeploymentConfigurationRouting(self) -> bool:
        try:
            manager = object.__new__(ManagerFunction)
            captured: Dict[str, Any] = {}

            def InitAgentHandleStub(**kwargs):
                captured.update(kwargs)
                return True

            manager.InitAgentHandle = InitAgentHandleStub
            manager.controller = ModuleController()
            result = ManagerFunction.DeployModule(
                manager,
                usePlanner=False)
            ok = (
                result == {"ok": True}
                and ManagerFunction.DEFAULT_OVERRIDE_CHECKPOINT_WITH_MODULE_PARAMS is False
                and captured == {
                    "usePlanner": False})
            print(
                f"Manager deployment configuration routing "
                f"{'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager deployment configuration routing error: {e}")
            return False

    def TestPauseStopImmediateExit(self) -> bool:
        try:
            manager = ManagerFunction.__new__(ManagerFunction)
            manager.controller = ModuleController()
            manager.controller.RequestPause()
            manager.controller.RequestStop()
            ok = (not manager.controller.ShouldPause()) and (not manager.WaitWhilePaused("paused"))
            print(f"Manager pause-stop {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager pause-stop error: {e}")
            return False

    def TestBackgroundExceptionStatus(self) -> bool:
        try:
            manager = ManagerFunction.__new__(ManagerFunction)
            manager.controller = ModuleController()
            manager.is_begin = False
            manager.br_thread = None

            def boom():
                raise RuntimeError("background smoke failure")

            started = manager.StartBackgroundTask(boom)
            manager.br_thread.join(timeout=2.0)
            status = manager.controller.GetStatus()
            ok = started and status["state"] == "error" and "background smoke failure" in status["message"]
            print(f"Manager background exception status {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager background exception status error: {e}")
            return False

    def TestEvaluationRuntimeRestoredBeforeSave(self) -> bool:
        caller_rng = None
        try:
            class FakeMemory:
                def __init__(self):
                    self.durable = torch.tensor([5.0])

                def ExportDurableState(self):
                    return {"durable": self.durable.detach().clone()}

                def ImportDurableState(self, state):
                    self.durable = state["durable"].detach().clone()

            class FakeWorld:
                def __init__(self):
                    self.physical = torch.tensor([7.0])

                def ExportPhysicalState(self):
                    return {"physical": self.physical.detach().clone()}

                def ImportPhysicalState(self, state):
                    self.physical = state["physical"].detach().clone()

            class FakeBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mem = FakeMemory()
                    self.world = FakeWorld()
                    self.runtime = torch.tensor([3.0])
                    self.training_graph = {"pending": object()}
                    self.child = nn.Linear(1, 1)
                    self.train()
                    self.child.eval()

                def RuntimeModule(self, module):
                    return module

                def ExportBuffers(self):
                    return {"runtime": self.runtime.detach().clone()}

                def ImportBuffers(self, state):
                    self.runtime = state["runtime"].detach().clone()

                def SuspendTransientTrainingGraph(self):
                    state = self.training_graph
                    self.training_graph = {}
                    return state

                def RestoreTransientTrainingGraph(self, state):
                    self.training_graph = state

            manager = ManagerFunction.__new__(ManagerFunction)
            brain = FakeBrain()
            split_entries: List[float] = []
            split_random: List[Tuple[float, float, float]] = []

            caller_rng = manager.CaptureRngState()
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            evaluation_start_rng = manager.CaptureRngState()

            def DrawRandom() -> Tuple[float, float, float]:
                return (
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(()).item()))

            def EvaluateValidation():
                split_entries.append(float(brain.runtime.item()))
                split_random.append(DrawRandom())
                brain.eval()
                brain.runtime.fill_(11.0)
                brain.world.physical.fill_(77.0)
                brain.mem.durable.fill_(55.0)
                return 1.25

            def EvaluateTest():
                split_entries.append(float(brain.runtime.item()))
                split_random.append(DrawRandom())
                brain.eval()
                brain.runtime.fill_(99.0)
                brain.world.physical.fill_(99.0)
                brain.mem.durable.fill_(99.0)
                return 2.5

            result = manager.EvaluateValidationAndTestWithRestoredBrainBuffers(
                brain,
                EvaluateValidation,
                EvaluateTest)
            saved_buffers = brain.ExportBuffers()
            random_after_evaluation = DrawRandom()
            manager.RestoreRngState(evaluation_start_rng)
            expected_random = DrawRandom()
            ok = (
                result == (1.25, 2.5)
                and split_entries == [3.0, 3.0]
                and split_random == [expected_random, expected_random]
                and random_after_evaluation == expected_random
                and brain.training
                and not brain.child.training
                and "pending" in brain.training_graph
                and torch.equal(saved_buffers["runtime"], torch.tensor([3.0]))
                and torch.equal(brain.world.physical, torch.tensor([7.0]))
                and torch.equal(brain.mem.durable, torch.tensor([5.0])))
            print(f"Manager evaluation runtime/RNG restore {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager evaluation runtime restore error: {e}")
            return False
        finally:
            if caller_rng is not None:
                manager.RestoreRngState(caller_rng)

    def TestDeploymentManifestRoutesOneCompleteGeneration(self) -> bool:
        try:
            import tempfile as tempfile_module

            manager = ManagerFunction.__new__(ManagerFunction)
            robot = Robot.CreateDefault()
            brain_build_spec = BrainBuildSpec.Compile(
                ModuleDim.CognitiveProfile(),
                robot.ContractView)
            with tempfile_module.TemporaryDirectory() as directory:
                root = Path(directory)
                configured_model = root / "model.pth"
                configured_memory = root / "memory.pth"
                missing_manifest_rejected = False
                try:
                    manager.ResolveDeploymentArtifactPaths(
                        configured_model,
                        calibrationId="calibration-a",
                        brainBuildSpec=brain_build_spec)
                except FileNotFoundError:
                    missing_manifest_rejected = True

                generation = root / ".model_deployments" / "generation-1"
                model = generation / "model.pth"
                world = generation / "world.pth"
                memory = generation / "memory.pth"
                for path in (model, world, memory):
                    manager.AtomicTorchSave({"complete": True}, path)
                manager.AtomicJsonSave({
                    "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                    "calibration_id": "calibration-a",
                    "description_id": (
                        brain_build_spec.contract_view.description_id),
                    "model_contract_id": brain_build_spec.model_signature,
                    "adapter_id": brain_build_spec.contract_view.adapter_id,
                    "generation": "generation-1",
                    "model_path": str(model),
                    "world_memory_path": str(world),
                    "memory_path": str(memory),
                }, manager.DeploymentManifestPath(configured_model))
                resolved = manager.ResolveDeploymentArtifactPaths(
                    configured_model,
                    calibrationId="calibration-a",
                    brainBuildSpec=brain_build_spec)
                mismatch_rejected = False
                try:
                    manager.ResolveDeploymentArtifactPaths(
                        configured_model,
                        calibrationId="calibration-b",
                        brainBuildSpec=brain_build_spec)
                except ValueError:
                    mismatch_rejected = True
                mixed_generation = dict(json.loads(
                    manager.DeploymentManifestPath(configured_model).read_text(
                        encoding="utf-8")))
                mixed_generation["memory_path"] = str(configured_memory.resolve())
                manager.AtomicTorchSave({"complete": True}, configured_memory)
                manager.AtomicJsonSave(
                    mixed_generation,
                    manager.DeploymentManifestPath(configured_model))
                try:
                    manager.ResolveDeploymentArtifactPaths(
                        configured_model,
                        calibrationId="calibration-a",
                        brainBuildSpec=brain_build_spec)
                    mixed_generation_rejected = False
                except ValueError:
                    mixed_generation_rejected = True
                ok = (
                    resolved == (str(model), str(world), str(memory))
                    and missing_manifest_rejected
                    and mismatch_rejected
                    and mixed_generation_rejected)
            print(f"Manager deployment generation routing {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager deployment generation routing error: {e}")
            return False

    def TestLoadCheckpointRestoresTopologyBeforeOptimizers(self) -> bool:
        try:
            import io

            events: List[str] = []

            class FakeMemory:
                def ValidateDurableState(self, state, *, expectedBatch):
                    if state != {"batch_size": expectedBatch}:
                        raise ValueError("invalid durable memory")
                    events.append("memory_validate")

                def ImportDurableState(self, state):
                    events.append("memory_import")

            class FakeWorld:
                def ValidateMemoryPayload(self, state):
                    events.append("world_validate")
                    return int(state["batch_size"]), 1

                def ImportMemoryPayload(self, state, *, batchSize):
                    if state["batch_size"] != batchSize:
                        raise ValueError("invalid World memory")
                    events.append("world_import")

            class FakeBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.calibration_id = "test-calibration"
                    self.description_id = "test-description"
                    self.model_contract_id = "test-model-contract"
                    self.adapter_id = "test-adapter"
                    self.register_buffer("buffer", torch.zeros(2))
                    self.mem = FakeMemory()

                def load_state_dict(self, state, strict):
                    events.append(f"brain_load:{strict}")
                    return super().load_state_dict(state, strict=strict)

                def ImportBuffers(self, state):
                    events.append("buffers")

                def ValidateBufferState(self, state):
                    if state.get("cognitive_state") == "invalid":
                        raise ValueError("invalid nested brain buffers")
                    events.append("buffers_validate")
                    return state.get("cognitive_state"), 1

            class FakeOptimizer:
                def __init__(self, name):
                    self.name = name

                def load_state_dict(self, state):
                    events.append(self.name)

            class FakeAgent:
                def __init__(self):
                    self.world = FakeWorld()
                    self.opt_actor = FakeOptimizer("opt_actor")
                    self.opt_critic = FakeOptimizer("opt_critic")
                    self.opt_world = FakeOptimizer("opt_world")

                def GetRuntimeWorld(self):
                    return self.world

                def ValidateOnlineCandidateState(self, state):
                    if state != {
                        "model_signature": "test-model-contract",
                        "wrappers": {},
                    }:
                        raise ValueError("unexpected online candidates")
                    events.append("candidates_validate")
                    return state["wrappers"]

                def ImportOnlineCandidateState(self, state):
                    if state != {
                        "model_signature": "test-model-contract",
                        "wrappers": {},
                    }:
                        raise ValueError("unexpected online candidates")
                    events.append("candidates")

                def SyncTrainableOptimizers(self):
                    events.append("sync")

            class FakeDataset:
                world_frame_id = "test-world"

                def __len__(self):
                    return 3

                def __getitem__(self, index):
                    return index

            manager = ManagerFunction.__new__(ManagerFunction)
            manager.device = torch.device("cpu")

            class FakeRobot:
                def Reset(self):
                    events.append("runtime")

            manager.robot = FakeRobot()
            checkpoint = {
                "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                "calibration_id": "test-calibration",
                "description_id": "test-description",
                "model_contract_id": "test-model-contract",
                "adapter_id": "test-adapter",
                "world_frame_id": "test-world",
                "epoch": 3,
                "next_batch_index": 0,
                "epoch_loss_sum": 0.0,
                "best_val": 0.25,
                "no_improve": 2,
                "train_stage": "full",
                "batch_size": 1,
                "online_learning": False,
                "brain": {"buffer": torch.zeros(2)},
                "online_candidates": {
                    "model_signature": "test-model-contract",
                    "wrappers": {},
                },
                "opt_actor": {},
                "opt_critic": {},
                "opt_world": {},
                "train_indices": [0],
                "val_indices": [1],
                "test_indices": [2],
                "processed_sample_count_total": 9,
                "rng": manager.CaptureRngState(),
                "buffers": {
                    **{name: None for name in BRAIN_RUNTIME_BUFFER_FIELDS},
                    "schema_version": BRAIN_RUNTIME_SCHEMA_VERSION,},
                "world_memory": {"batch_size": 1},
                "memory_durable": {"batch_size": 1},}
            payload = io.BytesIO()
            torch.save(checkpoint, payload)
            payload.seek(0)
            dataset = FakeDataset()
            resume_state = manager.LoadCheckpoint(
                FakeBrain(),
                FakeAgent(),
                dataset,
                payload,
                batchSize=1,
                trainStage="full",
                onlineLearning=False)
            success_events = list(events)

            def serialized(value):
                result = io.BytesIO()
                torch.save(value, result)
                result.seek(0)
                return result

            missing = dict(checkpoint)
            missing.pop("rng")
            try:
                manager.LoadCheckpoint(
                    FakeBrain(), FakeAgent(), dataset, serialized(missing),
                    batchSize=1, trainStage="full", onlineLearning=False)
                missing_rejected = False
            except ValueError:
                missing_rejected = True

            extra = dict(checkpoint)
            extra["unexpected_checkpoint_field"] = torch.zeros(())
            try:
                manager.LoadCheckpoint(
                    FakeBrain(), FakeAgent(), dataset, serialized(extra),
                    batchSize=1, trainStage="full", onlineLearning=False)
                extra_rejected = False
            except ValueError:
                extra_rejected = True

            malformed_buffers = dict(checkpoint)
            malformed_buffers["buffers"] = dict(checkpoint["buffers"])
            malformed_buffers["buffers"]["legacy_text_state"] = {
                "batch_size": 2}
            events.clear()
            try:
                manager.LoadCheckpoint(
                    FakeBrain(),
                    FakeAgent(),
                    dataset,
                    serialized(malformed_buffers),
                    batchSize=1,
                    trainStage="full",
                    onlineLearning=False)
                malformed_buffers_rejected_before_load = False
            except ValueError:
                malformed_buffers_rejected_before_load = not any(
                    event.startswith("brain_load")
                    for event in events)

            malformed_nested_buffers = dict(checkpoint)
            malformed_nested_buffers["buffers"] = dict(
                checkpoint["buffers"])
            malformed_nested_buffers["buffers"][
                "cognitive_state"] = "invalid"
            events.clear()
            try:
                manager.LoadCheckpoint(
                    FakeBrain(),
                    FakeAgent(),
                    dataset,
                    serialized(malformed_nested_buffers),
                    batchSize=1,
                    trainStage="full",
                    onlineLearning=False)
                malformed_nested_rejected_before_load = False
            except ValueError:
                malformed_nested_rejected_before_load = not any(
                    event.startswith("brain_load")
                    for event in events)

            malformed_candidates = dict(checkpoint)
            malformed_candidates["online_candidates"] = {
                "model_signature": "other-model-contract",
                "wrappers": {},
            }
            events.clear()
            try:
                manager.LoadCheckpoint(
                    FakeBrain(),
                    FakeAgent(),
                    dataset,
                    serialized(malformed_candidates),
                    batchSize=1,
                    trainStage="full",
                    onlineLearning=False)
                malformed_candidates_rejected_before_load = False
            except ValueError:
                malformed_candidates_rejected_before_load = not any(
                    event.startswith("brain_load")
                    for event in events)

            class FailingOptimizer(FakeOptimizer):
                def load_state_dict(self, state):
                    raise ValueError("optimizer topology mismatch")

            failing_agent = FakeAgent()
            failing_agent.opt_actor = FailingOptimizer("opt_actor")
            try:
                manager.LoadCheckpoint(
                    FakeBrain(), failing_agent, dataset, serialized(checkpoint),
                    batchSize=1, trainStage="full", onlineLearning=False)
                optimizer_error_propagated = False
            except ValueError as error:
                optimizer_error_propagated = (
                    str(error) == "optimizer topology mismatch")

            malformed_rng = dict(checkpoint)
            malformed_rng["rng"] = dict(checkpoint["rng"])
            malformed_rng["rng"].pop("cuda_all")
            try:
                manager.LoadCheckpoint(
                    FakeBrain(), FakeAgent(), dataset, serialized(malformed_rng),
                    batchSize=1, trainStage="full", onlineLearning=False)
                rng_error_propagated = False
            except ValueError as error:
                rng_error_propagated = (
                    str(error)
                    == "training RNG fields do not match the current schema")

            mismatch_rejected_before_load = True
            for keyword, value in (
                ("batchSize", 2),
                ("trainStage", "world"),
                ("onlineLearning", True),
            ):
                events.clear()
                arguments = {
                    "batchSize": 1,
                    "trainStage": "full",
                    "onlineLearning": False,}
                arguments[keyword] = value
                try:
                    manager.LoadCheckpoint(
                        FakeBrain(),
                        FakeAgent(),
                        dataset,
                        serialized(checkpoint),
                        **arguments)
                    mismatch_rejected_before_load = False
                except ValueError:
                    mismatch_rejected_before_load &= not any(
                        event.startswith("brain_load")
                        for event in events)

            ok = (
                success_events == [
                    "candidates_validate",
                    "world_validate",
                    "memory_validate",
                    "buffers_validate",
                    "brain_load:False",
                    "candidates",
                    "sync",
                    "opt_actor",
                    "opt_critic",
                    "opt_world",
                    "world_import",
                    "memory_import",
                    "buffers",
                    "runtime",]
                and resume_state.no_improve == 2
                and missing_rejected
                and extra_rejected
                and malformed_buffers_rejected_before_load
                and malformed_nested_rejected_before_load
                and malformed_candidates_rejected_before_load
                and optimizer_error_propagated
                and rng_error_propagated
                and mismatch_rejected_before_load)
            print(f"Manager checkpoint restore order {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager checkpoint restore order error: {e}")
            return False

    def TestLoadBrainWeightsStrictModelStateAndSync(self) -> bool:
        try:
            events: List[str] = []

            from FunctionTools import GrowableLoRALinear

            class FakeBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.calibration_id = "test-calibration"
                    self.description_id = "test-description"
                    self.model_contract_id = "test-model-contract"
                    self.adapter = GrowableLoRALinear(nn.Linear(2, 2))
                    self.adapter.Grow(1)

                def load_state_dict(self, state, strict):
                    events.append(f"load:{strict}")
                    return super().load_state_dict(state, strict=strict)

            class FakeAgent:
                def ResetOnlineCandidateState(self):
                    events.append("reset_candidates")

                def ClearTrainableOptimizerState(self):
                    events.append("clear_optimizer_state")

            brain = FakeBrain()
            manager = ManagerFunction.__new__(ManagerFunction)
            manager.LoadTorchPayload = lambda path: {
                "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION - 1,
                "calibration_id": "test-calibration",
                "model_contract_id": "test-model-contract",
                "brain": {}}
            old_schema_rejected = False
            try:
                manager.LoadBrainWeights(brain, "unused.pth")
            except ValueError:
                old_schema_rejected = True

            manager.LoadTorchPayload = lambda path: {
                "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                "calibration_id": "other-calibration",
                "model_contract_id": "test-model-contract",
                "brain": {}}
            calibration_mismatch_rejected = False
            try:
                manager.LoadBrainWeights(brain, "unused.pth")
            except ValueError:
                calibration_mismatch_rejected = True

            manager.LoadTorchPayload = lambda path: {
                "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                "calibration_id": "test-calibration",
                "model_contract_id": "other-model-contract",
                "brain": {}}
            model_mismatch_rejected = False
            try:
                manager.LoadBrainWeights(brain, "unused.pth")
            except ValueError:
                model_mismatch_rejected = True

            manager.LoadTorchPayload = lambda path: {
                "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                "calibration_id": "test-calibration",
                "model_contract_id": "test-model-contract",
                "brain": {
                    "adapter.target.weight": torch.randn_like(brain.adapter.target.weight),
                    "adapter.target.bias": torch.randn_like(brain.adapter.target.bias),
                    "adapter.topology_count": torch.zeros((), dtype=torch.long)}}
            manager.LoadBrainWeights(
                brain,
                "unused.pth",
                agent=FakeAgent())
            ok = (
                old_schema_rejected
                and calibration_mismatch_rejected
                and model_mismatch_rejected
                and events == [
                    "load:False",
                    "reset_candidates",
                    "clear_optimizer_state"]
                and len(brain.adapter.A_list) == 0)
            print(f"Manager parameter override restore order {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager parameter override restore order error: {e}")
            return False

    def TestTrainStageIsolationContract(self) -> bool:
        try:
            class FakeBrain:
                def __init__(self):
                    self.perc = nn.Identity()
                    self.attn = nn.Identity()
                    self.actor = nn.Identity()
                    self.world = nn.Identity()
                    self.critic = nn.Identity()
                    self.intention = nn.Identity()

            brain = FakeBrain()
            world = ManagerFunction.TrainStageOnlineWrappers(brain, "world")
            policy = ManagerFunction.TrainStageOnlineWrappers(brain, "policy")
            full = ManagerFunction.TrainStageOnlineWrappers(brain, "full")
            invalid_rejected = False
            try:
                ManagerFunction.NormalizeTrainStage("invalid")
            except ValueError:
                invalid_rejected = True
            non_string_rejected = False
            try:
                ManagerFunction.NormalizeTrainStage(1)  # type: ignore[arg-type]
            except TypeError:
                non_string_rejected = True

            ok = True
            ok &= world == [brain.world]
            ok &= brain.world not in policy
            ok &= brain.critic in policy and brain.actor not in policy
            ok &= brain.world in full and brain.critic in full
            ok &= brain.actor not in full
            ok &= ManagerFunction.TrainStageLossNames("world") == ("world",)
            ok &= ManagerFunction.TrainStageLossNames("policy") == ("critic", "policy")
            ok &= ManagerFunction.TrainStageLossNames("full") == ("world", "critic", "policy")
            ok &= invalid_rejected
            ok &= non_string_rejected

            world_param = nn.Parameter(torch.tensor(0.0))
            policy_param = nn.Parameter(torch.tensor(0.0))
            world_opt = torch.optim.SGD([world_param], lr=1.0)
            critic_opt = torch.optim.SGD([policy_param], lr=1.0)
            fake_agent = Agent.__new__(Agent)
            fake_agent.is_train = True
            fake_agent.opt_world = world_opt
            fake_agent.opt_critic = critic_opt
            fake_agent.opt_actor = critic_opt
            world_param.grad = torch.tensor(1.0)
            policy_param.grad = torch.tensor(100.0)
            torch.nn.utils.clip_grad_norm_(
                fake_agent.OptimizerParameters([world_opt]),
                1.0)
            ok &= float(world_param.grad.abs().item()) > 0.99
            ok &= float(policy_param.grad.abs().item()) == 100.0

            isolated_world = nn.Parameter(torch.tensor(1.0))
            isolated_critic = nn.Parameter(torch.tensor(2.0))
            isolated_policy = nn.Parameter(torch.tensor(3.0))
            shared = isolated_world + isolated_critic + isolated_policy
            world_objective = shared.square()
            critic_objective = (2.0 * shared).square()
            policy_objective = (3.0 * shared).square()
            world_objective.backward(inputs=[isolated_world], retain_graph=True)
            critic_objective.backward(inputs=[isolated_critic], retain_graph=True)
            policy_objective.backward(inputs=[isolated_policy])
            ok &= isolated_world.grad is not None
            ok &= isolated_critic.grad is not None
            ok &= isolated_policy.grad is not None
            ok &= float(isolated_world.grad.item()) == 12.0
            ok &= float(isolated_critic.grad.item()) == 48.0
            ok &= float(isolated_policy.grad.item()) == 108.0
            print(f"Manager train-stage isolation contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager train-stage isolation contract error: {e}")
            return False

    def TestTemporalEnvelopeProjection(self) -> bool:
        try:
            class FakeTemporal:
                action_epoch = torch.tensor([3], dtype=torch.long)
                kind_id = torch.tensor([5], dtype=torch.long)
                candidate_selected = torch.tensor([True])
                cache_selected = torch.tensor([False])
                hold_requested = torch.tensor([False])
                stop_requested = torch.tensor([True])
                failsafe = torch.tensor([True])

            class FakeAgent:
                def Act(self, request):
                    return ContractAgentActOutput(
                        action_request="request",
                        cognitive_readout="readout",
                        packed_target="target",
                        packed_temporal=FakeTemporal(),
                        decision={},
                        ocr=None,
                        intention_texts=[])

            class FakeFeedback:
                pass

            handle = AgentHandle.__new__(AgentHandle)
            handle.device = torch.device("cpu")
            handle.agent = FakeAgent()
            output = handle.ForwardStep(
                torch.zeros(1, 3, 2, 2),
                depth=torch.zeros(1, 1, 2, 2),
                depthValid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
                feedbackPacket=FakeFeedback())
            ok = (
                output.packed_target == "target"
                and output.packed_temporal is not None
                and bool(output.packed_temporal.failsafe.item()))
            print(
                f"Manager temporal envelope projection "
                f"{'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as exception:
            print(
                f"Manager temporal envelope projection error: "
                f"{exception}")
            return False

    def TestOcrTrainingArtifactFallback(self) -> bool:
        try:
            manager = ManagerFunction.__new__(ManagerFunction)
            manager.device = torch.device("cpu")
            with tempfile.TemporaryDirectory() as temporaryDirectory:
                artifactPath = Path(temporaryDirectory) / "ocr-parameters.pth"
                torch.save({"value": 7}, artifactPath)
                loaded, value = manager.TryLoadOCRTrainingArtifact(
                    str(artifactPath),
                    lambda path: torch.load(
                        path,
                        map_location="cpu",
                        weights_only=True)["value"],
                    logPrefix="TestOCR",
                    artifactName="parameters")
                incompatible, incompatibleValue = manager.TryLoadOCRTrainingArtifact(
                    str(artifactPath),
                    lambda path: (_ for _ in ()).throw(
                        ValueError("shape mismatch")),
                    logPrefix="TestOCR",
                    artifactName="parameters")
                missing, missingValue = manager.TryLoadOCRTrainingArtifact(
                    str(Path(temporaryDirectory) / "missing.pth"),
                    lambda path: None,
                    logPrefix="TestOCR",
                    artifactName="parameters")
            ok = (
                loaded
                and value == 7
                and not incompatible
                and incompatibleValue is None
                and not missing
                and missingValue is None)
            print(
                f"Manager OCR training artifact fallback "
                f"{'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as error:
            print(f"Manager OCR training artifact fallback error: {error}")
            return False

    def TestOcrCheckpointStrictContract(self) -> bool:
        caller_rng = None
        runtime_schema_version = TRAIN_CHECKPOINT_SCHEMA_VERSION
        try:
            import io

            strict_loads: List[Tuple[str, bool]] = []

            class FakeRecognizer(nn.Linear):
                def load_state_dict(self, state_dict, strict=True):
                    strict_loads.append(("recognizer", strict))
                    return super().load_state_dict(state_dict, strict=strict)

            class FakeOcrEngine(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone_weight = nn.Parameter(torch.ones(2))
                    self.recognizer = FakeRecognizer(2, 2)

                def load_state_dict(self, state_dict, strict=True):
                    strict_loads.append(("ocr", strict))
                    return super().load_state_dict(state_dict, strict=strict)

                def OcrMetadata(self):
                    return {
                        "vocab": ["<blank>", "a"],
                        "blank_index": 0,
                        "addon_cfg": {
                            "db_residual": True,
                            "rec_residual_rank": 1,
                            "width_aware_ctc": True,},}

            def serialized(value):
                result = io.BytesIO()
                torch.save(value, result)
                result.seek(0)
                return result

            manager = ManagerFunction.__new__(ManagerFunction)
            manager.device = torch.device("cpu")
            dataset = torch.utils.data.TensorDataset(torch.arange(6))
            engine = FakeOcrEngine()
            optimizer = torch.optim.AdamW(engine.parameters(), lr=1e-3)
            caller_rng = manager.CaptureRngState()
            globals()["TRAIN_CHECKPOINT_SCHEMA_VERSION"] = (
                OCR_CHECKPOINT_SCHEMA_VERSION + 1)
            saved_payloads: Dict[str, Any] = {}
            manager.AtomicTorchSave = lambda payload, path: saved_payloads.__setitem__(
                str(path), payload)
            manager.SaveOCRParameters(engine, "ocr-parameters.pth")
            manager.SaveOCRRecognizerParameters(
                engine,
                "ocr-recognizer-parameters.pth")
            manager.LoadOCRWeightsIntoEngine(
                engine,
                serialized(saved_payloads["ocr-parameters.pth"]))
            manager.LoadRecognizerWeightsIntoEngine(
                engine,
                serialized(saved_payloads["ocr-recognizer-parameters.pth"]))
            legacy_ocr_parameters = dict(
                saved_payloads["ocr-parameters.pth"])
            legacy_ocr_parameters["schema_version"] = 15
            legacy_recognizer_parameters = dict(
                saved_payloads["ocr-recognizer-parameters.pth"])
            legacy_recognizer_parameters["schema_version"] = 15
            try:
                manager.LoadOCRWeightsIntoEngine(
                    engine,
                    serialized(legacy_ocr_parameters))
                legacy_ocr_rejected = False
            except ValueError:
                legacy_ocr_rejected = True
            try:
                manager.LoadRecognizerWeightsIntoEngine(
                    engine,
                    serialized(legacy_recognizer_parameters))
                legacy_recognizer_rejected = False
            except ValueError:
                legacy_recognizer_rejected = True
            checkpoint = {
                "schema_version": OCR_CHECKPOINT_SCHEMA_VERSION,
                "epoch": 2,
                "best_val": 0.25,
                "ocr": engine.state_dict(),
                "ocr_meta": manager.CurrentOcrMetadata(engine),
                "optimizer": optimizer.state_dict(),
                "train_indices": [0, 1],
                "val_indices": [2, 3],
                "test_indices": [4, 5],
                "processed_sample_count_total": 12,
                "rng": manager.CaptureRngState(),
                "train_detection": True,
                "train_recognition": True,}
            restored = manager.LoadOCRCheckpoint(
                engine,
                optimizer,
                dataset,
                serialized(checkpoint),
                trainDetection=True,
                trainRecognition=True)
            legacy_checkpoint = dict(checkpoint)
            legacy_checkpoint["schema_version"] = 15
            try:
                manager.LoadOCRCheckpoint(
                    engine,
                    optimizer,
                    dataset,
                    serialized(legacy_checkpoint),
                    trainDetection=True,
                    trainRecognition=True)
                legacy_checkpoint_rejected = False
            except ValueError:
                legacy_checkpoint_rejected = True

            missing = dict(checkpoint)
            missing.pop("rng")
            try:
                manager.LoadOCRCheckpoint(
                    FakeOcrEngine(),
                    torch.optim.AdamW(FakeOcrEngine().parameters(), lr=1e-3),
                    dataset,
                    serialized(missing),
                    trainDetection=True,
                    trainRecognition=True)
                missing_rejected = False
            except ValueError:
                missing_rejected = True

            try:
                manager.LoadOCRCheckpoint(
                    engine,
                    optimizer,
                    dataset,
                    serialized(checkpoint),
                    trainDetection=False,
                    trainRecognition=True)
                mode_mismatch_rejected = False
            except ValueError:
                mode_mismatch_rejected = True

            legacy_parameter = {
                "ocr": engine.state_dict(),
                "brain": {
                    f"OCR.{name}": value
                    for name, value in engine.state_dict().items()},}
            try:
                manager.LoadOCRWeightsIntoEngine(
                    engine,
                    serialized(legacy_parameter))
                legacy_parameter_rejected = False
            except ValueError:
                legacy_parameter_rejected = True

            runtime_schema_parameter = dict(saved_payloads["ocr-parameters.pth"])
            runtime_schema_parameter["schema_version"] = (
                TRAIN_CHECKPOINT_SCHEMA_VERSION)
            try:
                manager.LoadOCRWeightsIntoEngine(
                    engine,
                    serialized(runtime_schema_parameter))
                runtime_schema_parameter_rejected = False
            except ValueError:
                runtime_schema_parameter_rejected = True

            recognizer_optimizer = torch.optim.AdamW(
                engine.recognizer.parameters(),
                lr=1e-3)
            recognizer_checkpoint = {
                "schema_version": OCR_CHECKPOINT_SCHEMA_VERSION,
                "epoch": 3,
                "best_val": 0.1,
                "recognizer": engine.recognizer.state_dict(),
                "ocr_meta": manager.CurrentOcrMetadata(engine),
                "optimizer": recognizer_optimizer.state_dict(),
                "train_indices": [0, 1],
                "val_indices": [2, 3],
                "test_indices": [4, 5],
                "processed_sample_count_total": 18,
                "rng": manager.CaptureRngState(),}
            recognizer_restored = manager.LoadOCRRecognizerCheckpoint(
                engine,
                recognizer_optimizer,
                dataset,
                serialized(recognizer_checkpoint))
            legacy_recognizer_checkpoint = dict(recognizer_checkpoint)
            legacy_recognizer_checkpoint["schema_version"] = 15
            try:
                manager.LoadOCRRecognizerCheckpoint(
                    engine,
                    recognizer_optimizer,
                    dataset,
                    serialized(legacy_recognizer_checkpoint))
                legacy_recognizer_checkpoint_rejected = False
            except ValueError:
                legacy_recognizer_checkpoint_rejected = True

            ok = (
                restored[0:3] == (2, 0.25, 12)
                and [list(split.indices) for split in restored[3:]]
                == [[0, 1], [2, 3], [4, 5]]
                and recognizer_restored[0:3] == (3, 0.1, 18)
                and legacy_ocr_rejected
                and legacy_recognizer_rejected
                and legacy_checkpoint_rejected
                and legacy_recognizer_checkpoint_rejected
                and missing_rejected
                and mode_mismatch_rejected
                and legacy_parameter_rejected
                and runtime_schema_parameter_rejected
                and TRAIN_CHECKPOINT_SCHEMA_VERSION
                == OCR_CHECKPOINT_SCHEMA_VERSION + 1
                and saved_payloads["ocr-parameters.pth"]["schema_version"]
                == OCR_CHECKPOINT_SCHEMA_VERSION
                and saved_payloads[
                    "ocr-recognizer-parameters.pth"]["schema_version"]
                == OCR_CHECKPOINT_SCHEMA_VERSION
                and strict_loads == [
                    ("ocr", True),
                    ("recognizer", True),
                    ("ocr", True),
                    ("recognizer", True),])
            print(
                f"Manager OCR strict checkpoint contract "
                f"{'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager OCR strict checkpoint contract error: {e}")
            return False
        finally:
            globals()["TRAIN_CHECKPOINT_SCHEMA_VERSION"] = runtime_schema_version
            if caller_rng is not None:
                manager.RestoreRngState(caller_rng)

    def TestSequentialLoaderRejectsEmptyRecurrentSplit(self) -> bool:
        try:
            dataset = torch.utils.data.TensorDataset(torch.arange(3))
            undersized_rejected = False
            try:
                SequentialTrajectoryLoader(dataset, batchSize=4)
            except ValueError:
                undersized_rejected = True

            remainder_dataset = torch.utils.data.TensorDataset(torch.arange(9))
            remainder_rejected = False
            try:
                SequentialTrajectoryLoader(remainder_dataset, batchSize=4)
            except ValueError:
                remainder_rejected = True

            full_dataset = torch.utils.data.TensorDataset(torch.arange(8))
            loader = SequentialTrajectoryLoader(full_dataset, batchSize=2)
            resumed = list(loader.IterFrom(2))
            ok = bool(
                undersized_rejected
                and remainder_rejected
                and len(loader) == 4
                and len(resumed) == 2
                and torch.equal(resumed[0][0], torch.tensor([2, 6])))
            print(
                f"Manager sequential-loader recurrent split "
                f"{'passed' if ok else 'failed'}")
            return ok
        except Exception as e:
            print(f"Manager sequential-loader recurrent split error: {e}")
            return False

    def TestCppDecisionWireContract(self) -> bool:
        original_converter = DataPreprocessor.ConvertCppPerceptionFrame
        processing_counts = {
            "convert": 0,
            "encode_feedback": 0,
            "decode_feedback": 0,
            "validate_feedback": 0,
        }

        def ConvertFrame(
            bitmap,
            reward,
            done,
            *,
            depthBitmap,
            depthValid,
            device,
            needVisualState,
        ):
            processing_counts["convert"] += 1
            if type(bitmap) is str:
                raise ValueError("invalid test image")
            return {
                "frames": torch.zeros(1, 3, 1, 1, device=device),
                "rewards": (
                    None
                    if reward is None
                    else torch.tensor([reward], device=device)),
                "dones": (
                    None
                    if done is None
                    else torch.tensor([done], device=device)),
                "depths": torch.zeros(1, 1, 1, 1, device=device),
                "depth_valid": torch.ones(
                    1, 1, 1, 1, device=device, dtype=torch.bool),
            }

        DataPreprocessor.ConvertCppPerceptionFrame = ConvertFrame
        try:
            manager = object.__new__(ManagerFunction)
            manager.device = torch.device("cpu")
            manager.robot = Robot.CreateDefault()
            manager.brain_build_spec = BrainBuildSpec.Compile(
                ModuleDim.CognitiveProfile(),
                manager.robot.ContractView)
            manager.perception_calibration_id = (
                manager.robot.ContractView.perception_projection.calibration_id)
            manager.active_sensor_stream_id = None
            manager.active_world_frame_id = None
            manager.last_sensor_sequence_index = None
            manager.pending_action_request = None
            manager.stream_terminated = False
            manager.agent_handle = object()
            captured: Dict[str, Any] = {}
            original_encode_feedback = manager.robot.EncodeFeedback
            original_decode_feedback = manager.robot.DecodeFeedback
            original_validate_feedback = manager.robot.ValidateFeedback

            def EncodeFeedback(*args, **kwargs):
                processing_counts["encode_feedback"] += 1
                return original_encode_feedback(*args, **kwargs)

            def DecodeFeedback(*args, **kwargs):
                processing_counts["decode_feedback"] += 1
                return original_decode_feedback(*args, **kwargs)

            def ValidateFeedback(*args, **kwargs):
                processing_counts["validate_feedback"] += 1
                return original_validate_feedback(*args, **kwargs)

            manager.robot.EncodeFeedback = EncodeFeedback
            manager.robot.DecodeFeedback = DecodeFeedback
            manager.robot.ValidateFeedback = ValidateFeedback

            target = PackedEndEffectorTarget(
                values=torch.zeros(
                    1,
                    manager.robot.ContractView.end_effector_target_layout.PackedDim),
                active=torch.zeros(
                    1,
                    manager.robot.ContractView.end_effector_count,
                    dtype=torch.bool),
                contract_id=manager.robot.ContractView.contract_id,
                model_signature=manager.robot.ContractView.model_signature,
                target_version=torch.zeros(1, dtype=torch.long),
                timestamp=torch.zeros(1))
            request = ActionRequest(
                request_id=torch.ones(1, dtype=torch.long),
                action_epoch=torch.zeros(1, dtype=torch.long),
                target=target,
                command_active=torch.zeros(1, dtype=torch.bool),
                hold_requested=torch.ones(1, dtype=torch.bool),
                stop_requested=torch.zeros(1, dtype=torch.bool),
                help_requested=torch.zeros(1, dtype=torch.bool),
                policy_path=torch.full(
                    (1,),
                    POLICY_PATH_FULL,
                    dtype=torch.long),
                planner_override=torch.zeros(1, dtype=torch.bool),
                temporal_kind_id=torch.zeros(1, dtype=torch.long),
                timestamp=torch.zeros(1))
            feature = torch.zeros(1, 1)
            readout = CognitiveReadout(
                schema_version=COGNITIVE_READOUT_SCHEMA_VERSION,
                model_signature=manager.brain_build_spec.model_signature,
                contract_id=manager.robot.ContractView.contract_id,
                request_id=request.request_id,
                timestamp=torch.zeros(1),
                row_valid=torch.ones(1, dtype=torch.bool),
                intention_feature=feature,
                intention_valid=torch.ones(1, dtype=torch.bool),
                intention_age=torch.zeros(1),
                world_belief_feature=feature,
                world_belief_valid=torch.ones(1, dtype=torch.bool),
                world_belief_age=torch.zeros(1),
                sensorimotor_evidence=feature,
                sensorimotor_valid=torch.ones(1, dtype=torch.bool),
                sensorimotor_age=torch.zeros(1),
                decision_feature=feature,
                decision_valid=torch.ones(1, dtype=torch.bool),
                decision_age=torch.zeros(1),
                compute_mode=torch.zeros(1, dtype=torch.long),
                policy_path=request.policy_path,
                planner_override=request.planner_override,
                option_id=torch.zeros(1, dtype=torch.long),
                option_valid=torch.zeros(1, dtype=torch.bool),
                temporal_kind_id=request.temporal_kind_id)
            act_output = ContractAgentActOutput(
                action_request=request,
                cognitive_readout=readout,
                packed_target=target,
                packed_temporal=None,
                decision={},
                ocr=None,
                intention_texts=["pick up the cup"])

            def CaptureForward(request, requestProvenance):
                if "request" not in captured:
                    captured["request"] = request
                    captured["request_provenance"] = requestProvenance
                    captured["first_processing_counts"] = dict(
                        processing_counts)
                cached_request_id = manager.robot.CachedRequestId
                captured.setdefault("committed_request_ids", []).append(
                    None
                    if cached_request_id is None
                    else int(cached_request_id.item()))
                return act_output

            manager.ForwardMaterializedContractBatch = CaptureForward
            feedback_payload = dict(
                manager.robot.BuildNeutralFeedbackPayload(
                    0.0,
                    manager.device))
            sensor_packet = {
                "schema_version": SENSOR_PACKET_WIRE_SCHEMA_VERSION,
                "stream_id": "stream-1",
                "sequence_index": 0,
                "frame_id": "frame-1",
                "calibration_id": manager.perception_calibration_id,
                "rgb_encoding": "rgb8",
                "depth_unit": "meter",
                "text_ext": ["pick up the cup"],
                "text_trust": [TEXT_TRUST_OPERATOR_COMMAND],
                "sample_actions": False,
                "deterministic_actor": True,
                "rgb": [[[0, 0, 0]]],
                "depth": [[1.0, 2.0], [3.0, 4.0]],
                "depth_valid": [[True, True], [True, False]],}
            result = ManagerFunction.AgentHandleForwardJson(
                manager,
                0.5,
                0.0,
                json.dumps(sensor_packet),
                json.dumps(feedback_payload),
                "null")
            response = json.loads(result)
            baseline_cached_request_id = (
                manager.robot.CachedRequestId.detach().clone())

            duplicate_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(sensor_packet),
                    json.dumps(feedback_payload),
                    "null")
            except ValueError:
                duplicate_rejected = True

            gap_sensor_packet = dict(sensor_packet)
            gap_sensor_packet["sequence_index"] = 2
            gap_sensor_packet["frame_id"] = "frame-3"
            gap_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(gap_sensor_packet),
                    json.dumps(feedback_payload),
                    "null")
            except ValueError:
                gap_rejected = True

            next_sensor_packet = dict(sensor_packet)
            next_sensor_packet["sequence_index"] = 1
            next_sensor_packet["frame_id"] = "frame-2"
            wrong_calibration_sensor = dict(next_sensor_packet)
            wrong_calibration_sensor["calibration_id"] = "other-calibration"
            calibration_mismatch_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(wrong_calibration_sensor),
                    json.dumps(feedback_payload),
                    "null")
            except ValueError:
                calibration_mismatch_rejected = True

            missing_result_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(next_sensor_packet),
                    json.dumps(feedback_payload),
                    "null")
            except (TypeError, ValueError):
                missing_result_rejected = True

            execution_payload = {
                "schema_version": manager.robot.ActionSchemaVersion,
                "request_id": [1],
                "action_epoch": [0],
                "applied_target": manager.robot.DecodeTarget(target),
                "execution_status": [[
                    SlotExecutionStatus.APPLIED.name
                    for _ in manager.robot.EndEffectors]],
                "execution_known": [[
                    True for _ in manager.robot.EndEffectors]],
                "hard_stop": [False],
                "help_accepted": [False],
                "timestamp": [0.1],
            }
            next_feedback_payload = dict(
                manager.robot.BuildNeutralFeedbackPayload(
                    0.2,
                    manager.device))

            def StatePreserved() -> bool:
                return bool(
                    torch.equal(
                        manager.robot.CachedRequestId,
                        baseline_cached_request_id)
                    and manager.pending_action_request is request
                    and manager.active_sensor_stream_id == "stream-1"
                    and manager.active_world_frame_id == "sensor_stream:stream-1"
                    and manager.last_sensor_sequence_index == 0
                    and not manager.stream_terminated)

            invalid_text_sensor = dict(next_sensor_packet)
            invalid_text_sensor["text_trust"] = ["invalid"]
            invalid_text_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(invalid_text_sensor),
                    json.dumps(next_feedback_payload),
                    json.dumps(execution_payload))
            except ValueError:
                invalid_text_rejected = True
            invalid_text_preserved = StatePreserved()

            invalid_image_sensor = dict(next_sensor_packet)
            invalid_image_sensor["rgb"] = "invalid"
            invalid_image_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(invalid_image_sensor),
                    json.dumps(next_feedback_payload),
                    json.dumps(execution_payload))
            except ValueError:
                invalid_image_rejected = True
            invalid_image_preserved = StatePreserved()

            invalid_feedback_payload = dict(next_feedback_payload)
            invalid_feedback_payload["contract_id"] = "invalid"
            invalid_feedback_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(next_sensor_packet),
                    json.dumps(invalid_feedback_payload),
                    json.dumps(execution_payload))
            except ValueError:
                invalid_feedback_rejected = True
            invalid_feedback_preserved = StatePreserved()

            early_feedback_payload = dict(next_feedback_payload)
            early_feedback_payload["timestamp"] = 0.05
            early_feedback_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(next_sensor_packet),
                    json.dumps(early_feedback_payload),
                    json.dumps(execution_payload))
            except ValueError:
                early_feedback_rejected = True
            early_feedback_preserved = StatePreserved()

            next_result = ManagerFunction.AgentHandleForwardJson(
                manager,
                0.5,
                1.0,
                json.dumps(next_sensor_packet),
                json.dumps(next_feedback_payload),
                json.dumps(execution_payload))
            next_response = json.loads(next_result)
            terminated_stream_rejected = False
            restarted_sensor_packet = dict(sensor_packet)
            restarted_sensor_packet["stream_id"] = "stream-2"
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.0,
                    0.0,
                    json.dumps(restarted_sensor_packet),
                    json.dumps(feedback_payload),
                    "null")
            except RuntimeError:
                terminated_stream_rejected = True

            failure_manager = object.__new__(ManagerFunction)
            failure_manager.device = torch.device("cpu")
            failure_manager.robot = Robot.CreateDefault()
            failure_manager.brain_build_spec = BrainBuildSpec.Compile(
                ModuleDim.CognitiveProfile(),
                failure_manager.robot.ContractView)
            failure_manager.perception_calibration_id = (
                failure_manager.robot.ContractView.perception_projection.calibration_id)
            failure_manager.active_sensor_stream_id = "stream-1"
            failure_manager.active_world_frame_id = "sensor_stream:stream-1"
            failure_manager.last_sensor_sequence_index = 0
            failure_manager.pending_action_request = request
            failure_manager.stream_terminated = False
            failure_manager.agent_handle = object()
            failure_manager.EncodeBrainFeedback(
                feedback_payload,
                batchSize=1)
            failure_capture: Dict[str, Any] = {"calls": 0}

            def FailForward(*args, **kwargs):
                failure_capture["calls"] += 1
                cached_request_id = failure_manager.robot.CachedRequestId
                failure_capture["request_id"] = (
                    None
                    if cached_request_id is None
                    else int(cached_request_id.item()))
                raise RuntimeError("brain failure")

            failure_manager.ForwardMaterializedContractBatch = FailForward
            postcommit_failure_raised = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    failure_manager,
                    0.5,
                    0.0,
                    json.dumps(next_sensor_packet),
                    json.dumps(next_feedback_payload),
                    json.dumps(execution_payload))
            except RuntimeError:
                postcommit_failure_raised = True
            postcommit_failure_terminated = bool(
                postcommit_failure_raised
                and failure_capture.get("request_id") == 1
                and failure_capture["calls"] == 1
                and failure_manager.robot.CachedRequestId is None
                and failure_manager.pending_action_request is None
                and failure_manager.active_sensor_stream_id is None
                and failure_manager.active_world_frame_id is None
                and failure_manager.last_sensor_sequence_index is None
                and failure_manager.stream_terminated)
            half_transaction_retry_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    failure_manager,
                    0.5,
                    0.0,
                    json.dumps(next_sensor_packet),
                    json.dumps(next_feedback_payload),
                    json.dumps(execution_payload))
            except RuntimeError:
                half_transaction_retry_rejected = True

            ok = (
                set(response) == {
                    "schema_version",
                    "request_provenance",
                    "action_request",
                    "cognitive_readout",
                    "intention_texts"}
                and response["schema_version"] == DECISION_WIRE_SCHEMA_VERSION
                and response["action_request"]["request_id"] == [1]
                and response["cognitive_readout"]["request_id"] == [1]
                and response["intention_texts"] == ["pick up the cup"]
                and next_response["request_provenance"][
                    "sequence_index"] == 1
                and next_response["action_request"] is None
                and type(captured["request"]) is ContractAgentActInput
                and captured["request"].text_ext == sensor_packet["text_ext"]
                and captured["request"].text_trust == sensor_packet["text_trust"]
                and captured["request"].sample_actions is False
                and captured["request"].deterministic_actor is True
                and torch.allclose(
                    captured["request"].reward,
                    torch.tensor([0.5]))
                and torch.allclose(
                    captured["request"].done,
                    torch.tensor([0.0]))
                and captured["first_processing_counts"] == {
                    "convert": 1,
                    "encode_feedback": 1,
                    "decode_feedback": 1,
                    "validate_feedback": 1,
                }
                and captured["request_provenance"] == {
                    "stream_id": "stream-1",
                    "sequence_index": 0,
                    "frame_id": "frame-1",
                    "calibration_id": manager.perception_calibration_id,
                    "world_frame_id": "sensor_stream:stream-1",
                    "description_id": (
                        manager.brain_build_spec.contract_view.description_id),
                    "model_contract_id": manager.brain_build_spec.model_signature,
                    "adapter_id": (
                        manager.brain_build_spec.contract_view.adapter_id),
                }
                and duplicate_rejected
                and gap_rejected
                and calibration_mismatch_rejected
                and missing_result_rejected
                and invalid_text_rejected
                and invalid_text_preserved
                and invalid_image_rejected
                and invalid_image_preserved
                and invalid_feedback_rejected
                and invalid_feedback_preserved
                and early_feedback_rejected
                and early_feedback_preserved
                and captured["committed_request_ids"] == [-1, 1]
                and terminated_stream_rejected
                and postcommit_failure_terminated
                and half_transaction_retry_rejected
                and failure_capture["calls"] == 1
                and manager.active_sensor_stream_id is None
                and manager.active_world_frame_id is None
                and manager.last_sensor_sequence_index is None
                and manager.pending_action_request is None
                and manager.stream_terminated
                and manager.robot.CachedRequestId is None)
            print(f"Manager C++ decision wire contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager C++ decision wire contract error: {e}")
            return False
        finally:
            DataPreprocessor.ConvertCppPerceptionFrame = original_converter

    def TestSingleFramePreprocessContract(self) -> bool:
        try:
            image_size = BasicParameters.IMAGE_SIZE
            rgb = torch.zeros(image_size, image_size, 3, dtype=torch.uint8)
            depth = torch.ones(image_size, image_size, dtype=torch.float32)
            depth_valid = torch.ones(image_size, image_size, dtype=torch.bool)
            pack = DataPreprocessor.PreprocessSingleFrame(
                rgb,
                0.5,
                0.0,
                depthBitmap=depth,
                depthValid=depth_valid,
                device=torch.device("cpu"))
            wrong_shape_rejected = False
            try:
                DataPreprocessor.PreprocessSingleFrame(
                    rgb[:-1],
                    None,
                    None,
                    depthBitmap=depth[:-1],
                    depthValid=depth_valid[:-1])
            except ValueError:
                wrong_shape_rejected = True
            nonfinite_reward_rejected = False
            try:
                DataPreprocessor.PreprocessSingleFrame(
                    rgb,
                    float("nan"),
                    None,
                    depthBitmap=depth,
                    depthValid=depth_valid)
            except ValueError:
                nonfinite_reward_rejected = True
            fractional_done_rejected = False
            try:
                DataPreprocessor.PreprocessSingleFrame(
                    rgb,
                    None,
                    0.5,
                    depthBitmap=depth,
                    depthValid=depth_valid)
            except ValueError:
                fractional_done_rejected = True
            ok = (
                tuple(pack["frames"].shape) == (1, 3, image_size, image_size)
                and tuple(pack["depths"].shape) == (1, 1, image_size, image_size)
                and tuple(pack["depth_valid"].shape) == (1, 1, image_size, image_size)
                and pack["frames"].dtype == torch.float32
                and pack["depths"].dtype == torch.float32
                and pack["depth_valid"].dtype == torch.bool
                and "perception_intrinsics" not in pack
                and wrong_shape_rejected
                and nonfinite_reward_rejected
                and fractional_done_rejected)
            print(f"Manager single-frame preprocess contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager single-frame preprocess contract error: {e}")
            return False

    def TestOfflinePreprocessStrictContract(self) -> bool:
        try:
            image_size = BasicParameters.IMAGE_SIZE
            rgb = torch.zeros(
                2, image_size, image_size, 3, dtype=torch.uint8)
            depth = torch.ones(
                2, image_size, image_size, dtype=torch.float32)
            depth_valid = torch.ones_like(depth, dtype=torch.bool)
            pack = DataPreprocessor.ConvertSensoryInputs(
                imgs=rgb,
                reward=torch.zeros(2),
                done=torch.zeros(2),
                depths=depth,
                depthValids=depth_valid,
                needVisualState=False)
            wrong_lattice_rejected = False
            try:
                DataPreprocessor.ConvertSensoryInputs(
                    imgs=rgb[:, :-1],
                    reward=torch.zeros(2),
                    done=torch.zeros(2),
                    depths=depth[:, :-1],
                    depthValids=depth_valid[:, :-1],
                    needVisualState=False)
            except ValueError:
                wrong_lattice_rejected = True
            wrong_mask_type_rejected = False
            try:
                DataPreprocessor.ConvertSensoryInputs(
                    imgs=rgb,
                    reward=torch.zeros(2),
                    done=torch.zeros(2),
                    depths=depth,
                    depthValids=depth_valid.float(),
                    needVisualState=False)
            except ValueError:
                wrong_mask_type_rejected = True
            wrong_feedback_shape_rejected = False
            try:
                DataPreprocessor.ConvertSensoryInputs(
                    imgs=rgb,
                    reward=torch.zeros(2, 1),
                    done=torch.zeros(2),
                    depths=depth,
                    depthValids=depth_valid,
                    needVisualState=False)
            except ValueError:
                wrong_feedback_shape_rejected = True
            out_of_range_reward_rejected = False
            try:
                DataPreprocessor.ConvertSensoryInputs(
                    imgs=rgb,
                    reward=torch.tensor([
                        float(BasicParameters.REWARD_MAX) + 1.0,
                        0.0]),
                    done=torch.zeros(2),
                    depths=depth,
                    depthValids=depth_valid,
                    needVisualState=False)
            except ValueError:
                out_of_range_reward_rejected = True
            ok = (
                tuple(pack["frames"].shape)
                == (2, 3, image_size, image_size)
                and tuple(pack["depths"].shape)
                == (2, 1, image_size, image_size)
                and torch.equal(pack["depths"], depth.unsqueeze(1))
                and wrong_lattice_rejected
                and wrong_mask_type_rejected
                and wrong_feedback_shape_rejected
                and out_of_range_reward_rejected
                and "perception_intrinsics" not in pack)
            print(f"Manager strict offline preprocess {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager offline preprocess contract error: {e}")
            return False

    def BuildAutonomousHarness(
        self,
        *,
        rolloutSteps: int = 4,
        terminated: bool = False,
        truncated: bool = True,
        executionKnown: bool = True,
        executionKnownSequence: Optional[Sequence[bool]] = None,
        inactiveUnknown: bool = False,
        executionStatus: Optional[SlotExecutionStatus] = None,
        executionStatusSequence: Optional[Sequence[SlotExecutionStatus]] = None,
        helpRequestedSequence: Optional[Sequence[bool]] = None,
        helpAcceptedSequence: Optional[Sequence[bool]] = None,
        actorCreditSequence: Optional[Sequence[bool]] = None,
        hardStop: bool = False,
        controllerOverride: bool = False,
        plannerOverride: bool = False,
    ) -> Tuple[
        AutonomousInteractionTrainer,
        DecisionExtractor,
        Any,
    ]:
        robot = Robot.CreateDefault()
        actor = TestDecisionMTool().MakeExtractor()
        config = AutonomousPpoConfig(
            rollout_steps=rolloutSteps,
            ppo_epochs=2,
            minibatch_size=2,
            discount=0.5,
            trace_decay=0.8,
            entropy_coefficient=0.0,
            help_request_cost=0.5)

        class FakePolicy:
            def __init__(self):
                self.step_index = 0
                self.value_scale = nn.Parameter(torch.tensor(0.25))
                self.planner_enabled = True
                self.value_calls = 0

            def Reset(self, observation: Any) -> None:
                self.step_index = 0
                actor.EnsureB(1)
                actor.ImportEligibilityState({
                    "trace": torch.full_like(actor.elig_plasticity.trace, 0.1),
                    "fast": torch.full_like(actor.elig_plasticity.fast, 0.05)}, 1)

            def SetPlannerEnabled(self, enabled: bool) -> None:
                self.planner_enabled = bool(enabled)

            def PlannerEnabled(self) -> bool:
                return bool(self.planner_enabled)

            def Step(self, observation: Any) -> AutonomousPolicyOutput:
                index = self.step_index
                paths = (
                    config.full_policy_path,
                    config.full_policy_path,
                    config.fast_policy_path,
                    config.detail_policy_path)
                candidates = (True, False, True, True)
                caches = (False, True, False, False)
                path = paths[min(index, len(paths) - 1)]
                base_inputs = TestDecisionMTool().MakeBaseInputs(actor, 1)
                base_inputs = {
                    name: value
                    for name, value in base_inputs.items()
                    if name not in {"sample", "deterministic"}}
                cached = torch.randn(1, actor.belief_dim)
                detail = torch.randn(1, actor.goal_decision_context_dim)
                eligibility = actor.ExportEligibilityState()
                row_index = torch.zeros(1, dtype=torch.long)
                if path == config.full_policy_path:
                    decision = actor(
                        **base_inputs,
                        sample=True,
                        deterministic=False)
                elif path == config.fast_policy_path:
                    decision = actor.ForwardFastRows(
                        row_index,
                        cached,
                        **base_inputs,
                        sample=True,
                        deterministic=False)
                else:
                    decision = actor.ForwardDetailRows(
                        row_index,
                        cached,
                        detail,
                        **base_inputs,
                        sample=True,
                        deterministic=False)
                candidate = candidates[min(index, len(candidates) - 1)]
                cache = caches[min(index, len(caches) - 1)]
                versions = (1, 1, 2, 3)
                epochs = (1, 1, 2, 3)
                version = versions[min(index, len(versions) - 1)]
                action_epoch = epochs[min(index, len(epochs) - 1)]
                timestamp = float(index + 1)
                help_requested = (
                    index == 2
                    if helpRequestedSequence is None
                    else bool(helpRequestedSequence[min(
                        index,
                        len(helpRequestedSequence) - 1)]))
                candidate = candidate or help_requested
                cache = cache and not help_requested
                actor_credit = (
                    candidate
                    if actorCreditSequence is None
                    else bool(actorCreditSequence[min(
                        index,
                        len(actorCreditSequence) - 1)]))
                target_active = torch.zeros(
                    1,
                    robot.ContractView.end_effector_count,
                    dtype=torch.bool)
                if not help_requested:
                    target_active[:, 0] = True
                target_values = torch.zeros(
                    1,
                    robot.ContractView.end_effector_target_layout.PackedDim)
                if not help_requested and index > 1:
                    target_values[:, 0] = 0.1 * float(index - 1)
                target = PackedEndEffectorTarget(
                    values=target_values,
                    active=target_active,
                    contract_id=robot.ContractView.contract_id,
                    model_signature=robot.ContractView.model_signature,
                    target_version=torch.tensor([version]),
                    timestamp=torch.tensor([timestamp]))
                request = ActionRequest(
                    request_id=torch.tensor([index + 1]),
                    action_epoch=torch.tensor([action_epoch]),
                    target=target,
                    command_active=torch.tensor([not help_requested]),
                    hold_requested=torch.tensor([False]),
                    stop_requested=torch.tensor([help_requested]),
                    help_requested=torch.tensor([help_requested]),
                    policy_path=torch.tensor([path]),
                    planner_override=torch.tensor([plannerOverride]),
                    temporal_kind_id=torch.tensor([0]),
                    timestamp=target.timestamp.clone())
                value_hidden = torch.as_tensor(
                    observation,
                    dtype=torch.float32).reshape(1, 1)
                value_conditioning = {
                    "valueHidden": value_hidden,
                    "valueBaseline": (
                        self.value_scale * value_hidden.reshape(-1)
                    ).detach().clone()}
                value = self.ReevaluateValue(value_conditioning)
                conditioning = actor.StorePolicyConditioning(decision)
                behavior_probability = actor.RecomputeActionLogProbability(
                    conditioning)["combinedActionLogProbability"]
                policy_snapshot = {
                    "policyPath": torch.tensor([path]),
                    "cachedDecisionFeature": cached,
                    "detailGoalFeature": detail,
                    "stateFeat": base_inputs["stateFeat"],
                    "intentFeat": base_inputs["intentFeat"],
                    "valueTensor": base_inputs["valueTensor"],
                    "vNextTensor": base_inputs["vNextTensor"],
                    "uncertainty": base_inputs["uncertainty"],
                    "confidence": base_inputs["confidence"],
                    "precision": base_inputs["precision"],
                    "risk": base_inputs["risk"],
                    "worldHzx": base_inputs["worldHzx"],
                    "prevFullDecisionState": base_inputs[
                        "prevDecisionState"],
                    "prevFastDecisionState": base_inputs[
                        "prevDecisionState"],
                    "prevDetailDecisionState": base_inputs[
                        "prevDecisionState"],
                    "prevFullLatentControl": base_inputs[
                        "prevLatentControl"],
                    "prevFastLatentControl": base_inputs[
                        "prevLatentControl"],
                    "prevDetailLatentControl": base_inputs[
                        "prevLatentControl"],
                    "prevActionEmbed": base_inputs["prevActionEmbed"],
                    "prevFullMapperHidden": base_inputs[
                        "prevMapperHidden"],
                    "prevFastMapperHidden": base_inputs[
                        "prevMapperHidden"],
                    "prevDetailMapperHidden": base_inputs[
                        "prevMapperHidden"],
                    "feedbackTdError": base_inputs["feedbackTdError"],
                    "prevFullOptionLogit": base_inputs[
                        "prevOptionLogit"],
                    "prevFastOptionLogit": base_inputs[
                        "prevOptionLogit"],
                    "prevDetailOptionLogit": base_inputs[
                        "prevOptionLogit"],
                    "uRaw": conditioning["uRaw"],
                    "optionIndex": conditioning["optionIndex"],
                    "eligibilityState": eligibility,
                    "eligibilityFrozen": actor.EligibilityFrozen(),
                }
                self.step_index += 1
                return AutonomousPolicyOutput(
                    action_request=request,
                    value_baseline=value,
                    behavior_log_probability=behavior_probability,
                    actor_credit_mask=torch.tensor([actor_credit]),
                    candidate_selected=torch.tensor([candidate]),
                    cache_selected=torch.tensor([cache]),
                    neutral_selected=torch.tensor([not candidate and not cache]),
                    controller_override=torch.tensor([
                        controllerOverride and index == 0]),
                    sensorimotor_inconsistency=torch.zeros(1),
                    sensorimotor_valid=torch.zeros(1, dtype=torch.bool),
                    policy_snapshot=AutonomousDecisionSnapshot(
                        conditioning=conditioning,
                        policy=policy_snapshot,
                        value_conditioning=value_conditioning))

            def Value(self, observation: Any) -> torch.Tensor:
                self.value_calls += 1
                value = torch.as_tensor(
                    observation,
                    dtype=torch.float32).reshape(-1)
                return self.value_scale * value

            def ValueAndReadout(
                self,
                observation: Any,
            ) -> Tuple[torch.Tensor, Any]:
                return self.Value(observation), None

            def ReevaluateValue(
                self,
                valueConditioning: Dict[str, torch.Tensor],
            ) -> torch.Tensor:
                return (
                    self.value_scale
                    * valueConditioning["valueHidden"].reshape(-1))

            def AfterOptimizerStep(self) -> None:
                return None

        class FakeEnvironment:
            def __init__(self):
                self.step_index = 0

            def Reset(self) -> Any:
                self.step_index = 0
                return torch.tensor([1.0])

            def MaterializeObservation(
                self,
                observation: Any,
                robotValue: Robot,
            ) -> Any:
                return observation

            def Step(
                self,
                actionRequest: ActionRequest,
            ) -> EmbodiedEnvironmentTransition:
                index = self.step_index
                self.step_index += 1
                final = self.step_index >= rolloutSteps
                known = (
                    executionKnown
                    if executionKnownSequence is None
                    else bool(executionKnownSequence[
                        min(index, len(executionKnownSequence) - 1)]))
                status = (
                    int(SlotExecutionStatus.STOPPED)
                    if actionRequest.help_requested.item()
                    else int(SlotExecutionStatus.APPLIED))
                if executionStatus is not None:
                    status = int(executionStatus)
                if executionStatusSequence is not None:
                    status = int(executionStatusSequence[min(
                        index,
                        len(executionStatusSequence) - 1)])
                if not known:
                    status = int(SlotExecutionStatus.UNKNOWN)
                if hardStop:
                    status = int(SlotExecutionStatus.STOPPED)
                applied_target = actionRequest.target
                if status == int(SlotExecutionStatus.STOPPED) or hardStop:
                    applied_target = PackedEndEffectorTarget(
                        values=torch.zeros_like(actionRequest.target.values),
                        active=torch.zeros_like(actionRequest.target.active),
                        contract_id=robot.ContractView.contract_id,
                        model_signature=robot.ContractView.model_signature,
                        target_version=actionRequest.target.target_version.clone(),
                        timestamp=actionRequest.timestamp.clone())
                elif (
                    not known
                    or status in (
                        int(SlotExecutionStatus.REJECTED),
                        int(SlotExecutionStatus.HELD))
                ):
                    cached_values = robot.CachedTargetValues
                    cached_active = robot.CachedTargetActive
                    cached_version = robot.CachedTargetVersion
                    applied_target = PackedEndEffectorTarget(
                        values=(
                            torch.zeros_like(actionRequest.target.values)
                            if cached_values is None
                            else cached_values.detach().clone()),
                        active=(
                            torch.zeros_like(actionRequest.target.active)
                            if cached_active is None
                            else cached_active.detach().clone()),
                        contract_id=robot.ContractView.contract_id,
                        model_signature=robot.ContractView.model_signature,
                        target_version=(
                            actionRequest.target.target_version.clone()
                            if cached_version is None
                            else cached_version.clamp_min(0).detach().clone()),
                        timestamp=actionRequest.timestamp.clone())
                status_tensor = torch.full(
                    (1, robot.ContractView.end_effector_count),
                    status,
                    dtype=torch.long)
                known_tensor = torch.full(
                    (1, robot.ContractView.end_effector_count),
                    known,
                    dtype=torch.bool)
                if inactiveUnknown:
                    status_tensor.fill_(int(SlotExecutionStatus.UNKNOWN))
                    known_tensor.zero_()
                    status_tensor[actionRequest.target.active] = int(
                        SlotExecutionStatus.APPLIED)
                    known_tensor[actionRequest.target.active] = True
                result = ActionExecutionResult(
                    request_id=actionRequest.request_id.clone(),
                    action_epoch=actionRequest.action_epoch.clone(),
                    applied_target=applied_target,
                    execution_status=status_tensor,
                    execution_known=known_tensor,
                    hard_stop=torch.tensor([hardStop]),
                    help_accepted=torch.tensor([
                        index == 2
                        if helpAcceptedSequence is None
                        else bool(helpAcceptedSequence[min(
                            index,
                            len(helpAcceptedSequence) - 1)])]),
                    timestamp=actionRequest.timestamp + 0.1)
                rewards = (1.0, 2.0, 3.0, 4.0)
                return EmbodiedEnvironmentTransition(
                    observation=torch.tensor([float(self.step_index + 1)]),
                    execution_result=result,
                    reward=torch.tensor([
                        rewards[min(index, len(rewards) - 1)]]),
                    terminated=torch.tensor([final and terminated]),
                    truncated=torch.tensor([final and truncated]))

        policy = FakePolicy()
        trainer = AutonomousInteractionTrainer(
            FakeEnvironment(),
            policy,
            actor,
            robot,
            torch.optim.Adam(actor.parameters(), lr=1e-3),
            torch.optim.Adam([policy.value_scale], lr=1e-3),
            config)
        return trainer, actor, policy

    def TestAutonomousInteractionPpo(self) -> bool:
        try:
            torch.manual_seed(71)
            trainer, actor, policy = self.BuildAutonomousHarness()
            rollout_state = actor.ExportEligibilityState()
            records = trainer.CollectRollout()
            expected_advantage = torch.tensor([2.5295, 3.7, 3.625])
            rollout_ok = bool(
                len(records) == 3
                and [record.policy_path for record in records] == [
                    POLICY_PATH_FULL,
                    POLICY_PATH_FAST,
                    POLICY_PATH_DETAIL]
                and [record.duration for record in records] == [2, 1, 1]
                and torch.allclose(torch.stack([
                    record.reward for record in records]),
                    torch.tensor([2.0, 2.5, 4.0]))
                and torch.allclose(torch.stack([
                    record.advantage for record in records]),
                    expected_advantage,
                    atol=1e-5,
                    rtol=1e-5)
                and records[-1].truncated
                and not records[-1].terminated
                and torch.allclose(records[-1].next_value, torch.tensor(1.25))
                and records[1].help_requested
                and records[1].help_accepted
                and policy.planner_enabled
                and not actor.EligibilityFrozen()
                and all(torch.equal(
                    rollout_state[name],
                    actor.ExportEligibilityState()[name])
                    for name in rollout_state))
            state_before = actor.ExportEligibilityState()
            actor_before = torch.cat([
                parameter.detach().flatten()
                for parameter in actor.parameters()]).clone()
            value_before = policy.value_scale.detach().clone()
            metrics = trainer.OptimizePpo(records)
            state_after = actor.ExportEligibilityState()
            actor_after = torch.cat([
                parameter.detach().flatten()
                for parameter in actor.parameters()])
            optimization_ok = bool(
                metrics["sample_count"] == 3.0
                and metrics["full_count"] == 2.0
                and metrics["fast_count"] == 2.0
                and metrics["detail_count"] == 2.0
                and metrics["help_acceptance"] == 1.0
                and not actor.EligibilityFrozen()
                and all(
                    torch.equal(state_before[name], state_after[name])
                    for name in state_before)
                and not torch.equal(actor_before, actor_after)
                and not torch.equal(value_before, policy.value_scale.detach()))
            return rollout_ok and optimization_ok
        except Exception as error:
            print(f"Manager autonomous PPO error: {error}")
            return False

    def TestAutonomousTerminationAndExclusion(self) -> bool:
        try:
            terminated_trainer, _, terminated_policy = self.BuildAutonomousHarness(
                rolloutSteps=1,
                terminated=True,
                truncated=False)
            terminated_records = terminated_trainer.CollectRollout()
            unknown_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=1,
                executionKnown=False)
            unknown_records = unknown_trainer.CollectRollout()
            override_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=1,
                controllerOverride=True)
            override_records = override_trainer.CollectRollout()
            planner_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=1,
                plannerOverride=True)
            planner_rejected = False
            try:
                planner_trainer.CollectRollout()
            except RuntimeError:
                planner_rejected = True
            return bool(
                len(terminated_records) == 1
                and terminated_records[0].terminated
                and not terminated_records[0].truncated
                and float(terminated_records[0].next_value.item()) == 0.0
                and terminated_policy.value_calls == 0
                and len(unknown_records) == 0
                and len(override_records) == 0
                and planner_rejected)
        except Exception as error:
            print(f"Manager autonomous termination error: {error}")
            return False

    def TestAutonomousExecutionCreditBoundaries(self) -> bool:
        try:
            interrupted_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=2,
                executionKnownSequence=(True, False))
            interrupted_records = interrupted_trainer.CollectRollout()
            partial_feedback_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=1,
                inactiveUnknown=True)
            partial_feedback_records = partial_feedback_trainer.CollectRollout()
            rejected_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=1,
                executionStatus=SlotExecutionStatus.REJECTED)
            rejected_records = rejected_trainer.CollectRollout()
            stopped_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=1,
                executionStatus=SlotExecutionStatus.STOPPED)
            stopped_records = stopped_trainer.CollectRollout()
            hard_stop_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=1,
                hardStop=True)
            hard_stop_records = hard_stop_trainer.CollectRollout()
            tolerance_trainer, _, tolerance_policy = (
                self.BuildAutonomousHarness(rolloutSteps=1))
            tolerance_trainer.robot.Reset()
            tolerance_observation = (
                tolerance_trainer.environment.MaterializeObservation(
                    tolerance_trainer.environment.Reset(),
                    tolerance_trainer.robot))
            tolerance_policy.Reset(tolerance_observation)
            tolerance_output = tolerance_policy.Step(tolerance_observation)
            tolerance_transition = tolerance_trainer.environment.Step(
                tolerance_output.action_request)
            tolerance_result = tolerance_transition.execution_result
            tolerant_values = tolerance_result.applied_target.values.clone()
            tolerant_values[:, 0] += 0.5 * float(
                tolerance_trainer.robot.ContractView
                .end_effector_target_tolerance[0])
            tolerant_target = PackedEndEffectorTarget(
                values=tolerant_values,
                active=tolerance_result.applied_target.active.clone(),
                contract_id=tolerance_result.applied_target.contract_id,
                model_signature=(
                    tolerance_result.applied_target.model_signature),
                target_version=(
                    tolerance_result.applied_target.target_version.clone()),
                timestamp=tolerance_result.applied_target.timestamp.clone())
            tolerant_result = ActionExecutionResult(
                request_id=tolerance_result.request_id.clone(),
                action_epoch=tolerance_result.action_epoch.clone(),
                applied_target=tolerant_target,
                execution_status=tolerance_result.execution_status.clone(),
                execution_known=tolerance_result.execution_known.clone(),
                hard_stop=tolerance_result.hard_stop.clone(),
                help_accepted=tolerance_result.help_accepted.clone(),
                timestamp=tolerance_result.timestamp.clone())
            tolerance_record = tolerance_trainer.CreateRecord(
                tolerance_output,
                tolerance_transition.execution_result,
                0,
                None)
            return bool(
                len(interrupted_records) == 1
                and interrupted_records[0].duration == 1
                and torch.allclose(
                    interrupted_records[0].reward,
                    torch.tensor(1.0))
                and torch.allclose(
                    interrupted_records[0].next_value,
                    torch.tensor(0.5))
                and len(partial_feedback_records) == 1
                and len(rejected_records) == 0
                and len(stopped_records) == 0
                and len(hard_stop_records) == 0
                and bool(tolerance_trainer.ExecutionMatchesRequest(
                    tolerance_output.action_request,
                    tolerant_result).item())
                and tolerance_trainer.RecordTargetMatches(
                    tolerance_record,
                    tolerant_result,
                    0))
        except Exception as error:
            print(f"Manager autonomous execution credit error: {error}")
            return False

    def TestAutonomousAppliedExecutionSegmentation(self) -> bool:
        try:
            def Collect(
                finalStatus: SlotExecutionStatus,
                known: bool = True,
                helpRequested: bool = False,
            ) -> List[AutonomousRolloutRecord]:
                trainer, _, _ = self.BuildAutonomousHarness(
                    rolloutSteps=3,
                    executionStatusSequence=(
                        SlotExecutionStatus.APPLIED,
                        SlotExecutionStatus.APPLIED,
                        finalStatus),
                    executionKnownSequence=(True, True, known),
                    helpRequestedSequence=(
                        False,
                        False,
                        helpRequested),
                    helpAcceptedSequence=(False, False, False))
                return trainer.CollectRollout()

            rejected = Collect(SlotExecutionStatus.REJECTED)
            held = Collect(SlotExecutionStatus.HELD)
            help_rejected = Collect(
                SlotExecutionStatus.REJECTED,
                helpRequested=True)
            modified = Collect(SlotExecutionStatus.MODIFIED)
            stopped = Collect(SlotExecutionStatus.STOPPED)
            unknown = Collect(
                SlotExecutionStatus.UNKNOWN,
                known=False)
            delayed_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=4,
                helpRequestedSequence=(False, False, True, True),
                helpAcceptedSequence=(False, False, False, True),
                actorCreditSequence=(True, False, True, True))
            delayed = delayed_trainer.CollectRollout()
            repeated_help_trainer, _, _ = self.BuildAutonomousHarness(
                rolloutSteps=2,
                helpRequestedSequence=(True, True),
                helpAcceptedSequence=(False, True),
                actorCreditSequence=(True, True))
            repeated_help = repeated_help_trainer.CollectRollout()
            continued = rejected + held
            disrupted = modified + stopped + unknown
            return bool(
                all(len(records) == 1 for records in (
                    rejected,
                    held,
                    modified,
                    stopped,
                    unknown))
                and all(record.duration == 3 for record in continued)
                and all(torch.allclose(
                    record.reward,
                    torch.tensor(2.75)) for record in continued)
                and all(record.applied_action_epoch == 1 for record in continued)
                and all(bool(record.applied_target_active[0].item()) for record in continued)
                and all(torch.equal(
                    record.applied_target_values,
                    torch.zeros_like(record.applied_target_values)) for record in continued)
                and len(help_rejected) == 2
                and help_rejected[0].duration == 2
                and torch.allclose(
                    help_rejected[0].reward,
                    torch.tensor(2.0))
                and not help_rejected[0].help_requested
                and help_rejected[0].trace_continues
                and help_rejected[1].duration == 1
                and torch.allclose(
                    help_rejected[1].reward,
                    torch.tensor(2.5))
                and help_rejected[1].help_requested
                and not help_rejected[1].help_accepted
                and help_rejected[1].help_pending
                and len(delayed) == 2
                and delayed[0].duration == 2
                and torch.allclose(delayed[0].reward, torch.tensor(2.0))
                and delayed[0].trace_continues
                and delayed[1].duration == 2
                and torch.allclose(delayed[1].reward, torch.tensor(4.25))
                and delayed[1].help_requested
                and delayed[1].help_accepted
                and not delayed[1].help_pending
                and len(repeated_help) == 1
                and repeated_help[0].duration == 2
                and torch.allclose(
                    repeated_help[0].reward,
                    torch.tensor(1.25))
                and repeated_help[0].help_requested
                and repeated_help[0].help_accepted
                and not repeated_help[0].help_pending
                and all(record.duration == 2 for record in disrupted)
                and all(torch.allclose(
                    record.reward,
                    torch.tensor(2.0)) for record in disrupted)
                and all(torch.allclose(
                    record.next_value,
                    torch.tensor(0.75)) for record in disrupted))
        except Exception as error:
            print(f"Manager applied execution segmentation error: {error}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "DeploymentConfigurationRouting": self.TestDeploymentConfigurationRouting(),
            "DeploymentManifestRoutesOneCompleteGeneration": self.TestDeploymentManifestRoutesOneCompleteGeneration(),
            "PauseStopImmediateExit": self.TestPauseStopImmediateExit(),
            "BackgroundExceptionStatus": self.TestBackgroundExceptionStatus(),
            "EvaluationRuntimeRestoredBeforeSave": self.TestEvaluationRuntimeRestoredBeforeSave(),
            "LoadCheckpointRestoresTopologyBeforeOptimizers": self.TestLoadCheckpointRestoresTopologyBeforeOptimizers(),
            "LoadBrainWeightsStrictModelStateAndSync": self.TestLoadBrainWeightsStrictModelStateAndSync(),
            "TrainStageIsolationContract": self.TestTrainStageIsolationContract(),
            "TemporalEnvelopeProjection": self.TestTemporalEnvelopeProjection(),
            "OcrTrainingArtifactFallback": self.TestOcrTrainingArtifactFallback(),
            "OcrCheckpointStrictContract": self.TestOcrCheckpointStrictContract(),
            "SequentialLoaderRejectsEmptyRecurrentSplit": self.TestSequentialLoaderRejectsEmptyRecurrentSplit(),
            "CppDecisionWireContract": self.TestCppDecisionWireContract(),
            "SingleFramePreprocessContract": self.TestSingleFramePreprocessContract(),
            "OfflinePreprocessStrictContract": self.TestOfflinePreprocessStrictContract(),
            "AutonomousInteractionPpo": self.TestAutonomousInteractionPpo(),
            "AutonomousTerminationAndExclusion": (
                self.TestAutonomousTerminationAndExclusion()),
            "AutonomousExecutionCreditBoundaries": (
                self.TestAutonomousExecutionCreditBoundaries()),
            "AutonomousAppliedExecutionSegmentation": (
                self.TestAutonomousAppliedExecutionSegmentation()),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nManager tests: {passed}/{len(results)} passed.")
        return results


class ContractInferenceAgent(Agent):
    def __init__(
        self,
        brain: BrainCore,
        brainBuildSpec: BrainBuildSpec,
        *,
        device: Union[str, torch.device],
        worldMemoryPath: Optional[str],
        memMemoryPath: Optional[str],
    ) -> None:
        if type(brainBuildSpec) is not BrainBuildSpec:
            raise TypeError("contract inference Agent requires BrainBuildSpec")
        self.brain_build_spec = brainBuildSpec
        super().__init__(
            brain,
            isTrain=False,
            device=device,
            worldMemoryPath=worldMemoryPath,
            memMemoryPath=memMemoryPath)

    def ValidateActRequest(
        self,
        request: ContractAgentActInput,
    ) -> int:
        if type(request) is not ContractAgentActInput:
            raise TypeError(
                "production Agent.Act requires ContractAgentActInput")
        if (
            not torch.is_tensor(request.frame)
            or request.frame.dim() != 4
            or int(request.frame.size(0)) < 1
        ):
            raise ValueError("contract inference frame must be a non-empty batch")
        batch_size = int(request.frame.size(0))
        if request.frame.device != self.device:
            raise ValueError("contract inference frame device does not match Agent")
        if (
            not torch.is_tensor(request.depth)
            or not torch.is_tensor(request.depth_valid)
            or request.depth.dim() != 4
            or tuple(request.depth_valid.shape) != tuple(request.depth.shape)
            or request.depth_valid.dtype != torch.bool
            or int(request.depth.size(0)) != batch_size
            or request.depth.device != self.device
            or request.depth_valid.device != self.device
        ):
            raise ValueError(
                "contract inference depth and validity must match the frame batch and device")
        if type(request.sample_actions) is not bool:
            raise TypeError("sample_actions must be a boolean")
        if type(request.deterministic_actor) is not bool:
            raise TypeError("deterministic_actor must be a boolean")
        return batch_size

    def Act(
        self,
        request: ContractAgentActInput,
    ) -> ContractAgentActOutput:
        self.ValidateActRequest(request)
        brain_step = ContractBrainStepInput(
            frame=request.frame,
            text_ext=request.text_ext,
            reward_ext=request.reward,
            done_flag=request.done,
            is_train=False,
            sample_actions=request.sample_actions,
            deterministic_actor=request.deterministic_actor,
            depth=request.depth,
            depth_valid=request.depth_valid,
            feedback_packet=request.feedback_packet,
            text_trust=request.text_trust)
        with torch.no_grad():
            step_out = self.brain.StepContract(brain_step)
        if type(step_out.decision) is not dict:
            raise TypeError("contract brain decision must be a dictionary")
        packedTarget = step_out.decision.get("packed_target")
        if type(packedTarget) is not PackedEndEffectorTarget:
            raise TypeError(
                "contract brain must return PackedEndEffectorTarget")
        self.CommitPendingWorldAutosave()
        return ContractAgentActOutput(
            action_request=step_out.action_request,
            cognitive_readout=step_out.cognitive_readout,
            packed_target=packedTarget,
            packed_temporal=step_out.decision.get("packed_temporal"),
            decision=step_out.decision,
            ocr=step_out.ocr,
            intention_texts=step_out.intention_texts)



class AgentHandle:
    def __init__(
        self,
        brainBuildSpec: Optional[BrainBuildSpec] = None,
        *,
        robot: Optional[Robot] = None,
        brainParameterPath: str = BasicParameters.MODULEPARAMETER_PATH,
        device: Optional[Union[str, torch.device]] = None,
        seqLen: int = BasicParameters.IMAGE_SEQ_LEN,
        usePlanner: bool = True,
        prioritizeExtStr: bool = True,
        saveModuleMessagerOutput: bool = True,):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.robot = robot if robot is not None else Robot.CreateDefault()
        self.brain_build_spec = (
            brainBuildSpec
            if brainBuildSpec is not None
            else BrainBuildSpec.Compile(
                ModuleDim.CognitiveProfile(),
                self.robot.ContractView))
        if type(self.brain_build_spec) is not BrainBuildSpec:
            raise TypeError("brainBuildSpec must be a BrainBuildSpec")
        contract_view = self.robot.ContractView
        if (
            self.brain_build_spec.contract_view.contract_id
            != contract_view.contract_id
            or self.brain_build_spec.contract_view.model_signature
            != contract_view.model_signature
        ):
            raise ValueError("BrainBuildSpec does not match the selected contract")
        projection = contract_view.perception_projection
        if projection is None:
            raise RuntimeError("robot contract has no perception calibration")

        parameter_path = str(brainParameterPath).strip()
        if parameter_path == "":
            raise ValueError("brainParameterPath must not be empty")

        (
            resolved_model_path,
            resolved_world_memory_path,
            resolved_memory_path,
        ) = ManagerFunction.ResolveDeploymentArtifactPaths(
            parameter_path,
            calibrationId=projection.calibration_id,
            brainBuildSpec=self.brain_build_spec)
        resolved_path = Path(resolved_model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"brain parameter file not found: {resolved_path}")

        self.brain = BrainCore(
            brainBuildSpec=self.brain_build_spec,
            device=self.device,
            seqLen=seqLen,
            prioritizeExtStr=prioritizeExtStr,
            plasticOnlineLearning=False,
            usePlanner=usePlanner,
            saveModuleMessagerOutput=saveModuleMessagerOutput,)

        self.agent = ContractInferenceAgent(
            self.brain,
            self.brain_build_spec,
            device=self.device,
            worldMemoryPath=resolved_world_memory_path,
            memMemoryPath=resolved_memory_path)

        self.agent.LoadBrainWeights(str(resolved_path))
        self.brain.eval()

    def ForwardStep(
        self,
        frame: torch.Tensor,
        *,
        textExt: Optional[List[Optional[str]]] = None,
        textTrust: Optional[List[str]] = None,
        reward: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        sampleActions: bool = True,
        deterministicActor: bool = False,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
    ) -> ContractAgentActOutput:
        act_output = self.agent.Act(ContractAgentActInput(
            frame=frame,
            text_ext=textExt,
            reward=reward,
            done=done,
            sample_actions=sampleActions,
            deterministic_actor=deterministicActor,
            depth=depth,
            depth_valid=depthValid,
            feedback_packet=feedbackPacket,
            text_trust=textTrust))
        return act_output

    def EncodeEmbodimentFeedback(
        self,
        feedbackPacket: BrainFeedbackPacket,
        *,
        batchSize: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.agent.EncodeEmbodimentFeedback(
            feedbackPacket,
            batchSize=batchSize)
