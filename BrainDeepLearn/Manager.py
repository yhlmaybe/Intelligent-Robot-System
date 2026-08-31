from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Optional, Sequence, Union
from pathlib import Path
from dataclasses import dataclass
import threading
import random
import time
import json
import math

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
from DecisionModule import TestDecisionMTool
from WorldModule import ContractWorldEmbodimentAdapter
from ValueEstimationModule import  TestValueEstimationMTool
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
    ContractAgentActInput,
    ContractAgentActOutput,
    ContractBrainStepInput,
    ContractOfflineBatch,
    ContractOfflineSample,
    DECISION_REQUEST_PROVENANCE_FIELDS,
    SENSOR_PACKET_WIRE_FIELDS,
    SENSOR_PACKET_WIRE_SCHEMA_VERSION,
    TEXT_TRUST_OCR_OBSERVED,
    TEXT_TRUST_OPERATOR_COMMAND,
    TEXT_TRUST_UNSAFE_EXTERNAL)
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    PackedEndEffectorTarget,
    Robot)


TRAIN_CHECKPOINT_SCHEMA_VERSION = BRAIN_RUNTIME_SCHEMA_VERSION
OCR_CHECKPOINT_SCHEMA_VERSION = 16
OCR_COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS = frozenset({15, 16})

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
        return True

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
    ) -> str:
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
        self.agent_handle.agent.BindWorldMemoryContext(
            requestProvenance["world_frame_id"],
            batchSize=batch_size)
        act_output = self.agent_handle.ForwardStep(
            converted["frames"],
            textExt=textExt,
            textTrust=textTrust,
            reward=converted["rewards"],
            done=converted["dones"],
            sampleActions=sampleActions,
            deterministicActor=deterministicActor,
            depth=converted["depths"],
            depthValid=converted["depth_valid"],
            feedbackPacket=feedback_packet)
        target = act_output.packed_target
        targetPayload = self.robot.DecodeTarget(target)
        result = json.dumps({
            "schema_version": 2,
            "request_provenance": dict(requestProvenance),
            "end_effector_target": targetPayload,
            "intention_texts": list(act_output.intention_texts),
        }, ensure_ascii=False, allow_nan=False)
        packedTemporal = act_output.packed_temporal
        if packedTemporal is None or not hasattr(
            packedTemporal,
            "candidate_selected",
        ):
            raise TypeError(
                "contract Agent output requires temporal dispatch selection")
        self.robot.CommitDispatchedTarget(
            target,
            packedTemporal.candidate_selected)
        if converted["dones"] is not None and bool(
            converted["dones"].gt(0.5).any().item()
        ):
            self.robot.Reset()
        return result

    def AgentHandleForwardJson(
        self,
        reward: Optional[float],
        done: Optional[float],
        sensorPacketJson: str,
        feedbackPayloadJson: str,
    ) -> str:
        sensor_packet = json.loads(sensorPacketJson)
        feedback_payload = json.loads(feedbackPayloadJson)
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
        if self.active_sensor_stream_id is None:
            if sequence_index != 0:
                raise ValueError("a sensor stream must begin at sequence_index 0")
        else:
            if sensor_packet["stream_id"] != self.active_sensor_stream_id:
                raise ValueError(
                    "sensor stream_id changed; initialize a new AgentHandle")
            if sequence_index != self.last_sensor_sequence_index + 1:
                raise ValueError(
                    "sensor sequence_index must increase by exactly one")
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
        result = self.ForwardContractBatch(
            sensor_packet["rgb"],
            reward,
            done,
            textExt=text_ext,
            textTrust=text_trust,
            sampleActions=sensor_packet["sample_actions"],
            deterministicActor=sensor_packet["deterministic_actor"],
            depthBitmap=sensor_packet["depth"],
            depthValid=sensor_packet["depth_valid"],
            feedbackPayload=feedback_payload,
            requestProvenance={
                "stream_id": sensor_packet["stream_id"],
                "sequence_index": sequence_index,
                "frame_id": sensor_packet["frame_id"],
                "calibration_id": sensor_packet["calibration_id"],
                "world_frame_id": world_context_id,
                "description_id": (
                    self.brain_build_spec.contract_view.description_id),
                "model_contract_id": self.brain_build_spec.model_signature,
                "adapter_id": self.brain_build_spec.contract_view.adapter_id,
            })
        self.active_sensor_stream_id = sensor_packet["stream_id"]
        self.active_world_frame_id = world_context_id
        self.last_sensor_sequence_index = sequence_index
        return result

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
            or payload["schema_version"] not in OCR_COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS
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
            or payload["schema_version"] not in OCR_COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS
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
            or ckpt["schema_version"] not in OCR_COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS
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
            or ckpt["schema_version"] not in OCR_COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS
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
            engine = OCREngineExtractor().to(self.device)
            has_resume_ckpt = resume and Path(ckptPath).exists()

            self.ConfigureOCRTrainingTargets(
                engine,
                trainDetection=trainDetection,
                trainRecognition=trainRecognition,)

            trainable_params = [p for p in engine.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise RuntimeError("no trainable OCR parameters selected")
            optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-2)

            start_epoch = 0
            best_val = float("inf")
            processed_sample_count_total = 0
            train_ds = val_ds = test_ds = None

            testSplit = 0.1
            if has_resume_ckpt:
                start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds = self.LoadOCRCheckpoint(
                    engine,
                    optimizer,
                    ds,
                    ckptPath,
                    trainDetection=trainDetection,
                    trainRecognition=trainRecognition)
            else:
                if recognizerInitPath and Path(recognizerInitPath).exists():
                    self.LoadRecognizerWeightsIntoEngine(engine, recognizerInitPath)
                elif not trainRecognition:
                    if not Path(outPath).is_file():
                        raise FileNotFoundError(
                            "detect-only OCR training requires the current OCR "
                            "parameter artifact or recognizerInitPath")
                    self.LoadOCRWeightsIntoEngine(engine, outPath)
                    
            parameters_overridden = self.ApplyParameterOverrideAfterResume(
                enabled=overrideCheckpointWithModuleParams,
                parameterPath=outPath,
                loadFn=lambda path: self.LoadOCRWeightsIntoEngine(engine, path),
                logPrefix="TrainOCR")
            if parameters_overridden:
                optimizer.state.clear()
                best_val = float("inf")
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
            engine = OCREngineExtractor().to(self.device)
            self.ConfigureOCRTrainingTargets(
                engine,
                trainDetection=False,
                trainRecognition=True,)

            trainable_params = [p for p in engine.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise RuntimeError("no trainable OCR recognizer parameters selected")
            optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-2)

            start_epoch = 0
            best_val = float("inf")
            processed_sample_count_total = 0
            train_ds = val_ds = test_ds = None

            testSplit = 0.1
            if resume and Path(ckptPath).exists():
                start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds = self.LoadOCRRecognizerCheckpoint(
                    engine, optimizer, ds, ckptPath)
                
            parameters_overridden = self.ApplyParameterOverrideAfterResume(
                enabled=overrideCheckpointWithModuleParams,
                parameterPath=outPath,
                loadFn=lambda path: self.LoadRecognizerWeightsIntoEngine(engine, path),
                logPrefix="TrainOCRRec")
            if parameters_overridden:
                optimizer.state.clear()
                best_val = float("inf")
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
        world_batch_size, _ = world._ValidateMemoryPayload(ckpt["world_memory"])
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
                def _ValidateMemoryPayload(self, state):
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
            manager.LoadOCRWeightsIntoEngine(
                engine,
                serialized(legacy_ocr_parameters))
            manager.LoadRecognizerWeightsIntoEngine(
                engine,
                serialized(legacy_recognizer_parameters))
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
            legacy_restored = manager.LoadOCRCheckpoint(
                engine,
                optimizer,
                dataset,
                serialized(legacy_checkpoint),
                trainDetection=True,
                trainRecognition=True)

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
            legacy_recognizer_restored = manager.LoadOCRRecognizerCheckpoint(
                engine,
                recognizer_optimizer,
                dataset,
                serialized(legacy_recognizer_checkpoint))

            ok = (
                restored[0:3] == (2, 0.25, 12)
                and [list(split.indices) for split in restored[3:]]
                == [[0, 1], [2, 3], [4, 5]]
                and recognizer_restored[0:3] == (3, 0.1, 18)
                and legacy_restored[0:3] == (2, 0.25, 12)
                and legacy_recognizer_restored[0:3] == (3, 0.1, 18)
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
                    ("recognizer", True),
                    ("ocr", True),
                    ("ocr", True),
                    ("recognizer", True),
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
        try:
            manager = object.__new__(ManagerFunction)
            manager.device = torch.device("cpu")
            manager.robot = Robot.CreateDefault()
            manager.brain_build_spec = BrainBuildSpec.Compile(
                ModuleDim.CognitiveProfile(),
                manager.robot.ContractView)
            manager.perception_calibration_id = "test-calibration"
            manager.active_sensor_stream_id = None
            manager.active_world_frame_id = None
            manager.last_sensor_sequence_index = None
            captured: Dict[str, Any] = {}

            def CaptureForward(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return "ok"

            manager.ForwardContractBatch = CaptureForward
            feedback_payload = dict(
                manager.robot.BuildNeutralFeedbackPayload(
                    0.0,
                    manager.device))
            sensor_packet = {
                "schema_version": SENSOR_PACKET_WIRE_SCHEMA_VERSION,
                "stream_id": "stream-1",
                "sequence_index": 0,
                "frame_id": "frame-1",
                "calibration_id": "test-calibration",
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
                json.dumps(feedback_payload))

            duplicate_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(sensor_packet),
                    json.dumps(feedback_payload))
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
                    json.dumps(feedback_payload))
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
                    json.dumps(feedback_payload))
            except ValueError:
                calibration_mismatch_rejected = True

            ok = (
                result == "ok"
                and captured["args"] == (sensor_packet["rgb"], 0.5, 0.0)
                and captured["kwargs"]["feedbackPayload"] == feedback_payload
                and captured["kwargs"]["textExt"] == sensor_packet["text_ext"]
                and captured["kwargs"]["textTrust"] == sensor_packet["text_trust"]
                and captured["kwargs"]["sampleActions"] is False
                and captured["kwargs"]["deterministicActor"] is True
                and captured["kwargs"]["depthBitmap"] == sensor_packet["depth"]
                and captured["kwargs"]["depthValid"] == sensor_packet["depth_valid"]
                and captured["kwargs"]["requestProvenance"] == {
                    "stream_id": "stream-1",
                    "sequence_index": 0,
                    "frame_id": "frame-1",
                    "calibration_id": "test-calibration",
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
                and manager.active_sensor_stream_id == "stream-1"
                and manager.active_world_frame_id == "sensor_stream:stream-1"
                and manager.last_sensor_sequence_index == 0)
            print(f"Manager C++ decision wire contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager C++ decision wire contract error: {e}")
            return False

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
            "OcrCheckpointStrictContract": self.TestOcrCheckpointStrictContract(),
            "SequentialLoaderRejectsEmptyRecurrentSplit": self.TestSequentialLoaderRejectsEmptyRecurrentSplit(),
            "CppDecisionWireContract": self.TestCppDecisionWireContract(),
            "SingleFramePreprocessContract": self.TestSingleFramePreprocessContract(),
            "OfflinePreprocessStrictContract": self.TestOfflinePreprocessStrictContract(),}
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
