from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
from types import SimpleNamespace
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
from WorldModule import  TestWorldMTool
from ValueEstimationModule import  TestValueEstimationMTool
from ConsciousnessModule import TestConsciousMTool
from IntentionModule import TestIntentionMTool
from OCRModule import TestOCRMTool, OCREngineExtractor, IouXyxy
from DataPreprocess import DataPreprocessor, DataResizeMeta, OfflineGameDataset, OfflineOCRDataset, OfflineOCRRecognitionDataset
from AGICore import (
    Agent,
    BRAIN_RUNTIME_BUFFER_FIELDS,
    BRAIN_RUNTIME_SCHEMA_VERSION,
    BrainCore,
    ExportDeploymentModelState,
    ExportBrainModelState,
    IsWorldRuntimeStateKey,
    LoadBrainModelState,
    LoadDeploymentModelState,
    TestAGICoreMTool)
from Config import BasicParameters
from CoreTypes import (
    AgentActInput,
    CameraCalibration,
    ExpectedOfflineSensorManifest,
    ExpectedRobotStateWireMetadata,
    ROBOT_STATE_FIELDS,
    ROBOT_STATE_MASK_FIELDS,
    ROBOT_STATE_SCALAR_MASK_FIELDS,
    ROBOT_STATE_WIRE_SCHEMA_VERSION,
    SENSOR_PACKET_WIRE_FIELDS,
    SENSOR_PACKET_WIRE_SCHEMA_VERSION,
    RobotState,
    ValidateOfflineSensorManifest,
    ValidateRobotStateWirePacket,
    ValidateRobotTensorContract,
    ValidateRobotObserverCalibration,
    TEXT_TRUST_OCR_OBSERVED,
    TEXT_TRUST_OPERATOR_COMMAND,
    TEXT_TRUST_UNSAFE_EXTERNAL)
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import (
    CompiledRobotMorphology,
    RobotMorphologyModule)


TRAIN_CHECKPOINT_SCHEMA_VERSION = BRAIN_RUNTIME_SCHEMA_VERSION
OCR_CHECKPOINT_SCHEMA_VERSION = 16
OCR_COMPATIBLE_CHECKPOINT_SCHEMA_VERSIONS = frozenset({15, 16})

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
    """Splits one chronological frame stream into batchSize contiguous segments so
    each batch slot stays temporally continuous across iterations: slot j yields
    frames j*steps .. j*steps+steps-1 in order, matching the recurrent state's
    batch-as-parallel-streams semantics."""

    def __init__(self, dataset: Dataset, *, batchSize: int):
        self.dataset = dataset
        self.batch_size = int(batchSize)
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
            yield torch.utils.data.default_collate(items)


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
        robotUrdfPath: str = BasicParameters.ROBOT_URDF_PATH,
        robotSrdfPath: str = BasicParameters.ROBOT_SRDF_PATH,
        robotSemanticPath: Optional[str] = BasicParameters.ROBOT_SEMANTIC_PATH,
        robotMorphology: Optional[CompiledRobotMorphology] = None,):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.robot_morphology = (
            robotMorphology
            if robotMorphology is not None
            else self.LoadRobotMorphology(
                robotUrdfPath,
                robotSrdfPath,
                robotSemanticPath))
        ValidateRobotTensorContract(self.robot_morphology)
        self.controller = ModuleController()

        self.br_thread: Optional[threading.Thread] = None
        self.message_thread: Optional[threading.Thread] = None
        self.is_begin = False
        self.overrideCheckpointWithModuleParams = (
            self.DEFAULT_OVERRIDE_CHECKPOINT_WITH_MODULE_PARAMS)
        self.agent_handle: Optional[AgentHandle] = None
        self.camera_calibration_id: Optional[str] = None
        self.active_sensor_stream_id: Optional[str] = None
        self.active_world_frame_id: Optional[str] = None
        self.last_sensor_sequence_index: Optional[int] = None
        self.json_queue = None

        self.test = {
            "perception": TestPerceptionMTool(),
            "attention": TestAttentionMTool(),
            "memory": TestMemoryMTool(),
            "decision": TestDecisionMTool(),
            "world": TestWorldMTool(),
            "value": TestValueEstimationMTool(),
            "consciousness": TestConsciousMTool(),
            "OCR": TestOCRMTool(),
            "intention": TestIntentionMTool(),
            "AGICore": TestAGICoreMTool(),
            "manager": TestManagerMTool(),}

    @staticmethod
    def LoadRobotMorphology(
        urdfPath: str = BasicParameters.ROBOT_URDF_PATH,
        srdfPath: str = BasicParameters.ROBOT_SRDF_PATH,
        semanticPath: Optional[str] = BasicParameters.ROBOT_SEMANTIC_PATH,
        observerFrameName: str = BasicParameters.ROBOT_OBSERVER_FRAME_NAME,
        observerCalibrationId: str = BasicParameters.ROBOT_OBSERVER_CALIBRATION_ID,
    ) -> CompiledRobotMorphology:
        overlay = None
        if semanticPath is not None and str(semanticPath).strip():
            overlay = RobotMorphologyModule._ReadJson(str(semanticPath))
            observer_configured = (
                overlay.get("observer") is not None
                or overlay.get("observer_endpoint") is not None)
            if observer_configured:
                expected = {
                    "observer_frame_name": observerFrameName,
                    "observer_calibration_id": observerCalibrationId,
                }
                for name, value in expected.items():
                    if type(value) is not str or not value:
                        raise ValueError(
                            f"configured robot {name} must be non-empty")
                    if overlay.get(name, value) != value:
                        raise ValueError(
                            f"robot semantic {name} does not match configuration")
                    overlay[name] = value
        return RobotMorphologyModule().FromMoveIt(
            urdfPath,
            srdfPath,
            overlay)

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
        """Run evaluation without changing the following training trajectory."""
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
                        # Assign directly so intentionally mixed train/eval subtrees are preserved.
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
        
    @staticmethod
    def LoadCameraCalibration(
        calibrationPath: str = BasicParameters.CAMERA_CALIBRATION_PATH,
        ) -> CameraCalibration:
        """Load K once for the fixed, already rectified 512x512 RGB-D lattice."""
        calibration = json.loads(Path(calibrationPath).read_text(encoding="utf-8"))
        expected_fields = {
            "version",
            "calibration_id",
            "camera_name",
            "sensor_type",
            "rgb_encoding",
            "depth_unit",
            "depth_representation",
            "rgb_depth_alignment",
            "rectification",
            "synchronization",
            "coordinate_frame",
            "pixel_convention",
            "image",
            "camera_intrinsics",
        }
        if type(calibration) is not dict or set(calibration) != expected_fields:
            raise ValueError("camera calibration fields do not match the current schema")
        if calibration["version"] != "3.0":
            raise ValueError("camera calibration version must be 3.0")
        if (
            type(calibration["calibration_id"]) is not str
            or not calibration["calibration_id"]
        ):
            raise ValueError("camera calibration_id must be a non-empty string")
        expected_contract = {
            "sensor_type": "rgbd",
            "rgb_encoding": "rgb8",
            "depth_unit": "meter",
            "depth_representation": "optical_axis_z",
            "rgb_depth_alignment": "registered_to_rgb",
            "rectification": "rectified",
            "synchronization": "synchronized_exposure",
        }
        for name, expected in expected_contract.items():
            if calibration[name] != expected:
                raise ValueError(f"camera calibration {name} must be {expected!r}")
        coordinate_frame = calibration["coordinate_frame"]
        if (
            type(coordinate_frame) is not dict
            or set(coordinate_frame) != {
                "camera_frame",
                "handedness",
                "x_axis_positive",
                "y_axis_positive",
                "z_axis_positive",
            }
            or type(coordinate_frame["camera_frame"]) is not str
            or not coordinate_frame["camera_frame"]
            or coordinate_frame["handedness"] != "right_handed"
            or coordinate_frame["x_axis_positive"] != "right"
            or coordinate_frame["y_axis_positive"] != "down"
            or coordinate_frame["z_axis_positive"] != "forward"
        ):
            raise ValueError("camera calibration optical-frame convention is invalid")
        if calibration["pixel_convention"] != {
            "pixel_centers": "integer_coordinates",
            "resampling": "align_corners_false",
        }:
            raise ValueError("camera calibration pixel convention is invalid")

        image = calibration["image"]
        if type(image) is not dict or set(image) != {"width", "height"}:
            raise ValueError("camera calibration image fields must be width and height")
        if image != {
            "width": BasicParameters.IMAGE_SIZE,
            "height": BasicParameters.IMAGE_SIZE,
        }:
            raise ValueError(
                f"camera calibration must use the fixed "
                f"{BasicParameters.IMAGE_SIZE}x{BasicParameters.IMAGE_SIZE} lattice")

        intrinsics = calibration["camera_intrinsics"]
        if (
            type(intrinsics) is not dict
            or set(intrinsics) != {"model", "fx", "fy", "cx", "cy", "skew"}
            or intrinsics["model"] != "pinhole"
        ):
            raise ValueError("camera intrinsics do not match the pinhole schema")
        intrinsic_names = ("fx", "fy", "cx", "cy", "skew")
        if any(type(intrinsics[name]) not in (int, float) for name in intrinsic_names):
            raise TypeError("camera intrinsics must be JSON numbers")
        values = [float(intrinsics[name]) for name in intrinsic_names]
        if not np.isfinite(values).all() or min(values[0], values[1]) <= 0.0:
            raise ValueError("camera intrinsics must be finite with positive focal lengths")
        camera_intrinsics = torch.tensor([
            [values[0], values[4], values[2]],
            [0.0, values[1], values[3]],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        return CameraCalibration(
            calibration_id=calibration["calibration_id"],
            frame_name=calibration["coordinate_frame"]["camera_frame"],
            intrinsics=camera_intrinsics)

    @staticmethod
    def TensorizeRobotState(
        state: Dict[str, Any],
        device: torch.device,
        *,
        batched: bool,
        robotContract: Any,) -> RobotState:
        ValidateRobotTensorContract(robotContract)
        if type(state) is not dict or set(state) != set(ROBOT_STATE_FIELDS):
            raise ValueError(f"RobotState fields must be exactly {sorted(ROBOT_STATE_FIELDS)}")

        tensors: Dict[str, torch.Tensor] = {}
        for name in ROBOT_STATE_FIELDS:
            try:
                value = torch.as_tensor(state[name])
            except Exception as error:
                raise ValueError(f"RobotState {name} has an invalid value") from error
            if (
                name in ROBOT_STATE_MASK_FIELDS
                or name in ROBOT_STATE_SCALAR_MASK_FIELDS
            ):
                if value.dtype != torch.bool:
                    raise TypeError(f"RobotState {name} must contain booleans")
                value = value.to(device=device)
            elif value.dtype == torch.bool or not (
                value.is_floating_point()
                or value.dtype in (
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                )
            ):
                raise TypeError(f"RobotState {name} must contain real numbers")
            elif not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"RobotState {name} must contain only finite values")
            elif name == "executed_action_id":
                if bool((value < 0).any().item()) or bool(
                    (value != value.round()).any().item()
                ):
                    raise ValueError(
                        "RobotState executed_action_id must contain non-negative integers")
                value = value.to(device=device, dtype=torch.long)
            else:
                value = value.to(device=device, dtype=torch.float32)
            tensors[name] = value

        joint_dof_count = int(robotContract.joint_dof_count)
        node_count = int(robotContract.node_count)
        endpoint_count = int(robotContract.endpoint_count)
        joint_shape = (
            (None, joint_dof_count)
            if batched else
            (joint_dof_count,))
        endpoint_shape = (
            (None, endpoint_count, ModuleDim.DecisionEndpointPoseDim)
            if batched else
            (endpoint_count, ModuleDim.DecisionEndpointPoseDim))
        endpoint_mask_shape = (
            (None, endpoint_count)
            if batched else
            (endpoint_count,))
        node_pose_shape = (None, node_count, 7) if batched else (node_count, 7)
        node_twist_shape = (None, node_count, 6) if batched else (node_count, 6)
        node_mask_shape = (None, node_count) if batched else (node_count,)
        observer_pose_shape = (None, 7) if batched else (7,)
        base_orientation_shape = (None, 4) if batched else (4,)
        gravity_shape = (None, 3) if batched else (3,)
        actual_joint_shape = tuple(tensors["joint_position"].shape)
        actual_node_pose_shape = tuple(tensors["node_pose_world"].shape)
        actual_node_twist_shape = tuple(tensors["node_twist_world"].shape)
        actual_endpoint_shape = tuple(tensors["endpoint_pose"].shape)
        actual_planner_shape = tuple(tensors["planner_expected_endpoint_pose"].shape)
        actual_observer_pose_shape = tuple(
            tensors["observer_pose_world"].shape)
        actual_base_orientation_shape = tuple(
            tensors["base_orientation_world"].shape)
        actual_gravity_shape = tuple(tensors["gravity_direction_world"].shape)
        for name in (
            "joint_position",
            "joint_velocity",
            "joint_effort",
            "joint_observed",
            "joint_healthy",
            "joint_controllable",
        ):
            shape = tuple(tensors[name].shape)
            if (
                len(shape) != len(joint_shape)
                or any(
                    expected is not None and actual != expected
                    for actual, expected in zip(shape, joint_shape))
            ):
                raise ValueError(
                    f"RobotState {name} does not match external joint variables")
        for name, expected_shape in (
            ("node_pose_world", node_pose_shape),
            ("node_twist_world", node_twist_shape),
        ):
            shape = tuple(tensors[name].shape)
            if (
                len(shape) != len(expected_shape)
                or any(
                    expected is not None and actual != expected
                    for actual, expected in zip(shape, expected_shape))
            ):
                raise ValueError(
                    f"RobotState {name} does not match external nodes")
        for name in ("node_observed", "node_healthy"):
            shape = tuple(tensors[name].shape)
            if (
                len(shape) != len(node_mask_shape)
                or any(
                    expected is not None and actual != expected
                    for actual, expected in zip(shape, node_mask_shape))
            ):
                raise ValueError(
                    f"RobotState {name} does not match external nodes")
        if (
            len(actual_endpoint_shape) != len(endpoint_shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(actual_endpoint_shape, endpoint_shape))
        ):
            raise ValueError(
                "RobotState endpoint_pose does not match external endpoints")
        if (
            len(actual_planner_shape) != len(endpoint_shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(actual_planner_shape, endpoint_shape))
        ):
            raise ValueError(
                "RobotState planner_expected_endpoint_pose does not match "
                "external endpoints")
        for name in (
            "endpoint_observed",
            "endpoint_healthy",
            "endpoint_controllable",
        ):
            shape = tuple(tensors[name].shape)
            if (
                len(shape) != len(endpoint_mask_shape)
                or any(
                    expected is not None and actual != expected
                    for actual, expected in zip(shape, endpoint_mask_shape))
            ):
                raise ValueError(
                    f"RobotState {name} does not match external endpoints")
        if (
            len(actual_observer_pose_shape) != len(observer_pose_shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(
                    actual_observer_pose_shape,
                    observer_pose_shape))
        ):
            raise ValueError(
                "RobotState observer_pose_world must have shape [B, 7] or [7]")
        if (
            len(actual_base_orientation_shape) != len(base_orientation_shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(
                    actual_base_orientation_shape,
                    base_orientation_shape))
        ):
            raise ValueError(
                "RobotState base_orientation_world must have shape [B, 4] or [4]")
        if (
            len(actual_gravity_shape) != len(gravity_shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(actual_gravity_shape, gravity_shape))
        ):
            raise ValueError(
                "RobotState gravity_direction_world must have shape [B, 3] or [3]")

        batch_size = actual_endpoint_shape[0] if batched else None
        scalar_shape = (batch_size,) if batched else ()
        if any(
            tuple(tensors[name].shape) != scalar_shape
            for name in ROBOT_STATE_FIELDS
            if name not in (
                "joint_position",
                "joint_velocity",
                "joint_effort",
                "joint_observed",
                "joint_healthy",
                "joint_controllable",
                "node_pose_world",
                "node_twist_world",
                "node_observed",
                "node_healthy",
                "endpoint_pose",
                "endpoint_observed",
                "endpoint_healthy",
                "endpoint_controllable",
                "observer_pose_world",
                "base_orientation_world",
                "gravity_direction_world",
                "planner_expected_endpoint_pose")
        ):
            raise ValueError(f"RobotState scalar fields must have shape {scalar_shape}")
        if batched and any(
            shape[0] != batch_size
            for shape in (
                actual_joint_shape,
                actual_node_pose_shape,
                actual_node_twist_shape,
                actual_planner_shape,
                actual_observer_pose_shape,
                actual_base_orientation_shape,
                actual_gravity_shape,
                tuple(tensors["joint_observed"].shape),
                tuple(tensors["node_observed"].shape),
                tuple(tensors["endpoint_observed"].shape))
        ):
            raise ValueError("RobotState fields must have one batch size")

        node_observed = tensors["node_observed"]
        if bool((tensors["node_healthy"] & ~node_observed).any().item()):
            raise ValueError("RobotState node_healthy requires node_observed")
        node_pose = tensors["node_pose_world"]
        node_identity = node_pose.new_zeros(node_pose.shape)
        node_identity[..., 6] = 1.0
        if (
            bool((~node_observed).any().item())
            and not torch.allclose(
                node_pose[~node_observed],
                node_identity[~node_observed],
                rtol=0.0,
                atol=1e-6)
        ):
            raise ValueError("RobotState unavailable node poses must be identity")
        if (
            bool((~node_observed).any().item())
            and bool((tensors["node_twist_world"][~node_observed]
                != 0.0).any().item())
        ):
            raise ValueError("RobotState unavailable node twists must be zero")
        node_quaternion_norm = node_pose[..., 3:7].norm(dim=-1)
        if not torch.allclose(
            node_quaternion_norm[node_observed],
            torch.ones_like(node_quaternion_norm[node_observed]),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError("RobotState node pose quaternions must have unit length")

        observer_pose = tensors["observer_pose_world"].reshape(-1, 7)
        observer_pose_valid = tensors["observer_pose_valid"].reshape(-1)
        observer_identity = observer_pose.new_zeros(observer_pose.shape)
        observer_identity[..., 6] = 1.0
        if (
            not robotContract.observer_valid
            and bool(observer_pose_valid.any().item())
        ):
            raise ValueError(
                "RobotState observer pose cannot be valid without an observer")
        observer_unavailable = ~observer_pose_valid
        if (
            bool(observer_unavailable.any().item())
            and not torch.allclose(
                observer_pose[observer_unavailable],
                observer_identity[observer_unavailable],
                rtol=0.0,
                atol=1e-6)
        ):
            raise ValueError(
                "RobotState unavailable observer pose must be identity")
        observer_quaternion_norm = observer_pose[..., 3:7].norm(dim=-1)
        if not torch.allclose(
            observer_quaternion_norm[observer_pose_valid],
            torch.ones_like(observer_quaternion_norm[observer_pose_valid]),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                "RobotState observer pose quaternion must have unit length")
        if (
            robotContract.observer_valid
            and robotContract.observer_attachment_kind == "link"
            and robotContract.observer_frame_name
            == robotContract.observer_attachment_name
        ):
            attachment_pose = node_pose[..., robotContract.observer_node_index, :]
            attachment_valid = node_observed[
                ..., robotContract.observer_node_index]
            comparable = observer_pose_valid & attachment_valid.reshape(-1)
            if (
                bool(comparable.any().item())
                and not torch.allclose(
                    observer_pose[comparable],
                    attachment_pose.reshape(-1, 7)[comparable],
                    rtol=0.0,
                    atol=1e-5)
            ):
                raise ValueError(
                    "RobotState observer and attachment node poses are inconsistent")

        joint_observed = tensors["joint_observed"]
        if bool((tensors["joint_healthy"] & ~joint_observed).any().item()):
            raise ValueError("RobotState joint_healthy requires joint_observed")
        for name in ("joint_position", "joint_velocity", "joint_effort"):
            if bool((tensors[name][~joint_observed] != 0.0).any().item()):
                raise ValueError(
                    f"RobotState unobserved {name} values must be zero")
        static_joint_controllable = robotContract.joint_variable_commandable.to(
            device=device)
        if batched:
            static_joint_controllable = static_joint_controllable.unsqueeze(0)
        if bool((tensors["joint_controllable"] &
            ~static_joint_controllable).any().item()):
            raise ValueError(
                "RobotState joint_controllable activates a passive joint")
        endpoint_observed = tensors["endpoint_observed"]
        if bool((tensors["endpoint_healthy"] & ~endpoint_observed).any().item()):
            raise ValueError(
                "RobotState endpoint_healthy requires endpoint_observed")
        endpoint_identity = tensors["endpoint_pose"].new_zeros(
            tensors["endpoint_pose"].shape)
        endpoint_identity[..., 6] = 1.0
        if (
            bool((~endpoint_observed).any().item())
            and not torch.allclose(
                tensors["endpoint_pose"][~endpoint_observed],
                endpoint_identity[~endpoint_observed],
                rtol=0.0,
                atol=1e-6)
        ):
            raise ValueError(
                "RobotState unobserved endpoint poses must be identity")
        static_endpoint_controllable = robotContract.endpoint_task_mask.any(
            dim=-1).to(device=device)
        if batched:
            static_endpoint_controllable = static_endpoint_controllable.unsqueeze(0)
        if bool((tensors["endpoint_controllable"] &
            ~static_endpoint_controllable).any().item()):
            raise ValueError(
                "RobotState endpoint_controllable lacks a task contract")

        for name in (
            "planner_progress",
        ):
            value = tensors[name]
            if bool(((value < 0.0) | (value > 1.0)).any().item()):
                raise ValueError(f"RobotState {name} must be in [0, 1]")
        for name in (
            "planner_executing",
            "planner_reached",
            "planner_failed",
            "model_command_executed",
        ):
            value = tensors[name]
            if bool(((value != 0.0) & (value != 1.0)).any().item()):
                raise ValueError(f"RobotState {name} must contain only binary 0/1 flags")
        if bool((tensors["planner_tracking_error"] < 0.0).any().item()):
            raise ValueError("RobotState planner_tracking_error must be non-negative")
        planner_terminal_count = (
            tensors["planner_executing"]
            + tensors["planner_reached"]
            + tensors["planner_failed"])
        if bool((planner_terminal_count > 1).any().item()):
            raise ValueError(
                "RobotState planner_executing/planner_reached/planner_failed "
                "must be mutually exclusive")
        model_command_executed = tensors["model_command_executed"].eq(1.0)
        feedback_without_id = (
            model_command_executed
            & (tensors["executed_action_id"] == 0))
        id_without_feedback = (
            ~model_command_executed
            & (tensors["executed_action_id"] != 0))
        if bool((feedback_without_id | id_without_feedback).any().item()):
            raise ValueError(
                "RobotState model_command_executed and executed_action_id must "
                "identify the same executed model command")
        endpoint_quaternion_norm = tensors["endpoint_pose"][..., 3:7].norm(
            dim=-1)
        if not torch.allclose(
            endpoint_quaternion_norm[endpoint_observed],
            torch.ones_like(endpoint_quaternion_norm[endpoint_observed]),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                "RobotState endpoint_pose quaternions must have unit length")
        planner_quaternion_norm = tensors[
            "planner_expected_endpoint_pose"][..., 3:7].norm(dim=-1)
        if not torch.allclose(
            planner_quaternion_norm,
            torch.ones_like(planner_quaternion_norm),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                "RobotState planner endpoint quaternions must have unit length")
        base_orientation_norm = tensors["base_orientation_world"].norm(dim=-1)
        if not torch.allclose(
            base_orientation_norm,
            torch.ones_like(base_orientation_norm),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                "RobotState base_orientation_world must have unit length")
        gravity_norm = tensors["gravity_direction_world"].norm(dim=-1)
        if not torch.allclose(
            gravity_norm,
            torch.ones_like(gravity_norm),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError(
                "RobotState gravity_direction_world must have unit length")
        return {name: tensors[name] for name in ROBOT_STATE_FIELDS}  # type: ignore[return-value]

    def InitAgentHandle(
        self,
        usePlanner: bool = True,):
        calibration = self.LoadCameraCalibration()
        self.camera_calibration_id = calibration.calibration_id
        self.agent_handle = AgentHandle(
            calibration=calibration,
            robotMorphology=self.robot_morphology,
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

    def _ForwardValidatedBatch(
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
        robotState: Dict[str, Any],
        requestProvenance: Dict[str, Any],):
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")

        converted = DataPreprocessor.ConvertCppCameraFrame(
            bitmap=bitmap,
            reward=reward,
            done=done,
            depthBitmap=depthBitmap,
            depthValid=depthValid,
            device=self.device,
            needVisualState=False,)
        robot_state = self.TensorizeRobotState(
            robotState,
            self.device,
            batched=True,
            robotContract=self.robot_morphology)
        batch_size = int(converted["frames"].size(0))
        self.agent_handle.agent.BindWorldMemoryContext(
            requestProvenance["world_frame_id"],
            batchSize=batch_size)
        if any(
            int(value.size(0)) != batch_size
            for value in robot_state.values()
        ):
            raise ValueError("RGB-D and RobotState must have one batch size")

        act_out = self.agent_handle.ForwardStep(
            converted["frames"],
            textExt=textExt,
            textTrust=textTrust,
            reward=converted["rewards"],
            done=converted["dones"],
            sampleActions=sampleActions,
            deterministicActor=deterministicActor,
            depth=converted["depths"],
            depthValid=converted["depth_valid"],
            robotState=robot_state,)
        return self.agent_handle.agent.UnpackActPacked(
            act_out,
            requestProvenance=requestProvenance)

    def AgentHandleForwardJson(
        self,
        reward: Optional[float],
        done: Optional[float],
        sensorPacketJson: str,
        robotStateJson: str,) -> str:
        """C++ single-frame wire protocol.

        ``sensorPacketJson`` contains one fixed-lattice synchronized RGB-D frame.
        ``robotStateJson`` contains the unbatched fields defined by
        :class:`CoreTypes.RobotState`; this adapter adds the one-frame batch axis.
        The returned command is a proposal: this adapter neither validates
        hardware feasibility nor enforces its timeout budget.
        """
        sensor_packet = json.loads(sensorPacketJson)
        robot_packet = json.loads(robotStateJson)
        if type(sensor_packet) is not dict or set(sensor_packet) != set(SENSOR_PACKET_WIRE_FIELDS):
            raise ValueError("sensor packet fields do not match the current schema")
        if (
            type(sensor_packet["schema_version"]) is not int
            or sensor_packet["schema_version"] != SENSOR_PACKET_WIRE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported sensor packet schema")
        if self.camera_calibration_id is None:
            raise RuntimeError("agent_handle has not been initialized")
        ValidateRobotStateWirePacket(
            robot_packet,
            self.camera_calibration_id,
            self.robot_morphology)
        if sensor_packet["calibration_id"] != self.camera_calibration_id:
            raise ValueError("sensor packet calibration_id does not match configured K")
        if type(sensor_packet["stream_id"]) is not str or not sensor_packet["stream_id"]:
            raise ValueError("sensor stream_id must be a non-empty string")
        if (
            type(sensor_packet["sequence_index"]) is not int
            or sensor_packet["sequence_index"] < 0
        ):
            raise ValueError(
                "sensor sequence_index must be a non-negative integer")
        if sensor_packet["stream_id"] != robot_packet["stream_id"]:
            raise ValueError("RGB-D and RobotState stream_id must match")
        if sensor_packet["sequence_index"] != robot_packet["sequence_index"]:
            raise ValueError("RGB-D and RobotState sequence_index must match")
        if sensor_packet["frame_id"] != robot_packet["frame_id"]:
            raise ValueError("RGB-D and RobotState frame_id must match")
        if not isinstance(sensor_packet["frame_id"], str) or not sensor_packet["frame_id"]:
            raise ValueError("frame_id must be a non-empty string")
        if sensor_packet["rgb_encoding"] != "rgb8" or sensor_packet["depth_unit"] != "meter":
            raise ValueError("sensor packet must contain rgb8 and metre depth")
        sequence_index = sensor_packet["sequence_index"]
        if self.active_sensor_stream_id is None:
            if sequence_index != 0:
                raise ValueError("a sensor stream must begin at sequence_index 0")
        else:
            if sensor_packet["stream_id"] != self.active_sensor_stream_id:
                raise ValueError(
                    "sensor stream_id changed; initialize a new AgentHandle")
            if robot_packet["world_frame_id"] != self.active_world_frame_id:
                raise ValueError(
                    "world_frame_id changed within an active sensor stream")
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
        robot_state = {
            name: [robot_packet[name]]
            for name in ROBOT_STATE_FIELDS}
        result = self._ForwardValidatedBatch(
            sensor_packet["rgb"],
            reward,
            done,
            textExt=text_ext,
            textTrust=text_trust,
            sampleActions=sensor_packet["sample_actions"],
            deterministicActor=sensor_packet["deterministic_actor"],
            depthBitmap=sensor_packet["depth"],
            depthValid=sensor_packet["depth_valid"],
            robotState=robot_state,
            requestProvenance={
                "stream_id": sensor_packet["stream_id"],
                "sequence_index": sequence_index,
                "frame_id": sensor_packet["frame_id"],
                "calibration_id": sensor_packet["calibration_id"],
                "world_frame_id": robot_packet["world_frame_id"],
                "description_id": self.robot_morphology.description_id,
                "model_contract_id": self.robot_morphology.model_contract_id,
                "adapter_id": self.robot_morphology.adapter_id,
            })
        self.active_sensor_stream_id = sensor_packet["stream_id"]
        self.active_world_frame_id = robot_packet["world_frame_id"]
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
        onlineLearning:bool = False,
        isTest: bool = False, 
        overrideCheckpointWithModuleParams: Optional[bool] = None, 
        saveEverySampleCount = 2000,
        trainStage: str = "full",):
        train_stage = self.NormalizeTrainStage(trainStage)
        ckpt_path = BasicParameters.CKPT_PATH_TEST if isTest else BasicParameters.CKPT_PATH_TRAIN
        out_path = BasicParameters.MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.MODULEPARAMETER_PATH
        wm_mem_path = (
            BasicParameters.WORLD_MEMORY_PATH_TEST_TRAIN
            if isTest
            else BasicParameters.WORLD_MEMORY_PATH_TRAIN)
        mem_mem_path = (
            BasicParameters.MEMORY_MEMORY_PATH_TEST_TRAIN
            if isTest
            else BasicParameters.MEMORY_MEMORY_PATH_TRAIN)
        override_enabled = bool(self.overrideCheckpointWithModuleParams) if overrideCheckpointWithModuleParams is None else bool(overrideCheckpointWithModuleParams)

        return self.StartBackgroundTask(
            self.TrainLoop,
            args=(epochs, batchSize, valSplit, resume, onlineLearning),
            kwargs={
                "worldMemPath": wm_mem_path,
                "memMemPath": mem_mem_path,
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "overrideCheckpointWithModuleParams": override_enabled,
                "saveEverySampleCount": saveEverySampleCount,
                "trainStage": train_stage,
                "isTest": isTest,})

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
        """Contiguous time-range split (train | val | test) so frame order survives;
        random_split would shuffle the stream and destroy the recurrent state's
        temporal continuity."""
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
            robot_state_dir = Path(BasicParameters.DATA_ROBOT_STATE_PATH)
            sensor_manifest_path = Path(BasicParameters.DATA_SENSOR_MANIFEST_PATH)
            texts_dir = Path(BasicParameters.DATA_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            reward_dir = root / "reward"
            done_dir = root / "done"
            depth_dir = root / "depth"
            depth_valid_dir = root / "depth_valid"
            robot_state_dir = root / "robot_state"
            sensor_manifest_path = root / "sensor_manifest.json"
            texts_dir = root / "texts"

        required_dirs = [
            frames_dir,
            reward_dir,
            done_dir,
            depth_dir,
            depth_valid_dir,
            robot_state_dir]
        if (
            not all(p.exists() for p in required_dirs)
            or not sensor_manifest_path.is_file()
        ):
            return False
        try:
            calibration = self.LoadCameraCalibration()
            sensor_manifest = json.loads(
                sensor_manifest_path.read_text(encoding="utf-8"))
            ValidateOfflineSensorManifest(
                sensor_manifest,
                calibration.calibration_id,
                self.robot_morphology,
                calibration.frame_name)
        except (OSError, TypeError, ValueError):
            return False

        counts = [
            len(sorted(frames_dir.glob("*.png"))),
            len(sorted(reward_dir.glob("*.npy"))),
            len(sorted(done_dir.glob("*.npy"))),
            len(DataPreprocessor.ListDepthFiles(depth_dir)),
            len(DataPreprocessor.ListDepthFiles(depth_valid_dir)),
            len(DataPreprocessor.ListJsonFiles(robot_state_dir)),]
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
        robotContract: CompiledRobotMorphology,) -> Tuple[str, str, str]:
        ValidateRobotTensorContract(robotContract)
        manifest_path = cls.DeploymentManifestPath(modelPath)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"deployment manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if type(manifest) is not dict or set(manifest) != DEPLOYMENT_MANIFEST_FIELDS:
            raise ValueError("deployment manifest fields do not match the current schema")
        if manifest["schema_version"] != TRAIN_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("deployment manifest schema is unsupported")
        if manifest["calibration_id"] != calibrationId:
            raise ValueError("deployment manifest calibration_id does not match configured K")
        if manifest["description_id"] != robotContract.description_id:
            raise ValueError(
                "deployment manifest description_id does not match the robot")
        if manifest["model_contract_id"] != robotContract.model_contract_id:
            raise ValueError(
                "deployment manifest model_contract_id does not match the foundation")
        if manifest["adapter_id"] != robotContract.adapter_id:
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
            "model_contract_id": brain.robot_model_contract_id,
            "brain": brain_state,}, out_path)

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
        if payload["model_contract_id"] != brain.robot_model_contract_id:
            raise ValueError(
                "brain parameter model_contract_id does not match the foundation")
        brain_state = payload["brain"]
        if type(brain_state) is not dict:
            raise TypeError("brain model state must be a dictionary")

        LoadDeploymentModelState(brain, brain_state)
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

            def collate_ocr_batch(batch):
                imgs, boxes, texts, ignore_flags = zip(*batch)
                return list(imgs), list(boxes), list(texts), list(ignore_flags)

            train_dl = self.CreateDataLoader(
                train_ds,
                batchSize=batchSize,
                shuffle=True,
                collateFn=collate_ocr_batch)

            val_dl = self.CreateDataLoader(
                val_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_ocr_batch)

            test_dl = self.CreateDataLoader(
                test_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_ocr_batch)

            patience = 10
            min_delta = 1e-4
            no_improve = 0
            box_metric_threshold = 0.95
            text_metric_threshold = 0.95
            box_iou_threshold = 0.7
            validation_interval = max(1, int(epochs) // 10)

            def eval_split(dl):
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
                    avg_val, avg_val_box_loss, avg_val_rec_loss, val_box_hmean, val_text_char_acc = eval_split(val_dl)
                    test_loss, test_box_loss, test_rec_loss, test_box_hmean, test_text_char_acc = eval_split(test_dl)

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

            def collate_rec_batch(batch):
                imgs, texts, ignore_flags = zip(*batch)
                return list(imgs), list(texts), list(ignore_flags)

            train_dl = self.CreateDataLoader(
                train_ds,
                batchSize=batchSize,
                shuffle=True,
                collateFn=collate_rec_batch)

            val_dl = self.CreateDataLoader(
                val_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_rec_batch)

            test_dl = self.CreateDataLoader(
                test_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_rec_batch)

            patience = 5
            min_delta = 1e-4
            no_improve = 0

            def eval_split(dl):
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
                avg_val, val_acc = eval_split(val_dl)
                test_loss, test_acc = eval_split(test_dl)

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
        onlineLearning = False, 
        *, 
        saveEverySampleCount = 2000,
        isTest: bool = False, 
        worldMemPath: str = None, 
        memMemPath: str = None, 
        ckptPath: str = None,
        outPath: str = None,
        overrideCheckpointWithModuleParams: bool = False,
        trainStage: str = "full",):
        try:
            trainStage = self.NormalizeTrainStage(trainStage)
            critic_stage_enabled = trainStage in ("full", "policy")
            ckptPath = ckptPath or BasicParameters.CKPT_PATH_TRAIN
            outPath = outPath or (BasicParameters.MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.MODULEPARAMETER_PATH)

            self.ResetControllerFlags()
    
            calibration = self.LoadCameraCalibration()
            ds = OfflineGameDataset(
                calibrationId=calibration.calibration_id,
                robotMorphology=self.robot_morphology,
                sensorFrameName=calibration.frame_name,
                isTest=isTest)

            brain = BrainCore(
                calibration=calibration,
                robotMorphology=self.robot_morphology,
                device=self.device,
                plasticOnlineLearning=onlineLearning,
                enablePerceptionSupervision=True)

            resume_checkpoint_exists = resume and Path(ckptPath).exists()
            if overrideCheckpointWithModuleParams:
                (
                    deployment_model_path,
                    initial_world_memory_path,
                    initial_memory_path,
                ) = self.ResolveDeploymentArtifactPaths(
                    outPath,
                    calibrationId=calibration.calibration_id,
                    robotContract=self.robot_morphology)
            else:
                deployment_model_path = str(outPath)
                initial_world_memory_path = worldMemPath
                initial_memory_path = memMemPath
            agent = Agent(
                brain,
                isTrain=True,
                device=self.device,
                worldMemoryPath=initial_world_memory_path,
                memMemoryPath=initial_memory_path)
            agent.BindWorldMemoryContext(
                ds.world_frame_id,
                batchSize=batchSize,
                loadPersistent=(
                    not resume_checkpoint_exists
                    and overrideCheckpointWithModuleParams))
            if onlineLearning and not resume_checkpoint_exists:
                agent.ResetOnlineCandidateState()

            start_epoch = 0
            best_val = float("inf")
            processed_sample_count_total = 0
            no_improve = 0
            resume_next_batch_index = 0
            resume_epoch_loss_sum = 0.0
            train_ds = val_ds = test_ds = None

            testSplit = 0.1

            if resume_checkpoint_exists:
                resume_state = self.LoadCheckpoint(
                    brain,
                    agent,
                    ds,
                    ckptPath,
                    batchSize=batchSize,
                    trainStage=trainStage,
                    onlineLearning=onlineLearning)
                start_epoch = resume_state.epoch
                resume_next_batch_index = resume_state.next_batch_index
                resume_epoch_loss_sum = resume_state.epoch_loss_sum
                best_val = resume_state.best_val
                no_improve = resume_state.no_improve
                processed_sample_count_total = (
                    resume_state.processed_sample_count_total)
                train_ds = resume_state.train_dataset
                val_ds = resume_state.validation_dataset
                test_ds = resume_state.test_dataset
                
            parameters_overridden = self.ApplyParameterOverrideAfterResume(
                enabled=overrideCheckpointWithModuleParams,
                parameterPath=deployment_model_path,
                loadFn=lambda path: self.LoadBrainWeights(brain, path, agent=agent),
                logPrefix="Train")
            if parameters_overridden:
                resume_next_batch_index = 0
                resume_epoch_loss_sum = 0.0
                best_val = float("inf")
                no_improve = 0
            if onlineLearning and (not resume_checkpoint_exists or parameters_overridden):
                agent.UpdateWrappers(
                    self.TrainStageOnlineWrappers(brain, trainStage),
                    "autogrow")
            train_ds, val_ds, test_ds = self.SplitDatasetSequential(
                ds,
                valSplit=valSplit,
                testSplit=testSplit,
                trainDataset=train_ds,
                valDataset=val_ds,
                testDataset=test_ds)

            train_dl = SequentialTrajectoryLoader(train_ds, batchSize=batchSize)
            val_dl = SequentialTrajectoryLoader(val_ds, batchSize=batchSize)
            test_dl = SequentialTrajectoryLoader(test_ds, batchSize=batchSize)

            patience = 5 
            min_delta = 1e-4 
            def unpack_batch(batch):
                img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, robot_state_b, synthetic_targets = batch

                ext_text_b = [
                    None if (t is None or str(t).strip() == "") else str(t)
                    for t in ext_text_b]

                return img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, robot_state_b, synthetic_targets

            def move_target_batch(targets: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    name: value.to(self.device) if torch.is_tensor(value) else value
                    for name, value in targets.items()}

            def BuildTrainCheckpointPayload(
                epochValue: int,
                *,
                nextBatchIndex: int,
                epochLossSum: float,) -> Dict[str, Any]:
                buffers = brain.ExportBuffers()
                world_memory = agent.GetRuntimeWorld().ExportMemoryPayload()
                return {
                    "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                    "calibration_id": calibration.calibration_id,
                    "description_id": brain.robot_description_id,
                    "model_contract_id": brain.robot_model_contract_id,
                    "adapter_id": brain.robot_adapter_id,
                    "world_frame_id": ds.world_frame_id,
                    "epoch": int(epochValue),
                    "next_batch_index": int(nextBatchIndex),
                    "epoch_loss_sum": float(epochLossSum),
                    "best_val": best_val,
                    "no_improve": int(no_improve),
                    "train_stage": trainStage,
                    "batch_size": int(batchSize),
                    "online_learning": bool(onlineLearning),
                    "brain": ExportBrainModelState(brain),
                    "online_candidates": agent.ExportOnlineCandidateState(),
                    "opt_actor": agent.opt_actor.state_dict(),
                    "opt_critic": agent.opt_critic.state_dict(),
                    "opt_world": agent.opt_world.state_dict(),
                    "train_indices": list(train_ds.indices),
                    "val_indices": list(val_ds.indices),
                    "test_indices": list(test_ds.indices),
                    "processed_sample_count_total": processed_sample_count_total,
                    "rng": self.CaptureRngState(),
                    "buffers": buffers,
                    "world_memory": world_memory,
                    "memory_durable": brain.mem.ExportDurableState(),}

            def SaveTrainArtifacts(
                epochValue: int,
                *,
                nextBatchIndex: int = 0,
                epochLossSum: float = 0.0,
                publishDeployment: bool = False,
                logPeriodic: bool = False,) -> None:
                checkpoint_payload = BuildTrainCheckpointPayload(
                    epochValue,
                    nextBatchIndex=nextBatchIndex,
                    epochLossSum=epochLossSum)
                if publishDeployment:
                    training_modes = [
                        (module, bool(module.training))
                        for module in brain.modules()]
                    try:
                        if onlineLearning:
                            agent.UpdateAllWrappers("commit")
                        generation = (
                            f"epoch-{int(epochValue):08d}-"
                            f"samples-{int(processed_sample_count_total):012d}-"
                            f"{time.time_ns()}")
                        model_path = Path(outPath)
                        generation_dir = (
                            model_path.parent
                            / f".{model_path.stem}_deployments"
                            / generation)
                        generation_model_path = generation_dir / model_path.name
                        generation_world_path = generation_dir / (
                            Path(worldMemPath).name
                            if worldMemPath
                            else "world_memory.pth")
                        generation_memory_path = generation_dir / (
                            Path(memMemPath).name
                            if memMemPath
                            else "memory.pth")
                        generation_dir.mkdir(parents=True, exist_ok=False)
                        agent.SaveWorldMemory(
                            str(generation_world_path))
                        agent.SaveAgentMemory(str(generation_memory_path))
                        self.SaveModuleParameters(
                            brain,
                            str(generation_model_path))
                        self.AtomicJsonSave({
                            "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                            "calibration_id": calibration.calibration_id,
                            "description_id": brain.robot_description_id,
                            "model_contract_id": brain.robot_model_contract_id,
                            "adapter_id": brain.robot_adapter_id,
                            "generation": generation,
                            "model_path": str(generation_model_path.resolve()),
                            "world_memory_path": str(generation_world_path.resolve()),
                            "memory_path": str(generation_memory_path.resolve()),
                        }, self.DeploymentManifestPath(outPath))
                    finally:
                        if onlineLearning:
                            self.ImportTrainingCheckpointState(
                                brain,
                                agent,
                                checkpoint_payload,
                                batchSize=batchSize)
                            for module, was_training in training_modes:
                                module.training = was_training
                self.AtomicTorchSave(checkpoint_payload, ckptPath)
                if logPeriodic:
                    print(
                        f"[Train] periodic save at processed_sample_count_total={processed_sample_count_total} "
                        f"(epoch {ep + 1}, batch {bi})")

            self.controller.SetStatus("training", "Training started", epoch=start_epoch, total_epochs=epochs, batch=0, total_batches=len(train_dl), visual=self.controller.EmptyVisualStatus(touch=True),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

                if not self.WaitWhilePaused("Training paused"):
                    if self.controller.ShouldStop():
                        self.controller.SetStatus("stopped", "Training stopped")
                    break

                brain.train()
                resume_this_epoch = (
                    ep == start_epoch and resume_next_batch_index > 0)
                if resume_this_epoch:
                    if resume_next_batch_index > len(train_dl):
                        raise ValueError(
                            "checkpoint next_batch_index exceeds the train split")
                    epoch_loss = resume_epoch_loss_sum
                    nb = resume_next_batch_index
                    train_iterator = train_dl.IterFrom(resume_next_batch_index)
                    train_enumeration_start = resume_next_batch_index + 1
                else:
                    epoch_loss = 0.0
                    nb = 0
                    agent.ResetBrainState(B=batchSize, isOnlineLearning=onlineLearning)
                    train_iterator = iter(train_dl)
                    train_enumeration_start = 1

                for bi, batch in enumerate(
                    train_iterator,
                    start=train_enumeration_start):
                    if self.controller.ShouldStop():
                        break
                    img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, robot_state_b, synthetic_targets_b = unpack_batch(batch)

                    visual_enabled = self.controller.IsVisualStateEnabled()
                    pack = DataPreprocessor.ConvertRobotInputs(
                        imgs=img_b,
                        reward=reward_b,
                        done=done_b,
                        depths=depth_b,
                        depthValids=depth_valid_b,
                        device=self.device,
                        needVisualState=visual_enabled,)
                    
                    frames = pack["frames"]
                    original_images = pack["original_images"]
                    resize_meta = pack["resize_meta"]
                    depth_t = pack["depths"]
                    depth_valid_t = pack["depth_valid"]
                    reward_t = pack["rewards"]
                    done_t = pack["dones"]
                    synthetic_targets_t = move_target_batch(synthetic_targets_b)
                    robot_state_t = self.TensorizeRobotState(
                        robot_state_b,
                        self.device,
                        batched=True,
                        robotContract=self.robot_morphology)
                    perception_targets = dict(synthetic_targets_t)
                    perception_targets.update({
                        "rgb": frames,
                        "depth": depth_t,
                        "depth_valid": depth_valid_t,})

                    if self.controller.ShouldResetHebbian():
                        agent.ResetHebbianMemory()
                        self.controller.RequestCancelResetHebbian()

                    act_out = agent.Act(AgentActInput(
                        frame=frames,
                        text_ext=ext_text_b,
                        reward=reward_t,
                        done=done_t,
                        sample_actions=True,
                        deterministic_actor=False,
                        depth=depth_t,
                        depth_valid=depth_valid_t,
                        perception_targets=perception_targets,
                        robot_state=robot_state_t,
                        compute_critic_loss=critic_stage_enabled,
                        text_trust=[TEXT_TRUST_OPERATOR_COMMAND for _ in range(len(ext_text_b))]))

                    if act_out is None:
                        continue

                    model_loss = act_out.loss
                    transport_delayed_loss = act_out.transport_delayed_loss
                    ocr_items = act_out.ocr

                    optimization_losses = act_out.optimization_losses
                    if set(optimization_losses) != {"world", "critic", "policy"}:
                        raise RuntimeError(
                            "BrainCore must expose world/critic/policy optimization losses")
                    selected_loss_names = self.TrainStageLossNames(trainStage)
                    current_stage_loss = sum(
                        (optimization_losses[name] for name in selected_loss_names),
                        model_loss.new_zeros(()))
                    loss = current_stage_loss
                    if critic_stage_enabled:
                        loss = loss + transport_delayed_loss.detach()

                    agent.opt_world.zero_grad(set_to_none=True)
                    agent.opt_critic.zero_grad(set_to_none=True)
                    agent.opt_actor.zero_grad(set_to_none=True)

                    world_parameters = agent.OptimizerParameters([agent.opt_world])
                    critic_parameters = agent.OptimizerParameters([agent.opt_critic])
                    policy_parameters = agent.OptimizerParameters([agent.opt_actor])

                    transport_capture_delayed = {"captured": 0.0, "grad_norm": 0.0, "accum_steps": 0.0}
                    if critic_stage_enabled and transport_delayed_loss.requires_grad:
                        # The delayed critic graph owns independent transport snapshots;
                        # restricting backward to current parameters would exclude those
                        # leaves and silently suppress their gradient hooks.
                        transport_delayed_loss.backward()
                        transport_capture_delayed = agent.CaptureCriticTransportGrad()

                    current_backward_jobs = []
                    if trainStage in ("full", "world"):
                        current_backward_jobs.append(
                            (optimization_losses["world"], world_parameters))
                    if critic_stage_enabled:
                        current_backward_jobs.append(
                            (optimization_losses["critic"], critic_parameters))
                    if trainStage in ("full", "policy"):
                        current_backward_jobs.append(
                            (optimization_losses["policy"], policy_parameters))
                    current_backward_jobs = [
                        (objective, parameters)
                        for objective, parameters in current_backward_jobs
                        if objective.requires_grad and len(parameters) > 0]
                    for job_index, (objective, parameters) in enumerate(current_backward_jobs):
                        objective.backward(
                            inputs=parameters,
                            retain_graph=(job_index + 1 < len(current_backward_jobs)))
                    transport_capture_current = {"captured": 0.0, "grad_norm": 0.0, "accum_steps": 0.0}
                    if critic_stage_enabled:
                        transport_capture_current = agent.CaptureCriticTransportGrad()
                    else:
                        agent.ClearCriticTransportGradAccumulator()
                    transport_capture = {
                        "captured": transport_capture_delayed["captured"] + transport_capture_current["captured"],
                        "grad_norm": (
                            transport_capture_delayed["grad_norm"] ** 2
                            + transport_capture_current["grad_norm"] ** 2) ** 0.5,
                        "accum_steps": max(
                            transport_capture_delayed["accum_steps"],
                            transport_capture_current["accum_steps"]),}
                    transport_apply = {"updated": 0.0, "grad_norm": 0.0, "scale": 0.0}
                    if critic_stage_enabled:
                        transport_apply = agent.ApplyCriticTransportManualGrad()

                    if trainStage == "world":
                        stage_optimizers = [agent.opt_world]
                    elif trainStage == "policy":
                        stage_optimizers = [agent.opt_critic, agent.opt_actor]
                    else:
                        stage_optimizers = [agent.opt_world, agent.opt_critic, agent.opt_actor]
                    torch.nn.utils.clip_grad_norm_(
                        agent.OptimizerParameters(stage_optimizers),
                        1.0)

                    # Phased curriculum: "world" trains the world model only,
                    # "policy" trains critic+actor on a frozen world model, "full" trains all.
                    if trainStage in ("full", "world"):
                        agent.opt_world.step()
                    if trainStage in ("full", "policy"):
                        agent.opt_critic.step()
                        agent.opt_actor.step()
                    if critic_stage_enabled:
                        agent.AfterOptimizerStep()
                    if onlineLearning:
                        stage_wrappers = self.TrainStageOnlineWrappers(brain, trainStage)
                        agent.UpdateWrappers(stage_wrappers, "accumulategrads")
                        agent.UpdateWrappers(stage_wrappers, "autogrow")

                    previous_processed_sample_count_total = processed_sample_count_total
                    processed_sample_count_total += int(frames.size(0))
                    if self.ShouldTriggerPeriodicSave(
                        previous_processed_sample_count_total,
                        processed_sample_count_total,
                        saveEverySampleCount):
                        SaveTrainArtifacts(
                            ep,
                            nextBatchIndex=bi,
                            epochLossSum=epoch_loss + float(loss.item()),
                            logPeriodic=True)

                    epoch_loss += float(loss.item())
                    nb += 1

                    visual_payload = None
                    if visual_enabled and original_images:
                        batch_ocr_items = (ocr_items[0] if (ocr_items is not None and len(ocr_items) > 0) else [])
                        batch_ocr_texts = [
                            str(item.get("text", "")).strip()
                            for item in batch_ocr_items
                            if str(item.get("text", "")).strip() != ""]
                        resize_meta_0 = resize_meta[0] if resize_meta else None
                        visual_payload = self.BuildVisualPayload(
                            original_images[0],
                            ocrTexts=batch_ocr_texts,
                            ocrItems=batch_ocr_items,
                            resizeMeta=resize_meta_0,
                            title="Train",
                            extraLines=[
                                f"epoch {ep + 1}/{epochs}",
                                f"batch {bi}/{len(train_dl)}"],)

                    status_kwargs = {
                        "epoch": ep + 1,
                        "total_epochs": epochs,
                        "batch": bi,
                        "total_batches": len(train_dl),
                        "train_loss": float(loss.item()),}
                    if visual_payload is not None:
                        status_kwargs["visual"] = visual_payload

                    transport_status = ""
                    if transport_capture["captured"] > 0.0 or transport_apply["updated"] > 0.0:
                        transport_status = (
                            f" transport_grad={transport_capture['grad_norm']:.3e}"
                            f" transport_update={int(transport_apply['updated'])}")
                    self.controller.SetStatus("training", f"Training...{transport_status}", **status_kwargs)

                    if self.controller.ShouldStop():
                        break
                    if not self.WaitWhilePaused("Training paused"):
                        break

                resume_next_batch_index = 0
                resume_epoch_loss_sum = 0.0

                avg_train = epoch_loss / max(1, nb)

                self.controller.SetStatus("training", f"Epoch {ep+1}/{epochs} done, avg_train={avg_train:.4f}", epoch=ep + 1, total_epochs=epochs,)

                SaveTrainArtifacts(
                    ep,
                    nextBatchIndex=nb,
                    epochLossSum=epoch_loss)
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

                def eval_split(dl):
                    brain.eval()
                    agent.ResetBrainState(B=batchSize, isOnlineLearning=onlineLearning)
                    split_loss = 0.0
                    split_batches = 0

                    with torch.no_grad():
                        for batch in dl:
                            img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, robot_state_b, synthetic_targets_b = unpack_batch(batch)

                            v_pack = DataPreprocessor.ConvertRobotInputs(
                                imgs=img_b,
                                reward=reward_b,
                                done=done_b,
                                depths=depth_b,
                                depthValids=depth_valid_b,
                                device=self.device,
                                needVisualState=False,)
                            
                            v_frames = v_pack["frames"]
                            v_depth_t = v_pack["depths"]
                            v_depth_valid_t = v_pack["depth_valid"]
                            v_reward_t = v_pack["rewards"]
                            v_done_t = v_pack["dones"]
                            v_synthetic_targets_t = move_target_batch(synthetic_targets_b)
                            v_robot_state_t = self.TensorizeRobotState(
                                robot_state_b,
                                self.device,
                                batched=True,
                                robotContract=self.robot_morphology)
                            v_perception_targets = dict(v_synthetic_targets_t)
                            v_perception_targets.update({
                                "rgb": v_frames,
                                "depth": v_depth_t,
                                "depth_valid": v_depth_valid_t,})

                            act_out = agent.Act(AgentActInput(
                                frame=v_frames,
                                text_ext=ext_text_b,
                                reward=v_reward_t,
                                done=v_done_t,
                                sample_actions=True,
                                deterministic_actor=True,
                                depth=v_depth_t,
                                depth_valid=v_depth_valid_t,
                                perception_targets=v_perception_targets,
                                robot_state=v_robot_state_t,
                                compute_critic_loss=critic_stage_enabled,
                                text_trust=[TEXT_TRUST_OPERATOR_COMMAND for _ in range(len(ext_text_b))]))
                            
                            if act_out is None:
                                continue

                            v_optimization_losses = act_out.optimization_losses
                            if set(v_optimization_losses) != {"world", "critic", "policy"}:
                                raise RuntimeError(
                                    "BrainCore must expose world/critic/policy optimization losses")
                            v_loss_names = self.TrainStageLossNames(trainStage)
                            v_model_loss = sum(
                                (v_optimization_losses[name] for name in v_loss_names),
                                act_out.loss.new_zeros(()))
                            if critic_stage_enabled:
                                v_model_loss = (
                                    v_model_loss
                                    + act_out.transport_delayed_loss)

                            split_loss += float(v_model_loss.item())
                            split_batches += 1

                    avg_split_loss = split_loss / max(1, split_batches)
                    return avg_split_loss

                avg_val, test_loss = self.EvaluateValidationAndTestWithRestoredBrainBuffers(
                    brain,
                    lambda: eval_split(val_dl),
                    lambda: eval_split(test_dl))

                improved = (best_val - avg_val) > min_delta
                if improved:
                    best_val = avg_val
                    no_improve = 0
                else:
                    no_improve += 1
                SaveTrainArtifacts(
                    ep + 1,
                    publishDeployment=improved)

                self.controller.SetStatus(
                    "training",
                    (f"Epoch {ep+1}/{epochs} done | " f"train {avg_train:.4f} | " f"val {avg_val:.4f} | " f"test {test_loss:.4f}"), val_loss=avg_val,)

                if no_improve >= patience:
                    self.controller.SetStatus("completed", "Validation stabilized, early stop.")
                    break

                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

            else:
                self.controller.SetStatus("completed", "Training completed")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Training error: {e}", trace=tb)
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
        if ckpt["description_id"] != brain.robot_description_id:
            raise ValueError(
                "training checkpoint description_id does not match the robot")
        if ckpt["model_contract_id"] != brain.robot_model_contract_id:
            raise ValueError(
                "training checkpoint model_contract_id does not match the foundation")
        if ckpt["adapter_id"] != brain.robot_adapter_id:
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
        self.RestoreRngState(checkpoint["rng"])

    def TestPerceptionModule(self):
        return self.RunNamedTest("perception")

    def TestAttentionModule(self):
        return self.RunNamedTest("attention")

    def TestMemoryModule(self):
        return self.RunNamedTest("memory")

    def TestDecisionModule(self):
        return self.RunNamedTest("decision")

    def TestWorldModule(self):
        return self.RunNamedTest("world")

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
        dataRoot: str = BasicParameters.DATA_ROOT_PATH_TEST,
        nSamples: int = 64,
        epochs: int = 1,
        batchSize: int = 1,
        valSplit: float = 0.2,
        seed: int = 42,) -> Dict[str, Any]:
        if self.is_begin:
            return {"ok": False, "msg": "StartTraining returns False (training may already be running)"} 

        try:
            root = Path(dataRoot)
            BasicParameters.Set("DATA_ROOT_PATH_TEST", str(root))
            BasicParameters.Set(
                "DATA_SENSOR_MANIFEST_PATH_TEST",
                str(root / "sensor_manifest.json"))
            BasicParameters.Set(
                "DATA_DEPTH_PATH_TEST",
                str(root / "depth"))
            BasicParameters.Set(
                "DATA_DEPTH_VALID_PATH_TEST",
                str(root / "depth_valid"))
            BasicParameters.Set(
                "DATA_ROBOT_STATE_PATH_TEST",
                str(root / "robot_state"))
            BasicParameters.Set(
                "DATA_NORMAL_PATH_TEST",
                str(root / "normal"))
            BasicParameters.Set(
                "DATA_SEMANTIC_SEGMENTATION_PATH_TEST",
                str(root / "semantic_segmentation"))
            BasicParameters.Set(
                "DATA_INSTANCE_SEGMENTATION_PATH_TEST",
                str(root / "instance_segmentation"))
            BasicParameters.Set(
                "DATA_SYNTHETIC_SUPERVISION_PATH_TEST",
                str(root / "synthetic_supervision"))

            if not self.HasGameDataset(root):
                if iio is None:
                    raise RuntimeError("imageio.v3 error")

                rng = np.random.default_rng(seed)
                if root.exists():
                    shutil.rmtree(root)
                (root / "frames").mkdir(parents=True, exist_ok=True)
                (root / "reward").mkdir(parents=True, exist_ok=True)
                (root / "done").mkdir(parents=True, exist_ok=True)
                (root / "depth").mkdir(parents=True, exist_ok=True)
                (root / "depth_valid").mkdir(parents=True, exist_ok=True)
                (root / "robot_state").mkdir(parents=True, exist_ok=True)
                (root / "texts").mkdir(parents=True, exist_ok=True)

                H, W = BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE
                calibration = self.LoadCameraCalibration()
                sensor_manifest = ExpectedOfflineSensorManifest(
                    calibration.calibration_id,
                    self.robot_morphology,
                    calibration.frame_name)
                (root / "sensor_manifest.json").write_text(
                    json.dumps(sensor_manifest),
                    encoding="utf-8")

                endpoint_pose = np.zeros((
                    self.robot_morphology.endpoint_count,
                    ModuleDim.DecisionEndpointPoseDim), dtype=np.float32)
                endpoint_pose[..., 6] = 1.0
                joint_state = np.zeros(
                    self.robot_morphology.joint_dof_count,
                    dtype=np.float32)
                joint_observed = np.ones(
                    self.robot_morphology.joint_dof_count,
                    dtype=np.bool_)
                joint_controllable = (
                    self.robot_morphology.joint_variable_commandable.cpu().numpy())
                node_pose = np.zeros(
                    (self.robot_morphology.node_count, 7),
                    dtype=np.float32)
                node_pose[..., 6] = 1.0
                node_twist = np.zeros(
                    (self.robot_morphology.node_count, 6),
                    dtype=np.float32)
                node_observed = np.zeros(
                    self.robot_morphology.node_count,
                    dtype=np.bool_)
                endpoint_observed = np.ones(
                    self.robot_morphology.endpoint_count,
                    dtype=np.bool_)
                endpoint_controllable = (
                    self.robot_morphology.endpoint_task_mask.any(
                        dim=-1).cpu().numpy())
                base_orientation_world = np.zeros(4, dtype=np.float32)
                base_orientation_world[3] = 1.0
                robot_metadata = ExpectedRobotStateWireMetadata(
                    self.robot_morphology)

                templates = ["move left", "move right", "move forward", "move back",
                            "use skill", "defend", "attack", "pickup item",
                            "open menu", "retreat",]

                for i in range(nSamples):
                    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
                    iio.imwrite(str(root / "frames" / f"{i:05d}.png"), img)

                    depth = rng.uniform(0.1, 2.0, size=(H, W)).astype(np.float32)
                    np.save(str(root / "depth" / f"{i:05d}.npy"), depth)
                    np.save(
                        str(root / "depth_valid" / f"{i:05d}.npy"),
                        np.ones((H, W), dtype=np.bool_))

                    reward = rng.normal(loc=0.0, scale=2.0, size=(1,)).astype(np.float32)
                    np.save(str(root / "reward" / f"{i:05d}.npy"), reward)

                    done = rng.normal(loc=0.0, scale=2.0, size=(1,)).astype(np.float32)
                    np.save(str(root / "done" / f"{i:05d}.npy"), done)

                    frame_id = f"{i:05d}"
                    robot_packet = {
                        "schema_version": ROBOT_STATE_WIRE_SCHEMA_VERSION,
                        "stream_id": "offline-test",
                        "sequence_index": i,
                        "frame_id": frame_id,
                        "calibration_id": calibration.calibration_id,
                        "world_frame_id": "test_world",
                        **robot_metadata,
                        "joint_position": joint_state.tolist(),
                        "joint_velocity": joint_state.tolist(),
                        "joint_effort": joint_state.tolist(),
                        "joint_observed": joint_observed.tolist(),
                        "joint_healthy": joint_observed.tolist(),
                        "joint_controllable": joint_controllable.tolist(),
                        "node_pose_world": node_pose.tolist(),
                        "node_twist_world": node_twist.tolist(),
                        "node_observed": node_observed.tolist(),
                        "node_healthy": node_observed.tolist(),
                        "endpoint_pose": endpoint_pose.tolist(),
                        "endpoint_observed": endpoint_observed.tolist(),
                        "endpoint_healthy": endpoint_observed.tolist(),
                        "endpoint_controllable": endpoint_controllable.tolist(),
                        "observer_pose_world": [
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        "observer_pose_valid": False,
                        "base_orientation_world": base_orientation_world.tolist(),
                        "gravity_direction_world": [0.0, 0.0, -1.0],
                        "planner_expected_endpoint_pose": endpoint_pose.tolist(),
                        "planner_progress": 0.0,
                        "planner_tracking_error": 0.0,
                        "planner_executing": 0.0,
                        "planner_reached": 0.0,
                        "planner_failed": 0.0,
                        "model_command_executed": 0.0,
                        "executed_action_id": 0,}
                    (root / "robot_state" / f"{frame_id}.json").write_text(
                        json.dumps(robot_packet),
                        encoding="utf-8")

                    if rng.random() < 0.35:
                        ext_text = ""
                    else:
                        n_words = int(rng.integers(1, 4))
                        picks = rng.choice(templates, size=n_words, replace=False).tolist()
                        ext_text = " | ".join(picks)
                    with open(root / "texts" / f"{i:05d}.txt", "w", encoding="utf-8") as f:
                        f.write(ext_text)

                print(f"[SmokeTest] created random game dataset at: {root}")
            else:
                print(f"[SmokeTest] use existing game dataset at: {root}")

            print("[SmokeTest] start train...")

            """ok = self.StartTraining(epochs=epochs,batchSize=batchSize,valSplit=valSplit,resume=False, onlineLearning=onlineLearning, isTest=True)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"ok": False, "msg": "StartTraining returns False (training may already be running)"}

            self.message_thread = threading.Thread(target=self.MonitorTraining,args=(),daemon=False,)
            self.message_thread.start()"""

            self.TrainLoop(epochs, batchSize, valSplit, False, onlineLearning, isTest=True, worldMemPath=BasicParameters.WORLD_MEMORY_PATH_TEST_TRAIN, memMemPath=BasicParameters.MEMORY_MEMORY_PATH_TEST_TRAIN,ckptPath=BasicParameters.CKPT_PATH_TEST)
        
            return {"ok": True}

        except Exception as e:
            print(f"TestModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise



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
        trainStage: str = "full",) -> Dict[str, Any]:
        try:
            if saveEverySampleCount is None:
                saveEverySampleCount = BasicParameters.SAVE_EVERY_SAMPLE_COUNT
            if not self.HasGameDataset():
                print(
                    "[Train] no dataset found, please prepare these folders first: "
                    f"{BasicParameters.DATA_FRAMES_PATH}, "
                    f"{BasicParameters.DATA_REWARD_PATH}, "
                    f"{BasicParameters.DATA_DONE_PATH}, "
                    f"{BasicParameters.DATA_DEPTH_PATH}.")
                return {"ok": False, "msg": "no dataset"}

            print(
                "[Train] use configured dataset folders: "
                f"{BasicParameters.DATA_FRAMES_PATH}, "
                f"{BasicParameters.DATA_REWARD_PATH}, "
                f"{BasicParameters.DATA_DONE_PATH}, "
                f"{BasicParameters.DATA_DEPTH_PATH}.")

            ok = self.StartTraining(
                epochs=epochs, 
                batchSize=batchSize, 
                valSplit=valSplit, 
                resume=isResume, 
                onlineLearning=onlineLearning,
                isTest=False,
                overrideCheckpointWithModuleParams=overrideCheckpointWithModuleParams,
                saveEverySampleCount=saveEverySampleCount,
                trainStage=trainStage,)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"ok": False, "msg": "StartTraining returns False (training may already be running)"}

            self.StartMessageMonitor(self.MonitorTraining)

            return {"ok": True}

        except Exception as e:
            print(f"ModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def DeployModule(
        self,
        usePlanner: bool = True,) -> Dict[str, Any]:
        """Initialize the push-stream runtime used by AgentHandleForward(Json).

        RGB-D and measured RobotState remain caller-owned synchronized inputs;
        deployment never opens a camera or fabricates depth/robot poses.
        """
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
    @staticmethod
    def MakeRobotMorphology() -> CompiledRobotMorphology:
        return ManagerFunction.LoadRobotMorphology()

    def TestDeploymentConfigurationRouting(self) -> bool:
        try:
            manager = object.__new__(ManagerFunction)
            captured: Dict[str, Any] = {}

            def init_agent_handle(**kwargs):
                captured.update(kwargs)
                return True

            manager.InitAgentHandle = init_agent_handle
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

            def draw_random() -> Tuple[float, float, float]:
                return (
                    random.random(),
                    float(np.random.random()),
                    float(torch.rand(()).item()))

            def evaluate_validation():
                split_entries.append(float(brain.runtime.item()))
                split_random.append(draw_random())
                brain.eval()
                brain.runtime.fill_(11.0)
                brain.world.physical.fill_(77.0)
                brain.mem.durable.fill_(55.0)
                return 1.25

            def evaluate_test():
                split_entries.append(float(brain.runtime.item()))
                split_random.append(draw_random())
                brain.eval()
                brain.runtime.fill_(99.0)
                brain.world.physical.fill_(99.0)
                brain.mem.durable.fill_(99.0)
                return 2.5

            result = manager.EvaluateValidationAndTestWithRestoredBrainBuffers(
                brain,
                evaluate_validation,
                evaluate_test)
            saved_buffers = brain.ExportBuffers()
            random_after_evaluation = draw_random()
            manager.RestoreRngState(evaluation_start_rng)
            expected_random = draw_random()
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
            robot_morphology = self.MakeRobotMorphology()
            with tempfile_module.TemporaryDirectory() as directory:
                root = Path(directory)
                configured_model = root / "model.pth"
                configured_world = root / "world.pth"
                configured_memory = root / "memory.pth"
                missing_manifest_rejected = False
                try:
                    manager.ResolveDeploymentArtifactPaths(
                        configured_model,
                        calibrationId="calibration-a",
                        robotContract=robot_morphology)
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
                    "description_id": robot_morphology.description_id,
                    "model_contract_id": robot_morphology.model_contract_id,
                    "adapter_id": robot_morphology.adapter_id,
                    "generation": "generation-1",
                    "model_path": str(model),
                    "world_memory_path": str(world),
                    "memory_path": str(memory),
                }, manager.DeploymentManifestPath(configured_model))
                resolved = manager.ResolveDeploymentArtifactPaths(
                    configured_model,
                    calibrationId="calibration-a",
                    robotContract=robot_morphology)
                mismatch_rejected = False
                try:
                    manager.ResolveDeploymentArtifactPaths(
                        configured_model,
                        calibrationId="calibration-b",
                        robotContract=robot_morphology)
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
                        robotContract=robot_morphology)
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
                    self.robot_description_id = "test-robot"
                    self.robot_model_contract_id = "test-model-contract"
                    self.robot_adapter_id = "test-adapter"
                    self.register_buffer("buffer", torch.zeros(2))
                    self.mem = FakeMemory()

                def load_state_dict(self, state, strict):
                    events.append(f"brain_load:{strict}")
                    return super().load_state_dict(state, strict=strict)

                def ImportBuffers(self, state):
                    events.append("buffers")

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

                def ImportOnlineCandidateState(self, state):
                    if state != {}:
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
            checkpoint = {
                "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                "calibration_id": "test-calibration",
                "description_id": "test-robot",
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
                "online_candidates": {},
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
            extra["legacy_camera_pose"] = torch.zeros(7)
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
                "world_validate",
                "memory_validate",
                "brain_load:False",
                "candidates",
                "sync",
                "opt_actor",
                "opt_critic",
                "opt_world",
                "world_import",
                "memory_import",
                "buffers",]
                and resume_state.no_improve == 2
                and missing_rejected
                and extra_rejected
                and malformed_buffers_rejected_before_load
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
                    self.robot_description_id = "test-robot"
                    self.robot_model_contract_id = "test-model-contract"
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
            manager.robot_morphology = self.MakeRobotMorphology()
            manager.camera_calibration_id = "test-camera"
            manager.active_sensor_stream_id = None
            manager.active_world_frame_id = None
            manager.last_sensor_sequence_index = None
            captured: Dict[str, Any] = {}

            def capture_forward(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return "ok"

            manager._ForwardValidatedBatch = capture_forward
            endpoint_pose = torch.zeros(
                manager.robot_morphology.endpoint_count,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            planner_pose = endpoint_pose.clone()
            base_orientation_world = torch.tensor([0.0, 0.0, 0.0, 1.0])
            joint_state = torch.zeros(
                manager.robot_morphology.joint_dof_count)
            joint_observed = torch.ones(
                manager.robot_morphology.joint_dof_count,
                dtype=torch.bool)
            node_pose = torch.zeros(
                manager.robot_morphology.node_count,
                7)
            node_pose[..., 6] = 1.0
            node_twist = torch.zeros(
                manager.robot_morphology.node_count,
                6)
            node_observed = torch.zeros(
                manager.robot_morphology.node_count,
                dtype=torch.bool)
            endpoint_observed = torch.ones(
                manager.robot_morphology.endpoint_count,
                dtype=torch.bool)
            robot_packet = {
                "schema_version": ROBOT_STATE_WIRE_SCHEMA_VERSION,
                "stream_id": "stream-1",
                "sequence_index": 0,
                "frame_id": "frame-1",
                "calibration_id": "test-camera",
                "world_frame_id": "map-v1",
                **ExpectedRobotStateWireMetadata(manager.robot_morphology),
                "joint_position": joint_state.tolist(),
                "joint_velocity": joint_state.tolist(),
                "joint_effort": joint_state.tolist(),
                "joint_observed": joint_observed.tolist(),
                "joint_healthy": joint_observed.tolist(),
                "joint_controllable": (
                    manager.robot_morphology.joint_variable_commandable.tolist()),
                "node_pose_world": node_pose.tolist(),
                "node_twist_world": node_twist.tolist(),
                "node_observed": node_observed.tolist(),
                "node_healthy": node_observed.tolist(),
                "endpoint_pose": endpoint_pose.tolist(),
                "endpoint_observed": endpoint_observed.tolist(),
                "endpoint_healthy": endpoint_observed.tolist(),
                "endpoint_controllable": (
                    manager.robot_morphology.endpoint_task_mask.any(
                        dim=-1).tolist()),
                "observer_pose_world": [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "observer_pose_valid": False,
                "base_orientation_world": base_orientation_world.tolist(),
                "gravity_direction_world": [0.0, 0.0, -1.0],
                "planner_expected_endpoint_pose": planner_pose.tolist(),
                "planner_progress": 0.2,
                "planner_tracking_error": 0.1,
                "planner_executing": 1.0,
                "planner_reached": 0.0,
                "planner_failed": 0.0,
                "model_command_executed": 0.0,
                "executed_action_id": 0,}
            sensor_packet = {
                "schema_version": SENSOR_PACKET_WIRE_SCHEMA_VERSION,
                "stream_id": "stream-1",
                "sequence_index": 0,
                "frame_id": "frame-1",
                "calibration_id": "test-camera",
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
                json.dumps(robot_packet))
            robot_state = captured["kwargs"]["robotState"]

            duplicate_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(sensor_packet),
                    json.dumps(robot_packet))
            except ValueError:
                duplicate_rejected = True

            gap_sensor_packet = dict(sensor_packet)
            gap_sensor_packet["sequence_index"] = 2
            gap_sensor_packet["frame_id"] = "frame-3"
            gap_robot_packet = dict(robot_packet)
            gap_robot_packet["sequence_index"] = 2
            gap_robot_packet["frame_id"] = "frame-3"
            gap_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(gap_sensor_packet),
                    json.dumps(gap_robot_packet))
            except ValueError:
                gap_rejected = True

            next_sensor_packet = dict(sensor_packet)
            next_sensor_packet["sequence_index"] = 1
            next_sensor_packet["frame_id"] = "frame-2"
            next_robot_packet = dict(robot_packet)
            next_robot_packet["sequence_index"] = 1
            next_robot_packet["frame_id"] = "frame-2"
            next_robot_packet["world_frame_id"] = "map-v2"
            world_frame_change_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(next_sensor_packet),
                    json.dumps(next_robot_packet))
            except ValueError:
                world_frame_change_rejected = True

            wrong_calibration_sensor = dict(next_sensor_packet)
            wrong_calibration_sensor["calibration_id"] = "other-camera"
            next_robot_packet["world_frame_id"] = "map-v1"
            calibration_mismatch_rejected = False
            try:
                ManagerFunction.AgentHandleForwardJson(
                    manager,
                    0.5,
                    0.0,
                    json.dumps(wrong_calibration_sensor),
                    json.dumps(next_robot_packet))
            except ValueError:
                calibration_mismatch_rejected = True

            ok = (
                result == "ok"
                and captured["args"] == (sensor_packet["rgb"], 0.5, 0.0)
                and tuple(torch.as_tensor(robot_state["endpoint_pose"]).shape)
                == (
                    1,
                    manager.robot_morphology.endpoint_count,
                    ModuleDim.DecisionEndpointPoseDim)
                and tuple(torch.as_tensor(
                    robot_state["planner_expected_endpoint_pose"]).shape)
                == (
                    1,
                    manager.robot_morphology.endpoint_count,
                    ModuleDim.DecisionEndpointPoseDim)
                and tuple(torch.as_tensor(
                    robot_state["base_orientation_world"]).shape) == (1, 4)
                and tuple(torch.as_tensor(
                    robot_state["gravity_direction_world"]).shape) == (1, 3)
                and tuple(torch.as_tensor(
                    robot_state["model_command_executed"]).shape) == (1,)
                and tuple(torch.as_tensor(
                    robot_state["executed_action_id"]).shape) == (1,)
                and "camera_pose_world" not in robot_state
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
                    "calibration_id": "test-camera",
                    "world_frame_id": "map-v1",
                    "description_id": manager.robot_morphology.description_id,
                    "model_contract_id": (
                        manager.robot_morphology.model_contract_id),
                    "adapter_id": manager.robot_morphology.adapter_id,
                }
                and duplicate_rejected
                and gap_rejected
                and world_frame_change_rejected
                and calibration_mismatch_rejected
                and manager.active_sensor_stream_id == "stream-1"
                and manager.active_world_frame_id == "map-v1"
                and manager.last_sensor_sequence_index == 0)
            print(f"Manager C++ decision wire contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager C++ decision wire contract error: {e}")
            return False

    def TestRobotStatePhysicalReferenceContract(self) -> bool:
        try:
            robot_morphology = self.MakeRobotMorphology()
            batch_size = 2
            endpoint_pose = torch.zeros(
                batch_size,
                robot_morphology.endpoint_count,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            base_orientation_world = torch.zeros(batch_size, 4)
            base_orientation_world[..., 3] = 1.0
            scalar = torch.zeros(batch_size)
            joint_state = torch.zeros(
                batch_size,
                robot_morphology.joint_dof_count)
            joint_observed = torch.ones_like(joint_state, dtype=torch.bool)
            node_pose = torch.zeros(
                batch_size,
                robot_morphology.node_count,
                7)
            node_pose[..., 6] = 1.0
            node_twist = torch.zeros(
                batch_size,
                robot_morphology.node_count,
                6)
            node_observed = torch.zeros(
                batch_size,
                robot_morphology.node_count,
                dtype=torch.bool)
            endpoint_observed = torch.ones(
                batch_size,
                robot_morphology.endpoint_count,
                dtype=torch.bool)
            observer_pose_world = torch.zeros(batch_size, 7)
            observer_pose_world[..., 6] = 1.0
            state = {
                "joint_position": joint_state,
                "joint_velocity": joint_state.clone(),
                "joint_effort": joint_state.clone(),
                "joint_observed": joint_observed,
                "joint_healthy": joint_observed.clone(),
                "joint_controllable": robot_morphology.joint_variable_commandable.unsqueeze(
                    0).expand(
                        batch_size, -1).clone(),
                "node_pose_world": node_pose,
                "node_twist_world": node_twist,
                "node_observed": node_observed,
                "node_healthy": node_observed.clone(),
                "endpoint_pose": endpoint_pose,
                "endpoint_observed": endpoint_observed,
                "endpoint_healthy": endpoint_observed.clone(),
                "endpoint_controllable": robot_morphology.endpoint_task_mask.any(
                    dim=-1).unsqueeze(0).expand(
                        batch_size, -1).clone(),
                "observer_pose_world": observer_pose_world,
                "observer_pose_valid": torch.zeros(
                    batch_size, dtype=torch.bool),
                "base_orientation_world": base_orientation_world,
                "gravity_direction_world": torch.tensor(
                    [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
                "planner_expected_endpoint_pose": endpoint_pose.clone(),
                "planner_progress": scalar,
                "planner_tracking_error": scalar,
                "planner_executing": scalar,
                "planner_reached": scalar,
                "planner_failed": scalar,
                "model_command_executed": scalar,
                "executed_action_id": torch.zeros(batch_size, dtype=torch.long),}
            converted = ManagerFunction.TensorizeRobotState(
                state,
                torch.device("cpu"),
                batched=True,
                robotContract=robot_morphology)
            unbatched_state = {
                name: value[0]
                for name, value in state.items()}
            converted_unbatched = ManagerFunction.TensorizeRobotState(
                unbatched_state,
                torch.device("cpu"),
                batched=False,
                robotContract=robot_morphology)

            node_state = dict(state)
            node_state["node_pose_world"] = node_pose.clone()
            node_state["node_twist_world"] = node_twist.clone()
            node_state["node_observed"] = node_observed.clone()
            node_state["node_healthy"] = node_observed.clone()
            node_state["node_pose_world"][0, 0, 0] = 0.25
            node_state["node_twist_world"][0, 0, 1] = 0.5
            node_state["node_observed"][0, 0] = True
            node_state["node_healthy"][0, 0] = True
            converted_node_state = ManagerFunction.TensorizeRobotState(
                node_state,
                torch.device("cpu"),
                batched=True,
                robotContract=robot_morphology)

            observer_morphology = RobotMorphologyModule().FromMoveIt(
                    BasicParameters.ROBOT_URDF_PATH,
                    BasicParameters.ROBOT_SRDF_PATH,
                    {
                        "sensors": [{
                            "name": "head_camera",
                            "type": "rgbd",
                            "link": "base_link",
                        }],
                        "observer": {"sensor": "head_camera"},
                        "observer_frame_name": "camera_optical",
                        "observer_calibration_id": "observer-calibration",
                    })
            ValidateRobotObserverCalibration(
                observer_morphology,
                "observer-calibration",
                "camera_optical")
            observer_state = dict(state)
            observer_state["observer_pose_world"] = observer_pose_world.clone()
            observer_state["observer_pose_world"][0, :3] = torch.tensor(
                [0.1, 0.2, 0.3])
            observer_state["observer_pose_valid"] = torch.tensor(
                [True, False])
            converted_observer_state = ManagerFunction.TensorizeRobotState(
                observer_state,
                torch.device("cpu"),
                batched=True,
                robotContract=observer_morphology)

            missing_rejected = False
            missing = dict(state)
            missing.pop("gravity_direction_world")
            try:
                ManagerFunction.TensorizeRobotState(
                    missing,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                missing_rejected = True

            base_quaternion_rejected = False
            bad_base = dict(state)
            bad_base["base_orientation_world"] = base_orientation_world.clone()
            bad_base["base_orientation_world"][0] = 0.0
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_base,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                base_quaternion_rejected = True

            gravity_rejected = False
            bad_gravity = dict(state)
            bad_gravity["gravity_direction_world"] = torch.tensor(
                [[0.0, 0.0, -9.81], [0.0, 1.0, 0.0]])
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_gravity,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                gravity_rejected = True

            bool_rejected = False
            bad_bool = dict(state)
            bad_bool["planner_executing"] = torch.zeros(
                batch_size, dtype=torch.bool)
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_bool,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except TypeError:
                bool_rejected = True

            planner_range_rejected = False
            bad_range = dict(state)
            bad_range["planner_progress"] = torch.tensor([1.1, 0.0])
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_range,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                planner_range_rejected = True

            fractional_flag_rejected = False
            bad_flag = dict(state)
            bad_flag["planner_executing"] = torch.tensor([0.25, 0.0])
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_flag,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                fractional_flag_rejected = True

            planner_conflict_rejected = False
            bad_status = dict(state)
            bad_status["planner_executing"] = torch.tensor([1.0, 0.0])
            bad_status["planner_reached"] = torch.tensor([1.0, 0.0])
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_status,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                planner_conflict_rejected = True

            action_provenance_rejected = False
            bad_action_id = dict(state)
            bad_action_id["executed_action_id"] = torch.tensor([1, 0])
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_action_id,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                action_provenance_rejected = True

            endpoint_shape_rejected = False
            bad_endpoint_shape = dict(state)
            bad_endpoint_shape["planner_expected_endpoint_pose"] = (
                state["planner_expected_endpoint_pose"][:, :-1].clone())
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_endpoint_shape,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                endpoint_shape_rejected = True

            mask_dtype_rejected = False
            bad_mask_dtype = dict(state)
            bad_mask_dtype["joint_observed"] = torch.ones_like(joint_state)
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_mask_dtype,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except TypeError:
                mask_dtype_rejected = True

            invalid_joint_health_rejected = False
            bad_joint_health = dict(state)
            bad_joint_health["joint_observed"] = joint_observed.clone()
            bad_joint_health["joint_healthy"] = joint_observed.clone()
            bad_joint_health["joint_observed"][0, 0] = False
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_joint_health,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                invalid_joint_health_rejected = True

            invalid_endpoint_observation_rejected = False
            bad_endpoint_observation = dict(state)
            bad_endpoint_observation["endpoint_observed"] = (
                endpoint_observed.clone())
            bad_endpoint_observation["endpoint_healthy"] = (
                endpoint_observed.clone())
            bad_endpoint_observation["endpoint_observed"][0, 0] = False
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_endpoint_observation,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                invalid_endpoint_observation_rejected = True

            unavailable_node_pose_rejected = False
            bad_node_pose = dict(state)
            bad_node_pose["node_pose_world"] = node_pose.clone()
            bad_node_pose["node_pose_world"][0, 0, 0] = 1.0
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_node_pose,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                unavailable_node_pose_rejected = True

            unavailable_node_twist_rejected = False
            bad_node_twist = dict(state)
            bad_node_twist["node_twist_world"] = node_twist.clone()
            bad_node_twist["node_twist_world"][0, 0, 0] = 1.0
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_node_twist,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                unavailable_node_twist_rejected = True

            invalid_node_observation_rejected = False
            bad_node_observation = dict(state)
            bad_node_observation["node_observed"] = node_observed.clone()
            bad_node_observation["node_healthy"] = node_observed.clone()
            bad_node_observation["node_healthy"][0, 0] = True
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_node_observation,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                invalid_node_observation_rejected = True

            absent_observer_valid_rejected = False
            bad_absent_observer_valid = dict(state)
            bad_absent_observer_valid["observer_pose_valid"] = torch.tensor(
                [True, False])
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_absent_observer_valid,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=robot_morphology)
            except ValueError:
                absent_observer_valid_rejected = True

            unavailable_observer_pose_rejected = False
            bad_observer_pose = dict(observer_state)
            bad_observer_pose["observer_pose_world"] = observer_pose_world.clone()
            bad_observer_pose["observer_pose_world"][1, 0] = 1.0
            try:
                ManagerFunction.TensorizeRobotState(
                    bad_observer_pose,
                    torch.device("cpu"),
                    batched=True,
                    robotContract=observer_morphology)
            except ValueError:
                unavailable_observer_pose_rejected = True

            observer_calibration_rejected = False
            try:
                ValidateRobotObserverCalibration(
                    observer_morphology,
                    "wrong-calibration",
                    "camera_optical")
            except ValueError:
                observer_calibration_rejected = True

            observer_frame_rejected = False
            try:
                ValidateRobotObserverCalibration(
                    observer_morphology,
                    "observer-calibration",
                    "wrong-frame")
            except ValueError:
                observer_frame_rejected = True

            sensor_frame_name = "sensor-frame-a"
            sensor_manifest = ExpectedOfflineSensorManifest(
                "calibration-a",
                robot_morphology,
                sensor_frame_name)
            ValidateOfflineSensorManifest(
                sensor_manifest,
                "calibration-a",
                robot_morphology,
                sensor_frame_name)
            calibration_mismatch_rejected = False
            try:
                ValidateOfflineSensorManifest(
                    sensor_manifest,
                    "calibration-b",
                    robot_morphology,
                    sensor_frame_name)
            except ValueError:
                calibration_mismatch_rejected = True

            motion_frame_rejected = False
            wrong_motion_frame = dict(sensor_manifest)
            wrong_motion_frame["object_motion_frame"] = "world"
            try:
                ValidateOfflineSensorManifest(
                    wrong_motion_frame,
                    "calibration-a",
                    robot_morphology,
                    sensor_frame_name)
            except ValueError:
                motion_frame_rejected = True

            old_manifest_rejected = False
            old_manifest = dict(sensor_manifest)
            old_manifest["schema_version"] = 2
            try:
                ValidateOfflineSensorManifest(
                    old_manifest,
                    "calibration-a",
                    robot_morphology,
                    sensor_frame_name)
            except ValueError:
                old_manifest_rejected = True

            ontology_order_rejected = False
            wrong_ontology_order = dict(sensor_manifest)
            wrong_ontology_order["entity_realm_names"] = list(reversed(
                sensor_manifest["entity_realm_names"]))
            try:
                ValidateOfflineSensorManifest(
                    wrong_ontology_order,
                    "calibration-a",
                    robot_morphology,
                    sensor_frame_name)
            except ValueError:
                ontology_order_rejected = True

            ok = (
                tuple(converted["base_orientation_world"].shape)
                == (batch_size, 4)
                and tuple(converted["endpoint_pose"].shape) == (
                    batch_size,
                    robot_morphology.endpoint_count,
                    ModuleDim.DecisionEndpointPoseDim)
                and tuple(converted["joint_position"].shape) == (
                    batch_size,
                    robot_morphology.joint_dof_count)
                and tuple(converted["joint_observed"].shape) == (
                    batch_size,
                    robot_morphology.joint_dof_count)
                and tuple(converted["node_pose_world"].shape) == (
                    batch_size,
                    robot_morphology.node_count,
                    7)
                and tuple(converted["node_twist_world"].shape) == (
                    batch_size,
                    robot_morphology.node_count,
                    6)
                and tuple(converted["node_observed"].shape) == (
                    batch_size,
                    robot_morphology.node_count)
                and tuple(converted_unbatched["node_pose_world"].shape)
                == (robot_morphology.node_count, 7)
                and tuple(converted_unbatched["node_twist_world"].shape)
                == (robot_morphology.node_count, 6)
                and tuple(converted_unbatched["observer_pose_world"].shape)
                == (7,)
                and tuple(converted_unbatched["observer_pose_valid"].shape)
                == ()
                and not bool(converted["node_observed"].any().item())
                and torch.all(converted["node_pose_world"][..., 6]
                    == 1.0).item()
                and torch.count_nonzero(
                    converted["node_pose_world"][..., :6]).item() == 0
                and torch.count_nonzero(
                    converted["node_twist_world"]).item() == 0
                and converted_node_state["node_observed"][0, 0].item()
                and converted_node_state["node_pose_world"][0, 0, 0].item()
                == 0.25
                and converted_node_state["node_twist_world"][0, 0, 1].item()
                == 0.5
                and tuple(converted["observer_pose_world"].shape)
                == (batch_size, 7)
                and tuple(converted["observer_pose_valid"].shape)
                == (batch_size,)
                and not bool(converted["observer_pose_valid"].any().item())
                and converted_observer_state["observer_pose_valid"][0].item()
                and not converted_observer_state[
                    "observer_pose_valid"][1].item()
                and torch.allclose(
                    converted_observer_state["observer_pose_world"][0, :3],
                    torch.tensor([0.1, 0.2, 0.3]))
                and observer_morphology.observer_valid
                and not observer_morphology.observer_controllable
                and tuple(converted["gravity_direction_world"].shape)
                == (batch_size, 3)
                and converted["executed_action_id"].dtype == torch.long
                and missing_rejected
                and base_quaternion_rejected
                and gravity_rejected
                and bool_rejected
                and planner_range_rejected
                and fractional_flag_rejected
                and planner_conflict_rejected
                and action_provenance_rejected
                and endpoint_shape_rejected
                and mask_dtype_rejected
                and invalid_joint_health_rejected
                and invalid_endpoint_observation_rejected
                and unavailable_node_pose_rejected
                and unavailable_node_twist_rejected
                and invalid_node_observation_rejected
                and absent_observer_valid_rejected
                and unavailable_observer_pose_rejected
                and observer_calibration_rejected
                and observer_frame_rejected
                and robot_morphology.joint_dof_count == 22
                and not robot_morphology.observer_valid
                and calibration_mismatch_rejected
                and motion_frame_rejected
                and old_manifest_rejected
                and ontology_order_rejected)
            print(
                f"Manager RobotState physical reference contract "
                f"{'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager RobotState physical reference contract error: {e}")
            return False

    def TestVirtualJointStateTensorization(self) -> bool:
        try:
            fixed = self.MakeRobotMorphology()
            variants = {"fixed": fixed}
            for joint_type in ("planar", "floating"):
                source = fixed.ToJson()
                source["srdf"]["virtual_joints"][0]["type"] = joint_type
                variants[joint_type] = RobotMorphologyModule().FromJson(source)

            def make_state(contract: CompiledRobotMorphology) -> Dict[str, torch.Tensor]:
                joint_state = torch.arange(
                    contract.joint_dof_count,
                    dtype=torch.float32)
                joint_observed = torch.ones(
                    contract.joint_dof_count,
                    dtype=torch.bool)
                node_pose = torch.zeros(contract.node_count, 7)
                node_pose[..., 6] = 1.0
                node_observed = torch.zeros(
                    contract.node_count,
                    dtype=torch.bool)
                endpoint_pose = torch.zeros(
                    contract.endpoint_count,
                    ModuleDim.DecisionEndpointPoseDim)
                endpoint_pose[..., 6] = 1.0
                endpoint_observed = torch.zeros(
                    contract.endpoint_count,
                    dtype=torch.bool)
                return {
                    "joint_position": joint_state,
                    "joint_velocity": joint_state.clone(),
                    "joint_effort": joint_state.clone(),
                    "joint_observed": joint_observed,
                    "joint_healthy": joint_observed.clone(),
                    "joint_controllable": (
                        contract.joint_variable_commandable.clone()),
                    "node_pose_world": node_pose,
                    "node_twist_world": torch.zeros(contract.node_count, 6),
                    "node_observed": node_observed,
                    "node_healthy": node_observed.clone(),
                    "endpoint_pose": endpoint_pose,
                    "endpoint_observed": endpoint_observed,
                    "endpoint_healthy": endpoint_observed.clone(),
                    "endpoint_controllable": endpoint_observed.clone(),
                    "observer_pose_world": torch.tensor([
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                    "observer_pose_valid": torch.tensor(False),
                    "base_orientation_world": torch.tensor([
                        0.0, 0.0, 0.0, 1.0]),
                    "gravity_direction_world": torch.tensor([
                        0.0, 0.0, -1.0]),
                    "planner_expected_endpoint_pose": endpoint_pose.clone(),
                    "planner_progress": torch.tensor(0.0),
                    "planner_tracking_error": torch.tensor(0.0),
                    "planner_executing": torch.tensor(0.0),
                    "planner_reached": torch.tensor(0.0),
                    "planner_failed": torch.tensor(0.0),
                    "model_command_executed": torch.tensor(0.0),
                    "executed_action_id": torch.tensor(0),
                }

            converted = {
                name: ManagerFunction.TensorizeRobotState(
                    make_state(contract),
                    torch.device("cpu"),
                    batched=False,
                    robotContract=contract)
                for name, contract in variants.items()
            }
            floating_state = make_state(variants["floating"])
            floating_metadata = ExpectedRobotStateWireMetadata(
                variants["floating"])
            floating_packet = {
                "schema_version": ROBOT_STATE_WIRE_SCHEMA_VERSION,
                "stream_id": "floating-stream",
                "sequence_index": 0,
                "frame_id": "floating-frame",
                "calibration_id": "test-camera",
                "world_frame_id": "world",
                **floating_metadata,
                **{
                    name: value.tolist()
                    for name, value in floating_state.items()},
            }
            ValidateRobotStateWirePacket(
                floating_packet,
                "test-camera",
                variants["floating"])
            truncated_packet = dict(floating_packet)
            truncated_packet["joint_position"] = floating_packet[
                "joint_position"][1:]
            truncated_wire_rejected = False
            try:
                ValidateRobotStateWirePacket(
                    truncated_packet,
                    "test-camera",
                    variants["floating"])
            except ValueError:
                truncated_wire_rejected = True
            missing_virtual_state = make_state(variants["floating"])
            for name in (
                "joint_position",
                "joint_velocity",
                "joint_effort",
                "joint_observed",
                "joint_healthy",
                "joint_controllable",
            ):
                missing_virtual_state[name] = missing_virtual_state[name][6:]
            missing_virtual_state_rejected = False
            try:
                ManagerFunction.TensorizeRobotState(
                    missing_virtual_state,
                    torch.device("cpu"),
                    batched=False,
                    robotContract=variants["floating"])
            except ValueError:
                missing_virtual_state_rejected = True
            missing_observer_contract = dict(vars(fixed))
            missing_observer_contract.pop("observer_attachment_name")
            missing_observer_contract_rejected = False
            try:
                ValidateRobotTensorContract(SimpleNamespace(
                    **missing_observer_contract))
            except TypeError:
                missing_observer_contract_rejected = True
            ok = (
                fixed.joint_count == fixed.node_count
                and fixed.joint_dof_count == 22
                and variants["planar"].joint_dof_count == 25
                and variants["floating"].joint_dof_count == 28
                and len({
                    fixed.model_contract_id,
                    variants["planar"].model_contract_id,
                    variants["floating"].model_contract_id,
                }) == 3
                and tuple(converted["fixed"]["joint_position"].shape) == (22,)
                and tuple(converted["planar"]["joint_position"].shape) == (25,)
                and tuple(converted["floating"]["joint_position"].shape) == (28,)
                and floating_metadata["joint_names"][0] == "virtual_joint"
                and floating_metadata["joint_type"][0]
                == ModuleDim.RobotJointTypeNames.index("floating")
                and floating_metadata["joint_parent_node"][0] == -1
                and floating_metadata["joint_child_node"][0] == 0
                and floating_metadata["joint_variable_local_index"][:6]
                == list(range(6))
                and torch.equal(
                    converted["floating"]["joint_position"][:6],
                    torch.arange(6, dtype=torch.float32))
                and bool(converted["floating"]["joint_observed"][:6].all().item())
                and bool(converted["floating"]["joint_healthy"][:6].all().item())
                and not bool(converted["floating"][
                    "joint_controllable"][:6].any().item())
                and truncated_wire_rejected
                and missing_virtual_state_rejected
                and missing_observer_contract_rejected)
            print(
                f"Manager virtual joint state tensorization "
                f"{'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager virtual joint state tensorization error: {e}")
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
                and "camera_intrinsics" not in pack
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
            pack = DataPreprocessor.ConvertRobotInputs(
                imgs=rgb,
                reward=torch.zeros(2),
                done=torch.zeros(2),
                depths=depth,
                depthValids=depth_valid,
                needVisualState=False)
            wrong_lattice_rejected = False
            try:
                DataPreprocessor.ConvertRobotInputs(
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
                DataPreprocessor.ConvertRobotInputs(
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
                DataPreprocessor.ConvertRobotInputs(
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
                DataPreprocessor.ConvertRobotInputs(
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
                and "camera_intrinsics" not in pack)
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
            "OcrCheckpointStrictContract": self.TestOcrCheckpointStrictContract(),
            "SequentialLoaderRejectsEmptyRecurrentSplit": self.TestSequentialLoaderRejectsEmptyRecurrentSplit(),
            "RobotStatePhysicalReferenceContract": self.TestRobotStatePhysicalReferenceContract(),
            "VirtualJointStateTensorization": self.TestVirtualJointStateTensorization(),
            "CppDecisionWireContract": self.TestCppDecisionWireContract(),
            "SingleFramePreprocessContract": self.TestSingleFramePreprocessContract(),
            "OfflinePreprocessStrictContract": self.TestOfflinePreprocessStrictContract(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nManager tests: {passed}/{len(results)} passed.")
        return results


class AgentHandle:
    def __init__(
        self,
        calibration: CameraCalibration,
        *,
        robotMorphology: Optional[CompiledRobotMorphology] = None,
        robotUrdfPath: str = BasicParameters.ROBOT_URDF_PATH,
        robotSrdfPath: str = BasicParameters.ROBOT_SRDF_PATH,
        robotSemanticPath: Optional[str] = BasicParameters.ROBOT_SEMANTIC_PATH,
        brainParameterPath: str = BasicParameters.MODULEPARAMETER_PATH,
        device: Optional[Union[str, torch.device]] = None,
        seqLen: int = BasicParameters.IMAGE_SEQ_LEN,
        usePlanner: bool = True,
        prioritizeExtStr: bool = True,
        saveModuleMessagerOutput: bool = True,):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.robot_morphology = (
            robotMorphology
            if robotMorphology is not None
            else ManagerFunction.LoadRobotMorphology(
                robotUrdfPath,
                robotSrdfPath,
                robotSemanticPath,
                observerFrameName=calibration.frame_name,
                observerCalibrationId=calibration.calibration_id))
        ValidateRobotTensorContract(self.robot_morphology)
        ValidateRobotObserverCalibration(
            self.robot_morphology,
            calibration.calibration_id,
            calibration.frame_name)

        parameter_path = str(brainParameterPath).strip()
        if parameter_path == "":
            raise ValueError("brainParameterPath must not be empty")

        (
            resolved_model_path,
            resolved_world_memory_path,
            resolved_memory_path,
        ) = ManagerFunction.ResolveDeploymentArtifactPaths(
            parameter_path,
            calibrationId=calibration.calibration_id,
            robotContract=self.robot_morphology)
        resolved_path = Path(resolved_model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"brain parameter file not found: {resolved_path}")

        self.brain = BrainCore(
            calibration=calibration,
            robotMorphology=self.robot_morphology,
            device=self.device,
            seqLen=seqLen,
            prioritizeExtStr=prioritizeExtStr,
            plasticOnlineLearning=False,
            usePlanner=usePlanner,
            saveModuleMessagerOutput=saveModuleMessagerOutput,)

        self.agent = Agent(
            self.brain,
            isTrain=False,
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
        perceptionTargets: Optional[Dict[str, torch.Tensor]] = None,
        robotState: RobotState,):
        return self.agent.Act(AgentActInput(
            frame=frame,
            text_ext=textExt,
            reward=reward,
            done=done,
            sample_actions=sampleActions,
            deterministic_actor=deterministicActor,
            depth=depth,
            depth_valid=depthValid,
            perception_targets=perceptionTargets,
            robot_state=robotState,
            text_trust=textTrust))
