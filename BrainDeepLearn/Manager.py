from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import threading
import random
import time
import json

import numpy as np
import torch
import torch.nn as nn
import shutil
import traceback
import os

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
from AGICore import Agent, BRAIN_RUNTIME_SCHEMA_VERSION, BrainCore, TestAGICoreMTool
from Config import BasicParameters
from CoreTypes import (
    AgentActInput,
    ROBOT_STATE_WIRE_SCHEMA_VERSION,
    SENSOR_PACKET_WIRE_SCHEMA_VERSION,
    RobotState,
    TEXT_TRUST_OCR_OBSERVED,
    TEXT_TRUST_OPERATOR_COMMAND,
    TEXT_TRUST_UNSAFE_EXTERNAL)
from ModuleMessagerManager import ModuleDim
from FunctionTools import SynchronizeGrowableLoRATopologyForFullLoad


TRAIN_CHECKPOINT_SCHEMA_VERSION = BRAIN_RUNTIME_SCHEMA_VERSION

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
        self.steps = len(dataset) // self.batch_size

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        for t in range(self.steps):
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
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.controller = ModuleController()

        self.br_thread: Optional[threading.Thread] = None
        self.message_thread: Optional[threading.Thread] = None
        self.is_begin = False
        self.overrideCheckpointWithModuleParams = True
        self.agent_handle: Optional[AgentHandle] = None
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
    def NormalizeTrainStage(trainStage: str) -> str:
        stage = str(trainStage).strip().lower()
        if stage not in ("full", "world", "policy"):
            raise ValueError("trainStage must be one of: full/world/policy")
        return stage

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
        training_rng = self.CaptureRngState()
        training_modes = [
            (module, bool(module.training))
            for module in brain.modules()]
        training_graph = brain.SuspendTransientTrainingGraph()
        try:
            return evaluate()
        finally:
            try:
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
        
    def InitAgentHandle(self):
        self.agent_handle = AgentHandle()
        return True

    def SetJsonQueue(self, queue):
        self.json_queue = queue
        return True

    def SetParameterReceiver(self, reward = None, done = None, textExt = None):
        self.controller.SetParameterReceiver(reward=reward, done=done, textExt=textExt)
        return True

    def SetCameraIntrinsics(
        self,
        cameraIntrinsics: Union[np.ndarray, torch.Tensor],
        sourceSize: Tuple[int, int]) -> None:
        """One-shot calibration: scales K from (cameraResH, cameraResW) to the
        perception input grid and writes it into the perception's K buffer.
        Subsequent AgentHandleForward() calls no longer take cameraIntrinsics."""
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")
        self.agent_handle.SetCameraIntrinsics(cameraIntrinsics, sourceSize=sourceSize)

    def AgentHandleForward(
        self,
        cameraIndex: int = 0,
        reward: Optional[float] = None,
        done: Optional[float] = None,
        textExt: Optional[List[Optional[str]]] = None,
        textTrust: Optional[List[str]] = None,
        sampleActions: bool = True,
        deterministicActor: bool = False,
        *,
        depthBitmap: Union[List[Any], np.ndarray, torch.Tensor],
        depthValid: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
        depthScaleMeters: float = 1.0,
        robotState: RobotState,):
        if self.agent_handle is None:
            raise RuntimeError("agent_handle has not been initialized")
        if iio is None:
            raise RuntimeError("imageio.v3 cant use")

        frame_np = iio.imread(f"<video{int(cameraIndex)}>", index=0)
        if frame_np is None:
            raise RuntimeError(f"cannot read frame from camera {int(cameraIndex)}")

        converted = DataPreprocessor.ConvertCppCameraFrame(
            bitmap=frame_np,
            reward=reward,
            done=done,
            depthBitmap=depthBitmap,
            depthValid=depthValid,
            depthScaleMeters=depthScaleMeters,
            device=self.device,
            needVisualState=False,)

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
            robotState=robotState,)
        return self.agent_handle.agent.UnpackActPacked(act_out)

    def AgentHandleForwardJson(
        self,
        cameraIndex: int,
        reward: Optional[float],
        done: Optional[float],
        sensorPacketJson: str,
        robotStateJson: str,) -> str:
        """C++ single-frame wire protocol.

        ``sensorPacketJson`` contains explicit text trust, action sampling mode,
        unbatched ``depth``, ``depth_valid`` and ``depth_scale_meters``.
        ``robotStateJson`` contains the unbatched fields defined by
        :class:`CoreTypes.RobotState`; this adapter adds the one-frame batch axis.
        The returned command is a proposal: this adapter neither validates
        hardware feasibility nor enforces its timeout budget.
        """
        sensor_packet = json.loads(sensorPacketJson)
        robot_packet = json.loads(robotStateJson)
        if int(sensor_packet["schema_version"]) != SENSOR_PACKET_WIRE_SCHEMA_VERSION:
            raise ValueError("unsupported sensor packet schema")
        if int(robot_packet["schema_version"]) != ROBOT_STATE_WIRE_SCHEMA_VERSION:
            raise ValueError("unsupported robot state packet schema")
        if tuple(robot_packet["endpoint_names"]) != ModuleDim.DecisionEndpointNames:
            raise ValueError("robot endpoint order does not match DecisionEndpointNames")
        if robot_packet["pose_frame"] != "world":
            raise ValueError("robot poses must use the world frame")
        if robot_packet["pose_unit"] != "meter":
            raise ValueError("robot poses must use metres")
        if robot_packet["quaternion_order"] != "xyzw":
            raise ValueError("robot pose quaternions must use XYZW order")
        text_ext = sensor_packet["text_ext"]
        text_trust = sensor_packet["text_trust"]
        if len(text_ext) != 1 or len(text_trust) != 1:
            raise ValueError("single-frame sensor text_ext/text_trust must have length 1")
        if text_trust[0] not in (
            TEXT_TRUST_OCR_OBSERVED,
            TEXT_TRUST_OPERATOR_COMMAND,
            TEXT_TRUST_UNSAFE_EXTERNAL,
        ):
            raise ValueError("unsupported text_trust value")

        def batched_tensor(name: str) -> torch.Tensor:
            return torch.as_tensor(
                robot_packet[name],
                device=self.device,
                dtype=torch.float32).unsqueeze(0)

        robot_state: RobotState = {
            "endpoint_pose": batched_tensor("endpoint_pose"),
            "camera_pose_world": batched_tensor("camera_pose_world"),
            "planner_expected_endpoint_pose": batched_tensor(
                "planner_expected_endpoint_pose"),
            "planner_progress": batched_tensor("planner_progress"),
            "planner_tracking_error": batched_tensor("planner_tracking_error"),
            "planner_executing": batched_tensor("planner_executing"),
            "planner_reached": batched_tensor("planner_reached"),
            "planner_failed": batched_tensor("planner_failed"),
            "model_command_executed": batched_tensor(
                "model_command_executed"),}
        if tuple(robot_state["endpoint_pose"].shape) != (
            1,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseDim,
        ):
            raise ValueError("endpoint_pose must have wire shape [13, 7]")
        if tuple(robot_state["planner_expected_endpoint_pose"].shape) != (
            1,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseDim,
        ):
            raise ValueError(
                "planner_expected_endpoint_pose must have wire shape [13, 7]")
        if tuple(robot_state["camera_pose_world"].shape) != (1, ModuleDim.PstPoseDim):
            raise ValueError("camera_pose_world must have wire shape [7]")
        scalar_fields = (
            "planner_progress",
            "planner_tracking_error",
            "planner_executing",
            "planner_reached",
            "planner_failed",
            "model_command_executed")
        if any(tuple(robot_state[name].shape) != (1,) for name in scalar_fields):
            raise ValueError("planner and provenance fields must be wire scalars")
        return self.AgentHandleForward(
            cameraIndex,
            reward,
            done,
            textExt=text_ext,
            textTrust=text_trust,
            sampleActions=bool(sensor_packet["sample_actions"]),
            deterministicActor=bool(sensor_packet["deterministic_actor"]),
            depthBitmap=sensor_packet["depth"],
            depthValid=sensor_packet["depth_valid"],
            depthScaleMeters=float(sensor_packet["depth_scale_meters"]),
            robotState=robot_state)

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
        saveEverySampleCount = 2000,):
        ckpt_path = BasicParameters.CKPT_PATH_TEST if isTest else BasicParameters.CKPT_PATH_TRAIN
        out_path = BasicParameters.MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.MODULEPARAMETER_PATH
        wm_mem_path = BasicParameters.WORLD_MEMORY_PATH_TEST if isTest else BasicParameters.WORLD_MEMORY_PATH
        mem_mem_path = BasicParameters.MEMORY_MEMORY_PATH_TEST if isTest else BasicParameters.MEMORY_MEMORY_PATH
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
            texts_dir = Path(BasicParameters.DATA_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            reward_dir = root / "reward"
            done_dir = root / "done"
            depth_dir = root / "depth"
            texts_dir = root / "texts"

        required_dirs = [frames_dir, reward_dir, done_dir, depth_dir]
        if not all(p.exists() for p in required_dirs):
            return False

        counts = [
            len(sorted(frames_dir.glob("*.png"))),
            len(sorted(reward_dir.glob("*.npy"))),
            len(sorted(done_dir.glob("*.npy"))),
            len(DataPreprocessor.ListDepthFiles(depth_dir)),]
        if counts[0] == 0 or len(set(counts)) != 1:
            return False

        if texts_dir.exists():
            text_count = len(sorted(texts_dir.glob("*.txt")))
            if text_count not in (0, counts[0]):
                return False
        return True

    def BuildDefaultRobotState(
        self,
        batchSize: int,
        *,
        device: torch.device,
        dtype: torch.dtype) -> RobotState:
        endpoint_pose = torch.zeros(
            int(batchSize),
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseDim,
            device=device,
            dtype=dtype)
        endpoint_pose[..., 6] = 1.0
        camera_pose_world = torch.zeros(int(batchSize), ModuleDim.PstPoseDim, device=device, dtype=dtype)
        camera_pose_world[:, 6] = 1.0
        planner_scalar = torch.zeros(int(batchSize), device=device, dtype=dtype)
        return {
            "endpoint_pose": endpoint_pose,
            "camera_pose_world": camera_pose_world,
            "planner_expected_endpoint_pose": endpoint_pose.clone(),
            "planner_progress": planner_scalar,
            "planner_tracking_error": planner_scalar,
            "planner_executing": planner_scalar,
            "planner_reached": planner_scalar,
            "planner_failed": planner_scalar,
            "model_command_executed": torch.zeros_like(planner_scalar),}

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

    def SaveModuleParameters(self, brain: BrainCore, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        brain_state = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in brain.state_dict().items()}

        torch.save({
            "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
            "brain": brain_state,}, str(out_path))

    def SaveOCRParameters(self, engine: OCREngineExtractor, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ocr_state = {k: v.detach().cpu() for k, v in engine.state_dict().items()}
        brain_state = {f"OCR.{k}": v for k, v in ocr_state.items()}
        ocr_meta = engine.OcrMetadata()

        torch.save({
            "ocr": ocr_state,
            "brain": brain_state,
            "vocab": ocr_meta["vocab"],
            "ocr_meta": ocr_meta,
            "legacy_prefixes": ocr_meta["legacy_prefixes"],
            "addon_cfg": ocr_meta["addon_cfg"],}, str(out_path))

    def SaveOCRRecognizerParameters(self, engine: OCREngineExtractor, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rec_state = {k: v.detach().cpu() for k, v in engine.recognizer.state_dict().items()}
        ocr_state = {f"recognizer.{k}": v for k, v in rec_state.items()}
        brain_state = {f"OCR.recognizer.{k}": v for k, v in rec_state.items()}
        ocr_meta = engine.OcrMetadata()

        torch.save({
            "recognizer": rec_state,
            "ocr": ocr_state,
            "brain": brain_state,
            "vocab": ocr_meta["vocab"],
            "ocr_meta": ocr_meta,
            "addon_cfg": ocr_meta["addon_cfg"],}, str(out_path))

    def CaptureRngState(self) -> Dict[str, Any]:
        state = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),}
        if torch.cuda.is_available():
            state["cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    def RestoreRngState(self, state: Dict[str, Any]) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        np.random.set_state(state["numpy"])
        if torch.cuda.is_available() and "cuda_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_all"])

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
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"Safe mode loading failed: {e}, try the normal mode")
            return torch.load(path, map_location=self.device)

    def LoadBrainWeights(
        self,
        brain: BrainCore,
        path: str,
        *,
        agent: Optional[Agent] = None,) -> None:
        payload = self.LoadTorchPayload(path)

        if isinstance(payload, dict) and "brain" in payload:
            checkpoint_version = int(payload.get("schema_version", 1))
            if checkpoint_version not in (1, 2, 3, TRAIN_CHECKPOINT_SCHEMA_VERSION):
                raise ValueError(
                    f"unsupported brain parameter schema {checkpoint_version}")
            brain_state = payload["brain"]
        elif isinstance(payload, dict):
            brain_state = payload
        else:
            raise TypeError(f"checkpoint {path} has invalid brain weights payload")

        SynchronizeGrowableLoRATopologyForFullLoad(brain, brain_state)
        brain_state = brain.UpgradeDecisionStateDict(brain_state)
        brain.ResizeStateBuffersForLoad(brain_state)
        brain.load_state_dict(brain_state, strict=False)
        if agent is not None:
            agent.SyncTrainableOptimizers()
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
            print(f"[{logPrefix}] parameter override skipped: parameter path is empty")
            return False

        override_path = Path(parameterPath)
        if not override_path.exists():
            print(f"[{logPrefix}] parameter override skipped, file not found: {override_path}")
            return False

        loadFn(str(override_path))
        print(f"[{logPrefix}] checkpoint weights overridden from parameter file: {override_path}")
        return True

    def FilterLoadableStateDict(
        self,
        module: nn.Module,
        stateDict: Dict[str, Any],
        *,
        logPrefix: str,) -> Dict[str, Any]:
        current = module.state_dict()
        filtered: Dict[str, Any] = {}
        skipped: List[str] = []

        for k, v in stateDict.items():
            if k in current and isinstance(v, torch.Tensor):
                if tuple(current[k].shape) != tuple(v.shape):
                    skipped.append(str(k))
                    continue
            filtered[k] = v

        if skipped:
            preview = ", ".join(skipped[:8])
            more = "" if len(skipped) <= 8 else f", ... +{len(skipped) - 8}"
            print(f"[{logPrefix}] skipped shape-mismatched keys: {preview}{more}")

        return filtered

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
        payload = torch.load(path, map_location=self.device)

        ocr_state = None
        if isinstance(payload, dict):
            if "ocr" in payload:
                ocr_state = payload["ocr"]
            elif "brain" in payload:
                ocr_state = {
                    k[len("OCR."):]: v
                    for k, v in payload["brain"].items()
                    if k.startswith("OCR.")}
            elif all(str(k).startswith(("backbone.", "dbHead.", "recognizer.")) for k in payload.keys()):
                ocr_state = payload

        if not ocr_state:
            raise KeyError(f"checkpoint {path} has no OCR weights")

        ocr_state = self.FilterLoadableStateDict(engine, ocr_state, logPrefix="LoadOCR")
        engine.load_state_dict(ocr_state, strict=False)

    def LoadRecognizerWeightsIntoEngine(self, engine: OCREngineExtractor, path: str) -> None:
        payload = torch.load(path, map_location=self.device)

        rec_state = None
        if isinstance(payload, dict):
            if "recognizer" in payload:
                rec_state = payload["recognizer"]
            elif "ocr" in payload:
                rec_state = {
                    k[len("recognizer."):]: v
                    for k, v in payload["ocr"].items()
                    if k.startswith("recognizer.")}
            elif "brain" in payload:
                rec_state = {
                    k[len("OCR.recognizer."):]: v
                    for k, v in payload["brain"].items()
                    if k.startswith("OCR.recognizer.")}
            elif all(str(k).startswith("recognizer.") for k in payload.keys()):
                rec_state = {k[len("recognizer."):]: v for k, v in payload.items()}
            elif all(not str(k).startswith("backbone.") and not str(k).startswith("dbHead.") for k in payload.keys()):
                rec_state = payload

        if not rec_state:
            raise KeyError(f"checkpoint {path} has no recognizer weights")

        rec_state = self.FilterLoadableStateDict(engine.recognizer, rec_state, logPrefix="LoadOCRRec")
        engine.recognizer.load_state_dict(rec_state, strict=False)

    def LoadOCRCheckpoint(
        self,
        engine: OCREngineExtractor,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        path: str,
        *,
        allowOptimizerMismatch: bool = False,):
        ckpt = torch.load(path, map_location=self.device)

        if "ocr" in ckpt:
            ocr_state = self.FilterLoadableStateDict(engine, ckpt["ocr"], logPrefix="LoadOCRCkpt")
            engine.load_state_dict(ocr_state, strict=False)
        elif "brain" in ckpt:
            ocr_state = {
                k[len("OCR."):]: v
                for k, v in ckpt["brain"].items()
                if k.startswith("OCR.")}
            if len(ocr_state) == 0:
                raise KeyError(f"checkpoint {path} has no OCR weights")
            ocr_state = self.FilterLoadableStateDict(engine, ocr_state, logPrefix="LoadOCRCkpt")
            engine.load_state_dict(ocr_state, strict=False)
        else:
            raise KeyError(f"checkpoint {path} has no 'ocr' or 'brain' field")

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                if not allowOptimizerMismatch:
                    raise
                print("[LoadOCRCkpt] optimizer state skipped because parameter groups changed")

        if "rng" in ckpt:
            self.RestoreRngState(ckpt["rng"])

        train_ds = val_ds = test_ds = None
        if ckpt.get("train_indices") is not None:
            train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
            val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
            if ckpt.get("test_indices") is not None:
                test_ds = torch.utils.data.Subset(dataset, ckpt["test_indices"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        processed_sample_count_total = int(ckpt.get("processed_sample_count_total", 0))
        return start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds

    def LoadOCRRecognizerCheckpoint(
        self,
        engine: OCREngineExtractor,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        path: str,):
        ckpt = torch.load(path, map_location=self.device)

        if "recognizer" in ckpt:
            rec_state = self.FilterLoadableStateDict(engine.recognizer, ckpt["recognizer"], logPrefix="LoadOCRRecCkpt")
            engine.recognizer.load_state_dict(rec_state, strict=False)
        elif "ocr" in ckpt:
            rec_state = {
                k[len("recognizer."):]: v
                for k, v in ckpt["ocr"].items()
                if k.startswith("recognizer.")}
            if len(rec_state) == 0:
                raise KeyError(f"checkpoint {path} has no recognizer weights")
            rec_state = self.FilterLoadableStateDict(engine.recognizer, rec_state, logPrefix="LoadOCRRecCkpt")
            engine.recognizer.load_state_dict(rec_state, strict=False)
        elif "brain" in ckpt:
            rec_state = {
                k[len("OCR.recognizer."):]: v
                for k, v in ckpt["brain"].items()
                if k.startswith("OCR.recognizer.")}
            if len(rec_state) == 0:
                raise KeyError(f"checkpoint {path} has no recognizer weights")
            rec_state = self.FilterLoadableStateDict(engine.recognizer, rec_state, logPrefix="LoadOCRRecCkpt")
            engine.recognizer.load_state_dict(rec_state, strict=False)
        else:
            raise KeyError(f"checkpoint {path} has no recognizer field")

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                print("[LoadOCRRecCkpt] optimizer state skipped because parameter groups changed")

        if "rng" in ckpt:
            self.RestoreRngState(ckpt["rng"])

        train_ds = val_ds = test_ds = None
        if ckpt.get("train_indices") is not None:
            train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
            val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
            if ckpt.get("test_indices") is not None:
                test_ds = torch.utils.data.Subset(dataset, ckpt["test_indices"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        processed_sample_count_total = int(ckpt.get("processed_sample_count_total", 0))
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
        overrideCheckpointWithModuleParams: bool = True,):
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
                    allowOptimizerMismatch=True)
            else:
                if recognizerInitPath and Path(recognizerInitPath).exists():
                    self.LoadRecognizerWeightsIntoEngine(engine, recognizerInitPath)
                elif not trainRecognition:
                    init_candidates = [Path(outPath), Path(ckptPath)]
                    init_error = None
                    loaded = False
                    for init_path in init_candidates:
                        if not init_path.exists():
                            continue
                        try:
                            self.LoadOCRWeightsIntoEngine(engine, str(init_path))
                            loaded = True
                            break
                        except Exception as e:
                            init_error = e

                    if not loaded:
                        msg = (
                            "detect-only OCR training requires existing OCR weights "
                            "or recognizerInitPath to preserve recognizer parameters")
                        if init_error is not None:
                            raise RuntimeError(msg) from init_error
                        raise FileNotFoundError(msg)
                    
            self.ApplyParameterOverrideAfterResume(
                enabled=overrideCheckpointWithModuleParams,
                parameterPath=outPath,
                loadFn=lambda path: self.LoadOCRWeightsIntoEngine(engine, path),
                logPrefix="TrainOCR")
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
                ocr_meta = engine.OcrMetadata()
                return {
                    "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                    "epoch": int(epochValue),
                    "best_val": best_val,
                    "ocr": engine.state_dict(),
                    "vocab": ocr_meta["vocab"],
                    "ocr_meta": ocr_meta,
                    "legacy_prefixes": ocr_meta["legacy_prefixes"],
                    "addon_cfg": ocr_meta["addon_cfg"],
                    "optimizer": optimizer.state_dict(),
                    "train_indices": list(train_ds.indices) if hasattr(train_ds, "indices") else None,
                    "val_indices": list(val_ds.indices) if hasattr(val_ds, "indices") else None,
                    "test_indices": list(test_ds.indices) if hasattr(test_ds, "indices") else None,
                    "processed_sample_count_total": processed_sample_count_total,
                    "rng": self.CaptureRngState(),}

            def SaveOCRTrainingArtifacts(epochValue: int, *, logPeriodic: bool = False) -> None:
                self.SaveOCRParameters(engine, outPath)
                ckpt_dir = Path(ckptPath).parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(BuildOCRCheckpointPayload(epochValue), ckptPath)
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
                
            self.ApplyParameterOverrideAfterResume(
                enabled=overrideCheckpointWithModuleParams,
                parameterPath=outPath,
                loadFn=lambda path: self.LoadRecognizerWeightsIntoEngine(engine, path),
                logPrefix="TrainOCRRec")
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
                ocr_meta = engine.OcrMetadata()
                return {
                    "epoch": int(epochValue),
                    "best_val": best_val,
                    "recognizer": engine.recognizer.state_dict(),
                    "vocab": ocr_meta["vocab"],
                    "ocr_meta": ocr_meta,
                    "addon_cfg": ocr_meta["addon_cfg"],
                    "optimizer": optimizer.state_dict(),
                    "train_indices": list(train_ds.indices) if hasattr(train_ds, "indices") else None,
                    "val_indices": list(val_ds.indices) if hasattr(val_ds, "indices") else None,
                    "test_indices": list(test_ds.indices) if hasattr(test_ds, "indices") else None,
                    "processed_sample_count_total": processed_sample_count_total,
                    "rng": self.CaptureRngState(),}

            def SaveOCRRecognizerTrainingArtifacts(epochValue: int, *, logPeriodic: bool = False) -> None:
                self.SaveOCRRecognizerParameters(engine, outPath)
                ckpt_dir = Path(ckptPath).parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(BuildOCRRecognizerCheckpointPayload(epochValue), ckptPath)
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
    
            ds = OfflineGameDataset(isTest=isTest)

            brain = BrainCore(
                device=self.device,
                plasticHebbian=True,
                plasticOnlineLearning=onlineLearning,
                enablePerceptionSupervision=True)

            agent = Agent(brain, isTrain=True, device=self.device, worldMemoryPath=worldMemPath, memMemoryPath=memMemPath,)

            if onlineLearning:
                agent.UpdateWrappers(
                    self.TrainStageOnlineWrappers(brain, trainStage),
                    "autogrow")

            start_epoch = 0
            best_val = float("inf")
            processed_sample_count_total = 0
            train_ds = val_ds = test_ds = None

            testSplit = 0.1

            if resume and Path(ckptPath).exists():
                start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds = self.LoadCheckpoint(brain, agent, ds, ckptPath)
                
            self.ApplyParameterOverrideAfterResume(
                enabled=overrideCheckpointWithModuleParams,
                parameterPath=outPath,
                loadFn=lambda path: self.LoadBrainWeights(brain, path, agent=agent),
                logPrefix="Train")

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
            no_improve = 0

            def unpack_batch(batch):
                img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, action_b, synthetic_targets = batch

                ext_text_b = [
                    None if (t is None or str(t).strip() == "") else str(t)
                    for t in ext_text_b]

                return img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, action_b, synthetic_targets

            def move_target_batch(targets: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    name: value.to(self.device) if torch.is_tensor(value) else value
                    for name, value in targets.items()}

            def BuildTrainCheckpointPayload(epochValue: int) -> Dict[str, Any]:
                return {
                    "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                    "epoch": int(epochValue),
                    "best_val": best_val,
                    "brain": brain.state_dict(),
                    "opt_actor": agent.opt_actor.state_dict(),
                    "opt_critic": agent.opt_critic.state_dict(),
                    "opt_world": agent.opt_world.state_dict(),
                    "train_indices": list(train_ds.indices)
                    if hasattr(train_ds, "indices")
                    else None,
                    "val_indices": list(val_ds.indices)
                    if hasattr(val_ds, "indices")
                    else None,
                    "test_indices": list(test_ds.indices)
                    if hasattr(test_ds, "indices")
                    else None,
                    "processed_sample_count_total": processed_sample_count_total,
                    "rng": self.CaptureRngState(),
                    "buffers": brain.ExportBuffers(),}

            def SaveTrainArtifacts(epochValue: int, *, logPeriodic: bool = False) -> None:
                agent.SaveRuntimeMemories()
                self.SaveModuleParameters(brain, outPath)
                ckpt_dir = Path(ckptPath).parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(BuildTrainCheckpointPayload(epochValue), ckptPath)
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
                    break

                brain.train()
                epoch_loss = 0.0
                nb = 0

                agent.ResetBrainState(B=batchSize, isOnlineLearning=onlineLearning)

                for bi, batch in enumerate(train_dl, start=1):
                    img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, action_b, synthetic_targets_b = unpack_batch(batch)

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
                    robot_state_t = self.BuildDefaultRobotState(
                        int(frames.size(0)),
                        device=self.device,
                        dtype=frames.dtype)
                    synthetic_targets_t = move_target_batch(synthetic_targets_b)
                    action_t = action_b.to(self.device)
                    robot_state_t["endpoint_pose"] = action_t
                    robot_state_t["planner_expected_endpoint_pose"] = robot_state_t["endpoint_pose"]
                    if "camera_pose_world" in synthetic_targets_t:
                        robot_state_t["camera_pose_world"] = synthetic_targets_t["camera_pose_world"]
                    perception_targets = {
                        "depth": depth_t,
                        "depth_valid": depth_valid_t,}
                    perception_targets.update(synthetic_targets_t)
                    perception_targets["rgb"] = frames
                    perception_targets["depth"] = depth_t
                    perception_targets["depth_valid"] = depth_valid_t

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
                        text_trust=[TEXT_TRUST_OPERATOR_COMMAND for _ in range(len(ext_text_b))]))

                    if act_out is None:
                        continue

                    model_loss = act_out.loss
                    transport_delayed_loss = act_out.transport_delayed_loss
                    ocr_items = act_out.ocr

                    loss = model_loss

                    agent.opt_world.zero_grad(set_to_none=True)
                    agent.opt_critic.zero_grad(set_to_none=True)
                    agent.opt_actor.zero_grad(set_to_none=True)

                    transport_capture_delayed = {"captured": 0.0, "grad_norm": 0.0, "accum_steps": 0.0}
                    if critic_stage_enabled and transport_delayed_loss.requires_grad:
                        transport_delayed_loss.backward()
                        transport_capture_delayed = agent.CaptureCriticTransportGrad()

                    loss.backward()
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
                        SaveTrainArtifacts(ep, logPeriodic=True)

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

                avg_train = epoch_loss / max(1, nb)

                self.controller.SetStatus("training", f"Epoch {ep+1}/{epochs} done, avg_train={avg_train:.4f}", epoch=ep + 1, total_epochs=epochs,)

                if self.controller.ShouldStop():
                    break

                def eval_split(dl):
                    brain.eval()
                    agent.ResetBrainState(B=batchSize, isOnlineLearning=onlineLearning)
                    split_loss = 0.0
                    split_batches = 0

                    with torch.no_grad():
                        for batch in dl:
                            img_b, reward_b, done_b, depth_b, depth_valid_b, ext_text_b, action_b, synthetic_targets_b = unpack_batch(batch)

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
                            v_robot_state_t = self.BuildDefaultRobotState(
                                int(v_frames.size(0)),
                                device=self.device,
                                dtype=v_frames.dtype)
                            v_synthetic_targets_t = move_target_batch(synthetic_targets_b)
                            v_action_t = action_b.to(self.device)
                            v_robot_state_t["endpoint_pose"] = v_action_t
                            v_robot_state_t["planner_expected_endpoint_pose"] = v_robot_state_t["endpoint_pose"]
                            if "camera_pose_world" in v_synthetic_targets_t:
                                v_robot_state_t["camera_pose_world"] = v_synthetic_targets_t["camera_pose_world"]
                            v_perception_targets = {
                                "depth": v_depth_t,
                                "depth_valid": v_depth_valid_t,}
                            v_perception_targets.update(v_synthetic_targets_t)
                            v_perception_targets["rgb"] = v_frames
                            v_perception_targets["depth"] = v_depth_t
                            v_perception_targets["depth_valid"] = v_depth_valid_t

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
                                text_trust=[TEXT_TRUST_OPERATOR_COMMAND for _ in range(len(ext_text_b))]))
                            
                            if act_out is None:
                                continue

                            v_model_loss = act_out.loss

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
                    SaveTrainArtifacts(ep + 1)
                else:
                    no_improve += 1

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
        if "brain" not in raw:
            raise KeyError(f"checkpoint {ckpt_path} has no 'brain' field")

        params = {
            "schema_version": int(raw.get("schema_version", 1)),
            "brain": raw["brain"],}

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

        torch.save(params, str(out_path))
        print(f"[ExportParamsOnly] saved params to {out_path}")


    def LoadCheckpoint(self, brain: BrainCore, agent: Agent, dataset: Dataset, path: str = None):
        ckpt = torch.load(path,  map_location=self.device)
        checkpoint_version = int(ckpt.get("schema_version", 1))
        if checkpoint_version not in (1, 2, 3, TRAIN_CHECKPOINT_SCHEMA_VERSION):
            raise ValueError(
                f"unsupported training checkpoint schema {checkpoint_version}")
        brain_state = ckpt["brain"]
        SynchronizeGrowableLoRATopologyForFullLoad(brain, brain_state)
        brain_state = brain.UpgradeDecisionStateDict(brain_state)
        brain.ResizeStateBuffersForLoad(brain_state)
        brain.load_state_dict(
            brain_state,
            strict=checkpoint_version == TRAIN_CHECKPOINT_SCHEMA_VERSION)
        agent.SyncTrainableOptimizers()
        if checkpoint_version == TRAIN_CHECKPOINT_SCHEMA_VERSION:
            try:
                agent.opt_actor.load_state_dict(ckpt["opt_actor"])
            except ValueError:
                print("[LoadCheckpoint] opt_actor state skipped because parameter groups changed")
            try:
                agent.opt_critic.load_state_dict(ckpt["opt_critic"])
            except ValueError:
                print("[LoadCheckpoint] opt_critic state skipped because parameter groups changed")
            try:
                agent.opt_world.load_state_dict(ckpt["opt_world"])
            except ValueError:
                print("[LoadCheckpoint] opt_world state skipped because parameter groups changed")
        else:
            print("[LoadCheckpoint] legacy optimizer states skipped after schema migration")
        agent.ClearTransientCandidateOptimizerState()

        if "buffers" in ckpt:
            brain.ImportBuffers(ckpt["buffers"])

        if "rng" in ckpt:
            self.RestoreRngState(ckpt["rng"])

        train_ds = val_ds = test_ds = None
        if ckpt.get("train_indices") is not None:
            train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
            val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
            if ckpt.get("test_indices") is not None:
                test_ds = torch.utils.data.Subset(dataset, ckpt["test_indices"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        processed_sample_count_total = int(ckpt.get("processed_sample_count_total", 0))
        return start_epoch, best_val, processed_sample_count_total, train_ds, val_ds, test_ds

    def StartDeployment(self, cameraIndex: int = 0, useHebbian: bool = True):
        return self.StartBackgroundTask(
            self.DeployLoop,
            args=(cameraIndex,),
            kwargs={"useHebbian": useHebbian,})


    def DeployLoop(self, cameraIndex: int,* ,useHebbian: bool = True,):
        try:
            self.ResetControllerFlags()

            brain = BrainCore(
                device=self.device,
                plasticHebbian=useHebbian,
                plasticOnlineLearning=False,)

            model_path = BasicParameters.MODULEPARAMETER_PATH

            if os.path.exists(model_path):
                self.LoadBrainWeights(brain, model_path)
            else:
                msg = f"The module file is not exit: {model_path}"
                print(msg)
                self.controller.SetStatus("error", msg)
                return 

            brain.eval()

            agent = Agent(brain,isTrain=False,device=self.device,worldMemoryPath=BasicParameters.WORLD_MEMORY_PATH,memMemoryPath=BasicParameters.MEMORY_MEMORY_PATH,)
            agent.ResetBrainState(isOnlineLearning=False)

            if iio is None:
                raise RuntimeError("imageio.v3 cant use")

            self.controller.SetStatus("is_begin", "Deployment started", visual=self.controller.EmptyVisualStatus(touch=True))

            while not self.controller.ShouldStop():
                frame_np = iio.imread(f"<video{int(cameraIndex)}>", index=0)
                if frame_np is None:
                    raise RuntimeError(f"cannot read frame from camera {int(cameraIndex)}")

                visual_enabled = self.controller.IsVisualStateEnabled()
                frame_arr = np.asarray(frame_np)
                depth_np = np.ones(frame_arr.shape[:2], dtype=np.float32)
                depth_valid_np = np.ones(frame_arr.shape[:2], dtype=bool)
                pack = DataPreprocessor.ConvertRobotInputs(
                    imgs=frame_np,
                    reward=None,
                    done=None,
                    depths=depth_np,
                    depthValids=depth_valid_np,
                    device=self.device,
                    needVisualState=visual_enabled,)

                frames = pack["frames"]
                original_images = pack["original_images"]
                resize_meta = pack["resize_meta"]
                depth_t = pack["depths"]
                depth_valid_t = pack["depth_valid"]
                robot_state_t = self.BuildDefaultRobotState(
                    int(frames.size(0)),
                    device=self.device,
                    dtype=frames.dtype)

                if self.controller.ShouldResetHebbian():
                    agent.ResetHebbianMemory()
                    self.controller.RequestCancelResetHebbian()

                parameters = self.controller.GetParameterReceiver()

                reward_param = parameters["reward"]
                done_param = parameters["done"]
                text_param = parameters["textExt"]

                reward_tensor = None
                if reward_param is not None:
                    reward_tensor = torch.tensor([[float(reward_param)]], dtype=torch.float32, device=self.device)

                done_tensor = None
                if done_param is not None:
                    done_tensor = torch.tensor([[float(done_param)]], dtype=torch.float32, device=self.device)

                text_ext = None
                if text_param is not None:
                    if isinstance(text_param, (list, tuple)):
                        text_ext = [None if (item is None or str(item).strip() == "") else str(item) for item in text_param]
                    else:
                        text_value = str(text_param).strip()
                        text_ext = [None if text_value == "" else text_value]

                act_out = agent.Act(AgentActInput(
                    frame=frames,
                    text_ext=text_ext,
                    reward=reward_tensor,
                    done=done_tensor,
                    sample_actions=True,
                    deterministic_actor=True,
                    depth=depth_t,
                    depth_valid=depth_valid_t,
                    robot_state=robot_state_t,
                    text_trust=(
                        [TEXT_TRUST_OPERATOR_COMMAND for _ in range(len(text_ext))]
                        if text_ext is not None else None)))
                if act_out is None:
                    continue

                act_json = agent.UnpackActPacked(act_out)
                if self.json_queue is not None:
                    self.json_queue.clearandpush(act_json)
               
                status_kwargs = {}

                if visual_enabled and original_images:
                    ocr_items = act_out.ocr
                    deploy_items = (ocr_items[0] if (ocr_items is not None and len(ocr_items) > 0) else [])
                    deploy_texts = [
                        str(item.get("text", "")).strip()
                        for item in deploy_items
                        if str(item.get("text", "")).strip() != ""]
                    resize_meta_0 = resize_meta[0] if resize_meta else None
                    visual_payload = None

                    visual_payload = self.BuildVisualPayload(
                        original_images[0],
                        ocrTexts=deploy_texts,
                        ocrItems=deploy_items,
                        resizeMeta=resize_meta_0,
                        title="Deploy",
                        extraLines=deploy_texts,)

                    if visual_payload is not None:
                        status_kwargs["visual"] = visual_payload

                self.controller.SetStatus("is_begin", act_json, **status_kwargs)

            self.controller.SetStatus("stopped", "Deployment stopped")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Deployment error: {e}", trace=tb)
        finally:
            self.is_begin = False



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
                (root / "texts").mkdir(parents=True, exist_ok=True)

                H, W = BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE

                templates = ["move left", "move right", "move forward", "move back",
                            "use skill", "defend", "attack", "pickup item",
                            "open menu", "retreat",]

                for i in range(nSamples):
                    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
                    iio.imwrite(str(root / "frames" / f"{i:05d}.png"), img)

                    depth = rng.uniform(0.1, 2.0, size=(H, W)).astype(np.float32)
                    np.save(str(root / "depth" / f"{i:05d}.npy"), depth)

                    reward = rng.normal(loc=0.0, scale=2.0, size=(1,)).astype(np.float32)
                    np.save(str(root / "reward" / f"{i:05d}.npy"), reward)

                    done = rng.normal(loc=0.0, scale=2.0, size=(1,)).astype(np.float32)
                    np.save(str(root / "done" / f"{i:05d}.npy"), done)

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

            self.TrainLoop(epochs, batchSize, valSplit, False, onlineLearning, isTest=True, worldMemPath=BasicParameters.WORLD_MEMORY_PATH_TEST, memMemPath=BasicParameters.MEMORY_MEMORY_PATH_TEST,ckptPath=BasicParameters.CKPT_PATH_TEST)
        
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
        saveEverySampleCount: Optional[int] = None,) -> Dict[str, Any]:
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
                saveEverySampleCount=saveEverySampleCount,)

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
        cameraIndex: int = 0,
        useHebbian: bool = True,) -> Dict[str, Any]:
        try:
            ok = self.StartDeployment(cameraIndex=cameraIndex, useHebbian=useHebbian,)

            if not ok:
                print("StartDeployment returns False (deployment may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorDeployment)

            return {"ok": True}

        except Exception as e:
            print(f"DeployModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

class TestManagerMTool:
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
            class FakeBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.runtime = torch.tensor([3.0])
                    self.training_graph = {"pending": object()}
                    self.child = nn.Linear(1, 1)
                    self.train()
                    self.child.eval()

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
                return 1.25

            def evaluate_test():
                split_entries.append(float(brain.runtime.item()))
                split_random.append(draw_random())
                brain.eval()
                brain.runtime.fill_(99.0)
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
                and torch.equal(saved_buffers["runtime"], torch.tensor([3.0])))
            print(f"Manager evaluation runtime/RNG restore {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager evaluation runtime restore error: {e}")
            return False
        finally:
            if caller_rng is not None:
                manager.RestoreRngState(caller_rng)

    def TestLoadCheckpointRestoresTopologyBeforeOptimizers(self) -> bool:
        try:
            import io

            events: List[str] = []

            class FakeBrain(nn.Module):
                def __init__(self):
                    super().__init__()

                def ResizeStateBuffersForLoad(self, state):
                    events.append("resize")

                def UpgradeDecisionStateDict(self, state):
                    events.append("migrate")
                    return state

                def load_state_dict(self, state, strict):
                    events.append(f"brain_load:{strict}")

                def ImportBuffers(self, state):
                    events.append("buffers")

            class FakeOptimizer:
                def __init__(self, name):
                    self.name = name

                def load_state_dict(self, state):
                    events.append(self.name)

            class FakeAgent:
                def __init__(self):
                    self.opt_actor = FakeOptimizer("opt_actor")
                    self.opt_critic = FakeOptimizer("opt_critic")
                    self.opt_world = FakeOptimizer("opt_world")

                def SyncTrainableOptimizers(self):
                    events.append("sync")

                def ClearTransientCandidateOptimizerState(self):
                    events.append("clear_candidates")

            checkpoint = {
                "schema_version": TRAIN_CHECKPOINT_SCHEMA_VERSION,
                "brain": {"buffer": torch.zeros(2)},
                "opt_actor": {},
                "opt_critic": {},
                "opt_world": {},
                "buffers": {},}
            payload = io.BytesIO()
            torch.save(checkpoint, payload)
            payload.seek(0)
            manager = ManagerFunction.__new__(ManagerFunction)
            manager.device = torch.device("cpu")
            manager.LoadCheckpoint(FakeBrain(), FakeAgent(), [], payload)
            ok = events == [
                "migrate",
                "resize",
                "brain_load:True",
                "sync",
                "opt_actor",
                "opt_critic",
                "opt_world",
                "clear_candidates",
                "buffers",]
            print(f"Manager checkpoint restore order {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager checkpoint restore order error: {e}")
            return False

    def TestLoadBrainWeightsResizesAndSyncsOverride(self) -> bool:
        try:
            events: List[str] = []

            from FunctionTools import GrowableLoRALinear

            class FakeBrain(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.adapter = GrowableLoRALinear(nn.Linear(2, 2))
                    self.adapter.Grow(1)

                def ResizeStateBuffersForLoad(self, state):
                    events.append("resize")

                def UpgradeDecisionStateDict(self, state):
                    events.append("migrate")
                    return state

                def load_state_dict(self, state, strict):
                    events.append(f"load:{strict}")

            class FakeAgent:
                def SyncTrainableOptimizers(self):
                    events.append("sync")

                def ClearTrainableOptimizerState(self):
                    events.append("clear_optimizer_state")

            brain = FakeBrain()
            manager = ManagerFunction.__new__(ManagerFunction)
            manager.LoadTorchPayload = lambda path: {
                "brain": {
                    "adapter.target.weight": torch.randn_like(brain.adapter.target.weight),
                    "adapter.target.bias": torch.randn_like(brain.adapter.target.bias)}}
            manager.LoadBrainWeights(
                brain,
                "unused.pth",
                agent=FakeAgent())
            ok = events == [
                "migrate",
                "resize",
                "load:False",
                "sync",
                "clear_optimizer_state"] and len(brain.adapter.A_list) == 0
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

            ok = True
            ok &= world == [brain.world]
            ok &= brain.world not in policy
            ok &= brain.critic in policy and brain.actor not in policy
            ok &= brain.world in full and brain.critic in full
            ok &= brain.actor not in full
            ok &= invalid_rejected

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
            print(f"Manager train-stage isolation contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager train-stage isolation contract error: {e}")
            return False

    def TestCppDecisionWireContract(self) -> bool:
        try:
            manager = object.__new__(ManagerFunction)
            manager.device = torch.device("cpu")
            captured: Dict[str, Any] = {}

            def capture_forward(*args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return "ok"

            manager.AgentHandleForward = capture_forward
            endpoint_pose = torch.zeros(
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            robot_packet = {
                "schema_version": ROBOT_STATE_WIRE_SCHEMA_VERSION,
                "endpoint_names": list(ModuleDim.DecisionEndpointNames),
                "pose_frame": "world",
                "pose_unit": "meter",
                "quaternion_order": "xyzw",
                "endpoint_pose": endpoint_pose.tolist(),
                "camera_pose_world": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "planner_expected_endpoint_pose": endpoint_pose.tolist(),
                "planner_progress": 0.2,
                "planner_tracking_error": 0.1,
                "planner_executing": 1.0,
                "planner_reached": 0.0,
                "planner_failed": 0.0,
                "model_command_executed": 1.0,}
            sensor_packet = {
                "schema_version": SENSOR_PACKET_WIRE_SCHEMA_VERSION,
                "text_ext": ["pick up the cup"],
                "text_trust": [TEXT_TRUST_OPERATOR_COMMAND],
                "sample_actions": False,
                "deterministic_actor": True,
                "depth": [[1.0, 2.0], [3.0, 4.0]],
                "depth_valid": [[True, True], [True, False]],
                "depth_scale_meters": 0.001,}
            result = ManagerFunction.AgentHandleForwardJson(
                manager,
                2,
                0.5,
                0.0,
                json.dumps(sensor_packet),
                json.dumps(robot_packet))
            robot_state = captured["kwargs"]["robotState"]
            ok = (
                result == "ok"
                and captured["args"] == (2, 0.5, 0.0)
                and tuple(robot_state["endpoint_pose"].shape)
                == (1, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim)
                and tuple(robot_state["camera_pose_world"].shape) == (1, ModuleDim.PstPoseDim)
                and tuple(robot_state["model_command_executed"].shape) == (1,)
                and captured["kwargs"]["textExt"] == sensor_packet["text_ext"]
                and captured["kwargs"]["textTrust"] == sensor_packet["text_trust"]
                and captured["kwargs"]["sampleActions"] is False
                and captured["kwargs"]["deterministicActor"] is True
                and captured["kwargs"]["depthBitmap"] == sensor_packet["depth"]
                and captured["kwargs"]["depthValid"] == sensor_packet["depth_valid"]
                and captured["kwargs"]["depthScaleMeters"] == 0.001)
            print(f"Manager C++ decision wire contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Manager C++ decision wire contract error: {e}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "PauseStopImmediateExit": self.TestPauseStopImmediateExit(),
            "BackgroundExceptionStatus": self.TestBackgroundExceptionStatus(),
            "EvaluationRuntimeRestoredBeforeSave": self.TestEvaluationRuntimeRestoredBeforeSave(),
            "LoadCheckpointRestoresTopologyBeforeOptimizers": self.TestLoadCheckpointRestoresTopologyBeforeOptimizers(),
            "LoadBrainWeightsResizesAndSyncsOverride": self.TestLoadBrainWeightsResizesAndSyncsOverride(),
            "TrainStageIsolationContract": self.TestTrainStageIsolationContract(),
            "CppDecisionWireContract": self.TestCppDecisionWireContract(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nManager tests: {passed}/{len(results)} passed.")
        return results


class AgentHandle:
    def __init__(
        self,
        *,
        brainParameterPath: str = BasicParameters.MODULEPARAMETER_PATH,
        device: Optional[str] = None,
        worldMemoryPath: str = BasicParameters.WORLD_MEMORY_PATH,
        memMemoryPath: str = BasicParameters.MEMORY_MEMORY_PATH,
        seqLen: int = BasicParameters.IMAGE_SEQ_LEN,
        plasticHebbian: bool = True,
        prioritizeExtStr: bool = True,
        saveModuleMessagerOutput: bool = True,):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        parameter_path = str(brainParameterPath).strip()
        if parameter_path == "":
            raise ValueError("brainParameterPath must not be empty")

        resolved_path = Path(parameter_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"brain parameter file not found: {resolved_path}")

        self.brain = BrainCore(
            device=self.device,
            seqLen=seqLen,
            plasticHebbian=plasticHebbian,
            prioritizeExtStr=prioritizeExtStr,
            plasticOnlineLearning=False,
            saveModuleMessagerOutput=saveModuleMessagerOutput,)

        self.agent = Agent(
            self.brain,
            isTrain=False,
            device=self.device,
            worldMemoryPath=worldMemoryPath,
            memMemoryPath=memMemoryPath)

        self.agent.LoadBrainWeights(str(resolved_path))
        self.brain.eval()

    def SetCameraIntrinsics(
        self,
        cameraIntrinsics: Union[np.ndarray, torch.Tensor],
        sourceSize: Optional[Tuple[int, int]] = None) -> None:
        k = torch.as_tensor(cameraIntrinsics)
        self.agent.SetCameraIntrinsics(k, sourceSize=sourceSize)

    def ForwardStep(
        self,
        frame: torch.Tensor,
        *,
        textExt: Optional[List[Optional[str]]] = None,
        textTrust: Optional[List[str]] = None,
        reward: Optional[int] = None,
        done: Optional[int] = None,
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
