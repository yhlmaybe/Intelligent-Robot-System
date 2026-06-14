from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
import threading
import queue
import random
import ast
import json

import numpy as np
import torch
import torch.nn as nn
import traceback
import os
import math
import copy
import inspect

#import debugpy

from dataclasses import dataclass, field
from collections import deque

from PerceptionModule import PerceiveExtractor, PerceptionOnlineWrapper, PerceptionRecallLoss, TopDownContext, VisualState
from AttentionModule import AttentionExtractor, AttentionOnlineWrapper
from MemoryModule import MemoryExtractor, MemoryType
from DecisionModule import DecisionExtractor, DecisionPlannerExtractor
from DecisionDecoupler import DecisionDecouplerV2, RelativePoseError
from WorldModule import RSSMWorldModel, WorldOnlineWrapper
from ValueEstimationModule import ValueEstimationExtractor,ValueEstimationOnlineWrapper
from ConsciousnessModule import ConsciousnessExtractor, ConsciousnessOutput
from IntentionModule import IntentionExtractor, IntentionOnlineWrapper
from OCRModule import OCREngineExtractor
from PhysicalStateModule import PhysicalStateExtractor, PhysicalStateLoss
from GoalModule import GoalGrounding, SatisfactionCheckModule, FourLevelGoalManager
from NeuroSymbolicModule import NeuroSymbolicExtractor
from TemporalExecutionModule import TemporalExecutionGateExtractor, DISPATCH, REDISPATCH, CONTINUE
from ModuleMessagerManager import ModuleDim, ModuleMessagerManager
 


def ToDevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

class BasicParameters:
    IMAGE_SIZE = 512

    IMAGE_SEQ_LEN = 16

    IMAGE_RM_LEN = math.ceil(IMAGE_SEQ_LEN * 1 / 10)

    MEMORY_CALLBACK_LEN = 16

    REWARD_MIN = -10.0

    REWARD_MAX = 10.0

    CONSCIOUSNESSTEM = 1 * 1024

    SAVE_EVERY_SAMPLE_COUNT = 500

    DATA_ROOT_PATH = "BrainDeepLearn/Data"
    OCR_DATA_ROOT_PATH = "BrainDeepLearn/Data/OCR"

    DATA_FRAMES_PATH = "BrainDeepLearn/Data/frames"
    DATA_REWARD_PATH = "BrainDeepLearn/Data/reward"
    DATA_DONE_PATH = "BrainDeepLearn/Data/done"
    DATA_DEPTH_PATH = "BrainDeepLearn/Data/depth"
    DATA_DEPTH_VALID_PATH = "BrainDeepLearn/Data/depth_valid"
    DATA_ACTIONS_PATH = "BrainDeepLearn/Data/actions"
    DATA_DEPTH_SCALE_METERS = 1.0
    DATA_TEXTS_PATH = "BrainDeepLearn/Data/texts"
    OCR_FRAMES_PATH = "BrainDeepLearn/Data/OCR/frames"
    OCR_TEXTS_PATH = "BrainDeepLearn/Data/OCR/OCRTexts"
    OCR_RECOGNIZER_FRAMES_PATH = "BrainDeepLearn/Data/OCRRecognition/frames"
    OCR_RECOGNIZER_TEXTS_PATH = "BrainDeepLearn/Data/OCRRecognition/OCRTexts"

    MEMORY_MEMORY_PATH = "BrainDeepLearn/Data/MemoryMemory.pt"
    WORLD_MEMORY_PATH = "BrainDeepLearn/Data/WorldMemory.pt"
    MODULEPARAMETER_PATH = "BrainDeepLearn/Data/module_parameter.pth"
    OCR_MODULEPARAMETER_PATH = "BrainDeepLearn/Data/ocr_module_parameter.pth"
    OCR_RECOGNIZER_MODULEPARAMETER_PATH = "BrainDeepLearn/Data/ocr_recognizer_parameter.pth"

    OCR_RECOGNIZER_DATA_ROOT_PATH = "BrainDeepLearn/Data/OCRRecognition"
    CKPT_PATH_TRAIN = "BrainDeepLearn/Data/training_checkpoint.pth"
    OCR_CKPT_PATH_TRAIN = "BrainDeepLearn/Data/ocr_training_checkpoint.pth"
    OCR_RECOGNIZER_CKPT_PATH_TRAIN = "BrainDeepLearn/Data/ocr_recognizer_training_checkpoint.pth"

    MEMORY_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/MemoryMemory.pt"
    WORLD_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/WorldMemory.pt"
    MODULEPARAMETER_PATH_TEST = "BrainDeepLearn/TestData/module_parameter.pth"
    OCR_MODULEPARAMETER_PATH_TEST = "BrainDeepLearn/TestData/ocr_module_parameter.pth"
    OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST = "BrainDeepLearn/TestData/ocr_recognizer_parameter.pth"
    DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData"
    DATA_DEPTH_PATH_TEST = "BrainDeepLearn/TestData/depth"
    DATA_DEPTH_VALID_PATH_TEST = "BrainDeepLearn/TestData/depth_valid"
    OCR_DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData/OCR"
    OCR_RECOGNIZER_DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData/OCRRecognition"
    CKPT_PATH_TEST = "BrainDeepLearn/TestData/training_test_checkpoint.pth"
    OCR_CKPT_PATH_TEST = "BrainDeepLearn/TestData/ocr_training_checkpoint.pth"
    OCR_RECOGNIZER_CKPT_PATH_TEST = "BrainDeepLearn/TestData/ocr_recognizer_training_checkpoint.pth"

    @classmethod
    def Get(cls, name: str):
        attrName = str(name).strip()
        if not cls.IsConfigAttribute(attrName):
            raise AttributeError(f"BasicParameters has no attribute: {attrName}")
        return getattr(cls, attrName)

    @classmethod
    def Set(cls, name: str, value: str) -> bool:
        try:
            attrName = str(name).strip()
            if not cls.IsConfigAttribute(attrName):
                return False
            if not isinstance(value, str):
                return False

            currentValue = getattr(cls, attrName)
            parsedValue = cls.ParseValueFromString(value, currentValue)
            setattr(cls, attrName, parsedValue)
            cls.RefreshDerivedParameters(attrName)
            return True
        except Exception:
            return False

    @classmethod
    def GetStringDict(cls) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for name in cls.__dict__.keys():
            if cls.IsConfigAttribute(name):
                result[str(name)] = str(getattr(cls, name))
        return result

    @classmethod
    def ParseValueFromString(cls, value: str, currentValue: Any):
        text = value.strip()

        if isinstance(currentValue, bool):
            textLower = text.lower()
            if textLower in ("true", "1", "yes", "on"):
                return True
            if textLower in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"cannot parse bool from: {value}")

        if isinstance(currentValue, int) and not isinstance(currentValue, bool):
            return int(text)

        if isinstance(currentValue, float):
            return float(text)

        if isinstance(currentValue, str):
            return value

        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = text

        if currentValue is None:
            return parsed

        currentType = type(currentValue)
        if isinstance(parsed, currentType):
            return parsed

        return currentType(parsed)

    @classmethod
    def RefreshDerivedParameters(cls, changedName: str = ""):
        if changedName == "IMAGE_SEQ_LEN":
            cls.IMAGE_RM_LEN = math.ceil(cls.IMAGE_SEQ_LEN * 1 / 10)

    @classmethod
    def IsConfigAttribute(cls, name: str) -> bool:
        if str(name).strip() == "":
            return False
        if name not in cls.__dict__:
            return False
        value = cls.__dict__[name]
        if isinstance(value, (classmethod, staticmethod)):
            return False
        return not callable(getattr(cls, name))


@dataclass
class BrainStepTrace:
    ObsImg: Optional[torch.Tensor] = None
    ActionEmbed: Optional[torch.Tensor] = None

    PercBuffer: Optional[list[torch.Tensor]] = None
    VisualBuffer: Optional[List[VisualState]] = None
    VisualStateNow: Optional[VisualState] = None
    OcrSemantic: Optional[torch.Tensor] = None
    IntentHint: Optional[torch.Tensor] = None
    PercFeat: Optional[torch.Tensor] = None
    AttnFeat: Optional[torch.Tensor] = None
    MemFeat: Optional[torch.Tensor] = None
    WorldState: Optional[torch.Tensor] = None
    WorldDeltaTransport: Optional[torch.Tensor] = None
    WorldDeltaPhysics: Optional[torch.Tensor] = None
    ConsciousnessState: Optional[torch.Tensor] = None
    IntentionState: Optional[torch.Tensor] = None
    Reward: Optional[torch.Tensor] = None
    Done: Optional[torch.Tensor] = None
    ValueFeat: Optional[torch.Tensor] = None
    ActionEntropy: Optional[torch.Tensor] = None

    EntropyPrev: Optional[torch.Tensor] = None
    UncertaintyPrev: Optional[torch.Tensor] = None
    TdErrorPrev: Optional[torch.Tensor] = None

    extras: Dict[str, Any] = field(default_factory=dict)


class BrainCore(nn.Module):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        seqLen: int = BasicParameters.IMAGE_SEQ_LEN,
        plasticHebbian: bool = True,
        prioritizeExtStr: bool = True,
        plasticOnlineLearning: bool = False,
        usePlanner: bool = True,
        plannerTeacherMode: bool = True,
        enablePerceptionSupervision: bool = False,
        saveModuleMessagerOutput: bool = True,
        needTrace: bool = True,
        slowPeriod: int = 4,):
        super().__init__()
        self.SEQ_LEN = seqLen
        self.slow_period = int(slowPeriod)
        self.is_online_learning = plasticOnlineLearning
        self.prioritize_ext_str = prioritizeExtStr
        self.need_trace = bool(needTrace)
        self.use_planner = bool(usePlanner)
        self.planner_teacher_mode = bool(plannerTeacherMode)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.perc = PerceiveExtractor(
            imgSize=BasicParameters.IMAGE_SIZE,
            embedDim=ModuleDim.PerceptionEmbed,
            useHebbian=plasticHebbian,
            enableRecallAuxiliary=enablePerceptionSupervision)
        self.perception_recall_loss = PerceptionRecallLoss() if enablePerceptionSupervision else None
        
        self.attn = AttentionExtractor(
            embedDim=ModuleDim.AttentionFeat,
            sequenceLength=seqLen, 
            hebbianRate=(0.01 if plasticHebbian else 0.0), 
            useHebbian=plasticHebbian,
            structuredDim=ModuleDim.PerceptionEmbed,
            goalDim=ModuleDim.IntentionFeat,
            objectTokenCount=self.perc.object_token_count)
        
        self.mem = MemoryExtractor(
            inputDim=ModuleDim.AttentionFeat,
            ssmStateDim=ModuleDim.MemoryItem,
            memoryDim=ModuleDim.MemoryItem,
            outputDim=ModuleDim.MemoryFeat,
            hebbAlpha=(0.15 if plasticHebbian else 0.0), 
            useHebbian=plasticHebbian,
            emotionDim=ModuleDim.ValueEstimationOutEmotion)
        
        self.value_tensor_dim = 512
        self.actor = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat,
            intentDim=ModuleDim.IntentionFeat,
            includeNoSkill=True,
            useHebb=plasticHebbian,
            valueTensorDim=self.value_tensor_dim,
            vNextTensorDim=self.value_tensor_dim,
            beliefDim=ModuleDim.DecisionBeliefDim,
            decisionDynDim=ModuleDim.DecisionDynDim,
            latentControlDim=ModuleDim.LatentControlDim,
            mapperEmbedDim=ModuleDim.MapperHiddenDim,)
        
        self.world = RSSMWorldModel(
            visionDim=ModuleDim.AttentionFeat, 
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
            # K_PST is decoupled from K_observed: PST is a persistent world memory
            # that must accumulate identities across frames, so it gets its own slot count.
            physicalSlots=ModuleDim.PstSlots,
            physicalSlotDim=ModuleDim.PstSlotDim,
            physicalPoseDim=ModuleDim.PstPoseDim,
            physicalAttrDim=ModuleDim.PstAttrDim,
            physicalIdDim=ModuleDim.PstIdDim,
            physicalRelDim=ModuleDim.PstRelDim,
            physicalRelationClasses=ModuleDim.PstRelationClasses,
            physicalSemanticDim=ModuleDim.PstSemanticDim,
            physicalStateDim=ModuleDim.PstStateDim,
            physicalAffordanceDim=ModuleDim.PstAffordanceDim,
            physicalTextDim=ModuleDim.PstTextDim,
            physicalSymbolDim=ModuleDim.PstSymbolClasses)

        self.critic = ValueEstimationExtractor(
            memoryDim=ModuleDim.MemoryFeat,
            attnDim=ModuleDim.AttentionFeat,
            stateDim=ModuleDim.WorldFeat,
            emotionDim=ModuleDim.ValueEstimationOutEmotion,
            useHebb=plasticHebbian,
            valueTensorDim=self.value_tensor_dim,)
        
        self.conscious = ConsciousnessExtractor(
            memItemDim=ModuleDim.MemoryItem,
            worldItemDim=ModuleDim.WorldMemoryItem,
            intentDim=ModuleDim.ConsciousnessState,
            useHebb=plasticHebbian)

        self.intention = IntentionExtractor(
            dimSem=ModuleDim.IntentionFeat,
            consSelfDim=int(self.conscious.self_dim),
            consIntentDim=int(self.conscious.intent_dim))

        # Embodied-AGI v2: tensorized physical state, hierarchical goals, coarse-to-fine.
        self.pst_builder = PhysicalStateExtractor(
            inObjectDim=ModuleDim.PerceptionEmbed)
        self.pst_loss = PhysicalStateLoss()
        world_latent_dim = ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState
        self.goal_manager = FourLevelGoalManager(
            worldLatentDim=world_latent_dim,
            pstSummaryDim=ModuleDim.PstSlotDim,
            intentDim=ModuleDim.IntentionFeat)
        self.goal_grounding = GoalGrounding()
        self.decision_decoupler = DecisionDecouplerV2(decisionDim=ModuleDim.DecisionBeliefDim)
        self.satisfaction = SatisfactionCheckModule()
        self.neuro_symbolic = NeuroSymbolicExtractor()
        self.temporal_gate = TemporalExecutionGateExtractor()

        self.OCR = OCREngineExtractor()

        self.history_len = int(BasicParameters.MEMORY_CALLBACK_LEN)

        if plasticOnlineLearning:
            self.perc = PerceptionOnlineWrapper(self.perc)
            self.attn = AttentionOnlineWrapper(self.attn)
            self.world = WorldOnlineWrapper(self.world)
            self.critic =ValueEstimationOnlineWrapper(self.critic)
            self.intention = IntentionOnlineWrapper(self.intention)

        self.planner = None
        if self.use_planner or self.planner_teacher_mode:
            self.planner = DecisionPlannerExtractor().BuildPlanner(
                worldModel=self.world,
                wmIsOnlineWrapper=plasticOnlineLearning,
                decisionDecoupler=self.decision_decoupler,
                N=64, elite=8, iters=3,
                temperature=1.0, momentum=0.15,
                minVar=1e-4)

        self.buf_B = 0

        self.extra_mem = None
        self.thread_end = True
        self.smooth_queue: queue.Queue = queue.Queue()
        self.smooth_worker = threading.Thread(target=self.SmoothWorkerLoop, daemon=True)
        self.smooth_worker.start()

        self.mem_copy = copy.deepcopy(self.mem)
        self.attn_copy = copy.deepcopy(self.attn)
        self.critic_copy = copy.deepcopy(self.critic)
        self.moduleMessager = ModuleMessagerManager(maxSteps=256)
        self.save_module_messager_output = bool(saveModuleMessagerOutput)

        self.ResetBuffers(B=1, isOnlineLearning=self.is_online_learning,device=self.device)

    def SetModuleMessagerEnabled(self, enabled: bool):
        self.save_module_messager_output = bool(enabled)

    def SetTraceEnabled(self, enabled: bool):
        self.need_trace = bool(enabled)

    def RuntimeModule(self, mod: nn.Module) -> nn.Module:
        return mod.base if hasattr(mod, "base") else mod

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
        modules = (
            self.RuntimeModule(self.perc),
            self.RuntimeModule(self.attn),
            self.RuntimeModule(self.actor),
            self.RuntimeModule(self.critic),
            self.RuntimeModule(self.mem),
            self.RuntimeModule(self.conscious),)
        seen = set()
        for mod in modules:
            if mod is None or id(mod) in seen:
                continue
            seen.add(id(mod))
            reset_fn = getattr(mod, "ResetHebbianMemory", None)
            if reset_fn is None:
                continue
            params = inspect.signature(reset_fn).parameters
            if doneMask is not None and "doneMask" in params:
                reset_fn(doneMask=doneMask)
            else:
                reset_fn()

    @torch.no_grad()
    def ResizeStateBuffersForLoad(self, stateDict: Dict[str, Any]) -> None:
        modules = dict(self.named_modules())
        device = next(self.parameters()).device

        for key, value in stateDict.items():
            if not isinstance(value, torch.Tensor):
                continue

            parts = str(key).split(".")
            if len(parts) < 2:
                continue

            module_name = ".".join(parts[:-1])
            buffer_name = parts[-1]
            module = modules.get(module_name, None)
            if module is None or buffer_name not in module._buffers:
                continue

            current = module._buffers.get(buffer_name)
            if not isinstance(current, torch.Tensor):
                continue
            if tuple(current.shape) == tuple(value.shape):
                continue

            module._buffers[buffer_name] = torch.zeros(
                tuple(value.shape),
                device=device,
                dtype=value.dtype)

    def DetachVisualState(self, state: Optional[VisualState], *, clone: bool = False) -> Optional[VisualState]:
        if state is None:
            return None

        def d(t: torch.Tensor) -> torch.Tensor:
            out = t.detach()
            return out.clone() if clone else out

        return VisualState(
            IntegratedFeat=d(state.IntegratedFeat),
            GlobalFeat=d(state.GlobalFeat),
            VentralFeat=d(state.VentralFeat),
            DorsalFeat=d(state.DorsalFeat),
            MotionToken=d(state.MotionToken),
            QualityToken=d(state.QualityToken),
            PredErrorToken=d(state.PredErrorToken),
            ObjectTokens=d(state.ObjectTokens),
            PatchTokens=d(state.PatchTokens),
            SemanticNodes={k: d(v) for k, v in state.SemanticNodes.items()},
            Auxiliary={k: d(v) for k, v in state.Auxiliary.items() if isinstance(v, torch.Tensor)},)

    def DetachRuntimeObject(self, obj: Any, *, clone: bool = False) -> Any:
        if isinstance(obj, torch.Tensor):
            out = obj.detach()
            return out.clone() if clone else out
        if isinstance(obj, dict):
            return {k: self.DetachRuntimeObject(v, clone=clone) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.DetachRuntimeObject(v, clone=clone) for v in obj]
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

    def SetCameraIntrinsics(
        self,
        intrinsics: torch.Tensor,
        sourceSize: Optional[Tuple[int, int]] = None) -> None:
        """Register the calibration matrix on the perception module so subsequent
        Step()/Act() calls no longer need cameraIntrinsics. Pass sourceSize=(H, W)
        when the K is calibrated for the raw camera resolution; omit it when the K
        is already scaled to BasicParameters.IMAGE_SIZE."""
        self.perc.SetCameraIntrinsics(intrinsics, sourceSize=sourceSize)

    def BuildTopDownContext(self) -> TopDownContext:
        return TopDownContext(
            PredictedVisual=self.prev_predicted_visual,
            Precision=self.prev_precision,
            MemoryCue=self.prev_mem,)

    def BuildVisualSequenceTensors(
        self,
        visualStates: List[VisualState],
        *,
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        recent = visualStates[-self.SEQ_LEN:]
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

        for idx, vs in enumerate(recent, start=start):
            integrated[:, idx] = vs.IntegratedFeat
            object_seq[:, idx] = vs.ObjectTokens
            motion_seq[:, idx] = vs.MotionToken
            quality_seq[:, idx] = vs.QualityToken
            pred_seq[:, idx] = vs.PredErrorToken
            key_padding_mask[:, idx] = False

        return integrated.contiguous(), object_seq.contiguous(), motion_seq.contiguous(), quality_seq.contiguous(), pred_seq.contiguous(), key_padding_mask

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

    @torch.no_grad()
    def ResetBuffers(self, B: int = 1, isOnlineLearning: bool = False, device: Optional[torch.device] = None):
        device = device or self.device

        def z(*s, dtype=torch.float32):
            return torch.zeros(B, *s, device=device, dtype=dtype)

        self.prev_mem = z(ModuleDim.MemoryFeat)
        self.prev_attn = z(ModuleDim.AttentionFeat)

        self.prev_world_h = z(ModuleDim.WorldOutHState)
        self.prev_world_z = z(ModuleDim.WorldOutZState)
        self.prev_world_x = z(ModuleDim.WorldOutXState)
        self.prev_world_s = z(ModuleDim.WorldFeat)
        self.prev_done_flag = torch.ones(B, device=device, dtype=torch.bool)

        if isOnlineLearning and hasattr(self.actor, "base"):
            self.prev_option_logit = z(self.actor.base.num_options)
            actor_runtime = self.actor.base
        else:
            self.prev_option_logit = z(self.actor.num_options)
            actor_runtime = self.actor

        self.prev_decision_state = z(int(actor_runtime.dyn_dim))
        self.prev_latent_control = z(int(actor_runtime.u_dim))
        self.prev_decision_feedback_embed = z(int(actor_runtime.action_embed_dim))
        self.prev_executed_decision_feedback_embed = z(int(actor_runtime.action_embed_dim))
        self.prev_mapper_hidden = z(int(actor_runtime.mapper_hidden_dim))
        self.prev_td_error = torch.zeros(B, device=device, dtype=torch.float32)
        self.temporal_active_mask = z()
        self.temporal_action_age = z()
        self.temporal_feedback_age = z()
        self.temporal_action_epoch = z()
        self.temporal_invoke_drift = z()
        self.temporal_active_kind = torch.zeros(B, device=device, dtype=torch.long)
        self.active_motion_command = None
        # Last frame's committed target pose. Next frame the measured endpoint pose is compared
        # against it to expose command-vs-achieved tracking error.
        self.prev_target_endpoint_pose = z(ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim)
        self.prev_target_endpoint_pose[..., 6] = 1.0

        self.prev_entropy = z()

        self.prev_visual_state = None
        self.prev_predicted_visual = None
        self.prev_precision = torch.ones(B, device=device, dtype=torch.float32)
        self.prev_goal_bias = z(ModuleDim.IntentionFeat)
        self.prev_self_sem = None
        self.prev_intent_sem = z(ModuleDim.IntentionFeat)

        self.prev_refinement_dir = z(ModuleDim.RefinementDim)
        self.prev_failure_count = z()

        # Previous frame's absolute camera pose (camera->world). camera_motion is derived
        # from this and the current pose; None means "no previous frame" (first step / post-done).
        self.prev_camera_pose_world = None

        self.perc_buffer = []
        self.visual_state_buffer = []

        self.history = deque(maxlen=self.history_len)

        self.slow_step_count = 0
        self.slow_cache = None

        self.buf_B = B

    def RelativeCameraMotion(self, prevPose, curPose):
        """Derive the inter-frame camera_motion from two absolute camera->world poses.
        Returns the current camera expressed in the previous camera frame (so a current 3D
        point maps to the previous frame as X_prev = R(q) X_cur + t), or identity on the
        first frame, or None when no camera pose is supplied. xyz + xyzw quaternion."""
        if curPose is None:
            return None
        if prevPose is None:
            identity = curPose.new_zeros(curPose.size(0), ModuleDim.PstPoseDim)
            identity[:, 6] = 1.0
            return identity
        prev_t, prev_q = prevPose[:, :3], prevPose[:, 3:7]
        cur_t, cur_q = curPose[:, :3], curPose[:, 3:7]
        prev_q_inv = prev_q * prev_q.new_tensor([-1.0, -1.0, -1.0, 1.0])  # conjugate of a unit quaternion
        rotation = RSSMWorldModel.QuaternionMultiply(prev_q_inv, cur_q)
        translation = RSSMWorldModel.QuaternionRotate(prev_q_inv, cur_t - prev_t)
        return torch.cat([translation, nn.functional.normalize(rotation, dim=-1)], dim=-1)


    def Step(
        self,
        frame: torch.Tensor,  # [B, C, H, W]
        textExt: Optional[List[Optional[str]]] = None,
        rewardExt: Optional[torch.Tensor] = None, # [B, 1]
        doneFlag: Optional[torch.Tensor] = None, # [B, 1]
        *,
        isTrain: bool = False,
        sampleActions: bool = True,
        deterministicActor: bool = False,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        robotPhysicalContext: torch.Tensor,
        interactionContext: torch.Tensor,
        perceptionTargets: Optional[Dict[str, torch.Tensor]] = None,
        robotState: Dict[str, torch.Tensor],) -> Dict[str, Any]:

        if self.is_online_learning and not isTrain:
            raise RuntimeError(f"Wrappers can only be used during training, but isTrain is {isTrain}, isUseWrappers is {self.is_online_learning}")

        isBeginStep = True

        def saveModuleOutput(moduleName: str, output: Any):
            nonlocal isBeginStep
            if not self.save_module_messager_output:
                return
            self.moduleMessager.SaveModuleOutput(moduleName, output, isBeginStep=isBeginStep)
            isBeginStep = False

        B, dev = frame.size(0), frame.device
        if self.buf_B != B:
            self.ResetBuffers(B=B, isOnlineLearning=self.is_online_learning, device=dev)
        # Measured (proprioceptive feedback) endpoint pose; the tracking error against last
        # frame's commanded setpoint is folded into its encoding so drift/limits are perceived.
        endpoint_pose = robotState["endpoint_pose"]
        planner_expected_endpoint_pose = robotState["planner_expected_endpoint_pose"]
        endpoint_tracking_error = RelativePoseError(self.prev_target_endpoint_pose, endpoint_pose)
        planner_endpoint_tracking_error = RelativePoseError(planner_expected_endpoint_pose, endpoint_pose)
        endpoint_pose_encoding = self.decision_decoupler.EncodeEndpointPose(
            endpoint_pose,
            endpoint_tracking_error,
            planner_endpoint_tracking_error)

        world_action_feedback = self.prev_executed_decision_feedback_embed

        def normalize_external_signal(
            x: Optional[torch.Tensor],
            *,
            clamp01: bool = False,
            clampReward: bool = False) -> Optional[torch.Tensor]:
            if x is None:
                return None
            out = x.detach().view(B)
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            if clampReward:
                out = out.clamp(float(BasicParameters.REWARD_MIN), float(BasicParameters.REWARD_MAX))
            return out.clamp(0.0, 1.0) if clamp01 else out

        reward_ext = normalize_external_signal(rewardExt, clampReward=True)
        done_ext = normalize_external_signal(doneFlag, clamp01=True)
        
        if self.extra_mem and self.thread_end:
            self.mem.MergeMemoryState(self.extra_mem)
            self.extra_mem =None


        def init_shadow_module_parms():
            mem_state = self.mem.ExportState()
            self.mem_copy.EnsureB(B, device=dev, dtype=self.prev_mem.dtype)
            self.mem_copy.ImportState(mem_state)
            self.mem_copy.pending = self.DetachRuntimeObject(self.mem.pending, clone=True)
            attn_state = self.attn.ExportState()
            self.attn_copy.ImportState(attn_state)
            critic_state = self.critic.ExportState()
            self.critic_copy.ImportState(critic_state)   

        if self.need_trace and not isTrain and reward_ext is not None and self.history and self.thread_end:
            self.thread_end = False
            init_shadow_module_parms()
            # Traces are detached/cloned at creation, so a shallow snapshot is enough.
            self.smooth_queue.put((list(self.history), reward_ext, "Reward", self.attn_copy, self.mem_copy, self.critic_copy))

        if self.need_trace and not isTrain and done_ext is not None and self.history and self.thread_end:
            self.thread_end = False
            init_shadow_module_parms()
            self.smooth_queue.put((list(self.history), done_ext, "Done", self.attn_copy, self.mem_copy, self.critic_copy))

        B, C, H, W = frame.shape

        if isTrain:
            prev_world_h_for_prediction = self.prev_world_h.detach()
            prev_world_z_for_prediction = self.prev_world_z.detach()
            prev_world_x_for_prediction = self.prev_world_x.detach()
            prev_done_for_prediction = self.prev_done_flag.detach().clone()
            prev_physical_state_for_prediction = self.RuntimeModule(self.world).ExportPhysicalState()

        top_down = self.BuildTopDownContext()
        # camera_motion is derived from consecutive absolute camera poses, never taken as a
        # label. The current camera->world pose is part of robotState; the relative motion
        # to the previous frame is computed against self.prev_camera_pose_world.
        camera_pose_world = robotState["camera_pose_world"]
        camera_motion_from_prev = self.RelativeCameraMotion(self.prev_camera_pose_world, camera_pose_world)
        prev_visual_for_loss = self.prev_visual_state
        visual_state = self.perc(
            frame,
            prevVisualState=prev_visual_for_loss,
            topDownContext=top_down,
            depth=depth,
            depthValid=depthValid,
            cameraMotion=camera_motion_from_prev)
        perc_feats = visual_state.IntegratedFeat # [B, D_perc]
        # Slow/fast split: OCR, consciousness, intention and the long/mid goal stack run
        # every slow_period steps; an external text command forces an immediate refresh.
        text_override = textExt is not None and any(t is not None and str(t).strip() != "" for t in textExt)
        slow_refresh = (
            self.slow_cache is None
            or (self.slow_step_count % self.slow_period == 0)
            or text_override)
        self.slow_step_count += 1
        if slow_refresh:
            ocr_items = self.OCR(frame)
            fuse_ocr = self.OCR.ExportFusedTexts() # List[List[str]]
            ocr_semantic = self.EncodeOcrSemantic(fuse_ocr, batchSize=B, device=dev)
        else:
            ocr_items = self.slow_cache["ocr_items"]
            fuse_ocr = self.slow_cache["fuse_ocr"]
            ocr_semantic = self.slow_cache["ocr_semantic"]

        visual_seq_src = self.visual_state_buffer + [visual_state]
        if len(visual_seq_src) > self.SEQ_LEN:
            visual_seq_src = visual_seq_src[-self.SEQ_LEN:]

        percs_seq, object_seq, motion_seq, quality_seq, pred_error_seq, key_padding_mask = self.BuildVisualSequenceTensors(
            visual_seq_src,
            batchSize=B,
            device=dev,
            dtype=frame.dtype)

        self.visual_state_buffer = [
            self.DetachVisualState(v)
            for v in visual_seq_src]
        self.perc_buffer = [v.IntegratedFeat for v in self.visual_state_buffer if v is not None]
        self.prev_visual_state = self.DetachVisualState(visual_state)

        saveModuleOutput("Perception", {
            "feat": perc_feats,
            "visual_state": {
                "global": visual_state.GlobalFeat,
                "ventral": visual_state.VentralFeat,
                "dorsal": visual_state.DorsalFeat,
                "motion": visual_state.MotionToken,
                "quality": visual_state.QualityToken,
                "pred_error": visual_state.PredErrorToken,
                "objects_mean": visual_state.ObjectTokens.mean(dim=1),
                "metric_depth": visual_state.Auxiliary["MetricDepth"],
                "monocular_depth": visual_state.Auxiliary["MonocularDepth"],
                "sensor_depth_reliability": visual_state.Auxiliary["SensorDepthReliability"],
                "object_geometry": visual_state.Auxiliary["ObjectGeometry"],},
            "key_padding_mask": key_padding_mask,})

        # --- Embodied-AGI v2: build and merge the Physical State Tensor before world inference ---
        semantic_view = self.pst_builder.SemanticWorldView(visual_state.SemanticNodes)
        node_mask = semantic_view["M"]
        geometry_valid = visual_state.Auxiliary["ObjectGeometryValid"]
        physical_out = self.pst_builder(
            visual_state.ObjectTokens,
            visual_state.Auxiliary["ObjectMotion"],
            visual_state.Auxiliary["ObjectGeometry"],
            node_mask,
            geometry_valid,
            robotContext=robotPhysicalContext,
            interactionContext=interactionContext)
        semantic_view = {**semantic_view, "M": physical_out["ObservationMask"]}
        observed_pst = {
            **physical_out,
            **semantic_view}
        world_runtime = self.RuntimeModule(self.world)
        pst = world_runtime.UpdatePhysicalState(
            observed_pst,
            robotState=robotState,
            executedActionEmbed=world_action_feedback)
        pst["U"] = self.mem.usage_bank.SlotReadout(pst["C"], pst["ARaw"]) * pst["M"].unsqueeze(-1)
        pst_summary = self.pst_builder.SlotSummary(pst["SRaw"], pst["MphysRaw"])
        
        w_preview = self.RuntimeModule(self.world).StepPriorOnly(
            hPrev=self.prev_world_h,
            zPrev=self.prev_world_z,
            s4xPrev=self.prev_world_x,
            physicalState=pst,
            actionEnc=world_action_feedback,
            sample=False)
        saveModuleOutput("WorldPreview", w_preview)

        s_t = w_preview["s_next"] # [B, D_world]
        r_t = w_preview["r_pred"].detach() # [B]
        d_t = w_preview["d_prob"].detach() # [B]
        d_tr = w_preview["d_tr"] # Optional[[B, D_world]]
        d_ph = w_preview["d_ph"] # Optional[[B, D_world]]

        value_reward = reward_ext if (isTrain and reward_ext is not None) else r_t
        value_done = done_ext if (isTrain and done_ext is not None) else d_t

        if self.is_online_learning:
            value_kwargs = {
                "rewardModel": value_reward,
                "policyEntropyPrev": self.prev_entropy,
                "doneModel": value_done,
                "worldDeltaTransport": d_tr,
                "worldDeltaPhysics": d_ph}
            value_x = {"memoryPrev": self.prev_mem,"attnPrev": self.prev_attn, "state": s_t} # memoryPrev:[B, D_mem], attnPrev:[B, D_attn], state:[B, D_world]
            critic_out = self.critic(x=value_x, **value_kwargs)
        else:
            critic_out = self.critic(memoryPrev=self.prev_mem,attnPrev=self.prev_attn,state=s_t,
                                     rewardModel=value_reward,doneModel=value_done,
                                     policyEntropyPrev=self.prev_entropy,
                                     worldDeltaTransport=d_tr,worldDeltaPhysics=d_ph,)
        saveModuleOutput("ValueEstimation", critic_out)

        value_current = critic_out.value
        value_next_current = critic_out.valueNext

        td_sig = critic_out.tdError.detach() # [B]
        unc_sig = critic_out.uncertainty.detach() # [B]
        precision_sig = critic_out.precision.detach() # [B]
        emotion_sig = critic_out.emotion.detach() # [B, D_emotion]
        value_comps = critic_out.rComps
        risk_sig = value_comps["risk"].detach()
        confidence_sig = value_comps["confidence"].detach()

        self.prev_precision = precision_sig.detach()

        atten_out = self.attn(
            percs_seq,
            keyPaddingMask=key_padding_mask,
            tdError=td_sig,
            uncertainty=unc_sig,
            objectSeq=object_seq,
            motionSeq=motion_seq,
            qualitySeq=quality_seq,
            predErrorSeq=pred_error_seq,
            goalBias=self.prev_goal_bias,
            precision=precision_sig,
            applyPlasticity=True) # [B, D_attn]
        saveModuleOutput("Attention", atten_out)

        intent_hint_for_memory = self.prev_intent_sem
        mem_feat = self.mem(
            atten_out,
            tdError=td_sig,
            emotion=emotion_sig,
            reward=r_t,
            visualState=visual_state,
            ocrSemantic=ocr_semantic,
            intentHint=intent_hint_for_memory,
            uncertainty=unc_sig,
            risk=risk_sig,
            confidence=confidence_sig) # [B, D_mem], [B, D_mem]
        saveModuleOutput("Memory", mem_feat)

        if isTrain:
            if self.is_online_learning:
                wm_kwargs = {"actionEnc": world_action_feedback,
                             "reward": reward_ext, "done": done_ext,
                             "physicalState": pst}
                w_out = self.world(atten_out, **wm_kwargs)
            else:
                w_out = self.world.ForwardTrain(visionIn=atten_out,
                                                physicalState=pst,
                                                actionEnc=world_action_feedback,
                                                reward=reward_ext, done=done_ext)
        else:
            w_out = self.world.StepPosterior(visionIn=atten_out,
                                             actionEnc=world_action_feedback,
                                             physicalState=pst,
                                             sample=False)
        saveModuleOutput("World", w_out)

        s_t = w_out["s_next"] # [B, D_world]
        r_t = w_out["r_pred"].detach() # [B]
        d_t = w_out["d_prob"].detach() # [B]
        d_tr = w_out["d_tr"] # Optional[[B, D_world]]
        d_ph = w_out["d_ph"] # Optional[[B, D_world]]

        next_visual_prediction = w_out["reconstructed_visual_state"]

        if done_ext is not None:
            done_now = done_ext> 0.5
        else:
            done_now = d_t > 0.5

        if B == 1 and bool(done_now.item()):
            self.prev_predicted_visual = None
        else:
            self.prev_predicted_visual = self.DetachRuntimeObject(next_visual_prediction)

        self.prev_world_s = s_t.detach()
        self.prev_done_flag = done_now.detach()

        if slow_refresh:
            memory_bank = self.mem.ExportMemoryBank(topk = BasicParameters.CONSCIOUSNESSTEM) # Optional[Dict[str, Tensor]]
            world_bank = self.world.ExportWorldMemoryBank(topk = BasicParameters.CONSCIOUSNESSTEM) # Optional[Dict[str, Tensor]]

            conscious_out = self.conscious(memoryBank=memory_bank, worldBank=world_bank) # self_sem/intention_sem: [B, D_cons]

            intent_sem, sym_probs, intention_extras = self.intention(
                conscious_out.self_sem,
                conscious_out.intent_sem,
                ocrTexts=fuse_ocr,
                extTexts=textExt,
                prioritizeExt=self.prioritize_ext_str,) # [B, D_intent], [B, K_sym], Dict[str, Tensor]
        else:
            conscious_out = ConsciousnessOutput(
                self_sem=self.slow_cache["self_sem"],
                intent_sem=self.slow_cache["cons_intent_sem"],
                extras=self.slow_cache["cons_extras"],)
            intent_sem = self.slow_cache["intent_sem"]
            sym_probs = self.slow_cache["sym_probs"]
            intention_extras = self.slow_cache["intention_extras"]

        saveModuleOutput("Consciousness", {
            "self_sem": conscious_out.self_sem,
            "intent_sem": conscious_out.intent_sem,
            "extras": conscious_out.extras,})

        saveModuleOutput("OCR", {
            "items": ocr_items,
            "texts": fuse_ocr,})

        intention_texts = [] if intention_extras is None else intention_extras.get("recall_texts", [])
        saveModuleOutput("Intention", {
            "intent_sem": intent_sem,
            "sym_probs": sym_probs,
            "extras": intention_extras,
            "ocr_texts": fuse_ocr,
            "ext_texts": textExt,})

        self.prev_self_sem = conscious_out.self_sem.detach()
        self.prev_intent_sem = intent_sem.detach()
        self.prev_goal_bias = intent_sem.detach()

        world_hzx_now = torch.cat([
            w_out["h_next"].detach(),
            w_out["z_next"],
            w_out["x_next"].detach(),], dim=-1)

        if slow_refresh:
            goals = self.goal_manager(
                worldLatent=world_hzx_now,
                pstSummary=pst_summary,
                intentEmbed=intent_sem,
                refinementDir=self.prev_refinement_dir,)
            self.slow_cache = {
                "ocr_items": ocr_items,
                "fuse_ocr": fuse_ocr,
                "ocr_semantic": ocr_semantic.detach(),
                "self_sem": conscious_out.self_sem.detach(),
                "cons_intent_sem": conscious_out.intent_sem.detach(),
                "cons_extras": self.DetachRuntimeObject(conscious_out.extras),
                "intent_sem": intent_sem.detach(),
                "sym_probs": sym_probs.detach(),
                "intention_extras": self.DetachRuntimeObject(intention_extras),
                "g_ultimate": goals["g_ultimate"].detach(),
                "g_long": goals["g_long"].detach(),
                "g_mid": goals["g_mid"].detach(),
                "ultimate_index": goals["ultimate_index"],
                "long_index": goals["long_index"],
                "mid_index": goals["mid_index"],}
        else:
            g_short = self.goal_manager.ShortGoal(
                self.slow_cache["g_ultimate"],
                self.slow_cache["g_mid"],
                pst_summary,
                self.prev_refinement_dir)
            goals = {
                "g_ultimate": self.slow_cache["g_ultimate"],
                "g_long": self.slow_cache["g_long"],
                "g_mid": self.slow_cache["g_mid"],
                "g_short": g_short,
                "ultimate_index": self.slow_cache["ultimate_index"],
                "long_index": self.slow_cache["long_index"],
                "mid_index": self.slow_cache["mid_index"],}
        grounding = self.goal_grounding(goals["g_short"], intent_sem, pst)

        saveModuleOutput("PhysicalState", {
            "P": pst["P"], "ARaw": pst["ARaw"], "M": pst["M"], "MphysRaw": pst["MphysRaw"], "R": pst["R"], "U": pst["U"],
            "LevelProb": pst["LevelProb"],
            "ObjectClassProb": pst["ObjectClassProb"],
            "PartClassProb": pst["PartClassProb"],
            "ParentProb": pst["ParentProb"],
            "Size": pst["Size"], "StateRaw": pst["StateRaw"], "AffordanceRaw": pst["AffordanceRaw"],
            "ExternalRelationProbRaw": pst["ExternalRelationProbRaw"],
            "MotionRaw": pst["MotionRaw"], "MovingProbRaw": pst["MovingProbRaw"],
            "ContactProbRaw": pst["ContactProbRaw"], "ContactForceRaw": pst["ContactForceRaw"],
            "Visibility": pst["Visibility"], "Occlusion": pst["Occlusion"],
            "HasTextProb": pst["HasTextProb"], "TextEmbed": pst["TextEmbed"],
            "SymbolProb": pst["SymbolProb"],
            "InteractionSuccessProb": pst["InteractionSuccessProb"],
            "Observed": pst["Observed"], "LastSeen": pst["LastSeen"],
            "ExecutedAction": pst["ExecutedAction"],
            "pst_binding": w_out["pst_binding"], "summary": pst_summary,
            "node_valid_mask": pst["M"]})
        saveModuleOutput("Goals", {
            "g_ultimate": goals["g_ultimate"],
            "g_long": goals["g_long"], "g_mid": goals["g_mid"], "g_short": goals["g_short"],
            "ultimate_index": goals["ultimate_index"],
            "long_index": goals["long_index"], "mid_index": goals["mid_index"]})
        saveModuleOutput("GoalGrounding", {
            "referenced_object_probs": grounding["referenced_object_probs"],
            "reference_confidence": grounding["reference_confidence"],
            "no_slot_prob": grounding["no_slot_prob"],
            "reference_distribution": grounding["reference_distribution"],
            "subgoal_skill_logits": grounding["subgoal_skill_logits"],
            "subgoal_slot_logits": grounding["subgoal_slot_logits"]})

        aug_actor_kwargs = {
            "uncertainty": unc_sig,
            "confidence": confidence_sig,
            "worldHzx": world_hzx_now,
            "prevDecisionState": self.prev_decision_state,
            "prevLatentControl": self.prev_latent_control,
            "prevActionEmbed": self.prev_decision_feedback_embed,
            "prevMapperHidden": self.prev_mapper_hidden,
            "prevTdError": self.prev_td_error,}
        base_act_out = self.actor(stateFeat=mem_feat,intentFeat=intent_sem,sample=sampleActions,
                                  deterministic=deterministicActor,prevOptionLogit=self.prev_option_logit,
                                  valueTensor=value_current, vNextTensor=value_next_current,
                                  **aug_actor_kwargs)
        decision_uncertainty = base_act_out["decision_uncertainty"].detach()
        world_abstract = self.RuntimeModule(self.world).BuildWorldAbstract(
            w_out,
            pst,
            pst_summary,
            unc_sig,
            confidence_sig,)

        # --- Embodied-AGI v2: endpoint pose feature -> neuro-symbolic plan -> decoupled endpoint command ---
        satisfaction_out = self.satisfaction(
            goals["g_short"],
            pst["SRaw"],
            pst["MphysRaw"],
            endpoint_pose_encoding.endpoint_pose_tokens,
            endpoint_pose_encoding.endpoint_pose_feat,)

        referenced = grounding["reference_distribution"].clamp_min(1e-6)
        intent_novelty = -(referenced * referenced.log()).sum(dim=-1) / math.log(referenced.size(-1))
        recent_failure = (self.prev_failure_count / 5.0).clamp(0.0, 1.0)
        temporal_context = self.temporal_gate.BuildContext(
            activeMask=self.temporal_active_mask,
            actionAge=self.temporal_action_age,
            feedbackAge=self.temporal_feedback_age,
            noSlotProb=grounding["no_slot_prob"].detach(),
            referenceConfidence=grounding["reference_confidence"].detach(),
            satisfactionProb=satisfaction_out["p_satisfied"].detach(),
            safetyRisk=risk_sig,
            interruptRisk=torch.maximum(risk_sig, decision_uncertainty),
            observationFreshness=1.0 - grounding["no_slot_prob"].detach(),
            canInterrupt=torch.ones_like(risk_sig),
            hardStop=(risk_sig > 0.98).float(),
            plannerProgress=robotState["planner_progress"],
            plannerTrackingError=robotState["planner_tracking_error"],
            plannerExecuting=robotState["planner_executing"],
            plannerReached=robotState["planner_reached"],
            plannerFailed=robotState["planner_failed"],
            plannerCanceled=robotState["planner_canceled"],)
        temporal_goal = self.goal_manager.TemporalGoal(goals["g_short"], temporal_context.feat)
        goals["temporal_goal"] = temporal_goal
        belief_feat = base_act_out["belief"]
        neuro_symbolic_out = self.neuro_symbolic(
            pst=pst,
            goalEmbed=goals["g_short"],
            worldBelief=world_hzx_now,
            decisionBelief=belief_feat,
            endpointPoseFeat=endpoint_pose_encoding.endpoint_pose_feat,
            uncertainty=torch.maximum(unc_sig, decision_uncertainty),
            novelty=risk_sig,
            recentFailure=recent_failure,
            intentNovelty=intent_novelty,
            satisfactionProb=satisfaction_out["p_satisfied"].detach(),
            referenced=grounding["referenced_object_probs"].detach(),
            referenceConfidence=grounding["reference_confidence"].detach(),
            noSlotProb=grounding["no_slot_prob"].detach(),
            temporalContextFeat=temporal_context.feat,
            returnExplain=self.save_module_messager_output,)
        act_out = self.actor.RefineWithNeuroSymbolic(
            base_act_out,
            neuro_symbolic_out,
            endpoint_pose_encoding.endpoint_pose_feat,
            world_abstract["world_hzx"],
            world_abstract["pst_summary"],
            temporal_context.feat,
            temporal_goal,)
        saveModuleOutput("Decision", act_out)

        decoupled_decision = self.decision_decoupler(
            decisionBackbone=act_out["decision_feature"],
            planLatent=act_out["decoder_plan_latent"],
            subgoalFeature=act_out["decoder_subgoal_feature"],
            constraintTokens=act_out["decoder_constraint_tokens"],
            endpointPoseEncoding=endpoint_pose_encoding,
            baseEndpointPose=endpoint_pose,)

        # Kept before the planner override so the CEM elites can supervise the network heads.
        network_decision_tensor = decoupled_decision.decision_tensor
        planner_prior = None
        if self.planner is not None:
            with torch.no_grad():
                planner_prior = self.planner.Plan(
                    decisionTensor=decoupled_decision.decision_tensor.detach(),
                    endpointPose=endpoint_pose,
                    endpointPoseEncoding=endpoint_pose_encoding,
                    h0=self.prev_world_h,
                    z0=self.prev_world_z,
                    x0=self.prev_world_x,
                    physicalState=pst,
                    returnDiagnostics=self.planner_teacher_mode,)
            if self.use_planner:
                planned_decision_tensor = self.decision_decoupler.MaskDecisionTensor(planner_prior["decision_tensor"].detach())
                planner_prior["decision_tensor"] = planned_decision_tensor
                planned_target_pose = self.decision_decoupler.DecodeEndpointPose(endpoint_pose, planned_decision_tensor)
                planned_feedback_embed = self.decision_decoupler.EncodeDecisionFeedback(
                    planned_decision_tensor,
                    planned_target_pose,
                    endpoint_pose_encoding,)
                decoupled_decision.decision_tensor = planned_decision_tensor
                decoupled_decision.target_endpoint_pose = planned_target_pose
                decoupled_decision.decision_feedback_embed = planned_feedback_embed
        candidate_motion_command = self.decision_decoupler.ToMotionCommand(decoupled_decision)
        candidate_safety_risk = 1.0 - decoupled_decision.safety_scores.mean(dim=-1)
        gate_context = self.temporal_gate.BuildContext(
            activeMask=self.temporal_active_mask,
            actionAge=self.temporal_action_age,
            feedbackAge=self.temporal_feedback_age,
            noSlotProb=grounding["no_slot_prob"].detach(),
            referenceConfidence=grounding["reference_confidence"].detach(),
            satisfactionProb=satisfaction_out["p_satisfied"].detach(),
            safetyRisk=torch.maximum(risk_sig, candidate_safety_risk),
            interruptRisk=torch.maximum(act_out["temporal_decision"]["p_interrupt"], risk_sig),
            observationFreshness=1.0 - grounding["no_slot_prob"].detach(),
            canInterrupt=torch.ones_like(risk_sig),
            hardStop=(torch.maximum(risk_sig, candidate_safety_risk) > 0.98).float(),
            plannerProgress=robotState["planner_progress"],
            plannerTrackingError=robotState["planner_tracking_error"],
            plannerExecuting=robotState["planner_executing"],
            plannerReached=robotState["planner_reached"],
            plannerFailed=robotState["planner_failed"],
            plannerCanceled=robotState["planner_canceled"],)
        active_motion_command = self.active_motion_command if self.active_motion_command is not None else candidate_motion_command
        # STAC-style drift integral over the active option's lifetime:
        # per-step invoke drift plus same-operator target-binding drift.
        invoke_drift = (
            self.temporal_invoke_drift
            + act_out["temporal_decision"]["invoke_delta"]
            + act_out["temporal_decision"]["reference_drift"])
        temporal_envelope = self.temporal_gate(
            gate_context,
            act_out["temporal_decision"],
            candidate_motion_command,
            active_motion_command,
            endpoint_pose,
            self.temporal_action_epoch,
            invoke_drift,)
        motion_command = temporal_envelope.motion_command
        executed_feedback_embed = self.decision_decoupler.EncodeDecisionFeedback(
            motion_command.decision_tensor,
            motion_command.target_endpoint_pose,
            endpoint_pose_encoding,)
        kind_id = temporal_envelope.kind_id
        start_mask = ((kind_id == DISPATCH) | (kind_id == REDISPATCH)).float()
        continue_mask = (kind_id == CONTINUE).float()
        self.temporal_action_epoch = self.temporal_action_epoch + start_mask
        temporal_active_next = torch.maximum(start_mask, continue_mask * self.temporal_active_mask)
        self.temporal_action_age = continue_mask * (self.temporal_action_age + 1.0)
        self.temporal_feedback_age = continue_mask * (self.temporal_feedback_age + 1.0)
        self.temporal_invoke_drift = (invoke_drift * continue_mask).detach()
        self.temporal_active_mask = temporal_active_next
        self.temporal_active_kind = kind_id.detach()
        self.active_motion_command = motion_command
        self.prev_target_endpoint_pose = motion_command.target_endpoint_pose.detach()

        act_out["satisfaction"] = satisfaction_out
        act_out["neuro_symbolic"] = neuro_symbolic_out
        act_out["decoupled_decision"] = decoupled_decision
        act_out["candidate_motion_command"] = candidate_motion_command
        act_out["motion_command"] = motion_command
        act_out["temporal_context"] = gate_context
        act_out["temporal_envelope"] = temporal_envelope
        act_out["goals"] = goals
        act_out["physical_state"] = pst
        act_out["world_abstract"] = world_abstract
        act_out["observed_physical_state"] = observed_pst
        act_out["goal_grounding"] = grounding
        act_out["planner_prior"] = planner_prior

        not_satisfied = self.satisfaction.IsNotSatisfied(satisfaction_out["p_satisfied"])
        self.prev_failure_count = (self.prev_failure_count + 1.0) * not_satisfied
        self.prev_refinement_dir = satisfaction_out["refinement_dir"].detach()

        saveModuleOutput("DecisionDecoupler", decoupled_decision)
        saveModuleOutput("TemporalExecution", temporal_envelope)
        saveModuleOutput("MotionCommand", motion_command)
        saveModuleOutput("Satisfaction", satisfaction_out)
        saveModuleOutput("NeuroSymbolic", neuro_symbolic_out)

        decision_feedback_embed = executed_feedback_embed
        entropy_actor = act_out["entropy"] # [B]
        next_option_logit = act_out["prevOptionLogit_next"].detach() # [B, K_option]

        self.prev_option_logit = next_option_logit # [B, K_option]

        self.prev_mem = mem_feat.detach() # [B, D_mem]
        self.prev_attn = atten_out.detach() # [B, D_attn]
        self.prev_world_h = w_out["h_next"].detach() # [B, D_world_h]
        self.prev_world_z = w_out["z_next"].detach() # [B, D_world_z] (PST-conditioned stochastic latent)
        self.prev_world_x = w_out["x_next"].detach() # [B, D_world_x]
        self.prev_camera_pose_world = camera_pose_world.detach()

        self.prev_entropy = entropy_actor.detach() # [B]
        self.prev_decision_state = act_out["decision_state_next"]
        self.prev_latent_control = act_out["latent_control_next"]
        self.prev_decision_feedback_embed = decision_feedback_embed.detach()
        self.prev_executed_decision_feedback_embed = executed_feedback_embed.detach()
        self.prev_mapper_hidden = act_out["mapper"]["hidden_next"]
        self.prev_td_error = td_sig
        if bool(done_now.any().item()):
            self.ResetHebbianMemory(doneMask=done_now)
            self.neuro_symbolic.ResetPlan(doneMask=done_now)
            self.slow_step_count = 0
            if self.perception_recall_loss is not None:
                self.perception_recall_loss.ResetIdentityBank()
            # New episode: the next frame has no valid previous visual/camera state.
            self.prev_camera_pose_world = None
            self.prev_visual_state = None
            self.visual_state_buffer = []
            self.perc_buffer = []
            done_keep = (1.0 - done_now.float())
            self.temporal_active_mask = self.temporal_active_mask * done_keep
            self.temporal_action_age = self.temporal_action_age * done_keep
            self.temporal_feedback_age = self.temporal_feedback_age * done_keep
            # New episode: no prior command, so reset the tracking reference to identity.
            self.prev_target_endpoint_pose = self.prev_target_endpoint_pose * done_keep.view(-1, 1, 1)
            self.prev_target_endpoint_pose[..., 6] += (1.0 - done_keep).view(-1, 1)
            self.temporal_invoke_drift = self.temporal_invoke_drift * done_keep
            self.active_motion_command = None

        if self.need_trace and not isTrain:
            def trace_tensor(t: torch.Tensor) -> torch.Tensor:
                return t.detach() if isinstance(t, torch.Tensor) else t

            trace = BrainStepTrace(
                PercBuffer=copy.deepcopy(self.perc_buffer), # List[[B, D_perc]]
                VisualBuffer=[
                    self.DetachVisualState(v, clone=True)
                    for v in self.visual_state_buffer],
                VisualStateNow=self.DetachVisualState(visual_state, clone=True),
                OcrSemantic=trace_tensor(ocr_semantic),
                IntentHint=trace_tensor(intent_hint_for_memory),
                ObsImg=fuse_ocr, # List[List[str]]
                ActionEmbed=trace_tensor(decision_feedback_embed),

                PercFeat=trace_tensor(perc_feats), # [B, D_perc]
                AttnFeat=trace_tensor(atten_out), # [B, D_attn]
                MemFeat=trace_tensor(mem_feat), # [B, D_mem]
                WorldState=trace_tensor(s_t), # [B, D_world]
                WorldDeltaTransport=trace_tensor(d_tr), # [B, D_world]
                WorldDeltaPhysics=trace_tensor(d_ph), # [B, D_world]
                ConsciousnessState=trace_tensor(conscious_out.intent_sem), # [B, D_cons]
                IntentionState=trace_tensor(intent_sem), # [B, D_intent]
                Reward=trace_tensor(r_t), # [B]
                Done=trace_tensor(d_t), # [B]
                ActionEntropy=trace_tensor(entropy_actor), # [B]

                extras= {},)
        
            self.history.append(trace)

        losses = {}

        if isTrain:
            world_loss = w_out["loss"]

            mem_loss = self.mem.GetInternalLoss()

            critic_loss = critic_out.loss if (critic_out.loss is not None) else world_loss.new_zeros(())
            critic_current_loss = critic_loss
            critic_transport_delayed_loss = world_loss.new_zeros(())
            if critic_out.extras is not None:
                critic_current_loss = critic_out.extras.get("loss_current_graph", critic_current_loss)
                critic_transport_delayed_loss = critic_out.extras.get(
                    "loss_transport_delayed_graph",
                    critic_transport_delayed_loss)

            conscious_loss = world_loss.new_zeros(())
            intention_loss = world_loss.new_zeros(())
            if slow_refresh:
                if conscious_out.extras is not None:
                    conscious_loss = conscious_out.extras.get("loss", conscious_loss)
                intention_loss, _ = self.intention.GetInternalLoss(sym_probs)

            perception_loss = world_loss.new_zeros(())
            perception_recall_losses: Dict[str, torch.Tensor] = {}
            if perceptionTargets is not None:
                perception_loss = self.perc.ComputePerceptionLoss(
                    visual_state,
                    depthTarget=perceptionTargets["depth"],
                    depthTargetValid=perceptionTargets["depth_valid"],
                    prevVisualState=prev_visual_for_loss,
                    cameraMotion=camera_motion_from_prev)
                if self.perception_recall_loss is not None:
                    recall_out = self.RuntimeModule(self.perc).recall_heads(visual_state)
                    perception_recall_losses = self.perception_recall_loss(recall_out, perceptionTargets)
                    perception_loss = perception_loss + perception_recall_losses["loss"]

            world_prediction_loss = world_loss.new_zeros(())
            world_prediction_losses: Dict[str, torch.Tensor] = {}
            if (B == 1
                and not bool(prev_done_for_prediction.any().item())):
                pred_train = self.world.PredictNextVisualFromPosterior(
                    prev_world_h_for_prediction,
                    prev_world_z_for_prediction,
                    prev_world_x_for_prediction,
                    physicalState=prev_physical_state_for_prediction,
                    actionEnc=world_action_feedback,
                    sample=False,)
                world_prediction_losses = self.world.ComputePredictionLoss(
                    predictedVisual=pred_train["predicted_visual"],
                    reconstructedVisualState=pred_train["reconstructed_visual_state"],
                    targetVisualState=visual_state,
                    precision=precision_sig,)
                world_prediction_loss = world_prediction_losses.get("loss_pred_total", world_prediction_loss)

            actor_loss = world_loss.new_zeros(())
            value_consistency_loss = world_loss.new_zeros(())
            # Magnitude-aware goal progress (projection on the decoded mid-goal direction)
            # plus the active-inference epistemic term shape the option advantage.
            world_hzx_prev = torch.cat([
                prev_world_h_for_prediction,
                prev_world_z_for_prediction,
                prev_world_x_for_prediction], dim=-1)
            alive_prev = 1.0 - prev_done_for_prediction.float()
            world_delta = (world_hzx_now.detach() - world_hzx_prev) * alive_prev.unsqueeze(-1)
            goal_progress = self.goal_manager.ProjectedProgress(world_delta, goals["g_mid"])
            epistemic_bonus = act_out["efe"]["epistemic"] / float(self.RuntimeModule(self.actor).u_dim)
            advantage = (td_sig + 0.1 * goal_progress + 0.05 * epistemic_bonus).detach()
            logp_terms = []
            if "logp_option" in act_out.get("option", {}):
                logp_terms.append(act_out["option"]["logp_option"])
            if len(logp_terms) > 0:
                logp_sum = torch.stack(logp_terms, dim=0).sum(dim=0)
                actor_loss = -(advantage * logp_sum).mean()
            goal_progress_loss = -goal_progress.mean()

            planner_distill_loss = world_loss.new_zeros(())
            if planner_prior is not None:
                planner_distill_loss = nn.functional.smooth_l1_loss(
                    network_decision_tensor,
                    planner_prior["decision_tensor"].detach())

            # Embodied-AGI v2 auxiliary objectives. Goal/codebook terms only exist on
            # slow-refresh steps where the long/mid heads actually ran.
            goal_align_loss = world_loss.new_zeros(())
            codebook_util_loss = world_loss.new_zeros(())
            if slow_refresh:
                goal_align_loss = self.goal_manager.AlignmentLoss(goals["g_ultimate"], intent_sem)
                codebook_util_loss = (
                    self.goal_manager.ultimate_head.UtilizationLoss(goals["ultimate_logits"])
                    + self.goal_manager.long_head.UtilizationLoss(goals["long_logits"])
                    + self.goal_manager.mid_head.UtilizationLoss(goals["mid_logits"]))
            symbolic_sparsity_loss = neuro_symbolic_out.invoke_mask.mean()
            grounding_loss = intent_novelty.mean()
            refinement_reg_loss = satisfaction_out["refinement_dir"].square().mean()
            # p_satisfied drives symbolic goal_done and the failure counter, so train it
            # against the real interaction success label.
            satisfaction_losses = self.satisfaction.SatisfactionLoss(
                satisfaction_out["sat_logits"], perceptionTargets["interaction_success"])
            satisfaction_loss = satisfaction_losses["total"]
            temporal_kind_loss = world_loss.new_zeros(())
            if "temporal_kind" in perceptionTargets:
                temporal_kind_loss = nn.functional.cross_entropy(
                    temporal_envelope.kind_logits, perceptionTargets["temporal_kind"])
            temporal_duration_loss = world_loss.new_zeros(())
            if "temporal_duration_ms" in perceptionTargets:
                temporal_duration_loss = nn.functional.smooth_l1_loss(
                    temporal_envelope.duration_ms / 1000.0,
                    perceptionTargets["temporal_duration_ms"] / 1000.0)
            v2_aux_loss = (
                goal_align_loss
                + 0.05 * symbolic_sparsity_loss
                + 0.05 * grounding_loss
                + 0.01 * refinement_reg_loss
                + 0.5 * satisfaction_loss
                + 0.05 * goal_progress_loss
                + 0.01 * codebook_util_loss
                + 0.5 * temporal_kind_loss
                + 0.05 * temporal_duration_loss)

            physical_loss = world_loss.new_zeros(())
            physical_losses: Dict[str, torch.Tensor] = {}
            if perceptionTargets is not None and "node_valid" in perceptionTargets:
                physical_losses = self.pst_loss(observed_pst, perceptionTargets)
                physical_loss = physical_losses["loss"]

            total_current_loss = (
                world_loss
                + mem_loss
                + critic_current_loss
                + conscious_loss
                + intention_loss
                + 0.05 * perception_loss
                + 0.05 * world_prediction_loss
                + 0.1 * actor_loss
                + planner_distill_loss
                + 0.05 * value_consistency_loss
                + 0.05 * v2_aux_loss
                + physical_loss)
            total_loss = total_current_loss

            losses["goal_align_loss"] = goal_align_loss
            losses["symbolic_sparsity_loss"] = symbolic_sparsity_loss
            losses["grounding_loss"] = grounding_loss
            losses["refinement_reg_loss"] = refinement_reg_loss
            losses["goal_progress_loss"] = goal_progress_loss
            losses["codebook_util_loss"] = codebook_util_loss
            losses["planner_distill_loss"] = planner_distill_loss
            losses["temporal_kind_loss"] = temporal_kind_loss
            losses["temporal_duration_loss"] = temporal_duration_loss
            losses["satisfaction_loss"] = satisfaction_loss
            losses["sat_success_loss"] = satisfaction_losses["sat_success_loss"]
            losses["v2_aux_loss"] = v2_aux_loss
            losses["physical_loss"] = physical_loss
            for name, value in perception_recall_losses.items():
                losses[f"perception_recall_{name}"] = value
            for name, value in physical_losses.items():
                if name != "loss":
                    losses[f"physical_{name}"] = value

            losses["world_loss"] = world_loss
            losses["memory_loss"] = mem_loss
            losses["critic_loss"] = critic_loss
            losses["critic_current_loss"] = critic_current_loss
            losses["critic_transport_delayed_loss"] = critic_transport_delayed_loss
            losses["conscious_loss"] = conscious_loss
            losses["intention_loss"] = intention_loss
            losses["perception_loss"] = perception_loss
            losses["world_prediction_loss"] = world_prediction_loss
            losses["actor_loss"] = actor_loss
            losses["value_consistency_loss"] = value_consistency_loss
            for name, value in world_prediction_losses.items():
                losses[f"world_{name}"] = value
            losses["total_current_loss"] = total_current_loss
            losses["total_loss"] = total_loss
            saveModuleOutput("Losses", losses)

        return {
            "decision": act_out,
            "world": {"state": s_t, "reward": r_t, "done": d_t}, # state:[B, D_world], reward/done:[B]
            "critic": critic_out,
            "features": {
                "perc": percs_seq,
                "attn": atten_out,
                "mem": mem_feat,
                "visualState": visual_state,
                "precision": precision_sig,
                "topDown": top_down,
                "keyPaddingMask": key_padding_mask}, # perc:[B, T, D_perc], attn:[B, D_attn], mem:[B, D_mem]
            "OCR": ocr_items,
            "intention_texts": intention_texts,
            "losses": losses}
        

    @torch.no_grad()
    def ExportBuffers(self) -> Dict[str, Any]:
        def runtime_module(mod: nn.Module) -> nn.Module:
            return mod.base if hasattr(mod, "base") else mod

        world_mod = runtime_module(self.world)
        attn_mod = runtime_module(self.attn)
        critic_mod = runtime_module(self.critic)

        h, z, x = world_mod.ExportState()
        A_prev = getattr(world_mod, "_A_prev", None)

        return {
            "prev_mem": self.prev_mem.detach().clone(),
            "prev_attn": self.prev_attn.detach().clone(),
            "prev_option_logit": self.prev_option_logit.detach().clone(),
            "prev_entropy": self.prev_entropy.detach().clone(),
            "prev_decision_state": self.prev_decision_state.detach().clone(),
            "prev_latent_control": self.prev_latent_control.detach().clone(),
            "prev_decision_feedback_embed": self.prev_decision_feedback_embed.detach().clone(),
            "prev_executed_decision_feedback_embed": self.prev_executed_decision_feedback_embed.detach().clone(),
            "prev_target_endpoint_pose": self.prev_target_endpoint_pose.detach().clone(),
            "prev_camera_pose_world": None if self.prev_camera_pose_world is None else self.prev_camera_pose_world.detach().clone(),
            "temporal_active_mask": self.temporal_active_mask.detach().clone(),
            "temporal_action_age": self.temporal_action_age.detach().clone(),
            "temporal_feedback_age": self.temporal_feedback_age.detach().clone(),
            "temporal_action_epoch": self.temporal_action_epoch.detach().clone(),
            "temporal_invoke_drift": self.temporal_invoke_drift.detach().clone(),
            "temporal_active_kind": self.temporal_active_kind.detach().clone(),
            "active_motion_command": self.DetachRuntimeObject(self.active_motion_command, clone=True),
            "prev_mapper_hidden": self.prev_mapper_hidden.detach().clone(),
            "prev_td_error": self.prev_td_error.detach().clone(),
            "world_state": {
                "h": h.detach().clone(),
                "z": z.detach().clone(),
                "x": x.detach().clone(),
                "s": self.prev_world_s.detach().clone(),
                "done": self.prev_done_flag.detach().clone(),
                "A_prev": None if A_prev is None else A_prev.detach().clone(),},
            "world_physical_state": world_mod.ExportPhysicalState(),
            "mem_state": self.mem.ExportState(),
            "mem_pending": copy.deepcopy(self.mem.pending),
            "attn_state": attn_mod.ExportState(),
            "critic_state": critic_mod.ExportState(),
            "perc_buffer": [t.detach().clone() for t in self.perc_buffer],
            "prev_visual_state": self.DetachVisualState(self.prev_visual_state, clone=True),
            "prev_predicted_visual": self.DetachRuntimeObject(self.prev_predicted_visual, clone=True),
            "prev_precision": self.prev_precision.detach().clone(),
            "prev_goal_bias": self.prev_goal_bias.detach().clone(),
            "prev_self_sem": None if self.prev_self_sem is None else self.prev_self_sem.detach().clone(),
            "prev_intent_sem": self.prev_intent_sem.detach().clone(),
            "visual_state_buffer": [
                self.DetachVisualState(v, clone=True)
                for v in self.visual_state_buffer
                if v is not None],
            "ocr_state": {
                "temporal_step": int(self.OCR._temporal_step),
                "last_batch_size": int(self.OCR._last_batch_size),
                "last_ocr_texts_batch": copy.deepcopy(self.OCR._last_ocr_texts_batch),
                "tracks_by_bi": copy.deepcopy(self.OCR._tracks_by_bi),},
            "history": copy.deepcopy(list(self.history)),
            "extra_mem": None if self.extra_mem is None else copy.deepcopy(self.extra_mem),
            "neuro_symbolic_plan": self.neuro_symbolic.ExportPlanState(),}

    @torch.no_grad()
    def ImportBuffers(self, state: Dict[str, Any]):
        def runtime_module(mod: nn.Module) -> nn.Module:
            return mod.base if hasattr(mod, "base") else mod

        def move_to_device(x, device: torch.device):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, list):
                return [move_to_device(v, device) for v in x]
            if isinstance(x, tuple):
                return tuple(move_to_device(v, device) for v in x)
            if isinstance(x, deque):
                return deque((move_to_device(v, device) for v in x), maxlen=x.maxlen)
            if isinstance(x, dict):
                return {k: move_to_device(v, device) for k, v in x.items()}
            if hasattr(x, "__dataclass_fields__"):
                vals = {name: move_to_device(getattr(x, name), device) for name in x.__dataclass_fields__.keys()}
                return type(x)(**vals)
            return x

        device = next(self.parameters()).device
        state = move_to_device(copy.deepcopy(state), device)

        world_mod = runtime_module(self.world)
        attn_mod = runtime_module(self.attn)
        critic_mod = runtime_module(self.critic)

        self.prev_mem = state["prev_mem"]
        self.prev_attn = state["prev_attn"]
        self.prev_option_logit = state["prev_option_logit"]
        prev_entropy = state["prev_entropy"]
        if isinstance(prev_entropy, torch.Tensor) and (prev_entropy.dim() > 1) and (prev_entropy.size(-1) == 1):
            prev_entropy = prev_entropy.squeeze(-1)
        self.prev_entropy = prev_entropy

        self.prev_decision_state = state["prev_decision_state"]
        self.prev_latent_control = state["prev_latent_control"]
        self.prev_decision_feedback_embed = state["prev_decision_feedback_embed"]
        self.prev_executed_decision_feedback_embed = state["prev_executed_decision_feedback_embed"]
        self.prev_target_endpoint_pose = state["prev_target_endpoint_pose"]
        self.prev_camera_pose_world = state["prev_camera_pose_world"]
        self.temporal_active_mask = state["temporal_active_mask"]
        self.temporal_action_age = state["temporal_action_age"]
        self.temporal_feedback_age = state["temporal_feedback_age"]
        self.temporal_action_epoch = state["temporal_action_epoch"]
        self.temporal_invoke_drift = state["temporal_invoke_drift"]
        self.temporal_active_kind = state["temporal_active_kind"]
        self.active_motion_command = state["active_motion_command"]
        self.prev_mapper_hidden = state["prev_mapper_hidden"]
        self.prev_td_error = state["prev_td_error"]

        world_state = state["world_state"]
        world_mod.ImportState(world_state["h"], world_state["z"], world_state["x"])
        world_mod._A_prev = None if world_state["A_prev"] is None else world_state["A_prev"].detach().clone()
        world_mod.ImportPhysicalState(state["world_physical_state"])
        self.prev_world_h = world_state["h"].detach().clone()
        self.prev_world_z = world_state["z"].detach().clone()
        self.prev_world_x = world_state["x"].detach().clone()
        self.prev_world_s = world_state["s"]
        self.prev_done_flag = world_state["done"]

        self.mem.EnsureB(int(self.prev_mem.size(0)), device=device, dtype=self.prev_mem.dtype)
        self.mem.ImportState(state["mem_state"], importGws=True, importLtm=True, importSym=True)
        self.mem.pending = state["mem_pending"]
        attn_mod.ImportState(state["attn_state"])
        critic_mod.ImportState(state["critic_state"])

        self.perc_buffer = state["perc_buffer"]
        self.prev_visual_state = state["prev_visual_state"]
        self.prev_predicted_visual = state["prev_predicted_visual"]
        self.prev_precision = state["prev_precision"]
        self.prev_goal_bias = state["prev_goal_bias"]
        self.prev_self_sem = state["prev_self_sem"]
        self.prev_intent_sem = state["prev_intent_sem"]
        self.visual_state_buffer = state["visual_state_buffer"]

        ocr_state = state["ocr_state"]
        self.OCR._temporal_step = int(ocr_state["temporal_step"])
        self.OCR._last_batch_size = int(ocr_state["last_batch_size"])
        self.OCR._last_ocr_texts_batch = ocr_state["last_ocr_texts_batch"]
        self.OCR._tracks_by_bi = ocr_state["tracks_by_bi"]

        self.history = deque(state["history"], maxlen=self.history_len)
        self.extra_mem = state["extra_mem"]
        self.neuro_symbolic.ImportPlanState(state["neuro_symbolic_plan"])
        self.thread_end = True
        self.slow_step_count = 0
        self.slow_cache = None
        self.buf_B = int(self.prev_mem.size(0))


    def SmoothCorrection(
        self,
        wmSeq: torch.Tensor,
        extLast: torch.Tensor,
        *,
        q: float = 0.05,
        rWm: float = 0.5,
        rExt: float = 0.05,
        initVar: float = 1.0,) -> torch.Tensor:
        B, T = wmSeq.shape

        device = wmSeq.device
        dtype = wmSeq.dtype

        q_t = torch.as_tensor(float(q), device=device, dtype=dtype)
        r_wm_t = torch.as_tensor(float(rWm), device=device, dtype=dtype)
        r_ext_t = torch.as_tensor(float(rExt), device=device, dtype=dtype)

        x_filt = wmSeq.new_zeros(B, T)
        P_filt = wmSeq.new_zeros(B, T)

        x_filt[:, 0] = wmSeq[:, 0]
        P_filt[:, 0] = wmSeq.new_full((B,), float(initVar))

        for t in range(1, T):
            x_prior = x_filt[:, t - 1]
            P_prior = P_filt[:, t - 1] + q_t

            z_t = wmSeq[:, t]

            K_t = P_prior / (P_prior + r_wm_t + 1e-8)

            x_post = x_prior + K_t * (z_t - x_prior)
            P_post = (1.0 - K_t) * P_prior

            x_filt[:, t] = x_post
            P_filt[:, t] = P_post

        x_T = x_filt[:, -1]
        P_T = P_filt[:, -1]
        ext_vec = extLast

        K_ext = P_T / (P_T + r_ext_t + 1e-8)  
        x_T_corr = x_T + K_ext * (ext_vec - x_T) 
        P_T_corr = (1.0 - K_ext) * P_T 

        x_filt[:, -1] = x_T_corr
        P_filt[:, -1] = P_T_corr

        x_smooth = x_filt.clone()
        P_smooth = P_filt.clone()

        for t in range(T - 2, -1, -1):
            P_t = P_filt[:, t]
            P_tp = P_t + q_t
            C_t = P_t / (P_tp + 1e-8)

            x_smooth[:, t] = x_filt[:, t] + C_t * (x_smooth[:, t + 1] - x_filt[:, t])
            P_smooth[:, t] = P_t + C_t * C_t * (P_smooth[:, t + 1] - P_tp)

        return x_smooth  # [B,T]

    def SmoothWorkerLoop(self):
        while True:
            task = self.smooth_queue.get()
            self.SmoothWork(*task)

    def SmoothWork(self, historyRef, lastRef, signal: str, attenModule: nn.Module, memModule: nn.Module, criticModule: nn.Module):
        try:
            visual_buffer_list = []
            visual_state_list = []
            ocr_semantic_list = []
            intent_hint_list = []
            atten_list = []
            mem_list = []
            world_state_list = []
            world_dtr_list = []
            world_dph_list = []
            reward_list = []
            done_list = []
            entropy_list = []

            for tr in historyRef:
                visual_buffer_list.append(tr.VisualBuffer)
                visual_state_list.append(tr.VisualStateNow)
                ocr_semantic_list.append(tr.OcrSemantic)
                intent_hint_list.append(tr.IntentHint)
                atten_list.append(tr.AttnFeat)
                mem_list.append(tr.MemFeat)
                world_state_list.append(tr.WorldState)
                world_dtr_list.append(tr.WorldDeltaTransport)
                world_dph_list.append(tr.WorldDeltaPhysics)
                reward_list.append(tr.Reward) 
                done_list.append(tr.Done)  
                entropy_list.append(tr.ActionEntropy)

            ref_seq = next((t for t in reward_list + done_list if isinstance(t, torch.Tensor)), None)
            if ref_seq is None:
                return

            B = int(ref_seq.size(0))
            ref_device = ref_seq.device
            ref_dtype = ref_seq.dtype

            def normalize_trace_signal(
                x: Optional[torch.Tensor],
                *,
                clamp01: bool = False,
                clampReward: bool = False) -> Optional[torch.Tensor]:
                if x is None:
                    return None
                out = x.detach().view(B)
                out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                if clampReward:
                    out = out.clamp(float(BasicParameters.REWARD_MIN), float(BasicParameters.REWARD_MAX))
                return out.clamp(0.0, 1.0) if clamp01 else out

            reward_list = [normalize_trace_signal(x, clampReward=True) for x in reward_list]
            done_list = [normalize_trace_signal(x, clamp01=True) for x in done_list]
            last_ref = normalize_trace_signal(
                lastRef,
                clamp01=(signal == "Done"),
                clampReward=(signal == "Reward"))
            if last_ref is None:
                return

            if signal == "Reward":
                seq_list = reward_list
            elif signal == "Done":
                seq_list = done_list
            else:
                return

            if len(seq_list) <= 1:
                return

            wm_seq = torch.stack(seq_list, dim=1).contiguous()  # [B, T]

            smoothed = self.SmoothCorrection(wmSeq=wm_seq, extLast=last_ref)

            smoothed_list = list(smoothed.unbind(dim=1))

            start = int(memModule.time_step.min().item())

            with torch.no_grad():
                for i in range(1, len(smoothed_list)):
                    if signal == "Reward":
                        reward_in = smoothed_list[i]
                    else: # Done
                        reward_in = reward_list[i]

                    value = criticModule(
                        memoryPrev=mem_list[i-1],
                        attnPrev=atten_list[i-1],
                        state=world_state_list[i],
                        rewardModel=reward_list[i],
                        policyEntropyPrev=entropy_list[i-1],
                        doneModel=done_list[i],
                        worldDeltaTransport=world_dtr_list[i],
                        worldDeltaPhysics=world_dph_list[i],)

                    td_sig = value.tdError.detach()
                    unc_sig = value.uncertainty.detach()
                    precision_sig = value.precision.detach()
                    emotion_sig = value.emotion.detach()
                    value_comps = value.rComps
                    risk_sig = value_comps["risk"].detach()
                    confidence_sig = value_comps["confidence"].detach()

                    percs_seq, object_seq, motion_seq, quality_seq, pred_error_seq, key_padding_mask = self.BuildVisualSequenceTensors(
                        visual_buffer_list[i],
                        batchSize=reward_in.size(0),
                        device=reward_in.device,
                        dtype=reward_in.dtype)
                    atten_out = attenModule(
                        percs_seq,
                        keyPaddingMask=key_padding_mask,
                        tdError=td_sig,
                        uncertainty=unc_sig,
                        objectSeq=object_seq,
                        motionSeq=motion_seq,
                        qualitySeq=quality_seq,
                        predErrorSeq=pred_error_seq,
                        goalBias=intent_hint_list[i],
                        precision=precision_sig,
                        applyPlasticity=True)
                    memModule(
                        atten_out,
                        tdError=td_sig,
                        emotion=emotion_sig,
                        reward=reward_in,
                        visualState=visual_state_list[i],
                        ocrSemantic=ocr_semantic_list[i],
                        intentHint=intent_hint_list[i],
                        uncertainty=unc_sig,
                        risk=risk_sig,
                        confidence=confidence_sig,
                        sourceLabel=MemoryType.SRC_IMAGINE)

                memModule.FlushPendingWrites()

            extra_state = memModule.ExportState(step=start)
            extra_state["memory_delta_base_step"] = torch.tensor(start, device=last_ref.device, dtype=torch.long)
            extra_state["memory_delta_new_step"] = memModule.time_step.detach().max()
            extra_state["memory_delta_kind"] = torch.tensor(1 if signal == "Reward" else 2, device=last_ref.device, dtype=torch.long)
            self.extra_mem = extra_state

        except Exception as e:
            print("[SmoothWork] error:", repr(e))
        finally:
            self.thread_end = True
            


class Agent:
    def __init__(self,
        brain: BrainCore,
        isTrain: bool,
        device: Union[str, torch.device] = "cpu",
        *,
        worldMemoryPath: str = None,
        memMemoryPath: str = None):

        self.device = torch.device(device)

        self.brain = brain
        self.is_train = isTrain

        self.wm_mem_path = worldMemoryPath
        self.mem_mem_path = memMemoryPath

        if self.wm_mem_path is not None:
            self.LoadWorldMemory(self.wm_mem_path)
        else:
            print(f"{self.wm_mem_path} is None")

        if self.mem_mem_path is not None:
            self.LoadAgentMemory(self.mem_mem_path)
        else:
            print(f"{self.mem_mem_path} is None")

        self.brain.to(self.device)

        self.ResetHebbianMemory()

        if isTrain:
            actor_params = self.CollectTrainableParams(
                self.brain.perc,
                self.brain.attn,
                self.brain.mem,
                self.brain.actor,
                self.brain.conscious,
                self.brain.intention,
                self.brain.pst_builder,
                self.brain.goal_manager,
                self.brain.goal_grounding,
                self.brain.decision_decoupler,
                self.brain.satisfaction,
                self.brain.neuro_symbolic,
                self.brain.temporal_gate)
        
            self.opt_actor = torch.optim.Adam(actor_params, lr=3e-4)

            self.opt_critic = torch.optim.Adam(self.brain.critic.parameters(), lr=2e-4)
            self.transport_manual_lr = 2e-4
            self.transport_manual_max_norm = 1.0
            self.transport_manual_weight_decay = 0.0

            self.opt_world = torch.optim.Adam(self.brain.world.parameters(), lr=2e-4)

    def CollectTrainableParams(self, *modules: nn.Module) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set[int] = set()
        for mod in modules:
            for p in mod.parameters():
                if (not p.requires_grad) or (id(p) in seen):
                    continue
                seen.add(id(p))
                params.append(p)
        return params
    
    def EnsureFile(self, path: str) -> bool:
        dir_ = os.path.dirname(path)
        if dir_ and (not os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
            print(f"[EnsureFile] created directory: {os.path.abspath(dir_)}")

        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[EnsureFile] file already exists: {os.path.abspath(path)} (size={size} bytes)")
            return False

        return True

    def GetRuntimeWorld(self):
        return self.brain.world.base if self.brain.is_online_learning else self.brain.world

    def LoadWorldMemory(self, path: str):
        self.EnsureFile(path)

        world = self.GetRuntimeWorld()
        world._use_memory = True
        world._mem_path = path

        valid = False
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                obj = torch.load(path, weights_only=False)
                if isinstance(obj, dict):
                    valid = all(k in obj for k in (
                        "mem_keys",
                        "mem_vals",
                        "mem_imp",
                        "mem_steps",
                        "mem_size",
                        "mem_global_step",))
            except Exception:
                valid = False

        if not valid:
            world.SaveMemory(path)

        world.LoadMemory(path, mapLocation=None, strict=False)

    def LoadAgentMemory(self, path: str):
        self.EnsureFile(path)

        mem = self.brain.mem
        valid = False
        obj = None
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                obj = torch.load(path, weights_only=False)
                if isinstance(obj, dict):
                    valid = ("state_dict" in obj) or ("h_state" in obj)
            except Exception:
                valid = False

        if not valid:
            mem.InitMemoryDocument(path)
            obj = torch.load(path, weights_only=False)

        if not isinstance(obj, dict):
            raise TypeError(f"Unexpected memory file format: {type(obj).__name__}")

        if "state_dict" in obj:
            mem.LoadState(path)
        else:
            mem.ImportState(obj, importGws=True, importLtm=True, importSym=True)

    def SaveRuntimeMemories(self):
        if self.wm_mem_path is not None:
            world = self.GetRuntimeWorld()
            world._use_memory = True
            world._mem_path = self.wm_mem_path
            world.SaveMemory(self.wm_mem_path)

        if self.mem_mem_path is not None:
            self.brain.mem.SaveState(self.mem_mem_path)

    def LoadTorchPayload(self, path: str):
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"Safe mode loading failed: {e}, try the normal mode")
            return torch.load(path, map_location=self.device)

    def LoadBrainWeights(self, path: str):
        payload = self.LoadTorchPayload(path)

        if isinstance(payload, dict) and "brain" in payload:
            brain_state = payload["brain"]
        elif isinstance(payload, dict):
            brain_state = payload
        else:
            raise TypeError(f"checkpoint {path} has invalid brain weights payload")

        if not self.brain.perc.recall_heads.enable_auxiliary:
            auxiliary_prefixes = (
                "perc.recall_heads.reconstruction_head.",
                "perc.recall_heads.patch_trunk.",
                "perc.recall_heads.patch_class_logits.",
                "perc.recall_heads.patch_depth.",
                "perc.recall_heads.patch_normal.")
            brain_state = {
                name: value
                for name, value in brain_state.items()
                if not name.startswith(auxiliary_prefixes)}
        self.brain.ResizeStateBuffersForLoad(brain_state)
        self.brain.load_state_dict(brain_state, strict=True)

    def LoadBrainStateDict(self, brainState: Dict[str, Any], strict: bool):
        self.brain.ResizeStateBuffersForLoad(brainState)
        self.brain.load_state_dict(brainState, strict=bool(strict))

    def ExportModuleMessagerData(self, nSteps: int = 0):
        return self.brain.moduleMessager.ExportDict(nSteps=nSteps)

    def SetCameraIntrinsics(
        self,
        intrinsics: torch.Tensor,
        sourceSize: Optional[Tuple[int, int]] = None) -> None:
        self.brain.SetCameraIntrinsics(intrinsics, sourceSize=sourceSize)

    def Act(
        self,
        frame: torch.Tensor, # [B,C,H,W]
        *,
        textExt: Optional[List[Optional[str]]] = None,
        reward: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        sampleActions: bool = True,
        deterministicActor: bool = False,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        robotPhysicalContext: torch.Tensor,
        interactionContext: torch.Tensor,
        perceptionTargets: Optional[Dict[str, torch.Tensor]] = None,
        robotState: Dict[str, torch.Tensor],):

        if self.is_train:
            step_out = self.brain.Step(
                frame,
                textExt,
                rewardExt=reward,
                doneFlag=done,
                isTrain=self.is_train,
                sampleActions=sampleActions,
                deterministicActor=deterministicActor,
                depth=depth,
                depthValid=depthValid,
                robotPhysicalContext=robotPhysicalContext,
                interactionContext=interactionContext,
                perceptionTargets=perceptionTargets,
                robotState=robotState,)
            motion_command = step_out["decision"]["motion_command"]
            act_out = {
                "motion_command": motion_command,
                "target_endpoint_pose": motion_command.target_endpoint_pose,
                "temporal_envelope": step_out["decision"]["temporal_envelope"],}
            act_out["decision"] = step_out["decision"]
            act_out["loss"] = step_out["losses"]["total_current_loss"]
            act_out["transport_delayed_loss"] = step_out["losses"]["critic_transport_delayed_loss"]
            act_out["total_loss"] = step_out["losses"]["total_loss"]
            act_out["physical_loss"] = step_out["losses"]["physical_loss"]
            act_out["OCR"] = step_out["OCR"]
            act_out["intention_texts"] = step_out.get("intention_texts", [])
            return act_out
        else:
            with torch.no_grad():
                step_out = self.brain.Step(
                    frame,
                    textExt,
                    rewardExt=reward,
                    doneFlag=done,
                    isTrain=self.is_train,
                    sampleActions=sampleActions,
                    deterministicActor=deterministicActor,
                    depth=depth,
                    depthValid=depthValid,
                    robotPhysicalContext=robotPhysicalContext,
                    interactionContext=interactionContext,
                    perceptionTargets=perceptionTargets,
                    robotState=robotState,)
                motion_command = step_out["decision"]["motion_command"]
                act_out = {
                    "motion_command": motion_command,
                    "target_endpoint_pose": motion_command.target_endpoint_pose,
                    "temporal_envelope": step_out["decision"]["temporal_envelope"],}
                act_out["decision"] = step_out["decision"]
                act_out["loss"] = None
                act_out["OCR"] = step_out["OCR"]
                act_out["intention_texts"] = step_out.get("intention_texts", [])
                return act_out


    def UnpackActPacked(self, actOut: Optional[Dict[str, Any]]) -> str:
        intention_texts: List[str] = []
        motion_command_value = actOut["motion_command"]
        intention_texts_value = actOut["intention_texts"]
        if isinstance(intention_texts_value, list):
            intention_texts = [str(item) for item in intention_texts_value]
        else:
            intention_texts = [str(intention_texts_value)]

        def to_json_scalar_or_list(value: Any) -> Any:
            if torch.is_tensor(value):
                tensor = value.detach().float().cpu()
                if tensor.dim() > 0:
                    tensor = tensor[0]
                if tensor.numel() == 1:
                    return float(tensor.reshape(-1)[0].item())
                return tensor.tolist()
            if isinstance(value, dict):
                return {str(k): to_json_scalar_or_list(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [to_json_scalar_or_list(v) for v in value]
            return value

        motion_command = {
            "decision_tensor": to_json_scalar_or_list(motion_command_value.decision_tensor),
            "target_endpoint_pose": to_json_scalar_or_list(motion_command_value.target_endpoint_pose),
            "endpoint_names": list(motion_command_value.endpoint_names),
            "gripper_cmd": to_json_scalar_or_list(motion_command_value.gripper_cmd),
            "mode_logits": to_json_scalar_or_list(motion_command_value.mode_logits),
            "safety_scores": to_json_scalar_or_list(motion_command_value.safety_scores),}
        temporal_value = actOut["temporal_envelope"]
        kind_id_tensor = temporal_value.kind_id.detach().cpu()
        kind_id0 = int(kind_id_tensor.reshape(-1)[0].item())
        temporal_envelope = {
            "kind": temporal_value.kind_names[kind_id0],
            "kind_id": to_json_scalar_or_list(temporal_value.kind_id),
            "kind_logits": to_json_scalar_or_list(temporal_value.kind_logits),
            "primitive_names": list(temporal_value.kind_names),
            "action_id": to_json_scalar_or_list(temporal_value.action_id),
            "action_epoch": to_json_scalar_or_list(temporal_value.action_epoch),
            "reason_logits": to_json_scalar_or_list(temporal_value.reason_logits),
            "duration_ms": to_json_scalar_or_list(temporal_value.duration_ms),
            "soft_timeout_ms": to_json_scalar_or_list(temporal_value.soft_timeout_ms),
            "hard_timeout_ms": to_json_scalar_or_list(temporal_value.hard_timeout_ms),
            "publish_motion_command": to_json_scalar_or_list(temporal_value.publish_motion_command),
            "reuse_active_motion_command": to_json_scalar_or_list(temporal_value.reuse_active_motion_command),
            "publish_stop_command": to_json_scalar_or_list(temporal_value.publish_stop_command),
            "publish_hold_command": to_json_scalar_or_list(temporal_value.publish_hold_command),
            "same_operator": to_json_scalar_or_list(temporal_value.same_operator),
            "operator_changed": to_json_scalar_or_list(temporal_value.operator_changed),
            "invoke_delta": to_json_scalar_or_list(temporal_value.invoke_delta),
            "reference_drift": to_json_scalar_or_list(temporal_value.reference_drift),
            "invoke_drift": to_json_scalar_or_list(temporal_value.invoke_drift),}

        return json.dumps({
            "motion_command": motion_command,
            "temporal_envelope": temporal_envelope,
            "intention_texts": intention_texts,}, ensure_ascii=False)


    def Save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.SaveRuntimeMemories()

        payload = {
            "brain": self.brain.state_dict(),
            "buffers": self.brain.ExportBuffers(),
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "wm_mem_path": self.wm_mem_path,
            "mem_mem_path": self.mem_mem_path,}
        
        if self.is_train:
            payload["opt_actor"] = self.opt_actor.state_dict()
            payload["opt_critic"] = self.opt_critic.state_dict()
            payload["opt_world"] = self.opt_world.state_dict()
        
        if torch.cuda.is_available():
            payload["rng_cuda_all"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)

    def Load(self, path: str, strict: bool = True, mapLocation: Optional[Union[str, torch.device]] = None):
        payload = torch.load(path, map_location=mapLocation or self.device, weights_only=False)

        if isinstance(payload, dict) and ("brain" in payload):
            self.LoadBrainStateDict(payload["brain"], strict=strict)

            if self.is_train:
                if "opt_actor" in payload:
                    try:
                        self.opt_actor.load_state_dict(payload["opt_actor"])
                    except ValueError:
                        print("[Agent.Load] opt_actor state skipped because parameter groups changed")
                if "opt_critic" in payload:
                    self.opt_critic.load_state_dict(payload["opt_critic"])
                if "opt_world" in payload:
                    self.opt_world.load_state_dict(payload["opt_world"])

            if "buffers" in payload:
                self.brain.ImportBuffers(payload["buffers"])

            try:
                if "rng_py" in payload:
                    random.setstate(payload["rng_py"])
                if "rng_np" in payload:
                    np.random.set_state(payload["rng_np"])
                if "rng_torch" in payload:
                    torch.set_rng_state(payload["rng_torch"])
                if torch.cuda.is_available() and ("rng_cuda_all" in payload):
                    torch.cuda.set_rng_state_all(payload["rng_cuda_all"])
            except Exception:
                traceback.print_exc()
        else:
            self.LoadBrainStateDict(payload, strict=strict)

        self.SaveRuntimeMemories()


    def ResetBrainState(self, B: int = 1, isOnlineLearning: Optional[bool] = None):
        if isOnlineLearning is None:
            isOnlineLearning = self.brain.is_online_learning

        if isOnlineLearning:
            self.brain.world.base.ResetState(batchSize=B)
            self.brain.critic.base.ResetState()
        else:
            self.brain.world.ResetState(batchSize=B)
            self.brain.critic.ResetState()

        self.brain.mem.SoftReset() 
        self.brain.conscious.ResetState()
        self.brain.OCR.ResetTemporal()

        self.brain.extra_mem = None
        self.brain.thread_end = True

        self.brain.ResetBuffers(B=B, isOnlineLearning=isOnlineLearning, device=self.device)

    def ResetHebbianMemory(self):
        self.brain.ResetHebbianMemory()

    def CaptureCriticTransportGrad(self) -> Dict[str, float]:
        critic = self.brain.critic
        if hasattr(critic, "CaptureTransportGrad"):
            return critic.CaptureTransportGrad(clearParamGrad=True)
        return {"captured": 0.0, "grad_norm": 0.0, "accum_steps": 0.0}

    def ApplyCriticTransportManualGrad(self) -> Dict[str, float]:
        critic = self.brain.critic
        if hasattr(critic, "ApplyTransportManualGrad"):
            return critic.ApplyTransportManualGrad(
                lr=self.transport_manual_lr,
                maxNorm=self.transport_manual_max_norm,
                weightDecay=self.transport_manual_weight_decay,
                clear=True)
        return {"updated": 0.0, "grad_norm": 0.0, "scale": 1.0}

    def AfterOptimizerStep(self):
        if hasattr(self.brain.critic, "AfterOptimizerStep"):
            self.brain.critic.AfterOptimizerStep()





    def UpdateWrappers(self, wrappers, action: str, **kwargs):
        results = []
        for w in wrappers:
            if not hasattr(w, "Update"):
                continue
            out = w.Update(action, **kwargs)
            results.append(out)
        return results
    
    def UpdateAllWrappers(self, action: str, **kwargs):
        wrappers = [self.brain.perc, self.brain.attn, self.brain.actor, self.brain.world, self.brain.critic, self.brain.intention]
        results = []
        for w in wrappers:
            if not hasattr(w, "Update"):
                continue
            out = w.Update(action, **kwargs)
            results.append(out)
        return results

    def GetModuleParamsCount(self, onlyTrainable: bool = True):
        def count_module_params(module: Optional[nn.Module]) -> int:
            if module is None:
                return 0

            total = 0
            for p in module.parameters():
                if onlyTrainable and not p.requires_grad:
                    continue
                total += p.numel()
            return total

        core_module_names = (
            "perc",
            "attn",
            "mem",
            "actor",
            "world",
            "critic",
            "conscious",
            "intention",
            "OCR",)

        per_module_counts: Dict[str, int] = {}
        for name in core_module_names:
            module = getattr(self.brain, name, None)
            if isinstance(module, nn.Module):
                per_module_counts[name] = count_module_params(module)

        aux_module_counts: Dict[str, int] = {}
        for child_name, child in self.brain.named_children():
            if child_name in per_module_counts:
                continue
            aux_module_counts[child_name] = count_module_params(child)

        core_total = sum(per_module_counts.values())
        aux_total = sum(aux_module_counts.values())
        full_total = count_module_params(self.brain)

        kind = "trainable" if onlyTrainable else "all"
        print(f"===== Core module parameter counts ({kind}) =====")
        for name, n in per_module_counts.items():
            print(f"{name:15s}: {n:,}")
        print("----------------------------------")
        print(f"TOTAL core {kind} params: {core_total:,}")

        if aux_module_counts:
            print("===== Auxiliary module parameter counts =====")
            for name, n in aux_module_counts.items():
                print(f"{name:15s}: {n:,}")
            print("----------------------------------")
            print(f"TOTAL auxiliary {kind} params: {aux_total:,}")

        if full_total != (core_total + aux_total):
            print(f"TOTAL full {kind} params: {full_total:,}")

        return core_total
