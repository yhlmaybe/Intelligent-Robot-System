from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
import threading
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

#import debugpy

from dataclasses import dataclass, field
from collections import deque

from PerceptionModule import PerceiveExtractor, PerceptionOnlineWrapper, TopDownContext, VisualState
from AttentionModule import AttentionExtractor, AttentionOnlineWrapper
from MemoryModule import MemoryExtractor, MemoryType
from DecisionModule import DecisionExtractor, DecisionOnlineWrapper, RAW_KEYBOARD_LAYOUT, DecisionPlannerExtractor, StableLogProbBernoulli
from WorldModule import RSSMWorldModel, WorldOnlineWrapper
from ValueEstimationModule import ValueEstimationExtractor,ValueEstimationOnlineWrapper
from ConsciousnessModule import ConsciousnessExtractor
from IntentionModule import IntentionExtractor, IntentionOnlineWrapper
from OCRModule import OCREngineExtractor
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

    CONSCIOUSNESSTEM = 1 * 1024

    SAVE_EVERY_SAMPLE_COUNT = 500

    DATA_ROOT_PATH = "BrainDeepLearn/Data"
    OCR_DATA_ROOT_PATH = "BrainDeepLearn/Data/OCR"

    DATA_FRAMES_PATH = "BrainDeepLearn/Data/frames"
    DATA_KEYS_PATH = "BrainDeepLearn/Data/keys"
    DATA_MOUSE_CLICK_PATH = "BrainDeepLearn/Data/mouse_click"
    DATA_MOUSE_MOVE_PATH = "BrainDeepLearn/Data/mouse_move"
    DATA_REWARD_PATH = "BrainDeepLearn/Data/reward"
    DATA_DONE_PATH = "BrainDeepLearn/Data/done"
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
    Keys: Optional[torch.Tensor] = None
    MouseClick: Optional[torch.Tensor] = None
    MouseDelta: Optional[torch.Tensor] = None

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
        saveModuleMessagerOutput: bool = True,):
        super().__init__()
        self.SEQ_LEN = seqLen
        self.is_online_learning = plasticOnlineLearning
        self.prioritize_ext_str = prioritizeExtStr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.perc = PerceiveExtractor(
            imgSize=BasicParameters.IMAGE_SIZE,
            embedDim=ModuleDim.PerceptionEmbed,
            useHebbian=plasticHebbian)
        
        self.attn = AttentionExtractor(
            embedDim=ModuleDim.AttentionFeat,
            sequenceLength=seqLen, 
            hebbianRate=(0.01 if plasticHebbian else 0.0), 
            useHebbian=plasticHebbian)
        
        self.mem = MemoryExtractor(
            inputDim=ModuleDim.AttentionFeat,
            ssmStateDim=ModuleDim.MemoryItem,
            memoryDim=ModuleDim.MemoryItem,
            outputDim=ModuleDim.MemoryFeat,
            hebbAlpha=(0.15 if plasticHebbian else 0.0), 
            useHebbian=plasticHebbian,
            emotionDim=ModuleDim.ValueEstimationOutEmotion)
        
        self.actor = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat, 
            intentDim=ModuleDim.IntentionFeat,
            includeNoSkill=True, 
            useHebb=plasticHebbian)
        
        self.world = RSSMWorldModel(
            visionDim=ModuleDim.AttentionFeat, 
            deterDim=ModuleDim.WorldOutHState,
            stochDim=ModuleDim.WorldOutZState,
            stateDim=ModuleDim.WorldFeat,
            ssmDim=ModuleDim.WorldOutXState,
            useMemory=True,
            globalFeatDim=ModuleDim.PerceptionFeat,
            objectTokenDim=ModuleDim.PerceptionEmbed,
            numObjectTokens=16,
            motionPredDim=ModuleDim.PerceptionEmbed,
            legacyFeatDim=ModuleDim.PerceptionFeat)

        self.critic = ValueEstimationExtractor(
            memoryDim=ModuleDim.MemoryFeat, 
            attnDim=ModuleDim.AttentionFeat, 
            stateDim=ModuleDim.WorldFeat,
            emotionDim=ModuleDim.ValueEstimationOutEmotion,
            useHebb=plasticHebbian)
        
        self.conscious = ConsciousnessExtractor(
            memItemDim=ModuleDim.MemoryItem,
            worldItemDim=ModuleDim.WorldMemoryItem,
            intentDim=ModuleDim.ConsciousnessState,
            useHebb=plasticHebbian)

        self.intention = IntentionExtractor(
            dimSem=ModuleDim.IntentionFeat,
            consSelfDim=int(self.conscious.self_dim),
            consIntentDim=int(self.conscious.intent_dim))

        self.OCR = OCREngineExtractor()

        self.history_len = int(BasicParameters.MEMORY_CALLBACK_LEN)

        if plasticOnlineLearning:
            self.perc = PerceptionOnlineWrapper(self.perc)
            self.attn = AttentionOnlineWrapper(self.attn)
            self.actor = DecisionOnlineWrapper(self.actor)
            self.world = WorldOnlineWrapper(self.world)
            self.critic =ValueEstimationOnlineWrapper(self.critic)
            self.intention = IntentionOnlineWrapper(self.intention)

        self.use_planner = usePlanner

        self.planner = None
        if self.use_planner:
            planner_keyboard_layout = {"default": RAW_KEYBOARD_LAYOUT}
            self.planner = DecisionPlannerExtractor().BuildPlanner(
                worldModel=self.world,
                wmIsOnlineWrapper=plasticOnlineLearning,
                KEYBOARD_LAYOUT=planner_keyboard_layout,
                horizon=5, N=64, elite=8, iters=3,
                gamma=0.99, temperature=1.0, momentum=0.15,
                minVar=1e-4, epsBern=1e-4)

        self.max_code = int(max(RAW_KEYBOARD_LAYOUT.values()))
        self.buf_B = 0

        self.extra_mem = None
        self.thread_end = True
        self.ex_thread: Optional[threading.Thread] = None

        self.mem_copy = copy.deepcopy(self.mem)
        self.attn_copy = copy.deepcopy(self.attn)
        self.critic_copy = copy.deepcopy(self.critic)
        self.moduleMessager = ModuleMessagerManager(maxSteps=256)
        self.save_module_messager_output = bool(saveModuleMessagerOutput)

        self.ResetBuffers(B=1, isOnlineLearning=self.is_online_learning,device=self.device)

    def SetModuleMessagerEnabled(self, enabled: bool):
        self.save_module_messager_output = bool(enabled)

    def RuntimeModule(self, mod: nn.Module) -> nn.Module:
        return mod.base if hasattr(mod, "base") else mod

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
            LegacyFeat=d(state.LegacyFeat),
            GlobalFeat=d(state.GlobalFeat),
            VentralFeat=d(state.VentralFeat),
            DorsalFeat=d(state.DorsalFeat),
            MotionToken=d(state.MotionToken),
            QualityToken=d(state.QualityToken),
            PredErrorToken=d(state.PredErrorToken),
            ObjectTokens=d(state.ObjectTokens),
            PatchTokens=d(state.PatchTokens),
            NextState={k: d(v) for k, v in state.NextState.items() if isinstance(v, torch.Tensor)},)

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

    def BuildTopDownContext(self) -> TopDownContext:
        return TopDownContext(
            GoalBias=self.prev_goal_bias,
            PredictedVisual=self.prev_predicted_visual,
            Precision=self.prev_precision,
            SelfSemantic=self.prev_self_sem,
            IntentSemantic=self.prev_intent_sem,
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

        legacy = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionFeat, device=device, dtype=dtype)
        object_seq = torch.zeros(batchSize, self.SEQ_LEN, 16, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        motion_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        quality_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        pred_seq = torch.zeros(batchSize, self.SEQ_LEN, ModuleDim.PerceptionEmbed, device=device, dtype=dtype)
        key_padding_mask = torch.ones(batchSize, self.SEQ_LEN, device=device, dtype=torch.bool)

        for idx, vs in enumerate(recent, start=start):
            legacy[:, idx] = vs.LegacyFeat.to(device=device, dtype=dtype)
            obj = vs.ObjectTokens.to(device=device, dtype=dtype)
            k = min(object_seq.size(2), obj.size(1))
            object_seq[:, idx, :k] = obj[:, :k]
            motion_seq[:, idx] = vs.MotionToken.to(device=device, dtype=dtype)
            quality_seq[:, idx] = vs.QualityToken.to(device=device, dtype=dtype)
            pred_seq[:, idx] = vs.PredErrorToken.to(device=device, dtype=dtype)
            key_padding_mask[:, idx] = False

        return legacy.contiguous(), object_seq.contiguous(), motion_seq.contiguous(), quality_seq.contiguous(), pred_seq.contiguous(), key_padding_mask

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

        self.prev_keys = z(self.max_code + 1)
        self.prev_clicks = z(2)
        self.prev_mouse = z(2)

        if isOnlineLearning:
            self.prev_option_logit = z(self.actor.base.num_options)
        else:
            self.prev_option_logit = z(self.actor.num_options)

        self.prev_entropy = z() 

        self.prev_visual_state = None
        self.prev_predicted_visual = None
        self.prev_precision = torch.ones(B, device=device, dtype=torch.float32)
        self.prev_goal_bias = z(ModuleDim.IntentionFeat)
        self.prev_self_sem = None
        self.prev_intent_sem = z(ModuleDim.IntentionFeat)

        self.buf_B = B

        self.perc_buffer = []
        self.visual_state_buffer = []

        self.history = deque(maxlen=self.history_len)


    def Step(
        self,
        frame: torch.Tensor,  # [B, C, H, W]
        textExt: Optional[List[Optional[str]]] = None,
        rewardExt: Optional[torch.Tensor] = None, # [B, 1]
        doneFlag: Optional[torch.Tensor] = None, # [B, 1]
        *,
        isTrain: bool = False,
        sampleActions: bool = True,
        deterministicActor: bool = False,) -> Dict[str, Any]:

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
        
        if self.extra_mem and self.thread_end:
            self.mem.MergeMemoryState(self.extra_mem)
            self.extra_mem =None


        def init_shadow_module_parms():
            mem_state = self.mem.ExportState()
            self.mem_copy.ImportState(mem_state)
            attn_state = self.attn.ExportState()
            self.attn_copy.ImportState(attn_state)
            critic_state = self.critic.ExportState()
            self.critic_copy.ImportState(critic_state)   

        if not isTrain and rewardExt is not None and self.history and self.thread_end:
            self.thread_end = False

            init_shadow_module_parms()

            historyRef_copy = copy.deepcopy(self.history)
            
            ex_thread = threading.Thread(
                target=self.SmoothWork,
                args=(historyRef_copy, rewardExt, "Reward", self.attn_copy, self.mem_copy, self.critic_copy),
                daemon=True)
            ex_thread.start()
        
        if not isTrain and doneFlag is not None and self.history and self.thread_end:
            self.thread_end = False

            init_shadow_module_parms()               

            historyRef_copy = copy.deepcopy(self.history)
            
            ex_thread = threading.Thread(
                target=self.SmoothWork,
                args=(historyRef_copy, doneFlag, "Done", self.attn_copy, self.mem_copy, self.critic_copy),
                daemon=True)
            ex_thread.start()

        B, C, H, W = frame.shape

        if isTrain:
            prev_world_h_for_prediction = self.prev_world_h.detach()
            prev_world_z_for_prediction = self.prev_world_z.detach()
            prev_world_x_for_prediction = self.prev_world_x.detach()
            prev_done_for_prediction = self.prev_done_flag.detach().clone()

        top_down = self.BuildTopDownContext()
        prev_visual_for_loss = self.prev_visual_state
        visual_state = self.perc(
            frame,
            prevVisualState=prev_visual_for_loss,
            topDownContext=top_down)
        perc_feats = visual_state.LegacyFeat # [B, D_perc]
        ocr_items = self.OCR(frame)
        fuse_ocr = self.OCR.ExportFusedTexts() # List[List[str]]
        ocr_semantic = self.EncodeOcrSemantic(fuse_ocr, batchSize=B, device=dev)

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
        self.perc_buffer = [v.LegacyFeat for v in self.visual_state_buffer if v is not None]
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
                "objects_mean": visual_state.ObjectTokens.mean(dim=1),},
            "key_padding_mask": key_padding_mask,})

        world_vis_in = self.attn(
            percs_seq,
            keyPaddingMask=key_padding_mask,
            objectSeq=object_seq,
            motionSeq=motion_seq,
            qualitySeq=quality_seq,
            predErrorSeq=pred_error_seq,
            goalBias=top_down.GoalBias,
            precision=top_down.Precision) # [B, D_attn]
        
        if isTrain:
            if self.is_online_learning:
                wm_kwargs = {"keysVec": self.prev_keys, "mouseClick": self.prev_clicks, "mouseSeq": self.prev_mouse,
                             "reward": rewardExt, "done": doneFlag}
                
                w_out = self.world(world_vis_in, **wm_kwargs)
            else: 
                w_out = self.world.ForwardTrain(visionIn=world_vis_in, keysVec=self.prev_keys,
                                                mouseClick=self.prev_clicks, mouseSeq=self.prev_mouse,
                                                reward=rewardExt, done=doneFlag)
        else:
            a_enc_prev = self.world.action_encoder(self.prev_keys, self.prev_mouse, self.prev_clicks) # [B, D_act]
            w_out = self.world.StepPosterior(visionIn=world_vis_in, actionEnc=a_enc_prev, sample=False)
        saveModuleOutput("World", w_out)

        s_t = w_out["s_next"] # [B, D_world]
        r_t = w_out["r_pred"].detach() # [B]
        d_t = w_out["d_prob"].detach() # [B]
        d_tr = w_out["d_tr"] # Optional[[B, D_world]]
        d_ph = w_out["d_ph"] # Optional[[B, D_world]]

        self.prev_world_h = w_out["h_next"].detach() # [B, D_world_h]
        self.prev_world_z = w_out["z_next"].detach() # [B, D_world_z]

        if isTrain:
            _, _, x_next = self.world.ExportState()
            self.prev_world_x = x_next.detach() # [B, D_world_x]
        else:
            self.prev_world_x = w_out["x_next"].detach() # [B, D_world_x]

        next_visual_prediction = w_out["reconstructed_visual_state"]

        if doneFlag is not None:
            done_now = doneFlag.detach().view(B) > 0.5
        else:
            done_now = d_t.detach().view(B) > 0.5

        if B == 1 and bool(done_now.item()):
            self.prev_predicted_visual = None
        else:
            self.prev_predicted_visual = self.DetachRuntimeObject(next_visual_prediction)

        self.prev_world_s = s_t.detach()
        self.prev_done_flag = done_now.detach()

        if self.is_online_learning:
            value_kwargs = {
                "rewardExt": r_t,
                "policyEntropyPrev": self.prev_entropy,
                "done": d_t,
                "worldDeltaTransport": d_tr,
                "worldDeltaPhysics": d_ph}
            value_x = {"memory": self.prev_mem,"attn": self.prev_attn, "state": s_t} # memory:[B, D_mem], attn:[B, D_attn], state:[B, D_world]
            critic_out = self.critic(x=value_x, **value_kwargs)
        else:
            critic_out = self.critic(memory=self.prev_mem,attn=self.prev_attn,state=s_t,rewardExt=r_t,
                                     policyEntropyPrev=self.prev_entropy,done=d_t,
                                     worldDeltaTransport=d_tr,worldDeltaPhysics=d_ph,)
        saveModuleOutput("ValueEstimation", critic_out)

        td_sig = critic_out.tdError.detach() # [B]
        unc_sig = critic_out.uncertainty.detach() # [B]
        precision_sig = getattr(critic_out, "precision", (1.0 - unc_sig).clamp(0.05, 1.0)).detach() # [B]
        emotion_sig = critic_out.emotion.detach() # [B, D_emotion]
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
            goalBias=top_down.GoalBias,
            precision=precision_sig) # [B, D_attn]
        saveModuleOutput("Attention", atten_out)

        intent_hint_for_memory = self.prev_intent_sem
        mem_feat = self.mem(
            atten_out,
            tdError=td_sig,
            emotion=emotion_sig,
            reward=r_t,
            visualState=visual_state,
            ocrSemantic=ocr_semantic,
            intentHint=intent_hint_for_memory) # [B, D_mem], [B, D_mem]
        saveModuleOutput("Memory", mem_feat)

        memory_bank = self.mem.ExportMemoryBank(topk = BasicParameters.CONSCIOUSNESSTEM) # Optional[Dict[str, Tensor]]
        world_bank = self.world.ExportWorldMemoryBank(topk = BasicParameters.CONSCIOUSNESSTEM) # Optional[Dict[str, Tensor]]

        conscious_out = self.conscious(memoryBank=memory_bank, worldBank=world_bank) # self_sem/intention_sem: [B, D_cons]
        saveModuleOutput("Consciousness", {
            "self_sem": conscious_out.self_sem,
            "intent_sem": conscious_out.intent_sem,
            "extras": conscious_out.extras,})

        saveModuleOutput("OCR", {
            "items": ocr_items,
            "texts": fuse_ocr,})

        intent_sem, sym_probs, intention_extras = self.intention(
            conscious_out.self_sem,
            conscious_out.intent_sem,
            ocrTexts=fuse_ocr,
            extTexts=textExt,
            prioritizeExt=self.prioritize_ext_str,) # [B, D_intent], [B, K_sym], Dict[str, Tensor]
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

        if self.is_online_learning:
            actor_kwargs = {"sample": sampleActions, "deterministic": deterministicActor, "prevOptionLogit": 
                            self.prev_option_logit, "intentFeat":intent_sem}
            act_out = self.actor(x=mem_feat,**actor_kwargs)
        else:    
            act_out = self.actor(stateFeat=mem_feat,intentFeat=intent_sem,sample=sampleActions,
                                deterministic=deterministicActor,prevOptionLogit=self.prev_option_logit)

        if self.use_planner:
            prior = None
            with torch.no_grad():
                mouseMu = act_out["mouse"]["mu"].detach() # [B, 2]
                mouseLogstd = act_out["mouse"]["logstd"].detach() # [B, 2]
                keysLogits = act_out["keyboard"]["keys_logits"].detach() # [B, K_key]
                clickLogits = act_out["mouse"]["click_logits"].detach() # [B, 2]

                prior = self.planner.Plan(keysLogits=keysLogits, mouseMu=mouseMu, mouseLogstd=mouseLogstd,
                                          clickLogits=clickLogits,
                                          h0=self.prev_world_h,z0=self.prev_world_z,x0=self.prev_world_x)
        
            if self.is_online_learning:
                actor_kwargs = {"sample": sampleActions, "deterministic": deterministicActor, "prevOptionLogit": 
                                    self.prev_option_logit, "intentFeat":intent_sem, "prior": prior}
                act_out = self.actor(x=mem_feat,**actor_kwargs)
            else:
                act_out = self.actor(stateFeat=mem_feat,intentFeat=intent_sem,sample=sampleActions,
                                    deterministic=deterministicActor,prevOptionLogit=self.prev_option_logit,prior=prior)
        saveModuleOutput("Decision", act_out)

        keys_act = act_out["keyboard"]["keys_act"] # [B, K_key]
        click_sample = act_out["mouse"]["click_sample"] # [B, 2]
        mouse_a = act_out["mouse"]["a"] # [B,2]
        entropy_actor = act_out["entropy"] # [B]
        next_option_logit = act_out["prevOptionLogit_next"].detach() # [B, K_option]

        self.prev_option_logit = next_option_logit # [B, K_option]

        self.prev_mem = mem_feat.detach() # [B, D_mem]
        self.prev_attn = atten_out.detach() # [B, D_attn]
        self.prev_keys = keys_act.detach() # [B, K_key]
        self.prev_clicks = click_sample.detach() # [B, 2]
        self.prev_mouse = mouse_a.detach() # [B, 2]

        self.prev_entropy = entropy_actor.detach() # [B]

        if not isTrain:
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
                Keys=trace_tensor(keys_act), # [B, K_key]
                MouseClick=trace_tensor(click_sample), # [B, 2]
                MouseDelta=trace_tensor(mouse_a), # [B, 2]

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

            conscious_loss = world_loss.new_zeros(())
            if conscious_out.extras is not None:
                conscious_loss = conscious_out.extras.get("loss", conscious_loss)

            intention_loss, _ = self.intention.GetInternalLoss(sym_probs)

            perception_loss = self.perc.ComputePerceptionLoss(
                visual_state,
                prevVisualState=prev_visual_for_loss)

            world_prediction_loss = world_loss.new_zeros(())
            world_prediction_losses: Dict[str, torch.Tensor] = {}
            if (B == 1
                and not bool(prev_done_for_prediction.any().item())):
                pred_train = self.world.PredictNextVisualFromPosterior(
                    prev_world_h_for_prediction,
                    prev_world_z_for_prediction,
                    prev_world_x_for_prediction,
                    actionEnc=None,
                    sample=False,)
                world_prediction_losses = self.world.ComputePredictionLoss(
                    predictedVisual=pred_train["predicted_visual"],
                    reconstructedVisualState=pred_train["reconstructed_visual_state"],
                    targetVisualState=visual_state,
                    precision=precision_sig,)
                world_prediction_loss = world_prediction_losses.get("loss_pred_total", world_prediction_loss)

            total_loss = (
                world_loss
                + mem_loss
                + critic_loss
                + conscious_loss
                + intention_loss
                + 0.05 * perception_loss
                + 0.05 * world_prediction_loss)
            
            losses["world_loss"] = world_loss
            losses["memory_loss"] = mem_loss
            losses["critic_loss"] = critic_loss
            losses["conscious_loss"] = conscious_loss
            losses["intention_loss"] = intention_loss
            losses["perception_loss"] = perception_loss
            losses["world_prediction_loss"] = world_prediction_loss
            for name, value in world_prediction_losses.items():
                losses[f"world_{name}"] = value
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
            "prev_keys": self.prev_keys.detach().clone(),
            "prev_clicks": self.prev_clicks.detach().clone(),
            "prev_mouse": self.prev_mouse.detach().clone(),
            "prev_option_logit": self.prev_option_logit.detach().clone(),
            "prev_entropy": self.prev_entropy.detach().clone(),
            "world_state": {
                "h": h.detach().clone(),
                "z": z.detach().clone(),
                "x": x.detach().clone(),
                "s": self.prev_world_s.detach().clone(),
                "done": self.prev_done_flag.detach().clone(),
                "A_prev": None if A_prev is None else A_prev.detach().clone(),},
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
            "extra_mem": None if self.extra_mem is None else copy.deepcopy(self.extra_mem),}

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
        self.prev_keys = state["prev_keys"]
        self.prev_clicks = state["prev_clicks"]
        self.prev_mouse = state["prev_mouse"]
        self.prev_option_logit = state["prev_option_logit"]
        prev_entropy = state["prev_entropy"]
        if isinstance(prev_entropy, torch.Tensor) and (prev_entropy.dim() > 1) and (prev_entropy.size(-1) == 1):
            prev_entropy = prev_entropy.squeeze(-1)
        self.prev_entropy = prev_entropy

        world_state = state["world_state"]
        world_mod.ImportState(world_state["h"], world_state["z"], world_state["x"])
        world_mod._A_prev = None if world_state["A_prev"] is None else world_state["A_prev"].detach().clone()
        self.prev_world_h = world_state["h"].detach().clone()
        self.prev_world_z = world_state["z"].detach().clone()
        self.prev_world_x = world_state["x"].detach().clone()
        self.prev_world_s = world_state.get(
            "s",
            torch.zeros(self.prev_mem.size(0), ModuleDim.WorldFeat, device=device, dtype=self.prev_mem.dtype))
        self.prev_done_flag = world_state.get(
            "done",
            torch.ones(self.prev_mem.size(0), device=device, dtype=torch.bool))

        self.mem.ImportState(state["mem_state"], importGws=True, importLtm=True, importSym=True)
        self.mem.pending = state["mem_pending"]
        attn_mod.ImportState(state["attn_state"])
        critic_mod.ImportState(state["critic_state"])

        self.perc_buffer = state["perc_buffer"]
        self.prev_visual_state = state.get("prev_visual_state", None)
        self.prev_predicted_visual = state.get("prev_predicted_visual", None)
        self.prev_precision = state.get("prev_precision", torch.ones(self.prev_mem.size(0), device=device, dtype=self.prev_mem.dtype))
        self.prev_goal_bias = state.get("prev_goal_bias", torch.zeros(self.prev_mem.size(0), ModuleDim.IntentionFeat, device=device, dtype=self.prev_mem.dtype))
        self.prev_self_sem = state.get("prev_self_sem", None)
        self.prev_intent_sem = state.get("prev_intent_sem", torch.zeros(self.prev_mem.size(0), ModuleDim.IntentionFeat, device=device, dtype=self.prev_mem.dtype))
        self.visual_state_buffer = state.get("visual_state_buffer", [])
        if not self.perc_buffer and self.visual_state_buffer:
            self.perc_buffer = [v.LegacyFeat for v in self.visual_state_buffer if v is not None]

        ocr_state = state["ocr_state"]
        self.OCR._temporal_step = int(ocr_state["temporal_step"])
        self.OCR._last_batch_size = int(ocr_state["last_batch_size"])
        self.OCR._last_ocr_texts_batch = ocr_state["last_ocr_texts_batch"]
        self.OCR._tracks_by_bi = ocr_state["tracks_by_bi"]

        self.history = deque(state["history"], maxlen=self.history_len)
        self.extra_mem = state["extra_mem"]
        self.thread_end = True
        self.ex_thread = None
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
        ext_vec = extLast.to(device=device, dtype=dtype)

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

    def SmoothWork(self, historyRef, lastRef, signal: str, attenModule: torch.Module, memModule: torch.Module, criticModule: torch.Module):
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

            if signal == "Reward":
                seq_list = reward_list
            elif signal == "Done":
                seq_list = done_list
            else:
                return

            if len(seq_list) <= 1:
                return

            wm_seq = torch.stack(seq_list, dim=1).contiguous()  # [B, T]

            smoothed = self.SmoothCorrection(wmSeq=wm_seq, extLast=lastRef)

            smoothed_list = list(smoothed.unbind(dim=1))

            start = int(memModule.time_step.min().item())

            with torch.no_grad():
                for i in range(1, len(smoothed_list)):
                    if signal == "Reward":
                        reward_in = smoothed_list[i]
                        done_in = done_list[i]
                    else: # Done
                        reward_in = reward_list[i]
                        done_in = smoothed_list[i]

                    value = criticModule(
                        memory=mem_list[i-1],
                        attn=atten_list[i-1],
                        state=world_state_list[i],
                        rewardExt=reward_in,
                        policyEntropyPrev=entropy_list[i-1],
                        done=done_in,
                        worldDeltaTransport=world_dtr_list[i],
                        worldDeltaPhysics=world_dph_list[i],)

                    td_sig = value.tdError.detach()
                    unc_sig = value.uncertainty.detach()
                    precision_sig = getattr(value, "precision", (1.0 - unc_sig).clamp(0.05, 1.0)).detach()
                    emotion_sig = value.emotion.detach()

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
                        precision=precision_sig)
                    memModule(
                        atten_out,
                        tdError=td_sig,
                        emotion=emotion_sig,
                        reward=reward_in,
                        visualState=visual_state_list[i],
                        ocrSemantic=ocr_semantic_list[i],
                        intentHint=intent_hint_list[i],
                        sourceLabel=MemoryType.SRC_IMAGINE)

            self.extra_mem = memModule.ExportState(step=start)

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
                self.brain.intention)
        
            self.opt_actor = torch.optim.Adam(actor_params, lr=3e-4)

            self.opt_critic = torch.optim.Adam(self.brain.critic.parameters(), lr=2e-4)

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

        self.brain.ResizeStateBuffersForLoad(brain_state)
        self.brain.load_state_dict(brain_state, strict=False)

    def ExportModuleMessagerData(self, nSteps: int = 0):
        return self.brain.moduleMessager.ExportDict(nSteps=nSteps)

    def Act(
        self,
        frame: torch.Tensor, # [B,C,H,W]
        *,
        textExt: Optional[List[Optional[str]]] = None,
        reward: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        sampleActions: bool = True,
        deterministicActor: bool = False,):

        frame = frame.to(self.device)
        if reward is not None:
            reward = reward.to(self.device)
        if done is not None:
            done = done.to(self.device)

        def build_action_output(decision_out: Dict[str, Any]) -> Dict[str, Any]:
            mouse_head = decision_out["mouse"]
            kb_head = decision_out["keyboard"]

            if sampleActions:
                keys = kb_head["keys_act"] # [B, K_key]
                clicks = mouse_head["click_sample"] # [B, 2]
                mouse = mouse_head["a"] # [B, 2]
            else:
                keys = (torch.sigmoid(kb_head["keys_logits"]) > 0.5).float() # [B, K_key]
                clicks = (torch.sigmoid(mouse_head["click_logits"]) > 0.5).float() # [B, 2]
                mouse = mouse_head["mu"] # [B, 2]

            return {
                "keys": keys,
                "mouse_clicks": clicks,
                "mouse_move": mouse,}

        if self.is_train:
            step_out = self.brain.Step(frame,textExt,rewardExt=reward,doneFlag=done,isTrain=self.is_train,sampleActions=sampleActions,deterministicActor=deterministicActor,)
            if step_out is None: return None
            act_out = build_action_output(step_out["decision"])
            act_out["loss"] = step_out["losses"]["total_loss"]
            act_out["OCR"] = step_out["OCR"]
            act_out["intention_texts"] = step_out.get("intention_texts", [])
            return act_out
        else:  
            with torch.no_grad():  
                step_out = self.brain.Step(frame,textExt,rewardExt=reward,doneFlag=done,isTrain=self.is_train,sampleActions=sampleActions,deterministicActor=deterministicActor,)
                if step_out is None: return None
                act_out = build_action_output(step_out["decision"])
                act_out["loss"] = None
                act_out["OCR"] = step_out["OCR"]
                act_out["intention_texts"] = step_out.get("intention_texts", [])
                return act_out


    def UnpackActPacked(self, actOut: Optional[Dict[str, Any]]) -> str:
        key_names: List[str] = []
        mouse_clicks: List[str] = []
        mouse_move: Optional[Dict[str, float]] = None
        intention_texts: List[str] = []

        if actOut is None:
            return json.dumps({
                "key_names": key_names,
                "mouse_clicks": mouse_clicks,
                "mouse_move": mouse_move,
                "intention_texts": intention_texts,}, ensure_ascii=False)

        keys_tensor = actOut.get("keys")
        clicks_tensor = actOut.get("mouse_clicks")
        mouse_tensor = actOut.get("mouse_move")
        intention_texts_value = actOut.get("intention_texts")

        keys_tensor = keys_tensor.detach().float().cpu() if keys_tensor is not None else None
        clicks_tensor = clicks_tensor.detach().float().cpu() if clicks_tensor is not None else None
        mouse_tensor = mouse_tensor.detach().float().cpu() if mouse_tensor is not None else None

        if isinstance(intention_texts_value, list):
            intention_texts = [str(item) for item in intention_texts_value]
        elif intention_texts_value is not None:
            intention_texts = [str(intention_texts_value)]

        index_to_key = {int(key_index): str(key_name) for key_name, key_index in RAW_KEYBOARD_LAYOUT.items()}

        if keys_tensor is not None:
            active_indices = (keys_tensor[0] > 0.5).nonzero(as_tuple=False).view(-1).tolist()
            key_names = [index_to_key[int(idx)] for idx in active_indices if int(idx) in index_to_key]

        if clicks_tensor is not None:
            if float(clicks_tensor[0, 0].item()) > 0.5:
                mouse_clicks.append("left")
            if float(clicks_tensor[0, 1].item()) > 0.5:
                mouse_clicks.append("right")

        if mouse_tensor is not None:
            mouse_move = {
                "x": float(mouse_tensor[0, 0].item()),
                "y": float(mouse_tensor[0, 1].item()),}

        return json.dumps({
            "key_names": key_names,
            "mouse_clicks": mouse_clicks,
            "mouse_move": mouse_move,
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
            self.brain.ResizeStateBuffersForLoad(payload["brain"])
            self.brain.load_state_dict(payload["brain"], strict=strict)

            if self.is_train:
                if "opt_actor" in payload:
                    self.opt_actor.load_state_dict(payload["opt_actor"])
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
            self.brain.ResizeStateBuffersForLoad(payload)
            self.brain.load_state_dict(payload, strict=strict)

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
        self.brain.ex_thread = None

        self.brain.ResetBuffers(B=B, isOnlineLearning=isOnlineLearning, device=self.device)

    def ResetHebbianMemory(self):
        if self.brain.is_online_learning:
            self.brain.perc.base.ResetHebbianMemory()
            self.brain.attn.base.ResetHebbianMemory()
            self.brain.actor.base.ResetHebbianMemory()
            self.brain.critic.base.ResetHebbianMemory()
        else:
            self.brain.perc.ResetHebbianMemory()
            self.brain.attn.ResetHebbianMemory()
            self.brain.actor.ResetHebbianMemory()
            self.brain.critic.ResetHebbianMemory()

        self.brain.mem.ResetHebbianMemory()
        self.brain.conscious.ResetHebbianMemory()





    def UpdateWrappers(self, wrappers, action: str, **kwargs):
        results = []
        for w in wrappers:
            out = w.Update(action, **kwargs)
            results.append(out)
        return results
    
    def UpdateAllWrappers(self, action: str, **kwargs):
        wrappers = [self.brain.perc, self.brain.attn, self.brain.actor, self.brain.world, self.brain.critic, self.brain.intention]
        results = []
        for w in wrappers:
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
