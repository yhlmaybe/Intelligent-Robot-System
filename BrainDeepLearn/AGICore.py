from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
import threading
import random
import ast

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

from PerceptionModule import PerceiveExtractor, PerceptionOnlineWrapper
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
            useMemory=True)

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

        self.prev_keys = z(self.max_code + 1)
        self.prev_clicks = z(2)
        self.prev_mouse = z(2)

        if isOnlineLearning:
            self.prev_option_logit = z(self.actor.base.num_options)
        else:
            self.prev_option_logit = z(self.actor.num_options)

        self.prev_entropy = z() 

        self.buf_B = B

        self.perc_buffer = []

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

        perc_feats = self.perc(frame) # [B, D_perc]
        ocr_items = self.OCR(frame)

        self.perc_buffer.append(perc_feats)
        saveModuleOutput("Perception", {
            "feat": perc_feats,})

        if len(self.perc_buffer) > self.SEQ_LEN:
            del self.perc_buffer[:BasicParameters.IMAGE_RM_LEN]
        elif len(self.perc_buffer) < self.SEQ_LEN:
            return None 

        percs_seq = torch.stack(self.perc_buffer, dim=1).contiguous() # [B, T, D_perc]

        with torch.no_grad():
            world_vis_in = self.attn(percs_seq) # [B, D_attn]
        
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
        emotion_sig = critic_out.emotion.detach() # [B, D_emotion]

        atten_out = self.attn(percs_seq, tdError=td_sig, uncertainty=unc_sig) # [B, D_attn]
        saveModuleOutput("Attention", atten_out)

        mem_feat = self.mem(atten_out, tdError=td_sig,emotion=emotion_sig,reward=r_t) # [B, D_mem], [B, D_mem]
        saveModuleOutput("Memory", mem_feat)

        memory_bank = self.mem.ExportMemoryBank(topk = BasicParameters.CONSCIOUSNESSTEM) # Optional[Dict[str, Tensor]]
        world_bank = self.world.ExportWorldMemoryBank(topk = BasicParameters.CONSCIOUSNESSTEM) # Optional[Dict[str, Tensor]]

        conscious_out = self.conscious(memoryBank=memory_bank, worldBank=world_bank) # self_sem/intention_sem: [B, D_cons]
        saveModuleOutput("Consciousness", {
            "self_sem": conscious_out.self_sem,
            "intent_sem": conscious_out.intent_sem,
            "extras": conscious_out.extras,})

        fuse_ocr = self.OCR.ExportFusedTexts() # List[List[str]]
        saveModuleOutput("OCR", {
            "items": ocr_items,
            "texts": fuse_ocr,})

        intent_sem, sym_probs, intention_extras = self.intention(
            conscious_out.self_sem,
            conscious_out.intent_sem,
            ocrTexts=fuse_ocr,
            extTexts=textExt,
            prioritizeExt=self.prioritize_ext_str,) # [B, D_intent], [B, K_sym], Dict[str, Tensor]
        saveModuleOutput("Intention", {
            "intent_sem": intent_sem,
            "sym_probs": sym_probs,
            "extras": intention_extras,
            "ocr_texts": fuse_ocr,
            "ext_texts": textExt,})

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
            trace = BrainStepTrace(
                PercBuffer=copy.deepcopy(self.perc_buffer), # List[[B, D_perc]]
                ObsImg=fuse_ocr, # List[List[str]]
                Keys=keys_act, # [B, K_key]
                MouseClick=click_sample, # [B, 2]
                MouseDelta=mouse_a, # [B, 2]

                PercFeat=perc_feats, # [B, D_perc]
                AttnFeat=atten_out, # [B, D_attn]
                MemFeat=mem_feat, # [B, D_mem]
                WorldState=s_t, # [B, D_world]
                WorldDeltaTransport=d_tr, # [B, D_world]
                WorldDeltaPhysics=d_ph, # [B, D_world]
                ConsciousnessState=conscious_out.intent_sem, # [B, D_cons]
                IntentionState=intent_sem, # [B, D_intent]
                Reward=r_t, # [B]
                Done=d_t, # [B]
                ActionEntropy=entropy_actor, # [B]

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

            total_loss = world_loss + mem_loss + critic_loss + conscious_loss + intention_loss
            
            losses["world_loss"] = world_loss
            losses["memory_loss"] = mem_loss
            losses["critic_loss"] = critic_loss
            losses["conscious_loss"] = conscious_loss
            losses["intention_loss"] = intention_loss
            losses["total_loss"] = total_loss
            saveModuleOutput("Losses", losses)

        return {
            "decision": act_out,
            "world": {"state": s_t, "reward": r_t, "done": d_t}, # state:[B, D_world], reward/done:[B]
            "critic": critic_out,
            "features": {"perc": percs_seq, "attn": atten_out, "mem": mem_feat}, # perc:[B, T, D_perc], attn:[B, D_attn], mem:[B, D_mem]
            "OCR": ocr_items,
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
                "A_prev": None if A_prev is None else A_prev.detach().clone(),},
            "mem_state": self.mem.ExportState(),
            "mem_pending": copy.deepcopy(self.mem.pending),
            "attn_state": attn_mod.ExportState(),
            "critic_state": critic_mod.ExportState(),
            "perc_buffer": [t.detach().clone() for t in self.perc_buffer],
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

        self.mem.ImportState(state["mem_state"], importGws=True, importLtm=True, importSym=True)
        self.mem.pending = state["mem_pending"]
        attn_mod.ImportState(state["attn_state"])
        critic_mod.ImportState(state["critic_state"])

        self.perc_buffer = state["perc_buffer"]

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
            per_buffer_list = []
            atten_list = []
            mem_list = []
            world_state_list = []
            world_dtr_list = []
            world_dph_list = []
            reward_list = []
            done_list = []
            entropy_list = []

            for tr in historyRef:
                per_buffer_list.append(tr.PercBuffer)
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
                    emotion_sig = value.emotion.detach()

                    percs_seq = torch.stack(per_buffer_list[i], dim=1).contiguous()
                    atten_out = attenModule(percs_seq, tdError=td_sig, uncertainty=unc_sig)
                    memModule(atten_out, tdError=td_sig, emotion=emotion_sig, reward=reward_in, sourceLabel=MemoryType.SRC_IMAGINE)

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

        def pack_action(decision_out: Dict[str, Any]):
            mouse_head = decision_out["mouse"]
            kb_head = decision_out["keyboard"]

            if sampleActions:
                keys = kb_head["keys_act"] # [B, K_key]
                clicks = mouse_head["click_sample"] # [B, 2]
                mouse = mouse_head["a"] # [B, 2]
                return keys, clicks, mouse

            keys = (torch.sigmoid(kb_head["keys_logits"]) > 0.5).float() # [B, K_key]
            clicks = (torch.sigmoid(mouse_head["click_logits"]) > 0.5).float() # [B, 2]
            mouse = mouse_head["mu"] # [B, 2]
            return keys, clicks, mouse

        if self.is_train:
            step_out = self.brain.Step(frame,textExt,rewardExt=reward,doneFlag=done,isTrain=self.is_train,sampleActions=sampleActions,deterministicActor=deterministicActor,)
            if step_out is None: return None
            total_loss = step_out["losses"]["total_loss"]
            packed = pack_action(step_out["decision"])

            return (*packed, total_loss, step_out["OCR"])
        else:  
            with torch.no_grad():  
                step_out = self.brain.Step(frame,textExt,rewardExt=reward,doneFlag=done,isTrain=self.is_train,sampleActions=sampleActions,deterministicActor=deterministicActor,)
                if step_out is None: return None
                packed = pack_action(step_out["decision"])

                return (*packed, step_out["OCR"])


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
