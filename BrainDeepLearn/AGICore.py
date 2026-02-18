from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
import threading
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import os
import math
import copy

#import debugpy

from dataclasses import dataclass, field
from collections import deque

from PerceptionModule import PerceiveExtractor, PerceptionOnlineWrapper
from AttentionModule import AttentionExtractor, AttentionOnlineWrapper
from MemoryModule import MemoryExtractor
from DecisionModule import DecisionExtractor, DecisionOnlineWrapper, RAW_KEYBOARD_LAYOUT, DecisionPlannerExtractor, StableLogProbBernoulli
from WorldModule import RSSMWorldModel, WorldOnlineWrapper
from ValueEstimationModule import ValueEstimationExtractor,ValueEstimationOnlineWrapper
from ConsciousnessModule import ConsciousnessExtractor
from IntentionModule import IntentionExtractor, IntentionOnlineWrapper
from OCRModule import OCREngineExtractor
from ModuleDimensionManager import ModuleDim
 


def ToDevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

class BasicParameters:
    IMAGE_SIZE = 512
    IMAGE_SEQ_LEN = 16
    IMAGE_RM_LEN = math.ceil(IMAGE_SEQ_LEN * 1 / 10)

    MEMORY_CALLBACK_LEN = 16

    MEMORY_MEMORY_PATH = "BrainDeepLearn/Data/MemoryMemory.pt"
    WORLD_MEMORY_PATH = "BrainDeepLearn/Data/WorldMemory.pt"
    MODULEPARAMETER_PATH = "BrainDeepLearn/Data/module_parameter.pth"
    DATA_ROOT_PATH = "BrainDeepLearn/Data"
    CKPT_PATH_TRAIN = "BrainDeepLearn/Data/training_checkpoint.pth"

    MEMORY_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/MemoryMemory.pt"
    WORLD_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/WorldMemory.pt"
    DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData"
    CKPT_PATH_TEST = "BrainDeepLearn/TestData/training_test_checkpoint.pth"


@dataclass
class BrainStepTrace:
    ObsImg: Optional[torch.Tensor] = None
    KeyVec: Optional[torch.Tensor] = None
    MouseDelta: Optional[torch.Tensor] = None

    PercBuffer: Optional[list[torch.Tensor]] = None
    PercFeat: Optional[torch.Tensor] = None
    AttnFeat: Optional[torch.Tensor] = None
    MemFeat: Optional[torch.Tensor] = None
    WorldState: Optional[torch.Tensor] = None
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
        usePlanner: bool = True,):
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
            consDim=ModuleDim.ConsciousnessState)

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

        self.planner = DecisionPlannerExtractor().BuildPlanner(
            worldModel=self.world,
            wmIsOnlineWrapper=plasticOnlineLearning,
            RAW_KEYBOARD_LAYOUT=RAW_KEYBOARD_LAYOUT,
            includeNoSkill=True,
            horizon=5, N=64, elite=8, iters=3,
            gamma=0.99, temperature=1.0, momentum=0.15,
            laplace=1.0, minVar=1e-4, epsBern=1e-4)

        all_codes = []
        for grp in RAW_KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        self.max_code = max(all_codes)
        self.keyvec_dim = (self.max_code + 1) + 2  # 106

        self.buf_B = 0

        self.extra_mem = None
        self.thread_end = True
        self.ex_thread: Optional[threading.Thread] = None

        self.mem_copy = copy.deepcopy(self.mem)
        self.attn_copy = copy.deepcopy(self.attn)
        self.critic_copy = copy.deepcopy(self.critic)

        self.ResetBuffers(B=1, isOnlineLearning=self.is_online_learning,device=self.device)

    def InitShadowModule(self):
        self.mem_copy.load_state_dict(self.mem.state_dict(), strict=True)
        self.attn_copy.load_state_dict(self.attn.state_dict(), strict=True)
        self.critic_copy.load_state_dict(self.critic.state_dict(), strict=True)

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

        self.prev_key_vec = z(self.keyvec_dim) 
        self.prev_mouse = z(2)

        if isOnlineLearning:
            self.prev_option_onehot = z(self.actor.base.num_options)
        else:
            self.prev_option_onehot = z(self.actor.num_options)

        self.prev_entropy = z(1) 

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

        B, dev = frame.size[0], frame.device
        if (self.buf_B != B) or (self.prev_mem.device != dev):
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

        perc_feats = self.perc(frame) # [B,D]
        perc_ocr = self.OCR(frame) 

        self.perc_buffer.append(perc_feats)

        if len(self.perc_buffer) > self.SEQ_LEN:
            del self.perc_buffer[:BasicParameters.IMAGE_RM_LEN]
        elif len(self.perc_buffer) < self.SEQ_LEN:
            return None 

        percs_seq = torch.stack(self.perc_buffer, dim=1).contiguous()

        with torch.no_grad():
            world_vis_in = self.attn(percs_seq)
        
        if self.is_online_learning:
            a_enc_prev = self.world.base.action_encoder(self.prev_key_vec, self.prev_mouse)
        else:
            a_enc_prev = self.world.action_encoder(self.prev_key_vec, self.prev_mouse)

        if isTrain:
            if self.is_online_learning:
                wm_kwargs = {"keysVec": self.prev_key_vec,"mouseSeq": self.prev_mouse,"h0": self.prev_world_h,
                             "z0": self.prev_world_z,"rewardSeq": rewardExt,"doneSeq": doneFlag}
                
                w_out = self.world(world_vis_in, **wm_kwargs)
            else: 
                w_out = self.world.ForwardTrainSeq(visionSeq=world_vis_in,keysVec= self.prev_key_vec, 
                                                   mouseSeq=self.prev_mouse, h0=self.prev_world_h, z0=self.prev_world_z,
                                                   rewardSeq=rewardExt,doneSeq=doneFlag)
        else:
            w_out = self.world.StepPosterior(self.prev_world_h, self.prev_world_z, 
                                            visionIn=world_vis_in, actionEnc=a_enc_prev, sample=False)

        s_t = w_out["s_next"] # World State
        r_t = w_out["r_pred"].detach() # Prediction Rewards
        d_t = w_out["d_prob"].detach() # Termination Probability
        d_tr = w_out.get("d_tr", None)
        d_ph = w_out.get("d_ph", None)

        self.prev_world_h = w_out["h_next"].detach()
        self.prev_world_z = w_out["z_next"].detach()
        self.prev_world_x = w_out["x_next"].detach()

        if self.is_online_learning:
            value_kwargs = {
                "rewardExt": r_t,
                "policyEntropyPrev": self.prev_entropy,
                "done": d_t,
                "worldDeltaTransport": d_tr,
                "worldDeltaPhysics": d_ph}
            value_x = {"memory": self.prev_mem,"attn": self.prev_attn, "state": s_t}
            critic_out = self.critic(x=value_x, **value_kwargs)
        else:
            critic_out = self.critic(memory=self.prev_mem,attn=self.prev_attn,state=s_t,rewardExt=r_t,
                                     policyEntropyPrev=self.prev_entropy,done=d_t,
                                     worldDeltaTransport=d_tr,worldDeltaPhysics=d_ph,)

        td_sig = critic_out.tdError.detach()
        rInt_sig = critic_out.rInt.detach()
        unc_sig = critic_out.uncertainty.detach()
        emotion_sig = critic_out.emotion.detach()

        atten_out = self.attn(percs_seq, tdError=td_sig, uncertainty=unc_sig)

        mem_feat, mem_recall = self.mem(atten_out, tdError=td_sig,emotion=emotion_sig,reward=rInt_sig)

        memory_bank = self.mem.ExportMemoryBank(B, self.device)
        world_bank = self.world.ExportWorldMemoryBank(B, self.device)

        conscious_out = self.conscious(memoryBank=memory_bank, worldBank=world_bank)

        conscious_state = conscious_out.intent_sem

        fuse_ocr = self.OCR.ExportFusedTexts()

        intent_sem, sym_probs, intention_extras = self.intention(conscious_state, ocrTexts=fuse_ocr, 
                                                                 extTexts=textExt,prioritizeExt=self.prioritize_ext_str)

        with torch.no_grad():
            if self.is_online_learning:
                actor_kwargs = {"sample": sampleActions, "deterministic": deterministicActor, "prevOptionOnehot": 
                                self.prev_option_onehot, "intentFeat":intent_sem}
                act_out = self.actor(x=mem_feat,**actor_kwargs)
            else:    
                act_out = self.actor(stateFeat=mem_feat,intentFeat=intent_sem,sample=sampleActions,
                                     deterministic=deterministicActor,prevOptionOnehot=self.prev_option_onehot)

        mouseMu = act_out["mouse"]["mu"].detach()
        mouseLogstd = act_out["mouse"]["logstd"].detach() 
        skillLogits = act_out["keyboard"]["skill_logits"].detach()
        baseLogits = act_out["keyboard"]["base_logits"].detach()
        extraLogits = act_out["keyboard"]["extra_logits"].detach()
        clickLogits = act_out["mouse"]["click_logits"].detach()

        prior = None
        if self.use_planner:
            with torch.no_grad():
                prior = self.planner.Plan(mouseMu=mouseMu,mouseLogstd=mouseLogstd,skillLogits=skillLogits,
                                          baseLogits=baseLogits,extraLogits=extraLogits,clickLogits=clickLogits,
                                          h0=self.prev_world_h,z0=self.prev_world_z,x0=self.prev_world_x)
        
        if self.is_online_learning:
            actor_kwargs = {"sample": sampleActions, "deterministic": deterministicActor, "prevOptionOnehot": 
                                self.prev_option_onehot, "intentFeat":intent_sem, "prior": prior}
            act_out = self.actor(x=mem_feat,**actor_kwargs)
        else:
            act_out = self.actor(stateFeat=mem_feat,intentFeat=intent_sem,sample=sampleActions,
                                 deterministic=deterministicActor,prevOptionOnehot=self.prev_option_onehot,prior=prior)

        key_vec = act_out["key_vec"] # [B,106]
        mouse_a = act_out["mouse"]["a"] # [B,2]
        entropy_actor = act_out["entropy"]

        if "option" in act_out and ("opt_onehot" in act_out["option"]):
            next_opt = act_out["option"]["opt_onehot"].detach()
        else:
            logits = act_out["option"]["logits"].detach()
            idx = torch.argmax(logits, dim=-1)
            next_opt = F.one_hot(idx, num_classes=self.actor.num_options).float()

        self.prev_option_onehot = next_opt.detach()

        self.prev_mem = mem_feat.detach()
        self.prev_attn = atten_out.detach()
        self.prev_key_vec = key_vec.detach()
        self.prev_mouse = mouse_a.detach()

        self.prev_entropy = entropy_actor.detach()

        if not isTrain:
            trace = BrainStepTrace(
                PercBuffer=copy.deepcopy(self.perc_buffer),
                ObsImg=fuse_ocr,
                KeyVec=key_vec,
                MouseDelta=mouse_a,

                PercFeat=perc_feats,
                AttnFeat=atten_out,
                MemFeat=mem_feat,
                WorldState=s_t,
                ConsciousnessState=conscious_state,
                IntentionState=intent_sem,
                Reward=r_t,
                Done=d_t,
                ActionEntropy=entropy_actor,

                extras= {},)
        
            self.history.append(trace)

        losses = {}

        if isTrain:
            world_loss_main = w_out["loss"]
            losses["world_loss"] = world_loss_main
            losses["world_loss_recon"] = w_out.get("loss_recon", 0.0)
            losses["world_loss_reward"] = w_out.get("loss_reward", 0.0)
            losses["world_loss_done"] = w_out.get("loss_done", 0.0)
            losses["world_loss_kl"] = w_out.get("loss_kl", 0.0)
            losses["world_loss_ns"] = w_out.get("loss_ns", 0.0)
            losses["world_loss_ns_distill"] = w_out.get("loss_ns_distill", 0.0)
            losses["world_loss_ns_prior_logic"] = w_out.get("loss_ns_prior_logic", 0.0)

            critic_loss = critic_out.loss
            losses["critic_loss"] = critic_loss

            ent = entropy_actor.mean()
            ent_loss = -0.001 * ent

            adv = td_sig.detach() 
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

            mask = (1.0 - doneFlag.detach()) if (doneFlag is not None) else 1.0

            kb = act_out.get("keyboard", {})
            ms = act_out.get("mouse", {})
            op = act_out.get("option", {})

            pg = entropy_actor.new_tensor(0.0)

            if "logp_base" in kb: pg += -(mask * adv * kb["logp_base"]).mean()
            if "logp_extra" in kb: pg += -(mask * adv * kb["logp_extra"]).mean()
            if "logp_skill" in kb: pg += -(mask * adv * kb["logp_skill"]).mean()
            if "logp" in ms: pg += -(mask * adv * ms["logp"]).mean()

            if ("click_logits" in ms) and ("click_sample" in ms):
                logp_click = StableLogProbBernoulli(ms["click_logits"], ms["click_sample"])
                pg += -(mask * adv * logp_click).mean()

            if "logp_option" in op: pg += -(mask * adv * op["logp_option"]).mean()
            if "logp_beta" in op: pg += -(mask * adv * op["logp_beta"]).mean()

            actor_loss = ent_loss + 0.01 * pg

            losses["actor_loss"] = actor_loss
            losses["entropy"] = ent

            mem_loss = self.mem.GetInternalLoss()
            losses["memory_loss"] = mem_loss

            conscious_loss = conscious_out.extras["loss"]
            losses["conscious_loss"] = conscious_loss

            intention_loss = self.intention.GetInternalLoss(sym_probs)
            losses["intention_loss"] = intention_loss

            total_loss = world_loss_main + critic_loss + actor_loss + mem_loss + conscious_loss +intention_loss
            losses["total_loss"] = total_loss

        return {
            "decision": act_out,
            "world": {"state": s_t, "reward": r_t, "done": d_t},
            "critic": critic_out,
            "features": {"perc": percs_seq, "attn": atten_out, "mem": mem_feat, "mem_recall": mem_recall},
            "losses": losses}
        

    @torch.no_grad()
    def ExportBuffers(self) -> Dict[str, Any]:
        h, z, x = self.world.ExportState()
        return {
            "prev_mem": self.prev_mem,
            "prev_attn": self.prev_attn,
            "prev_key_vec": self.prev_key_vec,
            "prev_mouse": self.prev_mouse,
            "prev_option_onehot": self.prev_option_onehot,
            "prev_entropy": self.prev_entropy,
            "prev_world_h": self.prev_world_h,
            "prev_world_z": self.prev_world_z,
            "prev_world_x": self.prev_world_x,
            "world_h": h, 
            "world_z": z,
            "world_x": x,
            "perc_buffer": self.perc_buffer,
            "buf_B": self.prev_mem.size(0)}

    @torch.no_grad()
    def ImportBuffers(self, state: Dict[str, Any]):
        device = next(self.parameters()).device

        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

        if "perc_buffer" in state:
            buf = state["perc_buffer"]
            if isinstance(buf, list):
                new_buf = []
                for t in buf:
                    if isinstance(t, torch.Tensor):
                        new_buf.append(t.to(device))
                state["perc_buffer"] = new_buf

        self.prev_mem = state["prev_mem"]
        self.prev_attn = state["prev_attn"]
        self.prev_key_vec = state["prev_key_vec"]
        self.prev_mouse = state["prev_mouse"]
        self.prev_option_onehot = state["prev_option_onehot"]
        self.prev_entropy = state["prev_entropy"]
        self.prev_world_h = state["prev_world_h"]
        self.prev_world_z = state["prev_world_z"]
        self.prev_world_x = state["prev_world_x"]
        self.world.ImportState(state["world_h"], state["world_z"], state["world_x"])
        self.perc_buffer = state["perc_buffer"]
        self.buf_B = self.prev_mem.size(0)


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

        x_filt = wmSeq.new_zeros(B, T, 1) 
        P_filt = wmSeq.new_zeros(B, T, 1) 

        x_filt[:, 0, :] = wmSeq[:, 0:1] 
        P_filt[:, 0, :] = wmSeq.new_full((B, 1), float(initVar))

        for t in range(1, T):
            x_prior = x_filt[:, t - 1, :] 
            P_prior = P_filt[:, t - 1, :] + q_t 

            z_t = wmSeq[:, t:t+1]

            K_t = P_prior / (P_prior + r_wm_t + 1e-8) 

            x_post = x_prior + K_t * (z_t - x_prior) 
            P_post = (1.0 - K_t) * P_prior 

            x_filt[:, t, :] = x_post
            P_filt[:, t, :] = P_post

        x_T = x_filt[:, -1, :]
        P_T = P_filt[:, -1, :]

        ext_vec = extLast.to(device=device, dtype=dtype)

        K_ext = P_T / (P_T + r_ext_t + 1e-8)  
        x_T_corr = x_T + K_ext * (ext_vec - x_T) 
        P_T_corr = (1.0 - K_ext) * P_T 

        x_filt[:, -1, :] = x_T_corr
        P_filt[:, -1, :] = P_T_corr

        x_smooth = x_filt.clone()
        P_smooth = P_filt.clone()

        for t in range(T - 2, -1, -1):
            P_t = P_filt[:, t, :] 
            P_tp = P_t + q_t 
            C_t = P_t / (P_tp + 1e-8) 

            x_smooth[:, t, :] = x_filt[:, t, :] + C_t * (x_smooth[:, t + 1, :] - x_filt[:, t, :])
            P_smooth[:, t, :] = P_t + C_t * C_t * (P_smooth[:, t + 1, :] - P_tp)

        return x_smooth.squeeze(-1)  # [B,T]

    def SmoothWork(self, historyRef, lastRef, signal: str, 
                   attenModule: torch.Module, memModule: torch.Module, criticModule: torch.Module):
        try:
            per_buffer_list = []
            atten_list = []
            mem_list = []
            world_state_list = []
            reward_list = []
            done_list = []
            entropy_list = []

            for tr in historyRef:
                per_buffer_list.append(tr.PercBuffer)
                atten_list.append(tr.AttnFeat)
                mem_list.append(tr.MemFeat)
                world_state_list.append(tr.WorldState)
                reward_list.append(tr.Reward) 
                done_list.append(tr.Done)  
                entropy_list.append(tr.ActionEntropy)

            if signal == "Reward":
                seq_list = reward_list
            elif signal == "Done":
                seq_list = done_list
            else:
                return
            
            wm_seq = torch.cat(seq_list, dim=1)  # [B, T]
            
            smoothed = self.SmoothCorrection(wmSeq=wm_seq, extLast=lastRef)

            smoothed_list = list(smoothed.split(1, dim=1))

            start = int(memModule.time_step)

            for i in range(len(smoothed_list)):
                if i == 0: 
                    continue
                if signal == "Reward":
                    value = criticModule(memory=mem_list[i-1],attn=atten_list[i-1],state=world_state_list[i],
                                        rewardExt=smoothed_list[i],policyEntropyPrev=entropy_list[i-1],done=done_list[i],)
                else: # Done
                    value = criticModule(memory=mem_list[i-1],attn=atten_list[i-1],state=world_state_list[i],
                                        rewardExt=reward_list[i],policyEntropyPrev=entropy_list[i-1],done=smoothed_list[i],)

                td_sig = value.tdError
                rInt_sig = value.rInt
                unc_sig = value.uncertainty
                emotion_sig = value.emotion

                percs_seq = torch.stack(per_buffer_list[i], dim=1).contiguous()
                atten_out = attenModule(percs_seq, tdError=td_sig, uncertainty=unc_sig)

                _, _ = memModule(atten_out, tdError=td_sig,emotion=emotion_sig,reward=rInt_sig)
            
            self.extra_mem = memModule.ExportMemoryState(step=start)

            self.thread_end = True

        except Exception as e:
            print("[SmoothWork] error:", repr(e))
            


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
            self.EnsureFile(self.wm_mem_path) 
            if self.brain.is_online_learning:
                self.brain.world.base.InitWorldMemoryDocument(self.wm_mem_path)
                self.brain.world.base.SetMemoryOption(True, self.wm_mem_path)
            else:
                self.brain.world.InitWorldMemoryDocument(self.wm_mem_path)
                self.brain.world.SetMemoryOption(True, self.wm_mem_path)
        else:
            print(f"{self.wm_mem_path} is None")

        if self.mem_mem_path is not None:
            self.EnsureFile(self.mem_mem_path) 
            self.brain.mem.InitMemoryDocument(self.mem_mem_path)
            self.brain.mem.LoadState(self.mem_mem_path)
        else:
            print(f"{self.mem_mem_path} is None")

        self.brain.to(self.device)

        self.ResetHebbianMemory()

        if isTrain:
            actor_params = (
                list(self.brain.perc.parameters())
                + list(self.brain.attn.parameters())
                + ([] if self.brain.is_online_learning else list(self.brain.mem.parameters()))
                + list(self.brain.actor.parameters()))
        
            self.opt_actor = torch.optim.Adam(actor_params, lr=3e-4)

            self.opt_critic = torch.optim.Adam(self.brain.critic.parameters(), lr=2e-4)
            self.opt_world = torch.optim.Adam(self.brain.world.parameters(), lr=2e-4)

    def EnsureFile(self, path: str) -> bool:
        dir_ = os.path.dirname(path)
        if dir_ and (not os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
            print(f"[EnsureFile] created directory: {os.path.abspath(dir_)}")

        created = False
        if not os.path.exists(path):
            torch.save({}, path)
            created = True
            print(f"[EnsureFile] created file: {os.path.abspath(path)}")
        else:
            size = os.path.getsize(path)
            print(f"[EnsureFile] file already exists: {os.path.abspath(path)} (size={size} bytes)")

        return created

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

        if self.is_train:
            out = self.brain.Step(frame,textExt,rewardExt=reward,doneFlag=done,isTrain=self.is_train,sampleActions=sampleActions,deterministicActor=deterministicActor,)
            if out is None: return None
            key_vec = out["decision"]["key_vec"] # [B, 106]
            mouse = out["decision"]["mouse"]["a"] # [B, 2]
            total_loss = out["losses"]["total_loss"]
            return key_vec, mouse, total_loss
        else:  
            with torch.no_grad():  
                out = self.brain.Step(frame,textExt,rewardExt=reward,doneFlag=done,isTrain=self.is_train,sampleActions=sampleActions,deterministicActor=deterministicActor,)
                if out is None: return None
                key_vec = out["decision"]["key_vec"] # [B, 106]
                mouse = out["decision"]["mouse"]["a"] # [B, 2]
                return key_vec, mouse 


    def Save(self, path: str):
        if not self.is_train: return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "brain": self.brain.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_world": self.opt_world.state_dict(),
            "buffers": self.brain.ExportBuffers(),
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),}
        
        if torch.cuda.is_available():
            payload["rng_cuda_all"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)

    def Load(self, path: str, strict: bool = True, mapLocation: Optional[Union[str, torch.device]] = None):
        if not self.is_train: return
        payload = torch.load(path, map_location=mapLocation or self.device)
        self.brain.load_state_dict(payload["brain"], strict=strict)
        self.opt_actor.load_state_dict(payload["opt_actor"])
        self.opt_critic.load_state_dict(payload["opt_critic"])
        self.opt_world.load_state_dict(payload["opt_world"])
        self.brain.ImportBuffers(payload["buffers"])
        try:
            random.setstate(payload["rng_py"])
            np.random.set_state(payload["rng_np"])
            torch.set_rng_state(payload["rng_torch"])
            if torch.cuda.is_available() and ("rng_cuda_all" in payload):
                torch.cuda.set_rng_state_all(payload["rng_cuda_all"])
        except Exception:
            traceback.print_exc()


    def ResetBrainState(self, B: int, isOnlineLearning: bool):
        if isOnlineLearning:
            self.brain.world.base.ResetState(batchSize=B)
            self.brain.critic.base.ResetState(batchSize=B)
        else:
            self.brain.world.ResetState(batchSize=B)
            self.brain.critic.ResetState(batchSize=B)

        self.brain.mem.SoftReset() 
        self.brain.conscious.ResetState()

        self.brain.ResetBuffers(B=B, isOnlineLearning=isOnlineLearning, device=self.device)

    def ResetHebbianMemory(self):
        if self.brain.is_online_learning:
            self.brain.perc.base.ResetHebbianMemory()
            self.brain.attn.base.ResetHebbianMemory()
            self.brain.actor.base.ResetHebbianMemory()
            self.brain.critic.base.ResetHebbianMemory()
            self.brain.conscious.base.ResetHebbianMemory()
        else:
            self.brain.perc.ResetHebbianMemory()
            self.brain.attn.ResetHebbianMemory()
            self.brain.actor.ResetHebbianMemory()
            self.brain.critic.ResetHebbianMemory()
            self.brain.conscious.ResetHebbianMemory()

        self.brain.mem.ResetHebbianMemory()
        self.brain.conscious.ResetHebbianMemory()

    def ConvertNpImagesKeysMouses(
        self,
        imgs: np.ndarray,
        keys: np.ndarray,
        mouse: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        *,
        size: Optional[Tuple[int, int]] = (BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE),
        device: Optional[torch.device] = None,) -> Dict[str, List]:

        if imgs is not None:
            img_tensor = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous().float() / 255.0  # [B,3,H,W]
            if size is not None:
                out_h, out_w = size
                img_tensor = F.interpolate(img_tensor,size=(out_h, out_w),mode="bilinear",align_corners=False,antialias=True,)
            if device is not None: img_tensor = img_tensor.to(device)
        else: img_tensor = None

        def convert_tensor(x):
            if x is not None:
                x_tensor = torch.from_numpy(x).float()
                if device is not None:
                    x_tensor = x_tensor.to(device)
            else: x_tensor = None
            return x_tensor

        key_tensor = convert_tensor(keys)
        mouse_tensor = convert_tensor(mouse)
        reward_tensor = convert_tensor(reward)
        done_tensor = convert_tensor(done)

        return {"frames": img_tensor, "keys": key_tensor, "mouses": mouse_tensor, "rewards": reward_tensor, "dones": done_tensor,}


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
        per_module_counts = {}
        for child_name, child in self.brain.named_children():
            n = 0
            for p in child.parameters():
                if onlyTrainable and not p.requires_grad:
                    continue
                n += p.numel()
            per_module_counts[child_name] = n

        total = 0
        for p in self.brain.parameters():
            if onlyTrainable and not p.requires_grad:
                continue
            total += p.numel()

        summed_children = sum(per_module_counts.values())
        other = total - summed_children

        kind = "trainable" if onlyTrainable else "all"
        print(f"===== Parameter counts ({kind}) =====")
        for name, n in per_module_counts.items():
            print(f"{name:15s}: {n:,}")
        if other > 0:
            print(f"{'(other)':15s}: {other:,}")
        print("----------------------------------")
        print(f"TOTAL {kind} params: {total:,}")
        return total
