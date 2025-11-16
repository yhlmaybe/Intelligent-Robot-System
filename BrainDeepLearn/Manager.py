from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import threading
import collections
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import traceback
import os
import math

import debugpy

from torch.utils.data import Dataset, DataLoader

from PerceptionModule import PerceiveExtractor, PerceptionOnlineWrapper, TestPerceptionMTool
from AttentionModule import AttentionExtractor, AttentionOnlineWrapper, TestAttentionMTool
from MemoryModule import MemoryExtractor, TestMemoryMTool
from DecisionModule import DecisionExtractor, DecisionOnlineWrapper, KEYBOARD_LAYOUT, TestDecisionMTool, DecisionPlannerExtractor
from WorldModule import RSSMWorldModel, WorldModelOnlineWrapper, TestWorldMTool
from ValueEstimationModule import ValueEstimationExtractor,ValueEstimationOnlineWrapper, TestValueEstimationMTool

try:
    import imageio.v3 as iio
except Exception:
    iio = None  


def ToDevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

class BasicParameters:
    IMAGE_SIZE = 512
    IMAGE_SEQ_LEN = 16
    IMAGE_RM_LEN = math.ceil(IMAGE_SEQ_LEN * 9 / 10)

    MEMORY_MEMORY_PATH = "BrainDeepLearn/Data/MemoryMemory.pt"
    WORLD_MEMORY_PATH = "BrainDeepLearn/Data/WorldMemory.pt"
    MODULEPARAMETER_PATH = "BrainDeepLearn/Data/module_parameter.pth"
    DATA_ROOT_PATH = "BrainDeepLearn/Data"
    CKPT_PATH_TRAIN = "BrainDeepLearn/Data/training_checkpoint.pth"

    MEMORY_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/MemoryMemory.pt"
    WORLD_MEMORY_PATH_TEST = "BrainDeepLearn/TestData/WorldMemory.pt"
    DATA_ROOT_PATH_TEST = "BrainDeepLearn/TestData"
    CKPT_PATH_TEST = "BrainDeepLearn/TestData/training_test_checkpoint.pth"



class BrainCore(nn.Module):
    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        seqLen: int = BasicParameters.IMAGE_SEQ_LEN,
        plasticHebbian: bool = True,
        plasticOnlineLearning: bool = False,
        usePlanner: bool = True,):
        super().__init__()
        self.SEQ_LEN = seqLen
        self.is_online_learning = plasticOnlineLearning
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perc = PerceiveExtractor(imgSize=BasicParameters.IMAGE_SIZE,useHebbian=plasticHebbian)
        self.attn = AttentionExtractor(sequenceLength=seqLen, hebbianRate=(0.01 if plasticHebbian else 0.0), useHebbian=plasticHebbian)
        self.mem = MemoryExtractor(hebbAlpha=(0.15 if plasticHebbian else 0.0), useHebbian=plasticHebbian)
        self.actor = DecisionExtractor(stateDim=768, includeNoSkill=True, useHebb=plasticHebbian)
        self.world = RSSMWorldModel(visionDim=1024)
        self.critic = ValueEstimationExtractor(memoryDim=768, attnDim=1024, stateDim=512, useHebb=plasticHebbian)

        if plasticOnlineLearning:
            self.perc = PerceptionOnlineWrapper(self.perc)
            self.attn = AttentionOnlineWrapper(self.attn)
            self.actor = DecisionOnlineWrapper(self.actor)
            self.world = WorldModelOnlineWrapper(self.world)
            self.critic =ValueEstimationOnlineWrapper(self.critic)

        self.use_planner = usePlanner

        self.planner = DecisionPlannerExtractor().BuildPlanner(
            worldModel=self.world,
            wmIsOnlineWrapper=plasticOnlineLearning,
            KEYBOARD_LAYOUT=KEYBOARD_LAYOUT,
            includeNoSkill=True,
            horizon=5, N=64, elite=8, iters=3,
            gamma=0.99, temperature=1.0, momentum=0.15,
            laplace=1.0, minVar=1e-4, epsBern=1e-4)

        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        self.max_code = max(all_codes)
        self.keyvec_dim = (self.max_code + 1) + 2  # 106

        self._buf_B = 0
        self.ResetBuffers(B=1, isOnlineLearning=self.is_online_learning,device=self.device)
        self.to(self.device)

    @torch.no_grad()
    def ResetBuffers(self, B: int = 1, isOnlineLearning: bool = False, device: Optional[torch.device] = None):
        device = device or self.device

        def z(*s, dtype=torch.float32):
            return torch.zeros(B, *s, device=device, dtype=dtype)

        self.prev_mem = z(768)
        self.prev_attn = z(1024)
        self.prev_state = z(256)

        self.prev_key_vec = z(self.keyvec_dim) 
        self.prev_mouse = z(2)

        if isOnlineLearning:
            self.prev_option_onehot = z(self.actor.base.num_options)
        else:
            self.prev_option_onehot = z(self.actor.num_options)

        self.prev_reward = z()
        self.prev_done = z() 
        self.prev_entropy = z() 
        self.prev_unc = z() 
        self.prev_td = z() 

        self._buf_B = B

    @torch.no_grad()
    def EnsureB(self, B: int, isOnlineLearning, device: torch.device):
        if (self._buf_B != B) or (self.prev_mem.device != device):
            self.ResetBuffers(B, isOnlineLearning, device)
            if isOnlineLearning:
                self.world.base.ResetHidden(batchSize=B)
            else:
                self.world.ResetHidden(batchSize=B)

    def Step(
        self,
        frames: torch.Tensor,  # [B, T=SEQ_LEN, C, H, W]
        rewardExt: Optional[torch.Tensor] = None, # [B]
        doneFlag: Optional[torch.Tensor] = None, # [B]
        *,
        isTrain: bool = False,
        sampleActions: bool = True,
        deterministicActor: bool = False,) -> Dict[str, Any]:

        if self.is_online_learning and not isTrain:
            raise RuntimeError(f"Wrappers can only be used during training, but isTrain is {isTrain}, isUseWrappers is {self.is_online_learning}")

        B, T, C, H, W = frames.shape
        dev = frames.device
        self.EnsureB(B,self.is_online_learning, dev)

        if T != self.SEQ_LEN:
            raise ValueError(f"Expected sequence length {self.SEQ_LEN}, but got {T}. "f"frames.shape={tuple(frames.shape)}")

        # [B,T,C,H,W] -> [B*T,C,H,W]
        x = frames.view(B * T, C, H, W).contiguous()

        perc_feats = self.perc(x) # [B*T,D]

        percs_seq = perc_feats.view(B, T, -1).contiguous() # [B, T, D]

        with torch.no_grad():
            world_vis_in = self.attn(percs_seq)
        
        if self.is_online_learning:
            a_enc_prev = self.world.base.action_encoder(self.prev_key_vec, self.prev_mouse)
        else:
            a_enc_prev = self.world.action_encoder(self.prev_key_vec, self.prev_mouse)

        hPrev, zPrev = self.world.ExportState()

        if isTrain:
            if self.is_online_learning:
                wm_kwargs = {"keysVec": self.prev_key_vec,"mouseSeq": self.prev_mouse,"h0": hPrev,"z0": zPrev,"rewardSeq": rewardExt,"doneSeq": doneFlag}
                w_out = self.world(world_vis_in, **wm_kwargs)
            else: 
                w_out = self.world.ForwardTrainSeq(visionSeq=world_vis_in,keysVec= self.prev_key_vec, mouseSeq=self.prev_mouse, h0=hPrev, z0=zPrev,rewardSeq=rewardExt,doneSeq=doneFlag)
        else:
            w_out = self.world.StepPosterior(hPrev, zPrev, visionIn=world_vis_in, actionEnc=a_enc_prev, sample=False)

        hPrev, zPrev = self.world.ExportState()

        s_t = w_out["s_next"] # World State
        r_t = w_out["r_pred"] # Prediction Rewards
        d_t = w_out["d_prob"] # Termination Probability

        if self.is_online_learning:
            value_kwargs = {"rewardExt": r_t, "policyEntropyPrev": self.prev_entropy, "done": d_t}
            value_x = {"memory": self.prev_mem,"attn": self.prev_attn, "state": s_t}
            critic_out = self.critic(x=value_x, tdError=self.prev_td, uncertainty= self.prev_unc, **value_kwargs)
        else:
            critic_out = self.critic(memory=self.prev_mem,attn=self.prev_attn,state=s_t,rewardExt=r_t,policyEntropyPrev=self.prev_entropy,
                                    uncertaintyTeacher=self.prev_unc,tdErrorPrev=self.prev_td,done=d_t,)

        td_sig = critic_out.tdError
        rInt_sig = critic_out.rInt
        unc_sig = critic_out.uncertainty

        atten_out = self.attn(percs_seq, tdError=td_sig, uncertainty=unc_sig)

        mem_feat, mem_recall = self.mem(atten_out, tdError=td_sig,reward=rInt_sig)

        mem_extra_loss = self.mem.GetInternalLoss()

        with torch.no_grad():
            if self.is_online_learning:
                actor_kwargs = {"sample": sampleActions, "deterministic": deterministicActor, "prevOptionOnehot": self.prev_option_onehot}
                act_out = self.actor(x=mem_feat,**actor_kwargs)
            else:    
                act_out = self.actor(stateFeat=mem_feat,sample=sampleActions,deterministic=deterministicActor,prevOptionOnehot=self.prev_option_onehot)

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
                                          baseLogits=baseLogits,extraLogits=extraLogits,clickLogits=clickLogits,h0=hPrev.detach(),z0=zPrev.detach())
        
        if self.is_online_learning:
            actor_kwargs["prior"] = prior
            act_out = self.actor(x=mem_feat,kwargs=actor_kwargs)
        else:
            act_out = self.actor(stateFeat=mem_feat,sample=sampleActions,deterministic=deterministicActor,prevOptionOnehot=self.prev_option_onehot,prior=prior)

        key_vec = act_out["key_vec"] # [B,106]
        mouse_a = act_out["mouse"]["a"]  # [B,2]
        entropy_scalar = act_out["entropy"]

        if "option" in act_out and ("opt_onehot" in act_out["option"]):
            next_opt = act_out["option"]["opt_onehot"].detach()
        else:
            logits = act_out["option"]["logits"].detach()
            idx = torch.argmax(logits, dim=-1)
            next_opt = F.one_hot(idx, num_classes=self.actor.num_options).float()
        self.prev_option_onehot = next_opt

        self.prev_mem = mem_feat.detach()
        self.prev_attn = atten_out.detach()
        self.prev_state = s_t.detach()
        self.prev_key_vec = key_vec.detach()
        self.prev_mouse = mouse_a.detach()

        self.prev_reward = r_t.detach().view(B)
        self.prev_done = (doneFlag.detach() if doneFlag is not None else d_t.detach()).view(B)
        self.prev_entropy = entropy_scalar.detach()
        self.prev_unc = unc_sig.detach()
        self.prev_td = td_sig.detach()

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
        else:
            world_loss_main = torch.zeros((), device=dev)

        critic_loss = critic_out.loss
        losses["critic_loss"] = critic_loss
        ent = entropy_scalar.mean()
        ent_loss = -0.001 * ent 
        actor_logp_loss = 0.0

        if "option" in act_out:
            opt_out = act_out["option"]
            if "logp_option" in opt_out:
                actor_logp_loss = actor_logp_loss - 0.01 * opt_out["logp_option"].mean()
            if "logp_beta" in opt_out:
                actor_logp_loss = actor_logp_loss - 0.01 * opt_out["logp_beta"].mean()
        
        actor_loss = ent_loss + actor_logp_loss
        losses["actor_loss"] = actor_loss
        losses["entropy"] = ent
        losses["memory_loss"] = mem_extra_loss

        total_loss = world_loss_main + critic_loss + actor_loss + mem_extra_loss
        losses["total_loss"] = total_loss

        return {
            "decision": act_out,
            "world": {"state": s_t, "reward": r_t, "done": d_t},
            "critic": critic_out,
            "features": {"perc": percs_seq, "attn": atten_out, "mem": mem_feat, "mem_recall": mem_recall},
            "losses": losses}
        

    @torch.no_grad()
    def ExportBuffers(self) -> Dict[str, Any]:
        h, z = self.world.ExportState()
        return {
            "prev_mem": self.prev_mem,
            "prev_attn": self.prev_attn,
            "prev_state": self.prev_state,
            "prev_key_vec": self.prev_key_vec,
            "prev_mouse": self.prev_mouse,
            "prev_option_onehot": self.prev_option_onehot,
            "prev_reward": self.prev_reward,
            "prev_done": self.prev_done,
            "prev_entropy": self.prev_entropy,
            "prev_unc": self.prev_unc,
            "prev_td": self.prev_td,
            "world_h": h, 
            "world_z": z,}

    @torch.no_grad()
    def ImportBuffers(self, state: Dict[str, Any]):
        device = next(self.parameters()).device
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
        self.prev_mem = state["prev_mem"]
        self.prev_attn = state["prev_attn"]
        self.prev_state = state["prev_state"]
        self.prev_key_vec = state["prev_key_vec"]
        self.prev_mouse = state["prev_mouse"]
        self.prev_option_onehot = state["prev_option_onehot"]
        self.prev_reward = state["prev_reward"]
        self.prev_done = state["prev_done"]
        self.prev_entropy = state["prev_entropy"]
        self.prev_unc = state["prev_unc"]
        self.prev_td = state["prev_td"]
        self.world.ImportState(state["world_h"], state["world_z"])



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
                + list(self.brain.mem.parameters())
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
        frames: torch.Tensor, # [B,T,C,H,W]
        *,
        reward: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
        sampleActions: bool = True,
        deterministicActor: bool = False,):

        frames = frames.to(self.device)
        B, T, C, H, W = frames.shape
        if T != self.brain.SEQ_LEN:
                raise ValueError(f"Expected frames with sequence length {self.brain.SEQ_LEN}, but got {T}. " f"Shape received: {tuple(frames.shape)}.")

        out = self.brain.Step(frames,rewardExt=reward,doneFlag=done,isTrain=self.is_train,sampleActions=sampleActions,deterministicActor=deterministicActor,)
        
        key_vec = out["decision"]["key_vec"] # [B, 106]
        mouse = out["decision"]["mouse"]["a"] # [B, 2]
        total_loss = out["losses"]["total_loss"]

        return key_vec, mouse, total_loss

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


    def ResetBrainState(self, isOnlineLearning: bool):
        if isOnlineLearning:
            self.brain.world.base.ResetHidden(batchSize=1)
        else:
            self.brain.world.ResetHidden(batchSize=1)

        self.brain.ResetBuffers(B=1, isOnlineLearning=isOnlineLearning, device=self.device)

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

    def StackNpImagesKeysMouses(
        self,
        imgs: List[np.ndarray],
        keys: Optional[List[Union[np.ndarray, torch.Tensor, float, int]]] = None,
        mouse: Optional[List[Union[np.ndarray, torch.Tensor, float, int]]] = None,
        reward: Optional[List[Union[np.ndarray, torch.Tensor, float, int]]] = None,
        done: Optional[List[Union[np.ndarray, torch.Tensor, float, int]]] = None,
        *,
        B: int,
        T: int,
        size: Optional[Tuple[int, int]] = (BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE),
        device: Optional[torch.device] = None,) -> Dict[str, List]:
        assert len(imgs) == B * T, f"got {len(imgs)} images, but B*T = {B*T}"

        img_tensors: List[torch.Tensor] = []
        for i, im in enumerate(imgs):
            if not isinstance(im, np.ndarray):
                raise TypeError(f"image {i} is {type(im)}, expected np.ndarray")
            if im.ndim != 3 or im.shape[2] != 3:
                raise ValueError(f"image {i} has shape {im.shape}, expected [H,W,3]")
            t = torch.from_numpy(im).permute(2, 0, 1).contiguous().float() / 255.0  # [3,H,W]
            img_tensors.append(t)

        x = torch.stack(img_tensors, dim=0)  # [B*T, 3, H, W]

        if size is not None:
            out_h, out_w = size
            x = F.interpolate(x,size=(out_h, out_w),mode="bilinear",align_corners=False,antialias=True,)

        BxT, C, H, W = x.shape
        x = x.view(B, T, C, H, W)


        def PickLastFromSeq(
            seq: List[Union[np.ndarray, torch.Tensor, float, int]],
            *,
            B: int,
            T: int,
            clamp_min: Optional[float] = None,
            clamp_max: Optional[float] = None,
            squeeze_scalar: bool = False, ) -> torch.Tensor:
            assert len(seq) == B * T, f"got {len(seq)} items, but need B*T={B*T}"
            picked = []
            for b in range(B):
                idx = b * T + (T - 1)
                v = seq[idx]
                if isinstance(v, np.ndarray):
                    vt = torch.from_numpy(v).float()
                elif isinstance(v, torch.Tensor):
                    vt = v.float()
                else:
                    vt = torch.tensor(float(v), dtype=torch.float32)

                if squeeze_scalar:
                    vt = vt.view(-1)[0]  
                picked.append(vt)

            out = torch.stack(picked, dim=0) 
            if (clamp_min is not None) or (clamp_max is not None):
                out = torch.clamp(out,clamp_min if clamp_min is not None else -float("inf"), clamp_max if clamp_max is not None else float("inf"))
            return out

        key_tensor = PickLastFromSeq(keys,  B=B, T=T) if keys  is not None else None
        mouse_tensor = PickLastFromSeq(mouse, B=B, T=T) if mouse is not None else None
        reward_tensor = PickLastFromSeq(reward, B=B, T=T, clamp_min=-10.0, clamp_max=10.0, squeeze_scalar=True) if reward is not None else None
        done_tensor = PickLastFromSeq(done, B=B, T=T, clamp_min=0.0, clamp_max=1.0, squeeze_scalar=True) if done is not None else None

        if device is not None:
            x = x.to(device)
            if key_tensor is not None:
                key_tensor = key_tensor.to(device)
            if mouse_tensor is not None:
                mouse_tensor = mouse_tensor.to(device)
            if reward_tensor is not None:
                reward_tensor = reward_tensor.to(device)
            if done_tensor is not None:
                done_tensor = done_tensor.to(device)

        return {"frames": x, "keys": key_tensor, "mouses": mouse_tensor, "rewards": reward_tensor, "dones": done_tensor,}


    def UpdateWrappers(self, wrappers, action: str, **kwargs):
        results = []
        for w in wrappers:
            out = w.Update(action, **kwargs)
            results.append(out)
        return results
    
    def UpdateAllWrappers(self, action: str, **kwargs):
        wrappers = [self.brain.perc, self.brain.attn, self.brain.actor, self.brain.world, self.brain.critic]
        results = []
        for w in wrappers:
            out = w.Update(action, **kwargs)
            results.append(out)
        return results


class OfflineGameDataset(Dataset):
    def __init__(self, root: str) -> None:
        p = Path(root)
        self.imgs = sorted((p / "frames").glob("*.png"))
        self.keys = sorted((p / "keys").glob("*.npy"))
        self.mouse = sorted((p / "mouse").glob("*.npy"))
        self.reward = sorted((p / "reward").glob("*.npy")) 
        self.done = sorted((p / "done").glob("*.npy")) 
        assert len(self.imgs) == len(self.keys) == len(self.mouse)==len(self.reward) == len(self.done), "frames/keys/mouse/reward/done The number of files is inconsistent."

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        imgs = iio.imread(self.imgs[idx])
        keys = np.load(self.keys[idx]).astype(np.float32)
        mouse = np.load(self.mouse[idx]).astype(np.float32)
        reward = np.load(self.reward[idx]).astype(np.float32)
        done = np.load(self.done[idx]).astype(np.float32)
        return imgs, keys, mouse, reward, done




class ModuleController:
    def __init__(self):
        self._lock = threading.Lock()
        self.status: Dict[str, Any] = {
            "state": "idle",
            "epoch": 0, "total_epochs": 0,
            "batch": 0, "total_batches": 0,
            "train_loss": 0.0, "val_loss": 0.0,
            "message": "Waiting to start",
            "trace": ""}
        self.stop_requested = False
        self.pause_requested = False
        self.reset_hebbian = False

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
        
    def ResteStatus(self):
        self.status: Dict[str, Any] = {
            "state": "idle",
            "epoch": 0, "total_epochs": 0,
            "batch": 0, "total_batches": 0,
            "train_loss": 0.0, "val_loss": 0.0,
            "message": "Waiting to start",
            "trace": ""}

    def RequestStop(self):
        with self._lock:
            self.stop_requested = True

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


class ManagerFunction:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.controller = ModuleController()

        self.br_thread: Optional[threading.Thread] = None
        self.message_thread: Optional[threading.Thread] = None
        self.is_begin = False

        self.test = {
            "perception": TestPerceptionMTool(),
            "attention": TestAttentionMTool(),
            "memory": TestMemoryMTool(),
            "decision": TestDecisionMTool(),
            "world": TestWorldMTool(),
            "value": TestValueEstimationMTool(),}

    def StartTraining(self, epochs: int = 5, batchSize: int = 32, valSplit: float = 0.1, resume: bool = True, onlineLearning:bool = False,isTest: bool = False):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.is_begin = True

        ckpt_path = BasicParameters.CKPT_PATH_TEST if isTest else BasicParameters.CKPT_PATH_TRAIN
        wm_mem_path = BasicParameters.WORLD_MEMORY_PATH_TEST if isTest else BasicParameters.WORLD_MEMORY_PATH
        mem_mem_path = BasicParameters.MEMORY_MEMORY_PATH_TEST if isTest else BasicParameters.MEMORY_MEMORY_PATH
        root = BasicParameters.DATA_ROOT_PATH_TEST if isTest else BasicParameters.DATA_ROOT_PATH

        self.br_thread = threading.Thread(
            target=self.TrainLoop, args=(
                root, epochs, batchSize, valSplit, resume, onlineLearning),
                kwargs={"worldMemPath": wm_mem_path, "memMemPath": mem_mem_path, "ckptPath": ckpt_path}, 
                daemon=False)
        self.br_thread.start()
        return True

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

    def TrainLoop(self,root: str, epochs: int, batchSize: int, valSplit: float, resume: bool, onlineLearning = False, *, worldMemPath: str = None, memMemPath: str = None, ckptPath: str = None,):
        try:
            torch.autograd.set_detect_anomaly(True)

            ds = OfflineGameDataset(root)

            brain = BrainCore(device=self.device, plasticHebbian=True, plasticOnlineLearning=onlineLearning, usePlanner=False,)

            agent = Agent(brain, isTrain=True, device=self.device, worldMemoryPath=worldMemPath, memMemoryPath=memMemPath,)

            if onlineLearning:
                agent.UpdateAllWrappers("autogrow")

            SEQ_LEN = agent.brain.SEQ_LEN

            start_epoch = 0
            best_val = float("inf")
            train_ds, val_ds = None, None

            testSplit = 0.1

            if resume and Path(ckptPath).exists():
                start_epoch, best_val, train_ds, val_ds = self.LoadCheckpoint(brain, agent, ds, ckptPath)

            if train_ds is None:
                n_total = len(ds)
                n_test = int(n_total * testSplit)
                n_val = int(n_total * valSplit)
                n_train = n_total - n_val - n_test
                train_ds, val_ds, test_ds = torch.utils.data.random_split(
                    ds, [n_train, n_val, n_test])
            else:
                n_used = len(train_ds) + len(val_ds)
                n_left = len(ds) - n_used
                if n_left > 0:
                    test_ds, = torch.utils.data.random_split(ds, [n_left])
                else:
                    test_ds = val_ds

            train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True,)
            val_dl = DataLoader(val_ds, batch_size=batchSize, shuffle=False, num_workers=0)
            test_dl = DataLoader(test_ds, batch_size=batchSize, shuffle=False, num_workers=0)

            bce = nn.BCELoss()
            mse = nn.MSELoss()

            all_codes = []
            for grp in KEYBOARD_LAYOUT.values():
                all_codes += list(grp.values())

            max_code = max(all_codes)
            keys_dim = max_code + 1 + 2

            patience = 5 
            min_delta = 1e-4 
            no_improve = 0
            target_acc = 0.90 
            max_gap = 0.1

            self.controller.SetStatus("training", "Training started", epoch=start_epoch, total_epochs=epochs, batch=0, total_batches=len(train_dl),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

                while self.controller.ShouldPause():
                    self.controller.SetStatus("paused", "Training paused")
                    time.sleep(0.2)

                brain.train()
                epoch_loss = 0.0
                nb = 0

                img_buf: List[np.ndarray] = []
                key_buf: List[Any] = []
                mouse_buf: List[Any] = []
                reward_buf: List[Any] = []
                done_buf: List[Any] = []

                agent.ResetBrainState(isOnlineLearning=onlineLearning)

                for bi, (img_b, key_b, mouse_b, reward_b, done_b) in enumerate(train_dl, start=1):
                    B_cur = img_b.shape[0]

                    for i in range(B_cur):
                        img_item = img_b[i]
                        if isinstance(img_item, torch.Tensor):
                            img_np = img_item.cpu().numpy()
                        else:
                            img_np = img_item

                        img_buf.append(img_np)
                        key_buf.append(key_b[i])
                        mouse_buf.append(mouse_b[i])
                        reward_buf.append(reward_b[i])
                        done_buf.append(done_b[i])

                        if len(img_buf) == SEQ_LEN * batchSize:
                            pack = agent.StackNpImagesKeysMouses(imgs=img_buf, keys=key_buf, mouse=mouse_buf, reward=reward_buf, done=done_buf, B=batchSize, T=SEQ_LEN, device=self.device,)

                            frames = pack["frames"]
                            keys_t = pack["keys"]
                            mouse_t = pack["mouses"]
                            reward_t = pack["rewards"]
                            done_t = pack["dones"]

                            if self.controller.ShouldResetHebbian():
                                agent.ResetHebbianMemory()
                                self.controller.RequestCancelResetHebbian()

                            key_pred, mouse_pred, model_loss = agent.Act(frames, reward=reward_t, done=done_t, deterministicActor=False,)

                            bc_loss = torch.zeros((), device=self.device)
                            if keys_t is not None:
                                K_use = min(keys_t.size(1), keys_dim)
                                bc_loss = bc_loss + bce(key_pred[:, :K_use], keys_t[:, :K_use].float())
                            if mouse_t is not None:
                                bc_loss = bc_loss + 0.05 * mse(mouse_pred, mouse_t)

                            loss = model_loss + bc_loss

                            agent.opt_world.zero_grad(set_to_none=True)
                            agent.opt_critic.zero_grad(set_to_none=True)
                            agent.opt_actor.zero_grad(set_to_none=True)

                            loss.backward()
                            if onlineLearning:
                                agent.UpdateAllWrappers("accumulategrads")
                                agent.UpdateAllWrappers("autogrow")

                            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)

                            for name, p in brain.named_parameters():
                                if p.grad is None:
                                    print("NO GRAD:", name)
                                elif not torch.isfinite(p.grad).all():
                                    print("BAD GRAD:", name, p.grad.min(), p.grad.max(),)

                            agent.opt_world.step()
                            agent.opt_critic.step()
                            agent.opt_actor.step()

                            epoch_loss += float(loss.item())
                            nb += 1

                            self.controller.SetStatus("training", "Training...", epoch=ep + 1, total_epochs=epochs, batch=bi, total_batches=len(train_dl), train_loss=float(loss.item()),)

                            img_buf.clear()
                            key_buf.clear()
                            mouse_buf.clear()
                            reward_buf.clear()
                            done_buf.clear()

                    if self.controller.ShouldStop():
                        break
                    while self.controller.ShouldPause():
                        self.controller.SetStatus("paused", "Training paused")
                        time.sleep(0.2)

                avg_train = epoch_loss / max(1, nb)

                self.controller.SetStatus("training", f"Epoch {ep+1}/{epochs} done, avg_train={avg_train:.4f}", epoch=ep + 1, total_epochs=epochs,)

                if self.controller.ShouldStop():
                    break

                def eval_split(dl):
                    brain.eval()
                    split_loss = 0.0
                    split_batches = 0
                    total_correct = 0
                    total_elems = 0

                    v_img_buf: List[np.ndarray] = []
                    v_key_buf: List[Any] = []
                    v_mouse_buf: List[Any] = []
                    v_reward_buf: List[Any] = []
                    v_done_buf: List[Any] = []

                    with torch.no_grad():
                        for (img_b, key_b, mouse_b, reward_b, done_b) in dl:
                            B_cur = img_b.shape[0]
                            for i in range(B_cur):
                                img_item = img_b[i]
                                if isinstance(img_item, torch.Tensor):
                                    img_np = img_item.cpu().numpy()
                                else:
                                    img_np = img_item

                                v_img_buf.append(img_np)
                                v_key_buf.append(key_b[i])
                                v_mouse_buf.append(mouse_b[i])
                                v_reward_buf.append(reward_b[i])
                                v_done_buf.append(done_b[i])

                                if len(v_img_buf) == SEQ_LEN * batchSize:
                                    v_pack = agent.StackNpImagesKeysMouses(imgs=v_img_buf, keys=v_key_buf, mouse=v_mouse_buf, reward=v_reward_buf, done=v_done_buf, B=batchSize, T=SEQ_LEN, device=self.device,)

                                    v_frames = v_pack["frames"]
                                    v_keys_t = v_pack["keys"]
                                    v_mouse_t = v_pack["mouses"]

                                    v_key_pred, v_mouse_pred, _ = agent.Act(v_frames, reward=None, done=None, deterministicActor=True,)

                                    cur_loss = torch.zeros((), device=self.device)
                                    if v_keys_t is not None:
                                        K_use = min(v_key_pred.size(1), v_keys_t.size(1), keys_dim,)
                                        cur_loss = cur_loss + bce(v_key_pred[:, :K_use], v_keys_t[:, :K_use].float(),)

                                        pred_bin = (v_key_pred[:, :K_use] > 0.5).float()
                                        tgt_bin = v_keys_t[:, :K_use].float()
                                        total_correct += (pred_bin == tgt_bin).float().sum().item()
                                        total_elems += pred_bin.numel()

                                    if v_mouse_t is not None:
                                        cur_loss = cur_loss + 0.05 * mse(v_mouse_pred, v_mouse_t)

                                    split_loss += float(cur_loss.item())
                                    split_batches += 1

                                    v_img_buf.clear()
                                    v_key_buf.clear()
                                    v_mouse_buf.clear()
                                    v_reward_buf.clear()
                                    v_done_buf.clear()

                    avg_split_loss = split_loss / max(1, split_batches)
                    split_acc = (total_correct / total_elems if total_elems > 0 else 0.0)
                    return avg_split_loss, split_acc

                avg_val, val_acc = eval_split(val_dl)
                test_loss, test_acc = eval_split(test_dl)

                improved = (best_val - avg_val) > min_delta
                if improved:
                    best_val = avg_val
                    no_improve = 0

                    ckpt = {
                        "epoch": ep + 1,
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
                        "rng": {
                            "python": random.getstate(),
                            "torch": torch.get_rng_state(),
                            "numpy": np.random.get_state(),},
                        "buffers": brain.ExportBuffers(),}
                    torch.save(ckpt, ckptPath)
                else:
                    no_improve += 1

                self.controller.SetStatus(
                    "training",
                    (f"Epoch {ep+1}/{epochs} done | " f"train {avg_train:.4f} | " f"val {avg_val:.4f}, acc={val_acc:.3f} | " f"test {test_loss:.4f}, acc={test_acc:.3f}"), val_loss=avg_val,)

                if (val_acc >= target_acc and test_acc >= target_acc and abs(val_acc - test_acc) <= max_gap):
                    self.controller.SetStatus("completed", f"Val/Test accuracies high & close: val={val_acc:.3f}, test={test_acc:.3f}",)
                    agent.UpdateAllWrappers("commit")
                    break

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

        params = {"brain": raw["brain"],}

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
        brain.load_state_dict(ckpt["brain"])
        agent.opt_actor.load_state_dict(ckpt["opt_actor"])
        agent.opt_critic.load_state_dict(ckpt["opt_critic"])
        agent.opt_world.load_state_dict(ckpt["opt_world"])

        brain.ImportBuffers(ckpt["buffers"])
        random.setstate(ckpt["rng"]["python"])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        np.random.set_state(ckpt["rng"]["numpy"])

        if ckpt.get("train_indices") is not None:
            train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
            val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
        else:
            train_ds = val_ds = None

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        return start_epoch, best_val, train_ds, val_ds

    def StartDeployment(self, cameraIndex: int = 0, useHebbian: bool = True, usePlanner: bool = True):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.is_begin = True
        self.controller.stop_requested = False 
        self.br_thread = threading.Thread(target=self.DeployLoop,args=(cameraIndex,),kwargs={"useHebbian": useHebbian, "usePlanner": usePlanner},daemon=False,)
        self.br_thread.start()
        return True


    def DeployLoop(self, cameraIndex: int,* ,useHebbian: bool = True, usePlanner: bool = True,):
        try:
            brain = BrainCore(device=self.device,plasticHebbian=useHebbian,plasticOnlineLearning=False,usePlanner=usePlanner,)

            model_path = BasicParameters.MODULEPARAMETER_PATH

            if os.path.exists(model_path):
                try:
                    sd = torch.load(model_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    print(f"Safe mode loading failed: {e}, try the normal mode")
                    sd = torch.load(model_path, map_location=self.device)
            else:
                msg = f"The module file is not exit: {model_path}"
                print(msg)
                sd = None
                self.controller.SetStatus("error", msg)
                return 

            if isinstance(sd, dict) and "brain" in sd:
                brain.load_state_dict(sd["brain"], strict=False)
            else:
                brain.load_state_dict(sd, strict=False)

            brain.to(self.device)
            brain.eval()

            agent = Agent(brain,isTrain=False,device=self.device,worldMemoryPath=BasicParameters.WORLD_MEMORY_PATH,memMemoryPath=BasicParameters.MEMORY_MEMORY_PATH,)
            agent.ResetBrainState(isOnlineLearning=False)

            seq_len = brain.SEQ_LEN
            rm_len = BasicParameters.IMAGE_RM_LEN

            if iio is None:
                raise RuntimeError("imageio.v3 cant use")

            self.controller.SetStatus("is_begin", "Deployment started")

            code_to_name: dict[int, str] = {}
            all_codes = []
            for grp, mp in KEYBOARD_LAYOUT.items():
                for name, code in mp.items():
                    code_to_name.setdefault(code, name)
                    all_codes.append(code)
            max_code = max(all_codes)

            frame_buf: List[np.ndarray] = []

            with iio.imopen(f"<video{cameraIndex}>", "r") as cam:
                for frame_np in cam:
                    if self.controller.ShouldStop():
                        break

                    frame_buf.append(frame_np)

                    if len(frame_buf) < seq_len:
                        continue

                    pack = agent.StackNpImagesKeysMouses(imgs=frame_buf,B=1, T=seq_len,device=self.device,)
                    frames = pack["frames"]

                    if self.controller.ShouldResetHebbian():
                        agent.ResetHebbianMemory()
                        self.controller.RequestCancelResetHebbian()

                    key_vec, mouse, _ = agent.Act(frames,reward=None,done=None,sampleActions=True,deterministicActor=True,)

                    kv = key_vec[0].detach().cpu()
                    ms = mouse[0].detach().cpu()

                    pressed_names: list[str] = []
                    for code in range(max_code + 1):
                        if float(kv[code]) > 0.5:
                            pressed_names.append(code_to_name.get(code, f"Key{code}"))

                    if float(kv[max_code + 1]) > 0.5:
                        pressed_names.append("MouseLeft")
                    if float(kv[max_code + 2]) > 0.5:
                        pressed_names.append("MouseRight")

                    names_str = "[" + ", ".join(pressed_names) + "]" if pressed_names else "[]"

                    self.controller.SetStatus("is_begin",(f"keys:{names_str} "f"mouse:dx={float(ms[0]):.3f},dy={float(ms[1]):.3f}"),)

                    drop_n = len(frame_buf) - rm_len
                    del frame_buf[:drop_n]

            self.controller.SetStatus("stopped", "Deployment stopped")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Deployment error: {e}", trace=tb)
        finally:
            self.is_begin = False



    def TestPerceptionModule(self):
        t = self.test["perception"]
        return t.RunAll()

    def TestAttentionModule(self):
        t = self.test["attention"]
        return t.RunAll()

    def TestMemoryModule(self):
        t = self.test["memory"]
        return t.RunAll()

    def TestDecisionModule(self):
        t = self.test["decision"]
        return t.RunAll()

    def TestWorldModule(self):
        t = self.test["world"]
        return t.RunAll()

    def TestValueEstimationModule(self):
        t = self.test["value"]
        return t.RunAll()
    

    def MonitorTraining(self):
        try:
            while True:
                st = self.GetCurrentStatus()
                print(
                    f"[TRAIN] {st['state']} | epoch {st['epoch']}/{st['total_epochs']} "
                    f"| batch {st['batch']}/{st['total_batches']} "
                    f"| train_loss={st['train_loss']:.4f} | msg={st['message']}")

                if st["state"] == "error":
                    trace = st.get("trace")
                    if trace:
                        print("\n====== TRAIN ERROR TRACEBACK ======\n")
                        print(trace)
                        print("===================================\n")

                if st["state"] in ("completed", "stopped", "error"):
                    self.controller.ResteStatus()
                    break

                time.sleep(1)

        except Exception as e:
            print(f"[MonitorTraining] monitor raised: {e}")
            print(traceback.format_exc())


    def MonitorDeployment(self):
        try:
            while True:
                st = self.GetCurrentStatus()
                print(f"[DEPLOY] {st['state']} | msg={st['message']}")

                if st["state"] == "error":
                    trace = st.get("trace")
                    if trace:
                        print("\n====== DEPLOY ERROR TRACEBACK ======\n")
                        print(trace)
                        print("====================================\n")

                if st["state"] in ("stopped", "error"):
                    self.controller.ResteStatus()
                    break

                time.sleep(0.5)

        except Exception as e:
            print(f"[MonitorDeployment] monitor raised: {e}")
            print(traceback.format_exc())


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
            return {"False": False, "msg": "StartTraining returns False (training may already be running)"} 

        try:
            if iio is None:
                raise RuntimeError("imageio.v3 error")

            rng = np.random.default_rng(seed)
            root = Path(dataRoot)
            if root.exists():
                shutil.rmtree(root)
            (root / "frames").mkdir(parents=True, exist_ok=True)
            (root / "keys").mkdir(parents=True, exist_ok=True)
            (root / "mouse").mkdir(parents=True, exist_ok=True)
            (root / "reward").mkdir(parents=True, exist_ok=True)
            (root / "done").mkdir(parents=True, exist_ok=True)

            all_codes = []
            for grp in KEYBOARD_LAYOUT.values():
                all_codes += list(grp.values())
            max_code = max(all_codes)
            keys_dim = max_code + 1 + 2

            base_codes = [KEYBOARD_LAYOUT["base_keys"][k] for k in KEYBOARD_LAYOUT["base_keys"]]
            extra_codes = []
            for grp in ["menu_keys", "system_keys", "alpha_keys"]:
                extra_codes += [KEYBOARD_LAYOUT[grp][k] for k in KEYBOARD_LAYOUT[grp]]
            skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in KEYBOARD_LAYOUT["skill_keys"]]

            H, W = BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE
            for i in range(nSamples):
                img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
                iio.imwrite(str(root / "frames" / f"{i:05d}.png"), img)

                keys = np.zeros((keys_dim,), dtype=np.float32)
                for code in base_codes:
                    keys[code] = 1.0 if rng.random() < 0.10 else 0.0
                for code in extra_codes:
                    keys[code] = 1.0 if rng.random() < 0.05 else 0.0
                if rng.random() >= 0.50:
                    keys[rng.choice(skill_codes)] = 1.0

                keys[max_code + 1] = 1.0 if rng.random() < 0.15 else 0.0
                keys[max_code + 2] = 1.0 if rng.random() < 0.05 else 0.0
                np.save(str(root / "keys" / f"{i:05d}.npy"), keys)

                mouse = rng.normal(loc=0.0, scale=2.0, size=(2,)).astype(np.float32)
                np.save(str(root / "mouse" / f"{i:05d}.npy"), mouse)

                reward = rng.normal(loc=0.0, scale=2.0, size=(2,)).astype(np.float32)
                np.save(str(root / "reward" / f"{i:05d}.npy"), reward)

                done = rng.normal(loc=0.0, scale=2.0, size=(2,)).astype(np.float32)
                np.save(str(root / "done" / f"{i:05d}.npy"), done)

            print("[SmokeTest] start train...")

            ok = self.StartTraining(epochs=epochs,batchSize=batchSize,valSplit=valSplit,resume=False, onlineLearning=onlineLearning, isTest=True)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"False": False, "msg": "StartTraining returns False (training may already be running)"}

            self.message_thread = threading.Thread(target=self.MonitorTraining,args=(),daemon=False,)
            self.message_thread.start()

            #self.TrainLoop(root, epochs, batchSize, valSplit, False, onlineLearning, worldMemPath=BasicParameters.WORLD_MEMORY_PATH_TEST, memMemPath=BasicParameters.MEMORY_MEMORY_PATH_TEST,ckptPath=BasicParameters.CKPT_PATH_TEST)
        
            return {"ok": True}

        except Exception as e:
            print(f"TestModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def TrainModule(
        self,
        onlineLearning: bool,
        epochs: int = 6,
        batchSize: int = 1,
        valSplit: float = 0.2,
        isResume: bool = False,
        *,
        dataRoot: str = BasicParameters.DATA_ROOT_PATH,) -> Dict[str, Any]:
        try:
            root = Path(dataRoot)

            def has_existing_data(p: Path) -> bool:
                frames_dir = p / "frames"
                keys_dir = p / "keys"
                mouse_dir = p / "mouse"
                reward_dir = p / "reward"
                done_dir = p / "done"
                if not (frames_dir.exists() and keys_dir.exists() and mouse_dir.exists() and reward_dir.exists() and done_dir.exists()):
                    return False
                if not any(frames_dir.glob("*.png")):
                    return False
                return True

            if not has_existing_data(root):
                print(f"[Train] no dataset found at {root}, please prepare frames/keys/mouse/reward/done first.")
                return {"ok": False, "msg": "no dataset"}

            print(f"[Train] use existing dataset at: {root}")

            ok = self.StartTraining(epochs=epochs, batchSize=batchSize, valSplit=valSplit, resume=isResume, onlineLearning=onlineLearning,isTest=False,)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"False": False, "msg": "StartTraining returns False (training may already be running)"}

            self.message_thread = threading.Thread(target=self.MonitorTraining,args=(),daemon=False,)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"ModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def DeployModule(
        self,
        cameraIndex: int = 0,
        useHebbian: bool = True,
        usePlanner: bool = True,) -> Dict[str, Any]:
        try:
            ok = self.StartDeployment(cameraIndex=cameraIndex, useHebbian=useHebbian, usePlanner=usePlanner,)

            if not ok:
                print("StartDeployment returns False (deployment may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.message_thread = threading.Thread(target=self.MonitorDeployment,args=(),daemon=False,)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"DeployModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise