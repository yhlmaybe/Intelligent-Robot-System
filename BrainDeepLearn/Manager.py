from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import threading
import collections
import random
import time
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import traceback

from torch.utils.data import Dataset, DataLoader

from PerceptionModule import PerceiveExtractor, TestPerceptionMTool
from AttentionModule import AttentionExtractor, TestAttentionMTool
from MemoryModule import MemoryExtractor, TestMemoryMTool
from DecisionModule import DecisionExtractor, KEYBOARD_LAYOUT, TestDecisionMTool, DecisionPlannerExtractor
from WorldModule import RSSMWorldModel, TestWorldMTool
from ValueEstimationModule import ValueEstimationExtractor, TestValueEstimationMTool

try:
    import imageio.v3 as iio
except Exception:
    iio = None  


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


class BrainCore(nn.Module):
    SEQ_LEN = 16
    def __init__(self, device: Optional[torch.device] = None, plasticHebbian: bool = True, plasticMeta: bool = True, usePlanner: bool = True):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perc = PerceiveExtractor(useHebbian=plasticHebbian)
        if hasattr(self.perc, "plastic_on"):
            self.perc.plastic_on = plasticMeta

        self.attn = AttentionExtractor(hebbianRate=0.01 if plasticHebbian else 0.0)

        self.mem = MemoryExtractor(hebbAlpha=0.15 if plasticHebbian else 0.0, useMeta=plasticMeta)

        self.actor = DecisionExtractor(stateDim=768, useHebbOnline=plasticHebbian)

        self.world = RSSMWorldModel(visionDim=1024, actionDim=128, deterDim=256, stochDim=32, stateDim=256, useDecoder=True)
        self.world.ResetHidden(batchSize=1, device=self.device) 

        self.critic = ValueEstimationExtractor(memoryDim=768, attnDim=1024, stateDim=256)

        self.use_planner = usePlanner
        self.plan_horizon = 5
        self.planner = DecisionPlannerExtractor().BuildPlanner(
            worldModel=self.world, 
            KEYBOARD_LAYOUT=KEYBOARD_LAYOUT,
            includeNoSkill=True,
            horizon=self.plan_horizon,
            N=64,elite=8,iters=3,       
            gamma=0.99, temperature=1.0, momentum=0.15,
            laplace=1.0, minVar=1e-4, epsBern=1e-4)

        self._buf_B = 0
        self.ResetBuffers(B=1, device=self.device)

        self.to(self.device)

    @torch.no_grad()
    def ResetBuffers(self, B: int = 1, device: Optional[torch.device] = None):
        
        device = device or self.device

        def z(*s, dtype=torch.float32):
            return torch.zeros(B, *s, device=device, dtype=dtype)

        self.prev_mem = z(768)
        self.prev_attn = z(1024)
        self.prev_state = z(256)

        self.prev_keys128 = z(128)
        self.prev_mouse = z(2)

        self.prev_reward = z()
        self.prev_done = z()

        self.prev_entropy = z()
        self.prev_unc = z()
        self.prev_td = z()

        self.perc_buf = z(self.SEQ_LEN, 1024)
        self.tlen = torch.zeros(B, dtype=torch.long, device=device)
        self.have_prev = False
        self._buf_B = B

    @torch.no_grad()
    def EnsureB(self, B: int, device: torch.device):
        if (self._buf_B != B) or (self.prev_mem.device != device):
            self.ResetBuffers(B, device)
            self.world.ResetHidden(batchSize=B, device=device)

    @torch.no_grad()
    def PushPerc(self, feat_p: torch.Tensor):
        B = feat_p.size(0)
        self.perc_buf[:, :-1] = self.perc_buf[:, 1:].clone()
        self.perc_buf[:, -1] = feat_p
        self.tlen = torch.clamp(self.tlen + 1, max=self.SEQ_LEN)

    def Step(self, 
             frame: torch.Tensor,
             rewardExt: Optional[torch.Tensor] = None,
             doneFlag: Optional[torch.Tensor] = None,
             *,
             sampleActions: bool = True,
             deterministicActor: bool = False) -> Dict[str, Any]:
        B, dev = frame.size(0), frame.device
        self.EnsureB(B, dev)

        feat_p = self.perc(frame)
        self.PushPerc(feat_p)
        seq = self.perc_buf[:, -int(self.tlen.min().item() or 1):]

        a_enc = self.world.action_encoder(self.prev_keys128, self.prev_mouse)
        hPrev, zPrev = self.world.ExportState()
        world_out = self.world.StepPosterior(hPrev, zPrev, visionIn=feat_p, actionEnc=a_enc, deterministicZ=False)
        s_t = world_out["s_next"]
        r_t = world_out["r_pred"]
        d_t = world_out["d_prob"]

        if self.have_prev:
            critic_out = self.critic(
                memoryPrev=self.prev_mem,
                attnPrev=self.prev_attn,
                stateCurr=s_t,
                rewardExt=(self.prev_reward if rewardExt is None else rewardExt),
                nextValue=None,
                done=(self.prev_done if doneFlag is None else doneFlag).float(),
                policyEntropyPrev=self.prev_entropy.squeeze(-1),
                uncertainty=self.prev_unc.squeeze(-1),
                tdErrorPrev=self.prev_td.squeeze(-1),)
        else:
            zeros = torch.zeros(B, device=dev)
            critic_out = self.critic(
                memoryPrev=self.prev_mem,
                attnPrev=self.prev_attn,
                stateCurr=s_t,
                rewardExt=zeros,
                nextValue=None,
                done=zeros,
                policyEntropyPrev=zeros,
                uncertainty=zeros,
                tdErrorPrev=zeros,)

        td_prev = critic_out.tdErrorDe.reshape(B, 1)
        ent_prev = critic_out.entropy.reshape(B, 1)
        unc_prev = critic_out.uncertainty.reshape(B, 1)

        feat_a = self.attn(seq, tdError=td_prev)

        feat_m, mem_recall = self.mem(
            feat_a,
            tdError=td_prev,
            entropy=ent_prev,
            reward=critic_out.rewardUsed.reshape(B, 1).detach(),
            uncertainty=unc_prev,)

        if self.use_planner:

            prior = self.planner.Plan(returnTrajectories=False)

            act_out = self.actor(
                feat_m,
                sample=sampleActions,
                deterministic=deterministicActor,
                prior=prior,mixW=0.30,
                updateHebb=True,
                returnKeys128=True,
                applyConstraints=True)
        else:
            act_out = self.actor(
                feat_m,
                sample=sampleActions,
                deterministic=deterministicActor,
                prior=(self.planner.Plan(returnTrajectories=False) if self.use_planner else None),
                mixW=0.30,
                updateHebb=True,
                returnKeys128=True,
                applyConstraints=True)
            
        if sampleActions:
            keys128 = act_out.get("keys128", act_out.get("keys128_raw"))
            mouse_delta = act_out["mouse"]["a"]
        else:
            keys128 = None
            mouse_delta = act_out["mouse"]["mu"]

        entropy_scalar = act_out["entropy"].view(B, 1)

        self.prev_mem = feat_m.detach()
        self.prev_attn = feat_a.detach()
        self.prev_state = s_t.detach()

        if keys128 is not None:
            self.prev_keys128 = keys128.detach()
        self.prev_mouse = mouse_delta.detach()

        self.prev_reward = (r_t.detach() if rewardExt is None else rewardExt.detach()).reshape(B)
        self.prev_done = (d_t.detach() if doneFlag  is None else doneFlag.detach()).reshape(B)
        self.prev_entropy = entropy_scalar.detach()
        self.prev_unc = critic_out.uncertainty.view(B, 1).detach()
        self.prev_td = critic_out.tdErrorDe.view(B, 1).detach()
        self.have_prev = True

        return {
            "decision": act_out,
            "world": {"state": s_t, "reward": r_t, "done": d_t},
            "critic": critic_out,
            "features": {"perc": feat_p, "attn": feat_a, "mem": feat_m, "mem_recall": mem_recall},}


    @torch.no_grad()
    def ExportBuffers(self) -> Dict[str, Any]:
        return {
            "prev_mem": self.prev_mem,
            "prev_attn": self.prev_attn,
            "prev_state": self.prev_state,
            "prev_keys128": self.prev_keys128,
            "prev_mouse": self.prev_mouse,
            "prev_reward": self.prev_reward,
            "prev_done": self.prev_done,
            "prev_entropy": self.prev_entropy,
            "prev_unc": self.prev_unc,
            "prev_td": self.prev_td,
            "perc_buf": self.perc_buf,
            "tlen": self.tlen,
            "have_prev": self.have_prev,
            "world_h": self.world.ExportState()[0],
            "world_z": self.world.ExportState()[1],}

    @torch.no_grad()
    def ImportBuffers(self, state: Dict[str, Any]):
        device = next(self.parameters()).device
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
        self.prev_mem = state["prev_mem"]
        self.prev_attn = state["prev_attn"]
        self.prev_state = state["prev_state"]
        self.prev_keys128 = state["prev_keys128"]
        self.prev_mouse = state["prev_mouse"]
        self.prev_reward = state["prev_reward"]
        self.prev_done = state["prev_done"]
        self.prev_entropy = state["prev_entropy"]
        self.prev_unc = state["prev_unc"]
        self.prev_td = state["prev_td"]
        self.perc_buf = state["perc_buf"]
        self.tlen = state["tlen"]
        self.have_prev = bool(state["have_prev"])
        self.world.ImportState(state["world_h"], state["world_z"])


class ReplayBuf(collections.deque):
    push = collections.deque.append
    def Sample(self, n): return random.sample(self, n)


class Agent:
    def __init__(self, brain: BrainCore, device="cpu"):
        self.brain = brain.to(device)
        self.device = device

        actor_p = list(brain.perc.parameters()) + list(brain.attn.parameters()) + \
                  list(brain.mem.parameters()) + list(brain.actor.parameters())
        self.opt_a = torch.optim.Adam(actor_p, lr=3e-4)
        self.opt_c = torch.optim.Adam(brain.critic.parameters(), lr=2e-4)
        self.opt_w = torch.optim.Adam(brain.world.parameters(), lr=2e-4)

        self.buf = ReplayBuf(maxlen=100_000)

    def PreprocessRgb(self, frameNp: np.ndarray, outHw: int = 224) -> torch.Tensor:
        if frameNp.ndim == 3 and frameNp.shape[2] == 3:
            img = torch.from_numpy(frameNp).permute(2, 0, 1).float() / 255.0  # C,H,W
            _, H, W = img.shape
            side = min(H, W)
            top = (H - side) // 2
            left = (W - side) // 2
            img = img[:, top:top+side, left:left+side]
            img = F.interpolate(img.unsqueeze(0), (outHw, outHw), mode='bilinear', align_corners=False).squeeze(0)
            return img
        raise ValueError("Expected frame_np as HxWx3 uint8/float array.")

    def Prep(self, imgsNp: Union[np.ndarray, List[np.ndarray]], device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(imgsNp, torch.Tensor):
            imgsNp = [x.cpu().numpy() for x in imgsNp]
        elif isinstance(imgsNp, np.ndarray):
            imgsNp = [imgsNp]
        t = torch.stack([self.PreprocessRgb(i) for i in imgsNp])
        return t.to(device or self.device)

    @torch.no_grad()
    def Act(self, frameNp: np.ndarray, reward: float = 0.0, done: bool = False):
        fr = self.Prep(frameNp)
        out = self.brain.Step(fr, 
                              rewardExt=torch.tensor([reward], 
                              device=self.device),
                              doneFlag=torch.tensor([done], 
                              device=self.device),
                              sampleActions=True, 
                              deterministicActor=False)

        keys128 = out["decision"]["keys128"].squeeze(0).cpu().numpy().astype(np.float32)  # (128,)
        mouse = out["decision"]["mouse"]["a"].squeeze(0).cpu().numpy().astype(np.float32)  # (2,)
        entropy = float(out["decision"]["entropy"].mean().item())

        self.buf.push(fr.squeeze(0).cpu(), keys128.copy(), mouse.copy(), reward, done,
                      out["world"]["state"].squeeze(0).cpu(),
                      out["critic"].value.detach().squeeze(0).cpu())

        return keys128, mouse, entropy


class OfflineGameDataset(Dataset):
    def __init__(self, root: str) -> None:
        p = Path(root)
        self.imgs = sorted((p / "frames").glob("*.png"))
        self.keys = sorted((p / "keys").glob("*.npy"))
        self.mouse = sorted((p / "mouse").glob("*.npy"))
        assert len(self.imgs) == len(self.keys) == len(self.mouse), "frames/keys/mouse 文件数不一致"

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = iio.imread(self.imgs[idx])
        keys = np.load(self.keys[idx]).astype(np.float32)
        mouse = np.load(self.mouse[idx]).astype(np.float32)
        return img, keys, mouse


class TrainingController:
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

    def RequestStop(self):
        with self._lock:
            self.stop_requested = True

    def RequestPause(self):
        with self._lock:
            self.pause_requested = True

    def RequestResume(self):
        with self._lock:
            self.pause_requested = False

    def ShouldStop(self) -> bool:
        with self._lock:
            return self.stop_requested

    def ShouldPause(self) -> bool:
        with self._lock:
            return self.pause_requested


class ManagerFunction:
    def __init__(self, device: Optional[str] = None, path : str = "BrainDeepLearn/Data/training_checkpoint.pth"):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.controller = TrainingController()

        self.training_thread: Optional[threading.Thread] = None
        self.is_training = False

        self.deploy_thread: Optional[threading.Thread] = None
        self.deploying = False

        self.checkpoint_path = path

        self.test = {
            "perception": TestPerceptionMTool(),
            "attention": TestAttentionMTool(),
            "memory": TestMemoryMTool(),
            "decision": TestDecisionMTool(),
            "world": TestWorldMTool(),
            "value": TestValueEstimationMTool(),}

    def StartTraining(self, root: str, epochs: int = 5, batchSize: int = 32, valSplit: float = 0.1, resume: bool = True):
        if self.is_training:
            self.controller.SetStatus("error", "Training is already running")
            return False
        self.is_training = True
        self.training_thread = threading.Thread(target=self.TrainLoop, args=(root, epochs, batchSize, valSplit, resume), daemon=True)
        self.training_thread.start()
        return True

    def StopTraining(self):
        if self.is_training:
            self.controller.RequestStop()
            if self.training_thread is not None:
                self.training_thread.join()
            self.is_training = False
            return True
        return False

    def PauseTraining(self):
        if self.is_training:
            self.controller.RequestPause()
            return True
        return False

    def ResumeTraining(self):
        if self.is_training:
            self.controller.RequestResume()
            return True
        return False

    def GetTrainingStatus(self):
        return self.controller.GetStatus()

    def TrainLoop(self, root: str, epochs: int, batchSize: int, valSplit: float, resume: bool):
        try:
            torch.autograd.set_detect_anomaly(True)

            ds = OfflineGameDataset(root)

            brain = BrainCore(device=self.device, plasticHebbian=True, plasticMeta=True, usePlanner=False)
            agent = Agent(brain, device=self.device)

            start_epoch = 0
            best_val = float("inf")
            train_ds, val_ds = None, None

            if resume and Path(self.checkpoint_path).exists():
                start_epoch, best_val, train_ds, val_ds = self.LoadCheckpoint(brain, agent, ds)

            if train_ds is None:
                n_train = int(len(ds) * (1 - valSplit))
                train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])

            train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=True, num_workers=2, pin_memory=True)
            val_dl = DataLoader(val_ds,   batch_size=batchSize, shuffle=False, num_workers=2)

            bce = nn.BCELoss()
            mse = nn.MSELoss()

            base_codes  = [KEYBOARD_LAYOUT["base_keys"][k] for k in KEYBOARD_LAYOUT["base_keys"]]
            extra_codes = []
            for grp in ["menu_keys", "system_keys", "alpha_keys"]:
                extra_codes += [KEYBOARD_LAYOUT[grp][k] for k in KEYBOARD_LAYOUT[grp]]
            skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in KEYBOARD_LAYOUT["skill_keys"]]
            all_codes   = []
            for grp in KEYBOARD_LAYOUT.values():
                all_codes += list(grp.values())
            max_code   = max(all_codes)
            keys_dim   = max_code + 1 + 2 

            self.controller.SetStatus(
                "training", "Training started",
                epoch=start_epoch, total_epochs=epochs,
                batch=0, total_batches=len(train_dl))

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

                for bi, (imgs_np, keys_np, mouse_np) in enumerate(train_dl, start=1):
                    if self.controller.ShouldStop():
                        break
                    while self.controller.ShouldPause():
                        self.controller.SetStatus("paused", "Training paused")
                        time.sleep(0.2)

                    imgs = agent.Prep(imgs_np).to(self.device)
                    keys = torch.from_numpy(keys_np).float().to(self.device) if isinstance(keys_np, np.ndarray) else torch.from_numpy(np.stack(keys_np)).float().to(self.device)
                    mouse = torch.from_numpy(mouse_np).float().to(self.device) if isinstance(mouse_np, np.ndarray) else torch.from_numpy(np.stack(mouse_np)).float().to(self.device)

                    pred_keys_prob_list = []
                    pred_mouse_list     = []
                    B = imgs.size(0)
                    for i in range(B):
                        brain.world.ResetHidden(batchSize=1, device=self.device)
                        brain.ResetBuffers(B=1, device=self.device)
                        brain.have_prev = False

                        out = brain.Step(
                            imgs[i:i+1],
                            rewardExt=torch.zeros(1, device=self.device),
                            doneFlag=torch.zeros(1, device=self.device),
                            sampleActions=True,
                            deterministicActor=True)

                        kb = out["decision"]["keyboard"]
                        ms = out["decision"]["mouse"]

                        base_p   = torch.sigmoid(kb["base_logits"]).squeeze(0)
                        extra_p  = torch.sigmoid(kb["extra_logits"]).squeeze(0)
                        click_p  = torch.sigmoid(ms["click_logits"]).squeeze(0)
                        skill_p  = torch.softmax(kb["skill_logits"], dim=-1).squeeze(0)
                        mouse_mu = ms["mu"].squeeze(0)

                        vec = torch.zeros(keys_dim, device=self.device)
                        for j, code in enumerate(base_codes):
                            vec[code] = base_p[j]
                        for j, code in enumerate(extra_codes):
                            vec[code] = extra_p[j]
                        for j, code in enumerate(skill_codes):
                            vec[code] = skill_p[j]

                        vec[max_code + 1 : max_code + 3] = click_p

                        pred_keys_prob_list.append(vec)
                        pred_mouse_list.append(mouse_mu)

                    pred_keys_prob = torch.stack(pred_keys_prob_list)
                    pred_mouse     = torch.stack(pred_mouse_list)

                    K = min(pred_keys_prob.size(1), keys.size(1))
                    bc_loss = bce(pred_keys_prob[:, :K], keys[:, :K]) + 0.05 * mse(pred_mouse, mouse)

                    agent.opt_a.zero_grad()
                    try:
                        bc_loss.backward()
                    except Exception as e:
                        traceback.print_exc()
                        raise

                    torch.nn.utils.clip_grad_norm_(
                        list(brain.perc.parameters()) +
                        list(brain.attn.parameters()) +
                        list(brain.mem.parameters()) +
                        list(brain.actor.parameters()),
                        1.0)
                    
                    agent.opt_a.step()

                    epoch_loss += float(bc_loss.item())
                    nb += 1

                    self.controller.SetStatus(
                        "training", "Training...",
                       epoch=ep + 1, total_epochs=epochs,
                        batch=bi, total_batches=len(train_dl),
                        train_loss=float(bc_loss.item()))

                if self.controller.ShouldStop():
                    break

                brain.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    nbv = 0
                    for imgs_np, keys_np, mouse_np in val_dl:
                        imgs  = agent.Prep(imgs_np).to(self.device)
                        keys  = torch.from_numpy(keys_np).float().to(self.device) if isinstance(keys_np, np.ndarray) else torch.from_numpy(np.stack(keys_np)).float().to(self.device)
                        mouse = torch.from_numpy(mouse_np).float().to(self.device) if isinstance(mouse_np, np.ndarray) else torch.from_numpy(np.stack(mouse_np)).float().to(self.device)

                        pred_keys_prob_list = []
                        pred_mouse_list     = []
                        B = imgs.size(0)
                        for i in range(B):
                            brain.world.ResetHidden(batchSize=1, device=self.device)
                            brain.ResetBuffers(B=1, device=self.device)
                            brain.have_prev = False

                            out = brain.Step(
                                imgs[i:i+1],
                                rewardExt=torch.zeros(1, device=self.device),
                                doneFlag=torch.zeros(1, device=self.device),
                                sampleActions=True,
                                deterministicActor=True)

                            kb = out["decision"]["keyboard"]
                            ms = out["decision"]["mouse"]

                            base_p   = torch.sigmoid(kb["base_logits"]).squeeze(0)
                            extra_p  = torch.sigmoid(kb["extra_logits"]).squeeze(0)
                            click_p  = torch.sigmoid(ms["click_logits"]).squeeze(0)
                            skill_p  = torch.softmax(kb["skill_logits"], dim=-1).squeeze(0)
                            mouse_mu = ms["mu"].squeeze(0)

                            vec = torch.zeros(keys_dim, device=self.device)
                            for j, code in enumerate(base_codes):
                                vec[code] = base_p[j]
                            for j, code in enumerate(extra_codes):
                                vec[code] = extra_p[j]
                            for j, code in enumerate(skill_codes):
                                vec[code] = skill_p[j]
                            vec[max_code + 1 : max_code + 3] = click_p

                            pred_keys_prob_list.append(vec)
                            pred_mouse_list.append(mouse_mu)

                        pred_keys_prob = torch.stack(pred_keys_prob_list)
                        pred_mouse = torch.stack(pred_mouse_list)

                        K = min(pred_keys_prob.size(1), keys.size(1))
                        loss = bce(pred_keys_prob[:, :K], keys[:, :K]) + 0.05 * mse(pred_mouse, mouse)
                        val_loss += float(loss.item())
                        nbv += 1

                    avg_train = epoch_loss / max(1, nb)
                    avg_val = val_loss / max(1, nbv)

                best_val = min(best_val, avg_val)
                ckpt = {
                    "epoch": ep + 1,
                    "best_val": best_val,
                    "brain": brain.state_dict(),
                    "opt_a": agent.opt_a.state_dict(),
                    "opt_c": agent.opt_c.state_dict(),
                    "opt_w": agent.opt_w.state_dict(),
                    "train_indices": list(train_ds.indices) if hasattr(train_ds, "indices") else None,
                    "val_indices": list(val_ds.indices) if hasattr(val_ds, "indices") else None,
                    "rng": {
                        "python": random.getstate(),
                        "torch": torch.get_rng_state(),
                        "numpy": np.random.get_state(),},

                    "buffers": brain.ExportBuffers(),}
                
                torch.save(ckpt, self.checkpoint_path)

                self.controller.SetStatus(
                    "training",
                    f"Epoch {ep+1}/{epochs} done | train {avg_train:.4f} | val {avg_val:.4f}",
                    val_loss=avg_val)

            if self.controller.ShouldStop():
                self.controller.SetStatus("stopped", "Training stopped")
            else:
                self.controller.SetStatus("completed", "Training completed")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Training error: {e}", trace=tb)
        finally:
            self.is_training = False

    def LoadCheckpoint(self, brain: BrainCore, agent: Agent, dataset: Dataset):
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        brain.load_state_dict(ckpt["brain"])
        agent.opt_a.load_state_dict(ckpt["opt_a"])
        agent.opt_c.load_state_dict(ckpt["opt_c"])
        agent.opt_w.load_state_dict(ckpt["opt_w"])

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

    def StartDeployment(self, ckptPath: str, cameraIndex: int = 0,
                        fps: int = 30, onlineLearn: bool = False,
                        onlineEvery: float = 2.0, safetyLrScale: float = 0.1):
        if self.deploying:
            return False
        self.deploying = True
        self.deploy_thread = threading.Thread(
            target=self.DeployLoop,
            args=(ckptPath, cameraIndex, fps, onlineLearn, onlineEvery, safetyLrScale),
            daemon=True)
        
        self.deploy_thread.start()
        return True

    def StopDeployment(self):
        if self.deploying:
            self.controller.RequestStop()
            if self.deploy_thread is not None:
                self.deploy_thread.join()
            self.deploying = False
            return True
        return False

    def DeployLoop(self,ckptPath: str,cameraIndex: int,fps: int,onlineLearn: bool,onlineEvery: float,safetyLrScale: float,useHebbian: bool = True,useMeta: bool = True,usePlanner: bool = True,):
        try:
            brain = BrainCore(device=self.device,plasticHebbian=useHebbian,plasticMeta=useMeta,usePlanner=usePlanner,)
            sd = torch.load(ckptPath, map_location=self.device)
            if isinstance(sd, dict):
                if "brain" in sd and isinstance(sd["brain"], dict):
                    state = sd["brain"]
                elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    state = sd["state_dict"]
                else:
                    state = sd 
            else:
                state = sd

            incompat = brain.load_state_dict(state, strict=False)
            if hasattr(incompat, "missing_keys") and (len(incompat.missing_keys) or len(incompat.unexpected_keys)):
                print("[DeployLoop] load_state_dict(strict=False) -> "f"missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}")

            brain.eval()

            agent = Agent(brain, device=self.device)
            base_lr = agent.opt_a.param_groups[0]["lr"]

            if iio is None:
                raise RuntimeError("imageio.v3 cant use")

            base_codes = [KEYBOARD_LAYOUT["base_keys"][k] for k in KEYBOARD_LAYOUT["base_keys"]]
            extra_codes = []
            for grp in ["menu_keys", "system_keys", "alpha_keys"]:
                extra_codes += [KEYBOARD_LAYOUT[grp][k] for k in KEYBOARD_LAYOUT[grp]]
            skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in KEYBOARD_LAYOUT["skill_keys"]]
            all_codes = []
            for grp in KEYBOARD_LAYOUT.values():
                all_codes += list(grp.values())
            max_code = max(all_codes)
            keys_dim = max_code + 1 + 2
            bce = nn.BCELoss()
            mse = nn.MSELoss()

            self.controller.SetStatus("deploying", "Deployment started")
            last_update = time.time()
            with iio.imopen(f"<video{cameraIndex}>", "r") as cam:
                for frame in cam:
                    if self.controller.ShouldStop():
                        break
                    t0 = time.time()

                    keys128, mouse, ent = agent.Act(frame, reward=0.0, done=False)

                    if onlineLearn and (time.time() - last_update >= onlineEvery) and len(agent.buf) >= 32:
                        for g in agent.opt_a.param_groups:
                            g["lr"] = base_lr * safetyLrScale

                        batch = 32
                        samples = list(agent.buf)[-batch:]
                        fr, kb, ms, rw, dn, st, val = map(list, zip(*samples))
                        imgs    = torch.stack(fr).to(self.device)
                        keys_t  = torch.from_numpy(np.stack(kb)).float().to(self.device)
                        mouse_t = torch.from_numpy(np.stack(ms)).float().to(self.device)

                        torch.set_grad_enabled(True)
                        brain.have_prev = False
                        brain.world.ResetHidden(batchSize=imgs.size(0), device=self.device)
                        brain.ResetBuffers(B=imgs.size(0), device=self.device)

                        pred_keys_prob_list, pred_mouse_list = [], []
                        for i in range(imgs.size(0)):
                            out = brain.Step(
                                imgs[i:i+1],
                                rewardExt=torch.zeros(1, device=self.device),
                                doneFlag=torch.zeros(1, device=self.device),
                                sampleActions=True,
                                deterministicActor=True,)
                            
                            kb_o, ms_o = out["decision"]["keyboard"], out["decision"]["mouse"]
                            base_p  = torch.sigmoid(kb_o["base_logits"]).squeeze(0)
                            extra_p = torch.sigmoid(kb_o["extra_logits"]).squeeze(0)
                            click_p = torch.sigmoid(ms_o["click_logits"]).squeeze(0)
                            skill_p = torch.softmax(kb_o["skill_logits"], dim=-1).squeeze(0)
                            mouse_mu = ms_o["mu"].squeeze(0)

                            vec = torch.zeros(keys_dim, device=self.device)
                            for j, code in enumerate(base_codes):  vec[code] = base_p[j]
                            for j, code in enumerate(extra_codes): vec[code] = extra_p[j]
                            for j, code in enumerate(skill_codes): vec[code] = skill_p[j]
                            vec[max_code + 1 : max_code + 3] = click_p

                            pred_keys_prob_list.append(vec)
                            pred_mouse_list.append(mouse_mu)

                        pred_k = torch.stack(pred_keys_prob_list)
                        pred_m = torch.stack(pred_mouse_list)
                        K = min(pred_k.size(1), keys_t.size(1))
                        loss = bce(pred_k[:, :K], keys_t[:, :K]) + 0.05 * mse(pred_m, mouse_t)

                        agent.opt_a.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(brain.perc.parameters()) +
                            list(brain.attn.parameters()) +
                            list(brain.mem.parameters()) +
                            list(brain.actor.parameters()),
                            1.0)
                        
                        agent.opt_a.step()
                        torch.set_grad_enabled(False)

                        for g in agent.opt_a.param_groups:
                            g["lr"] = base_lr
                        last_update = time.time()
                        self.controller.SetStatus("deploying", f"Online update: loss={float(loss.item()):.4f}")

                    latency = time.time() - t0
                    kb_bits = "".join("1" if v > 0.5 else "0" for v in keys128[:8])
                    self.controller.SetStatus("deploying",
                        f"keys:{kb_bits} mouse:dx={mouse[0]:.2f},dy={mouse[1]:.2f} | entropy={ent:.3f} | latency={latency*1000:.1f}ms")
                    time.sleep(max(0.0, 1.0 / max(1, fps) - latency))

            self.controller.SetStatus("stopped", "Deployment stopped")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Deployment error: {e}", trace=tb)
        finally:
            self.deploying = False

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
    

    def TestTrainAndDeploy(
        self,
        *,
        dataRoot: str = "BrainDeepLearn/Data",
        nSamples: int = 64,
        epochs: int = 1,
        batchSize: int = 8,
        val_split: float = 0.2,
        ckpt_path: Optional[str] = None,
        deploy_frames: int = 24,
        fps: int = 12,
        seed: int = 42,
        cleanup: bool = False,
        onlineLearn: bool = False,) -> Dict[str, Any]:

        if iio is None:
            raise RuntimeError("imageio.v3 error")

        rng = np.random.default_rng(seed)
        root = Path(dataRoot)
        if root.exists():
            shutil.rmtree(root)
        (root / "frames").mkdir(parents=True, exist_ok=True)
        (root / "keys").mkdir(parents=True, exist_ok=True)
        (root / "mouse").mkdir(parents=True, exist_ok=True)

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

        H, W = 256, 256
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

        ckpt = ckpt_path or self.checkpoint_path
        print("[SmokeTest] start train...")
        ok = self.StartTraining(
            root=str(root),
            epochs=epochs,
            batchSize=batchSize,
            valSplit=val_split,
            resume=False,)
        
        if not ok:
            raise RuntimeError("StartTraining returns False (training may already be running)")

        while True:
            st = self.GetTrainingStatus()
            print(
                f"[TRAIN] {st['state']} | epoch {st['epoch']}/{st['total_epochs']} "
                f"| batch {st['batch']}/{st['total_batches']} "
                f"| train_loss={st['train_loss']:.4f} | msg={st['message']}")
            
            if st["state"] in ("completed", "stopped", "error"):
                break
            time.sleep(0.5)

        if st["state"] == "error":
            if st.get("trace"):
                print("\n====== TRAIN TRACEBACK ======\n" + st["trace"] + "\n==============================\n")
            raise RuntimeError(f"train error: {st['message']}")
        train_status = st
        print("[SmokeTest] train complete, checkpoint:", ckpt)

        print("[SmokeTest] prepare video header frame...")

        frame_files = sorted((root / "frames").glob("*.png"))[:deploy_frames]
        fake_frames = [iio.imread(str(p)) for p in frame_files]

        class _DummyCam:
            def __init__(self, frames):
                self.frames = frames
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
            def __iter__(self):
                for f in self.frames:
                    yield f

        _orig_imopen = iio.imopen

        def FakeImopen(_, __):
            return _DummyCam(fake_frames)

        iio.imopen = FakeImopen
        try:
            print("[SmokeTest] Start deployment (using fake camera)...")
            ok = self.StartDeployment(
                ckptPath=ckpt,
                cameraIndex=0,
                fps=fps,
                onlineLearn=onlineLearn,
                onlineEvery=2.0,
                safetyLrScale=0.1,)
            
            if not ok:
                raise RuntimeError("StartDeployment returns False (a deployment may already be running)")

            while True:
                st = self.GetTrainingStatus()
                print(f"[DEPLOY] {st['state']} | msg={st['message']}")
                if st["state"] in ("stopped", "error"):
                    break
                time.sleep(0.3)

            if st["state"] == "error":
                raise RuntimeError(f"deploy error: {st['message']}")
            deploy_status = st
            print("[SmokeTest] deploy pass")
        finally:
            iio.imopen = _orig_imopen
            if cleanup:
                try:
                    shutil.rmtree(root, ignore_errors=True)
                except Exception:
                    pass

        return {
            "train_status": train_status,
            "deploy_status": deploy_status,
            "checkpoint": ckpt,
            "data_root": str(root),}
