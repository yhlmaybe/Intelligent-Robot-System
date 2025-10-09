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


def ToDevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x


class BrainCore(nn.Module):
    SEQ_LEN = 16
    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        plasticHebbian: bool = True,
        plasticMeta: bool = True,
        usePlanner: bool = True,):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.perc = PerceiveExtractor(useHebbian=plasticHebbian)
        if hasattr(self.perc, "plastic_on"):
            self.perc.plastic_on = plasticMeta

        self.attn = AttentionExtractor(hebbianRate=(0.01 if plasticHebbian else 0.0))
        self.mem = MemoryExtractor(hebbAlpha=(0.15 if plasticHebbian else 0.0), useMeta=plasticMeta)

        # 决策器输入维度对齐 Memory 输出：768
        self.actor = DecisionExtractor(stateDim=768, includeNoSkill=True, useHebbOnline=plasticHebbian)

        # 世界模型：视觉维度使用感知输出的 1024，内部状态 256
        self.world = RSSMWorldModel(
            visionDim=1024, actionDim=128, deterDim=256, stochDim=32, stateDim=256, useDecoder=True
        )
        self.world.ResetHidden(batchSize=1, device=self.device)

        # 价值估计：接收 prev memory(768) + prev attn(1024) + 当前世界状态(256)
        self.critic = ValueEstimationExtractor(memoryDim=768, attnDim=1024, stateDim=256)

        # 规划器（CEM），用于生成 prior 混合到 actor
        self.use_planner = usePlanner
        self.planner = DecisionPlannerExtractor().BuildPlanner(
            worldModel=self.world,
            KEYBOARD_LAYOUT=KEYBOARD_LAYOUT,
            includeNoSkill=True,
            horizon=5, N=64, elite=8, iters=3,
            gamma=0.99, temperature=1.0, momentum=0.15,
            laplace=1.0, minVar=1e-4, epsBern=1e-4
        )

        # === 键盘向量长度：max_code+1(离散按键) + 2(左右键点击) ===
        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        self.max_code = max(all_codes)
        self.keyvec_dim = (self.max_code + 1) + 2  # 106

        # 运行缓存
        self._buf_B = 0
        self.ResetBuffers(B=1, device=self.device)
        self.to(self.device)

    # ----------------- 缓存管理 -----------------
    @torch.no_grad()
    def ResetBuffers(self, B: int = 1, device: Optional[torch.device] = None):
        device = device or self.device

        def z(*s, dtype=torch.float32):
            return torch.zeros(B, *s, device=device, dtype=dtype)

        # 上一时刻的 B 与 C
        self.prev_mem = z(768)
        self.prev_attn = z(1024)
        # 世界隐藏状态的语义投影
        self.prev_state = z(256)

        # 上一时刻的动作（供世界模型使用）
        self.prev_key_vec = z(self.keyvec_dim)  # [B,106]
        self.prev_mouse = z(2)

        # 教师信号（传给价值估计/注意力/记忆）
        self.prev_reward = z()         # [B]
        self.prev_done = z()           # [B]
        self.prev_entropy = z()        # [B,1]   来自 actor 的策略熵聚合指标
        self.prev_unc = z()            # [B,1]   来自 critic 的不确定性
        self.prev_td = z()             # [B,1]   来自 critic 的 TD 误差

        # 感知帧序列（供注意力）
        self.perc_buf = z(self.SEQ_LEN, 1024)  # [B,T,1024]
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
        # shift-left
        self.perc_buf[:, :-1] = self.perc_buf[:, 1:].clone()
        self.perc_buf[:, -1] = feat_p
        self.tlen = torch.clamp(self.tlen + 1, max=self.SEQ_LEN)

    # ----------------- 核心一步 -----------------
    def Step(
        self,
        frame: torch.Tensor,                      # [B,3,H,W]
        rewardExt: Optional[torch.Tensor] = None, # [B] 或 None
        doneFlag: Optional[torch.Tensor] = None,  # [B] 或 None
        *,
        sampleActions: bool = True,               # True=随机; False=确定性
        deterministicActor: bool = False,
    ) -> Dict[str, Any]:
        B, dev = frame.size(0), frame.device
        self.EnsureB(B, dev)

        # 1) 感知 A
        feat_p = self.perc(frame)                 # [B,1024]
        self.PushPerc(feat_p)
        L = int(max(1, min(self.SEQ_LEN, int(self.tlen.min().item()))))
        seq = self.perc_buf[:, -L:]               # [B,L,1024]

        # 2) 世界模型（使用上一步动作）
        a_enc_prev = self.world.action_encoder(self.prev_key_vec, self.prev_mouse)
        hPrev, zPrev = self.world.ExportState()
        w_out = self.world.StepPosterior(hPrev, zPrev, visionIn=feat_p, actionEnc=a_enc_prev, sample=False)
        s_t = w_out["s_next"]     # [B,256]
        r_t = w_out["r_pred"]     # [B]
        d_t = w_out["d_prob"]     # [B]

        # 3) 价值估计（E + 上一帧 B/C）
        if self.have_prev:
            rew_prev = (rewardExt if rewardExt is not None else self.prev_reward).view(B)
            done_prev = (doneFlag if doneFlag is not None else self.prev_done).view(B)
            pe_prev = self.prev_entropy.view(B)          # [B]
            unc_teacher = self.prev_unc.view(B)          # [B]
            td_prev = self.prev_td.view(B)               # [B]
            mem_prev = self.prev_mem
            attn_prev = self.prev_attn
        else:
            zeros = torch.zeros(B, device=dev)
            rew_prev = zeros
            done_prev = zeros
            pe_prev = zeros
            unc_teacher = zeros
            td_prev = zeros
            mem_prev = torch.zeros_like(self.prev_mem)
            attn_prev = torch.zeros_like(self.prev_attn)

        critic_out = self.critic(
            memory=mem_prev,
            attn=attn_prev,
            state=s_t,
            rewardExt=rew_prev,
            policyEntropyPrev=pe_prev,
            uncertaintyTeacher=unc_teacher,
            tdErrorPrev=td_prev,
            done=done_prev,
            edgeIndex=None,
            edgeWeight=None,
        )
        # 提取指导信号
        td_sig = critic_out.tdError.view(B, 1)                      # [B,1]
        ent_sig = critic_out.rComps.get("entropy", torch.zeros(B, device=dev)).view(B, 1)
        unc_sig = critic_out.uncertainty.view(B, 1)

        # 4) 注意力 B（受 TD 影响）
        feat_b = self.attn(seq, tdError=td_sig)                     # [B,1024]

        # 5) 记忆 C（受 TD/熵/奖励/不确定性影响）
        # 这里作为奖励信号，使用内在奖励 rInt（与外部奖励无关，更稳定）
        feat_c, mem_recall = self.mem(
            feat_b,
            tdError=td_sig,
            entropy=ent_sig,
            reward=critic_out.rInt.view(B, 1).detach(),
            uncertainty=unc_sig,
        )  # feat_c: [B,768]

        # 6) 决策 D（可混合规划器的 prior）
        final_det = bool(deterministicActor or (not sampleActions))
        prior = None
        if self.use_planner and (self.planner is not None):
            with torch.no_grad():
                prior = self.planner.Plan(returnTrajectories=False)

        act_out = self.actor(
            stateFeat=feat_c,
            sample=True,                      # 始终 True，以生成 key_vec / mouse.a
            deterministic=final_det,          # 由 sampleActions + deterministicActor 共同决定
            prior=prior, mixW=0.30,
            updateHebb=True,
            returnKeysVec=True,
            applyConstraints=True,
        )

        key_vec = act_out["key_vec"]              # [B,106]
        mouse_a = act_out["mouse"]["a"]           # [B,2]
        entropy_scalar = act_out["entropy"].view(B, 1)

        # 7) 写回 “上一时刻” 缓存
        self.prev_mem = feat_c.detach()
        self.prev_attn = feat_b.detach()
        self.prev_state = s_t.detach()
        self.prev_key_vec = key_vec.detach()
        self.prev_mouse = mouse_a.detach()

        # 这里做一个“上一时刻奖励/终止”的定义：
        # 优先使用外部 reward/done，否则用世界模型回归的 r_t/d_t
        self.prev_reward = (rewardExt.detach() if rewardExt is not None else r_t.detach()).view(B)
        self.prev_done = (doneFlag.detach() if doneFlag is not None else d_t.detach()).view(B)
        self.prev_entropy = entropy_scalar.detach()
        self.prev_unc = unc_sig.detach()
        self.prev_td = td_sig.detach()
        self.have_prev = True

        return {
            "decision": act_out,
            "world": {"state": s_t, "reward": r_t, "done": d_t},
            "critic": critic_out,
            "features": {"perc": feat_p, "attn": feat_b, "mem": feat_c, "mem_recall": mem_recall},
        }

    # ----------------- 运行态保存/恢复 -----------------
    @torch.no_grad()
    def ExportBuffers(self) -> Dict[str, Any]:
        h, z = self.world.ExportState()
        return {
            "prev_mem": self.prev_mem,
            "prev_attn": self.prev_attn,
            "prev_state": self.prev_state,
            "prev_key_vec": self.prev_key_vec,
            "prev_mouse": self.prev_mouse,
            "prev_reward": self.prev_reward,
            "prev_done": self.prev_done,
            "prev_entropy": self.prev_entropy,
            "prev_unc": self.prev_unc,
            "prev_td": self.prev_td,
            "perc_buf": self.perc_buf,
            "tlen": self.tlen,
            "have_prev": torch.tensor([int(self.have_prev)], device=self.prev_mem.device),
            "world_h": h, "world_z": z,
        }

    @torch.no_grad()
    def ImportBuffers(self, state: Dict[str, Any]):
        device = next(self.parameters()).device
        # 搬运到当前设备
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
        self.prev_mem = state["prev_mem"]
        self.prev_attn = state["prev_attn"]
        self.prev_state = state["prev_state"]
        self.prev_key_vec = state["prev_key_vec"]
        self.prev_mouse = state["prev_mouse"]
        self.prev_reward = state["prev_reward"]
        self.prev_done = state["prev_done"]
        self.prev_entropy = state["prev_entropy"]
        self.prev_unc = state["prev_unc"]
        self.prev_td = state["prev_td"]
        self.perc_buf = state["perc_buf"]
        self.tlen = state["tlen"]
        self.have_prev = bool(int(state["have_prev"].view(-1)[0].item()))
        self.world.ImportState(state["world_h"], state["world_z"])


class Agent:
    """
    负责数据预处理、一步执行、保存/恢复（含优化器与随机种子）。
    训练循环你可以在外层组织，此处仅提供基本接口。
    """
    def __init__(self, brain: BrainCore, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self.brain = brain.to(self.device)

        # 优化器划分（你也可以按需调整分组/学习率）
        actor_params = (
            list(self.brain.perc.parameters())
            + list(self.brain.attn.parameters())
            + list(self.brain.mem.parameters())
            + list(self.brain.actor.parameters())
        )
        self.opt_actor = torch.optim.Adam(actor_params, lr=3e-4)
        self.opt_critic = torch.optim.Adam(self.brain.critic.parameters(), lr=2e-4)
        self.opt_world = torch.optim.Adam(self.brain.world.parameters(), lr=2e-4)

    # --------- 简单图像预处理 ---------
    def _preprocess_rgb(self, frame_np: np.ndarray, out_hw: int = 224) -> torch.Tensor:
        if frame_np.ndim == 3 and frame_np.shape[2] == 3:
            img = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0  # C,H,W
            _, H, W = img.shape
            side = min(H, W)
            top = (H - side) // 2
            left = (W - side) // 2
            img = img[:, top:top + side, left:left + side]
            img = F.interpolate(img.unsqueeze(0), (out_hw, out_hw), mode='bilinear', align_corners=False).squeeze(0)
            return img
        raise ValueError("Expected frame_np as HxWx3 array.")

    def prep(self, imgs: Union[np.ndarray, List[np.ndarray]], device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(imgs, torch.Tensor):
            return imgs.to(device or self.device)
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        t = torch.stack([self._preprocess_rgb(i) for i in imgs], dim=0)  # [B,3,H,W]
        return t.to(device or self.device)

    @torch.no_grad()
    def act(
        self,
        frame_np: np.ndarray,
        reward: float = 0.0,
        done: bool = False,
        *,
        sample_actions: bool = True,
        deterministic_actor: bool = False,
    ):
        frame = self.prep(frame_np)  # [1,3,H,W]
        out = self.brain.Step(
            frame,
            rewardExt=torch.tensor([reward], device=self.device, dtype=torch.float32),
            doneFlag=torch.tensor([float(done)], device=self.device, dtype=torch.float32),
            sampleActions=sample_actions,
            deterministicActor=deterministic_actor,
        )
        key_vec = out["decision"]["key_vec"].squeeze(0).cpu().numpy().astype(np.float32)  # (106,)
        mouse = out["decision"]["mouse"]["a"].squeeze(0).cpu().numpy().astype(np.float32) # (2,)
        entropy = float(out["decision"]["entropy"].mean().item())
        return key_vec, mouse, entropy

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "brain": self.brain.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_world": self.opt_world.state_dict(),
            "buffers": self.brain.ExportBuffers(),
            "rng_py": random.getstate(),
            "rng_np": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["rng_cuda_all"] = torch.cuda.get_rng_state_all()
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True, map_location: Optional[Union[str, torch.device]] = None):
        payload = torch.load(path, map_location=map_location or self.device)
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
            pass



class ReplayBuf(collections.deque):
    push = collections.deque.append
    def Sample(self, n): return random.sample(self, n)



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
