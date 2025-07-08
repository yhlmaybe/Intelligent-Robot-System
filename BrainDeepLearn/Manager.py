from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
from PerceptionModule import PerceiveExtractor, PerceiveExtractorMetaWrapper
from AttentionModule import AttentionExtractor
from MemoryModule import MemoryExtractor
from DecisionModule import DecisionExtractor, KEYBOARD_LAYOUT
from WorldModule import WorldModelExtractor, ActionEncoder, WorldModelSeqRNN
from ValueEstimationModule import ValueEstimationExtractor
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random, collections
import pathlib, imageio.v3 as iio
import time
import torch.utils
import torch.utils.data


class BrainCore(nn.Module):
    SEQ_LEN = 16
    def __init__(self, plastic_hebbian: bool = True, plastic_meta: bool = True):
        super().__init__()

        base = PerceiveExtractor(useHebbian=plastic_hebbian)
        self.perc = (PerceiveExtractorMetaWrapper(base) if plastic_meta else base)
        if hasattr(self.perc, "plastic_on"):
            self.perc.plastic_on = plastic_meta

        self.attn = AttentionExtractor(hebbianRate=0.01 if plastic_hebbian else 0.0)

        self.mem  = MemoryExtractor(hebbAlpha=0.15 if plastic_hebbian else 0.0, useMeta = plastic_meta)

        self.actor = DecisionExtractor()
        for m in self.actor.modules():
            if hasattr(m, "hebbian_on"):
                m.hebbian_on = plastic_hebbian

        self.world  = WorldModelSeqRNN()
        self.critic = ValueEstimationExtractor()
        self.act_enc= ActionEncoder()
        self.ResetBuffers()

    def ResetBuffers(self, B: int = 1, device: Optional[torch.device] = None):
        device = device or next(self.parameters()).device         
        z = lambda *s, dtype=torch.float32: torch.zeros(B, *s, device=device, dtype=dtype)

        self.prev_mem = z(768) 
        self.prev_attn = z(512)        
        self.prev_state = z(256)    
        self.prev_act = z(128)        

        self.prev_rew = z(1)       
        self.prev_done = z(1)
        self.prev_ent = z(1)
        self.prev_unc = z(1)

        self.perc_buf = z(self.SEQ_LEN, 512) 
        self.ptr = 0
        self.have_prev = False

    def Write(self, feat: torch.Tensor):
        if feat.size(0) != self.perc_buf.size(0):
            self.perc_buf = self.perc_buf.new_zeros(feat.size(0), self.SEQ_LEN, 512)
            self.ptr = 0
            
        self.perc_buf[:, self.ptr] = feat
        self.ptr = (self.ptr + 1) % self.SEQ_LEN

    def Seq(self):
        idx=[(self.ptr+i)%self.SEQ_LEN for i in range(self.SEQ_LEN)]
        return self.perc_buf[:,idx]  # [B,S,512]

    def Step(self, frame, reward, done):
        B, dev = frame.size(0), frame.device
        if not self.have_prev or self.prev_mem.size(0) != B:
            self.ResetBuffers(B, dev)

        feat_p = self.perc(frame) # [B, 512]
        self.Write(feat_p)
        seq = self.Seq()

        state = self.world(seq.detach(), self.prev_act.detach())

        if self.have_prev:
            cf = self.critic(
                self.prev_mem,
                self.prev_attn,
                self.prev_state,
                reward=self.prev_rew,
                nextValue=None,
                done=self.prev_done,
                policyEntropy=self.prev_ent)
            
            td_prev = cf.tdErrorDe
            ent_prev = cf.entropy
            unc_prev = cf.uncertainty
        else:                             
            td_prev = ent_prev = unc_prev = torch.zeros(
                B, 1, device=dev)

        feat_a = self.attn(seq, tdError=td_prev)
        feat_m, _ = self.mem(
            feat_a,
            tdError=td_prev,
            entropy=ent_prev,
            reward=self.prev_rew,
            uncertainty=unc_prev,)

        out = self.actor(feat_m)

        kb_vec = self.actor.kb.ToKeyboardVector(out["keyboard"],False).to(dev)
        act_emb= self.act_enc(kb_vec, out["mouse"])
        
        value,_,_ = self.critic(feat_m, feat_a, state)

        self.prev_mem,self.prev_attn,self.prev_state = feat_m.detach(), feat_a.detach(), state.detach()
        self.prev_act,self.prev_rew,self.prev_done = act_emb.detach(), reward.detach(), done.float()

        self.prev_ent  = out["entropy"].detach()
        self.have_prev=True
        return out, state.detach(), value.detach()

    def Imagine(self, featM0: torch.Tensor, state0: torch.Tensor, horizon: int = 5):
        device = state0.device
        s, f = state0, featM0               
        mem_state = self.mem.GetState()     

        states, values = [], []
        tmp_state = mem_state              

        for _ in range(horizon):
            act = self.actor(f)
            kb = self.actor.kb.ToKeyboardVector(act["keyboard"], False).to(device)
            a_emb = self.act_enc(kb, act["mouse"])

            s = self.world.ImagineStep(a_emb)
            states.append(s)

            if not hasattr(self, "state2attn"):
                self.state2attn = nn.Linear(self.world.state_dim, self.attn.output_dim, bias=False).to(device)
            attn_proxy = self.state2attn(s)

            f, tmp_state = self.mem.Step(attn_proxy, tmp_state, tdError=None)

            v, _, _ = self.critic(f, attn_proxy, s)
            values.append(v.squeeze(-1))

        self.mem.SetState(mem_state)

        return torch.stack(states, dim=1), torch.stack(values, dim=1)

class ReplayBuf(collections.deque):
    push=collections.deque.append
    def Sample(self,n): return random.sample(self,n)

class Agent:
    def __init__(self, brain:BrainCore, device="cpu", horizon=5, online_imagine=True):
        
        self.brain=brain.to(device); self.device=device
        self.horizon=horizon; self.online_imagine=online_imagine

        actor_p=list(brain.perc.parameters()) + list(brain.attn.parameters()) + list(brain.mem.parameters()) + list(brain.actor.parameters())
        self.opt_a=torch.optim.Adam(actor_p, 3e-4)
        self.opt_c=torch.optim.Adam(brain.critic.parameters(), 2e-4)
        self.opt_w=torch.optim.Adam(brain.world.parameters(),  2e-4)
        self.buf=ReplayBuf(maxlen=100_000)

    def PreprocessRgb(self, frameNp: np.ndarray, outHw: int = 224) -> torch.Tensor:
        if frameNp.shape[2] == 3 and frameNp[..., 0].mean() > frameNp[..., 2].mean():
            frameNp = frameNp[..., ::-1] # BGR→RGB

        img = torch.from_numpy(frameNp).permute(2, 0, 1).float() / 255.  # C×H×W

        _, H, W = img.shape
        side = min(H, W)
        top  = (H - side) // 2
        left = (W - side) // 2
        img = img[:, top:top+side, left:left+side]      # C×side×side

        img = F.interpolate(img.unsqueeze(0), (outHw, outHw), mode='bilinear', align_corners=False).squeeze(0)
        return img   # 3×224×224

    def Prep(self, imgs_np: Union[np.ndarray, List[np.ndarray]],device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(imgs_np, np.ndarray):
            imgs_np = [imgs_np]
        tensor = torch.stack([self.PreprocessRgb(i) for i in imgs_np])
        return tensor.to(device or self.device)

    def Act(self, frame_np, reward: float = 0.0, done: bool = False):
        fr = self.Prep(frame_np)                   
        out, s, v = self.brain.Step(fr,torch.tensor([reward], device=self.device),torch.tensor([done],   device=self.device))

        kb = self.brain.actor.kb.ToKeyboardVector(out["keyboard"], False).cpu().numpy()[0].astype(np.float32)  

        mouse = out["mouse"].squeeze(0).cpu().numpy().astype(np.float32)

        self.buf.push(fr.squeeze(0).cpu(), kb.copy(), mouse.copy(),reward, done, s.cpu(), v.cpu())

        return kb, mouse, out["entropy"].item()
    


    @staticmethod
    def Lam(rw: torch.Tensor, v: torch.Tensor, y: float = 0.99, lmbda: float = 0.95):
        """
        rw, v: [B, T]  —— r_t  &  V_t
        return: [B, T] —— GAE/λ-return
        """
        B, T = v.shape
        ret = torch.zeros_like(v)
        fut = torch.zeros(B, device=v.device)

        for t in reversed(range(T)):
            fut = rw[:, t] + v[:, t] + y * lmbda * fut
            ret[:, t] = fut

        return ret

    def Update(self, batch: int = 64):
        if len(self.buf) < batch or not self.online_imagine:
            return

        fr, kb, ms, rw, dn, st, _ = map(list, zip(*self.buf.Sample(batch)))

        fr = torch.stack(fr).to(self.device)
        kb = torch.from_numpy(np.stack(kb)).float().to(self.device)
        ms = torch.from_numpy(np.stack(ms)).float().to(self.device)
        st = torch.stack(st).to(self.device)
        rw = torch.tensor(rw, device=self.device).unsqueeze(1).float()
        dn = torch.tensor(dn, device=self.device).unsqueeze(1).float()

        with torch.no_grad():                  
            s_enc = self.brain.perc(fr)
        act_emb = self.brain.act_enc(kb, ms)

        pred_s, pred_r, pred_d = self.brain.world.ForwardTrain(s_enc.detach(), act_emb, return_predictions=True)
        
        w_loss = (F.mse_loss(pred_s, st) + F.mse_loss(pred_r, rw) + 0.1 * F.binary_cross_entropy_with_logits(pred_d, dn))
        
        self.opt_w.zero_grad()
        w_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.world.parameters(), 1.0)
        self.opt_w.step()

        with torch.no_grad():
            feat_a = self.brain.attn(s_enc.unsqueeze(1).repeat(1, self.brain.SEQ_LEN, 1))[:, -1]
            
            feat_m, _ = self.brain.mem(feat_a)

            _, im_v = self.brain.Imagine(featM0=feat_m, state0=st, horizon=self.horizon)
            
        target = self.Lam(rw.repeat(1, self.horizon), im_v)[:, 0] # [B]

        pred, _, _ = self.brain.critic(st, st, st) # [B,1]
        c_loss = F.mse_loss(pred.squeeze(), target.detach())

        self.opt_c.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.critic.parameters(), 1.0)
        self.opt_c.step()

        feat_m_detached = feat_m.detach()
        out_pol = self.brain.actor(feat_m_detached)
        logits = out_pol["logits_kb"] # [B, 8]

        dist = torch.distributions.Bernoulli(logits=logits)
        logp = dist.log_prob(kb[:, :8]).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)

        adv = (target - pred.detach().squeeze())
        a_loss = -(logp * adv).mean() - 0.01 * entropy.mean()

        self.opt_a.zero_grad()
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.brain.perc.parameters()) +
            list(self.brain.attn.parameters()) +
            list(self.brain.mem.parameters()) +
            list(self.brain.actor.parameters()), 1.0)
        self.opt_a.step()

        return {"w": w_loss.item(),"c": c_loss.item(),"a": a_loss.item(),
    }


class OfflineGameDataset(Dataset):
    def __init__(self, root: str) -> None:
        p = pathlib.Path(root)
        self.imgs: List[pathlib.Path] = sorted((p / "frames").glob("*.png"))
        self.keys: List[pathlib.Path] = sorted((p / "keys").glob("*.npy"))
        self.mouse: List[pathlib.Path] = sorted((p / "mouse").glob("*.npy"))
        assert len(self.imgs) == len(self.keys) == len(self.mouse)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img: np.ndarray = iio.imread(self.imgs[idx])  # (H, W, C) uint8
        keys: np.ndarray = np.load(self.keys[idx]).astype(np.float32)  # (104,)
        mouse: np.ndarray = np.load(self.mouse[idx]).astype(np.float32)  # (2,)
        return img, keys, mouse

class ManagerFunction():
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.original_lr: Optional[float] = None

    def Train(self, root: str, epochs: int = 5, batchSize: int = 32, val_split: float = 0.1, imagine_horizon: int = 5) -> None:
        """
        root: Root directory of the dataset
        epochs: Number of training rounds
        batch_size: Batch size
        val_split: Validation set ratio (0-1)
        imagine_horizon: Imagine and deduce the step size
        """
        ds = OfflineGameDataset(root)
        n_train = int(len(ds) * (1 - val_split))
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds)-n_train])
        
        train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=True, num_workers=4, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=batchSize, shuffle=False, num_workers=2)

        brain = BrainCore(plastic_hebbian=True, plastic_meta=True).to(self.device)
        agent = Agent(brain, device=self.device, horizon=imagine_horizon, online_imagine=True)
        
        self.original_lr = agent.opt_a.param_groups[0]["lr"]
        
        bce = torch.nn.BCELoss()
        mse = torch.nn.MSELoss()
        best_val = float('inf')

        for ep in range(epochs):
            brain.train()
            loss_sum = 0.0
            n = 0
            
            for imgs_np, keys_np, mouse_np in train_dl:
                imgs = agent.Prep(imgs_np).to(self.device) 
                keys = torch.from_numpy(keys_np).to(self.device)
                mouse = torch.from_numpy(mouse_np).to(self.device)
                
                brain.have_prev = False
                
                pred_k, pred_m, states, values = [], [], [], []
                for i in range(imgs.size(0)):
                    frame = imgs[i].unsqueeze(0)
                    out, state, value = brain.Step(
                        frame, 
                        torch.zeros(1, device=self.device), 
                        torch.zeros(1, device=self.device))
                    
                    pred_k.append(brain.actor.kb.ToKeyboardVector(out["keyboard"], False))
                    pred_m.append(out["mouse"].squeeze(0))
                    states.append(state.squeeze(0))
                    values.append(value.squeeze(0))
                
                pred_k = torch.stack(pred_k)
                pred_m = torch.stack(pred_m)
                states = torch.stack(states)
                values = torch.stack(values)
                
                bc_loss = bce(pred_k, keys) + 0.05 * mse(pred_m, mouse)
                
                agent.opt_a.zero_grad()
                bc_loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                agent.opt_a.step()
                
                for i in range(len(imgs_np)):
                    agent.buf.push(
                        imgs_np[i], 
                        keys_np[i], 
                        mouse_np[i],
                        0.0,  # reward
                        False, # done
                        states[i].detach().cpu(), 
                        values[i].detach().cpu())
                
                agent.Update(batch=batchSize)
                
                loss_sum += bc_loss.item()
                n += 1
            
            avg_loss = loss_sum / n
            print(f"[Epoch {ep+1}/{epochs}] Train Loss: {avg_loss:.4f}")
            
            brain.eval()
            val_loss = 0.0
            m = 0
            
            with torch.no_grad():
                for imgs_np, keys_np, mouse_np in val_dl:
                    imgs = agent.Prep(imgs_np).to(self.device)
                    keys = torch.from_numpy(keys_np).to(self.device)
                    mouse = torch.from_numpy(mouse_np).to(self.device)
                    
                    brain.have_prev = False
                    
                    pred_k, pred_m = [], []
                    for i in range(imgs.size(0)):
                        frame = imgs[i].unsqueeze(0)
                        out, _, _ = brain.Step(
                            frame, 
                            torch.zeros(1, device=self.device), 
                            torch.zeros(1, device=self.device)
                        )
                        pred_k.append(brain.actor.kb.ToKeyboardVector(out["keyboard"], False))
                        pred_m.append(out["mouse"].squeeze(0))
                    
                    pred_k = torch.stack(pred_k)
                    pred_m = torch.stack(pred_m)
                    
                    val_loss += (bce(pred_k, keys) + 0.05 * mse(pred_m, mouse)).item()
                    m += 1
            
            avg_val = val_loss / m
            print(f"Val Loss: {avg_val:.4f}")
            
            if avg_val < best_val:
                best_val = avg_val
                torch.save(brain.state_dict(), "brain_game_best.pth")
                print("保存最佳模型")
        
        torch.save(brain.state_dict(), "brain_game_final.pth")
        print("训练完成")

    
    def Deploy(self, ckpt: str, camera_index: int = 0, fps: int = 30, online_learn: bool = False, online_every: float = 2.0, safety_lr: float = 0.1) -> None:

        brain = BrainCore(plastic_hebbian=False, plastic_meta=False).to(self.device)
        brain.load_state_dict(torch.load(ckpt, map_location=self.device))
        agent = Agent(brain, device=self.device, online_imagine=online_learn)
        
        if self.original_lr is None:
            self.original_lr = agent.opt_a.param_groups[0]["lr"]
        
        last_update = time.time()
        frame_count = 0
        total_processing_time = 0
        
        try:
            with iio.imopen(f"<video{camera_index}>", "r") as cam:
                cam_shape = cam.properties().shape
                print(f"摄像头已启动 | 分辨率: {cam_shape[1]}x{cam_shape[0]} | 目标FPS: {fps}")
                
                for frame in cam:
                    start_time = time.time()
                    frame_count += 1
                    
                    frame_t = agent.PreprocessRgb(frame).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        out, state, value = brain.Step(
                            frame_t,
                            torch.zeros(1, device=self.device),
                            torch.zeros(1, device=self.device)
                        )
                        kb_vec = brain.actor.kb.ToKeyboardVector(out["keyboard"], False).cpu().numpy()
                        mouse = out["mouse"].squeeze(0).cpu().numpy()
                    
                    if online_learn:
                        agent.buf.push(
                            frame,
                            kb_vec.copy(),
                            mouse.copy(),
                            0.0,  # reward
                            False, # done
                            state.squeeze(0).cpu(), 
                            value.cpu())
                        
                        current_time = time.time()
                        if current_time - last_update >= online_every:
                            for g in agent.opt_a.param_groups:
                                g["lr"] = self.original_lr * safety_lr
                            
                            agent.Update(batch=32)

                            for g in agent.opt_a.param_groups:
                                g["lr"] = self.original_lr     

                            last_update = current_time
                            print(f"• 在线更新 | 学习率: {self.original_lr * safety_lr:.6f}")
                    
                    process_time = time.time() - start_time
                    total_processing_time += process_time
                    
                    if frame_count % fps == 0:
                        avg_process_time = total_processing_time / fps
                        real_fps = 1.0 / avg_process_time if avg_process_time > 0 else fps
                        
                        keys_status = "".join("1" if k > 0.5 else "0" for k in kb_vec[:8])
                        print(f"[帧 {frame_count}] FPS≈{real_fps:.1f} "
                              f"按键: {keys_status} "
                              f"鼠标: Δx={mouse[0]:.2f}, Δy={mouse[1]:.2f}")
                        
                        total_processing_time = 0
                    
                    target_delay = 1.0 / fps
                    sleep_time = max(0, target_delay - process_time)
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("操作已停止")
        finally:
            if online_learn and hasattr(agent, 'buf') and len(agent.buf) > 0:
                torch.save(brain.state_dict(), "brain_game_online.pth")
                print("✓ 在线学习模型已保存")

