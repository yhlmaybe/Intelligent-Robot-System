from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from DecisionModule import KEYBOARD_LAYOUT


def ClampLogStd(logstd: torch.Tensor, low: float = -6.0, high: float = 2.0) -> torch.Tensor:
    return torch.clamp(logstd, low, high)

def KLDiagNormal(muQ: torch.Tensor, logstdQ: torch.Tensor, muP: torch.Tensor, logstdP: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(2 * logstdQ)
    var_p = torch.exp(2 * logstdP)
    kl = 0.5 * (((var_q + (muQ - muP) ** 2) / var_p).sum(-1)+ 2 * (logstdP - logstdQ).sum(-1) - muQ.size(-1))
    return kl

def BalancedKL(muQ: torch.Tensor, logstdQ: torch.Tensor, muP: torch.Tensor, logstdP: torch.Tensor, alpha: float = 0.8, freeNats: float = 1.0) -> torch.Tensor:
    mu_p_sg, logstd_p_sg = muP.detach(), logstdP.detach()
    mu_q_sg, logstd_q_sg = muQ.detach(), logstdQ.detach()

    kl_qp = KLDiagNormal(muQ, logstdQ, mu_p_sg, logstd_p_sg)
    kl_pq = KLDiagNormal(mu_q_sg, logstd_q_sg, muP, logstdP)

    kl = alpha * kl_qp + (1.0 - alpha) * kl_pq
    if freeNats and freeNats > 0:
        kl = torch.relu(kl - freeNats)
    return kl  



class ActionEncoder(nn.Module):
    def __init__(self, numDiscrete: int = 128, contDim: int = 2, outDim: int = 128):
        super().__init__()
        self.disc_proj = nn.Linear(numDiscrete, outDim, bias=False)
        self.cont_net = nn.Sequential(
            nn.Linear(contDim, 64), nn.ReLU(), nn.Linear(64, outDim))
        
        self.fuse = nn.Sequential(nn.Linear(outDim * 2, outDim), nn.Tanh())
        nn.init.zeros_(self.disc_proj.weight)

    def forward(self, keysOnehot: torch.Tensor, mouseDelta: Optional[torch.Tensor] = None) -> torch.Tensor:
        disc_vec = self.disc_proj(keysOnehot.float())
        if mouseDelta is None:
            return disc_vec
        cont_vec = self.cont_net(mouseDelta.float())
        return self.fuse(torch.cat([disc_vec, cont_vec], dim=-1))



class RSSMWorldModel(nn.Module):
    def __init__(self,visionDim: int = 512, actionDim: int = 128,deterDim: int = 256,stochDim: int = 32,stateDim: int = 256, useDecoder: bool = True):
        super().__init__()
        self.vision_dim = visionDim
        self.action_dim = actionDim
        self.deter_dim = deterDim
        self.stoch_dim = stochDim
        self.state_dim = stateDim
        self.use_decoder = useDecoder

        self.obs_enc = nn.Sequential(
            nn.LayerNorm(visionDim),
            nn.Linear(visionDim, stateDim),
            nn.GELU(),
            nn.LayerNorm(stateDim),
            nn.Linear(stateDim, stochDim),)

        self.action_encoder = ActionEncoder(numDiscrete=128, contDim=2, outDim=actionDim)
        self.act_proj = nn.Sequential(nn.Linear(actionDim, stochDim), nn.LayerNorm(stochDim), nn.Tanh())

        self.gru = nn.GRUCell(input_size=stochDim + stochDim, hidden_size=deterDim)

        self.prior_net = nn.Sequential(nn.Linear(deterDim, 2 * stochDim))

        self.post_net = nn.Sequential(nn.Linear(deterDim + stochDim, 2 * stochDim))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + stochDim),
            nn.Linear(deterDim + stochDim, stateDim),
            nn.LayerNorm(stateDim),)

        self.rew_head = nn.Sequential(nn.Linear(stateDim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.done_head = nn.Sequential(nn.Linear(stateDim, 256), nn.ReLU(), nn.Linear(256, 1))
        nn.init.zeros_(self.rew_head[-1].bias)
        nn.init.zeros_(self.done_head[-1].bias)

        if useDecoder:
            self.obs_dec = nn.Sequential(
                nn.Linear(stateDim, stateDim), nn.GELU(),
                nn.Linear(stateDim, visionDim))

        self.ResetHidden()

    def ResetHidden(self, batchSize: int = 1, device: torch.device | str = "cpu"):
        device = torch.device(device)
        self._h = torch.zeros(batchSize, self.deter_dim, device=device)
        self._z = torch.zeros(batchSize, self.stoch_dim, device=device)

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._h, self._z

    def ImportState(self, h: torch.Tensor, z: torch.Tensor):
        self._h = h.clone()
        self._z = z.clone()

    @torch.no_grad()
    def StepPriorOnly(self,
                      hPrev: torch.Tensor, zPrev: torch.Tensor,
                      actionEnc: torch.Tensor,
                      sample: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a_t = self.act_proj(actionEnc)  
        h_next = self.gru(torch.cat([zPrev, a_t], dim=-1), hPrev) 

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        if sample:
            eps = torch.randn_like(logstd_p)
            z_next = mu_p + eps * torch.exp(logstd_p)
        else:
            z_next = mu_p 

        s_next = self.state_proj(torch.cat([h_next, z_next], dim=-1)) 
        r_pred = self.rew_head(s_next).squeeze(-1)
        d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)
        return h_next, z_next, s_next, r_pred, d_prob

    def StepPosterior(self,hPrev: torch.Tensor, zPrev: torch.Tensor,visionIn: torch.Tensor, actionEnc: torch.Tensor,deterministicZ: bool = False) -> Dict[str, torch.Tensor]:
        e_t = self.obs_enc(visionIn)        
        a_t = self.act_proj(actionEnc)       

        h_next = self.gru(torch.cat([zPrev, a_t], dim=-1), hPrev)

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = ClampLogStd(logstd_p)

        mu_q, logstd_q = self.post_net(torch.cat([h_next, e_t], dim=-1)).chunk(2, dim=-1)
        logstd_q = ClampLogStd(logstd_q)

        if deterministicZ:
            z_t = mu_q
        else:
            std_q = torch.exp(logstd_q)
            z_t = mu_q + torch.randn_like(std_q) * std_q

        s_next = self.state_proj(torch.cat([h_next, z_t], dim=-1))
        r_pred = self.rew_head(s_next).squeeze(-1)
        d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)

        out = {
            "h_next": h_next, "z_next": z_t, "s_next": s_next,
            "r_pred": r_pred, "d_prob": d_prob,
            "mu_p": mu_p, "logstd_p": logstd_p,
            "mu_q": mu_q, "logstd_q": logstd_q,}

        if self.use_decoder:
            recon = self.obs_dec(s_next)
            out["recon"] = recon
            out["recon_target"] = visionIn

        self._h = h_next.detach()
        self._z = z_t.detach()

        return out

    def ForwardTrainSeq(self,
                        visionSeq: torch.Tensor, 
                        keys128Seq: torch.Tensor, # [B,T,128]
                        mouseSeq: torch.Tensor, # [B,T,2]
                        h0: Optional[torch.Tensor] = None,
                        z0: Optional[torch.Tensor] = None,
                        rewardSeq: Optional[torch.Tensor] = None, # [B,T]
                        doneSeq: Optional[torch.Tensor] = None, # [B,T]
                        alphaKl: float = 0.8,
                        freeNats: float = 1.0,
                        reconCoef: float = 1.0,
                        rewardCoef: float = 1.0,
                        doneCoef: float = 1.0) -> Dict[str, torch.Tensor]:
        
        B, T, _ = visionSeq.shape
        device = visionSeq.device
        if h0 is None: h0 = torch.zeros(B, self.deter_dim, device=device)
        if z0 is None: z0 = torch.zeros(B, self.stoch_dim, device=device)

        h, z = h0, z0
        loss_recon = torch.tensor(0., device=device)
        loss_reward = torch.tensor(0., device=device)
        loss_done = torch.tensor(0., device=device)
        loss_kl = torch.tensor(0., device=device)

        for t in range(T):
            a_enc = self.action_encoder(keys128Seq[:, t], mouseSeq[:, t]) # [B, actionDim]
            a_enc = self.act_proj(a_enc)

            h_next = self.gru(torch.cat([z, a_enc], dim=-1), h)

            mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
            logstd_p = ClampLogStd(logstd_p)

            e_t = self.obs_enc(visionSeq[:, t])
            mu_q, logstd_q = self.post_net(torch.cat([h_next, e_t], dim=-1)).chunk(2, dim=-1)
            logstd_q = ClampLogStd(logstd_q)

            std_q = torch.exp(logstd_q)
            z = mu_q + torch.randn_like(std_q) * std_q

            s = self.state_proj(torch.cat([h_next, z], dim=-1))
            r = self.rew_head(s).squeeze(-1)           
            d_logit = self.done_head(s).squeeze(-1)  
            d = torch.sigmoid(d_logit)

            if self.use_decoder:
                recon = self.obs_dec(s)
                loss_recon = loss_recon + F.mse_loss(recon, visionSeq[:, t], reduction='mean')

            if rewardSeq is None:
                loss_reward = loss_reward + F.mse_loss(r, torch.zeros_like(r), reduction='mean')
            else:
                loss_reward = loss_reward + F.mse_loss(r, rewardSeq[:, t], reduction='mean')

            if doneSeq is None:
                loss_done = loss_done + F.binary_cross_entropy_with_logits(d_logit, torch.zeros_like(d_logit), reduction='mean')
            else:
                loss_done = loss_done + F.binary_cross_entropy_with_logits(d_logit, doneSeq[:, t].float(), reduction='mean')

            kl_t = BalancedKL(mu_q, logstd_q, mu_p, logstd_p, alpha=alphaKl, freeNats=freeNats).mean()
            loss_kl = loss_kl + kl_t

            h = h_next

        T_inv = 1.0 / float(T)
        loss = reconCoef * loss_recon * T_inv + rewardCoef * loss_reward * T_inv + doneCoef * loss_done * T_inv + loss_kl * T_inv

        return {
            "loss": loss,
            "loss_recon": loss_recon * T_inv,
            "loss_reward": loss_reward * T_inv,
            "loss_done": loss_done * T_inv,
            "loss_kl": loss_kl * T_inv,}



class WMAdapterForPlanner(nn.Module):
    def __init__(self,
                 wm: RSSMWorldModel,
                 maxCode: int,
                 baseCodes: List[int],
                 skillCodes: List[int],
                 extraCodes: List[int],     
                 noSkillId: Optional[int],
                 deterministicZ: bool = True):
        super().__init__()
        self.wm = wm
        self.max_code = maxCode
        self.register_buffer("base_codes_buf",  torch.tensor(baseCodes,  dtype=torch.long))
        self.register_buffer("skill_codes_buf", torch.tensor(skillCodes, dtype=torch.long))
        self.register_buffer("extra_codes_buf", torch.tensor(extraCodes, dtype=torch.long))
        self.no_skill_id = noSkillId
        self.deterministic_z = deterministicZ

    @staticmethod
    def MakeKeys(baseAct: torch.Tensor, # [B, n_base]
                 skillIdx: torch.Tensor, # [B]
                 extraAct: torch.Tensor, # [B, n_extra] 
                 clicks: torch.Tensor, # [B, 2]      
                 maxCode: int,
                 baseCodes: torch.Tensor, # [n_base]
                 skillCodes: torch.Tensor, # [n_skill]
                 extraCodes: torch.Tensor, # [n_extra]
                 noSkillId: Optional[int]) -> torch.Tensor:
        B = baseAct.size(0)
        device = baseAct.device

        key_vec = torch.zeros(B, maxCode + 1, device=device)

        for i, code in enumerate(baseCodes.tolist()):
            key_vec[:, code] = baseAct[:, i]

        for i, code in enumerate(extraCodes.tolist()):
            key_vec[:, code] = extraAct[:, i]

        if noSkillId is None:
            chosen = skillIdx
            valid = torch.ones_like(chosen, dtype=torch.bool)
        else:
            valid = (skillIdx != noSkillId)
            chosen = skillIdx.clamp(max=len(skillCodes) - 1)
        if valid.any():
            sel_codes = skillCodes[chosen[valid]]
            key_vec[valid, sel_codes] = 1.0

        keys128 = torch.cat([key_vec, clicks], dim=-1)
        return keys128

    @torch.no_grad()
    def Step(self,
             aMouse: torch.Tensor, # [B,2]
             aSkill: torch.Tensor, # [B]
             aBase: Optional[torch.Tensor]  = None, # [B, n_base]
             aExtra: Optional[torch.Tensor] = None, # [B, n_extra]  
             aClicks: Optional[torch.Tensor] = None, # [B, 2]      
             h0: Optional[torch.Tensor] = None, # [B, deterDim]
             z0: Optional[torch.Tensor] = None # [B, stochDim]
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = aMouse.size(0)
        device = aMouse.device

        if aBase is None:
            aBase = torch.zeros(B, self.base_codes_buf.numel(), device=device)
        if aExtra is None:
            aExtra = torch.zeros(B, self.extra_codes_buf.numel(), device=device)  
        if aClicks is None:
            aClicks = torch.zeros(B, 2, device=device) 

        keys128 = self.MakeKeys(
            aBase, aSkill, aExtra, aClicks, self.max_code,
            self.base_codes_buf.to(device),
            self.skill_codes_buf.to(device),
            self.extra_codes_buf.to(device),
            self.no_skill_id)
        
        a_enc = self.wm.action_encoder(keys128, aMouse) # [B, actionDim]

        if h0 is None or z0 is None:
            h_prev, z_prev = self.wm.ExportState()
            if h_prev is None or h_prev.size(0) != B or h_prev.device != device:
                h_prev = torch.zeros(B, self.wm.deter_dim, device=device)
                z_prev = torch.zeros(B, self.wm.stoch_dim, device=device)
        else:
            h_prev, z_prev = h0, z0

        h1, z1, s1, r, d = self.wm.StepPriorOnly(h_prev, z_prev, a_enc, sample=not self.deterministic_z)

        return s1, r, d






class TestWorldMTool:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(0)

        self.base_codes = [KEYBOARD_LAYOUT["base_keys"][k] for k in KEYBOARD_LAYOUT["base_keys"]]
        self.skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in KEYBOARD_LAYOUT["skill_keys"]]
        self.extra_codes = []
        for grp in ["menu_keys", "system_keys", "alpha_keys"]:
            self.extra_codes += [KEYBOARD_LAYOUT[grp][k] for k in KEYBOARD_LAYOUT[grp]]

        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        self.max_code = max(all_codes)  

        self.wm = RSSMWorldModel(visionDim=512, actionDim=128, deterDim=256, stochDim=32, stateDim=256, useDecoder=True).to(self.device)
        self.wm.ResetHidden(batchSize=4, device=self.device)

    def TestActionEncoder(self):
        try:
            enc = ActionEncoder(numDiscrete=128, contDim=2, outDim=128).to(self.device)
            B = 3
            keys128 = torch.zeros(B, 128, device=self.device)
            keys128[:, 17] = 1.0  
            keys128[:, 57] = 1.0  
            mouse = torch.randn(B, 2, device=self.device)

            y1 = enc(keys128, mouse)
            y2 = enc(keys128, None)

            ok = (y1.shape == (B, 128)) and (y2.shape == (B, 128))
            if ok:
                print("ActionEncoder test passed.")
                return True
            else:
                print(f"ActionEncoder output shape mismatch: {y1.shape}, {y2.shape}")
                return False
        except Exception as e:
            print("ActionEncoder test crash:", type(e).__name__, e)
            return False

    def TestRSSMStepPosterior(self):
        try:
            B = 4
            vision = torch.randn(B, 512, device=self.device)
            keys128 = torch.zeros(B, 128, device=self.device)
            keys128[:, 17] = 1.0; keys128[:, 57] = 1.0
            mouse = torch.randn(B, 2, device=self.device)
            a_enc = self.wm.action_encoder(keys128, mouse)

            h0, z0 = self.wm.ExportState()
            out = self.wm.StepPosterior(h0, z0, vision, a_enc, deterministicZ=False)

            ok_shapes = (
                out["h_next"].shape == (B, self.wm.deter_dim) and
                out["z_next"].shape == (B, self.wm.stoch_dim) and
                out["s_next"].shape == (B, self.wm.state_dim) and
                out["r_pred"].shape == (B,) and
                out["d_prob"].shape == (B,))

            h1, z1 = self.wm.ExportState()
            changed = (not torch.allclose(h0, h1)) or (not torch.allclose(z0, z1))

            in_range = (out["d_prob"].min() >= 0.0) and (out["d_prob"].max() <= 1.0)

            if ok_shapes and changed and in_range:
                print("RSSM StepPosterior test passed.")
                return True
            else:
                print(f"RSSM StepPosterior failed. shapes_ok={ok_shapes}, state_changed={changed}, d_in_range={in_range}")
                return False
        except Exception as e:
            print("RSSM StepPosterior test crash:", type(e).__name__, e)
            return False

    def TestRSSMStepPriorOnly(self):
        try:
            B = 4
            keys128 = torch.zeros(B, 128, device=self.device); keys128[:, 30] = 1.0 
            mouse = torch.randn(B, 2, device=self.device)
            a_enc = self.wm.action_encoder(keys128, mouse)

            h0, z0 = self.wm.ExportState()
            h_before, z_before = h0.clone(), z0.clone()

            h1, z1, s1, r, d = self.wm.StepPriorOnly(h0, z0, a_enc, sample=False)

            ok_shapes = (
                h1.shape == (B, self.wm.deter_dim) and
                z1.shape == (B, self.wm.stoch_dim) and
                s1.shape == (B, self.wm.state_dim) and
                r.shape == (B,) and
                d.shape == (B,))

            hin, zin = self.wm.ExportState()
            not_written = torch.allclose(hin, h_before) and torch.allclose(zin, z_before)

            if ok_shapes and not_written:
                print("RSSM StepPriorOnly test passed.")
                return True
            else:
                print(f"RSSM StepPriorOnly failed. shapes_ok={ok_shapes}, state_not_written={not_written}")
                return False
        except Exception as e:
            print("RSSM StepPriorOnly test crash:", type(e).__name__, e)
            return False

    def TestForwardTrainSeq(self):
        try:
            B, T = 2, 3
            vision_seq = torch.randn(B, T, 512, device=self.device)
            keys128_seq = torch.zeros(B, T, 128, device=self.device)
            keys128_seq[:, :, 17] = 1.0 
            mouse_seq = torch.randn(B, T, 2, device=self.device)

            out = self.wm.ForwardTrainSeq(
                visionSeq=vision_seq,
                keys128Seq=keys128_seq,
                mouseSeq=mouse_seq,
                rewardSeq=None,
                doneSeq=None,
                alphaKl=0.8,
                freeNats=1.0,
                reconCoef=1.0,
                rewardCoef=1.0,
                doneCoef=1.0,)
            
            loss = out["loss"]
            if not torch.isfinite(loss):
                print("ForwardTrainSeq loss is not finite.")
                return False

            loss.backward()
            print("RSSM ForwardTrainSeq test passed. loss =", float(loss.item()))
            return True
        except Exception as e:
            print("RSSM ForwardTrainSeq test crash:", type(e).__name__, e)
            return False


    def TestWMAdapterForPlanner(self):
        try:
            adapter = WMAdapterForPlanner(
                wm=self.wm,
                maxCode=self.max_code,
                baseCodes=self.base_codes,
                skillCodes=self.skill_codes,
                extraCodes=self.extra_codes,
                noSkillId=None,          
                deterministicZ=True).to(self.device)

            B = 3
            a_mouse = torch.randn(B, 2, device=self.device)
            a_base  = (torch.rand(B, len(self.base_codes),  device=self.device) > 0.5).float()
            a_extra = (torch.rand(B, len(self.extra_codes), device=self.device) > 0.5).float()
            a_click = (torch.rand(B, 2, device=self.device) > 0.5).float()
            a_skill = torch.randint(low=0, high=len(self.skill_codes), size=(B,), device=self.device)

            s1, r, d = adapter.Step(aMouse=a_mouse, aSkill=a_skill, aBase=a_base, aExtra=a_extra, aClicks=a_click)

            ok_shapes = (s1.shape == (B, self.wm.state_dim) and r.shape == (B,) and d.shape == (B,))
            if ok_shapes:
                print("WMAdapterForPlanner.Step test passed.")
                return True
            else:
                print(f"WMAdapterForPlanner.Step output shape mismatch: s1={s1.shape}, r={r.shape}, d={d.shape}")
                return False
        except Exception as e:
            print("WMAdapterForPlanner.Step test crash:", type(e).__name__, e)
            return False

    def RunAll(self):
        results = []
        results.append(self.TestActionEncoder())
        results.append(self.TestRSSMStepPosterior())
        results.append(self.TestRSSMStepPriorOnly())
        results.append(self.TestForwardTrainSeq())
        results.append(self.TestWMAdapterForPlanner())

        passed = sum(1 for x in results if x)
        total = len(results)
        print(f"\nWorldModel test summary: {passed}/{total} passed.")
        return all(results)
