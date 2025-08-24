from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


KEYBOARD_LAYOUT: Dict[str, Dict[str, int]] = {
    "base_keys": {
        "W": 17, "A": 30, "S": 31, "D": 32,
        "Space": 57, "Shift": 42, "Ctrl": 29, "Esc": 1
    },
    "skill_keys": {
        "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
        "Q": 16, "E": 18, "R": 19, "F": 33,
        "F1": 59, "F2": 60, "F3": 61, "F4": 62,
        "G": 34, "T": 20, "V": 47, "B": 48
    },
    "menu_keys": {
        "Tab": 15, "I": 23, "M": 50, "J": 36, "K": 37,
        "L": 38, "U": 22, "O": 24, "P": 25,
        "F5": 63, "F6": 64,
        "Insert": 110, "Delete": 111, "Home": 102, "End": 107
    },
    "system_keys": {
        "Enter": 28, "Backspace": 14, "CapsLock": 58, "Win": 125, "Alt": 56
    },
    "alpha_keys": {
        "a": 70, "b": 71, "c": 72, "d": 73, "e": 74, "f": 75, "g": 76, "h": 77,
        "i": 78, "j": 79, "k": 80, "l": 81, "m": 82, "n": 83, "o": 84, "p": 85,
        "q": 86, "r": 87, "s": 88, "t": 89, "u": 90, "v": 91, "w": 92, "x": 93,
        "y": 94, "z": 95
    }
}


def ClampLogstd(logstd: torch.Tensor, low: float = -5.0, high: float = 2.0) -> torch.Tensor:
    return torch.clamp(logstd, low, high)

def StableLogProbBernoulli(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return (actions * (-F.softplus(-logits)) + (1.0 - actions) * (-F.softplus(logits))).sum(-1)

def EntropyBernoulliFromLogits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    return -(p * (p.clamp_min(1e-8)).log() + (1 - p) * ((1 - p).clamp_min(1e-8)).log()).sum(-1)

def MixLogits(base: torch.Tensor, prior: Optional[torch.Tensor], w: float) -> torch.Tensor:
    if prior is None:
        return base
    return (1.0 - w) * base + w * prior

def MixGauss(mu: torch.Tensor, logstd: torch.Tensor, priorMu: Optional[torch.Tensor], priorVar: Optional[torch.Tensor], w: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if (priorMu is None) or (priorVar is None):
        return mu, logstd
    var = torch.exp(2.0 * logstd)
    var_mix = (1.0 - w) * var + w * priorVar
    mu_mix = (1.0 - w) * mu + w * priorMu
    logstd_mix = 0.5 * torch.log(var_mix.clamp_min(1e-10))
    return mu_mix, logstd_mix


class HebbianPlasticityLayer(nn.Module):
    def __init__(self, inDim: int, outDim: int, rate: float = 1e-3, decay: float = 0.995, maxRowNorm: float = 2.0):
        super().__init__()
        self.rate = rate
        self.decay = decay
        self.max_row_norm = maxRowNorm
        self.base = nn.Parameter(torch.randn(outDim, inDim) * 0.02)
        self.register_buffer("hebb", torch.zeros(outDim, inDim))

    @torch.no_grad()
    def Project(self):
        w = self.hebb
        row_norm = w.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        scale = (self.max_row_norm / row_norm).clamp_max(1.0)
        self.hebb.mul_(scale)

    def forward(self, x: torch.Tensor, update: bool = False):
        w = self.base + self.hebb
        out = F.linear(x, w)
        if update and not x.requires_grad:
            pre = x.detach()
            post = out.detach()
            B = pre.size(0)
            delta = self.rate * (torch.einsum("bo,bi->oi", post, pre) / max(1, B) - (post.pow(2).mean(0, keepdim=True).t() * self.hebb))
            self.hebb.mul_(self.decay).add_(delta)
            self.Project()
        return out


class MouseActor(nn.Module):
    def __init__(self, inDim: int = 256, hidden: int = 256, actDim: int = 2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(inDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        
        self.mu_head = nn.Linear(hidden, actDim)
        self.logstd_head = nn.Linear(hidden, actDim)
        self.click_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def Params(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(feat)
        mu = self.mu_head(h)
        logstd = ClampLogstd(self.logstd_head(h))
        click_logits = self.click_head(h)
        return mu, logstd, click_logits


class KeyboardActor(nn.Module):
    def __init__(self, inDim: int = 256,baseKeyNames: Optional[List[str]] = None,skillNames: Optional[List[str]] = None,includeNoSkill: bool = True,hidden: int = 256):
        super().__init__()
        baseKeyNames = baseKeyNames or list(KEYBOARD_LAYOUT["base_keys"].keys())
        skillNames = skillNames or list(KEYBOARD_LAYOUT["skill_keys"].keys())

        extra_codes, extra_map = [], []
        for grp in ["menu_keys", "system_keys", "alpha_keys"]:
            for name, code in KEYBOARD_LAYOUT[grp].items():
                extra_codes.append(code); extra_map.append((grp, name))
        self.extra_codes = extra_codes
        self.extra_map = extra_map

        self.base_key_names = baseKeyNames
        self.skill_names = skillNames
        self.include_no_skill = includeNoSkill
        self.num_base = len(baseKeyNames)
        self.num_skill = len(skillNames) + (1 if includeNoSkill else 0)
        self.num_extra = len(self.extra_codes)

        self.backbone = nn.Sequential(
            nn.Linear(inDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        
        self.base_head = nn.Linear(hidden, self.num_base)   
        self.skill_head = nn.Linear(hidden, self.num_skill)  
        self.extra_head = nn.Linear(hidden, self.num_extra) 

    def Logits(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(feat)
        return self.base_head(h), self.skill_head(h), self.extra_head(h)


class OptionPolicy(nn.Module):
    def __init__(self, zDim: int = 256, numOptions: int = 8, psiDim: int = 32, hidden: int = 256):
        super().__init__()
        self.num_options = numOptions
        self.psi_dim = psiDim
        self.enc = nn.Sequential(nn.Linear(zDim, hidden), nn.ReLU())
        self.pi_o = nn.Linear(hidden, numOptions)
        self.psi_head = nn.Linear(hidden, psiDim)
        self.beta_head = nn.Sequential(nn.Linear(zDim + numOptions, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor, prevOOnehot: Optional[torch.Tensor] = None):
        h = self.enc(z)
        logits_o = self.pi_o(h)
        psi = self.psi_head(h)
        beta = None
        if prevOOnehot is not None:
            beta = torch.sigmoid(self.beta_head(torch.cat([z, prevOOnehot], dim=-1)))
        return logits_o, psi, beta


class DecisionExtractor(nn.Module):
    def __init__(
        self,
        stateDim: int = 768,
        includeNoSkill: bool = True,
        useHebbOnline: bool = False,
        optionNum: int = 8,
        psiDim: int = 32,
        *,
        entropyWeights: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        logstdBounds: Tuple[float, float] = (-5.0, 2.0),):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(stateDim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU())
        
        self.hebb = HebbianPlasticityLayer(512, 512)
        self.to_z = nn.Linear(512, 256)
        self.use_hebb_online = useHebbOnline

        base_names  = list(KEYBOARD_LAYOUT["base_keys"].keys())
        skill_names = list(KEYBOARD_LAYOUT["skill_keys"].keys())
        self.base_codes = [KEYBOARD_LAYOUT["base_keys"][k] for k in base_names]
        self.skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in skill_names]
        self.extra_groups = ["menu_keys", "system_keys", "alpha_keys"]
        self.extra_codes, self.extra_names = [], []
        for g in self.extra_groups:
            for name, code in KEYBOARD_LAYOUT[g].items():
                self.extra_codes.append(code)
                self.extra_names.append((g, name))
        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        self.max_code = max(all_codes)
        self.no_skill_id = len(self.skill_codes) if includeNoSkill else None

        self.keyboard = KeyboardActor(256, base_names, skill_names, includeNoSkill=includeNoSkill)
        self.mouse = MouseActor(256)
        self.option = OptionPolicy(zDim=256, numOptions=optionNum, psiDim=psiDim)

        self.register_buffer("entropy_w", torch.tensor(entropyWeights, dtype=torch.float32))
        self.logstd_low = float(logstdBounds[0])
        self.logstd_high = float(logstdBounds[1])

    def Encode(self, stateFeat: torch.Tensor, updateHebb: bool) -> torch.Tensor:
        x = self.feature_net(stateFeat)
        x = self.hebb(x, update=(self.use_hebb_online and updateHebb))
        z = F.relu(self.to_z(x))
        return z

    def ToKeys128(self, baseAct: torch.Tensor, extraAct: torch.Tensor, skillIdx: torch.Tensor, clicks: torch.Tensor) -> torch.Tensor:
        B = baseAct.size(0); device = baseAct.device
        vec = torch.zeros(B, self.max_code + 1 + 2, device=device)

        for i, code in enumerate(self.base_codes):
            vec[:, code] = baseAct[:, i]

        for i, code in enumerate(self.extra_codes):
            vec[:, code] = extraAct[:, i]

        if self.no_skill_id is None:
            chosen = skillIdx; valid = torch.ones_like(chosen, dtype=torch.bool)
        else:
            valid = (skillIdx != self.no_skill_id)
            chosen = skillIdx.clamp(max=len(self.skill_codes) - 1)
        if valid.any():
            sel_codes = torch.tensor(self.skill_codes, device=device)[chosen[valid]]
            vec[valid, sel_codes] = 1.0

        vec[:, self.max_code + 1:self.max_code + 3] = clicks
        return vec

    @staticmethod
    def ApplyConstraints(vec128: torch.Tensor):
        max_scan = vec128.size(1) - 2
        W = KEYBOARD_LAYOUT["base_keys"]["W"]
        S = KEYBOARD_LAYOUT["base_keys"]["S"]
        A = KEYBOARD_LAYOUT["base_keys"]["A"]
        D = KEYBOARD_LAYOUT["base_keys"]["D"]
        conflict = (vec128[:, W] > 0.5) & (vec128[:, S] > 0.5)
        vec128[conflict, S] = 0.0
        conflict = (vec128[:, A] > 0.5) & (vec128[:, D] > 0.5)
        vec128[conflict, D] = 0.0
        for b in range(vec128.size(0)):
            on = (vec128[b, :max_scan] > 0.5).nonzero(as_tuple=False).squeeze(1)
            if on.numel() > 6:
                vec128[b, on[6:]] = 0.0

    def EntropyComponents(
        self,
        baseLogits: torch.Tensor,
        extraLogits: torch.Tensor,
        skillLogits: torch.Tensor,
        logstd: torch.Tensor,) -> Dict[str, torch.Tensor]:
        
        ent_base = EntropyBernoulliFromLogits(baseLogits)
        ent_extra = EntropyBernoulliFromLogits(extraLogits)
        ent_skill = torch.distributions.Categorical(logits=skillLogits).entropy()
        ent_mouse = (0.5 * (1.0 + math.log(2 * math.pi)) + logstd).sum(-1)

        n_base = max(1, baseLogits.size(-1))
        n_extra = max(1, extraLogits.size(-1))
        n_skill = max(2, skillLogits.size(-1)) 
        base_norm  = ent_base / n_base
        extra_norm = ent_extra / n_extra
        skill_norm = ent_skill / math.log(n_skill)

        l, h = self.logstd_low, self.logstd_high
        mouse_norm = ((logstd.clamp(l, h) - l) / (h - l)).mean(-1)

        return {
            "ent_base": ent_base, "ent_extra": ent_extra,
            "ent_skill": ent_skill, "ent_mouse": ent_mouse,
            "base_norm": base_norm, "extra_norm": extra_norm,
            "skill_norm": skill_norm, "mouse_norm": mouse_norm,}

    def AggregateEntropy(self, comps: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = self.entropy_w 
        return (
            w[0] * comps["base_norm"]
          + w[1] * comps["extra_norm"]
          + w[2] * comps["skill_norm"]
          + w[3] * comps["mouse_norm"])

    def forward(
        self,
        stateFeat: torch.Tensor,                            
        *,
        sample: bool = False,                               
        deterministic: bool = False,                           
        prevOptionOnehot: Optional[torch.Tensor] = None,     
        prior: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,  
        mixW: float = 0.25,                              
        updateHebb: bool = False,                      
        returnKeys128: bool = True,                        
        applyConstraints: bool = True) -> Dict[str, torch.Tensor]:
        B = stateFeat.size(0)

        z = self.Encode(stateFeat, updateHebb=updateHebb)

        option_logits, psi, beta = self.option(z, prevOptionOnehot)

        base_logits, skill_logits, extra_logits = self.keyboard.Logits(z)
        mu, logstd, click_logits = self.mouse.Params(z)

        if prior is not None:
            base_logits = MixLogits(base_logits,  prior.get("base",  {}).get("logits", None),  mixW)
            extra_logits = MixLogits(extra_logits, prior.get("extra", {}).get("logits", None),  mixW)
            skill_logits = MixLogits(skill_logits, prior.get("skill", {}).get("logits", None),  mixW)
            mu, logstd = MixGauss(mu, logstd, prior.get("mouse", {}).get("mu",  None), prior.get("mouse", {}).get("var", None), mixW)
            click_logits = MixLogits(click_logits, prior.get("click", {}).get("logits", None), mixW)

        comps = self.EntropyComponents(base_logits, extra_logits, skill_logits, logstd)
        entropy_scalar = self.AggregateEntropy(comps)  

        out: Dict[str, any] = {
            "z": z,
            "entropy": entropy_scalar, 
            "option": {"logits": option_logits, "psi": psi, "beta": beta},
            "keyboard": {"base_logits":  base_logits,"skill_logits": skill_logits,"extra_logits": extra_logits,},
            "mouse": {"mu": mu, "logstd": logstd, "click_logits": click_logits},
            "entropy_components": {
                "base": comps["ent_base"], "extra": comps["ent_extra"],
                "skill": comps["ent_skill"], "mouse": comps["ent_mouse"],
                "base_norm": comps["base_norm"], "extra_norm": comps["extra_norm"],
                "skill_norm": comps["skill_norm"], "mouse_norm": comps["mouse_norm"],},}

        if sample:
            if deterministic:
                base_act = (torch.sigmoid(base_logits)  > 0.5).float()
                extra_act = (torch.sigmoid(extra_logits) > 0.5).float()
                skill_idx = torch.argmax(skill_logits, dim=-1)
                mouse_a = mu
                clicks = (torch.sigmoid(click_logits) > 0.5).float()

                logp_base = StableLogProbBernoulli(base_logits, base_act)
                logp_extra = StableLogProbBernoulli(extra_logits, extra_act)
                logp_skill = torch.distributions.Categorical(logits=skill_logits).log_prob(skill_idx)
                logp_mouse = -0.5 * (((mouse_a - mu) / torch.exp(logstd)).pow(2) + 2 * logstd + math.log(2 * math.pi)).sum(-1)
            else:
                base_prob = torch.sigmoid(base_logits)
                extra_prob = torch.sigmoid(extra_logits)
                base_act = torch.bernoulli(base_prob)
                extra_act = torch.bernoulli(extra_prob)
                skill_idx = torch.distributions.Categorical(logits=skill_logits).sample()

                std = torch.exp(logstd)
                eps = torch.randn_like(std)
                mouse_a = mu + eps * std

                click_prob = torch.sigmoid(click_logits)
                clicks = torch.bernoulli(click_prob)

                logp_base = StableLogProbBernoulli(base_logits, base_act)
                logp_extra = StableLogProbBernoulli(extra_logits, extra_act)
                logp_skill = torch.distributions.Categorical(logits=skill_logits).log_prob(skill_idx)
                logp_mouse = -0.5 * (((mouse_a - mu) / std).pow(2) + 2 * logstd + math.log(2 * math.pi)).sum(-1)

            out["keyboard"].update({
                "base_act": base_act, "extra_act": extra_act, "skill_idx": skill_idx,
                "logp_base": logp_base, "logp_extra": logp_extra, "logp_skill": logp_skill,})
            
            out["mouse"].update({"a": mouse_a, "logp": logp_mouse, "click_sample": clicks})

            if returnKeys128:
                keys128_raw = self.ToKeys128(base_act, extra_act, skill_idx, clicks)
                out["keys128_raw"] = keys128_raw
                if applyConstraints:
                    keys128 = keys128_raw.clone()
                    self.ApplyConstraints(keys128)
                    out["keys128"] = keys128
        return out

    def ResetHebbianMemory(self, value: float = 0.0):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, HebbianPlasticityLayer):
                    m.hebb.fill_(value)



class CEMPlanner(nn.Module):
    def __init__(self,worldModel: nn.Module,baseCodes: List[int],skillCodes: List[int],extraCodes: List[int],maxCode: int,hasNoSkill: bool = True,    horizon: int = 5, N: int = 64,
                 elite: int = 8,iters: int = 3,gamma: float = 0.99,temperature: float = 1.0,momentum: float = 0.15,laplace: float = 1.0,minVar: float = 1e-4,epsBern: float = 1e-4):
        super().__init__()
        self.wm = worldModel
        self.horizon = int(horizon)
        self.N = int(N)
        self.elite = int(elite)
        self.iters = int(iters)
        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.laplace = float(laplace)
        self.min_var = float(minVar)
        self.eps_bern = float(epsBern)
        self.has_no_skill = bool(hasNoSkill)

        self.max_code = int(maxCode)
        self.register_buffer("base_codes_buf",  torch.tensor(baseCodes,  dtype=torch.long))
        self.register_buffer("skill_codes_buf", torch.tensor(skillCodes, dtype=torch.long))
        self.register_buffer("extra_codes_buf", torch.tensor(extraCodes, dtype=torch.long))

        self.n_base  = self.base_codes_buf.numel()
        self.n_skill = self.skill_codes_buf.numel() + (1 if self.has_no_skill else 0)  
        self.n_extra = self.extra_codes_buf.numel()

    @staticmethod
    def LogitsFromProb(p: torch.Tensor, eps: float) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return p.log() - (1.0 - p).log()

    @staticmethod
    def AssembleKeys128(baseAct: torch.Tensor,extraAct: torch.Tensor,skillIdx: torch.Tensor,clickAct: torch.Tensor,baseCodes: torch.Tensor,
                          skillCodes: torch.Tensor,extraCodes: torch.Tensor,maxCode: int,hasNoSkill: bool) -> torch.Tensor:
        B = baseAct.size(0)
        device = baseAct.device
        keys = torch.zeros(B, maxCode + 1, device=device)

        for i, code in enumerate(baseCodes.tolist()):
            keys[:, code] = baseAct[:, i]

        for i, code in enumerate(extraCodes.tolist()):
            keys[:, code] = extraAct[:, i]

        if hasNoSkill:
            no_skill_id = skillCodes.numel()  
            valid = (skillIdx != no_skill_id)
            chosen = skillIdx.clamp_max(skillCodes.numel() - 1)
            if valid.any():
                sel_codes = skillCodes[chosen[valid]]
                keys[valid, sel_codes] = 1.0
        else:
            sel_codes = skillCodes[skillIdx]
            keys[torch.arange(B, device=device), sel_codes] = 1.0

        keys128 = torch.cat([keys, clickAct], dim=-1)  
        return keys128

    @torch.no_grad()
    def Plan(self,
             mouseMu: Optional[torch.Tensor] = None,         
             mouseLogstd: Optional[torch.Tensor] = None,     
             skillLogits: Optional[torch.Tensor] = None,     
             baseLogits: Optional[torch.Tensor] = None,       
             extraLogits: Optional[torch.Tensor] = None,    
             clickLogits: Optional[torch.Tensor] = None,     
             h0: Optional[torch.Tensor] = None,
             z0: Optional[torch.Tensor] = None,
             returnTrajectories: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:

        with torch.no_grad():
            if mouseMu is not None:
                B = mouseMu.size(0)
                device = mouseMu.device
            elif skillLogits is not None:
                B = skillLogits.size(0)
                device = skillLogits.device
            elif baseLogits is not None:
                B = baseLogits.size(0); device = baseLogits.device
            else:
                h_cur, z_cur = self.wm.ExportState()
                if h_cur is None:
                    raise ValueError("batch size/device cannot be inferred; Please provide at least one distributed parameter or (h0,z0)")
                B = h_cur.size(0); device = h_cur.device

        if mouseMu is None:
            mouseMu = torch.zeros(B, 2, device=device)
        if mouseLogstd is None:
            mouseLogstd = torch.zeros(B, 2, device=device) 
        if skillLogits is None:
            skillLogits = torch.zeros(B, self.n_skill, device=device)
        if baseLogits is None:
            baseLogits = torch.zeros(B, self.n_base, device=device)
        if extraLogits is None:
            extraLogits = torch.zeros(B, self.n_extra, device=device)
        if clickLogits is None:
            clickLogits = torch.zeros(B, 2, device=device)

        H, N, E = self.horizon, self.N, self.elite
        mu_t = mouseMu.unsqueeze(0).repeat(H, 1, 1)
        std_t = torch.exp(mouseLogstd).unsqueeze(0).repeat(H, 1, 1)
        logits_s = skillLogits.unsqueeze(0).repeat(H, 1, 1)
        logits_b = baseLogits.unsqueeze(0).repeat(H, 1, 1)
        logits_e = extraLogits.unsqueeze(0).repeat(H, 1, 1)
        logits_c = clickLogits.unsqueeze(0).repeat(H, 1, 1)

        h_prev, z_prev = (h0, z0)
        if h_prev is None or z_prev is None:
            h_prev, z_prev = self.wm.ExportState()
            if h_prev is None or h_prev.size(0) != B or h_prev.device != device:
                h_prev = torch.zeros(B, self.wm.deter_dim, device=device)
                z_prev = torch.zeros(B, self.wm.stoch_dim, device=device)

        for _ in range(self.iters):
            eps = torch.randn(H, B, N, 2, device=device)
            mouse_seq = mu_t.unsqueeze(2) + eps * std_t.unsqueeze(2)

            skill_seq = []
            for t in range(H):
                dist = torch.distributions.Categorical(logits=logits_s[t])
                idx = dist.sample((N,)).transpose(0, 1).contiguous()
                skill_seq.append(idx)
            skill_seq = torch.stack(skill_seq, dim=0)

            pb = torch.sigmoid(logits_b)
            pe = torch.sigmoid(logits_e)
            pc = torch.sigmoid(logits_c)
            base_seq  = (torch.rand(H, B, N, self.n_base, device=device) < pb.unsqueeze(2)).float()
            extra_seq = (torch.rand(H, B, N, self.n_extra, device=device) < pe.unsqueeze(2)).float()
            click_seq = (torch.rand(H, B, N, 2,device=device) < pc.unsqueeze(2)).float()

            h = h_prev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
            z = z_prev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()

            score = torch.zeros(B, N, device=device)
            cont  = torch.ones(B, N, device=device)

            for t in range(H):
                a_mouse_t = mouse_seq[t].reshape(B * N, 2)
                a_skill_t = skill_seq[t].reshape(B * N)
                a_base_t  = base_seq[t].reshape(B * N, self.n_base)
                a_extra_t = extra_seq[t].reshape(B * N, self.n_extra)
                a_click_t = click_seq[t].reshape(B * N, 2)

                keys128 = self.AssembleKeys128(
                    a_base_t, a_extra_t, a_skill_t, a_click_t,
                    self.base_codes_buf, self.skill_codes_buf, self.extra_codes_buf,
                    self.max_code, self.has_no_skill)

                a_enc = self.wm.action_encoder(keys128, a_mouse_t)
                try:
                    h, z, s_next, r_t, d_t = self.wm.StepPriorOnly(h, z, a_enc, sample=False)
                except TypeError:
                    h, z, s_next, r_t, d_t = self.wm.StepPriorOnly(h, z, a_enc)

                r_t = r_t.view(B, N)
                d_t = d_t.view(B, N)

                score = score + cont * (self.gamma ** t) * r_t
                cont  = cont * (1.0 - d_t)


            topk = torch.topk(score, k=E, dim=1).indices
            elite_scores = score.gather(1, topk)
            if self.temperature <= 0:
                w = torch.ones_like(elite_scores) / E
            else:
                w = F.softmax(elite_scores / self.temperature, dim=1)
            w_exp = w.unsqueeze(-1)
            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, E)

            for t in range(H):
                elite_mouse_t = mouse_seq[t][b_idx, topk, :]
                mu_new  = (w_exp * elite_mouse_t).sum(dim=1)
                diff    = elite_mouse_t - mu_new.unsqueeze(1)
                var_new = (w_exp * (diff * diff)).sum(dim=1).clamp_min(self.min_var)
                std_new = var_new.sqrt()
                mu_t[t]  = self.momentum * mu_t[t]  + (1 - self.momentum) * mu_new
                std_t[t] = self.momentum * std_t[t] + (1 - self.momentum) * std_new


                elite_skill_t = skill_seq[t][b_idx, topk]
                counts = torch.zeros(B, self.n_skill, device=device)
                for e in range(E):
                    idx_e = elite_skill_t[:, e]
                    counts.scatter_add_(1, idx_e.unsqueeze(1), w[:, e].unsqueeze(1))
                logits_new_s = (counts + self.laplace).log()
                logits_s[t] = self.momentum * logits_s[t] + (1 - self.momentum) * logits_new_s


                elite_base_t = base_seq[t][b_idx, topk, :]
                p_hat_b = (w_exp * elite_base_t).sum(dim=1).clamp(self.eps_bern, 1 - self.eps_bern)
                logits_new_b = self.LogitsFromProb(p_hat_b, self.eps_bern)
                logits_b[t] = self.momentum * logits_b[t] + (1 - self.momentum) * logits_new_b

                elite_extra_t = extra_seq[t][b_idx, topk, :]
                p_hat_e = (w_exp * elite_extra_t).sum(dim=1).clamp(self.eps_bern, 1 - self.eps_bern)
                logits_new_e = self.LogitsFromProb(p_hat_e, self.eps_bern)
                logits_e[t] = self.momentum * logits_e[t] + (1 - self.momentum) * logits_new_e

                elite_click_t = click_seq[t][b_idx, topk, :]
                p_hat_c = (w_exp * elite_click_t).sum(dim=1).clamp(self.eps_bern, 1 - self.eps_bern)
                logits_new_c = self.LogitsFromProb(p_hat_c, self.eps_bern)
                logits_c[t] = self.momentum * logits_c[t] + (1 - self.momentum) * logits_new_c


        mouse_mu0  = mu_t[0]
        mouse_var0 = (std_t[0] * std_t[0])
        out = {
            "mouse": {"mu": mouse_mu0, "var": mouse_var0},
            "skill": {"logits": logits_s[0]},
            "base":  {"logits": logits_b[0]},
            "extra": {"logits": logits_e[0]},
            "click": {"logits": logits_c[0]}}

        if returnTrajectories:
            out["diagnostics"] = {
                "mu_seq": mu_t, "std_seq": std_t,
                "skill_logits_seq": logits_s,
                "base_logits_seq":  logits_b,
                "extra_logits_seq": logits_e,
                "click_logits_seq": logits_c}
        return out


class DecisionPlannerExtractor:
    def __init__(self):
        pass

    def BuildPlanner(self, worldModel: nn.Module,KEYBOARD_LAYOUT: Dict[str, Dict[str, int]],includeNoSkill: bool = True,**cemKwargs) -> CEMPlanner:
        base_codes  = [KEYBOARD_LAYOUT["base_keys"][k]  for k in KEYBOARD_LAYOUT["base_keys"].keys()]
        skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in KEYBOARD_LAYOUT["skill_keys"].keys()]
        extra_codes = []
        for grp in ["menu_keys", "system_keys", "alpha_keys"]:
            for _, code in KEYBOARD_LAYOUT[grp].items():
                extra_codes.append(code)
        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        max_code = max(all_codes)

        return CEMPlanner(worldModel=worldModel,baseCodes=base_codes,skillCodes=skill_codes,extraCodes=extra_codes,maxCode=max_code,hasNoSkill=includeNoSkill,**cemKwargs)




class ConstantActionEncoder(nn.Module):
    def __init__(self, outDim=128):
        super().__init__()
        self.register_buffer("vec", torch.zeros(1, outDim))
    def forward(self, keysOnehot, mouseDelta):
        B = keysOnehot.size(0)
        return self.vec.expand(B, -1).contiguous()

class WorldModelSimulation(nn.Module):
    def __init__(self, actionDim=128, deterDim=256, stochDim=32, stateDim=256):
        super().__init__()
        self.deter_dim = deterDim
        self.stoch_dim = stochDim
        self.state_dim = stateDim

        self.action_encoder = ConstantActionEncoder(outDim=actionDim)

        self.act_proj = nn.Sequential(nn.Linear(actionDim, stochDim), nn.Tanh())
        self.gru = nn.GRUCell(input_size=stochDim + stochDim, hidden_size=deterDim)
        self.prior_head = nn.Linear(deterDim, 2 * stochDim)  # -> mu, logstd
        self.state_proj = nn.Sequential(nn.Linear(deterDim + stochDim, stateDim), nn.LayerNorm(stateDim))
        self.rew_head = nn.Linear(stateDim, 1)
        self.done_head = nn.Linear(stateDim, 1)

        self.register_buffer("_h", torch.zeros(1, deterDim))
        self.register_buffer("_z", torch.zeros(1, stochDim))

    def ResetHidden(self, B: int = 1, device: Optional[torch.device] = None):
        if device is None:
            device = self._h.device
        self._h = torch.zeros(B, self.deter_dim, device=device)
        self._z = torch.zeros(B, self.stoch_dim, device=device)

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._h, self._z

    @torch.no_grad()
    def StepPriorOnly(self, hPrev: torch.Tensor, zPrev: torch.Tensor, aEnc: torch.Tensor, sample: bool = False):
        a = self.act_proj(aEnc)
        h_next = self.gru(torch.cat([zPrev, a], dim=-1), hPrev) 
        mu_p, logstd_p = self.prior_head(h_next).chunk(2, dim=-1)
        logstd_p = torch.clamp(logstd_p, -6.0, 2.0)
        z_next = mu_p + torch.randn_like(mu_p) * torch.exp(logstd_p) if sample else mu_p

        s_next = self.state_proj(torch.cat([h_next, z_next], dim=-1)) 
        r_pred = self.rew_head(s_next).squeeze(-1)
        d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1) 

        self._h = h_next.detach()
        self._z = z_next.detach()
        return h_next, z_next, s_next, r_pred, d_prob


class TestDecisionMTool:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.base_names  = list(KEYBOARD_LAYOUT["base_keys"].keys())
        self.skill_names = list(KEYBOARD_LAYOUT["skill_keys"].keys())
        self.extra_groups = ["menu_keys", "system_keys", "alpha_keys"]
        self.base_codes  = [KEYBOARD_LAYOUT["base_keys"][k]  for k in self.base_names]
        self.skill_codes = [KEYBOARD_LAYOUT["skill_keys"][k] for k in self.skill_names]
        self.extra_codes = []
        for g in self.extra_groups:
            self.extra_codes += [c for _, c in KEYBOARD_LAYOUT[g].items()]
        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        self.max_code = max(all_codes) 
        self.keys128_dim = self.max_code + 1 + 2 
        self.num_base  = len(self.base_codes) 
        self.num_skill = len(self.skill_names) + 1
        self.num_extra = len(self.extra_codes)

    def ResetHebbianMemory(self, hebb_layer: nn.Module):
        if hasattr(hebb_layer, "ResetHebbianMemory"):
            hebb_layer.ResetHebbianMemory()
        elif hasattr(hebb_layer, "hebb"):
            with torch.no_grad():
                hebb_layer.hebb.zero_()

    def TestHebbianPlasticityLayer(self) -> bool:
        layer = HebbianPlasticityLayer(inDim=512, outDim=512).to(self.device)
        x = torch.randn(4, 512, device=self.device)
        y = layer(x, update=False)
        ok = (y.shape == (4, 512))
        y2 = layer(x, update=True)
        ok = ok and (y2.shape == (4, 512))
        self.ResetHebbianMemory(layer)
        print("HebbianPlasticityLayer test {}.".format("passed" if ok else "failed"))
        return ok

    def TestDecisionExtractorNoPrior(self) -> bool:
        model = DecisionExtractor(stateDim=768, includeNoSkill=True, useHebbOnline=False).to(self.device)
        model.eval()
        B = 3
        state_feat = torch.randn(B, 768, device=self.device)

        out0 = model(state_feat,sample=False,prior=None,updateHebb=False)
        ok = True
        kb = out0["keyboard"]
        ms = out0["mouse"]
        ok = ok and (kb["base_logits"].shape  == (B, self.num_base))
        ok = ok and (kb["skill_logits"].shape == (B, self.num_skill))
        ok = ok and (kb["extra_logits"].shape == (B, self.num_extra))
        ok = ok and (ms["mu"].shape == (B, 2) and ms["logstd"].shape == (B, 2) and ms["click_logits"].shape == (B, 2))

        out1 = model(state_feat,sample=True,deterministic=False,prior=None,updateHebb=False,returnKeys128=True,applyConstraints=True)
        ok = ok and ("keys128_raw" in out1)
        ok = ok and ("keys128" in out1)
        ok = ok and (out1["keys128"].shape == (B, self.keys128_dim))
        print("DecisionExtractor (no prior) test {}.".format("passed" if ok else "failed"))
        return ok

    def TestCEMPlanner(self) -> bool:
        wm = WorldModelSimulation().to(self.device)
        wm.ResetHidden(B=2, device=self.device)

        planner = CEMPlanner(
            worldModel=wm,
            baseCodes=self.base_codes,
            skillCodes=self.skill_codes,
            extraCodes=self.extra_codes,
            maxCode=self.max_code,
            hasNoSkill=True,
            horizon=3, N=16, elite=4, iters=2).to(self.device)

        prior = planner.Plan(returnTrajectories=False)
        ok = True
        ok = ok and ("mouse" in prior and "mu" in prior["mouse"] and "var" in prior["mouse"])
        ok = ok and (prior["mouse"]["mu"].shape == (2, 2) and prior["mouse"]["var"].shape == (2, 2))
        ok = ok and (prior["skill"]["logits"].shape == (2, self.num_skill))
        ok = ok and (prior["base"]["logits"].shape  == (2, self.num_base))
        ok = ok and (prior["extra"]["logits"].shape == (2, self.num_extra))
        ok = ok and (prior["click"]["logits"].shape == (2, 2))
        print("CEMPlanner test {}.".format("passed" if ok else "failed"))
        return ok

    def TestDecisionExtractorWithPrior(self) -> bool:
        wm = WorldModelSimulation().to(self.device)
        wm.ResetHidden(B=2, device=self.device)
        planner = CEMPlanner(worldModel=wm,baseCodes=self.base_codes,skillCodes=self.skill_codes,extraCodes=self.extra_codes,maxCode=self.max_code,hasNoSkill=True,horizon=3, N=16, elite=4, iters=2).to(self.device)
        prior = planner.Plan(returnTrajectories=False)

        model = DecisionExtractor(stateDim=768, includeNoSkill=True, useHebbOnline=True).to(self.device)
        model.eval()
        state_feat = torch.randn(2, 768, device=self.device)

        out = model(state_feat,sample=True,deterministic=False,prior=prior,mixW=0.3,updateHebb=True, returnKeys128=True,applyConstraints=True)

        ok = True
        ok = ok and ("keys128" in out and out["keys128"].shape == (2, self.keys128_dim))
        ok = ok and (out["keyboard"]["base_logits"].shape  == (2, self.num_base))
        ok = ok and (out["keyboard"]["skill_logits"].shape == (2, self.num_skill))
        ok = ok and (out["keyboard"]["extra_logits"].shape == (2, self.num_extra))
        ok = ok and (out["mouse"]["mu"].shape == (2, 2))
        print("DecisionExtractor (with prior) test {}.".format("passed" if ok else "failed"))
        return ok

    def TestIntegrationEndToEnd(self) -> bool:
        wm = WorldModelSimulation().to(self.device)
        wm.ResetHidden(B=1, device=self.device)

        planner = CEMPlanner(worldModel=wm,baseCodes=self.base_codes,skillCodes=self.skill_codes,extraCodes=self.extra_codes,maxCode=self.max_code,hasNoSkill=True,horizon=3, N=8, elite=2, iters=2).to(self.device)
        prior = planner.Plan(returnTrajectories=False)

        dec = DecisionExtractor(stateDim=768, includeNoSkill=True, useHebbOnline=False).to(self.device)
        dec.eval()
        state_feat = torch.randn(1, 768, device=self.device)

        out = dec(state_feat,sample=True,deterministic=True,prior=prior,mixW=0.5,updateHebb=False,returnKeys128=True,applyConstraints=True)
        ok = ("keys128" in out and out["keys128"].shape == (1, self.keys128_dim))
        print("Integration (CEM -> Decision) test {}.".format("passed" if ok else "failed"))
        return ok

    def RunAll(self):
        results = []
        results.append(self.TestHebbianPlasticityLayer())
        results.append(self.TestDecisionExtractorNoPrior())
        results.append(self.TestCEMPlanner())
        results.append(self.TestDecisionExtractorWithPrior())
        results.append(self.TestIntegrationEndToEnd())
        passed = sum(1 for x in results if x)
        print(f"[DecisionModule Tests] {passed}/{len(results)} passed.")
        return all(results)
