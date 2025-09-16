from __future__ import annotations
import math
from typing import Dict, Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


KEYBOARD_LAYOUT = {
    "base_keys": {
        "Esc": 0,
        "W": 32, "A": 45, "S": 46, "D": 47,
        "Space": 72,
        "Shift": 57,
        "Ctrl": 69 
    },

    "skill_keys": {
        "1": 17, "2": 18, "3": 19, "4": 20, "5": 21,
        "Q": 31, "E": 33, "R": 34, "F": 48,
        "F1": 1, "F2": 2, "F3": 3, "F4": 4,
        "G": 49, "T": 35, "V": 61, "B": 62
    },

    "menu_keys": {
        "Tab": 30, "I": 38, "M": 64, "J": 51, "K": 52, "L": 53,
        "U": 37, "O": 39, "P": 40,
        "F5": 5, "F6": 6,
        "Insert": 77, "Delete": 80, "Home": 78, "End": 81
    },

    "system_keys": {
        "Enter": 56, "Backspace": 29, "CapsLock": 44,
        "Win": 70, "Alt": 71, "RCtrl": 76, "RShift": 68, "RAlt": 73, "RWin": 74
    },

    "alpha_keys": {
        "F7": 7, "F8": 8, "F9": 9, "F10": 10, "F11": 11, "F12": 12,
        "PrintScreen": 13, "ScrollLock": 14, "Pause": 15,

        "Grave": 16,
        "6": 22, "7": 23, "8": 24, "9": 25, "0": 26,
        "Minus": 27, "Equal": 28,

        "TildeBackslash": 43, 
        "Y": 36,
        "LeftBracket": 41, "RightBracket": 42,

        "H": 50,
        "Semicolon": 54, "Apostrophe": 55,

        "Z": 58, "X": 59, "C": 60,
        "N": 63,
        "Comma": 65, "Dot": 66, "Slash": 67,

        "PageUp": 79, "PageDown": 82,
        "ArrowUp": 83, "ArrowLeft": 84, "ArrowDown": 85, "ArrowRight": 86,

        "Menu": 75,

        "NumLock": 87, "NumpadDivide": 88, "NumpadMultiply": 89, "NumpadMinus": 90,
        "Numpad7": 91, "Numpad8": 92, "Numpad9": 93, "NumpadPlus": 94,
        "Numpad4": 95, "Numpad5": 96, "Numpad6": 97,
        "Numpad1": 98, "Numpad2": 99, "Numpad3": 100, "NumpadEnter": 101,
        "Numpad0": 102, "NumpadDot": 103
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
        if update:
            with torch.no_grad():
                pre = x.detach()
                post = out.detach()
                B = pre.size(0)
                delta = self.rate * (torch.einsum("bo,bi->oi", post, pre) / max(1, B) - (post.pow(2).mean(0, keepdim=True).t() * self.hebb))
                self.hebb.copy_(self.hebb * self.decay + delta)
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
    def __init__(self, zDim=256, numOptions=16, psiDim=128, hidden=256):
        super().__init__()
        self.K = numOptions
        self.enc = nn.Sequential(nn.Linear(zDim, hidden), nn.ReLU())

        self.pi_o = nn.Linear(hidden, self.K)

        self.trans = nn.Parameter(torch.zeros(self.K, self.K)) 

        self.psi_head = nn.Linear(hidden, self.K * psiDim)
        self.psiDim = psiDim

        self.beta_head = nn.Sequential(
            nn.Linear(hidden + self.K, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        
        nn.init.constant_(self.beta_head[-1].bias, -2.2)

    def forward(self, z, prevOnehot=None):
        h = self.enc(z)
        logits_base = self.pi_o(h)

        if prevOnehot is not None:
            prev = prevOnehot.detach()
            logits_o = logits_base + prev @ self.trans
            beta = torch.sigmoid(self.beta_head(torch.cat([h, prev], dim=-1)))
        else:
            logits_o = logits_base
            beta = torch.sigmoid(self.beta_head(torch.cat([h, torch.zeros_like(logits_base)], dim=-1)))

        psi_all = self.psi_head(h).view(-1, self.K, self.psiDim)
        return logits_o, psi_all, beta


class DecisionExtractor(nn.Module):
    def __init__(
        self,
        stateDim: int = 768,
        includeNoSkill: bool = True,
        useHebbOnline: bool = False,
        optionNum: int = 16,
        psiDim: int = 128,
        *,
        entropyWeights: Tuple[float, float, float, float] = (0.3, 0.2, 0.4, 0.1),
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

        self.num_options = optionNum

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

    def ToKeysVec(self, baseAct: torch.Tensor, extraAct: torch.Tensor, skillIdx: torch.Tensor, clicks: torch.Tensor) -> torch.Tensor:
        B = baseAct.size(0)
        device = baseAct.device
        vec = torch.zeros(B, self.max_code + 1 + 2, device=device)

        for i, code in enumerate(self.base_codes):
            vec[:, code] = baseAct[:, i]

        for i, code in enumerate(self.extra_codes):
            vec[:, code] = extraAct[:, i]

        if self.no_skill_id is None:
            chosen = skillIdx
            valid = torch.ones_like(chosen, dtype=torch.bool)
        else:
            valid = (skillIdx != self.no_skill_id)
            chosen = skillIdx.clamp(max=len(self.skill_codes) - 1)
        if valid.any():
            sel_codes = torch.tensor(self.skill_codes, device=device)[chosen[valid]]
            vec[valid, sel_codes] = 1.0

        vec[:, self.max_code + 1:self.max_code + 3] = clicks
        return vec

    @staticmethod
    def ApplyConstraints(keyVec: torch.Tensor) -> torch.Tensor:
        x = keyVec
        max_scan = x.size(1) - 2

        W = KEYBOARD_LAYOUT["base_keys"]["W"]
        S = KEYBOARD_LAYOUT["base_keys"]["S"]
        A = KEYBOARD_LAYOUT["base_keys"]["A"]
        D = KEYBOARD_LAYOUT["base_keys"]["D"]

        ws_conf = ((x[:, W] > 0.5) & (x[:, S] > 0.5)).unsqueeze(1)
        ad_conf = ((x[:, A] > 0.5) & (x[:, D] > 0.5)).unsqueeze(1)
        x2 = x.clone()
        x2[:, S:S+1] = torch.where(ws_conf, x2.new_zeros(x2[:, S:S+1].shape), x2[:, S:S+1])
        x2[:, D:D+1] = torch.where(ad_conf, x2.new_zeros(x2[:, D:D+1].shape), x2[:, D:D+1])

        if max_scan > 0:
            pressed = (x2[:, :max_scan] > 0.5)
            rank = torch.cumsum(pressed.to(torch.int16), dim=1)
            keep = (rank <= 6) | (~pressed)
            x2 = torch.cat([x2[:, :max_scan] * keep.float(), x2[:, max_scan:]], dim=1)
        return x2

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
        sample: bool = True,                               
        deterministic: bool = False,                           
        prevOptionOnehot: Optional[torch.Tensor] = None,     
        prior: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,  
        mixW: float = 0.25,                              
        updateHebb: bool = False,                      
        returnKeysVec: bool = True,                        
        applyConstraints: bool = True) -> Dict[str, torch.Tensor]:
        
        B = stateFeat.size(0)

        z = self.Encode(stateFeat, updateHebb=updateHebb)

        if prevOptionOnehot is not None:
            prevOptionOnehot = prevOptionOnehot.detach().clone()

        option_logits, psi_all, beta = self.option(z, prevOptionOnehot)

        base_logits, skill_logits, extra_logits = self.keyboard.Logits(z)

        mu, logstd, click_logits = self.mouse.Params(z)

        if prior is not None:
            base_logits = MixLogits(base_logits,  prior.get("base", {}).get("logits", None),  mixW)
            extra_logits = MixLogits(extra_logits, prior.get("extra", {}).get("logits", None),  mixW)
            skill_logits = MixLogits(skill_logits, prior.get("skill", {}).get("logits", None),  mixW)
            mu, logstd = MixGauss(mu, logstd, prior.get("mouse", {}).get("mu",  None), prior.get("mouse", {}).get("var", None), mixW)
            click_logits = MixLogits(click_logits, prior.get("click", {}).get("logits", None), mixW)

        comps = self.EntropyComponents(base_logits, extra_logits, skill_logits, logstd)

        entropy_scalar = self.AggregateEntropy(comps)  

        out: Dict[str, any] = {
            "z": z,
            "entropy": entropy_scalar, 
            "option": {"logits": option_logits, "psi_all": psi_all, "beta": beta},
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
                clicks = (torch.sigmoid(click_logits) > 0.5).float()

                mouse_a = mu

                logp_base = StableLogProbBernoulli(base_logits, base_act)
                logp_extra = StableLogProbBernoulli(extra_logits, extra_act)
                logp_skill = torch.distributions.Categorical(logits=skill_logits).log_prob(skill_idx)
                logp_mouse = -(logstd.sum(-1) + 0.5 * logstd.size(-1) * math.log(2 * math.pi))
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

            device = z.device
            if prevOptionOnehot is not None:
                prev_idx = torch.argmax(prevOptionOnehot, dim=-1)
            else:
                prev_idx = torch.zeros(B, dtype=torch.long, device=device)

            if deterministic:
                if beta is None:
                    terminate = torch.ones(B, 1, device=device)
                else:
                    terminate = (beta > 0.5).float()
                new_idx = torch.argmax(option_logits, dim=-1)
            else:
                if beta is None:
                    terminate = torch.ones(B, 1, device=device)
                else:
                    terminate = torch.bernoulli(beta.clamp(1e-6, 1-1e-6))
                new_idx = torch.distributions.Categorical(logits=option_logits).sample()

            term_mask = terminate.squeeze(-1).bool()

            opt_idx = torch.where(term_mask, new_idx, prev_idx)

            psi = psi_all[torch.arange(B, device=device), opt_idx]

            dist_opt = torch.distributions.Categorical(logits=option_logits)
            logp_new = dist_opt.log_prob(new_idx)
            logp_opt = torch.where(term_mask, logp_new, torch.zeros_like(logp_new))

            if beta is None:
                log_beta = torch.zeros(B, device=device)
            else:
                b = beta.clamp(1e-6, 1-1e-6).squeeze(-1)
                t = terminate.squeeze(-1)
                log_beta = t * b.log() + (1 - t) * (1 - b).log()

            out["option"].update({
                "opt_idx": opt_idx,
                "terminate": terminate,
                "psi": psi,
                "logp_option": logp_opt,
                "logp_beta": log_beta,})
            
            if "opt_idx" in out["option"]:
                opt_idx = out["option"]["opt_idx"]
                opt_onehot = torch.nn.functional.one_hot(
                    opt_idx, num_classes=self.num_options).float().to(opt_idx.device)
                out["option"]["opt_onehot"] = opt_onehot.detach()  

            if returnKeysVec:
                keyvec_raw = self.ToKeysVec(base_act, extra_act, skill_idx, clicks)
                out["keyvec_raw"] = keyvec_raw
                out["key_vec"] = self.ApplyConstraints(keyvec_raw) if applyConstraints else keyvec_raw

        return out

    def ResetHebbianMemory(self, value: float = 0.0):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, HebbianPlasticityLayer):
                    m.hebb.fill_(value)



class CEMPlanner(nn.Module):
    def __init__(self,worldModel: nn.Module,baseCodes: List[int],skillCodes: List[int],extraCodes: List[int],maxCode: int,hasNoSkill: bool = True, horizon: int = 5, N: int = 64,
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
    def AssembleKeyVec(baseAct: torch.Tensor,extraAct: torch.Tensor,skillIdx: torch.Tensor,clickAct: torch.Tensor,baseCodes: torch.Tensor,
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

        key_vec = torch.cat([keys, clickAct], dim=-1)  
        return key_vec

    @torch.no_grad()
    def Plan(self,
             mouseMu: Optional[torch.Tensor] = None,         
             mouseLogstd: Optional[torch.Tensor] = None,     
             skillLogits: Optional[torch.Tensor] = None,     
             baseLogits: Optional[torch.Tensor] = None,       
             extraLogits: Optional[torch.Tensor] = None,    
             clickLogits: Optional[torch.Tensor] = None,     
             h0: Optional[torch.Tensor] = None, # Deterministic hidden states of the world model
             z0: Optional[torch.Tensor] = None, # Random hidden states of the world model
             returnTrajectories: bool = False) -> Dict[str, Dict[str, torch.Tensor]]:

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
            cont = torch.ones(B, N, device=device)

            for t in range(H):
                a_mouse_t = mouse_seq[t].reshape(B * N, 2)
                a_skill_t = skill_seq[t].reshape(B * N)
                a_base_t  = base_seq[t].reshape(B * N, self.n_base)
                a_extra_t = extra_seq[t].reshape(B * N, self.n_extra)
                a_click_t = click_seq[t].reshape(B * N, 2)

                key_vec = self.AssembleKeyVec(
                    a_base_t, a_extra_t, a_skill_t, a_click_t,
                    self.base_codes_buf, self.skill_codes_buf, self.extra_codes_buf,
                    self.max_code, self.has_no_skill)

                a_enc = self.wm.action_encoder(key_vec, a_mouse_t)
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
        base_codes = [KEYBOARD_LAYOUT["base_keys"][k] for k in KEYBOARD_LAYOUT["base_keys"].keys()]
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





class TestDecisionMTool:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.base_names = list(KEYBOARD_LAYOUT["base_keys"].keys())
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
        self.keyvec_dim = self.max_code + 1 + 2 
        self.num_base = len(self.base_codes)
        self.num_skill = len(self.skill_names) + 1 
        self.num_extra = len(self.extra_codes)

    class MockActionEncoder(nn.Module):
        def __init__(self, keyDim: int, outDim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(keyDim + 2, 128), nn.ReLU(),
                nn.Linear(128, outDim))
            
        def forward(self, keysOnehot: torch.Tensor, mouseDelta: torch.Tensor):
            x = torch.cat([keysOnehot.float(), mouseDelta.float()], dim=-1)
            return self.net(x)

    class MockWorldModel(nn.Module):
        def __init__(self, keyDim: int = 106, actionDim: int = 128, deterDim: int = 256, stochDim: int = 32, stateDim: int = 256):
            super().__init__()
            self.deter_dim = deterDim
            self.stoch_dim = stochDim
            self.state_dim = stateDim
            self.action_dim = actionDim

            self.action_encoder = TestDecisionMTool.MockActionEncoder(keyDim, actionDim)

            self.act_proj = nn.Sequential(nn.Linear(actionDim, stochDim), nn.Tanh())
            self.gru = nn.GRUCell(input_size=stochDim + stochDim, hidden_size=deterDim)
            self.prior_head = nn.Linear(deterDim, 2 * stochDim)
            self.state_proj = nn.Linear(deterDim + stochDim, stateDim)
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
            z_next = mu_p

            s_next = self.state_proj(torch.cat([h_next, z_next], dim=-1))
            r_pred = self.rew_head(s_next).squeeze(-1)
            d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)
            return h_next, z_next, s_next, r_pred, d_prob

    def ResetHebbianMemory(self, hebb_layer: nn.Module):
        try:
            if hasattr(hebb_layer, "ResetHebbianMemory"):
                hebb_layer.ResetHebbianMemory()
            elif hasattr(hebb_layer, "hebb"):
                with torch.no_grad():
                    hebb_layer.hebb.zero_()
        except Exception as e:
            print("ResetHebbianMemory error:", type(e).__name__, e)

    def TestHebbianPlasticityLayer(self) -> bool:
        try:
            layer = HebbianPlasticityLayer(inDim=512, outDim=512).to(self.device)
            x = torch.randn(4, 512, device=self.device)
            y = layer(x, update=False)
            ok = (y.shape == (4, 512))
            y2 = layer(x, update=True)
            ok = ok and (y2.shape == (4, 512))
            self.ResetHebbianMemory(layer)
            print("HebbianPlasticityLayer test {}.".format("passed" if ok else "failed"))
            return ok
        except Exception as e:
            print("HebbianPlasticityLayer test crash:", type(e).__name__, e)
            return False

    def TestDecisionExtractorNoPrior(self) -> bool:
        try:
            model = DecisionExtractor(stateDim=1024, includeNoSkill=True, useHebbOnline=False).to(self.device)
            model.eval()
            B = 3
            state_feat = torch.randn(B, 1024, device=self.device)

            out0 = model(state_feat, sample=False, prior=None, updateHebb=False)
            ok = True
            kb = out0["keyboard"]; ms = out0["mouse"]
            ok = ok and (kb["base_logits"].shape  == (B, self.num_base))
            ok = ok and (kb["skill_logits"].shape == (B, self.num_skill))
            ok = ok and (kb["extra_logits"].shape == (B, self.num_extra))
            ok = ok and (ms["mu"].shape == (B, 2) and ms["logstd"].shape == (B, 2) and ms["click_logits"].shape == (B, 2))

            out1 = model(state_feat, sample=True, deterministic=False, prior=None, updateHebb=False, returnKeysVec=True, applyConstraints=True)
            ok = ok and ("keyvec_raw" in out1) and ("key_vec" in out1)
            ok = ok and (out1["key_vec"].shape == (B, self.keyvec_dim))
            print("DecisionExtractor (no prior) test {}.".format("passed" if ok else "failed"))
            return ok
        except Exception as e:
            print("DecisionExtractorNoPrior test crash:", type(e).__name__, e)
            return False

    def _build_planner(self, horizon=3, N=16, elite=4, iters=2):
        try:
            wm = self.MockWorldModel(keyDim=self.keyvec_dim, actionDim=128, deterDim=256, stochDim=32, stateDim=256).to(self.device)
            wm.ResetHidden(B=2, device=self.device)
            planner = CEMPlanner(
                worldModel=wm,
                baseCodes=self.base_codes,
                skillCodes=self.skill_codes,
                extraCodes=self.extra_codes,
                maxCode=self.max_code,
                hasNoSkill=True,
                horizon=horizon, N=N, elite=elite, iters=iters).to(self.device)
            return wm, planner
        except Exception as e:
            print("_build_planner crash:", type(e).__name__, e)
            return None, None

    def TestCEMPlanner(self) -> bool:
        try:
            _, planner = self._build_planner(horizon=3, N=16, elite=4, iters=2)
            if planner is None:
                return False
            prior = planner.Plan(returnTrajectories=False)
            ok = True
            ok = ok and ("mouse" in prior and "mu" in prior["mouse"] and "var" in prior["mouse"])
            ok = ok and (prior["mouse"]["mu"].shape == (2, 2) and prior["mouse"]["var"].shape == (2, 2))
            ok = ok and (prior["skill"]["logits"].shape == (2, self.num_skill))
            ok = ok and (prior["base"]["logits"].shape == (2, self.num_base))
            ok = ok and (prior["extra"]["logits"].shape == (2, self.num_extra))
            ok = ok and (prior["click"]["logits"].shape == (2, 2))
            print("CEMPlanner test {}.".format("passed" if ok else "failed"))
            return ok
        except Exception as e:
            print("CEMPlanner test crash:", type(e).__name__, e)
            return False

    def TestDecisionExtractorWithPrior(self) -> bool:
        try:
            _, planner = self._build_planner(horizon=3, N=16, elite=4, iters=2)
            if planner is None:
                return False
            prior = planner.Plan(returnTrajectories=False)

            model = DecisionExtractor(stateDim=1024, includeNoSkill=True, useHebbOnline=True).to(self.device)
            model.eval()
            model.num_options = model.option.K

            state_feat = torch.randn(2, 1024, device=self.device)
            out = model(state_feat, sample=True, deterministic=False, prior=prior, mixW=0.3, updateHebb=True, returnKeysVec=True, applyConstraints=True)

            ok = True
            ok = ok and ("key_vec" in out and out["key_vec"].shape == (2, self.keyvec_dim))
            ok = ok and (out["keyboard"]["base_logits"].shape == (2, self.num_base))
            ok = ok and (out["keyboard"]["skill_logits"].shape == (2, self.num_skill))
            ok = ok and (out["keyboard"]["extra_logits"].shape == (2, self.num_extra))
            ok = ok and (out["mouse"]["mu"].shape == (2, 2))
            print("DecisionExtractor (with prior) test {}.".format("passed" if ok else "failed"))
            return ok
        except Exception as e:
            print("DecisionExtractorWithPrior test crash:", type(e).__name__, e)
            return False

    def TestIntegrationEndToEnd(self) -> bool:
        try:
            _, planner = self._build_planner(horizon=3, N=8, elite=2, iters=2)
            if planner is None:
                return False
            planner.wm.ResetHidden(B=1, device=self.device)
            prior = planner.Plan(returnTrajectories=False)

            dec = DecisionExtractor(stateDim=1024, includeNoSkill=True, useHebbOnline=False).to(self.device)
            dec.eval()
            dec.num_options = dec.option.K 
            state_feat = torch.randn(1, 1024, device=self.device)

            out = dec(state_feat, sample=True, deterministic=True, prior=prior, mixW=0.5, updateHebb=False, returnKeysVec=True, applyConstraints=True)
            ok = ("key_vec" in out and out["key_vec"].shape == (1, self.keyvec_dim))
            print("Integration (CEM -> Decision) test {}.".format("passed" if ok else "failed"))
            return ok
        except Exception as e:
            print("IntegrationEndToEnd test crash:", type(e).__name__, e)
            return False

    class Teacher(nn.Module):
        def __init__(self, inDim, numBase, numSkill, numExtra):
            super().__init__()
            h = 384
            self.backbone = nn.Sequential(
                nn.Linear(inDim, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),)
            
            self.base_head = nn.Linear(h, numBase)
            self.extra_head = nn.Linear(h, numExtra)
            self.skill_head = nn.Linear(h, numSkill)
            self.mouse_mu = nn.Linear(h, 2)
            self.mouse_ls = nn.Linear(h, 2)
            self.click_head = nn.Linear(h, 2)

        def forward(self, x):
            h = self.backbone(x)
            return {
                "base_logits":  self.base_head(h),
                "extra_logits": self.extra_head(h),
                "skill_logits": self.skill_head(h),
                "mouse_mu":     self.mouse_mu(h),
                "mouse_logstd": torch.clamp(self.mouse_ls(h), -5.0, 2.0),
                "click_logits": self.click_head(h),}

    def SupervisedLoss(self, out, tgt):
        bce = nn.BCEWithLogitsLoss()
        loss = 0.0
        loss += bce(out["keyboard"]["base_logits"], torch.sigmoid(tgt["base_logits"]))
        loss += bce(out["keyboard"]["extra_logits"], torch.sigmoid(tgt["extra_logits"]))
        loss += bce(out["mouse"]["click_logits"], torch.sigmoid(tgt["click_logits"]))
        labels = torch.argmax(tgt["skill_logits"], dim=-1)
        loss += F.cross_entropy(out["keyboard"]["skill_logits"], labels)
        loss += F.mse_loss(out["mouse"]["mu"], tgt["mouse_mu"])
        loss += F.mse_loss(out["mouse"]["logstd"], tgt["mouse_logstd"])
        return loss

    def TrainStepSmoke(self):
        try:
            model = DecisionExtractor(stateDim=1024, includeNoSkill=True, useHebbOnline=False).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            B = 8
            state_feat = torch.randn(B, 1024, device=self.device)

            num_options = model.option.K
            prev_opt = torch.randint(0, num_options, (B,), device=self.device)
            prev_onehot = F.one_hot(prev_opt, num_classes=num_options).float()

            out = model(state_feat, sample=False, deterministic=False, prior=None, mixW=0.0, updateHebb=False, prevOptionOnehot=prev_onehot, returnKeysVec=False, applyConstraints=False)

            kb = out["keyboard"]; ms = out["mouse"]
            base_tgt = torch.rand_like(kb["base_logits"])
            extra_tgt = torch.rand_like(kb["extra_logits"])
            bce = torch.nn.BCEWithLogitsLoss()
            loss_base = bce(kb["base_logits"], base_tgt)
            loss_extra = bce(kb["extra_logits"], extra_tgt)

            num_skill = kb["skill_logits"].size(-1)
            skill_tgt = torch.randint(0, num_skill, (B,), device=self.device)
            loss_skill = F.cross_entropy(kb["skill_logits"], skill_tgt)

            mu, logstd = ms["mu"], ms["logstd"]
            std = torch.exp(logstd)
            mouse_tgt = torch.randn_like(mu)
            loss_mouse = 0.5 * (((mouse_tgt - mu) / std) ** 2 + 2 * logstd + math.log(2 * math.pi)).sum(dim=-1).mean()

            click_tgt = torch.rand_like(ms["click_logits"])
            loss_click = bce(ms["click_logits"], click_tgt)

            main_loss = loss_base + loss_extra + loss_skill + loss_mouse + loss_click

            opt_dict   = out["option"]
            opt_logits = opt_dict["logits"]

            if "psi" in opt_dict:
                psi = opt_dict["psi"]
            else:
                psi_all = opt_dict["psi_all"] 
                idx = torch.argmax(opt_logits, dim=-1)
                b_idx = torch.arange(B, device=self.device)
                psi = psi_all[b_idx, idx]

            beta = opt_dict["beta"].squeeze(-1)
            psi_dim = psi.size(-1)
            opt_tgt = torch.randint(0, num_options, (B,), device=self.device)
            psi_tgt = torch.randn(B, psi_dim, device=self.device)
            beta_tgt = torch.rand(B, device=self.device)

            loss_opt_ce = F.cross_entropy(opt_logits, opt_tgt)
            loss_psi_mse = F.mse_loss(psi, psi_tgt)
            loss_beta_bce = F.binary_cross_entropy(beta, beta_tgt)

            total = main_loss + (0.1 * loss_opt_ce + 0.05 * loss_psi_mse + 0.05 * loss_beta_bce)

            opt.zero_grad(set_to_none=True)
            total.backward()

            bad = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    if (p.grad is None) or (not torch.isfinite(p.grad).all()) or (p.grad.abs().sum() == 0):
                        bad.append(n)
            if bad:
                print("Decision TrainStepSmoke failed:\n\n Bad grad at:", bad)
                return False

            opt.step()
            print("Decision TrainStepSmoke passed.")
            return True
        except Exception as e:
            print("TrainStepSmoke crash:", type(e).__name__, e)
            return False

    def NoNanAfterManySteps(self, steps: int = 40) -> bool:
        try:
            in_dim = 512
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebbOnline=False).to(self.device)
            teacher = self.Teacher(in_dim, self.num_base, self.num_skill, self.num_extra).to(self.device)
            for p in teacher.parameters():
                p.requires_grad_(False)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            for t in range(steps):
                x = torch.randn(16, in_dim, device=self.device)
                with torch.no_grad():
                    tgt = teacher(x)
                out = model(x, sample=False, prior=None, updateHebb=False)
                loss = self.SupervisedLoss(out, tgt)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"
                opt.step()
            print("Decision NoNanAfterManySteps passed.")
            return True
        except AssertionError as e:
            print("Decision NoNanAfterManySteps failed:\n", e)
            return False
        except Exception as e:
            print("Decision NoNanAfterManySteps error:\n", type(e).__name__, e)
            return False

    def ParamsActuallyChange(self, steps: int = 20) -> bool:
        try:
            in_dim = 512
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebbOnline=False).to(self.device)
            teacher = self.Teacher(in_dim, self.num_base, self.num_skill, self.num_extra).to(self.device)
            for p in teacher.parameters():
                p.requires_grad_(False)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            with torch.no_grad():
                w0_feat = model.feature_net[0].weight.clone()
                w0_kbd = model.keyboard.base_head.weight.clone()
                w0_mu = model.mouse.mu_head.weight.clone()

            for _ in range(steps):
                x = torch.randn(16, in_dim, device=self.device)
                with torch.no_grad():
                    tgt = teacher(x)
                out = model(x, sample=False, prior=None, updateHebb=False)
                loss = self.SupervisedLoss(out, tgt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                d_feat = (w0_feat - model.feature_net[0].weight).norm().item()
                d_kbd = (w0_kbd - model.keyboard.base_head.weight).norm().item()
                d_mu = (w0_mu - model.mouse.mu_head.weight).norm().item()

            changed = any(d > 1e-6 for d in [d_feat, d_kbd, d_mu])
            assert changed, f"Parameters barely changed: feat={d_feat:.3e}, kbd={d_kbd:.3e}, mu={d_mu:.3e}"
            print("Decision ParamsActuallyChange passed.")
            return True
        except AssertionError as e:
            print("Decision ParamsActuallyChange failed:\n", e)
            return False
        except Exception as e:
            print("Decision ParamsActuallyChange error:\n", type(e).__name__, e)
            return False

    def TestNormalTrainingConvergence(self, steps: int = 120, logEvery: int = 30) -> bool:
        try:
            in_dim = 512
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebbOnline=False).to(self.device)
            teacher = self.Teacher(in_dim, self.num_base, self.num_skill, self.num_extra).to(self.device)
            for p in teacher.parameters():
                p.requires_grad_(False)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            B = 32
            xfix = torch.randn(B, in_dim, device=self.device)
            with torch.no_grad():
                tgtfix = teacher(xfix)

            with torch.no_grad():
                start = self.SupervisedLoss(model(xfix, sample=False, prior=None, updateHebb=False), tgtfix).item()

            for t in range(1, steps + 1):
                out = model(xfix, sample=False, prior=None, updateHebb=False)
                loss = self.SupervisedLoss(out, tgtfix)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                if (t % logEvery) == 0 or t == 1:
                    print(f"[DecisionTrain] step {t}/{steps} | loss={loss.item():.6f}")

            with torch.no_grad():
                end = self.SupervisedLoss(model(xfix, sample=False, prior=None, updateHebb=False), tgtfix).item()

            print(f"\n[DecisionTrain] loss start={start:.6f} -> end={end:.6f}")
            assert end <= 0.8 * start, "Training did not show sufficient convergence (drop < 20%)."
            print("Decision TestNormalTrainingConvergence passed.")
            return True
        except AssertionError as e:
            print("Decision TestNormalTrainingConvergence failed:\n", e)
            return False
        except Exception as e:
            print("Decision TestNormalTrainingConvergence error:\n", type(e).__name__, e)
            return False

    def RunAll(self):
        try:
            results = []
            results.append(self.TestHebbianPlasticityLayer())
            results.append(self.TestDecisionExtractorNoPrior())
            results.append(self.TestCEMPlanner())
            results.append(self.TestDecisionExtractorWithPrior())
            results.append(self.TestIntegrationEndToEnd())
            results.append(self.TrainStepSmoke())
            results.append(self.NoNanAfterManySteps())
            results.append(self.ParamsActuallyChange())
            results.append(self.TestNormalTrainingConvergence())

            passed = sum(1 for x in results if x)
            print(f"[DecisionModule Tests] {passed}/{len(results)} passed.")
            return all(results)
        except Exception as e:
            print("RunAll crash:", type(e).__name__, e)
            return False

