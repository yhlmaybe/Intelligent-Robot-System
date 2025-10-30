from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
from FunctionTools import GetParameterSScale, SiteSpec, BaseOnlineWrapper


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
    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).sum(-1)

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



class LoRALinearAdapter(nn.Module):
    def __init__(self, target_linear: nn.Linear):
        super().__init__()
        assert isinstance(target_linear, nn.Linear)
        self.target = target_linear
        self.in_f = target_linear.in_features
        self.out_f = target_linear.out_features

        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()  
        self.alpha = nn.ParameterList() 

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if addRank <= 0: return
        if init is None: init = {}
        dev, dt = self.target.weight.device, self.target.weight.dtype

        A = init.get("A", torch.randn(addRank, self.in_f, device=dev, dtype=dt) * 1e-4)
        B = init.get("B", torch.randn(self.out_f, addRank, device=dev, dtype=dt) * 1e-4)
        s = init.get("scale", 1e-2)

        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.tensor(s, device=dev, dtype=dt))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A); self.B_list.append(B); self.alpha.append(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.target.weight
        if len(self.A_list) > 0:
            dW = W.new_zeros(self.out_f, self.in_f)
            for A, B, s in zip(self.A_list, self.B_list, self.alpha):
                s_eff = torch.tanh(s) * GetParameterSScale(s) 
                dW = dW + s_eff * (B @ A)
            W = W + dW
        return F.linear(x, W, self.target.bias)


class MatLoRAAdapter(nn.Module):
    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.M, self.N = int(rows), int(cols)
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()

        self.register_buffer("_anchor", torch.empty(0))

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if addRank <= 0: return
        if init is None: init = {}

        dev, dt = self._anchor.device, self._anchor.dtype

        A = init.get("A", torch.randn(addRank, self.N, device=dev, dtype=dt) * 1e-4)
        B = init.get("B", torch.randn(self.M, addRank, device=dev, dtype=dt) * 1e-4)
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A); self.B_list.append(B); self.alpha.append(s)

    def forward(self, baseMatrix: torch.Tensor) -> torch.Tensor:
        M_eff = baseMatrix
        if len(self.A_list) > 0:
            d = baseMatrix.new_zeros(self.M, self.N)
            for A, B, s in zip(self.A_list, self.B_list, self.alpha):
                s_eff = torch.tanh(s) * GetParameterSScale(s)
                d = d + s_eff * (B @ A)
            M_eff = M_eff + d
        return M_eff


class HebbianPlasticityLayer(nn.Module):
    def __init__(self, inDim: int, outDim: int, rate: float = 1e-3, decay: float = 0.995, maxRowNorm: float = 2.0):
        super().__init__()
        self.rate = rate
        self.decay = decay
        self.max_row_norm = maxRowNorm
        self.base = nn.Parameter(torch.randn(outDim, inDim) * 0.02)
        self.register_buffer("hebb", torch.zeros(outDim, inDim))

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
    def __init__(self, inDim: int = 512, hidden: int = 256, actDim: int = 2, logstdBounds=(-5.0, 2.0)):
        super().__init__()
        self._ls_low, self._ls_high = logstdBounds

        self.backbone = nn.Sequential(
            nn.Linear(inDim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        
        self.mu_head = nn.Linear(hidden, actDim)
        self.logstd_head = nn.Linear(hidden, actDim)
        self.click_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 2))

    def Params(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(feat)
        mu = self.mu_head(h)
        logstd = ClampLogstd(self.logstd_head(h), self._ls_low, self._ls_high)
        click_logits = self.click_head(h)
        return mu, logstd, click_logits


class KeyboardActor(nn.Module):
    def __init__(self, inDim: int = 512,baseKeyNames: Optional[List[str]] = None,skillNames: Optional[List[str]] = None,includeNoSkill: bool = True,hidden: int = 256):
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
    def __init__(self, zDim=512, numOptions=16, psiDim=128, hidden=256):
        super().__init__()
        self.K = numOptions
        self.enc = nn.Sequential(nn.Linear(zDim, hidden), nn.ReLU())

        self.pi_o = nn.Linear(hidden, self.K)

        self.trans = nn.Parameter(torch.zeros(self.K, self.K)) 

        self.trans_adapter = MatLoRAAdapter(self.K, self.K)

        self.psi_head = nn.Linear(hidden, self.K * psiDim)
        self.psiDim = psiDim

        self.beta_head = nn.Sequential(
            nn.Linear(hidden + self.K, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        
        nn.init.constant_(self.beta_head[-1].bias, -0.5)

        self.psi_amp_global = nn.Parameter(torch.tensor(1.0))
        self.psi_amp_per_option = nn.Parameter(torch.ones(self.K))

    def forward(self, z, prevOnehot=None):
        h = self.enc(z)
        logits_base = self.pi_o(h)

        if prevOnehot is not None:
            prev = prevOnehot.detach()
            trans_eff = self.trans_adapter(self.trans)
            trans_eff = torch.nan_to_num(trans_eff, nan=0.0).clamp(-10.0, 10.0)
            logits_o = logits_base + prev @ trans_eff
            beta = torch.sigmoid(self.beta_head(torch.cat([h, prev], dim=-1))).clamp(1e-6, 1.0 - 1e-6)
        else:
            logits_o = logits_base
            beta = torch.sigmoid(self.beta_head(torch.cat([h, torch.zeros_like(logits_base)], dim=-1))).clamp(1e-6, 1.0 - 1e-6)

        psi_all = self.psi_head(h).view(-1, self.K, self.psiDim)
        psi_all = psi_all * self.psi_amp_global * self.psi_amp_per_option.view(1, self.K, 1)

        return logits_o, psi_all, beta


class SwiGLUBlock(nn.Module):
    def __init__(self, dim=768, hidden=1024, p=0.1, layerscale=1e-2):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden * 2)  
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(p)
        self.gamma = nn.Parameter(torch.ones(dim) * layerscale)  

    def forward(self, x):
        h = self.ln(x)
        a, b = self.fc1(h).chunk(2, dim=-1)
        h = F.silu(a) * b 
        h = self.fc2(h)
        return x + self.drop(h * self.gamma)

class DecisionExtractor(nn.Module):
    def __init__(
        self,
        stateDim: int = 768,
        includeNoSkill: bool = True,
        useHebb: bool = False,
        optionNum: int = 80,
        psiDim: int = 384,
        *,
        entropyWeights: Tuple[float, float, float, float] = (0.3, 0.2, 0.4, 0.1),
        logstdBounds: Tuple[float, float] = (-5.0, 2.0),):
        super().__init__()

        self.feature_net = nn.Sequential(
            SwiGLUBlock(dim=stateDim, hidden=1024, p=0.1),
            SwiGLUBlock(dim=stateDim, hidden=1024, p=0.1))
        
        self.hebb = HebbianPlasticityLayer(stateDim, stateDim)
        self.to_z = nn.Linear(stateDim, 512)
        self.use_hebb_online = useHebb

        base_names = list(KEYBOARD_LAYOUT["base_keys"].keys())
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

        self.keyboard = KeyboardActor(512, base_names, skill_names, includeNoSkill=includeNoSkill)
        self.mouse = MouseActor(512, logstdBounds=logstdBounds)
        self.option = OptionPolicy(zDim=512, numOptions=optionNum, psiDim=psiDim)

        self.register_buffer("entropy_w", torch.tensor(entropyWeights, dtype=torch.float32))
        self.logstd_low = float(logstdBounds[0])
        self.logstd_high = float(logstdBounds[1])

        self.InstallAdaptersMandatory()

        self.dim_base = self.keyboard.base_head.target.out_features
        self.dim_extra = self.keyboard.extra_head.target.out_features
        self.dim_skill = self.keyboard.skill_head.target.out_features
        self.dim_mu = 2
        self.dim_ls = 2
        self.dim_click = 2

        self.psi_to = nn.ModuleDict({
            "base": nn.Sequential( nn.Linear(psiDim, 64), nn.SiLU(),nn.Linear(64, self.dim_base)),
            "extra": nn.Sequential(nn.Linear(psiDim, 64), nn.SiLU(), nn.Linear(64, self.dim_extra)),
            "skill": nn.Sequential(nn.Linear(psiDim, 64), nn.SiLU(),nn.Linear(64, self.dim_skill)),
            "mu": nn.Sequential(nn.Linear(psiDim, 32), nn.SiLU(),nn.Linear(32, self.dim_mu)),
            "logstd": nn.Sequential(nn.Linear(psiDim, 32), nn.SiLU(),nn.Linear(32, self.dim_ls)),
            "click": nn.Sequential(nn.Linear(psiDim, 32), nn.SiLU(),nn.Linear(32, self.dim_click)),})

        K = optionNum
        self.psi_amp = nn.ParameterDict({
            "base": nn.Parameter(torch.full((K, 1), -4.0)),
            "extra": nn.Parameter(torch.full((K, 1), -4.0)),
            "skill": nn.Parameter(torch.full((K, 1), -4.0)),
            "mu": nn.Parameter(torch.full((K, 1), -4.0)),
            "logstd": nn.Parameter(torch.full((K, 1), -4.0)),
            "click": nn.Parameter(torch.full((K, 1), -4.0)),})

        self.g_base = nn.Parameter(torch.tensor(0.5))
        self.g_extra = nn.Parameter(torch.tensor(0.5))
        self.g_skill = nn.Parameter(torch.tensor(0.5))
        self.g_mu = nn.Parameter(torch.tensor(0.5))
        self.g_ls = nn.Parameter(torch.tensor(0.5))  
        self.g_click = nn.Parameter(torch.tensor(0.5))

    def InstallAdaptersMandatory(self):
        def wrap_linear(parent, name: str):
            lin = getattr(parent, name)
            assert isinstance(lin, nn.Linear), f"{name} must be nn.Linear"
            setattr(parent, name, LoRALinearAdapter(lin))

        wrap_linear(self, "to_z")

        wrap_linear(self.keyboard, "base_head")
        wrap_linear(self.keyboard, "skill_head")
        wrap_linear(self.keyboard, "extra_head")

        wrap_linear(self.mouse, "mu_head")
        wrap_linear(self.mouse, "logstd_head")

        wrap_linear(self.mouse.click_head, "0")
        wrap_linear(self.mouse.click_head, "2")

        wrap_linear(self.option, "pi_o")
        wrap_linear(self.option, "psi_head")

        wrap_linear(self.option.beta_head, "0")
        wrap_linear(self.option.beta_head, "2")

        for blk in self.feature_net:
            if isinstance(blk, SwiGLUBlock):
                wrap_linear(blk, "fc1")
                wrap_linear(blk, "fc2")

    @staticmethod
    def Safe(x: torch.Tensor, clip: float = 60.0) -> torch.Tensor:
         return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)

    @staticmethod
    def SafeSoftmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
         return torch.softmax(DecisionExtractor.Safe(logits, 60.0), dim=dim)

    def Encode(self, stateFeat: torch.Tensor) -> torch.Tensor:
        x = self.feature_net(stateFeat)
        x = self.hebb(x, update=(self.use_hebb_online))
        z = F.silu(self.to_z(x))
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
            pressed = x2[:, :max_scan]
            k = min(6, pressed.size(1))
            topk = pressed.topk(k, dim=1).indices
            mask = torch.zeros_like(pressed).scatter(1, topk, 1.0)
            x2 = torch.cat([pressed * mask, x2[:, max_scan:]], dim=1)
        return x2

    def EntropyComponents(self,baseLogits: torch.Tensor,extraLogits: torch.Tensor,skillLogits: torch.Tensor,logstd: torch.Tensor,) -> Dict[str, torch.Tensor]:

        def clean_logits(x):
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            return x.clamp(-60.0, 60.0)

        baseLogits = clean_logits(baseLogits)
        extraLogits = clean_logits(extraLogits)
        skillLogits = clean_logits(skillLogits)

        ent_base = EntropyBernoulliFromLogits(baseLogits)
        ent_extra = EntropyBernoulliFromLogits(extraLogits)

        s = skillLogits - skillLogits.logsumexp(dim=-1, keepdim=True)
        p = s.exp().clamp(1e-12, 1.0) 
        ent_skill = -(p * s).sum(dim=-1)

        ent_mouse = (0.5 * (1.0 + math.log(2 * math.pi)) + logstd).sum(-1)

        n_base = max(1, baseLogits.size(-1))
        n_extra = max(1, extraLogits.size(-1))
        n_skill = max(2, skillLogits.size(-1))
        base_norm = ent_base / n_base
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
        sample: bool = True, # Whether to perform sampling output specific actions
        deterministic: bool = False, # True indicates that the sample will be taken from the mean or a greedy sample; False indicates that the sample will be taken randomly (to increase the exploratory nature of the process)
        prevOptionOnehot: Optional[torch.Tensor] = None,     
        prior: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,  
        mixW: float = 0.25,                              
        returnKeysVec: bool = True,                        
        applyConstraints: bool = True) -> Dict[str, torch.Tensor]:
        
        B = stateFeat.size(0)

        z = self.Encode(stateFeat)
        if prevOptionOnehot is not None:
            prevOptionOnehot = prevOptionOnehot.detach().clone()
        
        option_logits, psi_all, beta = self.option(z, prevOptionOnehot)
        option_logits = self.Safe(option_logits, 60.0)
        psi_all = self.Safe(psi_all, 30.0) 

        p_new = self.SafeSoftmax(option_logits, dim=-1)

        if prevOptionOnehot is not None:
            w_t = (1.0 - beta) * prevOptionOnehot + beta * p_new
        else:
            w_t = p_new

        base_direct, skill_direct, extra_direct = self.keyboard.Logits(z)
        mu_direct, logstd_direct, click_direct = self.mouse.Params(z)

        def mix_psi_per_branch(amp_param: torch.nn.Parameter) -> torch.Tensor:
            amp = torch.sigmoid(amp_param).unsqueeze(0) 
            return (w_t.unsqueeze(-1) * psi_all * amp).sum(dim=1)

        psi_cond_base = mix_psi_per_branch(self.psi_amp["base"])
        psi_cond_extra = mix_psi_per_branch(self.psi_amp["extra"])
        psi_cond_skill = mix_psi_per_branch(self.psi_amp["skill"])
        psi_cond_mu = mix_psi_per_branch(self.psi_amp["mu"])
        psi_cond_logstd  = mix_psi_per_branch(self.psi_amp["logstd"])
        psi_cond_click = mix_psi_per_branch(self.psi_amp["click"])

        base_psi = self.psi_to["base"](psi_cond_base)
        extra_psi = self.psi_to["extra"](psi_cond_extra)
        skill_psi = self.psi_to["skill"](psi_cond_skill)
        mu_psi = self.psi_to["mu"](psi_cond_mu)
        ls_psi = self.psi_to["logstd"](psi_cond_logstd)
        click_psi = self.psi_to["click"](psi_cond_click)

        w_base = F.softplus(self.g_base) / (F.softplus(self.g_base) + 1.0)
        w_extra = F.softplus(self.g_extra) / (F.softplus(self.g_extra) + 1.0)
        w_skill = F.softplus(self.g_skill) / (F.softplus(self.g_skill) + 1.0)
        w_mu = F.softplus(self.g_mu) / (F.softplus(self.g_mu) + 1.0)
        w_ls = F.softplus(self.g_ls) / (F.softplus(self.g_ls) + 1.0)
        w_click = F.softplus(self.g_click) / (F.softplus(self.g_click) + 1.0)

        base_logits = w_base * base_psi + (1.0 - w_base) * base_direct
        extra_logits = w_extra * extra_psi + (1.0 - w_extra) * extra_direct
        skill_logits = w_skill * skill_psi + (1.0 - w_skill) * skill_direct
        mu = w_mu * mu_psi + (1.0 - w_mu) * mu_direct

        std_psi = torch.exp(ls_psi)
        std_dir = torch.exp(logstd_direct)
        var_mix = (w_ls * (std_psi ** 2) + (1.0 - w_ls) * (std_dir ** 2)).clamp_min(1e-12)
        logstd = 0.5 * torch.log(var_mix)
        logstd = ClampLogstd(logstd, self.logstd_low, self.logstd_high)

        click_logits = w_click * click_psi + (1.0 - w_click) * click_direct

        if prior is not None:
            base_logits = MixLogits(base_logits, prior.get("base", {}).get("logits", None), mixW)
            extra_logits = MixLogits(extra_logits, prior.get("extra", {}).get("logits", None), mixW)
            skill_logits = MixLogits(skill_logits, prior.get("skill", {}).get("logits", None), mixW)
            mu, logstd = MixGauss(mu, logstd, prior.get("mouse", {}).get("mu", None), prior.get("mouse", {}).get("var", None), mixW)
            click_logits = MixLogits(click_logits, prior.get("click", {}).get("logits", None), mixW)

        comps = self.EntropyComponents(base_logits, extra_logits, skill_logits, logstd)

        entropy_scalar = self.AggregateEntropy(comps)  

        def sanitize_(x: torch.Tensor, clip: float = 60.0) -> torch.Tensor:
            return torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)

        out: Dict[str, any] = {
            "z": z,
            "entropy": entropy_scalar, 
            "option": {"logits": option_logits, "psi_all": psi_all, "beta": beta},
            "keyboard": {
                "base_logits":  base_logits,
                "skill_logits": skill_logits,
                "extra_logits": extra_logits,},
            "mouse": {
                "mu": mu,
                "logstd": logstd,
                "click_logits": sanitize_(click_logits),},
            "entropy_components": {
                "base": comps["ent_base"], "extra": comps["ent_extra"],
                "skill": comps["ent_skill"], "mouse": comps["ent_mouse"],
                "base_norm": comps["base_norm"], "extra_norm": comps["extra_norm"],
                "skill_norm": comps["skill_norm"], "mouse_norm": comps["mouse_norm"],},}


        if sample:
            base_logits_s = sanitize_(base_logits)
            extra_logits_s = sanitize_(extra_logits)
            skill_logits_s = sanitize_(skill_logits)
            click_logits_s = sanitize_(click_logits)
            option_logits_s = sanitize_(option_logits)
            beta_s = torch.nan_to_num(beta, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1-1e-6)

            if deterministic:
                base_act = (torch.sigmoid(base_logits_s)  > 0.5).float()
                extra_act = (torch.sigmoid(extra_logits_s) > 0.5).float()
                skill_idx = torch.argmax(skill_logits_s, dim=-1)
                clicks = (torch.sigmoid(click_logits_s) > 0.5).float()

                mouse_a = mu

                logp_base = StableLogProbBernoulli(base_logits_s,  base_act)
                logp_extra = StableLogProbBernoulli(extra_logits_s, extra_act)
                logp_skill = torch.distributions.Categorical(logits=skill_logits_s).log_prob(skill_idx)
                
                LOG_TWO_PI = math.log(2.0 * math.pi)
                logp_mouse = -0.5 * (2.0 * logstd + LOG_TWO_PI).sum(-1)
            else:
                base_prob  = torch.sigmoid(base_logits_s).clamp(1e-6, 1.0 - 1e-6)
                extra_prob = torch.sigmoid(extra_logits_s).clamp(1e-6, 1.0 - 1e-6)

                base_act = torch.bernoulli(base_prob)
                extra_act = torch.bernoulli(extra_prob)

                skill_idx = torch.distributions.Categorical(logits=skill_logits_s).sample()

                std = torch.exp(logstd)
                eps = torch.randn_like(std)
                mouse_a = mu + eps * std

                click_prob = torch.sigmoid(click_logits_s).clamp(1e-6, 1.0 - 1e-6)
                clicks = torch.bernoulli(click_prob)

                logp_base = StableLogProbBernoulli(base_logits_s,  base_act)
                logp_extra = StableLogProbBernoulli(extra_logits_s, extra_act)
                logp_skill = torch.distributions.Categorical(logits=skill_logits_s).log_prob(skill_idx)
                dist_mouse = torch.distributions.Normal(mu, std)
                logp_mouse = dist_mouse.log_prob(mouse_a.detach()).sum(-1)

            out["keyboard"].update({
                "base_act": base_act, "extra_act": extra_act, "skill_idx": skill_idx,
                "logp_base": logp_base, "logp_extra": logp_extra, "logp_skill": logp_skill,})
            out["mouse"].update({"a": mouse_a, "logp": logp_mouse, "click_sample": clicks})

            device = z.device
            prev_idx = torch.argmax(prevOptionOnehot, dim=-1) if prevOptionOnehot is not None else torch.zeros(B, dtype=torch.long, device=device)

            if deterministic:
                terminate = (beta_s > 0.5).float()
                new_idx = torch.argmax(option_logits_s, dim=-1)
            else:
                terminate = torch.bernoulli(beta_s)
                new_idx = torch.distributions.Categorical(logits=option_logits_s).sample()

            term_mask = terminate.squeeze(-1).bool()
            opt_idx = torch.where(term_mask, new_idx, prev_idx)

            psi = psi_all[torch.arange(B, device=device), opt_idx]

            dist_opt = torch.distributions.Categorical(logits=option_logits_s)
            logp_new = dist_opt.log_prob(new_idx)
            logp_opt = torch.where(term_mask, logp_new, torch.zeros_like(logp_new))

            b = beta_s.squeeze(-1)
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
                opt_onehot = torch.nn.functional.one_hot(opt_idx, num_classes=self.num_options).float().to(opt_idx.device)
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



class DecisionOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: DecisionExtractor,
        *,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankFeat: int = 64,
        maxRankToZ: int = 64,
        maxRankKbd: int = 64,
        maxRankMouse: int = 64,
        maxRankClick0: int = 32,
        maxRankClick2: int = 32,
        maxRankPi: int = 64,
        maxRankPsi: int = 64,
        maxRankBeta0: int = 32,
        maxRankBeta2: int = 32,
        maxRankTrans: int = 64,):
        self.maxRankFeat = int(maxRankFeat)
        self.maxRankToZ = int(maxRankToZ)
        self.maxRankKbd = int(maxRankKbd)
        self.maxRankMouse = int(maxRankMouse)
        self.maxRankClick0 = int(maxRankClick0)
        self.maxRankClick2 = int(maxRankClick2)
        self.maxRankPi = int(maxRankPi)
        self.maxRankPsi = int(maxRankPsi)
        self.maxRankBeta0 = int(maxRankBeta0)
        self.maxRankBeta2 = int(maxRankBeta2)
        self.maxRankTrans = int(maxRankTrans)
        super().__init__(
            base=base,
            initRankEach=initRankEach,
            autoRank=autoRank,
            evThreshold=evThreshold,
            gradEma=gradEma,)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        def alloc_lin(addRank, device, dtype, inDim, outDim):
            A = nn.Parameter(torch.randn(addRank, inDim,  device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype) * 1e-4)
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_lin(a, b, s):
            return torch.tanh(s) * GetParameterSScale(s) * (b @ a)

        def alloc_mat(addRank, device, dtype, N, M):
            A = nn.Parameter(torch.randn(addRank, N, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(M, addRank, device=device, dtype=dtype) * 1e-4)
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_mat(a, b, s):
            return torch.tanh(s) * GetParameterSScale(s) * (b @ a)

        L = 1

        toz_in = self.base.to_z.target.in_features
        toz_out = self.base.to_z.target.out_features

        kbd_base_in = self.base.keyboard.base_head.target.in_features
        kbd_base_out = self.base.keyboard.base_head.target.out_features
        kbd_skill_in = self.base.keyboard.skill_head.target.in_features
        kbd_skill_out = self.base.keyboard.skill_head.target.out_features
        kbd_extra_in = self.base.keyboard.extra_head.target.in_features
        kbd_extra_out = self.base.keyboard.extra_head.target.out_features

        mouse_mu_in = self.base.mouse.mu_head.target.in_features
        mouse_mu_out = self.base.mouse.mu_head.target.out_features
        mouse_ls_in = self.base.mouse.logstd_head.target.in_features
        mouse_ls_out = self.base.mouse.logstd_head.target.out_features

        click0_in = self.base.mouse.click_head[0].target.in_features
        click0_out = self.base.mouse.click_head[0].target.out_features
        click2_in = self.base.mouse.click_head[2].target.in_features
        click2_out = self.base.mouse.click_head[2].target.out_features

        K = self.base.option.K
        opt_pi_in = self.base.option.pi_o.target.in_features
        opt_pi_out = self.base.option.pi_o.target.out_features
        opt_psi_in = self.base.option.psi_head.target.in_features
        opt_psi_out = self.base.option.psi_head.target.out_features
        opt_b0_in = self.base.option.beta_head[0].target.in_features
        opt_b0_out = self.base.option.beta_head[0].target.out_features
        opt_b2_in = self.base.option.beta_head[2].target.in_features
        opt_b2_out = self.base.option.beta_head[2].target.out_features

        specs = {
            "toz": SiteSpec("toz", L, toz_in, toz_out, self.maxRankToZ, lambda r, d, dt: alloc_lin(r, d, dt, toz_in, toz_out), compose_lin),

            "kbd_base": SiteSpec("kbd_base", L, kbd_base_in, kbd_base_out, self.maxRankKbd, lambda r, d, dt: alloc_lin(r, d, dt, kbd_base_in, kbd_base_out), compose_lin),
            "kbd_skill": SiteSpec("kbd_skill", L, kbd_skill_in, kbd_skill_out, self.maxRankKbd, lambda r, d, dt: alloc_lin(r, d, dt, kbd_skill_in, kbd_skill_out), compose_lin),
            "kbd_extra": SiteSpec("kbd_extra", L, kbd_extra_in, kbd_extra_out, self.maxRankKbd, lambda r, d, dt: alloc_lin(r, d, dt, kbd_extra_in, kbd_extra_out), compose_lin),

            "mouse_mu": SiteSpec("mouse_mu", L, mouse_mu_in, mouse_mu_out, self.maxRankMouse, lambda r, d, dt: alloc_lin(r, d, dt, mouse_mu_in, mouse_mu_out), compose_lin),
            "mouse_ls": SiteSpec("mouse_ls", L, mouse_ls_in, mouse_ls_out, self.maxRankMouse, lambda r, d, dt: alloc_lin(r, d, dt, mouse_ls_in, mouse_ls_out), compose_lin),

            "click0": SiteSpec("click0", L, click0_in, click0_out, self.maxRankClick0, lambda r, d, dt: alloc_lin(r, d, dt, click0_in, click0_out), compose_lin),
            "click2": SiteSpec("click2", L, click2_in, click2_out, self.maxRankClick2, lambda r, d, dt: alloc_lin(r, d, dt, click2_in, click2_out), compose_lin),

            "opt_pi": SiteSpec("opt_pi", L, opt_pi_in, opt_pi_out, self.maxRankPi, lambda r, d, dt: alloc_lin(r, d, dt, opt_pi_in, opt_pi_out), compose_lin),
            "opt_psi": SiteSpec("opt_psi", L, opt_psi_in, opt_psi_out, self.maxRankPsi, lambda r, d, dt: alloc_lin(r, d, dt, opt_psi_in, opt_psi_out), compose_lin),
            "opt_beta0": SiteSpec("opt_beta0", L, opt_b0_in, opt_b0_out, self.maxRankBeta0, lambda r, d, dt: alloc_lin(r, d, dt, opt_b0_in, opt_b0_out), compose_lin),
            "opt_beta2": SiteSpec("opt_beta2", L, opt_b2_in, opt_b2_out, self.maxRankBeta2, lambda r, d, dt: alloc_lin(r, d, dt, opt_b2_in, opt_b2_out), compose_lin),

            "opt_trans": SiteSpec("opt_trans", L, K, K, min(self.maxRankTrans, K), lambda r, d, dt: alloc_mat(r, d, dt, K, K), compose_mat),}

        for i, blk in enumerate(self.base.feature_net):
            if isinstance(blk, SwiGLUBlock):
                fc1_in, fc1_out = blk.fc1.target.in_features, blk.fc1.target.out_features
                fc2_in, fc2_out = blk.fc2.target.in_features, blk.fc2.target.out_features
                specs[f"feat{i}_fc1"] = SiteSpec(
                    f"feat{i}_fc1", L, fc1_in, fc1_out, self.maxRankFeat,
                    lambda r, d, dt, _in=fc1_in, _out=fc1_out: alloc_lin(r, d, dt, _in, _out),
                    compose_lin,)
                specs[f"feat{i}_fc2"] = SiteSpec(
                    f"feat{i}_fc2", L, fc2_in, fc2_out, self.maxRankFeat,
                    lambda r, d, dt, _in=fc2_in, _out=fc2_out: alloc_lin(r, d, dt, _in, _out),
                    compose_lin,)

        return specs

    @torch.no_grad()
    def ForwardWithDeltas(
        self,
        x: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]],
        **kwargs,) -> Dict[str, Any]:

        D = deltasPerLayer[0] if (deltasPerLayer and len(deltasPerLayer) > 0) else {}

        prior: Optional[Dict[str, Dict[str, torch.Tensor]]] = kwargs.pop("prior", None)
        mixW: float = float(kwargs.pop("mixW", 0.25))
        prevOptionOnehot: Optional[torch.Tensor] = kwargs.pop("prevOptionOnehot", None)
        if (prevOptionOnehot is None) and (keyPaddingMask is not None):
            prevOptionOnehot = keyPaddingMask

        B = x.size(0)
        device = x.device
        K = self.base.option.K

        has_prev = (prevOptionOnehot is not None) and (prevOptionOnehot.dim() == 2) and (prevOptionOnehot.size(1) == K)
        prev = (prevOptionOnehot.detach().to(dtype=x.dtype, device=device) if has_prev else torch.zeros(B, K, dtype=x.dtype, device=device))

        for i, blk in enumerate(self.base.feature_net):
            if isinstance(blk, SwiGLUBlock):
                h_norm = blk.ln(x)

                fc1_out = blk.fc1(h_norm) 
                d_fc1 = D.get(f"feat{i}_fc1", None)
                if d_fc1 is not None:
                    fc1_out = fc1_out + F.linear(h_norm, d_fc1, bias=None)

                a, b = fc1_out.chunk(2, dim=-1)
                mid = F.silu(a) * b

                fc2_out = blk.fc2(mid)  
                d_fc2 = D.get(f"feat{i}_fc2", None)
                if d_fc2 is not None:
                    fc2_out = fc2_out + F.linear(mid, d_fc2, bias=None)

                x = x + blk.drop(fc2_out * blk.gamma)
            else:
                x = blk(x)

        x = self.base.hebb(x, update=self.base.use_hebb_online)

        z_lin = self.base.to_z(x)

        if D.get("toz") is not None:
            z_lin = z_lin + F.linear(x, D["toz"], bias=None)
        z = F.silu(z_lin)

        h_k = self.base.keyboard.backbone(z)

        base_logits = self.base.keyboard.base_head(h_k)
        skill_logits = self.base.keyboard.skill_head(h_k)
        extra_logits = self.base.keyboard.extra_head(h_k)

        if D.get("kbd_base") is not None: base_logits = base_logits + F.linear(h_k, D["kbd_base"], bias=None)
        if D.get("kbd_skill") is not None: skill_logits = skill_logits + F.linear(h_k, D["kbd_skill"], bias=None)
        if D.get("kbd_extra") is not None: extra_logits = extra_logits + F.linear(h_k, D["kbd_extra"], bias=None)

        h_m = self.base.mouse.backbone(z)

        mu = self.base.mouse.mu_head(h_m)
        logstd = self.base.mouse.logstd_head(h_m)
        if D.get("mouse_mu") is not None: mu = mu + F.linear(h_m, D["mouse_mu"], bias=None)
        if D.get("mouse_ls") is not None: logstd = logstd + F.linear(h_m, D["mouse_ls"], bias=None)
        logstd = torch.clamp(logstd, self.base.logstd_low, self.base.logstd_high)

        c0 = self.base.mouse.click_head[0](h_m)
        if D.get("click0") is not None: c0 = c0 + F.linear(h_m, D["click0"], bias=None)
        c0 = F.relu(c0)

        click_logits = self.base.mouse.click_head[2](c0)
        if D.get("click2") is not None: click_logits = click_logits + F.linear(c0, D["click2"], bias=None)

        h_o = self.base.option.enc(z)

        opt_logits_base = self.base.option.pi_o(h_o)
        if D.get("opt_pi") is not None:
            opt_logits_base = opt_logits_base + F.linear(h_o, D["opt_pi"], bias=None)

        psi_flat = self.base.option.psi_head(h_o)
        if D.get("opt_psi") is not None:
            psi_flat = psi_flat + F.linear(h_o, D["opt_psi"], bias=None)

        K = self.base.option.K

        psi_all = psi_flat.view(-1, K, self.base.option.psiDim)
        psi_all = psi_all * self.base.option.psi_amp_global * self.base.option.psi_amp_per_option.view(1, K, 1)
        psi_all = self.base.Safe(psi_all, 30.0)

        b0_in = torch.cat([h_o, (prev if has_prev else torch.zeros_like(prev))], dim=-1)

        b0 = self.base.option.beta_head[0](b0_in)
        if D.get("opt_beta0") is not None:
            b0 = b0 + F.linear(b0_in, D["opt_beta0"], bias=None)
        b0 = F.relu(b0)
        beta = self.base.option.beta_head[2](b0)
        if D.get("opt_beta2") is not None:
            beta = beta + F.linear(b0, D["opt_beta2"], bias=None)
        beta = torch.sigmoid(beta).clamp(1e-6, 1.0 - 1.0e-6)

        trans_eff = self.base.option.trans_adapter(self.base.option.trans)
        if D.get("opt_trans") is not None:
            trans_eff = trans_eff + D["opt_trans"]

        trans_eff = torch.nan_to_num(trans_eff, nan=0.0).clamp(-10.0, 10.0)

        option_logits = self.base.Safe(opt_logits_base + (prev @ trans_eff if has_prev else 0.0), 60.0)

        p_new = self.base.SafeSoftmax(option_logits, dim=-1)

        w_t = (1.0 - beta) * prev + beta * p_new if has_prev else p_new

        sp = F.softplus
        def mix_psi(amp_param: torch.nn.Parameter) -> torch.Tensor:
            amp = torch.sigmoid(amp_param).unsqueeze(0) 
            return (w_t.unsqueeze(-1) * psi_all * amp).sum(dim=1)

        psi_cond_base = mix_psi(self.base.psi_amp["base"])
        psi_cond_extra = mix_psi(self.base.psi_amp["extra"])
        psi_cond_skill = mix_psi(self.base.psi_amp["skill"])
        psi_cond_mu = mix_psi(self.base.psi_amp["mu"])
        psi_cond_ls = mix_psi(self.base.psi_amp["logstd"])
        psi_cond_click = mix_psi(self.base.psi_amp["click"])

        base_psi = self.base.psi_to["base"](psi_cond_base)
        extra_psi = self.base.psi_to["extra"](psi_cond_extra)
        skill_psi = self.base.psi_to["skill"](psi_cond_skill)
        mu_psi = self.base.psi_to["mu"](psi_cond_mu)
        ls_psi = self.base.psi_to["logstd"](psi_cond_ls)
        click_psi = self.base.psi_to["click"](psi_cond_click)

        def gate(gparam: torch.nn.Parameter) -> torch.Tensor:
            s = sp(gparam)
            return s / (s + 1.0)

        w_base = gate(self.base.g_base)
        w_extra = gate(self.base.g_extra)
        w_skill = gate(self.base.g_skill)
        w_mu = gate(self.base.g_mu)
        w_ls = gate(self.base.g_ls)
        w_click = gate(self.base.g_click)

        base_logits = w_base * base_psi + (1.0 - w_base) * base_logits
        extra_logits = w_extra * extra_psi + (1.0 - w_extra) * extra_logits
        skill_logits = w_skill * skill_psi + (1.0 - w_skill) * skill_logits

        mu = w_mu * mu_psi + (1.0 - w_mu) * mu

        std_psi = torch.exp(ls_psi)
        std_dir = torch.exp(logstd)
        var_mix = (w_ls * (std_psi ** 2) + (1.0 - w_ls) * (std_dir ** 2)).clamp_min(1e-12)
        logstd = 0.5 * torch.log(var_mix)
        logstd = torch.clamp(logstd, self.base.logstd_low, self.base.logstd_high)

        click_logits = w_click * click_psi + (1.0 - w_click) * click_logits
        
        click_logits = self.base.Safe(click_logits, 60.0)

        if prior is not None:
            base_logits = MixLogits(base_logits,  prior.get("base",  {}).get("logits", None), mixW)
            extra_logits = MixLogits(extra_logits, prior.get("extra", {}).get("logits", None), mixW)
            skill_logits = MixLogits(skill_logits, prior.get("skill", {}).get("logits", None), mixW)
            mu, logstd = MixGauss(mu, logstd, prior.get("mouse", {}).get("mu",  None), prior.get("mouse", {}).get("var", None), mixW)
            
            click_logits = MixLogits(click_logits, prior.get("click", {}).get("logits", None), mixW)
 

        comps = self.base.EntropyComponents(base_logits, extra_logits, skill_logits, logstd)
        entropy_scalar = self.base.AggregateEntropy(comps)

        return {
            "z": z,
            "entropy": entropy_scalar,
            "entropy_components": {
                "base": comps["ent_base"], "extra": comps["ent_extra"],
                "skill": comps["ent_skill"], "mouse": comps["ent_mouse"],
                "base_norm": comps["base_norm"], "extra_norm": comps["extra_norm"],
                "skill_norm": comps["skill_norm"], "mouse_norm": comps["mouse_norm"],},
            "keyboard": {
                "base_logits":  base_logits,
                "skill_logits": skill_logits,
                "extra_logits": extra_logits,},
            "mouse": {
                "mu": mu,
                "logstd": logstd,
                "click_logits": click_logits,},
            "option": {
                "logits": option_logits,
                "psi_all": psi_all,
                "beta": beta,},}

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        if layerIdx != 0:
            return False

        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "toz":
            self.base.to_z.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        if site == "kbd_base":
            self.base.keyboard.base_head.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        if site == "kbd_skill":
            self.base.keyboard.skill_head.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        if site == "kbd_extra":
            self.base.keyboard.extra_head.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        if site == "mouse_mu":
            self.base.mouse.mu_head.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        if site == "mouse_ls":
            self.base.mouse.logstd_head.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        if site == "click0":
            self.base.mouse.click_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        if site == "click2":
            self.base.mouse.click_head[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        if site == "opt_pi":
            self.base.option.pi_o.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        if site == "opt_psi":
            self.base.option.psi_head.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        if site == "opt_beta0":
            self.base.option.beta_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        if site == "opt_beta2":
            self.base.option.beta_head[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        if site == "opt_trans":
            self.base.option.trans_adapter.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        
        if site.startswith("feat") and ("_fc" in site):
            head, tail = site.split("_", 1)   
            idx = int(head.replace("feat", ""))
            blk = self.base.feature_net[idx]
            getattr(blk, tail).Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True

        raise ValueError(f"Unknown site: {site}")



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
        self.register_buffer("base_codes_buf", torch.tensor(baseCodes, dtype=torch.long))
        self.register_buffer("skill_codes_buf", torch.tensor(skillCodes, dtype=torch.long))
        self.register_buffer("extra_codes_buf", torch.tensor(extraCodes, dtype=torch.long))

        self.n_base = self.base_codes_buf.numel()
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

            def prob_(x): return torch.sigmoid(x).clamp(1e-6, 1.0 - 1e-6)

            pb, pe, pc = prob_(logits_b), prob_(logits_e), prob_(logits_c)

            base_seq = (torch.rand(H, B, N, self.n_base, device=device) < pb.unsqueeze(2)).float()
            extra_seq = (torch.rand(H, B, N, self.n_extra, device=device) < pe.unsqueeze(2)).float()
            click_seq = (torch.rand(H, B, N, 2,device=device) < pc.unsqueeze(2)).float()

            h = h_prev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
            z = z_prev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()

            score = torch.zeros(B, N, device=device)
            cont = torch.ones(B, N, device=device)

            for t in range(H):
                a_mouse_t = mouse_seq[t].reshape(B * N, 2)
                a_skill_t = skill_seq[t].reshape(B * N)
                a_base_t = base_seq[t].reshape(B * N, self.n_base)
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
                cont = cont * (1.0 - d_t)


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
                mu_new = (w_exp * elite_mouse_t).sum(dim=1)
                diff = elite_mouse_t - mu_new.unsqueeze(1)
                var_new = (w_exp * (diff * diff)).sum(dim=1).clamp_min(self.min_var)
                std_new = var_new.sqrt()
                mu_t[t] = self.momentum * mu_t[t] + (1 - self.momentum) * mu_new
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

        mouse_mu0 = mu_t[0]
        mouse_var0 = (std_t[0] * std_t[0])
        out = {
            "mouse": {"mu": mouse_mu0, "var": mouse_var0},
            "skill": {"logits": logits_s[0]},
            "base": {"logits": logits_b[0]},
            "extra": {"logits": logits_e[0]},
            "click": {"logits": logits_c[0]}}

        if returnTrajectories:
            out["diagnostics"] = {
                "mu_seq": mu_t, "std_seq": std_t,
                "skill_logits_seq": logits_s,
                "base_logits_seq": logits_b,
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
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(0)

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

        self.num_base = len(self.base_codes)
        self.num_skill = len(self.skill_names) + 1 
        self.num_extra = len(self.extra_codes)
        self.keyvec_dim = self.max_code + 1 + 2

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

        def ResetHidden(self, B: int = 1, device: torch.device | None = None):
            if device is None:
                device = self._h.device
            self._h = torch.zeros(B, self.deter_dim, device=device)
            self._z = torch.zeros(B, self.stoch_dim, device=device)

        def ExportState(self):
            return self._h, self._z

        @torch.no_grad()
        def StepPriorOnly(self, hPrev, zPrev, aEnc, sample: bool = False):
            a = self.act_proj(aEnc)
            h_next = self.gru(torch.cat([zPrev, a], dim=-1), hPrev)
            mu_p, logstd_p = self.prior_head(h_next).chunk(2, dim=-1)
            logstd_p = torch.clamp(logstd_p, -6.0, 2.0)
            z_next = mu_p
            s_next = self.state_proj(torch.cat([h_next, z_next], dim=-1))
            r_pred = self.rew_head(s_next).squeeze(-1)
            d_prob = torch.sigmoid(self.done_head(s_next)).squeeze(-1)
            return h_next, z_next, s_next, r_pred, d_prob

    def BuildPlanner(self, horizon=3, N=16, elite=4, iters=2):
        try:
            wm = self.MockWorldModel(keyDim=self.keyvec_dim, actionDim=128, deterDim=128, stochDim=16, stateDim=128).to(self.device)
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
            print("BuildPlanner error:", type(e).__name__, e)
            return None, None

    def DecisionOnlyLoss(self, out: Dict[str, Any], adv: Optional[Dict[str, torch.Tensor]] = None, entCoef: float = 0.0, *, returnBreakdown: bool = False,) -> torch.Tensor | Dict[str, Any]:
        device = out["mouse"]["mu"].device
        B = out["mouse"]["mu"].size(0)
        adv = adv or {}

        def adv_(name: str, shape) -> torch.Tensor:
            a = adv.get(name, None)
            if a is None:
                a = torch.randn(shape, device=device)
            return a.detach()

        kb, ms, op = out["keyboard"], out["mouse"], out["option"]

        loss_core = torch.tensor(0.0, device=device)
        terms: Dict[str, torch.Tensor] = {}

        if "logp_base" in kb:
            t = -(adv_("base", (B,)) * kb["logp_base"]).mean()
            terms["base"] = t; loss_core = loss_core + t
        if "logp_extra" in kb:
            t = -(adv_("extra", (B,)) * kb["logp_extra"]).mean()
            terms["extra"] = t; loss_core = loss_core + t
        if "logp_skill" in kb:
            t = -(adv_("skill", (B,)) * kb["logp_skill"]).mean()
            terms["skill"] = t; loss_core = loss_core + t

        if "logp" in ms:
            t = -(adv_("mouse", (B,)) * ms["logp"]).mean()
            terms["mouse"] = t; loss_core = loss_core + t

        if ("click_logits" in ms) and ("click_sample" in ms):
            logp_click = StableLogProbBernoulli(ms["click_logits"], ms["click_sample"])
            t = -(adv_("click", (B,)) * logp_click).mean()
            terms["click"] = t; loss_core = loss_core + t

        if "logp_option" in op:
            t = -(adv_("option", (B,)) * op["logp_option"]).mean()
            terms["option"] = t; loss_core = loss_core + t
        if "logp_beta" in op:
            t = -(adv_("beta", (B,)) * op["logp_beta"]).mean()
            terms["beta"] = t; loss_core = loss_core + t

        ent_term = torch.tensor(0.0, device=device)
        if entCoef != 0.0 and ("entropy" in out):
            ent_term = - entCoef * out["entropy"].mean()

        loss = loss_core + ent_term

        if not returnBreakdown:
            return loss

        breakdown = {k: float(v.detach()) for k, v in terms.items()}
        return {
            "total": float(loss.detach()),
            "core": float(loss_core.detach()),
            "entropy": float(ent_term.detach()),
            "terms": breakdown,}

    def TestHebbLayer(self) -> bool:
        try:
            layer = HebbianPlasticityLayer(128, 64).to(self.device)
            x = torch.randn(8, 128, device=self.device)
            y0 = layer(x, update=False)
            y1 = layer(x, update=True)
            if y0.shape != (8, 64) or y1.shape != (8, 64):
                print("HebbianPlasticityLayer shape does not match")
                return False
            with torch.no_grad():
                changed = layer.hebb.abs().sum().item() > 0
            if not changed:
                print("The Hebb buffer is not updated online")
                return False
            print("HebbianPlasticityLayer pass")
            return True
        except Exception as e:
            print("HebbianPlasticityLayer error:", type(e).__name__, e)
            return False

    def TestLoraLinearAdapter(self) -> bool:
        try:
            lin = nn.Linear(32, 16).to(self.device)
            adap = LoRALinearAdapter(lin).to(self.device)
            x = torch.randn(4, 32, device=self.device)
            y_base = adap(x)
            adap.Grow(addRank=3)
            y_aug = adap(x)
            A, B, s = adap.A_list[0], adap.B_list[0], adap.alpha[0]
            s_eff = torch.tanh(s) * GetParameterSScale(s)
            delta = F.linear(x, s_eff * (B @ A), bias=None)
            diff = (y_aug - y_base - delta).abs().max().item()
            if diff >= 1e-5:
                print(f"LoRA linear increment inconsistency: max|Δ|={diff:.2e}")
                return False
            print("LoRALinearAdapter.Grow & forward pass")
            return True
        except Exception as e:
            print("LoRALinearAdapter error:", type(e).__name__, e)
            return False 

    def TestMatloraAdapter(self) -> bool:
        try:
            M, N = 10, 7
            base = torch.randn(M, N, device=self.device)
            adap = MatLoRAAdapter(M, N).to(self.device)

            out0 = adap(base)

            adap.Grow(addRank=1)
            adap.Grow(addRank=1, freezeOld=True)

            out1 = adap(base)

            A0, B0, s0 = adap.A_list[0], adap.B_list[0], adap.alpha[0]
            A1, B1, s1 = adap.A_list[1], adap.B_list[1], adap.alpha[1]
            s0_eff = torch.tanh(s0) * GetParameterSScale(s0)
            s1_eff = torch.tanh(s1) * GetParameterSScale(s1)
            expect = base + s0_eff * (B0 @ A0) + s1_eff * (B1 @ A1)

            err = (out1 - expect).abs().max().item()
            if out0.shape != base.shape or err >= 1e-5:
                print(f"MatLoRAAdapter inconsistency: err={err:.2e}")
                return False
            print("MatLoRAAdapter.Grow & forward pass")
            return True
        except Exception as e:
            print("MatLoRAAdapter error:", type(e).__name__, e)
            return False

    def TestDecisionForwardShapes(self) -> bool:
        try:
            model = DecisionExtractor(stateDim=1024, includeNoSkill=True, useHebb=False).to(self.device)
            model.eval()
            B = 3
            x = torch.randn(B, 1024, device=self.device)
            out = model(x, sample=False, prior=None, returnKeysVec=False, applyConstraints=True)
            kb, ms, opt = out["keyboard"], out["mouse"], out["option"]
            checks = [
                kb["base_logits"].shape  == (B, self.num_base),
                kb["skill_logits"].shape == (B, self.num_skill),
                kb["extra_logits"].shape == (B, self.num_extra),
                ms["mu"].shape == (B, 2),
                ms["logstd"].shape == (B, 2),
                ms["click_logits"].shape == (B, 2),
                opt["psi_all"].shape == (B, model.num_options, model.option.psiDim),]
            if not all(checks):
                print("DecisionExtractor forward output dimension does not match")
                return False
            out2 = model(x, sample=True, deterministic=False, prior=None, returnKeysVec=True, applyConstraints=True)
            key_vec = out2["key_vec"]
            if key_vec.shape != (B, self.keyvec_dim):
                print("key_vec shape does not match")
                return False
            pressed = (key_vec[:, :self.max_code+1] > 0.5).sum(-1).max().item()
            if pressed > 6:
                print(f"Constraint failed: Number of keys={pressed}")
                return False
            print("DecisionExtractor Forward/Sampling/Constraint pass")
            return True
        except Exception as e:
            print("DecisionExtractor forward error:", type(e).__name__, e)
            return False

    def TestOptionPrevAndTrans(self) -> bool:
        try:
            model = DecisionExtractor(stateDim=256, includeNoSkill=True, useHebb=False).to(self.device)
            model.eval()
            K = model.num_options
            model.option.trans_adapter.Grow(addRank=1)

            x = torch.randn(2, 256, device=self.device)
            prev0 = torch.zeros(2, K, device=self.device)
            prev1 = torch.zeros(2, K, device=self.device); prev1[:, 0] = 1.0

            out0 = model(x, sample=False, prevOptionOnehot=prev0, returnKeysVec=False)
            out1 = model(x, sample=False, prevOptionOnehot=prev1, returnKeysVec=False)

            with torch.no_grad():
                h = model.option.enc(F.silu(model.to_z(model.hebb(model.feature_net(x), update=False))))
                trans_eff = model.option.trans_adapter(model.option.trans)
                expect = prev1 @ trans_eff
            diff = (out1["option"]["logits"] - out0["option"]["logits"] - expect).abs().max().item()
            if diff >= 1e-5:
                print(f"prev@trans_eff mismatched functions: diff={diff:.2e}")
                return False
            print("OptionPolicy prev/transferMatrix pass")
            return True
        except Exception as e:
            print("OptionPolicy error:", type(e).__name__, e)
            return False

    def TestCemPlanner(self) -> bool:
        try:
            _, planner = self.BuildPlanner()
            if planner is None:
                return False
            prior = planner.Plan(returnTrajectories=False)
            ok = True
            ok &= prior["mouse"]["mu"].shape == (2, 2)
            ok &= prior["mouse"]["var"].shape == (2, 2)
            ok &= prior["skill"]["logits"].shape == (2, self.num_skill)
            ok &= prior["base"]["logits"].shape == (2, self.num_base)
            ok &= prior["extra"]["logits"].shape == (2, self.num_extra)
            if not ok:
                print("CEMPlanner output shape does not match")
                return False
            print("CEMPlanner.Plan pass")
            return True
        except Exception as e:
            print("CEMPlanner error:", type(e).__name__, e)
            return False

    def TestForwardWithDeltasInjection(self) -> bool:
        try:
            model = DecisionExtractor(stateDim=128, includeNoSkill=True, useHebb=False).to(self.device)
            wrapper = DecisionOnlineWrapper(model, initRankEach=0, autoRank=False)
            model.eval()
            B = 4
            x = torch.randn(B, 128, device=self.device)

            h = model.keyboard.backbone(F.silu(model.to_z(model.hebb(model.feature_net(x), update=False))))
            out_dim = model.keyboard.base_head.target.out_features
            in_dim = model.keyboard.base_head.target.in_features
            deltaW = torch.randn(out_dim, in_dim, device=self.device) * 1e-3
            D = {"kbd_base": deltaW}

            outD = wrapper.ForwardWithDeltas(x, keyPaddingMask=None, tdError=None, uncertainty=None, deltasPerLayer=[D])
            out0 = wrapper.ForwardWithDeltas(x, keyPaddingMask=None, tdError=None, uncertainty=None, deltasPerLayer=[{}])

            sp = F.softplus
            w_base = sp(model.g_base) / (sp(model.g_base) + 1.0) 
            expect = (1.0 - w_base) * F.linear(h, deltaW, bias=None)
            
            err = (outD["keyboard"]["base_logits"] - out0["keyboard"]["base_logits"] - expect).abs().max().item()
            if err >= 1e-5:
                print(f"ForwardWithDeltas injection mismatch: err={err:.2e}")
                return False
            print("DecisionOnlineWrapper.ForwardWithDeltas injection pass")
            return True
        except Exception as e:
            print("ForwardWithDeltas error:", type(e).__name__, e)
            return False

    def TestCommitOneGrowsLora(self) -> bool:
        try:
            model = DecisionExtractor(stateDim=128, includeNoSkill=True, useHebb=False).to(self.device)
            wrapper = DecisionOnlineWrapper(model, initRankEach=0, autoRank=False)

            tgt = model.keyboard.base_head
            r_before = len(tgt.A_list)
            in_dim = tgt.target.in_features
            out_dim = tgt.target.out_features
            addRank = 2
            A = torch.randn(addRank, in_dim, device=self.device) * 1e-4
            B = torch.zeros(out_dim, addRank, device=self.device) * 1e-4

            x = torch.randn(5, 128, device=self.device)
            h = model.keyboard.backbone(F.relu(model.to_z(model.hebb(model.feature_net(x), update=False))))
            with torch.no_grad():
                y_base = model.keyboard.base_head(h).detach()

            ok = wrapper.CommitOne("kbd_base", layerIdx=0, a=A, b=B, scale=1e-3)
            if not ok or len(tgt.A_list) != r_before + 1:
                print("CommitOne did not increase LoRA rank")
                return False

            y_after = model.keyboard.base_head(h)

            A_new, B_new, s_new = tgt.A_list[-1], tgt.B_list[-1], tgt.alpha[-1]

            s_eff = torch.tanh(s_new) * GetParameterSScale(s_new)
            expect_delta = F.linear(h, s_eff * (B_new @ A_new), bias=None)

            err = (y_after - y_base - expect_delta).abs().max().item()
            if err >= 1e-5:
                print(f"CommitOne increment value does not match: err={err:.2e}")
                return False
            print("CommitOne -> LoRA growth and value verification pass")
            return True
        except Exception as e:
            print("CommitOne error:", type(e).__name__, e)
            return False

    def TestActionsOnly(self, steps: int = 80) -> bool:
        try:
            in_dim, B = 512, 32
            torch.manual_seed(123)

            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebb=False).to(self.device)
            model.train()

            model.option.trans_adapter.Grow(addRank=2, freezeOld=False)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            x_fix = torch.randn(B, in_dim, device=self.device)
            K = model.num_options
            prev_fix = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            with torch.no_grad():
                out0 = model(x_fix, sample=True, deterministic=False, prior=None, prevOptionOnehot=prev_fix, returnKeysVec=False)
                start_loss = self.DecisionOnlyLoss(out0, entCoef=0.0).item()

            for t in range(steps):
                x = torch.randn(B, in_dim, device=self.device)
                prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

                self.ZeroAllGrads(model)
                out = model(x, sample=True, deterministic=False, prior=None, prevOptionOnehot=prev, returnKeysVec=False)
                adv = {
                    "option": torch.full((B,), 1.5, device=self.device),
                    "beta": torch.full((B,), 0.5, device=self.device),
                    "skill": torch.ones(B, device=self.device),
                    "base": torch.ones(B, device=self.device),
                    "extra": torch.ones(B, device=self.device),
                    "mouse": torch.ones(B, device=self.device),
                    "click": torch.ones(B, device=self.device),}
                
                loss = self.DecisionOnlyLoss(out, adv=adv, entCoef=0.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                def g(p): 
                    return None if (p.grad is None) else float(p.grad.norm().item())

                print("β bias grad:", g(model.option.beta_head[-1].target.bias))
                print("trans base grad:", g(model.option.trans))

                ta = model.option.trans_adapter
                if len(ta.A_list) > 0:
                    print("trans LoRA A grad:", g(ta.A_list[-1]))
                    print("trans LoRA B grad:", g(ta.B_list[-1]))
                    print("trans LoRA s grad:", g(ta.alpha[-1]))

                for name, seq in model.psi_to.items():
                    for m in seq.modules():
                        if isinstance(m, nn.Linear):
                            print(f"psi_to[{name}] weight grad:", g(m.weight))

                if t == 0:
                    def any_grad_lora(adp):
                        base_has = self.HasGrad(adp.target.weight) or (adp.target.bias is not None and self.HasGrad(adp.target.bias))
                        lora_has = any(self.HasGrad(p) for p in list(adp.A_list) + list(adp.B_list) + list(adp.alpha))
                        return base_has or lora_has

                    ok_pi = any_grad_lora(model.option.pi_o)
                    ok_psi = any_grad_lora(model.option.psi_head)
                    ok_b0 = any_grad_lora(model.option.beta_head[0])
                    ok_b2 = any_grad_lora(model.option.beta_head[2])

                    ok_trans_base = self.HasGrad(model.option.trans)
                    ta = model.option.trans_adapter
                    ok_trans_lora = (len(ta.A_list) > 0) and any(self.HasGrad(p) for p in list(ta.A_list) + list(ta.B_list) + list(ta.alpha))

                    def has_seq_grad(seq):
                        oks = []
                        for m in seq.modules():
                            if isinstance(m, nn.Linear):
                                oks.append(self.HasGrad(m.weight))
                        return all(oks) and (len(oks) > 0)

                    ok_psito = all(has_seq_grad(seq) for seq in model.psi_to.values())

                    if not (ok_pi and ok_psi and ok_b0 and ok_b2 and ok_trans_base and ok_trans_lora and ok_psito):
                        print("Key option/psi/trans path missing gradient under decision-only loss: ",
                              dict(pi=ok_pi, psi=ok_psi, beta0=ok_b0, beta2=ok_b2, trans_base=ok_trans_base, trans_lora=ok_trans_lora, psi_to=ok_psito))
                        return False

                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"Non-finite gradient: {n}")
                        return False

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            with torch.no_grad():
                out1 = model(x_fix, sample=True, deterministic=False, prior=None, prevOptionOnehot=prev_fix, returnKeysVec=False)
                end_loss = self.DecisionOnlyLoss(out1, entCoef=0.0).item()

            print(f"[ActionsOnly(decision-only)] loss {start_loss:.4f} -> {end_loss:.4f}")
            ok_improve = (end_loss <= 0.8 * start_loss) or ((start_loss - end_loss) >= 0.05)
            if not ok_improve:
                print("Fixed batch drop insufficiency")
                return False

            print("Decision-only training: option/psi/beta/trans_adapter get gradients and loss decreases, pass")
            return True

        except Exception as e:
            print("Actions-only (decision-only) test error:", type(e).__name__, e)
            return False

    def TestTrainStepSmoke(self) -> bool:
        try:
            in_dim = 512
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebb=False).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            B = 8
            x = torch.randn(B, in_dim, device=self.device)
            K = model.num_options
            prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            out = model(x, sample=True, deterministic=False, prior=None, prevOptionOnehot=prev, returnKeysVec=False)
            loss = self.DecisionOnlyLoss(out, entCoef=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if (model.feature_net[0].fc1.target.weight.grad is None) or (model.keyboard.base_head.target.weight.grad is None) or (model.mouse.mu_head.target.weight.grad is None):
                print("Training Smoke: Critical Gradient Missing (decision-only)")
                return False
            
            opt.step()
            print("Training smoke (decision-only loss) pass")
            return True
        except Exception as e:
            print("Training smoke error:", type(e).__name__, e)
            return False

    def TestNoNanManySteps(self, steps: int = 40) -> bool:
        try:
            in_dim = 512
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebb=False).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            B = 16
            for t in range(steps):
                x = torch.randn(B, in_dim, device=self.device)
                K = model.num_options
                prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()
                out = model(x, sample=True, deterministic=False, prior=None, prevOptionOnehot=prev, returnKeysVec=False)

                loss = self.DecisionOnlyLoss(out, entCoef=0.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"Non-finite grad at step {t}, {n}")
                        return False
                opt.step()
            print("Multi-step training (decision-only) without NaN/Inf, pass")
            return True
        except Exception as e:
            print("Multi-step training error:", type(e).__name__, e)
            return False

    def TestParamsChange(self, steps: int = 20) -> bool:
        try:
            in_dim = 512
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebb=False).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            with torch.no_grad():
                w_feat0 = model.feature_net[0].fc1.target.weight.clone()
                w_kbd0 = model.keyboard.base_head.target.weight.clone()
                w_mu0 = model.mouse.mu_head.target.weight.clone()

            B = 16
            for _ in range(steps):
                x = torch.randn(B, in_dim, device=self.device)
                K = model.num_options
                prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()
                out = model(x, sample=True, deterministic=False, prior=None, prevOptionOnehot=prev, returnKeysVec=False)
                loss = self.DecisionOnlyLoss(out, entCoef=0.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                d_feat = (w_feat0 - model.feature_net[0].fc1.target.weight).norm().item()
                d_kbd = (w_kbd0 - model.keyboard.base_head.target.weight).norm().item()
                d_mu = (w_mu0 - model.mouse.mu_head.target.weight).norm().item()
            if not any(d > 1e-6 for d in [d_feat, d_kbd, d_mu]):
                print(f"Parameter changes are too small: feat={d_feat:.3e}, kbd={d_kbd:.3e}, mu={d_mu:.3e}")
                return False
            print("Parameters change after decision-only training, pass")
            return True
        except Exception as e:
            print("Parameter change test error:", type(e).__name__, e)
            return False
        
    def TestConvergence(self, steps: int = 400, logEvery: int = 20) -> bool:
        device = self.device

        try:
            in_dim = 512
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebb=False,logstdBounds=(-4.0, 0.5)).to(device)
            model.train()

            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

            B = 32
            xfix = torch.randn(B, in_dim, device=device)
            K = model.num_options
            prevfix = F.one_hot(torch.randint(0, K, (B,), device=device), num_classes=K).float()

            def metrics(out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
                kb, ms, op = out["keyboard"], out["mouse"], out["option"]
                base_p = torch.sigmoid(kb["base_logits"]).clamp(1e-6, 1-1e-6)
                extra_p = torch.sigmoid(kb["extra_logits"]).clamp(1e-6, 1-1e-6)
                base_cnt = (base_p > 0.5).float().sum(dim=1)
                extra_cnt = (extra_p > 0.5).float().sum(dim=1)

                mu, logstd = ms["mu"], ms["logstd"]
                mu_energy = mu.pow(2).sum(dim=1)
                click_p  = torch.sigmoid(ms["click_logits"]).mean(dim=1)

                opt_p0 = F.softmax(op["logits"], dim=-1)[:, 0]
                beta_m = op["beta"].squeeze(-1)

                base_norm = base_cnt / max(1, self.num_base)
                extra_norm = extra_cnt / max(1, self.num_extra)

                proxy = (0.20 * base_norm + 0.35 * extra_norm + 0.20 * mu_energy + 0.12 * click_p + 0.08 * beta_m + 0.05 * (1.0 - opt_p0))

                return {
                    "proxy": proxy,
                    "base_cnt": base_cnt, "extra_cnt": extra_cnt,
                    "mu_energy": mu_energy, "click_p": click_p,
                    "opt0": opt_p0, "beta": beta_m,}

            def shaping_loss(out: Dict[str, Any]) -> torch.Tensor:
                kb, ms, op = out["keyboard"], out["mouse"], out["option"]

                extra_p = torch.sigmoid(kb["extra_logits"]).clamp(1e-6, 1-1e-6)
                click_p = torch.sigmoid(ms["click_logits"]).clamp(1e-6, 1-1e-6)
                mu, logstd = ms["mu"], ms["logstd"]
                beta_m = op["beta"].squeeze(-1)

                L_extra = extra_p.mean() 
                L_click = click_p.mean() 
                L_beta = beta_m.mean() 
                L_mu = mu.pow(2).mean()
                L_sig = torch.exp(logstd).mean()

                target0 = torch.zeros(mu.shape[0], dtype=torch.long, device=mu.device)
                L_opt = F.cross_entropy(op["logits"], target0) 

                base_p = torch.sigmoid(kb["base_logits"]).clamp(1e-6, 1-1e-6)
                L_base = base_p.mean() * 0.2

                w_extra, w_click, w_beta = 1.0, 0.5, 0.5
                w_mu, w_sig, w_opt, w_base = 0.3, 0.2, 0.6, 0.1

                return ( w_extra * L_extra + w_click * L_click + w_beta * L_beta + w_mu * L_mu + w_sig * L_sig + w_opt * L_opt + w_base * L_base)

            def zero_adv(out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
                B = out["mouse"]["mu"].shape[0]
                z = torch.zeros(B, device=device)
                return {"base": z, "extra": z, "skill": z, "mouse": z, "click": z, "option": z, "beta": z}

            with torch.no_grad():
                out0 = model(xfix, sample=True, deterministic=False, prevOptionOnehot=prevfix, returnKeysVec=False)
                m0 = metrics(out0)
                start_proxy = float(m0["proxy"].mean().item())
                start_pack = {k: float(v.mean().item()) for k, v in m0.items() if k != "proxy"}
                print(f"[DecisionOnlyTrain] START proxy={start_proxy:.6f} | "
                      f"base={start_pack['base_cnt']:.3f} extra={start_pack['extra_cnt']:.3f} "
                      f"mu2={start_pack['mu_energy']:.3f} click={start_pack['click_p']:.3f} "
                      f"opt0={start_pack['opt0']:.3f} beta={start_pack['beta']:.3f}")

            ent_coef = 0.0 
            for t in range(1, steps + 1):
                out = model(xfix, sample=True, deterministic=False, prevOptionOnehot=prevfix, returnKeysVec=False)

                base_loss = self.DecisionOnlyLoss(out, adv=zero_adv(out), entCoef=ent_coef)
                shape_loss = shaping_loss(out)

                loss = base_loss + shape_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

                if (t % logEvery) == 0 or t == 1:
                    with torch.no_grad():
                        m = metrics(out)
                        proxy = float(m["proxy"].mean().item())
                        print(
                            f"[DecisionOnlyTrain] step {t}/{steps} | proxy={proxy:.6f} "
                            f"| base={m['base_cnt'].mean():.2f} extra={m['extra_cnt'].mean():.2f} "
                            f"mu2={m['mu_energy'].mean():.3f} click={m['click_p'].mean():.3f} "
                            f"opt0={m['opt0'].mean():.3f} beta={m['beta'].mean():.3f}")

            with torch.no_grad():
                out1 = model(xfix, sample=True, deterministic=False, prevOptionOnehot=prevfix, returnKeysVec=False)
                m1 = metrics(out1)
                end_proxy = float(m1["proxy"].mean().item())
                end_pack = {k: float(v.mean().item()) for k, v in m1.items() if k != "proxy"}

            print(f"[DecisionOnlyTrain] proxy {start_proxy:.6f} -> {end_proxy:.6f}")

            rel_drop = (start_proxy - end_proxy) / max(1e-9, abs(start_proxy))
            ok_checks = 0
            ok_checks += int(end_pack["extra_cnt"] <= start_pack["extra_cnt"] * 0.90)
            ok_checks += int(end_pack["mu_energy"] <= start_pack["mu_energy"] * 0.90)
            ok_checks += int(end_pack["click_p"] <= start_pack["click_p"] * 0.90)
            ok_checks += int(end_pack["beta"] <= start_pack["beta"] * 0.90)
            ok_checks += int(end_pack["opt0"] >= start_pack["opt0"] * 1.05)

            if (rel_drop >= 0.10) and (ok_checks >= 3):
                print("Fixed set convergence via behavior proxy, pass")
                return True
            else:
                print(f"Convergence insufficient: rel_drop={rel_drop:.3f}, checks_passed={ok_checks}/5")
                return False

        except Exception as e:
            print("Convergence test error:", type(e).__name__, e)
            return False

    def PregrowAllLora(self, model: nn.Module, rank: int = 2, scale: float = 1e-3, freezeOld: bool = True) -> bool:
        try:
            cnt = 0
            for m in model.modules():
                if isinstance(m, LoRALinearAdapter) or isinstance(m, MatLoRAAdapter):
                    m.Grow(addRank=rank, freezeOld=freezeOld)
                    cnt += 1
            if cnt == 0:
                print("No LoRA adapter modules found")
                return False
            return True
        except Exception as e:
            print("PregrowAllLora error:", type(e).__name__, e)
            return False

    def ZeroAllGrads(self, model: nn.Module):
        for p in model.parameters():
            p.grad = None

    def SetLoraTrainMode(self, model: nn.Module, mode: str) -> bool:
        assert mode in ("lora_only", "base_only", "hybrid")
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None

        def set_lora_linear(m: LoRALinearAdapter):
            if mode in ("base_only", "hybrid"):
                if m.target.bias is not None: m.target.bias.requires_grad_(True)
                m.target.weight.requires_grad_(True)
            if mode in ("lora_only", "hybrid"):
                for p in list(m.A_list) + list(m.B_list) + list(m.alpha):
                    p.requires_grad_(True)

        def set_mat_lora(m: MatLoRAAdapter, trans_param: torch.nn.Parameter):
            if mode in ("base_only", "hybrid"):
                trans_param.requires_grad_(True)
            if mode in ("lora_only", "hybrid"):
                for p in list(m.A_list) + list(m.B_list) + list(m.alpha):
                    p.requires_grad_(True)

        for mod in model.modules():
            if isinstance(mod, LoRALinearAdapter):
                set_lora_linear(mod)

        if hasattr(model, "option"):
            set_mat_lora(model.option.trans_adapter, model.option.trans)

        if mode in ("base_only", "hybrid"):
            for name, p in model.named_parameters():
                if ("A_list" in name) or ("B_list" in name) or ("alpha" in name):
                    continue
                if name.startswith("option.trans"):
                    continue
                if name.startswith(("feature_net.", "keyboard.backbone.","mouse.backbone.", "psi_to.", "hebb.base")):
                    p.requires_grad_(True)
        return True

    @staticmethod
    def HasGrad(p: torch.Tensor, threshold: float = 1e-12) -> bool:
        return ((p is not None) and getattr(p, "requires_grad", False) and (p.grad is not None) and torch.isfinite(p.grad).all() and (p.grad.abs().max().item() > threshold))

    def TestGradRoutingLora(self, mode: str = "lora_only") -> bool:
        try:
            in_dim, B = 512, 32
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebb=False).to(self.device)

            if not self.PregrowAllLora(model, rank=2, scale=1e-3, freezeOld=True):
                return False
            if not self.SetLoraTrainMode(model, mode):
                return False

            model.train()
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

            x = torch.randn(B, in_dim, device=self.device)
            K = model.num_options
            prev_onehot = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            self.ZeroAllGrads(model)
            out = model(x, sample=True, deterministic=False, prior=None, prevOptionOnehot=prev_onehot, returnKeysVec=False)

            adv = {
                "option": torch.full((B,), 1.5, device=self.device),
                "beta": torch.full((B,), 0.7, device=self.device),
                "skill": torch.ones(B, device=self.device),
                "base": torch.ones(B, device=self.device),
                "extra": torch.ones(B, device=self.device),
                "mouse": torch.ones(B, device=self.device),
                "click": torch.ones(B, device=self.device),}

            loss = self.DecisionOnlyLoss(out, adv=adv, entCoef=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)

            def dep_any(params: List[torch.Tensor]) -> bool:
                cand = [p for p in params if isinstance(p, torch.Tensor) and p.requires_grad]
                if len(cand) == 0:
                    return False
                try:
                    gs = torch.autograd.grad(loss, cand, retain_graph=True, allow_unused=True)
                except RuntimeError:
                    return False
                return any(g is not None for g in gs)

            ok_all = True

            for m in model.modules():
                if isinstance(m, LoRALinearAdapter):
                    base_has = False
                    if m.target.weight is not None:
                        base_has |= self.HasGrad(m.target.weight, threshold=1e-12)
                    if getattr(m.target, "bias", None) is not None:
                        base_has |= self.HasGrad(m.target.bias, threshold=1e-12)

                    lora_params  = list(m.A_list) + list(m.B_list) + list(m.alpha)
                    lora_has_num = any(self.HasGrad(p, threshold=1e-12) for p in lora_params)
                    lora_has_dep = dep_any(lora_params)

                    if mode == "lora_only":
                        cond = (not base_has) and (lora_has_num or lora_has_dep)
                    elif mode == "base_only":
                        cond = base_has and (not lora_has_num) and (not lora_has_dep)
                    else: 
                        cond = base_has and (lora_has_num or lora_has_dep)

                    if not cond:
                        print(f"Grad routing does not comply with policy {mode} @ LoRA({m.target.in_features}->{m.target.out_features})")
                        ok_all = False

            for m in model.option.modules():
                if isinstance(m, MatLoRAAdapter):
                    trans_base_has = self.HasGrad(model.option.trans, threshold=1e-12)

                    lora_params = list(m.A_list) + list(m.B_list) + list(m.alpha)
                    lora_has_num = any(self.HasGrad(p, threshold=1e-12) for p in lora_params)
                    lora_has_dep = dep_any(lora_params)

                    if mode == "lora_only":
                        cond = (not trans_base_has) and (lora_has_num or lora_has_dep)
                    elif mode == "base_only":
                        cond = trans_base_has and (not lora_has_num) and (not lora_has_dep)
                    else: 
                        cond = trans_base_has and (lora_has_num or lora_has_dep)

                    if not cond:
                        print("MatLoRA grad routing does not comply with policy", mode)
                        ok_all = False

            def seq_has_grad(seq, thr=1e-12):
                for _, p in seq.named_parameters(recurse=True):
                    if self.HasGrad(p, threshold=thr):
                        return True
                return False
            ok_psito = all(seq_has_grad(seq) for seq in model.psi_to.values())
            if not ok_psito:
                print("psi_to path has no visible grads (FYI)")

            if not ok_all:
                return False

            opt.step()
            print(f"Grad routing policy verification (decision-only loss), pass  {mode}")
            return True

        except Exception as e:
            print("Grad routing test error:", type(e).__name__, e)
            return False

    def StressTestPlannerAndDecision(self,horizon: int = 6, N: int = 96, elite: int = 12,iters: int = 3, train_steps: int = 150) -> bool:
        try:
            wm, planner = self.BuildPlanner(horizon=horizon, N=N, elite=elite, iters=iters)
            if planner is None:
                return False
            in_dim, B = 512, 32

            planner.wm.ResetHidden(B=B, device=self.device)

            prior = planner.Plan(returnTrajectories=False)
            
            model = DecisionExtractor(stateDim=in_dim, includeNoSkill=True, useHebb=True).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=2e-4)

            x = torch.randn(B, in_dim, device=self.device)
            K = model.num_options
            prev_onehot = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            for t in range(train_steps):
                out = model(x, sample=True, deterministic=False, prior=prior, mixW=0.3, prevOptionOnehot=prev_onehot, returnKeysVec=False)
                loss = self.DecisionOnlyLoss(out, entCoef=0.01)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"Stress: Non-finite gradient {n} @ step {t}")
                        return False
                opt.step()

            print("Stress testing (planner+decision, decision-only loss), pass")
            return True
        except Exception as e:
            print("Stress test error:", type(e).__name__, e)
            return False

    def TestGradRoutingAllModes(self) -> bool:
        ok1 = self.TestGradRoutingLora("lora_only")
        ok2 = self.TestGradRoutingLora("base_only")
        ok3 = self.TestGradRoutingLora("hybrid")
        return ok1 and ok2 and ok3

    def RunAll(self):
        results = {
            "HebbianPlasticityLayer": self.TestHebbLayer(),
            "LoRALinearAdapter": self.TestLoraLinearAdapter(),
            "MatLoRAAdapter": self.TestMatloraAdapter(),
            "DecisionForwardShapes": self.TestDecisionForwardShapes(),
            "OptionPrevAndTrans": self.TestOptionPrevAndTrans(),
            "CEMPlanner": self.TestCemPlanner(),
            "ForwardWithDeltas": self.TestForwardWithDeltasInjection(),
            "CommitOneGrowsLora": self.TestCommitOneGrowsLora(),
            "TrainStepSmoke": self.TestTrainStepSmoke(),
            "NoNanManySteps": self.TestNoNanManySteps(),
            "ParamsChange": self.TestParamsChange(),
            "Convergence": self.TestConvergence(),
            "GradRoutingAllModes": self.TestGradRoutingAllModes(),
            "StressTestPlannerAndDecision": self.StressTestPlannerAndDecision(),
            "ActionsOnly": self.TestActionsOnly(),}
    
        passed = sum(1 for v in results.values() if v)
        print(f"\n[DecisionModule Tests] {passed}/{len(results)} passed.")
        return results