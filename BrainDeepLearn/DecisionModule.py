from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Any
from FunctionTools import GetParametersScale, SiteSpec, BaseOnlineWrapper, AGICoreModule
from collections import OrderedDict


RAW_KEYBOARD_LAYOUT = OrderedDict([
    ("Esc", 0),

    ("F1", 1), ("F2", 2), ("F3", 3), ("F4", 4), ("F5", 5), ("F6", 6),
    ("F7", 7), ("F8", 8), ("F9", 9), ("F10", 10), ("F11", 11), ("F12", 12),

    ("PrintScreen", 13), ("ScrollLock", 14), ("Pause", 15),

    ("Grave", 16),

    ("1", 17), ("2", 18), ("3", 19), ("4", 20), ("5", 21),
    ("6", 22), ("7", 23), ("8", 24), ("9", 25), ("0", 26),

    ("Minus", 27), ("Equal", 28),

    ("Backspace", 29),
    ("Tab", 30),

    ("Q", 31), ("W", 32), ("E", 33), ("R", 34), ("T", 35), ("Y", 36),
    ("U", 37), ("I", 38), ("O", 39), ("P", 40),

    ("LeftBracket", 41), ("RightBracket", 42),
    ("TildeBackslash", 43),

    ("CapsLock", 44),

    ("A", 45), ("S", 46), ("D", 47), ("F", 48), ("G", 49), ("H", 50),
    ("J", 51), ("K", 52), ("L", 53),

    ("Semicolon", 54), ("Apostrophe", 55),

    ("Enter", 56),

    ("Shift", 57),

    ("Z", 58), ("X", 59), ("C", 60), ("V", 61), ("B", 62), ("N", 63), ("M", 64),

    ("Comma", 65), ("Dot", 66), ("Slash", 67),

    ("RShift", 68),

    ("Ctrl", 69),
    ("Win", 70),
    ("Alt", 71),

    ("Space", 72),

    ("RAlt", 73),
    ("RWin", 74),
    ("Menu", 75),

    ("RCtrl", 76),

    ("Insert", 77), ("Home", 78), ("PageUp", 79),
    ("Delete", 80), ("End", 81), ("PageDown", 82),

    ("ArrowUp", 83), ("ArrowLeft", 84), ("ArrowDown", 85), ("ArrowRight", 86),

    ("NumLock", 87),
    ("NumpadDivide", 88), ("NumpadMultiply", 89), ("NumpadMinus", 90),
    ("Numpad7", 91), ("Numpad8", 92), ("Numpad9", 93), ("NumpadPlus", 94),
    ("Numpad4", 95), ("Numpad5", 96), ("Numpad6", 97),
    ("Numpad1", 98), ("Numpad2", 99), ("Numpad3", 100), ("NumpadEnter", 101),
    ("Numpad0", 102), ("NumpadDot", 103),
])


MAX_KEY_CODE = max(RAW_KEYBOARD_LAYOUT.values())
KEY_DIM = MAX_KEY_CODE + 1


def StableLogProbBernoulli(logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return (actions * (-F.softplus(-logits)) + (1.0 - actions) * (-F.softplus(logits))).sum(-1, keepdim=True)

def EntropyBernoulliFromLogits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log()).sum(-1, keepdim=True)

def MixLogits(base: torch.Tensor, prior: Optional[torch.Tensor], w: float) -> torch.Tensor:
    if prior is None:
        return base
    return (1.0 - w) * base + w * prior

def MixGauss(mu: torch.Tensor, logstd: torch.Tensor, priorMu: Optional[torch.Tensor], priorVar: Optional[torch.Tensor], w: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if (priorMu is None) or (priorVar is None):
        return mu, logstd
    var = torch.exp(2.0 * logstd)
    var_mix = (1-w)*var + w*priorVar + w*(1-w)*(mu - priorMu).square()
    mu_mix = (1.0 - w) * mu + w * priorMu
    logstd_mix = 0.5 * torch.log(var_mix.clamp_min(1e-10))
    return mu_mix, logstd_mix


class LoRALinearAdapter(nn.Module):
    def __init__(self, targetLinear: nn.Linear):
        super().__init__()
        assert isinstance(targetLinear, nn.Linear)
        self.target = targetLinear
        self.in_f = targetLinear.in_features
        self.out_f = targetLinear.out_features

        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()  
        self.alpha = nn.ParameterList() 

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if addRank <= 0: return
        if init is None: init = {}
        factory = {"device": self.target.weight.device, "dtype": self.target.weight.dtype}

        A = init.get("A", torch.randn(addRank, self.in_f, **factory) * 1e-4)
        B = init.get("B", torch.randn(self.out_f, addRank, **factory) * 1e-4)
        s = init.get("scale", 1e-2)

        A = nn.Parameter(A.contiguous().to(**factory))
        B = nn.Parameter(B.contiguous().to(**factory))
        s = nn.Parameter(torch.tensor(s, **factory))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A); self.B_list.append(B); self.alpha.append(s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.target.weight
        if len(self.A_list) > 0:
            dW = W.new_zeros(self.out_f, self.in_f)
            for A, B, s in zip(self.A_list, self.B_list, self.alpha):
                s_eff = torch.tanh(s) * GetParametersScale(s) 
                dW = dW + s_eff * (B @ A)
            W = W + dW
        return F.linear(x, W, self.target.bias)


class MatLoRAAdapter(AGICoreModule):
    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.M, self.N = int(rows), int(cols)
        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList() 
        self.alpha = nn.ParameterList()

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if addRank <= 0: return
        if init is None: init = {}

        factory = {"device": self.device, "dtype": self.dtype}

        A = init.get("A", torch.randn(addRank, self.N, **factory) * 1e-4)
        B = init.get("B", torch.randn(self.M, addRank, **factory) * 1e-4)
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(**factory))
        B = nn.Parameter(B.contiguous().to(**factory))
        s = nn.Parameter(torch.as_tensor(s, **factory))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A); self.B_list.append(B); self.alpha.append(s)

    def forward(self, baseMatrix: torch.Tensor) -> torch.Tensor:
        M_eff = baseMatrix
        if len(self.A_list) > 0:
            d = baseMatrix.new_zeros(self.M, self.N)
            for A, B, s in zip(self.A_list, self.B_list, self.alpha):
                s_eff = torch.tanh(s) * GetParametersScale(s)
                d = d + s_eff * (B @ A)
            M_eff = M_eff + d
        return M_eff


class HebbianPlasticityLayer(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        outDim: int,
        rate: float = 1e-3,
        decay: float = 0.995, 
        maxRowNorm: float = 2.0, 
        useHebbian: bool = True,
        applyScale: float = 0.25,):
        super().__init__()
        self.in_dim = int(inDim)
        self.out_dim = int(outDim)

        self.hebb_rate = float(rate)
        self.ema_alpha = float(decay)
        self.mem_norm_cap = float(maxRowNorm)
        self.apply_scale = float(applyScale)
        self.use_hebbian = bool(useHebbian)

        factory = {"device": self.device, "dtype": self.dtype}
        self.base = nn.Parameter(torch.randn(self.out_dim, self.in_dim, **factory) * 0.02)

        self.register_buffer("hebb", torch.empty(0), persistent=True)

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        if int(self.hebb.size(0)) != B:
            self.hebb = torch.zeros(B, self.out_dim, self.in_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):  # x: [B, inDim]

        if not self.use_hebbian:
            return F.linear(x, self.base)

        B = int(x.size(0))
        self.EnsureB(B, self.device, self.dtype)

        w_eff = self.base.unsqueeze(0) + self.apply_scale * self.hebb.detach() # [B, out, in]
        out = torch.einsum("bi,boi->bo", x, w_eff) # [B, out]

        with torch.no_grad():
            pre = x.detach() # [B, in]
            post = out.detach() # [B, out]

            hebb_term = torch.einsum("bo,bi->boi", post, pre) # [B, out, in]

            y2 = post.square().unsqueeze(-1) # [B, out, 1]
            decay_term = y2 * self.hebb # [B, out, in]

            delta = self.hebb_rate * (hebb_term - decay_term) # [B, out, in]

            a = float(self.ema_alpha)
            self.hebb = a * self.hebb + (1-a) * delta

            if self.mem_norm_cap > 0.0:
                flat = self.hebb.reshape(B, -1)
                nrm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8) # [B,1]
                scale = (self.mem_norm_cap / nrm).clamp_max(1.0) # [B,1]
                self.hebb = self.hebb * scale.view(B, 1, 1)

        return out



class MouseActor(AGICoreModule):
    def __init__(self, inDim: int = 512, hidden: int = 512, actDim: int = 2):
        super().__init__()

        self.backbone = nn.Sequential(nn.Linear(inDim, hidden), nn.SiLU(),nn.Linear(hidden, hidden), nn.SiLU())
        
        self.mu_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, actDim))

        self.logstd_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, actDim))

        self.click_head = nn.Sequential(nn.Linear(hidden, hidden // 4), nn.SiLU(), nn.Linear(hidden // 4, 2))


    def Params(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(feat)
        mu = self.mu_head(h)
        logstd = self.logstd_head(h)
        click_logits = self.click_head(h)
        return mu, logstd, click_logits # [B, actDim], [B, actDim], [B, 2]


class KeyboardActor(AGICoreModule):
    def __init__(
        self,
        inDim: int = 512,
        keyDim: Optional[int] = None,
        hidden: int = 512,
        headHidden: Optional[int] = None,
        pDrop: float = 0.0,
        useLayerNorm: bool = True,):
        super().__init__()
        self.key_dim = int(keyDim if keyDim is not None else KEY_DIM)
        hidden = int(hidden)
        headHidden = int(headHidden if headHidden is not None else hidden // 2)

        layers = []
        if useLayerNorm:
            layers.append(nn.LayerNorm(inDim))
        layers += [
            nn.Linear(inDim, hidden),
            nn.SiLU(),
            nn.Dropout(pDrop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(pDrop),]
        
        self.backbone = nn.Sequential(*layers)

        self.keys_head = nn.Sequential(
            nn.Linear(hidden, headHidden),
            nn.SiLU(),
            nn.Dropout(pDrop),
            nn.Linear(headHidden, self.key_dim),)

    def Logits(self, feat: torch.Tensor) -> torch.Tensor: # feat: [B, inDim]
        h = self.backbone(feat) # [B, hidden]
        logits = self.keys_head(h) # [B, key_dim]
        return logits # log-odds


class OptionPolicy(AGICoreModule):
    def __init__(self, zDim=512, numOptions=16, psiDim=128, hidden=256):
        super().__init__()
        self.K = numOptions
        self.enc = nn.Sequential(nn.Linear(zDim, hidden), nn.SiLU())

        self.pi_o = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.K),)

        self.trans = nn.Parameter(torch.zeros(self.K, self.K)) 

        self.trans_adapter = MatLoRAAdapter(self.K, self.K)

        self.psi_head = nn.Linear(hidden, self.K * psiDim)
        self.psiDim = psiDim
        
        self.psi_amp_global = nn.Parameter(torch.tensor(1.0))
        self.psi_amp_per_option = nn.Parameter(torch.ones(self.K))

    def forward(self, z, prevLogitsOpt=None):
        h = self.enc(z) # [B, hidden]
        logits_base = self.pi_o(h) # [B, numOptions]

        if prevLogitsOpt is not None:
            prev = prevLogitsOpt.detach()
            trans_eff = self.trans_adapter(self.trans)
            trans_eff = torch.nan_to_num(trans_eff, nan=0.0).clamp(-10.0, 10.0)
            logits_o = logits_base + prev @ trans_eff
        else:
            logits_o = logits_base

        psi_all = self.psi_head(h).view(-1, self.K, self.psiDim)
        psi_all = psi_all * self.psi_amp_global * self.psi_amp_per_option.view(1, self.K, 1)

        return logits_o, psi_all # [B, K], [B, K, psiDim]


class SwiGLUBlock(AGICoreModule):
    def __init__(self, dim=768, drop=0.1, layerscale=1e-2):
        super().__init__()
        self.hidden=3*dim
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, self.hidden * 2)  
        self.fc2 = nn.Linear(self.hidden, dim)
        self.drop = nn.Dropout(drop)
        self.gamma = nn.Parameter(torch.ones(dim) * layerscale)  

    def forward(self, x): # x: [B, D]
        h = self.ln(x) # [B, D]
        a, b = self.fc1(h).chunk(2, dim=-1) # [B, hidden], [B, hidden]
        h = F.silu(a) * b 
        h = self.fc2(h)
        return x + self.drop(h * self.gamma) # [B, D]


class IntentFusion(AGICoreModule):
    def __init__(
        self,
        stateDim: int,
        intentDim: int,
        hidden: int = 1024,
        numHeads: int = 8,
        numIntentTokens: int = 4,
        bilinearRank: int = 64,
        drop: float = 0.1,
        layerscaleInit: float = 1e-2,):
        super().__init__()
        self.state_dim = int(stateDim)
        self.intent_dim = int(intentDim)
        self.num_intent_tokens = int(numIntentTokens)
        self.rank = int(bilinearRank)

        self.ln_s = nn.LayerNorm(self.state_dim)
        self.ln_i = nn.LayerNorm(self.intent_dim)

        self.i_to_s = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),)

        self.film_gain = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),)
        
        self.film_bias = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),)

        self.bilin_s = nn.Linear(self.state_dim, self.rank, bias=False)
        self.bilin_i = nn.Linear(self.intent_dim, self.rank, bias=False)
        self.bilin_o = nn.Sequential(
            nn.Linear(self.rank, self.state_dim),
            nn.Dropout(drop),)

        self.intent_to_tokens = nn.Sequential(
            nn.Linear(self.intent_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.num_intent_tokens * self.state_dim),)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.state_dim,
            num_heads=numHeads,
            dropout=drop,
            batch_first=True,)

        self.attn_out = nn.Sequential(
            nn.Linear(self.state_dim, self.state_dim),
            nn.Dropout(drop),)

        fuse_in = self.state_dim * 6
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fuse_in, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, self.state_dim),)

        self.gate = nn.Sequential(
            nn.Linear(self.state_dim * 3, self.state_dim),
            nn.Sigmoid(),)

        self.layerscale = nn.Parameter(torch.ones(self.state_dim) * layerscaleInit)

        nn.init.zeros_(self.fuse_mlp[-1].weight)
        nn.init.zeros_(self.fuse_mlp[-1].bias)

    def forward(self, state: torch.Tensor, intent: torch.Tensor) -> torch.Tensor:

        s = self.ln_s(state) # [B, D]
        i = self.ln_i(intent) # [B, I]

        i_proj = self.i_to_s(i) # [B, D]

        gain = torch.tanh(self.film_gain(i)) # [B, D]
        bias = self.film_bias(i) # [B, D]
        film = state * (1.0 + gain) + bias # [B, D]
        film_n = self.ln_s(film) # [B, D]

        inter_mul = s * i_proj # [B, D]

        bilin = self.bilin_o(self.bilin_s(s) * self.bilin_i(i))   # [B, D]

        tokens = self.intent_to_tokens(i).view(-1, self.num_intent_tokens, self.state_dim)  # [B, T, D]
        q = s.unsqueeze(1)  # [B, 1, D]
        ctx, _ = self.cross_attn(q, tokens, tokens, need_weights=False)  # [B, 1, D]
        ctx = self.attn_out(ctx.squeeze(1))  # [B, D]

        fuse_vec = torch.cat([s, i_proj, film_n, inter_mul, bilin, ctx], dim=-1) # [B, 6D]
        delta = self.fuse_mlp(fuse_vec)  # [B, D]

        g = self.gate(torch.cat([s, i_proj, ctx], dim=-1)) # [B, D]

        out = state + (g * self.layerscale) * delta
        return out


class DecisionExtractor(AGICoreModule):
    def __init__(
        self,
        stateDim: int = 1024,
        useHebb: bool = True,
        optionNum: int = 80,
        hiddenDim: int = 1024,
        psiDim: int = 1024,
        intentDim: int = 1024,
        *,
        entropyWeights: Tuple[float, float, float] = (0.6, 0.2, 0.2),): # keys, click, mouse
        super().__init__()

        self.stateDim = int(stateDim)
        self.intentDim = int(intentDim)

        self.fuser = IntentFusion(
            stateDim=self.stateDim,
            intentDim=self.intentDim,
            hidden=1024, 
            numHeads=8,  
            numIntentTokens=4, 
            bilinearRank=64,)

        self.feature_net = nn.Sequential(
            SwiGLUBlock(dim=self.stateDim, drop=0.1, layerscale=1e-2),
            SwiGLUBlock(dim=self.stateDim, drop=0.1, layerscale=1e-2),)

        self.hebb = HebbianPlasticityLayer(self.stateDim, self.stateDim, useHebbian=useHebb)
        self.to_z = nn.Linear(self.stateDim, hiddenDim)
        self.use_hebb_online = bool(useHebb)

        self.max_code = int(MAX_KEY_CODE)
        self.key_dim = int(KEY_DIM)
        self.num_options = int(optionNum)

        self.keyboard = KeyboardActor(hiddenDim, keyDim=self.key_dim)

        self.act_dim = 2
        self.mouse = MouseActor(inDim=hiddenDim, hidden=hiddenDim, actDim=self.act_dim)
        self.option = OptionPolicy(zDim=hiddenDim, numOptions=self.num_options, psiDim=psiDim, hidden=psiDim)

        self.register_buffer("entropy_w", torch.tensor(entropyWeights, dtype=torch.float32))

        self.InstallAdaptersMandatory()

        self.dim_click = 2

        self.psi_trunk = nn.Sequential(
            SwiGLUBlock(dim=psiDim, drop=0.1, layerscale=1e-2),
            SwiGLUBlock(dim=psiDim, drop=0.1, layerscale=1e-2),)

        self.psi_to = nn.ModuleDict({
            "kbd_keys": nn.Sequential(
                nn.LayerNorm(psiDim),
                nn.Linear(psiDim, 1024),
                nn.SiLU(),
                nn.Linear(1024, self.key_dim),),
            "mu": nn.Sequential(
                nn.LayerNorm(psiDim),
                nn.Linear(psiDim, 512),
                nn.SiLU(),
                nn.Linear(512, self.act_dim),),
            "logstd": nn.Sequential(
                nn.LayerNorm(psiDim),
                nn.Linear(psiDim, 512),
                nn.SiLU(),
                nn.Linear(512, self.act_dim),),
            "click": nn.Sequential(
                nn.LayerNorm(psiDim),
                nn.Linear(psiDim, 512),
                nn.SiLU(),
                nn.Linear(512, self.dim_click),),})

    def InstallAdaptersMandatory(self):
        def wrap_linear(parent: nn.Module, name: str):
            lin = getattr(parent, name)
            assert isinstance(lin, nn.Linear), f"{type(parent).__name__}.{name} must be nn.Linear, got {type(lin)}"
            setattr(parent, name, LoRALinearAdapter(lin))

        wrap_linear(self, "to_z")

        wrap_linear(self.keyboard.keys_head, "0")
        wrap_linear(self.keyboard.keys_head, "3")

        wrap_linear(self.mouse.mu_head, "0")
        wrap_linear(self.mouse.mu_head, "2")

        wrap_linear(self.mouse.logstd_head, "0")
        wrap_linear(self.mouse.logstd_head, "2")

        wrap_linear(self.mouse.click_head, "0")
        wrap_linear(self.mouse.click_head, "2")

        wrap_linear(self.option.pi_o, "0")
        wrap_linear(self.option.pi_o, "2")

        wrap_linear(self.option, "psi_head")

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


    def Encode(self, stateFeat: torch.Tensor, intentFeat: torch.Tensor) -> torch.Tensor:
        x = self.fuser(stateFeat, intentFeat)
        x = self.feature_net(x)
        x = self.hebb(x)
        z = F.silu(self.to_z(x))
        return z

    def ToKeysVec(self, keysAct: torch.Tensor, clicks: torch.Tensor) -> torch.Tensor:
        B = keysAct.size(0)
        vec = keysAct.new_zeros(B, self.max_code + 1 + 2)
        vec[:, :self.max_code + 1] = keysAct[:, :self.max_code + 1]
        vec[:, self.max_code + 1:self.max_code + 3] = clicks
        return vec

    def EntropyComponents(
        self, 
        keysLogits, # [B, K]
        clickLogits, # [B, 2]
        logstd): # [B, 2]
        def clean_logits(x):
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
            return x.clamp(-60.0, 60.0)

        keysLogits = clean_logits(keysLogits) # [B, K]
        clickLogits = clean_logits(clickLogits) # [B, 2]

        ent_keys = EntropyBernoulliFromLogits(keysLogits) # [B, 1]
        ent_click = EntropyBernoulliFromLogits(clickLogits) # [B, 1]
        ent_mouse = (0.5 * (1.0 + math.log(2 * math.pi)) + logstd).sum(-1, keepdim=True) # [B, 1]

        n_keys = max(1, keysLogits.size(-1))
        keys_norm = ent_keys / float(n_keys) # [B, 1]
        click_norm = ent_click / 2.0 # [B, 1]

        mouse_norm = logstd.mean(-1, keepdim=True) # [B, 1]

        return {
            "ent_keys": ent_keys, "ent_click": ent_click, "ent_mouse": ent_mouse,
            "keys_norm": keys_norm, "click_norm": click_norm, "mouse_norm": mouse_norm,}  # [B, 1]

    def AggregateEntropy(self, comps: Dict[str, torch.Tensor]) -> torch.Tensor:
        w = self.entropy_w
        return w[0] * comps["keys_norm"] + w[1] * comps["click_norm"] + w[2] * comps["mouse_norm"]

    def forward(
        self,
        stateFeat: torch.Tensor,
        intentFeat: torch.Tensor,
        *,
        sample: bool = True,
        deterministic: bool = False,
        prevOptionLogit: Optional[torch.Tensor] = None,
        prior: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        mixW: float = 0.3,
        returnKeysVec: bool = True,) -> Dict[str, torch.Tensor]:

        B = stateFeat.size(0)
        z = self.Encode(stateFeat, intentFeat) # [B, hiddenDim]

        option_logits, psi_all = self.option(z, prevOptionLogit) # [B, optionNum], [B, optionNum, psiDim], [B, 1]

        w_t = self.SafeSoftmax(option_logits, dim=-1)

        keys_direct_logit = self.keyboard.Logits(z)  # [B, KEY_DIM]
        mu_direct, logstd_direct, click_direct_logit = self.mouse.Params(z) # [B, actDim], [B, actDim], [B, 2]

        psi_mix = (w_t.unsqueeze(-1) * psi_all).sum(dim=1)

        psi_h = self.psi_trunk(psi_mix)  

        keys_psi_logit = self.psi_to["kbd_keys"](psi_h) # [B, KEY_DIM]
        mu_psi = self.psi_to["mu"](psi_h) # [B, act_dim]
        ls_psi = self.psi_to["logstd"](psi_h) # [B, act_dim]
        click_psi_logit = self.psi_to["click"](psi_h) # [B, 2]

        u_keys_dir = EntropyBernoulliFromLogits(keys_direct_logit) / float(self.key_dim) # [B,1]
        u_keys_psi = EntropyBernoulliFromLogits(keys_psi_logit) / float(self.key_dim) # [B,1]

        u_click_dir = EntropyBernoulliFromLogits(click_direct_logit) / 2.0 # [B,1]
        u_click_psi = EntropyBernoulliFromLogits(click_psi_logit) / 2.0 # [B,1]

        alpha_keys = torch.exp(u_keys_dir - u_keys_psi).clamp(0.24, 4) # [B,1]
        alpha_click = torch.exp(u_click_dir - u_click_psi).clamp(0.24, 4) # [B,1]

        keys_logits = keys_direct_logit + alpha_keys * keys_psi_logit # [B, KEY_DIM]
        click_logits = click_direct_logit + alpha_click * click_psi_logit # [B, 2]

        u_mouse_dir = logstd_direct.mean(dim=-1, keepdim=True) # [B,1]
        u_mouse_psi = ls_psi.mean(dim=-1, keepdim=True) # [B,1]

        alpha_mouse = torch.exp(u_mouse_dir - u_mouse_psi).clamp(0.24, 4) # [B,1]

        tau_d = torch.exp(-2.0 * logstd_direct) # [B,2]
        tau_p = torch.exp(-2.0 * ls_psi) # [B,2]

        tau = (tau_d + alpha_mouse * tau_p).clamp_min(1e-12) # [B,2]

        mu = (tau_d * mu_direct + alpha_mouse * tau_p * mu_psi) / tau # [B,2]
        logstd = -0.5 * torch.log(tau) # [B,2]

        if prior is not None:
            prior_keys_logits = prior.get("keys", {}).get("logits", None)
            if prior_keys_logits is not None:
                p0 = torch.sigmoid(keys_logits).clamp(1e-6, 1.0 - 1e-6) # [B, KEY_DIM]
                p1 = torch.sigmoid(prior_keys_logits).clamp(1e-6, 1.0 - 1e-6) # [B, KEY_DIM]
                p = (1.0 - mixW) * p0 + mixW * p1 # [B, KEY_DIM]
                keys_logits = torch.log(p) - torch.log1p(-p) # [B, KEY_DIM]

            prior_click_logits = prior.get("click", {}).get("logits", None)
            if prior_click_logits is not None:
                p0 = torch.sigmoid(click_logits).clamp(1e-6, 1.0 - 1e-6) # [B, 2]
                p1 = torch.sigmoid(prior_click_logits).clamp(1e-6, 1.0 - 1e-6) # [B, 2]
                p  = (1.0 - mixW) * p0 + mixW * p1 # [B, 2]
                click_logits = torch.log(p) - torch.log1p(-p) # [B, 2]

            mu, logstd = MixGauss(
                mu,
                logstd,
                prior.get("mouse", {}).get("mu", None),
                prior.get("mouse", {}).get("var", None),
                mixW,)

        comps = self.EntropyComponents(keys_logits, click_logits, logstd)
        entropy_scalar = self.AggregateEntropy(comps)

        out: Dict[str, Any] = {
            "z": z, # [B, hiddenDim]
            "entropy": entropy_scalar, # [B,1]
            "option": {
                "logits": option_logits, # [B, K]
                "psi_all": psi_all, # [B, K, psiDim]
                "w_t": w_t, # [B, K]
                "psi_mix": psi_mix,}, # [B, psiDim]
            
            "keyboard": {"keys_logits": keys_logits}, # [B, KEY_DIM]
            "mouse": {"mu": mu, "logstd": logstd, "click_logits": click_logits}, # mu/logstd: [B,2], click_logits:[B,2]
            "entropy_components": {
                "keys": comps["ent_keys"], "click": comps["ent_click"], "mouse": comps["ent_mouse"],
                "keys_norm": comps["keys_norm"], "click_norm": comps["click_norm"], "mouse_norm": comps["mouse_norm"],},
            "prevOptionLogit_next": option_logits.detach(),} # [B, K]

        if sample:
            LOG_TWO_PI = math.log(2.0 * math.pi)
            if deterministic:
                keys_act = (torch.sigmoid(keys_logits) > 0.5).float() # [B, KEY_DIM]
                clicks = (torch.sigmoid(click_logits) > 0.5).float() # [B, 2]
                mouse_a = mu # [B, 2]

                logp_keys = StableLogProbBernoulli(keys_logits, keys_act) # [B,1]
                logp_click = StableLogProbBernoulli(click_logits, clicks) # [B,1]
                logp_mouse = (-logstd - 0.5 * LOG_TWO_PI).sum(dim=-1, keepdim=True) # [B,1]
            else:
                keys_prob = torch.sigmoid(keys_logits).clamp(1e-6, 1.0 - 1e-6) # [B, KEY_DIM]
                keys_act = torch.bernoulli(keys_prob) # [B, KEY_DIM]

                click_prob = torch.sigmoid(click_logits).clamp(1e-6, 1.0 - 1e-6) # [B, 2]
                clicks = torch.bernoulli(click_prob) # [B, 2]

                std = torch.exp(logstd).clamp_min(1e-6) # [B, 2]
                eps = torch.randn_like(std) # [B, 2]
                mouse_a = mu + eps * std # [B, 2]

                logp_keys = StableLogProbBernoulli(keys_logits, keys_act) # [B,1]
                logp_click = StableLogProbBernoulli(click_logits, clicks) # [B,1]

                z_norm = (mouse_a.detach() - mu) / std # [B,2]
                logp_mouse = (-0.5 * (z_norm.square() + 2.0 * logstd + LOG_TWO_PI)).sum(dim=-1, keepdim=True)  # [B,1]

            out["keyboard"].update({"keys_act": keys_act, "logp_keys": logp_keys})
            out["mouse"].update({"a": mouse_a, "click_sample": clicks, "logp_mouse": logp_mouse, "logp_click": logp_click})

            if deterministic:
                opt_idx = torch.argmax(option_logits, dim=-1) # [B]
            else:
                opt_idx = torch.distributions.Categorical(probs=w_t).sample() # [B]

            opt_logp_all = F.log_softmax(option_logits, dim=-1) # [B,K]
            logp_option = opt_logp_all.gather(1, opt_idx.view(-1, 1)) # [B,1]

            out["option"].update({
                "opt_idx": opt_idx, # [B]
                "logp_option": logp_option,}) # [B,1]

            if returnKeysVec:
                out["keyvec_raw"] = self.ToKeysVec(keys_act, clicks) # [B, KEY_DIM+2]

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
        maxRankTrans: int = 64,):
        self.maxRankFeat = int(maxRankFeat)
        self.maxRankToZ = int(maxRankToZ)
        self.maxRankKbd = int(maxRankKbd)
        self.maxRankMouse = int(maxRankMouse)
        self.maxRankClick0 = int(maxRankClick0)
        self.maxRankClick2 = int(maxRankClick2)
        self.maxRankPi = int(maxRankPi)
        self.maxRankPsi = int(maxRankPsi)
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
            return torch.tanh(s) * GetParametersScale(s) * (b @ a) # [out, in]

        def alloc_mat(addRank, device, dtype, N, M):
            A = nn.Parameter(torch.randn(addRank, N, device=device, dtype=dtype) * 1e-4)
            B = nn.Parameter(torch.zeros(M, addRank, device=device, dtype=dtype) * 1e-4)
            s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
            return A, B, s

        def compose_mat(a, b, s):
            return torch.tanh(s) * GetParametersScale(s) * (b @ a) # [M, N]

        L = 1 

        toz_in = self.base.to_z.target.in_features
        toz_out = self.base.to_z.target.out_features

        kbd0_in = self.base.keyboard.keys_head[0].target.in_features
        kbd0_out = self.base.keyboard.keys_head[0].target.out_features
        kbd3_in = self.base.keyboard.keys_head[3].target.in_features
        kbd3_out = self.base.keyboard.keys_head[3].target.out_features

        mu0_in = self.base.mouse.mu_head[0].target.in_features
        mu0_out = self.base.mouse.mu_head[0].target.out_features
        mu2_in = self.base.mouse.mu_head[2].target.in_features
        mu2_out = self.base.mouse.mu_head[2].target.out_features

        ls0_in = self.base.mouse.logstd_head[0].target.in_features
        ls0_out = self.base.mouse.logstd_head[0].target.out_features
        ls2_in = self.base.mouse.logstd_head[2].target.in_features
        ls2_out = self.base.mouse.logstd_head[2].target.out_features

        c0_in = self.base.mouse.click_head[0].target.in_features
        c0_out = self.base.mouse.click_head[0].target.out_features
        c2_in = self.base.mouse.click_head[2].target.in_features
        c2_out = self.base.mouse.click_head[2].target.out_features

        K = self.base.option.K
        opt_pi0_in = self.base.option.pi_o[0].target.in_features
        opt_pi0_out = self.base.option.pi_o[0].target.out_features
        opt_pi2_in = self.base.option.pi_o[2].target.in_features
        opt_pi2_out = self.base.option.pi_o[2].target.out_features

        opt_psi_in = self.base.option.psi_head.target.in_features
        opt_psi_out = self.base.option.psi_head.target.out_features

        specs = {
            "toz": SiteSpec("toz", L, toz_in, toz_out, self.maxRankToZ,
                            lambda r, d, dt: alloc_lin(r, d, dt, toz_in, toz_out), compose_lin),

            "kbd0": SiteSpec("kbd0", L, kbd0_in, kbd0_out, self.maxRankKbd,
                             lambda r, d, dt: alloc_lin(r, d, dt, kbd0_in, kbd0_out), compose_lin),
            "kbd3": SiteSpec("kbd3", L, kbd3_in, kbd3_out, self.maxRankKbd,
                             lambda r, d, dt: alloc_lin(r, d, dt, kbd3_in, kbd3_out), compose_lin),

            "mouse_mu0": SiteSpec("mouse_mu0", L, mu0_in, mu0_out, self.maxRankMouse,
                                  lambda r, d, dt: alloc_lin(r, d, dt, mu0_in, mu0_out), compose_lin),
            "mouse_mu2": SiteSpec("mouse_mu2", L, mu2_in, mu2_out, self.maxRankMouse,
                                  lambda r, d, dt: alloc_lin(r, d, dt, mu2_in, mu2_out), compose_lin),

            "mouse_ls0": SiteSpec("mouse_ls0", L, ls0_in, ls0_out, self.maxRankMouse,
                                  lambda r, d, dt: alloc_lin(r, d, dt, ls0_in, ls0_out), compose_lin),
            "mouse_ls2": SiteSpec("mouse_ls2", L, ls2_in, ls2_out, self.maxRankMouse,
                                  lambda r, d, dt: alloc_lin(r, d, dt, ls2_in, ls2_out), compose_lin),

            "click0": SiteSpec("click0", L, c0_in, c0_out, self.maxRankClick0,
                               lambda r, d, dt: alloc_lin(r, d, dt, c0_in, c0_out), compose_lin),
            "click2": SiteSpec("click2", L, c2_in, c2_out, self.maxRankClick2,
                               lambda r, d, dt: alloc_lin(r, d, dt, c2_in, c2_out), compose_lin),

            "opt_pi0": SiteSpec("opt_pi0", L, opt_pi0_in, opt_pi0_out, self.maxRankPi,
                                lambda r, d, dt: alloc_lin(r, d, dt, opt_pi0_in, opt_pi0_out), compose_lin),
            "opt_pi2": SiteSpec("opt_pi2", L, opt_pi2_in, opt_pi2_out, self.maxRankPi,
                                lambda r, d, dt: alloc_lin(r, d, dt, opt_pi2_in, opt_pi2_out), compose_lin),

            "opt_psi": SiteSpec("opt_psi", L, opt_psi_in, opt_psi_out, self.maxRankPsi,
                                lambda r, d, dt: alloc_lin(r, d, dt, opt_psi_in, opt_psi_out), compose_lin),

            "opt_trans": SiteSpec("opt_trans", L, K, K, min(self.maxRankTrans, K),
                                  lambda r, d, dt: alloc_mat(r, d, dt, K, K), compose_mat),}

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
        x,  # stateFeat
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,) -> Dict[str, Any]:
        # kwargs
        sample: bool = bool(kwargs.get("sample", True))
        deterministic: bool = bool(kwargs.get("deterministic", False))
        prevOptionLogit: Optional[torch.Tensor] = kwargs.get("prevOptionLogit", None)
        prior: Optional[Dict[str, Dict[str, torch.Tensor]]] = kwargs.get("prior", None)
        mixW: float = float(kwargs.get("mixW", 0.3))
        returnKeysVec: bool = bool(kwargs.get("returnKeysVec", True))
        intentFeat: Optional[torch.Tensor] = kwargs.get("intentFeat", None)

        if intentFeat is None:
            raise ValueError("DecisionOnlineWrapper.ForwardWithDeltas requires intentFeat in kwargs")

        D = deltasPerLayer[0] if (deltasPerLayer and len(deltasPerLayer) > 0) else {}

        B = x.size(0)
        device = x.device
        K = self.base.option.K

        has_prev = prevOptionLogit is not None

        prev = (
            prevOptionLogit.detach().to(dtype=x.dtype, device=device)
            if has_prev
            else torch.zeros(B, K, dtype=x.dtype, device=device))

        h = self.base.fuser(x, intentFeat)

        for i, blk in enumerate(self.base.feature_net):
            if isinstance(blk, SwiGLUBlock):
                h_norm = blk.ln(h)

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

                h = h + blk.drop(fc2_out * blk.gamma)
            else:
                h = blk(h)

        h = self.base.hebb(h)

        z_lin = self.base.to_z(h)
        if D.get("toz") is not None:
            z_lin = z_lin + F.linear(h, D["toz"], bias=None)
        z = F.silu(z_lin)

        h_o = self.base.option.enc(z)

        t = self.base.option.pi_o[0](h_o)
        if D.get("opt_pi0") is not None:
            t = t + F.linear(h_o, D["opt_pi0"], bias=None)
        t = self.base.option.pi_o[1](t) 
        option_logits = self.base.option.pi_o[2](t)
        if D.get("opt_pi2") is not None:
            option_logits = option_logits + F.linear(t, D["opt_pi2"], bias=None)

        trans_eff = self.base.option.trans_adapter(self.base.option.trans)
        if D.get("opt_trans") is not None:
            trans_eff = trans_eff + D["opt_trans"]
        trans_eff = torch.nan_to_num(trans_eff, nan=0.0).clamp(-10.0, 10.0)

        if has_prev:
            option_logits = option_logits + prev @ trans_eff

        w_t = self.base.SafeSoftmax(option_logits, dim=-1)

        psi_flat = self.base.option.psi_head(h_o)
        if D.get("opt_psi") is not None:
            psi_flat = psi_flat + F.linear(h_o, D["opt_psi"], bias=None)

        psi_all = psi_flat.view(B, K, self.base.option.psiDim)
        psi_all = psi_all * self.base.option.psi_amp_global * self.base.option.psi_amp_per_option.view(1, K, 1)

        psi_mix = (w_t.unsqueeze(-1) * psi_all).sum(dim=1)
        psi_h = self.base.psi_trunk(psi_mix)

        hk = self.base.keyboard.backbone(z)
        kt = self.base.keyboard.keys_head[0](hk)
        if D.get("kbd0") is not None:
            kt = kt + F.linear(hk, D["kbd0"], bias=None)
        kt = self.base.keyboard.keys_head[1](kt) 
        kt = self.base.keyboard.keys_head[2](kt) 
        keys_direct_logit = self.base.keyboard.keys_head[3](kt)
        if D.get("kbd3") is not None:
            keys_direct_logit = keys_direct_logit + F.linear(kt, D["kbd3"], bias=None)

        hm = self.base.mouse.backbone(z)

        mt = self.base.mouse.mu_head[0](hm)
        if D.get("mouse_mu0") is not None:
            mt = mt + F.linear(hm, D["mouse_mu0"], bias=None)
        mt = self.base.mouse.mu_head[1](mt) 
        mu_direct = self.base.mouse.mu_head[2](mt)
        if D.get("mouse_mu2") is not None:
            mu_direct = mu_direct + F.linear(mt, D["mouse_mu2"], bias=None)

        lt = self.base.mouse.logstd_head[0](hm)
        if D.get("mouse_ls0") is not None:
            lt = lt + F.linear(hm, D["mouse_ls0"], bias=None)
        lt = self.base.mouse.logstd_head[1](lt) 
        logstd_direct = self.base.mouse.logstd_head[2](lt)
        if D.get("mouse_ls2") is not None:
            logstd_direct = logstd_direct + F.linear(lt, D["mouse_ls2"], bias=None)

        ct = self.base.mouse.click_head[0](hm)
        if D.get("click0") is not None:
            ct = ct + F.linear(hm, D["click0"], bias=None)
        ct = self.base.mouse.click_head[1](ct) 
        click_direct_logit = self.base.mouse.click_head[2](ct)
        if D.get("click2") is not None:
            click_direct_logit = click_direct_logit + F.linear(ct, D["click2"], bias=None)

        keys_psi_logit = self.base.psi_to["kbd_keys"](psi_h)
        mu_psi = self.base.psi_to["mu"](psi_h)
        ls_psi = self.base.psi_to["logstd"](psi_h)
        click_psi_logit = self.base.psi_to["click"](psi_h)

        u_keys_dir = EntropyBernoulliFromLogits(keys_direct_logit) / float(self.base.key_dim)
        u_keys_psi = EntropyBernoulliFromLogits(keys_psi_logit) / float(self.base.key_dim)

        u_click_dir = EntropyBernoulliFromLogits(click_direct_logit) / 2.0
        u_click_psi = EntropyBernoulliFromLogits(click_psi_logit) / 2.0

        alpha_keys = torch.exp(u_keys_dir - u_keys_psi).clamp(0.24, 4.0)
        alpha_click = torch.exp(u_click_dir - u_click_psi).clamp(0.24, 4.0)

        keys_logits = keys_direct_logit + alpha_keys * keys_psi_logit
        click_logits = click_direct_logit + alpha_click * click_psi_logit

        u_mouse_dir = logstd_direct.mean(dim=-1, keepdim=True)
        u_mouse_psi = ls_psi.mean(dim=-1, keepdim=True)
        alpha_mouse = torch.exp(u_mouse_dir - u_mouse_psi).clamp(0.24, 4.0)

        tau_d = torch.exp(-2.0 * logstd_direct)
        tau_p = torch.exp(-2.0 * ls_psi)
        tau = (tau_d + alpha_mouse * tau_p).clamp_min(1e-12)

        mu = (tau_d * mu_direct + alpha_mouse * tau_p * mu_psi) / tau
        logstd = -0.5 * torch.log(tau)

        if prior is not None:
            prior_keys_logits = prior.get("keys", {}).get("logits", None)
            if prior_keys_logits is not None:
                p0 = torch.sigmoid(keys_logits).clamp(1e-6, 1.0 - 1e-6)
                p1 = torch.sigmoid(prior_keys_logits).clamp(1e-6, 1.0 - 1e-6)
                p = (1.0 - mixW) * p0 + mixW * p1
                keys_logits = torch.log(p) - torch.log1p(-p)

            prior_click_logits = prior.get("click", {}).get("logits", None)
            if prior_click_logits is not None:
                p0 = torch.sigmoid(click_logits).clamp(1e-6, 1.0 - 1e-6)
                p1 = torch.sigmoid(prior_click_logits).clamp(1e-6, 1.0 - 1e-6)
                p = (1.0 - mixW) * p0 + mixW * p1
                click_logits = torch.log(p) - torch.log1p(-p)

            mu, logstd = MixGauss(
                mu,
                logstd,
                prior.get("mouse", {}).get("mu", None),
                prior.get("mouse", {}).get("var", None),
                mixW,)

        comps = self.base.EntropyComponents(keys_logits, click_logits, logstd)
        entropy_scalar = self.base.AggregateEntropy(comps)

        out: Dict[str, Any] = {
            "z": z,
            "entropy": entropy_scalar,
            "option": {
                "logits": option_logits,
                "psi_all": psi_all,
                "w_t": w_t,
                "psi_mix": psi_mix,},

            "keyboard": {"keys_logits": keys_logits},
            "mouse": {"mu": mu, "logstd": logstd, "click_logits": click_logits},
            "entropy_components": {
                "keys": comps["ent_keys"], "click": comps["ent_click"], "mouse": comps["ent_mouse"],
                "keys_norm": comps["keys_norm"], "click_norm": comps["click_norm"], "mouse_norm": comps["mouse_norm"],},

            "prevOptionLogit_next": option_logits.detach(),}

        if sample:
            LOG_TWO_PI = math.log(2.0 * math.pi)

            if deterministic:
                keys_act = (torch.sigmoid(keys_logits) > 0.5).float()
                clicks = (torch.sigmoid(click_logits) > 0.5).float()
                mouse_a = mu

                logp_keys = StableLogProbBernoulli(keys_logits, keys_act)
                logp_click = StableLogProbBernoulli(click_logits, clicks)
                logp_mouse = (-logstd - 0.5 * LOG_TWO_PI).sum(dim=-1, keepdim=True)
            else:
                keys_prob = torch.sigmoid(keys_logits).clamp(1e-6, 1.0 - 1e-6)
                keys_act = torch.bernoulli(keys_prob)

                click_prob = torch.sigmoid(click_logits).clamp(1e-6, 1.0 - 1e-6)
                clicks = torch.bernoulli(click_prob)

                std = torch.exp(logstd).clamp_min(1e-6)
                eps = torch.randn_like(std)
                mouse_a = mu + eps * std

                logp_keys = StableLogProbBernoulli(keys_logits, keys_act)
                logp_click = StableLogProbBernoulli(click_logits, clicks)

                z_norm = (mouse_a.detach() - mu) / std
                logp_mouse = (-0.5 * (z_norm.square() + 2.0 * logstd + LOG_TWO_PI)).sum(dim=-1, keepdim=True)

            out["keyboard"].update({"keys_act": keys_act, "logp_keys": logp_keys})
            out["mouse"].update({"a": mouse_a, "click_sample": clicks, "logp_mouse": logp_mouse, "logp_click": logp_click})

            if deterministic:
                opt_idx = torch.argmax(w_t, dim=-1)
            else:
                opt_idx = torch.distributions.Categorical(probs=w_t).sample()

            logp_option = w_t.clamp_min(1e-12).log().gather(1, opt_idx.view(-1, 1))

            out["option"].update({"opt_idx": opt_idx, "logp_option": logp_option})

            if returnKeysVec:
                out["keyvec_raw"] = self.base.ToKeysVec(keys_act, clicks)

        return out

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        if layerIdx != 0:
            return False

        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "toz":
            self.base.to_z.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site == "kbd0":
            self.base.keyboard.keys_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "kbd3":
            self.base.keyboard.keys_head[3].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site == "mouse_mu0":
            self.base.mouse.mu_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "mouse_mu2":
            self.base.mouse.mu_head[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site == "mouse_ls0":
            self.base.mouse.logstd_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "mouse_ls2":
            self.base.mouse.logstd_head[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site == "click0":
            self.base.mouse.click_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "click2":
            self.base.mouse.click_head[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site == "opt_pi0":
            self.base.option.pi_o[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "opt_pi2":
            self.base.option.pi_o[2].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site == "opt_psi":
            self.base.option.psi_head.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site == "opt_trans":
            self.base.option.trans_adapter.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True

        elif site.startswith("feat") and ("_fc" in site):
            head, tail = site.split("_", 1)
            idx = int(head.replace("feat", ""))
            blk = self.base.feature_net[idx]
            getattr(blk, tail).Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
        else:
            raise ValueError(f"Unknown site: {site}")
        return True


class CEMPlanner(AGICoreModule):
    def __init__(
        self,
        worldModel: nn.Module,
        wmIsOnlineWrapper: bool,
        maxCode: int,
        horizon: int = 5,
        N: int = 64,
        elite: int = 8,
        iters: int = 3,
        gamma: float = 0.99,
        temperature: float = 1.0,
        momentum: float = 0.15,
        minVar: float = 1e-4,
        epsBern: float = 1e-4,):
        super().__init__()
        self.wm = worldModel
        self.wm_is_online_wrapper = bool(wmIsOnlineWrapper)

        self.horizon = int(horizon)
        self.N = int(N)
        self.elite = int(elite)
        self.iters = int(iters)

        self.gamma = float(gamma)
        self.temperature = float(temperature)
        self.momentum = float(momentum)

        self.min_var = float(minVar)
        self.eps_bern = float(epsBern)

        self.max_code = int(maxCode)
        self.key_dim = self.max_code + 1  # keys logits dim

    def LogitsFromProb(self, p: torch.Tensor, eps: float) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return p.log() - (1.0 - p).log()

    def SigmoidProb(self, logits: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.sigmoid(logits).clamp(eps, 1.0 - eps)

    @torch.no_grad()
    def Plan(
        self,
        keysLogits: torch.Tensor, # [B, key_dim]
        mouseMu: torch.Tensor, # [B, 2]
        mouseLogstd: torch.Tensor, # [B, 2]
        clickLogits: torch.Tensor, # [B, 2]
        h0: torch.Tensor,
        z0: torch.Tensor,
        x0: torch.Tensor,
        returnTrajectories: bool = False,) -> Dict[str, Dict[str, torch.Tensor]]:

        B = int(keysLogits.size(0))
        device = self.device
        dtype = self.dtype


        H, N = self.horizon, self.N
        E = min(self.elite, N)

        logits_k = keysLogits.unsqueeze(0).repeat(H, 1, 1).contiguous() # [H,B,K]
        logits_c = clickLogits.unsqueeze(0).repeat(H, 1, 1).contiguous() # [H,B,2]

        mu_t = mouseMu.unsqueeze(0).repeat(H, 1, 1).contiguous() # [H,B,2]
        std_t = torch.exp(mouseLogstd).unsqueeze(0).repeat(H, 1, 1).contiguous() # [H,B,2]
        std_t = std_t.clamp_min(1e-6)

        h_prev, z_prev, x_prev = h0, z0, x0

        for _ in range(self.iters):
            eps_noise = torch.randn(H, B, N, 2, device=device, dtype=dtype)
            mouse_seq = mu_t.unsqueeze(2) + eps_noise * std_t.unsqueeze(2)

            pk = self.SigmoidProb(logits_k, 1e-6) # [H,B,K]
            pc = self.SigmoidProb(logits_c, 1e-6) # [H,B,2]

            keys_seq = (torch.rand(H, B, N, self.key_dim, device=device, dtype=dtype) < pk.unsqueeze(2)).float()# [H,B,N,K]
            click_seq = (torch.rand(H, B, N, 2, device=device, dtype=dtype) < pc.unsqueeze(2)).float() # [H,B,N,2]

            h = h_prev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
            z = z_prev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
            x = x_prev.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()

            score = torch.zeros(B, N, device=device, dtype=dtype)
            cont = torch.ones(B, N, device=device, dtype=dtype)

            for t in range(H):
                a_mouse_t = mouse_seq[t].reshape(B * N, 2)
                a_keys_t = keys_seq[t].reshape(B * N, self.key_dim)
                a_click_t = click_seq[t].reshape(B * N, 2)

                key_vec = torch.cat([a_keys_t, a_click_t], dim=-1)

                if self.wm_is_online_wrapper:
                    a_enc = self.wm.base.action_encoder(key_vec, a_mouse_t)
                    h, z, _s_next, x, r_t, d_t = self.wm.StepPriorWithDeltas(h, z, x, a_enc, sample=False)
                else:
                    a_enc = self.wm.action_encoder(key_vec, a_mouse_t)
                    h, z, _s_next, x, r_t, d_t = self.wm.StepPriorOnly(h, z, x, a_enc, sample=False)

                r_t = r_t.view(B, N)
                d_t = d_t.view(B, N)

                score = score + cont * ((self.gamma ** t) * r_t)
                cont = cont * (1.0 - d_t)

            topk = torch.topk(score, k=E, dim=1).indices # [B,E]
            elite_scores = score.gather(1, topk) # [B,E]

            if self.temperature <= 0.0:
                w = torch.full_like(elite_scores, 1.0 / float(E))
            else:
                w = F.softmax(elite_scores / float(self.temperature), dim=1)
            w_exp = w.unsqueeze(-1) # [B,E,1]

            b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, E) # [B,E]

            for t in range(H):
                elite_mouse_t = mouse_seq[t][b_idx, topk, :] # [B,E,2]
                mu_new = (w_exp * elite_mouse_t).sum(dim=1) # [B,2]
                diff = elite_mouse_t - mu_new.unsqueeze(1)
                var_new = (w_exp * (diff * diff)).sum(dim=1).clamp_min(self.min_var)
                std_new = var_new.sqrt()

                mu_t[t] = self.momentum * mu_t[t] + (1.0 - self.momentum) * mu_new
                std_t[t] = self.momentum * std_t[t] + (1.0 - self.momentum) * std_new
                std_t[t] = std_t[t].clamp_min(1e-6)

                elite_keys_t = keys_seq[t][b_idx, topk, :] # [B,E,K]
                p_hat_k = (w_exp * elite_keys_t).sum(dim=1) # [B,K]
                p_hat_k = p_hat_k.clamp(self.eps_bern, 1.0 - self.eps_bern)

                logits_new_k = self.LogitsFromProb(p_hat_k, self.eps_bern)
                logits_k[t] = self.momentum * logits_k[t] + (1.0 - self.momentum) * logits_new_k

                elite_click_t = click_seq[t][b_idx, topk, :] # [B,E,2]
                p_hat_c = (w_exp * elite_click_t).sum(dim=1) # [B,2]
                p_hat_c = p_hat_c.clamp(self.eps_bern, 1.0 - self.eps_bern)

                logits_new_c = self.LogitsFromProb(p_hat_c, self.eps_bern)
                logits_c[t] = self.momentum * logits_c[t] + (1.0 - self.momentum) * logits_new_c

        out: Dict[str, Dict[str, torch.Tensor]] = {
            "mouse": {"mu": mu_t[0], "var": (std_t[0] * std_t[0])},
            "keys": {"logits": logits_k[0]},
            "click": {"logits": logits_c[0]},}

        if returnTrajectories:
            out["diagnostics"] = {
                "mu_seq": mu_t,
                "std_seq": std_t,
                "keys_logits_seq": logits_k,
                "click_logits_seq": logits_c,}
            
        return out


class DecisionPlannerExtractor:
    def __init__(self):
        pass

    def BuildPlanner(
        self,
        worldModel: nn.Module,
        wmIsOnlineWrapper: bool,
        KEYBOARD_LAYOUT: Dict[str, Dict[str, int]],
        **cemKwargs: Any,) -> CEMPlanner:
        all_codes = []
        for grp in KEYBOARD_LAYOUT.values():
            all_codes += list(grp.values())
        max_code = int(max(all_codes))

        return CEMPlanner(
            worldModel=worldModel,
            wmIsOnlineWrapper=wmIsOnlineWrapper,
            maxCode=max_code,
            **cemKwargs,)


class TestDecisionMTool:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(0)

        self.max_code = int(MAX_KEY_CODE)
        self.key_dim = int(KEY_DIM)  
        self.keyvec_dim = int(KEY_DIM + 2)    


    class MockActionEncoder(nn.Module):
        def __init__(self, keyVecDim: int, outDim: int):
            super().__init__()
            self.key_vec_dim = int(keyVecDim)
            self.out_dim = int(outDim)
            self.net = nn.Sequential(
                nn.Linear(self.key_vec_dim + 2, 128), nn.ReLU(),
                nn.Linear(128, self.out_dim))

        def forward(self, keyVec: torch.Tensor, mouseDelta: torch.Tensor):
            x = torch.cat([keyVec.float(), mouseDelta.float()], dim=-1)
            return self.net(x)

    class MockWorldModel(nn.Module):
        def __init__(self, keyVecDim: int, actionDim: int = 128, deterDim: int = 128, stochDim: int = 16, ssmDim: int = 64, stateDim: int = 128):
            super().__init__()
            self.deter_dim = int(deterDim)
            self.stoch_dim = int(stochDim)
            self.ssm_dim = int(ssmDim)
            self.state_dim = int(stateDim)
            self.action_dim = int(actionDim)

            self.action_encoder = TestDecisionMTool.MockActionEncoder(keyVecDim, actionDim)

            self.act_proj = nn.Sequential(nn.Linear(actionDim, stochDim), nn.Tanh())
            self.gru = nn.GRUCell(input_size=stochDim + stochDim, hidden_size=deterDim)

            self.prior_head = nn.Linear(deterDim, 2 * stochDim)
            self.state_proj = nn.Linear(deterDim + stochDim, stateDim)

            self.rew_head = nn.Linear(stateDim, 1)
            self.done_head = nn.Linear(stateDim, 1)

            self.register_buffer("_h", torch.zeros(1, self.deter_dim))
            self.register_buffer("_z", torch.zeros(1, self.stoch_dim))
            self.register_buffer("_x", torch.zeros(1, self.ssm_dim))

        def ResetHidden(self, B: int = 1, device: torch.device | None = None):
            device = device or self._h.device
            self._h = torch.zeros(B, self.deter_dim, device=device)
            self._z = torch.zeros(B, self.stoch_dim, device=device)
            self._x = torch.zeros(B, self.ssm_dim, device=device)

        def ExportState(self):
            return self._h, self._z, self._x

        @torch.no_grad()
        def StepPriorOnly(self, hPrev, zPrev, xPrev, aEnc, sample: bool = False):
            a = self.act_proj(aEnc) # [B, stochDim]
            h_next = self.gru(torch.cat([zPrev, a], dim=-1), hPrev)

            mu_p, logstd_p = self.prior_head(h_next).chunk(2, dim=-1)
            logstd_p = torch.clamp(logstd_p, -6.0, 2.0)

            z_next = mu_p

            s_next = self.state_proj(torch.cat([h_next, z_next], dim=-1))   # [B, stateDim]
            r_pred = self.rew_head(s_next) # [B,1]
            d_prob = torch.sigmoid(self.done_head(s_next)) 

            x_next = 0.98 * xPrev + 0.02 * torch.tanh(F.pad(z_next, (0, max(0, xPrev.size(-1) - z_next.size(-1))))[:, :xPrev.size(-1)])

            return h_next, z_next, s_next, x_next, r_pred, d_prob


    def BuildPlanner(self, B: int = 2, horizon: int = 3, N: int = 16, elite: int = 4, iters: int = 2):
        wm = self.MockWorldModel(keyVecDim=self.keyvec_dim, actionDim=128, deterDim=128, stochDim=16, ssmDim=64, stateDim=128).to(self.device)
        wm.ResetHidden(B=B, device=self.device)

        planner = CEMPlanner(
            worldModel=wm,
            wmIsOnlineWrapper=False,
            maxCode=self.max_code,
            horizon=horizon, N=N, elite=elite, iters=iters,).to(self.device)
        return wm, planner

    def DecisionOnlyLoss(
        self,
        out: Dict[str, Any],
        adv: Optional[Dict[str, torch.Tensor]] = None,
        entCoef: float = 0.0,
        *,
        returnBreakdown: bool = False,) -> torch.Tensor | Dict[str, Any]:
        device = out["mouse"]["mu"].device
        B = int(out["mouse"]["mu"].size(0))
        adv = adv or {}

        def adv_(name: str, shape) -> torch.Tensor:
            a = adv.get(name, None)
            if a is None:
                a = torch.randn(shape, device=device)
            return a.detach()

        kb, ms, op = out["keyboard"], out["mouse"], out["option"]

        loss_core = torch.tensor(0.0, device=device)
        terms: Dict[str, torch.Tensor] = {}

        if "logp_keys" in kb:
            t = -(adv_("keys", (B, 1)) * kb["logp_keys"]).mean()
            terms["keys"] = t
            loss_core = loss_core + t

        if "logp_mouse" in ms:
            t = -(adv_("mouse", (B, 1)) * ms["logp_mouse"]).mean()
            terms["mouse"] = t
            loss_core = loss_core + t

        if "logp_click" in ms:
            t = -(adv_("click", (B, 1)) * ms["logp_click"]).mean()
            terms["click"] = t
            loss_core = loss_core + t

        if "logp_option" in op:
            t = -(adv_("option", (B, 1)) * op["logp_option"]).mean()
            terms["option"] = t
            loss_core = loss_core + t

        ent_term = torch.tensor(0.0, device=device)
        if entCoef != 0.0 and ("entropy" in out):
            ent_term = -float(entCoef) * out["entropy"].mean()

        loss = loss_core + ent_term

        if not returnBreakdown:
            return loss

        breakdown = {k: float(v.detach()) for k, v in terms.items()}
        return {
            "total": float(loss.detach()),
            "core": float(loss_core.detach()),
            "entropy": float(ent_term.detach()),
            "terms": breakdown,}


    def ZeroAllGrads(self, model: nn.Module):
        for p in model.parameters():
            p.grad = None

    @staticmethod
    def HasGrad(p: torch.Tensor, threshold: float = 1e-12) -> bool:
        return (
            (p is not None)
            and getattr(p, "requires_grad", False)
            and (p.grad is not None)
            and torch.isfinite(p.grad).all()
            and (p.grad.abs().max().item() > threshold))

    def PregrowAllLora(self, model: nn.Module, rank: int = 2, freezeOld: bool = True) -> bool:
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

    def SetLoraTrainMode(self, model: nn.Module, mode: str) -> bool:
        assert mode in ("lora_only", "base_only", "hybrid")

        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None

        def set_lora_linear(m: LoRALinearAdapter):
            if mode in ("base_only", "hybrid"):
                m.target.weight.requires_grad_(True)
                if m.target.bias is not None:
                    m.target.bias.requires_grad_(True)
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
                if name.startswith(("feature_net.", "keyboard.backbone.", "mouse.backbone.", "psi_to.", "hebb.base")):
                    p.requires_grad_(True)

        return True

    def TestHebbLayer(self) -> bool:
        try:
            layer = HebbianPlasticityLayer(128, 64).to(self.device)
            x = torch.randn(8, 128, device=self.device)

            y0 = layer(x)
            y1 = layer(x)

            if y0.shape != (8, 64) or y1.shape != (8, 64):
                print("HebbianPlasticityLayer shape mismatch")
                return False

            with torch.no_grad():
                changed = layer.hebb.abs().sum().item() > 0
            if not changed:
                print("Hebbian buffer not updated online")
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
            s_eff = torch.tanh(s) * GetParametersScale(s)
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
            s0_eff = torch.tanh(s0) * GetParametersScale(s0)
            s1_eff = torch.tanh(s1) * GetParametersScale(s1)
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
            model = DecisionExtractor(
                stateDim=256, intentDim=256,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)
            model.eval()

            B = 3
            x = torch.randn(B, 256, device=self.device)
            intent = torch.randn(B, 256, device=self.device)

            out = model(x, intent, sample=False, prior=None, returnKeysVec=False)
            kb, ms, opt = out["keyboard"], out["mouse"], out["option"]

            checks = [
                kb["keys_logits"].shape == (B, self.key_dim),
                ms["mu"].shape == (B, 2),
                ms["logstd"].shape == (B, 2),
                ms["click_logits"].shape == (B, 2),
                opt["logits"].shape == (B, model.num_options),
                opt["psi_all"].shape == (B, model.num_options, model.option.psiDim),
                opt["w_t"].shape == (B, model.num_options),
                out["entropy"].shape == (B, 1),
                out["prevOptionLogit_next"].shape == (B, model.num_options),]
            
            if not all(checks):
                print("DecisionExtractor forward output dimension mismatch")
                return False

            out2 = model(x, intent, sample=True, deterministic=False, prior=None, returnKeysVec=True)
            kv = out2["keyvec_raw"]
            if kv.shape != (B, self.key_dim + 2):
                print("keyvec_raw shape mismatch")
                return False

            keys_act = out2["keyboard"]["keys_act"]
            clicks = out2["mouse"]["click_sample"]
            if not torch.isfinite(keys_act).all() or not torch.isfinite(clicks).all():
                print("Non-finite sampled actions")
                return False
            if not (((keys_act == 0) | (keys_act == 1)).all() and ((clicks == 0) | (clicks == 1)).all()):
                print("keys_act/click_sample not binary")
                return False

            print("DecisionExtractor Forward/Sampling shapes pass")
            return True
        except Exception as e:
            print("DecisionExtractor forward error:", type(e).__name__, e)
            return False

    def TestOptionPrevAndTrans(self) -> bool:
        try:
            model = DecisionExtractor(
                stateDim=128, intentDim=128,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)
            model.eval()

            K = model.num_options
            model.option.trans_adapter.Grow(addRank=1, freezeOld=False)

            x = torch.randn(2, 128, device=self.device)
            intent = torch.randn(2, 128, device=self.device)
            prev0 = torch.zeros(2, K, device=self.device)
            prev1 = torch.zeros(2, K, device=self.device)
            prev1[:, 0] = 1.0

            out0 = model(x, intent, sample=False, prevOptionLogit=prev0, returnKeysVec=False)
            out1 = model(x, intent, sample=False, prevOptionLogit=prev1, returnKeysVec=False)

            with torch.no_grad():
                trans_eff = model.option.trans_adapter(model.option.trans)
                trans_eff = torch.nan_to_num(trans_eff, nan=0.0).clamp(-10.0, 10.0)
                expect = prev1 @ trans_eff

            diff = (out1["option"]["logits"] - out0["option"]["logits"] - expect).abs().max().item()
            if diff >= 1e-5:
                print(f"prev@trans_eff mismatch: diff={diff:.2e}")
                return False

            print("OptionPolicy prev/transferMatrix pass")
            return True
        except Exception as e:
            print("OptionPolicy error:", type(e).__name__, e)
            return False

    def TestCemPlanner(self) -> bool:
        try:
            B = 2
            wm, planner = self.BuildPlanner(B=B, horizon=3, N=16, elite=4, iters=2)
            h0, z0, x0 = wm.ExportState()

            keysLogits = torch.zeros(B, self.key_dim, device=self.device)
            mouseMu = torch.zeros(B, 2, device=self.device)
            mouseLogstd = torch.zeros(B, 2, device=self.device)
            clickLogits = torch.zeros(B, 2, device=self.device)

            prior = planner.Plan(keysLogits, mouseMu, mouseLogstd, clickLogits, h0, z0, x0, returnTrajectories=True)

            ok = True
            ok &= prior["mouse"]["mu"].shape == (B, 2)
            ok &= prior["mouse"]["var"].shape == (B, 2)
            ok &= prior["keys"]["logits"].shape == (B, self.key_dim)
            ok &= prior["click"]["logits"].shape == (B, 2)

            if not ok:
                print("CEMPlanner output shape mismatch")
                return False

            diag = prior.get("diagnostics", {})
            if diag:
                ok &= diag["mu_seq"].shape[0] == planner.horizon
                ok &= diag["keys_logits_seq"].shape[0] == planner.horizon
                if not ok:
                    print("CEMPlanner diagnostics shape mismatch")
                    return False

            print("CEMPlanner.Plan pass")
            return True
        except Exception as e:
            print("CEMPlanner error:", type(e).__name__, e)
            return False

    def TestForwardWithDeltasInjection(self) -> bool:
        try:
            model = DecisionExtractor(
                stateDim=128, intentDim=128,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)
            wrapper = DecisionOnlineWrapper(model, initRankEach=0, autoRank=False)
            model.eval()

            with torch.no_grad():
                last_lin = model.psi_to["kbd_keys"][-1]
                if isinstance(last_lin, nn.Linear):
                    last_lin.weight.zero_()
                    if last_lin.bias is not None:
                        last_lin.bias.zero_()

            B = 4
            x = torch.randn(B, 128, device=self.device)
            intent = torch.randn(B, 128, device=self.device)

            with torch.no_grad():
                z = model.Encode(x, intent)
                hk = model.keyboard.backbone(z)
                kt = model.keyboard.keys_head[0](hk)
                kt = model.keyboard.keys_head[1](kt)
                kt = model.keyboard.keys_head[2](kt)

            out_dim = model.keyboard.keys_head[3].target.out_features
            in_dim = model.keyboard.keys_head[3].target.in_features
            deltaW = torch.randn(out_dim, in_dim, device=self.device) * 1e-3
            D = {"kbd3": deltaW}

            outD = wrapper.ForwardWithDeltas(x, deltasPerLayer=[D], intentFeat=intent, sample=False, returnKeysVec=False)
            out0 = wrapper.ForwardWithDeltas(x, deltasPerLayer=[{}], intentFeat=intent, sample=False, returnKeysVec=False)

            expect = F.linear(kt, deltaW, bias=None)
            err = (outD["keyboard"]["keys_logits"] - out0["keyboard"]["keys_logits"] - expect).abs().max().item()
            if err >= 1e-5:
                print(f"ForwardWithDeltas kbd3 injection mismatch: err={err:.2e}")
                return False

            print("DecisionOnlineWrapper.ForwardWithDeltas injection pass")
            return True
        except Exception as e:
            print("ForwardWithDeltas error:", type(e).__name__, e)
            return False

    def TestCommitOneGrowsLora(self) -> bool:
        try:
            model = DecisionExtractor(
                stateDim=128, intentDim=128,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)
            wrapper = DecisionOnlineWrapper(model, initRankEach=0, autoRank=False)
            model.eval()

            tgt = model.keyboard.keys_head[3]
            r_before = len(tgt.A_list)

            B = 5
            x = torch.randn(B, 128, device=self.device)
            intent = torch.randn(B, 128, device=self.device)

            with torch.no_grad():
                z = model.Encode(x, intent)
                hk = model.keyboard.backbone(z)
                kt = model.keyboard.keys_head[0](hk)
                kt = model.keyboard.keys_head[1](kt)
                kt = model.keyboard.keys_head[2](kt)
                y_base = tgt(kt).detach()

            addRank = 2
            in_dim = tgt.target.in_features
            out_dim = tgt.target.out_features
            A = torch.randn(addRank, in_dim, device=self.device) * 1e-4
            Bm = torch.zeros(out_dim, addRank, device=self.device) * 1e-4

            ok = wrapper.CommitOne("kbd3", layerIdx=0, a=A, b=Bm, scale=1e-3)
            if (not ok) or (len(tgt.A_list) != r_before + 1):
                print("CommitOne did not increase LoRA rank on kbd3")
                return False

            y_after = tgt(kt)
            A_new, B_new, s_new = tgt.A_list[-1], tgt.B_list[-1], tgt.alpha[-1]
            s_eff = torch.tanh(s_new) * GetParametersScale(s_new)
            expect_delta = F.linear(kt, s_eff * (B_new @ A_new), bias=None)

            err = (y_after - y_base - expect_delta).abs().max().item()
            if err >= 1e-5:
                print(f"CommitOne increment mismatch: err={err:.2e}")
                return False

            print("CommitOne -> LoRA growth and value verification pass")
            return True
        except Exception as e:
            print("CommitOne error:", type(e).__name__, e)
            return False

    def TestTrainStepSmoke(self) -> bool:
        try:
            model = DecisionExtractor(
                stateDim=256, intentDim=256,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            B = 8
            x = torch.randn(B, 256, device=self.device)
            intent = torch.randn(B, 256, device=self.device)
            K = model.num_options
            prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            out = model(x, intent, sample=True, deterministic=False, prior=None, prevOptionLogit=prev, returnKeysVec=False)
            loss = self.DecisionOnlyLoss(out, entCoef=0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if (model.feature_net[0].fc1.target.weight.grad is None) or \
               (model.keyboard.keys_head[3].target.weight.grad is None) or \
               (model.mouse.mu_head[2].target.weight.grad is None) or \
               (model.option.pi_o[2].target.weight.grad is None) or \
               (model.option.psi_head.target.weight.grad is None):
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
            model = DecisionExtractor(
                stateDim=256, intentDim=256,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            B = 16
            K = model.num_options

            for t in range(steps):
                x = torch.randn(B, 256, device=self.device)
                intent = torch.randn(B, 256, device=self.device)
                prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

                out = model(x, intent, sample=True, deterministic=False, prior=None, prevOptionLogit=prev, returnKeysVec=False)
                loss = self.DecisionOnlyLoss(out, entCoef=0.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"Non-finite grad at step {t}, {n}")
                        return False

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            print("Multi-step training (decision-only) without NaN/Inf, pass")
            return True
        except Exception as e:
            print("Multi-step training error:", type(e).__name__, e)
            return False

    def TestParamsChange(self, steps: int = 20) -> bool:
        try:
            model = DecisionExtractor(
                stateDim=256, intentDim=256,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)
            
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            with torch.no_grad():
                w_feat0 = model.feature_net[0].fc1.target.weight.clone()
                w_kbd3 = model.keyboard.keys_head[3].target.weight.clone()
                w_mu2 = model.mouse.mu_head[2].target.weight.clone()

            B = 16
            K = model.num_options
            for _ in range(steps):
                x = torch.randn(B, 256, device=self.device)
                intent = torch.randn(B, 256, device=self.device)
                prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

                out = model(x, intent, sample=True, deterministic=False, prior=None, prevOptionLogit=prev, returnKeysVec=False)
                loss = self.DecisionOnlyLoss(out, entCoef=0.0)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                d_feat = (w_feat0 - model.feature_net[0].fc1.target.weight).norm().item()
                d_kbd = (w_kbd3 - model.keyboard.keys_head[3].target.weight).norm().item()
                d_mu = (w_mu2 - model.mouse.mu_head[2].target.weight).norm().item()

            if not any(d > 1e-6 for d in [d_feat, d_kbd, d_mu]):
                print(f"Parameter changes too small: feat={d_feat:.3e}, kbd3={d_kbd:.3e}, mu2={d_mu:.3e}")
                return False

            print("Parameters change after decision-only training, pass")
            return True
        except Exception as e:
            print("Parameter change test error:", type(e).__name__, e)
            return False

    def TestGradRoutingLora(self, mode: str = "lora_only") -> bool:
        try:
            in_dim, B = 256, 32
            model = DecisionExtractor(
                stateDim=in_dim, intentDim=in_dim,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)

            if not self.PregrowAllLora(model, rank=2, freezeOld=True):
                return False
            if not self.SetLoraTrainMode(model, mode):
                return False

            model.train()
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

            x = torch.randn(B, in_dim, device=self.device)
            intent = torch.randn(B, in_dim, device=self.device)
            K = model.num_options
            prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            self.ZeroAllGrads(model)
            out = model(x, intent, sample=True, deterministic=False, prior=None, prevOptionLogit=prev, returnKeysVec=False)

            adv = {
                "keys": torch.ones(B, 1, device=self.device),
                "mouse": torch.ones(B, 1, device=self.device),
                "click": torch.ones(B, 1, device=self.device),
                "option": torch.ones(B, 1, device=self.device),}
            
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
                        base_has |= self.HasGrad(m.target.weight)
                    if getattr(m.target, "bias", None) is not None:
                        base_has |= self.HasGrad(m.target.bias)

                    lora_params = list(m.A_list) + list(m.B_list) + list(m.alpha)
                    lora_has_num = any(self.HasGrad(p) for p in lora_params)
                    lora_has_dep = dep_any(lora_params)

                    if mode == "lora_only":
                        cond = (not base_has) and (lora_has_num or lora_has_dep)
                    elif mode == "base_only":
                        cond = base_has and (not lora_has_num) and (not lora_has_dep)
                    else:
                        cond = base_has and (lora_has_num or lora_has_dep)

                    if not cond:
                        print(f"Grad routing fails {mode} @ LoRA({m.target.in_features}->{m.target.out_features})")
                        ok_all = False

            trans_base_has = self.HasGrad(model.option.trans)
            ta = model.option.trans_adapter
            lora_params = list(ta.A_list) + list(ta.B_list) + list(ta.alpha)
            lora_has_num = any(self.HasGrad(p) for p in lora_params)
            lora_has_dep = dep_any(lora_params)

            if mode == "lora_only":
                cond = (not trans_base_has) and (lora_has_num or lora_has_dep)
            elif mode == "base_only":
                cond = trans_base_has and (not lora_has_num) and (not lora_has_dep)
            else:
                cond = trans_base_has and (lora_has_num or lora_has_dep)

            if not cond:
                print("MatLoRA grad routing fails", mode)
                ok_all = False

            if not ok_all:
                return False

            opt.step()
            print(f"Grad routing policy verification pass: {mode}")
            return True
        except Exception as e:
            print("Grad routing test error:", type(e).__name__, e)
            return False

    def TestGradRoutingAllModes(self) -> bool:
        ok1 = self.TestGradRoutingLora("lora_only")
        ok2 = self.TestGradRoutingLora("base_only")
        ok3 = self.TestGradRoutingLora("hybrid")
        return ok1 and ok2 and ok3

    def StressTestPlannerAndDecision(self, horizon: int = 6, N: int = 64, elite: int = 10, iters: int = 3, train_steps: int = 120) -> bool:
        try:
            B = 32
            wm, planner = self.BuildPlanner(B=B, horizon=horizon, N=N, elite=elite, iters=iters)

            h0, z0, x0 = wm.ExportState()
            keysLogits0 = torch.zeros(B, self.key_dim, device=self.device)
            mouseMu0 = torch.zeros(B, 2, device=self.device)
            mouseLogstd0 = torch.zeros(B, 2, device=self.device)
            clickLogits0 = torch.zeros(B, 2, device=self.device)

            plan_out = planner.Plan(keysLogits0, mouseMu0, mouseLogstd0, clickLogits0, h0, z0, x0, returnTrajectories=False)

            prior = {
                "keys": {"logits": plan_out["keys"]["logits"]},
                "click": {"logits": plan_out["click"]["logits"]},
                "mouse": {"mu": plan_out["mouse"]["mu"], "var": plan_out["mouse"]["var"]},}

            model = DecisionExtractor(
                stateDim=256, intentDim=256,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=True).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=2e-4)

            x = torch.randn(B, 256, device=self.device)
            intent = torch.randn(B, 256, device=self.device)
            K = model.num_options
            prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            for t in range(train_steps):
                out = model(x, intent, sample=True, deterministic=False, prior=prior, mixW=0.3, prevOptionLogit=prev, returnKeysVec=False)
                loss = self.DecisionOnlyLoss(out, entCoef=0.01)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"Stress: Non-finite gradient {n} @ step {t}")
                        return False

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            print("Stress testing (planner+decision, decision-only loss), pass")
            return True
        except Exception as e:
            print("Stress test error:", type(e).__name__, e)
            return False


    def TestPlannerEffectiveness(self) -> bool:
        try:
            class ToyActionEncoder(nn.Module):
                def __init__(self, keyVecDim: int):
                    super().__init__()
                    self.key_vec_dim = int(keyVecDim)

                def forward(self, keyVec: torch.Tensor, mouseDelta: torch.Tensor) -> torch.Tensor:
                    return torch.cat([keyVec.float(), mouseDelta.float()], dim=-1)

            class ToyRewardWorldModel(nn.Module):
                def __init__(
                    self,
                    keyVecDim: int,
                    deterDim: int = 8,
                    stochDim: int = 4,
                    ssmDim: int = 6,
                    stateDim: int = 8,
                    target_xy: Tuple[float, float] = (0.8, -0.4),
                    space_code: int = 72,):
                    super().__init__()
                    self.key_vec_dim = int(keyVecDim)
                    self.deter_dim = int(deterDim)
                    self.stoch_dim = int(stochDim)
                    self.ssm_dim = int(ssmDim)
                    self.state_dim = int(stateDim)

                    self.action_encoder = ToyActionEncoder(self.key_vec_dim)

                    self.register_buffer("_h", torch.zeros(1, self.deter_dim))
                    self.register_buffer("_z", torch.zeros(1, self.stoch_dim))
                    self.register_buffer("_x", torch.zeros(1, self.ssm_dim))

                    self.register_buffer("_target", torch.tensor(target_xy, dtype=torch.float32).view(1, 2))
                    self.space_code = int(space_code)

                def ResetHidden(self, B: int = 1, device: torch.device | None = None):
                    device = device or self._h.device
                    self._h = torch.zeros(B, self.deter_dim, device=device)
                    self._z = torch.zeros(B, self.stoch_dim, device=device)
                    self._x = torch.zeros(B, self.ssm_dim, device=device)

                def ExportState(self):
                    return self._h, self._z, self._x

                @torch.no_grad()
                def StepPriorOnly(self, hPrev, zPrev, xPrev, aEnc, sample: bool = False):
                    keyvec = aEnc[:, : self.key_vec_dim] 
                    mouse = aEnc[:, self.key_vec_dim : ] 

                    target = self._target.to(device=mouse.device, dtype=mouse.dtype)
                    dist2 = (mouse - target).square().sum(dim=-1, keepdim=True)

                    click0 = keyvec[:, (self.key_vec_dim - 2):(self.key_vec_dim - 1)]
                    space = keyvec[:, self.space_code:self.space_code + 1]

                    r = (-dist2) + 1.2 * space + 0.4 * click0

                    h_next = hPrev
                    z_next = zPrev
                    s_next = torch.zeros(aEnc.size(0), self.state_dim, device=aEnc.device, dtype=aEnc.dtype)
                    x_next = xPrev
                    d_prob = torch.zeros(aEnc.size(0), 1, device=aEnc.device, dtype=aEnc.dtype)

                    return h_next, z_next, s_next, x_next, r, d_prob

            B = 4
            horizon = 4
            N = 128
            elite = 16
            iters = 4

            space_code = int(RAW_KEYBOARD_LAYOUT.get("Space", 72))

            wm = ToyRewardWorldModel(
                keyVecDim=self.keyvec_dim,
                deterDim=8,
                stochDim=4,
                ssmDim=6,
                stateDim=8,
                target_xy=(0.8, -0.4),
                space_code=space_code,).to(self.device)
            wm.ResetHidden(B=B, device=self.device)
            h0, z0, x0 = wm.ExportState()

            planner = CEMPlanner(
                worldModel=wm,
                wmIsOnlineWrapper=False,
                maxCode=self.max_code,
                horizon=horizon,
                N=N,
                elite=elite,
                iters=iters,
                gamma=0.99,
                temperature=1.0,
                momentum=0.15,
                minVar=1e-4,
                epsBern=1e-4,).to(self.device)

            keysLogits0 = torch.zeros(B, self.key_dim, device=self.device)
            mouseMu0 = torch.zeros(B, 2, device=self.device)
            mouseLogstd0 = torch.zeros(B, 2, device=self.device)
            clickLogits0 = torch.zeros(B, 2, device=self.device)

            prior = planner.Plan(keysLogits0, mouseMu0, mouseLogstd0, clickLogits0, h0, z0, x0, returnTrajectories=False)

            target = torch.tensor([0.8, -0.4], device=self.device, dtype=torch.float32).view(1, 2)
            mu_plan = prior["mouse"]["mu"].to(dtype=torch.float32)
            dist2_plan = (mu_plan - target).square().sum(dim=-1).mean().item()
            dist2_base = (mouseMu0.to(dtype=torch.float32) - target).square().sum(dim=-1).mean().item()

            p_space_plan = torch.sigmoid(prior["keys"]["logits"][:, space_code]).mean().item()
            p_click0_plan = torch.sigmoid(prior["click"]["logits"][:, 0]).mean().item()

            p_space_base = 0.5
            p_click0_base = 0.5

            ok = True
            ok &= (dist2_plan < dist2_base * 0.90) 
            ok &= (p_space_plan > p_space_base + 0.15)
            ok &= (p_click0_plan > p_click0_base + 0.10)

            if not ok:
                print(
                    "Planner effectiveness failed:",
                    f"dist2 base={dist2_base:.4f} plan={dist2_plan:.4f}, "
                    f"p(space) base={p_space_base:.3f} plan={p_space_plan:.3f}, "
                    f"p(click0) base={p_click0_base:.3f} plan={p_click0_plan:.3f}")
                return False

            print("CEMPlanner effectiveness (toy reward) pass")
            return True
        except Exception as e:
            print("Planner effectiveness error:", type(e).__name__, e)
            return False


    def TestLossDecreaseSupervised(self, steps: int = 160) -> bool:
        try:
            model = DecisionExtractor(
                stateDim=256, intentDim=256,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=False).to(self.device)

            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=2e-3)

            B = 32
            x = torch.randn(B, 256, device=self.device)
            intent = torch.randn(B, 256, device=self.device)

            K = model.num_options
            prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            tgt_keys = torch.zeros(B, self.key_dim, device=self.device)
            for code in [RAW_KEYBOARD_LAYOUT.get("W", 32), RAW_KEYBOARD_LAYOUT.get("Space", 72), RAW_KEYBOARD_LAYOUT.get("Shift", 57)]:
                if 0 <= int(code) < self.key_dim:
                    tgt_keys[:, int(code)] = 1.0

            tgt_click = torch.tensor([1.0, 0.0], device=self.device).view(1, 2).repeat(B, 1)
            tgt_mouse = torch.tensor([0.35, -0.25], device=self.device).view(1, 2).repeat(B, 1)
            tgt_opt = torch.full((B,), 3, device=self.device, dtype=torch.long)

            bce = nn.BCEWithLogitsLoss(reduction="mean")
            mse = nn.MSELoss(reduction="mean")
            ce = nn.CrossEntropyLoss(reduction="mean")

            losses: List[float] = []

            for t in range(int(steps)):
                out = model(
                    x, intent,
                    sample=False,
                    prevOptionLogit=prev,
                    prior=None,
                    returnKeysVec=False,)

                keys_logits = out["keyboard"]["keys_logits"]
                click_logits = out["mouse"]["click_logits"]
                mu = out["mouse"]["mu"]
                opt_logits = out["option"]["logits"]

                loss_keys = bce(keys_logits, tgt_keys)
                loss_click = bce(click_logits, tgt_click)
                loss_mouse = mse(mu, tgt_mouse)
                loss_opt = ce(opt_logits, tgt_opt)

                loss = loss_keys + 0.5 * loss_click + 2.0 * loss_mouse + 0.5 * loss_opt

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

                losses.append(float(loss.detach().cpu().item()))

            head = sum(losses[:10]) / 10.0
            tail = sum(losses[-10:]) / 10.0

            if not (tail < head * 0.70):
                print(f"Supervised loss did not decrease enough: head={head:.6f} tail={tail:.6f}")
                return False

            print(f"Supervised loss decrease pass. head={head:.6f} -> tail={tail:.6f}")
            return True
        except Exception as e:
            print("Loss decrease test error:", type(e).__name__, e)
            return False


    def TestAllParamsHaveGrad(self) -> bool:
        try:
            model = DecisionExtractor(
                stateDim=256, intentDim=256,
                hiddenDim=128, psiDim=64,
                optionNum=16, useHebb=True).to(self.device)

            if not self.PregrowAllLora(model, rank=2, freezeOld=False):
                return False

            model.train()

            B = 16
            x = torch.randn(B, 256, device=self.device)
            intent = torch.randn(B, 256, device=self.device)

            K = model.num_options
            prev = F.one_hot(torch.randint(0, K, (B,), device=self.device), num_classes=K).float()

            tgt_keys = torch.zeros(B, self.key_dim, device=self.device)
            for code in [RAW_KEYBOARD_LAYOUT.get("W", 32), RAW_KEYBOARD_LAYOUT.get("Space", 72)]:
                if 0 <= int(code) < self.key_dim:
                    tgt_keys[:, int(code)] = 1.0

            tgt_click = torch.tensor([1.0, 1.0], device=self.device).view(1, 2).repeat(B, 1)
            tgt_mouse = torch.tensor([0.2, -0.1], device=self.device).view(1, 2).repeat(B, 1)
            tgt_opt = torch.randint(0, K, (B,), device=self.device, dtype=torch.long)

            bce = nn.BCEWithLogitsLoss(reduction="mean")
            mse = nn.MSELoss(reduction="mean")
            ce = nn.CrossEntropyLoss(reduction="mean")

            out = model(
                x, intent,
                sample=False,
                prevOptionLogit=prev, 
                prior=None,
                returnKeysVec=False,)

            keys_logits = out["keyboard"]["keys_logits"]
            click_logits = out["mouse"]["click_logits"]
            mu = out["mouse"]["mu"]
            logstd = out["mouse"]["logstd"]
            opt_logits = out["option"]["logits"]

            loss = (
                bce(keys_logits, tgt_keys)
                + 0.5 * bce(click_logits, tgt_click)
                + 2.0 * mse(mu, tgt_mouse)
                + 0.1 * mse(logstd, torch.full_like(logstd, -1.0))
                + 0.5 * ce(opt_logits, tgt_opt))

            for p in model.parameters():
                p.grad = None

            loss.backward()

            missing: List[str] = []
            nonfinite: List[str] = []

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    missing.append(name)
                    continue
                if not torch.isfinite(p.grad).all():
                    nonfinite.append(name)

            if missing or nonfinite:
                if missing:
                    print("Missing grad (first 20):", missing[:20])
                if nonfinite:
                    print("Non-finite grad (first 20):", nonfinite[:20])
                return False

            print("All-params gradient coverage pass (base + LoRA)")
            return True
        except Exception as e:
            print("All-params grad test error:", type(e).__name__, e)
            return False


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
            "GradRoutingAllModes": self.TestGradRoutingAllModes(),
            "StressTestPlannerAndDecision": self.StressTestPlannerAndDecision(),
            "PlannerEffectiveness": self.TestPlannerEffectiveness(),
            "LossDecreaseSupervised": self.TestLossDecreaseSupervised(),
            "AllParamsHaveGrad": self.TestAllParamsHaveGrad(),}

        passed = sum(1 for v in results.values() if v)
        print(f"\n[DecisionModule Tests] {passed}/{len(results)} passed.")
        return results
