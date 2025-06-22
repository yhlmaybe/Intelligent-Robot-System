from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

KEYBOARD_LAYOUT: Dict[str, Dict[str, int]] = {
    "base_keys": {
        "W": 17, "A": 30, "S": 31, "D": 32,
        "Space": 57, "Shift": 42, "Ctrl": 29, "Esc": 1,
    },
    "skill_keys": {
        "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
        "Q": 16, "E": 18, "R": 19, "F": 33,
        "F1": 59, "F2": 60, "F3": 61, "F4": 62,
        "Mouse4": 110, "Mouse5": 111,
        "G": 34, "T": 20, "V": 47, "B": 48,
    },
    "menu_keys": {
        "Tab": 15, "I": 23, "M": 50, "J": 36, "K": 37,
        "L": 38, "U": 22, "O": 24, "P": 25,
        "F5": 63, "F6": 64,
        "Insert": 110, "Delete": 111, "Home": 102, "End": 107,
    },
    "system_keys": {
        "Enter": 28, "Backspace": 14, "CapsLock": 58, "Win": 125, "Alt": 56,
    },
    "alpha_keys": {chr(i): i - 93 for i in range(97, 123)},  # a‑z
}



class HebbianPlasticityLayer(nn.Module):
    def __init__(self, inDim: int, outDim: int, rate: float = 1e-2, decay: float = 0.995):
        super().__init__()
        self.rate = rate
        self.decay = decay
        self.base = nn.Parameter(torch.randn(outDim, inDim) * 0.02)
        self.hebb = nn.Parameter(torch.zeros(outDim, inDim), requires_grad=False)

    def forward(self, x: torch.Tensor, update: bool = True):
        """x: [B, inDim]"""
        weight = self.base + self.hebb
        out = F.linear(x, weight)  # [B, outDim]

        if self.training and update:
            pre = x  # [B, I]
            post = out  # [B, O]
            # ΔW_b = η (postᵀ · pre − (post ⊙ post) ⊙ W)
            delta = self.rate * (
                torch.einsum("bo,bi->oi", post, pre) / x.size(0)
                - (post.pow(2).mean(0, keepdim=True).t() * self.hebb)
            )
            with torch.no_grad():
                self.hebb.mul_(self.decay).add_(delta)
                self.hebb.data = F.layer_norm(self.hebb.data, (self.hebb.size(1),))
        return F.relu(out)


class MetaLearnerBlock(nn.Module):
    def __init__(self, featDim: int, contextDim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=featDim,
            hidden_size=contextDim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.to_meta = nn.Sequential(
            nn.Linear(2 * contextDim, 256), nn.GELU(), nn.Linear(256, 2 * featDim)
        )

    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor]):
        """x: [B, F] ; mem: [B, T, F] or None"""
        if mem is None:
            ctx = x.unsqueeze(1)
        else:
            ctx = torch.cat([mem, x.unsqueeze(1)], dim=1)
        out, _ = self.lstm(ctx)
        last = out[:, -1]
        meta = self.to_meta(last)  # [B, 2F]
        return meta, ctx  

class TextInputDecoder(nn.Module):
    def __init__(self, inDim: int, hidden: int = 64, maxLen: int = 10):
        super().__init__()
        self.max_len = maxLen
        self.mode_detector = nn.Sequential(nn.Linear(inDim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.char_embed = nn.Embedding(128, hidden)
        self.lstm = nn.LSTM(inDim + hidden, hidden, 1, batch_first=True)
        self.to_char = nn.Linear(hidden, 128)

    def forward(self, feat: torch.Tensor, prev: Optional[torch.Tensor]):
        """feat: [B, F]"""
        batch = feat.size(0)
        mode_prob = torch.sigmoid(self.mode_detector(feat)).squeeze(-1)  # [B]
        char_logits = None
        next_chars = prev

        active = mode_prob > 0.5
        if active.any():
            if prev is None:
                prev = torch.zeros(batch, self.max_len, dtype=torch.long, device=feat.device)
            h = torch.zeros(1, batch, self.lstm.hidden_size, device=feat.device)
            c = torch.zeros_like(h)
            logits = []
            for t in range(self.max_len):
                emb = self.char_embed(prev[:, t])
                inp = torch.cat([feat, emb], dim=-1).unsqueeze(1)
                o, (h, c) = self.lstm(inp, (h, c))
                logit = self.to_char(o.squeeze(1))  # [B, 128]
                logits.append(logit)
                next_char = logit.argmax(-1)
                if t + 1 < self.max_len:
                    prev[:, t + 1] = next_char
            char_logits = torch.stack(logits, dim=1)  # [B, T, 128]
            next_chars = prev
        return mode_prob, char_logits, next_chars

class DynamicKeySelector(nn.Module):
    def __init__(self, inDim: int, numKeys: int = 96, topK: int = 4):
        super().__init__()
        self.top_k = topK
        self.emb = nn.Embedding(numKeys, inDim)
        nn.init.uniform_(self.emb.weight, -0.1, 0.1)
        self.query_proj = nn.Sequential(nn.Linear(inDim, inDim), nn.ReLU())
        self.attn = nn.MultiheadAttention(inDim, num_heads=8, batch_first=True)
        self.register_buffer("key_ids", torch.arange(numKeys))

    def forward(self, feat: torch.Tensor):
        B = feat.size(0)
        q = self.query_proj(feat).unsqueeze(1)  # [B,1,F]
        keys = self.emb(self.key_ids).unsqueeze(0).expand(B, -1, -1)  # [B,K,F]
        attn_out, attn_w = self.attn(q, keys, keys, need_weights=True)
        w = attn_w.squeeze(1)  # [B,K]
        top_p, top_i = torch.topk(w, self.top_k, dim=1)
        return top_i, F.softmax(top_p, dim=-1)

class KeyboardActionModel(nn.Module):
    def __init__(self, inDim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(inDim, 512)
        self.hebb = HebbianPlasticityLayer(512, 512)
        self.fc2 = nn.Linear(512, 256)

        self.base_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(8)
        ])

        self.dynamic = DynamicKeySelector(256, num_keys=96, top_k=4)

        self.text = TextInputDecoder(256)

    def forward(self, feat: torch.Tensor, memory: Optional[torch.Tensor], prevTxt: Optional[torch.Tensor]):
        x = F.relu(self.fc1(feat))
        x = self.hebb(x)
        x = F.relu(self.fc2(x))

        meta, new_mem = None, None  

        base_probs = [torch.sigmoid(h(x)) for h in self.base_heads]  # list[Tensor([B,1])]
        dyn_idx, dyn_p = self.dynamic(x)
        tm_prob, char_logits, next_txt = self.text(x, prevTxt)
        return {
            "base_probs": base_probs,
            "dyn_idx": dyn_idx,
            "dyn_p": dyn_p,
            "text_mode": tm_prob,
            "char_logits": char_logits,
            "memory": new_mem,
            "next_text": next_txt,
        }


    @staticmethod
    def SampleBinary(prob: torch.Tensor, det: bool):
        return (prob > 0.5).float() if det else torch.bernoulli(prob)

    def ToKeyboardVector(self, out: Dict, det: bool = False) -> torch.Tensor:
        vec = torch.zeros(104, device=out["base_probs"][0].device)
        for i, p in enumerate(out["base_probs"]):
            act = self.SampleBinary(p.squeeze(), det)
            key = list(KEYBOARD_LAYOUT["base_keys"].values())[i]
            vec[key] = act
        B_idx = out["dyn_idx"][0]
        B_p = out["dyn_p"][0]
        for j in range(B_idx.size(0)):
            act = self.SampleBinary(B_p[j], det)
            vec[B_idx[j] + 8] = act
        if out["text_mode"][0] > 0.5:
            vec[KEYBOARD_LAYOUT["system_keys"]["Enter"]] = 1.0
        return vec


class MouseActionModel(nn.Module):
    def __init__(self, inDim: int = 256, maxSpeed: float = 5.0):
        super().__init__()
        self.max_speed = maxSpeed
        self.move = nn.Sequential(nn.Linear(inDim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        self.click = nn.Sequential(nn.Linear(inDim, 32), nn.ReLU(), nn.Linear(32, 2))
        self.register_buffer("mov_avg", torch.zeros(2))

    def forward(self, feat: torch.Tensor):
        move_raw = self.move(feat)
        delta = self.max_speed * torch.tanh(move_raw)
        if self.training:
            self.mov_avg.mul_(0.8).add_(0.2 * delta.detach().mean(0))
        else:
            delta = 0.7 * delta + 0.3 * self.mov_avg
        click_p = torch.sigmoid(self.click(feat))
        return delta, click_p

    def ToMouseAction(self, out: Tuple[torch.Tensor, torch.Tensor], det: bool = False):
        delta, p = out
        l = (p[0] > 0.5).float() if det else torch.bernoulli(p[0])
        r = (p[1] > 0.5).float() if det else torch.bernoulli(p[1])
        return {"delta": delta.tolist(), "clicks": [l.item(), r.item()]}


class DecisionModule(nn.Module):
    def __init__(self, stateDim: int = 768):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Linear(stateDim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU()
        )
        self.hebb_meta = nn.Sequential(HebbianPlasticityLayer(512, 512), nn.ReLU(), nn.Linear(512, 256))
        # Actor sub‑heads
        self.kb = KeyboardActionModel(256)
        self.mouse = MouseActionModel(256)
        # Critic
        self.value_head = nn.Linear(256, 1)
        # Memory cache
        self.memory = None
        self.prev_text = None

    def GetValue(self, state: torch.Tensor):
        f = self.feat_net(state)
        f = self.hebb_meta(f)
        return self.value_head(f).squeeze(-1)

    def forward(self, state: torch.Tensor):
        feat = self.feat_net(state)
        feat = self.hebb_meta(feat)
        kb_out = self.kb(feat, self.memory, self.prev_text)
        self.memory = kb_out["memory"]
        self.prev_text = kb_out["next_text"]
        mouse_out = self.mouse(feat)
        return {"keyboard": kb_out, "mouse": mouse_out, "value": self.value_head(feat).squeeze(-1)}

    def Act(self, state: torch.Tensor, det: bool = False):
        with torch.no_grad():
            out = self(state)
        kb_vec = self.kb.ToKeyboardVector(out["keyboard"], det)
        mouse_act = self.mouse.ToMouseAction(out["mouse"], det)
        self.ApplyConstraints(kb_vec)
        return {"keys": kb_vec.cpu().numpy(), "mouse_delta": mouse_act["delta"], "mouse_clicks": mouse_act["clicks"]}

    @staticmethod
    def ApplyConstraints(vec: torch.Tensor):
        W, S = KEYBOARD_LAYOUT["base_keys"]["W"], KEYBOARD_LAYOUT["base_keys"]["S"]
        if vec[W] and vec[S]:
            vec[S] = 0
        A, D = KEYBOARD_LAYOUT["base_keys"]["A"], KEYBOARD_LAYOUT["base_keys"]["D"]
        if vec[A] and vec[D]:
            vec[D] = 0
        if (vec > 0.5).sum() > 6:
            active = torch.where(vec > 0.5)[0]
            vec[active[6:]] = 0


class HybridPolicyTrainer:
    def __init__(self, model: DecisionModule, lr: float = 1e-4, inner_lr: float = 1e-2, gamma: float = 0.99):
        self.model = model
        self.gamma = gamma
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self.inner_lr = inner_lr

    def Value(self, s):
        return self.model.GetValue(s)
    

    def ComputeLoss(self, batch: Dict, weights: Optional[Dict[str, torch.Tensor]] = None):
        states = batch["state"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_s = batch["next_state"]
        dones = batch["done"]

        if weights is not None:
            out = self.FwdWithWeights(states, weights)
            val = self.ValueWithWeights(states, weights)
            next_val = self.ValueWithWeights(next_s, weights)
        else:
            out = self.model(states)
            val = out["value"]
            next_val = self.model.GetValue(next_s)

        logp = self.LogProb(actions, out)
        adv = rewards + self.gamma * next_val * (1 - dones) - val
        policy_loss = -(logp * adv.detach()).mean()
        value_loss = F.mse_loss(val, rewards + self.gamma * next_val * (1 - dones))
        return policy_loss + 0.5 * value_loss

    def LogProb(self, acts: Dict, out: Dict):
        logp = 0.0
        kb_out = out["keyboard"]
        for i, prob in enumerate(kb_out["base_probs"]):
            key_code = list(KEYBOARD_LAYOUT["base_keys"].values())[i]
            a = acts["keys"][:, key_code]
            p = prob.squeeze(1)
            logp = logp + a * torch.log(p + 1e-7) + (1 - a) * torch.log(1 - p + 1e-7)
        return logp.mean()

    def FwdWithWeights(self, s, w):
        orig = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(w, strict=False)
        out = self.model(s)
        self.model.load_state_dict(orig, strict=False)
        return out

    def ValueWithWeights(self, s, w):
        orig = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(w, strict=False)
        v = self.model.GetValue(s)
        self.model.load_state_dict(orig, strict=False)
        return v
