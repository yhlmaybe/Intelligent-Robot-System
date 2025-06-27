from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional

KEYBOARD_LAYOUT: Dict[str, Dict[str, int]] = {
    "base_keys": {
        "W": 17, "A": 30, "S": 31, "D": 32,
        "Space": 57, "Shift": 42, "Ctrl": 29, "Esc": 1},
    "skill_keys": {
        "1": 2, "2": 3, "3": 4, "4": 5, "5": 6,
        "Q": 16, "E": 18, "R": 19, "F": 33,
        "F1": 59, "F2": 60, "F3": 61, "F4": 62,
        "Mouse4": 110, "Mouse5": 111,
        "G": 34, "T": 20, "V": 47, "B": 48},
    "menu_keys": {
        "Tab": 15, "I": 23, "M": 50, "J": 36, "K": 37,
        "L": 38, "U": 22, "O": 24, "P": 25,
        "F5": 63, "F6": 64,
        "Insert": 110, "Delete": 111, "Home": 102, "End": 107},
    "system_keys": {
        "Enter": 28, "Backspace": 14, "CapsLock": 58, "Win": 125, "Alt": 56},
    "alpha_keys": {chr(i): i - 93 for i in range(97, 123)}} # a‑z


class HebbianPlasticityLayer(nn.Module):
    def __init__(self, inDim: int, outDim: int, rate: float = 1e-2, decay: float = 0.995):
        super().__init__()
        self.rate = rate
        self.decay = decay
        self.base = nn.Parameter(torch.randn(outDim, inDim) * 0.02)
        self.hebb = nn.Parameter(torch.zeros(outDim, inDim), requires_grad=False)

    def forward(self, x: torch.Tensor, update: bool = True):
        #x: [B, inDim]
        weight = self.base + self.hebb
        out = F.linear(x, weight)  # [B, outDim]

        if self.training and update:
            pre = x  # [B, I]
            post = out  # [B, O]
            # ΔW_b = η (postᵀ · pre − (post ⊙ post) ⊙ W)
            delta = self.rate * (
                torch.einsum("bo,bi->oi", post, pre) / x.size(0)
                - (post.pow(2).mean(0, keepdim=True).t() * self.hebb))
            with torch.no_grad():
                self.hebb.mul_(self.decay).add_(delta)
                self.hebb.data = F.layer_norm(self.hebb.data, (self.hebb.size(1),))
        return F.relu(out)


class MetaLearnerBlock(nn.Module):
    def __init__(self, featDim: int, contextDim: int = 128, maxLen: int = 16):
        super().__init__()
        self.max_len = maxLen
        self.rnn = nn.LSTM(featDim, contextDim, num_layers=2, batch_first=True, bidirectional=True)
        self.to_meta = nn.Sequential(
            nn.Linear(2*contextDim, 2*featDim),
            nn.GELU())
        
    def forward(self, x: torch.Tensor, mem: Optional[torch.Tensor]):
        if mem is None:
            seq = x.unsqueeze(1)                                 # (B,1,F)
        else:
            seq = torch.cat([mem, x.unsqueeze(1)], dim=1)        # append
            seq = seq[:, -self.max_len:]                         # truncate
        out, _ = self.rnn(seq)
        meta = self.to_meta(out[:, -1])                          # last hidden
        return meta, seq.detach()                                # return new memory (no grad)

class TextInputDecoder(nn.Module):
    def __init__(self, inDim: int, hidden: int = 64, maxLen: int = 10):
        super().__init__()
        self.max_len = maxLen
        self.mode_detector = nn.Sequential(nn.Linear(inDim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.char_embed = nn.Embedding(128, hidden)
        self.lstm = nn.LSTM(inDim + hidden, hidden, 1, batch_first=True)
        self.to_char = nn.Linear(hidden, 128)

    def forward(self, feat: torch.Tensor, prev: Optional[torch.Tensor]):
        #feat: [B, F]
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
            nn.Sequential(nn.Linear(256, 64), 
                          nn.ReLU(), 
                          nn.Linear(64, 1)) for _ in range(8)])
        self.dynamic = DynamicKeySelector(256, numKeys=96, topK=4)
        self.text    = TextInputDecoder(256)

        self.meta_blk = MetaLearnerBlock(256)

    def forward(self,
                feat: torch.Tensor,               # (B,256)
                memory: Optional[torch.Tensor],   # (B,T,256) or None
                prevTxt: Optional[torch.Tensor]):
        x = F.relu(self.fc1(feat))
        x = self.hebb(x)
        x = F.relu(self.fc2(x))

        meta_vec, new_mem = self.meta_blk(x, memory)     # (B,512) split as [Δfast, Δgate]
        delta = torch.tanh(meta_vec[:, :256])            # residual additive term
        gate  = torch.sigmoid(meta_vec[:, 256:])         # gating term
        x = x + gate * delta

        base_probs = [torch.sigmoid(h(x)) for h in self.base_heads]  # 8 Bernoulli
        base_H = torch.stack([
            -(p * (p+1e-8).log() + (1-p) * (1-p+1e-8).log()).squeeze(1)
            for p in base_probs], dim=1).mean(1)                     # (B)

        dyn_idx, dyn_p = self.dynamic(x)
        dyn_H = -(dyn_p * (dyn_p+1e-8).log()).sum(1) / math.log(dyn_p.size(1)+1e-8)

        tm_prob, char_logits, next_txt = self.text(x, prevTxt)
        text_H = -(tm_prob * (tm_prob+1e-8).log() + (1-tm_prob) * (1-tm_prob+1e-8).log())

        entropy = (base_H + dyn_H + text_H) / 3.0

        return {
            "base_probs": base_probs,
            "dyn_idx":    dyn_idx,
            "dyn_p":      dyn_p,
            "text_mode":  tm_prob,
            "char_logits": char_logits,
            "next_text":  next_txt,
            "memory":     new_mem,   # <-- feed back to outer loop
            "entropy":    entropy}
        

    @staticmethod
    def SampleBinary(prob: torch.Tensor, det=False):
        return (prob > 0.5).float() if det else torch.bernoulli(prob)

    def ToKeyboardVector(self, out: Dict, det: bool=False) -> torch.Tensor:
        vec = torch.zeros(104, device=out["base_probs"][0].device)
        for i,p in enumerate(out["base_probs"]):
            act = self.SampleBinary(p.squeeze(), det)
            vec[list(KEYBOARD_LAYOUT["base_keys"].values())[i]] = act
        idx, prob = out["dyn_idx"], out["dyn_p"]
        for j in range(idx.size(1)):
            vec[idx[:,j]+8] = self.SampleBinary(prob[:,j], det)
        enter_code = KEYBOARD_LAYOUT["system_keys"]["Enter"]
        vec[enter_code] = (out["text_mode"] > 0.5).float()
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


class DecisionExtractor(nn.Module):
    def __init__(self, stateDim: int = 768):
        super().__init__()

        self.feature_net = nn.Sequential(
            nn.Linear(stateDim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU())
        
        self.hebb_meta  = nn.Sequential(
            HebbianPlasticityLayer(512,512), 
            nn.ReLU(), 
            nn.Linear(512,256))

        self.kb = KeyboardActionModel(256)
        self.mouse = MouseActionModel(256)

        self.kb_memory: Optional[torch.Tensor] = None
        self.prev_text: Optional[torch.Tensor] = None

    def forward(self, state_feat: torch.Tensor):
        feat = self.feature_net(state_feat)
        feat = self.hebb_meta(feat)

        kb_out = self.kb(feat, self.kb_memory, self.prev_text)
        self.kb_memory = kb_out["memory"]
        self.prev_text = kb_out["next_text"]

        mouse_out = self.mouse(feat)

        mouse_entropy = -((mouse_out[1] * (mouse_out[1]+1e-8).log() + (1-mouse_out[1])*(1-mouse_out[1]+1e-8).log()).sum(1)/2)
        entropy = (kb_out["entropy"] + mouse_entropy) / 2

        return {"keyboard": kb_out, "mouse":     mouse_out, "entropy":   entropy}

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
