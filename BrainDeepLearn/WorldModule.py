from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionEncoder(nn.Module):
    def __init__(self, numDiscrete: int = 128, contDim: int = 2, outDim: int = 128):
        super().__init__()
        
        self.disc_proj = nn.Linear(numDiscrete, outDim, bias=False)
        self.cont_net = nn.Sequential(
            nn.Linear(contDim, 64), nn.ReLU(), nn.Linear(64, outDim))
        
        self.fuse = nn.Sequential(nn.Linear(outDim*2, outDim), nn.Tanh())
        nn.init.zeros_(self.disc_proj.weight)

    def forward(
        self,
        keysOnehot: torch.Tensor, # [B,128]
        mouseDelta: Optional[torch.Tensor] = None # [B,2]
        ) -> torch.Tensor: # [B,128]

        disc_vec = self.disc_proj(keysOnehot.float())

        if mouseDelta is None:
            return disc_vec
        
        cont_vec = self.cont_net(mouseDelta.float())
        return self.fuse(torch.cat([disc_vec, cont_vec], dim=-1))

class WorldModelExtractor(nn.Module):
    def __init__(
        self,
        visionDim: int = 512,
        actionDim: int = 128,
        latentDim: int = 256,
        deterDim: int = 256,
        stateDim: int = 256):

        super().__init__()
        self.latent_dim = latentDim
        self.deter_dim = deterDim
        self.state_dim = stateDim

        self.action_encoder = ActionEncoder(
            numDiscrete=128,
            contDim=2,
            outDim=actionDim)

        self.act_enc = nn.Sequential(
            nn.Linear(actionDim, latentDim),
            nn.LayerNorm(latentDim),
            nn.Tanh())

        self.obs_enc = nn.Sequential(
            nn.LayerNorm(visionDim),
            nn.Linear(visionDim, latentDim),
            nn.GELU(),
            nn.LayerNorm(latentDim))
        
        self.gru = nn.GRUCell(latentDim*2, deterDim)
        self.post_net  = nn.Sequential(nn.Linear(deterDim+latentDim+deterDim, 512), nn.GELU(), nn.Linear(512, latentDim*2))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim+latentDim),
            nn.Linear(deterDim+latentDim, stateDim),
            nn.LayerNorm(stateDim))
        
        self.rew_head  = nn.Linear(stateDim, 1)
        self.done_head = nn.Linear(stateDim, 1)
        nn.init.zeros_(self.rew_head.bias)
        nn.init.zeros_(self.done_head.bias)

        self.InitWeights()
        self.ResetHidden()

    def Step(
        self,
        visionIn: torch.Tensor, # [B,512]
        actionEnc: torch.Tensor, # [B,128]
        isLaten: bool = False) -> torch.Tensor: # [B,256]

        if isLaten:
            e_t = visionIn
        else:
            e_t = self.obs_enc(visionIn) # [B,latent]
        
        a_t = self.act_enc(actionEnc) # [B,latent]

        gru_in = torch.cat([self._z_prev, a_t], dim=-1)

        h_prev = self._h
        self._h = self.gru(gru_in, self._h)
        post_in = torch.cat([self._h, e_t, h_prev], dim=-1)

        mu_q, log_std_q = self.post_net(post_in).chunk(2, dim=-1)

        std_q = F.softplus(log_std_q) + 1e-4
        z_t = mu_q + torch.randn_like(std_q) * std_q

        state_raw = torch.cat([self._h, z_t], dim=-1)
        state_feat = self.state_proj(state_raw)
        self._z_prev = z_t.detach()
        
        return state_feat

    def forward(
        self,
        visionIn: torch.Tensor, # [B,512]
        keysOnehot: torch.Tensor, # [B,128]
        mouseDelta: torch.Tensor, # [B,2]
        reset: bool = False) -> torch.Tensor: # [B,256]
        if reset or self._h is None or self._h.device!=visionIn.device or self._h.size(0)!=visionIn.size(0):
            self.ResetHidden(visionIn.size(0), visionIn.device)

        actionEnc = self.action_encoder(keysOnehot, mouseDelta)

        return self.Step(visionIn, actionEnc)

    def ForwardTrain(
        self,
        visionIn: torch.Tensor,
        actionEnc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        state_feat = self.Step(visionIn, actionEnc)
        pred_r = self.rew_head(state_feat) # [B,1]
        pred_d = self.done_head(state_feat) # [B,1]
        return state_feat, pred_r, pred_d

    @staticmethod
    def PackAction(
        actOut: Dict[str, List],
        device: torch.device | str = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:
        
        key_tensor = torch.tensor(
            actOut["keys"] + actOut["mouse_clicks"],
            dtype=torch.float32,
            device=device).unsqueeze(0)

        mouse_tensor = torch.tensor(
            actOut["mouse_delta"],
            dtype=torch.float32,
            device=device).unsqueeze(0)
        return key_tensor, mouse_tensor

    def ResetHidden(
        self,
        batchSize: int = 1,
        device: torch.device | str = "cpu"):

        device = torch.device(device)
        self._h = torch.zeros(batchSize, self.deter_dim, device=device)
        self._z_prev = torch.zeros(batchSize, self.latent_dim, device=device)

    def InitWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
                if hasattr(m, "bias_hh"):
                    hs = m.hidden_size
                    m.bias_hh.data[hs:2*hs].fill_(-1.0)

class WorldModelSeqRNN(WorldModelExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.seq_encoder = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=self.latent_dim,
            batch_first=True)
        
        for name, param in self.seq_encoder.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        with torch.no_grad():
            if hasattr(self.seq_encoder, 'bias_ih_l0'):
                bias_ih = self.seq_encoder.bias_ih_l0
                update_idx = slice(self.latent_dim, 2*self.latent_dim)
                bias_ih.data[update_idx].fill_(-1.0)
    
    def forward(
        self,
        visionSeq: torch.Tensor,  # [B,S,512]
        keysOnehot: torch.Tensor,  # [B,128]
        mouseDelta: torch.Tensor,  # [B,2]
        reset: bool = False) -> torch.Tensor: # [B,256]

        if reset or self._h is None or self._h.size(0)!=visionSeq.size(0):
            self.ResetHidden(visionSeq.size(0), visionSeq.device)

        e_t = self.EncodeSeq(visionSeq)

        actionEnc = self.action_encoder(keysOnehot, mouseDelta)

        return self.Step(e_t, actionEnc, isLaten=True)  
    
    def EncodeSeq(self, visionSeq: torch.Tensor) -> torch.Tensor:
        B, S, _ = visionSeq.shape
        frames_flat = visionSeq.reshape(B * S, -1)
        latents_flat = self.obs_enc(frames_flat)
        latents = latents_flat.view(B, S, self.latent_dim)
        
        _, hidden = self.seq_encoder(latents)

        return hidden[-1]  