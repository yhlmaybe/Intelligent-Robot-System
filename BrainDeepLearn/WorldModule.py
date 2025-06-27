from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionEncoder(nn.Module):
    def __init__(self, numDiscrete: int = 104, contDim: int = 2, outDim: int = 128):
        super().__init__()
        self.disc_proj = nn.Linear(numDiscrete, outDim, bias=False)
        self.cont_net = nn.Sequential(
            nn.Linear(contDim, 64), 
            nn.ReLU(), 
            nn.Linear(64, outDim))
        
        self.fuse = nn.Sequential(
            nn.Linear(outDim * 2, outDim), 
            nn.Tanh())

        nn.init.zeros_(self.disc_proj.weight)

    def forward(
        self, 
        keysOnehot: torch.Tensor, 
        mouseDelta: Optional[torch.Tensor] = None) -> torch.Tensor:

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

        self.obs_enc = nn.Sequential(
            nn.LayerNorm(visionDim),
            nn.Linear(visionDim, latentDim),
            nn.GELU(),
            nn.LayerNorm(latentDim))
        
        self.act_enc = nn.Sequential(
            nn.Linear(actionDim, latentDim), 
            nn.LayerNorm(latentDim), 
            nn.Tanh())

        self.gru = nn.GRUCell(latentDim * 2, deterDim)

        self.prior_net = nn.Sequential(
            nn.Linear(deterDim, 512), 
            nn.GELU(), 
            nn.Linear(512, latentDim * 2))
        
        self.post_net = nn.Sequential(
            nn.Linear(deterDim + latentDim + deterDim, 512),
            nn.GELU(),
            nn.Linear(512, latentDim * 2))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + latentDim),
            nn.Linear(deterDim + latentDim, stateDim),
            nn.LayerNorm(stateDim))

        self.InitWeights()
        self.ResetHidden()

    def forward(
        self,
        visionIn: torch.Tensor,  # (B, vision_dim)
        actionPrev: torch.Tensor,  # (B, action_dim)
        reset: bool = False) -> torch.Tensor:  # (B, state_dim)

        B, device = visionIn.shape[0], visionIn.device
        
        if reset or self._h is None or self._h.shape[0] != B:
            self.ResetHidden(B, device)

        h_prev = self._h.clone() 
        
        e_t = self.obs_enc(visionIn)  # (B, latent)
        a_t = self.act_enc(actionPrev)  # (B, latent)
        
        gru_input = torch.cat([self._z_prev, a_t], dim=-1)
        self._h = self.gru(gru_input, self._h)
        
        post_inp = torch.cat([self._h, e_t, h_prev], dim=-1)
        mu_q, log_std_q = self.post_net(post_inp).chunk(2, dim=-1)
        std_q = F.softplus(log_std_q) + 1e-4
        
        z_t = mu_q + torch.randn_like(std_q) * std_q
        
        state_raw = torch.cat([self._h, z_t], dim=-1)
        state_feat = self.state_proj(state_raw)
        
        self._z_prev = z_t.detach()
        
        return state_feat

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
                    m.bias_hh.data[m.hidden_size:2*m.hidden_size].fill_(-1.0)

    @staticmethod
    def PackAction(
        actOut: Dict[str, List], 
        device: str | torch.device = "cuda") -> Tuple[torch.Tensor, torch.Tensor]:

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

    def DetachHidden(self):
        self._h = self._h.detach()
        self._z_prev = self._z_prev.detach()

    @torch.no_grad()
    def ImagineStep(
        self, 
        action: torch.Tensor) -> torch.Tensor:  # (B, state_dim)

        a_t = self.act_enc(action)
        self._h = self.gru(torch.cat([self._z_prev, a_t], dim=-1), self._h)
        
        mu_p, log_std_p = self.prior_net(self._h).chunk(2, dim=-1)
        std_p = F.softplus(log_std_p) + 1e-4
        
        z_t = mu_p + torch.randn_like(std_p) * std_p
        
        state = self.state_proj(torch.cat([self._h, z_t], dim=-1))
        
        self._z_prev = z_t.detach()
        
        return state

    @torch.no_grad()
    def ImagineRollout(
        self,
        actions: torch.Tensor,  # (B, T, action_dim)
        initialState: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:  # (B, T, state_dim)

        B, T, _ = actions.shape
        
        orig_h, orig_z = self._h.clone(), self._z_prev.clone()
        
        if initialState is not None:
            self._h, self._z_prev = initialState
        
        states = []
        for t in range(T):
            states.append(self.ImagineStep(actions[:, t]))
        
        self._h, self._z_prev = orig_h, orig_z
        
        return torch.stack(states, dim=1)