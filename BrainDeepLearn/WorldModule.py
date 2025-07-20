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

        self.action_encoder = ActionEncoder(
            numDiscrete = 128,
            contDim = 2,
            outDim = actionDim )
        

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

        self.dec_fc1 = nn.Linear(256, 4096)
        self.dec_fc2 = nn.Linear(4096, 12*56*56) 
        
        self.pixel_shuffle = nn.PixelShuffle(2)  
        
        self.refine_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1))
        
        nn.init.kaiming_normal_(self.dec_fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.dec_fc2.weight, nonlinearity='linear')
        nn.init.kaiming_normal_(self.refine_conv[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.refine_conv[2].weight, nonlinearity='linear')

        self.rew_head  = nn.Linear(stateDim, 1)   # Prediction r_t
        self.done_head = nn.Linear(stateDim, 1)   # Prediction d_t (logits)

        nn.init.zeros_(self.rew_head.bias)
        nn.init.zeros_(self.done_head.bias)

        self.InitWeights()
        self.ResetHidden()

    def forward(
        self,
        visionIn: torch.Tensor,  # (B, vision_dim)
        keysOnehot: torch.Tensor,  # (B,128)
        mouseDelta: torch.Tensor,  # (B,2)
        reset: bool = False) -> torch.Tensor:  # (B, state_dim)

        B, device = visionIn.shape[0], visionIn.device
        
        if self._h is None or self._h.device != device or self._h.shape[0] != B or reset:
            self.ResetHidden(B, device)

        h_prev = self._h.clone() 
        
        e_t = self.obs_enc(visionIn)  # (B, latent)

        a128 = self.action_encoder(keysOnehot, mouseDelta) # [B,128]
        a_t = self.act_enc(a128) # [B,256]
        
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

    def ForwardTrain(self,visionIn: torch.Tensor, 
                     actionPrev: torch.Tensor, 
                     return_predictions: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        state_feat = self.forward(visionIn, actionPrev, reset=False)
        pred_r = self.rew_head(state_feat)  # [B, 1]
        pred_d = self.done_head(state_feat)  # [B, 1]  logits

        if return_predictions:
            return state_feat, pred_r, pred_d
        else:
            return state_feat, pred_r, pred_d

    @staticmethod
    def PackAction(actOut: Dict[str, List], device="cuda") -> Tuple[torch.Tensor,torch.Tensor]:
        key_tensor = torch.tensor(
            actOut["keys"] + actOut["mouse_clicks"],
            dtype=torch.float32, device=device).unsqueeze(0)  # [1,128]

        mouse_tensor = torch.tensor(
            actOut["mouse_delta"],
            dtype=torch.float32, device=device).unsqueeze(0)# [1,2]

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

    def DecodeState(self, z: torch.Tensor) -> torch.Tensor:
         B = z.size(0)
        
         x = F.relu(self.dec_fc1(z))
         x = self.dec_fc2(x)
         x = x.view(B, 12, 56, 56)  
         
         x = self.pixel_shuffle(x)   # [B, 3, 112, 112]
         
         low_freq = F.interpolate(z.view(B, 1, 16, 16), size=112, mode="bilinear")
         x = x + low_freq.repeat(1, 3, 1, 1) * 0.2
         
         x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        
         x = self.refine_conv(x)
        
         return torch.sigmoid(x)

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
    

class WorldModelSeqRNN(WorldModelExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_gru: nn.GRU = nn.GRU(input_size=self.latent_dim,hidden_size=self.latent_dim,batch_first=True)

    def EncodeSeq(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape  
        x_flat = x.reshape(B * S, D)
        lat_flat = self.obs_enc(x_flat)
        lat = lat_flat.view(B, S, self.latent_dim)
        _, hT = self.obs_gru(lat)
        return hT.squeeze(0)

    def forward(self,visionSeq: torch.Tensor,actionPrev: torch.Tensor,reset: bool = False) -> torch.Tensor:

        
        B = visionSeq.size(0)  
        dev = visionSeq.device  
        
        if reset or self._h is None or self._h.size(0) != B:
            self.ResetHidden(B, dev)  
        
        h_prev = self._h.clone()
        
        e_t = self.EncodeSeq(visionSeq)
        
        a_t = self.act_enc(actionPrev)
        
        gru_input = torch.cat([self._z_prev, a_t], dim=-1)
        self._h = self.gru(gru_input, self._h)
        
        post_in = torch.cat([self._h, e_t, h_prev], dim=-1)
        
        post_out = self.post_net(post_in)
        mu, loga = post_out.chunk(2, dim=-1)
        
        a = F.softplus(loga) + 1e-4
        
        z = mu + torch.randn_like(a) * a
        
        state_input = torch.cat([self._h, z], dim=-1)
        state = self.state_proj(state_input)
        
        self._z_prev = z.detach()
        
        return state