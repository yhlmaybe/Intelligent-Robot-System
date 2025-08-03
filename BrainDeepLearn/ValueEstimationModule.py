from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticForward(NamedTuple):
    value: torch.Tensor  # [B]
    tdError: Optional[torch.Tensor] # [B] or None
    tdErrorDe: Optional[torch.Tensor] # [B] or None (detach)
    entropy: Optional[torch.Tensor] # Actor‑side entropy, for logging
    uncertainty: Optional[torch.Tensor] # σ_V (if head enabled)



class ValueEstimationExtractor(nn.Module):
    #Evaluate the value of the previous step based on the data of the previous step, and thereby regulate the current step 
    def __init__(self,
                 memoryDim: int = 768,
                 attnDim: int = 512,
                 stateDim: int = 256,
                 *,
                 hiddenDim: int = 512,
                 gamma: float = 0.99,
                 useLayerNorm: bool = False,
                 valueLossType: str = "mse",
                 useUncertHead: bool = True) -> None:
        super().__init__()
        self.gamma           = gamma
        self.value_loss_type = valueLossType.lower()
        self.use_layer_norm  = useLayerNorm
        self.use_uncert      = useUncertHead

        in_dim = memoryDim + attnDim + stateDim
        self.fc1 = nn.Linear(in_dim, hiddenDim)
        self.fc2 = nn.Linear(hiddenDim, hiddenDim)
        if useLayerNorm:
            self.norm1 = nn.LayerNorm(hiddenDim)
            self.norm2 = nn.LayerNorm(hiddenDim)

        self.value_head  = nn.Linear(hiddenDim, 1)
        self.uncert_head = nn.Linear(hiddenDim, 1) if useUncertHead else None

        self.InitWeights()

    #returns CriticForward
    def forward(self,
                memoryOut: torch.Tensor, # (B,768)  from pre-menory
                attnOut: torch.Tensor, # (B,512)  from pre-attn
                stateFeat: torch.Tensor, # (B,256)  from world state
                *,
                policyEntropy: Optional[torch.Tensor] = None, # from pre-decision
                reward: Optional[torch.Tensor] = None,
                nextValue: Optional[torch.Tensor] = None,
                done: Optional[torch.Tensor] = None) -> CriticForward:

        x = torch.cat([memoryOut, attnOut, stateFeat], dim=-1)
        x = F.relu(self.fc1(x))
        if self.use_layer_norm:
            x = self.norm1(x)
        x = F.relu(self.fc2(x))
        if self.use_layer_norm:
            x = self.norm2(x)

        value = self.value_head(x).squeeze(-1)  # (B)
        uncert = None
        if self.uncert_head is not None:
            uncert = F.softplus(self.uncert_head(x).squeeze(-1))

        td_error, td_error_de = self.TdAdvantage(value, reward, nextValue, done)

        return CriticForward(value=value,tdError=td_error,tdErrorDe=td_error_de,entropy=policyEntropy,uncertainty=uncert)

    def ValueLoss(self, vPred: torch.Tensor, target: torch.Tensor, *, clipDelta: Optional[float] = None) -> torch.Tensor:
        if self.value_loss_type == "huber":
            loss_elem = F.smooth_l1_loss(vPred, target, reduction="none")
        else:
            loss_elem = F.mse_loss(vPred, target, reduction="none")

        if clipDelta is not None:
            v_clip = vPred + (vPred - vPred.detach()).clamp(-clipDelta, clipDelta)
            loss_clip = F.mse_loss(v_clip, target, reduction="none")
            loss_elem = torch.max(loss_elem, loss_clip)
            
        return loss_elem.mean()


    def TdAdvantage(self,
                    value: torch.Tensor,
                    reward: Optional[torch.Tensor],
                    nextValue: Optional[torch.Tensor],
                    done: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        #Computes tdError and detached tdError; returns (None,None) if reward None
        if reward is None:
            return None, None
        device = value.device
        B = value.size(0)

        r = reward.to(device).view(-1) if isinstance(reward, torch.Tensor) else torch.full((B,), reward, device=device)
        d = done.to(device).float().view(-1) if isinstance(done, torch.Tensor) else torch.full((B,), float(done or 0), device=device)
        nv = nextValue.to(device).view(-1) if nextValue is not None else torch.zeros(B, device=device)

        td_target = r + self.gamma * nv * (1 - d)
        td_error  = td_target - value
        return td_error, td_error.detach()

    def InitWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def Save(self, path: str):
        torch.save({"state_dict": self.state_dict(),
                    "gamma": self.gamma,
                    "use_layer_norm": self.use_layer_norm,
                    "value_loss_type": self.value_loss_type}, path)

    @classmethod
    def Load(cls, path: str, mapLocation: Optional[str] = None) -> "ValueEstimationExtractor":
        ckpt = torch.load(path, map_location=mapLocation)
        model = cls(gamma=ckpt["gamma"], useLayerNorm=ckpt["use_layer_norm"], valueLossType=ckpt["value_loss_type"],
                    memoryDim=768, attnDim=512, stateDim=256)  # supply dims as used
        model.load_state_dict(ckpt["state_dict"], strict=False)
        return model