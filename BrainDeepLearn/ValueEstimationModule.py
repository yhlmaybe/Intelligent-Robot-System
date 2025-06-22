from __future__ import annotations
from typing import Optional, Tuple, Dict, Union, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class ForwardResult(NamedTuple):
    value: torch.Tensor                         # [B]
    td_error: Optional[torch.Tensor]            # [B]
    entropy: torch.Tensor                       # [B]
    advantage: Optional[torch.Tensor]           # [B]
    kl_div: Optional[torch.Tensor]              # [B]
    policy_out: Union[
        torch.Tensor,                           # logits  [B, A] (discrete)
        Tuple[torch.Tensor, torch.Tensor]]      # (μ, log σ) [B, A] (continuous)
    probs: Optional[torch.Tensor]               # [B, A]  (discrete)
    action_sample: Optional[torch.Tensor]       # [B, A]  (continuous sample)
    log_prob: Optional[torch.Tensor]            # [B]


#Versatile RL / BC head supporting discrete & continuous actions.
class ValueEstimationModule(nn.Module):
    """
    """
    def __init__(
        self,
        memoryDim: int,
        attnDim: int,
        stateDim: int,
        actionDim: int,
        *,
        optionalActionDim: int = 0,
        embedAction: bool = False,
        actionEmbedDim: int = 16,
        gamma: float = 0.99,
        isContinuous: bool = False,
        useLayerNorm: bool = False,
        valueLossType: str = "mse",
        hiddenDim: int = 128) -> None:
        super().__init__()

        self.gamma = gamma
        self.is_continuous = isContinuous
        self.value_loss_type = valueLossType
        self.action_dim = actionDim
        self.use_layer_norm = useLayerNorm
        self.embed_action = embedAction

        if embedAction:
            self.action_embed = nn.Embedding(actionDim, actionEmbedDim)
            act_feat_dim = actionEmbedDim
        else:
            act_feat_dim = optionalActionDim

        input_dim = memoryDim + attnDim + stateDim + act_feat_dim
        self.fc1 = nn.Linear(input_dim, hiddenDim)
        self.fc2 = nn.Linear(hiddenDim, hiddenDim)

        if useLayerNorm:
            self.norm1 = nn.LayerNorm(hiddenDim)
            self.norm2 = nn.LayerNorm(hiddenDim)

        self.value_head = nn.Linear(hiddenDim, 1)

        if isContinuous:
            self.policy_head = nn.Linear(hiddenDim, actionDim * 2)
        else:
            self.policy_head = nn.Linear(hiddenDim, actionDim)

        self._config: Dict[str, Union[int, float, bool, str]] = {
            "memory_dim": memoryDim,
            "attn_dim": attnDim,
            "state_dim": stateDim,
            "action_dim": actionDim,
            "optional_action_dim": optionalActionDim,
            "embed_action": embedAction,
            "action_embed_dim": actionEmbedDim,
            "gamma": gamma,
            "is_continuous": isContinuous,
            "use_layer_norm": useLayerNorm,
            "value_loss_type": valueLossType,
            "hidden_dim": hiddenDim}

        self.InitWeights()

    def InitWeights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if m is self.value_head else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        memory_out: torch.Tensor,                 # [B, memory_dim]
        attn_out: torch.Tensor,                   # [B, attn_dim]
        state_feat: torch.Tensor,                 # [B, state_dim]
        *,
        optional_action: Optional[torch.Tensor] = None,
        old_policy_probs: Optional[torch.Tensor] = None,         # discrete
        old_policy_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (μ, logσ) continuous
        reward: Optional[torch.Tensor] = None,
        next_value: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None) -> ForwardResult:

        device = memory_out.device

        if optional_action is not None and self.embed_action:
            act_feat = self.action_embed(optional_action.long())
            x = torch.cat([memory_out, attn_out, state_feat, act_feat], dim=-1)
        elif optional_action is not None:
            x = torch.cat([memory_out, attn_out, state_feat, optional_action], dim=-1)
        else:
            x = torch.cat([memory_out, attn_out, state_feat], dim=-1)

        x = F.relu(self.fc1(x))
        if self.use_layer_norm:
            x = self.norm1(x)

        x = F.relu(self.fc2(x))
        if self.use_layer_norm:
            x = self.norm2(x)

        value: torch.Tensor = self.value_head(x).squeeze(-1)      # [B]

        if self.is_continuous:
            mu, log_std = self.policy_head(x).chunk(2, dim=-1)
            log_std = torch.clamp(log_std, -20.0, 2.0)
            std = log_std.exp()

            policy_dist = dist.Normal(mu, std)

            action_sample = policy_dist.rsample()
            entropy = policy_dist.entropy().sum(-1)         # [B]

            if optional_action is not None:
                log_prob = policy_dist.log_prob(optional_action).sum(-1)  # [B]
            else:
                log_prob = policy_dist.log_prob(action_sample).sum(-1)

            probs = None
            policy_out = (mu, log_std)

            old_dist = None
            if old_policy_params is not None:
                old_mu, old_log_std = old_policy_params
                old_dist = dist.Normal(old_mu.to(device), old_log_std.exp().to(device))

            kl_div = self.ComputeKlDivergence(
                old_dist=old_dist,
                current_dist=policy_dist,
                device=device,
            )

        else:
            logits = self.policy_head(x)                    # [B, A]
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-8)
            entropy = -(probs * log_probs).sum(-1)          # [B]

        if optional_action is not None:
            action_indices = optional_action.long().unsqueeze(-1)
            log_prob = log_probs.gather(1, action_indices).squeeze(-1)
        else:
            s_action = torch.multinomial(probs, 1).squeeze(-1)
            log_prob = log_probs.gather(1, s_action.unsqueeze(-1)).squeeze(-1)

            action_sample = None
            policy_out = logits

            kl_div = self.ComputeKlDivergence(
                old_probs=old_policy_probs,
                current_log_probs=log_probs,
                device=device)

        td_error, advantage = self.ComputeTdAdvantage(
            value, reward, next_value, done, device)

        return ForwardResult(
            value=value,
            td_error=td_error,
            entropy=entropy,
            advantage=advantage,
            kl_div=kl_div,
            policy_out=policy_out,
            probs=probs,
            action_sample=action_sample,
            log_prob=log_prob)


    def ComputeTdAdvantage(
        self,
        value: torch.Tensor,
        reward: Optional[torch.Tensor],
        nextValue: Optional[torch.Tensor],
        done: Optional[torch.Tensor],
        device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if reward is None:
            return None, None

        B = value.shape[0]

        reward_tensor = (
            reward.to(device).view(-1)
            if isinstance(reward, torch.Tensor)
            else torch.full((B,), reward, dtype=torch.float32, device=device))
        done_mask = (
            done.to(device).float().view(-1)
            if isinstance(done, torch.Tensor)
            else torch.full((B,), float(done or 0), dtype=torch.float32, device=device))

        if nextValue is not None:
            next_value_tensor = (
                nextValue.to(device).view(-1)
                if isinstance(nextValue, torch.Tensor)
                else torch.full((B,), nextValue, dtype=torch.float32, device=device))
        else:
            next_value_tensor = torch.zeros(B, dtype=torch.float32, device=device)

        td_target = reward_tensor + self.gamma * next_value_tensor * (1.0 - done_mask)
        td_error = td_target - value
        advantage = td_error.detach()

        return td_error, advantage

    @staticmethod
    def ComputeKlDivergence(
        *,
        oldDist: Optional[dist.Distribution] = None,
        currentDist: Optional[dist.Distribution] = None,
        oldProbs: Optional[torch.Tensor] = None,
        currentLogProbs: Optional[torch.Tensor] = None,
        device: torch.device) -> Optional[torch.Tensor]:

        if oldDist is not None and currentDist is not None:
            # kl_divergence returns [B, A] → sum over A
            return dist.kl_divergence(oldDist, currentDist).sum(-1)

        if oldProbs is None or currentLogProbs is None:
            return None

        old_probs = oldProbs.to(device)
        eps = 1e-8
        log_old = torch.log(old_probs + eps)
        kl = (old_probs * (log_old - currentLogProbs)).sum(-1)  # [B]
        return kl


    def ComputeImitationLoss(
        self,
        logitsOrMu: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        expertAction: torch.Tensor) -> torch.Tensor:
        if self.is_continuous:
            mu, _ = logitsOrMu
            return F.mse_loss(mu, expertAction)
        else:
            logits = logitsOrMu
            return F.cross_entropy(logits, expertAction.long())

    # RL loss (policy + value + entropy + KL)
    def ComputeRlLoss(
        self,
        value: torch.Tensor,
        logProbAction: torch.Tensor,
        *,
        advantage: torch.Tensor,
        targetValue: torch.Tensor,
        entropy: torch.Tensor,
        oldLogProbAction: Optional[torch.Tensor] = None,
        entropyWeight: float = 0.01,
        valueWeight: float = 0.5,
        klWeight: float = 0.0,
        clipValue: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:

        policy_loss = -(advantage * logProbAction).mean()

        if self.value_loss_type == "huber":
            value_loss_elem = F.smooth_l1_loss(value, targetValue, reduction="none")
        else:
            value_loss_elem = F.mse_loss(value, targetValue, reduction="none")

        if clipValue is not None:
            v_clipped = value + (value - value.detach()).clamp(-clipValue, clipValue)
            v_loss_clip = F.mse_loss(v_clipped, targetValue, reduction="none")
            value_loss_elem = torch.max(value_loss_elem, v_loss_clip)

        value_loss = value_loss_elem.mean()

        entropy_loss = -entropyWeight * entropy.mean()

        kl_loss = torch.torch.Tensor(0.0, device=value.device)
        if oldLogProbAction is not None and klWeight > 0.0:
            kl_div = (oldLogProbAction - logProbAction).mean()
            kl_loss = klWeight * kl_div

        total_loss = policy_loss + valueWeight * value_loss + entropy_loss + kl_loss

        info = {
            "total": float(total_loss.item()),
            "policy": float(policy_loss.item()),
            "value": float(value_loss.item()),
            "entropy": float(entropy.mean().item()),
            "kl": float(kl_loss.item()) if klWeight > 0 else 0.0}
        return total_loss, info

    # GAE (vectorised for efficiency)
    @staticmethod
    def ComputeGae(
        rewards: torch.Tensor,      # [T, B]
        values: torch.Tensor,       # [T, B]
        dones: torch.Tensor,        # [T, B]  (1 = done)
        lastValue: torch.Tensor,    # [B]
        *,
        gamma: float = 0.99,
        lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        T, B = rewards.shape
        device = rewards.device

        advantages = torch.zeros((T, B), device=device)
        last_advantage = torch.zeros(B, device=device)

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_value = lastValue if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
            advantages[t] = last_advantage

        returns = advantages + values
        return advantages, returns

    def Save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self._config}, path)

    @classmethod
    def Load(cls, path: str, map_location: Optional[str] = None) -> "ValueEstimationModule":
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model