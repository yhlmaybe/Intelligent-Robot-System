from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import AGICoreModule


LOG_TWO_PI = math.log(2.0 * math.pi)


class BinaryActionDecoderBase(AGICoreModule):
    interface_kind = "binary"
    requires_deterministic_decision = False

    def __init__(self, actionEncodeDim: int = 256):
        super().__init__()
        self.action_encode_dim = int(actionEncodeDim)

    def Decode(
        self,
        actionEncode: torch.Tensor,
        *,
        sample: bool = True,
        deterministic: bool = False,) -> Dict[str, Any]:
        dist = self.Distribution(actionEncode)
        action = self.Sample(dist, deterministic=(deterministic or not sample))
        return {
            "action_dist": dist,
            "action_sample": action,
            "entropy": self.Entropy(dist),
            "interface_kind": self.interface_kind,}

    def Encode(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def Distribution(self, actionEncode: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def Sample(self, dist: Dict[str, torch.Tensor], *, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def Entropy(self, dist: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def ImitationLoss(
        self,
        dist: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class NumericActionDecoderBase(AGICoreModule):
    interface_kind = "numeric"
    requires_deterministic_decision = True

    def __init__(self, actionEncodeDim: int = 256):
        super().__init__()
        self.action_encode_dim = int(actionEncodeDim)

    def Decode(
        self,
        actionEncode: torch.Tensor,
        *,
        sample: bool = True,
        deterministic: bool = False,) -> Dict[str, Any]:
        command = self.Command(actionEncode)
        return {
            "action_command": command,
            "action_sample": command,
            "action_dist": {},
            "entropy": actionEncode.new_zeros(actionEncode.size(0)),
            "interface_kind": self.interface_kind,}

    def Encode(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def Command(self, actionEncode: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def CommandLoss(
        self,
        command: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        losses = []
        out: Dict[str, torch.Tensor] = {}
        for name, pred in command.items():
            target_value = target[name].to(device=pred.device, dtype=pred.dtype)
            if target_value.shape != pred.shape:
                target_value = target_value.view_as(pred)
            cur = F.smooth_l1_loss(pred, target_value)
            out[name] = cur
            losses.append(cur)
        total = torch.stack(losses).sum()
        out["total"] = total
        return out


class MouseKeyboardActionDecoder(BinaryActionDecoderBase):
    def __init__(
        self,
        actionEncodeDim: int,
        keyDim: int,
        actDim: int = 2,
        hidden: int = 256,
        drop: float = 0.05,
        mouseWeight: float = 0.05,
        logstdRange: tuple = (-6.0, 2.0),
    ):
        super().__init__(actionEncodeDim=actionEncodeDim)
        self.key_dim = int(keyDim)
        self.act_dim = int(actDim)
        self.hidden = int(hidden)
        self.drop = float(drop)
        self.mouse_weight = float(mouseWeight)
        self.logstd_lo, self.logstd_hi = float(logstdRange[0]), float(logstdRange[1])

        self.trunk = nn.Sequential(
            nn.LayerNorm(self.action_encode_dim),
            nn.Linear(self.action_encode_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.keys_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.key_dim),
        )
        self.click_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )
        self.mouse_mu_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.act_dim),
        )
        self.mouse_logstd_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, self.act_dim),
        )
        action_input_dim = self.key_dim + 2 + self.act_dim
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_input_dim),
            nn.Linear(action_input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.action_encode_dim),
            nn.LayerNorm(self.action_encode_dim),)

    def Encode(self, action: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys = action["keys"].float()
        mouse = action["mouse"].float()
        click = action["click"].float()
        return self.action_encoder(torch.cat([keys, mouse, click], dim=-1))

    def Distribution(self, actionEncode: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(actionEncode)
        return {
            "keys_logits": self.keys_head(h),
            "click_logits": self.click_head(h),
            "mouse_mu": self.mouse_mu_head(h),
            "mouse_logstd": self.mouse_logstd_head(h).clamp(self.logstd_lo, self.logstd_hi),
        }

    def Sample(self, dist: Dict[str, torch.Tensor], *, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        keys_logits = dist["keys_logits"]
        click_logits = dist["click_logits"]
        mu = dist["mouse_mu"]
        logstd = dist["mouse_logstd"]

        if deterministic:
            keys = (torch.sigmoid(keys_logits) > 0.5).float()
            click = (torch.sigmoid(click_logits) > 0.5).float()
            mouse = mu
            logp_mouse = (-logstd - 0.5 * LOG_TWO_PI).sum(dim=-1)
        else:
            keys_prob = torch.sigmoid(keys_logits).clamp(1e-6, 1.0 - 1e-6)
            keys = torch.bernoulli(keys_prob)
            click_prob = torch.sigmoid(click_logits).clamp(1e-6, 1.0 - 1e-6)
            click = torch.bernoulli(click_prob)
            std = torch.exp(logstd).clamp_min(1e-6)
            mouse = mu + torch.randn_like(std) * std
            zn = (mouse - mu) / std
            logp_mouse = (-0.5 * (zn.square() + 2.0 * logstd + LOG_TWO_PI)).sum(dim=-1)

        logp_keys = (keys * (-F.softplus(-keys_logits)) + (1.0 - keys) * (-F.softplus(keys_logits))).sum(-1)
        logp_click = (click * (-F.softplus(-click_logits)) + (1.0 - click) * (-F.softplus(click_logits))).sum(-1)
        return {
            "keys": keys,
            "click": click,
            "mouse": mouse,
            "logp_keys": logp_keys,
            "logp_click": logp_click,
            "logp_mouse": logp_mouse,
            "logp_total": logp_keys + logp_click + logp_mouse,
        }

    def Entropy(self, dist: Dict[str, torch.Tensor]) -> torch.Tensor:
        keys_prob = torch.sigmoid(dist["keys_logits"]).clamp(1e-6, 1.0 - 1e-6)
        click_prob = torch.sigmoid(dist["click_logits"]).clamp(1e-6, 1.0 - 1e-6)
        keys = -(keys_prob * keys_prob.log() + (1.0 - keys_prob) * (1.0 - keys_prob).log()).sum(-1)
        click = -(click_prob * click_prob.log() + (1.0 - click_prob) * (1.0 - click_prob).log()).sum(-1)
        mouse = (0.5 * (1.0 + LOG_TWO_PI) + dist["mouse_logstd"]).sum(-1)
        return keys + click + mouse

    def GaussianNll(self, mu: torch.Tensor, logstd: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        zn = (target - mu) * torch.exp(-logstd)
        return 0.5 * (zn.square() + 2.0 * logstd + LOG_TWO_PI).sum(dim=-1).mean()

    def ImitationLoss(
        self,
        dist: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        keys = F.binary_cross_entropy_with_logits(dist["keys_logits"], target["keys"].float())
        click = F.binary_cross_entropy_with_logits(dist["click_logits"], target["click"].float())
        mouse = self.GaussianNll(dist["mouse_mu"], dist["mouse_logstd"], target["mouse"].float())
        total = keys + click + self.mouse_weight * mouse
        return {"total": total, "keys": keys, "click": click, "mouse": mouse}

class JointActionCommandBase(NumericActionDecoderBase):
    """
    Base for deterministic robot-joint command heads.

    A concrete robot can subclass this and map DecisionModule output to fields
    such as joint_angles, joint_velocities, torque, or gripper.
    """

    def __init__(
        self,
        actionEncodeDim: int,
        jointDim: Optional[int] = None,
        jointDefinition: Optional[Dict[str, Dict[str, Any]]] = None,):
        super().__init__(actionEncodeDim=actionEncodeDim)
        self.joint_dim = int(jointDim or 0)
        self.joint_definition: Dict[str, Any] = {}
        self.joint_names: List[str] = []
        self.joint_index: Dict[str, int] = {}
        self.register_buffer("joint_min", torch.empty(0), persistent=False)
        self.register_buffer("joint_max", torch.empty(0), persistent=False)
        if jointDefinition is not None:
            self.InitializeJointDefinition(jointDefinition)

    def InitializeJointDefinition(self, jointDefinition: Dict[str, Dict[str, Any]]):
        """Register external robot joint definitions keyed by joint name."""
        names = [str(name) for name in jointDefinition.keys()]
        definition = {str(name): dict(spec) for name, spec in jointDefinition.items()}
        self.joint_names = names
        self.joint_definition = definition
        self.joint_index = {name: i for i, name in enumerate(names)}
        self.joint_dim = len(names)
        self.RefreshJointLimitBuffers()
        return self

    @staticmethod
    def SpecFloat(spec: Dict[str, Any], keys, default: float) -> float:
        for key in keys:
            if key in spec and spec[key] is not None:
                return float(spec[key])
        limits = spec.get("limit", spec.get("limits", None))
        if isinstance(limits, dict):
            return JointActionCommandBase.SpecFloat(limits, keys, default)
        return float(default)

    def RefreshJointLimitBuffers(self):
        mins: List[float] = []
        maxs: List[float] = []
        for name in self.joint_names:
            spec = self.joint_definition.get(name, {})
            mins.append(self.SpecFloat(
                spec,
                ("min", "lower", "angle_min", "min_angle", "lower_limit", "limit_min"),
                -float("inf")))
            maxs.append(self.SpecFloat(
                spec,
                ("max", "upper", "angle_max", "max_angle", "upper_limit", "limit_max"),
                float("inf")))
        self.joint_min = torch.tensor(mins, device=self.device, dtype=self.dtype)
        self.joint_max = torch.tensor(maxs, device=self.device, dtype=self.dtype)

    def ApplyJointLimits(self, jointAngles: torch.Tensor) -> torch.Tensor:
        joint_min = self.joint_min.to(device=jointAngles.device, dtype=jointAngles.dtype).view(1, -1)
        joint_max = self.joint_max.to(device=jointAngles.device, dtype=jointAngles.dtype).view(1, -1)
        bounded = torch.isfinite(joint_min) & torch.isfinite(joint_max)
        safe_min = torch.where(bounded, joint_min, torch.zeros_like(joint_min))
        safe_max = torch.where(bounded, joint_max, torch.ones_like(joint_max))
        center = 0.5 * (safe_min + safe_max)
        radius = 0.5 * (safe_max - safe_min).clamp_min(1e-6)
        limited = center + radius * torch.tanh(jointAngles)
        jointAngles = torch.where(bounded, limited, jointAngles)
        has_min = torch.isfinite(joint_min)
        has_max = torch.isfinite(joint_max)
        jointAngles = torch.where(has_min, torch.maximum(jointAngles, joint_min), jointAngles)
        jointAngles = torch.where(has_max, torch.minimum(jointAngles, joint_max), jointAngles)
        return jointAngles

    def JointAngles(self, actionEncode: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def JointCommandTensor(self, action: Any) -> torch.Tensor:
        vals = [action[name].view(action[name].size(0), -1)[:, 0].float() for name in self.joint_names]
        return torch.stack(vals, dim=-1)

    def FormatJointCommand(self, jointAngles: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: jointAngles[:, idx] for name, idx in self.joint_index.items()}

    def Command(self, actionEncode: torch.Tensor) -> Dict[str, torch.Tensor]:
        joint_angles = self.ApplyJointLimits(self.JointAngles(actionEncode))
        return self.FormatJointCommand(joint_angles)


class JointActionDecoder(JointActionCommandBase):
    def __init__(
        self,
        actionEncodeDim: int,
        jointDefinition: Dict[str, Dict[str, Any]],
        *,
        hidden: int = 256,
        drop: float = 0.05,
        outputScale: float = 1.0,):
        super().__init__(actionEncodeDim=actionEncodeDim, jointDefinition=jointDefinition)
        self.hidden = int(hidden)
        self.drop = float(drop)
        self.output_scale = float(outputScale)
        self.net = nn.Sequential(
            nn.LayerNorm(self.action_encode_dim),
            nn.Linear(self.action_encode_dim, self.hidden),
            nn.SiLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.joint_dim),)
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(self.joint_dim),
            nn.Linear(self.joint_dim, self.hidden),
            nn.SiLU(),
            nn.Dropout(self.drop),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.action_encode_dim),
            nn.LayerNorm(self.action_encode_dim),)

    def JointAngles(self, actionEncode: torch.Tensor) -> torch.Tensor:
        return self.net(actionEncode) * self.output_scale

    def Encode(self, action: Any) -> torch.Tensor:
        return self.action_encoder(self.JointCommandTensor(action))


class DecisionLosses:
    @staticmethod
    def ActionEncodeLoss(
        actionDecoder: Union[BinaryActionDecoderBase, NumericActionDecoderBase],
        decisionOut: Dict[str, Any],
        target: Dict[str, torch.Tensor],
        *,
        reconstructWeight: float = 1.0,) -> Dict[str, torch.Tensor]:
        action_encode = decisionOut["action_encode"]
        target_encode = actionDecoder.Encode(target)
        align = F.smooth_l1_loss(action_encode, target_encode.detach())

        decoded_target = actionDecoder.Decode(target_encode, sample=False, deterministic=True)
        if isinstance(actionDecoder, BinaryActionDecoderBase):
            reconstruct = actionDecoder.ImitationLoss(decoded_target["action_dist"], target)["total"]
        else:
            reconstruct = actionDecoder.CommandLoss(decoded_target["action_command"], target)["total"]
        total = align + float(reconstructWeight) * reconstruct
        return {
            "total": total,
            "align": align,
            "reconstruct": reconstruct,
            "target_action_encode": target_encode,}
