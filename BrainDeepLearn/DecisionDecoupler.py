from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim


@dataclass
class EndpointPoseEncoding:
    endpoint_pose_tokens: torch.Tensor
    endpoint_pose_feat: torch.Tensor


@dataclass
class MotionCommand:
    decision_tensor: torch.Tensor
    target_endpoint_pose: torch.Tensor
    endpoint_names: Tuple[str, ...]
    gripper_cmd: torch.Tensor
    mode_logits: torch.Tensor
    safety_scores: torch.Tensor


@dataclass
class DecoupledDecision:
    z_task: torch.Tensor
    z_motion: torch.Tensor
    z_dyn: torch.Tensor
    z_constraint: torch.Tensor
    z_uncertainty: torch.Tensor
    decision_tensor: torch.Tensor
    target_endpoint_pose: torch.Tensor
    decision_feedback_embed: torch.Tensor
    gripper_cmd: torch.Tensor
    mode_logits: torch.Tensor
    safety_scores: torch.Tensor
    explanation_tokens: torch.Tensor


def NormalizePose(pose: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(pose[..., 3:7], dim=-1, eps=1e-6)
    return torch.cat([pose[..., :3], quat], dim=-1)


def AxisAngleToQuat(axisAngle: torch.Tensor) -> torch.Tensor:
    angle = axisAngle.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    half = 0.5 * angle
    quat_xyz = axisAngle / angle * torch.sin(half)
    quat_w = torch.cos(half)
    return F.normalize(torch.cat([quat_xyz, quat_w], dim=-1), dim=-1, eps=1e-6)


def QuatMultiply(qA: torch.Tensor, qB: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = qA.unbind(dim=-1)
    bx, by, bz, bw = qB.unbind(dim=-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def ApplyPoseDelta(basePose: torch.Tensor, delta6: torch.Tensor) -> torch.Tensor:
    pos = basePose[..., :3] + delta6[..., :3]
    quat = QuatMultiply(basePose[..., 3:7], AxisAngleToQuat(delta6[..., 3:6]))
    return NormalizePose(torch.cat([pos, quat], dim=-1))


def QuatConjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def QuatToAxisAngle(quat: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(quat, dim=-1, eps=1e-6)
    quat = quat * (1.0 - 2.0 * (quat[..., 3:4] < 0.0).to(quat.dtype))  # w >= 0: shortest-path
    sin_half = quat[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6)
    angle = 2.0 * torch.atan2(sin_half, quat[..., 3:4].clamp(-1.0, 1.0))
    return quat[..., :3] / sin_half * angle


def RelativePoseError(commandedPose: torch.Tensor, measuredPose: torch.Tensor) -> torch.Tensor:
    """Tracking error of an achieved (measured) pose against the pose that was commanded:
    the translation residual plus the body-frame axis-angle rotation carrying the command
    onto the measurement. This is how actuator drift, joint limits and unreached setpoints
    (a commanded 30 deg that only reached 29 deg) enter the model as an explicit signal."""
    translation = measuredPose[..., :3] - commandedPose[..., :3]
    relative_quat = QuatMultiply(QuatConjugate(commandedPose[..., 3:7]), measuredPose[..., 3:7])
    return torch.cat([translation, QuatToAxisAngle(relative_quat)], dim=-1)


class EndpointPoseEncoder(AGICoreModule):
    def __init__(
        self,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        poseDim: int = ModuleDim.DecisionEndpointPoseDim,
        embedDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
    ):
        super().__init__()
        self.endpoint_count = int(endpointCount)
        self.pose_dim = int(poseDim)
        self.embed_dim = int(embedDim)
        self.action_dim = int(actionDim)
        self.endpoint_embed = nn.Parameter(torch.zeros(1, self.endpoint_count, self.embed_dim))
        # Each endpoint token carries both where it is (measured pose) and how well it tracked
        # its last command (tracking error), so the whole decision/temporal stack downstream
        # perceives actuator drift / limits, not just the current pose.
        self.token_net = nn.Sequential(
            nn.LayerNorm(self.pose_dim + self.action_dim * 2),
            nn.Linear(self.pose_dim + self.action_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.summary_net = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(
        self,
        endpointPose: torch.Tensor,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,) -> EndpointPoseEncoding:
        tokens = self.token_net(torch.cat([endpointPose, targetTrackingError, plannerTrackingError], dim=-1)) + self.endpoint_embed
        feat = self.summary_net(tokens.mean(dim=1))
        return EndpointPoseEncoding(endpoint_pose_tokens=tokens, endpoint_pose_feat=feat)


class EndpointPoseDecoder(AGICoreModule):
    def __init__(
        self,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        actionDim: int = ModuleDim.DecisionActionDim,
    ):
        super().__init__()
        action_mask = torch.ones(int(endpointCount), int(actionDim))
        for idx in ModuleDim.DecisionRotationOnlyEndpoints:
            action_mask[idx, :3] = 0.0
        self.register_buffer(
            "action_mask",
            action_mask.view(1, int(endpointCount), int(actionDim)),
            persistent=False,
        )

    def forward(self, baseEndpointPose: torch.Tensor, decisionTensor: torch.Tensor) -> torch.Tensor:
        return ApplyPoseDelta(baseEndpointPose, decisionTensor)


class DecisionFeedbackEncoder(AGICoreModule):
    def __init__(
        self,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        poseDim: int = ModuleDim.DecisionEndpointPoseDim,
        actionDim: int = ModuleDim.DecisionActionDim,
        poseFeatDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        outDim: int = ModuleDim.DecisionFeedbackEmbedDim,
        hidden: int = 512,
    ):
        super().__init__()
        self.endpoint_count = int(endpointCount)
        self.pose_dim = int(poseDim)
        self.action_dim = int(actionDim)
        self.pose_feat_dim = int(poseFeatDim)
        self.out_dim = int(outDim)
        in_dim = self.endpoint_count * self.action_dim + self.endpoint_count * self.pose_dim + self.pose_feat_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(
        self,
        decisionTensor: torch.Tensor,
        targetEndpointPose: torch.Tensor,
        endpointPoseFeat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([
            decisionTensor.reshape(decisionTensor.size(0), self.endpoint_count * self.action_dim),
            targetEndpointPose.reshape(targetEndpointPose.size(0), self.endpoint_count * self.pose_dim),
            endpointPoseFeat,
        ], dim=-1)
        return self.net(x)


class LatentFactorProjector(AGICoreModule):
    def __init__(
        self,
        inputDim: int,
        hidden: int = 512,
        taskDim: int = 256,
        motionDim: int = 256,
        dynDim: int = 128,
        constraintDim: int = 128,
        uncertaintyDim: int = 64,
    ):
        super().__init__()
        self.task_dim = int(taskDim)
        self.motion_dim = int(motionDim)
        self.dyn_dim = int(dynDim)
        self.constraint_dim = int(constraintDim)
        self.uncertainty_dim = int(uncertaintyDim)
        self.total_dim = self.task_dim + self.motion_dim + self.dyn_dim + self.constraint_dim + self.uncertainty_dim
        self.net = nn.Sequential(
            nn.LayerNorm(inputDim),
            nn.Linear(inputDim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.total_dim),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        factors = self.net(x)
        z_task, z_motion, z_dyn, z_constraint, z_uncertainty = torch.split(
            factors,
            [self.task_dim, self.motion_dim, self.dyn_dim, self.constraint_dim, self.uncertainty_dim],
            dim=-1,
        )
        return {
            "z_task": z_task,
            "z_motion": z_motion,
            "z_dyn": z_dyn,
            "z_constraint": z_constraint,
            "z_uncertainty": z_uncertainty,
        }


class TaskMotionCrossAttention(AGICoreModule):
    def __init__(
        self,
        queryDim: int = 256,
        tokenDim: int = 128,
        numHeads: int = 4,
    ):
        super().__init__()
        self.token_proj = nn.Linear(tokenDim, queryDim)
        self.attn = nn.MultiheadAttention(queryDim, int(numHeads), batch_first=True)
        self.norm_q = nn.LayerNorm(queryDim)
        self.norm_kv = nn.LayerNorm(queryDim)

    def forward(
        self,
        zTask: torch.Tensor,
        zMotion: torch.Tensor,
        constraintTokens: torch.Tensor,
    ) -> torch.Tensor:
        query = self.norm_q(torch.stack([zTask, zMotion], dim=1))
        tokens = self.norm_kv(self.token_proj(constraintTokens))
        out, _ = self.attn(query, tokens, tokens, need_weights=False)
        return out.reshape(out.size(0), -1)


class SE3ActionHead(AGICoreModule):
    def __init__(
        self,
        inputDim: int,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 512,
    ):
        super().__init__()
        self.endpoint_count = int(endpointCount)
        self.action_dim = int(actionDim)
        self.net = nn.Sequential(
            nn.LayerNorm(inputDim),
            nn.Linear(inputDim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.endpoint_count * self.action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), self.endpoint_count, self.action_dim)


class ChunkDynamicsHead(SE3ActionHead):
    def __init__(
        self,
        inputDim: int,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
    ):
        super().__init__(
            inputDim=inputDim,
            endpointCount=endpointCount,
            actionDim=actionDim,
            hidden=hidden,
        )


class ResidualErrorCompensator(SE3ActionHead):
    def __init__(
        self,
        inputDim: int,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
    ):
        super().__init__(
            inputDim=inputDim,
            endpointCount=endpointCount,
            actionDim=actionDim,
            hidden=hidden,
        )


class ConstraintHead(AGICoreModule):
    def __init__(
        self,
        inputDim: int,
        gripperCount: int = ModuleDim.ArmCount,
        modeDim: int = ModuleDim.ActTypeDim,
        safetyDim: int = 5,
        hidden: int = 256,
    ):
        super().__init__()
        self.gripper_count = int(gripperCount)
        self.trunk = nn.Sequential(
            nn.LayerNorm(inputDim),
            nn.Linear(inputDim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mode_head = nn.Linear(hidden, int(modeDim))
        self.safety_head = nn.Linear(hidden, int(safetyDim))
        self.gripper_head = nn.Linear(hidden, self.gripper_count)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "mode_logits": self.mode_head(h),
            "safety_scores": torch.sigmoid(self.safety_head(h)),
            "gripper_cmd": torch.sigmoid(self.gripper_head(h)).unsqueeze(-1),
        }


class DecisionDecouplerV2(AGICoreModule):
    def __init__(
        self,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        planDim: int = 256,
        subgoalFeatureDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        constraintTokenDim: int = 128,
        endpointPoseFeatDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        feedbackEmbedDim: int = ModuleDim.DecisionFeedbackEmbedDim,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        poseDim: int = ModuleDim.DecisionEndpointPoseDim,
        actionDim: int = ModuleDim.DecisionActionDim,
    ):
        super().__init__()
        self.decision_dim = int(decisionDim)
        self.plan_dim = int(planDim)
        self.subgoal_feature_dim = int(subgoalFeatureDim)
        self.constraint_token_dim = int(constraintTokenDim)
        self.endpoint_pose_feat_dim = int(endpointPoseFeatDim)
        self.feedback_embed_dim = int(feedbackEmbedDim)
        self.endpoint_count = int(endpointCount)
        self.pose_dim = int(poseDim)
        self.action_dim = int(actionDim)
        self.endpoint_names = tuple(ModuleDim.DecisionEndpointNames)

        self.endpoint_pose_encoder = EndpointPoseEncoder(
            endpointCount=self.endpoint_count,
            poseDim=self.pose_dim,
            embedDim=self.endpoint_pose_feat_dim,
        )
        self.endpoint_pose_decoder = EndpointPoseDecoder()
        self.decision_feedback_encoder = DecisionFeedbackEncoder(
            endpointCount=self.endpoint_count,
            poseDim=self.pose_dim,
            actionDim=self.action_dim,
            poseFeatDim=self.endpoint_pose_feat_dim,
            outDim=self.feedback_embed_dim,
        )

        factor_input = self.decision_dim + self.plan_dim + self.subgoal_feature_dim + self.endpoint_pose_feat_dim
        self.factor_projector = LatentFactorProjector(factor_input)
        self.cross_attention = TaskMotionCrossAttention(tokenDim=self.constraint_token_dim)

        z_total = 256 + 256 + 128 + 128 + 64
        cross_dim = 256 * 2
        action_input = z_total + self.endpoint_pose_feat_dim + self.plan_dim + self.subgoal_feature_dim + cross_dim
        dyn_input = 128 + self.endpoint_pose_feat_dim + cross_dim
        constraint_input = 128 + 64 + self.endpoint_pose_feat_dim + cross_dim
        residual_input = 64 + self.endpoint_pose_feat_dim + self.subgoal_feature_dim

        self.action_head = SE3ActionHead(
            action_input,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
        )
        self.dynamics_head = ChunkDynamicsHead(
            dyn_input,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
        )
        self.constraint_head = ConstraintHead(constraint_input)
        self.residual_compensator = ResidualErrorCompensator(
            residual_input,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
        )
        self.explanation_head = nn.Sequential(
            nn.LayerNorm(z_total + cross_dim),
            nn.Linear(z_total + cross_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.constraint_token_dim),
        )

    def EncodeEndpointPose(
        self,
        endpointPose: torch.Tensor,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,) -> EndpointPoseEncoding:
        return self.endpoint_pose_encoder(endpointPose, targetTrackingError, plannerTrackingError)

    def MaskDecisionTensor(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        mask = self.endpoint_pose_decoder.action_mask.reshape(
            *([1] * (decisionTensor.dim() - 2)),
            self.endpoint_count,
            self.action_dim)
        return decisionTensor * mask

    def DecodeEndpointPose(self, baseEndpointPose: torch.Tensor, decisionTensor: torch.Tensor) -> torch.Tensor:
        return self.endpoint_pose_decoder(baseEndpointPose, self.MaskDecisionTensor(decisionTensor))

    def EncodeDecisionFeedback(
        self,
        decisionTensor: torch.Tensor,
        targetEndpointPose: torch.Tensor,
        endpointPoseEncoding: EndpointPoseEncoding,
    ) -> torch.Tensor:
        return self.decision_feedback_encoder(
            decisionTensor,
            targetEndpointPose,
            endpointPoseEncoding.endpoint_pose_feat,
        )

    def ToMotionCommand(self, decision: DecoupledDecision) -> MotionCommand:
        return MotionCommand(
            decision_tensor=decision.decision_tensor,
            target_endpoint_pose=decision.target_endpoint_pose,
            endpoint_names=self.endpoint_names,
            gripper_cmd=decision.gripper_cmd,
            mode_logits=decision.mode_logits,
            safety_scores=decision.safety_scores,
        )

    def forward(
        self,
        decisionBackbone: torch.Tensor,
        planLatent: torch.Tensor,
        subgoalFeature: torch.Tensor,
        constraintTokens: torch.Tensor,
        endpointPoseEncoding: EndpointPoseEncoding,
        baseEndpointPose: torch.Tensor,
    ) -> DecoupledDecision:
        endpoint_feat = endpointPoseEncoding.endpoint_pose_feat
        factors = self.factor_projector(torch.cat([decisionBackbone, planLatent, subgoalFeature, endpoint_feat], dim=-1))
        cross = self.cross_attention(factors["z_task"], factors["z_motion"], constraintTokens)
        z_cat = torch.cat([
            factors["z_task"],
            factors["z_motion"],
            factors["z_dyn"],
            factors["z_constraint"],
            factors["z_uncertainty"],
        ], dim=-1)

        action_in = torch.cat([z_cat, endpoint_feat, planLatent, subgoalFeature, cross], dim=-1)
        dyn_in = torch.cat([factors["z_dyn"], endpoint_feat, cross], dim=-1)
        constraint_in = torch.cat([factors["z_constraint"], factors["z_uncertainty"], endpoint_feat, cross], dim=-1)
        residual_in = torch.cat([factors["z_uncertainty"], endpoint_feat, subgoalFeature], dim=-1)

        decision_tensor = self.MaskDecisionTensor(
            self.action_head(action_in)
            + 0.1 * self.dynamics_head(dyn_in)
            + 0.1 * self.residual_compensator(residual_in))
        target_endpoint_pose = self.DecodeEndpointPose(baseEndpointPose, decision_tensor)
        decision_feedback_embed = self.EncodeDecisionFeedback(
            decision_tensor,
            target_endpoint_pose,
            endpointPoseEncoding,
        )
        constraint_out = self.constraint_head(constraint_in)
        explanation_tokens = self.explanation_head(torch.cat([z_cat, cross], dim=-1))

        return DecoupledDecision(
            z_task=factors["z_task"],
            z_motion=factors["z_motion"],
            z_dyn=factors["z_dyn"],
            z_constraint=factors["z_constraint"],
            z_uncertainty=factors["z_uncertainty"],
            decision_tensor=decision_tensor,
            target_endpoint_pose=target_endpoint_pose,
            decision_feedback_embed=decision_feedback_embed,
            gripper_cmd=constraint_out["gripper_cmd"],
            mode_logits=constraint_out["mode_logits"],
            safety_scores=constraint_out["safety_scores"],
            explanation_tokens=explanation_tokens,
        )
