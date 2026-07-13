from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim


SAFETY_MARGIN_NAMES = (
    "model_translation_step_margin",
    "model_rotation_step_margin",
    "learned_current_state_risk_margin",
    "learned_current_state_confidence",
    "learned_current_state_precision",
)


@dataclass
class EndpointPoseEncoding:
    endpoint_pose_tokens: torch.Tensor
    endpoint_pose_feat: torch.Tensor


@dataclass
class MotionCommand:
    """Endpoint motion proposal emitted by the learned decision stack.

    ``decision_tensor`` is a local/body-frame SE(3) increment (metres and
    axis-angle radians); ``target_endpoint_pose`` is an absolute world-frame
    pose using XYZW quaternions. ``decision_dof_mask`` identifies modelled
    increment components. ``safety_scores`` are advisory model margins only;
    they do not prove IK feasibility, joint limits, collision freedom or
    actuator safety.
    """

    decision_tensor: torch.Tensor
    target_endpoint_pose: torch.Tensor
    endpoint_names: Tuple[str, ...]
    decision_dof_mask: torch.Tensor
    gripper_cmd: torch.Tensor
    gripper_valid: torch.Tensor
    mode_logits: torch.Tensor
    mode_valid: torch.Tensor
    safety_scores: torch.Tensor
    safety_names: Tuple[str, ...]


@dataclass
class DecoupledDecision:
    z_task: torch.Tensor
    z_motion: torch.Tensor
    z_dyn: torch.Tensor
    z_constraint: torch.Tensor
    z_uncertainty: torch.Tensor
    decision_latent: torch.Tensor
    decision_tensor: torch.Tensor
    target_endpoint_pose: torch.Tensor
    gripper_cmd: torch.Tensor
    gripper_valid: torch.Tensor
    mode_logits: torch.Tensor
    mode_valid: torch.Tensor
    safety_scores: torch.Tensor
    explanation_tokens: torch.Tensor


def NormalizePose(pose: torch.Tensor) -> torch.Tensor:
    quat = F.normalize(pose[..., 3:7], dim=-1, eps=1e-6)
    canonical_index = quat.abs().argmax(dim=-1, keepdim=True)
    canonical_component = torch.gather(quat, dim=-1, index=canonical_index)
    quat = quat * (1.0 - 2.0 * (canonical_component < 0.0).to(quat.dtype))
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


def QuatRotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = quat[..., :3]
    uv = torch.cross(q_xyz, vector, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vector + 2.0 * (quat[..., 3:4] * uv + uuv)


def ApplyPoseDelta(basePose: torch.Tensor, delta6: torch.Tensor) -> torch.Tensor:
    """Compose a local body-frame translation and axis-angle rotation with a base pose."""
    base_pose = NormalizePose(basePose)
    pos = base_pose[..., :3] + QuatRotate(base_pose[..., 3:7], delta6[..., :3])
    quat = QuatMultiply(base_pose[..., 3:7], AxisAngleToQuat(delta6[..., 3:6]))
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
    """Body-frame SE(3) increment carrying ``commandedPose`` onto ``measuredPose``.

    It is the inverse of :func:`ApplyPoseDelta` over the local action domain: translation
    is expressed in the commanded/body frame and rotation is the shortest-path body-frame
    axis-angle. The same representation therefore describes measured execution increments,
    command tracking residuals, and network actions.
    """
    commanded_pose = NormalizePose(commandedPose)
    measured_pose = NormalizePose(measuredPose)
    commanded_quat_inv = QuatConjugate(commanded_pose[..., 3:7])
    translation = QuatRotate(
        commanded_quat_inv,
        measured_pose[..., :3] - commanded_pose[..., :3])
    relative_quat = QuatMultiply(commanded_quat_inv, measured_pose[..., 3:7])
    return torch.cat([translation, QuatToAxisAngle(relative_quat)], dim=-1)


def DecisionActionMask(
    endpointCount: int = ModuleDim.DecisionEndpointCount,
    actionDim: int = ModuleDim.DecisionActionDim,
) -> torch.Tensor:
    mask = torch.ones(int(endpointCount), int(actionDim))
    for idx in ModuleDim.DecisionRotationOnlyEndpoints:
        mask[idx, :3] = 0.0
        mask[idx, 5] = 0.0
    return mask


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
        # A fixed code preserves endpoint identity when loading older checkpoints whose
        # learned endpoint embeddings may all be equal.
        position = torch.arange(self.endpoint_count, dtype=torch.float32).unsqueeze(1)
        frequency = 10000.0 ** (
            -torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
            / max(self.embed_dim, 1))
        endpoint_identity = torch.zeros(self.endpoint_count, self.embed_dim, dtype=torch.float32)
        endpoint_identity[:, 0::2] = torch.sin(position * frequency)
        endpoint_identity[:, 1::2] = torch.cos(
            position * frequency[:endpoint_identity[:, 1::2].size(1)])
        self.register_buffer(
            "endpoint_identity",
            endpoint_identity.unsqueeze(0),
            persistent=False,
        )
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
        endpointPose = NormalizePose(endpointPose)
        tokens = (
            self.token_net(torch.cat([endpointPose, targetTrackingError, plannerTrackingError], dim=-1))
            + self.endpoint_embed
            + self.endpoint_identity)
        # Keep identity attached through a nonlinear map; averaging first would reduce
        # every endpoint embedding to the same input-independent constant.
        feat = self.summary_net(tokens).mean(dim=1)
        return EndpointPoseEncoding(endpoint_pose_tokens=tokens, endpoint_pose_feat=feat)


class ActionConstraintProjector(AGICoreModule):
    def __init__(
        self,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        actionDim: int = ModuleDim.DecisionActionDim,
        translationLimit: float = 0.05,
        rotationLimit: float = 0.25,
    ):
        super().__init__()
        action_mask = DecisionActionMask(endpointCount, actionDim)
        self.register_buffer(
            "action_mask",
            action_mask.view(1, int(endpointCount), int(actionDim)),
            persistent=False,
        )
        action_limit = torch.empty(int(endpointCount), int(actionDim))
        action_limit[:, :3] = float(translationLimit)
        action_limit[:, 3:] = float(rotationLimit)
        self.register_buffer(
            "action_limit",
            action_limit.view(1, int(endpointCount), int(actionDim)),
            persistent=True,)

    def forward(self, decisionLatent: torch.Tensor) -> torch.Tensor:
        masked = decisionLatent * self.action_mask
        translation = (
            masked[..., :3]
            / (1.0 + masked[..., :3].norm(dim=-1, keepdim=True))
            * self.action_limit[..., :1])
        rotation = (
            masked[..., 3:]
            / (1.0 + masked[..., 3:].norm(dim=-1, keepdim=True))
            * self.action_limit[..., 3:4])
        return torch.cat([translation, rotation], dim=-1)

    def Normalize(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return decisionTensor / self.action_limit * self.action_mask

    def Mask(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        mask = self.action_mask.reshape(
            *([1] * (decisionTensor.dim() - 2)),
            self.action_mask.size(-2),
            self.action_mask.size(-1))
        return decisionTensor * mask


class EndpointPoseDecoder(AGICoreModule):
    def __init__(self):
        super().__init__()

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
        active_flat_indices = DecisionActionMask(
            self.endpoint_count,
            self.action_dim).reshape(-1).nonzero().flatten()
        self.register_buffer(
            "active_flat_indices",
            active_flat_indices,
            persistent=False)
        self.net = nn.Sequential(
            nn.LayerNorm(inputDim),
            nn.Linear(inputDim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(active_flat_indices.numel())),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active = self.net(x)
        full = active.new_zeros(x.size(0), self.endpoint_count * self.action_dim)
        full = full.scatter(
            1,
            self.active_flat_indices.unsqueeze(0).expand(x.size(0), -1),
            active)
        return full.view(x.size(0), self.endpoint_count, self.action_dim)


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


class EndpointActionRefiner(AGICoreModule):
    """Shared local residual head that preserves per-endpoint pose/error identity."""

    def __init__(
        self,
        endpointTokenDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        subgoalDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        motionDim: int = 256,
        dynamicsDim: int = 128,
        uncertaintyDim: int = 64,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
    ):
        super().__init__()
        local_dim = (
            int(endpointTokenDim)
            + int(subgoalDim)
            + int(motionDim)
            + int(dynamicsDim)
            + int(uncertaintyDim))
        self.net = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(actionDim)),)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        endpointTokens: torch.Tensor,
        zMotion: torch.Tensor,
        zDynamics: torch.Tensor,
        zUncertainty: torch.Tensor,
        subgoalFeature: torch.Tensor,) -> torch.Tensor:
        endpoint_count = endpointTokens.size(1)
        global_local_context = torch.cat([
            zMotion,
            zDynamics,
            zUncertainty,
            subgoalFeature,
        ], dim=-1).unsqueeze(1).expand(-1, endpoint_count, -1)
        return self.net(torch.cat([
            endpointTokens,
            global_local_context,
        ], dim=-1))


class DiscreteCommandContract(AGICoreModule):
    def __init__(
        self,
        gripperCount: int = ModuleDim.ArmCount,
        modeDim: int = ModuleDim.ActTypeDim,
    ):
        super().__init__()
        self.gripper_count = int(gripperCount)
        self.mode_dim = int(modeDim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "mode_logits": x.new_zeros(x.size(0), self.mode_dim),
            "gripper_cmd": x.new_full((x.size(0), self.gripper_count, 1), 0.5),
            "mode_valid": torch.zeros(x.size(0), device=x.device, dtype=torch.bool),
            "gripper_valid": torch.zeros(x.size(0), device=x.device, dtype=torch.bool),
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
        translationLimit: float = 0.05,
        rotationLimit: float = 0.25,
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
        self.action_projector = ActionConstraintProjector(
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
            translationLimit=translationLimit,
            rotationLimit=rotationLimit,)
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
        self.discrete_command_contract = DiscreteCommandContract()
        self.residual_compensator = ResidualErrorCompensator(
            residual_input,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
        )
        self.endpoint_action_refiner = EndpointActionRefiner(
            endpointTokenDim=self.endpoint_pose_feat_dim,
            subgoalDim=self.subgoal_feature_dim,
            motionDim=self.factor_projector.motion_dim,
            dynamicsDim=self.factor_projector.dyn_dim,
            uncertaintyDim=self.factor_projector.uncertainty_dim,
            actionDim=self.action_dim,)

    def EncodeEndpointPose(
        self,
        endpointPose: torch.Tensor,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,) -> EndpointPoseEncoding:
        return self.endpoint_pose_encoder(endpointPose, targetTrackingError, plannerTrackingError)

    def MaskDecisionTensor(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return self.action_projector.Mask(decisionTensor)

    def ProjectDecisionLatent(self, decisionLatent: torch.Tensor) -> torch.Tensor:
        return self.action_projector(decisionLatent)

    def NormalizeDecisionTensor(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return self.action_projector.Normalize(decisionTensor)

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
        decision_dof_mask = self.action_projector.action_mask.expand(
            decision.decision_tensor.size(0), -1, -1).bool()
        return MotionCommand(
            decision_tensor=decision.decision_tensor,
            target_endpoint_pose=decision.target_endpoint_pose,
            endpoint_names=self.endpoint_names,
            decision_dof_mask=decision_dof_mask,
            gripper_cmd=decision.gripper_cmd,
            gripper_valid=decision.gripper_valid,
            mode_logits=decision.mode_logits,
            mode_valid=decision.mode_valid,
            safety_scores=decision.safety_scores,
            safety_names=SAFETY_MARGIN_NAMES,
        )

    def RebaseMotionCommand(
        self,
        command: MotionCommand,
        currentEndpointPose: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,) -> MotionCommand:
        target_endpoint_pose = command.target_endpoint_pose.clone()
        for endpoint_index in ModuleDim.DecisionRotationOnlyEndpoints:
            target_endpoint_pose[:, endpoint_index, :3] = (
                currentEndpointPose[:, endpoint_index, :3])
        remaining_decision = RelativePoseError(
            currentEndpointPose,
            target_endpoint_pose)
        remaining_decision = self.MaskDecisionTensor(remaining_decision)
        return replace(
            command,
            decision_tensor=remaining_decision,
            target_endpoint_pose=target_endpoint_pose,
            safety_scores=self.SafetyScores(
                remaining_decision,
                risk,
                confidence,
                precision))

    def SafetyScores(
        self,
        decisionTensor: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,) -> torch.Tensor:
        normalized = self.NormalizeDecisionTensor(decisionTensor).abs()
        translation_margin = 1.0 - normalized[..., :3].norm(dim=-1).amax(dim=-1)
        rotation_margin = 1.0 - normalized[..., 3:].norm(dim=-1).amax(dim=-1)
        return torch.stack([
            translation_margin,
            rotation_margin,
            1.0 - risk,
            confidence,
            precision,], dim=-1)

    def ReplaceAction(
        self,
        decision: DecoupledDecision,
        decisionLatent: torch.Tensor,
        decisionTensor: torch.Tensor,
        baseEndpointPose: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,) -> DecoupledDecision:
        decision_tensor = self.MaskDecisionTensor(decisionTensor)
        target_endpoint_pose = self.DecodeEndpointPose(baseEndpointPose, decision_tensor)
        safety_scores = self.SafetyScores(
            decision_tensor,
            risk,
            confidence,
            precision,)
        return replace(
            decision,
            decision_latent=decisionLatent,
            decision_tensor=decision_tensor,
            target_endpoint_pose=target_endpoint_pose,
            safety_scores=safety_scores,)

    def forward(
        self,
        decisionBackbone: torch.Tensor,
        planLatent: torch.Tensor,
        subgoalFeature: torch.Tensor,
        constraintTokens: torch.Tensor,
        endpointPoseEncoding: EndpointPoseEncoding,
        baseEndpointPose: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,
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
        residual_in = torch.cat([factors["z_uncertainty"], endpoint_feat, subgoalFeature], dim=-1)

        decision_latent = (
            self.action_head(action_in)
            + 0.1 * self.dynamics_head(dyn_in)
            + 0.1 * self.residual_compensator(residual_in)
            + 0.1 * self.endpoint_action_refiner(
                endpointPoseEncoding.endpoint_pose_tokens,
                factors["z_motion"],
                factors["z_dyn"],
                factors["z_uncertainty"],
                subgoalFeature))
        decision_tensor = self.ProjectDecisionLatent(decision_latent)
        target_endpoint_pose = self.DecodeEndpointPose(baseEndpointPose, decision_tensor)
        constraint_out = self.discrete_command_contract(factors["z_constraint"])
        safety_scores = self.SafetyScores(decision_tensor, risk, confidence, precision)
        explanation_tokens = constraintTokens.mean(dim=1)

        return DecoupledDecision(
            z_task=factors["z_task"],
            z_motion=factors["z_motion"],
            z_dyn=factors["z_dyn"],
            z_constraint=factors["z_constraint"],
            z_uncertainty=factors["z_uncertainty"],
            decision_latent=decision_latent,
            decision_tensor=decision_tensor,
            target_endpoint_pose=target_endpoint_pose,
            gripper_cmd=constraint_out["gripper_cmd"],
            gripper_valid=constraint_out["gripper_valid"],
            mode_logits=constraint_out["mode_logits"],
            mode_valid=constraint_out["mode_valid"],
            safety_scores=safety_scores,
            explanation_tokens=explanation_tokens,
        )
