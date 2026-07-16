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
    body_pose_feat: torch.Tensor


@dataclass
class EndpointControlEncoding:
    endpoint_control_tokens: torch.Tensor
    control_feedback_feat: torch.Tensor


@dataclass
class DecisionRobotStateEncoding:
    body_pose: EndpointPoseEncoding
    endpoint_control: EndpointControlEncoding


@dataclass
class MotionCommand:
    """Endpoint motion proposal emitted by the learned decision stack.

    ``decision_tensor`` is a local/body-frame SE(3) increment (metres and
    axis-angle radians); ``target_endpoint_pose`` is an absolute world-frame
    pose using XYZW quaternions. ``decision_dof_mask`` declares the physically
    modelled action coordinates: 12 full SE(3) endpoints plus all three camera
    rotations, exactly 75 active DOFs. Camera translation is never modelled.
    ``safety_scores`` are advisory model margins only;
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
    gripper_cmd: torch.Tensor
    gripper_valid: torch.Tensor
    mode_logits: torch.Tensor
    mode_valid: torch.Tensor
    safety_scores: torch.Tensor
    explanation_tokens: torch.Tensor


def CanonicalizeQuaternion(quat: torch.Tensor) -> torch.Tensor:
    canonical_index = quat.abs().argmax(dim=-1, keepdim=True)
    canonical_component = torch.gather(quat, dim=-1, index=canonical_index)
    sign = torch.where(
        canonical_component < 0.0,
        -torch.ones_like(canonical_component),
        torch.ones_like(canonical_component))
    return quat * sign


def NormalizePose(pose: torch.Tensor) -> torch.Tensor:
    quat = CanonicalizeQuaternion(
        F.normalize(pose[..., 3:7], dim=-1, eps=1e-6))
    return torch.cat([pose[..., :3], quat], dim=-1)


def AxisAngleToQuat(axisAngle: torch.Tensor) -> torch.Tensor:
    angle = axisAngle.norm(dim=-1, keepdim=True)
    half = 0.5 * angle
    quat_xyz = axisAngle * (0.5 * torch.sinc(angle / (2.0 * torch.pi)))
    quat_w = torch.cos(half)
    return torch.cat([quat_xyz, quat_w], dim=-1)


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


def RelativePose(referencePose: torch.Tensor, targetPose: torch.Tensor) -> torch.Tensor:
    """Return ``T_reference_target`` from two absolute poses in a shared frame."""
    reference_pose = NormalizePose(referencePose)
    target_pose = NormalizePose(targetPose)
    reference_quat_inv = QuatConjugate(reference_pose[..., 3:7])
    translation = QuatRotate(
        reference_quat_inv,
        target_pose[..., :3] - reference_pose[..., :3])
    relative_quat = QuatMultiply(
        reference_quat_inv,
        target_pose[..., 3:7])
    return NormalizePose(torch.cat([translation, relative_quat], dim=-1))


def RelativePoseError(commandedPose: torch.Tensor, measuredPose: torch.Tensor) -> torch.Tensor:
    """Body-frame SE(3) increment carrying ``commandedPose`` onto ``measuredPose``.

    It is the inverse of :func:`ApplyPoseDelta` over the local action domain: translation
    is expressed in the commanded/body frame and rotation is the shortest-path body-frame
    axis-angle. The same representation therefore describes measured execution increments,
    command tracking residuals, and network actions.
    """
    relative_pose = RelativePose(commandedPose, measuredPose)
    return torch.cat([
        relative_pose[..., :3],
        QuatToAxisAngle(relative_pose[..., 3:7])], dim=-1)


def DecisionActionMask(
    endpointCount: int = ModuleDim.DecisionEndpointCount,
    actionDim: int = ModuleDim.DecisionActionDim,
) -> torch.Tensor:
    mask = torch.ones(int(endpointCount), int(actionDim))
    for endpoint_index in ModuleDim.DecisionRotationOnlyEndpoints:
        if endpoint_index < int(endpointCount):
            mask[endpoint_index, :3] = 0.0
    return mask


def FlattenActiveDecisionTensor(
    decisionTensor: torch.Tensor,
) -> torch.Tensor:
    """Flatten the strict 12xSE(3) body plus 3-DOF camera boundary."""
    body = decisionTensor[
        ..., :ModuleDim.DecisionBodyEndpointCount, :].flatten(start_dim=-2)
    camera_rotation = decisionTensor[
        ..., ModuleDim.DecisionCameraEndpointIndex, 3:6]
    return torch.cat([body, camera_rotation], dim=-1)


def UnflattenActiveDecisionTensor(
    activeDecision: torch.Tensor,
) -> torch.Tensor:
    """Restore the 13x6 carrier without creating camera-translation coordinates."""
    body_dof = ModuleDim.DecisionBodyDofCount
    body = activeDecision[..., :body_dof].reshape(
        *activeDecision.shape[:-1],
        ModuleDim.DecisionBodyEndpointCount,
        ModuleDim.DecisionActionDim)
    camera = torch.cat([
        activeDecision.new_zeros(*activeDecision.shape[:-1], 3),
        activeDecision[..., body_dof:]], dim=-1).unsqueeze(-2)
    return torch.cat([body, camera], dim=-2)


class EndpointPoseEncoder(AGICoreModule):
    def __init__(
        self,
        endpointCount: int = ModuleDim.DecisionBodyEndpointCount,
        poseDim: int = ModuleDim.DecisionEndpointPoseDim,
        embedDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        hidden: int = 256,
    ):
        super().__init__()
        self.endpoint_count = int(endpointCount)
        self.pose_dim = int(poseDim)
        self.embed_dim = int(embedDim)
        self.endpoint_embed = nn.Parameter(torch.zeros(1, self.endpoint_count, self.embed_dim))
        # A fixed code preserves endpoint identity independently of learned embeddings.
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
        # The pose input is body-relative proprioception, never raw world coordinates.
        self.token_net = nn.Sequential(
            nn.LayerNorm(self.pose_dim),
            nn.Linear(self.pose_dim, hidden),
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
        endpointPoseRelative: torch.Tensor,) -> EndpointPoseEncoding:
        endpoint_pose_relative = NormalizePose(endpointPoseRelative)
        tokens = (
            self.token_net(endpoint_pose_relative)
            + self.endpoint_embed
            + self.endpoint_identity)
        body_feat = self.summary_net(tokens).mean(dim=1)
        return EndpointPoseEncoding(
            endpoint_pose_tokens=tokens,
            body_pose_feat=body_feat)


class EndpointControlEncoder(AGICoreModule):
    def __init__(
        self,
        embedDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        hidden: int = 256,
    ):
        super().__init__()
        self.endpoint_count = ModuleDim.DecisionEndpointCount
        self.action_dim = ModuleDim.DecisionActionDim
        self.embed_dim = int(embedDim)
        endpoint_control_dim = self.action_dim * 2
        camera_control_dim = ModuleDim.DecisionCameraDofCount * 2
        self.token_net = nn.Sequential(
            nn.LayerNorm(endpoint_control_dim),
            nn.Linear(endpoint_control_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)
        self.camera_token_net = nn.Sequential(
            nn.LayerNorm(camera_control_dim),
            nn.Linear(camera_control_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(2 * ModuleDim.DecisionActiveDofCount),
            nn.Linear(2 * ModuleDim.DecisionActiveDofCount, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)

    def forward(
        self,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,) -> EndpointControlEncoding:
        body_control = torch.cat([
            targetTrackingError[:, :ModuleDim.DecisionBodyEndpointCount],
            plannerTrackingError[:, :ModuleDim.DecisionBodyEndpointCount]],
            dim=-1)
        camera_control = torch.cat([
            targetTrackingError[
                :, ModuleDim.DecisionCameraEndpointIndex, 3:6],
            plannerTrackingError[
                :, ModuleDim.DecisionCameraEndpointIndex, 3:6]], dim=-1)
        active_control = torch.cat([
            FlattenActiveDecisionTensor(targetTrackingError),
            FlattenActiveDecisionTensor(plannerTrackingError)], dim=-1)
        return EndpointControlEncoding(
            endpoint_control_tokens=torch.cat([
                self.token_net(body_control),
                self.camera_token_net(camera_control).unsqueeze(1)], dim=1),
            control_feedback_feat=self.summary_net(active_control))


class ActionConstraintProjector(AGICoreModule):
    def __init__(
        self,
        endpointCount: int = ModuleDim.DecisionEndpointCount,
        actionDim: int = ModuleDim.DecisionActionDim,
        translationLimit: float = 0.05,
        rotationLimit: float = 0.25,
    ):
        super().__init__()
        action_limit = torch.empty(int(endpointCount), int(actionDim))
        action_limit[:, :3] = float(translationLimit)
        action_limit[:, 3:] = float(rotationLimit)
        self.register_buffer(
            "action_limit",
            action_limit.view(1, int(endpointCount), int(actionDim)),
            persistent=True,)
        self.register_buffer(
            "action_mask",
            DecisionActionMask(endpointCount, actionDim).view(
                1, int(endpointCount), int(actionDim)),
            persistent=True,)

    def forward(self, decisionLatent: torch.Tensor) -> torch.Tensor:
        translation = (
            decisionLatent[..., :3]
            / (1.0 + decisionLatent[..., :3].norm(dim=-1, keepdim=True))
            * self.action_limit[..., :1])
        rotation = (
            decisionLatent[..., 3:]
            / (1.0 + decisionLatent[..., 3:].norm(dim=-1, keepdim=True))
            * self.action_limit[..., 3:4])
        return torch.cat([translation, rotation], dim=-1) * self.action_mask

    def Mask(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return decisionTensor * self.action_mask

    def Normalize(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return decisionTensor / self.action_limit * self.action_mask


class EndpointActionEncoder(AGICoreModule):
    def __init__(
        self,
        outDim: int = ModuleDim.EndpointActionEmbedDim,
        hidden: int = 512,
    ):
        super().__init__()
        self.endpoint_count = ModuleDim.DecisionEndpointCount
        self.action_dim = ModuleDim.DecisionActionDim
        self.out_dim = int(outDim)
        in_dim = ModuleDim.DecisionActiveDofCount
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
        decisionTensor: torch.Tensor,) -> torch.Tensor:
        return self.net(FlattenActiveDecisionTensor(decisionTensor))


class DecisionFeedbackEncoder(AGICoreModule):
    """Closed-loop decision feedback kept separate from the World efference copy."""

    def __init__(
        self,
        stateFeatureDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        physicalReferenceDim: int = ModuleDim.RobotPhysicalReferenceDim,
        outDim: int = ModuleDim.EndpointActionEmbedDim,
        hidden: int = 512,):
        super().__init__()
        self.endpoint_count = ModuleDim.DecisionEndpointCount
        self.action_dim = ModuleDim.DecisionActionDim
        in_dim = (
            ModuleDim.DecisionActiveDofCount
            + 2 * int(stateFeatureDim)
            + int(physicalReferenceDim))
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(outDim)),
            nn.LayerNorm(int(outDim)),)

    def forward(
        self,
        decisionTensor: torch.Tensor,
        robotStateEncoding: DecisionRobotStateEncoding,
        robotPhysicalReference: torch.Tensor,) -> torch.Tensor:
        return self.net(torch.cat([
            FlattenActiveDecisionTensor(decisionTensor),
            robotStateEncoding.body_pose.body_pose_feat,
            robotStateEncoding.endpoint_control.control_feedback_feat,
            robotPhysicalReference], dim=-1))


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
        full = active.new_zeros(
            x.size(0), self.endpoint_count * self.action_dim)
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
    """Local residual heads with a strict body-SE(3)/camera-SO(3) split."""

    def __init__(
        self,
        endpointPoseTokenDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        endpointControlTokenDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        subgoalDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        motionDim: int = 256,
        dynamicsDim: int = 128,
        uncertaintyDim: int = 64,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
    ):
        super().__init__()
        local_dim = (
            int(endpointPoseTokenDim)
            + int(endpointControlTokenDim)
            + int(subgoalDim)
            + int(motionDim)
            + int(dynamicsDim)
            + int(uncertaintyDim))
        self.net = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(actionDim)),)
        self.camera_rotation_net = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, ModuleDim.DecisionCameraDofCount),)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.camera_rotation_net[-1].weight)
        nn.init.zeros_(self.camera_rotation_net[-1].bias)

    def forward(
        self,
        endpointPoseTokens: torch.Tensor,
        endpointControlTokens: torch.Tensor,
        zMotion: torch.Tensor,
        zDynamics: torch.Tensor,
        zUncertainty: torch.Tensor,
        subgoalFeature: torch.Tensor,) -> torch.Tensor:
        endpoint_count = endpointPoseTokens.size(1)
        global_local_context = torch.cat([
            zMotion,
            zDynamics,
            zUncertainty,
            subgoalFeature,
        ], dim=-1).unsqueeze(1).expand(-1, endpoint_count, -1)
        local_features = torch.cat([
            endpointPoseTokens,
            endpointControlTokens,
            global_local_context,
        ], dim=-1)
        body_action = self.net(local_features[
            :, :ModuleDim.DecisionBodyEndpointCount])
        camera_rotation = self.camera_rotation_net(local_features[
            :, ModuleDim.DecisionCameraEndpointIndex])
        camera_action = torch.cat([
            camera_rotation.new_zeros(
                camera_rotation.size(0),
                ModuleDim.DecisionActionDim - ModuleDim.DecisionCameraDofCount),
            camera_rotation], dim=-1).unsqueeze(1)
        return torch.cat([body_action, camera_action], dim=1)


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
        poseFeatureDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        physicalReferenceDim: int = ModuleDim.RobotPhysicalReferenceDim,
        endpointActionEmbedDim: int = ModuleDim.EndpointActionEmbedDim,
        translationLimit: float = 0.05,
        rotationLimit: float = 0.25,
    ):
        super().__init__()
        self.decision_dim = int(decisionDim)
        self.plan_dim = int(planDim)
        self.subgoal_feature_dim = int(subgoalFeatureDim)
        self.constraint_token_dim = int(constraintTokenDim)
        self.pose_feature_dim = int(poseFeatureDim)
        self.physical_reference_dim = int(physicalReferenceDim)
        self.endpoint_action_embed_dim = int(endpointActionEmbedDim)
        self.endpoint_count = ModuleDim.DecisionEndpointCount
        self.body_endpoint_count = ModuleDim.DecisionBodyEndpointCount
        self.pose_dim = ModuleDim.DecisionEndpointPoseDim
        self.action_dim = ModuleDim.DecisionActionDim
        self.endpoint_pose_encoder = EndpointPoseEncoder(
            endpointCount=self.body_endpoint_count,
            poseDim=self.pose_dim,
            embedDim=self.pose_feature_dim,
        )
        self.endpoint_control_encoder = EndpointControlEncoder(
            embedDim=self.pose_feature_dim,
        )
        self.action_projector = ActionConstraintProjector(
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
            translationLimit=translationLimit,
            rotationLimit=rotationLimit,)
        self.endpoint_action_encoder = EndpointActionEncoder(
            outDim=self.endpoint_action_embed_dim,
        )
        self.decision_feedback_encoder = DecisionFeedbackEncoder(
            stateFeatureDim=self.pose_feature_dim,
            physicalReferenceDim=self.physical_reference_dim,
            outDim=self.endpoint_action_embed_dim,)
        self.camera_state_token = nn.Sequential(
            nn.LayerNorm(self.physical_reference_dim),
            nn.Linear(self.physical_reference_dim, self.pose_feature_dim),
            nn.SiLU(),
            nn.Linear(self.pose_feature_dim, self.pose_feature_dim),
            nn.LayerNorm(self.pose_feature_dim),)

        factor_input = (
            self.decision_dim
            + self.plan_dim
            + self.subgoal_feature_dim
            + self.pose_feature_dim
            + self.physical_reference_dim
            + self.pose_feature_dim)
        self.factor_projector = LatentFactorProjector(factor_input)
        self.cross_attention = TaskMotionCrossAttention(tokenDim=self.constraint_token_dim)

        z_total = 256 + 256 + 128 + 128 + 64
        cross_dim = 256 * 2
        action_input = (
            z_total
            + self.pose_feature_dim
            + self.plan_dim
            + self.subgoal_feature_dim
            + cross_dim)
        dyn_input = 128 + self.pose_feature_dim + cross_dim
        residual_input = (
            64 + self.pose_feature_dim + self.subgoal_feature_dim)

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
            endpointPoseTokenDim=self.pose_feature_dim,
            endpointControlTokenDim=self.pose_feature_dim,
            subgoalDim=self.subgoal_feature_dim,
            motionDim=self.factor_projector.motion_dim,
            dynamicsDim=self.factor_projector.dyn_dim,
            uncertaintyDim=self.factor_projector.uncertainty_dim,
            actionDim=self.action_dim,)

    def ProjectDecisionLatent(self, decisionLatent: torch.Tensor) -> torch.Tensor:
        return self.action_projector(decisionLatent)

    def MaskDecisionTensor(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return self.action_projector.Mask(decisionTensor)

    def NormalizeDecisionTensor(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return self.action_projector.Normalize(decisionTensor)

    def EncodeRobotState(
        self,
        bodyEndpointPoseRelative: torch.Tensor,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,) -> DecisionRobotStateEncoding:
        return DecisionRobotStateEncoding(
            body_pose=self.endpoint_pose_encoder(bodyEndpointPoseRelative),
            endpoint_control=self.endpoint_control_encoder(
                targetTrackingError,
                plannerTrackingError))

    def EncodeEndpointAction(
        self,
        decisionTensor: torch.Tensor,) -> torch.Tensor:
        return self.endpoint_action_encoder(
            self.MaskDecisionTensor(decisionTensor))

    def EncodeDecisionFeedback(
        self,
        decisionTensor: torch.Tensor,
        robotStateEncoding: DecisionRobotStateEncoding,
        robotPhysicalReference: torch.Tensor,) -> torch.Tensor:
        return self.decision_feedback_encoder(
            self.MaskDecisionTensor(decisionTensor),
            robotStateEncoding,
            robotPhysicalReference)

    def CameraMotionFromDecisionTensor(
        self,
        decisionTensor: torch.Tensor,) -> torch.Tensor:
        camera_delta = self.MaskDecisionTensor(decisionTensor)[
            ..., ModuleDim.DecisionCameraEndpointIndex, :]
        return AxisAngleToQuat(camera_delta[..., 3:6])

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
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,) -> DecoupledDecision:
        decision_tensor = self.MaskDecisionTensor(decisionTensor)
        safety_scores = self.SafetyScores(
            decision_tensor,
            risk,
            confidence,
            precision,)
        return replace(
            decision,
            decision_latent=self.MaskDecisionTensor(decisionLatent),
            decision_tensor=decision_tensor,
            safety_scores=safety_scores,)

    def forward(
        self,
        decisionBackbone: torch.Tensor,
        planLatent: torch.Tensor,
        subgoalFeature: torch.Tensor,
        constraintTokens: torch.Tensor,
        robotStateEncoding: DecisionRobotStateEncoding,
        robotPhysicalReference: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,
    ) -> DecoupledDecision:
        endpoint_pose_encoding = robotStateEncoding.body_pose
        endpoint_control_encoding = robotStateEncoding.endpoint_control
        factors = self.factor_projector(torch.cat([
            decisionBackbone,
            planLatent,
            subgoalFeature,
            endpoint_pose_encoding.body_pose_feat,
            robotPhysicalReference,
            endpoint_control_encoding.control_feedback_feat], dim=-1))
        cross = self.cross_attention(factors["z_task"], factors["z_motion"], constraintTokens)
        z_cat = torch.cat([
            factors["z_task"],
            factors["z_motion"],
            factors["z_dyn"],
            factors["z_constraint"],
            factors["z_uncertainty"],
        ], dim=-1)

        action_in = torch.cat([
            z_cat,
            endpoint_pose_encoding.body_pose_feat,
            planLatent,
            subgoalFeature,
            cross], dim=-1)
        dyn_in = torch.cat([
            factors["z_dyn"],
            endpoint_pose_encoding.body_pose_feat,
            cross], dim=-1)
        residual_in = torch.cat([
            factors["z_uncertainty"],
            endpoint_pose_encoding.body_pose_feat,
            subgoalFeature], dim=-1)
        camera_state_token = self.camera_state_token(
            robotPhysicalReference).unsqueeze(1)
        endpoint_pose_tokens = torch.cat([
            endpoint_pose_encoding.endpoint_pose_tokens,
            camera_state_token], dim=1)

        decision_latent = self.MaskDecisionTensor(
            self.action_head(action_in)
            + 0.1 * self.dynamics_head(dyn_in)
            + 0.1 * self.residual_compensator(residual_in)
            + 0.1 * self.endpoint_action_refiner(
                endpoint_pose_tokens,
                endpoint_control_encoding.endpoint_control_tokens,
                factors["z_motion"],
                factors["z_dyn"],
                factors["z_uncertainty"],
                subgoalFeature))
        decision_tensor = self.ProjectDecisionLatent(decision_latent)
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
            gripper_cmd=constraint_out["gripper_cmd"],
            gripper_valid=constraint_out["gripper_valid"],
            mode_logits=constraint_out["mode_logits"],
            mode_valid=constraint_out["mode_valid"],
            safety_scores=safety_scores,
            explanation_tokens=explanation_tokens,
        )
