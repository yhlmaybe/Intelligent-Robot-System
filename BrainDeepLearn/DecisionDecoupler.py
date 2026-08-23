from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import (
    BODY_CAPABILITY_NAMES,
    BODY_ROLE_NAMES,
    BODY_SIDE_NAMES,
    JOINT_TYPE_NAMES,)


SAFETY_MARGIN_NAMES = (
    "model_translation_step_margin",
    "model_rotation_step_margin",
    "learned_current_state_risk_margin",
    "learned_current_state_confidence",
    "learned_current_state_precision",
)


def _ActualCount(value: Optional[int], name: str) -> int:
    if value is None:
        raise TypeError(f"{name} must be provided from robot morphology")
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


@dataclass
class EndpointPoseEncoding:
    endpoint_pose_tokens: torch.Tensor
    endpoint_pose_feat: torch.Tensor


@dataclass
class EndpointControlEncoding:
    endpoint_control_tokens: torch.Tensor
    control_feedback_feat: torch.Tensor


@dataclass
class DecisionRobotStateEncoding:
    endpoint_pose: EndpointPoseEncoding
    endpoint_control: EndpointControlEncoding
    joint_feature: torch.Tensor
    joint_tokens: torch.Tensor
    joint_state_valid: torch.Tensor
    joint_controllable: torch.Tensor
    endpoint_state_valid: torch.Tensor
    endpoint_controllable: torch.Tensor


@dataclass
class MotionCommand:
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
    joint_variable_command: torch.Tensor
    joint_variable_command_mask: torch.Tensor
    joint_variable_names: Tuple[str, ...]


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
    joint_variable_command: torch.Tensor
    joint_variable_command_mask: torch.Tensor


def _EndpointContract(
    robotMorphology: Optional[Any],
) -> Tuple[torch.Tensor, Tuple[str, ...]]:
    action_dim = int(ModuleDim.RobotControlAxisDim)
    if robotMorphology is None:
        raise TypeError("robot morphology is required")
    endpoint_count = int(robotMorphology.endpoint_count)
    gripper_count = int(robotMorphology.gripper_count)
    if endpoint_count < 0 or gripper_count < 0:
        raise ValueError("morphology control counts must be non-negative")
    endpoint_task_mask = torch.as_tensor(
        robotMorphology.endpoint_task_mask, dtype=torch.bool).detach().cpu()
    if tuple(endpoint_task_mask.shape) != (endpoint_count, action_dim):
        raise ValueError("morphology endpoint task mask does not match count")
    endpoint_names = tuple(str(name) for name in robotMorphology.endpoint_names)
    if len(endpoint_names) != endpoint_count:
        raise ValueError("morphology endpoint names do not match endpoint count")
    if len(tuple(robotMorphology.gripper_names)) != gripper_count:
        raise ValueError("morphology gripper names do not match gripper count")
    return endpoint_task_mask, endpoint_names


def _EndpointSemanticDescriptor(
    robotMorphology: Optional[Any],
) -> torch.Tensor:
    if robotMorphology is None:
        raise TypeError("robot morphology is required")
    endpoint_count = int(robotMorphology.endpoint_count)
    if not hasattr(robotMorphology, "EndpointSemanticDescriptor"):
        raise TypeError("morphology endpoint descriptor is missing")
    semantic = robotMorphology.EndpointSemanticDescriptor()
    required = (
        "controllable",
        "topology_depth",
        "task_mask",
        "role",
        "side",
        "capability",
        "node_role",
        "node_side",
        "node_capability",
        "parent_node_index",
        "parent_role",
        "parent_side",
        "parent_capability",
        "group_role_membership",
        "group_side_membership",
        "group_capability",
    )
    missing = tuple(name for name in required if name not in semantic)
    if missing:
        raise TypeError(
            "morphology endpoint descriptor is incomplete: "
            + ", ".join(missing))

    def vector(name: str, dtype: torch.dtype) -> torch.Tensor:
        value = torch.as_tensor(
            semantic[name], dtype=dtype).detach().cpu()
        if tuple(value.shape) != (endpoint_count,):
            raise ValueError(
                f"morphology endpoint descriptor {name} shape is invalid")
        return value

    def matrix(
        name: str,
        width: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            semantic[name], dtype=dtype).detach().cpu()
        if tuple(value.shape) != (endpoint_count, int(width)):
            raise ValueError(
                f"morphology endpoint descriptor {name} shape is invalid")
        return value

    role_names = ("role", "node_role", "parent_role")
    side_names = ("side", "node_side", "parent_side")
    role = {name: vector(name, torch.long) for name in role_names}
    side = {name: vector(name, torch.long) for name in side_names}
    parent_index = vector("parent_node_index", torch.long)
    parent_valid = parent_index.ge(0)
    for name in ("role", "node_role"):
        if endpoint_count and bool(
            ((role[name] < 0) | (role[name] >= len(BODY_ROLE_NAMES))).any().item()
        ):
            raise ValueError("morphology endpoint body role is invalid")
    for name in ("side", "node_side"):
        if endpoint_count and bool(
            ((side[name] < 0) | (side[name] >= len(BODY_SIDE_NAMES))).any().item()
        ):
            raise ValueError("morphology endpoint body side is invalid")
    if bool((
        ((role["parent_role"] < 0)
         | (role["parent_role"] >= len(BODY_ROLE_NAMES)))
        & parent_valid
    ).any().item()):
        raise ValueError("morphology endpoint parent role is invalid")
    if bool((
        ((side["parent_side"] < 0)
         | (side["parent_side"] >= len(BODY_SIDE_NAMES)))
        & parent_valid
    ).any().item()):
        raise ValueError("morphology endpoint parent side is invalid")
    parent_valid_f = parent_valid.to(torch.float32).unsqueeze(-1)
    topology_depth = vector("topology_depth", torch.float32)
    node_count = int(robotMorphology.node_count)
    if node_count < 1:
        raise ValueError("robot morphology must contain a node")
    task_mask = matrix(
        "task_mask",
        ModuleDim.RobotControlAxisDim,
        torch.bool).to(torch.float32)
    controllable = vector("controllable", torch.bool)
    if not torch.equal(controllable, task_mask.bool().any(dim=-1)):
        raise ValueError("morphology endpoint controllability is inconsistent")
    return torch.cat([
        F.one_hot(
            role["role"],
            num_classes=len(BODY_ROLE_NAMES)).to(torch.float32),
        F.one_hot(
            side["side"],
            num_classes=len(BODY_SIDE_NAMES)).to(torch.float32),
        matrix("capability", len(BODY_CAPABILITY_NAMES)),
        F.one_hot(
            role["node_role"],
            num_classes=len(BODY_ROLE_NAMES)).to(torch.float32),
        F.one_hot(
            side["node_side"],
            num_classes=len(BODY_SIDE_NAMES)).to(torch.float32),
        matrix("node_capability", len(BODY_CAPABILITY_NAMES)),
        F.one_hot(
            role["parent_role"].clamp(0, len(BODY_ROLE_NAMES) - 1),
            num_classes=len(BODY_ROLE_NAMES)).to(torch.float32)
            * parent_valid_f,
        F.one_hot(
            side["parent_side"].clamp(0, len(BODY_SIDE_NAMES) - 1),
            num_classes=len(BODY_SIDE_NAMES)).to(torch.float32)
            * parent_valid_f,
        matrix("parent_capability", len(BODY_CAPABILITY_NAMES))
            * parent_valid_f,
        matrix("group_role_membership", len(BODY_ROLE_NAMES)),
        matrix("group_side_membership", len(BODY_SIDE_NAMES)),
        matrix("group_capability", len(BODY_CAPABILITY_NAMES)),
        (topology_depth / float(node_count)).unsqueeze(-1),
        task_mask,
        controllable.to(torch.float32).unsqueeze(-1),
    ], dim=-1)


def _JointSemanticContract(
    robotMorphology: Optional[Any],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    descriptor_dim = (
        len(JOINT_TYPE_NAMES)
        + 20
        + 3 * len(BODY_ROLE_NAMES)
        + 3 * len(BODY_SIDE_NAMES)
        + 3 * len(BODY_CAPABILITY_NAMES))
    if robotMorphology is None:
        raise TypeError("robot morphology is required")
    required = (
        "node_count",
        "joint_dof_count",
        "joint_variable_names",
        "joint_variable_commandable",
        "joint_lower",
        "joint_upper",
        "joint_effort_limit",
        "joint_velocity_limit",
        "joint_variable_command_delta_scale",
        "joint_variable_unit",
        "joint_variable_command_representation",
        "joint_variable_command_reference",
        "joint_variable_command_range",
        "joint_variable_command_limit_policy",
        "JointSemanticDescriptor",
    )
    missing = tuple(
        name for name in required if not hasattr(robotMorphology, name))
    if missing:
        raise TypeError(
            "morphology joint semantic contract is incomplete: "
            + ", ".join(missing))
    joint_count = int(robotMorphology.joint_dof_count)
    if joint_count < 0:
        raise ValueError("morphology joint dof count is negative")
    if len(tuple(robotMorphology.joint_variable_names)) != joint_count:
        raise ValueError("morphology joint variable names do not match dof count")
    joint_units = tuple(robotMorphology.joint_variable_unit)
    if len(joint_units) != joint_count or any(
        unit not in ("radian", "meter") for unit in joint_units
    ):
        raise ValueError("morphology joint variable units are invalid")
    if robotMorphology.joint_variable_command_representation != (
        "normalized_position_delta"
    ):
        raise ValueError("morphology joint command representation is invalid")
    if robotMorphology.joint_variable_command_reference != (
        "current_measured_position_at_sensor_frame_exposure"
    ):
        raise ValueError("morphology joint command reference is invalid")
    if tuple(robotMorphology.joint_variable_command_range) != (-1.0, 1.0):
        raise ValueError("morphology joint command range is invalid")
    if robotMorphology.joint_variable_command_limit_policy != (
        "clamp_finite_limits_wrap_unbounded_rotation"
    ):
        raise ValueError("morphology joint command limit policy is invalid")
    commandable = torch.as_tensor(
        robotMorphology.joint_variable_commandable,
        dtype=torch.bool).detach().cpu()
    lower = torch.as_tensor(
        robotMorphology.joint_lower,
        dtype=torch.float32).detach().cpu()
    upper = torch.as_tensor(
        robotMorphology.joint_upper,
        dtype=torch.float32).detach().cpu()
    effort_limit = torch.as_tensor(
        robotMorphology.joint_effort_limit,
        dtype=torch.float32).detach().cpu()
    velocity_limit = torch.as_tensor(
        robotMorphology.joint_velocity_limit,
        dtype=torch.float32).detach().cpu()
    command_delta_scale = torch.as_tensor(
        robotMorphology.joint_variable_command_delta_scale,
        dtype=torch.float32).detach().cpu()
    for value in (
        commandable,
        lower,
        upper,
        effort_limit,
        velocity_limit,
        command_delta_scale,
    ):
        if tuple(value.shape) != (joint_count,):
            raise ValueError("morphology joint variable shape does not match dof count")
    semantic = robotMorphology.JointSemanticDescriptor()
    semantic_names = (
        "commandable",
        "local_index",
        "topology_depth",
        "joint_type",
        "joint_axis",
        "child_role",
        "child_side",
        "child_capability",
        "parent_node_index",
        "parent_role",
        "parent_side",
        "parent_capability",
        "group_role_membership",
        "group_side_membership",
        "group_capability",
        "lower_limit_normalized",
        "upper_limit_normalized",
        "position_lower_limit_valid",
        "position_upper_limit_valid",
        "effort_limit_normalized",
        "effort_limit_valid",
        "velocity_limit_normalized",
        "velocity_limit_valid",
        "command_delta_scale",
    )
    semantic_missing = tuple(
        name for name in semantic_names if name not in semantic)
    if semantic_missing:
        raise TypeError(
            "morphology joint descriptor is incomplete: "
            + ", ".join(semantic_missing))

    def vector(name: str, dtype: torch.dtype) -> torch.Tensor:
        value = torch.as_tensor(
            semantic[name], dtype=dtype).detach().cpu()
        if tuple(value.shape) != (joint_count,):
            raise ValueError(
                f"morphology joint descriptor {name} shape is invalid")
        return value

    def matrix(
        name: str,
        width: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        value = torch.as_tensor(
            semantic[name], dtype=dtype).detach().cpu()
        if tuple(value.shape) != (joint_count, int(width)):
            raise ValueError(
                f"morphology joint descriptor {name} shape is invalid")
        return value

    semantic_commandable = vector("commandable", torch.bool)
    if not torch.equal(semantic_commandable, commandable):
        raise ValueError("morphology joint descriptor commandability is inconsistent")
    semantic_command_delta_scale = vector(
        "command_delta_scale", torch.float32)
    if not torch.equal(semantic_command_delta_scale, command_delta_scale):
        raise ValueError("morphology joint command scale is inconsistent")
    if bool((
        ~torch.isfinite(command_delta_scale)
        | command_delta_scale.le(0.0)
    ).any().item()):
        raise ValueError("morphology joint command scale is invalid")
    local_index = vector("local_index", torch.long)
    joint_type = vector("joint_type", torch.long)
    if joint_count and bool(
        ((local_index < 0) | (local_index >= 6)).any().item()
    ):
        raise ValueError("morphology joint variable local index is invalid")
    if joint_count and bool(
        ((joint_type < 0) | (joint_type >= len(JOINT_TYPE_NAMES))).any().item()
    ):
        raise ValueError("morphology joint type is invalid")
    child_role = vector("child_role", torch.long)
    child_side = vector("child_side", torch.long)
    if joint_count and bool(
        ((child_role < 0) | (child_role >= len(BODY_ROLE_NAMES))).any().item()
    ):
        raise ValueError("morphology child body role is invalid")
    if joint_count and bool(
        ((child_side < 0) | (child_side >= len(BODY_SIDE_NAMES))).any().item()
    ):
        raise ValueError("morphology child body side is invalid")
    parent_index = vector("parent_node_index", torch.long)
    parent_valid = parent_index.ge(0)
    parent_role = vector("parent_role", torch.long)
    parent_side = vector("parent_side", torch.long)
    if bool((
        ((parent_role < 0) | (parent_role >= len(BODY_ROLE_NAMES)))
        & parent_valid
    ).any().item()):
        raise ValueError("morphology parent body role is invalid")
    if bool((
        ((parent_side < 0) | (parent_side >= len(BODY_SIDE_NAMES)))
        & parent_valid
    ).any().item()):
        raise ValueError("morphology parent body side is invalid")
    parent_valid_f = parent_valid.to(torch.float32).unsqueeze(-1)
    topology_depth = vector("topology_depth", torch.float32)
    node_count = int(robotMorphology.node_count)
    if node_count < 1:
        raise ValueError("morphology node count must be positive")

    def limit(name: str, valid_name: str) -> torch.Tensor:
        value = torch.nan_to_num(vector(name, torch.float32))
        value_valid = vector(valid_name, torch.bool).to(torch.float32)
        return torch.stack([value * value_valid, value_valid], dim=-1)

    descriptor = torch.cat([
        F.one_hot(
            joint_type,
            num_classes=len(JOINT_TYPE_NAMES)).to(torch.float32),
        torch.nan_to_num(matrix("joint_axis", 3)),
        F.one_hot(local_index, num_classes=6).to(torch.float32),
        (topology_depth / float(node_count)).unsqueeze(-1),
        F.one_hot(
            child_role,
            num_classes=len(BODY_ROLE_NAMES)).to(torch.float32),
        F.one_hot(
            child_side,
            num_classes=len(BODY_SIDE_NAMES)).to(torch.float32),
        matrix("child_capability", len(BODY_CAPABILITY_NAMES)),
        F.one_hot(
            parent_role.clamp(0, len(BODY_ROLE_NAMES) - 1),
            num_classes=len(BODY_ROLE_NAMES)).to(torch.float32)
            * parent_valid_f,
        F.one_hot(
            parent_side.clamp(0, len(BODY_SIDE_NAMES) - 1),
            num_classes=len(BODY_SIDE_NAMES)).to(torch.float32)
            * parent_valid_f,
        matrix("parent_capability", len(BODY_CAPABILITY_NAMES))
            * parent_valid_f,
        matrix("group_role_membership", len(BODY_ROLE_NAMES)),
        matrix("group_side_membership", len(BODY_SIDE_NAMES)),
        matrix("group_capability", len(BODY_CAPABILITY_NAMES)),
        commandable.to(torch.float32).unsqueeze(-1),
        limit("lower_limit_normalized", "position_lower_limit_valid"),
        limit("upper_limit_normalized", "position_upper_limit_valid"),
        limit("effort_limit_normalized", "effort_limit_valid"),
        limit("velocity_limit_normalized", "velocity_limit_valid"),
        (command_delta_scale / (1.0 + command_delta_scale)).unsqueeze(-1),
    ], dim=-1)
    if tuple(descriptor.shape) != (joint_count, descriptor_dim):
        raise RuntimeError("morphology joint descriptor width is inconsistent")
    return (
        commandable,
        descriptor,
        lower,
        upper,
        effort_limit,
        velocity_limit,
        joint_type,
        local_index,
        command_delta_scale,)


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
    endpointCount: Optional[int] = None,
    actionDim: int = ModuleDim.DecisionActionDim,
    robotMorphology: Optional[Any] = None,
) -> torch.Tensor:
    task_mask, _ = _EndpointContract(robotMorphology)
    endpoint_count = (
        task_mask.size(0) if endpointCount is None else int(endpointCount))
    action_dim = int(actionDim)
    if endpoint_count != task_mask.size(0) or action_dim != task_mask.size(1):
        raise ValueError("requested decision mask does not match morphology")
    return task_mask.to(dtype=torch.float32)


def MaskRobotPhysicalReference(
    robotPhysicalReference: torch.Tensor,) -> torch.Tensor:
    valid = robotPhysicalReference[:, -1:].clamp(0.0, 1.0)
    return torch.cat([
        robotPhysicalReference[:, :-1] * valid,
        valid], dim=-1)


def FlattenActiveDecisionTensor(
    decisionTensor: torch.Tensor,
    actionMask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    value = decisionTensor
    if actionMask is not None:
        value = value * actionMask.to(
            device=value.device, dtype=value.dtype)
    return value.flatten(start_dim=-2)


def UnflattenActiveDecisionTensor(
    activeDecision: torch.Tensor,
    endpointCount: Optional[int] = None,
    actionDim: int = ModuleDim.DecisionActionDim,
) -> torch.Tensor:
    action_dim = int(actionDim)
    endpoint_count = (
        activeDecision.size(-1) // action_dim
        if endpointCount is None
        else int(endpointCount))
    if activeDecision.size(-1) != endpoint_count * action_dim:
        raise ValueError("flat decision tensor does not match endpoint count")
    return activeDecision.reshape(
        *activeDecision.shape[:-1], endpoint_count, action_dim)


class JointStateEncoder(AGICoreModule):
    def __init__(
        self,
        jointCount: Optional[int] = None,
        embedDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        hidden: int = 256,
        jointCommandable: Optional[torch.Tensor] = None,
        jointDescriptor: Optional[torch.Tensor] = None,
        jointLower: Optional[torch.Tensor] = None,
        jointUpper: Optional[torch.Tensor] = None,
        jointEffortLimit: Optional[torch.Tensor] = None,
        jointVelocityLimit: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.joint_count = _ActualCount(jointCount, "jointCount")
        self.embed_dim = int(embedDim)
        if jointCommandable is None or jointDescriptor is None:
            raise TypeError("joint morphology tensors are required")
        static_commandable = torch.as_tensor(
            jointCommandable, dtype=torch.bool)
        descriptor = torch.as_tensor(
            jointDescriptor, dtype=torch.float32)
        if tuple(static_commandable.shape) != (self.joint_count,):
            raise ValueError("joint commandability does not match dof count")
        if descriptor.dim() != 2 or descriptor.size(0) != self.joint_count:
            raise ValueError("joint descriptor does not match dof count")
        self.register_buffer(
            "joint_commandable",
            static_commandable.view(1, -1),
            persistent=False)
        self.register_buffer(
            "joint_descriptor", descriptor.unsqueeze(0), persistent=False)
        for name, value in (
            ("joint_lower", jointLower),
            ("joint_upper", jointUpper),
            ("joint_effort_limit", jointEffortLimit),
            ("joint_velocity_limit", jointVelocityLimit),
        ):
            if value is None:
                raise TypeError(f"{name} is required from robot morphology")
            tensor = torch.as_tensor(value, dtype=torch.float32)
            if tuple(tensor.shape) != (self.joint_count,):
                raise ValueError(f"{name} does not match joint dof count")
            self.register_buffer(
                name, tensor.view(1, -1), persistent=False)
        local_dim = 4 + int(descriptor.size(1))
        self.token_net = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)

    def forward(
        self,
        jointPosition: torch.Tensor,
        jointVelocity: torch.Tensor,
        jointEffort: torch.Tensor,
        jointObserved: torch.Tensor,
        jointHealthy: torch.Tensor,
        jointControllable: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        values = (
            jointPosition,
            jointVelocity,
            jointEffort,
            jointObserved,
            jointHealthy,
            jointControllable,)
        if any(
            value.dim() != 2 or value.size(1) != self.joint_count
            for value in values
        ):
            raise ValueError("joint runtime state does not match dof count")
        batch_size = jointPosition.size(0)
        if any(value.size(0) != batch_size for value in values):
            raise ValueError("joint runtime state batch sizes do not match")
        dtype = self.token_net[1].weight.dtype
        device = jointPosition.device
        position = torch.nan_to_num(jointPosition.to(dtype=dtype))
        velocity = torch.nan_to_num(
            jointVelocity.to(device=device, dtype=dtype))
        effort = torch.nan_to_num(
            jointEffort.to(device=device, dtype=dtype))
        observed = jointObserved.to(device=device, dtype=torch.bool)
        healthy = jointHealthy.to(device=device, dtype=torch.bool)
        runtime_controllable = jointControllable.to(
            device=device, dtype=torch.bool)
        state_valid = observed & healthy
        controllable = (
            state_valid
            & runtime_controllable
            & self.joint_commandable.to(device=device))
        lower = self.joint_lower.to(device=device, dtype=dtype)
        upper = self.joint_upper.to(device=device, dtype=dtype)
        finite_bounds = (
            torch.isfinite(lower)
            & torch.isfinite(upper)
            & (upper > lower))
        safe_lower = torch.where(
            finite_bounds, lower, torch.zeros_like(lower))
        safe_upper = torch.where(
            finite_bounds, upper, torch.zeros_like(upper))
        center = 0.5 * (safe_lower + safe_upper)
        half_range = 0.5 * (safe_upper - safe_lower).clamp_min(1e-6)
        position_feature = torch.where(
            finite_bounds,
            torch.tanh((position - center) / half_range),
            torch.tanh(position))

        def scale_feature(
            value: torch.Tensor,
            limit: torch.Tensor,
        ) -> torch.Tensor:
            finite_limit = torch.isfinite(limit) & (limit > 0.0)
            safe_limit = torch.where(
                finite_limit, limit, torch.ones_like(limit))
            return torch.where(
                finite_limit,
                torch.tanh(value / safe_limit),
                torch.tanh(value))

        velocity_feature = scale_feature(
            velocity,
            self.joint_velocity_limit.to(device=device, dtype=dtype))
        effort_feature = scale_feature(
            effort,
            self.joint_effort_limit.to(device=device, dtype=dtype))
        state_mask = state_valid.to(dtype=dtype).unsqueeze(-1)
        descriptor = self.joint_descriptor.to(
            device=device, dtype=dtype).expand(batch_size, -1, -1)
        local_state = torch.cat([
            position_feature.unsqueeze(-1),
            velocity_feature.unsqueeze(-1),
            effort_feature.unsqueeze(-1),
            controllable.to(dtype=dtype).unsqueeze(-1),
            descriptor,], dim=-1)
        tokens = self.token_net(local_state) * state_mask
        count = state_mask.sum(dim=1).clamp_min(1.0)
        mean = tokens.sum(dim=1) / count
        variance = (
            (tokens - mean.unsqueeze(1)).square()
            * state_mask).sum(dim=1) / count
        joint_feature = self.summary_net(torch.cat([
            mean,
            variance,], dim=-1))
        joint_feature = joint_feature * state_valid.any(
            dim=1, keepdim=True).to(dtype=dtype)
        return tokens, joint_feature, state_valid, controllable


class EndpointPoseEncoder(AGICoreModule):
    def __init__(
        self,
        endpointCount: Optional[int] = None,
        poseDim: int = ModuleDim.DecisionEndpointPoseDim,
        embedDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        hidden: int = 256,
        endpointDescriptor: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.endpoint_count = _ActualCount(endpointCount, "endpointCount")
        self.pose_dim = int(poseDim)
        self.embed_dim = int(embedDim)
        if endpointDescriptor is None:
            raise TypeError("endpoint descriptor is required")
        descriptor = torch.as_tensor(
            endpointDescriptor, dtype=torch.float32)
        if descriptor.dim() != 2 or descriptor.size(0) != self.endpoint_count:
            raise ValueError("endpoint descriptor does not match count")
        self.register_buffer(
            "endpoint_descriptor", descriptor.unsqueeze(0), persistent=False)
        self.token_net = nn.Sequential(
            nn.LayerNorm(self.pose_dim),
            nn.Linear(self.pose_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.descriptor_net = nn.Sequential(
            nn.LayerNorm(int(descriptor.size(1))),
            nn.Linear(int(descriptor.size(1)), hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def forward(
        self,
        endpointPoseRelative: torch.Tensor,
        endpointStateValid: torch.Tensor,
    ) -> EndpointPoseEncoding:
        if endpointPoseRelative.dim() != 3 or tuple(
            endpointPoseRelative.shape[1:]
        ) != (self.endpoint_count, self.pose_dim):
            raise ValueError("endpoint pose does not match endpoint count")
        batch_size = endpointPoseRelative.size(0)
        expected_mask_shape = (batch_size, self.endpoint_count)
        if tuple(endpointStateValid.shape) != expected_mask_shape:
            raise ValueError("endpoint state validity does not match count")
        if endpointStateValid.dtype != torch.bool:
            raise TypeError("endpoint state validity must be boolean")
        if endpointStateValid.device != endpointPoseRelative.device:
            raise ValueError("endpoint state validity device does not match pose")
        state_valid = endpointStateValid
        safe_pose = torch.where(
            state_valid.unsqueeze(-1),
            torch.nan_to_num(endpointPoseRelative),
            torch.zeros_like(endpointPoseRelative))
        endpoint_pose_relative = NormalizePose(safe_pose)
        tokens = (
            self.token_net(endpoint_pose_relative)
            + self.descriptor_net(self.endpoint_descriptor))
        valid = state_valid.to(dtype=tokens.dtype).unsqueeze(-1)
        tokens = tokens * valid
        endpoint_feat = (
            (self.summary_net(tokens) * valid).sum(dim=1)
            / valid.sum(dim=1).clamp_min(1.0))
        endpoint_feat = endpoint_feat * state_valid.any(
            dim=1, keepdim=True).to(dtype=endpoint_feat.dtype)
        return EndpointPoseEncoding(
            endpoint_pose_tokens=tokens,
            endpoint_pose_feat=endpoint_feat)


class EndpointControlEncoder(AGICoreModule):
    def __init__(
        self,
        endpointCount: Optional[int] = None,
        actionDim: int = ModuleDim.DecisionActionDim,
        embedDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        hidden: int = 256,
        actionMask: Optional[torch.Tensor] = None,
        endpointDescriptor: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.endpoint_count = _ActualCount(endpointCount, "endpointCount")
        self.action_dim = int(actionDim)
        self.embed_dim = int(embedDim)
        endpoint_control_dim = self.action_dim * 2
        self.token_net = nn.Sequential(
            nn.LayerNorm(endpoint_control_dim),
            nn.Linear(endpoint_control_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)
        if endpointDescriptor is None or actionMask is None:
            raise TypeError("endpoint control morphology tensors are required")
        descriptor = torch.as_tensor(
            endpointDescriptor, dtype=torch.float32)
        if descriptor.dim() != 2 or descriptor.size(0) != self.endpoint_count:
            raise ValueError("endpoint descriptor does not match count")
        self.register_buffer(
            "endpoint_descriptor", descriptor.unsqueeze(0), persistent=False)
        self.descriptor_net = nn.Sequential(
            nn.LayerNorm(int(descriptor.size(1))),
            nn.Linear(int(descriptor.size(1)), hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(2 * self.embed_dim),
            nn.Linear(2 * self.embed_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.embed_dim),
            nn.LayerNorm(self.embed_dim),)
        action_mask = torch.as_tensor(actionMask, dtype=torch.bool)
        if tuple(action_mask.shape) != (
            self.endpoint_count, self.action_dim
        ):
            raise ValueError("endpoint control mask does not match count")
        self.register_buffer(
            "action_mask",
            action_mask.view(1, self.endpoint_count, self.action_dim),
            persistent=False)

    def forward(
        self,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,
        endpointStateValid: torch.Tensor,
        endpointControllable: torch.Tensor,
    ) -> EndpointControlEncoding:
        expected_shape = (
            targetTrackingError.size(0),
            self.endpoint_count,
            self.action_dim)
        if tuple(targetTrackingError.shape) != expected_shape:
            raise ValueError("target tracking error does not match endpoint count")
        if tuple(plannerTrackingError.shape) != expected_shape:
            raise ValueError("planner tracking error does not match endpoint count")
        batch_size = targetTrackingError.size(0)
        expected_mask_shape = (batch_size, self.endpoint_count)
        if tuple(endpointStateValid.shape) != expected_mask_shape:
            raise ValueError("endpoint state validity does not match count")
        if tuple(endpointControllable.shape) != expected_mask_shape:
            raise ValueError("endpoint controllability does not match count")
        if endpointStateValid.dtype != torch.bool:
            raise TypeError("endpoint state validity must be boolean")
        if endpointControllable.dtype != torch.bool:
            raise TypeError("endpoint controllability must be boolean")
        if endpointStateValid.device != targetTrackingError.device:
            raise ValueError("endpoint state validity device does not match errors")
        if endpointControllable.device != targetTrackingError.device:
            raise ValueError("endpoint controllability device does not match errors")
        state_valid = endpointStateValid
        controllable = endpointControllable
        endpoint_valid = state_valid & controllable
        action_mask = (
            self.action_mask.to(device=targetTrackingError.device)
            & endpoint_valid.unsqueeze(-1))
        mask = action_mask.to(dtype=targetTrackingError.dtype)
        target_error = torch.where(
            action_mask,
            torch.nan_to_num(targetTrackingError),
            torch.zeros_like(targetTrackingError)) * mask
        planner_error = torch.where(
            action_mask,
            torch.nan_to_num(plannerTrackingError),
            torch.zeros_like(plannerTrackingError)) * mask
        endpoint_control = torch.cat([
            target_error,
            planner_error], dim=-1)
        endpoint_valid_f = endpoint_valid.to(
            dtype=targetTrackingError.dtype).unsqueeze(-1)
        control_tokens = (
            self.token_net(endpoint_control)
            + self.descriptor_net(self.endpoint_descriptor)) * endpoint_valid_f
        count = endpoint_valid_f.sum(dim=1).clamp_min(1.0)
        mean = control_tokens.sum(dim=1) / count
        variance = (
            (control_tokens - mean.unsqueeze(1)).square()
            * endpoint_valid_f).sum(dim=1) / count
        control_feedback_feat = self.summary_net(torch.cat([
            mean,
            variance,], dim=-1))
        control_feedback_feat = control_feedback_feat * endpoint_valid.any(
            dim=1, keepdim=True).to(dtype=control_feedback_feat.dtype)
        return EndpointControlEncoding(
            endpoint_control_tokens=(
                control_tokens),
            control_feedback_feat=control_feedback_feat)


class ActionConstraintProjector(AGICoreModule):
    def __init__(
        self,
        endpointCount: Optional[int] = None,
        actionDim: int = ModuleDim.DecisionActionDim,
        translationLimit: float = 0.05,
        rotationLimit: float = 0.25,
        actionMask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        endpoint_count = _ActualCount(endpointCount, "endpointCount")
        if actionMask is None:
            raise TypeError("endpoint action mask is required")
        action_limit = torch.empty(endpoint_count, int(actionDim))
        action_limit[:, :3] = float(translationLimit)
        action_limit[:, 3:] = float(rotationLimit)
        self.register_buffer(
            "action_limit",
            action_limit.view(1, endpoint_count, int(actionDim)),
            persistent=True,)
        action_mask = torch.as_tensor(actionMask, dtype=torch.float32)
        if tuple(action_mask.shape) != (
            endpoint_count, int(actionDim)
        ):
            raise ValueError("action constraint mask does not match endpoint count")
        self.register_buffer(
            "action_mask",
            action_mask.view(
                1, endpoint_count, int(actionDim)),
            persistent=False,)

    def forward(self, decisionLatent: torch.Tensor) -> torch.Tensor:
        decisionLatent = decisionLatent * self.action_mask
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
        endpointCount: Optional[int] = None,
        actionDim: int = ModuleDim.DecisionActionDim,
        actionMask: Optional[torch.Tensor] = None,
        endpointDescriptor: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.endpoint_count = _ActualCount(endpointCount, "endpointCount")
        self.action_dim = int(actionDim)
        self.out_dim = int(outDim)
        if endpointDescriptor is None or actionMask is None:
            raise TypeError("endpoint action morphology tensors are required")
        descriptor = torch.as_tensor(
            endpointDescriptor, dtype=torch.float32)
        if descriptor.dim() != 2 or descriptor.size(0) != self.endpoint_count:
            raise ValueError("endpoint descriptor does not match count")
        mask = torch.as_tensor(actionMask, dtype=torch.bool)
        if tuple(mask.shape) != (
            self.endpoint_count, self.action_dim
        ):
            raise ValueError("endpoint action mask does not match count")
        self.register_buffer(
            "action_mask",
            mask.view(1, self.endpoint_count, self.action_dim),
            persistent=False)
        self.register_buffer(
            "endpoint_descriptor", descriptor.unsqueeze(0), persistent=False)
        local_dim = self.action_dim + int(descriptor.size(1))
        self.token_net = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),)
        self.net = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

    def forward(
        self,
        decisionTensor: torch.Tensor,) -> torch.Tensor:
        mask = self.action_mask.to(
            device=decisionTensor.device, dtype=decisionTensor.dtype)
        valid = mask.any(dim=-1, keepdim=True).to(decisionTensor.dtype)
        descriptor = self.endpoint_descriptor.to(
            device=decisionTensor.device,
            dtype=decisionTensor.dtype).expand(decisionTensor.size(0), -1, -1)
        token = self.token_net(torch.cat([
            decisionTensor * mask,
            descriptor,], dim=-1)) * valid
        count = valid.sum(dim=1).clamp_min(1.0)
        mean = token.sum(dim=1) / count
        variance = (
            (token - mean.unsqueeze(1)).square() * valid).sum(dim=1) / count
        encoded = self.net(torch.cat([mean, variance], dim=-1))
        return encoded * self.action_mask.any().to(
            device=encoded.device, dtype=encoded.dtype)


class JointVariableActionEncoder(AGICoreModule):
    def __init__(
        self,
        jointCount: Optional[int] = None,
        outDim: int = ModuleDim.EndpointActionEmbedDim,
        hidden: int = 256,
        jointCommandable: Optional[torch.Tensor] = None,
        jointDescriptor: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.joint_count = _ActualCount(jointCount, "jointCount")
        self.out_dim = int(outDim)
        if jointCommandable is None or jointDescriptor is None:
            raise TypeError("joint action morphology tensors are required")
        commandable = torch.as_tensor(
            jointCommandable, dtype=torch.bool)
        descriptor = torch.as_tensor(
            jointDescriptor, dtype=torch.float32)
        if tuple(commandable.shape) != (self.joint_count,):
            raise ValueError("joint action commandability does not match dof count")
        if descriptor.dim() != 2 or descriptor.size(0) != self.joint_count:
            raise ValueError("joint action descriptor does not match dof count")
        self.register_buffer(
            "joint_commandable",
            commandable.view(1, self.joint_count),
            persistent=False)
        self.register_buffer(
            "joint_descriptor",
            descriptor.unsqueeze(0),
            persistent=False)
        local_dim = 1 + int(descriptor.size(1))
        self.token_net = nn.Sequential(
            nn.LayerNorm(local_dim),
            nn.Linear(local_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),)
        self.summary_net = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.out_dim),)

    def _Encode(
        self,
        jointVariableCommand: torch.Tensor,
        jointVariableMask: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(jointVariableCommand.shape) != tuple(jointVariableMask.shape):
            raise ValueError("joint action command and mask shapes do not match")
        if jointVariableCommand.dim() != 2 or (
            jointVariableCommand.size(1) != self.joint_count
        ):
            raise ValueError("joint action does not match dof count")
        command_mask = jointVariableMask.to(
            device=jointVariableCommand.device, dtype=torch.bool)
        dtype = self.token_net[1].weight.dtype
        command = torch.nan_to_num(
            jointVariableCommand.to(dtype=dtype)).clamp(-1.0, 1.0)
        descriptor = self.joint_descriptor.to(
            device=command.device,
            dtype=dtype).expand(command.size(0), -1, -1)
        token_input = torch.cat([command.unsqueeze(-1), descriptor], dim=-1)
        zero_input = torch.cat([
            torch.zeros_like(command).unsqueeze(-1),
            descriptor,], dim=-1)
        mask = command_mask.to(dtype=dtype).unsqueeze(-1)
        token = (self.token_net(token_input) - self.token_net(zero_input)) * mask
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = token.sum(dim=1) / count
        variance = (
            (token - mean.unsqueeze(1)).square() * mask).sum(dim=1) / count
        summary = torch.cat([mean, variance], dim=-1)
        zero_summary = torch.zeros_like(summary)
        return self.summary_net(summary) - self.summary_net(zero_summary)

    def forward(
        self,
        jointVariableCommand: torch.Tensor,
        jointVariableMask: torch.Tensor,
    ) -> torch.Tensor:
        commandable_mask = (
            jointVariableMask.to(
                device=jointVariableCommand.device, dtype=torch.bool)
            & self.joint_commandable.to(
                device=jointVariableCommand.device))
        return self._Encode(jointVariableCommand, commandable_mask)

    def EncodeMeasured(
        self,
        jointVariableDelta: torch.Tensor,
        jointMeasurementMask: torch.Tensor,
    ) -> torch.Tensor:
        return self._Encode(jointVariableDelta, jointMeasurementMask)


class DecisionFeedbackEncoder(AGICoreModule):
    """Closed-loop decision feedback kept separate from the World efference copy."""

    def __init__(
        self,
        stateFeatureDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        physicalReferenceDim: int = ModuleDim.RobotPhysicalReferenceDim,
        outDim: int = ModuleDim.EndpointActionEmbedDim,
        hidden: int = 512,
        actionMask: Optional[torch.Tensor] = None,
        jointCommandable: Optional[torch.Tensor] = None,):
        super().__init__()
        if actionMask is None or jointCommandable is None:
            raise TypeError("decision feedback morphology tensors are required")
        in_dim = (
            int(outDim)
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
        action_available = bool(
            torch.as_tensor(
                actionMask, dtype=torch.bool).any().item()
            or torch.as_tensor(
                jointCommandable, dtype=torch.bool).any().item())
        self.register_buffer(
            "action_available",
            torch.tensor(action_available, dtype=torch.bool),
            persistent=False)

    def forward(
        self,
        endpointActionFeature: torch.Tensor,
        robotStateEncoding: DecisionRobotStateEncoding,
        robotPhysicalReference: torch.Tensor,) -> torch.Tensor:
        physical_reference = MaskRobotPhysicalReference(
            robotPhysicalReference)
        encoded = self.net(torch.cat([
            endpointActionFeature,
            robotStateEncoding.endpoint_pose.endpoint_pose_feat,
            robotStateEncoding.endpoint_control.control_feedback_feat,
            physical_reference], dim=-1))
        endpoint_evidence = (
            robotStateEncoding.endpoint_state_valid.any(dim=1)
            & robotStateEncoding.endpoint_controllable.any(dim=1))
        joint_evidence = (
            robotStateEncoding.joint_state_valid.any(dim=1)
            & robotStateEncoding.joint_controllable.any(dim=1))
        evidence_valid = self.action_available.expand(
            encoded.size(0)) & (endpoint_evidence | joint_evidence)
        return encoded * evidence_valid.to(
            device=encoded.device,
            dtype=encoded.dtype).unsqueeze(-1)


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
        endpointCount: Optional[int] = None,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 512,
        localDim: int = ModuleDim.DecisionEndpointPoseFeatDim * 2,
    ):
        super().__init__()
        self.endpoint_count = _ActualCount(endpointCount, "endpointCount")
        self.action_dim = int(actionDim)
        self.context_net = nn.Sequential(
            nn.LayerNorm(inputDim),
            nn.Linear(inputDim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden + int(localDim)),
            nn.Linear(hidden + int(localDim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.action_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        endpointContext: torch.Tensor,
    ) -> torch.Tensor:
        if endpointContext.size(1) != self.endpoint_count:
            raise ValueError("endpoint context does not match count")
        context = self.context_net(x).unsqueeze(1).expand(
            -1, self.endpoint_count, -1)
        return self.net(torch.cat([context, endpointContext], dim=-1))


class JointVariableCommandHead(AGICoreModule):
    def __init__(
        self,
        contextDim: int,
        tokenDim: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.token_dim = int(tokenDim)
        self.context_net = nn.Sequential(
            nn.LayerNorm(int(contextDim)),
            nn.Linear(int(contextDim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),)
        self.command_net = nn.Sequential(
            nn.LayerNorm(hidden + self.token_dim),
            nn.Linear(hidden + self.token_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),)
        nn.init.zeros_(self.command_net[-1].weight)
        nn.init.zeros_(self.command_net[-1].bias)

    def forward(
        self,
        context: torch.Tensor,
        jointTokens: torch.Tensor,
        commandMask: torch.Tensor,
    ) -> torch.Tensor:
        if jointTokens.dim() != 3 or jointTokens.size(-1) != self.token_dim:
            raise ValueError("joint command tokens have an invalid shape")
        if tuple(commandMask.shape) != tuple(jointTokens.shape[:2]):
            raise ValueError("joint command mask does not match joint variables")
        if context.size(0) != jointTokens.size(0):
            raise ValueError("joint command context batch does not match")
        shared_context = self.context_net(context).unsqueeze(1).expand(
            -1, jointTokens.size(1), -1)
        command = torch.tanh(self.command_net(torch.cat([
            shared_context,
            jointTokens,], dim=-1)).squeeze(-1))
        return command * commandMask.to(
            device=command.device, dtype=command.dtype)


class ChunkDynamicsHead(SE3ActionHead):
    def __init__(
        self,
        inputDim: int,
        endpointCount: Optional[int] = None,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
        localDim: int = ModuleDim.DecisionEndpointPoseFeatDim * 2,
    ):
        super().__init__(
            inputDim=inputDim,
            endpointCount=endpointCount,
            actionDim=actionDim,
            hidden=hidden,
            localDim=localDim,
        )


class ResidualErrorCompensator(SE3ActionHead):
    def __init__(
        self,
        inputDim: int,
        endpointCount: Optional[int] = None,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
        localDim: int = ModuleDim.DecisionEndpointPoseFeatDim * 2,
    ):
        super().__init__(
            inputDim=inputDim,
            endpointCount=endpointCount,
            actionDim=actionDim,
            hidden=hidden,
            localDim=localDim,
        )


class EndpointActionRefiner(AGICoreModule):
    def __init__(
        self,
        endpointCount: Optional[int] = None,
        endpointPoseTokenDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        endpointControlTokenDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        subgoalDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        motionDim: int = 256,
        dynamicsDim: int = 128,
        uncertaintyDim: int = 64,
        actionDim: int = ModuleDim.DecisionActionDim,
        hidden: int = 256,
        actionMask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.endpoint_count = _ActualCount(endpointCount, "endpointCount")
        self.action_dim = int(actionDim)
        if actionMask is None:
            raise TypeError("endpoint refiner action mask is required")
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
        action_mask = torch.as_tensor(actionMask, dtype=torch.bool)
        if tuple(action_mask.shape) != (
            self.endpoint_count, self.action_dim
        ):
            raise ValueError("endpoint refiner mask does not match count")
        self.register_buffer(
            "action_mask",
            action_mask.view(1, self.endpoint_count, self.action_dim),
            persistent=False)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

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
        action = self.net(local_features)
        return action * self.action_mask.to(
            device=action.device, dtype=action.dtype)


class DiscreteCommandContract(AGICoreModule):
    def __init__(
        self,
        gripperCount: Optional[int] = None,
        modeDim: int = ModuleDim.ActTypeDim,
    ):
        super().__init__()
        self.gripper_count = _ActualCount(gripperCount, "gripperCount")
        self.mode_dim = int(modeDim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "mode_logits": x.new_zeros(x.size(0), self.mode_dim),
            "gripper_cmd": x.new_full((x.size(0), self.gripper_count, 1), 0.5),
            "mode_valid": torch.zeros(x.size(0), device=x.device, dtype=torch.bool),
            "gripper_valid": torch.ones(
                x.size(0),
                self.gripper_count,
                device=x.device,
                dtype=torch.bool),
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
        robotMorphology: Optional[Any] = None,
    ):
        super().__init__()
        if robotMorphology is None:
            raise TypeError("robot morphology is required")
        endpoint_task_mask, endpoint_names = _EndpointContract(robotMorphology)
        endpoint_descriptor = _EndpointSemanticDescriptor(robotMorphology)
        (
            joint_commandable,
            joint_descriptor,
            joint_lower,
            joint_upper,
            joint_effort_limit,
            joint_velocity_limit,
            joint_type,
            joint_local_index,
            joint_command_delta_scale,
        ) = _JointSemanticContract(robotMorphology)
        self.decision_dim = int(decisionDim)
        self.plan_dim = int(planDim)
        self.subgoal_feature_dim = int(subgoalFeatureDim)
        self.constraint_token_dim = int(constraintTokenDim)
        self.pose_feature_dim = int(poseFeatureDim)
        self.physical_reference_dim = int(physicalReferenceDim)
        self.endpoint_action_embed_dim = int(endpointActionEmbedDim)
        self.endpoint_count = int(endpoint_task_mask.size(0))
        self.joint_count = int(joint_commandable.numel())
        self.gripper_count = int(robotMorphology.gripper_count)
        self.pose_dim = ModuleDim.DecisionEndpointPoseDim
        self.action_dim = ModuleDim.RobotControlAxisDim
        self.endpoint_names = endpoint_names
        self.joint_variable_names = tuple(
            str(name) for name in robotMorphology.joint_variable_names)
        if len(self.joint_variable_names) != self.joint_count:
            raise ValueError("joint variable names do not match dof count")
        self.register_buffer(
            "endpoint_task_mask",
            endpoint_task_mask.view(
                1, self.endpoint_count, self.action_dim),
            persistent=False)
        self.register_buffer(
            "joint_variable_commandable",
            joint_commandable.view(1, self.joint_count),
            persistent=False)
        self.register_buffer(
            "joint_type",
            joint_type.view(1, self.joint_count),
            persistent=False)
        self.register_buffer(
            "joint_local_index",
            joint_local_index.view(1, self.joint_count),
            persistent=False)
        self.register_buffer(
            "joint_command_delta_scale",
            joint_command_delta_scale.view(1, self.joint_count),
            persistent=False)
        self.endpoint_pose_encoder = EndpointPoseEncoder(
            endpointCount=self.endpoint_count,
            poseDim=self.pose_dim,
            embedDim=self.pose_feature_dim,
            endpointDescriptor=endpoint_descriptor,
        )
        self.endpoint_control_encoder = EndpointControlEncoder(
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
            embedDim=self.pose_feature_dim,
            actionMask=endpoint_task_mask,
            endpointDescriptor=endpoint_descriptor,
        )
        self.joint_state_encoder = JointStateEncoder(
            jointCount=self.joint_count,
            embedDim=self.pose_feature_dim,
            jointCommandable=joint_commandable,
            jointDescriptor=joint_descriptor,
            jointLower=joint_lower,
            jointUpper=joint_upper,
            jointEffortLimit=joint_effort_limit,
            jointVelocityLimit=joint_velocity_limit,)
        self.action_projector = ActionConstraintProjector(
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
            translationLimit=translationLimit,
            rotationLimit=rotationLimit,
            actionMask=endpoint_task_mask,)
        self.endpoint_action_encoder = EndpointActionEncoder(
            outDim=self.endpoint_action_embed_dim,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
            actionMask=endpoint_task_mask,
            endpointDescriptor=endpoint_descriptor,
        )
        self.joint_variable_action_encoder = JointVariableActionEncoder(
            jointCount=self.joint_count,
            outDim=self.endpoint_action_embed_dim,
            jointCommandable=joint_commandable,
            jointDescriptor=joint_descriptor,)
        self.decision_feedback_encoder = DecisionFeedbackEncoder(
            stateFeatureDim=self.pose_feature_dim,
            physicalReferenceDim=self.physical_reference_dim,
            outDim=self.endpoint_action_embed_dim,
            actionMask=endpoint_task_mask,
            jointCommandable=joint_commandable,)
        factor_input = (
            self.decision_dim
            + self.plan_dim
            + self.subgoal_feature_dim
            + self.pose_feature_dim
            + self.physical_reference_dim
            + self.pose_feature_dim)
        self.factor_projector = LatentFactorProjector(factor_input)
        self.joint_factor_residual = nn.Sequential(
            nn.LayerNorm(self.pose_feature_dim),
            nn.Linear(self.pose_feature_dim, 256),
            nn.SiLU(),
            nn.Linear(256, self.factor_projector.total_dim),)
        nn.init.zeros_(self.joint_factor_residual[-1].weight)
        nn.init.zeros_(self.joint_factor_residual[-1].bias)
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
        joint_action_input = (
            z_total
            + self.plan_dim
            + self.subgoal_feature_dim
            + self.physical_reference_dim
            + cross_dim)

        self.action_head = SE3ActionHead(
            action_input,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
        )
        self.joint_variable_command_head = JointVariableCommandHead(
            contextDim=joint_action_input,
            tokenDim=self.pose_feature_dim,)
        self.dynamics_head = ChunkDynamicsHead(
            dyn_input,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
        )
        self.discrete_command_contract = DiscreteCommandContract(
            gripperCount=self.gripper_count)
        self.residual_compensator = ResidualErrorCompensator(
            residual_input,
            endpointCount=self.endpoint_count,
            actionDim=self.action_dim,
        )
        self.endpoint_action_refiner = EndpointActionRefiner(
            endpointCount=self.endpoint_count,
            endpointPoseTokenDim=self.pose_feature_dim,
            endpointControlTokenDim=self.pose_feature_dim,
            subgoalDim=self.subgoal_feature_dim,
            motionDim=self.factor_projector.motion_dim,
            dynamicsDim=self.factor_projector.dyn_dim,
            uncertaintyDim=self.factor_projector.uncertainty_dim,
            actionDim=self.action_dim,
            actionMask=endpoint_task_mask,)
        self.internal_decision_feature_encoder = nn.Sequential(
            nn.LayerNorm(self.decision_dim),
            nn.Linear(
                self.decision_dim,
                self.endpoint_action_embed_dim),
            nn.SiLU(),
            nn.Linear(
                self.endpoint_action_embed_dim,
                self.endpoint_action_embed_dim),
            nn.LayerNorm(self.endpoint_action_embed_dim),)

    def ProjectDecisionLatent(self, decisionLatent: torch.Tensor) -> torch.Tensor:
        return self.action_projector(decisionLatent)

    def MaskDecisionTensor(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return self.action_projector.Mask(decisionTensor)

    def NormalizeDecisionTensor(self, decisionTensor: torch.Tensor) -> torch.Tensor:
        return self.action_projector.Normalize(decisionTensor)

    def MaskJointVariableCommand(
        self,
        jointVariableCommand: torch.Tensor,
        runtimeMask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (*jointVariableCommand.shape[:-1], self.joint_count)
        if tuple(jointVariableCommand.shape) != expected:
            raise ValueError("joint variable command does not match dof count")
        static_mask = self.joint_variable_commandable.to(
            device=jointVariableCommand.device).expand_as(jointVariableCommand)
        if tuple(runtimeMask.shape) != tuple(jointVariableCommand.shape):
            raise ValueError("joint variable runtime mask shape is invalid")
        if runtimeMask.dtype != torch.bool:
            raise TypeError("joint variable runtime mask must be boolean")
        if runtimeMask.device != jointVariableCommand.device:
            raise ValueError("joint variable runtime mask device does not match command")
        static_mask = static_mask & runtimeMask
        return torch.nan_to_num(jointVariableCommand).clamp(-1.0, 1.0) * (
            static_mask.to(dtype=jointVariableCommand.dtype))

    def NormalizeMeasuredJointDelta(
        self,
        previousPosition: torch.Tensor,
        currentPosition: torch.Tensor,
        measurementMask: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = (previousPosition.size(0), self.joint_count)
        if tuple(previousPosition.shape) != expected_shape:
            raise ValueError("previous joint position does not match dof count")
        if tuple(currentPosition.shape) != expected_shape:
            raise ValueError("current joint position does not match dof count")
        if tuple(measurementMask.shape) != expected_shape:
            raise ValueError("joint delta mask does not match dof count")
        device = currentPosition.device
        dtype = self.joint_command_delta_scale.dtype
        previous = previousPosition.to(device=device, dtype=dtype)
        current = currentPosition.to(device=device, dtype=dtype)
        finite_state = torch.isfinite(previous) & torch.isfinite(current)
        delta = torch.nan_to_num(current - previous)
        joint_type = self.joint_type.to(device=device)
        local_index = self.joint_local_index.to(device=device)
        continuous = joint_type.eq(JOINT_TYPE_NAMES.index("continuous"))
        planar_rotation = (
            joint_type.eq(JOINT_TYPE_NAMES.index("planar"))
            & local_index.eq(2))
        floating_rotation = (
            joint_type.eq(JOINT_TYPE_NAMES.index("floating"))
            & local_index.ge(3))
        wrap_rotation = continuous | planar_rotation | floating_rotation
        wrapped_delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        delta = torch.where(wrap_rotation, wrapped_delta, delta)
        scale = self.joint_command_delta_scale.to(
            device=device, dtype=dtype)
        normalized = (delta / scale).clamp(-1.0, 1.0)
        mask = (
            measurementMask.to(device=device, dtype=torch.bool)
            & finite_state)
        return normalized * mask.to(dtype=normalized.dtype)

    def EncodeRobotState(
        self,
        endpointPoseRelative: torch.Tensor,
        targetTrackingError: torch.Tensor,
        plannerTrackingError: torch.Tensor,
        jointPosition: torch.Tensor,
        jointVelocity: torch.Tensor,
        jointEffort: torch.Tensor,
        jointObserved: torch.Tensor,
        jointHealthy: torch.Tensor,
        jointControllable: torch.Tensor,
        endpointStateValid: torch.Tensor,
        endpointControllable: torch.Tensor,
    ) -> DecisionRobotStateEncoding:
        if endpointPoseRelative.dim() != 3:
            raise ValueError("endpoint pose must include a batch dimension")
        batch_size = endpointPoseRelative.size(0)
        expected_endpoint_shape = (batch_size, self.endpoint_count)
        if tuple(endpointStateValid.shape) != expected_endpoint_shape:
            raise ValueError("endpoint state validity does not match count")
        if tuple(endpointControllable.shape) != expected_endpoint_shape:
            raise ValueError("endpoint controllability does not match count")
        endpoint_state_valid = endpointStateValid.to(
            device=endpointPoseRelative.device, dtype=torch.bool)
        static_controllable = self.endpoint_task_mask.any(dim=-1).to(
            device=endpointPoseRelative.device).expand(batch_size, -1)
        endpoint_controllable = (
            endpointControllable.to(
                device=endpointPoseRelative.device, dtype=torch.bool)
                & static_controllable
                & endpoint_state_valid)
        (
            joint_tokens,
            joint_feature,
            joint_state_valid,
            joint_controllable,
        ) = self.joint_state_encoder(
            jointPosition,
            jointVelocity,
            jointEffort,
            jointObserved,
            jointHealthy,
            jointControllable,)
        return DecisionRobotStateEncoding(
            endpoint_pose=self.endpoint_pose_encoder(
                endpointPoseRelative,
                endpoint_state_valid),
            endpoint_control=self.endpoint_control_encoder(
                targetTrackingError,
                plannerTrackingError,
                endpoint_state_valid,
                endpoint_controllable),
            joint_feature=joint_feature,
            joint_tokens=joint_tokens,
            joint_state_valid=joint_state_valid,
            joint_controllable=joint_controllable,
            endpoint_state_valid=endpoint_state_valid,
            endpoint_controllable=endpoint_controllable,)

    def EncodeEndpointAction(
        self,
        decisionTensor: torch.Tensor,) -> torch.Tensor:
        return self.endpoint_action_encoder(
            self.MaskDecisionTensor(decisionTensor))

    def EncodeInternalDecisionFeature(
        self,
        decisionFeature: torch.Tensor,
    ) -> torch.Tensor:
        if decisionFeature.dim() != 2 or int(
            decisionFeature.size(1)
        ) != self.decision_dim:
            raise ValueError("internal decision feature shape is invalid")
        return self.internal_decision_feature_encoder(decisionFeature)

    def EncodeAction(
        self,
        decisionTensor: torch.Tensor,
        jointVariableCommand: torch.Tensor,
        jointVariableMask: torch.Tensor,
    ) -> torch.Tensor:
        if decisionTensor.size(0) != jointVariableCommand.size(0):
            raise ValueError("endpoint and joint action batches do not match")
        joint_command = self.MaskJointVariableCommand(
            jointVariableCommand,
            jointVariableMask)
        return (
            self.EncodeEndpointAction(decisionTensor)
            + self.joint_variable_action_encoder(
                joint_command,
                jointVariableMask))

    def EncodeMeasuredAction(
        self,
        decisionTensor: torch.Tensor,
        jointVariableDelta: torch.Tensor,
        jointMeasurementMask: torch.Tensor,
    ) -> torch.Tensor:
        if decisionTensor.size(0) != jointVariableDelta.size(0):
            raise ValueError("endpoint and measured joint action batches do not match")
        return (
            self.EncodeEndpointAction(decisionTensor)
            + self.joint_variable_action_encoder.EncodeMeasured(
                jointVariableDelta,
                jointMeasurementMask))

    def EncodeDecisionFeedback(
        self,
        decisionTensor: torch.Tensor,
        robotStateEncoding: DecisionRobotStateEncoding,
        robotPhysicalReference: torch.Tensor,
        jointVariableCommand: torch.Tensor,
        jointVariableMask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decision_feedback_encoder(
            self.EncodeAction(
                decisionTensor,
                jointVariableCommand,
                jointVariableMask),
            robotStateEncoding,
            robotPhysicalReference)

    def CameraMotionFromDecisionTensor(
        self,
        decisionTensor: torch.Tensor,) -> torch.Tensor:
        identity = decisionTensor.new_zeros(
            *decisionTensor.shape[:-2], ModuleDim.ObserverMotionDim)
        identity[..., -1] = 1.0
        return identity

    def SafetyScores(
        self,
        decisionTensor: torch.Tensor,
        risk: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,) -> torch.Tensor:
        normalized = self.NormalizeDecisionTensor(decisionTensor).abs()
        if normalized.size(-2):
            translation_margin = (
                1.0
                - normalized[..., :3].norm(dim=-1).amax(dim=-1))
            rotation_margin = (
                1.0
                - normalized[..., 3:].norm(dim=-1).amax(dim=-1))
        else:
            translation_margin = risk.new_ones(risk.shape)
            rotation_margin = risk.new_ones(risk.shape)
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
        batch_size = decisionBackbone.size(0)
        if tuple(robotStateEncoding.joint_feature.shape) != (
            batch_size, self.pose_feature_dim
        ):
            raise ValueError("joint feature does not match decision batch")
        if tuple(robotStateEncoding.joint_tokens.shape) != (
            batch_size, self.joint_count, self.pose_feature_dim
        ):
            raise ValueError("joint tokens do not match decision batch")
        expected_joint_mask = (
            batch_size, self.joint_count)
        if tuple(
            robotStateEncoding.joint_state_valid.shape
        ) != expected_joint_mask:
            raise ValueError("joint state validity does not match dof count")
        if tuple(
            robotStateEncoding.joint_controllable.shape
        ) != expected_joint_mask:
            raise ValueError("joint controllability does not match dof count")
        expected_endpoint_mask = (batch_size, self.endpoint_count)
        if tuple(
            robotStateEncoding.endpoint_state_valid.shape
        ) != expected_endpoint_mask:
            raise ValueError("endpoint state validity does not match count")
        if tuple(
            robotStateEncoding.endpoint_controllable.shape
        ) != expected_endpoint_mask:
            raise ValueError("endpoint controllability does not match count")
        endpoint_pose_encoding = robotStateEncoding.endpoint_pose
        endpoint_control_encoding = robotStateEncoding.endpoint_control
        physical_reference = MaskRobotPhysicalReference(
            robotPhysicalReference)
        factors = self.factor_projector(torch.cat([
            decisionBackbone,
            planLatent,
            subgoalFeature,
            endpoint_pose_encoding.endpoint_pose_feat,
            physical_reference,
            endpoint_control_encoding.control_feedback_feat], dim=-1))
        joint_residual = self.joint_factor_residual(
            robotStateEncoding.joint_feature)
        joint_residual = joint_residual * robotStateEncoding.joint_state_valid.any(
            dim=1, keepdim=True).to(dtype=joint_residual.dtype)
        (
            joint_task,
            joint_motion,
            joint_dyn,
            joint_constraint,
            joint_uncertainty,
        ) = torch.split(joint_residual, [
            self.factor_projector.task_dim,
            self.factor_projector.motion_dim,
            self.factor_projector.dyn_dim,
            self.factor_projector.constraint_dim,
            self.factor_projector.uncertainty_dim,], dim=-1)
        factors["z_task"] = factors["z_task"] + joint_task
        factors["z_motion"] = factors["z_motion"] + joint_motion
        factors["z_dyn"] = factors["z_dyn"] + joint_dyn
        factors["z_constraint"] = factors["z_constraint"] + joint_constraint
        factors["z_uncertainty"] = factors["z_uncertainty"] + joint_uncertainty
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
            endpoint_pose_encoding.endpoint_pose_feat,
            planLatent,
            subgoalFeature,
            cross], dim=-1)
        dyn_in = torch.cat([
            factors["z_dyn"],
            endpoint_pose_encoding.endpoint_pose_feat,
            cross], dim=-1)
        residual_in = torch.cat([
            factors["z_uncertainty"],
            endpoint_pose_encoding.endpoint_pose_feat,
            subgoalFeature], dim=-1)
        endpoint_pose_tokens = endpoint_pose_encoding.endpoint_pose_tokens
        endpoint_context = torch.cat([
            endpoint_pose_tokens,
            endpoint_control_encoding.endpoint_control_tokens,], dim=-1)

        runtime_action_mask = (
            self.endpoint_task_mask.to(device=decisionBackbone.device)
            & robotStateEncoding.endpoint_state_valid.unsqueeze(-1)
            & robotStateEncoding.endpoint_controllable.unsqueeze(-1))
        decision_latent = (
            self.action_head(action_in, endpoint_context)
            + 0.1 * self.dynamics_head(dyn_in, endpoint_context)
            + 0.1 * self.residual_compensator(
                residual_in, endpoint_context)
            + 0.1 * self.endpoint_action_refiner(
                endpoint_pose_tokens,
                endpoint_control_encoding.endpoint_control_tokens,
                factors["z_motion"],
                factors["z_dyn"],
                factors["z_uncertainty"],
                subgoalFeature)) * runtime_action_mask.to(
                    dtype=decisionBackbone.dtype)
        decision_tensor = self.ProjectDecisionLatent(
            decision_latent) * runtime_action_mask.to(
                dtype=decisionBackbone.dtype)
        joint_command_mask = robotStateEncoding.joint_controllable
        joint_command_context = torch.cat([
            z_cat,
            planLatent,
            subgoalFeature,
            physical_reference,
            cross,], dim=-1)
        joint_variable_command = self.MaskJointVariableCommand(
            self.joint_variable_command_head(
                joint_command_context,
                robotStateEncoding.joint_tokens,
                joint_command_mask),
            joint_command_mask)
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
            joint_variable_command=joint_variable_command,
            joint_variable_command_mask=joint_command_mask,
        )
