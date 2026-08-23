from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import torch

from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import CONTROL_DOF_NAMES


TEXT_TRUST_OCR_OBSERVED = "ocr_observed"
TEXT_TRUST_OPERATOR_COMMAND = "operator_command"
TEXT_TRUST_UNSAFE_EXTERNAL = "unsafe_external"
ROBOT_STATE_WIRE_SCHEMA_VERSION = 14
SENSOR_PACKET_WIRE_SCHEMA_VERSION = 4
DECISION_WIRE_SCHEMA_VERSION = 10
OFFLINE_SENSOR_MANIFEST_SCHEMA_VERSION = 10


@dataclass(frozen=True)
class CameraCalibration:
    calibration_id: str
    frame_name: str
    intrinsics: torch.Tensor


ROBOT_STATE_WIRE_METADATA_FIELDS = (
    "schema_version",
    "stream_id",
    "sequence_index",
    "frame_id",
    "calibration_id",
    "world_frame_id",
    "description_id",
    "model_contract_id",
    "adapter_id",
    "node_names",
    "joint_names",
    "joint_type_names",
    "joint_type",
    "joint_parent_node",
    "joint_child_node",
    "joint_variable_names",
    "joint_variable_joint_index",
    "joint_variable_child_node",
    "joint_variable_local_index",
    "group_names",
    "joint_variable_commandable",
    "joint_lower",
    "joint_upper",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_variable_command_representation",
    "joint_variable_command_reference",
    "joint_variable_command_range",
    "joint_variable_command_delta_scale",
    "joint_variable_unit",
    "joint_variable_command_limit_policy",
    "endpoint_names",
    "endpoint_task_mask",
    "observer_attachment_name",
    "observer_attachment_kind",
    "observer_frame_name",
    "observer_calibration_id",
    "observer_control_joint_indices",
    "observer_control_group_index",
    "observer_pose_frame",
    "observer_pose_convention",
    "observer_pose_time_reference",
    "node_pose_frame",
    "node_pose_convention",
    "node_pose_time_reference",
    "node_twist_frame",
    "node_twist_convention",
    "node_twist_time_reference",
    "node_twist_linear_unit",
    "node_twist_angular_unit",
    "pose_frame",
    "pose_convention",
    "pose_time_reference",
    "pose_unit",
    "quaternion_order",
    "pose_handedness",
    "base_orientation_convention",
    "gravity_convention",
)

SENSOR_PACKET_WIRE_FIELDS = (
    "schema_version",
    "stream_id",
    "sequence_index",
    "frame_id",
    "calibration_id",
    "rgb_encoding",
    "depth_unit",
    "text_ext",
    "text_trust",
    "sample_actions",
    "deterministic_actor",
    "rgb",
    "depth",
    "depth_valid",
)

DECISION_REQUEST_PROVENANCE_FIELDS = (
    "stream_id",
    "sequence_index",
    "frame_id",
    "calibration_id",
    "world_frame_id",
    "description_id",
    "model_contract_id",
    "adapter_id",
)

OFFLINE_SENSOR_MANIFEST_FIELDS = (
    "schema_version",
    "robot_state_wire_schema_version",
    "calibration_id",
    "rgb_encoding",
    "depth_unit",
    "depth_representation",
    "rgb_depth_alignment",
    "rectification",
    "synchronization",
    "object_motion_frame",
    "object_motion_representation",
    "object_motion_reference",
    "object_motion_translation_unit",
    "object_motion_quaternion_order",
    "entity_motion_ontology_version",
    "entity_realm_names",
    "entity_agency_names",
    "motion_layer_names",
    "ontology_relation_names",
    "layer_agency_representation",
    "surface_content_motion_frame",
    "description_id",
    "model_contract_id",
    "adapter_id",
    "robot_control_axis_dim",
    "robot_body_role_names",
    "robot_body_side_names",
    "robot_body_capability_names",
    "node_names",
    "joint_names",
    "joint_type_names",
    "joint_type",
    "joint_parent_node",
    "joint_child_node",
    "group_names",
    "endpoint_names",
    "gripper_names",
    "sensor_names",
    "sensor_types",
    "joint_variable_names",
    "node_count",
    "joint_count",
    "group_count",
    "endpoint_count",
    "gripper_count",
    "sensor_count",
    "joint_dof_count",
    "commandable_joint_dof_count",
    "task_control_coordinate_count",
    "joint_variable_commandable",
    "joint_lower",
    "joint_upper",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_variable_command_representation",
    "joint_variable_command_reference",
    "joint_variable_command_range",
    "joint_variable_command_delta_scale",
    "joint_variable_unit",
    "joint_variable_command_limit_policy",
    "joint_variable_joint_index",
    "joint_variable_child_node",
    "joint_variable_local_index",
    "parent_index",
    "node_role",
    "node_side",
    "node_capability",
    "group_role",
    "group_side",
    "group_capability",
    "endpoint_to_node",
    "endpoint_task_mask",
    "endpoint_role",
    "endpoint_side",
    "endpoint_capability",
    "sensor_to_node",
    "sensor_role",
    "sensor_side",
    "sensor_capability",
    "observer_valid",
    "observer_controllable",
    "observer_attachment_name",
    "observer_frame_name",
    "observer_calibration_id",
    "observer_attachment_kind",
    "observer_attachment_index",
    "observer_node_index",
    "observer_sensor_index",
    "observer_endpoint_index",
    "observer_control_joint_indices",
    "observer_control_group_index",
    "observer_pose_frame",
    "observer_pose_convention",
    "observer_pose_time_reference",
    "node_pose_frame",
    "node_pose_convention",
    "node_pose_time_reference",
    "node_twist_frame",
    "node_twist_convention",
    "node_twist_time_reference",
    "node_twist_linear_unit",
    "node_twist_angular_unit",
    "node_runtime_fields",
    "observer_runtime_fields",
)


class RobotState(TypedDict):
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    joint_effort: torch.Tensor
    joint_observed: torch.Tensor
    joint_healthy: torch.Tensor
    joint_controllable: torch.Tensor
    node_pose_world: torch.Tensor
    node_twist_world: torch.Tensor
    node_observed: torch.Tensor
    node_healthy: torch.Tensor
    endpoint_pose: torch.Tensor
    endpoint_observed: torch.Tensor
    endpoint_healthy: torch.Tensor
    endpoint_controllable: torch.Tensor
    observer_pose_world: torch.Tensor
    observer_pose_valid: torch.Tensor
    base_orientation_world: torch.Tensor
    gravity_direction_world: torch.Tensor
    planner_expected_endpoint_pose: torch.Tensor
    planner_progress: torch.Tensor
    planner_tracking_error: torch.Tensor
    planner_executing: torch.Tensor
    planner_reached: torch.Tensor
    planner_failed: torch.Tensor
    model_command_executed: torch.Tensor
    executed_action_id: torch.Tensor


ROBOT_STATE_FIELDS = tuple(RobotState.__annotations__)
ROBOT_STATE_MASK_FIELDS = (
    "joint_observed",
    "joint_healthy",
    "joint_controllable",
    "node_observed",
    "node_healthy",
    "endpoint_observed",
    "endpoint_healthy",
    "endpoint_controllable",
)
ROBOT_STATE_SCALAR_MASK_FIELDS = ("observer_pose_valid",)
ROBOT_STATE_WIRE_FIELDS = ROBOT_STATE_WIRE_METADATA_FIELDS + ROBOT_STATE_FIELDS


def ValidateRobotTensorContract(contract: Any) -> None:
    required = {
        "description_id", "model_contract_id", "adapter_id",
        "node_names", "joint_names", "group_names",
        "endpoint_names", "joint_variable_names", "gripper_names",
        "sensor_names", "sensor_types", "node_count", "joint_count",
        "group_count", "endpoint_count", "joint_dof_count",
        "commandable_joint_dof_count", "task_control_coordinate_count",
        "gripper_count", "sensor_count", "parent_index",
        "joint_parent_node", "joint_child_node", "joint_type",
        "joint_variable_commandable",
        "joint_variable_joint_index", "joint_variable_child_node",
        "joint_variable_local_index", "joint_lower", "joint_upper",
        "joint_effort_limit", "joint_velocity_limit",
        "joint_variable_command_delta_scale", "joint_variable_unit",
        "joint_variable_command_representation",
        "joint_variable_command_reference", "joint_variable_command_range",
        "joint_variable_command_limit_policy", "group_node_mask",
        "group_joint_mask", "node_role", "node_side", "node_capability",
        "group_role", "group_side", "group_capability",
        "endpoint_to_node", "endpoint_task_mask", "endpoint_role",
        "endpoint_side", "endpoint_capability",
        "gripper_endpoint_index", "sensor_to_node",
        "sensor_role", "sensor_side", "sensor_capability", "observer_valid",
        "observer_controllable", "observer_attachment_name",
        "observer_frame_name", "observer_calibration_id",
        "observer_attachment_kind",
        "observer_attachment_index", "observer_node_index",
        "observer_sensor_index", "observer_endpoint_index",
        "observer_control_joint_indices", "observer_control_group_index",
        "group_dof_count",
    }
    missing = sorted(name for name in required if not hasattr(contract, name))
    if missing:
        raise TypeError(f"robot tensor contract fields are missing: {missing}")
    for name in (
        "description_id", "model_contract_id", "adapter_id",
    ):
        if type(getattr(contract, name)) is not str or not getattr(contract, name):
            raise ValueError(f"robot tensor contract {name} must be a non-empty string")
    counts = (
        "node_count",
        "joint_count",
        "group_count",
        "endpoint_count",
        "joint_dof_count",
        "commandable_joint_dof_count",
        "task_control_coordinate_count",
        "gripper_count",
        "sensor_count",
    )
    for name in counts:
        value = getattr(contract, name)
        minimum = 1 if name == "node_count" else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"robot tensor contract {name} is invalid")
    if contract.commandable_joint_dof_count > contract.joint_dof_count:
        raise ValueError("robot tensor contract commandable joint count is invalid")
    if contract.task_control_coordinate_count > (
        contract.endpoint_count * ModuleDim.RobotControlAxisDim
    ):
        raise ValueError("robot tensor contract task coordinate count is invalid")
    names_and_counts = (
        ("node_names", "node_count"),
        ("joint_names", "joint_count"),
        ("group_names", "group_count"),
        ("endpoint_names", "endpoint_count"),
        ("joint_variable_names", "joint_dof_count"),
        ("gripper_names", "gripper_count"),
        ("sensor_names", "sensor_count"),
    )
    for names_field, count_field in names_and_counts:
        names = getattr(contract, names_field)
        count = getattr(contract, count_field)
        if (
            type(names) is not tuple
            or len(names) != count
            or any(type(name) is not str or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                f"robot tensor contract {names_field} does not match {count_field}")
    if (
        type(contract.sensor_types) is not tuple
        or len(contract.sensor_types) != contract.sensor_count
        or any(type(name) is not str or not name for name in contract.sensor_types)
    ):
        raise ValueError("robot tensor contract sensor_types does not match sensor_count")
    tensor_specs = (
        ("parent_index", torch.long, (contract.node_count,)),
        ("joint_parent_node", torch.long, (contract.joint_count,)),
        ("joint_child_node", torch.long, (contract.joint_count,)),
        ("joint_type", torch.long, (contract.joint_count,)),
        ("joint_variable_commandable", torch.bool, (contract.joint_dof_count,)),
        ("joint_variable_joint_index", torch.long, (contract.joint_dof_count,)),
        ("joint_variable_child_node", torch.long, (contract.joint_dof_count,)),
        ("joint_variable_local_index", torch.long, (contract.joint_dof_count,)),
        ("joint_lower", torch.float32, (contract.joint_dof_count,)),
        ("joint_upper", torch.float32, (contract.joint_dof_count,)),
        ("joint_effort_limit", torch.float32, (contract.joint_dof_count,)),
        ("joint_velocity_limit", torch.float32, (contract.joint_dof_count,)),
        ("joint_variable_command_delta_scale", torch.float32, (
            contract.joint_dof_count,)),
        ("group_node_mask", torch.bool, (contract.group_count, contract.node_count)),
        ("group_joint_mask", torch.bool, (contract.group_count, contract.joint_count)),
        ("node_role", torch.long, (contract.node_count,)),
        ("node_side", torch.long, (contract.node_count,)),
        ("node_capability", torch.bool, (contract.node_count, ModuleDim.RobotBodyCapabilityDim)),
        ("group_role", torch.long, (contract.group_count,)),
        ("group_side", torch.long, (contract.group_count,)),
        ("group_capability", torch.bool, (contract.group_count, ModuleDim.RobotBodyCapabilityDim)),
        ("endpoint_to_node", torch.long, (contract.endpoint_count,)),
        ("endpoint_task_mask", torch.bool, (contract.endpoint_count, ModuleDim.RobotControlAxisDim)),
        ("endpoint_role", torch.long, (contract.endpoint_count,)),
        ("endpoint_side", torch.long, (contract.endpoint_count,)),
        ("endpoint_capability", torch.bool, (contract.endpoint_count, ModuleDim.RobotBodyCapabilityDim)),
        ("gripper_endpoint_index", torch.long, (contract.gripper_count,)),
        ("sensor_to_node", torch.long, (contract.sensor_count,)),
        ("sensor_role", torch.long, (contract.sensor_count,)),
        ("sensor_side", torch.long, (contract.sensor_count,)),
        ("sensor_capability", torch.bool, (contract.sensor_count, ModuleDim.RobotBodyCapabilityDim)),
    )
    for name, dtype, shape in tensor_specs:
        value = getattr(contract, name)
        if not torch.is_tensor(value) or value.dtype != dtype or tuple(value.shape) != shape:
            raise ValueError(
                f"robot tensor contract {name} must be {dtype} with shape {shape}")
    node_count = contract.node_count
    joint_count = contract.joint_count
    joint_dof_count = contract.joint_dof_count
    group_count = contract.group_count
    endpoint_count = contract.endpoint_count
    sensor_count = contract.sensor_count
    parent_index = contract.parent_index.detach().cpu()
    if bool(((parent_index < -1) | (parent_index >= node_count)).any().item()):
        raise ValueError("robot tensor contract parent_index contains an invalid node")
    roots = 0
    for node in range(node_count):
        current = node
        visited = set()
        while current >= 0:
            if current in visited:
                raise ValueError("robot tensor contract parent graph contains a cycle")
            visited.add(current)
            current = int(parent_index[current].item())
        if int(parent_index[node].item()) == -1:
            roots += 1
    if roots != 1:
        raise ValueError("robot tensor contract must contain one kinematic root")
    joint_parent = contract.joint_parent_node.detach().cpu()
    joint_child = contract.joint_child_node.detach().cpu()
    joint_type = contract.joint_type.detach().cpu()
    if bool(((joint_parent < -1) | (joint_parent >= node_count)).any().item()):
        raise ValueError(
            "robot tensor contract joint_parent_node contains an invalid node")
    if bool(((joint_child < 0) | (joint_child >= node_count)).any().item()):
        raise ValueError(
            "robot tensor contract joint_child_node contains an invalid node")
    if bool(((joint_type < 0) | (
        joint_type >= ModuleDim.RobotJointTypeClasses)).any().item()
    ):
        raise ValueError("robot tensor contract joint_type is invalid")
    root_node = int(torch.nonzero(
        parent_index < 0,
        as_tuple=False).flatten()[0].item())
    internal_joint = joint_parent.ge(0)
    external_joint = ~internal_joint
    internal_child = joint_child[internal_joint]
    external_child = joint_child[external_joint]
    if (
        int(internal_joint.sum().item()) != node_count - 1
        or internal_child.unique().numel() != internal_child.numel()
        or bool((parent_index[internal_child]
            != joint_parent[internal_joint]).any().item())
        or set(internal_child.tolist())
        != set(torch.nonzero(parent_index >= 0, as_tuple=False).flatten().tolist())
    ):
        raise ValueError("robot tensor contract joint graph is inconsistent")
    if (
        bool((external_child != root_node).any().item())
        or external_child.unique().numel() != external_child.numel()
        or bool((~torch.isin(
            joint_type[external_joint],
            torch.tensor([0, 4, 5], dtype=torch.long))).any().item())
    ):
        raise ValueError("robot tensor contract virtual joint graph is inconsistent")
    variable_joint = contract.joint_variable_joint_index.detach().cpu()
    variable_child = contract.joint_variable_child_node.detach().cpu()
    variable_local = contract.joint_variable_local_index.detach().cpu()
    if bool(((variable_joint < 0) | (variable_joint >= joint_count)).any().item()):
        raise ValueError("robot tensor contract joint runtime layout contains an invalid joint")
    if bool(((variable_child < 0) | (variable_child >= node_count)).any().item()):
        raise ValueError("robot tensor contract joint runtime layout contains an invalid node")
    if bool((variable_local < 0).any().item()):
        raise ValueError("robot tensor contract joint runtime local indices are invalid")
    if bool((variable_child != joint_child[variable_joint]).any().item()):
        raise ValueError("robot tensor contract joint runtime child mapping is invalid")
    for joint_index in range(joint_count):
        local = variable_local[variable_joint == joint_index]
        if sorted(local.tolist()) != list(range(local.numel())):
            raise ValueError("robot tensor contract joint local layout is invalid")
    virtual_dof = {0: 0, 4: 3, 5: 6}
    for joint_index in torch.nonzero(
        external_joint,
        as_tuple=False).flatten().tolist():
        variables = variable_joint.eq(joint_index)
        if int(variables.sum().item()) != virtual_dof[int(joint_type[joint_index].item())]:
            raise ValueError(
                "robot tensor contract virtual joint DOF layout is invalid")
        if bool(contract.joint_variable_commandable.detach().cpu()[
            variables].any().item()
        ):
            raise ValueError(
                "robot tensor contract virtual joint variables cannot be commandable")
    if (
        type(contract.group_dof_count) is not tuple
        or len(contract.group_dof_count) != group_count
        or any(type(value) is not int or value < 0 for value in contract.group_dof_count)
    ):
        raise ValueError("robot tensor contract group dof counts are invalid")
    for group_index in range(group_count):
        expected_group_dof = int(contract.group_joint_mask[
            group_index, variable_joint].sum().item())
        if contract.group_dof_count[group_index] != expected_group_dof:
            raise ValueError("robot tensor contract group dof count is inconsistent")
    if int(contract.joint_variable_commandable.sum().item()) != contract.commandable_joint_dof_count:
        raise ValueError("robot tensor contract commandable joint count does not match its mask")
    if bool((contract.joint_lower > contract.joint_upper).any().item()):
        raise ValueError("robot tensor contract joint limits are invalid")
    if bool((contract.joint_effort_limit < 0.0).any().item()) or bool((contract.joint_velocity_limit < 0.0).any().item()):
        raise ValueError("robot tensor contract joint runtime limits are invalid")
    if (
        bool((~torch.isfinite(
            contract.joint_variable_command_delta_scale)).any().item())
        or bool((contract.joint_variable_command_delta_scale <= 0.0).any().item())
    ):
        raise ValueError("robot joint command delta scale is invalid")
    if (
        type(contract.joint_variable_unit) is not tuple
        or len(contract.joint_variable_unit) != joint_dof_count
        or any(unit not in ("meter", "radian") for unit in contract.joint_variable_unit)
    ):
        raise ValueError("robot joint variable units are invalid")
    virtual_units = {
        0: (),
        4: ("meter", "meter", "radian"),
        5: (
            "meter", "meter", "meter",
            "radian", "radian", "radian"),
    }
    virtual_scales = {
        0: (),
        4: (1.0, 1.0, torch.pi),
        5: (1.0, 1.0, 1.0, torch.pi, torch.pi, torch.pi),
    }
    for joint_index in torch.nonzero(
        external_joint,
        as_tuple=False).flatten().tolist():
        variables = variable_joint.eq(joint_index)
        variable_indices = torch.nonzero(
            variables,
            as_tuple=False).flatten()
        type_index = int(joint_type[joint_index].item())
        if tuple(
            contract.joint_variable_unit[index]
            for index in variable_indices.tolist()
        ) != virtual_units[type_index]:
            raise ValueError("robot virtual joint variable units are invalid")
        if not torch.allclose(
            contract.joint_variable_command_delta_scale.detach().cpu()[
                variable_indices],
            torch.tensor(virtual_scales[type_index], dtype=torch.float32),
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("robot virtual joint command scales are invalid")
        if (
            not bool(torch.isneginf(contract.joint_lower.detach().cpu()[
                variable_indices]).all().item())
            or not bool(torch.isposinf(contract.joint_upper.detach().cpu()[
                variable_indices]).all().item())
            or not bool(torch.isposinf(
                contract.joint_effort_limit.detach().cpu()[
                    variable_indices]).all().item())
            or not bool(torch.isposinf(
                contract.joint_velocity_limit.detach().cpu()[
                    variable_indices]).all().item())
        ):
            raise ValueError("robot virtual joint limits are invalid")
    if (
        contract.joint_variable_command_representation
        != "normalized_position_delta"
        or contract.joint_variable_command_reference
        != "current_measured_position_at_sensor_frame_exposure"
        or contract.joint_variable_command_range != (-1.0, 1.0)
        or contract.joint_variable_command_limit_policy
        != "clamp_finite_limits_wrap_unbounded_rotation"
    ):
        raise ValueError("robot joint variable command semantics are invalid")
    semantic_specs = (
        ("node", contract.node_role, contract.node_side, contract.node_capability, node_count),
        ("group", contract.group_role, contract.group_side, contract.group_capability, group_count),
        ("endpoint", contract.endpoint_role, contract.endpoint_side, contract.endpoint_capability, endpoint_count),
        ("sensor", contract.sensor_role, contract.sensor_side, contract.sensor_capability, sensor_count),
    )
    for kind, role, side, capability, count in semantic_specs:
        if bool(((role < 0) | (role >= ModuleDim.RobotBodyRoleClasses)).any().item()):
            raise ValueError(f"robot tensor contract {kind} role is invalid")
        if bool(((side < 0) | (side >= ModuleDim.RobotBodySideClasses)).any().item()):
            raise ValueError(f"robot tensor contract {kind} side is invalid")
    endpoint_to_node = contract.endpoint_to_node.detach().cpu()
    if bool(((endpoint_to_node < 0) | (endpoint_to_node >= node_count)).any().item()):
        raise ValueError("robot tensor contract endpoint_to_node contains an invalid node")
    if int(contract.endpoint_task_mask.sum().item()) != contract.task_control_coordinate_count:
        raise ValueError("robot tensor contract task coordinate count does not match its mask")
    sensor_to_node = contract.sensor_to_node.detach().cpu()
    if bool(((sensor_to_node < 0) | (sensor_to_node >= node_count)).any().item()):
        raise ValueError("robot tensor contract sensor_to_node contains an invalid node")
    gripper_endpoint_index = contract.gripper_endpoint_index.detach().cpu()
    if bool(((gripper_endpoint_index < 0) | (gripper_endpoint_index >= endpoint_count)).any().item()):
        raise ValueError("robot tensor contract gripper endpoint is invalid")
    if type(contract.observer_valid) is not bool or type(contract.observer_controllable) is not bool:
        raise TypeError("robot tensor contract observer flags must be booleans")
    observer_attachment_name = contract.observer_attachment_name
    observer_frame_name = contract.observer_frame_name
    observer_calibration_id = contract.observer_calibration_id
    for name, value in (
        ("observer_attachment_name", observer_attachment_name),
        ("observer_frame_name", observer_frame_name),
        ("observer_calibration_id", observer_calibration_id),
    ):
        if type(value) is not str:
            raise TypeError(f"robot tensor contract {name} must be a string")
    observer_indices = (
        contract.observer_attachment_index,
        contract.observer_node_index,
        contract.observer_sensor_index,
        contract.observer_endpoint_index,
    )
    if any(type(value) is not int for value in observer_indices):
        raise TypeError("robot tensor contract observer indices must be integers")
    observer_control = contract.observer_control_joint_indices
    if (
        not torch.is_tensor(observer_control)
        or observer_control.dtype != torch.long
        or observer_control.ndim != 1
    ):
        raise ValueError(
            "robot tensor contract observer control indices are invalid")
    observer_control = observer_control.detach().cpu()
    if type(contract.observer_control_group_index) is not int:
        raise TypeError("robot observer control group index must be an integer")
    if (
        bool(((observer_control < 0) | (
            observer_control >= joint_dof_count)).any().item())
        or observer_control.unique().numel() != observer_control.numel()
        or bool((~contract.joint_variable_commandable.detach().cpu()[
            observer_control]).any().item())
    ):
        raise ValueError("robot observer control joint mapping is invalid")
    if contract.observer_control_group_index != -1 and not (
        0 <= contract.observer_control_group_index < group_count
    ):
        raise ValueError("robot observer control group index is invalid")
    if contract.observer_control_group_index >= 0 and bool(
        (~contract.group_joint_mask.detach().cpu()[
            contract.observer_control_group_index,
            variable_joint[observer_control]]).any().item()
    ):
        raise ValueError("robot observer control variables are outside their group")
    if observer_control.numel():
        if not hasattr(contract, "JointSemanticDescriptor"):
            raise TypeError("robot contract lacks joint semantic descriptors")
        joint_types = contract.JointSemanticDescriptor()[
            "joint_type"].detach().cpu()[observer_control]
        rotational_types = torch.tensor([
            1,
            2,
        ], dtype=torch.long)
        if bool((~torch.isin(joint_types, rotational_types)).any().item()):
            raise ValueError("robot observer controls must be rotational joints")
    if contract.observer_controllable != bool(observer_control.numel()):
        raise ValueError("robot observer controllability does not match its mapping")
    if not contract.observer_valid:
        if (
            observer_attachment_name
            or observer_frame_name
            or observer_calibration_id
            or contract.observer_attachment_kind != "none"
            or any(value != -1 for value in observer_indices)
            or contract.observer_controllable
            or observer_control.numel()
            or contract.observer_control_group_index != -1
        ):
            raise ValueError("robot tensor contract absent observer attachment is invalid")
        return
    if (
        not observer_attachment_name
        or not observer_frame_name
        or not observer_calibration_id
    ):
        raise ValueError("robot tensor contract observer metadata is incomplete")
    if contract.observer_attachment_kind not in ("link", "endpoint", "sensor"):
        raise ValueError("robot tensor contract observer attachment kind is invalid")
    if not (0 <= contract.observer_node_index < node_count):
        raise ValueError("robot tensor contract observer node is invalid")
    if contract.observer_attachment_kind == "link":
        valid_attachment = (
            contract.observer_attachment_index == contract.observer_node_index
            and observer_attachment_name
            == contract.node_names[contract.observer_node_index]
            and contract.observer_sensor_index == -1
            and contract.observer_endpoint_index == -1)
    elif contract.observer_attachment_kind == "sensor":
        valid_attachment = (
            0 <= contract.observer_sensor_index < sensor_count
            and contract.observer_attachment_index == contract.observer_sensor_index
            and observer_attachment_name
            == contract.sensor_names[contract.observer_sensor_index]
            and contract.observer_endpoint_index == -1
            and int(sensor_to_node[contract.observer_sensor_index].item()) == contract.observer_node_index)
    else:
        valid_attachment = (
            0 <= contract.observer_endpoint_index < endpoint_count
            and contract.observer_attachment_index == contract.observer_endpoint_index
            and observer_attachment_name
            == contract.endpoint_names[contract.observer_endpoint_index]
            and contract.observer_sensor_index == -1
            and int(endpoint_to_node[contract.observer_endpoint_index].item()) == contract.observer_node_index)
    if not valid_attachment:
        raise ValueError("robot tensor contract observer attachment indices are invalid")
    if (
        contract.observer_attachment_kind == "endpoint"
        and bool(contract.endpoint_task_mask[
            contract.observer_endpoint_index].any().item())
    ):
        raise ValueError("robot observer endpoint cannot define task axes")


def ValidateRobotObserverCalibration(
    robotContract: Any,
    calibrationId: str,
    observerFrameName: str,
) -> None:
    ValidateRobotTensorContract(robotContract)
    if type(calibrationId) is not str or not calibrationId:
        raise ValueError("calibrationId must be a non-empty string")
    if type(observerFrameName) is not str or not observerFrameName:
        raise ValueError("observerFrameName must be a non-empty string")
    if not robotContract.observer_valid:
        return
    if robotContract.observer_calibration_id != calibrationId:
        raise ValueError(
            "robot observer calibration_id does not match the configured sensor")
    if robotContract.observer_frame_name != observerFrameName:
        raise ValueError(
            "robot observer frame does not match the configured sensor")


def ExpectedRobotStateWireMetadata(robotContract: Any) -> Dict[str, Any]:
    ValidateRobotTensorContract(robotContract)
    def finite_or_none(value: torch.Tensor) -> List[Optional[float]]:
        return [
            float(item) if torch.isfinite(item) else None
            for item in value.detach().cpu().flatten()
        ]
    return {
        "description_id": robotContract.description_id,
        "model_contract_id": robotContract.model_contract_id,
        "adapter_id": robotContract.adapter_id,
        "node_names": list(robotContract.node_names),
        "joint_names": list(robotContract.joint_names),
        "joint_type_names": list(ModuleDim.RobotJointTypeNames),
        "joint_type": robotContract.joint_type.detach().cpu().tolist(),
        "joint_parent_node": (
            robotContract.joint_parent_node.detach().cpu().tolist()),
        "joint_child_node": (
            robotContract.joint_child_node.detach().cpu().tolist()),
        "joint_variable_names": list(robotContract.joint_variable_names),
        "joint_variable_joint_index": (
            robotContract.joint_variable_joint_index.detach().cpu().tolist()),
        "joint_variable_child_node": (
            robotContract.joint_variable_child_node.detach().cpu().tolist()),
        "joint_variable_local_index": (
            robotContract.joint_variable_local_index.detach().cpu().tolist()),
        "group_names": list(robotContract.group_names),
        "joint_variable_commandable": robotContract.joint_variable_commandable.detach().cpu().tolist(),
        "joint_lower": finite_or_none(robotContract.joint_lower),
        "joint_upper": finite_or_none(robotContract.joint_upper),
        "joint_effort_limit": finite_or_none(robotContract.joint_effort_limit),
        "joint_velocity_limit": finite_or_none(robotContract.joint_velocity_limit),
        "joint_variable_command_representation": robotContract.joint_variable_command_representation,
        "joint_variable_command_reference": robotContract.joint_variable_command_reference,
        "joint_variable_command_range": list(robotContract.joint_variable_command_range),
        "joint_variable_command_delta_scale": robotContract.joint_variable_command_delta_scale.detach().cpu().tolist(),
        "joint_variable_unit": list(robotContract.joint_variable_unit),
        "joint_variable_command_limit_policy": robotContract.joint_variable_command_limit_policy,
        "endpoint_names": list(robotContract.endpoint_names),
        "endpoint_task_mask": (
            robotContract.endpoint_task_mask.detach().cpu().tolist()),
        "observer_attachment_name": robotContract.observer_attachment_name,
        "observer_attachment_kind": robotContract.observer_attachment_kind,
        "observer_frame_name": robotContract.observer_frame_name,
        "observer_calibration_id": robotContract.observer_calibration_id,
        "observer_control_joint_indices": robotContract.observer_control_joint_indices.detach().cpu().tolist(),
        "observer_control_group_index": robotContract.observer_control_group_index,
        "observer_pose_frame": "world",
        "observer_pose_convention": "T_world_observer",
        "observer_pose_time_reference": "sensor_frame_exposure",
        "node_pose_frame": "world",
        "node_pose_convention": "T_world_node",
        "node_pose_time_reference": "sensor_frame_exposure",
        "node_twist_frame": "world",
        "node_twist_convention": "linear_xyz_angular_xyz",
        "node_twist_time_reference": "sensor_frame_exposure",
        "node_twist_linear_unit": "meter_per_second",
        "node_twist_angular_unit": "radian_per_second",
        "pose_frame": "world",
        "pose_convention": "T_world_endpoint",
        "pose_time_reference": "sensor_frame_exposure",
        "pose_unit": "meter",
        "quaternion_order": "xyzw",
        "pose_handedness": "right_handed",
        "base_orientation_convention": "q_world_base_xyzw",
        "gravity_convention": "unit_acceleration_direction_world",}


def ExpectedRobotCommandContract(robotContract: Any) -> Dict[str, Any]:
    metadata = ExpectedRobotStateWireMetadata(robotContract)
    fields = (
        "model_contract_id",
        "adapter_id",
        "joint_names",
        "joint_type_names",
        "joint_type",
        "joint_parent_node",
        "joint_child_node",
        "joint_variable_names",
        "joint_variable_joint_index",
        "joint_variable_child_node",
        "joint_variable_local_index",
        "group_names",
        "joint_variable_commandable",
        "joint_lower",
        "joint_upper",
        "joint_effort_limit",
        "joint_velocity_limit",
        "joint_variable_command_representation",
        "joint_variable_command_reference",
        "joint_variable_command_range",
        "joint_variable_command_delta_scale",
        "joint_variable_unit",
        "joint_variable_command_limit_policy",
    )
    return {
        **{name: metadata[name] for name in fields},
        "endpoint_task_command_representation": "local_body_se3_delta",
        "endpoint_task_command_component_order": list(CONTROL_DOF_NAMES),
        "endpoint_task_command_translation_unit": "meter",
        "endpoint_task_command_rotation_representation": "axis_angle",
        "endpoint_task_command_rotation_unit": "radian",
        "endpoint_task_command_reference": (
            "current_endpoint_pose_at_sensor_frame_exposure"),
        "endpoint_task_command_composition": (
            "T_world_target_equals_T_world_endpoint_times_T_endpoint_delta"),
        "observer_control_joint_indices": (
            robotContract.observer_control_joint_indices.detach().cpu().tolist()),
        "observer_control_group_index": (
            robotContract.observer_control_group_index),
        "observer_control_group_name": (
            None
            if robotContract.observer_control_group_index < 0
            else robotContract.group_names[
                robotContract.observer_control_group_index]),
    }


def ValidateRobotStateWirePacket(
    packet: Any,
    calibrationId: str,
    robotContract: Any,) -> None:
    if type(packet) is not dict or set(packet) != set(ROBOT_STATE_WIRE_FIELDS):
        raise ValueError("robot packet fields do not match the current schema")
    if (
        type(packet["schema_version"]) is not int
        or packet["schema_version"] != ROBOT_STATE_WIRE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported robot state packet schema")
    if type(packet["frame_id"]) is not str or not packet["frame_id"]:
        raise ValueError("RobotState frame_id must be a non-empty string")
    if type(packet["stream_id"]) is not str or not packet["stream_id"]:
        raise ValueError("RobotState stream_id must be a non-empty string")
    if type(packet["sequence_index"]) is not int or packet["sequence_index"] < 0:
        raise ValueError("RobotState sequence_index must be a non-negative integer")
    if packet["calibration_id"] != calibrationId:
        raise ValueError(
            "RobotState calibration_id does not match the configured sensor")
    if type(packet["world_frame_id"]) is not str or not packet["world_frame_id"]:
        raise ValueError("RobotState world_frame_id must be a non-empty string")
    for name, expected in ExpectedRobotStateWireMetadata(robotContract).items():
        if packet[name] != expected:
            raise ValueError(
                f"robot packet {name} does not match the current contract")
    if (
        robotContract.observer_valid
        and robotContract.observer_calibration_id != calibrationId
    ):
        raise ValueError(
            "RobotState calibration_id does not match the observer attachment")
    joint_dof_count = robotContract.joint_dof_count
    node_count = robotContract.node_count
    endpoint_count = robotContract.endpoint_count
    for name in (
        "joint_position",
        "joint_velocity",
        "joint_effort",
        "joint_observed",
        "joint_healthy",
        "joint_controllable",
    ):
        if type(packet[name]) is not list or len(packet[name]) != joint_dof_count:
            raise ValueError(
                f"RobotState {name} must contain {joint_dof_count} active joint variables")
    if type(packet["node_pose_world"]) is not list or len(
        packet["node_pose_world"]
    ) != node_count:
        raise ValueError(
            f"RobotState node_pose_world must contain {node_count} active nodes")
    if type(packet["node_twist_world"]) is not list or len(
        packet["node_twist_world"]
    ) != node_count:
        raise ValueError(
            f"RobotState node_twist_world must contain {node_count} active nodes")
    for name in ("node_observed", "node_healthy"):
        if type(packet[name]) is not list or len(packet[name]) != node_count:
            raise ValueError(
                f"RobotState {name} must contain {node_count} active nodes")
    for name in ("endpoint_pose", "planner_expected_endpoint_pose"):
        if type(packet[name]) is not list or len(packet[name]) != endpoint_count:
            raise ValueError(
                f"RobotState {name} must contain {endpoint_count} active endpoints")
    for name in (
        "endpoint_observed",
        "endpoint_healthy",
        "endpoint_controllable",
    ):
        if type(packet[name]) is not list or len(packet[name]) != endpoint_count:
            raise ValueError(
                f"RobotState {name} must contain {endpoint_count} active endpoints")
    for name in ROBOT_STATE_MASK_FIELDS:
        if any(type(value) is not bool for value in packet[name]):
            raise TypeError(f"RobotState {name} must contain booleans")
    node_pose_rows = packet["node_pose_world"]
    if any(
        type(row) is not list
        or len(row) != 7
        or any(type(value) not in (int, float) for value in row)
        for row in node_pose_rows
    ):
        raise ValueError(
            "RobotState node_pose_world must contain seven real values per node")
    node_pose_tensor = torch.as_tensor(node_pose_rows, dtype=torch.float32)
    if not bool(torch.isfinite(node_pose_tensor).all().item()):
        raise ValueError(
            "RobotState node_pose_world must contain only finite values")
    node_twist_rows = packet["node_twist_world"]
    if any(
        type(row) is not list
        or len(row) != 6
        or any(type(value) not in (int, float) for value in row)
        for row in node_twist_rows
    ):
        raise ValueError(
            "RobotState node_twist_world must contain six real values per node")
    node_twist_tensor = torch.as_tensor(node_twist_rows, dtype=torch.float32)
    if not bool(torch.isfinite(node_twist_tensor).all().item()):
        raise ValueError(
            "RobotState node_twist_world must contain only finite values")
    node_observed = torch.as_tensor(packet["node_observed"])
    node_identity = node_pose_tensor.new_zeros(node_pose_tensor.shape)
    node_identity[..., 6] = 1.0
    if (
        bool((~node_observed).any().item())
        and not torch.allclose(
            node_pose_tensor[~node_observed],
            node_identity[~node_observed],
            rtol=0.0,
            atol=1e-6)
    ):
        raise ValueError("RobotState unavailable node poses must be identity")
    if (
        bool((~node_observed).any().item())
        and bool((node_twist_tensor[~node_observed] != 0.0).any().item())
    ):
        raise ValueError("RobotState unavailable node twists must be zero")
    node_quaternion_norm = node_pose_tensor[..., 3:7].norm(dim=-1)
    if not torch.allclose(
        node_quaternion_norm[node_observed],
        torch.ones_like(node_quaternion_norm[node_observed]),
        rtol=1e-3,
        atol=1e-3,
    ):
        raise ValueError("RobotState node pose quaternions must have unit length")
    if any(
        healthy and not observed
        for healthy, observed in zip(
            packet["node_healthy"], packet["node_observed"])
    ):
        raise ValueError("RobotState node_healthy requires node_observed")
    if type(packet["observer_pose_valid"]) is not bool:
        raise TypeError("RobotState observer_pose_valid must be a boolean")
    observer_pose = packet["observer_pose_world"]
    if (
        type(observer_pose) is not list
        or len(observer_pose) != 7
        or any(type(value) not in (int, float) for value in observer_pose)
    ):
        raise ValueError("RobotState observer_pose_world must contain seven values")
    try:
        observer_pose_tensor = torch.as_tensor(observer_pose, dtype=torch.float32)
    except Exception as error:
        raise ValueError(
            "RobotState observer_pose_world must contain real numbers") from error
    if not bool(torch.isfinite(observer_pose_tensor).all().item()):
        raise ValueError(
            "RobotState observer_pose_world must contain only finite values")
    identity = observer_pose_tensor.new_tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    if not robotContract.observer_valid or not packet["observer_pose_valid"]:
        if packet["observer_pose_valid"] or not torch.allclose(
            observer_pose_tensor,
            identity,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "RobotState unavailable observer pose must be invalid identity")
    elif not torch.allclose(
        observer_pose_tensor[3:7].norm(),
        observer_pose_tensor.new_tensor(1.0),
        rtol=1e-3,
        atol=1e-3,
    ):
        raise ValueError("RobotState observer pose quaternion must have unit length")
    if (
        robotContract.observer_valid
        and robotContract.observer_attachment_kind == "link"
        and robotContract.observer_frame_name
        == robotContract.observer_attachment_name
        and packet["observer_pose_valid"]
        and packet["node_observed"][robotContract.observer_node_index]
        and not torch.allclose(
            observer_pose_tensor,
            node_pose_tensor[robotContract.observer_node_index],
            rtol=0.0,
            atol=1e-5)
    ):
        raise ValueError(
            "RobotState observer and attachment node poses are inconsistent")


def ExpectedOfflineSensorManifest(
    calibrationId: str,
    robotContract: Any,
    sensorFrameName: str,) -> Dict[str, Any]:
    ValidateRobotObserverCalibration(
        robotContract,
        calibrationId,
        sensorFrameName)
    node_count = robotContract.node_count
    joint_count = robotContract.joint_count
    group_count = robotContract.group_count
    endpoint_count = robotContract.endpoint_count
    gripper_count = robotContract.gripper_count
    sensor_count = robotContract.sensor_count
    joint_dof_count = robotContract.joint_dof_count
    command_contract = ExpectedRobotCommandContract(robotContract)
    return {
        "schema_version": OFFLINE_SENSOR_MANIFEST_SCHEMA_VERSION,
        "robot_state_wire_schema_version": ROBOT_STATE_WIRE_SCHEMA_VERSION,
        "calibration_id": calibrationId,
        "rgb_encoding": "rgb8",
        "depth_unit": "meter",
        "depth_representation": "optical_axis_z",
        "rgb_depth_alignment": "registered_to_rgb",
        "rectification": "rectified",
        "synchronization": "synchronized_exposure",
        "object_motion_frame": sensorFrameName,
        "object_motion_representation": "se3_spatial_delta",
        "object_motion_reference": (
            "previous_to_current_with_optional_observer_pose_reference"
            if robotContract.observer_valid
            else "previous_to_current_without_observer_pose_reference"),
        "object_motion_translation_unit": "meter",
        "object_motion_quaternion_order": "xyzw",
        "entity_motion_ontology_version": 1,
        "entity_realm_names": list(ModuleDim.PstRealmNames),
        "entity_agency_names": list(ModuleDim.PstAgencyNames),
        "motion_layer_names": list(ModuleDim.PstMotionLayerNames),
        "ontology_relation_names": list(
            ModuleDim.PstOntologyRelationNames),
        "layer_agency_representation": (
            "per_motion_layer_categorical_with_valid_mask"),
        "surface_content_motion_frame": "canonical_surface_uv",
        "description_id": robotContract.description_id,
        "model_contract_id": robotContract.model_contract_id,
        "adapter_id": robotContract.adapter_id,
        "robot_control_axis_dim": ModuleDim.RobotControlAxisDim,
        "robot_body_role_names": list(ModuleDim.RobotBodyRoleNames),
        "robot_body_side_names": list(ModuleDim.RobotBodySideNames),
        "robot_body_capability_names": list(
            ModuleDim.RobotBodyCapabilityNames),
        "node_names": list(robotContract.node_names),
        "joint_names": list(robotContract.joint_names),
        "joint_type_names": list(ModuleDim.RobotJointTypeNames),
        "joint_type": robotContract.joint_type.detach().cpu().tolist(),
        "joint_parent_node": (
            robotContract.joint_parent_node.detach().cpu().tolist()),
        "joint_child_node": (
            robotContract.joint_child_node.detach().cpu().tolist()),
        "group_names": list(robotContract.group_names),
        "endpoint_names": list(robotContract.endpoint_names),
        "gripper_names": list(robotContract.gripper_names),
        "sensor_names": list(robotContract.sensor_names),
        "sensor_types": list(robotContract.sensor_types),
        "joint_variable_names": list(robotContract.joint_variable_names),
        "node_count": node_count,
        "joint_count": joint_count,
        "group_count": group_count,
        "endpoint_count": endpoint_count,
        "gripper_count": gripper_count,
        "sensor_count": sensor_count,
        "joint_dof_count": joint_dof_count,
        "commandable_joint_dof_count": (
            robotContract.commandable_joint_dof_count),
        "task_control_coordinate_count": (
            robotContract.task_control_coordinate_count),
        "joint_variable_commandable": robotContract.joint_variable_commandable.detach().cpu().tolist(),
        "joint_lower": command_contract["joint_lower"],
        "joint_upper": command_contract["joint_upper"],
        "joint_effort_limit": command_contract["joint_effort_limit"],
        "joint_velocity_limit": command_contract["joint_velocity_limit"],
        "joint_variable_command_representation": command_contract[
            "joint_variable_command_representation"],
        "joint_variable_command_reference": command_contract[
            "joint_variable_command_reference"],
        "joint_variable_command_range": command_contract[
            "joint_variable_command_range"],
        "joint_variable_command_delta_scale": command_contract[
            "joint_variable_command_delta_scale"],
        "joint_variable_unit": command_contract["joint_variable_unit"],
        "joint_variable_command_limit_policy": command_contract[
            "joint_variable_command_limit_policy"],
        "joint_variable_joint_index": robotContract.joint_variable_joint_index.detach().cpu().tolist(),
        "joint_variable_child_node": robotContract.joint_variable_child_node.detach().cpu().tolist(),
        "joint_variable_local_index": robotContract.joint_variable_local_index.detach().cpu().tolist(),
        "parent_index": robotContract.parent_index.detach().cpu().tolist(),
        "node_role": robotContract.node_role.detach().cpu().tolist(),
        "node_side": robotContract.node_side.detach().cpu().tolist(),
        "node_capability": robotContract.node_capability.detach().cpu().tolist(),
        "group_role": robotContract.group_role.detach().cpu().tolist(),
        "group_side": robotContract.group_side.detach().cpu().tolist(),
        "group_capability": robotContract.group_capability.detach().cpu().tolist(),
        "endpoint_to_node": robotContract.endpoint_to_node.detach().cpu().tolist(),
        "endpoint_task_mask": robotContract.endpoint_task_mask.detach().cpu().tolist(),
        "endpoint_role": robotContract.endpoint_role.detach().cpu().tolist(),
        "endpoint_side": robotContract.endpoint_side.detach().cpu().tolist(),
        "endpoint_capability": robotContract.endpoint_capability.detach().cpu().tolist(),
        "sensor_to_node": robotContract.sensor_to_node.detach().cpu().tolist(),
        "sensor_role": robotContract.sensor_role.detach().cpu().tolist(),
        "sensor_side": robotContract.sensor_side.detach().cpu().tolist(),
        "sensor_capability": robotContract.sensor_capability.detach().cpu().tolist(),
        "observer_valid": robotContract.observer_valid,
        "observer_controllable": robotContract.observer_controllable,
        "observer_attachment_name": robotContract.observer_attachment_name,
        "observer_frame_name": robotContract.observer_frame_name,
        "observer_calibration_id": robotContract.observer_calibration_id,
        "observer_attachment_kind": robotContract.observer_attachment_kind,
        "observer_attachment_index": robotContract.observer_attachment_index,
        "observer_node_index": robotContract.observer_node_index,
        "observer_sensor_index": robotContract.observer_sensor_index,
        "observer_endpoint_index": robotContract.observer_endpoint_index,
        "observer_control_joint_indices": command_contract[
            "observer_control_joint_indices"],
        "observer_control_group_index": command_contract[
            "observer_control_group_index"],
        "observer_pose_frame": "world",
        "observer_pose_convention": "T_world_observer",
        "observer_pose_time_reference": "sensor_frame_exposure",
        "node_pose_frame": "world",
        "node_pose_convention": "T_world_node",
        "node_pose_time_reference": "sensor_frame_exposure",
        "node_twist_frame": "world",
        "node_twist_convention": "linear_xyz_angular_xyz",
        "node_twist_time_reference": "sensor_frame_exposure",
        "node_twist_linear_unit": "meter_per_second",
        "node_twist_angular_unit": "radian_per_second",
        "node_runtime_fields": [
            "node_pose_world",
            "node_twist_world",
            "node_observed",
            "node_healthy",
        ],
        "observer_runtime_fields": [
            "observer_pose_world",
            "observer_pose_valid",
        ],
    }


def ValidateOfflineSensorManifest(
    manifest: Any,
    calibrationId: str,
    robotContract: Any,
    sensorFrameName: str,) -> None:
    if (
        type(manifest) is not dict
        or set(manifest) != set(OFFLINE_SENSOR_MANIFEST_FIELDS)
    ):
        raise ValueError(
            "offline sensor manifest fields do not match the current schema")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != OFFLINE_SENSOR_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported offline sensor manifest schema")
    expected = ExpectedOfflineSensorManifest(
        calibrationId,
        robotContract,
        sensorFrameName)
    for name, value in expected.items():
        if manifest[name] != value:
            raise ValueError(
                f"offline sensor manifest {name} does not match the current contract")


@dataclass
class BrainStepInput:
    frame: torch.Tensor
    text_ext: Optional[List[Optional[str]]]
    reward_ext: Optional[torch.Tensor]
    done_flag: Optional[torch.Tensor]
    is_train: bool
    sample_actions: bool
    deterministic_actor: bool
    depth: torch.Tensor
    depth_valid: torch.Tensor
    perception_targets: Optional[Dict[str, torch.Tensor]]
    robot_state: RobotState
    text_trust: Optional[List[str]] = None
    compute_critic_loss: bool = True


@dataclass
class AgentActInput:
    frame: torch.Tensor
    text_ext: Optional[List[Optional[str]]]
    reward: Optional[torch.Tensor]
    done: Optional[torch.Tensor]
    sample_actions: bool
    deterministic_actor: bool
    depth: torch.Tensor
    depth_valid: torch.Tensor
    robot_state: RobotState
    perception_targets: Optional[Dict[str, torch.Tensor]] = None
    text_trust: Optional[List[str]] = None
    compute_critic_loss: bool = True


@dataclass
class BrainStepOutput:
    decision: Dict[str, Any]
    world: Dict[str, torch.Tensor]
    critic: Any
    features: Dict[str, Any]
    ocr: Any
    intention_texts: List[str]
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    stages: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentActOutput:
    motion_command: Any
    temporal_envelope: Any
    decision: Dict[str, Any]
    loss: Optional[torch.Tensor]
    ocr: Any
    intention_texts: List[str]
    optimization_losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    total_loss: Optional[torch.Tensor] = None
    transport_delayed_loss: Optional[torch.Tensor] = None
    physical_loss: Optional[torch.Tensor] = None


@dataclass
class PerceptionPhysicalStage:
    visual_state: Any
    perc_feats: torch.Tensor
    percs_seq: torch.Tensor
    object_seq: torch.Tensor
    motion_seq: torch.Tensor
    quality_seq: torch.Tensor
    pred_error_seq: torch.Tensor
    key_padding_mask: torch.Tensor
    prev_visual_for_loss: Any
    ocr_items: Any
    fuse_ocr: List[List[str]]
    ocr_semantic: torch.Tensor
    slow_refresh: bool
    text_control_refresh: bool
    pst: Dict[str, torch.Tensor]
    observed_pst: Dict[str, torch.Tensor]
    pst_summary: torch.Tensor
    world_action_feedback: torch.Tensor


@dataclass
class ValueMemoryWorldStage:
    w_preview: Dict[str, torch.Tensor]
    w_out: Dict[str, torch.Tensor]
    s_t: torch.Tensor
    r_t: torch.Tensor
    d_t: torch.Tensor
    d_tr: Any
    d_ph: Any
    done_now: torch.Tensor
    critic_out: Any
    value_current: torch.Tensor
    value_next_current: torch.Tensor
    td_sig: torch.Tensor
    unc_sig: torch.Tensor
    precision_sig: torch.Tensor
    emotion_sig: torch.Tensor
    risk_sig: torch.Tensor
    confidence_sig: torch.Tensor
    atten_out: torch.Tensor
    mem_feat: torch.Tensor
    memory_reward: torch.Tensor
    intent_hint_for_memory: torch.Tensor
    prospective_visual_prediction: Any


@dataclass
class CognitionGoalStage:
    conscious_out: Any
    intent_sem: torch.Tensor
    sym_probs: torch.Tensor
    intention_extras: Dict[str, Any]
    intention_texts: List[str]
    world_hzx_now: torch.Tensor
    goals: Dict[str, torch.Tensor]
    grounding: Dict[str, torch.Tensor]


@dataclass
class ExecutionStage:
    act_out: Dict[str, Any]
    motion_command: Any
    candidate_motion_command: Any
    temporal_envelope: Any
    temporal_context: Any
    satisfaction_prob: torch.Tensor
    neuro_symbolic_out: Any
    decoupled_decision: Any
    planner_prior: Optional[Dict[str, torch.Tensor]]
    network_decision_feature: torch.Tensor
    prospective_action_embed: torch.Tensor
    temporal_goal: Dict[str, torch.Tensor]
    world_abstract: Dict[str, torch.Tensor]
    reference_uncertainty: torch.Tensor
