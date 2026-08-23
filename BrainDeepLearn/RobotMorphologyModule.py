from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import json
import math
import xml.etree.ElementTree as ET

import torch


class Realm(IntEnum):
    SELF_BODY = 0
    EXTERNAL_PHYSICAL = 1
    VIRTUAL_CONTENT = 2
    VISUAL_EFFECT = 3
    UNKNOWN = 4


class Agency(IntEnum):
    SELF_CAUSED = 0
    EXTERNAL_CAUSED = 1
    AUTONOMOUS = 2
    MIXED = 3
    UNKNOWN = 4


class MotionLayer(IntEnum):
    OBSERVER_MOTION = 0
    CARRIER_MOTION = 1
    ARTICULATION_MOTION = 2
    SURFACE_CONTENT_MOTION = 3
    PHOTOMETRIC_CHANGE = 4


class OntologyRelation(IntEnum):
    DISPLAYED_ON = 0
    HELD_BY = 1
    MOVING_WITH = 2
    ATTACHED_TO_SELF = 3
    CONTACTING_SELF = 4
    REFLECTED_IN = 5
    SHADOW_OF = 6
    OCCLUDES = 7
    INSIDE_DISPLAY_REGION = 8


EntityRealm = Realm
EntityAgency = Agency


REALM_NAMES: Tuple[str, ...] = tuple(item.name.lower() for item in Realm)
AGENCY_NAMES: Tuple[str, ...] = tuple(item.name.lower() for item in Agency)
MOTION_LAYER_NAMES: Tuple[str, ...] = tuple(
    item.name.lower() for item in MotionLayer)
ONTOLOGY_RELATION_NAMES: Tuple[str, ...] = tuple(
    item.name.lower() for item in OntologyRelation)


CONTROL_DOF_NAMES: Tuple[str, ...] = (
    "translation_x",
    "translation_y",
    "translation_z",
    "rotation_x",
    "rotation_y",
    "rotation_z",
)


JOINT_TYPE_NAMES: Tuple[str, ...] = (
    "fixed",
    "revolute",
    "continuous",
    "prismatic",
    "planar",
    "floating",
)


BODY_ROLE_NAMES: Tuple[str, ...] = (
    "root",
    "torso",
    "head",
    "arm",
    "hand",
    "finger",
    "leg",
    "foot",
    "sensor",
    "tool",
    "other",
)


BODY_SIDE_NAMES: Tuple[str, ...] = (
    "left",
    "right",
    "center",
    "none",
)


BODY_CAPABILITY_NAMES: Tuple[str, ...] = (
    "manipulation",
    "support",
    "locomotion",
    "grasp",
    "observe",
    "contact",
    "balance",
)


DEFAULT_VIRTUAL_SLOT_COUNT = 32

ENTITY_ONTOLOGY_BASE_DIM = (
    len(REALM_NAMES)
    + len(MOTION_LAYER_NAMES)
    + len(MOTION_LAYER_NAMES) * len(AGENCY_NAMES)
    + len(AGENCY_NAMES)
    + 1
    + 1
    + 1
    + 1
    + 1
    + 2
    + 1
    + 2
    + 1
    + 1
)


def EntityOntologyTokenDim(selfPartSemanticDim: int) -> int:
    if type(selfPartSemanticDim) is not int or selfPartSemanticDim < 1:
        raise ValueError("selfPartSemanticDim must be a positive integer")
    return ENTITY_ONTOLOGY_BASE_DIM + selfPartSemanticDim


def _ValidateContiguousEnum(enumType) -> None:
    values = tuple(int(item.value) for item in enumType)
    expected = tuple(range(len(values)))
    if values != expected:
        raise ValueError(
            f"{enumType.__name__} values must be contiguous from zero: "
            f"expected {expected}, got {values}")


def _ValidateRobotEntityOntology() -> None:
    for enum_type in (Realm, Agency, MotionLayer, OntologyRelation):
        _ValidateContiguousEnum(enum_type)

    if not (
        len(Realm) == 5
        and len(Agency) == 5
        and len(MotionLayer) == 5
        and len(OntologyRelation) == 9
    ):
        raise ValueError("entity-motion ontology cardinalities changed")

    expected_names = {
        Realm: (
            "self_body",
            "external_physical",
            "virtual_content",
            "visual_effect",
            "unknown"),
        Agency: (
            "self_caused",
            "external_caused",
            "autonomous",
            "mixed",
            "unknown"),
        MotionLayer: (
            "observer_motion",
            "carrier_motion",
            "articulation_motion",
            "surface_content_motion",
            "photometric_change"),
        OntologyRelation: (
            "displayed_on",
            "held_by",
            "moving_with",
            "attached_to_self",
            "contacting_self",
            "reflected_in",
            "shadow_of",
            "occludes",
            "inside_display_region"),
    }
    for enum_type, names in expected_names.items():
        actual = tuple(item.name.lower() for item in enum_type)
        if actual != names:
            raise ValueError(
                f"{enum_type.__name__} stable names changed: "
                f"expected {names}, got {actual}")

    if CONTROL_DOF_NAMES != (
        "translation_x",
        "translation_y",
        "translation_z",
        "rotation_x",
        "rotation_y",
        "rotation_z",
    ):
        raise ValueError("self-body control-axis order changed")
    if EntityOntologyTokenDim(1) != ENTITY_ONTOLOGY_BASE_DIM + 1:
        raise ValueError("entity ontology token dimension is invalid")

    if DEFAULT_VIRTUAL_SLOT_COUNT != 32:
        raise ValueError("the virtual-content slot budget must remain 32")


_ValidateRobotEntityOntology()


ROBOT_MORPHOLOGY_SCHEMA_VERSION = 3
ROBOT_MORPHOLOGY_PARSER_VERSION = 5
ROBOT_MODEL_CONTRACT_VERSION = 7


@dataclass(frozen=True)
class CompiledRobotMorphology:
    description_id: str
    model_contract_id: str
    adapter_id: str
    robot_name: str
    node_names: Tuple[str, ...]
    joint_names: Tuple[str, ...]
    joint_variable_names: Tuple[str, ...]
    group_names: Tuple[str, ...]
    endpoint_names: Tuple[str, ...]
    gripper_names: Tuple[str, ...]
    sensor_names: Tuple[str, ...]
    sensor_types: Tuple[str, ...]
    node_count: int
    joint_count: int
    joint_dof_count: int
    commandable_joint_dof_count: int
    task_control_coordinate_count: int
    group_count: int
    endpoint_count: int
    gripper_count: int
    sensor_count: int
    parent_index: torch.Tensor
    joint_parent_node: torch.Tensor
    joint_child_node: torch.Tensor
    joint_type: torch.Tensor
    joint_variable_commandable: torch.Tensor
    joint_variable_joint_index: torch.Tensor
    joint_variable_child_node: torch.Tensor
    joint_variable_local_index: torch.Tensor
    joint_lower: torch.Tensor
    joint_upper: torch.Tensor
    joint_effort_limit: torch.Tensor
    joint_velocity_limit: torch.Tensor
    joint_variable_command_delta_scale: torch.Tensor
    joint_variable_unit: Tuple[str, ...]
    joint_variable_command_representation: str
    joint_variable_command_reference: str
    joint_variable_command_range: Tuple[float, float]
    joint_variable_command_limit_policy: str
    group_node_mask: torch.Tensor
    group_joint_mask: torch.Tensor
    node_role: torch.Tensor
    node_side: torch.Tensor
    node_capability: torch.Tensor
    group_role: torch.Tensor
    group_side: torch.Tensor
    group_capability: torch.Tensor
    endpoint_to_node: torch.Tensor
    endpoint_task_mask: torch.Tensor
    endpoint_role: torch.Tensor
    endpoint_side: torch.Tensor
    endpoint_capability: torch.Tensor
    gripper_endpoint_index: torch.Tensor
    sensor_to_node: torch.Tensor
    sensor_role: torch.Tensor
    sensor_side: torch.Tensor
    sensor_capability: torch.Tensor
    observer_valid: bool
    observer_controllable: bool
    observer_attachment_name: str
    observer_frame_name: str
    observer_calibration_id: str
    observer_attachment_kind: str
    observer_attachment_index: int
    observer_node_index: int
    observer_sensor_index: int
    observer_endpoint_index: int
    observer_control_joint_indices: torch.Tensor
    observer_control_group_index: int
    group_dof_count: Tuple[int, ...]
    diagnostics: Tuple[str, ...]
    canonical_json: str

    def ToJson(self) -> Dict[str, Any]:
        return json.loads(self.canonical_json)

    def _NodeDepth(self) -> torch.Tensor:
        depth = torch.full_like(self.parent_index, -1)
        for node_index in range(self.node_count):
            current = node_index
            value = 0
            while int(self.parent_index[current].item()) >= 0:
                current = int(self.parent_index[current].item())
                value += 1
            depth[node_index] = value
        return depth

    @staticmethod
    def _NormalizedFinite(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = torch.isfinite(value)
        normalized = torch.zeros_like(value)
        normalized[valid] = value[valid] / (1.0 + value[valid].abs())
        return normalized, valid

    def JointSemanticDescriptor(self) -> Dict[str, torch.Tensor]:
        count = self.joint_dof_count
        device = self.joint_variable_joint_index.device
        child_node = self.joint_variable_child_node.clone()
        parent_node = self.parent_index[child_node].clone()
        node_depth = self._NodeDepth()
        topology_depth = node_depth[child_node].clone()
        child_role = self.node_role[child_node].clone()
        child_side = self.node_side[child_node].clone()
        child_capability = self.node_capability[child_node].clone()
        parent_role = torch.full_like(child_role, -1)
        parent_side = torch.full_like(child_side, -1)
        parent_capability = torch.zeros_like(child_capability)
        has_parent = parent_node.ge(0)
        if bool(has_parent.any().item()):
            parent_role[has_parent] = self.node_role[parent_node[has_parent]]
            parent_side[has_parent] = self.node_side[parent_node[has_parent]]
            parent_capability[has_parent] = self.node_capability[
                parent_node[has_parent]]
        group_role = torch.zeros(
            count,
            len(BODY_ROLE_NAMES),
            dtype=torch.bool,
            device=device)
        group_side = torch.zeros(
            count,
            len(BODY_SIDE_NAMES),
            dtype=torch.bool,
            device=device)
        group_capability = torch.zeros_like(child_capability)
        for variable_index in range(self.joint_dof_count):
            joint_index = int(
                self.joint_variable_joint_index[variable_index].item())
            groups = torch.nonzero(
                self.group_joint_mask[:, joint_index],
                as_tuple=False).flatten()
            for group_index in groups.tolist():
                group_role[
                    variable_index,
                    int(self.group_role[group_index].item())] = True
                group_side[
                    variable_index,
                    int(self.group_side[group_index].item())] = True
                group_capability[variable_index] |= self.group_capability[
                    group_index]
        source = self.ToJson()
        joints = {item["name"]: item for item in source["urdf"]["joints"]}
        joint_type = self.joint_type[
            self.joint_variable_joint_index].clone()
        joint_axis = torch.zeros(count, 3, dtype=torch.float32, device=device)
        for variable_index in range(self.joint_dof_count):
            joint_index = int(
                self.joint_variable_joint_index[variable_index].item())
            joint_name = self.joint_names[joint_index]
            local_index = int(
                self.joint_variable_local_index[variable_index].item())
            if int(self.joint_parent_node[joint_index].item()) < 0:
                axis = [0.0, 0.0, 0.0]
                if int(joint_type[variable_index].item()) == JOINT_TYPE_NAMES.index(
                    "planar"
                ):
                    axis[local_index if local_index < 2 else 2] = 1.0
                elif int(joint_type[variable_index].item()) == JOINT_TYPE_NAMES.index(
                    "floating"
                ):
                    axis[local_index % 3] = 1.0
                else:
                    raise ValueError("SRDF virtual joint variable type is invalid")
            else:
                joint_kind = joints[joint_name]["type"]
                if joint_kind == "planar":
                    normal = torch.as_tensor(
                        joints[joint_name]["axis"],
                        dtype=torch.float32,
                        device=device)
                    normal = normal / normal.norm()
                    reference = normal.new_tensor([1.0, 0.0, 0.0])
                    if float(normal[0].abs().item()) > 0.9:
                        reference = normal.new_tensor([0.0, 1.0, 0.0])
                    tangent_x = reference - (reference * normal).sum() * normal
                    tangent_x = tangent_x / tangent_x.norm()
                    tangent_y = torch.linalg.cross(normal, tangent_x)
                    axis = (tangent_x, tangent_y, normal)[local_index]
                elif joint_kind == "floating":
                    axis = torch.eye(
                        3,
                        dtype=torch.float32,
                        device=device)[local_index % 3]
                else:
                    axis = joints[joint_name]["axis"]
            axis_tensor = torch.as_tensor(
                axis,
                dtype=torch.float32,
                device=device)
            joint_axis[variable_index] = axis_tensor / axis_tensor.norm()
        lower, position_lower_valid = self._NormalizedFinite(self.joint_lower)
        upper, position_upper_valid = self._NormalizedFinite(self.joint_upper)
        effort, effort_valid = self._NormalizedFinite(self.joint_effort_limit)
        velocity, velocity_valid = self._NormalizedFinite(
            self.joint_velocity_limit)
        return {
            "commandable": self.joint_variable_commandable.clone(),
            "joint_index": self.joint_variable_joint_index.clone(),
            "local_index": self.joint_variable_local_index.clone(),
            "child_node_index": child_node,
            "parent_node_index": parent_node,
            "topology_depth": topology_depth,
            "joint_type": joint_type,
            "joint_axis": joint_axis,
            "child_role": child_role,
            "child_side": child_side,
            "child_capability": child_capability,
            "parent_role": parent_role,
            "parent_side": parent_side,
            "parent_capability": parent_capability,
            "group_role_membership": group_role,
            "group_side_membership": group_side,
            "group_capability": group_capability,
            "lower_limit_normalized": lower,
            "upper_limit_normalized": upper,
            "position_lower_limit_valid": position_lower_valid,
            "position_upper_limit_valid": position_upper_valid,
            "effort_limit_normalized": effort,
            "effort_limit_valid": effort_valid,
            "velocity_limit_normalized": velocity,
            "velocity_limit_valid": velocity_valid,
            "command_delta_scale": (
                self.joint_variable_command_delta_scale.clone()),
        }

    def NodeSemanticDescriptor(self) -> Dict[str, torch.Tensor]:
        device = self.parent_index.device
        parent_node = self.parent_index.clone()
        topology_depth = self._NodeDepth()
        parent_role = torch.full_like(self.node_role, -1)
        parent_side = torch.full_like(self.node_side, -1)
        parent_capability = torch.zeros_like(self.node_capability)
        has_parent = parent_node.ge(0)
        if bool(has_parent.any().item()):
            parent_role[has_parent] = self.node_role[parent_node[has_parent]]
            parent_side[has_parent] = self.node_side[parent_node[has_parent]]
            parent_capability[has_parent] = self.node_capability[
                parent_node[has_parent]]
        in_degree = has_parent.to(dtype=torch.long)
        out_degree = torch.zeros(
            self.node_count,
            dtype=torch.long,
            device=device)
        if bool(has_parent.any().item()):
            out_degree.scatter_add_(
                0,
                parent_node[has_parent],
                torch.ones_like(parent_node[has_parent]))
        group_role = torch.zeros(
            self.node_count,
            len(BODY_ROLE_NAMES),
            dtype=torch.bool,
            device=device)
        group_side = torch.zeros(
            self.node_count,
            len(BODY_SIDE_NAMES),
            dtype=torch.bool,
            device=device)
        group_capability = torch.zeros_like(self.node_capability)
        for node_index in range(self.node_count):
            groups = torch.nonzero(
                self.group_node_mask[:, node_index],
                as_tuple=False).flatten()
            for group_index in groups.tolist():
                group_role[
                    node_index,
                    int(self.group_role[group_index].item())] = True
                group_side[
                    node_index,
                    int(self.group_side[group_index].item())] = True
                group_capability[node_index] |= self.group_capability[
                    group_index]
        return {
            "node_index": torch.arange(
                self.node_count,
                dtype=torch.long,
                device=device),
            "parent_node_index": parent_node,
            "topology_depth": topology_depth,
            "is_root": parent_node.lt(0),
            "is_leaf": out_degree.eq(0),
            "in_degree": in_degree,
            "out_degree": out_degree,
            "role": self.node_role.clone(),
            "side": self.node_side.clone(),
            "capability": self.node_capability.clone(),
            "parent_role": parent_role,
            "parent_side": parent_side,
            "parent_capability": parent_capability,
            "group_role_membership": group_role,
            "group_side_membership": group_side,
            "group_capability": group_capability,
        }

    def EndpointSemanticDescriptor(self) -> Dict[str, torch.Tensor]:
        count = self.endpoint_count
        device = self.endpoint_to_node.device
        node_index = self.endpoint_to_node.clone()
        parent_node = self.parent_index[node_index].clone()
        node_depth = self._NodeDepth()
        topology_depth = node_depth[node_index].clone()
        node_role = self.node_role[node_index].clone()
        node_side = self.node_side[node_index].clone()
        node_capability = self.node_capability[node_index].clone()
        parent_role = torch.full_like(node_role, -1)
        parent_side = torch.full_like(node_side, -1)
        parent_capability = torch.zeros_like(node_capability)
        has_parent = parent_node.ge(0)
        parent_role[has_parent] = self.node_role[parent_node[has_parent]]
        parent_side[has_parent] = self.node_side[parent_node[has_parent]]
        parent_capability[has_parent] = self.node_capability[
            parent_node[has_parent]]
        group_role = torch.zeros(
            count,
            len(BODY_ROLE_NAMES),
            dtype=torch.bool,
            device=device)
        group_side = torch.zeros(
            count,
            len(BODY_SIDE_NAMES),
            dtype=torch.bool,
            device=device)
        group_capability = torch.zeros_like(node_capability)
        for endpoint_index in range(self.endpoint_count):
            endpoint_node = int(node_index[endpoint_index].item())
            groups = torch.nonzero(
                self.group_node_mask[:, endpoint_node],
                as_tuple=False).flatten()
            for group_index in groups.tolist():
                group_role[
                    endpoint_index,
                    int(self.group_role[group_index].item())] = True
                group_side[
                    endpoint_index,
                    int(self.group_side[group_index].item())] = True
                group_capability[endpoint_index] |= self.group_capability[
                    group_index]
        return {
            "controllable": self.endpoint_task_mask.any(dim=-1),
            "node_index": node_index,
            "parent_node_index": parent_node,
            "topology_depth": topology_depth,
            "task_mask": self.endpoint_task_mask.clone(),
            "role": self.endpoint_role.clone(),
            "side": self.endpoint_side.clone(),
            "capability": self.endpoint_capability.clone(),
            "node_role": node_role,
            "node_side": node_side,
            "node_capability": node_capability,
            "parent_role": parent_role,
            "parent_side": parent_side,
            "parent_capability": parent_capability,
            "group_role_membership": group_role,
            "group_side_membership": group_side,
            "group_capability": group_capability,
        }


class RobotMorphologyModule:
    @staticmethod
    def _ReadJson(source: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
        if isinstance(source, Mapping):
            return dict(source)
        value = json.loads(Path(source).read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise TypeError("robot morphology JSON root must be an object")
        return value

    @staticmethod
    def _FiniteVector(value: Optional[str], size: int, default: Sequence[float]) -> List[float]:
        if value is None:
            result = [float(item) for item in default]
        else:
            parts = value.split()
            if len(parts) != size:
                raise ValueError("robot vector has an invalid dimension")
            result = [float(item) for item in parts]
        if any(not math.isfinite(item) for item in result):
            raise ValueError("robot vector must be finite")
        return result

    @staticmethod
    def _Canonical(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): RobotMorphologyModule._Canonical(value[key])
                for key in sorted(value)
            }
        if isinstance(value, (list, tuple)):
            return [RobotMorphologyModule._Canonical(item) for item in value]
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("robot morphology contains a non-finite number")
            if value == 0.0:
                return 0.0
            return float(format(value, ".17g"))
        if value is None or type(value) in (str, int, bool):
            return value
        raise TypeError("robot morphology contains an unsupported value")

    @classmethod
    def _CanonicalJson(cls, value: Dict[str, Any]) -> str:
        return json.dumps(
            cls._Canonical(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False)

    @staticmethod
    def _JointDof(joint_type: str) -> int:
        dof = {
            "fixed": 0,
            "revolute": 1,
            "continuous": 1,
            "prismatic": 1,
            "planar": 3,
            "floating": 6,
        }
        if joint_type not in dof:
            raise ValueError(f"unsupported URDF joint type {joint_type!r}")
        return dof[joint_type]

    @staticmethod
    def _VariableNames(joint: Dict[str, Any]) -> Tuple[str, ...]:
        if joint.get("mimic") is not None:
            return ()
        dof = RobotMorphologyModule._JointDof(joint["type"])
        if dof == 0:
            return ()
        if dof == 1:
            return (joint["name"],)
        suffixes = (
            ("x", "y", "theta")
            if dof == 3
            else ("x", "y", "z", "roll", "pitch", "yaw"))
        return tuple(f"{joint['name']}/{suffix}" for suffix in suffixes)

    @staticmethod
    def _ParseUrdf(path: Union[str, Path]) -> Dict[str, Any]:
        root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
        if root.tag != "robot" or not root.attrib.get("name"):
            raise ValueError("URDF root must be a named robot")
        links = []
        for element in root.findall("link"):
            name = element.attrib.get("name")
            if not name:
                raise ValueError("URDF link must have a name")
            links.append({"name": name})
        joints = []
        for element in root.findall("joint"):
            name = element.attrib.get("name")
            joint_type = element.attrib.get("type")
            parent_element = element.find("parent")
            child_element = element.find("child")
            if (
                not name
                or not joint_type
                or parent_element is None
                or child_element is None
                or not parent_element.attrib.get("link")
                or not child_element.attrib.get("link")
            ):
                raise ValueError("URDF joint is incomplete")
            origin_element = element.find("origin")
            axis_element = element.find("axis")
            limit_element = element.find("limit")
            mimic_element = element.find("mimic")
            limit = None
            if limit_element is not None:
                limit = {
                    key: float(limit_element.attrib[key])
                    for key in ("lower", "upper", "effort", "velocity")
                    if key in limit_element.attrib
                }
                if any(not math.isfinite(value) for value in limit.values()):
                    raise ValueError("URDF joint limit must be finite")
            mimic = None
            if mimic_element is not None:
                target = mimic_element.attrib.get("joint")
                if not target:
                    raise ValueError("URDF mimic joint must name its source")
                mimic = {
                    "joint": target,
                    "multiplier": float(mimic_element.attrib.get("multiplier", "1")),
                    "offset": float(mimic_element.attrib.get("offset", "0")),
                }
                if any(not math.isfinite(value) for value in (
                    mimic["multiplier"], mimic["offset"]
                )):
                    raise ValueError("URDF mimic parameters must be finite")
            joint = {
                "name": name,
                "type": joint_type,
                "parent": parent_element.attrib["link"],
                "child": child_element.attrib["link"],
                "origin_xyz": RobotMorphologyModule._FiniteVector(
                    None if origin_element is None else origin_element.attrib.get("xyz"),
                    3,
                    (0.0, 0.0, 0.0)),
                "origin_rpy": RobotMorphologyModule._FiniteVector(
                    None if origin_element is None else origin_element.attrib.get("rpy"),
                    3,
                    (0.0, 0.0, 0.0)),
                "axis": RobotMorphologyModule._FiniteVector(
                    None if axis_element is None else axis_element.attrib.get("xyz"),
                    3,
                    (1.0, 0.0, 0.0)),
                "limit": limit,
                "mimic": mimic,
            }
            RobotMorphologyModule._JointDof(joint_type)
            joints.append(joint)
        return {
            "name": root.attrib["name"],
            "links": sorted(links, key=lambda item: item["name"]),
            "joints": sorted(joints, key=lambda item: item["name"]),
        }

    @staticmethod
    def _ParseSrdf(path: Union[str, Path]) -> Dict[str, Any]:
        root = ET.fromstring(Path(path).read_text(encoding="utf-8"))
        if root.tag != "robot" or not root.attrib.get("name"):
            raise ValueError("SRDF root must be a named robot")
        groups = []
        for element in root.findall("group"):
            name = element.attrib.get("name")
            if not name:
                raise ValueError("SRDF group must have a name")
            groups.append({
                "name": name,
                "joints": sorted(
                    item.attrib["name"] for item in element.findall("joint")),
                "links": sorted(
                    item.attrib["name"] for item in element.findall("link")),
                "subgroups": sorted(
                    item.attrib["name"] for item in element.findall("group")),
                "chains": sorted([
                    {
                        "base_link": item.attrib["base_link"],
                        "tip_link": item.attrib["tip_link"],
                    }
                    for item in element.findall("chain")
                ], key=lambda item: (item["base_link"], item["tip_link"])),
            })
        passive_joints = sorted(
            item.attrib["name"] for item in root.findall("passive_joint"))
        end_effectors = []
        for item in root.findall("end_effector"):
            required = ("name", "parent_link", "group")
            if any(not item.attrib.get(name) for name in required):
                raise ValueError("SRDF end_effector is incomplete")
            end_effectors.append({
                "name": item.attrib["name"],
                "parent_link": item.attrib["parent_link"],
                "group": item.attrib["group"],
                "parent_group": item.attrib.get("parent_group"),
            })
        virtual_joints = []
        for item in root.findall("virtual_joint"):
            required = ("name", "type", "parent_frame", "child_link")
            if any(not item.attrib.get(name) for name in required):
                raise ValueError("SRDF virtual_joint is incomplete")
            virtual_joints.append({name: item.attrib[name] for name in required})
        group_states = []
        for item in root.findall("group_state"):
            if not item.attrib.get("name") or not item.attrib.get("group"):
                raise ValueError("SRDF group_state is incomplete")
            group_states.append({
                "name": item.attrib["name"],
                "group": item.attrib["group"],
                "joints": sorted([
                    {
                        "name": joint.attrib["name"],
                        "value": [float(value) for value in joint.attrib["value"].split()],
                    }
                    for joint in item.findall("joint")
                ], key=lambda joint: joint["name"]),
            })
            if any(
                not math.isfinite(value)
                for joint in group_states[-1]["joints"]
                for value in joint["value"]
            ):
                raise ValueError("SRDF group_state values must be finite")
        disabled_collisions = sorted([
            {
                "link1": item.attrib["link1"],
                "link2": item.attrib["link2"],
                "reason": item.attrib.get("reason", ""),
            }
            for item in root.findall("disable_collisions")
        ], key=lambda item: (item["link1"], item["link2"], item["reason"]))
        return {
            "name": root.attrib["name"],
            "groups": sorted(groups, key=lambda item: item["name"]),
            "passive_joints": passive_joints,
            "end_effectors": sorted(end_effectors, key=lambda item: item["name"]),
            "virtual_joints": sorted(virtual_joints, key=lambda item: item["name"]),
            "group_states": sorted(group_states, key=lambda item: item["name"]),
            "disabled_collisions": disabled_collisions,
        }

    def FromMoveIt(
        self,
        urdfPath: Union[str, Path],
        srdfPath: Union[str, Path],
        overlay: Optional[Union[str, Path, Mapping[str, Any]]] = None,
    ) -> CompiledRobotMorphology:
        control = {} if overlay is None else self._ReadJson(overlay)
        return self.Compile({
            "schema_version": ROBOT_MORPHOLOGY_SCHEMA_VERSION,
            "urdf": self._ParseUrdf(urdfPath),
            "srdf": self._ParseSrdf(srdfPath),
            "control": control,
        })

    def FromJson(
        self,
        source: Union[str, Path, Mapping[str, Any]],
    ) -> CompiledRobotMorphology:
        return self.Compile(self._ReadJson(source))

    @staticmethod
    def _UniqueNames(items: Iterable[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
        output: Dict[str, Dict[str, Any]] = {}
        for item in items:
            if type(item) is not dict:
                raise TypeError("robot morphology named items must be objects")
            name = item.get(field)
            if type(name) is not str or not name:
                raise ValueError(f"robot morphology {field} must be a non-empty string")
            if name in output:
                raise ValueError(f"duplicate robot morphology name {name!r}")
            output[name] = dict(item)
        return output

    @staticmethod
    def _SemanticValues(
        item: Mapping[str, Any],
        *,
        defaultRole: str,
        defaultCapabilities: Sequence[str] = (),
    ) -> Tuple[str, str, Tuple[str, ...]]:
        role = item.get("role", defaultRole)
        side = item.get("side", "none")
        capabilities = item.get("capabilities", list(defaultCapabilities))
        if role not in BODY_ROLE_NAMES:
            raise ValueError(f"unsupported robot body role {role!r}")
        if side not in BODY_SIDE_NAMES:
            raise ValueError(f"unsupported robot body side {side!r}")
        if (
            type(capabilities) is not list
            or any(
                type(capability) is not str
                or capability not in BODY_CAPABILITY_NAMES
                for capability in capabilities)
            or len(set(capabilities)) != len(capabilities)
        ):
            raise ValueError("robot body capabilities are invalid")
        ordered = tuple(
            capability for capability in BODY_CAPABILITY_NAMES
            if capability in capabilities)
        return role, side, ordered

    @classmethod
    def _SemanticOverlay(
        cls,
        items: Any,
        names: Sequence[str],
        *,
        kind: str,
        defaultRole: str,
    ) -> Dict[str, Tuple[str, str, Tuple[str, ...]]]:
        result = {
            name: cls._SemanticValues({}, defaultRole=defaultRole)
            for name in names}
        if items is None:
            return result
        if type(items) is not list:
            raise TypeError(f"control {kind} semantics must be an array")
        specs = cls._UniqueNames(items, "name")
        unknown = sorted(set(specs) - set(names))
        if unknown:
            raise ValueError(
                f"control {kind} semantics reference unknown names: {unknown}")
        for name, spec in specs.items():
            if set(spec) - {"name", "role", "side", "capabilities"}:
                raise ValueError(f"control {kind} semantic fields are invalid")
            result[name] = cls._SemanticValues(
                spec,
                defaultRole=defaultRole)
        return result

    @staticmethod
    def _TopologicalLinks(
        links: Dict[str, Dict[str, Any]],
        joints: Dict[str, Dict[str, Any]],
    ) -> Tuple[Tuple[str, ...], Dict[str, Optional[str]], Dict[str, str]]:
        parent_of: Dict[str, Optional[str]] = {name: None for name in links}
        parent_joint: Dict[str, str] = {}
        children: Dict[str, List[str]] = {name: [] for name in links}
        for joint in joints.values():
            parent = joint["parent"]
            child = joint["child"]
            if parent not in links or child not in links:
                raise ValueError("URDF joint references an unknown link")
            if parent_of[child] is not None:
                raise ValueError("URDF link has more than one parent joint")
            parent_of[child] = parent
            parent_joint[child] = joint["name"]
            children[parent].append(child)
        roots = sorted(name for name, parent in parent_of.items() if parent is None)
        if len(roots) != 1:
            raise ValueError("URDF must contain exactly one kinematic root")
        ordered: List[str] = []
        queue: List[str] = roots[:]
        while queue:
            node = queue.pop(0)
            ordered.append(node)
            queue.extend(sorted(children[node]))
        if len(ordered) != len(links):
            raise ValueError("URDF kinematic graph contains a cycle")
        return tuple(ordered), parent_of, parent_joint

    @staticmethod
    def _PathLinks(
        base: str,
        tip: str,
        parent_of: Dict[str, Optional[str]],
    ) -> Tuple[str, ...]:
        path: List[str] = []
        current: Optional[str] = tip
        while current is not None and current != base:
            path.append(current)
            current = parent_of[current]
        if current != base:
            raise ValueError("SRDF chain is not a directed kinematic path")
        path.append(base)
        return tuple(reversed(path))

    def _ResolveGroups(
        self,
        group_items: Sequence[Dict[str, Any]],
        links: Dict[str, Dict[str, Any]],
        joints: Dict[str, Dict[str, Any]],
        parent_of: Dict[str, Optional[str]],
        parent_joint: Dict[str, str],
    ) -> Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]:
        groups = self._UniqueNames(group_items, "name")
        resolved: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}
        active: set = set()

        def resolve(name: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
            if name in resolved:
                return resolved[name]
            if name not in groups:
                raise ValueError(f"SRDF references unknown subgroup {name!r}")
            if name in active:
                raise ValueError("SRDF subgroup graph contains a cycle")
            active.add(name)
            group = groups[name]
            for field in ("links", "joints", "subgroups", "chains"):
                if type(group.get(field, [])) is not list:
                    raise TypeError(f"SRDF group {field} must be an array")
            for field in ("links", "joints", "subgroups"):
                values = group.get(field, [])
                if (
                    any(type(value) is not str or not value for value in values)
                    or len(set(values)) != len(values)
                ):
                    raise ValueError(f"SRDF group {field} is invalid")
            chains = group.get("chains", [])
            chain_keys = []
            for chain in chains:
                if (
                    type(chain) is not dict
                    or set(chain) != {"base_link", "tip_link"}
                    or type(chain["base_link"]) is not str
                    or not chain["base_link"]
                    or type(chain["tip_link"]) is not str
                    or not chain["tip_link"]
                ):
                    raise ValueError("SRDF group chain is invalid")
                chain_keys.append((chain["base_link"], chain["tip_link"]))
            if len(set(chain_keys)) != len(chain_keys):
                raise ValueError("duplicate SRDF group chain")
            link_names = set(group.get("links", []))
            joint_names = set(group.get("joints", []))
            for link_name in tuple(link_names):
                if link_name not in links:
                    raise ValueError("SRDF group references an unknown link")
                if link_name in parent_joint:
                    joint_names.add(parent_joint[link_name])
            for joint_name in tuple(joint_names):
                if joint_name not in joints:
                    virtual_joints = {
                        item["name"]: item
                        for item in self._current_srdf.get("virtual_joints", [])}
                    if joint_name in virtual_joints:
                        link_names.add(
                            virtual_joints[joint_name]["child_link"])
                        continue
                    raise ValueError("SRDF group references an unknown joint")
                link_names.add(joints[joint_name]["child"])
            for chain in group.get("chains", []):
                base = chain["base_link"]
                tip = chain["tip_link"]
                if base not in links or tip not in links:
                    raise ValueError("SRDF chain references an unknown link")
                chain_links = self._PathLinks(base, tip, parent_of)
                link_names.update(chain_links)
                joint_names.update(
                    parent_joint[link]
                    for link in chain_links[1:])
            for subgroup in group.get("subgroups", []):
                sub_links, sub_joints = resolve(subgroup)
                link_names.update(sub_links)
                joint_names.update(sub_joints)
            active.remove(name)
            result = (tuple(sorted(link_names)), tuple(sorted(joint_names)))
            resolved[name] = result
            return result

        for group_name in groups:
            resolve(group_name)
        return resolved

    @staticmethod
    def _EndpointLink(
        endpoint: Dict[str, Any],
        resolved_groups: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]],
        parent_of: Dict[str, Optional[str]],
    ) -> Tuple[str, Optional[str]]:
        group_name = endpoint["group"]
        if group_name not in resolved_groups:
            raise ValueError("SRDF end_effector references an unknown group")
        group_links = set(resolved_groups[group_name][0])
        leaves = sorted(
            link for link in group_links
            if not any(parent_of.get(other) == link for other in group_links))
        if len(leaves) == 1:
            return leaves[0], None
        raise ValueError(
            f"SRDF end_effector {endpoint['name']!r} has ambiguous group leaves")

    def Compile(self, source: Mapping[str, Any]) -> CompiledRobotMorphology:
        source = dict(source)
        if source.get("schema_version") != ROBOT_MORPHOLOGY_SCHEMA_VERSION:
            raise ValueError("unsupported robot morphology schema")
        if type(source.get("urdf")) is not dict or type(source.get("srdf")) is not dict:
            raise TypeError("robot morphology requires urdf and srdf objects")
        urdf = dict(source["urdf"])
        srdf = dict(source["srdf"])
        control = dict(source.get("control", {}))
        if set(control) - {
            "nodes",
            "groups",
            "endpoints",
            "sensors",
            "observer",
            "observer_endpoint",
            "observer_frame_name",
            "observer_calibration_id",
            "grippers",
        }:
            raise ValueError("robot control semantics contain unsupported fields")
        if urdf.get("name") != srdf.get("name"):
            raise ValueError("URDF and SRDF robot names do not match")
        robot_name = urdf.get("name")
        if type(robot_name) is not str or not robot_name:
            raise ValueError("robot morphology name must be non-empty")
        for owner, fields in (
            (urdf, ("links", "joints")),
            (srdf, (
                "groups",
                "passive_joints",
                "end_effectors",
                "virtual_joints",
                "group_states",
                "disabled_collisions",
            )),
        ):
            for field in fields:
                if type(owner.get(field, [])) is not list:
                    raise TypeError(f"robot morphology {field} must be an array")
        links = self._UniqueNames(urdf.get("links", []), "name")
        joints = self._UniqueNames(urdf.get("joints", []), "name")
        if not links:
            raise ValueError("robot morphology must contain at least one link")
        for joint in joints.values():
            if not {"name", "type", "parent", "child"}.issubset(joint):
                raise ValueError("URDF joint is incomplete")
            if (
                type(joint["parent"]) is not str
                or not joint["parent"]
                or type(joint["child"]) is not str
                or not joint["child"]
            ):
                raise ValueError("URDF joint links are invalid")
            self._JointDof(joint["type"])
        node_names, parent_of, parent_joint = self._TopologicalLinks(links, joints)
        node_index = {name: index for index, name in enumerate(node_names)}
        node_semantics = self._SemanticOverlay(
            control.get("nodes"),
            node_names,
            kind="node",
            defaultRole="other")
        for joint in joints.values():
            if joint["parent"] not in links or joint["child"] not in links:
                raise ValueError("URDF joint references an unknown link")
            for field in ("origin_xyz", "origin_rpy"):
                value = joint.get(field, [0.0, 0.0, 0.0])
                if (
                    type(value) is not list
                    or len(value) != 3
                    or any(type(item) not in (int, float) for item in value)
                    or any(not math.isfinite(float(item)) for item in value)
                ):
                    raise ValueError(f"URDF joint {field} must contain three finite values")
            joint_type = joint["type"]
            dof = self._JointDof(joint_type)
            axis = joint.get("axis")
            if (
                type(axis) is not list
                or len(axis) != 3
                or any(type(value) not in (int, float) for value in axis)
                or any(not math.isfinite(float(value)) for value in axis)
            ):
                raise ValueError("URDF joint axis must contain three finite values")
            if dof > 0:
                axis_norm = math.sqrt(sum(float(value) ** 2 for value in axis))
                if axis_norm <= 1e-8:
                    raise ValueError("URDF movable joint axis must be non-zero")
                joint["axis"] = [float(value) / axis_norm for value in axis]
            limit = joint.get("limit")
            if type(limit) is dict and set(limit) - {
                "lower", "upper", "effort", "velocity"
            }:
                raise ValueError("URDF joint limit fields are invalid")
            if joint_type in ("revolute", "prismatic"):
                if (
                    type(limit) is not dict
                    or not {"lower", "upper", "effort", "velocity"}.issubset(limit)
                ):
                    raise ValueError(
                        "URDF revolute and prismatic joints require complete limits")
            elif joint_type == "continuous":
                if (
                    type(limit) is not dict
                    or not {"effort", "velocity"}.issubset(limit)
                    or "lower" in limit
                    or "upper" in limit
                ):
                    raise ValueError(
                        "URDF continuous joints require effort and velocity limits")
            elif limit is not None:
                raise ValueError(
                    "URDF fixed, planar, and floating joints cannot declare limits")
            if limit is not None:
                if (
                    type(limit) is not dict
                    or any(type(value) not in (int, float) for value in limit.values())
                    or any(not math.isfinite(float(value)) for value in limit.values())
                ):
                    raise ValueError("URDF joint limits must be finite numbers")
                if (
                    float(limit.get("effort", 0.0)) < 0.0
                    or float(limit.get("velocity", 0.0)) < 0.0
                ):
                    raise ValueError(
                        "URDF joint effort and velocity limits must be non-negative")
                if (
                    joint_type in ("revolute", "prismatic")
                    and float(limit["lower"]) > float(limit["upper"])
                ):
                    raise ValueError("URDF joint lower limit exceeds upper limit")
            if joint.get("mimic") is not None:
                mimic = joint["mimic"]
                if (
                    type(mimic) is not dict
                    or set(mimic) != {"joint", "multiplier", "offset"}
                    or type(mimic.get("multiplier")) not in (int, float)
                    or type(mimic.get("offset")) not in (int, float)
                    or not math.isfinite(float(mimic["multiplier"]))
                    or not math.isfinite(float(mimic["offset"]))
                ):
                    raise ValueError("URDF mimic parameters are invalid")
                target = mimic.get("joint")
                if target not in joints:
                    raise ValueError("URDF mimic references an unknown joint")
                target_type = joints[target]["type"]
                rotational = {"revolute", "continuous"}
                compatible = (
                    joint_type in rotational and target_type in rotational
                ) or (
                    joint_type == "prismatic" and target_type == "prismatic")
                if not compatible:
                    raise ValueError("URDF mimic joint dimension does not match source")
        for joint_name in joints:
            visited = set()
            current = joint_name
            while joints[current].get("mimic") is not None:
                if current in visited:
                    raise ValueError("URDF mimic graph contains a cycle")
                visited.add(current)
                current = joints[current]["mimic"]["joint"]
        virtual_joints = srdf.get("virtual_joints", [])
        virtual_joint_by_name = self._UniqueNames(virtual_joints, "name")
        if set(virtual_joint_by_name) & set(joints):
            raise ValueError("SRDF virtual_joint conflicts with a URDF joint name")
        virtual_children = set()
        for virtual_joint in virtual_joints:
            if set(virtual_joint) != {
                "name", "type", "parent_frame", "child_link"
            }:
                raise ValueError("SRDF virtual_joint fields are invalid")
            child_link = virtual_joint["child_link"]
            if child_link not in links:
                raise ValueError("SRDF virtual_joint child_link is unknown")
            if parent_of[child_link] is not None:
                raise ValueError("SRDF virtual_joint child_link must be the URDF root")
            if child_link in virtual_children:
                raise ValueError("SRDF virtual_joint child_link is duplicated")
            virtual_children.add(child_link)
            if virtual_joint["type"] not in ("fixed", "floating", "planar"):
                raise ValueError("SRDF virtual_joint type is unsupported")
            if (
                type(virtual_joint["parent_frame"]) is not str
                or not virtual_joint["parent_frame"]
            ):
                raise ValueError("SRDF virtual_joint parent_frame is invalid")
            if virtual_joint["parent_frame"] in links:
                raise ValueError("SRDF virtual_joint parent_frame must be external")
        joint_definitions = dict(joints)
        for virtual_joint in virtual_joints:
            joint_definitions[virtual_joint["name"]] = {
                "name": virtual_joint["name"],
                "type": virtual_joint["type"],
                "parent": None,
                "child": virtual_joint["child_link"],
                "origin_xyz": [0.0, 0.0, 0.0],
                "origin_rpy": [0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, 0.0],
                "limit": None,
                "mimic": None,
            }
        self._current_srdf = srdf
        try:
            resolved_groups = self._ResolveGroups(
                srdf.get("groups", []), links, joints, parent_of, parent_joint)
        finally:
            del self._current_srdf
        independent_joints = {
            name
            for name, joint in joints.items()
            if self._JointDof(joint["type"]) > 0
            and joint.get("mimic") is None
        }
        covered_joints = {
            name
            for _, group_joints in resolved_groups.values()
            for name in group_joints
            if name in joints
        }
        uncovered_joints = sorted(independent_joints - covered_joints)
        if uncovered_joints:
            raise ValueError(
                "independent URDF joints are missing from SRDF planning groups: "
                f"{uncovered_joints}")
        end_effectors = self._UniqueNames(srdf.get("end_effectors", []), "name")
        for end_effector in end_effectors.values():
            if not {"name", "parent_link", "group"}.issubset(end_effector):
                raise ValueError("SRDF end_effector is incomplete")
            if end_effector["parent_link"] not in links:
                raise ValueError("SRDF end_effector parent_link is unknown")
            if end_effector["group"] not in resolved_groups:
                raise ValueError("SRDF end_effector group is unknown")
            parent_group = end_effector.get("parent_group")
            if parent_group is not None and parent_group not in resolved_groups:
                raise ValueError("SRDF end_effector parent_group is unknown")
            if (
                parent_group is not None
                and end_effector["parent_link"]
                not in resolved_groups[parent_group][0]
            ):
                raise ValueError(
                    "SRDF end_effector parent_link is outside parent_group")
        for collision in srdf.get("disabled_collisions", []):
            if type(collision) is not dict or not {"link1", "link2"}.issubset(collision):
                raise ValueError("SRDF disabled collision is incomplete")
            if collision["link1"] not in links or collision["link2"] not in links:
                raise ValueError("SRDF disabled collision references an unknown link")
        passive_items = srdf.get("passive_joints", [])
        if any(type(name) is not str or not name for name in passive_items):
            raise ValueError("SRDF passive_joint name is invalid")
        passive = set(passive_items)
        if len(passive) != len(passive_items):
            raise ValueError("duplicate SRDF passive_joint")
        if any(name not in joints for name in passive):
            raise ValueError("SRDF passive_joint references an unknown joint")
        invalid_passive = sorted(passive - independent_joints)
        if invalid_passive:
            raise ValueError(
                "SRDF passive_joint must reference an independent movable joint: "
                f"{invalid_passive}")
        joint_names = tuple(sorted(
            joint_definitions,
            key=lambda name: (
                node_index[joint_definitions[name]["child"]],
                name)))
        variable_names: List[str] = []
        variable_limits: List[Tuple[float, float]] = []
        variable_effort_limits: List[float] = []
        variable_velocity_limits: List[float] = []
        variable_command_delta_scales: List[float] = []
        variable_units: List[str] = []
        variable_commandable: List[bool] = []
        joint_type_indices: List[int] = []
        variable_joint_indices: List[int] = []
        variable_child_nodes: List[int] = []
        variable_local_indices: List[int] = []
        joint_variable_slices: Dict[str, Tuple[int, int]] = {}
        for joint_index, joint_name in enumerate(joint_names):
            joint = joint_definitions[joint_name]
            joint_type_indices.append(JOINT_TYPE_NAMES.index(joint["type"]))
            start = len(variable_names)
            names = self._VariableNames(joint)
            variable_names.extend(names)
            variable_commandable.extend(
                joint_name in joints and joint_name not in passive
                for _ in names)
            dof = len(names)
            limit = joint.get("limit")
            effort_limit = (
                float(limit.get("effort", math.inf))
                if limit is not None else math.inf)
            velocity_limit = (
                float(limit.get("velocity", math.inf))
                if limit is not None else math.inf)
            if effort_limit < 0.0 or velocity_limit < 0.0:
                raise ValueError("URDF joint effort and velocity limits must be non-negative")
            if dof == 1 and joint["type"] != "continuous" and limit is not None:
                lower = float(limit.get("lower", -math.inf))
                upper = float(limit.get("upper", math.inf))
            else:
                lower = -math.inf
                upper = math.inf
            if lower > upper:
                raise ValueError("URDF joint lower limit exceeds upper limit")
            variable_limits.extend((lower, upper) for _ in range(dof))
            variable_effort_limits.extend(effort_limit for _ in range(dof))
            variable_velocity_limits.extend(velocity_limit for _ in range(dof))
            for local_index in range(dof):
                joint_type = joint["type"]
                rotational = (
                    joint_type in ("revolute", "continuous")
                    or joint_type == "planar" and local_index == 2
                    or joint_type == "floating" and local_index >= 3
                )
                variable_units.append("radian" if rotational else "meter")
                if math.isfinite(lower) and math.isfinite(upper):
                    command_scale = upper - lower
                else:
                    command_scale = math.pi if rotational else 1.0
                if not math.isfinite(command_scale) or command_scale <= 0.0:
                    raise ValueError("joint command delta scale must be positive")
                variable_command_delta_scales.append(command_scale)
            variable_joint_indices.extend(joint_index for _ in range(dof))
            variable_child_nodes.extend(
                node_index[joint["child"]] for _ in range(dof))
            variable_local_indices.extend(range(dof))
            joint_variable_slices[joint_name] = (start, len(variable_names))
        if len(set(variable_names)) != len(variable_names):
            raise ValueError("robot joint variable names are not unique")
        group_state_keys = set()
        for state in srdf.get("group_states", []):
            if type(state) is not dict or not {"name", "group", "joints"}.issubset(state):
                raise ValueError("SRDF group_state is incomplete")
            if (
                type(state["name"]) is not str
                or not state["name"]
                or type(state["group"]) is not str
                or not state["group"]
                or type(state["joints"]) is not list
            ):
                raise ValueError("SRDF group_state fields are invalid")
            state_key = (state["group"], state["name"])
            if state_key in group_state_keys:
                raise ValueError("duplicate SRDF group_state")
            group_state_keys.add(state_key)
            if state["group"] not in resolved_groups:
                raise ValueError("SRDF group_state references an unknown group")
            allowed_joints = set(resolved_groups[state["group"]][1])
            state_joints = self._UniqueNames(state["joints"], "name")
            for item in state_joints.values():
                joint_name = item["name"]
                if (
                    joint_name not in joint_definitions
                    or joint_name not in allowed_joints
                ):
                    raise ValueError("SRDF group_state references an invalid joint")
                if (
                    joint_name not in independent_joints
                    and not (
                        joint_name in virtual_joint_by_name
                        and self._JointDof(
                            joint_definitions[joint_name]["type"]) > 0)
                ):
                    raise ValueError(
                        "SRDF group_state must reference an independent movable joint")
                expected_dof = self._JointDof(
                    joint_definitions[joint_name]["type"])
                value = item.get("value")
                if (
                    type(value) is not list
                    or len(value) != expected_dof
                    or any(type(entry) not in (int, float) for entry in value)
                    or any(not math.isfinite(float(entry)) for entry in value)
                ):
                    raise ValueError("SRDF group_state joint value dimension is invalid")
                limit = joint_definitions[joint_name].get("limit")
                if expected_dof == 1 and joint_definitions[joint_name]["type"] != "continuous" and limit is not None:
                    scalar_value = float(value[0])
                    if scalar_value < float(limit["lower"]) or scalar_value > float(limit["upper"]):
                        raise ValueError("SRDF group_state joint value exceeds URDF limits")
        group_names = tuple(sorted(resolved_groups))
        group_semantics = self._SemanticOverlay(
            control.get("groups"),
            group_names,
            kind="group",
            defaultRole="other")
        group_dof_count = tuple(
            sum(
                joint_variable_slices[name][1] - joint_variable_slices[name][0]
                for name in resolved_groups[group_name][1]
                if name in joint_variable_slices)
            for group_name in group_names)
        diagnostics: List[str] = [
            f"virtual_joint {item['name']} {item['type']} state requires external joint runtime fields"
            for item in virtual_joints
            if item["type"] != "fixed"]
        overlay_endpoints = control.get("endpoints")
        endpoint_specs: List[Dict[str, Any]] = []
        endpoint_semantic_overrides: Dict[str, Dict[str, Any]] = {}
        semantic_endpoint_fields = {
            "name",
            "role",
            "side",
            "capabilities",
            "task_mask",
        }
        if overlay_endpoints is None or (
            type(overlay_endpoints) is list
            and all(
                type(item) is dict
                and set(item).issubset(semantic_endpoint_fields)
                for item in overlay_endpoints)
        ):
            for endpoint in srdf.get("end_effectors", []):
                link, diagnostic = self._EndpointLink(
                    endpoint, resolved_groups, parent_of)
                if diagnostic is not None:
                    diagnostics.append(diagnostic)
                endpoint_specs.append({
                    "name": endpoint["name"],
                    "link": link,
                    "task_mask": [False] * len(CONTROL_DOF_NAMES),
                })
            endpoint_specs.sort(key=lambda item: item["name"])
            if overlay_endpoints is not None:
                endpoint_semantic_overrides = self._UniqueNames(
                    overlay_endpoints,
                    "name")
            for endpoint in endpoint_specs:
                override = endpoint_semantic_overrides.get(
                    endpoint["name"], {})
                if "task_mask" in override:
                    endpoint["task_mask"] = override["task_mask"]
                else:
                    diagnostics.append(
                        f"endpoint {endpoint['name']} has no declared task axes and is non-commandable")
        else:
            if type(overlay_endpoints) is not list:
                raise TypeError("control endpoints must be an array")
            endpoint_specs = [dict(item) for item in overlay_endpoints]
        endpoint_by_name = self._UniqueNames(endpoint_specs, "name")
        endpoint_names = tuple(endpoint_by_name)
        unknown_endpoint_semantics = sorted(
            set(endpoint_semantic_overrides) - set(endpoint_names))
        if unknown_endpoint_semantics:
            raise ValueError(
                "control endpoint semantics reference unknown names: "
                f"{unknown_endpoint_semantics}")
        endpoint_links: List[str] = []
        endpoint_masks: List[Tuple[bool, ...]] = []
        endpoint_semantics: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {}
        for endpoint_name in endpoint_names:
            endpoint = endpoint_by_name[endpoint_name]
            if (
                not {"name", "link", "task_mask"}.issubset(endpoint)
                or set(endpoint) - {
                    "name",
                    "link",
                    "task_mask",
                    "role",
                    "side",
                    "capabilities",
                }
            ):
                raise ValueError("control endpoint fields are invalid")
            link = endpoint.get("link")
            if link not in links:
                raise ValueError("control endpoint references an unknown link")
            task_mask = endpoint.get("task_mask")
            if (
                type(task_mask) is not list
                or len(task_mask) != len(CONTROL_DOF_NAMES)
                or any(type(value) is not bool for value in task_mask)
            ):
                raise ValueError("control endpoint task_mask is invalid")
            endpoint_links.append(link)
            endpoint_masks.append(tuple(task_mask))
            endpoint_semantic = dict(endpoint)
            endpoint_semantic.update(
                endpoint_semantic_overrides.get(endpoint_name, {}))
            endpoint_semantics[endpoint_name] = self._SemanticValues(
                endpoint_semantic,
                defaultRole="other")
        sensor_specs = control.get("sensors", [])
        if type(sensor_specs) is not list:
            raise TypeError("control sensors must be an array")
        sensor_by_name = self._UniqueNames(sensor_specs, "name")
        sensor_names = tuple(sensor_by_name)
        sensor_types: List[str] = []
        sensor_links: List[str] = []
        sensor_semantics: Dict[str, Tuple[str, str, Tuple[str, ...]]] = {}
        for sensor_name in sensor_names:
            sensor = sensor_by_name[sensor_name]
            if (
                not {"name", "type", "link"}.issubset(sensor)
                or set(sensor) - {
                    "name",
                    "type",
                    "link",
                    "role",
                    "side",
                    "capabilities",
                }
            ):
                raise ValueError("control sensor fields are invalid")
            sensor_type = sensor.get("type")
            sensor_link = sensor.get("link")
            if type(sensor_type) is not str or not sensor_type:
                raise ValueError("control sensor type must be a non-empty string")
            if sensor_link not in links:
                raise ValueError("control sensor references an unknown link")
            sensor_types.append(sensor_type)
            sensor_links.append(sensor_link)
            sensor_semantics[sensor_name] = self._SemanticValues(
                sensor,
                defaultRole="sensor",
                defaultCapabilities=("observe",))
        if control.get("observer") is not None and control.get("observer_endpoint") is not None:
            raise ValueError("observer and observer_endpoint cannot both be configured")
        observer_spec = control.get("observer")
        if observer_spec is None and control.get("observer_endpoint") is not None:
            observer_spec = {"endpoint": control["observer_endpoint"]}
        if type(observer_spec) is str:
            matches = []
            if observer_spec in sensor_by_name:
                matches.append("sensor")
            if observer_spec in endpoint_by_name:
                matches.append("endpoint")
            if observer_spec in links:
                matches.append("link")
            if len(matches) != 1:
                raise ValueError("observer name must resolve to one attachment")
            observer_spec = {matches[0]: observer_spec}
        observer_control_variables: List[str] = []
        observer_control_group: Optional[str] = None
        if observer_spec is not None:
            if type(observer_spec) is not dict:
                raise ValueError("observer attachment is invalid")
            attachment_keys = tuple(
                key for key in ("sensor", "endpoint", "link")
                if key in observer_spec)
            if (
                len(attachment_keys) != 1
                or set(observer_spec) - {
                    "sensor",
                    "endpoint",
                    "link",
                    "control_joint_variables",
                    "control_group",
                }
            ):
                raise ValueError("observer attachment is invalid")
            observer_control_variables = observer_spec.get(
                "control_joint_variables", [])
            observer_control_group = observer_spec.get("control_group")
            if (
                type(observer_control_variables) is not list
                or any(
                    type(name) is not str or not name
                    for name in observer_control_variables)
                or len(set(observer_control_variables))
                != len(observer_control_variables)
            ):
                raise ValueError(
                    "observer control_joint_variables are invalid")
            if observer_control_group is not None and (
                type(observer_control_group) is not str
                or not observer_control_group
                or observer_control_group not in resolved_groups
            ):
                raise ValueError("observer control_group is invalid")
            observer_spec = {
                attachment_keys[0]: observer_spec[attachment_keys[0]]}
        observer_attachment_kind = "none"
        observer_attachment_index = -1
        observer_node_index = -1
        observer_sensor_index = -1
        observer_endpoint_index = -1
        observer_control_group_index = -1
        observer_control_joint_indices: List[int] = []
        observer_attachment_name: Optional[str] = None
        if observer_spec is not None:
            observer_attachment_kind, observer_name = next(iter(observer_spec.items()))
            if type(observer_name) is not str or not observer_name:
                raise ValueError("observer attachment name must be non-empty")
            observer_attachment_name = observer_name
            if observer_attachment_kind == "sensor":
                if observer_name not in sensor_by_name:
                    raise ValueError("observer references an unknown sensor")
                observer_sensor_index = sensor_names.index(observer_name)
                observer_attachment_index = observer_sensor_index
                observer_node_index = node_index[sensor_links[observer_sensor_index]]
            elif observer_attachment_kind == "endpoint":
                if observer_name not in endpoint_by_name:
                    raise ValueError("observer references an unknown endpoint")
                observer_endpoint_index = endpoint_names.index(observer_name)
                observer_attachment_index = observer_endpoint_index
                observer_node_index = node_index[endpoint_links[observer_endpoint_index]]
            else:
                if observer_name not in links:
                    raise ValueError("observer references an unknown link")
                observer_node_index = node_index[observer_name]
                observer_attachment_index = observer_node_index
            observer_node_name = node_names[observer_node_index]
            if observer_control_group is not None:
                observer_control_group_index = group_names.index(
                    observer_control_group)
                group_joint_names = resolved_groups[
                    observer_control_group][1]
                group_variables = [
                    variable_names[variable_index]
                    for joint_name in group_joint_names
                    if joint_name in joint_variable_slices
                    for variable_index in range(
                        joint_variable_slices[joint_name][0],
                        joint_variable_slices[joint_name][1])
                ]
                if observer_control_variables:
                    if set(observer_control_variables) != set(group_variables):
                        raise ValueError(
                            "observer control variables do not match control_group")
                else:
                    observer_control_variables = group_variables
            variable_index_by_name = {
                name: index for index, name in enumerate(variable_names)}
            unknown_observer_variables = sorted(
                set(observer_control_variables) - set(variable_index_by_name))
            if unknown_observer_variables:
                raise ValueError(
                    "observer controls reference unknown joint variables: "
                    f"{unknown_observer_variables}")
            ancestor_joints = set()
            current_node = observer_node_name
            while parent_of[current_node] is not None:
                ancestor_joints.add(parent_joint[current_node])
                current_node = parent_of[current_node]
            for variable_name in observer_control_variables:
                variable_index = variable_index_by_name[variable_name]
                joint_name = joint_names[
                    variable_joint_indices[variable_index]]
                if joint_definitions[joint_name]["type"] not in (
                    "revolute", "continuous"
                ):
                    raise ValueError(
                        "observer controls must be revolute or continuous")
                if joint_name not in ancestor_joints:
                    raise ValueError(
                        "observer control joint must affect the observer attachment")
                if not variable_commandable[variable_index]:
                    raise ValueError(
                        "observer control joint variable is not commandable")
                observer_control_joint_indices.append(variable_index)
            observer_control_joint_indices.sort()
        observer_valid = observer_spec is not None
        observer_frame_name = control.get("observer_frame_name")
        observer_calibration_id = control.get("observer_calibration_id")
        if observer_valid:
            if type(observer_frame_name) is not str or not observer_frame_name:
                raise ValueError("observer_frame_name must be configured")
            if (
                type(observer_calibration_id) is not str
                or not observer_calibration_id
            ):
                raise ValueError("observer_calibration_id must be configured")
        else:
            if observer_frame_name is not None or observer_calibration_id is not None:
                raise ValueError(
                    "observer frame and calibration require an observer attachment")
            observer_frame_name = ""
            observer_calibration_id = ""
        if (
            observer_attachment_kind == "endpoint"
            and any(endpoint_masks[observer_endpoint_index])
        ):
            raise ValueError(
                "observer endpoint task axes cannot define camera joint control")
        observer_controllable = bool(observer_control_joint_indices)
        gripper_specs = control.get("grippers", [])
        if type(gripper_specs) is not list:
            raise TypeError("control grippers must be an array")
        gripper_by_name = self._UniqueNames(gripper_specs, "name")
        gripper_names = tuple(gripper_by_name)
        gripper_endpoint_indices: List[int] = []
        for name in gripper_names:
            if set(gripper_by_name[name]) != {"name", "endpoint"}:
                raise ValueError("control gripper fields are invalid")
            endpoint_name = gripper_by_name[name].get("endpoint")
            if endpoint_name not in endpoint_by_name:
                raise ValueError("gripper references an unknown endpoint")
            gripper_endpoint_indices.append(endpoint_names.index(endpoint_name))
        canonical = {
            "schema_version": ROBOT_MORPHOLOGY_SCHEMA_VERSION,
            "parser_version": ROBOT_MORPHOLOGY_PARSER_VERSION,
            "urdf": urdf,
            "srdf": srdf,
            "control": control,
            "compiled": {
                "node_names": list(node_names),
                "joint_names": list(joint_names),
                "joint_variable_names": variable_names,
                "joint_variable_commandable": variable_commandable,
                "joint_variable_joint_index": variable_joint_indices,
                "joint_variable_child_node": variable_child_nodes,
                "joint_variable_local_index": variable_local_indices,
                "joint_type": joint_type_indices,
                "joint_variable_command_delta_scale": (
                    variable_command_delta_scales),
                "joint_variable_unit": variable_units,
                "group_names": list(group_names),
                "endpoint_names": list(endpoint_names),
                "endpoint_links": endpoint_links,
                "endpoint_task_mask": [list(mask) for mask in endpoint_masks],
                "node_semantics": node_semantics,
                "group_semantics": group_semantics,
                "endpoint_semantics": endpoint_semantics,
                "sensor_names": list(sensor_names),
                "sensor_types": sensor_types,
                "sensor_links": sensor_links,
                "sensor_semantics": sensor_semantics,
                "observer_attachment": (
                    None
                    if observer_attachment_name is None
                    else {
                        observer_attachment_kind: observer_attachment_name}),
                "observer_frame_name": observer_frame_name,
                "observer_calibration_id": observer_calibration_id,
                "observer_control_joint_indices": (
                    observer_control_joint_indices),
                "observer_control_group": observer_control_group,
                "gripper_names": list(gripper_names),
                "gripper_endpoint_indices": gripper_endpoint_indices,
            },
        }
        canonical_json = self._CanonicalJson(canonical)
        description_id = sha256(canonical_json.encode("utf-8")).hexdigest()
        model_contract_json = self._CanonicalJson({
            "model_contract_version": ROBOT_MODEL_CONTRACT_VERSION,
            "actual_dimensions": {
                "links": len(node_names),
                "joints": len(joint_names),
                "joint_variables": len(variable_names),
                "groups": len(group_names),
                "endpoints": len(endpoint_names),
                "grippers": len(gripper_names),
                "sensors": len(sensor_names),
                "control_axes": len(CONTROL_DOF_NAMES),
            },
            "semantics": {
                "roles": BODY_ROLE_NAMES,
                "sides": BODY_SIDE_NAMES,
                "capabilities": BODY_CAPABILITY_NAMES,
                "control_axes": CONTROL_DOF_NAMES,
                "joint_types": JOINT_TYPE_NAMES,
                "joint_dof_source": (
                    "urdf_joint_type_axis_limits_and_srdf_virtual_joint_type"),
                "virtual_joint_variables": {
                    "fixed": (),
                    "planar": ("x", "y", "theta"),
                    "floating": (
                        "x", "y", "z", "roll", "pitch", "yaw"),
                    "frame": "srdf_parent_frame_to_urdf_root",
                    "rotation_convention": (
                        "fixed_axis_roll_x_pitch_y_yaw_z"),
                    "velocity": "time_derivative_of_position_coordinates",
                    "effort": (
                        "generalized_force_newton_and_torque_newton_meter"),
                    "state_source": (
                        "external_robot_state_only_no_node_pose_inference"),
                    "commandable": False,
                },
                "endpoint_task_axes": (
                    "external_explicit_only_default_false"),
                "endpoint_task_command": {
                    "representation": "local_body_se3_delta",
                    "component_order": CONTROL_DOF_NAMES,
                    "translation_unit": "meter",
                    "rotation_representation": "axis_angle",
                    "rotation_unit": "radian",
                    "reference": (
                        "current_endpoint_pose_at_sensor_frame_exposure"),
                    "composition": (
                        "T_world_target_equals_T_world_endpoint_times_T_endpoint_delta"),
                },
                "observer_control": "urdf_joint_variables_only",
                "joint_variable_command": {
                    "representation": "normalized_position_delta",
                    "range": (-1.0, 1.0),
                    "reference": (
                        "current_measured_position_at_sensor_frame_exposure"),
                    "delta_scale": (
                        "finite_position_span_else_pi_radian_or_one_meter"),
                    "target": "reference_plus_command_times_delta_scale",
                    "limit_policy": (
                        "clamp_finite_limits_wrap_unbounded_rotation"),
                },
                "pose": {
                    "dimension": 7,
                    "translation_axes": ("x", "y", "z"),
                    "translation_unit": "meter",
                    "quaternion_order": "xyzw",
                    "handedness": "right_handed",
                    "world_transform_convention": "T_world_entity",
                },
                "twist": {
                    "dimension": 6,
                    "component_order": (
                        "linear_x",
                        "linear_y",
                        "linear_z",
                        "angular_x",
                        "angular_y",
                        "angular_z"),
                    "frame": "world",
                    "linear_unit": "meter_per_second",
                    "angular_unit": "radian_per_second",
                },
                "time_reference": "sensor_frame_exposure",
                "base_orientation_convention": "q_world_base_xyzw",
                "gravity_convention": (
                    "unit_acceleration_direction_world"),
                "joint_descriptor_version": 2,
                "node_descriptor_version": 1,
                "endpoint_descriptor_version": 1,
                "limit_normalization": (
                    "finite_x_over_one_plus_abs_x_with_valid_mask"),
            },
            "static_tensor_fields": {
                "parent_index": ("L",),
                "joint_parent_node": ("J",),
                "joint_child_node": ("J",),
                "joint_type": ("J",),
                "joint_variable_commandable": ("Q",),
                "joint_variable_joint_index": ("Q",),
                "joint_variable_child_node": ("Q",),
                "joint_variable_local_index": ("Q",),
                "joint_lower": ("Q",),
                "joint_upper": ("Q",),
                "joint_effort_limit": ("Q",),
                "joint_velocity_limit": ("Q",),
                "joint_variable_command_delta_scale": ("Q",),
                "group_node_mask": ("G", "L"),
                "group_joint_mask": ("G", "J"),
                "node_role": ("L",),
                "node_side": ("L",),
                "node_capability": ("L", "capability"),
                "group_role": ("G",),
                "group_side": ("G",),
                "group_capability": ("G", "capability"),
                "endpoint_to_node": ("E",),
                "endpoint_task_mask": ("E", "control_axis"),
                "endpoint_role": ("E",),
                "endpoint_side": ("E",),
                "endpoint_capability": ("E", "capability"),
                "gripper_endpoint_index": ("R",),
                "sensor_to_node": ("S",),
                "sensor_role": ("S",),
                "sensor_side": ("S",),
                "sensor_capability": ("S", "capability"),
            },
            "node_runtime_fields": (
                "node_pose_world",
                "node_twist_world",
                "node_observed",
                "node_healthy",
            ),
            "joint_runtime_fields": (
                "joint_position",
                "joint_velocity",
                "joint_effort",
                "joint_observed",
                "joint_healthy",
                "joint_controllable",
            ),
            "endpoint_runtime_fields": (
                "endpoint_pose",
                "endpoint_observed",
                "endpoint_healthy",
                "endpoint_controllable",
            ),
            "observer_runtime_fields": (
                "observer_pose_world",
                "observer_pose_valid",
            ),
            "observer_motion": {
                "dimension": 7,
                "representation": "relative_se3_translation_xyzw",
                "transform_convention": (
                    "T_previous_observer_current_observer"),
                "translation_frame": "previous_observer",
                "time_pair": (
                    "previous_sensor_exposure_to_current_sensor_exposure"),
                "translation_unit": "meter",
                "quaternion_order": "xyzw",
            },
            "physical_reference_fields": (
                "base_orientation_world",
                "gravity_direction_world",
            ),
            "planner_runtime_fields": (
                "planner_expected_endpoint_pose",
                "planner_progress",
                "planner_tracking_error",
                "planner_executing",
                "planner_reached",
                "planner_failed",
            ),
            "execution_feedback_fields": (
                "model_command_executed",
                "executed_action_id",
            ),
        })
        model_contract_id = sha256(
            model_contract_json.encode("utf-8")).hexdigest()
        def contract_number(value: float) -> Union[float, str]:
            if math.isnan(value):
                raise ValueError("robot contract limit cannot be NaN")
            if value == math.inf:
                return "positive_infinity"
            if value == -math.inf:
                return "negative_infinity"
            return value
        adapter_json = self._CanonicalJson({
            "model_contract_id": model_contract_id,
            "robot_name": robot_name,
            "node_names": node_names,
            "parent_index": [
                -1 if parent_of[name] is None else node_index[parent_of[name]]
                for name in node_names],
            "joint_names": joint_names,
            "joint_types": [
                joint_definitions[name]["type"] for name in joint_names],
            "joint_axes": [
                joint_definitions[name].get("axis", [])
                for name in joint_names],
            "joint_origins_xyz": [
                joint_definitions[name].get(
                    "origin_xyz", [0.0, 0.0, 0.0])
                for name in joint_names],
            "joint_origins_rpy": [
                joint_definitions[name].get(
                    "origin_rpy", [0.0, 0.0, 0.0])
                for name in joint_names],
            "joint_mimic": [
                joint_definitions[name].get("mimic")
                for name in joint_names],
            "joint_variable_names": variable_names,
            "joint_variable_commandable": variable_commandable,
            "joint_variable_joint_index": variable_joint_indices,
            "joint_variable_child_node": variable_child_nodes,
            "joint_variable_local_index": variable_local_indices,
            "joint_variable_limits": [
                [contract_number(lower), contract_number(upper)]
                for lower, upper in variable_limits],
            "joint_variable_effort_limits": [
                contract_number(value) for value in variable_effort_limits],
            "joint_variable_velocity_limits": [
                contract_number(value) for value in variable_velocity_limits],
            "joint_variable_command_delta_scale": (
                variable_command_delta_scales),
            "joint_variable_unit": variable_units,
            "group_names": group_names,
            "group_nodes": {
                name: sorted(resolved_groups[name][0])
                for name in group_names},
            "group_joints": {
                name: sorted(resolved_groups[name][1])
                for name in group_names},
            "virtual_joints": virtual_joints,
            "endpoint_names": endpoint_names,
            "endpoint_links": endpoint_links,
            "endpoint_task_mask": endpoint_masks,
            "node_semantics": node_semantics,
            "group_semantics": group_semantics,
            "endpoint_semantics": endpoint_semantics,
            "sensor_names": sensor_names,
            "sensor_types": sensor_types,
            "sensor_links": sensor_links,
            "sensor_semantics": sensor_semantics,
            "observer_attachment": (
                None
                if observer_attachment_name is None
                else {observer_attachment_kind: observer_attachment_name}),
            "observer_frame_name": observer_frame_name,
            "observer_calibration_id": observer_calibration_id,
            "observer_control_joint_indices": observer_control_joint_indices,
            "observer_control_group": observer_control_group,
            "gripper_names": gripper_names,
            "gripper_endpoint_indices": gripper_endpoint_indices,
        })
        adapter_id = sha256(adapter_json.encode("utf-8")).hexdigest()

        def semantic_tensors(
            names: Sequence[str],
            semantics: Mapping[str, Tuple[str, str, Tuple[str, ...]]],
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            role = torch.full((len(names),), -1, dtype=torch.long)
            side = torch.full((len(names),), -1, dtype=torch.long)
            capability = torch.zeros(
                len(names),
                len(BODY_CAPABILITY_NAMES),
                dtype=torch.bool)
            for index, name in enumerate(names):
                role_name, side_name, capability_names = semantics[name]
                role[index] = BODY_ROLE_NAMES.index(role_name)
                side[index] = BODY_SIDE_NAMES.index(side_name)
                for capability_name in capability_names:
                    capability[
                        index,
                        BODY_CAPABILITY_NAMES.index(capability_name)] = True
            return role, side, capability

        node_role, node_side, node_capability = semantic_tensors(
            node_names,
            node_semantics)
        parent_index = torch.full(
            (len(node_names),), -1, dtype=torch.long)
        for index, name in enumerate(node_names):
            parent = parent_of[name]
            parent_index[index] = -1 if parent is None else node_index[parent]
        joint_parent_node = torch.full(
            (len(joint_names),), -1, dtype=torch.long)
        joint_child_node = torch.full_like(joint_parent_node, -1)
        for index, name in enumerate(joint_names):
            joint = joint_definitions[name]
            joint_parent_node[index] = (
                -1
                if joint["parent"] is None
                else node_index[joint["parent"]])
            joint_child_node[index] = node_index[joint["child"]]
        joint_type = torch.tensor(joint_type_indices, dtype=torch.long)
        joint_variable_commandable = torch.tensor(
            variable_commandable, dtype=torch.bool)
        joint_variable_joint_index = torch.tensor(
            variable_joint_indices, dtype=torch.long)
        joint_variable_child_node = torch.tensor(
            variable_child_nodes, dtype=torch.long)
        joint_variable_local_index = torch.tensor(
            variable_local_indices, dtype=torch.long)
        joint_lower = torch.tensor(
            [value[0] for value in variable_limits], dtype=torch.float32)
        joint_upper = torch.tensor(
            [value[1] for value in variable_limits], dtype=torch.float32)
        joint_effort_limit = torch.tensor(
            variable_effort_limits, dtype=torch.float32)
        joint_velocity_limit = torch.tensor(
            variable_velocity_limits, dtype=torch.float32)
        joint_variable_command_delta_scale = torch.tensor(
            variable_command_delta_scales, dtype=torch.float32)
        observer_control_joint_index = torch.tensor(
            observer_control_joint_indices, dtype=torch.long)
        group_node_mask = torch.zeros(
            len(group_names),
            len(node_names),
            dtype=torch.bool)
        group_joint_mask = torch.zeros(
            len(group_names),
            len(joint_names),
            dtype=torch.bool)
        group_role, group_side, group_capability = semantic_tensors(
            group_names,
            group_semantics)
        joint_index = {name: index for index, name in enumerate(joint_names)}
        for group_index, group_name in enumerate(group_names):
            group_links, group_joints = resolved_groups[group_name]
            for link in group_links:
                group_node_mask[group_index, node_index[link]] = True
            for joint in group_joints:
                if joint in joint_index:
                    group_joint_mask[group_index, joint_index[joint]] = True
        endpoint_to_node = torch.full(
            (len(endpoint_names),), -1, dtype=torch.long)
        endpoint_task_mask = torch.zeros(
            len(endpoint_names),
            len(CONTROL_DOF_NAMES),
            dtype=torch.bool)
        endpoint_role, endpoint_side, endpoint_capability = semantic_tensors(
            endpoint_names,
            endpoint_semantics)
        for index, (link, mask) in enumerate(zip(endpoint_links, endpoint_masks)):
            endpoint_to_node[index] = node_index[link]
            endpoint_task_mask[index] = torch.tensor(mask, dtype=torch.bool)
        gripper_endpoint_index = torch.full(
            (len(gripper_names),), -1, dtype=torch.long)
        for index, endpoint_index in enumerate(gripper_endpoint_indices):
            gripper_endpoint_index[index] = endpoint_index
        sensor_to_node = torch.full(
            (len(sensor_names),), -1, dtype=torch.long)
        for index, link in enumerate(sensor_links):
            sensor_to_node[index] = node_index[link]
        sensor_role, sensor_side, sensor_capability = semantic_tensors(
            sensor_names,
            sensor_semantics)
        return CompiledRobotMorphology(
            description_id=description_id,
            model_contract_id=model_contract_id,
            adapter_id=adapter_id,
            robot_name=robot_name,
            node_names=node_names,
            joint_names=joint_names,
            joint_variable_names=tuple(variable_names),
            group_names=group_names,
            endpoint_names=endpoint_names,
            gripper_names=gripper_names,
            sensor_names=sensor_names,
            sensor_types=tuple(sensor_types),
            node_count=len(node_names),
            joint_count=len(joint_names),
            joint_dof_count=len(variable_names),
            commandable_joint_dof_count=sum(variable_commandable),
            task_control_coordinate_count=sum(
                sum(mask) for mask in endpoint_masks),
            group_count=len(group_names),
            endpoint_count=len(endpoint_names),
            gripper_count=len(gripper_names),
            sensor_count=len(sensor_names),
            parent_index=parent_index,
            joint_parent_node=joint_parent_node,
            joint_child_node=joint_child_node,
            joint_type=joint_type,
            joint_variable_commandable=joint_variable_commandable,
            joint_variable_joint_index=joint_variable_joint_index,
            joint_variable_child_node=joint_variable_child_node,
            joint_variable_local_index=joint_variable_local_index,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
            joint_effort_limit=joint_effort_limit,
            joint_velocity_limit=joint_velocity_limit,
            joint_variable_command_delta_scale=(
                joint_variable_command_delta_scale),
            joint_variable_unit=tuple(variable_units),
            joint_variable_command_representation=(
                "normalized_position_delta"),
            joint_variable_command_reference=(
                "current_measured_position_at_sensor_frame_exposure"),
            joint_variable_command_range=(-1.0, 1.0),
            joint_variable_command_limit_policy=(
                "clamp_finite_limits_wrap_unbounded_rotation"),
            group_node_mask=group_node_mask,
            group_joint_mask=group_joint_mask,
            node_role=node_role,
            node_side=node_side,
            node_capability=node_capability,
            group_role=group_role,
            group_side=group_side,
            group_capability=group_capability,
            endpoint_to_node=endpoint_to_node,
            endpoint_task_mask=endpoint_task_mask,
            endpoint_role=endpoint_role,
            endpoint_side=endpoint_side,
            endpoint_capability=endpoint_capability,
            gripper_endpoint_index=gripper_endpoint_index,
            sensor_to_node=sensor_to_node,
            sensor_role=sensor_role,
            sensor_side=sensor_side,
            sensor_capability=sensor_capability,
            observer_valid=observer_valid,
            observer_controllable=observer_controllable,
            observer_attachment_name=(observer_attachment_name or ""),
            observer_frame_name=observer_frame_name,
            observer_calibration_id=observer_calibration_id,
            observer_attachment_kind=observer_attachment_kind,
            observer_attachment_index=observer_attachment_index,
            observer_node_index=observer_node_index,
            observer_sensor_index=observer_sensor_index,
            observer_endpoint_index=observer_endpoint_index,
            observer_control_joint_indices=observer_control_joint_index,
            observer_control_group_index=observer_control_group_index,
            group_dof_count=group_dof_count,
            diagnostics=tuple(sorted(set(diagnostics))),
            canonical_json=canonical_json)

class TestRobotMorphologyMTool:
    def RunSyntheticTopologies(self) -> Dict[str, bool]:
        module = RobotMorphologyModule()
        minimal = module.FromJson({
            "schema_version": ROBOT_MORPHOLOGY_SCHEMA_VERSION,
            "urdf": {
                "name": "minimal",
                "links": [{"name": "root"}],
                "joints": [],
            },
            "srdf": {
                "name": "minimal",
                "groups": [],
                "passive_joints": [],
                "end_effectors": [],
                "virtual_joints": [],
                "group_states": [],
                "disabled_collisions": [],
            },
            "control": {},
        })
        virtual_morphologies = {}
        for joint_type in ("fixed", "planar", "floating"):
            virtual_source = minimal.ToJson()
            virtual_source["srdf"]["virtual_joints"] = [{
                "name": "world_joint",
                "type": joint_type,
                "parent_frame": "world",
                "child_link": "root",
            }]
            virtual_morphologies[joint_type] = module.FromJson(
                virtual_source)
        fixed_virtual = virtual_morphologies["fixed"]
        planar_virtual = virtual_morphologies["planar"]
        floating_virtual = virtual_morphologies["floating"]
        invalid_virtual_parent = fixed_virtual.ToJson()
        invalid_virtual_parent["srdf"]["virtual_joints"][0][
            "parent_frame"] = "root"
        invalid_virtual_parent_rejected = False
        try:
            module.FromJson(invalid_virtual_parent)
        except ValueError:
            invalid_virtual_parent_rejected = True
        rich = module.FromJson({
            "schema_version": ROBOT_MORPHOLOGY_SCHEMA_VERSION,
            "urdf": {
                "name": "rich",
                "links": [
                    {"name": "root"},
                    {"name": "shoulder_link"},
                    {"name": "camera_link"},
                    {"name": "tool_link"},
                ],
                "joints": [
                    {
                        "name": "shoulder",
                        "type": "revolute",
                        "parent": "root",
                        "child": "shoulder_link",
                        "origin_xyz": [0.0, 0.0, 0.0],
                        "origin_rpy": [0.0, 0.0, 0.0],
                        "axis": [0.0, 0.0, 2.0],
                        "limit": {
                            "lower": -1.0,
                            "upper": 1.0,
                            "effort": 2.0,
                            "velocity": 3.0,
                        },
                        "mimic": None,
                    },
                    {
                        "name": "camera_pan",
                        "type": "revolute",
                        "parent": "shoulder_link",
                        "child": "camera_link",
                        "origin_xyz": [0.0, 0.0, 0.1],
                        "origin_rpy": [0.0, 0.0, 0.0],
                        "axis": [0.0, 1.0, 0.0],
                        "limit": {
                            "lower": -0.5,
                            "upper": 0.5,
                            "effort": 1.0,
                            "velocity": 1.0,
                        },
                        "mimic": None,
                    },
                    {
                        "name": "tool_spin",
                        "type": "continuous",
                        "parent": "shoulder_link",
                        "child": "tool_link",
                        "origin_xyz": [0.0, 0.0, 0.2],
                        "origin_rpy": [0.0, 0.0, 0.0],
                        "axis": [1.0, 0.0, 0.0],
                        "limit": {"effort": 1.0, "velocity": 2.0},
                        "mimic": None,
                    },
                ],
            },
            "srdf": {
                "name": "rich",
                "groups": [
                    {
                        "name": "arm",
                        "joints": ["shoulder", "tool_spin"],
                        "links": [],
                        "subgroups": [],
                        "chains": [],
                    },
                    {
                        "name": "camera",
                        "joints": ["camera_pan"],
                        "links": [],
                        "subgroups": [],
                        "chains": [],
                    },
                ],
                "passive_joints": [],
                "end_effectors": [
                    {
                        "name": "tool_endpoint",
                        "parent_link": "tool_link",
                        "group": "arm",
                        "parent_group": None,
                    },
                    {
                        "name": "camera_endpoint",
                        "parent_link": "camera_link",
                        "group": "camera",
                        "parent_group": None,
                    },
                ],
                "virtual_joints": [],
                "group_states": [{
                    "name": "home",
                    "group": "arm",
                    "joints": [
                        {"name": "shoulder", "value": [0.0]},
                        {"name": "tool_spin", "value": [0.0]},
                    ],
                }],
                "disabled_collisions": [],
            },
            "control": {
                "nodes": [{
                    "name": "camera_link",
                    "role": "sensor",
                    "side": "center",
                    "capabilities": ["observe"],
                }],
                "groups": [{
                    "name": "arm",
                    "role": "arm",
                    "side": "right",
                    "capabilities": ["manipulation"],
                }],
                "endpoints": [
                    {
                        "name": "camera_endpoint",
                        "task_mask": [False, False, False, False, False, False],
                    },
                    {
                        "name": "tool_endpoint",
                        "task_mask": [True, False, False, False, False, True],
                    },
                ],
                "grippers": [{
                    "name": "tool_gripper",
                    "endpoint": "tool_endpoint",
                }],
                "sensors": [
                    {"name": "camera", "type": "rgbd", "link": "camera_link"},
                    {"name": "force", "type": "force", "link": "tool_link"},
                ],
                "observer": {
                    "sensor": "camera",
                    "control_group": "camera",
                },
                "observer_frame_name": "camera_optical",
                "observer_calibration_id": "rich-camera",
            },
        })
        multidof = module.FromJson({
            "schema_version": ROBOT_MORPHOLOGY_SCHEMA_VERSION,
            "urdf": {
                "name": "multidof",
                "links": [
                    {"name": "root"},
                    {"name": "plane"},
                    {"name": "slide"},
                ],
                "joints": [
                    {
                        "name": "planar_base",
                        "type": "planar",
                        "parent": "root",
                        "child": "plane",
                        "origin_xyz": [0.0, 0.0, 0.0],
                        "origin_rpy": [0.0, 0.0, 0.0],
                        "axis": [0.0, 0.0, 1.0],
                        "limit": None,
                        "mimic": None,
                    },
                    {
                        "name": "slide_axis",
                        "type": "prismatic",
                        "parent": "plane",
                        "child": "slide",
                        "origin_xyz": [0.0, 0.0, 0.0],
                        "origin_rpy": [0.0, 0.0, 0.0],
                        "axis": [1.0, 0.0, 0.0],
                        "limit": {
                            "lower": 0.0,
                            "upper": 0.5,
                            "effort": 4.0,
                            "velocity": 0.5,
                        },
                        "mimic": None,
                    },
                ],
            },
            "srdf": {
                "name": "multidof",
                "groups": [{
                    "name": "body",
                    "joints": ["planar_base", "slide_axis"],
                    "links": [],
                    "subgroups": [],
                    "chains": [],
                }],
                "passive_joints": [],
                "end_effectors": [],
                "virtual_joints": [],
                "group_states": [{
                    "name": "home",
                    "group": "body",
                    "joints": [
                        {"name": "planar_base", "value": [0.0, 0.0, 0.0]},
                        {"name": "slide_axis", "value": [0.0]},
                    ],
                }],
                "disabled_collisions": [],
            },
            "control": {},
        })
        invalid_observer_control_source = rich.ToJson()
        for joint in invalid_observer_control_source["urdf"]["joints"]:
            if joint["name"] == "camera_pan":
                joint["type"] = "prismatic"
        invalid_observer_control_rejected = False
        try:
            module.FromJson(invalid_observer_control_source)
        except ValueError:
            invalid_observer_control_rejected = True
        carrier_motion_source = rich.ToJson()
        for joint in carrier_motion_source["urdf"]["joints"]:
            if joint["name"] == "camera_pan":
                joint["type"] = "prismatic"
        carrier_motion_source["control"]["observer"].pop("control_group")
        carrier_motion = module.FromJson(carrier_motion_source)
        mimic_source = rich.ToJson()
        mimic_source["urdf"]["links"].append({"name": "mirror_link"})
        mimic_source["urdf"]["joints"].append({
            "name": "mirror_joint",
            "type": "revolute",
            "parent": "root",
            "child": "mirror_link",
            "origin_xyz": [0.0, 0.0, 0.0],
            "origin_rpy": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "limit": {
                "lower": -1.0,
                "upper": 1.0,
                "effort": 1.0,
                "velocity": 1.0,
            },
            "mimic": {
                "joint": "shoulder",
                "multiplier": 1.0,
                "offset": 0.0,
            },
        })
        mimic_one = module.FromJson(mimic_source)
        mimic_source["urdf"]["joints"][-1]["mimic"]["multiplier"] = -1.0
        mimic_two = module.FromJson(mimic_source)
        invalid_virtual_state = planar_virtual.ToJson()
        invalid_virtual_state["srdf"]["groups"] = [{
            "name": "root",
            "joints": ["world_joint"],
            "links": [],
            "subgroups": [],
            "chains": [],
        }]
        invalid_virtual_state["srdf"]["group_states"] = [{
            "name": "home",
            "group": "root",
            "joints": [{"name": "world_joint", "value": [0.0, 0.0]}],
        }]
        invalid_virtual_state_rejected = False
        try:
            module.FromJson(invalid_virtual_state)
        except ValueError:
            invalid_virtual_state_rejected = True
        rich_node_descriptor = rich.NodeSemanticDescriptor()
        return {
            "minimal_actual_dimensions": (
                (minimal.node_count, minimal.joint_count, minimal.joint_dof_count)
                == (1, 0, 0)
                and (minimal.group_count, minimal.endpoint_count)
                == (0, 0)
                and (minimal.gripper_count, minimal.sensor_count) == (0, 0)),
            "fixed_virtual_actual_dimensions": (
                fixed_virtual.node_count == 1
                and fixed_virtual.joint_count == 1
                and fixed_virtual.joint_dof_count == 0
                and tuple(fixed_virtual.joint_type.shape) == (1,)
                and tuple(fixed_virtual.joint_parent_node.tolist()) == (-1,)
                and tuple(fixed_virtual.joint_child_node.tolist()) == (0,)),
            "planar_virtual_actual_dimensions": (
                planar_virtual.node_count == 1
                and planar_virtual.joint_count == 1
                and planar_virtual.joint_dof_count == 3
                and planar_virtual.joint_variable_names == (
                    "world_joint/x",
                    "world_joint/y",
                    "world_joint/theta")
                and torch.equal(
                    planar_virtual.joint_variable_local_index,
                    torch.tensor([0, 1, 2]))
                and bool((~planar_virtual.joint_variable_commandable).all().item())
                and bool((planar_virtual.JointSemanticDescriptor()[
                    "joint_type"] == JOINT_TYPE_NAMES.index("planar")).all().item())
                and torch.equal(
                    planar_virtual.JointSemanticDescriptor()["joint_axis"],
                    torch.eye(3))),
            "floating_virtual_actual_dimensions": (
                floating_virtual.node_count == 1
                and floating_virtual.joint_count == 1
                and floating_virtual.joint_dof_count == 6
                and floating_virtual.joint_variable_names == (
                    "world_joint/x",
                    "world_joint/y",
                    "world_joint/z",
                    "world_joint/roll",
                    "world_joint/pitch",
                    "world_joint/yaw")
                and torch.equal(
                    floating_virtual.joint_variable_local_index,
                    torch.arange(6))
                and bool((~floating_virtual.joint_variable_commandable).all().item())
                and bool((floating_virtual.JointSemanticDescriptor()[
                    "joint_type"] == JOINT_TYPE_NAMES.index("floating")).all().item())
                and torch.equal(
                    floating_virtual.JointSemanticDescriptor()["joint_axis"],
                    torch.cat((torch.eye(3), torch.eye(3)), dim=0))),
            "virtual_dimension_model_isolation": len({
                minimal.model_contract_id,
                fixed_virtual.model_contract_id,
                planar_virtual.model_contract_id,
                floating_virtual.model_contract_id,
            }) == 4,
            "virtual_state_dimension_strict": (
                invalid_virtual_state_rejected
                and invalid_virtual_parent_rejected
                and any(
                    "external joint runtime fields" in item
                    for item in floating_virtual.diagnostics)),
            "rich_actual_dimensions": (
                (rich.node_count, rich.joint_count, rich.joint_dof_count)
                == (4, 3, 3)
                and (rich.group_count, rich.endpoint_count) == (2, 2)
                and (rich.gripper_count, rich.sensor_count) == (1, 2)),
            "multidof_actual_dimensions": (
                (multidof.node_count, multidof.joint_count, multidof.joint_dof_count)
                == (3, 2, 4)
                and (multidof.group_count, multidof.endpoint_count) == (1, 0)),
            "multidof_axis_basis": torch.equal(
                multidof.JointSemanticDescriptor()["joint_axis"],
                torch.cat((torch.eye(3), torch.tensor([[1.0, 0.0, 0.0]])))),
            "actual_dimension_model_isolation": len({
                minimal.model_contract_id,
                rich.model_contract_id,
                multidof.model_contract_id,
            }) == 3,
            "external_semantic_control": (
                rich.task_control_coordinate_count == 2
                and rich.commandable_joint_dof_count == 3
                and rich.observer_valid
                and rich.observer_controllable
                and rich.observer_control_joint_indices.numel() == 1
                and tuple(rich.endpoint_task_mask.shape) == (2, 6)),
            "axis_and_node_descriptor": (
                torch.allclose(
                    rich.JointSemanticDescriptor()["joint_axis"][0].norm(),
                    torch.tensor(1.0))
                and tuple(rich_node_descriptor["role"].shape) == (4,)
                and int(rich_node_descriptor["is_root"].sum().item()) == 1),
            "observer_control_vs_carrier_motion": (
                invalid_observer_control_rejected
                and carrier_motion.observer_valid
                and not carrier_motion.observer_controllable
                and carrier_motion.observer_control_joint_indices.numel() == 0),
            "mimic_adapter_identity": (
                mimic_one.model_contract_id == mimic_two.model_contract_id
                and mimic_one.adapter_id != mimic_two.adapter_id
                and mimic_one.description_id != mimic_two.description_id),
        }

    def RunCurrentMoveIt(self, projectRoot: Union[str, Path]) -> Dict[str, bool]:
        module = RobotMorphologyModule()
        root = Path(projectRoot)
        compiled = module.FromMoveIt(
            root / "Configure/Arm_R_SLDASM.urdf",
            root / "Configure/Arm_R_SLDASM.srdf")
        roundtrip = module.FromJson(compiled.ToJson())
        runtime_joint_name = compiled.joint_names[int(
            compiled.joint_variable_joint_index[0].item())]
        passive_source = compiled.ToJson()
        passive_source["srdf"]["passive_joints"] = [runtime_joint_name]
        passive = module.FromJson(passive_source)
        planar_root_source = compiled.ToJson()
        planar_root_source["srdf"]["virtual_joints"][0]["type"] = "planar"
        planar_root = module.FromJson(planar_root_source)
        floating_root_source = compiled.ToJson()
        floating_root_source["srdf"]["virtual_joints"][0]["type"] = "floating"
        floating_root = module.FromJson(floating_root_source)
        continuous_source = compiled.ToJson()
        for joint in continuous_source["urdf"]["joints"]:
            if joint["name"] == runtime_joint_name:
                joint["type"] = "continuous"
                joint["limit"].pop("lower", None)
                joint["limit"].pop("upper", None)
                break
        continuous = module.FromJson(continuous_source)
        limit_source = compiled.ToJson()
        for joint in limit_source["urdf"]["joints"]:
            if joint["name"] == runtime_joint_name:
                joint["limit"]["velocity"] += 0.5
                break
        changed_limit = module.FromJson(limit_source)
        group_source = compiled.ToJson()
        group_source["srdf"]["groups"][0]["links"].extend((
            "base_link",
            "WristArth_Link",
            "Palm_Link",
        ))
        changed_group = module.FromJson(group_source)
        multidof_source = compiled.ToJson()
        multidof_joint_name = runtime_joint_name
        for joint in multidof_source["urdf"]["joints"]:
            if joint["name"] == multidof_joint_name:
                joint["type"] = "planar"
                joint["limit"] = None
                break
        for state in multidof_source["srdf"]["group_states"]:
            for joint in state["joints"]:
                if joint["name"] == multidof_joint_name:
                    joint["value"] = [joint["value"][0], 0.0, 0.0]
        multidof = module.FromJson(multidof_source)
        observer_name = compiled.endpoint_names[3]
        future_observer = module.FromMoveIt(
            root / "Configure/Arm_R_SLDASM.urdf",
            root / "Configure/Arm_R_SLDASM.srdf",
            {
                "observer_endpoint": observer_name,
                "observer_frame_name": "camera_optical",
                "observer_calibration_id": "test-camera",
            })
        controlled_endpoint = module.FromMoveIt(
            root / "Configure/Arm_R_SLDASM.urdf",
            root / "Configure/Arm_R_SLDASM.srdf",
            {
                "endpoints": [{
                    "name": observer_name,
                    "task_mask": [False, False, False, True, True, True],
                }],
            })
        controlled_endpoint_index = controlled_endpoint.endpoint_names.index(
            observer_name)
        observer_task_axes_rejected = False
        try:
            module.FromMoveIt(
                root / "Configure/Arm_R_SLDASM.urdf",
                root / "Configure/Arm_R_SLDASM.srdf",
                {
                    "endpoints": [{
                        "name": observer_name,
                        "task_mask": [False, False, False, True, True, True],
                    }],
                    "observer_endpoint": observer_name,
                    "observer_frame_name": "camera_optical",
                    "observer_calibration_id": "test-camera",
                })
        except ValueError:
            observer_task_axes_rejected = True
        semantic = module.FromMoveIt(
            root / "Configure/Arm_R_SLDASM.urdf",
            root / "Configure/Arm_R_SLDASM.srdf",
            {
                "nodes": [{
                    "name": "base_link",
                    "role": "root",
                    "side": "center",
                    "capabilities": ["balance"],
                }],
                "groups": [{
                    "name": "r_arm",
                    "role": "arm",
                    "side": "right",
                    "capabilities": ["manipulation"],
                }],
                "endpoints": [{
                    "name": "wrist_end",
                    "role": "hand",
                    "side": "right",
                    "capabilities": ["manipulation", "grasp"],
                }],
                "sensors": [{
                    "name": "head_camera",
                    "type": "rgbd",
                    "link": "base_link",
                }],
                "observer": {"sensor": "head_camera"},
                "observer_frame_name": "camera_optical",
                "observer_calibration_id": "test-camera",
            })
        semantic_roundtrip = module.FromJson(semantic.ToJson())
        joint_descriptor = semantic.JointSemanticDescriptor()
        node_descriptor = semantic.NodeSemanticDescriptor()
        endpoint_descriptor = semantic.EndpointSemanticDescriptor()
        passive_joint_descriptor = passive.JointSemanticDescriptor()
        base_index = semantic.node_names.index("base_link")
        arm_index = semantic.group_names.index("r_arm")
        wrist_index = semantic.endpoint_names.index("wrist_end")
        return {
            "links_28": compiled.node_count == 28,
            "joints_28": compiled.joint_count == 28,
            "joint_dof_22": compiled.joint_dof_count == 22,
            "commandable_joint_dof_22": (
                compiled.commandable_joint_dof_count == 22),
            "task_coordinates_default_zero": (
                compiled.task_control_coordinate_count == 0),
            "groups_12": compiled.group_count == 12,
            "endpoints_11": compiled.endpoint_count == 11,
            "observer_absent": (
                not compiled.observer_valid
                and compiled.observer_endpoint_index == -1),
            "external_observer_seam": (
                future_observer.observer_valid
                and not future_observer.observer_controllable
                and future_observer.observer_attachment_name == observer_name
                and future_observer.observer_frame_name == "camera_optical"
                and future_observer.observer_calibration_id == "test-camera"
                and future_observer.observer_endpoint_index == 3
                and future_observer.description_id != compiled.description_id),
            "external_task_axes_only": (
                not controlled_endpoint.observer_valid
                and controlled_endpoint.task_control_coordinate_count == 3
                and not bool(controlled_endpoint.endpoint_task_mask[
                    controlled_endpoint_index, :3].any().item())
                and bool(controlled_endpoint.endpoint_task_mask[
                    controlled_endpoint_index, 3:].all().item())
                and observer_task_axes_rejected),
            "adapter_execution_semantics": (
                passive.model_contract_id == compiled.model_contract_id
                and passive.adapter_id != compiled.adapter_id
                and passive.description_id != compiled.description_id
                and passive.commandable_joint_dof_count
                < compiled.commandable_joint_dof_count),
            "adapter_limits_and_groups": (
                changed_limit.model_contract_id == compiled.model_contract_id
                and changed_limit.adapter_id != compiled.adapter_id
                and changed_group.model_contract_id == compiled.model_contract_id
                and changed_group.adapter_id != compiled.adapter_id),
            "adapter_multidof_layout": (
                multidof.model_contract_id != compiled.model_contract_id
                and multidof.adapter_id != compiled.adapter_id
                and multidof.joint_dof_count == compiled.joint_dof_count + 2
                and torch.equal(
                    multidof.joint_variable_local_index[:3],
                    torch.tensor([0, 1, 2]))),
            "external_virtual_root_seam": (
                planar_root.model_contract_id != compiled.model_contract_id
                and floating_root.model_contract_id != compiled.model_contract_id
                and floating_root.model_contract_id
                != planar_root.model_contract_id
                and planar_root.adapter_id != compiled.adapter_id
                and floating_root.adapter_id != compiled.adapter_id
                and planar_root.joint_count == compiled.joint_count
                and floating_root.joint_count == compiled.joint_count
                and planar_root.joint_dof_count
                == compiled.joint_dof_count + 3
                and floating_root.joint_dof_count
                == compiled.joint_dof_count + 6
                and not bool(planar_root.joint_variable_commandable[:3].any().item())
                and not bool(floating_root.joint_variable_commandable[:6].any().item())
                and any(
                    "external joint runtime fields" in item
                    for item in floating_root.diagnostics)),
            "unbounded_limit_adapter_identity": (
                continuous.model_contract_id == compiled.model_contract_id
                and continuous.adapter_id != compiled.adapter_id
                and not bool(continuous.JointSemanticDescriptor()[
                    "position_lower_limit_valid"][0].item())
                and not bool(continuous.JointSemanticDescriptor()[
                    "position_upper_limit_valid"][0].item())),
            "undeclared_endpoint_axes_masked": (
                not bool(compiled.endpoint_task_mask[:11].any().item())
                and sum(
                    "has no declared task axes" in item
                    for item in compiled.diagnostics) == 11),
            "json_roundtrip": (
                roundtrip.description_id == compiled.description_id
                and torch.equal(
                    roundtrip.endpoint_task_mask,
                    compiled.endpoint_task_mask)),
            "actual_dimension_contract_isolation": (
                semantic.model_contract_id != compiled.model_contract_id
                and semantic.adapter_id != compiled.adapter_id
                and semantic.description_id != compiled.description_id),
            "semantic_overlay": (
                int(semantic.node_role[base_index].item())
                == BODY_ROLE_NAMES.index("root")
                and int(semantic.node_side[base_index].item())
                == BODY_SIDE_NAMES.index("center")
                and bool(semantic.node_capability[
                    base_index,
                    BODY_CAPABILITY_NAMES.index("balance")].item())
                and int(semantic.group_role[arm_index].item())
                == BODY_ROLE_NAMES.index("arm")
                and int(semantic.endpoint_role[wrist_index].item())
                == BODY_ROLE_NAMES.index("hand")),
            "sensor_observer_attachment": (
                semantic.sensor_count == 1
                and semantic.observer_valid
                and not semantic.observer_controllable
                and semantic.observer_attachment_kind == "sensor"
                and semantic.observer_attachment_name == "head_camera"
                and semantic.observer_frame_name == "camera_optical"
                and semantic.observer_calibration_id == "test-camera"
                and semantic.observer_sensor_index == 0
                and semantic.observer_endpoint_index == -1
                and semantic.observer_node_index == base_index),
            "joint_runtime_layout": (
                bool((semantic.joint_variable_joint_index >= 0).all().item())
                and bool((semantic.joint_variable_child_node >= 0).all().item())
                and bool((semantic.joint_variable_local_index >= 0).all().item())),
            "shared_semantic_descriptors": (
                tuple(node_descriptor["capability"].shape)
                == (semantic.node_count, len(BODY_CAPABILITY_NAMES))
                and int(node_descriptor["is_root"].sum().item()) == 1
                and torch.equal(
                    node_descriptor["is_leaf"],
                    node_descriptor["out_degree"].eq(0))
                and bool((node_descriptor["topology_depth"] >= 0).all().item())
                and tuple(node_descriptor["group_capability"].shape)
                == (semantic.node_count, len(BODY_CAPABILITY_NAMES))
                and
                tuple(joint_descriptor["joint_axis"].shape)
                == (semantic.joint_dof_count, 3)
                and tuple(joint_descriptor["group_capability"].shape)
                == (
                    semantic.joint_dof_count,
                    len(BODY_CAPABILITY_NAMES))
                and bool((joint_descriptor[
                    "topology_depth"] >= 0).all().item())
                and torch.equal(
                    passive_joint_descriptor["commandable"],
                    passive.joint_variable_commandable)
                and tuple(endpoint_descriptor["task_mask"].shape)
                == (
                    semantic.endpoint_count,
                    len(CONTROL_DOF_NAMES))
                and torch.equal(
                    endpoint_descriptor["task_mask"],
                    semantic.endpoint_task_mask)),
            "semantic_json_roundtrip": (
                semantic_roundtrip.description_id == semantic.description_id
                and semantic_roundtrip.adapter_id == semantic.adapter_id
                and semantic_roundtrip.model_contract_id
                == semantic.model_contract_id
                and semantic_roundtrip.observer_attachment_name
                == semantic.observer_attachment_name
                and semantic_roundtrip.observer_frame_name
                == semantic.observer_frame_name
                and semantic_roundtrip.observer_calibration_id
                == semantic.observer_calibration_id),
        }
