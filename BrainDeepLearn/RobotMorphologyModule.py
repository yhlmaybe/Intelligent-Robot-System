from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, Union
import json
import math

import torch


ROBOT_EMBODIMENT_SCHEMA_VERSION = 13
FEEDBACK_TIMESTAMP_UNIT = "s"
FEEDBACK_TIMESTAMP_REFERENCE = "monotonic_relative"


class ModelSignatureCompiler:
    @staticmethod
    def CanonicalJson(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False)

    @classmethod
    def Compile(cls, payload: Mapping[str, Any]) -> str:
        if not isinstance(payload, Mapping):
            raise TypeError("model signature payload must be a mapping")
        canonical = cls.CanonicalJson(dict(payload)).encode("utf-8")
        return sha256(canonical).hexdigest()

    @classmethod
    def CompileBrainBuild(
        cls,
        cognitivePayload: Mapping[str, Any],
        contractView: Any,
        schemaVersion: int,
    ) -> str:
        if not isinstance(cognitivePayload, Mapping) or not cognitivePayload:
            raise TypeError("cognitive signature payload must be non-empty")
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("brain signature requires a contract view")
        if type(schemaVersion) is not int or schemaVersion < 1:
            raise ValueError("brain build schema version must be positive")
        return cls.Compile({
            "schema_version": schemaVersion,
            "cognitive": dict(cognitivePayload),
            "contract": {
                "schema_version": contractView.schema_version,
                "contract_id": contractView.contract_id,
                "model_shape_id": contractView.model_shape_id,
                "model_signature": contractView.model_signature,
            },
            "embodiment": {
                name: getattr(contractView.model_shape, name)
                for name in contractView.model_shape.__dataclass_fields__
            },
        })


class JointType(IntEnum):
    FIXED = 0
    REVOLUTE = 1
    CONTINUOUS = 2
    PRISMATIC = 3


class EndEffectorType(IntEnum):
    WRIST = 0
    FINGERTIP = 1
    TOOL = 2
    SENSOR_ACTUATOR = 3
    OTHER = 4


@dataclass(frozen=True)
class PackedLayout:
    offsets: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.offsets or self.offsets[0] != 0:
            raise ValueError("packed offsets must start at zero")
        if any(type(value) is not int for value in self.offsets):
            raise TypeError("packed offsets must be integers")
        if any(
            right < left
            for left, right in zip(self.offsets[:-1], self.offsets[1:])
        ):
            raise ValueError("packed offsets must be monotonic")

    @property
    def SlotCount(self) -> int:
        return len(self.offsets) - 1

    @property
    def PackedDim(self) -> int:
        return self.offsets[-1]

    def Width(self, slotIndex: int) -> int:
        index = int(slotIndex)
        if index < 0 or index >= self.SlotCount:
            raise IndexError("packed slot index is out of range")
        return self.offsets[index + 1] - self.offsets[index]

    def Slice(self, slotIndex: int) -> slice:
        index = int(slotIndex)
        self.Width(index)
        return slice(self.offsets[index], self.offsets[index + 1])

    @classmethod
    def FromWidths(cls, widths: Sequence[int]) -> "PackedLayout":
        offsets = [0]
        for width in widths:
            if type(width) is not int or width < 0:
                raise ValueError("packed widths must be non-negative integers")
            offsets.append(offsets[-1] + width)
        return cls(tuple(offsets))


@dataclass(frozen=True)
class PackedView:
    source_slot_count: int
    indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.source_slot_count) is not int or self.source_slot_count < 0:
            raise ValueError("packed view source count must be non-negative")
        if any(type(index) is not int for index in self.indices):
            raise TypeError("packed view indices must be integers")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("packed view indices must be unique")
        if any(
            index < 0 or index >= self.source_slot_count
            for index in self.indices
        ):
            raise ValueError("packed view index is out of range")

@dataclass(frozen=True)
class PackedTensor:
    values: Tuple[float, ...]
    offsets: Tuple[int, ...]
    shapes: Tuple[Tuple[int, int], ...]

    def __post_init__(self) -> None:
        layout = PackedLayout(self.offsets)
        if layout.SlotCount != len(self.shapes):
            raise ValueError("packed tensor shape count is inconsistent")
        if layout.PackedDim != len(self.values):
            raise ValueError("packed tensor value count is inconsistent")
        if any(not math.isfinite(float(value)) for value in self.values):
            raise ValueError("packed tensor values must be finite")
        for index, shape in enumerate(self.shapes):
            if (
                len(shape) != 2
                or any(type(value) is not int or value < 0 for value in shape)
                or shape[0] * shape[1] != layout.Width(index)
            ):
                raise ValueError("packed tensor matrix shape is invalid")

    def Matrix(
        self,
        slotIndex: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        index = int(slotIndex)
        if index < 0 or index >= len(self.shapes):
            raise IndexError("packed tensor slot index is out of range")
        start = self.offsets[index]
        end = self.offsets[index + 1]
        return torch.tensor(
            self.values[start:end],
            device=device,
            dtype=dtype).reshape(self.shapes[index])

    @classmethod
    def FromMatrices(
        cls,
        matrices: Sequence[Sequence[Sequence[float]]],
    ) -> "PackedTensor":
        values = []
        widths = []
        shapes = []
        for matrix in matrices:
            rows = tuple(tuple(float(value) for value in row) for row in matrix)
            if not rows or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("packed matrices must be non-empty rectangular matrices")
            flat = tuple(value for row in rows for value in row)
            if any(not math.isfinite(value) for value in flat):
                raise ValueError("packed matrices must be finite")
            values.extend(flat)
            widths.append(len(flat))
            shapes.append((len(rows), len(rows[0])))
        layout = PackedLayout.FromWidths(widths)
        return cls(tuple(values), layout.offsets, tuple(shapes))


@dataclass(frozen=True)
class EmbodimentShape:
    joint_token_count: int
    joint_static_descriptor_dim: int
    joint_feedback_packed_dim: int
    end_effector_token_count: int
    end_effector_static_descriptor_dim: int
    end_effector_target_packed_dim: int
    hierarchy_edge_count: int
    perception_view_dim: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"embodiment shape {name} must be non-negative")
        if self.joint_token_count < 1:
            raise ValueError("embodiment must contain joint coordinates")
        if self.end_effector_token_count < 1:
            raise ValueError("embodiment must contain end effectors")
        if self.joint_static_descriptor_dim < 1:
            raise ValueError("joint static descriptors must be non-empty")
        if self.end_effector_static_descriptor_dim < 1:
            raise ValueError("end-effector static descriptors must be non-empty")
        if self.joint_feedback_packed_dim < 1:
            raise ValueError("joint feedback encoding must be non-empty")

@dataclass(frozen=True)
class UrdfJointDescription:
    joint_id: str
    joint_type: JointType
    parent_link_id: str
    child_link_id: str
    parent_joint_id: Optional[str]
    origin_translation: Tuple[float, float, float]
    origin_quaternion_xyzw: Tuple[float, float, float, float]
    translation_axis: Tuple[float, float, float]
    rotation_axis: Tuple[float, float, float]
    position_lower: float
    position_upper: float
    velocity_limit: float
    periodic: bool

    def __post_init__(self) -> None:
        RobotMorphologyModule.ValidateIdentifier(self.joint_id, "joint_id")
        if type(self.joint_type) is not JointType:
            raise TypeError("joint_type must be JointType")
        RobotMorphologyModule.ValidateIdentifier(self.parent_link_id, "parent_link_id")
        RobotMorphologyModule.ValidateIdentifier(self.child_link_id, "child_link_id")
        if self.parent_joint_id is not None:
            RobotMorphologyModule.ValidateIdentifier(self.parent_joint_id, "parent_joint_id")
            if self.parent_joint_id == self.joint_id:
                raise ValueError("a joint cannot be its own parent")
        RobotMorphologyModule.ValidateFiniteTuple(self.origin_translation, 3, "origin_translation")
        quaternion = RobotMorphologyModule.ValidateFiniteTuple(
            self.origin_quaternion_xyzw, 4, "origin_quaternion_xyzw")
        if abs(sum(value * value for value in quaternion) - 1.0) > 1e-6:
            raise ValueError("origin_quaternion_xyzw must be a unit quaternion")
        translation = RobotMorphologyModule.ValidateFiniteTuple(
            self.translation_axis, 3, "translation_axis")
        rotation = RobotMorphologyModule.ValidateFiniteTuple(
            self.rotation_axis, 3, "rotation_axis")
        translationNorm = math.sqrt(sum(value * value for value in translation))
        rotationNorm = math.sqrt(sum(value * value for value in rotation))
        if self.joint_type is JointType.FIXED:
            if translationNorm > 1e-12 or rotationNorm > 1e-12:
                raise ValueError("fixed joints cannot expose a motion axis")
        elif self.joint_type is JointType.PRISMATIC:
            if abs(translationNorm - 1.0) > 1e-6 or rotationNorm > 1e-12:
                raise ValueError("prismatic joints require one translation axis")
        else:
            if abs(rotationNorm - 1.0) > 1e-6 or translationNorm > 1e-12:
                raise ValueError("rotational joints require one rotation axis")
        for value in (
            self.position_lower,
            self.position_upper,
            self.velocity_limit,
        ):
            if not math.isfinite(float(value)):
                raise ValueError("joint limits must be finite")
        if float(self.position_lower) >= float(self.position_upper):
            raise ValueError("joint position limits must be ordered")
        if float(self.velocity_limit) <= 0.0:
            raise ValueError("joint velocity limit must be positive")
        if type(self.periodic) is not bool:
            raise TypeError("joint periodic flag must be boolean")
        if self.periodic and self.joint_type not in (
            JointType.REVOLUTE,
            JointType.CONTINUOUS,
        ):
            raise ValueError("only rotational joints can be periodic")


JointDefinition = UrdfJointDescription


@dataclass(frozen=True)
class SrdfEndEffectorDescription:
    effector_id: str
    effector_type: EndEffectorType
    parent_effector_id: Optional[str]
    joint_ids: Tuple[str, ...]
    translation_basis: Tuple[Tuple[float, ...], ...]
    rotation_basis: Tuple[Tuple[float, ...], ...]
    target_lower: Tuple[float, ...]
    target_upper: Tuple[float, ...]
    terminal_translation: Tuple[float, float, float]
    terminal_quaternion_xyzw: Tuple[float, float, float, float]
    reference_frame_id: str
    is_perception_slot: bool
    progress_enter: float
    progress_exit: float
    dwell_cycles: int
    translation_error_scale: float = 1.0
    rotation_error_scale: float = 1.0

    def __post_init__(self) -> None:
        RobotMorphologyModule.ValidateIdentifier(self.effector_id, "effector_id")
        if type(self.effector_type) is not EndEffectorType:
            raise TypeError("effector_type must be EndEffectorType")
        if self.parent_effector_id is not None:
            RobotMorphologyModule.ValidateIdentifier(self.parent_effector_id, "parent_effector_id")
            if self.parent_effector_id == self.effector_id:
                raise ValueError("an end effector cannot parent itself")
        if not self.joint_ids:
            raise ValueError("an end effector must reference a joint chain")
        if any(type(value) is not str or not value for value in self.joint_ids):
            raise ValueError("end-effector joint identifiers must be non-empty")
        if len(set(self.joint_ids)) != len(self.joint_ids):
            raise ValueError("end-effector joint identifiers must be unique")
        translation = RobotMorphologyModule.ValidateBasis(self.translation_basis, "translation_basis")
        rotation = RobotMorphologyModule.ValidateBasis(self.rotation_basis, "rotation_basis")
        targetDim = len(translation[0]) + len(rotation[0])
        if targetDim < 1:
            raise ValueError("an end effector must expose a target coordinate")
        lower = RobotMorphologyModule.ValidateFiniteTuple(
            self.target_lower,
            targetDim,
            "target_lower")
        upper = RobotMorphologyModule.ValidateFiniteTuple(
            self.target_upper,
            targetDim,
            "target_upper")
        if any(
            lowerValue >= upperValue
            for lowerValue, upperValue in zip(lower, upper)
        ):
            raise ValueError("end-effector target limits must be ordered")
        RobotMorphologyModule.ValidateFiniteTuple(
            self.terminal_translation, 3, "terminal_translation")
        quaternion = RobotMorphologyModule.ValidateFiniteTuple(
            self.terminal_quaternion_xyzw, 4, "terminal_quaternion_xyzw")
        if abs(sum(value * value for value in quaternion) - 1.0) > 1e-6:
            raise ValueError("terminal_quaternion_xyzw must be a unit quaternion")
        RobotMorphologyModule.ValidateIdentifier(self.reference_frame_id, "reference_frame_id")
        if type(self.is_perception_slot) is not bool:
            raise TypeError("is_perception_slot must be boolean")
        if self.is_perception_slot:
            if len(translation[0]) != 0 or len(rotation[0]) < 1:
                raise ValueError("perception actuators must be pure rotation")
            if self.effector_type is not EndEffectorType.SENSOR_ACTUATOR:
                raise ValueError("perception actuators require sensor semantics")
        if (
            not math.isfinite(float(self.progress_enter))
            or not math.isfinite(float(self.progress_exit))
            or float(self.progress_enter) <= 0.0
            or float(self.progress_exit) <= float(self.progress_enter)
        ):
            raise ValueError("progress thresholds must define positive hysteresis")
        if type(self.dwell_cycles) is not int or self.dwell_cycles < 1:
            raise ValueError("dwell_cycles must be a positive integer")
        if (
            not math.isfinite(float(self.translation_error_scale))
            or not math.isfinite(float(self.rotation_error_scale))
            or float(self.translation_error_scale) <= 0.0
            or float(self.rotation_error_scale) <= 0.0
        ):
            raise ValueError("end-effector error scales must be positive")

    @property
    def TargetDim(self) -> int:
        return len(self.translation_basis[0]) + len(self.rotation_basis[0])


EndEffector = SrdfEndEffectorDescription


@dataclass(frozen=True)
class PerceptionCalibrationBinding:
    component_id: str
    calibration_id: str
    frame_id: str
    projection_matrix: Tuple[Tuple[float, ...], ...]
    reference_size: Tuple[int, int]
    primary: bool

    def __post_init__(self) -> None:
        RobotMorphologyModule.ValidateIdentifier(self.component_id, "component_id")
        RobotMorphologyModule.ValidateIdentifier(self.calibration_id, "calibration_id")
        RobotMorphologyModule.ValidateIdentifier(self.frame_id, "frame_id")
        matrix = tuple(
            tuple(float(value) for value in row)
            for row in self.projection_matrix)
        if (
            len(matrix) != 3
            or any(len(row) != 3 for row in matrix)
            or any(not math.isfinite(value) for row in matrix for value in row)
            or matrix[0][0] <= 0.0
            or matrix[1][1] <= 0.0
            or matrix[2] != (0.0, 0.0, 1.0)
        ):
            raise ValueError("perception projection must be a calibrated matrix")
        if (
            len(self.reference_size) != 2
            or any(type(value) is not int or value < 1 for value in self.reference_size)
        ):
            raise ValueError("perception reference size must be positive")
        if type(self.primary) is not bool:
            raise TypeError("perception primary flag must be boolean")


@dataclass(frozen=True)
class PerceptionProjectionView:
    calibration_id: str
    reference_frame_id: str
    projection_matrix: Tuple[Tuple[float, ...], ...]
    reference_size: Tuple[int, int]


@dataclass(frozen=True)
class UrdfRobotDescription:
    robot_name: str
    description_id: str
    joints: Tuple[UrdfJointDescription, ...]

    def __post_init__(self) -> None:
        RobotMorphologyModule.ValidateIdentifier(self.robot_name, "robot_name")
        RobotMorphologyModule.ValidateIdentifier(self.description_id, "description_id")
        if not self.joints:
            raise ValueError("URDF description must expose joint coordinates")
        if any(type(value) is not UrdfJointDescription for value in self.joints):
            raise TypeError("URDF joints must be UrdfJointDescription values")
        identifiers = tuple(value.joint_id for value in self.joints)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("URDF joint identifiers must be unique")


@dataclass(frozen=True)
class SrdfSemanticDescription:
    robot_name: str
    description_id: str
    semantic_definition_id: str
    end_effectors: Tuple[SrdfEndEffectorDescription, ...]
    perception_calibrations: Tuple[PerceptionCalibrationBinding, ...] = ()

    def __post_init__(self) -> None:
        RobotMorphologyModule.ValidateIdentifier(self.robot_name, "robot_name")
        RobotMorphologyModule.ValidateIdentifier(self.description_id, "description_id")
        RobotMorphologyModule.ValidateIdentifier(
            self.semantic_definition_id, "semantic_definition_id")
        if not self.end_effectors:
            raise ValueError("SRDF description must expose end effectors")
        if any(
            type(value) is not SrdfEndEffectorDescription
            for value in self.end_effectors
        ):
            raise TypeError("SRDF end effectors have the wrong type")


class UrdfReaderProtocol(Protocol):
    def Read(self, source: Union[str, Path]) -> UrdfRobotDescription:
        ...


class SrdfReaderProtocol(Protocol):
    def Read(self, source: Union[str, Path]) -> SrdfSemanticDescription:
        ...


@dataclass(frozen=True)
class RobotDefinition:
    profile_id: str
    description_id: str
    semantic_definition_id: str
    adapter_id: str
    joints: Tuple[JointDefinition, ...]
    end_effectors: Tuple[EndEffector, ...]
    perception_calibrations: Tuple[PerceptionCalibrationBinding, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "description_id",
            "semantic_definition_id",
            "adapter_id",
        ):
            RobotMorphologyModule.ValidateIdentifier(getattr(self, name), name)
        if not self.joints or any(
            type(value) is not JointDefinition for value in self.joints
        ):
            raise TypeError("robot joints must be JointDefinition values")
        if any(value.joint_type is JointType.FIXED for value in self.joints):
            raise ValueError(
                "fixed URDF transforms must be folded before coordinate compilation")
        if not self.end_effectors or any(
            type(value) is not EndEffector for value in self.end_effectors
        ):
            raise TypeError("robot end effectors must be EndEffector values")
        jointIds = tuple(value.joint_id for value in self.joints)
        effectorIds = tuple(value.effector_id for value in self.end_effectors)
        if len(set(jointIds)) != len(jointIds):
            raise ValueError("robot joint identifiers must be unique")
        if len(set(effectorIds)) != len(effectorIds):
            raise ValueError("robot end-effector identifiers must be unique")
        jointSet = set(jointIds)
        effectorSet = set(effectorIds)
        if any(
            jointId not in jointSet
            for effector in self.end_effectors
            for jointId in effector.joint_ids
        ):
            raise ValueError("end-effector chains must reference robot joints")
        if any(
            effector.parent_effector_id is not None
            and effector.parent_effector_id not in effectorSet
            for effector in self.end_effectors
        ):
            raise ValueError("end-effector parents must reference end effectors")
        jointById = {value.joint_id: value for value in self.joints}
        effectorById = {
            value.effector_id: value for value in self.end_effectors}
        for effector in self.end_effectors:
            chain = tuple(jointById[value] for value in effector.joint_ids)
            if chain[0].parent_joint_id is not None or any(
                child.parent_joint_id != parent.joint_id
                for parent, child in zip(chain[:-1], chain[1:])
            ):
                raise ValueError("end-effector joint chains must be continuous from a root")
            if effector.parent_effector_id is not None:
                parentChain = effectorById[
                    effector.parent_effector_id].joint_ids
                if effector.joint_ids[:len(parentChain)] != parentChain:
                    raise ValueError("child end-effector chains must extend their parent chain")
            if (
                effector.reference_frame_id != "world"
                and (
                    effector.reference_frame_id not in effectorSet
                    or effector.reference_frame_id == effector.effector_id)
            ):
                raise ValueError("end-effector reference frames must be world or another endpoint")
            if effector.is_perception_slot:
                if any(
                    joint.joint_type not in (
                        JointType.REVOLUTE,
                        JointType.CONTINUOUS)
                    for joint in chain
                ):
                    raise ValueError("pure-rotation perception chains require rotational joints")
                if any(
                    math.sqrt(sum(value * value for value in joint.origin_translation))
                    > 1e-9
                    for joint in chain[1:]
                ) or math.sqrt(sum(
                    value * value
                    for value in effector.terminal_translation)) > 1e-9:
                    raise ValueError("perception rotation axes and optical center must share one pivot")
        calibrationIds = tuple(
            value.calibration_id for value in self.perception_calibrations)
        if len(set(calibrationIds)) != len(calibrationIds):
            raise ValueError("perception calibration identifiers must be unique")
        perceptionIds = {
            value.effector_id
            for value in self.end_effectors
            if value.is_perception_slot
        }
        if any(
            value.component_id not in perceptionIds
            for value in self.perception_calibrations
        ):
            raise ValueError("calibration must reference a perception end effector")
        if sum(
            int(value.primary) for value in self.perception_calibrations
        ) != int(bool(perceptionIds)):
            raise ValueError("robot definition requires one primary projection")

@dataclass(frozen=True)
class RobotEmbodimentContractView:
    schema_version: int
    description_id: str
    semantic_definition_id: str
    contract_id: str
    model_shape_id: str
    adapter_id: str
    model_signature: str
    timestamp_unit: str
    timestamp_reference: str
    joint_count: int
    end_effector_count: int
    joint_feedback_layout: PackedLayout
    end_effector_target_layout: PackedLayout
    static_joint_tokens: Tuple[Tuple[float, ...], ...]
    static_end_effector_tokens: Tuple[Tuple[float, ...], ...]
    joint_translation_basis: PackedTensor
    joint_rotation_basis: PackedTensor
    joint_lower: Tuple[float, ...]
    joint_upper: Tuple[float, ...]
    joint_velocity_limit: Tuple[float, ...]
    joint_periodic: Tuple[bool, ...]
    joint_rotational: Tuple[bool, ...]
    end_effector_translation_basis: PackedTensor
    end_effector_rotation_basis: PackedTensor
    end_effector_target_lower: Tuple[float, ...]
    end_effector_target_upper: Tuple[float, ...]
    end_effector_joint_chain_offsets: Tuple[int, ...]
    end_effector_joint_chain_indices: Tuple[int, ...]
    parent_index: Tuple[int, ...]
    topological_layers: Tuple[Tuple[int, ...], ...]
    root_mask: Tuple[bool, ...]
    child_mask: Tuple[bool, ...]
    independent_mask: Tuple[bool, ...]
    subtree_offsets: Tuple[int, ...]
    subtree_indices: Tuple[int, ...]
    perception_view: PackedView
    perception_projection: Optional[PerceptionProjectionView]
    primary_perception_view_index: Optional[int]
    progress_enter: Tuple[float, ...]
    progress_exit: Tuple[float, ...]
    dwell_cycles: Tuple[int, ...]
    translation_error_scale: Tuple[float, ...]
    rotation_error_scale: Tuple[float, ...]
    model_shape: EmbodimentShape

    def __post_init__(self) -> None:
        self.Validate()

    def Validate(self) -> None:
        for name in (
            "description_id",
            "semantic_definition_id",
            "contract_id",
            "model_shape_id",
            "adapter_id",
            "model_signature",
            "timestamp_unit",
            "timestamp_reference",
        ):
            RobotMorphologyModule.ValidateIdentifier(getattr(self, name), name)
        if type(self.schema_version) is not int or self.schema_version < 1:
            raise ValueError("contract schema version must be positive")
        if type(self.joint_count) is not int or self.joint_count < 1:
            raise ValueError("contract joint count must be positive")
        if type(self.end_effector_count) is not int or self.end_effector_count < 1:
            raise ValueError("contract end-effector count must be positive")
        if self.joint_feedback_layout.SlotCount != self.joint_count:
            raise ValueError("joint feedback layout does not match joint count")
        if self.end_effector_target_layout.SlotCount != self.end_effector_count:
            raise ValueError("target layout does not match end-effector count")
        if len(self.static_joint_tokens) != self.joint_count:
            raise ValueError("joint static tokens do not match joint count")
        if len(self.static_end_effector_tokens) != self.end_effector_count:
            raise ValueError("end-effector static tokens do not match endpoint count")
        if len({len(value) for value in self.static_joint_tokens}) != 1:
            raise ValueError("joint static token widths must be uniform")
        if len({len(value) for value in self.static_end_effector_tokens}) != 1:
            raise ValueError("end-effector static token widths must be uniform")
        if any(
            not math.isfinite(float(value))
            for token in self.static_joint_tokens + self.static_end_effector_tokens
            for value in token
        ):
            raise ValueError("static tokens must be finite")
        if self.joint_translation_basis.shapes != ((3, 1),) * self.joint_count:
            raise ValueError("joint translation bases must be three by one")
        if self.joint_rotation_basis.shapes != ((3, 1),) * self.joint_count:
            raise ValueError("joint rotation bases must be three by one")
        if len(self.end_effector_translation_basis.shapes) != self.end_effector_count:
            raise ValueError("endpoint translation basis count is invalid")
        if len(self.end_effector_rotation_basis.shapes) != self.end_effector_count:
            raise ValueError("endpoint rotation basis count is invalid")
        for index in range(self.end_effector_count):
            translationShape = self.end_effector_translation_basis.shapes[index]
            rotationShape = self.end_effector_rotation_basis.shapes[index]
            if translationShape[0] != 3 or rotationShape[0] != 3:
                raise ValueError("endpoint motion bases must have three rows")
            if self.end_effector_target_layout.Width(index) != (
                translationShape[1] + rotationShape[1]
            ):
                raise ValueError("endpoint target layout width is inconsistent")
        if (
            len(self.end_effector_target_lower)
            != self.end_effector_target_layout.PackedDim
            or len(self.end_effector_target_upper)
            != self.end_effector_target_layout.PackedDim
            or any(
                not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or float(lower) >= float(upper)
                for lower, upper in zip(
                    self.end_effector_target_lower,
                    self.end_effector_target_upper)
            )
        ):
            raise ValueError("endpoint compact target limits are invalid")
        for values, name in (
            (self.joint_lower, "joint_lower"),
            (self.joint_upper, "joint_upper"),
            (self.joint_velocity_limit, "joint_velocity_limit"),
            (self.joint_periodic, "joint_periodic"),
            (self.joint_rotational, "joint_rotational"),
        ):
            if len(values) != self.joint_count:
                raise ValueError(f"{name} does not match joint count")
        if any(
            not math.isfinite(float(value))
            for values in (
                self.joint_lower,
                self.joint_upper,
                self.joint_velocity_limit,
            )
            for value in values
        ):
            raise ValueError("joint contract limits must be finite")
        if any(
            lower >= upper or velocity <= 0.0
            for lower, upper, velocity in zip(
                self.joint_lower,
                self.joint_upper,
                self.joint_velocity_limit)
        ):
            raise ValueError("joint contract limits are invalid")
        if any(
            type(value) is not bool
            for values in (self.joint_periodic, self.joint_rotational)
            for value in values
        ):
            raise ValueError("joint contract motion flags must be boolean")
        for index in range(self.joint_count):
            expectedWidth = (
                3
                if self.joint_periodic[index]
                else 4
                if self.joint_rotational[index]
                else 2)
            if self.joint_feedback_layout.Width(index) != expectedWidth:
                raise ValueError("joint feedback layout does not match joint motion")
            translation = self.joint_translation_basis.Matrix(
                index,
                dtype=torch.float64)
            rotation = self.joint_rotation_basis.Matrix(
                index,
                dtype=torch.float64)
            hasTranslation = bool(torch.count_nonzero(translation).item())
            hasRotation = bool(torch.count_nonzero(rotation).item())
            if (
                self.joint_rotational[index] != hasRotation
                or self.joint_rotational[index] == hasTranslation
                or self.joint_periodic[index] and not hasRotation
            ):
                raise ValueError("joint motion flags do not match joint bases")
        if len(self.end_effector_joint_chain_offsets) != self.end_effector_count + 1:
            raise ValueError("endpoint chain offsets do not match endpoint count")
        chainLayout = PackedLayout(self.end_effector_joint_chain_offsets)
        if (
            chainLayout.PackedDim != len(self.end_effector_joint_chain_indices)
            or any(
                chainLayout.Width(index) < 1
                for index in range(self.end_effector_count))
        ):
            raise ValueError("endpoint chain offsets do not match indices")
        if any(
            value < 0 or value >= self.joint_count
            for value in self.end_effector_joint_chain_indices
        ):
            raise ValueError("endpoint chain joint index is out of range")
        for endpointIndex in range(self.end_effector_count):
            chain = self.end_effector_joint_chain_indices[
                chainLayout.Slice(endpointIndex)]
            if any(right <= left for left, right in zip(chain[:-1], chain[1:])):
                raise ValueError("endpoint joint chains must be strictly ordered")
        if len(self.parent_index) != self.end_effector_count:
            raise ValueError("endpoint parent index count is invalid")
        if any(
            type(parentIndex) is not int
            or parentIndex < -1
            or parentIndex >= self.end_effector_count
            or parentIndex == endpointIndex
            for endpointIndex, parentIndex in enumerate(self.parent_index)
        ):
            raise ValueError("endpoint parent indices are invalid")
        (
            expectedLayers,
            expectedRoot,
            expectedChild,
            expectedIndependent,
            expectedSubtreeOffsets,
            expectedSubtreeIndices,
        ) = RobotMorphologyModule.CompileHierarchyIndices(self.parent_index)
        if (
            self.topological_layers != expectedLayers
            or self.root_mask != expectedRoot
            or self.child_mask != expectedChild
            or self.independent_mask != expectedIndependent
            or self.subtree_offsets != expectedSubtreeOffsets
            or self.subtree_indices != expectedSubtreeIndices
        ):
            raise ValueError("endpoint hierarchy compilation is inconsistent")
        if self.perception_view.source_slot_count != self.end_effector_count:
            raise ValueError("perception view source count is invalid")
        for index in self.perception_view.indices:
            if self.end_effector_translation_basis.shapes[index][1] != 0:
                raise ValueError("perception endpoint cannot translate")
            if self.end_effector_rotation_basis.shapes[index][1] < 1:
                raise ValueError("perception endpoint must rotate")
        if self.perception_projection is None:
            if self.primary_perception_view_index is not None:
                raise ValueError("empty perception projection cannot have a primary view")
        elif (
            type(self.primary_perception_view_index) is not int
            or self.primary_perception_view_index < 0
            or self.primary_perception_view_index >= len(self.perception_view.indices)
        ):
            raise ValueError("primary perception view index is out of range")
        if len(self.progress_enter) != self.end_effector_count:
            raise ValueError("progress enter thresholds are invalid")
        if len(self.progress_exit) != self.end_effector_count:
            raise ValueError("progress exit thresholds are invalid")
        if len(self.dwell_cycles) != self.end_effector_count:
            raise ValueError("progress dwell counts are invalid")
        if (
            len(self.translation_error_scale) != self.end_effector_count
            or len(self.rotation_error_scale) != self.end_effector_count
            or any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for values in (
                    self.translation_error_scale,
                    self.rotation_error_scale)
                for value in values)
        ):
            raise ValueError("endpoint error scales are invalid")
        if any(
            enter <= 0.0 or exitValue <= enter
            for enter, exitValue in zip(self.progress_enter, self.progress_exit)
        ):
            raise ValueError("progress hysteresis thresholds are invalid")
        if any(type(value) is not int or value < 1 for value in self.dwell_cycles):
            raise ValueError("progress dwell counts must be positive")
        if type(self.model_shape) is not EmbodimentShape:
            raise TypeError("contract model shape has the wrong type")
        expectedShape = EmbodimentShape(
            joint_token_count=self.joint_count,
            joint_static_descriptor_dim=len(self.static_joint_tokens[0]),
            joint_feedback_packed_dim=self.joint_feedback_layout.PackedDim,
            end_effector_token_count=self.end_effector_count,
            end_effector_static_descriptor_dim=len(
                self.static_end_effector_tokens[0]),
            end_effector_target_packed_dim=self.end_effector_target_layout.PackedDim,
            hierarchy_edge_count=sum(value >= 0 for value in self.parent_index),
            perception_view_dim=len(self.perception_view.indices))
        if self.model_shape != expectedShape:
            raise ValueError("contract model shape is inconsistent")


@dataclass(frozen=True)
class RobotEmbodimentContract:
    schema_version: int
    description_id: str
    semantic_definition_id: str
    contract_id: str
    model_shape_id: str
    adapter_id: str
    model_signature: str
    joint_ids: Tuple[str, ...]
    end_effector_ids: Tuple[str, ...]
    joint_parent_index: Tuple[int, ...]
    joint_origin_translation: Tuple[Tuple[float, float, float], ...]
    joint_origin_quaternion_xyzw: Tuple[Tuple[float, float, float, float], ...]
    end_effector_terminal_translation: Tuple[Tuple[float, float, float], ...]
    end_effector_terminal_quaternion_xyzw: Tuple[Tuple[float, float, float, float], ...]
    end_effector_reference_frame: Tuple[str, ...]
    view: RobotEmbodimentContractView

    def __post_init__(self) -> None:
        for name in (
            "description_id",
            "semantic_definition_id",
            "contract_id",
            "model_shape_id",
            "adapter_id",
            "model_signature",
        ):
            if getattr(self, name) != getattr(self.view, name):
                raise ValueError(f"contract and view {name} values differ")
        if self.schema_version != self.view.schema_version:
            raise ValueError("contract and view schema versions differ")
        if len(self.joint_ids) != self.view.joint_count:
            raise ValueError("contract joint identifiers do not match view")
        if len(self.end_effector_ids) != self.view.end_effector_count:
            raise ValueError("contract endpoint identifiers do not match view")
        if len(set(self.joint_ids)) != len(self.joint_ids):
            raise ValueError("contract joint identifiers must be unique")
        if len(set(self.end_effector_ids)) != len(self.end_effector_ids):
            raise ValueError("contract endpoint identifiers must be unique")
        for identifier in self.joint_ids + self.end_effector_ids:
            RobotMorphologyModule.ValidateIdentifier(identifier, "contract component identifier")
        if len(self.joint_parent_index) != self.view.joint_count:
            raise ValueError("joint parent indices do not match joint count")
        if any(
            type(parentIndex) is not int
            or parentIndex < -1
            or parentIndex >= jointIndex
            for jointIndex, parentIndex in enumerate(self.joint_parent_index)
        ):
            raise ValueError("joint parent indices must be topologically ordered")
        if len(self.joint_origin_translation) != self.view.joint_count:
            raise ValueError("joint origins do not match joint count")
        if len(self.joint_origin_quaternion_xyzw) != self.view.joint_count:
            raise ValueError("joint orientations do not match joint count")
        if len(self.end_effector_terminal_translation) != self.view.end_effector_count:
            raise ValueError("endpoint translations do not match endpoint count")
        if len(self.end_effector_terminal_quaternion_xyzw) != self.view.end_effector_count:
            raise ValueError("endpoint orientations do not match endpoint count")
        if len(self.end_effector_reference_frame) != self.view.end_effector_count:
            raise ValueError("endpoint reference frames do not match endpoint count")
        for values in (
            self.joint_origin_translation,
            self.end_effector_terminal_translation,
        ):
            for value in values:
                RobotMorphologyModule.ValidateFiniteTuple(value, 3, "contract translation")
        for values in (
            self.joint_origin_quaternion_xyzw,
            self.end_effector_terminal_quaternion_xyzw,
        ):
            for value in values:
                quaternion = RobotMorphologyModule.ValidateFiniteTuple(
                    value,
                    4,
                    "contract quaternion")
                if abs(sum(component * component for component in quaternion) - 1.0) > 1e-6:
                    raise ValueError("contract quaternions must have unit norm")
        endpointSet = set(self.end_effector_ids)
        if any(
            type(reference) is not str
            or not reference
            or reference != "world" and reference not in endpointSet
            for reference in self.end_effector_reference_frame
        ):
            raise ValueError("contract endpoint reference frames are invalid")
        endpointChains = []
        chainLayout = PackedLayout(
            self.view.end_effector_joint_chain_offsets)
        for endpointIndex in range(self.view.end_effector_count):
            chain = self.view.end_effector_joint_chain_indices[
                chainLayout.Slice(endpointIndex)]
            if self.joint_parent_index[chain[0]] != -1 or any(
                self.joint_parent_index[childIndex] != parentIndex
                for parentIndex, childIndex in zip(chain[:-1], chain[1:])
            ):
                raise ValueError("endpoint chains must follow the joint hierarchy")
            endpointChains.append(chain)
        for endpointIndex, parentIndex in enumerate(self.view.parent_index):
            if parentIndex >= 0 and endpointChains[endpointIndex][
                :len(endpointChains[parentIndex])
            ] != endpointChains[parentIndex]:
                raise ValueError("child endpoint chains must extend parent endpoint chains")
        expectedShapeId = ModelSignatureCompiler.Compile({
            "kind": "embodiment_shape",
            "shape": {
                name: getattr(self.view.model_shape, name)
                for name in self.view.model_shape.__dataclass_fields__
            },
        })
        if self.model_shape_id != expectedShapeId:
            raise ValueError("contract model shape identity is inconsistent")
        expectedSignature = self.CompileContentSignature(
            self.schema_version,
            self.description_id,
            self.semantic_definition_id,
            self.adapter_id,
            self.joint_ids,
            self.end_effector_ids,
            self.joint_parent_index,
            self.joint_origin_translation,
            self.joint_origin_quaternion_xyzw,
            self.end_effector_terminal_translation,
            self.end_effector_terminal_quaternion_xyzw,
            self.end_effector_reference_frame,
            self.view)
        if self.model_signature != expectedSignature:
            raise ValueError("contract content does not match its model signature")
        expectedContractId = ModelSignatureCompiler.Compile({
            "kind": "robot_contract",
            "model_signature": expectedSignature,
        })
        if self.contract_id != expectedContractId:
            raise ValueError("contract identity does not match its model signature")

    @staticmethod
    def CompileContentSignature(
        schemaVersion: int,
        descriptionId: str,
        semanticDefinitionId: str,
        adapterId: str,
        jointIds: Sequence[str],
        endEffectorIds: Sequence[str],
        jointParentIndex: Sequence[int],
        jointOriginTranslation: Sequence[Sequence[float]],
        jointOriginQuaternionXyzw: Sequence[Sequence[float]],
        endEffectorTerminalTranslation: Sequence[Sequence[float]],
        endEffectorTerminalQuaternionXyzw: Sequence[Sequence[float]],
        endEffectorReferenceFrame: Sequence[str],
        contractView: Union[RobotEmbodimentContractView, Mapping[str, Any]],
    ) -> str:
        viewPayload = (
            asdict(contractView)
            if type(contractView) is RobotEmbodimentContractView
            else {
                name: asdict(value)
                if hasattr(value, "__dataclass_fields__")
                else value
                for name, value in contractView.items()
            })
        for name in ("contract_id", "model_shape_id", "model_signature"):
            viewPayload.pop(name, None)
        return ModelSignatureCompiler.Compile({
            "schema_version": schemaVersion,
            "description_id": descriptionId,
            "semantic_definition_id": semanticDefinitionId,
            "adapter_id": adapterId,
            "joint_ids": tuple(jointIds),
            "end_effector_ids": tuple(endEffectorIds),
            "joint_parent_index": tuple(jointParentIndex),
            "joint_origin_translation": tuple(map(tuple, jointOriginTranslation)),
            "joint_origin_quaternion_xyzw": tuple(map(tuple, jointOriginQuaternionXyzw)),
            "end_effector_terminal_translation": tuple(map(tuple, endEffectorTerminalTranslation)),
            "end_effector_terminal_quaternion_xyzw": tuple(map(tuple, endEffectorTerminalQuaternionXyzw)),
            "end_effector_reference_frame": tuple(endEffectorReferenceFrame),
            "view": viewPayload,
        })

class RobotMorphologyModule:
    IdentityBasis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    EmptyBasis = ((), (), ())

    @staticmethod
    def ValidateIdentifier(value: Any, fieldName: str) -> str:
        if type(value) is not str or not value:
            raise ValueError(f"{fieldName} must be a non-empty string")
        return value

    @staticmethod
    def ValidateFiniteTuple(
        values: Sequence[float],
        width: int,
        fieldName: str,
    ) -> Tuple[float, ...]:
        result = tuple(float(value) for value in values)
        if len(result) != width or any(not math.isfinite(value) for value in result):
            raise ValueError(f"{fieldName} must contain {width} finite values")
        return result

    @staticmethod
    def ValidateBasis(
        basis: Sequence[Sequence[float]],
        fieldName: str,
    ) -> Tuple[Tuple[float, ...], ...]:
        result = tuple(tuple(float(value) for value in row) for row in basis)
        if len(result) != 3 or any(len(row) != len(result[0]) for row in result):
            raise ValueError(f"{fieldName} must be a rectangular three-row basis")
        if any(not math.isfinite(value) for row in result for value in row):
            raise ValueError(f"{fieldName} must be finite")
        if result[0] and int(torch.linalg.matrix_rank(
            torch.tensor(result, dtype=torch.float64)
        ).item()) != len(result[0]):
            raise ValueError(f"{fieldName} columns must be linearly independent")
        return result

    @staticmethod
    def ValidateTensor(
        value: torch.Tensor,
        shape: Tuple[int, ...],
        fieldName: str,
        floating: bool,
    ) -> None:
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"{fieldName} has the wrong shape")
        if floating and (
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(f"{fieldName} must be finite floating point")
        if not floating and value.dtype != torch.bool:
            raise ValueError(f"{fieldName} must be boolean")

    @staticmethod
    def FromUrdfSrdf(
        urdfSource: Union[str, Path],
        srdfSource: Union[str, Path],
        profileId: str,
        adapterId: str,
        urdfReader: UrdfReaderProtocol,
        srdfReader: SrdfReaderProtocol,
    ) -> RobotDefinition:
        urdf = urdfReader.Read(urdfSource)
        srdf = srdfReader.Read(srdfSource)
        if type(urdf) is not UrdfRobotDescription or type(srdf) is not SrdfSemanticDescription:
            raise TypeError("robot readers returned an invalid description")
        if urdf.robot_name != srdf.robot_name or urdf.description_id != srdf.description_id:
            raise ValueError("URDF and SRDF identities do not match")
        return RobotDefinition(
            profile_id=RobotMorphologyModule.ValidateIdentifier(profileId, "profileId"),
            description_id=urdf.description_id,
            semantic_definition_id=srdf.semantic_definition_id,
            adapter_id=RobotMorphologyModule.ValidateIdentifier(adapterId, "adapterId"),
            joints=urdf.joints,
            end_effectors=srdf.end_effectors,
            perception_calibrations=srdf.perception_calibrations)

    @staticmethod
    def CompileHierarchyIndices(
        parentIndex: Sequence[int],
    ) -> Tuple[
        Tuple[Tuple[int, ...], ...],
        Tuple[bool, ...],
        Tuple[bool, ...],
        Tuple[bool, ...],
        Tuple[int, ...],
        Tuple[int, ...],
    ]:
        count = len(parentIndex)
        depths = []
        for index in range(count):
            visited = set()
            cursor = index
            depth = -1
            while cursor >= 0:
                if cursor in visited:
                    raise ValueError("end-effector hierarchy must be acyclic")
                visited.add(cursor)
                depth += 1
                cursor = parentIndex[cursor]
            depths.append(depth)
        layers = tuple(
            tuple(index for index, depth in enumerate(depths) if depth == layer)
            for layer in range(max(depths) + 1))
        rootMask = tuple(value < 0 for value in parentIndex)
        childMask = tuple(value >= 0 for value in parentIndex)
        independentMask = tuple(
            rootMask[index] and index not in parentIndex
            for index in range(count))
        subtrees = []
        for root in range(count):
            subtree = []
            for candidate in range(count):
                cursor = candidate
                while cursor >= 0 and cursor != root:
                    cursor = parentIndex[cursor]
                if cursor == root:
                    subtree.append(candidate)
            subtrees.append(tuple(subtree))
        layout = PackedLayout.FromWidths(tuple(map(len, subtrees)))
        return (
            layers,
            rootMask,
            childMask,
            independentMask,
            layout.offsets,
            tuple(index for subtree in subtrees for index in subtree),
        )

    @staticmethod
    def CompileJointTokens(
        joints: Sequence[JointDefinition],
    ) -> Tuple[Tuple[float, ...], ...]:
        indexById = {joint.joint_id: index for index, joint in enumerate(joints)}
        depths = []
        for index, joint in enumerate(joints):
            depth = 0
            parentId = joint.parent_joint_id
            while parentId is not None:
                parentIndex = indexById.get(parentId, index)
                if parentIndex >= index:
                    raise ValueError("joint parents must precede child coordinates")
                depth += 1
                parentId = joints[parentIndex].parent_joint_id
            depths.append(depth)
        maxDepth = max(depths)
        return tuple(tuple(float(value) for value in (
            *(float(typeIndex == int(joint.joint_type)) for typeIndex in range(len(JointType))),
            *joint.translation_axis,
            *joint.rotation_axis,
            joint.position_lower,
            joint.position_upper,
            joint.velocity_limit,
            float(joint.periodic),
            float(depths[index]) / float(max(maxDepth, 1)),
        )) for index, joint in enumerate(joints))

    @staticmethod
    def CompileEndEffectorTokens(
        endEffectors: Sequence[EndEffector],
        parentIndex: Sequence[int],
        layers: Sequence[Sequence[int]],
    ) -> Tuple[Tuple[float, ...], ...]:
        depths = [0] * len(endEffectors)
        for depth, layer in enumerate(layers):
            for index in layer:
                depths[index] = depth
        maxDepth = max(depths)
        maxChain = max(len(effector.joint_ids) for effector in endEffectors)
        tokens = []
        for index, effector in enumerate(endEffectors):
            translation = torch.tensor(effector.translation_basis, dtype=torch.float64)
            rotation = torch.tensor(effector.rotation_basis, dtype=torch.float64)
            tokens.append(tuple(float(value) for value in (
                *(float(typeIndex == int(effector.effector_type)) for typeIndex in range(len(EndEffectorType))),
                *(float(bool(torch.linalg.vector_norm(translation[row]).item())) for row in range(3)),
                *(float(bool(torch.linalg.vector_norm(rotation[row]).item())) for row in range(3)),
                float(parentIndex[index] < 0),
                float(parentIndex[index] >= 0),
                float(effector.is_perception_slot),
                float(depths[index]) / float(max(maxDepth, 1)),
                float(len(effector.joint_ids)) / float(maxChain),
            )))
        return tuple(tokens)

    @staticmethod
    def CompileDefinition(
        definition: RobotDefinition,
    ) -> RobotEmbodimentContract:
        if type(definition) is not RobotDefinition:
            raise TypeError("definition must be RobotDefinition")
        parentIndexById = {
            value.joint_id: index for index, value in enumerate(definition.joints)
        }
        jointParentIndex = tuple(
            -1
            if value.parent_joint_id is None
            else parentIndexById[value.parent_joint_id]
            for value in definition.joints)
        effectorIndexById = {
            value.effector_id: index
            for index, value in enumerate(definition.end_effectors)
        }
        parentIndex = tuple(
            -1
            if value.parent_effector_id is None
            else effectorIndexById[value.parent_effector_id]
            for value in definition.end_effectors)
        (
            topologicalLayers,
            rootMask,
            childMask,
            independentMask,
            subtreeOffsets,
            subtreeIndices,
        ) = RobotMorphologyModule.CompileHierarchyIndices(parentIndex)
        jointStaticTokens = RobotMorphologyModule.CompileJointTokens(
            definition.joints)
        endpointStaticTokens = RobotMorphologyModule.CompileEndEffectorTokens(
            definition.end_effectors,
            parentIndex,
            topologicalLayers)
        jointFeedbackLayout = PackedLayout.FromWidths(tuple(
            3
            if value.periodic
            else 4
            if value.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS)
            else 2
            for value in definition.joints))
        endpointTargetLayout = PackedLayout.FromWidths(tuple(
            value.TargetDim for value in definition.end_effectors))
        jointIndexById = {
            value.joint_id: index
            for index, value in enumerate(definition.joints)
        }
        endpointChains = tuple(
            tuple(jointIndexById[jointId] for jointId in value.joint_ids)
            for value in definition.end_effectors)
        endpointChainLayout = PackedLayout.FromWidths(tuple(map(len, endpointChains)))
        perceptionIndices = tuple(
            index
            for index, value in enumerate(definition.end_effectors)
            if value.is_perception_slot)
        projectionBinding = next(
            (
                value
                for value in definition.perception_calibrations
                if value.primary
            ),
            None)
        projection = (
            None
            if projectionBinding is None
            else PerceptionProjectionView(
                calibration_id=projectionBinding.calibration_id,
                reference_frame_id=projectionBinding.frame_id,
                projection_matrix=projectionBinding.projection_matrix,
                reference_size=projectionBinding.reference_size)
        )
        primaryPerceptionViewIndex = (
            None
            if projectionBinding is None
            else perceptionIndices.index(next(
                index
                for index, value in enumerate(definition.end_effectors)
                if value.effector_id == projectionBinding.component_id))
        )
        shape = EmbodimentShape(
            joint_token_count=len(definition.joints),
            joint_static_descriptor_dim=len(jointStaticTokens[0]),
            joint_feedback_packed_dim=jointFeedbackLayout.PackedDim,
            end_effector_token_count=len(definition.end_effectors),
            end_effector_static_descriptor_dim=len(endpointStaticTokens[0]),
            end_effector_target_packed_dim=endpointTargetLayout.PackedDim,
            hierarchy_edge_count=sum(value >= 0 for value in parentIndex),
            perception_view_dim=len(perceptionIndices))
        shapeId = ModelSignatureCompiler.Compile({
            "kind": "embodiment_shape",
            "shape": {
                name: getattr(shape, name) for name in shape.__dataclass_fields__
            },
        })
        viewFields = dict(
            schema_version=ROBOT_EMBODIMENT_SCHEMA_VERSION,
            description_id=definition.description_id,
            semantic_definition_id=definition.semantic_definition_id,
            model_shape_id=shapeId,
            adapter_id=definition.adapter_id,
            timestamp_unit=FEEDBACK_TIMESTAMP_UNIT,
            timestamp_reference=FEEDBACK_TIMESTAMP_REFERENCE,
            joint_count=len(definition.joints),
            end_effector_count=len(definition.end_effectors),
            joint_feedback_layout=jointFeedbackLayout,
            end_effector_target_layout=endpointTargetLayout,
            static_joint_tokens=jointStaticTokens,
            static_end_effector_tokens=endpointStaticTokens,
            joint_translation_basis=PackedTensor.FromMatrices(tuple(
                tuple((value.translation_axis[row],) for row in range(3))
                for value in definition.joints)),
            joint_rotation_basis=PackedTensor.FromMatrices(tuple(
                tuple((value.rotation_axis[row],) for row in range(3))
                for value in definition.joints)),
            joint_lower=tuple(value.position_lower for value in definition.joints),
            joint_upper=tuple(value.position_upper for value in definition.joints),
            joint_velocity_limit=tuple(
                value.velocity_limit for value in definition.joints),
            joint_periodic=tuple(value.periodic for value in definition.joints),
            joint_rotational=tuple(
                value.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS)
                for value in definition.joints),
            end_effector_translation_basis=PackedTensor.FromMatrices(tuple(
                value.translation_basis for value in definition.end_effectors)),
            end_effector_rotation_basis=PackedTensor.FromMatrices(tuple(
                value.rotation_basis for value in definition.end_effectors)),
            end_effector_target_lower=tuple(
                coordinate
                for value in definition.end_effectors
                for coordinate in value.target_lower),
            end_effector_target_upper=tuple(
                coordinate
                for value in definition.end_effectors
                for coordinate in value.target_upper),
            end_effector_joint_chain_offsets=endpointChainLayout.offsets,
            end_effector_joint_chain_indices=tuple(
                index for chain in endpointChains for index in chain),
            parent_index=parentIndex,
            topological_layers=topologicalLayers,
            root_mask=rootMask,
            child_mask=childMask,
            independent_mask=independentMask,
            subtree_offsets=subtreeOffsets,
            subtree_indices=subtreeIndices,
            perception_view=PackedView(
                source_slot_count=len(definition.end_effectors),
                indices=perceptionIndices),
            perception_projection=projection,
            primary_perception_view_index=primaryPerceptionViewIndex,
            progress_enter=tuple(
                value.progress_enter for value in definition.end_effectors),
            progress_exit=tuple(
                value.progress_exit for value in definition.end_effectors),
            dwell_cycles=tuple(
                value.dwell_cycles for value in definition.end_effectors),
            translation_error_scale=tuple(
                value.translation_error_scale for value in definition.end_effectors),
            rotation_error_scale=tuple(
                value.rotation_error_scale for value in definition.end_effectors),
            model_shape=shape)
        jointIds = tuple(value.joint_id for value in definition.joints)
        endEffectorIds = tuple(
            value.effector_id for value in definition.end_effectors)
        jointOriginTranslation = tuple(
            value.origin_translation for value in definition.joints)
        jointOriginQuaternionXyzw = tuple(
            value.origin_quaternion_xyzw for value in definition.joints)
        endEffectorTerminalTranslation = tuple(
            value.terminal_translation for value in definition.end_effectors)
        endEffectorTerminalQuaternionXyzw = tuple(
            value.terminal_quaternion_xyzw for value in definition.end_effectors)
        endEffectorReferenceFrame = tuple(
            value.reference_frame_id for value in definition.end_effectors)
        modelSignature = RobotEmbodimentContract.CompileContentSignature(
            ROBOT_EMBODIMENT_SCHEMA_VERSION,
            definition.description_id,
            definition.semantic_definition_id,
            definition.adapter_id,
            jointIds,
            endEffectorIds,
            jointParentIndex,
            jointOriginTranslation,
            jointOriginQuaternionXyzw,
            endEffectorTerminalTranslation,
            endEffectorTerminalQuaternionXyzw,
            endEffectorReferenceFrame,
            viewFields)
        contractId = ModelSignatureCompiler.Compile({
            "kind": "robot_contract",
            "model_signature": modelSignature,
        })
        view = RobotEmbodimentContractView(
            contract_id=contractId,
            model_signature=modelSignature,
            **viewFields)
        return RobotEmbodimentContract(
            schema_version=ROBOT_EMBODIMENT_SCHEMA_VERSION,
            description_id=definition.description_id,
            semantic_definition_id=definition.semantic_definition_id,
            contract_id=contractId,
            model_shape_id=shapeId,
            adapter_id=definition.adapter_id,
            model_signature=modelSignature,
            joint_ids=jointIds,
            end_effector_ids=endEffectorIds,
            joint_parent_index=jointParentIndex,
            joint_origin_translation=jointOriginTranslation,
            joint_origin_quaternion_xyzw=jointOriginQuaternionXyzw,
            end_effector_terminal_translation=endEffectorTerminalTranslation,
            end_effector_terminal_quaternion_xyzw=(
                endEffectorTerminalQuaternionXyzw),
            end_effector_reference_frame=endEffectorReferenceFrame,
            view=view)


    @staticmethod
    def CreateRevoluteJoint(
        jointId: str,
        parentJointId: Optional[str],
        originTranslation: Sequence[float],
        rotationAxis: Sequence[float],
        lower: float,
        upper: float,
        velocity: float,
        periodic: bool = False,
    ) -> JointDefinition:
        parentLink = "robot_base" if parentJointId is None else f"{parentJointId}_link"
        return JointDefinition(
            joint_id=jointId,
            joint_type=(JointType.CONTINUOUS if periodic else JointType.REVOLUTE),
            parent_link_id=parentLink,
            child_link_id=f"{jointId}_link",
            parent_joint_id=parentJointId,
            origin_translation=tuple(float(value) for value in originTranslation),
            origin_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            translation_axis=(0.0, 0.0, 0.0),
            rotation_axis=tuple(float(value) for value in rotationAxis),
            position_lower=float(lower),
            position_upper=float(upper),
            velocity_limit=float(velocity),
            periodic=bool(periodic))


    @staticmethod
    def CreateAnthropomorphicArm(
        sideId: str,
        sideSign: float,
    ) -> Tuple[Tuple[JointDefinition, ...], Tuple[EndEffector, ...]]:
        upperNames = (
            "shoulder_flexion",
            "shoulder_abduction",
            "shoulder_rotation",
            "elbow_flexion",
            "forearm_rotation",
            "wrist_flexion",
            "wrist_deviation",
        )
        upperAxes = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        upperOrigins = (
            (0.24 * sideSign, 0.0, 0.45),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, -0.30, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, -0.26, 0.0),
            (0.0, 0.0, 0.0),
        )
        upperLimits = (
            (-2.79, 2.09, 2.4),
            (-1.57, 2.79, 2.4),
            (-1.57, 1.57, 2.4),
            (0.0, 2.62, 2.8),
            (-1.57, 1.57, 3.0),
            (-1.22, 1.22, 3.0),
            (-0.61, 0.61, 3.0),
        )
        joints = []
        upperIds = []
        parentId = None
        for name, axis, origin, limits in zip(
            upperNames,
            upperAxes,
            upperOrigins,
            upperLimits,
        ):
            jointId = f"{sideId}_{name}"
            joints.append(RobotMorphologyModule.CreateRevoluteJoint(
                jointId,
                parentId,
                origin,
                axis,
                limits[0],
                limits[1],
                limits[2]))
            upperIds.append(jointId)
            parentId = jointId
        fingerSpecifications = (
            (
                "thumb",
                (0.045 * sideSign, -0.015, -0.005),
                ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                ((-0.70, 0.70), (-0.35, 1.40), (0.0, 1.40), (0.0, 1.40)),
                (0.0, -0.025, 0.0),
            ),
            (
                "index",
                (0.030 * sideSign, -0.040, 0.0),
                ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                ((-0.35, 1.57), (-0.35, 0.35), (0.0, 1.75), (0.0, 1.40)),
                (0.0, -0.025, 0.0),
            ),
            (
                "middle",
                (0.010 * sideSign, -0.045, 0.0),
                ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                ((-0.35, 1.57), (-0.25, 0.25), (0.0, 1.75), (0.0, 1.40)),
                (0.0, -0.027, 0.0),
            ),
            (
                "ring",
                (-0.010 * sideSign, -0.042, 0.0),
                ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                ((-0.35, 1.57), (-0.30, 0.30), (0.0, 1.75), (0.0, 1.40)),
                (0.0, -0.024, 0.0),
            ),
            (
                "little",
                (-0.030 * sideSign, -0.035, 0.0),
                ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                ((-0.35, 1.57), (-0.40, 0.40), (0.0, 1.75), (0.0, 1.40)),
                (0.0, -0.021, 0.0),
            ),
        )
        fingerChains = []
        for fingerName, baseOrigin, axes, limits, terminal in fingerSpecifications:
            chain = []
            parentId = upperIds[-1]
            origins = (
                baseOrigin,
                (0.0, -0.035, 0.0),
                (0.0, -0.025, 0.0),
                (0.0, -0.020, 0.0),
            )
            for coordinateIndex in range(4):
                jointId = f"{sideId}_{fingerName}_{coordinateIndex}"
                joints.append(RobotMorphologyModule.CreateRevoluteJoint(
                    jointId,
                    parentId,
                    origins[coordinateIndex],
                    axes[coordinateIndex],
                    limits[coordinateIndex][0],
                    limits[coordinateIndex][1],
                    4.0))
                chain.append(jointId)
                parentId = jointId
            fingerChains.append((fingerName, tuple(chain), terminal))
        wristId = f"{sideId}_wrist"
        endEffectors = [EndEffector(
            effector_id=wristId,
            effector_type=EndEffectorType.WRIST,
            parent_effector_id=None,
            joint_ids=tuple(upperIds),
            translation_basis=RobotMorphologyModule.IdentityBasis,
            rotation_basis=RobotMorphologyModule.IdentityBasis,
            target_lower=(-1.5, -1.5, -1.5, -math.pi, -math.pi, -math.pi),
            target_upper=(1.5, 1.5, 1.5, math.pi, math.pi, math.pi),
            terminal_translation=(0.0, 0.0, 0.0),
            terminal_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            reference_frame_id="world",
            is_perception_slot=False,
            progress_enter=0.025,
            progress_exit=0.040,
            dwell_cycles=3)]
        for fingerName, chain, terminal in fingerChains:
            endEffectors.append(EndEffector(
                effector_id=f"{sideId}_{fingerName}_tip",
                effector_type=EndEffectorType.FINGERTIP,
                parent_effector_id=wristId,
                joint_ids=tuple(upperIds) + chain,
                translation_basis=RobotMorphologyModule.IdentityBasis,
                rotation_basis=RobotMorphologyModule.IdentityBasis,
                target_lower=(-0.35, -0.35, -0.35, -math.pi, -math.pi, -math.pi),
                target_upper=(0.35, 0.35, 0.35, math.pi, math.pi, math.pi),
                terminal_translation=terminal,
                terminal_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
                reference_frame_id=wristId,
                is_perception_slot=False,
                progress_enter=0.012,
                progress_exit=0.020,
                dwell_cycles=3))
        return tuple(joints), tuple(endEffectors)


    @staticmethod
    def TemporaryDefinition() -> RobotDefinition:
        leftJoints, leftEffectors = RobotMorphologyModule.CreateAnthropomorphicArm(
            "left", 1.0)
        rightJoints, rightEffectors = RobotMorphologyModule.CreateAnthropomorphicArm(
            "right", -1.0)
        cameraJoints = (
            RobotMorphologyModule.CreateRevoluteJoint(
                "camera_yaw",
                None,
                (0.0, 0.0, 0.65),
                (0.0, 0.0, 1.0),
                -math.pi,
                math.pi,
                1.5,
                True),
            RobotMorphologyModule.CreateRevoluteJoint(
                "camera_pitch",
                "camera_yaw",
                (0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                -0.5 * math.pi,
                0.5 * math.pi,
                1.5),
            RobotMorphologyModule.CreateRevoluteJoint(
                "camera_roll",
                "camera_pitch",
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                -math.pi,
                math.pi,
                1.5,
                True),
        )
        camera = EndEffector(
            effector_id="camera",
            effector_type=EndEffectorType.SENSOR_ACTUATOR,
            parent_effector_id=None,
            joint_ids=tuple(value.joint_id for value in cameraJoints),
            translation_basis=RobotMorphologyModule.EmptyBasis,
            rotation_basis=RobotMorphologyModule.IdentityBasis,
            target_lower=(-math.pi, -math.pi, -math.pi),
            target_upper=(math.pi, math.pi, math.pi),
            terminal_translation=(0.0, 0.0, 0.0),
            terminal_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            reference_frame_id="world",
            is_perception_slot=True,
            progress_enter=0.02,
            progress_exit=0.04,
            dwell_cycles=2)
        definition = RobotDefinition(
            profile_id="temporary_dual_anthropomorphic_arm_camera",
            description_id="temporary_dual_anthropomorphic_arm_camera_urdf",
            semantic_definition_id="temporary_dual_anthropomorphic_arm_camera_srdf",
            adapter_id="external_endpoint_ik_adapter",
            joints=leftJoints + rightJoints + cameraJoints,
            end_effectors=leftEffectors + rightEffectors + (camera,),
            perception_calibrations=(PerceptionCalibrationBinding(
                component_id="camera",
                calibration_id="temporary_camera_projection",
                frame_id="camera_optical_pivot",
                projection_matrix=(
                    (384.0, 0.0, 255.5),
                    (0.0, 384.0, 255.5),
                    (0.0, 0.0, 1.0),
                ),
                reference_size=(512, 512),
                primary=True),))
        return definition


    @staticmethod
    def CompileTemporary() -> RobotEmbodimentContract:
        return RobotMorphologyModule.CompileDefinition(
            RobotMorphologyModule.TemporaryDefinition())


    @staticmethod
    def CompileActive() -> RobotEmbodimentContract:
        return RobotMorphologyModule.CompileTemporary()


@dataclass(frozen=True)
class RawJointFeedback:
    position: torch.Tensor
    velocity: torch.Tensor
    contract_id: str
    timestamp: torch.Tensor
    sample_index: torch.Tensor

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("raw feedback validation requires a contract view")
        if self.contract_id != contractView.contract_id:
            raise ValueError("raw feedback contract identity does not match")
        if not torch.is_tensor(self.position) or self.position.dim() != 2:
            raise ValueError("joint position must be a batched matrix")
        batchSize = int(self.position.size(0))
        expected = (batchSize, contractView.joint_count)
        RobotMorphologyModule.ValidateTensor(self.position, expected, "joint position", True)
        RobotMorphologyModule.ValidateTensor(self.velocity, expected, "joint velocity", True)
        if self.velocity.device != self.position.device:
            raise ValueError("joint position and velocity must share a device")
        if self.velocity.dtype != self.position.dtype:
            raise ValueError("joint position and velocity must share a dtype")
        RobotMorphologyModule.ValidateTensor(
            self.timestamp,
            (batchSize,),
            "feedback timestamp",
            True)
        if self.timestamp.device != self.position.device:
            raise ValueError("feedback timestamp must share the feedback device")
        if self.timestamp.dtype != self.position.dtype:
            raise ValueError("feedback timestamp must share the feedback dtype")
        if bool((self.timestamp < 0.0).any().item()):
            raise ValueError("feedback timestamp cannot be negative")
        if (
            not torch.is_tensor(self.sample_index)
            or tuple(self.sample_index.shape) != (batchSize,)
            or self.sample_index.dtype != torch.long
            or self.sample_index.device != self.position.device
            or bool((self.sample_index < 0).any().item())
        ):
            raise ValueError("sample_index must be a non-negative long vector")
        lower = torch.tensor(
            contractView.joint_lower,
            device=self.position.device,
            dtype=self.position.dtype).unsqueeze(0)
        upper = torch.tensor(
            contractView.joint_upper,
            device=self.position.device,
            dtype=self.position.dtype).unsqueeze(0)
        periodic = torch.tensor(
            contractView.joint_periodic,
            device=self.position.device,
            dtype=torch.bool).unsqueeze(0)
        tolerance = 16.0 * torch.finfo(self.position.dtype).eps
        outside = (~periodic) & (
            (self.position < lower - tolerance)
            | (self.position > upper + tolerance))
        if bool(outside.any().item()):
            raise ValueError("joint position exceeds the morphology limits")
        velocityLimit = torch.tensor(
            contractView.joint_velocity_limit,
            device=self.velocity.device,
            dtype=self.velocity.dtype).unsqueeze(0)
        if bool((self.velocity.abs() > velocityLimit + tolerance).any().item()):
            raise ValueError("joint velocity exceeds the morphology limits")


@dataclass(frozen=True)
class ExpandedEndEffectorTarget:
    translation: torch.Tensor
    rotation_vector: torch.Tensor
    translation_active: torch.Tensor
    rotation_active: torch.Tensor
    active: torch.Tensor
    contract_id: str
    model_signature: str
    target_version: torch.Tensor
    timestamp: torch.Tensor


@dataclass(frozen=True)
class PackedEndEffectorTarget:
    values: torch.Tensor
    active: torch.Tensor
    contract_id: str
    model_signature: str
    target_version: torch.Tensor
    timestamp: torch.Tensor

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("target validation requires a contract view")
        if self.contract_id != contractView.contract_id:
            raise ValueError("target contract identity does not match")
        if self.model_signature != contractView.model_signature:
            raise ValueError("target model signature does not match")
        if not torch.is_tensor(self.values) or self.values.dim() != 2:
            raise ValueError("packed endpoint targets must be a batched matrix")
        batchSize = int(self.values.size(0))
        RobotMorphologyModule.ValidateTensor(
            self.values,
            (batchSize, contractView.end_effector_target_layout.PackedDim),
            "packed endpoint targets",
            True)
        RobotMorphologyModule.ValidateTensor(
            self.active,
            (batchSize, contractView.end_effector_count),
            "endpoint active mask",
            False)
        if self.active.device != self.values.device:
            raise ValueError("target active mask must share the target device")
        lower = torch.tensor(
            contractView.end_effector_target_lower,
            device=self.values.device,
            dtype=self.values.dtype).unsqueeze(0)
        upper = torch.tensor(
            contractView.end_effector_target_upper,
            device=self.values.device,
            dtype=self.values.dtype).unsqueeze(0)
        tolerance = 16.0 * torch.finfo(self.values.dtype).eps
        for endpointIndex in range(contractView.end_effector_count):
            targetSlice = contractView.end_effector_target_layout.Slice(
                endpointIndex)
            active = self.active[:, endpointIndex]
            if bool((
                active.unsqueeze(-1)
                & (
                    (self.values[:, targetSlice] < lower[:, targetSlice] - tolerance)
                    | (self.values[:, targetSlice] > upper[:, targetSlice] + tolerance)
                )
            ).any().item()):
                raise ValueError("active endpoint targets exceed morphology limits")
        for endpointIndex, parentIndex in enumerate(contractView.parent_index):
            if parentIndex >= 0 and bool((
                self.active[:, endpointIndex]
                & ~self.active[:, parentIndex]
            ).any().item()):
                raise ValueError("active child targets require an active parent target")
        if (
            not torch.is_tensor(self.target_version)
            or tuple(self.target_version.shape) != (batchSize,)
            or self.target_version.dtype != torch.long
            or self.target_version.device != self.values.device
            or bool((self.target_version < 0).any().item())
        ):
            raise ValueError("target_version must be a non-negative long vector")
        RobotMorphologyModule.ValidateTensor(
            self.timestamp,
            (batchSize,),
            "target timestamp",
            True)
        if self.timestamp.device != self.values.device:
            raise ValueError("target timestamp must share the target device")
        if self.timestamp.dtype != self.values.dtype:
            raise ValueError("target timestamp must share the target dtype")
        if bool((self.timestamp < 0.0).any().item()):
            raise ValueError("target timestamp cannot be negative")

    def Expand(
        self,
        contractView: RobotEmbodimentContractView,
    ) -> ExpandedEndEffectorTarget:
        self.Validate(contractView)
        return self.ExpandValidated(contractView)

    def ExpandValidated(
        self,
        contractView: RobotEmbodimentContractView,
    ) -> ExpandedEndEffectorTarget:
        batchSize = int(self.values.size(0))
        translations = []
        rotationVectors = []
        translationActive = []
        rotationActive = []
        for endpointIndex in range(contractView.end_effector_count):
            targetSlice = contractView.end_effector_target_layout.Slice(
                endpointIndex)
            coordinates = torch.where(
                self.active[:, endpointIndex].unsqueeze(-1),
                self.values[:, targetSlice],
                torch.zeros_like(self.values[:, targetSlice]))
            translationBasis = contractView.end_effector_translation_basis.Matrix(
                endpointIndex,
                device=self.values.device,
                dtype=self.values.dtype)
            rotationBasis = contractView.end_effector_rotation_basis.Matrix(
                endpointIndex,
                device=self.values.device,
                dtype=self.values.dtype)
            translationWidth = int(translationBasis.size(1))
            rotationWidth = int(rotationBasis.size(1))
            translationCoordinates = coordinates[:, :translationWidth]
            rotationCoordinates = coordinates[
                :,
                translationWidth:translationWidth + rotationWidth]
            translations.append(
                translationCoordinates @ translationBasis.transpose(0, 1))
            rotationVectors.append(
                rotationCoordinates @ rotationBasis.transpose(0, 1))
            translationActive.append(translationWidth > 0)
            rotationActive.append(rotationWidth > 0)
        return ExpandedEndEffectorTarget(
            translation=torch.stack(translations, dim=1),
            rotation_vector=torch.stack(rotationVectors, dim=1),
            translation_active=torch.tensor(
                translationActive,
                device=self.values.device,
                dtype=torch.bool).unsqueeze(0).expand(batchSize, -1),
            rotation_active=torch.tensor(
                rotationActive,
                device=self.values.device,
                dtype=torch.bool).unsqueeze(0).expand(batchSize, -1),
            active=self.active,
            contract_id=self.contract_id,
            model_signature=self.model_signature,
            target_version=self.target_version,
            timestamp=self.timestamp)


@dataclass(frozen=True)
class BrainFeedbackPacket:
    joint_features: torch.Tensor
    joint_valid: torch.Tensor
    endpoint_valid: torch.Tensor
    progress: torch.Tensor
    reached: torch.Tensor
    child_enabled: torch.Tensor
    target_active: torch.Tensor
    target_version: torch.Tensor
    perception_rotation: torch.Tensor
    perception_rotation_delta: torch.Tensor
    perception_angular_velocity: torch.Tensor
    perception_valid: torch.Tensor
    contract_id: str
    model_signature: str
    timestamp: torch.Tensor
    sample_index: torch.Tensor

    @property
    def values(self) -> torch.Tensor:
        return self.joint_features

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("feedback validation requires a contract view")
        if self.contract_id != contractView.contract_id:
            raise ValueError("feedback contract identity does not match")
        if self.model_signature != contractView.model_signature:
            raise ValueError("feedback model signature does not match")
        if not torch.is_tensor(self.joint_features) or self.joint_features.dim() != 2:
            raise ValueError("joint features must be a batched matrix")
        batchSize = int(self.joint_features.size(0))
        device = self.joint_features.device
        RobotMorphologyModule.ValidateTensor(
            self.joint_features,
            (batchSize, contractView.joint_feedback_layout.PackedDim),
            "joint features",
            True)
        for value, shape, name in (
            (
                self.joint_valid,
                (batchSize, contractView.joint_count),
                "joint_valid"),
            (
                self.endpoint_valid,
                (batchSize, contractView.end_effector_count),
                "endpoint_valid"),
            (
                self.reached,
                (batchSize, contractView.end_effector_count),
                "reached"),
            (
                self.child_enabled,
                (batchSize, contractView.end_effector_count),
                "child_enabled"),
            (
                self.target_active,
                (batchSize, contractView.end_effector_count),
                "target_active"),
            (
                self.perception_valid,
                (batchSize, len(contractView.perception_view.indices)),
                "perception_valid"),
        ):
            RobotMorphologyModule.ValidateTensor(value, shape, name, False)
            if value.device != device:
                raise ValueError(f"{name} must share the packet device")
        RobotMorphologyModule.ValidateTensor(
            self.progress,
            (batchSize, contractView.end_effector_count),
            "progress",
            True)
        RobotMorphologyModule.ValidateTensor(
            self.perception_rotation,
            (batchSize, len(contractView.perception_view.indices), 4),
            "perception_rotation",
            True)
        RobotMorphologyModule.ValidateTensor(
            self.perception_rotation_delta,
            (batchSize, len(contractView.perception_view.indices), 4),
            "perception_rotation_delta",
            True)
        RobotMorphologyModule.ValidateTensor(
            self.perception_angular_velocity,
            (batchSize, len(contractView.perception_view.indices), 3),
            "perception_angular_velocity",
            True)
        for value in (
            self.progress,
            self.perception_rotation,
            self.perception_rotation_delta,
            self.perception_angular_velocity,
        ):
            if value.device != device:
                raise ValueError("feedback tensors must share one device")
            if value.dtype != self.joint_features.dtype:
                raise ValueError("feedback tensors must share one dtype")
        for quaternion, fieldName in (
            (self.perception_rotation, "perception rotations"),
            (self.perception_rotation_delta, "perception rotation deltas"),
        ):
            quaternionNorm = torch.linalg.vector_norm(
                quaternion,
                dim=-1)
            if quaternionNorm.numel() > 0 and not bool(
                torch.isclose(
                    quaternionNorm,
                    torch.ones_like(quaternionNorm),
                    atol=1e-5,
                    rtol=1e-5).all().item()
            ):
                raise ValueError(f"{fieldName} must be unit quaternions")
        if bool(((self.progress < 0.0) | (self.progress > 1.0)).any().item()):
            raise ValueError("hierarchy progress must lie in the unit interval")
        if (
            not torch.is_tensor(self.target_version)
            or tuple(self.target_version.shape) != (batchSize,)
            or self.target_version.dtype != torch.long
            or self.target_version.device != device
            or bool((self.target_version < -1).any().item())
        ):
            raise ValueError("target_version must be a long vector greater than or equal to minus one")
        RobotMorphologyModule.ValidateTensor(
            self.timestamp,
            (batchSize,),
            "feedback timestamp",
            True)
        if self.timestamp.device != device:
            raise ValueError("feedback timestamp must share the packet device")
        if self.timestamp.dtype != self.joint_features.dtype:
            raise ValueError("feedback timestamp must share the packet dtype")
        if (
            not torch.is_tensor(self.sample_index)
            or tuple(self.sample_index.shape) != (batchSize,)
            or self.sample_index.dtype != torch.long
            or self.sample_index.device != device
            or bool((self.sample_index < 0).any().item())
        ):
            raise ValueError("sample_index must be a non-negative long vector")

    def IndexSelectRows(self, rowIndex: torch.Tensor) -> "BrainFeedbackPacket":
        if (
            not torch.is_tensor(rowIndex)
            or rowIndex.dim() != 1
            or rowIndex.dtype != torch.long
            or rowIndex.device != self.joint_features.device
        ):
            raise ValueError("feedback row indices must be a compatible long vector")
        return BrainFeedbackPacket(
            joint_features=self.joint_features.index_select(0, rowIndex),
            joint_valid=self.joint_valid.index_select(0, rowIndex),
            endpoint_valid=self.endpoint_valid.index_select(0, rowIndex),
            progress=self.progress.index_select(0, rowIndex),
            reached=self.reached.index_select(0, rowIndex),
            child_enabled=self.child_enabled.index_select(0, rowIndex),
            target_active=self.target_active.index_select(0, rowIndex),
            target_version=self.target_version.index_select(0, rowIndex),
            perception_rotation=(
                self.perception_rotation.index_select(0, rowIndex)),
            perception_rotation_delta=(
                self.perception_rotation_delta.index_select(0, rowIndex)),
            perception_angular_velocity=(
                self.perception_angular_velocity.index_select(0, rowIndex)),
            perception_valid=self.perception_valid.index_select(0, rowIndex),
            contract_id=self.contract_id,
            model_signature=self.model_signature,
            timestamp=self.timestamp.index_select(0, rowIndex),
            sample_index=self.sample_index.index_select(0, rowIndex))

    def RepeatCandidates(self, candidateCount: int) -> "BrainFeedbackPacket":
        if type(candidateCount) is not int or candidateCount < 1:
            raise ValueError("candidateCount must be a positive integer")
        batchSize = int(self.joint_features.size(0))
        rowIndex = torch.arange(
            batchSize,
            device=self.joint_features.device,
            dtype=torch.long).repeat_interleave(candidateCount)
        return self.IndexSelectRows(rowIndex)


@dataclass(frozen=True)
class HierarchyProgress:
    progress: torch.Tensor
    reached: torch.Tensor
    child_enabled: torch.Tensor


@dataclass(frozen=True)
class EndEffectorPose:
    translation: torch.Tensor
    rotation: torch.Tensor


class RobotEmbodimentRuntime:
    @staticmethod
    def QuaternionToMatrix(quaternion: torch.Tensor) -> torch.Tensor:
        normalized = quaternion / torch.linalg.vector_norm(
            quaternion,
            dim=-1,
            keepdim=True).clamp_min(torch.finfo(quaternion.dtype).eps)
        x, y, z, w = normalized.unbind(dim=-1)
        return torch.stack((
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ), dim=-1).reshape(quaternion.shape[:-1] + (3, 3))


    @staticmethod
    def RotationVectorToMatrix(rotationVector: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.vector_norm(rotationVector, dim=-1, keepdim=True)
        epsilon = torch.finfo(rotationVector.dtype).eps
        axis = rotationVector / angle.clamp_min(epsilon)
        x, y, z = axis.unbind(dim=-1)
        zero = torch.zeros_like(x)
        skew = torch.stack((
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ), dim=-1).reshape(rotationVector.shape[:-1] + (3, 3))
        identity = torch.eye(
            3,
            device=rotationVector.device,
            dtype=rotationVector.dtype).expand(rotationVector.shape[:-1] + (3, 3))
        sinAngle = torch.sin(angle).unsqueeze(-1)
        oneMinusCosine = (1.0 - torch.cos(angle)).unsqueeze(-1)
        matrix = identity + sinAngle * skew + oneMinusCosine * (skew @ skew)
        small = angle.squeeze(-1) < math.sqrt(epsilon)
        smallVector = rotationVector
        sx, sy, sz = smallVector.unbind(dim=-1)
        smallSkew = torch.stack((
            zero, -sz, sy,
            sz, zero, -sx,
            -sy, sx, zero,
        ), dim=-1).reshape(rotationVector.shape[:-1] + (3, 3))
        smallMatrix = identity + smallSkew + 0.5 * (smallSkew @ smallSkew)
        return torch.where(small.unsqueeze(-1).unsqueeze(-1), smallMatrix, matrix)


    @staticmethod
    def MatrixToRotationVector(matrix: torch.Tensor) -> torch.Tensor:
        if (
            not torch.is_tensor(matrix)
            or matrix.dim() < 2
            or tuple(matrix.shape[-2:]) != (3, 3)
            or not matrix.is_floating_point()
            or not bool(torch.isfinite(matrix).all().item())
        ):
            raise ValueError("rotation logarithm requires finite three by three matrices")
        epsilon = torch.finfo(matrix.dtype).eps
        trace = matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        angle = torch.acos(cosine)
        vector = torch.stack((
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ), dim=-1)
        sine = torch.sin(angle)
        scale = angle / (2.0 * sine).clamp_min(epsilon)
        result = vector * scale.unsqueeze(-1)
        small = angle < math.sqrt(epsilon)
        smallResult = 0.5 * vector
        diagonal = matrix.diagonal(dim1=-2, dim2=-1)
        root = torch.sqrt(((diagonal + 1.0) * 0.5).clamp_min(0.0))
        denominator = (4.0 * root).clamp_min(math.sqrt(epsilon))
        symmetricXy = matrix[..., 0, 1] + matrix[..., 1, 0]
        symmetricXz = matrix[..., 0, 2] + matrix[..., 2, 0]
        symmetricYz = matrix[..., 1, 2] + matrix[..., 2, 1]
        candidateX = torch.stack((
            root[..., 0],
            symmetricXy / denominator[..., 0],
            symmetricXz / denominator[..., 0],
        ), dim=-1)
        candidateY = torch.stack((
            symmetricXy / denominator[..., 1],
            root[..., 1],
            symmetricYz / denominator[..., 1],
        ), dim=-1)
        candidateZ = torch.stack((
            symmetricXz / denominator[..., 2],
            symmetricYz / denominator[..., 2],
            root[..., 2],
        ), dim=-1)
        candidates = torch.stack((candidateX, candidateY, candidateZ), dim=-2)
        largest = diagonal.argmax(dim=-1)
        gatherIndex = largest.unsqueeze(-1).unsqueeze(-1).expand(
            largest.shape + (1, 3))
        axis = candidates.gather(-2, gatherIndex).squeeze(-2)
        axis = axis / torch.linalg.vector_norm(
            axis,
            dim=-1,
            keepdim=True).clamp_min(math.sqrt(epsilon))
        orientation = (axis * vector).sum(dim=-1, keepdim=True)
        axis = torch.where(orientation < 0.0, -axis, axis)
        nearPiResult = axis * angle.unsqueeze(-1)
        nearPi = angle > math.pi - max(1e-4, 32.0 * epsilon)
        result = torch.where(small.unsqueeze(-1), smallResult, result)
        return torch.where(nearPi.unsqueeze(-1), nearPiResult, result)


    @staticmethod
    def RotationVectorToQuaternion(rotationVector: torch.Tensor) -> torch.Tensor:
        angle = torch.linalg.vector_norm(rotationVector, dim=-1, keepdim=True)
        half = 0.5 * angle
        scale = torch.sin(half) / angle.clamp_min(torch.finfo(rotationVector.dtype).eps)
        small = angle < math.sqrt(torch.finfo(rotationVector.dtype).eps)
        scale = torch.where(small, 0.5 - angle * angle / 48.0, scale)
        quaternion = torch.cat((rotationVector * scale, torch.cos(half)), dim=-1)
        return quaternion / torch.linalg.vector_norm(
            quaternion,
            dim=-1,
            keepdim=True).clamp_min(torch.finfo(rotationVector.dtype).eps)

    def __init__(self, contract: RobotEmbodimentContract) -> None:
        if type(contract) is not RobotEmbodimentContract:
            raise TypeError("runtime requires a RobotEmbodimentContract")
        self.Contract = contract
        self.ContractView = contract.view
        self.CachedTargetValues: Optional[torch.Tensor] = None
        self.CachedTargetActive: Optional[torch.Tensor] = None
        self.CachedTargetVersion: Optional[torch.Tensor] = None
        self.CachedTargetTimestamp: Optional[torch.Tensor] = None
        self.DwellState: Optional[torch.Tensor] = None
        self.ReachedState: Optional[torch.Tensor] = None
        self.LastTimestamp: Optional[torch.Tensor] = None
        self.LastSampleIndex: Optional[torch.Tensor] = None
        self.LastPerceptionRotation: Optional[torch.Tensor] = None
        self.KinematicTensorCache = {}

    def ResetRuntimeState(self) -> None:
        self.CachedTargetValues = None
        self.CachedTargetActive = None
        self.CachedTargetVersion = None
        self.CachedTargetTimestamp = None
        self.DwellState = None
        self.ReachedState = None
        self.LastTimestamp = None
        self.LastSampleIndex = None
        self.LastPerceptionRotation = None

    def BuildNeutralFeedback(
        self,
        timestamp: float,
        sampleIndex: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RawJointFeedback:
        if not math.isfinite(float(timestamp)) or float(timestamp) < 0.0:
            raise ValueError("neutral feedback timestamp must be non-negative")
        if type(sampleIndex) is not int or sampleIndex < 0:
            raise ValueError("neutral feedback sample index must be non-negative")
        position = torch.tensor([
            0.0 if periodic else 0.5 * (lower + upper)
            for lower, upper, periodic in zip(
                self.ContractView.joint_lower,
                self.ContractView.joint_upper,
                self.ContractView.joint_periodic)
        ], device=device, dtype=dtype).unsqueeze(0)
        return RawJointFeedback(
            position=position,
            velocity=torch.zeros_like(position),
            contract_id=self.ContractView.contract_id,
            timestamp=torch.tensor(
                [float(timestamp)],
                device=device,
                dtype=dtype),
            sample_index=torch.tensor(
                [sampleIndex],
                device=device,
                dtype=torch.long))

    def EncodeTargetPayload(
        self,
        target: PackedEndEffectorTarget,
    ) -> Mapping[str, Any]:
        expanded = target.Expand(self.ContractView)
        batchSize = int(target.values.size(0))
        translation = []
        rotation = []
        for batchIndex in range(batchSize):
            translationRow = []
            rotationRow = []
            for endpointIndex in range(self.ContractView.end_effector_count):
                translationRow.append(
                    expanded.translation[batchIndex, endpointIndex]
                    .detach().cpu().tolist()
                    if bool((
                        expanded.translation_active[batchIndex, endpointIndex]
                        & target.active[batchIndex, endpointIndex]
                    ).item())
                    else None)
                rotationRow.append(
                    expanded.rotation_vector[batchIndex, endpointIndex]
                    .detach().cpu().tolist()
                    if bool((
                        expanded.rotation_active[batchIndex, endpointIndex]
                        & target.active[batchIndex, endpointIndex]
                    ).item())
                    else None)
            translation.append(translationRow)
            rotation.append(rotationRow)
        return {
            "contract_id": target.contract_id,
            "model_signature": target.model_signature,
            "target_version": target.target_version.detach().cpu().tolist(),
            "timestamp": target.timestamp.detach().cpu().tolist(),
            "slot_ids": list(self.Contract.end_effector_ids),
            "reference_frame_ids": list(
                self.Contract.end_effector_reference_frame),
            "active": target.active.detach().cpu().tolist(),
            "translation": translation,
            "rotation_vector": rotation,
        }

    def ValidateTemporalOrder(self, rawFeedback: RawJointFeedback) -> None:
        if self.LastTimestamp is None:
            return
        if (
            tuple(self.LastTimestamp.shape) != tuple(rawFeedback.timestamp.shape)
            or self.LastTimestamp.device != rawFeedback.timestamp.device
        ):
            raise ValueError("runtime feedback batch identity changed without reset")
        if bool((rawFeedback.timestamp <= self.LastTimestamp).any().item()):
            raise ValueError("feedback timestamps must increase monotonically")
        if self.LastSampleIndex is None or bool(
            (rawFeedback.sample_index <= self.LastSampleIndex).any().item()
        ):
            raise ValueError("feedback sample indices must increase monotonically")

    def EncodeJointFeatures(self, rawFeedback: RawJointFeedback) -> torch.Tensor:
        view = self.ContractView
        features = []
        for jointIndex in range(view.joint_count):
            position = rawFeedback.position[:, jointIndex]
            velocity = rawFeedback.velocity[:, jointIndex]
            normalizedVelocity = velocity / float(
                view.joint_velocity_limit[jointIndex])
            if view.joint_periodic[jointIndex]:
                token = torch.stack((
                    torch.sin(position),
                    torch.cos(position),
                    normalizedVelocity,
                ), dim=-1)
            else:
                lower = float(view.joint_lower[jointIndex])
                upper = float(view.joint_upper[jointIndex])
                normalizedPosition = (
                    2.0 * (position - lower) / (upper - lower) - 1.0)
                if view.joint_rotational[jointIndex]:
                    token = torch.stack((
                        normalizedPosition,
                        torch.sin(position),
                        torch.cos(position),
                        normalizedVelocity,
                    ), dim=-1)
                else:
                    token = torch.stack((
                        normalizedPosition,
                        normalizedVelocity,
                    ), dim=-1)
            features.append(token)
        return torch.cat(features, dim=-1)

    def GetKinematicTensors(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, ...]:
        key = (device.type, device.index, dtype)
        cached = self.KinematicTensorCache.get(key)
        if cached is not None:
            return cached
        view = self.ContractView
        jointTranslationAxis = torch.tensor(
            view.joint_translation_basis.values,
            device=device,
            dtype=dtype).reshape(view.joint_count, 3)
        jointRotationAxis = torch.tensor(
            view.joint_rotation_basis.values,
            device=device,
            dtype=dtype).reshape(view.joint_count, 3)
        jointOriginTranslation = torch.tensor(
            self.Contract.joint_origin_translation,
            device=device,
            dtype=dtype)
        jointOriginRotation = RobotEmbodimentRuntime.QuaternionToMatrix(torch.tensor(
            self.Contract.joint_origin_quaternion_xyzw,
            device=device,
            dtype=dtype))
        endpointTerminalTranslation = torch.tensor(
            self.Contract.end_effector_terminal_translation,
            device=device,
            dtype=dtype)
        endpointTerminalRotation = RobotEmbodimentRuntime.QuaternionToMatrix(torch.tensor(
            self.Contract.end_effector_terminal_quaternion_xyzw,
            device=device,
            dtype=dtype))
        endpointJointIndex = torch.tensor(tuple(
            view.end_effector_joint_chain_indices[
                view.end_effector_joint_chain_offsets[index + 1] - 1]
            for index in range(view.end_effector_count)
        ), device=device, dtype=torch.long)
        endpointIndexById = {
            identifier: index
            for index, identifier in enumerate(self.Contract.end_effector_ids)
        }
        referenceIndex = torch.tensor(tuple(
            -1
            if reference == "world"
            else endpointIndexById[reference]
            for reference in self.Contract.end_effector_reference_frame
        ), device=device, dtype=torch.long)
        perceptionIndex = torch.tensor(
            view.perception_view.indices,
            device=device,
            dtype=torch.long)
        cached = (
            jointTranslationAxis,
            jointRotationAxis,
            jointOriginTranslation,
            jointOriginRotation,
            endpointTerminalTranslation,
            endpointTerminalRotation,
            endpointJointIndex,
            referenceIndex,
            perceptionIndex,
        )
        self.KinematicTensorCache[key] = cached
        return cached

    def ForwardKinematics(self, position: torch.Tensor) -> EndEffectorPose:
        if (
            not torch.is_tensor(position)
            or position.dim() != 2
            or int(position.size(1)) != self.ContractView.joint_count
            or not position.is_floating_point()
            or not bool(torch.isfinite(position).all().item())
        ):
            raise ValueError("forward kinematics requires finite batched joint positions")
        batchSize = int(position.size(0))
        device = position.device
        dtype = position.dtype
        (
            jointTranslationAxis,
            jointRotationAxis,
            jointOriginTranslation,
            jointOriginRotation,
            endpointTerminalTranslation,
            endpointTerminalRotation,
            endpointJointIndex,
            referenceIndex,
            _,
        ) = self.GetKinematicTensors(device, dtype)
        identity = torch.eye(
            3,
            device=device,
            dtype=dtype).unsqueeze(0).expand(batchSize, -1, -1)
        zero = torch.zeros(batchSize, 3, device=device, dtype=dtype)
        jointTranslations = []
        jointRotations = []
        for jointIndex, parentIndex in enumerate(self.Contract.joint_parent_index):
            parentTranslation = (
                zero
                if parentIndex < 0
                else jointTranslations[parentIndex])
            parentRotation = (
                identity
                if parentIndex < 0
                else jointRotations[parentIndex])
            originTranslation = parentTranslation + torch.matmul(
                parentRotation,
                jointOriginTranslation[jointIndex]
                .reshape(1, 3, 1)
                .expand(batchSize, -1, -1)).squeeze(-1)
            originRotation = (
                parentRotation
                @ jointOriginRotation[jointIndex]
                .unsqueeze(0)
                .expand(batchSize, -1, -1))
            coordinate = position[:, jointIndex].unsqueeze(-1)
            translation = originTranslation + torch.matmul(
                originRotation,
                (coordinate * jointTranslationAxis[jointIndex])
                .unsqueeze(-1)).squeeze(-1)
            rotation = originRotation @ RobotEmbodimentRuntime.RotationVectorToMatrix(
                coordinate * jointRotationAxis[jointIndex])
            jointTranslations.append(translation)
            jointRotations.append(rotation)
        jointTranslation = torch.stack(jointTranslations, dim=1)
        jointRotation = torch.stack(jointRotations, dim=1)
        worldTranslation = jointTranslation.index_select(
            1,
            endpointJointIndex)
        worldRotation = jointRotation.index_select(1, endpointJointIndex)
        worldTranslation = worldTranslation + torch.matmul(
            worldRotation,
            endpointTerminalTranslation.unsqueeze(0).unsqueeze(-1),
        ).squeeze(-1)
        worldRotation = (
            worldRotation
            @ endpointTerminalRotation.unsqueeze(0))
        worldReference = referenceIndex.lt(0)
        safeReferenceIndex = referenceIndex.clamp_min(0)
        referenceTranslation = worldTranslation.index_select(
            1,
            safeReferenceIndex)
        referenceRotation = worldRotation.index_select(
            1,
            safeReferenceIndex)
        referenceTranslation = torch.where(
            worldReference.reshape(1, -1, 1),
            torch.zeros_like(referenceTranslation),
            referenceTranslation)
        referenceRotation = torch.where(
            worldReference.reshape(1, -1, 1, 1),
            torch.eye(3, device=device, dtype=dtype).reshape(1, 1, 3, 3),
            referenceRotation)
        outputRotation = referenceRotation.transpose(-1, -2) @ worldRotation
        outputTranslation = torch.matmul(
            referenceRotation.transpose(-1, -2),
            (worldTranslation - referenceTranslation).unsqueeze(-1),
        ).squeeze(-1)
        return EndEffectorPose(
            translation=outputTranslation,
            rotation=outputRotation)

    def AllocateTargetState(
        self,
        target: PackedEndEffectorTarget,
    ) -> None:
        batchSize = int(target.values.size(0))
        endpointCount = self.ContractView.end_effector_count
        self.CachedTargetValues = torch.zeros_like(target.values)
        self.CachedTargetActive = torch.zeros(
            batchSize,
            endpointCount,
            device=target.values.device,
            dtype=torch.bool)
        self.CachedTargetVersion = torch.full(
            (batchSize,),
            -1,
            device=target.values.device,
            dtype=torch.long)
        self.CachedTargetTimestamp = torch.zeros_like(target.timestamp)
        self.DwellState = torch.zeros(
            batchSize,
            endpointCount,
            device=target.values.device,
            dtype=torch.long)
        self.ReachedState = torch.zeros(
            batchSize,
            endpointCount,
            device=target.values.device,
            dtype=torch.bool)

    def SetDispatchedTargets(
        self,
        target: PackedEndEffectorTarget,
        dispatchedMask: Optional[torch.Tensor] = None,
    ) -> None:
        target.Validate(self.ContractView)
        batchSize = int(target.values.size(0))
        if dispatchedMask is None:
            dispatched = torch.ones(
                batchSize,
                device=target.values.device,
                dtype=torch.bool)
        else:
            RobotMorphologyModule.ValidateTensor(
                dispatchedMask,
                (batchSize,),
                "dispatched mask",
                False)
            if dispatchedMask.device != target.values.device:
                raise ValueError("dispatched mask must share the target device")
            dispatched = dispatchedMask
        if not bool(dispatched.any().item()):
            return
        canonicalTargetValues = torch.zeros_like(target.values)
        for endpointIndex in range(self.ContractView.end_effector_count):
            targetSlice = self.ContractView.end_effector_target_layout.Slice(
                endpointIndex)
            canonicalTargetValues[:, targetSlice] = torch.where(
                target.active[:, endpointIndex].unsqueeze(-1),
                target.values[:, targetSlice],
                torch.zeros_like(target.values[:, targetSlice]))
        if self.CachedTargetValues is None:
            self.AllocateTargetState(target)
        if (
            tuple(self.CachedTargetValues.shape) != tuple(target.values.shape)
            or self.CachedTargetValues.device != target.values.device
            or self.CachedTargetValues.dtype != target.values.dtype
        ):
            raise ValueError("target batch identity changed without runtime reset")
        currentVersion = self.CachedTargetVersion
        if bool((dispatched & (target.target_version < currentVersion)).any().item()):
            raise ValueError("dispatched target versions cannot move backwards")
        sameVersion = dispatched & target.target_version.eq(currentVersion)
        if bool(sameVersion.any().item()):
            rows = torch.nonzero(sameVersion, as_tuple=False).squeeze(-1)
            if not bool(torch.equal(
                canonicalTargetValues.index_select(0, rows),
                self.CachedTargetValues.index_select(0, rows))
            ) or not bool(torch.equal(
                target.active.index_select(0, rows),
                self.CachedTargetActive.index_select(0, rows))
            ):
                raise ValueError("one target version cannot identify different targets")
        changedSlots = torch.zeros_like(self.CachedTargetActive)
        for endpointIndex in range(self.ContractView.end_effector_count):
            targetSlice = self.ContractView.end_effector_target_layout.Slice(
                endpointIndex)
            changedSlots[:, endpointIndex] = dispatched & (
                target.active[:, endpointIndex].ne(
                    self.CachedTargetActive[:, endpointIndex])
                | canonicalTargetValues[:, targetSlice].ne(
                    self.CachedTargetValues[:, targetSlice]).any(dim=-1))
        resetSlots = changedSlots.clone()
        for endpointIndex in range(self.ContractView.end_effector_count):
            start = self.ContractView.subtree_offsets[endpointIndex]
            end = self.ContractView.subtree_offsets[endpointIndex + 1]
            subtree = self.ContractView.subtree_indices[start:end]
            resetSlots[:, list(subtree)] |= changedSlots[
                :, endpointIndex].unsqueeze(-1)
        rows = torch.nonzero(dispatched, as_tuple=False).squeeze(-1)
        self.CachedTargetValues.index_copy_(
            0,
            rows,
            canonicalTargetValues.index_select(0, rows).detach())
        self.CachedTargetActive.index_copy_(
            0,
            rows,
            target.active.index_select(0, rows).detach())
        self.CachedTargetVersion.index_copy_(
            0,
            rows,
            target.target_version.index_select(0, rows).detach())
        self.CachedTargetTimestamp.index_copy_(
            0,
            rows,
            target.timestamp.index_select(0, rows).detach())
        self.DwellState.masked_fill_(resetSlots, 0)
        self.ReachedState.masked_fill_(resetSlots, False)

    def EvaluateHierarchy(
        self,
        pose: EndEffectorPose,
        endpointValid: torch.Tensor,
    ) -> HierarchyProgress:
        batchSize = int(pose.translation.size(0))
        endpointCount = self.ContractView.end_effector_count
        device = pose.translation.device
        dtype = pose.translation.dtype
        RobotMorphologyModule.ValidateTensor(
            endpointValid,
            (batchSize, endpointCount),
            "endpointValid",
            False)
        progress = torch.zeros(
            batchSize,
            endpointCount,
            device=device,
            dtype=dtype)
        reached = torch.zeros(
            batchSize,
            endpointCount,
            device=device,
            dtype=torch.bool)
        enabled = torch.zeros_like(reached)
        rootMask = torch.tensor(
            self.ContractView.root_mask,
            device=device,
            dtype=torch.bool).unsqueeze(0).expand(batchSize, -1)
        if self.CachedTargetValues is None:
            enabled = rootMask & endpointValid
            return HierarchyProgress(
                progress=progress,
                reached=reached,
                child_enabled=enabled)
        if (
            tuple(self.CachedTargetValues.shape[:1]) != (batchSize,)
            or self.CachedTargetValues.device != device
            or self.CachedTargetValues.dtype != dtype
        ):
            raise ValueError("cached targets do not match feedback batch identity")
        target = PackedEndEffectorTarget(
            values=self.CachedTargetValues,
            active=self.CachedTargetActive,
            contract_id=self.ContractView.contract_id,
            model_signature=self.ContractView.model_signature,
            target_version=self.CachedTargetVersion,
            timestamp=self.CachedTargetTimestamp)
        expanded = target.ExpandValidated(self.ContractView)
        for layer in self.ContractView.topological_layers:
            for endpointIndex in layer:
                parentIndex = self.ContractView.parent_index[endpointIndex]
                if parentIndex < 0:
                    hierarchyEnabled = endpointValid[:, endpointIndex]
                else:
                    hierarchyEnabled = (
                        reached[:, parentIndex]
                        & endpointValid[:, endpointIndex])
                enabled[:, endpointIndex] = hierarchyEnabled
                endpointExecuting = (
                    hierarchyEnabled
                    & self.CachedTargetActive[:, endpointIndex])
                translationWidth = self.ContractView.end_effector_translation_basis.shapes[
                    endpointIndex][1]
                rotationWidth = self.ContractView.end_effector_rotation_basis.shapes[
                    endpointIndex][1]
                errorSquared = torch.zeros(
                    batchSize,
                    device=device,
                    dtype=dtype)
                if translationWidth > 0:
                    translationError = (
                        expanded.translation[:, endpointIndex]
                        - pose.translation[:, endpointIndex])
                    translationScale = float(
                        self.ContractView.translation_error_scale[endpointIndex])
                    errorSquared = errorSquared + (
                        translationError / translationScale).square().sum(dim=-1)
                if rotationWidth > 0:
                    targetRotation = RobotEmbodimentRuntime.RotationVectorToMatrix(
                        expanded.rotation_vector[:, endpointIndex])
                    relativeRotation = targetRotation @ pose.rotation[
                        :, endpointIndex].transpose(-1, -2)
                    rotationError = RobotEmbodimentRuntime.MatrixToRotationVector(relativeRotation)
                    rotationScale = float(
                        self.ContractView.rotation_error_scale[endpointIndex])
                    errorSquared = errorSquared + (
                        rotationError / rotationScale).square().sum(dim=-1)
                error = torch.sqrt(errorSquared)
                enter = float(self.ContractView.progress_enter[endpointIndex])
                exitValue = float(self.ContractView.progress_exit[endpointIndex])
                progress[:, endpointIndex] = torch.where(
                    endpointExecuting,
                    (1.0 + error / exitValue).reciprocal(),
                    torch.zeros_like(error))
                wasReached = self.ReachedState[:, endpointIndex]
                insideEnter = endpointExecuting & error.le(enter)
                remainReached = endpointExecuting & wasReached & error.le(exitValue)
                dwell = torch.where(
                    insideEnter & ~wasReached,
                    self.DwellState[:, endpointIndex] + 1,
                    torch.zeros_like(self.DwellState[:, endpointIndex]))
                newlyReached = dwell.ge(
                    int(self.ContractView.dwell_cycles[endpointIndex]))
                endpointReached = remainReached | newlyReached
                self.DwellState[:, endpointIndex] = dwell
                self.ReachedState[:, endpointIndex] = endpointReached
                reached[:, endpointIndex] = endpointReached
        return HierarchyProgress(
            progress=progress,
            reached=reached,
            child_enabled=enabled)

    def ComputePerceptionMotion(
        self,
        pose: EndEffectorPose,
        timestamp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batchSize = int(pose.rotation.size(0))
        perceptionIndices = self.GetKinematicTensors(
            pose.rotation.device,
            pose.rotation.dtype)[8]
        current = pose.rotation.index_select(1, perceptionIndices)
        perceptionCount = int(current.size(1))
        currentQuaternion = RobotEmbodimentRuntime.RotationVectorToQuaternion(
            RobotEmbodimentRuntime.MatrixToRotationVector(current))
        identityQuaternion = torch.zeros(
            batchSize,
            perceptionCount,
            4,
            device=pose.rotation.device,
            dtype=pose.rotation.dtype)
        identityQuaternion[..., -1] = 1.0
        angularVelocity = torch.zeros(
            batchSize,
            perceptionCount,
            3,
            device=pose.rotation.device,
            dtype=pose.rotation.dtype)
        valid = torch.zeros(
            batchSize,
            perceptionCount,
            device=pose.rotation.device,
            dtype=torch.bool)
        if perceptionCount == 0:
            self.LastPerceptionRotation = current.detach().clone()
            return currentQuaternion, identityQuaternion, angularVelocity, valid
        if self.LastPerceptionRotation is not None:
            deltaMatrix = self.LastPerceptionRotation.transpose(-1, -2) @ current
            rotationVector = RobotEmbodimentRuntime.MatrixToRotationVector(deltaMatrix)
            deltaTime = (timestamp - self.LastTimestamp).reshape(
                batchSize, 1, 1)
            angularVelocity = rotationVector / deltaTime
            identityQuaternion = RobotEmbodimentRuntime.RotationVectorToQuaternion(rotationVector)
            valid = torch.ones_like(valid)
        self.LastPerceptionRotation = current.detach().clone()
        return currentQuaternion, identityQuaternion, angularVelocity, valid

    def EncodeFeedback(
        self,
        rawFeedback: RawJointFeedback,
    ) -> BrainFeedbackPacket:
        if type(rawFeedback) is not RawJointFeedback:
            raise TypeError("runtime feedback must be RawJointFeedback")
        rawFeedback.Validate(self.ContractView)
        self.ValidateTemporalOrder(rawFeedback)
        jointFeatures = self.EncodeJointFeatures(rawFeedback)
        jointValid = torch.ones(
            rawFeedback.position.shape,
            device=rawFeedback.position.device,
            dtype=torch.bool)
        endpointValid = torch.ones(
            int(rawFeedback.position.size(0)),
            self.ContractView.end_effector_count,
            device=rawFeedback.position.device,
            dtype=torch.bool)
        pose = self.ForwardKinematics(rawFeedback.position)
        hierarchy = self.EvaluateHierarchy(pose, endpointValid)
        (
            perceptionRotation,
            perceptionRotationDelta,
            perceptionAngularVelocity,
            perceptionValid,
        ) = self.ComputePerceptionMotion(pose, rawFeedback.timestamp)
        batchSize = int(rawFeedback.position.size(0))
        if self.CachedTargetActive is None:
            targetActive = torch.zeros(
                batchSize,
                self.ContractView.end_effector_count,
                device=rawFeedback.position.device,
                dtype=torch.bool)
            targetVersion = torch.full(
                (batchSize,),
                -1,
                device=rawFeedback.position.device,
                dtype=torch.long)
        else:
            targetActive = self.CachedTargetActive.clone()
            targetVersion = self.CachedTargetVersion.clone()
        packet = BrainFeedbackPacket(
            joint_features=jointFeatures,
            joint_valid=jointValid,
            endpoint_valid=endpointValid,
            progress=hierarchy.progress,
            reached=hierarchy.reached,
            child_enabled=hierarchy.child_enabled,
            target_active=targetActive,
            target_version=targetVersion,
            perception_rotation=perceptionRotation,
            perception_rotation_delta=perceptionRotationDelta,
            perception_angular_velocity=perceptionAngularVelocity,
            perception_valid=perceptionValid,
            contract_id=self.ContractView.contract_id,
            model_signature=self.ContractView.model_signature,
            timestamp=rawFeedback.timestamp,
            sample_index=rawFeedback.sample_index)
        self.LastTimestamp = rawFeedback.timestamp.detach().clone()
        self.LastSampleIndex = rawFeedback.sample_index.detach().clone()
        return packet
