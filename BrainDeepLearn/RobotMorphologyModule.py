from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from hashlib import sha256
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
import json
import math

import torch


class JointType(IntEnum):
    REVOLUTE = 0
    CONTINUOUS = 1
    PRISMATIC = 2


class EndEffectorType(IntEnum):
    WRIST = 0
    FINGERTIP = 1
    TOOL = 2
    SENSOR_ACTUATOR = 3
    OTHER = 4


@dataclass(frozen=True)
class PackedLayout:
    offsets: Tuple[int, ...]

    @property
    def SlotCount(self) -> int:
        return len(self.offsets) - 1

    @property
    def PackedDim(self) -> int:
        return self.offsets[-1]

    def Width(self, slotIndex: int) -> int:
        return self.offsets[slotIndex + 1] - self.offsets[slotIndex]

    def Slice(self, slotIndex: int) -> slice:
        return slice(self.offsets[slotIndex], self.offsets[slotIndex + 1])

    @classmethod
    def FromWidths(cls, widths: Sequence[int]) -> "PackedLayout":
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1] + width)
        return cls(tuple(offsets))


@dataclass(frozen=True)
class PackedTensor:
    values: Tuple[float, ...]
    offsets: Tuple[int, ...]
    shapes: Tuple[Tuple[int, int], ...]

    def Matrix(
        self,
        slotIndex: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        start = self.offsets[slotIndex]
        end = self.offsets[slotIndex + 1]
        return torch.tensor(
            self.values[start:end],
            device=device,
            dtype=dtype).reshape(self.shapes[slotIndex])

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
            flat = tuple(value for row in rows for value in row)
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
    end_effector_feedback_packed_dim: int
    end_effector_target_packed_dim: int
    hierarchy_edge_count: int
    perception_view_dim: int
    perception_motion_packed_dim: int


@dataclass(frozen=True)
class JointDefinition:
    joint_id: str
    joint_type: JointType
    translation_axis: Tuple[float, float, float]
    rotation_axis: Tuple[float, float, float]
    position_lower: float
    position_upper: float
    velocity_limit: float
    perception_effector_id: Optional[str] = None


@dataclass(frozen=True)
class EndEffector:
    effector_id: str
    effector_type: EndEffectorType
    parent_effector_id: Optional[str]
    translation_basis: Tuple[Tuple[float, ...], ...]
    rotation_basis: Tuple[Tuple[float, ...], ...]
    target_lower: Tuple[float, ...]
    target_upper: Tuple[float, ...]
    reference_frame_id: str
    is_perception_slot: bool
    progress_enter: float
    progress_exit: float
    dwell_cycles: int
    translation_error_scale: float = 1.0
    rotation_error_scale: float = 1.0

    @property
    def TargetDim(self) -> int:
        return len(self.translation_basis[0]) + len(self.rotation_basis[0])


@dataclass(frozen=True)
class PerceptionCalibration:
    effector_id: str
    calibration_id: str
    frame_id: str
    projection_matrix: Tuple[Tuple[float, ...], ...]
    reference_size: Tuple[int, int]
    primary: bool


@dataclass(frozen=True)
class PerceptionProjectionView:
    calibration_id: str
    reference_frame_id: str
    projection_matrix: Tuple[Tuple[float, ...], ...]
    reference_size: Tuple[int, int]


@dataclass(frozen=True)
class RobotDefinition:
    profile_id: str
    description_id: str
    semantic_definition_id: str
    adapter_id: str
    joints: Tuple[JointDefinition, ...]
    end_effectors: Tuple[EndEffector, ...]
    perception_calibrations: Tuple[PerceptionCalibration, ...]


@dataclass(frozen=True)
class RobotEmbodimentContractView:
    schema_version: int
    description_id: str
    semantic_definition_id: str
    contract_id: str
    model_shape_id: str
    adapter_id: str
    model_signature: str
    rotation_chart_limit: float
    joint_count: int
    end_effector_count: int
    joint_feedback_layout: PackedLayout
    end_effector_feedback_layout: PackedLayout
    end_effector_target_layout: PackedLayout
    perception_motion_layout: PackedLayout
    static_joint_tokens: Tuple[Tuple[float, ...], ...]
    static_end_effector_tokens: Tuple[Tuple[float, ...], ...]
    end_effector_translation_basis: PackedTensor
    end_effector_rotation_basis: PackedTensor
    end_effector_target_lower: Tuple[float, ...]
    end_effector_target_upper: Tuple[float, ...]
    parent_index: Tuple[int, ...]
    topological_layers: Tuple[Tuple[int, ...], ...]
    root_mask: Tuple[bool, ...]
    child_mask: Tuple[bool, ...]
    independent_mask: Tuple[bool, ...]
    perception_view_indices: Tuple[int, ...]
    perception_projection: Optional[PerceptionProjectionView]
    primary_perception_view_index: Optional[int]
    model_shape: EmbodimentShape


@dataclass(frozen=True)
class PackedEndEffectorTarget:
    values: torch.Tensor
    active: torch.Tensor
    contract_id: str
    model_signature: str
    target_version: torch.Tensor
    timestamp: torch.Tensor

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
        if self.contract_id != contractView.contract_id or self.model_signature != contractView.model_signature:
            raise ValueError("target identity does not match robot contract")
        if (
            not torch.is_tensor(self.values)
            or self.values.dim() != 2
            or self.values.dtype != torch.float32
        ):
            raise ValueError("packed targets must be a float32 batched matrix")
        batchSize = int(self.values.size(0))
        Robot.ValidateTensor(
            self.values,
            (batchSize, contractView.end_effector_target_layout.PackedDim),
            "packed targets",
            True)
        Robot.ValidateTensor(
            self.active,
            (batchSize, contractView.end_effector_count),
            "target active mask",
            False)
        if self.active.device != self.values.device:
            raise ValueError("target tensors must share a device")
        values = self.values.to(dtype=torch.float64)
        lower = torch.tensor(
            contractView.end_effector_target_lower,
            device=self.values.device,
            dtype=torch.float64).unsqueeze(0)
        upper = torch.tensor(
            contractView.end_effector_target_upper,
            device=self.values.device,
            dtype=torch.float64).unsqueeze(0)
        tolerance = 1e-6
        for endpointIndex in range(contractView.end_effector_count):
            targetSlice = contractView.end_effector_target_layout.Slice(endpointIndex)
            active = self.active[:, endpointIndex].unsqueeze(-1)
            outside = (
                (values[:, targetSlice] < lower[:, targetSlice] - tolerance)
                | (values[:, targetSlice] > upper[:, targetSlice] + tolerance))
            if bool((active & outside).any().item()):
                raise ValueError("active target exceeds end-effector limits")
            translationWidth = contractView.end_effector_translation_basis.shapes[endpointIndex][1]
            rotationWidth = contractView.end_effector_rotation_basis.shapes[endpointIndex][1]
            if rotationWidth:
                coordinates = values[:, targetSlice][
                    :, translationWidth:translationWidth + rotationWidth]
                basis = contractView.end_effector_rotation_basis.Matrix(
                    endpointIndex,
                    self.values.device,
                    torch.float64)
                norm = torch.linalg.vector_norm(
                    coordinates @ basis.transpose(0, 1),
                    dim=-1)
                if bool((
                    self.active[:, endpointIndex]
                    & norm.gt(contractView.rotation_chart_limit + tolerance)
                ).any().item()):
                    raise ValueError("active rotation target leaves the principal chart")
        for endpointIndex, parentIndex in enumerate(contractView.parent_index):
            if parentIndex >= 0 and bool((self.active[:, endpointIndex] & ~self.active[:, parentIndex]).any().item()):
                raise ValueError("active child target requires its parent target")
        if (
            not torch.is_tensor(self.target_version)
            or tuple(self.target_version.shape) != (batchSize,)
            or self.target_version.dtype != torch.long
            or self.target_version.device != self.values.device
            or bool(self.target_version.lt(0).any().item())
        ):
            raise ValueError("target version must be a non-negative long vector")
        Robot.ValidateTensor(self.timestamp, (batchSize,), "target timestamp", True)
        if self.timestamp.device != self.values.device or bool(self.timestamp.lt(0.0).any().item()):
            raise ValueError("target timestamp is invalid")


@dataclass(frozen=True)
class BrainFeedbackPacket:
    joint_features: torch.Tensor
    end_effector_features: torch.Tensor
    endpoint_present: torch.Tensor
    progress: torch.Tensor
    reached: torch.Tensor
    child_enabled: torch.Tensor
    target_active: torch.Tensor
    target_version: torch.Tensor
    perception_rotation: torch.Tensor
    perception_rotation_delta: torch.Tensor
    perception_angular_velocity: torch.Tensor
    perception_motion_features: torch.Tensor
    perception_motion_present: torch.Tensor
    contract_id: str
    model_signature: str
    timestamp: torch.Tensor

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
        if self.contract_id != contractView.contract_id or self.model_signature != contractView.model_signature:
            raise ValueError("feedback identity does not match robot contract")
        if not torch.is_tensor(self.joint_features) or self.joint_features.dim() != 2:
            raise ValueError("joint features must be a batched matrix")
        batchSize = int(self.joint_features.size(0))
        device = self.joint_features.device
        dtype = self.joint_features.dtype
        floatingFields = (
            (self.joint_features, (batchSize, contractView.joint_feedback_layout.PackedDim)),
            (self.end_effector_features, (batchSize, contractView.end_effector_feedback_layout.PackedDim)),
            (self.progress, (batchSize, contractView.end_effector_count)),
            (self.perception_rotation, (batchSize, len(contractView.perception_view_indices), 4)),
            (self.perception_rotation_delta, (batchSize, len(contractView.perception_view_indices), 4)),
            (self.perception_angular_velocity, (batchSize, len(contractView.perception_view_indices), 3)),
            (self.perception_motion_features, (batchSize, contractView.perception_motion_layout.PackedDim)),
        )
        booleanFields = (
            (self.endpoint_present, (batchSize, contractView.end_effector_count)),
            (self.reached, (batchSize, contractView.end_effector_count)),
            (self.child_enabled, (batchSize, contractView.end_effector_count)),
            (self.target_active, (batchSize, contractView.end_effector_count)),
            (self.perception_motion_present, (batchSize, len(contractView.perception_view_indices))),
        )
        for value, shape in floatingFields:
            Robot.ValidateTensor(value, shape, "feedback field", True)
            if value.device != device or value.dtype != dtype:
                raise ValueError("floating feedback fields must share device and dtype")
        for value, shape in booleanFields:
            Robot.ValidateTensor(value, shape, "feedback mask", False)
            if value.device != device:
                raise ValueError("feedback masks must share the packet device")
        Robot.ValidateTensor(self.timestamp, (batchSize,), "feedback timestamp", True)
        if self.timestamp.device != device or bool(self.timestamp.lt(0.0).any().item()):
            raise ValueError("feedback timestamp must share the packet device")
        if (
            not torch.is_tensor(self.target_version)
            or tuple(self.target_version.shape) != (batchSize,)
            or self.target_version.dtype != torch.long
            or self.target_version.device != device
            or bool(self.target_version.lt(-1).any().item())
        ):
            raise ValueError("feedback target version is invalid")
    def IndexSelectRows(self, rowIndex: torch.Tensor) -> "BrainFeedbackPacket":
        return BrainFeedbackPacket(
            joint_features=self.joint_features.index_select(0, rowIndex),
            end_effector_features=self.end_effector_features.index_select(0, rowIndex),
            endpoint_present=self.endpoint_present.index_select(0, rowIndex),
            progress=self.progress.index_select(0, rowIndex),
            reached=self.reached.index_select(0, rowIndex),
            child_enabled=self.child_enabled.index_select(0, rowIndex),
            target_active=self.target_active.index_select(0, rowIndex),
            target_version=self.target_version.index_select(0, rowIndex),
            perception_rotation=self.perception_rotation.index_select(0, rowIndex),
            perception_rotation_delta=self.perception_rotation_delta.index_select(0, rowIndex),
            perception_angular_velocity=self.perception_angular_velocity.index_select(0, rowIndex),
            perception_motion_features=self.perception_motion_features.index_select(0, rowIndex),
            perception_motion_present=self.perception_motion_present.index_select(0, rowIndex),
            contract_id=self.contract_id,
            model_signature=self.model_signature,
            timestamp=self.timestamp.index_select(0, rowIndex))

    def RepeatCandidates(self, candidateCount: int) -> "BrainFeedbackPacket":
        rows = torch.arange(
            int(self.joint_features.size(0)),
            device=self.joint_features.device,
            dtype=torch.long).repeat_interleave(candidateCount)
        return self.IndexSelectRows(rows)


class Robot:
    SchemaVersion = 24
    RotationStateWidth = 6
    PrincipalRotationLimit = math.pi - 1e-4
    PerceptionTranslationTolerance = 1e-5
    IdentityBasis = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    EmptyBasis = ((), (), ())
    FeedbackFields = frozenset({
        "position",
        "velocity",
        "end_effector_translation",
        "end_effector_rotation_xyzw",
        "end_effector_present",
        "contract_id",
        "timestamp",
    })

    def __init__(self, definition: RobotDefinition) -> None:
        (
            self.PerceptionJointIndices,
            self.PerceptionVelocityAxes,
        ) = self.ValidateDefinition(definition)
        self.Joints = definition.joints
        self.EndEffectors = definition.end_effectors
        self.EndEffectorIds = tuple(value.effector_id for value in self.EndEffectors)
        self.ReferenceFrameIds = tuple(value.reference_frame_id for value in self.EndEffectors)
        self.JointLower = tuple(float(value.position_lower) for value in self.Joints)
        self.JointUpper = tuple(float(value.position_upper) for value in self.Joints)
        self.JointVelocityLimit = tuple(float(value.velocity_limit) for value in self.Joints)
        self.JointPeriodic = tuple(
            value.joint_type is JointType.CONTINUOUS
            for value in self.Joints)
        self.JointRotational = tuple(value.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS) for value in self.Joints)
        self.ProgressEnter = tuple(float(value.progress_enter) for value in self.EndEffectors)
        self.ProgressExit = tuple(float(value.progress_exit) for value in self.EndEffectors)
        self.DwellCycles = tuple(int(value.dwell_cycles) for value in self.EndEffectors)
        self.TranslationErrorScale = tuple(float(value.translation_error_scale) for value in self.EndEffectors)
        self.RotationErrorScale = tuple(float(value.rotation_error_scale) for value in self.EndEffectors)
        self.ContractView = self.CompileContractView(definition)
        self.TranslationProjection = self.CompileBasisProjection(
            self.ContractView.end_effector_translation_basis)
        self.RotationProjection = self.CompileBasisProjection(
            self.ContractView.end_effector_rotation_basis)
        self.Reset()

    @staticmethod
    def CompileSignature(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False).encode("utf-8")
        return sha256(canonical).hexdigest()

    @classmethod
    def CompileBrainBuildSignature(
        cls,
        cognitivePayload: Mapping[str, Any],
        contractView: RobotEmbodimentContractView,
        schemaVersion: int,
    ) -> str:
        return cls.CompileSignature({
            "schema_version": schemaVersion,
            "cognitive": dict(cognitivePayload),
            "contract": {
                "schema_version": contractView.schema_version,
                "description_id": contractView.description_id,
                "semantic_definition_id": contractView.semantic_definition_id,
                "contract_id": contractView.contract_id,
                "model_shape_id": contractView.model_shape_id,
                "adapter_id": contractView.adapter_id,
                "model_signature": contractView.model_signature,
                "rotation_chart_limit": contractView.rotation_chart_limit,
            },
            "embodiment": asdict(contractView.model_shape),
        })

    @staticmethod
    def ValidateTensor(
        value: torch.Tensor,
        shape: Tuple[int, ...],
        fieldName: str,
        floating: bool,
    ) -> None:
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"{fieldName} has the wrong shape")
        if floating:
            if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{fieldName} must be finite floating point")
        elif value.dtype != torch.bool:
            raise ValueError(f"{fieldName} must be boolean")

    @staticmethod
    def ValidateIdentifier(value: Any, fieldName: str) -> None:
        if type(value) is not str or not value:
            raise ValueError(f"{fieldName} must be a non-empty string")

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
        if result[0]:
            matrix = torch.tensor(result, dtype=torch.float64)
            if int(torch.linalg.matrix_rank(matrix).item()) != len(result[0]):
                raise ValueError(f"{fieldName} columns must be independent")
        return result

    @staticmethod
    def CompileBasisProjection(bases: PackedTensor) -> PackedTensor:
        matrices = []
        for endpointIndex in range(len(bases.shapes)):
            basis = bases.Matrix(endpointIndex, dtype=torch.float64)
            projection = torch.linalg.pinv(
                basis,
                atol=0.0,
                rtol=0.0).transpose(0, 1)
            matrices.append(tuple(
                tuple(float(value) for value in row)
                for row in projection.tolist()))
        return PackedTensor.FromMatrices(tuple(matrices))

    @classmethod
    def CompilePerceptionJointBindings(
        cls,
        joints: Sequence[JointDefinition],
        endEffectors: Sequence[EndEffector],
    ) -> Tuple[Tuple[Tuple[int, ...], ...], PackedTensor]:
        perceptionById = {
            endpoint.effector_id: endpoint
            for endpoint in endEffectors
            if endpoint.is_perception_slot
        }
        bindings = {
            endpointId: []
            for endpointId in perceptionById
        }
        for jointIndex, joint in enumerate(joints):
            binding = joint.perception_effector_id
            if binding is None:
                continue
            if binding not in perceptionById:
                raise ValueError(
                    "perception joint binding must target a perception slot")
            if joint.joint_type not in (
                JointType.REVOLUTE,
                JointType.CONTINUOUS,
            ):
                raise ValueError(
                    "perception joint binding requires a rotational joint")
            bindings[binding].append(jointIndex)
        jointIndices = []
        velocityAxes = []
        for endpoint in perceptionById.values():
            indices = tuple(bindings[endpoint.effector_id])
            if not indices:
                raise ValueError(
                    "perception slot requires bound rotational joints")
            basis = torch.tensor(
                endpoint.rotation_basis,
                dtype=torch.float64)
            axes = torch.tensor(tuple(
                joints[jointIndex].rotation_axis
                for jointIndex in indices), dtype=torch.float64).transpose(0, 1)
            allowedRank = int(torch.linalg.matrix_rank(basis).item())
            if (
                int(torch.linalg.matrix_rank(axes).item()) != allowedRank
                or int(torch.linalg.matrix_rank(
                    torch.cat((basis, axes), dim=1)).item()) != allowedRank
            ):
                raise ValueError(
                    "perception joint axes must span the allowed rotation subspace")
            jointIndices.append(indices)
            velocityAxes.append(tuple(
                tuple(float(value) for value in joints[jointIndex].rotation_axis)
                for jointIndex in indices))
        return tuple(jointIndices), PackedTensor.FromMatrices(tuple(velocityAxes))

    @classmethod
    def ValidateDefinition(
        cls,
        definition: RobotDefinition,
    ) -> Tuple[Tuple[Tuple[int, ...], ...], PackedTensor]:
        if type(definition) is not RobotDefinition:
            raise TypeError("robot definition has the wrong type")
        for name in ("profile_id", "description_id", "semantic_definition_id", "adapter_id"):
            cls.ValidateIdentifier(getattr(definition, name), name)
        if not definition.joints or any(type(value) is not JointDefinition for value in definition.joints):
            raise ValueError("robot definition requires joint coordinates")
        if not definition.end_effectors or any(type(value) is not EndEffector for value in definition.end_effectors):
            raise ValueError("robot definition requires end effectors")
        jointIds = tuple(value.joint_id for value in definition.joints)
        endpointIds = tuple(value.effector_id for value in definition.end_effectors)
        if len(set(jointIds)) != len(jointIds) or len(set(endpointIds)) != len(endpointIds):
            raise ValueError("robot component identifiers must be unique")
        for joint in definition.joints:
            cls.ValidateIdentifier(joint.joint_id, "joint id")
            if type(joint.joint_type) is not JointType:
                raise ValueError("robot joints must be movable scalar coordinates")
            translation = tuple(float(value) for value in joint.translation_axis)
            rotation = tuple(float(value) for value in joint.rotation_axis)
            if len(translation) != 3 or len(rotation) != 3 or any(not math.isfinite(value) for value in translation + rotation):
                raise ValueError("joint axes must contain three finite values")
            translationNorm = math.sqrt(sum(value * value for value in translation))
            rotationNorm = math.sqrt(sum(value * value for value in rotation))
            if joint.joint_type is JointType.PRISMATIC:
                validAxis = abs(translationNorm - 1.0) <= 1e-6 and rotationNorm <= 1e-12
            else:
                validAxis = abs(rotationNorm - 1.0) <= 1e-6 and translationNorm <= 1e-12
            if not validAxis:
                raise ValueError("joint motion description is inconsistent")
            if (
                not math.isfinite(float(joint.position_lower))
                or not math.isfinite(float(joint.position_upper))
                or not math.isfinite(float(joint.velocity_limit))
                or joint.position_lower >= joint.position_upper
                or joint.velocity_limit <= 0.0
            ):
                raise ValueError("joint limits are invalid")
        endpointSet = set(endpointIds)
        for endpoint in definition.end_effectors:
            cls.ValidateIdentifier(endpoint.effector_id, "end-effector id")
            if type(endpoint.effector_type) is not EndEffectorType:
                raise ValueError("end-effector type is invalid")
            if endpoint.parent_effector_id is not None and endpoint.parent_effector_id not in endpointSet:
                raise ValueError("end-effector parent is unknown")
            cls.ValidateIdentifier(endpoint.reference_frame_id, "reference frame")
            translation = cls.ValidateBasis(endpoint.translation_basis, "translation basis")
            rotation = cls.ValidateBasis(endpoint.rotation_basis, "rotation basis")
            targetDim = len(translation[0]) + len(rotation[0])
            if targetDim < 1 or len(endpoint.target_lower) != targetDim or len(endpoint.target_upper) != targetDim:
                raise ValueError("end-effector target dimensions are inconsistent")
            if any(
                not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or lower >= upper
                for lower, upper in zip(endpoint.target_lower, endpoint.target_upper)
            ):
                raise ValueError("end-effector target limits are invalid")
            if (
                type(endpoint.is_perception_slot) is not bool
                or any(not math.isfinite(float(value)) for value in (
                    endpoint.progress_enter,
                    endpoint.progress_exit,
                    endpoint.translation_error_scale,
                    endpoint.rotation_error_scale))
                or endpoint.progress_enter <= 0.0
                or endpoint.progress_exit <= endpoint.progress_enter
                or type(endpoint.dwell_cycles) is not int
                or endpoint.dwell_cycles < 1
                or endpoint.translation_error_scale <= 0.0
                or endpoint.rotation_error_scale <= 0.0
            ):
                raise ValueError("end-effector execution semantics are invalid")
            if endpoint.is_perception_slot and (
                len(translation[0]) != 0
                or len(rotation[0]) < 1
            ):
                raise ValueError("perception actuator must expose pure rotation")
        perceptionIds = {value.effector_id for value in definition.end_effectors if value.is_perception_slot}
        primaryCount = 0
        calibrationIds = set()
        for calibration in definition.perception_calibrations:
            if (
                type(calibration) is not PerceptionCalibration
                or calibration.effector_id not in perceptionIds
                or type(calibration.primary) is not bool
            ):
                raise ValueError("perception calibration component is invalid")
            cls.ValidateIdentifier(calibration.calibration_id, "calibration id")
            cls.ValidateIdentifier(calibration.frame_id, "calibration frame")
            if calibration.calibration_id in calibrationIds:
                raise ValueError("perception calibration identifiers must be unique")
            calibrationIds.add(calibration.calibration_id)
            matrix = tuple(tuple(float(value) for value in row) for row in calibration.projection_matrix)
            if (
                len(matrix) != 3
                or any(len(row) != 3 for row in matrix)
                or any(not math.isfinite(value) for row in matrix for value in row)
                or matrix[0][0] <= 0.0
                or matrix[1][1] <= 0.0
                or tuple(matrix[2]) != (0.0, 0.0, 1.0)
            ):
                raise ValueError("perception projection must be a finite three-by-three matrix")
            if len(calibration.reference_size) != 2 or any(type(value) is not int or value < 1 for value in calibration.reference_size):
                raise ValueError("perception reference size is invalid")
            primaryCount += int(calibration.primary)
        if primaryCount != int(bool(perceptionIds)):
            raise ValueError("robot definition requires one primary perception calibration")
        return cls.CompilePerceptionJointBindings(
            definition.joints,
            definition.end_effectors)

    @staticmethod
    def CompileHierarchy(
        parentIndex: Sequence[int],
    ) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[bool, ...], Tuple[bool, ...], Tuple[bool, ...]]:
        count = len(parentIndex)
        depths = []
        for index in range(count):
            visited = set()
            cursor = index
            depth = -1
            while cursor >= 0:
                if cursor in visited:
                    raise ValueError("end-effector hierarchy must be an acyclic graph")
                visited.add(cursor)
                depth += 1
                cursor = parentIndex[cursor]
            depths.append(depth)
        layers = tuple(
            tuple(index for index, depth in enumerate(depths) if depth == layer)
            for layer in range(max(depths) + 1))
        rootMask = tuple(parent < 0 for parent in parentIndex)
        childMask = tuple(parent >= 0 for parent in parentIndex)
        independentMask = tuple(rootMask[index] and index not in parentIndex for index in range(count))
        return layers, rootMask, childMask, independentMask

    @staticmethod
    def CompileShapeId(
        shape: EmbodimentShape,
        jointLayout: PackedLayout,
        feedbackLayout: PackedLayout,
        targetLayout: PackedLayout,
        perceptionLayout: PackedLayout,
    ) -> str:
        return Robot.CompileSignature({
            "kind": "embodiment_shape",
            "shape": asdict(shape),
            "joint_feedback_widths": tuple(jointLayout.Width(index) for index in range(jointLayout.SlotCount)),
            "end_effector_feedback_widths": tuple(feedbackLayout.Width(index) for index in range(feedbackLayout.SlotCount)),
            "end_effector_target_widths": tuple(targetLayout.Width(index) for index in range(targetLayout.SlotCount)),
            "perception_motion_widths": tuple(perceptionLayout.Width(index) for index in range(perceptionLayout.SlotCount)),
        })

    @staticmethod
    def CompileJointTokens(joints: Sequence[JointDefinition]) -> Tuple[Tuple[float, ...], ...]:
        return tuple(tuple(float(value) for value in (
            *(float(typeIndex == int(joint.joint_type)) for typeIndex in range(len(JointType))),
            *joint.translation_axis,
            *joint.rotation_axis,
            joint.position_lower,
            joint.position_upper,
            joint.velocity_limit,
        )) for joint in joints)

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
        tokens = []
        for index, endpoint in enumerate(endEffectors):
            translationWidth = len(endpoint.translation_basis[0])
            rotationWidth = len(endpoint.rotation_basis[0])
            tokens.append(tuple(float(value) for value in (
                *(float(typeIndex == int(endpoint.effector_type)) for typeIndex in range(len(EndEffectorType))),
                *(endpoint.translation_basis[row][column] if column < translationWidth else 0.0 for row in range(3) for column in range(3)),
                *(float(column < translationWidth) for column in range(3)),
                *(endpoint.rotation_basis[row][column] if column < rotationWidth else 0.0 for row in range(3) for column in range(3)),
                *(float(column < rotationWidth) for column in range(3)),
                float(parentIndex[index] < 0),
                float(parentIndex[index] >= 0),
                float(endpoint.is_perception_slot),
                float(depths[index]) / float(max(maxDepth, 1)),
            )))
        return tuple(tokens)

    def CompileContractView(
        self,
        definition: RobotDefinition,
    ) -> RobotEmbodimentContractView:
        endpointIndex = {value.effector_id: index for index, value in enumerate(self.EndEffectors)}
        parentIndex = tuple(
            -1 if value.parent_effector_id is None else endpointIndex[value.parent_effector_id]
            for value in self.EndEffectors)
        layers, rootMask, childMask, independentMask = self.CompileHierarchy(parentIndex)
        jointTokens = self.CompileJointTokens(self.Joints)
        endpointTokens = self.CompileEndEffectorTokens(self.EndEffectors, parentIndex, layers)
        jointLayout = PackedLayout.FromWidths(tuple(
            3 if value.joint_type is JointType.CONTINUOUS
            else 4 if value.joint_type is JointType.REVOLUTE
            else 2
            for value in self.Joints))
        targetLayout = PackedLayout.FromWidths(tuple(value.TargetDim for value in self.EndEffectors))
        feedbackLayout = PackedLayout.FromWidths(tuple(
            len(value.translation_basis[0]) + (self.RotationStateWidth if value.rotation_basis[0] else 0)
            for value in self.EndEffectors))
        perceptionIndices = tuple(index for index, value in enumerate(self.EndEffectors) if value.is_perception_slot)
        perceptionLayout = PackedLayout.FromWidths(tuple(
            4 + len(self.EndEffectors[index].rotation_basis[0])
            for index in perceptionIndices))
        primary = next((
            value
            for value in definition.perception_calibrations
            if value.primary), None)
        projection = None if primary is None else PerceptionProjectionView(
            calibration_id=primary.calibration_id,
            reference_frame_id=primary.frame_id,
            projection_matrix=primary.projection_matrix,
            reference_size=primary.reference_size)
        primaryIndex = None if primary is None else perceptionIndices.index(
            endpointIndex[primary.effector_id])
        shape = EmbodimentShape(
            joint_token_count=len(self.Joints),
            joint_static_descriptor_dim=len(jointTokens[0]),
            joint_feedback_packed_dim=jointLayout.PackedDim,
            end_effector_token_count=len(self.EndEffectors),
            end_effector_static_descriptor_dim=len(endpointTokens[0]),
            end_effector_feedback_packed_dim=feedbackLayout.PackedDim,
            end_effector_target_packed_dim=targetLayout.PackedDim,
            hierarchy_edge_count=sum(value >= 0 for value in parentIndex),
            perception_view_dim=len(perceptionIndices),
            perception_motion_packed_dim=perceptionLayout.PackedDim)
        shapeId = self.CompileShapeId(shape, jointLayout, feedbackLayout, targetLayout, perceptionLayout)
        translationBasis = PackedTensor.FromMatrices(tuple(value.translation_basis for value in self.EndEffectors))
        rotationBasis = PackedTensor.FromMatrices(tuple(value.rotation_basis for value in self.EndEffectors))
        targetLower = tuple(value for endpoint in self.EndEffectors for value in endpoint.target_lower)
        targetUpper = tuple(value for endpoint in self.EndEffectors for value in endpoint.target_upper)
        modelSignature = self.CompileSignature({
            "schema_version": self.SchemaVersion,
            "definition": asdict(definition),
            "joint_feedback_layout": jointLayout.offsets,
            "end_effector_feedback_layout": feedbackLayout.offsets,
            "end_effector_target_layout": targetLayout.offsets,
            "perception_motion_layout": perceptionLayout.offsets,
            "joint_tokens": jointTokens,
            "end_effector_tokens": endpointTokens,
            "parent_index": parentIndex,
            "topological_layers": layers,
        })
        contractId = self.CompileSignature({"kind": "robot_contract", "model_signature": modelSignature})
        view = RobotEmbodimentContractView(
            schema_version=self.SchemaVersion,
            description_id=definition.description_id,
            semantic_definition_id=definition.semantic_definition_id,
            contract_id=contractId,
            model_shape_id=shapeId,
            adapter_id=definition.adapter_id,
            model_signature=modelSignature,
            rotation_chart_limit=self.PrincipalRotationLimit,
            joint_count=len(self.Joints),
            end_effector_count=len(self.EndEffectors),
            joint_feedback_layout=jointLayout,
            end_effector_feedback_layout=feedbackLayout,
            end_effector_target_layout=targetLayout,
            perception_motion_layout=perceptionLayout,
            static_joint_tokens=jointTokens,
            static_end_effector_tokens=endpointTokens,
            end_effector_translation_basis=translationBasis,
            end_effector_rotation_basis=rotationBasis,
            end_effector_target_lower=targetLower,
            end_effector_target_upper=targetUpper,
            parent_index=parentIndex,
            topological_layers=layers,
            root_mask=rootMask,
            child_mask=childMask,
            independent_mask=independentMask,
            perception_view_indices=perceptionIndices,
            perception_projection=projection,
            primary_perception_view_index=primaryIndex,
            model_shape=shape)
        return view

    @classmethod
    def ParseUrdf(
        cls,
        source: Any,
        reader: Callable[[Any], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return reader(source)

    @classmethod
    def ParseSrdf(
        cls,
        source: Any,
        reader: Callable[[Any], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return reader(source)

    @classmethod
    def FromUrdfSrdf(
        cls,
        urdfSource: Any,
        srdfSource: Any,
        profileId: str,
        adapterId: str,
        urdfReader: Callable[[Any], Mapping[str, Any]],
        srdfReader: Callable[[Any], Mapping[str, Any]],
    ) -> "Robot":
        urdf = cls.ParseUrdf(urdfSource, urdfReader)
        srdf = cls.ParseSrdf(srdfSource, srdfReader)
        if urdf["robot_name"] != srdf["robot_name"] or urdf["description_id"] != srdf["description_id"]:
            raise ValueError("URDF and SRDF identities do not match")
        definition = RobotDefinition(
            profile_id=profileId,
            description_id=urdf["description_id"],
            semantic_definition_id=srdf["semantic_definition_id"],
            adapter_id=adapterId,
            joints=tuple(urdf["joints"]),
            end_effectors=tuple(srdf["end_effectors"]),
            perception_calibrations=tuple(srdf["perception_calibrations"]))
        return cls(definition)

    @classmethod
    def CreateRevoluteJoint(
        cls,
        jointId: str,
        axis: Sequence[float],
        lower: float,
        upper: float,
        velocity: float,
        periodic: bool = False,
        perceptionEffectorId: Optional[str] = None,
    ) -> JointDefinition:
        return JointDefinition(
            joint_id=jointId,
            joint_type=JointType.CONTINUOUS if periodic else JointType.REVOLUTE,
            translation_axis=(0.0, 0.0, 0.0),
            rotation_axis=tuple(float(value) for value in axis),
            position_lower=float(lower),
            position_upper=float(upper),
            velocity_limit=float(velocity),
            perception_effector_id=perceptionEffectorId)

    @classmethod
    def CreateAnthropomorphicArm(
        cls,
        sideId: str,
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
        upperLimits = (
            (-2.79, 2.09, 2.4),
            (-1.57, 2.79, 2.4),
            (-1.57, 1.57, 2.4),
            (0.0, 2.62, 2.8),
            (-1.57, 1.57, 3.0),
            (-1.22, 1.22, 3.0),
            (-0.61, 0.61, 3.0),
        )
        joints = [
            cls.CreateRevoluteJoint(f"{sideId}_{name}", axis, lower, upper, velocity)
            for name, axis, (lower, upper, velocity) in zip(upperNames, upperAxes, upperLimits)
        ]
        fingerSpecifications = (
            ("thumb", ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((-0.70, 0.70), (-0.35, 1.40), (0.0, 1.40), (0.0, 1.40))),
            ("index", ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((-0.35, 1.57), (-0.35, 0.35), (0.0, 1.75), (0.0, 1.40))),
            ("middle", ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((-0.35, 1.57), (-0.25, 0.25), (0.0, 1.75), (0.0, 1.40))),
            ("ring", ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((-0.35, 1.57), (-0.30, 0.30), (0.0, 1.75), (0.0, 1.40))),
            ("little", ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), ((-0.35, 1.57), (-0.40, 0.40), (0.0, 1.75), (0.0, 1.40))),
        )
        fingerNames = []
        for fingerName, axes, limits in fingerSpecifications:
            for coordinateIndex, (axis, bounds) in enumerate(zip(axes, limits)):
                joints.append(cls.CreateRevoluteJoint(
                    f"{sideId}_{fingerName}_{coordinateIndex}",
                    axis,
                    bounds[0],
                    bounds[1],
                    4.0))
            fingerNames.append(fingerName)
        wristId = f"{sideId}_wrist"
        endpoints = [EndEffector(
            effector_id=wristId,
            effector_type=EndEffectorType.WRIST,
            parent_effector_id=None,
            translation_basis=cls.IdentityBasis,
            rotation_basis=cls.IdentityBasis,
            target_lower=(-1.5, -1.5, -1.5, -math.pi, -math.pi, -math.pi),
            target_upper=(1.5, 1.5, 1.5, math.pi, math.pi, math.pi),
            reference_frame_id="world",
            is_perception_slot=False,
            progress_enter=0.025,
            progress_exit=0.040,
            dwell_cycles=3)]
        endpoints.extend(EndEffector(
            effector_id=f"{sideId}_{fingerName}_tip",
            effector_type=EndEffectorType.FINGERTIP,
            parent_effector_id=wristId,
            translation_basis=cls.IdentityBasis,
            rotation_basis=cls.IdentityBasis,
            target_lower=(-0.35, -0.35, -0.35, -math.pi, -math.pi, -math.pi),
            target_upper=(0.35, 0.35, 0.35, math.pi, math.pi, math.pi),
            reference_frame_id=wristId,
            is_perception_slot=False,
            progress_enter=0.012,
            progress_exit=0.020,
            dwell_cycles=3) for fingerName in fingerNames)
        return tuple(joints), tuple(endpoints)

    @classmethod
    def CreateDefault(cls) -> "Robot":
        leftJoints, leftEndpoints = cls.CreateAnthropomorphicArm("left")
        rightJoints, rightEndpoints = cls.CreateAnthropomorphicArm("right")
        cameraJoints = (
            cls.CreateRevoluteJoint(
                "camera_yaw",
                (0.0, 0.0, 1.0),
                -math.pi,
                math.pi,
                1.5,
                periodic=True,
                perceptionEffectorId="camera"),
            cls.CreateRevoluteJoint(
                "camera_pitch",
                (0.0, 1.0, 0.0),
                -0.5 * math.pi,
                0.5 * math.pi,
                1.5,
                perceptionEffectorId="camera"),
            cls.CreateRevoluteJoint(
                "camera_roll",
                (1.0, 0.0, 0.0),
                -math.pi,
                math.pi,
                1.5,
                periodic=True,
                perceptionEffectorId="camera"),
        )
        camera = EndEffector(
            effector_id="camera",
            effector_type=EndEffectorType.SENSOR_ACTUATOR,
            parent_effector_id=None,
            translation_basis=cls.EmptyBasis,
            rotation_basis=cls.IdentityBasis,
            target_lower=(-math.pi, -math.pi, -math.pi),
            target_upper=(math.pi, math.pi, math.pi),
            reference_frame_id="world",
            is_perception_slot=True,
            progress_enter=0.020,
            progress_exit=0.040,
            dwell_cycles=2)
        return cls(RobotDefinition(
            profile_id="temporary_dual_anthropomorphic_arm_camera",
            description_id="temporary_dual_anthropomorphic_arm_camera_urdf",
            semantic_definition_id="temporary_dual_anthropomorphic_arm_camera_srdf",
            adapter_id="external_endpoint_pose_adapter",
            joints=leftJoints + rightJoints + cameraJoints,
            end_effectors=leftEndpoints + rightEndpoints + (camera,),
            perception_calibrations=(PerceptionCalibration(
                effector_id="camera",
                calibration_id="temporary_camera_projection",
                frame_id="camera_optical_pivot",
                projection_matrix=(
                    (384.0, 0.0, 255.5),
                    (0.0, 384.0, 255.5),
                    (0.0, 0.0, 1.0),
                ),
                reference_size=(512, 512),
                primary=True),)))

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
        matrix = identity + torch.sin(angle).unsqueeze(-1) * skew + (1.0 - torch.cos(angle)).unsqueeze(-1) * (skew @ skew)
        sx, sy, sz = rotationVector.unbind(dim=-1)
        smallSkew = torch.stack((
            zero, -sz, sy,
            sz, zero, -sx,
            -sy, sx, zero,
        ), dim=-1).reshape(rotationVector.shape[:-1] + (3, 3))
        smallMatrix = identity + smallSkew + 0.5 * (smallSkew @ smallSkew)
        return torch.where(
            angle.squeeze(-1).lt(math.sqrt(epsilon)).unsqueeze(-1).unsqueeze(-1),
            smallMatrix,
            matrix)

    @staticmethod
    def MatrixToQuaternion(matrix: torch.Tensor) -> torch.Tensor:
        m00 = matrix[..., 0, 0]
        m01 = matrix[..., 0, 1]
        m02 = matrix[..., 0, 2]
        m10 = matrix[..., 1, 0]
        m11 = matrix[..., 1, 1]
        m12 = matrix[..., 1, 2]
        m20 = matrix[..., 2, 0]
        m21 = matrix[..., 2, 1]
        m22 = matrix[..., 2, 2]
        squared = torch.stack((
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ), dim=-1)
        roots = torch.sqrt(squared.clamp_min(torch.finfo(matrix.dtype).tiny))
        candidates = torch.stack((
            torch.stack((roots[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, roots[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, roots[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, roots[..., 3].square()), dim=-1),
        ), dim=-2)
        candidates = candidates / (2.0 * roots.unsqueeze(-1)).clamp_min(0.1)
        best = roots.argmax(dim=-1)
        quaternionWxyz = candidates.gather(
            -2,
            best.unsqueeze(-1).unsqueeze(-1).expand(best.shape + (1, 4))).squeeze(-2)
        quaternion = torch.cat((quaternionWxyz[..., 1:], quaternionWxyz[..., :1]), dim=-1)
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion,
            dim=-1,
            keepdim=True).clamp_min(torch.finfo(matrix.dtype).eps)
        signValue = quaternion[..., 3]
        for componentIndex in range(3):
            signValue = torch.where(signValue.eq(0.0), quaternion[..., componentIndex], signValue)
        sign = torch.where(signValue.lt(0.0), -torch.ones_like(signValue), torch.ones_like(signValue))
        return quaternion * sign.unsqueeze(-1)

    @classmethod
    def MatrixToRotationVector(cls, matrix: torch.Tensor) -> torch.Tensor:
        quaternion = cls.MatrixToQuaternion(matrix)
        vector = quaternion[..., :3]
        scalar = quaternion[..., 3].clamp(0.0, 1.0)
        sineHalf = torch.linalg.vector_norm(vector, dim=-1)
        angle = 2.0 * torch.atan2(sineHalf, scalar)
        epsilon = torch.finfo(matrix.dtype).eps
        squared = sineHalf.square()
        scale = torch.where(
            sineHalf.gt(math.sqrt(epsilon)),
            angle / sineHalf.clamp_min(epsilon),
            2.0 + squared / 3.0 + 3.0 * squared.square() / 20.0)
        return vector * scale.unsqueeze(-1)

    def Reset(self) -> None:
        self.CachedTargetValues: Optional[torch.Tensor] = None
        self.CachedTargetActive: Optional[torch.Tensor] = None
        self.CachedTargetVersion: Optional[torch.Tensor] = None
        self.DwellState: Optional[torch.Tensor] = None
        self.ReachedState: Optional[torch.Tensor] = None
        self.LastTimestamp: Optional[torch.Tensor] = None
        self.LastPerceptionRotation: Optional[torch.Tensor] = None
        self.LastPerceptionStatePresent: Optional[torch.Tensor] = None
        self.PerceptionTranslationReference: Optional[torch.Tensor] = None
        self.PerceptionTranslationKnown: Optional[torch.Tensor] = None

    def BuildNeutralFeedbackPayload(
        self,
        timestamp: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Mapping[str, Any]:
        position = torch.tensor([
            0.0 if periodic else 0.5 * (lower + upper)
            for lower, upper, periodic in zip(self.JointLower, self.JointUpper, self.JointPeriodic)
        ], device=device, dtype=dtype)
        rotation = torch.zeros(
            self.ContractView.end_effector_count,
            4,
            device=device,
            dtype=dtype)
        rotation[:, 3] = 1.0
        return {
            "position": position.cpu().tolist(),
            "velocity": torch.zeros_like(position).cpu().tolist(),
            "end_effector_translation": torch.zeros(
                self.ContractView.end_effector_count,
                3,
                device=device,
                dtype=dtype).cpu().tolist(),
            "end_effector_rotation_xyzw": rotation.cpu().tolist(),
            "end_effector_present": [False] * self.ContractView.end_effector_count,
            "contract_id": self.ContractView.contract_id,
            "timestamp": float(timestamp),
        }

    def ValidatePayload(self, payload: Any) -> None:
        if not isinstance(payload, Mapping) or set(payload) != self.FeedbackFields:
            raise ValueError("robot feedback payload fields do not match")
        if torch.as_tensor(payload["end_effector_present"]).dtype != torch.bool:
            raise TypeError("end-effector presence must contain booleans")

    def DecodeFeedback(
        self,
        rawPayload: Any,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor]:
        payloads = tuple(rawPayload) if isinstance(rawPayload, (tuple, list)) else None
        if payloads is not None:
            if not payloads:
                raise ValueError("robot feedback batch cannot be empty")
            for payload in payloads:
                self.ValidatePayload(payload)
            contractIds = {payload["contract_id"] for payload in payloads}
            if len(contractIds) != 1:
                raise ValueError("robot feedback batch must share one contract")
            position = torch.stack(tuple(torch.as_tensor(
                payload["position"],
                device=device,
                dtype=torch.float32).reshape(-1) for payload in payloads))
            velocity = torch.stack(tuple(torch.as_tensor(
                payload["velocity"],
                device=device,
                dtype=torch.float32).reshape(-1) for payload in payloads))
            translation = torch.stack(tuple(torch.as_tensor(
                payload["end_effector_translation"],
                device=device,
                dtype=torch.float32) for payload in payloads))
            rotation = torch.stack(tuple(torch.as_tensor(
                payload["end_effector_rotation_xyzw"],
                device=device,
                dtype=torch.float32) for payload in payloads))
            present = torch.stack(tuple(torch.as_tensor(
                payload["end_effector_present"],
                device=device,
                dtype=torch.bool) for payload in payloads))
            timestamp = torch.tensor(tuple(float(torch.as_tensor(payload["timestamp"]).item()) for payload in payloads), device=device, dtype=torch.float64)
            contractId = next(iter(contractIds))
        else:
            self.ValidatePayload(rawPayload)
            position = torch.as_tensor(rawPayload["position"], device=device, dtype=torch.float32)
            velocity = torch.as_tensor(rawPayload["velocity"], device=device, dtype=torch.float32)
            translation = torch.as_tensor(rawPayload["end_effector_translation"], device=device, dtype=torch.float32)
            rotation = torch.as_tensor(rawPayload["end_effector_rotation_xyzw"], device=device, dtype=torch.float32)
            present = torch.as_tensor(rawPayload["end_effector_present"], device=device, dtype=torch.bool)
            if position.dim() == 1:
                position = position.unsqueeze(0)
            if velocity.dim() == 1:
                velocity = velocity.unsqueeze(0)
            if translation.dim() == 2:
                translation = translation.unsqueeze(0)
            if rotation.dim() == 2:
                rotation = rotation.unsqueeze(0)
            if present.dim() == 1:
                present = present.unsqueeze(0)
            timestamp = torch.as_tensor(rawPayload["timestamp"], device=device, dtype=torch.float64).reshape(-1)
            contractId = rawPayload["contract_id"]
        return position, velocity, translation, rotation, present, contractId, timestamp

    def ValidateFeedback(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        endpointPresent: torch.Tensor,
        contractId: str,
        timestamp: torch.Tensor,
        batchSize: Optional[int],
    ) -> None:
        if contractId != self.ContractView.contract_id:
            raise ValueError("robot feedback contract identity does not match")
        actualBatch = int(position.size(0)) if position.dim() == 2 else 0
        if batchSize is not None and actualBatch != int(batchSize):
            raise ValueError("robot feedback batch does not match sensory batch")
        self.ValidateTensor(position, (actualBatch, len(self.Joints)), "joint position", True)
        self.ValidateTensor(velocity, (actualBatch, len(self.Joints)), "joint velocity", True)
        self.ValidateTensor(translation, (actualBatch, len(self.EndEffectors), 3), "end-effector translation", True)
        self.ValidateTensor(rotation, (actualBatch, len(self.EndEffectors), 4), "end-effector rotation", True)
        self.ValidateTensor(endpointPresent, (actualBatch, len(self.EndEffectors)), "end-effector presence", False)
        self.ValidateTensor(timestamp, (actualBatch,), "feedback timestamp", True)
        if bool(timestamp.lt(0.0).any().item()):
            raise ValueError("feedback timestamp cannot be negative")
        lower = position.new_tensor(self.JointLower).unsqueeze(0)
        upper = position.new_tensor(self.JointUpper).unsqueeze(0)
        periodic = torch.tensor(self.JointPeriodic, device=position.device, dtype=torch.bool).unsqueeze(0)
        tolerance = 16.0 * torch.finfo(position.dtype).eps
        if bool(((~periodic) & ((position < lower - tolerance) | (position > upper + tolerance))).any().item()):
            raise ValueError("joint position exceeds robot limits")
        velocityLimit = velocity.new_tensor(self.JointVelocityLimit).unsqueeze(0)
        if bool(velocity.abs().gt(velocityLimit + tolerance).any().item()):
            raise ValueError("joint velocity exceeds robot limits")
        quaternionNorm = torch.linalg.vector_norm(rotation, dim=-1)
        if bool((endpointPresent & quaternionNorm.sub(1.0).abs().gt(
            64.0 * torch.finfo(rotation.dtype).eps
        )).any().item()):
            raise ValueError("end-effector rotations must be unit quaternions")
        if self.LastTimestamp is not None:
            if tuple(self.LastTimestamp.shape) != tuple(timestamp.shape):
                raise ValueError("feedback batch identity changed without robot reset")
            if bool(timestamp.le(self.LastTimestamp).any().item()):
                raise ValueError("feedback time must increase")
        if self.CachedTargetValues is not None and (
            tuple(self.CachedTargetValues.shape[:1]) != (actualBatch,)
            or self.CachedTargetValues.device != position.device
            or self.CachedTargetValues.dtype != position.dtype
        ):
            raise ValueError("cached target does not match feedback batch identity")

    def EncodeJointFeatures(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        features = []
        for jointIndex in range(len(self.Joints)):
            coordinate = position[:, jointIndex]
            normalizedVelocity = velocity[:, jointIndex] / self.JointVelocityLimit[jointIndex]
            if self.JointPeriodic[jointIndex]:
                feature = torch.stack((torch.sin(coordinate), torch.cos(coordinate), normalizedVelocity), dim=-1)
            else:
                normalizedPosition = 2.0 * (coordinate - self.JointLower[jointIndex]) / (self.JointUpper[jointIndex] - self.JointLower[jointIndex]) - 1.0
                if self.JointRotational[jointIndex]:
                    feature = torch.stack((normalizedPosition, torch.sin(coordinate), torch.cos(coordinate), normalizedVelocity), dim=-1)
                else:
                    feature = torch.stack((normalizedPosition, normalizedVelocity), dim=-1)
            features.append(feature)
        return torch.cat(features, dim=-1)

    def EncodeEndEffectorFeatures(
        self,
        translation: torch.Tensor,
        rotationQuaternion: torch.Tensor,
        endpointPresent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rotationMatrix = self.QuaternionToMatrix(rotationQuaternion)
        features = []
        for endpointIndex in range(len(self.EndEffectors)):
            translationBasis = self.ContractView.end_effector_translation_basis.Matrix(
                endpointIndex,
                translation.device,
                translation.dtype)
            rotationBasis = self.ContractView.end_effector_rotation_basis.Matrix(
                endpointIndex,
                translation.device,
                translation.dtype)
            components = []
            if translationBasis.size(1):
                projection = self.TranslationProjection.Matrix(
                    endpointIndex,
                    translation.device,
                    translation.dtype)
                coordinates = translation[:, endpointIndex] @ projection
                components.append(coordinates)
            if rotationBasis.size(1):
                components.append(torch.cat((
                    rotationMatrix[:, endpointIndex, :, 0],
                    rotationMatrix[:, endpointIndex, :, 1]), dim=-1))
            feature = torch.cat(components, dim=-1)
            features.append(torch.where(
                endpointPresent[:, endpointIndex].unsqueeze(-1),
                feature,
                torch.zeros_like(feature)))
        return torch.cat(features, dim=-1), rotationMatrix

    def ExpandTarget(
        self,
        values: torch.Tensor,
        active: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        translations = []
        rotationVectors = []
        for endpointIndex in range(len(self.EndEffectors)):
            targetSlice = self.ContractView.end_effector_target_layout.Slice(endpointIndex)
            coordinates = torch.where(
                active[:, endpointIndex].unsqueeze(-1),
                values[:, targetSlice],
                torch.zeros_like(values[:, targetSlice]))
            translationBasis = self.ContractView.end_effector_translation_basis.Matrix(
                endpointIndex,
                values.device,
                values.dtype)
            rotationBasis = self.ContractView.end_effector_rotation_basis.Matrix(
                endpointIndex,
                values.device,
                values.dtype)
            translationWidth = int(translationBasis.size(1))
            rotationWidth = int(rotationBasis.size(1))
            translations.append(coordinates[:, :translationWidth] @ translationBasis.transpose(0, 1))
            rotationVectors.append(coordinates[:, translationWidth:translationWidth + rotationWidth] @ rotationBasis.transpose(0, 1))
        return torch.stack(translations, dim=1), torch.stack(rotationVectors, dim=1)

    def DecodeTarget(self, target: PackedEndEffectorTarget) -> Mapping[str, Any]:
        target.Validate(self.ContractView)
        translation, rotationVector = self.ExpandTarget(target.values, target.active)
        translations = []
        rotations = []
        for batchIndex in range(int(target.values.size(0))):
            translationRow = []
            rotationRow = []
            for endpointIndex in range(len(self.EndEffectors)):
                endpointActive = bool(target.active[batchIndex, endpointIndex].item())
                translationRow.append(
                    translation[batchIndex, endpointIndex].detach().cpu().tolist()
                    if endpointActive and self.EndEffectors[endpointIndex].translation_basis[0]
                    else None)
                rotationRow.append(
                    rotationVector[batchIndex, endpointIndex].detach().cpu().tolist()
                    if endpointActive and self.EndEffectors[endpointIndex].rotation_basis[0]
                    else None)
            translations.append(translationRow)
            rotations.append(rotationRow)
        return {
            "contract_id": target.contract_id,
            "model_signature": target.model_signature,
            "target_version": target.target_version.detach().cpu().tolist(),
            "timestamp": target.timestamp.detach().cpu().tolist(),
            "slot_ids": list(self.EndEffectorIds),
            "reference_frame_ids": list(self.ReferenceFrameIds),
            "active": target.active.detach().cpu().tolist(),
            "translation": translations,
            "rotation_vector": rotations,
        }

    def CommitDispatchedTarget(
        self,
        target: PackedEndEffectorTarget,
        dispatchedMask: Optional[torch.Tensor] = None,
    ) -> None:
        batchSize = int(target.values.size(0))
        if dispatchedMask is None:
            dispatched = torch.ones(batchSize, device=target.values.device, dtype=torch.bool)
        else:
            self.ValidateTensor(dispatchedMask, (batchSize,), "dispatched mask", False)
            if dispatchedMask.device != target.values.device:
                raise ValueError("dispatched mask must share the target device")
            dispatched = dispatchedMask
        if not bool(dispatched.any().item()):
            return
        canonicalValues = torch.zeros_like(target.values)
        for endpointIndex in range(len(self.EndEffectors)):
            targetSlice = self.ContractView.end_effector_target_layout.Slice(endpointIndex)
            canonicalValues[:, targetSlice] = torch.where(
                target.active[:, endpointIndex].unsqueeze(-1),
                target.values[:, targetSlice],
                torch.zeros_like(target.values[:, targetSlice]))
        if self.CachedTargetValues is None:
            endpointCount = len(self.EndEffectors)
            self.CachedTargetValues = torch.zeros_like(target.values)
            self.CachedTargetActive = torch.zeros(batchSize, endpointCount, device=target.values.device, dtype=torch.bool)
            self.CachedTargetVersion = torch.full((batchSize,), -1, device=target.values.device, dtype=torch.long)
            self.DwellState = torch.zeros(batchSize, endpointCount, device=target.values.device, dtype=torch.long)
            self.ReachedState = torch.zeros(batchSize, endpointCount, device=target.values.device, dtype=torch.bool)
        if (
            tuple(self.CachedTargetValues.shape) != tuple(target.values.shape)
            or self.CachedTargetValues.device != target.values.device
            or self.CachedTargetValues.dtype != target.values.dtype
        ):
            raise ValueError("target batch identity changed without robot reset")
        if bool((dispatched & target.target_version.lt(self.CachedTargetVersion)).any().item()):
            raise ValueError("dispatched target version cannot move backwards")
        sameVersion = dispatched & target.target_version.eq(self.CachedTargetVersion)
        if bool(sameVersion.any().item()):
            rows = torch.nonzero(sameVersion, as_tuple=False).squeeze(-1)
            sameValues = torch.equal(
                canonicalValues.index_select(0, rows),
                self.CachedTargetValues.index_select(0, rows))
            sameActive = torch.equal(
                target.active.index_select(0, rows),
                self.CachedTargetActive.index_select(0, rows))
            if not sameValues or not sameActive:
                raise ValueError("one target version cannot identify different targets")
        changed = torch.zeros_like(self.CachedTargetActive)
        for endpointIndex in range(len(self.EndEffectors)):
            targetSlice = self.ContractView.end_effector_target_layout.Slice(endpointIndex)
            changed[:, endpointIndex] = dispatched & (
                target.active[:, endpointIndex].ne(self.CachedTargetActive[:, endpointIndex])
                | canonicalValues[:, targetSlice].ne(self.CachedTargetValues[:, targetSlice]).any(dim=-1))
        reset = changed.clone()
        for layer in self.ContractView.topological_layers[1:]:
            for endpointIndex in layer:
                reset[:, endpointIndex] |= reset[:, self.ContractView.parent_index[endpointIndex]]
        rows = torch.nonzero(dispatched, as_tuple=False).squeeze(-1)
        self.CachedTargetValues.index_copy_(0, rows, canonicalValues.index_select(0, rows).detach())
        self.CachedTargetActive.index_copy_(0, rows, target.active.index_select(0, rows).detach())
        self.CachedTargetVersion.index_copy_(0, rows, target.target_version.index_select(0, rows).detach())
        self.DwellState.masked_fill_(reset, 0)
        self.ReachedState.masked_fill_(reset, False)

    def EvaluateHierarchy(
        self,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        endpointPresent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batchSize = int(translation.size(0))
        endpointCount = len(self.EndEffectors)
        progress = translation.new_zeros(batchSize, endpointCount)
        reached = torch.zeros(batchSize, endpointCount, device=translation.device, dtype=torch.bool)
        enabled = torch.zeros_like(reached)
        if self.CachedTargetValues is None:
            rootMask = torch.tensor(self.ContractView.root_mask, device=translation.device, dtype=torch.bool).unsqueeze(0)
            return progress, reached, rootMask & endpointPresent
        _, targetRotationVector = self.ExpandTarget(
            self.CachedTargetValues,
            self.CachedTargetActive)
        for layer in self.ContractView.topological_layers:
            for endpointIndex in layer:
                parentIndex = self.ContractView.parent_index[endpointIndex]
                hierarchyEnabled = endpointPresent[:, endpointIndex] if parentIndex < 0 else reached[:, parentIndex] & endpointPresent[:, endpointIndex]
                enabled[:, endpointIndex] = hierarchyEnabled
                executing = hierarchyEnabled & self.CachedTargetActive[:, endpointIndex]
                errorSquared = translation.new_zeros(batchSize)
                targetSlice = self.ContractView.end_effector_target_layout.Slice(
                    endpointIndex)
                translationWidth = self.ContractView.end_effector_translation_basis.shapes[
                    endpointIndex][1]
                rotationWidth = self.ContractView.end_effector_rotation_basis.shapes[
                    endpointIndex][1]
                if translationWidth:
                    projection = self.TranslationProjection.Matrix(
                        endpointIndex,
                        translation.device,
                        translation.dtype)
                    currentCoordinates = (
                        translation[:, endpointIndex] @ projection)
                    targetCoordinates = self.CachedTargetValues[
                        :, targetSlice.start:targetSlice.start + translationWidth]
                    error = (
                        targetCoordinates - currentCoordinates
                    ) / self.TranslationErrorScale[endpointIndex]
                    errorSquared = errorSquared + error.square().sum(dim=-1)
                if rotationWidth:
                    targetRotation = self.RotationVectorToMatrix(targetRotationVector[:, endpointIndex])
                    relativeRotation = targetRotation @ rotation[:, endpointIndex].transpose(-1, -2)
                    projection = self.RotationProjection.Matrix(
                        endpointIndex,
                        rotation.device,
                        rotation.dtype)
                    error = (
                        self.MatrixToRotationVector(relativeRotation)
                        @ projection
                    ) / self.RotationErrorScale[endpointIndex]
                    errorSquared = errorSquared + error.square().sum(dim=-1)
                errorNorm = torch.sqrt(errorSquared)
                progress[:, endpointIndex] = torch.where(
                    executing,
                    (1.0 + errorNorm / self.ProgressExit[endpointIndex]).reciprocal(),
                    torch.zeros_like(errorNorm))
                wasReached = self.ReachedState[:, endpointIndex]
                inside = executing & errorNorm.le(self.ProgressEnter[endpointIndex])
                remain = executing & wasReached & errorNorm.le(self.ProgressExit[endpointIndex])
                dwell = torch.where(
                    inside & ~wasReached,
                    self.DwellState[:, endpointIndex] + 1,
                    torch.zeros_like(self.DwellState[:, endpointIndex]))
                endpointReached = remain | dwell.ge(self.DwellCycles[endpointIndex])
                self.DwellState[:, endpointIndex] = dwell
                self.ReachedState[:, endpointIndex] = endpointReached
                reached[:, endpointIndex] = endpointReached
        return progress, reached, enabled

    def EncodePerceptionAngularVelocity(
        self,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        angularVelocity = []
        for perceptionIndex, jointIndices in enumerate(
            self.PerceptionJointIndices
        ):
            index = torch.tensor(
                jointIndices,
                device=velocity.device,
                dtype=torch.long)
            axes = self.PerceptionVelocityAxes.Matrix(
                perceptionIndex,
                velocity.device,
                velocity.dtype)
            angularVelocity.append(
                velocity.index_select(1, index) @ axes)
        if angularVelocity:
            return torch.stack(angularVelocity, dim=1)
        return velocity.new_zeros(int(velocity.size(0)), 0, 3)

    def ComputePerceptionMotion(
        self,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        endpointPresent: torch.Tensor,
        jointAngularVelocity: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batchSize = int(rotation.size(0))
        perceptionIndices = torch.tensor(self.ContractView.perception_view_indices, device=rotation.device, dtype=torch.long)
        current = rotation.index_select(1, perceptionIndices)
        currentTranslation = translation.index_select(1, perceptionIndices)
        currentStatePresent = endpointPresent.index_select(1, perceptionIndices)
        perceptionCount = int(current.size(1))
        currentQuaternion = self.MatrixToQuaternion(current)
        deltaQuaternion = rotation.new_zeros(batchSize, perceptionCount, 4)
        deltaQuaternion[..., 3] = 1.0
        angularVelocity = torch.zeros_like(jointAngularVelocity)
        motionPresent = torch.zeros(batchSize, perceptionCount, device=rotation.device, dtype=torch.bool)
        if perceptionCount == 0:
            return currentQuaternion, deltaQuaternion, angularVelocity, motionPresent
        if self.PerceptionTranslationReference is None:
            self.PerceptionTranslationReference = torch.zeros_like(currentTranslation)
            self.PerceptionTranslationKnown = torch.zeros_like(currentStatePresent)
        drift = torch.linalg.vector_norm(currentTranslation - self.PerceptionTranslationReference, dim=-1)
        if bool((currentStatePresent & self.PerceptionTranslationKnown & drift.gt(self.PerceptionTranslationTolerance)).any().item()):
            raise ValueError("pure-rotation perception endpoint translated")
        establish = currentStatePresent & ~self.PerceptionTranslationKnown
        self.PerceptionTranslationReference = torch.where(
            establish.unsqueeze(-1),
            currentTranslation,
            self.PerceptionTranslationReference).detach().clone()
        self.PerceptionTranslationKnown = (self.PerceptionTranslationKnown | currentStatePresent).detach().clone()
        if self.LastPerceptionRotation is None:
            self.LastPerceptionRotation = current.detach().clone()
            self.LastPerceptionStatePresent = torch.zeros_like(currentStatePresent)
        motionPresent = currentStatePresent & self.LastPerceptionStatePresent
        deltaMatrix = self.LastPerceptionRotation.transpose(-1, -2) @ current
        computedDelta = self.MatrixToQuaternion(deltaMatrix)
        angularVelocity = torch.where(
            motionPresent.unsqueeze(-1),
            jointAngularVelocity,
            angularVelocity)
        deltaQuaternion = torch.where(motionPresent.unsqueeze(-1), computedDelta, deltaQuaternion)
        self.LastPerceptionRotation = torch.where(
            currentStatePresent.unsqueeze(-1).unsqueeze(-1),
            current,
            self.LastPerceptionRotation).detach().clone()
        self.LastPerceptionStatePresent = currentStatePresent.detach().clone()
        return currentQuaternion, deltaQuaternion, angularVelocity, motionPresent

    def EncodePerceptionMotion(
        self,
        rotationDelta: torch.Tensor,
        angularVelocity: torch.Tensor,
        motionPresent: torch.Tensor,
    ) -> torch.Tensor:
        features = []
        for perceptionIndex, endpointIndex in enumerate(self.ContractView.perception_view_indices):
            projection = self.RotationProjection.Matrix(
                endpointIndex,
                angularVelocity.device,
                angularVelocity.dtype)
            compactVelocity = angularVelocity[:, perceptionIndex] @ projection
            feature = torch.cat((rotationDelta[:, perceptionIndex], compactVelocity), dim=-1)
            features.append(torch.where(
                motionPresent[:, perceptionIndex].unsqueeze(-1),
                feature,
                torch.zeros_like(feature)))
        return torch.cat(features, dim=-1) if features else angularVelocity.new_zeros(int(angularVelocity.size(0)), 0)

    def EncodeFeedback(
        self,
        rawPayload: Any,
        device: torch.device,
        batchSize: Optional[int] = None,
    ) -> BrainFeedbackPacket:
        (
            position,
            velocity,
            translation,
            rotationQuaternion,
            endpointPresent,
            contractId,
            timestamp,
        ) = self.DecodeFeedback(rawPayload, torch.device(device))
        self.ValidateFeedback(
            position,
            velocity,
            translation,
            rotationQuaternion,
            endpointPresent,
            contractId,
            timestamp,
            batchSize,
        )
        jointFeatures = self.EncodeJointFeatures(position, velocity)
        endpointFeatures, rotation = self.EncodeEndEffectorFeatures(
            translation,
            rotationQuaternion,
            endpointPresent)
        jointAngularVelocity = self.EncodePerceptionAngularVelocity(velocity)
        (
            perceptionRotation,
            perceptionRotationDelta,
            perceptionAngularVelocity,
            perceptionMotionPresent,
        ) = self.ComputePerceptionMotion(
            translation,
            rotation,
            endpointPresent,
            jointAngularVelocity)
        progress, reached, childEnabled = self.EvaluateHierarchy(
            translation,
            rotation,
            endpointPresent)
        perceptionMotionFeatures = self.EncodePerceptionMotion(
            perceptionRotationDelta,
            perceptionAngularVelocity,
            perceptionMotionPresent)
        actualBatch = int(position.size(0))
        if self.CachedTargetActive is None:
            targetActive = torch.zeros(actualBatch, len(self.EndEffectors), device=position.device, dtype=torch.bool)
            targetVersion = torch.full((actualBatch,), -1, device=position.device, dtype=torch.long)
        else:
            targetActive = self.CachedTargetActive.clone()
            targetVersion = self.CachedTargetVersion.clone()
        packet = BrainFeedbackPacket(
            joint_features=jointFeatures,
            end_effector_features=endpointFeatures,
            endpoint_present=endpointPresent,
            progress=progress,
            reached=reached,
            child_enabled=childEnabled,
            target_active=targetActive,
            target_version=targetVersion,
            perception_rotation=perceptionRotation,
            perception_rotation_delta=perceptionRotationDelta,
            perception_angular_velocity=perceptionAngularVelocity,
            perception_motion_features=perceptionMotionFeatures,
            perception_motion_present=perceptionMotionPresent,
            contract_id=self.ContractView.contract_id,
            model_signature=self.ContractView.model_signature,
            timestamp=timestamp)
        self.LastTimestamp = timestamp.detach().clone()
        return packet
