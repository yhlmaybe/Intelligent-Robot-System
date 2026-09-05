from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple
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


class SlotExecutionStatus(IntEnum):
    UNKNOWN = 0
    APPLIED = 1
    MODIFIED = 2
    REJECTED = 3
    HELD = 4
    STOPPED = 5


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
    joint_ids: Tuple[str, ...]
    translation_error_scale: float = 1.0
    rotation_error_scale: float = 1.0
    observation_timeout: float = 0.25

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
    joints: Tuple[JointDefinition, ...]
    end_effectors: Tuple[EndEffector, ...]
    perception_calibrations: Tuple[PerceptionCalibration, ...]


@dataclass(frozen=True)
class RobotEmbodimentContractView:
    rotation_chart_limit: float
    joint_count: int
    end_effector_count: int
    joint_feedback_layout: PackedLayout
    end_effector_feedback_layout: PackedLayout
    end_effector_target_layout: PackedLayout
    perception_motion_layout: PackedLayout
    static_joint_tokens: Tuple[Tuple[float, ...], ...]
    static_end_effector_tokens: Tuple[Tuple[float, ...], ...]
    effector_joint_offsets: Tuple[int, ...]
    effector_joint_indices: Tuple[int, ...]
    end_effector_translation_basis: PackedTensor
    end_effector_rotation_basis: PackedTensor
    end_effector_target_lower: Tuple[float, ...]
    end_effector_target_upper: Tuple[float, ...]
    end_effector_target_tolerance: Tuple[float, ...]
    parent_index: Tuple[int, ...]
    topological_layers: Tuple[Tuple[int, ...], ...]
    root_mask: Tuple[bool, ...]
    child_mask: Tuple[bool, ...]
    independent_mask: Tuple[bool, ...]
    observation_timeout: Tuple[float, ...]
    perception_view_indices: Tuple[int, ...]
    perception_projection: Optional[PerceptionProjectionView]
    primary_perception_view_index: Optional[int]
    model_shape: EmbodimentShape

    def TargetSlotsMatch(
        self,
        leftValues: torch.Tensor,
        leftActive: torch.Tensor,
        rightValues: torch.Tensor,
        rightActive: torch.Tensor,
    ) -> torch.Tensor:
        tolerance = leftValues.new_tensor(
            self.end_effector_target_tolerance).unsqueeze(0)
        coordinateMatch = (leftValues - rightValues).abs().le(tolerance)
        slotMatch = torch.zeros_like(leftActive)
        for endpointIndex in range(self.end_effector_count):
            targetSlice = self.end_effector_target_layout.Slice(endpointIndex)
            activeMatch = leftActive[:, endpointIndex].eq(
                rightActive[:, endpointIndex])
            valuesMatch = coordinateMatch[:, targetSlice].all(dim=-1)
            slotMatch[:, endpointIndex] = (
                activeMatch
                & (~leftActive[:, endpointIndex] | valuesMatch))
        return slotMatch

    def TargetRowsMatch(
        self,
        leftValues: torch.Tensor,
        leftActive: torch.Tensor,
        rightValues: torch.Tensor,
        rightActive: torch.Tensor,
    ) -> torch.Tensor:
        return self.TargetSlotsMatch(
            leftValues,
            leftActive,
            rightValues,
            rightActive).all(dim=-1)


@dataclass(frozen=True)
class PackedEndEffectorTarget:
    values: torch.Tensor
    active: torch.Tensor
    target_version: torch.Tensor
    timestamp: torch.Tensor

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
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
        tolerance = torch.tensor(
            contractView.end_effector_target_tolerance,
            device=self.values.device,
            dtype=torch.float64).unsqueeze(0)
        for endpointIndex in range(contractView.end_effector_count):
            targetSlice = contractView.end_effector_target_layout.Slice(endpointIndex)
            active = self.active[:, endpointIndex].unsqueeze(-1)
            outside = (
                (
                    values[:, targetSlice]
                    < lower[:, targetSlice] - tolerance[:, targetSlice])
                | (
                    values[:, targetSlice]
                    > upper[:, targetSlice] + tolerance[:, targetSlice]))
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
                physicalRotation = coordinates @ basis.transpose(0, 1)
                norm = torch.linalg.vector_norm(physicalRotation, dim=-1)
                rotationSlice = slice(
                    targetSlice.start + translationWidth,
                    targetSlice.start + translationWidth + rotationWidth)
                rotationTolerance = tolerance[:, rotationSlice]
                mappedTolerance = torch.linalg.vector_norm(
                    rotationTolerance @ basis.abs().transpose(0, 1),
                    dim=-1)
                epsilon = torch.finfo(torch.float64).eps
                productRoundoff = (
                    rotationWidth * epsilon
                    / (1.0 - rotationWidth * epsilon))
                normRoundoff = (
                    int(basis.size(0)) * epsilon
                    / (1.0 - int(basis.size(0)) * epsilon))
                arithmeticTolerance = (
                    productRoundoff
                    * torch.linalg.vector_norm(
                        coordinates.abs() @ basis.abs().transpose(0, 1),
                        dim=-1)
                    + normRoundoff * norm)
                chartTolerance = mappedTolerance + arithmeticTolerance
                if bool((
                    self.active[:, endpointIndex]
                    & norm.gt(
                        contractView.rotation_chart_limit + chartTolerance)
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
class ActionRequest:
    request_id: torch.Tensor
    action_epoch: torch.Tensor
    target: PackedEndEffectorTarget
    command_active: torch.Tensor
    hold_requested: torch.Tensor
    stop_requested: torch.Tensor
    help_requested: torch.Tensor
    policy_path: torch.Tensor
    planner_override: torch.Tensor
    temporal_kind_id: torch.Tensor
    timestamp: torch.Tensor

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
        self.target.Validate(contractView)
        batchSize = int(self.target.values.size(0))
        device = self.target.values.device
        booleanFields = (
            self.command_active,
            self.hold_requested,
            self.stop_requested,
            self.help_requested,
            self.planner_override,
        )
        integerFields = (
            self.request_id,
            self.action_epoch,
            self.policy_path,
            self.temporal_kind_id,
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.dtype != torch.bool
            or value.device != device
            for value in booleanFields
        ):
            raise ValueError("action request masks must match the target batch")
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.dtype != torch.long
            or value.device != device
            or bool(value.lt(0).any().item())
            for value in integerFields
        ):
            raise ValueError("action request identifiers must be non-negative long vectors")
        Robot.ValidateTensor(self.timestamp, (batchSize,), "action request timestamp", True)
        if self.timestamp.device != device or bool(self.timestamp.lt(0.0).any().item()):
            raise ValueError("action request timestamp must match the target batch")
        if bool((self.stop_requested & self.target.active.any(dim=-1)).any().item()):
            raise ValueError("a stop request cannot carry an active target")


@dataclass(frozen=True)
class ActionExecutionResult:
    request_id: torch.Tensor
    action_epoch: torch.Tensor
    applied_target: PackedEndEffectorTarget
    execution_status: torch.Tensor
    execution_known: torch.Tensor
    hard_stop: torch.Tensor
    help_accepted: torch.Tensor
    timestamp: torch.Tensor

    def Validate(
        self,
        request: ActionRequest,
        contractView: RobotEmbodimentContractView,
    ) -> None:
        request.Validate(contractView)
        self.applied_target.Validate(contractView)
        batchSize = int(request.target.values.size(0))
        endpointCount = contractView.end_effector_count
        device = request.target.values.device
        if (
            self.applied_target.values.device != device
            or self.applied_target.values.dtype != request.target.values.dtype
            or int(self.applied_target.values.size(0)) != batchSize
        ):
            raise ValueError("applied target must match the action request batch")
        integerFields = (self.request_id, self.action_epoch)
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.dtype != torch.long
            or value.device != device
            or bool(value.lt(0).any().item())
            for value in integerFields
        ):
            raise ValueError("action execution identifiers must be non-negative long vectors")
        if not torch.equal(self.request_id, request.request_id):
            raise ValueError("action execution result does not match the request id")
        if not torch.equal(self.action_epoch, request.action_epoch):
            raise ValueError("action execution result does not match the action epoch")
        if (
            not torch.is_tensor(self.execution_status)
            or tuple(self.execution_status.shape) != (batchSize, endpointCount)
            or self.execution_status.dtype != torch.long
            or self.execution_status.device != device
            or bool(self.execution_status.lt(int(SlotExecutionStatus.UNKNOWN)).any().item())
            or bool(self.execution_status.gt(int(SlotExecutionStatus.STOPPED)).any().item())
        ):
            raise ValueError("slot execution status is invalid")
        booleanFields = (
            (self.execution_known, (batchSize, endpointCount)),
            (self.hard_stop, (batchSize,)),
            (self.help_accepted, (batchSize,)),
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != shape
            or value.dtype != torch.bool
            or value.device != device
            for value, shape in booleanFields
        ):
            raise ValueError("action execution masks must match the request batch")
        unknown = self.execution_status.eq(int(SlotExecutionStatus.UNKNOWN))
        if bool((unknown != ~self.execution_known).any().item()):
            raise ValueError("unknown execution status must match execution knowledge")
        Robot.ValidateTensor(self.timestamp, (batchSize,), "action execution timestamp", True)
        if (
            self.timestamp.device != device
            or bool(self.timestamp.lt(request.timestamp).any().item())
        ):
            raise ValueError("action execution timestamp precedes its request")
        if bool((self.help_accepted & ~request.help_requested).any().item()):
            raise ValueError("help cannot be accepted without a help request")
        if bool((self.hard_stop & self.applied_target.active.any(dim=-1)).any().item()):
            raise ValueError("hard stop requires an empty applied target")


@dataclass(frozen=True)
class BrainFeedbackPacket:
    joint_features: torch.Tensor
    end_effector_features: torch.Tensor
    endpoint_present: torch.Tensor
    progress: torch.Tensor
    reached: torch.Tensor
    phase_enabled: torch.Tensor
    phase_known: torch.Tensor
    observation_age: torch.Tensor
    applied_target_values: torch.Tensor
    applied_target_active: torch.Tensor
    applied_target_version: torch.Tensor
    applied_action_epoch: torch.Tensor
    execution_status: torch.Tensor
    execution_known: torch.Tensor
    execution_relevant: torch.Tensor
    execution_result_known: torch.Tensor
    hard_stop: torch.Tensor
    help_accepted: torch.Tensor
    perception_rotation: torch.Tensor
    perception_rotation_delta: torch.Tensor
    perception_angular_velocity: torch.Tensor
    perception_motion_features: torch.Tensor
    perception_motion_present: torch.Tensor
    timestamp: torch.Tensor

    def Validate(self, contractView: RobotEmbodimentContractView) -> None:
        if not torch.is_tensor(self.joint_features) or self.joint_features.dim() != 2:
            raise ValueError("joint features must be a batched matrix")
        batchSize = int(self.joint_features.size(0))
        device = self.joint_features.device
        dtype = self.joint_features.dtype
        floatingFields = (
            (self.joint_features, (batchSize, contractView.joint_feedback_layout.PackedDim)),
            (self.end_effector_features, (batchSize, contractView.end_effector_feedback_layout.PackedDim)),
            (self.progress, (batchSize, contractView.end_effector_count)),
            (self.observation_age, (batchSize, contractView.end_effector_count)),
            (self.applied_target_values, (batchSize, contractView.end_effector_target_layout.PackedDim)),
            (self.perception_rotation, (batchSize, len(contractView.perception_view_indices), 4)),
            (self.perception_rotation_delta, (batchSize, len(contractView.perception_view_indices), 4)),
            (self.perception_angular_velocity, (batchSize, len(contractView.perception_view_indices), 3)),
            (self.perception_motion_features, (batchSize, contractView.perception_motion_layout.PackedDim)),
        )
        booleanFields = (
            (self.endpoint_present, (batchSize, contractView.end_effector_count)),
            (self.reached, (batchSize, contractView.end_effector_count)),
            (self.phase_enabled, (batchSize, contractView.end_effector_count)),
            (self.phase_known, (batchSize, contractView.end_effector_count)),
            (self.applied_target_active, (batchSize, contractView.end_effector_count)),
            (self.execution_known, (batchSize, contractView.end_effector_count)),
            (self.execution_relevant, (batchSize, contractView.end_effector_count)),
            (self.execution_result_known, (batchSize,)),
            (self.hard_stop, (batchSize,)),
            (self.help_accepted, (batchSize,)),
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
            not torch.is_tensor(self.applied_target_version)
            or tuple(self.applied_target_version.shape) != (batchSize,)
            or self.applied_target_version.dtype != torch.long
            or self.applied_target_version.device != device
            or bool(self.applied_target_version.lt(-1).any().item())
        ):
            raise ValueError("feedback applied target version is invalid")
        if (
            not torch.is_tensor(self.applied_action_epoch)
            or tuple(self.applied_action_epoch.shape) != (batchSize,)
            or self.applied_action_epoch.dtype != torch.long
            or self.applied_action_epoch.device != device
            or bool(self.applied_action_epoch.lt(0).any().item())
        ):
            raise ValueError("feedback applied action epoch is invalid")
        if (
            not torch.is_tensor(self.execution_status)
            or tuple(self.execution_status.shape) != (batchSize, contractView.end_effector_count)
            or self.execution_status.dtype != torch.long
            or self.execution_status.device != device
            or bool(self.execution_status.lt(int(SlotExecutionStatus.UNKNOWN)).any().item())
            or bool(self.execution_status.gt(int(SlotExecutionStatus.STOPPED)).any().item())
        ):
            raise ValueError("feedback execution status is invalid")
        unknown = self.execution_status.eq(int(SlotExecutionStatus.UNKNOWN))
        if bool((unknown != ~self.execution_known).any().item()):
            raise ValueError("feedback execution status knowledge is inconsistent")
        if bool((self.phase_enabled & ~self.phase_known).any().item()):
            raise ValueError("an unknown hierarchy phase cannot be enabled")
        if bool((self.hard_stop & self.applied_target_active.any(dim=-1)).any().item()):
            raise ValueError("hard stop feedback cannot expose an active target")
        any_relevant = self.execution_relevant.any(dim=-1)
        result_known = torch.where(
            any_relevant,
            (self.execution_known | ~self.execution_relevant).all(dim=-1),
            self.execution_known.all(dim=-1))
        if bool((self.execution_result_known != result_known).any().item()):
            raise ValueError("feedback execution result knowledge is inconsistent")
        if bool(self.observation_age.lt(0.0).any().item()):
            raise ValueError("feedback observation age cannot be negative")
        if bool((self.applied_target_version.lt(0) & self.applied_target_active.any(dim=-1)).any().item()):
            raise ValueError("active applied target requires an applied target version")
        for endpointIndex, parentIndex in enumerate(contractView.parent_index):
            targetSlice = contractView.end_effector_target_layout.Slice(endpointIndex)
            inactiveValues = self.applied_target_values[:, targetSlice].ne(0.0).any(dim=-1)
            if bool((~self.applied_target_active[:, endpointIndex] & inactiveValues).any().item()):
                raise ValueError("inactive applied target coordinates must be zero")
            if parentIndex >= 0 and bool((
                self.applied_target_active[:, endpointIndex]
                & ~self.applied_target_active[:, parentIndex]
            ).any().item()):
                raise ValueError("active applied child target requires its parent target")

    def IndexSelectRows(self, rowIndex: torch.Tensor) -> "BrainFeedbackPacket":
        return BrainFeedbackPacket(
            joint_features=self.joint_features.index_select(0, rowIndex),
            end_effector_features=self.end_effector_features.index_select(0, rowIndex),
            endpoint_present=self.endpoint_present.index_select(0, rowIndex),
            progress=self.progress.index_select(0, rowIndex),
            reached=self.reached.index_select(0, rowIndex),
            phase_enabled=self.phase_enabled.index_select(0, rowIndex),
            phase_known=self.phase_known.index_select(0, rowIndex),
            observation_age=self.observation_age.index_select(0, rowIndex),
            applied_target_values=self.applied_target_values.index_select(0, rowIndex),
            applied_target_active=self.applied_target_active.index_select(0, rowIndex),
            applied_target_version=self.applied_target_version.index_select(0, rowIndex),
            applied_action_epoch=self.applied_action_epoch.index_select(0, rowIndex),
            execution_status=self.execution_status.index_select(0, rowIndex),
            execution_known=self.execution_known.index_select(0, rowIndex),
            execution_relevant=self.execution_relevant.index_select(0, rowIndex),
            execution_result_known=self.execution_result_known.index_select(0, rowIndex),
            hard_stop=self.hard_stop.index_select(0, rowIndex),
            help_accepted=self.help_accepted.index_select(0, rowIndex),
            perception_rotation=self.perception_rotation.index_select(0, rowIndex),
            perception_rotation_delta=self.perception_rotation_delta.index_select(0, rowIndex),
            perception_angular_velocity=self.perception_angular_velocity.index_select(0, rowIndex),
            perception_motion_features=self.perception_motion_features.index_select(0, rowIndex),
            perception_motion_present=self.perception_motion_present.index_select(0, rowIndex),
            timestamp=self.timestamp.index_select(0, rowIndex))

    def RepeatCandidates(self, candidateCount: int) -> "BrainFeedbackPacket":
        rows = torch.arange(
            int(self.joint_features.size(0)),
            device=self.joint_features.device,
            dtype=torch.long).repeat_interleave(candidateCount)
        return self.IndexSelectRows(rows)


class Robot:
    RotationStateWidth = 6
    PrincipalRotationLimit = math.pi - 1e-4
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
        "timestamp",
    })
    TargetPayloadFields = frozenset({
        "target_version",
        "timestamp",
        "slot_ids",
        "reference_frame_ids",
        "active",
        "translation",
        "rotation_vector",
    })
    ActionRequestPayloadFields = frozenset({
        "request_id",
        "action_epoch",
        "command_active",
        "hold_requested",
        "stop_requested",
        "help_requested",
        "policy_path",
        "planner_override",
        "temporal_kind_id",
        "timestamp",
        "end_effector_target",
    })
    ActionExecutionResultPayloadFields = frozenset({
        "request_id",
        "action_epoch",
        "applied_target",
        "execution_status",
        "execution_known",
        "hard_stop",
        "help_accepted",
        "timestamp",
    })

    def __init__(self, definition: RobotDefinition) -> None:
        self.PerceptionJointIndices = self.ValidateDefinition(definition)
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
        self.ObservationTimeout = tuple(float(value.observation_timeout) for value in self.EndEffectors)
        self.ContractView = self.CompileContractView(definition)
        self.TranslationProjection = self.CompileBasisProjection(
            self.ContractView.end_effector_translation_basis)
        self.RotationProjection = self.CompileBasisProjection(
            self.ContractView.end_effector_rotation_basis)
        self.Reset()

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
    def CompileTargetCoordinateTolerance(
        cls,
        endEffectors: Sequence[EndEffector],
    ) -> Tuple[float, ...]:
        epsilon = float(torch.finfo(torch.float32).eps)
        tolerances = []
        for endpoint in endEffectors:
            coordinateOffset = 0
            for basisRows in (
                endpoint.translation_basis,
                endpoint.rotation_basis,
            ):
                basis = torch.tensor(basisRows, dtype=torch.float64)
                coordinateCount = int(basis.size(1))
                if coordinateCount == 0:
                    continue
                condition = float(torch.linalg.cond(basis).item())
                operationCount = 2 * int(basis.size(0))
                roundoff = (
                    operationCount * epsilon
                    / (1.0 - operationCount * epsilon))
                for coordinateIndex in range(coordinateCount):
                    lower = float(endpoint.target_lower[
                        coordinateOffset + coordinateIndex])
                    upper = float(endpoint.target_upper[
                        coordinateOffset + coordinateIndex])
                    scale = max(
                        1.0,
                        abs(lower),
                        abs(upper),
                        upper - lower)
                    tolerances.append(roundoff * condition * scale)
                coordinateOffset += coordinateCount
        return tuple(tolerances)

    @classmethod
    def CompilePerceptionJointBindings(
        cls,
        joints: Sequence[JointDefinition],
        endEffectors: Sequence[EndEffector],
    ) -> Tuple[Tuple[int, ...], ...]:
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
        for endpoint in perceptionById.values():
            indices = tuple(bindings[endpoint.effector_id])
            if not indices:
                raise ValueError(
                    "perception slot requires bound rotational joints")
            jointIndices.append(indices)
        return tuple(jointIndices)

    @classmethod
    def ValidateDefinition(
        cls,
        definition: RobotDefinition,
    ) -> Tuple[Tuple[int, ...], ...]:
        if type(definition) is not RobotDefinition:
            raise TypeError("robot definition has the wrong type")
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
        jointSet = set(jointIds)
        for endpoint in definition.end_effectors:
            cls.ValidateIdentifier(endpoint.effector_id, "end-effector id")
            if type(endpoint.effector_type) is not EndEffectorType:
                raise ValueError("end-effector type is invalid")
            if endpoint.parent_effector_id is not None and endpoint.parent_effector_id not in endpointSet:
                raise ValueError("end-effector parent is unknown")
            if not endpoint.joint_ids or any(
                jointId not in jointSet
                for jointId in endpoint.joint_ids
            ):
                raise ValueError("end-effector joints are invalid")
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
                or not math.isfinite(float(endpoint.observation_timeout))
                or endpoint.observation_timeout <= 0.0
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
        jointIndex = {
            value.joint_id: index
            for index, value in enumerate(self.Joints)}
        effectorJointWidths = tuple(
            len(endpoint.joint_ids)
            for endpoint in self.EndEffectors)
        effectorJointOffsets = PackedLayout.FromWidths(
            effectorJointWidths).offsets
        effectorJointIndices = tuple(
            jointIndex[jointId]
            for endpoint in self.EndEffectors
            for jointId in endpoint.joint_ids)
        translationBasis = PackedTensor.FromMatrices(tuple(value.translation_basis for value in self.EndEffectors))
        rotationBasis = PackedTensor.FromMatrices(tuple(value.rotation_basis for value in self.EndEffectors))
        targetLower = tuple(value for endpoint in self.EndEffectors for value in endpoint.target_lower)
        targetUpper = tuple(value for endpoint in self.EndEffectors for value in endpoint.target_upper)
        targetTolerance = self.CompileTargetCoordinateTolerance(
            self.EndEffectors)
        view = RobotEmbodimentContractView(
            rotation_chart_limit=self.PrincipalRotationLimit,
            joint_count=len(self.Joints),
            end_effector_count=len(self.EndEffectors),
            joint_feedback_layout=jointLayout,
            end_effector_feedback_layout=feedbackLayout,
            end_effector_target_layout=targetLayout,
            perception_motion_layout=perceptionLayout,
            static_joint_tokens=jointTokens,
            static_end_effector_tokens=endpointTokens,
            effector_joint_offsets=effectorJointOffsets,
            effector_joint_indices=effectorJointIndices,
            end_effector_translation_basis=translationBasis,
            end_effector_rotation_basis=rotationBasis,
            end_effector_target_lower=targetLower,
            end_effector_target_upper=targetUpper,
            end_effector_target_tolerance=targetTolerance,
            parent_index=parentIndex,
            topological_layers=layers,
            root_mask=rootMask,
            child_mask=childMask,
            independent_mask=independentMask,
            observation_timeout=tuple(
                float(value.observation_timeout)
                for value in self.EndEffectors),
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
        urdfReader: Callable[[Any], Mapping[str, Any]],
        srdfReader: Callable[[Any], Mapping[str, Any]],
    ) -> "Robot":
        urdf = cls.ParseUrdf(urdfSource, urdfReader)
        srdf = cls.ParseSrdf(srdfSource, srdfReader)
        definition = RobotDefinition(
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
        armJointIds = tuple(joint.joint_id for joint in joints)
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
            dwell_cycles=3,
            joint_ids=armJointIds)]
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
            dwell_cycles=3,
            joint_ids=armJointIds + tuple(
                f"{sideId}_{fingerName}_{coordinateIndex}"
                for coordinateIndex in range(4)))
            for fingerName in fingerNames)
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
            dwell_cycles=2,
            joint_ids=tuple(joint.joint_id for joint in cameraJoints))
        return cls(RobotDefinition(
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
        self.CachedActionEpoch: Optional[torch.Tensor] = None
        self.CachedRequestId: Optional[torch.Tensor] = None
        self.CachedExecutionStatus: Optional[torch.Tensor] = None
        self.CachedExecutionKnown: Optional[torch.Tensor] = None
        self.CachedExecutionRelevant: Optional[torch.Tensor] = None
        self.CachedExecutionResultKnown: Optional[torch.Tensor] = None
        self.CachedExecutionTimestamp: Optional[torch.Tensor] = None
        self.CachedHardStop: Optional[torch.Tensor] = None
        self.CachedHelpAccepted: Optional[torch.Tensor] = None
        self.DwellState: Optional[torch.Tensor] = None
        self.ReachedState: Optional[torch.Tensor] = None
        self.ProgressState: Optional[torch.Tensor] = None
        self.ObservationAgeState: Optional[torch.Tensor] = None
        self.ObservationKnownState: Optional[torch.Tensor] = None
        self.LastTimestamp: Optional[torch.Tensor] = None
        self.LastPerceptionRotation: Optional[torch.Tensor] = None
        self.LastPerceptionStatePresent: Optional[torch.Tensor] = None

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        payloads = tuple(rawPayload) if isinstance(rawPayload, (tuple, list)) else None
        if payloads is not None:
            if not payloads:
                raise ValueError("robot feedback batch cannot be empty")
            for payload in payloads:
                self.ValidatePayload(payload)
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
        return position, velocity, translation, rotation, present, timestamp

    def ValidateFeedback(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        endpointPresent: torch.Tensor,
        timestamp: torch.Tensor,
        batchSize: Optional[int],
    ) -> None:
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
        if (
            self.CachedExecutionTimestamp is not None
            and bool(timestamp.lt(self.CachedExecutionTimestamp).any().item())
        ):
            raise ValueError("feedback time precedes the applied action result")
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
            "target_version": target.target_version.detach().cpu().tolist(),
            "timestamp": target.timestamp.detach().cpu().tolist(),
            "slot_ids": list(self.EndEffectorIds),
            "reference_frame_ids": list(self.ReferenceFrameIds),
            "active": target.active.detach().cpu().tolist(),
            "translation": translations,
            "rotation_vector": rotations,
        }

    def BuildActionRequest(
        self,
        target: PackedEndEffectorTarget,
        requestId: torch.Tensor,
        actionEpoch: torch.Tensor,
        commandActive: torch.Tensor,
        holdRequested: torch.Tensor,
        stopRequested: torch.Tensor,
        helpRequested: torch.Tensor,
        policyPath: torch.Tensor,
        plannerOverride: torch.Tensor,
        temporalKindId: torch.Tensor,
        timestamp: Optional[torch.Tensor] = None,
    ) -> ActionRequest:
        request = ActionRequest(
            request_id=requestId,
            action_epoch=actionEpoch,
            target=target,
            command_active=commandActive,
            hold_requested=holdRequested,
            stop_requested=stopRequested,
            help_requested=helpRequested,
            policy_path=policyPath,
            planner_override=plannerOverride,
            temporal_kind_id=temporalKindId,
            timestamp=target.timestamp if timestamp is None else timestamp)
        request.Validate(self.ContractView)
        return request

    def EncodeActionRequest(
        self,
        request: ActionRequest,
    ) -> Mapping[str, Any]:
        request.Validate(self.ContractView)
        return {
            "request_id": request.request_id.detach().cpu().tolist(),
            "action_epoch": request.action_epoch.detach().cpu().tolist(),
            "command_active": request.command_active.detach().cpu().tolist(),
            "hold_requested": request.hold_requested.detach().cpu().tolist(),
            "stop_requested": request.stop_requested.detach().cpu().tolist(),
            "help_requested": request.help_requested.detach().cpu().tolist(),
            "policy_path": request.policy_path.detach().cpu().tolist(),
            "planner_override": request.planner_override.detach().cpu().tolist(),
            "temporal_kind_id": request.temporal_kind_id.detach().cpu().tolist(),
            "timestamp": request.timestamp.detach().cpu().tolist(),
            "end_effector_target": self.DecodeTarget(request.target),
        }

    @staticmethod
    def DecodeActionVector(
        value: Any,
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
        fieldName: str,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device)
        if tuple(tensor.shape) != shape:
            raise ValueError(fieldName + " has the wrong shape")
        if dtype is torch.bool:
            if tensor.dtype != torch.bool:
                raise TypeError(fieldName + " must contain booleans")
        elif tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError(fieldName + " must contain integers")
        return tensor.to(dtype=dtype)

    def EncodeTargetPayload(
        self,
        payload: Any,
        device: torch.device,
    ) -> PackedEndEffectorTarget:
        device = torch.device(device)
        if not isinstance(payload, Mapping) or set(payload) != self.TargetPayloadFields:
            raise ValueError("applied target payload fields do not match")
        if (
            tuple(payload["slot_ids"]) != self.EndEffectorIds
            or tuple(payload["reference_frame_ids"]) != self.ReferenceFrameIds
        ):
            raise ValueError("applied target payload does not match the robot contract")
        endpointCount = len(self.EndEffectors)
        activeValue = torch.as_tensor(payload["active"])
        if activeValue.dim() != 2 or int(activeValue.size(1)) != endpointCount:
            raise ValueError("applied target active mask has the wrong shape")
        batchSize = int(activeValue.size(0))
        active = self.DecodeActionVector(
            payload["active"],
            (batchSize, endpointCount),
            device,
            torch.bool,
            "applied target active mask")
        translations = payload["translation"]
        rotations = payload["rotation_vector"]
        if (
            not isinstance(translations, (tuple, list))
            or not isinstance(rotations, (tuple, list))
            or len(translations) != batchSize
            or len(rotations) != batchSize
            or any(
                not isinstance(row, (tuple, list))
                or len(row) != endpointCount
                for row in tuple(translations) + tuple(rotations)
            )
        ):
            raise ValueError("applied target poses have the wrong shape")
        values = torch.zeros(
            batchSize,
            self.ContractView.end_effector_target_layout.PackedDim,
            device=device,
            dtype=torch.float32)
        tolerance = 1e-5
        for batchIndex in range(batchSize):
            for endpointIndex in range(endpointCount):
                targetSlice = self.ContractView.end_effector_target_layout.Slice(endpointIndex)
                translationWidth = self.ContractView.end_effector_translation_basis.shapes[endpointIndex][1]
                rotationWidth = self.ContractView.end_effector_rotation_basis.shapes[endpointIndex][1]
                translationValue = translations[batchIndex][endpointIndex]
                rotationValue = rotations[batchIndex][endpointIndex]
                if not bool(active[batchIndex, endpointIndex].item()):
                    if translationValue is not None or rotationValue is not None:
                        raise ValueError("inactive applied target pose must be absent")
                    continue
                if translationWidth:
                    physicalTranslation = torch.as_tensor(
                        translationValue,
                        device=device,
                        dtype=torch.float32)
                    self.ValidateTensor(
                        physicalTranslation,
                        (3,),
                        "applied target translation",
                        True)
                    projection = self.TranslationProjection.Matrix(
                        endpointIndex,
                        device,
                        torch.float32)
                    basis = self.ContractView.end_effector_translation_basis.Matrix(
                        endpointIndex,
                        device,
                        torch.float32)
                    coordinates = physicalTranslation @ projection
                    if bool((coordinates @ basis.transpose(0, 1) - physicalTranslation).abs().max().gt(tolerance).item()):
                        raise ValueError("applied target translation leaves the allowed subspace")
                    values[
                        batchIndex,
                        targetSlice.start:targetSlice.start + translationWidth,
                    ] = coordinates
                elif translationValue is not None:
                    raise ValueError("applied target contains forbidden translation")
                if rotationWidth:
                    physicalRotation = torch.as_tensor(
                        rotationValue,
                        device=device,
                        dtype=torch.float32)
                    self.ValidateTensor(
                        physicalRotation,
                        (3,),
                        "applied target rotation",
                        True)
                    projection = self.RotationProjection.Matrix(
                        endpointIndex,
                        device,
                        torch.float32)
                    basis = self.ContractView.end_effector_rotation_basis.Matrix(
                        endpointIndex,
                        device,
                        torch.float32)
                    coordinates = physicalRotation @ projection
                    if bool((coordinates @ basis.transpose(0, 1) - physicalRotation).abs().max().gt(tolerance).item()):
                        raise ValueError("applied target rotation leaves the allowed subspace")
                    values[
                        batchIndex,
                        targetSlice.start + translationWidth:targetSlice.stop,
                    ] = coordinates
                elif rotationValue is not None:
                    raise ValueError("applied target contains forbidden rotation")
        target = PackedEndEffectorTarget(
            values=values,
            active=active,
            target_version=self.DecodeActionVector(
                payload["target_version"],
                (batchSize,),
                device,
                torch.long,
                "applied target version"),
            timestamp=torch.as_tensor(
                payload["timestamp"],
                device=device,
                dtype=torch.float64))
        target.Validate(self.ContractView)
        return target

    def DecodeActionExecutionResult(
        self,
        payload: Any,
        pendingRequest: ActionRequest,
        device: torch.device,
    ) -> ActionExecutionResult:
        device = torch.device(device)
        pendingRequest.Validate(self.ContractView)
        if pendingRequest.target.values.device != device:
            raise ValueError("pending action request does not match the result device")
        if not isinstance(payload, Mapping) or set(payload) != self.ActionExecutionResultPayloadFields:
            raise ValueError("action execution result fields do not match")
        batchSize = int(pendingRequest.target.values.size(0))
        endpointCount = len(self.EndEffectors)
        statusRows = payload["execution_status"]
        if (
            not isinstance(statusRows, (tuple, list))
            or len(statusRows) != batchSize
            or any(
                not isinstance(row, (tuple, list))
                or len(row) != endpointCount
                for row in statusRows)
        ):
            raise ValueError("action execution status has the wrong shape")
        statusValues = []
        for row in statusRows:
            statusRow = []
            for value in row:
                if type(value) is str:
                    statusRow.append(int(SlotExecutionStatus[value]))
                elif type(value) is int:
                    statusRow.append(int(SlotExecutionStatus(value)))
                else:
                    raise TypeError("action execution status values are invalid")
            statusValues.append(tuple(statusRow))
        result = ActionExecutionResult(
            request_id=self.DecodeActionVector(
                payload["request_id"],
                (batchSize,),
                device,
                torch.long,
                "action execution request id"),
            action_epoch=self.DecodeActionVector(
                payload["action_epoch"],
                (batchSize,),
                device,
                torch.long,
                "action execution epoch"),
            applied_target=self.EncodeTargetPayload(
                payload["applied_target"],
                device),
            execution_status=torch.tensor(
                statusValues,
                device=device,
                dtype=torch.long),
            execution_known=self.DecodeActionVector(
                payload["execution_known"],
                (batchSize, endpointCount),
                device,
                torch.bool,
                "action execution knowledge"),
            hard_stop=self.DecodeActionVector(
                payload["hard_stop"],
                (batchSize,),
                device,
                torch.bool,
                "action execution hard stop"),
            help_accepted=self.DecodeActionVector(
                payload["help_accepted"],
                (batchSize,),
                device,
                torch.bool,
                "action execution help acceptance"),
            timestamp=torch.as_tensor(
                payload["timestamp"],
                device=device,
                dtype=torch.float64))
        result.Validate(pendingRequest, self.ContractView)
        return result

    def CanonicalizeTarget(
        self,
        target: PackedEndEffectorTarget,
    ) -> torch.Tensor:
        target.Validate(self.ContractView)
        canonicalValues = torch.zeros_like(target.values)
        for endpointIndex in range(len(self.EndEffectors)):
            targetSlice = self.ContractView.end_effector_target_layout.Slice(endpointIndex)
            canonicalValues[:, targetSlice] = torch.where(
                target.active[:, endpointIndex].unsqueeze(-1),
                target.values[:, targetSlice],
                torch.zeros_like(target.values[:, targetSlice]))
        return canonicalValues

    def InitializeExecutionState(self, template: torch.Tensor) -> None:
        batchSize = int(template.size(0))
        endpointCount = len(self.EndEffectors)
        self.CachedTargetValues = torch.zeros(
            batchSize,
            self.ContractView.end_effector_target_layout.PackedDim,
            device=template.device,
            dtype=template.dtype)
        self.CachedTargetActive = torch.zeros(
            batchSize,
            endpointCount,
            device=template.device,
            dtype=torch.bool)
        self.CachedTargetVersion = torch.full(
            (batchSize,),
            -1,
            device=template.device,
            dtype=torch.long)
        self.CachedActionEpoch = torch.zeros(
            batchSize,
            device=template.device,
            dtype=torch.long)
        self.CachedRequestId = torch.full(
            (batchSize,),
            -1,
            device=template.device,
            dtype=torch.long)
        self.CachedExecutionStatus = torch.full(
            (batchSize, endpointCount),
            int(SlotExecutionStatus.UNKNOWN),
            device=template.device,
            dtype=torch.long)
        self.CachedExecutionKnown = torch.zeros(
            batchSize,
            endpointCount,
            device=template.device,
            dtype=torch.bool)
        self.CachedExecutionRelevant = torch.zeros(
            batchSize,
            endpointCount,
            device=template.device,
            dtype=torch.bool)
        self.CachedExecutionResultKnown = torch.zeros(
            batchSize,
            device=template.device,
            dtype=torch.bool)
        self.CachedExecutionTimestamp = torch.zeros(
            batchSize,
            device=template.device,
            dtype=torch.float64)
        self.CachedHardStop = torch.zeros(
            batchSize,
            device=template.device,
            dtype=torch.bool)
        self.CachedHelpAccepted = torch.zeros(
            batchSize,
            device=template.device,
            dtype=torch.bool)
        self.DwellState = torch.zeros(
            batchSize,
            endpointCount,
            device=template.device,
            dtype=torch.long)
        self.ReachedState = torch.zeros(
            batchSize,
            endpointCount,
            device=template.device,
            dtype=torch.bool)
        self.ProgressState = torch.zeros(
            batchSize,
            endpointCount,
            device=template.device,
            dtype=template.dtype)

    def CommitAppliedTarget(
        self,
        request: ActionRequest,
        result: ActionExecutionResult,
    ) -> None:
        result.Validate(request, self.ContractView)
        target = result.applied_target
        canonicalValues = self.CanonicalizeTarget(target)
        requestedValues = self.CanonicalizeTarget(request.target)
        if self.CachedTargetValues is None:
            self.InitializeExecutionState(target.values)
        if (
            tuple(self.CachedTargetValues.shape) != tuple(target.values.shape)
            or self.CachedTargetValues.device != target.values.device
            or self.CachedTargetValues.dtype != target.values.dtype
        ):
            raise ValueError("target batch identity changed without robot reset")
        if bool(target.target_version.lt(self.CachedTargetVersion).any().item()):
            raise ValueError("applied target version cannot move backwards")
        if bool(result.action_epoch.lt(self.CachedActionEpoch).any().item()):
            raise ValueError("applied action epoch cannot move backwards")
        requestSlotMatches = self.ContractView.TargetSlotsMatch(
            canonicalValues,
            target.active,
            requestedValues,
            request.target.active)
        cachedSlotMatches = self.ContractView.TargetSlotsMatch(
            canonicalValues,
            target.active,
            self.CachedTargetValues,
            self.CachedTargetActive)
        previousActive = self.CachedTargetActive.detach().clone()
        for endpointIndex in range(len(self.EndEffectors)):
            status = result.execution_status[:, endpointIndex]
            exactRequest = requestSlotMatches[:, endpointIndex]
            unchanged = cachedSlotMatches[:, endpointIndex]
            if bool((status.eq(int(SlotExecutionStatus.APPLIED)) & ~exactRequest).any().item()):
                raise ValueError("applied slot status requires the requested target")
            if bool((
                (
                    status.eq(int(SlotExecutionStatus.UNKNOWN))
                    | status.eq(int(SlotExecutionStatus.REJECTED))
                    | status.eq(int(SlotExecutionStatus.HELD))
                )
                & ~unchanged
            ).any().item()):
                raise ValueError("unknown, rejected or held slot status requires the previous target")
            if bool((
                status.eq(int(SlotExecutionStatus.STOPPED))
                & target.active[:, endpointIndex]
            ).any().item()):
                raise ValueError("stopped slot status requires an inactive target")
        rowMatchesRequest = requestSlotMatches.all(dim=-1)
        rowMatchesCache = cachedSlotMatches.all(dim=-1)
        relevant = (
            previousActive
            | request.target.active
            | target.active)
        requestedSlotsApplied = (
            ~request.target.active
            | (
                result.execution_known
                & result.execution_status.eq(
                    int(SlotExecutionStatus.APPLIED)))
        ).all(dim=-1)
        stoppedSlots = result.execution_status.eq(
            int(SlotExecutionStatus.STOPPED))
        stoppedRequest = (
            (request.stop_requested | request.help_requested)
            & torch.where(
                relevant.any(dim=-1),
                (stoppedSlots | ~relevant).all(dim=-1),
                stoppedSlots.all(dim=-1)))
        requestSnapshot = (
            rowMatchesRequest
            & (
                (
                    request.target.active.any(dim=-1)
                    & requestedSlotsApplied)
                | stoppedRequest))
        previousStatus = (
            result.execution_status.eq(
                int(SlotExecutionStatus.UNKNOWN))
            | result.execution_status.eq(
                int(SlotExecutionStatus.REJECTED))
            | result.execution_status.eq(
                int(SlotExecutionStatus.HELD)))
        previousSnapshot = (
            rowMatchesCache
            & torch.where(
                relevant.any(dim=-1),
                (previousStatus | ~relevant).all(dim=-1),
                previousStatus.all(dim=-1)))
        requestVersionValid = (
            requestSnapshot
            & target.target_version.eq(request.target.target_version))
        cacheVersionValid = (
            previousSnapshot
            & (
                (
                    self.CachedTargetVersion.ge(0)
                    & target.target_version.eq(
                        self.CachedTargetVersion))
                | (
                    self.CachedTargetVersion.lt(0)
                    & target.target_version.eq(
                        request.target.target_version))))
        compositeSnapshot = ~(requestSnapshot | previousSnapshot)
        compositeVersionValid = (
            compositeSnapshot
            & target.target_version.gt(self.CachedTargetVersion))
        if bool((~(
            requestVersionValid
            | cacheVersionValid
            | compositeVersionValid
        )).any().item()):
            raise ValueError("applied target version does not identify its snapshot")
        sameVersion = target.target_version.eq(self.CachedTargetVersion)
        if bool(sameVersion.any().item()):
            rows = torch.nonzero(sameVersion, as_tuple=False).squeeze(-1)
            sameTarget = self.ContractView.TargetRowsMatch(
                canonicalValues.index_select(0, rows),
                target.active.index_select(0, rows),
                self.CachedTargetValues.index_select(0, rows),
                self.CachedTargetActive.index_select(0, rows),
            )
            if not bool(sameTarget.all().item()):
                raise ValueError("one target version cannot identify different targets")
            canonicalValues = torch.where(
                sameVersion.unsqueeze(-1),
                self.CachedTargetValues,
                canonicalValues)
        changed = ~cachedSlotMatches
        requestMatches = (
            rowMatchesRequest
            & target.target_version.eq(request.target.target_version))
        requestApplied = (
            request.target.active.any(dim=-1)
            & requestMatches
            & requestedSlotsApplied)
        ownerChanged = (
            changed.any(dim=-1)
            | requestApplied
            | result.help_accepted
            | (
                result.execution_known
                & result.execution_status.eq(
                    int(SlotExecutionStatus.MODIFIED))
                & relevant).any(dim=-1))
        ownerEpoch = torch.where(
            ownerChanged,
            result.action_epoch,
            self.CachedActionEpoch)
        reset = changed.clone()
        for layer in self.ContractView.topological_layers[1:]:
            for endpointIndex in layer:
                reset[:, endpointIndex] |= reset[:, self.ContractView.parent_index[endpointIndex]]
        dwellState = self.DwellState.masked_fill(reset, 0)
        reachedState = self.ReachedState.masked_fill(reset, False)
        progressState = self.ProgressState.masked_fill(reset, 0.0)
        self.CachedTargetValues = canonicalValues.detach().clone()
        self.CachedTargetActive = target.active.detach().clone()
        self.CachedTargetVersion = target.target_version.detach().clone()
        self.CachedActionEpoch = ownerEpoch.detach().clone()
        self.CachedRequestId = result.request_id.detach().clone()
        self.CachedExecutionStatus = result.execution_status.detach().clone()
        self.CachedExecutionKnown = result.execution_known.detach().clone()
        executionRelevant = relevant
        anyRelevant = executionRelevant.any(dim=-1)
        self.CachedExecutionRelevant = executionRelevant.detach().clone()
        self.CachedExecutionResultKnown = torch.where(
            anyRelevant,
            (result.execution_known | ~executionRelevant).all(dim=-1),
            result.execution_known.all(dim=-1)).detach().clone()
        self.CachedExecutionTimestamp = result.timestamp.detach().clone()
        self.CachedHardStop = result.hard_stop.detach().clone()
        self.CachedHelpAccepted = result.help_accepted.detach().clone()
        self.DwellState = dwellState.detach().clone()
        self.ReachedState = reachedState.detach().clone()
        self.ProgressState = progressState.detach().clone()

    def EvaluateHierarchy(
        self,
        translation: torch.Tensor,
        rotation: torch.Tensor,
        endpointPresent: torch.Tensor,
        timestamp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batchSize = int(translation.size(0))
        endpointCount = len(self.EndEffectors)
        if self.CachedTargetValues is None:
            self.InitializeExecutionState(translation.new_zeros(
                batchSize,
                self.ContractView.end_effector_target_layout.PackedDim))
        if self.ObservationAgeState is None:
            self.ObservationAgeState = translation.new_zeros(
                batchSize,
                endpointCount)
            self.ObservationKnownState = torch.zeros(
                batchSize,
                endpointCount,
                device=translation.device,
                dtype=torch.bool)
        elapsed = (
            torch.zeros_like(timestamp)
            if self.LastTimestamp is None
            else timestamp - self.LastTimestamp
        ).to(dtype=translation.dtype).unsqueeze(-1)
        self.ObservationAgeState = torch.where(
            endpointPresent,
            torch.zeros_like(self.ObservationAgeState),
            self.ObservationAgeState + elapsed).detach().clone()
        self.ObservationKnownState = (
            self.ObservationKnownState | endpointPresent
        ).detach().clone()
        progress = self.ProgressState.clone()
        reached = self.ReachedState.clone()
        phaseEnabled = torch.zeros(
            batchSize,
            endpointCount,
            device=translation.device,
            dtype=torch.bool)
        phaseKnown = torch.zeros_like(phaseEnabled)
        _, targetRotationVector = self.ExpandTarget(
            self.CachedTargetValues,
            self.CachedTargetActive)
        for layer in self.ContractView.topological_layers:
            for endpointIndex in layer:
                parentIndex = self.ContractView.parent_index[endpointIndex]
                if parentIndex < 0:
                    endpointPhaseKnown = torch.ones(
                        batchSize,
                        device=translation.device,
                        dtype=torch.bool)
                    endpointPhaseEnabled = endpointPhaseKnown
                else:
                    parentTargetActive = self.CachedTargetActive[:, parentIndex]
                    parentBlocked = (
                        ~phaseEnabled[:, parentIndex]
                        | ~parentTargetActive)
                    parentConfirmed = self.ObservationKnownState[:, parentIndex]
                    endpointPhaseKnown = (
                        phaseKnown[:, parentIndex]
                        & (parentBlocked | parentConfirmed))
                    endpointPhaseEnabled = (
                        phaseKnown[:, parentIndex]
                        & ~parentBlocked
                        & parentConfirmed
                        & self.ReachedState[:, parentIndex])
                phaseKnown[:, endpointIndex] = endpointPhaseKnown
                phaseEnabled[:, endpointIndex] = endpointPhaseEnabled
                targetActive = self.CachedTargetActive[:, endpointIndex]
                executing = endpointPhaseEnabled & targetActive
                observedExecution = executing & endpointPresent[:, endpointIndex]
                errorSquared = translation.new_zeros(batchSize)
                targetSlice = self.ContractView.end_effector_target_layout.Slice(
                    endpointIndex)
                translationWidth = self.ContractView.end_effector_translation_basis.shapes[
                    endpointIndex][1]
                rotationWidth = self.ContractView.end_effector_rotation_basis.shapes[
                    endpointIndex][1]
                if translationWidth:
                    basis = self.ContractView.end_effector_translation_basis.Matrix(
                        endpointIndex,
                        translation.device,
                        translation.dtype)
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
                    physicalError = error @ basis.transpose(0, 1)
                    errorSquared = errorSquared + physicalError.square().sum(dim=-1)
                if rotationWidth:
                    targetRotation = self.RotationVectorToMatrix(targetRotationVector[:, endpointIndex])
                    relativeRotation = targetRotation @ rotation[:, endpointIndex].transpose(-1, -2)
                    basis = self.ContractView.end_effector_rotation_basis.Matrix(
                        endpointIndex,
                        rotation.device,
                        rotation.dtype)
                    projection = self.RotationProjection.Matrix(
                        endpointIndex,
                        rotation.device,
                        rotation.dtype)
                    error = (
                        self.MatrixToRotationVector(relativeRotation)
                        @ projection
                    ) / self.RotationErrorScale[endpointIndex]
                    physicalError = error @ basis.transpose(0, 1)
                    errorSquared = errorSquared + physicalError.square().sum(dim=-1)
                errorNorm = torch.sqrt(errorSquared)
                measuredProgress = (
                    1.0 + errorNorm / self.ProgressExit[endpointIndex]
                ).reciprocal()
                wasReached = self.ReachedState[:, endpointIndex]
                inside = observedExecution & errorNorm.le(
                    self.ProgressEnter[endpointIndex])
                remain = observedExecution & wasReached & errorNorm.le(
                    self.ProgressExit[endpointIndex])
                measuredDwell = torch.where(
                    inside & ~wasReached,
                    self.DwellState[:, endpointIndex] + 1,
                    torch.zeros_like(self.DwellState[:, endpointIndex]))
                measuredReached = remain | measuredDwell.ge(
                    self.DwellCycles[endpointIndex])
                clearState = (
                    ~targetActive
                    | (endpointPhaseKnown & ~endpointPhaseEnabled))
                self.DwellState[:, endpointIndex] = torch.where(
                    observedExecution,
                    measuredDwell,
                    torch.where(
                        clearState,
                        torch.zeros_like(measuredDwell),
                        self.DwellState[:, endpointIndex]))
                self.ReachedState[:, endpointIndex] = torch.where(
                    observedExecution,
                    measuredReached,
                    torch.where(
                        clearState,
                        torch.zeros_like(wasReached),
                        wasReached))
                self.ProgressState[:, endpointIndex] = torch.where(
                    observedExecution,
                    measuredProgress,
                    torch.where(
                        clearState,
                        torch.zeros_like(measuredProgress),
                        self.ProgressState[:, endpointIndex]))
                progress[:, endpointIndex] = self.ProgressState[:, endpointIndex]
                reached[:, endpointIndex] = self.ReachedState[:, endpointIndex]
        self.DwellState = self.DwellState.detach().clone()
        self.ReachedState = self.ReachedState.detach().clone()
        self.ProgressState = self.ProgressState.detach().clone()
        return (
            progress,
            reached,
            phaseEnabled,
            phaseKnown,
            self.ObservationAgeState.clone())

    def ComputePerceptionMotion(
        self,
        rotation: torch.Tensor,
        endpointPresent: torch.Tensor,
        elapsed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batchSize = int(rotation.size(0))
        perceptionIndices = torch.tensor(self.ContractView.perception_view_indices, device=rotation.device, dtype=torch.long)
        current = rotation.index_select(1, perceptionIndices)
        currentStatePresent = endpointPresent.index_select(1, perceptionIndices)
        perceptionCount = int(current.size(1))
        currentQuaternion = self.MatrixToQuaternion(current)
        deltaQuaternion = rotation.new_zeros(batchSize, perceptionCount, 4)
        deltaQuaternion[..., 3] = 1.0
        angularVelocity = rotation.new_zeros(batchSize, perceptionCount, 3)
        motionPresent = torch.zeros(batchSize, perceptionCount, device=rotation.device, dtype=torch.bool)
        if perceptionCount == 0:
            return currentQuaternion, deltaQuaternion, angularVelocity, motionPresent
        if self.LastPerceptionRotation is None:
            self.LastPerceptionRotation = current.detach().clone()
            self.LastPerceptionStatePresent = torch.zeros_like(currentStatePresent)
        motionPresent = currentStatePresent & self.LastPerceptionStatePresent
        deltaMatrix = self.LastPerceptionRotation.transpose(-1, -2) @ current
        spatialDeltaMatrix = current @ self.LastPerceptionRotation.transpose(-1, -2)
        computedDelta = self.MatrixToQuaternion(deltaMatrix)
        computedAngularVelocity = self.MatrixToRotationVector(spatialDeltaMatrix) / elapsed.to(
            dtype=rotation.dtype).view(batchSize, 1, 1).clamp_min(
                torch.finfo(rotation.dtype).eps)
        angularVelocity = torch.where(
            motionPresent.unsqueeze(-1),
            computedAngularVelocity,
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
            timestamp,
        ) = self.DecodeFeedback(rawPayload, torch.device(device))
        self.ValidateFeedback(
            position,
            velocity,
            translation,
            rotationQuaternion,
            endpointPresent,
            timestamp,
            batchSize,
        )
        jointFeatures = self.EncodeJointFeatures(position, velocity)
        endpointFeatures, rotation = self.EncodeEndEffectorFeatures(
            translation,
            rotationQuaternion,
            endpointPresent)
        elapsed = (
            torch.zeros_like(timestamp)
            if self.LastTimestamp is None
            else timestamp - self.LastTimestamp)
        (
            perceptionRotation,
            perceptionRotationDelta,
            perceptionAngularVelocity,
            perceptionMotionPresent,
        ) = self.ComputePerceptionMotion(
            rotation,
            endpointPresent,
            elapsed)
        (
            progress,
            reached,
            phaseEnabled,
            phaseKnown,
            observationAge,
        ) = self.EvaluateHierarchy(
            translation,
            rotation,
            endpointPresent,
            timestamp)
        perceptionMotionFeatures = self.EncodePerceptionMotion(
            perceptionRotationDelta,
            perceptionAngularVelocity,
            perceptionMotionPresent)
        packet = BrainFeedbackPacket(
            joint_features=jointFeatures,
            end_effector_features=endpointFeatures,
            endpoint_present=endpointPresent,
            progress=progress,
            reached=reached,
            phase_enabled=phaseEnabled,
            phase_known=phaseKnown,
            observation_age=observationAge,
            applied_target_values=self.CachedTargetValues.clone(),
            applied_target_active=self.CachedTargetActive.clone(),
            applied_target_version=self.CachedTargetVersion.clone(),
            applied_action_epoch=self.CachedActionEpoch.clone(),
            execution_status=self.CachedExecutionStatus.clone(),
            execution_known=self.CachedExecutionKnown.clone(),
            execution_relevant=self.CachedExecutionRelevant.clone(),
            execution_result_known=self.CachedExecutionResultKnown.clone(),
            hard_stop=self.CachedHardStop.clone(),
            help_accepted=self.CachedHelpAccepted.clone(),
            perception_rotation=perceptionRotation,
            perception_rotation_delta=perceptionRotationDelta,
            perception_angular_velocity=perceptionAngularVelocity,
            perception_motion_features=perceptionMotionFeatures,
            perception_motion_present=perceptionMotionPresent,
            timestamp=timestamp)
        self.LastTimestamp = timestamp.detach().clone()
        return packet


class TestRobotMorphologyMTool:
    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or torch.device("cpu")

    def MakeTarget(
        self,
        robot: Robot,
        activeIndices: Sequence[int],
        version: int,
        timestamp: float,
    ) -> PackedEndEffectorTarget:
        contract = robot.ContractView
        values = torch.zeros(
            1,
            contract.end_effector_target_layout.PackedDim,
            device=self.device)
        active = torch.zeros(
            1,
            contract.end_effector_count,
            device=self.device,
            dtype=torch.bool)
        active[:, tuple(activeIndices)] = True
        return PackedEndEffectorTarget(
            values=values,
            active=active,
            target_version=torch.tensor(
                [version],
                device=self.device,
                dtype=torch.long),
            timestamp=torch.tensor(
                [timestamp],
                device=self.device))

    def MakeRequest(
        self,
        robot: Robot,
        target: PackedEndEffectorTarget,
        requestId: int,
        actionEpoch: int,
        stopRequested: bool = False,
    ) -> ActionRequest:
        device = target.values.device
        return robot.BuildActionRequest(
            target=target,
            requestId=torch.tensor([requestId], device=device),
            actionEpoch=torch.tensor([actionEpoch], device=device),
            commandActive=torch.tensor(
                [not stopRequested],
                device=device),
            holdRequested=torch.tensor([False], device=device),
            stopRequested=torch.tensor([stopRequested], device=device),
            helpRequested=torch.tensor([False], device=device),
            policyPath=torch.tensor([0], device=device),
            plannerOverride=torch.tensor([False], device=device),
            temporalKindId=torch.tensor([0], device=device),
            timestamp=target.timestamp.clone())

    def MakeResult(
        self,
        request: ActionRequest,
        target: PackedEndEffectorTarget,
        status: SlotExecutionStatus,
        timestamp: float,
        hardStop: bool = False,
    ) -> ActionExecutionResult:
        batchSize = int(target.values.size(0))
        endpointCount = int(target.active.size(1))
        device = target.values.device
        return ActionExecutionResult(
            request_id=request.request_id.clone(),
            action_epoch=request.action_epoch.clone(),
            applied_target=target,
            execution_status=torch.full(
                (batchSize, endpointCount),
                int(status),
                device=device,
                dtype=torch.long),
            execution_known=torch.ones(
                batchSize,
                endpointCount,
                device=device,
                dtype=torch.bool),
            hard_stop=torch.tensor([hardStop], device=device),
            help_accepted=torch.tensor([False], device=device),
            timestamp=torch.tensor([timestamp], device=device))

    def TestDefaultProfile(self) -> bool:
        robot = Robot.CreateDefault()
        leftJoints, leftEndpoints = Robot.CreateAnthropomorphicArm("left")
        rightJoints, rightEndpoints = Robot.CreateAnthropomorphicArm("right")
        assert len(leftJoints) == 27
        assert len(rightJoints) == 27
        assert len(leftEndpoints) + len(rightEndpoints) == 12
        assert len(robot.EndEffectors) == 13
        assert len(robot.ContractView.perception_view_indices) == 1
        perceptionIndex = robot.ContractView.perception_view_indices[0]
        assert robot.ContractView.end_effector_translation_basis.shapes[
            perceptionIndex][1] == 0
        assert robot.ContractView.end_effector_rotation_basis.shapes[
            perceptionIndex][1] > 0
        return True

    def TestAppliedTransactionAndHierarchy(self) -> bool:
        robot = Robot.CreateDefault()
        contract = robot.ContractView
        target = self.MakeTarget(robot, (0, 1), 0, 0.0)
        request = self.MakeRequest(robot, target, 11, 3)
        encodedRequest = robot.EncodeActionRequest(request)
        assert set(encodedRequest) == robot.ActionRequestPayloadFields
        result = robot.DecodeActionExecutionResult({
            "request_id": [11],
            "action_epoch": [3],
            "applied_target": robot.DecodeTarget(target),
            "execution_status": [[
                SlotExecutionStatus.APPLIED.name
                for _ in robot.EndEffectors]],
            "execution_known": [[True for _ in robot.EndEffectors]],
            "hard_stop": [False],
            "help_accepted": [False],
            "timestamp": [0.1],
        }, request, self.device)
        robot.CommitAppliedTarget(request, result)
        payload = robot.BuildNeutralFeedbackPayload(
            0.2,
            self.device)
        payload["end_effector_present"][0] = True
        packet = robot.EncodeFeedback(payload, self.device)
        assert int(robot.DwellState[0, 0].item()) == 1
        payload["end_effector_present"][0] = False
        payload["timestamp"] = 0.3
        packet = robot.EncodeFeedback(payload, self.device)
        assert int(robot.DwellState[0, 0].item()) == 1
        payload["end_effector_present"][0] = True
        for timestamp in (0.4, 0.5):
            payload["timestamp"] = timestamp
            packet = robot.EncodeFeedback(payload, self.device)
            packet.Validate(contract)
        assert bool(packet.reached[0, 0].item())
        assert bool(packet.phase_enabled[0, 1].item())
        payload["end_effector_present"][0] = False
        payload["timestamp"] = 0.6
        packet = robot.EncodeFeedback(payload, self.device)
        assert bool(packet.reached[0, 0].item())
        assert bool(packet.phase_enabled[0, 1].item())
        assert bool(packet.phase_known[0, 1].item())
        payload["timestamp"] = 0.9
        packet = robot.EncodeFeedback(payload, self.device)
        assert bool(packet.reached[0, 0].item())
        assert bool(packet.phase_enabled[0, 1].item())
        assert bool(packet.phase_known[0, 1].item())
        from TemporalExecutionModule import (
            CONTINUE,
            PACKED_TEMPORAL_KIND_NAMES,
            PackedTemporalEvent,
            PackedTemporalExecutionGate,
            PackedTemporalProposal,
        )
        temporalGate = PackedTemporalExecutionGate(contract)
        kindScores = torch.full(
            (1, len(PACKED_TEMPORAL_KIND_NAMES)),
            -1.0,
            device=self.device)
        kindScores[:, CONTINUE] = 1.0
        temporalDecision = temporalGate.Step(
            packet,
            target,
            PackedTemporalProposal(
                kind_scores=kindScores,
                same_operator=torch.ones(1, device=self.device),
                operator_changed=torch.zeros(1, device=self.device),
                invoke_delta=torch.zeros(1, device=self.device),
                reference_drift=torch.zeros(1, device=self.device),
                redispatch_score=torch.zeros(1, device=self.device),
                interrupt_score=torch.zeros(1, device=self.device),
                duration_ms=torch.ones(1, device=self.device),
                soft_timeout_ms=torch.full((1,), 10.0, device=self.device),
                hard_timeout_ms=torch.full((1,), 20.0, device=self.device)),
            PackedTemporalEvent(
                candidate_ready=torch.zeros(
                    1, device=self.device, dtype=torch.bool),
                redispatch_requested=torch.zeros(
                    1, device=self.device, dtype=torch.bool),
                cancel_requested=torch.zeros(
                    1, device=self.device, dtype=torch.bool),
                planner_failed=torch.zeros(
                    1, device=self.device, dtype=torch.bool),
                plan_reached=torch.zeros(
                    1, device=self.device, dtype=torch.bool),
                active_risk=torch.zeros(1, device=self.device),
                candidate_risk=torch.zeros(1, device=self.device)),
            torch.zeros(1, device=self.device))
        assert int(temporalDecision.kind_id.item()) == CONTINUE
        assert not bool(temporalDecision.stop_requested.item())
        stopTarget = self.MakeTarget(robot, (), 1, 1.0)
        stopRequest = self.MakeRequest(
            robot,
            stopTarget,
            12,
            4,
            True)
        stopResult = self.MakeResult(
            stopRequest,
            stopTarget,
            SlotExecutionStatus.STOPPED,
            1.1,
            True)
        robot.CommitAppliedTarget(stopRequest, stopResult)
        assert not bool(robot.CachedTargetActive.any().item())
        assert bool(robot.CachedHardStop.item())
        assert not bool(robot.ReachedState.any().item())
        assert bool(robot.CachedExecutionRelevant[0, 0].item())
        assert bool(robot.CachedExecutionRelevant[0, 1].item())
        assert bool(robot.CachedExecutionResultKnown.item())
        return True

    def TestAppliedTargetOwnerEpoch(self) -> bool:
        robot = Robot.CreateDefault()
        targetA = self.MakeTarget(robot, (0,), 0, 0.0)
        requestA = self.MakeRequest(robot, targetA, 21, 1)
        resultA = self.MakeResult(
            requestA,
            targetA,
            SlotExecutionStatus.APPLIED,
            0.1)
        robot.CommitAppliedTarget(requestA, resultA)
        targetBValues = targetA.values.clone()
        targetBValues[:, 0] = 0.25
        targetB = PackedEndEffectorTarget(
            values=targetBValues,
            active=targetA.active.clone(),
            target_version=torch.tensor(
                [1], device=self.device, dtype=torch.long),
            timestamp=torch.tensor([0.2], device=self.device))
        requestB = self.MakeRequest(robot, targetB, 22, 2)
        rejectedB = self.MakeResult(
            requestB,
            targetA,
            SlotExecutionStatus.REJECTED,
            0.3)
        robot.CommitAppliedTarget(requestB, rejectedB)
        assert int(robot.CachedActionEpoch.item()) == 1
        assert torch.equal(robot.CachedTargetValues, targetA.values)
        appliedB = self.MakeResult(
            requestB,
            targetB,
            SlotExecutionStatus.APPLIED,
            0.4)
        robot.CommitAppliedTarget(requestB, appliedB)
        assert int(robot.CachedActionEpoch.item()) == 2
        assert torch.equal(robot.CachedTargetValues, targetB.values)
        return True

    def TestContractTargetCoordinateComparison(self) -> bool:
        robot = Robot.CreateDefault()
        contract = robot.ContractView
        target = self.MakeTarget(robot, (0,), 0, 0.0)
        withinValues = target.values.clone()
        withinValues[:, 0] += 5.0e-7
        assert bool(contract.TargetRowsMatch(
            target.values,
            target.active,
            withinValues,
            target.active).item())
        outsideValues = target.values.clone()
        outsideValues[:, 0] += 1.0e-3
        assert not bool(contract.TargetRowsMatch(
            target.values,
            target.active,
            outsideValues,
            target.active).item())
        inactive = target.active.clone()
        inactive[:, 0] = False
        assert not bool(contract.TargetRowsMatch(
            target.values,
            target.active,
            target.values,
            inactive).item())
        return True

    def TestAppliedTargetUsesContractTolerance(self) -> bool:
        robot = Robot.CreateDefault()
        requested = self.MakeTarget(robot, (0,), 0, 0.0)
        request = self.MakeRequest(robot, requested, 31, 1)
        appliedValues = requested.values.clone()
        appliedValues[:, 0] += 5.0e-7
        applied = PackedEndEffectorTarget(
            values=appliedValues,
            active=requested.active.clone(),
            target_version=requested.target_version.clone(),
            timestamp=torch.tensor([0.1], device=self.device))
        result = self.MakeResult(
            request,
            applied,
            SlotExecutionStatus.APPLIED,
            0.1)
        robot.CommitAppliedTarget(request, result)
        assert int(robot.CachedActionEpoch.item()) == 1
        return True

    def TestTargetBoundsUseContractTolerance(self) -> bool:
        robot = Robot.CreateDefault()
        contract = robot.ContractView
        target = self.MakeTarget(robot, (0,), 0, 0.0)
        upper = contract.end_effector_target_upper[0]
        tolerance = contract.end_effector_target_tolerance[0]
        withinValues = target.values.clone()
        withinValues[:, 0] = upper + 0.5 * tolerance
        within = PackedEndEffectorTarget(
            values=withinValues,
            active=target.active.clone(),
            target_version=target.target_version.clone(),
            timestamp=target.timestamp.clone())
        within.Validate(contract)
        outsideValues = target.values.clone()
        outsideValues[:, 0] = upper + 2.0 * tolerance
        outside = PackedEndEffectorTarget(
            values=outsideValues,
            active=target.active.clone(),
            target_version=target.target_version.clone(),
            timestamp=target.timestamp.clone())
        rejected = False
        try:
            outside.Validate(contract)
        except ValueError:
            rejected = True
        assert rejected
        return True

    def TestAppliedTargetVersionIsStrict(self) -> bool:
        robot = Robot.CreateDefault()
        target = self.MakeTarget(robot, (0,), 0, 0.0)
        firstRequest = self.MakeRequest(robot, target, 41, 1)
        robot.CommitAppliedTarget(
            firstRequest,
            self.MakeResult(
                firstRequest,
                target,
                SlotExecutionStatus.APPLIED,
                0.1))
        nextTarget = PackedEndEffectorTarget(
            values=target.values.clone(),
            active=target.active.clone(),
            target_version=torch.tensor(
                [1], device=self.device, dtype=torch.long),
            timestamp=torch.tensor([0.2], device=self.device))
        nextRequest = self.MakeRequest(robot, nextTarget, 42, 2)
        rejected = False
        try:
            robot.CommitAppliedTarget(
                nextRequest,
                self.MakeResult(
                    nextRequest,
                    target,
                    SlotExecutionStatus.APPLIED,
                    0.3))
        except ValueError:
            rejected = True
        assert rejected
        return True

    def TestSameVersionTargetSnapsToCache(self) -> bool:
        robot = Robot.CreateDefault()
        target = self.MakeTarget(robot, (0,), 0, 0.0)
        firstRequest = self.MakeRequest(robot, target, 51, 1)
        robot.CommitAppliedTarget(
            firstRequest,
            self.MakeResult(
                firstRequest,
                target,
                SlotExecutionStatus.APPLIED,
                0.1))
        heldValues = target.values.clone()
        heldValues[:, 0] += 5.0e-7
        heldTarget = PackedEndEffectorTarget(
            values=heldValues,
            active=target.active.clone(),
            target_version=target.target_version.clone(),
            timestamp=torch.tensor([0.2], device=self.device))
        heldRequest = self.MakeRequest(robot, target, 52, 1)
        robot.CommitAppliedTarget(
            heldRequest,
            self.MakeResult(
                heldRequest,
                heldTarget,
                SlotExecutionStatus.HELD,
                0.2))
        assert torch.equal(robot.CachedTargetValues, target.values)
        return True

    def TestCompositeAppliedTargetSnapshot(self) -> bool:
        robot = Robot.CreateDefault()
        contract = robot.ContractView
        targetA = self.MakeTarget(robot, (0, 1), 0, 0.0)
        requestA = self.MakeRequest(robot, targetA, 61, 1)
        robot.CommitAppliedTarget(
            requestA,
            self.MakeResult(
                requestA,
                targetA,
                SlotExecutionStatus.APPLIED,
                0.1))
        valuesB = targetA.values.clone()
        valuesB[:, contract.end_effector_target_layout.Slice(0).start] = 0.25
        valuesB[:, contract.end_effector_target_layout.Slice(1).start] = 0.15
        targetB = PackedEndEffectorTarget(
            values=valuesB,
            active=targetA.active.clone(),
            target_version=torch.tensor(
                [1], device=self.device, dtype=torch.long),
            timestamp=torch.tensor([0.2], device=self.device))
        requestB = self.MakeRequest(robot, targetB, 62, 2)
        compositeValues = targetA.values.clone()
        compositeValues[:, contract.end_effector_target_layout.Slice(0)] = (
            targetB.values[:, contract.end_effector_target_layout.Slice(0)])
        composite = PackedEndEffectorTarget(
            values=compositeValues,
            active=targetA.active.clone(),
            target_version=torch.tensor(
                [1], device=self.device, dtype=torch.long),
            timestamp=torch.tensor([0.3], device=self.device))
        result = self.MakeResult(
            requestB,
            composite,
            SlotExecutionStatus.HELD,
            0.3)
        status = result.execution_status.clone()
        status[:, 0] = int(SlotExecutionStatus.APPLIED)
        result = ActionExecutionResult(
            request_id=result.request_id,
            action_epoch=result.action_epoch,
            applied_target=result.applied_target,
            execution_status=status,
            execution_known=result.execution_known,
            hard_stop=result.hard_stop,
            help_accepted=result.help_accepted,
            timestamp=result.timestamp)
        robot.CommitAppliedTarget(requestB, result)
        assert int(robot.CachedTargetVersion.item()) == 1
        assert int(robot.CachedActionEpoch.item()) == 2
        assert torch.equal(robot.CachedTargetValues, compositeValues)
        return True

    def TestPerceptionTranslationDoesNotInvalidateMotion(self) -> bool:
        robot = Robot.CreateDefault()
        contract = robot.ContractView
        endpointIndex = contract.perception_view_indices[0]
        viewIndex = contract.perception_view_indices.index(endpointIndex)
        payload = robot.BuildNeutralFeedbackPayload(0.1, self.device)
        payload["end_effector_present"][endpointIndex] = True
        robot.EncodeFeedback(payload, self.device)
        payload["end_effector_translation"][endpointIndex] = [1.0, -2.0, 3.0]
        payload["timestamp"] = 0.2
        translated = robot.EncodeFeedback(payload, self.device)
        assert bool(translated.perception_motion_present[0, viewIndex].item())
        assert torch.allclose(
            translated.perception_angular_velocity[0, viewIndex],
            torch.zeros(3, device=self.device))
        angle = 0.1
        payload["end_effector_translation"][endpointIndex] = [-3.0, 2.0, 1.0]
        payload["end_effector_rotation_xyzw"][endpointIndex] = [
            0.0,
            0.0,
            math.sin(0.5 * angle),
            math.cos(0.5 * angle),
        ]
        payload["timestamp"] = 0.3
        rotated = robot.EncodeFeedback(payload, self.device)
        assert torch.allclose(
            rotated.perception_angular_velocity[0, viewIndex],
            torch.tensor([0.0, 0.0, 1.0], device=self.device),
            atol=1.0e-5,
            rtol=1.0e-5)
        previousMatrix = Robot.RotationVectorToMatrix(torch.tensor(
            [[0.0, 0.0, angle]], device=self.device))
        incrementMatrix = Robot.RotationVectorToMatrix(torch.tensor(
            [[angle, 0.0, 0.0]], device=self.device))
        nextMatrix = previousMatrix @ incrementMatrix
        payload["end_effector_rotation_xyzw"][endpointIndex] = (
            Robot.MatrixToQuaternion(nextMatrix)[0].tolist())
        payload["timestamp"] = 0.4
        composed = robot.EncodeFeedback(payload, self.device)
        expectedAxis = previousMatrix[0] @ torch.tensor(
            [1.0, 0.0, 0.0], device=self.device)
        assert torch.allclose(
            composed.perception_angular_velocity[0, viewIndex],
            expectedAxis,
            atol=1.0e-5,
            rtol=1.0e-5)
        return True

    def TestBasisMetricIsCoordinateInvariant(self) -> bool:
        def BuildRobot(
            profileId: str,
            basis: Tuple[Tuple[float, ...], ...],
        ) -> Robot:
            return Robot(RobotDefinition(
                joints=(Robot.CreateRevoluteJoint(
                    f"{profileId}_joint",
                    (1.0, 0.0, 0.0),
                    -1.0,
                    1.0,
                    1.0),),
                end_effectors=(EndEffector(
                    effector_id=f"{profileId}_endpoint",
                    effector_type=EndEffectorType.OTHER,
                    parent_effector_id=None,
                    translation_basis=basis,
                    rotation_basis=Robot.EmptyBasis,
                    target_lower=(-2.0, -2.0),
                    target_upper=(2.0, 2.0),
                    reference_frame_id="external",
                    is_perception_slot=False,
                    progress_enter=0.1,
                    progress_exit=2.0,
                    dwell_cycles=1,
                    joint_ids=(f"{profileId}_joint",)),),
                perception_calibrations=()))

        def Evaluate(
            robot: Robot,
            coordinates: Tuple[float, float],
        ) -> torch.Tensor:
            contract = robot.ContractView
            target = PackedEndEffectorTarget(
                values=torch.tensor(
                    [coordinates],
                    device=self.device),
                active=torch.ones(
                    1,
                    contract.end_effector_count,
                    device=self.device,
                    dtype=torch.bool),
                target_version=torch.zeros(
                    1,
                    device=self.device,
                    dtype=torch.long),
                timestamp=torch.zeros(1, device=self.device))
            request = self.MakeRequest(robot, target, 1, 1)
            robot.CommitAppliedTarget(
                request,
                self.MakeResult(
                    request,
                    target,
                    SlotExecutionStatus.APPLIED,
                    0.1))
            payload = robot.BuildNeutralFeedbackPayload(0.2, self.device)
            payload["end_effector_present"][0] = True
            return robot.EncodeFeedback(
                payload,
                self.device).progress[:, 0]

        orthogonal = BuildRobot(
            "orthogonal_basis",
            ((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
        transformed = BuildRobot(
            "transformed_basis",
            ((2.0, 1.0), (0.0, 1.0), (0.0, 0.0)))
        orthogonalProgress = Evaluate(orthogonal, (1.0, 1.0))
        transformedProgress = Evaluate(transformed, (0.0, 1.0))
        assert torch.allclose(
            orthogonalProgress,
            transformedProgress,
            atol=1.0e-6,
            rtol=1.0e-6)
        return True

    def RunAll(self) -> Mapping[str, bool]:
        return {
            "DefaultProfile": self.TestDefaultProfile(),
            "AppliedTransactionAndHierarchy": self.TestAppliedTransactionAndHierarchy(),
            "AppliedTargetOwnerEpoch": self.TestAppliedTargetOwnerEpoch(),
            "ContractTargetCoordinateComparison": self.TestContractTargetCoordinateComparison(),
            "AppliedTargetUsesContractTolerance": self.TestAppliedTargetUsesContractTolerance(),
            "TargetBoundsUseContractTolerance": self.TestTargetBoundsUseContractTolerance(),
            "AppliedTargetVersionIsStrict": self.TestAppliedTargetVersionIsStrict(),
            "SameVersionTargetSnapsToCache": self.TestSameVersionTargetSnapsToCache(),
            "CompositeAppliedTargetSnapshot": self.TestCompositeAppliedTargetSnapshot(),
            "PerceptionTranslationDoesNotInvalidateMotion": self.TestPerceptionTranslationDoesNotInvalidateMotion(),
            "BasisMetricIsCoordinateInvariant": self.TestBasisMetricIsCoordinateInvariant(),
        }
