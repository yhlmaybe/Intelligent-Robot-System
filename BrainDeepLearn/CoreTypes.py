from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import torch

from ModuleMessagerManager import ModuleDim


TEXT_TRUST_OCR_OBSERVED = "ocr_observed"
TEXT_TRUST_OPERATOR_COMMAND = "operator_command"
TEXT_TRUST_UNSAFE_EXTERNAL = "unsafe_external"
ROBOT_STATE_WIRE_SCHEMA_VERSION = 7
SENSOR_PACKET_WIRE_SCHEMA_VERSION = 4
DECISION_WIRE_SCHEMA_VERSION = 7
OFFLINE_SENSOR_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CameraCalibration:
    calibration_id: str
    intrinsics: torch.Tensor


ROBOT_STATE_WIRE_METADATA_FIELDS = (
    "schema_version",
    "stream_id",
    "sequence_index",
    "frame_id",
    "calibration_id",
    "world_frame_id",
    "endpoint_names",
    "controlled_endpoint_names",
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
)

OFFLINE_SENSOR_MANIFEST_FIELDS = (
    "schema_version",
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
)


class RobotState(TypedDict):
    """Measured robot/planner state for the current frame.

    Poses are batched, expressed in the world frame in metres, and use XYZW
    quaternions. ``endpoint_pose`` follows ``ModuleDim.RobotStateEndpointNames``
    and is ``[B, 13, 7]``. The first 12 rows are the ten fingertips and two
    wrists. The ``camera_optical`` row uses a fixed optical-center translation
    plus an XYZW unit quaternion; the quaternion represents exactly three
    rotational DOFs, while its xyz carrier is not a camera motion DOF. There is
    deliberately no second camera-pose field. ``base_orientation_world`` is an
    XYZW unit quaternion shaped ``[B, 4]``. It is used only to remove the
    arbitrary world-frame gauge from the camera orientation; base translation
    is fixed hardware geometry and never enters a learned layer.
    ``gravity_direction_world`` is the
    unit, dimensionless direction of gravitational acceleration in the world
    frame (down), shaped ``[B, 3]``; it is not accelerometer specific force and
    does not include the 9.81 m/s^2 magnitude. Both physical references use the
    current sensor-frame exposure time. ``planner_expected_endpoint_pose``
    follows all 13 ``ModuleDim.DecisionEndpointNames`` and is ``[B, 13, 7]``.
    The strict action boundary is 12 full SE(3) endpoints plus three camera
    rotations, for 75 active DOFs. Camera target translation remains the fixed
    optical-center translation. Planner/provenance scalars are ``[B]``.

    ``model_command_executed`` states whether the measured transition into this
    frame was produced by the model command emitted on the preceding frame.
    When it is true, ``executed_action_id`` must identify that command; zero is
    the no-model-command sentinel. These are feedback provenance, not action
    labels. An executor-side rejection reports ``model_command_executed = 0``,
    ``executed_action_id = 0`` and ``planner_failed = 1``.
    """

    endpoint_pose: torch.Tensor
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
ROBOT_STATE_WIRE_FIELDS = ROBOT_STATE_WIRE_METADATA_FIELDS + ROBOT_STATE_FIELDS


def ExpectedRobotStateWireMetadata() -> Dict[str, Any]:
    return {
        "endpoint_names": list(ModuleDim.RobotStateEndpointNames),
        "controlled_endpoint_names": list(ModuleDim.DecisionEndpointNames),
        "pose_frame": "world",
        "pose_convention": "T_world_endpoint",
        "pose_time_reference": "sensor_frame_exposure",
        "pose_unit": "meter",
        "quaternion_order": "xyzw",
        "pose_handedness": "right_handed",
        "base_orientation_convention": "q_world_base_xyzw",
        "gravity_convention": "unit_acceleration_direction_world",}


def ValidateRobotStateWirePacket(
    packet: Any,
    calibrationId: str,) -> None:
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
            "RobotState calibration_id does not match the configured camera")
    if type(packet["world_frame_id"]) is not str or not packet["world_frame_id"]:
        raise ValueError("RobotState world_frame_id must be a non-empty string")
    for name, expected in ExpectedRobotStateWireMetadata().items():
        if packet[name] != expected:
            raise ValueError(
                f"robot packet {name} does not match the current contract")


def ValidateOfflineSensorManifest(
    manifest: Any,
    calibrationId: str,) -> None:
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
    expected = {
        "calibration_id": calibrationId,
        "rgb_encoding": "rgb8",
        "depth_unit": "meter",
        "depth_representation": "optical_axis_z",
        "rgb_depth_alignment": "registered_to_rgb",
        "rectification": "rectified",
        "synchronization": "synchronized_exposure",
        "object_motion_frame": "current_camera_optical",
        "object_motion_representation": "se3_spatial_delta",
        "object_motion_reference": (
            "previous_to_current_after_camera_egomotion_compensation"),
        "object_motion_translation_unit": "meter",
        "object_motion_quaternion_order": "xyzw",}
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
    network_decision_tensor: torch.Tensor
    prospective_action_embed: torch.Tensor
    temporal_goal: Dict[str, torch.Tensor]
    world_abstract: Dict[str, torch.Tensor]
    reference_uncertainty: torch.Tensor
