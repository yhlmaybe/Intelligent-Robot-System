from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

import torch


TEXT_TRUST_OCR_OBSERVED = "ocr_observed"
TEXT_TRUST_OPERATOR_COMMAND = "operator_command"
TEXT_TRUST_UNSAFE_EXTERNAL = "unsafe_external"
ROBOT_STATE_WIRE_SCHEMA_VERSION = 1
SENSOR_PACKET_WIRE_SCHEMA_VERSION = 2
DECISION_WIRE_SCHEMA_VERSION = 2


class RobotState(TypedDict):
    """Measured robot/planner state for the current frame.

    Poses are batched, expressed in the world frame in metres, and use XYZW
    quaternions. Endpoint tensors follow ``ModuleDim.DecisionEndpointNames``:
    ``endpoint_pose`` and ``planner_expected_endpoint_pose`` are ``[B, 13, 7]``;
    ``camera_pose_world`` is ``[B, 7]``; planner/provenance scalars are ``[B]``.

    ``model_command_executed`` states whether the measured transition into this
    frame was produced by the model command emitted on the preceding frame. It
    is feedback provenance, not an action label. An executor-side rejection
    reports ``model_command_executed = 0`` and ``planner_failed = 1``.
    """

    endpoint_pose: torch.Tensor
    camera_pose_world: torch.Tensor
    planner_expected_endpoint_pose: torch.Tensor
    planner_progress: torch.Tensor
    planner_tracking_error: torch.Tensor
    planner_executing: torch.Tensor
    planner_reached: torch.Tensor
    planner_failed: torch.Tensor
    model_command_executed: torch.Tensor


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
    target_endpoint_pose: torch.Tensor
    temporal_envelope: Any
    decision: Dict[str, Any]
    loss: Optional[torch.Tensor]
    ocr: Any
    intention_texts: List[str]
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
    camera_pose_world: torch.Tensor
    camera_motion_from_prev: Any
    prev_visual_for_loss: Any
    ocr_items: Any
    fuse_ocr: List[List[str]]
    ocr_semantic: torch.Tensor
    slow_refresh: bool
    text_control_refresh: bool
    pst: Dict[str, torch.Tensor]
    observed_pst: Dict[str, torch.Tensor]
    pst_summary: torch.Tensor
    endpoint_pose: torch.Tensor
    endpoint_pose_encoding: Any
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
    next_visual_prediction: Any


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
    executed_feedback_embed: torch.Tensor
    temporal_goal: Dict[str, torch.Tensor]
    world_abstract: Dict[str, torch.Tensor]
    intent_novelty: torch.Tensor
