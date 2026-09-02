from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import torch

from ModuleMessagerManager import CognitiveDimProfile
from RobotMorphologyModule import (
    ActionRequest,
    BrainFeedbackPacket,
    EmbodimentShape,
    PackedEndEffectorTarget,
    RobotEmbodimentContractView,
)


TEXT_TRUST_OCR_OBSERVED = "ocr_observed"
TEXT_TRUST_OPERATOR_COMMAND = "operator_command"
TEXT_TRUST_UNSAFE_EXTERNAL = "unsafe_external"
SENSOR_PACKET_WIRE_SCHEMA_VERSION = 6
DECISION_WIRE_SCHEMA_VERSION = 12
BRAIN_BUILD_SPEC_SCHEMA_VERSION = 13
POLICY_OPTION_COUNT = 81
ASSIST_OPTION_ID = 80
COGNITIVE_READOUT_SCHEMA_VERSION = 2
POLICY_PATH_FULL = 0
POLICY_PATH_FAST = 1
POLICY_PATH_DETAIL = 2
POLICY_PATH_NONE = 3


@dataclass(frozen=True)
class BrainBuildSpec:
    cognitive: CognitiveDimProfile
    embodiment: EmbodimentShape
    contract_view: RobotEmbodimentContractView
    policy_option_count: int
    assist_option_id: int
    model_signature: str

    @staticmethod
    def CompileModelSignature(
        cognitive: CognitiveDimProfile,
        contractView: RobotEmbodimentContractView,
        policyOptionCount: int = POLICY_OPTION_COUNT,
        assistOptionId: int = ASSIST_OPTION_ID,
    ) -> str:
        if type(cognitive) is not CognitiveDimProfile:
            raise TypeError("cognitive must be a CognitiveDimProfile")
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError(
                "contractView must be a RobotEmbodimentContractView")

        embodiment = contractView.model_shape
        if type(embodiment) is not EmbodimentShape:
            raise TypeError("contractView.model_shape must be an EmbodimentShape")

        cognitive_payload = {
            name: getattr(cognitive, name)
            for name in cognitive.__dataclass_fields__}
        cognitive_payload["policy_option_count"] = int(policyOptionCount)
        cognitive_payload["assist_option_id"] = int(assistOptionId)
        return contractView.CompileBrainBuildSignature(
            cognitive_payload,
            BRAIN_BUILD_SPEC_SCHEMA_VERSION)

    @classmethod
    def Compile(
        cls,
        cognitive: CognitiveDimProfile,
        contractView: RobotEmbodimentContractView,
        policyOptionCount: int = POLICY_OPTION_COUNT,
        assistOptionId: int = ASSIST_OPTION_ID,
    ) -> "BrainBuildSpec":
        option_count = int(policyOptionCount)
        assist_option_id = int(assistOptionId)
        if option_count < 2 or not 0 <= assist_option_id < option_count:
            raise ValueError("assist option must belong to the policy option set")
        model_signature = cls.CompileModelSignature(
            cognitive,
            contractView,
            option_count,
            assist_option_id)
        return cls(
            cognitive=cognitive,
            embodiment=contractView.model_shape,
            contract_view=contractView,
            policy_option_count=option_count,
            assist_option_id=assist_option_id,
            model_signature=model_signature,
        )

    def IsCheckpointCompatible(self, checkpointModelSignature: Any) -> bool:
        return (
            type(checkpointModelSignature) is str
            and checkpointModelSignature == self.model_signature
        )

    def ValidateCheckpointCompatibility(
        self,
        checkpointModelSignature: Any,
    ) -> None:
        if not self.IsCheckpointCompatible(checkpointModelSignature):
            raise ValueError(
                "full checkpoint model_signature does not match BrainBuildSpec")

    def CognitiveProfilePayload(self) -> Dict[str, int]:
        payload = {
            name: getattr(self.cognitive, name)
            for name in self.cognitive.__dataclass_fields__}
        payload["policy_option_count"] = self.policy_option_count
        payload["assist_option_id"] = self.assist_option_id
        return payload

    def ValidateCognitiveProfileCompatibility(
        self,
        cognitivePayload: Any,
    ) -> None:
        expected = self.CognitiveProfilePayload()
        if type(cognitivePayload) is not dict or set(cognitivePayload) != set(expected):
            raise ValueError(
                "cognitive backbone profile fields do not match")
        cognitive_fields = self.cognitive.__dataclass_fields__
        if any(
            type(cognitivePayload[name]) is not int
            or cognitivePayload[name] <= 0
            for name in cognitive_fields
        ):
            raise ValueError(
                "cognitive backbone profile dimensions must be positive integers")
        option_count = cognitivePayload["policy_option_count"]
        assist_option_id = cognitivePayload["assist_option_id"]
        if (
            type(option_count) is not int
            or option_count < 2
            or type(assist_option_id) is not int
            or not 0 <= assist_option_id < option_count
        ):
            raise ValueError(
                "cognitive backbone policy option profile is invalid")
        if cognitivePayload != expected:
            raise ValueError(
                "cognitive backbone requires an identical CognitiveDimProfile")

    def ValidateFeedbackPacket(
        self,
        feedbackPacket: BrainFeedbackPacket,
        *,
        batchSize: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if type(feedbackPacket) is not BrainFeedbackPacket:
            raise TypeError("brain input requires a BrainFeedbackPacket")
        feedbackPacket.Validate(self.contract_view)
        packet_batch_size = int(feedbackPacket.joint_features.size(0))
        if batchSize is not None and packet_batch_size != int(batchSize):
            raise ValueError(
                "feedback packet batch size does not match sensory input")
        if (
            device is not None
            and feedbackPacket.joint_features.device != torch.device(device)
        ):
            raise ValueError(
                "feedback packet device does not match the brain input device")


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


@dataclass(frozen=True)
class ContractBrainStepInput:
    frame: torch.Tensor
    text_ext: Optional[List[Optional[str]]]
    reward_ext: Optional[torch.Tensor]
    done_flag: Optional[torch.Tensor]
    is_train: bool
    sample_actions: bool
    deterministic_actor: bool
    depth: torch.Tensor
    depth_valid: torch.Tensor
    feedback_packet: BrainFeedbackPacket
    text_trust: Optional[List[str]] = None
    perception_targets: Dict[str, torch.Tensor] = field(
        default_factory=dict)


@dataclass(frozen=True)
class ContractAgentActInput:
    frame: torch.Tensor
    text_ext: Optional[List[Optional[str]]]
    reward: Optional[torch.Tensor]
    done: Optional[torch.Tensor]
    sample_actions: bool
    deterministic_actor: bool
    depth: torch.Tensor
    depth_valid: torch.Tensor
    feedback_packet: BrainFeedbackPacket
    text_trust: Optional[List[str]] = None
    perception_targets: Dict[str, torch.Tensor] = field(
        default_factory=dict)


@dataclass(frozen=True)
class CognitiveReadout:
    schema_version: int
    model_signature: str
    contract_id: str
    request_id: torch.Tensor
    timestamp: torch.Tensor
    row_valid: torch.Tensor
    intention_feature: torch.Tensor
    intention_valid: torch.Tensor
    intention_age: torch.Tensor
    world_belief_feature: torch.Tensor
    world_belief_valid: torch.Tensor
    world_belief_age: torch.Tensor
    sensorimotor_evidence: torch.Tensor
    sensorimotor_valid: torch.Tensor
    sensorimotor_age: torch.Tensor
    decision_feature: torch.Tensor
    decision_valid: torch.Tensor
    decision_age: torch.Tensor
    compute_mode: torch.Tensor
    policy_path: torch.Tensor
    planner_override: torch.Tensor
    option_id: torch.Tensor
    option_valid: torch.Tensor
    temporal_kind_id: torch.Tensor

    def Validate(self, buildSpec: BrainBuildSpec) -> None:
        if (
            self.schema_version != COGNITIVE_READOUT_SCHEMA_VERSION
            or self.model_signature != buildSpec.model_signature
            or self.contract_id != buildSpec.contract_view.contract_id
        ):
            raise ValueError("cognitive readout identity does not match")
        if not torch.is_tensor(self.timestamp) or self.timestamp.dim() != 1:
            raise ValueError("cognitive readout timestamp must be a vector")
        batchSize = int(self.timestamp.size(0))
        device = self.timestamp.device
        floatingVectors = (
            self.timestamp,
            self.intention_age,
            self.world_belief_age,
            self.sensorimotor_age,
            self.decision_age,
        )
        featureMatrices = (
            self.intention_feature,
            self.world_belief_feature,
            self.sensorimotor_evidence,
            self.decision_feature,
        )
        booleanVectors = (
            self.row_valid,
            self.intention_valid,
            self.world_belief_valid,
            self.sensorimotor_valid,
            self.decision_valid,
            self.planner_override,
            self.option_valid,
        )
        integerVectors = (
            self.request_id,
            self.compute_mode,
            self.policy_path,
            self.option_id,
            self.temporal_kind_id,
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.device != device
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            or bool(value.lt(0.0).any().item())
            for value in floatingVectors
        ):
            raise ValueError("cognitive readout ages are invalid")
        if any(
            not torch.is_tensor(value)
            or value.dim() != 2
            or int(value.size(0)) != batchSize
            or int(value.size(1)) < 1
            or value.device != device
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
            for value in featureMatrices
        ):
            raise ValueError("cognitive readout features are invalid")
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.device != device
            or value.dtype != torch.bool
            for value in booleanVectors
        ):
            raise ValueError("cognitive readout validity is invalid")
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != (batchSize,)
            or value.device != device
            or value.dtype != torch.long
            or bool(value.lt(0).any().item())
            for value in integerVectors
        ):
            raise ValueError("cognitive readout identifiers are invalid")
        if bool(self.option_id.ge(buildSpec.policy_option_count).any().item()):
            raise ValueError("cognitive readout option is outside the policy")
        if bool(self.policy_path.gt(POLICY_PATH_NONE).any().item()):
            raise ValueError("cognitive readout policy path is invalid")


class CognitiveReadoutDecoderProtocol(Protocol):
    def Decode(self, readout: CognitiveReadout) -> Any:
        ...


@dataclass(frozen=True)
class ContractAgentActOutput:
    action_request: ActionRequest
    cognitive_readout: CognitiveReadout
    packed_target: PackedEndEffectorTarget
    packed_temporal: Any
    decision: Dict[str, Any]
    ocr: Any
    intention_texts: List[str]


@dataclass(frozen=True)
class ContractOfflineSample:
    image: Any
    reward: Any
    done: Any
    depth: Any
    depth_valid: Any
    text_ext: Optional[str]
    feedback_payload: Any
    perception_targets: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContractOfflineBatch:
    frames: torch.Tensor
    rewards: Optional[torch.Tensor]
    dones: Optional[torch.Tensor]
    depths: torch.Tensor
    depth_valid: torch.Tensor
    text_ext: List[Optional[str]]
    feedback_packet: BrainFeedbackPacket
    perception_targets: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class BrainStepOutput:
    action_request: ActionRequest
    cognitive_readout: CognitiveReadout
    decision: Dict[str, Any]
    world: Dict[str, torch.Tensor]
    critic: Any
    features: Dict[str, Any]
    ocr: Any
    intention_texts: List[str]
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    stages: Dict[str, Any] = field(default_factory=dict)
