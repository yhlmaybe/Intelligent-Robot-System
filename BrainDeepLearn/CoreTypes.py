from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ModuleMessagerManager import CognitiveDimProfile
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    EmbodimentShape,
    PackedEndEffectorTarget,
    Robot,
    RobotEmbodimentContractView,
)


TEXT_TRUST_OCR_OBSERVED = "ocr_observed"
TEXT_TRUST_OPERATOR_COMMAND = "operator_command"
TEXT_TRUST_UNSAFE_EXTERNAL = "unsafe_external"
SENSOR_PACKET_WIRE_SCHEMA_VERSION = 5
DECISION_WIRE_SCHEMA_VERSION = 11
BRAIN_BUILD_SPEC_SCHEMA_VERSION = 12


@dataclass(frozen=True)
class BrainBuildSpec:
    cognitive: CognitiveDimProfile
    embodiment: EmbodimentShape
    contract_view: RobotEmbodimentContractView
    model_signature: str

    @staticmethod
    def CompileModelSignature(
        cognitive: CognitiveDimProfile,
        contractView: RobotEmbodimentContractView,
    ) -> str:
        if type(cognitive) is not CognitiveDimProfile:
            raise TypeError("cognitive must be a CognitiveDimProfile")
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError(
                "contractView must be a RobotEmbodimentContractView")

        embodiment = contractView.model_shape
        if type(embodiment) is not EmbodimentShape:
            raise TypeError("contractView.model_shape must be an EmbodimentShape")

        return Robot.CompileBrainBuildSignature(
            {
                name: getattr(cognitive, name)
                for name in cognitive.__dataclass_fields__
            },
            contractView,
            BRAIN_BUILD_SPEC_SCHEMA_VERSION)

    @classmethod
    def Compile(
        cls,
        cognitive: CognitiveDimProfile,
        contractView: RobotEmbodimentContractView,
    ) -> "BrainBuildSpec":
        model_signature = cls.CompileModelSignature(cognitive, contractView)
        return cls(
            cognitive=cognitive,
            embodiment=contractView.model_shape,
            contract_view=contractView,
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
        return {
            name: getattr(self.cognitive, name)
            for name in self.cognitive.__dataclass_fields__}

    def ValidateCognitiveProfileCompatibility(
        self,
        cognitivePayload: Any,
    ) -> None:
        expected = self.CognitiveProfilePayload()
        if type(cognitivePayload) is not dict or set(cognitivePayload) != set(expected):
            raise ValueError(
                "cognitive backbone profile fields do not match")
        if any(type(value) is not int or value <= 0 for value in cognitivePayload.values()):
            raise ValueError(
                "cognitive backbone profile dimensions must be positive integers")
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
class ContractAgentActOutput:
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
    decision: Dict[str, Any]
    world: Dict[str, torch.Tensor]
    critic: Any
    features: Dict[str, Any]
    ocr: Any
    intention_texts: List[str]
    losses: Dict[str, torch.Tensor] = field(default_factory=dict)
    stages: Dict[str, Any] = field(default_factory=dict)
