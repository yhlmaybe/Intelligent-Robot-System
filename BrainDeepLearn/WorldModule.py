from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import SiteSpec, BaseOnlineWrapper, AGICoreModule, GrowableLoRALinear, GetParametersScale, HungarianAssignment
from ModuleMessagerManager import ModuleDim
from PhysicalStateModule import (
    ContractPhysicalStateAdapter,
    PSTWorldBinder)
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    RobotEmbodimentContractView,
    SlotExecutionStatus,)


ROTATION_QUATERNION_DIM = 4
SelfRealmIndex = 0
VirtualRealmIndex = 2
EffectRealmIndex = 3
WorldConsciousSourceEntity = 8
WorldConsciousSourceHistory = 9


class ContractWorldFeedbackAdapter(nn.Module):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        cognitiveDim: int,
    ) -> None:
        super().__init__()
        if type(cognitiveDim) is not int or cognitiveDim < 1:
            raise ValueError("cognitiveDim must be a positive integer")
        self.ContractView = contractView
        self.CognitiveDim = int(cognitiveDim)
        self.BodyAdapter = ContractPhysicalStateAdapter(
            contractView,
            cognitiveDim)
        self.EndpointAdapters = nn.ModuleList()
        for endpointIndex in range(contractView.end_effector_count):
            endpointWidth = contractView.end_effector_feedback_layout.Width(
                endpointIndex)
            inputNormalization = (
                nn.Identity()
                if endpointWidth == 1
                else nn.LayerNorm(endpointWidth))
            self.EndpointAdapters.append(nn.Sequential(
                inputNormalization,
                nn.Linear(endpointWidth, self.CognitiveDim),
                nn.SiLU()))
        self.SlotNorm = nn.LayerNorm(self.CognitiveDim)
        self.OutputAdapter = nn.Sequential(
            nn.LayerNorm(cognitiveDim),
            nn.Linear(cognitiveDim, cognitiveDim),
            nn.SiLU())

    def forward(
        self,
        feedback: BrainFeedbackPacket,
    ) -> Dict[str, torch.Tensor]:
        body = self.BodyAdapter(feedback)
        direct_tokens = torch.stack(tuple(
            adapter(feedback.end_effector_features[
                ...,
                self.ContractView.end_effector_feedback_layout.Slice(
                    endpointIndex)])
            for endpointIndex, adapter in enumerate(
                self.EndpointAdapters)), dim=1)
        direct_tokens = direct_tokens * feedback.endpoint_present.to(
            dtype=direct_tokens.dtype).unsqueeze(-1)
        slot_weight = body["SlotWeight"]
        slot_tokens = self.SlotNorm(
            body["SlotBodyTokens"] + direct_tokens)
        slot_tokens = slot_tokens * slot_weight.unsqueeze(-1)
        summary = slot_tokens.sum(dim=1) / slot_weight.sum(
            dim=1,
            keepdim=True).clamp_min(1.0)
        return {
            **body,
            "SlotFeedbackTokens": slot_tokens,
            "EncodedFeedback": self.OutputAdapter(summary),
        }




class ContractWorldFeedbackPredictor(nn.Module):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        cognitiveDim: int,
    ) -> None:
        super().__init__()
        if type(cognitiveDim) is not int or cognitiveDim < 1:
            raise ValueError("cognitiveDim must be a positive integer")
        self.ContractView = contractView
        self.CognitiveDim = int(cognitiveDim)
        self.register_buffer(
            "StaticJointTokens",
            torch.tensor(
                contractView.static_joint_tokens,
                dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            "StaticEndpointTokens",
            torch.tensor(
                contractView.static_end_effector_tokens,
                dtype=torch.float32),
            persistent=True)
        self.JointStaticAdapter = nn.Linear(
            contractView.model_shape.joint_static_descriptor_dim,
            self.CognitiveDim)
        self.EndpointStaticAdapter = nn.Linear(
            contractView.model_shape.end_effector_static_descriptor_dim,
            self.CognitiveDim)
        self.JointFeedbackHeads = nn.ModuleList([
            nn.Linear(
                self.CognitiveDim,
                2 * contractView.joint_feedback_layout.Width(jointIndex))
            for jointIndex in range(contractView.joint_count)
        ])
        self.EndpointFeedbackHeads = nn.ModuleList([
            nn.Linear(
                self.CognitiveDim,
                2 * contractView.end_effector_feedback_layout.Width(endpointIndex))
            for endpointIndex in range(contractView.end_effector_count)
        ])
        self.EndpointStatusHead = nn.Linear(self.CognitiveDim, 5)

    @staticmethod
    def MaskedMean(
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weight = mask.to(dtype=value.dtype)
        while weight.dim() < value.dim():
            weight = weight.unsqueeze(-1)
        weight = weight.expand_as(value)
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    def ComputeLoss(
        self,
        prediction: Dict[str, torch.Tensor],
        feedback: BrainFeedbackPacket,
        sampleMask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if feedback.progress.dtype != feedback.joint_features.dtype:
            raise ValueError("progress and joint features must share one dtype")
        required = {
            "PackedJointFeatures",
            "PackedJointLogVariance",
            "PackedEndEffectorFeatures",
            "PackedEndEffectorLogVariance",
            "Progress",
            "ProgressLogVariance",
            "ReachedLogits",
            "LatentRisk",
            "LatentFeasibility",
            "LatentRiskKnown",
            "LatentFeasibilityKnown",
        }
        if set(prediction) != required:
            raise ValueError("feedback prediction fields do not match the contract")
        batch_size = int(feedback.joint_features.size(0))
        if sampleMask is None:
            sample_mask = torch.ones(
                batch_size,
                device=feedback.joint_features.device,
                dtype=torch.bool)
        elif (
            not torch.is_tensor(sampleMask)
            or tuple(sampleMask.shape) != (batch_size,)
            or sampleMask.device != feedback.joint_features.device
            or sampleMask.dtype != torch.bool
        ):
            raise ValueError("sampleMask must be a batched boolean mask")
        else:
            sample_mask = sampleMask
        endpoint_shape = (
            batch_size,
            self.ContractView.end_effector_count)
        for name in ("PackedJointFeatures", "PackedJointLogVariance"):
            packed_prediction = prediction[name]
            if (
                not torch.is_tensor(packed_prediction)
                or tuple(packed_prediction.shape)
                != tuple(feedback.joint_features.shape)
                or not packed_prediction.is_floating_point()
                or packed_prediction.device != feedback.joint_features.device
                or packed_prediction.dtype != feedback.joint_features.dtype
                or not bool(torch.isfinite(packed_prediction).all().item())
            ):
                raise ValueError("packed joint prediction has the wrong shape")
        for name in (
            "PackedEndEffectorFeatures",
            "PackedEndEffectorLogVariance",
        ):
            endpoint_prediction = prediction[name]
            if (
                not torch.is_tensor(endpoint_prediction)
                or tuple(endpoint_prediction.shape)
                != tuple(feedback.end_effector_features.shape)
                or not endpoint_prediction.is_floating_point()
                or endpoint_prediction.device != feedback.joint_features.device
                or endpoint_prediction.dtype != feedback.joint_features.dtype
                or not bool(torch.isfinite(endpoint_prediction).all().item())
            ):
                raise ValueError("packed endpoint prediction has the wrong shape")
        for name in (
            "Progress",
            "ProgressLogVariance",
            "ReachedLogits",
            "LatentRisk",
            "LatentFeasibility",
        ):
            value = prediction[name]
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != endpoint_shape
                or not value.is_floating_point()
                or value.device != feedback.joint_features.device
                or value.dtype != feedback.joint_features.dtype
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError("endpoint prediction has the wrong shape")
        for name in ("LatentRiskKnown", "LatentFeasibilityKnown"):
            value = prediction[name]
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != endpoint_shape
                or value.dtype != torch.bool
                or value.device != feedback.joint_features.device
                or bool(value.any().item())
            ):
                raise ValueError("latent supervision availability is invalid")
        packed_mask = sample_mask.unsqueeze(-1).expand_as(
            feedback.joint_features)
        endpoint_mask = torch.cat([
            feedback.endpoint_present[:, endpointIndex].unsqueeze(-1).expand(
                -1,
                self.ContractView.end_effector_feedback_layout.Width(
                    endpointIndex))
            for endpointIndex in range(self.ContractView.end_effector_count)
        ], dim=-1)
        endpoint_mask = endpoint_mask & sample_mask.unsqueeze(-1)
        target_mask = (
            feedback.endpoint_present
            & feedback.applied_target_active
            & sample_mask.unsqueeze(-1))
        joint_error = (
            prediction["PackedJointFeatures"]
            - feedback.joint_features.detach())
        joint_nll = 0.5 * (
            joint_error.square()
            * torch.exp(-prediction["PackedJointLogVariance"])
            + prediction["PackedJointLogVariance"])
        endpoint_error = (
            prediction["PackedEndEffectorFeatures"]
            - feedback.end_effector_features.detach())
        endpoint_nll = 0.5 * (
            endpoint_error.square()
            * torch.exp(-prediction["PackedEndEffectorLogVariance"])
            + prediction["PackedEndEffectorLogVariance"])
        progress_error = prediction["Progress"] - feedback.progress.detach()
        progress_nll = 0.5 * (
            progress_error.square()
            * torch.exp(-prediction["ProgressLogVariance"])
            + prediction["ProgressLogVariance"])
        loss_joint_features = self.MaskedMean(joint_nll, packed_mask)
        loss_progress = self.MaskedMean(progress_nll, target_mask)
        loss_endpoint_features = self.MaskedMean(
            endpoint_nll,
            endpoint_mask)
        loss_reached = self.MaskedMean(
            F.binary_cross_entropy_with_logits(
                prediction["ReachedLogits"],
                feedback.reached.to(dtype=feedback.joint_features.dtype),
                reduction="none"),
            target_mask)
        loss = (
            loss_joint_features
            + loss_endpoint_features
            + loss_progress
            + loss_reached)
        joint_weight = packed_mask.to(dtype=joint_nll.dtype)
        endpoint_weight = endpoint_mask.to(dtype=endpoint_nll.dtype)
        progress_weight = target_mask.to(dtype=progress_nll.dtype)
        reached_nll = F.binary_cross_entropy_with_logits(
            prediction["ReachedLogits"],
            feedback.reached.to(dtype=feedback.joint_features.dtype),
            reduction="none")
        joint_standardized = joint_error.abs() * torch.exp(
            -0.5 * prediction["PackedJointLogVariance"])
        endpoint_standardized = endpoint_error.abs() * torch.exp(
            -0.5 * prediction["PackedEndEffectorLogVariance"])
        progress_standardized = progress_error.abs() * torch.exp(
            -0.5 * prediction["ProgressLogVariance"])
        joint_robust = torch.where(
            joint_standardized.le(1.0),
            0.5 * joint_standardized.square(),
            joint_standardized - 0.5)
        endpoint_robust = torch.where(
            endpoint_standardized.le(1.0),
            0.5 * endpoint_standardized.square(),
            endpoint_standardized - 0.5)
        progress_robust = torch.where(
            progress_standardized.le(1.0),
            0.5 * progress_standardized.square(),
            progress_standardized - 0.5)
        surprise_numerator = (
            (joint_robust * joint_weight).sum(dim=-1)
            + (endpoint_robust * endpoint_weight).sum(dim=-1)
            + (progress_robust * progress_weight).sum(dim=-1)
            + (reached_nll * progress_weight).sum(dim=-1))
        surprise_count = (
            joint_weight.sum(dim=-1)
            + endpoint_weight.sum(dim=-1)
            + 2.0 * progress_weight.sum(dim=-1))
        surprise_valid = surprise_count.gt(0.0)
        normalized_surprise = torch.where(
            surprise_valid,
            torch.sqrt((surprise_numerator / surprise_count.clamp_min(1.0))
                       .clamp_min(0.0)),
            torch.zeros_like(surprise_count))
        return {
            "loss": loss,
            "loss_joint_features": loss_joint_features,
            "loss_endpoint_features": loss_endpoint_features,
            "loss_progress": loss_progress,
            "loss_reached": loss_reached,
            "normalized_surprise": normalized_surprise,
            "surprise_valid": surprise_valid,
        }

    def forward(self, priorWorldState: torch.Tensor) -> Dict[str, torch.Tensor]:
        if (
            not torch.is_tensor(priorWorldState)
            or priorWorldState.dim() != 2
            or int(priorWorldState.size(-1)) != self.CognitiveDim
            or not priorWorldState.is_floating_point()
            or not bool(torch.isfinite(priorWorldState).all().item())
        ):
            raise ValueError("priorWorldState must be a finite fixed-width feature")
        joint_static = self.JointStaticAdapter(
            self.StaticJointTokens.to(
                device=priorWorldState.device,
                dtype=priorWorldState.dtype)).unsqueeze(0)
        joint_context = priorWorldState.unsqueeze(1) + joint_static
        packed_joint_distribution = torch.cat([
            head(joint_context[:, jointIndex])
            for jointIndex, head in enumerate(self.JointFeedbackHeads)
        ], dim=-1)
        packed_joint_mean = []
        packed_joint_log_variance = []
        offset = 0
        for jointIndex in range(self.ContractView.joint_count):
            width = self.ContractView.joint_feedback_layout.Width(jointIndex)
            distribution = packed_joint_distribution[
                :, offset:offset + 2 * width]
            mean, log_variance = distribution.split(width, dim=-1)
            packed_joint_mean.append(mean)
            packed_joint_log_variance.append(log_variance.clamp(-8.0, 4.0))
            offset += 2 * width
        endpoint_static = self.EndpointStaticAdapter(
            self.StaticEndpointTokens.to(
                device=priorWorldState.device,
                dtype=priorWorldState.dtype)).unsqueeze(0)
        endpoint_context = priorWorldState.unsqueeze(1) + endpoint_static
        packed_endpoint_distribution = torch.cat([
            head(endpoint_context[:, endpointIndex])
            for endpointIndex, head in enumerate(
                self.EndpointFeedbackHeads)
        ], dim=-1)
        packed_endpoint_mean = []
        packed_endpoint_log_variance = []
        offset = 0
        for endpointIndex in range(self.ContractView.end_effector_count):
            width = self.ContractView.end_effector_feedback_layout.Width(
                endpointIndex)
            distribution = packed_endpoint_distribution[
                :, offset:offset + 2 * width]
            mean, log_variance = distribution.split(width, dim=-1)
            packed_endpoint_mean.append(mean)
            packed_endpoint_log_variance.append(
                log_variance.clamp(-8.0, 4.0))
            offset += 2 * width
        status = self.EndpointStatusHead(endpoint_context)
        unknown = torch.zeros(
            priorWorldState.size(0),
            self.ContractView.end_effector_count,
            device=priorWorldState.device,
            dtype=torch.bool)
        return {
            "PackedJointFeatures": torch.cat(packed_joint_mean, dim=-1),
            "PackedJointLogVariance": torch.cat(
                packed_joint_log_variance,
                dim=-1),
            "PackedEndEffectorFeatures": torch.cat(
                packed_endpoint_mean,
                dim=-1),
            "PackedEndEffectorLogVariance": torch.cat(
                packed_endpoint_log_variance,
                dim=-1),
            "Progress": torch.sigmoid(status[..., 0]),
            "ProgressLogVariance": status[..., 1].clamp(-8.0, 4.0),
            "ReachedLogits": status[..., 2],
            "LatentRisk": torch.sigmoid(status[..., 3]),
            "LatentFeasibility": torch.sigmoid(status[..., 4]),
            "LatentRiskKnown": unknown,
            "LatentFeasibilityKnown": unknown.clone(),
        }


class ContractWorldEmbodimentAdapter(nn.Module):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        cognitiveDim: int,
        actionDim: int,
    ) -> None:
        super().__init__()
        if type(cognitiveDim) is not int or cognitiveDim < 1:
            raise ValueError("cognitiveDim must be a positive integer")
        self.ContractView = contractView
        self.CognitiveDim = int(cognitiveDim)
        self.FeedbackAdapter = ContractWorldFeedbackAdapter(
            contractView,
            cognitiveDim)
        self.FeedbackPredictor = ContractWorldFeedbackPredictor(
            contractView,
            cognitiveDim)
        self.ActionDim = int(actionDim)
        self.ExecutionActionStateDim = len(SlotExecutionStatus) + 4
        self.ExecutionActionAdapter = nn.Sequential(
            nn.LayerNorm(
                self.ActionDim + self.ExecutionActionStateDim),
            nn.Linear(
                self.ActionDim + self.ExecutionActionStateDim,
                self.ActionDim * 2),
            nn.SiLU(),
            nn.Linear(self.ActionDim * 2, self.ActionDim),
            nn.LayerNorm(self.ActionDim))
        self.PerceptionRotationAdapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(
                    contractView.perception_motion_layout.Width(
                        perceptionIndex)),
                nn.Linear(
                    contractView.perception_motion_layout.Width(
                        perceptionIndex),
                    self.CognitiveDim),
                nn.SiLU())
            for perceptionIndex in range(
                len(contractView.perception_view_indices))
        ])
        self.ExecutionStateAdapter = nn.Sequential(
            nn.Linear(10, self.CognitiveDim),
            nn.SiLU(),
            nn.Linear(self.CognitiveDim, self.CognitiveDim))
        self.ActivityAdapter = nn.Sequential(
            nn.Linear(8, self.CognitiveDim),
            nn.SiLU(),
            nn.Linear(self.CognitiveDim, self.CognitiveDim))
        self.SlotFusion = nn.Sequential(
            nn.LayerNorm(self.CognitiveDim * 5),
            nn.Linear(self.CognitiveDim * 5, self.CognitiveDim * 2),
            nn.SiLU(),
            nn.Linear(self.CognitiveDim * 2, self.CognitiveDim))
        self.ParentMessageAdapter = nn.Sequential(
            nn.LayerNorm(self.CognitiveDim),
            nn.Linear(self.CognitiveDim, self.CognitiveDim, bias=False),
            nn.SiLU())
        self.ChildMessageAdapter = nn.Sequential(
            nn.LayerNorm(self.CognitiveDim),
            nn.Linear(self.CognitiveDim, self.CognitiveDim, bias=False),
            nn.SiLU())
        self.GraphNorm = nn.LayerNorm(self.CognitiveDim)
        self.TransitionAdapter = nn.Sequential(
            nn.LayerNorm(self.CognitiveDim * 5),
            nn.Linear(self.CognitiveDim * 5, self.CognitiveDim * 2),
            nn.SiLU(),
            nn.Linear(self.CognitiveDim * 2, self.CognitiveDim))

        parent_graph = torch.zeros(
            contractView.end_effector_count,
            contractView.end_effector_count,
            dtype=torch.float32)
        child_graph = torch.zeros_like(parent_graph)
        for childIndex, parentIndex in enumerate(contractView.parent_index):
            if parentIndex >= 0:
                parent_graph[childIndex, parentIndex] = 1.0
                child_graph[parentIndex, childIndex] = 1.0
        self.register_buffer(
            "ParentGraph",
            parent_graph,
            persistent=True)
        self.register_buffer(
            "ChildGraph",
            child_graph,
            persistent=True)

    def EncodeExecutionAction(
        self,
        action: torch.Tensor,
        executionStatus: torch.Tensor,
        executionRelevant: torch.Tensor,
        executionKnown: torch.Tensor,
        executionResultKnown: torch.Tensor,
        hardStop: torch.Tensor,
        helpAccepted: torch.Tensor,
        targetActive: torch.Tensor,
    ) -> torch.Tensor:
        dtype = action.dtype
        status = F.one_hot(
            executionStatus,
            num_classes=len(SlotExecutionStatus)).to(dtype=dtype)
        relevant = torch.where(
            executionRelevant.any(dim=-1, keepdim=True),
            executionRelevant,
            torch.ones_like(executionRelevant))
        status_weight = relevant.to(dtype=dtype).unsqueeze(-1)
        status = (status * status_weight).sum(dim=1) / status_weight.sum(
            dim=1).clamp_min(1.0)
        state = torch.cat((
            status,
            executionResultKnown.to(dtype=dtype).unsqueeze(-1),
            hardStop.to(dtype=dtype).unsqueeze(-1),
            helpAccepted.to(dtype=dtype).unsqueeze(-1),
            targetActive.to(dtype=dtype).mean(
                dim=-1,
                keepdim=True),
        ), dim=-1)
        actionKnown = (
            targetActive
            & executionRelevant
            & executionKnown).any(dim=-1)
        known_action = action * actionKnown.to(
            dtype=dtype).unsqueeze(-1)
        return known_action + self.ExecutionActionAdapter(torch.cat((
            known_action,
            state,
        ), dim=-1))

    @staticmethod
    def PropagateGraph(
        slotTokens: torch.Tensor,
        slotWeight: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        weighted_graph = (
            adjacency.to(
                device=slotTokens.device,
                dtype=slotTokens.dtype).unsqueeze(0)
            * slotWeight.unsqueeze(1))
        normalizer = weighted_graph.sum(
            dim=-1,
            keepdim=True).clamp_min(1.0)
        return torch.bmm(weighted_graph, slotTokens) / normalizer

    @staticmethod
    def MaskedMoments(
        slotTokens: torch.Tensor,
        slotWeight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = slotWeight.unsqueeze(-1)
        normalizer = weight.sum(dim=1).clamp_min(1.0)
        mean = (slotTokens * weight).sum(dim=1) / normalizer
        variance = (
            (slotTokens - mean.unsqueeze(1)).square() * weight
        ).sum(dim=1) / normalizer
        return mean, variance

    def EncodeExecutionState(
        self,
        feedback: BrainFeedbackPacket,
    ) -> torch.Tensor:
        dtype = feedback.joint_features.dtype
        endpoint_present = feedback.endpoint_present.to(dtype=dtype)
        active_present = (
            feedback.endpoint_present
            & feedback.applied_target_active).to(dtype=dtype)
        active_count = active_present.sum(dim=-1).clamp_min(1.0)
        progress = (
            feedback.progress * active_present
        ).sum(dim=-1) / active_count
        reached = (
            feedback.reached.to(dtype=dtype) * active_present
        ).sum(dim=-1) / active_count
        phase_enabled = feedback.phase_enabled.to(dtype=dtype).mean(dim=-1)
        phase_known = feedback.phase_known.to(dtype=dtype).mean(dim=-1)
        target_active = feedback.applied_target_active.to(
            dtype=dtype).mean(dim=-1)
        endpoint_present_fraction = endpoint_present.mean(dim=-1)
        if int(feedback.perception_motion_present.size(-1)) > 0:
            perception_motion_present_fraction = feedback.perception_motion_present.to(
                dtype=dtype).mean(dim=-1)
        else:
            perception_motion_present_fraction = torch.zeros_like(progress)
        execution_relevant = torch.where(
            feedback.execution_relevant.any(dim=-1, keepdim=True),
            feedback.execution_relevant,
            torch.ones_like(feedback.execution_relevant))
        execution_known = (
            feedback.execution_known
            & execution_relevant).to(dtype=dtype).sum(dim=-1) / (
                execution_relevant.to(dtype=dtype).sum(
                    dim=-1).clamp_min(1.0))
        execution_result_known = feedback.execution_result_known.to(
            dtype=dtype)
        action_epoch = feedback.applied_action_epoch.to(dtype=dtype)
        action_epoch = action_epoch / (1.0 + action_epoch)
        state = torch.stack((
            progress,
            reached,
            phase_enabled,
            phase_known,
            target_active,
            endpoint_present_fraction,
            perception_motion_present_fraction,
            execution_known,
            execution_result_known,
            action_epoch,
        ), dim=-1)
        return self.ExecutionStateAdapter(state)

    def EncodePerceptionRotation(
        self,
        feedback: BrainFeedbackPacket,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotation_tokens = []
        for perceptionIndex, adapter in enumerate(
            self.PerceptionRotationAdapters
        ):
            rotation_tokens.append(adapter(
                feedback.perception_motion_features[
                    ...,
                    self.ContractView.perception_motion_layout.Slice(
                        perceptionIndex)]))
        batch_size = int(feedback.joint_features.size(0))
        if rotation_tokens:
            perception_tokens = torch.stack(rotation_tokens, dim=1)
            perception_motion_present = feedback.perception_motion_present
            perception_tokens = perception_tokens * perception_motion_present.to(
                dtype=perception_tokens.dtype).unsqueeze(-1)
        else:
            perception_tokens = feedback.joint_features.new_zeros(
                batch_size,
                0,
                self.CognitiveDim)
            perception_motion_present = feedback.endpoint_present.new_zeros(batch_size, 0)
        full_tokens = feedback.joint_features.new_zeros(
            batch_size,
            self.ContractView.end_effector_count,
            self.CognitiveDim)
        if rotation_tokens:
            perception_index = torch.tensor(
                self.ContractView.perception_view_indices,
                dtype=torch.long,
                device=feedback.joint_features.device)
            full_tokens = full_tokens.index_copy(
                1,
                perception_index,
                perception_tokens)
        return full_tokens, perception_tokens, perception_motion_present

    def EncodeFeedback(
        self,
        feedback: BrainFeedbackPacket,
    ) -> Dict[str, torch.Tensor]:
        return self.FeedbackAdapter(feedback)

    def PredictFeedback(
        self,
        priorWorldState: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.FeedbackPredictor(priorWorldState)

    def ComputeFeedbackLoss(
        self,
        prediction: Dict[str, torch.Tensor],
        feedback: BrainFeedbackPacket,
        sampleMask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.FeedbackPredictor.ComputeLoss(
            prediction,
            feedback,
            sampleMask=sampleMask)

    def EncodeTransition(
        self,
        feedback: BrainFeedbackPacket,
    ) -> Dict[str, torch.Tensor]:
        encoded_feedback = self.FeedbackAdapter(feedback)
        control_feedback = encoded_feedback["ControlFeedbackFeature"]
        control_slots = control_feedback.unsqueeze(1).expand(
            -1,
            self.ContractView.end_effector_count,
            -1)
        execution_state = self.EncodeExecutionState(feedback)
        full_rotation, perception_rotation, perception_motion_present = (
            self.EncodePerceptionRotation(feedback))
        activity = self.ActivityAdapter(torch.stack((
            feedback.applied_target_active.to(
                dtype=feedback.joint_features.dtype),
            feedback.reached.to(dtype=feedback.joint_features.dtype),
            feedback.phase_enabled.to(dtype=feedback.joint_features.dtype),
            feedback.phase_known.to(dtype=feedback.joint_features.dtype),
            feedback.endpoint_present.to(dtype=feedback.joint_features.dtype),
            feedback.execution_known.to(dtype=feedback.joint_features.dtype),
            feedback.execution_relevant.to(
                dtype=feedback.joint_features.dtype),
            feedback.execution_status.to(
                dtype=feedback.joint_features.dtype) / float(
                    max(int(value) for value in SlotExecutionStatus)),
        ), dim=-1))

        slot_mask = torch.ones_like(
            feedback.endpoint_present,
            dtype=feedback.joint_features.dtype)
        slot_weight = slot_mask
        local_tokens = self.SlotFusion(torch.cat((
            encoded_feedback["SlotFeedbackTokens"],
            control_slots,
            full_rotation,
            activity,
            execution_state.unsqueeze(1).expand(
                -1,
                self.ContractView.end_effector_count,
                -1),
        ), dim=-1))
        local_tokens = local_tokens * slot_mask.unsqueeze(-1)
        parent_message = self.ParentMessageAdapter(self.PropagateGraph(
            local_tokens,
            slot_weight,
            self.ParentGraph))
        child_message = self.ChildMessageAdapter(self.PropagateGraph(
            local_tokens,
            slot_weight,
            self.ChildGraph))
        transition_slots = self.GraphNorm(
            local_tokens + parent_message + child_message)
        transition_slots = transition_slots * slot_mask.unsqueeze(-1)
        slot_mean, slot_variance = self.MaskedMoments(
            transition_slots,
            slot_weight)
        encoded_transition = self.TransitionAdapter(torch.cat((
            encoded_feedback["EncodedFeedback"],
            control_feedback,
            slot_mean,
            slot_variance,
            execution_state,
        ), dim=-1))
        return {
            **encoded_feedback,
            "EncodedControlFeedback": control_feedback,
            "ExecutionStateFeature": execution_state,
            "PerceptionRotationTokens": perception_rotation,
            "PerceptionMotionPresent": perception_motion_present,
            "TransitionSlotTokens": transition_slots,
            "TransitionSlotMean": slot_mean,
            "TransitionSlotVariance": slot_variance,
            "EncodedTransition": encoded_transition,
        }


PERSISTENT_PHYSICAL_STATE_FIELDS = (
    "SlotState",
    "SpatialWorld",
    "ARaw",
    "SlotPresence",
    "MphysRaw",
    "PerceptualPresence",
    "GeometryValidMask",
    "PhysicalEntityProb",
    "PhysicalInteractionProb",
    "RealmProb",
    "MotionLayerProb",
    "LayerAgencyProb",
    "AgencyProb",
    "BodyMembershipProb",
    "SelfPartProb",
    "SelfPartSemantic",
    "IdentityKey",
    "PairwiseRelationWorld",
    "PairRelationLastSeen",
    "ExternalRelationProbRaw",
    "Semantic",
    "Size",
    "StateRaw",
    "AffordanceRaw",
    "MotionWorldRaw",
    "CarrierMotionWorldRaw",
    "ArticulationMotionWorldRaw",
    "ContentMotionUV",
    "ContentChangeProb",
    "MovingProbRaw",
    "ContactProbRaw",
    "ContactForceRaw",
    "ContactPointWorldRaw",
    "ParentProb",
    "DisplaySurfaceProb",
    "SurfaceParentProb",
    "SurfaceUV",
    "SurfaceUVConfidence",
    "VerificationConfidence",
    "OntologyRelationProb",
    "Visibility",
    "Occlusion",
    "HasTextProb",
    "TextEmbed",
    "SymbolProb",
    "Observed",
    "LastSeen",
    "Step",
)
MODEL_PHYSICAL_GEOMETRY_FIELDS = (
    "SpatialFrame",
    "MotionObserverRaw",
    "CarrierMotionObserverRaw",
    "ArticulationMotionObserverRaw",
    "ContactPointObserverRaw",
    "PairwiseRelationObserver",
)
MODEL_SEMANTIC_VIEW_FIELDS = (
    "LevelProb",
    "ObjectClassProb",
    "PartClassProb",
)
PERSISTENT_WORLD_GEOMETRY_FIELDS = (
    "SpatialWorld",
    "MotionWorldRaw",
    "CarrierMotionWorldRaw",
    "ArticulationMotionWorldRaw",
    "ContactPointWorldRaw",
    "PairwiseRelationWorld",
)
MODEL_PHYSICAL_STATE_FIELDS = tuple(
    name
    for name in PERSISTENT_PHYSICAL_STATE_FIELDS
    if name not in PERSISTENT_WORLD_GEOMETRY_FIELDS
) + MODEL_PHYSICAL_GEOMETRY_FIELDS + MODEL_SEMANTIC_VIEW_FIELDS
OBSERVED_PHYSICAL_STATE_FIELDS = (
    "SlotState",
    "SpatialFrame",
    "ARaw",
    "ObservedSlotMask",
    "MphysRaw",
    "PerceptualPresence",
    "GeometryValidMask",
    "PhysicalEntityProb",
    "PhysicalInteractionProb",
    "RealmProb",
    "MotionLayerProb",
    "LayerAgencyProb",
    "AgencyProb",
    "BodyMembershipProb",
    "SelfPartProb",
    "SelfPartSemantic",
    "IdentityKey",
    "Semantic",
    "ExternalRelationProbRaw",
    "Size",
    "StateRaw",
    "AffordanceRaw",
    "MotionObserverRaw",
    "CarrierMotionObserverRaw",
    "ArticulationMotionObserverRaw",
    "ContentMotionUV",
    "ContentChangeProb",
    "MovingProbRaw",
    "ContactProbRaw",
    "ContactForceRaw",
    "ContactPointObserverRaw",
    "Visibility",
    "Occlusion",
    "HasTextProb",
    "TextEmbed",
    "SymbolProb",
    "PairwiseRelationObserver",
    "ParentProb",
    "DisplaySurfaceProb",
    "SurfaceParentProb",
    "SurfaceUV",
    "SurfaceUVConfidence",
    "VerificationConfidence",
    "OntologyRelationProb",
)

WORLD_MEMORY_SCHEMA_VERSION = 8
WORLD_MEMORY_TENSOR_FIELDS = (
    "mem_keys",
    "mem_vals",
    "mem_imp",
    "mem_steps",
    "mem_size",
    "mem_global_step",
    "pst_slot_state",
    "pst_spatial_world",
    "pst_attribute",
    "pst_slot_presence",
    "pst_entity_prob",
    "pst_entity_id",
    "pst_slot_generation",
    "pst_next_entity_id",
    "last_observed_to_world_slot",
    "pst_perceptual_presence",
    "pst_geometry_valid",
    "pst_physical_interaction",
    "pst_realm",
    "pst_motion_layer",
    "pst_layer_agency",
    "pst_agency",
    "pst_body_membership",
    "pst_self_part",
    "pst_self_part_semantic",
    "pst_identity_key",
    "pst_pairwise_relation",
    "pst_pair_last_seen",
    "pst_external_relation",
    "pst_semantic",
    "pst_size",
    "pst_state",
    "pst_affordance",
    "pst_motion",
    "pst_carrier_motion",
    "pst_articulation_motion",
    "pst_content_motion",
    "pst_content_change",
    "pst_moving",
    "pst_contact",
    "pst_contact_force",
    "pst_contact_point",
    "pst_parent",
    "pst_display_surface",
    "pst_surface_parent",
    "pst_surface_uv",
    "pst_surface_uv_confidence",
    "pst_verification",
    "pst_ontology_relation",
    "pst_visibility",
    "pst_occlusion",
    "pst_has_text",
    "pst_text",
    "pst_entity_text_semantic",
    "pst_entity_text_confidence",
    "pst_entity_text_revision",
    "pst_entity_text_changed",
    "pst_symbol",
    "pst_observed",
    "pst_last_seen",
    "pst_step",
)
WORLD_MEMORY_PAYLOAD_FIELDS = frozenset((
    "world_memory_schema_version",
    "calibration_id",
    "world_frame_id",
    "batch_size",
    "pst_contact_point_frame",
    *WORLD_MEMORY_TENSOR_FIELDS,
))


def KLDiagNormal(muQ: torch.Tensor, logstdQ: torch.Tensor, muP: torch.Tensor, logstdP: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(2 * logstdQ)
    var_p = torch.exp(2 * logstdP)
    kl = 0.5 * (((var_q + (muQ - muP) ** 2) / var_p).sum(-1) + 2 * (logstdP - logstdQ).sum(-1) - muQ.size(-1))
    return kl


def BalancedKL(muQ: torch.Tensor,logstdQ: torch.Tensor,muP: torch.Tensor,logstdP: torch.Tensor,alpha: float = 0.8,freeNats: float = 1.0,) -> torch.Tensor:
    mu_p_sg, logstd_p_sg = muP.detach(), logstdP.detach()
    mu_q_sg, logstd_q_sg = muQ.detach(), logstdQ.detach()
    kl_qp = KLDiagNormal(muQ, logstdQ, mu_p_sg, logstd_p_sg)
    kl_pq = KLDiagNormal(mu_q_sg, logstd_q_sg, muP, logstdP)
    kl = alpha * kl_qp + (1.0 - alpha) * kl_pq
    if freeNats and freeNats > 0:
        kl = torch.relu(kl - freeNats)
    return kl


@dataclass
class PredictedVisualPack:
    GlobalFeat: torch.Tensor
    ObjectTokens: torch.Tensor
    MotionPred: torch.Tensor
    IntegratedFeat: torch.Tensor


class PredictedVisualHead(nn.Module):
    def __init__(
        self,
        stateDim: int,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        integratedFeatDim: int = 1024,):
        super().__init__()
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.integrated_feat_dim = int(integratedFeatDim)

        hidden = max(int(stateDim), self.global_feat_dim, self.object_token_dim * 2)

        def head(outDim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(int(stateDim)),
                nn.Linear(int(stateDim), hidden),
                nn.GELU(),
                nn.Linear(hidden, int(outDim)),)

        self.global_head = head(self.global_feat_dim)
        self.object_head = head(self.num_object_tokens * self.object_token_dim)
        self.motion_head = head(self.motion_pred_dim)
        self.integrated_head = head(self.integrated_feat_dim)

    def forward(self, state: torch.Tensor) -> PredictedVisualPack:
        B = int(state.size(0))
        return PredictedVisualPack(
            GlobalFeat=self.global_head(state),
            ObjectTokens=self.object_head(state).view(B, self.num_object_tokens, self.object_token_dim),
            MotionPred=self.motion_head(state),
            IntegratedFeat=self.integrated_head(state),)

class VisualReconstructor(nn.Module):
    def __init__(
        self,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        integratedFeatDim: int = 1024,):
        super().__init__()
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.integrated_feat_dim = int(integratedFeatDim)
        self.recon_dim = min(256, self.object_token_dim)
        self.head_count = 8

        scene_in_dim = self.global_feat_dim + self.integrated_feat_dim

        self.scene_encoder = nn.Sequential(
            nn.LayerNorm(scene_in_dim),
            nn.Linear(scene_in_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(self.motion_pred_dim),
            nn.Linear(self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(self.object_token_dim),
            nn.Linear(self.object_token_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.slot_context_fuser = nn.Sequential(
            nn.LayerNorm(self.recon_dim * 3),
            nn.Linear(self.recon_dim * 3, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.slot_presence_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, 1),)

        self.factor_realm_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, ModuleDim.PstRealmClasses))
        self.factor_motion_layer_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, ModuleDim.PstMotionLayerClasses))
        self.factor_layer_agency_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(
                self.recon_dim,
                ModuleDim.PstMotionLayerClasses * ModuleDim.PstAgencyClasses))
        self.factor_surface_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, 1))
        self.factor_surface_uv_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, 2))
        self.factor_content_motion_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, 2))
        self.factor_content_change_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, 1))
        self.factor_confidence_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, 1))

        self.slot_norm = nn.LayerNorm(self.recon_dim)

        self.slot_relation_attn = nn.MultiheadAttention(
            self.recon_dim,
            self.head_count,
            batch_first=True,)

        self.scene_from_objects = nn.MultiheadAttention(
            self.recon_dim,
            self.head_count,
            batch_first=True,)

        self.scene_to_objects = nn.MultiheadAttention(
            self.recon_dim,
            self.head_count,
            batch_first=True,)

        self.scene_norm = nn.LayerNorm(self.recon_dim)

        self.motion_to_object_gate = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.recon_dim),
            nn.Sigmoid(),)

        self.slot_ffn = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.recon_dim * 2),
            nn.GELU(),
            nn.Linear(self.recon_dim * 2, self.recon_dim),)

        self.context_to_object = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.object_token_dim),)

        self.slot_to_object = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.object_token_dim),)

        self.object_decoder = nn.Sequential(
            nn.LayerNorm(self.recon_dim * 3),
            nn.Linear(self.recon_dim * 3, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.object_token_dim),)

        self.object_out_norm = nn.LayerNorm(self.object_token_dim)

        self.motion_reconstructor = nn.Sequential(
            nn.LayerNorm(self.motion_pred_dim + self.object_token_dim * 2),
            nn.Linear(self.motion_pred_dim + self.object_token_dim * 2, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.motion_pred_dim),)

        self.motion_out_norm = nn.LayerNorm(self.motion_pred_dim)

        self.global_reconstructor = nn.Sequential(
            nn.LayerNorm(self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim),
            nn.Linear(self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.global_feat_dim),)

        self.global_out_norm = nn.LayerNorm(self.global_feat_dim)

        self.integrated_reconstructor = nn.Sequential(
            nn.LayerNorm(self.integrated_feat_dim + self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim),
            nn.Linear(self.integrated_feat_dim + self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.integrated_feat_dim),)

        self.integrated_out_norm = nn.LayerNorm(self.integrated_feat_dim)

        self.pred_error_basis = nn.Sequential(
            nn.LayerNorm(self.global_feat_dim + self.integrated_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim),
            nn.Linear(self.global_feat_dim + self.integrated_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.global_feat_dim),)

    def forward(self, predictedVisual: PredictedVisualPack) -> Dict[str, torch.Tensor]:
        scene_context = self.scene_encoder(torch.cat([
            predictedVisual.GlobalFeat,
            predictedVisual.IntegratedFeat,], dim=-1))

        motion_context = self.motion_encoder(predictedVisual.MotionPred)

        slot_base = self.slot_encoder(predictedVisual.ObjectTokens)

        slot_state = self.slot_context_fuser(torch.cat([
            slot_base,
            scene_context.unsqueeze(1).expand(-1, self.num_object_tokens, -1),
            motion_context.unsqueeze(1).expand(-1, self.num_object_tokens, -1),], dim=-1))

        slot_norm = self.slot_norm(slot_state)
        slot_relation, _ = self.slot_relation_attn(
            slot_norm,
            slot_norm,
            slot_norm,
            need_weights=False)
        slot_state = slot_state + slot_relation + self.slot_ffn(slot_state)

        slot_presence_logits = self.slot_presence_head(slot_state).squeeze(-1)
        slot_weight = F.softmax(slot_presence_logits, dim=-1)
        slot_summary = (slot_state * slot_weight.unsqueeze(-1)).sum(dim=1) # [B,256]
        scene_seed = self.scene_norm(scene_context + motion_context + slot_summary).unsqueeze(1)
        slot_norm = self.slot_norm(slot_state)

        scene_context, _ = self.scene_from_objects(
            scene_seed,
            slot_norm,
            slot_norm,
            need_weights=False)

        scene_token = self.scene_norm(scene_seed + scene_context)

        object_scene_delta, _ = self.scene_to_objects(
            slot_norm,
            scene_token,
            scene_token,
            need_weights=False)

        motion_gate = self.motion_to_object_gate(motion_context).unsqueeze(1)

        slot_state = self.slot_norm(
            slot_state
            + motion_gate * object_scene_delta
            + self.slot_ffn(slot_state))

        slot_presence_logits = self.slot_presence_head(slot_state).squeeze(-1)
        slot_weight = F.softmax(slot_presence_logits, dim=-1)
        object_summary_internal = (slot_state * slot_weight.unsqueeze(-1)).sum(dim=1)
        scene_summary_internal = scene_token.squeeze(1) # [B,256]

        object_tokens = self.object_out_norm(
            predictedVisual.ObjectTokens
            + self.object_decoder(torch.cat([
                slot_state,
                scene_summary_internal.unsqueeze(1).expand(-1, self.num_object_tokens, -1),
                motion_context.unsqueeze(1).expand(-1, self.num_object_tokens, -1),], dim=-1)))

        slot_state_object = self.slot_to_object(slot_state)
        scene_summary = self.context_to_object(scene_summary_internal)
        object_summary = self.context_to_object(object_summary_internal)

        motion_delta = self.motion_reconstructor(torch.cat([
            predictedVisual.MotionPred,
            scene_summary,
            object_summary,], dim=-1))

        motion_pred = self.motion_out_norm(predictedVisual.MotionPred + motion_delta)

        global_delta = self.global_reconstructor(torch.cat([
            predictedVisual.GlobalFeat,
            scene_summary,
            object_summary,
            motion_pred,], dim=-1))

        global_feat = self.global_out_norm(predictedVisual.GlobalFeat + global_delta)

        integrated_delta = self.integrated_reconstructor(torch.cat([
            predictedVisual.IntegratedFeat,
            global_feat,
            scene_summary,
            object_summary,
            motion_pred,], dim=-1))

        integrated_feat = self.integrated_out_norm(predictedVisual.IntegratedFeat + integrated_delta)
        slot_entropy = -(slot_weight * F.log_softmax(slot_presence_logits, dim=-1)).sum(dim=-1)
        slot_confidence = 1.0 - slot_entropy / slot_entropy.new_tensor(float(self.num_object_tokens)).log()
        scene_agreement = (F.cosine_similarity(scene_summary, object_summary, dim=-1) + 1.0) * 0.5
        prior_confidence = (0.5 + 0.5 * slot_confidence) * scene_agreement
        realm_prob = F.softmax(self.factor_realm_head(slot_state), dim=-1)
        motion_layer_prob = torch.sigmoid(
            self.factor_motion_layer_head(slot_state))
        layer_agency_prob = F.softmax(
            self.factor_layer_agency_head(slot_state).view(
                slot_state.size(0),
                slot_state.size(1),
                ModuleDim.PstMotionLayerClasses,
                ModuleDim.PstAgencyClasses),
            dim=-1)
        agency_mass = (
            motion_layer_prob.unsqueeze(-1) * layer_agency_prob
        ).sum(dim=-2)
        layer_mass = motion_layer_prob.sum(dim=-1, keepdim=True)
        agency_prob = agency_mass / (layer_mass + 1e-6)
        unknown_agency = torch.zeros_like(agency_prob)
        unknown_agency[..., -1] = 1.0
        agency_prob = torch.where(
            layer_mass > 1e-6,
            agency_prob,
            unknown_agency)
        factor_prior_confidence = (
            torch.sigmoid(self.factor_confidence_head(slot_state).squeeze(-1))
            * torch.sigmoid(slot_presence_logits)
            * prior_confidence.unsqueeze(-1))

        return {
            "IntegratedFeat": integrated_feat, # [B,1024]
            "GlobalFeat": global_feat, # [B,1024]
            "ObjectTokens": object_tokens, # [B,K,512]
            "MotionPred": motion_pred, # [B,512]
            "SlotState": slot_state_object, # [B,K,512]
            "SlotPresenceLogits": slot_presence_logits, # [B,K]
            "SceneSummary": scene_summary, # [B,512]
            "ObjectSummary": object_summary, # [B,512]
            "PriorConfidence": prior_confidence, # [B]
            "RealmProb": realm_prob,
            "MotionLayerProb": motion_layer_prob,
            "LayerAgencyProb": layer_agency_prob,
            "ObjectAgencyProb": agency_prob,
            "DisplaySurfaceProb": torch.sigmoid(
                self.factor_surface_head(slot_state).squeeze(-1)),
            "SurfaceUV": torch.sigmoid(
                self.factor_surface_uv_head(slot_state)),
            "ContentMotionUV": torch.tanh(
                self.factor_content_motion_head(slot_state)),
            "ContentChangeProb": torch.sigmoid(
                self.factor_content_change_head(slot_state).squeeze(-1)),
            "FactorPriorConfidence": factor_prior_confidence,
            "PredErrorBasis": self.pred_error_basis(torch.cat([
                global_feat,
                integrated_feat,
                scene_summary,
                object_summary,
                motion_pred,], dim=-1)),} # [B,1024]

    def PairwiseCosine(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(tokens, dim=-1, eps=1e-6)
        return torch.matmul(normed, normed.transpose(1, 2))

    def SoftAlignObjects(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = float(source.size(-1)) ** 0.5
        weights = F.softmax(torch.matmul(source, target.transpose(1, 2)) / scale, dim=-1)
        return torch.matmul(weights, target)

    def InverseMappingLoss(
        self,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        sampleMask: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        target_objects = targetVisualState.ObjectTokens.detach()
        target_motion = targetVisualState.MotionToken.detach()
        target_object_valid = targetVisualState.Auxiliary[
            "ObjectGeometryValid"].detach().squeeze(-1)
        target_object_prob = F.softmax(
            targetVisualState.SemanticNodes["node_logits"].detach(), dim=-1)[..., 1]
        target_object_weight = target_object_prob * target_object_valid
        target_weight_sum = target_object_weight.sum(dim=-1, keepdim=True)
        target_slot_weight = torch.where(
            target_weight_sum > 0.0,
            target_object_weight / target_weight_sum.clamp_min(1e-8),
            torch.zeros_like(target_object_weight))
        batch_size = int(target_objects.size(0))
        if sampleMask.dtype != torch.bool:
            raise TypeError(f"sampleMask must be bool, got {sampleMask.dtype}")
        if tuple(sampleMask.shape) != (batch_size,):
            raise ValueError(
                f"sampleMask must have shape ({batch_size},), got {tuple(sampleMask.shape)}")
        if sampleMask.device != target_objects.device:
            raise ValueError(
                f"sampleMask must be on {target_objects.device}, got {sampleMask.device}")
        base_sample = sampleMask
        object_sample = base_sample & (target_weight_sum.squeeze(-1) > 0.0)

        def MaskedMeanLocal(per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            numerator = torch.where(
                mask, per_sample, torch.zeros_like(per_sample)).sum()
            return numerator / mask.sum().clamp_min(1.0)

        def ObjectMean(per_sample: torch.Tensor) -> torch.Tensor:
            return MaskedMeanLocal(per_sample, object_sample)

        object_tokens = reconstructedVisualState["ObjectTokens"]
        slot_state = reconstructedVisualState["SlotState"]
        slot_logits = reconstructedVisualState["SlotPresenceLogits"]
        scene_summary = reconstructedVisualState["SceneSummary"]
        object_summary = reconstructedVisualState["ObjectSummary"]

        aligned_objects_for_target = self.SoftAlignObjects(target_objects, object_tokens)
        loss_inverse_object = ObjectMean(
            F.smooth_l1_loss(aligned_objects_for_target, target_objects, reduction="none").mean(dim=-1)
            .mul(target_slot_weight).sum(dim=-1))

        target_norm = F.normalize(target_objects, dim=-1, eps=1e-6)
        slot_norm = F.normalize(slot_state, dim=-1, eps=1e-6)
        target_slot_similarity = torch.matmul(target_norm, slot_norm.transpose(1, 2))
        slot_match_score = target_slot_similarity.max(dim=-1).values
        loss_inverse_slot = ObjectMean(
            ((1.0 - slot_match_score) * target_slot_weight).sum(dim=-1))

        target_to_slot_weight = F.softmax(target_slot_similarity, dim=-1)
        target_relation = self.PairwiseCosine(target_objects)
        aligned_slots_for_target = torch.matmul(target_to_slot_weight, slot_state)
        slot_relation = self.PairwiseCosine(aligned_slots_for_target)
        relation_weight = target_slot_weight.unsqueeze(1) * target_slot_weight.unsqueeze(2)
        loss_inverse_relation = ObjectMean((
            F.smooth_l1_loss(slot_relation, target_relation, reduction="none")
            * relation_weight).sum(dim=(1, 2)))

        pred_slot_prob = F.softmax(slot_logits, dim=-1)
        pred_presence_for_target = torch.matmul(target_to_slot_weight, pred_slot_prob.unsqueeze(-1)).squeeze(-1)
        pred_presence_for_target = pred_presence_for_target / pred_presence_for_target.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        presence_kl = torch.where(
            target_slot_weight > 0.0,
            target_slot_weight * (
                target_slot_weight.clamp_min(1e-8).log()
                - pred_presence_for_target.clamp_min(1e-8).log()),
            torch.zeros_like(target_slot_weight))
        loss_inverse_presence = ObjectMean(presence_kl.sum(dim=-1))

        target_summary = (target_objects * target_slot_weight.unsqueeze(-1)).sum(dim=1)
        loss_inverse_scene = ObjectMean(
            1.0 - F.cosine_similarity(scene_summary, target_summary, dim=-1))
        loss_inverse_summary = ObjectMean(
            F.smooth_l1_loss(object_summary, target_summary, reduction="none").mean(dim=-1))

        loss_inverse_motion = MaskedMeanLocal(F.smooth_l1_loss(
            F.normalize(reconstructedVisualState["MotionPred"], dim=-1, eps=1e-6),
            F.normalize(target_motion, dim=-1, eps=1e-6),
            reduction="none").mean(dim=-1), base_sample)

        loss_inverse_total = (
            loss_inverse_object
            + loss_inverse_slot
            + 0.5 * loss_inverse_relation
            + 0.25 * loss_inverse_presence
            + 0.5 * loss_inverse_scene
            + 0.25 * loss_inverse_summary
            + 0.25 * loss_inverse_motion)

        return {
            "loss_pred_inverse_object": loss_inverse_object,
            "loss_pred_inverse_slot": loss_inverse_slot,
            "loss_pred_inverse_relation": loss_inverse_relation,
            "loss_pred_inverse_presence": loss_inverse_presence,
            "loss_pred_inverse_scene": loss_inverse_scene,
            "loss_pred_inverse_summary": loss_inverse_summary,
            "loss_pred_inverse_motion": loss_inverse_motion,
            "loss_pred_inverse_total": loss_inverse_total,}

class LowRankMultiplicativeFlow(AGICoreModule):







    def __init__(
        self,
        stateDim: int,
        inputDim: int,
        rank: int = 64,
        maxMix: float = 0.10,
        targetScale: float = 0.50,):
        super().__init__()
        self.state_dim = int(stateDim)
        self.input_dim = int(inputDim)
        self.rank = min(int(rank), self.state_dim)
        self.max_mix = float(maxMix)
        self.target_scale = float(targetScale)

        self.left = GrowableLoRALinear(nn.Linear(self.state_dim, self.rank, bias=False))
        self.right = GrowableLoRALinear(nn.Linear(self.state_dim, self.rank, bias=False))
        self.context = GrowableLoRALinear(nn.Linear(self.input_dim, self.rank, bias=False))
        self.out = GrowableLoRALinear(nn.Linear(self.rank, self.state_dim, bias=False))
        self.selectivity = GrowableLoRALinear(nn.Linear(self.input_dim, 1, bias=True))
        self.gain = nn.Parameter(torch.tensor(0.50))

        state_anchor = self.BuildGroupedAnalysis(self.rank, self.state_dim)
        context_anchor = self.BuildGroupedAnalysis(self.rank, self.input_dim)
        self.register_buffer("left_anchor", state_anchor, persistent=False)
        self.register_buffer(
            "right_anchor",
            torch.roll(state_anchor, shifts=max(1, self.rank // 3), dims=0),
            persistent=False)
        self.register_buffer("context_anchor", context_anchor, persistent=False)
        self.register_buffer("out_anchor", state_anchor.t().contiguous(), persistent=False)

        nn.init.xavier_uniform_(self.left.target.weight)
        nn.init.xavier_uniform_(self.right.target.weight)
        nn.init.xavier_uniform_(self.context.target.weight)
        nn.init.xavier_uniform_(self.out.target.weight, gain=0.5)
        nn.init.zeros_(self.selectivity.target.weight)
        nn.init.zeros_(self.selectivity.target.bias)

    @staticmethod
    def BuildGroupedAnalysis(outDim: int, inDim: int) -> torch.Tensor:
        matrix = torch.zeros(int(outDim), int(inDim))
        columns = torch.arange(int(inDim))
        matrix[columns.remainder(int(outDim)), columns] = 1.0
        row_norm = matrix.square().sum(dim=1, keepdim=True).sqrt()
        return matrix / torch.where(
            row_norm > 0.0,
            row_norm,
            torch.ones_like(row_norm))

    @staticmethod
    def NormalizeWeight(weight: torch.Tensor) -> torch.Tensor:
        work_weight = (
            weight.float()
            if weight.dtype in (torch.float16, torch.bfloat16)
            else weight)
        normalized = work_weight / torch.sqrt(
            1.0 + work_weight.square().sum())
        return normalized.to(dtype=weight.dtype)

    def ProjectWeight(
        self,
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        anchor: torch.Tensor,
        ) -> torch.Tensor:
        learned = F.linear(value, self.NormalizeWeight(weight), bias)
        anchored = F.linear(value, anchor)
        return 0.5 * (anchored + learned)

    def Project(
        self,
        value: torch.Tensor,
        layer: GrowableLoRALinear,
        anchor: torch.Tensor,
        ) -> torch.Tensor:
        weight = layer.target.weight
        delta = layer.DeltaWeight()
        if delta is not None:
            weight = weight + delta
        return self.ProjectWeight(
            value, weight, layer.target.bias, anchor)

    def BoundedInputs(self, state: torch.Tensor, control: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tanh(state), torch.tanh(control)

    def Interaction(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        context: torch.Tensor,
        ) -> torch.Tensor:
        return torch.tanh(left) * torch.tanh(right + context)

    def Target(self, state: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        return self.target_scale * (
            torch.tanh(state) + 0.5 * torch.tanh(drive))

    def Mix(self, selectivity: torch.Tensor) -> torch.Tensor:
        gain = self.max_mix * torch.tanh(self.gain).square()
        return gain * torch.sigmoid(selectivity)

    def forward(
        self,
        state: torch.Tensor,
        control: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        bounded_state, bounded_control = self.BoundedInputs(state, control)
        interaction = self.Interaction(
            self.Project(bounded_state, self.left, self.left_anchor),
            self.Project(bounded_state, self.right, self.right_anchor),
            self.Project(bounded_control, self.context, self.context_anchor))
        return (
            self.Target(
                state,
                self.Project(interaction, self.out, self.out_anchor)),
            self.Mix(self.selectivity(bounded_control)))


class S4DCell(AGICoreModule):
    def __init__(self, inDim: int, deterDim: int, ssmDim: int = 512, dt: float = 1.0, dropout: float = 0.0, ffnMult: int = 4):
        super().__init__()
        self.in_dim = int(inDim)
        self.deter_dim = int(deterDim)
        self.ssm_dim = int(ssmDim)
        self.dt = float(dt)
        self.min_decay_rate = 0.005
        self.max_decay_rate = 1.0



        decay_rates = torch.exp(torch.linspace(
            torch.log(torch.tensor(0.01)),
            torch.log(torch.tensor(0.80)),
            steps=self.ssm_dim))
        decay_fraction = (
            (decay_rates - self.min_decay_rate)
            / (self.max_decay_rate - self.min_decay_rate))
        self.theta = nn.Parameter(torch.log(
            decay_fraction / (1.0 - decay_fraction)))

        self.in_to_ssm = GrowableLoRALinear(nn.Linear(self.in_dim, self.ssm_dim, bias=True))
        self.ssm_to_deter = GrowableLoRALinear(nn.Linear(self.ssm_dim, self.deter_dim, bias=True))
        self.in_to_deter = GrowableLoRALinear(nn.Linear(self.in_dim, self.deter_dim, bias=True))
        self.gate = GrowableLoRALinear(nn.Linear(self.in_dim, self.ssm_dim, bias=True))
        self.out_gate = GrowableLoRALinear(nn.Linear(self.ssm_dim, self.deter_dim, bias=True))
        self.nonlinear_flow = LowRankMultiplicativeFlow(
            stateDim=self.ssm_dim,
            inputDim=self.in_dim,
            rank=min(64, self.ssm_dim))

        self.ln_y = nn.LayerNorm(self.deter_dim)
        self.ln_ffn = nn.LayerNorm(self.deter_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.deter_dim, ffnMult * self.deter_dim, bias=True),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffnMult * self.deter_dim, self.deter_dim, bias=True),)

        self.register_buffer("x", torch.zeros(1, self.ssm_dim), persistent=True)

    def EnsureB(self, B: int):
        B = int(B)
        if self.x.size(0) != B:
            self.x = self.x.new_zeros(B, self.ssm_dim)

    def DecayRates(self, aDiag: torch.Tensor) -> torch.Tensor:
        return (
            self.min_decay_rate
            + (self.max_decay_rate - self.min_decay_rate) * torch.sigmoid(aDiag))

    def CayleyStep(self, aDiag: torch.Tensor, x: torch.Tensor, Bu: torch.Tensor, dt: float):
        A = -self.DecayRates(aDiag)
        k = 0.5 * dt * A
        num = (1 + k) * x + dt * Bu
        denom = (1 - k).clamp_min(1e-6)
        return num / denom

    def LinearStateTransition(
        self,
        x: torch.Tensor,
        linearTarget: torch.Tensor,
        ) -> torch.Tensor:


        decay = self.DecayRates(self.theta)
        return self.CayleyStep(
            self.theta,
            x,
            decay * linearTarget,
            self.dt)

    def StateTransition(
        self,
        linearState: torch.Tensor,
        nonlinearTarget: torch.Tensor,
        nonlinearMix: torch.Tensor,
        ) -> torch.Tensor:


        return (
            (1.0 - nonlinearMix) * linearState
            + nonlinearMix * nonlinearTarget)

    def ResetState(self, batch):
        self.x = torch.zeros(batch, self.ssm_dim, device=self.device, dtype=self.dtype)

    def Step(self, zPrev: torch.Tensor, action: torch.Tensor, *, updateState: bool = True) -> torch.Tensor:
        u = torch.cat([zPrev, action], dim=-1)
        g = torch.sigmoid(self.gate(u))
        linear_target = self.in_to_ssm(u) * g

        linear_state = self.LinearStateTransition(self.x, linear_target)
        nonlinear_target, nonlinear_mix = self.nonlinear_flow(linear_state, u)
        x_next = self.StateTransition(
            linear_state, nonlinear_target, nonlinear_mix)
        y_lin = self.ssm_to_deter(x_next) + self.in_to_deter(u)
        y_glu = y_lin * torch.sigmoid(self.out_gate(x_next))
        y = self.ln_y(y_glu)
        y = y + self.ffn(self.ln_ffn(y))

        if updateState:
            self.x = x_next.detach()
        return y # [B, D] deterministic state

    def StepWithX(self, zPrev: torch.Tensor, action: torch.Tensor, x: torch.Tensor):
        u = torch.cat([zPrev, action], dim=-1)
        g = torch.sigmoid(self.gate(u))
        linear_target = self.in_to_ssm(u) * g

        linear_state = self.LinearStateTransition(x, linear_target)
        nonlinear_target, nonlinear_mix = self.nonlinear_flow(linear_state, u)
        x_next = self.StateTransition(
            linear_state, nonlinear_target, nonlinear_mix)
        y_lin = self.ssm_to_deter(x_next) + self.in_to_deter(u)
        y_glu = y_lin * torch.sigmoid(self.out_gate(x_next))
        y = self.ln_y(y_glu)
        y = y + self.ffn(self.ln_ffn(y))

        return y, x_next.detach() # y: [B, D] deterministic state



class PhysRefinerHead(AGICoreModule):
    def __init__(
        self,
        deterDim: int,
        actDim: int,
        projDim: int = 256,
        hidden: int = 512,
        dt: float = 1.0,
        substeps: int = 2,
        lambdaWorkCons: float = 0.10,
        lambdaForceSmooth: float = 0.05,
        lambdaDelta: float = 0.01,
        clampResidualRatio: float = 0.50,
        dampP: float = 0.00, ):
        super().__init__()
        self.D = int(deterDim)
        self.A = int(actDim)
        self.P = int(projDim)
        assert self.P % 2 == 0, f"projDim must be even, got {self.P}"
        self.Q = self.P // 2

        self.dt = float(dt)
        self.substeps = int(max(1, substeps))
        self.clamp_ratio = float(clampResidualRatio)
        self.dampP = float(dampP)

        self.l_work = float(lambdaWorkCons)
        self.l_smooth = float(lambdaForceSmooth)
        self.l_delta = float(lambdaDelta)

        self.to_qp = GrowableLoRALinear(nn.Linear(self.D, self.P, bias=True))
        self.from_qp = GrowableLoRALinear(nn.Linear(self.P, self.D, bias=True))

        self.HNet = nn.Sequential(
            GrowableLoRALinear(nn.Linear(self.P, hidden, bias=True)),
            nn.Softplus(),
            GrowableLoRALinear(nn.Linear(hidden, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, 1, bias=True)),)

        self.force_net = nn.Sequential(
            GrowableLoRALinear(nn.Linear(self.D + self.A, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.Q, bias=True)),)

        self.g_force = GrowableLoRALinear(nn.Linear(self.D + self.A, self.Q, bias=True))
        self.g_phys  = GrowableLoRALinear(nn.Linear(self.D + self.A, self.D, bias=True))

        self.g_fuse = GrowableLoRALinear(nn.Linear(self.D + self.A + self.D, self.D, bias=True))

    def HAndGrad(self, qp: torch.Tensor, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.HNet(qp)
        g = torch.autograd.grad(
            H.sum(), qp,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,)[0] # [B,P]
        return H, g

    def SymplecticLeapfrog(self, q: torch.Tensor, p: torch.Tensor, dt: float, create_graph: bool
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qp0 = torch.cat([q, p], dim=-1)
        H0, g0 = self.HAndGrad(qp0, create_graph=create_graph)
        dH_dq0, _ = g0.chunk(2, dim=-1)

        p_half = p - 0.5 * dt * dH_dq0

        qp_mid = torch.cat([q, p_half], dim=-1)
        _, gm = self.HAndGrad(qp_mid, create_graph=create_graph)
        _, dH_dp_mid = gm.chunk(2, dim=-1)

        q1 = q + dt * dH_dp_mid

        qp_for_p = torch.cat([q1, p_half], dim=-1)
        H1, g2 = self.HAndGrad(qp_for_p, create_graph=create_graph)
        dH_dq2, _ = g2.chunk(2, dim=-1)

        p1 = p_half - 0.5 * dt * dH_dq2
        return q1, p1, H0, H1, dH_dp_mid

    def ClampResidual(self, delta: torch.Tensor, base: torch.Tensor, ratio: float) -> torch.Tensor:
        eps = 1e-8
        dnorm = delta.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        bnorm = base.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-3
        maxn = ratio * bnorm + eps
        scale = (maxn / dnorm).clamp(max=1.0)
        return delta * scale

    def forward(
        self,
        hPrev: torch.Tensor,
        action: torch.Tensor,
        hS4: torch.Tensor,):

        training_mode = bool(self.training and torch.is_grad_enabled())
        inference_mode_active = torch.is_inference_mode_enabled()

        create_graph = bool(training_mode)

        dt_sub = self.dt / float(self.substeps)

        if training_mode:
            H_start = None
            H_end = None
            work_acc = hPrev.new_zeros(hPrev.size(0), 1) # [B,1]
            smooth_acc = hPrev.new_tensor(0.0)

        with torch.inference_mode(False), torch.enable_grad():
            h_prev_work = hPrev.detach().clone() if inference_mode_active else hPrev
            action_work = action.detach().clone() if inference_mode_active else action
            qp = self.to_qp(h_prev_work)
            if not qp.requires_grad:
                qp = qp.detach().requires_grad_(True)
            q, p = qp.chunk(2, dim=-1)

            for i in range(self.substeps):
                h_cur = self.from_qp(torch.cat([q, p], dim=-1))

                fa0_inp = torch.cat([h_cur, action_work], dim=-1)
                F0 = self.force_net(fa0_inp) * torch.sigmoid(self.g_force(fa0_inp)) # [B,Q]

                if self.dampP > 0.0:
                    p = p * p.new_tensor(-self.dampP * dt_sub).exp()

                p = p + 0.5 * dt_sub * F0

                q, p, H0, H1, dH_dp_mid = self.SymplecticLeapfrog(q, p, dt_sub, create_graph=create_graph)

                if training_mode:
                    if i == 0:
                        H_start = H0
                    H_end = H1

                h_mid = self.from_qp(torch.cat([q, p], dim=-1))
                fa1_inp = torch.cat([h_mid, action_work], dim=-1)
                F1 = self.force_net(fa1_inp) * torch.sigmoid(self.g_force(fa1_inp)) # [B,Q]

                p = p + 0.5 * dt_sub * F1

                if training_mode:
                    smooth_acc = smooth_acc + (F1 - F0).pow(2).mean()
                    F_avg = 0.5 * (F0 + F1)
                    work_acc = work_acc + (dH_dp_mid * F_avg).sum(dim=-1, keepdim=True) * dt_sub

        h_phys_raw = self.from_qp(torch.cat([q, p], dim=-1))

        d_corr = h_phys_raw - hS4 # [B,D]

        gph = torch.sigmoid(self.g_phys(torch.cat([hPrev, action], dim=-1))) # [B,D]
        d_corr = d_corr * gph

        base = hS4 - hPrev # [B,D]
        d_corr = self.ClampResidual(d_corr, base, ratio=self.clamp_ratio)

        alpha = torch.sigmoid(self.g_fuse(torch.cat([hPrev, action, hS4], dim=-1))) # [B,D]
        h_fused = hS4 + alpha * d_corr

        if not training_mode:
            return h_fused, None, None

        if (H_start is None) or (H_end is None):
            e_work = hPrev.new_tensor(0.0)
        else:
            denom = H_start.detach().abs().mean().clamp_min(1e-6)
            dH = (H_end - H_start)
            e_work = ((dH - work_acc) / denom).pow(2).mean()

        e_smooth = smooth_acc / float(self.substeps)
        e_delta = d_corr.pow(2).mean()

        loss = (
            self.l_work * e_work
            + self.l_smooth * e_smooth
            + self.l_delta * e_delta)

        aux: Dict[str, torch.Tensor] = {}
        aux = {
            "L_work": e_work.detach(),
            "L_smooth": e_smooth.detach(),
            "L_delta": e_delta.detach(),}

        return h_fused, loss, aux


class NeSyHead(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        K: int,
        hidden: int = 1024,
        experts: int = 8,
        dropout: float = 0.1,
        *,
        temperature: float = 1.0,
        noisyGating: bool = True,
        noiseStd: float = 0.1,
        expertDropout: float = 0.0,):
        super().__init__()
        self.K = int(K)
        self.E = int(experts)

        self.temperature = float(max(1e-6, temperature))
        self.noisyGating = bool(noisyGating)
        self.noiseStd = float(noiseStd)
        self.expertDropout = float(expertDropout)

        self.input_ln = nn.LayerNorm(inDim)

        self.gate = nn.Linear(inDim, self.E)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.K),)for _ in range(self.E)])

        self.out_scale_log = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

    def GetAuxLoss(self) -> torch.Tensor:
        return self.aux_loss

    def GateWeights(
        self,
        x_aligned: torch.Tensor,
        *,
        deterministic: bool = False,
        updateAux: bool = True,
        sampleMask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        logits = self.gate(x_aligned) # [B,E]

        if self.training and not deterministic and self.noisyGating and (self.noiseStd > 0.0):
            logits = logits + torch.randn_like(logits) * self.noiseStd

        if self.training and not deterministic and (self.expertDropout > 0.0):
            keep = (torch.rand_like(logits) > self.expertDropout)
            all_drop = (~keep).all(dim=-1)
            if all_drop.any():
                rand_idx = torch.randint(0, self.E, (int(all_drop.sum().item()),), device=self.device)
                keep[all_drop] = False
                keep[all_drop, rand_idx] = True
            logits = logits.masked_fill(
                ~keep,
                torch.finfo(logits.dtype).min)

        w = F.softmax((logits / self.temperature).float(), dim=-1) # [B,E]

        if updateAux and self.training and not deterministic:
            if sampleMask is None:
                importance = w.mean(dim=0) # [E]
                self.aux_loss = float(self.E) * (importance.pow(2).sum())
            else:
                if (
                    not torch.is_tensor(sampleMask)
                    or tuple(sampleMask.shape) != (x_aligned.size(0),)
                    or sampleMask.device != x_aligned.device
                    or sampleMask.dtype != torch.bool
                ):
                    raise ValueError("sampleMask must be a batched boolean mask")
                weight = sampleMask.to(dtype=w.dtype).unsqueeze(-1)
                importance = (w * weight).sum(dim=0) / weight.sum().clamp_min(1.0)
                self.aux_loss = float(self.E) * (importance.pow(2).sum())
        elif updateAux:
            self.aux_loss = x_aligned.new_zeros(())

        return w

    def forward(
        self,
        x: torch.Tensor,
        *,
        deterministic: bool = False,
        updateAux: bool = True,
        sampleMask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x_aligned = self.input_ln(x) # [B,inDim]
        w = self.GateWeights(
            x_aligned,
            deterministic=deterministic,
            updateAux=updateAux,
            sampleMask=sampleMask) # [B,E]

        if deterministic:
            expert_outputs = []
            for expert in self.experts:
                hidden = expert[0](x_aligned)
                hidden = expert[1](hidden)
                hidden = F.dropout(hidden, p=float(expert[2].p), training=False)
                hidden = expert[3](hidden)
                hidden = expert[4](hidden)
                hidden = F.dropout(hidden, p=float(expert[5].p), training=False)
                expert_outputs.append(expert[6](hidden))
            expert_logits = torch.stack(expert_outputs, dim=1)
        else:
            expert_logits = torch.stack([e(x_aligned) for e in self.experts], dim=1)

        out_logits = (w.unsqueeze(-1) * expert_logits).sum(dim=1) # [B,K]

        scale = torch.exp(self.out_scale_log).clamp(1e-3, 100.0)
        out_logits = out_logits * scale

        return out_logits # [B,K]



class GeometricLinear(AGICoreModule):
    def __init__(self, inFeatures, outFeatures, wrapLinear=None, gain=0.1):
        super().__init__()
        lin = nn.Linear(inFeatures, outFeatures, bias=True)
        nn.init.orthogonal_(lin.weight, gain=gain)
        nn.init.zeros_(lin.bias)
        self.linear = wrapLinear(lin) if wrapLinear is not None else lin

    def forward(self, x): return self.linear(x)

class FilmResidual(AGICoreModule):
    def __init__(self, hidden, alpha=0.1, wrapLinear=None):
        super().__init__()
        self.alpha = float(alpha)
        self.ln = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            GeometricLinear(hidden, hidden, wrapLinear),
            nn.GELU(),
            nn.LayerNorm(hidden),
            GeometricLinear(hidden, hidden, wrapLinear),)

    def forward(self, h, gx, bx):
        y = (1.0 + gx) * h + bx
        y = self.ln(y)
        y = self.ff(y)
        return h + self.alpha * y

class ConnNet(AGICoreModule):
    def __init__(self,
        stateDim: int,
        actDim: int,
        *,
        hidden: int = 512,
        numBlocks: int = 3,
        rank: int = 8,
        useFull: bool = True,
        useLowrank: bool = True,
        dt: float = 1.0,
        lambdaFro: float = 1e-4,
        lambdaL1: float = 1e-5,
        lambdaSmooth: float = 3e-5,
        normClip: float = 0.8,
        wrapLinear=None):
        super().__init__()
        self.S = int(stateDim)
        self.A = int(actDim)
        self.H = int(hidden)
        self.r = int(rank)
        self.use_full = bool(useFull)
        self.use_lowrank = bool(useLowrank)
        self.dt = float(dt)
        self.lambda_fro = float(lambdaFro)
        self.lambda_l1 = float(lambdaL1)
        self.lambda_smooth = float(lambdaSmooth)
        self.norm_clip = float(normClip)

        self.enc_s = nn.Sequential(
            nn.LayerNorm(self.S),
            GeometricLinear(self.S, self.H, wrapLinear),
            nn.GELU(),)

        self.enc_a = nn.Sequential(
            nn.LayerNorm(self.A),
            GeometricLinear(self.A, self.H, wrapLinear),
            nn.GELU(),)

        self.film_gamma_a = GeometricLinear(self.H, self.H, wrapLinear)
        self.film_beta_a = GeometricLinear(self.H, self.H, wrapLinear)

        self.blocks = nn.ModuleList([FilmResidual(self.H, alpha=0.1, wrapLinear=wrapLinear) for _ in range(numBlocks)])

        self.head_uv = GeometricLinear(self.H, 2 * self.S * self.r, wrapLinear)
        self.head_full = GeometricLinear(self.H, self.S * self.S, wrapLinear)
        nBranches = int(self.use_lowrank) + int(self.use_full)
        self.mix = GeometricLinear(self.H, max(1, nBranches), wrapLinear)

    def BuildLowrank(self, h):
        uv = self.head_uv(h)
        U, V = uv.split(self.S * self.r, dim=-1)
        U = U.view(-1, self.S, self.r)
        V = V.view(-1, self.S, self.r)
        return U @ V.transpose(1, 2) - V @ U.transpose(1, 2)

    def BuildFull(self, h):
        M = self.head_full(h).view(-1, self.S, self.S)
        return 0.5 * (M - M.transpose(1, 2))

    def TransportApply(self, A: torch.Tensor, sBase: torch.Tensor) -> torch.Tensor:
        B, S = A.size(0), A.size(1)
        dt = self.dt

        I = torch.eye(S, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, S, S)
        lhs = I - 0.5 * dt * A
        rhs_vec = torch.einsum("bij,bj->bi", I + 0.5 * dt * A, sBase)

        lhs = lhs.contiguous()
        rhs = rhs_vec.unsqueeze(-1).contiguous()

        solve_dtype = (
            torch.float32
            if lhs.device.type == "cpu" and lhs.dtype in (torch.float16, torch.bfloat16)
            else lhs.dtype)
        cayley = torch.linalg.solve(
            lhs.to(dtype=solve_dtype),
            rhs.to(dtype=solve_dtype)).squeeze(-1).to(dtype=sBase.dtype)

        return cayley # [B, S]


    def ComputeGeomReg(self, A, prevA=None, sampleMask=None):
        def BatchMean(value: torch.Tensor) -> torch.Tensor:
            if sampleMask is None:
                return value.mean()
            weight = sampleMask.to(dtype=value.dtype)
            per_row = value.reshape(value.size(0), -1).mean(dim=-1)
            return (per_row * weight).sum() / weight.sum().clamp_min(1.0)

        reg = self.lambda_fro * BatchMean(A.pow(2))
        if self.use_full and self.lambda_l1 > 0:
            reg = reg + self.lambda_l1 * BatchMean(A.abs())
        if (prevA is not None) and (self.lambda_smooth > 0):
            reg = reg + self.lambda_smooth * BatchMean(
                (A - prevA).pow(2))
        return reg

    def forward(self, sBase: torch.Tensor, actPrev: torch.Tensor) -> torch.Tensor:
        B = sBase.size(0)
        hs = self.enc_s(sBase)
        ha = self.enc_a(actPrev)

        g = torch.tanh(self.film_gamma_a(ha))
        b = self.film_beta_a(ha)

        h = hs
        for blk in self.blocks:
            h = blk(h, g, b)

        A_list = []
        if self.use_lowrank:
            A_list.append(self.BuildLowrank(h))
        if self.use_full:
            A_list.append(self.BuildFull(h))

        if not A_list:
            A = torch.zeros(B, self.S, self.S, device=sBase.device, dtype=sBase.dtype)
        elif len(A_list) == 1:
            A = A_list[0]
        else:
            w = F.softmax(self.mix(h), dim=-1)
            A = w[:, :1].view(B, 1, 1) * A_list[0] + w[:, 1:2].view(B, 1, 1) * A_list[1]

        if self.norm_clip and self.norm_clip > 0:
            fro = A.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), self.norm_clip / fro).view(B, 1, 1)
            A = A * scale
        return A



class SoftNeSyStructure(AGICoreModule):
    def __init__(self, k: int, gExcl: int = 8, gAlo: int = 8, tauInit: float = 1.0, lambdaDag: float = 1e-3):
        super().__init__()
        self.K = int(k)
        self.Ge = int(gExcl)
        self.Ga = int(gAlo)
        self.lambda_dag = float(lambdaDag)
        self.tau = nn.Parameter(torch.tensor(float(tauInit)))
        self.M_excl = nn.Parameter(torch.randn(self.Ge, self.K) * 0.01)
        self.M_alo = nn.Parameter(torch.randn(self.Ga, self.K) * 0.01)
        self.E = nn.Parameter(torch.zeros(self.K, self.K))
        self.register_buffer("_eye", torch.eye(self.K))

    def MixExclusive(self, P: torch.Tensor, temp: float) -> torch.Tensor:
        eps = 1e-6
        Wg = F.softmax(self.M_excl, dim=-1) # [Ge, K]
        logP = torch.log(P.clamp(eps, 1 - eps)) / max(1e-6, temp) # [B, K]
        g = logP.unsqueeze(1) + torch.log(Wg.unsqueeze(0).clamp(eps)) # [B, Ge, K]
        g_sm = F.softmax(g, dim=-1) # [B, Ge, K]
        Wk = F.softmax(self.M_excl.t(), dim=-1)
        P_new = torch.einsum("bgk,kg->bk", g_sm, Wk)
        return P_new # [B, K]

    def EnforceAlo(self, P: torch.Tensor, tau: float) -> torch.Tensor:
        eps = 1e-6
        Wa = F.softmax(self.M_alo, dim=-1)
        group_vals = (P.unsqueeze(1) * Wa.unsqueeze(0)).max(dim=-1).values
        scale = torch.where(group_vals < tau, tau / (group_vals + eps), torch.ones_like(group_vals)) # [B, Ga]
        P_scaled = P.clone()
        Wk = F.softmax(self.M_alo.t(), dim=-1)
        s = torch.einsum("bg,kg->bk", scale, Wk)
        P_scaled = P_scaled * s
        return P_scaled # [B, K]

    def ApplyImplications(self, P: torch.Tensor, alpha: float) -> torch.Tensor:
        eps = 1e-6
        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        contrib = P.unsqueeze(2) * W.unsqueeze(0) # [B, K, K]
        implied = contrib.max(dim=1).values # [B, K]
        Q = torch.maximum(P, alpha * implied).clamp(eps, 1 - eps)
        return Q # [B, K]

    def ProjectTrain(self, P: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        eps = 1e-6
        t = float(max(1e-3, temp))

        soft = max(1e-3, 0.25 * t)

        P1 = self.MixExclusive(P, temp=t).clamp(eps, 1.0 - eps) # [B,K]

        aloTau = P1.new_tensor(0.60)
        Wa = F.softmax(self.M_alo, dim=-1) # [Ga,K]

        v = (P1.unsqueeze(1) * Wa.unsqueeze(0)) # [B,Ga,K]
        attn = F.softmax(v / soft, dim=-1) # [B,Ga,K]
        group_vals = (attn * v).sum(dim=-1).clamp_min(eps) # [B,Ga]

        deficiency = F.softplus((aloTau - group_vals) / soft) * soft # [B,Ga]

        scale = 1.0 + deficiency / group_vals # [B,Ga]
        scale = scale.clamp(1.0, 10.0)

        Wk = F.softmax(self.M_alo.t(), dim=-1) # [K,Ga]
        s = torch.einsum("bg,kg->bk", scale, Wk) # [B,K]
        P2 = (P1 * s).clamp(eps, 1.0 - eps) # [B,K]

        implAlpha = P2.new_tensor(1.0)
        W = torch.sigmoid(self.E) * (1.0 - self._eye) # [K,K]

        contrib = P2.unsqueeze(2) * W.unsqueeze(0) # [B,K,K]
        w_imp = F.softmax(contrib / soft, dim=1)
        implied = (w_imp * contrib).sum(dim=1) # [B,K]

        b = (implAlpha * implied).clamp(eps, 1.0 - eps) # [B,K]

        smax = torch.sigmoid((b - P2) / soft) # [B,K]
        Q = (1.0 - smax) * P2 + smax * b
        Q = Q.clamp(eps, 1.0 - eps)

        return Q # [B,K]

    @torch.no_grad()
    def ProjectRuntime(self, P: torch.Tensor, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        Q = self.MixExclusive(P, temp)
        Q = self.EnforceAlo(Q, aloTau)
        Q = self.ApplyImplications(Q, implAlpha)

        Ge = F.softmax(self.M_excl, dim=-1) # [Ge, K]
        gprob = F.softmax((torch.log(Q)).unsqueeze(1) + torch.log(Ge.unsqueeze(0)), dim=-1) # [B, Ge, K]
        excl_pen = 0.5 * ((gprob.sum(-1) ** 2) - (gprob ** 2).sum(-1)).mean(dim=-1)

        Ga = F.softmax(self.M_alo, dim=-1) # [Ga, K]
        alo_val = (Q.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values # [B, Ga]
        alo_pen = F.relu(aloTau - alo_val).mean(dim=-1) # [B]

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl_pen = (W.unsqueeze(0) * F.relu(Q.unsqueeze(2) - Q.unsqueeze(1))).mean(dim=(1,2))

        pen = excl_pen + alo_pen + impl_pen
        pen = (1.0 - torch.exp(-pen)).clamp(0.0, 1.0)
        return Q.clamp(1e-6, 1.0 - 1e-6), pen

    def LogicLosses(self, P: torch.Tensor, lambdaExcl: float, lambdaAlo: float, lambdaImpl: float, aloTau: float = 0.60):
        Ge = F.softmax(self.M_excl, dim=-1)
        g = (torch.log(P.clamp(1e-6, 1-1e-6))).unsqueeze(1) + torch.log(Ge.unsqueeze(0).clamp(1e-6)) # [B, Ge, K]
        g_sm = F.softmax(g, dim=-1) # [B, Ge, K]
        excl = 0.5 * ((g_sm.sum(-1)**2) - (g_sm**2).sum(-1)) # [B, Ge]
        excl = excl.mean()

        Ga = F.softmax(self.M_alo, dim=-1) # [Ga, K]
        top1 = (P.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values # [B, Ga]
        aloTau_t = top1.new_tensor(float(aloTau))
        alo = (F.relu(aloTau_t - top1) ** 2).mean()

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl = (W.unsqueeze(0) * F.relu(P.unsqueeze(2) - P.unsqueeze(1))).mean()

        loss = lambdaExcl * excl + lambdaAlo * alo + lambdaImpl * impl

        reg = 1e-4 * W.mean()

        Ge_sm = F.softmax(self.M_excl, dim=-1).clamp_min(1e-6)
        Ga_sm = F.softmax(self.M_alo,  dim=-1).clamp_min(1e-6)
        reg = reg + 1e-3 * (
            (Ge_sm * torch.log(Ge_sm)).sum() / float(self.Ge) +
            (Ga_sm * torch.log(Ga_sm)).sum() / float(self.Ga))

        A = (W * W) / float(self.K) # [K,K]
        dag = torch.trace(torch.matrix_exp(A.float())) - float(self.K)
        dag = dag.to(dtype=P.dtype, device=P.device)
        reg = reg + self.lambda_dag * dag

        loss = loss + reg
        stats = {"excl": excl.detach(), "alo": alo.detach(), "impl": impl.detach()}
        return loss, stats


class FiLMHResidual(AGICoreModule):
    def __init__(
        self,
        baseDim: int,
        rediusDim: int,
        hidden: int = 512,
        dropout: float = 0.1,
        filmScale: float = 0.10,
        outLayerNorm: bool = True,):
        super().__init__()
        self.D = int(baseDim)
        self.Z = int(rediusDim)
        self.H = int(hidden)
        self.film_scale = float(filmScale)
        self.use_out_ln = bool(outLayerNorm)

        self.ln_h = nn.LayerNorm(self.D)
        self.ln_e = nn.LayerNorm(self.Z)

        self.e_to_gb = GrowableLoRALinear(nn.Linear(self.Z, 2 * self.D, bias=True))

        self.e_to_h = GrowableLoRALinear(nn.Linear(self.Z, self.D, bias=True))

        self.delta_ln = nn.LayerNorm(4 * self.D)
        self.delta_mlp = nn.Sequential(
            GrowableLoRALinear(nn.Linear(4 * self.D, self.H, bias=True)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            GrowableLoRALinear(nn.Linear(self.H, self.D, bias=True)),)

        self.to_gate = GrowableLoRALinear(nn.Linear(2 * self.D, self.D, bias=True))

        self.out_ln = nn.LayerNorm(self.D)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h0 = self.ln_h(h) # [B,D]
        e0 = self.ln_e(e) # [B,Z]

        gamma, beta = self.e_to_gb(e0).chunk(2, dim=-1) # [B,D],[B,D]
        gamma = self.film_scale * torch.tanh(gamma) # [B,D]
        beta  = self.film_scale * torch.tanh(beta) # [B,D]

        h_film = (1.0 + gamma) * h0 + beta # [B,D]

        e_h = self.e_to_h(e0) # [B,D]
        e_h = self.film_scale * torch.tanh(e_h) # [B,D]

        feat = torch.cat([h_film, e_h, h_film * e_h, h_film - e_h], dim=-1) # [B,4D]
        feat = self.delta_ln(feat) # [B,4D]
        delta = self.delta_mlp(feat) # [B,D]

        gate_in = torch.cat([h_film, e_h], dim=-1) # [B,2D]
        gate = torch.sigmoid(self.to_gate(gate_in)) # [B,D]

        h_out = h + gate * delta # [B,D]

        if self.use_out_ln:
            h_out = self.out_ln(h_out) # [B,D]
        return h_out

class KeyEmbed(AGICoreModule):
    def __init__(self, Z: int, keyDim: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.Z = int(Z)
        self.keyDim = int(keyDim)
        H = int(hidden)

        self.ln_e = nn.LayerNorm(self.Z)
        self.ln_a = nn.LayerNorm(self.Z)

        self.to_gb = GrowableLoRALinear(nn.Linear(self.Z, 2 * self.Z, bias=True))

        self.ln_feat = nn.LayerNorm(4 * self.Z)
        self.mlp1 = GrowableLoRALinear(nn.Linear(4 * self.Z, H, bias=True))
        self.mlp2 = GrowableLoRALinear(nn.Linear(H, self.keyDim, bias=True))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, base: torch.Tensor, actionEmbed: torch.Tensor) -> torch.Tensor:
        e = self.ln_e(base) # [B,Z]
        a = self.ln_a(actionEmbed) # [B,Z]

        gamma, beta = self.to_gb(a).chunk(2, dim=-1) # [B,Z],[B,Z]
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)

        e_film = (1.0 + gamma) * e + beta # [B,Z]

        feat = torch.cat([e_film, a, e_film * a, e_film - a], dim=-1) # [B,4Z]
        feat = self.ln_feat(feat)

        h = F.silu(self.mlp1(feat)) # [B,H]
        h = self.drop(h)
        k = self.mlp2(h) # [B,keyDim]

        k = F.normalize(k, dim=-1, eps=1e-6) # [B,keyDim]
        return k




class RSSMWorldModel(AGICoreModule):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        visionDim: int = ModuleDim.PerceptionFeat,
        actionDim: int = ModuleDim.DecisionActionFeatureDim,
        deterDim: int = 512,
        stochDim: int = 64,
        stateDim: int = 512,
        ssmDim: int = 512,
        useDecoder: bool = True,
        useMemory: bool = True,
        memoryCapacity: int = 16384,
        memoryPath: Optional[str] = None,
        memoryAutosaveEvery: int = 0,
        nsEnabled: bool = True,
        nsLambdaExclusive: float = 1e-2,
        nsLambdaAtLeastOne: float = 1e-2,
        nsLambdaImplication: float = 1e-2,
        memTopK = 4,
        memTemp: float = 1.0,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        integratedFeatDim: int = 1024,
        physicalSlots: int = ModuleDim.PstSlots,
        physicalSlotDim: int = ModuleDim.PstSlotDim,
        spatialFrameDim: int = ModuleDim.PstPoseDim,
        physicalAttrDim: int = ModuleDim.PstAttrDim,
        physicalIdDim: int = ModuleDim.PstIdDim,
        physicalRelDim: int = ModuleDim.PstRelDim,
        physicalRelationClasses: int = ModuleDim.PstRelationClasses,
        physicalSemanticDim: int = ModuleDim.PstSemanticDim,
        physicalStateDim: int = ModuleDim.PstStateDim,
        physicalAffordanceDim: int = ModuleDim.PstAffordanceDim,
        physicalTextDim: int = ModuleDim.PstTextDim,
        physicalSymbolDim: int = ModuleDim.PstSymbolClasses,
        physicalObservationThreshold: float = 0.5,
        physicalIdentityThreshold: float = 0.75,
        physicalConfidenceDecay: float = 0.995,):
        super().__init__()
        self.vision_dim = visionDim
        self.action_dim = actionDim
        self.deter_dim = deterDim
        self.stoch_dim = stochDim
        self.state_dim = stateDim
        self.use_decoder = useDecoder
        self.ssm_dim = ssmDim
        self.reward_min = -10.0
        self.reward_max = 10.0

        self._mem_topk: int = int(memTopK)
        self._mem_temp: float = float(memTemp)
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.integrated_feat_dim = int(integratedFeatDim)
        self.physical_slots = int(physicalSlots)
        self.physical_slot_dim = int(physicalSlotDim)
        self.physical_spatial_dim = int(spatialFrameDim)
        self.physical_attr_dim = int(physicalAttrDim)
        self.physical_id_dim = int(physicalIdDim)
        self.physical_rel_dim = int(physicalRelDim)
        self.physical_relation_classes = int(physicalRelationClasses)
        self.physical_semantic_dim = int(physicalSemanticDim)
        self.physical_state_dim = int(physicalStateDim)
        self.physical_affordance_dim = int(physicalAffordanceDim)
        self.physical_text_dim = int(physicalTextDim)
        self.entity_text_semantic_dim = 512
        self.physical_symbol_dim = int(physicalSymbolDim)
        self.physical_observation_threshold = float(physicalObservationThreshold)
        self.physical_identity_threshold = float(physicalIdentityThreshold)
        self.physical_confidence_decay = float(physicalConfidenceDecay)
        self.self_part_count = int(contractView.end_effector_count)
        self.self_part_semantic_dim = int(
            ModuleDim.PstSelfPartSemanticDim)
        self.embodiment_state_dim = ModuleDim.PstSlotDim
        self.embodiment_context_dim = ModuleDim.PstSlotDim
        self.observer_valid = bool(len(contractView.perception_view_indices) > 0)
        self.contract_embodiment_adapter = ContractWorldEmbodimentAdapter(
            contractView,
            self.embodiment_state_dim,
            self.action_dim)
        entity_bank_input_dim = (
            self.physical_slot_dim
            + ModuleDim.PstRealmClasses
            + ModuleDim.PstMotionLayerClasses
            + ModuleDim.PstAgencyClasses
            + 1
            + self.self_part_semantic_dim
            + 2
            + 2
            + 1
            + 1
            + 2
            + 1
            + 1
            + self.entity_text_semantic_dim
            + 3)
        self.entity_conscious_encoder = nn.Sequential(
            nn.LayerNorm(entity_bank_input_dim),
            nn.Linear(entity_bank_input_dim, self.state_dim * 2),
            nn.SiLU(),
            nn.Linear(self.state_dim * 2, self.state_dim),
            nn.LayerNorm(self.state_dim),)

        self._A_prev = None
        self._A_prev_valid = None

        self.obs_enc = nn.Sequential(
            nn.LayerNorm(visionDim),
            GrowableLoRALinear(nn.Linear(visionDim, stateDim, bias=True)),
            nn.GELU(),
            nn.LayerNorm(stateDim),
            GrowableLoRALinear(nn.Linear(stateDim, stochDim, bias=True)),)

        self.act_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(actionDim, stochDim, bias=True)),
            nn.LayerNorm(stochDim),
            nn.Tanh(),)

        self.s4 = S4DCell(inDim=stochDim + stochDim, deterDim=deterDim, ssmDim=self.ssm_dim, dt=1.0)

        self.prior_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim, 2 * stochDim, bias=True)))

        self.post_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim + stochDim, 2 * stochDim, bias=True)))

        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + stochDim),
            GrowableLoRALinear(nn.Linear(deterDim + stochDim, stateDim, bias=True)),
            nn.LayerNorm(stateDim),)

        self.rdone_ln = nn.LayerNorm(2 * stateDim + stochDim)

        self.rdone_trunk = nn.Sequential(
            GrowableLoRALinear(nn.Linear(2 * stateDim + stochDim, 512, bias=True)),
            nn.SiLU(),
            nn.Dropout(0.1),
            GrowableLoRALinear(nn.Linear(512, 256, bias=True)),
            nn.SiLU(),)

        self.rew_head = nn.Sequential(GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)

        self.done_head = nn.Sequential(GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)

        self.information_gain_head = nn.Sequential(
            nn.LayerNorm(256),
            GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)
        information_gain_context_dim = (
            deterDim
            + stochDim
            + stochDim
            + self.embodiment_context_dim)
        self.information_gain_context = nn.Sequential(
            nn.LayerNorm(information_gain_context_dim),
            GrowableLoRALinear(nn.Linear(
                information_gain_context_dim,
                256,
                bias=True)),
            nn.SiLU(),)

        self.obs_dec = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, stateDim, bias=True)),
            nn.GELU(),
            GrowableLoRALinear(nn.Linear(stateDim, visionDim, bias=True)),)

        self._use_memory = bool(useMemory)
        self._mem_capacity = int(memoryCapacity)
        self._mem_path = memoryPath
        self._mem_autosave_every = int(memoryAutosaveEvery)
        self._mem_add_count = 0
        self._mem_autosave_pending = False
        self._memory_calibration_id: Optional[str] = None
        self._memory_world_frame_id: Optional[str] = None

        self.register_buffer("_mem_keys", torch.zeros(1, self._mem_capacity, stochDim))
        self.register_buffer("_mem_vals", torch.zeros(1, self._mem_capacity, stateDim))
        self.register_buffer("_mem_size", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_mem_imp", torch.zeros(1, self._mem_capacity))
        self.register_buffer("_mem_steps", torch.zeros(1, self._mem_capacity, dtype=torch.long))
        self.register_buffer("_mem_global_step", torch.zeros(1, dtype=torch.long))

        self.register_buffer("_pst_slot_state", torch.zeros(1, self.physical_slots, self.physical_slot_dim))
        self.register_buffer("_pst_spatial_world", torch.zeros(1, self.physical_slots, self.physical_spatial_dim))
        self.register_buffer("_pst_attribute", torch.zeros(1, self.physical_slots, self.physical_attr_dim))
        self.register_buffer("_pst_slot_presence", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_entity_prob", torch.zeros(1, self.physical_slots))
        self.register_buffer(
            "_pst_entity_id",
            torch.full((1, self.physical_slots), -1, dtype=torch.long))
        self.register_buffer(
            "_pst_slot_generation",
            torch.zeros(1, self.physical_slots, dtype=torch.long))
        self.register_buffer(
            "_pst_next_entity_id",
            torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            "_last_observed_to_world_slot",
            torch.full((1, 0), -1, dtype=torch.long))
        self.register_buffer("_pst_perceptual_presence", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_geometry_valid", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_physical_interaction", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_realm", torch.zeros(1, self.physical_slots, ModuleDim.PstRealmClasses))
        self.register_buffer("_pst_motion_layer", torch.zeros(1, self.physical_slots, ModuleDim.PstMotionLayerClasses))
        self.register_buffer("_pst_layer_agency", torch.zeros(1, self.physical_slots, ModuleDim.PstMotionLayerClasses, ModuleDim.PstAgencyClasses))
        self.register_buffer("_pst_agency", torch.zeros(1, self.physical_slots, ModuleDim.PstAgencyClasses))
        self.register_buffer("_pst_body_membership", torch.zeros(1, self.physical_slots))
        self.register_buffer(
            "_pst_self_part",
            torch.zeros(1, self.physical_slots, self.self_part_count))
        self.register_buffer(
            "_pst_self_part_semantic",
            torch.zeros(
                1,
                self.physical_slots,
                self.self_part_semantic_dim))
        self.register_buffer("_pst_identity_key", torch.zeros(1, self.physical_slots, self.physical_id_dim))
        self.register_buffer("_pst_pairwise_relation", torch.zeros(1, self.physical_slots, self.physical_slots, self.physical_rel_dim))
        self.register_buffer(
            "_pst_pair_last_seen",
            torch.zeros(1, self.physical_slots, self.physical_slots, dtype=torch.long))
        self.register_buffer("_pst_external_relation", torch.zeros(1, self.physical_slots, self.physical_relation_classes))
        self.register_buffer("_pst_semantic", torch.zeros(1, self.physical_slots, self.physical_semantic_dim))
        self.register_buffer("_pst_size", torch.zeros(1, self.physical_slots, 3))
        self.register_buffer("_pst_state", torch.zeros(1, self.physical_slots, self.physical_state_dim))
        self.register_buffer("_pst_affordance", torch.zeros(1, self.physical_slots, self.physical_affordance_dim))
        self.register_buffer("_pst_motion", torch.zeros(1, self.physical_slots, self.physical_spatial_dim))
        self.register_buffer("_pst_carrier_motion", torch.zeros(1, self.physical_slots, self.physical_spatial_dim))
        self.register_buffer("_pst_articulation_motion", torch.zeros(1, self.physical_slots, self.physical_spatial_dim))
        self.register_buffer("_pst_content_motion", torch.zeros(1, self.physical_slots, 2))
        self.register_buffer("_pst_content_change", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_moving", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_contact", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_contact_force", torch.zeros(1, self.physical_slots, 2))
        self.register_buffer("_pst_contact_point", torch.zeros(1, self.physical_slots, 3))
        self.register_buffer("_pst_parent", torch.zeros(1, self.physical_slots, self.physical_slots))
        self.register_buffer("_pst_display_surface", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_surface_parent", torch.zeros(1, self.physical_slots, self.physical_slots + 1))
        self.register_buffer("_pst_surface_uv", torch.zeros(1, self.physical_slots, 2))
        self.register_buffer("_pst_surface_uv_confidence", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_verification", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_ontology_relation", torch.zeros(1, self.physical_slots, self.physical_slots, ModuleDim.PstOntologyRelationClasses))
        self.register_buffer("_pst_visibility", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_occlusion", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_has_text", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_text", torch.zeros(1, self.physical_slots, self.physical_text_dim))
        self.register_buffer(
            "_pst_entity_text_semantic",
            torch.zeros(1, self.physical_slots, self.entity_text_semantic_dim))
        self.register_buffer(
            "_pst_entity_text_confidence",
            torch.zeros(1, self.physical_slots))
        self.register_buffer(
            "_pst_entity_text_revision",
            torch.zeros(1, self.physical_slots, dtype=torch.long))
        self.register_buffer(
            "_pst_entity_text_changed",
            torch.zeros(1, self.physical_slots, dtype=torch.bool))
        self.register_buffer("_pst_symbol", torch.zeros(1, self.physical_slots, self.physical_symbol_dim))
        self.register_buffer("_pst_observed", torch.zeros(1, self.physical_slots, dtype=torch.bool))
        self.register_buffer("_pst_last_seen", torch.zeros(1, self.physical_slots, dtype=torch.long))
        self.register_buffer("_pst_step", torch.zeros(1, dtype=torch.long))
        self._mem_imp_lr = 0.10

        self._ns_enabled = bool(nsEnabled)

        self._ns_K: int = 128
        self.ns_struct = SoftNeSyStructure(k=self._ns_K, gExcl=30, gAlo=30, tauInit=1.0)

        self.ns_head_prior = NeSyHead(deterDim, self._ns_K, hidden=1024, experts=4)
        self.ns_head_post = NeSyHead(deterDim + stochDim, self._ns_K, hidden=1024, experts=4)

        self.ns_to_delta_mu = nn.Linear(self._ns_K, stochDim)
        self.ns_gate_mu = nn.Linear(deterDim + stochDim, stochDim)
        self.ns_gate_mu_post = nn.Linear(deterDim + 2 * stochDim, stochDim)

        self.key_emb = KeyEmbed(Z=stochDim, keyDim=stochDim)

        self.state_state_film = FiLMHResidual(baseDim=self.state_dim, rediusDim=self.state_dim, hidden=512)

        self.ns_lambda_excl = float(nsLambdaExclusive)
        self.ns_lambda_alo = float(nsLambdaAtLeastOne)
        self.ns_lambda_impl = float(nsLambdaImplication)

        self.ResetState(batchSize=1)

        self.conn = ConnNet(stateDim=stateDim,actDim=stochDim,wrapLinear=GrowableLoRALinear)

        self.phys_refiner = PhysRefinerHead(deterDim=self.deter_dim,actDim=self.stoch_dim)

        self.mix_gate = nn.Sequential(GrowableLoRALinear(nn.Linear(3 * self.state_dim, 3)))
        self.embodiment_context_proj = nn.Sequential(
            nn.LayerNorm(
                self.embodiment_state_dim
                + self.action_dim
                + self.physical_slot_dim
                + ROTATION_QUATERNION_DIM
                + 1),
            nn.Linear(
                self.embodiment_state_dim
                + self.action_dim
                + self.physical_slot_dim
                + ROTATION_QUATERNION_DIM
                + 1,
                self.embodiment_context_dim * 2),
            nn.SiLU(),
            nn.Linear(
                self.embodiment_context_dim * 2,
                self.embodiment_context_dim),
            nn.LayerNorm(self.embodiment_context_dim))
        self.embodied_action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim + self.embodiment_state_dim + self.embodiment_context_dim),
            GrowableLoRALinear(nn.Linear(self.action_dim + self.embodiment_state_dim + self.embodiment_context_dim, self.action_dim, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(self.action_dim, self.action_dim, bias=True)),
            nn.LayerNorm(self.action_dim),)

        self.future_action_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, max(self.state_dim, self.action_dim)),
            nn.GELU(),
            nn.Linear(max(self.state_dim, self.action_dim), self.action_dim),
            nn.LayerNorm(self.action_dim),)

        self.predicted_visual_head = PredictedVisualHead(
            stateDim=self.state_dim,
            globalFeatDim=self.global_feat_dim,
            objectTokenDim=self.object_token_dim,
            numObjectTokens=self.num_object_tokens,
            motionPredDim=self.motion_pred_dim,
            integratedFeatDim=self.integrated_feat_dim,)

        self.visual_reconstructor = VisualReconstructor(
            globalFeatDim=self.global_feat_dim,
            objectTokenDim=self.object_token_dim,
            numObjectTokens=self.num_object_tokens,
            motionPredDim=self.motion_pred_dim,
            integratedFeatDim=self.integrated_feat_dim,)

        self.pst_binder = PSTWorldBinder(
            hDim=self.deter_dim,
            zDim=self.stoch_dim,
            xDim=self.ssm_dim,
            actionDim=self.action_dim,
            embodimentDim=self.embodiment_context_dim,
            slotDim=self.physical_slot_dim,
            idDim=self.physical_id_dim,
            poseDim=self.physical_spatial_dim,
            attrDim=self.physical_attr_dim,
            semanticDim=self.physical_semantic_dim,
            stateDim=self.physical_state_dim,
            affordanceDim=self.physical_affordance_dim,
            relDim=self.physical_rel_dim,
            relationClasses=self.physical_relation_classes)

        world_abstract_dim = (
            self.deter_dim
            + self.stoch_dim
            + self.ssm_dim
            + self.state_dim
            + 2 * self.physical_slot_dim
            + self.embodiment_context_dim
            + 4)

        self.world_abstract_projector = nn.Sequential(
            nn.LayerNorm(world_abstract_dim),
            nn.Linear(world_abstract_dim, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),)

    def EnsurePhysicalMemory(self, B: int):
        if self._pst_slot_state.size(0) == B:
            return
        K = self.physical_slots
        self._pst_slot_state = self._pst_slot_state.new_zeros(B, K, self.physical_slot_dim)
        self._pst_spatial_world = self._pst_spatial_world.new_zeros(B, K, self.physical_spatial_dim)
        self._pst_attribute = self._pst_attribute.new_zeros(B, K, self.physical_attr_dim)
        self._pst_slot_presence = self._pst_slot_presence.new_zeros(B, K)
        self._pst_entity_prob = self._pst_entity_prob.new_zeros(B, K)
        self._pst_entity_id = self._pst_entity_id.new_full((B, K), -1)
        self._pst_slot_generation = self._pst_slot_generation.new_zeros(B, K)
        self._pst_next_entity_id = self._pst_next_entity_id.new_zeros(B)
        self._last_observed_to_world_slot = (
            self._last_observed_to_world_slot.new_full((B, 0), -1))
        self._pst_perceptual_presence = self._pst_perceptual_presence.new_zeros(B, K)
        self._pst_geometry_valid = self._pst_geometry_valid.new_zeros(B, K)
        self._pst_physical_interaction = self._pst_physical_interaction.new_zeros(B, K)
        self._pst_realm = self._pst_realm.new_zeros(B, K, ModuleDim.PstRealmClasses)
        self._pst_motion_layer = self._pst_motion_layer.new_zeros(B, K, ModuleDim.PstMotionLayerClasses)
        self._pst_layer_agency = self._pst_layer_agency.new_zeros(B, K, ModuleDim.PstMotionLayerClasses, ModuleDim.PstAgencyClasses)
        self._pst_agency = self._pst_agency.new_zeros(B, K, ModuleDim.PstAgencyClasses)
        self._pst_body_membership = self._pst_body_membership.new_zeros(B, K)
        self._pst_self_part = self._pst_self_part.new_zeros(
            B, K, self.self_part_count)
        self._pst_self_part_semantic = self._pst_self_part_semantic.new_zeros(
            B, K, self.self_part_semantic_dim)
        self._pst_identity_key = self._pst_identity_key.new_zeros(B, K, self.physical_id_dim)
        self._pst_pairwise_relation = self._pst_pairwise_relation.new_zeros(B, K, K, self.physical_rel_dim)
        self._pst_pair_last_seen = self._pst_pair_last_seen.new_zeros(B, K, K)
        self._pst_external_relation = self._pst_external_relation.new_zeros(B, K, self.physical_relation_classes)
        self._pst_semantic = self._pst_semantic.new_zeros(B, K, self.physical_semantic_dim)
        self._pst_size = self._pst_size.new_zeros(B, K, 3)
        self._pst_state = self._pst_state.new_zeros(B, K, self.physical_state_dim)
        self._pst_affordance = self._pst_affordance.new_zeros(B, K, self.physical_affordance_dim)
        self._pst_motion = self._pst_motion.new_zeros(B, K, self.physical_spatial_dim)
        self._pst_carrier_motion = self._pst_carrier_motion.new_zeros(B, K, self.physical_spatial_dim)
        self._pst_articulation_motion = self._pst_articulation_motion.new_zeros(B, K, self.physical_spatial_dim)
        self._pst_content_motion = self._pst_content_motion.new_zeros(B, K, 2)
        self._pst_content_change = self._pst_content_change.new_zeros(B, K)
        self._pst_moving = self._pst_moving.new_zeros(B, K)
        self._pst_contact = self._pst_contact.new_zeros(B, K)
        self._pst_contact_force = self._pst_contact_force.new_zeros(B, K, 2)
        self._pst_contact_point = self._pst_contact_point.new_zeros(B, K, 3)
        self._pst_parent = self._pst_parent.new_zeros(B, K, K)
        self._pst_display_surface = self._pst_display_surface.new_zeros(B, K)
        self._pst_surface_parent = self._pst_surface_parent.new_zeros(B, K, K + 1)
        self._pst_surface_uv = self._pst_surface_uv.new_zeros(B, K, 2)
        self._pst_surface_uv_confidence = self._pst_surface_uv_confidence.new_zeros(B, K)
        self._pst_verification = self._pst_verification.new_zeros(B, K)
        self._pst_ontology_relation = self._pst_ontology_relation.new_zeros(B, K, K, ModuleDim.PstOntologyRelationClasses)
        self._pst_visibility = self._pst_visibility.new_zeros(B, K)
        self._pst_occlusion = self._pst_occlusion.new_zeros(B, K)
        self._pst_has_text = self._pst_has_text.new_zeros(B, K)
        self._pst_text = self._pst_text.new_zeros(B, K, self.physical_text_dim)
        self._pst_entity_text_semantic = (
            self._pst_entity_text_semantic.new_zeros(
                B, K, self.entity_text_semantic_dim))
        self._pst_entity_text_confidence = (
            self._pst_entity_text_confidence.new_zeros(B, K))
        self._pst_entity_text_revision = (
            self._pst_entity_text_revision.new_zeros(B, K))
        self._pst_entity_text_changed = (
            self._pst_entity_text_changed.new_zeros(B, K))
        self._pst_symbol = self._pst_symbol.new_zeros(B, K, self.physical_symbol_dim)
        self._pst_observed = self._pst_observed.new_zeros(B, K)
        self._pst_last_seen = self._pst_last_seen.new_zeros(B, K)
        self._pst_step = self._pst_step.new_zeros(B)

    def EnsureB(self, B: int):
        B = int(B)
        cap = int(self._mem_capacity)
        self.EnsurePhysicalMemory(B)

        if self._mem_keys.size(0) != B:
            self._mem_keys = self._mem_keys.new_zeros(B, cap, self.stoch_dim)
            self._mem_vals = self._mem_vals.new_zeros(B, cap, self.state_dim)
            self._mem_imp = self._mem_imp.new_zeros(B, cap)
            self._mem_steps = self._mem_steps.new_zeros(B, cap)
            self._mem_size = self._mem_size.new_zeros(B)
            self._mem_global_step = self._mem_global_step.new_zeros(B)

        if self._h.size(0) != B:
            self._h = self.NewZeros(B, self.deter_dim)
            self._z = self.NewZeros(B, self.stoch_dim)
            self._A_prev = None
            self._A_prev_valid = None

        self.s4.EnsureB(B)

    def ResolveCommitMask(
        self,
        commitMask: Optional[torch.Tensor],
        batchSize: int,
    ) -> torch.Tensor:
        runtime_reference = next(self.parameters(), None)
        if runtime_reference is None:
            runtime_reference = next(self.buffers(), None)
        runtime_device = (
            torch.device("cpu")
            if runtime_reference is None
            else runtime_reference.device)
        if commitMask is None:
            return torch.ones(
                int(batchSize),
                device=runtime_device,
                dtype=torch.bool)
        if (
            not torch.is_tensor(commitMask)
            or tuple(commitMask.shape) != (int(batchSize),)
            or commitMask.device != runtime_device
            or commitMask.dtype != torch.bool
        ):
            raise ValueError("commitMask must be a batched boolean mask")
        return commitMask

    def MergeCommittedRows(
        self,
        update: torch.Tensor,
        previous: torch.Tensor,
        commitMask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not torch.is_tensor(update)
            or not torch.is_tensor(previous)
            or update.shape != previous.shape
            or update.dim() < 1
            or update.device != previous.device
            or update.dtype != previous.dtype
        ):
            raise ValueError("committed world states must share shape, device, and dtype")
        mask = self.ResolveCommitMask(commitMask, int(update.size(0)))
        while mask.dim() < update.dim():
            mask = mask.unsqueeze(-1)
        return torch.where(mask, update, previous)

    def MaskedBatchMean(
        self,
        value: torch.Tensor,
        sampleMask: torch.Tensor,
    ) -> torch.Tensor:
        mask = self.ResolveCommitMask(sampleMask, int(value.size(0)))
        per_row = value.reshape(value.size(0), -1).mean(dim=-1)
        weight = mask.to(dtype=per_row.dtype)
        return (per_row * weight).sum() / weight.sum().clamp_min(1.0)

    def PhysicalRuntimeStateNames(self) -> Tuple[str, ...]:
        return tuple(
            name
            for name in self._buffers
            if name.startswith("_pst_")
            or name == "_last_observed_to_world_slot")

    @torch.no_grad()
    def CapturePhysicalRuntimeRows(
        self,
        preserveMask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask = self.ResolveCommitMask(
            preserveMask,
            int(self._pst_slot_state.size(0)))
        return {
            name: getattr(self, name)[mask].detach().clone()
            for name in self.PhysicalRuntimeStateNames()}

    @torch.no_grad()
    def RestorePhysicalRuntimeRows(
        self,
        snapshot: Dict[str, torch.Tensor],
        preserveMask: torch.Tensor,
    ) -> None:
        mask = self.ResolveCommitMask(
            preserveMask,
            int(self._pst_slot_state.size(0)))
        selected = int(mask.sum().item())
        expected = set(self.PhysicalRuntimeStateNames())
        if not isinstance(snapshot, dict) or set(snapshot) != expected:
            raise ValueError("physical runtime snapshot is incomplete")
        for name in self.PhysicalRuntimeStateNames():
            target = getattr(self, name)
            value = snapshot[name]
            if (
                not torch.is_tensor(value)
                or value.size(0) != selected
                or value.device != target.device
                or value.dtype != target.dtype
                or tuple(value.shape[1:]) != tuple(target.shape[1:])
            ):
                raise ValueError("physical runtime snapshot is invalid")
            target[mask] = value


    def BindMemoryContext(self, calibrationId: str, worldFrameId: str) -> None:
        if not isinstance(calibrationId, str) or not calibrationId.strip():
            raise ValueError("calibrationId must be a non-empty string")
        if not isinstance(worldFrameId, str) or not worldFrameId.strip():
            raise ValueError("worldFrameId must be a non-empty string")
        context = (calibrationId, worldFrameId)
        current = (self._memory_calibration_id, self._memory_world_frame_id)
        if current != (None, None) and current != context:
            raise RuntimeError(
                "world memory context is already bound to "
                f"calibration_id={current[0]!r}, world_frame_id={current[1]!r}")
        self._memory_calibration_id, self._memory_world_frame_id = context

    def RequireMemoryContext(self) -> Tuple[str, str]:
        if self._memory_calibration_id is None or self._memory_world_frame_id is None:
            raise RuntimeError(
                "world memory context is unbound; call BindMemoryContext before saving or loading")
        return self._memory_calibration_id, self._memory_world_frame_id

    def ValidateMemoryPayload(self, payload: Any) -> Tuple[int, int]:
        if not isinstance(payload, dict):
            raise TypeError("world memory payload must be a dictionary")
        actual_fields = frozenset(payload)
        if actual_fields != WORLD_MEMORY_PAYLOAD_FIELDS:
            missing = sorted(WORLD_MEMORY_PAYLOAD_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - WORLD_MEMORY_PAYLOAD_FIELDS)
            raise ValueError(
                f"world memory payload fields mismatch: missing={missing}, unexpected={unexpected}")
        schema_version = payload["world_memory_schema_version"]
        if type(schema_version) is not int or schema_version != WORLD_MEMORY_SCHEMA_VERSION:
            raise ValueError(
                "world memory schema mismatch: "
                f"expected {WORLD_MEMORY_SCHEMA_VERSION}, got {schema_version!r}")
        calibration_id, world_frame_id = self.RequireMemoryContext()
        if payload["calibration_id"] != calibration_id:
            raise ValueError(
                "world memory calibration_id mismatch: "
                f"expected {calibration_id!r}, got {payload['calibration_id']!r}")
        if payload["world_frame_id"] != world_frame_id:
            raise ValueError(
                "world memory world_frame_id mismatch: "
                f"expected {world_frame_id!r}, got {payload['world_frame_id']!r}")
        if payload["pst_contact_point_frame"] != "world":
            raise ValueError("world memory pst_contact_point_frame must be 'world'")

        for field in WORLD_MEMORY_TENSOR_FIELDS:
            if not torch.is_tensor(payload[field]):
                raise TypeError(f"world memory field {field!r} must be a tensor")
        keys = payload["mem_keys"]
        if keys.ndim != 3:
            raise ValueError(f"world memory mem_keys must be rank 3, got {keys.ndim}")
        B, C = int(keys.size(0)), int(keys.size(1))
        if B < 1:
            raise ValueError("world memory batch size must be positive")
        if type(payload["batch_size"]) is not int or payload["batch_size"] != B:
            raise ValueError(
                "world memory batch_size must match the stored tensor batch")
        observed_mapping = payload["last_observed_to_world_slot"]
        if observed_mapping.ndim != 2 or int(observed_mapping.size(0)) != B:
            raise ValueError(
                "world memory last_observed_to_world_slot must have shape "
                f"[B, Kobs], got {tuple(observed_mapping.shape)}")
        Kobs = int(observed_mapping.size(1))
        K = self.physical_slots
        expected_shapes = {
            "mem_keys": (B, C, self.stoch_dim),
            "mem_vals": (B, C, self.state_dim),
            "mem_imp": (B, C),
            "mem_steps": (B, C),
            "mem_size": (B,),
            "mem_global_step": (B,),
            "pst_slot_state": (B, K, self.physical_slot_dim),
            "pst_spatial_world": (B, K, self.physical_spatial_dim),
            "pst_attribute": (B, K, self.physical_attr_dim),
            "pst_slot_presence": (B, K),
            "pst_entity_prob": (B, K),
            "pst_entity_id": (B, K),
            "pst_slot_generation": (B, K),
            "pst_next_entity_id": (B,),
            "last_observed_to_world_slot": (B, Kobs),
            "pst_perceptual_presence": (B, K),
            "pst_geometry_valid": (B, K),
            "pst_physical_interaction": (B, K),
            "pst_realm": (B, K, ModuleDim.PstRealmClasses),
            "pst_motion_layer": (B, K, ModuleDim.PstMotionLayerClasses),
            "pst_layer_agency": (B, K, ModuleDim.PstMotionLayerClasses, ModuleDim.PstAgencyClasses),
            "pst_agency": (B, K, ModuleDim.PstAgencyClasses),
            "pst_body_membership": (B, K),
            "pst_self_part": (B, K, self.self_part_count),
            "pst_self_part_semantic": (
                B, K, self.self_part_semantic_dim),
            "pst_identity_key": (B, K, self.physical_id_dim),
            "pst_pairwise_relation": (B, K, K, self.physical_rel_dim),
            "pst_pair_last_seen": (B, K, K),
            "pst_external_relation": (B, K, self.physical_relation_classes),
            "pst_semantic": (B, K, self.physical_semantic_dim),
            "pst_size": (B, K, 3),
            "pst_state": (B, K, self.physical_state_dim),
            "pst_affordance": (B, K, self.physical_affordance_dim),
            "pst_motion": (B, K, self.physical_spatial_dim),
            "pst_carrier_motion": (B, K, self.physical_spatial_dim),
            "pst_articulation_motion": (B, K, self.physical_spatial_dim),
            "pst_content_motion": (B, K, 2),
            "pst_content_change": (B, K),
            "pst_moving": (B, K),
            "pst_contact": (B, K),
            "pst_contact_force": (B, K, 2),
            "pst_contact_point": (B, K, 3),
            "pst_parent": (B, K, K),
            "pst_display_surface": (B, K),
            "pst_surface_parent": (B, K, K + 1),
            "pst_surface_uv": (B, K, 2),
            "pst_surface_uv_confidence": (B, K),
            "pst_verification": (B, K),
            "pst_ontology_relation": (B, K, K, ModuleDim.PstOntologyRelationClasses),
            "pst_visibility": (B, K),
            "pst_occlusion": (B, K),
            "pst_has_text": (B, K),
            "pst_text": (B, K, self.physical_text_dim),
            "pst_entity_text_semantic": (
                B, K, self.entity_text_semantic_dim),
            "pst_entity_text_confidence": (B, K),
            "pst_entity_text_revision": (B, K),
            "pst_entity_text_changed": (B, K),
            "pst_symbol": (B, K, self.physical_symbol_dim),
            "pst_observed": (B, K),
            "pst_last_seen": (B, K),
            "pst_step": (B,),}
        for field, expected in expected_shapes.items():
            actual = tuple(payload[field].shape)
            if actual != expected:
                raise ValueError(
                    f"world memory field {field!r} must have shape {expected}, got {actual}")

        integer_fields = (
            "mem_steps", "mem_size", "mem_global_step",
            "pst_entity_id", "pst_slot_generation", "pst_next_entity_id",
            "last_observed_to_world_slot", "pst_pair_last_seen",
            "pst_entity_text_revision", "pst_last_seen", "pst_step")
        for field in integer_fields:
            if payload[field].dtype != torch.long:
                raise TypeError(f"world memory field {field!r} must have dtype torch.long")
        bool_fields = {"pst_entity_text_changed", "pst_observed"}
        for field in bool_fields:
            if payload[field].dtype != torch.bool:
                raise TypeError(
                    f"world memory field {field!r} must have dtype torch.bool")
        float_fields = set(WORLD_MEMORY_TENSOR_FIELDS) - set(integer_fields) - bool_fields
        for field in float_fields:
            if not payload[field].is_floating_point():
                raise TypeError(f"world memory field {field!r} must be floating point")
            if not bool(torch.isfinite(payload[field]).all().item()):
                raise ValueError(
                    f"world memory field {field!r} must contain only finite values")
        if bool(((payload["mem_size"] < 0) | (payload["mem_size"] > C)).any().item()):
            raise ValueError("world memory mem_size must be within the stored memory capacity")
        if bool((payload["mem_global_step"] < 0).any().item()):
            raise ValueError("world memory mem_global_step must be non-negative")
        entity_id = payload["pst_entity_id"]
        slot_generation = payload["pst_slot_generation"]
        next_entity_id = payload["pst_next_entity_id"]
        if bool((entity_id < -1).any().item()):
            raise ValueError("world memory pst_entity_id must be at least -1")
        if bool((slot_generation < 0).any().item()):
            raise ValueError("world memory pst_slot_generation must be non-negative")
        if bool((next_entity_id < 0).any().item()):
            raise ValueError("world memory pst_next_entity_id must be non-negative")
        if bool((payload["pst_entity_text_revision"] < 0).any().item()):
            raise ValueError(
                "world memory pst_entity_text_revision must be non-negative")
        text_confidence = payload["pst_entity_text_confidence"]
        if bool(((text_confidence < 0.0) | (text_confidence > 1.0)).any().item()):
            raise ValueError(
                "world memory pst_entity_text_confidence must be a probability")
        valid_entity = entity_id >= 0
        if bool((valid_entity & (entity_id >= next_entity_id.unsqueeze(1))).any().item()):
            raise ValueError(
                "world memory pst_entity_id must be smaller than pst_next_entity_id")
        if bool((valid_entity & (slot_generation <= 0)).any().item()):
            raise ValueError(
                "world memory live entity ids must have positive slot generation")
        for b in range(B):
            lane_ids = entity_id[b, valid_entity[b]]
            if lane_ids.numel() != torch.unique(lane_ids).numel():
                raise ValueError("world memory pst_entity_id must be unique within each batch lane")
        if bool(((observed_mapping < -1) | (observed_mapping >= K)).any().item()):
            raise ValueError(
                "world memory last_observed_to_world_slot must contain -1 or a valid slot index")
        if Kobs > 0:
            safe_mapping = observed_mapping.clamp_min(0)
            mapped_entity = entity_id.gather(1, safe_mapping)
            if bool(((observed_mapping >= 0) & (mapped_entity < 0)).any().item()):
                raise ValueError(
                    "world memory observed associations must reference assigned entity ids")
        return B, C

    @torch.no_grad()
    def ExportMemoryPayload(self) -> Dict[str, Any]:
        if not self._use_memory:
            raise RuntimeError("world memory is disabled")
        calibration_id, world_frame_id = self.RequireMemoryContext()

        B = int(self._mem_keys.size(0))
        maxN = int(self._mem_size.max().item()) if B > 0 else 0
        if maxN < 0 or maxN > self._mem_capacity:
            raise ValueError("world memory size exceeds its configured capacity")
        payload = {
            "world_memory_schema_version": WORLD_MEMORY_SCHEMA_VERSION,
            "calibration_id": calibration_id,
            "world_frame_id": world_frame_id,
            "batch_size": B,
            "pst_contact_point_frame": "world",
            "mem_keys": self._mem_keys[:, :maxN].detach().cpu(),
            "mem_vals": self._mem_vals[:, :maxN].detach().cpu(),
            "mem_imp": self._mem_imp[:, :maxN].detach().cpu(),
            "mem_steps": self._mem_steps[:, :maxN].detach().cpu(),
            "mem_size": self._mem_size.detach().cpu(),
            "mem_global_step": self._mem_global_step.detach().cpu(),
            "pst_slot_state": self._pst_slot_state.detach().cpu(),
            "pst_spatial_world": self._pst_spatial_world.detach().cpu(),
            "pst_attribute": self._pst_attribute.detach().cpu(),
            "pst_slot_presence": self._pst_slot_presence.detach().cpu(),
            "pst_entity_prob": self._pst_entity_prob.detach().cpu(),
            "pst_entity_id": self._pst_entity_id.detach().cpu(),
            "pst_slot_generation": self._pst_slot_generation.detach().cpu(),
            "pst_next_entity_id": self._pst_next_entity_id.detach().cpu(),
            "last_observed_to_world_slot": (
                self._last_observed_to_world_slot.detach().cpu()),
            "pst_perceptual_presence": self._pst_perceptual_presence.detach().cpu(),
            "pst_geometry_valid": self._pst_geometry_valid.detach().cpu(),
            "pst_physical_interaction": self._pst_physical_interaction.detach().cpu(),
            "pst_realm": self._pst_realm.detach().cpu(),
            "pst_motion_layer": self._pst_motion_layer.detach().cpu(),
            "pst_layer_agency": self._pst_layer_agency.detach().cpu(),
            "pst_agency": self._pst_agency.detach().cpu(),
            "pst_body_membership": self._pst_body_membership.detach().cpu(),
            "pst_self_part": self._pst_self_part.detach().cpu(),
            "pst_self_part_semantic": self._pst_self_part_semantic.detach().cpu(),
            "pst_identity_key": self._pst_identity_key.detach().cpu(),
            "pst_pairwise_relation": self._pst_pairwise_relation.detach().cpu(),
            "pst_pair_last_seen": self._pst_pair_last_seen.detach().cpu(),
            "pst_external_relation": self._pst_external_relation.detach().cpu(),
            "pst_semantic": self._pst_semantic.detach().cpu(),
            "pst_size": self._pst_size.detach().cpu(),
            "pst_state": self._pst_state.detach().cpu(),
            "pst_affordance": self._pst_affordance.detach().cpu(),
            "pst_motion": self._pst_motion.detach().cpu(),
            "pst_carrier_motion": self._pst_carrier_motion.detach().cpu(),
            "pst_articulation_motion": self._pst_articulation_motion.detach().cpu(),
            "pst_content_motion": self._pst_content_motion.detach().cpu(),
            "pst_content_change": self._pst_content_change.detach().cpu(),
            "pst_moving": self._pst_moving.detach().cpu(),
            "pst_contact": self._pst_contact.detach().cpu(),
            "pst_contact_force": self._pst_contact_force.detach().cpu(),
            "pst_contact_point": self._pst_contact_point.detach().cpu(),
            "pst_parent": self._pst_parent.detach().cpu(),
            "pst_display_surface": self._pst_display_surface.detach().cpu(),
            "pst_surface_parent": self._pst_surface_parent.detach().cpu(),
            "pst_surface_uv": self._pst_surface_uv.detach().cpu(),
            "pst_surface_uv_confidence": self._pst_surface_uv_confidence.detach().cpu(),
            "pst_verification": self._pst_verification.detach().cpu(),
            "pst_ontology_relation": self._pst_ontology_relation.detach().cpu(),
            "pst_visibility": self._pst_visibility.detach().cpu(),
            "pst_occlusion": self._pst_occlusion.detach().cpu(),
            "pst_has_text": self._pst_has_text.detach().cpu(),
            "pst_text": self._pst_text.detach().cpu(),
            "pst_entity_text_semantic": (
                self._pst_entity_text_semantic.detach().cpu()),
            "pst_entity_text_confidence": (
                self._pst_entity_text_confidence.detach().cpu()),
            "pst_entity_text_revision": (
                self._pst_entity_text_revision.detach().cpu()),
            "pst_entity_text_changed": (
                self._pst_entity_text_changed.detach().cpu()),
            "pst_symbol": self._pst_symbol.detach().cpu(),
            "pst_observed": self._pst_observed.detach().cpu(),
            "pst_last_seen": self._pst_last_seen.detach().cpu(),
            "pst_step": self._pst_step.detach().cpu(),}
        self.ValidateMemoryPayload(payload)
        return payload

    def SaveMemory(self, path: Optional[str] = None) -> None:
        p = path or self._mem_path
        if not p:
            raise ValueError("world memory path is not configured")
        dirpath = os.path.dirname(p)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(p)}.",
            suffix=".tmp",
            dir=dirpath or ".")
        os.close(fd)
        try:
            torch.save(self.ExportMemoryPayload(), temporary_path)
            os.replace(temporary_path, p)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def LoadMemory(
        self,
        path: str,
        *,
        batchSize: int,
        mapLocation: Optional[str] = None,) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if os.path.getsize(path) == 0:
            raise ValueError(f"world memory file is empty: {path}")

        payload = torch.load(path, map_location=mapLocation, weights_only=False)
        self.ImportMemoryPayload(payload, batchSize=batchSize)

    @torch.no_grad()
    def ImportMemoryPayload(
        self,
        payload: Dict[str, Any],
        *,
        batchSize: int,) -> None:
        if not self._use_memory:
            raise RuntimeError("world memory is disabled")
        self.RequireMemoryContext()
        if type(batchSize) is not int or batchSize < 1:
            raise ValueError("world memory batchSize must be a positive integer")
        Bf, Cf = self.ValidateMemoryPayload(payload)
        if Bf != batchSize:
            raise ValueError(
                f"world memory batch size mismatch: expected {batchSize}, got {Bf}")
        new_cap = max(self._mem_capacity, Cf)
        dev = self.device
        dtyp = self.dtype

        new_keys = torch.zeros(Bf, new_cap, self.stoch_dim, device=dev, dtype=dtyp)
        new_vals = torch.zeros(Bf, new_cap, self.state_dim, device=dev, dtype=dtyp)
        new_imp = torch.zeros(Bf, new_cap, device=dev, dtype=dtyp)
        new_steps = torch.zeros(Bf, new_cap, device=dev, dtype=torch.long)
        new_keys[:, :Cf].copy_(payload["mem_keys"].to(device=dev, dtype=dtyp))
        new_vals[:, :Cf].copy_(payload["mem_vals"].to(device=dev, dtype=dtyp))
        new_imp[:, :Cf].copy_(payload["mem_imp"].to(device=dev, dtype=dtyp))
        new_steps[:, :Cf].copy_(payload["mem_steps"].to(device=dev, dtype=torch.long))

        self._mem_capacity = new_cap
        self._mem_keys = new_keys
        self._mem_vals = new_vals
        self._mem_imp = new_imp
        self._mem_steps = new_steps
        self._mem_size = payload["mem_size"].to(device=dev, dtype=torch.long)
        self._mem_global_step = payload["mem_global_step"].to(device=dev, dtype=torch.long)
        self.EnsureB(Bf)

        float_buffers = {
            "_pst_slot_state": "pst_slot_state",
            "_pst_spatial_world": "pst_spatial_world",
            "_pst_attribute": "pst_attribute",
            "_pst_slot_presence": "pst_slot_presence",
            "_pst_entity_prob": "pst_entity_prob",
            "_pst_perceptual_presence": "pst_perceptual_presence",
            "_pst_physical_interaction": "pst_physical_interaction",
            "_pst_realm": "pst_realm",
            "_pst_motion_layer": "pst_motion_layer",
            "_pst_layer_agency": "pst_layer_agency",
            "_pst_agency": "pst_agency",
            "_pst_body_membership": "pst_body_membership",
            "_pst_self_part": "pst_self_part",
            "_pst_self_part_semantic": "pst_self_part_semantic",
            "_pst_identity_key": "pst_identity_key",
            "_pst_pairwise_relation": "pst_pairwise_relation",
            "_pst_external_relation": "pst_external_relation",
            "_pst_semantic": "pst_semantic",
            "_pst_size": "pst_size",
            "_pst_state": "pst_state",
            "_pst_affordance": "pst_affordance",
            "_pst_motion": "pst_motion",
            "_pst_carrier_motion": "pst_carrier_motion",
            "_pst_articulation_motion": "pst_articulation_motion",
            "_pst_content_motion": "pst_content_motion",
            "_pst_content_change": "pst_content_change",
            "_pst_moving": "pst_moving",
            "_pst_contact": "pst_contact",
            "_pst_contact_force": "pst_contact_force",
            "_pst_contact_point": "pst_contact_point",
            "_pst_parent": "pst_parent",
            "_pst_display_surface": "pst_display_surface",
            "_pst_surface_parent": "pst_surface_parent",
            "_pst_surface_uv": "pst_surface_uv",
            "_pst_surface_uv_confidence": "pst_surface_uv_confidence",
            "_pst_verification": "pst_verification",
            "_pst_ontology_relation": "pst_ontology_relation",
            "_pst_visibility": "pst_visibility",
            "_pst_occlusion": "pst_occlusion",
            "_pst_has_text": "pst_has_text",
            "_pst_text": "pst_text",
            "_pst_entity_text_semantic": "pst_entity_text_semantic",
            "_pst_entity_text_confidence": "pst_entity_text_confidence",
            "_pst_symbol": "pst_symbol",}
        for buffer_name, field in float_buffers.items():
            getattr(self, buffer_name).copy_(payload[field].to(device=dev, dtype=dtyp))
        self._pst_pair_last_seen.copy_(payload["pst_pair_last_seen"].to(device=dev, dtype=torch.long))
        self._pst_geometry_valid.copy_(payload["pst_geometry_valid"].to(device=dev, dtype=dtyp))
        self._pst_entity_id.copy_(payload["pst_entity_id"].to(device=dev, dtype=torch.long))
        self._pst_slot_generation.copy_(
            payload["pst_slot_generation"].to(device=dev, dtype=torch.long))
        self._pst_next_entity_id.copy_(
            payload["pst_next_entity_id"].to(device=dev, dtype=torch.long))
        self._last_observed_to_world_slot = payload[
            "last_observed_to_world_slot"].to(device=dev, dtype=torch.long).clone()
        self._pst_observed.copy_(payload["pst_observed"].to(device=dev, dtype=torch.bool))
        self._pst_entity_text_revision.copy_(
            payload["pst_entity_text_revision"].to(
                device=dev, dtype=torch.long))
        self._pst_entity_text_changed.copy_(
            payload["pst_entity_text_changed"].to(
                device=dev, dtype=torch.bool))
        self._pst_last_seen.copy_(payload["pst_last_seen"].to(device=dev, dtype=torch.long))
        self._pst_step.copy_(payload["pst_step"].to(device=dev, dtype=torch.long))

    @torch.no_grad()
    def ResetPhysicalState(self, doneMask: Optional[torch.Tensor] = None) -> None:
        buffers = (
            self._pst_slot_state,
            self._pst_spatial_world,
            self._pst_attribute,
            self._pst_slot_presence,
            self._pst_entity_prob,
            self._pst_slot_generation,
            self._pst_next_entity_id,
            self._pst_perceptual_presence,
            self._pst_geometry_valid,
            self._pst_physical_interaction,
            self._pst_realm,
            self._pst_motion_layer,
            self._pst_layer_agency,
            self._pst_agency,
            self._pst_body_membership,
            self._pst_self_part,
            self._pst_self_part_semantic,
            self._pst_identity_key,
            self._pst_pairwise_relation,
            self._pst_pair_last_seen,
            self._pst_external_relation,
            self._pst_semantic,
            self._pst_size,
            self._pst_state,
            self._pst_affordance,
            self._pst_motion,
            self._pst_carrier_motion,
            self._pst_articulation_motion,
            self._pst_content_motion,
            self._pst_content_change,
            self._pst_moving,
            self._pst_contact,
            self._pst_contact_force,
            self._pst_contact_point,
            self._pst_parent,
            self._pst_display_surface,
            self._pst_surface_parent,
            self._pst_surface_uv,
            self._pst_surface_uv_confidence,
            self._pst_verification,
            self._pst_ontology_relation,
            self._pst_visibility,
            self._pst_occlusion,
            self._pst_has_text,
            self._pst_text,
            self._pst_entity_text_semantic,
            self._pst_entity_text_confidence,
            self._pst_entity_text_revision,
            self._pst_entity_text_changed,
            self._pst_symbol,
            self._pst_observed,
            self._pst_last_seen,
            self._pst_step,)
        if doneMask is None:
            for buffer in buffers:
                buffer.zero_()
            self._pst_entity_id.fill_(-1)
            self._last_observed_to_world_slot.fill_(-1)
            return

        mask = doneMask.to(device=self._pst_slot_state.device, dtype=torch.bool).view(-1)
        if mask.numel() != self._pst_slot_state.size(0):
            raise ValueError(
                f"doneMask must have {self._pst_slot_state.size(0)} elements, got {mask.numel()}")
        if not bool(mask.any().item()):
            return
        for buffer in buffers:
            buffer[mask] = 0
        self._pst_entity_id[mask] = -1
        self._last_observed_to_world_slot[mask] = -1

    @torch.no_grad()
    def ResetEpisodeState(self, doneMask: torch.Tensor) -> None:
        mask = doneMask.to(device=self._h.device, dtype=torch.bool).view(-1)
        if mask.numel() != self._h.size(0):
            raise ValueError(f"doneMask must have {self._h.size(0)} elements, got {mask.numel()}")
        if not bool(mask.any().item()):
            return
        self._h[mask] = 0
        self._z[mask] = 0
        self.s4.x[mask] = 0
        if self._A_prev is not None and self._A_prev.size(0) == mask.numel():
            self._A_prev[mask] = 0
        if (
            self._A_prev_valid is not None
            and self._A_prev_valid.size(0) == mask.numel()
        ):
            self._A_prev_valid[mask] = False
        self.ResetPhysicalState(mask)

    def ResetMemory(self):
        if self._use_memory:
            self._mem_keys.zero_()
            self._mem_vals.zero_()
            self._mem_imp.zero_()
            self._mem_steps.zero_()
            self._mem_size.zero_()
            self._mem_global_step.zero_()
            self._mem_add_count = 0
            self._mem_autosave_pending = False
        self.ResetPhysicalState()

    def HasMemoryAutosaveRequest(self) -> bool:
        return bool(self._mem_autosave_pending)

    def AcknowledgeMemoryAutosaveRequest(self) -> None:
        self._mem_autosave_pending = False

    @torch.no_grad()
    def ReorderMemorySteps(self):
        if not self._use_memory:
            return

        B = int(self._mem_size.size(0))
        cap = int(self._mem_capacity)
        device = self.device

        if cap <= 0 or B <= 0:
            self._mem_steps.zero_()
            self._mem_global_step.zero_()
            return

        slots = torch.arange(cap, device=device).view(1, cap)
        valid = slots < self._mem_size.view(B, 1) # [B, cap]

        if not bool(valid.any().item()):
            self._mem_steps.zero_()
            self._mem_global_step.zero_()
            return

        max_step = torch.iinfo(self._mem_steps.dtype).max
        metric = torch.where(valid, self._mem_steps, torch.full_like(self._mem_steps, max_step))
        order = torch.argsort(metric, dim=1, descending=False)

        new_steps = torch.zeros_like(self._mem_steps)
        ranks = torch.arange(1, cap + 1, device=device, dtype=torch.long).view(1, cap).expand(B, cap)
        rank_valid = ranks <= self._mem_size.view(B, 1)
        assign = torch.where(rank_valid, ranks, torch.zeros_like(ranks))
        new_steps.scatter_(1, order, assign)
        new_steps = torch.where(valid, new_steps, torch.zeros_like(new_steps))

        self._mem_steps.copy_(new_steps)
        self._mem_global_step.copy_(self._mem_size)

    @torch.no_grad()
    def MemAdd(
        self,
        keyE: torch.Tensor, # [B, Z]
        valH: torch.Tensor, # [B, D]
        imp: torch.Tensor,
        commitMask: Optional[torch.Tensor] = None,): # [B]

        if not self._use_memory:
            return

        B = int(keyE.size(0))
        commit_mask = self.ResolveCommitMask(commitMask, B)
        if not bool(commit_mask.any().item()):
            return

        cap = int(self._mem_capacity)

        size = self._mem_size # [B]
        has_space = size < cap # [B]
        idx_replace = torch.argmin(self._mem_imp, dim=1) # [B]
        idx = torch.where(has_space, size, idx_replace).long() # [B]

        self._mem_global_step.add_(commit_mask.to(dtype=torch.long))
        self._mem_size = torch.where(
            commit_mask & has_space,
            size + 1,
            size) # [B]

        bidx = torch.nonzero(commit_mask, as_tuple=False).flatten()
        active_idx = idx.index_select(0, bidx)

        self._mem_keys[bidx, active_idx] = keyE.index_select(0, bidx)
        self._mem_vals[bidx, active_idx] = valH.index_select(0, bidx)
        self._mem_imp[bidx, active_idx] = imp.index_select(0, bidx)
        self._mem_steps[bidx, active_idx] = self._mem_global_step.index_select(
            0, bidx)

        if self._mem_autosave_every > 0:
            self._mem_add_count += 1
            if self._mem_add_count % self._mem_autosave_every == 0:
                if self._mem_path:
                    self.SaveMemory(self._mem_path)
                else:
                    self._mem_autosave_pending = True

    def MemRetrieve(
        self,
        queryE: torch.Tensor,
        *,
        updateImportance: bool = True,
        commitMask: Optional[torch.Tensor] = None,
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._use_memory:
            return None

        B = int(queryE.size(0))
        commit_mask = self.ResolveCommitMask(commitMask, B)

        filled = self._mem_size # [B]
        filled_max = int(filled.max().item())

        if filled_max <= 0:
            return None

        cap = int(self._mem_capacity)
        K_req = max(int(self._mem_topk), 1)
        K = min(K_req, filled_max, cap)
        temp = max(float(self._mem_temp), 1e-6)

        keys = self._mem_keys # [B,C,Z]
        vals = self._mem_vals # [B,C,S]

        q = queryE # [B,Z]
        k = keys

        sims = torch.einsum("bd,bnd->bn", q, k) # [B,C]

        idx = torch.arange(cap, device=sims.device).view(1, cap)
        valid = idx < self._mem_size.view(B, 1)
        sims = sims.masked_fill(
            ~valid,
            torch.finfo(sims.dtype).min)

        top_vals, top_idx = torch.topk(sims, k=K, dim=-1) # [B,K]

        logits = top_vals / temp
        weights = F.softmax(logits, dim=-1) # [B,K]

        has_memory = self._mem_size > 0
        weights = weights * has_memory.view(B, 1).to(weights.dtype)

        gathered = vals.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, vals.size(-1))) # [B,K,D]
        mem_h = (weights.unsqueeze(-1) * gathered).sum(dim=1) # [B,D]

        if updateImportance:
            with torch.no_grad():
                inc = weights.detach() # [B,K] in [0,1], sum=1
                cur = self._mem_imp.gather(1, top_idx) # [B,K]
                lr = float(getattr(self, "_mem_imp_lr", 0.10))
                new = cur + lr * (1.0 - cur) * inc
                new = torch.where(commit_mask.unsqueeze(-1), new, cur)
                self._mem_imp.scatter_(1, top_idx, new.clamp_(0.0, 1.0))

        return mem_h, has_memory # [B,D], [B]

    @staticmethod
    def QuaternionMultiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lx, ly, lz, lw = left.unbind(dim=-1)
        rx, ry, rz, rw = right.unbind(dim=-1)
        return torch.stack([
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,], dim=-1)

    @staticmethod
    def QuaternionRotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        q_vec = quat[..., :3]
        t = 2.0 * torch.cross(q_vec, vector, dim=-1)
        return vector + quat[..., 3:4] * t + torch.cross(q_vec, t, dim=-1)

    @staticmethod
    def NormalizeQuaternion(quat: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(quat.float(), dim=-1, eps=1e-6).to(quat.dtype)
        identity = torch.zeros_like(normalized)
        identity[..., 3] = 1.0
        return torch.where(
            quat.norm(dim=-1, keepdim=True) > 1e-6,
            normalized,
            identity)


    def ValidateContractObserverRotation(
        self,
        observerRotationWorld: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(observerRotationWorld):
            raise TypeError("observer rotation must be a tensor")
        if (
            observerRotationWorld.ndim != 2
            or int(observerRotationWorld.size(-1)) != ROTATION_QUATERNION_DIM
            or int(observerRotationWorld.size(0)) < 1
        ):
            raise ValueError("observer rotation must have shape [B, 4]")
        if not observerRotationWorld.is_floating_point():
            raise TypeError("observer rotation must be floating point")
        if observerRotationWorld.device != self.device:
            raise ValueError("observer rotation device does not match world model")
        if observerRotationWorld.dtype != self.dtype:
            raise ValueError("observer rotation dtype does not match world model")
        if not bool(torch.isfinite(observerRotationWorld).all().item()):
            raise ValueError("observer rotation must contain only finite values")
        norm = observerRotationWorld.norm(dim=-1)
        if not bool(torch.allclose(
            norm,
            torch.ones_like(norm),
            atol=1e-5,
            rtol=1e-5)):
            raise ValueError("observer rotation must be a unit quaternion")

    def EffectiveObserverRotation(
        self,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        if self.observer_valid:
            return observerRotationWorld
        identity = torch.zeros_like(observerRotationWorld)
        identity[..., 3] = 1.0
        return identity

    @staticmethod
    def ExpandObserverRotation(
        observerRotationWorld: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        rotation = observerRotationWorld
        while rotation.dim() < target.dim():
            rotation = rotation.unsqueeze(1)
        return rotation


    @torch.no_grad()
    def ComposeSpatialFrame(self, parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        while parent.dim() < child.dim():
            parent = parent.unsqueeze(1)
        parent_q = self.NormalizeQuaternion(parent[..., 3:7])
        child_q = self.NormalizeQuaternion(child[..., 3:7])
        translation = parent[..., :3] + self.QuaternionRotate(parent_q, child[..., :3])
        rotation = F.normalize(self.QuaternionMultiply(
            parent_q, child_q).float(), dim=-1, eps=1e-6).to(child.dtype)
        pivot_index = rotation.abs().argmax(dim=-1, keepdim=True)
        pivot = rotation.gather(-1, pivot_index)
        rotation = torch.where(pivot < 0.0, -rotation, rotation)
        return torch.cat([translation, rotation], dim=-1)


    @torch.no_grad()
    def SpatialToWorldWithObserver(
        self,
        pose: torch.Tensor,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        rotation = self.ExpandObserverRotation(
            observerRotationWorld,
            pose)
        observer_q = self.NormalizeQuaternion(rotation)
        translation = self.QuaternionRotate(
            observer_q,
            pose[..., :3])
        pose_rotation = self.NormalizeQuaternion(pose[..., 3:7])
        world_rotation = self.NormalizeQuaternion(
            self.QuaternionMultiply(observer_q, pose_rotation))
        pivot = world_rotation.gather(
            -1,
            world_rotation.abs().argmax(dim=-1, keepdim=True))
        world_rotation = torch.where(
            pivot < 0.0,
            -world_rotation,
            world_rotation)
        return torch.cat([translation, world_rotation], dim=-1)


    @torch.no_grad()
    def MotionToWorldWithObserver(
        self,
        motionObserver: torch.Tensor,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        observer_q = self.NormalizeQuaternion(self.ExpandObserverRotation(
            observerRotationWorld,
            motionObserver))
        motion_observer_q = self.NormalizeQuaternion(motionObserver[..., 3:7])
        observer_q_inverse = torch.cat([-observer_q[..., :3], observer_q[..., 3:4]], dim=-1)
        motion_world_q = self.NormalizeQuaternion(self.QuaternionMultiply(
            self.QuaternionMultiply(observer_q, motion_observer_q),
            observer_q_inverse))
        pivot = motion_world_q.gather(
            -1, motion_world_q.abs().argmax(dim=-1, keepdim=True))
        motion_world_q = torch.where(pivot < 0.0, -motion_world_q, motion_world_q)
        return torch.cat([
            self.QuaternionRotate(observer_q, motionObserver[..., :3]),
            motion_world_q], dim=-1)


    @torch.no_grad()
    def SpatialToObserverWithObserver(
        self,
        poseWorld: torch.Tensor,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        observer_q = self.NormalizeQuaternion(self.ExpandObserverRotation(
            observerRotationWorld,
            poseWorld))
        observer_q_inverse = torch.cat(
            [-observer_q[..., :3], observer_q[..., 3:4]], dim=-1)
        translation = self.QuaternionRotate(
            observer_q_inverse,
            poseWorld[..., :3])
        rotation = self.NormalizeQuaternion(self.QuaternionMultiply(
            observer_q_inverse,
            self.NormalizeQuaternion(poseWorld[..., 3:7])))
        pivot = rotation.gather(
            -1, rotation.abs().argmax(dim=-1, keepdim=True))
        rotation = torch.where(pivot < 0.0, -rotation, rotation)
        return torch.cat([translation, rotation], dim=-1)


    @torch.no_grad()
    def MotionToObserverWithObserver(
        self,
        motionWorld: torch.Tensor,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        observer_q = self.NormalizeQuaternion(self.ExpandObserverRotation(
            observerRotationWorld,
            motionWorld))
        observer_q_inverse = torch.cat(
            [-observer_q[..., :3], observer_q[..., 3:4]], dim=-1)
        rotation = self.NormalizeQuaternion(self.QuaternionMultiply(
            self.QuaternionMultiply(
                observer_q_inverse,
                self.NormalizeQuaternion(motionWorld[..., 3:7])),
            observer_q))
        pivot = rotation.gather(
            -1, rotation.abs().argmax(dim=-1, keepdim=True))
        rotation = torch.where(pivot < 0.0, -rotation, rotation)
        return torch.cat([
            self.QuaternionRotate(observer_q_inverse, motionWorld[..., :3]),
            rotation], dim=-1)


    @torch.no_grad()
    def WeightedPointToObserverWithObserver(
        self,
        weightedPointWorld: torch.Tensor,
        pointWeight: torch.Tensor,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        observer_q = self.NormalizeQuaternion(self.ExpandObserverRotation(
            observerRotationWorld,
            weightedPointWorld))
        observer_q_inverse = torch.cat(
            [-observer_q[..., :3], observer_q[..., 3:4]], dim=-1)
        return self.QuaternionRotate(
            observer_q_inverse,
            weightedPointWorld)


    @torch.no_grad()
    def PairwiseRelationToObserverWithObserver(
        self,
        pairwiseRelationWorld: torch.Tensor,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        observer_q = self.NormalizeQuaternion(observerRotationWorld)
        observer_q_inverse = torch.cat(
            [-observer_q[..., :3], observer_q[..., 3:4]], dim=-1)
        observer_q_inverse = observer_q_inverse[:, None, None, :].expand(
            -1,
            pairwiseRelationWorld.size(1),
            pairwiseRelationWorld.size(2),
            -1)
        return torch.cat([
            self.QuaternionRotate(
                observer_q_inverse,
                pairwiseRelationWorld[..., :3]),
            pairwiseRelationWorld[..., 3:]], dim=-1)


    @torch.no_grad()
    def BuildContractModelPhysicalState(
        self,
        persistentState: Dict[str, torch.Tensor],
        observerRotationWorld: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.ValidateContractObserverRotation(observerRotationWorld)
        return self.BuildModelPhysicalStateWithObserver(
            persistentState,
            self.EffectiveObserverRotation(observerRotationWorld))

    @torch.no_grad()
    def BuildModelPhysicalStateWithObserver(
        self,
        persistentState: Dict[str, torch.Tensor],
        observerRotationWorld: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        model_state = {
            name: persistentState[name]
            for name in PERSISTENT_PHYSICAL_STATE_FIELDS
            if name not in PERSISTENT_WORLD_GEOMETRY_FIELDS}
        spatial_frame = self.SpatialToObserverWithObserver(
            persistentState["SpatialWorld"],
            observerRotationWorld)
        identity_spatial = torch.zeros_like(spatial_frame)
        identity_spatial[..., 6] = 1.0
        spatial_frame = torch.where(
            persistentState["GeometryValidMask"].unsqueeze(-1) > 0.5,
            spatial_frame,
            identity_spatial)
        model_state.update({
            "SpatialFrame": spatial_frame,
            "MotionObserverRaw": self.MotionToObserverWithObserver(
                persistentState["MotionWorldRaw"],
                observerRotationWorld),
            "CarrierMotionObserverRaw": self.MotionToObserverWithObserver(
                persistentState["CarrierMotionWorldRaw"],
                observerRotationWorld),
            "ArticulationMotionObserverRaw": self.MotionToObserverWithObserver(
                persistentState["ArticulationMotionWorldRaw"],
                observerRotationWorld),
            "ContactPointObserverRaw": self.WeightedPointToObserverWithObserver(
                persistentState["ContactPointWorldRaw"],
                persistentState["ContactProbRaw"],
                observerRotationWorld),
            "PairwiseRelationObserver": self.PairwiseRelationToObserverWithObserver(
                persistentState["PairwiseRelationWorld"],
                observerRotationWorld),
            "LevelProb": persistentState["Semantic"][..., :3],
            "ObjectClassProb": persistentState["Semantic"][
                ..., 3:3 + ModuleDim.PstObjectClasses],
            "PartClassProb": persistentState["Semantic"][
                ..., 3 + ModuleDim.PstObjectClasses:],})
        return model_state


    @torch.no_grad()
    def WeightedPointToWorldWithObserver(
        self,
        weightedPointObserver: torch.Tensor,
        pointWeight: torch.Tensor,
        observerRotationWorld: torch.Tensor,
    ) -> torch.Tensor:
        observer_q = self.NormalizeQuaternion(self.ExpandObserverRotation(
            observerRotationWorld,
            weightedPointObserver))
        weighted_point = self.QuaternionRotate(
            observer_q,
            weightedPointObserver)
        return weighted_point

    @torch.no_grad()
    def ExportPhysicalState(self) -> Dict[str, torch.Tensor]:
        return {
            "SlotState": self._pst_slot_state.detach().clone(),
            "SpatialWorld": self._pst_spatial_world.detach().clone(),
            "ARaw": self._pst_attribute.detach().clone(),
            "SlotPresence": self._pst_slot_presence.detach().clone(),
            "MphysRaw": self._pst_entity_prob.detach().clone(),
            "PerceptualPresence": self._pst_perceptual_presence.detach().clone(),
            "GeometryValidMask": self._pst_geometry_valid.detach().clone(),
            "PhysicalEntityProb": self._pst_entity_prob.detach().clone(),
            "PhysicalInteractionProb": self._pst_physical_interaction.detach().clone(),
            "RealmProb": self._pst_realm.detach().clone(),
            "MotionLayerProb": self._pst_motion_layer.detach().clone(),
            "LayerAgencyProb": self._pst_layer_agency.detach().clone(),
            "AgencyProb": self._pst_agency.detach().clone(),
            "BodyMembershipProb": self._pst_body_membership.detach().clone(),
            "SelfPartProb": self._pst_self_part.detach().clone(),
            "SelfPartSemantic": self._pst_self_part_semantic.detach().clone(),
            "IdentityKey": self._pst_identity_key.detach().clone(),
            "PairwiseRelationWorld": self._pst_pairwise_relation.detach().clone(),
            "PairRelationLastSeen": self._pst_pair_last_seen.detach().clone(),
            "ExternalRelationProbRaw": self._pst_external_relation.detach().clone(),
            "Semantic": self._pst_semantic.detach().clone(),
            "Size": self._pst_size.detach().clone(),
            "StateRaw": self._pst_state.detach().clone(),
            "AffordanceRaw": self._pst_affordance.detach().clone(),
            "MotionWorldRaw": self._pst_motion.detach().clone(),
            "CarrierMotionWorldRaw": self._pst_carrier_motion.detach().clone(),
            "ArticulationMotionWorldRaw": self._pst_articulation_motion.detach().clone(),
            "ContentMotionUV": self._pst_content_motion.detach().clone(),
            "ContentChangeProb": self._pst_content_change.detach().clone(),
            "MovingProbRaw": self._pst_moving.detach().clone(),
            "ContactProbRaw": self._pst_contact.detach().clone(),
            "ContactForceRaw": self._pst_contact_force.detach().clone(),
            "ContactPointWorldRaw": self._pst_contact_point.detach().clone(),
            "ParentProb": self._pst_parent.detach().clone(),
            "DisplaySurfaceProb": self._pst_display_surface.detach().clone(),
            "SurfaceParentProb": self._pst_surface_parent.detach().clone(),
            "SurfaceUV": self._pst_surface_uv.detach().clone(),
            "SurfaceUVConfidence": self._pst_surface_uv_confidence.detach().clone(),
            "VerificationConfidence": self._pst_verification.detach().clone(),
            "OntologyRelationProb": self._pst_ontology_relation.detach().clone(),
            "Visibility": self._pst_visibility.detach().clone(),
            "Occlusion": self._pst_occlusion.detach().clone(),
            "HasTextProb": self._pst_has_text.detach().clone(),
            "TextEmbed": self._pst_text.detach().clone(),
            "EntityTextSemantic": (
                self._pst_entity_text_semantic.detach().clone()),
            "EntityTextConfidence": (
                self._pst_entity_text_confidence.detach().clone()),
            "EntityTextRevision": (
                self._pst_entity_text_revision.detach().clone()),
            "EntityTextChanged": (
                self._pst_entity_text_changed.detach().clone()),
            "SymbolProb": self._pst_symbol.detach().clone(),
            "Observed": self._pst_observed.detach().clone(),
            "LastSeen": self._pst_last_seen.detach().clone(),
            "Step": self._pst_step.detach().clone(),}

    @torch.no_grad()
    def ExportPhysicalAssociations(self) -> Dict[str, torch.Tensor]:
        return {
            "ObservedToWorldSlot": (
                self._last_observed_to_world_slot.detach().clone()),
            "WorldEntityId": self._pst_entity_id.detach().clone(),
            "WorldSlotGeneration": (
                self._pst_slot_generation.detach().clone()),}

    @torch.no_grad()
    def UpdateEntityTextState(
        self,
        textState: Dict[str, torch.Tensor],
    ) -> None:
        required = {
            "EntityTextSemantic",
            "EntityTextConfidence",
            "EntityTextRevision",
            "EntityTextChanged",
            "EntityId",
            "SlotGeneration",}
        if set(textState) != required:
            raise ValueError("entity text state fields mismatch")
        B, K = self._pst_entity_id.shape
        expected = {
            "EntityTextSemantic": (B, K, self.entity_text_semantic_dim),
            "EntityTextConfidence": (B, K),
            "EntityTextRevision": (B, K),
            "EntityTextChanged": (B, K),
            "EntityId": (B, K),
            "SlotGeneration": (B, K),}
        for name, shape in expected.items():
            value = textState[name]
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(
                    f"entity text field {name!r} must have shape {shape}")
            if value.device != self._pst_entity_id.device:
                raise ValueError(
                    f"entity text field {name!r} must use the world device")
        integer_fields = {
            "EntityTextRevision", "EntityId", "SlotGeneration"}
        for name in integer_fields:
            if textState[name].dtype != torch.long:
                raise TypeError(
                    f"entity text field {name!r} must use torch.long")
        if textState["EntityTextChanged"].dtype != torch.bool:
            raise TypeError(
                "entity text field 'EntityTextChanged' must use torch.bool")
        for name in ("EntityTextSemantic", "EntityTextConfidence"):
            if not textState[name].is_floating_point():
                raise TypeError(
                    f"entity text field {name!r} must be floating point")
            if not bool(torch.isfinite(textState[name]).all().item()):
                raise ValueError(
                    f"entity text field {name!r} must be finite")
        confidence = textState["EntityTextConfidence"]
        if bool(((confidence < 0.0) | (confidence > 1.0)).any().item()):
            raise ValueError("entity text confidence must be a probability")
        if bool((textState["EntityTextRevision"] < 0).any().item()):
            raise ValueError("entity text revision must be non-negative")
        live = self._pst_entity_id >= 0
        if bool((live & (
            textState["EntityId"] != self._pst_entity_id)).any().item()):
            raise ValueError("entity text identity does not match world identity")
        if bool((live & (
            textState["SlotGeneration"]
            != self._pst_slot_generation)).any().item()):
            raise ValueError(
                "entity text generation does not match world generation")
        self._pst_entity_text_semantic.copy_(torch.where(
            live.unsqueeze(-1),
            textState["EntityTextSemantic"].to(self._pst_entity_text_semantic.dtype),
            torch.zeros_like(self._pst_entity_text_semantic)))
        self._pst_entity_text_confidence.copy_(torch.where(
            live,
            confidence.to(self._pst_entity_text_confidence.dtype),
            torch.zeros_like(self._pst_entity_text_confidence)))
        self._pst_entity_text_revision.copy_(torch.where(
            live,
            textState["EntityTextRevision"],
            torch.zeros_like(self._pst_entity_text_revision)))
        self._pst_entity_text_changed.copy_(
            textState["EntityTextChanged"] & live)
        self._pst_has_text.copy_(torch.maximum(
            self._pst_has_text,
            self._pst_entity_text_confidence))



    @torch.no_grad()
    def ImportPhysicalState(self, physicalState: Dict[str, torch.Tensor]) -> None:
        reference = physicalState["SlotState"]
        self.EnsurePhysicalMemory(int(reference.size(0)))

        def CopyFrom(buffer: torch.Tensor, key: str) -> None:
            buffer.copy_(physicalState[key])

        self._pst_slot_state.zero_()
        self._pst_spatial_world.zero_()
        self._pst_attribute.zero_()
        self._pst_slot_presence.zero_()
        self._pst_entity_prob.zero_()
        self._pst_entity_id.fill_(-1)
        self._pst_slot_generation.zero_()
        self._pst_next_entity_id.zero_()
        self._last_observed_to_world_slot = (
            self._last_observed_to_world_slot.new_full(
                (int(reference.size(0)), 0), -1))
        self._pst_perceptual_presence.zero_()
        self._pst_geometry_valid.zero_()
        self._pst_physical_interaction.zero_()
        self._pst_realm.zero_()
        self._pst_motion_layer.zero_()
        self._pst_layer_agency.zero_()
        self._pst_agency.zero_()
        self._pst_body_membership.zero_()
        self._pst_self_part.zero_()
        self._pst_self_part_semantic.zero_()
        self._pst_identity_key.zero_()
        self._pst_pairwise_relation.zero_()
        self._pst_pair_last_seen.zero_()
        self._pst_external_relation.zero_()
        self._pst_semantic.zero_()
        self._pst_size.zero_()
        self._pst_state.zero_()
        self._pst_affordance.zero_()
        self._pst_motion.zero_()
        self._pst_carrier_motion.zero_()
        self._pst_articulation_motion.zero_()
        self._pst_content_motion.zero_()
        self._pst_content_change.zero_()
        self._pst_moving.zero_()
        self._pst_contact.zero_()
        self._pst_contact_force.zero_()
        self._pst_contact_point.zero_()
        self._pst_parent.zero_()
        self._pst_display_surface.zero_()
        self._pst_surface_parent.zero_()
        self._pst_surface_uv.zero_()
        self._pst_surface_uv_confidence.zero_()
        self._pst_verification.zero_()
        self._pst_ontology_relation.zero_()
        self._pst_visibility.zero_()
        self._pst_occlusion.zero_()
        self._pst_has_text.zero_()
        self._pst_text.zero_()
        self._pst_entity_text_semantic.zero_()
        self._pst_entity_text_confidence.zero_()
        self._pst_entity_text_revision.zero_()
        self._pst_entity_text_changed.zero_()
        self._pst_symbol.zero_()
        self._pst_observed.zero_()
        self._pst_last_seen.zero_()
        CopyFrom(self._pst_slot_state, "SlotState")
        CopyFrom(self._pst_spatial_world, "SpatialWorld")
        CopyFrom(self._pst_attribute, "ARaw")
        CopyFrom(self._pst_slot_presence, "SlotPresence")
        CopyFrom(self._pst_entity_prob, "MphysRaw")
        CopyFrom(self._pst_perceptual_presence, "PerceptualPresence")
        CopyFrom(self._pst_geometry_valid, "GeometryValidMask")
        CopyFrom(self._pst_physical_interaction, "PhysicalInteractionProb")
        CopyFrom(self._pst_realm, "RealmProb")
        CopyFrom(self._pst_motion_layer, "MotionLayerProb")
        CopyFrom(self._pst_layer_agency, "LayerAgencyProb")
        CopyFrom(self._pst_agency, "AgencyProb")
        CopyFrom(self._pst_body_membership, "BodyMembershipProb")
        CopyFrom(self._pst_self_part, "SelfPartProb")
        CopyFrom(self._pst_self_part_semantic, "SelfPartSemantic")
        CopyFrom(self._pst_identity_key, "IdentityKey")
        CopyFrom(self._pst_pairwise_relation, "PairwiseRelationWorld")
        CopyFrom(self._pst_pair_last_seen, "PairRelationLastSeen")
        CopyFrom(self._pst_external_relation, "ExternalRelationProbRaw")
        CopyFrom(self._pst_semantic, "Semantic")
        CopyFrom(self._pst_size, "Size")
        CopyFrom(self._pst_state, "StateRaw")
        CopyFrom(self._pst_affordance, "AffordanceRaw")
        CopyFrom(self._pst_motion, "MotionWorldRaw")
        CopyFrom(self._pst_carrier_motion, "CarrierMotionWorldRaw")
        CopyFrom(self._pst_articulation_motion, "ArticulationMotionWorldRaw")
        CopyFrom(self._pst_content_motion, "ContentMotionUV")
        CopyFrom(self._pst_content_change, "ContentChangeProb")
        CopyFrom(self._pst_moving, "MovingProbRaw")
        CopyFrom(self._pst_contact, "ContactProbRaw")
        CopyFrom(self._pst_contact_force, "ContactForceRaw")
        CopyFrom(self._pst_contact_point, "ContactPointWorldRaw")
        CopyFrom(self._pst_parent, "ParentProb")
        CopyFrom(self._pst_display_surface, "DisplaySurfaceProb")
        CopyFrom(self._pst_surface_parent, "SurfaceParentProb")
        CopyFrom(self._pst_surface_uv, "SurfaceUV")
        CopyFrom(self._pst_surface_uv_confidence, "SurfaceUVConfidence")
        CopyFrom(self._pst_verification, "VerificationConfidence")
        CopyFrom(self._pst_ontology_relation, "OntologyRelationProb")
        CopyFrom(self._pst_visibility, "Visibility")
        CopyFrom(self._pst_occlusion, "Occlusion")
        CopyFrom(self._pst_has_text, "HasTextProb")
        CopyFrom(self._pst_text, "TextEmbed")
        CopyFrom(self._pst_entity_text_semantic, "EntityTextSemantic")
        CopyFrom(self._pst_entity_text_confidence, "EntityTextConfidence")
        CopyFrom(self._pst_entity_text_revision, "EntityTextRevision")
        CopyFrom(self._pst_entity_text_changed, "EntityTextChanged")
        CopyFrom(self._pst_symbol, "SymbolProb")
        CopyFrom(self._pst_observed, "Observed")
        CopyFrom(self._pst_last_seen, "LastSeen")
        CopyFrom(self._pst_step, "Step")
        for b in range(int(reference.size(0))):
            active = torch.nonzero(
                self._pst_slot_presence[b] > 1e-6,
                as_tuple=False).flatten()
            if active.numel() == 0:
                continue
            entity_ids = torch.arange(
                active.numel(),
                device=self._pst_entity_id.device,
                dtype=torch.long)
            self._pst_entity_id[b, active] = entity_ids
            self._pst_slot_generation[b, active] = 1
            self._pst_next_entity_id[b] = active.numel()

    @staticmethod
    def MatchLegalPhysicalSlots(
        cost: torch.Tensor,
        legal: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cost.ndim != 2 or legal.shape != cost.shape:
            raise ValueError("cost and legal must have the same two-dimensional shape")
        active_count, incoming_count = int(cost.size(0)), int(cost.size(1))
        assignment_cost = cost.float()
        augmented_size = active_count + incoming_count
        illegal_cost = 1e6
        unmatched_cost = 1.0
        augmented = assignment_cost.new_full(
            (augmented_size, augmented_size), illegal_cost)
        augmented[:active_count, :incoming_count] = torch.where(
            legal,
            assignment_cost,
            assignment_cost.new_full(assignment_cost.shape, illegal_cost))
        augmented[:active_count, incoming_count:] = unmatched_cost
        augmented[active_count:, :incoming_count] = unmatched_cost
        augmented[active_count:, incoming_count:] = 0.0
        assigned_rows, assigned_cols = HungarianAssignment(augmented)
        real_assignment = (assigned_rows < active_count) & (assigned_cols < incoming_count)
        return assigned_rows[real_assignment], assigned_cols[real_assignment]

    def ValidatePhysicalUpdateInputsWithObserver(
        self,
        observedPst: Dict[str, torch.Tensor],
        observerRotationWorld: torch.Tensor,
        embodimentState: torch.Tensor,
        ) -> Tuple[int, int]:
        if not isinstance(observedPst, dict):
            raise TypeError("observedPst must be a dictionary of tensors")
        missing = sorted(set(OBSERVED_PHYSICAL_STATE_FIELDS).difference(observedPst))
        if missing:
            raise KeyError(f"observedPst is missing required fields: {missing}")

        slot_state = observedPst["SlotState"]
        if not torch.is_tensor(slot_state) or slot_state.ndim != 3:
            actual = tuple(slot_state.shape) if torch.is_tensor(slot_state) else type(slot_state).__name__
            raise ValueError(f"observedPst['SlotState'] must have shape [B, K, D], got {actual}")
        B, K = int(slot_state.size(0)), int(slot_state.size(1))
        if B <= 0 or K <= 0:
            raise ValueError("observedPst must contain at least one batch row and one slot")

        expected_shapes = {
            "SlotState": (B, K, self.physical_slot_dim),
            "SpatialFrame": (B, K, self.physical_spatial_dim),
            "ARaw": (B, K, self.physical_attr_dim),
            "ObservedSlotMask": (B, K),
            "MphysRaw": (B, K),
            "PerceptualPresence": (B, K),
            "GeometryValidMask": (B, K),
            "PhysicalEntityProb": (B, K),
            "PhysicalInteractionProb": (B, K),
            "RealmProb": (B, K, ModuleDim.PstRealmClasses),
            "MotionLayerProb": (B, K, ModuleDim.PstMotionLayerClasses),
            "LayerAgencyProb": (B, K, ModuleDim.PstMotionLayerClasses, ModuleDim.PstAgencyClasses),
            "AgencyProb": (B, K, ModuleDim.PstAgencyClasses),
            "BodyMembershipProb": (B, K),
            "SelfPartProb": (B, K, self.self_part_count),
            "SelfPartSemantic": (
                B, K, self.self_part_semantic_dim),
            "IdentityKey": (B, K, self.physical_id_dim),
            "Semantic": (B, K, self.physical_semantic_dim),
            "ExternalRelationProbRaw": (B, K, self.physical_relation_classes),
            "Size": (B, K, 3),
            "StateRaw": (B, K, self.physical_state_dim),
            "AffordanceRaw": (B, K, self.physical_affordance_dim),
            "MotionObserverRaw": (B, K, self.physical_spatial_dim),
            "CarrierMotionObserverRaw": (B, K, self.physical_spatial_dim),
            "ArticulationMotionObserverRaw": (B, K, self.physical_spatial_dim),
            "ContentMotionUV": (B, K, 2),
            "ContentChangeProb": (B, K),
            "MovingProbRaw": (B, K),
            "ContactProbRaw": (B, K),
            "ContactForceRaw": (B, K, 2),
            "ContactPointObserverRaw": (B, K, 3),
            "Visibility": (B, K),
            "Occlusion": (B, K),
            "HasTextProb": (B, K),
            "TextEmbed": (B, K, self.physical_text_dim),
            "SymbolProb": (B, K, self.physical_symbol_dim),
            "PairwiseRelationObserver": (B, K, K, self.physical_rel_dim),
            "ParentProb": (B, K, K),}
        expected_shapes.update({
            "DisplaySurfaceProb": (B, K),
            "SurfaceParentProb": (B, K, K + 1),
            "SurfaceUV": (B, K, 2),
            "SurfaceUVConfidence": (B, K),
            "VerificationConfidence": (B, K),
            "OntologyRelationProb": (
                B, K, K, ModuleDim.PstOntologyRelationClasses),})
        external_shapes = {
            "observerRotationWorld": (
                observerRotationWorld,
                (B, ROTATION_QUATERNION_DIM)),
            "embodimentState": (
                embodimentState,
                (B, self.embodiment_state_dim)),}
        tensors = {
            **{name: observedPst[name] for name in expected_shapes},
            **{name: value for name, (value, _) in external_shapes.items()}}
        shapes = {
            **expected_shapes,
            **{name: shape for name, (_, shape) in external_shapes.items()}}
        model_device, model_dtype = self.device, self.dtype
        for name, value in tensors.items():
            if not torch.is_tensor(value):
                raise TypeError(f"{name} must be a tensor")
            if tuple(value.shape) != shapes[name]:
                raise ValueError(
                    f"{name} must have shape {shapes[name]}, got {tuple(value.shape)}")
            if value.device != model_device:
                raise ValueError(f"{name} must be on {model_device}, got {value.device}")
            if value.dtype != model_dtype:
                raise ValueError(f"{name} must have dtype {model_dtype}, got {value.dtype}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")
        if not torch.equal(
            observedPst["MphysRaw"],
            observedPst["PhysicalEntityProb"]):
            raise ValueError(
                "MphysRaw must be exactly PhysicalEntityProb")
        if bool((
            observedPst["PhysicalInteractionProb"]
            > observedPst["MphysRaw"] + 1e-6).any().item()):
            raise ValueError(
                "PhysicalInteractionProb must not exceed PhysicalEntityProb")
        realm_class = observedPst["RealmProb"].argmax(dim=-1)
        virtual_or_effect = (
            realm_class.eq(VirtualRealmIndex)
            | realm_class.eq(EffectRealmIndex))
        invalid_independent_geometry = virtual_or_effect & (
            observedPst["GeometryValidMask"].gt(0.0)
            | observedPst["MphysRaw"].gt(0.0)
            | observedPst["PhysicalInteractionProb"].gt(0.0))
        if bool(invalid_independent_geometry.any().item()):
            raise ValueError(
                "virtual/effect slots cannot own independent 3D geometry or "
                "physical interaction state")
        return B, K


    @torch.no_grad()
    def UpdateContractPhysicalState(
        self,
        observedPst: Dict[str, torch.Tensor],
        observerRotationWorld: torch.Tensor,
        embodimentState: torch.Tensor,
        observerValid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self.ValidateContractObserverRotation(observerRotationWorld)
        return self.UpdatePhysicalStateWithObserver(
            observedPst,
            self.EffectiveObserverRotation(observerRotationWorld),
            embodimentState,
            observerValid)

    @torch.no_grad()
    def UpdatePhysicalStateWithObserver(
        self,
        observedPst: Dict[str, torch.Tensor],
        observerRotationWorld: torch.Tensor,
        embodimentState: torch.Tensor,
        observerValid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, _ = self.ValidatePhysicalUpdateInputsWithObserver(
            observedPst,
            observerRotationWorld,
            embodimentState)
        if not torch.is_tensor(observerValid):
            raise TypeError("observer validity must be a tensor")
        if tuple(observerValid.shape) != (B,):
            raise ValueError("observer validity does not match batch")
        if observerValid.device != observerRotationWorld.device:
            raise ValueError("observer validity device does not match rotation")
        if observerValid.dtype != torch.bool:
            raise TypeError("observer validity must be boolean")
        runtime_observer_valid = observerValid & bool(self.observer_valid)
        self.EnsurePhysicalMemory(B)
        observed_p_world = self.SpatialToWorldWithObserver(
            observedPst["SpatialFrame"].detach(),
            observerRotationWorld)
        geometry_valid = (
            observedPst["GeometryValidMask"].detach()
            * runtime_observer_valid.to(
                observedPst["GeometryValidMask"].dtype).unsqueeze(-1))
        identity_spatial = torch.zeros_like(observed_p_world)
        identity_spatial[..., 6] = 1.0
        observed_p_world = torch.where(
            geometry_valid.unsqueeze(-1) > 0.5,
            observed_p_world,
            identity_spatial)
        observed_motion_world = self.MotionToWorldWithObserver(
            observedPst["MotionObserverRaw"].detach(),
            observerRotationWorld)
        observed_carrier_world = self.MotionToWorldWithObserver(
            observedPst["CarrierMotionObserverRaw"].detach(),
            observerRotationWorld)
        observed_articulation_world = self.MotionToWorldWithObserver(
            observedPst["ArticulationMotionObserverRaw"].detach(),
            observerRotationWorld)
        observed_motion_world = torch.where(
            geometry_valid.unsqueeze(-1) > 0.5,
            observed_motion_world,
            identity_spatial)
        observed_carrier_world = torch.where(
            geometry_valid.unsqueeze(-1) > 0.5,
            observed_carrier_world,
            identity_spatial)
        observed_articulation_world = torch.where(
            geometry_valid.unsqueeze(-1) > 0.5,
            observed_articulation_world,
            identity_spatial)
        observed_contact_world = self.WeightedPointToWorldWithObserver(
            observedPst["ContactPointObserverRaw"].detach(),
            observedPst["ContactProbRaw"].detach(),
            observerRotationWorld)
        observed_contact_world = (
            observed_contact_world
            * geometry_valid.unsqueeze(-1))
        observed_m = observedPst["ObservedSlotMask"].detach()
        observed_m_phys = observedPst["MphysRaw"].detach() # [B, K]
        observed_presence = observedPst["PerceptualPresence"].detach()
        observed_c = observedPst["IdentityKey"].detach()

        self._pst_step.add_(1)
        self._pst_slot_presence.mul_(self.physical_confidence_decay)
        effect_probability = self._pst_realm[..., EffectRealmIndex]
        self._pst_slot_presence.mul_(1.0 - 0.20 * effect_probability)
        self._pst_observed.zero_()

        self._last_observed_to_world_slot = (
            self._last_observed_to_world_slot.new_full(
                observed_m.shape, -1))
        incoming_to_memory = self._last_observed_to_world_slot

        def WriteSlot(b: int, source_idx: int, target_idx: int, reset_pairwise_relations: bool) -> None:
            if reset_pairwise_relations:
                self._pst_has_text[b, target_idx].zero_()
                self._pst_text[b, target_idx].zero_()
                self._pst_entity_text_semantic[b, target_idx].zero_()
                self._pst_entity_text_confidence[b, target_idx].zero_()
                self._pst_entity_text_revision[b, target_idx].zero_()
                self._pst_entity_text_changed[b, target_idx].zero_()
                self._pst_symbol[b, target_idx].zero_()
                self._pst_pairwise_relation[b, target_idx].zero_()
                self._pst_pairwise_relation[b, :, target_idx].zero_()
                self._pst_pair_last_seen[b, target_idx].zero_()
                self._pst_pair_last_seen[b, :, target_idx].zero_()
                self._pst_parent[b, target_idx].zero_()
                self._pst_parent[b, :, target_idx].zero_()
                self._pst_surface_parent[b, target_idx].zero_()
                self._pst_surface_parent[b, :, target_idx].zero_()
                self._pst_ontology_relation[b, target_idx].zero_()
                self._pst_ontology_relation[b, :, target_idx].zero_()
            if reset_pairwise_relations or self._pst_entity_id[b, target_idx] < 0:
                self._pst_entity_id[b, target_idx] = self._pst_next_entity_id[b]
                self._pst_next_entity_id[b].add_(1)
                self._pst_slot_generation[b, target_idx].add_(1)
            incoming_to_memory[b, source_idx] = target_idx
            self._pst_slot_state[b, target_idx] = observedPst["SlotState"][b, source_idx].detach()
            self._pst_spatial_world[b, target_idx] = observed_p_world[b, source_idx]
            self._pst_attribute[b, target_idx] = observedPst["ARaw"][b, source_idx].detach()
            self._pst_slot_presence[b, target_idx] = observed_presence[b, source_idx]
            self._pst_entity_prob[b, target_idx] = observed_m_phys[b, source_idx]
            self._pst_perceptual_presence[b, target_idx] = observed_presence[b, source_idx]
            self._pst_geometry_valid[b, target_idx] = geometry_valid[b, source_idx]
            self._pst_physical_interaction[b, target_idx] = observedPst["PhysicalInteractionProb"][b, source_idx].detach()
            self._pst_realm[b, target_idx] = observedPst["RealmProb"][b, source_idx].detach()
            self._pst_motion_layer[b, target_idx] = observedPst["MotionLayerProb"][b, source_idx].detach()
            self._pst_layer_agency[b, target_idx] = observedPst["LayerAgencyProb"][b, source_idx].detach()
            self._pst_agency[b, target_idx] = observedPst["AgencyProb"][b, source_idx].detach()
            self._pst_self_part_semantic[b, target_idx] = observedPst[
                "SelfPartSemantic"][b, source_idx].detach()
            self._pst_body_membership[b, target_idx] = observedPst["BodyMembershipProb"][b, source_idx].detach()
            self._pst_self_part[b, target_idx] = observedPst["SelfPartProb"][b, source_idx].detach()
            self._pst_identity_key[b, target_idx] = observed_c[b, source_idx]
            self._pst_semantic[b, target_idx] = observedPst["Semantic"][b, source_idx].detach()
            self._pst_external_relation[b, target_idx] = observedPst["ExternalRelationProbRaw"][b, source_idx].detach()
            self._pst_size[b, target_idx] = observedPst["Size"][b, source_idx].detach()
            self._pst_state[b, target_idx] = observedPst["StateRaw"][b, source_idx].detach()
            self._pst_affordance[b, target_idx] = observedPst["AffordanceRaw"][b, source_idx].detach()
            self._pst_motion[b, target_idx] = observed_motion_world[b, source_idx]
            self._pst_carrier_motion[b, target_idx] = observed_carrier_world[b, source_idx]
            self._pst_articulation_motion[b, target_idx] = observed_articulation_world[b, source_idx]
            self._pst_content_motion[b, target_idx] = observedPst["ContentMotionUV"][b, source_idx].detach()
            self._pst_content_change[b, target_idx] = observedPst["ContentChangeProb"][b, source_idx].detach()
            self._pst_moving[b, target_idx] = observedPst["MovingProbRaw"][b, source_idx].detach()
            self._pst_contact[b, target_idx] = observedPst["ContactProbRaw"][b, source_idx].detach()
            self._pst_contact_force[b, target_idx] = observedPst["ContactForceRaw"][b, source_idx].detach()
            self._pst_contact_point[b, target_idx] = observed_contact_world[b, source_idx]
            self._pst_visibility[b, target_idx] = observedPst["Visibility"][b, source_idx].detach()
            self._pst_occlusion[b, target_idx] = observedPst["Occlusion"][b, source_idx].detach()
            self._pst_has_text[b, target_idx] = observedPst["HasTextProb"][b, source_idx].detach()
            self._pst_text[b, target_idx] = observedPst["TextEmbed"][b, source_idx].detach()
            self._pst_symbol[b, target_idx] = observedPst["SymbolProb"][b, source_idx].detach()
            self._pst_display_surface[b, target_idx] = observedPst["DisplaySurfaceProb"][b, source_idx].detach()
            self._pst_surface_uv[b, target_idx] = observedPst["SurfaceUV"][b, source_idx].detach()
            self._pst_surface_uv_confidence[b, target_idx] = observedPst["SurfaceUVConfidence"][b, source_idx].detach()
            self._pst_verification[b, target_idx] = observedPst["VerificationConfidence"][b, source_idx].detach()
            self._pst_observed[b, target_idx] = True
            self._pst_last_seen[b, target_idx] = self._pst_step[b]

        for b in range(B):
            incoming = torch.nonzero(
                observed_m[b] >= self.physical_observation_threshold,
                as_tuple=False).flatten()
            if incoming.numel() == 0:
                continue

            incoming = incoming[torch.argsort(observed_m[b, incoming], descending=True)]
            assigned_source = torch.zeros(observed_m.size(1), device=observed_m.device, dtype=torch.bool)
            active_idx = torch.nonzero(self._pst_slot_presence[b] > 1e-6, as_tuple=False).flatten()

            if active_idx.numel() > 0:
                similarity = torch.matmul(self._pst_identity_key[b, active_idx], observed_c[b, incoming].t())
                active_realm = self._pst_realm[
                    b, active_idx].argmax(dim=-1)
                incoming_realm = observedPst["RealmProb"][
                    b, incoming].argmax(dim=-1)
                realm_compatible = (
                    active_realm.unsqueeze(1)
                    == incoming_realm.unsqueeze(0))
                active_self = active_realm.eq(SelfRealmIndex)
                incoming_self = incoming_realm.eq(SelfRealmIndex)
                active_self_part = self._pst_self_part[
                    b, active_idx].argmax(dim=-1)
                incoming_self_part = observedPst["SelfPartProb"][
                    b, incoming].argmax(dim=-1)
                self_part_compatible = (
                    active_self_part.unsqueeze(1)
                    == incoming_self_part.unsqueeze(0))
                self_identity_compatible = (
                    ~(active_self.unsqueeze(1) | incoming_self.unsqueeze(0))
                    | (
                        active_self.unsqueeze(1)
                        & incoming_self.unsqueeze(0)
                        & self_part_compatible))
                pose_delta = (
                    self._pst_spatial_world[b, active_idx, :3].unsqueeze(1)
                    - observed_p_world[b, incoming, :3].unsqueeze(0))
                pose_comparable = (
                    self._pst_geometry_valid[b, active_idx].unsqueeze(1)
                    * geometry_valid[b, incoming].unsqueeze(0))
                pose_cost = (
                    torch.tanh(pose_delta.norm(dim=-1)) * pose_comparable)
                last_seen_age = (self._pst_step[b].float() - self._pst_last_seen[b, active_idx].float()).clamp_min(0.0)
                age_cost = last_seen_age / (last_seen_age + 1.0)
                cost = (
                    (1.0 - similarity)
                    + 0.25 * pose_cost
                    + 0.05 * age_cost.unsqueeze(1)
                    - 0.10 * self._pst_slot_presence[b, active_idx].unsqueeze(1)
                    - 0.05 * observed_m[b, incoming].unsqueeze(0)) # [N_active, N_incoming]
                active_local, incoming_local = self.MatchLegalPhysicalSlots(
                    cost,
                    (
                        similarity >= self.physical_identity_threshold
                    ) & realm_compatible & self_identity_compatible)

                for active_pos_t, incoming_pos_t in zip(active_local, incoming_local):
                    active_pos = int(active_pos_t.item())
                    incoming_pos = int(incoming_pos_t.item())
                    source_idx = int(incoming[incoming_pos].item())
                    target_idx = int(active_idx[active_pos].item())
                    assigned_source[source_idx] = True
                    WriteSlot(b, source_idx, target_idx, False)

            for source_idx in incoming[~assigned_source[incoming]].tolist():
                empty_slots = torch.nonzero(self._pst_slot_presence[b] <= 1e-6, as_tuple=False).flatten()
                if empty_slots.numel() > 0:
                    target_idx = int(empty_slots[0].item())
                else:
                    slot_age = self._pst_step[b].float() - self._pst_last_seen[b].float()
                    physical_importance = self._pst_slot_presence[b] * self._pst_entity_prob[b]
                    interaction_importance = 0.5 * self._pst_moving[b] + 0.5 * self._pst_contact[b]
                    semantic_importance = (
                        0.5 * self._pst_visibility[b]
                        + 0.25 * self._pst_has_text[b]
                        + 0.25 * self._pst_symbol[b].amax(dim=-1))
                    parent_importance = torch.maximum(
                        self._pst_parent[b].amax(dim=0),
                        self._pst_parent[b].amax(dim=1))
                    relation_importance = torch.maximum(
                        parent_importance,
                        self._pst_external_relation[b].amax(dim=-1))
                    slot_importance = (
                        0.45 * physical_importance
                        + 0.25 * interaction_importance
                        + 0.20 * semantic_importance
                        + 0.10 * relation_importance)
                    replacement_score = slot_age * (1.0 - slot_importance)
                    replaceable = self._pst_body_membership[b] <= 0.5
                    if not bool(replaceable.any().item()):
                        continue
                    replacement_score = replacement_score.masked_fill(
                        ~replaceable,
                        torch.finfo(replacement_score.dtype).min)
                    target_idx = int(torch.argmax(replacement_score).item())
                WriteSlot(b, int(source_idx), target_idx, True)

            mapped_sources = torch.nonzero(incoming_to_memory[b] >= 0, as_tuple=False).flatten()
            if mapped_sources.numel() > 0:
                memory_targets = incoming_to_memory[b, mapped_sources]
                mem_grid = memory_targets.view(-1, 1).expand(-1, memory_targets.numel())
                src_grid = mapped_sources.view(-1, 1).expand(-1, mapped_sources.numel())
                self._pst_pairwise_relation[b, mem_grid, mem_grid.t()] = observedPst["PairwiseRelationObserver"][b, src_grid, src_grid.t()].detach()
                self._pst_pair_last_seen[b, mem_grid, mem_grid.t()] = self._pst_step[b]
                self._pst_parent[b, mem_grid, mem_grid.t()] = observedPst["ParentProb"][b, src_grid, src_grid.t()].detach()
                self._pst_ontology_relation[b, mem_grid, mem_grid.t()] = observedPst["OntologyRelationProb"][b, src_grid, src_grid.t()].detach()
                observed_surface_parent = observedPst["SurfaceParentProb"][b]
                for source_idx in mapped_sources.tolist():
                    target_idx = int(incoming_to_memory[b, source_idx].item())
                    self._pst_surface_parent[b, target_idx].zero_()
                    parent_sources = torch.nonzero(
                        incoming_to_memory[b] >= 0,
                        as_tuple=False).flatten()
                    parent_targets = incoming_to_memory[b, parent_sources]
                    self._pst_surface_parent[b, target_idx, parent_targets] = (
                        observed_surface_parent[source_idx, parent_sources])
                    unmapped_parent = incoming_to_memory[b] < 0
                    self._pst_surface_parent[b, target_idx, -1] = (
                        observed_surface_parent[source_idx, -1]
                        + observed_surface_parent[
                            source_idx, :observed_m.size(1)][unmapped_parent].sum())

        relative = self._pst_spatial_world[..., :3].unsqueeze(1) - self._pst_spatial_world[..., :3].unsqueeze(2)
        distance = relative.norm(dim=-1, keepdim=True)
        off_diagonal = ~torch.eye(self.physical_slots, device=self._pst_slot_presence.device, dtype=torch.bool)
        semantic_pair_valid = (
            (self._pst_slot_presence > 1e-6).unsqueeze(2)
            & (self._pst_slot_presence > 1e-6).unsqueeze(1)
            & off_diagonal.unsqueeze(0)).unsqueeze(-1)
        geometry_pair_valid = (
            semantic_pair_valid
            & (self._pst_geometry_valid > 0.5).unsqueeze(2).unsqueeze(-1)
            & (self._pst_geometry_valid > 0.5).unsqueeze(1).unsqueeze(-1))
        self._pst_pairwise_relation[..., :4].masked_fill_(
            ~geometry_pair_valid, 0.0)
        self._pst_pairwise_relation[..., 4:].masked_fill_(
            ~semantic_pair_valid, 0.0)
        self._pst_pair_last_seen.masked_fill_(
            ~semantic_pair_valid.squeeze(-1), 0)
        self._pst_pairwise_relation[..., :3] = (
            relative * geometry_pair_valid.to(relative.dtype))
        self._pst_pairwise_relation[..., 3:4] = (
            distance * geometry_pair_valid.to(distance.dtype))
        self._pst_parent.masked_fill_(
            ~semantic_pair_valid.squeeze(-1), 0.0)
        self._pst_ontology_relation.masked_fill_(
            ~semantic_pair_valid, 0.0)

        return self.BuildModelPhysicalStateWithObserver(
            self.ExportPhysicalState(),
            observerRotationWorld)

    def PhysicalSlotSummary(self, S: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        weight = M.unsqueeze(-1)
        return (S * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)


    def EncodeContractTransition(
        self,
        feedbackPacket: BrainFeedbackPacket,
    ) -> torch.Tensor:
        if self.contract_embodiment_adapter is None:
            raise RuntimeError(
                "world model is not bound to an embodiment contract")
        transition = self.contract_embodiment_adapter.EncodeTransition(
            feedbackPacket)
        encoded = transition["EncodedTransition"]
        if int(encoded.size(-1)) != self.embodiment_state_dim:
            raise RuntimeError(
                "contract transition does not match world physical width")
        return encoded

    def EncodeContractExecutionAction(
        self,
        action: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
    ) -> torch.Tensor:
        return self.contract_embodiment_adapter.EncodeExecutionAction(
            action,
            feedbackPacket.execution_status,
            feedbackPacket.execution_relevant,
            feedbackPacket.execution_known,
            feedbackPacket.execution_result_known,
            feedbackPacket.hard_stop,
            feedbackPacket.help_accepted,
            feedbackPacket.applied_target_active)

    def EncodeAssumedAppliedAction(
        self,
        action: torch.Tensor,
        targetActive: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, endpoint_count = targetActive.shape
        execution_status = torch.full(
            (batch_size, endpoint_count),
            int(SlotExecutionStatus.APPLIED),
            device=action.device,
            dtype=torch.long)
        row_mask = torch.ones(
            batch_size,
            device=action.device,
            dtype=torch.bool)
        return self.contract_embodiment_adapter.EncodeExecutionAction(
            action,
            execution_status,
            targetActive,
            torch.ones_like(targetActive),
            row_mask,
            torch.zeros_like(row_mask),
            torch.zeros_like(row_mask),
            targetActive)

    def EncodeContractEmbodiment(
        self,
        feedbackPacket: BrainFeedbackPacket,
    ) -> Dict[str, torch.Tensor]:
        if self.contract_embodiment_adapter is None:
            raise RuntimeError(
                "world model is not bound to an embodiment contract")
        return self.contract_embodiment_adapter.EncodeTransition(
            feedbackPacket)




    def PredictContractFeedback(
        self,
        priorWorldState: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.contract_embodiment_adapter is None:
            raise RuntimeError(
                "world model is not bound to an embodiment contract")
        return self.contract_embodiment_adapter.PredictFeedback(
            priorWorldState)

    def ComputeContractFeedbackLoss(
        self,
        prediction: Dict[str, torch.Tensor],
        feedbackPacket: BrainFeedbackPacket,
        sampleMask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.contract_embodiment_adapter is None:
            raise RuntimeError(
                "world model is not bound to an embodiment contract")
        return self.contract_embodiment_adapter.ComputeFeedbackLoss(
            prediction,
            feedbackPacket,
            sampleMask=sampleMask)

    def BuildEmbodiedAction(
        self,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(observerMotionValid):
            raise TypeError("observer motion validity must be a tensor")
        if tuple(observerMotionValid.shape) != (observerMotion.size(0),):
            raise ValueError("observer motion validity must be [B]")
        if observerMotionValid.device != observerMotion.device:
            raise ValueError("observer motion validity device does not match motion")
        if observerMotionValid.dtype != torch.bool:
            raise TypeError("observer motion validity must be boolean")
        self.ValidateContractObserverRotation(observerMotion)
        observer_valid = observerMotionValid & bool(self.observer_valid)
        spatial_summary = self.PhysicalSlotSummary(
            physicalState["SlotState"],
            physicalState["SlotPresence"])
        observer_rotation = self.EffectiveObserverRotation(observerMotion)
        identity_rotation = torch.zeros_like(observer_rotation)
        identity_rotation[..., 3] = 1.0
        observer_rotation = torch.where(
            observer_valid.unsqueeze(-1),
            observer_rotation,
            identity_rotation)
        embodiment_context = self.embodiment_context_proj(torch.cat((
            embodimentState,
            actionEnc,
            spatial_summary,
            observer_rotation,
            observer_valid.to(dtype=actionEnc.dtype).unsqueeze(-1),
        ), dim=-1))
        embodied_action = self.embodied_action_proj(torch.cat([
            actionEnc,
            embodimentState,
            embodiment_context], dim=-1))
        return embodied_action, embodiment_context

    def BindPhysicalMu(
        self,
        worldH: torch.Tensor,
        worldZMu: torch.Tensor,
        worldX: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentContext: torch.Tensor,
        sampleMask: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:
        binding = self.pst_binder(worldH, worldZMu, worldX, physicalState, actionEnc, embodimentContext)
        pst_summary_target = self.PhysicalSlotSummary(
            physicalState["SlotState"],
            binding["slot_binding_weight"]).detach()
        if sampleMask is None:
            loss_pst_bind = (
                0.01 * binding["delta_mu"].square().mean()
                + 0.10 * F.smooth_l1_loss(
                    binding["pst_summary_pred"],
                    pst_summary_target,
                    reduction="mean")
                + 0.001 * binding["bind_gate"].mean())
        else:
            loss_pst_bind = (
                0.01 * self.MaskedBatchMean(
                    binding["delta_mu"].square(), sampleMask)
                + 0.10 * self.MaskedBatchMean(
                    F.smooth_l1_loss(
                        binding["pst_summary_pred"],
                        pst_summary_target,
                        reduction="none"),
                    sampleMask)
                + 0.001 * self.MaskedBatchMean(
                    binding["bind_gate"], sampleMask))
        binding["loss_pst_bind"] = loss_pst_bind
        binding["embodiment_context"] = embodimentContext
        return binding

    def BoundReward(self, reward: torch.Tensor) -> torch.Tensor:
        scale = max(abs(float(self.reward_min)), abs(float(self.reward_max)), 1e-6)
        return scale * torch.tanh(reward / scale)

    def PredictInformationGain(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.softplus(
            self.information_gain_head(hidden).squeeze(-1))

    def ExpectedInformationGain(
        self,
        worldState: torch.Tensor,
        priorMean: torch.Tensor,
        actionState: torch.Tensor,
        embodimentContext: torch.Tensor,
    ) -> torch.Tensor:
        context = self.information_gain_context(torch.cat([
            worldState,
            priorMean,
            actionState,
            embodimentContext], dim=-1))
        return self.PredictInformationGain(context)

    @staticmethod
    def RealizedInformationGain(
        posteriorMean: torch.Tensor,
        posteriorLogStd: torch.Tensor,
        priorMean: torch.Tensor,
        priorLogStd: torch.Tensor,
    ) -> torch.Tensor:
        variance_ratio = torch.exp((
            2.0 * (posteriorLogStd - priorLogStd)).clamp(-30.0, 30.0))
        mean_delta = (posteriorMean - priorMean).square() * torch.exp(
            (-2.0 * priorLogStd).clamp(-30.0, 30.0))
        return (
            priorLogStd
            - posteriorLogStd
            + 0.5 * (variance_ratio + mean_delta - 1.0)
        ).mean(dim=-1).clamp_min(0.0)

    def RewardDoneTrunk(self, inp: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        normalized = self.rdone_ln(inp)
        if not deterministic:
            return self.rdone_trunk(normalized)
        hidden = self.rdone_trunk[0](normalized)
        hidden = self.rdone_trunk[1](hidden)
        hidden = F.dropout(hidden, p=float(self.rdone_trunk[2].p), training=False)
        hidden = self.rdone_trunk[3](hidden)
        return self.rdone_trunk[4](hidden)

    def ResetState(self, batchSize: int = 1):
        device, dtype = self.device, self.dtype
        B = int(batchSize)

        self._h = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
        self._z = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)
        self.s4.ResetState(B)
        self._A_prev = None
        self._A_prev_valid = None


        self.EnsurePhysicalMemory(B)
        if self._use_memory:
            self.EnsureB(B)

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._h, self._z, self.s4.x

    def ImportState(self, h: torch.Tensor, z: torch.Tensor, s4x: torch.Tensor):
        self._h = h.detach().to(device=self.device, dtype=self.dtype).clone()
        self._z = z.detach().to(device=self.device, dtype=self.dtype).clone()
        self.s4.x = s4x.detach().to(device=self.device, dtype=self.dtype).clone()

    def BuildPredictedVisual(self, state: torch.Tensor) -> Dict[str, Any]:
        predicted_visual = self.predicted_visual_head(state)
        reconstructed = self.visual_reconstructor(predicted_visual)
        return {"predicted_visual": predicted_visual,"reconstructed_visual_state": reconstructed,}

    def ObjectAttentionError(self, predictedObjects: torch.Tensor, targetObjects: torch.Tensor) -> torch.Tensor:
        D = int(targetObjects.size(-1))
        scores = torch.matmul(targetObjects, predictedObjects.transpose(1, 2)) / max(float(D) ** 0.5, 1.0)
        weights = F.softmax(scores, dim=-1)
        aligned_pred = torch.matmul(weights, predictedObjects)
        return (aligned_pred - targetObjects).pow(2).mean(dim=(1, 2))

    def ComputePredictionLoss(
        self,
        predictedVisual: PredictedVisualPack,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        precision: torch.Tensor,
        sampleMask: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        target = {
            "GlobalFeat": targetVisualState.GlobalFeat.detach(),
            "ObjectTokens": targetVisualState.ObjectTokens.detach(),
            "IntegratedFeat": targetVisualState.IntegratedFeat.detach(),
            "MotionPred": targetVisualState.MotionToken.detach(),}

        global_err = (predictedVisual.GlobalFeat - target["GlobalFeat"]).pow(2).mean(dim=-1)
        object_err = self.ObjectAttentionError(predictedVisual.ObjectTokens, target["ObjectTokens"])
        integrated_err = (predictedVisual.IntegratedFeat - target["IntegratedFeat"]).pow(2).mean(dim=-1)
        motion_err = (predictedVisual.MotionPred - target["MotionPred"]).pow(2).mean(dim=-1)

        recon_err = (
            (reconstructedVisualState["GlobalFeat"] - target["GlobalFeat"]).pow(2).mean(dim=-1)
            + 0.5 * self.ObjectAttentionError(reconstructedVisualState["ObjectTokens"], target["ObjectTokens"])
            + 0.25 * (reconstructedVisualState["MotionPred"] - target["MotionPred"]).pow(2).mean(dim=-1)
            + 0.5 * (reconstructedVisualState["IntegratedFeat"] - target["IntegratedFeat"]).pow(2).mean(dim=-1))
        basis_err = (reconstructedVisualState["PredErrorBasis"] - target["GlobalFeat"]).pow(2).mean(dim=-1)

        target_auxiliary = targetVisualState.Auxiliary
        alignment = F.softmax(
            torch.matmul(
                reconstructedVisualState["ObjectTokens"],
                target["ObjectTokens"].transpose(1, 2))
            / (float(target["ObjectTokens"].size(-1)) ** 0.5),
            dim=-1)

        def AlignSlot(value: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bij,bj...->bi...", alignment, value.detach())

        factor_presence = AlignSlot(
            target_auxiliary["PerceptualPresence"])
        factor_denominator = factor_presence.sum(dim=-1) + 1e-6
        realm_target = AlignSlot(target_auxiliary["RealmProb"])
        realm_err = (
            -realm_target
            * torch.log(reconstructedVisualState["RealmProb"] + 1e-8)
        ).sum(dim=-1)
        layer_target = AlignSlot(target_auxiliary["MotionLayerProb"])
        layer_err = F.binary_cross_entropy(
            reconstructedVisualState["MotionLayerProb"],
            layer_target,
            reduction="none").mean(dim=-1)
        layer_agency_target = AlignSlot(
            target_auxiliary["LayerAgencyProb"])
        layer_agency_err = (
            -layer_agency_target
            * torch.log(
                reconstructedVisualState["LayerAgencyProb"] + 1e-8)
        ).sum(dim=-1).mean(dim=-1)
        display_target = AlignSlot(
            target_auxiliary["DisplaySurfaceProb"])
        display_err = F.binary_cross_entropy(
            reconstructedVisualState["DisplaySurfaceProb"],
            display_target,
            reduction="none")
        surface_uv_target = AlignSlot(target_auxiliary["SurfaceUV"])
        surface_uv_err = (
            reconstructedVisualState["SurfaceUV"] - surface_uv_target
        ).square().mean(dim=-1) * display_target
        content_motion_target = AlignSlot(
            target_auxiliary["ContentMotionUV"])
        content_motion_err = (
            reconstructedVisualState["ContentMotionUV"]
            - content_motion_target
        ).square().mean(dim=-1) * layer_target[..., 3]
        content_change_target = AlignSlot(
            target_auxiliary["ContentChangeProb"])
        content_change_err = F.binary_cross_entropy(
            reconstructedVisualState["ContentChangeProb"],
            content_change_target,
            reduction="none")
        factor_confidence_target = AlignSlot(
            target_auxiliary["VerificationConfidence"])
        factor_confidence_err = (
            reconstructedVisualState["FactorPriorConfidence"]
            - factor_confidence_target).square()
        factor_err = (
            (
                realm_err
                + layer_err
                + layer_agency_err
                + display_err
                + surface_uv_err
                + content_motion_err
                + content_change_err
                + factor_confidence_err)
            * factor_presence
        ).sum(dim=-1) / factor_denominator

        per_sample = (
            global_err
            + object_err
            + 0.5 * integrated_err
            + 0.25 * motion_err
            + 0.5 * recon_err
            + 0.1 * basis_err
            + 0.25 * factor_err)
        batch_size = int(per_sample.size(0))
        if sampleMask.dtype != torch.bool:
            raise TypeError(f"sampleMask must be bool, got {sampleMask.dtype}")
        if tuple(sampleMask.shape) != (batch_size,):
            raise ValueError(
                f"sampleMask must have shape ({batch_size},), got {tuple(sampleMask.shape)}")
        if sampleMask.device != per_sample.device:
            raise ValueError(
                f"sampleMask must be on {per_sample.device}, got {sampleMask.device}")

        def MaskedMeanLocal(values: torch.Tensor) -> torch.Tensor:
            numerator = torch.where(
                sampleMask, values, torch.zeros_like(values)).sum()
            return numerator / sampleMask.sum().clamp_min(1.0)

        if tuple(precision.shape) != (batch_size,):
            raise ValueError(
                f"precision must have shape ({batch_size},), got {tuple(precision.shape)}")
        if precision.device != per_sample.device:
            raise ValueError(
                f"precision must be on {per_sample.device}, got {precision.device}")
        p = precision.detach()
        precision_loss = MaskedMeanLocal(p * per_sample)
        inverse_losses = self.visual_reconstructor.InverseMappingLoss(
            reconstructedVisualState,
            targetVisualState,
            sampleMask=sampleMask)
        total_loss = precision_loss + 0.25 * inverse_losses["loss_pred_inverse_total"]

        losses = {
            "loss_pred_global": MaskedMeanLocal(global_err),
            "loss_pred_object": MaskedMeanLocal(object_err),
            "loss_pred_integrated": MaskedMeanLocal(integrated_err),
            "loss_pred_motion": MaskedMeanLocal(motion_err),
            "loss_pred_recon": MaskedMeanLocal(recon_err),
            "loss_pred_basis": MaskedMeanLocal(basis_err),
            "loss_pred_entity_motion_factors": MaskedMeanLocal(factor_err),
            "loss_pred_precision": precision_loss,
            "loss_pred_total": total_loss,}
        losses.update(inverse_losses)
        return losses

    def ValidatePriorRolloutSequenceInputs(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalStateSequence: Dict[str, torch.Tensor],
        actionEncSequence: torch.Tensor,
        embodimentStateSequence: torch.Tensor,
        observerMotionSequence: torch.Tensor,
        observerMotionValidSequence: torch.Tensor,
    ) -> Tuple[int, int]:
        if not torch.is_tensor(actionEncSequence):
            raise TypeError("actionEncSequence must be a tensor")
        if actionEncSequence.ndim != 3:
            raise ValueError("actionEncSequence must have shape [B, T, A]")
        B, T, action_dim = map(int, actionEncSequence.shape)
        if B < 1 or T < 1 or action_dim != self.action_dim:
            raise ValueError(
                f"actionEncSequence must have shape [B, T, {self.action_dim}]")
        expected_float_shapes = {
            "hPrev": (hPrev, (B, self.deter_dim)),
            "zPrev": (zPrev, (B, self.stoch_dim)),
            "s4xPrev": (s4xPrev, (B, self.ssm_dim)),
            "actionEncSequence": (
                actionEncSequence,
                (B, T, self.action_dim)),
            "embodimentStateSequence": (
                embodimentStateSequence,
                (B, T, self.embodiment_state_dim)),
            "observerMotionSequence": (
                observerMotionSequence,
                (B, T, ROTATION_QUATERNION_DIM)),
        }
        for name, (value, expected_shape) in expected_float_shapes.items():
            if not torch.is_tensor(value):
                raise TypeError(f"{name} must be a tensor")
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {tuple(value.shape)}")
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if value.device != self.device:
                raise ValueError(
                    f"{name} must be on {self.device}, got {value.device}")
            if value.dtype != self.dtype:
                raise ValueError(
                    f"{name} must have dtype {self.dtype}, got {value.dtype}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")
        if not torch.is_tensor(observerMotionValidSequence):
            raise TypeError("observerMotionValidSequence must be a tensor")
        if tuple(observerMotionValidSequence.shape) != (B, T):
            raise ValueError(
                "observerMotionValidSequence must have shape [B, T]")
        if observerMotionValidSequence.device != self.device:
            raise ValueError(
                "observerMotionValidSequence device does not match world model")
        if observerMotionValidSequence.dtype != torch.bool:
            raise TypeError("observerMotionValidSequence must be boolean")
        self.ValidateContractObserverRotation(
            observerMotionSequence.reshape(B * T, ROTATION_QUATERNION_DIM))
        if not isinstance(physicalStateSequence, dict):
            raise TypeError("physicalStateSequence must be a dictionary")
        required_fields = set(MODEL_PHYSICAL_STATE_FIELDS)
        actual_fields = set(physicalStateSequence)
        missing = sorted(required_fields.difference(actual_fields))
        extra = sorted(actual_fields.difference(required_fields))
        if missing or extra:
            raise ValueError(
                f"physicalStateSequence fields mismatch; missing={missing}, extra={extra}")
        K = self.physical_slots
        level_dim = min(3, self.physical_semantic_dim)
        object_dim = max(min(
            self.physical_semantic_dim,
            3 + ModuleDim.PstObjectClasses) - 3, 0)
        part_dim = max(
            self.physical_semantic_dim
            - 3
            - ModuleDim.PstObjectClasses,
            0)
        expected_shapes = {
            "SlotState": (B, T, K, self.physical_slot_dim),
            "ARaw": (B, T, K, self.physical_attr_dim),
            "SlotPresence": (B, T, K),
            "MphysRaw": (B, T, K),
            "PerceptualPresence": (B, T, K),
            "GeometryValidMask": (B, T, K),
            "PhysicalEntityProb": (B, T, K),
            "PhysicalInteractionProb": (B, T, K),
            "RealmProb": (B, T, K, ModuleDim.PstRealmClasses),
            "MotionLayerProb": (B, T, K, ModuleDim.PstMotionLayerClasses),
            "LayerAgencyProb": (
                B,
                T,
                K,
                ModuleDim.PstMotionLayerClasses,
                ModuleDim.PstAgencyClasses),
            "AgencyProb": (B, T, K, ModuleDim.PstAgencyClasses),
            "BodyMembershipProb": (B, T, K),
            "SelfPartProb": (B, T, K, self.self_part_count),
            "SelfPartSemantic": (
                B,
                T,
                K,
                self.self_part_semantic_dim),
            "IdentityKey": (B, T, K, self.physical_id_dim),
            "PairRelationLastSeen": (B, T, K, K),
            "ExternalRelationProbRaw": (
                B,
                T,
                K,
                self.physical_relation_classes),
            "Semantic": (B, T, K, self.physical_semantic_dim),
            "Size": (B, T, K, 3),
            "StateRaw": (B, T, K, self.physical_state_dim),
            "AffordanceRaw": (B, T, K, self.physical_affordance_dim),
            "ContentMotionUV": (B, T, K, 2),
            "ContentChangeProb": (B, T, K),
            "MovingProbRaw": (B, T, K),
            "ContactProbRaw": (B, T, K),
            "ContactForceRaw": (B, T, K, 2),
            "ParentProb": (B, T, K, K),
            "DisplaySurfaceProb": (B, T, K),
            "SurfaceParentProb": (B, T, K, K + 1),
            "SurfaceUV": (B, T, K, 2),
            "SurfaceUVConfidence": (B, T, K),
            "VerificationConfidence": (B, T, K),
            "OntologyRelationProb": (
                B,
                T,
                K,
                K,
                ModuleDim.PstOntologyRelationClasses),
            "Visibility": (B, T, K),
            "Occlusion": (B, T, K),
            "HasTextProb": (B, T, K),
            "TextEmbed": (B, T, K, self.physical_text_dim),
            "SymbolProb": (B, T, K, self.physical_symbol_dim),
            "Observed": (B, T, K),
            "LastSeen": (B, T, K),
            "Step": (B, T),
            "SpatialFrame": (B, T, K, self.physical_spatial_dim),
            "MotionObserverRaw": (
                B,
                T,
                K,
                self.physical_spatial_dim),
            "CarrierMotionObserverRaw": (
                B,
                T,
                K,
                self.physical_spatial_dim),
            "ArticulationMotionObserverRaw": (
                B,
                T,
                K,
                self.physical_spatial_dim),
            "ContactPointObserverRaw": (B, T, K, 3),
            "PairwiseRelationObserver": (
                B,
                T,
                K,
                K,
                self.physical_rel_dim),
            "LevelProb": (B, T, K, level_dim),
            "ObjectClassProb": (B, T, K, object_dim),
            "PartClassProb": (B, T, K, part_dim),
        }
        boolean_fields = {"Observed"}
        integer_fields = {"PairRelationLastSeen", "LastSeen", "Step"}
        for name in MODEL_PHYSICAL_STATE_FIELDS:
            value = physicalStateSequence[name]
            if not torch.is_tensor(value):
                raise TypeError(f"physicalStateSequence[{name!r}] must be a tensor")
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"physicalStateSequence[{name!r}] must have shape "
                    f"{expected_shapes[name]}, got {tuple(value.shape)}")
            if value.device != self.device:
                raise ValueError(
                    f"physicalStateSequence[{name!r}] must be on {self.device}")
            expected_dtype = (
                torch.bool
                if name in boolean_fields
                else torch.long
                if name in integer_fields
                else self.dtype)
            if value.dtype != expected_dtype:
                raise TypeError(
                    f"physicalStateSequence[{name!r}] must have dtype "
                    f"{expected_dtype}")
            if value.is_floating_point() and not bool(
                torch.isfinite(value).all().item()
            ):
                raise ValueError(
                    f"physicalStateSequence[{name!r}] must contain only finite values")
        return B, T

    def StackPriorRolloutSteps(
        self,
        stepOutputs: List[Dict[str, torch.Tensor]],
        batchSize: int,
    ) -> Dict[str, torch.Tensor]:
        if not stepOutputs:
            raise ValueError("stepOutputs must not be empty")

        def StackValues(values: List[Any], path: str) -> Any:
            first = values[0]
            if isinstance(first, dict):
                expected_keys = set(first)
                for value in values:
                    if not isinstance(value, dict) or set(value) != expected_keys:
                        raise RuntimeError(
                            f"prior rollout output fields changed at {path}")
                return {
                    name: StackValues(
                        [value[name] for value in values],
                        f"{path}.{name}")
                    for name in first}
            if not torch.is_tensor(first):
                raise TypeError(
                    f"prior rollout output at {path} must be a tensor or dictionary")
            for value in values:
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"prior rollout output at {path} changed type")
                if (
                    tuple(value.shape) != tuple(first.shape)
                    or value.device != first.device
                    or value.dtype != first.dtype
                ):
                    raise RuntimeError(
                        f"prior rollout output at {path} changed tensor schema")
                if value.is_floating_point() and not bool(
                    torch.isfinite(value).all().item()
                ):
                    raise FloatingPointError(
                        f"prior rollout output at {path} is nonfinite")
            if first.ndim == 0:
                return torch.stack(values, dim=0)
            if int(first.size(0)) != batchSize:
                raise RuntimeError(
                    f"prior rollout output at {path} is not batch aligned")
            return torch.stack(values, dim=1)

        return StackValues(stepOutputs, "prior")

    @torch.no_grad()
    def BuildPriorRolloutSequence(
        self,
        stepFunction: Callable[..., Dict[str, torch.Tensor]],
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalStateSequence: Dict[str, torch.Tensor],
        actionEncSequence: torch.Tensor,
        embodimentStateSequence: torch.Tensor,
        observerMotionSequence: torch.Tensor,
        observerMotionValidSequence: torch.Tensor,
        sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if type(sample) is not bool:
            raise TypeError("sample must be boolean")
        B, T = self.ValidatePriorRolloutSequenceInputs(
            hPrev,
            zPrev,
            s4xPrev,
            physicalStateSequence,
            actionEncSequence,
            embodimentStateSequence,
            observerMotionSequence,
            observerMotionValidSequence)
        steps = []
        h, z, x = hPrev, zPrev, s4xPrev
        for horizon_index in range(T):
            physical_state = {
                name: value[:, horizon_index]
                for name, value in physicalStateSequence.items()}
            step = stepFunction(
                h,
                z,
                x,
                actionEncSequence[:, horizon_index],
                physicalState=physical_state,
                embodimentState=embodimentStateSequence[:, horizon_index],
                observerMotion=observerMotionSequence[:, horizon_index],
                observerMotionValid=observerMotionValidSequence[
                    :, horizon_index],
                sample=sample)
            steps.append(step)
            h, z, x = step["h_next"], step["z_next"], step["x_next"]
        return self.StackPriorRolloutSteps(steps, B)

    @torch.no_grad()
    def PriorRolloutSequence(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalStateSequence: Dict[str, torch.Tensor],
        actionEncSequence: torch.Tensor,
        embodimentStateSequence: torch.Tensor,
        observerMotionSequence: torch.Tensor,
        observerMotionValidSequence: torch.Tensor,
        sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.BuildPriorRolloutSequence(
            self.StepPriorOnly,
            hPrev,
            zPrev,
            s4xPrev,
            physicalStateSequence,
            actionEncSequence,
            embodimentStateSequence,
            observerMotionSequence,
            observerMotionValidSequence,
            sample=sample)

    def PriorRolloutFromStateAction(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1))

        embodied_action, embodiment_context = self.BuildEmbodiedAction(
            physicalState,
            actionEnc,
            embodimentState,
            observerMotion,
            observerMotionValid)
        a_t = self.act_proj(embodied_action)

        h_next, x_next = self.s4.StepWithX(zPrev, a_t, s4xPrev)
        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True)
            dmu = self.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate = self.ComputeNeuroSymbolicGate(base_gate, pen, conf)
            mu_p = mu_p + gate * dmu

        mu_p_raw = mu_p
        pst_binding = self.BindPhysicalMu(h_next, mu_p, x_next, physicalState, embodied_action, embodiment_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1))

        A_t = self.conn(s_prev_base, a_t)
        s_transport = self.conn.TransportApply(A_t, s_prev_base)
        h_phys, _, _ = self.phys_refiner(hPrev, a_t, h_next)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        w = F.softmax(self.mix_gate(torch.cat([s_base, d_tr, d_ph], dim=-1)), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        trunk = self.rdone_trunk(self.rdone_ln(torch.cat([s_base, s_next, a_t], dim=-1)))
        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": x_next,
            "s_next": s_next,
            "action_enc": actionEnc,
            "embodied_action": embodied_action,
            "embodiment_context": embodiment_context,
            "r_pred": self.BoundReward(self.rew_head(trunk).squeeze(-1)),
            "d_prob": torch.sigmoid(self.done_head(trunk).squeeze(-1)),
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

    def PredictNextVisualFromPosterior(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool = False,) -> Dict[str, Any]:
        rollout = self.PriorRolloutFromStateAction(
            hPrev=h,
            zPrev=z,
            s4xPrev=s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            embodimentState=embodimentState,
            observerMotion=observerMotion,
            observerMotionValid=observerMotionValid,
            sample=sample,)
        pred = self.BuildPredictedVisual(rollout["s_next"])
        pred["prior_rollout"] = rollout
        return pred

    def PredictNextVisualWithStationaryObserver(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        sample: bool = False,) -> Dict[str, Any]:
        return self.PredictNextVisualFromPosterior(
            h,
            z,
            s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            embodimentState=embodimentState,
            observerMotion=self.StationaryObserverMotion(actionEnc),
            observerMotionValid=torch.zeros(
                actionEnc.size(0),
                device=actionEnc.device,
                dtype=torch.bool),
            sample=sample,)

    def BuildWorldAbstract(
        self,
        worldOut: Dict[str, torch.Tensor],
        physicalState: Dict[str, torch.Tensor],
        pstSummary: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,) -> Dict[str, torch.Tensor]:
        world_hzx = torch.cat([
            worldOut["h_next"],
            worldOut["z_next"],
            worldOut["x_next"],
        ], dim=-1)
        pst_context = worldOut["pst_binding"]["pst_context"]
        embodiment_context = worldOut["pst_binding"]["embodiment_context"]
        scalar = torch.stack([
            worldOut["r_pred"],
            worldOut["d_prob"],
            uncertainty.view(-1),
            confidence.view(-1),
        ], dim=-1)
        abstract_feat = self.world_abstract_projector(torch.cat([
            world_hzx,
            worldOut["s_next"],
            pstSummary,
            pst_context,
            embodiment_context,
            scalar,
        ], dim=-1))
        return {
            "world_hzx": world_hzx,
            "world_state": worldOut["s_next"],
            "pst_summary": pstSummary,
            "pst_context": pst_context,
            "embodiment_context": embodiment_context,
            "slot_binding_weight": worldOut["pst_binding"]["slot_binding_weight"],
            "reward_pred": worldOut["r_pred"],
            "done_prob": worldOut["d_prob"],
            "uncertainty": uncertainty,
            "confidence": confidence,
            "abstract_feat": abstract_feat,
            "slot_presence_mask": physicalState["SlotPresence"],
            "physical_entity_mask": physicalState["SlotPresence"] * physicalState["MphysRaw"],}

    @torch.no_grad()
    def ScoreDecisionImaginations(
        self,
        h0: torch.Tensor,
        z0: torch.Tensor,
        x0: torch.Tensor,
        actionEncCandidates: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        embodimentState: torch.Tensor,
        perceptionRotationCandidates: torch.Tensor,
        perceptionRotationValidCandidates: torch.Tensor,
        gamma: float = 0.99,
        physicalStateSequence: Optional[Dict[str, torch.Tensor]] = None,
        embodimentStateSequence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not torch.is_tensor(actionEncCandidates):
            raise TypeError("actionEncCandidates must be a tensor")
        if actionEncCandidates.ndim != 4:
            raise ValueError(
                "actionEncCandidates must have shape [B, N, T, A]")
        B, N, T, A = actionEncCandidates.shape
        if B < 1 or N < 1 or T < 1 or A != self.action_dim:
            raise ValueError(
                f"actionEncCandidates must have shape [B, N, T, {self.action_dim}]")
        if not torch.is_tensor(perceptionRotationCandidates):
            raise TypeError("perceptionRotationCandidates must be a tensor")
        if not torch.is_tensor(perceptionRotationValidCandidates):
            raise TypeError(
                "perceptionRotationValidCandidates must be a tensor")
        if tuple(perceptionRotationCandidates.shape) != (
            B, N, T, ROTATION_QUATERNION_DIM
        ):
            raise ValueError(
                "candidate perception rotation must have shape [B, N, T, 4]")
        if tuple(perceptionRotationValidCandidates.shape) != (B, N, T):
            raise ValueError(
                "candidate perception rotation validity must have shape [B, N, T]")
        if perceptionRotationValidCandidates.dtype != torch.bool:
            raise TypeError("candidate perception rotation validity must be boolean")
        if (physicalStateSequence is None) != (embodimentStateSequence is None):
            raise ValueError(
                "physical and embodiment state sequences must be provided together")
        if T > 1 and physicalStateSequence is None:
            raise ValueError(
                "multi-step imagination requires explicit physical and embodiment state sequences")
        if physicalStateSequence is None:
            if not isinstance(physicalState, dict):
                raise TypeError("physicalState must be a dictionary")
            candidate_physical_sequence = {}
            for name, value in physicalState.items():
                if not torch.is_tensor(value) or value.ndim < 1:
                    raise TypeError(f"physicalState[{name!r}] must be a tensor")
                if int(value.size(0)) != B:
                    raise ValueError(
                        f"physicalState[{name!r}] batch does not match candidates")
                candidate_physical_sequence[name] = value.unsqueeze(
                    1).unsqueeze(2).expand(B, N, 1, *value.shape[1:])
            if not torch.is_tensor(embodimentState):
                raise TypeError("embodimentState must be a tensor")
            if tuple(embodimentState.shape) != (
                B,
                self.embodiment_state_dim,
            ):
                raise ValueError("embodimentState shape does not match candidates")
            candidate_embodiment_sequence = embodimentState.unsqueeze(
                1).unsqueeze(2).expand(
                    B,
                    N,
                    1,
                    self.embodiment_state_dim)
        else:
            if not isinstance(physicalStateSequence, dict):
                raise TypeError("physicalStateSequence must be a dictionary")
            required_fields = set(MODEL_PHYSICAL_STATE_FIELDS)
            if set(physicalStateSequence) != required_fields:
                raise ValueError(
                    "physicalStateSequence fields do not match model physical state")
            candidate_physical_sequence = {}
            for name, value in physicalStateSequence.items():
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"physicalStateSequence[{name!r}] must be a tensor")
                if value.ndim < 3 or tuple(value.shape[:3]) != (B, N, T):
                    raise ValueError(
                        f"physicalStateSequence[{name!r}] must begin with [B, N, T]")
                candidate_physical_sequence[name] = value
            if (
                not torch.is_tensor(embodimentStateSequence)
                or tuple(embodimentStateSequence.shape)
                != (B, N, T, self.embodiment_state_dim)
            ):
                raise ValueError(
                    "embodimentStateSequence must have shape [B, N, T, E]")
            candidate_embodiment_sequence = embodimentStateSequence
        for name, value, width in (
            ("h0", h0, self.deter_dim),
            ("z0", z0, self.stoch_dim),
            ("x0", x0, self.ssm_dim),
        ):
            if not torch.is_tensor(value) or tuple(value.shape) != (B, width):
                raise ValueError(f"{name} shape does not match candidates")
        gamma_value = float(gamma)
        if not bool(torch.isfinite(
            actionEncCandidates.new_tensor(gamma_value)).item()
        ):
            raise ValueError("gamma must be finite")
        if gamma_value < 0.0 or gamma_value > 1.0:
            raise ValueError("gamma must be in [0, 1]")
        h = h0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
        z = z0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
        x = x0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
        flattened_physical_sequence = {
            name: value.reshape(B * N, T, *value.shape[3:]).contiguous()
            for name, value in candidate_physical_sequence.items()}
        flattened_embodiment_sequence = candidate_embodiment_sequence.reshape(
            B * N,
            T,
            self.embodiment_state_dim).contiguous()
        rollout = self.PriorRolloutSequence(
            h,
            z,
            x,
            physicalStateSequence=flattened_physical_sequence,
            actionEncSequence=actionEncCandidates.reshape(
                B * N,
                T,
                A),
            embodimentStateSequence=flattened_embodiment_sequence,
            observerMotionSequence=perceptionRotationCandidates.reshape(
                B * N,
                T,
                ROTATION_QUATERNION_DIM),
            observerMotionValidSequence=perceptionRotationValidCandidates.reshape(
                B * N,
                T),
            sample=False)
        reward = rollout["r_pred"].reshape(B, N, T)
        done = rollout["d_prob"].reshape(B, N, T)
        continuation_before = torch.cat([
            done.new_ones(B, N, 1),
            torch.cumprod(1.0 - done[..., :-1], dim=-1),
        ], dim=-1)
        discount = actionEncCandidates.new_tensor(gamma_value).pow(
            torch.arange(T, device=actionEncCandidates.device))
        score = (
            continuation_before
            * discount.view(1, 1, T)
            * reward
        ).sum(dim=-1)
        cont = torch.prod(1.0 - done, dim=-1)
        return {
            "score": score,
            "continue_prob": cont,
            "terminal_h": rollout["h_next"][:, -1].view(B, N, -1),
            "terminal_z": rollout["z_next"][:, -1].view(B, N, -1),
            "terminal_x": rollout["x_next"][:, -1].view(B, N, -1),}

    def NsProjectProbs(self, P: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        return self.ns_struct.ProjectTrain(P, temp=temp)

    @torch.no_grad()
    def NsProjectRuntime(self, P: torch.Tensor, *, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        return self.ns_struct.ProjectRuntime(P, aloTau=aloTau, implAlpha=implAlpha, temp=temp)

    def NsConfidence(self, P: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        P = P.clamp(eps, 1 - eps) # [B,K]
        H = -(P * torch.log(P) + (1 - P) * torch.log(1 - P)) # [B,K]
        Hmax = P.new_tensor(0.6931471805599453)
        conf = (1.0 - H / Hmax).clamp(0.0, 1.0) # [B,K]
        return conf

    @staticmethod
    def ComputeNeuroSymbolicGate(
        baseGate: torch.Tensor,
        penalty: torch.Tensor,
        confidence: torch.Tensor,
        ) -> torch.Tensor:

        return (
            baseGate
            * (1.0 - 0.40 * penalty.view(-1, 1))
            * (0.6 + 0.4 * confidence))

    def ComputeMemoryImportance(
        self,
        rewardPrediction: torch.Tensor,
        doneProbability: torch.Tensor,
        nsProbability: Optional[torch.Tensor],
        nsPenalty: Optional[torch.Tensor],
        ) -> torch.Tensor:

        reward_score = torch.tanh(rewardPrediction.detach().abs())
        done_score = doneProbability.detach()
        if self._ns_enabled:
            ns_confidence = self.NsConfidence(
                nsProbability.detach()).mean(dim=-1)
            ns_score = (
                (1.0 - nsPenalty.detach())
                * (0.5 + 0.5 * ns_confidence))
        else:
            ns_score = torch.full_like(reward_score, 0.5)
        return 0.60 * ns_score + 0.25 * reward_score + 0.15 * done_score

    def NsLogicLosses(
        self,
        probs: torch.Tensor,
        sampleMask: Optional[torch.Tensor] = None,
    ):
        if sampleMask is not None:
            mask = self.ResolveCommitMask(sampleMask, int(probs.size(0)))
            if not bool(mask.any().item()):
                return probs.sum() * 0.0, {}
            probs = probs[mask]
        loss, stats = self.ns_struct.LogicLosses(
            probs,
            lambdaExcl=self.ns_lambda_excl,
            lambdaAlo=self.ns_lambda_alo,
            lambdaImpl=self.ns_lambda_impl,
            aloTau=0.6,)

        return loss, stats

    @torch.no_grad()
    def StepStationaryObserverPriorOnly(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        embodimentState: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        return self.StepPriorOnly(
            hPrev,
            zPrev,
            s4xPrev,
            actionEnc,
            physicalState=physicalState,
            embodimentState=embodimentState,
            observerMotion=self.StationaryObserverMotion(actionEnc),
            observerMotionValid=torch.zeros(
                actionEnc.size(0),
                device=actionEnc.device,
                dtype=torch.bool),
            sample=sample,)

    @staticmethod
    def StationaryObserverMotion(reference: torch.Tensor) -> torch.Tensor:
        observer_motion = reference.new_zeros(
            reference.size(0), ROTATION_QUATERNION_DIM)
        observer_motion[:, -1] = 1.0
        return observer_motion

    @torch.no_grad()
    def StepPriorOnly(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:

        B = actionEnc.size(0)
        device, dtype = self.device, self.dtype

        if hPrev is None or zPrev is None or s4xPrev is None:
            hPrev = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
            zPrev = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)
            s4xPrev = torch.zeros(B, self.ssm_dim, device=device, dtype=dtype)

        embodied_action, embodiment_context = self.BuildEmbodiedAction(
            physicalState,
            actionEnc,
            embodimentState,
            observerMotion,
            observerMotionValid)
        a_t = self.act_proj(embodied_action)
        h_next, s4x_next = self.s4.StepWithX(zPrev, a_t, s4xPrev) # h_next: [B, deterDim], s4x_next: [B, ssmDim]

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1) # [B, stochDim]
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_prior(
                h_next, deterministic=True, updateAux=False) # [B,K]
            P_raw = torch.sigmoid(ns_logits) # [B,K]
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # Q:[B,K], pen:[B]
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True) # [B,1]

            dmu = self.ns_to_delta_mu(Q) # [B, stochDim]

            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, dmu], dim=-1))) # [B, stochDim]
            gate = self.ComputeNeuroSymbolicGate(base_gate, pen, conf) # [B, stochDim]

            mu_p = mu_p + gate * dmu # [B, stochDim]

        mu_p_raw = mu_p
        pst_binding = self.BindPhysicalMu(h_next, mu_p, s4x_next, physicalState, embodied_action, embodiment_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            logstd_p = logstd_p.clamp(-7.0, 2.0)
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p # [B, stochDim]

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1)) # [B, stateDim]
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1)) # [B, stateDim]

        A_t = self.conn(s_prev_base, a_t) # [B, stateDim, stateDim]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B, stateDim]

        h_phys, _, _ = self.phys_refiner(hPrev, a_t, h_next)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))

        d_tr = s_transport - s_base # [B,stateDim]
        d_ph = s_phys - s_base # [B,stateDim]
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3 * stateDim]

        logits = self.mix_gate(g_in) # [B,3]
        w = F.softmax(logits, dim=-1) # [B,3]
        s_next = w[:,0:1]*s_base + w[:,1:2]*s_transport + w[:,2:3]*s_phys # [B,stateDim]

        inp = torch.cat([s_base, s_next, a_t], dim=-1) # [B, 2 * stateDim + stochDim]
        h = self.RewardDoneTrunk(inp, deterministic=True)

        r_pred = self.BoundReward(self.rew_head(h).squeeze(-1)) # [B]
        d_logit = self.done_head(h).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]
        information_gain_pred = self.ExpectedInformationGain(
            h_next,
            mu_p,
            a_t,
            embodiment_context)

        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": s4x_next,
            "s_next": s_next,
            "embodied_action": embodied_action,
            "embodiment_context": embodiment_context,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "information_gain_pred": information_gain_pred,
            "mu_p": mu_p,
            "logstd_p": logstd_p,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}


    def StepPosterior(
        self,
        visionIn: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        *,
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        transitionPhysicalState: Dict[str, torch.Tensor],
        transitionEmbodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool = False,
        commitState: bool = True,
        updateMemory: bool = True,
        commitMask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:

        if type(commitState) is not bool or type(updateMemory) is not bool:
            raise TypeError("posterior state controls must be booleans")
        if updateMemory and not commitState:
            raise ValueError("posterior memory updates require state commit")
        B = int(visionIn.size(0))
        self.EnsureB(B)
        commit_mask = self.ResolveCommitMask(commitMask, B)

        raw_e = self.obs_enc(visionIn) # [B, stochDim]
        transition_embodied_action, transition_embodiment_context = self.BuildEmbodiedAction(
            transitionPhysicalState,
            actionEnc,
            transitionEmbodimentState,
            observerMotion,
            observerMotionValid)
        observation_embodied_action, observation_embodiment_context = self.BuildEmbodiedAction(
            physicalState,
            actionEnc,
            embodimentState,
            observerMotion,
            observerMotionValid)
        a_t = self.act_proj(transition_embodied_action) # [B, stochDim]
        key = self.key_emb(raw_e, a_t) # [B, stochDim]

        h_pred, x_next = self.s4.StepWithX(
            self._z,
            a_t,
            self.s4.x) # [B, deterDim]
        # [B, ssmDim]

        mu_p, logstd_p = self.prior_net(h_pred).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)
        if self._ns_enabled:
            prior_logits = self.ns_head_prior(
                h_pred,
                deterministic=True,
                updateAux=False)
            prior_probability = torch.sigmoid(prior_logits)
            prior_projected, prior_penalty = self.NsProjectRuntime(
                prior_probability,
                aloTau=0.60,
                implAlpha=1.0,
                temp=1.0)
            prior_confidence = self.NsConfidence(
                prior_projected).mean(dim=-1, keepdim=True)
            prior_delta = self.ns_to_delta_mu(prior_projected)
            prior_base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([
                h_pred,
                prior_delta,
            ], dim=-1)))
            prior_gate = self.ComputeNeuroSymbolicGate(
                prior_base_gate,
                prior_penalty,
                prior_confidence)
            mu_p = mu_p + prior_gate * prior_delta
        mu_p_raw = mu_p
        prior_binding = self.BindPhysicalMu(
            h_pred,
            mu_p,
            x_next,
            transitionPhysicalState,
            transition_embodied_action,
            transition_embodiment_context,
            sampleMask=commit_mask)
        mu_p = prior_binding["bound_mu"]

        mu_q, logstd_q = self.post_net(torch.cat([h_pred, raw_e], dim=-1)).chunk(2, dim=-1) # [B,stochDim]
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        if self._ns_enabled:
            post_aux_state = (
                None
                if commitState
                else self.ns_head_post.aux_loss.detach().clone())
            ns_logits = self.ns_head_post(torch.cat([h_pred, raw_e], dim=-1)) # [B,K]
            if post_aux_state is not None:
                self.ns_head_post.aux_loss.copy_(post_aux_state)
            P_raw = torch.sigmoid(ns_logits) # [B,K]
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # Q:[B,K], pen:[B]
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True) # [B,1]

            dmu = self.ns_to_delta_mu(Q) # [B, stochDim]

            base_gate = torch.sigmoid(self.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu], dim=-1))) # [B, stochDim]
            gate = self.ComputeNeuroSymbolicGate(base_gate, pen, conf) # [B, stochDim]

            mu_q = mu_q + gate * dmu # [B, stochDim]

        mu_q_raw = mu_q
        pst_binding = self.BindPhysicalMu(
            h_pred,
            mu_q,
            x_next,
            physicalState,
            observation_embodied_action,
            observation_embodiment_context,
            sampleMask=commit_mask)
        mu_q = pst_binding["bound_mu"]

        if sample:
            z_next = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
        else:
            z_next = mu_q

        s_base = self.state_proj(torch.cat([h_pred, z_next], dim=-1)) # [B, stateDim]
        s_prev_base = self.state_proj(torch.cat([self._h, self._z], dim=-1)) # [B, stateDim]

        A_t = self.conn(s_prev_base, a_t) # [B,S,S]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B,S]

        h_phys, _, _ = self.phys_refiner(self._h, a_t, h_pred) # [B,D]
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1)) # [B,S]

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3S]
        logits = self.mix_gate(g_in) # [B,3]
        w = F.softmax(logits, dim=-1) # [B,3]
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        dynamics_state = s_next
        inp = torch.cat([s_base, dynamics_state, a_t], dim=-1)
        h = self.rdone_trunk(self.rdone_ln(inp))
        r_pred = self.BoundReward(self.rew_head(h).squeeze(-1)) # [B]
        d_logit = self.done_head(h).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]
        information_gain_pred = self.ExpectedInformationGain(
            h_pred,
            mu_p,
            a_t,
            transition_embodiment_context)

        if self._use_memory:
            with torch.no_grad():
                mem_retrieved = self.MemRetrieve(
                    key,
                    updateImportance=updateMemory,
                    commitMask=commit_mask)
                if updateMemory:
                    imp = self.ComputeMemoryImportance(
                        r_pred,
                        d_prob,
                        Q if self._ns_enabled else None,
                        pen if self._ns_enabled else None)
                    self.MemAdd(
                        key.detach(),
                        dynamics_state.detach(),
                        imp.detach(),
                        commitMask=commit_mask)

                if mem_retrieved is not None:
                    mem_s, mem_mask = mem_retrieved
                    s_memory = self.state_state_film(dynamics_state, mem_s)
                    s_next = torch.where(mem_mask.view(B, 1), s_memory, dynamics_state)

        out: Dict[str, torch.Tensor] = {
            "h_next": h_pred,
            "z_next": z_next,
            "z_next_raw": mu_q_raw,
            "x_next": x_next,
            "s_next": s_next,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "information_gain_pred": information_gain_pred,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "mu_p": mu_p,
            "mu_p_raw": mu_p_raw,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "mu_q_raw": mu_q_raw,
            "logstd_q": logstd_q,
            "transition_embodied_action": transition_embodied_action,
            "transition_embodiment_context": transition_embodiment_context,
            "posterior_embodied_action": observation_embodied_action,
            "posterior_embodiment_context": observation_embodiment_context,
            "pst_binding_prior": prior_binding,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

        if self._ns_enabled:
            out["ns_logits"] = ns_logits
            out["ns_Q"] = Q
            out["ns_pen"] = pen

        if self.use_decoder:
            out["recon"] = self.obs_dec(s_next)
            out["recon_target"] = visionIn

        if commitState:
            self.s4.x = self.MergeCommittedRows(
                x_next.detach(),
                self.s4.x,
                commit_mask)
            self._h = self.MergeCommittedRows(
                h_pred.detach(),
                self._h,
                commit_mask)
            self._z = self.MergeCommittedRows(
                z_next.detach(),
                self._z,
                commit_mask)
        else:
            preview_importance = self.ComputeMemoryImportance(
                r_pred,
                d_prob,
                Q if self._ns_enabled else None,
                pen if self._ns_enabled else None)
            out.update({
                "posterior_preview_base_h": self._h.detach().clone(),
                "posterior_preview_base_z": self._z.detach().clone(),
                "posterior_preview_base_x": self.s4.x.detach().clone(),
                "posterior_preview_memory_size": (
                    self._mem_size.detach().clone()),
                "posterior_preview_memory_step": (
                    self._mem_global_step.detach().clone()),
                "posterior_preview_key": key.detach(),
                "posterior_preview_state": dynamics_state.detach(),
                "posterior_preview_importance": (
                    preview_importance.detach()),})

        return out

    def CommitPosteriorPreview(
        self,
        preview: Dict[str, torch.Tensor],
        commitMask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        required = (
            "h_next",
            "z_next",
            "x_next",
            "posterior_preview_base_h",
            "posterior_preview_base_z",
            "posterior_preview_base_x",
            "posterior_preview_memory_size",
            "posterior_preview_memory_step",
            "posterior_preview_key",
            "posterior_preview_state",
            "posterior_preview_importance",
        )
        if type(preview) is not dict or any(
            name not in preview
            for name in required
        ):
            raise ValueError("posterior preview is incomplete")
        for name in required:
            value = preview[name]
            if (
                not torch.is_tensor(value)
                or value.device != self.device
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError("posterior preview tensors are invalid")
        if (
            not torch.equal(preview["posterior_preview_base_h"], self._h)
            or not torch.equal(
                preview["posterior_preview_base_z"],
                self._z)
            or not torch.equal(
                preview["posterior_preview_base_x"],
                self.s4.x)
            or not torch.equal(
                preview["posterior_preview_memory_size"],
                self._mem_size)
            or not torch.equal(
                preview["posterior_preview_memory_step"],
                self._mem_global_step)
        ):
            raise RuntimeError("posterior preview is stale")
        expected_shapes = {
            "h_next": tuple(self._h.shape),
            "z_next": tuple(self._z.shape),
            "x_next": tuple(self.s4.x.shape),
            "posterior_preview_key": (
                int(self._h.size(0)),
                self.stoch_dim),
            "posterior_preview_state": (
                int(self._h.size(0)),
                self.state_dim),
            "posterior_preview_importance": (int(self._h.size(0)),),
        }
        if any(
            tuple(preview[name].shape) != shape
            for name, shape in expected_shapes.items()
        ):
            raise ValueError("posterior preview shapes are invalid")
        commit_mask = self.ResolveCommitMask(
            commitMask,
            int(self._h.size(0)))
        if self._use_memory:
            with torch.no_grad():
                self.MemRetrieve(
                    preview["posterior_preview_key"],
                    updateImportance=True,
                    commitMask=commit_mask)
                self.MemAdd(
                    preview["posterior_preview_key"],
                    preview["posterior_preview_state"],
                    preview["posterior_preview_importance"],
                    commitMask=commit_mask)
        self.s4.x = self.MergeCommittedRows(
            preview["x_next"].detach(),
            self.s4.x,
            commit_mask)
        self._h = self.MergeCommittedRows(
            preview["h_next"].detach(),
            self._h,
            commit_mask)
        self._z = self.MergeCommittedRows(
            preview["z_next"].detach(),
            self._z,
            commit_mask)
        if self._ns_enabled:
            self.ns_head_post.aux_loss.zero_()
        return {
            name: value
            for name, value in preview.items()
            if not name.startswith("posterior_preview_")}


    def ForwardTrain(
        self,
        visionIn: torch.Tensor, # [B, visionDim]
        physicalState: Dict[str, torch.Tensor],
        reward: Optional[torch.Tensor] = None, # [B]
        done: Optional[torch.Tensor] = None, # [B]
        *,
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        transitionPhysicalState: Dict[str, torch.Tensor],
        transitionEmbodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: Optional[bool] = None,
        updateMemory: Optional[bool] = None,
        commitMask: Optional[torch.Tensor] = None,
        alphaKl: float = 0.8,
        freeNats: float = 1.0,
        reconCoef: float = 1.0,
        rewardCoef: float = 1.0,
        doneCoef: float = 1.0,
        nsCoef: float = 1.0,
        nsDistillCoef: float = 0.0,
        nsPriorLogicCoef: float = 1e-3,
        physCoef: float = 1e-4,
        pstBindCoef: float = 0.05,) -> Dict[str, torch.Tensor]:

        B = visionIn.size(0)
        self.EnsureB(B)
        commit_mask = self.ResolveCommitMask(commitMask, int(B))
        sample = bool(self.training) if sample is None else bool(sample)
        update_memory = bool(self.training) if updateMemory is None else bool(updateMemory)
        update_memory = update_memory and bool(self.training)

        h0 = self._h
        z0 = self._z

        a_enc = actionEnc
        transition_embodied_action, transition_embodiment_context = self.BuildEmbodiedAction(
            transitionPhysicalState,
            a_enc,
            transitionEmbodimentState,
            observerMotion,
            observerMotionValid)
        observation_embodied_action, observation_embodiment_context = self.BuildEmbodiedAction(
            physicalState,
            a_enc,
            embodimentState,
            observerMotion,
            observerMotionValid)
        a_t = self.act_proj(transition_embodied_action) # [B, stochDim]

        h_pred, x_next = self.s4.StepWithX(z0, a_t, self.s4.x) # [B,D]

        mu_p, logstd_p = self.prior_net(h_pred).chunk(2, dim=-1) # [B,stochDim]
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        logits_pr = None
        P_pr_train = None
        ns_prior_logic = visionIn.new_tensor(0.0)

        if self._ns_enabled:
            logits_pr = self.ns_head_prior(
                h_pred,
                sampleMask=commit_mask) # [B,K]
            P_pr_raw = torch.sigmoid(logits_pr) # [B,K]
            P_pr_train = self.NsProjectProbs(P_pr_raw) # [B,K]

            dmu_p = self.ns_to_delta_mu(P_pr_train) # [B,stochDim]
            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_pred, dmu_p], dim=-1))) # [B,stochDim]

            _, pen_pr = self.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # [B]

            conf = self.NsConfidence(P_pr_train).mean(dim=-1, keepdim=True) # [B,1]
            gate = self.ComputeNeuroSymbolicGate(
                base_gate, pen_pr, conf) # [B,stochDim]

            mu_p = mu_p + gate * dmu_p

            if nsPriorLogicCoef > 0.0:
                ns_prior_logic, _ = self.NsLogicLosses(
                    P_pr_train,
                    sampleMask=commit_mask)

        mu_p_raw = mu_p
        pst_binding_prior = self.BindPhysicalMu(
            h_pred,
            mu_p,
            x_next,
            transitionPhysicalState,
            transition_embodied_action,
            transition_embodiment_context,
            sampleMask=commit_mask)
        mu_p = pst_binding_prior["bound_mu"]

        raw_e = self.obs_enc(visionIn) # [B,stochDim]

        mu_q, logstd_q = self.post_net(torch.cat([h_pred, raw_e], dim=-1)).chunk(2, dim=-1) # [B,stochDim]
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        ns_loss = visionIn.new_tensor(0.0)
        ns_distill = visionIn.new_tensor(0.0)
        logits_q = None
        Q_train = None
        pen_q = None

        if self._ns_enabled:
            logits_q = self.ns_head_post(
                torch.cat([h_pred, raw_e], dim=-1),
                sampleMask=commit_mask) # [B,K]
            P_q_raw = torch.sigmoid(logits_q) # [B,K]
            Q_train = self.NsProjectProbs(P_q_raw) # [B,K]

            dmu_q = self.ns_to_delta_mu(Q_train) # [B,stochDim]
            base_gate_q = torch.sigmoid(self.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu_q], dim=-1))) # [B,stochDim]

            _, pen_q = self.NsProjectRuntime(P_q_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # [B]

            conf_q = self.NsConfidence(Q_train).mean(dim=-1, keepdim=True) # [B,1]
            gate_q = self.ComputeNeuroSymbolicGate(
                base_gate_q, pen_q, conf_q)

            mu_q = mu_q + gate_q * dmu_q

            ns_loss, _ = self.NsLogicLosses(
                Q_train,
                sampleMask=commit_mask)

            if (logits_pr is not None) and (nsDistillCoef > 0.0):
                with torch.no_grad():
                    P_teacher = torch.sigmoid(logits_q) # [B,K]
                ns_distill = self.MaskedBatchMean(
                    F.binary_cross_entropy_with_logits(
                        logits_pr,
                        P_teacher,
                        reduction="none"),
                    commit_mask)

        mu_q_raw = mu_q
        pst_binding_posterior = self.BindPhysicalMu(
            h_pred,
            mu_q,
            x_next,
            physicalState,
            observation_embodied_action,
            observation_embodiment_context,
            sampleMask=commit_mask)
        mu_q = pst_binding_posterior["bound_mu"]

        if sample:
            z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q) # [B,stochDim]
        else:
            z1 = mu_q

        s_base = self.state_proj(torch.cat([h_pred, z1], dim=-1)) # [B,S]
        s_prev_base = self.state_proj(torch.cat([h0, z0], dim=-1)) # [B,S]

        A_t = self.conn(s_prev_base, a_t) # [B,S,S]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B,S]

        previous_connection_valid = (
            self._A_prev_valid
            if (
                self._A_prev_valid is not None
                and tuple(self._A_prev_valid.shape) == (int(B),)
            )
            else torch.zeros(int(B), device=A_t.device, dtype=torch.bool))
        if self._A_prev is not None and self._A_prev.shape == A_t.shape:
            previous_connection = self._A_prev
            previous_mask = previous_connection_valid.view(int(B), 1, 1)
            prevA = torch.where(
                previous_mask,
                previous_connection,
                A_t.detach())
        else:
            previous_connection = torch.zeros_like(A_t)
            prevA = None
        reg_A = self.conn.ComputeGeomReg(
            A_t,
            prevA,
            sampleMask=commit_mask)
        self._A_prev = self.MergeCommittedRows(
            A_t.detach(),
            previous_connection,
            commit_mask)
        self._A_prev_valid = previous_connection_valid | commit_mask

        active_rows = torch.nonzero(commit_mask, as_tuple=False).flatten()
        h_phys = h_pred # h_phys:[B,D]
        phys_loss = visionIn.new_zeros(())
        if active_rows.numel() > 0:
            active_h_phys, active_phys_loss, _ = self.phys_refiner(
                h0.index_select(0, active_rows),
                a_t.index_select(0, active_rows),
                h_pred.index_select(0, active_rows))
            h_phys = h_phys.index_copy(0, active_rows, active_h_phys)
            if active_phys_loss is not None:
                phys_loss = active_phys_loss
        s_phys = self.state_proj(torch.cat([h_phys, z1], dim=-1)) # [B,S]

        d_tr = s_transport - s_base # [B,S]
        d_ph = s_phys - s_base # [B,S]
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3S]
        w = F.softmax(self.mix_gate(g_in), dim=-1) # [B,3]
        s1 = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys # [B,S]

        dynamics_state = s1
        inp = torch.cat([s_base, dynamics_state, a_t], dim=-1) # [B,2S+stochDim]
        trunk = self.rdone_trunk(self.rdone_ln(inp)) # [B,256]
        r_pred = self.BoundReward(self.rew_head(trunk).squeeze(-1)) # [B]
        d_logit = self.done_head(trunk).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]
        information_gain_pred = self.ExpectedInformationGain(
            h_pred,
            mu_p,
            a_t,
            transition_embodiment_context)

        if self._use_memory:
            key = self.key_emb(raw_e, a_t) # [B,stochDim]
            mem_retrieved = self.MemRetrieve(
                key,
                updateImportance=update_memory,
                commitMask=commit_mask)
            if update_memory:
                imp = self.ComputeMemoryImportance(
                    r_pred, d_prob, Q_train, pen_q)
                self.MemAdd(
                    key.detach(),
                    dynamics_state.detach(),
                    imp.detach(),
                    commitMask=commit_mask)

            if mem_retrieved is not None:
                mem_s, mem_mask = mem_retrieved
                s_memory = self.state_state_film(dynamics_state, mem_s)
                s1 = torch.where(mem_mask.view(B, 1), s_memory, dynamics_state)

        loss_recon = visionIn.new_tensor(0.0)
        recon_error = visionIn.new_zeros(B)
        recon = None
        if self.use_decoder:
            recon = self.obs_dec(s1) # [B, visionDim]
            normalized_shape = (int(recon.size(-1)),)
            target = F.layer_norm(
                visionIn.detach(),
                normalized_shape=normalized_shape)
            recon_n = F.layer_norm(
                recon,
                normalized_shape=normalized_shape)

            recon_error = (recon_n - target).pow(2).mean(dim=-1)
            loss_recon = self.MaskedBatchMean(
                recon_error,
                commit_mask)

        aux_moe = visionIn.new_tensor(0.0)
        if self._ns_enabled:
            aux_moe = self.ns_head_prior.GetAuxLoss() + self.ns_head_post.GetAuxLoss()

        if reward is None:
            loss_reward = visionIn.new_zeros(())
        else:
            reward_target = reward.view(B).clamp(float(self.reward_min), float(self.reward_max))
            loss_reward = self.MaskedBatchMean(
                F.mse_loss(r_pred, reward_target, reduction="none"),
                commit_mask)
        if done is None:
            loss_done = visionIn.new_zeros(())
        else:
            loss_done = self.MaskedBatchMean(
                F.binary_cross_entropy_with_logits(
                    d_logit,
                    done.view(B).to(d_logit.dtype),
                    reduction="none"),
                commit_mask)

        information_gain_target = self.RealizedInformationGain(
            mu_q,
            logstd_q,
            mu_p,
            logstd_p).detach()
        loss_information_gain = self.MaskedBatchMean(
            F.smooth_l1_loss(
                torch.log1p(information_gain_pred),
                torch.log1p(information_gain_target),
                reduction="none"),
            commit_mask)

        loss_kl = self.MaskedBatchMean(
            BalancedKL(
                mu_q,
                logstd_q,
                mu_p,
                logstd_p,
                alpha=alphaKl,
                freeNats=freeNats),
            commit_mask)
        loss_pst_bind = 0.5 * (
            pst_binding_prior["loss_pst_bind"]
            + pst_binding_posterior["loss_pst_bind"])

        self.s4.x = self.MergeCommittedRows(
            x_next.detach(),
            self.s4.x,
            commit_mask)
        self._h = self.MergeCommittedRows(
            h_pred.detach(),
            self._h,
            commit_mask)
        self._z = self.MergeCommittedRows(
            z1.detach(),
            self._z,
            commit_mask)

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + 0.05 * loss_information_gain
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic
            + physCoef * phys_loss
            + pstBindCoef * loss_pst_bind
            + reg_A
            + 1e-1 * aux_moe)

        out: Dict[str, Any] = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_information_gain": loss_information_gain,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "loss_phys": phys_loss,
            "loss_pst_bind": loss_pst_bind,
            "loss_conn_reg": reg_A,

            "h_next": h_pred,
            "z_next": z1,
            "z_next_raw": mu_q_raw,
            "x_next": x_next,
            "s_next": s1,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "information_gain_pred": information_gain_pred,
            "information_gain_target": information_gain_target,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "recon_error": recon_error,
            "mu_p": mu_p,
            "mu_p_raw": mu_p_raw,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "mu_q_raw": mu_q_raw,
            "logstd_q": logstd_q,
            "action_enc": a_enc,
            "transition_embodied_action": transition_embodied_action,
            "transition_embodiment_context": transition_embodiment_context,
            "posterior_embodied_action": observation_embodied_action,
            "posterior_embodiment_context": observation_embodiment_context,
            "pst_binding": pst_binding_posterior,
            "pst_binding_prior": pst_binding_prior,
            "pst_binding_posterior": pst_binding_posterior,}

        if self.use_decoder and recon is not None:
            out["recon"] = recon
            out["recon_target"] = visionIn

        return out


    def ExportConsciousBank(
        self,
        topk: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        B = int(self._pst_slot_state.size(0))
        budget = max(0, int(topk))
        empty_tokens = torch.zeros(
            B, 0, self.state_dim,
            device=self.device, dtype=self.dtype)
        empty_valid = torch.zeros(
            B, 0,
            device=self.device, dtype=torch.bool)
        empty_real = torch.zeros(
            B, 0,
            device=self.device, dtype=self.dtype)
        empty_source = torch.zeros(
            B, 0,
            device=self.device, dtype=torch.long)
        if budget == 0:
            return {
                "tokens": empty_tokens,
                "valid": empty_valid,
                "source": empty_source,
                "age": empty_real,
                "confidence": empty_real,
                "staleness": empty_real,}
        entity_features = torch.cat([
            self._pst_slot_state,
            self._pst_realm,
            self._pst_motion_layer,
            self._pst_agency,
            self._pst_body_membership.unsqueeze(-1),
            self._pst_self_part_semantic,
            self._pst_entity_prob.unsqueeze(-1),
            self._pst_physical_interaction.unsqueeze(-1),
            self._pst_content_motion,
            self._pst_content_change.unsqueeze(-1),
            self._pst_display_surface.unsqueeze(-1),
            self._pst_surface_uv,
            self._pst_surface_uv_confidence.unsqueeze(-1),
            self._pst_verification.unsqueeze(-1),
            self._pst_entity_text_semantic,
            self._pst_entity_text_confidence.unsqueeze(-1),
            torch.tanh(
                self._pst_entity_text_revision.to(self.dtype).unsqueeze(-1)
                / 16.0),
            self._pst_entity_text_changed.to(self.dtype).unsqueeze(-1),], dim=-1)
        entity_valid = (
            (self._pst_entity_id >= 0)
            & (self._pst_slot_presence > 0.0))
        entity_tokens = self.entity_conscious_encoder(entity_features)
        entity_tokens = entity_tokens * entity_valid.unsqueeze(-1).to(
            entity_tokens.dtype)
        entity_scores = (
            self._pst_slot_presence
            * (1.0
               + self._pst_verification
               + self._pst_entity_text_confidence))
        entity_scores = entity_scores.masked_fill(~entity_valid, -torch.inf)
        entity_age = (
            self._pst_step.unsqueeze(-1) - self._pst_last_seen
        ).clamp_min(0).to(dtype=self.dtype)
        entity_confidence = (
            self._pst_slot_presence
            * (0.5 + 0.5 * self._pst_verification)
        ).clamp(0.0, 1.0)
        entity_staleness = 1.0 - torch.exp(-entity_age / 32.0)
        entity_source = torch.full_like(
            self._pst_entity_id,
            WorldConsciousSourceEntity,
            dtype=torch.long)

        if self._use_memory:
            cap = int(self._mem_vals.size(1))
            ar = torch.arange(
                cap,
                device=self._mem_vals.device).view(1, cap)
            history_valid = ar < self._mem_size.view(B, 1)
            history_tokens = self._mem_vals
            history_scores = self._mem_imp.masked_fill(
                ~history_valid, -torch.inf)
            history_age = (
                self._mem_global_step.unsqueeze(-1) - self._mem_steps
            ).clamp_min(0).to(dtype=self.dtype)
            history_confidence = torch.sigmoid(self._mem_imp)
            history_staleness = 1.0 - torch.exp(-history_age / 64.0)
            history_source = torch.full(
                (B, cap),
                WorldConsciousSourceHistory,
                device=self.device,
                dtype=torch.long)
        else:
            history_tokens = empty_tokens
            history_valid = empty_valid
            history_scores = empty_real
            history_age = empty_real
            history_confidence = empty_real
            history_staleness = empty_real
            history_source = empty_source
        all_tokens = torch.cat([entity_tokens, history_tokens], dim=1)
        all_valid = torch.cat([entity_valid, history_valid], dim=1)
        all_scores = torch.cat([entity_scores, history_scores], dim=1)
        all_source = torch.cat([entity_source, history_source], dim=1)
        all_age = torch.cat([entity_age, history_age], dim=1)
        all_confidence = torch.cat([
            entity_confidence,
            history_confidence,
        ], dim=1)
        all_staleness = torch.cat([
            entity_staleness,
            history_staleness,
        ], dim=1)
        count = min(budget, int(all_tokens.size(1)))
        if count == 0:
            return {
                "tokens": all_tokens[:, :0],
                "valid": all_valid[:, :0],
                "source": all_source[:, :0],
                "age": all_age[:, :0],
                "confidence": all_confidence[:, :0],
                "staleness": all_staleness[:, :0],}
        _, indices = torch.topk(all_scores, k=count, dim=1)
        tokens = torch.gather(
            all_tokens,
            1,
            indices.unsqueeze(-1).expand(B, count, self.state_dim))
        valid = torch.gather(all_valid, 1, indices)
        source = torch.gather(all_source, 1, indices)
        age = torch.gather(all_age, 1, indices)
        confidence = torch.gather(all_confidence, 1, indices)
        staleness = torch.gather(all_staleness, 1, indices)
        tokens = tokens * valid.unsqueeze(-1).to(tokens.dtype)
        return {
            "tokens": tokens,
            "valid": valid,
            "source": source,
            "age": age,
            "confidence": confidence,
            "staleness": staleness,}

class WorldOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: nn.Module,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        *,
        maxRank: int = 64,
        maxRankSmall: int = 16,
        maxRankHuge: int = 8,
        maxRankConnHeadUV: int = 16,
        maxRankConnHeadFull: int = 8,):
        self.maxRank = int(maxRank)
        self.maxRankSmall = int(maxRankSmall)
        self.maxRankHuge = int(maxRankHuge)
        self.maxRankConnHeadUV = int(maxRankConnHeadUV)
        self.maxRankConnHeadFull = int(maxRankConnHeadFull)
        super().__init__(
            base,
            initRankEach=initRankEach,
            autoRank=autoRank,
            evThreshold=evThreshold,
            gradEma=gradEma,)


        self.RestoreBaseTrainabilityAfterCommit()

    def ComputeNeuroSymbolicGate(
        self,
        baseGate: torch.Tensor,
        penalty: torch.Tensor,
        confidence: torch.Tensor,
        ) -> torch.Tensor:
        return self.base.ComputeNeuroSymbolicGate(
            baseGate, penalty, confidence)

    def ComputeMemoryImportance(
        self,
        rewardPrediction: torch.Tensor,
        doneProbability: torch.Tensor,
        nsProbability: Optional[torch.Tensor],
        nsPenalty: Optional[torch.Tensor],
        ) -> torch.Tensor:
        return self.base.ComputeMemoryImportance(
            rewardPrediction,
            doneProbability,
            nsProbability,
            nsPenalty)

    def DirectOnlineHeads(self) -> Tuple[nn.Module, ...]:
        return (
            self.base.contract_embodiment_adapter,
            self.base.embodiment_context_proj[3],
            self.base.embodied_action_proj[3],
            self.base.pst_binder.delta_mu[1],
            self.base.pst_binder.bind_gate[1],
            self.base.information_gain_context[1],
            self.base.information_gain_head,
            self.base.world_abstract_projector,
            self.base.entity_conscious_encoder,)

    def RestoreBaseTrainabilityAfterCommit(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for head in self.DirectOnlineHeads():
            for parameter in head.parameters():
                parameter.requires_grad_(True)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        V = int(self.base.vision_dim)
        A = int(self.base.action_dim)
        D = int(self.base.deter_dim)
        Z = int(self.base.stoch_dim)
        S = int(self.base.state_dim)
        X = int(self.base.ssm_dim)

        conn = self.base.conn
        phys = self.base.phys_refiner
        key = self.base.key_emb
        film = self.base.state_state_film
        nonlinear_flow = self.base.s4.nonlinear_flow

        def mk(
            name: str,
            inDim: int,
            outDim: int,
            maxRank: int,
            adapterStd: float = 1e-4,
            adapterScale: float = 1e-2,
            ) -> SiteSpec:
            inDim_i, outDim_i, maxRank_i = int(inDim), int(outDim), int(maxRank)

            def alloc(addRank: int, device: torch.device, dtype: torch.dtype):
                A_ = nn.Parameter(
                    torch.randn(addRank, inDim_i, device=device, dtype=dtype)
                    * float(adapterStd)) # [r,in]
                B_ = nn.Parameter(torch.zeros(outDim_i, addRank, device=device, dtype=dtype)) # [out,r]
                s_ = nn.Parameter(torch.tensor(float(adapterScale), device=device, dtype=dtype))
                return A_, B_, s_

            def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                s_eff = torch.tanh(s) * GetParametersScale(s)
                return s_eff * (b @ a) # [out,in]

            return SiteSpec(name, 1, inDim_i, outDim_i, maxRank_i, alloc, compose)

        specs: Dict[str, SiteSpec] = {}

        specs["obs_enc0"] = mk("obs_enc0", V, S, self.maxRank)
        specs["obs_enc1"] = mk("obs_enc1", S, Z, self.maxRank)

        specs["act_proj"] = mk("act_proj", A, Z, self.maxRank)

        specs["s4_in_to_ssm"] = mk("s4_in_to_ssm", 2 * Z, X, self.maxRank)
        specs["s4_ssm_to_deter"] = mk("s4_ssm_to_deter", X, D, self.maxRank)
        specs["s4_in_to_deter"] = mk("s4_in_to_deter", 2 * Z, D, self.maxRank)
        specs["s4_gate"] = mk("s4_gate", 2 * Z, X, self.maxRank)
        specs["s4_out_gate"] = mk("s4_out_gate", X, D, self.maxRank)
        specs["s4_nonlinear_left"] = mk(
            "s4_nonlinear_left", X, nonlinear_flow.rank, self.maxRank,
            adapterStd=X ** -0.5, adapterScale=0.5)
        specs["s4_nonlinear_right"] = mk(
            "s4_nonlinear_right", X, nonlinear_flow.rank, self.maxRank,
            adapterStd=X ** -0.5, adapterScale=0.5)
        specs["s4_nonlinear_context"] = mk(
            "s4_nonlinear_context", 2 * Z, nonlinear_flow.rank, self.maxRank,
            adapterStd=(2 * Z) ** -0.5, adapterScale=0.5)
        specs["s4_nonlinear_out"] = mk(
            "s4_nonlinear_out", nonlinear_flow.rank, X, self.maxRank,
            adapterStd=nonlinear_flow.rank ** -0.5, adapterScale=0.5)
        specs["s4_nonlinear_selectivity"] = mk(
            "s4_nonlinear_selectivity", 2 * Z, 1, self.maxRankSmall,
            adapterStd=(2 * Z) ** -0.5, adapterScale=0.5)

        specs["prior"] = mk("prior", D, 2 * Z, self.maxRank)
        specs["post"] = mk("post", D + Z, 2 * Z, self.maxRank)
        specs["state_proj"] = mk("state_proj", D + Z, S, self.maxRank)

        specs["rdone0"] = mk("rdone0", 2 * S + Z, 512, self.maxRank)
        specs["rdone1"] = mk("rdone1", 512, 256, self.maxRank)
        specs["rew"] = mk("rew", 256, 1, self.maxRankSmall)
        specs["done"] = mk("done", 256, 1, self.maxRankSmall)

        specs["obs_dec0"] = mk("obs_dec0", S, S, self.maxRank)
        specs["obs_dec1"] = mk("obs_dec1", S, V, self.maxRank)

        specs["mix_gate"] = mk("mix_gate", 3 * S, 3, self.maxRankSmall)

        specs["key_to_gb"] = mk("key_to_gb", Z, 2 * Z, self.maxRank)
        specs["key_mlp1"] = mk("key_mlp1", 4 * Z, int(key.mlp1.out_f), self.maxRank)
        specs["key_mlp2"] = mk("key_mlp2", int(key.mlp2.in_f), int(key.mlp2.out_f), self.maxRank)

        specs["ssfilm_e_to_gb"] = mk("ssfilm_e_to_gb", int(film.e_to_gb.in_f), int(film.e_to_gb.out_f), self.maxRank)
        specs["ssfilm_e_to_h"] = mk("ssfilm_e_to_h", int(film.e_to_h.in_f), int(film.e_to_h.out_f), self.maxRank)
        specs["ssfilm_delta0"] = mk("ssfilm_delta0", int(film.delta_mlp[0].in_f), int(film.delta_mlp[0].out_f), self.maxRank)
        specs["ssfilm_delta1"] = mk("ssfilm_delta1", int(film.delta_mlp[3].in_f), int(film.delta_mlp[3].out_f), self.maxRank)
        specs["ssfilm_to_gate"] = mk("ssfilm_to_gate", int(film.to_gate.in_f), int(film.to_gate.out_f), self.maxRank)

        specs["conn_enc_s"] = mk("conn_enc_s", int(conn.enc_s[1].linear.in_f), int(conn.enc_s[1].linear.out_f), self.maxRank)
        specs["conn_enc_a"] = mk("conn_enc_a", int(conn.enc_a[1].linear.in_f), int(conn.enc_a[1].linear.out_f), self.maxRank)

        specs["conn_film_gamma_a"] = mk("conn_film_gamma_a", int(conn.film_gamma_a.linear.in_f), int(conn.film_gamma_a.linear.out_f), self.maxRank)
        specs["conn_film_beta_a"] = mk("conn_film_beta_a", int(conn.film_beta_a.linear.in_f), int(conn.film_beta_a.linear.out_f), self.maxRank)

        for i, blk in enumerate(conn.blocks):
            specs[f"conn_blk{i}_ff0"] = mk(f"conn_blk{i}_ff0", int(blk.ff[0].linear.in_f), int(blk.ff[0].linear.out_f), self.maxRank)
            specs[f"conn_blk{i}_ff1"] = mk(f"conn_blk{i}_ff1", int(blk.ff[3].linear.in_f), int(blk.ff[3].linear.out_f), self.maxRank)

        if conn.use_lowrank:
            specs["conn_head_uv"] = mk("conn_head_uv", int(conn.head_uv.linear.in_f), int(conn.head_uv.linear.out_f), self.maxRankConnHeadUV)
        if conn.use_full:
            specs["conn_head_full"] = mk("conn_head_full", int(conn.head_full.linear.in_f), int(conn.head_full.linear.out_f), self.maxRankConnHeadFull)

        specs["conn_mix"] = mk("conn_mix", int(conn.mix.linear.in_f), int(conn.mix.linear.out_f), self.maxRankSmall)

        specs["phys_to_qp"] = mk("phys_to_qp", int(phys.to_qp.in_f), int(phys.to_qp.out_f), self.maxRank)
        specs["phys_from_qp"] = mk("phys_from_qp", int(phys.from_qp.in_f), int(phys.from_qp.out_f), self.maxRank)

        specs["phys_H0"] = mk("phys_H0", int(phys.HNet[0].in_f), int(phys.HNet[0].out_f), self.maxRank)
        specs["phys_H1"] = mk("phys_H1", int(phys.HNet[2].in_f), int(phys.HNet[2].out_f), self.maxRank)
        specs["phys_H2"] = mk("phys_H2", int(phys.HNet[4].in_f), int(phys.HNet[4].out_f), self.maxRankSmall)

        specs["phys_force0"] = mk("phys_force0", int(phys.force_net[0].in_f), int(phys.force_net[0].out_f), self.maxRank)
        specs["phys_force1"] = mk("phys_force1", int(phys.force_net[2].in_f), int(phys.force_net[2].out_f), self.maxRank)

        specs["phys_g_force"] = mk("phys_g_force", int(phys.g_force.in_f), int(phys.g_force.out_f), self.maxRank)
        specs["phys_g_phys"] = mk("phys_g_phys", int(phys.g_phys.in_f), int(phys.g_phys.out_f), self.maxRank)
        specs["phys_g_fuse"] = mk("phys_g_fuse", int(phys.g_fuse.in_f), int(phys.g_fuse.out_f), self.maxRank)

        return specs

    def EffW(self, gll, d2: Optional[torch.Tensor]) -> torch.Tensor:
        W = gll.target.weight
        d0 = gll.DeltaWeight()
        if d0 is not None:
            W = W + d0
        if d2 is not None:
            W = W + d2
        return W

    def Lin(self, x: torch.Tensor, gll, d2: Optional[torch.Tensor]) -> torch.Tensor:
        return F.linear(x, self.EffW(gll, d2), gll.target.bias)


    def ObsEnc(self, visionIn: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.base.obs_enc[0](visionIn)
        x = self.Lin(x, self.base.obs_enc[1], d.get("obs_enc0"))
        x = self.base.obs_enc[2](x)
        x = self.base.obs_enc[3](x)
        x = self.Lin(x, self.base.obs_enc[4], d.get("obs_enc1"))
        return x # [B,Z]

    def ActProj(self, actionEnc: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.Lin(actionEnc, self.base.act_proj[0], d.get("act_proj"))
        x = self.base.act_proj[1](x)
        x = self.base.act_proj[2](x)
        return x # [B,Z]

    def Prior(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(h, self.base.prior_net[0], d.get("prior"))

    def Post(self, hz: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(hz, self.base.post_net[0], d.get("post"))

    def StateProj(self, hz: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.base.state_proj[0](hz)
        x = self.Lin(x, self.base.state_proj[1], d.get("state_proj"))
        x = self.base.state_proj[2](x)
        return x # [B,S]

    def RdoneTrunk(
        self,
        inp: torch.Tensor,
        d: Dict[str, Optional[torch.Tensor]],
        *,
        deterministic: bool = False,
        ) -> torch.Tensor:
        x = self.Lin(inp, self.base.rdone_trunk[0], d.get("rdone0"))
        x = self.base.rdone_trunk[1](x)
        if deterministic:
            x = F.dropout(x, p=float(self.base.rdone_trunk[2].p), training=False)
        else:
            x = self.base.rdone_trunk[2](x)
        x = self.Lin(x, self.base.rdone_trunk[3], d.get("rdone1"))
        x = self.base.rdone_trunk[4](x)
        return x # [B,256]

    def Rew(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.base.BoundReward(self.Lin(h, self.base.rew_head[0], d.get("rew")))

    def Done(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(h, self.base.done_head[0], d.get("done"))

    def ObsDec(self, s: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.Lin(s, self.base.obs_dec[0], d.get("obs_dec0"))
        x = self.base.obs_dec[1](x)
        x = self.Lin(x, self.base.obs_dec[2], d.get("obs_dec1"))
        return x

    def MixGate(self, g_in: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(g_in, self.base.mix_gate[0], d.get("mix_gate"))


    def S4NonlinearFlow(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        d: Dict[str, Optional[torch.Tensor]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        flow = self.base.s4.nonlinear_flow
        bounded_state, bounded_control = flow.BoundedInputs(x, u)
        interaction = flow.Interaction(
            flow.ProjectWeight(
                bounded_state,
                self.EffW(flow.left, d.get("s4_nonlinear_left")),
                flow.left.target.bias,
                flow.left_anchor),
            flow.ProjectWeight(
                bounded_state,
                self.EffW(flow.right, d.get("s4_nonlinear_right")),
                flow.right.target.bias,
                flow.right_anchor),
            flow.ProjectWeight(
                bounded_control,
                self.EffW(flow.context, d.get("s4_nonlinear_context")),
                flow.context.target.bias,
                flow.context_anchor))
        return (
            flow.Target(
                x,
                flow.ProjectWeight(
                    interaction,
                    self.EffW(flow.out, d.get("s4_nonlinear_out")),
                    flow.out.target.bias,
                    flow.out_anchor)),
            flow.Mix(self.Lin(
                bounded_control,
                flow.selectivity,
                d.get("s4_nonlinear_selectivity"))))


    def S4Step(self, zPrev: torch.Tensor, a_t: torch.Tensor, *, updateState: bool, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        s4 = self.base.s4
        u = torch.cat([zPrev, a_t], dim=-1) # [B,2Z]

        g = torch.sigmoid(self.Lin(u, s4.gate, d.get("s4_gate"))) # [B,X]
        linear_target = self.Lin(
            u, s4.in_to_ssm, d.get("s4_in_to_ssm")) * g # [B,X]

        linear_state = s4.LinearStateTransition(s4.x, linear_target)
        nonlinear_target, nonlinear_mix = self.S4NonlinearFlow(
            linear_state, u, d)
        x_next = s4.StateTransition(
            linear_state, nonlinear_target, nonlinear_mix)
        y_lin = self.Lin(x_next, s4.ssm_to_deter, d.get("s4_ssm_to_deter")) + self.Lin(u, s4.in_to_deter, d.get("s4_in_to_deter"))
        y_glu = y_lin * torch.sigmoid(self.Lin(x_next, s4.out_gate, d.get("s4_out_gate")))
        y = s4.ln_y(y_glu)
        y = y + s4.ffn(s4.ln_ffn(y))

        if updateState:
            s4.x = x_next.detach()
        return y # [B,D]

    def S4StepWithX(self, zPrev: torch.Tensor, a_t: torch.Tensor, x: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        s4 = self.base.s4
        u = torch.cat([zPrev, a_t], dim=-1)

        g = torch.sigmoid(self.Lin(u, s4.gate, d.get("s4_gate")))
        linear_target = self.Lin(
            u, s4.in_to_ssm, d.get("s4_in_to_ssm")) * g

        linear_state = s4.LinearStateTransition(x, linear_target)
        nonlinear_target, nonlinear_mix = self.S4NonlinearFlow(
            linear_state, u, d)
        x_next = s4.StateTransition(
            linear_state, nonlinear_target, nonlinear_mix)
        y_lin = self.Lin(x_next, s4.ssm_to_deter, d.get("s4_ssm_to_deter")) + self.Lin(u, s4.in_to_deter, d.get("s4_in_to_deter"))
        y_glu = y_lin * torch.sigmoid(self.Lin(x_next, s4.out_gate, d.get("s4_out_gate")))
        y = s4.ln_y(y_glu)
        y = y + s4.ffn(s4.ln_ffn(y))
        return y, x_next.detach()


    def KeyEmbed(self, base_e: torch.Tensor, actionEmbed: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        ke = self.base.key_emb
        e = ke.ln_e(base_e)
        a = ke.ln_a(actionEmbed)

        gb = self.Lin(a, ke.to_gb, d.get("key_to_gb"))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)

        e_film = (1.0 + gamma) * e + beta

        feat = torch.cat([e_film, a, e_film * a, e_film - a], dim=-1)
        feat = ke.ln_feat(feat)

        h = F.silu(self.Lin(feat, ke.mlp1, d.get("key_mlp1")))
        h = ke.drop(h)
        k = self.Lin(h, ke.mlp2, d.get("key_mlp2"))

        k = F.normalize(k, dim=-1, eps=1e-6)
        return k


    def FiLMHResidual(self, h: torch.Tensor, e: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        fr = self.base.state_state_film
        h0 = fr.ln_h(h)
        e0 = fr.ln_e(e)

        gb = self.Lin(e0, fr.e_to_gb, d.get("ssfilm_e_to_gb"))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = fr.film_scale * torch.tanh(gamma)
        beta = fr.film_scale * torch.tanh(beta)

        h_film = (1.0 + gamma) * h0 + beta

        e_h = self.Lin(e0, fr.e_to_h, d.get("ssfilm_e_to_h"))
        e_h = fr.film_scale * torch.tanh(e_h)

        feat = torch.cat([h_film, e_h, h_film * e_h, h_film - e_h], dim=-1)
        feat = fr.delta_ln(feat)

        y = self.Lin(feat, fr.delta_mlp[0], d.get("ssfilm_delta0"))
        y = fr.delta_mlp[1](y)
        y = fr.delta_mlp[2](y)
        delta = self.Lin(y, fr.delta_mlp[3], d.get("ssfilm_delta1"))

        gate_in = torch.cat([h_film, e_h], dim=-1)
        gate = torch.sigmoid(self.Lin(gate_in, fr.to_gate, d.get("ssfilm_to_gate")))

        h_out = h + gate * delta
        if fr.use_out_ln:
            h_out = fr.out_ln(h_out)
        return h_out

    def ConnNet(self, sBase: torch.Tensor, actPrev: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        conn = self.base.conn
        B = int(sBase.size(0))

        hs = conn.enc_s[0](sBase)
        hs = self.Lin(hs, conn.enc_s[1].linear, d.get("conn_enc_s"))
        hs = conn.enc_s[2](hs)

        ha = conn.enc_a[0](actPrev)
        ha = self.Lin(ha, conn.enc_a[1].linear, d.get("conn_enc_a"))
        ha = conn.enc_a[2](ha)

        g = torch.tanh(self.Lin(ha, conn.film_gamma_a.linear, d.get("conn_film_gamma_a")))
        b = self.Lin(ha, conn.film_beta_a.linear, d.get("conn_film_beta_a"))

        h = hs
        for i, blk in enumerate(conn.blocks):
            y = (1.0 + g) * h + b
            y = blk.ln(y)

            y = self.Lin(y, blk.ff[0].linear, d.get(f"conn_blk{i}_ff0"))
            y = blk.ff[1](y)
            y = blk.ff[2](y)
            y = self.Lin(y, blk.ff[3].linear, d.get(f"conn_blk{i}_ff1"))

            h = h + blk.alpha * y

        A_list: List[torch.Tensor] = []

        if conn.use_lowrank:
            uv = self.Lin(h, conn.head_uv.linear, d.get("conn_head_uv"))
            U, Vv = uv.split(conn.S * conn.r, dim=-1)
            U = U.view(B, conn.S, conn.r)
            Vv = Vv.view(B, conn.S, conn.r)
            A_list.append(U @ Vv.transpose(1, 2) - Vv @ U.transpose(1, 2))

        if conn.use_full:
            M = self.Lin(h, conn.head_full.linear, d.get("conn_head_full")).view(B, conn.S, conn.S)
            A_list.append(0.5 * (M - M.transpose(1, 2)))

        if not A_list:
            A = torch.zeros(B, conn.S, conn.S, device=sBase.device, dtype=sBase.dtype)
        elif len(A_list) == 1:
            A = A_list[0]
        else:
            w = F.softmax(self.Lin(h, conn.mix.linear, d.get("conn_mix")), dim=-1)
            A = w[:, :1].view(B, 1, 1) * A_list[0] + w[:, 1:2].view(B, 1, 1) * A_list[1]

        if conn.norm_clip and conn.norm_clip > 0:
            fro = A.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), conn.norm_clip / fro).view(B, 1, 1)
            A = A * scale

        return A # [B,S,S]


    def PhysRefiner(self, hPrev: torch.Tensor, action: torch.Tensor, hS4: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]):
        pr = self.base.phys_refiner
        training_mode = bool(self.training and torch.is_grad_enabled())
        inference_mode_active = torch.is_inference_mode_enabled()
        create_graph = bool(training_mode)
        dt_sub = pr.dt / float(pr.substeps)

        if training_mode:
            H_start = None
            H_end = None
            work_acc = hPrev.new_zeros(hPrev.size(0), 1)
            smooth_acc = hPrev.new_tensor(0.0)

        def HNet(qp: torch.Tensor) -> torch.Tensor:
            x = self.Lin(qp, pr.HNet[0], d.get("phys_H0"))
            x = pr.HNet[1](x)
            x = self.Lin(x, pr.HNet[2], d.get("phys_H1"))
            x = pr.HNet[3](x)
            x = self.Lin(x, pr.HNet[4], d.get("phys_H2"))
            return x # [B,1]

        def ForceNet(fa: torch.Tensor) -> torch.Tensor:
            x = self.Lin(fa, pr.force_net[0], d.get("phys_force0"))
            x = pr.force_net[1](x)
            x = self.Lin(x, pr.force_net[2], d.get("phys_force1"))
            return x # [B,Q]

        def HAndGrad(qp: torch.Tensor, create_graph_: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            H = HNet(qp)
            g = torch.autograd.grad(
                H.sum(), qp,
                create_graph=create_graph_,
                retain_graph=create_graph_,
                allow_unused=False,
            )[0] # [B,P]
            return H, g

        def SymplecticLeapfrog(q: torch.Tensor, p: torch.Tensor, dt: float, create_graph_: bool):
            qp0 = torch.cat([q, p], dim=-1)
            H0, g0 = HAndGrad(qp0, create_graph_=create_graph_)
            dH_dq0, _ = g0.chunk(2, dim=-1)

            p_half = p - 0.5 * dt * dH_dq0

            qp_mid = torch.cat([q, p_half], dim=-1)
            _, gm = HAndGrad(qp_mid, create_graph_=create_graph_)
            _, dH_dp_mid = gm.chunk(2, dim=-1)

            q1 = q + dt * dH_dp_mid

            qp_for_p = torch.cat([q1, p_half], dim=-1)
            H1, g2 = HAndGrad(qp_for_p, create_graph_=create_graph_)
            dH_dq2, _ = g2.chunk(2, dim=-1)

            p1 = p_half - 0.5 * dt * dH_dq2
            return q1, p1, H0, H1, dH_dp_mid

        def ClampResidual(delta_: torch.Tensor, base_: torch.Tensor, ratio: float) -> torch.Tensor:
            eps = 1e-8
            dnorm = delta_.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
            bnorm = base_.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-3
            maxn = ratio * bnorm + eps
            scale = (maxn / dnorm).clamp(max=1.0)
            return delta_ * scale

        with torch.inference_mode(False), torch.enable_grad():
            h_prev_work = hPrev.detach().clone() if inference_mode_active else hPrev
            action_work = action.detach().clone() if inference_mode_active else action
            qp = self.Lin(h_prev_work, pr.to_qp, d.get("phys_to_qp"))

            if not qp.requires_grad:
                qp = qp.detach().requires_grad_(True)

            q, p = qp.chunk(2, dim=-1)

            for i in range(pr.substeps):
                h_cur = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))

                fa0_inp = torch.cat([h_cur, action_work], dim=-1)
                F0 = ForceNet(fa0_inp) * torch.sigmoid(self.Lin(fa0_inp, pr.g_force, d.get("phys_g_force")))

                if pr.dampP > 0.0:
                    p = p * p.new_tensor(-pr.dampP * dt_sub).exp()

                p = p + 0.5 * dt_sub * F0

                q, p, H0, H1, dH_dp_mid = SymplecticLeapfrog(q, p, dt_sub, create_graph_=create_graph)

                if training_mode:
                    if i == 0:
                        H_start = H0
                    H_end = H1

                h_mid = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))
                fa1_inp = torch.cat([h_mid, action_work], dim=-1)
                F1 = ForceNet(fa1_inp) * torch.sigmoid(self.Lin(fa1_inp, pr.g_force, d.get("phys_g_force")))

                p = p + 0.5 * dt_sub * F1

                if training_mode:
                    smooth_acc = smooth_acc + (F1 - F0).pow(2).mean()
                    F_avg = 0.5 * (F0 + F1)
                    work_acc = work_acc + (dH_dp_mid * F_avg).sum(dim=-1, keepdim=True) * dt_sub

        h_phys_raw = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))

        d_corr = h_phys_raw - hS4
        gph = torch.sigmoid(self.Lin(torch.cat([hPrev, action], dim=-1), pr.g_phys, d.get("phys_g_phys")))
        d_corr = d_corr * gph

        base_ = hS4 - hPrev
        d_corr = ClampResidual(d_corr, base_, ratio=pr.clamp_ratio)

        alpha = torch.sigmoid(self.Lin(torch.cat([hPrev, action, hS4], dim=-1), pr.g_fuse, d.get("phys_g_fuse")))
        h_fused = hS4 + alpha * d_corr

        if not training_mode:
            return h_fused, None, None

        if (H_start is None) or (H_end is None):
            e_work = hPrev.new_tensor(0.0)
        else:
            denom = H_start.detach().abs().mean().clamp_min(1e-6)
            dH = (H_end - H_start)
            e_work = ((dH - work_acc) / denom).pow(2).mean()

        e_smooth = smooth_acc / float(pr.substeps)
        e_delta = d_corr.pow(2).mean()

        loss = pr.l_work * e_work + pr.l_smooth * e_smooth + pr.l_delta * e_delta
        aux = {"L_work": e_work.detach(), "L_smooth": e_smooth.detach(), "L_delta": e_delta.detach()}
        return h_fused, loss, aux


    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,) -> Dict[str, torch.Tensor]:

        visionIn = x
        actionEnc = kwargs["actionEnc"]
        reward = kwargs.get("reward")
        done = kwargs.get("done")
        physicalState = kwargs["physicalState"]
        embodimentState = kwargs["embodimentState"]
        transitionPhysicalState = kwargs["transitionPhysicalState"]
        transitionEmbodimentState = kwargs["transitionEmbodimentState"]
        observerMotion = kwargs["observerMotion"]
        observerMotionValid = kwargs["observerMotionValid"]

        sample_arg = kwargs.get("sample")
        sample = bool(self.training) if sample_arg is None else bool(sample_arg)
        update_memory_arg = kwargs.get("updateMemory")
        update_memory = bool(self.training) if update_memory_arg is None else bool(update_memory_arg)
        update_memory = update_memory and bool(self.training)
        alphaKl = kwargs.get("alphaKl", 0.8)
        freeNats = kwargs.get("freeNats", 1.0)
        reconCoef = kwargs.get("reconCoef", 1.0)
        rewardCoef = kwargs.get("rewardCoef", 1.0)
        doneCoef = kwargs.get("doneCoef", 1.0)
        nsCoef = kwargs.get("nsCoef", 1.0)
        nsDistillCoef = kwargs.get("nsDistillCoef", 0.0)
        nsPriorLogicCoef = kwargs.get("nsPriorLogicCoef", 1e-3)
        physCoef = kwargs.get("physCoef", 1e-4)
        pstBindCoef = kwargs.get("pstBindCoef", 0.05)

        B = int(visionIn.size(0))
        self.base.EnsureB(B)
        commit_mask = self.base.ResolveCommitMask(
            kwargs.get("commitMask"),
            B)

        d = deltasPerLayer[0] if (deltasPerLayer is not None) else {}

        h0 = self.base._h
        z0 = self.base._z

        a_enc = actionEnc
        transition_embodied_action, transition_embodiment_context = self.base.BuildEmbodiedAction(
            transitionPhysicalState,
            a_enc,
            transitionEmbodimentState,
            observerMotion,
            observerMotionValid)
        observation_embodied_action, observation_embodiment_context = self.base.BuildEmbodiedAction(
            physicalState,
            a_enc,
            embodimentState,
            observerMotion,
            observerMotionValid)
        a_t = self.ActProj(transition_embodied_action, d) # [B, stochDim]

        h_pred, x_next = self.S4StepWithX(
            z0,
            a_t,
            self.base.s4.x,
            d) # [B, deterDim]

        mu_p, logstd_p = self.Prior(h_pred, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        logits_pr = None
        P_pr_train = None
        ns_prior_logic = visionIn.new_tensor(0.0)

        if self.base._ns_enabled:
            logits_pr = self.base.ns_head_prior(
                h_pred,
                sampleMask=commit_mask)
            P_pr_raw = torch.sigmoid(logits_pr)
            P_pr_train = self.base.NsProjectProbs(P_pr_raw)

            dmu_p = self.base.ns_to_delta_mu(P_pr_train)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_pred, dmu_p], dim=-1)))

            _, pen_pr = self.base.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)

            conf = self.base.NsConfidence(P_pr_train).mean(dim=-1, keepdim=True)
            gate = self.ComputeNeuroSymbolicGate(
                base_gate, pen_pr, conf)

            mu_p = mu_p + gate * dmu_p

            if nsPriorLogicCoef > 0.0:
                ns_prior_logic, _ = self.base.NsLogicLosses(
                    P_pr_train,
                    sampleMask=commit_mask)

        mu_p_raw = mu_p
        pst_binding_prior = self.base.BindPhysicalMu(
            h_pred,
            mu_p,
            x_next,
            transitionPhysicalState,
            transition_embodied_action,
            transition_embodiment_context,
            sampleMask=commit_mask)
        mu_p = pst_binding_prior["bound_mu"]

        raw_e = self.ObsEnc(visionIn, d) # [B, stochDim]

        mu_q, logstd_q = self.Post(torch.cat([h_pred, raw_e], dim=-1), d).chunk(2, dim=-1)
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        ns_loss = visionIn.new_tensor(0.0)
        ns_distill = visionIn.new_tensor(0.0)
        logits_q = None
        Q_train = None
        pen_q = None

        if self.base._ns_enabled:
            logits_q = self.base.ns_head_post(
                torch.cat([h_pred, raw_e], dim=-1),
                sampleMask=commit_mask)
            P_q_raw = torch.sigmoid(logits_q)
            Q_train = self.base.NsProjectProbs(P_q_raw)

            dmu_q = self.base.ns_to_delta_mu(Q_train)
            base_gate_q = torch.sigmoid(self.base.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu_q], dim=-1)))

            _, pen_q = self.base.NsProjectRuntime(P_q_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)

            conf_q = self.base.NsConfidence(Q_train).mean(dim=-1, keepdim=True)
            gate_q = self.ComputeNeuroSymbolicGate(
                base_gate_q, pen_q, conf_q)

            mu_q = mu_q + gate_q * dmu_q

            ns_loss, _ = self.base.NsLogicLosses(
                Q_train,
                sampleMask=commit_mask)

            if (logits_pr is not None) and (nsDistillCoef > 0.0):
                with torch.no_grad():
                    P_teacher = torch.sigmoid(logits_q)
                ns_distill = self.base.MaskedBatchMean(
                    F.binary_cross_entropy_with_logits(
                        logits_pr,
                        P_teacher,
                        reduction="none"),
                    commit_mask)

        mu_q_raw = mu_q
        pst_binding_posterior = self.base.BindPhysicalMu(
            h_pred,
            mu_q,
            x_next,
            physicalState,
            observation_embodied_action,
            observation_embodiment_context,
            sampleMask=commit_mask)
        mu_q = pst_binding_posterior["bound_mu"]

        if sample:
            z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
        else:
            z1 = mu_q

        s_base = self.StateProj(torch.cat([h_pred, z1], dim=-1), d)
        s_prev_base = self.StateProj(torch.cat([h0, z0], dim=-1), d)

        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)

        previous_connection_valid = (
            self.base._A_prev_valid
            if (
                self.base._A_prev_valid is not None
                and tuple(self.base._A_prev_valid.shape) == (B,)
            )
            else torch.zeros(B, device=A_t.device, dtype=torch.bool))
        if self.base._A_prev is not None and self.base._A_prev.shape == A_t.shape:
            previous_connection = self.base._A_prev
            previous_mask = previous_connection_valid.view(B, 1, 1)
            prevA = torch.where(
                previous_mask,
                previous_connection,
                A_t.detach())
        else:
            previous_connection = torch.zeros_like(A_t)
            prevA = None
        reg_A = self.base.conn.ComputeGeomReg(
            A_t,
            prevA,
            sampleMask=commit_mask)
        self.base._A_prev = self.base.MergeCommittedRows(
            A_t.detach(),
            previous_connection,
            commit_mask)
        self.base._A_prev_valid = previous_connection_valid | commit_mask

        active_rows = torch.nonzero(commit_mask, as_tuple=False).flatten()
        h_phys = h_pred
        phys_loss = visionIn.new_zeros(())
        if active_rows.numel() > 0:
            active_h_phys, active_phys_loss, _ = self.PhysRefiner(
                h0.index_select(0, active_rows),
                a_t.index_select(0, active_rows),
                h_pred.index_select(0, active_rows),
                d)
            h_phys = h_phys.index_copy(0, active_rows, active_h_phys)
            if active_phys_loss is not None:
                phys_loss = active_phys_loss
        s_phys = self.StateProj(torch.cat([h_phys, z1], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1)
        w = F.softmax(self.MixGate(g_in, d), dim=-1)
        s1 = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        dynamics_state = s1
        inp = torch.cat([s_base, dynamics_state, a_t], dim=-1)
        trunk = self.RdoneTrunk(self.base.rdone_ln(inp), d)
        r_pred = self.Rew(trunk, d).squeeze(-1)
        d_logit = self.Done(trunk, d).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)
        information_gain_pred = self.base.ExpectedInformationGain(
            h_pred,
            mu_p,
            a_t,
            transition_embodiment_context)

        if self.base._use_memory:
            key = self.KeyEmbed(raw_e, a_t, d)
            mem_retrieved = self.base.MemRetrieve(
                key,
                updateImportance=update_memory,
                commitMask=commit_mask)
            if update_memory:
                imp = self.ComputeMemoryImportance(
                    r_pred, d_prob, Q_train, pen_q)
                self.base.MemAdd(
                    key.detach(),
                    dynamics_state.detach(),
                    imp.detach(),
                    commitMask=commit_mask)

            if mem_retrieved is not None:
                mem_s, mem_mask = mem_retrieved
                s_memory = self.FiLMHResidual(dynamics_state, mem_s, d)
                s1 = torch.where(mem_mask.view(B, 1), s_memory, dynamics_state)

        loss_recon = visionIn.new_tensor(0.0)
        recon_error = visionIn.new_zeros(B)
        recon = None
        if self.base.use_decoder:
            recon = self.ObsDec(s1, d)
            normalized_shape = (int(recon.size(-1)),)
            target = F.layer_norm(
                visionIn.detach(),
                normalized_shape=normalized_shape)
            recon_n = F.layer_norm(
                recon,
                normalized_shape=normalized_shape)

            recon_error = (recon_n - target).pow(2).mean(dim=-1)
            loss_recon = self.base.MaskedBatchMean(
                recon_error,
                commit_mask)

        aux_moe = visionIn.new_tensor(0.0)
        if self.base._ns_enabled:
            aux_moe = self.base.ns_head_prior.GetAuxLoss() + self.base.ns_head_post.GetAuxLoss()

        if reward is None:
            loss_reward = visionIn.new_zeros(())
        else:
            reward_target = reward.view(B).clamp(float(self.base.reward_min), float(self.base.reward_max))
            loss_reward = self.base.MaskedBatchMean(
                F.mse_loss(r_pred, reward_target, reduction="none"),
                commit_mask)
        if done is None:
            loss_done = visionIn.new_zeros(())
        else:
            loss_done = self.base.MaskedBatchMean(
                F.binary_cross_entropy_with_logits(
                    d_logit,
                    done.view(B).to(d_logit.dtype),
                    reduction="none"),
                commit_mask)
        information_gain_target = self.base.RealizedInformationGain(
            mu_q,
            logstd_q,
            mu_p,
            logstd_p).detach()
        loss_information_gain = self.base.MaskedBatchMean(
            F.smooth_l1_loss(
                torch.log1p(information_gain_pred),
                torch.log1p(information_gain_target),
                reduction="none"),
            commit_mask)
        loss_kl = self.base.MaskedBatchMean(
            BalancedKL(
                mu_q,
                logstd_q,
                mu_p,
                logstd_p,
                alpha=alphaKl,
                freeNats=freeNats),
            commit_mask)
        loss_pst_bind = 0.5 * (
            pst_binding_prior["loss_pst_bind"]
            + pst_binding_posterior["loss_pst_bind"])

        self.base.s4.x = self.base.MergeCommittedRows(
            x_next.detach(),
            self.base.s4.x,
            commit_mask)
        self.base._h = self.base.MergeCommittedRows(
            h_pred.detach(),
            self.base._h,
            commit_mask)
        self.base._z = self.base.MergeCommittedRows(
            z1.detach(),
            self.base._z,
            commit_mask)

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + 0.05 * loss_information_gain
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic
            + physCoef * phys_loss
            + pstBindCoef * loss_pst_bind
            + reg_A
            + 1e-1 * aux_moe)

        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_information_gain": loss_information_gain,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "loss_phys": phys_loss,
            "loss_pst_bind": loss_pst_bind,
            "loss_conn_reg": reg_A,
            "h_next": h_pred,
            "z_next": z1,
            "z_next_raw": mu_q_raw,
            "x_next": x_next,
            "s_next": s1,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "information_gain_pred": information_gain_pred,
            "information_gain_target": information_gain_target,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "recon_error": recon_error,
            "mu_p": mu_p,
            "mu_p_raw": mu_p_raw,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "mu_q_raw": mu_q_raw,
            "logstd_q": logstd_q,
            "action_enc": a_enc,
            "transition_embodied_action": transition_embodied_action,
            "transition_embodiment_context": transition_embodiment_context,
            "posterior_embodied_action": observation_embodied_action,
            "posterior_embodiment_context": observation_embodiment_context,
            "pst_binding": pst_binding_posterior,
            "pst_binding_prior": pst_binding_prior,
            "pst_binding_posterior": pst_binding_posterior,}

        if self.base.use_decoder and recon is not None:
            out["recon"] = recon
            out["recon_target"] = visionIn

        return out

    @torch.no_grad()
    def StepStationaryObserverPriorOnly(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        embodimentState: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        return self.StepPriorOnly(
            hPrev,
            zPrev,
            s4xPrev,
            actionEnc,
            physicalState=physicalState,
            embodimentState=embodimentState,
            observerMotion=self.base.StationaryObserverMotion(actionEnc),
            observerMotionValid=torch.zeros(
                actionEnc.size(0),
                device=actionEnc.device,
                dtype=torch.bool),
            sample=sample,)

    @torch.no_grad()
    def StepPriorOnly(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool = False,
        ) -> Dict[str, torch.Tensor]:
        deltas = [self.ComposeLayerDelta(layerIdx) for layerIdx in range(self.layerCount)]
        return self.StepPriorWithDeltas(
            hPrev,
            zPrev,
            s4xPrev,
            actionEnc,
            sample=sample,
            deltasPerLayer=deltas,
            physicalState=physicalState,
            embodimentState=embodimentState,
            observerMotion=observerMotion,
            observerMotionValid=observerMotionValid)

    @torch.no_grad()
    def StepPriorWithDeltas(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,
        deltasPerLayer: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        **kwargs,) -> Dict[str, torch.Tensor]:
        physicalState = kwargs["physicalState"]
        embodimentState = kwargs["embodimentState"]
        observerMotion = kwargs["observerMotion"]
        observerMotionValid = kwargs["observerMotionValid"]
        B = int(actionEnc.size(0))
        device, dtype = self.base.device, self.base.dtype

        d = deltasPerLayer[0] if (deltasPerLayer is not None) else {}

        if hPrev is None or zPrev is None or s4xPrev is None:
            hPrev = torch.zeros(B, self.base.deter_dim, device=device, dtype=dtype)
            zPrev = torch.zeros(B, self.base.stoch_dim, device=device, dtype=dtype)
            s4xPrev = torch.zeros(B, self.base.ssm_dim, device=device, dtype=dtype)

        embodied_action, embodiment_context = self.base.BuildEmbodiedAction(
            physicalState,
            actionEnc,
            embodimentState,
            observerMotion,
            observerMotionValid)
        a_t = self.ActProj(embodied_action, d)
        h_next, s4x_next = self.S4StepWithX(zPrev, a_t, s4xPrev, d)

        mu_p, logstd_p = self.Prior(h_next, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self.base._ns_enabled:
            ns_logits = self.base.ns_head_prior(
                h_next, deterministic=True, updateAux=False)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.base.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.base.NsConfidence(Q).mean(dim=-1, keepdim=True)

            dmu = self.base.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate = self.ComputeNeuroSymbolicGate(base_gate, pen, conf)

            mu_p = mu_p + gate * dmu

        mu_p_raw = mu_p
        pst_binding = self.base.BindPhysicalMu(h_next, mu_p, s4x_next, physicalState, embodied_action, embodiment_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.StateProj(torch.cat([h_next, z_next], dim=-1), d)
        s_prev_base = self.StateProj(torch.cat([hPrev, zPrev], dim=-1), d)

        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)

        h_phys, _, _ = self.PhysRefiner(hPrev, a_t, h_next, d)
        s_phys = self.StateProj(torch.cat([h_phys, z_next], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1)

        w = F.softmax(self.MixGate(g_in, d), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        inp = torch.cat([s_base, s_next, a_t], dim=-1)
        h = self.RdoneTrunk(self.base.rdone_ln(inp), d, deterministic=True)

        r_pred = self.Rew(h, d).squeeze(-1)
        d_logit = self.Done(h, d).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)
        information_gain_pred = self.base.ExpectedInformationGain(
            h_next,
            mu_p,
            a_t,
            embodiment_context)

        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": s4x_next,
            "s_next": s_next,
            "embodied_action": embodied_action,
            "embodiment_context": embodiment_context,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "information_gain_pred": information_gain_pred,
            "mu_p": mu_p,
            "logstd_p": logstd_p,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.base.ExportState()

    @torch.no_grad()
    def PriorRolloutSequence(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalStateSequence: Dict[str, torch.Tensor],
        actionEncSequence: torch.Tensor,
        embodimentStateSequence: torch.Tensor,
        observerMotionSequence: torch.Tensor,
        observerMotionValidSequence: torch.Tensor,
        sample: bool = False,
    ) -> Dict[str, torch.Tensor]:
        return self.base.BuildPriorRolloutSequence(
            self.StepPriorOnly,
            hPrev,
            zPrev,
            s4xPrev,
            physicalStateSequence,
            actionEncSequence,
            embodimentStateSequence,
            observerMotionSequence,
            observerMotionValidSequence,
            sample=sample)

    def PriorRolloutFromStateAction(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        d = self.ComposeLayerDelta(0)
        return self.PriorRolloutFromStateActionWithDeltas(
            hPrev,
            zPrev,
            s4xPrev,
            physicalState=physicalState,
            actionEnc=actionEnc,
            embodimentState=embodimentState,
            observerMotion=observerMotion,
            observerMotionValid=observerMotionValid,
            sample=sample,
            d=d)

    def PriorRolloutFromStateActionWithDeltas(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool,
        d: Dict[str, Optional[torch.Tensor]],) -> Dict[str, torch.Tensor]:
        s_prev_base = self.StateProj(torch.cat([hPrev, zPrev], dim=-1), d)

        embodied_action, embodiment_context = self.base.BuildEmbodiedAction(
            physicalState,
            actionEnc,
            embodimentState,
            observerMotion,
            observerMotionValid)
        a_t = self.ActProj(embodied_action, d)
        h_next, x_next = self.S4StepWithX(zPrev, a_t, s4xPrev, d)
        mu_p, logstd_p = self.Prior(h_next, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self.base._ns_enabled:
            ns_logits = self.base.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.base.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.base.NsConfidence(Q).mean(dim=-1, keepdim=True)
            dmu = self.base.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate = self.ComputeNeuroSymbolicGate(base_gate, pen, conf)
            mu_p = mu_p + gate * dmu

        mu_p_raw = mu_p
        pst_binding = self.base.BindPhysicalMu(h_next, mu_p, x_next, physicalState, embodied_action, embodiment_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.StateProj(torch.cat([h_next, z_next], dim=-1), d)
        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)
        h_phys, _, _ = self.PhysRefiner(hPrev, a_t, h_next, d)
        s_phys = self.StateProj(torch.cat([h_phys, z_next], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        w = F.softmax(self.MixGate(torch.cat([s_base, d_tr, d_ph], dim=-1), d), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        trunk = self.RdoneTrunk(self.base.rdone_ln(torch.cat([s_base, s_next, a_t], dim=-1)), d)
        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": x_next,
            "s_next": s_next,
            "action_enc": actionEnc,
            "embodied_action": embodied_action,
            "embodiment_context": embodiment_context,
            "r_pred": self.Rew(trunk, d).squeeze(-1),
            "d_prob": torch.sigmoid(self.Done(trunk, d).squeeze(-1)),
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

    def PredictNextVisualFromPosterior(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool = False,) -> Dict[str, Any]:
        d = self.ComposeLayerDelta(0)
        return self.PredictNextVisualFromPosteriorWithDeltas(
            h,
            z,
            s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            embodimentState=embodimentState,
            observerMotion=observerMotion,
            observerMotionValid=observerMotionValid,
            sample=sample,
            d=d)

    def PredictNextVisualWithStationaryObserver(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        sample: bool = False,) -> Dict[str, Any]:
        return self.PredictNextVisualFromPosterior(
            h,
            z,
            s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            embodimentState=embodimentState,
            observerMotion=self.base.StationaryObserverMotion(actionEnc),
            observerMotionValid=torch.zeros(
                actionEnc.size(0),
                device=actionEnc.device,
                dtype=torch.bool),
            sample=sample,)

    def PredictNextVisualFromPosteriorWithDeltas(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentState: torch.Tensor,
        observerMotion: torch.Tensor,
        observerMotionValid: torch.Tensor,
        sample: bool,
        d: Dict[str, Optional[torch.Tensor]],) -> Dict[str, Any]:
        rollout = self.PriorRolloutFromStateActionWithDeltas(
            h,
            z,
            s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            embodimentState=embodimentState,
            observerMotion=observerMotion,
            observerMotionValid=observerMotionValid,
            sample=sample,
            d=d)
        pred = self.base.BuildPredictedVisual(rollout["s_next"])
        pred["prior_rollout"] = rollout
        return pred

    def ComputePredictionLoss(
        self,
        predictedVisual: PredictedVisualPack,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        precision: torch.Tensor,
        sampleMask: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        return self.base.ComputePredictionLoss(
            predictedVisual=predictedVisual,
            reconstructedVisualState=reconstructedVisualState,
            targetVisualState=targetVisualState,
            precision=precision,
            sampleMask=sampleMask,)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        r = int(a.size(0))
        if r <= 0 or a.numel() == 0 or b.numel() == 0 or abs(float(scale)) < 1e-12:
            return False

        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "obs_enc0":
            self.base.obs_enc[1].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "obs_enc1":
            self.base.obs_enc[4].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "act_proj":
            self.base.act_proj[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "s4_in_to_ssm":
            self.base.s4.in_to_ssm.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_ssm_to_deter":
            self.base.s4.ssm_to_deter.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_in_to_deter":
            self.base.s4.in_to_deter.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_gate":
            self.base.s4.gate.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_out_gate":
            self.base.s4.out_gate.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_nonlinear_left":
            self.base.s4.nonlinear_flow.left.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_nonlinear_right":
            self.base.s4.nonlinear_flow.right.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_nonlinear_context":
            self.base.s4.nonlinear_flow.context.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_nonlinear_out":
            self.base.s4.nonlinear_flow.out.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_nonlinear_selectivity":
            self.base.s4.nonlinear_flow.selectivity.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "prior":
            self.base.prior_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "post":
            self.base.post_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "state_proj":
            self.base.state_proj[1].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "rdone0":
            self.base.rdone_trunk[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "rdone1":
            self.base.rdone_trunk[3].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "rew":
            self.base.rew_head[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "done":
            self.base.done_head[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "obs_dec0":
            self.base.obs_dec[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "obs_dec1":
            self.base.obs_dec[2].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "mix_gate":
            self.base.mix_gate[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "key_to_gb":
            self.base.key_emb.to_gb.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "key_mlp1":
            self.base.key_emb.mlp1.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "key_mlp2":
            self.base.key_emb.mlp2.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "ssfilm_e_to_gb":
            self.base.state_state_film.e_to_gb.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_e_to_h":
            self.base.state_state_film.e_to_h.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_delta0":
            self.base.state_state_film.delta_mlp[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_delta1":
            self.base.state_state_film.delta_mlp[3].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_to_gate":
            self.base.state_state_film.to_gate.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "conn_enc_s":
            self.base.conn.enc_s[1].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_enc_a":
            self.base.conn.enc_a[1].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_film_gamma_a":
            self.base.conn.film_gamma_a.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_film_beta_a":
            self.base.conn.film_beta_a.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site.startswith("conn_blk") and (("_ff0" in site) or ("_ff1" in site)):
            s0 = site.replace("conn_blk", "")
            i_str, which = s0.split("_", 1)
            i = int(i_str)
            if which == "ff0":
                self.base.conn.blocks[i].ff[0].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
            elif which == "ff1":
                self.base.conn.blocks[i].ff[3].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
            else:
                raise ValueError(site)

        elif site == "conn_head_uv":
            self.base.conn.head_uv.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_head_full":
            self.base.conn.head_full.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_mix":
            self.base.conn.mix.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_to_qp":
            self.base.phys_refiner.to_qp.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_from_qp":
            self.base.phys_refiner.from_qp.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_H0":
            self.base.phys_refiner.HNet[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_H1":
            self.base.phys_refiner.HNet[2].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_H2":
            self.base.phys_refiner.HNet[4].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_force0":
            self.base.phys_refiner.force_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_force1":
            self.base.phys_refiner.force_net[2].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_g_force":
            self.base.phys_refiner.g_force.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_g_phys":
            self.base.phys_refiner.g_phys.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_g_fuse":
            self.base.phys_refiner.g_fuse.Grow(r, init=init, freezeOld=self.freezeOldPar)

        else:
            raise ValueError(f"Unknown site: {site}")

        return True
