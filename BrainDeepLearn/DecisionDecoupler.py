from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunctionTools import AGICoreModule
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    PackedEndEffectorTarget,
    RobotEmbodimentContractView,
)


@dataclass(frozen=True)
class PackedDecisionContext:
    plan_latent: torch.Tensor
    subgoal_feature: torch.Tensor
    context_feature: torch.Tensor
    constraint_tokens: torch.Tensor
    constraint_valid: torch.Tensor
    slot_legal: torch.Tensor
    risk: torch.Tensor
    confidence: torch.Tensor
    precision: torch.Tensor
    slot_relevance: Optional[torch.Tensor] = None
    slot_selection_mask: Optional[torch.Tensor] = None
    previous_target_values: Optional[torch.Tensor] = None
    previous_target_active: Optional[torch.Tensor] = None

    def IndexSelectRows(
        self,
        rowIndex: torch.Tensor,
    ) -> "PackedDecisionContext":
        if (
            not torch.is_tensor(rowIndex)
            or rowIndex.dim() != 1
            or rowIndex.dtype != torch.long
            or rowIndex.device != self.plan_latent.device
        ):
            raise ValueError("rowIndex must be a one-dimensional long tensor on the context device")
        batchSize = int(self.plan_latent.size(0))
        if rowIndex.numel() > 0 and bool(
            ((rowIndex < 0) | (rowIndex >= batchSize)).any().item()
        ):
            raise IndexError("rowIndex contains an out-of-range context row")

        def Select(value: torch.Tensor) -> torch.Tensor:
            if not torch.is_tensor(value) or value.dim() < 1:
                raise ValueError("decision context fields must have a batch dimension")
            if int(value.size(0)) != batchSize:
                raise ValueError("decision context fields must share one batch size")
            if value.device != rowIndex.device:
                raise ValueError("decision context fields must share the rowIndex device")
            return value.index_select(0, rowIndex)

        return PackedDecisionContext(
            plan_latent=Select(self.plan_latent),
            subgoal_feature=Select(self.subgoal_feature),
            context_feature=Select(self.context_feature),
            constraint_tokens=Select(self.constraint_tokens),
            constraint_valid=Select(self.constraint_valid),
            slot_legal=Select(self.slot_legal),
            risk=Select(self.risk),
            confidence=Select(self.confidence),
            precision=Select(self.precision),
            slot_relevance=(
                None
                if self.slot_relevance is None
                else Select(self.slot_relevance)),
            slot_selection_mask=(
                None
                if self.slot_selection_mask is None
                else Select(self.slot_selection_mask)),
            previous_target_values=(
                None
                if self.previous_target_values is None
                else Select(self.previous_target_values)),
            previous_target_active=(
                None
                if self.previous_target_active is None
                else Select(self.previous_target_active)))


@dataclass(frozen=True)
class PackedDecoupledDecision:
    target: PackedEndEffectorTarget
    world_action_feature: torch.Tensor
    z_task: torch.Tensor
    z_motion: torch.Tensor
    z_dynamics: torch.Tensor
    z_constraint: torch.Tensor
    z_uncertainty: torch.Tensor
    safety_scores: torch.Tensor
    safety_logits: torch.Tensor
    legality_logits: torch.Tensor
    slot_legal: torch.Tensor
    slot_available: torch.Tensor
    hierarchy_enabled: torch.Tensor
    selection_logits: torch.Tensor
    selection_probability: torch.Tensor
    slot_selected: torch.Tensor
    slot_executable: torch.Tensor
    explanation_tokens: torch.Tensor


@dataclass(frozen=True)
class PackedPerceptionRotationEfference:
    rotation_delta: torch.Tensor
    valid: torch.Tensor
    contract_id: str


class PackedHierarchicalDecisionDecoder(AGICoreModule):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        decisionDim: int,
        slotTokenDim: int = 128,
        feedbackTokenDim: int = 128,
        hierarchyDim: int = 128,
        hiddenDim: int = 256,
        planDim: Optional[int] = None,
        subgoalDim: Optional[int] = None,
        contextDim: Optional[int] = None,
        constraintTokenDim: Optional[int] = None,
        taskDim: int = 256,
        motionDim: int = 256,
        dynamicsDim: int = 128,
        constraintDim: int = 128,
        uncertaintyDim: int = 64,
        attentionHeads: int = 4,
        worldActionDim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("packed decoder requires an embodiment contract view")
        contractView.Validate()
        worldActionDim = decisionDim if worldActionDim is None else worldActionDim
        planDim = decisionDim if planDim is None else planDim
        subgoalDim = decisionDim if subgoalDim is None else subgoalDim
        contextDim = decisionDim if contextDim is None else contextDim
        constraintTokenDim = (
            decisionDim if constraintTokenDim is None else constraintTokenDim)
        dimensions = (
            decisionDim,
            worldActionDim,
            slotTokenDim,
            feedbackTokenDim,
            hierarchyDim,
            hiddenDim,
            planDim,
            subgoalDim,
            contextDim,
            constraintTokenDim,
            taskDim,
            motionDim,
            dynamicsDim,
            constraintDim,
            uncertaintyDim,
            attentionHeads,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("packed decoder dimensions must be positive integers")
        actual_attention_heads = max(
            headCount
            for headCount in range(1, int(attentionHeads) + 1)
            if int(hiddenDim) % headCount == 0)

        self.contract_view = contractView
        self.decision_dim = int(decisionDim)
        self.world_action_dim = int(worldActionDim)
        self.slot_token_dim = int(slotTokenDim)
        self.feedback_token_dim = int(feedbackTokenDim)
        self.hierarchy_dim = int(hierarchyDim)
        self.hidden_dim = int(hiddenDim)
        self.plan_dim = int(planDim)
        self.subgoal_dim = int(subgoalDim)
        self.context_dim = int(contextDim)
        self.constraint_token_dim = int(constraintTokenDim)
        self.task_dim = int(taskDim)
        self.motion_dim = int(motionDim)
        self.dynamics_dim = int(dynamicsDim)
        self.constraint_dim = int(constraintDim)
        self.uncertainty_dim = int(uncertaintyDim)
        self.factor_dim = (
            self.task_dim
            + self.motion_dim
            + self.dynamics_dim
            + self.constraint_dim
            + self.uncertainty_dim)
        self.slot_count = int(contractView.end_effector_count)
        self.joint_count = int(contractView.joint_count)
        self.target_packed_dim = int(
            contractView.end_effector_target_layout.PackedDim)
        self.feedback_packed_dim = int(
            contractView.joint_feedback_layout.PackedDim)
        self.topological_layers = tuple(
            tuple(int(slotIndex) for slotIndex in layer)
            for layer in contractView.topological_layers)
        self.parent_index = tuple(int(value) for value in contractView.parent_index)
        self.child_indices = tuple(
            tuple(
                childIndex
                for childIndex, parentIndex in enumerate(self.parent_index)
                if parentIndex == slotIndex)
            for slotIndex in range(self.slot_count))
        self.target_offsets = tuple(
            int(value)
            for value in contractView.end_effector_target_layout.offsets)
        self.feedback_offsets = tuple(
            int(value) for value in contractView.joint_feedback_layout.offsets)
        self.endpoint_joint_chain_offsets = tuple(
            int(value)
            for value in contractView.end_effector_joint_chain_offsets)
        self.endpoint_joint_chain_indices = tuple(
            int(value)
            for value in contractView.end_effector_joint_chain_indices)
        self.slot_relevance_gain = nn.Parameter(torch.tensor(0.0))

        static_slot_tokens = torch.tensor(
            contractView.static_end_effector_tokens,
            dtype=torch.float32)
        static_joint_tokens = torch.tensor(
            contractView.static_joint_tokens,
            dtype=torch.float32)
        self.register_buffer(
            "static_slot_tokens",
            static_slot_tokens,
            persistent=True)
        self.register_buffer(
            "static_joint_tokens",
            static_joint_tokens,
            persistent=True)
        self.register_buffer(
            "root_mask",
            torch.tensor(contractView.root_mask, dtype=torch.bool),
            persistent=True)
        self.register_buffer(
            "child_mask",
            torch.tensor(contractView.child_mask, dtype=torch.bool),
            persistent=True)
        self.register_buffer(
            "independent_mask",
            torch.tensor(contractView.independent_mask, dtype=torch.bool),
            persistent=True)
        self.register_buffer(
            "target_lower",
            torch.tensor(
                contractView.end_effector_target_lower,
                dtype=torch.float32),
            persistent=True)
        self.register_buffer(
            "target_upper",
            torch.tensor(
                contractView.end_effector_target_upper,
                dtype=torch.float32),
            persistent=True)

        descriptor_dim = int(static_slot_tokens.size(-1))
        static_normalizer = (
            nn.Identity()
            if descriptor_dim == 1
            else nn.LayerNorm(descriptor_dim))
        self.static_slot_encoder = nn.Sequential(
            static_normalizer,
            nn.Linear(descriptor_dim, self.slot_token_dim),
            nn.SiLU(),
            nn.LayerNorm(self.slot_token_dim),
        )
        joint_descriptor_dim = int(static_joint_tokens.size(-1))
        joint_static_normalizer = (
            nn.Identity()
            if joint_descriptor_dim == 1
            else nn.LayerNorm(joint_descriptor_dim))
        self.static_joint_encoder = nn.Sequential(
            joint_static_normalizer,
            nn.Linear(joint_descriptor_dim, self.feedback_token_dim),
            nn.SiLU(),
            nn.LayerNorm(self.feedback_token_dim),
        )
        self.joint_feedback_adapters = nn.ModuleList()
        self.joint_token_fuser = nn.Sequential(
            nn.LayerNorm(2 * self.feedback_token_dim),
            nn.Linear(2 * self.feedback_token_dim, self.feedback_token_dim),
            nn.SiLU(),
            nn.LayerNorm(self.feedback_token_dim),
        )
        self.slot_decoders = nn.ModuleList()
        self.slot_dynamics_heads = nn.ModuleList()
        self.slot_residual_heads = nn.ModuleList()
        self.slot_safety_heads = nn.ModuleList()
        self.slot_legality_heads = nn.ModuleList()
        self.slot_selection_heads = nn.ModuleList()
        self.parent_output_encoders = nn.ModuleList()
        self.world_action_adapters = nn.ModuleList()
        self.dynamic_state_encoder = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.feedback_token_dim),
            nn.SiLU(),
            nn.LayerNorm(self.feedback_token_dim),
        )
        self.child_feedback_encoder = nn.Sequential(
            nn.LayerNorm(self.feedback_token_dim),
            nn.Linear(self.feedback_token_dim, self.hierarchy_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hierarchy_dim),
        )
        global_feedback_dim = (
            self.slot_token_dim
            + 2 * self.feedback_token_dim
            + self.hierarchy_dim)
        self.global_feedback_encoder = nn.Sequential(
            nn.LayerNorm(global_feedback_dim),
            nn.Linear(global_feedback_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        factor_input_dim = (
            self.decision_dim
            + self.plan_dim
            + self.subgoal_dim
            + self.context_dim
            + self.hidden_dim)
        self.factor_projector = nn.Sequential(
            nn.LayerNorm(factor_input_dim),
            nn.Linear(factor_input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.factor_dim),
        )
        self.factor_query_encoder = nn.ModuleList((
            nn.Linear(self.task_dim, self.hidden_dim),
            nn.Linear(self.motion_dim, self.hidden_dim),
        ))
        self.constraint_token_encoder = nn.Sequential(
            nn.LayerNorm(self.constraint_token_dim),
            nn.Linear(self.constraint_token_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.constraint_attention = nn.MultiheadAttention(
            self.hidden_dim,
            actual_attention_heads,
            batch_first=True)
        self.constraint_attention_norm = nn.LayerNorm(self.hidden_dim)
        decoder_input_dim = (
            self.factor_dim
            + self.plan_dim
            + self.subgoal_dim
            + self.context_dim
            + 2 * self.hidden_dim
            + self.slot_token_dim
            + 2 * self.feedback_token_dim
            + 2 * self.hierarchy_dim)
        selection_input_dim = (
            factor_input_dim
            + self.slot_token_dim
            + 2 * self.feedback_token_dim
            + self.hierarchy_dim)
        for jointIndex in range(self.joint_count):
            feedback_width = contractView.joint_feedback_layout.Width(jointIndex)
            if feedback_width < 1:
                raise ValueError("joint feedback widths must be positive")
            feedback_normalizer = (
                nn.Identity()
                if feedback_width == 1
                else nn.LayerNorm(feedback_width))
            self.joint_feedback_adapters.append(nn.Sequential(
                feedback_normalizer,
                nn.Linear(feedback_width, self.feedback_token_dim),
                nn.SiLU(),
                nn.LayerNorm(self.feedback_token_dim),
            ))
        for slotIndex in range(self.slot_count):
            target_width = contractView.end_effector_target_layout.Width(
                slotIndex)
            if target_width < 1:
                raise ValueError("end-effector target widths must be positive")
            self.slot_decoders.append(nn.Sequential(
                nn.LayerNorm(decoder_input_dim),
                nn.Linear(decoder_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, target_width),
            ))
            self.slot_dynamics_heads.append(nn.Sequential(
                nn.LayerNorm(decoder_input_dim),
                nn.Linear(decoder_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, target_width),
            ))
            self.slot_residual_heads.append(nn.Sequential(
                nn.LayerNorm(decoder_input_dim),
                nn.Linear(decoder_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, target_width),
            ))
            self.slot_safety_heads.append(nn.Sequential(
                nn.LayerNorm(selection_input_dim),
                nn.Linear(selection_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            ))
            self.slot_legality_heads.append(nn.Sequential(
                nn.LayerNorm(selection_input_dim),
                nn.Linear(selection_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            ))
            self.slot_selection_heads.append(nn.Sequential(
                nn.LayerNorm(selection_input_dim),
                nn.Linear(selection_input_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            ))
            self.parent_output_encoders.append(nn.Sequential(
                nn.Linear(target_width, self.hierarchy_dim),
                nn.SiLU(),
                nn.LayerNorm(self.hierarchy_dim),
            ))
            self.world_action_adapters.append(nn.Linear(
                target_width,
                self.world_action_dim,
                bias=False))

    def BuildNeutralContext(
        self,
        decisionBackbone: torch.Tensor,
    ) -> PackedDecisionContext:
        batch_size = int(decisionBackbone.size(0))
        return PackedDecisionContext(
            plan_latent=decisionBackbone.new_zeros(batch_size, self.plan_dim),
            subgoal_feature=decisionBackbone.new_zeros(
                batch_size, self.subgoal_dim),
            context_feature=decisionBackbone.new_zeros(
                batch_size, self.context_dim),
            constraint_tokens=decisionBackbone.new_zeros(
                batch_size, 1, self.constraint_token_dim),
            constraint_valid=torch.ones(
                batch_size,
                1,
                dtype=torch.bool,
                device=decisionBackbone.device),
            slot_legal=torch.ones(
                batch_size,
                self.slot_count,
                dtype=torch.bool,
                device=decisionBackbone.device),
            risk=decisionBackbone.new_zeros(batch_size),
            confidence=decisionBackbone.new_ones(batch_size),
            precision=decisionBackbone.new_ones(batch_size),
        )

    def ValidateContext(
        self,
        decisionContext: PackedDecisionContext,
        decisionBackbone: torch.Tensor,
    ) -> None:
        if type(decisionContext) is not PackedDecisionContext:
            raise TypeError("packed decoder context has an invalid type")
        batch_size = int(decisionBackbone.size(0))
        device = decisionBackbone.device
        expected = (
            (decisionContext.plan_latent, (batch_size, self.plan_dim)),
            (decisionContext.subgoal_feature, (batch_size, self.subgoal_dim)),
            (decisionContext.context_feature, (batch_size, self.context_dim)),
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != shape
            or not value.is_floating_point()
            or value.device != device
            or value.dtype != decisionBackbone.dtype
            or not bool(torch.isfinite(value).all().item())
            for value, shape in expected
        ):
            raise ValueError("packed decoder context features are invalid")
        tokens = decisionContext.constraint_tokens
        token_valid = decisionContext.constraint_valid
        if (
            not torch.is_tensor(tokens)
            or tokens.dim() != 3
            or tuple(tokens.shape[:1]) != (batch_size,)
            or int(tokens.size(-1)) != self.constraint_token_dim
            or int(tokens.size(1)) < 1
            or not tokens.is_floating_point()
            or tokens.device != device
            or tokens.dtype != decisionBackbone.dtype
            or not bool(torch.isfinite(tokens).all().item())
        ):
            raise ValueError("packed decoder constraint tokens are invalid")
        if (
            not torch.is_tensor(token_valid)
            or tuple(token_valid.shape) != tuple(tokens.shape[:2])
            or token_valid.dtype != torch.bool
            or token_valid.device != device
            or not bool(token_valid.any(dim=-1).all().item())
        ):
            raise ValueError("packed decoder constraint validity is invalid")
        if (
            not torch.is_tensor(decisionContext.slot_legal)
            or tuple(decisionContext.slot_legal.shape)
            != (batch_size, self.slot_count)
            or decisionContext.slot_legal.dtype != torch.bool
            or decisionContext.slot_legal.device != device
        ):
            raise ValueError("packed decoder legality mask is invalid")
        selection_mask = decisionContext.slot_selection_mask
        if selection_mask is not None and (
            not torch.is_tensor(selection_mask)
            or tuple(selection_mask.shape) != (batch_size, self.slot_count)
            or selection_mask.dtype != torch.bool
            or selection_mask.device != device
        ):
            raise ValueError("packed decoder selection mask is invalid")
        relevance = decisionContext.slot_relevance
        if relevance is not None and (
            not torch.is_tensor(relevance)
            or tuple(relevance.shape) != (batch_size, self.slot_count)
            or not relevance.is_floating_point()
            or relevance.device != device
            or relevance.dtype != decisionBackbone.dtype
            or not bool(torch.isfinite(relevance).all().item())
            or bool((relevance < 0.0).any().item())
        ):
            raise ValueError("packed decoder slot relevance is invalid")
        previous_values = decisionContext.previous_target_values
        previous_active = decisionContext.previous_target_active
        if (previous_values is None) != (previous_active is None):
            raise ValueError("previous target values and activity must be provided together")
        if previous_values is not None and (
            not torch.is_tensor(previous_values)
            or tuple(previous_values.shape)
            != (batch_size, self.target_packed_dim)
            or not previous_values.is_floating_point()
            or previous_values.device != device
            or previous_values.dtype != decisionBackbone.dtype
            or not bool(torch.isfinite(previous_values).all().item())
            or not torch.is_tensor(previous_active)
            or tuple(previous_active.shape) != (batch_size, self.slot_count)
            or previous_active.dtype != torch.bool
            or previous_active.device != device
        ):
            raise ValueError("previous end-effector target is invalid")
        if previous_values is not None and previous_active is not None:
            tolerance = 16.0 * torch.finfo(previous_values.dtype).eps
            for slotIndex, parentIndex in enumerate(self.parent_index):
                if parentIndex >= 0 and bool((
                    previous_active[:, slotIndex]
                    & ~previous_active[:, parentIndex]
                ).any().item()):
                    raise ValueError("previous child targets require active ancestors")
            for slotIndex in range(self.slot_count):
                targetSlice = slice(
                    self.target_offsets[slotIndex],
                    self.target_offsets[slotIndex + 1])
                lower = self.target_lower[targetSlice].to(
                    device=device,
                    dtype=previous_values.dtype)
                upper = self.target_upper[targetSlice].to(
                    device=device,
                    dtype=previous_values.dtype)
                if bool((
                    previous_active[:, slotIndex].unsqueeze(-1)
                    & (
                        (previous_values[:, targetSlice] < lower - tolerance)
                        | (previous_values[:, targetSlice] > upper + tolerance)
                    )
                ).any().item()):
                    raise ValueError("previous active targets exceed morphology limits")
        for value in (
            decisionContext.risk,
            decisionContext.confidence,
            decisionContext.precision,
        ):
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != (batch_size,)
                or not value.is_floating_point()
                or value.device != device
                or value.dtype != decisionBackbone.dtype
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError("packed decoder confidence signals are invalid")
        if bool((decisionContext.risk < 0.0).any().item()) or bool(
            (decisionContext.risk > 1.0).any().item()
        ):
            raise ValueError("packed decoder risk must be normalized")
        if bool((decisionContext.confidence < 0.0).any().item()) or bool(
            (decisionContext.confidence > 1.0).any().item()
        ):
            raise ValueError("packed decoder confidence must be normalized")
        if bool((decisionContext.precision < 0.0).any().item()):
            raise ValueError("packed decoder precision cannot be negative")

    def ValidateDecisionBackbone(
        self,
        decisionBackbone: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(decisionBackbone):
            raise TypeError("decision backbone must be a tensor")
        if tuple(decisionBackbone.shape[1:]) != (self.decision_dim,):
            raise ValueError("decision backbone shape does not match the decoder")
        if not decisionBackbone.is_floating_point() or not bool(
            torch.isfinite(decisionBackbone).all().item()
        ):
            raise ValueError("decision backbone must be finite floating point")

    def ResolveExecutionState(
        self,
        decisionBackbone: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(decisionBackbone.size(0))
        device = decisionBackbone.device
        dtype = decisionBackbone.dtype
        if type(feedbackPacket) is not BrainFeedbackPacket:
            raise TypeError("feedback must be a BrainFeedbackPacket")
        feedbackPacket.Validate(self.contract_view)
        if int(feedbackPacket.joint_features.size(0)) != batch_size:
            raise ValueError("feedback batch does not match decision backbone")
        if feedbackPacket.joint_features.device != device:
            raise ValueError("feedback must share the decision device")
        feedback_values = feedbackPacket.joint_features.to(dtype=dtype)
        available = feedbackPacket.endpoint_valid
        enabled = feedbackPacket.child_enabled

        root_mask = self.root_mask.to(device=device).unsqueeze(0)
        enabled = enabled | root_mask
        return feedback_values, available, enabled

    def EncodeFeedbackState(
        self,
        decisionBackbone: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
        feedbackValues: torch.Tensor,
        staticTokens: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = int(decisionBackbone.size(0))
        dtype = decisionBackbone.dtype
        device = decisionBackbone.device
        available = feedbackPacket.endpoint_valid
        dynamic_state = torch.stack([
            feedbackPacket.progress,
            feedbackPacket.reached.to(dtype=dtype),
            feedbackPacket.child_enabled.to(dtype=dtype),
            feedbackPacket.target_active.to(dtype=dtype),
            feedbackPacket.endpoint_valid.to(dtype=dtype),
        ], dim=-1).to(dtype=dtype)

        dynamic_tokens = self.dynamic_state_encoder(dynamic_state)
        static_joint_tokens = self.static_joint_encoder(
            self.static_joint_tokens.to(device=device, dtype=dtype))
        joint_tokens = []
        for jointIndex in range(self.joint_count):
            feedback_slice = slice(
                self.feedback_offsets[jointIndex],
                self.feedback_offsets[jointIndex + 1])
            feedback_token = self.joint_feedback_adapters[jointIndex](
                feedbackValues[:, feedback_slice])
            joint_token = self.joint_token_fuser(torch.cat([
                feedback_token,
                static_joint_tokens[jointIndex].unsqueeze(0).expand(
                    batch_size, -1),
            ], dim=-1))
            joint_token = joint_token * feedbackPacket.joint_valid[
                :, jointIndex].to(dtype=dtype).unsqueeze(-1)
            joint_tokens.append(joint_token)
        joint_token_tensor = torch.stack(joint_tokens, dim=1)

        feedback_tokens = []
        for slotIndex in range(self.slot_count):
            chain_start = self.endpoint_joint_chain_offsets[slotIndex]
            chain_end = self.endpoint_joint_chain_offsets[slotIndex + 1]
            chain_index = torch.tensor(
                self.endpoint_joint_chain_indices[chain_start:chain_end],
                dtype=torch.long,
                device=device)
            chain_tokens = joint_token_tensor.index_select(1, chain_index)
            chain_valid = feedbackPacket.joint_valid.index_select(
                1, chain_index)
            chain_weight = chain_valid.to(dtype=dtype).unsqueeze(-1)
            feedback_token = (
                chain_tokens * chain_weight).sum(dim=1) / chain_weight.sum(
                    dim=1).clamp_min(1.0)
            feedback_token = feedback_token * available[:, slotIndex].to(
                dtype=dtype).unsqueeze(-1)
            feedback_tokens.append(feedback_token)
        feedback_token_tensor = torch.stack(feedback_tokens, dim=1)
        dynamic_tokens = dynamic_tokens * available.to(dtype=dtype).unsqueeze(-1)

        child_contexts = []
        for slotIndex in range(self.slot_count):
            children = self.child_indices[slotIndex]
            if not children:
                child_contexts.append(decisionBackbone.new_zeros(
                    batch_size,
                    self.hierarchy_dim))
                continue
            child_index = torch.tensor(children, dtype=torch.long, device=device)
            child_tokens = feedback_token_tensor.index_select(1, child_index)
            child_valid = available.index_select(1, child_index)
            child_weight = child_valid.to(dtype=dtype).unsqueeze(-1)
            child_mean = (child_tokens * child_weight).sum(dim=1) / child_weight.sum(
                dim=1).clamp_min(1.0)
            child_contexts.append(self.child_feedback_encoder(child_mean))
        child_context = torch.stack(child_contexts, dim=1)

        static_batch = staticTokens.unsqueeze(0).expand(batch_size, -1, -1)
        global_slot_tokens = torch.cat([
            static_batch,
            feedback_token_tensor,
            dynamic_tokens,
            child_context,
        ], dim=-1)
        global_weight = available.to(dtype=dtype).unsqueeze(-1)
        global_feedback = (global_slot_tokens * global_weight).sum(
            dim=1) / global_weight.sum(dim=1).clamp_min(1.0)
        global_feedback = self.global_feedback_encoder(global_feedback)
        return (
            tuple(feedback_tokens),
            dynamic_tokens,
            child_context,
            global_feedback,
            available.to(dtype=dtype),
        )

    def AttendConstraints(
        self,
        zTask: torch.Tensor,
        zMotion: torch.Tensor,
        decisionContext: PackedDecisionContext,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = torch.stack([
            self.factor_query_encoder[0](zTask),
            self.factor_query_encoder[1](zMotion),
        ], dim=1)
        tokens = self.constraint_token_encoder(
            decisionContext.constraint_tokens)
        attended, _ = self.constraint_attention(
            query,
            tokens,
            tokens,
            key_padding_mask=~decisionContext.constraint_valid,
            need_weights=False)
        attended = self.constraint_attention_norm(attended + query)
        token_weight = decisionContext.constraint_valid.to(
            dtype=decisionContext.constraint_tokens.dtype).unsqueeze(-1)
        explanation = (
            decisionContext.constraint_tokens * token_weight).sum(
                dim=1) / token_weight.sum(dim=1).clamp_min(1.0)
        return attended.flatten(start_dim=1), explanation

    def SelectSlots(
        self,
        selectionShared: torch.Tensor,
        staticTokens: torch.Tensor,
        feedbackTokens: Tuple[torch.Tensor, ...],
        dynamicTokens: torch.Tensor,
        childContext: torch.Tensor,
        slotEligible: torch.Tensor,
        slotRelevance: Optional[torch.Tensor],
        slotSelectionMask: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size = int(selectionShared.size(0))
        selection_values = []
        safety_values = []
        legality_values = []
        for slotIndex in range(self.slot_count):
            selector_input = torch.cat([
                selectionShared,
                staticTokens[slotIndex].unsqueeze(0).expand(batch_size, -1),
                feedbackTokens[slotIndex],
                dynamicTokens[:, slotIndex],
                childContext[:, slotIndex],
            ], dim=-1)
            selection_values.append(
                self.slot_selection_heads[slotIndex](selector_input).squeeze(-1))
            safety_values.append(
                self.slot_safety_heads[slotIndex](selector_input).squeeze(-1))
            legality_values.append(
                self.slot_legality_heads[slotIndex](selector_input).squeeze(-1))
        selection_logits = torch.stack(selection_values, dim=-1)
        safety_logits = torch.stack(safety_values, dim=-1)
        legality_logits = torch.stack(legality_values, dim=-1)
        if slotRelevance is not None:
            log_relevance = slotRelevance.clamp_min(1e-8).log()
            eligible_weight = slotEligible.to(
                dtype=selection_logits.dtype)
            centered_relevance = log_relevance - (
                log_relevance * eligible_weight
            ).sum(dim=-1, keepdim=True) / eligible_weight.sum(
                dim=-1, keepdim=True).clamp_min(1.0)
            selection_logits = (
                selection_logits
                + torch.sigmoid(self.slot_relevance_gain)
                * centered_relevance)
        selection_probability = torch.sigmoid(selection_logits)
        if slotSelectionMask is not None:
            slot_selected = slotSelectionMask
        else:
            slot_selected = selection_logits > 0.0
            root_eligible = (
                slotEligible
                & self.root_mask.to(
                    device=selection_logits.device).unsqueeze(0))
            missing_root = (
                root_eligible.any(dim=-1)
                & ~(slot_selected & root_eligible).any(dim=-1))
            fallback_index = selection_logits.masked_fill(
                ~root_eligible,
                -torch.inf).argmax(dim=-1)
            fallback = F.one_hot(
                fallback_index,
                num_classes=self.slot_count).to(dtype=torch.bool)
            slot_selected = slot_selected | (
                fallback & missing_root.unsqueeze(-1))
        hard_gate = slot_selected.to(dtype=selection_probability.dtype)
        straight_through_gate = (
            hard_gate
            + selection_probability
            - selection_probability.detach())
        return (
            selection_logits,
            selection_probability,
            slot_selected,
            straight_through_gate,
            safety_logits,
            legality_logits,
        )

    def EncodeWorldAction(
        self,
        target: PackedEndEffectorTarget,
    ) -> torch.Tensor:
        if type(target) is not PackedEndEffectorTarget:
            raise TypeError("world action encoding requires an end-effector target")
        target.Validate(self.contract_view)
        if (
            target.values.device != self.static_slot_tokens.device
            or target.values.dtype != self.static_slot_tokens.dtype
        ):
            raise ValueError("end-effector targets must share the decoder device and dtype")

        encoded_slots = []
        for slotIndex, adapter in enumerate(self.world_action_adapters):
            slotSlice = slice(
                self.target_offsets[slotIndex],
                self.target_offsets[slotIndex + 1])
            slotOutput = torch.where(
                target.active[:, slotIndex].unsqueeze(-1),
                target.values[:, slotSlice],
                torch.zeros_like(target.values[:, slotSlice]))
            slot_feature = adapter(slotOutput)
            encoded_slots.append(slot_feature * target.active[:, slotIndex].to(
                dtype=slot_feature.dtype).unsqueeze(-1))

        active_count = target.active.to(
            dtype=encoded_slots[0].dtype).sum(dim=-1, keepdim=True)
        return torch.stack(encoded_slots, dim=1).sum(dim=1) / active_count.clamp_min(1.0)

    def DecodePerceptionRotationEfference(
        self,
        target: PackedEndEffectorTarget,
        feedbackPacket: BrainFeedbackPacket,
    ) -> PackedPerceptionRotationEfference:
        if type(target) is not PackedEndEffectorTarget:
            raise TypeError("perception efference requires an end-effector target")
        target.Validate(self.contract_view)
        if type(feedbackPacket) is not BrainFeedbackPacket:
            raise TypeError("perception efference requires encoded feedback")
        feedbackPacket.Validate(self.contract_view)
        batch_size = int(target.values.size(0))
        if (
            int(feedbackPacket.joint_features.size(0)) != batch_size
            or feedbackPacket.joint_features.device != target.values.device
            or feedbackPacket.joint_features.dtype != target.values.dtype
        ):
            raise ValueError("perception efference target and feedback must match")
        rotations = []
        validity = []
        for view_index, slot_index in enumerate(
            self.contract_view.perception_view.indices
        ):
            target_slice = self.contract_view.end_effector_target_layout.Slice(
                slot_index)
            translation_basis = (
                self.contract_view.end_effector_translation_basis.Matrix(
                    slot_index,
                    device=target.values.device,
                    dtype=target.values.dtype))
            rotation_basis = self.contract_view.end_effector_rotation_basis.Matrix(
                slot_index,
                device=target.values.device,
                dtype=target.values.dtype)
            translation_width = int(translation_basis.size(1))
            rotation_width = int(rotation_basis.size(1))
            compact = target.values[:, target_slice]
            compact_rotation = compact[
                :, translation_width:translation_width + rotation_width]
            compact_rotation = torch.where(
                target.active[:, slot_index].unsqueeze(-1),
                compact_rotation,
                torch.zeros_like(compact_rotation))
            tangent = compact_rotation.matmul(rotation_basis.transpose(0, 1))
            angle = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True)
            vector_scale = 0.5 * torch.sinc(angle / (2.0 * math.pi))
            target_quaternion = torch.cat([
                tangent * vector_scale,
                torch.cos(0.5 * angle),
            ], dim=-1)
            target_quaternion = F.normalize(target_quaternion, dim=-1)
            current_quaternion = feedbackPacket.perception_rotation[
                :, view_index]
            cx, cy, cz, cw = current_quaternion.unbind(dim=-1)
            tx, ty, tz, tw = target_quaternion.unbind(dim=-1)
            quaternion = torch.stack((
                cw * tx - cx * tw - cy * tz + cz * ty,
                cw * ty + cx * tz - cy * tw - cz * tx,
                cw * tz - cx * ty + cy * tx - cz * tw,
                cw * tw + cx * tx + cy * ty + cz * tz,
            ), dim=-1)
            quaternion = F.normalize(quaternion, dim=-1)
            quaternion = torch.where(
                quaternion[:, 3:4] < 0.0,
                -quaternion,
                quaternion)
            active = (
                target.active[:, slot_index]
                & feedbackPacket.endpoint_valid[:, slot_index])
            identity = torch.zeros_like(quaternion)
            identity[:, -1] = 1.0
            rotations.append(torch.where(
                active.unsqueeze(-1), quaternion, identity))
            validity.append(active)

        if rotations:
            rotation_delta = torch.stack(rotations, dim=1)
            valid = torch.stack(validity, dim=1)
        else:
            rotation_delta = target.values.new_zeros(batch_size, 0, 4)
            valid = torch.zeros(
                batch_size,
                0,
                dtype=torch.bool,
                device=target.values.device)
        return PackedPerceptionRotationEfference(
            rotation_delta=rotation_delta,
            valid=valid,
            contract_id=target.contract_id)

    def Decode(
        self,
        decisionBackbone: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
        decisionContext: Optional[PackedDecisionContext] = None,
    ) -> PackedDecoupledDecision:
        self.ValidateDecisionBackbone(decisionBackbone)
        if decisionContext is None:
            decisionContext = self.BuildNeutralContext(decisionBackbone)
        self.ValidateContext(decisionContext, decisionBackbone)
        (
            feedback_values,
            slot_available,
            hierarchy_enabled,
        ) = self.ResolveExecutionState(
            decisionBackbone,
            feedbackPacket)
        batch_size = int(decisionBackbone.size(0))
        static_tokens = self.static_slot_encoder(
            self.static_slot_tokens.to(
                device=decisionBackbone.device,
                dtype=decisionBackbone.dtype))
        (
            feedback_tokens,
            dynamic_tokens,
            child_context,
            global_feedback,
            endpoint_validity,
        ) = self.EncodeFeedbackState(
            decisionBackbone,
            feedbackPacket,
            feedback_values,
            static_tokens)
        factor_input = torch.cat([
            decisionBackbone,
            decisionContext.plan_latent,
            decisionContext.subgoal_feature,
            decisionContext.context_feature,
            global_feedback,
        ], dim=-1)
        slot_eligible = (
            slot_available
            & hierarchy_enabled
            & decisionContext.slot_legal)
        (
            selection_logits,
            selection_probability,
            slot_selected,
            selection_gate,
            safety_logits,
            legality_logits,
        ) = self.SelectSlots(
            factor_input,
            static_tokens,
            feedback_tokens,
            dynamic_tokens,
            child_context,
            slot_eligible,
            decisionContext.slot_relevance,
            decisionContext.slot_selection_mask)
        slot_active = slot_eligible & slot_selected
        ancestor_required = torch.zeros_like(slot_active)
        for layer in reversed(self.topological_layers):
            for slotIndex in layer:
                parentIndex = self.parent_index[slotIndex]
                if parentIndex >= 0:
                    slot_active[:, slotIndex] = (
                        slot_active[:, slotIndex]
                        & slot_eligible[:, parentIndex])
                    ancestor_required[:, parentIndex] = (
                        ancestor_required[:, parentIndex]
                        | slot_active[:, slotIndex])
                    slot_active[:, parentIndex] = (
                        slot_active[:, parentIndex]
                        | slot_active[:, slotIndex])
        held_slots = torch.zeros_like(slot_active)
        if (
            decisionContext.previous_target_values is not None
            and decisionContext.previous_target_active is not None
        ):
            held_slots = (
                feedbackPacket.reached
                & decisionContext.previous_target_active
                & ancestor_required
                & slot_eligible)
        slot_selected = slot_selected | slot_active
        selection_gate = (
            slot_active.to(dtype=selection_probability.dtype)
            + selection_probability
            - selection_probability.detach())
        factor_values = self.factor_projector(factor_input)
        (
            z_task,
            z_motion,
            z_dynamics,
            z_constraint,
            z_uncertainty,
        ) = torch.split(factor_values, (
            self.task_dim,
            self.motion_dim,
            self.dynamics_dim,
            self.constraint_dim,
            self.uncertainty_dim,
        ), dim=-1)
        constraint_context, explanation_tokens = self.AttendConstraints(
            z_task,
            z_motion,
            decisionContext)
        hierarchy_context = tuple(
            decisionBackbone.new_zeros(batch_size, self.hierarchy_dim)
            for _ in range(self.slot_count))
        slot_outputs = [
            decisionBackbone.new_zeros(
                batch_size,
                self.target_offsets[index + 1] - self.target_offsets[index])
            for index in range(self.slot_count)
        ]
        slot_contexts = list(hierarchy_context)

        shared_context = torch.cat([
            factor_values,
            decisionContext.plan_latent,
            decisionContext.subgoal_feature,
            decisionContext.context_feature,
            constraint_context,
        ], dim=-1)
        for layer in self.topological_layers:
            for slotIndex in layer:
                parent_index = self.parent_index[slotIndex]
                target_slice = slice(
                    self.target_offsets[slotIndex],
                    self.target_offsets[slotIndex + 1])
                held_rows = torch.nonzero(
                    held_slots[:, slotIndex],
                    as_tuple=False).flatten()
                if held_rows.numel() > 0:
                    if decisionContext.previous_target_values is None:
                        raise RuntimeError("held targets require previous target values")
                    held_output = decisionContext.previous_target_values[
                        :, target_slice].index_select(0, held_rows)
                    slot_outputs[slotIndex] = slot_outputs[
                        slotIndex].index_copy(
                            0,
                            held_rows,
                            held_output)
                    slot_contexts[slotIndex] = slot_contexts[
                        slotIndex].index_copy(
                            0,
                            held_rows,
                            self.parent_output_encoders[
                                slotIndex](held_output))
                active_rows = torch.nonzero(
                    slot_active[:, slotIndex] & ~held_slots[:, slotIndex],
                    as_tuple=False).flatten()
                if active_rows.numel() == 0:
                    continue
                if parent_index < 0:
                    parent_context = decisionBackbone.new_zeros(
                        active_rows.numel(),
                        self.hierarchy_dim)
                else:
                    parent_context = slot_contexts[parent_index].index_select(
                        0,
                        active_rows)
                decoder_input = torch.cat([
                    shared_context.index_select(0, active_rows),
                    static_tokens[slotIndex].unsqueeze(0).expand(
                        active_rows.numel(), -1),
                    feedback_tokens[slotIndex].index_select(0, active_rows),
                    dynamic_tokens[:, slotIndex].index_select(0, active_rows),
                    child_context[:, slotIndex].index_select(0, active_rows),
                    parent_context,
                ], dim=-1)
                decoded_latent = (
                    self.slot_decoders[slotIndex](decoder_input)
                    + 0.1 * self.slot_dynamics_heads[slotIndex](decoder_input)
                    + 0.1 * self.slot_residual_heads[slotIndex](decoder_input))
                normalized = torch.tanh(decoded_latent) * selection_gate[
                    :, slotIndex].index_select(0, active_rows).unsqueeze(-1)
                lower = self.target_lower[target_slice].to(
                    device=decoded_latent.device,
                    dtype=decoded_latent.dtype)
                upper = self.target_upper[target_slice].to(
                    device=decoded_latent.device,
                    dtype=decoded_latent.dtype)
                decoded = (
                    lower.unsqueeze(0)
                    + 0.5
                    * (normalized + 1.0)
                    * (upper - lower).unsqueeze(0))
                slot_outputs[slotIndex] = slot_outputs[slotIndex].index_copy(
                    0,
                    active_rows,
                    decoded)
                slot_contexts[slotIndex] = slot_contexts[slotIndex].index_copy(
                    0,
                    active_rows,
                    self.parent_output_encoders[slotIndex](decoded))

        target = PackedEndEffectorTarget(
            values=torch.cat(slot_outputs, dim=-1),
            active=slot_active,
            contract_id=self.contract_view.contract_id,
            model_signature=self.contract_view.model_signature,
            target_version=feedbackPacket.target_version + 1,
            timestamp=feedbackPacket.timestamp)
        target.Validate(self.contract_view)
        world_action_feature = self.EncodeWorldAction(target)
        safety_scores = torch.stack([
            torch.sigmoid(safety_logits),
            endpoint_validity,
            (1.0 - decisionContext.risk).unsqueeze(-1).expand(
                -1, self.slot_count),
            decisionContext.confidence.unsqueeze(-1).expand(
                -1, self.slot_count),
            (
                decisionContext.precision
                / (1.0 + decisionContext.precision)
            ).unsqueeze(-1).expand(-1, self.slot_count),
        ], dim=-1)
        return PackedDecoupledDecision(
            target=target,
            world_action_feature=world_action_feature,
            z_task=z_task,
            z_motion=z_motion,
            z_dynamics=z_dynamics,
            z_constraint=z_constraint,
            z_uncertainty=z_uncertainty,
            safety_scores=safety_scores,
            safety_logits=safety_logits,
            legality_logits=legality_logits,
            slot_legal=decisionContext.slot_legal,
            slot_available=slot_available,
            hierarchy_enabled=hierarchy_enabled,
            selection_logits=selection_logits,
            selection_probability=selection_probability,
            slot_selected=slot_selected,
            slot_executable=slot_active,
            explanation_tokens=explanation_tokens)

    def forward(
        self,
        decisionBackbone: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
        decisionContext: Optional[PackedDecisionContext] = None,
    ) -> PackedEndEffectorTarget:
        return self.Decode(
            decisionBackbone,
            feedbackPacket,
            decisionContext).target


class DecisionDecouplerV2(AGICoreModule):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        decisionDim: int,
        slotTokenDim: int = 128,
        feedbackTokenDim: int = 128,
        hierarchyDim: int = 128,
        hiddenDim: int = 256,
        planDim: Optional[int] = None,
        subgoalDim: Optional[int] = None,
        contextDim: Optional[int] = None,
        constraintTokenDim: Optional[int] = None,
        taskDim: int = 256,
        motionDim: int = 256,
        dynamicsDim: int = 128,
        constraintDim: int = 128,
        uncertaintyDim: int = 64,
        attentionHeads: int = 4,
        worldActionDim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.decoder = PackedHierarchicalDecisionDecoder(
            contractView=contractView,
            decisionDim=decisionDim,
            slotTokenDim=slotTokenDim,
            feedbackTokenDim=feedbackTokenDim,
            hierarchyDim=hierarchyDim,
            hiddenDim=hiddenDim,
            planDim=planDim,
            subgoalDim=subgoalDim,
            contextDim=contextDim,
            constraintTokenDim=constraintTokenDim,
            taskDim=taskDim,
            motionDim=motionDim,
            dynamicsDim=dynamicsDim,
            constraintDim=constraintDim,
            uncertaintyDim=uncertaintyDim,
            attentionHeads=attentionHeads,
            worldActionDim=worldActionDim)

    @property
    def ContractView(self) -> RobotEmbodimentContractView:
        return self.decoder.contract_view

    def BuildNeutralContext(
        self,
        decisionFeature: torch.Tensor,
    ) -> PackedDecisionContext:
        return self.decoder.BuildNeutralContext(decisionFeature)

    def DecodeContract(
        self,
        decisionFeature: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
        decisionContext: Optional[PackedDecisionContext] = None,
    ) -> PackedDecoupledDecision:
        return self.decoder.Decode(
            decisionBackbone=decisionFeature,
            feedbackPacket=feedbackPacket,
            decisionContext=decisionContext)

    def EncodeWorldAction(
        self,
        target: PackedEndEffectorTarget,
    ) -> torch.Tensor:
        return self.decoder.EncodeWorldAction(target)

    def TrainingConstraintLoss(
        self,
        decision: PackedDecoupledDecision,
        feedbackPacket: BrainFeedbackPacket,
        decisionContext: PackedDecisionContext,
        continuationMask: Optional[torch.Tensor] = None,
    ) -> dict:
        if type(decision) is not PackedDecoupledDecision:
            raise TypeError("constraint loss requires PackedDecoupledDecision")
        if type(decisionContext) is not PackedDecisionContext:
            raise TypeError("constraint loss requires PackedDecisionContext")
        feedbackPacket.Validate(self.ContractView)
        target = decision.target
        target.Validate(self.ContractView)
        if target.contract_id != self.ContractView.contract_id:
            raise ValueError("constraint loss contract identity mismatch")
        batch_size = int(feedbackPacket.joint_features.size(0))
        slot_shape = (batch_size, self.ContractView.end_effector_count)
        if (
            tuple(target.values.shape) != (
                batch_size,
                self.ContractView.end_effector_target_layout.PackedDim)
            or target.values.device != feedbackPacket.joint_features.device
            or not target.values.is_floating_point()
            or not bool(torch.isfinite(target.values).all().item())
        ):
            raise ValueError("constraint loss target values are invalid")
        bool_fields = (
            target.active,
            decision.slot_legal,
            decision.slot_available,
            decision.hierarchy_enabled,
            decision.slot_executable,
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != slot_shape
            or value.dtype != torch.bool
            or value.device != target.values.device
            for value in bool_fields
        ):
            raise ValueError("constraint loss decision masks are invalid")
        if not torch.equal(target.active, decision.slot_executable):
            raise ValueError("constraint loss executable mask is inconsistent")
        real_fields = (
            decision.safety_logits,
            decision.legality_logits,
            decision.selection_logits,
            decision.selection_probability,
        )
        if any(
            not torch.is_tensor(value)
            or tuple(value.shape) != slot_shape
            or not value.is_floating_point()
            or value.device != target.values.device
            or value.dtype != target.values.dtype
            or not bool(torch.isfinite(value).all().item())
            for value in real_fields
        ):
            raise ValueError("constraint loss decision scores are invalid")
        if not torch.equal(decisionContext.slot_legal, decision.slot_legal):
            raise ValueError("constraint loss legality context is inconsistent")
        relevance = decisionContext.slot_relevance
        if relevance is not None and (
            tuple(relevance.shape) != slot_shape
            or relevance.device != target.values.device
            or relevance.dtype != target.values.dtype
            or not bool(torch.isfinite(relevance).all().item())
            or bool((relevance < 0.0).any().item())
        ):
            raise ValueError("constraint loss slot relevance is invalid")
        demonstrated_active = decisionContext.slot_selection_mask
        if demonstrated_active is not None and (
            tuple(demonstrated_active.shape) != slot_shape
            or demonstrated_active.dtype != torch.bool
            or demonstrated_active.device != target.values.device
        ):
            raise ValueError("constraint loss selection supervision is invalid")
        if continuationMask is None:
            continuation_mask = torch.zeros(
                batch_size,
                dtype=torch.bool,
                device=target.values.device)
        else:
            if (
                not torch.is_tensor(continuationMask)
                or tuple(continuationMask.shape) != (batch_size,)
                or continuationMask.dtype != torch.bool
                or continuationMask.device != target.values.device
            ):
                raise ValueError("constraint loss continuation mask is invalid")
            continuation_mask = continuationMask

        def MaskedMean(
            values: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            weight = mask.to(dtype=values.dtype)
            return (values * weight).sum() / weight.sum().clamp_min(1.0)

        hierarchy_loss = target.values.new_zeros(())
        action_safety_loss = target.values.new_zeros(())
        target_continuity_loss = target.values.new_zeros(())
        continuity_count = target.values.new_zeros(())
        previous_values = decisionContext.previous_target_values
        previous_active = decisionContext.previous_target_active
        for slotIndex in range(self.ContractView.end_effector_count):
            target_slice = self.ContractView.end_effector_target_layout.Slice(
                slotIndex)
            values = target.values[:, target_slice]
            lower = self.decoder.target_lower[target_slice].to(
                device=values.device,
                dtype=values.dtype)
            upper = self.decoder.target_upper[target_slice].to(
                device=values.device,
                dtype=values.dtype)
            midpoint = 0.5 * (lower + upper)
            activeValues = torch.where(
                target.active[:, slotIndex].unsqueeze(-1),
                values,
                midpoint.unsqueeze(0))
            normalizedValues = (
                2.0 * (activeValues - lower.unsqueeze(0))
                / (upper - lower).unsqueeze(0)
                - 1.0)
            normalizedValues = normalizedValues * target.active[
                :, slotIndex].to(dtype=values.dtype).unsqueeze(-1)
            energy = normalizedValues.square().mean(dim=-1)
            disabled = ~feedbackPacket.child_enabled[:, slotIndex]
            hierarchy_loss = hierarchy_loss + (
                energy * disabled.to(dtype=energy.dtype)).mean()
            action_safety_loss = action_safety_loss + (
                energy * decisionContext.risk).mean()
            if previous_values is not None and previous_active is not None:
                continuity_valid = (
                    continuation_mask
                    & decision.slot_executable[:, slotIndex]
                    & previous_active[:, slotIndex])
                normalizedPrevious = (
                    2.0
                    * (
                        torch.where(
                            previous_active[:, slotIndex].unsqueeze(-1),
                            previous_values[:, target_slice],
                            midpoint.unsqueeze(0))
                        - lower.unsqueeze(0))
                    / (upper - lower).unsqueeze(0)
                    - 1.0)
                continuity_error = F.smooth_l1_loss(
                    normalizedValues,
                    normalizedPrevious,
                    reduction="none").mean(dim=-1)
                continuity_weight = continuity_valid.to(dtype=values.dtype)
                target_continuity_loss = target_continuity_loss + (
                    continuity_error * continuity_weight).sum()
                continuity_count = continuity_count + continuity_weight.sum()
        target_continuity_loss = (
            target_continuity_loss
            / continuity_count.clamp_min(1.0))

        operational_target = (
            feedbackPacket.endpoint_valid
            & decision.hierarchy_enabled
            & decision.slot_legal)
        classification_mask = torch.ones_like(operational_target)
        legality_loss = MaskedMean(
            F.binary_cross_entropy_with_logits(
                decision.legality_logits,
                operational_target.to(dtype=target.values.dtype),
                reduction="none"),
            classification_mask)
        safety_target = (
            (1.0 - decisionContext.risk).unsqueeze(-1)
            * feedbackPacket.endpoint_valid.to(dtype=target.values.dtype))
        safety_prediction_loss = MaskedMean(
            F.binary_cross_entropy_with_logits(
                decision.safety_logits,
                safety_target.detach(),
                reduction="none"),
            classification_mask)

        if demonstrated_active is not None:
            selection_target = (
                demonstrated_active
                & operational_target).to(dtype=target.values.dtype)
        elif relevance is not None:
            selection_target = (
                relevance
                * operational_target.to(dtype=target.values.dtype))
            selection_target = selection_target / selection_target.sum(
                dim=-1,
                keepdim=True).clamp_min(1.0)
        else:
            selection_target = operational_target.to(
                dtype=target.values.dtype)
        selection_loss = MaskedMean(
            F.binary_cross_entropy_with_logits(
                decision.selection_logits,
                selection_target.detach(),
                reduction="none"),
            classification_mask)
        scale = 1.0 / float(max(self.ContractView.end_effector_count, 1))
        return {
            "hierarchy": hierarchy_loss * scale,
            "safety": action_safety_loss * scale,
            "legality": legality_loss,
            "safety_prediction": safety_prediction_loss,
            "selection": selection_loss,
            "target_continuity": target_continuity_loss}

    def DecodePerceptionRotationEfference(
        self,
        target: PackedEndEffectorTarget,
        feedbackPacket: BrainFeedbackPacket,
    ) -> PackedPerceptionRotationEfference:
        return self.decoder.DecodePerceptionRotationEfference(
            target,
            feedbackPacket)

    def forward(
        self,
        decisionFeature: torch.Tensor,
        feedbackPacket: BrainFeedbackPacket,
        decisionContext: Optional[PackedDecisionContext] = None,
    ) -> PackedEndEffectorTarget:
        return self.DecodeContract(
            decisionFeature=decisionFeature,
            feedbackPacket=feedbackPacket,
            decisionContext=decisionContext).target
