from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import AGICoreModule, HungarianAssignment, BuildReferenceWeights
from ModuleMessagerManager import ModuleDim
from PerceptionModule import PerceiveExtractor, PerceptionRecallLoss, TopDownContext, VisualState
from RobotMorphologyModule import (
    BrainFeedbackPacket,
    RobotEmbodimentContractView)


SelfRealmIndex = 0
VirtualRealmIndex = 2
EffectRealmIndex = 3
CarrierMotionIndex = 1
ArticulationMotionIndex = 2
SurfaceContentMotionIndex = 3


class ContractPhysicalStateAdapter(nn.Module):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        bodyDim: int,
        controlFeedbackDim: Optional[int] = None,
        embodimentContextDim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("physical state adapter requires a contract view")
        contractView.Validate()
        if type(bodyDim) is not int or bodyDim < 1:
            raise ValueError("bodyDim must be a positive integer")
        if controlFeedbackDim is None:
            controlFeedbackDim = bodyDim
        if embodimentContextDim is None:
            embodimentContextDim = bodyDim
        if type(controlFeedbackDim) is not int or controlFeedbackDim < 1:
            raise ValueError(
                "controlFeedbackDim must be a positive integer")
        if type(embodimentContextDim) is not int or embodimentContextDim < 1:
            raise ValueError(
                "embodimentContextDim must be a positive integer")
        self.ContractView = contractView
        self.BodyDim = int(bodyDim)
        self.ControlFeedbackDim = int(controlFeedbackDim)
        self.EmbodimentContextDim = int(embodimentContextDim)
        static_joint_tokens = torch.tensor(
            contractView.static_joint_tokens,
            dtype=torch.float32)
        static_slot_tokens = torch.tensor(
            contractView.static_end_effector_tokens,
            dtype=torch.float32)
        endpoint_joint_mask = torch.zeros(
            contractView.end_effector_count,
            contractView.joint_count,
            dtype=torch.bool)
        for endpointIndex in range(contractView.end_effector_count):
            begin = contractView.end_effector_joint_chain_offsets[endpointIndex]
            end = contractView.end_effector_joint_chain_offsets[
                endpointIndex + 1]
            endpoint_joint_mask[
                endpointIndex,
                list(contractView.end_effector_joint_chain_indices[begin:end])
            ] = True
        self.register_buffer(
            "StaticJointTokens",
            static_joint_tokens,
            persistent=True)
        self.register_buffer(
            "StaticSlotTokens",
            static_slot_tokens,
            persistent=True)
        self.register_buffer(
            "EndpointJointMask",
            endpoint_joint_mask,
            persistent=True)
        self.JointStaticAdapter = nn.Linear(
            contractView.model_shape.joint_static_descriptor_dim,
            self.BodyDim)
        self.JointFeedbackAdapters = nn.ModuleList([
            nn.Sequential(
                (
                    nn.Identity()
                    if contractView.joint_feedback_layout.Width(jointIndex) == 1
                    else nn.LayerNorm(
                        contractView.joint_feedback_layout.Width(jointIndex))),
                nn.Linear(
                    contractView.joint_feedback_layout.Width(jointIndex),
                    self.BodyDim),
                nn.SiLU())
            for jointIndex in range(contractView.joint_count)
        ])
        self.JointOutputNorm = (
            nn.Identity()
            if self.BodyDim == 1
            else nn.LayerNorm(self.BodyDim))
        self.StaticAdapter = nn.Linear(
            contractView.model_shape.end_effector_static_descriptor_dim,
            self.BodyDim)
        self.EndpointQueryAdapter = nn.Linear(
            contractView.model_shape.end_effector_static_descriptor_dim,
            self.BodyDim)
        self.StatusAdapter = nn.Sequential(
            nn.Linear(8, self.BodyDim),
            nn.SiLU(),
            nn.Linear(self.BodyDim, self.BodyDim))
        self.OutputNorm = (
            nn.Identity()
            if self.BodyDim == 1
            else nn.LayerNorm(self.BodyDim))
        self.ControlJointAdapter = nn.Linear(
            self.BodyDim,
            self.ControlFeedbackDim)
        self.ControlStaticAdapter = nn.Linear(
            contractView.model_shape.end_effector_static_descriptor_dim,
            self.ControlFeedbackDim)
        self.ControlActivityAdapter = nn.Sequential(
            nn.Linear(8, self.ControlFeedbackDim),
            nn.SiLU(),
            nn.Linear(self.ControlFeedbackDim, self.ControlFeedbackDim))
        self.ControlOutputNorm = (
            nn.Identity()
            if self.ControlFeedbackDim == 1
            else nn.LayerNorm(self.ControlFeedbackDim))
        self.ContextJointAdapter = nn.Linear(
            self.BodyDim,
            self.EmbodimentContextDim)
        self.ContextEndpointAdapter = nn.Linear(
            self.BodyDim,
            self.EmbodimentContextDim)
        self.InvalidContextState = nn.Parameter(
            torch.zeros(self.EmbodimentContextDim))
        self.ContextOutputNorm = (
            nn.Identity()
            if self.EmbodimentContextDim == 1
            else nn.LayerNorm(self.EmbodimentContextDim))

    def ValidatePacket(self, feedback: BrainFeedbackPacket) -> None:
        if type(feedback) is not BrainFeedbackPacket:
            raise TypeError("physical state adapter requires BrainFeedbackPacket")
        feedback.Validate(self.ContractView)
        if feedback.joint_features.device != self.StaticJointTokens.device:
            raise ValueError(
                "physical state packet and adapter must share one device")
        if feedback.joint_features.dtype != self.StaticJointTokens.dtype:
            raise ValueError(
                "physical state packet and adapter must share one floating dtype")
        if feedback.progress.dtype != feedback.joint_features.dtype:
            raise ValueError("progress and joint features must share one dtype")

    def EncodeJointState(
        self,
        feedback: BrainFeedbackPacket,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dynamic_tokens = []
        for jointIndex, adapter in enumerate(self.JointFeedbackAdapters):
            dynamic_tokens.append(adapter(feedback.joint_features[
                ..., self.ContractView.joint_feedback_layout.Slice(jointIndex)]))
        dynamic = torch.stack(dynamic_tokens, dim=1)
        static = self.JointStaticAdapter(
            self.StaticJointTokens.to(
                device=dynamic.device,
                dtype=dynamic.dtype)).unsqueeze(0)
        joint_weight = feedback.joint_valid.to(dtype=dynamic.dtype)
        joint_tokens = self.JointOutputNorm(dynamic + static)
        joint_tokens = joint_tokens * joint_weight.unsqueeze(-1)
        joint_summary = joint_tokens.sum(dim=1) / joint_weight.sum(
            dim=1,
            keepdim=True).clamp_min(1.0)
        return joint_tokens, joint_weight, joint_summary

    def PoolEndpointChains(
        self,
        jointTokens: torch.Tensor,
        feedback: BrainFeedbackPacket,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self.EndpointQueryAdapter(
            self.StaticSlotTokens.to(
                device=jointTokens.device,
                dtype=jointTokens.dtype))
        score = torch.einsum(
            "bjd,ed->bej",
            jointTokens,
            queries) / float(self.BodyDim) ** 0.5
        chain_mask = self.EndpointJointMask.to(
            device=jointTokens.device).unsqueeze(0)
        valid_mask = chain_mask & feedback.joint_valid.unsqueeze(1)
        score = score.masked_fill(
            ~valid_mask,
            torch.finfo(score.dtype).min)
        weight = torch.softmax(score, dim=-1)
        weight = weight * valid_mask.to(dtype=weight.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weight.dtype).eps)
        endpoint_tokens = torch.bmm(weight, jointTokens)
        chain_size = chain_mask.sum(dim=-1).clamp_min(1)
        chain_ratio = valid_mask.sum(dim=-1).to(
            dtype=jointTokens.dtype) / chain_size.to(dtype=jointTokens.dtype)
        return endpoint_tokens, chain_ratio

    @staticmethod
    def EncodeEndpointStatus(
        feedback: BrainFeedbackPacket,
        chainRatio: torch.Tensor,
    ) -> torch.Tensor:
        dtype = feedback.joint_features.dtype
        target_active = feedback.target_active.to(dtype=dtype)
        child_enabled = feedback.child_enabled.to(dtype=dtype)
        progress = feedback.progress
        return torch.stack((
            progress,
            feedback.reached.to(dtype=dtype),
            child_enabled,
            target_active,
            feedback.endpoint_valid.to(dtype=dtype),
            chainRatio,
            progress * child_enabled,
            target_active * (1.0 - progress),
        ), dim=-1)

    def EncodeControlFeedback(
        self,
        jointSummary: torch.Tensor,
        endpointStatus: torch.Tensor,
        slotWeight: torch.Tensor,
    ) -> torch.Tensor:
        static = self.ControlStaticAdapter(
            self.StaticSlotTokens.to(
                device=jointSummary.device,
                dtype=jointSummary.dtype)).unsqueeze(0)
        endpoint_tokens = (
            static + self.ControlActivityAdapter(endpointStatus))
        endpoint_summary = (
            endpoint_tokens * slotWeight.unsqueeze(-1)
        ).sum(dim=1) / slotWeight.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.ControlOutputNorm(
            self.ControlJointAdapter(jointSummary) + endpoint_summary)

    def EncodeEmbodimentContext(
        self,
        jointSummary: torch.Tensor,
        bodySummary: torch.Tensor,
        jointWeight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        context_valid = jointWeight.gt(0.0).any(dim=-1)
        valid_feature = (
            self.ContextJointAdapter(jointSummary)
            + self.ContextEndpointAdapter(bodySummary))
        invalid_feature = self.InvalidContextState.unsqueeze(0).expand(
            jointSummary.size(0), -1)
        context_feature = torch.where(
            context_valid.unsqueeze(-1),
            valid_feature,
            invalid_feature)
        return self.ContextOutputNorm(context_feature), context_valid

    def forward(
        self,
        feedback: BrainFeedbackPacket,
    ) -> Dict[str, torch.Tensor]:
        self.ValidatePacket(feedback)
        joint_tokens, joint_weight, joint_summary = self.EncodeJointState(
            feedback)
        endpoint_dynamic, chain_ratio = self.PoolEndpointChains(
            joint_tokens,
            feedback)
        static = self.StaticAdapter(
            self.StaticSlotTokens.to(
                device=endpoint_dynamic.device,
                dtype=endpoint_dynamic.dtype)).unsqueeze(0)
        status = self.EncodeEndpointStatus(feedback, chain_ratio)
        slot_weight = feedback.endpoint_valid.to(dtype=endpoint_dynamic.dtype)
        slot_tokens = self.OutputNorm(
            endpoint_dynamic + static + self.StatusAdapter(status))
        slot_tokens = slot_tokens * slot_weight.unsqueeze(-1)
        body_summary = slot_tokens.sum(dim=1) / slot_weight.sum(
            dim=1,
            keepdim=True).clamp_min(1.0)
        control_feedback = self.EncodeControlFeedback(
            joint_summary,
            status,
            slot_weight)
        context_feature, context_valid = self.EncodeEmbodimentContext(
            joint_summary,
            body_summary,
            joint_weight)
        return {
            "SlotBodyTokens": slot_tokens,
            "BodySummary": body_summary,
            "SlotWeight": slot_weight,
            "JointStateTokens": joint_tokens,
            "JointWeight": joint_weight,
            "JointSummary": joint_summary,
            "ControlFeedbackFeature": control_feedback,
            "EmbodimentContextFeature": context_feature,
            "EmbodimentContextValid": context_valid,
            "StaticSlotTokens": static.expand(
                endpoint_dynamic.size(0), -1, -1),
        }


class SlotCrossAttention(AGICoreModule):
    def __init__(self, queryDim: int, sourceDim: int, numHeads: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(queryDim)
        self.source_norm = nn.LayerNorm(sourceDim)
        self.attn = nn.MultiheadAttention(
            embed_dim=queryDim,
            num_heads=numHeads,
            kdim=sourceDim,
            vdim=sourceDim,
            batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        sourceWeight: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(query)
        source = self.source_norm(source)
        B, N = sourceWeight.shape
        T = query.size(1)
        H = self.attn.num_heads
        source_visible = sourceWeight > 0.0
        empty_source = ~source_visible.any(dim=1)
        source_visible = source_visible.clone()
        source_visible[empty_source, 0] = True
        log_weight = torch.log(sourceWeight.clamp_min(1e-6))
        log_weight = log_weight.masked_fill(~source_visible, torch.finfo(log_weight.dtype).min)
        attn_mask = log_weight.view(B, 1, 1, N).expand(-1, H, T, -1).reshape(B * H, T, N)
        out, _ = self.attn(query=query, key=source, value=source, attn_mask=attn_mask, need_weights=False)
        return out


class SlotRefineLayer(AGICoreModule):
    def __init__(self, slotDim: int, numHeads: int = 4):
        super().__init__()
        self.attn = SlotCrossAttention(slotDim, slotDim, numHeads=numHeads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(slotDim),
            nn.Linear(slotDim, slotDim * 2),
            nn.SiLU(),
            nn.Linear(slotDim * 2, slotDim))

    def forward(self, S: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        slot_weight = M.unsqueeze(-1)
        S = S * slot_weight
        S = S + self.attn(S, S, M) * slot_weight
        S = S + self.ffn(S) * slot_weight
        return S * slot_weight


class PhysicalStateExtractor(AGICoreModule):
    @staticmethod
    def MakeHead(inDim: int, outDim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(inDim),
            nn.Linear(inDim, inDim),
            nn.SiLU(),
            nn.Linear(inDim, outDim))

    @staticmethod
    def MakeDeepHead(inDim: int, outDim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(inDim),
            nn.Linear(inDim, inDim),
            nn.SiLU(),
            nn.Linear(inDim, inDim),
            nn.SiLU(),
            nn.Linear(inDim, outDim))

    @staticmethod
    def SemanticWorldView(nodes: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        level_prob = F.softmax(nodes["level_logits"], dim=-1)
        object_prob = F.softmax(nodes["object_class_logits"], dim=-1)
        part_prob = F.softmax(nodes["part_class_logits"], dim=-1)
        text_prob = F.softmax(nodes["has_text_logits"], dim=-1)
        symbol_prob = F.softmax(nodes["symbol_logits"], dim=-1)
        physical_semantic = torch.cat([level_prob, object_prob, part_prob], dim=-1)
        identity_embed = F.normalize(nodes["identity_embed"], dim=-1, eps=1e-6)
        identity_key = F.normalize(
            torch.cat([identity_embed, 0.25 * physical_semantic], dim=-1),
            dim=-1,
            eps=1e-6)
        return {
            "NodePresence": F.softmax(nodes["node_logits"], dim=-1)[..., 1],
            "SpatialFrame": nodes["spatial_frame"],
            "Size": nodes["size_3d"],
            "BBox2D": nodes["bbox_2d"],
            "IdentityKey": identity_key,
            "Semantic": physical_semantic,
            "LevelProb": level_prob,
            "ObjectClassProb": object_prob,
            "PartClassProb": part_prob,
            "ParentProb": F.softmax(nodes["parent_logits"], dim=-1),
            "Visibility": nodes["visible_ratio"],
            "Occlusion": nodes["occlusion_ratio"],
            "HasTextProb": text_prob[..., 1],
            "TextEmbed": nodes["text_embed"],
            "SymbolProb": symbol_prob}

    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        inObjectDim: int = ModuleDim.PerceptionEmbed,
        motionDim: int = ModuleDim.PerceptionEmbed,
        slotDim: int = ModuleDim.PstSlotDim,
        poseDim: int = ModuleDim.PstPoseDim,
        attrDim: int = ModuleDim.PstAttrDim,
        affordanceDim: int = ModuleDim.PstAffordanceDim,
        stateDim: int = ModuleDim.PstStateDim,
        relationClasses: int = ModuleDim.PstRelationClasses,
        numSlotLayers: int = 3,
        numHeads: int = 4,
        nodeMaskThreshold: float = 0.5,
        geometryMaskThreshold: float = 0.5):
        super().__init__()
        self.slot_dim = int(slotDim)
        self.pose_dim = int(poseDim)
        self.attr_dim = int(attrDim)
        self.affordance_dim = int(affordanceDim)
        self.state_dim = int(stateDim)
        self.relation_classes = int(relationClasses)
        self.node_mask_threshold = float(nodeMaskThreshold)
        self.geometry_mask_threshold = float(geometryMaskThreshold)
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError(
                "physical state requires an immutable contract view")
        contractView.Validate()
        node_descriptor = torch.tensor(
            contractView.static_end_effector_tokens,
            dtype=torch.float32)
        self.self_part_count = int(contractView.end_effector_count)
        self.register_buffer(
            "self_part_node_descriptor",
            node_descriptor,
            persistent=False)

        self.in_proj = nn.Linear(inObjectDim, self.slot_dim)
        self.motion_in_proj = nn.Linear(motionDim, self.slot_dim)
        self.objectness_head = self.MakeHead(self.slot_dim, 1)
        self.physicality_head = self.MakeDeepHead(self.slot_dim, 1)
        self.interaction_head = self.MakeDeepHead(self.slot_dim, 1)

        self.slot_layers = nn.ModuleList([
            SlotRefineLayer(self.slot_dim, numHeads=numHeads) for _ in range(int(numSlotLayers))])

        self.slot_post = self.MakeHead(self.slot_dim, self.slot_dim)
        self.state_head = self.MakeHead(self.slot_dim, self.state_dim)
        self.attribute_head = self.MakeHead(self.slot_dim, self.attr_dim)
        self.affordance_head = self.MakeDeepHead(self.slot_dim, self.affordance_dim)
        self.motion_head = self.MakeHead(self.slot_dim, self.pose_dim)
        self.moving_head = self.MakeHead(self.slot_dim, 1)
        self.contact_head = self.MakeDeepHead(self.slot_dim, 6)
        relation_hidden = self.slot_dim // 2
        self.relation_subject = nn.Linear(self.slot_dim, relation_hidden)
        self.relation_object = nn.Linear(self.slot_dim, relation_hidden)
        self.relation_geometry = nn.Linear(3, relation_hidden)

        self.relation_head = nn.Sequential(
            nn.Linear(relation_hidden, relation_hidden),
            nn.SiLU(),
            nn.Linear(relation_hidden, self.relation_classes))

        self.external_relation_head = self.MakeHead(self.slot_dim, self.relation_classes)

        self.realm_head = self.MakeDeepHead(
            self.slot_dim, ModuleDim.PstRealmClasses)
        self.motion_layer_head = self.MakeDeepHead(
            self.slot_dim, ModuleDim.PstMotionLayerClasses)
        self.layer_agency_head = self.MakeDeepHead(
            self.slot_dim,
            ModuleDim.PstMotionLayerClasses * ModuleDim.PstAgencyClasses)
        self.body_membership_head = self.MakeDeepHead(self.slot_dim, 1)
        descriptor_dim = int(self.self_part_node_descriptor.size(-1))
        self.self_part_semantic_dim = ModuleDim.PstSelfPartSemanticDim
        descriptor_normalizer = (
            nn.Identity()
            if descriptor_dim == 1
            else nn.LayerNorm(descriptor_dim))
        self.self_part_node_encoder = nn.Sequential(
            descriptor_normalizer,
            nn.Linear(descriptor_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.LayerNorm(self.slot_dim))
        self.self_part_slot_query = self.MakeHead(
            self.slot_dim,
            self.slot_dim)
        self.self_part_node_key = self.MakeHead(
            self.slot_dim,
            self.slot_dim)
        semantic_descriptor_normalizer = (
            nn.Identity()
            if descriptor_dim == 1
            else nn.LayerNorm(descriptor_dim))
        self.self_part_semantic_proj = nn.Sequential(
            semantic_descriptor_normalizer,
            nn.Linear(descriptor_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(
                self.slot_dim,
                self.self_part_semantic_dim))
        self.contract_body_key = self.MakeHead(
            self.slot_dim,
            self.slot_dim)
        self.contract_body_semantic = self.MakeHead(
            self.slot_dim,
            self.self_part_semantic_dim)

        self.carrier_motion_head = self.MakeDeepHead(
            self.slot_dim, self.pose_dim)
        self.articulation_motion_head = self.MakeDeepHead(
            self.slot_dim, self.pose_dim)
        self.content_motion_head = self.MakeDeepHead(self.slot_dim, 2)
        self.content_change_head = self.MakeDeepHead(self.slot_dim, 1)

        self.display_surface_head = self.MakeDeepHead(self.slot_dim, 1)
        self.surface_parent_subject = nn.Linear(self.slot_dim, self.slot_dim)
        self.surface_parent_object = nn.Linear(self.slot_dim, self.slot_dim)
        self.surface_parent_null = self.MakeHead(self.slot_dim, 1)
        self.surface_uv_head = self.MakeDeepHead(self.slot_dim, 2)
        self.surface_uv_confidence_head = self.MakeDeepHead(self.slot_dim, 1)
        self.verification_head = self.MakeDeepHead(self.slot_dim, 1)

        ontology_hidden = self.slot_dim // 2
        self.ontology_relation_subject = nn.Linear(
            self.slot_dim, ontology_hidden)
        self.ontology_relation_object = nn.Linear(
            self.slot_dim, ontology_hidden)
        self.ontology_relation_head = nn.Sequential(
            nn.LayerNorm(ontology_hidden),
            nn.Linear(ontology_hidden, ontology_hidden),
            nn.SiLU(),
            nn.Linear(ontology_hidden, ModuleDim.PstOntologyRelationClasses))

    def NormalizePose(self, rawPose: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(rawPose[..., 3:7], dim=-1, eps=1e-6)
        pivot_index = quat.abs().argmax(dim=-1, keepdim=True)
        pivot = quat.gather(-1, pivot_index)
        quat = torch.where(pivot < 0.0, -quat, quat)
        return torch.cat([rawPose[..., :3], quat], dim=-1)

    def BuildRelations(
        self,
        S: torch.Tensor,
        P: torch.Tensor,
        semanticPairMask: torch.Tensor,
        geometryPairMask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        relative = P[..., :3].unsqueeze(1) - P[..., :3].unsqueeze(2)
        distance = relative.norm(dim=-1, keepdim=True)
        pair = (
            self.relation_subject(S).unsqueeze(2)
            + self.relation_object(S).unsqueeze(1)
            + self.relation_geometry(relative))

        relation_logits = self.relation_head(F.silu(pair))
        relation_prob = F.softmax(relation_logits, dim=-1)
        geometry = torch.cat([relative, distance], dim=-1)
        relation = torch.cat([
            geometry * geometryPairMask.unsqueeze(-1),
            relation_prob * semanticPairMask.unsqueeze(-1),], dim=-1)
        return relation, relation_logits

    def BuildSurfaceParents(
        self,
        slots: torch.Tensor,
        perceptualMask: torch.Tensor,
        displaySurfaceProb: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = float(self.slot_dim) ** -0.5
        logits = torch.matmul(
            self.surface_parent_subject(slots),
            self.surface_parent_object(slots).transpose(1, 2)) * scale
        parent_valid = perceptualMask.unsqueeze(1) * displaySurfaceProb.unsqueeze(1)
        logits = logits + (parent_valid + 1e-6).log()
        self_parent = torch.eye(
            logits.size(1),
            device=logits.device,
            dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(
            self_parent,
            torch.finfo(logits.dtype).min)
        null_logit = self.surface_parent_null(slots).squeeze(-1)
        probability = F.softmax(
            torch.cat([logits, null_logit.unsqueeze(-1)], dim=-1),
            dim=-1)
        subject_valid = perceptualMask.unsqueeze(-1)
        null_distribution = torch.zeros_like(probability)
        null_distribution[..., -1] = 1.0
        probability = torch.where(
            subject_valid > 0.0,
            probability,
            null_distribution)
        return probability, logits

    @staticmethod
    def MarginalAgency(
        motionLayerProb: torch.Tensor,
        layerAgencyProb: torch.Tensor,
        ) -> torch.Tensor:
        weighted = (
            motionLayerProb.unsqueeze(-1) * layerAgencyProb
        ).sum(dim=-2)
        layer_mass = motionLayerProb.sum(dim=-1, keepdim=True)
        marginal = weighted / (layer_mass + 1e-6)
        no_motion = layer_mass <= 1e-6
        unknown = torch.zeros_like(marginal)
        unknown[..., -1] = 1.0
        return torch.where(no_motion, unknown, marginal)

    def ValidateContractBodyState(
        self,
        objectTokens: torch.Tensor,
        slotBodyTokens: Optional[torch.Tensor],
        slotWeight: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if slotBodyTokens is None or slotWeight is None:
            raise ValueError(
                "contract body schema requires slot body tokens and weights")
        expected_tokens = (
            int(objectTokens.size(0)),
            self.self_part_count,
            self.slot_dim)
        expected_weight = (
            int(objectTokens.size(0)),
            self.self_part_count)
        if tuple(slotBodyTokens.shape) != expected_tokens:
            raise ValueError("slotBodyTokens has invalid shape")
        if tuple(slotWeight.shape) != expected_weight:
            raise ValueError("slotWeight has invalid shape")
        if not slotBodyTokens.is_floating_point():
            raise TypeError("slotBodyTokens must use floating point values")
        if not slotWeight.is_floating_point():
            raise TypeError("slotWeight must use floating point values")
        if slotBodyTokens.device != objectTokens.device:
            raise ValueError("slotBodyTokens must share the object token device")
        if slotWeight.device != objectTokens.device:
            raise ValueError("slotWeight must share the object token device")
        if not torch.isfinite(slotBodyTokens).all():
            raise ValueError("slotBodyTokens must be finite")
        if not torch.isfinite(slotWeight).all():
            raise ValueError("slotWeight must be finite")
        if bool((slotWeight < 0.0).any().item()):
            raise ValueError("slotWeight must be nonnegative")
        return (
            slotBodyTokens.to(dtype=objectTokens.dtype),
            slotWeight.to(dtype=objectTokens.dtype))

    def forward(
        self,
        objectTokens: torch.Tensor, # [B, K, D_obj]
        objectMotion: torch.Tensor, # [B, K, D_motion]
        objectGeometry: torch.Tensor, # [B, K, 7]
        nodeMask: torch.Tensor,
        geometryValid: torch.Tensor,
        slotBodyTokens: torch.Tensor,
        slotWeight: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        observation_mask = (
            nodeMask > self.node_mask_threshold
        ).to(objectTokens.dtype)
        geometry_mask = (
            geometryValid.squeeze(-1) > self.geometry_mask_threshold
        ).to(objectTokens.dtype)
        feature_mask = observation_mask

        S_raw = self.in_proj(objectTokens) + self.motion_in_proj(objectMotion) # [B, K, 128]

        for layer in self.slot_layers:
            S_raw = layer(S_raw, feature_mask)

        S_feature = self.slot_post(S_raw) # [B, K, 128]
        S_raw = S_feature * observation_mask.unsqueeze(-1)

        perceptual_presence_logits = self.objectness_head(S_feature).squeeze(-1)
        perceptual_presence = (
            torch.sigmoid(perceptual_presence_logits) * observation_mask)
        entity_available = perceptual_presence.unsqueeze(-1)

        realm_logits = self.realm_head(S_feature)
        realm_prob = F.softmax(realm_logits, dim=-1)
        realm_class = realm_prob.argmax(dim=-1)
        independent_3d_mask = (
            realm_class.ne(VirtualRealmIndex)
            & realm_class.ne(EffectRealmIndex)
        ).to(objectTokens.dtype)
        canonical_geometry_mask = geometry_mask * independent_3d_mask

        mphys_logits = self.physicality_head(S_feature).squeeze(-1)
        physical_entity_prob = (
            torch.sigmoid(mphys_logits)
            * perceptual_presence
            * independent_3d_mask)
        interaction_logits = self.interaction_head(S_feature).squeeze(-1)
        physical_interaction_prob = (
            torch.sigmoid(interaction_logits)
            * physical_entity_prob
            * canonical_geometry_mask)
        geometry_available = (
            physical_entity_prob * canonical_geometry_mask).unsqueeze(-1)

        state_logits = self.state_head(S_feature) # [B, K, PstStateDim]
        state_raw = torch.sigmoid(state_logits) * entity_available

        attr_pred = self.attribute_head(S_feature)
        attr_raw = attr_pred * entity_available

        affordance_logits = self.affordance_head(S_feature)
        affordance_raw = torch.sigmoid(affordance_logits) * entity_available

        motion_pred = self.NormalizePose(self.motion_head(S_feature))
        identity_motion = torch.zeros_like(motion_pred)
        identity_motion[..., 6] = 1.0
        motion_raw = self.NormalizePose(
            identity_motion
            + (motion_pred - identity_motion) * geometry_available)

        moving_logits = self.moving_head(S_feature).squeeze(-1)
        moving_raw = (
            torch.sigmoid(moving_logits)
            * physical_entity_prob
            * canonical_geometry_mask)

        contact_raw = self.contact_head(S_feature) # [B, K, 6]
        contact_logits = contact_raw[..., 0]
        contact_prob_raw = (
            torch.sigmoid(contact_logits) * physical_interaction_prob)
        contact_force_pred = F.softplus(contact_raw[..., 1:3])
        contact_point_pred = contact_raw[..., 3:6]
        contact_weight = contact_prob_raw.unsqueeze(-1)
        contact_force_raw = contact_force_pred * contact_weight
        contact_point_raw = contact_point_pred * contact_weight

        external_relation_logits = self.external_relation_head(S_feature)
        external_relation_raw = (
            torch.sigmoid(external_relation_logits) * entity_available)
        semantic_pair_mask = (
            observation_mask.unsqueeze(1) * observation_mask.unsqueeze(2))
        geometry_pair_mask = (
            canonical_geometry_mask.unsqueeze(1)
            * canonical_geometry_mask.unsqueeze(2)
            * semantic_pair_mask)

        off_diagonal = 1.0 - torch.eye(
            observation_mask.size(1), device=observation_mask.device, dtype=observation_mask.dtype)

        semantic_pair_mask = (
            semantic_pair_mask * off_diagonal.unsqueeze(0))
        geometry_pair_mask = geometry_pair_mask * off_diagonal.unsqueeze(0)
        pairwise_relation, relation_logits_raw = self.BuildRelations(
            S_feature,
            objectGeometry,
            semantic_pair_mask,
            geometry_pair_mask)

        motion_layer_logits = self.motion_layer_head(S_feature)
        motion_layer_prob = torch.sigmoid(motion_layer_logits)
        layer_agency_logits = self.layer_agency_head(S_feature).view(
            S_feature.size(0),
            S_feature.size(1),
            ModuleDim.PstMotionLayerClasses,
            ModuleDim.PstAgencyClasses)
        layer_agency_prob = F.softmax(layer_agency_logits, dim=-1)
        agency_prob = self.MarginalAgency(
            motion_layer_prob,
            layer_agency_prob)

        body_membership_logits = self.body_membership_head(
            S_feature).squeeze(-1)
        body_membership_prob = (
            torch.sigmoid(body_membership_logits)
            * realm_prob[..., SelfRealmIndex]
            * perceptual_presence)
        node_descriptor = self.self_part_node_descriptor.to(
            device=S_feature.device,
            dtype=S_feature.dtype)
        encoded_node = self.self_part_node_encoder(node_descriptor)
        contract_body_tokens, contract_slot_weight = (
            self.ValidateContractBodyState(
                objectTokens,
                slotBodyTokens,
                slotWeight))
        dynamic_node = self.contract_body_key(contract_body_tokens)
        dynamic_node = dynamic_node * contract_slot_weight.unsqueeze(-1)
        node_key = self.self_part_node_key(
            encoded_node.unsqueeze(0) + dynamic_node)
        slot_query = self.self_part_slot_query(S_feature)
        raw_self_part_logits = torch.einsum(
            "bkd,bnd->bkn",
            slot_query,
            node_key) * (float(self.slot_dim) ** -0.5)
        contract_part_valid = contract_slot_weight > 0.0
        log_weight = torch.log(
            contract_slot_weight.clamp_min(
                torch.finfo(contract_slot_weight.dtype).tiny))
        log_weight = log_weight.masked_fill(
            ~contract_part_valid,
            torch.finfo(raw_self_part_logits.dtype).min)
        raw_self_part_logits = raw_self_part_logits + log_weight.unsqueeze(1)
        empty_contract_body = ~contract_part_valid.any(dim=-1)
        raw_self_part_logits = torch.where(
            empty_contract_body.view(-1, 1, 1),
            torch.zeros_like(raw_self_part_logits),
            raw_self_part_logits)
        self_part_logits = raw_self_part_logits
        self_part_distribution = F.softmax(
            self_part_logits,
            dim=-1)
        self_part_distribution = (
            self_part_distribution
            * contract_part_valid.unsqueeze(1).to(
                dtype=self_part_distribution.dtype))
        self_part_distribution = torch.where(
            empty_contract_body.view(-1, 1, 1),
            torch.zeros_like(self_part_distribution),
            self_part_distribution)
        self_part_prob = (
            self_part_distribution
            * body_membership_prob.unsqueeze(-1))
        weighted_self_part_descriptor = torch.einsum(
            "bkn,nd->bkd",
            self_part_prob,
            node_descriptor)
        self_part_mass = self_part_prob.sum(dim=-1, keepdim=True)
        self_part_semantic = (
            self.self_part_semantic_proj(weighted_self_part_descriptor)
            * self_part_mass)
        dynamic_semantic = self.contract_body_semantic(contract_body_tokens)
        dynamic_semantic = dynamic_semantic * contract_slot_weight.unsqueeze(-1)
        self_part_semantic = self_part_semantic + torch.einsum(
            "bkn,bnd->bkd",
            self_part_prob,
            dynamic_semantic)

        carrier_motion_pred = self.NormalizePose(
            self.carrier_motion_head(S_feature))
        articulation_motion_pred = self.NormalizePose(
            self.articulation_motion_head(S_feature))
        carrier_weight = (
            geometry_available
            * motion_layer_prob[..., CarrierMotionIndex].unsqueeze(-1))
        articulation_weight = (
            geometry_available
            * motion_layer_prob[..., ArticulationMotionIndex].unsqueeze(-1))
        carrier_motion_raw = self.NormalizePose(
            identity_motion
            + (carrier_motion_pred - identity_motion) * carrier_weight)
        articulation_motion_raw = self.NormalizePose(
            identity_motion
            + (articulation_motion_pred - identity_motion) * articulation_weight)

        content_motion_pred = torch.tanh(self.content_motion_head(S_feature))
        content_layer = motion_layer_prob[
            ..., SurfaceContentMotionIndex]
        content_motion_uv = (
            content_motion_pred
            * content_layer.unsqueeze(-1)
            * perceptual_presence.unsqueeze(-1))
        content_change_logits = self.content_change_head(
            S_feature).squeeze(-1)
        content_change_prob = (
            torch.sigmoid(content_change_logits) * perceptual_presence)

        display_surface_logits = self.display_surface_head(
            S_feature).squeeze(-1)
        display_surface_prob = (
            torch.sigmoid(display_surface_logits) * perceptual_presence)
        surface_parent_prob, surface_parent_logits = self.BuildSurfaceParents(
            S_feature,
            observation_mask,
            display_surface_prob)
        surface_uv = (
            torch.sigmoid(self.surface_uv_head(S_feature))
            * perceptual_presence.unsqueeze(-1))
        surface_uv_confidence_logits = self.surface_uv_confidence_head(
            S_feature).squeeze(-1)
        surface_uv_confidence = (
            torch.sigmoid(surface_uv_confidence_logits)
            * perceptual_presence)
        verification_logits = self.verification_head(
            S_feature).squeeze(-1)
        verification_confidence = (
            torch.sigmoid(verification_logits) * perceptual_presence)

        ontology_pair = (
            self.ontology_relation_subject(S_feature).unsqueeze(2)
            + self.ontology_relation_object(S_feature).unsqueeze(1))
        ontology_relation_logits = self.ontology_relation_head(
            F.silu(ontology_pair))
        ontology_relation_prob = (
            torch.sigmoid(ontology_relation_logits)
            * semantic_pair_mask.unsqueeze(-1))

        return {
            "SlotState": S_raw,
            "ObservationMask": observation_mask,
            "PerceptualPresenceLogits": perceptual_presence_logits,
            "PerceptualPresence": perceptual_presence,
            "GeometryValidMask": canonical_geometry_mask,
            "MphysLogits": mphys_logits,
            "MphysRaw": physical_entity_prob,
            "PhysicalEntityProb": physical_entity_prob,
            "PhysicalInteractionLogits": interaction_logits,
            "PhysicalInteractionProb": physical_interaction_prob,
            "AttributePred": attr_pred,
            "ARaw": attr_raw,
            "StateLogits": state_logits,
            "StateRaw": state_raw,
            "AffordanceLogits": affordance_logits,
            "AffordanceRaw": affordance_raw,
            "MotionPred": motion_pred,
            "MotionObserverRaw": motion_raw,
            "MovingLogits": moving_logits,
            "MovingProbRaw": moving_raw,
            "ContactLogits": contact_logits,
            "ContactProbRaw": contact_prob_raw,
            "ContactForcePred": contact_force_pred,
            "ContactForceRaw": contact_force_raw,
            "ContactPointPred": contact_point_pred,
            "ContactPointObserverRaw": contact_point_raw,
            "PairwiseRelationObserver": pairwise_relation,
            "RelationLogitsRaw": relation_logits_raw,
            "ExternalRelationLogits": external_relation_logits,
            "ExternalRelationProbRaw": external_relation_raw,
            "RealmLogits": realm_logits,
            "RealmProb": realm_prob,
            "MotionLayerLogits": motion_layer_logits,
            "MotionLayerProb": motion_layer_prob,
            "LayerAgencyLogits": layer_agency_logits,
            "LayerAgencyProb": layer_agency_prob,
            "AgencyProb": agency_prob,
            "BodyMembershipLogits": body_membership_logits,
            "BodyMembershipProb": body_membership_prob,
            "SelfPartLogits": self_part_logits,
            "SelfPartProb": self_part_prob,
            "SelfPartSemantic": self_part_semantic,
            "SelfPartDynamicWeight": contract_slot_weight,
            "CarrierMotionPred": carrier_motion_pred,
            "CarrierMotionObserverRaw": carrier_motion_raw,
            "ArticulationMotionPred": articulation_motion_pred,
            "ArticulationMotionObserverRaw": articulation_motion_raw,
            "ContentMotionPred": content_motion_pred,
            "ContentMotionUV": content_motion_uv,
            "ContentChangeLogits": content_change_logits,
            "ContentChangeProb": content_change_prob,
            "DisplaySurfaceLogits": display_surface_logits,
            "DisplaySurfaceProb": display_surface_prob,
            "SurfaceParentLogits": surface_parent_logits,
            "SurfaceParentProb": surface_parent_prob,
            "SurfaceUV": surface_uv,
            "SurfaceUVConfidenceLogits": surface_uv_confidence_logits,
            "SurfaceUVConfidence": surface_uv_confidence,
            "VerificationLogits": verification_logits,
            "VerificationConfidence": verification_confidence,
            "OntologyRelationLogits": ontology_relation_logits,
            "OntologyRelationProb": ontology_relation_prob}

    def SlotSummary(self, S: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        weight = M.unsqueeze(-1)
        return (S * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)


class PhysicalStateLoss(nn.Module):
    def __init__(
        self,
        stateWeight: float = 1.0,
        attributeWeight: float = 1.0,
        affordanceWeight: float = 2.0,
        relationWeight: float = 1.0,
        motionWeight: float = 1.0,
        contactWeight: float = 1.0,
        mphysWeight: float = 1.0,
        mphysNegativeWeight: float = 0.05,
        ontologyWeight: float = 1.0,
        factorMotionWeight: float = 1.0,
        causalAgencyWeight: float = 0.05,):
        super().__init__()
        self.state_weight = float(stateWeight)
        self.attribute_weight = float(attributeWeight)
        self.affordance_weight = float(affordanceWeight)
        self.relation_weight = float(relationWeight)
        self.motion_weight = float(motionWeight)
        self.contact_weight = float(contactWeight)
        self.mphys_weight = float(mphysWeight)
        self.mphys_negative_weight = float(mphysNegativeWeight)
        self.ontology_weight = float(ontologyWeight)
        self.factor_motion_weight = float(factorMotionWeight)
        self.causal_agency_weight = float(causalAgencyWeight)

    @staticmethod
    def QuaternionAngle(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred_q = pred[..., 3:7]
        tgt_q = tgt[..., 3:7]
        dot = (pred_q * tgt_q).sum(dim=-1).abs().clamp(0.0, 1.0)
        return 2.0 * torch.atan2(torch.sqrt((1.0 - dot * dot).clamp_min(0.0)), dot.clamp_min(1e-6))

    def PoseCost(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        translation = torch.cdist(pred[..., :3], tgt[..., :3], p=1)
        pred_q = pred[..., 3:7]
        tgt_q = tgt[..., 3:7]
        dot = torch.matmul(pred_q, tgt_q.t()).abs().clamp(0.0, 1.0)
        angle = 2.0 * torch.atan2(torch.sqrt((1.0 - dot * dot).clamp_min(0.0)), dot.clamp_min(1e-6))
        return translation + angle

    def PoseLossEach(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        translation = F.smooth_l1_loss(pred[..., :3], tgt[..., :3], reduction="none").mean(dim=-1)
        return translation + self.QuaternionAngle(pred, tgt)

    def MatchedNodes(self, pst: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gt = torch.nonzero(targets["node_valid"][b], as_tuple=False).flatten()
        if gt.numel() == 0:
            empty = torch.empty(0, device=pst["SlotState"].device, dtype=torch.long)
            return empty, empty
        candidate = torch.arange(
            pst["MphysLogits"].size(1),
            device=pst["SlotState"].device)
        levels = targets["node_level"][b, gt]
        object_score = pst["ObjectClassProb"][b, candidate][:, targets["object_classes"][b, gt]]
        part_score = pst["PartClassProb"][b, candidate][:, targets["part_classes"][b, gt]]
        class_score = torch.where((levels == 0).unsqueeze(0), object_score, part_score)
        pose_cost = self.PoseCost(
            pst["SpatialFrame"][b, candidate],
            targets["spatial_frame"][b, gt])
        pose_cost = pose_cost * targets["pose_valid"][b, gt].unsqueeze(0)
        bbox_cost = torch.cdist(
            pst["BBox2D"][b, candidate],
            targets["bbox_2d"][b, gt],
            p=1)
        cost = (
            pose_cost
            + 0.5 * bbox_cost
            - torch.sigmoid(
                pst["PerceptualPresenceLogits"][b, candidate]).unsqueeze(1)
            - pst["LevelProb"][b, candidate][:, levels]
            - class_score)
        pred_local, local = HungarianAssignment(cost)
        return candidate[pred_local], gt[local]

    @staticmethod
    def MaskedMean(loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return (loss * valid).sum() / valid.sum().clamp_min(1.0)

    @staticmethod
    def MeanTerms(terms: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(terms).mean()

    @staticmethod
    def NormalizeAgencyProbability(
        probability: torch.Tensor,
    ) -> torch.Tensor:
        if not probability.is_floating_point():
            raise TypeError("causal agency probability must be floating point")
        if probability.size(-1) != ModuleDim.PstAgencyClasses:
            raise ValueError("causal agency probability class width is invalid")
        if not bool(torch.isfinite(probability).all().item()):
            raise ValueError("causal agency probability must be finite")
        if bool((probability < 0.0).any().item()):
            raise ValueError("causal agency probability must be nonnegative")
        mass = probability.sum(dim=-1, keepdim=True)
        if bool((mass <= 0.0).any().item()):
            raise ValueError("causal agency probability mass must be positive")
        return probability / mass

    def CausalAgencyLoss(
        self,
        pst: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batchIndex: int,
        pred: torch.Tensor,
        gt: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        probability = pst.get("CausalLayerAgencyProb")
        evidence_valid = pst.get("CausalAgencyEvidenceValid")
        if probability is None or evidence_valid is None:
            return zero
        if probability.shape != pst["LayerAgencyProb"].shape:
            raise ValueError("causal agency probability shape is invalid")
        if evidence_valid.shape != probability.shape[:2]:
            raise ValueError("causal agency evidence validity shape is invalid")
        matched_probability = probability[batchIndex, pred]
        matched_target = targets["agency_by_layer"][batchIndex, gt]
        matched_valid = targets[
            "agency_by_layer_valid"][batchIndex, gt].bool()
        matched_valid = (
            matched_valid
            & evidence_valid[batchIndex, pred].bool().unsqueeze(-1))
        if not bool(matched_valid.any().item()):
            return matched_probability.sum() * 0.0
        selected_probability = self.NormalizeAgencyProbability(
            matched_probability[matched_valid])
        selected_target = matched_target[matched_valid]
        return F.nll_loss(
            selected_probability.clamp_min(
                torch.finfo(selected_probability.dtype).tiny).log(),
            selected_target,
            reduction="mean")

    def forward(self, pst: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        presence_terms: List[torch.Tensor] = []
        mphys_terms: List[torch.Tensor] = []
        interaction_terms: List[torch.Tensor] = []
        realm_terms: List[torch.Tensor] = []
        motion_layer_terms: List[torch.Tensor] = []
        layer_agency_terms: List[torch.Tensor] = []
        causal_agency_terms: List[torch.Tensor] = []
        body_terms: List[torch.Tensor] = []
        self_part_terms: List[torch.Tensor] = []
        carrier_terms: List[torch.Tensor] = []
        articulation_terms: List[torch.Tensor] = []
        content_motion_terms: List[torch.Tensor] = []
        content_change_terms: List[torch.Tensor] = []
        display_surface_terms: List[torch.Tensor] = []
        surface_parent_terms: List[torch.Tensor] = []
        surface_uv_terms: List[torch.Tensor] = []
        surface_uv_confidence_terms: List[torch.Tensor] = []
        verification_terms: List[torch.Tensor] = []
        ontology_relation_terms: List[torch.Tensor] = []
        consistency_terms: List[torch.Tensor] = []
        state_terms: List[torch.Tensor] = []
        attribute_terms: List[torch.Tensor] = []
        affordance_terms: List[torch.Tensor] = []
        relation_terms: List[torch.Tensor] = []
        external_relation_terms: List[torch.Tensor] = []
        motion_terms: List[torch.Tensor] = []
        moving_terms: List[torch.Tensor] = []
        contact_terms: List[torch.Tensor] = []
        force_terms: List[torch.Tensor] = []
        point_terms: List[torch.Tensor] = []
        relation_weight = pst["SlotState"].new_ones(ModuleDim.PstRelationClasses)
        relation_weight[0] = 0.1
        for b in range(pst["SlotState"].size(0)):
            pred, gt = self.MatchedNodes(pst, targets, b)
            zero = pst["SlotState"][b].sum() * 0.0
            presence_target = torch.zeros_like(
                pst["PerceptualPresenceLogits"][b])
            presence_weight = torch.full_like(
                presence_target, self.mphys_negative_weight)
            presence_target[pred] = 1.0
            presence_weight[pred] = 1.0
            presence_raw = F.binary_cross_entropy_with_logits(
                pst["PerceptualPresenceLogits"][b],
                presence_target,
                reduction="none")
            presence_terms.append(
                (presence_raw * presence_weight).sum()
                / presence_weight.sum().clamp_min(1.0))
            if gt.numel() == 0:
                mphys_terms.append(zero)
                interaction_terms.append(zero)
                realm_terms.append(zero)
                motion_layer_terms.append(zero)
                layer_agency_terms.append(zero)
                causal_agency_terms.append(zero)
                body_terms.append(zero)
                self_part_terms.append(zero)
                carrier_terms.append(zero)
                articulation_terms.append(zero)
                content_motion_terms.append(zero)
                content_change_terms.append(zero)
                display_surface_terms.append(zero)
                surface_parent_terms.append(zero)
                surface_uv_terms.append(zero)
                surface_uv_confidence_terms.append(zero)
                verification_terms.append(zero)
                ontology_relation_terms.append(zero)
                consistency_terms.append(zero)
                state_terms.append(zero)
                attribute_terms.append(zero)
                affordance_terms.append(zero)
                relation_terms.append(zero)
                external_relation_terms.append(zero)
                motion_terms.append(zero)
                moving_terms.append(zero)
                contact_terms.append(zero)
                force_terms.append(zero)
                point_terms.append(zero)
                continue

            physical_valid = targets["physical_entity_valid"][b, gt]
            mphys_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["MphysLogits"][b, pred],
                    targets["physical_entity"][b, gt],
                    reduction="none"),
                physical_valid))
            interaction_valid = targets[
                "physical_interaction_valid"][b, gt]
            interaction_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["PhysicalInteractionLogits"][b, pred],
                    targets["physical_interaction"][b, gt],
                    reduction="none"),
                interaction_valid))

            realm_valid = targets["realm_valid"][b, gt]
            realm_raw = F.cross_entropy(
                pst["RealmLogits"][b, pred],
                targets["realm"][b, gt],
                reduction="none")
            realm_terms.append(self.MaskedMean(realm_raw, realm_valid))

            layer_valid = targets["motion_layer_valid"][b, gt]
            motion_layer_raw = F.binary_cross_entropy_with_logits(
                pst["MotionLayerLogits"][b, pred],
                targets["motion_layer_multi_hot"][b, gt],
                reduction="none").mean(dim=-1)
            motion_layer_terms.append(self.MaskedMean(
                motion_layer_raw, layer_valid))

            agency_valid = targets["agency_by_layer_valid"][b, gt]
            layer_agency_raw = F.cross_entropy(
                pst["LayerAgencyLogits"][b, pred].reshape(
                    -1, ModuleDim.PstAgencyClasses),
                targets["agency_by_layer"][b, gt].reshape(-1),
                reduction="none").view_as(agency_valid)
            layer_agency_terms.append(self.MaskedMean(
                layer_agency_raw, agency_valid))
            causal_agency_terms.append(self.CausalAgencyLoss(
                pst,
                targets,
                b,
                pred,
                gt,
                zero))

            body_valid = targets["body_membership_valid"][b, gt]
            body_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["BodyMembershipLogits"][b, pred],
                    targets["body_membership"][b, gt],
                    reduction="none"),
                body_valid))
            self_part_valid = targets["self_part_valid"][b, gt]
            self_part_raw = F.cross_entropy(
                pst["SelfPartLogits"][b, pred],
                targets["self_part_id"][b, gt],
                reduction="none")
            self_part_terms.append(self.MaskedMean(
                self_part_raw, self_part_valid))

            carrier_valid = targets["carrier_motion_valid"][b, gt]
            carrier_terms.append(self.MaskedMean(
                self.PoseLossEach(
                    pst["CarrierMotionPred"][b, pred],
                    targets["carrier_motion"][b, gt]),
                carrier_valid))
            articulation_valid = targets[
                "articulation_motion_valid"][b, gt]
            articulation_terms.append(self.MaskedMean(
                self.PoseLossEach(
                    pst["ArticulationMotionPred"][b, pred],
                    targets["articulation_motion"][b, gt]),
                articulation_valid))
            content_motion_valid = targets[
                "content_motion_uv_valid"][b, gt]
            content_motion_terms.append(self.MaskedMean(
                F.smooth_l1_loss(
                    pst["ContentMotionPred"][b, pred],
                    targets["content_motion_uv"][b, gt],
                    reduction="none").mean(dim=-1),
                content_motion_valid))
            content_change_valid = targets[
                "content_change_valid"][b, gt]
            content_change_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["ContentChangeLogits"][b, pred],
                    targets["content_change"][b, gt],
                    reduction="none"),
                content_change_valid))

            display_valid = targets["display_surface_valid"][b, gt]
            display_surface_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["DisplaySurfaceLogits"][b, pred],
                    targets["display_surface"][b, gt],
                    reduction="none"),
                display_valid))
            surface_uv_valid = targets["surface_uv_valid"][b, gt]
            surface_uv_terms.append(self.MaskedMean(
                F.smooth_l1_loss(
                    pst["SurfaceUV"][b, pred],
                    targets["surface_uv"][b, gt],
                    reduction="none").mean(dim=-1),
                surface_uv_valid))
            surface_uv_confidence_terms.append(
                F.binary_cross_entropy_with_logits(
                    pst["SurfaceUVConfidenceLogits"][b, pred],
                    surface_uv_valid.to(
                        pst["SurfaceUVConfidenceLogits"].dtype)))
            verification_valid = targets[
                "verification_confidence_valid"][b, gt]
            verification_terms.append(self.MaskedMean(
                F.smooth_l1_loss(
                    pst["VerificationConfidence"][b, pred],
                    targets["verification_confidence"][b, gt],
                    reduction="none"),
                verification_valid))

            surface_parent_valid = targets[
                "surface_parent_valid"][b, gt]
            parent_target_gt = targets["surface_parent_index"][b, gt]
            gt_to_pred = torch.full(
                (targets["node_valid"].size(1),),
                -1,
                device=pred.device,
                dtype=torch.long)
            gt_to_pred[gt] = pred
            parent_target = torch.full_like(
                parent_target_gt, pst["SlotState"].size(1))
            has_parent = parent_target_gt >= 0
            parent_target[has_parent] = gt_to_pred[
                parent_target_gt[has_parent]]
            mapped_parent = (~has_parent) | (parent_target >= 0)
            surface_parent_raw = -torch.log(torch.gather(
                pst["SurfaceParentProb"][b, pred],
                1,
                parent_target.clamp_min(0).unsqueeze(-1)
            ).squeeze(-1) + 1e-6)
            surface_parent_terms.append(self.MaskedMean(
                surface_parent_raw,
                surface_parent_valid & mapped_parent))

            ontology_target = targets[
                "ontology_relation_multi_hot"][b][
                    gt.unsqueeze(1), gt.unsqueeze(0)]
            ontology_valid = targets[
                "ontology_relation_valid"][b][
                    gt.unsqueeze(1), gt.unsqueeze(0)]
            ontology_logits = pst["OntologyRelationLogits"][b][
                pred.unsqueeze(1), pred.unsqueeze(0)]
            ontology_raw = F.binary_cross_entropy_with_logits(
                ontology_logits,
                ontology_target,
                reduction="none").mean(dim=-1)
            ontology_pair_valid = (
                ontology_valid
                & ~torch.eye(
                    pred.numel(), device=pred.device, dtype=torch.bool))
            ontology_relation_terms.append(self.MaskedMean(
                ontology_raw, ontology_pair_valid))

            realm_prob = pst["RealmProb"][b, pred]
            physical_prob = torch.sigmoid(pst["MphysLogits"][b, pred])
            interaction_prob = torch.sigmoid(
                pst["PhysicalInteractionLogits"][b, pred])
            body_prob = torch.sigmoid(
                pst["BodyMembershipLogits"][b, pred])
            virtual_or_effect = (
                realm_prob[..., VirtualRealmIndex]
                + realm_prob[..., EffectRealmIndex])
            ontology_consistency = (
                physical_prob * virtual_or_effect
                + interaction_prob * (1.0 - physical_prob)
                + body_prob * (
                    1.0 - realm_prob[..., SelfRealmIndex]))
            consistency_terms.append(ontology_consistency.mean())
            state_valid = targets["node_state_valid"][b, gt]
            state_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["StateLogits"][b, pred],
                    targets["node_state"][b, gt],
                    reduction="none").mean(dim=-1),
                state_valid))
            attr_valid = targets["node_attributes_valid"][b, gt]
            attribute_terms.append(self.MaskedMean(
                F.smooth_l1_loss(pst["AttributePred"][b, pred], targets["node_attributes"][b, gt], reduction="none").mean(dim=-1),
                attr_valid))
            affordance_valid = targets["affordance_valid"][b, gt]
            affordance_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["AffordanceLogits"][b, pred],
                    targets["affordance"][b, gt],
                    reduction="none").mean(dim=-1),
                affordance_valid))
            relation_target = targets["relation_type"][b][gt.unsqueeze(1), gt.unsqueeze(0)]
            pair_logits = pst["RelationLogitsRaw"][b][pred.unsqueeze(1), pred.unsqueeze(0)]
            pair_valid = (
                targets["relation_valid"][b][gt.unsqueeze(1), gt.unsqueeze(0)]
                & ~torch.eye(pred.numel(), device=pred.device, dtype=torch.bool))
            pair_loss = F.cross_entropy(
                pair_logits.reshape(-1, ModuleDim.PstRelationClasses),
                relation_target.reshape(-1),
                weight=relation_weight,
                reduction="none").view(pred.numel(), pred.numel())
            relation_terms.append(self.MaskedMean(pair_loss, pair_valid))
            external_valid = targets["external_relation_valid"][b, gt]
            external_relation_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["ExternalRelationLogits"][b, pred],
                    targets["external_relation"][b, gt],
                    reduction="none").mean(dim=-1),
                external_valid))
            motion_valid = targets["motion_valid"][b, gt]
            motion_terms.append(self.MaskedMean(
                self.PoseLossEach(pst["MotionPred"][b, pred], targets["motion"][b, gt]),
                motion_valid))
            moving_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["MovingLogits"][b, pred],
                    targets["is_moving"][b, gt],
                    reduction="none"),
                motion_valid))
            contact_valid = targets["contact_valid"][b, gt]
            contact_terms.append(self.MaskedMean(
                F.binary_cross_entropy_with_logits(
                    pst["ContactLogits"][b, pred],
                    targets["contact"][b, gt],
                    reduction="none"),
                contact_valid))
            contacted = contact_valid & targets["contact"][b, gt].bool()
            force_terms.append(self.MaskedMean(
                F.smooth_l1_loss(
                    pst["ContactForcePred"][b, pred],
                    targets["contact_force"][b, gt],
                    reduction="none").mean(dim=-1),
                contacted))
            point_terms.append(self.MaskedMean(
                F.smooth_l1_loss(
                    pst["ContactPointPred"][b, pred],
                    targets["contact_point_observer"][b, gt],
                    reduction="none").mean(dim=-1),
                contacted))

        loss_presence = self.MeanTerms(presence_terms)
        loss_mphys = self.MeanTerms(mphys_terms)
        loss_interaction = self.MeanTerms(interaction_terms)
        loss_realm = self.MeanTerms(realm_terms)
        loss_motion_layer = self.MeanTerms(motion_layer_terms)
        loss_layer_agency = self.MeanTerms(layer_agency_terms)
        loss_causal_agency = self.MeanTerms(causal_agency_terms)
        loss_body = self.MeanTerms(body_terms)
        loss_self_part = self.MeanTerms(self_part_terms)
        loss_carrier = self.MeanTerms(carrier_terms)
        loss_articulation = self.MeanTerms(articulation_terms)
        loss_content_motion = self.MeanTerms(content_motion_terms)
        loss_content_change = self.MeanTerms(content_change_terms)
        loss_display_surface = self.MeanTerms(display_surface_terms)
        loss_surface_parent = self.MeanTerms(surface_parent_terms)
        loss_surface_uv = self.MeanTerms(surface_uv_terms)
        loss_surface_uv_confidence = self.MeanTerms(
            surface_uv_confidence_terms)
        loss_verification = self.MeanTerms(verification_terms)
        loss_ontology_relation = self.MeanTerms(ontology_relation_terms)
        loss_ontology_consistency = self.MeanTerms(consistency_terms)
        loss_state = self.MeanTerms(state_terms)
        loss_attribute = self.MeanTerms(attribute_terms)
        loss_affordance = self.MeanTerms(affordance_terms)
        loss_relation = self.MeanTerms(relation_terms)
        loss_external_relation = self.MeanTerms(external_relation_terms)
        loss_motion = self.MeanTerms(motion_terms)
        loss_moving = self.MeanTerms(moving_terms)
        loss_contact = self.MeanTerms(contact_terms)
        loss_force = self.MeanTerms(force_terms)
        loss_point = self.MeanTerms(point_terms)
        total = (
            self.mphys_weight * (loss_presence + loss_mphys + loss_interaction)
            + self.state_weight * loss_state
            + self.attribute_weight * loss_attribute
            + self.affordance_weight * loss_affordance
            + self.relation_weight * (loss_relation + loss_external_relation)
            + self.motion_weight * (loss_motion + loss_moving)
            + self.contact_weight * (loss_contact + loss_force + loss_point)
            + self.ontology_weight * (
                loss_realm
                + loss_motion_layer
                + loss_layer_agency
                + self.causal_agency_weight * loss_causal_agency
                + loss_body
                + loss_self_part
                + loss_content_change
                + loss_display_surface
                + loss_surface_parent
                + loss_surface_uv
                + loss_surface_uv_confidence
                + loss_verification
                + loss_ontology_relation
                + 0.25 * loss_ontology_consistency)
            + self.factor_motion_weight * (
                loss_carrier + loss_articulation + loss_content_motion))
        return {
            "loss_perceptual_presence": loss_presence,
            "loss_mphys": loss_mphys,
            "loss_physical_interaction": loss_interaction,
            "loss_realm": loss_realm,
            "loss_motion_layer": loss_motion_layer,
            "loss_layer_agency": loss_layer_agency,
            "loss_causal_agency": loss_causal_agency,
            "loss_body_membership": loss_body,
            "loss_self_part": loss_self_part,
            "loss_carrier_motion": loss_carrier,
            "loss_articulation_motion": loss_articulation,
            "loss_content_motion": loss_content_motion,
            "loss_content_change": loss_content_change,
            "loss_display_surface": loss_display_surface,
            "loss_surface_parent": loss_surface_parent,
            "loss_surface_uv": loss_surface_uv,
            "loss_surface_uv_confidence": loss_surface_uv_confidence,
            "loss_verification": loss_verification,
            "loss_ontology_relation": loss_ontology_relation,
            "loss_ontology_consistency": loss_ontology_consistency,
            "loss_state": loss_state,
            "loss_attributes": loss_attribute,
            "loss_affordance": loss_affordance,
            "loss_relation": loss_relation,
            "loss_external_relation": loss_external_relation,
            "loss_motion": loss_motion,
            "loss_moving": loss_moving,
            "loss_contact": loss_contact,
            "loss_contact_force": loss_force,
            "loss_contact_point": loss_point,
            "loss": total}


class PerceptionPhysicalTrainer(nn.Module):
    def __init__(
        self,
        projectionMatrix: torch.Tensor,
        contractView: RobotEmbodimentContractView,
        physicalWeight: float = 1.0,
        recallLossKwargs: Optional[Dict[str, Any]] = None,
        **perceptionKwargs: Any):
        super().__init__()
        perceptionKwargs = dict(perceptionKwargs)
        perceptionKwargs["enableRecallAuxiliary"] = True
        self.perception = PerceiveExtractor(
            projectionMatrix=projectionMatrix,
            **perceptionKwargs)
        self.recall_loss = PerceptionRecallLoss(**({} if recallLossKwargs is None else recallLossKwargs))
        self.physical = PhysicalStateExtractor(
            contractView=contractView,
            inObjectDim=self.perception.embed_dim,
            motionDim=self.perception.embed_dim)
        self.physical_loss = PhysicalStateLoss()
        self.physical_weight = float(physicalWeight)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        topDownContext: TopDownContext,
        targets: Dict[str, torch.Tensor],
        observerRotation: torch.Tensor,
        prevVisualValid: torch.Tensor,
        slotBodyTokens: torch.Tensor,
        slotWeight: torch.Tensor,
        prevVisualState: Optional[VisualState] = None,
        observerRotationValid: Optional[torch.Tensor] = None,
        ) -> Dict[str, Any]:
        visual_state = self.perception(
            rgb,
            topDownContext=topDownContext,
            depth=depth,
            depthValid=depthValid,
            prevVisualState=prevVisualState,
            observerRotation=observerRotation,
            prevVisualValid=prevVisualValid,
            observerRotationValid=observerRotationValid)

        recall_out = self.perception.recall_heads(visual_state)

        representation_loss = self.perception.ComputePerceptionLoss(
            visual_state,
            depthTarget=targets["depth"],
            depthTargetValid=targets["depth_valid"],
            prevVisualState=prevVisualState,
            observerRotation=observerRotation,
            prevVisualValid=prevVisualValid,
            observerRotationValid=observerRotationValid)

        recall_losses = self.recall_loss(recall_out, targets)

        semantic_view = self.physical.SemanticWorldView(visual_state.SemanticNodes)

        node_mask = semantic_view["NodePresence"]

        geometry_valid = visual_state.Auxiliary["ObjectGeometryValid"]

        physical_out = self.physical(
            visual_state.ObjectTokens,
            visual_state.Auxiliary["ObjectMotion"],
            visual_state.Auxiliary["ObjectGeometry"],
            node_mask,
            geometry_valid,
            slotBodyTokens,
            slotWeight)

        pst = {
            **physical_out,
            **semantic_view,
            "ObservedSlotMask": physical_out["ObservationMask"]}

        physical_losses = self.physical_loss(pst, targets)
        perception_loss = representation_loss + recall_losses["loss"]
        total = perception_loss + self.physical_weight * physical_losses["loss"]

        return {
            "visual_state": visual_state,
            "physical_state": pst,
            "recall_out": recall_out,
            "loss_representation": representation_loss,
            "loss_perception": perception_loss,
            "loss_physical": physical_losses["loss"],
            "loss": total,
            "recall_losses": recall_losses,
            "physical_losses": physical_losses}


class PSTWorldBinder(AGICoreModule):
    def __init__(
        self,
        hDim: int = ModuleDim.WorldOutHState,
        zDim: int = ModuleDim.WorldOutZState,
        xDim: int = ModuleDim.WorldOutXState,
        actionDim: int = ModuleDim.DecisionActionFeatureDim,
        embodimentDim: int = ModuleDim.PstSlotDim,
        slotDim: int = ModuleDim.PstSlotDim,
        idDim: int = ModuleDim.PstIdDim,
        poseDim: int = ModuleDim.PstPoseDim,
        attrDim: int = ModuleDim.PstAttrDim,
        semanticDim: int = ModuleDim.PstSemanticDim,
        stateDim: int = ModuleDim.PstStateDim,
        affordanceDim: int = ModuleDim.PstAffordanceDim,
        relDim: int = ModuleDim.PstRelDim,
        relationClasses: int = ModuleDim.PstRelationClasses,
        queryCount: int = 8,
        numHeads: int = 4,
        memoryDecayHorizon: float = 64.0,
        memoryScale: float = 0.5):
        super().__init__()
        self.slot_dim = int(slotDim)
        self.z_dim = int(zDim)
        self.query_count = int(queryCount)
        self.memory_decay_horizon = float(memoryDecayHorizon)
        self.memory_scale = float(memoryScale)

        content_dim = (
            int(slotDim)
            + int(poseDim)
            + int(attrDim)
            + 3
            + int(stateDim)
            + int(affordanceDim)
            + int(poseDim)
            + 1
            + 1
            + 2
            + 3
            + 1
            + 1
            + int(semanticDim)
            + int(relationClasses)
            + int(relDim)
            + ModuleDim.PstRealmClasses
            + ModuleDim.PstAgencyClasses
            + ModuleDim.PstMotionLayerClasses
            + ModuleDim.PstMotionLayerClasses * ModuleDim.PstAgencyClasses
            + 1
            + ModuleDim.PstSelfPartSemanticDim
            + 4
            + 2 * int(poseDim)
            + 2
            + 1
            + 1
            + 2
            + 1
            + 1
            + ModuleDim.PstOntologyRelationClasses
            + 2)
        world_dim = int(hDim) + int(zDim) + int(xDim) + int(actionDim) + int(embodimentDim)

        self.address_proj = nn.Sequential(
            nn.LayerNorm(int(idDim)),
            nn.Linear(int(idDim), self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim))

        self.content_proj = nn.Sequential(
            nn.LayerNorm(content_dim),
            nn.Linear(content_dim, self.slot_dim * 2),
            nn.SiLU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim))

        self.query_proj = nn.Sequential(
            nn.LayerNorm(world_dim),
            nn.Linear(world_dim, self.query_count * self.slot_dim))

        self.query_offset = nn.Parameter(torch.zeros(self.query_count, self.slot_dim))

        self.query_norm = nn.LayerNorm(self.slot_dim)
        self.key_norm = nn.LayerNorm(self.slot_dim)
        self.value_norm = nn.LayerNorm(self.slot_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.slot_dim,
            num_heads=int(numHeads),
            kdim=self.slot_dim,
            vdim=self.slot_dim,
            batch_first=True)
        self.self_attn_norm = nn.LayerNorm(self.slot_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.slot_dim,
            num_heads=int(numHeads),
            batch_first=True)

        self.query_ffn = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.slot_dim * 2),
            nn.SiLU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim))

        self.query_pool = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, 1))

        self.context_post = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim))

        self.delta_mu = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.z_dim))

        self.bind_gate = nn.Sequential(
            nn.LayerNorm(world_dim + self.slot_dim),
            nn.Linear(world_dim + self.slot_dim, self.z_dim))

        self.summary_pred = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, self.slot_dim))

    def SlotBindingWeights(
        self,
        physicalState: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        reference_weights = BuildReferenceWeights(
            physicalState,
            physicalState["Step"].view(-1, 1).float(),
            physicalState["PerceptualPresence"],
            memoryScale=self.memory_scale,
            memoryDecayHorizon=self.memory_decay_horizon)
        observed = physicalState["Observed"]
        observed_weight = torch.where(
            observed,
            reference_weights.observed_weight,
            torch.zeros_like(reference_weights.observed_weight))
        memory_active = ~observed & (physicalState["SlotPresence"] > 0.0)
        memory_weight = torch.where(
            memory_active,
            reference_weights.memory_weight,
            torch.zeros_like(reference_weights.memory_weight))
        return (
            observed_weight + memory_weight,
            observed_weight,
            memory_weight)

    def RelationSummary(self, physicalState: Dict[str, torch.Tensor], slotWeight: torch.Tensor) -> torch.Tensor:
        relation = physicalState["PairwiseRelationObserver"]
        pair_last_seen = physicalState["PairRelationLastSeen"]
        pair_age = (
            physicalState["Step"].view(-1, 1, 1).to(pair_last_seen.dtype)
            - pair_last_seen).clamp_min(0).to(relation.dtype)
        pair_seen = pair_last_seen > 0
        semantic_recency = (
            torch.exp(-pair_age / max(self.memory_decay_horizon, 1e-6))
            * pair_seen.to(relation.dtype))
        relation = torch.cat([
            relation[..., :4],
            relation[..., 4:] * semantic_recency.unsqueeze(-1),], dim=-1)
        K = int(relation.size(1))
        slot_valid = slotWeight > 0.0
        off_diagonal = ~torch.eye(K, device=relation.device, dtype=torch.bool)
        pair_valid = (
            slot_valid.unsqueeze(2)
            & slot_valid.unsqueeze(1)
            & off_diagonal.unsqueeze(0))
        relation = torch.where(
            pair_valid.unsqueeze(-1), relation, torch.zeros_like(relation))
        neighbor_weight = slotWeight.unsqueeze(1) * pair_valid.to(slotWeight.dtype)
        denom = neighbor_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return (relation * neighbor_weight.unsqueeze(-1)).sum(dim=2) / denom

    def OntologyRelationSummary(
        self,
        physicalState: Dict[str, torch.Tensor],
        slotWeight: torch.Tensor,
        ) -> torch.Tensor:
        relation = physicalState["OntologyRelationProb"]
        K = int(relation.size(1))
        slot_valid = slotWeight > 0.0
        off_diagonal = ~torch.eye(K, device=relation.device, dtype=torch.bool)
        pair_valid = (
            slot_valid.unsqueeze(2)
            & slot_valid.unsqueeze(1)
            & off_diagonal.unsqueeze(0))
        neighbor_weight = slotWeight.unsqueeze(1) * pair_valid.to(slotWeight.dtype)
        denom = neighbor_weight.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return (
            relation * neighbor_weight.unsqueeze(-1)
        ).sum(dim=2) / denom

    def WeightedCrossAttention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sourceWeight: torch.Tensor) -> torch.Tensor:
        query = self.query_norm(query)
        key = self.key_norm(key)
        value = self.value_norm(value)
        B, N = sourceWeight.shape
        T = query.size(1)
        H = self.cross_attn.num_heads
        source_visible = sourceWeight > 0.0
        empty_source = ~source_visible.any(dim=1)
        source_visible = source_visible.clone()
        source_visible[empty_source, 0] = True
        log_weight = torch.log(sourceWeight.clamp_min(1e-6))
        log_weight = log_weight.masked_fill(~source_visible, torch.finfo(log_weight.dtype).min)
        attn_mask = log_weight.view(B, 1, 1, N).expand(-1, H, T, -1).reshape(B * H, T, N)
        out, _ = self.cross_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            need_weights=False)
        return out * (~empty_source).to(out.dtype).view(B, 1, 1)

    def forward(
        self,
        worldH: torch.Tensor,
        worldZMu: torch.Tensor,
        worldX: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        embodimentContext: torch.Tensor) -> Dict[str, torch.Tensor]:
        world_context = torch.cat([worldH, worldZMu, worldX, actionEnc, embodimentContext], dim=-1)
        slot_binding_weight, observed_weight, memory_weight = self.SlotBindingWeights(physicalState)
        slot_valid = slot_binding_weight > 0.0
        slot_mask = slot_valid.unsqueeze(-1)
        has_source = slot_valid.any(dim=1, keepdim=True).to(world_context.dtype)
        relation_summary = self.RelationSummary(physicalState, slot_binding_weight)
        ontology_relation_summary = self.OntologyRelationSummary(
            physicalState, slot_binding_weight)
        surface_parent = physicalState["SurfaceParentProb"]
        surface_parent_summary = torch.stack([
            surface_parent[..., :-1].amax(dim=-1),
            surface_parent[..., -1],], dim=-1)
        slot_content = torch.cat([
            physicalState["SlotState"],
            physicalState["SpatialFrame"],
            physicalState["ARaw"],
            physicalState["Size"],
            physicalState["StateRaw"],
            physicalState["AffordanceRaw"],
            physicalState["MotionObserverRaw"],
            physicalState["MovingProbRaw"].unsqueeze(-1),
            physicalState["ContactProbRaw"].unsqueeze(-1),
            physicalState["ContactForceRaw"],
            physicalState["ContactPointObserverRaw"],
            physicalState["Visibility"].unsqueeze(-1),
            physicalState["Occlusion"].unsqueeze(-1),
            physicalState["Semantic"],
            physicalState["ExternalRelationProbRaw"],
            relation_summary,
            physicalState["RealmProb"],
            physicalState["AgencyProb"],
            physicalState["MotionLayerProb"],
            physicalState["LayerAgencyProb"].flatten(-2),
            physicalState["BodyMembershipProb"].unsqueeze(-1),
            physicalState["SelfPartSemantic"],
            physicalState["PerceptualPresence"].unsqueeze(-1),
            physicalState["GeometryValidMask"].unsqueeze(-1),
            physicalState["MphysRaw"].unsqueeze(-1),
            physicalState["PhysicalInteractionProb"].unsqueeze(-1),
            physicalState["CarrierMotionObserverRaw"],
            physicalState["ArticulationMotionObserverRaw"],
            physicalState["ContentMotionUV"],
            physicalState["ContentChangeProb"].unsqueeze(-1),
            physicalState["DisplaySurfaceProb"].unsqueeze(-1),
            physicalState["SurfaceUV"],
            physicalState["SurfaceUVConfidence"].unsqueeze(-1),
            physicalState["VerificationConfidence"].unsqueeze(-1),
            ontology_relation_summary,
            surface_parent_summary], dim=-1)
        slot_content = torch.where(slot_mask, slot_content, torch.zeros_like(slot_content))
        identity_key = torch.where(
            slot_mask,
            physicalState["IdentityKey"],
            torch.zeros_like(physicalState["IdentityKey"]))

        address_key = self.address_proj(identity_key)
        content_value = self.content_proj(slot_content)
        query = self.query_proj(world_context).view(worldH.size(0), self.query_count, self.slot_dim)
        query = query + self.query_offset.unsqueeze(0)
        query = query + self.WeightedCrossAttention(query, address_key, content_value, slot_binding_weight)
        self_out, _ = self.self_attn(
            query=self.self_attn_norm(query),
            key=self.self_attn_norm(query),
            value=self.self_attn_norm(query),
            need_weights=False)
        query = query + self_out
        query = query + self.query_ffn(query)
        query_pool_weight = F.softmax(self.query_pool(query).squeeze(-1), dim=1).unsqueeze(-1)
        pst_context = self.context_post((query * query_pool_weight).sum(dim=1)) * has_source
        bind_gate = (
            torch.sigmoid(self.bind_gate(torch.cat([world_context, pst_context], dim=-1)))
            * has_source)
        delta_mu = bind_gate * self.delta_mu(pst_context)
        bound_mu = worldZMu + delta_mu
        return {
            "bound_mu": bound_mu,
            "delta_mu": delta_mu,
            "bind_gate": bind_gate,
            "pst_context": pst_context,
            "slot_binding_weight": slot_binding_weight,
            "observed_weight": observed_weight,
            "memory_weight": memory_weight,
            "query_pool_weight": query_pool_weight.squeeze(-1),
            "pst_summary_pred": self.summary_pred(pst_context) * has_source}
