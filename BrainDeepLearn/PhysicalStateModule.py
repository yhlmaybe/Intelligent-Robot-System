from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import AGICoreModule, HungarianAssignment, BuildReferenceWeights
from ModuleMessagerManager import ModuleDim
from PerceptionModule import PerceiveExtractor, PerceptionRecallLoss, TopDownContext, VisualState
from RobotMorphologyModule import (
    CompiledRobotMorphology,
    EntityAgency,
    EntityRealm,
    MotionLayer)

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
    def BuildSelfPartNodeDescriptors(
        robotMorphology: CompiledRobotMorphology,
        ) -> torch.Tensor:
        node_count = int(robotMorphology.node_count)
        semantic = robotMorphology.NodeSemanticDescriptor()

        def Tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(
                semantic[name],
                dtype=dtype).detach().cpu()

        role = Tensor("role", torch.long).reshape(node_count)
        side = Tensor("side", torch.long).reshape(node_count)
        capability = Tensor("capability", torch.float32).reshape(
            node_count,
            ModuleDim.RobotBodyCapabilityDim)
        parent_index = Tensor(
            "parent_node_index", torch.long).reshape(node_count)
        parent_role = Tensor("parent_role", torch.long).reshape(node_count)
        parent_side = Tensor("parent_side", torch.long).reshape(node_count)
        parent_capability = Tensor(
            "parent_capability", torch.float32).reshape(
                node_count,
                ModuleDim.RobotBodyCapabilityDim)
        group_role = Tensor(
            "group_role_membership", torch.float32).reshape(
                node_count,
                ModuleDim.RobotBodyRoleClasses)
        group_side = Tensor(
            "group_side_membership", torch.float32).reshape(
                node_count,
                ModuleDim.RobotBodySideClasses)
        group_capability = Tensor(
            "group_capability", torch.float32).reshape(
                node_count,
                ModuleDim.RobotBodyCapabilityDim)
        topology_depth = Tensor(
            "topology_depth", torch.float32).reshape(node_count, 1)
        in_degree = Tensor("in_degree", torch.float32).reshape(node_count, 1)
        out_degree = Tensor(
            "out_degree", torch.float32).reshape(node_count, 1)
        is_root = Tensor("is_root", torch.float32).reshape(node_count, 1)
        is_leaf = Tensor("is_leaf", torch.float32).reshape(node_count, 1)
        parent_present = parent_index.ge(0).to(torch.float32).unsqueeze(-1)
        parent_role_feature = F.one_hot(
            parent_role.clamp(0, ModuleDim.RobotBodyRoleClasses - 1),
            num_classes=ModuleDim.RobotBodyRoleClasses).to(torch.float32)
        parent_side_feature = F.one_hot(
            parent_side.clamp(0, ModuleDim.RobotBodySideClasses - 1),
            num_classes=ModuleDim.RobotBodySideClasses).to(torch.float32)
        parent_role_feature *= parent_present
        parent_side_feature *= parent_present
        parent_capability *= parent_present
        descriptor = torch.cat([
            F.one_hot(
                role,
                num_classes=ModuleDim.RobotBodyRoleClasses).to(torch.float32),
            F.one_hot(
                side,
                num_classes=ModuleDim.RobotBodySideClasses).to(torch.float32),
            capability,
            parent_role_feature,
            parent_side_feature,
            parent_capability,
            group_role,
            group_side,
            group_capability,
            topology_depth / (1.0 + topology_depth),
            in_degree / (1.0 + in_degree),
            out_degree / (1.0 + out_degree),
            parent_present,
            is_root,
            is_leaf,], dim=-1)
        return descriptor

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
            "PoseCamera": nodes["pose_camera"],
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
        robotMorphology: CompiledRobotMorphology,
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
        node_descriptor = self.BuildSelfPartNodeDescriptors(robotMorphology)
        self.self_part_count = int(robotMorphology.node_count)
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
        self.self_part_node_encoder = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
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
        self.self_part_semantic_proj = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(
                self.slot_dim,
                self.self_part_semantic_dim))
        self.self_part_geometry_log_scale = nn.Parameter(torch.tensor(0.0))
        self.self_part_geometry_gain = nn.Parameter(torch.tensor(-2.944439))

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

    def forward(
        self,
        objectTokens: torch.Tensor, # [B, K, D_obj]
        objectMotion: torch.Tensor, # [B, K, D_motion]
        objectGeometry: torch.Tensor, # [B, K, 7]
        nodeMask: torch.Tensor,
        geometryValid: torch.Tensor,
        bodyNodePoseCamera: Optional[torch.Tensor] = None,
        bodyNodeObserved: Optional[torch.Tensor] = None,
        bodyNodeHealthy: Optional[torch.Tensor] = None,
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

        # Keep an unmasked training feature so a missed object can still receive
        # positive supervision.  Runtime state remains masked at the output seam.
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
            realm_class.ne(int(EntityRealm.VIRTUAL_CONTENT))
            & realm_class.ne(int(EntityRealm.VISUAL_EFFECT))
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
            * realm_prob[..., int(EntityRealm.SELF_BODY)]
            * perceptual_presence)
        node_descriptor = self.self_part_node_descriptor.to(
            dtype=S_feature.dtype)
        node_key = self.self_part_node_key(
            self.self_part_node_encoder(node_descriptor))
        slot_query = self.self_part_slot_query(S_feature)
        raw_self_part_logits = torch.einsum(
            "bkd,nd->bkn",
            slot_query,
            node_key) * (float(self.slot_dim) ** -0.5)
        body_state = (
            bodyNodePoseCamera,
            bodyNodeObserved,
            bodyNodeHealthy)
        if any(value is not None for value in body_state):
            if any(value is None for value in body_state):
                raise ValueError("body node state is incomplete")
            B = int(objectTokens.size(0))
            expected_pose = (B, self.self_part_count, 7)
            expected_mask = (B, self.self_part_count)
            if tuple(bodyNodePoseCamera.shape) != expected_pose:
                raise ValueError("bodyNodePoseCamera has invalid shape")
            if tuple(bodyNodeObserved.shape) != expected_mask:
                raise ValueError("bodyNodeObserved has invalid shape")
            if tuple(bodyNodeHealthy.shape) != expected_mask:
                raise ValueError("bodyNodeHealthy has invalid shape")
            body_pose = bodyNodePoseCamera.to(
                device=objectTokens.device,
                dtype=objectTokens.dtype)
            body_available = (
                bodyNodeObserved.to(
                    device=objectTokens.device,
                    dtype=torch.bool)
                & bodyNodeHealthy.to(
                    device=objectTokens.device,
                    dtype=torch.bool)
                & torch.isfinite(body_pose).all(dim=-1))
            body_position = torch.where(
                body_available.unsqueeze(-1),
                body_pose[..., :3],
                torch.zeros_like(body_pose[..., :3]))
            object_available = (
                geometry_mask.bool()
                & torch.isfinite(objectGeometry[..., :3]).all(dim=-1))
            object_position = torch.where(
                object_available.unsqueeze(-1),
                objectGeometry[..., :3],
                torch.zeros_like(objectGeometry[..., :3]))
            distance = torch.linalg.vector_norm(
                object_position.unsqueeze(2)
                - body_position.unsqueeze(1),
                dim=-1)
            scale = F.softplus(self.self_part_geometry_log_scale) + 0.05
            gain = 4.0 * torch.sigmoid(self.self_part_geometry_gain)
            geometry_evidence = -gain * torch.tanh(distance / scale)
            evidence_valid = (
                object_available.unsqueeze(-1)
                & body_available.unsqueeze(1))
            raw_self_part_logits = raw_self_part_logits + torch.where(
                evidence_valid,
                geometry_evidence,
                torch.zeros_like(geometry_evidence))
        self_part_logits = raw_self_part_logits
        self_part_prob = (
            F.softmax(self_part_logits, dim=-1)
            * body_membership_prob.unsqueeze(-1))
        weighted_self_part_descriptor = torch.einsum(
            "bkn,nd->bkd",
            self_part_prob,
            node_descriptor)
        self_part_mass = self_part_prob.sum(dim=-1, keepdim=True)
        self_part_semantic = (
            self.self_part_semantic_proj(weighted_self_part_descriptor)
            * self_part_mass)

        carrier_motion_pred = self.NormalizePose(
            self.carrier_motion_head(S_feature))
        articulation_motion_pred = self.NormalizePose(
            self.articulation_motion_head(S_feature))
        carrier_weight = (
            geometry_available
            * motion_layer_prob[..., int(MotionLayer.CARRIER_MOTION)].unsqueeze(-1))
        articulation_weight = (
            geometry_available
            * motion_layer_prob[..., int(MotionLayer.ARTICULATION_MOTION)].unsqueeze(-1))
        carrier_motion_raw = self.NormalizePose(
            identity_motion
            + (carrier_motion_pred - identity_motion) * carrier_weight)
        articulation_motion_raw = self.NormalizePose(
            identity_motion
            + (articulation_motion_pred - identity_motion) * articulation_weight)

        content_motion_pred = torch.tanh(self.content_motion_head(S_feature))
        content_layer = motion_layer_prob[
            ..., int(MotionLayer.SURFACE_CONTENT_MOTION)]
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
            "MotionCameraRaw": motion_raw,
            "MovingLogits": moving_logits,
            "MovingProbRaw": moving_raw,
            "ContactLogits": contact_logits,
            "ContactProbRaw": contact_prob_raw,
            "ContactForcePred": contact_force_pred,
            "ContactForceRaw": contact_force_raw,
            "ContactPointPred": contact_point_pred,
            "ContactPointCameraRaw": contact_point_raw,
            "PairwiseRelationCamera": pairwise_relation,
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
            "CarrierMotionPred": carrier_motion_pred,
            "CarrierMotionCameraRaw": carrier_motion_raw,
            "ArticulationMotionPred": articulation_motion_pred,
            "ArticulationMotionCameraRaw": articulation_motion_raw,
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
        factorMotionWeight: float = 1.0,):
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
            pst["PoseCamera"][b, candidate],
            targets["pose_camera"][b, gt])
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

    def forward(self, pst: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        presence_terms: List[torch.Tensor] = []
        mphys_terms: List[torch.Tensor] = []
        interaction_terms: List[torch.Tensor] = []
        realm_terms: List[torch.Tensor] = []
        motion_layer_terms: List[torch.Tensor] = []
        layer_agency_terms: List[torch.Tensor] = []
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
                realm_prob[..., int(EntityRealm.VIRTUAL_CONTENT)]
                + realm_prob[..., int(EntityRealm.VISUAL_EFFECT)])
            ontology_consistency = (
                physical_prob * virtual_or_effect
                + interaction_prob * (1.0 - physical_prob)
                + body_prob * (
                    1.0 - realm_prob[..., int(EntityRealm.SELF_BODY)]))
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
                    targets["contact_point_camera"][b, gt],
                    reduction="none").mean(dim=-1),
                contacted))

        loss_presence = self.MeanTerms(presence_terms)
        loss_mphys = self.MeanTerms(mphys_terms)
        loss_interaction = self.MeanTerms(interaction_terms)
        loss_realm = self.MeanTerms(realm_terms)
        loss_motion_layer = self.MeanTerms(motion_layer_terms)
        loss_layer_agency = self.MeanTerms(layer_agency_terms)
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
        cameraIntrinsics: torch.Tensor,
        robotMorphology: CompiledRobotMorphology,
        physicalWeight: float = 1.0,
        recallLossKwargs: Optional[Dict[str, Any]] = None,
        **perceptionKwargs: Any):
        super().__init__()
        perceptionKwargs = dict(perceptionKwargs)
        perceptionKwargs["enableRecallAuxiliary"] = True
        self.perception = PerceiveExtractor(
            cameraIntrinsics=cameraIntrinsics,
            **perceptionKwargs)
        self.recall_loss = PerceptionRecallLoss(**({} if recallLossKwargs is None else recallLossKwargs))
        self.physical = PhysicalStateExtractor(
            robotMorphology=robotMorphology,
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
        cameraMotion: torch.Tensor,
        prevVisualValid: torch.Tensor,
        prevVisualState: Optional[VisualState] = None) -> Dict[str, Any]:
        visual_state = self.perception(
            rgb,
            topDownContext=topDownContext,
            depth=depth,
            depthValid=depthValid,
            prevVisualState=prevVisualState,
            cameraMotion=cameraMotion,
            prevVisualValid=prevVisualValid)
        
        recall_out = self.perception.recall_heads(visual_state)

        representation_loss = self.perception.ComputePerceptionLoss(
            visual_state,
            depthTarget=targets["depth"],
            depthTargetValid=targets["depth_valid"],
            prevVisualState=prevVisualState,
            cameraMotion=cameraMotion,
            prevVisualValid=prevVisualValid)
        
        recall_losses = self.recall_loss(recall_out, targets)

        semantic_view = self.physical.SemanticWorldView(visual_state.SemanticNodes)

        node_mask = semantic_view["NodePresence"]

        geometry_valid = visual_state.Auxiliary["ObjectGeometryValid"]

        physical_out = self.physical(
            visual_state.ObjectTokens,
            visual_state.Auxiliary["ObjectMotion"],
            visual_state.Auxiliary["ObjectGeometry"],
            node_mask,
            geometry_valid)
        
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
        actionDim: int = 256,
        robotWorldDim: int = ModuleDim.PstSlotDim,
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
        world_dim = int(hDim) + int(zDim) + int(xDim) + int(actionDim) + int(robotWorldDim)

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
        relation = physicalState["PairwiseRelationCamera"]
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
        robotWorldContext: torch.Tensor) -> Dict[str, torch.Tensor]:
        world_context = torch.cat([worldH, worldZMu, worldX, actionEnc, robotWorldContext], dim=-1)
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
            physicalState["PoseCamera"],
            physicalState["ARaw"],
            physicalState["Size"],
            physicalState["StateRaw"],
            physicalState["AffordanceRaw"],
            physicalState["MotionCameraRaw"],
            physicalState["MovingProbRaw"].unsqueeze(-1),
            physicalState["ContactProbRaw"].unsqueeze(-1),
            physicalState["ContactForceRaw"],
            physicalState["ContactPointCameraRaw"],
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
            physicalState["CarrierMotionCameraRaw"],
            physicalState["ArticulationMotionCameraRaw"],
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


class TestPhysicalStateMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def AssertFinite(self, value: torch.Tensor, name: str) -> None:
        assert torch.isfinite(value).all(), f"{name} contains non-finite values"

    def MakeRobotMorphology(
        self,
        nodeCount: int = 4,
        permutation: Optional[torch.Tensor] = None,
        ) -> Any:
        node_count = int(nodeCount)
        if node_count < 1:
            raise ValueError("nodeCount must be positive")
        node_role = torch.empty(node_count, dtype=torch.long)
        node_side = torch.empty(node_count, dtype=torch.long)
        node_capability = torch.zeros(
            node_count,
            ModuleDim.RobotBodyCapabilityDim,
            dtype=torch.bool)
        parent_index = torch.full((node_count,), -1, dtype=torch.long)
        index = torch.arange(node_count)
        node_role[:] = index % ModuleDim.RobotBodyRoleClasses
        node_side[:] = index * 3 % ModuleDim.RobotBodySideClasses
        node_capability[
            index,
            index % ModuleDim.RobotBodyCapabilityDim] = True
        if node_count > 1:
            parent_index[1:] = torch.arange(node_count - 1) // 2

        if permutation is not None:
            order = torch.as_tensor(
                permutation,
                dtype=torch.long).reshape(-1)
            if order.numel() != node_count:
                raise ValueError("permutation has invalid length")
            if not torch.equal(
                torch.sort(order).values,
                torch.arange(node_count)):
                raise ValueError("permutation is invalid")
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(node_count)
            old_parent = parent_index.clone()[order]
            node_role = node_role.clone()[order]
            node_side = node_side.clone()[order]
            node_capability = node_capability.clone()[order]
            parent_index = torch.where(
                old_parent.ge(0),
                inverse[old_parent.clamp_min(0)],
                old_parent)

        def NodeSemanticDescriptor() -> Dict[str, torch.Tensor]:
            has_parent = parent_index.ge(0)
            parent_role = torch.full_like(node_role, -1)
            parent_side = torch.full_like(node_side, -1)
            parent_capability = torch.zeros_like(node_capability)
            parent_role[has_parent] = node_role[parent_index[has_parent]]
            parent_side[has_parent] = node_side[parent_index[has_parent]]
            parent_capability[has_parent] = node_capability[
                parent_index[has_parent]]
            topology_depth = torch.zeros(node_count, dtype=torch.long)
            out_degree = torch.zeros(node_count, dtype=torch.long)
            for node_index in range(node_count):
                current = node_index
                while int(parent_index[current].item()) >= 0:
                    current = int(parent_index[current].item())
                    topology_depth[node_index] += 1
            if bool(has_parent.any().item()):
                out_degree.scatter_add_(
                    0,
                    parent_index[has_parent],
                    torch.ones_like(parent_index[has_parent]))
            return {
                "parent_node_index": parent_index,
                "topology_depth": topology_depth,
                "is_root": ~has_parent,
                "is_leaf": out_degree.eq(0),
                "in_degree": has_parent.to(torch.long),
                "out_degree": out_degree,
                "role": node_role,
                "side": node_side,
                "capability": node_capability,
                "parent_role": parent_role,
                "parent_side": parent_side,
                "parent_capability": parent_capability,
                "group_role_membership": torch.zeros(
                    node_count,
                    ModuleDim.RobotBodyRoleClasses,
                    dtype=torch.bool),
                "group_side_membership": torch.zeros(
                    node_count,
                    ModuleDim.RobotBodySideClasses,
                    dtype=torch.bool),
                "group_capability": torch.zeros_like(node_capability),}

        return SimpleNamespace(
            node_count=node_count,
            parent_index=parent_index,
            node_role=node_role,
            node_side=node_side,
            node_capability=node_capability,
            NodeSemanticDescriptor=NodeSemanticDescriptor)

    def MakeTopDownContext(self, model, B: int, dtype: torch.dtype = torch.float32) -> TopDownContext:
        runtime = model.base if hasattr(model, "base") else model
        return TopDownContext(
            PredictedVisual=None,
            Precision=torch.ones(B, device=self.device, dtype=dtype),
            MemoryCue=torch.zeros(B, int(runtime.integrated_dim), device=self.device, dtype=dtype),)

    def MakeCameraIntrinsics(self, imageSize: int) -> torch.Tensor:
        focal_length = 0.75 * float(imageSize)
        principal_point = 0.5 * (float(imageSize) - 1.0)
        return torch.tensor([
            [focal_length, 0.0, principal_point],
            [0.0, focal_length, principal_point],
            [0.0, 0.0, 1.0],
        ], device=self.device)

    def MakeExtractorInputs(
        self,
        B: int = 2,
        K: int = 4,
        tokenDim: int = 32) -> Dict[str, torch.Tensor]:
        object_tokens = torch.randn(B, K, tokenDim, device=self.device)
        object_motion = torch.randn(B, K, tokenDim, device=self.device)
        object_geometry = torch.randn(B, K, 7, device=self.device)
        object_geometry[..., 6] = 1.0
        node_mask = torch.ones(B, K, device=self.device)
        geometry_valid = torch.ones(B, K, 1, device=self.device)
        return {
            "object_tokens": object_tokens,
            "object_motion": object_motion,
            "object_geometry": object_geometry,
            "node_mask": node_mask,
            "geometry_valid": geometry_valid,}

    def MakeSemanticNodes(self, B: int = 2, K: int = 4) -> Dict[str, torch.Tensor]:
        parent_logits = torch.randn(B, K, K, device=self.device)
        eye = torch.eye(K, device=self.device, dtype=torch.bool)
        parent_logits = parent_logits.masked_fill(eye.unsqueeze(0), torch.finfo(parent_logits.dtype).min)
        pose = torch.randn(B, K, ModuleDim.PstPoseDim, device=self.device) * 0.1
        pose[..., 2] = 1.0
        pose[..., 6] = 1.0
        return {
            "node_logits": torch.randn(B, K, 2, device=self.device),
            "level_logits": torch.randn(B, K, 3, device=self.device),
            "object_class_logits": torch.randn(B, K, ModuleDim.PstObjectClasses, device=self.device),
            "part_class_logits": torch.randn(B, K, ModuleDim.PstPartClasses, device=self.device),
            "parent_logits": parent_logits,
            "pose_camera": pose,
            "size_3d": torch.rand(B, K, 3, device=self.device).clamp_min(0.05),
            "bbox_2d": torch.rand(B, K, 4, device=self.device),
            "visible_ratio": torch.rand(B, K, device=self.device),
            "occlusion_ratio": torch.rand(B, K, device=self.device) * 0.2,
            "has_text_logits": torch.randn(B, K, 2, device=self.device),
            "text_embed": F.normalize(torch.randn(B, K, ModuleDim.PstTextDim, device=self.device), dim=-1, eps=1e-6),
            "symbol_logits": torch.randn(B, K, ModuleDim.PstSymbolClasses, device=self.device),
            "identity_embed": F.normalize(torch.randn(B, K, ModuleDim.PstIdentityDim, device=self.device), dim=-1, eps=1e-6),}

    def MakeTargets(
        self,
        B: int = 2,
        K: int = 4,
        H: int = 32,
        W: int = 32) -> Dict[str, torch.Tensor]:
        pose = torch.zeros(B, K, ModuleDim.PstPoseDim, device=self.device)
        pose[..., 2] = 1.0
        pose[..., 6] = 1.0
        valid = torch.ones(B, K, device=self.device, dtype=torch.bool)
        level = torch.zeros(B, K, device=self.device, dtype=torch.long)
        parent = torch.full((B, K), -1, device=self.device, dtype=torch.long)
        object_class = torch.ones(B, K, device=self.device, dtype=torch.long)
        part_class = torch.zeros(B, K, device=self.device, dtype=torch.long)
        if K > 1:
            level[:, 1:] = 1
            parent[:, 1:] = 0
            object_class[:, 1:] = 0
            part_class[:, 1:] = 1
        relation = torch.zeros(B, K, K, device=self.device, dtype=torch.long)
        relation_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        relation_valid = relation_valid & ~torch.eye(K, device=self.device, dtype=torch.bool).unsqueeze(0)
        masks = torch.zeros(B, K, H, W, device=self.device, dtype=torch.bool)
        masks[:, 0, : H // 2, : W // 2] = True
        if K > 1:
            masks[:, 1:, H // 2:, W // 2:] = True
        depth = torch.ones(B, 1, H, W, device=self.device)
        normal = torch.zeros(B, 3, H, W, device=self.device)
        normal[:, 2] = 1.0
        realm = torch.full(
            (B, K),
            int(EntityRealm.EXTERNAL_PHYSICAL),
            device=self.device,
            dtype=torch.long)
        motion_layer = torch.zeros(
            B, K, ModuleDim.PstMotionLayerClasses, device=self.device)
        motion_layer[..., int(MotionLayer.CARRIER_MOTION)] = 1.0
        agency_by_layer = torch.full(
            (B, K, ModuleDim.PstMotionLayerClasses),
            int(EntityAgency.UNKNOWN),
            device=self.device,
            dtype=torch.long)
        agency_valid = torch.zeros_like(agency_by_layer, dtype=torch.bool)
        agency_valid[..., int(MotionLayer.CARRIER_MOTION)] = True
        return {
            "rgb": torch.rand(B, 3, H, W, device=self.device),
            "depth": depth,
            "depth_valid": torch.ones_like(depth, dtype=torch.bool),
            "normal": normal,
            "normal_valid": torch.ones_like(depth, dtype=torch.bool),
            "semantic_segmentation": torch.zeros(B, H, W, device=self.device, dtype=torch.long),
            "scene_class": torch.ones(B, device=self.device, dtype=torch.long),
            "global_labels": torch.ones(B, ModuleDim.PstGlobalLabels, device=self.device),
            "node_valid": valid,
            "node_level": level,
            "parent_index": parent,
            "object_classes": object_class,
            "part_classes": part_class,
            "track_id": torch.arange(K, device=self.device).unsqueeze(0).expand(B, -1),
            "pose_camera": pose,
            "pose_world": pose.clone(),
            "pose_valid": valid,
            "geometry_valid": valid,
            "size_3d": torch.ones(B, K, 3, device=self.device) * 0.1,
            "bbox_2d": torch.ones(B, K, 4, device=self.device) * 0.25,
            "node_instance_masks": masks,
            "visible_ratio": torch.ones(B, K, device=self.device),
            "occlusion_ratio": torch.zeros(B, K, device=self.device),
            "has_text": torch.zeros(B, K, device=self.device, dtype=torch.long),
            "text_embed": torch.zeros(B, K, ModuleDim.PstTextDim, device=self.device),
            "symbol_type": torch.zeros(B, K, device=self.device, dtype=torch.long),
            "node_state": torch.zeros(B, K, ModuleDim.PstStateDim, device=self.device),
            "node_state_valid": valid,
            "node_attributes": torch.zeros(B, K, ModuleDim.PstAttrDim, device=self.device),
            "node_attributes_valid": level == 0,
            "relation_type": relation,
            "relation_valid": relation_valid,
            "external_relation": torch.zeros(B, K, ModuleDim.PstRelationClasses, device=self.device),
            "external_relation_valid": valid,
            "motion": pose,
            "motion_valid": valid,
            "is_moving": torch.zeros(B, K, device=self.device),
            "affordance": torch.zeros(B, K, ModuleDim.PstAffordanceDim, device=self.device),
            "affordance_valid": level == 0,
            "contact": torch.zeros(B, K, device=self.device),
            "contact_valid": valid,
            "contact_force": torch.zeros(B, K, 2, device=self.device),
            "contact_point_camera": torch.zeros(B, K, 3, device=self.device),
            "physical_entity": torch.ones(B, K, device=self.device),
            "physical_entity_valid": valid,
            "physical_interaction": torch.ones(B, K, device=self.device),
            "physical_interaction_valid": valid,
            "realm": realm,
            "realm_valid": valid,
            "motion_layer_multi_hot": motion_layer,
            "motion_layer_valid": valid,
            "agency_by_layer": agency_by_layer,
            "agency_by_layer_valid": agency_valid,
            "body_membership": torch.zeros(B, K, device=self.device),
            "body_membership_valid": valid,
            "self_part_id": torch.zeros(B, K, device=self.device, dtype=torch.long),
            "self_part_valid": torch.zeros_like(valid),
            "carrier_motion": pose.clone(),
            "carrier_motion_valid": valid,
            "articulation_motion": pose.clone(),
            "articulation_motion_valid": torch.zeros_like(valid),
            "content_motion_uv": torch.zeros(B, K, 2, device=self.device),
            "content_motion_uv_valid": torch.zeros_like(valid),
            "content_change": torch.zeros(B, K, device=self.device),
            "content_change_valid": valid,
            "display_surface": torch.zeros(B, K, device=self.device),
            "display_surface_valid": valid,
            "surface_parent_index": torch.full(
                (B, K), -1, device=self.device, dtype=torch.long),
            "surface_parent_valid": valid,
            "surface_uv": torch.zeros(B, K, 2, device=self.device),
            "surface_uv_valid": torch.zeros_like(valid),
            "verification_confidence": torch.ones(B, K, device=self.device),
            "verification_confidence_valid": valid,
            "ontology_relation_multi_hot": torch.zeros(
                B, K, K, ModuleDim.PstOntologyRelationClasses,
                device=self.device),
            "ontology_relation_valid": relation_valid,}

    def MakePhysicalState(
        self,
        B: int = 2,
        K: int = 4,
        tokenDim: int = 32,
        robotMorphology: Optional[Any] = None,
        ) -> Tuple[PhysicalStateExtractor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        model = PhysicalStateExtractor(
            robotMorphology=(
                self.MakeRobotMorphology()
                if robotMorphology is None
                else robotMorphology),
            inObjectDim=tokenDim,
            motionDim=tokenDim,
            numSlotLayers=1,
            numHeads=4).to(self.device)
        inputs = self.MakeExtractorInputs(B=B, K=K, tokenDim=tokenDim)
        out = model(
            inputs["object_tokens"],
            inputs["object_motion"],
            inputs["object_geometry"],
            inputs["node_mask"],
            inputs["geometry_valid"])
        semantic = model.SemanticWorldView(self.MakeSemanticNodes(B=B, K=K))
        pst = {**out, **semantic, "ObservedSlotMask": out["ObservationMask"]}
        pst["Observed"] = torch.ones(B, K, device=self.device, dtype=torch.bool)
        pst["Step"] = torch.full((B,), 8, device=self.device, dtype=torch.long)
        pst["LastSeen"] = torch.arange(K, device=self.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        pst["PairRelationLastSeen"] = torch.zeros(
            B, K, K, device=self.device, dtype=torch.long)
        pst["SlotPresence"] = pst["ObservedSlotMask"]
        return model, pst, self.MakeTargets(B=B, K=K)

    def TestSlotCrossAttention(self) -> bool:
        try:
            B, T, N, D = 2, 3, 5, 16
            layer = SlotCrossAttention(D, D, numHeads=4).to(self.device)
            query = torch.randn(B, T, D, device=self.device, requires_grad=True)
            source = torch.randn(B, N, D, device=self.device)
            weight = torch.rand(B, N, device=self.device)
            weight[0].zero_()
            out = layer(query, source, weight)
            assert tuple(out.shape) == (B, T, D), f"unexpected output shape {tuple(out.shape)}"
            self.AssertFinite(out, "SlotCrossAttention output")
            out.square().mean().backward()
            assert query.grad is not None, "query gradient was not produced"
            print("SlotCrossAttention test passed.")
            return True
        except Exception as e:
            print(f"SlotCrossAttention test failed: {type(e).__name__}: {e}")
            return False

    def TestSlotRefineLayer(self) -> bool:
        try:
            B, K, D = 2, 5, 16
            layer = SlotRefineLayer(D, numHeads=4).to(self.device)
            slots = torch.randn(B, K, D, device=self.device)
            mask = torch.ones(B, K, device=self.device)
            mask[0].zero_()
            out = layer(slots, mask)
            assert tuple(out.shape) == (B, K, D), f"unexpected output shape {tuple(out.shape)}"
            self.AssertFinite(out, "SlotRefineLayer output")
            print("SlotRefineLayer test passed.")
            return True
        except Exception as e:
            print(f"SlotRefineLayer test failed: {type(e).__name__}: {e}")
            return False

    def TestPhysicalStateExtractorForwardShapes(self) -> bool:
        try:
            B, K, token_dim = 2, 4, 32
            model = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device)
            inputs = self.MakeExtractorInputs(B=B, K=K, tokenDim=token_dim)
            inputs["node_mask"][:, -1] = 0.0
            out = model(
                inputs["object_tokens"],
                inputs["object_motion"],
                inputs["object_geometry"],
                inputs["node_mask"],
                inputs["geometry_valid"])
            assert tuple(out["SlotState"].shape) == (B, K, ModuleDim.PstSlotDim)
            assert tuple(out["PairwiseRelationCamera"].shape) == (B, K, K, ModuleDim.PstRelDim)
            assert tuple(out["RelationLogitsRaw"].shape) == (B, K, K, ModuleDim.PstRelationClasses)
            assert tuple(out["SelfPartLogits"].shape) == (B, K, 4)
            assert tuple(out["SelfPartProb"].shape) == (B, K, 4)
            assert tuple(out["SelfPartSemantic"].shape) == (
                B,
                K,
                ModuleDim.PstSelfPartSemanticDim)
            assert bool((out["ObservationMask"][:, -1] == 0.0).all().item())
            quat_norm = out["MotionCameraRaw"][..., 3:7].norm(dim=-1)
            assert bool(torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-4))
            assert bool((out["SlotState"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["ARaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["StateRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["AffordanceRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["ContactForceRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["ContactPointCameraRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["MotionCameraRaw"][:, -1, :3].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["MotionCameraRaw"][:, -1, 3:6].abs().sum(dim=-1) == 0.0).all().item())
            assert bool(torch.allclose(out["MotionCameraRaw"][:, -1, 6], torch.ones(B, device=self.device), atol=1e-5))
            for name, value in out.items():
                self.AssertFinite(value, f"PhysicalStateExtractor {name}")
            print("PhysicalStateExtractor forward shape test passed.")
            return True
        except Exception as e:
            print(f"PhysicalStateExtractor forward shape test failed: {type(e).__name__}: {e}")
            return False

    def TestSelfPartMorphologyPermutation(self) -> bool:
        try:
            B, K, token_dim, node_count = 2, 3, 32, 5
            order = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
            reference = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(
                    nodeCount=node_count),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device).eval()
            permuted = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(
                    nodeCount=node_count,
                    permutation=order),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device).eval()
            state = reference.state_dict()
            assert "self_part_node_descriptor" not in state
            permuted.load_state_dict(state)
            inputs = self.MakeExtractorInputs(B=B, K=K, tokenDim=token_dim)
            with torch.no_grad():
                reference_out = reference(
                    inputs["object_tokens"],
                    inputs["object_motion"],
                    inputs["object_geometry"],
                    inputs["node_mask"],
                    inputs["geometry_valid"])
                permuted_out = permuted(
                    inputs["object_tokens"],
                    inputs["object_motion"],
                    inputs["object_geometry"],
                    inputs["node_mask"],
                    inputs["geometry_valid"])
            full_order = order.to(self.device)
            assert torch.allclose(
                permuted_out["SelfPartLogits"],
                reference_out["SelfPartLogits"].index_select(
                    -1,
                    full_order),
                atol=1e-6,
                rtol=1e-6)
            assert torch.allclose(
                permuted_out["SelfPartProb"],
                reference_out["SelfPartProb"].index_select(
                    -1,
                    full_order),
                atol=1e-6,
                rtol=1e-6)
            assert torch.allclose(
                permuted_out["SelfPartSemantic"],
                reference_out["SelfPartSemantic"],
                atol=1e-6,
                rtol=1e-6)
            print("SelfPart morphology permutation test passed.")
            return True
        except Exception as e:
            print(
                "SelfPart morphology permutation test failed: "
                f"{type(e).__name__}: {e}")
            return False

    def TestSelfPartDynamicCardinality(self) -> bool:
        try:
            B, K, token_dim = 2, 3, 32
            small = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(
                    nodeCount=4),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device).eval()
            large = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(
                    nodeCount=7),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device).eval()
            large.load_state_dict(small.state_dict())
            inputs = self.MakeExtractorInputs(B=B, K=K, tokenDim=token_dim)
            with torch.no_grad():
                small_out = small(
                    inputs["object_tokens"],
                    inputs["object_motion"],
                    inputs["object_geometry"],
                    inputs["node_mask"],
                    inputs["geometry_valid"])
                large_out = large(
                    inputs["object_tokens"],
                    inputs["object_motion"],
                    inputs["object_geometry"],
                    inputs["node_mask"],
                    inputs["geometry_valid"])
            assert tuple(small_out["SelfPartLogits"].shape) == (B, K, 4)
            assert tuple(large_out["SelfPartLogits"].shape) == (B, K, 7)
            assert tuple(small_out["SelfPartSemantic"].shape) == (
                B, K, ModuleDim.PstSelfPartSemanticDim)
            assert tuple(large_out["SelfPartSemantic"].shape) == (
                B, K, ModuleDim.PstSelfPartSemanticDim)
            print("SelfPart dynamic cardinality test passed.")
            return True
        except Exception as e:
            print(
                "SelfPart dynamic cardinality test failed: "
                f"{type(e).__name__}: {e}")
            return False

    def TestSelfPartGeometryEvidence(self) -> bool:
        try:
            B, K, L, token_dim = 1, 2, 4, 32
            model = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(nodeCount=L),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device).eval()
            inputs = self.MakeExtractorInputs(B=B, K=K, tokenDim=token_dim)
            inputs["object_geometry"][0, 0, :3] = 0.0
            body_pose = torch.zeros(B, L, 7, device=self.device)
            body_pose[..., 6] = 1.0
            body_pose[0, :, 0] = torch.arange(
                L,
                device=self.device,
                dtype=body_pose.dtype)
            body_observed = torch.ones(
                B, L, device=self.device, dtype=torch.bool)
            body_healthy = torch.ones_like(body_observed)
            with torch.no_grad():
                baseline = model(
                    inputs["object_tokens"],
                    inputs["object_motion"],
                    inputs["object_geometry"],
                    inputs["node_mask"],
                    inputs["geometry_valid"])
                grounded = model(
                    inputs["object_tokens"],
                    inputs["object_motion"],
                    inputs["object_geometry"],
                    inputs["node_mask"],
                    inputs["geometry_valid"],
                    bodyNodePoseCamera=body_pose,
                    bodyNodeObserved=body_observed,
                    bodyNodeHealthy=body_healthy)
                invalid_inputs = {
                    name: value.clone() for name, value in inputs.items()}
                invalid_inputs["geometry_valid"][0, 0] = False
                invalid_baseline = model(
                    invalid_inputs["object_tokens"],
                    invalid_inputs["object_motion"],
                    invalid_inputs["object_geometry"],
                    invalid_inputs["node_mask"],
                    invalid_inputs["geometry_valid"])
                invalid_grounded = model(
                    invalid_inputs["object_tokens"],
                    invalid_inputs["object_motion"],
                    invalid_inputs["object_geometry"],
                    invalid_inputs["node_mask"],
                    invalid_inputs["geometry_valid"],
                    bodyNodePoseCamera=body_pose,
                    bodyNodeObserved=body_observed,
                    bodyNodeHealthy=body_healthy)
            delta = (
                grounded["SelfPartLogits"]
                - baseline["SelfPartLogits"])
            assert delta[0, 0, 0] > delta[0, 0, 1]
            assert delta.min() >= -4.0
            assert delta.max() <= 0.0
            assert torch.allclose(
                invalid_grounded["SelfPartLogits"][0, 0],
                invalid_baseline["SelfPartLogits"][0, 0],
                atol=1e-6,
                rtol=1e-6)
            model.train()
            model.zero_grad(set_to_none=True)
            learned = model(
                inputs["object_tokens"],
                inputs["object_motion"],
                inputs["object_geometry"],
                inputs["node_mask"],
                inputs["geometry_valid"],
                bodyNodePoseCamera=body_pose,
                bodyNodeObserved=body_observed,
                bodyNodeHealthy=body_healthy)
            learned["SelfPartLogits"][..., 1:].mean().backward()
            assert model.self_part_geometry_gain.grad is not None
            assert model.self_part_geometry_log_scale.grad is not None
            assert torch.isfinite(model.self_part_geometry_gain.grad)
            assert torch.isfinite(model.self_part_geometry_log_scale.grad)
            print("SelfPart geometry evidence test passed.")
            return True
        except Exception as e:
            print(
                "SelfPart geometry evidence test failed: "
                f"{type(e).__name__}: {e}")
            return False

    def TestSemanticWorldView(self) -> bool:
        try:
            B, K = 2, 4
            view = PhysicalStateExtractor.SemanticWorldView(self.MakeSemanticNodes(B=B, K=K))
            assert tuple(view["NodePresence"].shape) == (B, K)
            assert tuple(view["PoseCamera"].shape) == (B, K, ModuleDim.PstPoseDim)
            assert tuple(view["IdentityKey"].shape) == (B, K, ModuleDim.PstIdDim)
            assert tuple(view["Semantic"].shape) == (B, K, ModuleDim.PstSemanticDim)
            assert tuple(view["ParentProb"].shape) == (B, K, K)
            for name, value in view.items():
                self.AssertFinite(value, f"SemanticWorldView {name}")
            print("SemanticWorldView test passed.")
            return True
        except Exception as e:
            print(f"SemanticWorldView test failed: {type(e).__name__}: {e}")
            return False

    def TestIdentityKeySeparatesSameSemanticEntities(self) -> bool:
        try:
            nodes = self.MakeSemanticNodes(B=1, K=2)
            nodes["identity_embed"].zero_()
            nodes["identity_embed"][0, 0, 0] = 1.0
            nodes["identity_embed"][0, 1, 1] = 10.0
            for name in ("level_logits", "object_class_logits", "part_class_logits"):
                nodes[name].fill_(-100.0)
                nodes[name][..., 0] = 100.0

            identity_key = PhysicalStateExtractor.SemanticWorldView(nodes)["IdentityKey"]
            identity_dim = int(nodes["identity_embed"].size(-1))
            identity_norm = identity_key[..., :identity_dim].norm(dim=-1)
            similarity = F.cosine_similarity(identity_key[:, 0], identity_key[:, 1], dim=-1)

            assert torch.allclose(
                identity_norm[:, 0], identity_norm[:, 1], atol=1e-6), (
                "IdentityKey must normalize identity embeddings before semantic fusion")
            assert float(similarity.item()) < 0.5, (
                f"orthogonal identities with equal semantics are too similar: {float(similarity.item()):.6f}")
            print("IdentityKey same-semantic separation test passed.")
            return True
        except Exception as e:
            print(f"IdentityKey same-semantic separation test failed: {type(e).__name__}: {e}")
            return False

    def TestPhysicalStateLossFiniteAndBackward(self) -> bool:
        try:
            model, pst, targets = self.MakePhysicalState(B=2, K=4, tokenDim=32)
            loss_mod = PhysicalStateLoss().to(self.device)
            losses = loss_mod(pst, targets)
            expected = {
                "loss_mphys",
                "loss_state",
                "loss_attributes",
                "loss_affordance",
                "loss_relation",
                "loss_external_relation",
                "loss_motion",
                "loss_moving",
                "loss_contact",
                "loss_contact_force",
                "loss_contact_point",
                "loss",}
            assert expected.issubset(losses.keys()), f"missing losses: {expected.difference(losses.keys())}"
            self.AssertFinite(losses["loss"], "PhysicalStateLoss total")
            model.zero_grad(set_to_none=True)
            losses["loss"].backward()
            grad_norm = sum(
                float(p.grad.detach().abs().sum().item())
                for p in model.parameters()
                if p.grad is not None)
            assert grad_norm > 0.0, "no gradients flowed through PhysicalStateExtractor"
            print("PhysicalStateLoss finite/backward test passed.")
            return True
        except Exception as e:
            print(f"PhysicalStateLoss finite/backward test failed: {type(e).__name__}: {e}")
            return False

    def TestMissedObservationStillReceivesPositiveSupervision(self) -> bool:
        try:
            B, K, token_dim = 1, 1, 32
            model = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device)
            inputs = self.MakeExtractorInputs(B=B, K=K, tokenDim=token_dim)
            inputs["node_mask"].zero_()
            physical = model(
                inputs["object_tokens"],
                inputs["object_motion"],
                inputs["object_geometry"],
                inputs["node_mask"],
                inputs["geometry_valid"])
            pst = {
                **physical,
                **model.SemanticWorldView(self.MakeSemanticNodes(B=B, K=K)),}
            losses = PhysicalStateLoss().to(self.device)(
                pst,
                self.MakeTargets(B=B, K=K))
            model.zero_grad(set_to_none=True)
            (
                losses["loss_perceptual_presence"]
                + losses["loss_state"]
            ).backward()
            objectness_grad = sum(
                float(parameter.grad.abs().sum().item())
                for parameter in model.objectness_head.parameters()
                if parameter.grad is not None)
            state_grad = sum(
                float(parameter.grad.abs().sum().item())
                for parameter in model.state_head.parameters()
                if parameter.grad is not None)
            ok = bool(
                torch.count_nonzero(physical["ObservationMask"]).item() == 0
                and losses["loss_perceptual_presence"].item() > 0.0
                and objectness_grad > 0.0
                and state_grad > 0.0)
            print(
                f"MissedObservationPositiveSupervision "
                f"{'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"MissedObservationPositiveSupervision failed: {type(e).__name__}: {e}")
            return False

    def TestPSTWorldBinderShapes(self) -> bool:
        try:
            B, K = 2, 4
            _, pst, _ = self.MakePhysicalState(B=B, K=K, tokenDim=32)
            observed = torch.tensor(
                [[True, False, True, False], [False, True, False, True]],
                device=self.device)
            pst["Observed"] = observed
            pst["SlotPresence"] = torch.ones(B, K, device=self.device)
            pst["MphysRaw"] = torch.ones(B, K, device=self.device)
            pst["LastSeen"] = torch.tensor([[8, 4, 8, 0], [2, 8, 3, 8]], device=self.device)
            pst["Step"] = torch.full((B,), 8, device=self.device, dtype=torch.long)
            binder = PSTWorldBinder().to(self.device)
            out = binder(
                torch.randn(B, ModuleDim.WorldOutHState, device=self.device),
                torch.randn(B, ModuleDim.WorldOutZState, device=self.device),
                torch.randn(B, ModuleDim.WorldOutXState, device=self.device),
                pst,
                torch.randn(B, 256, device=self.device),
                torch.randn(B, ModuleDim.PstSlotDim, device=self.device))
            assert tuple(out["bound_mu"].shape) == (B, ModuleDim.WorldOutZState)
            assert tuple(out["pst_context"].shape) == (B, ModuleDim.PstSlotDim)
            assert tuple(out["slot_binding_weight"].shape) == (B, K)
            assert bool((out["memory_weight"][observed] == 0.0).all().item())
            query_sum = out["query_pool_weight"].sum(dim=-1)
            assert bool(torch.allclose(query_sum, torch.ones_like(query_sum), atol=1e-5))
            for name, value in out.items():
                self.AssertFinite(value, f"PSTWorldBinder {name}")
            print("PSTWorldBinder shape/memory test passed.")
            return True
        except Exception as e:
            print(f"PSTWorldBinder shape/memory test failed: {type(e).__name__}: {e}")
            return False

    def TestVirtualRealmCanonicalPhysicalProjection(self) -> bool:
        try:
            B, K, token_dim = 1, 2, 32
            model = PhysicalStateExtractor(
                robotMorphology=self.MakeRobotMorphology(),
                inObjectDim=token_dim,
                motionDim=token_dim,
                numSlotLayers=1,
                numHeads=4).to(self.device).eval()
            with torch.no_grad():
                for parameter in model.realm_head.parameters():
                    parameter.zero_()
                model.realm_head[-1].bias.fill_(-10.0)
                model.realm_head[-1].bias[
                    int(EntityRealm.VIRTUAL_CONTENT)] = 10.0
                for head in (
                    model.objectness_head,
                    model.physicality_head,
                    model.interaction_head,
                    model.moving_head,
                    model.contact_head,
                ):
                    for parameter in head.parameters():
                        parameter.zero_()
                    head[-1].bias.fill_(10.0)
                inputs = self.MakeExtractorInputs(
                    B=B, K=K, tokenDim=token_dim)
                out = model(
                    inputs["object_tokens"],
                    inputs["object_motion"],
                    inputs["object_geometry"],
                    inputs["node_mask"],
                    inputs["geometry_valid"])
            identity = torch.zeros_like(out["MotionCameraRaw"])
            identity[..., 6] = 1.0
            ok = bool(
                out["PerceptualPresence"].amin().item() > 0.99
                and torch.count_nonzero(out["GeometryValidMask"]).item() == 0
                and torch.count_nonzero(out["MphysRaw"]).item() == 0
                and torch.count_nonzero(
                    out["PhysicalInteractionProb"]).item() == 0
                and torch.equal(out["MotionCameraRaw"], identity)
                and torch.equal(out["CarrierMotionCameraRaw"], identity)
                and torch.equal(out["ArticulationMotionCameraRaw"], identity)
                and torch.count_nonzero(out["MovingProbRaw"]).item() == 0
                and torch.count_nonzero(out["ContactProbRaw"]).item() == 0
                and torch.count_nonzero(
                    out["PairwiseRelationCamera"][..., :4]).item() == 0)
            print(
                "Virtual realm canonical physical projection "
                f"{'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(
                "Virtual realm canonical physical projection failed: "
                f"{type(e).__name__}: {e}")
            return False

    def TestPSTWorldBinderEmptyAndInvalidMask(self) -> bool:
        try:
            B, K = 2, 4
            _, pst, _ = self.MakePhysicalState(B=B, K=K, tokenDim=32)
            pst["Observed"][0].zero_()
            pst["SlotPresence"][0].zero_()
            pst["MphysRaw"][0].fill_(float("nan"))
            vector_fields = (
                "SlotState", "PoseCamera", "ARaw", "Size", "StateRaw",
                "AffordanceRaw", "MotionCameraRaw", "ContactForceRaw", "ContactPointCameraRaw",
                "Semantic", "ExternalRelationProbRaw", "IdentityKey")
            scalar_fields = (
                "MovingProbRaw", "ContactProbRaw", "Visibility", "Occlusion")
            for key in vector_fields:
                pst[key][0].fill_(float("nan"))
            for key in scalar_fields:
                pst[key][0].fill_(float("nan"))
            pst["PairwiseRelationCamera"][0].fill_(float("nan"))

            binder = PSTWorldBinder().to(self.device).eval()
            world_z = torch.randn(B, ModuleDim.WorldOutZState, device=self.device)
            out = binder(
                torch.randn(B, ModuleDim.WorldOutHState, device=self.device),
                world_z,
                torch.randn(B, ModuleDim.WorldOutXState, device=self.device),
                pst,
                torch.randn(B, 256, device=self.device),
                torch.randn(B, ModuleDim.PstSlotDim, device=self.device))
            for name, value in out.items():
                self.AssertFinite(value, f"PSTWorldBinder masked {name}")
            ok = (
                float(out["slot_binding_weight"][0].abs().sum().item()) == 0.0
                and float(out["pst_context"][0].abs().sum().item()) == 0.0
                and float(out["bind_gate"][0].abs().sum().item()) == 0.0
                and float(out["delta_mu"][0].abs().sum().item()) == 0.0
                and torch.equal(out["bound_mu"][0], world_z[0]))
            print(f"PSTWorldBinder empty/invalid mask {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"PSTWorldBinder empty/invalid mask failed: {type(e).__name__}: {e}")
            return False

    def TestPerceptionPhysicalTrainerForwardSmoke(self) -> bool:
        try:
            B, H, W, K = 1, 32, 32, 4
            trainer = PerceptionPhysicalTrainer(
                cameraIntrinsics=self.MakeCameraIntrinsics(H),
                robotMorphology=self.MakeRobotMorphology(),
                physicalWeight=0.25,
                recallLossKwargs={"identityBankSize": 16},
                imgSize=H,
                patchSize=1,
                embedDim=32,
                numHeads=4,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=K,).to(self.device)
            frames = torch.rand(B, 3, H, W, device=self.device)
            depth = torch.ones(B, 1, H, W, device=self.device)
            depth_valid = torch.ones_like(depth, dtype=torch.bool)
            targets = self.MakeTargets(B=B, K=K, H=H, W=W)
            targets["rgb"] = frames
            targets["depth"] = depth
            targets["depth_valid"] = depth_valid
            camera_motion = torch.zeros(
                B, ModuleDim.ObserverMotionDim, device=self.device)
            camera_motion[:, 6] = 1.0
            out = trainer(
                frames,
                depth,
                depth_valid,
                self.MakeTopDownContext(trainer.perception, B, frames.dtype),
                targets,
                cameraMotion=camera_motion,
                prevVisualValid=torch.zeros(B, device=self.device, dtype=torch.bool))
            assert "physical_state" in out and "physical_losses" in out
            assert tuple(out["physical_state"]["SlotState"].shape[:2]) == (B, K)
            self.AssertFinite(out["loss"], "PerceptionPhysicalTrainer loss")
            out["loss"].backward()
            print("PerceptionPhysicalTrainer forward smoke test passed.")
            return True
        except Exception as e:
            print(f"PerceptionPhysicalTrainer forward smoke test failed: {type(e).__name__}: {e}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "SlotCrossAttention": self.TestSlotCrossAttention(),
            "SlotRefineLayer": self.TestSlotRefineLayer(),
            "PhysicalStateExtractorForwardShapes": self.TestPhysicalStateExtractorForwardShapes(),
            "SelfPartMorphologyPermutation": self.TestSelfPartMorphologyPermutation(),
            "SelfPartDynamicCardinality": self.TestSelfPartDynamicCardinality(),
            "SelfPartGeometryEvidence": self.TestSelfPartGeometryEvidence(),
            "SemanticWorldView": self.TestSemanticWorldView(),
            "IdentityKeySeparatesSameSemanticEntities": self.TestIdentityKeySeparatesSameSemanticEntities(),
            "PhysicalStateLossFiniteAndBackward": self.TestPhysicalStateLossFiniteAndBackward(),
            "MissedObservationPositiveSupervision": self.TestMissedObservationStillReceivesPositiveSupervision(),
            "PSTWorldBinderShapes": self.TestPSTWorldBinderShapes(),
            "VirtualRealmCanonicalPhysicalProjection": self.TestVirtualRealmCanonicalPhysicalProjection(),
            "PSTWorldBinderEmptyAndInvalidMask": self.TestPSTWorldBinderEmptyAndInvalidMask(),
            "PerceptionPhysicalTrainerForwardSmoke": self.TestPerceptionPhysicalTrainerForwardSmoke(),}
        passed = sum(1 for value in results.values() if value)
        print(f"\n[PhysicalStateModule Tests] {passed}/{len(results)} passed.")
        return results
