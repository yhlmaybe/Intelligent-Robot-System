from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import AGICoreModule, HungarianAssignment, BuildReferenceWeights
from ModuleMessagerManager import ModuleDim
from PerceptionModule import PerceiveExtractor, PerceptionRecallLoss, TopDownContext, VisualState

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
            "PoseCamera": nodes["pose_camera"],
            "Size": nodes["size_3d"],
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

        self.in_proj = nn.Linear(inObjectDim, self.slot_dim)
        self.motion_in_proj = nn.Linear(motionDim, self.slot_dim)
        self.objectness_head = self.MakeHead(self.slot_dim, 1)

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
        pairMask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        relative = P[..., :3].unsqueeze(1) - P[..., :3].unsqueeze(2)
        distance = relative.norm(dim=-1, keepdim=True)
        pair = (
            self.relation_subject(S).unsqueeze(2)
            + self.relation_object(S).unsqueeze(1)
            + self.relation_geometry(relative))
        
        relation_logits = self.relation_head(F.silu(pair))
        relation_prob = F.softmax(relation_logits, dim=-1)
        relation = torch.cat([relative, distance, relation_prob], dim=-1)
        relation = relation * pairMask.unsqueeze(-1)
        return relation, relation_logits

    def forward(
        self,
        objectTokens: torch.Tensor, # [B, K, D_obj]
        objectMotion: torch.Tensor, # [B, K, D_motion]
        objectGeometry: torch.Tensor, # [B, K, 7]
        nodeMask: torch.Tensor,
        geometryValid: torch.Tensor) -> Dict[str, torch.Tensor]:
        observation_mask = (
            (nodeMask > self.node_mask_threshold)
            & (geometryValid.squeeze(-1) > self.geometry_mask_threshold)).to(objectTokens.dtype) # [B, K]
        
        S_raw = self.in_proj(objectTokens) + self.motion_in_proj(objectMotion) # [B, K, 128]

        for layer in self.slot_layers:
            S_raw = layer(S_raw, observation_mask)

        S_raw = self.slot_post(S_raw) * observation_mask.unsqueeze(-1) # [B, K, 128]

        mphys_logits = self.objectness_head(S_raw).squeeze(-1)
        mphys_raw = torch.sigmoid(mphys_logits) # [B, K]
        physical_available = mphys_raw * observation_mask
        slot_available = physical_available.unsqueeze(-1)

        state_logits = self.state_head(S_raw) # [B, K, PstStateDim]
        state_raw = torch.sigmoid(state_logits) * slot_available

        attr_pred = self.attribute_head(S_raw)
        attr_raw = attr_pred * slot_available

        affordance_logits = self.affordance_head(S_raw)
        affordance_raw = torch.sigmoid(affordance_logits) * slot_available

        motion_pred = self.NormalizePose(self.motion_head(S_raw))
        identity_motion = torch.zeros_like(motion_pred)
        identity_motion[..., 6] = 1.0
        motion_raw = self.NormalizePose(identity_motion + (motion_pred - identity_motion) * slot_available)

        moving_logits = self.moving_head(S_raw).squeeze(-1)
        moving_raw = torch.sigmoid(moving_logits) * physical_available

        contact_raw = self.contact_head(S_raw) # [B, K, 6]
        contact_logits = contact_raw[..., 0]
        contact_prob_raw = torch.sigmoid(contact_logits) * physical_available
        contact_force_pred = F.softplus(contact_raw[..., 1:3])
        contact_point_pred = contact_raw[..., 3:6]
        contact_weight = contact_prob_raw.unsqueeze(-1)
        contact_force_raw = contact_force_pred * contact_weight
        contact_point_raw = contact_point_pred * contact_weight

        external_relation_logits = self.external_relation_head(S_raw)
        external_relation_raw = torch.sigmoid(external_relation_logits) * slot_available
        pair_mask = observation_mask.unsqueeze(1) * observation_mask.unsqueeze(2)

        off_diagonal = 1.0 - torch.eye(
            observation_mask.size(1), device=observation_mask.device, dtype=observation_mask.dtype)
        
        pair_mask = pair_mask * off_diagonal.unsqueeze(0) # [B, K, K]
        pairwise_relation, relation_logits_raw = self.BuildRelations(S_raw, objectGeometry, pair_mask)
        
        return {
            "SlotState": S_raw,
            "ObservationMask": observation_mask,
            "MphysLogits": mphys_logits,
            "MphysRaw": physical_available,
            "AttributePred": attr_pred,
            "ARaw": attr_raw,
            "StateLogits": state_logits,
            "StateRaw": state_raw,
            "AffordanceLogits": affordance_logits,
            "AffordanceRaw": affordance_raw,
            "MotionPred": motion_pred,
            "MotionRaw": motion_raw,
            "MovingLogits": moving_logits,
            "MovingProbRaw": moving_raw,
            "ContactLogits": contact_logits,
            "ContactProbRaw": contact_prob_raw,
            "ContactForcePred": contact_force_pred,
            "ContactForceRaw": contact_force_raw,
            "ContactPointPred": contact_point_pred,
            "ContactPointRaw": contact_point_raw,
            "PairwiseRelation": pairwise_relation,
            "RelationLogitsRaw": relation_logits_raw,
            "ExternalRelationLogits": external_relation_logits,
            "ExternalRelationProbRaw": external_relation_raw}

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
        mphysNegativeWeight: float = 0.05):
        super().__init__()
        self.state_weight = float(stateWeight)
        self.attribute_weight = float(attributeWeight)
        self.affordance_weight = float(affordanceWeight)
        self.relation_weight = float(relationWeight)
        self.motion_weight = float(motionWeight)
        self.contact_weight = float(contactWeight)
        self.mphys_weight = float(mphysWeight)
        self.mphys_negative_weight = float(mphysNegativeWeight)

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
        candidate = torch.nonzero(pst["ObservedSlotMask"][b] > 0.0, as_tuple=False).flatten()
        if candidate.numel() == 0:
            empty = torch.empty(0, device=pst["SlotState"].device, dtype=torch.long)
            return empty, empty
        levels = targets["node_level"][b, gt]
        object_score = pst["ObjectClassProb"][b, candidate][:, targets["object_classes"][b, gt]]
        part_score = pst["PartClassProb"][b, candidate][:, targets["part_classes"][b, gt]]
        class_score = torch.where((levels == 0).unsqueeze(0), object_score, part_score)
        cost = (
            2.0 * self.PoseCost(pst["PoseCamera"][b, candidate], targets["pose_camera"][b, gt])
            - torch.sigmoid(pst["MphysLogits"][b, candidate]).unsqueeze(1)
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
        mphys_terms: List[torch.Tensor] = []
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
            mphys_target = torch.zeros_like(pst["MphysLogits"][b])
            mphys_weight = torch.full_like(pst["MphysLogits"][b], self.mphys_negative_weight)
            mphys_target[pred] = 1.0
            mphys_weight[pred] = 1.0
            mphys_raw = F.binary_cross_entropy_with_logits(
                pst["MphysLogits"][b],
                mphys_target,
                reduction="none")
            mphys_terms.append((mphys_raw * mphys_weight).sum() / mphys_weight.sum().clamp_min(1.0))
            if gt.numel() == 0:
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

        loss_mphys = self.MeanTerms(mphys_terms)
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
            self.mphys_weight * loss_mphys
            + self.state_weight * loss_state
            + self.attribute_weight * loss_attribute
            + self.affordance_weight * loss_affordance
            + self.relation_weight * (loss_relation + loss_external_relation)
            + self.motion_weight * (loss_motion + loss_moving)
            + self.contact_weight * (loss_contact + loss_force + loss_point))
        return {
            "loss_mphys": loss_mphys,
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
        physicalWeight: float = 1.0,
        recallLossKwargs: Optional[Dict[str, Any]] = None,
        **perceptionKwargs: Any):
        super().__init__()
        perceptionKwargs = dict(perceptionKwargs)
        perceptionKwargs["enableRecallAuxiliary"] = True
        self.perception = PerceiveExtractor(**perceptionKwargs)
        self.recall_loss = PerceptionRecallLoss(**({} if recallLossKwargs is None else recallLossKwargs))
        self.physical = PhysicalStateExtractor(
            inObjectDim=self.perception.embed_dim,
            motionDim=self.perception.embed_dim)
        self.physical_loss = PhysicalStateLoss()
        self.physical_weight = float(physicalWeight)

    def SetCameraIntrinsics(
        self,
        intrinsics: torch.Tensor,
        sourceSize: Optional[Tuple[int, int]] = None) -> None:
        self.perception.SetCameraIntrinsics(intrinsics, sourceSize=sourceSize)

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        topDownContext: TopDownContext,
        targets: Dict[str, torch.Tensor],
        prevVisualState: Optional[VisualState] = None,
        cameraMotion: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        visual_state = self.perception(
            rgb,
            topDownContext=topDownContext,
            depth=depth,
            depthValid=depthValid,
            prevVisualState=prevVisualState,
            cameraMotion=cameraMotion)
        
        recall_out = self.perception.recall_heads(visual_state)

        representation_loss = self.perception.ComputePerceptionLoss(
            visual_state,
            depthTarget=targets["depth"],
            depthTargetValid=targets["depth_valid"],
            prevVisualState=prevVisualState,
            cameraMotion=cameraMotion)
        
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
            + int(relDim))
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
            memoryScale=self.memory_scale,
            memoryDecayHorizon=self.memory_decay_horizon)
        clean = lambda value: torch.nan_to_num(
            value, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        return (
            clean(reference_weights.slot_weight),
            clean(reference_weights.observed_weight),
            clean(reference_weights.memory_weight))

    def RelationSummary(self, physicalState: Dict[str, torch.Tensor], slotWeight: torch.Tensor) -> torch.Tensor:
        relation = physicalState["PairwiseRelation"]
        if "PairRelationLastSeen" in physicalState and "Step" in physicalState:
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
        if "ContactPointWorldRaw" in physicalState:
            contact_point = physicalState["ContactPointWorldRaw"]
        elif "ContactPointWorld" in physicalState:
            contact_point = physicalState["ContactPointWorld"]
        else:
            contact_point = physicalState["ContactPointRaw"]
        slot_content = torch.cat([
            physicalState["SlotState"],
            physicalState["PoseWorld"],
            physicalState["ARaw"],
            physicalState["Size"],
            physicalState["StateRaw"],
            physicalState["AffordanceRaw"],
            physicalState["MotionRaw"],
            physicalState["MovingProbRaw"].unsqueeze(-1),
            physicalState["ContactProbRaw"].unsqueeze(-1),
            physicalState["ContactForceRaw"],
            contact_point,
            physicalState["Visibility"].unsqueeze(-1),
            physicalState["Occlusion"].unsqueeze(-1),
            physicalState["Semantic"],
            physicalState["ExternalRelationProbRaw"],
            relation_summary], dim=-1)
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

    def MakeTopDownContext(self, model, B: int, dtype: torch.dtype = torch.float32) -> TopDownContext:
        runtime = model.base if hasattr(model, "base") else model
        return TopDownContext(
            PredictedVisual=None,
            Precision=torch.ones(B, device=self.device, dtype=dtype),
            MemoryCue=torch.zeros(B, int(runtime.integrated_dim), device=self.device, dtype=dtype),)

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
        camera_pose_world = torch.zeros(B, ModuleDim.PstPoseDim, device=self.device)
        camera_pose_world[:, 6] = 1.0
        return {
            "rgb": torch.rand(B, 3, H, W, device=self.device),
            "depth": depth,
            "depth_valid": torch.ones_like(depth, dtype=torch.bool),
            "normal": normal,
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
            "pose_world": pose,
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
            "camera_pose_world": camera_pose_world,}

    def MakePhysicalState(
        self,
        B: int = 2,
        K: int = 4,
        tokenDim: int = 32) -> Tuple[PhysicalStateExtractor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        model = PhysicalStateExtractor(
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
        pst["PoseWorld"] = pst["PoseCamera"]
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
            assert tuple(out["PairwiseRelation"].shape) == (B, K, K, ModuleDim.PstRelDim)
            assert tuple(out["RelationLogitsRaw"].shape) == (B, K, K, ModuleDim.PstRelationClasses)
            assert bool((out["ObservationMask"][:, -1] == 0.0).all().item())
            quat_norm = out["MotionRaw"][..., 3:7].norm(dim=-1)
            assert bool(torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-4))
            assert bool((out["SlotState"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["ARaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["StateRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["AffordanceRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["ContactForceRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["ContactPointRaw"][:, -1].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["MotionRaw"][:, -1, :3].abs().sum(dim=-1) == 0.0).all().item())
            assert bool((out["MotionRaw"][:, -1, 3:6].abs().sum(dim=-1) == 0.0).all().item())
            assert bool(torch.allclose(out["MotionRaw"][:, -1, 6], torch.ones(B, device=self.device), atol=1e-5))
            for name, value in out.items():
                self.AssertFinite(value, f"PhysicalStateExtractor {name}")
            print("PhysicalStateExtractor forward shape test passed.")
            return True
        except Exception as e:
            print(f"PhysicalStateExtractor forward shape test failed: {type(e).__name__}: {e}")
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

    def TestPSTWorldBinderEmptyAndInvalidMask(self) -> bool:
        try:
            B, K = 2, 4
            _, pst, _ = self.MakePhysicalState(B=B, K=K, tokenDim=32)
            pst["Observed"][0].zero_()
            pst["SlotPresence"][0].zero_()
            pst["MphysRaw"][0].fill_(float("nan"))
            vector_fields = (
                "SlotState", "PoseWorld", "ARaw", "Size", "StateRaw",
                "AffordanceRaw", "MotionRaw", "ContactForceRaw", "ContactPointRaw",
                "Semantic", "ExternalRelationProbRaw", "IdentityKey")
            scalar_fields = (
                "MovingProbRaw", "ContactProbRaw", "Visibility", "Occlusion")
            for key in vector_fields:
                pst[key][0].fill_(float("nan"))
            for key in scalar_fields:
                pst[key][0].fill_(float("nan"))
            pst["PairwiseRelation"][0].fill_(float("nan"))

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
                physicalWeight=0.25,
                recallLossKwargs={"identityBankSize": 16},
                imgSize=H,
                patchSize=1,
                embedDim=32,
                numHeads=4,
                numLayers=1,
                baseChannels=8,
                objectTokenCount=K,
                useHebbian=False).to(self.device)
            frames = torch.rand(B, 3, H, W, device=self.device)
            depth = torch.ones(B, 1, H, W, device=self.device)
            depth_valid = torch.ones_like(depth, dtype=torch.bool)
            targets = self.MakeTargets(B=B, K=K, H=H, W=W)
            targets["rgb"] = frames
            targets["depth"] = depth
            targets["depth_valid"] = depth_valid
            out = trainer(
                frames,
                depth,
                depth_valid,
                self.MakeTopDownContext(trainer.perception, B, frames.dtype),
                targets)
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
            "SemanticWorldView": self.TestSemanticWorldView(),
            "IdentityKeySeparatesSameSemanticEntities": self.TestIdentityKeySeparatesSameSemanticEntities(),
            "PhysicalStateLossFiniteAndBackward": self.TestPhysicalStateLossFiniteAndBackward(),
            "PSTWorldBinderShapes": self.TestPSTWorldBinderShapes(),
            "PSTWorldBinderEmptyAndInvalidMask": self.TestPSTWorldBinderEmptyAndInvalidMask(),
            "PerceptionPhysicalTrainerForwardSmoke": self.TestPerceptionPhysicalTrainerForwardSmoke(),}
        passed = sum(1 for value in results.values() if value)
        print(f"\n[PhysicalStateModule Tests] {passed}/{len(results)} passed.")
        return results
