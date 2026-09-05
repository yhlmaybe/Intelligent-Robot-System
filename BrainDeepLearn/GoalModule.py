from __future__ import annotations
from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import RobotEmbodimentContractView


class CodebookGoalHead(AGICoreModule):
    def __init__(self, contextDim: int, groups: int, codes: int, goalDim: int, hidden: int = 256):
        super().__init__()
        self.groups = int(groups)
        self.codes = int(codes)
        self.code_dim = self.groups * self.codes

        self.manager = nn.Sequential(
            nn.LayerNorm(contextDim),
            nn.Linear(contextDim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.code_dim),)

        self.decoder = nn.Sequential(
            nn.Linear(self.code_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, goalDim),)

        self.register_buffer("code_usage", torch.full((self.groups, self.codes), 1.0 / self.codes), persistent=True)

    @staticmethod
    def StraightThroughOneHot(logits: torch.Tensor) -> torch.Tensor:
        soft = F.softmax(logits, dim=-1)
        idx = soft.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0) # [B, groups, codes]
        return hard + soft - soft.detach()

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.manager(context).view(context.size(0), self.groups, self.codes)
        onehot = self.StraightThroughOneHot(logits)

        with torch.no_grad():
            self.code_usage.mul_(0.99).add_(0.01 * onehot.detach().mean(dim=0))

        code = onehot.view(context.size(0), self.code_dim)
        goal = self.decoder(code)
        return {"goal": goal, "logits": logits, "code": code, "index": logits.argmax(dim=-1), "usage": self.code_usage}

    def UtilizationLoss(self, logits: torch.Tensor) -> torch.Tensor:
        probability = F.softmax(logits.float(), dim=-1)
        prob = probability.mean(dim=0)
        probability_floor = max(
            1e-8,
            float(torch.finfo(prob.dtype).tiny))
        soft_balance = (
            math.log(self.codes)
            + (prob * prob.clamp_min(
                probability_floor).log()).sum(dim=-1)).mean()

        sample_prob = probability
        hard_index = sample_prob.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(sample_prob).scatter_(-1, hard_index, 1.0)
        hard_st = hard + sample_prob - sample_prob.detach()
        hard_usage = hard_st.mean(dim=0)
        hard_balance = (
            math.log(self.codes)
            + (hard_usage * hard_usage.clamp_min(
                probability_floor).log()).sum(dim=-1)
        ).mean()
        return soft_balance + 0.25 * hard_balance

    @torch.no_grad()
    def ResetDeadCodes(self, threshold: float = 0.05):
        dead = (self.code_usage < threshold / self.codes).view(-1)
        rows = dead.nonzero(as_tuple=False).flatten()
        head = self.manager[-1]
        head.weight[rows] = torch.randn_like(head.weight[rows]) * 0.02
        head.bias[rows] = 0.0
        self.code_usage.view(-1)[rows] = 1.0 / self.codes

    def Decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code)


class GoalGrounding(AGICoreModule):
    def __init__(
        self,
        goalDim: int = ModuleDim.GoalShortDim,
        intentDim: int = ModuleDim.IntentionFeat,
        slotDim: int = ModuleDim.PstSlotDim,
        usageDim: int = ModuleDim.PstUsageDim,
        numSkills: int = ModuleDim.UsageNumSkills,
        paramDim: int = ModuleDim.UsageParamDim,
        subgoalSteps: int = 4,
        numHeads: int = 4,):
        super().__init__()
        self.slot_dim = int(slotDim)
        self.subgoal_steps = int(subgoalSteps)

        self.goal_intent_proj = nn.Sequential(
            nn.LayerNorm(int(goalDim) + int(intentDim)),
            nn.Linear(int(goalDim) + int(intentDim), self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),)

        reference_context_dim = int(goalDim) + int(intentDim) + 8
        self.reference_memory_scale_head = nn.Sequential(
            nn.LayerNorm(reference_context_dim),
            nn.Linear(reference_context_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, 1),)

        self.slot_ground_encoder = nn.Sequential(
            nn.LayerNorm(self.slot_dim + int(usageDim) + 4),
            nn.Linear(self.slot_dim + int(usageDim) + 4, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.LayerNorm(self.slot_dim),)

        ontology_dim = (
            5
            + 5
            + 5 * 5
            + 5
            + 1
            + ModuleDim.PstSelfPartSemanticDim
            + 1
            + 1
            + 2
            + 1
            + 2
            + 1)
        self.ontology_slot_encoder = nn.Sequential(
            nn.LayerNorm(ontology_dim),
            nn.Linear(ontology_dim, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.LayerNorm(self.slot_dim),)
        self.ontology_slot_gain = nn.Parameter(torch.tensor(-2.944439))
        self.entity_text_feature_dim = 515
        self.entity_text_slot_encoder = nn.Sequential(
            nn.LayerNorm(self.entity_text_feature_dim),
            nn.Linear(self.entity_text_feature_dim, self.slot_dim * 2),
            nn.SiLU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim),
            nn.LayerNorm(self.slot_dim),)
        self.entity_text_slot_residual = nn.Sequential(
            nn.LayerNorm(self.slot_dim * 2),
            nn.Linear(self.slot_dim * 2, self.slot_dim * 2),
            nn.SiLU(),
            nn.Linear(self.slot_dim * 2, self.slot_dim),)
        self.entity_text_slot_gate = nn.Sequential(
            nn.LayerNorm(self.slot_dim * 2),
            nn.Linear(self.slot_dim * 2, self.slot_dim),
            nn.Sigmoid(),)
        self.entity_text_slot_gain = nn.Parameter(torch.tensor(-2.944439))
        nn.init.zeros_(self.entity_text_slot_residual[-1].weight)
        nn.init.zeros_(self.entity_text_slot_residual[-1].bias)

        self.ground_attn = nn.MultiheadAttention(self.slot_dim, int(numHeads), batch_first=True)
        self.grounded_query_norm = nn.LayerNorm(self.slot_dim)
        self.no_slot_token = nn.Parameter(torch.randn(1, 1, self.slot_dim) * 0.02)

        self.no_slot_head = nn.Sequential(
            nn.LayerNorm(self.slot_dim),
            nn.Linear(self.slot_dim, 1),)

        self.subgoal_query = nn.Parameter(torch.randn(self.subgoal_steps, self.slot_dim) * 0.02)
        decomp_layer = nn.TransformerDecoderLayer(
            d_model=self.slot_dim,
            nhead=int(numHeads),
            dim_feedforward=self.slot_dim * 4,
            dropout=0.05,
            batch_first=True,
            norm_first=True,)
        self.decomposer = nn.TransformerDecoder(decomp_layer, num_layers=2)
        self.skill_head = nn.Linear(self.slot_dim, int(numSkills))
        self.slot_head = nn.Linear(self.slot_dim, 1)
        self.param_head = nn.Linear(self.slot_dim, int(paramDim))

    @staticmethod
    def MaskFloor(value: torch.Tensor) -> float:
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise TypeError("goal grounding mask requires a floating tensor")
        return float(torch.finfo(value.dtype).min)

    def BuildSemanticReferenceWeights(
        self,
        physicalState: Dict[str, torch.Tensor],
        memoryScale: torch.Tensor,
        memoryDecayHorizon: float = 32.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        presence = physicalState["PerceptualPresence"]
        observed = physicalState["Observed"].float()
        current_step = physicalState["Step"].view(-1, 1).float()
        memory_age = (
            current_step - physicalState["LastSeen"].float()
        ).clamp_min(0.0)
        memory_recency = torch.exp(-memory_age / float(memoryDecayHorizon))
        observed_weight = presence * observed
        memory_weight = (
            memoryScale
            * presence
            * physicalState["SlotPresence"]
            * (1.0 - observed)
            * memory_recency)
        return (
            observed_weight,
            memory_weight,
            observed_weight + memory_weight,
            memory_recency,)

    def BuildSemanticScaleContext(
        self,
        observedPhysicalState: Dict[str, torch.Tensor],
        demandQuery: torch.Tensor,
    ) -> torch.Tensor:
        observed_strength = (
            observedPhysicalState["ObservedSlotMask"]
            * observedPhysicalState["PerceptualPresence"])
        demand = F.normalize(demandQuery, dim=-1, eps=1e-6)
        slot = F.normalize(
            observedPhysicalState["SlotState"], dim=-1, eps=1e-6)
        demand_match = torch.einsum("bkd,bd->bk", slot, demand).add(1.0).mul(0.5)
        matched_strength = observed_strength * demand_match
        unmatched_strength = observed_strength * (1.0 - demand_match)
        top_match = torch.topk(matched_strength, k=2, dim=1).values
        observed_total = observed_strength.sum(dim=1, keepdim=True).clamp_min(1e-6)
        observed_max = observed_strength.amax(dim=1, keepdim=True)
        best_match = top_match[:, :1]
        second_match = top_match[:, 1:2]
        mean_match = matched_strength.sum(dim=1, keepdim=True) / observed_total
        ambiguity = second_match / best_match.clamp_min(1e-6)
        unresolved = unmatched_strength.sum(dim=1, keepdim=True) / observed_total
        return torch.cat([
            observed_strength.mean(dim=1, keepdim=True),
            observed_max,
            best_match,
            mean_match,
            1.0 - best_match,
            1.0 - observed_max,
            ambiguity,
            unresolved,], dim=-1)

    def BuildOntologySlotContext(
        self,
        physicalState: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        K = int(physicalState["SlotState"].size(1))
        surface_parent = physicalState["SurfaceParentProb"]
        parent_summary = torch.stack([
            surface_parent[..., :K].amax(dim=-1),
            surface_parent[..., K],], dim=-1)
        ontology = torch.cat([
            physicalState["RealmProb"],
            physicalState["MotionLayerProb"],
            physicalState["LayerAgencyProb"].flatten(-2),
            physicalState["AgencyProb"],
            physicalState["BodyMembershipProb"].unsqueeze(-1),
            physicalState["SelfPartSemantic"],
            physicalState["PhysicalInteractionProb"].unsqueeze(-1),
            physicalState["VerificationConfidence"].unsqueeze(-1),
            physicalState["SurfaceUV"],
            physicalState["SurfaceUVConfidence"].unsqueeze(-1),
            parent_summary,
            physicalState["PerceptualPresence"].unsqueeze(-1),], dim=-1)
        return self.ontology_slot_encoder(ontology)

    def BuildEntityTextSlotContext(
        self,
        physicalState: Dict[str, torch.Tensor],
        slotTensor: torch.Tensor,
    ) -> torch.Tensor:
        required = (
            "EntityTextSemantic",
            "EntityTextConfidence",
            "EntityTextRevision",
            "EntityTextChanged",)
        if any(name not in physicalState for name in required):
            return torch.zeros_like(slotTensor)
        confidence = physicalState["EntityTextConfidence"].unsqueeze(-1)
        revision = torch.tanh(
            physicalState["EntityTextRevision"].to(slotTensor.dtype).unsqueeze(-1)
            / 16.0)
        changed = physicalState["EntityTextChanged"].to(
            slotTensor.dtype).unsqueeze(-1)
        features = torch.cat([
            physicalState["EntityTextSemantic"].to(slotTensor.dtype),
            confidence,
            revision,
            changed,], dim=-1)
        code = self.entity_text_slot_encoder(features)
        combined = torch.cat([slotTensor, code], dim=-1)
        residual = self.entity_text_slot_residual(combined)
        gate = self.entity_text_slot_gate(combined)
        return 0.25 * confidence * torch.sigmoid(
            self.entity_text_slot_gain) * gate * residual

    def forward(
        self,
        goalEmbed: torch.Tensor,
        intentEmbed: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        observedPhysicalState: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        slot_tensor = physicalState["SlotState"]
        goal_intent = torch.cat([goalEmbed, intentEmbed], dim=-1)
        query_vec = self.goal_intent_proj(goal_intent)
        reference_context = self.BuildSemanticScaleContext(
            observedPhysicalState,
            query_vec) # [B, 8]

        memory_scale = torch.sigmoid(self.reference_memory_scale_head(torch.cat([goal_intent, reference_context], dim=-1)))

        (
            observed_weight,
            memory_weight,
            slot_weight,
            memory_recency,
        ) = self.BuildSemanticReferenceWeights(
            physicalState,
            memory_scale)

        slot_input = torch.cat([
            slot_tensor,
            physicalState["U"],
            observed_weight.unsqueeze(-1),
            memory_weight.unsqueeze(-1),
            slot_weight.unsqueeze(-1),
            memory_recency.unsqueeze(-1),], dim=-1)

        slot_embed = (
            self.slot_ground_encoder(slot_input)
            + torch.sigmoid(self.ontology_slot_gain)
            * self.BuildOntologySlotContext(physicalState))
        slot_embed = slot_embed + self.BuildEntityTextSlotContext(
            physicalState,
            slot_embed)
        B, _, _ = slot_embed.shape
        no_slot_token = self.no_slot_token.expand(B, 1, self.slot_dim)
        memory_tokens = torch.cat([slot_embed, no_slot_token], dim=1) # [B, K + 1, D]
        invalid_slot = slot_weight <= 0.0
        key_padding = torch.cat([
            invalid_slot,
            torch.zeros(B, 1, device=slot_weight.device, dtype=torch.bool),], dim=1)

        grounded, _ = self.ground_attn(
            query_vec.unsqueeze(1),
            memory_tokens,
            memory_tokens,
            key_padding_mask=key_padding)
        grounded_query = self.grounded_query_norm(query_vec + grounded.squeeze(1))

        subgoal_q = self.subgoal_query.unsqueeze(0).expand(B, self.subgoal_steps, self.slot_dim) # [B, S, D]
        decoded = self.decomposer(subgoal_q, memory_tokens, memory_key_padding_mask=key_padding)
        subgoal_skill_logits = self.skill_head(decoded)

        subgoal_step_logits = (
            self.slot_head(decoded).squeeze(-1)
            + torch.logsumexp(subgoal_skill_logits, dim=-1)
            + self.param_head(decoded).tanh().mean(dim=-1))

        subgoal_step_weight = F.softmax(subgoal_step_logits, dim=-1)
        subgoal_query = (decoded * subgoal_step_weight.unsqueeze(-1)).sum(dim=1)

        reference_prior = slot_weight.clamp_min(1e-6).log()
        query_slot_logits = (
            torch.einsum("bd,bkd->bk", grounded_query, slot_embed)
            / (float(self.slot_dim) ** 0.5)
            + reference_prior)
        subgoal_slot_logits = (
            torch.einsum("bd,bkd->bk", subgoal_query, slot_embed)
            / (float(self.slot_dim) ** 0.5)
            + reference_prior)
        mask_floor = self.MaskFloor(query_slot_logits)
        query_slot_logits = query_slot_logits.masked_fill(
            invalid_slot,
            mask_floor)
        subgoal_slot_logits = subgoal_slot_logits.masked_fill(
            invalid_slot,
            mask_floor)
        slot_logits = query_slot_logits + subgoal_slot_logits - reference_prior
        slot_logits = slot_logits.masked_fill(invalid_slot, mask_floor)

        no_slot_logit = self.no_slot_head(grounded_query + subgoal_query).squeeze(-1)
        reference_distribution = F.softmax(torch.cat([slot_logits, no_slot_logit.unsqueeze(-1)], dim=-1), dim=-1)
        query_reference_distribution = F.softmax(torch.cat([
            query_slot_logits,
            self.no_slot_head(grounded_query).squeeze(-1).unsqueeze(-1),
        ], dim=-1), dim=-1)
        subgoal_reference_distribution = F.softmax(torch.cat([
            subgoal_slot_logits,
            self.no_slot_head(subgoal_query).squeeze(-1).unsqueeze(-1),
        ], dim=-1), dim=-1)
        agreement_mean = 0.5 * (
            query_reference_distribution + subgoal_reference_distribution)
        grounding_consistency_loss = 0.5 * (
            (
                query_reference_distribution
                * (
                    query_reference_distribution.clamp_min(1e-6).log()
                    - agreement_mean.clamp_min(1e-6).log())
            ).sum(dim=-1)
            + (
                subgoal_reference_distribution
                * (
                    subgoal_reference_distribution.clamp_min(1e-6).log()
                    - agreement_mean.clamp_min(1e-6).log())
            ).sum(dim=-1)
        ).mean()
        referenced = reference_distribution[:, :-1]
        no_reference_prob = reference_distribution[:, -1]
        referenced_entity_summary = (slot_embed * referenced.unsqueeze(-1)).sum(dim=1)

        reference_confidence = referenced.sum(dim=-1)
        output = {
            "referenced_object_probs": referenced,
            "reference_distribution": reference_distribution,
            "query_reference_distribution": query_reference_distribution,
            "subgoal_reference_distribution": subgoal_reference_distribution,
            "grounding_consistency_loss": grounding_consistency_loss,
            "referenced_entity_summary": referenced_entity_summary,
            "reference_confidence": reference_confidence,
            "no_reference_prob": no_reference_prob,
            "semantic_reference_probs": referenced,
            "semantic_reference_distribution": reference_distribution,
            "semantic_reference_confidence": reference_confidence,}
        if "EntityId" in physicalState and "EntityGeneration" in physicalState:
            selected_slot = referenced.argmax(dim=-1)
            batch_index = torch.arange(
                B,
                device=selected_slot.device)
            selected_entity = physicalState["EntityId"][
                batch_index,
                selected_slot]
            selected_generation = physicalState["EntityGeneration"][
                batch_index,
                selected_slot]
            has_reference = reference_confidence > no_reference_prob
            output["referenced_entity_id"] = torch.where(
                has_reference,
                selected_entity,
                torch.full_like(selected_entity, -1))
            output["referenced_entity_generation"] = torch.where(
                has_reference,
                selected_generation,
                torch.full_like(selected_generation, -1))
        return output


class TemporalGoalHead(AGICoreModule):
    def __init__(
        self,
        shortGoalDim: int = ModuleDim.GoalShortDim,
        temporalContextDim: int = ModuleDim.TemporalContextDim,
        hidden: int = 128,
        defaultSoftTimeoutMs: float = 1000.0,
        defaultHardTimeoutMs: float = 5000.0,):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(shortGoalDim) + int(temporalContextDim)),
            nn.Linear(int(shortGoalDim) + int(temporalContextDim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),)

        self.mode_head = nn.Linear(hidden, ModuleDim.TemporalPrimitiveCount)
        self.soft_timeout_head = nn.Linear(hidden, 1)
        self.hard_timeout_grace_ms = (
            float(defaultHardTimeoutMs) - float(defaultSoftTimeoutMs))
        nn.init.zeros_(self.mode_head.weight)
        nn.init.zeros_(self.mode_head.bias)
        nn.init.zeros_(self.soft_timeout_head.weight)
        with torch.no_grad():
            soft_seconds = float(defaultSoftTimeoutMs) / 1000.0
            self.soft_timeout_head.bias.fill_(math.log(math.expm1(soft_seconds)))

    def forward(self, goalTemporal: torch.Tensor, temporalContextFeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(torch.cat([goalTemporal, temporalContextFeat], dim=-1))
        soft = F.softplus(self.soft_timeout_head(h)).squeeze(-1) * 1000.0
        hard = soft + self.hard_timeout_grace_ms
        return {
            "goal_mode_logits": self.mode_head(h),
            "goal_timeout_soft_ms": soft,
            "goal_timeout_hard_ms": hard,}


class HierarchicalGoalFusion(AGICoreModule):
    def __init__(
        self,
        ultimateDim: int = ModuleDim.GoalUltimateDim,
        longDim: int = ModuleDim.GoalLongDim,
        midDim: int = ModuleDim.GoalMidDim,
        shortDim: int = ModuleDim.GoalShortDim,
        fusionDim: int = ModuleDim.GoalShortDim,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        numHeads: int = 4,):
        super().__init__()
        self.fusion_dim = int(fusionDim)
        self.ultimate_proj = nn.Sequential(nn.LayerNorm(int(ultimateDim)), nn.Linear(int(ultimateDim), self.fusion_dim))
        self.long_proj = nn.Sequential(nn.LayerNorm(int(longDim)), nn.Linear(int(longDim), self.fusion_dim))
        self.mid_proj = nn.Sequential(nn.LayerNorm(int(midDim)), nn.Linear(int(midDim), self.fusion_dim))
        self.short_proj = nn.Sequential(nn.LayerNorm(int(shortDim)), nn.Linear(int(shortDim), self.fusion_dim))
        self.distill_norm = nn.LayerNorm(self.fusion_dim)
        self.register_buffer("level_decay", torch.tensor([0.125, 0.25, 0.5, 1.0]).view(1, 4, 1), persistent=False)
        self.role_query = nn.Parameter(torch.randn(3, self.fusion_dim) * 0.02)
        self.role_attn = nn.MultiheadAttention(self.fusion_dim, int(numHeads), batch_first=True)
        self.role_norm = nn.LayerNorm(self.fusion_dim)
        self.symbolic_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.SiLU(),
            nn.Linear(self.fusion_dim, int(shortDim)),)
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.fusion_dim),
            nn.SiLU(),
            nn.Linear(self.fusion_dim, int(shortDim)),)
        self.decision_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, int(decisionDim)),
            nn.SiLU(),
            nn.Linear(int(decisionDim), int(decisionDim)),
            nn.LayerNorm(int(decisionDim)),)

    def forward(
        self,
        gUltimate: torch.Tensor,
        gLong: torch.Tensor,
        gMid: torch.Tensor,
        gShort: torch.Tensor,) -> Dict[str, torch.Tensor]:
        ultimate = self.ultimate_proj(gUltimate)
        long = self.distill_norm(self.long_proj(gLong) + 0.5 * ultimate)
        mid = self.distill_norm(self.mid_proj(gMid) + 0.5 * long + 0.25 * ultimate)
        short = self.distill_norm(self.short_proj(gShort) + 0.5 * mid + 0.25 * long + 0.125 * ultimate)
        tokens = torch.stack([ultimate, long, mid, short], dim=1) * self.level_decay
        query = self.role_query.unsqueeze(0).expand(gShort.size(0), 3, self.fusion_dim)
        role, _ = self.role_attn(query, tokens, tokens)
        role = self.role_norm(role + query)
        return {
            "goal_symbolic": self.symbolic_head(role[:, 0] + short),
            "goal_temporal": self.temporal_head(role[:, 1] + 0.5 * (mid + short)),
            "goal_decision": self.decision_head(role[:, 2]),}


class FourLevelGoalManager(AGICoreModule):
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        worldLatentDim: int,
        pstSummaryDim: int = ModuleDim.PstSlotDim,
        intentDim: int = ModuleDim.IntentionFeat,
        ultimateDim: int = ModuleDim.GoalUltimateDim,
        longDim: int = ModuleDim.GoalLongDim,
        midDim: int = ModuleDim.GoalMidDim,
        shortDim: int = ModuleDim.GoalShortDim,
        taskRelationDim: int = 16,
        taskObjectDim: int = 32,
        capabilityDim: int = 32,
        taskContextDim: int = 128,):
        super().__init__()
        self.contract_view = contractView
        self.endpoint_count = int(contractView.end_effector_count)
        static_end_effector_tokens = torch.tensor(
            contractView.static_end_effector_tokens,
            dtype=torch.float32)
        if (
            static_end_effector_tokens.dim() != 2
            or int(static_end_effector_tokens.size(0)) != self.endpoint_count
            or int(static_end_effector_tokens.size(1)) < 1
        ):
            raise ValueError("goal contract end-effector descriptors are invalid")
        self.register_buffer(
            "static_end_effector_tokens",
            static_end_effector_tokens,
            persistent=False)
        self.register_buffer(
            "root_mask",
            torch.tensor(contractView.root_mask, dtype=torch.bool),
            persistent=False)
        self.parent_index = tuple(int(value) for value in contractView.parent_index)
        self.topological_layers = tuple(
            tuple(int(index) for index in layer)
            for layer in contractView.topological_layers)
        self.ultimate_dim = int(ultimateDim)
        self.long_dim = int(longDim)
        self.mid_dim = int(midDim)
        self.short_dim = int(shortDim)
        self.task_relation_dim = int(taskRelationDim)
        self.task_object_dim = int(taskObjectDim)
        self.capability_dim = int(capabilityDim)
        self.task_context_dim = int(taskContextDim)

        ctx_dim = worldLatentDim + pstSummaryDim + intentDim
        self.context_dim = int(ctx_dim)
        self.task_relation_encoder = nn.Sequential(
            nn.LayerNorm(self.task_relation_dim),
            nn.Linear(self.task_relation_dim, self.task_context_dim),
            nn.SiLU(),)
        self.task_object_encoder = nn.Sequential(
            nn.LayerNorm(self.task_object_dim),
            nn.Linear(self.task_object_dim, self.task_context_dim),
            nn.SiLU(),)
        self.requirement_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, self.task_context_dim),
            nn.SiLU(),)
        self.task_context_encoder = nn.Sequential(
            nn.LayerNorm(self.task_context_dim * 3),
            nn.Linear(self.task_context_dim * 3, self.task_context_dim * 2),
            nn.SiLU(),
            nn.Linear(self.task_context_dim * 2, self.task_context_dim),
            nn.LayerNorm(self.task_context_dim),)
        descriptor_dim = int(static_end_effector_tokens.size(-1))
        self.endpoint_capability_adapter = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, self.capability_dim),
            nn.SiLU(),)
        self.capability_encoder = nn.Sequential(
            nn.LayerNorm(self.capability_dim),
            nn.Linear(self.capability_dim, self.task_context_dim),
            nn.SiLU(),
            nn.Linear(self.task_context_dim, self.task_context_dim),)
        self.capability_query = nn.Sequential(
            nn.LayerNorm(self.task_context_dim),
            nn.Linear(self.task_context_dim, self.task_context_dim),)
        self.task_to_context = nn.Sequential(
            nn.LayerNorm(self.task_context_dim * 2),
            nn.Linear(self.task_context_dim * 2, self.context_dim),
            nn.Tanh(),)
        self.task_context_gain = nn.Parameter(
            torch.zeros(self.context_dim))
        self.ultimate_head = CodebookGoalHead(
            ctx_dim,
            ModuleDim.GoalUltimateCodebookGroups,
            ModuleDim.GoalUltimateCodebookCodes,
            self.ultimate_dim)

        self.long_head = CodebookGoalHead(
            ctx_dim + self.ultimate_dim,
            ModuleDim.GoalLongCodebookGroups,
            ModuleDim.GoalLongCodebookCodes,
            self.long_dim)

        self.mid_head = CodebookGoalHead(
            ctx_dim + self.ultimate_dim + self.long_dim,
            ModuleDim.GoalMidCodebookGroups,
            ModuleDim.GoalMidCodebookCodes,
            self.mid_dim)

        short_in = self.ultimate_dim + self.long_dim + self.mid_dim + pstSummaryDim
        self.short_head = nn.Sequential(
            nn.LayerNorm(short_in),
            nn.Linear(short_in, 256),
            nn.SiLU(),
            nn.Linear(256, self.short_dim),)

        self.mid_to_world = nn.Linear(self.mid_dim, worldLatentDim)
        self.goal_fusion = HierarchicalGoalFusion(
            ultimateDim=self.ultimate_dim,
            longDim=self.long_dim,
            midDim=self.mid_dim,
            shortDim=self.short_dim)
        self.temporal_goal_head = TemporalGoalHead(shortGoalDim=self.short_dim)

    @property
    def ContractView(self) -> RobotEmbodimentContractView:
        return self.contract_view

    def NormalizeRequirement(
        self,
        value: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,) -> torch.Tensor:
        if value is None:
            return torch.zeros(batchSize, device=device, dtype=dtype)
        normalized = torch.as_tensor(value, device=device, dtype=dtype)
        if normalized.dim() == 2 and normalized.size(-1) == 1:
            normalized = normalized.squeeze(-1)
        if normalized.shape != (batchSize,):
            raise ValueError(
                f"{name} must have shape [{batchSize}], got {tuple(normalized.shape)}")
        if not bool(torch.isfinite(normalized).all().item()):
            raise ValueError(f"{name} must be finite")
        return normalized.clamp(0.0, 1.0)

    def BuildTaskRequirements(
        self,
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,
        taskRelation: Optional[torch.Tensor],
        taskObject: Optional[torch.Tensor],
        precisionRequirement: Optional[torch.Tensor],
        timeRequirement: Optional[torch.Tensor],
        terminationRequirement: Optional[torch.Tensor],
        activePerceptionRequirement: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if taskRelation is None:
            relation = torch.zeros(
                batchSize,
                self.task_relation_dim,
                device=device,
                dtype=dtype)
        else:
            relation = taskRelation.to(device=device, dtype=dtype)
            if relation.shape != (batchSize, self.task_relation_dim):
                raise ValueError(
                    f"taskRelation must have shape [{batchSize}, {self.task_relation_dim}], got {tuple(relation.shape)}")
        if taskObject is None:
            task_object = torch.zeros(
                batchSize,
                self.task_object_dim,
                device=device,
                dtype=dtype)
        else:
            task_object = taskObject.to(device=device, dtype=dtype)
            if task_object.shape != (batchSize, self.task_object_dim):
                raise ValueError(
                    f"taskObject must have shape [{batchSize}, {self.task_object_dim}], got {tuple(task_object.shape)}")
        if not bool(torch.isfinite(relation).all().item()):
            raise ValueError("taskRelation must be finite")
        if not bool(torch.isfinite(task_object).all().item()):
            raise ValueError("taskObject must be finite")

        precision = self.NormalizeRequirement(
            precisionRequirement, batchSize, device, dtype, "precisionRequirement")
        time = self.NormalizeRequirement(
            timeRequirement, batchSize, device, dtype, "timeRequirement")
        termination = self.NormalizeRequirement(
            terminationRequirement, batchSize, device, dtype, "terminationRequirement")
        active_perception = self.NormalizeRequirement(
            activePerceptionRequirement,
            batchSize,
            device,
            dtype,
            "activePerceptionRequirement")
        requirements = torch.stack([
            precision,
            time,
            termination,
            active_perception,], dim=-1)
        relation_code = self.task_relation_encoder(relation)
        object_code = self.task_object_encoder(task_object)
        requirement_code = self.requirement_encoder(requirements)
        context = self.task_context_encoder(torch.cat([
            relation_code,
            object_code,
            requirement_code,], dim=-1))
        return {
            "task_context": context,
            "task_relation": relation,
            "task_object": task_object,
            "precision_requirement": precision,
            "time_requirement": time,
            "termination_requirement": termination,
            "active_perception_requirement": active_perception,}

    def BindCapabilities(
        self,
        taskContext: torch.Tensor,
        endpointAvailable: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        B = int(taskContext.size(0))
        descriptors = self.endpoint_capability_adapter(
            self.static_end_effector_tokens.to(
                device=taskContext.device,
                dtype=taskContext.dtype)).unsqueeze(0).expand(B, -1, -1)
        if endpointAvailable is None:
            available = torch.ones(
                B,
                self.endpoint_count,
                dtype=torch.bool,
                device=taskContext.device)
        else:
            available = endpointAvailable.to(
                device=taskContext.device,
                dtype=torch.bool)
            if available.shape != (B, self.endpoint_count):
                raise ValueError(
                    f"endpointAvailable must have shape [{B}, {self.endpoint_count}], got {tuple(available.shape)}")
        codes = self.capability_encoder(descriptors)
        query = self.capability_query(taskContext)
        logits = torch.einsum("bd,bnd->bn", query, codes)
        logits = logits / math.sqrt(float(self.task_context_dim))
        relevance = F.softmax(
            logits.masked_fill(
                ~available,
                torch.finfo(logits.dtype).min),
            dim=-1)
        relevance = relevance * available.to(dtype=relevance.dtype)
        relevance = relevance / relevance.sum(
            dim=-1,
            keepdim=True).clamp_min(max(
                1e-8,
                float(torch.finfo(relevance.dtype).tiny)))
        summary = torch.einsum("bn,bnd->bd", relevance, codes)
        return {
            "capability_relevance": relevance,
            "capability_summary": summary,}

    def ResolveEndpointActivity(
        self,
        endpointAvailable: Optional[torch.Tensor],
        hierarchyEnabled: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (int(batchSize), self.endpoint_count)
        if endpointAvailable is None:
            available = torch.ones(shape, dtype=torch.bool, device=device)
        else:
            available = endpointAvailable.to(device=device, dtype=torch.bool)
            if tuple(available.shape) != shape:
                raise ValueError("endpointAvailable does not match the contract")
        if hierarchyEnabled is None:
            enabled = self.root_mask.to(device=device).unsqueeze(0).expand(shape)
        else:
            enabled = hierarchyEnabled.to(device=device, dtype=torch.bool)
            if tuple(enabled.shape) != shape:
                raise ValueError("hierarchyEnabled does not match the contract")
            enabled = enabled | self.root_mask.to(
                device=device).unsqueeze(0)
        active = available & enabled
        for endpointIndex, parentIndex in enumerate(self.parent_index):
            if parentIndex >= 0 and bool(
                (active[:, endpointIndex] & ~active[:, parentIndex]).any().item()
            ):
                raise ValueError("active child endpoints require an active parent")
        return active

    def BuildSubtreeState(
        self,
        endpointRelevance: torch.Tensor,
        endpointActive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            endpointRelevance.dim() != 2
            or tuple(endpointRelevance.shape) != tuple(endpointActive.shape)
            or int(endpointRelevance.size(-1)) != self.endpoint_count
            or not endpointRelevance.is_floating_point()
            or endpointActive.dtype != torch.bool
            or endpointRelevance.device != endpointActive.device
            or not bool(torch.isfinite(endpointRelevance).all().item())
            or bool((endpointRelevance < 0.0).any().item())
        ):
            raise ValueError("endpoint subtree state is invalid")
        subtree_relevance = endpointRelevance.clone()
        subtree_active = endpointActive.clone()
        for layer in reversed(self.topological_layers):
            for endpointIndex in layer:
                parentIndex = self.parent_index[endpointIndex]
                if parentIndex < 0:
                    continue
                subtree_relevance[:, parentIndex] = (
                    subtree_relevance[:, parentIndex]
                    + subtree_relevance[:, endpointIndex])
                subtree_active[:, parentIndex] = (
                    subtree_active[:, parentIndex]
                    | subtree_active[:, endpointIndex])
        return subtree_relevance, subtree_active

    def forward(
        self,
        worldLatent: torch.Tensor, # [B, WorldFeat]
        pstSummary: torch.Tensor, # [B, PstSlotDim]
        intentEmbed: torch.Tensor, # [B, IntentionFeat]
        *,
        taskRelation: Optional[torch.Tensor] = None,
        taskObject: Optional[torch.Tensor] = None,
        precisionRequirement: Optional[torch.Tensor] = None,
        timeRequirement: Optional[torch.Tensor] = None,
        terminationRequirement: Optional[torch.Tensor] = None,
        activePerceptionRequirement: Optional[torch.Tensor] = None,
        endpointAvailable: Optional[torch.Tensor] = None,
        hierarchyEnabled: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        ctx = torch.cat([worldLatent, pstSummary, intentEmbed], dim=-1)
        task = self.BuildTaskRequirements(
            batchSize=int(ctx.size(0)),
            device=ctx.device,
            dtype=ctx.dtype,
            taskRelation=taskRelation,
            taskObject=taskObject,
            precisionRequirement=precisionRequirement,
            timeRequirement=timeRequirement,
            terminationRequirement=terminationRequirement,
            activePerceptionRequirement=activePerceptionRequirement)
        capability = self.BindCapabilities(
            taskContext=task["task_context"],
            endpointAvailable=endpointAvailable)
        has_task = any(value is not None for value in (
            taskRelation,
            taskObject,
            precisionRequirement,
            timeRequirement,
            terminationRequirement,
            activePerceptionRequirement,
            endpointAvailable,))
        if has_task:
            task_residual = self.task_to_context(torch.cat([
                task["task_context"],
                capability["capability_summary"],], dim=-1))
            ctx = ctx + torch.tanh(
                self.task_context_gain).unsqueeze(0) * task_residual
        ultimate_out = self.ultimate_head(ctx)

        long_out = self.long_head(torch.cat([ctx, ultimate_out["goal"]], dim=-1))
        mid_out = self.mid_head(torch.cat([ctx, ultimate_out["goal"], long_out["goal"]], dim=-1))
        g_short = self.short_head(torch.cat([ultimate_out["goal"], long_out["goal"], mid_out["goal"], pstSummary], dim=-1))
        fused = self.FuseGoals(ultimate_out["goal"], long_out["goal"], mid_out["goal"], g_short)

        output = {
            "g_ultimate": ultimate_out["goal"],
            "g_long": long_out["goal"],
            "g_mid": mid_out["goal"],
            "g_short": g_short,
            "goal_symbolic": fused["goal_symbolic"],
            "goal_temporal": fused["goal_temporal"],
            "goal_decision": fused["goal_decision"],
            "ultimate_logits": ultimate_out["logits"],
            "long_logits": long_out["logits"],
            "mid_logits": mid_out["logits"],}
        output.update(task)
        output.update(capability)
        endpoint_active = self.ResolveEndpointActivity(
            endpointAvailable=endpointAvailable,
            hierarchyEnabled=hierarchyEnabled,
            batchSize=int(ctx.size(0)),
            device=ctx.device)
        subtree_relevance, subtree_active = self.BuildSubtreeState(
            capability["capability_relevance"],
            endpoint_active)
        output["endpoint_relevance"] = capability["capability_relevance"]
        output["subtree_relevance"] = subtree_relevance
        output["endpoint_active"] = endpoint_active
        output["subtree_active"] = subtree_active
        return output

    def TemporalGoal(self, goalTemporal: torch.Tensor, temporalContextFeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.temporal_goal_head(goalTemporal, temporalContextFeat)

    def ShortGoal(
        self,
        gUltimate: torch.Tensor,
        gLong: torch.Tensor,
        gMid: torch.Tensor,
        pstSummary: torch.Tensor,) -> torch.Tensor:
        return self.short_head(torch.cat([gUltimate, gLong, gMid, pstSummary], dim=-1))

    def FuseGoals(
        self,
        gUltimate: torch.Tensor,
        gLong: torch.Tensor,
        gMid: torch.Tensor,
        gShort: torch.Tensor,) -> Dict[str, torch.Tensor]:
        return self.goal_fusion(gUltimate, gLong, gMid, gShort)

    def ProjectedProgress(self, worldDelta: torch.Tensor, gMid: torch.Tensor) -> torch.Tensor:
        direction = F.normalize(self.mid_to_world(gMid), dim=-1, eps=1e-6)
        return (worldDelta * direction).sum(dim=-1)


class TestGoalMTool:
    def __init__(
        self,
        contractView: RobotEmbodimentContractView,
        device=None,
    ):
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("goal tests require an embodiment contract view")
        self.contract_view = contractView
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def WorldLatentDim(self) -> int:
        return ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState

    def MakeManager(self) -> FourLevelGoalManager:
        return FourLevelGoalManager(
            contractView=self.contract_view,
            worldLatentDim=self.WorldLatentDim(),
            pstSummaryDim=ModuleDim.PstSlotDim,
            intentDim=ModuleDim.IntentionFeat).to(self.device)

    def MakeGoalInputs(self, B: int = 2) -> Dict[str, torch.Tensor]:
        return {
            "worldLatent": torch.randn(B, self.WorldLatentDim(), device=self.device),
            "pstSummary": torch.randn(B, ModuleDim.PstSlotDim, device=self.device),
            "intentEmbed": torch.randn(B, ModuleDim.IntentionFeat, device=self.device),}

    def MakeGroundingState(self, B: int, K: int) -> Dict[str, torch.Tensor]:
        realm = torch.zeros(B, K, 5, device=self.device)
        realm[..., 0] = 1.0
        agency = torch.zeros(B, K, 5, device=self.device)
        agency[..., 4] = 1.0
        motion_layer = torch.zeros(B, K, 5, device=self.device)
        layer_agency = torch.zeros(B, K, 5, 5, device=self.device)
        layer_agency[..., 4] = 1.0
        self_part = torch.zeros(
            B, K, ModuleDim.PstSelfPartSemanticDim, device=self.device)
        surface_parent = torch.zeros(B, K, K + 1, device=self.device)
        surface_parent[..., K] = 1.0
        return {
            "SlotState": torch.randn(B, K, ModuleDim.PstSlotDim, device=self.device),
            "U": torch.randn(B, K, ModuleDim.PstUsageDim, device=self.device),
            "MphysRaw": torch.ones(B, K, device=self.device),
            "Observed": torch.ones(B, K, device=self.device),
            "LastSeen": torch.full((B, K), 4, device=self.device),
            "Step": torch.full((B,), 4, device=self.device, dtype=torch.long),
            "SlotPresence": torch.ones(B, K, device=self.device),
            "ObservedSlotMask": torch.ones(B, K, device=self.device),
            "PerceptualPresence": torch.ones(B, K, device=self.device),
            "PhysicalInteractionProb": torch.ones(B, K, device=self.device),
            "RealmProb": realm,
            "MotionLayerProb": motion_layer,
            "LayerAgencyProb": layer_agency,
            "AgencyProb": agency,
            "BodyMembershipProb": torch.zeros(B, K, device=self.device),
            "SelfPartSemantic": self_part,
            "SurfaceParentProb": surface_parent,
            "SurfaceUV": torch.full((B, K, 2), 0.5, device=self.device),
            "SurfaceUVConfidence": torch.ones(B, K, device=self.device),
            "DisplaySurfaceProb": torch.ones(B, K, device=self.device),
            "VerificationConfidence": torch.ones(B, K, device=self.device),
            "EntityTextSemantic": torch.randn(B, K, 512, device=self.device),
            "EntityTextConfidence": torch.ones(B, K, device=self.device),
            "EntityTextRevision": torch.ones(
                B, K, device=self.device, dtype=torch.long),
            "EntityTextChanged": torch.ones(
                B, K, device=self.device, dtype=torch.bool),
            "EntityId": torch.arange(
                K, device=self.device, dtype=torch.long).view(1, K).expand(B, K),
            "EntityGeneration": torch.ones(
                B, K, device=self.device, dtype=torch.long),}

    def AssertFinite(self, value: torch.Tensor, name: str) -> None:
        assert torch.isfinite(value).all(), f"{name} contains non-finite values"

    def TestFourLevelForwardShapes(self) -> bool:
        try:
            B = 2
            manager = self.MakeManager().eval()
            inputs = self.MakeGoalInputs(B)
            with torch.no_grad():
                out = manager(**inputs)
            assert tuple(out["g_ultimate"].shape) == (B, ModuleDim.GoalUltimateDim)
            assert tuple(out["g_long"].shape) == (B, ModuleDim.GoalLongDim)
            assert tuple(out["g_mid"].shape) == (B, ModuleDim.GoalMidDim)
            assert tuple(out["g_short"].shape) == (B, ModuleDim.GoalShortDim)
            assert tuple(out["goal_symbolic"].shape) == (B, ModuleDim.GoalShortDim)
            assert tuple(out["goal_temporal"].shape) == (B, ModuleDim.GoalShortDim)
            assert tuple(out["goal_decision"].shape) == (B, ModuleDim.DecisionBeliefDim)
            assert tuple(out["ultimate_logits"].shape) == (
                B,
                ModuleDim.GoalUltimateCodebookGroups,
                ModuleDim.GoalUltimateCodebookCodes)
            assert tuple(out["long_logits"].shape) == (
                B,
                ModuleDim.GoalLongCodebookGroups,
                ModuleDim.GoalLongCodebookCodes)
            assert tuple(out["mid_logits"].shape) == (
                B,
                ModuleDim.GoalMidCodebookGroups,
                ModuleDim.GoalMidCodebookCodes)
            for name, value in out.items():
                self.AssertFinite(value.float(), f"FourLevelGoalManager {name}")
            print("FourLevelGoalManager forward shape test passed.")
            return True
        except Exception as e:
            print(f"FourLevelGoalManager forward shape test failed: {type(e).__name__}: {e}")
            return False

    def TestShortGoalFastPathMatchesForward(self) -> bool:
        try:
            B = 2
            manager = self.MakeManager().eval()
            inputs = self.MakeGoalInputs(B)
            with torch.no_grad():
                out = manager(**inputs)
                fast = manager.ShortGoal(
                    out["g_ultimate"],
                    out["g_long"],
                    out["g_mid"],
                    inputs["pstSummary"])
            assert torch.allclose(fast, out["g_short"], atol=1e-6)
            print("FourLevelGoalManager ShortGoal fast-path test passed.")
            return True
        except Exception as e:
            print(f"FourLevelGoalManager ShortGoal fast-path test failed: {type(e).__name__}: {e}")
            return False

    def TestTemporalGoalShapes(self) -> bool:
        try:
            B = 2
            manager = self.MakeManager().eval()
            inputs = self.MakeGoalInputs(B)
            temporal_context = torch.randn(B, ModuleDim.TemporalContextDim, device=self.device)
            with torch.no_grad():
                goals = manager(**inputs)
                out = manager.TemporalGoal(goals["goal_temporal"], temporal_context)
            assert tuple(out["goal_mode_logits"].shape) == (B, ModuleDim.TemporalPrimitiveCount)
            assert tuple(out["goal_timeout_soft_ms"].shape) == (B,)
            assert tuple(out["goal_timeout_hard_ms"].shape) == (B,)
            assert torch.count_nonzero(out["goal_mode_logits"]).item() == 0
            assert not hasattr(manager.temporal_goal_head, "hard_timeout_head")
            assert manager.temporal_goal_head.hard_timeout_grace_ms == 4000.0
            assert torch.allclose(
                out["goal_timeout_soft_ms"],
                torch.full_like(out["goal_timeout_soft_ms"], 1000.0),
                atol=1e-4)
            assert torch.allclose(
                out["goal_timeout_hard_ms"],
                torch.full_like(out["goal_timeout_hard_ms"], 5000.0),
                atol=1e-4)
            for name, value in out.items():
                self.AssertFinite(value, f"TemporalGoal {name}")
            print("TemporalGoal shape test passed.")
            return True
        except Exception as e:
            print(f"TemporalGoal shape test failed: {type(e).__name__}: {e}")
            return False

    def TestTemporalTimeoutGradientSemantics(self) -> bool:
        try:
            B = 2
            head = TemporalGoalHead(
                shortGoalDim=8,
                temporalContextDim=ModuleDim.TemporalContextDim,
                hidden=16).to(self.device)
            out = head(
                torch.randn(B, 8, device=self.device),
                torch.randn(B, ModuleDim.TemporalContextDim, device=self.device))
            out["goal_timeout_soft_ms"].mean().backward()
            assert head.soft_timeout_head.weight.grad is not None
            assert head.soft_timeout_head.bias.grad is not None
            assert not hasattr(head, "hard_timeout_head")
            assert torch.allclose(
                out["goal_timeout_hard_ms"] - out["goal_timeout_soft_ms"],
                torch.full_like(out["goal_timeout_soft_ms"], 4000.0))
            print("TemporalGoal timeout gradient semantics passed.")
            return True
        except Exception as e:
            print(f"TemporalGoal timeout gradient semantics failed: {type(e).__name__}: {e}")
            return False

    def TestGoalGroundingShapes(self) -> bool:
        try:
            B, K = 2, 4
            grounding = GoalGrounding().to(self.device).eval()
            goal = torch.randn(B, ModuleDim.GoalShortDim, device=self.device)
            intent = torch.randn(B, ModuleDim.IntentionFeat, device=self.device)
            physical_state = self.MakeGroundingState(B, K)
            physical_state["Observed"] = torch.tensor(
                [[1, 1, 0, 0], [1, 0, 1, 0]], device=self.device)
            physical_state["LastSeen"] = torch.tensor(
                [[8, 8, 4, 0], [8, 3, 8, 0]], device=self.device)
            physical_state["Step"] = torch.full(
                (B,), 8, device=self.device, dtype=torch.long)
            with torch.no_grad():
                out = grounding(goal, intent, physical_state, physical_state)
            assert tuple(out["referenced_object_probs"].shape) == (B, K)
            assert tuple(out["reference_distribution"].shape) == (B, K + 1)
            assert tuple(out["referenced_entity_summary"].shape) == (B, ModuleDim.PstSlotDim)
            assert tuple(out["reference_confidence"].shape) == (B,)
            assert tuple(out["no_reference_prob"].shape) == (B,)
            for name, value in out.items():
                self.AssertFinite(value, f"GoalGrounding {name}")
            print("GoalGrounding shape test passed.")
            return True
        except Exception as e:
            print(f"GoalGrounding shape test failed: {type(e).__name__}: {e}")
            return False

    def TestGoalManagerBackward(self) -> bool:
        try:
            manager = self.MakeManager()
            inputs = self.MakeGoalInputs(B=2)
            out = manager(**inputs)
            progress = manager.ProjectedProgress(
                torch.randn(2, self.WorldLatentDim(), device=self.device),
                out["g_mid"])
            loss = (
                out["g_short"].square().mean()
                + 0.01 * out["goal_symbolic"].square().mean()
                + 0.01 * out["goal_temporal"].square().mean()
                + 0.01 * out["goal_decision"].square().mean()
                + 0.01 * manager.ultimate_head.UtilizationLoss(out["ultimate_logits"])
                + 0.01 * manager.long_head.UtilizationLoss(out["long_logits"])
                + 0.01 * manager.mid_head.UtilizationLoss(out["mid_logits"])
                - 0.01 * progress.mean())
            loss.backward()
            grad_norm = sum(
                float(p.grad.detach().abs().sum().item())
                for p in manager.parameters()
                if p.grad is not None)
            assert grad_norm > 0.0
            print("FourLevelGoalManager backward test passed.")
            return True
        except Exception as e:
            print(f"FourLevelGoalManager backward test failed: {type(e).__name__}: {e}")
            return False

    def TestAbstractTaskRequirementsAndExternalActivity(self) -> bool:
        try:
            B = 2
            manager = FourLevelGoalManager(
                contractView=self.contract_view,
                worldLatentDim=self.WorldLatentDim(),
                pstSummaryDim=ModuleDim.PstSlotDim,
                intentDim=ModuleDim.IntentionFeat,
                taskRelationDim=6,
                taskObjectDim=7,
                capabilityDim=5).to(self.device)
            N = manager.endpoint_count
            inputs = self.MakeGoalInputs(B)
            relation = torch.randn(
                B, 6, device=self.device, requires_grad=True)
            task_object = torch.randn(
                B, 7, device=self.device, requires_grad=True)
            available = torch.ones(
                B, N, device=self.device, dtype=torch.bool)
            enabled = manager.root_mask.to(
                device=self.device).unsqueeze(0).expand(B, -1).clone()
            for endpointIndex, parentIndex in enumerate(manager.parent_index):
                if parentIndex >= 0:
                    enabled[:, endpointIndex] = True
                    break
            out = manager(
                **inputs,
                taskRelation=relation,
                taskObject=task_object,
                precisionRequirement=torch.tensor(
                    [[0.8], [0.4]], device=self.device),
                timeRequirement=torch.tensor(
                    [[0.6], [0.3]], device=self.device),
                terminationRequirement=torch.tensor(
                    [[0.7], [0.5]], device=self.device),
                activePerceptionRequirement=torch.tensor(
                    [[0.9], [0.2]], device=self.device),
                endpointAvailable=available,
                hierarchyEnabled=enabled)
            expected_active = available & enabled
            assert tuple(out["task_context"].shape) == (
                B, manager.task_context_dim)
            assert tuple(out["capability_relevance"].shape) == (B, N)
            assert tuple(out["capability_summary"].shape) == (
                B, manager.task_context_dim)
            assert torch.equal(out["endpoint_active"], expected_active)
            assert torch.count_nonzero(
                out["capability_relevance"].masked_select(~available)).item() == 0
            assert torch.allclose(
                out["precision_requirement"],
                torch.tensor([0.8, 0.4], device=self.device))
            loss = (
                out["g_short"].square().mean()
                + out["task_context"].square().mean()
                + out["capability_summary"].square().mean())
            loss.backward()
            assert relation.grad is not None
            assert task_object.grad is not None
            assert float(relation.grad.abs().sum().item()) > 0.0
            assert float(task_object.grad.abs().sum().item()) > 0.0
            capability_grad = sum(
                float(parameter.grad.abs().sum().item())
                for parameter in manager.endpoint_capability_adapter.parameters()
                if parameter.grad is not None)
            assert capability_grad > 0.0
            print("AbstractTaskRequirementsAndExternalActivity passed.")
            return True
        except Exception as e:
            print(
                f"AbstractTaskRequirementsAndExternalActivity failed: {type(e).__name__}: {e}")
            return False

    def TestHardCodebookCollapsePenalty(self) -> bool:
        try:
            codes = 4
            head = CodebookGoalHead(
                contextDim=8,
                groups=1,
                codes=codes,
                goalDim=8,
                hidden=8).to(self.device)
            collapsed_logits = torch.zeros(
                codes, 1, codes, device=self.device, requires_grad=True)
            collapsed_loss = head.UtilizationLoss(collapsed_logits)
            collapsed_loss.backward()
            balanced_logits = torch.full(
                (codes, 1, codes), -8.0, device=self.device)
            balanced_logits[
                torch.arange(codes, device=self.device),
                0,
                torch.arange(codes, device=self.device)] = 8.0
            balanced_loss = head.UtilizationLoss(balanced_logits)
            ok = bool(
                collapsed_loss.item() > balanced_loss.item()
                and collapsed_logits.grad is not None
                and collapsed_logits.grad.abs().sum().item() > 0.0)
            print(f"HardCodebookCollapsePenalty {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"HardCodebookCollapsePenalty failed: {type(e).__name__}: {e}")
            return False

    def TestGroundingConsistencyGradient(self) -> bool:
        try:
            B, K = 2, 4
            grounding = GoalGrounding().to(self.device).train()
            physical_state = self.MakeGroundingState(B, K)
            out = grounding(
                torch.randn(B, ModuleDim.GoalShortDim, device=self.device),
                torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
                physical_state,
                physical_state)
            out["grounding_consistency_loss"].backward()
            query_grad = sum(
                float(parameter.grad.abs().sum().item())
                for parameter in grounding.goal_intent_proj.parameters()
                if parameter.grad is not None)
            subgoal_grad = sum(
                float(parameter.grad.abs().sum().item())
                for parameter in grounding.decomposer.parameters()
                if parameter.grad is not None)

            no_slot_state = dict(physical_state)
            no_slot_state["MphysRaw"] = torch.zeros(B, K, device=self.device)
            no_slot_state["Observed"] = torch.zeros(B, K, device=self.device)
            no_slot_state["SlotPresence"] = torch.zeros(B, K, device=self.device)
            no_slot_state["ObservedSlotMask"] = torch.zeros(B, K, device=self.device)
            no_slot_state["PerceptualPresence"] = torch.zeros(B, K, device=self.device)
            no_slot = grounding(
                torch.randn(B, ModuleDim.GoalShortDim, device=self.device),
                torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
                no_slot_state,
                no_slot_state)
            ok = bool(
                query_grad > 0.0
                and subgoal_grad > 0.0
                and torch.allclose(
                    no_slot["no_reference_prob"],
                    torch.ones_like(no_slot["no_reference_prob"]),
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    no_slot["referenced_entity_summary"],
                    torch.zeros_like(no_slot["referenced_entity_summary"]),
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    no_slot["grounding_consistency_loss"],
                    torch.zeros_like(no_slot["grounding_consistency_loss"]),
                    atol=1e-6,
                    rtol=1e-6))
            print(f"GroundingConsistencyGradient {'passed' if ok else 'failed'}.")
            return ok
        except Exception as e:
            print(f"GroundingConsistencyGradient failed: {type(e).__name__}: {e}")
            return False

    def TestEntityTextGrounding(self) -> bool:
        try:
            B, K = 2, 4
            grounding = GoalGrounding().to(self.device).train()
            state = self.MakeGroundingState(B, K)
            state["EntityTextConfidence"][:, 0] = 0.0
            slot_tensor = state["SlotState"]
            optimizer = torch.optim.SGD(grounding.parameters(), lr=1e-2)
            neutral = grounding.BuildEntityTextSlotContext(state, slot_tensor)
            assert torch.count_nonzero(neutral).item() == 0
            optimizer.zero_grad(set_to_none=True)
            neutral.sum().backward()
            optimizer.step()

            grounding.zero_grad(set_to_none=True)
            enriched = grounding.BuildEntityTextSlotContext(state, slot_tensor)
            assert torch.count_nonzero(enriched[:, 0]).item() == 0
            assert torch.count_nonzero(enriched[:, 1:]).item() > 0
            enriched[:, 1:].square().mean().backward()
            assert grounding.entity_text_slot_encoder[1].weight.grad is not None
            assert grounding.entity_text_slot_residual[1].weight.grad is not None
            assert grounding.entity_text_slot_gate[1].weight.grad is not None
            assert grounding.entity_text_slot_gain.grad is not None
            output = grounding(
                torch.randn(B, ModuleDim.GoalShortDim, device=self.device),
                torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
                state,
                state)
            assert output["referenced_entity_id"].shape == (B,)
            assert output["referenced_entity_generation"].shape == (B,)
            print("EntityTextGrounding passed.")
            return True
        except Exception as e:
            print(f"EntityTextGrounding failed: {type(e).__name__}: {e}")
            return False

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "FourLevelForwardShapes": self.TestFourLevelForwardShapes(),
            "ShortGoalFastPathMatchesForward": self.TestShortGoalFastPathMatchesForward(),
            "TemporalGoalShapes": self.TestTemporalGoalShapes(),
            "TemporalTimeoutGradientSemantics": self.TestTemporalTimeoutGradientSemantics(),
            "GoalGroundingShapes": self.TestGoalGroundingShapes(),
            "HardCodebookCollapsePenalty": self.TestHardCodebookCollapsePenalty(),
            "GroundingConsistencyGradient": self.TestGroundingConsistencyGradient(),
            "EntityTextGrounding": self.TestEntityTextGrounding(),
            "AbstractTaskRequirementsAndExternalActivity": self.TestAbstractTaskRequirementsAndExternalActivity(),
            "GoalManagerBackward": self.TestGoalManagerBackward(),}
        passed = sum(1 for value in results.values() if value)
        print(f"\n[GoalModule Tests] {passed}/{len(results)} passed.")
        return results
