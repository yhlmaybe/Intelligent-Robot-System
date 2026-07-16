from __future__ import annotations
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import AGICoreModule, BuildReferenceWeights, BuildReferenceScaleContext
from ModuleMessagerManager import ModuleDim


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
        prob = F.softmax(logits, dim=-1).mean(dim=0)
        soft_balance = (
            math.log(self.codes)
            + (prob * prob.clamp_min(1e-8).log()).sum(dim=-1)).mean()

        sample_prob = F.softmax(logits, dim=-1)
        hard_index = sample_prob.argmax(dim=-1, keepdim=True)
        hard = torch.zeros_like(sample_prob).scatter_(-1, hard_index, 1.0)
        hard_st = hard + sample_prob - sample_prob.detach()
        hard_usage = hard_st.mean(dim=0)
        hard_balance = (
            math.log(self.codes)
            + (hard_usage * hard_usage.clamp_min(1e-8).log()).sum(dim=-1)
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

    def forward(
        self,
        goalEmbed: torch.Tensor,
        intentEmbed: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        observedPhysicalState: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        slot_tensor = physicalState["SlotState"]
        current_step = physicalState["Step"].view(-1, 1).float()
        goal_intent = torch.cat([goalEmbed, intentEmbed], dim=-1)
        query_vec = self.goal_intent_proj(goal_intent)
        reference_context = BuildReferenceScaleContext(
            observedPhysicalState,
            query_vec) # [B, 8]

        memory_scale = torch.sigmoid(self.reference_memory_scale_head(torch.cat([goal_intent, reference_context], dim=-1)))

        reference_weights = BuildReferenceWeights(
            physicalState,
            current_step,
            memoryScale=memory_scale,
            memoryDecayHorizon=32.0)

        observed_weight = reference_weights.observed_weight
        memory_weight = reference_weights.memory_weight
        memory_recency = reference_weights.memory_recency
        slot_weight = reference_weights.slot_weight

        slot_input = torch.cat([
            slot_tensor,
            physicalState["U"],
            observed_weight.unsqueeze(-1),
            memory_weight.unsqueeze(-1),
            slot_weight.unsqueeze(-1),
            memory_recency.unsqueeze(-1),], dim=-1)

        slot_embed = self.slot_ground_encoder(slot_input) # [B, K, D]
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
        query_slot_logits = query_slot_logits.masked_fill(invalid_slot, -1e9)
        subgoal_slot_logits = subgoal_slot_logits.masked_fill(invalid_slot, -1e9)
        slot_logits = query_slot_logits + subgoal_slot_logits - reference_prior
        slot_logits = slot_logits.masked_fill(invalid_slot, -1e9)

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
        no_slot_prob = reference_distribution[:, -1]
        referenced_slot_summary = (slot_embed * referenced.unsqueeze(-1)).sum(dim=1)

        reference_confidence = referenced.sum(dim=-1)

        return {
            "referenced_object_probs": referenced,
            "reference_distribution": reference_distribution,
            "query_reference_distribution": query_reference_distribution,
            "subgoal_reference_distribution": subgoal_reference_distribution,
            "grounding_consistency_loss": grounding_consistency_loss,
            "referenced_slot_summary": referenced_slot_summary,
            "reference_confidence": reference_confidence,
            "no_slot_prob": no_slot_prob,}


class TemporalGoalHead(AGICoreModule):
    """Predict nominal/soft duration and derive a policy hard-deadline grace.

    Synthetic supervision supplies an explicitly-valid soft-duration label.
    There is no independent hard-timeout label, so the hard deadline is
    derived as a fixed grace beyond the learned soft timeout.
    """

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
        worldLatentDim: int,
        pstSummaryDim: int = ModuleDim.PstSlotDim,
        intentDim: int = ModuleDim.IntentionFeat,
        ultimateDim: int = ModuleDim.GoalUltimateDim,
        longDim: int = ModuleDim.GoalLongDim,
        midDim: int = ModuleDim.GoalMidDim,
        shortDim: int = ModuleDim.GoalShortDim,):
        super().__init__()
        self.ultimate_dim = int(ultimateDim)
        self.long_dim = int(longDim)
        self.mid_dim = int(midDim)
        self.short_dim = int(shortDim)

        ctx_dim = worldLatentDim + pstSummaryDim + intentDim
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

    def forward(
        self,
        worldLatent: torch.Tensor, # [B, WorldFeat]
        pstSummary: torch.Tensor, # [B, PstSlotDim]
        intentEmbed: torch.Tensor, # [B, IntentionFeat]
        ) -> Dict[str, torch.Tensor]:
        ctx = torch.cat([worldLatent, pstSummary, intentEmbed], dim=-1)
        ultimate_out = self.ultimate_head(ctx)

        long_out = self.long_head(torch.cat([ctx, ultimate_out["goal"]], dim=-1))
        mid_out = self.mid_head(torch.cat([ctx, ultimate_out["goal"], long_out["goal"]], dim=-1))
        g_short = self.short_head(torch.cat([ultimate_out["goal"], long_out["goal"], mid_out["goal"], pstSummary], dim=-1))
        fused = self.FuseGoals(ultimate_out["goal"], long_out["goal"], mid_out["goal"], g_short)

        return {
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
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def WorldLatentDim(self) -> int:
        return ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState

    def MakeManager(self) -> FourLevelGoalManager:
        return FourLevelGoalManager(
            worldLatentDim=self.WorldLatentDim(),
            pstSummaryDim=ModuleDim.PstSlotDim,
            intentDim=ModuleDim.IntentionFeat).to(self.device)

    def MakeGoalInputs(self, B: int = 2) -> Dict[str, torch.Tensor]:
        return {
            "worldLatent": torch.randn(B, self.WorldLatentDim(), device=self.device),
            "pstSummary": torch.randn(B, ModuleDim.PstSlotDim, device=self.device),
            "intentEmbed": torch.randn(B, ModuleDim.IntentionFeat, device=self.device),}

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
            physical_state = {
                "SlotState": torch.randn(B, K, ModuleDim.PstSlotDim, device=self.device),
                "U": torch.randn(B, K, ModuleDim.PstUsageDim, device=self.device),
                "MphysRaw": torch.ones(B, K, device=self.device),
                "Observed": torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], device=self.device),
                "LastSeen": torch.tensor([[8, 8, 4, 0], [8, 3, 8, 0]], device=self.device),
                "Step": torch.full((B,), 8, device=self.device, dtype=torch.long),
                "SlotPresence": torch.ones(B, K, device=self.device),
                "ObservedSlotMask": torch.ones(B, K, device=self.device),}
            with torch.no_grad():
                out = grounding(goal, intent, physical_state, physical_state)
            assert tuple(out["referenced_object_probs"].shape) == (B, K)
            assert tuple(out["reference_distribution"].shape) == (B, K + 1)
            assert tuple(out["referenced_slot_summary"].shape) == (B, ModuleDim.PstSlotDim)
            assert tuple(out["reference_confidence"].shape) == (B,)
            assert tuple(out["no_slot_prob"].shape) == (B,)
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
            physical_state = {
                "SlotState": torch.randn(B, K, ModuleDim.PstSlotDim, device=self.device),
                "U": torch.randn(B, K, ModuleDim.PstUsageDim, device=self.device),
                "MphysRaw": torch.ones(B, K, device=self.device),
                "Observed": torch.ones(B, K, device=self.device),
                "LastSeen": torch.full((B, K), 4, device=self.device),
                "Step": torch.full((B,), 4, device=self.device, dtype=torch.long),
                "SlotPresence": torch.ones(B, K, device=self.device),
                "ObservedSlotMask": torch.ones(B, K, device=self.device),}
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
            no_slot = grounding(
                torch.randn(B, ModuleDim.GoalShortDim, device=self.device),
                torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
                no_slot_state,
                no_slot_state)
            ok = bool(
                query_grad > 0.0
                and subgoal_grad > 0.0
                and torch.allclose(
                    no_slot["no_slot_prob"],
                    torch.ones_like(no_slot["no_slot_prob"]),
                    atol=1e-6,
                    rtol=1e-6)
                and torch.allclose(
                    no_slot["referenced_slot_summary"],
                    torch.zeros_like(no_slot["referenced_slot_summary"]),
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

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "FourLevelForwardShapes": self.TestFourLevelForwardShapes(),
            "ShortGoalFastPathMatchesForward": self.TestShortGoalFastPathMatchesForward(),
            "TemporalGoalShapes": self.TestTemporalGoalShapes(),
            "TemporalTimeoutGradientSemantics": self.TestTemporalTimeoutGradientSemantics(),
            "GoalGroundingShapes": self.TestGoalGroundingShapes(),
            "HardCodebookCollapsePenalty": self.TestHardCodebookCollapsePenalty(),
            "GroundingConsistencyGradient": self.TestGroundingConsistencyGradient(),
            "GoalManagerBackward": self.TestGoalManagerBackward(),}
        passed = sum(1 for value in results.values() if value)
        print(f"\n[GoalModule Tests] {passed}/{len(results)} passed.")
        return results
