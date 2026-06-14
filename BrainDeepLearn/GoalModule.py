from __future__ import annotations
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from FunctionTools import AGICoreModule
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
        hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
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
        return (math.log(self.codes) + (prob * prob.clamp_min(1e-8).log()).sum(dim=-1)).mean()

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
    """Ground the current short-term goal onto physical slots and candidate subgoals."""

    def __init__(
        self,
        goalDim: int = ModuleDim.GoalShortDim,
        intentDim: int = ModuleDim.IntentionFeat,
        slotDim: int = ModuleDim.PstSlotDim,
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
        self.slot_ground_encoder = nn.Sequential(
            nn.LayerNorm(self.slot_dim + 4),
            nn.Linear(self.slot_dim + 4, self.slot_dim),
            nn.SiLU(),
            nn.Linear(self.slot_dim, self.slot_dim),
            nn.LayerNorm(self.slot_dim),)
        self.ground_attn = nn.MultiheadAttention(self.slot_dim, int(numHeads), batch_first=True)
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
        physicalState: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        slot_tensor = physicalState["SRaw"]
        m = physicalState["MphysRaw"]
        observed = physicalState["Observed"].float()
        last_seen = physicalState["LastSeen"].float()
        current_step = last_seen.amax(dim=1, keepdim=True)
        memory_age = (current_step - last_seen).clamp_min(0.0)
        memory_recency = torch.exp(-memory_age / 32.0)
        observed_weight = m * observed
        memory_weight = 0.5 * m * physicalState["M"] * (1.0 - observed) * memory_recency
        slot_weight = observed_weight + memory_weight

        B, _, _ = slot_tensor.shape
        slot_input = torch.cat([
            slot_tensor,
            observed_weight.unsqueeze(-1),
            memory_weight.unsqueeze(-1),
            slot_weight.unsqueeze(-1),
            memory_recency.unsqueeze(-1),], dim=-1)
        slot_embed = self.slot_ground_encoder(slot_input)
        masked_slots = slot_embed * slot_weight.unsqueeze(-1)
        no_slot_token = self.no_slot_token.expand(B, 1, self.slot_dim)
        memory_tokens = torch.cat([masked_slots, no_slot_token], dim=1)
        key_padding = torch.cat([
            slot_weight <= 0.0,
            torch.zeros(B, 1, device=slot_weight.device, dtype=torch.bool),], dim=1)
        query_vec = self.goal_intent_proj(torch.cat([goalEmbed, intentEmbed], dim=-1))
        query = query_vec.unsqueeze(1)
        grounded, attn_weights = self.ground_attn(
            query,
            memory_tokens,
            memory_tokens,
            key_padding_mask=key_padding)
        slot_logits = torch.einsum("bd,bkd->bk", query_vec, slot_embed) / (float(self.slot_dim) ** 0.5)
        slot_logits = slot_logits + slot_weight.clamp_min(1e-6).log()
        slot_logits = slot_logits.masked_fill(slot_weight <= 0.0, -1e9)
        no_slot_logit = self.no_slot_head(query_vec).squeeze(-1)
        reference_distribution = F.softmax(torch.cat([slot_logits, no_slot_logit.unsqueeze(-1)], dim=-1), dim=-1)
        referenced = reference_distribution[:, :-1]
        no_slot_prob = reference_distribution[:, -1]
        reference_confidence = referenced.sum(dim=-1)

        subgoal_q = self.subgoal_query.unsqueeze(0).expand(B, self.subgoal_steps, self.slot_dim)
        decoded = self.decomposer(subgoal_q, memory_tokens, memory_key_padding_mask=key_padding)
        return {
            "grounded_intention": grounded.squeeze(1),
            "referenced_object_probs": referenced,
            "reference_distribution": reference_distribution,
            "reference_confidence": reference_confidence,
            "no_slot_prob": no_slot_prob,
            "observed_reference_weight": observed_weight,
            "memory_reference_weight": memory_weight,
            "subgoal_skill_logits": self.skill_head(decoded),
            "subgoal_slot_logits": self.slot_head(decoded).squeeze(-1),
            "subgoal_param_delta": self.param_head(decoded),}


class TemporalGoalHead(AGICoreModule):
    def __init__(
        self,
        shortGoalDim: int = ModuleDim.GoalShortDim,
        temporalContextDim: int = ModuleDim.TemporalContextDim,
        hidden: int = 128,):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(shortGoalDim) + int(temporalContextDim)),
            nn.Linear(int(shortGoalDim) + int(temporalContextDim), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),)
        self.mode_head = nn.Linear(hidden, ModuleDim.TemporalPrimitiveCount)
        self.hold_head = nn.Linear(hidden, 1)
        self.replan_head = nn.Linear(hidden, 1)
        self.soft_timeout_head = nn.Linear(hidden, 1)
        self.hard_timeout_head = nn.Linear(hidden, 1)

    def forward(self, gShort: torch.Tensor, temporalContextFeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(torch.cat([gShort, temporalContextFeat], dim=-1))
        soft = F.softplus(self.soft_timeout_head(h)).squeeze(-1) * 1000.0
        hard = soft + F.softplus(self.hard_timeout_head(h)).squeeze(-1) * 1000.0
        return {
            "goal_mode_logits": self.mode_head(h),
            "goal_hold_score": torch.sigmoid(self.hold_head(h)).squeeze(-1),
            "goal_replan_score": torch.sigmoid(self.replan_head(h)).squeeze(-1),
            "goal_timeout_soft_ms": soft,
            "goal_timeout_hard_ms": hard,}


class FourLevelGoalManager(AGICoreModule):
    """Ultimate / long / mid / short hierarchical goal stack.

    g_U is the stable mission/ultimate objective. g_L and g_M decompose that mission
    into current task and phase goals. g_S is continuous and drives the immediate
    action-level objective.
    """

    def __init__(
        self,
        worldLatentDim: int,
        pstSummaryDim: int = ModuleDim.PstSlotDim,
        intentDim: int = ModuleDim.IntentionFeat,
        ultimateDim: int = ModuleDim.GoalUltimateDim,
        longDim: int = ModuleDim.GoalLongDim,
        midDim: int = ModuleDim.GoalMidDim,
        shortDim: int = ModuleDim.GoalShortDim,
        refinementDim: int = ModuleDim.RefinementDim,):
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

        short_in = self.ultimate_dim + self.mid_dim + pstSummaryDim + refinementDim
        self.short_head = nn.Sequential(
            nn.LayerNorm(short_in),
            nn.Linear(short_in, 256),
            nn.SiLU(),
            nn.Linear(256, self.short_dim),)

        # decode goals back into world-model latent space for cosine-progress reward
        self.mid_to_world = nn.Linear(self.mid_dim, worldLatentDim)
        # Anchor language intent onto the mission goal, so user/task text can shape
        # the stable objective above the task/phase decomposition.
        self.intent_to_ultimate = nn.Linear(intentDim, self.ultimate_dim)
        self.temporal_goal_head = TemporalGoalHead(shortGoalDim=self.short_dim)

    def forward(
        self,
        worldLatent: torch.Tensor,
        pstSummary: torch.Tensor,
        intentEmbed: torch.Tensor,
        refinementDir: torch.Tensor,) -> Dict[str, torch.Tensor]:
        ctx = torch.cat([worldLatent, pstSummary, intentEmbed], dim=-1)
        ultimate_out = self.ultimate_head(ctx)
        long_out = self.long_head(torch.cat([ctx, ultimate_out["goal"]], dim=-1))
        mid_out = self.mid_head(torch.cat([ctx, ultimate_out["goal"], long_out["goal"]], dim=-1))
        g_short = self.short_head(torch.cat([ultimate_out["goal"], mid_out["goal"], pstSummary, refinementDir], dim=-1))

        return {
            "g_ultimate": ultimate_out["goal"],
            "g_long": long_out["goal"],
            "g_mid": mid_out["goal"],
            "g_short": g_short,
            "ultimate_logits": ultimate_out["logits"],
            "long_logits": long_out["logits"],
            "mid_logits": mid_out["logits"],
            "ultimate_index": ultimate_out["index"],
            "long_index": long_out["index"],
            "mid_index": mid_out["index"],
            "mid_world_goal": self.mid_to_world(mid_out["goal"]),
            "intent_ultimate_anchor": self.intent_to_ultimate(intentEmbed),}

    def TemporalGoal(self, gShort: torch.Tensor, temporalContextFeat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.temporal_goal_head(gShort, temporalContextFeat)

    def ShortGoal(
        self,
        gUltimate: torch.Tensor,
        gMid: torch.Tensor,
        pstSummary: torch.Tensor,
        refinementDir: torch.Tensor,) -> torch.Tensor:
        return self.short_head(torch.cat([gUltimate, gMid, pstSummary, refinementDir], dim=-1))

    def AlignmentLoss(self, gUltimate: torch.Tensor, intentEmbed: torch.Tensor) -> torch.Tensor:
        # 10% of the gradient flows into g_ultimate so language can reshape the mission space.
        anchored = 0.9 * gUltimate.detach() + 0.1 * gUltimate
        return F.mse_loss(self.intent_to_ultimate(intentEmbed), anchored)

    def CosineProgress(self, worldDelta: torch.Tensor, gMid: torch.Tensor) -> torch.Tensor:
        return F.cosine_similarity(worldDelta, self.mid_to_world(gMid), dim=-1)

    def ProjectedProgress(self, worldDelta: torch.Tensor, gMid: torch.Tensor) -> torch.Tensor:
        """Magnitude-aware progress: length of the world delta projected on the
        decoded mid-goal direction."""
        direction = F.normalize(self.mid_to_world(gMid), dim=-1, eps=1e-6)
        return (worldDelta * direction).sum(dim=-1)


class SatisfactionCheckModule(AGICoreModule):
    """Check whether the short-term goal is satisfied and emit the next refinement direction."""

    def __init__(
        self,
        shortGoalDim: int = ModuleDim.GoalShortDim,
        slotDim: int = ModuleDim.PstSlotDim,
        endpointPoseFeatDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        refinementDim: int = ModuleDim.RefinementDim,
        hidden: int = 128,
        numHeads: int = 4,
        numLayers: int = 2,
        satisfactionThreshold: float = 0.85,):
        super().__init__()
        self.hidden = int(hidden)
        self.endpoint_pose_feat_dim = int(endpointPoseFeatDim)
        self.satisfaction_threshold = float(satisfactionThreshold)
        self.slot_proj = nn.Linear(slotDim, self.hidden)
        self.goal_token = nn.Linear(shortGoalDim, self.hidden)
        self.endpoint_token = nn.Linear(self.endpoint_pose_feat_dim, self.hidden)
        self.endpoint_summary_token = nn.Linear(self.endpoint_pose_feat_dim, self.hidden)

        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=int(numHeads),
            dim_feedforward=self.hidden * 4,
            dropout=0.05,
            batch_first=True,
            norm_first=True,)
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(numLayers))

        self.sat_head = nn.Linear(self.hidden, 1)
        self.refine_head = nn.Linear(self.hidden, int(refinementDim))
        # Calibration temperature trained jointly with the BCE success loss.
        self.sat_log_temperature = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        gShort: torch.Tensor,
        slotTensor: torch.Tensor,
        slotMask: torch.Tensor,
        endpointPoseTokens: torch.Tensor,
        endpointPoseFeat: torch.Tensor,) -> Dict[str, torch.Tensor]:
        B, _, _ = slotTensor.shape
        slots = self.slot_proj(slotTensor)
        goal = self.goal_token(gShort).unsqueeze(1)
        endpoint_summary = self.endpoint_summary_token(endpointPoseFeat).unsqueeze(1)
        endpoint_tokens = self.endpoint_token(endpointPoseTokens)
        tokens = torch.cat([goal, endpoint_summary, endpoint_tokens, slots], dim=1)

        pad = torch.zeros(B, 2 + endpoint_tokens.size(1), device=slotMask.device, dtype=torch.bool)
        key_padding = torch.cat([pad, slotMask <= 0.5], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding)

        summary = encoded[:, 0]
        sat_logits = self.sat_head(summary).squeeze(-1) * torch.exp(-self.sat_log_temperature)
        return {
            "sat_logits": sat_logits,
            "p_satisfied": torch.sigmoid(sat_logits),
            "refinement_dir": self.refine_head(summary),}

    def IsNotSatisfied(self, pSatisfied: torch.Tensor) -> torch.Tensor:
        return pSatisfied < self.satisfaction_threshold

    def SuccessLoss(self, satLogits: torch.Tensor, successLabel: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(satLogits, successLabel)

    def SatisfactionLoss(
        self,
        satLogits: torch.Tensor,
        successLabel: torch.Tensor,) -> Dict[str, torch.Tensor]:
        loss = self.SuccessLoss(satLogits, successLabel)
        return {
            "total": loss,
            "sat_success_loss": loss,
            "sat_target": successLabel,}


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
            "intentEmbed": torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
            "refinementDir": torch.randn(B, ModuleDim.RefinementDim, device=self.device),}

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
            assert tuple(out["ultimate_index"].shape) == (B, ModuleDim.GoalUltimateCodebookGroups)
            assert tuple(out["long_index"].shape) == (B, ModuleDim.GoalLongCodebookGroups)
            assert tuple(out["mid_index"].shape) == (B, ModuleDim.GoalMidCodebookGroups)
            assert tuple(out["mid_world_goal"].shape) == (B, self.WorldLatentDim())
            assert tuple(out["intent_ultimate_anchor"].shape) == (B, ModuleDim.GoalUltimateDim)
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
                    out["g_mid"],
                    inputs["pstSummary"],
                    inputs["refinementDir"])
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
                out = manager.TemporalGoal(goals["g_short"], temporal_context)
            assert tuple(out["goal_mode_logits"].shape) == (B, ModuleDim.TemporalPrimitiveCount)
            assert tuple(out["goal_hold_score"].shape) == (B,)
            assert tuple(out["goal_replan_score"].shape) == (B,)
            assert tuple(out["goal_timeout_soft_ms"].shape) == (B,)
            assert tuple(out["goal_timeout_hard_ms"].shape) == (B,)
            for name, value in out.items():
                self.AssertFinite(value, f"TemporalGoal {name}")
            print("TemporalGoal shape test passed.")
            return True
        except Exception as e:
            print(f"TemporalGoal shape test failed: {type(e).__name__}: {e}")
            return False

    def TestGoalGroundingShapes(self) -> bool:
        try:
            B, K = 2, 4
            grounding = GoalGrounding().to(self.device).eval()
            goal = torch.randn(B, ModuleDim.GoalShortDim, device=self.device)
            intent = torch.randn(B, ModuleDim.IntentionFeat, device=self.device)
            physical_state = {
                "SRaw": torch.randn(B, K, ModuleDim.PstSlotDim, device=self.device),
                "MphysRaw": torch.ones(B, K, device=self.device),
                "Observed": torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], device=self.device),
                "LastSeen": torch.tensor([[8, 8, 4, 0], [8, 3, 8, 0]], device=self.device),
                "M": torch.ones(B, K, device=self.device),}
            with torch.no_grad():
                out = grounding(goal, intent, physical_state)
            assert tuple(out["grounded_intention"].shape) == (B, ModuleDim.PstSlotDim)
            assert tuple(out["referenced_object_probs"].shape) == (B, K)
            assert tuple(out["reference_distribution"].shape) == (B, K + 1)
            assert tuple(out["subgoal_skill_logits"].shape) == (B, grounding.subgoal_steps, ModuleDim.UsageNumSkills)
            assert tuple(out["subgoal_slot_logits"].shape) == (B, grounding.subgoal_steps)
            assert tuple(out["subgoal_param_delta"].shape) == (B, grounding.subgoal_steps, ModuleDim.UsageParamDim)
            for name, value in out.items():
                self.AssertFinite(value, f"GoalGrounding {name}")
            print("GoalGrounding shape test passed.")
            return True
        except Exception as e:
            print(f"GoalGrounding shape test failed: {type(e).__name__}: {e}")
            return False

    def TestSatisfactionCheckShapesAndLoss(self) -> bool:
        try:
            B, K = 2, 4
            model = SatisfactionCheckModule().to(self.device)
            out = model(
                torch.randn(B, ModuleDim.GoalShortDim, device=self.device),
                torch.randn(B, K, ModuleDim.PstSlotDim, device=self.device),
                torch.ones(B, K, device=self.device),
                torch.randn(B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
                torch.randn(B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device))
            assert tuple(out["sat_logits"].shape) == (B,)
            assert tuple(out["p_satisfied"].shape) == (B,)
            assert tuple(out["refinement_dir"].shape) == (B, ModuleDim.RefinementDim)
            losses = model.SatisfactionLoss(out["sat_logits"], torch.rand(B, device=self.device))
            losses["total"].backward()
            grad_norm = sum(
                float(p.grad.detach().abs().sum().item())
                for p in model.parameters()
                if p.grad is not None)
            assert grad_norm > 0.0
            print("SatisfactionCheckModule shape/loss test passed.")
            return True
        except Exception as e:
            print(f"SatisfactionCheckModule shape/loss test failed: {type(e).__name__}: {e}")
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
                + manager.AlignmentLoss(out["g_ultimate"], inputs["intentEmbed"])
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

    def RunAll(self) -> Dict[str, bool]:
        results = {
            "FourLevelForwardShapes": self.TestFourLevelForwardShapes(),
            "ShortGoalFastPathMatchesForward": self.TestShortGoalFastPathMatchesForward(),
            "TemporalGoalShapes": self.TestTemporalGoalShapes(),
            "GoalGroundingShapes": self.TestGoalGroundingShapes(),
            "SatisfactionCheckShapesAndLoss": self.TestSatisfactionCheckShapesAndLoss(),
            "GoalManagerBackward": self.TestGoalManagerBackward(),}
        passed = sum(1 for value in results.values() if value)
        print(f"\n[GoalModule Tests] {passed}/{len(results)} passed.")
        return results
