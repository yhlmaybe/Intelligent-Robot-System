from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from DecisionDecoupler import DecisionDecouplerV2, EndpointPoseEncoding, MotionCommand, RelativePoseError, SAFETY_MARGIN_NAMES
from FunctionTools import AGICoreModule
from ModuleMessagerManager import ModuleDim
from NeuroSymbolicModule import FAILURE_CAUSES, OPERATORS, NeuroSymbolicOutput


class BeliefAssembler(AGICoreModule):
    def __init__(
        self,
        memDim: int,
        intentDim: int,
        valueDim: int,
        vNextDim: int,
        worldHzxDim: int,
        beliefDim: int = 1024,
        hidden: int = 1024,
        drop: float = 0.05,):
        super().__init__()
        self.belief_dim = int(beliefDim)
        self.world_hzx_dim = int(worldHzxDim)

        self.ln_mem = nn.LayerNorm(memDim)
        self.ln_intent = nn.LayerNorm(intentDim)
        self.ln_value = nn.LayerNorm(valueDim)
        self.ln_vnext = nn.LayerNorm(vNextDim)
        self.ln_world = nn.LayerNorm(worldHzxDim)

        self.p_mem = nn.Linear(memDim, beliefDim)
        self.p_intent = nn.Linear(intentDim, beliefDim)
        self.p_value = nn.Linear(valueDim, beliefDim)
        self.p_vnext = nn.Linear(vNextDim, beliefDim)
        self.p_world = nn.Linear(worldHzxDim, beliefDim)
        self.p_scalar = nn.Linear(4, beliefDim)

        fuse_in = beliefDim * 6
        self.fuse = nn.Sequential(
            nn.LayerNorm(fuse_in),
            nn.Linear(fuse_in, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, beliefDim),)
        self.gate = nn.Sequential(
            nn.Linear(fuse_in, beliefDim),
            nn.Sigmoid(),)
        self.source_gate = nn.Linear(fuse_in, 6)
        self.ln_out = nn.LayerNorm(beliefDim)
        self.layerscale = nn.Parameter(torch.ones(beliefDim) * 1e-2)

        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)
        nn.init.zeros_(self.source_gate.weight)
        nn.init.zeros_(self.source_gate.bias)

    def forward(
        self,
        memFeat: torch.Tensor,
        intentFeat: torch.Tensor,
        valueTensor: torch.Tensor,
        vNextTensor: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,
        risk: torch.Tensor,
        worldHzx: torch.Tensor,) -> torch.Tensor:
        scalars = torch.stack([uncertainty, confidence, precision, risk], dim=-1)

        m = self.p_mem(self.ln_mem(memFeat))
        i = self.p_intent(self.ln_intent(intentFeat))
        v = self.p_value(self.ln_value(valueTensor))
        vn = self.p_vnext(self.ln_vnext(vNextTensor))
        w = self.p_world(self.ln_world(worldHzx))
        s = self.p_scalar(scalars)

        fused_in = torch.cat([m, i, v, vn, w, s], dim=-1)
        source_weights = F.softmax(self.source_gate(fused_in), dim=-1) * 6.0
        weighted_sources = (
            torch.stack([m, i, v, vn, w, s], dim=1)
            * source_weights.unsqueeze(-1))
        source_sum = weighted_sources.sum(dim=1)
        gated_fused_in = weighted_sources.flatten(start_dim=1)
        delta = self.fuse(gated_fused_in)
        gate = self.gate(gated_fused_in)
        return self.ln_out(source_sum + gate * self.layerscale * delta)


class LatentControlInferer(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        dynDim: int,
        uDim: int,
        hidden: int = 256,
        drop: float = 0.05,):
        super().__init__()
        self.u_dim = int(uDim)
        in_dim = int(beliefDim) + int(dynDim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),)
        self.mu_head = nn.Linear(hidden, uDim)
        self.logvar_head = nn.Linear(hidden, uDim)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.logvar_head.weight)
        with torch.no_grad():
            self.logvar_head.bias.fill_(-1.0)

    def forward(self, belief: torch.Tensor, decisionState: torch.Tensor):
        h = self.trunk(torch.cat([belief, decisionState], dim=-1))
        return self.mu_head(h), self.logvar_head(h).clamp(-8.0, 4.0)


class PredictiveDecisionCore(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        uDim: int,
        dynDim: int,
        nSteps: int = 2,
        hidden: int = 256,
        drop: float = 0.05,):
        super().__init__()
        self.n_steps = int(nSteps)
        self.dyn_dim = int(dynDim)
        in_dim = int(dynDim) + int(beliefDim) + int(uDim) + 1

        self.f = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dynDim),
            nn.Tanh(),)
        self.belief_to_dyn = nn.Linear(beliefDim, dynDim, bias=False)
        self.dt = nn.Parameter(torch.tensor(0.5))
        self.pull_gain = nn.Parameter(torch.tensor(0.1))
        self.drift_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.zeros_(self.f[-2].weight)
        nn.init.zeros_(self.f[-2].bias)

    def Drift(
        self,
        h: torch.Tensor,
        belief: torch.Tensor,
        u: torch.Tensor,
        precision: torch.Tensor,) -> torch.Tensor:
        x = torch.cat([h, belief, u, precision], dim=-1)
        return F.softplus(self.drift_scale) * self.f(x)

    def forward(
        self,
        prevState: torch.Tensor,
        belief: torch.Tensor,
        u: torch.Tensor,
        precision: torch.Tensor,) -> torch.Tensor:
        h = prevState
        dt = torch.sigmoid(self.dt) / float(self.n_steps)
        target = self.belief_to_dyn(belief)
        pull_rate = F.softplus(self.pull_gain) * precision
        for _ in range(self.n_steps):
            k1 = self.Drift(h, belief, u, precision)
            k2 = self.Drift(h + dt * k1, belief, u, precision)
            h_drift = h + 0.5 * dt * (k1 + k2)
            decay = torch.exp(-pull_rate * dt)
            h = target + (h_drift - target) * decay
        return h


class PredictionErrorHead(AGICoreModule):
    def __init__(self, dynDim: int, beliefDim: int, hidden: int = 256, drop: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dynDim),
            nn.Linear(dynDim, hidden),
            nn.SiLU(),
            nn.Dropout(drop),
            nn.Linear(hidden, beliefDim),)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, decisionState: torch.Tensor) -> torch.Tensor:
        return self.net(decisionState)


class OptionTransitionPrior(AGICoreModule):
    """Learned option-to-option prior with a neutral uniform initialization."""

    def __init__(self, optionCount: int):
        super().__init__()
        self.option_count = int(optionCount)
        self.transition_logits = nn.Parameter(
            torch.zeros(self.option_count, self.option_count))

    def forward(self, previousLogits: torch.Tensor) -> torch.Tensor:
        previous_log_prob = F.log_softmax(previousLogits.detach(), dim=-1)
        transition_log_prob = F.log_softmax(self.transition_logits, dim=-1)
        next_log_prob = torch.logsumexp(
            previous_log_prob.unsqueeze(-1) + transition_log_prob,
            dim=-2)
        return next_log_prob + math.log(float(self.option_count))


class EligibilityTracePlasticityLayer(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        outDim: int,
        lam: float = 0.9,
        eta: float = 1e-3,
        gamma: float = 1e-2,
        applyScale: float = 0.25,
        maxRowNorm: float = 2.0,
        enabled: bool = True,):
        super().__init__()
        self.in_dim = int(inDim)
        self.out_dim = int(outDim)
        self.lam = float(lam)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.apply_scale = float(applyScale)
        self.max_row_norm = float(maxRowNorm)
        self.enabled = bool(enabled)

        self.base = nn.Parameter(torch.randn(outDim, inDim, device=self.device, dtype=self.dtype) * 0.02)
        self.register_buffer("trace", torch.empty(0), persistent=True)
        self.register_buffer("fast", torch.empty(0), persistent=True)

    def EnsureBatch(self, B: int, device: torch.device, dtype: torch.dtype):
        if int(self.trace.size(0)) != B:
            self.trace = torch.zeros(B, self.out_dim, self.in_dim, device=device, dtype=dtype)
            self.fast = torch.zeros(B, self.out_dim, self.in_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, neuromod: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return F.linear(x, self.base)

        B = int(x.size(0))
        self.EnsureBatch(B, self.device, self.dtype)

        with torch.no_grad():
            mod = neuromod.detach().view(B, 1, 1)
            self.fast = (1.0 - self.gamma) * self.fast + self.eta * mod * self.trace

            if self.max_row_norm > 0.0:
                flat = self.fast.reshape(B, -1)
                nrm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
                scale = (self.max_row_norm / nrm).clamp_max(1.0)
                self.fast = self.fast * scale.view(B, 1, 1)

        w_eff = self.base.unsqueeze(0) + self.apply_scale * self.fast.detach()
        return torch.einsum("bi,boi->bo", x, w_eff)

    def Commit(self, pre: torch.Tensor, post: torch.Tensor, executeMask: torch.Tensor):
        if not self.enabled:
            return
        B = int(pre.size(0))
        self.EnsureBatch(B, self.device, self.dtype)
        with torch.no_grad():
            outer = torch.einsum("bo,bi->boi", post.detach(), pre.detach())
            write = executeMask.view(B, 1, 1)
            self.trace = self.lam * self.trace + (1.0 - self.lam) * write * outer

    def ClearTrace(self, invalidMask: torch.Tensor):
        if not self.enabled or self.trace.numel() == 0:
            return
        with torch.no_grad():
            self.trace.masked_fill_(invalidMask.view(-1, 1, 1).bool(), 0.0)

    def Reset(self, doneMask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            if doneMask is None:
                self.trace.zero_()
                self.fast.zero_()
                return
            keep = (1.0 - doneMask.view(-1).to(self.trace.dtype)).view(-1, 1, 1)
            self.trace.mul_(keep)
            self.fast.mul_(keep)


class NeuroSymbolicConditioner(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        planDim: int = 256,
        subgoalFeatureDim: int = ModuleDim.DecisionEndpointPoseFeatDim,
        constraintTokens: int = 8,
        constraintTokenDim: int = 128,
        operatorDim: int = len(OPERATORS),
        failureDim: int = len(FAILURE_CAUSES),
        hiddenDim: int = 1024,):
        super().__init__()
        self.belief_dim = int(beliefDim)
        self.plan_dim = int(planDim)
        self.subgoal_feature_dim = int(subgoalFeatureDim)
        self.constraint_tokens = int(constraintTokens)
        self.constraint_token_dim = int(constraintTokenDim)
        self.operator_dim = int(operatorDim)
        self.failure_dim = int(failureDim)

        raw_dim = (
            self.plan_dim
            + self.subgoal_feature_dim
            + self.constraint_tokens * self.constraint_token_dim
            + self.operator_dim
            + self.failure_dim
            + 1)
        self.symbol_projector = nn.Sequential(
            nn.LayerNorm(raw_dim),
            nn.Linear(raw_dim, hiddenDim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hiddenDim, self.belief_dim),
            nn.LayerNorm(self.belief_dim),)
        self.plan_refiner = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.plan_dim),
            nn.Linear(self.belief_dim + self.plan_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.plan_dim),
            nn.LayerNorm(self.plan_dim),)
        self.subgoal_refiner = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.subgoal_feature_dim),
            nn.Linear(self.belief_dim + self.subgoal_feature_dim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, self.subgoal_feature_dim),
            nn.LayerNorm(self.subgoal_feature_dim),)
        self.constraint_context = nn.Linear(self.belief_dim, self.constraint_token_dim)
        self.constraint_refiner = nn.Sequential(
            nn.LayerNorm(2 * self.constraint_token_dim),
            nn.Linear(2 * self.constraint_token_dim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, self.constraint_token_dim),
            nn.LayerNorm(self.constraint_token_dim),)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.operator_dim + self.failure_dim + 1),
            nn.Linear(self.belief_dim + self.operator_dim + self.failure_dim + 1, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, 1),)

    def forward(self, neuroSymbolic: NeuroSymbolicOutput) -> Dict[str, torch.Tensor]:
        B = neuroSymbolic.plan_latent.size(0)
        constraint_flat = neuroSymbolic.constraint_tokens.reshape(B, -1)
        invoke = neuroSymbolic.invoke_mask.view(B, 1)
        risk_cause_logits = getattr(
            neuroSymbolic,
            "risk_cause_logits",
            neuroSymbolic.failure_cause_logits)
        raw = torch.cat([
            neuroSymbolic.plan_latent,
            neuroSymbolic.subgoal_feature,
            constraint_flat,
            neuroSymbolic.operator_logits,
            risk_cause_logits,
            invoke,
        ], dim=-1)
        context = self.symbol_projector(raw)
        plan_latent = self.plan_refiner(torch.cat([context, neuroSymbolic.plan_latent], dim=-1))
        subgoal_feature = self.subgoal_refiner(torch.cat([context, neuroSymbolic.subgoal_feature], dim=-1))
        token_context = self.constraint_context(context).unsqueeze(1).expand(
            B,
            self.constraint_tokens,
            self.constraint_token_dim,)
        constraint_tokens = self.constraint_refiner(torch.cat([
            neuroSymbolic.constraint_tokens,
            token_context,
        ], dim=-1))
        energy = self.energy_head(torch.cat([
            context,
            neuroSymbolic.operator_logits,
            risk_cause_logits,
            invoke,
        ], dim=-1)).squeeze(-1)
        return {
            "context": context,
            "plan_latent": plan_latent,
            "subgoal_feature": subgoal_feature,
            "constraint_tokens": constraint_tokens,
            "energy": energy,}


class TemporalDecisionHead(AGICoreModule):
    """Combines symbolic/goal timing without changing temporal label meanings."""

    def __init__(
        self,
        primitiveCount: int = ModuleDim.TemporalPrimitiveCount,):
        super().__init__()
        self.primitive_count = int(primitiveCount)

    def forward(
        self,
        neuroSymbolic: NeuroSymbolicOutput,
        temporalGoal: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
        kind_logits = neuroSymbolic.temporal_logits + temporalGoal["goal_mode_logits"]
        soft_timeout = temporalGoal["goal_timeout_soft_ms"]
        hard_timeout = temporalGoal["goal_timeout_hard_ms"]
        return {
            "kind_logits": kind_logits,
            "duration_ms": soft_timeout,
            "soft_timeout_ms": soft_timeout,
            "hard_timeout_ms": hard_timeout,
            "p_interrupt": neuroSymbolic.interrupt_guard_score,
            "redispatch_score": neuroSymbolic.redispatch_guard_score,
            "same_operator": neuroSymbolic.same_operator,
            "operator_changed": neuroSymbolic.operator_changed,
            "invoke_delta": neuroSymbolic.invoke_delta,
            "reference_drift": neuroSymbolic.reference_drift,}




class DecisionExtractor(AGICoreModule):
    def __init__(
        self,
        stateDim: int = 1024,
        useHebb: bool = True,
        optionNum: int = 80,
        hiddenDim: int = 1024,
        psiDim: int = 1024,
        intentDim: int = 1024,
        includeNoSkill: bool = True,
        *,
        valueTensorDim: int = 512,
        vNextTensorDim: int = 512,
        worldHDim: int = 512,
        worldZDim: int = 64,
        worldXDim: int = 64,
        beliefDim: int = 1024,
        decisionDynDim: int = 256,
        latentControlDim: int = 64,
        mapperEmbedDim: int = 256,
        actionEmbedDim: int = ModuleDim.DecisionFeedbackEmbedDim,):
        super().__init__()
        self.stateDim = int(stateDim)
        self.intentDim = int(intentDim)
        self.includeNoSkill = bool(includeNoSkill)
        self.value_tensor_dim = int(valueTensorDim)
        self.v_next_tensor_dim = int(vNextTensorDim)
        self.world_h_dim = int(worldHDim)
        self.world_z_dim = int(worldZDim)
        self.world_x_dim = int(worldXDim)
        self.world_hzx_dim = self.world_h_dim + self.world_z_dim + self.world_x_dim
        self.belief_dim = int(beliefDim)
        self.dyn_dim = int(decisionDynDim)
        self.u_dim = int(latentControlDim)
        self.action_embed_dim = int(actionEmbedDim)
        self.mapper_hidden_dim = int(mapperEmbedDim)
        self.num_options = int(optionNum)
        self.psi_dim = int(psiDim)

        self.belief_assembler = BeliefAssembler(
            memDim=self.stateDim,
            intentDim=self.intentDim,
            valueDim=self.value_tensor_dim,
            vNextDim=self.v_next_tensor_dim,
            worldHzxDim=self.world_hzx_dim,
            beliefDim=self.belief_dim,)
        self.latent_inferer = LatentControlInferer(
            beliefDim=self.belief_dim,
            dynDim=self.dyn_dim,
            uDim=self.u_dim,)
        self.predictive_core = PredictiveDecisionCore(
            beliefDim=self.belief_dim,
            uDim=self.u_dim,
            dynDim=self.dyn_dim,
            nSteps=2,)
        self.belief_predictor = PredictionErrorHead(
            dynDim=self.dyn_dim,
            beliefDim=self.belief_dim,)
        self.elig_plasticity = EligibilityTracePlasticityLayer(
            inDim=self.u_dim,
            outDim=self.u_dim,
            enabled=useHebb,)
        action_ctx_dim = self.u_dim + self.dyn_dim + 2 + self.action_embed_dim
        self.action_context = nn.Sequential(
            nn.LayerNorm(action_ctx_dim),
            nn.Linear(action_ctx_dim, self.mapper_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(self.mapper_hidden_dim, self.mapper_hidden_dim),
            nn.SiLU(),)
        option_in_dim = self.dyn_dim + self.u_dim + self.mapper_hidden_dim
        self.option_head = nn.Sequential(
            nn.LayerNorm(option_in_dim),
            nn.Linear(option_in_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.num_options),)
        self.option_transition_prior = OptionTransitionPrior(self.num_options)
        self.option_psi_head = nn.Sequential(
            nn.LayerNorm(option_in_dim),
            nn.Linear(option_in_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.num_options * self.psi_dim),)
        self.option_to_belief = nn.Sequential(
            nn.LayerNorm(self.psi_dim),
            nn.Linear(self.psi_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.belief_dim),)

        self.nesy_conditioner = NeuroSymbolicConditioner(
            beliefDim=self.belief_dim,
            planDim=256,
            subgoalFeatureDim=ModuleDim.DecisionEndpointPoseFeatDim,
            constraintTokens=8,
            constraintTokenDim=128,)
        refine_in_dim = (
            self.belief_dim
            + self.dyn_dim
            + self.u_dim
            + self.world_hzx_dim
            + ModuleDim.PstSlotDim
            + ModuleDim.DecisionEndpointPoseFeatDim
            + self.belief_dim
            + self.belief_dim)
        self.nesy_decision_refiner = nn.Sequential(
            nn.LayerNorm(refine_in_dim),
            nn.Linear(refine_in_dim, hiddenDim),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hiddenDim, self.belief_dim),)
        self.nesy_decision_gate = nn.Sequential(
            nn.LayerNorm(refine_in_dim),
            nn.Linear(refine_in_dim, self.belief_dim),
            nn.Sigmoid(),)
        self.final_decision_norm = nn.LayerNorm(self.belief_dim)
        self.decoder_plan_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + 256),
            nn.Linear(self.belief_dim + 256, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, 256),
            nn.LayerNorm(256),)
        self.decoder_subgoal_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + 2 * ModuleDim.DecisionEndpointPoseFeatDim),
            nn.Linear(self.belief_dim + 2 * ModuleDim.DecisionEndpointPoseFeatDim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, ModuleDim.DecisionEndpointPoseFeatDim),
            nn.LayerNorm(ModuleDim.DecisionEndpointPoseFeatDim),)
        self.decoder_constraint_seed = nn.Linear(self.belief_dim, 8 * 128)
        self.decoder_constraint_head = nn.Sequential(
            nn.LayerNorm(2 * 128),
            nn.Linear(2 * 128, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, 128),
            nn.LayerNorm(128),)
        self.decision_energy_head = nn.Sequential(
            nn.LayerNorm(self.belief_dim + self.world_hzx_dim + ModuleDim.DecisionEndpointPoseFeatDim),
            nn.Linear(self.belief_dim + self.world_hzx_dim + ModuleDim.DecisionEndpointPoseFeatDim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, 1),)
        self.temporal_decision_head = TemporalDecisionHead()

    def OptionLogits(
        self,
        optionPolicyInput: torch.Tensor,
        optionPriorLogit: torch.Tensor,) -> torch.Tensor:
        return (
            self.option_head(optionPolicyInput)
            + 0.05 * optionPriorLogit
            + self.option_transition_prior(optionPriorLogit))

    def OptionLogProb(
        self,
        optionPolicyInput: torch.Tensor,
        optionPriorLogit: torch.Tensor,
        optionIndex: torch.Tensor,) -> torch.Tensor:
        logits = self.OptionLogits(optionPolicyInput, optionPriorLogit)
        return F.log_softmax(logits, dim=-1).gather(1, optionIndex.view(-1, 1)).squeeze(1)

    def PredictBelief(self, decisionState: torch.Tensor) -> torch.Tensor:
        return self.belief_predictor(decisionState)

    def CommitEligibility(
        self,
        eligibilityPre: torch.Tensor,
        eligibilityPost: torch.Tensor,
        executeMask: torch.Tensor,) -> None:
        self.elig_plasticity.Commit(eligibilityPre, eligibilityPost, executeMask)

    def ClearInvalidEligibility(self, invalidMask: torch.Tensor) -> None:
        self.elig_plasticity.ClearTrace(invalidMask)

    def forward(
        self,
        stateFeat: torch.Tensor,
        intentFeat: torch.Tensor,
        *,
        valueTensor: torch.Tensor,
        vNextTensor: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,
        precision: torch.Tensor,
        risk: torch.Tensor,
        worldHzx: torch.Tensor,
        prevDecisionState: torch.Tensor,
        prevLatentControl: torch.Tensor,
        prevActionEmbed: torch.Tensor,
        prevMapperHidden: torch.Tensor,
        feedbackTdError: torch.Tensor,
        prevOptionLogit: torch.Tensor,
        sample: bool = True,
        deterministic: bool = False,) -> Dict[str, Any]:

        B = stateFeat.size(0)
        value_in = valueTensor.reshape(B, self.value_tensor_dim)
        v_next_in = vNextTensor.reshape(B, self.v_next_tensor_dim)
        belief = self.belief_assembler(
            memFeat=stateFeat,
            intentFeat=intentFeat,
            valueTensor=value_in,
            vNextTensor=v_next_in,
            uncertainty=uncertainty,
            confidence=confidence,
            precision=precision,
            risk=risk,
            worldHzx=worldHzx,)

        u_mu, u_logvar = self.latent_inferer(belief, prevDecisionState)
        if sample and not deterministic:
            u_t = u_mu + torch.exp(0.5 * u_logvar) * torch.randn_like(u_mu)
        else:
            u_t = u_mu
        u_pre = 0.8 * u_t + 0.2 * prevLatentControl

        neuromod = feedbackTdError * confidence * (1.0 - uncertainty)
        u_plastic = self.elig_plasticity(u_pre, neuromod)
        u_t = u_pre + 0.1 * u_plastic

        decision_state = self.predictive_core(prevDecisionState, belief, u_t, precision.view(-1, 1))
        scalars = torch.cat([uncertainty.view(-1, 1), confidence.view(-1, 1)], dim=-1)
        mapper_hidden = self.action_context(torch.cat([
            u_t,
            decision_state,
            scalars,
            prevActionEmbed,], dim=-1))
        mapper_hidden_next = 0.5 * prevMapperHidden + 0.5 * mapper_hidden

        option_in = torch.cat([decision_state, u_t, mapper_hidden_next], dim=-1)
        option_logits = self.OptionLogits(option_in, prevOptionLogit.detach())
        option_log_probs = F.log_softmax(option_logits, dim=-1)
        w_t = option_log_probs.exp()
        psi_all = self.option_psi_head(option_in).view(B, self.num_options, self.psi_dim)
        if sample and not deterministic:
            opt_idx = torch.distributions.Categorical(probs=w_t).sample()
            option_hard = F.one_hot(opt_idx, num_classes=self.num_options).to(w_t.dtype)
            option_weight = option_hard + w_t - w_t.detach()
        else:
            opt_idx = option_logits.argmax(dim=-1)
            option_weight = F.one_hot(opt_idx, num_classes=self.num_options).to(w_t.dtype)
        psi_selected = (option_weight.unsqueeze(-1) * psi_all).sum(dim=1)
        option_context = self.option_to_belief(psi_selected)

        gaussian_entropy = (
            0.5 * (1.0 + math.log(2.0 * math.pi))
            + 0.5 * u_logvar
            + math.log(0.8)).sum(dim=-1)
        option_entropy = -(w_t * option_log_probs).sum(dim=-1)
        entropy_scalar = gaussian_entropy + option_entropy

        out: Dict[str, Any] = {
            "z": decision_state,
            "entropy": entropy_scalar,
            "option": {
                "logits": option_logits,
                "psi_all": psi_all,
                "w_t": w_t,
                "psi_selected": psi_selected,
                "option_context": option_context,
                "opt_idx": opt_idx,
                "logp_option": option_log_probs.gather(1, opt_idx.view(-1, 1)).squeeze(1),
                "policy_input": option_in,
                "prior_logits": prevOptionLogit.detach(),},
            "prevOptionLogit_next": option_logits.detach(),
            "belief": belief,
            "decision_state": decision_state,
            "decision_state_next": decision_state.detach(),
            "decision_uncertainty": uncertainty,
            "latent_control": {
                "u": u_t,
                "mu": u_mu,
                "logvar": u_logvar,},
            "latent_control_next": u_t.detach(),
            "mapper": {
                "hidden": mapper_hidden,
                "hidden_next": mapper_hidden_next.detach(),},
            "eligibility": {
                "pre": u_pre.detach(),
                "post": u_plastic.detach(),},}

        return out

    def RefineWithNeuroSymbolic(
        self,
        baseActOut: Dict[str, Any],
        neuroSymbolic: NeuroSymbolicOutput,
        endpointPoseFeat: torch.Tensor,
        worldHzx: torch.Tensor,
        pstSummary: torch.Tensor,
        goalDecisionContext: torch.Tensor,
        temporalGoal: Dict[str, torch.Tensor],) -> Dict[str, Any]:
        nesy = self.nesy_conditioner(neuroSymbolic)
        latent_u = baseActOut["latent_control"]["u"]
        refine_in = torch.cat([
            baseActOut["belief"],
            baseActOut["decision_state"],
            latent_u,
            worldHzx,
            pstSummary,
            endpointPoseFeat,
            goalDecisionContext,
            nesy["context"],
        ], dim=-1)
        delta = self.nesy_decision_refiner(refine_in)
        gate = self.nesy_decision_gate(refine_in)
        decision_feature = self.final_decision_norm(
            baseActOut["belief"]
            + baseActOut["option"]["option_context"]
            + gate * delta)
        decoder_plan_latent = self.decoder_plan_head(torch.cat([
            decision_feature,
            nesy["plan_latent"],
        ], dim=-1))
        decoder_subgoal_feature = self.decoder_subgoal_head(torch.cat([
            decision_feature,
            nesy["subgoal_feature"],
            endpointPoseFeat,
        ], dim=-1))
        B = decision_feature.size(0)
        constraint_seed = self.decoder_constraint_seed(decision_feature).view(B, 8, 128)
        decoder_constraint_tokens = self.decoder_constraint_head(torch.cat([
            nesy["constraint_tokens"],
            constraint_seed,
        ], dim=-1))
        decision_energy = self.decision_energy_head(torch.cat([
            decision_feature,
            worldHzx,
            endpointPoseFeat,
        ], dim=-1)).squeeze(-1) + nesy["energy"]
        temporal_decision = self.temporal_decision_head(
            neuroSymbolic,
            temporalGoal)

        baseActOut["base_belief"] = baseActOut["belief"]
        baseActOut["belief_final"] = decision_feature
        baseActOut["decision_feature"] = decision_feature
        baseActOut["decoder_plan_latent"] = decoder_plan_latent
        baseActOut["decoder_subgoal_feature"] = decoder_subgoal_feature
        baseActOut["decoder_constraint_tokens"] = decoder_constraint_tokens
        baseActOut["neuro_symbolic_condition"] = {
            "context": nesy["context"],
            "gate": gate,
            "symbolic_energy": nesy["energy"],}
        baseActOut["decision_energy"] = decision_energy
        baseActOut["temporal_decision"] = temporal_decision
        return baseActOut

    def ResetHebbianMemory(self, value: float = 0.0, doneMask: Optional[torch.Tensor] = None):
        for m in self.modules():
            if isinstance(m, EligibilityTracePlasticityLayer):
                m.Reset(doneMask=doneMask)




class CEMPlanner(AGICoreModule):
    def __init__(
        self,
        worldModel: nn.Module,
        wmIsOnlineWrapper: bool,
        decisionDecoupler: DecisionDecouplerV2,
        N: int = 64,
        elite: int = 8,
        iters: int = 3,
        temperature: float = 1.0,
        momentum: float = 0.15,
        minVar: float = 1e-4,
        candidateChunkSize: int = 8,
    ):
        super().__init__()
        self.wm = worldModel
        self.wm_is_online_wrapper = bool(wmIsOnlineWrapper)
        self.decision_decoupler = decisionDecoupler
        self.N = int(N)
        self.elite = int(elite)
        self.iters = int(iters)
        self.temperature = float(temperature)
        self.momentum = float(momentum)
        self.min_var = float(minVar)
        self.candidate_chunk_size = max(1, int(candidateChunkSize))

    @torch.no_grad()
    def Plan(
        self,
        decisionLatent: torch.Tensor,
        endpointPose: torch.Tensor,
        endpointPoseEncoding: EndpointPoseEncoding,
        h0: torch.Tensor,
        z0: torch.Tensor,
        x0: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        robotSelfState: torch.Tensor,
        returnDiagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = int(decisionLatent.size(0))
        N = self.N
        E = min(self.elite, N)
        active_mask = self.decision_decoupler.MaskDecisionTensor(
            torch.ones_like(decisionLatent))
        mu = decisionLatent * active_mask
        std = torch.ones_like(decisionLatent) * active_mask

        def score_latent_candidates(
            latent_candidates: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            candidate_count = int(latent_candidates.size(1))
            actions = self.decision_decoupler.ProjectDecisionLatent(latent_candidates)
            score_chunks = []
            for start in range(0, candidate_count, self.candidate_chunk_size):
                end = min(start + self.candidate_chunk_size, candidate_count)
                chunk_size = end - start

                def expand_candidates(value: torch.Tensor) -> torch.Tensor:
                    return value.unsqueeze(1).expand(
                        B, chunk_size, *value.shape[1:]).reshape(
                            B * chunk_size, *value.shape[1:]).contiguous()

                h = expand_candidates(h0)
                z = expand_candidates(z0)
                x = expand_candidates(x0)
                pose = expand_candidates(endpointPose)
                action = actions[:, start:end].reshape(
                    B * chunk_size,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim)
                encoding = EndpointPoseEncoding(
                    endpoint_pose_tokens=expand_candidates(
                        endpointPoseEncoding.endpoint_pose_tokens),
                    endpoint_pose_feat=expand_candidates(
                        endpointPoseEncoding.endpoint_pose_feat),)
                target_pose = self.decision_decoupler.DecodeEndpointPose(pose, action)
                action_enc = self.decision_decoupler.EncodeDecisionFeedback(
                    action, target_pose, encoding)
                physical_state_candidates = {
                    key: expand_candidates(value)
                    for key, value in physicalState.items()}
                prior = self.wm.StepPriorOnly(
                    h,
                    z,
                    x,
                    action_enc,
                    physicalState=physical_state_candidates,
                    robotSelfState=expand_candidates(robotSelfState),
                    sample=False)
                score_chunks.append(
                    (prior["r_pred"] - prior["d_prob"]).view(B, chunk_size))
            return torch.cat(score_chunks, dim=1), actions

        for _ in range(self.iters):
            noise = torch.randn_like(std.unsqueeze(1).expand(B, N, -1, -1))
            latent_samples = mu.unsqueeze(1) + noise * std.unsqueeze(1)
            score, _ = score_latent_candidates(latent_samples)

            topk = torch.topk(score, k=E, dim=1).indices
            elite_scores = score.gather(1, topk)
            if self.temperature <= 0.0:
                weights = torch.full_like(elite_scores, 1.0 / float(E))
            else:
                weights = F.softmax(elite_scores / float(self.temperature), dim=1)
            b_idx = torch.arange(B, device=decisionLatent.device).unsqueeze(1).expand(B, E)
            w = weights.unsqueeze(-1).unsqueeze(-1)
            elite_latent = latent_samples[b_idx, topk]
            mu_new = (w * elite_latent).sum(dim=1)
            var_new = (
                (w * (elite_latent - mu_new.unsqueeze(1)).square())
                .sum(dim=1)
                .clamp_min(self.min_var)
                * active_mask)
            std_new = var_new.sqrt()
            mu = (
                self.momentum * mu
                + (1.0 - self.momentum) * mu_new) * active_mask
            std = (
                self.momentum * std
                + (1.0 - self.momentum) * std_new) * active_mask

        population_return = (weights * elite_scores).sum(dim=1)
        final_score, final_action = score_latent_candidates(mu.unsqueeze(1))
        out = {
            "decision_latent": mu,
            "decision_tensor": final_action[:, 0],
            "expected_return": final_score[:, 0],}
        if returnDiagnostics:
            out["diagnostics"] = {
                "std": std,
                "elite_population_return": population_return,}
        return out


class DecisionPlannerExtractor:
    def BuildPlanner(
        self,
        worldModel: nn.Module,
        wmIsOnlineWrapper: bool,
        decisionDecoupler: DecisionDecouplerV2,
        **cemKwargs: Any,
    ) -> CEMPlanner:
        return CEMPlanner(
            worldModel=worldModel,
            wmIsOnlineWrapper=wmIsOnlineWrapper,
            decisionDecoupler=decisionDecoupler,
            **cemKwargs,
        )


class TestDecisionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def TestDecisionExtractorIOShapes(self) -> bool:
        B = 2
        model = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat,
            intentDim=ModuleDim.IntentionFeat,
            hiddenDim=256,
            psiDim=256,
            optionNum=8,
            useHebb=False,
        ).to(self.device)
        out = model(
            torch.randn(B, ModuleDim.MemoryFeat, device=self.device),
            torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
            valueTensor=torch.randn(B, model.value_tensor_dim, device=self.device),
            vNextTensor=torch.randn(B, model.v_next_tensor_dim, device=self.device),
            prevOptionLogit=torch.zeros(B, 8, device=self.device),
            uncertainty=torch.zeros(B, device=self.device),
            confidence=torch.ones(B, device=self.device),
            precision=torch.ones(B, device=self.device),
            risk=torch.zeros(B, device=self.device),
            worldHzx=torch.randn(B, ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState, device=self.device),
            prevDecisionState=torch.zeros(B, ModuleDim.DecisionDynDim, device=self.device),
            prevLatentControl=torch.zeros(B, ModuleDim.LatentControlDim, device=self.device),
            prevActionEmbed=torch.zeros(B, ModuleDim.DecisionFeedbackEmbedDim, device=self.device),
            prevMapperHidden=torch.zeros(B, ModuleDim.MapperHiddenDim, device=self.device),
            feedbackTdError=torch.zeros(B, device=self.device),
        )
        return (
            set(out) == {
                "z", "entropy", "option", "prevOptionLogit_next", "belief",
                "decision_state", "decision_state_next", "decision_uncertainty",
                "latent_control", "latent_control_next", "mapper", "eligibility"}
            and set(out["option"]) == {
                "logits", "psi_all", "w_t", "psi_selected", "option_context",
                "opt_idx", "logp_option", "policy_input", "prior_logits"}
            and set(out["latent_control"]) == {"u", "mu", "logvar"}
            and set(out["mapper"]) == {"hidden", "hidden_next"}
            and set(out["eligibility"]) == {"pre", "post"}
            and out["belief"].shape == (B, ModuleDim.DecisionBeliefDim)
            and out["decision_state_next"].shape == (B, ModuleDim.DecisionDynDim)
            and out["latent_control_next"].shape == (B, ModuleDim.LatentControlDim)
            and out["mapper"]["hidden_next"].shape == (B, ModuleDim.MapperHiddenDim)
        )

    def TestAdaptiveBeliefSourceGate(self) -> bool:
        torch.manual_seed(17)
        model = BeliefAssembler(
            memDim=4,
            intentDim=4,
            valueDim=4,
            vNextDim=4,
            worldHzxDim=4,
            beliefDim=8,
            hidden=8,
            drop=0.0).to(self.device).eval()
        inputs = {
            "memFeat": torch.randn(2, 4, device=self.device),
            "intentFeat": torch.randn(2, 4, device=self.device),
            "valueTensor": torch.randn(2, 4, device=self.device),
            "vNextTensor": torch.randn(2, 4, device=self.device),
            "uncertainty": torch.rand(2, device=self.device),
            "confidence": torch.rand(2, device=self.device),
            "precision": torch.rand(2, device=self.device),
            "risk": torch.rand(2, device=self.device),
            "worldHzx": torch.randn(2, 4, device=self.device),}
        with torch.no_grad():
            m = model.p_mem(model.ln_mem(inputs["memFeat"]))
            i = model.p_intent(model.ln_intent(inputs["intentFeat"]))
            v = model.p_value(model.ln_value(inputs["valueTensor"]))
            vn = model.p_vnext(model.ln_vnext(inputs["vNextTensor"]))
            w = model.p_world(model.ln_world(inputs["worldHzx"]))
            scalars = torch.stack([
                inputs["uncertainty"],
                inputs["confidence"],
                inputs["precision"],
                inputs["risk"],
            ], dim=-1)
            s = model.p_scalar(scalars)
            fused = torch.cat([m, i, v, vn, w, s], dim=-1)
            expected = model.ln_out(
                m + i + v + vn + w + s
                + model.gate(fused) * model.layerscale * model.fuse(fused))
            neutral = model(**inputs)

            changed_inputs = dict(inputs)
            changed_inputs["intentFeat"] = (
                inputs["intentFeat"]
                * inputs["intentFeat"].new_tensor([4.0, -3.0, 2.0, -1.0])
                + inputs["intentFeat"].new_tensor([3.0, -2.0, 1.0, 0.0]))
            changed_neutral = model(**changed_inputs)

            model.fuse[-1].weight.normal_(0.0, 0.1)
            model.source_gate.bias.copy_(
                torch.tensor([20.0, -20.0, -20.0, -20.0, -20.0, -20.0], device=self.device))
            intent_a = model(**inputs)
            intent_b = model(**changed_inputs)
        return bool(
            torch.allclose(neutral, expected, atol=1e-6, rtol=1e-6)
            and not torch.allclose(neutral, changed_neutral)
            and torch.allclose(intent_a, intent_b, atol=1e-5, rtol=1e-5))

    def TestLearnedOptionTransitionPrior(self) -> bool:
        torch.manual_seed(19)
        model = DecisionExtractor(
            stateDim=8,
            intentDim=8,
            valueTensorDim=4,
            vNextTensorDim=4,
            worldHDim=2,
            worldZDim=1,
            worldXDim=1,
            beliefDim=16,
            decisionDynDim=4,
            latentControlDim=2,
            mapperEmbedDim=4,
            actionEmbedDim=4,
            hiddenDim=16,
            psiDim=4,
            optionNum=3,
            useHebb=False).to(self.device)
        policy_input = torch.randn(1, 10, device=self.device)
        previous_logits = torch.tensor(
            [[12.0, -12.0, -12.0]], device=self.device)
        base_logits = model.option_head(policy_input) + 0.05 * previous_logits
        neutral_logits = model.OptionLogits(policy_input, previous_logits)
        with torch.no_grad():
            model.option_transition_prior.transition_logits[0, 2] = 8.0
        learned_logits = model.OptionLogits(policy_input, previous_logits)
        transition_bias = learned_logits - base_logits
        loss = -model.OptionLogProb(
            policy_input,
            previous_logits,
            torch.tensor([1], device=self.device)).mean()
        loss.backward()
        transition_grad = model.option_transition_prior.transition_logits.grad
        transition_row_grad = transition_grad.norm(dim=-1)
        return bool(
            torch.allclose(neutral_logits, base_logits, atol=1e-6, rtol=1e-6)
            and int(transition_bias.argmax(dim=-1).item()) == 2
            and transition_grad is not None
            and torch.isfinite(transition_grad).all().item()
            and torch.count_nonzero(transition_grad).item() > 0
            and transition_row_grad[0] > 1000.0 * transition_row_grad[1:].amax())

    def TestEndpointTokensDirectlyConditionLocalAction(self) -> bool:
        torch.manual_seed(21)
        decoupler = DecisionDecouplerV2(decisionDim=32).to(self.device).eval()
        with torch.no_grad():
            initially_neutral = (
                torch.count_nonzero(
                    decoupler.endpoint_action_refiner.net[-1].weight).item() == 0
                and torch.count_nonzero(
                    decoupler.endpoint_action_refiner.net[-1].bias).item() == 0)
            decoupler.endpoint_action_refiner.net[-1].weight.normal_(0.0, 0.05)
        B = 1
        endpoint_tokens = torch.randn(
            B,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseFeatDim,
            device=self.device)
        changed_tokens = endpoint_tokens.clone()
        changed_tokens[:, 0] = changed_tokens[:, 0] + 0.25
        endpoint_feat = torch.randn(
            B,
            ModuleDim.DecisionEndpointPoseFeatDim,
            device=self.device)
        base_pose = torch.zeros(
            B,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseDim,
            device=self.device)
        base_pose[..., 6] = 1.0
        common = {
            "decisionBackbone": torch.randn(B, 32, device=self.device),
            "planLatent": torch.randn(B, 256, device=self.device),
            "subgoalFeature": torch.randn(
                B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
            "constraintTokens": torch.randn(B, 8, 128, device=self.device),
            "baseEndpointPose": base_pose,
            "risk": torch.zeros(B, device=self.device),
            "confidence": torch.ones(B, device=self.device),
            "precision": torch.ones(B, device=self.device),}
        first = decoupler(
            endpointPoseEncoding=EndpointPoseEncoding(endpoint_tokens, endpoint_feat),
            **common).decision_tensor
        second = decoupler(
            endpointPoseEncoding=EndpointPoseEncoding(changed_tokens, endpoint_feat),
            **common).decision_tensor
        difference = (second - first).abs()
        return bool(
            initially_neutral
            and difference[:, 0].amax().item() > 1e-7
            and torch.count_nonzero(difference[:, 1:] > 1e-7).item() == 0)

    def TestStructuralEnhancementsReceiveActionGradient(self) -> bool:
        torch.manual_seed(22)
        model = DecisionExtractor(
            stateDim=16,
            intentDim=16,
            valueTensorDim=8,
            vNextTensorDim=8,
            worldHDim=4,
            worldZDim=2,
            worldXDim=2,
            beliefDim=32,
            decisionDynDim=8,
            latentControlDim=4,
            mapperEmbedDim=8,
            actionEmbedDim=6,
            hiddenDim=32,
            psiDim=8,
            optionNum=3,
            useHebb=False).to(self.device)
        decoupler = DecisionDecouplerV2(decisionDim=32).to(self.device)
        B = 2
        out = model(
            torch.randn(B, 16, device=self.device),
            torch.randn(B, 16, device=self.device),
            valueTensor=torch.randn(B, 8, device=self.device),
            vNextTensor=torch.randn(B, 8, device=self.device),
            prevOptionLogit=torch.randn(B, 3, device=self.device),
            uncertainty=torch.rand(B, device=self.device),
            confidence=torch.rand(B, device=self.device),
            precision=torch.rand(B, device=self.device),
            risk=torch.rand(B, device=self.device),
            worldHzx=torch.randn(B, 8, device=self.device),
            prevDecisionState=torch.randn(B, 8, device=self.device),
            prevLatentControl=torch.randn(B, 4, device=self.device),
            prevActionEmbed=torch.randn(B, 6, device=self.device),
            prevMapperHidden=torch.randn(B, 8, device=self.device),
            feedbackTdError=torch.zeros(B, device=self.device),
            sample=True,
            deterministic=False,)
        decision_feature = model.final_decision_norm(
            out["belief"] + out["option"]["option_context"])
        base_pose = torch.zeros(
            B,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseDim,
            device=self.device)
        base_pose[..., 6] = 1.0
        zero_error = torch.zeros(
            B,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionActionDim,
            device=self.device)
        endpoint_encoding = decoupler.EncodeEndpointPose(
            base_pose, zero_error, zero_error)
        decision = decoupler(
            decision_feature,
            torch.randn(B, 256, device=self.device),
            torch.randn(
                B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
            torch.randn(B, 8, 128, device=self.device),
            endpoint_encoding,
            base_pose,
            risk=torch.rand(B, device=self.device),
            confidence=torch.rand(B, device=self.device),
            precision=torch.rand(B, device=self.device))
        target = torch.randn_like(decision.decision_tensor)
        F.smooth_l1_loss(decision.decision_tensor, target).backward()
        gradients = (
            model.belief_assembler.source_gate.weight.grad,
            model.option_transition_prior.transition_logits.grad,
            decoupler.endpoint_action_refiner.net[-1].weight.grad,)
        return all(
            grad is not None
            and torch.isfinite(grad).all().item()
            and torch.count_nonzero(grad).item() > 0
            for grad in gradients)

    def TestPredictiveCoreContractsForAnyPrecision(self) -> bool:
        torch.manual_seed(23)
        core = PredictiveDecisionCore(
            beliefDim=8,
            uDim=4,
            dynDim=8,
            nSteps=2,
            hidden=16,
            drop=0.0).to(self.device).eval()
        belief = torch.randn(4, 8, device=self.device)
        target = core.belief_to_dyn(belief)
        offset = torch.randn_like(target)
        prev = target + offset
        u = torch.zeros(4, 4, device=self.device)
        precision = torch.tensor([0.1, 1.0, 20.0, 100.0], device=self.device).view(-1, 1)
        out = core(prev, belief, u, precision)
        ratio = (out - target).norm(dim=-1) / offset.norm(dim=-1)
        h = prev[-1:]
        for _ in range(1000):
            h = core(h, belief[-1:], u[-1:], precision[-1:])
        return bool(
            torch.all(ratio[1:] < ratio[:-1]).item()
            and torch.all(ratio < 1.0).item()
            and torch.isfinite(h).all().item()
            and F.softplus(core.drift_scale).item() > 0.0)

    def TestEligibilityUsesOldTraceAndHonorsDisable(self) -> bool:
        disabled = EligibilityTracePlasticityLayer(3, 2, enabled=False).to(self.device)
        x = torch.ones(2, 3, device=self.device)
        disabled(x, torch.ones(2, device=self.device))
        disabled_ok = disabled.trace.numel() == 0 and disabled.fast.numel() == 0

        layer = EligibilityTracePlasticityLayer(
            3, 2, lam=0.5, eta=0.1, gamma=0.0,
            applyScale=1.0, maxRowNorm=0.0, enabled=True).to(self.device)
        layer.EnsureBatch(2, self.device, x.dtype)
        layer.trace.fill_(1.0)
        layer.fast.zero_()
        post = layer(x, torch.tensor([1.0, 0.0], device=self.device))
        fast_after_feedback = layer.fast.detach().clone()
        layer.Commit(x, post, torch.tensor([1.0, 0.0], device=self.device))
        layer.fast[1].fill_(1.0)
        layer.ClearTrace(torch.tensor([False, True], device=self.device))
        return bool(
            disabled_ok
            and torch.allclose(fast_after_feedback[0], torch.full_like(fast_after_feedback[0], 0.1))
            and torch.count_nonzero(fast_after_feedback[1]).item() == 0
            and torch.count_nonzero(layer.trace[1]).item() == 0
            and torch.all(layer.fast[1] == 1.0).item())

    def TestOptionSelectionChangesPhysicalAction(self) -> bool:
        torch.manual_seed(29)
        model = DecisionExtractor(
            stateDim=16,
            intentDim=16,
            valueTensorDim=8,
            vNextTensorDim=8,
            worldHDim=4,
            worldZDim=2,
            worldXDim=2,
            beliefDim=32,
            decisionDynDim=8,
            latentControlDim=4,
            mapperEmbedDim=8,
            actionEmbedDim=6,
            hiddenDim=32,
            psiDim=8,
            optionNum=2,
            useHebb=False).to(self.device).eval()
        decoupler = DecisionDecouplerV2(decisionDim=32).to(self.device).eval()
        with torch.no_grad():
            model.option_head[-1].weight.zero_()
            model.option_psi_head[-1].weight.zero_()
            psi_bias = model.option_psi_head[-1].bias.view(2, 8)
            psi_bias[0].copy_(torch.linspace(-1.0, 1.0, 8, device=self.device))
            psi_bias[1].copy_(torch.linspace(1.0, -1.0, 8, device=self.device))

        inputs = {
            "stateFeat": torch.randn(1, 16, device=self.device),
            "intentFeat": torch.randn(1, 16, device=self.device),
            "valueTensor": torch.randn(1, 8, device=self.device),
            "vNextTensor": torch.randn(1, 8, device=self.device),
            "uncertainty": torch.zeros(1, device=self.device),
            "confidence": torch.ones(1, device=self.device),
            "precision": torch.ones(1, device=self.device),
            "risk": torch.zeros(1, device=self.device),
            "worldHzx": torch.randn(1, 8, device=self.device),
            "prevDecisionState": torch.zeros(1, 8, device=self.device),
            "prevLatentControl": torch.zeros(1, 4, device=self.device),
            "prevActionEmbed": torch.zeros(1, 6, device=self.device),
            "prevMapperHidden": torch.zeros(1, 8, device=self.device),
            "feedbackTdError": torch.zeros(1, device=self.device),
            "prevOptionLogit": torch.zeros(1, 2, device=self.device),
            "sample": True,
            "deterministic": True,}
        pose = torch.zeros(1, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim, device=self.device)
        pose[..., 6] = 1.0
        zero_error = pose.new_zeros(1, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim)
        encoding = decoupler.EncodeEndpointPose(pose, zero_error, zero_error)
        plan = torch.randn(1, 256, device=self.device)
        subgoal = torch.randn(1, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device)
        constraints = torch.randn(1, 8, 128, device=self.device)

        def action_for(option: int):
            with torch.no_grad():
                model.option_head[-1].bias.copy_(
                    torch.tensor([10.0, -10.0] if option == 0 else [-10.0, 10.0], device=self.device))
            out = model(**inputs)
            feature = model.final_decision_norm(
                out["belief"] + out["option"]["option_context"])
            action = decoupler(
                feature, plan, subgoal, constraints, encoding, pose,
                risk=inputs["risk"],
                confidence=inputs["confidence"],
                precision=inputs["precision"])
            return out, action.decision_tensor

        _, action0 = action_for(0)
        out1, action1 = action_for(1)
        model.zero_grad(set_to_none=True)
        decoupler.zero_grad(set_to_none=True)
        action1.square().mean().backward()
        psi_grad = model.option_psi_head[-1].bias.grad
        return bool(
            not torch.allclose(action0, action1)
            and psi_grad is not None
            and torch.count_nonzero(psi_grad).item() > 0
            and int(out1["option"]["opt_idx"].item()) == 1)

    def TestPhysicalProjectionAndSafetySemantics(self) -> bool:
        decoupler = DecisionDecouplerV2(decisionDim=32).to(self.device).eval()
        latent = torch.full(
            (2, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim),
            100.0,
            device=self.device)
        action = decoupler.ProjectDecisionLatent(latent)
        normalized = decoupler.NormalizeDecisionTensor(action)
        zero_latent = torch.zeros_like(latent, requires_grad=True)
        decoupler.ProjectDecisionLatent(zero_latent).sum().backward()
        zero_jacobian_probe = zero_latent.grad
        active_mask = decoupler.action_projector.action_mask.expand_as(zero_jacobian_probe)
        camera = ModuleDim.DecisionEndpointNames.index("camera")
        safety_zero = decoupler.SafetyScores(
            torch.zeros_like(action),
            torch.zeros(2, device=self.device),
            torch.ones(2, device=self.device),
            torch.ones(2, device=self.device))
        safety_edge = decoupler.SafetyScores(
            action,
            torch.zeros(2, device=self.device),
            torch.ones(2, device=self.device),
            torch.ones(2, device=self.device))
        active_dof_count = int(decoupler.action_projector.action_mask.sum().item())
        action_heads_parameterize_only_active_dofs = all(
            head.net[-1].out_features == active_dof_count
            for head in (
                decoupler.action_head,
                decoupler.dynamics_head,
                decoupler.residual_compensator))
        base_pose = torch.zeros(
            1,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionEndpointPoseDim,
            device=self.device)
        half_turn = math.pi / 4.0
        base_pose[..., 2] = 0.2
        base_pose[..., 5] = math.sin(half_turn)
        base_pose[..., 6] = math.cos(half_turn)
        local_delta = torch.zeros(
            1,
            ModuleDim.DecisionEndpointCount,
            ModuleDim.DecisionActionDim,
            device=self.device)
        local_delta[..., 0] = 0.03
        local_delta[..., 3] = 0.1
        local_delta = decoupler.MaskDecisionTensor(local_delta)
        target_pose = decoupler.DecodeEndpointPose(base_pose, local_delta)
        recovered_delta = RelativePoseError(base_pose, target_pose)
        current_pose = base_pose.clone()
        current_pose[..., :3] = (
            base_pose[..., :3]
            + base_pose.new_tensor([0.0, 0.02, 0.0]))
        command = MotionCommand(
            decision_tensor=local_delta,
            target_endpoint_pose=target_pose,
            endpoint_names=ModuleDim.DecisionEndpointNames,
            decision_dof_mask=decoupler.action_projector.action_mask.bool(),
            gripper_cmd=torch.full(
                (1, ModuleDim.ArmCount, 1),
                0.5,
                device=self.device),
            gripper_valid=torch.zeros(1, device=self.device, dtype=torch.bool),
            mode_logits=torch.zeros(
                1,
                ModuleDim.ActTypeDim,
                device=self.device),
            mode_valid=torch.zeros(1, device=self.device, dtype=torch.bool),
            safety_scores=torch.ones(1, 5, device=self.device),
            safety_names=SAFETY_MARGIN_NAMES)
        rebased_command = decoupler.RebaseMotionCommand(
            command,
            current_pose,
            torch.zeros(1, device=self.device),
            torch.ones(1, device=self.device),
            torch.ones(1, device=self.device))
        expected_rebased_delta = local_delta.clone()
        expected_rebased_delta[..., 0] = 0.01
        expected_rebased_delta = decoupler.MaskDecisionTensor(expected_rebased_delta)
        expected_rebased_target = target_pose.clone()
        for endpoint_index in ModuleDim.DecisionRotationOnlyEndpoints:
            expected_rebased_target[:, endpoint_index, :3] = (
                current_pose[:, endpoint_index, :3])
        translated_endpoint_indices = tuple(
            index
            for index in range(ModuleDim.DecisionEndpointCount)
            if index not in ModuleDim.DecisionRotationOnlyEndpoints)
        return bool(
            action_heads_parameterize_only_active_dofs
            and torch.all(zero_jacobian_probe[active_mask.bool()] > 0.0).item()
            and torch.count_nonzero(zero_jacobian_probe[~active_mask.bool()]).item() == 0
            and torch.all(normalized.abs() <= 1.0).item()
            and torch.all(normalized[..., :3].norm(dim=-1) <= 1.0 + 1e-6).item()
            and torch.all(normalized[..., 3:].norm(dim=-1) <= 1.0 + 1e-6).item()
            and torch.count_nonzero(action[:, camera, :3]).item() == 0
            and torch.count_nonzero(action[:, camera, 5]).item() == 0
            and torch.all(safety_edge[:, :2] < safety_zero[:, :2]).item()
            and torch.allclose(recovered_delta, local_delta, atol=1e-5, rtol=1e-5)
            and torch.allclose(
                target_pose[:, translated_endpoint_indices, :2],
                base_pose[:, translated_endpoint_indices, :2]
                + base_pose.new_tensor([0.0, 0.03]),
                atol=1e-5,
                rtol=1e-5)
            and torch.allclose(
                rebased_command.decision_tensor,
                expected_rebased_delta,
                atol=1e-5,
                rtol=1e-5)
            and torch.equal(
                rebased_command.target_endpoint_pose,
                expected_rebased_target))

    def TestNeuroSymbolicRefineShapes(self) -> bool:
        B = 2
        model = DecisionExtractor(
            stateDim=ModuleDim.MemoryFeat,
            intentDim=ModuleDim.IntentionFeat,
            hiddenDim=256,
            psiDim=256,
            optionNum=8,
            useHebb=False,
        ).to(self.device)
        base = model(
            torch.randn(B, ModuleDim.MemoryFeat, device=self.device),
            torch.randn(B, ModuleDim.IntentionFeat, device=self.device),
            valueTensor=torch.randn(B, model.value_tensor_dim, device=self.device),
            vNextTensor=torch.randn(B, model.v_next_tensor_dim, device=self.device),
            prevOptionLogit=torch.zeros(B, 8, device=self.device),
            uncertainty=torch.zeros(B, device=self.device),
            confidence=torch.ones(B, device=self.device),
            precision=torch.ones(B, device=self.device),
            risk=torch.zeros(B, device=self.device),
            worldHzx=torch.randn(B, ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState, device=self.device),
            prevDecisionState=torch.zeros(B, ModuleDim.DecisionDynDim, device=self.device),
            prevLatentControl=torch.zeros(B, ModuleDim.LatentControlDim, device=self.device),
            prevActionEmbed=torch.zeros(B, ModuleDim.DecisionFeedbackEmbedDim, device=self.device),
            prevMapperHidden=torch.zeros(B, ModuleDim.MapperHiddenDim, device=self.device),
            feedbackTdError=torch.zeros(B, device=self.device),
        )
        nesy = NeuroSymbolicOutput(
            facts=[],
            operator_logits=torch.randn(B, len(OPERATORS), device=self.device),
            plan_steps=[],
            operator_rationales=[],
            plan_latent=torch.randn(B, 256, device=self.device),
            subgoal_feature=torch.randn(B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
            constraint_tokens=torch.randn(B, 8, 128, device=self.device),
            risk_cause_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            risk_cause_raw_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            failure_cause_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            failure_cause_raw_logits=torch.randn(B, len(FAILURE_CAUSES), device=self.device),
            failure_gate_logits=torch.randn(B, device=self.device),
            failure_gate=torch.sigmoid(torch.randn(B, device=self.device)),
            invoke_mask=torch.sigmoid(torch.randn(B, device=self.device)),
            same_operator=torch.zeros(B, device=self.device),
            operator_changed=torch.zeros(B, device=self.device),
            invoke_delta=torch.zeros(B, device=self.device),
            reference_drift=torch.zeros(B, device=self.device),
            temporal_logits=torch.randn(B, ModuleDim.TemporalPrimitiveCount, device=self.device),
            temporal_reason_logits=torch.randn(B, ModuleDim.TemporalReasonDim, device=self.device),
            continue_guard_score=torch.sigmoid(torch.randn(B, device=self.device)),
            interrupt_guard_score=torch.sigmoid(torch.randn(B, device=self.device)),
            redispatch_guard_score=torch.sigmoid(torch.randn(B, device=self.device)),)
        temporal_goal = {
            "goal_mode_logits": torch.randn(B, ModuleDim.TemporalPrimitiveCount, device=self.device),
            "goal_timeout_soft_ms": torch.rand(B, device=self.device) * 1000.0,
            "goal_timeout_hard_ms": torch.rand(B, device=self.device) * 2000.0,}
        out = model.RefineWithNeuroSymbolic(
            base,
            nesy,
            torch.randn(B, ModuleDim.DecisionEndpointPoseFeatDim, device=self.device),
            torch.randn(B, ModuleDim.WorldOutHState + ModuleDim.WorldOutZState + ModuleDim.WorldOutXState, device=self.device),
            torch.randn(B, ModuleDim.PstSlotDim, device=self.device),
            torch.randn(B, ModuleDim.DecisionBeliefDim, device=self.device),
            temporal_goal,)
        return (
            out["decision_feature"].shape == (B, ModuleDim.DecisionBeliefDim)
            and out["decoder_plan_latent"].shape == (B, 256)
            and out["decoder_subgoal_feature"].shape == (B, ModuleDim.DecisionEndpointPoseFeatDim)
            and out["decoder_constraint_tokens"].shape == (B, 8, 128)
            and out["decision_energy"].shape == (B,)
            and out["temporal_decision"]["kind_logits"].shape == (B, ModuleDim.TemporalPrimitiveCount)
        )

    def TestCEMPlannerUsesUnifiedPrior(self) -> bool:
        try:
            class FakeWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.calls = 0

                def StepPriorOnly(
                    self,
                    h,
                    z,
                    x,
                    actionEnc,
                    *,
                    physicalState,
                    robotSelfState,
                    sample=False,):
                    self.calls += 1
                    return {
                        "r_pred": actionEnc.new_zeros(actionEnc.size(0)),
                        "d_prob": actionEnc.new_zeros(actionEnc.size(0)),}

            class FakeDecoupler(nn.Module):
                def MaskDecisionTensor(self, value):
                    masked = value.clone()
                    camera = ModuleDim.DecisionEndpointNames.index("camera")
                    masked[..., camera, :3] = 0.0
                    masked[..., camera, 5] = 0.0
                    return masked

                def ProjectDecisionLatent(self, value):
                    return value

                def EncodeEndpointPose(self, pose, targetError, plannerError):
                    feat = pose.new_zeros(pose.size(0), ModuleDim.DecisionEndpointPoseFeatDim)
                    return EndpointPoseEncoding(
                        endpoint_pose_tokens=pose.new_zeros(
                            pose.size(0),
                            ModuleDim.DecisionEndpointCount,
                            ModuleDim.DecisionEndpointPoseFeatDim),
                        endpoint_pose_feat=feat)

                def DecodeEndpointPose(self, pose, action):
                    return pose

                def EncodeDecisionFeedback(self, action, targetPose, encoding):
                    return action.new_zeros(action.size(0), ModuleDim.DecisionFeedbackEmbedDim)

            B = 1
            world = FakeWorld()
            decoupler = FakeDecoupler()
            planner = CEMPlanner(
                worldModel=world,
                wmIsOnlineWrapper=True,
                decisionDecoupler=decoupler,
                N=2,
                elite=1,
                iters=1)
            endpoint_pose = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            endpoint_encoding = decoupler.EncodeEndpointPose(
                endpoint_pose,
                torch.zeros(B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim),
                torch.zeros(B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim))
            out = planner.Plan(
                decisionLatent=torch.zeros(
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim),
                endpointPose=endpoint_pose,
                endpointPoseEncoding=endpoint_encoding,
                h0=torch.zeros(B, 2),
                z0=torch.zeros(B, 2),
                x0=torch.zeros(B, 2),
                physicalState={"SlotPresence": torch.ones(B, 1)},
                robotSelfState=torch.zeros(B, 2))
            camera = ModuleDim.DecisionEndpointNames.index("camera")
            ok = (
                world.calls == 2
                and torch.count_nonzero(out["decision_latent"][:, camera, :3]).item() == 0
                and torch.count_nonzero(out["decision_latent"][:, camera, 5]).item() == 0)
            print(f"CEM unified prior contract {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"CEM unified prior contract error: {e}")
            return False

    def TestCEMPlannerChunksCandidateBatch(self) -> bool:
        try:
            class FakeWorld(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.batch_sizes: List[int] = []

                def StepPriorOnly(
                    self,
                    h,
                    z,
                    x,
                    actionEnc,
                    *,
                    physicalState,
                    robotSelfState,
                    sample=False,):
                    self.batch_sizes.append(int(actionEnc.size(0)))
                    assert int(physicalState["SlotPresence"].size(0)) == int(actionEnc.size(0))
                    return {
                        "r_pred": actionEnc.new_zeros(actionEnc.size(0)),
                        "d_prob": actionEnc.new_zeros(actionEnc.size(0)),}

            class FakeDecoupler(nn.Module):
                def MaskDecisionTensor(self, value):
                    return value

                def ProjectDecisionLatent(self, value):
                    return value

                def EncodeEndpointPose(self, pose, targetError, plannerError):
                    return EndpointPoseEncoding(
                        endpoint_pose_tokens=pose.new_zeros(
                            pose.size(0),
                            ModuleDim.DecisionEndpointCount,
                            ModuleDim.DecisionEndpointPoseFeatDim),
                        endpoint_pose_feat=pose.new_zeros(
                            pose.size(0), ModuleDim.DecisionEndpointPoseFeatDim))

                def DecodeEndpointPose(self, pose, action):
                    return pose

                def EncodeDecisionFeedback(self, action, targetPose, encoding):
                    return action.new_zeros(action.size(0), ModuleDim.DecisionFeedbackEmbedDim)

            B = 2
            world = FakeWorld()
            decoupler = FakeDecoupler()
            planner = CEMPlanner(
                worldModel=world,
                wmIsOnlineWrapper=False,
                decisionDecoupler=decoupler,
                N=5,
                elite=1,
                iters=1,
                candidateChunkSize=2)
            endpoint_pose = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            endpoint_encoding = decoupler.EncodeEndpointPose(
                endpoint_pose,
                torch.zeros(B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim),
                torch.zeros(B, ModuleDim.DecisionEndpointCount, ModuleDim.DecisionActionDim))
            planner.Plan(
                decisionLatent=torch.zeros(
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim),
                endpointPose=endpoint_pose,
                endpointPoseEncoding=endpoint_encoding,
                h0=torch.zeros(B, 2),
                z0=torch.zeros(B, 2),
                x0=torch.zeros(B, 2),
                physicalState={"SlotPresence": torch.ones(B, 3)},
                robotSelfState=torch.zeros(B, 2))
            ok = world.batch_sizes == [B * 2, B * 2, B, B]
            print(f"CEM candidate chunking {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"CEM candidate chunking error: {e}")
            return False

    def TestCEMPlannerPreservesMeasuredEndpointEncoding(self) -> bool:
        try:
            class FakeWorld(nn.Module):
                def StepPriorOnly(
                    self,
                    h,
                    z,
                    x,
                    actionEnc,
                    *,
                    physicalState,
                    robotSelfState,
                    sample=False,):
                    return {
                        "r_pred": actionEnc.new_zeros(actionEnc.size(0)),
                        "d_prob": actionEnc.new_zeros(actionEnc.size(0)),}

            class FakeDecoupler(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.feedback_features: List[torch.Tensor] = []

                def MaskDecisionTensor(self, value):
                    return value

                def ProjectDecisionLatent(self, value):
                    return value

                def EncodeEndpointPose(self, pose, targetError, plannerError):
                    return EndpointPoseEncoding(
                        endpoint_pose_tokens=pose.new_zeros(
                            pose.size(0),
                            ModuleDim.DecisionEndpointCount,
                            ModuleDim.DecisionEndpointPoseFeatDim),
                        endpoint_pose_feat=pose.new_zeros(
                            pose.size(0), ModuleDim.DecisionEndpointPoseFeatDim))

                def DecodeEndpointPose(self, pose, action):
                    return pose

                def EncodeDecisionFeedback(self, action, targetPose, encoding):
                    self.feedback_features.append(encoding.endpoint_pose_feat.detach().clone())
                    return action.new_zeros(action.size(0), ModuleDim.DecisionFeedbackEmbedDim)

            B = 2
            decoupler = FakeDecoupler()
            planner = CEMPlanner(
                worldModel=FakeWorld(),
                wmIsOnlineWrapper=False,
                decisionDecoupler=decoupler,
                N=3,
                elite=1,
                iters=1,
                candidateChunkSize=2)
            endpoint_pose = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            measured_feat = torch.stack([
                torch.full((ModuleDim.DecisionEndpointPoseFeatDim,), 3.0),
                torch.full((ModuleDim.DecisionEndpointPoseFeatDim,), 7.0)])
            endpoint_encoding = EndpointPoseEncoding(
                endpoint_pose_tokens=torch.zeros(
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionEndpointPoseFeatDim),
                endpoint_pose_feat=measured_feat)
            planner.Plan(
                decisionLatent=torch.zeros(
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim),
                endpointPose=endpoint_pose,
                endpointPoseEncoding=endpoint_encoding,
                h0=torch.zeros(B, 2),
                z0=torch.zeros(B, 2),
                x0=torch.zeros(B, 2),
                physicalState={"SlotPresence": torch.ones(B, 1)},
                robotSelfState=torch.zeros(B, 2))
            observed = torch.cat(decoupler.feedback_features, dim=0)
            expected = torch.cat([
                measured_feat[:, None].expand(B, 2, -1).reshape(B * 2, -1),
                measured_feat,
                measured_feat], dim=0)
            ok = torch.equal(observed, expected)
            print(f"CEM measured endpoint encoding {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"CEM measured endpoint encoding error: {e}")
            return False

    def TestCEMPlannerScoresImmediateRewardBeforeDone(self) -> bool:
        try:
            class FakeWorld(nn.Module):
                def StepPriorOnly(
                    self,
                    h,
                    z,
                    x,
                    actionEnc,
                    *,
                    physicalState,
                    robotSelfState,
                    sample=False,):
                    if int(actionEnc.size(0)) == 1:
                        return {
                            "r_pred": actionEnc.new_tensor([10.0]),
                            "d_prob": actionEnc.new_tensor([1.0]),}
                    return {
                        "r_pred": actionEnc.new_tensor([1.0, 10.0]),
                        "d_prob": actionEnc.new_tensor([0.0, 1.0]),}

            class FakeDecoupler(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.actions: List[torch.Tensor] = []

                def MaskDecisionTensor(self, value):
                    return value

                def ProjectDecisionLatent(self, value):
                    return value

                def DecodeEndpointPose(self, pose, action):
                    return pose

                def EncodeDecisionFeedback(self, action, targetPose, encoding):
                    self.actions.append(action.detach().clone())
                    return action.new_zeros(
                        action.size(0), ModuleDim.DecisionFeedbackEmbedDim)

            torch.manual_seed(7)
            B = 1
            decoupler = FakeDecoupler()
            planner = CEMPlanner(
                worldModel=FakeWorld(),
                wmIsOnlineWrapper=False,
                decisionDecoupler=decoupler,
                N=2,
                elite=1,
                iters=1,
                momentum=0.0,
                candidateChunkSize=2)
            endpoint_pose = torch.zeros(
                B,
                ModuleDim.DecisionEndpointCount,
                ModuleDim.DecisionEndpointPoseDim)
            endpoint_pose[..., 6] = 1.0
            endpoint_encoding = EndpointPoseEncoding(
                endpoint_pose_tokens=torch.zeros(
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionEndpointPoseFeatDim),
                endpoint_pose_feat=torch.zeros(
                    B, ModuleDim.DecisionEndpointPoseFeatDim))
            out = planner.Plan(
                decisionLatent=torch.zeros(
                    B,
                    ModuleDim.DecisionEndpointCount,
                    ModuleDim.DecisionActionDim),
                endpointPose=endpoint_pose,
                endpointPoseEncoding=endpoint_encoding,
                h0=torch.zeros(B, 2),
                z0=torch.zeros(B, 2),
                x0=torch.zeros(B, 2),
                physicalState={"SlotPresence": torch.ones(B, 1)},
                robotSelfState=torch.zeros(B, 2))
            expected = decoupler.actions[0][1:2]
            ok = (
                torch.allclose(out["decision_tensor"], expected)
                and torch.allclose(out["expected_return"], out["expected_return"].new_tensor([9.0])))
            print(f"CEM immediate terminal reward {'passed' if ok else 'failed'}.")
            return bool(ok)
        except Exception as e:
            print(f"CEM immediate terminal reward error: {e}")
            return False

    def RunAllTests(self) -> Dict[str, bool]:
        return {
            "DecisionExtractorIOShapes": self.TestDecisionExtractorIOShapes(),
            "AdaptiveBeliefSourceGate": self.TestAdaptiveBeliefSourceGate(),
            "LearnedOptionTransitionPrior": self.TestLearnedOptionTransitionPrior(),
            "EndpointTokensDirectlyConditionLocalAction": self.TestEndpointTokensDirectlyConditionLocalAction(),
            "StructuralEnhancementsReceiveActionGradient": self.TestStructuralEnhancementsReceiveActionGradient(),
            "PredictiveCoreContractsForAnyPrecision": self.TestPredictiveCoreContractsForAnyPrecision(),
            "EligibilityUsesOldTraceAndHonorsDisable": self.TestEligibilityUsesOldTraceAndHonorsDisable(),
            "OptionSelectionChangesPhysicalAction": self.TestOptionSelectionChangesPhysicalAction(),
            "PhysicalProjectionAndSafetySemantics": self.TestPhysicalProjectionAndSafetySemantics(),
            "NeuroSymbolicRefineShapes": self.TestNeuroSymbolicRefineShapes(),
            "CEMPlannerUsesUnifiedPrior": self.TestCEMPlannerUsesUnifiedPrior(),
            "CEMPlannerChunksCandidateBatch": self.TestCEMPlannerChunksCandidateBatch(),
            "CEMPlannerPreservesMeasuredEndpointEncoding": self.TestCEMPlannerPreservesMeasuredEndpointEncoding(),
            "CEMPlannerScoresImmediateRewardBeforeDone": self.TestCEMPlannerScoresImmediateRewardBeforeDone(),}
