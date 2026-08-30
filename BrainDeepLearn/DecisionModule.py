from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        outDim: int,):
        super().__init__()
        self.in_dim = int(inDim)
        self.out_dim = int(outDim)

        self.base = nn.Parameter(torch.randn(outDim, inDim, device=self.device, dtype=self.dtype) * 0.02)
        self.register_buffer("trace", torch.empty(0), persistent=False)
        self.register_buffer("fast", torch.empty(0), persistent=False)

    def EnsureB(self, B: int):
        if int(self.trace.size(0)) != B:
            self.trace = self.base.new_zeros(B, self.out_dim, self.in_dim)
            self.fast = self.base.new_zeros(B, self.out_dim, self.in_dim)

    def forward(self, x: torch.Tensor, neuromod: torch.Tensor) -> torch.Tensor:
        B = int(x.size(0))
        fast = self.fast.detach().clone()
        w_eff = self.base.unsqueeze(0) + 0.25 * fast
        out = torch.einsum("bi,boi->bo", x, w_eff)

        with torch.no_grad():
            mod = neuromod.detach().view(B, 1, 1)
            self.fast = 0.99 * self.fast + 1e-3 * mod * self.trace
            flat = self.fast.reshape(B, -1)
            nrm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
            scale = (2.0 / nrm).clamp_max(1.0)
            self.fast = self.fast * scale.view(B, 1, 1)

        return out

    def Commit(self, pre: torch.Tensor, post: torch.Tensor, executeMask: torch.Tensor):
        B = int(pre.size(0))
        with torch.no_grad():
            outer = torch.einsum("bo,bi->boi", post.detach(), pre.detach())
            write = executeMask.view(B, 1, 1)
            self.trace = 0.9 * self.trace + 0.1 * write * outer

    def ClearTrace(self, invalidMask: torch.Tensor):
        if self.trace.numel() == 0:
            return
        with torch.no_grad():
            self.trace.masked_fill_(invalidMask.view(-1, 1, 1).bool(), 0.0)

    def Reset(self, doneMask: Optional[torch.Tensor] = None):
        with torch.no_grad():
            if doneMask is None:
                self.trace.zero_()
                self.fast.zero_()
                return
            mask = doneMask.view(-1)
            if mask.numel() != self.trace.size(0):
                raise ValueError("Decision Hebbian reset mask must match its batch size")
            self.trace[mask] = 0
            self.fast[mask] = 0


class NeuroSymbolicConditioner(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        planDim: int = 256,
        subgoalFeatureDim: int = 128,
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


class FastDecisionPolicy(AGICoreModule):
    def __init__(
        self,
        sourceDims: Tuple[int, ...],
        beliefDim: int,
        dynDim: int,
        latentDim: int,
        optionCount: int,
        mapperDim: int,
        sourceDim: int,
        hiddenDim: int,
    ):
        super().__init__()
        self.source_encoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(int(source_dim)),
                nn.Linear(int(source_dim), int(sourceDim)),
                nn.SiLU())
            for source_dim in sourceDims])
        fused_dim = len(sourceDims) * int(sourceDim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, int(hiddenDim)),
            nn.SiLU(),
            nn.Linear(int(hiddenDim), int(hiddenDim)),
            nn.SiLU())
        self.output_dims = (
            int(beliefDim),
            int(dynDim),
            int(latentDim),
            int(latentDim),
            int(optionCount),
            int(mapperDim),
            int(beliefDim))
        self.residual_head = nn.Linear(
            int(hiddenDim),
            sum(self.output_dims))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        sources: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        if len(sources) != len(self.source_encoders):
            raise ValueError("Fast decision source count does not match")
        encoded = [
            encoder(source)
            for encoder, source in zip(self.source_encoders, sources)]
        hidden = self.trunk(torch.cat(encoded, dim=-1))
        residual = self.residual_head(hidden)
        return tuple(torch.split(residual, self.output_dims, dim=-1))


class DetailDecisionPolicy(AGICoreModule):
    def __init__(
        self,
        beliefDim: int,
        contextDim: int,
        dynDim: int,
        latentDim: int,
        optionCount: int,
        mapperDim: int,
        hiddenDim: int,
    ):
        super().__init__()
        self.belief_encoder = nn.Sequential(
            nn.LayerNorm(int(beliefDim)),
            nn.Linear(int(beliefDim), int(hiddenDim)),
            nn.SiLU())
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(int(contextDim)),
            nn.Linear(int(contextDim), int(hiddenDim)),
            nn.SiLU())
        self.trunk = nn.Sequential(
            nn.LayerNorm(2 * int(hiddenDim)),
            nn.Linear(2 * int(hiddenDim), int(hiddenDim)),
            nn.SiLU())
        self.output_dims = (
            int(beliefDim),
            int(dynDim),
            int(latentDim),
            int(latentDim),
            int(optionCount),
            int(mapperDim),
            int(beliefDim))
        self.residual_head = nn.Linear(
            int(hiddenDim),
            sum(self.output_dims))
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        belief: torch.Tensor,
        detailContext: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        hidden = self.trunk(torch.cat([
            self.belief_encoder(belief),
            self.context_encoder(detailContext)], dim=-1))
        residual = self.residual_head(hidden)
        return tuple(torch.split(residual, self.output_dims, dim=-1))




class DecisionExtractor(AGICoreModule):
    def __init__(
        self,
        stateDim: int = 1024,
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
        actionEmbedDim: int = 256,
        planFeatureDim: int = 256,
        subgoalFeatureDim: int = 128,
        constraintTokenCount: int = 8,
        constraintTokenDim: int = 128,
        bodyStateFeatureDim: int = 128,
        controlFeedbackFeatureDim: int = 128,
        embodimentContextFeatureDim: int = 8,
        embodimentStateFeatureDim: int = 128,
        sceneStateFeatureDim: int = 128,
        goalDecisionContextDim: Optional[int] = None,):
        super().__init__()
        dimensions = {
            "state": stateDim,
            "intent": intentDim,
            "value tensor": valueTensorDim,
            "next value tensor": vNextTensorDim,
            "world deterministic": worldHDim,
            "world stochastic": worldZDim,
            "world sequence": worldXDim,
            "belief": beliefDim,
            "decision dynamics": decisionDynDim,
            "latent control": latentControlDim,
            "mapper embedding": mapperEmbedDim,
            "action embedding": actionEmbedDim,
            "plan feature": planFeatureDim,
            "subgoal feature": subgoalFeatureDim,
            "constraint token count": constraintTokenCount,
            "constraint token": constraintTokenDim,
            "body-state feature": bodyStateFeatureDim,
            "control-feedback feature": controlFeedbackFeatureDim,
            "embodiment-context feature": embodimentContextFeatureDim,
            "embodiment-state feature": embodimentStateFeatureDim,
            "scene-state feature": sceneStateFeatureDim,
        }
        if goalDecisionContextDim is not None:
            dimensions["goal-decision context"] = goalDecisionContextDim
        invalid_dimensions = [
            name
            for name, value in dimensions.items()
            if type(value) is not int or int(value) < 1]
        if invalid_dimensions:
            raise ValueError(
                "Decision cognitive dimensions must be positive integers: "
                + ", ".join(invalid_dimensions))
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
        self.plan_feature_dim = int(planFeatureDim)
        self.subgoal_feature_dim = int(subgoalFeatureDim)
        self.constraint_token_count = int(constraintTokenCount)
        self.constraint_token_dim = int(constraintTokenDim)
        self.body_state_feature_dim = int(bodyStateFeatureDim)
        self.control_feedback_feature_dim = int(controlFeedbackFeatureDim)
        self.embodiment_context_feature_dim = int(
            embodimentContextFeatureDim)
        self.embodiment_state_feature_dim = int(embodimentStateFeatureDim)
        self.scene_state_feature_dim = int(sceneStateFeatureDim)
        self.goal_decision_context_dim = int(
            self.belief_dim
            if goalDecisionContextDim is None
            else goalDecisionContextDim)

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
            outDim=self.u_dim,)
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
            planDim=self.plan_feature_dim,
            subgoalFeatureDim=self.subgoal_feature_dim,
            constraintTokens=self.constraint_token_count,
            constraintTokenDim=self.constraint_token_dim,)
        embodied_state_input_dim = (
            self.body_state_feature_dim
            + self.control_feedback_feature_dim
            + self.embodiment_context_feature_dim)
        self.embodied_state_encoder = nn.Sequential(
            nn.LayerNorm(embodied_state_input_dim),
            nn.Linear(embodied_state_input_dim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(
                hiddenDim // 2,
                self.embodiment_state_feature_dim),
            nn.LayerNorm(self.embodiment_state_feature_dim),)
        refine_in_dim = (
            self.belief_dim
            + self.dyn_dim
            + self.u_dim
            + self.world_hzx_dim
            + self.scene_state_feature_dim
            + self.goal_decision_context_dim
            + self.belief_dim
            + self.embodiment_state_feature_dim)
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
            nn.LayerNorm(self.belief_dim + self.plan_feature_dim),
            nn.Linear(self.belief_dim + self.plan_feature_dim, hiddenDim),
            nn.SiLU(),
            nn.Linear(hiddenDim, self.plan_feature_dim),
            nn.LayerNorm(self.plan_feature_dim),)
        self.decoder_subgoal_head = nn.Sequential(
            nn.LayerNorm(
                self.belief_dim
                + self.subgoal_feature_dim
                + self.embodiment_state_feature_dim),
            nn.Linear(
                self.belief_dim
                + self.subgoal_feature_dim
                + self.embodiment_state_feature_dim,
                hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, self.subgoal_feature_dim),
            nn.LayerNorm(self.subgoal_feature_dim),)
        self.decoder_constraint_seed = nn.Linear(
            self.belief_dim,
            self.constraint_token_count * self.constraint_token_dim)
        self.decoder_constraint_head = nn.Sequential(
            nn.LayerNorm(2 * self.constraint_token_dim),
            nn.Linear(2 * self.constraint_token_dim, hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, self.constraint_token_dim),
            nn.LayerNorm(self.constraint_token_dim),)
        self.decision_energy_head = nn.Sequential(
            nn.LayerNorm(
                self.belief_dim
                + self.world_hzx_dim
                + self.embodiment_state_feature_dim),
            nn.Linear(
                self.belief_dim
                + self.world_hzx_dim
                + self.embodiment_state_feature_dim,
                hiddenDim // 2),
            nn.SiLU(),
            nn.Linear(hiddenDim // 2, 1),)
        self.temporal_decision_head = TemporalDecisionHead()
        fast_source_dim = max(8, min(32, self.belief_dim // 32))
        fast_hidden_dim = max(32, min(128, self.belief_dim // 4))
        self.fast_policy = FastDecisionPolicy(
            sourceDims=(
                self.stateDim,
                self.intentDim,
                self.value_tensor_dim,
                self.v_next_tensor_dim,
                self.world_hzx_dim,
                self.dyn_dim,
                self.u_dim,
                self.action_embed_dim,
                self.mapper_hidden_dim,
                self.belief_dim,
                5),
            beliefDim=self.belief_dim,
            dynDim=self.dyn_dim,
            latentDim=self.u_dim,
            optionCount=self.num_options,
            mapperDim=self.mapper_hidden_dim,
            sourceDim=fast_source_dim,
            hiddenDim=fast_hidden_dim)
        self.detail_policy = DetailDecisionPolicy(
            beliefDim=self.belief_dim,
            contextDim=self.goal_decision_context_dim,
            dynDim=self.dyn_dim,
            latentDim=self.u_dim,
            optionCount=self.num_options,
            mapperDim=self.mapper_hidden_dim,
            hiddenDim=fast_hidden_dim)

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

    def CommitEligibilityRows(
        self,
        eligibilityPre: torch.Tensor,
        eligibilityPost: torch.Tensor,
        executeMask: torch.Tensor,
        updateMask: torch.Tensor,
    ) -> None:
        if (
            updateMask.dtype != torch.bool
            or updateMask.dim() != 1
            or int(updateMask.numel()) != int(eligibilityPre.size(0))
            or updateMask.device != eligibilityPre.device
        ):
            raise ValueError("Decision eligibility update mask is invalid")
        previous_trace = self.elig_plasticity.trace.detach().clone()
        self.elig_plasticity.Commit(
            eligibilityPre,
            eligibilityPost,
            executeMask & updateMask)
        self.elig_plasticity.trace = torch.where(
            updateMask.view(-1, 1, 1),
            self.elig_plasticity.trace,
            previous_trace)

    def ClearInvalidEligibility(self, invalidMask: torch.Tensor) -> None:
        self.elig_plasticity.ClearTrace(invalidMask)

    @torch.no_grad()
    def ExportEligibilityState(self) -> Dict[str, torch.Tensor]:
        return {
            "trace": self.elig_plasticity.trace.detach().clone(),
            "fast": self.elig_plasticity.fast.detach().clone()}

    @torch.no_grad()
    def ImportEligibilityState(
        self,
        state: Dict[str, torch.Tensor],
        batchSize: int,
    ) -> None:
        if type(state) is not dict or set(state) != {"trace", "fast"}:
            raise ValueError("Decision eligibility state fields do not match")
        trace = state["trace"]
        fast = state["fast"]
        expected = (int(batchSize), self.u_dim, self.u_dim)
        if (
            not torch.is_tensor(trace)
            or not torch.is_tensor(fast)
            or tuple(trace.shape) != expected
            or tuple(fast.shape) != expected
            or trace.device != self.elig_plasticity.base.device
            or fast.device != self.elig_plasticity.base.device
            or trace.dtype != self.elig_plasticity.base.dtype
            or fast.dtype != self.elig_plasticity.base.dtype
            or not bool(torch.isfinite(trace).all().item())
            or not bool(torch.isfinite(fast).all().item())
        ):
            raise ValueError("Decision eligibility state does not match")
        self.elig_plasticity.trace = trace.detach().clone()
        self.elig_plasticity.fast = fast.detach().clone()

    def EnsureB(self, B: int) -> None:
        self.elig_plasticity.EnsureB(B)

    def SelectFullBatchRows(
        self,
        value: torch.Tensor,
        rowIndex: torch.Tensor,
        fullBatchSize: int,
        fieldName: str,
    ) -> torch.Tensor:
        if (
            not torch.is_tensor(value)
            or value.dim() < 1
            or int(value.size(0)) != fullBatchSize
        ):
            raise ValueError(
                f"Decision {fieldName} must have full batch size {fullBatchSize}")
        if value.device != rowIndex.device:
            raise ValueError(
                f"Decision {fieldName} must share the rowIndex device")
        return value.index_select(0, rowIndex)

    def ValidateStatelessRows(
        self,
        rowIndex: torch.Tensor,
        fullBatchSize: int,
        device: torch.device,
    ) -> None:
        if (
            not torch.is_tensor(rowIndex)
            or rowIndex.dim() != 1
            or rowIndex.dtype != torch.long
            or rowIndex.device != device
        ):
            raise ValueError(
                "Decision rowIndex must be a one-dimensional long tensor on the input device")
        if bool(((rowIndex < 0) | (rowIndex >= fullBatchSize)).any().item()):
            raise IndexError("Decision rowIndex contains an out-of-range row")
        if int(torch.unique(rowIndex).numel()) != int(rowIndex.numel()):
            raise ValueError("Decision rowIndex must not contain duplicate rows")

    def ValidateStatelessFeature(
        self,
        value: torch.Tensor,
        rowCount: int,
        featureDim: int,
        fieldName: str,
    ) -> None:
        if (
            not torch.is_tensor(value)
            or value.dim() != 2
            or tuple(value.shape) != (int(rowCount), int(featureDim))
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(
                f"Decision {fieldName} must be finite with shape "
                f"({int(rowCount)}, {int(featureDim)})")

    def PrepareStatelessRows(
        self,
        rowIndex: torch.Tensor,
        cachedDecisionFeature: torch.Tensor,
        fullInputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        state_feat = fullInputs["stateFeat"]
        if not torch.is_tensor(state_feat) or state_feat.dim() != 2:
            raise ValueError("Decision stateFeat must have two dimensions")
        full_batch_size = int(state_feat.size(0))
        if full_batch_size < 1:
            raise ValueError("Decision full batch must contain at least one row")
        self.ValidateStatelessRows(
            rowIndex,
            full_batch_size,
            state_feat.device)
        if (
            not torch.is_tensor(cachedDecisionFeature)
            or cachedDecisionFeature.dim() != 2
            or tuple(cachedDecisionFeature.shape) != (
                full_batch_size,
                self.belief_dim)
            or cachedDecisionFeature.device != state_feat.device
            or not cachedDecisionFeature.is_floating_point()
            or not bool(torch.isfinite(cachedDecisionFeature).all().item())
        ):
            raise ValueError(
                "Decision cachedDecisionFeature must be a finite full-batch belief feature")
        selected = {
            name: self.SelectFullBatchRows(
                value,
                rowIndex,
                full_batch_size,
                name)
            for name, value in fullInputs.items()}
        row_count = int(rowIndex.numel())
        selected["valueTensor"] = selected["valueTensor"].reshape(
            row_count,
            self.value_tensor_dim)
        selected["vNextTensor"] = selected["vNextTensor"].reshape(
            row_count,
            self.v_next_tensor_dim)
        expected_features = (
            ("stateFeat", self.stateDim),
            ("intentFeat", self.intentDim),
            ("valueTensor", self.value_tensor_dim),
            ("vNextTensor", self.v_next_tensor_dim),
            ("worldHzx", self.world_hzx_dim),
            ("prevDecisionState", self.dyn_dim),
            ("prevLatentControl", self.u_dim),
            ("prevActionEmbed", self.action_embed_dim),
            ("prevMapperHidden", self.mapper_hidden_dim),
            ("prevOptionLogit", self.num_options))
        for field_name, feature_dim in expected_features:
            self.ValidateStatelessFeature(
                selected[field_name],
                row_count,
                feature_dim,
                field_name)
        for field_name in (
            "uncertainty",
            "confidence",
            "precision",
            "risk",
            "feedbackTdError",
        ):
            value = selected[field_name]
            if (
                not torch.is_tensor(value)
                or tuple(value.shape) != (row_count,)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(
                    f"Decision {field_name} must be a finite row scalar")
        cached = cachedDecisionFeature.index_select(0, rowIndex)
        return selected, cached

    def BuildStatelessOutput(
        self,
        selected: Dict[str, torch.Tensor],
        cachedDecisionFeature: torch.Tensor,
        residuals: Tuple[torch.Tensor, ...],
        sample: bool,
        deterministic: bool,
    ) -> Dict[str, Any]:
        (
            belief_residual,
            state_residual,
            latent_residual,
            logvar_residual,
            option_residual,
            mapper_residual,
            option_context_residual,
        ) = residuals
        belief = cachedDecisionFeature + torch.tanh(belief_residual)
        decision_state = (
            selected["prevDecisionState"]
            + torch.tanh(state_residual))
        latent_control = (
            selected["prevLatentControl"]
            + torch.tanh(latent_residual))
        latent_logvar = (-1.0 + logvar_residual).clamp(-8.0, 4.0)
        mapper_hidden = (
            selected["prevMapperHidden"]
            + torch.tanh(mapper_residual))
        option_prior_logits = selected["prevOptionLogit"].detach()
        option_logits = option_prior_logits + option_residual
        option_log_probs = F.log_softmax(option_logits, dim=-1)
        option_weights = option_log_probs.exp()
        row_count = int(belief.size(0))
        if row_count == 0:
            option_index = torch.empty(
                0,
                dtype=torch.long,
                device=belief.device)
        elif sample and not deterministic:
            option_index = torch.distributions.Categorical(
                probs=option_weights).sample()
        else:
            option_index = option_logits.argmax(dim=-1)
        option_psi = belief.new_zeros(
            row_count,
            self.num_options,
            self.psi_dim)
        selected_psi = belief.new_zeros(row_count, self.psi_dim)
        option_context = torch.tanh(option_context_residual)
        option_log_probability = (
            option_log_probs.gather(
                1,
                option_index.view(-1, 1)).squeeze(1)
            if row_count > 0
            else belief.new_empty(0))
        gaussian_entropy = (
            0.5 * (1.0 + math.log(2.0 * math.pi))
            + 0.5 * latent_logvar).sum(dim=-1)
        option_entropy = -(
            option_weights * option_log_probs).sum(dim=-1)
        policy_input = torch.cat([
            decision_state,
            latent_control,
            mapper_hidden], dim=-1)
        return {
            "z": decision_state,
            "entropy": gaussian_entropy + option_entropy,
            "option": {
                "logits": option_logits,
                "psi_all": option_psi,
                "w_t": option_weights,
                "psi_selected": selected_psi,
                "option_context": option_context,
                "opt_idx": option_index,
                "logp_option": option_log_probability,
                "policy_input": policy_input,
                "prior_logits": option_prior_logits,},
            "prevOptionLogit_next": option_logits.detach(),
            "belief": belief,
            "decision_state": decision_state,
            "decision_state_next": decision_state.detach(),
            "decision_uncertainty": selected["uncertainty"],
            "latent_control": {
                "u": latent_control,
                "mu": latent_control,
                "logvar": latent_logvar,},
            "latent_control_next": latent_control.detach(),
            "mapper": {
                "hidden": mapper_hidden,
                "hidden_next": mapper_hidden.detach(),},
            "eligibility": {
                "pre": selected["prevLatentControl"].detach(),
                "post": torch.zeros_like(selected["prevLatentControl"]),},}

    def ComputeStatelessResiduals(
        self,
        selected: Dict[str, torch.Tensor],
        cachedDecisionFeature: torch.Tensor,
        detailContext: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        scalar_source = torch.stack([
            selected["uncertainty"],
            selected["confidence"],
            selected["precision"],
            selected["risk"],
            selected["feedbackTdError"],
        ], dim=-1)
        residuals = self.fast_policy((
            selected["stateFeat"],
            selected["intentFeat"],
            selected["valueTensor"],
            selected["vNextTensor"],
            selected["worldHzx"],
            selected["prevDecisionState"],
            selected["prevLatentControl"],
            selected["prevActionEmbed"],
            selected["prevMapperHidden"],
            cachedDecisionFeature,
            scalar_source))
        if detailContext is None:
            return residuals
        fast_belief = cachedDecisionFeature + torch.tanh(residuals[0])
        detail_residuals = self.detail_policy(
            fast_belief,
            detailContext)
        return tuple(
            fast_residual + detail_residual
            for fast_residual, detail_residual in zip(
                residuals,
                detail_residuals))

    def ForwardFastRows(
        self,
        rowIndex: torch.Tensor,
        cachedDecisionFeature: torch.Tensor,
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
        sample: bool = False,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        full_inputs = {
            "stateFeat": stateFeat,
            "intentFeat": intentFeat,
            "valueTensor": valueTensor,
            "vNextTensor": vNextTensor,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "precision": precision,
            "risk": risk,
            "worldHzx": worldHzx,
            "prevDecisionState": prevDecisionState,
            "prevLatentControl": prevLatentControl,
            "prevActionEmbed": prevActionEmbed,
            "prevMapperHidden": prevMapperHidden,
            "feedbackTdError": feedbackTdError,
            "prevOptionLogit": prevOptionLogit}
        selected, cached = self.PrepareStatelessRows(
            rowIndex,
            cachedDecisionFeature,
            full_inputs)
        residuals = self.ComputeStatelessResiduals(selected, cached)
        return self.BuildStatelessOutput(
            selected,
            cached,
            residuals,
            sample,
            deterministic)

    def ForwardDetailRows(
        self,
        rowIndex: torch.Tensor,
        cachedDecisionFeature: torch.Tensor,
        detailContext: torch.Tensor,
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
        sample: bool = False,
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        full_inputs = {
            "stateFeat": stateFeat,
            "intentFeat": intentFeat,
            "valueTensor": valueTensor,
            "vNextTensor": vNextTensor,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "precision": precision,
            "risk": risk,
            "worldHzx": worldHzx,
            "prevDecisionState": prevDecisionState,
            "prevLatentControl": prevLatentControl,
            "prevActionEmbed": prevActionEmbed,
            "prevMapperHidden": prevMapperHidden,
            "feedbackTdError": feedbackTdError,
            "prevOptionLogit": prevOptionLogit}
        selected, cached = self.PrepareStatelessRows(
            rowIndex,
            cachedDecisionFeature,
            full_inputs)
        full_batch_size = int(stateFeat.size(0))
        selected_detail = self.SelectFullBatchRows(
            detailContext,
            rowIndex,
            full_batch_size,
            "detailContext")
        self.ValidateStatelessFeature(
            selected_detail,
            int(rowIndex.numel()),
            self.goal_decision_context_dim,
            "detailContext")
        residuals = self.ComputeStatelessResiduals(
            selected,
            cached,
            selected_detail)
        return self.BuildStatelessOutput(
            selected,
            cached,
            residuals,
            sample,
            deterministic)

    def FastDistillationLoss(
        self,
        fastActOut: Dict[str, Any],
        fullActOut: Dict[str, Any],
        stableMask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        fast_refined = "decision_feature" in fastActOut
        full_refined = "decision_feature" in fullActOut
        if fast_refined != full_refined:
            raise ValueError(
                "Decision distillation representations must be at the same refinement stage")
        fast_feature = (
            fastActOut["decision_feature"]
            if fast_refined
            else fastActOut["belief"])
        teacher_feature = (
            fullActOut["decision_feature"]
            if full_refined
            else fullActOut["belief"]).detach()
        if (
            not torch.is_tensor(stableMask)
            or stableMask.dim() != 1
            or stableMask.dtype != torch.bool
            or int(stableMask.numel()) != int(fast_feature.size(0))
            or stableMask.device != fast_feature.device
        ):
            raise ValueError(
                "Decision stableMask must be a full-batch boolean tensor")
        if tuple(fast_feature.shape) != tuple(teacher_feature.shape):
            raise ValueError(
                "Decision fast and full feature shapes do not match")
        if not bool(stableMask.any().item()):
            zero = fast_feature.sum() * 0.0
            return {
                "total": zero,
                "feature": zero,
                "dynamics": zero,
                "latent": zero,
                "option": zero,
                "context": zero,}
        fast_feature = fast_feature[stableMask]
        teacher_feature = teacher_feature[stableMask]
        feature_loss = F.l1_loss(fast_feature, teacher_feature)
        dynamics_loss = F.smooth_l1_loss(
            fastActOut["decision_state"][stableMask],
            fullActOut["decision_state"].detach()[stableMask])
        latent_loss = F.smooth_l1_loss(
            fastActOut["latent_control"]["u"][stableMask],
            fullActOut["latent_control"]["u"].detach()[stableMask])
        fast_log_prob = F.log_softmax(
            fastActOut["option"]["logits"][stableMask],
            dim=-1)
        teacher_log_prob = F.log_softmax(
            fullActOut["option"]["logits"].detach()[stableMask],
            dim=-1)
        option_loss = (
            fast_log_prob.exp()
            * (fast_log_prob - teacher_log_prob)).sum(dim=-1).mean()
        context_loss = F.smooth_l1_loss(
            fastActOut["option"]["option_context"][stableMask],
            fullActOut["option"]["option_context"].detach()[stableMask])
        total = (
            feature_loss
            + 0.25 * dynamics_loss
            + 0.25 * latent_loss
            + 0.1 * option_loss
            + 0.1 * context_loss)
        return {
            "total": total,
            "feature": feature_loss,
            "dynamics": dynamics_loss,
            "latent": latent_loss,
            "option": option_loss,
            "context": context_loss,}

    def BuildBeliefContextRows(
        self,
        rowIndex: torch.Tensor,
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
    ) -> torch.Tensor:
        if not torch.is_tensor(stateFeat) or stateFeat.dim() != 2:
            raise ValueError("Decision stateFeat must have two dimensions")
        full_batch_size = int(stateFeat.size(0))
        self.ValidateStatelessRows(
            rowIndex,
            full_batch_size,
            stateFeat.device)
        selected_state = self.SelectFullBatchRows(
            stateFeat,
            rowIndex,
            full_batch_size,
            "stateFeat")
        selected_intent = self.SelectFullBatchRows(
            intentFeat,
            rowIndex,
            full_batch_size,
            "intentFeat")
        selected_value = self.SelectFullBatchRows(
            valueTensor,
            rowIndex,
            full_batch_size,
            "valueTensor").reshape(-1, self.value_tensor_dim)
        selected_next_value = self.SelectFullBatchRows(
            vNextTensor,
            rowIndex,
            full_batch_size,
            "vNextTensor").reshape(-1, self.v_next_tensor_dim)
        return self.belief_assembler(
            memFeat=selected_state,
            intentFeat=selected_intent,
            valueTensor=selected_value,
            vNextTensor=selected_next_value,
            uncertainty=self.SelectFullBatchRows(
                uncertainty,
                rowIndex,
                full_batch_size,
                "uncertainty"),
            confidence=self.SelectFullBatchRows(
                confidence,
                rowIndex,
                full_batch_size,
                "confidence"),
            precision=self.SelectFullBatchRows(
                precision,
                rowIndex,
                full_batch_size,
                "precision"),
            risk=self.SelectFullBatchRows(
                risk,
                rowIndex,
                full_batch_size,
                "risk"),
            worldHzx=self.SelectFullBatchRows(
                worldHzx,
                rowIndex,
                full_batch_size,
                "worldHzx"))

    def ForwardContractRows(
        self,
        rowIndex: torch.Tensor,
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
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        if not torch.is_tensor(stateFeat) or stateFeat.dim() < 1:
            raise ValueError("Decision stateFeat must have a batch dimension")
        fullBatchSize = int(stateFeat.size(0))
        if fullBatchSize < 1:
            raise ValueError("Decision full batch must contain at least one row")
        if (
            not torch.is_tensor(rowIndex)
            or rowIndex.dim() != 1
            or rowIndex.dtype != torch.long
            or rowIndex.device != stateFeat.device
        ):
            raise ValueError(
                "Decision rowIndex must be a one-dimensional long tensor on the input device")
        if rowIndex.numel() < 1:
            raise ValueError("Decision rowIndex must select at least one row")
        if bool(((rowIndex < 0) | (rowIndex >= fullBatchSize)).any().item()):
            raise IndexError("Decision rowIndex contains an out-of-range row")
        if int(torch.unique(rowIndex).numel()) != int(rowIndex.numel()):
            raise ValueError("Decision rowIndex must not contain duplicate rows")
        fullInputs = {
            "stateFeat": stateFeat,
            "intentFeat": intentFeat,
            "valueTensor": valueTensor,
            "vNextTensor": vNextTensor,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "precision": precision,
            "risk": risk,
            "worldHzx": worldHzx,
            "prevDecisionState": prevDecisionState,
            "prevLatentControl": prevLatentControl,
            "prevActionEmbed": prevActionEmbed,
            "prevMapperHidden": prevMapperHidden,
            "feedbackTdError": feedbackTdError,
            "prevOptionLogit": prevOptionLogit}
        selectedInputs = {
            name: self.SelectFullBatchRows(
                value,
                rowIndex,
                fullBatchSize,
                name)
            for name, value in fullInputs.items()}
        fullRows = torch.arange(
            fullBatchSize,
            dtype=torch.long,
            device=rowIndex.device)
        if torch.equal(rowIndex, fullRows):
            return self(
                **fullInputs,
                sample=sample,
                deterministic=deterministic)

        originalState = self.ExportEligibilityState()
        try:
            self.EnsureB(fullBatchSize)
            fullState = self.ExportEligibilityState()
            selectedState = {
                name: value.index_select(0, rowIndex)
                for name, value in fullState.items()}
            self.ImportEligibilityState(
                selectedState,
                int(rowIndex.numel()))
            output = self(
                **selectedInputs,
                sample=sample,
                deterministic=deterministic)
            updatedState = self.ExportEligibilityState()
            mergedState = {
                name: fullState[name].index_copy(
                    0,
                    rowIndex,
                    updatedState[name])
                for name in fullState}
            self.ImportEligibilityState(mergedState, fullBatchSize)
        except BaseException:
            self.elig_plasticity.trace = originalState["trace"]
            self.elig_plasticity.fast = originalState["fast"]
            raise
        return output

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
        self.EnsureB(B)
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
        option_prior_logits = prevOptionLogit.detach()
        option_logits = self.OptionLogits(option_in, option_prior_logits)
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
                "prior_logits": option_prior_logits,},
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
        worldHzx: torch.Tensor,
        sceneStateFeature: torch.Tensor,
        goalDecisionContext: torch.Tensor,
        temporalGoal: Dict[str, torch.Tensor],
        bodyStateFeature: torch.Tensor,
        controlFeedbackFeature: torch.Tensor,
        embodimentContextFeature: torch.Tensor,) -> Dict[str, Any]:
        expected_features = (
            ("world", worldHzx, self.world_hzx_dim),
            ("scene state", sceneStateFeature, self.scene_state_feature_dim),
            (
                "goal-decision context",
                goalDecisionContext,
                self.goal_decision_context_dim),
            ("body state", bodyStateFeature, self.body_state_feature_dim),
            (
                "control feedback",
                controlFeedbackFeature,
                self.control_feedback_feature_dim),
            (
                "embodiment context",
                embodimentContextFeature,
                self.embodiment_context_feature_dim),)
        batch_size = int(baseActOut["belief"].size(0))
        for feature_name, feature, feature_dim in expected_features:
            if (
                not torch.is_tensor(feature)
                or feature.dim() != 2
                or tuple(feature.shape) != (batch_size, feature_dim)
                or not feature.is_floating_point()
                or not bool(torch.isfinite(feature).all().item())
            ):
                raise ValueError(
                    f"Decision {feature_name} feature must be finite with "
                    f"shape ({batch_size}, {feature_dim})")
        nesy = self.nesy_conditioner(neuroSymbolic)
        latent_u = baseActOut["latent_control"]["u"]
        embodied_state_feature = self.embodied_state_encoder(torch.cat([
            bodyStateFeature,
            controlFeedbackFeature,
            embodimentContextFeature], dim=-1))
        refine_in = torch.cat([
            baseActOut["belief"],
            baseActOut["decision_state"],
            latent_u,
            worldHzx,
            sceneStateFeature,
            goalDecisionContext,
            nesy["context"],
            embodied_state_feature,
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
            embodied_state_feature,
        ], dim=-1))
        B = decision_feature.size(0)
        constraint_seed = self.decoder_constraint_seed(decision_feature).view(
            B,
            self.constraint_token_count,
            self.constraint_token_dim)
        decoder_constraint_tokens = self.decoder_constraint_head(torch.cat([
            nesy["constraint_tokens"],
            constraint_seed,
        ], dim=-1))
        decision_energy = self.decision_energy_head(torch.cat([
            decision_feature,
            worldHzx,
            embodied_state_feature,
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
        baseActOut["embodied_state_feature"] = embodied_state_feature
        baseActOut["neuro_symbolic_condition"] = {
            "context": nesy["context"],
            "gate": gate,
            "symbolic_energy": nesy["energy"],}
        baseActOut["decision_energy"] = decision_energy
        baseActOut["temporal_decision"] = temporal_decision
        return baseActOut

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
        for m in self.modules():
            if isinstance(m, EligibilityTracePlasticityLayer):
                m.Reset(doneMask=doneMask)




class CEMPlanner(AGICoreModule):
    def __init__(
        self,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        N: int = 64,
        elite: int = 8,
        iters: int = 3,
        temperature: float = 1.0,
        momentum: float = 0.15,
        minVar: float = 1e-4,
        candidateChunkSize: int = 8,
    ) -> None:
        super().__init__()
        if type(decisionDim) is not int or decisionDim < 1:
            raise ValueError("CEM decision dimension must be positive")
        if type(N) is not int or N < 1:
            raise ValueError("CEM population must be positive")
        if type(elite) is not int or elite < 1:
            raise ValueError("CEM elite count must be positive")
        if type(iters) is not int or iters < 1:
            raise ValueError("CEM iteration count must be positive")
        if not math.isfinite(float(temperature)):
            raise ValueError("CEM temperature must be finite")
        if not math.isfinite(float(momentum)) or not 0.0 <= momentum <= 1.0:
            raise ValueError("CEM momentum must be in [0, 1]")
        if not math.isfinite(float(minVar)) or minVar <= 0.0:
            raise ValueError("CEM minimum variance must be positive and finite")
        if type(candidateChunkSize) is not int or candidateChunkSize < 1:
            raise ValueError("CEM candidate chunk size must be positive")
        self.DecisionDim = int(decisionDim)
        self.Population = int(N)
        self.Elite = min(int(elite), int(N))
        self.Iterations = int(iters)
        self.Temperature = float(temperature)
        self.Momentum = float(momentum)
        self.MinimumVariance = float(minVar)
        self.CandidateChunkSize = int(candidateChunkSize)

    def ValidateDecisionFeature(self, decisionFeature: torch.Tensor) -> None:
        if (
            not torch.is_tensor(decisionFeature)
            or decisionFeature.dim() != 2
            or int(decisionFeature.size(-1)) != self.DecisionDim
            or not decisionFeature.is_floating_point()
            or not bool(torch.isfinite(decisionFeature).all().item())
        ):
            raise ValueError(
                "CEM input must be a finite batched decision feature")

    def ScoreFeatureCandidates(
        self,
        featureCandidates: torch.Tensor,
        candidateEvaluator: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        scores, _ = self.EvaluateFeatureCandidates(
            featureCandidates,
            candidateEvaluator)
        return scores

    def NormalizeCandidateEvaluation(
        self,
        evaluation: Any,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(evaluation):
            scores = evaluation
            valid = torch.ones(
                reference.size(0),
                device=reference.device,
                dtype=torch.bool)
        elif (
            isinstance(evaluation, tuple)
            and len(evaluation) == 2
        ):
            scores, valid = evaluation
        else:
            raise TypeError(
                "candidate evaluator must return scores or scores with validity")
        if (
            not torch.is_tensor(scores)
            or tuple(scores.shape) != (reference.size(0),)
            or not scores.is_floating_point()
            or not bool(torch.isfinite(scores).all().item())
            or scores.device != reference.device
            or scores.dtype != reference.dtype
        ):
            raise ValueError(
                "candidate evaluator must return one finite score per feature")
        if (
            not torch.is_tensor(valid)
            or tuple(valid.shape) != (reference.size(0),)
            or valid.dtype != torch.bool
            or valid.device != reference.device
        ):
            raise ValueError(
                "candidate evaluator validity must match the feature batch")
        return scores, valid

    def EvaluateFeatureCandidates(
        self,
        featureCandidates: torch.Tensor,
        candidateEvaluator: Callable[[torch.Tensor], Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            not torch.is_tensor(featureCandidates)
            or featureCandidates.dim() != 3
            or int(featureCandidates.size(-1)) != self.DecisionDim
            or not featureCandidates.is_floating_point()
            or not bool(torch.isfinite(featureCandidates).all().item())
        ):
            raise ValueError("CEM candidates must be decision features")
        if not callable(candidateEvaluator):
            raise TypeError("CEM candidate evaluator must be callable")
        batch_size = int(featureCandidates.size(0))
        candidate_count = int(featureCandidates.size(1))
        if batch_size < 1 or candidate_count < 1:
            raise ValueError("CEM candidate population cannot be empty")
        score_chunks = []
        valid_chunks = []
        for start in range(0, candidate_count, self.CandidateChunkSize):
            end = min(start + self.CandidateChunkSize, candidate_count)
            chunk_size = end - start
            candidate_feature = featureCandidates[:, start:end].reshape(
                batch_size * chunk_size,
                self.DecisionDim)
            score, valid = self.NormalizeCandidateEvaluation(
                candidateEvaluator(candidate_feature),
                candidate_feature)
            score_chunks.append(score.reshape(batch_size, chunk_size))
            valid_chunks.append(valid.reshape(batch_size, chunk_size))
        return (
            torch.cat(score_chunks, dim=1),
            torch.cat(valid_chunks, dim=1))

    @torch.no_grad()
    def Plan(
        self,
        decisionFeature: torch.Tensor,
        candidateEvaluator: Callable[[torch.Tensor], Any],
        returnDiagnostics: bool = False,
    ) -> Dict[str, Any]:
        self.ValidateDecisionFeature(decisionFeature)
        batch_size = int(decisionFeature.size(0))
        mu = decisionFeature
        std = torch.ones_like(mu)
        batch_index = torch.arange(
            batch_size,
            device=decisionFeature.device).unsqueeze(1).expand(
                batch_size,
                self.Elite)
        weights = decisionFeature.new_full(
            (batch_size, self.Elite),
            1.0 / float(self.Elite))
        elite_scores = decisionFeature.new_zeros(
            batch_size,
            self.Elite)
        elite_valid = torch.zeros(
            batch_size,
            self.Elite,
            device=decisionFeature.device,
            dtype=torch.bool)
        best_feature = decisionFeature.clone()
        best_score = decisionFeature.new_full(
            (batch_size,),
            -torch.inf)
        best_valid = torch.zeros(
            batch_size,
            device=decisionFeature.device,
            dtype=torch.bool)
        for _ in range(self.Iterations):
            noise = torch.randn_like(std.unsqueeze(1).expand(
                batch_size,
                self.Population,
                self.DecisionDim))
            candidates = mu.unsqueeze(1) + noise * std.unsqueeze(1)
            score, valid = self.EvaluateFeatureCandidates(
                candidates,
                candidateEvaluator)
            ranked_score = score.masked_fill(~valid, -torch.inf)
            iteration_score, iteration_index = ranked_score.max(dim=1)
            iteration_valid = valid.any(dim=1)
            iteration_feature = candidates[
                torch.arange(
                    batch_size,
                    device=decisionFeature.device),
                iteration_index]
            better = (
                iteration_valid
                & (~best_valid | iteration_score.gt(best_score)))
            best_feature = torch.where(
                better.unsqueeze(-1),
                iteration_feature,
                best_feature)
            best_score = torch.where(
                better,
                iteration_score,
                best_score)
            best_valid = best_valid | iteration_valid
            top_index = torch.topk(
                ranked_score,
                k=self.Elite,
                dim=1).indices
            elite_scores = score.gather(1, top_index)
            elite_valid = valid.gather(1, top_index)
            has_valid_elite = elite_valid.any(dim=1)
            if self.Temperature <= 0.0:
                weights = elite_valid.to(dtype=elite_scores.dtype)
                weights = weights / weights.sum(
                    dim=1,
                    keepdim=True).clamp_min(1.0)
            else:
                logits = (elite_scores / self.Temperature).masked_fill(
                    ~elite_valid,
                    -torch.inf)
                logits = torch.where(
                    has_valid_elite.unsqueeze(-1),
                    logits,
                    torch.zeros_like(logits))
                weights = F.softmax(logits, dim=1)
                weights = weights * elite_valid.to(dtype=weights.dtype)
                weights = weights / weights.sum(
                    dim=1,
                    keepdim=True).clamp_min(1.0)
            elite_features = candidates[batch_index, top_index]
            weighted = weights.unsqueeze(-1)
            new_mu = (weighted * elite_features).sum(dim=1)
            new_variance = (
                weighted
                * (elite_features - new_mu.unsqueeze(1)).square()
            ).sum(dim=1).clamp_min(self.MinimumVariance)
            updated_mu = (
                self.Momentum * mu
                + (1.0 - self.Momentum) * new_mu)
            updated_std = (
                self.Momentum * std
                + (1.0 - self.Momentum) * new_variance.sqrt())
            mu = torch.where(
                has_valid_elite.unsqueeze(-1),
                updated_mu,
                mu)
            std = torch.where(
                has_valid_elite.unsqueeze(-1),
                updated_std,
                std)

        verification_score, verification_valid = (
            self.EvaluateFeatureCandidates(
                torch.stack([
                    best_feature,
                    decisionFeature,
                ], dim=1),
                candidateEvaluator))
        planner_valid = best_valid & verification_valid[:, 0]
        selected_feature = torch.where(
            planner_valid.unsqueeze(-1),
            best_feature,
            decisionFeature)
        final_score = torch.where(
            planner_valid,
            verification_score[:, 0],
            verification_score[:, 1])
        final_valid = torch.where(
            planner_valid,
            verification_valid[:, 0],
            verification_valid[:, 1])
        result: Dict[str, Any] = {
            "decision_feature": selected_feature,
            "expected_return": final_score,
            "valid": planner_valid,
            "fallback_used": ~planner_valid,
        }
        if returnDiagnostics:
            result["diagnostics"] = {
                "std": std,
                "elite_population_return": (
                    weights * elite_scores).sum(dim=1),
                "elite_valid": elite_valid,
                "selected_feature_valid": final_valid,
            }
        return result


class DecisionPlannerExtractor:
    def BuildPlanner(
        self,
        decisionDim: int = ModuleDim.DecisionBeliefDim,
        N: int = 64,
        elite: int = 8,
        iters: int = 3,
        temperature: float = 1.0,
        momentum: float = 0.15,
        minVar: float = 1e-4,
        candidateChunkSize: int = 8,
    ) -> CEMPlanner:
        return CEMPlanner(
            decisionDim=decisionDim,
            N=N,
            elite=elite,
            iters=iters,
            temperature=temperature,
            momentum=momentum,
            minVar=minVar,
            candidateChunkSize=candidateChunkSize)


class TestDecisionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")

    def MakeExtractor(self) -> DecisionExtractor:
        return DecisionExtractor(
            stateDim=12,
            optionNum=5,
            hiddenDim=32,
            psiDim=16,
            intentDim=10,
            valueTensorDim=8,
            vNextTensorDim=8,
            worldHDim=9,
            worldZDim=4,
            worldXDim=3,
            beliefDim=16,
            decisionDynDim=7,
            latentControlDim=5,
            mapperEmbedDim=11,
            actionEmbedDim=6,
            planFeatureDim=8,
            subgoalFeatureDim=7,
            constraintTokenCount=3,
            constraintTokenDim=5,
            bodyStateFeatureDim=6,
            controlFeedbackFeatureDim=4,
            embodimentContextFeatureDim=2,
            embodimentStateFeatureDim=8,
            sceneStateFeatureDim=9,
            goalDecisionContextDim=10).to(self.device)

    def MakeBaseInputs(
        self,
        model: DecisionExtractor,
        batchSize: int = 2,
    ) -> Dict[str, Any]:
        batch_size = int(batchSize)
        return {
            "stateFeat": torch.randn(
                batch_size,
                model.stateDim,
                device=self.device,
                requires_grad=True),
            "intentFeat": torch.randn(
                batch_size,
                model.intentDim,
                device=self.device,
                requires_grad=True),
            "valueTensor": torch.randn(
                batch_size,
                model.value_tensor_dim,
                device=self.device),
            "vNextTensor": torch.randn(
                batch_size,
                model.v_next_tensor_dim,
                device=self.device),
            "uncertainty": torch.full(
                (batch_size,),
                0.2,
                device=self.device),
            "confidence": torch.full(
                (batch_size,),
                0.8,
                device=self.device),
            "precision": torch.full(
                (batch_size,),
                0.7,
                device=self.device),
            "risk": torch.full(
                (batch_size,),
                0.1,
                device=self.device),
            "worldHzx": torch.randn(
                batch_size,
                model.world_hzx_dim,
                device=self.device),
            "prevDecisionState": torch.zeros(
                batch_size,
                model.dyn_dim,
                device=self.device),
            "prevLatentControl": torch.zeros(
                batch_size,
                model.u_dim,
                device=self.device),
            "prevActionEmbed": torch.zeros(
                batch_size,
                model.action_embed_dim,
                device=self.device),
            "prevMapperHidden": torch.zeros(
                batch_size,
                model.mapper_hidden_dim,
                device=self.device),
            "feedbackTdError": torch.ones(
                batch_size,
                device=self.device),
            "prevOptionLogit": torch.zeros(
                batch_size,
                model.num_options,
                device=self.device),
            "sample": False,
            "deterministic": True,
        }

    def MakeNeuroSymbolic(
        self,
        model: DecisionExtractor,
        batchSize: int,
    ) -> NeuroSymbolicOutput:
        batch_size = int(batchSize)
        temporal_count = int(ModuleDim.TemporalPrimitiveCount)
        failure_count = len(FAILURE_CAUSES)
        return NeuroSymbolicOutput(
            facts=[],
            operator_logits=torch.randn(
                batch_size,
                len(OPERATORS),
                device=self.device,
                requires_grad=True),
            plan_steps=[],
            operator_rationales=[],
            plan_latent=torch.randn(
                batch_size,
                model.plan_feature_dim,
                device=self.device,
                requires_grad=True),
            subgoal_feature=torch.randn(
                batch_size,
                model.subgoal_feature_dim,
                device=self.device,
                requires_grad=True),
            constraint_tokens=torch.randn(
                batch_size,
                model.constraint_token_count,
                model.constraint_token_dim,
                device=self.device,
                requires_grad=True),
            risk_cause_logits=torch.randn(
                batch_size,
                failure_count,
                device=self.device,
                requires_grad=True),
            risk_cause_raw_logits=torch.randn(
                batch_size,
                failure_count,
                device=self.device),
            failure_cause_logits=torch.randn(
                batch_size,
                failure_count,
                device=self.device),
            failure_cause_raw_logits=torch.randn(
                batch_size,
                failure_count,
                device=self.device),
            failure_gate_logits=torch.randn(
                batch_size,
                device=self.device),
            failure_gate=torch.sigmoid(torch.randn(
                batch_size,
                device=self.device)),
            invoke_mask=torch.ones(
                batch_size,
                device=self.device),
            same_operator=torch.ones(
                batch_size,
                device=self.device),
            operator_changed=torch.zeros(
                batch_size,
                device=self.device),
            invoke_delta=torch.zeros(
                batch_size,
                device=self.device),
            reference_drift=torch.zeros(
                batch_size,
                device=self.device),
            temporal_logits=torch.randn(
                batch_size,
                temporal_count,
                device=self.device),
            temporal_reason_logits=torch.randn(
                batch_size,
                temporal_count,
                device=self.device),
            continue_guard_score=torch.zeros(
                batch_size,
                device=self.device),
            interrupt_guard_score=torch.zeros(
                batch_size,
                device=self.device),
            redispatch_guard_score=torch.zeros(
                batch_size,
                device=self.device))

    def TestDecisionExtractorRelationshipsAndGradient(self) -> bool:
        try:
            torch.manual_seed(31)
            model = self.MakeExtractor()
            inputs = self.MakeBaseInputs(model)
            output = model(**inputs)
            loss = (
                output["belief"].square().mean()
                + output["decision_state"].square().mean()
                + output["latent_control"]["u"].square().mean()
                + output["option"]["logits"].square().mean())
            loss.backward()
            state_gradient = inputs["stateFeat"].grad
            intent_gradient = inputs["intentFeat"].grad
            return bool(
                tuple(output["belief"].shape) == (2, model.belief_dim)
                and tuple(output["decision_state"].shape) == (
                    2,
                    model.dyn_dim)
                and tuple(output["latent_control"]["u"].shape) == (
                    2,
                    model.u_dim)
                and tuple(output["option"]["logits"].shape) == (
                    2,
                    model.num_options)
                and tuple(output["mapper"]["hidden"].shape) == (
                    2,
                    model.mapper_hidden_dim)
                and state_gradient is not None
                and intent_gradient is not None
                and bool(torch.isfinite(state_gradient).all().item())
                and bool(torch.isfinite(intent_gradient).all().item())
                and float(state_gradient.abs().sum().item()) > 0.0
                and float(intent_gradient.abs().sum().item()) > 0.0)
        except Exception:
            return False

    def TestNeuroSymbolicRefinementAndGradient(self) -> bool:
        try:
            torch.manual_seed(37)
            model = self.MakeExtractor()
            inputs = self.MakeBaseInputs(model)
            base_output = model(**inputs)
            batch_size = int(inputs["stateFeat"].size(0))
            symbolic = self.MakeNeuroSymbolic(model, batch_size)
            scene_feature = torch.randn(
                batch_size,
                model.scene_state_feature_dim,
                device=self.device,
                requires_grad=True)
            goal_context = torch.randn(
                batch_size,
                model.goal_decision_context_dim,
                device=self.device,
                requires_grad=True)
            body_feature = torch.randn(
                batch_size,
                model.body_state_feature_dim,
                device=self.device,
                requires_grad=True)
            feedback_feature = torch.randn(
                batch_size,
                model.control_feedback_feature_dim,
                device=self.device,
                requires_grad=True)
            context_feature = torch.randn(
                batch_size,
                model.embodiment_context_feature_dim,
                device=self.device,
                requires_grad=True)
            temporal_goal = {
                "goal_mode_logits": torch.randn(
                    batch_size,
                    ModuleDim.TemporalPrimitiveCount,
                    device=self.device),
                "goal_timeout_soft_ms": torch.full(
                    (batch_size,),
                    250.0,
                    device=self.device),
                "goal_timeout_hard_ms": torch.full(
                    (batch_size,),
                    500.0,
                    device=self.device),
            }
            refined = model.RefineWithNeuroSymbolic(
                baseActOut=base_output,
                neuroSymbolic=symbolic,
                worldHzx=inputs["worldHzx"],
                sceneStateFeature=scene_feature,
                goalDecisionContext=goal_context,
                temporalGoal=temporal_goal,
                bodyStateFeature=body_feature,
                controlFeedbackFeature=feedback_feature,
                embodimentContextFeature=context_feature)
            loss = (
                refined["decision_feature"].square().mean()
                + refined["decoder_plan_latent"].square().mean()
                + refined["decoder_subgoal_feature"].square().mean()
                + refined["decoder_constraint_tokens"].square().mean()
                + refined["decision_energy"].square().mean())
            loss.backward()
            return bool(
                tuple(refined["decision_feature"].shape) == (
                    batch_size,
                    model.belief_dim)
                and tuple(refined["decoder_plan_latent"].shape) == (
                    batch_size,
                    model.plan_feature_dim)
                and tuple(refined["decoder_subgoal_feature"].shape) == (
                    batch_size,
                    model.subgoal_feature_dim)
                and tuple(refined["decoder_constraint_tokens"].shape) == (
                    batch_size,
                    model.constraint_token_count,
                    model.constraint_token_dim)
                and symbolic.plan_latent.grad is not None
                and body_feature.grad is not None
                and feedback_feature.grad is not None
                and context_feature.grad is not None
                and float(symbolic.plan_latent.grad.abs().sum().item()) > 0.0
                and float(body_feature.grad.abs().sum().item()) > 0.0
                and float(feedback_feature.grad.abs().sum().item()) > 0.0
                and float(context_feature.grad.abs().sum().item()) > 0.0)
        except Exception:
            return False

    def TestOptionAndEligibilityState(self) -> bool:
        try:
            model = self.MakeExtractor()
            batch_size = 2
            model.EnsureB(batch_size)
            pre = torch.randn(batch_size, model.u_dim, device=self.device)
            post = torch.randn(batch_size, model.u_dim, device=self.device)
            model.CommitEligibility(
                pre,
                post,
                torch.tensor([True, False], device=self.device))
            first_written = float(
                model.elig_plasticity.trace[0].abs().sum().item()) > 0.0
            second_empty = float(
                model.elig_plasticity.trace[1].abs().sum().item()) == 0.0
            model.ClearInvalidEligibility(torch.tensor(
                [True, False],
                device=self.device))
            cleared = float(
                model.elig_plasticity.trace.abs().sum().item()) == 0.0
            prior = model.option_transition_prior(torch.zeros(
                batch_size,
                model.num_options,
                device=self.device))
            return bool(
                first_written
                and second_empty
                and cleared
                and tuple(prior.shape) == (batch_size, model.num_options)
                and bool(torch.isfinite(prior).all().item()))
        except Exception:
            return False

    def TestCEMFeatureSearch(self) -> bool:
        try:
            torch.manual_seed(41)
            planner = CEMPlanner(
                decisionDim=4,
                N=48,
                elite=8,
                iters=4,
                temperature=0.5,
                candidateChunkSize=5)
            observed_shapes: List[Tuple[int, ...]] = []

            def EvaluateFeatures(features: torch.Tensor) -> torch.Tensor:
                observed_shapes.append(tuple(features.shape))
                return -features.square().sum(dim=-1)

            initial_feature = torch.full(
                (2, 4),
                2.0,
                device=self.device)
            result = planner.Plan(
                initial_feature,
                EvaluateFeatures,
                returnDiagnostics=True)
            optimized = result["decision_feature"]
            return bool(
                observed_shapes
                and all(
                    len(shape) == 2 and shape[1] == 4
                    for shape in observed_shapes)
                and max(shape[0] for shape in observed_shapes) <= 10
                and tuple(optimized.shape) == (2, 4)
                and tuple(result["expected_return"].shape) == (2,)
                and tuple(result["diagnostics"]["std"].shape) == (2, 4)
                and float(optimized.square().mean().item())
                < float(initial_feature.square().mean().item()))
        except Exception:
            return False

    def TestCEMFeatureValidation(self) -> bool:
        try:
            planner = CEMPlanner(
                decisionDim=3,
                N=4,
                elite=2,
                iters=1)
            invalid_input_rejected = False
            invalid_score_rejected = False
            try:
                planner.Plan(
                    torch.zeros(3),
                    lambda value: value.sum(dim=-1))
            except ValueError:
                invalid_input_rejected = True
            try:
                planner.Plan(
                    torch.zeros(2, 3),
                    lambda value: value)
            except ValueError:
                invalid_score_rejected = True
            return bool(invalid_input_rejected and invalid_score_rejected)
        except Exception:
            return False

    def RunAllTests(self) -> Dict[str, bool]:
        return {
            "DecisionExtractorRelationshipsAndGradient": (
                self.TestDecisionExtractorRelationshipsAndGradient()),
            "NeuroSymbolicRefinementAndGradient": (
                self.TestNeuroSymbolicRefinementAndGradient()),
            "OptionAndEligibilityState": self.TestOptionAndEligibilityState(),
            "CEMFeatureSearch": self.TestCEMFeatureSearch(),
            "CEMFeatureValidation": self.TestCEMFeatureValidation(),
        }
