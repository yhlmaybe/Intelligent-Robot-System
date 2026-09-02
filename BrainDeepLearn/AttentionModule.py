from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, List, Dict
from FunctionTools import GetParametersScale, SiteSpec, BaseOnlineWrapper, AGICoreModule, GrowableLoRALinear, RotaryEmbedding
from ModuleMessagerManager import ModuleDim




class SelectiveSSM(AGICoreModule):
    def __init__(
        self,
        embedDim: int,
        stateDim: int = 4,
        convKernel: int = 4,
        slowDtScale: float = 0.25,):
        super().__init__()
        E = embedDim
        N = stateDim
        self.E = E
        self.N = N
        self.slow_dt_scale = float(slowDtScale)

        self.A_log = nn.Parameter(torch.randn(E, N))

        self.D = nn.Parameter(torch.zeros(E))

        self.in_proj = nn.Linear(E, 2 * E)

        self.param_proj = nn.Linear(E, E + 2 * E * N)

        self.dt_bias = nn.Parameter(torch.full((E,), -3.0))

        self.dw_conv = nn.Conv1d(E, E, convKernel, groups=E, bias=True)

        self.out_norm = nn.LayerNorm(E)
        self.time_mix_gate = nn.Linear(E, E)

        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.param_proj.weight, gain=0.5)
        nn.init.zeros_(self.param_proj.bias)
        nn.init.zeros_(self.time_mix_gate.weight)
        nn.init.zeros_(self.time_mix_gate.bias)

    def CausalDwconv(self, u: torch.Tensor) -> torch.Tensor:
        x = u.transpose(1, 2)
        k = self.dw_conv.kernel_size[0]
        x = F.pad(x, (k - 1, 0))
        y = self.dw_conv(x)
        y = y.transpose(1, 2)
        return y

    def forward(
        self,
        x: torch.Tensor, # [B,S,E]
        tdError: torch.Tensor, # [B]
        uncertainty: torch.Tensor, # [B]
        keyPaddingMask: Optional[torch.Tensor] = None, # [B,S] bool，True=padding
        ) -> torch.Tensor:
        B, S, E = x.shape
        N = self.N

        surprise = tdError.abs()

        u_and_g = self.in_proj(x)
        u, g = torch.chunk(u_and_g, 2, dim=-1)
        u = F.silu(u)
        if keyPaddingMask is not None:
            u = u * (~keyPaddingMask).unsqueeze(-1)

        u = u + 0.5 * self.CausalDwconv(u)

        gate_bias = (0.5 * surprise - 0.5 * uncertainty)[:, None, None]
        g = torch.sigmoid(g + gate_bias)

        p = self.param_proj(u)
        dt_raw = p[..., :E]
        B_raw, C_raw = p[..., E:].split(E * N, dim=-1)
        B_t = torch.tanh(B_raw).view(B, S, E, N)
        C_t = torch.tanh(C_raw).view(B, S, E, N)
        dt = F.softplus(dt_raw + self.dt_bias.view(1, 1, E))
        dt = dt * (
            (1.0 + 0.50 * surprise[:, None, None])
            * (1.0 - 0.25 * uncertainty[:, None, None]))

        A_pos = F.softplus(self.A_log) + 1e-4

        y = torch.empty((B, S, E), device=x.device, dtype=x.dtype)
        fast_state = torch.zeros((B, E, N), device=x.device, dtype=x.dtype)
        slow_state = torch.zeros((B, E, N), device=x.device, dtype=x.dtype)

        has_mask = (keyPaddingMask is not None)

        for t in range(S):
            if has_mask:
                keep = (~keyPaddingMask[:, t]).view(B, 1, 1)
            else:
                keep = None

            u_t = u[:, t, :]
            dt_t = dt[:, t, :]

            decay = torch.exp(-dt_t.unsqueeze(-1) * A_pos.unsqueeze(0))
            inj = B_t[:, t, :, :] * u_t.unsqueeze(-1)

            new_fast = decay * fast_state + (1.0 - decay) * inj

            if keep is not None:
                fast_state = torch.where(keep, new_fast, fast_state)
            else:
                fast_state = new_fast

            slow_dt = dt_t * self.slow_dt_scale
            slow_decay = torch.exp(-slow_dt.unsqueeze(-1) * A_pos.unsqueeze(0))
            new_slow = slow_decay * slow_state + (1.0 - slow_decay) * inj
            if keep is not None:
                slow_state = torch.where(keep, new_slow, slow_state)
            else:
                slow_state = new_slow
            mix = torch.sigmoid(self.time_mix_gate(u_t)).unsqueeze(-1)
            state_for_read = mix * fast_state + (1.0 - mix) * slow_state

            out_t = (C_t[:, t, :, :] * state_for_read).sum(dim=-1) + self.D.unsqueeze(0) * u_t
            out_t = out_t * g[:, t, :]

            if keep is not None:
                out_t = out_t * keep.squeeze(-1)

            y[:, t, :] = out_t

        y = self.out_norm(y)
        if keyPaddingMask is not None:
            y = y * (~keyPaddingMask).unsqueeze(-1)
        return y

class MultiHeadAttention(AGICoreModule):
    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        attnDropout: float = 0.1,
        tdUncScale: float = 1.0,):
        super().__init__()
        assert embedDim % numHeads == 0, "AttentionModule embed_dim must be divisible by num_heads"
        self.embed_dim = embedDim
        self.num_heads = numHeads
        self.head_dim = embedDim // numHeads
        self.attn_dropout_p = attnDropout
        self.td_unc_scale = tdUncScale

        self.temp_w_td = nn.Parameter(torch.tensor(0.8))
        self.temp_w_unc = nn.Parameter(torch.tensor(0.5))
        self.attention_prior_gain = nn.Parameter(
            torch.zeros(numHeads))

        self.q_proj = nn.Linear(embedDim, embedDim)
        self.k_proj = nn.Linear(embedDim, embedDim)
        self.v_proj = nn.Linear(embedDim, embedDim)
        self.out_proj = nn.Linear(embedDim, embedDim)

        self.q_adapter = GrowableLoRALinear(self.q_proj)
        self.k_adapter = GrowableLoRALinear(self.k_proj)
        self.v_adapter = GrowableLoRALinear(self.v_proj)
        self.o_adapter = GrowableLoRALinear(self.out_proj)
        self.rope = RotaryEmbedding(self.head_dim)

        self.register_buffer(
            "U",
            torch.zeros(1, self.num_heads, self.head_dim, 8),
            persistent=False)
        self.register_buffer(
            "V",
            torch.zeros(1, self.num_heads, self.head_dim, 8),
            persistent=False)

        self.ResetParameters()

    @torch.no_grad()
    def EnsureB(self, B: int):
        if self.U.size(0) != B:
            self.U = self.U.new_zeros(
                B, self.num_heads, self.head_dim, 8)
            self.V = self.V.new_zeros(
                B, self.num_heads, self.head_dim, 8)

    def ModulateTau(self, tdError, uncertainty):
        surprise = tdError.abs()
        tau = torch.exp(
            0.25 * torch.tanh(self.temp_w_unc) * uncertainty
            - 0.25 * torch.tanh(self.temp_w_td) * surprise)
        return tau[:, None, None, None]


    def ResetParameters(self) -> None:
        for mod in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_normal_(mod.weight, gain=1 / math.sqrt(2))
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        self.ResetHebbianMemory()

    def ResetHebbianMemory(
        self,
        doneMask: Optional[torch.Tensor] = None,
        ) -> None:
        if doneMask is None:
            self.U.zero_()
            self.V.zero_()
            return
        mask = doneMask.view(-1)
        if mask.numel() != self.U.size(0):
            raise ValueError("Attention Hebbian reset mask must match its batch size")
        self.U[mask] = 0
        self.V[mask] = 0

    def ComputeNeuromodulation(self, tdError: torch.Tensor) -> torch.Tensor:
        neuromod = 1.0 + 0.25 * tdError.abs()
        return neuromod[:, None, None, None]

    def ComputeHebbMod(
        self,
        tdError: torch.Tensor,
        precision: torch.Tensor,) -> torch.Tensor:
        mod = (0.25 + 0.75 * tdError.abs()) * precision
        return mod[:, None, None, None]


    @torch.no_grad()
    def UpdateHebbianWeights(self, v: torch.Tensor, q: torch.Tensor, alpha_tensor: torch.Tensor, keep4: Optional[torch.Tensor] = None):
        v = v.detach()
        q = q.detach()

        if keep4 is not None:
            valid = keep4[:, 0, :, 0].bool()
            update_rows = valid.any(dim=-1)
            positions = torch.arange(
                v.size(2),
                device=v.device).unsqueeze(0)
            current_index = positions.masked_fill(
                ~valid,
                0).amax(dim=-1)
            gather_index = current_index.view(
                v.size(0),
                1,
                1,
                1).expand(-1, v.size(1), 1, v.size(3))
            v = v.gather(2, gather_index)
            q = q.gather(2, gather_index)
        else:
            update_rows = torch.ones(
                v.size(0),
                device=v.device,
                dtype=torch.bool)
            v = v[:, :, -1:]
            q = q[:, :, -1:]

        v = F.normalize(v, dim=-1, eps=1e-6)
        q = F.normalize(q, dim=-1, eps=1e-6)

        hebb = torch.einsum("bhse,bhsd->bhde", v, q)
        hebb = torch.tanh(hebb)

        def ClampFro(x: torch.Tensor, max_norm: float):
            n = torch.linalg.vector_norm(x, ord=2, dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            scale = torch.clamp(max_norm / n, max=1.0)
            return x * scale

        r = int(self.U.size(-1))
        M0 = torch.einsum("bhdr,bher->bhde", self.U, self.V)
        M1 = (1.0 - 0.01) * M0 + alpha_tensor * hebb
        U_s, S, Vh = torch.linalg.svd(M1, full_matrices=False)

        Sr = S[..., :r].clamp_min(0.0)
        sqrtSr = Sr.sqrt().unsqueeze(-2)

        Ur = U_s[..., :r] * sqrtSr
        Vr = Vh.transpose(-2, -1)[..., :r] * sqrtSr

        Ur = ClampFro(Ur, 1.5)
        Vr = ClampFro(Vr, 1.5)

        row_mask = update_rows.view(-1, 1, 1, 1)
        Ur = torch.where(row_mask, Ur, self.U)
        Vr = torch.where(row_mask, Vr, self.V)

        self.U.copy_(Ur)
        self.V.copy_(Vr)


    def Attend(
        self,
        q_lin: torch.Tensor,
        k_lin: torch.Tensor,
        v_lin: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        precision: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        headGate: Optional[torch.Tensor] = None,
        attentionBias: Optional[torch.Tensor] = None,) -> torch.Tensor:
        B, L, _ = q_lin.shape

        td = tdError
        unc = uncertainty
        td_eff = td * self.td_unc_scale
        unc_eff = unc * self.td_unc_scale

        neuromod = self.ComputeNeuromodulation(td_eff)
        hebb_mod = self.ComputeHebbMod(td_eff, precision)

        q = q_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_lin.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q_hebb = q

        q = q * neuromod
        if headGate is not None:
            q = q * headGate

        U = self.U.detach().clone()
        V = self.V.detach().clone()
        qU = torch.matmul(q_hebb, U)
        delt = torch.matmul(qU, V.transpose(-2, -1))

        delt = delt * hebb_mod
        v_fast = v + delt

        tau = self.ModulateTau(td_eff, unc_eff)
        q = q / tau

        q_attn = self.rope.Apply(q)
        k_attn = self.rope.Apply(k)

        d = q_attn.size(-1)
        scores = torch.matmul(q_attn, k_attn.transpose(-2, -1)) / math.sqrt(d)
        if attentionBias is not None:
            prior_gain = F.softplus(
                self.attention_prior_gain).view(1, self.num_heads, 1, 1)
            scores = scores + prior_gain * attentionBias[:, None, None, :]
        if keyPaddingMask is not None:
            mask = keyPaddingMask[:, None, None, :] # bool [B,1,1,L]
            scores = scores.masked_fill(mask, -torch.inf)

        weights = F.softmax(scores, dim=-1)
        if keyPaddingMask is not None:
            weights = weights.masked_fill(mask, 0.0)
        weights = F.dropout(weights, p=self.attn_dropout_p if self.training else 0.0, training=self.training)
        context = torch.matmul(weights, v_fast)
        out = context.transpose(1, 2).reshape(B, L, self.embed_dim)

        alpha = 0.01 * hebb_mod
        if keyPaddingMask is not None:
            keep4 = (~keyPaddingMask).view(B, 1, L, 1)
            self.UpdateHebbianWeights(v, q_hebb, alpha, keep4=keep4)
        else:
            self.UpdateHebbianWeights(v, q_hebb, alpha, keep4=None)

        return out

    def forward(
        self,
        query,
        key,
        value,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        precision: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        headGate: Optional[torch.Tensor] = None,
        attentionBias: Optional[torch.Tensor] = None,):
        q_lin = self.q_adapter(query)
        k_lin = self.k_adapter(key)
        v_lin = self.v_adapter(value)
        out = self.Attend(
            q_lin,
            k_lin,
            v_lin,
            tdError,
            uncertainty,
            precision,
            keyPaddingMask,
            headGate,
            attentionBias)
        return self.o_adapter(out)


class TemporalAttention(AGICoreModule):
    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        layerIdx: int = 0,
        slowDtScale: float = 0.25,):
        super().__init__()

        td_unc_scale = 1.0 / (layerIdx + 1)
        self.mhsa = MultiHeadAttention(
            embedDim,
            numHeads,
            tdUncScale=td_unc_scale)
        self.ssm = SelectiveSSM(
            embedDim,
            stateDim=16,
            convKernel=4,
            slowDtScale=slowDtScale)

        self.gamma = nn.Parameter(1e-1 * torch.ones(embedDim), requires_grad=True)

        self.mix_gate = nn.Sequential(
            nn.Linear(embedDim, embedDim),
            nn.SiLU(),
            nn.Linear(embedDim, 1))

        nn.init.constant_(self.mix_gate[-1].bias, -1.0)

        self.ffn = nn.Sequential(
            nn.Linear(embedDim, 4 * embedDim),
            nn.GELU(),
            nn.Linear(4 * embedDim, embedDim))

        self.gamma_ffn = nn.Parameter(1e-1 * torch.ones(embedDim), requires_grad=True)

        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(embedDim)
        self.norm_ffn = nn.LayerNorm(embedDim)

    def forward(
        self,
        x,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        precision: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor]=None,
        headGate: Optional[torch.Tensor] = None,
        channelGate: Optional[torch.Tensor] = None,
        attentionBias: Optional[torch.Tensor] = None,):
        residual = x
        x_norm = self.norm(x)

        mhsa_out = self.mhsa(
            x_norm,
            x_norm,
            x_norm,
            keyPaddingMask=keyPaddingMask,
            tdError=tdError,
            uncertainty=uncertainty,
            precision=precision,
            headGate=headGate,
            attentionBias=attentionBias)

        ssm_out = self.ssm(x_norm, keyPaddingMask=keyPaddingMask, tdError=tdError, uncertainty=uncertainty)

        w = torch.sigmoid(self.mix_gate(x_norm)) # [B,S,1]

        y = w * mhsa_out + (1 - w) * ssm_out # [B,S,E]
        if channelGate is not None:
            y = y * channelGate

        x = residual + self.dropout(y) * self.gamma

        residual = x
        x_norm = self.norm_ffn(x)

        ffn_out = self.ffn(x_norm)

        out = residual + self.dropout(ffn_out) * self.gamma_ffn

        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1) # [B,S,1]
            out = out * keep
        return out


class AttentionWorkspace(AGICoreModule):
    def __init__(
        self,
        latentDim: int,
        numLatents: int,
        iterations: int,
        temporalBasisCount: int,):
        super().__init__()
        self.latent_dim = int(latentDim)
        self.num_latents = int(numLatents)
        self.iterations = int(iterations)
        self.temporal_basis_count = int(temporalBasisCount)

        self.latent_queries = nn.Parameter(
            torch.randn(numLatents, latentDim) * 0.02)
        self.goal_gain = nn.Parameter(
            torch.randn(numLatents, latentDim) * 0.02)
        self.temporal_transforms = nn.Parameter(torch.empty(
            temporalBasisCount,
            numLatents,
            latentDim,
            latentDim))
        self.temporal_latent_prior = nn.Parameter(torch.zeros(
            temporalBasisCount,
            numLatents))
        self.aggregation_gain = nn.Parameter(torch.empty(
            numLatents,
            2,
            latentDim))
        self.source_gain = nn.Parameter(torch.ones(
            numLatents,
            2,
            latentDim))
        self.temporal_competition_gain = nn.Parameter(torch.tensor(3.98))
        self.object_competition_gain = nn.Parameter(torch.tensor(3.98))
        self.query_norm = nn.LayerNorm(latentDim)
        self.token_norm = nn.LayerNorm(latentDim)
        self.q_proj = nn.Linear(latentDim, latentDim, bias=False)
        self.k_proj = nn.Linear(latentDim, latentDim, bias=False)
        self.v_proj = nn.Linear(latentDim, latentDim, bias=False)
        self.out_proj = nn.Linear(latentDim, latentDim, bias=False)
        self.ffn_norm = nn.LayerNorm(latentDim)
        self.ffn = nn.Sequential(
            nn.Linear(latentDim, 2 * latentDim),
            nn.GELU(),
            nn.Linear(2 * latentDim, latentDim))
        identity = torch.eye(
            latentDim,
            device=self.temporal_transforms.device,
            dtype=self.temporal_transforms.dtype)
        with torch.no_grad():
            self.temporal_transforms.copy_(
                identity.view(1, 1, latentDim, latentDim)
                + 0.02 * torch.randn_like(self.temporal_transforms))
            self.aggregation_gain[:, 0].zero_()
            self.aggregation_gain[:, 1].fill_(1.0)

    def forward(
        self,
        tokens: torch.Tensor,
        objectCandidates: torch.Tensor,
        objectPriority: torch.Tensor,
        goal: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor]:
        B = tokens.size(0)
        latents = (
            self.latent_queries.unsqueeze(0).expand(B, -1, -1)
            + torch.tanh(self.goal_gain).unsqueeze(0)
            * goal.unsqueeze(1))
        token_state = self.token_norm(tokens)
        key = self.k_proj(token_state)
        object_state = self.token_norm(objectCandidates)
        object_key = self.k_proj(object_state)
        object_value = (
            self.v_proj(object_state)
            + 0.25 * objectCandidates)
        S = token_state.size(1)
        temporal_transforms = F.interpolate(
            self.temporal_transforms.permute(1, 2, 3, 0).reshape(
                1,
                self.num_latents * self.latent_dim * self.latent_dim,
                self.temporal_basis_count),
            size=S,
            mode="linear",
            align_corners=True).reshape(
                self.num_latents,
                self.latent_dim,
                self.latent_dim,
                S).permute(3, 0, 1, 2)
        temporal_latent_prior = F.interpolate(
            self.temporal_latent_prior.transpose(0, 1).unsqueeze(0),
            size=S,
            mode="linear",
            align_corners=True).squeeze(0)
        transformed_value = torch.einsum(
            "bsd,sldh->bslh",
            tokens,
            temporal_transforms)
        value = (
            self.v_proj(token_state).unsqueeze(2)
            + 0.25 * transformed_value)

        for _ in range(self.iterations):
            query = self.q_proj(self.query_norm(latents))
            scores = torch.matmul(
                query,
                key.transpose(-2, -1)) / math.sqrt(self.latent_dim)
            scores = scores + temporal_latent_prior.unsqueeze(0)
            scores = scores + 0.25 * torch.einsum(
                "bld,bsld->bls",
                query,
                transformed_value) / math.sqrt(self.latent_dim)
            scores = F.softplus(self.temporal_competition_gain) * scores
            object_scores = torch.einsum(
                "bld,bskd->blsk",
                query,
                object_key) / math.sqrt(self.latent_dim)
            object_scores = (
                F.softplus(self.object_competition_gain)
                * object_scores)
            if keyPaddingMask is not None:
                scores = scores.masked_fill(
                    keyPaddingMask.unsqueeze(1),
                    -torch.inf)
                object_scores = object_scores.masked_fill(
                    keyPaddingMask[:, None, :, None],
                    -torch.inf)
            allocation = torch.softmax(scores, dim=1)
            object_allocation = torch.softmax(object_scores, dim=1)
            if keyPaddingMask is None:
                valid_time_count = objectPriority.new_full(
                    (B, 1, 1, 1),
                    float(objectPriority.size(1)))
            else:
                valid_time_count = (~keyPaddingMask).sum(
                    dim=-1).view(B, 1, 1, 1)
            object_allocation = (
                object_allocation
                * valid_time_count
                * objectPriority.unsqueeze(1))
            if keyPaddingMask is not None:
                allocation = allocation.masked_fill(
                    keyPaddingMask.unsqueeze(1),
                    0.0)
                object_allocation = object_allocation.masked_fill(
                    keyPaddingMask[:, None, :, None],
                    0.0)
            allocation_mass = allocation.sum(
                dim=-1,
                keepdim=True)
            weights = allocation / (
                allocation_mass
                + allocation_mass.eq(0))
            mean_update = torch.einsum(
                "bls,bsld->bld",
                weights,
                value)
            mass_update = torch.einsum(
                "bls,bsld->bld",
                allocation,
                value)
            temporal_update = (
                self.aggregation_gain[:, 0].unsqueeze(0) * mean_update
                + self.aggregation_gain[:, 1].unsqueeze(0) * mass_update)
            object_allocation_mass = object_allocation.sum(
                dim=(2, 3),
                keepdim=True)
            object_weights = object_allocation / (
                object_allocation_mass
                + object_allocation_mass.eq(0))
            object_mean_update = torch.einsum(
                "blsk,bskd->bld",
                object_weights,
                object_value)
            object_mass_update = torch.einsum(
                "blsk,bskd->bld",
                object_allocation,
                object_value)
            object_update = (
                self.aggregation_gain[:, 0].unsqueeze(0)
                * object_mean_update
                + self.aggregation_gain[:, 1].unsqueeze(0)
                * object_mass_update)
            workspace_update = (
                self.source_gain[:, 0].unsqueeze(0) * temporal_update
                + self.source_gain[:, 1].unsqueeze(0) * object_update)
            latents = latents + self.out_proj(
                workspace_update)
            latents = latents + self.ffn(
                self.ffn_norm(latents))

        source_mass = torch.stack([
            allocation.sum(dim=(1, 2)),
            object_allocation.sum(dim=(1, 2, 3))], dim=-1)
        return latents, weights, object_weights, source_mass


class GoalConditionedHebbianFusion(AGICoreModule):
    def __init__(
        self,
        numModes: int,
        embedDim: int,):
        super().__init__()
        self.num_modes = int(numModes)
        self.embed_dim = int(embedDim)

        self.base_weights = nn.Parameter(torch.empty(
            self.num_modes,
            self.embed_dim,
            self.embed_dim))
        self.gate_head = nn.Sequential(
            nn.Linear(5 * self.embed_dim, 2 * self.embed_dim),
            nn.SiLU(),
            nn.Linear(2 * self.embed_dim, 1))
        self.register_buffer(
            "hebbian_memory",
            torch.zeros(
                1,
                self.num_modes,
                self.embed_dim,
                self.embed_dim),
            persistent=False)
        self.ResetParameters()

    def ResetParameters(self) -> None:
        identity = torch.eye(
            self.embed_dim,
            device=self.base_weights.device,
            dtype=self.base_weights.dtype)
        with torch.no_grad():
            self.base_weights.copy_(
                identity.unsqueeze(0)
                + 0.02 * torch.randn_like(self.base_weights))
        for module in self.gate_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        self.ResetHebbianMemory()

    @torch.no_grad()
    def EnsureB(self, B: int) -> None:
        if self.hebbian_memory.size(0) != B:
            self.hebbian_memory = self.hebbian_memory.new_zeros(
                B,
                self.num_modes,
                self.embed_dim,
                self.embed_dim)

    @torch.no_grad()
    def ResetHebbianMemory(
        self,
        doneMask: Optional[torch.Tensor] = None,
        ) -> None:
        if doneMask is None:
            self.hebbian_memory.zero_()
            return
        mask = doneMask.view(-1)
        if mask.numel() != self.hebbian_memory.size(0):
            raise ValueError(
                "Attention fusion reset mask must match its batch size")
        self.hebbian_memory[mask] = 0

    @torch.no_grad()
    def UpdateHebbianMemory(
        self,
        inputs: torch.Tensor,
        fused: torch.Tensor,
        alpha: torch.Tensor,
        updateRows: torch.Tensor,
        ) -> None:
        source = inputs.detach()
        target = fused.detach().unsqueeze(1).expand(
            -1,
            self.num_modes,
            -1)
        hebbian = torch.einsum(
            "bme,bmf->bmef",
            source,
            target) / math.sqrt(self.embed_dim)
        update_rate = alpha.view(-1, 1, 1, 1)
        updated = (
            0.99 * self.hebbian_memory
            + update_rate * hebbian)

        def FrobeniusCap(x: torch.Tensor) -> torch.Tensor:
            norm = torch.linalg.vector_norm(
                x,
                ord=2,
                dim=(-2, -1),
                keepdim=True)
            scale = torch.minimum(
                torch.ones_like(norm),
                (2.0 * math.sqrt(self.embed_dim)) / (norm + 1e-8))
            return x * scale

        updated = FrobeniusCap(updated)
        row_mask = updateRows.view(-1, 1, 1, 1)
        self.hebbian_memory.copy_(torch.where(
            row_mask,
            updated,
            self.hebbian_memory))

    def forward(
        self,
        inputs: torch.Tensor,
        goal: torch.Tensor,
        priorLogits: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        updateRows: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        base = torch.einsum(
            "bme,mef->bmf",
            inputs,
            self.base_weights)
        memory = self.hebbian_memory.detach().clone()
        fast = torch.einsum("bme,bmef->bmf", inputs, memory)
        transformed = base + 0.10 * fast

        prior_weights = torch.softmax(priorLogits, dim=-1)
        context = torch.einsum("bme,bm->be", inputs, prior_weights)
        context_modes = context.unsqueeze(1).expand(-1, self.num_modes, -1)
        goal_modes = goal.unsqueeze(1).expand(-1, self.num_modes, -1)
        gate_input = torch.cat([
            inputs,
            context_modes,
            inputs - context_modes,
            inputs * context_modes,
            goal_modes], dim=-1)
        content_logits = self.gate_head(gate_input).squeeze(-1)
        fusion_weights = torch.softmax(
            content_logits + priorLogits,
            dim=-1)
        fused = torch.einsum(
            "bmf,bm->bf",
            transformed,
            fusion_weights)

        alpha = 0.01 * (
            1.0
            + precision * (0.25 + 0.75 * tdError.abs()))
        self.UpdateHebbianMemory(
            inputs,
            fused,
            alpha,
            updateRows)
        return fused, fusion_weights



class AttentionExtractor(AGICoreModule):
    def __init__(
        self,
        embedDim: int = 1024,
        sequenceLength: int = 32,
        numHeads: int = 16,
        temporalLayers: int = 12,
        capsDim: int = 256,
        routingIterations: int = 6,
        routingOutCaps: int = 8,
        gradientClipVal: float = 1.0,
        structuredDim: Optional[int] = None,
        goalDim: Optional[int] = None,
        objectTokenCount: int = 16,
        useDistributedGating: bool = True,
        slowDtScale: float = 0.25,):
        super().__init__()

        self.sequence_length = int(sequenceLength)
        self.gradient_clip_val = gradientClipVal
        self.output_dim = embedDim
        self.num_heads = numHeads
        self.caps_dim = capsDim
        self.routing_out_caps = int(routingOutCaps)
        self.structured_dim = int(structuredDim if structuredDim is not None else max(1, embedDim // 2))
        self.goal_dim = int(goalDim if goalDim is not None else self.structured_dim)
        self.object_token_count = int(objectTokenCount)
        self.use_distributed_gating = bool(useDistributedGating)

        self.temporal_blocks: nn.ModuleList = nn.ModuleList([
            TemporalAttention(
                embedDim,
                numHeads,
                idx,
                slowDtScale=(
                    slowDtScale ** ((idx + 1) / temporalLayers)))
            for idx in range(temporalLayers)])

        self.caps_in_proj = nn.Sequential(
            nn.Linear(embedDim, self.caps_dim),
            nn.LayerNorm(self.caps_dim),
            nn.SiLU())
        self.workspace_goal_proj = nn.Sequential(
            nn.LayerNorm(self.goal_dim),
            nn.Linear(self.goal_dim, self.caps_dim, bias=False))
        self.object_workspace_proj = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, self.caps_dim),
            nn.SiLU())
        nn.init.zeros_(self.object_workspace_proj[1].bias)
        self.workspace = AttentionWorkspace(
            latentDim=self.caps_dim,
            numLatents=self.routing_out_caps,
            iterations=routingIterations,
            temporalBasisCount=sequenceLength)
        self.caps_out_proj = nn.Linear(self.caps_dim, embedDim)

        self.output_proj = nn.Sequential(
            nn.Linear(embedDim, embedDim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedDim * 2, embedDim),
            nn.LayerNorm(embedDim))

        self.object_pool_norm = nn.LayerNorm(self.structured_dim)
        self.object_pool_key = nn.Linear(
            self.structured_dim,
            self.structured_dim)
        nn.init.zeros_(self.object_pool_key.bias)
        self.goal_object_query = nn.Sequential(
            nn.LayerNorm(self.goal_dim),
            nn.Linear(
                self.goal_dim,
                self.structured_dim,
                bias=False))
        self.quality_reliability = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, 1))
        nn.init.zeros_(self.quality_reliability[-1].weight)
        nn.init.zeros_(self.quality_reliability[-1].bias)

        self.motion_salience_head = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, 1, bias=False))
        self.prediction_salience_head = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, 1, bias=False))
        self.quality_channel_reliability = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, embedDim))
        nn.init.zeros_(self.quality_channel_reliability[-1].bias)
        self.motion_salience_gain = nn.Parameter(torch.tensor(0.0))
        self.prediction_salience_gain = nn.Parameter(torch.tensor(0.0))
        self.novelty_gain = nn.Parameter(torch.tensor(0.0))
        self.quality_log_gain = nn.Parameter(torch.tensor(0.0))
        self.presence_gain = nn.Parameter(torch.tensor(1.85))

        self.ontology_object_feature_dim = (
            46 + ModuleDim.PstSelfPartSemanticDim)
        self.ontology_object_encoder = nn.Sequential(
            nn.LayerNorm(self.ontology_object_feature_dim),
            nn.Linear(
                self.ontology_object_feature_dim,
                self.structured_dim * 2),
            nn.SiLU(),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim),
            nn.LayerNorm(self.structured_dim),)
        self.ontology_object_residual = nn.Sequential(
            nn.LayerNorm(self.structured_dim * 2),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim * 2),
            nn.SiLU(),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim),)
        self.ontology_object_gate = nn.Sequential(
            nn.LayerNorm(self.structured_dim * 2),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim),
            nn.Sigmoid(),)
        self.ontology_object_residual_gain = nn.Parameter(
            torch.tensor(-2.944439))
        self.entity_text_feature_dim = 515
        self.entity_text_encoder = nn.Sequential(
            nn.LayerNorm(self.entity_text_feature_dim),
            nn.Linear(
                self.entity_text_feature_dim,
                self.structured_dim * 2),
            nn.SiLU(),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim),
            nn.LayerNorm(self.structured_dim),)
        self.entity_text_residual = nn.Sequential(
            nn.LayerNorm(self.structured_dim * 2),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim * 2),
            nn.SiLU(),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim),)
        self.entity_text_gate = nn.Sequential(
            nn.LayerNorm(self.structured_dim * 2),
            nn.Linear(
                self.structured_dim * 2,
                self.structured_dim),
            nn.Sigmoid(),)
        self.entity_text_gain = nn.Parameter(torch.tensor(-2.944439))
        nn.init.zeros_(self.entity_text_residual[-1].weight)
        nn.init.zeros_(self.entity_text_residual[-1].bias)
        self.object_salience_allocation_head = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, self.structured_dim),
            nn.SiLU(),
            nn.Linear(self.structured_dim, 1),)
        nn.init.zeros_(self.object_salience_allocation_head[-1].weight)
        nn.init.zeros_(self.object_salience_allocation_head[-1].bias)

        self.object_seq_proj = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, embedDim),
            nn.GELU())
        self.motion_seq_proj = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, embedDim),
            nn.GELU())
        nn.init.zeros_(self.motion_seq_proj[1].bias)
        self.pred_error_seq_proj = nn.Sequential(
            nn.LayerNorm(self.structured_dim),
            nn.Linear(self.structured_dim, embedDim),
            nn.GELU())
        nn.init.zeros_(self.pred_error_seq_proj[1].bias)
        self.content_gate = nn.Linear(2 * self.structured_dim, 2)
        nn.init.zeros_(self.content_gate.weight)
        nn.init.zeros_(self.content_gate.bias)

        self.goal_bias_proj = nn.Sequential(
            nn.LayerNorm(self.goal_dim),
            nn.Linear(self.goal_dim, embedDim),
            nn.GELU())
        nn.init.zeros_(self.goal_bias_proj[1].bias)
        gate_in_dim = self.goal_dim + 3
        gate_hidden = max(32, embedDim // 2)
        self.mod_gate = nn.Sequential(
            nn.LayerNorm(gate_in_dim),
            nn.Linear(gate_in_dim, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, numHeads + embedDim))
        nn.init.zeros_(self.mod_gate[-1].weight)
        nn.init.zeros_(self.mod_gate[-1].bias)
        self.independent_head_scale = nn.Parameter(torch.zeros(5, numHeads))
        self.independent_channel_scale = nn.Parameter(torch.zeros(5, embedDim))
        self.independent_attention_scale = nn.Parameter(
            torch.zeros(5, sequenceLength))
        self.inhibition_return_gain = nn.Parameter(torch.tensor(-4.0))
        self.inhibition_return_decay = nn.Parameter(torch.tensor(1.4))

        self.scene_query = nn.Parameter(
            torch.randn(embedDim) * 0.02)
        self.readout_query = nn.Linear(
            embedDim,
            embedDim,
            bias=False)
        self.readout_key = nn.Linear(
            embedDim,
            embedDim,
            bias=False)
        self.readout_value = nn.Linear(
            embedDim,
            embedDim,
            bias=False)
        self.readout_gate = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, 3))
        nn.init.zeros_(self.readout_gate[-1].weight)
        nn.init.zeros_(self.readout_gate[-1].bias)
        self.readout_fusion = GoalConditionedHebbianFusion(
            numModes=3,
            embedDim=embedDim)
        self.local_detail_query = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim, bias=False))
        self.local_detail_value = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim, bias=False))
        self.local_detail_gain = nn.Parameter(torch.tensor(0.0))
        self.local_detail_radius = max(1, int(round(math.sqrt(sequenceLength))))
        self.workspace_ignition_gain = nn.Parameter(torch.tensor(0.0))
        self.workspace_ignition_count = max(
            1,
            min(self.routing_out_caps, int(round(math.sqrt(self.routing_out_caps)))))
        self.fast_student_residual = nn.Sequential(
            nn.LayerNorm(2 * embedDim),
            nn.Linear(2 * embedDim, embedDim),
            nn.GELU(),
            nn.Linear(embedDim, embedDim))
        self.fast_student_norm = nn.LayerNorm(embedDim)
        self.fast_student_gain = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.fast_student_residual[-1].weight)
        nn.init.zeros_(self.fast_student_residual[-1].bias)
        self.detail_token_student = nn.Sequential(
            nn.LayerNorm(embedDim),
            nn.Linear(embedDim, embedDim),
            nn.GELU(),
            nn.Linear(embedDim, embedDim))
        self.detail_token_gain = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.detail_token_student[-1].weight)
        nn.init.zeros_(self.detail_token_student[-1].bias)
        self.detail_student_query = nn.Linear(
            embedDim,
            embedDim,
            bias=False)
        nn.init.eye_(self.detail_student_query.weight)
        self.detail_student_residual = nn.Sequential(
            nn.LayerNorm(2 * embedDim),
            nn.Linear(2 * embedDim, embedDim),
            nn.GELU(),
            nn.Linear(embedDim, embedDim))
        self.detail_student_norm = nn.LayerNorm(embedDim)
        self.detail_student_gain = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.detail_student_residual[-1].weight)
        nn.init.zeros_(self.detail_student_residual[-1].bias)

    def ClipGrads(self):
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)

    def SanitizeModulators(
        self,
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        precision: torch.Tensor,
        B: int,
        x: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tdError is None:
            tdError = x.new_zeros(B)
        if uncertainty is None:
            uncertainty = x.new_zeros(B)

        return tdError, uncertainty, precision

    def ComputeDistributedGates(
        self,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_distributed_gating:
            return None, None
        B = goalBias.size(0)
        gate_in = torch.cat([
            goalBias,
            precision[:, None],
            tdError[:, None],
            uncertainty[:, None]], dim=-1)
        logits = self.mod_gate(gate_in)
        head_logits, channel_logits = logits.split([self.num_heads, self.output_dim], dim=-1)
        head_gate = 1.0 + 0.25 * torch.tanh(head_logits).view(B, self.num_heads, 1, 1)
        channel_gate = 1.0 + 0.25 * torch.tanh(channel_logits).view(B, 1, self.output_dim)
        return head_gate, channel_gate

    def NormalizeSignal(
        self,
        signal: Optional[torch.Tensor],
        B: int,
        x: torch.Tensor,
        signalName: str,) -> torch.Tensor:
        if signal is None:
            return x.new_zeros(B)
        if not isinstance(signal, torch.Tensor):
            raise TypeError(f"{signalName} must be a tensor")
        if signal.device != x.device:
            raise ValueError(f"{signalName} must be on the input device")
        if signal.dtype != x.dtype:
            raise TypeError(f"{signalName} must use the input dtype")
        if signal.shape == (B, 1):
            signal = signal[:, 0]
        if signal.shape != (B,):
            raise ValueError(f"{signalName} must have shape {(B,)}")
        if not bool(torch.isfinite(signal).all().item()):
            raise ValueError(f"{signalName} must be finite")
        return signal

    def ResolveStagePadding(
        self,
        x: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        stageMask: Optional[torch.Tensor],) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        B, S = x.shape[:2]
        if keyPaddingMask is not None:
            if keyPaddingMask.dtype != torch.bool:
                raise TypeError("keyPaddingMask must use torch.bool")
            if keyPaddingMask.device != x.device:
                raise ValueError("keyPaddingMask must be on the input device")
            if keyPaddingMask.shape != (B, S):
                raise ValueError(f"keyPaddingMask must have shape {(B, S)}")
        if stageMask is None:
            active = (
                torch.ones(B, S, dtype=torch.bool, device=x.device)
                if keyPaddingMask is None
                else ~keyPaddingMask)
        else:
            if stageMask.dtype != torch.bool:
                raise TypeError("stageMask must use torch.bool")
            if stageMask.device != x.device:
                raise ValueError("stageMask must be on the input device")
            if stageMask.shape != (B, S):
                raise ValueError(f"stageMask must have shape {(B, S)}")
            active = stageMask if keyPaddingMask is None else stageMask & ~keyPaddingMask
        effective = ~active
        return effective, active

    def ResolveDetailMask(
        self,
        x: torch.Tensor,
        activeMask: torch.Tensor,
        localDetailMask: Optional[torch.Tensor],) -> Optional[torch.Tensor]:
        if localDetailMask is None:
            return None
        if localDetailMask.dtype != torch.bool:
            raise TypeError("localDetailMask must use torch.bool")
        if localDetailMask.device != x.device:
            raise ValueError("localDetailMask must be on the input device")
        if localDetailMask.shape != activeMask.shape:
            raise ValueError(
                f"localDetailMask must have shape {tuple(activeMask.shape)}")
        return localDetailMask & activeMask

    def ComputeIndependentModulation(
        self,
        signals: torch.Tensor,
        sequenceLength: int,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        head_delta = torch.einsum(
            "bn,nh->bh",
            signals,
            self.independent_head_scale)
        channel_delta = torch.einsum(
            "bn,nd->bd",
            signals,
            self.independent_channel_scale)
        temporal_basis = F.interpolate(
            self.independent_attention_scale.unsqueeze(0),
            size=sequenceLength,
            mode="linear",
            align_corners=True).squeeze(0)
        attention_delta = torch.einsum(
            "bn,ns->bs",
            signals,
            temporal_basis)
        head_gate = 1.0 + 0.25 * torch.tanh(head_delta).unsqueeze(-1).unsqueeze(-1)
        channel_gate = 1.0 + 0.25 * torch.tanh(channel_delta).unsqueeze(1)
        return head_gate, channel_gate, 0.25 * torch.tanh(attention_delta)

    def ApplyInhibitionOfReturn(
        self,
        logits: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, C = logits.shape
        trace = logits.new_zeros(B, C)
        traces = []
        adjusted = []
        gain = F.softplus(self.inhibition_return_gain)
        decay = torch.sigmoid(self.inhibition_return_decay)
        for index in range(S):
            valid = (
                torch.ones(B, 1, dtype=torch.bool, device=logits.device)
                if keyPaddingMask is None
                else ~keyPaddingMask[:, index:index + 1])
            traces.append(trace)
            current = logits[:, index] - gain * trace
            probability = torch.softmax(current, dim=-1)
            probability = torch.where(valid, probability, torch.zeros_like(probability))
            trace = torch.where(valid, decay * trace + probability, trace)
            adjusted.append(current)
        return torch.stack(adjusted, dim=1), torch.stack(traces, dim=1)

    def ObjectEvidence(self, objectSeq: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(
            objectSeq,
            ord=2,
            dim=-1) / math.sqrt(float(objectSeq.size(-1)))

    def EncodeOntologyObjectSequence(
        self,
        objectSeq: torch.Tensor,
        ontologyAuxSeq: Dict[str, torch.Tensor],) -> torch.Tensor:
        presence = ontologyAuxSeq["PerceptualPresence"].unsqueeze(-1)
        ontology = torch.cat([
            ontologyAuxSeq["EntityRealmProb"],
            ontologyAuxSeq["ObjectAgencyProb"],
            ontologyAuxSeq["ObjectMotionLayerProb"],
            ontologyAuxSeq["LayerAgencyProb"].flatten(-2),
            ontologyAuxSeq["BodyMembershipProb"].unsqueeze(-1),
            ontologyAuxSeq["SelfPartSemantic"],
            ontologyAuxSeq["PhysicalInteractionProb"].unsqueeze(-1),
            ontologyAuxSeq["ContentMotionUV"],
            ontologyAuxSeq["ContentChangeProb"].unsqueeze(-1),
            presence,
        ], dim=-1)
        ontology_code = self.ontology_object_encoder(ontology)
        combined = torch.cat([objectSeq, ontology_code], dim=-1)
        residual = self.ontology_object_residual(combined)
        gate = self.ontology_object_gate(combined)
        gain = 0.25 * torch.sigmoid(
            self.ontology_object_residual_gain)
        return objectSeq + presence * gain * gate * residual

    def EncodeTextEntityObjectSequence(
        self,
        objectSeq: torch.Tensor,
        textAuxSeq: Dict[str, torch.Tensor],) -> torch.Tensor:
        confidence = textAuxSeq["EntityTextConfidence"].unsqueeze(-1)
        revision = torch.tanh(
            textAuxSeq["EntityTextRevision"].to(objectSeq.dtype).unsqueeze(-1)
            / 16.0)
        changed = textAuxSeq["EntityTextChanged"].unsqueeze(-1)
        features = torch.cat([
            textAuxSeq["EntityTextSemantic"],
            confidence,
            revision,
            changed,], dim=-1)
        code = self.entity_text_encoder(features)
        combined = torch.cat([objectSeq, code], dim=-1)
        residual = self.entity_text_residual(combined)
        gate = self.entity_text_gate(combined)
        return objectSeq + 0.25 * confidence * torch.sigmoid(
            self.entity_text_gain) * gate * residual

    def BuildObjectTimeCompetition(
        self,
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        returnDiagnostics: bool = False,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor]:
        object_evidence = self.ObjectEvidence(objectSeq)
        objects = self.object_pool_norm(objectSeq)
        keys = self.object_pool_key(objects)
        goal_query = self.goal_object_query(goalBias)
        goal_scores = torch.einsum(
            "bd,bskd->bsk",
            goal_query,
            keys) / math.sqrt(self.structured_dim)

        novelty_tail = torch.log1p(
            (objects[:, 1:] - objects[:, :-1]).square().mean(dim=-1))
        if keyPaddingMask is not None:
            pair_valid = (
                ~keyPaddingMask[:, 1:]
                & ~keyPaddingMask[:, :-1])
            novelty_tail = novelty_tail * pair_valid.unsqueeze(-1)
        novelty = torch.cat([
            novelty_tail.new_zeros(
                novelty_tail.size(0),
                1,
                novelty_tail.size(2)),
            novelty_tail], dim=1)

        motion_salience = F.softplus(
            self.motion_salience_head(motionSeq).squeeze(-1))
        prediction_salience = F.softplus(
            self.prediction_salience_head(predErrorSeq).squeeze(-1))
        reliability_logit = self.quality_reliability(
            qualitySeq).squeeze(-1)
        reliability = torch.sigmoid(reliability_logit)
        log_reliability = F.logsigmoid(reliability_logit)

        salience = (
            F.softplus(self.motion_salience_gain) * motion_salience
            + F.softplus(self.prediction_salience_gain)
            * (1.0 + tdError.abs().unsqueeze(-1))
            * prediction_salience)
        K = objectSeq.size(2)
        log_object_count = math.log(float(K))
        presence_logit = F.softplus(self.presence_gain) * (
            torch.log1p(float(K) * object_evidence)
            - log_object_count)
        presence_bias = F.logsigmoid(presence_logit) + math.log(2.0)
        real_object_logits = (
            goal_scores
            + F.softplus(self.novelty_gain) * novelty
            + presence_bias)
        object_logits = torch.cat([
            real_object_logits,
            real_object_logits.new_zeros(
                real_object_logits.size(0),
                real_object_logits.size(1),
                1)], dim=-1)
        real_salience_allocation_logits = (
            self.object_salience_allocation_head(objectSeq).squeeze(-1))
        salience_allocation_logits = torch.cat([
            real_salience_allocation_logits,
            real_salience_allocation_logits.new_zeros(
                real_salience_allocation_logits.size(0),
                real_salience_allocation_logits.size(1),
                1)], dim=-1)
        salience_allocation = torch.softmax(
            salience_allocation_logits,
            dim=-1) * float(K + 1)
        competition_logits = (
            object_logits
            + salience.unsqueeze(-1) * salience_allocation
            + F.softplus(self.quality_log_gain)
            * log_reliability.unsqueeze(-1))
        competition_logits, inhibition_trace = self.ApplyInhibitionOfReturn(
            competition_logits,
            keyPaddingMask)

        if keyPaddingMask is not None:
            competition_logits = competition_logits.masked_fill(
                keyPaddingMask.unsqueeze(-1),
                -torch.inf)

        B, S, candidate_count = competition_logits.shape
        competition_attention = torch.softmax(
            competition_logits.reshape(B, S * candidate_count),
            dim=-1).reshape(B, S, candidate_count)
        if keyPaddingMask is not None:
            competition_attention = competition_attention.masked_fill(
                keyPaddingMask.unsqueeze(-1),
                0.0)

        object_attention = torch.softmax(
            object_logits
            - F.softplus(self.inhibition_return_gain) * inhibition_trace,
            dim=-1)
        if keyPaddingMask is not None:
            object_attention = object_attention * (
                ~keyPaddingMask).unsqueeze(-1)
        real_object_attention = object_attention[..., :K]
        object_summary = (
            objectSeq
            * real_object_attention.unsqueeze(-1)).sum(dim=2)
        object_content_evidence = (
            real_object_attention * object_evidence).sum(dim=-1)

        temporal_logits = torch.logsumexp(competition_logits, dim=-1)
        temporal_log_attention = F.log_softmax(
            temporal_logits,
            dim=-1)
        if keyPaddingMask is not None:
            temporal_log_attention = temporal_log_attention.masked_fill(
                keyPaddingMask,
                0.0)
        temporal_attention = temporal_log_attention.exp()
        if keyPaddingMask is not None:
            temporal_attention = temporal_attention.masked_fill(
                keyPaddingMask,
                0.0)
        attention_bias = (
            temporal_log_attention * precision.unsqueeze(-1))

        content_weights = torch.softmax(
            self.content_gate(torch.cat([
                object_summary,
                motionSeq], dim=-1)),
            dim=-1)
        content_terms = torch.stack([
            self.object_seq_proj(object_summary)
            * object_content_evidence.unsqueeze(-1),
            self.motion_seq_proj(motionSeq)], dim=2)
        content = (
            content_terms
            * content_weights.unsqueeze(-1)).sum(dim=2) * 2.0

        event_gate = prediction_salience / (1.0 + prediction_salience)
        event_content = (
            self.pred_error_seq_proj(predErrorSeq)
            * event_gate.unsqueeze(-1))
        channel_reliability = 2.0 * torch.sigmoid(
            self.quality_channel_reliability(qualitySeq))
        structured = (
            (content + event_content) * channel_reliability) * (
                reliability * precision.unsqueeze(-1)).unsqueeze(-1)
        result = (
            structured,
            content_weights,
            competition_attention,
            object_attention,
            temporal_attention,
            attention_bias,
            reliability)
        if returnDiagnostics:
            return result + (inhibition_trace,)
        return result

    def PrepareAttentionSequence(
        self,
        x: torch.Tensor,
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        tdError: Optional[torch.Tensor],
        uncertainty: Optional[torch.Tensor],
        novelty: Optional[torch.Tensor] = None,
        risk: Optional[torch.Tensor] = None,
        informationGain: Optional[torch.Tensor] = None,
        localDetailMask: Optional[torch.Tensor] = None,
        stageActiveMask: Optional[torch.Tensor] = None,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[torch.Tensor],
            Optional[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            Dict[str, Any]]:
        B = x.size(0)
        self.EnsureB(B)
        tdError, uncertainty, precision = self.SanitizeModulators(
            tdError,
            uncertainty,
            precision,
            B,
            x)
        head_gate, channel_gate = self.ComputeDistributedGates(
            goalBias,
            precision,
            tdError,
            uncertainty)
        if novelty is None:
            novelty_signal = torch.log1p(
                (objectSeq[:, 1:] - objectSeq[:, :-1]).square().mean(
                    dim=(1, 2, 3))) if objectSeq.size(1) > 1 else x.new_zeros(B)
        else:
            novelty_signal = self.NormalizeSignal(
                novelty,
                B,
                x,
                "novelty")
        risk_signal = self.NormalizeSignal(risk, B, x, "risk")
        information_signal = self.NormalizeSignal(
            informationGain,
            B,
            x,
            "informationGain")
        signals = torch.stack([
            tdError.abs(),
            precision,
            novelty_signal,
            risk_signal,
            information_signal], dim=-1)
        independent_head, independent_channel, independent_attention = (
            self.ComputeIndependentModulation(signals, x.size(1)))
        head_gate = independent_head if head_gate is None else head_gate * independent_head
        channel_gate = (
            independent_channel
            if channel_gate is None
            else channel_gate * independent_channel)

        (
            structured,
            content_weights,
            competition_attention,
            object_attention,
            temporal_attention,
            attention_bias,
            reliability,
            inhibition_trace,
        ) = self.BuildObjectTimeCompetition(
            objectSeq,
            motionSeq,
            qualitySeq,
            predErrorSeq,
            goalBias,
            precision,
            tdError,
            keyPaddingMask,
            returnDiagnostics=True)
        attention_bias = attention_bias + independent_attention
        if keyPaddingMask is not None:
            attention_bias = attention_bias.masked_fill(keyPaddingMask, 0.0)

        x = x + 0.0625 * structured
        goal_term = self.goal_bias_proj(goalBias)
        x = x + 0.10 * goal_term.unsqueeze(1)
        if keyPaddingMask is not None:
            x = x * (~keyPaddingMask).unsqueeze(-1)

        attention_mass_square = competition_attention.square().sum(
            dim=(1, 2))
        effective_capacity = torch.where(
            attention_mass_square > 0,
            attention_mass_square.reciprocal(),
            attention_mass_square)
        attention_transfer = 0.5 * (
            object_attention[:, 1:]
            - object_attention[:, :-1]).abs().sum(dim=-1)
        if keyPaddingMask is None:
            attention_transfer_rate = attention_transfer.mean(dim=-1)
        else:
            transfer_valid = (
                ~keyPaddingMask[:, 1:]
                & ~keyPaddingMask[:, :-1])
            transfer_mass = transfer_valid.sum(dim=-1)
            attention_transfer_rate = (
                attention_transfer * transfer_valid).sum(dim=-1) / (
                    transfer_mass
                    + transfer_mass.eq(0))
        extras: Dict[str, Any] = {
            "structured_terms": x.new_tensor(3.0),
            "structured_weights": content_weights,
            "object_time_attention": competition_attention,
            "object_attention": object_attention,
            "temporal_attention_prior": temporal_attention,
            "frame_reliability": reliability,
            "background_attention": object_attention[..., -1],
            "effective_attention_capacity": effective_capacity,
            "attention_transfer_rate": attention_transfer_rate,
            "goal_bias_norm": goal_term.detach().norm(dim=-1),
            "precision": precision.detach(),}
        extras["independent_modulation"] = signals
        extras["inhibition_trace"] = inhibition_trace
        extras["local_detail_mask"] = localDetailMask
        extras["stage_active_mask"] = (
            stageActiveMask
            if stageActiveMask is not None
            else torch.ones(B, x.size(1), dtype=torch.bool, device=x.device))
        if head_gate is not None:
            extras["head_gate_mean"] = head_gate.detach().mean(
                dim=(1, 2, 3))
            extras["channel_gate_mean"] = channel_gate.detach().mean(
                dim=(1, 2))
        return (
            x,
            tdError,
            uncertainty,
            precision,
            head_gate,
            channel_gate,
            goal_term,
            attention_bias,
            extras)

    def GoalConditionedReadout(
        self,
        x: torch.Tensor,
        objectSeq: torch.Tensor,
        goalBias: torch.Tensor,
        goalTerm: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        attentionBias: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        extras: Dict[str, Any],
        ) -> torch.Tensor:
        object_evidence = self.ObjectEvidence(objectSeq)
        object_candidates = (
            self.object_workspace_proj(objectSeq)
            * object_evidence.unsqueeze(-1))
        (
            workspace_latents,
            workspace_time_attention,
            workspace_object_attention,
            workspace_source_mass,
        ) = self.workspace(
            self.caps_in_proj(x),
            object_candidates,
            extras["object_time_attention"][..., :-1],
            self.workspace_goal_proj(goalBias),
            keyPaddingMask)
        workspace_tokens = F.layer_norm(
            self.caps_out_proj(workspace_latents),
            (self.output_dim,))

        query = (
            self.scene_query.unsqueeze(0)
            + self.readout_query(goalTerm))
        temporal_scores = torch.einsum(
            "bd,bsd->bs",
            query,
            self.readout_key(x)) / math.sqrt(self.output_dim)
        temporal_scores = temporal_scores + attentionBias
        if keyPaddingMask is not None:
            temporal_scores = temporal_scores.masked_fill(
                keyPaddingMask,
                -torch.inf)
        temporal_weights = torch.softmax(temporal_scores, dim=-1)
        if keyPaddingMask is not None:
            temporal_weights = temporal_weights.masked_fill(
                keyPaddingMask,
                0.0)
        temporal_readout = torch.einsum(
            "bs,bsd->bd",
            temporal_weights,
            self.readout_value(x))

        workspace_scores = torch.einsum(
            "bd,bld->bl",
            query,
            self.readout_key(workspace_tokens)) / math.sqrt(self.output_dim)
        workspace_weights = torch.softmax(workspace_scores, dim=-1)
        dense_workspace_readout = torch.einsum(
            "bl,bld->bd",
            workspace_weights,
            self.readout_value(workspace_tokens))
        ignition_count = min(
            self.workspace_ignition_count,
            int(workspace_scores.size(1)))
        ignition_index = torch.topk(
            workspace_scores,
            k=ignition_count,
            dim=-1).indices
        ignition_mask = torch.zeros_like(workspace_scores, dtype=torch.bool)
        ignition_mask.scatter_(1, ignition_index, True)
        ignition_scores = workspace_scores.masked_fill(
            ~ignition_mask,
            -torch.inf)
        ignition_weights = torch.softmax(ignition_scores, dim=-1)
        ignition_readout = torch.einsum(
            "bl,bld->bd",
            ignition_weights,
            self.readout_value(workspace_tokens))
        workspace_readout = (
            dense_workspace_readout
            + torch.tanh(self.workspace_ignition_gain) * ignition_readout)

        local_mask = extras["local_detail_mask"]
        if local_mask is None:
            center = extras["temporal_attention_prior"].argmax(dim=-1)
            positions = torch.arange(x.size(1), device=x.device).view(1, -1)
            local_mask = (
                positions - center.unsqueeze(-1)).abs() <= self.local_detail_radius
            if keyPaddingMask is not None:
                local_mask = local_mask & ~keyPaddingMask
        local_enabled = local_mask.any(dim=-1)
        fallback_index = extras["temporal_attention_prior"].argmax(dim=-1)
        fallback_mask = F.one_hot(
            fallback_index,
            num_classes=x.size(1)).to(dtype=torch.bool)
        safe_local_mask = local_mask | (
            fallback_mask & ~local_enabled.unsqueeze(-1))
        local_query = query + self.local_detail_query(goalTerm)
        local_scores = torch.einsum(
            "bd,bsd->bs",
            local_query,
            self.readout_key(x)) / math.sqrt(self.output_dim)
        local_scores = local_scores.masked_fill(~safe_local_mask, -torch.inf)
        local_weights = torch.softmax(local_scores, dim=-1)
        local_weights = (
            local_weights.masked_fill(~local_mask, 0.0)
            * local_enabled.to(dtype=local_weights.dtype).unsqueeze(-1))
        local_readout = torch.einsum(
            "bs,bsd->bd",
            local_weights,
            self.local_detail_value(x))
        full_temporal_readout = temporal_readout
        temporal_readout = (
            temporal_readout
            + torch.tanh(self.local_detail_gain) * local_readout)

        if keyPaddingMask is None:
            current_token = x[:, -1]
        else:
            positions = torch.arange(
                x.size(1),
                device=x.device).unsqueeze(0)
            current_index = positions.masked_fill(
                keyPaddingMask,
                0).amax(dim=-1)
            current_token = x[
                torch.arange(x.size(0), device=x.device),
                current_index]
        current_readout = self.readout_value(current_token)

        readout_context = (
            goalTerm
            + temporal_readout
            + workspace_readout
            + current_readout)
        readout_logits = self.readout_gate(readout_context)
        readout_weights = torch.softmax(readout_logits, dim=-1)
        readout_terms = torch.stack([
            temporal_readout,
            workspace_readout,
            current_readout], dim=1)
        if keyPaddingMask is None:
            update_rows = torch.ones(
                x.size(0),
                dtype=torch.bool,
                device=x.device)
        else:
            update_rows = (~keyPaddingMask).any(dim=-1)
        out, fusion_weights = self.readout_fusion(
            readout_terms,
            goalTerm,
            readout_logits,
            precision,
            tdError,
            update_rows)

        extras["temporal_readout_attention"] = temporal_weights
        extras["workspace_time_attention"] = workspace_time_attention
        extras["workspace_object_attention"] = (
            workspace_object_attention)
        extras["workspace_source_mass"] = workspace_source_mass
        extras["workspace_readout_attention"] = workspace_weights
        extras["workspace_ignition_mask"] = ignition_mask
        extras["workspace_ignition_attention"] = ignition_weights
        extras["local_detail_attention"] = local_weights
        extras["local_detail_readout"] = local_readout
        extras["full_temporal_readout"] = full_temporal_readout
        extras["readout_weights"] = readout_weights
        extras["fusion_weights"] = fusion_weights
        return self.output_proj(out)

    def ForwardPreparedFull(
        self,
        x: torch.Tensor,
        objectSeq: torch.Tensor,
        goalBias: torch.Tensor,
        goalTerm: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        headGate: Optional[torch.Tensor],
        channelGate: Optional[torch.Tensor],
        attentionBias: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        extras: Dict[str, Any],
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        for block in self.temporal_blocks:
            x = block(
                x,
                keyPaddingMask=keyPaddingMask,
                tdError=tdError,
                uncertainty=uncertainty,
                precision=precision,
                headGate=headGate,
                channelGate=channelGate,
                attentionBias=attentionBias)
        out = self.GoalConditionedReadout(
            x,
            objectSeq,
            goalBias,
            goalTerm,
            precision,
            tdError,
            attentionBias,
            keyPaddingMask,
            extras)
        return out, extras

    def LatestPreparedToken(
        self,
        x: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        ) -> torch.Tensor:
        if keyPaddingMask is None:
            return x[:, -1]
        positions = torch.arange(
            x.size(1),
            device=x.device).unsqueeze(0)
        current_index = positions.masked_fill(
            keyPaddingMask,
            0).amax(dim=-1)
        current = x[
            torch.arange(x.size(0), device=x.device),
            current_index]
        valid = (~keyPaddingMask).any(dim=-1).unsqueeze(-1)
        return current * valid.to(dtype=x.dtype)

    def ForwardFastPrepared(
        self,
        x: torch.Tensor,
        goalTerm: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        ) -> torch.Tensor:
        current = self.LatestPreparedToken(x, keyPaddingMask)
        residual = self.fast_student_residual(
            torch.cat([current, goalTerm], dim=-1))
        return self.fast_student_norm(
            current + torch.tanh(self.fast_student_gain) * residual)

    def ForwardDetailPrepared(
        self,
        x: torch.Tensor,
        goalTerm: torch.Tensor,
        localDetailMask: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(x.size(0))
        if localDetailMask is None:
            localDetailMask = torch.zeros(
                B,
                x.size(1),
                device=x.device,
                dtype=torch.bool)
        selected_index = localDetailMask.nonzero(as_tuple=False)
        selected_count = localDetailMask.sum(dim=-1)
        summary = x.new_zeros(B, x.size(-1))
        if selected_index.numel() > 0:
            selected = x[
                selected_index[:, 0],
                selected_index[:, 1]]
            selected = (
                selected
                + torch.tanh(self.detail_token_gain)
                * self.detail_token_student(selected))
            selected_query = self.detail_student_query(
                goalTerm).index_select(0, selected_index[:, 0])
            selected_score = (
                selected * selected_query).sum(dim=-1) / math.sqrt(
                    float(x.size(-1)))
            for rowIndex in range(B):
                row_selected = selected_index[:, 0].eq(rowIndex)
                if not bool(row_selected.any().item()):
                    continue
                row_weight = torch.softmax(
                    selected_score[row_selected],
                    dim=0)
                summary[rowIndex] = (
                    row_weight.unsqueeze(-1)
                    * selected[row_selected]).sum(dim=0)
        residual = self.detail_student_residual(
            torch.cat([summary, goalTerm], dim=-1))
        output = self.detail_student_norm(
            summary + torch.tanh(self.detail_student_gain) * residual)
        return output, selected_count

    def IndexAttentionExtras(
        self,
        extras: Dict[str, Any],
        rowIndex: torch.Tensor,
        batchSize: int,
        ) -> Dict[str, Any]:
        return {
            name: (
                value.index_select(0, rowIndex)
                if (
                    torch.is_tensor(value)
                    and value.dim() > 0
                    and int(value.size(0)) == batchSize)
                else value)
            for name, value in extras.items()}

    def ScatterAttentionExtras(
        self,
        extras: Dict[str, Any],
        update: Dict[str, Any],
        rowIndex: torch.Tensor,
        batchSize: int,
        ) -> Dict[str, Any]:
        merged = dict(extras)
        update_rows = int(rowIndex.numel())
        for name, value in update.items():
            if (
                torch.is_tensor(value)
                and value.dim() > 0
                and int(value.size(0)) == update_rows
            ):
                existing = merged.get(name)
                if (
                    torch.is_tensor(existing)
                    and existing.dim() > 0
                    and int(existing.size(0)) == batchSize
                    and existing.shape[1:] == value.shape[1:]
                ):
                    destination = existing.clone()
                else:
                    destination = value.new_zeros(
                        batchSize,
                        *value.shape[1:])
                destination.index_copy_(0, rowIndex, value)
                merged[name] = destination
            elif name not in merged:
                merged[name] = value
        return merged

    @torch.no_grad()
    def IndexAttentionState(
        self,
        state: Dict[str, Any],
        rowIndex: torch.Tensor,
        ) -> Dict[str, Any]:
        return {
            "fusion": state["fusion"].index_select(0, rowIndex),
            "mhsa": [
                {
                    "U": item["U"].index_select(0, rowIndex),
                    "V": item["V"].index_select(0, rowIndex),}
                for item in state["mhsa"]]}

    @torch.no_grad()
    def ScatterAttentionState(
        self,
        state: Dict[str, Any],
        rowIndex: torch.Tensor,
        ) -> None:
        self.readout_fusion.hebbian_memory.index_copy_(
            0,
            rowIndex,
            state["fusion"])
        for block, item in zip(self.temporal_blocks, state["mhsa"]):
            block.mhsa.U.index_copy_(0, rowIndex, item["U"])
            block.mhsa.V.index_copy_(0, rowIndex, item["V"])

    def RunFullPreparedRows(
        self,
        fullRunner: Any,
        rowIndex: torch.Tensor,
        x: torch.Tensor,
        objectSeq: torch.Tensor,
        goalBias: torch.Tensor,
        goalTerm: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        headGate: Optional[torch.Tensor],
        channelGate: Optional[torch.Tensor],
        attentionBias: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        extras: Dict[str, Any],
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B = int(x.size(0))

        def Select(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return (
                None
                if value is None
                else value.index_select(0, rowIndex))

        selected_extras = self.IndexAttentionExtras(
            extras,
            rowIndex,
            B)
        if int(rowIndex.numel()) == B:
            return fullRunner(
                x,
                objectSeq,
                goalBias,
                goalTerm,
                precision,
                tdError,
                uncertainty,
                headGate,
                channelGate,
                attentionBias,
                keyPaddingMask,
                selected_extras)
        full_state = self.ExportState()
        selected_state = self.IndexAttentionState(
            full_state,
            rowIndex)
        self.ImportState(selected_state)
        updated_state = None
        try:
            output, selected_extras = fullRunner(
                Select(x),
                Select(objectSeq),
                Select(goalBias),
                Select(goalTerm),
                Select(precision),
                Select(tdError),
                Select(uncertainty),
                Select(headGate),
                Select(channelGate),
                Select(attentionBias),
                Select(keyPaddingMask),
                selected_extras)
            updated_state = self.ExportState()
        finally:
            self.ImportState(full_state)
        if updated_state is not None:
            self.ScatterAttentionState(
                updated_state,
                rowIndex)
        return output, selected_extras

    def ConditionalComputeUnits(
        self,
        fullMask: torch.Tensor,
        fastMask: torch.Tensor,
        detailMask: torch.Tensor,
        localDetailMask: Optional[torch.Tensor],
        sequenceLength: int,
        trainStudents: bool,
        ) -> Dict[str, torch.Tensor]:
        dtype = torch.float32
        device = fullMask.device
        B = int(fullMask.size(0))
        full_unit_value = float(
            max(
                1,
                len(self.temporal_blocks) * int(sequenceLength)
                + self.workspace.iterations * self.routing_out_caps
                + 1))
        full_units = torch.full(
            (B,),
            full_unit_value,
            device=device,
            dtype=dtype)
        detail_count = (
            torch.zeros(B, device=device, dtype=dtype)
            if localDetailMask is None
            else localDetailMask.sum(dim=-1).to(dtype=dtype))
        fast_units = torch.ones(B, device=device, dtype=dtype)
        detail_units = detail_count + 1.0
        selected_units = (
            fullMask.to(dtype=dtype) * full_units
            + fastMask.to(dtype=dtype) * fast_units
            + detailMask.to(dtype=dtype) * detail_units)
        actual_units = (
            (fullMask | fastMask | detailMask).to(dtype=dtype) * full_units
            + fastMask.to(dtype=dtype) * fast_units
            + detailMask.to(dtype=dtype) * detail_units
            if trainStudents
            else selected_units)
        return {
            "selected_compute_units": selected_units,
            "actual_compute_units": actual_units,
            "full_compute_units": full_units,
            "selected_normalized_compute_fraction": (
                selected_units / full_units),
            "actual_normalized_compute_fraction": (
                actual_units / full_units),}

    def StudentDistillationLoss(
        self,
        student: torch.Tensor,
        teacher: torch.Tensor,
        sampleMask: torch.Tensor,
        ) -> torch.Tensor:
        if (
            student.dim() != 2
            or teacher.shape != student.shape
            or sampleMask.shape != student.shape[:1]
            or sampleMask.dtype != torch.bool
            or sampleMask.device != student.device
            or teacher.device != student.device
            or teacher.dtype != student.dtype
        ):
            raise ValueError(
                "attention distillation inputs must share [B, D]")
        per_row = F.smooth_l1_loss(
            student,
            teacher.detach(),
            reduction="none").mean(dim=-1)
        weight = sampleMask.to(dtype=student.dtype)
        return (per_row * weight).sum() / weight.sum().clamp_min(1.0)

    def ForwardConditional(
        self,
        x: torch.Tensor,
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        fullMask: torch.Tensor,
        fastMask: torch.Tensor,
        detailMask: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        returnExtras: bool = False,
        novelty: Optional[torch.Tensor] = None,
        risk: Optional[torch.Tensor] = None,
        informationGain: Optional[torch.Tensor] = None,
        stageMask: Optional[torch.Tensor] = None,
        localDetailMask: Optional[torch.Tensor] = None,
        trainStudents: bool = False,
        fullRunner: Optional[Any] = None,
        ) -> torch.Tensor:
        B = int(x.size(0))
        for name, mask in (
            ("fullMask", fullMask),
            ("fastMask", fastMask),
            ("detailMask", detailMask),
        ):
            if (
                not torch.is_tensor(mask)
                or mask.shape != (B,)
                or mask.dtype != torch.bool
                or mask.device != x.device
            ):
                raise ValueError(name + " must be boolean [B]")
        if bool((
            fullMask.to(dtype=torch.int8)
            + fastMask.to(dtype=torch.int8)
            + detailMask.to(dtype=torch.int8)
        ).gt(1).any().item()):
            raise ValueError("attention compute paths must be disjoint")
        keyPaddingMask, stage_active_mask = self.ResolveStagePadding(
            x,
            keyPaddingMask,
            stageMask)
        local_detail_mask = self.ResolveDetailMask(
            x,
            stage_active_mask,
            localDetailMask)
        (
            prepared,
            tdError,
            uncertainty,
            precision,
            head_gate,
            channel_gate,
            goal_term,
            attention_bias,
            extras,
        ) = self.PrepareAttentionSequence(
            x,
            objectSeq,
            motionSeq,
            qualitySeq,
            predErrorSeq,
            goalBias,
            precision,
            keyPaddingMask,
            tdError,
            uncertainty,
            novelty,
            risk,
            informationGain,
            local_detail_mask,
            stage_active_mask)
        runner = (
            self.ForwardPreparedFull
            if fullRunner is None
            else fullRunner)
        units = self.ConditionalComputeUnits(
            fullMask,
            fastMask,
            detailMask,
            local_detail_mask,
            int(x.size(1)),
            bool(trainStudents))
        extras.update(units)
        if bool(trainStudents):
            active_rows = (fullMask | fastMask | detailMask).nonzero(
                as_tuple=False).flatten()
            output = x.new_zeros(B, self.output_dim)
            if active_rows.numel() > 0:
                teacher_output, teacher_extras = self.RunFullPreparedRows(
                    runner,
                    active_rows,
                    prepared,
                    objectSeq,
                    goalBias,
                    goal_term,
                    precision,
                    tdError,
                    uncertainty,
                    head_gate,
                    channel_gate,
                    attention_bias,
                    keyPaddingMask,
                    extras)
                output.index_copy_(0, active_rows, teacher_output)
                extras = self.ScatterAttentionExtras(
                    extras,
                    teacher_extras,
                    active_rows,
                    B)
            if "local_detail_readout" not in extras:
                extras["local_detail_readout"] = x.new_zeros(
                    B,
                    self.output_dim)
            if "full_temporal_readout" not in extras:
                extras["full_temporal_readout"] = x.new_zeros(
                    B,
                    self.output_dim)
            fast_student = x.new_zeros(B, self.output_dim)
            fast_rows = fastMask.nonzero(as_tuple=False).flatten()
            if fast_rows.numel() > 0:
                fast_update = self.ForwardFastPrepared(
                    prepared.index_select(0, fast_rows),
                    goal_term.index_select(0, fast_rows),
                    None if keyPaddingMask is None else (
                        keyPaddingMask.index_select(0, fast_rows)))
                fast_student.index_copy_(0, fast_rows, fast_update)
            detail_student = x.new_zeros(B, self.output_dim)
            detail_count = torch.zeros(
                B,
                device=x.device,
                dtype=torch.long)
            detail_rows = detailMask.nonzero(as_tuple=False).flatten()
            if detail_rows.numel() > 0:
                detail_update, detail_update_count = (
                    self.ForwardDetailPrepared(
                        prepared.index_select(0, detail_rows),
                        goal_term.index_select(0, detail_rows),
                        None if local_detail_mask is None else (
                            local_detail_mask.index_select(0, detail_rows))))
                detail_student.index_copy_(
                    0,
                    detail_rows,
                    detail_update)
                detail_count.index_copy_(
                    0,
                    detail_rows,
                    detail_update_count)
            extras["fast_student_attention"] = fast_student
            extras["detail_student_attention"] = detail_student
            extras["detail_student_token_count"] = detail_count
            extras.update(units)
            if returnExtras:
                return output, extras
            return output
        output = x.new_zeros(B, self.output_dim)
        full_rows = fullMask.nonzero(as_tuple=False).flatten()
        if full_rows.numel() > 0:
            full_output, full_extras = self.RunFullPreparedRows(
                runner,
                full_rows,
                prepared,
                objectSeq,
                goalBias,
                goal_term,
                precision,
                tdError,
                uncertainty,
                head_gate,
                channel_gate,
                attention_bias,
                keyPaddingMask,
                extras)
            output.index_copy_(0, full_rows, full_output)
            extras = self.ScatterAttentionExtras(
                extras,
                full_extras,
                full_rows,
                B)
        fast_rows = fastMask.nonzero(as_tuple=False).flatten()
        if fast_rows.numel() > 0:
            fast_output = self.ForwardFastPrepared(
                prepared.index_select(0, fast_rows),
                goal_term.index_select(0, fast_rows),
                None if keyPaddingMask is None else keyPaddingMask.index_select(
                    0, fast_rows))
            output.index_copy_(0, fast_rows, fast_output)
        detail_rows = detailMask.nonzero(as_tuple=False).flatten()
        detail_count = torch.zeros(
            B,
            device=x.device,
            dtype=torch.long)
        if detail_rows.numel() > 0:
            detail_output, selected_count = self.ForwardDetailPrepared(
                prepared.index_select(0, detail_rows),
                goal_term.index_select(0, detail_rows),
                None if local_detail_mask is None else (
                    local_detail_mask.index_select(0, detail_rows)))
            output.index_copy_(0, detail_rows, detail_output)
            detail_count.index_copy_(0, detail_rows, selected_count)
        extras["detail_student_token_count"] = detail_count
        extras.update(units)
        if returnExtras:
            return output, extras
        return output

    def EnsureB(self, B: int) -> None:
        for block in self.temporal_blocks:
            block.mhsa.EnsureB(B)
        self.readout_fusion.EnsureB(B)

    def forward(
        self,
        x: torch.Tensor, # [B,S,E]
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None, # [-1 ,1] [B]
        uncertainty: Optional[torch.Tensor]=None, # [0 ,1] [B]
        returnExtras: bool = False,
        novelty: Optional[torch.Tensor] = None,
        risk: Optional[torch.Tensor] = None,
        informationGain: Optional[torch.Tensor] = None,
        stageMask: Optional[torch.Tensor] = None,
        localDetailMask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        keyPaddingMask, stage_active_mask = self.ResolveStagePadding(
            x,
            keyPaddingMask,
            stageMask)
        local_detail_mask = self.ResolveDetailMask(
            x,
            stage_active_mask,
            localDetailMask)
        (
            x,
            tdError,
            uncertainty,
            precision,
            head_gate,
            channel_gate,
            goal_term,
            attention_bias,
            extras,
        ) = self.PrepareAttentionSequence(
            x,
            objectSeq,
            motionSeq,
            qualitySeq,
            predErrorSeq,
            goalBias,
            precision,
            keyPaddingMask,
            tdError,
            uncertainty,
            novelty,
            risk,
            informationGain,
            local_detail_mask,
            stage_active_mask)

        for blk in self.temporal_blocks:
            x = blk(
                x,
                keyPaddingMask=keyPaddingMask,
                tdError=tdError,
                uncertainty=uncertainty,
                precision=precision,
                headGate=head_gate,
                channelGate=channel_gate,
                attentionBias=attention_bias)

        out = self.GoalConditionedReadout(
            x,
            objectSeq,
            goalBias,
            goal_term,
            precision,
            tdError,
            attention_bias,
            keyPaddingMask,
            extras)
        if returnExtras:
            return out, extras
        return out

    @torch.no_grad()
    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
        for block in self.temporal_blocks:
            block.mhsa.ResetHebbianMemory(doneMask=doneMask)
        self.readout_fusion.ResetHebbianMemory(doneMask=doneMask)

    @torch.no_grad()
    def ExportState(self) -> dict:
        st = {
            "fusion": self.readout_fusion.hebbian_memory.detach().clone(),
            "mhsa": []}

        for blk in self.temporal_blocks:
            mhsa = blk.mhsa
            st["mhsa"].append({
                "U": mhsa.U.detach().clone(),
                "V": mhsa.V.detach().clone(),})

        return st

    @torch.no_grad()
    def ImportState(self, st: dict):
        self.EnsureB(int(st["fusion"].size(0)))
        self.readout_fusion.hebbian_memory.copy_(st["fusion"])
        for block, saved in zip(self.temporal_blocks, st["mhsa"]):
            mhsa = block.mhsa
            mhsa.U.copy_(saved["U"])
            mhsa.V.copy_(saved["V"])


class AttentionOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: nn.Module,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankQ: int = 64,
        maxRankK: int = 64,
        maxRankV: int = 64,
        maxRankO: int = 64,):
        self.maxRankQ = int(maxRankQ)
        self.maxRankK = int(maxRankK)
        self.maxRankV = int(maxRankV)
        self.maxRankO = int(maxRankO)
        super().__init__(
            base,
            initRankEach=initRankEach,
            autoRank=autoRank,
            evThreshold=evThreshold,
            gradEma=gradEma,)
        self.EnableStudentTraining()
        self.EnableAttentionEnhancementTraining()
        self.register_load_state_dict_post_hook(
            self.RestoreStudentTraining)

    def EnableStudentTraining(self) -> None:
        prefixes = (
            "fast_student_",
            "detail_token_",
            "detail_student_")
        for name, parameter in self.base.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(True)

    def EnableAttentionEnhancementTraining(self) -> None:
        prefixes = (
            "independent_head_scale",
            "independent_channel_scale",
            "independent_attention_scale",
            "inhibition_return_gain",
            "inhibition_return_decay",
            "local_detail_query.",
            "local_detail_value.",
            "local_detail_gain",
            "workspace_ignition_gain")
        for name, parameter in self.base.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(True)

    def RestoreBaseTrainabilityAfterCommit(self) -> None:
        super().RestoreBaseTrainabilityAfterCommit()
        self.EnableStudentTraining()
        self.EnableAttentionEnhancementTraining()

    def RestoreStudentTraining(
        self,
        module: nn.Module,
        incompatibleKeys: Any,
        ) -> None:
        self.EnableStudentTraining()
        self.EnableAttentionEnhancementTraining()

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        L = len(self.base.temporal_blocks)
        assert L > 0, "AttentionExtractor.temporal_blocks is NULL"
        E = int(self.base.temporal_blocks[0].mhsa.embed_dim)

        def AllocLinear(addRank: int, device: torch.device, dtype: torch.dtype):
            A = nn.Parameter(torch.randn(addRank, E, device=device, dtype=dtype) * 1e-4) # [r, inDim]
            B = nn.Parameter(torch.zeros(E, addRank, device=device, dtype=dtype) * 1e-4) # [outDim, r]
            s = nn.Parameter(torch.tensor(1e-2, device=device, dtype=dtype))
            return A, B, s

        def ComposeLinear(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            s_eff = torch.tanh(s) * GetParametersScale(s)
            return s_eff * (b @ a)

        return {
            "q": SiteSpec("q", L, E, E, self.maxRankQ, AllocLinear, ComposeLinear),
            "k": SiteSpec("k", L, E, E, self.maxRankK, AllocLinear, ComposeLinear),
            "v": SiteSpec("v", L, E, E, self.maxRankV, AllocLinear, ComposeLinear),
            "o": SiteSpec("o", L, E, E, self.maxRankO, AllocLinear, ComposeLinear),}

    def ForwardWithDeltas(
        self,
        x: torch.Tensor, # [B,S,E]
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        *,
        objectSeq: torch.Tensor,
        motionSeq: torch.Tensor,
        qualitySeq: torch.Tensor,
        predErrorSeq: torch.Tensor,
        goalBias: torch.Tensor,
        precision: torch.Tensor,
        returnExtras: bool = False,
        **kwargs,) -> torch.Tensor:
        stage_mask = kwargs.pop("stageMask", None)
        local_detail_mask = kwargs.pop("localDetailMask", None)
        novelty = kwargs.pop("novelty", None)
        risk = kwargs.pop("risk", None)
        information_gain = kwargs.pop("informationGain", None)
        keyPaddingMask, stage_active_mask = self.base.ResolveStagePadding(
            x,
            keyPaddingMask,
            stage_mask)
        local_detail_mask = self.base.ResolveDetailMask(
            x,
            stage_active_mask,
            local_detail_mask)
        (
            h,
            tdError,
            uncertainty,
            precision,
            head_gate,
            channel_gate,
            goal_term,
            attention_bias,
            extras,
        ) = self.base.PrepareAttentionSequence(
            x,
            objectSeq,
            motionSeq,
            qualitySeq,
            predErrorSeq,
            goalBias,
            precision,
            keyPaddingMask,
            tdError,
            uncertainty,
            novelty,
            risk,
            information_gain,
            local_detail_mask,
            stage_active_mask)

        for layerIdx, blk in enumerate(self.base.temporal_blocks):
            h = self.ForwardBlockWithDeltas(
                blk=blk,
                x=h,
                keyPaddingMask=keyPaddingMask,
                tdError=tdError,
                uncertainty=uncertainty,
                precision=precision,
                delta=deltasPerLayer[layerIdx],
                headGate=head_gate,
                channelGate=channel_gate,
                attentionBias=attention_bias)

        out = self.base.GoalConditionedReadout(
            h,
            objectSeq,
            goalBias,
            goal_term,
            precision,
            tdError,
            attention_bias,
            keyPaddingMask,
            extras)
        if returnExtras:
            return out, extras
        return out

    def ForwardPreparedFullWithDeltas(
        self,
        x: torch.Tensor,
        objectSeq: torch.Tensor,
        goalBias: torch.Tensor,
        goalTerm: torch.Tensor,
        precision: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        headGate: Optional[torch.Tensor],
        channelGate: Optional[torch.Tensor],
        attentionBias: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        extras: Dict[str, Any],
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]],
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        for layerIdx, block in enumerate(self.base.temporal_blocks):
            x = self.ForwardBlockWithDeltas(
                blk=block,
                x=x,
                keyPaddingMask=keyPaddingMask,
                tdError=tdError,
                uncertainty=uncertainty,
                precision=precision,
                delta=deltasPerLayer[layerIdx],
                headGate=headGate,
                channelGate=channelGate,
                attentionBias=attentionBias)
        out = self.base.GoalConditionedReadout(
            x,
            objectSeq,
            goalBias,
            goalTerm,
            precision,
            tdError,
            attentionBias,
            keyPaddingMask,
            extras)
        return out, extras

    def ForwardConditional(self, *args, **kwargs) -> torch.Tensor:
        deltas = [
            self.ComposeLayerDelta(layerIdx)
            for layerIdx in range(self.layerCount)]

        def RunFull(
            x,
            objectSeq,
            goalBias,
            goalTerm,
            precision,
            tdError,
            uncertainty,
            headGate,
            channelGate,
            attentionBias,
            keyPaddingMask,
            extras,):
            return self.ForwardPreparedFullWithDeltas(
                x,
                objectSeq,
                goalBias,
                goalTerm,
                precision,
                tdError,
                uncertainty,
                headGate,
                channelGate,
                attentionBias,
                keyPaddingMask,
                extras,
                deltas)

        return self.base.ForwardConditional(
            *args,
            fullRunner=RunFull,
            **kwargs)

    def ResetHebbianMemory(self, doneMask: Optional[torch.Tensor] = None) -> None:
        self.base.ResetHebbianMemory(doneMask=doneMask)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        r = int(a.size(0))
        if r <= 0 or a.numel() == 0 or b.numel() == 0 or abs(float(scale)) < 1e-12:
            return False

        blk = self.base.temporal_blocks[layerIdx]
        mhsa = blk.mhsa
        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "q":
            mhsa.q_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        elif site == "k":
            mhsa.k_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        elif site == "v":
            mhsa.v_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        elif site == "o":
            mhsa.o_adapter.Grow(addRank=r, init=init, freezeOld=self.freezeOldPar)
        else:
            raise ValueError(f"Unknown site: {site}")
        return True

    def ForwardBlockWithDeltas(
        self,
        blk,
        x: torch.Tensor,
        keyPaddingMask: torch.Tensor,
        tdError: torch.Tensor,
        uncertainty: torch.Tensor,
        precision: torch.Tensor,
        delta: Dict[str, Optional[torch.Tensor]],
        headGate: Optional[torch.Tensor] = None,
        channelGate: Optional[torch.Tensor] = None,
        attentionBias: Optional[torch.Tensor] = None,) -> torch.Tensor:

        mhsa = blk.mhsa

        residual0 = x
        x_norm = blk.norm(x)

        def EffLinear(weight: torch.Tensor, adapter, d2: Optional[torch.Tensor]):
            W = weight
            d_base = adapter.DeltaWeight()
            if d_base is not None:
                W = W + d_base
            if d2 is not None:
                W = W + d2
            return W

        Wq = EffLinear(mhsa.q_proj.weight, mhsa.q_adapter, delta.get("q"))
        Wk = EffLinear(mhsa.k_proj.weight, mhsa.k_adapter, delta.get("k"))
        Wv = EffLinear(mhsa.v_proj.weight, mhsa.v_adapter, delta.get("v"))
        Wo = EffLinear(mhsa.out_proj.weight, mhsa.o_adapter, delta.get("o"))

        q_lin = F.linear(x_norm, Wq, mhsa.q_proj.bias)
        k_lin = F.linear(x_norm, Wk, mhsa.k_proj.bias)
        v_lin = F.linear(x_norm, Wv, mhsa.v_proj.bias)
        attended = mhsa.Attend(
            q_lin,
            k_lin,
            v_lin,
            tdError,
            uncertainty,
            precision,
            keyPaddingMask,
            headGate,
            attentionBias)
        mhsa_out = F.linear(attended, Wo, mhsa.out_proj.bias)

        ssm_out = blk.ssm(x_norm, keyPaddingMask=keyPaddingMask,tdError=tdError, uncertainty=uncertainty)

        w = torch.sigmoid(blk.mix_gate(x_norm))
        y = w * mhsa_out + (1.0 - w) * ssm_out
        if channelGate is not None:
            y = y * channelGate

        x1 = residual0 + blk.dropout(y) * blk.gamma

        residual1 = x1
        x1_norm = blk.norm_ffn(x1)
        ffn_out = blk.ffn(x1_norm)
        out2 = residual1 + blk.dropout(ffn_out) * blk.gamma_ffn

        if keyPaddingMask is not None:
            keep = (~keyPaddingMask).unsqueeze(-1)
            out2 = out2 * keep

        return out2


class TestAttentionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

        self.B = 2
        self.S = 16
        self.E = 32
        self.H = 4
        self.M = 3
        self.out_caps = 4

    def AttentionInputs(self, B: int, S: int, E: int, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        half = E // 2
        return {
            "objectSeq": torch.randn(B, S, 16, half, device=self.device, dtype=dtype),
            "motionSeq": torch.randn(B, S, half, device=self.device, dtype=dtype),
            "qualitySeq": torch.randn(B, S, half, device=self.device, dtype=dtype),
            "predErrorSeq": torch.randn(B, S, half, device=self.device, dtype=dtype),
            "goalBias": torch.randn(B, half, device=self.device, dtype=dtype),
            "precision": torch.ones(B, device=self.device, dtype=dtype),}

    def AttentionForward(self, model, x: torch.Tensor, **kwargs) -> torch.Tensor:
        args = self.AttentionInputs(int(x.size(0)), int(x.size(1)), int(x.size(2)), x.dtype)
        args.update(kwargs)
        return model(x, **args)

    def AdapterRankAndParams(self, adapter) -> Tuple[int, int]:
        rank_sum = 0
        param_cnt = 0
        if not hasattr(adapter, "A_list"):
            return 0, 0
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            rank_sum += int(A.shape[0])
            param_cnt += int(A.numel() + B.numel() + 1)
        return rank_sum, param_cnt

    def MhsaAllRanksAndParams(self, base) -> Tuple[List[Tuple[int,int,int,int]], Tuple[int,int,int,int]]:
        per_layer = []
        sum_q = sum_k = sum_v = sum_o = 0
        for blk in base.temporal_blocks:
            mhsa = blk.mhsa
            rq, _ = self.AdapterRankAndParams(mhsa.q_adapter)
            rk, _ = self.AdapterRankAndParams(mhsa.k_adapter)
            rv, _ = self.AdapterRankAndParams(mhsa.v_adapter)
            ro, _ = self.AdapterRankAndParams(mhsa.o_adapter)
            per_layer.append((rq, rk, rv, ro))
            sum_q += rq; sum_k += rk; sum_v += rv; sum_o += ro
        return per_layer, (sum_q, sum_k, sum_v, sum_o)

    def DeltaFromLinearAdapter(self, adapter) -> torch.Tensor:
        if (not hasattr(adapter, "A_list")) or len(adapter.A_list) == 0:
            out_f = getattr(adapter, "out_f", None)
            in_f = getattr(adapter, "in_f", None)
            if out_f is None or in_f is None:
                return torch.zeros(0, 0, device=self.device)
            return torch.zeros(out_f, in_f, device=self.device)
        out_f = adapter.out_f
        in_f = adapter.in_f
        delta = torch.zeros(out_f, in_f, device=adapter.A_list[0].device, dtype=adapter.A_list[0].dtype)
        for A, B, s in zip(adapter.A_list, adapter.B_list, adapter.alpha):
            s_eff = torch.tanh(s.detach()) * GetParametersScale(s.detach())
            delta = delta + s_eff * (B @ A)
        return delta

    def AttentionStateMaxAbsDiff(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        vals = [
            (a["fusion"] - b["fusion"]).abs().max().item()]
        for x, y in zip(a["mhsa"], b["mhsa"]):
            vals.append((x["U"] - y["U"]).abs().max().item())
            vals.append((x["V"] - y["V"]).abs().max().item())
        return max(vals)

    def TestFixedPlasticityUsesPreviousState(self):
        try:
            torch.manual_seed(444)
            attn = MultiHeadAttention(
                embedDim=64,
                numHeads=4).to(self.device).eval()
            B, S = 2, 16
            x = torch.randn(B, S, 64, device=self.device)
            td = torch.ones(B, device=self.device)
            unc = torch.zeros(B, device=self.device)
            attn.EnsureB(B)

            with torch.no_grad():
                first = attn(
                    x, x, x,
                    tdError=td,
                    uncertainty=unc,
                    precision=torch.ones_like(td))
                assert attn.U.norm().item() > 1e-8
                assert attn.V.norm().item() > 1e-8

                second = attn(
                    x, x, x,
                    tdError=td,
                    uncertainty=unc,
                    precision=torch.ones_like(td))
                assert not torch.allclose(
                    first,
                    second,
                    atol=1e-8,
                    rtol=1e-7)

                attn.ResetHebbianMemory()
                replay = attn(
                    x, x, x,
                    tdError=td,
                    uncertainty=unc,
                    precision=torch.ones_like(td))
                assert torch.allclose(
                    first,
                    replay,
                    atol=1e-7,
                    rtol=1e-6)
                assert attn.U.size(-1) == 8
                assert attn.V.size(-1) == 8
            print("FixedPlasticityUsesPreviousState passed.")
            return True
        except AssertionError as e:
            print(f"FixedPlasticityUsesPreviousState failed: {e}")
            return False
        except Exception as e:
            print(f"FixedPlasticityUsesPreviousState error: {e}")
            return False

    def TestSelectiveDoneReset(self):
        try:
            torch.manual_seed(445)
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=16,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2).to(self.device)
            model.eval()
            x = torch.randn(2, 16, 64, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x.dtype)
            with torch.no_grad():
                _ = model(
                    x,
                    tdError=torch.ones(2, device=self.device),
                    uncertainty=torch.zeros(2, device=self.device),
                    **args)
                st0 = model.ExportState()
                model.ResetHebbianMemory(doneMask=torch.zeros(2, dtype=torch.bool, device=self.device))
                st_false = model.ExportState()
                assert self.AttentionStateMaxAbsDiff(st0, st_false) < 1e-12, "all-false doneMask changed state"

                model.ResetHebbianMemory(doneMask=torch.tensor([True, False], device=self.device))
                fusion_now = model.readout_fusion.hebbian_memory
                assert fusion_now[0].abs().max().item() < 1e-12, "done row fusion memory not cleared"
                assert torch.allclose(
                    fusion_now[1],
                    st0["fusion"][1]), "non-done fusion memory changed"
                for blk_idx, s in enumerate(st0["mhsa"]):
                    mhsa_now = model.temporal_blocks[blk_idx].mhsa
                    assert mhsa_now.U[0].abs().max().item() < 1e-12 and mhsa_now.V[0].abs().max().item() < 1e-12, "done row U/V not cleared"
                    assert torch.allclose(mhsa_now.U[1], s["U"][1]) and torch.allclose(mhsa_now.V[1], s["V"][1]), "non-done row U/V changed"

                model.ResetHebbianMemory(doneMask=torch.ones(2, dtype=torch.bool, device=self.device))
                assert model.readout_fusion.hebbian_memory.abs().max().item() < 1e-12, "all-true doneMask did not clear fusion memory"
                for blk in model.temporal_blocks:
                    assert blk.mhsa.U.abs().max().item() < 1e-12 and blk.mhsa.V.abs().max().item() < 1e-12, "all-true doneMask did not clear MHSA"
            print("SelectiveDoneReset passed.")
            return True
        except AssertionError as e:
            print(f"SelectiveDoneReset failed: {e}")
            return False
        except Exception as e:
            print(f"SelectiveDoneReset error: {e}")
            return False

    def TestModulatorsPassThrough(self):
        try:
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=1,capsDim=16,routingIterations=2).to(self.device)
            model.eval()
            x = torch.randn(2, 16, 64, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x.dtype)
            td = torch.tensor([-0.5, 0.5], device=self.device)
            unc = torch.tensor([0.1, 0.7], device=self.device)
            precision = torch.tensor([0.2, 1.0], device=self.device)
            td_raw = torch.tensor([-1.0, 1.0], device=self.device)
            unc_raw = torch.tensor([0.0, 1.0], device=self.device)
            precision_raw = torch.tensor([0.05, 1.0], device=self.device)
            td_s, unc_s, precision_s = model.SanitizeModulators(td_raw, unc_raw, precision_raw, 2, x)
            assert td_s is td_raw and unc_s is unc_raw and precision_s is precision_raw, "modulators should pass through unchanged"
            with torch.no_grad():
                y, extras = model(x, tdError=td, uncertainty=unc, precision=precision, returnExtras=True, **{k: v for k, v in args.items() if k != "precision"})
            assert torch.isfinite(y).all(), "valid modulators produced non-finite output"
            assert torch.allclose(extras["precision"], precision), "precision should pass through unchanged"

            print("ModulatorsPassThrough passed.")
            return True
        except AssertionError as e:
            print(f"ModulatorsPassThrough failed: {e}")
            return False
        except Exception as e:
            print(f"ModulatorsPassThrough error: {e}")
            return False

    def TestRoutingAndDistributedGates(self):
        try:
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=1,capsDim=16,routingIterations=2).to(self.device)
            assert model.workspace.num_latents == 8
            assert tuple(model.workspace.latent_queries.shape) == (8, 16)
            B = 3
            goal = torch.randn(B, 32, device=self.device)
            td, unc, precision = model.SanitizeModulators(
                torch.randn(B, device=self.device),
                torch.rand(B, device=self.device),
                torch.ones(B, device=self.device),
                B,
                goal)
            head_gate, channel_gate = model.ComputeDistributedGates(goal, precision, td, unc)
            assert tuple(head_gate.shape) == (B, 4, 1, 1), f"head gate shape mismatch: {head_gate.shape}"
            assert tuple(channel_gate.shape) == (B, 1, 64), f"channel gate shape mismatch: {channel_gate.shape}"
            assert torch.allclose(head_gate, torch.ones_like(head_gate)), "head gate should initialize neutral"
            assert torch.allclose(channel_gate, torch.ones_like(channel_gate)), "channel gate should initialize neutral"
            print("RoutingAndDistributedGates passed.")
            return True
        except AssertionError as e:
            print(f"RoutingAndDistributedGates failed: {e}")
            return False

    def TestCapacityNotReduced(self):
        try:
            baseline_parameter_count = 657_724_557
            with torch.device("meta"):
                model = AttentionExtractor()
            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters())
            assert parameter_count >= baseline_parameter_count
            assert len(model.temporal_blocks) == 12
            ssm = model.temporal_blocks[0].ssm
            assert ssm.param_proj.out_features == (
                1024 * (1 + 2 * 16))
            assert tuple(model.workspace.temporal_transforms.shape) == (
                32,
                8,
                256,
                256)
            assert tuple(model.readout_fusion.base_weights.shape) == (
                3,
                1024,
                1024)
            assert tuple(model.readout_fusion.hebbian_memory.shape) == (
                1,
                3,
                1024,
                1024)
            print(
                "CapacityNotReduced passed. "
                f"parameters={parameter_count:,}")
            return True
        except AssertionError as e:
            print(f"CapacityNotReduced failed: {e}")
            return False
        except Exception as e:
            print(f"CapacityNotReduced error: {e}")
            return False
        except Exception as e:
            print(f"RoutingAndDistributedGates error: {e}")
            return False

    def TestDualTimeConstantSSM(self):
        try:
            ssm = SelectiveSSM(self.E, stateDim=4, convKernel=4, slowDtScale=0.25).to(self.device)
            assert ssm.param_proj.out_features == self.E * (1 + 2 * 4)
            ssm.train()
            x = torch.randn(self.B, self.S, self.E, device=self.device, requires_grad=True)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device)
            kpm[:, -2:] = True
            y = ssm(x, tdError=torch.randn(self.B, device=self.device), uncertainty=torch.rand(self.B, device=self.device), keyPaddingMask=kpm)
            assert y.shape == (self.B, self.S, self.E), f"dual SSM shape mismatch: {y.shape}"
            assert torch.isfinite(y).all(), "dual SSM output has non-finite values"
            assert y[:, -2:].abs().max().item() < 1e-6, "masked dual SSM positions should be zero"
            loss = y.square().mean()
            loss.backward()
            for n, p in ssm.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"dual SSM non-finite grad: {n}"
            print("DualTimeConstantSSM passed.")
            return True
        except AssertionError as e:
            print(f"DualTimeConstantSSM failed: {e}")
            return False
        except Exception as e:
            print(f"DualTimeConstantSSM error: {e}")
            return False

    def TestSimpleSSM(self):
        try:
            ssm = SelectiveSSM(self.E, stateDim=4, convKernel=4).to(self.device)
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device)
            kpm[:, -3:] = True
            unc = torch.randn(self.B, device=self.device)
            y = ssm(x, keyPaddingMask=kpm, tdError=torch.randn(self.B, device=self.device), uncertainty = unc)
            assert y.shape == (self.B, self.S, self.E), f"Output shape mismatch: {y.shape}"
            print("SimpleSSM test passed.")
            return True
        except AssertionError as e:
            print(f"SimpleSSM test failed: {e}")
            return False
        except Exception as e:
            print(f"SimpleSSM test error: {e}")
            return False

    def TestMultiHeadAttention(self):
        try:
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device); kpm[:, -3:] = True
            td_unc = torch.randn(self.B, device=self.device)
            attn = MultiHeadAttention(
                embedDim=self.E,
                numHeads=self.H).to(self.device).eval()
            attn.EnsureB(self.B)
            y1 = attn(
                x, x, x,
                keyPaddingMask=kpm,
                tdError=td_unc,
                uncertainty=td_unc.abs().sigmoid(),
                precision=torch.ones_like(td_unc))
            assert y1.shape == (self.B, self.S, self.E)
            assert not {"U", "V"} & set(attn.state_dict())
            assert attn.U.size(-1) == 8
            assert attn.V.size(-1) == 8
            assert attn.U.norm().item() > 0.0
            assert attn.V.norm().item() > 0.0
            print("MultiHeadAttention test passed.")
            return True
        except AssertionError as e:
            print(f"MultiHeadAttention test failed: {e}")
            return False
        except Exception as e:
            print(f"MultiHeadAttention test error: {e}")
            return False

    def TestTemporalAttention(self):
        try:
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device)
            ta = TemporalAttention(
                self.E,
                self.H,
                layerIdx=0).to(self.device)
            ta.mhsa.EnsureB(self.B)
            y = ta(
                x,
                keyPaddingMask=kpm,
                tdError=torch.randn(self.B, device=self.device).tanh(),
                uncertainty=torch.rand(self.B, device=self.device),
                precision=torch.ones(self.B, device=self.device))
            assert y.shape == (self.B, self.S, self.E)
            print("TemporalAttention test passed.")
            return True
        except AssertionError as e:
            print(f"TemporalAttention test failed: {e}")
            return False
        except Exception as e:
            print(f"TemporalAttention test error: {e}")
            return False

    def TestAttentionWorkspace(self):
        try:
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            mask = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device); mask[:, -2:] = True
            workspace = AttentionWorkspace(
                latentDim=self.E,
                numLatents=self.out_caps,
                iterations=3,
                temporalBasisCount=self.S).to(self.device)
            object_candidates = torch.randn(
                self.B,
                self.S,
                5,
                self.E,
                device=self.device)
            object_priority = torch.softmax(
                torch.randn(
                    self.B,
                    self.S,
                    5,
                    device=self.device).flatten(1),
                dim=-1).view(self.B, self.S, 5)
            y, weights, object_weights, source_mass = workspace(
                x,
                object_candidates,
                object_priority,
                torch.randn(self.B, self.E, device=self.device),
                mask)
            assert y.shape == (self.B, self.out_caps, self.E), f"Output shape mismatch: {y.shape}"
            assert weights.shape == (self.B, self.out_caps, self.S)
            assert object_weights.shape == (
                self.B,
                self.out_caps,
                self.S,
                5)
            assert source_mass.shape == (self.B, 2)
            assert torch.all(
                source_mass[:, 1] <= source_mass[:, 0] + 1e-5)
            assert torch.count_nonzero(weights[:, :, -2:]) == 0
            assert torch.count_nonzero(object_weights[:, :, -2:]) == 0
            assert torch.allclose(
                weights[:, :, :-2].sum(dim=-1),
                torch.ones(
                    self.B,
                    self.out_caps,
                    device=self.device))
            assert torch.allclose(
                object_weights[:, :, :-2].sum(dim=(2, 3)),
                torch.ones(
                    self.B,
                    self.out_caps,
                    device=self.device))
            normalized = F.normalize(weights[:, :, :-2], dim=-1)
            similarity = torch.matmul(
                normalized,
                normalized.transpose(-2, -1))
            off_diagonal = ~torch.eye(
                self.out_caps,
                dtype=torch.bool,
                device=self.device).unsqueeze(0)
            assert similarity.masked_select(off_diagonal).max() < 0.999
            object_normalized = F.normalize(
                object_weights[:, :, :-2].flatten(2),
                dim=-1)
            object_similarity = torch.matmul(
                object_normalized,
                object_normalized.transpose(-2, -1))
            assert object_similarity.masked_select(
                off_diagonal).max() < 0.99
            print("AttentionWorkspace test passed.")
            return True
        except AssertionError as e:
            print(f"AttentionWorkspace test failed: {e}")
            return False
        except Exception as e:
            print(f"AttentionWorkspace test error: {e}")
            return False

    def TestObjectTimeCompetition(self):
        try:
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=8,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2).to(self.device).eval()
            args = self.AttentionInputs(2, 8, 64)
            td = torch.tensor([-0.5, 0.5], device=self.device)
            mask = torch.zeros(2, 8, dtype=torch.bool, device=self.device)
            mask[:, :2] = True
            result = model.BuildObjectTimeCompetition(
                args["objectSeq"],
                args["motionSeq"],
                args["qualitySeq"],
                args["predErrorSeq"],
                args["goalBias"],
                args["precision"],
                td,
                mask)
            structured, weights, competition, objects, temporal, bias, reliability = result
            assert structured.shape == (2, 8, 64)
            assert weights.shape == (2, 8, 2)
            assert competition.shape == (2, 8, 17)
            assert objects.shape == (2, 8, 17)
            assert temporal.shape == bias.shape == reliability.shape == (2, 8)
            assert torch.allclose(competition.sum(dim=(1, 2)), torch.ones(2, device=self.device))
            assert torch.count_nonzero(competition[:, :2]) == 0
            print("ObjectTimeCompetition test passed.")
            return True
        except AssertionError as e:
            print(f"ObjectTimeCompetition test failed: {e}")
            return False
        except Exception as e:
            print(f"ObjectTimeCompetition test error: {e}")
            return False

    def TestOntologyObjectConditioning(self):
        try:
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=3,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2,
                objectTokenCount=4).to(self.device)
            B, S, K, D = 1, 3, 4, 32
            objects = torch.randn(
                B, S, K, D,
                device=self.device,
                requires_grad=True)
            auxiliary = {
                "EntityRealmProb": torch.softmax(torch.randn(B, S, K, 5, device=self.device), dim=-1),
                "ObjectAgencyProb": torch.softmax(torch.randn(B, S, K, 5, device=self.device), dim=-1),
                "ObjectMotionLayerProb": torch.sigmoid(torch.randn(B, S, K, 5, device=self.device)),
                "LayerAgencyProb": torch.softmax(torch.randn(B, S, K, 5, 5, device=self.device), dim=-1),
                "BodyMembershipProb": torch.sigmoid(torch.randn(B, S, K, device=self.device)),
                "SelfPartSemantic": torch.randn(
                    B, S, K, ModuleDim.PstSelfPartSemanticDim,
                    device=self.device),
                "PhysicalInteractionProb": torch.sigmoid(torch.randn(B, S, K, device=self.device)),
                "ContentMotionUV": torch.randn(B, S, K, 2, device=self.device),
                "ContentChangeProb": torch.sigmoid(torch.randn(B, S, K, device=self.device)),
                "PerceptualPresence": torch.sigmoid(torch.randn(B, S, K, device=self.device)),}
            enriched = model.EncodeOntologyObjectSequence(objects, auxiliary)
            assert enriched.shape == objects.shape

            changed = {name: value.clone() for name, value in auxiliary.items()}
            changed["EntityRealmProb"][:, :, 2] = torch.roll(
                changed["EntityRealmProb"][:, :, 2], shifts=1, dims=-1)
            enriched_changed = model.EncodeOntologyObjectSequence(
                objects,
                changed)
            unchanged_candidates = torch.tensor([0, 1, 3], device=self.device)
            assert torch.equal(
                enriched.index_select(2, unchanged_candidates),
                enriched_changed.index_select(2, unchanged_candidates))
            assert not torch.equal(
                enriched[:, :, 2],
                enriched_changed[:, :, 2])

            enriched.square().mean().backward()
            assert objects.grad is not None
            assert model.ontology_object_encoder[1].weight.grad is not None

            empty_objects = torch.zeros(
                B, S, K, D,
                device=self.device,
                requires_grad=True)
            sparse_auxiliary = {
                name: value.detach().clone()
                for name, value in auxiliary.items()}
            sparse_auxiliary["PerceptualPresence"].zero_()
            sparse_auxiliary["PerceptualPresence"][:, :, 1] = 1.0
            sparse_enriched = model.EncodeOntologyObjectSequence(
                empty_objects,
                sparse_auxiliary)
            absent = torch.tensor([0, 2, 3], device=self.device)
            assert torch.count_nonzero(
                sparse_enriched.index_select(2, absent)).item() == 0
            assert torch.count_nonzero(
                sparse_enriched[:, :, 1]).item() > 0
            sparse_enriched[:, :, 1].square().mean().backward()
            assert empty_objects.grad is not None
            assert torch.isfinite(empty_objects.grad).all()

            with torch.no_grad():
                allocation_hidden = model.object_salience_allocation_head[1]
                allocation_hidden.weight.copy_(torch.eye(D, device=self.device))
                allocation_hidden.bias.zero_()
                allocation_out = model.object_salience_allocation_head[-1]
                allocation_out.weight.zero_()
                allocation_out.weight[0, 0] = 1.0
                allocation_out.bias.zero_()
                probe = torch.zeros(B, S, K, D, device=self.device)
                probe[:, :, 0, 0] = 2.0
                probe[:, :, 0, 1] = -2.0
                probe[:, :, 1, 0] = -2.0
                probe[:, :, 1, 1] = 2.0
                allocation_logits = (
                    model.object_salience_allocation_head(probe).squeeze(-1))
            assert torch.all(
                allocation_logits[:, :, 0]
                > allocation_logits[:, :, 1])
            print("OntologyObjectConditioning passed.")
            return True
        except AssertionError as e:
            print(f"OntologyObjectConditioning failed: {e}")
            return False
        except Exception as e:
            print(f"OntologyObjectConditioning error: {e}")
            return False

    def TestEntityTextConditioning(self):
        try:
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=3,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2,
                objectTokenCount=4).to(self.device).train()
            B, S, K, D = 1, 3, 4, 32
            objects = torch.randn(B, S, K, D, device=self.device)
            confidence = torch.ones(B, S, K, device=self.device)
            confidence[:, :, 0] = 0.0
            auxiliary = {
                "EntityTextSemantic": torch.randn(
                    B, S, K, 512, device=self.device),
                "EntityTextConfidence": confidence,
                "EntityTextRevision": torch.ones(
                    B, S, K, device=self.device, dtype=torch.long),
                "EntityTextChanged": torch.ones(
                    B, S, K, device=self.device),}
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
            neutral = model.EncodeTextEntityObjectSequence(objects, auxiliary)
            assert torch.equal(neutral, objects)
            optimizer.zero_grad(set_to_none=True)
            neutral.square().mean().backward()
            optimizer.step()

            model.zero_grad(set_to_none=True)
            enriched = model.EncodeTextEntityObjectSequence(objects, auxiliary)
            assert torch.equal(enriched[:, :, 0], objects[:, :, 0])
            assert not torch.equal(enriched[:, :, 1:], objects[:, :, 1:])
            enriched[:, :, 1:].square().mean().backward()
            assert model.entity_text_encoder[1].weight.grad is not None
            assert model.entity_text_residual[1].weight.grad is not None
            assert model.entity_text_gate[1].weight.grad is not None
            assert model.entity_text_gain.grad is not None
            print("EntityTextConditioning passed.")
            return True
        except AssertionError as e:
            print(f"EntityTextConditioning failed: {e}")
            return False
        except Exception as e:
            print(f"EntityTextConditioning error: {e}")
            return False

    def TestGoalConditionedSelection(self):
        try:
            torch.manual_seed(446)
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=8,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2).to(self.device).eval()
            x = torch.randn(1, 8, 64, device=self.device)
            args = self.AttentionInputs(1, 8, 64)
            goal_a = args["goalBias"]
            goal_b = -goal_a
            with torch.no_grad():
                out_a, extras_a = model(
                    x,
                    returnExtras=True,
                    **args)
                model.ResetHebbianMemory()
                out_b, extras_b = model(
                    x,
                    returnExtras=True,
                    **{**args, "goalBias": goal_b})
            assert float((
                extras_a["object_time_attention"]
                - extras_b["object_time_attention"]
            ).abs().max()) > 1e-6
            assert float((out_a - out_b).abs().max()) > 1e-6
            print("GoalConditionedSelection test passed.")
            return True
        except AssertionError as e:
            print(f"GoalConditionedSelection test failed: {e}")
            return False
        except Exception as e:
            print(f"GoalConditionedSelection test error: {e}")
            return False

    def TestBottomUpSalienceAndReliability(self):
        try:
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=8,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2).to(self.device).eval()
            B, S, D = 1, 8, 32
            object_frame = torch.randn(
                B, 1, 16, D, device=self.device)
            objects = object_frame.expand(-1, S, -1, -1).clone()
            direction = torch.cat([
                torch.ones(D // 2, device=self.device),
                -torch.ones(D // 2, device=self.device)])
            motion = -direction.view(1, 1, D).expand(B, S, D).clone()
            motion[:, 3] = direction
            quality = torch.zeros(B, S, D, device=self.device)
            quality[:, 4, 0] = 10.0
            quality[:, 5, 0] = -10.0
            pred_error = torch.zeros_like(motion)
            goal = torch.zeros(B, D, device=self.device)
            with torch.no_grad():
                quality_head = model.quality_reliability[-1]
                quality_head.weight.zero_()
                quality_head.weight[:, 0] = 1.0
                quality_head.bias.zero_()
                salience_head = model.motion_salience_head[-1]
                salience_head.weight.copy_(direction.view(1, D) / D)
                result = model.BuildObjectTimeCompetition(
                    objects,
                    motion,
                    quality,
                    pred_error,
                    goal,
                    torch.ones(B, device=self.device),
                    torch.zeros(B, device=self.device),
                    None)
            temporal = result[4]
            reliability = result[6]
            assert torch.allclose(
                motion[:, 3].square().mean(),
                motion[:, 0].square().mean())
            assert temporal[0, 3] > temporal[0, 0]
            assert reliability[0, 4] > reliability[0, 5]
            assert temporal[0, 4] > temporal[0, 5]
            print("BottomUpSalienceAndReliability test passed.")
            return True
        except AssertionError as e:
            print(f"BottomUpSalienceAndReliability test failed: {e}")
            return False

    def TestObjectPresenceRejectsEmptyCandidates(self):
        try:
            K = 128
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=4,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2,
                objectTokenCount=K).to(self.device).eval()
            B, S, D = 1, 4, 32
            objects = torch.zeros(B, S, K, D, device=self.device)
            objects[:, :, 0] = torch.randn(B, S, D, device=self.device)
            objects.requires_grad_()
            zeros = torch.zeros(B, S, D, device=self.device)
            result = model.BuildObjectTimeCompetition(
                objects,
                zeros,
                zeros,
                zeros,
                zeros[:, 0],
                torch.ones(B, device=self.device),
                torch.zeros(B, device=self.device),
                None)
            object_attention = result[3]
            observed = object_attention[..., 0]
            empty = object_attention[..., 1:K]
            background = object_attention[..., -1]
            assert torch.all(observed > empty.amax(dim=-1))
            assert torch.all(empty.sum(dim=-1) < observed)
            assert torch.all(background > empty.amax(dim=-1))
            output = model(
                torch.randn(B, S, 64, device=self.device),
                objectSeq=objects,
                motionSeq=zeros,
                qualitySeq=zeros,
                predErrorSeq=zeros,
                goalBias=zeros[:, 0],
                precision=torch.ones(B, device=self.device),
                tdError=torch.zeros(B, device=self.device))
            output.square().mean().backward()
            assert objects.grad is not None
            assert torch.isfinite(objects.grad).all()
            print("ObjectPresenceRejectsEmptyCandidates passed.")
            return True
        except AssertionError as e:
            print(f"ObjectPresenceRejectsEmptyCandidates failed: {e}")
            return False
        except Exception as e:
            print(f"ObjectPresenceRejectsEmptyCandidates error: {e}")
            return False

    def TestAttentionMapsSupportAuxiliaryLearning(self):
        try:
            torch.manual_seed(448)
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=8,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2).to(self.device).train()
            x = torch.randn(2, 8, 64, device=self.device)
            args = self.AttentionInputs(2, 8, 64)
            _, extras = model(x, returnExtras=True, **args)
            supervised_maps = (
                "object_time_attention",
                "object_attention",
                "temporal_attention_prior",
                "workspace_time_attention",
                "workspace_object_attention",
                "temporal_readout_attention",
                "workspace_readout_attention",
                "readout_weights",
                "fusion_weights")
            assert all(extras[name].requires_grad for name in supervised_maps)
            loss = (
                -extras["object_time_attention"][:, -1, 0].log().mean()
                -extras["workspace_time_attention"][:, 0, -1].log().mean()
                -extras["readout_weights"][:, 0].log().mean())
            loss.backward()
            assert model.goal_object_query[-1].weight.grad is not None
            assert model.workspace.latent_queries.grad is not None
            assert model.readout_gate[-1].weight.grad is not None
            print("AttentionMapsSupportAuxiliaryLearning passed.")
            return True
        except AssertionError as e:
            print(f"AttentionMapsSupportAuxiliaryLearning failed: {e}")
            return False

    def TestFusionPlasticityUsesPreviousState(self):
        try:
            torch.manual_seed(449)
            fusion = GoalConditionedHebbianFusion(
                numModes=3,
                embedDim=32).to(self.device).eval()
            fusion.EnsureB(1)
            inputs = torch.randn(1, 3, 32, device=self.device)
            goal = torch.randn(1, 32, device=self.device)
            prior_logits = torch.randn(1, 3, device=self.device)
            precision = torch.ones(1, device=self.device)
            td = torch.ones(1, device=self.device)
            update_rows = torch.ones(
                1,
                dtype=torch.bool,
                device=self.device)
            with torch.no_grad():
                first, _ = fusion(
                    inputs,
                    goal,
                    prior_logits,
                    precision,
                    td,
                    update_rows)
                memory_after_first = fusion.hebbian_memory.clone()
                second, _ = fusion(
                    inputs,
                    goal,
                    prior_logits,
                    precision,
                    td,
                    update_rows)
                fusion.ResetHebbianMemory()
                replay, _ = fusion(
                    inputs,
                    goal,
                    prior_logits,
                    precision,
                    td,
                    update_rows)
            assert tuple(memory_after_first.shape) == (1, 3, 32, 32)
            assert memory_after_first.norm() > 0
            assert not torch.allclose(first, second)
            assert torch.allclose(first, replay, atol=1e-7, rtol=1e-6)
            print("FusionPlasticityUsesPreviousState passed.")
            return True
        except AssertionError as e:
            print(f"FusionPlasticityUsesPreviousState failed: {e}")
            return False
        except Exception as e:
            print(f"FusionPlasticityUsesPreviousState error: {e}")
            return False
        except Exception as e:
            print(f"AttentionMapsSupportAuxiliaryLearning error: {e}")
            return False
        except Exception as e:
            print(f"BottomUpSalienceAndReliability test error: {e}")
            return False

    def TestCurrentFrameOnlyHebbianUpdate(self):
        try:
            torch.manual_seed(447)
            attn = MultiHeadAttention(
                embedDim=64,
                numHeads=4).to(self.device).eval()
            first = torch.randn(1, 8, 64, device=self.device)
            second = torch.randn_like(first)
            second[:, -1] = first[:, -1]
            td = torch.tensor([0.5], device=self.device)
            uncertainty = torch.tensor([0.2], device=self.device)
            precision = torch.tensor([0.8], device=self.device)
            attn.EnsureB(1)
            with torch.no_grad():
                _ = attn(
                    first,
                    first,
                    first,
                    tdError=td,
                    uncertainty=uncertainty,
                    precision=precision)
                U_first = attn.U.clone()
                V_first = attn.V.clone()
                attn.ResetHebbianMemory()
                _ = attn(
                    second,
                    second,
                    second,
                    tdError=td,
                    uncertainty=uncertainty,
                    precision=precision)
            assert torch.allclose(attn.U, U_first)
            assert torch.allclose(attn.V, V_first)
            print("CurrentFrameOnlyHebbianUpdate test passed.")
            return True
        except AssertionError as e:
            print(f"CurrentFrameOnlyHebbianUpdate test failed: {e}")
            return False
        except Exception as e:
            print(f"CurrentFrameOnlyHebbianUpdate test error: {e}")
            return False

    def TestAllPaddingRowFinite(self):
        try:
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=8,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2).to(self.device).eval()
            x = torch.randn(2, 8, 64, device=self.device)
            args = self.AttentionInputs(2, 8, 64)
            mask = torch.zeros(2, 8, dtype=torch.bool, device=self.device)
            mask[0] = True
            with torch.no_grad():
                out, extras = model(
                    x,
                    keyPaddingMask=mask,
                    returnExtras=True,
                    **args)
            assert torch.isfinite(out).all()
            assert torch.count_nonzero(
                extras["object_time_attention"][0]) == 0
            assert torch.count_nonzero(
                extras["temporal_readout_attention"][0]) == 0
            for saved in model.ExportState()["mhsa"]:
                assert torch.count_nonzero(saved["U"][0]) == 0
                assert torch.count_nonzero(saved["V"][0]) == 0
            assert torch.count_nonzero(
                model.ExportState()["fusion"][0]) == 0
            print("AllPaddingRowFinite test passed.")
            return True
        except AssertionError as e:
            print(f"AllPaddingRowFinite test failed: {e}")
            return False
        except Exception as e:
            print(f"AllPaddingRowFinite test error: {e}")
            return False

    def TestVariableLengthWrapperParity(self):
        try:
            base = AttentionExtractor(
                embedDim=64,
                sequenceLength=16,
                numHeads=4,
                temporalLayers=1,
                capsDim=16,
                routingIterations=2).to(self.device).eval()
            wrapper = AttentionOnlineWrapper(
                base=base,
                initRankEach=0,
                autoRank=False).to(self.device).eval()
            x = torch.randn(2, 11, 64, device=self.device)
            args = self.AttentionInputs(2, 11, 64)
            mask = torch.zeros(2, 11, dtype=torch.bool, device=self.device)
            mask[:, :3] = True
            with torch.no_grad():
                base_out = base(
                    x,
                    keyPaddingMask=mask,
                    **args)
                base.ResetHebbianMemory()
                wrapper_out = wrapper(
                    x,
                    keyPaddingMask=mask,
                    **args)
            assert torch.allclose(
                base_out,
                wrapper_out,
                atol=1e-6,
                rtol=1e-5)
            print("VariableLengthWrapperParity test passed.")
            return True
        except AssertionError as e:
            print(f"VariableLengthWrapperParity test failed: {e}")
            return False
        except Exception as e:
            print(f"VariableLengthWrapperParity test error: {e}")
            return False

    def TestAttentionExtractor(self):
        try:
            model = AttentionExtractor(embedDim=self.E, sequenceLength=self.S, numHeads=self.H,temporalLayers=2, routingIterations=3, gradientClipVal=0.5).to(self.device)
            x = torch.randn(self.B, self.S, self.E, device=self.device)
            kpm = torch.zeros(self.B, self.S, dtype=torch.bool, device=self.device); kpm[:, -2:] = True
            td = torch.randn(self.B, device=self.device)
            y = self.AttentionForward(model, x, keyPaddingMask=kpm, tdError=td)
            assert y.shape == (self.B, self.E)

            loss = y.mean()
            loss.backward()
            model.ClipGrads()
            print("AttentionExtractor test passed.")
            return True
        except AssertionError as e:
            print(f"AttentionExtractor test failed: {e}")
            return False
        except Exception as e:
            print(f"AttentionExtractor test error: {e}")
            return False

    def TestAttentionExtractorIOShapes(self):
        try:
            batch_size = 2
            seq_len = 16
            embed_dim = 1024

            model = AttentionExtractor(
                embedDim=embed_dim,
                sequenceLength=seq_len,
                numHeads=16,
                temporalLayers=2,
                routingIterations=3,
                gradientClipVal=0.5).to(self.device)
            model.eval()

            x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
            kpm = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
            kpm[:, -2:] = True
            td = torch.randn(batch_size, device=self.device)
            uncertainty = torch.rand(batch_size, device=self.device)

            def PrintShape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            with torch.no_grad():
                PrintShape("input.x", x)
                PrintShape("input.keyPaddingMask", kpm)
                PrintShape("input.tdError", td)
                PrintShape("input.uncertainty", uncertainty)
                y = self.AttentionForward(model, x, keyPaddingMask=kpm, tdError=td, uncertainty=uncertainty)
                PrintShape("output.y", y)

            expected_out_shape = (batch_size, embed_dim)
            assert tuple(y.shape) == expected_out_shape, f"Output shape mismatch: {y.shape}"
            return True
        except AssertionError as e:
            print(f"TestAttentionExtractorIOShapes failed: {e}")
            return False
        except Exception as e:
            print(f"TestAttentionExtractorIOShapes error: {e}")
            return False

    def TrainStepSmoke(self):
        try:
            torch.manual_seed(123)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            x = torch.randn(8, 16, 64, device=self.device)
            td = torch.randn(8, device=self.device)
            y = torch.randn(8, 12, device=self.device)

            out = self.AttentionForward(model, x, keyPaddingMask=None, tdError=td)
            pred = head(out)
            loss = F.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.ClipGrads()

            grads_ok = True
            for _, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grads_ok = False; break
            assert grads_ok and head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Gradient invalid."
            opt.step()
            print("TrainStepSmoke passed.")
            return True
        except AssertionError as e:
            print(f"TrainStepSmoke failed: {e}")
            return False
        except Exception as e:
            print(f"TrainStepSmoke error: {e}")
            return False

    def NoNanAfterManySteps(self, steps: int = 30):
        try:
            torch.manual_seed(321)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            for t in range(steps):
                x = torch.randn(8, 16, 64, device=self.device)
                td = torch.randn(8, device=self.device)
                y = torch.randn(8, 12, device=self.device)

                pred = head(self.AttentionForward(model, x, tdError=td))
                loss = F.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                model.ClipGrads()
                for n, p in list(model.named_parameters()) + list(head.named_parameters()):
                    if p.grad is not None:
                        assert torch.isfinite(p.grad).all(), f"Non-finite grad at step {t}, {n}"
                opt.step()
            print("NoNanAfterManySteps passed.")
            return True
        except AssertionError as e:
            print(f"NoNanAfterManySteps failed: {e}")
            return False
        except Exception as e:
            print(f"NoNanAfterManySteps error: {e}")
            return False

    def ParamsActuallyChange(self, steps: int = 10):
        try:
            torch.manual_seed(111)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            with torch.no_grad():
                key_params = {
                    "mhsa_q_proj": next(p for p in model.temporal_blocks[0].mhsa.q_proj.parameters() if p.dim() >= 2),
                    "workspace_query": model.workspace.latent_queries,
                    "object_query": model.goal_object_query[1].weight,
                    "output_proj_w": next(p for p in model.output_proj[0].parameters() if p.dim() >= 2),
                    "head_w": head.weight}
                init_norms = {k: v.norm().item() for k, v in key_params.items()}

            for _ in range(steps):
                x = torch.randn(8, 16, 64, device=self.device)
                td = torch.randn(8, device=self.device)
                y = torch.randn(8, 12, device=self.device)
                pred = head(self.AttentionForward(model, x, tdError=td))
                loss = F.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                model.ClipGrads()
                opt.step()

            with torch.no_grad():
                new_norms = {k: v.norm().item() for k, v in key_params.items()}
            changed = any(abs(new_norms[k] - init_norms[k]) > 1e-6 for k in init_norms)
            assert changed, "Key parameters seem unchanged; suspected no updates."
            print("ParamsActuallyChange passed.")
            return True
        except AssertionError as e:
            print(f"ParamsActuallyChange failed: {e}")
            return False
        except Exception as e:
            print(f"ParamsActuallyChange error: {e}")
            return False

    def TestNormalTrainingConvergence(self, steps: int = 120, logEvery: int = 30):
        try:
            torch.manual_seed(222)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            B = 16
            data_x = torch.randn(B, 16, 64, device=self.device)
            data_y = torch.randn(B, 12, device=self.device)
            data_td = torch.randn(B, device=self.device)

            with torch.no_grad():
                start = F.mse_loss(head(self.AttentionForward(model, data_x, tdError=data_td)), data_y).item()

            for t in range(1, steps + 1):
                pred = head(self.AttentionForward(model, data_x, tdError=data_td))
                loss = F.mse_loss(pred, data_y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                model.ClipGrads()
                opt.step()
                if (t % logEvery) == 0 or t == 1:
                    print(f"[AttentionTrain] step {t}/{steps} | mse={loss.item():.6f}")

            with torch.no_grad():
                end = F.mse_loss(head(self.AttentionForward(model, data_x, tdError=data_td)), data_y).item()

            print(f"\n[AttentionTrain] loss start={start:.6f} -> end={end:.6f}")
            assert end <= 0.8 * start, "Training did not converge enough (<20% drop)."
            print("TestNormalTrainingConvergence passed.")
            return True
        except AssertionError as e:
            print(f"TestNormalTrainingConvergence failed: {e}")
            return False
        except Exception as e:
            print(f"TestNormalTrainingConvergence error: {e}")
            return False

    def WrapperForwardEqualWhenNoInitRank(self):
        try:
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)
            base.eval()
            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=False).to(self.device)
            wrapper.eval()

            x = torch.randn(3, 16, 64, device=self.device)
            kpm = torch.zeros(3, 16, dtype=torch.bool, device=self.device)
            args = self.AttentionInputs(3, 16, 64, x.dtype)
            with torch.no_grad():
                y_base = base(x, keyPaddingMask=kpm, tdError=None, **args)
                base.ResetHebbianMemory()
                y_wrap = wrapper(x, keyPaddingMask=kpm, tdError=None, **args)

            max_abs = (y_base - y_wrap).abs().max().item()
            assert max_abs < 1e-6, f"Wrapper forward differs when ranks=0: max_abs={max_abs:.3e}"
            print("WrapperForwardEqualWhenNoInitRank passed.")
            return True
        except AssertionError as e:
            print(f"WrapperForwardEqualWhenNoInitRank failed: {e}")
            return False
        except Exception as e:
            print(f"WrapperForwardEqualWhenNoInitRank error: {e}")
            return False

    def WrapperAPIBasics(self):
        try:
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)
            base.eval()
            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=True).to(self.device)
            wrapper.train()

            r = wrapper.Update("ranks")["ranks"]
            assert all(row.get("q",0)==0 and row.get("k",0)==0 and row.get("v",0)==0 and row.get("o",0)==0 for row in r["perLayer"])

            wrapper.Update("grow", growFactor=2.0, addEach=1)
            r2 = wrapper.Update("ranks")["ranks"]
            assert all(row["q"] >= 1 and row["k"] >= 1 and row["v"] >= 1 and row["o"] >= 1 for row in r2["perLayer"])

            wrapper.Update("accumulategrads")

            st = wrapper.Update("set", evThreshold=0.85, gradEma=0.8, **{"maxRank:q":64, "maxRank:k":64, "maxRank:v":64, "maxRank:o":64})
            assert st["ok"]

            wrapper.Update("rollback")
            r3 = wrapper.Update("ranks")["ranks"]
            assert all(row["q"] == 0 and row["k"] == 0 and row["v"] == 0 and row["o"] == 0 for row in r3["perLayer"])

            print("WrapperAPIBasics passed.")
            return True
        except AssertionError as e:
            print("WrapperAPIBasics failed:\n", e)
            return False
        except Exception as e:
            print("WrapperAPIBasics error:\n", e)
            return False

    def WrapperManualGrowTrainAndCommit(self):
        try:
            torch.manual_seed(11)
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)
            base.eval()

            wrapper = AttentionOnlineWrapper(base=base, initRankEach=2, autoRank=False).to(self.device)
            wrapper.train()

            head = nn.Linear(64, 12).to(self.device).train()
            opt = torch.optim.Adam(list(wrapper.CandParameters()) + list(head.parameters()), lr=3e-3)

            _ = wrapper.Update("grow", growFactor=2.0, addEach=0)

            for _ in range(6):
                x = torch.randn(8, 16, 64, device=self.device)
                td = torch.randn(8, device=self.device)
                y = torch.randn(8, 12, device=self.device)

                pred = head(self.AttentionForward(wrapper, x, tdError=td))
                loss = F.mse_loss(pred, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            expected = []
            for li in range(wrapper.layerCount):
                per = {}
                for site in ("q", "k", "v", "o"):
                    per[site] = wrapper.ComposeOne(site, li).detach().clone()
                expected.append(per)

            res = wrapper.Update("commit")
            assert res["ok"] and res["commit_stats"]["committed_triples"] > 0, "Nothing committed."

            per_layer_rp, sums = self.MhsaAllRanksAndParams(base)
            print(f"[ManualCommit] committed_triples={int(res['commit_stats']['committed_triples'])}, "f"committed_rank={int(res['commit_stats']['committed_rank'])}")
            print(f"[Injected ranks per-layer] {per_layer_rp}; sums(q,k,v,o)={sums}")

            atol, rtol = 1e-6, 1e-4
            for li, blk in enumerate(base.temporal_blocks):
                mhsa = blk.mhsa
                got_q = self.DeltaFromLinearAdapter(mhsa.q_adapter); exp_q = expected[li]["q"]
                got_k = self.DeltaFromLinearAdapter(mhsa.k_adapter); exp_k = expected[li]["k"]
                got_v = self.DeltaFromLinearAdapter(mhsa.v_adapter); exp_v = expected[li]["v"]
                got_o = self.DeltaFromLinearAdapter(mhsa.o_adapter); exp_o = expected[li]["o"]

                if not torch.allclose(exp_q, torch.zeros_like(exp_q)): assert torch.allclose(got_q, exp_q, atol=atol, rtol=rtol), f"[L{li}] q delta mismatch"
                if not torch.allclose(exp_k, torch.zeros_like(exp_k)): assert torch.allclose(got_k, exp_k, atol=atol, rtol=rtol), f"[L{li}] k delta mismatch"
                if not torch.allclose(exp_v, torch.zeros_like(exp_v)): assert torch.allclose(got_v, exp_v, atol=atol, rtol=rtol), f"[L{li}] v delta mismatch"
                if not torch.allclose(exp_o, torch.zeros_like(exp_o)): assert torch.allclose(got_o, exp_o, atol=atol, rtol=rtol), f"[L{li}] o delta mismatch"

            r = wrapper.Update("ranks")["ranks"]
            assert all(row["q"] == 0 and row["k"] == 0 and row["v"] == 0 and row["o"] == 0 for row in r["perLayer"])

            base.eval(); wrapper.eval()
            x_chk = torch.randn(2, 16, 64, device=self.device)
            kpm_chk = torch.zeros(2, 16, dtype=torch.bool, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x_chk.dtype)
            with torch.no_grad():
                y0 = base(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
                base.ResetHebbianMemory()
                y1 = wrapper(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
            assert torch.allclose(y0, y1, atol=1e-6, rtol=1e-4), "base vs wrapper mismatch after commit."

            print("WrapperManualGrowTrainAndCommit passed.")
            return True
        except AssertionError as e:
            print("WrapperManualGrowTrainAndCommit failed:\n", e)
            return False
        except Exception as e:
            print("WrapperManualGrowTrainAndCommit error:\n", e)
            return False

    def WrapperAutoInjectAndCommit(self):
        try:
            torch.manual_seed(2025)
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)

            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=True, evThreshold=0.9).to(self.device)

            _ = wrapper.Update("autogrow")
            r0 = wrapper.Update("ranks")["ranks"]
            seed_per_layer = [(row.get("q", 0), row.get("k", 0), row.get("v", 0), row.get("o", 0)) for row in r0["perLayer"]]
            seed_sums = (sum(r[0] for r in seed_per_layer),
                        sum(r[1] for r in seed_per_layer),
                        sum(r[2] for r in seed_per_layer),
                        sum(r[3] for r in seed_per_layer))
            print(f"[Seed ranks per-layer] {seed_per_layer}; sums(q,k,v,o)={seed_sums}")

            head = nn.Linear(64, 12).to(self.device)
            wrapper.train(); head.train()
            opt = torch.optim.Adam(list(wrapper.CandParameters()) + list(head.parameters()), lr=5e-3)

            for _ in range(8):
                x = torch.randn(16, 16, 64, device=self.device)
                td = torch.randn(16, device=self.device)
                y = torch.randn(16, 12, device=self.device)

                pred = head(self.AttentionForward(wrapper, x, tdError=td))
                loss = F.mse_loss(pred, y)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            res = wrapper.Update("commit")
            assert res["ok"], "Commit failed."
            stats = res.get("commit_stats", {})
            committed_rank = int(stats.get("committed_rank", 0))
            committed_triples = int(stats.get("committed_triples", 0))
            print(f"[AutoCommit] committed_triples={committed_triples}, committed_rank={committed_rank}")

            per_layer_rp, sums = self.MhsaAllRanksAndParams(base)
            print(f"[Injected ranks per-layer] {per_layer_rp}; sums(q,k,v,o)={sums}")
            assert (committed_rank > 0) == (sum(sums) > 0), "Committed rank/stat mismatch with adapters."

            base.eval(); wrapper.eval()
            x_chk = torch.randn(2, 16, 64, device=self.device)
            kpm_chk = torch.zeros(2, 16, dtype=torch.bool, device=self.device)
            args = self.AttentionInputs(2, 16, 64, x_chk.dtype)
            with torch.no_grad():
                y0 = base(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
                base.ResetHebbianMemory()
                y1 = wrapper(x_chk, keyPaddingMask=kpm_chk, tdError=None, **args)
            max_abs = (y0 - y1).abs().max().item()
            assert max_abs < 1e-6, f"Wrapper vs base mismatch after auto-commit: {max_abs:.3e}"

            print("WrapperAutoInjectAndCommit passed.")
            return True
        except AssertionError as e:
            print("WrapperAutoInjectAndCommit failed:\n", e)
            return False
        except Exception as e:
            print("WrapperAutoInjectAndCommit error:\n", e)
            return False

    def WrapperPipelineCompatible(self):
        try:
            torch.manual_seed(13)
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)

            wrapper = AttentionOnlineWrapper(base=base, initRankEach=2, autoRank=False).to(self.device)

            for site in wrapper.sites:
                for li in range(wrapper.layerCount):
                    for s_param in wrapper.cand[site][li]["s"]:
                        s_param.requires_grad_(False)

            wrapper.train();
            head = nn.Linear(64, 10).to(self.device)
            head.train()

            opt = torch.optim.Adam(list(head.parameters()) + list(wrapper.CandParameters()), lr=1e-3)

            x  = torch.randn(6, 16, 64, device=self.device)
            td = torch.randn(6, device=self.device)
            y  = torch.randn(6, 10, device=self.device)

            pred = head(self.AttentionForward(wrapper, x, tdError=td))
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Head grad invalid with wrapper."

            opt.step()
            print("WrapperPipelineCompatible passed.")
            return True

        except AssertionError as e:
            print("WrapperPipelineCompatible failed:", e)
            return False
        except Exception as e:
            print("WrapperPipelineCompatible error:", e)
            return False


    def GradCoverageReportAttention(self, min_ratio: float = 0.65):
        try:
            torch.manual_seed(7)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3, gradientClipVal=0.5).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.train(); head.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

            x  = torch.randn(8, 16, 64, device=self.device)
            td = torch.randn(8, device=self.device)
            y  = torch.randn(8, 12, device=self.device)

            pred = head(self.AttentionForward(model, x, tdError=td))
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            model.ClipGrads()

            named = dict(list(model.named_parameters()) + [('head.'+k, v) for k,v in head.named_parameters()])
            total_trainable = sum(1 for p in named.values() if p.requires_grad)
            total_with_grad = sum(1 for p in named.values() if (p.requires_grad and (p.grad is not None)))
            ratio = total_with_grad / max(1, total_trainable)

            must_have = [
                "temporal_blocks.0.mhsa.q_proj.weight",
                "temporal_blocks.0.mhsa.k_proj.weight",
                "temporal_blocks.0.mhsa.v_proj.weight",
                "temporal_blocks.0.mhsa.out_proj.weight",
                "temporal_blocks.0.ssm.in_proj.weight",
                "temporal_blocks.0.ssm.param_proj.weight",
                "workspace.latent_queries",
                "workspace.temporal_transforms",
                "workspace.temporal_latent_prior",
                "workspace.aggregation_gain",
                "workspace.source_gain",
                "object_workspace_proj.1.weight",
                "goal_object_query.1.weight",
                "quality_channel_reliability.1.weight",
                "content_gate.weight",
                "readout_query.weight",
                "readout_gate.1.weight",
                "readout_fusion.base_weights",
                "readout_fusion.gate_head.0.weight",
                "output_proj.0.weight",
                "head.weight",]
            missing = [n for n in must_have if (n in named) and (named[n].grad is None)]
            assert len(missing) == 0, f"The key layer does not get the gradient: {missing}"
            assert ratio >= min_ratio, f"Gradient coverage is too low: {ratio:.2%} < {min_ratio:.2%}"

            print(f"GradCoverageReportAttention passed. grad_ratio={ratio:.2%}")
            return True
        except AssertionError as e:
            print(f"GradCoverageReportAttention failed: {e}")
            return False
        except Exception as e:
            print(f"GradCoverageReportAttention error: {e}")
            return False

    def MaskInvariance(self):
        try:
            torch.manual_seed(8)
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)
            model.eval()
            B, S, E = 3, 16, 64
            x1 = torch.randn(B, S, E, device=self.device)
            x2 = x1.clone()

            kpm = torch.zeros(B, S, dtype=torch.bool, device=self.device)
            kpm[:, -4:] = True
            x2[:, -4:, :] += torch.randn(B, 4, E, device=self.device) * 10.0

            args = self.AttentionInputs(B, S, E, x1.dtype)
            with torch.no_grad():
                y1 = model(x1, keyPaddingMask=kpm, **args)
                model.ResetHebbianMemory()
                y2 = model(x2, keyPaddingMask=kpm, **args)

            max_abs = (y1 - y2).abs().max().item()
            assert max_abs < 5e-5, f"Mask invariance fails, max_abs={max_abs:.3e}"
            print("MaskInvariance passed.")
            return True
        except AssertionError as e:
            print(f"MaskInvariance failed: {e}")
            return False
        except Exception as e:
            print(f"MaskInvariance error: {e}")
            return False

    def HebbianMemoryLifecycleAttention(self):
        try:
            attn = MultiHeadAttention(
                embedDim=64,
                numHeads=4).to(self.device)
            attn.train()
            B,S,E = 2, 10, 64
            x = torch.randn(B, S, E, device=self.device)
            attn.EnsureB(B)

            U0 = attn.U.norm().item(); V0 = attn.V.norm().item()
            for _ in range(3):
                _ = attn(
                    x,
                    x,
                    x,
                    tdError=torch.randn(B, device=self.device).tanh(),
                    uncertainty=torch.rand(B, device=self.device),
                    precision=torch.ones(B, device=self.device))
            U1 = attn.U.norm().item(); V1 = attn.V.norm().item()
            assert U1 > U0 + 1e-8 and V1 > V0 + 1e-8, "MHSA Hebbian(U/V) not growing"

            attn.ResetHebbianMemory()
            assert attn.U.abs().max().item() < 1e-12 and attn.V.abs().max().item() < 1e-12, "MHSA Hebbian(U/V) not cleared"

            print("HebbianMemoryLifecycleAttention passed.")
            return True
        except AssertionError as e:
            print(f"HebbianMemoryLifecycleAttention failed: {e}")
            return False
        except Exception as e:
            print(f"HebbianMemoryLifecycleAttention error: {e}")
            return False

    def WrapperKeepsBaseEval(self):
        try:
            base = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)
            wrapper = AttentionOnlineWrapper(base=base, initRankEach=0, autoRank=True).to(self.device)
            wrapper.train()
            assert wrapper.training and (not base.training), "base should remain eval() when wrapper.train()"
            print("WrapperKeepsBaseEval passed.")
            return True
        except AssertionError as e:
            print(f"WrapperKeepsBaseEval failed: {e}")
            return False
        except Exception as e:
            print(f"WrapperKeepsBaseEval error: {e}")
            return False

    def SmallBatchSafety(self):
        try:
            model = AttentionExtractor(embedDim=64, sequenceLength=16, numHeads=4,temporalLayers=2, routingIterations=3).to(self.device)
            head = nn.Linear(64, 12).to(self.device)
            model.eval(); head.train()
            x = torch.randn(1, 16, 64, device=self.device)
            td = torch.randn(1, device=self.device)
            y = torch.randn(1, 12, device=self.device)

            pred = head(self.AttentionForward(model, x, tdError=td))
            loss = F.mse_loss(pred, y)
            head.zero_grad(set_to_none=True)
            loss.backward()
            assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all(), "Head gradient is abnormal when batch=1"
            print("SmallBatchSafety passed.")
            return True
        except AssertionError as e:
            print(f"SmallBatchSafety failed: {e}")
            return False
        except Exception as e:
            print(f"SmallBatchSafety error: {e}")
            return False

    def TestIndependentSignalsAndSparseSelection(self):
        try:
            torch.manual_seed(517)
            model = AttentionExtractor(
                embedDim=64,
                sequenceLength=8,
                numHeads=4,
                temporalLayers=1,
                routingIterations=2).to(self.device).eval()
            x = torch.randn(2, 8, 64, device=self.device)
            stage_mask = torch.ones(2, 8, dtype=torch.bool, device=self.device)
            stage_mask[:, :2] = False
            detail_mask = torch.zeros_like(stage_mask)
            detail_mask[:, 4:7] = True
            args = self.AttentionInputs(2, 8, 64, x.dtype)
            with torch.no_grad():
                out, extras = model(
                    x,
                    **args,
                    novelty=torch.tensor([0.2, 0.7], device=self.device),
                    risk=torch.tensor([0.8, 0.1], device=self.device),
                    informationGain=torch.tensor([0.3, 0.9], device=self.device),
                    stageMask=stage_mask,
                    localDetailMask=detail_mask,
                    returnExtras=True)
            assert out.shape == (2, 64)
            assert extras["independent_modulation"].shape == (2, 5)
            assert extras["inhibition_trace"].shape[:2] == (2, 8)
            assert extras["local_detail_attention"].shape == (2, 8)
            assert extras["workspace_ignition_mask"].dtype == torch.bool
            assert extras["temporal_attention_prior"][:, :2].abs().max().item() == 0.0
            assert extras["local_detail_attention"][~detail_mask].abs().max().item() == 0.0
            assert extras["workspace_ignition_mask"].sum(dim=-1).max().item() < model.routing_out_caps
            print("TestIndependentSignalsAndSparseSelection passed.")
            return True
        except AssertionError as e:
            print("TestIndependentSignalsAndSparseSelection failed:", e)
            return False
        except Exception as e:
            print("TestIndependentSignalsAndSparseSelection error:", e)
            return False

    def RunAll(self):
        results = {
            "SimpleSSM": self.TestSimpleSSM(),
            "MultiHeadAttention": self.TestMultiHeadAttention(),
            "TemporalAttention": self.TestTemporalAttention(),
            "AttentionWorkspace": self.TestAttentionWorkspace(),
            "ObjectTimeCompetition": self.TestObjectTimeCompetition(),
            "OntologyObjectConditioning": self.TestOntologyObjectConditioning(),
            "EntityTextConditioning": self.TestEntityTextConditioning(),
            "GoalConditionedSelection": self.TestGoalConditionedSelection(),
            "BottomUpSalienceAndReliability": self.TestBottomUpSalienceAndReliability(),
            "ObjectPresenceRejectsEmptyCandidates": self.TestObjectPresenceRejectsEmptyCandidates(),
            "AttentionMapsSupportAuxiliaryLearning": self.TestAttentionMapsSupportAuxiliaryLearning(),
            "CurrentFrameOnlyHebbianUpdate": self.TestCurrentFrameOnlyHebbianUpdate(),
            "FusionPlasticityUsesPreviousState": self.TestFusionPlasticityUsesPreviousState(),
            "AllPaddingRowFinite": self.TestAllPaddingRowFinite(),
            "VariableLengthWrapperParity": self.TestVariableLengthWrapperParity(),
            "AttentionExtractorForward": self.TestAttentionExtractor(),
            "AttentionExtractorIOShapes": self.TestAttentionExtractorIOShapes(),
            "TrainStepSmoke": self.TrainStepSmoke(),
            "NoNanAfterManySteps": self.NoNanAfterManySteps(),
            "ParamsActuallyChange": self.ParamsActuallyChange(),
            "NormalTrainingConvergence": self.TestNormalTrainingConvergence(),
            "WrapperForwardEqualWhenNoInitRank": self.WrapperForwardEqualWhenNoInitRank(),
            "WrapperAPIBasics": self.WrapperAPIBasics(),
            "WrapperManualGrowTrainAndCommit": self.WrapperManualGrowTrainAndCommit(),
            "WrapperAutoInjectAndCommit": self.WrapperAutoInjectAndCommit(),
            "WrapperPipelineCompatible": self.WrapperPipelineCompatible(),
            "GradCoverageReportAttention": self.GradCoverageReportAttention(),
            "MaskInvariance": self.MaskInvariance(),
            "HebbianMemoryLifecycleAttention": self.HebbianMemoryLifecycleAttention(),
            "WrapperKeepsBaseEval": self.WrapperKeepsBaseEval(),
            "FixedPlasticityUsesPreviousState": self.TestFixedPlasticityUsesPreviousState(),
            "SelectiveDoneReset": self.TestSelectiveDoneReset(),
            "ModulatorsPassThrough": self.TestModulatorsPassThrough(),
            "RoutingAndDistributedGates": self.TestRoutingAndDistributedGates(),
            "CapacityNotReduced": self.TestCapacityNotReduced(),
            "DualTimeConstantSSM": self.TestDualTimeConstantSSM(),
            "IndependentSignalsAndSparseSelection": self.TestIndependentSignalsAndSparseSelection(),
            "SmallBatchSafety": self.SmallBatchSafety(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\nAttention module tests (with wrapper): {passed}/{len(results)} passed.")
        return results
