from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from FunctionTools import SiteSpec, BaseOnlineWrapper, AGICoreModule, GrowableLoRALinear, GetParametersScale, HungarianAssignment
from ModuleMessagerManager import ModuleDim
from PhysicalStateModule import PSTWorldBinder


def KLDiagNormal(muQ: torch.Tensor, logstdQ: torch.Tensor, muP: torch.Tensor, logstdP: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(2 * logstdQ)
    var_p = torch.exp(2 * logstdP)
    kl = 0.5 * (((var_q + (muQ - muP) ** 2) / var_p).sum(-1) + 2 * (logstdP - logstdQ).sum(-1) - muQ.size(-1))
    return kl


def BalancedKL(muQ: torch.Tensor,logstdQ: torch.Tensor,muP: torch.Tensor,logstdP: torch.Tensor,alpha: float = 0.8,freeNats: float = 1.0,) -> torch.Tensor:
    mu_p_sg, logstd_p_sg = muP.detach(), logstdP.detach()
    mu_q_sg, logstd_q_sg = muQ.detach(), logstdQ.detach()
    kl_qp = KLDiagNormal(muQ, logstdQ, mu_p_sg, logstd_p_sg)
    kl_pq = KLDiagNormal(mu_q_sg, logstd_q_sg, muP, logstdP)
    kl = alpha * kl_qp + (1.0 - alpha) * kl_pq
    if freeNats and freeNats > 0:
        kl = torch.relu(kl - freeNats)
    return kl


@dataclass
class PredictedVisualPack:
    GlobalFeat: torch.Tensor
    ObjectTokens: torch.Tensor
    MotionPred: torch.Tensor
    IntegratedFeat: torch.Tensor


class PredictedVisualHead(nn.Module):
    def __init__(
        self,
        stateDim: int,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        integratedFeatDim: int = 1024,):
        super().__init__()
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.integrated_feat_dim = int(integratedFeatDim)

        hidden = max(int(stateDim), self.global_feat_dim, self.object_token_dim * 2)

        def head(outDim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(int(stateDim)),
                nn.Linear(int(stateDim), hidden),
                nn.GELU(),
                nn.Linear(hidden, int(outDim)),)

        self.global_head = head(self.global_feat_dim)
        self.object_head = head(self.num_object_tokens * self.object_token_dim)
        self.motion_head = head(self.motion_pred_dim)
        self.integrated_head = head(self.integrated_feat_dim)

    def forward(self, state: torch.Tensor) -> PredictedVisualPack:
        B = int(state.size(0))
        return PredictedVisualPack(
            GlobalFeat=self.global_head(state),
            ObjectTokens=self.object_head(state).view(B, self.num_object_tokens, self.object_token_dim),
            MotionPred=self.motion_head(state),
            IntegratedFeat=self.integrated_head(state),)

class VisualReconstructor(nn.Module):
    def __init__(
        self,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        integratedFeatDim: int = 1024,):
        super().__init__()
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.integrated_feat_dim = int(integratedFeatDim)
        self.recon_dim = min(256, self.object_token_dim)
        self.head_count = 8

        scene_in_dim = self.global_feat_dim + self.integrated_feat_dim

        self.scene_encoder = nn.Sequential(
            nn.LayerNorm(scene_in_dim),
            nn.Linear(scene_in_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.motion_encoder = nn.Sequential(
            nn.LayerNorm(self.motion_pred_dim),
            nn.Linear(self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(self.object_token_dim),
            nn.Linear(self.object_token_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.slot_context_fuser = nn.Sequential(
            nn.LayerNorm(self.recon_dim * 3),
            nn.Linear(self.recon_dim * 3, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.recon_dim),)

        self.slot_presence_head = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, 1),)

        self.slot_norm = nn.LayerNorm(self.recon_dim)

        self.slot_relation_attn = nn.MultiheadAttention(
            self.recon_dim,
            self.head_count,
            batch_first=True,)

        self.scene_from_objects = nn.MultiheadAttention(
            self.recon_dim,
            self.head_count,
            batch_first=True,)

        self.scene_to_objects = nn.MultiheadAttention(
            self.recon_dim,
            self.head_count,
            batch_first=True,)

        self.scene_norm = nn.LayerNorm(self.recon_dim)

        self.motion_to_object_gate = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.recon_dim),
            nn.Sigmoid(),)

        self.slot_ffn = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.recon_dim * 2),
            nn.GELU(),
            nn.Linear(self.recon_dim * 2, self.recon_dim),)

        self.context_to_object = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.object_token_dim),)

        self.slot_to_object = nn.Sequential(
            nn.LayerNorm(self.recon_dim),
            nn.Linear(self.recon_dim, self.object_token_dim),)

        self.object_decoder = nn.Sequential(
            nn.LayerNorm(self.recon_dim * 3),
            nn.Linear(self.recon_dim * 3, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.object_token_dim),)

        self.object_out_norm = nn.LayerNorm(self.object_token_dim)

        self.motion_reconstructor = nn.Sequential(
            nn.LayerNorm(self.motion_pred_dim + self.object_token_dim * 2),
            nn.Linear(self.motion_pred_dim + self.object_token_dim * 2, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.motion_pred_dim),)

        self.motion_out_norm = nn.LayerNorm(self.motion_pred_dim)

        self.global_reconstructor = nn.Sequential(
            nn.LayerNorm(self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim),
            nn.Linear(self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.global_feat_dim),)

        self.global_out_norm = nn.LayerNorm(self.global_feat_dim)

        self.integrated_reconstructor = nn.Sequential(
            nn.LayerNorm(self.integrated_feat_dim + self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim),
            nn.Linear(self.integrated_feat_dim + self.global_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.integrated_feat_dim),)

        self.integrated_out_norm = nn.LayerNorm(self.integrated_feat_dim)

        self.pred_error_basis = nn.Sequential(
            nn.LayerNorm(self.global_feat_dim + self.integrated_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim),
            nn.Linear(self.global_feat_dim + self.integrated_feat_dim + self.object_token_dim * 2 + self.motion_pred_dim, self.recon_dim),
            nn.GELU(),
            nn.Linear(self.recon_dim, self.global_feat_dim),)

    def forward(self, predictedVisual: PredictedVisualPack) -> Dict[str, torch.Tensor]:
        scene_context = self.scene_encoder(torch.cat([
            predictedVisual.GlobalFeat,
            predictedVisual.IntegratedFeat,], dim=-1))

        motion_context = self.motion_encoder(predictedVisual.MotionPred)

        slot_base = self.slot_encoder(predictedVisual.ObjectTokens)

        slot_state = self.slot_context_fuser(torch.cat([
            slot_base,
            scene_context.unsqueeze(1).expand(-1, self.num_object_tokens, -1),
            motion_context.unsqueeze(1).expand(-1, self.num_object_tokens, -1),], dim=-1))

        slot_norm = self.slot_norm(slot_state)
        slot_relation, _ = self.slot_relation_attn(
            slot_norm,
            slot_norm,
            slot_norm,
            need_weights=False)
        slot_state = slot_state + slot_relation + self.slot_ffn(slot_state)

        slot_presence_logits = self.slot_presence_head(slot_state).squeeze(-1)
        slot_weight = F.softmax(slot_presence_logits, dim=-1)
        slot_summary = (slot_state * slot_weight.unsqueeze(-1)).sum(dim=1) # [B,256]
        scene_seed = self.scene_norm(scene_context + motion_context + slot_summary).unsqueeze(1)
        slot_norm = self.slot_norm(slot_state)

        scene_context, _ = self.scene_from_objects(
            scene_seed,
            slot_norm,
            slot_norm,
            need_weights=False)

        scene_token = self.scene_norm(scene_seed + scene_context)

        object_scene_delta, _ = self.scene_to_objects(
            slot_norm,
            scene_token,
            scene_token,
            need_weights=False)

        motion_gate = self.motion_to_object_gate(motion_context).unsqueeze(1)

        slot_state = self.slot_norm(
            slot_state
            + motion_gate * object_scene_delta
            + self.slot_ffn(slot_state))

        slot_presence_logits = self.slot_presence_head(slot_state).squeeze(-1)
        slot_weight = F.softmax(slot_presence_logits, dim=-1)
        object_summary_internal = (slot_state * slot_weight.unsqueeze(-1)).sum(dim=1)
        scene_summary_internal = scene_token.squeeze(1) # [B,256]

        object_tokens = self.object_out_norm(
            predictedVisual.ObjectTokens
            + self.object_decoder(torch.cat([
                slot_state,
                scene_summary_internal.unsqueeze(1).expand(-1, self.num_object_tokens, -1),
                motion_context.unsqueeze(1).expand(-1, self.num_object_tokens, -1),], dim=-1)))

        slot_state_object = self.slot_to_object(slot_state)
        scene_summary = self.context_to_object(scene_summary_internal)
        object_summary = self.context_to_object(object_summary_internal)

        motion_delta = self.motion_reconstructor(torch.cat([
            predictedVisual.MotionPred,
            scene_summary,
            object_summary,], dim=-1))

        motion_pred = self.motion_out_norm(predictedVisual.MotionPred + motion_delta)

        global_delta = self.global_reconstructor(torch.cat([
            predictedVisual.GlobalFeat,
            scene_summary,
            object_summary,
            motion_pred,], dim=-1))

        global_feat = self.global_out_norm(predictedVisual.GlobalFeat + global_delta)

        integrated_delta = self.integrated_reconstructor(torch.cat([
            predictedVisual.IntegratedFeat,
            global_feat,
            scene_summary,
            object_summary,
            motion_pred,], dim=-1))

        integrated_feat = self.integrated_out_norm(predictedVisual.IntegratedFeat + integrated_delta)
        slot_entropy = -(slot_weight * F.log_softmax(slot_presence_logits, dim=-1)).sum(dim=-1)
        slot_confidence = 1.0 - slot_entropy / slot_entropy.new_tensor(float(self.num_object_tokens)).log()
        scene_agreement = (F.cosine_similarity(scene_summary, object_summary, dim=-1) + 1.0) * 0.5
        prior_confidence = (0.5 + 0.5 * slot_confidence) * scene_agreement

        return {
            "IntegratedFeat": integrated_feat, # [B,1024]
            "GlobalFeat": global_feat, # [B,1024]
            "ObjectTokens": object_tokens, # [B,K,512]
            "MotionPred": motion_pred, # [B,512]
            "SlotState": slot_state_object, # [B,K,512]
            "SlotPresenceLogits": slot_presence_logits, # [B,K]
            "SceneSummary": scene_summary, # [B,512]
            "ObjectSummary": object_summary, # [B,512]
            "PriorConfidence": prior_confidence, # [B]
            "PredErrorBasis": self.pred_error_basis(torch.cat([
                global_feat,
                integrated_feat,
                scene_summary,
                object_summary,
                motion_pred,], dim=-1)),} # [B,1024]

    def PairwiseCosine(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(tokens, dim=-1, eps=1e-6)
        return torch.matmul(normed, normed.transpose(1, 2))

    def SoftAlignObjects(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = float(source.size(-1)) ** 0.5
        weights = F.softmax(torch.matmul(source, target.transpose(1, 2)) / scale, dim=-1)
        return torch.matmul(weights, target)

    def InverseMappingLoss(
        self,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        sampleMask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        target_objects = targetVisualState.ObjectTokens.detach()
        target_motion = targetVisualState.MotionToken.detach()
        target_object_valid = torch.nan_to_num(
            targetVisualState.Auxiliary["ObjectGeometryValid"].detach().squeeze(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0).clamp(0.0, 1.0)
        target_object_prob = torch.nan_to_num(
            F.softmax(targetVisualState.SemanticNodes["node_logits"].detach(), dim=-1)[..., 1],
            nan=0.0,
            posinf=0.0,
            neginf=0.0).clamp(0.0, 1.0)
        target_object_weight = target_object_prob * target_object_valid
        target_weight_sum = target_object_weight.sum(dim=-1, keepdim=True)
        target_slot_weight = torch.where(
            target_weight_sum > 0.0,
            target_object_weight / target_weight_sum.clamp_min(1e-8),
            torch.zeros_like(target_object_weight))
        base_sample = target_objects.new_ones(target_objects.size(0))
        if sampleMask is not None:
            mask = sampleMask.detach().to(device=target_objects.device, dtype=target_objects.dtype).view(-1)
            if mask.numel() != target_objects.size(0):
                raise ValueError(
                    f"sampleMask must have {target_objects.size(0)} elements, got {mask.numel()}")
            base_sample = torch.nan_to_num(
                mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        object_sample = base_sample * (
            target_weight_sum.squeeze(-1) > 0.0).to(target_objects.dtype)

        def masked_mean(per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            numerator = (per_sample * mask).sum()
            return numerator / mask.sum().clamp_min(1.0)

        def object_mean(per_sample: torch.Tensor) -> torch.Tensor:
            return masked_mean(per_sample, object_sample)

        object_tokens = reconstructedVisualState["ObjectTokens"]
        slot_state = reconstructedVisualState["SlotState"]
        slot_logits = reconstructedVisualState["SlotPresenceLogits"]
        scene_summary = reconstructedVisualState["SceneSummary"]
        object_summary = reconstructedVisualState["ObjectSummary"]

        aligned_objects_for_target = self.SoftAlignObjects(target_objects, object_tokens)
        loss_inverse_object = object_mean(
            F.smooth_l1_loss(aligned_objects_for_target, target_objects, reduction="none").mean(dim=-1)
            .mul(target_slot_weight).sum(dim=-1))

        target_norm = F.normalize(target_objects, dim=-1, eps=1e-6)
        slot_norm = F.normalize(slot_state, dim=-1, eps=1e-6)
        target_slot_similarity = torch.matmul(target_norm, slot_norm.transpose(1, 2))
        slot_match_score = target_slot_similarity.max(dim=-1).values
        loss_inverse_slot = object_mean(
            ((1.0 - slot_match_score) * target_slot_weight).sum(dim=-1))

        target_to_slot_weight = F.softmax(target_slot_similarity, dim=-1)
        target_relation = self.PairwiseCosine(target_objects)
        aligned_slots_for_target = torch.matmul(target_to_slot_weight, slot_state)
        slot_relation = self.PairwiseCosine(aligned_slots_for_target)
        relation_weight = target_slot_weight.unsqueeze(1) * target_slot_weight.unsqueeze(2)
        loss_inverse_relation = object_mean((
            F.smooth_l1_loss(slot_relation, target_relation, reduction="none")
            * relation_weight).sum(dim=(1, 2)))

        pred_slot_prob = F.softmax(slot_logits, dim=-1)
        pred_presence_for_target = torch.matmul(target_to_slot_weight, pred_slot_prob.unsqueeze(-1)).squeeze(-1)
        pred_presence_for_target = pred_presence_for_target / pred_presence_for_target.sum(
            dim=-1, keepdim=True).clamp_min(1e-8)
        presence_kl = torch.where(
            target_slot_weight > 0.0,
            target_slot_weight * (
                target_slot_weight.clamp_min(1e-8).log()
                - pred_presence_for_target.clamp_min(1e-8).log()),
            torch.zeros_like(target_slot_weight))
        loss_inverse_presence = object_mean(presence_kl.sum(dim=-1))

        target_summary = (target_objects * target_slot_weight.unsqueeze(-1)).sum(dim=1)
        loss_inverse_scene = object_mean(
            1.0 - F.cosine_similarity(scene_summary, target_summary, dim=-1))
        loss_inverse_summary = object_mean(
            F.smooth_l1_loss(object_summary, target_summary, reduction="none").mean(dim=-1))

        loss_inverse_motion = masked_mean(F.smooth_l1_loss(
            F.normalize(reconstructedVisualState["MotionPred"], dim=-1, eps=1e-6),
            F.normalize(target_motion, dim=-1, eps=1e-6),
            reduction="none").mean(dim=-1), base_sample)

        loss_inverse_total = (
            loss_inverse_object
            + loss_inverse_slot
            + 0.5 * loss_inverse_relation
            + 0.25 * loss_inverse_presence
            + 0.5 * loss_inverse_scene
            + 0.25 * loss_inverse_summary
            + 0.25 * loss_inverse_motion)

        return {
            "loss_pred_inverse_object": loss_inverse_object,
            "loss_pred_inverse_slot": loss_inverse_slot,
            "loss_pred_inverse_relation": loss_inverse_relation,
            "loss_pred_inverse_presence": loss_inverse_presence,
            "loss_pred_inverse_scene": loss_inverse_scene,
            "loss_pred_inverse_summary": loss_inverse_summary,
            "loss_pred_inverse_motion": loss_inverse_motion,
            "loss_pred_inverse_total": loss_inverse_total,}

class S4DCell(AGICoreModule):
    def __init__(self, inDim: int, deterDim: int, ssmDim: int = 512, dt: float = 1.0, dropout: float = 0.0, ffnMult: int = 4):
        super().__init__()
        self.in_dim = int(inDim)
        self.deter_dim = int(deterDim)
        self.ssm_dim = int(ssmDim)
        self.dt = float(dt)

        self.theta = nn.Parameter(torch.randn(self.ssm_dim) * 0.1)

        self.in_to_ssm = GrowableLoRALinear(nn.Linear(self.in_dim, self.ssm_dim, bias=True))
        self.ssm_to_deter = GrowableLoRALinear(nn.Linear(self.ssm_dim, self.deter_dim, bias=True))
        self.in_to_deter = GrowableLoRALinear(nn.Linear(self.in_dim, self.deter_dim, bias=True))
        self.gate = GrowableLoRALinear(nn.Linear(self.in_dim, self.ssm_dim, bias=True))
        self.out_gate = GrowableLoRALinear(nn.Linear(self.ssm_dim, self.deter_dim, bias=True))

        self.ln_y = nn.LayerNorm(self.deter_dim)
        self.ln_ffn = nn.LayerNorm(self.deter_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.deter_dim, ffnMult * self.deter_dim, bias=True),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffnMult * self.deter_dim, self.deter_dim, bias=True),)

        self.register_buffer("x", torch.zeros(1, self.ssm_dim), persistent=True)

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        B = int(B)
        if self.x.size(0) != B or self.x.device != device or self.x.dtype != dtype:
            self.x = torch.zeros(int(B), self.ssm_dim, device=device, dtype=dtype)

    def CayleyStep(self, aDiag: torch.Tensor, x: torch.Tensor, Bu: torch.Tensor, dt: float):
        A = -F.softplus(aDiag)
        k = 0.5 * dt * A
        num = (1 + k) * x + dt * Bu
        denom = (1 - k).clamp_min(1e-6)
        return num / denom

    def ResetState(self, batch):
        self.x = torch.zeros(batch, self.ssm_dim, device=self.device, dtype=self.dtype)

    def Step(self, zPrev: torch.Tensor, action: torch.Tensor, *, updateState: bool = True) -> torch.Tensor:
        u = torch.cat([zPrev, action], dim=-1)
        g = torch.sigmoid(self.gate(u))
        Bu = self.in_to_ssm(u) * g

        x_next = self.CayleyStep(self.theta, self.x, Bu, self.dt)
        y_lin = self.ssm_to_deter(x_next) + self.in_to_deter(u)
        y_glu = y_lin * torch.sigmoid(self.out_gate(x_next))
        y = self.ln_y(y_glu)
        y = y + self.ffn(self.ln_ffn(y))

        if updateState:
            self.x = x_next.detach()
        return y # [B, D] deterministic state

    def StepWithX(self, zPrev: torch.Tensor, action: torch.Tensor, x: torch.Tensor): # zPrev: stochastic state
        u = torch.cat([zPrev, action], dim=-1)
        g = torch.sigmoid(self.gate(u))
        Bu = self.in_to_ssm(u) * g

        x_next = self.CayleyStep(self.theta, x, Bu, self.dt)
        y_lin = self.ssm_to_deter(x_next) + self.in_to_deter(u)
        y_glu = y_lin * torch.sigmoid(self.out_gate(x_next))
        y = self.ln_y(y_glu)
        y = y + self.ffn(self.ln_ffn(y))

        return y, x_next.detach() # y: [B, D] deterministic state
    


class PhysRefinerHead(AGICoreModule):
    def __init__(
        self,
        deterDim: int,
        actDim: int,
        projDim: int = 256,
        hidden: int = 512,
        dt: float = 1.0,
        substeps: int = 2,
        lambdaWorkCons: float = 0.10, 
        lambdaForceSmooth: float = 0.05, 
        lambdaDelta: float = 0.01, 
        clampResidualRatio: float = 0.50,  
        dampP: float = 0.00, ):
        super().__init__()
        self.D = int(deterDim)
        self.A = int(actDim)
        self.P = int(projDim)
        assert self.P % 2 == 0, f"projDim must be even, got {self.P}"
        self.Q = self.P // 2

        self.dt = float(dt)
        self.substeps = int(max(1, substeps))
        self.clamp_ratio = float(clampResidualRatio)
        self.dampP = float(dampP)

        self.l_work = float(lambdaWorkCons)
        self.l_smooth = float(lambdaForceSmooth)
        self.l_delta = float(lambdaDelta)

        self.to_qp = GrowableLoRALinear(nn.Linear(self.D, self.P, bias=True))
        self.from_qp = GrowableLoRALinear(nn.Linear(self.P, self.D, bias=True))

        self.H_net = nn.Sequential(
            GrowableLoRALinear(nn.Linear(self.P, hidden, bias=True)),
            nn.Softplus(),
            GrowableLoRALinear(nn.Linear(hidden, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, 1, bias=True)),)

        self.force_net = nn.Sequential(
            GrowableLoRALinear(nn.Linear(self.D + self.A, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.Q, bias=True)),)

        self.g_force = GrowableLoRALinear(nn.Linear(self.D + self.A, self.Q, bias=True)) 
        self.g_phys  = GrowableLoRALinear(nn.Linear(self.D + self.A, self.D, bias=True)) 

        self.g_fuse = GrowableLoRALinear(nn.Linear(self.D + self.A + self.D, self.D, bias=True))

    def HAndGrad(self, qp: torch.Tensor, create_graph: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        H = self.H_net(qp) # [B,1]
        g = torch.autograd.grad(
            H.sum(), qp,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=False,)[0] # [B,P]
        return H, g

    def SymplecticLeapfrog(self, q: torch.Tensor, p: torch.Tensor, dt: float, create_graph: bool
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qp0 = torch.cat([q, p], dim=-1)
        H0, g0 = self.HAndGrad(qp0, create_graph=create_graph)
        dH_dq0, _ = g0.chunk(2, dim=-1)

        p_half = p - 0.5 * dt * dH_dq0

        qp_mid = torch.cat([q, p_half], dim=-1)
        _, gm = self.HAndGrad(qp_mid, create_graph=create_graph)
        _, dH_dp_mid = gm.chunk(2, dim=-1)

        q1 = q + dt * dH_dp_mid

        qp_for_p = torch.cat([q1, p_half], dim=-1)
        H1, g2 = self.HAndGrad(qp_for_p, create_graph=create_graph)
        dH_dq2, _ = g2.chunk(2, dim=-1)

        p1 = p_half - 0.5 * dt * dH_dq2
        return q1, p1, H0, H1, dH_dp_mid 

    def ClampResidual(self, delta: torch.Tensor, base: torch.Tensor, ratio: float) -> torch.Tensor:
        eps = 1e-8
        dnorm = delta.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        bnorm = base.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-3
        maxn = ratio * bnorm + eps
        scale = (maxn / dnorm).clamp(max=1.0)
        return delta * scale

    def forward(
        self,
        hPrev: torch.Tensor,
        action: torch.Tensor,
        hS4: torch.Tensor,):

        training_mode = bool(self.training and torch.is_grad_enabled())
        inference_mode_active = torch.is_inference_mode_enabled()

        create_graph = bool(training_mode)

        dt_sub = self.dt / float(self.substeps)

        if training_mode:
            H_start = None
            H_end = None
            work_acc = hPrev.new_zeros(hPrev.size(0), 1) # [B,1]
            smooth_acc = hPrev.new_tensor(0.0)

        with torch.inference_mode(False), torch.enable_grad():
            h_prev_work = hPrev.detach().clone() if inference_mode_active else hPrev
            action_work = action.detach().clone() if inference_mode_active else action
            qp = self.to_qp(h_prev_work)
            if not qp.requires_grad:
                qp = qp.detach().requires_grad_(True)
            q, p = qp.chunk(2, dim=-1)

            for i in range(self.substeps):
                h_cur = self.from_qp(torch.cat([q, p], dim=-1))

                fa0_inp = torch.cat([h_cur, action_work], dim=-1)
                F0 = self.force_net(fa0_inp) * torch.sigmoid(self.g_force(fa0_inp)) # [B,Q]

                if self.dampP > 0.0:
                    p = p * p.new_tensor(-self.dampP * dt_sub).exp()

                p = p + 0.5 * dt_sub * F0

                q, p, H0, H1, dH_dp_mid = self.SymplecticLeapfrog(q, p, dt_sub, create_graph=create_graph)

                if training_mode:
                    if i == 0:
                        H_start = H0
                    H_end = H1

                h_mid = self.from_qp(torch.cat([q, p], dim=-1))
                fa1_inp = torch.cat([h_mid, action_work], dim=-1)
                F1 = self.force_net(fa1_inp) * torch.sigmoid(self.g_force(fa1_inp)) # [B,Q]

                p = p + 0.5 * dt_sub * F1

                if training_mode:
                    smooth_acc = smooth_acc + (F1 - F0).pow(2).mean()
                    F_avg = 0.5 * (F0 + F1)
                    work_acc = work_acc + (dH_dp_mid * F_avg).sum(dim=-1, keepdim=True) * dt_sub

        h_phys_raw = self.from_qp(torch.cat([q, p], dim=-1))

        d_corr = h_phys_raw - hS4 # [B,D]

        gph = torch.sigmoid(self.g_phys(torch.cat([hPrev, action], dim=-1))) # [B,D]
        d_corr = d_corr * gph

        base = hS4 - hPrev # [B,D]  
        d_corr = self.ClampResidual(d_corr, base, ratio=self.clamp_ratio)

        alpha = torch.sigmoid(self.g_fuse(torch.cat([hPrev, action, hS4], dim=-1))) # [B,D]
        h_fused = hS4 + alpha * d_corr

        if not training_mode:
            return h_fused, None, None

        if (H_start is None) or (H_end is None):
            e_work = hPrev.new_tensor(0.0)
        else:
            denom = H_start.detach().abs().mean().clamp_min(1e-6)
            dH = (H_end - H_start)
            e_work = ((dH - work_acc) / denom).pow(2).mean()

        e_smooth = smooth_acc / float(self.substeps)
        e_delta = d_corr.pow(2).mean()

        loss = (
            self.l_work * e_work
            + self.l_smooth * e_smooth
            + self.l_delta * e_delta)

        aux: Dict[str, torch.Tensor] = {}
        aux = {
            "L_work": e_work.detach(),
            "L_smooth": e_smooth.detach(),
            "L_delta": e_delta.detach(),}

        return h_fused, loss, aux


class NeSyHead(AGICoreModule):
    def __init__(
        self,
        inDim: int,
        K: int,
        hidden: int = 1024,
        experts: int = 8,
        dropout: float = 0.1,
        *,
        temperature: float = 1.0,
        noisyGating: bool = True,
        noiseStd: float = 0.1,
        expertDropout: float = 0.0,):
        super().__init__()
        self.K = int(K)
        self.E = int(experts)

        self.temperature = float(max(1e-6, temperature))
        self.noisyGating = bool(noisyGating)
        self.noiseStd = float(noiseStd)
        self.expertDropout = float(expertDropout)

        self.input_ln = nn.LayerNorm(inDim)

        self.gate = nn.Linear(inDim, self.E)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(inDim, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, self.K),)for _ in range(self.E)])

        self.out_scale_log = nn.Parameter(torch.tensor(0.0))

        self.register_buffer("aux_loss", torch.zeros(()), persistent=False)

    def GetAuxLoss(self) -> torch.Tensor:
        return self.aux_loss

    def GateWeights(
        self,
        x_aligned: torch.Tensor,
        *,
        deterministic: bool = False,
        updateAux: bool = True,
        ) -> torch.Tensor:
        logits = self.gate(x_aligned) # [B,E]

        if self.training and not deterministic and self.noisyGating and (self.noiseStd > 0.0):
            logits = logits + torch.randn_like(logits) * self.noiseStd

        if self.training and not deterministic and (self.expertDropout > 0.0):
            keep = (torch.rand_like(logits) > self.expertDropout)
            all_drop = (~keep).all(dim=-1)
            if all_drop.any():
                rand_idx = torch.randint(0, self.E, (int(all_drop.sum().item()),), device=self.device)
                keep[all_drop] = False
                keep[all_drop, rand_idx] = True
            logits = logits.masked_fill(~keep, -1e9)

        w = F.softmax((logits / self.temperature).float(), dim=-1) # [B,E]

        if updateAux and self.training and not deterministic:
            importance = w.mean(dim=0) # [E]
            self.aux_loss = float(self.E) * (importance.pow(2).sum())
        elif updateAux:
            self.aux_loss = x_aligned.new_zeros(())

        return w

    def forward(
        self,
        x: torch.Tensor,
        *,
        deterministic: bool = False,
        updateAux: bool = True,
        ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x_aligned = self.input_ln(x) # [B,inDim]
        w = self.GateWeights(
            x_aligned,
            deterministic=deterministic,
            updateAux=updateAux) # [B,E]

        if deterministic:
            expert_outputs = []
            for expert in self.experts:
                hidden = expert[0](x_aligned)
                hidden = expert[1](hidden)
                hidden = F.dropout(hidden, p=float(expert[2].p), training=False)
                hidden = expert[3](hidden)
                hidden = expert[4](hidden)
                hidden = F.dropout(hidden, p=float(expert[5].p), training=False)
                expert_outputs.append(expert[6](hidden))
            expert_logits = torch.stack(expert_outputs, dim=1)
        else:
            expert_logits = torch.stack([e(x_aligned) for e in self.experts], dim=1)

        out_logits = (w.unsqueeze(-1) * expert_logits).sum(dim=1) # [B,K]

        scale = torch.exp(self.out_scale_log).clamp(1e-3, 100.0)
        out_logits = out_logits * scale

        return out_logits # [B,K]



class GeometricLinear(AGICoreModule):
    def __init__(self, inFeatures, outFeatures, wrapLinear=None, gain=0.1):
        super().__init__()
        lin = nn.Linear(inFeatures, outFeatures, bias=True)
        nn.init.orthogonal_(lin.weight, gain=gain)
        nn.init.zeros_(lin.bias)
        self.linear = wrapLinear(lin) if wrapLinear is not None else lin

    def forward(self, x): return self.linear(x)

class FilmResidual(AGICoreModule):
    def __init__(self, hidden, alpha=0.1, wrapLinear=None):
        super().__init__()
        self.alpha = float(alpha)
        self.ln = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            GeometricLinear(hidden, hidden, wrapLinear),
            nn.GELU(),
            nn.LayerNorm(hidden),
            GeometricLinear(hidden, hidden, wrapLinear),)
        
    def forward(self, h, gx, bx):
        y = (1.0 + gx) * h + bx
        y = self.ln(y)
        y = self.ff(y)
        return h + self.alpha * y

class ConnNet(AGICoreModule):
    def __init__(self,
        stateDim: int,
        actDim: int,
        *,
        hidden: int = 512,
        numBlocks: int = 3,
        rank: int = 8,
        useFull: bool = True,
        useLowrank: bool = True,
        dt: float = 1.0,
        lambdaFro: float = 1e-4,
        lambdaL1: float = 1e-5,
        lambdaSmooth: float = 3e-5,
        normClip: float = 0.8,
        wrapLinear=None):
        super().__init__()
        self.S = int(stateDim)
        self.A = int(actDim)
        self.H = int(hidden)
        self.r = int(rank)
        self.use_full = bool(useFull)
        self.use_lowrank = bool(useLowrank)
        self.dt = float(dt)
        self.lambda_fro = float(lambdaFro)
        self.lambda_l1 = float(lambdaL1)
        self.lambda_smooth = float(lambdaSmooth)
        self.norm_clip = float(normClip)

        self.enc_s = nn.Sequential(
            nn.LayerNorm(self.S),
            GeometricLinear(self.S, self.H, wrapLinear),
            nn.GELU(),)
        
        self.enc_a = nn.Sequential(
            nn.LayerNorm(self.A),
            GeometricLinear(self.A, self.H, wrapLinear),
            nn.GELU(),)

        self.film_gamma_a = GeometricLinear(self.H, self.H, wrapLinear)
        self.film_beta_a = GeometricLinear(self.H, self.H, wrapLinear)

        self.blocks = nn.ModuleList([FilmResidual(self.H, alpha=0.1, wrapLinear=wrapLinear) for _ in range(numBlocks)])

        self.head_uv = GeometricLinear(self.H, 2 * self.S * self.r, wrapLinear)
        self.head_full = GeometricLinear(self.H, self.S * self.S, wrapLinear)
        nBranches = int(self.use_lowrank) + int(self.use_full)
        self.mix = GeometricLinear(self.H, max(1, nBranches), wrapLinear)

    def BuildLowrank(self, h):
        uv = self.head_uv(h)
        U, V = uv.split(self.S * self.r, dim=-1)
        U = U.view(-1, self.S, self.r)
        V = V.view(-1, self.S, self.r)
        return U @ V.transpose(1, 2) - V @ U.transpose(1, 2)

    def BuildFull(self, h):
        M = self.head_full(h).view(-1, self.S, self.S)
        return 0.5 * (M - M.transpose(1, 2))

    def TransportApply(self, A: torch.Tensor, sBase: torch.Tensor) -> torch.Tensor:
        B, S = A.size(0), A.size(1)
        dt = self.dt

        I = torch.eye(S, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, S, S)
        lhs = I - 0.5 * dt * A
        rhs_vec = torch.einsum("bij,bj->bi", I + 0.5 * dt * A, sBase)

        lhs = lhs.contiguous()
        rhs = rhs_vec.unsqueeze(-1).contiguous()

        solve_dtype = (
            torch.float32
            if lhs.device.type == "cpu" and lhs.dtype in (torch.float16, torch.bfloat16)
            else lhs.dtype)
        cayley = torch.linalg.solve(
            lhs.to(dtype=solve_dtype),
            rhs.to(dtype=solve_dtype)).squeeze(-1).to(dtype=sBase.dtype)

        return cayley # [B, S]


    def ComputeGeomReg(self, A, prevA=None):
        reg = self.lambda_fro * A.pow(2).mean()
        if self.use_full and self.lambda_l1 > 0:
            reg = reg + self.lambda_l1 * A.abs().mean()
        if (prevA is not None) and (self.lambda_smooth > 0):
            reg = reg + self.lambda_smooth * (A - prevA).pow(2).mean()
        return reg

    def forward(self, sBase: torch.Tensor, actPrev: torch.Tensor) -> torch.Tensor:
        B = sBase.size(0)
        hs = self.enc_s(sBase) 
        ha = self.enc_a(actPrev) 

        g = torch.tanh(self.film_gamma_a(ha))
        b = self.film_beta_a(ha) 

        h = hs
        for blk in self.blocks:
            h = blk(h, g, b) 

        A_list = []
        if self.use_lowrank:
            A_list.append(self.BuildLowrank(h))
        if self.use_full:
            A_list.append(self.BuildFull(h))

        if not A_list:
            A = torch.zeros(B, self.S, self.S, device=sBase.device, dtype=sBase.dtype)
        elif len(A_list) == 1:
            A = A_list[0]
        else:
            w = F.softmax(self.mix(h), dim=-1) 
            A = w[:, :1].view(B, 1, 1) * A_list[0] + w[:, 1:2].view(B, 1, 1) * A_list[1]

        if self.norm_clip and self.norm_clip > 0:
            fro = A.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), self.norm_clip / fro).view(B, 1, 1)
            A = A * scale
        return A



class SoftNeSyStructure(AGICoreModule):
    def __init__(self, k: int, gExcl: int = 8, gAlo: int = 8, tauInit: float = 1.0, lambdaDag: float = 1e-3):
        super().__init__()
        self.K = int(k)
        self.Ge = int(gExcl)
        self.Ga = int(gAlo)
        self.lambda_dag = float(lambdaDag)
        self.tau = nn.Parameter(torch.tensor(float(tauInit)))
        self.M_excl = nn.Parameter(torch.randn(self.Ge, self.K) * 0.01)
        self.M_alo = nn.Parameter(torch.randn(self.Ga, self.K) * 0.01)
        self.E = nn.Parameter(torch.zeros(self.K, self.K))
        self.register_buffer("_eye", torch.eye(self.K))

    def MixExclusive(self, P: torch.Tensor, temp: float) -> torch.Tensor:
        eps = 1e-6
        Wg = F.softmax(self.M_excl, dim=-1) # [Ge, K]
        logP = torch.log(P.clamp(eps, 1 - eps)) / max(1e-6, temp) # [B, K]
        g = logP.unsqueeze(1) + torch.log(Wg.unsqueeze(0).clamp(eps)) # [B, Ge, K]
        g_sm = F.softmax(g, dim=-1) # [B, Ge, K]
        Wk = F.softmax(self.M_excl.t(), dim=-1)
        P_new = torch.einsum("bgk,kg->bk", g_sm, Wk)
        return P_new # [B, K]

    def EnforceAlo(self, P: torch.Tensor, tau: float) -> torch.Tensor:
        eps = 1e-6
        Wa = F.softmax(self.M_alo, dim=-1)
        group_vals = (P.unsqueeze(1) * Wa.unsqueeze(0)).max(dim=-1).values
        scale = torch.where(group_vals < tau, tau / (group_vals + eps), torch.ones_like(group_vals)) # [B, Ga]
        P_scaled = P.clone()
        Wk = F.softmax(self.M_alo.t(), dim=-1)
        s = torch.einsum("bg,kg->bk", scale, Wk)
        P_scaled = P_scaled * s
        return P_scaled # [B, K]

    def ApplyImplications(self, P: torch.Tensor, alpha: float) -> torch.Tensor:
        eps = 1e-6
        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        contrib = P.unsqueeze(2) * W.unsqueeze(0) # [B, K, K]
        implied = contrib.max(dim=1).values # [B, K]
        Q = torch.maximum(P, alpha * implied).clamp(eps, 1 - eps)
        return Q # [B, K]

    def ProjectTrain(self, P: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        eps = 1e-6
        t = float(max(1e-3, temp))

        soft = max(1e-3, 0.25 * t)

        P1 = self.MixExclusive(P, temp=t).clamp(eps, 1.0 - eps) # [B,K]

        aloTau = P1.new_tensor(0.60)   
        Wa = F.softmax(self.M_alo, dim=-1) # [Ga,K]

        v = (P1.unsqueeze(1) * Wa.unsqueeze(0)) # [B,Ga,K]
        attn = F.softmax(v / soft, dim=-1) # [B,Ga,K]
        group_vals = (attn * v).sum(dim=-1).clamp_min(eps) # [B,Ga]  

        deficiency = F.softplus((aloTau - group_vals) / soft) * soft # [B,Ga] 

        scale = 1.0 + deficiency / group_vals # [B,Ga]
        scale = scale.clamp(1.0, 10.0)     

        Wk = F.softmax(self.M_alo.t(), dim=-1) # [K,Ga]
        s = torch.einsum("bg,kg->bk", scale, Wk) # [B,K]
        P2 = (P1 * s).clamp(eps, 1.0 - eps) # [B,K]

        implAlpha = P2.new_tensor(1.0)   
        W = torch.sigmoid(self.E) * (1.0 - self._eye) # [K,K]

        contrib = P2.unsqueeze(2) * W.unsqueeze(0) # [B,K,K]  
        w_imp = F.softmax(contrib / soft, dim=1)  
        implied = (w_imp * contrib).sum(dim=1) # [B,K]  

        b = (implAlpha * implied).clamp(eps, 1.0 - eps) # [B,K]

        smax = torch.sigmoid((b - P2) / soft) # [B,K]
        Q = (1.0 - smax) * P2 + smax * b
        Q = Q.clamp(eps, 1.0 - eps)

        return Q # [B,K]

    @torch.no_grad()
    def ProjectRuntime(self, P: torch.Tensor, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        Q = self.MixExclusive(P, temp)
        Q = self.EnforceAlo(Q, aloTau)
        Q = self.ApplyImplications(Q, implAlpha)

        Ge = F.softmax(self.M_excl, dim=-1) # [Ge, K]
        gprob = F.softmax((torch.log(Q)).unsqueeze(1) + torch.log(Ge.unsqueeze(0)), dim=-1) # [B, Ge, K]
        excl_pen = 0.5 * ((gprob.sum(-1) ** 2) - (gprob ** 2).sum(-1)).mean(dim=-1)

        Ga = F.softmax(self.M_alo, dim=-1) # [Ga, K]
        alo_val = (Q.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values # [B, Ga]
        alo_pen = F.relu(aloTau - alo_val).mean(dim=-1) # [B]

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl_pen = (W.unsqueeze(0) * F.relu(Q.unsqueeze(2) - Q.unsqueeze(1))).mean(dim=(1,2))

        pen = excl_pen + alo_pen + impl_pen
        pen = (1.0 - torch.exp(-pen)).clamp(0.0, 1.0)
        return Q.clamp(1e-6, 1.0 - 1e-6), pen

    def LogicLosses(self, P: torch.Tensor, lambdaExcl: float, lambdaAlo: float, lambdaImpl: float, aloTau: float = 0.60):
        Ge = F.softmax(self.M_excl, dim=-1)
        g = (torch.log(P.clamp(1e-6, 1-1e-6))).unsqueeze(1) + torch.log(Ge.unsqueeze(0).clamp(1e-6)) # [B, Ge, K]
        g_sm = F.softmax(g, dim=-1) # [B, Ge, K]
        excl = 0.5 * ((g_sm.sum(-1)**2) - (g_sm**2).sum(-1)) # [B, Ge]
        excl = excl.mean()

        Ga = F.softmax(self.M_alo, dim=-1) # [Ga, K]
        top1 = (P.unsqueeze(1) * Ga.unsqueeze(0)).max(-1).values # [B, Ga]
        aloTau_t = top1.new_tensor(float(aloTau))
        alo = (F.relu(aloTau_t - top1) ** 2).mean()

        W = torch.sigmoid(self.E) * (1.0 - self._eye)
        impl = (W.unsqueeze(0) * F.relu(P.unsqueeze(2) - P.unsqueeze(1))).mean()

        loss = lambdaExcl * excl + lambdaAlo * alo + lambdaImpl * impl

        reg = 1e-4 * W.mean()

        Ge_sm = F.softmax(self.M_excl, dim=-1).clamp_min(1e-6)
        Ga_sm = F.softmax(self.M_alo,  dim=-1).clamp_min(1e-6)
        reg = reg + 1e-3 * (
            (Ge_sm * torch.log(Ge_sm)).sum() / float(self.Ge) +
            (Ga_sm * torch.log(Ga_sm)).sum() / float(self.Ga))

        A = (W * W) / float(self.K)  # [K,K]
        dag = torch.trace(torch.matrix_exp(A.float())) - float(self.K)
        dag = dag.to(dtype=P.dtype, device=P.device)
        reg = reg + self.lambda_dag * dag

        loss = loss + reg
        stats = {"excl": excl.detach(), "alo": alo.detach(), "impl": impl.detach()}
        return loss, stats


class FiLMHResidual(AGICoreModule):
    def __init__(
        self,
        baseDim: int,
        rediusDim: int,  
        hidden: int = 512,
        dropout: float = 0.1,
        filmScale: float = 0.10,   
        outLayerNorm: bool = True,):
        super().__init__()
        self.D = int(baseDim)
        self.Z = int(rediusDim)
        self.H = int(hidden)
        self.film_scale = float(filmScale)
        self.use_out_ln = bool(outLayerNorm)

        self.ln_h = nn.LayerNorm(self.D)
        self.ln_e = nn.LayerNorm(self.Z)

        self.e_to_gb = GrowableLoRALinear(nn.Linear(self.Z, 2 * self.D, bias=True))

        self.e_to_h = GrowableLoRALinear(nn.Linear(self.Z, self.D, bias=True))

        self.delta_ln = nn.LayerNorm(4 * self.D)
        self.delta_mlp = nn.Sequential(
            GrowableLoRALinear(nn.Linear(4 * self.D, self.H, bias=True)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            GrowableLoRALinear(nn.Linear(self.H, self.D, bias=True)),)

        self.to_gate = GrowableLoRALinear(nn.Linear(2 * self.D, self.D, bias=True))

        self.out_ln = nn.LayerNorm(self.D)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h0 = self.ln_h(h)  # [B,D]
        e0 = self.ln_e(e)  # [B,Z]

        gamma, beta = self.e_to_gb(e0).chunk(2, dim=-1) # [B,D],[B,D]
        gamma = self.film_scale * torch.tanh(gamma) # [B,D]
        beta  = self.film_scale * torch.tanh(beta) # [B,D]

        h_film = (1.0 + gamma) * h0 + beta # [B,D]

        e_h = self.e_to_h(e0) # [B,D]
        e_h = self.film_scale * torch.tanh(e_h) # [B,D] 

        feat = torch.cat([h_film, e_h, h_film * e_h, h_film - e_h], dim=-1) # [B,4D]
        feat = self.delta_ln(feat) # [B,4D]
        delta = self.delta_mlp(feat) # [B,D]

        gate_in = torch.cat([h_film, e_h], dim=-1) # [B,2D]
        gate = torch.sigmoid(self.to_gate(gate_in)) # [B,D]

        h_out = h + gate * delta # [B,D]

        if self.use_out_ln:
            h_out = self.out_ln(h_out) # [B,D]
        return h_out

class KeyEmbed(AGICoreModule):
    def __init__(self, Z: int, keyDim: int = 256, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.Z = int(Z)
        self.keyDim = int(keyDim)
        H = int(hidden)

        self.ln_e = nn.LayerNorm(self.Z)
        self.ln_a = nn.LayerNorm(self.Z)

        self.to_gb = GrowableLoRALinear(nn.Linear(self.Z, 2 * self.Z, bias=True))

        self.ln_feat = nn.LayerNorm(4 * self.Z)
        self.mlp1 = GrowableLoRALinear(nn.Linear(4 * self.Z, H, bias=True))
        self.mlp2 = GrowableLoRALinear(nn.Linear(H, self.keyDim, bias=True))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, base: torch.Tensor, actionEmbed: torch.Tensor) -> torch.Tensor:
        e = self.ln_e(base) # [B,Z]
        a = self.ln_a(actionEmbed) # [B,Z]

        gamma, beta = self.to_gb(a).chunk(2, dim=-1) # [B,Z],[B,Z]
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)

        e_film = (1.0 + gamma) * e + beta # [B,Z]

        feat = torch.cat([e_film, a, e_film * a, e_film - a], dim=-1) # [B,4Z]
        feat = self.ln_feat(feat)

        h = F.silu(self.mlp1(feat)) # [B,H]
        h = self.drop(h)
        k = self.mlp2(h) # [B,keyDim]

        k = F.normalize(k, dim=-1, eps=1e-6) # [B,keyDim]
        return k


class RobotWorldRelationEncoder(AGICoreModule):
    def __init__(
        self,
        robotDim: int = ModuleDim.PstSlotDim,
        actionDim: int = ModuleDim.DecisionFeedbackEmbedDim,
        slotDim: int = ModuleDim.PstSlotDim,
        poseDim: int = ModuleDim.PstPoseDim,
        attrDim: int = ModuleDim.PstAttrDim,
        relDim: int = 36,
        affordanceDim: int = ModuleDim.PstAffordanceDim,
        relationClasses: int = ModuleDim.PstRelationClasses,
        stateDim: int = ModuleDim.PstStateDim,
        outputDim: int = ModuleDim.PstSlotDim,
        hidden: int = 256,
        pairChunkSize: int = 16,):
        super().__init__()
        if int(poseDim) < 7:
            raise ValueError(f"poseDim must contain xyz + quaternion (at least 7), got {poseDim}")
        if int(relDim) != int(relationClasses) + 4:
            raise ValueError(
                f"relDim must be 4 geometry values + relationClasses, got "
                f"relDim={relDim}, relationClasses={relationClasses}")

        self.robot_dim = int(robotDim)
        self.action_dim = int(actionDim)
        self.pose_dim = int(poseDim)
        self.relation_dim = int(relDim)
        self.output_dim = int(outputDim)
        self.pair_chunk_size = max(1, int(pairChunkSize))
        self._slot_field_dims = {
            "SlotState": int(slotDim),
            "PoseWorld": self.pose_dim,
            "ARaw": int(attrDim),
            "Size": 3,
            "StateRaw": int(stateDim),
            "AffordanceRaw": int(affordanceDim),
            "MotionRaw": self.pose_dim,
            "ExternalRelationProbRaw": int(relationClasses),
            "ContactForceRaw": 2,
        }
        slot_input_dim = (
            int(slotDim)
            + int(poseDim)
            + int(attrDim)
            + int(stateDim)
            + int(affordanceDim)
            + int(poseDim)
            + int(relationClasses)
            + 14)

        self.robot_action_proj = nn.Sequential(
            nn.LayerNorm(int(robotDim) + int(actionDim)),
            GrowableLoRALinear(nn.Linear(int(robotDim) + int(actionDim), hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),
            nn.LayerNorm(self.output_dim),)

        self.slot_proj = nn.Sequential(
            nn.LayerNorm(slot_input_dim),
            GrowableLoRALinear(nn.Linear(slot_input_dim, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),
            nn.LayerNorm(self.output_dim),)

        self.pair_geometry_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(4, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),)

        self.pair_relation_prob_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(int(relDim) - 4, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),)

        self.pair_relation_fuser = nn.Sequential(
            nn.LayerNorm(self.output_dim * 3),
            GrowableLoRALinear(nn.Linear(self.output_dim * 3, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),
            nn.LayerNorm(self.output_dim),)

        self.pair_message_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim * 3),
            GrowableLoRALinear(nn.Linear(self.output_dim * 3, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),
            nn.LayerNorm(self.output_dim),)

        self.pair_score = GrowableLoRALinear(nn.Linear(self.output_dim, 1, bias=False))

        self.slot_relation_norm = nn.LayerNorm(self.output_dim)

        self.slot_action_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim * 4),
            GrowableLoRALinear(nn.Linear(self.output_dim * 4, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),
            nn.LayerNorm(self.output_dim),)

        self.slot_score = GrowableLoRALinear(nn.Linear(self.output_dim, 1, bias=False))

        self.scene_stats_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(3, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),)

        self.relation_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim * 5),
            GrowableLoRALinear(nn.Linear(self.output_dim * 5, hidden, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(hidden, self.output_dim, bias=True)),
            nn.LayerNorm(self.output_dim),)

    @staticmethod
    def _canonicalize_pose_quaternion(pose: torch.Tensor) -> torch.Tensor:
        quaternion_raw = pose[..., 3:7]
        quaternion = F.normalize(quaternion_raw.float(), dim=-1, eps=1e-6).to(pose.dtype)
        identity = torch.zeros_like(quaternion)
        identity[..., 3] = 1.0
        quaternion = torch.where(
            quaternion_raw.norm(dim=-1, keepdim=True) > 1e-6,
            quaternion,
            identity)
        pivot_index = quaternion.abs().argmax(dim=-1, keepdim=True)
        pivot = quaternion.gather(-1, pivot_index)
        sign = torch.where(pivot < 0.0, -torch.ones_like(pivot), torch.ones_like(pivot))
        return torch.cat([pose[..., :3], quaternion * sign, pose[..., 7:]], dim=-1)

    @staticmethod
    def _masked_confidence_softmax(
        logits: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
        dim: int,
        ) -> torch.Tensor:
        output_dtype = logits.dtype
        work_dtype = (
            torch.float32
            if logits.dtype in (torch.float16, torch.bfloat16)
            else logits.dtype)
        logits = logits.to(dtype=work_dtype)
        confidence = torch.nan_to_num(
            confidence.to(dtype=work_dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0).clamp_min(0.0)
        valid = valid & (confidence > 0.0)
        tiny = torch.finfo(logits.dtype).tiny
        weighted_logits = logits + confidence.clamp_min(tiny).log()
        masked_logits = torch.where(
            valid,
            weighted_logits,
            torch.full_like(weighted_logits, torch.finfo(logits.dtype).min))
        probability = F.softmax(masked_logits, dim=dim)
        probability = torch.where(valid, probability, torch.zeros_like(probability))
        normalizer = probability.sum(dim=dim, keepdim=True)
        probability = torch.where(
            normalizer > 0.0,
            probability / normalizer.clamp_min(tiny),
            torch.zeros_like(probability))
        return probability.to(dtype=output_dtype)

    def _aggregate_pair_chunk(
        self,
        receiver_token: torch.Tensor,
        neighbor_token: torch.Tensor,
        pair_relation: torch.Tensor,
        pair_valid: torch.Tensor,
        pair_confidence: torch.Tensor,
        relation_recency: torch.Tensor,
        robot_action: torch.Tensor,
        ) -> torch.Tensor:
        """Aggregate one receiver chunk; checkpointed by ``forward`` during training."""
        pair_geometry_token = self.pair_geometry_proj(pair_relation[..., :4])
        pair_relation_prob_token = self.pair_relation_prob_proj(pair_relation[..., 4:])
        pair_relation_prob_token = pair_relation_prob_token * relation_recency.unsqueeze(-1)
        pair_token = self.pair_relation_fuser(torch.cat([
            pair_geometry_token,
            pair_relation_prob_token,
            pair_geometry_token * pair_relation_prob_token], dim=-1))

        receiver_count = int(receiver_token.size(1))
        active_count = int(neighbor_token.size(1))
        receiver_expanded = receiver_token.unsqueeze(2).expand(-1, -1, active_count, -1)
        neighbor_expanded = neighbor_token.unsqueeze(1).expand(-1, receiver_count, -1, -1)
        pair_message = self.pair_message_proj(torch.cat([
            receiver_expanded, neighbor_expanded, pair_token], dim=-1))
        pair_query = robot_action.unsqueeze(1).unsqueeze(2)
        pair_logits = self.pair_score(pair_message * pair_query).squeeze(-1)
        pair_prob = self._masked_confidence_softmax(
            pair_logits, pair_confidence, pair_valid, dim=-1)
        return (pair_message * pair_prob.unsqueeze(-1)).sum(dim=2)

    def _validate_inputs(
        self,
        robotSelfState: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        ) -> Tuple[int, int]:
        if not torch.is_tensor(robotSelfState):
            raise TypeError("robotSelfState must be a tensor")
        if not torch.is_tensor(actionEnc):
            raise TypeError("actionEnc must be a tensor")
        if robotSelfState.ndim != 2 or robotSelfState.size(-1) != self.robot_dim:
            raise ValueError(
                f"robotSelfState must have shape [B, {self.robot_dim}], got {tuple(robotSelfState.shape)}")
        if actionEnc.ndim != 2 or actionEnc.size(-1) != self.action_dim:
            raise ValueError(f"actionEnc must have shape [B, {self.action_dim}], got {tuple(actionEnc.shape)}")
        if actionEnc.size(0) != robotSelfState.size(0):
            raise ValueError("robotSelfState and actionEnc batch sizes must match")
        if not isinstance(physicalState, dict):
            raise TypeError("physicalState must be a dictionary of tensors")

        required = set(self._slot_field_dims) | {
            "SlotPresence", "MphysRaw", "ContactProbRaw", "MovingProbRaw",
            "Visibility", "Occlusion", "Observed", "LastSeen", "Step", "PairwiseRelation"}
        missing = sorted(required.difference(physicalState))
        if missing:
            raise KeyError(f"physicalState is missing required fields: {missing}")
        contact_point_keys = tuple(
            key for key in ("ContactPointWorldRaw", "ContactPointWorld", "ContactPointRaw")
            if key in physicalState)
        if not contact_point_keys:
            raise KeyError(
                "physicalState must contain ContactPointWorldRaw, ContactPointWorld, or ContactPointRaw")

        B = int(robotSelfState.size(0))
        slot_presence = physicalState["SlotPresence"]
        if not torch.is_tensor(slot_presence) or slot_presence.ndim != 2 or slot_presence.size(0) != B:
            actual = tuple(slot_presence.shape) if torch.is_tensor(slot_presence) else type(slot_presence).__name__
            raise ValueError(f"SlotPresence must have shape [B, K], got {actual}")
        K = int(slot_presence.size(1))
        if K <= 0:
            raise ValueError("physicalState must contain at least one slot")

        expected_shapes = {
            **{key: (B, K, dim) for key, dim in self._slot_field_dims.items()},
            "SlotPresence": (B, K),
            "MphysRaw": (B, K),
            "ContactProbRaw": (B, K),
            "MovingProbRaw": (B, K),
            "Visibility": (B, K),
            "Occlusion": (B, K),
            "Observed": (B, K),
            "LastSeen": (B, K),
            "Step": (B,),
            "PairwiseRelation": (B, K, K, self.relation_dim),}
        for key in contact_point_keys:
            expected_shapes[key] = (B, K, 3)
        if "PairRelationLastSeen" in physicalState:
            expected_shapes["PairRelationLastSeen"] = (B, K, K)
        for key, expected in expected_shapes.items():
            value = physicalState[key]
            if not torch.is_tensor(value) or tuple(value.shape) != expected:
                actual = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(f"physicalState[{key!r}] must have shape {expected}, got {actual}")
            if value.device != robotSelfState.device:
                raise ValueError(f"physicalState[{key!r}] must be on {robotSelfState.device}, got {value.device}")
        if actionEnc.device != robotSelfState.device:
            raise ValueError(f"actionEnc must be on {robotSelfState.device}, got {actionEnc.device}")
        return B, K

    def forward(
        self,
        robotSelfState: torch.Tensor, # [B, 128]
        physicalState: Dict[str, torch.Tensor], # Dict[str, Tensor]
        actionEnc: torch.Tensor, # [B, 256]
        ) -> torch.Tensor:
        """Encode the supplied relation without shifting the caller's time indices.

        Filtering callers intentionally supply the current measured robot/PST state together
        with the previously executed action feedback. Prospective callers instead supply the
        current state with a candidate action. The same encoder therefore supports both
        current-state estimation and forward rollout.
        """
        B, K = self._validate_inputs(robotSelfState, physicalState, actionEnc)
        robot_action = self.robot_action_proj(torch.cat([robotSelfState, actionEnc], dim=-1))

        confidence_dtype = physicalState["SlotState"].dtype
        slot_presence = physicalState["SlotPresence"].to(dtype=confidence_dtype)
        physical_confidence = physicalState["MphysRaw"].to(dtype=confidence_dtype)
        slot_presence = torch.nan_to_num(slot_presence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        physical_confidence = torch.nan_to_num(
            physical_confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        slot_weight = slot_presence * physical_confidence # [B, K]
        slot_valid = slot_weight > 0.0
        slot_mask = slot_valid.unsqueeze(-1)

        def safe_slot(name: str) -> torch.Tensor:
            value = physicalState[name]
            value_mask = slot_mask if value.ndim == 3 else slot_valid
            return torch.where(value_mask, value, torch.zeros_like(value))

        pose_world = self._canonicalize_pose_quaternion(safe_slot("PoseWorld"))
        motion_raw = self._canonicalize_pose_quaternion(safe_slot("MotionRaw"))
        contact_point_key = next(
            key for key in ("ContactPointWorldRaw", "ContactPointWorld", "ContactPointRaw")
            if key in physicalState)
        age = (physicalState["Step"].unsqueeze(1) - physicalState["LastSeen"]).clamp_min(0)
        recency = torch.where(
            slot_valid,
            1.0 / (1.0 + age.to(dtype=confidence_dtype)),
            torch.zeros_like(slot_weight))
        slot_input = torch.cat([
            safe_slot("SlotState"),
            pose_world,
            safe_slot("ARaw"),
            safe_slot("Size"),
            safe_slot("StateRaw"),
            safe_slot("AffordanceRaw"),
            motion_raw,
            safe_slot("ExternalRelationProbRaw"),
            safe_slot("ContactProbRaw").unsqueeze(-1),
            safe_slot("MovingProbRaw").unsqueeze(-1),
            safe_slot("ContactForceRaw"),
            safe_slot(contact_point_key),
            safe_slot("Visibility").unsqueeze(-1),
            safe_slot("Occlusion").unsqueeze(-1),
            safe_slot("Observed").to(dtype=confidence_dtype).unsqueeze(-1),
            recency.unsqueeze(-1),], dim=-1)
        slot_token = self.slot_proj(slot_input) # [B, K, 128]

        active_indices = slot_valid.any(dim=0).nonzero(as_tuple=False).flatten()
        neighbor_context = torch.zeros_like(slot_token)
        if active_indices.numel() > 0:
            active_slot_token = slot_token.index_select(1, active_indices)
            active_weight = slot_weight.index_select(1, active_indices).to(dtype=slot_token.dtype)
            active_valid = slot_valid.index_select(1, active_indices)
            active_count = int(active_indices.numel())
            context_chunks = []
            for start in range(0, active_count, self.pair_chunk_size):
                end = min(start + self.pair_chunk_size, active_count)
                receiver_indices = active_indices[start:end]
                pair_relation = physicalState["PairwiseRelation"].index_select(
                    1, receiver_indices).index_select(2, active_indices)
                pair_valid = active_valid[:, start:end].unsqueeze(2) & active_valid.unsqueeze(1)
                same_slot = receiver_indices.unsqueeze(1) == active_indices.unsqueeze(0)
                pair_valid = pair_valid & ~same_slot.unsqueeze(0)
                pair_relation = torch.where(
                    pair_valid.unsqueeze(-1), pair_relation, torch.zeros_like(pair_relation))

                relation_seen = pair_relation[..., 4:].abs().sum(dim=-1) > 0.0
                if "PairRelationLastSeen" in physicalState:
                    relation_last_seen = physicalState["PairRelationLastSeen"].index_select(
                        1, receiver_indices).index_select(2, active_indices)
                    current_step = physicalState["Step"].view(B, 1, 1)
                    relation_last_seen = torch.where(
                        pair_valid, relation_last_seen, current_step.expand_as(relation_last_seen))
                    relation_age = (
                        current_step - relation_last_seen).clamp_min(0)
                    relation_decay = torch.exp(
                        -relation_age.to(dtype=slot_token.dtype) / 64.0)
                    relation_recency = torch.where(
                        relation_seen & (relation_last_seen > 0),
                        relation_decay,
                        torch.zeros_like(relation_decay))
                else:
                    relation_recency = relation_seen.to(dtype=slot_token.dtype)
                pair_confidence = (
                    active_weight[:, start:end].unsqueeze(2) * active_weight.unsqueeze(1))
                chunk_inputs = (
                    active_slot_token[:, start:end],
                    active_slot_token,
                    pair_relation,
                    pair_valid,
                    pair_confidence,
                    relation_recency,
                    robot_action)
                if torch.is_grad_enabled():
                    chunk_context = checkpoint(
                        self._aggregate_pair_chunk,
                        *chunk_inputs,
                        use_reentrant=False)
                else:
                    chunk_context = self._aggregate_pair_chunk(*chunk_inputs)
                context_chunks.append(chunk_context)
            neighbor_context = neighbor_context.index_copy(
                1, active_indices, torch.cat(context_chunks, dim=1))

        relational_slot = self.slot_relation_norm(slot_token + neighbor_context)
        robot_query = robot_action.unsqueeze(1).expand(-1, relational_slot.size(1), -1)
        slot_action = self.slot_action_proj(torch.cat([
            relational_slot,
            robot_query,
            relational_slot * robot_query,
            relational_slot - robot_query,], dim=-1))

        slot_logits = self.slot_score(slot_action).squeeze(-1)
        slot_prob = self._masked_confidence_softmax(
            slot_logits, slot_weight.to(dtype=slot_logits.dtype), slot_valid, dim=-1)
        slot_context = (slot_action * slot_prob.unsqueeze(-1)).sum(dim=1)
        slot_spread = ((slot_action - slot_context.unsqueeze(1)).square() * slot_prob.unsqueeze(-1)).sum(dim=1)

        total_confidence = slot_weight.sum(dim=1)
        valid_count = slot_valid.sum(dim=1).to(dtype=total_confidence.dtype)
        mean_confidence = total_confidence / valid_count.clamp_min(1.0)
        scene_stats = torch.stack([
            torch.log1p(total_confidence),
            mean_confidence,
            valid_count / float(K),], dim=-1).to(dtype=slot_context.dtype)
        scene_stats_token = self.scene_stats_proj(scene_stats)
        scene_gate = (1.0 - torch.exp(-total_confidence)).to(dtype=slot_context.dtype).unsqueeze(-1)
        scene_gate = scene_gate * (valid_count > 0.0).to(dtype=slot_context.dtype).unsqueeze(-1)

        relation = self.relation_proj(torch.cat([
            robot_action,
            slot_context,
            slot_spread,
            robot_action * slot_context,
            scene_stats_token,], dim=-1))
        return relation * scene_gate


class RSSMWorldModel(AGICoreModule):
    def __init__(
        self,
        visionDim: int = 1024,
        actionDim: int = 256,
        deterDim: int = 512,
        stochDim: int = 64,
        stateDim: int = 512,
        ssmDim: int = 512,
        useDecoder: bool = True,
        useMemory: bool = True,
        memoryCapacity: int = 16384,
        memoryPath: Optional[str] = None,
        memoryAutosaveEvery: int = 0,
        nsEnabled: bool = True,
        nsLambdaExclusive: float = 1e-2,
        nsLambdaAtLeastOne: float = 1e-2,
        nsLambdaImplication: float = 1e-2,
        memTopK = 4,
        memTemp: float = 1.0,
        globalFeatDim: int = 1024,
        objectTokenDim: int = 512,
        numObjectTokens: int = 16,
        motionPredDim: int = 512,
        integratedFeatDim: int = 1024,
        physicalSlots: int = ModuleDim.PstSlots,
        physicalSlotDim: int = 128,
        physicalPoseDim: int = 7,
        physicalAttrDim: int = 32,
        physicalIdDim: int = 515,
        physicalRelDim: int = 36,
        physicalRelationClasses: int = 32,
        physicalSemanticDim: int = 387,
        physicalStateDim: int = 16,
        physicalAffordanceDim: int = 8,
        physicalTextDim: int = 4,
        physicalSymbolDim: int = 16,
        physicalObservationThreshold: float = 0.5,
        physicalIdentityThreshold: float = 0.75,
        physicalConfidenceDecay: float = 0.995,):
        super().__init__()

        self.vision_dim = visionDim
        self.action_dim = actionDim
        self.deter_dim = deterDim
        self.stoch_dim = stochDim
        self.state_dim = stateDim
        self.use_decoder = useDecoder
        self.ssm_dim = ssmDim
        self.reward_min = -10.0
        self.reward_max = 10.0

        self._mem_topk: int = int(memTopK)
        self._mem_temp: float = float(memTemp)
        self.global_feat_dim = int(globalFeatDim)
        self.object_token_dim = int(objectTokenDim)
        self.num_object_tokens = int(numObjectTokens)
        self.motion_pred_dim = int(motionPredDim)
        self.integrated_feat_dim = int(integratedFeatDim)
        self.physical_slots = int(physicalSlots)
        self.physical_slot_dim = int(physicalSlotDim)
        self.physical_pose_dim = int(physicalPoseDim)
        self.physical_attr_dim = int(physicalAttrDim)
        self.physical_id_dim = int(physicalIdDim)
        self.physical_rel_dim = int(physicalRelDim)
        self.physical_relation_classes = int(physicalRelationClasses)
        self.physical_semantic_dim = int(physicalSemanticDim)
        self.physical_state_dim = int(physicalStateDim)
        self.physical_affordance_dim = int(physicalAffordanceDim)
        self.physical_text_dim = int(physicalTextDim)
        self.physical_symbol_dim = int(physicalSymbolDim)
        self.physical_observation_threshold = float(physicalObservationThreshold)
        self.physical_identity_threshold = float(physicalIdentityThreshold)
        self.physical_confidence_decay = float(physicalConfidenceDecay)
        self.robot_self_dim = ModuleDim.PstSlotDim
        self.robot_world_dim = ModuleDim.PstSlotDim

        self._A_prev = None

        self.obs_enc = nn.Sequential(
            nn.LayerNorm(visionDim),
            GrowableLoRALinear(nn.Linear(visionDim, stateDim, bias=True)),
            nn.GELU(),
            nn.LayerNorm(stateDim),
            GrowableLoRALinear(nn.Linear(stateDim, stochDim, bias=True)),)

        self.act_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(actionDim, stochDim, bias=True)),
            nn.LayerNorm(stochDim),
            nn.Tanh(),)
        
        self.s4 = S4DCell(inDim=stochDim + stochDim, deterDim=deterDim, ssmDim=self.ssm_dim, dt=1.0)

        self.prior_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim, 2 * stochDim, bias=True)))

        self.post_net = nn.Sequential(GrowableLoRALinear(nn.Linear(deterDim + stochDim, 2 * stochDim, bias=True)))
        
        self.state_proj = nn.Sequential(
            nn.LayerNorm(deterDim + stochDim),
            GrowableLoRALinear(nn.Linear(deterDim + stochDim, stateDim, bias=True)),
            nn.LayerNorm(stateDim),)

        self.rdone_ln = nn.LayerNorm(2 * stateDim + stochDim)

        self.rdone_trunk = nn.Sequential(
            GrowableLoRALinear(nn.Linear(2 * stateDim + stochDim, 512, bias=True)),
            nn.SiLU(),
            nn.Dropout(0.1),
            GrowableLoRALinear(nn.Linear(512, 256, bias=True)),
            nn.SiLU(),)

        self.rew_head = nn.Sequential(GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)

        self.done_head = nn.Sequential(GrowableLoRALinear(nn.Linear(256, 1, bias=True)),)
        
        self.obs_dec = nn.Sequential(
            GrowableLoRALinear(nn.Linear(stateDim, stateDim, bias=True)),
            nn.GELU(),
            GrowableLoRALinear(nn.Linear(stateDim, visionDim, bias=True)),)

        self._use_memory = bool(useMemory)
        self._mem_capacity = int(memoryCapacity)
        self._mem_path = memoryPath
        self._mem_autosave_every = int(memoryAutosaveEvery)
        self._mem_add_count = 0

        self.register_buffer("_mem_keys", torch.zeros(1, self._mem_capacity, stochDim))
        self.register_buffer("_mem_vals", torch.zeros(1, self._mem_capacity, stateDim))
        self.register_buffer("_mem_size", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_mem_imp", torch.zeros(1, self._mem_capacity))
        self.register_buffer("_mem_steps", torch.zeros(1, self._mem_capacity, dtype=torch.long))
        self.register_buffer("_mem_global_step", torch.zeros(1, dtype=torch.long))

        self.register_buffer("_pst_slot_state", torch.zeros(1, self.physical_slots, self.physical_slot_dim))
        self.register_buffer("_pst_pose_world", torch.zeros(1, self.physical_slots, self.physical_pose_dim))
        self.register_buffer("_pst_attribute", torch.zeros(1, self.physical_slots, self.physical_attr_dim))
        self.register_buffer("_pst_slot_presence", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_entity_prob", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_identity_key", torch.zeros(1, self.physical_slots, self.physical_id_dim))
        self.register_buffer("_pst_pairwise_relation", torch.zeros(1, self.physical_slots, self.physical_slots, self.physical_rel_dim))
        self.register_buffer(
            "_pst_pair_last_seen",
            torch.zeros(1, self.physical_slots, self.physical_slots, dtype=torch.long))
        self.register_buffer("_pst_external_relation", torch.zeros(1, self.physical_slots, self.physical_relation_classes))
        self.register_buffer("_pst_semantic", torch.zeros(1, self.physical_slots, self.physical_semantic_dim))
        self.register_buffer("_pst_size", torch.zeros(1, self.physical_slots, 3))
        self.register_buffer("_pst_state", torch.zeros(1, self.physical_slots, self.physical_state_dim))
        self.register_buffer("_pst_affordance", torch.zeros(1, self.physical_slots, self.physical_affordance_dim))
        self.register_buffer("_pst_motion", torch.zeros(1, self.physical_slots, self.physical_pose_dim))
        self.register_buffer("_pst_moving", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_contact", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_contact_force", torch.zeros(1, self.physical_slots, 2))
        self.register_buffer("_pst_contact_point", torch.zeros(1, self.physical_slots, 3))
        self.register_buffer("_pst_parent", torch.zeros(1, self.physical_slots, self.physical_slots))
        self.register_buffer("_pst_visibility", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_occlusion", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_has_text", torch.zeros(1, self.physical_slots))
        self.register_buffer("_pst_text", torch.zeros(1, self.physical_slots, self.physical_text_dim))
        self.register_buffer("_pst_symbol", torch.zeros(1, self.physical_slots, self.physical_symbol_dim))
        self.register_buffer("_pst_observed", torch.zeros(1, self.physical_slots, dtype=torch.bool))
        self.register_buffer("_pst_last_seen", torch.zeros(1, self.physical_slots, dtype=torch.long))
        self.register_buffer("_pst_step", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_robot_self_state", torch.zeros(1, self.robot_self_dim))
        self.register_buffer("_robot_action_context", torch.zeros(1, self.action_dim))

        self._mem_imp_lr = 0.10

        self._ns_enabled = bool(nsEnabled)
 
        self._ns_K: int = 128
        self.ns_struct = SoftNeSyStructure(k=self._ns_K, gExcl=30, gAlo=30, tauInit=1.0)

        self.ns_head_prior = NeSyHead(deterDim, self._ns_K, hidden=1024, experts=4)
        self.ns_head_post = NeSyHead(deterDim + stochDim, self._ns_K, hidden=1024, experts=4)

        self.ns_to_delta_mu = nn.Linear(self._ns_K, stochDim)
        self.ns_gate_mu = nn.Linear(deterDim + stochDim, stochDim)
        self.ns_gate_mu_post = nn.Linear(deterDim + 2 * stochDim, stochDim)

        self.key_emb = KeyEmbed(Z=stochDim, keyDim=stochDim)

        self.state_state_film = FiLMHResidual(baseDim=self.state_dim, rediusDim=self.state_dim, hidden=512)

        self.ns_lambda_excl = float(nsLambdaExclusive)
        self.ns_lambda_alo = float(nsLambdaAtLeastOne)
        self.ns_lambda_impl = float(nsLambdaImplication)

        self.ResetState(batchSize=1)

        if self._use_memory and self._mem_path:
            self.LoadMemory(self._mem_path, mapLocation=None, strict=False)

        self.conn = ConnNet(stateDim=stateDim,actDim=stochDim,wrapLinear=GrowableLoRALinear)

        self.phys_refiner = PhysRefinerHead(deterDim=self.deter_dim,actDim=self.stoch_dim)

        self.mix_gate = nn.Sequential(GrowableLoRALinear(nn.Linear(3 * self.state_dim, 3)))
        self.robot_world_relation = RobotWorldRelationEncoder(
            robotDim=self.robot_self_dim,
            actionDim=self.action_dim,
            slotDim=self.physical_slot_dim,
            poseDim=self.physical_pose_dim,
            attrDim=self.physical_attr_dim,
            relDim=self.physical_rel_dim,
            affordanceDim=self.physical_affordance_dim,
            relationClasses=self.physical_relation_classes,
            stateDim=self.physical_state_dim,
            outputDim=self.robot_world_dim)
        self.embodied_action_proj = nn.Sequential(
            nn.LayerNorm(self.action_dim + self.robot_self_dim + self.robot_world_dim),
            GrowableLoRALinear(nn.Linear(self.action_dim + self.robot_self_dim + self.robot_world_dim, self.action_dim, bias=True)),
            nn.SiLU(),
            GrowableLoRALinear(nn.Linear(self.action_dim, self.action_dim, bias=True)),
            nn.LayerNorm(self.action_dim),)

        self.future_action_head = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, max(self.state_dim, self.action_dim)),
            nn.GELU(),
            nn.Linear(max(self.state_dim, self.action_dim), self.action_dim),
            nn.LayerNorm(self.action_dim),)

        self.predicted_visual_head = PredictedVisualHead(
            stateDim=self.state_dim,
            globalFeatDim=self.global_feat_dim,
            objectTokenDim=self.object_token_dim,
            numObjectTokens=self.num_object_tokens,
            motionPredDim=self.motion_pred_dim,
            integratedFeatDim=self.integrated_feat_dim,)

        self.visual_reconstructor = VisualReconstructor(
            globalFeatDim=self.global_feat_dim,
            objectTokenDim=self.object_token_dim,
            numObjectTokens=self.num_object_tokens,
            motionPredDim=self.motion_pred_dim,
            integratedFeatDim=self.integrated_feat_dim,)

        self.pst_binder = PSTWorldBinder(
            hDim=self.deter_dim,
            zDim=self.stoch_dim,
            xDim=self.ssm_dim,
            actionDim=self.action_dim,
            robotWorldDim=self.robot_world_dim,
            slotDim=self.physical_slot_dim,
            idDim=self.physical_id_dim,
            poseDim=self.physical_pose_dim,
            attrDim=self.physical_attr_dim,
            semanticDim=self.physical_semantic_dim,
            stateDim=self.physical_state_dim,
            affordanceDim=self.physical_affordance_dim,
            relDim=self.physical_rel_dim,
            relationClasses=self.physical_relation_classes)

        world_abstract_dim = (
            self.deter_dim
            + self.stoch_dim
            + self.ssm_dim
            + self.state_dim
            + 2 * self.physical_slot_dim
            + self.robot_world_dim
            + 4)

        self.world_abstract_projector = nn.Sequential(
            nn.LayerNorm(world_abstract_dim),
            nn.Linear(world_abstract_dim, self.state_dim),
            nn.SiLU(),
            nn.Linear(self.state_dim, self.state_dim),
            nn.LayerNorm(self.state_dim),)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,):
        timestamp_key = f"{prefix}_pst_pair_last_seen"
        if timestamp_key not in state_dict:
            relation_key = f"{prefix}_pst_pairwise_relation"
            presence_key = f"{prefix}_pst_slot_presence"
            step_key = f"{prefix}_pst_step"
            if all(key in state_dict for key in (relation_key, presence_key, step_key)):
                inferred = self.InferPairRelationLastSeen(
                    state_dict[relation_key],
                    state_dict[presence_key],
                    state_dict[step_key])
                has_relation = inferred.flatten(1).any(dim=1)
                step = state_dict[step_key].clone()
                step = torch.where(has_relation & (step < 1), torch.ones_like(step), step)
                state_dict[step_key] = step
                state_dict[timestamp_key] = self.InferPairRelationLastSeen(
                    state_dict[relation_key],
                    state_dict[presence_key],
                    step)
            else:
                state_dict[timestamp_key] = torch.zeros_like(self._pst_pair_last_seen)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs)

    def EnsurePhysicalMemory(self, B: int, device: torch.device, dtype: torch.dtype):
        if (self._pst_slot_state.size(0) == B
            and self._pst_slot_state.size(1) == self.physical_slots
            and self._pst_slot_state.size(2) == self.physical_slot_dim
            and tuple(self._pst_pair_last_seen.shape[1:]) == (self.physical_slots, self.physical_slots)
            and self._pst_semantic.size(2) == self.physical_semantic_dim
            and self._pst_slot_state.device == device
            and self._pst_slot_state.dtype == dtype):

            return
        K = self.physical_slots
        self._pst_slot_state = torch.zeros(B, K, self.physical_slot_dim, device=device, dtype=dtype)
        self._pst_pose_world = torch.zeros(B, K, self.physical_pose_dim, device=device, dtype=dtype)
        self._pst_attribute = torch.zeros(B, K, self.physical_attr_dim, device=device, dtype=dtype)
        self._pst_slot_presence = torch.zeros(B, K, device=device, dtype=dtype)
        self._pst_entity_prob = torch.zeros(B, K, device=device, dtype=dtype)
        self._pst_identity_key = torch.zeros(B, K, self.physical_id_dim, device=device, dtype=dtype)
        self._pst_pairwise_relation = torch.zeros(B, K, K, self.physical_rel_dim, device=device, dtype=dtype)
        self._pst_pair_last_seen = torch.zeros(B, K, K, device=device, dtype=torch.long)
        self._pst_external_relation = torch.zeros(B, K, self.physical_relation_classes, device=device, dtype=dtype)
        self._pst_semantic = torch.zeros(B, K, self.physical_semantic_dim, device=device, dtype=dtype)
        self._pst_size = torch.zeros(B, K, 3, device=device, dtype=dtype)
        self._pst_state = torch.zeros(B, K, self.physical_state_dim, device=device, dtype=dtype)
        self._pst_affordance = torch.zeros(B, K, self.physical_affordance_dim, device=device, dtype=dtype)
        self._pst_motion = torch.zeros(B, K, self.physical_pose_dim, device=device, dtype=dtype)
        self._pst_moving = torch.zeros(B, K, device=device, dtype=dtype)
        self._pst_contact = torch.zeros(B, K, device=device, dtype=dtype)
        self._pst_contact_force = torch.zeros(B, K, 2, device=device, dtype=dtype)
        self._pst_contact_point = torch.zeros(B, K, 3, device=device, dtype=dtype)
        self._pst_parent = torch.zeros(B, K, K, device=device, dtype=dtype)
        self._pst_visibility = torch.zeros(B, K, device=device, dtype=dtype)
        self._pst_occlusion = torch.zeros(B, K, device=device, dtype=dtype)
        self._pst_has_text = torch.zeros(B, K, device=device, dtype=dtype)
        self._pst_text = torch.zeros(B, K, self.physical_text_dim, device=device, dtype=dtype)
        self._pst_symbol = torch.zeros(B, K, self.physical_symbol_dim, device=device, dtype=dtype)
        self._pst_observed = torch.zeros(B, K, device=device, dtype=torch.bool)
        self._pst_last_seen = torch.zeros(B, K, device=device, dtype=torch.long)
        self._pst_step = torch.zeros(B, device=device, dtype=torch.long)
        self._robot_self_state = torch.zeros(B, self.robot_self_dim, device=device, dtype=dtype)
        self._robot_action_context = torch.zeros(B, self.action_dim, device=device, dtype=dtype)

    def EnsureB(self, B: int, device: torch.device, dtype: torch.dtype):
        B = int(B)
        cap = int(self._mem_capacity)
        self.EnsurePhysicalMemory(B, device, dtype)

        if self._mem_keys.size(0) != B:
            self._mem_keys = torch.zeros(B, cap, self.stoch_dim, device=device, dtype=dtype)
            self._mem_vals = torch.zeros(B, cap, self.state_dim, device=device, dtype=dtype)
            self._mem_imp = torch.zeros(B, cap, device=device, dtype=dtype)
            self._mem_steps = torch.zeros(B, cap, device=device, dtype=torch.long)
            self._mem_size = torch.zeros(B, device=device, dtype=torch.long)
            self._mem_global_step = torch.zeros(B, device=device, dtype=torch.long)

        if self._h.size(0) != B or self._h.device != device or self._h.dtype != dtype:
            self._h = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
            self._z = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)
            self._A_prev = None

        self.s4.EnsureB(B, device, dtype)


    def SaveMemory(self, path: Optional[str] = None):
        if not self._use_memory:
            return
        p = path or self._mem_path
        if not p:
            return

        dirpath = os.path.dirname(p)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        B = int(self._mem_keys.size(0))
        cap = int(self._mem_capacity)

        maxN = int(self._mem_size.max().item()) if B > 0 else 0
        maxN = max(0, min(maxN, cap))

        payload = {
            "pst_schema_version": 2,
            "pst_contact_point_frame": "world",
            "mem_keys": self._mem_keys[:, :maxN].detach().cpu(), # [B,maxN,Z]
            "mem_vals": self._mem_vals[:, :maxN].detach().cpu(), # [B,maxN,S]
            "mem_imp": self._mem_imp[:,  :maxN].detach().cpu(), # [B,maxN]
            "mem_steps": self._mem_steps[:, :maxN].detach().cpu(), # [B,maxN]
            "mem_size": self._mem_size.detach().cpu(), # [B]
            "mem_global_step": self._mem_global_step.detach().cpu(), # [B]
            "pst_slot_state": self._pst_slot_state.detach().cpu(),
            "pst_pose_world": self._pst_pose_world.detach().cpu(),
            "pst_attribute": self._pst_attribute.detach().cpu(),
            "pst_slot_presence": self._pst_slot_presence.detach().cpu(),
            "pst_entity_prob": self._pst_entity_prob.detach().cpu(),
            "pst_identity_key": self._pst_identity_key.detach().cpu(),
            "pst_pairwise_relation": self._pst_pairwise_relation.detach().cpu(),
            "pst_pair_last_seen": self._pst_pair_last_seen.detach().cpu(),
            "pst_external_relation": self._pst_external_relation.detach().cpu(),
            "pst_semantic": self._pst_semantic.detach().cpu(),
            "pst_size": self._pst_size.detach().cpu(),
            "pst_state": self._pst_state.detach().cpu(),
            "pst_affordance": self._pst_affordance.detach().cpu(),
            "pst_motion": self._pst_motion.detach().cpu(),
            "pst_moving": self._pst_moving.detach().cpu(),
            "pst_contact": self._pst_contact.detach().cpu(),
            "pst_contact_force": self._pst_contact_force.detach().cpu(),
            "pst_contact_point": self._pst_contact_point.detach().cpu(),
            "pst_parent": self._pst_parent.detach().cpu(),
            "pst_visibility": self._pst_visibility.detach().cpu(),
            "pst_occlusion": self._pst_occlusion.detach().cpu(),
            "pst_has_text": self._pst_has_text.detach().cpu(),
            "pst_text": self._pst_text.detach().cpu(),
            "pst_symbol": self._pst_symbol.detach().cpu(),
            "pst_observed": self._pst_observed.detach().cpu(),
            "pst_last_seen": self._pst_last_seen.detach().cpu(),
            "pst_step": self._pst_step.detach().cpu(),
            "robot_self_state": self._robot_self_state.detach().cpu(),
            "robot_action_context": self._robot_action_context.detach().cpu(),} # [B]

        torch.save(payload, p)

    def LoadMemory(self, path: str, mapLocation: Optional[str] = None, strict: bool = False):
        if (not self._use_memory):
            return
        if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
            if strict and (not os.path.exists(path)):
                raise FileNotFoundError(path)
            return

        payload = torch.load(path, map_location=mapLocation, weights_only=False)

        keys = payload["mem_keys"] # [B, C, stochDim]
        vals = payload["mem_vals"] # [B, C, stateDim]
        size = payload["mem_size"] # [B]
        imp = payload["mem_imp"] # [B, C]
        steps = payload["mem_steps"] # [B, C]
        global_step = payload["mem_global_step"] # [B]

        Bf = int(keys.size(0))
        Cf = int(keys.size(1))

        new_cap = int(max(self._mem_capacity, Cf))
        self._mem_capacity = new_cap

        dev = self.device
        dtyp = self.dtype

        new_keys = torch.zeros(Bf, new_cap, self.stoch_dim, device=dev, dtype=dtyp)
        new_vals = torch.zeros(Bf, new_cap, self.state_dim, device=dev, dtype=dtyp)
        new_imp = torch.zeros(Bf, new_cap, device=dev, dtype=dtyp)
        new_steps = torch.zeros(Bf, new_cap, device=dev, dtype=torch.long)
        new_size = torch.zeros(Bf, device=dev, dtype=torch.long)
        new_global_step = torch.zeros(Bf, device=dev, dtype=torch.long)

        new_keys[:, :Cf] = keys.to(device=dev, dtype=dtyp).contiguous()
        new_vals[:, :Cf] = vals.to(device=dev, dtype=dtyp).contiguous()
        new_imp[:,  :Cf] = imp.to(device=dev, dtype=dtyp).contiguous()
        new_steps[:, :Cf] = steps.to(device=dev, dtype=torch.long).contiguous()
        new_size[:] = size.to(device=dev, dtype=torch.long).clamp_(0, Cf)
        new_global_step[:] = global_step.to(device=dev, dtype=torch.long).view(-1)[:Bf]

        self._mem_keys = new_keys
        self._mem_vals = new_vals
        self._mem_imp = new_imp
        self._mem_steps = new_steps
        self._mem_size = new_size
        self._mem_global_step = new_global_step
        self.EnsureB(Bf, dev, dtyp)
        self._pst_slot_state.zero_()
        self._pst_pose_world.zero_()
        self._pst_attribute.zero_()
        self._pst_slot_presence.zero_()
        self._pst_entity_prob.zero_()
        self._pst_identity_key.zero_()
        self._pst_pairwise_relation.zero_()
        self._pst_pair_last_seen.zero_()
        self._pst_external_relation.zero_()
        self._pst_semantic.zero_()
        self._pst_size.zero_()
        self._pst_state.zero_()
        self._pst_affordance.zero_()
        self._pst_motion.zero_()
        self._pst_moving.zero_()
        self._pst_contact.zero_()
        self._pst_contact_force.zero_()
        self._pst_contact_point.zero_()
        self._pst_parent.zero_()
        self._pst_visibility.zero_()
        self._pst_occlusion.zero_()
        self._pst_has_text.zero_()
        self._pst_text.zero_()
        self._pst_symbol.zero_()
        self._pst_observed.zero_()
        self._pst_last_seen.zero_()
        self._pst_step.zero_()
        self._robot_self_state.zero_()
        self._robot_action_context.zero_()
        if "pst_slot_state" not in payload:
            return
        self._pst_slot_state.copy_(payload["pst_slot_state"].to(device=dev, dtype=dtyp))
        self._pst_pose_world.copy_(payload["pst_pose_world"].to(device=dev, dtype=dtyp))
        self._pst_attribute.copy_(payload["pst_attribute"].to(device=dev, dtype=dtyp))
        self._pst_slot_presence.copy_(payload["pst_slot_presence"].to(device=dev, dtype=dtyp))
        self._pst_entity_prob.copy_(payload["pst_entity_prob"].to(device=dev, dtype=dtyp))
        self._pst_identity_key.copy_(payload["pst_identity_key"].to(device=dev, dtype=dtyp))
        self._pst_pairwise_relation.copy_(payload["pst_pairwise_relation"].to(device=dev, dtype=dtyp))
        if "pst_pair_last_seen" in payload:
            self._pst_pair_last_seen.copy_(payload["pst_pair_last_seen"].to(device=dev, dtype=torch.long))
        self._pst_external_relation.copy_(payload["pst_external_relation"].to(device=dev, dtype=dtyp))
        self._pst_semantic.copy_(payload["pst_semantic"].to(device=dev, dtype=dtyp))
        self._pst_size.copy_(payload["pst_size"].to(device=dev, dtype=dtyp))
        self._pst_state.copy_(payload["pst_state"].to(device=dev, dtype=dtyp))
        self._pst_affordance.copy_(payload["pst_affordance"].to(device=dev, dtype=dtyp))
        self._pst_motion.copy_(payload["pst_motion"].to(device=dev, dtype=dtyp))
        self._pst_moving.copy_(payload["pst_moving"].to(device=dev, dtype=dtyp))
        self._pst_contact.copy_(payload["pst_contact"].to(device=dev, dtype=dtyp))
        self._pst_contact_force.copy_(payload["pst_contact_force"].to(device=dev, dtype=dtyp))
        if (
            int(payload.get("pst_schema_version", 1)) >= 2
            and payload.get("pst_contact_point_frame") == "world"
        ):
            self._pst_contact_point.copy_(payload["pst_contact_point"].to(device=dev, dtype=dtyp))
        self._pst_parent.copy_(payload["pst_parent"].to(device=dev, dtype=dtyp))
        self._pst_visibility.copy_(payload["pst_visibility"].to(device=dev, dtype=dtyp))
        self._pst_occlusion.copy_(payload["pst_occlusion"].to(device=dev, dtype=dtyp))
        self._pst_has_text.copy_(payload["pst_has_text"].to(device=dev, dtype=dtyp))
        self._pst_text.copy_(payload["pst_text"].to(device=dev, dtype=dtyp))
        self._pst_symbol.copy_(payload["pst_symbol"].to(device=dev, dtype=dtyp))
        self._pst_observed.copy_(payload["pst_observed"].to(device=dev, dtype=torch.bool))
        self._pst_last_seen.copy_(payload["pst_last_seen"].to(device=dev, dtype=torch.long))
        self._pst_step.copy_(payload["pst_step"].to(device=dev, dtype=torch.long))
        if "pst_pair_last_seen" not in payload:
            inferred = self.InferPairRelationLastSeen(
                self._pst_pairwise_relation,
                self._pst_slot_presence,
                self._pst_step)
            has_relation = inferred.flatten(1).any(dim=1)
            self._pst_step.copy_(torch.where(
                has_relation & (self._pst_step < 1),
                torch.ones_like(self._pst_step),
                self._pst_step))
            self._pst_pair_last_seen.copy_(self.InferPairRelationLastSeen(
                self._pst_pairwise_relation,
                self._pst_slot_presence,
                self._pst_step))
        self._robot_self_state.copy_(payload["robot_self_state"].to(device=dev, dtype=dtyp))
        self._robot_action_context.copy_(payload["robot_action_context"].to(device=dev, dtype=dtyp))

    @torch.no_grad()
    def ResetPhysicalState(self, doneMask: Optional[torch.Tensor] = None) -> None:
        buffers = (
            self._pst_slot_state,
            self._pst_pose_world,
            self._pst_attribute,
            self._pst_slot_presence,
            self._pst_entity_prob,
            self._pst_identity_key,
            self._pst_pairwise_relation,
            self._pst_pair_last_seen,
            self._pst_external_relation,
            self._pst_semantic,
            self._pst_size,
            self._pst_state,
            self._pst_affordance,
            self._pst_motion,
            self._pst_moving,
            self._pst_contact,
            self._pst_contact_force,
            self._pst_contact_point,
            self._pst_parent,
            self._pst_visibility,
            self._pst_occlusion,
            self._pst_has_text,
            self._pst_text,
            self._pst_symbol,
            self._pst_observed,
            self._pst_last_seen,
            self._pst_step,
            self._robot_self_state,
            self._robot_action_context,)
        if doneMask is None:
            for buffer in buffers:
                buffer.zero_()
            return

        mask = doneMask.to(device=self._pst_slot_state.device, dtype=torch.bool).view(-1)
        if mask.numel() != self._pst_slot_state.size(0):
            raise ValueError(
                f"doneMask must have {self._pst_slot_state.size(0)} elements, got {mask.numel()}")
        if not bool(mask.any().item()):
            return
        for buffer in buffers:
            buffer[mask] = 0

    @torch.no_grad()
    def ResetEpisodeState(self, doneMask: torch.Tensor) -> None:
        mask = doneMask.to(device=self._h.device, dtype=torch.bool).view(-1)
        if mask.numel() != self._h.size(0):
            raise ValueError(f"doneMask must have {self._h.size(0)} elements, got {mask.numel()}")
        if not bool(mask.any().item()):
            return
        self._h[mask] = 0
        self._z[mask] = 0
        self.s4.x[mask] = 0
        if self._A_prev is not None and self._A_prev.size(0) == mask.numel():
            self._A_prev[mask] = 0
        self.ResetPhysicalState(mask)

    def ResetMemory(self):
        if self._use_memory:
            self._mem_keys.zero_()
            self._mem_vals.zero_()
            self._mem_imp.zero_()
            self._mem_steps.zero_()
            self._mem_size.zero_()
            self._mem_global_step.zero_()
        self.ResetPhysicalState()

    @torch.no_grad()
    def ReorderMemorySteps(self):
        if not self._use_memory:
            return

        B = int(self._mem_size.size(0))
        cap = int(self._mem_capacity)
        device = self.device

        if cap <= 0 or B <= 0:
            self._mem_steps.zero_()
            self._mem_global_step.zero_()
            return

        slots = torch.arange(cap, device=device).view(1, cap)
        valid = slots < self._mem_size.view(B, 1) # [B, cap]

        if not bool(valid.any().item()):
            self._mem_steps.zero_()
            self._mem_global_step.zero_()
            return

        max_step = torch.iinfo(self._mem_steps.dtype).max
        metric = torch.where(valid, self._mem_steps, torch.full_like(self._mem_steps, max_step))
        order = torch.argsort(metric, dim=1, descending=False)

        new_steps = torch.zeros_like(self._mem_steps)
        ranks = torch.arange(1, cap + 1, device=device, dtype=torch.long).view(1, cap).expand(B, cap)
        rank_valid = ranks <= self._mem_size.view(B, 1)
        assign = torch.where(rank_valid, ranks, torch.zeros_like(ranks))
        new_steps.scatter_(1, order, assign)
        new_steps = torch.where(valid, new_steps, torch.zeros_like(new_steps))

        self._mem_steps.copy_(new_steps)
        self._mem_global_step.copy_(self._mem_size)

    @torch.no_grad()
    def MemAdd(
        self,
        keyE: torch.Tensor, # [B, Z]
        valH: torch.Tensor, # [B, D]
        imp: torch.Tensor,): # [B]
    
        if not self._use_memory:
            return

        B = int(keyE.size(0))

        cap = int(self._mem_capacity)

        size = self._mem_size # [B]  
        has_space = size < cap # [B]  
        idx_replace = torch.argmin(self._mem_imp, dim=1) # [B]
        idx = torch.where(has_space, size, idx_replace).long() # [B]

        self._mem_global_step.add_(1)
        self._mem_size = torch.where(has_space, size + 1, size) # [B]

        bidx = torch.arange(B, device=self.device)

        self._mem_keys[bidx, idx] = keyE # [B,Z]
        self._mem_vals[bidx, idx] = valH # [B,S]
        self._mem_imp[bidx, idx] = imp # [B]
        self._mem_steps[bidx, idx] = self._mem_global_step

        if self._mem_path and self._mem_autosave_every > 0:
            self._mem_add_count += 1
            if self._mem_add_count % self._mem_autosave_every == 0:
                self.SaveMemory(self._mem_path)

    def MemRetrieve(
        self,
        queryE: torch.Tensor,
        *,
        updateImportance: bool = True,
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self._use_memory:
            return None

        B = int(queryE.size(0))

        filled = self._mem_size # [B]
        filled_max = int(filled.max().item())

        if filled_max <= 0:
            return None

        cap = int(self._mem_capacity)
        K_req = max(int(self._mem_topk), 1)
        K = min(K_req, filled_max, cap)
        temp = max(float(self._mem_temp), 1e-6)

        keys = self._mem_keys # [B,C,Z]
        vals = self._mem_vals # [B,C,S]

        q = queryE # [B,Z]
        k = keys

        sims = torch.einsum("bd,bnd->bn", q, k) # [B,C]

        idx = torch.arange(cap, device=sims.device).view(1, cap)
        valid = idx < self._mem_size.view(B, 1)
        sims = sims.masked_fill(~valid, -1e9)

        top_vals, top_idx = torch.topk(sims, k=K, dim=-1) # [B,K]

        logits = top_vals / temp
        weights = F.softmax(logits, dim=-1) # [B,K]

        has_memory = self._mem_size > 0
        weights = weights * has_memory.view(B, 1).to(weights.dtype)

        gathered = vals.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, vals.size(-1))) # [B,K,D]
        mem_h = (weights.unsqueeze(-1) * gathered).sum(dim=1) # [B,D]

        if updateImportance:
            with torch.no_grad():
                inc = weights.detach() # [B,K] in [0,1], sum=1
                cur = self._mem_imp.gather(1, top_idx) # [B,K]
                lr = float(getattr(self, "_mem_imp_lr", 0.10))
                new = cur + lr * (1.0 - cur) * inc
                self._mem_imp.scatter_(1, top_idx, new.clamp_(0.0, 1.0))

        return mem_h, has_memory # [B,D], [B]

    @staticmethod
    def QuaternionMultiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        lx, ly, lz, lw = left.unbind(dim=-1)
        rx, ry, rz, rw = right.unbind(dim=-1)
        return torch.stack([
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,], dim=-1)

    @staticmethod
    def QuaternionRotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        q_vec = quat[..., :3]
        t = 2.0 * torch.cross(q_vec, vector, dim=-1)
        return vector + quat[..., 3:4] * t + torch.cross(q_vec, t, dim=-1)

    @staticmethod
    def NormalizeQuaternion(quat: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(quat.float(), dim=-1, eps=1e-6).to(quat.dtype)
        identity = torch.zeros_like(normalized)
        identity[..., 3] = 1.0
        return torch.where(
            quat.norm(dim=-1, keepdim=True) > 1e-6,
            normalized,
            identity)

    def ComposePose(self, parent: torch.Tensor, child: torch.Tensor) -> torch.Tensor:
        while parent.dim() < child.dim():
            parent = parent.unsqueeze(1)
        parent_q = self.NormalizeQuaternion(parent[..., 3:7])
        child_q = self.NormalizeQuaternion(child[..., 3:7])
        translation = parent[..., :3] + self.QuaternionRotate(parent_q, child[..., :3])
        rotation = F.normalize(self.QuaternionMultiply(parent_q, child_q).float(), dim=-1, eps=1e-6).to(child.dtype)
        pivot_index = rotation.abs().argmax(dim=-1, keepdim=True)
        pivot = rotation.gather(-1, pivot_index)
        rotation = torch.where(pivot < 0.0, -rotation, rotation)
        return torch.cat([translation, rotation], dim=-1)

    def PoseToWorld(
        self,
        pose: torch.Tensor,
        cameraPoseWorld: torch.Tensor,) -> torch.Tensor:
        return self.ComposePose(cameraPoseWorld, pose)

    def WeightedPointToWorld(
        self,
        weightedPointCamera: torch.Tensor,
        pointWeight: torch.Tensor,
        cameraPoseWorld: torch.Tensor,) -> torch.Tensor:
        camera = cameraPoseWorld
        while camera.dim() < weightedPointCamera.dim():
            camera = camera.unsqueeze(1)
        camera_q = self.NormalizeQuaternion(camera[..., 3:7])
        return (
            self.QuaternionRotate(camera_q, weightedPointCamera)
            + camera[..., :3] * pointWeight.unsqueeze(-1))

    def InferPairRelationLastSeen(
        self,
        pairwiseRelation: torch.Tensor,
        slotPresence: torch.Tensor,
        step: torch.Tensor,
        ) -> torch.Tensor:
        K = int(pairwiseRelation.size(1))
        semantic_seen = pairwiseRelation[..., 4:].abs().sum(dim=-1) > 0.0
        active = slotPresence > 1e-6
        off_diagonal = ~torch.eye(K, device=pairwiseRelation.device, dtype=torch.bool)
        valid = active.unsqueeze(2) & active.unsqueeze(1) & off_diagonal.unsqueeze(0)
        timestamp = step.to(device=pairwiseRelation.device, dtype=torch.long).view(-1, 1, 1)
        timestamp = timestamp.clamp_min(1).expand_as(semantic_seen)
        return torch.where(semantic_seen & valid, timestamp, torch.zeros_like(timestamp))

    @torch.no_grad()
    def ExportPhysicalState(self) -> Dict[str, torch.Tensor]:
        return {
            "SlotState": self._pst_slot_state.detach().clone(),
            "PoseWorld": self._pst_pose_world.detach().clone(),
            "ARaw": self._pst_attribute.detach().clone(),
            "SlotPresence": self._pst_slot_presence.detach().clone(),
            "MphysRaw": self._pst_entity_prob.detach().clone(),
            "IdentityKey": self._pst_identity_key.detach().clone(),
            "PairwiseRelation": self._pst_pairwise_relation.detach().clone(),
            "PairRelationLastSeen": self._pst_pair_last_seen.detach().clone(),
            "ExternalRelationProbRaw": self._pst_external_relation.detach().clone(),
            "Semantic": self._pst_semantic.detach().clone(),
            "LevelProb": self._pst_semantic[..., :3].detach().clone(),
            "ObjectClassProb": self._pst_semantic[..., 3:259].detach().clone(),
            "PartClassProb": self._pst_semantic[..., 259:].detach().clone(),
            "Size": self._pst_size.detach().clone(),
            "StateRaw": self._pst_state.detach().clone(),
            "AffordanceRaw": self._pst_affordance.detach().clone(),
            "MotionRaw": self._pst_motion.detach().clone(),
            "MovingProbRaw": self._pst_moving.detach().clone(),
            "ContactProbRaw": self._pst_contact.detach().clone(),
            "ContactForceRaw": self._pst_contact_force.detach().clone(),
            "ContactPointRaw": self._pst_contact_point.detach().clone(),
            "ContactPointWorldRaw": self._pst_contact_point.detach().clone(),
            "ParentProb": self._pst_parent.detach().clone(),
            "Visibility": self._pst_visibility.detach().clone(),
            "Occlusion": self._pst_occlusion.detach().clone(),
            "HasTextProb": self._pst_has_text.detach().clone(),
            "TextEmbed": self._pst_text.detach().clone(),
            "SymbolProb": self._pst_symbol.detach().clone(),
            "Observed": self._pst_observed.detach().clone(),
            "LastSeen": self._pst_last_seen.detach().clone(),
            "Step": self._pst_step.detach().clone(),}

    @torch.no_grad()
    def ExportRobotSelfState(self) -> Dict[str, torch.Tensor]:
        return {
            "RobotSelfState": self._robot_self_state.detach().clone(),
            "ExecutedAction": self._robot_action_context.detach().clone(),}

    @torch.no_grad()
    def ImportRobotSelfState(self, robotSelfState: Dict[str, torch.Tensor]) -> None:
        reference = robotSelfState["RobotSelfState"]
        self.EnsurePhysicalMemory(int(reference.size(0)), self.device, self.dtype)
        self._robot_self_state.copy_(robotSelfState["RobotSelfState"].to(
            device=self._robot_self_state.device, dtype=self._robot_self_state.dtype))
        self._robot_action_context.copy_(robotSelfState["ExecutedAction"].to(
            device=self._robot_action_context.device, dtype=self._robot_action_context.dtype))

    @torch.no_grad()
    def ImportPhysicalState(self, physicalState: Dict[str, torch.Tensor]) -> None:
        reference = physicalState["SlotState"]
        self.EnsurePhysicalMemory(int(reference.size(0)), self.device, self.dtype)

        def copy_from(buffer: torch.Tensor, key: str) -> None:
            buffer.copy_(physicalState[key].to(device=buffer.device, dtype=buffer.dtype))

        self._pst_slot_state.zero_()
        self._pst_pose_world.zero_()
        self._pst_attribute.zero_()
        self._pst_slot_presence.zero_()
        self._pst_entity_prob.zero_()
        self._pst_identity_key.zero_()
        self._pst_pairwise_relation.zero_()
        self._pst_pair_last_seen.zero_()
        self._pst_external_relation.zero_()
        self._pst_semantic.zero_()
        self._pst_size.zero_()
        self._pst_state.zero_()
        self._pst_affordance.zero_()
        self._pst_motion.zero_()
        self._pst_moving.zero_()
        self._pst_contact.zero_()
        self._pst_contact_force.zero_()
        self._pst_contact_point.zero_()
        self._pst_parent.zero_()
        self._pst_visibility.zero_()
        self._pst_occlusion.zero_()
        self._pst_has_text.zero_()
        self._pst_text.zero_()
        self._pst_symbol.zero_()
        self._pst_observed.zero_()
        self._pst_last_seen.zero_()
        copy_from(self._pst_slot_state, "SlotState")
        copy_from(self._pst_pose_world, "PoseWorld")
        copy_from(self._pst_attribute, "ARaw")
        copy_from(self._pst_slot_presence, "SlotPresence")
        copy_from(self._pst_entity_prob, "MphysRaw")
        copy_from(self._pst_identity_key, "IdentityKey")
        copy_from(self._pst_pairwise_relation, "PairwiseRelation")
        if "PairRelationLastSeen" in physicalState:
            copy_from(self._pst_pair_last_seen, "PairRelationLastSeen")
        copy_from(self._pst_external_relation, "ExternalRelationProbRaw")
        copy_from(self._pst_semantic, "Semantic")
        copy_from(self._pst_size, "Size")
        copy_from(self._pst_state, "StateRaw")
        copy_from(self._pst_affordance, "AffordanceRaw")
        copy_from(self._pst_motion, "MotionRaw")
        copy_from(self._pst_moving, "MovingProbRaw")
        copy_from(self._pst_contact, "ContactProbRaw")
        copy_from(self._pst_contact_force, "ContactForceRaw")
        contact_point_world = physicalState.get("ContactPointWorldRaw", physicalState.get("ContactPointWorld"))
        if contact_point_world is not None:
            self._pst_contact_point.copy_(contact_point_world.to(
                device=self._pst_contact_point.device, dtype=self._pst_contact_point.dtype))
        copy_from(self._pst_parent, "ParentProb")
        copy_from(self._pst_visibility, "Visibility")
        copy_from(self._pst_occlusion, "Occlusion")
        copy_from(self._pst_has_text, "HasTextProb")
        copy_from(self._pst_text, "TextEmbed")
        copy_from(self._pst_symbol, "SymbolProb")
        copy_from(self._pst_observed, "Observed")
        copy_from(self._pst_last_seen, "LastSeen")
        copy_from(self._pst_step, "Step")
        if "PairRelationLastSeen" not in physicalState:
            inferred = self.InferPairRelationLastSeen(
                self._pst_pairwise_relation,
                self._pst_slot_presence,
                self._pst_step)
            has_relation = inferred.flatten(1).any(dim=1)
            self._pst_step.copy_(torch.where(
                has_relation & (self._pst_step < 1),
                torch.ones_like(self._pst_step),
                self._pst_step))
            self._pst_pair_last_seen.copy_(self.InferPairRelationLastSeen(
                self._pst_pairwise_relation,
                self._pst_slot_presence,
                self._pst_step))

    @staticmethod
    def MatchLegalPhysicalSlots(
        cost: torch.Tensor,
        legal: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cost.ndim != 2 or legal.shape != cost.shape:
            raise ValueError("cost and legal must have the same two-dimensional shape")
        active_count, incoming_count = int(cost.size(0)), int(cost.size(1))
        assignment_cost = cost.float()
        augmented_size = active_count + incoming_count
        illegal_cost = 1e6
        unmatched_cost = 1.0
        augmented = assignment_cost.new_full(
            (augmented_size, augmented_size), illegal_cost)
        augmented[:active_count, :incoming_count] = torch.where(
            legal,
            assignment_cost,
            assignment_cost.new_full(assignment_cost.shape, illegal_cost))
        augmented[:active_count, incoming_count:] = unmatched_cost
        augmented[active_count:, :incoming_count] = unmatched_cost
        augmented[active_count:, incoming_count:] = 0.0
        assigned_rows, assigned_cols = HungarianAssignment(augmented)
        real_assignment = (assigned_rows < active_count) & (assigned_cols < incoming_count)
        return assigned_rows[real_assignment], assigned_cols[real_assignment]

    def ValidatePhysicalUpdateInputs(
        self,
        observedPst: Dict[str, torch.Tensor],
        cameraPoseWorld: torch.Tensor,
        robotSelfState: torch.Tensor,
        executedActionEmbed: torch.Tensor,
        ) -> Tuple[int, int]:
        if not isinstance(observedPst, dict):
            raise TypeError("observedPst must be a dictionary of tensors")
        required = {
            "SlotState", "PoseCamera", "ARaw", "ObservedSlotMask", "MphysRaw",
            "IdentityKey", "Semantic", "ExternalRelationProbRaw", "Size",
            "StateRaw", "AffordanceRaw", "MotionRaw", "MovingProbRaw",
            "ContactProbRaw", "ContactForceRaw", "ContactPointRaw", "Visibility",
            "Occlusion", "HasTextProb", "TextEmbed", "SymbolProb",
            "PairwiseRelation", "ParentProb"}
        missing = sorted(required.difference(observedPst))
        if missing:
            raise KeyError(f"observedPst is missing required fields: {missing}")

        slot_state = observedPst["SlotState"]
        if not torch.is_tensor(slot_state) or slot_state.ndim != 3:
            actual = tuple(slot_state.shape) if torch.is_tensor(slot_state) else type(slot_state).__name__
            raise ValueError(f"observedPst['SlotState'] must have shape [B, K, D], got {actual}")
        B, K = int(slot_state.size(0)), int(slot_state.size(1))
        if B <= 0 or K <= 0:
            raise ValueError("observedPst must contain at least one batch row and one slot")

        expected_shapes = {
            "SlotState": (B, K, self.physical_slot_dim),
            "PoseCamera": (B, K, self.physical_pose_dim),
            "ARaw": (B, K, self.physical_attr_dim),
            "ObservedSlotMask": (B, K),
            "MphysRaw": (B, K),
            "IdentityKey": (B, K, self.physical_id_dim),
            "Semantic": (B, K, self.physical_semantic_dim),
            "ExternalRelationProbRaw": (B, K, self.physical_relation_classes),
            "Size": (B, K, 3),
            "StateRaw": (B, K, self.physical_state_dim),
            "AffordanceRaw": (B, K, self.physical_affordance_dim),
            "MotionRaw": (B, K, self.physical_pose_dim),
            "MovingProbRaw": (B, K),
            "ContactProbRaw": (B, K),
            "ContactForceRaw": (B, K, 2),
            "ContactPointRaw": (B, K, 3),
            "Visibility": (B, K),
            "Occlusion": (B, K),
            "HasTextProb": (B, K),
            "TextEmbed": (B, K, self.physical_text_dim),
            "SymbolProb": (B, K, self.physical_symbol_dim),
            "PairwiseRelation": (B, K, K, self.physical_rel_dim),
            "ParentProb": (B, K, K),}
        external_shapes = {
            "cameraPoseWorld": (cameraPoseWorld, (B, self.physical_pose_dim)),
            "robotSelfState": (robotSelfState, (B, self.robot_self_dim)),
            "executedActionEmbed": (executedActionEmbed, (B, self.action_dim)),}
        tensors = {
            **{name: observedPst[name] for name in expected_shapes},
            **{name: value for name, (value, _) in external_shapes.items()}}
        shapes = {
            **expected_shapes,
            **{name: shape for name, (_, shape) in external_shapes.items()}}
        model_device, model_dtype = self.device, self.dtype
        for name, value in tensors.items():
            if not torch.is_tensor(value):
                raise TypeError(f"{name} must be a tensor")
            if tuple(value.shape) != shapes[name]:
                raise ValueError(
                    f"{name} must have shape {shapes[name]}, got {tuple(value.shape)}")
            if value.device != model_device:
                raise ValueError(f"{name} must be on {model_device}, got {value.device}")
            if value.dtype != model_dtype:
                raise ValueError(f"{name} must have dtype {model_dtype}, got {value.dtype}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")
        return B, K

    @torch.no_grad()
    def UpdatePhysicalState(
        self,
        observedPst: Dict[str, torch.Tensor],
        cameraPoseWorld: torch.Tensor,
        robotSelfState: torch.Tensor,
        executedActionEmbed: torch.Tensor,) -> Dict[str, torch.Tensor]:
        B, _ = self.ValidatePhysicalUpdateInputs(
            observedPst,
            cameraPoseWorld,
            robotSelfState,
            executedActionEmbed)
        self.EnsurePhysicalMemory(B, self.device, self.dtype)

        self._robot_self_state.copy_(robotSelfState.detach())
        self._robot_action_context.copy_(executedActionEmbed.detach())

        observed_p_world = self.PoseToWorld(observedPst["PoseCamera"].detach(), cameraPoseWorld)
        observed_contact_world = self.WeightedPointToWorld(
            observedPst["ContactPointRaw"].detach(),
            observedPst["ContactProbRaw"].detach(),
            cameraPoseWorld)
        observed_m = observedPst["ObservedSlotMask"].detach()
        observed_m_phys = observedPst["MphysRaw"].detach() # [B, K]
        observed_c = observedPst["IdentityKey"].detach()

        self._pst_step.add_(1)
        self._pst_slot_presence.mul_(self.physical_confidence_decay)
        self._pst_observed.zero_()

        incoming_to_memory = torch.full(
            observed_m.shape,
            -1,
            device=observed_m.device,
            dtype=torch.long)

        def write_slot(b: int, source_idx: int, target_idx: int, reset_pairwise_relations: bool) -> None:
            incoming_to_memory[b, source_idx] = target_idx
            if reset_pairwise_relations:
                self._pst_pairwise_relation[b, target_idx].zero_()
                self._pst_pairwise_relation[b, :, target_idx].zero_()
                self._pst_pair_last_seen[b, target_idx].zero_()
                self._pst_pair_last_seen[b, :, target_idx].zero_()
                self._pst_parent[b, target_idx].zero_()
                self._pst_parent[b, :, target_idx].zero_()
            self._pst_slot_state[b, target_idx] = observedPst["SlotState"][b, source_idx].detach()
            self._pst_pose_world[b, target_idx] = observed_p_world[b, source_idx]
            self._pst_attribute[b, target_idx] = observedPst["ARaw"][b, source_idx].detach()
            self._pst_slot_presence[b, target_idx] = observed_m[b, source_idx]
            self._pst_entity_prob[b, target_idx] = observed_m_phys[b, source_idx]
            self._pst_identity_key[b, target_idx] = observed_c[b, source_idx]
            self._pst_semantic[b, target_idx] = observedPst["Semantic"][b, source_idx].detach()
            self._pst_external_relation[b, target_idx] = observedPst["ExternalRelationProbRaw"][b, source_idx].detach()
            self._pst_size[b, target_idx] = observedPst["Size"][b, source_idx].detach()
            self._pst_state[b, target_idx] = observedPst["StateRaw"][b, source_idx].detach()
            self._pst_affordance[b, target_idx] = observedPst["AffordanceRaw"][b, source_idx].detach()
            self._pst_motion[b, target_idx] = observedPst["MotionRaw"][b, source_idx].detach()
            self._pst_moving[b, target_idx] = observedPst["MovingProbRaw"][b, source_idx].detach()
            self._pst_contact[b, target_idx] = observedPst["ContactProbRaw"][b, source_idx].detach()
            self._pst_contact_force[b, target_idx] = observedPst["ContactForceRaw"][b, source_idx].detach()
            self._pst_contact_point[b, target_idx] = observed_contact_world[b, source_idx]
            self._pst_visibility[b, target_idx] = observedPst["Visibility"][b, source_idx].detach()
            self._pst_occlusion[b, target_idx] = observedPst["Occlusion"][b, source_idx].detach()
            self._pst_has_text[b, target_idx] = observedPst["HasTextProb"][b, source_idx].detach()
            self._pst_text[b, target_idx] = observedPst["TextEmbed"][b, source_idx].detach()
            self._pst_symbol[b, target_idx] = observedPst["SymbolProb"][b, source_idx].detach()
            self._pst_observed[b, target_idx] = True
            self._pst_last_seen[b, target_idx] = self._pst_step[b]

        for b in range(B):
            incoming = torch.nonzero(
                observed_m[b] >= self.physical_observation_threshold,
                as_tuple=False).flatten()
            if incoming.numel() == 0:
                continue

            incoming = incoming[torch.argsort(observed_m[b, incoming], descending=True)]
            assigned_source = torch.zeros(observed_m.size(1), device=observed_m.device, dtype=torch.bool)
            active_idx = torch.nonzero(self._pst_slot_presence[b] > 1e-6, as_tuple=False).flatten()

            if active_idx.numel() > 0:
                similarity = torch.matmul(self._pst_identity_key[b, active_idx], observed_c[b, incoming].t())
                pose_delta = (
                    self._pst_pose_world[b, active_idx, :3].unsqueeze(1)
                    - observed_p_world[b, incoming, :3].unsqueeze(0))
                pose_cost = torch.tanh(pose_delta.norm(dim=-1))
                last_seen_age = (self._pst_step[b].float() - self._pst_last_seen[b, active_idx].float()).clamp_min(0.0)
                age_cost = last_seen_age / (last_seen_age + 1.0)
                cost = (
                    (1.0 - similarity)
                    + 0.25 * pose_cost
                    + 0.05 * age_cost.unsqueeze(1)
                    - 0.10 * self._pst_slot_presence[b, active_idx].unsqueeze(1)
                    - 0.05 * observed_m[b, incoming].unsqueeze(0)) # [N_active, N_incoming]
                active_local, incoming_local = self.MatchLegalPhysicalSlots(
                    cost,
                    similarity >= self.physical_identity_threshold)

                for active_pos_t, incoming_pos_t in zip(active_local, incoming_local):
                    active_pos = int(active_pos_t.item())
                    incoming_pos = int(incoming_pos_t.item())
                    source_idx = int(incoming[incoming_pos].item())
                    target_idx = int(active_idx[active_pos].item())
                    assigned_source[source_idx] = True
                    write_slot(b, source_idx, target_idx, False)

            for source_idx in incoming[~assigned_source[incoming]].tolist():
                empty_slots = torch.nonzero(self._pst_slot_presence[b] <= 1e-6, as_tuple=False).flatten()
                if empty_slots.numel() > 0:
                    target_idx = int(empty_slots[0].item())
                else:
                    slot_age = self._pst_step[b].float() - self._pst_last_seen[b].float()
                    physical_importance = self._pst_slot_presence[b] * self._pst_entity_prob[b]
                    interaction_importance = 0.5 * self._pst_moving[b] + 0.5 * self._pst_contact[b]
                    semantic_importance = (
                        0.5 * self._pst_visibility[b]
                        + 0.25 * self._pst_has_text[b]
                        + 0.25 * self._pst_symbol[b].amax(dim=-1))
                    parent_importance = torch.maximum(
                        self._pst_parent[b].amax(dim=0),
                        self._pst_parent[b].amax(dim=1))
                    relation_importance = torch.maximum(
                        parent_importance,
                        self._pst_external_relation[b].amax(dim=-1))
                    slot_importance = (
                        0.45 * physical_importance
                        + 0.25 * interaction_importance
                        + 0.20 * semantic_importance
                        + 0.10 * relation_importance)
                    replacement_score = slot_age * (1.0 - slot_importance)
                    target_idx = int(torch.argmax(replacement_score).item())
                write_slot(b, int(source_idx), target_idx, True)

            mapped_sources = torch.nonzero(incoming_to_memory[b] >= 0, as_tuple=False).flatten()
            if mapped_sources.numel() > 0:
                memory_targets = incoming_to_memory[b, mapped_sources]
                mem_grid = memory_targets.view(-1, 1).expand(-1, memory_targets.numel())
                src_grid = mapped_sources.view(-1, 1).expand(-1, mapped_sources.numel())
                self._pst_pairwise_relation[b, mem_grid, mem_grid.t()] = observedPst["PairwiseRelation"][b, src_grid, src_grid.t()].detach()
                self._pst_pair_last_seen[b, mem_grid, mem_grid.t()] = self._pst_step[b]
                self._pst_parent[b, mem_grid, mem_grid.t()] = observedPst["ParentProb"][b, src_grid, src_grid.t()].detach()

        relative = self._pst_pose_world[..., :3].unsqueeze(1) - self._pst_pose_world[..., :3].unsqueeze(2)
        distance = relative.norm(dim=-1, keepdim=True)
        off_diagonal = ~torch.eye(self.physical_slots, device=self._pst_slot_presence.device, dtype=torch.bool)
        pair_valid = (
            (self._pst_slot_presence > 1e-6).unsqueeze(2)
            & (self._pst_slot_presence > 1e-6).unsqueeze(1)
            & off_diagonal.unsqueeze(0)).unsqueeze(-1)
        self._pst_pairwise_relation.masked_fill_(~pair_valid, 0.0)
        self._pst_pair_last_seen.masked_fill_(~pair_valid.squeeze(-1), 0)
        self._pst_pairwise_relation[..., :3] = relative * pair_valid.to(relative.dtype)
        self._pst_pairwise_relation[..., 3:4] = distance * pair_valid.to(distance.dtype)
        self._pst_parent.masked_fill_(~pair_valid.squeeze(-1), 0.0)

        merged = self.ExportPhysicalState()
        if "_aux" in observedPst:
            merged["_aux"] = observedPst["_aux"]
        return merged

    def PhysicalSlotSummary(self, S: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        weight = M.unsqueeze(-1)
        return (S * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)

    def BuildEmbodiedAction(
        self,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build a relation-conditioned action while preserving the caller's time contract.

        Posterior/filtering paths use ``(PST_t, robot_t, executed_feedback_{t-1})`` to infer the
        current world state. Prior/prospective paths use ``(PST_t, robot_t, candidate_action_t)``
        to predict a possible next state. This deliberate asymmetry is not a causal leak.
        """
        robot_world_context = self.robot_world_relation(robotSelfState, physicalState, actionEnc)
        embodied_action = self.embodied_action_proj(torch.cat([
            actionEnc,
            robotSelfState,
            robot_world_context], dim=-1))
        return embodied_action, robot_world_context

    def BindPhysicalMu(
        self,
        worldH: torch.Tensor,
        worldZMu: torch.Tensor,
        worldX: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotWorldContext: torch.Tensor) -> Dict[str, torch.Tensor]:
        binding = self.pst_binder(worldH, worldZMu, worldX, physicalState, actionEnc, robotWorldContext)
        pst_summary_target = self.PhysicalSlotSummary(
            physicalState["SlotState"],
            binding["slot_binding_weight"]).detach()
        loss_pst_bind = (
            0.01 * binding["delta_mu"].square().mean()
            + 0.10 * F.smooth_l1_loss(binding["pst_summary_pred"], pst_summary_target, reduction="mean")
            + 0.001 * binding["bind_gate"].mean())
        binding["loss_pst_bind"] = loss_pst_bind
        binding["robot_world_context"] = robotWorldContext
        return binding

    def BoundReward(self, reward: torch.Tensor) -> torch.Tensor:
        scale = max(abs(float(self.reward_min)), abs(float(self.reward_max)), 1e-6)
        return scale * torch.tanh(reward / scale)

    def RewardDoneTrunk(self, inp: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        normalized = self.rdone_ln(inp)
        if not deterministic:
            return self.rdone_trunk(normalized)
        hidden = self.rdone_trunk[0](normalized)
        hidden = self.rdone_trunk[1](hidden)
        hidden = F.dropout(hidden, p=float(self.rdone_trunk[2].p), training=False)
        hidden = self.rdone_trunk[3](hidden)
        return self.rdone_trunk[4](hidden)

    def ResetState(self, batchSize: int = 1):
        device, dtype = self.device, self.dtype
        B = int(batchSize)

        self._h = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
        self._z = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)
        self.s4.ResetState(B)
        self._A_prev = None

        # PST is part of the world state even when episodic key/value memory is disabled.
        self.EnsurePhysicalMemory(B, device, dtype)
        if self._use_memory:
            self.EnsureB(B, device, self.dtype)

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._h, self._z, self.s4.x

    def ImportState(self, h: torch.Tensor, z: torch.Tensor, s4x: torch.Tensor):
        self._h = h.detach().to(device=self.device, dtype=self.dtype).clone()
        self._z = z.detach().to(device=self.device, dtype=self.dtype).clone()
        self.s4.x = s4x.detach().to(device=self.device, dtype=self.dtype).clone()

    def BuildPredictedVisual(self, state: torch.Tensor) -> Dict[str, Any]:
        predicted_visual = self.predicted_visual_head(state)
        reconstructed = self.visual_reconstructor(predicted_visual)
        return {"predicted_visual": predicted_visual,"reconstructed_visual_state": reconstructed,}

    def ObjectAttentionError(self, predictedObjects: torch.Tensor, targetObjects: torch.Tensor) -> torch.Tensor:
        D = int(targetObjects.size(-1))
        scores = torch.matmul(targetObjects, predictedObjects.transpose(1, 2)) / max(float(D) ** 0.5, 1.0)
        weights = F.softmax(scores, dim=-1)
        aligned_pred = torch.matmul(weights, predictedObjects)
        return (aligned_pred - targetObjects).pow(2).mean(dim=(1, 2))

    def ComputePredictionLoss(
        self,
        predictedVisual: PredictedVisualPack,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        precision: torch.Tensor,
        sampleMask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        target = {
            "GlobalFeat": targetVisualState.GlobalFeat.detach(),
            "ObjectTokens": targetVisualState.ObjectTokens.detach(),
            "IntegratedFeat": targetVisualState.IntegratedFeat.detach(),
            "MotionPred": targetVisualState.MotionToken.detach(),}

        global_err = (predictedVisual.GlobalFeat - target["GlobalFeat"]).pow(2).mean(dim=-1)
        object_err = self.ObjectAttentionError(predictedVisual.ObjectTokens, target["ObjectTokens"])
        integrated_err = (predictedVisual.IntegratedFeat - target["IntegratedFeat"]).pow(2).mean(dim=-1)
        motion_err = (predictedVisual.MotionPred - target["MotionPred"]).pow(2).mean(dim=-1)

        recon_err = (
            (reconstructedVisualState["GlobalFeat"] - target["GlobalFeat"]).pow(2).mean(dim=-1)
            + 0.5 * self.ObjectAttentionError(reconstructedVisualState["ObjectTokens"], target["ObjectTokens"])
            + 0.25 * (reconstructedVisualState["MotionPred"] - target["MotionPred"]).pow(2).mean(dim=-1)
            + 0.5 * (reconstructedVisualState["IntegratedFeat"] - target["IntegratedFeat"]).pow(2).mean(dim=-1))
        basis_err = (reconstructedVisualState["PredErrorBasis"] - target["GlobalFeat"]).pow(2).mean(dim=-1)

        per_sample = global_err + object_err + 0.5 * integrated_err + 0.25 * motion_err + 0.5 * recon_err + 0.1 * basis_err
        batch_size = int(per_sample.size(0))
        if sampleMask is None:
            sample_mask = per_sample.new_ones(batch_size)
        else:
            sample_mask = sampleMask.detach().to(
                device=per_sample.device, dtype=per_sample.dtype).view(-1)
            if sample_mask.numel() != batch_size:
                raise ValueError(
                    f"sampleMask must have {batch_size} elements, got {sample_mask.numel()}")
            sample_mask = torch.nan_to_num(
                sample_mask, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        def masked_mean(values: torch.Tensor) -> torch.Tensor:
            return (values * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)

        p = precision.detach().view(-1).clamp(0.05, 1.0)
        if p.numel() != batch_size:
            raise ValueError(f"precision must have {batch_size} elements, got {p.numel()}")
        precision_loss = masked_mean(p * per_sample)
        inverse_losses = self.visual_reconstructor.InverseMappingLoss(
            reconstructedVisualState,
            targetVisualState,
            sampleMask=sample_mask)
        total_loss = precision_loss + 0.25 * inverse_losses["loss_pred_inverse_total"]

        losses = {
            "loss_pred_global": masked_mean(global_err),
            "loss_pred_object": masked_mean(object_err),
            "loss_pred_integrated": masked_mean(integrated_err),
            "loss_pred_motion": masked_mean(motion_err),
            "loss_pred_recon": masked_mean(recon_err),
            "loss_pred_basis": masked_mean(basis_err),
            "loss_pred_precision": precision_loss,
            "loss_pred_total": total_loss,}
        losses.update(inverse_losses)
        return losses

    def PriorRolloutFromStateAction(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1))

        embodied_action, robot_world_context = self.BuildEmbodiedAction(physicalState, actionEnc, robotSelfState)
        a_t = self.act_proj(embodied_action)

        h_next, x_next = self.s4.StepWithX(zPrev, a_t, s4xPrev)
        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True)
            dmu = self.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            mu_p = mu_p + (base_gate * gate_scale).clamp(0.0, 1.0) * dmu

        mu_p_raw = mu_p
        pst_binding = self.BindPhysicalMu(h_next, mu_p, x_next, physicalState, embodied_action, robot_world_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1))

        A_t = self.conn(s_prev_base, a_t)
        s_transport = self.conn.TransportApply(A_t, s_prev_base)
        h_phys, _, _ = self.phys_refiner(hPrev, a_t, h_next)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        w = F.softmax(self.mix_gate(torch.cat([s_base, d_tr, d_ph], dim=-1)), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        trunk = self.rdone_trunk(self.rdone_ln(torch.cat([s_base, s_next, a_t], dim=-1)))
        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": x_next,
            "s_next": s_next,
            "action_enc": actionEnc,
            "embodied_action": embodied_action,
            "robot_world_context": robot_world_context,
            "r_pred": self.BoundReward(self.rew_head(trunk).squeeze(-1)),
            "d_prob": torch.sigmoid(self.done_head(trunk).squeeze(-1)),
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

    def PredictNextVisualFromPosterior(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor,
        sample: bool = False,) -> Dict[str, Any]:
        rollout = self.PriorRolloutFromStateAction(
            hPrev=h,
            zPrev=z,
            s4xPrev=s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            robotSelfState=robotSelfState,
            sample=sample,)
        pred = self.BuildPredictedVisual(rollout["s_next"])
        pred["prior_rollout"] = rollout
        return pred

    def BuildWorldAbstract(
        self,
        worldOut: Dict[str, torch.Tensor],
        physicalState: Dict[str, torch.Tensor],
        pstSummary: torch.Tensor,
        uncertainty: torch.Tensor,
        confidence: torch.Tensor,) -> Dict[str, torch.Tensor]:
        world_hzx = torch.cat([
            worldOut["h_next"],
            worldOut["z_next"],
            worldOut["x_next"],
        ], dim=-1)
        pst_context = worldOut["pst_binding"]["pst_context"]
        robot_world_context = worldOut["pst_binding"]["robot_world_context"]
        scalar = torch.stack([
            worldOut["r_pred"],
            worldOut["d_prob"],
            uncertainty.view(-1),
            confidence.view(-1),
        ], dim=-1)
        abstract_feat = self.world_abstract_projector(torch.cat([
            world_hzx,
            worldOut["s_next"],
            pstSummary,
            pst_context,
            robot_world_context,
            scalar,
        ], dim=-1))
        return {
            "world_hzx": world_hzx,
            "world_state": worldOut["s_next"],
            "pst_summary": pstSummary,
            "pst_context": pst_context,
            "robot_world_context": robot_world_context,
            "slot_binding_weight": worldOut["pst_binding"]["slot_binding_weight"],
            "reward_pred": worldOut["r_pred"],
            "done_prob": worldOut["d_prob"],
            "uncertainty": uncertainty,
            "confidence": confidence,
            "abstract_feat": abstract_feat,
            "slot_presence_mask": physicalState["SlotPresence"],
            "physical_entity_mask": physicalState["SlotPresence"] * physicalState["MphysRaw"],}

    @torch.no_grad()
    def ScoreDecisionImaginations(
        self,
        h0: torch.Tensor,
        z0: torch.Tensor,
        x0: torch.Tensor,
        actionEncCandidates: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        robotSelfState: torch.Tensor,
        gamma: float = 0.99,) -> Dict[str, torch.Tensor]:
        B, N, T, A = actionEncCandidates.shape
        if T != 1:
            raise ValueError(
                "ScoreDecisionImaginations currently supports T=1 only; "
                "multi-step rollout requires a predicted PST/robot transition")
        h = h0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
        z = z0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
        x = x0.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
        physical_state_candidates = {
            k: v.unsqueeze(1).expand(B, N, *v.shape[1:]).reshape(B * N, *v.shape[1:]).contiguous()
            for k, v in physicalState.items()
        }
        robot_self_candidates = robotSelfState.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1).contiguous()
        score = actionEncCandidates.new_zeros(B, N)
        cont = actionEncCandidates.new_ones(B, N)
        for t in range(T):
            prior = self.StepPriorOnly(
                h,
                z,
                x,
                actionEncCandidates[:, :, t].reshape(B * N, A),
                physicalState=physical_state_candidates,
                robotSelfState=robot_self_candidates,
                sample=False,)
            h, z, x = prior["h_next"], prior["z_next"], prior["x_next"]
            score = score + cont * ((float(gamma) ** t) * prior["r_pred"].view(B, N))
            cont = cont * (1.0 - prior["d_prob"].view(B, N))
        return {
            "score": score,
            "continue_prob": cont,
            "terminal_h": h.view(B, N, -1),
            "terminal_z": z.view(B, N, -1),
            "terminal_x": x.view(B, N, -1),}

    def NsProjectProbs(self, P: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
        return self.ns_struct.ProjectTrain(P, temp=temp)

    @torch.no_grad()
    def NsProjectRuntime(self, P: torch.Tensor, *, aloTau: float = 0.60, implAlpha: float = 1.0, temp: float = 1.0):
        return self.ns_struct.ProjectRuntime(P, aloTau=aloTau, implAlpha=implAlpha, temp=temp)

    def NsConfidence(self, P: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        P = P.clamp(eps, 1 - eps) # [B,K]
        H = -(P * torch.log(P) + (1 - P) * torch.log(1 - P)) # [B,K]
        Hmax = P.new_tensor(0.6931471805599453) # ln(2)
        conf = (1.0 - H / Hmax).clamp(0.0, 1.0) # [B,K]
        return conf

    def NsLogicLosses(self, probs: torch.Tensor):
        loss, stats = self.ns_struct.LogicLosses(
            probs,
            lambdaExcl=self.ns_lambda_excl,
            lambdaAlo=self.ns_lambda_alo,
            lambdaImpl=self.ns_lambda_impl,
            aloTau=0.6,)
        
        return loss, stats

    @torch.no_grad()
    def StepPriorOnly(
        self,
        hPrev: torch.Tensor, # deterministic state
        zPrev: torch.Tensor, # stochastic state
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        robotSelfState: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:

        B = actionEnc.size(0)
        device, dtype = self.device, self.dtype

        if hPrev is None or zPrev is None or s4xPrev is None:
            hPrev = torch.zeros(B, self.deter_dim, device=device, dtype=dtype)
            zPrev = torch.zeros(B, self.stoch_dim, device=device, dtype=dtype)
            s4xPrev = torch.zeros(B, self.ssm_dim, device=device, dtype=dtype)

        embodied_action, robot_world_context = self.BuildEmbodiedAction(physicalState, actionEnc, robotSelfState)
        a_t = self.act_proj(embodied_action)
        h_next, s4x_next = self.s4.StepWithX(zPrev, a_t, s4xPrev) # h_next: [B, deterDim], s4x_next: [B, ssmDim]

        mu_p, logstd_p = self.prior_net(h_next).chunk(2, dim=-1) # [B, stochDim]
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_prior(
                h_next, deterministic=True, updateAux=False) # [B,K]
            P_raw = torch.sigmoid(ns_logits) # [B,K]
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # Q:[B,K], pen:[B]
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True) # [B,1]

            dmu = self.ns_to_delta_mu(Q) # [B, stochDim]

            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_next, dmu], dim=-1))) # [B, stochDim]
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf) # [B,1]

            gate = (base_gate * gate_scale).clamp(0.0, 1.0) # [B, stochDim]

            mu_p = mu_p + gate * dmu # [B, stochDim]

        mu_p_raw = mu_p
        pst_binding = self.BindPhysicalMu(h_next, mu_p, s4x_next, physicalState, embodied_action, robot_world_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            logstd_p = logstd_p.clamp(-7.0, 2.0) 
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p # [B, stochDim]

        s_base = self.state_proj(torch.cat([h_next, z_next], dim=-1)) # [B, stateDim]
        s_prev_base = self.state_proj(torch.cat([hPrev, zPrev], dim=-1)) # [B, stateDim]

        A_t = self.conn(s_prev_base, a_t) # [B, stateDim, stateDim]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B, stateDim]

        h_phys, _, _ = self.phys_refiner(hPrev, a_t, h_next)
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1))

        d_tr = s_transport - s_base # [B,stateDim]
        d_ph = s_phys - s_base # [B,stateDim]
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3 * stateDim]

        logits = self.mix_gate(g_in) # [B,3]
        w = F.softmax(logits, dim=-1) # [B,3]
        s_next = w[:,0:1]*s_base + w[:,1:2]*s_transport + w[:,2:3]*s_phys # [B,stateDim]

        inp = torch.cat([s_base, s_next, a_t], dim=-1) # [B, 2 * stateDim + stochDim]
        h = self.RewardDoneTrunk(inp, deterministic=True)

        r_pred = self.BoundReward(self.rew_head(h).squeeze(-1)) # [B]
        d_logit = self.done_head(h).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]

        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": s4x_next,
            "s_next": s_next,
            "embodied_action": embodied_action,
            "robot_world_context": robot_world_context,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}


    def StepPosterior(
        self,
        visionIn: torch.Tensor,
        actionEnc: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        robotSelfState: torch.Tensor,
        sample: bool = False,  # False: Deterministic Forward, True: Reparameterized sampling with noise(More exploratory)
        ) -> Dict[str, torch.Tensor]:

        B = int(visionIn.size(0))
        device, dtype = self.device, self.dtype
        self.EnsureB(B, device, dtype)

        raw_e = self.obs_enc(visionIn) # [B, stochDim]
        embodied_action, robot_world_context = self.BuildEmbodiedAction(physicalState, actionEnc, robotSelfState)
        a_t = self.act_proj(embodied_action) # [B, stochDim]
        key = self.key_emb(raw_e, a_t) # [B, stochDim]

        h_pred = self.s4.Step(self._z, a_t, updateState=True) # [B, deterDim]
        x_next = self.s4.x # [B, ssmDim]

        mu_q, logstd_q = self.post_net(torch.cat([h_pred, raw_e], dim=-1)).chunk(2, dim=-1) # [B,stochDim]
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        if self._ns_enabled:
            ns_logits = self.ns_head_post(torch.cat([h_pred, raw_e], dim=-1)) # [B,K]
            P_raw = torch.sigmoid(ns_logits) # [B,K]
            Q, pen = self.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # Q:[B,K], pen:[B]
            conf = self.NsConfidence(Q).mean(dim=-1, keepdim=True) # [B,1]

            dmu = self.ns_to_delta_mu(Q) # [B, stochDim]

            base_gate = torch.sigmoid(self.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu], dim=-1))) # [B, stochDim]
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf) # [B,1]

            gate = (base_gate * gate_scale).clamp(0.0, 1.0) # [B, stochDim]

            mu_q = mu_q + gate * dmu # [B, stochDim]

        mu_q_raw = mu_q
        pst_binding = self.BindPhysicalMu(h_pred, mu_q, x_next, physicalState, embodied_action, robot_world_context)
        mu_q = pst_binding["bound_mu"]

        if sample:
            z_next = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
        else:
            z_next = mu_q

        s_base = self.state_proj(torch.cat([h_pred, z_next], dim=-1)) # [B, stateDim]
        s_prev_base = self.state_proj(torch.cat([self._h, self._z], dim=-1)) # [B, stateDim]

        A_t = self.conn(s_prev_base, a_t) # [B,S,S]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B,S]

        h_phys, _, _ = self.phys_refiner(self._h, a_t, h_pred) # [B,D]
        s_phys = self.state_proj(torch.cat([h_phys, z_next], dim=-1)) # [B,S]

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3S]
        logits = self.mix_gate(g_in) # [B,3]
        w = F.softmax(logits, dim=-1) # [B,3]
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        dynamics_state = s_next
        inp = torch.cat([s_base, dynamics_state, a_t], dim=-1)
        h = self.rdone_trunk(self.rdone_ln(inp))
        r_pred = self.BoundReward(self.rew_head(h).squeeze(-1)) # [B]
        d_logit = self.done_head(h).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]

        if self._use_memory:
            with torch.no_grad():
                mem_retrieved = self.MemRetrieve(key)
                r_score = torch.tanh(r_pred.detach().abs()).clamp(0.0, 1.0)
                d_score = d_prob.detach().clamp(0.0, 1.0)

                if self._ns_enabled:
                    conf_scalar = self.NsConfidence(Q).mean(dim=-1) # [B]
                    imp_ns = ((1.0 - pen).clamp(0.0, 1.0) * (0.5 + 0.5 * conf_scalar)).clamp(0.0, 1.0)
                else:
                    imp_ns = torch.full((B,), 0.5, device=device, dtype=dtype)

                imp = (0.60 * imp_ns + 0.25 * r_score + 0.15 * d_score).clamp(0.0, 1.0)
                self.MemAdd(key.detach(), dynamics_state.detach(), imp.detach())

                if mem_retrieved is not None:
                    mem_s, mem_mask = mem_retrieved
                    s_memory = self.state_state_film(dynamics_state, mem_s)
                    s_next = torch.where(mem_mask.view(B, 1), s_memory, dynamics_state)

        out: Dict[str, torch.Tensor] = {
            "h_next": h_pred,
            "z_next": z_next,
            "z_next_raw": mu_q_raw,
            "x_next": x_next,
            "s_next": s_next,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "mu_q": mu_q,
            "mu_q_raw": mu_q_raw,
            "logstd_q": logstd_q,
            "embodied_action": embodied_action,
            "robot_world_context": robot_world_context,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

        if self._ns_enabled:
            out["ns_logits"] = ns_logits
            out["ns_Q"] = Q
            out["ns_pen"] = pen

        if self.use_decoder:
            out["recon"] = self.obs_dec(s_next)
            out["recon_target"] = visionIn

        self._h = h_pred.detach()
        self._z = z_next.detach()

        return out


    def ForwardTrain(
        self,
        visionIn: torch.Tensor, # [B, visionDim]
        physicalState: Dict[str, torch.Tensor],
        reward: Optional[torch.Tensor] = None, # [B]
        done: Optional[torch.Tensor] = None, # [B]
        *,
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor,
        sample: Optional[bool] = None,
        updateMemory: Optional[bool] = None,
        alphaKl: float = 0.8,
        freeNats: float = 1.0,
        reconCoef: float = 1.0,
        rewardCoef: float = 1.0,
        doneCoef: float = 1.0,
        nsCoef: float = 1.0,
        nsDistillCoef: float = 1e-2,
        nsPriorLogicCoef: float = 1e-3,
        physCoef: float = 1e-4,
        pstBindCoef: float = 0.05,) -> Dict[str, torch.Tensor]:

        B = visionIn.size(0)
        device, dtype = self.device, self.dtype
        self.EnsureB(B, device, dtype)
        sample = bool(self.training) if sample is None else bool(sample)
        update_memory = bool(self.training) if updateMemory is None else bool(updateMemory)
        update_memory = update_memory and bool(self.training)

        h0 = self._h
        z0 = self._z

        a_enc = actionEnc
        embodied_action, robot_world_context = self.BuildEmbodiedAction(physicalState, a_enc, robotSelfState)
        a_t = self.act_proj(embodied_action) # [B, stochDim]

        h_pred = self.s4.Step(z0, a_t) # [B,D]

        mu_p, logstd_p = self.prior_net(h_pred).chunk(2, dim=-1) # [B,stochDim]
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        logits_pr = None
        P_pr_train = None
        ns_prior_logic = visionIn.new_tensor(0.0)

        if self._ns_enabled:
            logits_pr = self.ns_head_prior(h_pred) # [B,K]
            P_pr_raw = torch.sigmoid(logits_pr) # [B,K]
            P_pr_train = self.NsProjectProbs(P_pr_raw) # [B,K] 

            dmu_p = self.ns_to_delta_mu(P_pr_train) # [B,stochDim]
            base_gate = torch.sigmoid(self.ns_gate_mu(torch.cat([h_pred, dmu_p], dim=-1)))  # [B,stochDim]

            _, pen_pr = self.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)  # [B]

            conf = self.NsConfidence(P_pr_train).mean(dim=-1, keepdim=True)  # [B,1]
            gate_scale = (1.0 - 0.40 * pen_pr.view(-1, 1)) * (0.6 + 0.4 * conf) # [B,1]
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)  # [B,stochDim]

            mu_p = mu_p + gate * dmu_p

            if nsPriorLogicCoef > 0.0:
                ns_prior_logic, _ = self.NsLogicLosses(P_pr_train)

        mu_p_raw = mu_p
        pst_binding_prior = self.BindPhysicalMu(h_pred, mu_p, self.s4.x, physicalState, embodied_action, robot_world_context)
        mu_p = pst_binding_prior["bound_mu"]

        raw_e = self.obs_enc(visionIn) # [B,stochDim]

        mu_q, logstd_q = self.post_net(torch.cat([h_pred, raw_e], dim=-1)).chunk(2, dim=-1) # [B,stochDim]
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        ns_loss = visionIn.new_tensor(0.0)
        ns_distill = visionIn.new_tensor(0.0)
        logits_q = None
        Q_train = None
        pen_q = None

        if self._ns_enabled:
            logits_q = self.ns_head_post(torch.cat([h_pred, raw_e], dim=-1)) # [B,K]
            P_q_raw = torch.sigmoid(logits_q) # [B,K]
            Q_train = self.NsProjectProbs(P_q_raw) # [B,K]

            dmu_q = self.ns_to_delta_mu(Q_train) # [B,stochDim]
            base_gate_q = torch.sigmoid(self.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu_q], dim=-1))) # [B,stochDim]

            _, pen_q = self.NsProjectRuntime(P_q_raw, aloTau=0.60, implAlpha=1.0, temp=1.0) # [B]

            conf_q = self.NsConfidence(Q_train).mean(dim=-1, keepdim=True) # [B,1]
            gate_scale_q = (1.0 - 0.40 * pen_q.view(-1, 1)) * (0.6 + 0.4 * conf_q)
            gate_q = (base_gate_q * gate_scale_q).clamp(0.0, 1.0)

            mu_q = mu_q + gate_q * dmu_q

            ns_loss, _ = self.NsLogicLosses(Q_train)

            if (logits_pr is not None) and (nsDistillCoef > 0.0):
                with torch.no_grad():
                    P_teacher = torch.sigmoid(logits_q) # [B,K]
                ns_distill = F.binary_cross_entropy_with_logits(logits_pr, P_teacher, reduction="mean")

        mu_q_raw = mu_q
        pst_binding_posterior = self.BindPhysicalMu(h_pred, mu_q, self.s4.x, physicalState, embodied_action, robot_world_context)
        mu_q = pst_binding_posterior["bound_mu"]

        if sample:
            z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q) # [B,stochDim]
        else:
            z1 = mu_q

        s_base = self.state_proj(torch.cat([h_pred, z1], dim=-1)) # [B,S]
        s_prev_base = self.state_proj(torch.cat([h0, z0], dim=-1)) # [B,S]

        A_t = self.conn(s_prev_base, a_t) # [B,S,S]
        s_transport = self.conn.TransportApply(A_t, s_prev_base) # [B,S]

        prevA = self._A_prev if (self._A_prev is not None and self._A_prev.shape == A_t.shape) else None
        reg_A = self.conn.ComputeGeomReg(A_t, prevA)
        self._A_prev = A_t.detach()

        h_phys, phys_loss, _ = self.phys_refiner(h0, a_t, h_pred) # h_phys:[B,D]
        if phys_loss is None:
            phys_loss = visionIn.new_zeros(())
        s_phys = self.state_proj(torch.cat([h_phys, z1], dim=-1)) # [B,S]

        d_tr = s_transport - s_base # [B,S]
        d_ph = s_phys - s_base # [B,S] 
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1) # [B,3S]
        w = F.softmax(self.mix_gate(g_in), dim=-1) # [B,3]
        s1 = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys # [B,S]

        dynamics_state = s1
        inp = torch.cat([s_base, dynamics_state, a_t], dim=-1) # [B,2S+stochDim]
        trunk = self.rdone_trunk(self.rdone_ln(inp)) # [B,256]
        r_pred = self.BoundReward(self.rew_head(trunk).squeeze(-1)) # [B]
        d_logit = self.done_head(trunk).squeeze(-1) # [B]
        d_prob = torch.sigmoid(d_logit) # [B]

        if self._use_memory:
            key = self.key_emb(raw_e, a_t) # [B,stochDim]
            mem_retrieved = self.MemRetrieve(key, updateImportance=update_memory)
            if update_memory:
                r_score = torch.tanh(r_pred.detach().abs()).clamp(0.0, 1.0) # [B]
                d_score = d_prob.detach().clamp(0.0, 1.0) # [B]
                if self._ns_enabled and (Q_train is not None) and (pen_q is not None):
                    conf_scalar = self.NsConfidence(Q_train.detach()).mean(dim=-1) # [B]
                    imp_ns = ((1.0 - pen_q.detach()).clamp(0.0, 1.0) * (0.5 + 0.5 * conf_scalar)).clamp(0.0, 1.0)
                else:
                    imp_ns = visionIn.new_full((B,), 0.5)
                imp = (0.60 * imp_ns + 0.25 * r_score + 0.15 * d_score).clamp(0.0, 1.0)
                self.MemAdd(key.detach(), dynamics_state.detach(), imp.detach())

            if mem_retrieved is not None:
                mem_s, mem_mask = mem_retrieved
                s_memory = self.state_state_film(dynamics_state, mem_s)
                s1 = torch.where(mem_mask.view(B, 1), s_memory, dynamics_state)

        loss_recon = visionIn.new_tensor(0.0)
        recon_error = visionIn.new_zeros(B)
        recon = None
        if self.use_decoder:
            recon = self.obs_dec(s1)  # [B, visionDim]

            target = self.obs_enc[0](visionIn)  # nn.LayerNorm(visionDim)

            recon_n = F.layer_norm(
                recon,
                normalized_shape=(int(recon.size(-1)),),
                weight=self.obs_enc[0].weight,
                bias=self.obs_enc[0].bias,
                eps=self.obs_enc[0].eps,)

            recon_error = (recon_n - target).pow(2).mean(dim=-1)
            loss_recon = recon_error.mean()

        aux_moe = visionIn.new_tensor(0.0)
        if self._ns_enabled:
            aux_moe = self.ns_head_prior.GetAuxLoss() + self.ns_head_post.GetAuxLoss()

        if reward is None:
            loss_reward = visionIn.new_zeros(())
        else:
            reward_target = reward.view(B).clamp(float(self.reward_min), float(self.reward_max))
            loss_reward = F.mse_loss(r_pred, reward_target, reduction="mean")
        if done is None:
            loss_done = visionIn.new_zeros(())
        else:
            loss_done = F.binary_cross_entropy_with_logits(d_logit, done.view(B).to(d_logit.dtype), reduction="mean")

        loss_kl = BalancedKL(mu_q, logstd_q, mu_p, logstd_p, alpha=alphaKl, freeNats=freeNats).mean()
        loss_pst_bind = 0.5 * (
            pst_binding_prior["loss_pst_bind"]
            + pst_binding_posterior["loss_pst_bind"])

        self._h = h_pred.detach()
        self._z = z1.detach()

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic
            + physCoef * phys_loss
            + pstBindCoef * loss_pst_bind
            + reg_A
            + 1e-1 * aux_moe)

        out: Dict[str, Any] = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "loss_phys": phys_loss,
            "loss_pst_bind": loss_pst_bind,
            "loss_conn_reg": reg_A,

            "h_next": h_pred,
            "z_next": z1,
            "z_next_raw": mu_q_raw,
            "x_next": self.s4.x,
            "s_next": s1,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "recon_error": recon_error,
            "mu_p": mu_p,
            "mu_p_raw": mu_p_raw,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "mu_q_raw": mu_q_raw,
            "logstd_q": logstd_q,
            "action_enc": a_enc,
            "embodied_action": embodied_action,
            "robot_world_context": robot_world_context,
            "pst_binding": pst_binding_posterior,
            "pst_binding_prior": pst_binding_prior,
            "pst_binding_posterior": pst_binding_posterior,}

        if self.use_decoder and recon is not None:
            out["recon"] = recon
            out["recon_target"] = visionIn

        return out


    def ExportWorldMemoryBank(self, topk: int = 1024, onlyVals: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        if (not getattr(self, "_use_memory", False)):
            return None

        K = int(topk)
        if K <= 0:
            return None

        B = int(self._mem_vals.size(0))
        cap = int(self._mem_vals.size(1))

        filled = self._mem_size # [B] 
        if (filled <= 0).all():
            return None

        K = min(K, cap)
        K = min(K, int(filled.min().item()))
        if K <= 0:
            return None

        ar = torch.arange(cap, device=self._mem_vals.device).view(1, cap) # [1,cap]
        valid = ar < filled.view(B, 1) # [B,cap]
        scores = self._mem_imp.masked_fill(~valid, -1e9) # [B,cap]

        _, idx = torch.topk(scores, k=K, dim=-1) # [B,K]
        sel_steps = torch.gather(self._mem_steps, 1, idx) # [B,K]
        time_order = torch.argsort(sel_steps, dim=-1, descending=True)
        idx = torch.gather(idx, 1, time_order)

        out: Dict[str, torch.Tensor] = {} 

        Dv = int(self._mem_vals.size(-1))
        out["vals"] = torch.gather(self._mem_vals, 1, idx.unsqueeze(-1).expand(B, K, Dv)).contiguous()

        if onlyVals:
            return out

        out["size"] = filled.detach().clone() # [B]
        out["idx"]  = idx.contiguous() # [B,K]
        out["steps"] = torch.gather(self._mem_steps, 1, idx).contiguous() # [B,K]

        Dk = int(self._mem_keys.size(-1))
        out["keys"] = torch.gather(self._mem_keys, 1, idx.unsqueeze(-1).expand(B, K, Dk)).contiguous()

        out["imp"]  = torch.gather(self._mem_imp, 1, idx).contiguous() # [B,K]
        return out
    


class WorldOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: nn.Module,
        initRankEach: int = 0,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        *,
        maxRank: int = 64,
        maxRankSmall: int = 16,
        maxRankHuge: int = 8,
        maxRankConnHeadUV: int = 16,
        maxRankConnHeadFull: int = 8,):
        self.maxRank = int(maxRank)
        self.maxRankSmall = int(maxRankSmall)
        self.maxRankHuge = int(maxRankHuge)
        self.maxRankConnHeadUV = int(maxRankConnHeadUV)
        self.maxRankConnHeadFull = int(maxRankConnHeadFull)
        super().__init__(
            base,
            initRankEach=initRankEach,
            autoRank=autoRank,
            evThreshold=evThreshold,
            gradEma=gradEma,)
        # These embodiment-facing terminal heads adapt directly online. They are deliberately
        # outside the candidate LoRA rollback contract; the rest of the base remains frozen.
        self.RestoreBaseTrainabilityAfterCommit()

    def DirectOnlineHeads(self) -> Tuple[nn.Module, ...]:
        return (
            self.base.robot_world_relation.pair_score,
            self.base.robot_world_relation.slot_score,
            self.base.robot_world_relation.relation_proj[3],
            self.base.embodied_action_proj[3],
            self.base.pst_binder.delta_mu[1],
            self.base.pst_binder.bind_gate[1],)

    def RestoreBaseTrainabilityAfterCommit(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for head in self.DirectOnlineHeads():
            for parameter in head.parameters():
                parameter.requires_grad_(True)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        V = int(self.base.vision_dim)
        A = int(self.base.action_dim)
        D = int(self.base.deter_dim)
        Z = int(self.base.stoch_dim)
        S = int(self.base.state_dim)
        X = int(self.base.ssm_dim)

        conn = self.base.conn
        phys = self.base.phys_refiner
        key = self.base.key_emb
        film = self.base.state_state_film

        def mk(name: str, inDim: int, outDim: int, maxRank: int) -> SiteSpec:
            inDim_i, outDim_i, maxRank_i = int(inDim), int(outDim), int(maxRank)

            def alloc(addRank: int, device: torch.device, dtype: torch.dtype):
                A_ = nn.Parameter(torch.randn(addRank, inDim_i, device=device, dtype=dtype) * 1e-4) # [r,in]
                B_ = nn.Parameter(torch.zeros(outDim_i, addRank, device=device, dtype=dtype) * 1e-4) # [out,r]
                s_ = nn.Parameter(torch.tensor(1e-2, device=device, dtype=dtype))
                return A_, B_, s_

            def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
                s_eff = torch.tanh(s) * GetParametersScale(s)
                return s_eff * (b @ a) # [out,in]

            return SiteSpec(name, 1, inDim_i, outDim_i, maxRank_i, alloc, compose)

        specs: Dict[str, SiteSpec] = {}

        specs["obs_enc0"] = mk("obs_enc0", V, S, self.maxRank)
        specs["obs_enc1"] = mk("obs_enc1", S, Z, self.maxRank)

        specs["act_proj"] = mk("act_proj", A, Z, self.maxRank)

        specs["s4_in_to_ssm"] = mk("s4_in_to_ssm", 2 * Z, X, self.maxRank)
        specs["s4_ssm_to_deter"] = mk("s4_ssm_to_deter", X, D, self.maxRank)
        specs["s4_in_to_deter"] = mk("s4_in_to_deter", 2 * Z, D, self.maxRank)
        specs["s4_gate"] = mk("s4_gate", 2 * Z, X, self.maxRank)
        specs["s4_out_gate"] = mk("s4_out_gate", X, D, self.maxRank)

        specs["prior"] = mk("prior", D, 2 * Z, self.maxRank)
        specs["post"] = mk("post", D + Z, 2 * Z, self.maxRank)
        specs["state_proj"] = mk("state_proj", D + Z, S, self.maxRank)

        specs["rdone0"] = mk("rdone0", 2 * S + Z, 512, self.maxRank)
        specs["rdone1"] = mk("rdone1", 512, 256, self.maxRank)
        specs["rew"] = mk("rew", 256, 1, self.maxRankSmall)
        specs["done"] = mk("done", 256, 1, self.maxRankSmall)

        specs["obs_dec0"] = mk("obs_dec0", S, S, self.maxRank)
        specs["obs_dec1"] = mk("obs_dec1", S, V, self.maxRank)

        specs["mix_gate"] = mk("mix_gate", 3 * S, 3, self.maxRankSmall)

        specs["key_to_gb"] = mk("key_to_gb", Z, 2 * Z, self.maxRank)
        specs["key_mlp1"] = mk("key_mlp1", 4 * Z, int(key.mlp1.out_f), self.maxRank)
        specs["key_mlp2"] = mk("key_mlp2", int(key.mlp2.in_f), int(key.mlp2.out_f), self.maxRank)

        specs["ssfilm_e_to_gb"] = mk("ssfilm_e_to_gb", int(film.e_to_gb.in_f), int(film.e_to_gb.out_f), self.maxRank)
        specs["ssfilm_e_to_h"] = mk("ssfilm_e_to_h", int(film.e_to_h.in_f), int(film.e_to_h.out_f), self.maxRank)
        specs["ssfilm_delta0"] = mk("ssfilm_delta0", int(film.delta_mlp[0].in_f), int(film.delta_mlp[0].out_f), self.maxRank)
        specs["ssfilm_delta1"] = mk("ssfilm_delta1", int(film.delta_mlp[3].in_f), int(film.delta_mlp[3].out_f), self.maxRank)
        specs["ssfilm_to_gate"] = mk("ssfilm_to_gate", int(film.to_gate.in_f), int(film.to_gate.out_f), self.maxRank)

        specs["conn_enc_s"] = mk("conn_enc_s", int(conn.enc_s[1].linear.in_f), int(conn.enc_s[1].linear.out_f), self.maxRank)
        specs["conn_enc_a"] = mk("conn_enc_a", int(conn.enc_a[1].linear.in_f), int(conn.enc_a[1].linear.out_f), self.maxRank)

        specs["conn_film_gamma_a"] = mk("conn_film_gamma_a", int(conn.film_gamma_a.linear.in_f), int(conn.film_gamma_a.linear.out_f), self.maxRank)
        specs["conn_film_beta_a"] = mk("conn_film_beta_a", int(conn.film_beta_a.linear.in_f), int(conn.film_beta_a.linear.out_f), self.maxRank)

        for i, blk in enumerate(conn.blocks):
            specs[f"conn_blk{i}_ff0"] = mk(f"conn_blk{i}_ff0", int(blk.ff[0].linear.in_f), int(blk.ff[0].linear.out_f), self.maxRank)
            specs[f"conn_blk{i}_ff1"] = mk(f"conn_blk{i}_ff1", int(blk.ff[3].linear.in_f), int(blk.ff[3].linear.out_f), self.maxRank)

        if conn.use_lowrank:
            specs["conn_head_uv"] = mk("conn_head_uv", int(conn.head_uv.linear.in_f), int(conn.head_uv.linear.out_f), self.maxRankConnHeadUV)
        if conn.use_full:
            specs["conn_head_full"] = mk("conn_head_full", int(conn.head_full.linear.in_f), int(conn.head_full.linear.out_f), self.maxRankConnHeadFull)

        specs["conn_mix"] = mk("conn_mix", int(conn.mix.linear.in_f), int(conn.mix.linear.out_f), self.maxRankSmall)

        specs["phys_to_qp"] = mk("phys_to_qp", int(phys.to_qp.in_f), int(phys.to_qp.out_f), self.maxRank)
        specs["phys_from_qp"] = mk("phys_from_qp", int(phys.from_qp.in_f), int(phys.from_qp.out_f), self.maxRank)

        specs["phys_H0"] = mk("phys_H0", int(phys.H_net[0].in_f), int(phys.H_net[0].out_f), self.maxRank)
        specs["phys_H1"] = mk("phys_H1", int(phys.H_net[2].in_f), int(phys.H_net[2].out_f), self.maxRank)
        specs["phys_H2"] = mk("phys_H2", int(phys.H_net[4].in_f), int(phys.H_net[4].out_f), self.maxRankSmall)

        specs["phys_force0"] = mk("phys_force0", int(phys.force_net[0].in_f), int(phys.force_net[0].out_f), self.maxRank)
        specs["phys_force1"] = mk("phys_force1", int(phys.force_net[2].in_f), int(phys.force_net[2].out_f), self.maxRank)

        specs["phys_g_force"] = mk("phys_g_force", int(phys.g_force.in_f), int(phys.g_force.out_f), self.maxRank)
        specs["phys_g_phys"] = mk("phys_g_phys", int(phys.g_phys.in_f), int(phys.g_phys.out_f), self.maxRank)
        specs["phys_g_fuse"] = mk("phys_g_fuse", int(phys.g_fuse.in_f), int(phys.g_fuse.out_f), self.maxRank)

        return specs

    def EffW(self, gll, d2: Optional[torch.Tensor]) -> torch.Tensor:
        W = gll.target.weight
        d0 = gll.DeltaWeight()
        if d0 is not None:
            W = W + d0
        if d2 is not None:
            W = W + d2
        return W

    def Lin(self, x: torch.Tensor, gll, d2: Optional[torch.Tensor]) -> torch.Tensor:
        return F.linear(x, self.EffW(gll, d2), gll.target.bias)


    def ObsEnc(self, visionIn: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.base.obs_enc[0](visionIn)
        x = self.Lin(x, self.base.obs_enc[1], d.get("obs_enc0"))
        x = self.base.obs_enc[2](x)
        x = self.base.obs_enc[3](x)
        x = self.Lin(x, self.base.obs_enc[4], d.get("obs_enc1"))
        return x # [B,Z]

    def ActProj(self, actionEnc: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.Lin(actionEnc, self.base.act_proj[0], d.get("act_proj"))
        x = self.base.act_proj[1](x)
        x = self.base.act_proj[2](x)
        return x # [B,Z]

    def Prior(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(h, self.base.prior_net[0], d.get("prior"))

    def Post(self, hz: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(hz, self.base.post_net[0], d.get("post"))

    def StateProj(self, hz: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.base.state_proj[0](hz)
        x = self.Lin(x, self.base.state_proj[1], d.get("state_proj"))
        x = self.base.state_proj[2](x)
        return x  # [B,S]

    def RdoneTrunk(
        self,
        inp: torch.Tensor,
        d: Dict[str, Optional[torch.Tensor]],
        *,
        deterministic: bool = False,
        ) -> torch.Tensor:
        x = self.Lin(inp, self.base.rdone_trunk[0], d.get("rdone0"))
        x = self.base.rdone_trunk[1](x)
        if deterministic:
            x = F.dropout(x, p=float(self.base.rdone_trunk[2].p), training=False)
        else:
            x = self.base.rdone_trunk[2](x)
        x = self.Lin(x, self.base.rdone_trunk[3], d.get("rdone1"))
        x = self.base.rdone_trunk[4](x)
        return x  # [B,256]

    def Rew(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.base.BoundReward(self.Lin(h, self.base.rew_head[0], d.get("rew")))

    def Done(self, h: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(h, self.base.done_head[0], d.get("done"))

    def ObsDec(self, s: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        x = self.Lin(s, self.base.obs_dec[0], d.get("obs_dec0"))
        x = self.base.obs_dec[1](x)
        x = self.Lin(x, self.base.obs_dec[2], d.get("obs_dec1"))
        return x

    def MixGate(self, g_in: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        return self.Lin(g_in, self.base.mix_gate[0], d.get("mix_gate"))


    def S4_Step(self, zPrev: torch.Tensor, a_t: torch.Tensor, *, updateState: bool, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        s4 = self.base.s4
        u = torch.cat([zPrev, a_t], dim=-1)  # [B,2Z]

        g = torch.sigmoid(self.Lin(u, s4.gate, d.get("s4_gate"))) # [B,X]
        Bu = self.Lin(u, s4.in_to_ssm, d.get("s4_in_to_ssm")) * g # [B,X]

        x_next = s4.CayleyStep(s4.theta, s4.x, Bu, s4.dt)
        y_lin = self.Lin(x_next, s4.ssm_to_deter, d.get("s4_ssm_to_deter")) + self.Lin(u, s4.in_to_deter, d.get("s4_in_to_deter"))
        y_glu = y_lin * torch.sigmoid(self.Lin(x_next, s4.out_gate, d.get("s4_out_gate")))
        y = s4.ln_y(y_glu)
        y = y + s4.ffn(s4.ln_ffn(y))

        if updateState:
            s4.x = x_next.detach()
        return y # [B,D]

    def S4StepWithX(self, zPrev: torch.Tensor, a_t: torch.Tensor, x: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        s4 = self.base.s4
        u = torch.cat([zPrev, a_t], dim=-1)

        g = torch.sigmoid(self.Lin(u, s4.gate, d.get("s4_gate")))
        Bu = self.Lin(u, s4.in_to_ssm, d.get("s4_in_to_ssm")) * g

        x_next = s4.CayleyStep(s4.theta, x, Bu, s4.dt)
        y_lin = self.Lin(x_next, s4.ssm_to_deter, d.get("s4_ssm_to_deter")) + self.Lin(u, s4.in_to_deter, d.get("s4_in_to_deter"))
        y_glu = y_lin * torch.sigmoid(self.Lin(x_next, s4.out_gate, d.get("s4_out_gate")))
        y = s4.ln_y(y_glu)
        y = y + s4.ffn(s4.ln_ffn(y))
        return y, x_next.detach()


    def KeyEmbed(self, base_e: torch.Tensor, actionEmbed: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        ke = self.base.key_emb
        e = ke.ln_e(base_e)
        a = ke.ln_a(actionEmbed)

        gb = self.Lin(a, ke.to_gb, d.get("key_to_gb"))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = 0.10 * torch.tanh(gamma)
        beta = 0.10 * torch.tanh(beta)

        e_film = (1.0 + gamma) * e + beta

        feat = torch.cat([e_film, a, e_film * a, e_film - a], dim=-1)
        feat = ke.ln_feat(feat)

        h = F.silu(self.Lin(feat, ke.mlp1, d.get("key_mlp1")))
        h = ke.drop(h)
        k = self.Lin(h, ke.mlp2, d.get("key_mlp2"))

        k = F.normalize(k, dim=-1, eps=1e-6)
        return k


    def FiLMHResidual(self, h: torch.Tensor, e: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        fr = self.base.state_state_film
        h0 = fr.ln_h(h)
        e0 = fr.ln_e(e)

        gb = self.Lin(e0, fr.e_to_gb, d.get("ssfilm_e_to_gb"))
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = fr.film_scale * torch.tanh(gamma)
        beta = fr.film_scale * torch.tanh(beta)

        h_film = (1.0 + gamma) * h0 + beta

        e_h = self.Lin(e0, fr.e_to_h, d.get("ssfilm_e_to_h"))
        e_h = fr.film_scale * torch.tanh(e_h)

        feat = torch.cat([h_film, e_h, h_film * e_h, h_film - e_h], dim=-1)
        feat = fr.delta_ln(feat)

        y = self.Lin(feat, fr.delta_mlp[0], d.get("ssfilm_delta0"))
        y = fr.delta_mlp[1](y) # SiLU
        y = fr.delta_mlp[2](y) # Dropout
        delta = self.Lin(y, fr.delta_mlp[3], d.get("ssfilm_delta1"))

        gate_in = torch.cat([h_film, e_h], dim=-1)
        gate = torch.sigmoid(self.Lin(gate_in, fr.to_gate, d.get("ssfilm_to_gate")))

        h_out = h + gate * delta
        if fr.use_out_ln:
            h_out = fr.out_ln(h_out)
        return h_out

    def ConnNet(self, sBase: torch.Tensor, actPrev: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        conn = self.base.conn
        B = int(sBase.size(0))

        hs = conn.enc_s[0](sBase)
        hs = self.Lin(hs, conn.enc_s[1].linear, d.get("conn_enc_s"))
        hs = conn.enc_s[2](hs)

        ha = conn.enc_a[0](actPrev)
        ha = self.Lin(ha, conn.enc_a[1].linear, d.get("conn_enc_a"))
        ha = conn.enc_a[2](ha)

        g = torch.tanh(self.Lin(ha, conn.film_gamma_a.linear, d.get("conn_film_gamma_a")))
        b = self.Lin(ha, conn.film_beta_a.linear, d.get("conn_film_beta_a"))

        h = hs
        for i, blk in enumerate(conn.blocks):
            y = (1.0 + g) * h + b
            y = blk.ln(y)

            y = self.Lin(y, blk.ff[0].linear, d.get(f"conn_blk{i}_ff0"))
            y = blk.ff[1](y) 
            y = blk.ff[2](y) 
            y = self.Lin(y, blk.ff[3].linear, d.get(f"conn_blk{i}_ff1"))

            h = h + blk.alpha * y

        A_list: List[torch.Tensor] = []

        if conn.use_lowrank:
            uv = self.Lin(h, conn.head_uv.linear, d.get("conn_head_uv"))
            U, Vv = uv.split(conn.S * conn.r, dim=-1)
            U = U.view(B, conn.S, conn.r)
            Vv = Vv.view(B, conn.S, conn.r)
            A_list.append(U @ Vv.transpose(1, 2) - Vv @ U.transpose(1, 2))

        if conn.use_full:
            M = self.Lin(h, conn.head_full.linear, d.get("conn_head_full")).view(B, conn.S, conn.S)
            A_list.append(0.5 * (M - M.transpose(1, 2)))

        if not A_list:
            A = torch.zeros(B, conn.S, conn.S, device=sBase.device, dtype=sBase.dtype)
        elif len(A_list) == 1:
            A = A_list[0]
        else:
            w = F.softmax(self.Lin(h, conn.mix.linear, d.get("conn_mix")), dim=-1)
            A = w[:, :1].view(B, 1, 1) * A_list[0] + w[:, 1:2].view(B, 1, 1) * A_list[1]

        if conn.norm_clip and conn.norm_clip > 0:
            fro = A.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
            scale = torch.minimum(torch.ones_like(fro), conn.norm_clip / fro).view(B, 1, 1)
            A = A * scale

        return A # [B,S,S]


    def PhysRefiner(self, hPrev: torch.Tensor, action: torch.Tensor, hS4: torch.Tensor, d: Dict[str, Optional[torch.Tensor]]):
        pr = self.base.phys_refiner
        training_mode = bool(self.training and torch.is_grad_enabled())
        inference_mode_active = torch.is_inference_mode_enabled()
        create_graph = bool(training_mode)
        dt_sub = pr.dt / float(pr.substeps)

        if training_mode:
            H_start = None
            H_end = None
            work_acc = hPrev.new_zeros(hPrev.size(0), 1)
            smooth_acc = hPrev.new_tensor(0.0)

        def H_net(qp: torch.Tensor) -> torch.Tensor:
            x = self.Lin(qp, pr.H_net[0], d.get("phys_H0"))
            x = pr.H_net[1](x)  
            x = self.Lin(x, pr.H_net[2], d.get("phys_H1"))
            x = pr.H_net[3](x)  
            x = self.Lin(x, pr.H_net[4], d.get("phys_H2"))
            return x # [B,1]

        def Force_net(fa: torch.Tensor) -> torch.Tensor:
            x = self.Lin(fa, pr.force_net[0], d.get("phys_force0"))
            x = pr.force_net[1](x) 
            x = self.Lin(x, pr.force_net[2], d.get("phys_force1"))
            return x # [B,Q]

        def HAndGrad(qp: torch.Tensor, create_graph_: bool) -> Tuple[torch.Tensor, torch.Tensor]:
            H = H_net(qp) # [B,1]
            g = torch.autograd.grad(
                H.sum(), qp,
                create_graph=create_graph_,
                retain_graph=create_graph_,
                allow_unused=False,
            )[0]  # [B,P]
            return H, g

        def SymplecticLeapfrog(q: torch.Tensor, p: torch.Tensor, dt: float, create_graph_: bool):
            qp0 = torch.cat([q, p], dim=-1)
            H0, g0 = HAndGrad(qp0, create_graph_=create_graph_)
            dH_dq0, _ = g0.chunk(2, dim=-1)

            p_half = p - 0.5 * dt * dH_dq0

            qp_mid = torch.cat([q, p_half], dim=-1)
            _, gm = HAndGrad(qp_mid, create_graph_=create_graph_)
            _, dH_dp_mid = gm.chunk(2, dim=-1)

            q1 = q + dt * dH_dp_mid

            qp_for_p = torch.cat([q1, p_half], dim=-1)
            H1, g2 = HAndGrad(qp_for_p, create_graph_=create_graph_)
            dH_dq2, _ = g2.chunk(2, dim=-1)

            p1 = p_half - 0.5 * dt * dH_dq2
            return q1, p1, H0, H1, dH_dp_mid

        def ClampResidual(delta_: torch.Tensor, base_: torch.Tensor, ratio: float) -> torch.Tensor:
            eps = 1e-8
            dnorm = delta_.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
            bnorm = base_.pow(2).sum(dim=-1, keepdim=True).sqrt() + 1e-3
            maxn = ratio * bnorm + eps
            scale = (maxn / dnorm).clamp(max=1.0)
            return delta_ * scale

        with torch.inference_mode(False), torch.enable_grad():
            h_prev_work = hPrev.detach().clone() if inference_mode_active else hPrev
            action_work = action.detach().clone() if inference_mode_active else action
            qp = self.Lin(h_prev_work, pr.to_qp, d.get("phys_to_qp"))

            if not qp.requires_grad:
                qp = qp.detach().requires_grad_(True)
            
            q, p = qp.chunk(2, dim=-1)

            for i in range(pr.substeps):
                h_cur = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))

                fa0_inp = torch.cat([h_cur, action_work], dim=-1)
                F0 = Force_net(fa0_inp) * torch.sigmoid(self.Lin(fa0_inp, pr.g_force, d.get("phys_g_force")))

                if pr.dampP > 0.0:
                    p = p * p.new_tensor(-pr.dampP * dt_sub).exp()

                p = p + 0.5 * dt_sub * F0

                q, p, H0, H1, dH_dp_mid = SymplecticLeapfrog(q, p, dt_sub, create_graph_=create_graph)

                if training_mode:
                    if i == 0:
                        H_start = H0
                    H_end = H1

                h_mid = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))
                fa1_inp = torch.cat([h_mid, action_work], dim=-1)
                F1 = Force_net(fa1_inp) * torch.sigmoid(self.Lin(fa1_inp, pr.g_force, d.get("phys_g_force")))

                p = p + 0.5 * dt_sub * F1

                if training_mode:
                    smooth_acc = smooth_acc + (F1 - F0).pow(2).mean()
                    F_avg = 0.5 * (F0 + F1)
                    work_acc = work_acc + (dH_dp_mid * F_avg).sum(dim=-1, keepdim=True) * dt_sub

        h_phys_raw = self.Lin(torch.cat([q, p], dim=-1), pr.from_qp, d.get("phys_from_qp"))

        d_corr = h_phys_raw - hS4
        gph = torch.sigmoid(self.Lin(torch.cat([hPrev, action], dim=-1), pr.g_phys, d.get("phys_g_phys")))
        d_corr = d_corr * gph

        base_ = hS4 - hPrev
        d_corr = ClampResidual(d_corr, base_, ratio=pr.clamp_ratio)

        alpha = torch.sigmoid(self.Lin(torch.cat([hPrev, action, hS4], dim=-1), pr.g_fuse, d.get("phys_g_fuse")))
        h_fused = hS4 + alpha * d_corr

        if not training_mode:
            return h_fused, None, None

        if (H_start is None) or (H_end is None):
            e_work = hPrev.new_tensor(0.0)
        else:
            denom = H_start.detach().abs().mean().clamp_min(1e-6)
            dH = (H_end - H_start)
            e_work = ((dH - work_acc) / denom).pow(2).mean()

        e_smooth = smooth_acc / float(pr.substeps)
        e_delta = d_corr.pow(2).mean()

        loss = pr.l_work * e_work + pr.l_smooth * e_smooth + pr.l_delta * e_delta
        aux = {"L_work": e_work.detach(), "L_smooth": e_smooth.detach(), "L_delta": e_delta.detach()}
        return h_fused, loss, aux


    def ForwardWithDeltas(
        self,
        x,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: List[Dict[str, Optional[torch.Tensor]]] = None,
        **kwargs,) -> Dict[str, torch.Tensor]:

        visionIn = x
        actionEnc = kwargs["actionEnc"]
        reward = kwargs.get("reward")
        done = kwargs.get("done")
        physicalState = kwargs["physicalState"]
        robotSelfState = kwargs["robotSelfState"]

        sample_arg = kwargs.get("sample")
        sample = bool(self.training) if sample_arg is None else bool(sample_arg)
        update_memory_arg = kwargs.get("updateMemory")
        update_memory = bool(self.training) if update_memory_arg is None else bool(update_memory_arg)
        update_memory = update_memory and bool(self.training)
        alphaKl = kwargs.get("alphaKl", 0.8)
        freeNats = kwargs.get("freeNats", 1.0)
        reconCoef = kwargs.get("reconCoef", 1.0)
        rewardCoef = kwargs.get("rewardCoef", 1.0)
        doneCoef = kwargs.get("doneCoef", 1.0)
        nsCoef = kwargs.get("nsCoef", 1.0)
        nsDistillCoef = kwargs.get("nsDistillCoef", 1e-2)
        nsPriorLogicCoef = kwargs.get("nsPriorLogicCoef", 1e-3)
        physCoef = kwargs.get("physCoef", 1e-4)
        pstBindCoef = kwargs.get("pstBindCoef", 0.05)

        B = int(visionIn.size(0))
        device, dtype = self.base.device, self.base.dtype
        self.base.EnsureB(B, device, dtype)

        d = deltasPerLayer[0] if (deltasPerLayer is not None) else {}

        h0 = self.base._h
        z0 = self.base._z

        a_enc = actionEnc
        embodied_action, robot_world_context = self.base.BuildEmbodiedAction(physicalState, a_enc, robotSelfState)
        a_t = self.ActProj(embodied_action, d) # [B, stochDim]

        h_pred = self.S4_Step(z0, a_t, updateState=True, d=d) # [B, deterDim]

        mu_p, logstd_p = self.Prior(h_pred, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        logits_pr = None
        P_pr_train = None
        ns_prior_logic = visionIn.new_tensor(0.0)

        if self.base._ns_enabled:
            logits_pr = self.base.ns_head_prior(h_pred)
            P_pr_raw = torch.sigmoid(logits_pr)
            P_pr_train = self.base.NsProjectProbs(P_pr_raw)

            dmu_p = self.base.ns_to_delta_mu(P_pr_train)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_pred, dmu_p], dim=-1)))

            _, pen_pr = self.base.NsProjectRuntime(P_pr_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)

            conf = self.base.NsConfidence(P_pr_train).mean(dim=-1, keepdim=True)
            gate_scale = (1.0 - 0.40 * pen_pr.view(-1, 1)) * (0.6 + 0.4 * conf)
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)

            mu_p = mu_p + gate * dmu_p

            if nsPriorLogicCoef > 0.0:
                ns_prior_logic, _ = self.base.NsLogicLosses(P_pr_train)

        mu_p_raw = mu_p
        pst_binding_prior = self.base.BindPhysicalMu(h_pred, mu_p, self.base.s4.x, physicalState, embodied_action, robot_world_context)
        mu_p = pst_binding_prior["bound_mu"]

        raw_e = self.ObsEnc(visionIn, d) # [B, stochDim]

        mu_q, logstd_q = self.Post(torch.cat([h_pred, raw_e], dim=-1), d).chunk(2, dim=-1)
        logstd_q = logstd_q.clamp(-7.0, 2.0)

        ns_loss = visionIn.new_tensor(0.0)
        ns_distill = visionIn.new_tensor(0.0)
        logits_q = None
        Q_train = None
        pen_q = None

        if self.base._ns_enabled:
            logits_q = self.base.ns_head_post(torch.cat([h_pred, raw_e], dim=-1))
            P_q_raw = torch.sigmoid(logits_q)
            Q_train = self.base.NsProjectProbs(P_q_raw)

            dmu_q = self.base.ns_to_delta_mu(Q_train)
            base_gate_q = torch.sigmoid(self.base.ns_gate_mu_post(torch.cat([h_pred, raw_e, dmu_q], dim=-1)))

            _, pen_q = self.base.NsProjectRuntime(P_q_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)

            conf_q = self.base.NsConfidence(Q_train).mean(dim=-1, keepdim=True)
            gate_scale_q = (1.0 - 0.40 * pen_q.view(-1, 1)) * (0.6 + 0.4 * conf_q)
            gate_q = (base_gate_q * gate_scale_q).clamp(0.0, 1.0)

            mu_q = mu_q + gate_q * dmu_q

            ns_loss, _ = self.base.NsLogicLosses(Q_train)

            if (logits_pr is not None) and (nsDistillCoef > 0.0):
                with torch.no_grad():
                    P_teacher = torch.sigmoid(logits_q)
                ns_distill = F.binary_cross_entropy_with_logits(logits_pr, P_teacher, reduction="mean")

        mu_q_raw = mu_q
        pst_binding_posterior = self.base.BindPhysicalMu(h_pred, mu_q, self.base.s4.x, physicalState, embodied_action, robot_world_context)
        mu_q = pst_binding_posterior["bound_mu"]

        if sample:
            z1 = mu_q + torch.exp(logstd_q) * torch.randn_like(mu_q)
        else:
            z1 = mu_q

        s_base = self.StateProj(torch.cat([h_pred, z1], dim=-1), d)
        s_prev_base = self.StateProj(torch.cat([h0, z0], dim=-1), d)

        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)

        prevA = self.base._A_prev if (self.base._A_prev is not None and self.base._A_prev.shape == A_t.shape) else None
        reg_A = self.base.conn.ComputeGeomReg(A_t, prevA)
        self.base._A_prev = A_t.detach()

        h_phys, phys_loss, _ = self.PhysRefiner(h0, a_t, h_pred, d)
        if phys_loss is None:
            phys_loss = visionIn.new_zeros(())
        s_phys = self.StateProj(torch.cat([h_phys, z1], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1)
        w = F.softmax(self.MixGate(g_in, d), dim=-1)
        s1 = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        dynamics_state = s1
        inp = torch.cat([s_base, dynamics_state, a_t], dim=-1)
        trunk = self.RdoneTrunk(self.base.rdone_ln(inp), d)
        r_pred = self.Rew(trunk, d).squeeze(-1)
        d_logit = self.Done(trunk, d).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)

        if self.base._use_memory:
            key = self.KeyEmbed(raw_e, a_t, d)
            mem_retrieved = self.base.MemRetrieve(key, updateImportance=update_memory)
            if update_memory:
                r_score = torch.tanh(r_pred.detach().abs()).clamp(0.0, 1.0)
                d_score = d_prob.detach().clamp(0.0, 1.0)
                if self.base._ns_enabled and (Q_train is not None) and (pen_q is not None):
                    conf_scalar = self.base.NsConfidence(Q_train.detach()).mean(dim=-1)
                    imp_ns = ((1.0 - pen_q.detach()).clamp(0.0, 1.0) * (0.5 + 0.5 * conf_scalar)).clamp(0.0, 1.0)
                else:
                    imp_ns = visionIn.new_full((B,), 0.5)
                imp = (0.60 * imp_ns + 0.25 * r_score + 0.15 * d_score).clamp(0.0, 1.0)
                self.base.MemAdd(key.detach(), dynamics_state.detach(), imp.detach())

            if mem_retrieved is not None:
                mem_s, mem_mask = mem_retrieved
                s_memory = self.FiLMHResidual(dynamics_state, mem_s, d)
                s1 = torch.where(mem_mask.view(B, 1), s_memory, dynamics_state)

        loss_recon = visionIn.new_tensor(0.0)
        recon_error = visionIn.new_zeros(B)
        recon = None
        if self.base.use_decoder:
            recon = self.ObsDec(s1, d)

            target = self.base.obs_enc[0](visionIn)

            recon_n = F.layer_norm(
                recon,
                normalized_shape=(int(recon.size(-1)),),
                weight=self.base.obs_enc[0].weight,
                bias=self.base.obs_enc[0].bias,
                eps=self.base.obs_enc[0].eps,)

            recon_error = (recon_n - target).pow(2).mean(dim=-1)
            loss_recon = recon_error.mean()

        aux_moe = visionIn.new_tensor(0.0)
        if self.base._ns_enabled:
            aux_moe = self.base.ns_head_prior.GetAuxLoss() + self.base.ns_head_post.GetAuxLoss()

        if reward is None:
            loss_reward = visionIn.new_zeros(())
        else:
            reward_target = reward.view(B).clamp(float(self.base.reward_min), float(self.base.reward_max))
            loss_reward = F.mse_loss(r_pred, reward_target, reduction="mean")
        if done is None:
            loss_done = visionIn.new_zeros(())
        else:
            loss_done = F.binary_cross_entropy_with_logits(
                d_logit, done.view(B).to(d_logit.dtype), reduction="mean")
        loss_kl = BalancedKL(mu_q, logstd_q, mu_p, logstd_p, alpha=alphaKl, freeNats=freeNats).mean()
        loss_pst_bind = 0.5 * (
            pst_binding_prior["loss_pst_bind"]
            + pst_binding_posterior["loss_pst_bind"])

        self.base._h = h_pred.detach()
        self.base._z = z1.detach()

        loss = (
            reconCoef * loss_recon
            + rewardCoef * loss_reward
            + doneCoef * loss_done
            + loss_kl
            + nsCoef * ns_loss
            + nsDistillCoef * ns_distill
            + nsPriorLogicCoef * ns_prior_logic
            + physCoef * phys_loss
            + pstBindCoef * loss_pst_bind
            + reg_A
            + 1e-1 * aux_moe)

        out: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_reward": loss_reward,
            "loss_done": loss_done,
            "loss_kl": loss_kl,
            "loss_ns": ns_loss,
            "loss_ns_distill": ns_distill,
            "loss_ns_prior_logic": ns_prior_logic,
            "loss_phys": phys_loss,
            "loss_pst_bind": loss_pst_bind,
            "loss_conn_reg": reg_A,
            "h_next": h_pred,
            "z_next": z1,
            "z_next_raw": mu_q_raw,
            "x_next": self.base.s4.x,
            "s_next": s1,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "recon_error": recon_error,
            "mu_p": mu_p,
            "mu_p_raw": mu_p_raw,
            "logstd_p": logstd_p,
            "mu_q": mu_q,
            "mu_q_raw": mu_q_raw,
            "logstd_q": logstd_q,
            "action_enc": a_enc,
            "embodied_action": embodied_action,
            "robot_world_context": robot_world_context,
            "pst_binding": pst_binding_posterior,
            "pst_binding_prior": pst_binding_prior,
            "pst_binding_posterior": pst_binding_posterior,}
        
        if self.base.use_decoder and recon is not None:
            out["recon"] = recon
            out["recon_target"] = visionIn

        return out

    @torch.no_grad()
    def StepPriorOnly(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        robotSelfState: torch.Tensor,
        sample: bool = False,
        ) -> Dict[str, torch.Tensor]:
        deltas = [self.ComposeLayerDelta(layerIdx) for layerIdx in range(self.layerCount)]
        return self.StepPriorWithDeltas(
            hPrev,
            zPrev,
            s4xPrev,
            actionEnc,
            sample=sample,
            deltasPerLayer=deltas,
            physicalState=physicalState,
            robotSelfState=robotSelfState)

    @torch.no_grad()
    def StepPriorWithDeltas(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        actionEnc: torch.Tensor,
        sample: bool = False,
        deltasPerLayer: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        **kwargs,) -> Dict[str, torch.Tensor]:
        physicalState = kwargs["physicalState"]
        robotSelfState = kwargs["robotSelfState"]
        B = int(actionEnc.size(0))
        device, dtype = self.base.device, self.base.dtype

        d = deltasPerLayer[0] if (deltasPerLayer is not None) else {}

        if hPrev is None or zPrev is None or s4xPrev is None:
            hPrev = torch.zeros(B, self.base.deter_dim, device=device, dtype=dtype)
            zPrev = torch.zeros(B, self.base.stoch_dim, device=device, dtype=dtype)
            s4xPrev = torch.zeros(B, self.base.ssm_dim, device=device, dtype=dtype)

        embodied_action, robot_world_context = self.base.BuildEmbodiedAction(physicalState, actionEnc, robotSelfState)
        a_t = self.ActProj(embodied_action, d)
        h_next, s4x_next = self.S4StepWithX(zPrev, a_t, s4xPrev, d)

        mu_p, logstd_p = self.Prior(h_next, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self.base._ns_enabled:
            ns_logits = self.base.ns_head_prior(
                h_next, deterministic=True, updateAux=False)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.base.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.base.NsConfidence(Q).mean(dim=-1, keepdim=True)

            dmu = self.base.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            gate = (base_gate * gate_scale).clamp(0.0, 1.0)

            mu_p = mu_p + gate * dmu

        mu_p_raw = mu_p
        pst_binding = self.base.BindPhysicalMu(h_next, mu_p, s4x_next, physicalState, embodied_action, robot_world_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.StateProj(torch.cat([h_next, z_next], dim=-1), d)
        s_prev_base = self.StateProj(torch.cat([hPrev, zPrev], dim=-1), d)

        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)

        h_phys, _, _ = self.PhysRefiner(hPrev, a_t, h_next, d)
        s_phys = self.StateProj(torch.cat([h_phys, z_next], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        g_in = torch.cat([s_base, d_tr, d_ph], dim=-1)

        w = F.softmax(self.MixGate(g_in, d), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        inp = torch.cat([s_base, s_next, a_t], dim=-1)
        h = self.RdoneTrunk(self.base.rdone_ln(inp), d, deterministic=True)

        r_pred = self.Rew(h, d).squeeze(-1)
        d_logit = self.Done(h, d).squeeze(-1)
        d_prob = torch.sigmoid(d_logit)

        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": s4x_next,
            "s_next": s_next,
            "embodied_action": embodied_action,
            "robot_world_context": robot_world_context,
            "r_pred": r_pred,
            "d_prob": d_prob,
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

    def ExportState(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.base.ExportState()

    def PriorRolloutFromStateAction(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor,
        sample: bool = False,) -> Dict[str, torch.Tensor]:
        d = self.ComposeLayerDelta(0)
        return self.PriorRolloutFromStateActionWithDeltas(
            hPrev,
            zPrev,
            s4xPrev,
            physicalState=physicalState,
            actionEnc=actionEnc,
            robotSelfState=robotSelfState,
            sample=sample,
            d=d)

    def PriorRolloutFromStateActionWithDeltas(
        self,
        hPrev: torch.Tensor,
        zPrev: torch.Tensor,
        s4xPrev: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor,
        sample: bool,
        d: Dict[str, Optional[torch.Tensor]],) -> Dict[str, torch.Tensor]:
        s_prev_base = self.StateProj(torch.cat([hPrev, zPrev], dim=-1), d)

        embodied_action, robot_world_context = self.base.BuildEmbodiedAction(physicalState, actionEnc, robotSelfState)
        a_t = self.ActProj(embodied_action, d)
        h_next, x_next = self.S4StepWithX(zPrev, a_t, s4xPrev, d)
        mu_p, logstd_p = self.Prior(h_next, d).chunk(2, dim=-1)
        logstd_p = logstd_p.clamp(-7.0, 2.0)

        if self.base._ns_enabled:
            ns_logits = self.base.ns_head_prior(h_next)
            P_raw = torch.sigmoid(ns_logits)
            Q, pen = self.base.NsProjectRuntime(P_raw, aloTau=0.60, implAlpha=1.0, temp=1.0)
            conf = self.base.NsConfidence(Q).mean(dim=-1, keepdim=True)
            dmu = self.base.ns_to_delta_mu(Q)
            base_gate = torch.sigmoid(self.base.ns_gate_mu(torch.cat([h_next, dmu], dim=-1)))
            gate_scale = (1.0 - 0.40 * pen.view(-1, 1)) * (0.6 + 0.4 * conf)
            mu_p = mu_p + (base_gate * gate_scale).clamp(0.0, 1.0) * dmu

        mu_p_raw = mu_p
        pst_binding = self.base.BindPhysicalMu(h_next, mu_p, x_next, physicalState, embodied_action, robot_world_context)
        mu_p = pst_binding["bound_mu"]

        if sample:
            z_next = mu_p + torch.exp(logstd_p) * torch.randn_like(mu_p)
        else:
            z_next = mu_p

        s_base = self.StateProj(torch.cat([h_next, z_next], dim=-1), d)
        A_t = self.ConnNet(s_prev_base, a_t, d)
        s_transport = self.base.conn.TransportApply(A_t, s_prev_base)
        h_phys, _, _ = self.PhysRefiner(hPrev, a_t, h_next, d)
        s_phys = self.StateProj(torch.cat([h_phys, z_next], dim=-1), d)

        d_tr = s_transport - s_base
        d_ph = s_phys - s_base
        w = F.softmax(self.MixGate(torch.cat([s_base, d_tr, d_ph], dim=-1), d), dim=-1)
        s_next = w[:, 0:1] * s_base + w[:, 1:2] * s_transport + w[:, 2:3] * s_phys

        trunk = self.RdoneTrunk(self.base.rdone_ln(torch.cat([s_base, s_next, a_t], dim=-1)), d)
        return {
            "h_next": h_next,
            "z_next": z_next,
            "z_next_raw": mu_p_raw,
            "x_next": x_next,
            "s_next": s_next,
            "action_enc": actionEnc,
            "embodied_action": embodied_action,
            "robot_world_context": robot_world_context,
            "r_pred": self.Rew(trunk, d).squeeze(-1),
            "d_prob": torch.sigmoid(self.Done(trunk, d).squeeze(-1)),
            "d_tr": d_tr,
            "d_ph": d_ph,
            "pst_binding": pst_binding,
            "loss_pst_bind": pst_binding["loss_pst_bind"],}

    def PredictNextVisualFromPosterior(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor,
        sample: bool = False,) -> Dict[str, Any]:
        d = self.ComposeLayerDelta(0)
        return self.PredictNextVisualFromPosteriorWithDeltas(
            h,
            z,
            s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            robotSelfState=robotSelfState,
            sample=sample,
            d=d)

    def PredictNextVisualFromPosteriorWithDeltas(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        s4x: torch.Tensor,
        physicalState: Dict[str, torch.Tensor],
        actionEnc: torch.Tensor,
        robotSelfState: torch.Tensor,
        sample: bool,
        d: Dict[str, Optional[torch.Tensor]],) -> Dict[str, Any]:
        rollout = self.PriorRolloutFromStateActionWithDeltas(
            h,
            z,
            s4x,
            physicalState=physicalState,
            actionEnc=actionEnc,
            robotSelfState=robotSelfState,
            sample=sample,
            d=d)
        pred = self.base.BuildPredictedVisual(rollout["s_next"])
        pred["prior_rollout"] = rollout
        return pred

    def ComputePredictionLoss(
        self,
        predictedVisual: PredictedVisualPack,
        reconstructedVisualState: Dict[str, torch.Tensor],
        targetVisualState: Any,
        precision: torch.Tensor,
        sampleMask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        return self.base.ComputePredictionLoss(
            predictedVisual=predictedVisual,
            reconstructedVisualState=reconstructedVisualState,
            targetVisualState=targetVisualState,
            precision=precision,
            sampleMask=sampleMask,)

    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        r = int(a.size(0))
        if r <= 0 or a.numel() == 0 or b.numel() == 0 or abs(float(scale)) < 1e-12:
            return False

        init = {"A": a.detach().clone(), "B": b.detach().clone(), "scale": float(scale)}

        if site == "obs_enc0":
            self.base.obs_enc[1].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "obs_enc1":
            self.base.obs_enc[4].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "act_proj":
            self.base.act_proj[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "s4_in_to_ssm":
            self.base.s4.in_to_ssm.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_ssm_to_deter":
            self.base.s4.ssm_to_deter.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_in_to_deter":
            self.base.s4.in_to_deter.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_gate":
            self.base.s4.gate.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "s4_out_gate":
            self.base.s4.out_gate.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "prior":
            self.base.prior_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "post":
            self.base.post_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "state_proj":
            self.base.state_proj[1].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "rdone0":
            self.base.rdone_trunk[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "rdone1":
            self.base.rdone_trunk[3].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "rew":
            self.base.rew_head[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "done":
            self.base.done_head[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "obs_dec0":
            self.base.obs_dec[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "obs_dec1":
            self.base.obs_dec[2].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "mix_gate":
            self.base.mix_gate[0].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "key_to_gb":
            self.base.key_emb.to_gb.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "key_mlp1":
            self.base.key_emb.mlp1.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "key_mlp2":
            self.base.key_emb.mlp2.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "ssfilm_e_to_gb":
            self.base.state_state_film.e_to_gb.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_e_to_h":
            self.base.state_state_film.e_to_h.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_delta0":
            self.base.state_state_film.delta_mlp[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_delta1":
            self.base.state_state_film.delta_mlp[3].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "ssfilm_to_gate":
            self.base.state_state_film.to_gate.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "conn_enc_s":
            self.base.conn.enc_s[1].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_enc_a":
            self.base.conn.enc_a[1].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_film_gamma_a":
            self.base.conn.film_gamma_a.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_film_beta_a":
            self.base.conn.film_beta_a.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site.startswith("conn_blk") and (("_ff0" in site) or ("_ff1" in site)):
            s0 = site.replace("conn_blk", "")
            i_str, which = s0.split("_", 1)
            i = int(i_str)
            if which == "ff0":
                self.base.conn.blocks[i].ff[0].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
            elif which == "ff1":
                self.base.conn.blocks[i].ff[3].linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
            else:
                raise ValueError(site)

        elif site == "conn_head_uv":
            self.base.conn.head_uv.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_head_full":
            self.base.conn.head_full.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "conn_mix":
            self.base.conn.mix.linear.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_to_qp":
            self.base.phys_refiner.to_qp.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_from_qp":
            self.base.phys_refiner.from_qp.Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_H0":
            self.base.phys_refiner.H_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_H1":
            self.base.phys_refiner.H_net[2].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_H2":
            self.base.phys_refiner.H_net[4].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_force0":
            self.base.phys_refiner.force_net[0].Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_force1":
            self.base.phys_refiner.force_net[2].Grow(r, init=init, freezeOld=self.freezeOldPar)

        elif site == "phys_g_force":
            self.base.phys_refiner.g_force.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_g_phys":
            self.base.phys_refiner.g_phys.Grow(r, init=init, freezeOld=self.freezeOldPar)
        elif site == "phys_g_fuse":
            self.base.phys_refiner.g_fuse.Grow(r, init=init, freezeOld=self.freezeOldPar)

        else:
            raise ValueError(f"Unknown site: {site}")

        return True





class TestWorldMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wm = RSSMWorldModel(
            visionDim=64,
            actionDim=ModuleDim.DecisionFeedbackEmbedDim,
            deterDim=64,
            stochDim=16,
            stateDim=64,
            ssmDim=32,
            useDecoder=True,
            useMemory=False,
            nsEnabled=False,
            globalFeatDim=64,
            objectTokenDim=32,
            numObjectTokens=8,
            motionPredDim=32,
            integratedFeatDim=64,
            physicalSlots=16,
            physicalSlotDim=32,
            physicalPoseDim=7,
            physicalAttrDim=8,
            physicalIdDim=32,
            physicalRelDim=36,
            physicalRelationClasses=32,
            physicalSemanticDim=16,
            physicalStateDim=8,
            physicalAffordanceDim=4,
            physicalTextDim=4,
            physicalSymbolDim=8,).to(self.device)

    def MakePhysicalState(self, wm: RSSMWorldModel, B: int, activeSlots: int = 4) -> Dict[str, torch.Tensor]:
        K = wm.physical_slots
        D = wm.physical_slot_dim
        device = self.device
        M = torch.zeros(B, K, device=device)
        M[:, :activeSlots] = 1.0
        observed = torch.zeros(B, K, device=device, dtype=torch.bool)
        observed[:, :activeSlots] = True
        last_seen = torch.zeros(B, K, device=device, dtype=torch.long)
        step = torch.full((B,), 4, device=device, dtype=torch.long)
        return {
            "SlotPresence": M,
            "MphysRaw": M,
            "IdentityKey": F.normalize(torch.randn(B, K, wm.physical_id_dim, device=device), dim=-1, eps=1e-6),
            "SlotState": torch.randn(B, K, D, device=device) * M.unsqueeze(-1),
            "ObservedSlotMask": M,
            "PoseCamera": torch.randn(B, K, wm.physical_pose_dim, device=device) * M.unsqueeze(-1),
            "PoseWorld": torch.randn(B, K, wm.physical_pose_dim, device=device) * M.unsqueeze(-1),
            "ARaw": torch.randn(B, K, wm.physical_attr_dim, device=device) * M.unsqueeze(-1),
            "Size": torch.randn(B, K, 3, device=device) * M.unsqueeze(-1),
            "StateRaw": torch.randn(B, K, wm.physical_state_dim, device=device) * M.unsqueeze(-1),
            "AffordanceRaw": torch.randn(B, K, wm.physical_affordance_dim, device=device) * M.unsqueeze(-1),
            "MotionRaw": torch.randn(B, K, wm.physical_pose_dim, device=device) * M.unsqueeze(-1),
            "MovingProbRaw": torch.zeros(B, K, device=device),
            "ContactProbRaw": torch.zeros(B, K, device=device),
            "ContactForceRaw": torch.zeros(B, K, 2, device=device),
            "ContactPointRaw": torch.zeros(B, K, 3, device=device),
            "Visibility": M.clone(),
            "Occlusion": torch.zeros(B, K, device=device),
            "HasTextProb": torch.zeros(B, K, device=device),
            "TextEmbed": torch.zeros(B, K, wm.physical_text_dim, device=device),
            "SymbolProb": torch.zeros(B, K, wm.physical_symbol_dim, device=device),
            "Semantic": torch.randn(B, K, wm.physical_semantic_dim, device=device) * M.unsqueeze(-1),
            "ExternalRelationProbRaw": torch.zeros(B, K, wm.physical_relation_classes, device=device),
            "PairwiseRelation": torch.zeros(B, K, K, wm.physical_rel_dim, device=device),
            "ParentProb": torch.zeros(B, K, K, device=device),
            "Observed": observed,
            "LastSeen": last_seen,
            "Step": step,}

    def MakeRobotSelfState(self, wm: RSSMWorldModel, B: int) -> torch.Tensor:
        return torch.randn(B, wm.robot_self_dim, device=self.device)

    def TestPSTWorldBinderShapes(self) -> bool:
        B = 2
        wm = self.wm.eval()
        physical = self.MakePhysicalState(wm, B)
        out = wm.pst_binder(
            torch.randn(B, wm.deter_dim, device=self.device),
            torch.randn(B, wm.stoch_dim, device=self.device),
            torch.randn(B, wm.ssm_dim, device=self.device),
            physical,
            torch.randn(B, wm.action_dim, device=self.device),
            torch.randn(B, wm.robot_world_dim, device=self.device),)
        ok = (
            out["bound_mu"].shape == (B, wm.stoch_dim)
            and out["delta_mu"].shape == (B, wm.stoch_dim)
            and out["bind_gate"].shape == (B, wm.stoch_dim)
            and out["pst_context"].shape == (B, wm.physical_slot_dim)
            and out["slot_binding_weight"].shape == (B, wm.physical_slots))
        print(f"PSTWorldBinder shapes {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestRSSMStepPosterior(self) -> bool:
        B = 3
        wm = self.wm.eval()
        wm.ResetState(batchSize=B)
        out = wm.StepPosterior(
            torch.randn(B, wm.vision_dim, device=self.device),
            torch.randn(B, wm.action_dim, device=self.device),
            self.MakePhysicalState(wm, B),
            self.MakeRobotSelfState(wm, B),
            sample=False,)
        ok = (
            out["h_next"].shape == (B, wm.deter_dim)
            and out["z_next"].shape == (B, wm.stoch_dim)
            and out["s_next"].shape == (B, wm.state_dim)
            and out["r_pred"].shape == (B,)
            and out["d_prob"].shape == (B,))
        print(f"RSSM StepPosterior {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestRSSMStepPriorOnly(self) -> bool:
        B = 3
        wm = self.wm.eval()
        out = wm.StepPriorOnly(
            torch.randn(B, wm.deter_dim, device=self.device),
            torch.randn(B, wm.stoch_dim, device=self.device),
            torch.randn(B, wm.ssm_dim, device=self.device),
            torch.randn(B, wm.action_dim, device=self.device),
            self.MakePhysicalState(wm, B),
            self.MakeRobotSelfState(wm, B),
            sample=False,)
        ok = (
            out["h_next"].shape == (B, wm.deter_dim)
            and out["z_next"].shape == (B, wm.stoch_dim)
            and out["s_next"].shape == (B, wm.state_dim)
            and out["r_pred"].shape == (B,)
            and out["d_prob"].shape == (B,))
        print(f"RSSM StepPriorOnly {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestPriorRolloutIsPureAndDeterministic(self) -> bool:
        wm = self.wm.train()
        wm.ResetState(batchSize=1)
        wm.EnsureB(1, wm.device, wm.dtype)
        wm.ImportPhysicalState(self.MakePhysicalState(wm, 1, activeSlots=3))
        wm._h.normal_()
        wm._z.normal_()
        wm.s4.x.normal_()
        live_before = {
            name: value.detach().clone()
            for name, value in wm.named_buffers()
            if name.startswith("_pst_")
            or name.startswith("_mem_")
            or name in {"_h", "_z", "_robot_self_state", "_robot_action_context", "s4.x"}}

        B = 3
        physical = self.MakePhysicalState(wm, B, activeSlots=3)
        h = torch.randn(B, wm.deter_dim, device=self.device)
        z = torch.randn(B, wm.stoch_dim, device=self.device)
        x = torch.randn(B, wm.ssm_dim, device=self.device)
        action = torch.randn(B, wm.action_dim, device=self.device)
        robot = self.MakeRobotSelfState(wm, B)
        first = wm.StepPriorOnly(h, z, x, action, physical, robot)
        second = wm.StepPriorOnly(h, z, x, action, physical, robot)

        live_after = dict(wm.named_buffers())
        state_unchanged = all(
            name in live_after
            and tuple(live_after[name].shape) == tuple(before.shape)
            and torch.equal(live_after[name], before)
            for name, before in live_before.items())
        deterministic = all(
            torch.equal(first[name], second[name])
            for name in ("h_next", "z_next", "x_next", "s_next", "r_pred", "d_prob"))
        ok = state_unchanged and deterministic
        print(
            f"Prior rollout purity/determinism {'passed' if ok else 'failed'} "
            f"| state={state_unchanged}, deterministic={deterministic}")
        return bool(ok)

    def TestPriorRolloutIsDeterministicWithNeSy(self) -> bool:
        wm = self.wm.train()
        original_ns_enabled = wm._ns_enabled
        try:
            wm._ns_enabled = True
            wm.ns_head_prior.aux_loss.fill_(7.0)
            B = 1
            inputs = (
                torch.randn(B, wm.deter_dim, device=self.device),
                torch.randn(B, wm.stoch_dim, device=self.device),
                torch.randn(B, wm.ssm_dim, device=self.device),
                torch.randn(B, wm.action_dim, device=self.device))
            kwargs = {
                "physicalState": self.MakePhysicalState(wm, B, activeSlots=2),
                "robotSelfState": self.MakeRobotSelfState(wm, B)}
            first = wm.StepPriorOnly(*inputs, **kwargs)
            second = wm.StepPriorOnly(*inputs, **kwargs)
            deterministic = all(
                torch.equal(first[name], second[name])
                for name in ("h_next", "z_next", "x_next", "s_next", "r_pred", "d_prob"))
            state_unchanged = (
                wm.training
                and wm.ns_head_prior.training
                and float(wm.ns_head_prior.aux_loss.item()) == 7.0)
            ok = deterministic and state_unchanged
            print(
                f"NeSy prior determinism {'passed' if ok else 'failed'} "
                f"| deterministic={deterministic}, state={state_unchanged}")
            return bool(ok)
        finally:
            wm._ns_enabled = original_ns_enabled

    def TestPhysicsRefinerSupportsInferenceModeAndDamping(self) -> bool:
        wm = self.wm.eval()
        original_damping = wm.phys_refiner.dampP
        trainability = {name: parameter.requires_grad for name, parameter in wm.named_parameters()}
        try:
            wm.phys_refiner.dampP = 0.1
            B = 1
            h_prev = torch.randn(B, wm.deter_dim, device=self.device)
            action = torch.randn(B, wm.stoch_dim, device=self.device)
            h_s4 = torch.randn(B, wm.deter_dim, device=self.device)
            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()
            with torch.inference_mode():
                base_out, _, _ = wm.phys_refiner(h_prev, action, h_s4)
                online_out, _, _ = wrapper.PhysRefiner(h_prev, action, h_s4, {})
            ok = (
                bool(torch.isfinite(base_out).all().item())
                and bool(torch.isfinite(online_out).all().item()))
            print(f"Physics inference/damping {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as error:
            print(f"Physics inference/damping failed: {error}")
            return False
        finally:
            wm.phys_refiner.dampP = original_damping
            for name, parameter in wm.named_parameters():
                parameter.requires_grad_(trainability[name])

    def TestOnlinePriorRolloutUsesCandidatesWithoutMutatingState(self) -> bool:
        original_training = self.wm.training
        trainability = {
            name: parameter.requires_grad
            for name, parameter in self.wm.named_parameters()}
        wm = self.wm.eval()
        wm.ResetState(batchSize=1)
        wm.EnsureB(1, wm.device, wm.dtype)
        wm.ImportPhysicalState(self.MakePhysicalState(wm, 1, activeSlots=2))
        wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).train()
        spec = wrapper.sites["rew"]
        candidate_a, candidate_b, candidate_scale = spec.allocFn(
            1, wrapper.deviceRef, wrapper.dtypeRef)
        with torch.no_grad():
            candidate_a.fill_(0.05)
            candidate_b.fill_(0.10)
            candidate_scale.fill_(0.50)
        wrapper.cand["rew"][0]["A"].append(candidate_a)
        wrapper.cand["rew"][0]["B"].append(candidate_b)
        wrapper.cand["rew"][0]["s"].append(candidate_scale)
        live_before = {
            name: value.detach().clone()
            for name, value in wm.named_buffers()
            if name.startswith("_pst_")
            or name.startswith("_mem_")
            or name in {"_h", "_z", "_robot_self_state", "_robot_action_context", "s4.x"}}
        state_before = tuple(value.detach().clone() for value in wm.ExportState())
        connection_before = None if wm._A_prev is None else wm._A_prev.detach().clone()

        B = 2
        physical = self.MakePhysicalState(wm, B, activeSlots=2)
        inputs = (
            torch.randn(B, wm.deter_dim, device=self.device),
            torch.randn(B, wm.stoch_dim, device=self.device),
            torch.randn(B, wm.ssm_dim, device=self.device),
            torch.randn(B, wm.action_dim, device=self.device))
        kwargs = {
            "physicalState": physical,
            "robotSelfState": self.MakeRobotSelfState(wm, B)}
        automatic = wrapper.StepPriorOnly(*inputs, **kwargs)
        explicit = wrapper.StepPriorWithDeltas(
            *inputs,
            deltasPerLayer=[wrapper.ComposeLayerDelta(0)],
            **kwargs)
        base = wm.StepPriorOnly(*inputs, **kwargs)

        live_after = dict(wm.named_buffers())
        state_unchanged = all(
            name in live_after
            and tuple(live_after[name].shape) == tuple(before.shape)
            and torch.equal(live_after[name], before)
            for name, before in live_before.items())
        state_unchanged = state_unchanged and all(
            torch.equal(after, before)
            for after, before in zip(wm.ExportState(), state_before))
        if connection_before is None:
            state_unchanged = state_unchanged and wm._A_prev is None
        else:
            state_unchanged = (
                state_unchanged
                and wm._A_prev is not None
                and torch.equal(wm._A_prev, connection_before))
        composed = all(
            torch.equal(automatic[name], explicit[name])
            for name in ("h_next", "z_next", "x_next", "s_next", "r_pred", "d_prob"))
        candidate_effect = not torch.allclose(
            automatic["r_pred"], base["r_pred"], atol=1e-9, rtol=0.0)
        ok = state_unchanged and composed and candidate_effect
        print(
            f"Online prior rollout contract {'passed' if ok else 'failed'} "
            f"| state={state_unchanged}, composed={composed}, candidate={candidate_effect}")
        for name, parameter in wm.named_parameters():
            parameter.requires_grad_(trainability[name])
        wm.train(original_training)
        return bool(ok)

    def TestRobotSelfStateAffectsWorldDynamics(self) -> bool:
        B = 2
        wm = self.wm.eval()
        physical = self.MakePhysicalState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        robot_a = self.MakeRobotSelfState(wm, B)
        robot_b = robot_a + 1.0
        h = torch.randn(B, wm.deter_dim, device=self.device)
        z = torch.randn(B, wm.stoch_dim, device=self.device)
        x = torch.randn(B, wm.ssm_dim, device=self.device)

        prior_a = wm.StepPriorOnly(
            h,
            z,
            x,
            action,
            physical,
            robot_a,
            sample=False,)
        prior_b = wm.StepPriorOnly(
            h,
            z,
            x,
            action,
            physical,
            robot_b,
            sample=False,)

        vision = torch.randn(B, wm.vision_dim, device=self.device)
        wm.ResetState(batchSize=B)
        post_a = wm.StepPosterior(
            vision,
            action,
            physical,
            robot_a,
            sample=False,)
        wm.ResetState(batchSize=B)
        post_b = wm.StepPosterior(
            vision,
            action,
            physical,
            robot_b,
            sample=False,)

        prior_diff = float((prior_a["s_next"] - prior_b["s_next"]).abs().mean())
        post_diff = float((post_a["s_next"] - post_b["s_next"]).abs().mean())
        ok = prior_diff > 1e-7 and post_diff > 1e-7
        print(f"Robot self state affects world dynamics {'passed' if ok else 'failed'} | prior={prior_diff:.3e}, posterior={post_diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationUsesPairwiseRelations(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_a = self.MakePhysicalState(wm, B, activeSlots=3)
        physical_b = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_a.items()}
        physical_b["PairwiseRelation"].zero_()
        physical_b["PairwiseRelation"][0, 0, 1, 0] = 0.5
        physical_b["PairwiseRelation"][0, 0, 1, 3] = 0.5
        physical_b["PairwiseRelation"][0, 0, 1, 4] = 1.0
        physical_b["PairwiseRelation"][0, 1, 0, 0] = -0.5
        physical_b["PairwiseRelation"][0, 1, 0, 3] = 0.5
        physical_b["PairwiseRelation"][0, 1, 0, 5] = 1.0

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_a = wm.robot_world_relation(robot, physical_a, action)
        context_b = wm.robot_world_relation(robot, physical_b, action)
        diff = float((context_a - context_b).abs().mean())
        ok = context_a.shape == (B, wm.robot_world_dim) and diff > 1e-8
        print(f"Robot-world relation uses pairwise relations {'passed' if ok else 'failed'} | diff={diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationIgnoresInvalidSlots(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_a = self.MakePhysicalState(wm, B, activeSlots=1)
        physical_b = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_a.items()}
        physical_b["SlotState"][:, 1:] = torch.randn_like(physical_b["SlotState"][:, 1:]) * 100.0
        physical_b["PoseWorld"][:, 1:] = torch.randn_like(physical_b["PoseWorld"][:, 1:]) * 100.0
        physical_b["ARaw"][:, 1:] = torch.randn_like(physical_b["ARaw"][:, 1:]) * 100.0
        physical_b["StateRaw"][:, 1:] = torch.randn_like(physical_b["StateRaw"][:, 1:]) * 100.0
        physical_b["AffordanceRaw"][:, 1:] = torch.randn_like(physical_b["AffordanceRaw"][:, 1:]) * 100.0
        physical_b["MotionRaw"][:, 1:] = torch.randn_like(physical_b["MotionRaw"][:, 1:]) * 100.0
        physical_b["ExternalRelationProbRaw"][:, 1:] = torch.randn_like(physical_b["ExternalRelationProbRaw"][:, 1:]) * 100.0
        physical_b["ContactProbRaw"][:, 1:] = torch.randn_like(physical_b["ContactProbRaw"][:, 1:]) * 100.0
        physical_b["MovingProbRaw"][:, 1:] = torch.randn_like(physical_b["MovingProbRaw"][:, 1:]) * 100.0
        physical_b["ContactForceRaw"][:, 1:] = torch.randn_like(physical_b["ContactForceRaw"][:, 1:]) * 100.0
        physical_b["ContactPointRaw"][:, 1:] = torch.randn_like(physical_b["ContactPointRaw"][:, 1:]) * 100.0
        physical_b["PairwiseRelation"][:, 1:] = torch.randn_like(physical_b["PairwiseRelation"][:, 1:]) * 100.0
        physical_b["PairwiseRelation"][:, :, 1:] = torch.randn_like(physical_b["PairwiseRelation"][:, :, 1:]) * 100.0

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_a = wm.robot_world_relation(robot, physical_a, action)
        context_b = wm.robot_world_relation(robot, physical_b, action)
        diff = float((context_a - context_b).abs().max())
        ok = context_a.shape == (B, wm.robot_world_dim) and diff < 1e-6
        print(f"Robot-world relation ignores invalid slots {'passed' if ok else 'failed'} | max_diff={diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationMasksInvalidNonFiniteValues(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_a = self.MakePhysicalState(wm, B, activeSlots=2)
        physical_b = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_a.items()}
        vector_fields = [
            "SlotState", "PoseWorld", "ARaw", "Size", "StateRaw", "AffordanceRaw",
            "MotionRaw", "ExternalRelationProbRaw", "ContactForceRaw", "ContactPointRaw"]
        if "ContactPointWorldRaw" in physical_b:
            vector_fields.append("ContactPointWorldRaw")
        scalar_fields = ["ContactProbRaw", "MovingProbRaw", "Visibility", "Occlusion"]
        for key in vector_fields:
            physical_b[key][:, 2:] = torch.nan
        for key in scalar_fields:
            physical_b[key][:, 2:] = torch.nan
        physical_b["MphysRaw"][:, 2:] = torch.nan
        physical_b["PairwiseRelation"][:, 2:] = torch.nan
        physical_b["PairwiseRelation"][:, :, 2:] = torch.nan
        physical_b["PairwiseRelation"][:, 0, 0] = torch.nan
        physical_b["PairwiseRelation"][:, 1, 1] = torch.nan

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_a = wm.robot_world_relation(robot, physical_a, action)
        context_b = wm.robot_world_relation(robot, physical_b, action)
        diff = float((context_a - context_b).abs().max())
        ok = bool(torch.isfinite(context_b).all().item()) and diff < 1e-6
        print(f"Robot-world relation masks invalid non-finite values {'passed' if ok else 'failed'} | max_diff={diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationQuaternionDoubleCover(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_a = self.MakePhysicalState(wm, B, activeSlots=3)
        quaternion = F.normalize(torch.tensor(
            [[0.2, -0.3, 0.4, -0.5], [-0.7, 0.1, 0.2, 0.3], [0.4, 0.6, -0.1, 0.2]],
            device=self.device), dim=-1)
        physical_a["PoseWorld"][0, :3, 3:7] = quaternion
        physical_a["MotionRaw"][0, :3, 3:7] = quaternion.roll(1, dims=0)
        physical_b = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_a.items()}
        physical_b["PoseWorld"][0, :3, 3:7].mul_(-3.0)
        physical_b["MotionRaw"][0, :3, 3:7].mul_(-2.0)

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_a = wm.robot_world_relation(robot, physical_a, action)
        context_b = wm.robot_world_relation(robot, physical_b, action)
        diff = float((context_a - context_b).abs().max())
        ok = torch.allclose(context_a, context_b, atol=2e-6, rtol=1e-5)
        print(f"Robot-world relation quaternion double cover {'passed' if ok else 'failed'} | max_diff={diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationPreservesMetricScale(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_near = self.MakePhysicalState(wm, B, activeSlots=2)
        physical_near["PairwiseRelation"].zero_()
        forward_geometry = torch.tensor([0.1, 0.2, 0.3, 0.37416574], device=self.device)
        reverse_geometry = torch.tensor([-0.1, -0.2, -0.3, 0.37416574], device=self.device)
        physical_near["PairwiseRelation"][0, 0, 1, :4] = forward_geometry
        physical_near["PairwiseRelation"][0, 1, 0, :4] = reverse_geometry
        physical_near["PairwiseRelation"][0, 0, 1, 4] = 1.0
        physical_near["PairwiseRelation"][0, 1, 0, 5] = 1.0
        physical_far = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_near.items()}
        physical_far["PairwiseRelation"][0, 0, 1, :4].mul_(10.0)
        physical_far["PairwiseRelation"][0, 1, 0, :4].mul_(10.0)

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_near = wm.robot_world_relation(robot, physical_near, action)
        context_far = wm.robot_world_relation(robot, physical_far, action)
        diff = float((context_near - context_far).abs().mean())
        ok = diff > 1e-5
        print(f"Robot-world relation preserves metric scale {'passed' if ok else 'failed'} | diff={diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationDecaysStaleRelationProbabilities(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_fresh = self.MakePhysicalState(wm, B, activeSlots=3)
        physical_fresh["PairwiseRelation"][0, 0, 1, 4] = 1.0
        physical_fresh["PairwiseRelation"][0, 0, 2, 5] = 1.0
        physical_fresh["Step"].fill_(256)
        physical_fresh["PairRelationLastSeen"] = torch.full(
            (B, wm.physical_slots, wm.physical_slots),
            256,
            device=self.device,
            dtype=torch.long)
        physical_stale = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_fresh.items()}
        physical_stale["PairRelationLastSeen"].zero_()
        physical_without_semantics = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_stale.items()}
        physical_without_semantics["PairwiseRelation"][..., 4:].zero_()

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_fresh = wm.robot_world_relation(robot, physical_fresh, action)
        context_stale = wm.robot_world_relation(robot, physical_stale, action)
        context_without_semantics = wm.robot_world_relation(robot, physical_without_semantics, action)
        diff = float((context_fresh - context_stale).abs().mean())
        unseen_diff = float((context_stale - context_without_semantics).abs().max())
        ok = diff > 1e-5 and unseen_diff < 1e-6
        print(
            f"Robot-world relation decays stale relation probabilities {'passed' if ok else 'failed'} "
            f"| fresh_diff={diff:.3e}, unseen_diff={unseen_diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationPreservesSceneConfidence(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_high = self.MakePhysicalState(wm, B, activeSlots=3)
        physical_low = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_high.items()}
        physical_low["SlotPresence"][:, :3].mul_(0.1)

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_high = wm.robot_world_relation(robot, physical_high, action)
        context_low = wm.robot_world_relation(robot, physical_low, action)
        diff = float((context_high - context_low).abs().mean())
        high_norm = float(context_high.norm(dim=-1).mean())
        low_norm = float(context_low.norm(dim=-1).mean())
        ok = diff > 1e-5 and low_norm < high_norm
        print(
            f"Robot-world relation preserves scene confidence {'passed' if ok else 'failed'} | "
            f"diff={diff:.3e}, high_norm={high_norm:.3e}, low_norm={low_norm:.3e}")
        return bool(ok)

    def TestRobotWorldRelationLowConfidenceAMP(self) -> bool:
        logits = torch.zeros(1, 2, device=self.device, dtype=torch.float16)
        confidence = torch.tensor(
            [[1e-5, 1e-6]], device=self.device, dtype=torch.float16)
        probability = RobotWorldRelationEncoder._masked_confidence_softmax(
            logits,
            confidence,
            torch.ones_like(confidence, dtype=torch.bool),
            dim=-1)
        expected = confidence.float() / confidence.float().sum(dim=-1, keepdim=True)
        ok = (
            probability.dtype == torch.float16
            and torch.allclose(probability.float(), expected, atol=2e-3, rtol=2e-3))
        print(f"Robot-world relation low-confidence AMP {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestRobotWorldRelationEmptySceneIsZero(self) -> bool:
        B = 2
        wm = self.wm.eval()
        physical = self.MakePhysicalState(wm, B, activeSlots=0)
        context = wm.robot_world_relation(
            self.MakeRobotSelfState(wm, B),
            physical,
            torch.randn(B, wm.action_dim, device=self.device))
        max_value = float(context.abs().max())
        ok = context.shape == (B, wm.robot_world_dim) and bool(torch.isfinite(context).all().item()) and max_value == 0.0
        print(f"Robot-world relation empty scene is zero {'passed' if ok else 'failed'} | max={max_value:.3e}")
        return bool(ok)

    def TestRobotWorldRelationSlotPermutationInvariant(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical_a = self.MakePhysicalState(wm, B, activeSlots=4)
        physical_a["PairwiseRelation"][:, :4, :4, :4] = torch.randn(
            B, 4, 4, 4, device=self.device)
        relation_prob = torch.rand(B, 4, 4, wm.physical_relation_classes, device=self.device)
        physical_a["PairwiseRelation"][:, :4, :4, 4:] = relation_prob
        diagonal = torch.arange(4, device=self.device)
        physical_a["PairwiseRelation"][:, diagonal, diagonal] = 0.0

        permutation = torch.arange(wm.physical_slots - 1, -1, -1, device=self.device)
        physical_b = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in physical_a.items()}
        slot_fields = [
            "SlotPresence", "MphysRaw", "SlotState", "PoseWorld", "ARaw", "Size",
            "StateRaw", "AffordanceRaw", "MotionRaw", "ExternalRelationProbRaw",
            "ContactProbRaw", "MovingProbRaw", "ContactForceRaw", "ContactPointRaw",
            "Visibility", "Occlusion", "Observed", "LastSeen"]
        if "ContactPointWorldRaw" in physical_a:
            slot_fields.append("ContactPointWorldRaw")
        for key in slot_fields:
            physical_b[key] = physical_a[key].index_select(1, permutation)
        physical_b["PairwiseRelation"] = physical_a["PairwiseRelation"].index_select(
            1, permutation).index_select(2, permutation)
        if "PairRelationLastSeen" in physical_a:
            physical_b["PairRelationLastSeen"] = physical_a["PairRelationLastSeen"].index_select(
                1, permutation).index_select(2, permutation)

        robot = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        context_a = wm.robot_world_relation(robot, physical_a, action)
        context_b = wm.robot_world_relation(robot, physical_b, action)
        diff = float((context_a - context_b).abs().max())
        ok = diff < 1e-5
        print(f"Robot-world relation slot permutation invariant {'passed' if ok else 'failed'} | max_diff={diff:.3e}")
        return bool(ok)

    def TestRobotWorldRelationInputContract(self) -> bool:
        B = 1
        wm = self.wm.eval()
        physical = self.MakePhysicalState(wm, B, activeSlots=2)
        physical["PairwiseRelation"] = physical["PairwiseRelation"][..., :-1]
        try:
            wm.robot_world_relation(
                self.MakeRobotSelfState(wm, B),
                physical,
                torch.randn(B, wm.action_dim, device=self.device))
        except ValueError as error:
            ok = "PairwiseRelation" in str(error)
            print(f"Robot-world relation input contract {'passed' if ok else 'failed'}")
            return bool(ok)
        print("Robot-world relation input contract failed")
        return False

    def TestRobotSelfStateRoundTrip(self) -> bool:
        B = 2
        wm = self.wm.eval()
        wm.ResetState(batchSize=B)
        camera_pose = torch.zeros(B, wm.physical_pose_dim, device=self.device)
        camera_pose[:, 6] = 1.0
        robot_self = self.MakeRobotSelfState(wm, B)
        action = torch.randn(B, wm.action_dim, device=self.device)
        wm.UpdatePhysicalState(
            self.MakePhysicalState(wm, B),
            cameraPoseWorld=camera_pose,
            robotSelfState=robot_self,
            executedActionEmbed=action,)
        physical_export = wm.ExportPhysicalState()
        robot_export = wm.ExportRobotSelfState()
        wm._robot_self_state.zero_()
        wm._robot_action_context.zero_()
        wm.ImportRobotSelfState(robot_export)
        ok = (
            "RobotSelfState" not in physical_export
            and "ExecutedAction" not in physical_export
            and torch.allclose(wm._robot_self_state, robot_export["RobotSelfState"])
            and torch.allclose(wm._robot_action_context, robot_export["ExecutedAction"]))
        print(f"Robot self state roundtrip {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestWorldStateImportUsesModelPlacement(self) -> bool:
        wm = self.wm.eval()
        source_dtype = torch.float64 if wm.dtype != torch.float64 else torch.float32
        wm.ImportState(
            torch.randn(2, wm.deter_dim, device=self.device, dtype=source_dtype),
            torch.randn(2, wm.stoch_dim, device=self.device, dtype=source_dtype),
            torch.randn(2, wm.ssm_dim, device=self.device, dtype=source_dtype))
        ok = all(
            value.device == wm.device and value.dtype == wm.dtype
            for value in wm.ExportState())
        print(f"World state import placement {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestForwardTrainFiniteGrad(self) -> bool:
        B = 3
        wm = self.wm.train()
        wm.ResetState(batchSize=B)
        out = wm.ForwardTrain(
            torch.randn(B, wm.vision_dim, device=self.device),
            physicalState=self.MakePhysicalState(wm, B),
            actionEnc=torch.randn(B, wm.action_dim, device=self.device),
            robotSelfState=self.MakeRobotSelfState(wm, B),
            reward=torch.zeros(B, device=self.device),
            done=torch.zeros(B, device=self.device),
            sample=False,)
        loss = out["loss"]
        wm.zero_grad(set_to_none=True)
        loss.backward()
        ok = bool(torch.isfinite(loss).item())
        print(f"ForwardTrain finite grad {'passed' if ok else 'failed'}")
        return ok

    def TestForwardTrainEvalIsDeterministicAndMemoryReadOnly(self) -> bool:
        wm = self.wm.eval()
        original_use_memory = wm._use_memory
        try:
            wm._use_memory = True
            B = 2
            wm.ResetState(batchSize=B)
            wm.ResetMemory()
            wm.EnsureB(B, wm.device, wm.dtype)
            wm.MemAdd(
                F.normalize(torch.randn(B, wm.stoch_dim, device=self.device), dim=-1),
                torch.randn(B, wm.state_dim, device=self.device),
                torch.ones(B, device=self.device))
            memory_before = {
                name: value.detach().clone()
                for name, value in wm.named_buffers()
                if name.startswith("_mem_")}
            state_before = tuple(value.detach().clone() for value in wm.ExportState())
            vision = torch.randn(B, wm.vision_dim, device=self.device)
            physical = self.MakePhysicalState(wm, B, activeSlots=2)
            action = torch.randn(B, wm.action_dim, device=self.device)
            robot = self.MakeRobotSelfState(wm, B)

            first = wm.ForwardTrain(
                vision,
                physical,
                actionEnc=action,
                robotSelfState=robot,
                reward=None,
                done=None,
                updateMemory=False)
            wm.ImportState(*state_before)
            wm._A_prev = None
            second = wm.ForwardTrain(
                vision,
                physical,
                actionEnc=action,
                robotSelfState=robot,
                reward=None,
                done=None,
                updateMemory=False)

            memory_after = dict(wm.named_buffers())
            memory_unchanged = all(
                name in memory_after and torch.equal(memory_after[name], before)
                for name, before in memory_before.items())
            deterministic = all(
                torch.equal(first[name], second[name])
                for name in ("h_next", "z_next", "x_next", "s_next", "r_pred", "d_prob"))
            finite = bool(torch.isfinite(first["loss"]).item())
            optional_targets_zero = (
                float(first["loss_reward"].item()) == 0.0
                and float(first["loss_done"].item()) == 0.0)
            ok = memory_unchanged and deterministic and finite and optional_targets_zero
            print(
                f"ForwardTrain eval contract {'passed' if ok else 'failed'} "
                f"| memory={memory_unchanged}, deterministic={deterministic}, finite={finite}")
            return bool(ok)
        finally:
            wm._use_memory = original_use_memory
            wm.ResetMemory()

    def TestTrainingMemoryRetrievalBackpropagatesToKey(self) -> bool:
        wm = self.wm.train()
        original_use_memory = wm._use_memory
        original_topk = wm._mem_topk
        try:
            wm._use_memory = True
            wm._mem_topk = 2
            B = 1
            wm.ResetState(batchSize=B)
            wm.ResetMemory()
            wm.EnsureB(B, wm.device, wm.dtype)
            wm.MemAdd(
                F.normalize(torch.randn(B, wm.stoch_dim, device=self.device), dim=-1),
                torch.randn(B, wm.state_dim, device=self.device),
                torch.ones(B, device=self.device))
            wm.MemAdd(
                F.normalize(torch.randn(B, wm.stoch_dim, device=self.device), dim=-1),
                torch.randn(B, wm.state_dim, device=self.device),
                torch.ones(B, device=self.device))
            wm.zero_grad(set_to_none=True)
            out = wm.ForwardTrain(
                torch.randn(B, wm.vision_dim, device=self.device),
                physicalState=self.MakePhysicalState(wm, B, activeSlots=2),
                actionEnc=torch.randn(B, wm.action_dim, device=self.device),
                robotSelfState=self.MakeRobotSelfState(wm, B),
                reward=torch.zeros(B, device=self.device),
                done=torch.zeros(B, device=self.device),
                sample=False,
                updateMemory=False)
            out["loss"].backward()
            grad = wm.key_emb.to_gb.target.weight.grad
            ok = (
                grad is not None
                and bool(torch.isfinite(grad).all().item())
                and float(grad.abs().sum().item()) > 0.0)
            print(f"Memory key retrieval gradient {'passed' if ok else 'failed'}")
            return bool(ok)
        finally:
            wm._use_memory = original_use_memory
            wm._mem_topk = original_topk
            wm.ResetMemory()

    def TestRewardDoneUsePreMemoryDynamics(self) -> bool:
        wm = self.wm.eval()
        original_use_memory = wm._use_memory
        original_topk = wm._mem_topk
        try:
            wm._use_memory = True
            wm._mem_topk = 1
            B = 1
            vision = torch.randn(B, wm.vision_dim, device=self.device)
            action = torch.randn(B, wm.action_dim, device=self.device)
            physical = self.MakePhysicalState(wm, B, activeSlots=2)
            robot = self.MakeRobotSelfState(wm, B)
            initial = (
                torch.randn(B, wm.deter_dim, device=self.device),
                torch.randn(B, wm.stoch_dim, device=self.device),
                torch.randn(B, wm.ssm_dim, device=self.device))
            memory_key = F.normalize(
                torch.randn(B, wm.stoch_dim, device=self.device), dim=-1)

            def run(memory_value: torch.Tensor) -> Dict[str, torch.Tensor]:
                wm.ResetState(batchSize=B)
                wm.ResetMemory()
                wm.ImportState(*initial)
                wm.MemAdd(memory_key, memory_value, torch.ones(B, device=self.device))
                return wm.StepPosterior(
                    vision, action, physical, robot, sample=False)

            first = run(torch.zeros(B, wm.state_dim, device=self.device))
            second = run(torch.linspace(
                -100.0, 100.0, wm.state_dim, device=self.device).view(B, -1))
            state_changes = float((first["s_next"] - second["s_next"]).abs().max().item()) > 1e-7
            predictions_stable = (
                torch.equal(first["r_pred"], second["r_pred"])
                and torch.equal(first["d_prob"], second["d_prob"]))
            ok = state_changes and predictions_stable
            print(
                f"Reward/done pre-memory dynamics {'passed' if ok else 'failed'} "
                f"| state_changes={state_changes}, predictions={predictions_stable}")
            return bool(ok)
        finally:
            wm._use_memory = original_use_memory
            wm._mem_topk = original_topk
            wm.ResetMemory()

    def TestOnlineForwardEvalAcceptsValidationControls(self) -> bool:
        wm = self.wm.eval()
        original_use_memory = wm._use_memory
        trainability = {name: parameter.requires_grad for name, parameter in wm.named_parameters()}
        try:
            wm._use_memory = True
            B = 1
            wm.ResetState(batchSize=B)
            wm.ResetMemory()
            wm.EnsureB(B, wm.device, wm.dtype)
            wm.MemAdd(
                F.normalize(torch.randn(B, wm.stoch_dim, device=self.device), dim=-1),
                torch.randn(B, wm.state_dim, device=self.device),
                torch.ones(B, device=self.device))
            memory_before = {
                name: value.detach().clone()
                for name, value in wm.named_buffers()
                if name.startswith("_mem_")}
            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()
            out = wrapper(
                torch.randn(B, wm.vision_dim, device=self.device),
                actionEnc=torch.randn(B, wm.action_dim, device=self.device),
                robotSelfState=self.MakeRobotSelfState(wm, B),
                physicalState=self.MakePhysicalState(wm, B, activeSlots=2),
                reward=None,
                done=None,
                sample=False,
                updateMemory=False)
            memory_after = dict(wm.named_buffers())
            memory_unchanged = all(
                name in memory_after and torch.equal(memory_after[name], before)
                for name, before in memory_before.items())
            ok = (
                memory_unchanged
                and bool(torch.isfinite(out["loss"]).item())
                and float(out["loss_reward"].item()) == 0.0
                and float(out["loss_done"].item()) == 0.0)
            print(f"Online validation controls {'passed' if ok else 'failed'}")
            return bool(ok)
        finally:
            wm._use_memory = original_use_memory
            wm.ResetMemory()
            for name, parameter in wm.named_parameters():
                parameter.requires_grad_(trainability[name])

    def TestWorldForwardIOShapes(self) -> bool:
        B = 2
        wm = self.wm.train()
        wm.ResetState(batchSize=B)
        out = wm.ForwardTrain(
            torch.randn(B, wm.vision_dim, device=self.device),
            physicalState=self.MakePhysicalState(wm, B),
            actionEnc=torch.randn(B, wm.action_dim, device=self.device),
            robotSelfState=self.MakeRobotSelfState(wm, B),
            reward=torch.zeros(B, device=self.device),
            done=torch.zeros(B, device=self.device),
            sample=False,)
        ok = (
            out["h_next"].shape == (B, wm.deter_dim)
            and out["z_next"].shape == (B, wm.stoch_dim)
            and out["z_next_raw"].shape == (B, wm.stoch_dim)
            and out["x_next"].shape == (B, wm.ssm_dim)
            and out["s_next"].shape == (B, wm.state_dim)
            and out["action_enc"].shape == (B, wm.action_dim)
            and out["pst_binding"]["bound_mu"].shape == (B, wm.stoch_dim))
        print(f"WorldForward IO shapes {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestPredictionLossEmptyTargetsAreFinite(self) -> bool:
        wm = self.wm.eval()
        B = 2
        prediction = wm.BuildPredictedVisual(
            torch.randn(B, wm.state_dim, device=self.device))

        class TargetVisual:
            pass

        target = TargetVisual()
        target.GlobalFeat = torch.randn(B, wm.global_feat_dim, device=self.device)
        target.ObjectTokens = torch.randn(
            B, wm.num_object_tokens, wm.object_token_dim, device=self.device)
        target.IntegratedFeat = torch.randn(B, wm.integrated_feat_dim, device=self.device)
        target.MotionToken = torch.randn(B, wm.motion_pred_dim, device=self.device)
        target.Auxiliary = {
            "ObjectGeometryValid": torch.zeros(
                B, wm.num_object_tokens, 1, device=self.device)}
        target.SemanticNodes = {
            "node_logits": torch.randn(
                B, wm.num_object_tokens, 2, device=self.device)}
        losses = wm.ComputePredictionLoss(
            prediction["predicted_visual"],
            prediction["reconstructed_visual_state"],
            target,
            precision=torch.ones(B, device=self.device))
        finite = all(bool(torch.isfinite(value).item()) for value in losses.values())
        object_losses_zero = all(
            float(losses[name].item()) == 0.0
            for name in (
                "loss_pred_inverse_object",
                "loss_pred_inverse_slot",
                "loss_pred_inverse_relation",
                "loss_pred_inverse_presence",
                "loss_pred_inverse_scene",
                "loss_pred_inverse_summary"))
        motion_supervised = float(losses["loss_pred_inverse_motion"].item()) > 0.0
        ok = finite and object_losses_zero and motion_supervised
        print(f"Empty-target prediction loss {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestPredictionLossHonorsSampleMask(self) -> bool:
        wm = self.wm.eval()
        B = 2
        prediction = wm.BuildPredictedVisual(
            torch.randn(B, wm.state_dim, device=self.device))

        class TargetVisual:
            pass

        target = TargetVisual()
        target.GlobalFeat = torch.randn(B, wm.global_feat_dim, device=self.device)
        target.ObjectTokens = torch.randn(
            B, wm.num_object_tokens, wm.object_token_dim, device=self.device)
        target.IntegratedFeat = torch.randn(B, wm.integrated_feat_dim, device=self.device)
        target.MotionToken = torch.randn(B, wm.motion_pred_dim, device=self.device)
        target.Auxiliary = {
            "ObjectGeometryValid": torch.ones(
                B, wm.num_object_tokens, 1, device=self.device)}
        target.SemanticNodes = {
            "node_logits": torch.randn(
                B, wm.num_object_tokens, 2, device=self.device)}
        sample_mask = torch.tensor([1.0, 0.0], device=self.device)
        baseline = wm.ComputePredictionLoss(
            prediction["predicted_visual"],
            prediction["reconstructed_visual_state"],
            target,
            precision=torch.ones(B, device=self.device),
            sampleMask=sample_mask)

        target.GlobalFeat[1].add_(100.0)
        target.ObjectTokens[1].add_(100.0)
        target.IntegratedFeat[1].add_(100.0)
        target.MotionToken[1].add_(100.0)
        changed = wm.ComputePredictionLoss(
            prediction["predicted_visual"],
            prediction["reconstructed_visual_state"],
            target,
            precision=torch.ones(B, device=self.device),
            sampleMask=sample_mask)
        ok = all(
            torch.allclose(baseline[name], changed[name], atol=1e-7, rtol=0.0)
            for name in baseline)
        print(f"Prediction sample mask {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestWorldAbstractShapes(self) -> bool:
        B = 2
        wm = self.wm.train()
        wm.ResetState(batchSize=B)
        physical = self.MakePhysicalState(wm, B)
        out = wm.ForwardTrain(
            torch.randn(B, wm.vision_dim, device=self.device),
            physicalState=physical,
            actionEnc=torch.randn(B, wm.action_dim, device=self.device),
            robotSelfState=self.MakeRobotSelfState(wm, B),
            reward=torch.zeros(B, device=self.device),
            done=torch.zeros(B, device=self.device),
            sample=False,)
        abstract = wm.BuildWorldAbstract(
            out,
            physical,
            torch.randn(B, wm.physical_slot_dim, device=self.device),
            torch.zeros(B, device=self.device),
            torch.ones(B, device=self.device),)
        ok = (
            abstract["world_hzx"].shape == (B, wm.deter_dim + wm.stoch_dim + wm.ssm_dim)
            and abstract["world_state"].shape == (B, wm.state_dim)
            and abstract["pst_summary"].shape == (B, wm.physical_slot_dim)
            and abstract["pst_context"].shape == (B, wm.physical_slot_dim)
            and abstract["abstract_feat"].shape == (B, wm.state_dim)
            and abstract["slot_presence_mask"].shape == (B, wm.physical_slots)
            and abstract["physical_entity_mask"].shape == (B, wm.physical_slots))
        print(f"WorldAbstract shapes {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestScoreDecisionImaginationsShapes(self) -> bool:
        B, N, T = 2, 3, 1
        wm = self.wm.eval()
        out = wm.ScoreDecisionImaginations(
            torch.randn(B, wm.deter_dim, device=self.device),
            torch.randn(B, wm.stoch_dim, device=self.device),
            torch.randn(B, wm.ssm_dim, device=self.device),
            torch.randn(B, N, T, wm.action_dim, device=self.device),
            self.MakePhysicalState(wm, B),
            self.MakeRobotSelfState(wm, B),)
        shapes_ok = (
            out["score"].shape == (B, N)
            and out["continue_prob"].shape == (B, N)
            and out["terminal_h"].shape == (B, N, wm.deter_dim)
            and out["terminal_z"].shape == (B, N, wm.stoch_dim)
            and out["terminal_x"].shape == (B, N, wm.ssm_dim))
        rejects_stale_multistep = False
        try:
            wm.ScoreDecisionImaginations(
                torch.randn(B, wm.deter_dim, device=self.device),
                torch.randn(B, wm.stoch_dim, device=self.device),
                torch.randn(B, wm.ssm_dim, device=self.device),
                torch.randn(B, N, 2, wm.action_dim, device=self.device),
                self.MakePhysicalState(wm, B),
                self.MakeRobotSelfState(wm, B))
        except ValueError as error:
            rejects_stale_multistep = "T=1" in str(error)
        ok = shapes_ok and rejects_stale_multistep
        print(f"ScoreDecisionImaginations shapes {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestLossDecrease(self) -> bool:
        B = 8
        wm = RSSMWorldModel(
            visionDim=32,
            actionDim=ModuleDim.DecisionFeedbackEmbedDim,
            deterDim=32,
            stochDim=8,
            stateDim=32,
            ssmDim=16,
            useDecoder=False,
            useMemory=False,
            nsEnabled=False,
            physicalSlots=8,
            physicalSlotDim=32,
            physicalPoseDim=7,
            physicalAttrDim=8,
            physicalIdDim=32,
            physicalRelDim=36,
            physicalRelationClasses=32,
            physicalSemanticDim=16,
            physicalStateDim=8,
            physicalAffordanceDim=4,
            physicalTextDim=4,
            physicalSymbolDim=8,).to(self.device).train()
        opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
        vision = torch.randn(B, wm.vision_dim, device=self.device)
        action = torch.randn(B, wm.action_dim, device=self.device)
        reward = torch.ones(B, device=self.device) * 0.25
        done = torch.zeros(B, device=self.device)
        losses: List[float] = []
        for _ in range(12):
            wm.ResetState(batchSize=B)
            out = wm.ForwardTrain(
                vision,
                physicalState=self.MakePhysicalState(wm, B),
                actionEnc=action,
                robotSelfState=self.MakeRobotSelfState(wm, B),
                reward=reward,
                done=done,
                sample=False,)
            loss = out["loss_reward"] + out["loss_done"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        ok = losses[-1] <= losses[0]
        print(f"LossDecrease {'passed' if ok else 'failed'} | {losses[0]:.6f}->{losses[-1]:.6f}")
        return bool(ok)

    def TestConnRegReset(self) -> bool:
        B = 2
        wm = self.wm.train()
        wm.ResetState(batchSize=B)
        _ = wm.ForwardTrain(
            torch.randn(B, wm.vision_dim, device=self.device),
            physicalState=self.MakePhysicalState(wm, B),
            actionEnc=torch.randn(B, wm.action_dim, device=self.device),
            robotSelfState=self.MakeRobotSelfState(wm, B),
            reward=torch.zeros(B, device=self.device),
            done=torch.zeros(B, device=self.device),
            sample=False,)
        had_prev = wm._A_prev is not None
        wm.ResetState(batchSize=B)
        ok = had_prev and (wm._A_prev is None)
        print(f"ConnReg reset {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestConnTransportSupportsCPUHalfTypes(self) -> bool:
        conn = ConnNet(stateDim=4, actDim=2)
        ok = True
        try:
            for dtype in (torch.float16, torch.bfloat16):
                generator = torch.Generator(device="cpu").manual_seed(7)
                state = torch.randn(2, 4, generator=generator, dtype=torch.float32).to(dtype)
                raw = torch.randn(2, 4, 4, generator=generator, dtype=torch.float32)
                skew = (0.05 * (raw - raw.transpose(1, 2))).to(dtype)
                transported = conn.TransportApply(skew, state)
                ok = (
                    ok
                    and transported.dtype == dtype
                    and bool(torch.isfinite(transported).all().item()))
        except RuntimeError:
            ok = False
        print(f"Conn transport CPU half types {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestLoadLegacyMemoryOnlyPayload(self) -> bool:
        wm = self.wm.eval()
        original_use_memory = wm._use_memory
        try:
            wm._use_memory = True
            B, C = 1, 2
            payload = {
                "mem_keys": torch.randn(B, C, wm.stoch_dim),
                "mem_vals": torch.randn(B, C, wm.state_dim),
                "mem_imp": torch.tensor([[0.4, 0.8]]),
                "mem_steps": torch.tensor([[3, 7]], dtype=torch.long),
                "mem_size": torch.tensor([C], dtype=torch.long),
                "mem_global_step": torch.tensor([7], dtype=torch.long)}
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "legacy_world_memory.pt")
                torch.save(payload, path)
                wm.LoadMemory(path, strict=False)
            ok = (
                int(wm._mem_size[0].item()) == C
                and torch.allclose(wm._mem_keys[0, :C].cpu(), payload["mem_keys"][0])
                and float(wm._pst_slot_presence.abs().sum().item()) == 0.0
                and int(wm._pst_pair_last_seen.abs().sum().item()) == 0)
            print(f"Legacy memory-only load {'passed' if ok else 'failed'}")
            return bool(ok)
        finally:
            wm._use_memory = original_use_memory
            wm.ResetMemory()

    def TestExportWorldMemoryBank(self) -> bool:
        wm = RSSMWorldModel(
            visionDim=32,
            actionDim=ModuleDim.DecisionFeedbackEmbedDim,
            deterDim=32,
            stochDim=8,
            stateDim=16,
            ssmDim=16,
            useDecoder=False,
            useMemory=True,
            memoryCapacity=8,
            nsEnabled=False,).to(self.device).eval()
        B = 1
        wm.ResetState(batchSize=B)
        wm.ResetMemory()
        with torch.no_grad():
            wm._mem_size.fill_(3)
            wm._mem_imp.zero_()
            wm._mem_steps.zero_()
            wm._mem_vals.zero_()
            wm._mem_keys.zero_()
            wm._mem_imp[0, :3] = torch.tensor([0.2, 0.9, 0.5], device=self.device)
            wm._mem_steps[0, :3] = torch.tensor([1, 2, 3], device=self.device)
            wm._mem_vals[0, :3, 0] = torch.tensor([1.0, 2.0, 3.0], device=self.device)
            wm._mem_keys[0, :3, 0] = torch.tensor([4.0, 5.0, 6.0], device=self.device)
        out = wm.ExportWorldMemoryBank(topk=2)
        ok = (
            out is not None
            and out["vals"].shape == (B, 2, wm.state_dim)
            and out["keys"].shape == (B, 2, wm.stoch_dim)
            and out["idx"].shape == (B, 2)
            and out["steps"].shape == (B, 2))
        print(f"ExportWorldMemoryBank {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestExportWorldMemoryBankLatestFirst(self) -> bool:
        wm = RSSMWorldModel(
            visionDim=32,
            actionDim=ModuleDim.DecisionFeedbackEmbedDim,
            deterDim=32,
            stochDim=8,
            stateDim=16,
            ssmDim=16,
            useDecoder=False,
            useMemory=True,
            memoryCapacity=8,
            nsEnabled=False,).to(self.device).eval()
        wm.ResetState(batchSize=1)
        wm.ResetMemory()
        with torch.no_grad():
            wm._mem_size.fill_(3)
            wm._mem_imp[0, :3] = torch.tensor([0.9, 0.8, 0.7], device=self.device)
            wm._mem_steps[0, :3] = torch.tensor([30, 10, 20], device=self.device)
            wm._mem_vals[0, :3, 0] = torch.tensor([30.0, 10.0, 20.0], device=self.device)
        out = wm.ExportWorldMemoryBank(topk=3)
        ok = out is not None and torch.equal(out["steps"][0], torch.tensor([30, 20, 10], device=self.device))
        print(f"ExportWorldMemoryBank latest-first {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestReorderMemorySteps(self) -> bool:
        wm = RSSMWorldModel(
            visionDim=32,
            actionDim=ModuleDim.DecisionFeedbackEmbedDim,
            deterDim=32,
            stochDim=8,
            stateDim=16,
            ssmDim=16,
            useDecoder=False,
            useMemory=True,
            memoryCapacity=8,
            nsEnabled=False,).to(self.device).eval()
        wm.ResetState(batchSize=1)
        wm.ResetMemory()
        with torch.no_grad():
            wm._mem_size.fill_(3)
            wm._mem_steps[0, :3] = torch.tensor([30, 10, 20], device=self.device)
            wm._mem_imp[0, :3] = torch.tensor([0.9, 0.8, 0.7], device=self.device)
        wm.ReorderMemorySteps()
        ok = torch.equal(wm._mem_steps[0, :3], torch.tensor([3, 1, 2], device=self.device))
        print(f"ReorderMemorySteps {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestWrapperAPIBasics(self) -> bool:
        B = 3
        wm = self.wm.eval()
        wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()
        wm.ResetState(batchSize=B)
        out = wrapper(
            torch.randn(B, wm.vision_dim, device=self.device),
            actionEnc=torch.randn(B, wm.action_dim, device=self.device),
            physicalState=self.MakePhysicalState(wm, B),
            robotSelfState=self.MakeRobotSelfState(wm, B),
            reward=torch.zeros(B, device=self.device),
            done=torch.zeros(B, device=self.device),
            sample=False,)
        ok = (
            out["h_next"].shape == (B, wm.deter_dim)
            and out["z_next"].shape == (B, wm.stoch_dim)
            and out["s_next"].shape == (B, wm.state_dim)
            and out["d_prob"].shape == (B,))
        print(f"Wrapper API basics {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestForwardWithDeltasInjection(self) -> bool:
        B = 3
        wm = self.wm.eval()
        wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()
        vision = torch.randn(B, wm.vision_dim, device=self.device)
        action = torch.randn(B, wm.action_dim, device=self.device)
        physical = self.MakePhysicalState(wm, B)
        robot_self = self.MakeRobotSelfState(wm, B)
        reward = torch.zeros(B, device=self.device)
        done = torch.zeros(B, device=self.device)
        site = "act_proj"
        deltaW = torch.randn(*wm.act_proj[0].target.weight.shape, device=self.device) * 1e-3
        wm.ResetState(batchSize=B)
        out0 = wrapper.ForwardWithDeltas(
            vision, None, None, None, [{}],
            actionEnc=action, physicalState=physical, robotSelfState=robot_self, reward=reward, done=done, sample=False,)
        wm.ResetState(batchSize=B)
        out1 = wrapper.ForwardWithDeltas(
            vision, None, None, None, [{site: deltaW}],
            actionEnc=action, physicalState=physical, robotSelfState=robot_self, reward=reward, done=done, sample=False,)
        diff = float((out0["s_next"] - out1["s_next"]).abs().mean())
        ok = diff > 1e-8
        print(f"ForwardWithDeltas injection {'passed' if ok else 'failed'} | |delta|={diff:.3e}")
        return bool(ok)

    def TestCommitOneGrowAndValueChange(self) -> bool:
        B = 3
        wm = self.wm.eval()
        wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()
        lo = wm.act_proj[0]
        n0 = len(lo.A_list)
        A = torch.randn(2, lo.target.in_features, device=self.device) * 1e-2
        Bm = torch.randn(lo.target.out_features, 2, device=self.device) * 1e-2
        ok_commit = wrapper.CommitOne("act_proj", 0, A, Bm, 5.0)
        n1 = len(lo.A_list)
        vision = torch.randn(B, wm.vision_dim, device=self.device)
        action = torch.randn(B, wm.action_dim, device=self.device)
        physical = self.MakePhysicalState(wm, B)
        robot_self = self.MakeRobotSelfState(wm, B)
        wm.ResetState(batchSize=B)
        with torch.no_grad():
            last_s = lo.alpha[-1].clone()
            lo.alpha[-1].zero_()
            out0 = wrapper(vision, actionEnc=action, physicalState=physical, robotSelfState=robot_self, reward=torch.zeros(B, device=self.device), done=torch.zeros(B, device=self.device), sample=False)
            lo.alpha[-1].copy_(last_s)
            wm.ResetState(batchSize=B)
            out1 = wrapper(vision, actionEnc=action, physicalState=physical, robotSelfState=robot_self, reward=torch.zeros(B, device=self.device), done=torch.zeros(B, device=self.device), sample=False)
        diff = float((out0["s_next"] - out1["s_next"]).abs().mean())
        ok = ok_commit and n1 == n0 + 1 and diff > 1e-8
        print(f"CommitOne grow & effect {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestGradFlowCandidates(self) -> bool:
        B = 3
        wm = self.wm.train()
        wrapper = WorldOnlineWrapper(wm, initRankEach=1, autoRank=False).to(self.device).train()
        wm.ResetState(batchSize=B)
        out = wrapper(
            torch.randn(B, wm.vision_dim, device=self.device),
            actionEnc=torch.randn(B, wm.action_dim, device=self.device),
            physicalState=self.MakePhysicalState(wm, B),
            robotSelfState=self.MakeRobotSelfState(wm, B),
            reward=torch.zeros(B, device=self.device),
            done=torch.zeros(B, device=self.device),
            sample=False,)
        loss = out["loss"]
        wrapper.zero_grad(set_to_none=True)
        loss.backward()
        ok = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in wrapper.CandParameters())
        print(f"Grad flow candidates {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestWrapperUpdateInjectLoRA(self) -> bool:
        B = 3
        wm = self.wm.eval()
        wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).eval()
        wrapper.Update("reset", initRankEach=0)
        wrapper.Update("grow", growFactor=1.0, addEach=1)
        ok = "act_proj" in wrapper.cand and len(wrapper.cand["act_proj"][0]["A"]) > 0
        print(f"Wrapper Update-inject LoRA {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestPSTHungarianAssignmentIdentitySwap(self) -> bool:
        try:
            wm = self.wm.eval()
            B = 1
            wm.ResetState(batchSize=B)
            camera_pose = torch.zeros(B, wm.physical_pose_dim, device=self.device)
            camera_pose[:, 6] = 1.0
            robot_self = self.MakeRobotSelfState(wm, B)
            action = torch.zeros(B, wm.action_dim, device=self.device)
            id_a = torch.zeros(wm.physical_id_dim, device=self.device)
            id_b = torch.zeros(wm.physical_id_dim, device=self.device)
            id_a[0] = 1.0
            id_b[1] = 1.0

            def observed_state(first: torch.Tensor, second: torch.Tensor) -> Dict[str, torch.Tensor]:
                state = wm.ExportPhysicalState()
                for key, value in state.items():
                    if torch.is_tensor(value):
                        value.zero_()
                state["ObservedSlotMask"] = state["SlotPresence"]
                state["SlotPresence"][:, :2] = 1.0
                state["ObservedSlotMask"][:, :2] = 1.0
                state["MphysRaw"][:, :2] = 1.0
                state["Observed"][:, :2] = True
                state["IdentityKey"][0, 0] = first
                state["IdentityKey"][0, 1] = second
                state["PoseCamera"] = state["PoseWorld"]
                state["PoseCamera"][0, 0, 0] = 0.0
                state["PoseCamera"][0, 1, 0] = 1.0
                state["PoseCamera"][0, :2, 6] = 1.0
                return state

            wm.UpdatePhysicalState(observed_state(id_a, id_b), cameraPoseWorld=camera_pose, robotSelfState=robot_self, executedActionEmbed=action)
            wm.UpdatePhysicalState(observed_state(id_b, id_a), cameraPoseWorld=camera_pose, robotSelfState=robot_self, executedActionEmbed=action)
            merged = wm.ExportPhysicalState()
            slot0_a = float(torch.dot(merged["IdentityKey"][0, 0], id_a).item())
            slot1_b = float(torch.dot(merged["IdentityKey"][0, 1], id_b).item())
            ok = slot0_a > 0.99 and slot1_b > 0.99 and bool(merged["Observed"][0, :2].all().item())
            print(f"PST Hungarian identity swap {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"PST Hungarian identity swap error: {e}")
            return False

    def TestPhysicalUpdateRejectsNonFiniteBeforeMutation(self) -> bool:
        wm = self.wm.eval()
        B = 1
        wm.ResetState(batchSize=B)
        wm.ImportPhysicalState(self.MakePhysicalState(wm, B, activeSlots=2))
        before_physical = wm.ExportPhysicalState()
        before_robot = wm.ExportRobotSelfState()
        observed = self.MakePhysicalState(wm, B, activeSlots=2)
        observed["PoseCamera"][0, 0, 0] = torch.nan
        camera = torch.zeros(B, wm.physical_pose_dim, device=self.device)
        camera[:, 6] = 1.0
        rejected = False
        try:
            wm.UpdatePhysicalState(
                observed,
                camera,
                self.MakeRobotSelfState(wm, B),
                torch.zeros(B, wm.action_dim, device=self.device))
        except ValueError as error:
            rejected = "finite" in str(error)
        after_physical = wm.ExportPhysicalState()
        after_robot = wm.ExportRobotSelfState()
        unchanged = (
            all(torch.equal(before_physical[key], after_physical[key]) for key in before_physical)
            and all(torch.equal(before_robot[key], after_robot[key]) for key in before_robot))
        ok = rejected and unchanged
        print(
            f"PST non-finite prevalidation {'passed' if ok else 'failed'} "
            f"| rejected={rejected}, unchanged={unchanged}")
        return bool(ok)

    def TestPhysicalUpdateRejectsInvalidShapeBeforeMutation(self) -> bool:
        wm = self.wm.eval()
        B = 1
        wm.ResetState(batchSize=B)
        wm.ImportPhysicalState(self.MakePhysicalState(wm, B, activeSlots=2))
        before_physical = wm.ExportPhysicalState()
        before_robot = wm.ExportRobotSelfState()
        observed = self.MakePhysicalState(wm, B, activeSlots=2)
        observed["ParentProb"] = observed["ParentProb"][:, :, :-1]
        camera = torch.zeros(B, wm.physical_pose_dim, device=self.device)
        camera[:, 6] = 1.0
        rejected = False
        try:
            wm.UpdatePhysicalState(
                observed,
                camera,
                self.MakeRobotSelfState(wm, B),
                torch.zeros(B, wm.action_dim, device=self.device))
        except ValueError as error:
            rejected = "shape" in str(error)
        after_physical = wm.ExportPhysicalState()
        after_robot = wm.ExportRobotSelfState()
        unchanged = (
            all(torch.equal(before_physical[key], after_physical[key]) for key in before_physical)
            and all(torch.equal(before_robot[key], after_robot[key]) for key in before_robot))
        ok = rejected and unchanged
        print(
            f"PST shape prevalidation {'passed' if ok else 'failed'} "
            f"| rejected={rejected}, unchanged={unchanged}")
        return bool(ok)

    def TestPSTAssignmentKeepsLegalMatchWithUnmatchedDummies(self) -> bool:
        wm = self.wm.eval()
        B = 1
        wm.ResetState(batchSize=B)
        memory = self.MakePhysicalState(wm, B, activeSlots=2)
        memory["IdentityKey"].zero_()
        memory["IdentityKey"][0, 0, 0] = 1.0
        memory["IdentityKey"][0, 1, 1] = 1.0
        memory["PoseWorld"].zero_()
        memory["PoseWorld"][..., 6] = 1.0
        memory["PoseWorld"][0, 0, 0] = 10.0
        wm.ImportPhysicalState(memory)

        observed = self.MakePhysicalState(wm, B, activeSlots=2)
        observed["IdentityKey"].zero_()
        observed["IdentityKey"][0, 0, 0] = 0.8
        observed["IdentityKey"][0, 0, 1] = 0.6
        observed["IdentityKey"][0, 1, 0] = 0.7
        observed["IdentityKey"][0, 1, 2] = float(0.51 ** 0.5)
        observed["PoseCamera"].zero_()
        observed["PoseCamera"][..., 6] = 1.0
        observed["PoseCamera"][0, 1, 0] = 10.0
        observed["SlotState"][0, 0].fill_(42.0)
        camera = torch.zeros(B, wm.physical_pose_dim, device=self.device)
        camera[:, 6] = 1.0
        merged = wm.UpdatePhysicalState(
            observed,
            camera,
            self.MakeRobotSelfState(wm, B),
            torch.zeros(B, wm.action_dim, device=self.device))
        legal_match_kept = torch.allclose(
            merged["SlotState"][0, 0],
            observed["SlotState"][0, 0])
        print(f"PST assignment dummies {'passed' if legal_match_kept else 'failed'}")
        return bool(legal_match_kept)

    def TestPSTAssignmentSupportsHalfPrecisionCosts(self) -> bool:
        cost = torch.tensor(
            [[0.3, 0.1], [0.2, 0.9]],
            device=self.device,
            dtype=torch.float16)
        legal = torch.tensor(
            [[True, False], [False, False]],
            device=self.device)
        rows, cols = self.wm.MatchLegalPhysicalSlots(cost, legal)
        ok = (
            torch.equal(rows, torch.tensor([0], device=self.device))
            and torch.equal(cols, torch.tensor([0], device=self.device)))
        print(f"PST assignment half precision {'passed' if ok else 'failed'}")
        return bool(ok)

    def TestPSTReplacementClearsStaleRelations(self) -> bool:
        try:
            wm = self.wm.eval()
            B = 1
            wm.ResetState(batchSize=B)
            camera_pose = torch.zeros(B, wm.physical_pose_dim, device=self.device)
            camera_pose[:, 6] = 1.0
            robot_self = self.MakeRobotSelfState(wm, B)
            action = torch.zeros(B, wm.action_dim, device=self.device)

            physical = self.MakePhysicalState(wm, B, activeSlots=wm.physical_slots)
            physical["Step"].fill_(8)
            physical["LastSeen"].fill_(8)
            physical["LastSeen"][0, 0] = 0
            physical["IdentityKey"].zero_()
            physical["IdentityKey"][0, :, 0] = 1.0
            physical["PairwiseRelation"].zero_()
            physical["PairwiseRelation"][0, 0, 1, 4] = 1.0
            physical["PairwiseRelation"][0, 1, 0, 5] = 1.0
            physical["ParentProb"].zero_()
            physical["ParentProb"][0, 0, 1] = 1.0
            physical["ParentProb"][0, 1, 0] = 1.0
            wm.ImportPhysicalState(physical)

            observed = self.MakePhysicalState(wm, B, activeSlots=1)
            observed["ObservedSlotMask"].zero_()
            observed["MphysRaw"].zero_()
            observed["IdentityKey"].zero_()
            observed["PairwiseRelation"].zero_()
            observed["ParentProb"].zero_()
            observed["PoseCamera"].zero_()
            observed["PoseCamera"][0, 0, 6] = 1.0
            observed["ObservedSlotMask"][0, 0] = 1.0
            observed["MphysRaw"][0, 0] = 1.0
            observed["IdentityKey"][0, 0, 1] = 1.0

            wm.UpdatePhysicalState(
                observed,
                cameraPoseWorld=camera_pose,
                robotSelfState=robot_self,
                executedActionEmbed=action)
            merged = wm.ExportPhysicalState()
            stale_pair_relation = (
                merged["PairwiseRelation"][0, 0, 1, 4:].abs().sum()
                + merged["PairwiseRelation"][0, 1, 0, 4:].abs().sum())
            stale_parent = (
                merged["ParentProb"][0, 0, 1].abs()
                + merged["ParentProb"][0, 1, 0].abs())
            identity_replaced = float(torch.dot(
                merged["IdentityKey"][0, 0],
                observed["IdentityKey"][0, 0]).item()) > 0.99
            ok = identity_replaced and float(stale_pair_relation.item()) == 0.0 and float(stale_parent.item()) == 0.0
            print(f"PST replacement clears stale relations {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"PST replacement clears stale relations error: {e}")
            return False

    def TestPSTRelationMaskClearsSelfAndInactivePairs(self) -> bool:
        try:
            wm = self.wm.eval()
            B = 1
            wm.ResetState(batchSize=B)
            camera_pose = torch.zeros(B, wm.physical_pose_dim, device=self.device)
            camera_pose[:, 6] = 1.0
            robot_self = self.MakeRobotSelfState(wm, B)
            action = torch.zeros(B, wm.action_dim, device=self.device)

            physical = self.MakePhysicalState(wm, B, activeSlots=2)
            physical["PairwiseRelation"].zero_()
            physical["ParentProb"].zero_()
            physical["PairwiseRelation"][0, 0, 0, 4] = 1.0
            physical["PairwiseRelation"][0, 0, 3, 5] = 1.0
            physical["PairwiseRelation"][0, 3, 0, 6] = 1.0
            physical["ParentProb"][0, 0, 0] = 1.0
            physical["ParentProb"][0, 0, 3] = 1.0
            physical["ParentProb"][0, 3, 0] = 1.0
            wm.ImportPhysicalState(physical)

            observed = self.MakePhysicalState(wm, B, activeSlots=0)
            wm.UpdatePhysicalState(
                observed,
                cameraPoseWorld=camera_pose,
                robotSelfState=robot_self,
                executedActionEmbed=action)
            merged = wm.ExportPhysicalState()
            stale_pair_relation = (
                merged["PairwiseRelation"][0, 0, 0].abs().sum()
                + merged["PairwiseRelation"][0, 0, 3].abs().sum()
                + merged["PairwiseRelation"][0, 3, 0].abs().sum())
            stale_parent = (
                merged["ParentProb"][0, 0, 0].abs()
                + merged["ParentProb"][0, 0, 3].abs()
                + merged["ParentProb"][0, 3, 0].abs())
            ok = float(stale_pair_relation.item()) == 0.0 and float(stale_parent.item()) == 0.0
            print(f"PST relation mask clears self/inactive pairs {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"PST relation mask clears self/inactive pairs error: {e}")
            return False

    def TestContactPointStoredInWorldFrame(self) -> bool:
        try:
            wm = self.wm.eval()
            B = 1
            wm.ResetState(batchSize=B)
            wm.ResetPhysicalState()
            observed = self.MakePhysicalState(wm, B, activeSlots=1)
            observed["PoseCamera"].zero_()
            observed["PoseCamera"][0, 0, 0] = 1.0
            observed["PoseCamera"][0, 0, 6] = 1.0
            observed["ContactProbRaw"][0, 0] = 0.5
            # ContactPointRaw is already probability-weighted by PhysicalStateExtractor.
            observed["ContactPointRaw"][0, 0] = torch.tensor(
                [0.5, 0.0, 0.0], device=self.device)
            camera_pose = torch.tensor(
                [[10.0, 20.0, 30.0, 0.0, 0.0, 3.0 * 2.0 ** -0.5, 3.0 * 2.0 ** -0.5]],
                device=self.device)
            wm.UpdatePhysicalState(
                observed,
                cameraPoseWorld=camera_pose,
                robotSelfState=self.MakeRobotSelfState(wm, B),
                executedActionEmbed=torch.zeros(B, wm.action_dim, device=self.device))
            merged = wm.ExportPhysicalState()
            expected = torch.tensor([5.0, 10.5, 15.0], device=self.device)
            transformed = merged["ContactPointWorldRaw"][0, 0]
            expected_pose = torch.tensor([10.0, 21.0, 30.0], device=self.device)

            explicit_world_state = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in merged.items()
                if key != "ContactPointRaw"}
            explicit_context = wm.robot_world_relation(
                self.MakeRobotSelfState(wm, B),
                explicit_world_state,
                torch.zeros(B, wm.action_dim, device=self.device))
            explicit_binding = wm.pst_binder(
                torch.zeros(B, wm.deter_dim, device=self.device),
                torch.zeros(B, wm.stoch_dim, device=self.device),
                torch.zeros(B, wm.ssm_dim, device=self.device),
                explicit_world_state,
                torch.zeros(B, wm.action_dim, device=self.device),
                explicit_context)

            # Old runtime dictionaries only exposed an ambiguous camera-frame alias. Importing
            # them must not silently relabel that value as a world-frame coordinate.
            old_schema = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in merged.items()
                if key != "ContactPointWorldRaw"}
            old_schema["ContactPointRaw"].fill_(123.0)
            wm.ImportPhysicalState(old_schema)
            legacy_world = wm.ExportPhysicalState()["ContactPointWorldRaw"]
            ok = (
                torch.allclose(transformed, expected, atol=1e-5, rtol=1e-5)
                and torch.allclose(
                    merged["PoseWorld"][0, 0, :3], expected_pose, atol=1e-5, rtol=1e-5)
                and bool(torch.isfinite(explicit_context).all().item())
                and bool(torch.isfinite(explicit_binding["bound_mu"]).all().item())
                and float(legacy_world.abs().sum().item()) == 0.0)
            print(f"PST contact-point world frame {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"PST contact-point world frame error: {e}")
            return False

    def TestPairRelationRecencyTracksObservation(self) -> bool:
        try:
            wm = self.wm.eval()
            B = 1
            wm.ResetState(batchSize=B)
            wm.ResetPhysicalState()
            camera_pose = torch.zeros(B, wm.physical_pose_dim, device=self.device)
            camera_pose[:, 6] = 1.0
            robot_self = self.MakeRobotSelfState(wm, B)
            action = torch.zeros(B, wm.action_dim, device=self.device)
            observed = self.MakePhysicalState(wm, B, activeSlots=2)
            observed["PairwiseRelation"].zero_()
            observed["PairwiseRelation"][0, 0, 1, 4] = 1.0
            observed["PairwiseRelation"][0, 1, 0, 5] = 1.0
            wm.UpdatePhysicalState(observed, camera_pose, robot_self, action)
            first = wm.ExportPhysicalState()
            first_step = int(first["Step"][0].item())
            first_seen = first["PairRelationLastSeen"][0, :2, :2].clone()

            only_first = {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in observed.items()}
            only_first["ObservedSlotMask"][:, 1:] = 0.0
            only_first["MphysRaw"][:, 1:] = 0.0
            only_first["PairwiseRelation"].zero_()
            wm.UpdatePhysicalState(only_first, camera_pose, robot_self, action)
            second = wm.ExportPhysicalState()
            second_seen = second["PairRelationLastSeen"][0, :2, :2]
            ok = (
                first_step == 1
                and int(first_seen[0, 1].item()) == first_step
                and int(first_seen[1, 0].item()) == first_step
                and int(first_seen.diagonal().sum().item()) == 0
                and int(second["Step"][0].item()) == 2
                and int(second_seen[0, 1].item()) == first_step
                and int(second_seen[1, 0].item()) == first_step)
            print(f"PST pair-relation recency {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"PST pair-relation recency error: {e}")
            return False

    def TestPhysicalStateLegacyTimestampRoundTrip(self) -> bool:
        wm = self.wm.eval()
        B = 1
        try:
            legacy = self.MakePhysicalState(wm, B, activeSlots=2)
            legacy["PairwiseRelation"][0, 0, 1, 4] = 1.0
            legacy["PairwiseRelation"][0, 1, 0, 5] = 1.0
            legacy["Step"].zero_()
            legacy.pop("PairRelationLastSeen", None)
            for key, value in tuple(legacy.items()):
                if torch.is_tensor(value) and value.is_floating_point():
                    legacy[key] = value.to(dtype=torch.float64)
            wm.ImportPhysicalState(legacy)
            exported = wm.ExportPhysicalState()
            inferred = exported["PairRelationLastSeen"]
            inferred_ok = (
                int(exported["Step"][0].item()) == 1
                and
                int(inferred[0, 0, 1].item()) == int(exported["Step"][0].item())
                and int(inferred[0, 1, 0].item()) == int(exported["Step"][0].item()))
            model_placement_ok = (
                exported["SlotState"].device == wm.device
                and exported["SlotState"].dtype == wm.dtype)

            state = {
                key: value.detach().clone()
                for key, value in wm.state_dict().items()}
            persistent = "_pst_pair_last_seen" in state
            expected = inferred.clone()
            wm._pst_pair_last_seen.zero_()
            wm.load_state_dict(state, strict=True)
            state_roundtrip = torch.equal(wm._pst_pair_last_seen, expected)
            legacy_state = dict(state)
            legacy_state.pop("_pst_pair_last_seen", None)
            wm._pst_pair_last_seen.zero_()
            wm.load_state_dict(legacy_state, strict=True)
            legacy_state_roundtrip = torch.equal(wm._pst_pair_last_seen, expected)
            ok = (
                inferred_ok
                and model_placement_ok
                and persistent
                and state_roundtrip
                and legacy_state_roundtrip)
            print(
                f"Legacy pair timestamp roundtrip {'passed' if ok else 'failed'} "
                f"| inferred={inferred_ok}, placement={model_placement_ok}, persistent={persistent}")
            return bool(ok)
        finally:
            wm.to(device=self.device, dtype=torch.float32)
            wm.ResetState(batchSize=B)

    def TestPartialEpisodeResetClearsOnlyDoneRows(self) -> bool:
        try:
            wm = self.wm.eval()
            B = 2
            wm.ResetState(batchSize=B)
            wm.EnsureB(B, wm.device, wm.dtype)
            wm._h.fill_(1.0)
            wm._z.fill_(2.0)
            wm.s4.x.fill_(3.0)
            wm._A_prev = torch.full(
                (B, wm.state_dim, wm.state_dim), 4.0,
                device=self.device,
                dtype=wm.dtype)
            wm._pst_slot_presence.fill_(1.0)
            wm._pst_pairwise_relation.fill_(2.0)
            wm._pst_pair_last_seen.fill_(3)
            wm._pst_step.fill_(4)
            wm._robot_self_state.fill_(5.0)
            wm._robot_action_context.fill_(6.0)
            wm._mem_keys.fill_(7.0)
            wm._mem_size.fill_(1)
            memory_before = (wm._mem_keys.clone(), wm._mem_size.clone())

            wm.ResetEpisodeState(torch.tensor([True, False], device=self.device))
            runtime_buffers = (
                wm._h, wm._z, wm.s4.x, wm._A_prev,
                wm._pst_slot_presence, wm._pst_pairwise_relation,
                wm._pst_pair_last_seen, wm._pst_step,
                wm._robot_self_state, wm._robot_action_context)
            done_cleared = all(float(buffer[0].abs().sum().item()) == 0.0 for buffer in runtime_buffers)
            live_preserved = all(float(buffer[1].abs().sum().item()) > 0.0 for buffer in runtime_buffers)
            memory_preserved = (
                torch.equal(wm._mem_keys, memory_before[0])
                and torch.equal(wm._mem_size, memory_before[1]))
            ok = done_cleared and live_preserved and memory_preserved
            print(f"Partial episode reset {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Partial episode reset error: {e}")
            return False

    def TestOnlineRelationHeadsRemainTrainable(self) -> bool:
        try:
            wm = self.wm.train()
            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device).train()
            direct_heads = (
                wm.robot_world_relation.pair_score,
                wm.robot_world_relation.slot_score,
                wm.robot_world_relation.relation_proj[3],
                wm.embodied_action_proj[3],
                wm.pst_binder.delta_mu[1],
                wm.pst_binder.bind_gate[1])
            direct_trainable = all(
                parameter.requires_grad
                for head in direct_heads
                for parameter in head.parameters())
            upstream_frozen = not wm.robot_world_relation.slot_proj[1].target.weight.requires_grad

            wrapper.zero_grad(set_to_none=True)
            physical = self.MakePhysicalState(wm, 1, activeSlots=3)
            context = wm.robot_world_relation(
                self.MakeRobotSelfState(wm, 1),
                physical,
                torch.randn(1, wm.action_dim, device=self.device))
            probe = torch.linspace(0.5, 1.5, context.size(-1), device=self.device)
            (context * probe).sum().backward()
            relation_grad = wm.robot_world_relation.relation_proj[3].target.weight.grad
            relation_receives_grad = (
                relation_grad is not None
                and bool(torch.isfinite(relation_grad).all().item())
                and float(relation_grad.abs().sum().item()) > 0.0)
            ok = direct_trainable and upstream_frozen and relation_receives_grad
            print(f"Online relation heads trainable {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Online relation heads trainable error: {e}")
            return False

    def TestOnlineCommitKeepsOnlyDirectHeadsTrainable(self) -> bool:
        try:
            wm = self.wm.eval()
            wrapper = WorldOnlineWrapper(wm, initRankEach=0, autoRank=False).to(self.device)
            spec = wrapper.sites["act_proj"]
            candidate_a, candidate_b, candidate_scale = spec.allocFn(
                1, wrapper.deviceRef, wrapper.dtypeRef)
            slot = wrapper.cand["act_proj"][0]
            slot["A"].append(candidate_a)
            slot["B"].append(candidate_b)
            slot["s"].append(candidate_scale)
            result = wrapper.Update("commit")
            committed = (
                wm.act_proj[0].A_list[-1],
                wm.act_proj[0].B_list[-1],
                wm.act_proj[0].alpha[-1])
            direct_trainable = all(
                parameter.requires_grad
                for head in wrapper.DirectOnlineHeads()
                for parameter in head.parameters())
            ok = (
                result["commit_stats"]["committed_triples"] == 1.0
                and not any(parameter.requires_grad for parameter in committed)
                and direct_trainable
                and not wm.act_proj[0].target.weight.requires_grad)
            print(f"Online commit trainability contract {'passed' if ok else 'failed'}")
            return bool(ok)
        except Exception as e:
            print(f"Online commit trainability contract error: {e}")
            return False

    def RunAll(self) -> bool:
        results = {
            "PSTWorldBinderShapes": self.TestPSTWorldBinderShapes(),
            "RSSMStepPosterior": self.TestRSSMStepPosterior(),
            "RSSMStepPriorOnly": self.TestRSSMStepPriorOnly(),
            "PriorRolloutIsPureAndDeterministic": self.TestPriorRolloutIsPureAndDeterministic(),
            "PriorRolloutIsDeterministicWithNeSy": self.TestPriorRolloutIsDeterministicWithNeSy(),
            "PhysicsRefinerSupportsInferenceModeAndDamping": self.TestPhysicsRefinerSupportsInferenceModeAndDamping(),
            "OnlinePriorRolloutUsesCandidatesWithoutMutatingState": self.TestOnlinePriorRolloutUsesCandidatesWithoutMutatingState(),
            "RobotSelfStateAffectsWorldDynamics": self.TestRobotSelfStateAffectsWorldDynamics(),
            "RobotWorldRelationUsesPairwiseRelations": self.TestRobotWorldRelationUsesPairwiseRelations(),
            "RobotWorldRelationIgnoresInvalidSlots": self.TestRobotWorldRelationIgnoresInvalidSlots(),
            "RobotWorldRelationMasksInvalidNonFiniteValues": self.TestRobotWorldRelationMasksInvalidNonFiniteValues(),
            "RobotWorldRelationQuaternionDoubleCover": self.TestRobotWorldRelationQuaternionDoubleCover(),
            "RobotWorldRelationPreservesMetricScale": self.TestRobotWorldRelationPreservesMetricScale(),
            "RobotWorldRelationDecaysStaleRelationProbabilities": self.TestRobotWorldRelationDecaysStaleRelationProbabilities(),
            "RobotWorldRelationPreservesSceneConfidence": self.TestRobotWorldRelationPreservesSceneConfidence(),
            "RobotWorldRelationLowConfidenceAMP": self.TestRobotWorldRelationLowConfidenceAMP(),
            "RobotWorldRelationEmptySceneIsZero": self.TestRobotWorldRelationEmptySceneIsZero(),
            "RobotWorldRelationSlotPermutationInvariant": self.TestRobotWorldRelationSlotPermutationInvariant(),
            "RobotWorldRelationInputContract": self.TestRobotWorldRelationInputContract(),
            "RobotSelfStateRoundTrip": self.TestRobotSelfStateRoundTrip(),
            "WorldStateImportUsesModelPlacement": self.TestWorldStateImportUsesModelPlacement(),
            "ForwardTrainFiniteGrad": self.TestForwardTrainFiniteGrad(),
            "ForwardTrainEvalIsDeterministicAndMemoryReadOnly": self.TestForwardTrainEvalIsDeterministicAndMemoryReadOnly(),
            "TrainingMemoryRetrievalBackpropagatesToKey": self.TestTrainingMemoryRetrievalBackpropagatesToKey(),
            "RewardDoneUsePreMemoryDynamics": self.TestRewardDoneUsePreMemoryDynamics(),
            "OnlineForwardEvalAcceptsValidationControls": self.TestOnlineForwardEvalAcceptsValidationControls(),
            "WorldForwardIOShapes": self.TestWorldForwardIOShapes(),
            "PredictionLossEmptyTargetsAreFinite": self.TestPredictionLossEmptyTargetsAreFinite(),
            "PredictionLossHonorsSampleMask": self.TestPredictionLossHonorsSampleMask(),
            "WorldAbstractShapes": self.TestWorldAbstractShapes(),
            "ScoreDecisionImaginationsShapes": self.TestScoreDecisionImaginationsShapes(),
            "LossDecrease": self.TestLossDecrease(),
            "ConnRegReset": self.TestConnRegReset(),
            "ConnTransportSupportsCPUHalfTypes": self.TestConnTransportSupportsCPUHalfTypes(),
            "LoadLegacyMemoryOnlyPayload": self.TestLoadLegacyMemoryOnlyPayload(),
            "ExportWorldMemoryBank": self.TestExportWorldMemoryBank(),
            "ExportWorldMemoryBankLatestFirst": self.TestExportWorldMemoryBankLatestFirst(),
            "ReorderMemorySteps": self.TestReorderMemorySteps(),
            "WrapperAPIBasics": self.TestWrapperAPIBasics(),
            "ForwardWithDeltasInjection": self.TestForwardWithDeltasInjection(),
            "CommitOneGrowAndValueChange": self.TestCommitOneGrowAndValueChange(),
            "GradFlowCandidates": self.TestGradFlowCandidates(),
            "WrapperUpdateInjectLoRA": self.TestWrapperUpdateInjectLoRA(),
            "PSTHungarianAssignmentIdentitySwap": self.TestPSTHungarianAssignmentIdentitySwap(),
            "PhysicalUpdateRejectsNonFiniteBeforeMutation": self.TestPhysicalUpdateRejectsNonFiniteBeforeMutation(),
            "PhysicalUpdateRejectsInvalidShapeBeforeMutation": self.TestPhysicalUpdateRejectsInvalidShapeBeforeMutation(),
            "PSTAssignmentKeepsLegalMatchWithUnmatchedDummies": self.TestPSTAssignmentKeepsLegalMatchWithUnmatchedDummies(),
            "PSTAssignmentSupportsHalfPrecisionCosts": self.TestPSTAssignmentSupportsHalfPrecisionCosts(),
            "PSTReplacementClearsStaleRelations": self.TestPSTReplacementClearsStaleRelations(),
            "PSTRelationMaskClearsSelfAndInactivePairs": self.TestPSTRelationMaskClearsSelfAndInactivePairs(),
            "ContactPointStoredInWorldFrame": self.TestContactPointStoredInWorldFrame(),
            "PairRelationRecencyTracksObservation": self.TestPairRelationRecencyTracksObservation(),
            "PhysicalStateLegacyTimestampRoundTrip": self.TestPhysicalStateLegacyTimestampRoundTrip(),
            "PartialEpisodeResetClearsOnlyDoneRows": self.TestPartialEpisodeResetClearsOnlyDoneRows(),
            "OnlineRelationHeadsRemainTrainable": self.TestOnlineRelationHeadsRemainTrainable(),
            "OnlineCommitKeepsOnlyDirectHeadsTrainable": self.TestOnlineCommitKeepsOnlyDirectHeadsTrainable(),}
        passed = sum(1 for v in results.values() if v)
        print(f"\n[WorldModule Tests] {passed}/{len(results)} passed.")
        return passed == len(results)
