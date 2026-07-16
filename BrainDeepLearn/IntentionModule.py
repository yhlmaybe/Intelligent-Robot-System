from __future__ import annotations
from typing import Any, List, Tuple, Dict, Optional
from Config import BasicParameters
from CoreTypes import TEXT_TRUST_OPERATOR_COMMAND, TEXT_TRUST_UNSAFE_EXTERNAL
from FunctionTools import GetParametersScale, SiteSpec, BaseOnlineWrapper, AGICoreModule, GrowableLoRALinear, RoPEMultiheadAttention

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import math



class TextEncoder(AGICoreModule):
    def __init__(
        self,
        vocabSize: int,
        dimEmbed: int,
        dimHidden: int,
        numLayers: int = 2,
        dropout: float = 0.1,
        paddingIdx: int = 0,):
        super().__init__()
        self.padding_idx = int(paddingIdx)

        self.embedding = nn.Embedding(
            num_embeddings=vocabSize,
            embedding_dim=dimEmbed,
            padding_idx=self.padding_idx,)

        self.rnn_f = nn.GRU(
            input_size=dimEmbed,
            hidden_size=dimHidden,
            num_layers=numLayers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if numLayers > 1 else 0.0,)
        
        self.rnn_b = nn.GRU(
            input_size=dimEmbed,
            hidden_size=dimHidden,
            num_layers=numLayers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if numLayers > 1 else 0.0,)

        self.att_proj = nn.Linear(dimHidden * 2, 1)
        self.out_dim = dimHidden * 2

    def ReversePaddedSequence(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[:2]
        idx = torch.arange(T, device=self.device).unsqueeze(0).expand(B, T)  
        L = lengths.unsqueeze(1)
        rev_idx = torch.where(idx < L, (L - 1 - idx), idx)  
        rev_idx = rev_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)) 
        return x.gather(1, rev_idx) # [B, T, D]

    def forward(self, tokenIds: torch.Tensor) -> torch.Tensor:
        B, T = tokenIds.shape

        emb = self.embedding(tokenIds) # [B, T, E]

        mask = (tokenIds != self.padding_idx)  
        lengths = mask.long().sum(dim=1).clamp(min=1)  

        out_f, _ = self.rnn_f(emb) # [B, T, H]

        emb_rev = self.ReversePaddedSequence(emb, lengths)
        out_b_rev, _ = self.rnn_b(emb_rev) 
        out_b = self.ReversePaddedSequence(out_b_rev, lengths)

        out = torch.cat([out_f, out_b], dim=-1) # [B, T, 2H]

        scores = self.att_proj(out).squeeze(-1) # [B, T]
        scores = scores.masked_fill(~mask, float("-inf"))

        no_token = ~mask.any(dim=1)
        if no_token.any():
            scores = scores.clone()
            scores[no_token] = 0.0

        attn = F.softmax(scores, dim=-1)
        if no_token.any():
            attn = attn.clone()
            attn[no_token] = 0.0

        text_repr = (out * attn.unsqueeze(-1)).sum(dim=1) 
        return text_repr # [B, 2H]



class LangSymbolReasoner(AGICoreModule):
    def __init__(
        self,
        nSymbols: int,
        dimSem: int,
        hiddenDim: int = 512,
        alphaImp: float = 1.0,
        alphaCooc: float = 0.5,
        alphaContr: float = 1.0,
        dynamicScale: float = 0.25,):
        super().__init__()
        self.nSymbols = int(nSymbols)
        self.dimSem = int(dimSem)
        self.dynamic_scale = float(dynamicScale)

        self.relImp = nn.Parameter(torch.randn(self.dimSem, self.dimSem) * 0.02)
        self.relContr = nn.Parameter(torch.randn(self.dimSem, self.dimSem) * 0.02)
        self.relCooc = nn.Parameter(torch.randn(self.dimSem, self.dimSem) * 0.02)

        self.alphaImp = nn.Parameter(torch.tensor(float(alphaImp)))
        self.alphaCooc = nn.Parameter(torch.tensor(float(alphaCooc)))
        self.alphaContr = nn.Parameter(torch.tensor(float(alphaContr)))

        self.ctxBackbone = nn.Sequential(
            nn.Linear(self.dimSem, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, self.dimSem),
            nn.LayerNorm(self.dimSem),
            nn.GELU(),)

        self.alphaHead = nn.Linear(self.dimSem, 3)
        self.filmImp = nn.Linear(self.dimSem, self.dimSem * 2)
        self.filmCooc = nn.Linear(self.dimSem, self.dimSem * 2)
        self.filmContr = nn.Linear(self.dimSem, self.dimSem * 2)

        self.postMlp = nn.Sequential(
            nn.Linear(self.nSymbols * 2 + self.dimSem, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, self.nSymbols),)

    def BuildRelationMatrix(
        self,
        conceptEmb: torch.Tensor,
        relCore: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,) -> torch.Tensor:

        norm_emb = F.normalize(conceptEmb, dim=-1, eps=1e-6) # [K, D]
        scale = 1.0 / math.sqrt(float(self.dimSem))

        if (gamma is None) or (beta is None):
            interm = norm_emb @ relCore # [K, D]
            relation_matrix = interm @ norm_emb.t() # [K, K]
            return relation_matrix * scale

        B = gamma.size(0)
        emb = norm_emb.unsqueeze(0).expand(B, -1, -1) # [B, K, D]
        emb = emb * gamma.unsqueeze(1) + beta.unsqueeze(1)
        emb = F.normalize(emb, dim=-1, eps=1e-6) # [B, K, D]

        interm = torch.matmul(emb, relCore) # [B, K, D]
        relation_matrix = torch.matmul(interm, emb.transpose(1, 2)) # [B, K, K]
        return relation_matrix * scale

    def BuildFilmParams(self, ctxFeat: torch.Tensor, filmHead: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        film = filmHead(ctxFeat) # [B, 2D]
        gamma_raw, beta_raw = film.chunk(2, dim=-1) # [B, D]
        gamma = 1.0 + self.dynamic_scale * torch.tanh(gamma_raw)
        beta = self.dynamic_scale * torch.tanh(beta_raw)
        return gamma, beta # [B, D]

    def forward(
        self,
        symLogits: torch.Tensor, # [B, K]
        conceptEmb: torch.Tensor, # [K, D]
        ctx: Optional[torch.Tensor] = None, # [B, D]
        *,
        returnSupport: bool = False,):

        B, K = symLogits.shape

        if ctx is None:
            ctx = symLogits.new_zeros(B, self.dimSem)

        ctx_feat = self.ctxBackbone(ctx) # [B, D]
        alpha_delta = self.alphaHead(ctx_feat) # [B, 3]
        alpha_base = torch.stack([self.alphaImp, self.alphaCooc, self.alphaContr], dim=0).unsqueeze(0) # [1, 3]
        alpha_eff = torch.tanh(alpha_base + alpha_delta) # [B, 3]

        gamma_imp, beta_imp = self.BuildFilmParams(ctx_feat, self.filmImp)
        gamma_cooc, beta_cooc = self.BuildFilmParams(ctx_feat, self.filmCooc)
        gamma_contr, beta_contr = self.BuildFilmParams(ctx_feat, self.filmContr)

        sym_probs0 = torch.sigmoid(symLogits) # [B, K]

        w_imp = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relImp, gamma_imp, beta_imp)) # [B, K, K]
        w_contr = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relContr, gamma_contr, beta_contr)) # [B, K, K]
        w_cooc = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relCooc, gamma_cooc, beta_cooc)) # [B, K, K]

        diag_mask = torch.eye(K, device=self.device, dtype=torch.bool).unsqueeze(0)
        w_imp = w_imp.masked_fill(diag_mask, 0.0)
        w_contr = w_contr.masked_fill(diag_mask, 0.0)
        w_cooc = w_cooc.masked_fill(diag_mask, 0.0)

        support_imp = torch.bmm(sym_probs0.unsqueeze(1), w_imp).squeeze(1)
        support_cooc = torch.bmm(sym_probs0.unsqueeze(1), w_cooc).squeeze(1)
        support_contr = torch.bmm(sym_probs0.unsqueeze(1), w_contr).squeeze(1)

        combined_logits = (symLogits
            + alpha_eff[:, 0:1] * support_imp
            + alpha_eff[:, 1:2] * support_cooc
            - alpha_eff[:, 2:3] * support_contr)

        combined_probs = torch.sigmoid(combined_logits) # [B, K]
        mlp_input = torch.cat([sym_probs0, combined_probs, ctx_feat], dim=-1) # [B, 2K + D]
        delta_logits = self.postMlp(mlp_input) # [B, K]

        final_logits = combined_logits + delta_logits # [B, K]
        sym_probs = torch.sigmoid(final_logits) # [B, K]

        if not returnSupport:
            return sym_probs

        support = {
            "sym_probs0": sym_probs0,
            "support_imp": support_imp,
            "support_cooc": support_cooc,
            "support_contr": support_contr,
            "w_imp": w_imp,
            "w_cooc": w_cooc,
            "w_contr": w_contr,
            "alpha_eff": alpha_eff,
            "alpha_delta": alpha_delta,
            "gamma_imp": gamma_imp,
            "gamma_cooc": gamma_cooc,
            "gamma_contr": gamma_contr,
            "beta_imp": beta_imp,
            "beta_cooc": beta_cooc,
            "beta_contr": beta_contr,
            "ctx_feat": ctx_feat,}
        
        return sym_probs, support # sym_probs: [B, K]

    def GetInternalLoss(
        self,
        conceptEmb,
        symProbs,
        supportCache: Optional[Dict[str, torch.Tensor]] = None,
        lambdaSymmetry=1e-3,
        lambdaEntropy=1e-3,
        lambdaDynamic=5e-4):

        w_contr = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relContr)) # [K, K]
        w_cooc = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relCooc)) # [K, K]

        if w_contr.dim() == 2 and w_contr.size(0) == w_contr.size(1):
            w_contr = w_contr.clone()
            w_cooc = w_cooc.clone()
            w_contr.fill_diagonal_(0.0)
            w_cooc.fill_diagonal_(0.0)

        cooc_anti = 0.5 * (w_cooc - w_cooc.t())
        contr_anti = 0.5 * (w_contr - w_contr.t())

        loss_cooc_sym = cooc_anti.pow(2).mean() * lambdaSymmetry
        loss_contr_sym = contr_anti.pow(2).mean() * lambdaSymmetry
        loss_dynamic = symProbs.new_zeros(())

        if supportCache is not None:
            if ("w_cooc" in supportCache) and (supportCache["w_cooc"].dim() == 3):
                cooc_dyn = supportCache["w_cooc"]
                cooc_dyn_anti = 0.5 * (cooc_dyn - cooc_dyn.transpose(1, 2))
                loss_dynamic = loss_dynamic + cooc_dyn_anti.pow(2).mean() * lambdaSymmetry

            if ("w_contr" in supportCache) and (supportCache["w_contr"].dim() == 3):
                contr_dyn = supportCache["w_contr"]
                contr_dyn_anti = 0.5 * (contr_dyn - contr_dyn.transpose(1, 2))
                loss_dynamic = loss_dynamic + contr_dyn_anti.pow(2).mean() * lambdaSymmetry

            for k in ("gamma_imp", "gamma_cooc", "gamma_contr"):
                if k in supportCache:
                    loss_dynamic = loss_dynamic + (supportCache[k] - 1.0).pow(2).mean() * lambdaDynamic

            for k in ("beta_imp", "beta_cooc", "beta_contr"):
                if k in supportCache:
                    loss_dynamic = loss_dynamic + supportCache[k].pow(2).mean() * lambdaDynamic

            if "alpha_delta" in supportCache:
                loss_dynamic = loss_dynamic + supportCache["alpha_delta"].pow(2).mean() * lambdaDynamic

        eps = 1e-6
        p = symProbs.clamp(eps, 1.0 - eps)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        mean_entropy = entropy.mean()
        target_entropy = symProbs.new_tensor(0.5)
        loss_entropy = (mean_entropy - target_entropy).pow(2) * lambdaEntropy

        total_loss = loss_cooc_sym + loss_contr_sym + loss_entropy + loss_dynamic
        
        stats = {"reason_cooc_sym_pen": loss_cooc_sym.detach(),
                "reason_contr_sym_pen": loss_contr_sym.detach(),
                "reason_dynamic_pen": loss_dynamic.detach(),
                "reason_entropy": loss_entropy.detach(),}
        
        return total_loss, stats


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        *,
        activation: str = "gelu",):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(
            embedDim=d_model,
            numHeads=nhead,
            dropout=dropout,)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ff_drop = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        attn_in = self.norm1(src)
        attn_out, _ = self.self_attn(
            query=attn_in,
            key=attn_in,
            value=attn_in,
            keyPaddingMask=src_key_padding_mask,
            needWeights=False,)
        x = src + self.dropout1(attn_out)

        ff_in = self.norm2(x)
        ff = self.linear1(ff_in)
        ff = self.activation(ff)
        ff = self.ff_drop(ff)
        ff = self.linear2(ff)
        x = x + self.dropout2(ff)
        return x


class RoPETransformerEncoder(nn.Module):
    def __init__(self, layer: RoPETransformerEncoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(int(num_layers))])

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x


class RoPETransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        *,
        activation: str = "gelu",):
        super().__init__()
        self.self_attn = RoPEMultiheadAttention(
            embedDim=d_model,
            numHeads=nhead,
            dropout=dropout,)
        self.cross_attn = RoPEMultiheadAttention(
            embedDim=d_model,
            numHeads=nhead,
            dropout=dropout,)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ff_drop = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        self_in = self.norm1(tgt)
        self_out, _ = self.self_attn(
            query=self_in,
            key=self_in,
            value=self_in,
            keyPaddingMask=tgt_key_padding_mask,
            needWeights=False,
            attnMask=tgt_mask,)
        x = tgt + self.dropout1(self_out)

        cross_in = self.norm2(x)
        cross_out, _ = self.cross_attn(
            query=cross_in,
            key=memory,
            value=memory,
            keyPaddingMask=memory_key_padding_mask,
            needWeights=False,)
        x = x + self.dropout2(cross_out)

        ff_in = self.norm3(x)
        ff = self.linear1(ff_in)
        ff = self.activation(ff)
        ff = self.ff_drop(ff)
        ff = self.linear2(ff)
        x = x + self.dropout3(ff)
        return x


class RoPETransformerDecoder(nn.Module):
    def __init__(self, layer: RoPETransformerDecoderLayer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(int(num_layers))])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        x = tgt
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,)
        return x


class SymControlNet(nn.Module):
    def __init__(self, nSymbols: int, dimSem: int, nTokenSources: int = 4):
        super().__init__()
        self.nSymbols = int(nSymbols)
        self.dimSem = int(dimSem)
        self.n_token_sources = int(nTokenSources)

        inK = self.nSymbols * 5  # symProbs + sym_probs0 + imp + cooc + contr

        self.k2h = nn.Sequential(
            GrowableLoRALinear(nn.Linear(inK, dimSem)),
            nn.LayerNorm(dimSem),
            nn.GELU(),)

        self.gain_head = nn.Sequential(
            GrowableLoRALinear(nn.Linear(dimSem, 3)),)

        self.tok_head = nn.Sequential(
            GrowableLoRALinear(nn.Linear(dimSem, self.n_token_sources)), )

        self.film_head = nn.Sequential(
            GrowableLoRALinear(nn.Linear(dimSem, dimSem * 2)),)

        self.ctx_proj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(dimSem, dimSem)),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, dimSem),)

    def forward(
        self,
        symProbs: torch.Tensor,  # [B, K]
        support: Dict[str, torch.Tensor],
        conceptEmb: torch.Tensor, # [K, D]
        token_mask: torch.Tensor,) -> Dict[str, torch.Tensor]:

        featK = torch.cat([
            symProbs,
            support["sym_probs0"],
            support["support_imp"],
            support["support_cooc"],
            support["support_contr"],], dim=-1)

        h = self.k2h(featK) # [B, 5K]

        gains = torch.sigmoid(self.gain_head(h)) # [B, 3]
        g_ocr = gains[:, 0:1]
        g_ext = gains[:, 1:2]
        g_trans = gains[:, 2:3]

        tok_logits = self.tok_head(h) # [B, S]
        tok_w = F.softmax(tok_logits, dim=-1)  

        mask_f = token_mask.float()
        tok_w = tok_w * mask_f
        denom = tok_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        tok_w = tok_w / denom

        film = self.film_head(h) # [B, 2D]
        gamma_raw, beta = film.chunk(2, dim=-1)
        film_scale = torch.exp(gamma_raw.clamp(-4.0, 4.0))

        sym_ctx_raw = symProbs @ conceptEmb 
        sym_ctx = self.ctx_proj(sym_ctx_raw) 

        return {
            "h": h,
            "g_ocr": g_ocr,
            "g_ext": g_ext,
            "g_trans": g_trans,
            "tok_w": tok_w,  
            "film_scale": film_scale, 
            "beta": beta, 
            "sym_ctx": sym_ctx, }


class IntentionExtractor(AGICoreModule):
    def __init__(
        self,
        *,
        vocabSize: int = 6624,
        paddingIdx: int = 0,
        maxSeqLen: int = 64,
        recallSafetyMaxLen: Optional[int] = None,
        dimEmbed: int = 512,
        dimEncoderHidden: int = 512,
        numEncoderLayers: int = 3,
        encoderDropout: float = 0.1,
        dimSem: int = 512,
        consDim: int = 1024,
        consSelfDim: Optional[int] = None,
        consIntentDim: Optional[int] = None,
        nSymbols: int = 128,
        reasonSteps: int = 3,
        reasonerHiddenDim: int = 512,
        reasonerAlphaImp: float = 1.0,
        reasonerAlphaCooc: float = 0.5,
        reasonerAlphaContr: float = 1.0,
        reasonerDynamicScale: float = 0.25,
        lossLambdaSymmetry: float = 1e-3,
        lossLambdaEntropy: float = 1e-3,
        lossLambdaRecallCE: float = 0.25,
        lossLambdaRecallAlign: float = 0.05,
        nTextSlots: int = 4,
        chunkOverlapRatio: float = 0.5,
        ocrDictPath: Optional[str] = None,):
        super().__init__()

        self.pad_idx = int(paddingIdx)
        self.max_seq_len = int(maxSeqLen)
        self.n_text_slots = max(1, int(nTextSlots))
        self.ocr_observed_control_weight = 0.25
        overlap = float(chunkOverlapRatio)
        overlap = min(max(overlap, 0.0), 0.95)
        self.chunk_overlap_ratio = overlap
        self.chunk_stride = max(1, int(round(self.max_seq_len * (1.0 - overlap))))

        self.ch2id: Dict[str, int] = {}
        self.id2ch: List[str] = []

        if ocrDictPath is None:
            ocrDictPath = BasicParameters.OCR_DICT_PATH
        self.LoadOcrDict(ocrDictPath)

        if self.id2ch:
            self.vocab_size = len(self.id2ch) + 2
        else:
            self.vocab_size = max(3, int(vocabSize))
        self.eos_idx = int(self.vocab_size - 1)
        safety_len = self.max_seq_len * 4 if (recallSafetyMaxLen is None) else int(recallSafetyMaxLen)
        self.recall_safety_max_len = max(self.max_seq_len, safety_len)

        self.encoder = TextEncoder(
            vocabSize=self.vocab_size,
            dimEmbed=dimEmbed,
            dimHidden=dimEncoderHidden,
            numLayers=numEncoderLayers,
            dropout=encoderDropout,
            paddingIdx=self.pad_idx,)

        encoder_out_dim = self.encoder.out_dim

        self.semProj = nn.Sequential(
            GrowableLoRALinear(nn.Linear(encoder_out_dim, dimSem)),
            nn.LayerNorm(dimSem),
            nn.GELU(),)
        self.chunkFuseInNorm = nn.LayerNorm(dimSem)
        self.chunkFuseFwd = nn.Linear(dimSem, dimSem * 4)
        self.chunkFuseBwd = nn.Linear(dimSem, dimSem * 4)
        self.chunkStateProj = nn.Sequential(
            nn.Linear(dimSem * 2, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),)
        self.slotFuseInNorm = nn.LayerNorm(dimSem)
        self.slotQuery = nn.Parameter(torch.randn(self.n_text_slots, dimSem) * 0.02)
        self.slotDynQuery = nn.Sequential(
            nn.Linear(dimSem, dimSem * 2),
            nn.LayerNorm(dimSem * 2),
            nn.GELU(),
            nn.Linear(dimSem * 2, self.n_text_slots * dimSem),)
        self.slotMixGate = nn.Sequential(
            nn.Linear(dimSem, self.n_text_slots),
            nn.Sigmoid(),)

        def pick_heads(embed_dim: int) -> int:
            for h in (8, 4, 2, 1):
                if (embed_dim % h) == 0:
                    return h
            return 1

        attn_heads = pick_heads(dimSem)

        self.slotCrossAttn = RoPEMultiheadAttention(
            embedDim=dimSem,
            numHeads=attn_heads,
            dropout=encoderDropout,)
        self.slotPost = nn.Sequential(
            nn.Linear(dimSem, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),)
        self.chunkFuseOut = nn.Sequential(
            nn.Linear(dimSem * 2, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),)

        self.dimSem = int(dimSem)
        if consSelfDim is None:
            consSelfDim = consDim
        if consIntentDim is None:
            consIntentDim = consDim

        self.cons_self_dim = int(consSelfDim)
        self.cons_intent_dim = int(consIntentDim)

        self.consSelfProj = GrowableLoRALinear(nn.Linear(self.cons_self_dim, dimSem))
        self.consSelfNorm = nn.LayerNorm(dimSem)
        self.consIntentProj = GrowableLoRALinear(nn.Linear(self.cons_intent_dim, dimSem))
        self.consIntentNorm = nn.LayerNorm(dimSem)

        pair_hidden = dimSem * 2
        self.consPairNet = nn.Sequential(
            GrowableLoRALinear(nn.Linear(dimSem * 4, pair_hidden)),
            nn.LayerNorm(pair_hidden),
            nn.GELU(),
            nn.Linear(pair_hidden, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),)

        cons_encoder_layer = RoPETransformerEncoderLayer(
            d_model=dimSem,
            nhead=attn_heads,
            dim_feedforward=dimSem * 4,
            dropout=encoderDropout,
            activation="gelu",)
        self.consTokenTransformer = RoPETransformerEncoder(cons_encoder_layer, num_layers=2)

        self.consTokenGate = nn.Sequential(
            GrowableLoRALinear(nn.Linear(dimSem * 3, dimSem)),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, 3),)

        self.consFuseNorm = nn.LayerNorm(dimSem)

        self.conceptEmb = nn.Parameter(torch.randn(nSymbols, dimSem) * 0.02)
        self.conceptBias = nn.Parameter(torch.zeros(nSymbols))

        self.reasoner = LangSymbolReasoner(
            nSymbols=nSymbols,
            dimSem=dimSem,
            hiddenDim=reasonerHiddenDim,
            alphaImp=reasonerAlphaImp,
            alphaCooc=reasonerAlphaCooc,
            alphaContr=reasonerAlphaContr,
            dynamicScale=reasonerDynamicScale,)

        self.lossLambdaSymmetry = float(lossLambdaSymmetry)
        self.lossLambdaEntropy = float(lossLambdaEntropy)
        self.lossLambdaRecallCE = float(lossLambdaRecallCE)
        self.lossLambdaRecallAlign = float(lossLambdaRecallAlign)

        fuse_ocr_in = dimSem * 7
        self.fuse_ocr_gate = nn.Sequential(
            nn.Linear(fuse_ocr_in, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, 1),
            nn.Sigmoid(),)
        
        ocr_fc1 = self.fuse_ocr_gate[0]
        self.fuse_ocr_gate[0] = GrowableLoRALinear(ocr_fc1)

        fuse_ext_in = dimSem * 4
        self.fuse_ext_gate = nn.Sequential(
            nn.Linear(fuse_ext_in, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, 1),
            nn.Sigmoid(),)
        
        ext_fc1 = self.fuse_ext_gate[0]
        self.fuse_ext_gate[0] = GrowableLoRALinear(ext_fc1)

        self.beta_ocr = nn.Parameter(torch.tensor(0.1))
        self.beta_ext = nn.Parameter(torch.tensor(0.1))

        encoder_layer = RoPETransformerEncoderLayer(
            d_model=dimSem,
            nhead=attn_heads,
            dim_feedforward=dimSem * 4,
            dropout=encoderDropout,
            activation="gelu",)
        
        self.intentTransformer = RoPETransformerEncoder(encoder_layer, num_layers=2)
        self.beta_trans = nn.Parameter(torch.tensor(0.3))

        self.reason_steps = int(reasonSteps)

        self.n_token_sources = 2 + 2 * self.n_text_slots
        self.symCtrl = SymControlNet(nSymbols=nSymbols, dimSem=dimSem, nTokenSources=self.n_token_sources)

        self.beta_sym = nn.Parameter(torch.tensor(0.2))
        self.beta_update = nn.Parameter(torch.tensor(0.5))

        self.sym_norm = nn.LayerNorm(dimSem)

        self.recallStart = nn.Parameter(torch.zeros(dimSem))
        self.recallTokEmb = nn.Embedding(self.vocab_size, dimSem, padding_idx=self.pad_idx)
        self.recallCond = nn.Sequential(
            nn.Linear(dimSem, dimSem * 2),
            nn.LayerNorm(dimSem * 2),
            nn.GELU(),
            nn.Linear(dimSem * 2, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),)
        self.recallMemScore = nn.Sequential(
            nn.Linear(dimSem, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, 1),)
        self.recallInNorm = nn.LayerNorm(dimSem)
        self.recallInDrop = nn.Dropout(encoderDropout)

        recall_dec_layer = RoPETransformerDecoderLayer(
            d_model=dimSem,
            nhead=attn_heads,
            dim_feedforward=dimSem * 4,
            dropout=encoderDropout,
            activation="gelu",)
        self.recallDecoder = RoPETransformerDecoder(recall_dec_layer, num_layers=2)
        self.recallHead = nn.Linear(dimSem, self.vocab_size)

        self._last_reason_support: Optional[Dict[str, torch.Tensor]] = None
        self._last_recall_logits: Optional[torch.Tensor] = None
        self._last_recall_hidden: Optional[torch.Tensor] = None
        self._last_recall_targets: Optional[torch.Tensor] = None
        self._last_recall_valid: Optional[torch.Tensor] = None
        self._last_recall_cons_sem: Optional[torch.Tensor] = None


    def LoadOcrDict(self, dictPath: str) -> None:
        ch2id: Dict[str, int] = {}
        id2ch: List[str] = []

        with open(dictPath, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if not token:
                    continue
                ch = token
                if ch not in ch2id:
                    ch_id = len(id2ch) + 1
                    ch2id[ch] = ch_id
                    id2ch.append(ch)

        if len(ch2id) == 0:
            raise RuntimeError(f"OCR dict at {dictPath} is empty or invalid")

        self.ch2id = ch2id
        self.id2ch = id2ch

    def TextToTokenIds(self, text: Optional[str]) -> List[int]:
        if text is None:
            return []

        s = str(text).strip()
        if not s:
            return []

        ids: List[int] = []
        if self.ch2id:
            for ch in s:
                ch_id = self.ch2id.get(ch, None)
                if ch_id is not None:
                    ids.append(int(ch_id))
            return ids

        pieces = s.lower().split()
        span = max(1, self.eos_idx - 1) # map words into [1, eos_idx-1], excluding PAD/EOS
        for tok in pieces:
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "little", signed=False)
            idx = 1 + (h % span)
            ids.append(int(idx))
        return ids

    def BuildChunkStartIndices(self, length: int, stride: int) -> List[int]:
        T = self.max_seq_len
        if length <= 0:
            return [0]
        if length <= T:
            return [0]

        stride_eff = min(T, max(1, int(stride)))
        last_start = max(0, length - T)
        starts = list(range(0, last_start + 1, stride_eff))
        if len(starts) == 0:
            starts = [0]
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def TokenizeBatch(
        self,
        texts: List[Optional[str]],
        device: torch.device,
        *,
        stride: Optional[int] = None,
        appendEos: bool = False,) -> torch.Tensor:
        batch_size = len(texts)
        if batch_size == 0:
            return torch.full((0, 1, self.max_seq_len), self.pad_idx, dtype=torch.long, device=device)

        stride_eff = self.chunk_stride if (stride is None) else max(1, int(stride))
        all_ids: List[List[int]] = []
        for s in texts:
            ids = self.TextToTokenIds(s)
            if appendEos:
                ids = list(ids)
                if (len(ids) == 0) or (int(ids[-1]) != int(self.eos_idx)):
                    ids.append(int(self.eos_idx))
            all_ids.append(ids)
        all_starts: List[List[int]] = []
        chunk_counts: List[int] = []
        for ids in all_ids:
            starts = self.BuildChunkStartIndices(len(ids), stride_eff)
            all_starts.append(starts)
            chunk_counts.append(len(starts))

        n_chunks = max(1, max(chunk_counts))
        tokens = torch.full(
            (batch_size, n_chunks, self.max_seq_len),
            self.pad_idx,
            dtype=torch.long,
            device=device,)

        for i, ids in enumerate(all_ids):
            if len(ids) == 0:
                continue
            starts = all_starts[i]
            for c, start in enumerate(starts):
                end = min(start + self.max_seq_len, len(ids))
                seg = ids[start:end]
                if len(seg) == 0:
                    continue
                seg_t = torch.as_tensor(seg, dtype=torch.long, device=device)
                tokens[i, c, : seg_t.numel()] = seg_t

        return tokens # [B, N, T]

    def SelectiveScan(
        self,
        x: torch.Tensor, # [B, N, D]
        valid: torch.Tensor, # [B, N]
        proj: nn.Linear,
        *,
        reverse: bool = False,) -> Tuple[torch.Tensor, torch.Tensor]:

        B, N, D = x.shape  
        h = x.new_zeros(B, D) # [B, D]
        states = x.new_zeros(B, N, D) # [B, N, D]
        idx_iter = range(N - 1, -1, -1) if reverse else range(N) 

        for idx in idx_iter:
            xi = x[:, idx, :] # [B, D]
            gate_in_raw, gate_keep_raw, cand_raw, skip_raw = proj(xi).chunk(4, dim=-1) # [B, D]

            gate_in = torch.sigmoid(gate_in_raw) # [B, D]
            gate_keep = torch.sigmoid(gate_keep_raw) # [B, D]
            cand = torch.tanh(cand_raw) # [B, D]
            skip = torch.tanh(skip_raw) # [B, D]

            h_new = gate_keep * h + gate_in * cand + 0.1 * skip # [B, D]
            m = valid[:, idx].unsqueeze(-1).float() # [B, 1]
            h = m * h_new + (1.0 - m) * h # [B, D]
            states[:, idx, :] = h # [B, N, D]

        return states, h  

    def BuildSemanticSlots(
        self,
        contextualChunks: torch.Tensor, # [B, N, D]
        chunkValid: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, D = contextualChunks.shape  
        K = self.n_text_slots 

        has_chunk = chunkValid.any(dim=1) # [B]
        valid_f = chunkValid.float().unsqueeze(-1) # [B, N, 1]
        denom = valid_f.sum(dim=1).clamp(min=1.0) # [B, 1]
        summary = (contextualChunks * valid_f).sum(dim=1) / denom # [B, D]

        query_base = self.slotQuery.unsqueeze(0).expand(B, -1, -1) # [B, K, D]
        query_dyn = self.slotDynQuery(summary).view(B, K, D) # [B, K, D]
        mix_gate = self.slotMixGate(summary).unsqueeze(-1) # [B, K, 1]
        query = self.slotFuseInNorm(mix_gate * query_base + (1.0 - mix_gate) * query_dyn) # [B, K, D]

        safe_mask = chunkValid.clone() # [B, N]
        all_pad = ~safe_mask.any(dim=1) # [B]
        safe_mask[:, 0] |= all_pad

        slots_attn, _ = self.slotCrossAttn(
            query=query,  
            key=contextualChunks,  
            value=contextualChunks,  
            keyPaddingMask=~safe_mask,  
            needWeights=False,)
        slots = self.slotPost(slots_attn + query) # [B, K, D]

        slot_mask = has_chunk.unsqueeze(1).expand(B, K) # [B, K]
        slots = slots * slot_mask.unsqueeze(-1).float() # [B, K, D]
        return slots, slot_mask 

    def PoolChunkSemanticsWithSlots(
        self,
        chunkSem: torch.Tensor,
        chunkValid: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = self.chunkFuseInNorm(chunkSem) # [B, N, D]

        states_fwd, _ = self.SelectiveScan(x, chunkValid, self.chunkFuseFwd, reverse=False) # [B, N, D]
        states_bwd, _ = self.SelectiveScan(x, chunkValid, self.chunkFuseBwd, reverse=True) # [B, N, D]
        contextual = self.chunkStateProj(torch.cat([states_fwd, states_bwd], dim=-1)) # [B, N, D]
        contextual = contextual * chunkValid.unsqueeze(-1).float() # [B, N, D]

        slots, slot_mask = self.BuildSemanticSlots(contextual, chunkValid) # slots: [B, K, D], slot_mask: [B, K]
        slots_norm = self.slotFuseInNorm(slots) # [B, K, D]
        _, slot_fwd = self.SelectiveScan(slots_norm, slot_mask, self.chunkFuseFwd, reverse=False) # [B, D]
        _, slot_bwd = self.SelectiveScan(slots_norm, slot_mask, self.chunkFuseBwd, reverse=True) # [B, D]

        pooled = self.chunkFuseOut(torch.cat([slot_fwd, slot_bwd], dim=-1)) # [B, D]
        pooled = pooled * chunkValid.any(dim=1).unsqueeze(-1).float() # [B, D]
        return pooled, slots, slot_mask # pooled: [B, D] 


    def EncodeStringsWithSlots(
        self,
        texts: List[Optional[str]],
        device: torch.device,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids = self.TokenizeBatch(texts, device=device) # [B, N, T]
        B, N, T = token_ids.shape # scalars
        token_flat = token_ids.reshape(B * N, T) # [B*N, T]
        mask_valid_flat = token_flat.ne(self.pad_idx).any(dim=1) # [B*N]

        text_repr_flat = self.encoder(token_flat) # [B*N, E]
        lang_sem_flat = self.semProj(text_repr_flat) # [B*N, D]
        lang_sem_flat = lang_sem_flat * mask_valid_flat.unsqueeze(-1) # [B*N, D]

        chunk_sem = lang_sem_flat.view(B, N, self.dimSem) # [B, N, D]
        chunk_valid = mask_valid_flat.view(B, N) # [B, N]
        lang_sem, slot_sem, slot_mask = self.PoolChunkSemanticsWithSlots(chunk_sem, chunk_valid) # [B, D], [B, K, D], [B, K]
        return lang_sem, slot_sem, slot_mask 


    @staticmethod
    def MergeOcrTexts(ocrTexts: List[List[str]]) -> List[str]:
        merged: List[str] = []
        for lines in ocrTexts:
            if lines is None or len(lines) == 0:
                merged.append("")
            else:
                parts = [str(t).strip() for t in lines if t is not None and str(t).strip()]
                merged.append(" ".join(parts))
        return merged

    def BuildRecallTexts(
        self,
        batchSize: int,
        ocrTexts: Optional[List[List[str]]],
        extTexts: Optional[List[Optional[str]]],) -> List[str]:
        ocr_merged = [""] * batchSize if ocrTexts is None else self.MergeOcrTexts(ocrTexts)
        ext_norm: List[str] = [""] * batchSize
        if extTexts is not None:
            ext_norm = [("" if t is None else str(t).strip()) for t in extTexts]

        merged: List[str] = []
        for ocr_s, ext_s in zip(ocr_merged, ext_norm):
            o = ocr_s.strip()
            e = ext_s.strip()
            if o and e:
                merged.append(f"{o} {e}")
            elif o:
                merged.append(o)
            elif e:
                merged.append(e)
            else:
                merged.append("")
        return merged

    def BuildRecallDecoderInputs(
        self,
        recallTargets: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
        recallPrefixTokens: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if recallTargets is None:
            T = self.max_seq_len
            start = self.recallStart.view(1, 1, -1).expand(batchSize, T, -1)
            dec_in = self.recallInNorm(start)
            dec_in = self.recallInDrop(dec_in) # [B, T, D]
            return dec_in, None, None

        B, T_target = recallTargets.shape
        if recallPrefixTokens is not None:
            prefix_ids = recallPrefixTokens
        else:
            prefix_ids = None

        shifted = torch.full_like(recallTargets, self.pad_idx)
        if T_target > 1:
            shifted[:, 1:] = recallTargets[:, :-1]
        if prefix_ids is not None:
            shifted[:, 0] = prefix_ids

        tok_emb = self.recallTokEmb(shifted) # [B, T_target, D]
        start = self.recallStart.view(1, 1, -1).expand(B, 1, -1)
        first = start
        if prefix_ids is not None:
            use_prefix = prefix_ids.ne(self.pad_idx).view(B, 1, 1)
            first = torch.where(use_prefix, tok_emb[:, :1, :], start)
        if T_target > 1:
            dec_tokens = torch.cat([first, tok_emb[:, 1:, :]], dim=1)
        else:
            dec_tokens = first

        dec_in = self.recallInNorm(dec_tokens)
        dec_in = self.recallInDrop(dec_in)

        tgt_pad_mask = shifted.eq(self.pad_idx)
        tgt_pad_mask[:, 0] = False
        causal_mask = torch.triu(
            torch.ones((T_target, T_target), device=device, dtype=torch.bool),
            diagonal=1,)

        return dec_in, tgt_pad_mask, causal_mask

    def DecodeRecallFromConscious(
        self,
        recallSem: Optional[torch.Tensor], # [B, D] or [B, M, D]
        recallSemValid: Optional[torch.Tensor] = None,
        recallTargets: Optional[torch.Tensor] = None, # [B, T]
        recallPrefixTokens: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dtype = self.dtype

        if recallSem is not None:
            B = recallSem.size(0)
        elif recallTargets is not None:
            B = recallTargets.size(0)
        else:
            raise ValueError("DecodeRecallFromConscious: recallSem or recallTargets must be provided.")

        mem_tokens, mem_valid, mem_summary = self.NormalizeRecallMemory(
            recallSem=recallSem,
            recallSemValid=recallSemValid,
            batchSize=B,
            device=self.device,
            dtype=dtype,)
        has_sem = mem_valid.any(dim=1) # [B]
        cond_bias = self.recallCond(mem_summary).unsqueeze(1) # [B, 1, D]

        dec_in, tgt_pad_mask, causal_mask = self.BuildRecallDecoderInputs(
            recallTargets=recallTargets,
            batchSize=B,
            device=self.device,
            recallPrefixTokens=recallPrefixTokens,)
        
        query = dec_in + cond_bias # [B, T, D]
        mem_key_padding_mask = ~mem_valid # [B, M]
        all_mem_pad = mem_key_padding_mask.all(dim=1)
        if all_mem_pad.any():
            mem_key_padding_mask = mem_key_padding_mask.clone()
            mem_key_padding_mask[all_mem_pad, 0] = False

        hidden = self.recallDecoder(
            tgt=query,
            memory=mem_tokens,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_key_padding_mask,)
        
        logits = self.recallHead(hidden) # [B, T, V]
        
        return logits, hidden, has_sem

    def DecodeRecallChunked(
        self,
        recallSem: Optional[torch.Tensor],
        recallSemValid: Optional[torch.Tensor],
        recallTargets: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, N, T = recallTargets.shape
        D = self.dimSem

        mem_tokens: Optional[torch.Tensor] = None
        mem_valid: Optional[torch.Tensor] = None
        recall_flat: Optional[torch.Tensor] = None
        valid_flat: Optional[torch.Tensor] = None
        if recallSem is not None:
            mem_tokens, mem_valid, _ = self.NormalizeRecallMemory(
                recallSem=recallSem,
                recallSemValid=recallSemValid,
                batchSize=B,
                device=recallTargets.device,
                dtype=self.dtype,)
            M = int(mem_tokens.size(1))
            recall_flat = mem_tokens.unsqueeze(1).expand(B, N, M, self.dimSem).reshape(B * N, M, self.dimSem)
            valid_flat = mem_valid.unsqueeze(1).expand(B, N, M).reshape(B * N, M)
        elif recallSemValid is not None:
            raise ValueError("DecodeRecallChunked: recallSemValid is set but recallSem is None.")

        chunk_prefix = self.BuildRecallChunkPrefixTokens(recallTargets) # [B, N]
        prefix_flat = chunk_prefix.reshape(B * N) # [B*N]
        targets_flat = recallTargets.reshape(B * N, T)

        logits_flat, hidden_flat, has_sem_flat = self.DecodeRecallFromConscious(
            recallSem=recall_flat,
            recallSemValid=valid_flat,
            recallTargets=targets_flat,
            recallPrefixTokens=prefix_flat,)

        logits = logits_flat.view(B, N, T, -1)
        hidden = hidden_flat.view(B, N, T, D)
        has_sem = has_sem_flat.view(B, N)
        return logits, hidden, has_sem

    def PoolRecallMemory(
        self,
        recallMem: torch.Tensor, # [B, M, D]
        recallMemValid: torch.Tensor,) -> torch.Tensor:

        scores = self.recallMemScore(recallMem).squeeze(-1) # [B, M]
        safe_valid = recallMemValid.clone()
        all_pad = ~safe_valid.any(dim=1)
        safe_valid[:, 0] |= all_pad

        scores = scores.masked_fill(~safe_valid, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = attn * recallMemValid.float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        summary = (recallMem * attn.unsqueeze(-1)).sum(dim=1) # [B, D]
        summary = summary * recallMemValid.any(dim=1).unsqueeze(-1).float()
        return summary

    def NormalizeRecallMemory(
        self,
        recallSem: Optional[torch.Tensor],
        recallSemValid: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
        dtype: torch.dtype,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if recallSem is None:
            mem = torch.zeros(batchSize, 1, self.dimSem, device=device, dtype=dtype)
            valid = torch.zeros(batchSize, 1, dtype=torch.bool, device=device)
            summary = torch.zeros(batchSize, self.dimSem, device=device, dtype=dtype)
            return mem, valid, summary

        if recallSem.dim() == 2:
            mem = recallSem.unsqueeze(1) # [B, 1, D]
        elif recallSem.dim() == 3:
            mem = recallSem # [B, M, D]
        else:
            raise ValueError("NormalizeRecallMemory: recallSem must be [B, D] or [B, M, D].")

        B, M, _ = mem.shape
        if recallSemValid is None:
            valid = torch.ones(B, M, dtype=torch.bool, device=device)
        else:
            if recallSemValid.dim() == 1:
                v = recallSemValid.to(dtype=torch.bool).unsqueeze(1) # [B, 1]
                valid = v.expand(B, M) 
            elif recallSemValid.dim() == 2:
                valid = recallSemValid.to(dtype=torch.bool)
            else:
                raise ValueError("NormalizeRecallMemory: recallSemValid must be [B] or [B, M].")

        mem = mem * valid.unsqueeze(-1).float()
        summary = self.PoolRecallMemory(mem, valid) # [B, D]
        return mem, valid, summary

    def BuildRecallChunkPrefixTokens(self, recallTargets: torch.Tensor) -> torch.Tensor:
        B, N, _ = recallTargets.shape
        prefix = torch.full((B, N), self.pad_idx, dtype=torch.long, device=recallTargets.device)
        if N <= 1:
            return prefix

        valid_tok = recallTargets.ne(self.pad_idx) # [B, N, T]
        chunk_len = valid_tok.long().sum(dim=-1) # [B, N]
        has_chunk = chunk_len.gt(0) # [B, N]
        last_pos = (chunk_len - 1).clamp(min=0) # [B, N]
        last_tok = recallTargets.gather(dim=-1, index=last_pos.unsqueeze(-1)).squeeze(-1) # [B, N]

        prefix[:, 1:] = last_tok[:, :-1]
        prev_has = torch.zeros((B, N), dtype=torch.bool, device=recallTargets.device)
        prev_has[:, 1:] = has_chunk[:, :-1]
        link_mask = has_chunk & prev_has # [B, N]
        prefix = torch.where(link_mask, prefix, torch.full_like(prefix, self.pad_idx))
        return prefix

    def BuildRecallTargetsFromGenerated(
        self,
        tokenIds: torch.Tensor,
        *,
        stride: Optional[int] = None,) -> torch.Tensor:

        B, L = tokenIds.shape
        T = self.max_seq_len
        if B == 0:
            return torch.full((0, 1, T), self.pad_idx, dtype=torch.long, device=tokenIds.device)

        stride_eff = T if (stride is None) else max(1, int(stride))
        starts = self.BuildChunkStartIndices(int(L), stride_eff)
        N = max(1, len(starts))

        out = torch.full((B, N, T), self.pad_idx, dtype=torch.long, device=tokenIds.device)
        for n, st in enumerate(starts):
            ed = min(st + T, int(L))
            if ed <= st:
                continue
            out[:, n, : (ed - st)] = tokenIds[:, st:ed]
        return out # [B, N, T]

    def CacheRecallState(
        self,
        recallLogits: torch.Tensor,
        recallHidden: torch.Tensor,
        recallTargets: torch.Tensor,
        recallValid: torch.Tensor,
        consSem: Optional[torch.Tensor],) -> None:
        self._last_recall_logits = recallLogits
        self._last_recall_hidden = recallHidden
        self._last_recall_targets = recallTargets
        self._last_recall_valid = recallValid
        self._last_recall_cons_sem = None if consSem is None else consSem

    def ResetTransientLossCache(self) -> None:
        self._last_reason_support = None
        self.ResetRecallLossCache()

    def ResetRecallLossCache(self) -> None:
        self._last_recall_logits = None
        self._last_recall_hidden = None
        self._last_recall_targets = None
        self._last_recall_valid = None
        self._last_recall_cons_sem = None

    @staticmethod
    def BuildRecallSemanticFromTexts(
        semOcr: torch.Tensor,
        hasOcrMask: torch.Tensor,
        semExt: torch.Tensor,
        hasExtMask: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
        recall_sem = torch.stack([semOcr, semExt], dim=1)
        recall_valid = torch.stack([hasOcrMask, hasExtMask], dim=1)
        return (
            recall_sem * recall_valid.unsqueeze(-1),
            recall_valid)

    def RunRecallFromSemantic(
        self,
        recallSem: Optional[torch.Tensor], # [B, D] or [B, M, D]
        recallSemValid: Optional[torch.Tensor],
        batchSize: int,
        ocrTexts: Optional[List[List[str]]],
        extTexts: Optional[List[Optional[str]]],
        device: torch.device,) -> Dict[str, torch.Tensor]:

        self._last_recall_logits = None
        self._last_recall_hidden = None
        self._last_recall_targets = None
        self._last_recall_valid = None
        self._last_recall_cons_sem = None

        recall_texts = self.BuildRecallTexts(
            batchSize,
            ocrTexts=ocrTexts,
            extTexts=extTexts)

        recall_targets = self.TokenizeBatch(
            recall_texts,
            device=device,
            stride=self.max_seq_len,
            appendEos=True,) # [B, N, T]
        
        recall_target_valid = recall_targets.ne(self.pad_idx).any(dim=-1) # [B, N]

        recall_logits, recall_hidden, recall_has_sem = self.DecodeRecallChunked(
            recallSem=recallSem,
            recallSemValid=recallSemValid,
            recallTargets=recall_targets,)
        
        recall_valid = recall_target_valid & recall_has_sem # [B, N]
        recall_pred_ids = recall_logits.argmax(dim=-1) # [B, N, T]

        recall_cond = None
        if recallSem is not None:
            _, _, recall_summary = self.NormalizeRecallMemory(
                recallSem=recallSem,
                recallSemValid=recallSemValid,
                batchSize=batchSize,
                device=device,
                dtype=self.dtype,)
            recall_cond = recall_summary.unsqueeze(1).expand(batchSize, recall_targets.size(1), self.dimSem)

        self.CacheRecallState(
            recallLogits=recall_logits,
            recallHidden=recall_hidden,
            recallTargets=recall_targets,
            recallValid=recall_valid,
            consSem=None if recall_cond is None else recall_cond.detach(),)

        return {
            "recall_logits": recall_logits.detach(),
            "recall_targets": recall_targets.detach(),
            "recall_valid": recall_valid.detach(),
            "recall_pred_ids": recall_pred_ids.detach(),}

    @torch.no_grad()
    def TokenIdsToTexts(
        self, 
        tokenIds: torch.Tensor # [B, L]
        ) -> List[str]:

        use_dict = len(self.id2ch) > 0

        def decode_one_row(row: List[int]) -> Tuple[str, bool]:
            pieces: List[str] = []
            hit_eos = False
            for tid_raw in row:
                tid = int(tid_raw)
                if tid == self.eos_idx:
                    hit_eos = True
                    break
                if tid == self.pad_idx:
                    break
                if use_dict:
                    if 1 <= tid <= len(self.id2ch):
                        pieces.append(self.id2ch[tid - 1])
                    else:
                        pieces.append("[UNK]")
                else:
                    pieces.append(str(tid))
            text = "".join(pieces) if use_dict else " ".join(pieces)
            return text, hit_eos

        rows = tokenIds.detach().to("cpu").tolist()
        texts: List[str] = []
        for row in rows:
            txt, _ = decode_one_row(row)
            texts.append(txt)
        return texts

    @torch.no_grad()
    def RecallGenerateFromSemantic(
        self,
        recallSem: torch.Tensor, # [B, D]s
        *,
        maxLen: Optional[int] = None,) -> Tuple[torch.Tensor, List[str]]:

        B = int(recallSem.size(0))
        device = self.device
        total_len = self.recall_safety_max_len if (maxLen is None) else max(1, int(maxLen))

        pred_ids = torch.full((B, total_len), self.pad_idx, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        offset = 0
        while offset < total_len:
            if bool(finished.all()):
                break
            chunk_len = min(self.max_seq_len, total_len - offset)
            chunk_ids = torch.full((B, chunk_len), self.pad_idx, dtype=torch.long, device=device) # [B, chunk_len]

            if offset > 0:
                bridge = min(offset, self.max_seq_len - 1)
                prefix = pred_ids[:, offset - bridge:offset] # [B, bridge]
            else:
                prefix = pred_ids.new_empty((B, 0))

            for t in range(chunk_len):
                ctx = torch.cat([prefix, chunk_ids[:, :t]], dim=1) # [B, bridge + t]
                if int(ctx.size(1)) > (self.max_seq_len - 1):
                    ctx = ctx[:, -(self.max_seq_len - 1):]
                local_targets = torch.cat([
                    ctx,
                    torch.full((B, 1), self.pad_idx, dtype=torch.long, device=device),], dim=1)

                logits_t, _, _ = self.DecodeRecallFromConscious(
                    recallSem=recallSem,
                    recallTargets=local_targets,) #  logits_t: [B, T_local, V]
                
                next_id = logits_t[:, -1, :].argmax(dim=-1)
                next_id = torch.where(finished, torch.full_like(next_id, self.pad_idx), next_id)
                chunk_ids[:, t] = next_id
                finished = finished | next_id.eq(self.eos_idx)

            pred_ids[:, offset: offset + chunk_len] = chunk_ids
            offset += chunk_len

        has_any = pred_ids.ne(self.pad_idx).any(dim=0)
        if bool(has_any.any()):
            last_idx = int(torch.nonzero(has_any, as_tuple=False)[-1, 0]) + 1
            pred_ids = pred_ids[:, :last_idx]
        else:
            pred_ids = pred_ids[:, :1]

        texts = self.TokenIdsToTexts(pred_ids) # [B, L]
        return pred_ids, texts # [B, L(Token)], List[str]

    @torch.no_grad()
    def RecallGenerateFromConscious(
        self,
        selfState: Optional[torch.Tensor],
        intentState: Optional[torch.Tensor],
        *,
        maxLen: Optional[int] = None,) -> Tuple[torch.Tensor, List[str]]:
        intentSem, _, _ = self(
            selfState=selfState,
            intentState=intentState,
            ocrTexts=None,
            extTexts=None,
            prioritizeExt=False,)
        if intentSem.numel() == 0:
            empty_ids = torch.full((0, 1), self.pad_idx, dtype=torch.long, device=self.device)
            return empty_ids, []
        return self.RecallGenerateFromSemantic(intentSem, maxLen=maxLen)

    def InferBatchSize(
        self,
        selfState: Optional[torch.Tensor],
        intentState: Optional[torch.Tensor],
        ocrTexts: Optional[List[List[str]]],
        extTexts: Optional[List[Optional[str]]],) -> Optional[int]:
        batch_size: Optional[int] = None
        if selfState is not None:
            batch_size = int(selfState.size(0))
        if intentState is not None:
            if batch_size is None:
                batch_size = int(intentState.size(0))
            elif int(intentState.size(0)) != batch_size:
                raise ValueError(
                    f"IntentionExtractor: batch mismatch, self_sem={batch_size}, intent_sem={int(intentState.size(0))}")

        if ocrTexts is not None:
            if batch_size is None:
                batch_size = len(ocrTexts)
            elif len(ocrTexts) != batch_size:
                raise ValueError(f"IntentionExtractor: batch mismatch, consciousness={batch_size}, ocrTexts={len(ocrTexts)}")

        if extTexts is not None:
            if batch_size is None:
                batch_size = len(extTexts)
            elif len(extTexts) != batch_size:
                raise ValueError(f"IntentionExtractor: batch mismatch, consciousness/ocr={batch_size}, extTexts={len(extTexts)}")

        return batch_size

    def EncodeConsciousStates(
        self,
        selfState: Optional[torch.Tensor],
        intentState: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        dim = self.dimSem
        self_sem = torch.zeros(batchSize, dim, device=device)
        intent_sem = torch.zeros(batchSize, dim, device=device)
        extras: Dict[str, torch.Tensor] = {}

        if selfState is not None:
            self_in = selfState
            self_sem = self.consSelfNorm(self.consSelfProj(self_in)) # [B, D]
            extras["cons_self_sem"] = self_sem.detach()

        if intentState is not None:
            intent_in = intentState
            intent_sem = self.consIntentNorm(self.consIntentProj(intent_in)) # [B, D]
            extras["cons_intent_sem"] = intent_sem.detach()

        cons_sem: Optional[torch.Tensor] = None

        if (selfState is not None) and (intentState is not None):
            pair_feat = torch.cat([
                self_sem,
                intent_sem,
                torch.abs(self_sem - intent_sem),
                self_sem * intent_sem,], dim=-1)
            pair_sem = self.consPairNet(pair_feat) # [B, D]

            cons_tokens = torch.stack([self_sem, intent_sem, pair_sem], dim=1) # [B, 3, D]
            cons_tokens = self.consTokenTransformer(cons_tokens) # [B, 3, D]

            token_logits = self.consTokenGate(cons_tokens.reshape(batchSize, -1))
            token_weights = F.softmax(token_logits, dim=-1) # [B, 3]

            cons_sem = (cons_tokens * token_weights.unsqueeze(-1)).sum(dim=1) # [B, D]
            cons_sem = self.consFuseNorm(cons_sem + pair_sem) # [B, D]

            extras["cons_pair_sem"] = pair_sem.detach()
            extras["cons_token_weights"] = token_weights.detach()
        elif intentState is not None:
            cons_sem = intent_sem
        elif selfState is not None:
            cons_sem = self_sem

        return cons_sem, self_sem, intent_sem, extras

    def BuildTextTrustMasks(
        self,
        batchSize: int,
        textTrust: Optional[List[str]],
        device: torch.device,) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        trust = (
            [TEXT_TRUST_UNSAFE_EXTERNAL for _ in range(batchSize)]
            if textTrust is None
            else [str(item) for item in textTrust])
        ext_control = torch.tensor(
            [1.0 if item == TEXT_TRUST_OPERATOR_COMMAND else 0.0 for item in trust],
            device=device,
            dtype=torch.float32)
        ocr_control = torch.full(
            (batchSize,),
            float(self.ocr_observed_control_weight),
            device=device,
            dtype=torch.float32)
        return trust, ext_control, ocr_control

    def forward(
        self,
        selfState: Optional[torch.Tensor],
        intentState: Optional[torch.Tensor],
        ocrTexts: Optional[List[List[str]]] = None,
        extTexts: Optional[List[Optional[str]]] = None,
        *,
        prioritizeExt: bool = False,
        textTrust: Optional[List[str]] = None,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:

        device = self.device
        batch_size = self.InferBatchSize(selfState, intentState, ocrTexts, extTexts)
        if batch_size is None:
            self.ResetTransientLossCache()
            intent_zero = torch.zeros(0, self.dimSem, device=device)
            sym_zero = torch.zeros(0, int(self.conceptEmb.size(0)), device=device)
            return intent_zero, sym_zero, {}

        cons_sem, self_sem, intent_sem_cons, cons_extras = self.EncodeConsciousStates(
            selfState=selfState,
            intentState=intentState,
            batchSize=batch_size,
            device=device,) # cons_sem: [B, D]
        
        extras: Dict[str, Any] = dict(cons_extras)
        if cons_sem is not None:
            extras["cons_sem"] = cons_sem.detach()

        text_trust, ext_control, ocr_control = self.BuildTextTrustMasks(batch_size, textTrust, device)
        ext_control_mask = ext_control > 0.0
        extras["text_trust"] = list(text_trust)
        extras["ext_control_mask"] = ext_control_mask.detach()
        extras["ocr_control_weight"] = ocr_control.detach()
        observed_ext_texts = None
        if extTexts is not None:
            observed_ext_texts = [
                "" if text is None else str(text)
                for text in extTexts]

        if ocrTexts is not None:
            merged = self.MergeOcrTexts(ocrTexts)
            sem_ocr, ocr_slots, ocr_slot_mask = self.EncodeStringsWithSlots(merged, device=device) # [B, D], [B, K, D], [B, K]
            has_ocr_mask = ocr_slot_mask.any(dim=1)
        else:
            sem_ocr = torch.zeros(batch_size, self.dimSem, device=device)
            ocr_slots = torch.zeros(batch_size, self.n_text_slots, self.dimSem, device=device)
            ocr_slot_mask = torch.zeros(batch_size, self.n_text_slots, dtype=torch.bool, device=device)
            has_ocr_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if observed_ext_texts is not None:
            sem_ext, ext_slots, ext_slot_mask = self.EncodeStringsWithSlots(observed_ext_texts, device=device) # [B, D], [B, K, D], [B, K]
            has_ext_mask = ext_slot_mask.any(dim=1)
        else:
            sem_ext = torch.zeros(batch_size, self.dimSem, device=device)
            ext_slots = torch.zeros(batch_size, self.n_text_slots, self.dimSem, device=device)
            ext_slot_mask = torch.zeros(batch_size, self.n_text_slots, dtype=torch.bool, device=device)
            has_ext_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        has_ext_control_mask = has_ext_mask & ext_control_mask
        if (cons_sem is None) and (not has_ocr_mask.any()) and (not has_ext_control_mask.any()):
            self._last_reason_support = None
            if self.training and bool(has_ext_mask.any().item()):
                recall_sem, recall_valid = self.BuildRecallSemanticFromTexts(
                    semOcr=sem_ocr,
                    hasOcrMask=has_ocr_mask,
                    semExt=sem_ext,
                    hasExtMask=has_ext_mask)
                extras.update(self.RunRecallFromSemantic(
                    recallSem=recall_sem,
                    recallSemValid=recall_valid,
                    batchSize=batch_size,
                    ocrTexts=ocrTexts,
                    extTexts=extTexts,
                    device=device))
            else:
                self.ResetRecallLossCache()
            extras["sem_ext_observed"] = sem_ext.detach()
            return None, None, extras

        if cons_sem is not None:
            base = cons_sem # [B, D]
        else:
            base = torch.zeros(batch_size, self.dimSem, device=device)

        ext_control_float = ext_control.unsqueeze(-1)
        ext_slot_control = ext_control.view(batch_size, 1, 1)
        sem_ext_control = sem_ext * ext_control_float
        ext_slots_control = ext_slots * ext_slot_control
        ext_slot_mask_control = ext_slot_mask & ext_control_mask.view(batch_size, 1)
        ocr_control_float = ocr_control.unsqueeze(-1)

        ext_for_ocr = sem_ext_control

        feat_ocr = torch.cat([
                base,
                sem_ocr,
                torch.abs(base - sem_ocr),
                base * sem_ocr,
                ext_for_ocr,
                torch.abs(ext_for_ocr - sem_ocr),
                ext_for_ocr * sem_ocr,],dim=-1,) 

        gate_ocr = self.fuse_ocr_gate(feat_ocr) # [B, 1]

        sem_ocr_fused = gate_ocr * sem_ocr # [B, D]

        ocr_mask_float = has_ocr_mask.unsqueeze(-1).float() # [B, 1]
        base = base + self.beta_ocr * (sem_ocr_fused * ocr_mask_float * ocr_control_float) # [B, D]

        extras["sem_ocr_raw"] = sem_ocr.detach()
        extras["sem_ocr_fused"] = sem_ocr_fused.detach()
        extras["gate_ocr"] = gate_ocr.detach()
        extras["has_ocr_mask"] = has_ocr_mask.detach()
        extras["sem_ocr_slots"] = ocr_slots.detach()
        extras["sem_ocr_slot_mask"] = ocr_slot_mask.detach()

        feat_ext = torch.cat([
                base,
                sem_ext_control,
                torch.abs(base - sem_ext_control),
                base * sem_ext_control,],dim=-1,) # [B, 4D]

        gate_ext = self.fuse_ext_gate(feat_ext) # [B, 1]

        has_ext_mask_float = has_ext_control_mask.unsqueeze(-1).float()
        has_ext_mask_exp = has_ext_control_mask.unsqueeze(-1)

        if prioritizeExt:
            gamma = 0.5 + 0.5 * gate_ext # [B, 1]
            candidate = (1.0 - gamma) * base + gamma * sem_ext_control # [B, D]
            intentSem = torch.where(has_ext_mask_exp, candidate, base)

            extras["gamma_ext"] = gamma.detach()
        else:
            sem_ext_fused = gate_ext * sem_ext_control
            intentSem = base + self.beta_ext * (sem_ext_fused * has_ext_mask_float)
            extras["sem_ext_fused"] = sem_ext_fused.detach()

        extras["sem_ext_observed"] = sem_ext.detach()
        extras["sem_ext_controlled"] = sem_ext_control.detach()
        extras["gate_ext"] = gate_ext.detach()
        extras["has_ext_mask"] = has_ext_mask.detach()
        extras["sem_ext_slots"] = ext_slots.detach()
        extras["sem_ext_slot_mask"] = ext_slot_mask.detach()

        self_token_mask = torch.full((batch_size, 1), selfState is not None, dtype=torch.bool, device=device)
        intent_token_mask = torch.full((batch_size, 1), intentState is not None, dtype=torch.bool, device=device)
        tokens = torch.cat([
            self_sem.unsqueeze(1),
            intent_sem_cons.unsqueeze(1),
            ocr_slots * ocr_control.view(batch_size, 1, 1),
            ext_slots_control,], dim=1)
        token_mask = torch.cat([
            self_token_mask,
            intent_token_mask,
            ocr_slot_mask,
            ext_slot_mask_control,], dim=1)

        def safe_token_mask(token_mask_: torch.Tensor) -> torch.Tensor:
            safe = token_mask_.clone()
            all_pad = ~safe.any(dim=1)
            safe[:, 0] |= all_pad
            return safe

        trans_tokens: Optional[torch.Tensor] = None
        token_mask_safe = safe_token_mask(token_mask)
        src_key_padding_mask = ~token_mask_safe  # [B, S]

        if token_mask.any():
            trans_tokens = self.intentTransformer(tokens, src_key_padding_mask=src_key_padding_mask)

            mask_float = token_mask.float().unsqueeze(-1)
            sum_vec = (trans_tokens * mask_float).sum(dim=1)
            mask_sum = mask_float.sum(dim=1)
            denom = mask_sum.clamp(min=1.0)
            fused = sum_vec / denom
            intentSem = intentSem + self.beta_trans * fused

            extras["intent_trans_norm"] = fused.norm(dim=-1, keepdim=True).detach()
            extras["intent_trans_mask_sum"] = mask_sum.detach()

        abs_ext_ocr = torch.abs(sem_ext_control - sem_ocr)
        mul_ext_ocr = sem_ext_control * sem_ocr

        self._last_reason_support = None
        for t in range(self.reason_steps):
            symbol_logits = F.linear(intentSem, self.conceptEmb, self.conceptBias)
            symProbs_t, support = self.reasoner(
                symbol_logits,
                self.conceptEmb,
                ctx=intentSem,
                returnSupport=True,)

            ctrl = self.symCtrl(symProbs_t, support, self.conceptEmb, token_mask)

            base_ctx = intentSem 

            feat_ocr = torch.cat([
                base_ctx,
                sem_ocr,
                torch.abs(base_ctx - sem_ocr),
                base_ctx * sem_ocr,
                sem_ext_control,
                abs_ext_ocr,
                mul_ext_ocr,], dim=-1)

            gate_ocr = self.fuse_ocr_gate(feat_ocr)
            sem_ocr_fused = gate_ocr * sem_ocr
            base2 = base_ctx + torch.tanh(self.beta_ocr) * ctrl["g_ocr"] * (
                sem_ocr_fused * ocr_mask_float * ocr_control_float)

            feat_ext = torch.cat([
                base2,
                sem_ext_control,
                torch.abs(base2 - sem_ext_control),
                base2 * sem_ext_control,], dim=-1)

            gate_ext = self.fuse_ext_gate(feat_ext)

            if prioritizeExt:
                gamma0 = 0.5 + 0.5 * gate_ext
                gamma_eff = gamma0 * ctrl["g_ext"]
                candidate = (1.0 - gamma_eff) * base2 + gamma_eff * sem_ext_control
                intent2 = torch.where(has_ext_mask_exp, candidate, base2)
            else:
                sem_ext_fused = gate_ext * sem_ext_control
                intent2 = base2 + torch.tanh(self.beta_ext) * ctrl["g_ext"] * (
                    sem_ext_fused * has_ext_mask_float)

            if trans_tokens is not None:
                w = ctrl["tok_w"]
                w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                fused2 = (trans_tokens * w.unsqueeze(-1)).sum(dim=1)
                intent2 = intent2 + torch.tanh(self.beta_trans) * ctrl["g_trans"] * fused2

            intent2 = self.sym_norm(intent2 + torch.tanh(self.beta_sym) * ctrl["sym_ctx"])
            intent2 = self.sym_norm(intent2 * ctrl["film_scale"] + ctrl["beta"])

            u = torch.sigmoid(self.beta_update)
            intentSem = (1.0 - u) * intentSem + u * intent2

            extras["sym_probs_loop"] = symProbs_t.detach()
            extras["sym_ctrl_gains"] = torch.cat([ctrl["g_ocr"], ctrl["g_ext"], ctrl["g_trans"]], dim=-1).detach()
            extras["sym_tok_w"] = ctrl["tok_w"].detach()
            extras["sym_reason_alpha"] = support["alpha_eff"].detach()

        final_logits = F.linear(intentSem, self.conceptEmb, self.conceptBias)

        symProbs, final_support = self.reasoner(
            final_logits,
            self.conceptEmb,
            ctx=intentSem,
            returnSupport=True,)
        
        self._last_reason_support = final_support
        extras["reason_alpha_final"] = final_support["alpha_eff"].detach()

        if self.training:
            recall_sem, recall_valid = self.BuildRecallSemanticFromTexts(
                semOcr=sem_ocr,
                hasOcrMask=has_ocr_mask,
                semExt=sem_ext,
                hasExtMask=has_ext_mask)
            recall_output = self.RunRecallFromSemantic(
                recallSem=recall_sem,
                recallSemValid=recall_valid,
                batchSize=batch_size,
                ocrTexts=ocrTexts,
                extTexts=extTexts,
                device=device,)
            extras.update(recall_output)

            recall_pred_ids_train = recall_output.get("recall_pred_ids")
            if recall_pred_ids_train is not None:
                recall_pred_flat = recall_pred_ids_train.reshape(recall_pred_ids_train.size(0), -1)
                extras["recall_texts"] = self.TokenIdsToTexts(recall_pred_flat)
            else:
                extras["recall_texts"] = []
        else:
            self.ResetRecallLossCache()
            _, recall_texts = self.RecallGenerateFromSemantic(
                intentSem,
                maxLen=self.recall_safety_max_len,)
            extras["recall_texts"] = recall_texts

        return intentSem, symProbs, extras

    def GetInternalLoss(
        self,
        symProbs: torch.Tensor,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        reason_loss, stats = self.reasoner.GetInternalLoss(
            conceptEmb=self.conceptEmb,
            symProbs=symProbs,
            supportCache=self._last_reason_support,
            lambdaSymmetry=self.lossLambdaSymmetry,
            lambdaEntropy=self.lossLambdaEntropy,)

        recall_loss, recall_stats = self.GetRecallLoss()
        total_loss = reason_loss + recall_loss

        stats.update(recall_stats)

        return total_loss, stats

    def GetRecallLoss(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = self.conceptEmb.new_zeros(())

        if (self._last_recall_logits is None
            or self._last_recall_hidden is None
            or self._last_recall_targets is None
            or self._last_recall_valid is None):
            return zero, {
                "recall_loss_ce": zero.detach(),
                "recall_loss_align": zero.detach(),
                "recall_loss_total": zero.detach(),}

        logits = self._last_recall_logits
        hidden = self._last_recall_hidden
        targets = self._last_recall_targets
        valid_seq = self._last_recall_valid

        if logits.numel() == 0 or (not valid_seq.any()):
            return zero, {
                "recall_loss_ce": zero.detach(),
                "recall_loss_align": zero.detach(),
                "recall_loss_total": zero.detach(),}

        token_valid = targets.ne(self.pad_idx) & valid_seq.unsqueeze(-1)
        loss_ce = F.cross_entropy(
            logits[token_valid],
            targets[token_valid],)

        loss_align = zero
        if self._last_recall_cons_sem is not None:
            cons_sem = self._last_recall_cons_sem
            valid_hidden = hidden[valid_seq]
            valid_token = token_valid[valid_seq]
            hidden_avg = (
                torch.where(
                    valid_token.unsqueeze(-1),
                    valid_hidden,
                    torch.zeros_like(valid_hidden),).sum(dim=-2)
                / valid_token.sum(dim=-1, keepdim=True))
            cos = F.cosine_similarity(hidden_avg, cons_sem[valid_seq], dim=-1)
            loss_align = (1.0 - cos).mean()

        total = self.lossLambdaRecallCE * loss_ce + self.lossLambdaRecallAlign * loss_align

        stats = {
            "recall_loss_ce": loss_ce.detach(),
            "recall_loss_align": loss_align.detach(),
            "recall_loss_total": total.detach(),}
        
        return total, stats



class IntentionOnlineWrapper(BaseOnlineWrapper):
    def __init__(
        self,
        base: "IntentionExtractor",
        *,
        initRankEach: int = 4,
        autoRank: bool = True,
        evThreshold: float = 0.90,
        gradEma: float = 0.9,
        maxRankSem: int = 64,
        maxRankCons: int = 64,
        maxRankOcr: int = 64,
        maxRankExt: int = 64,
        maxRankSym: int = 64,):
        self.maxRankSem = int(maxRankSem)
        self.maxRankCons = int(maxRankCons)
        self.maxRankOcr = int(maxRankOcr)
        self.maxRankExt = int(maxRankExt)
        self.maxRankSym = int(maxRankSym)
        super().__init__(base,initRankEach=initRankEach,autoRank=autoRank,evThreshold=evThreshold,gradEma=gradEma,)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        base: "IntentionExtractor" = self.base 

        sem_lora: "GrowableLoRALinear" = base.semProj[0]
        cons_self_lora: "GrowableLoRALinear" = base.consSelfProj
        cons_intent_lora: "GrowableLoRALinear" = base.consIntentProj
        cons_pair_lora: "GrowableLoRALinear" = base.consPairNet[0]
        cons_gate_lora: "GrowableLoRALinear" = base.consTokenGate[0]
        ocr_lora: "GrowableLoRALinear" = base.fuse_ocr_gate[0]
        ext_lora: "GrowableLoRALinear" = base.fuse_ext_gate[0]
        sym_k2h: "GrowableLoRALinear" = base.symCtrl.k2h[0]
        sym_gain: "GrowableLoRALinear" = base.symCtrl.gain_head[0]
        sym_tok: "GrowableLoRALinear" = base.symCtrl.tok_head[0]
        sym_film: "GrowableLoRALinear" = base.symCtrl.film_head[0]
        sym_ctx: "GrowableLoRALinear" = base.symCtrl.ctx_proj[0]

        def make_alloc(inDim: int, outDim: int, maxRank: int):
            def alloc(addRank: int, device: torch.device, dtype: torch.dtype):
                A = nn.Parameter(torch.randn(addRank, inDim, device=device, dtype=dtype) * 1e-4)
                B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype) * 1e-4)
                s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
                return A, B, s
            return alloc

        def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParametersScale(s) * (b @ a)

        specs = {
            "sem": SiteSpec(
                name="sem",
                nLayers=1,
                inDim=int(sem_lora.in_f),
                outDim=int(sem_lora.out_f),
                maxRank=self.maxRankSem,
                allocFn=make_alloc(int(sem_lora.in_f), int(sem_lora.out_f), self.maxRankSem),
                composeFn=compose,),

            "cons_self": SiteSpec(
                name="cons_self",
                nLayers=1,
                inDim=int(cons_self_lora.in_f),
                outDim=int(cons_self_lora.out_f),
                maxRank=self.maxRankCons,
                allocFn=make_alloc(int(cons_self_lora.in_f), int(cons_self_lora.out_f), self.maxRankCons),
                composeFn=compose,),

            "cons_intent": SiteSpec(
                name="cons_intent",
                nLayers=1,
                inDim=int(cons_intent_lora.in_f),
                outDim=int(cons_intent_lora.out_f),
                maxRank=self.maxRankCons,
                allocFn=make_alloc(int(cons_intent_lora.in_f), int(cons_intent_lora.out_f), self.maxRankCons),
                composeFn=compose,),

            "cons_pair": SiteSpec(
                name="cons_pair",
                nLayers=1,
                inDim=int(cons_pair_lora.in_f),
                outDim=int(cons_pair_lora.out_f),
                maxRank=self.maxRankCons,
                allocFn=make_alloc(int(cons_pair_lora.in_f), int(cons_pair_lora.out_f), self.maxRankCons),
                composeFn=compose,),

            "cons_token_gate": SiteSpec(
                name="cons_token_gate",
                nLayers=1,
                inDim=int(cons_gate_lora.in_f),
                outDim=int(cons_gate_lora.out_f),
                maxRank=self.maxRankCons,
                allocFn=make_alloc(int(cons_gate_lora.in_f), int(cons_gate_lora.out_f), self.maxRankCons),
                composeFn=compose,),

            "ocr_gate": SiteSpec(
                name="ocr_gate",
                nLayers=1,
                inDim=int(ocr_lora.in_f),
                outDim=int(ocr_lora.out_f),
                maxRank=self.maxRankOcr,
                allocFn=make_alloc(int(ocr_lora.in_f), int(ocr_lora.out_f), self.maxRankOcr),
                composeFn=compose,),

            "ext_gate": SiteSpec(
                name="ext_gate",
                nLayers=1,
                inDim=int(ext_lora.in_f),
                outDim=int(ext_lora.out_f),
                maxRank=self.maxRankExt,
                allocFn=make_alloc(int(ext_lora.in_f), int(ext_lora.out_f), self.maxRankExt),
                composeFn=compose,),

            "sym_k2h": SiteSpec(
                name="sym_k2h",
                nLayers=1,
                inDim=int(sym_k2h.in_f),
                outDim=int(sym_k2h.out_f),
                maxRank=self.maxRankSym,
                allocFn=make_alloc(int(sym_k2h.in_f), int(sym_k2h.out_f), self.maxRankSym),
                composeFn=compose,),

            "sym_gain": SiteSpec(
                name="sym_gain",
                nLayers=1,
                inDim=int(sym_gain.in_f),
                outDim=int(sym_gain.out_f),
                maxRank=self.maxRankSym,
                allocFn=make_alloc(int(sym_gain.in_f), int(sym_gain.out_f), self.maxRankSym),
                composeFn=compose,),

            "sym_tok": SiteSpec(
                name="sym_tok",
                nLayers=1,
                inDim=int(sym_tok.in_f),
                outDim=int(sym_tok.out_f),
                maxRank=self.maxRankSym,
                allocFn=make_alloc(int(sym_tok.in_f), int(sym_tok.out_f), self.maxRankSym),
                composeFn=compose,),

            "sym_film": SiteSpec(
                name="sym_film",
                nLayers=1,
                inDim=int(sym_film.in_f),
                outDim=int(sym_film.out_f),
                maxRank=self.maxRankSym,
                allocFn=make_alloc(int(sym_film.in_f), int(sym_film.out_f), self.maxRankSym),
                composeFn=compose,),

            "sym_ctx": SiteSpec(
                name="sym_ctx",
                nLayers=1,
                inDim=int(sym_ctx.in_f),
                outDim=int(sym_ctx.out_f),
                maxRank=self.maxRankSym,
                allocFn=make_alloc(int(sym_ctx.in_f), int(sym_ctx.out_f), self.maxRankSym),
                composeFn=compose,),}
        
        return specs

    def forward(
        self,
        selfState: Optional[torch.Tensor],
        intentState: Optional[torch.Tensor],
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        **kwargs,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        deltas = [self.ComposeLayerDelta(layerIdx) for layerIdx in range(self.layerCount)]
        return self.ForwardWithDeltas(
            x=None,
            keyPaddingMask=keyPaddingMask,
            tdError=tdError,
            uncertainty=uncertainty,
            deltasPerLayer=deltas,
            selfState=selfState,
            intentState=intentState,
            **kwargs,)

    @staticmethod
    def LinearWithLora(
        mod: "GrowableLoRALinear",
        x: torch.Tensor,
        deltaW: Optional[torch.Tensor],) -> torch.Tensor:
        W = mod.target.weight
        b = mod.target.bias
        base_delta = mod.DeltaWeight()
        if base_delta is not None:
            W = W + base_delta

        if deltaW is not None:
            W = W + deltaW.to(device=W.device, dtype=W.dtype)

        return F.linear(x, W, b)


    def EncodeStringsWithDelta(
        self,
        base: "IntentionExtractor",
        texts: List[str],
        device: torch.device,
        delta_sem: Optional[torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        normed: List[str] = []
        for s in texts:
            if s is None:
                normed.append("")
            else:
                normed.append(str(s))

        token_ids = base.TokenizeBatch(normed, device=device) # [B, N, T]
        B, N, T = token_ids.shape
        token_flat = token_ids.reshape(B * N, T)
        mask_valid_flat = token_flat.ne(base.pad_idx).any(dim=1) # [B*N]

        text_repr = base.encoder(token_flat) 

        sem_lora: "GrowableLoRALinear" = base.semProj[0]
        h = self.LinearWithLora(sem_lora, text_repr, delta_sem)
        h = base.semProj[1](h)
        h = base.semProj[2](h)

        h = h * mask_valid_flat.unsqueeze(-1)
        chunk_sem = h.view(B, N, base.dimSem)
        chunk_valid = mask_valid_flat.view(B, N)
        pooled, slots, slot_mask = base.PoolChunkSemanticsWithSlots(chunk_sem, chunk_valid)
        return pooled, slots, slot_mask

    def GateWithDelta(
        self,
        gate_seq: nn.Sequential,
        x: torch.Tensor,
        delta_gate: Optional[torch.Tensor],) -> torch.Tensor:
        gate_lora: "GrowableLoRALinear" = gate_seq[0]

        h = self.LinearWithLora(gate_lora, x, delta_gate)
        h = gate_seq[1](h)
        h = gate_seq[2](h)
        h = gate_seq[3](h)
        h = gate_seq[4](h)
        return h

    def ConsPairWithDelta(
        self,
        pair_seq: nn.Sequential,
        x: torch.Tensor,
        delta_pair: Optional[torch.Tensor],) -> torch.Tensor:
        pair_lora: "GrowableLoRALinear" = pair_seq[0]
        h = self.LinearWithLora(pair_lora, x, delta_pair)
        h = pair_seq[1](h)
        h = pair_seq[2](h)
        h = pair_seq[3](h)
        h = pair_seq[4](h)
        h = pair_seq[5](h)
        return h

    def ConsTokenGateWithDelta(
        self,
        gate_seq: nn.Sequential,
        x: torch.Tensor,
        delta_gate: Optional[torch.Tensor],) -> torch.Tensor:
        gate_lora: "GrowableLoRALinear" = gate_seq[0]
        h = self.LinearWithLora(gate_lora, x, delta_gate)
        h = gate_seq[1](h)
        h = gate_seq[2](h)
        h = gate_seq[3](h)
        return h

    def EncodeConsciousStatesWithDelta(
        self,
        base: "IntentionExtractor",
        selfState: Optional[torch.Tensor],
        intentState: Optional[torch.Tensor],
        batchSize: int,
        device: torch.device,
        *,
        delta_cons_self: Optional[torch.Tensor],
        delta_cons_intent: Optional[torch.Tensor],
        delta_cons_pair: Optional[torch.Tensor],
        delta_cons_token_gate: Optional[torch.Tensor],) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        dim = base.dimSem
        self_sem = torch.zeros(batchSize, dim, device=device)
        intent_sem = torch.zeros(batchSize, dim, device=device)
        extras: Dict[str, torch.Tensor] = {}

        if selfState is not None:
            if selfState.size(-1) != base.cons_self_dim:
                raise ValueError(
                    f"IntentionOnlineWrapper: self_sem dim mismatch, expected {base.cons_self_dim}, got {int(selfState.size(-1))}.")
            self_in = selfState.to(device=device)
            self_lin = self.LinearWithLora(base.consSelfProj, self_in, delta_cons_self)
            self_sem = base.consSelfNorm(self_lin)
            extras["cons_self_sem"] = self_sem.detach()

        if intentState is not None:
            if intentState.size(-1) != base.cons_intent_dim:
                raise ValueError(
                    f"IntentionOnlineWrapper: intent_sem dim mismatch, expected {base.cons_intent_dim}, got {int(intentState.size(-1))}.")
            intent_in = intentState.to(device=device)
            intent_lin = self.LinearWithLora(base.consIntentProj, intent_in, delta_cons_intent)
            intent_sem = base.consIntentNorm(intent_lin)
            extras["cons_intent_sem"] = intent_sem.detach()

        cons_sem: Optional[torch.Tensor] = None
        if (selfState is not None) and (intentState is not None):
            pair_feat = torch.cat([
                self_sem,
                intent_sem,
                torch.abs(self_sem - intent_sem),
                self_sem * intent_sem,], dim=-1)

            pair_sem = self.ConsPairWithDelta(base.consPairNet, pair_feat, delta_cons_pair)
            cons_tokens = torch.stack([self_sem, intent_sem, pair_sem], dim=1)
            cons_tokens = base.consTokenTransformer(cons_tokens)

            token_logits = self.ConsTokenGateWithDelta(
                base.consTokenGate,
                cons_tokens.reshape(batchSize, -1),
                delta_cons_token_gate,)
            token_weights = F.softmax(token_logits, dim=-1)

            cons_sem = (cons_tokens * token_weights.unsqueeze(-1)).sum(dim=1)
            cons_sem = base.consFuseNorm(cons_sem + pair_sem)

            extras["cons_pair_sem"] = pair_sem.detach()
            extras["cons_token_weights"] = token_weights.detach()
        elif intentState is not None:
            cons_sem = intent_sem
        elif selfState is not None:
            cons_sem = self_sem

        return cons_sem, self_sem, intent_sem, extras

    def ForwardWithDeltas(
        self,
        x=None,
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        **kwargs,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:

        base: "IntentionExtractor" = self.base 
        device = base.conceptEmb.device

        self_state: Optional[torch.Tensor] = kwargs.get("selfState", None)
        intent_state: Optional[torch.Tensor] = kwargs.get("intentState", None)
        ocrTexts: Optional[List[List[str]]] = kwargs.get("ocrTexts", None)
        extTexts: Optional[List[Optional[str]]] = kwargs.get("extTexts", None)
        prioritizeExt: bool = bool(kwargs.get("prioritizeExt", False))
        textTrust: Optional[List[str]] = kwargs.get("textTrust", None)

        row = deltasPerLayer[0] if (deltasPerLayer is not None and len(deltasPerLayer) > 0) else {}

        delta_sem = row.get("sem", None)
        delta_cons_self = row.get("cons_self", None)
        delta_cons_intent = row.get("cons_intent", None)
        delta_cons_pair = row.get("cons_pair", None)
        delta_cons_token_gate = row.get("cons_token_gate", None)
        delta_ocr = row.get("ocr_gate", None)
        delta_ext = row.get("ext_gate", None)
        delta_sym_k2h = row.get("sym_k2h", None)
        delta_sym_gain = row.get("sym_gain", None)
        delta_sym_tok = row.get("sym_tok", None)
        delta_sym_film = row.get("sym_film", None)
        delta_sym_ctx = row.get("sym_ctx", None)

        batch_size = base.InferBatchSize(self_state, intent_state, ocrTexts, extTexts)
        if batch_size is None:
            base.ResetTransientLossCache()
            intent_zero = torch.zeros(0, base.dimSem, device=device)
            sym_zero = torch.zeros(0, int(base.conceptEmb.size(0)), device=device)
            return intent_zero, sym_zero, {}

        dimSem = base.dimSem
        cons_sem, self_sem, intent_sem_cons, cons_extras = self.EncodeConsciousStatesWithDelta(
            base=base,
            selfState=self_state,
            intentState=intent_state,
            batchSize=batch_size,
            device=device,
            delta_cons_self=delta_cons_self,
            delta_cons_intent=delta_cons_intent,
            delta_cons_pair=delta_cons_pair,
            delta_cons_token_gate=delta_cons_token_gate,)
        extras: Dict[str, torch.Tensor] = dict(cons_extras)
        if cons_sem is not None:
            extras["cons_sem"] = cons_sem.detach()
        text_trust, ext_control, ocr_control = base.BuildTextTrustMasks(batch_size, textTrust, device)
        ext_control_mask = ext_control > 0.0
        extras["text_trust"] = list(text_trust)
        extras["ext_control_mask"] = ext_control_mask.detach()
        extras["ocr_control_weight"] = ocr_control.detach()
        observed_ext_texts = None
        if extTexts is not None:
            observed_ext_texts = [
                "" if text is None else str(text)
                for text in extTexts]

        if ocrTexts is not None:
            merged = base.MergeOcrTexts(ocrTexts)
            sem_ocr, ocr_slots, ocr_slot_mask = self.EncodeStringsWithDelta(
                base, merged, device=device, delta_sem=delta_sem)
            has_ocr_mask = ocr_slot_mask.any(dim=1)
        else:
            sem_ocr = torch.zeros(batch_size, dimSem, device=device)
            ocr_slots = torch.zeros(batch_size, base.n_text_slots, dimSem, device=device)
            ocr_slot_mask = torch.zeros(batch_size, base.n_text_slots, dtype=torch.bool, device=device)
            has_ocr_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if observed_ext_texts is not None:
            sem_ext, ext_slots, ext_slot_mask = self.EncodeStringsWithDelta(
                base, observed_ext_texts, device=device, delta_sem=delta_sem)
            has_ext_mask = ext_slot_mask.any(dim=1)
        else:
            sem_ext = torch.zeros(batch_size, dimSem, device=device)
            ext_slots = torch.zeros(batch_size, base.n_text_slots, dimSem, device=device)
            ext_slot_mask = torch.zeros(batch_size, base.n_text_slots, dtype=torch.bool, device=device)
            has_ext_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        ext_control_float = ext_control.unsqueeze(-1)
        ext_slot_control = ext_control.view(batch_size, 1, 1)
        has_ext_control_mask = has_ext_mask & ext_control_mask
        sem_ext_control = sem_ext * ext_control_float
        ext_slots_control = ext_slots * ext_slot_control
        ext_slot_mask_control = ext_slot_mask & ext_control_mask.view(batch_size, 1)
        ocr_control_float = ocr_control.unsqueeze(-1)

        if (cons_sem is None) and (not has_ocr_mask.any()) and (not has_ext_control_mask.any()):
            base._last_reason_support = None
            if self.training and bool(has_ext_mask.any().item()):
                recall_sem, recall_valid = base.BuildRecallSemanticFromTexts(
                    semOcr=sem_ocr,
                    hasOcrMask=has_ocr_mask,
                    semExt=sem_ext,
                    hasExtMask=has_ext_mask)
                extras.update(base.RunRecallFromSemantic(
                    recallSem=recall_sem,
                    recallSemValid=recall_valid,
                    batchSize=batch_size,
                    ocrTexts=ocrTexts,
                    extTexts=extTexts,
                    device=device))
            else:
                base.ResetRecallLossCache()
            extras["sem_ext_observed"] = sem_ext.detach()
            return None, None, extras

        if cons_sem is not None:
            base_vec = cons_sem
        else:
            base_vec = torch.zeros(batch_size, dimSem, device=device)

        ext_for_ocr = sem_ext_control

        feat_ocr = torch.cat([
                base_vec,
                sem_ocr,
                torch.abs(base_vec - sem_ocr),
                base_vec * sem_ocr,
                ext_for_ocr,
                torch.abs(ext_for_ocr - sem_ocr),
                ext_for_ocr * sem_ocr,],dim=-1,)

        gate_ocr = self.GateWithDelta(base.fuse_ocr_gate, feat_ocr, delta_ocr)
        sem_ocr_fused = gate_ocr * sem_ocr

        ocr_mask_float = has_ocr_mask.unsqueeze(-1).float()
        base_vec = base_vec + base.beta_ocr * (sem_ocr_fused * ocr_mask_float * ocr_control_float)

        extras["sem_ocr_raw"] = sem_ocr.detach()
        extras["sem_ocr_fused"] = sem_ocr_fused.detach()
        extras["gate_ocr"] = gate_ocr.detach()
        extras["has_ocr_mask"] = has_ocr_mask.detach()
        extras["sem_ocr_slots"] = ocr_slots.detach()
        extras["sem_ocr_slot_mask"] = ocr_slot_mask.detach()

        feat_ext = torch.cat([
                base_vec,
                sem_ext_control,
                torch.abs(base_vec - sem_ext_control),
                base_vec * sem_ext_control,],dim=-1,)

        gate_ext = self.GateWithDelta(base.fuse_ext_gate, feat_ext, delta_ext)

        has_ext_mask_float = has_ext_control_mask.unsqueeze(-1).float()
        has_ext_mask_exp = has_ext_control_mask.unsqueeze(-1)

        if prioritizeExt:
            gamma = 0.5 + 0.5 * gate_ext
            candidate = (1.0 - gamma) * base_vec + gamma * sem_ext_control
            intentSem = torch.where(has_ext_mask_exp, candidate, base_vec)
            extras["gamma_ext"] = gamma.detach()
        else:
            sem_ext_fused = gate_ext * sem_ext_control
            intentSem = base_vec + base.beta_ext * (sem_ext_fused * has_ext_mask_float)
            extras["sem_ext_fused"] = sem_ext_fused.detach()

        extras["sem_ext_observed"] = sem_ext.detach()
        extras["sem_ext_controlled"] = sem_ext_control.detach()
        extras["gate_ext"] = gate_ext.detach()
        extras["has_ext_mask"] = has_ext_mask.detach()
        extras["sem_ext_slots"] = ext_slots.detach()
        extras["sem_ext_slot_mask"] = ext_slot_mask.detach()

        self_token_mask = torch.full((batch_size, 1), self_state is not None, dtype=torch.bool, device=device)
        intent_token_mask = torch.full((batch_size, 1), intent_state is not None, dtype=torch.bool, device=device)
        tokens = torch.cat([
            self_sem.unsqueeze(1),
            intent_sem_cons.unsqueeze(1),
            ocr_slots * ocr_control.view(batch_size, 1, 1),
            ext_slots_control,], dim=1)
        token_mask = torch.cat([
            self_token_mask,
            intent_token_mask,
            ocr_slot_mask,
            ext_slot_mask_control,], dim=1)

        def safe_token_mask(token_mask: torch.Tensor) -> torch.Tensor:
            safe = token_mask.clone()
            all_pad = ~safe.any(dim=1) 
            safe[:, 0] |= all_pad
            return safe

        token_mask_safe = safe_token_mask(token_mask)
        src_key_padding_mask = ~token_mask_safe 

        trans_out: Optional[torch.Tensor] = None
        if token_mask.any(): 
            trans_out = base.intentTransformer(tokens, src_key_padding_mask=src_key_padding_mask)

            mask_float = token_mask.float().unsqueeze(-1) 
            sum_vec = (trans_out * mask_float).sum(dim=1)
            mask_sum = mask_float.sum(dim=1)
            denom = mask_sum.clamp(min=1.0)
            fused = sum_vec / denom

            intentSem = intentSem + base.beta_trans * fused
            extras["intent_trans_norm"] = fused.norm(dim=-1, keepdim=True).detach()
            extras["intent_trans_mask_sum"] = mask_sum.detach()

        base._last_reason_support = None
        for t in range(int(base.reason_steps)):

            symbol_logits = F.linear(intentSem, base.conceptEmb, base.conceptBias)
            symProbs_t, support = base.reasoner(
                symbol_logits,
                base.conceptEmb,
                ctx=intentSem,
                returnSupport=True,)

            ctrl = self.SymCtrlWithDelta(
                base=base,
                symProbs=symProbs_t,
                support=support,
                token_mask=token_mask, 
                delta_sym_k2h=delta_sym_k2h,
                delta_sym_gain=delta_sym_gain,
                delta_sym_tok=delta_sym_tok,
                delta_sym_film=delta_sym_film,
                delta_sym_ctx=delta_sym_ctx,)

            base_ctx = intentSem

            feat_ocr2 = torch.cat([
                base_ctx, sem_ocr, torch.abs(base_ctx - sem_ocr), base_ctx * sem_ocr,
                sem_ext_control,
                torch.abs(sem_ext_control - sem_ocr),
                sem_ext_control * sem_ocr], dim=-1)

            gate_ocr2 = self.GateWithDelta(base.fuse_ocr_gate, feat_ocr2, delta_ocr)
            sem_ocr_fused2 = gate_ocr2 * sem_ocr
            base2 = base_ctx + torch.tanh(base.beta_ocr) * ctrl["g_ocr"] * (
                sem_ocr_fused2 * ocr_mask_float * ocr_control_float)

            feat_ext2 = torch.cat([
                base2,
                sem_ext_control,
                torch.abs(base2 - sem_ext_control),
                base2 * sem_ext_control], dim=-1)

            gate_ext2 = self.GateWithDelta(base.fuse_ext_gate, feat_ext2, delta_ext)

            if prioritizeExt:
                gamma0 = 0.5 + 0.5 * gate_ext2
                gamma_eff = gamma0 * ctrl["g_ext"]
                cand = (1.0 - gamma_eff) * base2 + gamma_eff * sem_ext_control
                intent2 = torch.where(has_ext_mask_exp, cand, base2)
            else:
                sem_ext_fused2 = gate_ext2 * sem_ext_control
                intent2 = base2 + torch.tanh(base.beta_ext) * ctrl["g_ext"] * (
                    sem_ext_fused2 * has_ext_mask_float)

            if trans_out is not None:
                w = ctrl["tok_w"]
                w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                fused2 = (trans_out * w.unsqueeze(-1)).sum(dim=1)
                intent2 = intent2 + torch.tanh(base.beta_trans) * ctrl["g_trans"] * fused2

            intent2 = base.sym_norm(intent2 + torch.tanh(base.beta_sym) * ctrl["sym_ctx"])
            intent2 = base.sym_norm(intent2 * ctrl["film_scale"] + ctrl["beta"])

            u = torch.sigmoid(base.beta_update)
            intentSem = (1.0 - u) * intentSem + u * intent2

            extras["sym_probs_loop"] = symProbs_t.detach()
            extras["sym_ctrl_gains"] = torch.cat([ctrl["g_ocr"], ctrl["g_ext"], ctrl["g_trans"]], dim=-1).detach()
            extras["sym_tok_w"] = ctrl["tok_w"].detach()
            extras["sym_reason_alpha"] = support["alpha_eff"].detach()

        final_logits = F.linear(intentSem, base.conceptEmb, base.conceptBias)
        symProbs, final_support = base.reasoner(
            final_logits,
            base.conceptEmb,
            ctx=intentSem,
            returnSupport=True,)
        base._last_reason_support = final_support
        extras["reason_alpha_final"] = final_support["alpha_eff"].detach()
        if self.training:
            recall_sem, recall_valid = base.BuildRecallSemanticFromTexts(
                semOcr=sem_ocr,
                hasOcrMask=has_ocr_mask,
                semExt=sem_ext,
                hasExtMask=has_ext_mask)
            extras.update(base.RunRecallFromSemantic(
                recallSem=recall_sem,
                recallSemValid=recall_valid,
                batchSize=batch_size,
                ocrTexts=ocrTexts,
                extTexts=extTexts,
                device=device,))
        else:
            base.ResetRecallLossCache()
            _, recall_texts = base.RecallGenerateFromSemantic(
                intentSem,
                maxLen=base.recall_safety_max_len,)
            extras["recall_texts"] = recall_texts
        return intentSem, symProbs, extras


    def SymCtrlWithDelta(
        self,
        base: "IntentionExtractor",
        symProbs: torch.Tensor,
        support: Dict[str, torch.Tensor],
        token_mask: torch.Tensor,
        *,
        delta_sym_k2h: Optional[torch.Tensor],
        delta_sym_gain: Optional[torch.Tensor],
        delta_sym_tok: Optional[torch.Tensor],
        delta_sym_film: Optional[torch.Tensor],
        delta_sym_ctx: Optional[torch.Tensor],) -> Dict[str, torch.Tensor]:

        featK = torch.cat([
            symProbs,
            support["sym_probs0"],
            support["support_imp"],
            support["support_cooc"],
            support["support_contr"],], dim=-1)

        h = self.LinearWithLora(base.symCtrl.k2h[0], featK, delta_sym_k2h)
        h = base.symCtrl.k2h[1](h)
        h = base.symCtrl.k2h[2](h)

        gains = self.LinearWithLora(base.symCtrl.gain_head[0], h, delta_sym_gain)
        gains = torch.sigmoid(gains)
        g_ocr = gains[:, 0:1]
        g_ext = gains[:, 1:2]
        g_trans = gains[:, 2:3]

        tok_logits = self.LinearWithLora(base.symCtrl.tok_head[0], h, delta_sym_tok)
        tok_w = F.softmax(tok_logits, dim=-1)
        mask_f = token_mask.float()
        tok_w = tok_w * mask_f
        denom = tok_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        tok_w = tok_w / denom

        film = self.LinearWithLora(base.symCtrl.film_head[0], h, delta_sym_film)
        gamma_raw, beta = film.chunk(2, dim=-1)
        film_scale = torch.exp(gamma_raw.clamp(-4.0, 4.0))

        sym_ctx_raw = symProbs @ base.conceptEmb
        sym_ctx = self.LinearWithLora(base.symCtrl.ctx_proj[0], sym_ctx_raw, delta_sym_ctx)
        sym_ctx = base.symCtrl.ctx_proj[1](sym_ctx)
        sym_ctx = base.symCtrl.ctx_proj[2](sym_ctx)
        sym_ctx = base.symCtrl.ctx_proj[3](sym_ctx)

        return {
            "h": h,
            "g_ocr": g_ocr,
            "g_ext": g_ext,
            "g_trans": g_trans,
            "tok_w": tok_w,
            "film_scale": film_scale, 
            "beta": beta,
            "sym_ctx": sym_ctx,}


    @torch.no_grad()
    def CommitOne(self, site: str, layerIdx: int, a: torch.Tensor, b: torch.Tensor, scale: float) -> bool:
        if layerIdx != 0:
            return False

        base: "IntentionExtractor" = self.base 
        init = {
            "A": a.detach().clone(),
            "B": b.detach().clone(),
            "scale": float(scale),}

        if site == "sem":
            base.semProj[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "cons_self":
            base.consSelfProj.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "cons_intent":
            base.consIntentProj.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "cons_pair":
            base.consPairNet[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "cons_token_gate":
            base.consTokenGate[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "ocr_gate":
            base.fuse_ocr_gate[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "ext_gate":
            base.fuse_ext_gate[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "sym_k2h":
            base.symCtrl.k2h[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "sym_gain":
            base.symCtrl.gain_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "sym_tok":
            base.symCtrl.tok_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "sym_film":
            base.symCtrl.film_head[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        elif site == "sym_ctx":
            base.symCtrl.ctx_proj[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar); return True
        else:
            raise ValueError(f"Unknown site: {site}")

    def GetInternalLoss(
        self,
        symProbs: torch.Tensor,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if hasattr(self.base, "GetInternalLoss"):
            return self.base.GetInternalLoss(symProbs) 
        raise RuntimeError("Base IntentionExtractor has no GetInternalLoss.")



class TestIntentionMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)

    def MakeTestModel(self) -> "IntentionExtractor":
        return IntentionExtractor(
            maxSeqLen=8,
            recallSafetyMaxLen=2,
            dimEmbed=32,
            dimEncoderHidden=16,
            numEncoderLayers=2,
            dimSem=32,
            consDim=64,
            nSymbols=12,
            reasonSteps=1,
            reasonerHiddenDim=32,
            nTextSlots=1,).to(self.device)

    def MakeDummyBatch(
        self,
        model: "IntentionExtractor",
        batch_size: int = 8,
        with_ocr: bool = True,
        with_ext: bool = True,
        with_cons: bool = True,
        compact_text: bool = False,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[List[str]]], Optional[List[Optional[str]]], torch.Tensor]:
        nSymbols = int(model.conceptEmb.size(0))
        selfState: Optional[torch.Tensor] = None
        intentState: Optional[torch.Tensor] = None
        ocrTexts: Optional[List[List[str]]] = None
        extTexts: Optional[List[Optional[str]]] = None

        if with_cons:
            selfState = torch.randn(batch_size, int(model.cons_self_dim), device=self.device)
            intentState = torch.randn(batch_size, int(model.cons_intent_dim), device=self.device)

        if with_ocr:
            if compact_text:
                ocrTexts = [["ab"] for _ in range(batch_size)]
            else:
                ocrTexts = [[f"ocr text {i} sample"] for i in range(batch_size)]
        if with_ext:
            if compact_text:
                extTexts = ["cd" for _ in range(batch_size)]
            else:
                extTexts = [f"external hint {i}" for i in range(batch_size)]

        targetSym = torch.randint(low=0,high=2,size=(batch_size, nSymbols),dtype=torch.float32,device=self.device,)
        return selfState, intentState, ocrTexts, extTexts, targetSym

    def ComputeBaseLossBundle(
        self,
        model: "IntentionExtractor",
        selfState: Optional[torch.Tensor],
        intentState: Optional[torch.Tensor],
        ocrTexts: Optional[List[List[str]]],
        extTexts: Optional[List[Optional[str]]],
        targetSym: torch.Tensor,
        *,
        prioritizeExt: bool = False,
        textTrust: Optional[List[str]] = None,) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        intentSem, symProbs, extras = model(
            selfState,
            intentState,
            ocrTexts=ocrTexts,
            extTexts=extTexts,
            prioritizeExt=prioritizeExt,
            textTrust=textTrust,)

        if symProbs is None:
            raise AssertionError("ComputeBaseLossBundle: symProbs is None.")

        loss_main = F.binary_cross_entropy(symProbs, targetSym)
        internal_loss, stats = model.GetInternalLoss(symProbs)
        loss_total = loss_main + 0.1 * internal_loss
        loss_reason_total = (
            stats["reason_cooc_sym_pen"]
            + stats["reason_contr_sym_pen"]
            + stats["reason_dynamic_pen"]
            + stats["reason_entropy"])

        bundle = {
            "loss_total": loss_total,
            "loss_main": loss_main,
            "loss_internal": internal_loss,
            "loss_reason_total": loss_reason_total,
            "loss_recall_total": stats["recall_loss_total"],}
        return bundle, symProbs, extras

    def TestIntentionExtractorIOShapes(self) -> bool:
        try:
            model = self.MakeTestModel()
            model.eval()

            B = 2
            selfState, intentState, ocrTexts, extTexts, _ = self.MakeDummyBatch(
                model,
                batch_size=B,
                with_ocr=True,
                with_ext=True,
                with_cons=True,
                compact_text=True,)

            def print_shape(name: str, tensor: torch.Tensor):
                print(f"{name}: {tuple(tensor.shape)}")

            def print_nested(prefix: str, obj):
                if isinstance(obj, torch.Tensor):
                    print_shape(prefix, obj)
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        next_prefix = f"{prefix}.{key}" if prefix else key
                        print_nested(next_prefix, value)

            with torch.no_grad():
                if selfState is not None:
                    print_shape("input.selfState", selfState)
                if intentState is not None:
                    print_shape("input.intentState", intentState)

                intentSem, symProbs, extras = model(
                    selfState,
                    intentState,
                    ocrTexts=ocrTexts,
                    extTexts=extTexts,
                    prioritizeExt=False,)

                if intentSem is not None:
                    print_shape("output.intentSem", intentSem)
                if symProbs is not None:
                    print_shape("output.symProbs", symProbs)
                print_nested("output.extras", extras)

            assert intentSem is not None and intentSem.shape == (B, model.dimSem)
            assert symProbs is not None and symProbs.shape == (B, int(model.conceptEmb.size(0)))
            return True
        except Exception as e:
            print("TestIntentionExtractorIOShapes error:", type(e).__name__, e)
            return False

    @staticmethod
    def BestSeen(history: List[float]) -> float:
        if len(history) <= 0:
            raise ValueError("BestSeen: history is empty.")
        return min(history)

    def GradCoverage(self, named: Dict[str, torch.nn.Parameter], min_ratio: float, must_have: List[str]) -> bool:
        total_trainable = sum(1 for p in named.values() if p.requires_grad)
        total_with_grad = sum(1 for p in named.values() if p.requires_grad and (p.grad is not None))
        ratio = total_with_grad / max(1, total_trainable)

        missing_names = [n for n in must_have if n not in named]
        if missing_names:
            print("Unknown parameter names (typo / changed module?):", missing_names)
            return False

        missing_grad = [n for n in must_have if named[n].requires_grad and (named[n].grad is None)]
        if missing_grad:
            print("Missing gradient parameters:", missing_grad)
            return False

        if ratio < min_ratio:
            print(f"Gradient coverage too low: {ratio:.2%} < {min_ratio:.2%}")
            return False

        print(f"Gradient coverage: {ratio:.2%}")
        return True


    def ForwardVariants(self) -> bool:
        try:
            model = IntentionExtractor().to(self.device)
            model.eval()

            dimSem = int(model.dimSem)
            nSymbols = int(model.conceptEmb.size(0))
            selfDim = int(model.cons_self_dim)
            intentDim = int(model.cons_intent_dim)

            B = 4

            self_only = torch.randn(B, selfDim, device=self.device)
            intent_only = torch.randn(B, intentDim, device=self.device)
            intentSem, symProbs, extras = model(self_only, intent_only, ocrTexts=None, extTexts=None)
            assert intentSem is not None and symProbs is not None
            assert intentSem.shape == (B, dimSem)
            assert symProbs.shape == (B, nSymbols)
            assert "recall_texts" in extras
            assert "recall_logits" not in extras
            assert "recall_targets" not in extras

            cons_self_only = torch.randn(B, selfDim, device=self.device)
            intentSem0, symProbs0, _ = model(cons_self_only, None, ocrTexts=None, extTexts=None)
            assert intentSem0 is not None and symProbs0 is not None
            assert intentSem0.shape == (B, dimSem)
            assert symProbs0.shape == (B, nSymbols)

            cons_int_only = torch.randn(B, intentDim, device=self.device)
            intentSem1, symProbs1, _ = model(None, cons_int_only, ocrTexts=None, extTexts=None)
            assert intentSem1 is not None and symProbs1 is not None
            assert intentSem1.shape == (B, dimSem)
            assert symProbs1.shape == (B, nSymbols)

            ocrTexts = [[f"hello {i}"] for i in range(B)]
            intentSem2, symProbs2, extras2 = model(None, None, ocrTexts=ocrTexts, extTexts=None)
            assert intentSem2 is not None and symProbs2 is not None
            assert intentSem2.shape == (B, dimSem)
            assert symProbs2.shape == (B, nSymbols)

            extTexts = [f"world {i}" for i in range(B)]
            intentSem3, symProbs3, extras3 = model(
                None,
                None,
                ocrTexts=None,
                extTexts=extTexts,
                textTrust=[TEXT_TRUST_OPERATOR_COMMAND for _ in range(B)])
            assert intentSem3 is not None and symProbs3 is not None
            assert intentSem3.shape == (B, dimSem)
            assert symProbs3.shape == (B, nSymbols)

            cons_full_self = torch.randn(B, selfDim, device=self.device)
            cons_full_intent = torch.randn(B, intentDim, device=self.device)
            ocr_full = [[f"ocr {i} text"] for i in range(B)]
            ext_full = [f"ext {i} text" for i in range(B)]

            intentSem4, symProbs4, extras4 = model(
                cons_full_self,
                cons_full_intent,
                ocrTexts=ocr_full,
                extTexts=ext_full,
                prioritizeExt=True,
                textTrust=[TEXT_TRUST_OPERATOR_COMMAND for _ in range(B)])
            assert intentSem4 is not None and symProbs4 is not None
            assert intentSem4.shape == (B, dimSem)
            assert symProbs4.shape == (B, nSymbols)

            for t in (symProbs, symProbs0, symProbs1, symProbs2, symProbs3, symProbs4):
                assert torch.isfinite(t).all()
                assert t.min().item() >= -1e-6 and t.max().item() <= 1.0 + 1e-6

            print("ForwardVariants passed.")
            return True
        except AssertionError as e:
            print("ForwardVariants failed:", e)
            return False
        except Exception as e:
            print("ForwardVariants error:", e)
            return False

    def TrainStepSmokeBase(self) -> bool:
        try:
            model = IntentionExtractor().to(self.device)
            model.train()

            selfState, intentState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(model, batch_size=8)

            intentSem, symProbs, extras = model(selfState, intentState, ocrTexts=ocrTexts, extTexts=extTexts)
            assert symProbs is not None
            assert torch.isfinite(symProbs).all()

            loss_main = F.binary_cross_entropy(symProbs, targetSym)

            internal_loss, stats = model.GetInternalLoss(symProbs)
            loss = loss_main + 0.1 * internal_loss

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            opt.zero_grad(set_to_none=True)
            loss.backward()

            named = dict(model.named_parameters())

            must_have = [
                "encoder.embedding.weight",
                "encoder.att_proj.weight",
                "semProj.0.target.weight",
                "chunkFuseFwd.weight",
                "chunkFuseBwd.weight",
                "chunkStateProj.0.weight",
                "slotFuseInNorm.weight",
                "slotQuery",
                "slotDynQuery.0.weight",
                "slotMixGate.0.weight",
                "slotCrossAttn.q_proj.weight",
                "slotPost.0.weight",
                "chunkFuseOut.0.weight",
                "consSelfProj.target.weight",
                "consIntentProj.target.weight",
                "consPairNet.0.target.weight",
                "consTokenTransformer.layers.0.self_attn.q_proj.weight",
                "consTokenGate.0.target.weight",
                "recallTokEmb.weight",
                "recallStart",
                "recallCond.0.weight",
                "intentTransformer.layers.0.self_attn.q_proj.weight",
                "recallDecoder.layers.0.self_attn.q_proj.weight",
                "recallDecoder.layers.0.cross_attn.q_proj.weight",
                "recallHead.weight",
                "fuse_ocr_gate.0.target.weight",
                "fuse_ext_gate.0.target.weight",
                "conceptEmb",
                "reasoner.relImp",
                "reasoner.relCooc",
                "reasoner.relContr",
                "symCtrl.k2h.0.target.weight",
                "symCtrl.gain_head.0.target.weight",
                "symCtrl.tok_head.0.target.weight",
                "symCtrl.film_head.0.target.weight",
                "symCtrl.ctx_proj.0.target.weight",
                "beta_sym",
                "beta_update",]

            for l in range(model.encoder.rnn_f.num_layers):
                must_have += [
                    f"encoder.rnn_f.weight_ih_l{l}",
                    f"encoder.rnn_f.weight_hh_l{l}",
                    f"encoder.rnn_b.weight_ih_l{l}",
                    f"encoder.rnn_b.weight_hh_l{l}",]
                
            ok_cov = self.GradCoverage(named, min_ratio=0.5, must_have=must_have)
            assert ok_cov, "Gradient coverage check failed."

            for n, p in named.items():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

            opt.step()
            print("TrainStepSmokeBase passed.")
            return True
        except AssertionError as e:
            print("TrainStepSmokeBase failed:", e)
            return False
        except Exception as e:
            print("TrainStepSmokeBase error:", e)
            return False

    def AllTrainableGradCoverageBase(self) -> bool:
        try:
            model = self.MakeTestModel()
            model.train()

            selfState, intentState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(model, batch_size=2, compact_text=True)
            losses, symProbs, _ = self.ComputeBaseLossBundle(
                model,
                selfState,
                intentState,
                ocrTexts,
                extTexts,
                targetSym,)

            model.zero_grad(set_to_none=True)
            losses["loss_total"].backward()

            missing = []
            non_finite = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    missing.append(name)
                    continue
                if not torch.isfinite(param.grad).all():
                    non_finite.append(name)

            assert len(missing) == 0, f"Missing grad for trainable params: {missing}"
            assert len(non_finite) == 0, f"Non-finite grad for params: {non_finite}"

            print(f"AllTrainableGradCoverageBase passed. params={sum(1 for p in model.parameters() if p.requires_grad)}")
            return True
        except AssertionError as e:
            print("AllTrainableGradCoverageBase failed:", e)
            return False
        except Exception as e:
            print("AllTrainableGradCoverageBase error:", e)
            return False

    def BranchGradientCoverageBase(self) -> bool:
        try:
            specs = [
                {"with_cons": True, "with_ocr": True, "with_ext": True, "prioritizeExt": False, "textTrust": None},
                {"with_cons": True, "with_ocr": False, "with_ext": False, "prioritizeExt": False, "textTrust": None},
                {"with_cons": False, "with_ocr": True, "with_ext": False, "prioritizeExt": False, "textTrust": None},
                {
                    "with_cons": False,
                    "with_ocr": False,
                    "with_ext": True,
                    "prioritizeExt": True,
                    "textTrust": [TEXT_TRUST_OPERATOR_COMMAND, TEXT_TRUST_OPERATOR_COMMAND],},]

            union: set = set()
            branch_grads: Dict[str, set] = {}
            ref_model = self.MakeTestModel()
            all_trainable = {n for n, p in ref_model.named_parameters() if p.requires_grad}

            for spec in specs:
                model = self.MakeTestModel()
                model.train()
                selfState, intentState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(
                    model,
                    batch_size=2,
                    with_cons=spec["with_cons"],
                    with_ocr=spec["with_ocr"],
                    with_ext=spec["with_ext"],
                    compact_text=True,)

                losses, _, _ = self.ComputeBaseLossBundle(
                    model,
                    selfState,
                    intentState,
                    ocrTexts,
                    extTexts,
                    targetSym,
                    prioritizeExt=bool(spec["prioritizeExt"]),
                    textTrust=spec["textTrust"],)

                model.zero_grad(set_to_none=True)
                losses["loss_total"].backward()

                got = {n for n, p in model.named_parameters() if p.requires_grad and (p.grad is not None)}
                union |= got
                key = f"cons={int(spec['with_cons'])},ocr={int(spec['with_ocr'])},ext={int(spec['with_ext'])},prio={int(spec['prioritizeExt'])}"
                branch_grads[key] = got

            missing_union = sorted(all_trainable - union)
            assert len(missing_union) == 0, f"Union grad coverage missing params: {missing_union}"

            assert "consSelfProj.target.weight" in branch_grads["cons=1,ocr=0,ext=0,prio=0"]
            assert "consIntentProj.target.weight" in branch_grads["cons=1,ocr=0,ext=0,prio=0"]
            assert "fuse_ocr_gate.0.target.weight" in branch_grads["cons=0,ocr=1,ext=0,prio=0"]
            assert "fuse_ext_gate.0.target.weight" in branch_grads["cons=0,ocr=0,ext=1,prio=1"]

            print(f"BranchGradientCoverageBase passed. union={len(union)}/{len(all_trainable)}")
            return True
        except AssertionError as e:
            print("BranchGradientCoverageBase failed:", e)
            return False
        except Exception as e:
            print("BranchGradientCoverageBase error:", e)
            return False

    def LossComponentsDecreaseBase(self, steps: int = 6) -> bool:
        try:
            model = self.MakeTestModel()
            model.train()

            selfState, intentState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(model, batch_size=2, compact_text=True)
            opt = torch.optim.Adam(model.parameters(), lr=3e-3)

            with torch.no_grad():
                start_losses, _, _ = self.ComputeBaseLossBundle(
                    model,
                    selfState,
                    intentState,
                    ocrTexts,
                    extTexts,
                    targetSym,)
                start_total = float(start_losses["loss_total"].item())
                start_main = float(start_losses["loss_main"].item())
                start_recall = float(start_losses["loss_recall_total"].item())

            hist_total: List[float] = []
            hist_main: List[float] = []
            hist_recall: List[float] = []

            for _ in range(int(steps)):
                losses, _, _ = self.ComputeBaseLossBundle(
                    model,
                    selfState,
                    intentState,
                    ocrTexts,
                    extTexts,
                    targetSym,)
                opt.zero_grad(set_to_none=True)
                losses["loss_total"].backward()
                opt.step()

                hist_total.append(float(losses["loss_total"].detach().item()))
                hist_main.append(float(losses["loss_main"].detach().item()))
                hist_recall.append(float(losses["loss_recall_total"].detach().item()))

            best_total = self.BestSeen(hist_total)
            best_main = self.BestSeen(hist_main)
            best_recall = self.BestSeen(hist_recall)

            assert best_total < start_total, f"total loss did not decrease: start={start_total:.6f}, best={best_total:.6f}"
            assert best_main < start_main, f"main loss did not decrease: start={start_main:.6f}, best={best_main:.6f}"
            assert best_recall < start_recall, f"recall loss did not decrease: start={start_recall:.6f}, best={best_recall:.6f}"

            print(
                f"LossComponentsDecreaseBase passed. total {start_total:.6f}->{best_total:.6f}, "
                f"main {start_main:.6f}->{best_main:.6f}, recall {start_recall:.6f}->{best_recall:.6f}")
            return True
        except AssertionError as e:
            print("LossComponentsDecreaseBase failed:", e)
            return False
        except Exception as e:
            print("LossComponentsDecreaseBase error:", e)
            return False

    def NormalTrainingConvergenceBase(self, steps: int = 80, logEvery: int = 20) -> bool:
        try:
            model = IntentionExtractor().to(self.device)
            model.train()

            selfState, intentState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(model, batch_size=12)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            with torch.no_grad():
                _, sym0, _ = model(selfState, intentState, ocrTexts=ocrTexts, extTexts=extTexts)
                start_main = F.binary_cross_entropy(sym0, targetSym).item()
                internal0, _ = model.GetInternalLoss(sym0)
                start = start_main + 0.1 * internal0.item()

            hist = []
            for t in range(1, steps + 1):
                intentSem, symProbs, extras = model(selfState, intentState, ocrTexts=ocrTexts, extTexts=extTexts)
                loss_main = F.binary_cross_entropy(symProbs, targetSym)
                internal_loss, _ = model.GetInternalLoss(symProbs)
                loss = loss_main + 0.1 * internal_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                val = float(loss.item())
                hist.append(val)
                if (t % logEvery) == 0 or t == 1:
                    print(f"[IntentionTrain] step {t}/{steps} | loss={val:.6f}")

            end = hist[-1]
            tail_mean = sum(hist[-10:]) / min(10, len(hist))
            print(f"[IntentionTrain] start={start:.6f} -> end={end:.6f}, tail_mean={tail_mean:.6f}")
            assert tail_mean <= 0.8 * start, "Training did not converge enough (<20% decline)."
            print("NormalTrainingConvergenceBase passed.")
            return True
        except AssertionError as e:
            print("NormalTrainingConvergenceBase failed:", e)
            return False
        except Exception as e:
            print("NormalTrainingConvergenceBase error:", e)
            return False


    def WrapperForwardEqualWhenNoInitRank(self) -> bool:
        try:
            base = self.MakeTestModel()
            base.eval()
            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.eval()

            selfState, intentState, ocrTexts, extTexts, _ = self.MakeDummyBatch(
                base,
                batch_size=2,
                compact_text=True,)

            with torch.no_grad():
                y_base = base(selfState, intentState, ocrTexts=ocrTexts, extTexts=extTexts, prioritizeExt=True)
                y_wrap = wrapper(selfState, intentState, ocrTexts=ocrTexts, extTexts=extTexts, prioritizeExt=True)

            intent_base, sym_base, extras_base = y_base
            intent_wrap, sym_wrap, extras_wrap = y_wrap

            assert intent_base is not None and intent_wrap is not None
            assert sym_base is not None and sym_wrap is not None

            max_abs_int = (intent_base - intent_wrap).abs().max().item()
            max_abs_sym = (sym_base - sym_wrap).abs().max().item()
            assert max_abs_int < 1e-6, f"intent mismatch: {max_abs_int:.3e}"
            assert max_abs_sym < 1e-6, f"symProbs mismatch: {max_abs_sym:.3e}"
            assert extras_wrap["recall_texts"] == extras_base["recall_texts"]
            for key in ("recall_logits", "recall_targets", "recall_valid", "recall_pred_ids"):
                assert key not in extras_wrap, f"wrapper eval exposed training-only {key}"
            assert base._last_recall_logits is None
            assert base._last_recall_hidden is None
            assert base._last_recall_targets is None
            assert base._last_recall_valid is None
            assert base._last_recall_cons_sem is None

            print("WrapperForwardEqualWhenNoInitRank passed.")
            return True
        except AssertionError as e:
            print("WrapperForwardEqualWhenNoInitRank failed:", e)
            return False
        except Exception as e:
            print("WrapperForwardEqualWhenNoInitRank error:", e)
            return False

    def WrapperAPIBasics(self) -> bool:
        try:
            base = IntentionExtractor().to(self.device)
            base.eval()
            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()

            r0 = wrapper.Update("ranks")["ranks"]
            for row in r0["perLayer"]:
                for k, v in row.items():
                    assert v == 0, f"expected 0 rank at {k}, got {v}"

            wrapper.Update("grow", growFactor=2.0, addEach=2)
            r1 = wrapper.Update("ranks")["ranks"]
            assert any(r1["sum"][k] > 0 for k in r1["sum"].keys())

            wrapper.Update("rollback")
            r2 = wrapper.Update("ranks")["ranks"]
            for row in r2["perLayer"]:
                for k, v in row.items():
                    assert v == 0

            print("WrapperAPIBasics passed.")
            return True
        except AssertionError as e:
            print("WrapperAPIBasics failed:", e)
            return False
        except Exception as e:
            print("WrapperAPIBasics error:", e)
            return False

    def WrapperKeepsBaseEval(self) -> bool:
        try:
            base = IntentionExtractor().to(self.device)
            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()
            assert wrapper.training and (not base.training), "wrapper.train() should not set base to train()."
            print("WrapperKeepsBaseEval passed.")
            return True
        except AssertionError as e:
            print("WrapperKeepsBaseEval failed:", e)
            return False
        except Exception as e:
            print("WrapperKeepsBaseEval error:", e)
            return False

    def WrapperCandGradSmoke(self) -> bool:
        try:
            base = IntentionExtractor().to(self.device)
            base.eval()

            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()

            wrapper.Update("grow", growFactor=1.0, addEach=4)

            selfState, intentState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(base, batch_size=6)

            opt = torch.optim.Adam(list(wrapper.CandParameters()), lr=3e-3)

            dummy_td = torch.ones(targetSym.size(0), 1, device=self.device)

            intentSem, symProbs, extras = wrapper(selfState, intentState, ocrTexts=ocrTexts, extTexts=extTexts, prioritizeExt=False, tdError=dummy_td)
            loss = F.binary_cross_entropy(symProbs, targetSym)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            wrapper.Update("accumulategrads")

            for p in wrapper.CandParameters():
                assert p.grad is not None, "Candidate parameter has no gradient."
                assert torch.isfinite(p.grad).all(), "Candidate parameter grad not finite."

            opt.step()
            print("WrapperCandGradSmoke passed.")
            return True
        except AssertionError as e:
            print("WrapperCandGradSmoke failed:", e)
            return False
        except Exception as e:
            print("WrapperCandGradSmoke error:", e)
            return False

    def WrapperCandidateConvergence(self, steps: int = 4) -> bool:
        try:
            base = self.MakeTestModel()
            base.eval()

            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()
            wrapper.Update("grow", growFactor=1.0, addEach=2)

            selfState, intentState, _, _, targetSym = self.MakeDummyBatch(
                base,
                batch_size=2,
                with_ocr=False,
                with_ext=False,
                with_cons=True,
                compact_text=True,)

            opt = torch.optim.Adam(list(wrapper.CandParameters()), lr=8e-3)

            with torch.no_grad():
                _, sym0, _ = wrapper(
                    selfState,
                    intentState,
                    ocrTexts=None,
                    extTexts=None,
                    prioritizeExt=False,)
                start = float(F.binary_cross_entropy(sym0, targetSym).item())

            hist: List[float] = []
            for _ in range(int(steps)):
                _, symProbs, _ = wrapper(
                    selfState,
                    intentState,
                    ocrTexts=None,
                    extTexts=None,
                    prioritizeExt=False,)
                loss = F.binary_cross_entropy(symProbs, targetSym)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()
                hist.append(float(loss.detach().item()))

            best = self.BestSeen(hist)
            assert best < start, f"wrapper candidate loss did not decrease: start={start:.6f}, best={best:.6f}"

            print(f"WrapperCandidateConvergence passed. start={start:.6f}, best={best:.6f}")
            return True
        except AssertionError as e:
            print("WrapperCandidateConvergence failed:", e)
            return False
        except Exception as e:
            print("WrapperCandidateConvergence error:", e)
            return False
        
    def WrapperManualGrowTrainAndCommit(self) -> bool:
        try:
            base = IntentionExtractor().to(self.device)
            base.eval()

            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()

            wrapper.Update("grow", growFactor=1.0, addEach=4)

            selfState, intentState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(base, batch_size=8)
            opt = torch.optim.Adam(list(wrapper.CandParameters()), lr=3e-3)

            steps = 10
            dummy_td = torch.ones(targetSym.size(0), 1, device=self.device)

            for _ in range(steps):
                intentSem, symProbs, extras = wrapper(
                    selfState,
                    intentState,
                    ocrTexts=ocrTexts,
                    extTexts=extTexts,
                    prioritizeExt=False,
                    tdError=dummy_td,)
                
                loss = F.binary_cross_entropy(symProbs, targetSym)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            sites = [
                "sem",
                "cons_self",
                "cons_intent",
                "cons_pair",
                "cons_token_gate",
                "ocr_gate",
                "ext_gate",
                "sym_k2h",
                "sym_gain",
                "sym_tok",
                "sym_film",
                "sym_ctx",]

            expected: Dict[str, torch.Tensor] = {}
            for site in sites:
                expected[site] = wrapper.ComposeOne(site, layerIdx=0).detach().clone()

            res = wrapper.Update("commit")
            assert res["ok"], "Commit failed."
            stats = res.get("commit_stats", {})
            print(
                f"[IntentionCommit] committed_rank={stats.get('committed_rank', 0)}, "
                f"triples={stats.get('committed_triples', 0)}")

            r_after = wrapper.Update("ranks")["ranks"]
            for row in r_after["perLayer"]:
                for k, v in row.items():
                    assert int(v) == 0, f"rank not cleared: site={k}, rank={v}"

            atol, rtol = 1e-6, 1e-4

            def site_to_mod(site: str) -> "GrowableLoRALinear":
                if site == "sem":
                    return base.semProj[0]
                if site == "cons_self":
                    return base.consSelfProj
                if site == "cons_intent":
                    return base.consIntentProj
                if site == "cons_pair":
                    return base.consPairNet[0]
                if site == "cons_token_gate":
                    return base.consTokenGate[0]
                if site == "ocr_gate":
                    return base.fuse_ocr_gate[0]
                if site == "ext_gate":
                    return base.fuse_ext_gate[0]
                if site == "sym_k2h":
                    return base.symCtrl.k2h[0]
                if site == "sym_gain":
                    return base.symCtrl.gain_head[0]
                if site == "sym_tok":
                    return base.symCtrl.tok_head[0]
                if site == "sym_film":
                    return base.symCtrl.film_head[0]
                if site == "sym_ctx":
                    return base.symCtrl.ctx_proj[0]
                raise KeyError(f"Unknown site: {site}")

            for site in sites:
                exp = expected[site]
                mod = site_to_mod(site)

                got = mod.DeltaWeight()
                if got is None:
                    got = torch.zeros_like(exp)
                else:
                    got = got.to(device=exp.device, dtype=exp.dtype)

                if not torch.allclose(got, exp, atol=atol, rtol=rtol):
                    max_abs = (got - exp).abs().max().item()
                    raise AssertionError(f"{site} delta mismatch, max_abs={max_abs:.3e}")

            base.eval()
            wrapper.eval()

            self_chk, intent_chk, ocr_chk, ext_chk, _ = self.MakeDummyBatch(base, batch_size=5)
            with torch.no_grad():
                ib, sb, _ = base(self_chk, intent_chk, ocrTexts=ocr_chk, extTexts=ext_chk)
                iw, sw, _ = wrapper(self_chk, intent_chk, ocrTexts=ocr_chk, extTexts=ext_chk)

            max_abs_int = (ib - iw).abs().max().item()
            max_abs_sym = (sb - sw).abs().max().item()
            assert max_abs_int < 1e-6, f"post-commit intent mismatch: {max_abs_int:.3e}"
            assert max_abs_sym < 1e-6, f"post-commit symProbs mismatch: {max_abs_sym:.3e}"

            print("WrapperManualGrowTrainAndCommit passed.")
            return True

        except AssertionError as e:
            print("WrapperManualGrowTrainAndCommit failed:", e)
            return False
        except Exception as e:
            print("WrapperManualGrowTrainAndCommit error:", e)
            return False

    def TextTrustPolicy(self) -> bool:
        try:
            model = self.MakeTestModel()
            model.eval()
            selfState, intentState, ocrTexts, extTexts, _ = self.MakeDummyBatch(
                model,
                batch_size=2,
                with_ocr=True,
                with_ext=True,
                with_cons=True,
                compact_text=True)
            with torch.no_grad():
                base_intent, _, _ = model(
                    selfState,
                    intentState,
                    ocrTexts=None,
                    extTexts=None,
                    prioritizeExt=True)
                unsafe_intent, _, unsafe_extras = model(
                    selfState,
                    intentState,
                    ocrTexts=None,
                    extTexts=extTexts,
                    prioritizeExt=True,
                    textTrust=[TEXT_TRUST_UNSAFE_EXTERNAL for _ in range(2)])
                default_intent, _, default_extras = model(
                    selfState,
                    intentState,
                    ocrTexts=None,
                    extTexts=extTexts,
                    prioritizeExt=True)
                operator_intent, _, operator_extras = model(
                    selfState,
                    intentState,
                    ocrTexts=ocrTexts,
                    extTexts=extTexts,
                    prioritizeExt=True,
                    textTrust=[TEXT_TRUST_OPERATOR_COMMAND for _ in range(2)])

            unsafe_delta = float((unsafe_intent - base_intent).abs().max().item())
            operator_delta = float((operator_intent - base_intent).abs().max().item())
            assert unsafe_delta < 1e-6, f"unsafe external changed control branch by {unsafe_delta:.3e}"
            assert torch.allclose(default_intent, unsafe_intent)
            assert operator_delta > 1e-7, "operator_command did not affect intent"
            assert bool(operator_extras["ext_control_mask"].all().item())
            assert not bool(unsafe_extras["ext_control_mask"].any().item())
            assert not bool(default_extras["ext_control_mask"].any().item())
            assert float(unsafe_extras["sem_ext_observed"].norm().item()) > 0.0
            assert torch.allclose(
                unsafe_extras["sem_ext_observed"],
                default_extras["sem_ext_observed"])
            assert torch.allclose(
                unsafe_extras["sem_ext_observed"],
                operator_extras["sem_ext_observed"])
            assert int(torch.count_nonzero(unsafe_extras["sem_ext_controlled"]).item()) == 0
            assert torch.allclose(
                operator_extras["sem_ext_controlled"],
                operator_extras["sem_ext_observed"])
            assert float(operator_extras["ocr_control_weight"].max().item()) < 1.0
            print("TextTrustPolicy passed.")
            return True
        except AssertionError as e:
            print("TextTrustPolicy failed:", e)
            return False
        except Exception as e:
            print("TextTrustPolicy error:", e)
            return False

    def RecallConditionAndTrust(self) -> bool:
        try:
            B = 2
            model = self.MakeTestModel()
            model.train()
            self_state = torch.randn(B, int(model.cons_self_dim), device=self.device)
            intent_state = torch.randn(B, int(model.cons_intent_dim), device=self.device)
            ocr_texts = [["ab"], ["a"]]
            unsafe_ext = ["cd", "d"]
            unsafe_trust = [TEXT_TRUST_UNSAFE_EXTERNAL for _ in range(B)]

            stale_targets = torch.ones(B, 1, model.max_seq_len, dtype=torch.long, device=self.device)
            stale_logits = torch.zeros(
                B, 1, model.max_seq_len, model.vocab_size, device=self.device)
            stale_hidden = torch.zeros(
                B, 1, model.max_seq_len, model.dimSem, device=self.device)
            stale_valid = torch.ones(B, 1, dtype=torch.bool, device=self.device)
            model.CacheRecallState(
                recallLogits=stale_logits,
                recallHidden=stale_hidden,
                recallTargets=stale_targets,
                recallValid=stale_valid,
                consSem=torch.zeros(B, 1, model.dimSem, device=self.device),)
            empty_intent, empty_probs, empty_extras = model(
                None,
                None,
                ocrTexts=None,
                extTexts=unsafe_ext,
                textTrust=unsafe_trust,)
            assert empty_intent is None and empty_probs is None
            assert "sem_ext_controlled" not in empty_extras
            empty_expected_texts = model.BuildRecallTexts(
                B,
                ocrTexts=None,
                extTexts=unsafe_ext,)
            empty_expected_targets = model.TokenizeBatch(
                empty_expected_texts,
                device=self.device,
                stride=model.max_seq_len,
                appendEos=True,)
            assert torch.equal(empty_extras["recall_targets"], empty_expected_targets)
            assert torch.equal(model._last_recall_targets, empty_expected_targets)
            assert model._last_recall_logits is not None
            assert model._last_recall_hidden is not None
            assert model._last_recall_valid is not None
            assert bool(model._last_recall_valid.any().item())
            assert model._last_recall_cons_sem is not None
            assert float(empty_extras["sem_ext_observed"].norm().item()) > 0.0

            intent_sem, sym_probs, extras = model(
                self_state,
                intent_state,
                ocrTexts=ocr_texts,
                extTexts=unsafe_ext,
                textTrust=unsafe_trust,)
            assert intent_sem is not None and sym_probs is not None

            expected_texts = model.BuildRecallTexts(
                B,
                ocrTexts=ocr_texts,
                extTexts=unsafe_ext,)
            expected_targets = model.TokenizeBatch(
                expected_texts,
                device=self.device,
                stride=model.max_seq_len,
                appendEos=True,)
            assert torch.equal(extras["recall_targets"], expected_targets)
            assert model._last_recall_cons_sem is not None
            recall_sem, recall_valid = model.BuildRecallSemanticFromTexts(
                semOcr=extras["sem_ocr_raw"],
                hasOcrMask=extras["has_ocr_mask"],
                semExt=extras["sem_ext_observed"],
                hasExtMask=extras["has_ext_mask"])
            _, _, recall_summary = model.NormalizeRecallMemory(
                recallSem=recall_sem,
                recallSemValid=recall_valid,
                batchSize=B,
                device=self.device,
                dtype=model.dtype,)
            expected_condition = recall_summary.detach().unsqueeze(1).expand(
                B,
                expected_targets.size(1),
                model.dimSem,)
            assert torch.allclose(model._last_recall_cons_sem, expected_condition)

            def text_encoder_grad_norm(trust: str) -> float:
                probe = self.MakeTestModel()
                probe.train()
                probe.zero_grad(set_to_none=True)
                _, probe_probs, _ = probe(
                    self_state,
                    intent_state,
                    ocrTexts=None,
                    extTexts=["cd", "cd"],
                    textTrust=[trust for _ in range(B)],)
                assert probe_probs is not None
                probe_loss, _ = probe.GetInternalLoss(probe_probs)
                probe_loss.backward()
                grad = probe.encoder.embedding.weight.grad
                return 0.0 if grad is None else float(grad.norm().item())

            unsafe_grad = text_encoder_grad_norm(TEXT_TRUST_UNSAFE_EXTERNAL)
            trusted_grad = text_encoder_grad_norm(TEXT_TRUST_OPERATOR_COMMAND)
            assert unsafe_grad > 1e-10, "observed unsafe text did not train recall encoder"
            assert trusted_grad > 1e-10, "trusted operator recall did not reach text encoder"

            print("RecallConditionAndTrust passed.")
            return True
        except AssertionError as e:
            print("RecallConditionAndTrust failed:", e)
            return False
        except Exception as e:
            print("RecallConditionAndTrust error:", e)
            return False

    def RecallLossMasksBeforeComputation(self) -> bool:
        try:
            model = self.MakeTestModel()
            model.train()

            B, N, T = 2, 2, 3
            targets = torch.full(
                (B, N, T),
                model.pad_idx,
                dtype=torch.long,
                device=self.device,)
            targets[0, 0, 0] = 1
            targets[0, 0, 1] = model.eos_idx
            targets[1, 1, 0] = 2
            targets[0, 1, 0] = 1
            targets[1, 0, 0] = 2
            valid_seq = torch.tensor(
                [[True, False], [False, True]],
                dtype=torch.bool,
                device=self.device,)
            token_valid = targets.ne(model.pad_idx) & valid_seq.unsqueeze(-1)

            logits_data = torch.randn(
                B, N, T, model.vocab_size, device=self.device)
            hidden_data = torch.randn(B, N, T, model.dimSem, device=self.device)
            logits_data[~token_valid] = float("inf")
            hidden_data[~token_valid] = float("inf")
            logits = logits_data.requires_grad_()
            hidden = hidden_data.requires_grad_()
            cons_sem = torch.randn(B, N, model.dimSem, device=self.device)

            model.CacheRecallState(
                recallLogits=logits,
                recallHidden=hidden,
                recallTargets=targets,
                recallValid=valid_seq,
                consSem=cons_sem,)
            loss, stats = model.GetRecallLoss()

            selected_hidden = torch.stack([
                hidden[0, 0, :2].mean(dim=0),
                hidden[1, 1, :1].mean(dim=0),])
            selected_cons = torch.stack([cons_sem[0, 0], cons_sem[1, 1]])
            expected_ce = F.cross_entropy(logits[token_valid], targets[token_valid])
            expected_align = (
                1.0 - F.cosine_similarity(selected_hidden, selected_cons, dim=-1)
                ).mean()
            expected = (
                model.lossLambdaRecallCE * expected_ce
                + model.lossLambdaRecallAlign * expected_align)

            assert bool(torch.isfinite(loss).item())
            assert torch.allclose(stats["recall_loss_ce"], expected_ce.detach())
            assert torch.allclose(stats["recall_loss_align"], expected_align.detach())
            assert torch.allclose(loss, expected)

            loss.backward()
            assert logits.grad is not None and hidden.grad is not None
            assert bool(torch.isfinite(logits.grad).all().item())
            assert bool(torch.isfinite(hidden.grad).all().item())
            assert int(torch.count_nonzero(logits.grad[~token_valid]).item()) == 0
            assert int(torch.count_nonzero(hidden.grad[~token_valid]).item()) == 0

            print("RecallLossMasksBeforeComputation passed.")
            return True
        except AssertionError as e:
            print("RecallLossMasksBeforeComputation failed:", e)
            return False
        except Exception as e:
            print("RecallLossMasksBeforeComputation error:", e)
            return False


    def RunAll(self) -> Dict[str, bool]:
        results = {
            "IntentionExtractorIOShapes": self.TestIntentionExtractorIOShapes(),
            "ForwardVariants": self.ForwardVariants(),
            "TrainStepSmokeBase": self.TrainStepSmokeBase(),
            "AllTrainableGradCoverageBase": self.AllTrainableGradCoverageBase(),
            "BranchGradientCoverageBase": self.BranchGradientCoverageBase(),
            "LossComponentsDecreaseBase": self.LossComponentsDecreaseBase(),
            "NormalTrainingConvergenceBase": self.NormalTrainingConvergenceBase(),
            "WrapperForwardEqualWhenNoInitRank": self.WrapperForwardEqualWhenNoInitRank(),
            "WrapperAPIBasics": self.WrapperAPIBasics(),
            "WrapperKeepsBaseEval": self.WrapperKeepsBaseEval(),
            "WrapperCandGradSmoke": self.WrapperCandGradSmoke(),
            "WrapperCandidateConvergence": self.WrapperCandidateConvergence(),
            "WrapperManualGrowTrainAndCommit": self.WrapperManualGrowTrainAndCommit(),
            "TextTrustPolicy": self.TextTrustPolicy(),
            "RecallConditionAndTrust": self.RecallConditionAndTrust(),
            "RecallLossMasksBeforeComputation": self.RecallLossMasksBeforeComputation(),}
        
        passed = sum(1 for v in results.values() if v)
        print(f"\nIntention module tests (with wrapper): {passed}/{len(results)} passed.")
        return results
