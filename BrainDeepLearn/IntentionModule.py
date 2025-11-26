from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
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

        self.rnn = nn.GRU(
            input_size=dimEmbed,
            hidden_size=dimHidden,
            num_layers=numLayers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if numLayers > 1 else 0.0,)

        self.att_proj = nn.Linear(dimHidden * 2, 1)
        self.out_dim = dimHidden * 2

    def forward(self, tokenIds: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len = tokenIds.shape

        emb = self.embedding(tokenIds) 

        mask = (tokenIds != self.padding_idx) 
        lengths = mask.long().sum(dim=1) 
        lengths_clamped = lengths.clamp(min=1)
        lengths_cpu = lengths_clamped.detach().cpu()

        packed = nn.utils.rnn.pack_padded_sequence(emb,lengths_cpu,batch_first=True,enforce_sorted=False,)

        out_packed, _ = self.rnn(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True, total_length=seq_len,) 

        scores = self.att_proj(out).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))

        no_token = ~mask.any(dim=1)

        if no_token.any():
            scores = scores.clone()
            scores[no_token] = 0.0

        attn = F.softmax(scores, dim=-1) 

        if no_token.any():
            attn = attn.clone()
            attn[no_token] = 0.0

        attn_exp = attn.unsqueeze(-1) 
        text_repr = (out * attn_exp).sum(dim=1)  

        return text_repr



class LangSymbolReasoner(nn.Module):
    def __init__(
        self,
        nSymbols: int,
        dimSem: int,
        hiddenDim: int = 512,
        alphaImp: float = 1.0,
        alphaCooc: float = 0.5,
        alphaContr: float = 1.0,):
        super().__init__()
        self.nSymbols = int(nSymbols)
        self.dimSem = int(dimSem)

        self.relImp = nn.Parameter(torch.randn(self.dimSem, self.dimSem) * 0.02)
        self.relContr = nn.Parameter(torch.randn(self.dimSem, self.dimSem) * 0.02)
        self.relCooc = nn.Parameter(torch.randn(self.dimSem, self.dimSem) * 0.02)

        self.alphaImp = nn.Parameter(torch.tensor(float(alphaImp)))
        self.alphaCooc = nn.Parameter(torch.tensor(float(alphaCooc)))
        self.alphaContr = nn.Parameter(torch.tensor(float(alphaContr)))

        self.postMlp = nn.Sequential(
            nn.Linear(self.nSymbols * 2, hiddenDim),
            nn.LayerNorm(hiddenDim),
            nn.GELU(),
            nn.Linear(hiddenDim, self.nSymbols),)

    def BuildRelationMatrix(self, conceptEmb: torch.Tensor, relCore: torch.Tensor) -> torch.Tensor:
        norm_emb = F.normalize(conceptEmb, dim=-1, eps=1e-6) 
        interm = norm_emb @ relCore  
        relation_matrix = interm @ norm_emb.t() 
        return relation_matrix

    def forward(
        self,
        symLogits: torch.Tensor,
        conceptEmb: torch.Tensor,) -> torch.Tensor:
        batch_size, symbol_count = symLogits.shape
        assert symbol_count == self.nSymbols, "IntentionExtractor: nSymbols mismatch in LangSymbolReasoner."

        sym_probs0 = torch.sigmoid(symLogits)  # [B, K]

        w_imp = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relImp))
        w_contr = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relContr))
        w_cooc = torch.tanh(self.BuildRelationMatrix(conceptEmb, self.relCooc))

        support_imp = sym_probs0 @ w_imp
        support_cooc = sym_probs0 @ w_cooc
        support_contr = sym_probs0 @ w_contr

        combined_logits = (symLogits
            + torch.tanh(self.alphaImp) * support_imp
            + torch.tanh(self.alphaCooc) * support_cooc
            - torch.tanh(self.alphaContr) * support_contr)

        combined_probs = torch.sigmoid(combined_logits)
        mlp_input = torch.cat([sym_probs0, combined_probs], dim=-1)
        delta_logits = self.postMlp(mlp_input)

        final_logits = combined_logits + delta_logits
        sym_probs = torch.sigmoid(final_logits)

        return sym_probs

    def GetInternalLoss(
        self,
        conceptEmb: torch.Tensor,
        symProbs: torch.Tensor,
        lambdaSymmetry: float = 1e-3,
        lambdaAntiSymmetry: float = 1e-3,
        lambdaEntropy: float = 1e-3,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        w_contr = self.BuildRelationMatrix(conceptEmb, self.relContr)
        w_cooc = self.BuildRelationMatrix(conceptEmb, self.relCooc)

        anti_sym_part = 0.5 * (w_cooc - w_cooc.t())
        loss_symmetry = anti_sym_part.pow(2).mean() * lambdaSymmetry

        contr_sym_part = 0.5 * (w_contr + w_contr.t())
        loss_anti_symmetry = contr_sym_part.pow(2).mean() * lambdaAntiSymmetry

        eps = 1e-6
        p = symProbs.clamp(eps, 1.0 - eps)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        mean_entropy = entropy.mean()

        target_entropy = symProbs.new_tensor(0.5)
        loss_entropy = ((mean_entropy - target_entropy) ** 2) * lambdaEntropy

        total_loss = loss_symmetry + loss_anti_symmetry + loss_entropy

        stats: Dict[str, torch.Tensor] = {
            "reason_symmetry": loss_symmetry.detach(),
            "reason_antisymmetry": loss_anti_symmetry.detach(),
            "reason_entropy": loss_entropy.detach(),}

        return total_loss, stats


class IntentionExtractor(nn.Module):
    def __init__(
        self,
        *,
        vocabSize: int = 6624,
        paddingIdx: int = 0,
        maxSeqLen: int = 64,
        dimEmbed: int = 512,
        dimEncoderHidden: int = 512,
        numEncoderLayers: int = 3,
        encoderDropout: float = 0.1,
        dimSem: int = 512,
        consDim: int = 1024,
        nSymbols: int = 128,
        reasonerHiddenDim: int = 512,
        reasonerAlphaImp: float = 1.0,
        reasonerAlphaCooc: float = 0.5,
        reasonerAlphaContr: float = 1.0,
        lossLambdaSymmetry: float = 1e-3,
        lossLambdaAntiSymmetry: float = 1e-3,
        lossLambdaEntropy: float = 1e-3,
        ocrDictPath: Optional[str] = "/home/yhl/Documents/Intelligent-Robot-System/BrainDeepLearn/ModuleSetting/OCRKeys.txt",):
        super().__init__()

        self.pad_idx = int(paddingIdx)
        self.max_seq_len = int(maxSeqLen)

        self.ch2id: Dict[str, int] = {}
        self.id2ch: List[str] = []

        if ocrDictPath is not None:
            try:
                self.LoadOcrDict(ocrDictPath)
            except FileNotFoundError:
                self.ch2id = {}
                self.id2ch = []

        if self.id2ch:
            self.vocab_size = len(self.id2ch) + 1
        else:
            self.vocab_size = int(vocabSize)

        self.encoder = TextEncoder(
            vocabSize=self.vocab_size,
            dimEmbed=dimEmbed,
            dimHidden=dimEncoderHidden,
            numLayers=numEncoderLayers,
            dropout=encoderDropout,
            paddingIdx=self.pad_idx,)

        encoder_out_dim = self.encoder.out_dim

        self.semProj = nn.Sequential(
            nn.Linear(encoder_out_dim, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),)

        self.dimSem = int(dimSem)

        self.consProj = nn.Linear(consDim, dimSem)
        self.consNorm = nn.LayerNorm(dimSem)

        self.conceptEmb = nn.Parameter(torch.randn(nSymbols, dimSem) * 0.02)
        self.conceptBias = nn.Parameter(torch.zeros(nSymbols))

        self.reasoner = LangSymbolReasoner(
            nSymbols=nSymbols,
            dimSem=dimSem,
            hiddenDim=reasonerHiddenDim,
            alphaImp=reasonerAlphaImp,
            alphaCooc=reasonerAlphaCooc,
            alphaContr=reasonerAlphaContr,)

        self.lossLambdaSymmetry = float(lossLambdaSymmetry)
        self.lossLambdaAntiSymmetry = float(lossLambdaAntiSymmetry)
        self.lossLambdaEntropy = float(lossLambdaEntropy)

        fuse_ocr_in = dimSem * 7
        self.fuse_ocr_gate = nn.Sequential(
            nn.Linear(fuse_ocr_in, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, 1),
            nn.Sigmoid(),)

        fuse_ext_in = dimSem * 4
        self.fuse_ext_gate = nn.Sequential(
            nn.Linear(fuse_ext_in, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, 1),
            nn.Sigmoid(),)

        self.beta_ocr = nn.Parameter(torch.tensor(0.1))
        self.beta_ext = nn.Parameter(torch.tensor(0.1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dimSem,
            nhead=8,
            dim_feedforward=dimSem * 4,
            dropout=encoderDropout,
            batch_first=True,
            activation="gelu",)
        
        self.intentTransformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.beta_trans = nn.Parameter(torch.tensor(0.3))

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

    def TokenizeBatch(self, texts: List[str], device: torch.device) -> torch.Tensor:
        batch_size = len(texts)
        tokens = torch.full(
            (batch_size, self.max_seq_len),
            self.pad_idx,
            dtype=torch.long,
            device=device,)

        if self.ch2id:
            for i, s in enumerate(texts):
                if s is None:
                    continue
                s = str(s).strip()
                if not s:
                    continue

                pos = 0
                for ch in s:
                    if pos >= self.max_seq_len:
                        break
                    ch_id = self.ch2id.get(ch, None)
                    if ch_id is None:
                        continue
                    tokens[i, pos] = ch_id
                    pos += 1
            return tokens

        for i, s in enumerate(texts):
            if s is None:
                continue
            s = str(s).strip().lower()
            if not s:
                continue
            pieces = s.split()
            for j, tok in enumerate(pieces[: self.max_seq_len]):
                h = hash(tok)
                idx = 1 + (abs(h) % (self.vocab_size - 1))
                tokens[i, j] = idx

        return tokens

    def EncodeStrings(
        self,
        texts: List[str],
        device: torch.device,) -> torch.Tensor:
        batch_size = len(texts)
        _ = batch_size

        valid = []
        normed = []
        for s in texts:
            if s is None:
                normed.append("")
                valid.append(False)
            else:
                s2 = str(s).strip()
                normed.append(s2)
                valid.append(len(s2) > 0)

        token_ids = self.TokenizeBatch(normed, device=device)
        text_repr = self.encoder(token_ids)
        lang_sem = self.semProj(text_repr)

        mask_valid = torch.tensor(valid, dtype=torch.bool, device=device)
        lang_sem = lang_sem * mask_valid.unsqueeze(-1)

        return lang_sem

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

    def forward(
        self,
        consState: Optional[torch.Tensor],
        ocrTexts: Optional[List[List[str]]] = None,
        extTexts: Optional[List[Optional[str]]] = None,
        *,
        prioritizeExt: bool = False,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:

        device = self.conceptEmb.device

        batch_size: Optional[int] = None
        if consState is not None:
            consState = consState.to(device)
            batch_size = consState.size(0)

        if ocrTexts is not None:
            if batch_size is None:
                batch_size = len(ocrTexts)
            elif len(ocrTexts) != batch_size:
                raise ValueError(f"IntentionExtractor: batch mismatch, consState={batch_size}, ocrTexts={len(ocrTexts)}")

        if extTexts is not None:
            if batch_size is None:
                batch_size = len(extTexts)
            elif len(extTexts) != batch_size:
                raise ValueError( f"IntentionExtractor: batch mismatch, consState/ocr vs extTexts={len(extTexts)}")

        if batch_size is None:
            return None, None, {}

        cons_sem: Optional[torch.Tensor] = None
        if consState is not None:
            cons_sem = self.consNorm(self.consProj(consState))
            has_cons_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            has_cons_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if ocrTexts is not None:
            merged = self.MergeOcrTexts(ocrTexts)
            sem_ocr = self.EncodeStrings(merged, device=device)
            has_ocr_mask = sem_ocr.abs().sum(dim=-1).gt(0)
        else:
            sem_ocr = torch.zeros(batch_size, self.dimSem, device=device)
            has_ocr_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if extTexts is not None:
            normed = [("" if t is None else str(t)) for t in extTexts]
            sem_ext = self.EncodeStrings(normed, device=device)
            has_ext_mask = sem_ext.abs().sum(dim=-1).gt(0)
        else:
            sem_ext = torch.zeros(batch_size, self.dimSem, device=device)
            has_ext_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        extras: Dict[str, torch.Tensor] = {}

        if (cons_sem is None) and (not has_ocr_mask.any()) and (not has_ext_mask.any()):
            return None, None, extras

        if cons_sem is not None:
            base = cons_sem
            extras["cons_sem"] = cons_sem.detach()
        else:
            base = torch.zeros(batch_size, self.dimSem, device=device)

        ext_for_ocr = sem_ext 

        feat_ocr = torch.cat([
                base,
                sem_ocr,
                torch.abs(base - sem_ocr),
                base * sem_ocr,
                ext_for_ocr,
                torch.abs(ext_for_ocr - sem_ocr),
                ext_for_ocr * sem_ocr,],dim=-1,) 

        gate_ocr = self.fuse_ocr_gate(feat_ocr)
        sem_ocr_fused = gate_ocr * sem_ocr 

        ocr_mask_float = has_ocr_mask.unsqueeze(-1).float()
        base = base + self.beta_ocr * (sem_ocr_fused * ocr_mask_float)

        extras["sem_ocr_raw"] = sem_ocr.detach()
        extras["sem_ocr_fused"] = sem_ocr_fused.detach()
        extras["gate_ocr"] = gate_ocr.detach()
        extras["has_ocr_mask"] = has_ocr_mask.detach()

        feat_ext = torch.cat([
                base,
                sem_ext,
                torch.abs(base - sem_ext),
                base * sem_ext,],dim=-1,) 

        gate_ext = self.fuse_ext_gate(feat_ext)

        has_ext_mask_float = has_ext_mask.unsqueeze(-1).float()
        has_ext_mask_exp = has_ext_mask.unsqueeze(-1) 

        if prioritizeExt:
            gamma = 0.5 + 0.5 * gate_ext  
            candidate = (1.0 - gamma) * base + gamma * sem_ext
            intentSem = torch.where(has_ext_mask_exp, candidate, base)

            extras["gamma_ext"] = gamma.detach()
        else:
            sem_ext_fused = gate_ext * sem_ext
            intentSem = base + self.beta_ext * (sem_ext_fused * has_ext_mask_float)
            extras["sem_ext_fused"] = sem_ext_fused.detach()

        extras["sem_ext_raw"] = sem_ext.detach()
        extras["gate_ext"] = gate_ext.detach()
        extras["has_ext_mask"] = has_ext_mask.detach()

        if cons_sem is not None:
            cons_token = cons_sem
        else:
            cons_token = torch.zeros(batch_size, self.dimSem, device=device)

        tokens = torch.stack([
                cons_token,
                sem_ocr,
                sem_ext,],dim=1,)

        token_mask = torch.stack([
                has_cons_mask,
                has_ocr_mask,
                has_ext_mask,], dim=1,) 

        if token_mask.any():
            src_key_padding_mask = ~token_mask 

            trans_out = self.intentTransformer(tokens,src_key_padding_mask=src_key_padding_mask,)

            mask_float = token_mask.float().unsqueeze(-1)  
            sum_vec = (trans_out * mask_float).sum(dim=1)
            denom = mask_float.sum(dim=1).clamp(min=1.0)
            fused = sum_vec / denom 

            intentSem = intentSem + self.beta_trans * fused

            extras["intent_trans_norm"] = fused.norm(dim=-1).detach()
            extras["intent_trans_mask_sum"] = mask_float.sum(dim=1).squeeze(-1).detach()

        symbol_logits = F.linear(intentSem, self.conceptEmb, self.conceptBias)
        symProbs = self.reasoner(symbol_logits, self.conceptEmb)

        return intentSem, symProbs, extras

    def GetInternalLoss(
        self,
        symProbs: torch.Tensor,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        total_loss, stats = self.reasoner.GetInternalLoss(
            conceptEmb=self.conceptEmb,
            symProbs=symProbs,
            lambdaSymmetry=self.lossLambdaSymmetry,
            lambdaAntiSymmetry=self.lossLambdaAntiSymmetry,
            lambdaEntropy=self.lossLambdaEntropy,)
        
        return total_loss, stats

