from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from FunctionTools import GetParameterSScale, SiteSpec, BaseOnlineWrapper

import torch
import torch.nn as nn
import torch.nn.functional as F



class IntentionLoRALinear(nn.Module):
    def __init__(self, targetLinear: nn.Linear):
        super().__init__()
        assert isinstance(targetLinear, nn.Linear)
        self.target = targetLinear

        self.A_list = nn.ParameterList()
        self.B_list = nn.ParameterList()
        self.alpha = nn.ParameterList() 

        self.out_f = int(targetLinear.out_features)
        self.in_f = int(targetLinear.in_features)

    @torch.no_grad()
    def Grow(self, addRank: int, init: dict = None, freezeOld: bool = True):
        if (addRank is None) or (addRank <= 0):
            return

        if init is None:
            init = {}

        dev = self.target.weight.device
        dt  = self.target.weight.dtype

        A = init.get("A", torch.randn(addRank, self.in_f, device=dev, dtype=dt) * 1e-4)
        B = init.get("B", torch.randn(self.out_f, addRank, device=dev, dtype=dt) * 1e-4)
        s = init.get("scale", 1e-3)

        A = nn.Parameter(A.contiguous().to(device=dev, dtype=dt))
        B = nn.Parameter(B.contiguous().to(device=dev, dtype=dt))
        s = nn.Parameter(torch.as_tensor(s, device=dev, dtype=dt))

        if freezeOld:
            for p in list(self.A_list) + list(self.B_list) + list(self.alpha):
                p.requires_grad_(False)

        self.A_list.append(A)
        self.B_list.append(B)
        self.alpha.append(s)

    def DeltaWeight(self) -> Optional[torch.Tensor]:
        if len(self.A_list) == 0:
            return None

        delta = self.target.weight.new_zeros(self.out_f, self.in_f)
        for A, B, s in zip(self.A_list, self.B_list, self.alpha):
            delta = delta + torch.tanh(s) * GetParameterSScale(s) * (B @ A)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.target.weight
        delta = self.DeltaWeight()
        if delta is not None:
            W = W + delta
        return F.linear(x, W, self.target.bias)



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
            IntentionLoRALinear(nn.Linear(encoder_out_dim, dimSem)),
            nn.LayerNorm(dimSem),
            nn.GELU(),)

        self.dimSem = int(dimSem)

        self.consProj = nn.Linear(consDim, dimSem)
        self.consProj = IntentionLoRALinear(self.consProj)
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
        
        ocr_fc1 = self.fuse_ocr_gate[0]
        self.fuse_ocr_gate[0] = IntentionLoRALinear(ocr_fc1)

        fuse_ext_in = dimSem * 4
        self.fuse_ext_gate = nn.Sequential(
            nn.Linear(fuse_ext_in, dimSem),
            nn.LayerNorm(dimSem),
            nn.GELU(),
            nn.Linear(dimSem, 1),
            nn.Sigmoid(),)
        
        ext_fc1 = self.fuse_ext_gate[0]
        self.fuse_ext_gate[0] = IntentionLoRALinear(ext_fc1)

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

    def EncodeStrings(self, texts: List[Optional[str]], device: torch.device) -> torch.Tensor:
        token_ids = self.TokenizeBatch(texts, device=device) 
        mask_valid = token_ids.ne(self.pad_idx).any(dim=1) 

        text_repr = self.encoder(token_ids)
        lang_sem = self.semProj(text_repr)
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

            extras["intent_trans_norm"] = fused.norm(dim=-1, keepdim=True).detach()
            extras["intent_trans_mask_sum"] = mask_float.sum(dim=1).detach()

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
        maxRankExt: int = 64,):
        self.maxRankSem = int(maxRankSem)
        self.maxRankCons = int(maxRankCons)
        self.maxRankOcr = int(maxRankOcr)
        self.maxRankExt = int(maxRankExt)
        super().__init__(base,initRankEach=initRankEach,autoRank=autoRank,evThreshold=evThreshold,gradEma=gradEma,)

    def BuildSiteSpecs(self) -> Dict[str, SiteSpec]:
        base: "IntentionExtractor" = self.base 

        sem_lora: "IntentionLoRALinear" = base.semProj[0]
        cons_lora: "IntentionLoRALinear" = base.consProj
        ocr_lora: "IntentionLoRALinear" = base.fuse_ocr_gate[0]
        ext_lora: "IntentionLoRALinear" = base.fuse_ext_gate[0]

        def make_alloc(inDim: int, outDim: int, maxRank: int):
            def alloc(addRank: int, device: torch.device, dtype: torch.dtype):
                A = nn.Parameter(torch.randn(addRank, inDim, device=device, dtype=dtype) * 1e-4)
                B = nn.Parameter(torch.zeros(outDim, addRank, device=device, dtype=dtype) * 1e-4)
                s = nn.Parameter(torch.tensor(1e-3, device=device, dtype=dtype))
                return A, B, s
            return alloc

        def compose(a: torch.Tensor, b: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return torch.tanh(s) * GetParameterSScale(s) * (b @ a)

        specs = {
            "sem": SiteSpec(
                name="sem",
                nLayers=1,
                inDim=int(sem_lora.in_f),
                outDim=int(sem_lora.out_f),
                maxRank=self.maxRankSem,
                allocFn=make_alloc(int(sem_lora.in_f), int(sem_lora.out_f), self.maxRankSem),
                composeFn=compose,),

            "cons": SiteSpec(
                name="cons",
                nLayers=1,
                inDim=int(cons_lora.in_f),
                outDim=int(cons_lora.out_f),
                maxRank=self.maxRankCons,
                allocFn=make_alloc(int(cons_lora.in_f), int(cons_lora.out_f), self.maxRankCons),
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
                composeFn=compose,),}
        return specs

    @staticmethod
    def LinearWithLora(
        mod: "IntentionLoRALinear",
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
        delta_sem: Optional[torch.Tensor],) -> torch.Tensor:

        normed: List[str] = []
        for s in texts:
            if s is None:
                normed.append("")
            else:
                normed.append(str(s))

        token_ids = base.TokenizeBatch(normed, device=device)

        mask_valid = token_ids.ne(base.pad_idx).any(dim=1) 

        text_repr = base.encoder(token_ids) 

        sem_lora: "IntentionLoRALinear" = base.semProj[0]
        h = self.LinearWithLora(sem_lora, text_repr, delta_sem)
        h = base.semProj[1](h)
        h = base.semProj[2](h)

        h = h * mask_valid.unsqueeze(-1)
        return h

    def GateWithDelta(
        self,
        gate_seq: nn.Sequential,
        x: torch.Tensor,
        delta_gate: Optional[torch.Tensor],) -> torch.Tensor:
        gate_lora: "IntentionLoRALinear" = gate_seq[0]

        h = self.LinearWithLora(gate_lora, x, delta_gate)
        h = gate_seq[1](h)
        h = gate_seq[2](h)
        h = gate_seq[3](h)
        h = gate_seq[4](h)
        return h

    def ForwardWithDeltas(
        self,
        x: Optional[torch.Tensor],
        keyPaddingMask: Optional[torch.Tensor] = None,
        tdError: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        deltasPerLayer: Optional[List[Dict[str, Optional[torch.Tensor]]]] = None,
        **kwargs,) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, torch.Tensor]]:

        base: "IntentionExtractor" = self.base 
        device = base.conceptEmb.device

        consState: Optional[torch.Tensor] = x
        ocrTexts: Optional[List[List[str]]] = kwargs.get("ocrTexts", None)
        extTexts: Optional[List[Optional[str]]] = kwargs.get("extTexts", None)
        prioritizeExt: bool = bool(kwargs.get("prioritizeExt", False))

        row = deltasPerLayer[0] if (deltasPerLayer is not None and len(deltasPerLayer) > 0) else {}
        delta_sem = row.get("sem", None)
        delta_cons = row.get("cons", None)
        delta_ocr = row.get("ocr_gate", None)
        delta_ext = row.get("ext_gate", None)

        batch_size: Optional[int] = None

        if consState is not None:
            batch_size = consState.size(0)

        if ocrTexts is not None:
            if batch_size is None:
                batch_size = len(ocrTexts)
            elif len(ocrTexts) != batch_size:
                raise ValueError(f"IntentionOnlineWrapper: batch mismatch, consState={batch_size}, ocrTexts={len(ocrTexts)}")

        if extTexts is not None:
            if batch_size is None:
                batch_size = len(extTexts)
            elif len(extTexts) != batch_size:
                raise ValueError(f"IntentionOnlineWrapper: batch mismatch, consState/ocr vs extTexts={len(extTexts)}")

        if batch_size is None:
            return None, None, {}

        dimSem = base.dimSem
        extras: Dict[str, torch.Tensor] = {}

        if consState is not None:
            cons_lora: "IntentionLoRALinear" = base.consProj
            cons_lin = self.LinearWithLora(cons_lora, consState, delta_cons)
            cons_sem = base.consNorm(cons_lin)
            has_cons_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            extras["cons_sem"] = cons_sem.detach()
        else:
            cons_sem = None
            has_cons_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if ocrTexts is not None:
            merged = base.MergeOcrTexts(ocrTexts)
            sem_ocr = self.EncodeStringsWithDelta(base, merged, device=device, delta_sem=delta_sem)
            has_ocr_mask = sem_ocr.abs().sum(dim=-1).gt(0)
        else:
            sem_ocr = torch.zeros(batch_size, dimSem, device=device)
            has_ocr_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if extTexts is not None:
            normed = [("" if t is None else str(t)) for t in extTexts]
            sem_ext = self.EncodeStringsWithDelta(base, normed, device=device, delta_sem=delta_sem)
            has_ext_mask = sem_ext.abs().sum(dim=-1).gt(0)
        else:
            sem_ext = torch.zeros(batch_size, dimSem, device=device)
            has_ext_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if (cons_sem is None) and (not has_ocr_mask.any()) and (not has_ext_mask.any()):
            return None, None, extras

        if cons_sem is not None:
            base_vec = cons_sem
        else:
            base_vec = torch.zeros(batch_size, dimSem, device=device)

        ext_for_ocr = sem_ext

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
        base_vec = base_vec + base.beta_ocr * (sem_ocr_fused * ocr_mask_float)

        extras["sem_ocr_raw"] = sem_ocr.detach()
        extras["sem_ocr_fused"] = sem_ocr_fused.detach()
        extras["gate_ocr"] = gate_ocr.detach()
        extras["has_ocr_mask"] = has_ocr_mask.detach()

        feat_ext = torch.cat([
                base_vec,
                sem_ext,
                torch.abs(base_vec - sem_ext),
                base_vec * sem_ext,],dim=-1,)

        gate_ext = self.GateWithDelta(base.fuse_ext_gate, feat_ext, delta_ext)

        has_ext_mask_float = has_ext_mask.unsqueeze(-1).float()
        has_ext_mask_exp = has_ext_mask.unsqueeze(-1)

        if prioritizeExt:
            gamma = 0.5 + 0.5 * gate_ext
            candidate = (1.0 - gamma) * base_vec + gamma * sem_ext
            intentSem = torch.where(has_ext_mask_exp, candidate, base_vec)
            extras["gamma_ext"] = gamma.detach()
        else:
            sem_ext_fused = gate_ext * sem_ext
            intentSem = base_vec + base.beta_ext * (sem_ext_fused * has_ext_mask_float)
            extras["sem_ext_fused"] = sem_ext_fused.detach()

        extras["sem_ext_raw"] = sem_ext.detach()
        extras["gate_ext"] = gate_ext.detach()
        extras["has_ext_mask"] = has_ext_mask.detach()

        if cons_sem is not None:
            cons_token = cons_sem
        else:
            cons_token = torch.zeros(batch_size, dimSem, device=device)

        tokens = torch.stack([
                cons_token,
                sem_ocr,
                sem_ext,],dim=1,) 

        token_mask = torch.stack([
                has_cons_mask,
                has_ocr_mask,
                has_ext_mask,],dim=1,)

        if token_mask.any():
            src_key_padding_mask = ~token_mask

            trans_out = base.intentTransformer(tokens, src_key_padding_mask=src_key_padding_mask)

            mask_float = token_mask.float().unsqueeze(-1) 
            sum_vec = (trans_out * mask_float).sum(dim=1)
            denom = mask_float.sum(dim=1).clamp(min=1.0)
            fused = sum_vec / denom

            intentSem = intentSem + base.beta_trans * fused

            extras["intent_trans_norm"] = fused.norm(dim=-1, keepdim=True).detach() 
            extras["intent_trans_mask_sum"] = mask_float.sum(dim=1).detach()

        symbol_logits = F.linear(intentSem, base.conceptEmb, base.conceptBias)
        symProbs = base.reasoner(symbol_logits, base.conceptEmb)

        return intentSem, symProbs, extras

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
        elif site == "cons":
            base.consProj.Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "ocr_gate":
            base.fuse_ocr_gate[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
        elif site == "ext_gate":
            base.fuse_ext_gate[0].Grow(addRank=a.size(0), init=init, freezeOld=self.freezeOldPar)
            return True
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


    def MakeDummyBatch(
        self,
        model: "IntentionExtractor",
        batch_size: int = 8,
        with_ocr: bool = True,
        with_ext: bool = True,
        with_cons: bool = True,) -> Tuple[Optional[torch.Tensor], Optional[List[List[str]]], Optional[List[Optional[str]]], torch.Tensor]:
        nSymbols = int(model.conceptEmb.size(0))
        consState = None
        ocrTexts: Optional[List[List[str]]] = None
        extTexts: Optional[List[Optional[str]]] = None

        if with_cons:
            consDim = int(model.consProj.in_f)
            consState = torch.randn(batch_size, consDim, device=self.device)

        if with_ocr:
            ocrTexts = [[f"ocr text {i} sample"] for i in range(batch_size)]
        if with_ext:
            extTexts = [f"external hint {i}" for i in range(batch_size)]

        targetSym = torch.randint(low=0,high=2,size=(batch_size, nSymbols),dtype=torch.float32,device=self.device,)
        return consState, ocrTexts, extTexts, targetSym

    def GradCoverage(
        self,
        named: Dict[str, torch.nn.Parameter],
        min_ratio: float,
        must_have: List[str],) -> bool:
        total_trainable = sum(1 for p in named.values() if p.requires_grad)
        total_with_grad = sum(1 for p in named.values() if p.requires_grad and (p.grad is not None))
        ratio = total_with_grad / max(1, total_trainable)

        missing = [n for n in must_have if n in named and (named[n].grad is None)]
        if missing:
            print("Missing gradient parameters:", missing)
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
            consDim = int(model.consProj.in_f)

            B = 4

            cons_only = torch.randn(B, consDim, device=self.device)
            intentSem, symProbs, extras = model(cons_only, ocrTexts=None, extTexts=None)
            assert intentSem is not None and symProbs is not None
            assert intentSem.shape == (B, dimSem)
            assert symProbs.shape == (B, nSymbols)

            ocrTexts = [[f"hello {i}"] for i in range(B)]
            intentSem2, symProbs2, extras2 = model(consState=None, ocrTexts=ocrTexts, extTexts=None)
            assert intentSem2 is not None and symProbs2 is not None
            assert intentSem2.shape == (B, dimSem)
            assert symProbs2.shape == (B, nSymbols)

            extTexts = [f"world {i}" for i in range(B)]
            intentSem3, symProbs3, extras3 = model(consState=None, ocrTexts=None, extTexts=extTexts)
            assert intentSem3 is not None and symProbs3 is not None
            assert intentSem3.shape == (B, dimSem)
            assert symProbs3.shape == (B, nSymbols)

            cons_full = torch.randn(B, consDim, device=self.device)
            ocr_full = [[f"ocr {i} text"] for i in range(B)]
            ext_full = [f"ext {i} text" for i in range(B)]

            intentSem4, symProbs4, extras4 = model(cons_full, ocrTexts=ocr_full, extTexts=ext_full, prioritizeExt=True)
            assert intentSem4 is not None and symProbs4 is not None
            assert intentSem4.shape == (B, dimSem)
            assert symProbs4.shape == (B, nSymbols)

            for t in (symProbs, symProbs2, symProbs3, symProbs4):
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

            consState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(model, batch_size=8)

            intentSem, symProbs, extras = model(consState, ocrTexts=ocrTexts, extTexts=extTexts)
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
                "encoder.rnn.weight_ih_l0",
                "encoder.att_proj.weight",
                "semProj.0.target.weight",
                "consProj.target.weight",
                "fuse_ocr_gate.0.target.weight",
                "fuse_ext_gate.0.target.weight",
                "conceptEmb",
                "reasoner.relImp",
                "reasoner.relCooc",
                "reasoner.relContr",]

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

    def NormalTrainingConvergenceBase(self, steps: int = 80, logEvery: int = 20) -> bool:
        try:
            model = IntentionExtractor().to(self.device)
            model.train()

            consState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(model, batch_size=12)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)

            with torch.no_grad():
                _, sym0, _ = model(consState, ocrTexts=ocrTexts, extTexts=extTexts)
                start_main = F.binary_cross_entropy(sym0, targetSym).item()
                internal0, _ = model.GetInternalLoss(sym0)
                start = start_main + 0.1 * internal0.item()

            hist = []
            for t in range(1, steps + 1):
                intentSem, symProbs, extras = model(consState, ocrTexts=ocrTexts, extTexts=extTexts)
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
            base = IntentionExtractor().to(self.device)
            base.eval()
            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.eval()

            consState, ocrTexts, extTexts, _ = self.MakeDummyBatch(base, batch_size=5)

            with torch.no_grad():
                y_base = base(consState, ocrTexts=ocrTexts, extTexts=extTexts, prioritizeExt=True)
                y_wrap = wrapper(consState, ocrTexts=ocrTexts, extTexts=extTexts, prioritizeExt=True)

            intent_base, sym_base, _ = y_base
            intent_wrap, sym_wrap, _ = y_wrap

            assert intent_base is not None and intent_wrap is not None
            assert sym_base is not None and sym_wrap is not None

            max_abs_int = (intent_base - intent_wrap).abs().max().item()
            max_abs_sym = (sym_base - sym_wrap).abs().max().item()
            assert max_abs_int < 1e-6, f"intent mismatch: {max_abs_int:.3e}"
            assert max_abs_sym < 1e-6, f"symProbs mismatch: {max_abs_sym:.3e}"

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
                assert row["sem"] == 0 and row["cons"] == 0 and row["ocr_gate"] == 0 and row["ext_gate"] == 0

            wrapper.Update("grow", growFactor=2.0, addEach=2)
            r1 = wrapper.Update("ranks")["ranks"]
            sum_sem = r1["sum"]["sem"]
            sum_cons = r1["sum"]["cons"]
            sum_ocr = r1["sum"]["ocr_gate"]
            sum_ext = r1["sum"]["ext_gate"]
            assert sum_sem > 0 and sum_cons > 0 and sum_ocr > 0 and sum_ext > 0

            wrapper.Update("rollback")
            r2 = wrapper.Update("ranks")["ranks"]
            for row in r2["perLayer"]:
                assert row["sem"] == 0 and row["cons"] == 0 and row["ocr_gate"] == 0 and row["ext_gate"] == 0

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

            consState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(base, batch_size=6)

            opt = torch.optim.Adam(list(wrapper.CandParameters()), lr=3e-3)

            dummy_td = torch.ones(consState.size(0), 1, device=self.device)

            intentSem, symProbs, extras = wrapper(consState, ocrTexts=ocrTexts, extTexts=extTexts, prioritizeExt=False, tdError=dummy_td)
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
        
    def WrapperManualGrowTrainAndCommit(self) -> bool:
        try:
            base = IntentionExtractor().to(self.device)
            base.eval()

            wrapper = IntentionOnlineWrapper(base=base, initRankEach=0).to(self.device)
            wrapper.train()

            wrapper.Update("grow", growFactor=1.0, addEach=4)

            consState, ocrTexts, extTexts, targetSym = self.MakeDummyBatch(base, batch_size=8)
            opt = torch.optim.Adam(list(wrapper.CandParameters()), lr=3e-3)

            steps = 10

            dummy_td = torch.ones(consState.size(0), 1, device=self.device)
            for _ in range(steps):
                intentSem, symProbs, extras = wrapper(consState, ocrTexts=ocrTexts, extTexts=extTexts, prioritizeExt=False,tdError=dummy_td)
                loss = F.binary_cross_entropy(symProbs, targetSym)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                wrapper.Update("accumulategrads")
                opt.step()

            sites = ["sem", "cons", "ocr_gate", "ext_gate"]
            expected: Dict[str, torch.Tensor] = {}
            for site in sites:
                mat = wrapper.ComposeOne(site, layerIdx=0).detach().clone()
                expected[site] = mat

            res = wrapper.Update("commit")
            assert res["ok"], "Commit failed."
            stats = res.get("commit_stats", {})
            print(f"[IntentionCommit] committed_rank={stats.get('committed_rank', 0)}, "
                  f"triples={stats.get('committed_triples', 0)}")

            r_after = wrapper.Update("ranks")["ranks"]
            for row in r_after["perLayer"]:
                assert row["sem"] == 0 and row["cons"] == 0 and row["ocr_gate"] == 0 and row["ext_gate"] == 0

            atol, rtol = 1e-6, 1e-4

            def delta_from_lora(mod) -> torch.Tensor:
                dw = mod.DeltaWeight()
                if dw is None:
                    return torch.zeros_like(expected["sem"])
                return dw.to(expected["sem"].device, expected["sem"].dtype)

            exp_sem = expected["sem"]
            if not torch.allclose(exp_sem, torch.zeros_like(exp_sem)):
                got_sem = delta_from_lora(base.semProj[0])
                assert torch.allclose(got_sem, exp_sem, atol=atol, rtol=rtol), \
                    f"sem delta mismatch, max_abs={(got_sem - exp_sem).abs().max().item():.3e}"

            exp_cons = expected["cons"]
            if not torch.allclose(exp_cons, torch.zeros_like(exp_cons)):
                got_cons = delta_from_lora(base.consProj)
                assert torch.allclose(got_cons, exp_cons, atol=atol, rtol=rtol), \
                    f"cons delta mismatch, max_abs={(got_cons - exp_cons).abs().max().item():.3e}"

            exp_ocr = expected["ocr_gate"]
            if not torch.allclose(exp_ocr, torch.zeros_like(exp_ocr)):
                got_ocr = delta_from_lora(base.fuse_ocr_gate[0])
                assert torch.allclose(got_ocr, exp_ocr, atol=atol, rtol=rtol), \
                    f"ocr_gate delta mismatch, max_abs={(got_ocr - exp_ocr).abs().max().item():.3e}"

            exp_ext = expected["ext_gate"]
            if not torch.allclose(exp_ext, torch.zeros_like(exp_ext)):
                got_ext = delta_from_lora(base.fuse_ext_gate[0])
                assert torch.allclose(got_ext, exp_ext, atol=atol, rtol=rtol), \
                    f"ext_gate delta mismatch, max_abs={(got_ext - exp_ext).abs().max().item():.3e}"

            base.eval()
            wrapper.eval()
            cons_chk, ocr_chk, ext_chk, _ = self.MakeDummyBatch(base, batch_size=5)
            with torch.no_grad():
                ib, sb, _ = base(cons_chk, ocrTexts=ocr_chk, extTexts=ext_chk)
                iw, sw, _ = wrapper(cons_chk, ocrTexts=ocr_chk, extTexts=ext_chk)

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


    def RunAll(self) -> Dict[str, bool]:
        results = {
            "ForwardVariants": self.ForwardVariants(),
            "TrainStepSmokeBase": self.TrainStepSmokeBase(),
            "NormalTrainingConvergenceBase": self.NormalTrainingConvergenceBase(),
            "WrapperForwardEqualWhenNoInitRank": self.WrapperForwardEqualWhenNoInitRank(),
            "WrapperAPIBasics": self.WrapperAPIBasics(),
            "WrapperKeepsBaseEval": self.WrapperKeepsBaseEval(),
            "WrapperCandGradSmoke": self.WrapperCandGradSmoke(),
            "WrapperManualGrowTrainAndCommit": self.WrapperManualGrowTrainAndCommit(),}
        
        passed = sum(1 for v in results.values() if v)
        print(f"\nIntention module tests (with wrapper): {passed}/{len(results)} passed.")
        return results