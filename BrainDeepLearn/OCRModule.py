from __future__ import annotations
from typing import Any, List, Tuple, Dict, Optional, Union, TypedDict
from dataclasses import dataclass
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math


def DefaultOcrVocabPath() -> str:
    return str(Path(__file__).resolve().parent / "ModuleSetting" / "OCRKeys.txt")


def WidthToCtcSteps(validWidths: torch.Tensor) -> torch.Tensor:
    w = validWidths.to(torch.long)
    w = torch.div(w, 2, rounding_mode="floor")
    w = torch.div(w, 2, rounding_mode="floor")
    w = (w - 1).clamp(min=1)
    return w


def NormText(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    return s

def CharNgrams(s: str, n: int = 2) -> set:
    s = NormText(s)
    if not s:
        return set()
    if len(s) <= n:
        return {s}
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def Jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def TextSim(a: str, b: str) -> float:
    return Jaccard(CharNgrams(a, 2), CharNgrams(b, 2))

def IouXyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / max(1e-6, union)

@dataclass
class OcrLineObs:  
    box: tuple[int, int, int, int]  
    text: str
    det_score: float
    rec_conf: float
    step: int

    @property
    def weight(self) -> float:
        return max(0.0, self.det_score) * max(0.0, self.rec_conf)

@dataclass
class OcrTrack:
    obs: deque 
    age: int 


class OcrItem(TypedDict):
    box: Tuple[int, int, int, int]
    text: str
    det_score: float
    rec_conf: float
    score: float


class ConvBNReLU(nn.Module):
    def __init__(self, inCh: int, outCh: int, kSize: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(inCh, outCh, kernel_size=kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outCh)
        self.act = nn.ReLU(inplace=True)

    def forward(
        self, 
        inputTensor: torch.Tensor # [B, C, H, W]
        ) -> torch.Tensor:
        x = self.conv(inputTensor)
        x = self.bn(x)
        x = self.act(x)

        # H' ((H + 2*padding - kSize)/stride) + 1
        # W' ((W + 2*padding - kSize)/stride) + 1
        return x # [B, outCh, H', W']


class DBBackbone(nn.Module):
    def __init__(self, inCh: int = 3, baseCh: int = 64):
        super().__init__()
        c = baseCh

        self.enc1 = nn.Sequential(
            ConvBNReLU(inCh, c),
            ConvBNReLU(c, c),)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            ConvBNReLU(c, c * 2),
            ConvBNReLU(c * 2, c * 2),)
        
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            ConvBNReLU(c * 2, c * 4),
            ConvBNReLU(c * 4, c * 4),)
        
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = nn.Sequential(
            ConvBNReLU(c * 4, c * 8),
            ConvBNReLU(c * 8, c * 8),)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)

        self.dec3 = nn.Sequential(
            ConvBNReLU(c * 8, c * 4),
            ConvBNReLU(c * 4, c * 4),)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)

        self.dec2 = nn.Sequential(
            ConvBNReLU(c * 4, c * 2),
            ConvBNReLU(c * 2, c * 2),)

        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)

        self.dec1 = nn.Sequential(
            ConvBNReLU(c * 2, c),
            ConvBNReLU(c, c),)

        self.outConv = ConvBNReLU(c, 256, kSize=3, stride=1, padding=1)

    def forward(self, inputTensor: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(inputTensor) 
        p1 = self.pool1(e1)

        e2 = self.enc2(p1) 
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3) 

        u3 = self.up3(e4) 
        x3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(x3)

        u2 = self.up2(d3)
        x2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(x2)

        u1 = self.up1(d2) 
        x1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(x1)

        out = self.outConv(d1)
        return out # [B, 256, H, W]


class DBHead(nn.Module):
    def __init__(self, inCh: int = 256):
        super().__init__()

        self.probConv = nn.Sequential(
            ConvBNReLU(inCh, 64, kSize=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1),)
        self.residual = DBResidualLogitHead(inCh=inCh)

    def ForwardWithLogits(self, featTensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_prob_logits = self.probConv(featTensor)
        delta_prob_logits = self.residual(featTensor)
        prob_logits = base_prob_logits + delta_prob_logits
        prob_map = torch.sigmoid(prob_logits)
        return {
            "prob_logits": prob_logits,
            "prob_map": prob_map,
            "base_prob_logits": base_prob_logits,
            "delta_prob_logits": delta_prob_logits,}

    def forward(self, featTensor: torch.Tensor) -> torch.Tensor:
        prob_map = self.ForwardWithLogits(featTensor)["prob_map"]
        return prob_map # [B, 1, H, W]


class DBResidualLogitHead(nn.Module):
    def __init__(self, inCh: int = 256, midCh: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inCh, midCh, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, midCh),
            nn.ReLU(inplace=True),
            nn.Conv2d(midCh, 1, kernel_size=1, stride=1, padding=0, bias=True),)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, featTensor: torch.Tensor) -> torch.Tensor:

        scale = 1.0 + torch.tanh(self.alpha)
        return scale * self.block(featTensor)


def BalancedBceLoss(predTensor: torch.Tensor, gtTensor: torch.Tensor, maskTensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    eps = 1e-6
    pred = predTensor.clamp(eps, 1.0 - eps)
    gt = gtTensor

    if maskTensor is None:
        mask = torch.ones_like(gt)
    else:
        mask = maskTensor

    valid = (mask > 0.5).float()
    pos = (gt > 0.5).float() * valid
    neg = (1.0 - (gt > 0.5).float()) * valid

    n_pos = pos.sum().clamp(min=1.0)
    n_neg = neg.sum().clamp(min=1.0)

    w_pos = n_neg / (n_pos + n_neg)
    w_neg = n_pos / (n_pos + n_neg)

    loss_pos = -w_pos * (pos * torch.log(pred)) 
    loss_neg = -w_neg * (neg * torch.log(1.0 - pred))

    loss = (loss_pos + loss_neg)
    return loss.sum() / valid.sum().clamp(min=1.0)


def BalancedBceWithLogitsLoss(logitsTensor: torch.Tensor, gtTensor: torch.Tensor, maskTensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    gt = gtTensor.float()

    if maskTensor is None:
        mask = torch.ones_like(gt)
    else:
        mask = maskTensor

    valid = (mask > 0.5).float()
    pos = (gt > 0.5).float() * valid
    neg = (1.0 - (gt > 0.5).float()) * valid

    n_pos = pos.sum().clamp(min=1.0)
    n_neg = neg.sum().clamp(min=1.0)

    w_pos = n_neg / (n_pos + n_neg)
    w_neg = n_pos / (n_pos + n_neg)
    weight = w_pos * pos + w_neg * neg

    loss = F.binary_cross_entropy_with_logits(
        logitsTensor,
        gt,
        weight=weight,
        reduction="sum",)
    return loss / valid.sum().clamp(min=1.0)


def DiceLoss(predTensor: torch.Tensor,gtTensor: torch.Tensor, maskTensor: Optional[torch.Tensor] = None) -> torch.Tensor:
    if maskTensor is None:
        mask = torch.ones_like(gtTensor)
    else:
        mask = maskTensor

    pred = predTensor * mask
    gt = gtTensor * mask

    inter = (pred * gt).sum()
    union = pred.sum() + gt.sum() + 1e-6
    loss = 1.0 - 2.0 * inter / union
    return loss


class DBLoss(nn.Module):
    def __init__(
        self,
        lambdaProb: float = 1.0,
        lambdaMask: float = 1.0,):
        super().__init__()
        self.lambdaProb = float(lambdaProb)
        self.lambdaMask = float(lambdaMask)

    def forward(
        self,
        probMap: torch.Tensor,
        gtBoxes: torch.Tensor,
        gtMask: Optional[torch.Tensor] = None,
        probLogits: Optional[torch.Tensor] = None,
        gtTextMask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        target = gtBoxes if gtTextMask is None else gtTextMask

        if gtMask is None:
            gtMask = torch.ones_like(target)

        if probLogits is None:
            loss_prob = BalancedBceLoss(probMap, target, gtMask)
        else:
            loss_prob = BalancedBceWithLogitsLoss(probLogits, target, gtMask)
        loss_mask = DiceLoss(probMap, target, gtMask)

        total = (self.lambdaProb * loss_prob
            + self.lambdaMask * loss_mask)

        stats = {"loss_prob": loss_prob.detach(),
            "loss_mask": loss_mask.detach(),}

        return total, stats



class CRNNRecognizer(nn.Module):
    def __init__(
        self,
        imgH: int = 32,
        inCh: int = 1,
        nClasses: int = 96, 
        rnnHidden: int = 256,
        residualRank: int = 64,):
        super().__init__()
        self.imgH = int(imgH)
        self.nClasses = int(nClasses)

        self.conv1 = nn.Sequential(
            nn.Conv2d(inCh, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),)
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),)
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),)

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnnHidden,
            num_layers=2,
            bidirectional=True,
            batch_first=False,)

        self.fc = nn.Linear(rnnHidden * 2, self.nClasses)

        self.ctcLoss = nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)
        self.residualHead = LowRankCTCResidualHead(
            inDim=512,
            rank=int(residualRank),
            nClasses=self.nClasses,) if int(residualRank) > 0 else None

    def EncodeVisual(self, imgsTensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(imgsTensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x

    def FeaturesToSeq(self, featTensor: torch.Tensor) -> torch.Tensor:
        feat = featTensor
        b, c, h, w = feat.size()
        if h != 1:
            feat = F.adaptive_avg_pool2d(feat, (1, w))
            b, c, h, w = feat.size()
        feat = feat.squeeze(2)
        feat = feat.permute(2, 0, 1)
        return feat

    def forward(
        self,
        imgsTensor: torch.Tensor,
        targetsTensor: Optional[torch.Tensor] = None,
        targetLengths: Optional[torch.Tensor] = None,
        inputLengths: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:

        x = self.EncodeVisual(imgsTensor)
        seq = self.FeaturesToSeq(x) 

        rnn_out, _ = self.rnn(seq)
        base_logits = self.fc(rnn_out)
        delta_logits = self.residualHead(seq) if self.residualHead is not None else None
        logits = base_logits if delta_logits is None else (base_logits + delta_logits)
        log_probs = F.log_softmax(logits, dim=-1)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "log_probs": log_probs,
            "base_logits": base_logits,}
        if delta_logits is not None:
            out["delta_logits"] = delta_logits

        if targetsTensor is not None and targetLengths is not None:
            t, b, _ = log_probs.size()
            if inputLengths is None:
                input_lengths = torch.full(
                    size=(b,),
                    fill_value=t,
                    dtype=torch.long,
                    device=log_probs.device,)
            else:
                input_lengths = inputLengths.to(device=log_probs.device, dtype=torch.long).clamp(min=1, max=t)
            target_lengths = targetLengths.to(device=log_probs.device, dtype=torch.long)
            
            per_sample = self.ctcLoss(
                log_probs,
                targetsTensor,
                input_lengths,
                target_lengths,)
            valid = input_lengths >= target_lengths
            per_sample = per_sample / target_lengths.clamp_min(1).to(per_sample.dtype)
            loss = per_sample[valid].mean() if valid.any() else (per_sample.mean() * 0.0)
            
            out["loss"] = loss
            out["input_lengths"] = input_lengths
            out["ctc_invalid_ratio"] = (~valid).float().mean()

        return out


class LowRankCTCResidualHead(nn.Module):
    def __init__(self, inDim: int = 512, rank: int = 64, nClasses: int = 96, kernelSize: int = 5):
        super().__init__()
        self.norm = nn.LayerNorm(inDim)
        self.down = nn.Linear(inDim, rank, bias=False)
        self.dw = nn.Conv1d(
            rank,
            rank,
            kernel_size=kernelSize,
            padding=kernelSize // 2,
            groups=rank,
            bias=False,)
        self.out = nn.Linear(rank, nClasses, bias=True)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, seqTensor: torch.Tensor) -> torch.Tensor:
        z = self.norm(seqTensor)
        z = self.down(z)
        z = z.permute(1, 2, 0)
        z = F.gelu(self.dw(z))
        z = z.permute(2, 0, 1)
        scale = 1.0 + torch.tanh(self.alpha)
        return scale * self.out(z)


class OCREngineExtractor(nn.Module):
    def __init__(
        self,
        vocabCharsPath: str = DefaultOcrVocabPath(),
        *,
        temporalSteps: int = 0, 
        fuseTopK: int = 8,
        fuseMinHits: int = 2,
        fuseIouThr: float = 0.30,
        fuseSimThr: float = 0.75,
        fuseDecay: float = 0.85,):
        super().__init__()

        self.temporalSteps = int(temporalSteps)
        self.fuseTopK = int(fuseTopK)
        self.fuseMinHits = max(1, int(fuseMinHits))
        self.fuseIouThr = float(fuseIouThr)
        self.fuseSimThr = float(fuseSimThr)
        self.fuseDecay = float(fuseDecay)
        if self.temporalSteps > 0:
            self.fuseMinHits = min(self.fuseMinHits, self.temporalSteps)

        self._temporal_step = 0
        self._tracks_by_bi: dict[int, list[OcrTrack]] = {}
        self._last_batch_size: int = 0
        self._last_ocr_texts_batch: List[List[str]] = []

        self.backbone = DBBackbone(inCh=3, baseCh=64)
        self.dbHead = DBHead(inCh=256)
        self.dbLoss = DBLoss()

        self.blankIndex = 0
        vocabChars = self.LoadOcrVocabFromTxt(vocabCharsPath)
        self.idx2Char = ["<blank>"] + list(vocabChars)
        self.char2Idx = {c: i for i, c in enumerate(self.idx2Char)}

        self.recognizer = CRNNRecognizer(imgH=32,inCh=1, nClasses=len(self.idx2Char), rnnHidden=256,)


    def OcrMetadata(self) -> Dict[str, Any]:
        return {
            "vocab": list(self.idx2Char),
            "blank_index": int(self.blankIndex),
            "legacy_prefixes": [
                "backbone.",
                "dbHead.probConv.",
                "recognizer.conv",
                "recognizer.rnn.",
                "recognizer.fc.",],
            "addon_cfg": {
                "db_residual": self.dbHead.residual is not None,
                "rec_residual_rank": (
                    int(self.recognizer.residualHead.down.out_features)
                    if self.recognizer.residualHead is not None else 0),
                "width_aware_ctc": True,},}

    def LoadOcrVocabFromTxt(self, dictPath: str, *, encoding: str = "utf-8") -> str:
        chars: List[str] = []
        seen = set()
        with open(dictPath, "r", encoding=encoding) as f:
            for line in f:
                token = line.strip()
                if not token:
                    continue
                ch = token
                if ch in seen:
                    continue
                seen.add(ch)
                chars.append(ch)
        if not chars:
            raise RuntimeError(f"OCR vocab file {dictPath} is empty or invalid")
        return "".join(chars)

    def ForwardDetect(
        self,
        imagesTensor: torch.Tensor,
        gtBoxes: Optional[torch.Tensor] = None,
        gtMask: Optional[torch.Tensor] = None,
        gtTextMask: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:
        feat = self.backbone(imagesTensor)
        det_out = self.dbHead.ForwardWithLogits(feat)

        out: Dict[str, torch.Tensor] = dict(det_out)

        target = gtBoxes if gtTextMask is None else gtTextMask
        if target is not None:
            loss, stats = self.dbLoss(
                probMap=det_out["prob_map"],
                probLogits=det_out["prob_logits"],
                gtBoxes=target,
                gtMask=gtMask,)

            out["loss"] = loss
            for k, v in stats.items():
                out[f"stat_{k}"] = v

        return out


    def ForwardRecognize(
        self,
        lineImgs: torch.Tensor,
        targetsTensor: Optional[torch.Tensor] = None,
        targetLengths: Optional[torch.Tensor] = None,
        inputLengths: Optional[torch.Tensor] = None,
        validWidths: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:

        if inputLengths is None and validWidths is not None:
            inputLengths = WidthToCtcSteps(validWidths)
        return self.recognizer(
            lineImgs,
            targetsTensor=targetsTensor,
            targetLengths=targetLengths,
            inputLengths=inputLengths,)


    def BitmapToBoxes(
        self,
        binMap: torch.Tensor,
        threshValue: float = 0.3,
        minArea: int = 10,) -> List[np.ndarray]:

        bm = binMap.detach().cpu().squeeze(0).numpy()
        mask = (bm > threshValue).astype(np.uint8)
        h, w = mask.shape

        visited = np.zeros_like(mask, dtype=bool)
        boxes: List[np.ndarray] = []

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for y in range(h):
            for x in range(w):
                if mask[y, x] == 0 or visited[y, x]:
                    continue
                queue = [(y, x)]
                visited[y, x] = True
                min_y = max_y = y
                min_x = max_x = x
                area = 0

                while queue:
                    cy, cx = queue.pop()
                    area += 1
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if mask[ny, nx] == 1 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                if area >= minArea:
                    boxes.append(np.array([min_x, min_y, max_x + 1, max_y + 1], dtype=np.int32))

        return boxes


    def CropAndResizeLines(
        self,
        imageTensor: torch.Tensor,
        boxes: List[np.ndarray],
        targetH: int = 32,
        maxW: int = 512,) -> torch.Tensor:
        line_imgs, _ = self.CropAndResizeLinesWithWidths(
            imageTensor,
            boxes,
            targetH=targetH,
            maxW=maxW,)
        return line_imgs


    def CropAndResizeLinesWithWidths(
        self,
        imageTensor: torch.Tensor,
        boxes: List[np.ndarray],
        targetH: int = 32,
        maxW: int = 512,) -> Tuple[torch.Tensor, torch.Tensor]:

        c, h_img, w_img = imageTensor.shape

        gray = (0.299 * imageTensor[0]
            + 0.587 * imageTensor[1]
            + 0.114 * imageTensor[2]) 

        device = imageTensor.device
        line_tensors: List[torch.Tensor] = []
        valid_widths: List[int] = []

        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = gray[y1:y2, x1:x2]  # [h,w]
            h, w = patch.shape
            if h < 1 or w < 1:
                continue

            patch = patch.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]

            scale = targetH / float(h)
            new_w = max(1, int(round(w * scale)))
            patch_resized = F.interpolate(
                patch,
                size=(targetH, new_w),
                mode="bilinear",
                align_corners=False,)  

            if new_w > maxW:
                patch_resized = patch_resized[:, :, :, :maxW]
                new_w = maxW

            pad = torch.zeros(1, 1, targetH, maxW, device=device, dtype=imageTensor.dtype)
            pad[:, :, :, :new_w] = patch_resized
            line_tensors.append(pad)
            valid_widths.append(int(new_w))

        if not line_tensors:
            return (
                torch.empty(0, 1, targetH, maxW, dtype=imageTensor.dtype, device=device),
                torch.empty(0, dtype=torch.long, device=device),)

        return (
            torch.cat(line_tensors, dim=0),
            torch.tensor(valid_widths, dtype=torch.long, device=device),)


    def forward(
        self,
        imagesTensor: torch.Tensor, # [B,3,H,W]
        binThresh: float = 0.3,
        minBoxArea: int = 10,) -> List[List[OcrItem]]:

        feat = self.backbone(imagesTensor) # [B,256,H,W]
        prob_map = self.dbHead(feat) # [B,1,H,W]

        bsz = imagesTensor.size(0)
        results_batch: List[List[Tuple[np.ndarray, str, float, float]]] = []

        if self.temporalSteps > 0 and self._tracks_by_bi:
            for k in list(self._tracks_by_bi.keys()):
                if k < 0 or k >= bsz:
                    del self._tracks_by_bi[k]

        for bi in range(bsz):
            pm = prob_map[bi] 
            img = imagesTensor[bi]  

            triplets: List[Tuple[np.ndarray, str, float, float]] = []
            frame_obs: List[OcrLineObs] = []  

            boxes = self.BitmapToBoxes(pm, threshValue=binThresh, minArea=minBoxArea)

            if len(boxes) != 0:
                line_imgs, valid_widths = self.CropAndResizeLinesWithWidths(img, boxes, targetH=32, maxW=512)

                if line_imgs.size(0) != 0:
                    rec_out = self.ForwardRecognize(line_imgs, validWidths=valid_widths)
                    pairs = self.CtcGreedyDecodeWithConf(
                        rec_out["log_probs"],
                        idx2Char=self.idx2Char,
                        blankIndex=self.blankIndex)

                    h_map, w_map = pm.shape[-2], pm.shape[-1]
                    pm_np = pm.detach().cpu().squeeze(0).numpy() 

                    for box, (text, rec_conf) in zip(boxes, pairs):
                        x1, y1, x2, y2 = box.tolist()
                        x1 = max(0, x1); y1 = max(0, y1)
                        x2 = min(w_map, x2); y2 = min(h_map, y2)

                        region = pm_np[y1:y2, x1:x2]
                        det_score = float(region.mean()) if region.size != 0 else 0.0

                        triplets.append((box, text, det_score, float(rec_conf)))
                        frame_obs.append(OcrLineObs(
                            box=(int(x1), int(y1), int(x2), int(y2)),
                            text=text,
                            det_score=det_score,
                            rec_conf=float(rec_conf),
                            step=self._temporal_step))

            results_batch.append(triplets)

            if self.temporalSteps > 0:
                self.UpdateTemporalTracks(bi, frame_obs)

        if self.temporalSteps > 0:
            self._temporal_step += 1

        ocr_items = self.OcrResultsToOcrItems(results_batch) 
        self._last_batch_size = bsz
        self._last_ocr_texts_batch = [[it["text"] for it in items] for items in ocr_items]
        return ocr_items


    def OcrResultsToOcrTexts(
        self,
        resultsBatch: List[List[Tuple[np.ndarray, str, float, float]]],
        scoreThresh: float = 0.3,) -> List[List[str]]:
        ocr_texts: List[List[str]] = []

        for triplets in resultsBatch:
            lines: List[str] = []
            for box, text, det_score, rec_conf in triplets:
                score = max(0.0, float(det_score)) * max(0.0, float(rec_conf))
                if score < scoreThresh:
                    continue
                t = str(text).strip()
                if not t:
                    continue
                lines.append(t)
            ocr_texts.append(lines)

        return ocr_texts


    def OcrResultsToOcrItems(
        self,
        resultsBatch: List[List[Tuple[np.ndarray, str, float, float]]],
        scoreThresh: float = 0.3,) -> List[List[OcrItem]]:

        out_batch: List[List[OcrItem]] = []
        for triplets in resultsBatch:
            items: List[OcrItem] = []
            for box, text, det_score, rec_conf in triplets:
                score = max(0.0, float(det_score)) * max(0.0, float(rec_conf))
                if score < scoreThresh:
                    continue
                t = str(text).strip()
                if not t:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.tolist()]  # xyxy
                items.append({
                    "box": (x1, y1, x2, y2),
                    "text": t,
                    "det_score": float(det_score),
                    "rec_conf": float(rec_conf),
                    "score": float(score),})
            out_batch.append(items)
        return out_batch


    def ResetTemporal(self):
        self._temporal_step = 0
        self._tracks_by_bi.clear()
        self._last_batch_size = 0
        self._last_ocr_texts_batch = []


    def CtcGreedyDecodeWithConf(
        self,
        logProbs: torch.Tensor, 
        idx2Char: List[str],
        blankIndex: int = 0,) -> List[tuple[str, float]]:
        preds = logProbs.argmax(dim=-1) 
        chosen_lp = logProbs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1) 
        chosen_p = torch.exp(chosen_lp).clamp(0.0, 1.0)  

        t, b = preds.shape
        out: List[tuple[str, float]] = []
        for li in range(b):
            seq = preds[:, li].tolist()
            pseq = chosen_p[:, li].detach().cpu().tolist()
            prev = None
            chars: List[str] = []
            confs: List[float] = []
            for idx, p in zip(seq, pseq):
                if idx == blankIndex:
                    prev = None
                    continue
                if prev == idx:
                    continue
                chars.append(idx2Char[idx])
                confs.append(float(p))
                prev = idx
            text = "".join(chars)
            conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0
            out.append((text, conf))
        return out


    def UpdateTemporalTracks(self, bi: int, frameObs: List[OcrLineObs]):
        tracks = self._tracks_by_bi.get(bi, [])
        for tr in tracks:
            tr.age += 1

        def match_track(obs: OcrLineObs) -> int:
            best_i = -1
            best_s = -1e9
            for i, tr in enumerate(tracks):
                last = tr.obs[-1]
                iou = IouXyxy(last.box, obs.box)
                sim = TextSim(last.text, obs.text)
                if (iou >= self.fuseIouThr) or (iou >= self.fuseIouThr * 0.5 and sim >= self.fuseSimThr):
                    s = 2.0 * iou + 1.0 * sim + 0.2 * obs.weight
                    if s > best_s:
                        best_s = s
                        best_i = i
            return best_i

        for obs in frameObs:
            ti = match_track(obs)
            if ti < 0:
                dq = deque([obs], maxlen=self.temporalSteps)
                tracks.append(OcrTrack(obs=dq, age=0))
            else:
                tracks[ti].obs.append(obs)
                tracks[ti].age = 0

        ttl = self.temporalSteps
        tracks = [tr for tr in tracks if tr.age <= ttl and len(tr.obs) > 0]
        self._tracks_by_bi[bi] = tracks



    def ExportFusedTextsForBi(self, bi: int) -> List[str]:
        if self.temporalSteps <= 0:
            if bi < 0 or bi >= len(self._last_ocr_texts_batch):
                return []
            return list(self._last_ocr_texts_batch[bi])

        tracks = self._tracks_by_bi.get(bi, [])
        if not tracks:
            return []

        cands: List[tuple[float, int, str]] = [] 

        for tr in tracks:
            votes = defaultdict(float)
            best_raw_for_norm = {}
            best_w_for_norm = defaultdict(float)

            for ob in tr.obs:
                nt = NormText(ob.text)
                if not nt:
                    continue
                age = (self._temporal_step - 1) - ob.step
                recency = self.fuseDecay ** max(0, age)
                w = ob.weight * recency
                votes[nt] += w
                if w > best_w_for_norm[nt]:
                    best_w_for_norm[nt] = w
                    best_raw_for_norm[nt] = ob.text

            if not votes:
                continue

            hits = len(tr.obs)
            if hits < self.fuseMinHits:
                continue

            best_nt = max(votes.items(), key=lambda kv: kv[1])[0]
            rep_text = best_raw_for_norm.get(best_nt, "")
            score = votes[best_nt] * math.sqrt(hits)
            if NormText(rep_text):
                cands.append((score, hits, rep_text))

        cands.sort(key=lambda x: (x[0], x[1], len(x[2])), reverse=True)

        out: List[str] = []
        for score, hits, text in cands:
            if len(out) >= self.fuseTopK:
                break
            if any(TextSim(text, p) >= max(self.fuseSimThr, 0.80) for p in out):
                continue
            out.append(text)

        return out

    def ExportFusedTexts(self, bi: Optional[int] = None) -> Union[List[str], List[List[str]]]:
        if bi is not None:
            return self.ExportFusedTextsForBi(int(bi))

        bsz = self._last_batch_size
        if bsz <= 0:
            if self.temporalSteps > 0 and self._tracks_by_bi:
                bsz = max(self._tracks_by_bi.keys()) + 1
            elif self._last_ocr_texts_batch:
                bsz = len(self._last_ocr_texts_batch)
            else:
                return []

        return [self.ExportFusedTextsForBi(i) for i in range(bsz)]



class TestOCRMTool:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)


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

    def MakeEngine(self) -> "OCREngineExtractor":
        try:
            engine = OCREngineExtractor().to(self.device)
            return engine
        except (TypeError, FileNotFoundError, RuntimeError) as e:
            print(f"OCREngineExtractor init failed ({e}), patching LoadOcrVocabFromTxt with dummy vocab.")

            OCREngineExtractor.LoadOcrVocabFromTxt = (
                lambda self, dictPath, *, encoding="utf-8": "0123456789ABCDEF"
            )
            engine = OCREngineExtractor().to(self.device)
            return engine


    def DetectForwardShapes(self) -> bool:
        try:
            backbone = DBBackbone(inCh=3, baseCh=64).to(self.device)
            head = DBHead(inCh=256).to(self.device)
            backbone.eval()
            head.eval()

            B, H, W = 2, 256, 256
            imgs = torch.randn(B, 3, H, W, device=self.device)

            with torch.no_grad():
                feat = backbone(imgs)
                det_out = head.ForwardWithLogits(feat)
                prob = head(feat)

            assert prob.shape[0] == B and prob.shape[1] == 1
            assert det_out["prob_map"].shape == prob.shape
            assert det_out["prob_logits"].shape == prob.shape
            assert det_out["base_prob_logits"].shape == prob.shape
            assert det_out["delta_prob_logits"].shape == prob.shape

            for t in (prob, det_out["prob_map"]):
                assert torch.isfinite(t).all()
                assert t.min().item() >= -1e-6 and t.max().item() <= 1.0 + 1e-6
            assert torch.isfinite(det_out["prob_logits"]).all()

            print("DetectForwardShapes passed.")
            return True
        except AssertionError as e:
            print("DetectForwardShapes failed:", e)
            return False
        except Exception as e:
            print("DetectForwardShapes error:", e)
            return False

    def DetectLossGradSmoke(self) -> bool:
        try:
            backbone = DBBackbone(inCh=3, baseCh=64).to(self.device)
            head = DBHead(inCh=256).to(self.device)
            crit = DBLoss().to(self.device)

            backbone.train()
            head.train()

            B, H, W = 2, 256, 256
            imgs = torch.randn(B, 3, H, W, device=self.device)

            feat = backbone(imgs)
            det_out = head.ForwardWithLogits(feat)
            prob = det_out["prob_map"]

            gtBoxes = (torch.rand_like(prob) > 0.7).float()
            gtMask = (torch.rand_like(prob) > 0.1).float()

            loss, stats = crit(
                probMap=prob,
                probLogits=det_out["prob_logits"],
                gtBoxes=gtBoxes,
                gtMask=gtMask,)

            opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=1e-3)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            named: Dict[str, nn.Parameter] = {}
            named.update({f"backbone.{k}": v for k, v in backbone.named_parameters()})
            named.update({f"dbHead.{k}": v for k, v in head.named_parameters()})

            must_have = [
                "backbone.enc1.0.conv.weight",
                "backbone.enc2.0.conv.weight",
                "backbone.outConv.conv.weight",
                "dbHead.probConv.0.conv.weight",
                "dbHead.residual.block.3.weight",
                ]

            ok_cov = self.GradCoverage(named, min_ratio=0.4, must_have=must_have)
            assert ok_cov, "DB detection grad coverage failed."

            for n, p in named.items():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

            opt.step()
            print("DetectLossGradSmoke passed.")
            return True
        except AssertionError as e:
            print("DetectLossGradSmoke failed:", e)
            return False
        except Exception as e:
            print("DetectLossGradSmoke error:", e)
            return False


    def RecognizeForwardShapes(self) -> bool:
        try:
            rec = CRNNRecognizer(imgH=32, inCh=1, nClasses=32, rnnHidden=128).to(self.device)
            rec.eval()

            B = 3
            imgs = torch.randn(B, 1, 32, 128, device=self.device)

            with torch.no_grad():
                out = rec(imgs)
                logits = out["logits"]
                log_probs = out["log_probs"]

            T, B2, C = log_probs.shape
            assert B2 == B and C == rec.nClasses
            assert T > 0

            assert torch.isfinite(logits).all()
            assert torch.isfinite(log_probs).all()

            print("RecognizeForwardShapes passed.")
            return True
        except AssertionError as e:
            print("RecognizeForwardShapes failed:", e)
            return False
        except Exception as e:
            print("RecognizeForwardShapes error:", e)
            return False

    def RecognizeForwardShapes512Square(self) -> bool:
        try:
            rec = CRNNRecognizer(imgH=32, inCh=3, nClasses=6634, rnnHidden=64).to(self.device)
            rec.eval()

            B, H, W = 1, 512, 512
            imgs = torch.randn(B, 3, H, W, device=self.device)

            with torch.no_grad():
                out = rec(imgs)
                logits = out["logits"]
                log_probs = out["log_probs"]

            T, B2, C = log_probs.shape
            assert B2 == B and C == rec.nClasses
            assert T > 0

            assert logits.shape == (T, B, rec.nClasses)
            assert torch.isfinite(logits).all()
            assert torch.isfinite(log_probs).all()

            print(f"RecognizeForwardShapes512Square passed. logits shape={tuple(logits.shape)}")
            return True
        except AssertionError as e:
            print("RecognizeForwardShapes512Square failed:", e)
            return False
        except Exception as e:
            print("RecognizeForwardShapes512Square error:", e)
            return False

    def RecognizeCtcGradSmoke(self) -> bool:
        try:
            nClasses = 32
            rec = CRNNRecognizer(imgH=32, inCh=1, nClasses=nClasses, rnnHidden=128).to(self.device)
            rec.train()

            B = 4
            imgs = torch.randn(B, 1, 32, 128, device=self.device)

            with torch.no_grad():
                out0 = rec(imgs)
                T = out0["log_probs"].size(0)

            max_len = max(1, T // 2)
            targetLengths = torch.randint(1, max_len + 1, (B,), device=self.device, dtype=torch.long)
            total_len = int(targetLengths.sum().item())
            targets = torch.randint(1, nClasses, (total_len,), device=self.device, dtype=torch.long)
            inputLengths = torch.full((B,), T, dtype=torch.long, device=self.device)

            out = rec(
                imgs,
                targetsTensor=targets,
                targetLengths=targetLengths,
                inputLengths=inputLengths,)
            assert "loss" in out
            assert "ctc_invalid_ratio" in out
            assert float(out["ctc_invalid_ratio"].item()) == 0.0
            loss = out["loss"]

            opt = torch.optim.Adam(rec.parameters(), lr=1e-3)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            named = dict(rec.named_parameters())
            must_have = [
                "conv1.0.weight",
                "conv2.0.weight",
                "rnn.weight_ih_l0",
                "fc.weight",
                "residualHead.out.weight",]

            ok_cov = self.GradCoverage(named, min_ratio=0.5, must_have=must_have)
            assert ok_cov, "CRNN grad coverage failed."

            for n, p in named.items():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

            opt.step()
            print("RecognizeCtcGradSmoke passed.")
            return True
        except AssertionError as e:
            print("RecognizeCtcGradSmoke failed:", e)
            return False
        except Exception as e:
            print("RecognizeCtcGradSmoke error:", e)
            return False


    def EngineForwardDetectAndLoss(self) -> bool:
        try:
            engine = self.MakeEngine()
            engine.train()

            B, H, W = 2, 256, 256
            imgs = torch.randn(B, 3, H, W, device=self.device)

            with torch.no_grad():
                feat_tmp = engine.backbone(imgs)
                prob_tmp = engine.dbHead(feat_tmp)

            gtBoxes = (torch.rand_like(prob_tmp) > 0.7).float()
            gtMask = (torch.rand_like(prob_tmp) > 0.1).float()

            out = engine.ForwardDetect(imgs, gtBoxes=gtBoxes, gtMask=gtMask,)
            assert "loss" in out
            assert "prob_logits" in out and out["prob_logits"].shape == out["prob_map"].shape
            loss = out["loss"]

            opt = torch.optim.Adam(list(engine.backbone.parameters()) + list(engine.dbHead.parameters()), lr=1e-3)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            named: Dict[str, nn.Parameter] = {}
            named.update({f"backbone.{k}": v for k, v in engine.backbone.named_parameters()})
            named.update({f"dbHead.{k}": v for k, v in engine.dbHead.named_parameters()})

            must_have = [
                "backbone.enc1.0.conv.weight",
                "backbone.enc4.1.conv.weight",
                "dbHead.probConv.0.conv.weight",
                "dbHead.residual.block.3.weight",
                ]

            ok_cov = self.GradCoverage(named, min_ratio=0.4, must_have=must_have)
            assert ok_cov, "Engine detect grad coverage failed."

            opt.step()
            print("EngineForwardDetectAndLoss passed.")
            return True
        except AssertionError as e:
            print("EngineForwardDetectAndLoss failed:", e)
            return False
        except Exception as e:
            print("EngineForwardDetectAndLoss error:", e)
            return False

    def EngineForwardRecognizeAndLoss(self) -> bool:
        try:
            engine = self.MakeEngine()
            rec = engine.recognizer.to(self.device)
            rec.train()

            B = 3
            imgs = torch.randn(B, 1, 32, 128, device=self.device)

            with torch.no_grad():
                out0 = rec(imgs)
                T = out0["log_probs"].size(0)

            max_len = max(1, T // 2)
            targetLengths = torch.randint(1, max_len + 1, (B,), device=self.device, dtype=torch.long)
            total_len = int(targetLengths.sum().item())
            targets = torch.randint(1, rec.nClasses, (total_len,), device=self.device, dtype=torch.long)
            validWidths = torch.full((B,), 128, dtype=torch.long, device=self.device)

            out = engine.ForwardRecognize(
                imgs,
                targetsTensor=targets,
                targetLengths=targetLengths,
                validWidths=validWidths,)
            assert "loss" in out
            assert "ctc_invalid_ratio" in out
            loss = out["loss"]

            opt = torch.optim.Adam(rec.parameters(), lr=1e-3)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            named = dict(rec.named_parameters())
            must_have = [
                "conv1.0.weight",
                "conv4.0.weight",
                "rnn.weight_ih_l0",
                "fc.weight",
                "residualHead.out.weight",]

            ok_cov = self.GradCoverage(named, min_ratio=0.5, must_have=must_have)
            assert ok_cov, "Engine recognizer grad coverage failed."

            for n, p in named.items():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

            opt.step()
            print("EngineForwardRecognizeAndLoss passed.")
            return True
        except AssertionError as e:
            print("EngineForwardRecognizeAndLoss failed:", e)
            return False
        except Exception as e:
            print("EngineForwardRecognizeAndLoss error:", e)
            return False

    def EngineFullForwardPipeline(self) -> bool:
        try:
            engine = self.MakeEngine()
            engine.eval()

            B, H, W = 2, 256, 256
            imgs = torch.randn(B, 3, H, W, device=self.device)

            with torch.no_grad():
                items_batch = engine(imgs, binThresh=0.3, minBoxArea=5)

            assert isinstance(items_batch, list)
            assert len(items_batch) == B

            for items in items_batch:
                assert isinstance(items, list)
                for it in items:
                    assert isinstance(it, dict)
                    assert isinstance(it.get("text"), str)
                    box = it.get("box")
                    assert isinstance(box, tuple) and len(box) == 4
                    assert all(isinstance(v, int) for v in box)
                    assert isinstance(it.get("det_score"), float)
                    assert isinstance(it.get("rec_conf"), float)
                    assert isinstance(it.get("score"), float)

            print("EngineFullForwardPipeline passed.")
            return True
        except AssertionError as e:
            print("EngineFullForwardPipeline failed:", e)
            return False
        except Exception as e:
            print("EngineFullForwardPipeline error:", e)
            return False

    def EngineJointTrainGradCoverage(self) -> bool:
        try:
            engine = self.MakeEngine()
            engine.train()

            B_det, H, W = 2, 256, 256
            imgs_det = torch.randn(B_det, 3, H, W, device=self.device)

            det_probe = engine.ForwardDetect(imgs_det)
            prob = det_probe["prob_map"]
            gtShrink = (torch.rand_like(prob) > 0.7).float()
            gtMask = (torch.rand_like(prob) > 0.1).float()

            det_out = engine.ForwardDetect(imgs_det, gtBoxes=gtShrink, gtMask=gtMask)
            loss_det = det_out["loss"]

            rec = engine.recognizer
            B_rec = 3
            imgs_rec = torch.randn(B_rec, 1, 32, 128, device=self.device)

            with torch.no_grad():
                out0 = rec(imgs_rec)
                T = out0["log_probs"].size(0)

            max_len = max(1, T // 2)
            targetLengths = torch.randint(1, max_len + 1, (B_rec,), device=self.device, dtype=torch.long)
            total_len = int(targetLengths.sum().item())
            targets = torch.randint(1, rec.nClasses, (total_len,), device=self.device, dtype=torch.long)

            validWidths = torch.full((B_rec,), 128, dtype=torch.long, device=self.device)
            out_rec = rec(
                imgs_rec,
                targetsTensor=targets,
                targetLengths=targetLengths,
                inputLengths=WidthToCtcSteps(validWidths),)
            loss_rec = out_rec["loss"]
            total_loss = loss_det + loss_rec

            opt = torch.optim.Adam(engine.parameters(), lr=1e-3)
            opt.zero_grad(set_to_none=True)
            total_loss.backward()

            named = dict(engine.named_parameters())
            must_have = [
                "backbone.enc1.0.conv.weight",
                "backbone.enc4.1.conv.weight",
                "dbHead.probConv.0.conv.weight",
                "dbHead.residual.block.3.weight",
                "recognizer.conv1.0.weight",
                "recognizer.rnn.weight_ih_l0",
                "recognizer.fc.weight",
                "recognizer.residualHead.out.weight",]

            ok_cov = self.GradCoverage(named, min_ratio=0.5, must_have=must_have)
            assert ok_cov, "Engine joint grad coverage failed."

            for n, p in named.items():
                if p.requires_grad and p.grad is not None:
                    assert torch.isfinite(p.grad).all(), f"Non-finite grad at {n}"

            opt.step()
            print("EngineJointTrainGradCoverage passed.")
            return True
        except AssertionError as e:
            print("EngineJointTrainGradCoverage failed:", e)
            return False
        except Exception as e:
            print("EngineJointTrainGradCoverage error:", e)
            return False


    def RunAll(self) -> Dict[str, bool]:
        results = {
            "DetectForwardShapes": self.DetectForwardShapes(),
            "DetectLossGradSmoke": self.DetectLossGradSmoke(),
            "RecognizeForwardShapes": self.RecognizeForwardShapes(),
            "RecognizeForwardShapes512Square": self.RecognizeForwardShapes512Square(),
            "RecognizeCtcGradSmoke": self.RecognizeCtcGradSmoke(),
            "EngineForwardDetectAndLoss": self.EngineForwardDetectAndLoss(),
            "EngineForwardRecognizeAndLoss": self.EngineForwardRecognizeAndLoss(),
            "EngineFullForwardPipeline": self.EngineFullForwardPipeline(),
            "EngineJointTrainGradCoverage": self.EngineJointTrainGradCoverage(),}
        
        passed = sum(1 for v in results.values() if v)
        print(f"\nOCR module tests: {passed}/{len(results)} passed.")
        return results
