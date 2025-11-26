from __future__ import annotations
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, inCh: int, outCh: int, kSize: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(inCh, outCh, kernel_size=kSize, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(outCh)
        self.act = nn.ReLU(inplace=True)

    def forward(self, inputTensor: torch.Tensor) -> torch.Tensor:
        x = self.conv(inputTensor)
        x = self.bn(x)
        x = self.act(x)
        return x


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
        return out


class DBHead(nn.Module):
    def __init__(self, inCh: int = 256, kValue: float = 50.0):
        super().__init__()
        self.k = float(kValue)

        self.probConv = nn.Sequential(
            ConvBNReLU(inCh, 64, kSize=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1),)

        self.threshConv = nn.Sequential(
            ConvBNReLU(inCh, 64, kSize=3, stride=1, padding=1),
            nn.Conv2d(64, 1, kernel_size=1),)

    def forward(self, featTensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prob_logits = self.probConv(featTensor)
        thresh_logits = self.threshConv(featTensor)

        prob_map = torch.sigmoid(prob_logits)
        thresh_map = torch.sigmoid(thresh_logits)

        bin_map = torch.sigmoid(self.k * (prob_map - thresh_map))

        return prob_map, thresh_map, bin_map


def BalancedBceLoss(predTensor: torch.Tensor, gtTensor: torch.Tensor, maskTensor: Optional[torch.Tensor] = None) -> torch.Tensor:

    eps = 1e-6
    pred = predTensor.clamp(eps, 1.0 - eps)
    gt = gtTensor

    if maskTensor is None:
        mask = torch.ones_like(gt)
    else:
        mask = maskTensor

    pos = (gt > 0.5).float()
    neg = 1.0 - pos

    n_pos = pos.sum().clamp(min=1.0)
    n_neg = neg.sum().clamp(min=1.0)

    w_pos = n_neg / (n_pos + n_neg)
    w_neg = n_pos / (n_pos + n_neg)

    loss_pos = -w_pos * (pos * torch.log(pred)) 
    loss_neg = -w_neg * (neg * torch.log(1.0 - pred))

    loss = (loss_pos + loss_neg) * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


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
        lambdaBin: float = 1.0,
        lambdaThresh: float = 10.0,):
        super().__init__()
        self.lambdaProb = float(lambdaProb)
        self.lambdaBin = float(lambdaBin)
        self.lambdaThresh = float(lambdaThresh)

    def forward(
        self,
        probMap: torch.Tensor,
        binMap: torch.Tensor,
        threshMap: torch.Tensor,
        gtShrink: torch.Tensor,
        gtThresh: torch.Tensor,
        gtMask: Optional[torch.Tensor] = None,) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if gtMask is None:
            gtMask = torch.ones_like(gtShrink)

        loss_prob = BalancedBceLoss(probMap, gtShrink, gtMask)
        loss_bin = DiceLoss(binMap, gtShrink, gtMask)

        loss_thresh = F.l1_loss(threshMap * gtMask, gtThresh * gtMask)

        total = (self.lambdaProb * loss_prob
            + self.lambdaBin * loss_bin
            + self.lambdaThresh * loss_thresh)

        stats = {"loss_prob": loss_prob.detach(),
            "loss_bin": loss_bin.detach(),
            "loss_thresh": loss_thresh.detach(),}
        
        return total, stats



class CRNNRecognizer(nn.Module):
    def __init__(
        self,
        imgH: int = 32,
        inCh: int = 1,
        nClasses: int = 96, 
        rnnHidden: int = 256,):
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

        self.ctcLoss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

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
        targetLengths: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:

        x = self.conv1(imgsTensor)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x) 

        seq = self.FeaturesToSeq(x) 

        rnn_out, _ = self.rnn(seq)
        logits = self.fc(rnn_out) 
        log_probs = F.log_softmax(logits, dim=-1)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "log_probs": log_probs,}

        if targetsTensor is not None and targetLengths is not None:
            t, b, _ = log_probs.size()
            input_lengths = torch.full(
                size=(b,),
                fill_value=t,
                dtype=torch.long,
                device=log_probs.device,)
            
            loss = self.ctcLoss(
                log_probs,
                targetsTensor,
                input_lengths,
                targetLengths,)
            
            out["loss"] = loss

        return out







class OCREngineExtractor(nn.Module):
    def __init__(
        self,
        vocabCharsPath: str = "/home/yhl/Documents/Intelligent-Robot-System/BrainDeepLearn/ModuleSetting/OCRKeys.txt",
        dbK: float = 50.0,):
        super().__init__()

        self.backbone = DBBackbone(inCh=3, baseCh=64)
        self.dbHead = DBHead(inCh=256, kValue=dbK)
        self.dbLoss = DBLoss()

        self.blankIndex = 0
        vocabChars = self.LoadOcrVocabFromTxt(vocabCharsPath)
        self.idx2Char = ["<blank>"] + list(vocabChars)
        self.char2Idx = {c: i for i, c in enumerate(self.idx2Char)}

        self.recognizer = CRNNRecognizer(
            imgH=32,
            inCh=1,
            nClasses=len(self.idx2Char),
            rnnHidden=256,)


    def LoadOcrVocabFromTxt(dictPath: str, *, encoding: str = "utf-8") -> str:
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

    def CtcGreedyDecode(self,
        logProbs: torch.Tensor,
        idx2Char: List[str],
        blankIndex: int = 0,) -> List[str]:

        t, b, c = logProbs.shape
        preds = logProbs.argmax(dim=-1) 

        results: List[str] = []
        for bi in range(b):
            seq = preds[:, bi].tolist()
            prev = None
            chars: List[str] = []
            for idx in seq:
                if idx == blankIndex:
                    prev = None
                    continue
                if prev == idx:
                    continue
                chars.append(idx2Char[idx])
                prev = idx
            results.append("".join(chars))
        return results

    def ForwardDetect(
        self,
        imagesTensor: torch.Tensor,
        gtShrink: Optional[torch.Tensor] = None,
        gtThresh: Optional[torch.Tensor] = None,
        gtMask: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:
        feat = self.backbone(imagesTensor)
        prob_map, thresh_map, bin_map = self.dbHead(feat)

        out: Dict[str, torch.Tensor] = {
            "prob_map": prob_map,
            "thresh_map": thresh_map,
            "bin_map": bin_map,}

        if gtShrink is not None and gtThresh is not None:
            loss, stats = self.dbLoss(
                probMap=prob_map,
                binMap=bin_map,
                threshMap=thresh_map,
                gtShrink=gtShrink,
                gtThresh=gtThresh,
                gtMask=gtMask,)
            
            out["loss"] = loss
            for k, v in stats.items():
                out[f"stat_{k}"] = v

        return out


    def ForwardRecognize(
        self,
        lineImgs: torch.Tensor,
        targetsTensor: Optional[torch.Tensor] = None,
        targetLengths: Optional[torch.Tensor] = None,) -> Dict[str, torch.Tensor]:

        return self.recognizer(lineImgs, targetsTensor, targetLengths)


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
        maxW: int = 256,) -> torch.Tensor:

        c, h_img, w_img = imageTensor.shape

        gray = (0.299 * imageTensor[0]
            + 0.587 * imageTensor[1]
            + 0.114 * imageTensor[2]) 

        device = imageTensor.device
        line_tensors: List[torch.Tensor] = []

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

        if not line_tensors:
            return torch.empty(0, 1, targetH, maxW, dtype=imageTensor.dtype, device=device)

        return torch.cat(line_tensors, dim=0) 


    def forward(
        self,
        imagesTensor: torch.Tensor,
        binThresh: float = 0.3,
        minBoxArea: int = 10,) -> List[List[Tuple[np.ndarray, str, float]]]:

        feat = self.backbone(imagesTensor)
        prob_map, thresh_map, bin_map = self.dbHead(feat)

        bsz = imagesTensor.size(0)
        results_batch: List[List[Tuple[np.ndarray, str, float]]] = []

        for bi in range(bsz):
            pm = prob_map[bi]
            bm = bin_map[bi] 
            img = imagesTensor[bi] 

            boxes = self.BitmapToBoxes(bm, threshValue=binThresh, minArea=minBoxArea)
            if len(boxes) == 0:
                results_batch.append([])
                continue

            line_imgs = self.CropAndResizeLines(img, boxes, targetH=32, maxW=256)
            if line_imgs.size(0) == 0:
                results_batch.append([])
                continue

            rec_out = self.recognizer(line_imgs)
            texts = self.CtcGreedyDecode(
                rec_out["log_probs"],
                idx2Char=self.idx2Char,
                blankIndex=self.blankIndex,)

            h_map, w_map = pm.shape[-2], pm.shape[-1]
            pm_np = pm.detach().cpu().squeeze(0).numpy()

            triplets: List[Tuple[np.ndarray, str, float]] = []
            for box, text in zip(boxes, texts):
                x1, y1, x2, y2 = box.tolist()
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_map, x2)
                y2 = min(h_map, y2)
                region = pm_np[y1:y2, x1:x2]
                if region.size == 0:
                    score = 0.0
                else:
                    score = float(region.mean())
                triplets.append((box, text, score))

            results_batch.append(triplets)

        return self.OcrResultsToOcrTexts(results_batch)


    def OcrResultsToOcrTexts(
        self,
        resultsBatch: List[List[Tuple[np.ndarray, str, float]]],
        scoreThresh: float = 0.3,) -> List[List[str]]:
        ocr_texts: List[List[str]] = []

        for triplets in resultsBatch:
            lines: List[str] = []
            for box, text, score in triplets:
                if score < scoreThresh:
                    continue
                t = str(text).strip()
                if not t:
                    continue
                lines.append(t)
            ocr_texts.append(lines)

        return ocr_texts
