from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import threading
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import traceback
import os

from torch.utils.data import Dataset, DataLoader

from PerceptionModule import TestPerceptionMTool
from AttentionModule import  TestAttentionMTool
from MemoryModule import  TestMemoryMTool
from DecisionModule import RAW_KEYBOARD_LAYOUT, TestDecisionMTool
from WorldModule import  TestWorldMTool
from ValueEstimationModule import  TestValueEstimationMTool
from ConsciousnessModule import TestConsciousMTool
from IntentionModule import TestIntentionMTool
from OCRModule import TestOCRMTool, OCREngineExtractor
from AGICore import Agent, BrainCore, BasicParameters

try:
    import imageio.v3 as iio
except Exception:
    iio = None  


class OfflineGameDataset(Dataset):
    def __init__(self, root: str) -> None:
        p = Path(root)
        self.imgs = sorted((p / "frames").glob("*.png"))
        self.keys = sorted((p / "keys").glob("*.npy"))
        self.mouse_click = sorted((p / "mouse_click").glob("*.npy"))
        self.mouse_move = sorted((p / "mouse_move").glob("*.npy"))
        self.reward = sorted((p / "reward").glob("*.npy")) 
        self.done = sorted((p / "done").glob("*.npy")) 
        self.texts = sorted((p / "texts").glob("*.txt"))

        assert len(self.imgs) == len(self.keys) == len(self.mouse_click) == len(self.mouse_move) == len(self.reward) == len(self.done), "frames/keys/mouse_click/mouse_move/reward/done The number of files is inconsistent."
        if self.texts:
            assert len(self.texts) == len(self.imgs), "texts The number of files is inconsistent."

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        imgs = iio.imread(self.imgs[idx])
        keys = np.load(self.keys[idx]).astype(np.float32)
        mouse_click = np.load(self.mouse_click[idx]).astype(np.float32)
        mouse_move = np.load(self.mouse_move[idx]).astype(np.float32)
        reward = np.load(self.reward[idx]).astype(np.float32)
        done = np.load(self.done[idx]).astype(np.float32)
        ext_text = None
        if self.texts:
            ext_text = self.texts[idx].read_text(encoding="utf-8").strip()
        return imgs, keys, mouse_click, mouse_move, reward, done, ext_text


class OfflineOCRDataset(Dataset):
    def __init__(self, root: str) -> None:
        p = Path(root)
        self.imgs = sorted((p / "frames").glob("*.png"))
        self.boxes = sorted((p / "boxes").glob("*.npy"))
        self.texts = sorted((p / "texts").glob("*.txt"))

        assert len(self.imgs) == len(self.boxes) == len(self.texts), "frames/boxes/texts The number of files is inconsistent."
        if len(self.imgs) == 0:
            raise RuntimeError(f"no OCR samples found under {p}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = iio.imread(self.imgs[idx])
        boxes = np.load(self.boxes[idx]).astype(np.float32)
        if boxes.ndim == 1:
            boxes = boxes.reshape(-1, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"OCR boxes must have shape [N, 4], but got {boxes.shape} from {self.boxes[idx]}")

        texts = self.texts[idx].read_text(encoding="utf-8").splitlines()
        if len(boxes) != len(texts):
            raise ValueError(
                f"OCR texts/boxes count mismatch at {self.texts[idx]} and {self.boxes[idx]}: "
                f"{len(texts)} vs {len(boxes)}")
        return img, boxes, texts



class ModuleController:
    def __init__(self):
        self._lock = threading.Lock()
        self.status: Dict[str, Any] = {
            "state": "idle",
            "epoch": 0, "total_epochs": 0,
            "batch": 0, "total_batches": 0,
            "train_loss": 0.0, "val_loss": 0.0,
            "message": "Waiting to start",
            "trace": ""}
        self.stop_requested = False
        self.pause_requested = False
        self.reset_hebbian = False

    def SetStatus(self, state: str, message: str, **kwargs):
        with self._lock:
            self.status["state"] = state
            self.status["message"] = message
            for k, v in kwargs.items():
                if k in self.status:
                    self.status[k] = v

    def GetStatus(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.status)
        
    def ResteStatus(self):
        self.status: Dict[str, Any] = {
            "state": "idle",
            "epoch": 0, "total_epochs": 0,
            "batch": 0, "total_batches": 0,
            "train_loss": 0.0, "val_loss": 0.0,
            "message": "Waiting to start",
            "trace": ""}

    def RequestStop(self):
        with self._lock:
            self.stop_requested = True

    def RequestPause(self):
        with self._lock:
            self.pause_requested = True

    def RequestResetHebbian(self):
        with self._lock:
            self.reset_hebbian = True

    def RequestCancelResetHebbian(self):
        with self._lock:
            self.reset_hebbian = False

    def RequestResume(self):
        with self._lock:
            self.pause_requested = False

    def ShouldStop(self) -> bool:
        with self._lock:
            return self.stop_requested

    def ShouldPause(self) -> bool:
        with self._lock:
            return self.pause_requested
        
    def ShouldResetHebbian(self) -> bool:
        with self._lock:
            return self.reset_hebbian


class ManagerFunction:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.controller = ModuleController()

        self.br_thread: Optional[threading.Thread] = None
        self.message_thread: Optional[threading.Thread] = None
        self.is_begin = False

        self.test = {
            "perception": TestPerceptionMTool(),
            "attention": TestAttentionMTool(),
            "memory": TestMemoryMTool(),
            "decision": TestDecisionMTool(),
            "world": TestWorldMTool(),
            "value": TestValueEstimationMTool(),
            "consciousness": TestConsciousMTool(),
            "OCR": TestOCRMTool(),
            "intention": TestIntentionMTool(),}

    def StartTraining(self, epochs: int = 5, batchSize: int = 32, valSplit: float = 0.1, resume: bool = True, onlineLearning:bool = False,isTest: bool = False):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.is_begin = True

        ckpt_path = BasicParameters.CKPT_PATH_TEST if isTest else BasicParameters.CKPT_PATH_TRAIN
        wm_mem_path = BasicParameters.WORLD_MEMORY_PATH_TEST if isTest else BasicParameters.WORLD_MEMORY_PATH
        mem_mem_path = BasicParameters.MEMORY_MEMORY_PATH_TEST if isTest else BasicParameters.MEMORY_MEMORY_PATH
        root = BasicParameters.DATA_ROOT_PATH_TEST if isTest else BasicParameters.DATA_ROOT_PATH

        self.br_thread = threading.Thread(
            target=self.TrainLoop, args=(
                root, epochs, batchSize, valSplit, resume, onlineLearning),
                kwargs={"worldMemPath": wm_mem_path, "memMemPath": mem_mem_path, "ckptPath": ckpt_path}, 
                daemon=False)
        self.br_thread.start()
        return True

    def StartOCRTraining(
        self,
        root: str,
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        onlineLearning: bool = False,
        *,
        ckptPath: str,
        outPath: str,):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.is_begin = True

        self.br_thread = threading.Thread(
            target=self.OCRTrainLoop,
            args=(root, epochs, batchSize, valSplit, resume),
            kwargs={
                "onlineLearning": onlineLearning,
                "ckptPath": ckptPath,
                "outPath": outPath,},
                
            daemon=False)
        self.br_thread.start()
        return True

    def Stop(self):
        if self.is_begin:
            self.controller.RequestStop()
            if self.br_thread is not None:
                self.br_thread.join()
                self.br_thread = None

            if self.message_thread is not None:
                self.message_thread.join()
                self.message_thread = None

            self.is_begin = False
            return True
        return False

    def Pause(self):
        if self.is_begin:
            self.controller.RequestPause()
            return True
        return False

    def Resume(self):
        if self.is_begin:
            self.controller.RequestResume()
            return True
        return False
    
    def ResetHebbianMemory(self):
        if self.is_begin:
            self.controller.RequestResetHebbian()
            return True
        return False

    def GetCurrentStatus(self):
        return self.controller.GetStatus()

    def PrepareOCRBatch(
        self,
        imgsBatch,
        boxesBatch,
        textsBatch,
        engine: OCREngineExtractor,
        *,
        device: Optional[torch.device] = None,
        imageSize: int = BasicParameters.IMAGE_SIZE,
        targetH: int = 32,
        maxW: int = 256,):
        device = device or self.device

        if isinstance(imgsBatch, tuple):
            imgsBatch = list(imgsBatch)
        if isinstance(boxesBatch, tuple):
            boxesBatch = list(boxesBatch)
        if isinstance(textsBatch, tuple):
            textsBatch = list(textsBatch)

        detect_imgs: List[torch.Tensor] = []
        gt_shrink_batch: List[torch.Tensor] = []
        gt_thresh_batch: List[torch.Tensor] = []
        gt_mask_batch: List[torch.Tensor] = []
        line_imgs_list: List[torch.Tensor] = []
        flat_targets: List[int] = []
        target_lengths: List[int] = []
        norm_texts: List[str] = []

        for img, raw_boxes, raw_texts in zip(imgsBatch, boxesBatch, textsBatch):
            if isinstance(img, np.ndarray):
                img_t = torch.from_numpy(img)
            elif isinstance(img, torch.Tensor):
                img_t = img.detach().cpu()
            else:
                raise TypeError(f"unsupported OCR image type: {type(img)}")

            if img_t.ndim == 2:
                img_t = img_t.unsqueeze(-1)
            if img_t.ndim != 3:
                raise ValueError(f"OCR image must have 2 or 3 dims, but got shape {tuple(img_t.shape)}")

            if img_t.shape[-1] in (1, 3):
                img_t = img_t.permute(2, 0, 1)
            elif img_t.shape[0] not in (1, 3):
                raise ValueError(f"OCR image channel layout is invalid: {tuple(img_t.shape)}")

            img_t = img_t.float()
            if img_t.max().item() > 1.0:
                img_t = img_t / 255.0

            if img_t.size(0) == 1:
                img_rgb = img_t.repeat(3, 1, 1)
            else:
                img_rgb = img_t[:3]

            _, h0, w0 = img_rgb.shape
            img_rgb = F.interpolate(
                img_rgb.unsqueeze(0),
                size=(imageSize, imageSize),
                mode="bilinear",
                align_corners=False,).squeeze(0)

            detect_imgs.append(img_rgb)

            gt_shrink = torch.zeros(1, imageSize, imageSize, dtype=img_rgb.dtype)
            gt_thresh = torch.zeros(1, imageSize, imageSize, dtype=img_rgb.dtype)
            gt_mask = torch.ones(1, imageSize, imageSize, dtype=img_rgb.dtype)

            boxes_np = np.asarray(raw_boxes, dtype=np.float32).reshape(-1, 4)
            texts = list(raw_texts)
            if len(boxes_np) != len(texts):
                raise ValueError(f"OCR texts/boxes count mismatch in batch: {len(texts)} vs {len(boxes_np)}")

            scale_x = float(imageSize) / float(max(1, w0))
            scale_y = float(imageSize) / float(max(1, h0))

            rec_boxes: List[np.ndarray] = []
            rec_texts: List[str] = []
            for box, raw_text in zip(boxes_np, texts):
                text = "".join(ch for ch in str(raw_text).strip() if ch in engine.char2Idx)
                if not text:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                x1 = int(round(x1 * scale_x))
                y1 = int(round(y1 * scale_y))
                x2 = int(round(x2 * scale_x))
                y2 = int(round(y2 * scale_y))

                x1 = max(0, min(imageSize - 1, x1))
                y1 = max(0, min(imageSize - 1, y1))
                x2 = max(x1 + 1, min(imageSize, x2))
                y2 = max(y1 + 1, min(imageSize, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                gt_thresh[:, y1:y2, x1:x2] = 1.0

                shrink_dx = max(1, int(round((x2 - x1) * 0.15)))
                shrink_dy = max(1, int(round((y2 - y1) * 0.15)))
                sx1 = min(max(0, x1 + shrink_dx), imageSize - 1)
                sy1 = min(max(0, y1 + shrink_dy), imageSize - 1)
                sx2 = max(sx1 + 1, min(imageSize, x2 - shrink_dx))
                sy2 = max(sy1 + 1, min(imageSize, y2 - shrink_dy))
                gt_shrink[:, sy1:sy2, sx1:sx2] = 1.0

                rec_boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))
                rec_texts.append(text)

            gt_shrink_batch.append(gt_shrink)
            gt_thresh_batch.append(gt_thresh)
            gt_mask_batch.append(gt_mask)

            if len(rec_boxes) == 0:
                continue

            line_imgs = engine.CropAndResizeLines(
                img_rgb,
                rec_boxes,
                targetH=targetH,
                maxW=maxW,)

            if line_imgs.size(0) == 0:
                continue

            line_imgs_list.append(line_imgs.cpu())
            for text in rec_texts:
                ids = [int(engine.char2Idx[ch]) for ch in text]
                flat_targets.extend(ids)
                target_lengths.append(len(ids))
                norm_texts.append(text)

        if len(detect_imgs) == 0:
            return None

        detect_imgs_t = torch.stack(detect_imgs, dim=0).to(device)
        gt_shrink_t = torch.stack(gt_shrink_batch, dim=0).to(device)
        gt_thresh_t = torch.stack(gt_thresh_batch, dim=0).to(device)
        gt_mask_t = torch.stack(gt_mask_batch, dim=0).to(device)

        if len(line_imgs_list) == 0:
            recog_imgs_t = torch.empty(0, 1, targetH, maxW, device=device, dtype=detect_imgs_t.dtype)
            targets_t = torch.empty(0, dtype=torch.long, device=device)
            target_lengths_t = torch.empty(0, dtype=torch.long, device=device)
        else:
            recog_imgs_t = torch.cat(line_imgs_list, dim=0).to(device)
            targets_t = torch.tensor(flat_targets, dtype=torch.long, device=device)
            target_lengths_t = torch.tensor(target_lengths, dtype=torch.long, device=device)

        return (
            detect_imgs_t,
            gt_shrink_t,
            gt_thresh_t,
            gt_mask_t,
            recog_imgs_t,
            targets_t,
            target_lengths_t,
            norm_texts,
        )

    def SaveOCRParameters(self, engine: OCREngineExtractor, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ocr_state = {k: v.detach().cpu() for k, v in engine.state_dict().items()}
        brain_state = {f"OCR.{k}": v for k, v in ocr_state.items()}

        torch.save({
            "ocr": ocr_state,
            "brain": brain_state,
        }, str(out_path))

    def LoadOCRCheckpoint(
        self,
        engine: OCREngineExtractor,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        path: str,):
        ckpt = torch.load(path, map_location=self.device)

        if "ocr" in ckpt:
            engine.load_state_dict(ckpt["ocr"], strict=True)
        elif "brain" in ckpt:
            ocr_state = {
                k[len("OCR."):]: v
                for k, v in ckpt["brain"].items()
                if k.startswith("OCR.")
            }
            if len(ocr_state) == 0:
                raise KeyError(f"checkpoint {path} has no OCR weights")
            engine.load_state_dict(ocr_state, strict=False)
        else:
            raise KeyError(f"checkpoint {path} has no 'ocr' or 'brain' field")

        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

        if "rng" in ckpt:
            random.setstate(ckpt["rng"]["python"])
            torch.set_rng_state(ckpt["rng"]["torch"].cpu())
            np.random.set_state(ckpt["rng"]["numpy"])

        train_ds = val_ds = test_ds = None
        if ckpt.get("train_indices") is not None:
            train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
            val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
            if ckpt.get("test_indices") is not None:
                test_ds = torch.utils.data.Subset(dataset, ckpt["test_indices"])

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        return start_epoch, best_val, train_ds, val_ds, test_ds

    def OCRTrainLoop(
        self,
        root: str,
        epochs: int,
        batchSize: int,
        valSplit: float,
        resume: bool,
        onlineLearning: bool = False,
        *,
        ckptPath: str,
        outPath: str,):
        try:
            torch.autograd.set_detect_anomaly(True)

            self.controller.stop_requested = False
            self.controller.pause_requested = False
            self.controller.reset_hebbian = False

            del onlineLearning

            engine = OCREngineExtractor().to(self.device)

            ds = OfflineOCRDataset(root)
            optimizer = torch.optim.Adam(engine.parameters(), lr=1e-3)

            start_epoch = 0
            best_val = float("inf")
            train_ds = val_ds = test_ds = None

            testSplit = 0.1
            if resume and Path(ckptPath).exists():
                start_epoch, best_val, train_ds, val_ds, test_ds = self.LoadOCRCheckpoint(
                    engine, optimizer, ds, ckptPath)

            if train_ds is None:
                n_total = len(ds)
                n_test = int(n_total * testSplit)
                n_val = int(n_total * valSplit)
                n_train = n_total - n_val - n_test
                train_ds, val_ds, test_ds = torch.utils.data.random_split(
                    ds, [n_train, n_val, n_test])
            elif test_ds is None:
                train_indices = list(train_ds.indices) if hasattr(train_ds, "indices") else list(range(len(train_ds)))
                val_indices = list(val_ds.indices) if hasattr(val_ds, "indices") else []
                used = set(train_indices) | set(val_indices)
                test_indices = [idx for idx in range(len(ds)) if idx not in used]
                test_ds = torch.utils.data.Subset(ds, test_indices) if len(test_indices) > 0 else val_ds

            def collate_ocr_batch(batch):
                imgs, boxes, texts = zip(*batch)
                return list(imgs), list(boxes), list(texts)

            pin_memory = bool(getattr(self.device, "type", "") == "cuda")
            train_dl = DataLoader(
                train_ds,
                batch_size=batchSize,
                shuffle=True,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=collate_ocr_batch,)
            val_dl = DataLoader(
                val_ds,
                batch_size=batchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=collate_ocr_batch,)
            test_dl = DataLoader(
                test_ds,
                batch_size=batchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=collate_ocr_batch,)

            patience = 5
            min_delta = 1e-4
            no_improve = 0

            def eval_split(dl):
                engine.eval()
                split_loss = 0.0
                split_batches = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for imgs_b, boxes_b, texts_b in dl:
                        prepared = self.PrepareOCRBatch(
                            imgs_b,
                            boxes_b,
                            texts_b,
                            engine,
                            device=self.device,)
                        if prepared is None:
                            continue

                        (
                            detect_imgs_t,
                            gt_shrink_t,
                            gt_thresh_t,
                            gt_mask_t,
                            recog_imgs_t,
                            targets_t,
                            target_lengths_t,
                            norm_texts,
                        ) = prepared

                        det_out = engine.ForwardDetect(
                            detect_imgs_t,
                            gtShrink=gt_shrink_t,
                            gtThresh=gt_thresh_t,
                            gtMask=gt_mask_t,)
                        det_loss = det_out["loss"]

                        rec_loss = det_loss.new_zeros(())
                        if recog_imgs_t.size(0) > 0 and targets_t.numel() > 0:
                            rec_out = engine.ForwardRecognize(
                                recog_imgs_t,
                                targetsTensor=targets_t,
                                targetLengths=target_lengths_t,)
                            rec_loss = rec_out["loss"]

                            pairs = engine.CtcGreedyDecodeWithConf(
                                rec_out["log_probs"],
                                idx2Char=engine.idx2Char,
                                blankIndex=engine.blankIndex,)
                            pred_texts = [txt for txt, _ in pairs]
                            correct += sum(int(pred == target) for pred, target in zip(pred_texts, norm_texts))
                            total += len(norm_texts)

                        loss = det_loss + rec_loss
                        split_loss += float(loss.item())
                        split_batches += 1

                avg_loss = split_loss / max(1, split_batches)
                acc = correct / max(1, total)
                return avg_loss, acc

            self.controller.SetStatus(
                "training",
                "OCR training started",
                epoch=start_epoch,
                total_epochs=epochs,
                batch=0,
                total_batches=len(train_dl),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR training stopped")
                    break

                while self.controller.ShouldPause():
                    self.controller.SetStatus("paused", "OCR training paused")
                    time.sleep(0.2)

                engine.train()

                epoch_loss = 0.0
                nb = 0

                for bi, batch in enumerate(train_dl, start=1):
                    imgs_b, boxes_b, texts_b = batch
                    prepared = self.PrepareOCRBatch(
                        imgs_b,
                        boxes_b,
                        texts_b,
                        engine,
                        device=self.device,)
                    if prepared is None:
                        continue

                    (
                        detect_imgs_t,
                        gt_shrink_t,
                        gt_thresh_t,
                        gt_mask_t,
                        recog_imgs_t,
                        targets_t,
                        target_lengths_t,
                        norm_texts,
                    ) = prepared

                    det_out = engine.ForwardDetect(
                        detect_imgs_t,
                        gtShrink=gt_shrink_t,
                        gtThresh=gt_thresh_t,
                        gtMask=gt_mask_t,)
                    det_loss = det_out["loss"]

                    rec_loss = det_loss.new_zeros(())
                    rec_out = None
                    if recog_imgs_t.size(0) > 0 and targets_t.numel() > 0:
                        rec_out = engine.ForwardRecognize(
                            recog_imgs_t,
                            targetsTensor=targets_t,
                            targetLengths=target_lengths_t,)
                        rec_loss = rec_out["loss"]

                    loss = det_loss + rec_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
                    optimizer.step()

                    with torch.no_grad():
                        batch_acc = 0.0
                        if rec_out is not None and len(norm_texts) > 0:
                            pairs = engine.CtcGreedyDecodeWithConf(
                                rec_out["log_probs"].detach(),
                                idx2Char=engine.idx2Char,
                                blankIndex=engine.blankIndex,)
                            pred_texts = [txt for txt, _ in pairs]
                            batch_acc = (
                                sum(int(pred == target) for pred, target in zip(pred_texts, norm_texts))
                                / max(1, len(norm_texts)))

                    epoch_loss += float(loss.item())
                    nb += 1

                    self.controller.SetStatus(
                        "training",
                        f"OCR training... acc={batch_acc:.3f}",
                        epoch=ep + 1,
                        total_epochs=epochs,
                        batch=bi,
                        total_batches=len(train_dl),
                        train_loss=float(loss.item()),)

                    if self.controller.ShouldStop():
                        break
                    while self.controller.ShouldPause():
                        self.controller.SetStatus("paused", "OCR training paused")
                        time.sleep(0.2)

                avg_train = epoch_loss / max(1, nb)
                avg_val, val_acc = eval_split(val_dl)
                test_loss, test_acc = eval_split(test_dl)

                improved = (best_val - avg_val) > min_delta
                if improved:
                    best_val = avg_val
                    no_improve = 0
                    self.SaveOCRParameters(engine, outPath)

                    ckpt_dir = Path(ckptPath).parent
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "epoch": ep + 1,
                        "best_val": best_val,
                        "ocr": engine.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_indices": list(train_ds.indices) if hasattr(train_ds, "indices") else None,
                        "val_indices": list(val_ds.indices) if hasattr(val_ds, "indices") else None,
                        "test_indices": list(test_ds.indices) if hasattr(test_ds, "indices") else None,
                        "rng": {
                            "python": random.getstate(),
                            "torch": torch.get_rng_state(),
                            "numpy": np.random.get_state(),
                        },
                    }, ckptPath)
                else:
                    no_improve += 1

                self.controller.SetStatus(
                    "training",
                    (
                        f"OCR epoch {ep+1}/{epochs} done | "
                        f"train {avg_train:.4f} | "
                        f"val {avg_val:.4f}, acc={val_acc:.3f} | "
                        f"test {test_loss:.4f}, acc={test_acc:.3f}"
                    ),
                    epoch=ep + 1,
                    total_epochs=epochs,
                    val_loss=avg_val,)

                if no_improve >= patience:
                    self.controller.SetStatus("completed", "OCR validation stabilized, early stop.")
                    break

                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR training stopped")
                    break

            else:
                self.controller.SetStatus("completed", "OCR training completed")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"OCR training error: {e}", trace=tb)
        finally:
            self.is_begin = False

    def TrainLoop(self,root: str, epochs: int, batchSize: int, valSplit: float, resume: bool, onlineLearning = False, *, worldMemPath: str = None, memMemPath: str = None, ckptPath: str = None,):
        try:
            torch.autograd.set_detect_anomaly(True)
            ckptPath = ckptPath or BasicParameters.CKPT_PATH_TRAIN

            self.controller.stop_requested = False
            self.controller.pause_requested = False
            self.controller.reset_hebbian = False

            ds = OfflineGameDataset(root)

            brain = BrainCore(device=self.device, plasticHebbian=True, plasticOnlineLearning=onlineLearning, usePlanner=False,)

            agent = Agent(brain, isTrain=True, device=self.device, worldMemoryPath=worldMemPath, memMemoryPath=memMemPath,)

            if onlineLearning:
                agent.UpdateAllWrappers("autogrow")

            start_epoch = 0
            best_val = float("inf")
            train_ds, val_ds = None, None

            testSplit = 0.1

            if resume and Path(ckptPath).exists():
                start_epoch, best_val, train_ds, val_ds = self.LoadCheckpoint(brain, agent, ds, ckptPath)

            if train_ds is None:
                n_total = len(ds)
                n_test = int(n_total * testSplit)
                n_val = int(n_total * valSplit)
                n_train = n_total - n_val - n_test
                train_ds, val_ds, test_ds = torch.utils.data.random_split(
                    ds, [n_train, n_val, n_test])
            else:
                train_indices = list(train_ds.indices) if hasattr(train_ds, "indices") else list(range(len(train_ds)))
                val_indices = list(val_ds.indices) if hasattr(val_ds, "indices") else []
                used = set(train_indices) | set(val_indices)
                test_indices = [idx for idx in range(len(ds)) if idx not in used]
                test_ds = torch.utils.data.Subset(ds, test_indices) if len(test_indices) > 0 else val_ds

            train_dl = DataLoader(train_ds, batch_size=batchSize, shuffle=False, num_workers=0, pin_memory=True,)
            val_dl = DataLoader(val_ds, batch_size=batchSize, shuffle=False, num_workers=0)
            test_dl = DataLoader(test_ds, batch_size=batchSize, shuffle=False, num_workers=0)

            mse = nn.MSELoss()

            patience = 5 
            min_delta = 1e-4 
            no_improve = 0
            target_acc = 0.90 
            max_gap = 0.1

            def unpack_batch(batch):
                img_b, key_b, mouse_click_b, mouse_move_b, reward_b, done_b, ext_text_b = batch

                if ext_text_b is not None:
                    if isinstance(ext_text_b, tuple):
                        ext_text_b = list(ext_text_b)
                    elif isinstance(ext_text_b, list):
                        ext_text_b = ext_text_b
                    else:
                        ext_text_b = [ext_text_b]

                    ext_text_b = [
                        None if (t is None or str(t).strip() == "") else str(t)
                        for t in ext_text_b]

                return img_b, key_b, mouse_click_b, mouse_move_b, reward_b, done_b, ext_text_b

            def compute_supervised_loss_and_metrics(
                key_pred: Optional[torch.Tensor],
                click_pred: Optional[torch.Tensor],
                mouse_move_pred: Optional[torch.Tensor],
                keys_t: Optional[torch.Tensor],
                mouse_click_t: Optional[torch.Tensor],
                mouse_move_t: Optional[torch.Tensor],):

                key_targets = keys_t.float()
                click_targets = mouse_click_t.float()
                move_targets = mouse_move_t.float()

                cur_loss = (
                    mse(key_pred.float(), key_targets)
                    + mse(click_pred.float(), click_targets)
                    + 0.05 * mse(mouse_move_pred.float(), move_targets))

                total_correct = float((key_pred.float() == key_targets).float().sum().item())
                total_correct += float((click_pred.float() == click_targets).float().sum().item())
                total_elems = int(key_targets.numel() + click_targets.numel())

                return cur_loss, total_correct, total_elems

            self.controller.SetStatus("training", "Training started", epoch=start_epoch, total_epochs=epochs, batch=0, total_batches=len(train_dl),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

                while self.controller.ShouldPause():
                    self.controller.SetStatus("paused", "Training paused")
                    time.sleep(0.2)

                brain.train()
                epoch_loss = 0.0
                nb = 0

                agent.ResetBrainState(B=batchSize, isOnlineLearning=onlineLearning)

                for bi, batch in enumerate(train_dl, start=1):
                    img_b, key_b, mouse_click_b, mouse_move_b, reward_b, done_b, ext_text_b = unpack_batch(batch)

                    pack = agent.ConvertNpImagesKeysMouses(
                        imgs=img_b,
                        keys=key_b,
                        mouseClick=mouse_click_b,
                        mouseMove=mouse_move_b,
                        reward=reward_b,
                        done=done_b,
                        device=self.device,)
                    
                    frames = pack["frames"]
                    keys_t = pack["keys"]
                    mouse_click_t = pack["mouse_clicks"]
                    mouse_move_t = pack["mouse_moves"]
                    reward_t = pack["rewards"]
                    done_t = pack["dones"]

                    if self.controller.ShouldResetHebbian():
                        agent.ResetHebbianMemory()
                        self.controller.RequestCancelResetHebbian()

                    act_out = agent.Act(
                        frames,
                        textExt=ext_text_b,
                        reward=reward_t,
                        done=done_t,
                        sampleActions=True,
                        deterministicActor=False,)
                    
                    if act_out is None:
                        continue

                    key_pred, click_pred, mouse_move_pred, model_loss = act_out
                    bc_loss, _, _ = compute_supervised_loss_and_metrics(
                        key_pred,
                        click_pred,
                        mouse_move_pred,
                        keys_t,
                        mouse_click_t,
                        mouse_move_t,)

                    loss = model_loss + bc_loss

                    agent.opt_world.zero_grad(set_to_none=True)
                    agent.opt_critic.zero_grad(set_to_none=True)
                    agent.opt_actor.zero_grad(set_to_none=True)

                    loss.backward()
                    if onlineLearning:
                        agent.UpdateAllWrappers("accumulategrads")
                        agent.UpdateAllWrappers("autogrow")

                    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)

                    for name, p in brain.named_parameters():
                        if not p.requires_grad:
                            continue
                        if p.grad is None:
                            print("NO GRAD:", name)
                        elif not torch.isfinite(p.grad).all():
                            print("BAD GRAD:", name, p.grad.min(), p.grad.max(),)

                    agent.opt_world.step()
                    agent.opt_critic.step()
                    agent.opt_actor.step()

                    epoch_loss += float(loss.item())
                    nb += 1

                    self.controller.SetStatus("training", "Training...", epoch=ep + 1, total_epochs=epochs, batch=bi, total_batches=len(train_dl), train_loss=float(loss.item()),)

                    if self.controller.ShouldStop():
                        break
                    while self.controller.ShouldPause():
                        self.controller.SetStatus("paused", "Training paused")
                        time.sleep(0.2)

                avg_train = epoch_loss / max(1, nb)

                self.controller.SetStatus("training", f"Epoch {ep+1}/{epochs} done, avg_train={avg_train:.4f}", epoch=ep + 1, total_epochs=epochs,)

                if self.controller.ShouldStop():
                    break

                def eval_split(dl):
                    brain.eval()
                    agent.ResetBrainState(B=batchSize, isOnlineLearning=onlineLearning)
                    split_loss = 0.0
                    split_batches = 0
                    total_correct = 0
                    total_elems = 0

                    with torch.no_grad():
                        for batch in dl:
                            img_b, key_b, mouse_click_b, mouse_move_b, reward_b, done_b, ext_text_b = unpack_batch(batch)

                            v_pack = agent.ConvertNpImagesKeysMouses(
                                imgs=img_b,
                                keys=key_b,
                                mouseClick=mouse_click_b,
                                mouseMove=mouse_move_b,
                                reward=reward_b,
                                done=done_b,
                                device=self.device,)
                            
                            v_frames = v_pack["frames"]
                            v_keys_t = v_pack["keys"]
                            v_mouse_click_t = v_pack["mouse_clicks"]
                            v_mouse_move_t = v_pack["mouse_moves"]

                            act_out = agent.Act(
                                v_frames,
                                textExt=ext_text_b,
                                reward=None,
                                done=None,
                                sampleActions=True,
                                deterministicActor=True,)
                            
                            if act_out is None:
                                continue

                            v_key_pred, v_click_pred, v_mouse_move_pred = act_out
                            cur_loss, correct, elems = compute_supervised_loss_and_metrics(
                                v_key_pred,
                                v_click_pred,
                                v_mouse_move_pred,
                                v_keys_t,
                                v_mouse_click_t,
                                v_mouse_move_t,)
                            
                            total_correct += correct
                            total_elems += elems

                            split_loss += float(cur_loss.item())
                            split_batches += 1

                    avg_split_loss = split_loss / max(1, split_batches)
                    split_acc = (total_correct / total_elems if total_elems > 0 else 0.0)
                    return avg_split_loss, split_acc

                avg_val, val_acc = eval_split(val_dl)
                test_loss, test_acc = eval_split(test_dl)

                improved = (best_val - avg_val) > min_delta
                if improved:
                    best_val = avg_val
                    no_improve = 0
                    agent.SaveRuntimeMemories()

                    ckpt = {
                        "epoch": ep + 1,
                        "best_val": best_val,
                        "brain": brain.state_dict(),
                        "opt_actor": agent.opt_actor.state_dict(),
                        "opt_critic": agent.opt_critic.state_dict(),
                        "opt_world": agent.opt_world.state_dict(),
                        "train_indices": list(train_ds.indices)
                        if hasattr(train_ds, "indices")
                        else None,
                        "val_indices": list(val_ds.indices)
                        if hasattr(val_ds, "indices")
                        else None,
                        "test_indices": list(test_ds.indices)
                        if hasattr(test_ds, "indices")
                        else None,
                        "rng": {
                            "python": random.getstate(),
                            "torch": torch.get_rng_state(),
                            "numpy": np.random.get_state(),},
                        "buffers": brain.ExportBuffers(),}
                    torch.save(ckpt, ckptPath)
                else:
                    no_improve += 1

                self.controller.SetStatus(
                    "training",
                    (f"Epoch {ep+1}/{epochs} done | " f"train {avg_train:.4f} | " f"val {avg_val:.4f}, acc={val_acc:.3f} | " f"test {test_loss:.4f}, acc={test_acc:.3f}"), val_loss=avg_val,)

                if (val_acc >= target_acc and test_acc >= target_acc and abs(val_acc - test_acc) <= max_gap):
                    self.controller.SetStatus("completed", f"Val/Test accuracies high & close: val={val_acc:.3f}, test={test_acc:.3f}",)
                    if onlineLearning:
                        agent.UpdateAllWrappers("commit")
                    break

                if no_improve >= patience:
                    self.controller.SetStatus("completed", "Validation stabilized, early stop.")
                    break

                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

            else:
                self.controller.SetStatus("completed", "Training completed")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Training error: {e}", trace=tb)
        finally:
            self.is_begin = False




    def ExportParamsFromCheckpoint(
        self,
        overwrite: bool = False,
        *,
        ckptPath: str = BasicParameters.CKPT_PATH_TRAIN,
        outPath: str = BasicParameters.MODULEPARAMETER_PATH,) -> None:

        ckpt_path = Path(ckptPath)
        out_path = Path(outPath)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        raw = torch.load(str(ckpt_path), map_location="cpu")
        if "brain" not in raw:
            raise KeyError(f"checkpoint {ckpt_path} has no 'brain' field")

        params = {"brain": raw["brain"],}

        if out_path.exists() and not overwrite:
            stem = out_path.stem
            suffix = out_path.suffix
            parent = out_path.parent

            idx = 1
            while True:
                cand = parent / f"{stem}_{idx}{suffix}"
                if not cand.exists():
                    out_path = cand
                    print(f"[ExportParamsOnly] target exists, save to {out_path} instead")
                    break
                idx += 1

        torch.save(params, str(out_path))
        print(f"[ExportParamsOnly] saved params to {out_path}")


    def LoadCheckpoint(self, brain: BrainCore, agent: Agent, dataset: Dataset, path: str = None):
        ckpt = torch.load(path,  map_location=self.device)
        brain.load_state_dict(ckpt["brain"])
        agent.opt_actor.load_state_dict(ckpt["opt_actor"])
        agent.opt_critic.load_state_dict(ckpt["opt_critic"])
        agent.opt_world.load_state_dict(ckpt["opt_world"])

        brain.ImportBuffers(ckpt["buffers"])
        random.setstate(ckpt["rng"]["python"])
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        np.random.set_state(ckpt["rng"]["numpy"])

        if ckpt.get("train_indices") is not None:
            train_ds = torch.utils.data.Subset(dataset, ckpt["train_indices"])
            val_ds = torch.utils.data.Subset(dataset, ckpt["val_indices"])
        else:
            train_ds = val_ds = None

        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        return start_epoch, best_val, train_ds, val_ds

    def StartDeployment(self, cameraIndex: int = 0, useHebbian: bool = True, usePlanner: bool = True):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.is_begin = True
        self.controller.stop_requested = False 
        self.br_thread = threading.Thread(target=self.DeployLoop,args=(cameraIndex,),kwargs={"useHebbian": useHebbian, "usePlanner": usePlanner},daemon=False,)
        self.br_thread.start()
        return True


    def DeployLoop(self, cameraIndex: int,* ,useHebbian: bool = True, usePlanner: bool = True,):
        try:
            brain = BrainCore(device=self.device,plasticHebbian=useHebbian,plasticOnlineLearning=False,usePlanner=usePlanner,)

            model_path = BasicParameters.MODULEPARAMETER_PATH

            if os.path.exists(model_path):
                try:
                    sd = torch.load(model_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    print(f"Safe mode loading failed: {e}, try the normal mode")
                    sd = torch.load(model_path, map_location=self.device)
            else:
                msg = f"The module file is not exit: {model_path}"
                print(msg)
                sd = None
                self.controller.SetStatus("error", msg)
                return 

            if isinstance(sd, dict) and "brain" in sd:
                brain.load_state_dict(sd["brain"], strict=False)
            else:
                brain.load_state_dict(sd, strict=False)

            brain.to(self.device)
            brain.eval()

            agent = Agent(brain,isTrain=False,device=self.device,worldMemoryPath=BasicParameters.WORLD_MEMORY_PATH,memMemoryPath=BasicParameters.MEMORY_MEMORY_PATH,)
            agent.ResetBrainState(isOnlineLearning=False)

            seq_len = brain.SEQ_LEN
            rm_len = BasicParameters.IMAGE_RM_LEN

            if iio is None:
                raise RuntimeError("imageio.v3 cant use")

            self.controller.SetStatus("is_begin", "Deployment started")

            code_to_name: dict[int, str] = {}
            all_codes = []
            for name, code in RAW_KEYBOARD_LAYOUT.items():
                code_to_name.setdefault(code, name)
                all_codes.append(code)
            max_code = max(all_codes)

            frame_buf: List[np.ndarray] = []

            with iio.imopen(f"<video{cameraIndex}>", "r") as cam:
                for frame_np in cam:
                    if self.controller.ShouldStop():
                        break

                    frame_buf.append(frame_np)

                    if len(frame_buf) < seq_len:
                        continue

                    pack = agent.StackNpImagesKeysMouses(imgs=frame_buf,B=1, T=seq_len,device=self.device,)
                    frames = pack["frames"]

                    if self.controller.ShouldResetHebbian():
                        agent.ResetHebbianMemory()
                        self.controller.RequestCancelResetHebbian()

                    keys, clicks, mouse = agent.Act(frames,reward=None,done=None,sampleActions=True,deterministicActor=True,)

                    kv = keys[0].detach().cpu()
                    ck = clicks[0].detach().cpu()
                    ms = mouse[0].detach().cpu()

                    pressed_names: list[str] = []
                    for code in range(max_code + 1):
                        if float(kv[code]) > 0.5:
                            pressed_names.append(code_to_name.get(code, f"Key{code}"))

                    if float(ck[0]) > 0.5:
                        pressed_names.append("MouseLeft")
                    if float(ck[1]) > 0.5:
                        pressed_names.append("MouseRight")

                    names_str = "[" + ", ".join(pressed_names) + "]" if pressed_names else "[]"

                    self.controller.SetStatus("is_begin",(f"keys:{names_str} "f"mouse:dx={float(ms[0]):.3f},dy={float(ms[1]):.3f}"),)

                    drop_n = len(frame_buf) - rm_len
                    del frame_buf[:drop_n]

            self.controller.SetStatus("stopped", "Deployment stopped")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Deployment error: {e}", trace=tb)
        finally:
            self.is_begin = False



    def TestPerceptionModule(self):
        t = self.test["perception"]
        return t.RunAll()

    def TestAttentionModule(self):
        t = self.test["attention"]
        return t.RunAll()

    def TestMemoryModule(self):
        t = self.test["memory"]
        return t.RunAll()

    def TestDecisionModule(self):
        t = self.test["decision"]
        return t.RunAll()

    def TestWorldModule(self):
        t = self.test["world"]
        return t.RunAll()

    def TestValueEstimationModule(self):
        t = self.test["value"]
        return t.RunAll()
    
    def TestConsciousnessModule(self):
        t = self.test["consciousness"]
        return t.RunAll()
    
    def TestIntentionModule(self):
        t = self.test["intention"]
        return t.RunAll()
    
    def TestOCRModule(self):
        t = self.test["OCR"]
        return t.RunAll()
    

    def MonitorTraining(self):
        try:
            while True:
                st = self.GetCurrentStatus()
                print(
                    f"[TRAIN] {st['state']} | epoch {st['epoch']}/{st['total_epochs']} "
                    f"| batch {st['batch']}/{st['total_batches']} "
                    f"| train_loss={st['train_loss']:.4f} | msg={st['message']}")

                if st["state"] == "error":
                    trace = st.get("trace")
                    if trace:
                        print("\n====== TRAIN ERROR TRACEBACK ======\n")
                        print(trace)
                        print("===================================\n")

                if st["state"] in ("completed", "stopped", "error"):
                    self.controller.ResteStatus()
                    break

                time.sleep(1)

        except Exception as e:
            print(f"[MonitorTraining] monitor raised: {e}")
            print(traceback.format_exc())


    def MonitorDeployment(self):
        try:
            while True:
                st = self.GetCurrentStatus()
                print(f"[DEPLOY] {st['state']} | msg={st['message']}")

                if st["state"] == "error":
                    trace = st.get("trace")
                    if trace:
                        print("\n====== DEPLOY ERROR TRACEBACK ======\n")
                        print(trace)
                        print("====================================\n")

                if st["state"] in ("stopped", "error"):
                    self.controller.ResteStatus()
                    break

                time.sleep(0.5)

        except Exception as e:
            print(f"[MonitorDeployment] monitor raised: {e}")
            print(traceback.format_exc())


    def TestModuleTrain(
        self,
        onlineLearning: bool,
        *,
        dataRoot: str = BasicParameters.DATA_ROOT_PATH_TEST,
        nSamples: int = 64,
        epochs: int = 1,
        batchSize: int = 1,
        valSplit: float = 0.2,
        seed: int = 42,) -> Dict[str, Any]:
        if self.is_begin:
            return {"False": False, "msg": "StartTraining returns False (training may already be running)"} 

        try:
            if iio is None:
                raise RuntimeError("imageio.v3 error")

            rng = np.random.default_rng(seed)
            root = Path(dataRoot)
            if root.exists():
                shutil.rmtree(root)
            (root / "frames").mkdir(parents=True, exist_ok=True)
            (root / "keys").mkdir(parents=True, exist_ok=True)
            (root / "mouse_click").mkdir(parents=True, exist_ok=True)
            (root / "mouse_move").mkdir(parents=True, exist_ok=True)
            (root / "reward").mkdir(parents=True, exist_ok=True)
            (root / "done").mkdir(parents=True, exist_ok=True)
            (root / "texts").mkdir(parents=True, exist_ok=True)

            all_codes = list(RAW_KEYBOARD_LAYOUT.values())
            max_code = max(all_codes)
            keys_dim = max_code + 1

            H, W = BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE

            templates = ["move left", "move right", "move forward", "move back",
                        "use skill", "defend", "attack", "pickup item",
                        "open menu", "retreat",]

            for i in range(nSamples):
                img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
                iio.imwrite(str(root / "frames" / f"{i:05d}.png"), img)

                keys = np.zeros((keys_dim,), dtype=np.float32)
                for code in all_codes:
                    keys[code] = 1.0 if rng.random() < 0.05 else 0.0

                np.save(str(root / "keys" / f"{i:05d}.npy"), keys)

                mouse_click = np.zeros((2,), dtype=np.float32)
                mouse_click[0] = 1.0 if rng.random() < 0.15 else 0.0
                mouse_click[1] = 1.0 if rng.random() < 0.05 else 0.0
                np.save(str(root / "mouse_click" / f"{i:05d}.npy"), mouse_click)

                mouse_move = rng.normal(loc=0.0, scale=2.0, size=(2,)).astype(np.float32)
                np.save(str(root / "mouse_move" / f"{i:05d}.npy"), mouse_move)

                reward = rng.normal(loc=0.0, scale=2.0, size=(1,)).astype(np.float32)
                np.save(str(root / "reward" / f"{i:05d}.npy"), reward)

                done = rng.normal(loc=0.0, scale=2.0, size=(1,)).astype(np.float32)
                np.save(str(root / "done" / f"{i:05d}.npy"), done)


                if rng.random() < 0.35:
                    ext_text = ""
                else:
                    n_words = int(rng.integers(1, 4)) 
                    picks = rng.choice(templates, size=n_words, replace=False).tolist()
                    ext_text = " | ".join(picks)
                with open(root / "texts" / f"{i:05d}.txt", "w", encoding="utf-8") as f:
                    f.write(ext_text)

            print("[SmokeTest] start train...")

            ok = self.StartTraining(epochs=epochs,batchSize=batchSize,valSplit=valSplit,resume=False, onlineLearning=onlineLearning, isTest=True)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"False": False, "msg": "StartTraining returns False (training may already be running)"}

            self.message_thread = threading.Thread(target=self.MonitorTraining,args=(),daemon=False,)
            self.message_thread.start()

            #self.TrainLoop(root, epochs, batchSize, valSplit, False, onlineLearning, worldMemPath=BasicParameters.WORLD_MEMORY_PATH_TEST, memMemPath=BasicParameters.MEMORY_MEMORY_PATH_TEST,ckptPath=BasicParameters.CKPT_PATH_TEST)
        
            return {"ok": True}

        except Exception as e:
            print(f"TestModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def TrainOCRModule(
        self,
        onlineLearning: bool,
        epochs: int = 6,
        batchSize: int = 8,
        valSplit: float = 0.2,
        isResume: bool = False,
        *,
        dataRoot: str = BasicParameters.DATA_ROOT_PATH,) -> Dict[str, Any]:
        try:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            boxes_dir = root / "boxes"
            texts_dir = root / "texts"

            if not frames_dir.exists() or not boxes_dir.exists() or not texts_dir.exists():
                print(f"[TrainOCR] no dataset found at {root}, please prepare frames/boxes/texts first.")
                return {"ok": False, "msg": "no ocr dataset"}

            if not any(frames_dir.glob("*.png")) or not any(boxes_dir.glob("*.npy")) or not any(texts_dir.glob("*.txt")):
                print(f"[TrainOCR] no OCR samples found at {root}.")
                return {"ok": False, "msg": "empty ocr dataset"}

            is_test_root = Path(dataRoot) == Path(BasicParameters.DATA_ROOT_PATH_TEST)
            ckpt_path = (
                "BrainDeepLearn/TestData/ocr_training_checkpoint.pth"
                if is_test_root
                else "BrainDeepLearn/Data/ocr_training_checkpoint.pth")
            out_path = (
                "BrainDeepLearn/TestData/ocr_module_parameter.pth"
                if is_test_root
                else "BrainDeepLearn/Data/ocr_module_parameter.pth")

            print(f"[TrainOCR] use existing OCR dataset at: {root}")

            ok = self.StartOCRTraining(
                str(root),
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                onlineLearning=onlineLearning,
                ckptPath=ckpt_path,
                outPath=out_path,)

            if not ok:
                print("StartOCRTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.message_thread = threading.Thread(
                target=self.MonitorTraining,
                args=(),
                daemon=False,)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"TrainOCRModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def TestOCRModuleTrain(self, onlineLearning: bool) -> Dict[str, Any]:
        return self.TrainOCRModule(
            onlineLearning=onlineLearning,
            epochs=1,
            batchSize=4,
            valSplit=0.2,
            isResume=False,
            dataRoot=BasicParameters.DATA_ROOT_PATH_TEST,)

    def TrainModule(
        self,
        onlineLearning: bool,
        epochs: int = 6,
        batchSize: int = 1,
        valSplit: float = 0.2,
        isResume: bool = False,
        *,
        dataRoot: str = BasicParameters.DATA_ROOT_PATH,) -> Dict[str, Any]:
        try:
            root = Path(dataRoot)

            def has_existing_data(p: Path) -> bool:
                frames_dir = p / "frames"
                keys_dir = p / "keys"
                mouse_click_dir = p / "mouse_click"
                mouse_move_dir = p / "mouse_move"
                reward_dir = p / "reward"
                done_dir = p / "done"
                if not (frames_dir.exists() and keys_dir.exists() and mouse_click_dir.exists() and mouse_move_dir.exists() and reward_dir.exists() and done_dir.exists()):
                    return False
                if not any(frames_dir.glob("*.png")):
                    return False
                return True

            if not has_existing_data(root):
                print(f"[Train] no dataset found at {root}, please prepare frames/keys/mouse_click/mouse_move/reward/done first.")
                return {"ok": False, "msg": "no dataset"}

            print(f"[Train] use existing dataset at: {root}")

            ok = self.StartTraining(epochs=epochs, batchSize=batchSize, valSplit=valSplit, resume=isResume, onlineLearning=onlineLearning,isTest=False,)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"False": False, "msg": "StartTraining returns False (training may already be running)"}

            self.message_thread = threading.Thread(target=self.MonitorTraining,args=(),daemon=False,)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"ModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def DeployModule(
        self,
        cameraIndex: int = 0,
        useHebbian: bool = True,
        usePlanner: bool = True,) -> Dict[str, Any]:
        try:
            ok = self.StartDeployment(cameraIndex=cameraIndex, useHebbian=useHebbian, usePlanner=usePlanner,)

            if not ok:
                print("StartDeployment returns False (deployment may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.message_thread = threading.Thread(target=self.MonitorDeployment,args=(),daemon=False,)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"DeployModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
