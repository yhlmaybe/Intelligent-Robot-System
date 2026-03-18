from __future__ import annotations
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
import threading
import random
import time

import numpy as np
import torch
import torch.nn as nn
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
from DataPreprocess import (
    DataPreprocessor,
    OfflineGameDataset,
    OfflineOCRDataset,
    OfflineOCRRecognitionDataset,
)
from AGICore import Agent, BrainCore, BasicParameters

try:
    import imageio.v3 as iio
except Exception:
    iio = None  


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
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        isTest: bool = False,
        *,
        trainDetection: bool = True,
        trainRecognition: bool = True,
        root: Optional[str] = None,):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.is_begin = True

        ckpt_path = BasicParameters.OCR_CKPT_PATH_TEST if isTest else BasicParameters.OCR_CKPT_PATH_TRAIN
        out_path = BasicParameters.OCR_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_MODULEPARAMETER_PATH
        recognizer_init_path = BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH
        root = root or (BasicParameters.DATA_ROOT_PATH_TEST if isTest else BasicParameters.DATA_ROOT_PATH)

        self.br_thread = threading.Thread(
            target=self.OCRTrainLoop,
            args=(root, epochs, batchSize, valSplit, resume),
            kwargs={
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "trainDetection": trainDetection,
                "trainRecognition": trainRecognition,
                "recognizerInitPath": recognizer_init_path,},

            daemon=False)
        self.br_thread.start()
        return True

    def StartOCRRecognitionTraining(
        self,
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        isTest: bool = False,
        *,
        root: Optional[str] = None,):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.is_begin = True

        ckpt_path = BasicParameters.OCR_RECOGNIZER_CKPT_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_CKPT_PATH_TRAIN
        out_path = BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH
        root = root or (BasicParameters.OCR_RECOGNIZER_DATA_ROOT_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_DATA_ROOT_PATH)

        self.br_thread = threading.Thread(
            target=self.OCRRecognitionTrainLoop,
            args=(root, epochs, batchSize, valSplit, resume),
            kwargs={
                "ckptPath": ckpt_path,
                "outPath": out_path,},
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



    def SaveOCRParameters(self, engine: OCREngineExtractor, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ocr_state = {k: v.detach().cpu() for k, v in engine.state_dict().items()}
        brain_state = {f"OCR.{k}": v for k, v in ocr_state.items()}

        torch.save({
            "ocr": ocr_state,
            "brain": brain_state,}, str(out_path))

    def SaveOCRRecognizerParameters(self, engine: OCREngineExtractor, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rec_state = {k: v.detach().cpu() for k, v in engine.recognizer.state_dict().items()}
        ocr_state = {f"recognizer.{k}": v for k, v in rec_state.items()}
        brain_state = {f"OCR.recognizer.{k}": v for k, v in rec_state.items()}

        torch.save({
            "recognizer": rec_state,
            "ocr": ocr_state,
            "brain": brain_state,}, str(out_path))

    def LoadOCRWeightsIntoEngine(self, engine: OCREngineExtractor, path: str) -> None:
        payload = torch.load(path, map_location=self.device)

        ocr_state = None
        if isinstance(payload, dict):
            if "ocr" in payload:
                ocr_state = payload["ocr"]
            elif "brain" in payload:
                ocr_state = {
                    k[len("OCR."):]: v
                    for k, v in payload["brain"].items()
                    if k.startswith("OCR.")}
            elif all(str(k).startswith(("backbone.", "dbHead.", "recognizer.")) for k in payload.keys()):
                ocr_state = payload

        if not ocr_state:
            raise KeyError(f"checkpoint {path} has no OCR weights")

        engine.load_state_dict(ocr_state, strict=False)

    def LoadRecognizerWeightsIntoEngine(self, engine: OCREngineExtractor, path: str) -> None:
        payload = torch.load(path, map_location=self.device)

        rec_state = None
        if isinstance(payload, dict):
            if "recognizer" in payload:
                rec_state = payload["recognizer"]
            elif "ocr" in payload:
                rec_state = {
                    k[len("recognizer."):]: v
                    for k, v in payload["ocr"].items()
                    if k.startswith("recognizer.")}
            elif "brain" in payload:
                rec_state = {
                    k[len("OCR.recognizer."):]: v
                    for k, v in payload["brain"].items()
                    if k.startswith("OCR.recognizer.")}
            elif all(str(k).startswith("recognizer.") for k in payload.keys()):
                rec_state = {k[len("recognizer."):]: v for k, v in payload.items()}
            elif all(not str(k).startswith("backbone.") and not str(k).startswith("dbHead.") for k in payload.keys()):
                rec_state = payload

        if not rec_state:
            raise KeyError(f"checkpoint {path} has no recognizer weights")

        engine.recognizer.load_state_dict(rec_state, strict=False)

    def LoadOCRCheckpoint(
        self,
        engine: OCREngineExtractor,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        path: str,
        *,
        allowOptimizerMismatch: bool = False,):
        ckpt = torch.load(path, map_location=self.device)

        if "ocr" in ckpt:
            engine.load_state_dict(ckpt["ocr"], strict=True)
        elif "brain" in ckpt:
            ocr_state = {
                k[len("OCR."):]: v
                for k, v in ckpt["brain"].items()
                if k.startswith("OCR.")}
            if len(ocr_state) == 0:
                raise KeyError(f"checkpoint {path} has no OCR weights")
            engine.load_state_dict(ocr_state, strict=False)
        else:
            raise KeyError(f"checkpoint {path} has no 'ocr' or 'brain' field")

        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                if not allowOptimizerMismatch:
                    raise

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

    def LoadOCRRecognizerCheckpoint(
        self,
        engine: OCREngineExtractor,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset,
        path: str,):
        ckpt = torch.load(path, map_location=self.device)

        if "recognizer" in ckpt:
            engine.recognizer.load_state_dict(ckpt["recognizer"], strict=True)
        elif "ocr" in ckpt:
            rec_state = {
                k[len("recognizer."):]: v
                for k, v in ckpt["ocr"].items()
                if k.startswith("recognizer.")}
            if len(rec_state) == 0:
                raise KeyError(f"checkpoint {path} has no recognizer weights")
            engine.recognizer.load_state_dict(rec_state, strict=False)
        elif "brain" in ckpt:
            rec_state = {
                k[len("OCR.recognizer."):]: v
                for k, v in ckpt["brain"].items()
                if k.startswith("OCR.recognizer.")}
            if len(rec_state) == 0:
                raise KeyError(f"checkpoint {path} has no recognizer weights")
            engine.recognizer.load_state_dict(rec_state, strict=False)
        else:
            raise KeyError(f"checkpoint {path} has no recognizer field")

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
        *,
        ckptPath: str,
        outPath: str,
        trainDetection: bool = True,
        trainRecognition: bool = True,
        recognizerInitPath: Optional[str] = None,):
        try:
            torch.autograd.set_detect_anomaly(True)

            self.controller.stop_requested = False
            self.controller.pause_requested = False
            self.controller.reset_hebbian = False

            if not trainDetection and not trainRecognition:
                raise ValueError("OCRTrainLoop requires trainDetection or trainRecognition to be True")

            ds = OfflineOCRDataset(root)
            engine = OCREngineExtractor().to(self.device)

            if recognizerInitPath and Path(recognizerInitPath).exists() and not (resume and Path(ckptPath).exists()):
                self.LoadRecognizerWeightsIntoEngine(engine, recognizerInitPath)
            elif not trainRecognition and not (resume and Path(ckptPath).exists()):
                init_candidates = [Path(outPath), Path(ckptPath)]
                init_error = None
                loaded = False
                for init_path in init_candidates:
                    if not init_path.exists():
                        continue
                    try:
                        self.LoadOCRWeightsIntoEngine(engine, str(init_path))
                        loaded = True
                        break
                    except Exception as e:
                        init_error = e

                if not loaded:
                    msg = (
                        "detect-only OCR training requires existing OCR weights "
                        "or recognizerInitPath to preserve recognizer parameters")
                    if init_error is not None:
                        raise RuntimeError(msg) from init_error
                    raise FileNotFoundError(msg)

            for p in engine.backbone.parameters():
                p.requires_grad = trainDetection
            for p in engine.dbHead.parameters():
                p.requires_grad = trainDetection
            for p in engine.recognizer.parameters():
                p.requires_grad = trainRecognition

            trainable_params = [p for p in engine.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise RuntimeError("no trainable OCR parameters selected")
            optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

            start_epoch = 0
            best_val = float("inf")
            train_ds = val_ds = test_ds = None

            testSplit = 0.1
            if resume and Path(ckptPath).exists():
                start_epoch, best_val, train_ds, val_ds, test_ds = self.LoadOCRCheckpoint(
                    engine,
                    optimizer,
                    ds,
                    ckptPath,
                    allowOptimizerMismatch=True)

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
                imgs, boxes, texts, ignore_flags = zip(*batch)
                return list(imgs), list(boxes), list(texts), list(ignore_flags)

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
                split_samples = 0
                total_correct = 0
                total_elems = 0

                with torch.no_grad():
                    for imgs_b, boxes_b, texts_b, ignore_b in dl:
                        for img, boxes, texts, ignore_flags in zip(imgs_b, boxes_b, texts_b, ignore_b):
                            sample: Dict[str, Any] = DataPreprocessor.PrepareOCRSample(
                                img, 
                                boxes,
                                texts,
                                ignoreFlags=ignore_flags,
                                char2Idx=engine.char2Idx,
                                imageSize=BasicParameters.IMAGE_SIZE,
                                targetH=32,
                                maxW=256,
                                device=self.device,)

                            zero = sample["detect_img"].new_zeros(())
                            det_loss = zero
                            if trainDetection:
                                detect_img = sample["detect_img"].unsqueeze(0) # [1, 3, H, W]
                                gt_shrink = sample["gt_shrink"].unsqueeze(0) # [1, 1, H, W]
                                gt_thresh = sample["gt_thresh"].unsqueeze(0) # [1, 1, H, W]
                                gt_mask = sample["gt_mask"].unsqueeze(0) # [1, 1, H, W]

                                det_out = engine.ForwardDetect(
                                    detect_img,
                                    gtShrink=gt_shrink,
                                    gtThresh=gt_thresh,
                                    gtMask=gt_mask,)
                                det_loss = det_out["loss"] # []

                            rec_loss = zero # []
                            recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                            targets = sample["targets"] # [sum(target_lengths)]
                            target_lengths = sample["target_lengths"] # [N_line]
                            norm_texts = sample["norm_texts"]

                            if trainRecognition and recog_imgs.size(0) > 0 and targets.numel() > 0:
                                rec_out = engine.ForwardRecognize(
                                    recog_imgs,
                                    targetsTensor=targets,
                                    targetLengths=target_lengths,)
                                rec_loss = rec_out["loss"] # []

                                pairs = engine.CtcGreedyDecodeWithConf(
                                    rec_out["log_probs"], # [T, N_line, C_vocab]
                                    idx2Char=engine.idx2Char,
                                    blankIndex=engine.blankIndex,)
                                pred_texts = [txt for txt, _ in pairs]
                                total_correct += sum(int(pred == target) for pred, target in zip(pred_texts, norm_texts))
                                total_elems += len(norm_texts)

                            split_loss += float((det_loss + rec_loss).item())
                            split_samples += 1

                avg_split_loss = split_loss / max(1, split_samples)
                split_acc = (total_correct / total_elems if total_elems > 0 else 0.0)
                return avg_split_loss, split_acc

            self.controller.SetStatus(
                "training",(
                    "OCR training started"
                    if trainDetection and trainRecognition else
                    "OCR training started (detect only)"
                    if trainDetection else
                    "OCR training started (recognize only)"),
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
                    imgs_b, boxes_b, texts_b, ignore_b = batch
                    sample_losses: List[torch.Tensor] = []
                    batch_correct = 0
                    batch_elems = 0

                    for img, boxes, texts, ignore_flags in zip(imgs_b, boxes_b, texts_b, ignore_b):
                        sample: Dict[str, Any] = DataPreprocessor.PrepareOCRSample(
                            img,  
                            boxes, 
                            texts,
                            ignoreFlags=ignore_flags,
                            char2Idx=engine.char2Idx,
                            imageSize=BasicParameters.IMAGE_SIZE,
                            targetH=32,
                            maxW=256,
                            device=self.device,)

                        zero = sample["detect_img"].new_zeros(())
                        det_loss = zero
                        if trainDetection:
                            detect_img = sample["detect_img"].unsqueeze(0) # [1, 3, H, W]
                            gt_shrink = sample["gt_shrink"].unsqueeze(0) # [1, 1, H, W]
                            gt_thresh = sample["gt_thresh"].unsqueeze(0) # [1, 1, H, W]
                            gt_mask = sample["gt_mask"].unsqueeze(0) # [1, 1, H, W]

                            det_out = engine.ForwardDetect(
                                detect_img,
                                gtShrink=gt_shrink,
                                gtThresh=gt_thresh,
                                gtMask=gt_mask,)
                            det_loss = det_out["loss"] # []

                        rec_loss = zero # []
                        recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                        targets = sample["targets"] # [sum(target_lengths)]
                        target_lengths = sample["target_lengths"] # [N_line]
                        norm_texts = sample["norm_texts"]

                        if trainRecognition and recog_imgs.size(0) > 0 and targets.numel() > 0:
                            rec_out = engine.ForwardRecognize(
                                recog_imgs,
                                targetsTensor=targets,
                                targetLengths=target_lengths,)
                            rec_loss = rec_out["loss"] # []

                            with torch.no_grad():
                                pairs = engine.CtcGreedyDecodeWithConf(
                                    rec_out["log_probs"].detach(), # [T, N_line, C_vocab]
                                    idx2Char=engine.idx2Char,
                                    blankIndex=engine.blankIndex,)
                                pred_texts = [txt for txt, _ in pairs]
                                batch_correct += sum(int(pred == target) for pred, target in zip(pred_texts, norm_texts))
                                batch_elems += len(norm_texts)

                        sample_losses.append(det_loss + rec_loss)

                    if len(sample_losses) == 0:
                        continue

                    loss = torch.stack(sample_losses).mean() # []
                    rec_acc = (batch_correct / batch_elems) if batch_elems > 0 else 0.0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    nb += 1

                    self.controller.SetStatus(
                        "training",
                        (
                            f"OCR training... rec_acc={rec_acc:.3f}"
                            if trainRecognition else
                            "OCR training..."),
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
                            "numpy": np.random.get_state(),},}, ckptPath)
                else:
                    no_improve += 1

                self.controller.SetStatus(
                    "training",(
                        f"OCR epoch {ep+1}/{epochs} done | "
                        f"train {avg_train:.4f} | "
                        f"val {avg_val:.4f}, acc={val_acc:.3f} | "
                        f"test {test_loss:.4f}, acc={test_acc:.3f}"),
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

    def OCRRecognitionTrainLoop(
        self,
        root: str,
        epochs: int,
        batchSize: int,
        valSplit: float,
        resume: bool,
        *,
        ckptPath: str,
        outPath: str,):
        try:
            torch.autograd.set_detect_anomaly(True)

            self.controller.stop_requested = False
            self.controller.pause_requested = False
            self.controller.reset_hebbian = False

            ds = OfflineOCRRecognitionDataset(root)
            engine = OCREngineExtractor().to(self.device)
            for p in engine.backbone.parameters():
                p.requires_grad = False
            for p in engine.dbHead.parameters():
                p.requires_grad = False
            for p in engine.recognizer.parameters():
                p.requires_grad = True

            optimizer = torch.optim.Adam(engine.recognizer.parameters(), lr=1e-3)

            start_epoch = 0
            best_val = float("inf")
            train_ds = val_ds = test_ds = None

            testSplit = 0.1
            if resume and Path(ckptPath).exists():
                start_epoch, best_val, train_ds, val_ds, test_ds = self.LoadOCRRecognizerCheckpoint(
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

            def collate_rec_batch(batch):
                imgs, texts, ignore_flags = zip(*batch)
                return list(imgs), list(texts), list(ignore_flags)

            pin_memory = bool(getattr(self.device, "type", "") == "cuda")
            train_dl = DataLoader(
                train_ds,
                batch_size=batchSize,
                shuffle=True,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=collate_rec_batch,)

            val_dl = DataLoader(
                val_ds,
                batch_size=batchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=collate_rec_batch,)

            test_dl = DataLoader(
                test_ds,
                batch_size=batchSize,
                shuffle=False,
                num_workers=0,
                pin_memory=pin_memory,
                collate_fn=collate_rec_batch,)

            patience = 5
            min_delta = 1e-4
            no_improve = 0

            def eval_split(dl):
                engine.eval()
                split_loss = 0.0
                split_samples = 0
                total_correct = 0
                total_elems = 0

                with torch.no_grad():
                    for imgs_b, texts_b, ignore_b in dl:
                        for img, text, ignore_flag in zip(imgs_b, texts_b, ignore_b):
                            sample = DataPreprocessor.PrepareOCRRecognitionSample(
                                img,
                                text,
                                ignoreFlag=ignore_flag,
                                char2Idx=engine.char2Idx,
                                targetH=32,
                                maxW=256,
                                device=self.device,)

                            recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                            targets = sample["targets"] # [sum(target_lengths)]
                            target_lengths = sample["target_lengths"] # [N_line]
                            norm_text = sample["norm_text"]
                            if recog_imgs.size(0) == 0 or targets.numel() == 0:
                                continue

                            rec_out = engine.ForwardRecognize(
                                recog_imgs,
                                targetsTensor=targets,
                                targetLengths=target_lengths,)
                            rec_loss = rec_out["loss"] # []

                            pairs = engine.CtcGreedyDecodeWithConf(
                                rec_out["log_probs"], # [T, N_line, C_vocab]
                                idx2Char=engine.idx2Char,
                                blankIndex=engine.blankIndex,)
                            pred_text = pairs[0][0] if len(pairs) > 0 else ""
                            total_correct += int(pred_text == norm_text)
                            total_elems += 1
                            split_loss += float(rec_loss.item())
                            split_samples += 1

                avg_split_loss = split_loss / max(1, split_samples)
                split_acc = (total_correct / total_elems if total_elems > 0 else 0.0)
                return avg_split_loss, split_acc

            self.controller.SetStatus(
                "training",
                "OCR recognizer training started",
                epoch=start_epoch,
                total_epochs=epochs,
                batch=0,
                total_batches=len(train_dl),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR recognizer training stopped")
                    break

                while self.controller.ShouldPause():
                    self.controller.SetStatus("paused", "OCR recognizer training paused")
                    time.sleep(0.2)

                engine.train()
                epoch_loss = 0.0
                nb = 0

                for bi, batch in enumerate(train_dl, start=1):
                    imgs_b, texts_b, ignore_b = batch
                    sample_losses: List[torch.Tensor] = []
                    batch_correct = 0
                    batch_elems = 0

                    for img, text, ignore_flag in zip(imgs_b, texts_b, ignore_b):
                        sample = DataPreprocessor.PrepareOCRRecognitionSample(
                            img,
                            text,
                            ignoreFlag=ignore_flag,
                            char2Idx=engine.char2Idx,
                            targetH=32,
                            maxW=256,
                            device=self.device,)

                        recog_imgs = sample["recog_imgs"] # [N_line, 1, targetH, maxW]
                        targets = sample["targets"] # [sum(target_lengths)]
                        target_lengths = sample["target_lengths"] # [N_line]
                        norm_text = sample["norm_text"]
                        if recog_imgs.size(0) == 0 or targets.numel() == 0:
                            continue

                        rec_out = engine.ForwardRecognize(
                            recog_imgs,
                            targetsTensor=targets,
                            targetLengths=target_lengths,)
                        rec_loss = rec_out["loss"] # []

                        with torch.no_grad():
                            pairs = engine.CtcGreedyDecodeWithConf(
                                rec_out["log_probs"].detach(), # [T, N_line, C_vocab]
                                idx2Char=engine.idx2Char,
                                blankIndex=engine.blankIndex,)
                            pred_text = pairs[0][0] if len(pairs) > 0 else ""
                            batch_correct += int(pred_text == norm_text)
                            batch_elems += 1

                        sample_losses.append(rec_loss)

                    if len(sample_losses) == 0:
                        continue

                    loss = torch.stack(sample_losses).mean() # []
                    rec_acc = (batch_correct / batch_elems) if batch_elems > 0 else 0.0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(engine.recognizer.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    nb += 1

                    self.controller.SetStatus(
                        "training",
                        f"OCR recognizer training... acc={rec_acc:.3f}",
                        epoch=ep + 1,
                        total_epochs=epochs,
                        batch=bi,
                        total_batches=len(train_dl),
                        train_loss=float(loss.item()),)

                    if self.controller.ShouldStop():
                        break

                    while self.controller.ShouldPause():
                        self.controller.SetStatus("paused", "OCR recognizer training paused")
                        time.sleep(0.2)

                avg_train = epoch_loss / max(1, nb)
                avg_val, val_acc = eval_split(val_dl)
                test_loss, test_acc = eval_split(test_dl)

                improved = (best_val - avg_val) > min_delta
                if improved:
                    best_val = avg_val
                    no_improve = 0
                    self.SaveOCRRecognizerParameters(engine, outPath)

                    ckpt_dir = Path(ckptPath).parent
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "epoch": ep + 1,
                        "best_val": best_val,
                        "recognizer": engine.recognizer.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "train_indices": list(train_ds.indices) if hasattr(train_ds, "indices") else None,
                        "val_indices": list(val_ds.indices) if hasattr(val_ds, "indices") else None,
                        "test_indices": list(test_ds.indices) if hasattr(test_ds, "indices") else None,
                        "rng": {
                            "python": random.getstate(),
                            "torch": torch.get_rng_state(),
                            "numpy": np.random.get_state(),},}, ckptPath)
                else:
                    no_improve += 1

                self.controller.SetStatus(
                    "training",(
                        f"OCR recognizer epoch {ep+1}/{epochs} done | "
                        f"train {avg_train:.4f} | "
                        f"val {avg_val:.4f}, acc={val_acc:.3f} | "
                        f"test {test_loss:.4f}, acc={test_acc:.3f}"),
                    epoch=ep + 1,
                    total_epochs=epochs,
                    val_loss=avg_val,)

                if no_improve >= patience:
                    self.controller.SetStatus("completed", "OCR recognizer validation stabilized, early stop.")
                    break

                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR recognizer training stopped")
                    break

            else:
                self.controller.SetStatus("completed", "OCR recognizer training completed")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"OCR recognizer training error: {e}", trace=tb)
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
                            v_reward_t = v_pack["rewards"]
                            v_done_t = v_pack["dones"]

                            act_out = agent.Act(
                                v_frames,
                                textExt=ext_text_b,
                                reward=v_reward_t,
                                done=v_done_t,
                                sampleActions=True,
                                deterministicActor=True,)
                            
                            if act_out is None:
                                continue

                            v_key_pred, v_click_pred, v_mouse_move_pred, v_model_loss = act_out
                            bc_loss, correct, elems = compute_supervised_loss_and_metrics(
                                v_key_pred,
                                v_click_pred,
                                v_mouse_move_pred,
                                v_keys_t,
                                v_mouse_click_t,
                                v_mouse_move_t,)
                            cur_loss = v_model_loss + bc_loss
                            
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

            """ok = self.StartTraining(epochs=epochs,batchSize=batchSize,valSplit=valSplit,resume=False, onlineLearning=onlineLearning, isTest=True)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"False": False, "msg": "StartTraining returns False (training may already be running)"}

            self.message_thread = threading.Thread(target=self.MonitorTraining,args=(),daemon=False,)
            self.message_thread.start()"""

            self.TrainLoop(root, epochs, batchSize, valSplit, False, onlineLearning, worldMemPath=BasicParameters.WORLD_MEMORY_PATH_TEST, memMemPath=BasicParameters.MEMORY_MEMORY_PATH_TEST,ckptPath=BasicParameters.CKPT_PATH_TEST)
        
            return {"ok": True}

        except Exception as e:
            print(f"TestModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise



    def TestOCRModuleTrain(
        self,
        *,
        dataRoot: str = BasicParameters.OCR_DATA_ROOT_PATH_TEST,
        nSamples: int = 32,
        epochs: int = 1,
        batchSize: int = 4,
        valSplit: float = 0.2,
        seed: int = 42,) -> Dict[str, Any]:
        if self.is_begin:
            return {"ok": False, "msg": "StartOCRTraining returns False (training may already be running)"}

        try:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            ocr_texts_dir = root / "OCRTexts"

            def has_existing_ocr_data(p: Path) -> bool:
                texts_dir = p / "OCRTexts"
                return (
                    (p / "frames").exists()
                    and texts_dir.exists()
                    and any((p / "frames").glob("*.png"))
                    and any(texts_dir.glob("*.txt"))
                    and (
                        any((p / "boxes").glob("*.npy"))
                        or any(DataPreprocessor.TextFileLooksLikeOCRAnnotations(tp) for tp in texts_dir.glob("*.txt"))))

            if not has_existing_ocr_data(root):
                if iio is None:
                    raise RuntimeError("imageio.v3 error")

                rng = np.random.default_rng(seed)
                if root.exists():
                    shutil.rmtree(root)
                frames_dir.mkdir(parents=True, exist_ok=True)
                ocr_texts_dir.mkdir(parents=True, exist_ok=True)

                vocab_path = Path("BrainDeepLearn/ModuleSetting/OCRKeys.txt")
                if not vocab_path.exists():
                    raise FileNotFoundError(f"OCR vocab file not found: {vocab_path}")

                vocab_chars = [
                    line.strip()
                    for line in vocab_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
                if len(vocab_chars) == 0:
                    raise RuntimeError(f"OCR vocab file is empty: {vocab_path}")
                safe_vocab_chars = [
                    ch for ch in vocab_chars
                    if ch not in {"\"", ",", "\n", "\r", "#"}]
                if len(safe_vocab_chars) == 0:
                    safe_vocab_chars = list("ABC123")

                h_img = BasicParameters.IMAGE_SIZE
                w_img = BasicParameters.IMAGE_SIZE

                for i in range(nSamples):
                    img = rng.integers(0, 32, size=(h_img, w_img, 3), dtype=np.uint8)
                    n_lines = int(rng.integers(1, 5))
                    anno_lines: List[str] = []

                    top_margin = 24
                    bottom_margin = 24
                    usable_h = max(80, h_img - top_margin - bottom_margin)
                    lane_h = max(28, usable_h // max(1, n_lines))

                    for li in range(n_lines):
                        line_h = int(rng.integers(24, min(56, lane_h + 1)))
                        y1_base = top_margin + li * lane_h
                        y1_jitter = int(rng.integers(0, max(1, lane_h - line_h + 1)))
                        y1 = min(h_img - line_h - 1, y1_base + y1_jitter)
                        y2 = min(h_img, y1 + line_h)

                        x1 = int(rng.integers(16, 64))
                        line_w = int(rng.integers(96, min(360, w_img - x1)))
                        x2 = min(w_img, x1 + line_w)

                        bg_val = int(rng.integers(120, 220))
                        img[y1:y2, x1:x2, :] = bg_val

                        text_len = int(rng.integers(2, 9))
                        text = "".join(rng.choice(safe_vocab_chars, size=text_len, replace=True).tolist())
                        ignore_flag = 1 if rng.random() < 0.25 else 0
                        if ignore_flag:
                            chars = list(text)
                            n_mask = max(1, min(len(chars), int(rng.integers(1, len(chars) + 1))))
                            for pos in rng.choice(len(chars), size=n_mask, replace=False).tolist():
                                chars[pos] = "#"
                            text = "".join(chars)
                        anno_lines.append(
                            f'{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{ignore_flag},"{text}"')

                        inner_y1 = min(y2 - 1, y1 + 4)
                        inner_y2 = max(inner_y1 + 1, y2 - 4)
                        cursor_x = x1 + 6
                        char_w = max(6, (max(1, x2 - x1 - 12)) // max(1, text_len))
                        visible_text = text.replace("#", safe_vocab_chars[0])
                        for ch in visible_text:
                            stripe_w = max(2, char_w // 3)
                            glyph_h = max(4, inner_y2 - inner_y1)
                            tone = int((hash(ch) % 120) + 40)
                            gx1 = min(x2 - 1, cursor_x + int(rng.integers(0, max(1, char_w - stripe_w + 1))))
                            gx2 = min(x2, gx1 + stripe_w)
                            gy1 = inner_y1
                            gy2 = min(inner_y2, gy1 + glyph_h)
                            img[gy1:gy2, gx1:gx2, :] = tone
                            cursor_x += char_w
                            if cursor_x >= x2 - 4:
                                break

                    iio.imwrite(str(frames_dir / f"{i:05d}.png"), img)
                    ocr_texts_dir.joinpath(f"{i:05d}.txt").write_text("\n".join(anno_lines), encoding="utf-8")

                print(f"[TestOCR] created random OCR dataset at: {root}")
            else:
                print(f"[TestOCR] use existing OCR dataset at: {root}")

            ok = self.StartOCRTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=False,
                isTest=True,
                trainDetection=True,
                trainRecognition=True,
                root=str(root),)

            if not ok:
                print("StartOCRTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.message_thread = threading.Thread(target=self.MonitorTraining, args=(), daemon=False)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"TestOCRModuleTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def TestOCRRecognitionTrain(
        self,
        *,
        dataRoot: str = BasicParameters.OCR_RECOGNIZER_DATA_ROOT_PATH_TEST,
        nSamples: int = 64,
        epochs: int = 1,
        batchSize: int = 8,
        valSplit: float = 0.2,
        seed: int = 42,) -> Dict[str, Any]:
        if self.is_begin:
            return {"ok": False, "msg": "StartOCRRecognitionTraining returns False (training may already be running)"}

        try:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

            def has_existing_rec_data(p: Path) -> bool:
                return (
                    (p / "frames").exists()
                    and (p / "OCRTexts").exists()
                    and any((p / "frames").glob("*.png"))
                    and any((p / "OCRTexts").glob("*.txt")))

            if not has_existing_rec_data(root):
                if iio is None:
                    raise RuntimeError("imageio.v3 error")

                rng = np.random.default_rng(seed)
                if root.exists():
                    shutil.rmtree(root)
                frames_dir.mkdir(parents=True, exist_ok=True)
                texts_dir.mkdir(parents=True, exist_ok=True)

                vocab_path = Path("BrainDeepLearn/ModuleSetting/OCRKeys.txt")
                if not vocab_path.exists():
                    raise FileNotFoundError(f"OCR vocab file not found: {vocab_path}")

                vocab_chars = [
                    line.strip()
                    for line in vocab_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
                if len(vocab_chars) == 0:
                    raise RuntimeError(f"OCR vocab file is empty: {vocab_path}")

                safe_vocab_chars = [
                    ch for ch in vocab_chars
                    if ch not in {"\"", ",", "\n", "\r", "#"}]
                if len(safe_vocab_chars) == 0:
                    safe_vocab_chars = list("ABC123")

                target_h = 32

                for i in range(nSamples):
                    text_len = int(rng.integers(2, 10))
                    text = "".join(rng.choice(safe_vocab_chars, size=text_len, replace=True).tolist())
                    ignore_flag = 1 if rng.random() < 0.2 else 0
                    stored_text = text
                    if ignore_flag:
                        chars = list(stored_text)
                        n_mask = max(1, min(len(chars), int(rng.integers(1, len(chars) + 1))))
                        for pos in rng.choice(len(chars), size=n_mask, replace=False).tolist():
                            chars[pos] = "#"
                        stored_text = "".join(chars)

                    img_w = int(rng.integers(72, 220))
                    img = np.full((target_h, img_w, 3), int(rng.integers(185, 235)), dtype=np.uint8)

                    cursor_x = 4
                    char_w = max(6, (img_w - 8) // max(1, text_len))
                    visible_text = stored_text.replace("#", safe_vocab_chars[0])
                    for ch in visible_text:
                        stripe_w = max(2, char_w // 3)
                        tone = int((hash(ch) % 120) + 40)
                        gx1 = min(img_w - 1, cursor_x + int(rng.integers(0, max(1, char_w - stripe_w + 1))))
                        gx2 = min(img_w, gx1 + stripe_w)
                        gy1 = int(rng.integers(4, 10))
                        gy2 = min(target_h, gy1 + int(rng.integers(16, 24)))
                        img[gy1:gy2, gx1:gx2, :] = tone
                        cursor_x += char_w
                        if cursor_x >= img_w - 4:
                            break

                    label_line = (
                        f'0,0,{img_w},0,{img_w},{target_h},0,{target_h},{ignore_flag},"{stored_text}"')

                    iio.imwrite(str(frames_dir / f"{i:05d}.png"), img)
                    texts_dir.joinpath(f"{i:05d}.txt").write_text(label_line, encoding="utf-8")

                print(f"[TestOCRRec] created random OCR recognition dataset at: {root}")
            else:
                print(f"[TestOCRRec] use existing OCR recognition dataset at: {root}")

            ok = self.StartOCRRecognitionTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=False,
                isTest=True,
                root=str(root),)

            if not ok:
                print("StartOCRRecognitionTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.message_thread = threading.Thread(target=self.MonitorTraining, args=(), daemon=False)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"TestOCRRecognitionTrain failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


    def TrainOCRModule(
        self,
        epochs: int = 6,
        batchSize: int = 4,
        valSplit: float = 0.2,
        isResume: bool = False,
        *,
        trainDetection: bool = True,
        trainRecognition: bool = True,
        dataRoot: str = BasicParameters.OCR_DATA_ROOT_PATH,) -> Dict[str, Any]:
        try:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            boxes_dir = root / "boxes"
            ocr_texts_dir = root / "OCRTexts"

            if not (trainDetection or trainRecognition):
                print("[TrainOCR] trainDetection and trainRecognition cannot both be False.")
                return {"ok": False, "msg": "no_train_target"}

            txt_files = sorted(ocr_texts_dir.glob("*.txt")) if ocr_texts_dir.exists() else []
            has_box_files = boxes_dir.exists() and any(boxes_dir.glob("*.npy"))
            has_text_annotations = any(DataPreprocessor.TextFileLooksLikeOCRAnnotations(p) for p in txt_files)

            if not (frames_dir.exists() and ocr_texts_dir.exists()):
                print(f"[TrainOCR] no dataset found at {root}, please prepare frames/OCRTexts first.")
                return {"ok": False, "msg": "no ocr dataset"}

            if not any(frames_dir.glob("*.png")) or not txt_files:
                print(f"[TrainOCR] no OCR samples found at {root}.")
                return {"ok": False, "msg": "empty ocr dataset"}

            if not has_box_files and not has_text_annotations:
                print(f"[TrainOCR] OCR labels at {root} need either boxes/*.npy or annotation txt lines with coords/flag/text.")
                return {"ok": False, "msg": "invalid ocr labels"}

            print(f"[TrainOCR] use existing OCR dataset at: {root}")

            ok = self.StartOCRTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                isTest=False,
                trainDetection=trainDetection,
                trainRecognition=trainRecognition,
                root=str(root),)

            if not ok:
                print("StartOCRTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.message_thread = threading.Thread(target=self.MonitorTraining, args=(), daemon=False)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"TrainOCRModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def TrainOCRRecognitionModule(
        self,
        onlineLearning: bool,
        epochs: int = 6,
        batchSize: int = 8,
        valSplit: float = 0.2,
        isResume: bool = False,
        *,
        dataRoot: str = BasicParameters.OCR_RECOGNIZER_DATA_ROOT_PATH,) -> Dict[str, Any]:
        try:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

            if not (frames_dir.exists() and texts_dir.exists()):
                print(f"[TrainOCRRec] no dataset found at {root}, please prepare frames/OCRTexts first.")
                return {"ok": False, "msg": "no ocr recognition dataset"}

            if not any(frames_dir.glob("*.png")) or not any(texts_dir.glob("*.txt")):
                print(f"[TrainOCRRec] no OCR recognition samples found at {root}.")
                return {"ok": False, "msg": "empty ocr recognition dataset"}

            print(f"[TrainOCRRec] use existing OCR recognition dataset at: {root}")

            ok = self.StartOCRRecognitionTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                isTest=False,
                root=str(root),)

            if not ok:
                print("StartOCRRecognitionTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.message_thread = threading.Thread(target=self.MonitorTraining, args=(), daemon=False)
            self.message_thread.start()

            return {"ok": True}

        except Exception as e:
            print(f"TrainOCRRecognitionModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise


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
