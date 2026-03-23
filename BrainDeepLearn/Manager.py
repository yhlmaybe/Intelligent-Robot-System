from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Optional, Union
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
from DataPreprocess import DataPreprocessor, DataResizeMeta, OfflineGameDataset, OfflineOCRDataset, OfflineOCRRecognitionDataset
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
            "trace": "",
            "visual": self.EmptyVisualStatus(),}
        self.stop_requested = False
        self.pause_requested = False
        self.reset_hebbian = False
        self.visual_state_enabled = True

    def EmptyVisualStatus(self, *, touch: bool = False) -> Dict[str, Any]:
        return {
            "bitmap": [],
            "text": "",
            "ocr_texts": [],
            "items": [],
            "updated_at": (time.time() if touch else 0.0),}

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
            "trace": "",
            "visual": self.EmptyVisualStatus(),}

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

    def SetVisualStateEnabled(self, enabled: bool):
        with self._lock:
            self.visual_state_enabled = bool(enabled)
            if not self.visual_state_enabled:
                self.status["visual"] = self.EmptyVisualStatus(touch=True)

    def IsVisualStateEnabled(self) -> bool:
        with self._lock:
            return self.visual_state_enabled


class ManagerFunction:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.controller = ModuleController()

        self.br_thread: Optional[threading.Thread] = None
        self.message_thread: Optional[threading.Thread] = None
        self.is_begin = False
        self.overrideCheckpointWithModuleParams = True

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

    def SetBasicParameters(self, name: str, value: str):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        return BasicParameters.Set(name=name, value=value)
    
    def GetBasicParameters(self, name: str):
        return BasicParameters.Get(name=name)
    
    def GetBasicParametersDict(self):
        return BasicParameters.GetStringDict()

    def SetOverrideCheckpointWithModuleParams(self, enabled: bool):
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False
        self.overrideCheckpointWithModuleParams = bool(enabled)
        return True

    def StartBackgroundTask(
        self,
        target: Callable[..., Any],
        *,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,) -> bool:
        if self.is_begin:
            self.controller.SetStatus("recur", "Training or Deploy is already running")
            return False

        self.is_begin = True
        self.br_thread = threading.Thread(
            target=target,
            args=args,
            kwargs=(kwargs or {}),
            daemon=False)
        self.br_thread.start()
        return True

    def StartTraining(
        self,
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        onlineLearning: bool = False,
        isTest: bool = False,
        *,
        overrideCheckpointWithModuleParams: Optional[bool] = None,):
        ckpt_path = BasicParameters.CKPT_PATH_TEST if isTest else BasicParameters.CKPT_PATH_TRAIN
        out_path = BasicParameters.MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.MODULEPARAMETER_PATH
        wm_mem_path = BasicParameters.WORLD_MEMORY_PATH_TEST if isTest else BasicParameters.WORLD_MEMORY_PATH
        mem_mem_path = BasicParameters.MEMORY_MEMORY_PATH_TEST if isTest else BasicParameters.MEMORY_MEMORY_PATH
        override_enabled = bool(self.overrideCheckpointWithModuleParams) if overrideCheckpointWithModuleParams is None else bool(overrideCheckpointWithModuleParams)

        return self.StartBackgroundTask(
            self.TrainLoop,
            args=(epochs, batchSize, valSplit, resume, onlineLearning),
            kwargs={
                "worldMemPath": wm_mem_path,
                "memMemPath": mem_mem_path,
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "isTest": isTest,
                "overrideCheckpointWithModuleParams": override_enabled,})

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
        overrideCheckpointWithModuleParams: Optional[bool] = None,):
        ckpt_path = BasicParameters.OCR_CKPT_PATH_TEST if isTest else BasicParameters.OCR_CKPT_PATH_TRAIN
        out_path = BasicParameters.OCR_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_MODULEPARAMETER_PATH
        recognizer_init_path = BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH
        override_enabled = bool(self.overrideCheckpointWithModuleParams) if overrideCheckpointWithModuleParams is None else bool(overrideCheckpointWithModuleParams)

        return self.StartBackgroundTask(
            self.OCRTrainLoop,
            args=(epochs, batchSize, valSplit, resume),
            kwargs={
                "isTest": isTest,
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "trainDetection": trainDetection,
                "trainRecognition": trainRecognition,
                "recognizerInitPath": recognizer_init_path,
                "overrideCheckpointWithModuleParams": override_enabled,})

    def StartOCRRecognitionTraining(
        self,
        epochs: int = 5,
        batchSize: int = 32,
        valSplit: float = 0.1,
        resume: bool = True,
        isTest: bool = False,
        *,
        overrideCheckpointWithModuleParams: Optional[bool] = None):
        ckpt_path = BasicParameters.OCR_RECOGNIZER_CKPT_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_CKPT_PATH_TRAIN
        out_path = BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.OCR_RECOGNIZER_MODULEPARAMETER_PATH
        override_enabled = bool(self.overrideCheckpointWithModuleParams) if overrideCheckpointWithModuleParams is None else bool(overrideCheckpointWithModuleParams)

        return self.StartBackgroundTask(
            self.OCRRecognitionTrainLoop,
            args=(epochs, batchSize, valSplit, resume),
            kwargs={
                "isTest": isTest,
                "ckptPath": ckpt_path,
                "outPath": out_path,
                "overrideCheckpointWithModuleParams": override_enabled,})

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

    def SetVisualStateEnabled(self, enabled: bool):
        self.controller.SetVisualStateEnabled(enabled)
        return True

    def GetVisualStateEnabled(self):
        return self.controller.IsVisualStateEnabled()

    def ResetControllerFlags(self) -> None:
        with self.controller._lock:
            self.controller.stop_requested = False
            self.controller.pause_requested = False
            self.controller.reset_hebbian = False

    def WaitWhilePaused(self, pausedMessage: str) -> None:
        while self.controller.ShouldPause():
            self.controller.SetStatus("paused", pausedMessage)
            time.sleep(2)

    def StartMessageMonitor(self, monitorFn: Callable[[], None]) -> None:
        self.message_thread = threading.Thread(target=monitorFn, args=(), daemon=False)
        self.message_thread.start()

    def RunNamedTest(self, testKey: str):
        return self.test[str(testKey)].RunAll()

    def CreateDataLoader(
        self,
        dataset: Dataset,
        *,
        batchSize: int,
        shuffle: bool,
        collateFn: Optional[Callable[..., Any]] = None,
        pinMemory: Optional[bool] = None,) -> DataLoader:
        loader_kwargs: Dict[str, Any] = {
            "batch_size": batchSize,
            "shuffle": shuffle,
            "num_workers": 0,
            "pin_memory": (
                bool(getattr(self.device, "type", "") == "cuda")
                if pinMemory is None else
                bool(pinMemory)),}
        if collateFn is not None:
            loader_kwargs["collate_fn"] = collateFn
        return DataLoader(dataset, **loader_kwargs)

    def SplitDataset(
        self,
        dataset: Dataset,
        *,
        valSplit: float,
        testSplit: float = 0.1,
        trainDataset: Optional[Dataset] = None,
        valDataset: Optional[Dataset] = None,
        testDataset: Optional[Dataset] = None,):
        if trainDataset is None:
            n_total = len(dataset)
            n_test = int(n_total * testSplit)
            n_val = int(n_total * valSplit)
            n_train = n_total - n_val - n_test
            trainDataset, valDataset, testDataset = torch.utils.data.random_split(
                dataset,
                [n_train, n_val, n_test])
        elif testDataset is None:
            train_indices = list(trainDataset.indices) if hasattr(trainDataset, "indices") else list(range(len(trainDataset)))
            val_indices = list(valDataset.indices) if hasattr(valDataset, "indices") else []
            used = set(train_indices) | set(val_indices)
            test_indices = [idx for idx in range(len(dataset)) if idx not in used]
            testDataset = torch.utils.data.Subset(dataset, test_indices) if test_indices else valDataset

        return trainDataset, valDataset, testDataset

    def HasGameDataset(self, dataRoot: Optional[Union[str, Path]] = None) -> bool:
        if dataRoot is None:
            frames_dir = Path(BasicParameters.DATA_FRAMES_PATH)
            keys_dir = Path(BasicParameters.DATA_KEYS_PATH)
            mouse_click_dir = Path(BasicParameters.DATA_MOUSE_CLICK_PATH)
            mouse_move_dir = Path(BasicParameters.DATA_MOUSE_MOVE_PATH)
            reward_dir = Path(BasicParameters.DATA_REWARD_PATH)
            done_dir = Path(BasicParameters.DATA_DONE_PATH)
            texts_dir = Path(BasicParameters.DATA_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            keys_dir = root / "keys"
            mouse_click_dir = root / "mouse_click"
            mouse_move_dir = root / "mouse_move"
            reward_dir = root / "reward"
            done_dir = root / "done"
            texts_dir = root / "texts"

        required_dirs = [frames_dir, keys_dir, mouse_click_dir, mouse_move_dir, reward_dir, done_dir]
        if not all(p.exists() for p in required_dirs):
            return False

        counts = [
            len(sorted(frames_dir.glob("*.png"))),
            len(sorted(keys_dir.glob("*.npy"))),
            len(sorted(mouse_click_dir.glob("*.npy"))),
            len(sorted(mouse_move_dir.glob("*.npy"))),
            len(sorted(reward_dir.glob("*.npy"))),
            len(sorted(done_dir.glob("*.npy"))),]
        if counts[0] == 0 or len(set(counts)) != 1:
            return False

        if texts_dir.exists():
            text_count = len(sorted(texts_dir.glob("*.txt")))
            if text_count not in (0, counts[0]):
                return False
        return True

    def HasOcrDataset(self, dataRoot: Optional[Union[str, Path]] = None) -> bool:
        if dataRoot is None:
            root = Path(BasicParameters.OCR_DATA_ROOT_PATH)
            frames_dir = Path(BasicParameters.OCR_FRAMES_PATH)
            texts_dir = Path(BasicParameters.OCR_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

        if not (frames_dir.exists() and texts_dir.exists()):
            return False

        frame_files = sorted(frames_dir.glob("*.png"))
        txt_files = sorted(texts_dir.glob("*.txt"))
        if not frame_files or not txt_files or len(frame_files) != len(txt_files):
            return False

        boxes_dir = root / "boxes"
        return any(boxes_dir.glob("*.npy")) or any(
            DataPreprocessor.TextFileLooksLikeOCRAnnotations(txt_path)
            for txt_path in txt_files)

    def HasOcrRecognitionDataset(self, dataRoot: Optional[Union[str, Path]] = None) -> bool:
        if dataRoot is None:
            frames_dir = Path(BasicParameters.OCR_RECOGNIZER_FRAMES_PATH)
            texts_dir = Path(BasicParameters.OCR_RECOGNIZER_TEXTS_PATH)
        else:
            root = Path(dataRoot)
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

        if not (frames_dir.exists() and texts_dir.exists()):
            return False

        frame_files = sorted(frames_dir.glob("*.png"))
        text_files = sorted(texts_dir.glob("*.txt"))
        return bool(frame_files) and bool(text_files) and len(frame_files) == len(text_files)

    def MonitorStatus(
        self,
        *,
        prefix: str,
        monitorName: str,
        renderStatus: Callable[[Dict[str, Any]], str],
        terminalStates: Tuple[str, ...],
        sleepSeconds: float,) -> None:
        try:
            while True:
                st = self.GetCurrentStatus()
                print(f"[{prefix}] {renderStatus(st)}")

                if st["state"] == "error":
                    trace = st.get("trace")
                    if trace:
                        print(f"\n====== {prefix} ERROR TRACEBACK ======\n")
                        print(trace)
                        print("====================================\n")

                if st["state"] in terminalStates:
                    self.controller.ResteStatus()
                    break

                time.sleep(sleepSeconds)

        except Exception as e:
            print(f"[{monitorName}] monitor raised: {e}")
            print(traceback.format_exc())

    def RestoreOcrItemsToOriginal(
        self,
        ocrItems: Optional[List[Dict[str, Any]]],
        resizeMeta: Optional[DataResizeMeta] = None,) -> List[Dict[str, Any]]:
        restored_items = [dict(item) for item in (ocrItems or [])]
        if not restored_items:
            return restored_items

        boxes: List[List[float]] = []
        valid_indices: List[int] = []
        for idx, item in enumerate(restored_items):
            box = item.get("box", None)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                item["box"] = (0, 0, 0, 0)
                continue
            boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            valid_indices.append(idx)

        if not boxes:
            return restored_items

        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if resizeMeta is not None:
            boxes_np = DataPreprocessor.RestoreBoxesXYXY(boxes_np, resizeMeta, clamp=True)

        for idx, box in zip(valid_indices, boxes_np):
            x1, y1, x2, y2 = [int(round(float(v))) for v in box.tolist()]
            restored_items[idx]["box"] = (x1, y1, x2, y2)

        return restored_items

    def DrawBoxesOnImage(
        self,
        image: Union[np.ndarray, torch.Tensor],
        ocrItems: Optional[List[Dict[str, Any]]],
        *,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            canvas = image.detach().cpu().numpy()
        else:
            canvas = np.asarray(image)

        if canvas.ndim == 2:
            canvas = canvas[..., None]
        if canvas.ndim != 3:
            raise ValueError(f"visual image must have 2 or 3 dims, but got {canvas.shape}")

        if canvas.shape[0] in (1, 3, 4) and canvas.shape[-1] not in (1, 3, 4):
            canvas = np.moveaxis(canvas, 0, -1)

        if canvas.shape[-1] == 1:
            canvas = np.repeat(canvas, 3, axis=-1)
        elif canvas.shape[-1] >= 3:
            canvas = canvas[..., :3]
        else:
            raise ValueError(f"visual image channel layout is invalid: {canvas.shape}")

        canvas = np.ascontiguousarray(np.array(canvas, copy=True))
        if canvas.size == 0:
            return canvas

        h_img, w_img = canvas.shape[:2]
        thick = max(1, int(thickness))

        for item in ocrItems or []:
            box = item.get("box", None)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(w_img - 1, x1))
            y1 = max(0, min(h_img - 1, y1))
            x2 = max(x1 + 1, min(w_img, x2))
            y2 = max(y1 + 1, min(h_img, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            canvas[y1:min(h_img, y1 + thick), x1:x2] = color
            canvas[max(0, y2 - thick):y2, x1:x2] = color
            canvas[y1:y2, x1:min(w_img, x1 + thick)] = color
            canvas[y1:y2, max(0, x2 - thick):x2] = color

        return canvas

    def FormatVisualText(
        self,
        title: str,
        ocrTexts: List[str],
        extraLines: Optional[List[str]] = None,) -> str:
        lines: List[str] = []
        title_text = str(title).strip()
        if title_text:
            lines.append(title_text)

        if extraLines:
            for line in extraLines:
                line_text = str(line).strip()
                if line_text:
                    lines.append(line_text)

        lines.append("OCR Text:")
        if ocrTexts:
            lines.extend(ocrTexts)
        else:
            lines.append("<empty>")

        return "\n".join(lines)

    def BuildVisualPayload(
        self,
        image: Optional[Union[np.ndarray, torch.Tensor]],
        *,
        ocrTexts: Optional[List[str]] = None,
        ocrItems: Optional[List[Dict[str, Any]]] = None,
        resizeMeta: Optional[DataResizeMeta] = None,
        title: str = "",
        extraLines: Optional[List[str]] = None,
        drawBoxes: bool = True,) -> Dict[str, Any]:
        updated_at = time.time()
        clean_texts = [str(text).strip() for text in (ocrTexts or []) if str(text).strip() != ""]
        restored_items = self.RestoreOcrItemsToOriginal(ocrItems, resizeMeta)

        if image is None:
            payload = self.controller.EmptyVisualStatus()
            payload["text"] = self.FormatVisualText(title, clean_texts, extraLines)
            payload["ocr_texts"] = clean_texts
            payload["items"] = restored_items
            payload["updated_at"] = updated_at
            return payload

        bitmap_rgb = self.DrawBoxesOnImage(
            image,
            restored_items if drawBoxes else None)

        return {
            "bitmap": bitmap_rgb.tolist(),
            "text": self.FormatVisualText(title, clean_texts, extraLines),
            "ocr_texts": clean_texts,
            "items": restored_items,
            "updated_at": updated_at,}

    def SaveModuleParameters(self, brain: BrainCore, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        brain_state = {
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in brain.state_dict().items()}

        torch.save({
            "brain": brain_state,}, str(out_path))

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

    def LoadTorchPayload(self, path: str):
        try:
            return torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"Safe mode loading failed: {e}, try the normal mode")
            return torch.load(path, map_location=self.device)

    def LoadBrainWeights(self, brain: BrainCore, path: str) -> None:
        payload = self.LoadTorchPayload(path)

        if isinstance(payload, dict) and "brain" in payload:
            brain_state = payload["brain"]
        elif isinstance(payload, dict):
            brain_state = payload
        else:
            raise TypeError(f"checkpoint {path} has invalid brain weights payload")

        brain.load_state_dict(brain_state, strict=False)

    def ApplyParameterOverrideAfterResume(
        self,
        *,
        enabled: bool,
        parameterPath: Optional[str],
        loadFn: Callable[[str], None],
        logPrefix: str,) -> bool:
        if not enabled:
            return False
        if not parameterPath:
            print(f"[{logPrefix}] parameter override skipped: parameter path is empty")
            return False

        override_path = Path(parameterPath)
        if not override_path.exists():
            print(f"[{logPrefix}] parameter override skipped, file not found: {override_path}")
            return False

        loadFn(str(override_path))
        print(f"[{logPrefix}] checkpoint weights overridden from parameter file: {override_path}")
        return True

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
        epochs: int,
        batchSize: int,
        valSplit: float,
        resume: bool,
        *,
        isTest: bool,
        ckptPath: str,
        outPath: str,
        trainDetection: bool = True,
        trainRecognition: bool = True,
        recognizerInitPath: Optional[str] = None,
        overrideCheckpointWithModuleParams: bool = False,):
        try:
            torch.autograd.set_detect_anomaly(True)

            self.ResetControllerFlags()

            if not trainDetection and not trainRecognition:
                raise ValueError("OCRTrainLoop requires trainDetection or trainRecognition to be True")

            ds = OfflineOCRDataset(isTest=isTest)
            engine = OCREngineExtractor().to(self.device)
            has_resume_ckpt = resume and Path(ckptPath).exists()

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
            if has_resume_ckpt:
                start_epoch, best_val, train_ds, val_ds, test_ds = self.LoadOCRCheckpoint(
                    engine,
                    optimizer,
                    ds,
                    ckptPath,
                    allowOptimizerMismatch=True)
                self.ApplyParameterOverrideAfterResume(
                    enabled=overrideCheckpointWithModuleParams,
                    parameterPath=outPath,
                    loadFn=lambda path: self.LoadOCRWeightsIntoEngine(engine, path),
                    logPrefix="TrainOCR")
            else:
                if recognizerInitPath and Path(recognizerInitPath).exists():
                    self.LoadRecognizerWeightsIntoEngine(engine, recognizerInitPath)
                elif not trainRecognition:
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

            train_ds, val_ds, test_ds = self.SplitDataset(
                ds,
                valSplit=valSplit,
                testSplit=testSplit,
                trainDataset=train_ds,
                valDataset=val_ds,
                testDataset=test_ds)

            def collate_ocr_batch(batch):
                imgs, boxes, texts, ignore_flags = zip(*batch)
                return list(imgs), list(boxes), list(texts), list(ignore_flags)

            train_dl = self.CreateDataLoader(
                train_ds,
                batchSize=batchSize,
                shuffle=True,
                collateFn=collate_ocr_batch)

            val_dl = self.CreateDataLoader(
                val_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_ocr_batch)

            test_dl = self.CreateDataLoader(
                test_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_ocr_batch)

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
                total_batches=len(train_dl),
                visual=self.controller.EmptyVisualStatus(touch=True),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR training stopped")
                    break

                self.WaitWhilePaused("OCR training paused")

                engine.train()
                epoch_loss = 0.0
                nb = 0

                for bi, batch in enumerate(train_dl, start=1):
                    imgs_b, boxes_b, texts_b, ignore_b = batch
                    sample_losses: List[torch.Tensor] = []
                    batch_correct = 0
                    batch_elems = 0
                    latest_visual = None

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
                        detect_img = sample["detect_img"].unsqueeze(0) # [1, 3, H, W]
                        det_loss = zero
                        if trainDetection:
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

                        pred_ocr_items: List[Dict[str, Any]] = []
                        pred_ocr_texts: List[str] = []
                        with torch.no_grad():
                            pred_ocr_batch = engine(detect_img.detach())
                            if pred_ocr_batch:
                                pred_ocr_items = pred_ocr_batch[0]
                            pred_ocr_texts = [
                                str(item.get("text", "")).strip()
                                for item in pred_ocr_items
                                if str(item.get("text", "")).strip() != ""]

                        if self.controller.IsVisualStateEnabled():
                            latest_visual = self.BuildVisualPayload(
                                img,
                                ocrTexts=pred_ocr_texts,
                                ocrItems=pred_ocr_items,
                                resizeMeta=sample["resize_meta"],
                                title="OCR Train",
                                extraLines=[
                                    f"epoch {ep + 1}/{epochs}",
                                    f"batch {bi}/{len(train_dl)}"],)

                        sample_losses.append(det_loss + rec_loss)

                    if not sample_losses:
                        continue

                    loss = torch.stack(sample_losses).mean() # []
                    rec_acc = (batch_correct / batch_elems) if batch_elems > 0 else 0.0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(engine.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    nb += 1

                    status_kwargs = {
                        "epoch": ep + 1,
                        "total_epochs": epochs,
                        "batch": bi,
                        "total_batches": len(train_dl),
                        "train_loss": float(loss.item()),}
                    if latest_visual is not None:
                        status_kwargs["visual"] = latest_visual

                    self.controller.SetStatus(
                        "training",(
                            f"OCR training... rec_acc={rec_acc:.3f}"
                            if trainRecognition else
                            "OCR training..."),
                        **status_kwargs)

                    if self.controller.ShouldStop():
                        break

                    self.WaitWhilePaused("OCR training paused")

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
        epochs: int,
        batchSize: int,
        valSplit: float,
        resume: bool,
        *,
        isTest: bool,
        ckptPath: str,
        outPath: str,
        overrideCheckpointWithModuleParams: bool = False,):
        try:
            torch.autograd.set_detect_anomaly(True)

            self.ResetControllerFlags()

            ds = OfflineOCRRecognitionDataset(isTest=isTest)
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
                self.ApplyParameterOverrideAfterResume(
                    enabled=overrideCheckpointWithModuleParams,
                    parameterPath=outPath,
                    loadFn=lambda path: self.LoadRecognizerWeightsIntoEngine(engine, path),
                    logPrefix="TrainOCRRec")

            train_ds, val_ds, test_ds = self.SplitDataset(
                ds,
                valSplit=valSplit,
                testSplit=testSplit,
                trainDataset=train_ds,
                valDataset=val_ds,
                testDataset=test_ds)

            def collate_rec_batch(batch):
                imgs, texts, ignore_flags = zip(*batch)
                return list(imgs), list(texts), list(ignore_flags)

            train_dl = self.CreateDataLoader(
                train_ds,
                batchSize=batchSize,
                shuffle=True,
                collateFn=collate_rec_batch)

            val_dl = self.CreateDataLoader(
                val_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_rec_batch)

            test_dl = self.CreateDataLoader(
                test_ds,
                batchSize=batchSize,
                shuffle=False,
                collateFn=collate_rec_batch)

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
                total_batches=len(train_dl),
                visual=self.controller.EmptyVisualStatus(touch=True),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "OCR recognizer training stopped")
                    break

                self.WaitWhilePaused("OCR recognizer training paused")

                engine.train()
                epoch_loss = 0.0
                nb = 0

                for bi, batch in enumerate(train_dl, start=1):
                    imgs_b, texts_b, ignore_b = batch
                    sample_losses: List[torch.Tensor] = []
                    batch_correct = 0
                    batch_elems = 0
                    latest_visual = None

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

                        if self.controller.IsVisualStateEnabled():
                            latest_visual = self.BuildVisualPayload(
                                img,
                                ocrTexts=([pred_text] if pred_text != "" else []),
                                title="OCR Recognition Train",
                                extraLines=[
                                    f"epoch {ep + 1}/{epochs}",
                                    f"batch {bi}/{len(train_dl)}"],
                                drawBoxes=False,)

                        sample_losses.append(rec_loss)

                    if not sample_losses:
                        continue

                    loss = torch.stack(sample_losses).mean() # []
                    rec_acc = (batch_correct / batch_elems) if batch_elems > 0 else 0.0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(engine.recognizer.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    nb += 1

                    status_kwargs = {
                        "epoch": ep + 1,
                        "total_epochs": epochs,
                        "batch": bi,
                        "total_batches": len(train_dl),
                        "train_loss": float(loss.item()),}
                    if latest_visual is not None:
                        status_kwargs["visual"] = latest_visual

                    self.controller.SetStatus(
                        "training",
                        f"OCR recognizer training... acc={rec_acc:.3f}",
                        **status_kwargs)

                    if self.controller.ShouldStop():
                        break

                    self.WaitWhilePaused("OCR recognizer training paused")

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



    def TrainLoop(self, epochs: int, batchSize: int, valSplit: float, resume: bool, onlineLearning = False, *, isTest: bool = False, worldMemPath: str = None, memMemPath: str = None, ckptPath: str = None, outPath: str = None, overrideCheckpointWithModuleParams: bool = False,):
        try:
            torch.autograd.set_detect_anomaly(True)
            ckptPath = ckptPath or BasicParameters.CKPT_PATH_TRAIN
            outPath = outPath or (BasicParameters.MODULEPARAMETER_PATH_TEST if isTest else BasicParameters.MODULEPARAMETER_PATH)

            self.ResetControllerFlags()

            ds = OfflineGameDataset(isTest=isTest)

            brain = BrainCore(device=self.device, plasticHebbian=True, plasticOnlineLearning=onlineLearning, usePlanner=False,)

            agent = Agent(brain, isTrain=True, device=self.device, worldMemoryPath=worldMemPath, memMemoryPath=memMemPath,)

            if onlineLearning:
                agent.UpdateAllWrappers("autogrow")

            start_epoch = 0
            best_val = float("inf")
            train_ds = val_ds = test_ds = None

            testSplit = 0.1

            if resume and Path(ckptPath).exists():
                start_epoch, best_val, train_ds, val_ds = self.LoadCheckpoint(brain, agent, ds, ckptPath)
                self.ApplyParameterOverrideAfterResume(
                    enabled=overrideCheckpointWithModuleParams,
                    parameterPath=outPath,
                    loadFn=lambda path: self.LoadBrainWeights(brain, path),
                    logPrefix="Train")

            train_ds, val_ds, test_ds = self.SplitDataset(
                ds,
                valSplit=valSplit,
                testSplit=testSplit,
                trainDataset=train_ds,
                valDataset=val_ds,
                testDataset=test_ds)

            train_dl = self.CreateDataLoader(
                train_ds,
                batchSize=batchSize,
                shuffle=False,
                pinMemory=True)
            val_dl = self.CreateDataLoader(
                val_ds,
                batchSize=batchSize,
                shuffle=False,
                pinMemory=False)
            test_dl = self.CreateDataLoader(
                test_ds,
                batchSize=batchSize,
                shuffle=False,
                pinMemory=False)

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

            self.controller.SetStatus("training", "Training started", epoch=start_epoch, total_epochs=epochs, batch=0, total_batches=len(train_dl), visual=self.controller.EmptyVisualStatus(touch=True),)

            for ep in range(start_epoch, epochs):
                if self.controller.ShouldStop():
                    self.controller.SetStatus("stopped", "Training stopped")
                    break

                self.WaitWhilePaused("Training paused")

                brain.train()
                epoch_loss = 0.0
                nb = 0

                agent.ResetBrainState(B=batchSize, isOnlineLearning=onlineLearning)

                for bi, batch in enumerate(train_dl, start=1):
                    img_b, key_b, mouse_click_b, mouse_move_b, reward_b, done_b, ext_text_b = unpack_batch(batch)

                    visual_enabled = self.controller.IsVisualStateEnabled()
                    pack = DataPreprocessor.ConvertNpImagesKeysMouses(
                        imgs=img_b,
                        keys=key_b,
                        mouseClick=mouse_click_b,
                        mouseMove=mouse_move_b,
                        reward=reward_b,
                        done=done_b,
                        device=self.device,
                        needVisualState=visual_enabled,)
                    
                    frames = pack["frames"]
                    original_images = pack["original_images"]
                    resize_meta = pack["resize_meta"]
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

                    key_pred, click_pred, mouse_move_pred, model_loss, ocr_items = act_out
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

                    visual_payload = None
                    if visual_enabled and original_images:
                        batch_ocr_items = (ocr_items[0] if (ocr_items is not None and len(ocr_items) > 0) else [])
                        batch_ocr_texts = [
                            str(item.get("text", "")).strip()
                            for item in batch_ocr_items
                            if str(item.get("text", "")).strip() != ""]
                        resize_meta_0 = resize_meta[0] if resize_meta else None
                        visual_payload = self.BuildVisualPayload(
                            original_images[0],
                            ocrTexts=batch_ocr_texts,
                            ocrItems=batch_ocr_items,
                            resizeMeta=resize_meta_0,
                            title="Train",
                            extraLines=[
                                f"epoch {ep + 1}/{epochs}",
                                f"batch {bi}/{len(train_dl)}"],)

                    status_kwargs = {
                        "epoch": ep + 1,
                        "total_epochs": epochs,
                        "batch": bi,
                        "total_batches": len(train_dl),
                        "train_loss": float(loss.item()),}
                    if visual_payload is not None:
                        status_kwargs["visual"] = visual_payload

                    self.controller.SetStatus("training", "Training...", **status_kwargs)

                    if self.controller.ShouldStop():
                        break
                    self.WaitWhilePaused("Training paused")

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

                            v_pack = DataPreprocessor.ConvertNpImagesKeysMouses(
                                imgs=img_b,
                                keys=key_b,
                                mouseClick=mouse_click_b,
                                mouseMove=mouse_move_b,
                                reward=reward_b,
                                done=done_b,
                                device=self.device,
                                needVisualState=False,)
                            
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

                            v_key_pred, v_click_pred, v_mouse_move_pred, v_model_loss, ocr_items = act_out
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
                    self.SaveModuleParameters(brain, outPath)

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
        return self.StartBackgroundTask(
            self.DeployLoop,
            args=(cameraIndex,),
            kwargs={"useHebbian": useHebbian, "usePlanner": usePlanner,})


    def DeployLoop(self, cameraIndex: int,* ,useHebbian: bool = True, usePlanner: bool = True,):
        try:
            self.ResetControllerFlags()

            brain = BrainCore(device=self.device,plasticHebbian=useHebbian,plasticOnlineLearning=False,usePlanner=usePlanner,)

            model_path = BasicParameters.MODULEPARAMETER_PATH

            if os.path.exists(model_path):
                self.LoadBrainWeights(brain, model_path)
            else:
                msg = f"The module file is not exit: {model_path}"
                print(msg)
                self.controller.SetStatus("error", msg)
                return 

            brain.to(self.device)
            brain.eval()

            agent = Agent(brain,isTrain=False,device=self.device,worldMemoryPath=BasicParameters.WORLD_MEMORY_PATH,memMemoryPath=BasicParameters.MEMORY_MEMORY_PATH,)
            agent.ResetBrainState(isOnlineLearning=False)

            if iio is None:
                raise RuntimeError("imageio.v3 cant use")

            self.controller.SetStatus("is_begin", "Deployment started", visual=self.controller.EmptyVisualStatus(touch=True))

            code_to_name: dict[int, str] = {}
            all_codes = []
            for name, code in RAW_KEYBOARD_LAYOUT.items():
                code_to_name.setdefault(code, name)
                all_codes.append(code)
            max_code = max(all_codes)

            with iio.imopen(f"<video{cameraIndex}>", "r") as cam:
                for frame_np in cam:
                    if self.controller.ShouldStop():
                        break

                    visual_enabled = self.controller.IsVisualStateEnabled()
                    pack = DataPreprocessor.ConvertNpImagesKeysMouses(
                        imgs=frame_np,
                        keys=None,
                        mouseClick=None,
                        mouseMove=None,
                        reward=None,
                        done=None,
                        device=self.device,
                        needVisualState=visual_enabled,)
                    
                    frames = pack["frames"]
                    original_images = pack["original_images"]
                    resize_meta = pack["resize_meta"]

                    if self.controller.ShouldResetHebbian():
                        agent.ResetHebbianMemory()
                        self.controller.RequestCancelResetHebbian()

                    act_out = agent.Act(
                        frames,
                        reward=None,
                        done=None,
                        sampleActions=True,
                        deterministicActor=True,)
                    if act_out is None:
                        continue

                    keys, clicks, mouse, ocr_items = act_out

                    kv = keys[0].detach().cpu()
                    ck = clicks[0].detach().cpu()
                    ms = mouse[0].detach().cpu()

                    pressed_names: list[str] = []
                    limit = min(max_code + 1, int(kv.numel()))
                    for code in range(limit):
                        if float(kv[code]) > 0.5:
                            pressed_names.append(code_to_name.get(code, f"Key{code}"))

                    if float(ck[0]) > 0.5:
                        pressed_names.append("MouseLeft")
                    if float(ck[1]) > 0.5:
                        pressed_names.append("MouseRight")

                    names_str = "[" + ", ".join(pressed_names) + "]" if pressed_names else "[]"

                    deploy_items = (ocr_items[0] if (ocr_items is not None and len(ocr_items) > 0) else [])
                    deploy_texts = [
                        str(item.get("text", "")).strip()
                        for item in deploy_items
                        if str(item.get("text", "")).strip() != ""]
                    resize_meta_0 = resize_meta[0] if resize_meta else None
                    visual_payload = None
                    if visual_enabled and original_images:
                        visual_payload = self.BuildVisualPayload(
                            original_images[0],
                            ocrTexts=deploy_texts,
                            ocrItems=deploy_items,
                            resizeMeta=resize_meta_0,
                            title="Deploy",
                            extraLines=[
                                f"keys:{names_str}",
                                f"mouse:dx={float(ms[0]):.3f},dy={float(ms[1]):.3f}"],)

                    status_kwargs = {}
                    if visual_payload is not None:
                        status_kwargs["visual"] = visual_payload

                    self.controller.SetStatus("is_begin",(f"keys:{names_str} "f"mouse:dx={float(ms[0]):.3f},dy={float(ms[1]):.3f}"), **status_kwargs)

            self.controller.SetStatus("stopped", "Deployment stopped")

        except Exception as e:
            tb = traceback.format_exc()
            self.controller.SetStatus("error", f"Deployment error: {e}", trace=tb)
        finally:
            self.is_begin = False



    def TestPerceptionModule(self):
        return self.RunNamedTest("perception")

    def TestAttentionModule(self):
        return self.RunNamedTest("attention")

    def TestMemoryModule(self):
        return self.RunNamedTest("memory")

    def TestDecisionModule(self):
        return self.RunNamedTest("decision")

    def TestWorldModule(self):
        return self.RunNamedTest("world")

    def TestValueEstimationModule(self):
        return self.RunNamedTest("value")
    
    def TestConsciousnessModule(self):
        return self.RunNamedTest("consciousness")
    
    def TestIntentionModule(self):
        return self.RunNamedTest("intention")
    
    def TestOCRModule(self):
        return self.RunNamedTest("OCR")
    

    def MonitorTraining(self):
        self.MonitorStatus(
            prefix="TRAIN",
            monitorName="MonitorTraining",
            renderStatus=lambda st: (
                f"{st['state']} | epoch {st['epoch']}/{st['total_epochs']} "
                f"| batch {st['batch']}/{st['total_batches']} "
                f"| train_loss={st['train_loss']:.4f} | msg={st['message']}"),
            terminalStates=("completed", "stopped", "error"),
            sleepSeconds=1.0,)


    def MonitorDeployment(self):
        self.MonitorStatus(
            prefix="DEPLOY",
            monitorName="MonitorDeployment",
            renderStatus=lambda st: f"{st['state']} | msg={st['message']}",
            terminalStates=("stopped", "error"),
            sleepSeconds=0.5,)


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
            root = Path(dataRoot)
            BasicParameters.Set("DATA_ROOT_PATH_TEST", str(root))

            if not self.HasGameDataset(root):
                if iio is None:
                    raise RuntimeError("imageio.v3 error")

                rng = np.random.default_rng(seed)
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

                print(f"[SmokeTest] created random game dataset at: {root}")
            else:
                print(f"[SmokeTest] use existing game dataset at: {root}")

            print("[SmokeTest] start train...")

            """ok = self.StartTraining(epochs=epochs,batchSize=batchSize,valSplit=valSplit,resume=False, onlineLearning=onlineLearning, isTest=True)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"False": False, "msg": "StartTraining returns False (training may already be running)"}

            self.message_thread = threading.Thread(target=self.MonitorTraining,args=(),daemon=False,)
            self.message_thread.start()"""

            self.TrainLoop(epochs, batchSize, valSplit, False, onlineLearning, isTest=True, worldMemPath=BasicParameters.WORLD_MEMORY_PATH_TEST, memMemPath=BasicParameters.MEMORY_MEMORY_PATH_TEST,ckptPath=BasicParameters.CKPT_PATH_TEST)
        
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
            BasicParameters.Set("OCR_DATA_ROOT_PATH_TEST", str(root))
            frames_dir = root / "frames"
            ocr_texts_dir = root / "OCRTexts"

            if not self.HasOcrDataset(root):
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
                trainRecognition=True,)

            if not ok:
                print("StartOCRTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

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
            BasicParameters.Set("OCR_RECOGNIZER_DATA_ROOT_PATH_TEST", str(root))
            frames_dir = root / "frames"
            texts_dir = root / "OCRTexts"

            if not self.HasOcrRecognitionDataset(root):
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
                isTest=True,)

            if not ok:
                print("StartOCRRecognitionTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

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
        overrideCheckpointWithModuleParams: Optional[bool] = None,) -> Dict[str, Any]:
        try:
            if not (trainDetection or trainRecognition):
                print("[TrainOCR] trainDetection and trainRecognition cannot both be False.")
                return {"ok": False, "msg": "no_train_target"}

            ocr_texts_dir = Path(BasicParameters.OCR_TEXTS_PATH)
            txt_files = sorted(ocr_texts_dir.glob("*.txt")) if ocr_texts_dir.exists() else []

            if not self.HasOcrDataset():
                print(
                    "[TrainOCR] no valid OCR dataset found, please prepare these folders first: "
                    f"{BasicParameters.OCR_FRAMES_PATH}, "
                    f"{BasicParameters.OCR_TEXTS_PATH}.")
                return {"ok": False, "msg": "invalid ocr dataset"}

            for txt_path in txt_files:
                try:
                    boxes, texts, ignore_flags = DataPreprocessor.LoadOCRAnnotations(txt_path)
                except Exception as e:
                    print(f"[TrainOCR] failed to parse OCR annotation file {txt_path}: {e}")
                    return {"ok": False, "msg": "invalid ocr labels"}

                if not boxes or not texts or len(ignore_flags) != len(texts):
                    print(f"[TrainOCR] OCR annotation file is empty or invalid: {txt_path}")
                    return {"ok": False, "msg": "invalid ocr labels"}

            print(
                "[TrainOCR] use configured OCR folders: "
                f"{BasicParameters.OCR_FRAMES_PATH}, "
                f"{BasicParameters.OCR_TEXTS_PATH}.")

            ok = self.StartOCRTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                isTest=False,
                trainDetection=trainDetection,
                trainRecognition=trainRecognition,
                overrideCheckpointWithModuleParams=overrideCheckpointWithModuleParams,)

            if not ok:
                print("StartOCRTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

            return {"ok": True}

        except Exception as e:
            print(f"TrainOCRModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

    def TrainOCRRecognitionModule(
        self,
        epochs: int = 6,
        batchSize: int = 8,
        valSplit: float = 0.2,
        isResume: bool = False,
        *,
        overrideCheckpointWithModuleParams: Optional[bool] = None,) -> Dict[str, Any]:
        try:
            if not self.HasOcrRecognitionDataset():
                print(
                    "[TrainOCRRec] no valid OCR recognition dataset found, please prepare these folders first: "
                    f"{BasicParameters.OCR_RECOGNIZER_FRAMES_PATH}, "
                    f"{BasicParameters.OCR_RECOGNIZER_TEXTS_PATH}.")
                return {"ok": False, "msg": "invalid ocr recognition dataset"}

            print(
                "[TrainOCRRec] use configured OCR recognition folders: "
                f"{BasicParameters.OCR_RECOGNIZER_FRAMES_PATH}, "
                f"{BasicParameters.OCR_RECOGNIZER_TEXTS_PATH}.")

            ok = self.StartOCRRecognitionTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                isTest=False,
                overrideCheckpointWithModuleParams=overrideCheckpointWithModuleParams,)

            if not ok:
                print("StartOCRRecognitionTraining returns False (training may already be running)")
                return {"ok": False, "msg": "already_running"}

            self.StartMessageMonitor(self.MonitorTraining)

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
        overrideCheckpointWithModuleParams: Optional[bool] = None,) -> Dict[str, Any]:
        try:
            if not self.HasGameDataset():
                print(
                    "[Train] no dataset found, please prepare these folders first: "
                    f"{BasicParameters.DATA_FRAMES_PATH}, "
                    f"{BasicParameters.DATA_KEYS_PATH}, "
                    f"{BasicParameters.DATA_MOUSE_CLICK_PATH}, "
                    f"{BasicParameters.DATA_MOUSE_MOVE_PATH}, "
                    f"{BasicParameters.DATA_REWARD_PATH}, "
                    f"{BasicParameters.DATA_DONE_PATH}.")
                return {"ok": False, "msg": "no dataset"}

            print(
                "[Train] use configured dataset folders: "
                f"{BasicParameters.DATA_FRAMES_PATH}, "
                f"{BasicParameters.DATA_KEYS_PATH}, "
                f"{BasicParameters.DATA_MOUSE_CLICK_PATH}, "
                f"{BasicParameters.DATA_MOUSE_MOVE_PATH}, "
                f"{BasicParameters.DATA_REWARD_PATH}, "
                f"{BasicParameters.DATA_DONE_PATH}.")

            ok = self.StartTraining(
                epochs=epochs,
                batchSize=batchSize,
                valSplit=valSplit,
                resume=isResume,
                onlineLearning=onlineLearning,
                isTest=False,
                overrideCheckpointWithModuleParams=overrideCheckpointWithModuleParams,)

            if not ok:
                print("StartTraining returns False (training may already be running)")
                return {"False": False, "msg": "StartTraining returns False (training may already be running)"}

            self.StartMessageMonitor(self.MonitorTraining)

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

            self.StartMessageMonitor(self.MonitorDeployment)

            return {"ok": True}

        except Exception as e:
            print(f"DeployModule failed with error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise
