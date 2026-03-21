from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import json
import math
import threading
import time
import numpy as np
import torch



class ModuleDim:
    PerceptionEmbed: int = 512
    PerceptionFeat: int = 2 * PerceptionEmbed

    AttentionFeat: int = PerceptionFeat

    WorldFeat: int = 512
    WorldOutHState: int = 512
    WorldOutZState: int = 64
    WorldOutXState: int = 64

    MemoryFeat = AttentionFeat

    MemoryItem: int = MemoryFeat
    WorldMemoryItem: int = WorldFeat

    ConsciousnessState: int = MemoryFeat

    IntentionFeat: int = 512

    ValueEstimationOutEmotion = 64


class TensorVisualProcessor:
    def __init__(
        self,
        bitmapSize: int = 64,
        collapseStdThr: float = 1e-6,
        collapseEntropyThr: float = 0.01,
        collapseDominanceThr: float = 0.98):
        self.bitmapSize = max(8, int(bitmapSize))
        self.collapseStdThr = float(collapseStdThr)
        self.collapseEntropyThr = float(collapseEntropyThr)
        self.collapseDominanceThr = float(collapseDominanceThr)
        self.Reset()

    def Reset(self):
        self.prevMaps: Dict[str, np.ndarray] = {}

    def ToJsonValue(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value

        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return str(value)

        if isinstance(value, dict):
            return {str(k): self.ToJsonValue(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self.ToJsonValue(v) for v in value]

        return str(value)

    def PrepareArray(self, value: Any):
        if isinstance(value, torch.Tensor):
            tensorCpu = value.detach().cpu()
            rawData = tensorCpu.tolist()
            analysisArray = tensorCpu.float().numpy()
            return "torch.Tensor", list(tensorCpu.shape), str(tensorCpu.dtype), rawData, analysisArray

        arrayValue = np.asarray(value)
        rawData = arrayValue.tolist()
        if np.issubdtype(arrayValue.dtype, np.number) or np.issubdtype(arrayValue.dtype, np.bool_):
            analysisArray = arrayValue.astype(np.float32, copy=False)
        else:
            analysisArray = None
        return "numpy.ndarray", list(arrayValue.shape), str(arrayValue.dtype), rawData, analysisArray

    def BuildChannelView(self, arrayValue: np.ndarray) -> np.ndarray:
        if arrayValue.ndim == 0:
            return arrayValue.reshape(1, 1, 1)
        if arrayValue.ndim == 1:
            return arrayValue.reshape(1, 1, -1)
        if arrayValue.ndim == 2:
            return arrayValue.reshape(1, arrayValue.shape[0], arrayValue.shape[1])
        if arrayValue.ndim == 3 and arrayValue.shape[-1] <= 4 and arrayValue.shape[0] > 4 and arrayValue.shape[1] > 4:
            return np.moveaxis(arrayValue, -1, 0)
        return arrayValue.reshape(-1, arrayValue.shape[-2], arrayValue.shape[-1])

    def ResizeMap(self, spatialMap: np.ndarray) -> np.ndarray:
        h, w = spatialMap.shape
        if h <= 0 or w <= 0:
            return np.zeros((self.bitmapSize, self.bitmapSize), dtype=np.float32)

        ys = np.linspace(0, h - 1, num=self.bitmapSize, dtype=np.int32)
        xs = np.linspace(0, w - 1, num=self.bitmapSize, dtype=np.int32)
        return spatialMap[np.ix_(ys, xs)]

    def BuildBitmap(self, spatialMap: np.ndarray):
        sampled = self.ResizeMap(spatialMap)
        sampled = np.nan_to_num(sampled.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        vMin = float(sampled.min()) if sampled.size > 0 else 0.0
        vMax = float(sampled.max()) if sampled.size > 0 else 0.0

        if sampled.size == 0 or abs(vMax - vMin) < 1e-8:
            normMap = np.zeros_like(sampled, dtype=np.float32)
        else:
            normMap = (sampled - vMin) / (vMax - vMin)

        bitmap = (normMap * 255.0).round().clip(0.0, 255.0).astype(np.uint8)
        return bitmap.tolist(), normMap

    def ComputeSpatialEntropy(self, spatialMap: np.ndarray) -> float:
        flat = np.abs(spatialMap).reshape(-1).astype(np.float64, copy=False)
        total = float(flat.sum())
        if total <= 1e-12 or flat.size == 0:
            return 0.0
        probs = flat / total
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        denom = max(1e-12, float(np.log(flat.size)))
        return entropy / denom

    def ComputeDifference(self, tensorName: str, bitmapMap: np.ndarray):
        prevMap = self.prevMaps.get(str(tensorName))
        diff = None
        if prevMap is not None and prevMap.shape == bitmapMap.shape:
            diff = float(np.mean(np.abs(bitmapMap - prevMap)))
        self.prevMaps[str(tensorName)] = bitmapMap.copy()
        return diff

    def DetectCollapse(self, channelEnergy: np.ndarray, spatialEntropy: float, channelView: np.ndarray) -> Dict[str, Any]:
        stdValue = float(np.std(channelView)) if channelView.size > 0 else 0.0
        energySum = float(channelEnergy.sum()) if channelEnergy.size > 0 else 0.0
        dominance = 1.0
        if energySum > 1e-12 and channelEnergy.size > 0:
            dominance = float(channelEnergy.max() / energySum)

        reasons = []
        if stdValue < self.collapseStdThr:
            reasons.append("low_std")
        if spatialEntropy < self.collapseEntropyThr:
            reasons.append("low_spatial_entropy")
        if dominance > self.collapseDominanceThr:
            reasons.append("single_channel_dominance")

        return {
            "is_collapsed": len(reasons) > 0,
            "collapse_reasons": reasons,
            "std": stdValue,
            "channel_dominance": dominance,}

    def ProcessTensor(self, tensorName: str, value: Any) -> Dict[str, Any]:
        tensorType, shape, dtypeText, rawData, analysisArray = self.PrepareArray(value)
        result: Dict[str, Any] = {
            "type": tensorType,
            "shape": shape,
            "dtype": dtypeText,
            "data": self.ToJsonValue(rawData),}

        if analysisArray is None or analysisArray.size == 0:
            result["bitmap"] = []
            result["max_response_position"] = {"channel": 0, "y": 0, "x": 0}
            result["strongest_channel"] = 0
            result["channel_energy_ranking"] = []
            result["spatial_entropy"] = 0.0
            result["collapse"] = {
                "is_collapsed": True,
                "collapse_reasons": ["non_numeric_or_empty"],
                "std": 0.0,
                "channel_dominance": 1.0,}
            result["difference_from_previous_frame"] = None
            return result

        channelView = self.BuildChannelView(analysisArray)
        absView = np.abs(channelView)
        channelEnergy = absView.reshape(absView.shape[0], -1).sum(axis=1)
        strongestChannel = int(np.argmax(channelEnergy)) if channelEnergy.size > 0 else 0

        maxIndex = int(np.argmax(absView)) if absView.size > 0 else 0
        maxC, maxY, maxX = np.unravel_index(maxIndex, absView.shape) if absView.size > 0 else (0, 0, 0)

        spatialMap = absView.sum(axis=0)
        bitmap, bitmapMap = self.BuildBitmap(spatialMap)
        spatialEntropy = self.ComputeSpatialEntropy(spatialMap)
        collapseInfo = self.DetectCollapse(channelEnergy, spatialEntropy, channelView)
        frameDiff = self.ComputeDifference(tensorName, bitmapMap)

        ranking = []
        order = np.argsort(-channelEnergy) if channelEnergy.size > 0 else []
        for idx in order:
            ranking.append({
                "channel": int(idx),
                "energy": float(channelEnergy[int(idx)]),})

        result["bitmap"] = bitmap
        result["max_response_position"] = {"channel": int(maxC), "y": int(maxY), "x": int(maxX)}
        result["strongest_channel"] = strongestChannel
        result["channel_energy_ranking"] = ranking
        result["spatial_entropy"] = float(spatialEntropy)
        result["collapse"] = collapseInfo
        result["difference_from_previous_frame"] = frameDiff
        return result


class ModuleMessagerManager:
    def __init__(self, maxSteps: int = 128):
        self.lock = threading.RLock()
        self.maxSteps = max(1, int(maxSteps))
        self.tensorVisualProcessor = TensorVisualProcessor()
        self.Reset()

    def Reset(self) -> None:
        with self.lock:
            now = time.time()
            self.steps: List[Dict[str, Any]] = []
            self.meta: Dict[str, Any] = {
                "created_at": now,
                "updated_at": now,
                "max_steps": self.maxSteps,}
            self.tensorVisualProcessor.Reset()

    def SetMaxSteps(self, maxSteps: int) -> None:
        with self.lock:
            self.maxSteps = max(1, int(maxSteps))
            now = time.time()
            self.meta["max_steps"] = self.maxSteps
            self.meta["updated_at"] = now
            self.TrimHistory()

    def SaveModuleOutput(self, moduleName: str, output: Any, isBeginStep: bool = False) -> int:
        if str(moduleName).strip() == "":
            raise ValueError("moduleName cannot be empty")

        with self.lock:
            now = time.time()
            if len(self.steps) == 0:
                stepId = 0
                self.steps.append({
                    "step": stepId,
                    "timestamp": now,
                    "updated_at": now,
                    "modules": {}})
            elif isBeginStep:
                stepId = int(self.steps[-1]["step"]) + 1
                self.steps.append({
                    "step": stepId,
                    "timestamp": now,
                    "updated_at": now,
                    "modules": {}})
            else:
                stepId = int(self.steps[-1]["step"])

            stepRecord = self.steps[-1]
            entry = {
                "module": str(moduleName),
                "step": stepId,
                "timestamp": now,
                "output": self.ToJsonSafe(output, tensorName=str(moduleName))}

            stepRecord["modules"][str(moduleName)] = deepcopy(entry)
            stepRecord["updated_at"] = now
            self.meta["updated_at"] = now
            self.TrimHistory()
            return stepId

    def GetStep(self):
        with self.lock:
            if len(self.steps) == 0:
                return None
            return deepcopy(self.steps[-1])

    def GetRecentSteps(self, nSteps: int = 0) -> List[Dict[str, Any]]:
        with self.lock:
            steps = self.steps
            if nSteps > 0:
                steps = steps[-max(0, int(nSteps)):]
            return deepcopy(steps)

    def GetModuleHistory(self, moduleName: str, nSteps: int = 0) -> List[Dict[str, Any]]:
        with self.lock:
            stepRecords = self.steps
            if nSteps > 0:
                stepRecords = stepRecords[-max(0, int(nSteps)):]

            history = []
            for stepRecord in stepRecords:
                if str(moduleName) in stepRecord["modules"]:
                    history.append(deepcopy(stepRecord["modules"][str(moduleName)]))
            return history

    def GetDatabaseSnapshot(self, nSteps: int = 0) -> Dict[str, Any]:
        with self.lock:
            stepRecords = self.steps
            if nSteps > 0:
                stepRecords = stepRecords[-max(0, int(nSteps)):]

            return {
                "meta": deepcopy(self.meta),
                "current_step": (-1 if len(self.steps) == 0 else int(self.steps[-1]["step"])),
                "stored_steps": len(self.steps),
                "steps": deepcopy(stepRecords),}

    def ExportDict(self, nSteps: int = 0) -> Dict[str, Any]:
        return self.GetDatabaseSnapshot(nSteps=nSteps)

    def ExportJson(
        self,
        nSteps: int = 0,
        *,
        indent: int = 2,
        ensureAscii: bool = False) -> str:
        snapshot = self.GetDatabaseSnapshot(nSteps=nSteps)
        return json.dumps(snapshot, ensure_ascii=ensureAscii, indent=indent)

    def ExportJason(
        self,
        nSteps: int = 0,
        *,
        indent: int = 2,
        ensureAscii: bool = False) -> str:
        return self.ExportJson(nSteps=nSteps, indent=indent, ensureAscii=ensureAscii)

    def SaveJson(
        self,
        path: Union[str, Path],
        nSteps: int = 0,
        *,
        indent: int = 2,
        ensureAscii: bool = False) -> Path:
        outPath = Path(path)
        outPath.parent.mkdir(parents=True, exist_ok=True)
        outPath.write_text(
            self.ExportJson(nSteps=nSteps, indent=indent, ensureAscii=ensureAscii),
            encoding="utf-8")
        return outPath

    def SaveJason(
        self,
        path: Union[str, Path],
        nSteps: int = 0,
        *,
        indent: int = 2,
        ensureAscii: bool = False) -> Path:
        return self.SaveJson(path, nSteps=nSteps, indent=indent, ensureAscii=ensureAscii)

    def TrimHistory(self) -> None:
        if len(self.steps) > self.maxSteps:
            self.steps = self.steps[-self.maxSteps:]

    def ToJsonSafe(self, value: Any, tensorName: str = "value") -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value

        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return str(value)

        if isinstance(value, Path):
            return str(value)

        if is_dataclass(value):
            return self.ToJsonSafe(asdict(value), tensorName=tensorName)

        if np is not None:
            if isinstance(value, np.generic):
                return self.ToJsonSafe(value.item(), tensorName=tensorName)
            if isinstance(value, np.ndarray):
                return self.tensorVisualProcessor.ProcessTensor(tensorName, value)

        if torch is not None and isinstance(value, torch.Tensor):
            return self.tensorVisualProcessor.ProcessTensor(tensorName, value)

        if isinstance(value, OrderedDict):
            return {str(k): self.ToJsonSafe(v, tensorName=f"{tensorName}.{k}") for k, v in value.items()}

        if isinstance(value, dict):
            return {str(k): self.ToJsonSafe(v, tensorName=f"{tensorName}.{k}") for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self.ToJsonSafe(v, tensorName=f"{tensorName}[{idx}]") for idx, v in enumerate(value)]

        if hasattr(value, "__dict__") and not isinstance(value, type):
            return {
                "type": value.__class__.__name__,
                "fields": self.ToJsonSafe(vars(value), tensorName=f"{tensorName}.fields")}

        return str(value)
