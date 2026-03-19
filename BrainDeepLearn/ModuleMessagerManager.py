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


class ModuleMessagerManager:
    def __init__(self, maxSteps: int = 128):
        self.lock = threading.RLock()
        self.maxSteps = max(1, int(maxSteps))
        self.Reset()

    def Reset(self) -> None:
        with self.lock:
            now = time.time()
            self.steps: List[Dict[str, Any]] = []
            self.meta: Dict[str, Any] = {
                "created_at": now,
                "updated_at": now,
                "max_steps": self.maxSteps,}

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
                "output": self.ToJsonSafe(output)}

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

    def ToJsonSafe(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, bool, int)):
            return value

        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return str(value)

        if isinstance(value, Path):
            return str(value)

        if is_dataclass(value):
            return self.ToJsonSafe(asdict(value))

        if np is not None:
            if isinstance(value, np.generic):
                return self.ToJsonSafe(value.item())
            if isinstance(value, np.ndarray):
                return self.ToJsonSafe(value.tolist())

        if torch is not None and isinstance(value, torch.Tensor):
            return {
                "type": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "data": self.ToJsonSafe(value.detach().cpu().tolist())}

        if isinstance(value, OrderedDict):
            return {str(k): self.ToJsonSafe(v) for k, v in value.items()}

        if isinstance(value, dict):
            return {str(k): self.ToJsonSafe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self.ToJsonSafe(v) for v in value]

        if hasattr(value, "__dict__") and not isinstance(value, type):
            return {
                "type": value.__class__.__name__,
                "fields": self.ToJsonSafe(vars(value))}

        return str(value)
