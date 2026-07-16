from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from Config import BasicParameters
from CoreTypes import (
    ROBOT_STATE_FIELDS,
    ValidateOfflineSensorManifest,
    ValidateRobotStateWirePacket)
from ModuleMessagerManager import ModuleDim

try:
    import imageio.v3 as iio
except Exception:
    iio = None


def LoadImageFirstFrame(path: Union[str, Path]) -> np.ndarray:
    if iio is None:
        raise RuntimeError("imageio.v3 cant use")

    try:
        return np.asarray(iio.imread(path, index=0))
    except Exception as e:
        raise ValueError(f"failed to read image {path}: {e}") from e


DEPTH_FILE_SUFFIXES = (".npy", ".npz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ARRAY_FILE_SUFFIXES = DEPTH_FILE_SUFFIXES


def ListDepthFiles(path: Union[str, Path]) -> List[Path]:
    depth_dir = Path(path)
    if not depth_dir.exists():
        return []
    return sorted([
        item for item in depth_dir.iterdir()
        if item.is_file() and item.suffix.lower() in DEPTH_FILE_SUFFIXES],
        key=lambda item: item.name)


def LoadDepthArray(path: Union[str, Path]) -> np.ndarray:
    depth_path = Path(path)
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(depth_path))
    if suffix == ".npz":
        payload = np.load(depth_path)
        if len(payload.files) == 0:
            raise ValueError(f"depth npz has no arrays: {depth_path}")
        return np.asarray(payload[payload.files[0]])
    return LoadImageFirstFrame(depth_path)


def ListArrayFiles(path: Union[str, Path]) -> List[Path]:
    array_dir = Path(path)
    if not array_dir.exists():
        return []
    return sorted([
        item for item in array_dir.iterdir()
        if item.is_file() and item.suffix.lower() in ARRAY_FILE_SUFFIXES],
        key=lambda item: item.name)


def ListJsonFiles(path: Union[str, Path]) -> List[Path]:
    json_dir = Path(path)
    if not json_dir.exists():
        return []
    return sorted([
        item for item in json_dir.iterdir()
        if item.is_file() and item.suffix.lower() == ".json"],
        key=lambda item: item.name)


class OfflineGameDataset(Dataset):
    def __init__(self, *, calibrationId: str, isTest: bool = False) -> None:
        self.calibration_id = calibrationId
        if isTest:
            p = Path(BasicParameters.DATA_ROOT_PATH_TEST)
            sensor_manifest_path = Path(
                BasicParameters.DATA_SENSOR_MANIFEST_PATH_TEST)
            self.imgs = sorted((p / "frames").glob("*.png"))
            self.reward = sorted((p / "reward").glob("*.npy"))
            self.done = sorted((p / "done").glob("*.npy"))
            self.depths = ListDepthFiles(getattr(BasicParameters, "DATA_DEPTH_PATH_TEST", p / "depth"))
            self.depth_valids = ListDepthFiles(getattr(BasicParameters, "DATA_DEPTH_VALID_PATH_TEST", p / "depth_valid"))
            self.texts = sorted((p / "texts").glob("*.txt"))
            self.robot_states = ListJsonFiles(BasicParameters.DATA_ROBOT_STATE_PATH_TEST)
            self.normals = ListArrayFiles(getattr(BasicParameters, "DATA_NORMAL_PATH_TEST", p / "normal"))
            self.semantic_segmentations = ListArrayFiles(getattr(BasicParameters, "DATA_SEMANTIC_SEGMENTATION_PATH_TEST", p / "semantic_segmentation"))
            self.instance_segmentations = ListArrayFiles(getattr(BasicParameters, "DATA_INSTANCE_SEGMENTATION_PATH_TEST", p / "instance_segmentation"))
            self.synthetic_annotations = ListJsonFiles(getattr(BasicParameters, "DATA_SYNTHETIC_SUPERVISION_PATH_TEST", p / "synthetic_supervision"))
        else:
            sensor_manifest_path = Path(BasicParameters.DATA_SENSOR_MANIFEST_PATH)
            self.imgs = sorted(Path(BasicParameters.DATA_FRAMES_PATH).glob("*.png"))
            self.reward = sorted(Path(BasicParameters.DATA_REWARD_PATH).glob("*.npy"))
            self.done = sorted(Path(BasicParameters.DATA_DONE_PATH).glob("*.npy"))
            self.depths = ListDepthFiles(BasicParameters.DATA_DEPTH_PATH)
            self.depth_valids = ListDepthFiles(BasicParameters.DATA_DEPTH_VALID_PATH)
            self.texts = sorted(Path(BasicParameters.DATA_TEXTS_PATH).glob("*.txt"))
            self.robot_states = ListJsonFiles(BasicParameters.DATA_ROBOT_STATE_PATH)
            self.normals = ListArrayFiles(BasicParameters.DATA_NORMAL_PATH)
            self.semantic_segmentations = ListArrayFiles(BasicParameters.DATA_SEMANTIC_SEGMENTATION_PATH)
            self.instance_segmentations = ListArrayFiles(BasicParameters.DATA_INSTANCE_SEGMENTATION_PATH)
            self.synthetic_annotations = ListJsonFiles(BasicParameters.DATA_SYNTHETIC_SUPERVISION_PATH)

        sensor_manifest = json.loads(
            sensor_manifest_path.read_text(encoding="utf-8"))
        ValidateOfflineSensorManifest(sensor_manifest, calibrationId)

        if not (
            len(self.imgs)
            == len(self.reward)
            == len(self.done)
            == len(self.depths)
            == len(self.depth_valids)
            == len(self.robot_states)
        ):
            raise ValueError(
                "frames/reward/done/depth/robot_state counts must match")
        frame_ids = [path.stem for path in self.imgs]
        required_streams = {
            "reward": self.reward,
            "done": self.done,
            "depth": self.depths,
            "depth_valid": self.depth_valids,
            "robot_state": self.robot_states,}
        for name, paths in required_streams.items():
            if [path.stem for path in paths] != frame_ids:
                raise ValueError(
                    f"{name} filenames must match frame identifiers exactly")
        if self.texts:
            if len(self.texts) != len(self.imgs):
                raise ValueError("text count must match frames")
            if [path.stem for path in self.texts] != frame_ids:
                raise ValueError("text filenames must match frame identifiers exactly")
        if self.synthetic_annotations:
            if any(
                len(paths) != len(self.imgs)
                for paths in (
                    self.synthetic_annotations,
                    self.normals,
                    self.semantic_segmentations,
                    self.instance_segmentations)
            ):
                raise ValueError(
                    "synthetic supervision stream counts must match frames")
            for name, paths in {
                "synthetic_supervision": self.synthetic_annotations,
                "normal": self.normals,
                "semantic_segmentation": self.semantic_segmentations,
                "instance_segmentation": self.instance_segmentations,}.items():
                if [path.stem for path in paths] != frame_ids:
                    raise ValueError(
                        f"{name} filenames must match frame identifiers exactly")

        self.robot_state_payloads: List[Dict[str, Any]] = []
        stream_id: Optional[str] = None
        world_frame_id: Optional[str] = None
        for sequence_index, (frame_id, state_path) in enumerate(zip(
            frame_ids,
            self.robot_states)):
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            ValidateRobotStateWirePacket(payload, self.calibration_id)
            if payload["frame_id"] != frame_id:
                raise ValueError(
                    "offline RobotState frame_id must match the RGB-D frame identifier")
            if payload["sequence_index"] != sequence_index:
                raise ValueError(
                    "offline RobotState sequence_index must match dataset order")
            if stream_id is None:
                stream_id = payload["stream_id"]
                world_frame_id = payload["world_frame_id"]
            elif (
                payload["stream_id"] != stream_id
                or payload["world_frame_id"] != world_frame_id
            ):
                raise ValueError(
                    "offline RobotState stream_id and world_frame_id must remain stable")
            self.robot_state_payloads.append(payload)
        if stream_id is None or world_frame_id is None:
            raise ValueError("offline RGB-D/RobotState dataset must not be empty")
        self.stream_id = stream_id
        self.world_frame_id = world_frame_id

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        imgs = LoadImageFirstFrame(self.imgs[idx])
        reward = np.load(self.reward[idx]).astype(np.float32)
        done = np.load(self.done[idx]).astype(np.float32)
        depth = LoadDepthArray(self.depths[idx])
        depth_valid = LoadDepthArray(self.depth_valids[idx])
        image_size = BasicParameters.IMAGE_SIZE
        if imgs.dtype != np.uint8 or imgs.shape != (image_size, image_size, 3):
            raise ValueError(
                f"offline RGB must be uint8 [{image_size}, {image_size}, 3]")
        if depth.dtype != np.float32 or depth.shape != (image_size, image_size):
            raise ValueError(
                f"offline depth must be float32 metres [{image_size}, {image_size}]")
        if depth_valid.dtype != np.bool_ or depth_valid.shape != depth.shape:
            raise ValueError(
                "offline depth_valid must be a bool mask matching depth")
        if not np.isfinite(depth).all():
            raise ValueError("offline depth must contain only finite metre values")
        if np.any(depth_valid & (depth <= 0.0)):
            raise ValueError("offline valid depth pixels must be positive")
        ext_text = ""
        if self.texts:
            ext_text = self.texts[idx].read_text(encoding="utf-8").strip()
        robot_state_payload = self.robot_state_payloads[idx]
        robot_state = {
            name: torch.as_tensor(robot_state_payload[name])
            for name in ROBOT_STATE_FIELDS}
        synthetic_targets: Dict[str, torch.Tensor] = {}
        if self.synthetic_annotations:
            annotation = json.loads(self.synthetic_annotations[idx].read_text(encoding="utf-8"))
            rgb_tensor = DataPreprocessor.ToImageTensor(imgs)
            depth_tensor, depth_valid_tensor = DataPreprocessor.ToDepthTensor(
                depth,
                depth_valid)
            normal_tensor = DataPreprocessor.ToNormalTensor(LoadDepthArray(self.normals[idx]))
            semantic_segmentation = DataPreprocessor.ToSegmentationTensor(LoadDepthArray(self.semantic_segmentations[idx]))
            instance_segmentation = DataPreprocessor.ToSegmentationTensor(LoadDepthArray(self.instance_segmentations[idx]))
            synthetic_targets = DataPreprocessor.TensorizeSyntheticSupervision(
                annotation,
                rgb_tensor,
                depth_tensor,
                depth_valid_tensor,
                normal_tensor,
                semantic_segmentation,
                instance_segmentation,
                maxNodes=ModuleDim.PstObservedSlots,
                textDim=ModuleDim.PstTextDim,
                stateDim=ModuleDim.PstStateDim,
                attrDim=ModuleDim.PstAttrDim,
                affordanceDim=ModuleDim.PstAffordanceDim,
                relationClasses=ModuleDim.PstRelationClasses)
        return imgs, reward, done, depth, depth_valid, ext_text, robot_state, synthetic_targets


class OfflineOCRDataset(Dataset):
    def __init__(self, isTest: bool = False) -> None:
        if isTest:
            p = Path(BasicParameters.OCR_DATA_ROOT_PATH_TEST)
            self.imgs = DataPreprocessor.ListImageFiles(p / "frames")
            self.texts = sorted((p / "OCRTexts").glob("*.txt"))
            self.boxes = sorted((p / "boxes").glob("*.npy"))
            self.use_text_annotations = False
            for txt_path in self.texts:
                if DataPreprocessor.TextFileLooksLikeOCRAnnotations(txt_path):
                    self.use_text_annotations = True
                    break
        else:
            self.imgs = DataPreprocessor.ListImageFiles(Path(BasicParameters.OCR_FRAMES_PATH))
            self.texts = sorted(Path(BasicParameters.OCR_TEXTS_PATH).glob("*.txt"))
            self.boxes = []
            p = Path(BasicParameters.OCR_DATA_ROOT_PATH)
            self.use_text_annotations = True

        assert len(self.imgs) == len(self.texts), "frames/OCRTexts The number of files is inconsistent."
        if len(self.imgs) == 0:
            raise RuntimeError(f"no OCR samples found under {p}")

        if not self.use_text_annotations:
            assert len(self.boxes) == len(self.imgs), "frames/boxes/OCRTexts The number of files is inconsistent."

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = LoadImageFirstFrame(self.imgs[idx])
        if self.use_text_annotations:
            boxes, texts, ignore_flags = DataPreprocessor.LoadOCRAnnotations(self.texts[idx])
        else:
            boxes = np.load(self.boxes[idx]).astype(np.float32)
            if boxes.ndim == 1:
                boxes = boxes.reshape(-1, 4)
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError(f"OCR boxes must have shape [N, 4], but got {boxes.shape} from {self.boxes[idx]}")

            texts = self.texts[idx].read_text(encoding="utf-8").splitlines()
            ignore_flags = np.zeros((len(texts),), dtype=np.float32)

        if len(boxes) != len(texts):
            raise ValueError(
                f"OCR texts/boxes count mismatch at {self.texts[idx]}: "
                f"{len(texts)} vs {len(boxes)}")
        if len(ignore_flags) != len(texts):
            raise ValueError(f"OCR ignore flags/texts mismatch at {self.texts[idx]}: {len(ignore_flags)} vs {len(texts)}")
        return img, boxes, texts, ignore_flags


class OfflineOCRRecognitionDataset(Dataset):
    def __init__(self, isTest: bool = False) -> None:
        if isTest:
            p = Path(BasicParameters.OCR_RECOGNIZER_DATA_ROOT_PATH_TEST)
            self.imgs = DataPreprocessor.ListImageFiles(p / "frames")
            self.texts = sorted((p / "OCRTexts").glob("*.txt"))
        else:
            self.imgs = DataPreprocessor.ListImageFiles(Path(BasicParameters.OCR_RECOGNIZER_FRAMES_PATH))
            self.texts = sorted(Path(BasicParameters.OCR_RECOGNIZER_TEXTS_PATH).glob("*.txt"))
            p = Path(BasicParameters.OCR_RECOGNIZER_DATA_ROOT_PATH)

        assert len(self.imgs) == len(self.texts), "frames/OCRTexts The number of files is inconsistent."
        if len(self.imgs) == 0:
            raise RuntimeError(f"no OCR recognition samples found under {p}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = LoadImageFirstFrame(self.imgs[idx])
        raw_lines = [
            line.strip()
            for line in self.texts[idx].read_text(encoding="utf-8").splitlines()
            if line.strip()]

        if len(raw_lines) == 0:
            text = ""
            ignore_flag = 1
        elif len(raw_lines) == 1 and DataPreprocessor.LooksLikeOCRAnnotationLine(raw_lines[0]):
            _, ignore_flag, text = DataPreprocessor.ParseOCRAnnotationLine(raw_lines[0])
        elif len(raw_lines) == 1:
            text = raw_lines[0]
            ignore_flag = 0
        else:
            raise ValueError(
                f"OCR recognition label file must contain exactly one non-empty line: {self.texts[idx]}")

        return img, text, np.float32(ignore_flag)


@dataclass
class DataResizeMeta:
    src_h: int
    src_w: int
    dst_h: int
    dst_w: int
    scale_x: float
    scale_y: float


class DataPreprocessor:
    SUPPORTED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

    @staticmethod
    def ListDepthFiles(path: Union[str, Path]) -> List[Path]:
        return ListDepthFiles(path)

    @staticmethod
    def ListJsonFiles(path: Union[str, Path]) -> List[Path]:
        return ListJsonFiles(path)

    @staticmethod
    def ListImageFiles(path: Union[str, Path]) -> List[Path]:
        img_dir = Path(path)
        if not img_dir.exists():
            return []

        return sorted([
                item for item in img_dir.iterdir()
                if item.is_file() and item.suffix.lower() in DataPreprocessor.SUPPORTED_IMAGE_SUFFIXES],
            key=lambda item: item.name)

    @staticmethod
    def SplitOCRCsvLine(line: str) -> List[str]:
        cleaned = str(line).strip().lstrip("\ufeff").strip()
        cleaned = cleaned.lstrip(",")
        if not cleaned:
            return []
        return next(csv.reader([cleaned], skipinitialspace=True))

    @staticmethod
    def ResolveOCRAnnotationLayout(parts: List[str]) -> Optional[Tuple[int, bool]]:
        for coord_count in (8, 4):
            if len(parts) < coord_count + 1:
                continue
            try:
                [float(parts[i]) for i in range(coord_count)]
            except Exception:
                continue
            has_ignore_flag = (
                len(parts) >= coord_count + 2
                and parts[coord_count].strip() in ("0", "1"))
            return coord_count, has_ignore_flag
        return None

    @staticmethod
    def LooksLikeOCRAnnotationLine(line: str) -> bool:
        parts = DataPreprocessor.SplitOCRCsvLine(line)
        return DataPreprocessor.ResolveOCRAnnotationLayout(parts) is not None

    @staticmethod
    def TextFileLooksLikeOCRAnnotations(path: Union[str, Path]) -> bool:
        txt_path = Path(path)
        if not txt_path.exists():
            return False

        for raw_line in txt_path.read_text(encoding="utf-8").splitlines():
            if raw_line.strip():
                return DataPreprocessor.LooksLikeOCRAnnotationLine(raw_line)
        return False

    @staticmethod
    def ParseOCRAnnotationLine(line: str) -> Tuple[np.ndarray, int, str]:
        parts = DataPreprocessor.SplitOCRCsvLine(line)
        if len(parts) < 5:
            raise ValueError(f"OCR annotation line has too few fields: {line!r}")

        layout = DataPreprocessor.ResolveOCRAnnotationLayout(parts)
        if layout is None:
            raise ValueError(f"OCR annotation line has invalid coordinates: {line!r}")
        coord_count, has_ignore_flag = layout

        coords = [float(parts[i]) for i in range(coord_count)]
        idx = coord_count

        if has_ignore_flag:
            ignore_flag = int(parts[idx].strip())
            idx += 1
        else:
            preview_text = ",".join(parts[idx:]).strip() if idx < len(parts) else ""
            ignore_flag = 1 if "#" in preview_text else 0

        text = ",".join(parts[idx:]).strip() if idx < len(parts) else ""
        if ignore_flag:
            text = text.replace("#", "")
        if coord_count == 8:
            xs = coords[0::2]
            ys = coords[1::2]
            box = np.asarray([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)
        else:
            x1, y1, x2, y2 = coords
            box = np.asarray([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)

        return box, int(ignore_flag), text

    @staticmethod
    def LoadOCRAnnotations(path: Union[str, Path]) -> Tuple[np.ndarray, List[str], np.ndarray]:
        boxes: List[np.ndarray] = []
        texts: List[str] = []
        ignore_flags: List[float] = []

        txt_path = Path(path)
        for line_no, raw_line in enumerate(txt_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                box, ignore_flag, text = DataPreprocessor.ParseOCRAnnotationLine(raw_line)
            except Exception as e:
                raise ValueError(f"failed to parse OCR annotation line {line_no} in {txt_path}: {e}") from e

            boxes.append(box.astype(np.float32))
            texts.append(text)
            ignore_flags.append(float(ignore_flag))

        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32), [], np.empty((0,), dtype=np.float32)

        return np.stack(boxes, axis=0).astype(np.float32), texts, np.asarray(ignore_flags, dtype=np.float32)

    @staticmethod
    def ToImageTensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            img_t = torch.from_numpy(image)
        elif isinstance(image, torch.Tensor):
            img_t = image.detach().cpu()
        else:
            raise TypeError(f"unsupported image type: {type(image)}")

        if img_t.ndim == 4:
            if img_t.shape[-1] in (1, 3, 4):
                img_t = img_t[0]
            elif img_t.shape[0] in (1, 3, 4):
                img_t = img_t[..., 0]
            else:
                raise ValueError(f"image must have 2 or 3 dims, but got shape {tuple(img_t.shape)}")

        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(-1)
        if img_t.ndim != 3:
            raise ValueError(f"image must have 2 or 3 dims, but got shape {tuple(img_t.shape)}")

        if img_t.shape[-1] in (1, 3, 4):
            img_t = img_t.permute(2, 0, 1)
        elif img_t.shape[0] not in (1, 3, 4):
            raise ValueError(f"image channel layout is invalid: {tuple(img_t.shape)}")

        img_t = img_t.float()
        if img_t.max().item() > 1.0:
            img_t = img_t / 255.0

        if img_t.size(0) == 1:
            img_t = img_t.repeat(3, 1, 1)
        else:
            img_t = img_t[:3]

        return img_t

    @staticmethod
    def ToNormalTensor(normal: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        normal_t = torch.as_tensor(normal).float()
        if normal_t.ndim == 3 and normal_t.shape[-1] == 3:
            normal_t = normal_t.permute(2, 0, 1)
        if normal_t.ndim != 3 or normal_t.size(0) != 3:
            raise ValueError(f"normal sample must have shape [3, H, W] or [H, W, 3], got {tuple(normal_t.shape)}")
        return F.normalize(normal_t, dim=0, eps=1e-6)

    @staticmethod
    def ToSegmentationTensor(segmentation: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        seg_t = torch.as_tensor(segmentation)
        if seg_t.ndim == 3 and seg_t.size(-1) == 1:
            seg_t = seg_t[..., 0]
        if seg_t.ndim == 3 and seg_t.size(0) == 1:
            seg_t = seg_t[0]
        if seg_t.ndim != 2:
            raise ValueError(f"segmentation sample must have shape [H, W], got {tuple(seg_t.shape)}")
        return seg_t.long()

    @staticmethod
    def ResizeImage(
        imageTensor: torch.Tensor,
        size: Union[int, Tuple[int, int]],
        *,
        antialias: bool = False,) -> Tuple[torch.Tensor, DataResizeMeta]:
        if isinstance(size, int):
            dst_h = int(size)
            dst_w = int(size)
        else:
            dst_h = int(size[0])
            dst_w = int(size[1])

        _, src_h, src_w = imageTensor.shape
        resized = F.interpolate(
            imageTensor.unsqueeze(0),
            size=(dst_h, dst_w),
            mode="bilinear",
            align_corners=False,
            antialias=bool(antialias),).squeeze(0)

        meta = DataResizeMeta(
            src_h=int(src_h),
            src_w=int(src_w),
            dst_h=int(dst_h),
            dst_w=int(dst_w),
            scale_x=float(dst_w) / float(max(1, src_w)),
            scale_y=float(dst_h) / float(max(1, src_h)),)
        return resized, meta

    @staticmethod
    def ToDepthTensor(
        depth: Union[np.ndarray, torch.Tensor],
        valid: Union[np.ndarray, torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_t = torch.as_tensor(depth)
        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0)
        elif depth_t.ndim == 3 and depth_t.shape[-1] == 1:
            depth_t = depth_t.permute(2, 0, 1)
        if depth_t.dtype != torch.float32:
            raise TypeError("depth sample must be float32 metres")
        if depth_t.ndim != 3 or depth_t.size(0) != 1:
            raise ValueError(
                f"depth sample must have shape [H, W] or [1, H, W], got "
                f"{tuple(depth_t.shape)}")
        valid_t = torch.as_tensor(valid)
        if valid_t.ndim == 2:
            valid_t = valid_t.unsqueeze(0)
        elif valid_t.ndim == 3 and valid_t.shape[-1] == 1:
            valid_t = valid_t.permute(2, 0, 1)
        if valid_t.dtype != torch.bool or valid_t.shape != depth_t.shape:
            raise ValueError(
                "depth valid mask must be bool and match the depth sample shape")
        if not bool(torch.isfinite(depth_t).all().item()):
            raise ValueError("depth sample must contain only finite metre values")
        if bool((valid_t & (depth_t <= 0.0)).any().item()):
            raise ValueError("valid depth pixels must be positive")
        return torch.where(valid_t, depth_t, torch.zeros_like(depth_t)), valid_t

    @staticmethod
    def TensorizeSyntheticSupervision(
        annotation: Dict[str, Any],
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depthValid: torch.Tensor,
        normal: torch.Tensor,
        semanticSegmentation: torch.Tensor,
        instanceSegmentation: torch.Tensor,
        *,
        maxNodes: int = 256,
        numGlobalLabels: int = 8,
        textDim: int = 4,
        stateDim: int = 16,
        attrDim: int = 32,
        affordanceDim: int = 8,
        relationClasses: int = 32,) -> Dict[str, torch.Tensor]:
        """Tensorize labels under the coordinate contract in the dataset manifest.

        ``motion`` is an SE(3) spatial delta from the previous object state to
        the current state after camera-egomotion compensation, expressed in the
        current camera-optical axes (metres plus an XYZW quaternion).
        """
        dtype = rgb.dtype
        device = rgb.device
        height, width = rgb.shape[-2:]
        coverage = annotation["coverage"]
        if type(coverage) is not dict or set(coverage) != {
            "relations_exhaustive",
            "contact_events_exhaustive",
        }:
            raise ValueError(
                "synthetic coverage must declare exhaustive relation and contact labels")
        if (
            coverage["relations_exhaustive"] is not True
            or coverage["contact_events_exhaustive"] is not True
        ):
            raise ValueError(
                "relation/contact negatives require exhaustive synthetic annotations")
        temporal = annotation["temporal"]
        if type(temporal["kind_valid"]) is not bool or type(
            temporal["duration_valid"]
        ) is not bool:
            raise TypeError("temporal validity fields must be booleans")
        temporal_kind = int(temporal["kind"])
        temporal_duration_ms = float(temporal["duration_ms"])
        temporal_kind_valid = temporal["kind_valid"]
        temporal_duration_valid = temporal["duration_valid"]
        if not 0 <= temporal_kind < ModuleDim.TemporalPrimitiveCount:
            raise ValueError("temporal.kind is outside the temporal primitive vocabulary")
        if not math.isfinite(temporal_duration_ms):
            raise ValueError("temporal.duration_ms must be finite")
        if temporal_duration_valid != (temporal_duration_ms > 0.0):
            raise ValueError(
                "temporal.duration_ms must be positive exactly when duration_valid is true")
        nodes: List[Tuple[Dict[str, Any], int]] = []

        def flatten(node: Dict[str, Any], parentIndex: int) -> None:
            index = len(nodes)
            nodes.append((node, parentIndex))
            for child in node["parts"]:
                flatten(child, index)

        for obj in annotation["objects"]:
            flatten(obj, -1)

        N = int(maxNodes)
        if N < 1 or len(nodes) > N:
            raise ValueError(
                f"synthetic annotation has {len(nodes)} nodes, maxNodes={N}")
        node_valid = torch.zeros(N, device=device, dtype=torch.bool)
        node_id = torch.full((N,), -1, device=device, dtype=torch.long)
        node_level = torch.zeros(N, device=device, dtype=torch.long)
        parent_index = torch.full((N,), -1, device=device, dtype=torch.long)
        object_classes = torch.zeros(N, device=device, dtype=torch.long)
        part_classes = torch.zeros(N, device=device, dtype=torch.long)
        track_id = torch.zeros(N, device=device, dtype=torch.long)
        pose_camera = torch.zeros(N, 7, device=device, dtype=dtype)
        pose_world = torch.zeros(N, 7, device=device, dtype=dtype)
        size_3d = torch.zeros(N, 3, device=device, dtype=dtype)
        bbox_2d = torch.zeros(N, 4, device=device, dtype=dtype)
        node_instance_masks = torch.zeros(N, height, width, device=device, dtype=torch.bool)
        visible_ratio = torch.zeros(N, device=device, dtype=dtype)
        occlusion_ratio = torch.zeros(N, device=device, dtype=dtype)
        node_state = torch.zeros(N, stateDim, device=device, dtype=dtype)
        node_state_valid = torch.zeros(N, device=device, dtype=torch.bool)
        node_attributes = torch.zeros(N, attrDim, device=device, dtype=dtype)
        node_attributes_valid = torch.zeros(N, device=device, dtype=torch.bool)
        has_text = torch.zeros(N, device=device, dtype=torch.long)
        text_embed = torch.zeros(N, textDim, device=device, dtype=dtype)
        symbol_type = torch.zeros(N, device=device, dtype=torch.long)
        node_lookup: Dict[int, int] = {}

        def validated_pose(value: Any, *, field: str) -> torch.Tensor:
            pose = torch.as_tensor(value, device=device, dtype=dtype)
            if tuple(pose.shape) != (7,) or not bool(torch.isfinite(pose).all().item()):
                raise ValueError(f"{field} must be a finite 7D pose")
            if not torch.allclose(
                pose[3:7].norm(),
                pose.new_tensor(1.0),
                rtol=1e-3,
                atol=1e-3,
            ):
                raise ValueError(f"{field} quaternion must have unit length")
            return pose

        for index, (node, parent) in enumerate(nodes):
            level = int(node["level"])
            current_node_id = int(node["node_id"])
            if current_node_id in node_lookup:
                raise ValueError(f"duplicate synthetic node_id {current_node_id}")
            node_valid[index] = True
            node_id[index] = current_node_id
            node_level[index] = level
            parent_index[index] = int(parent)
            track_id[index] = int(annotation["episode_id"]) * 1000000 + int(node["identity_id"])
            pose_camera[index] = validated_pose(
                node["pose_camera"], field=f"node {current_node_id} pose_camera")
            pose_world[index] = validated_pose(
                node["pose_world"], field=f"node {current_node_id} pose_world")
            node_size = torch.as_tensor(node["size_3d"], device=device, dtype=dtype)
            if (
                tuple(node_size.shape) != (3,)
                or not bool(torch.isfinite(node_size).all().item())
                or bool((node_size <= 0.0).any().item())
            ):
                raise ValueError(f"node {current_node_id} size_3d must be finite and positive")
            size_3d[index] = node_size
            xyxy = torch.tensor(node["bbox_2d"], device=device, dtype=dtype)
            if (
                tuple(xyxy.shape) != (4,)
                or not bool(torch.isfinite(xyxy).all().item())
                or not (
                    0.0 <= float(xyxy[0]) < float(xyxy[2]) <= float(width)
                    and 0.0 <= float(xyxy[1]) < float(xyxy[3]) <= float(height)
                )
            ):
                raise ValueError(f"node {current_node_id} bbox_2d is outside the image")
            bbox_2d[index] = xyxy / xyxy.new_tensor([width, height, width, height])
            node_instance_masks[index] = instanceSegmentation.eq(int(node["instance_id"]))
            visible_ratio[index] = float(node["visible_ratio"])
            occlusion_ratio[index] = float(node["occlusion_ratio"])
            if not (
                0.0 <= float(visible_ratio[index]) <= 1.0
                and 0.0 <= float(occlusion_ratio[index]) <= 1.0
            ):
                raise ValueError(
                    f"node {current_node_id} visibility ratios must be in [0, 1]")
            if level == 0:
                object_classes[index] = int(node["object_class"])
                node_state[index] = torch.tensor(node["object_state"], device=device, dtype=dtype)
                node_state_valid[index] = True
                node_attributes[index] = torch.tensor(node["object_attributes"], device=device, dtype=dtype)
                node_attributes_valid[index] = True
            else:
                part_classes[index] = int(node["part_class"])
                node_state[index] = torch.tensor(node["part_state"], device=device, dtype=dtype)
                node_state_valid[index] = True
                has_text[index] = int(node["has_text"])
                symbol_type[index] = int(node["symbol_type"])
                if int(node["has_text"]) == 1:
                    text_embed[index] = torch.tensor(node["text_embed"], device=device, dtype=dtype)
            node_lookup[current_node_id] = index

        relation_type = torch.zeros(N, N, device=device, dtype=torch.long)
        relation_valid = node_valid.unsqueeze(1) & node_valid.unsqueeze(0)
        relation_valid = relation_valid & ~torch.eye(N, device=device, dtype=torch.bool)
        external_relation = torch.zeros(N, relationClasses, device=device, dtype=dtype)
        external_relation_valid = node_valid.clone()
        for relation in annotation["relations"]:
            subject = int(relation["subject_node_id"])
            obj = int(relation["object_node_id"])
            relation_id = int(relation["relation_type"])
            if subject not in node_lookup:
                raise ValueError(f"relation references unknown subject node_id {subject}")
            if not 0 <= relation_id < relationClasses:
                raise ValueError(f"relation_type {relation_id} is outside the vocabulary")
            if obj in node_lookup:
                relation_type[node_lookup[subject], node_lookup[obj]] = relation_id
            else:
                external_relation[node_lookup[subject], relation_id] = 1.0

        motion = torch.zeros(N, 7, device=device, dtype=dtype)
        motion[:, 6] = 1.0
        motion_valid = torch.zeros(N, device=device, dtype=torch.bool)
        is_moving = torch.zeros(N, device=device, dtype=dtype)
        for entry in annotation["motion"]["object_motions_from_prev"]:
            entry_node_id = int(entry["node_id"])
            if entry_node_id not in node_lookup:
                raise ValueError(f"motion references unknown node_id {entry_node_id}")
            index = node_lookup[entry_node_id]
            if bool(motion_valid[index].item()):
                raise ValueError(f"duplicate motion label for node_id {entry_node_id}")
            motion[index] = validated_pose(
                entry["motion"], field=f"node {entry_node_id} motion")
            motion_valid[index] = True
            is_moving[index] = float(entry["is_moving"])
            if not 0.0 <= float(is_moving[index]) <= 1.0:
                raise ValueError(f"node {entry_node_id} is_moving must be in [0, 1]")

        affordance = torch.zeros(N, affordanceDim, device=device, dtype=dtype)
        affordance_valid = torch.zeros(N, device=device, dtype=torch.bool)
        affordance_keys = (
            "graspable", "pushable", "pressable", "pullable",
            "rotatable", "openable", "container", "support_surface")
        for entry in annotation["interaction"]["affordance_targets"]:
            entry_node_id = int(entry["node_id"])
            if entry_node_id not in node_lookup:
                raise ValueError(f"affordance references unknown node_id {entry_node_id}")
            index = node_lookup[entry_node_id]
            affordance_value = torch.tensor(
                [entry[name] for name in affordance_keys],
                device=device,
                dtype=dtype)
            if bool(((affordance_value < 0.0) | (affordance_value > 1.0)).any().item()):
                raise ValueError(f"node {entry_node_id} affordances must be in [0, 1]")
            affordance[index] = affordance_value
            affordance_valid[index] = True

        contact = torch.zeros(N, device=device, dtype=dtype)
        contact_valid = node_valid.clone()
        contact_force = torch.zeros(N, 2, device=device, dtype=dtype)
        contact_point_camera = torch.zeros(N, 3, device=device, dtype=dtype)
        for event in annotation["interaction"]["contact_events"]:
            entry_node_id = int(event["actor_b_node_id"])
            if entry_node_id not in node_lookup:
                raise ValueError(f"contact references unknown node_id {entry_node_id}")
            index = node_lookup[entry_node_id]
            contact[index] = 1.0
            force = torch.tensor(
                [event["normal_force_n"], event["tangential_force_n"]],
                device=device,
                dtype=dtype)
            point = torch.tensor(
                event["contact_point_camera"], device=device, dtype=dtype)
            if (
                tuple(point.shape) != (3,)
                or not bool(torch.isfinite(point).all().item())
                or not bool(torch.isfinite(force).all().item())
                or bool((force < 0.0).any().item())
            ):
                raise ValueError(f"contact for node {entry_node_id} is invalid")
            contact_force[index] = force
            contact_point_camera[index] = point

        global_labels = annotation["scene"]["global_labels"]
        if len(global_labels) != numGlobalLabels:
            raise ValueError(
                f"scene.global_labels must contain exactly {numGlobalLabels} values")

        return {
            "rgb": rgb,
            "depth": depth,
            "depth_valid": depthValid,
            "normal": normal,
            "normal_valid": (
                normal.norm(dim=0, keepdim=True) > 0.5
            ),
            "semantic_segmentation": semanticSegmentation.long(),
            "instance_segmentation": instanceSegmentation.long(),
            "scene_class": torch.tensor(annotation["scene"]["scene_class"], device=device, dtype=torch.long),
            "global_labels": torch.tensor(global_labels, device=device, dtype=dtype),
            "node_valid": node_valid,
            "node_id": node_id,
            "node_level": node_level,
            "parent_index": parent_index,
            "object_classes": object_classes,
            "part_classes": part_classes,
            "track_id": track_id,
            "pose_camera": pose_camera,
            "pose_world": pose_world,
            "size_3d": size_3d,
            "bbox_2d": bbox_2d,
            "node_instance_masks": node_instance_masks,
            "visible_ratio": visible_ratio,
            "occlusion_ratio": occlusion_ratio,
            "node_state": node_state,
            "node_state_valid": node_state_valid,
            "node_attributes": node_attributes,
            "node_attributes_valid": node_attributes_valid,
            "has_text": has_text,
            "text_embed": text_embed,
            "symbol_type": symbol_type,
            "relation_type": relation_type,
            "relation_valid": relation_valid,
            "external_relation": external_relation,
            "external_relation_valid": external_relation_valid,
            "motion": motion,
            "motion_valid": motion_valid,
            "is_moving": is_moving,
            "affordance": affordance,
            "affordance_valid": affordance_valid,
            "contact": contact,
            "contact_valid": contact_valid,
            "contact_force": contact_force,
            "contact_point_camera": contact_point_camera,
            "temporal_kind": torch.tensor(
                temporal_kind, device=device, dtype=torch.long),
            "temporal_kind_valid": torch.tensor(
                temporal_kind_valid, device=device, dtype=torch.bool),
            "temporal_duration_ms": torch.tensor(
                temporal_duration_ms, device=device, dtype=dtype),
            "temporal_duration_valid": torch.tensor(
                temporal_duration_valid, device=device, dtype=torch.bool),}

    @staticmethod
    def CollateSyntheticSupervision(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {name: torch.stack([sample[name] for sample in batch], dim=0) for name in batch[0]}

    @staticmethod
    def ScaleBoxesXYXY(
        boxes: Union[np.ndarray, torch.Tensor],
        resizeMeta: DataResizeMeta,
        *,
        clamp: bool = True,) -> np.ndarray:
        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        if boxes_np.size == 0:
            return boxes_np.reshape(0, 4)

        boxes_np[:, [0, 2]] *= float(resizeMeta.scale_x)
        boxes_np[:, [1, 3]] *= float(resizeMeta.scale_y)

        if clamp:
            boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0.0, float(resizeMeta.dst_w))
            boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0.0, float(resizeMeta.dst_h))

        return boxes_np

    @staticmethod
    def RestoreBoxesXYXY(
        boxes: Union[np.ndarray, torch.Tensor],
        resizeMeta: DataResizeMeta,
        *,
        clamp: bool = True,) -> np.ndarray:
        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
        if boxes_np.size == 0:
            return boxes_np.reshape(0, 4)

        boxes_np[:, [0, 2]] /= float(max(1e-6, resizeMeta.scale_x))
        boxes_np[:, [1, 3]] /= float(max(1e-6, resizeMeta.scale_y))

        if clamp:
            boxes_np[:, [0, 2]] = np.clip(boxes_np[:, [0, 2]], 0.0, float(resizeMeta.src_w))
            boxes_np[:, [1, 3]] = np.clip(boxes_np[:, [1, 3]], 0.0, float(resizeMeta.src_h))

        return boxes_np

    @staticmethod
    def PrepareImageAndBoxes(
        image: Union[np.ndarray, torch.Tensor],
        boxes: Union[np.ndarray, torch.Tensor],
        *,
        size: Union[int, Tuple[int, int]],) -> Tuple[torch.Tensor, np.ndarray, DataResizeMeta]:
        image_t = DataPreprocessor.ToImageTensor(image)
        resized_image_t, resize_meta = DataPreprocessor.ResizeImage(image_t, size=size)
        scaled_boxes = DataPreprocessor.ScaleBoxesXYXY(boxes, resize_meta, clamp=True)
        return resized_image_t, scaled_boxes, resize_meta

    @staticmethod
    def TensorizeFeedbackSignal(
        value: Any,
        *,
        name: str,
        batchSize: int,
        device: Optional[torch.device],) -> Optional[torch.Tensor]:
        """Validate external reward/done feedback once at the input seam."""
        if value is None:
            return None
        try:
            value_tensor = torch.as_tensor(value)
        except Exception as error:
            raise ValueError(f"{name} must contain real numbers") from error
        if value_tensor.dtype == torch.bool or not (
            value_tensor.is_floating_point()
            or value_tensor.dtype in (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
        ):
            raise TypeError(f"{name} must contain real numbers")
        if tuple(value_tensor.shape) != (batchSize,):
            raise ValueError(f"{name} must have shape [{batchSize}]")
        if not bool(torch.isfinite(value_tensor).all().item()):
            raise ValueError(f"{name} must contain only finite values")
        if name == "reward":
            if bool((
                (value_tensor < float(BasicParameters.REWARD_MIN))
                | (value_tensor > float(BasicParameters.REWARD_MAX))
            ).any().item()):
                raise ValueError(
                    f"reward must be in [{BasicParameters.REWARD_MIN}, "
                    f"{BasicParameters.REWARD_MAX}]")
        elif name == "done":
            if bool(((value_tensor != 0) & (value_tensor != 1)).any().item()):
                raise ValueError("done must contain only binary 0/1 flags")
        else:
            raise ValueError(f"unsupported feedback signal: {name}")
        return value_tensor.to(device=device, dtype=torch.float32)

    @staticmethod
    def ConvertRobotInputs(
        imgs: Union[np.ndarray, torch.Tensor],
        reward: Optional[Union[np.ndarray, torch.Tensor]],
        done: Optional[Union[np.ndarray, torch.Tensor]],
        *,
        device: Optional[torch.device] = None,
        needVisualState: bool = True,
        depths: Union[np.ndarray, torch.Tensor],
        depthValids: Union[np.ndarray, torch.Tensor],) -> Dict[str, Any]:
        image_size = BasicParameters.IMAGE_SIZE
        img_tensor = torch.as_tensor(imgs)
        depth_tensor = torch.as_tensor(depths)
        depth_valid_tensor = torch.as_tensor(depthValids)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0)
        if depth_valid_tensor.ndim == 2:
            depth_valid_tensor = depth_valid_tensor.unsqueeze(0)
        if (
            img_tensor.dtype != torch.uint8
            or tuple(img_tensor.shape[1:]) != (image_size, image_size, 3)
        ):
            raise ValueError(
                f"RGB must be uint8 [B, {image_size}, {image_size}, 3]")
        if (
            depth_tensor.dtype != torch.float32
            or tuple(depth_tensor.shape[1:]) != (image_size, image_size)
        ):
            raise ValueError(
                f"depth must be float32 metres [B, {image_size}, {image_size}]")
        if (
            depth_valid_tensor.dtype != torch.bool
            or tuple(depth_valid_tensor.shape) != tuple(depth_tensor.shape)
        ):
            raise ValueError("depth_valid must be a bool mask matching depth")
        batch_size = int(img_tensor.size(0))
        if int(depth_tensor.size(0)) != batch_size:
            raise ValueError("RGB, depth and depth_valid batch sizes must match")
        if not bool(torch.isfinite(depth_tensor).all().item()):
            raise ValueError("depth must contain only finite metre values")
        if bool((depth_valid_tensor & (depth_tensor <= 0.0)).any().item()):
            raise ValueError("valid depth pixels must be positive")

        original_images: List[np.ndarray] = []
        resize_meta: List[DataResizeMeta] = []
        if needVisualState:
            original_images = [
                np.array(sample.detach().cpu().numpy(), copy=True)
                for sample in img_tensor]
            resize_meta = [
                DataResizeMeta(
                    src_h=image_size,
                    src_w=image_size,
                    dst_h=image_size,
                    dst_w=image_size,
                    scale_x=1.0,
                    scale_y=1.0)
                for _ in range(batch_size)]

        frame_tensor = img_tensor.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        depth_tensor = depth_tensor.unsqueeze(1)
        depth_valid_tensor = depth_valid_tensor.unsqueeze(1)
        if device is not None:
            frame_tensor = frame_tensor.to(device)
            depth_tensor = depth_tensor.to(device)
            depth_valid_tensor = depth_valid_tensor.to(device)

        return {
            "frames": frame_tensor,
            "original_images": original_images,
            "resize_meta": resize_meta,
            "depths": depth_tensor,
            "depth_valid": depth_valid_tensor,
            "rewards": DataPreprocessor.TensorizeFeedbackSignal(
                reward,
                name="reward",
                batchSize=batch_size,
                device=device),
            "dones": DataPreprocessor.TensorizeFeedbackSignal(
                done,
                name="done",
                batchSize=batch_size,
                device=device),}

    @staticmethod
    def PreprocessSingleFrame(
        bitmap: Union[List[Any], np.ndarray, torch.Tensor],
        reward: Optional[float],
        done: Optional[float],
        *,
        depthBitmap: Union[List[Any], np.ndarray, torch.Tensor],
        depthValid: Union[List[Any], np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
        needVisualState: bool = False,) -> Dict[str, Any]:
        if isinstance(bitmap, torch.Tensor):
            rgb = bitmap
        else:
            bitmap_array = np.asarray(bitmap)
            if not np.issubdtype(bitmap_array.dtype, np.integer):
                raise TypeError("wire RGB input must contain integers")
            if bitmap_array.size and (
                int(bitmap_array.min()) < 0 or int(bitmap_array.max()) > 255
            ):
                raise ValueError("wire RGB values must be in [0, 255]")
            rgb = torch.from_numpy(bitmap_array.astype(np.uint8, copy=False))

        if isinstance(depthBitmap, torch.Tensor):
            depth = depthBitmap
        else:
            depth_array = np.asarray(depthBitmap)
            if not np.issubdtype(depth_array.dtype, np.floating):
                raise TypeError("wire depth input must contain metre values")
            depth = torch.from_numpy(depth_array.astype(np.float32, copy=False))

        if isinstance(depthValid, torch.Tensor):
            depth_valid = depthValid
        else:
            valid_array = np.asarray(depthValid)
            if valid_array.dtype != np.bool_:
                raise TypeError("wire depth valid input must contain booleans")
            depth_valid = torch.from_numpy(valid_array)

        image_size = BasicParameters.IMAGE_SIZE
        if rgb.dtype != torch.uint8:
            raise TypeError("RGB input must use uint8 rgb8 samples")
        if tuple(rgb.shape) != (image_size, image_size, 3):
            raise ValueError(
                f"RGB input must have shape [{image_size}, {image_size}, 3]")
        if depth.dtype != torch.float32:
            raise TypeError("depth input must use float32 metre values")
        if tuple(depth.shape) != (image_size, image_size):
            raise ValueError(
                f"depth input must have shape [{image_size}, {image_size}]")
        if depth_valid.dtype != torch.bool:
            raise TypeError("depth valid input must use bool samples")
        if tuple(depth_valid.shape) != (image_size, image_size):
            raise ValueError(
                f"depth valid input must have shape [{image_size}, {image_size}]")
        if not bool(torch.isfinite(depth).all().item()):
            raise ValueError("depth input must contain only finite metre values")
        if bool((depth_valid & (depth <= 0.0)).any().item()):
            raise ValueError("valid depth pixels must be positive")

        original_images: List[np.ndarray] = []
        resize_meta: List[DataResizeMeta] = []
        if needVisualState:
            original_images.append(np.array(rgb.detach().cpu().numpy(), copy=True))
            resize_meta.append(DataResizeMeta(
                src_h=image_size,
                src_w=image_size,
                dst_h=image_size,
                dst_w=image_size,
                scale_x=1.0,
                scale_y=1.0,))

        frame = rgb.permute(2, 0, 1).contiguous().unsqueeze(0).float().div_(255.0)
        depth = depth.unsqueeze(0).unsqueeze(0)
        depth_valid = depth_valid.unsqueeze(0).unsqueeze(0)
        if device is not None:
            frame = frame.to(device)
            depth = depth.to(device)
            depth_valid = depth_valid.to(device)

        reward_value = DataPreprocessor.TensorizeFeedbackSignal(
            None if reward is None else [reward],
            name="reward",
            batchSize=1,
            device=device)
        done_value = DataPreprocessor.TensorizeFeedbackSignal(
            None if done is None else [done],
            name="done",
            batchSize=1,
            device=device)

        return {
            "frames": frame,
            "original_images": original_images,
            "resize_meta": resize_meta,
            "depths": depth,
            "depth_valid": depth_valid,
            "rewards": reward_value,
            "dones": done_value,}

    @staticmethod
    def ConvertCppCameraFrame(
        bitmap: Union[List[Any], np.ndarray, torch.Tensor],
        reward: Optional[float],
        done: Optional[float],
        *,
        depthBitmap: Union[List[Any], np.ndarray, torch.Tensor],
        depthValid: Union[List[Any], np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
        needVisualState: bool = False,) -> Dict[str, Any]:
        return DataPreprocessor.PreprocessSingleFrame(
            bitmap,
            reward,
            done,
            depthBitmap=depthBitmap,
            depthValid=depthValid,
            device=device,
            needVisualState=needVisualState,)

    @staticmethod
    def CropAndResizeLineImagesWithMeta(
        imageTensor: torch.Tensor,
        boxes: Union[np.ndarray, torch.Tensor],
        *,
        targetH: int = 32,
        maxW: int = 512,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c, h_img, w_img = imageTensor.shape
        if c < 3:
            raise ValueError(f"imageTensor channel count must be at least 3, but got {c}")

        gray = (
            0.299 * imageTensor[0]
            + 0.587 * imageTensor[1]
            + 0.114 * imageTensor[2])

        device = imageTensor.device
        line_tensors: List[torch.Tensor] = []
        valid_widths: List[int] = []
        truncated_flags: List[bool] = []
        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

        for box in boxes_np:
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            patch = gray[y1:y2, x1:x2]
            h, w = patch.shape
            if h < 1 or w < 1:
                continue

            patch = patch.unsqueeze(0).unsqueeze(0)

            scale = targetH / float(h)
            new_w = max(1, int(round(w * scale)))
            truncated = new_w > maxW
            patch_resized = F.interpolate(
                patch,
                size=(targetH, new_w),
                mode="bilinear",
                align_corners=False,)

            if new_w > maxW:
                patch_resized = patch_resized[:, :, :, :maxW]
                new_w = maxW

            pad = torch.zeros(1, 1, targetH, maxW, dtype=imageTensor.dtype, device=device)
            pad[:, :, :, :new_w] = patch_resized
            line_tensors.append(pad)
            valid_widths.append(int(new_w))
            truncated_flags.append(bool(truncated))

        if not line_tensors:
            return (
                torch.empty(0, 1, targetH, maxW, dtype=imageTensor.dtype, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.bool, device=device),)

        return (
            torch.cat(line_tensors, dim=0),
            torch.tensor(valid_widths, dtype=torch.long, device=device),
            torch.tensor(truncated_flags, dtype=torch.bool, device=device),)

    @staticmethod
    def NormalizeTextLine(text: Optional[str], char2Idx: dict) -> str:
        if text is None:
            return ""
        cleaned = str(text).strip()
        return "".join(ch for ch in cleaned if ch in char2Idx)

    @staticmethod
    def OCRBuildDbTargets(
        boxes: Union[np.ndarray, torch.Tensor],
        *,
        imageHeight: int,
        imageWidth: int,
        dtype: torch.dtype = torch.float32,) -> Tuple[torch.Tensor, torch.Tensor]:
        gt_boxes = torch.zeros(1, imageHeight, imageWidth, dtype=dtype)
        gt_mask = torch.ones(1, imageHeight, imageWidth, dtype=dtype)

        boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for box in boxes_np:
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            x1 = int(round(x1))
            y1 = int(round(y1))
            x2 = int(round(x2))
            y2 = int(round(y2))

            x1 = max(0, min(imageWidth - 1, x1))
            y1 = max(0, min(imageHeight - 1, y1))
            x2 = max(x1 + 1, min(imageWidth, x2))
            y2 = max(y1 + 1, min(imageHeight, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            gt_boxes[:, y1:y2, x1:x2] = 1.0

        return gt_boxes, gt_mask

    @staticmethod
    def PrepareOCRSample(
        image: Union[np.ndarray, torch.Tensor],
        boxes: Union[np.ndarray, torch.Tensor],
        texts: List[str],
        *,
        ignoreFlags: Optional[Union[np.ndarray, torch.Tensor, List[float], List[int]]] = None,
        char2Idx: dict,
        imageSize: int,
        targetH: int = 32,
        maxW: int = 512,
        device: Optional[torch.device] = None,) -> Dict[str, Any]:
        img_rgb, boxes_np, resize_meta = DataPreprocessor.PrepareImageAndBoxes(
            image,
            boxes,
            size=imageSize,)

        texts_list = list(texts)
        if len(boxes_np) != len(texts_list):
            raise ValueError(f"OCR texts/boxes count mismatch in sample: {len(texts_list)} vs {len(boxes_np)}")
        if ignoreFlags is None:
            ignore_flags_np = np.zeros((len(texts_list),), dtype=np.float32)
        else:
            ignore_flags_np = np.asarray(ignoreFlags, dtype=np.float32).reshape(-1)
        if len(ignore_flags_np) != len(texts_list):
            raise ValueError(f"OCR ignore flags/texts count mismatch in sample: {len(ignore_flags_np)} vs {len(texts_list)}")

        det_boxes: List[np.ndarray] = []
        rec_boxes: List[np.ndarray] = []
        rec_texts: List[str] = []

        for box, raw_text, ignore_flag in zip(boxes_np, texts_list, ignore_flags_np):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(imageSize - 1, x1))
            y1 = max(0, min(imageSize - 1, y1))
            x2 = max(x1 + 1, min(imageSize, x2))
            y2 = max(y1 + 1, min(imageSize, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            det_boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))

            if bool(ignore_flag):
                continue
            text = DataPreprocessor.NormalizeTextLine(raw_text, char2Idx)
            if not text:
                continue

            rec_boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))
            rec_texts.append(text)

        gt_boxes, gt_mask = DataPreprocessor.OCRBuildDbTargets(
            det_boxes,
            imageHeight=imageSize,
            imageWidth=imageSize,
            dtype=img_rgb.dtype,)

        if len(rec_boxes) == 0:
            recog_imgs = torch.empty(0, 1, targetH, maxW, dtype=img_rgb.dtype)
            recog_widths = torch.empty(0, dtype=torch.long)
            recog_truncated = torch.empty(0, dtype=torch.bool)
            targets = torch.empty(0, dtype=torch.long)
            target_lengths = torch.empty(0, dtype=torch.long)
        else:
            recog_imgs, recog_widths, recog_truncated = DataPreprocessor.CropAndResizeLineImagesWithMeta(
                img_rgb,
                rec_boxes,
                targetH=targetH,
                maxW=maxW,)

            flat_targets: List[int] = []
            target_lengths_list: List[int] = []
            for text in rec_texts:
                ids = [int(char2Idx[ch]) for ch in text]
                flat_targets.extend(ids)
                target_lengths_list.append(len(ids))

            targets = torch.tensor(flat_targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths_list, dtype=torch.long)

        if device is not None:
            img_rgb = img_rgb.to(device)
            gt_boxes = gt_boxes.to(device)
            gt_mask = gt_mask.to(device)
            recog_imgs = recog_imgs.to(device)
            recog_widths = recog_widths.to(device)
            recog_truncated = recog_truncated.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

        return {
            "detect_img": img_rgb,
            "scaled_boxes": boxes_np,
            "det_boxes": det_boxes,
            "rec_boxes": rec_boxes,
            "gt_boxes": gt_boxes,
            "gt_mask": gt_mask,
            "recog_imgs": recog_imgs,
            "recog_widths": recog_widths,
            "recog_truncated": recog_truncated,
            "targets": targets,
            "target_lengths": target_lengths,
            "norm_texts": rec_texts,
            "ignore_flags": ignore_flags_np,
            "resize_meta": resize_meta,}

    @staticmethod
    def PrepareOCRRecognitionSample(
        image: Union[np.ndarray, torch.Tensor],
        text: Optional[str],
        *,
        ignoreFlag: Union[bool, int, float] = False,
        char2Idx: dict,
        targetH: int = 32,
        maxW: int = 512,
        device: Optional[torch.device] = None,) -> Dict[str, Any]:
        image_t = DataPreprocessor.ToImageTensor(image)
        _, h_img, w_img = image_t.shape

        recog_imgs, recog_widths, recog_truncated = DataPreprocessor.CropAndResizeLineImagesWithMeta(
            image_t,
            np.asarray([[0.0, 0.0, float(w_img), float(h_img)]], dtype=np.float32),
            targetH=targetH,
            maxW=maxW,)

        ignore = bool(ignoreFlag)
        norm_text = "" if ignore else DataPreprocessor.NormalizeTextLine(text, char2Idx)

        if recog_imgs.size(0) == 0 or not norm_text:
            targets = torch.empty(0, dtype=torch.long)
            target_lengths = torch.empty(0, dtype=torch.long)
        else:
            ids = [int(char2Idx[ch]) for ch in norm_text]
            targets = torch.tensor(ids, dtype=torch.long)
            target_lengths = torch.tensor([len(ids)], dtype=torch.long)

        if device is not None:
            recog_imgs = recog_imgs.to(device)
            recog_widths = recog_widths.to(device)
            recog_truncated = recog_truncated.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

        return {
            "recog_imgs": recog_imgs,
            "recog_widths": recog_widths,
            "recog_truncated": recog_truncated,
            "targets": targets,
            "target_lengths": target_lengths,
            "norm_text": norm_text,
            "ignore": ignore,}
