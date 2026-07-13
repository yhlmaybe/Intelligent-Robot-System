from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from Config import BasicParameters
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
    def __init__(self, isTest: bool = False) -> None:
        if isTest:
            p = Path(BasicParameters.DATA_ROOT_PATH_TEST)
            self.imgs = sorted((p / "frames").glob("*.png"))
            self.reward = sorted((p / "reward").glob("*.npy"))
            self.done = sorted((p / "done").glob("*.npy"))
            self.depths = ListDepthFiles(getattr(BasicParameters, "DATA_DEPTH_PATH_TEST", p / "depth"))
            self.depth_valids = ListDepthFiles(getattr(BasicParameters, "DATA_DEPTH_VALID_PATH_TEST", p / "depth_valid"))
            self.texts = sorted((p / "texts").glob("*.txt"))
            self.actions = sorted((p / "actions").glob("*.npy"))
            self.normals = ListArrayFiles(getattr(BasicParameters, "DATA_NORMAL_PATH_TEST", p / "normal"))
            self.semantic_segmentations = ListArrayFiles(getattr(BasicParameters, "DATA_SEMANTIC_SEGMENTATION_PATH_TEST", p / "semantic_segmentation"))
            self.instance_segmentations = ListArrayFiles(getattr(BasicParameters, "DATA_INSTANCE_SEGMENTATION_PATH_TEST", p / "instance_segmentation"))
            self.synthetic_annotations = ListJsonFiles(getattr(BasicParameters, "DATA_SYNTHETIC_SUPERVISION_PATH_TEST", p / "synthetic_supervision"))
        else:
            self.imgs = sorted(Path(BasicParameters.DATA_FRAMES_PATH).glob("*.png"))
            self.reward = sorted(Path(BasicParameters.DATA_REWARD_PATH).glob("*.npy"))
            self.done = sorted(Path(BasicParameters.DATA_DONE_PATH).glob("*.npy"))
            self.depths = ListDepthFiles(BasicParameters.DATA_DEPTH_PATH)
            self.depth_valids = ListDepthFiles(BasicParameters.DATA_DEPTH_VALID_PATH)
            self.texts = sorted(Path(BasicParameters.DATA_TEXTS_PATH).glob("*.txt"))
            self.actions = sorted(Path(BasicParameters.DATA_ACTIONS_PATH).glob("*.npy"))
            self.normals = ListArrayFiles(BasicParameters.DATA_NORMAL_PATH)
            self.semantic_segmentations = ListArrayFiles(BasicParameters.DATA_SEMANTIC_SEGMENTATION_PATH)
            self.instance_segmentations = ListArrayFiles(BasicParameters.DATA_INSTANCE_SEGMENTATION_PATH)
            self.synthetic_annotations = ListJsonFiles(BasicParameters.DATA_SYNTHETIC_SUPERVISION_PATH)

        assert len(self.imgs) == len(self.reward) == len(self.done) == len(self.depths), "frames/reward/done/depth The number of files is inconsistent."
        if self.depth_valids:
            assert len(self.depth_valids) == len(self.imgs), "depth_valid The number of files is inconsistent."
        if self.texts:
            assert len(self.texts) == len(self.imgs), "texts The number of files is inconsistent."
        if self.actions:
            assert len(self.actions) == len(self.imgs), "actions The number of files is inconsistent."
        if self.synthetic_annotations:
            assert len(self.synthetic_annotations) == len(self.imgs), "synthetic_supervision The number of files is inconsistent."
            assert len(self.normals) == len(self.imgs), "normal The number of files is inconsistent."
            assert len(self.semantic_segmentations) == len(self.imgs), "semantic_segmentation The number of files is inconsistent."
            assert len(self.instance_segmentations) == len(self.imgs), "instance_segmentation The number of files is inconsistent."

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        imgs = LoadImageFirstFrame(self.imgs[idx])
        reward = np.load(self.reward[idx]).astype(np.float32)
        done = np.load(self.done[idx]).astype(np.float32)
        depth = LoadDepthArray(self.depths[idx])
        if self.depth_valids:
            depth_valid = LoadDepthArray(self.depth_valids[idx]).astype(bool)
        else:
            depth_valid = np.isfinite(depth) & (depth > 0)
        ext_text = ""
        if self.texts:
            ext_text = self.texts[idx].read_text(encoding="utf-8").strip()
        # Per-frame executed endpoint poses [endpoint_count, 7] (xyz + xyzw quaternion);
        # identity poses until the recording pipeline supplies real actions.
        if self.actions:
            action = np.load(self.actions[idx]).astype(np.float32)
        else:
            action = np.zeros((ModuleDim.DecisionEndpointCount, ModuleDim.DecisionEndpointPoseDim), dtype=np.float32)
            action[:, 6] = 1.0
        synthetic_targets: Dict[str, torch.Tensor] = {}
        if self.synthetic_annotations:
            annotation = json.loads(self.synthetic_annotations[idx].read_text(encoding="utf-8"))
            rgb_tensor = DataPreprocessor.ToImageTensor(imgs)
            depth_tensor, depth_valid_tensor = DataPreprocessor.ToDepthTensor(
                depth,
                depth_valid,
                depthScaleMeters=BasicParameters.DATA_DEPTH_SCALE_METERS)
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
        return imgs, reward, done, depth, depth_valid, ext_text, action, synthetic_targets


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
        valid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        *,
        depthScaleMeters: float = 1.0,) -> Tuple[torch.Tensor, torch.Tensor]:
        depth_t = torch.as_tensor(depth).float()
        if depth_t.ndim == 2:
            depth_t = depth_t.unsqueeze(0)
        elif depth_t.ndim == 3 and depth_t.shape[-1] == 1:
            depth_t = depth_t.permute(2, 0, 1)
        if depth_t.ndim != 3 or depth_t.size(0) != 1:
            raise ValueError(f"depth sample must have shape [H, W] or [1, H, W], got {tuple(depth_t.shape)}")
        depth_t = depth_t * float(depthScaleMeters)
        depth_valid = torch.isfinite(depth_t) & (depth_t > 0.0)

        if valid is not None:
            valid_t = torch.as_tensor(valid)
            if valid_t.ndim == 2:
                valid_t = valid_t.unsqueeze(0)
            elif valid_t.ndim == 3 and valid_t.shape[-1] == 1:
                valid_t = valid_t.permute(2, 0, 1)
            if valid_t.shape != depth_t.shape:
                raise ValueError(f"depth valid mask must match depth sample shape, got {tuple(valid_t.shape)}")
            depth_valid = depth_valid & valid_t.bool()

        depth_t = torch.where(depth_valid, depth_t, torch.zeros_like(depth_t))
        return depth_t, depth_valid

    @staticmethod
    def ResizeDepth(
        depthTensor: torch.Tensor,
        validTensor: torch.Tensor,
        size: Union[int, Tuple[int, int]],) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(size, int):
            dst_size = (int(size), int(size))
        else:
            dst_size = (int(size[0]), int(size[1]))
        valid_float = validTensor.float()
        inverse = torch.where(validTensor, depthTensor.clamp_min(1e-6).reciprocal(), torch.zeros_like(depthTensor))
        valid_weight = F.interpolate(valid_float.unsqueeze(0), size=dst_size, mode="area").squeeze(0)
        inverse_sum = F.interpolate(inverse.unsqueeze(0), size=dst_size, mode="area").squeeze(0)
        valid = valid_weight > 1e-6
        resized_inverse = inverse_sum / valid_weight.clamp_min(1e-6)
        depth = resized_inverse.clamp_min(1e-6).reciprocal()
        return torch.where(valid, depth, torch.zeros_like(depth)), valid

    @staticmethod
    def ResizeCameraIntrinsics(
        cameraIntrinsics: Union[np.ndarray, torch.Tensor],
        sourceSize: Tuple[int, int],
        targetSize: Tuple[int, int],
        *,
        device: Optional[torch.device] = None,) -> torch.Tensor:
        intrinsics = torch.as_tensor(cameraIntrinsics).float().clone()
        src_h, src_w = int(sourceSize[0]), int(sourceSize[1])
        dst_h, dst_w = int(targetSize[0]), int(targetSize[1])
        sx = float(dst_w) / float(max(1, src_w))
        sy = float(dst_h) / float(max(1, src_h))
        intrinsics[..., 0, 0] *= sx
        intrinsics[..., 1, 1] *= sy
        intrinsics[..., 0, 2] *= sx
        intrinsics[..., 1, 2] *= sy
        return intrinsics.to(device) if device is not None else intrinsics

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
        dtype = rgb.dtype
        device = rgb.device
        height, width = rgb.shape[-2:]
        nodes: List[Tuple[Dict[str, Any], int]] = []

        def flatten(node: Dict[str, Any], parentIndex: int) -> None:
            index = len(nodes)
            nodes.append((node, parentIndex))
            for child in node["parts"]:
                flatten(child, index)

        for obj in annotation["objects"]:
            flatten(obj, -1)

        N = int(maxNodes)
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

        for index, (node, parent) in enumerate(nodes):
            level = int(node["level"])
            node_valid[index] = True
            node_id[index] = int(node["node_id"])
            node_level[index] = level
            parent_index[index] = int(parent)
            track_id[index] = int(annotation["episode_id"]) * 1000000 + int(node["identity_id"])
            pose_camera[index] = torch.tensor(node["pose_camera"], device=device, dtype=dtype)
            pose_world[index] = torch.tensor(node["pose_world"], device=device, dtype=dtype)
            size_3d[index] = torch.tensor(node["size_3d"], device=device, dtype=dtype)
            xyxy = torch.tensor(node["bbox_2d"], device=device, dtype=dtype)
            bbox_2d[index] = xyxy / xyxy.new_tensor([width, height, width, height])
            node_instance_masks[index] = instanceSegmentation.eq(int(node["instance_id"]))
            visible_ratio[index] = float(node["visible_ratio"])
            occlusion_ratio[index] = float(node["occlusion_ratio"])
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
            node_lookup[int(node["node_id"])] = index

        relation_type = torch.zeros(N, N, device=device, dtype=torch.long)
        relation_valid = node_valid.unsqueeze(1) & node_valid.unsqueeze(0)
        relation_valid = relation_valid & ~torch.eye(N, device=device, dtype=torch.bool)
        external_relation = torch.zeros(N, relationClasses, device=device, dtype=dtype)
        external_relation_valid = node_valid.clone()
        for relation in annotation["relations"]:
            subject = int(relation["subject_node_id"])
            obj = int(relation["object_node_id"])
            if subject in node_lookup and obj in node_lookup:
                relation_type[node_lookup[subject], node_lookup[obj]] = int(relation["relation_type"])
            elif subject in node_lookup:
                external_relation[node_lookup[subject], int(relation["relation_type"])] = 1.0

        motion = torch.zeros(N, 7, device=device, dtype=dtype)
        motion[:, 6] = 1.0
        motion_valid = node_valid.clone()
        is_moving = torch.zeros(N, device=device, dtype=dtype)
        for entry in annotation["motion"]["object_motions_from_prev"]:
            index = node_lookup[int(entry["node_id"])]
            motion[index] = torch.tensor(entry["motion"], device=device, dtype=dtype)
            is_moving[index] = float(entry["is_moving"])

        affordance = torch.zeros(N, affordanceDim, device=device, dtype=dtype)
        affordance_valid = torch.zeros(N, device=device, dtype=torch.bool)
        affordance_keys = (
            "graspable", "pushable", "pressable", "pullable",
            "rotatable", "openable", "container", "support_surface")
        for entry in annotation["interaction"]["affordance_targets"]:
            index = node_lookup[int(entry["node_id"])]
            affordance[index] = torch.tensor([entry[name] for name in affordance_keys], device=device, dtype=dtype)
            affordance_valid[index] = True

        contact = torch.zeros(N, device=device, dtype=dtype)
        contact_valid = node_valid.clone()
        contact_force = torch.zeros(N, 2, device=device, dtype=dtype)
        contact_point_camera = torch.zeros(N, 3, device=device, dtype=dtype)
        for event in annotation["interaction"]["contact_events"]:
            index = node_lookup[int(event["actor_b_node_id"])]
            contact[index] = 1.0
            contact_force[index] = torch.tensor(
                [event["normal_force_n"], event["tangential_force_n"]],
                device=device,
                dtype=dtype)
            contact_point_camera[index] = torch.tensor(event["contact_point_camera"], device=device, dtype=dtype)

        return {
            "rgb": rgb,
            "depth": depth,
            "depth_valid": depthValid,
            "normal": normal,
            "semantic_segmentation": semanticSegmentation.long(),
            "instance_segmentation": instanceSegmentation.long(),
            "scene_class": torch.tensor(annotation["scene"]["scene_class"], device=device, dtype=torch.long),
            "global_labels": torch.tensor(annotation["scene"]["global_labels"][:numGlobalLabels], device=device, dtype=dtype),
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
            # Per-frame absolute camera pose (camera->world, xyz + xyzw quaternion). The
            # inter-frame camera_motion is derived from consecutive poses in BrainCore.Step,
            # so the dataset only stores each frame's own pose, never a relative transform.
            "camera_pose_world": torch.tensor(annotation["camera"]["pose_world"], device=device, dtype=dtype)}

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
    def ConvertRobotInputs(
        imgs: Optional[Union[np.ndarray, torch.Tensor]],
        reward: Optional[Union[np.ndarray, torch.Tensor]],
        done: Optional[Union[np.ndarray, torch.Tensor]],
        *,
        size: Optional[Tuple[int, int]] = (BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE),
        device: Optional[torch.device] = None,
        needVisualState: bool = True,
        depths: Optional[Union[np.ndarray, torch.Tensor]] = None,
        depthValids: Optional[Union[np.ndarray, torch.Tensor]] = None,
        depthScaleMeters: float = 1.0,
        cameraIntrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,) -> Dict[str, Any]:

        original_images: List[np.ndarray] = []
        resize_meta: List[DataResizeMeta] = []
        # Source image size, captured regardless of needVisualState, so the camera
        # intrinsics can be rescaled to the same target grid as RGB/depth.
        image_source_size: Optional[Tuple[int, int]] = None
        image_target_size: Optional[Tuple[int, int]] = None

        if imgs is not None:
            img_tensor = torch.as_tensor(imgs)
            if img_tensor.ndim == 3:
                image_samples = [img_tensor]
            elif img_tensor.ndim == 4:
                image_samples = [img_tensor[i] for i in range(int(img_tensor.shape[0]))]
            else:
                raise ValueError(f"Unexpected image batch shape: {tuple(img_tensor.shape)}")

            resized_samples: List[torch.Tensor] = []
            for sample in image_samples:
                if needVisualState:
                    if isinstance(sample, torch.Tensor):
                        original_image = sample.detach().cpu().numpy()
                    else:
                        original_image = np.asarray(sample)
                    original_images.append(np.array(original_image, copy=True))

                sample_tensor = DataPreprocessor.ToImageTensor(sample)

                if size is None:
                    resized_tensor = sample_tensor.clone()
                    _, src_h, src_w = sample_tensor.shape
                    meta = DataResizeMeta(
                        src_h=int(src_h),
                        src_w=int(src_w),
                        dst_h=int(src_h),
                        dst_w=int(src_w),
                        scale_x=1.0,
                        scale_y=1.0,)
                else:
                    resized_tensor, meta = DataPreprocessor.ResizeImage(
                        sample_tensor,
                        size=size,
                        antialias=True,)

                if image_source_size is None:
                    image_source_size = (int(meta.src_h), int(meta.src_w))
                    image_target_size = (int(meta.dst_h), int(meta.dst_w))

                resized_samples.append(resized_tensor)
                if needVisualState:
                    resize_meta.append(meta)

            img_tensor = torch.stack(resized_samples, dim=0) if len(resized_samples) > 0 else torch.empty(0)
            if device is not None:
                img_tensor = img_tensor.to(device)
        else:
            img_tensor = None

        depth_tensor = None
        depth_valid_tensor = None
        if depths is not None:
            depth_batch = torch.as_tensor(depths)
            expected_batch = None if img_tensor is None else int(img_tensor.size(0))
            if depth_batch.ndim == 2:
                depth_samples = [depth_batch]
            elif depth_batch.ndim == 3:
                if expected_batch is not None and expected_batch > 1 and int(depth_batch.size(0)) == expected_batch:
                    depth_samples = [depth_batch[i] for i in range(expected_batch)]
                else:
                    depth_samples = [depth_batch]
            elif depth_batch.ndim == 4:
                depth_samples = [depth_batch[i] for i in range(int(depth_batch.size(0)))]
            else:
                raise ValueError(f"Unexpected depth batch shape: {tuple(depth_batch.shape)}")
            if expected_batch is not None and len(depth_samples) != expected_batch:
                raise ValueError(f"RGB/depth batch mismatch: {expected_batch} vs {len(depth_samples)}")

            valid_samples: List[Optional[torch.Tensor]] = [None] * len(depth_samples)
            if depthValids is not None:
                valid_batch = torch.as_tensor(depthValids)
                if valid_batch.ndim == 2:
                    valid_samples = [valid_batch]
                elif valid_batch.ndim == 3 and len(depth_samples) > 1 and int(valid_batch.size(0)) == len(depth_samples):
                    valid_samples = [valid_batch[i] for i in range(len(depth_samples))]
                elif valid_batch.ndim == 3:
                    valid_samples = [valid_batch]
                elif valid_batch.ndim == 4:
                    valid_samples = [valid_batch[i] for i in range(int(valid_batch.size(0)))]
                else:
                    raise ValueError(f"Unexpected depth valid batch shape: {tuple(valid_batch.shape)}")
                if len(valid_samples) != len(depth_samples):
                    raise ValueError("depth/depth valid batch mismatch")

            resized_depth: List[torch.Tensor] = []
            resized_valid: List[torch.Tensor] = []
            for sample, sample_valid in zip(depth_samples, valid_samples):
                one_depth, one_valid = DataPreprocessor.ToDepthTensor(
                    sample,
                    sample_valid,
                    depthScaleMeters=depthScaleMeters)
                if size is not None:
                    one_depth, one_valid = DataPreprocessor.ResizeDepth(one_depth, one_valid, size)
                resized_depth.append(one_depth)
                resized_valid.append(one_valid)
            depth_tensor = torch.stack(resized_depth, dim=0)
            depth_valid_tensor = torch.stack(resized_valid, dim=0)
            if device is not None:
                depth_tensor = depth_tensor.to(device)
                depth_valid_tensor = depth_valid_tensor.to(device)

        # Camera intrinsics: rescale to the same target grid as RGB/depth when supplied.
        # If not passed, leave as None; if passed but no resize happened, return as-is.
        camera_intrinsics_tensor = None
        if cameraIntrinsics is not None:
            if image_source_size is not None and image_target_size is not None and size is not None:
                camera_intrinsics_tensor = DataPreprocessor.ResizeCameraIntrinsics(
                    cameraIntrinsics,
                    sourceSize=image_source_size,
                    targetSize=image_target_size,
                    device=device)
            else:
                camera_intrinsics_tensor = torch.as_tensor(cameraIntrinsics).float()
                if device is not None:
                    camera_intrinsics_tensor = camera_intrinsics_tensor.to(device)

        def convert_tensor(value: Optional[Union[np.ndarray, torch.Tensor]]):
            if value is None:
                return None
            value_tensor = torch.as_tensor(value).float()
            if device is not None:
                value_tensor = value_tensor.to(device)
            return value_tensor

        return {
            "frames": img_tensor,
            "original_images": original_images,
            "resize_meta": resize_meta,
            "depths": depth_tensor,
            "depth_valid": depth_valid_tensor,
            "camera_intrinsics": camera_intrinsics_tensor,
            "rewards": convert_tensor(reward),
            "dones": convert_tensor(done),}

    @staticmethod
    def ConvertCppCameraFrame(
        bitmap: Union[List[Any], np.ndarray, torch.Tensor],
        reward: Optional[float],
        done: Optional[float],
        *,
        depthBitmap: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
        depthValid: Optional[Union[List[Any], np.ndarray, torch.Tensor]] = None,
        depthScaleMeters: float = 1.0,
        cameraIntrinsics: Optional[Union[np.ndarray, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        needVisualState: bool = False,) -> Dict[str, Any]:
        if isinstance(bitmap, torch.Tensor):
            bitmap_value: Union[np.ndarray, torch.Tensor] = bitmap
        else:
            bitmap_value = np.asarray(bitmap)

        reward_value = None if reward is None else np.asarray([float(reward)], dtype=np.float32)
        done_value = None if done is None else np.asarray([float(done)], dtype=np.float32)

        return DataPreprocessor.ConvertRobotInputs(
            imgs=bitmap_value,
            reward=reward_value,
            done=done_value,
            size=(BasicParameters.IMAGE_SIZE, BasicParameters.IMAGE_SIZE),
            device=device,
            needVisualState=needVisualState,
            depths=depthBitmap,
            depthValids=depthValid,
            depthScaleMeters=depthScaleMeters,
            cameraIntrinsics=cameraIntrinsics,)

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
