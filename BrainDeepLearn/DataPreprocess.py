from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from Config import BasicParameters
from CoreTypes import ContractOfflineSample
from ModuleMessagerManager import ModuleDim
from RobotMorphologyModule import RobotEmbodimentContractView


class Realm(IntEnum):
    SELF_BODY = 0
    EXTERNAL_PHYSICAL = 1
    VIRTUAL_CONTENT = 2
    VISUAL_EFFECT = 3
    UNKNOWN = 4


class Agency(IntEnum):
    SELF_CAUSED = 0
    EXTERNAL_CAUSED = 1
    AUTONOMOUS = 2
    MIXED = 3
    UNKNOWN = 4


class MotionLayer(IntEnum):
    OBSERVER_MOTION = 0
    CARRIER_MOTION = 1
    ARTICULATION_MOTION = 2
    SURFACE_CONTENT_MOTION = 3
    PHOTOMETRIC_CHANGE = 4


class OntologyRelation(IntEnum):
    DISPLAYED_ON = 0
    HELD_BY = 1
    MOVING_WITH = 2
    ATTACHED_TO_SELF = 3
    CONTACTING_SELF = 4
    REFLECTED_IN = 5
    SHADOW_OF = 6
    OCCLUDES = 7
    INSIDE_DISPLAY_REGION = 8


REALM_NAMES: Tuple[str, ...] = tuple(value.name.lower() for value in Realm)
AGENCY_NAMES: Tuple[str, ...] = tuple(value.name.lower() for value in Agency)
MOTION_LAYER_NAMES: Tuple[str, ...] = tuple(
    value.name.lower() for value in MotionLayer)
ONTOLOGY_RELATION_NAMES: Tuple[str, ...] = tuple(
    value.name.lower() for value in OntologyRelation)

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


def ResolveContractFeedbackPath(isTest: bool) -> str:
    configured_name = (
        "DATA_FEEDBACK_PATH_TEST" if isTest else "DATA_FEEDBACK_PATH")
    value = getattr(BasicParameters, configured_name, None)
    if type(value) is not str or not value:
        raise ValueError(f"{configured_name} must be configured")
    return value


def ValidateContractOfflineSensorManifest(
    manifest: Any,
    calibrationId: str,
    sensorFrameName: str,
) -> None:
    if type(manifest) is not dict:
        raise TypeError("offline sensor manifest must be a mapping")
    expected = {
        "calibration_id": calibrationId,
        "rgb_encoding": "rgb8",
        "depth_unit": "meter",
        "depth_representation": "optical_axis_z",
        "rgb_depth_alignment": "registered_to_rgb",
        "rectification": "rectified",
        "synchronization": "synchronized_exposure",
        "object_motion_frame": sensorFrameName,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(
                f"offline sensor manifest {name} does not match the sensory contract")


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
    def __init__(
        self,
        *,
        calibrationId: str,
        sensorFrameName: str,
        contractView: Optional[RobotEmbodimentContractView] = None,
        isTest: bool = False,
    ) -> None:
        if type(calibrationId) is not str or not calibrationId:
            raise ValueError("calibrationId must be a non-empty string")
        if type(sensorFrameName) is not str or not sensorFrameName:
            raise ValueError("sensorFrameName must be a non-empty string")
        self.calibration_id = calibrationId
        self.sensor_frame_name = sensorFrameName
        self.world_frame_id = f"offline:{sensorFrameName}"
        if contractView is not None:
            if type(contractView) is not RobotEmbodimentContractView:
                raise TypeError("contractView must be an immutable contract view")
        self.contract_view = contractView
        if isTest:
            root = Path(BasicParameters.DATA_ROOT_PATH_TEST)
            sensor_manifest_path = Path(
                BasicParameters.DATA_SENSOR_MANIFEST_PATH_TEST)
            self.imgs = sorted((root / "frames").glob("*.png"))
            self.reward = sorted((root / "reward").glob("*.npy"))
            self.done = sorted((root / "done").glob("*.npy"))
            self.depths = ListDepthFiles(getattr(
                BasicParameters,
                "DATA_DEPTH_PATH_TEST",
                root / "depth"))
            self.depth_valids = ListDepthFiles(getattr(
                BasicParameters,
                "DATA_DEPTH_VALID_PATH_TEST",
                root / "depth_valid"))
            self.texts = sorted((root / "texts").glob("*.txt"))
            self.feedbacks = ListJsonFiles(ResolveContractFeedbackPath(True))
            self.normals = ListArrayFiles(getattr(
                BasicParameters,
                "DATA_NORMAL_PATH_TEST",
                root / "normal"))
            self.semantic_segmentations = ListArrayFiles(getattr(
                BasicParameters,
                "DATA_SEMANTIC_SEGMENTATION_PATH_TEST",
                root / "semantic_segmentation"))
            self.instance_segmentations = ListArrayFiles(getattr(
                BasicParameters,
                "DATA_INSTANCE_SEGMENTATION_PATH_TEST",
                root / "instance_segmentation"))
            self.synthetic_annotations = ListJsonFiles(getattr(
                BasicParameters,
                "DATA_SYNTHETIC_SUPERVISION_PATH_TEST",
                root / "synthetic_supervision"))
        else:
            sensor_manifest_path = Path(
                BasicParameters.DATA_SENSOR_MANIFEST_PATH)
            self.imgs = sorted(Path(
                BasicParameters.DATA_FRAMES_PATH).glob("*.png"))
            self.reward = sorted(Path(
                BasicParameters.DATA_REWARD_PATH).glob("*.npy"))
            self.done = sorted(Path(
                BasicParameters.DATA_DONE_PATH).glob("*.npy"))
            self.depths = ListDepthFiles(BasicParameters.DATA_DEPTH_PATH)
            self.depth_valids = ListDepthFiles(
                BasicParameters.DATA_DEPTH_VALID_PATH)
            self.texts = sorted(Path(
                BasicParameters.DATA_TEXTS_PATH).glob("*.txt"))
            self.feedbacks = ListJsonFiles(ResolveContractFeedbackPath(False))
            self.normals = ListArrayFiles(BasicParameters.DATA_NORMAL_PATH)
            self.semantic_segmentations = ListArrayFiles(
                BasicParameters.DATA_SEMANTIC_SEGMENTATION_PATH)
            self.instance_segmentations = ListArrayFiles(
                BasicParameters.DATA_INSTANCE_SEGMENTATION_PATH)
            self.synthetic_annotations = ListJsonFiles(
                BasicParameters.DATA_SYNTHETIC_SUPERVISION_PATH)

        sensor_manifest = json.loads(
            sensor_manifest_path.read_text(encoding="utf-8"))
        ValidateContractOfflineSensorManifest(
            sensor_manifest,
            self.calibration_id,
            self.sensor_frame_name)
        if not (
            len(self.imgs)
            == len(self.reward)
            == len(self.done)
            == len(self.depths)
            == len(self.depth_valids)
            == len(self.feedbacks)
        ):
            raise ValueError(
                "frames/reward/done/depth/feedback counts must match")
        frame_ids = [path.stem for path in self.imgs]
        required_streams = {
            "reward": self.reward,
            "done": self.done,
            "depth": self.depths,
            "depth_valid": self.depth_valids,
            "feedback": self.feedbacks,
        }
        for name, paths in required_streams.items():
            if [path.stem for path in paths] != frame_ids:
                raise ValueError(
                    f"{name} filenames must match frame identifiers exactly")
        if self.texts:
            if len(self.texts) != len(self.imgs):
                raise ValueError("text count must match frames")
            if [path.stem for path in self.texts] != frame_ids:
                raise ValueError(
                    "text filenames must match frame identifiers exactly")
        supervision_streams = (
            self.synthetic_annotations,
            self.normals,
            self.semantic_segmentations,
            self.instance_segmentations,
        )
        if any(bool(paths) for paths in supervision_streams):
            if self.contract_view is None:
                raise ValueError(
                    "synthetic supervision requires an embodiment contract view")
            if any(len(paths) != len(self.imgs) for paths in supervision_streams):
                raise ValueError(
                    "synthetic supervision stream counts must match frames")
            for name, paths in {
                "synthetic_supervision": self.synthetic_annotations,
                "normal": self.normals,
                "semantic_segmentation": self.semantic_segmentations,
                "instance_segmentation": self.instance_segmentations,
            }.items():
                if [path.stem for path in paths] != frame_ids:
                    raise ValueError(
                        f"{name} filenames must match frame identifiers exactly")
        self.feedback_payloads = tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in self.feedbacks)
        if len(self.feedback_payloads) < 1:
            raise ValueError("offline sensory dataset must not be empty")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> ContractOfflineSample:
        imgs = LoadImageFirstFrame(self.imgs[idx])
        reward_values = np.asarray(
            np.load(self.reward[idx]), dtype=np.float32).reshape(-1)
        done_values = np.asarray(
            np.load(self.done[idx]), dtype=np.float32).reshape(-1)
        if reward_values.size != 1 or done_values.size != 1:
            raise ValueError("offline reward and done must contain one scalar")
        reward = np.float32(reward_values[0])
        done = np.float32(done_values[0])
        depth = LoadDepthArray(self.depths[idx])
        depth_valid = LoadDepthArray(self.depth_valids[idx])
        image_size = BasicParameters.IMAGE_SIZE
        if imgs.dtype != np.uint8 or imgs.shape != (
            image_size,
            image_size,
            3,
        ):
            raise ValueError(
                f"offline RGB must be uint8 [{image_size}, {image_size}, 3]")
        if depth.dtype != np.float32 or depth.shape != (
            image_size,
            image_size,
        ):
            raise ValueError(
                f"offline depth must be float32 metres [{image_size}, {image_size}]")
        if depth_valid.dtype != np.bool_ or depth_valid.shape != depth.shape:
            raise ValueError(
                "offline depth_valid must be a bool mask matching depth")
        if not np.isfinite(depth).all():
            raise ValueError(
                "offline depth must contain only finite metre values")
        if np.any(depth_valid & (depth <= 0.0)):
            raise ValueError(
                "offline valid depth pixels must be positive")
        ext_text = None
        if self.texts:
            text = self.texts[idx].read_text(encoding="utf-8").strip()
            ext_text = text if text else None
        perception_targets: Dict[str, torch.Tensor] = {}
        if self.synthetic_annotations:
            annotation = json.loads(
                self.synthetic_annotations[idx].read_text(encoding="utf-8"))
            rgb_tensor = DataPreprocessor.ToImageTensor(imgs)
            depth_tensor, depth_valid_tensor = DataPreprocessor.ToDepthTensor(
                depth,
                depth_valid)
            normal_tensor = DataPreprocessor.ToNormalTensor(
                LoadDepthArray(self.normals[idx]))
            semantic_segmentation = DataPreprocessor.ToSegmentationTensor(
                LoadDepthArray(self.semantic_segmentations[idx]))
            instance_segmentation = DataPreprocessor.ToSegmentationTensor(
                LoadDepthArray(self.instance_segmentations[idx]))
            if self.contract_view is None:
                raise RuntimeError("synthetic supervision contract is unavailable")
            perception_targets = DataPreprocessor.TensorizeSyntheticSupervision(
                annotation,
                rgb_tensor,
                depth_tensor,
                depth_valid_tensor,
                normal_tensor,
                semantic_segmentation,
                instance_segmentation,
                contractView=self.contract_view)
        return ContractOfflineSample(
            image=imgs,
            reward=reward,
            done=done,
            depth=depth,
            depth_valid=depth_valid,
            text_ext=ext_text,
            feedback_payload=self.feedback_payloads[idx],
            perception_targets=perception_targets)


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
    def ExpectedSyntheticOntologyVocabularyContract(
        contractView: RobotEmbodimentContractView,
        maxNodes: int = ModuleDim.PstObservedSlots,
    ) -> Dict[str, Any]:
        if type(contractView) is not RobotEmbodimentContractView:
            raise TypeError("synthetic supervision requires a contract view")
        if type(maxNodes) is not int or maxNodes < 1:
            raise ValueError("synthetic observed capacity must be positive")
        return {
            "realm_names": list(REALM_NAMES),
            "agency_names": list(AGENCY_NAMES),
            "motion_layer_names": list(MOTION_LAYER_NAMES),
            "ontology_relation_names": list(ONTOLOGY_RELATION_NAMES),
            "self_part_slot_count": contractView.end_effector_count,
            "self_part_parent_indices": list(contractView.parent_index),
            "observed_slot_capacity": int(maxNodes),
            "virtual_slot_capacity": ModuleDim.PstVirtualSlots,
        }

    @staticmethod
    def ValidateSyntheticValueType(value: Any, kind: str, field: str) -> None:
        if isinstance(value, list):
            for child in value:
                DataPreprocessor.ValidateSyntheticValueType(
                    child,
                    kind,
                    field)
            return
        if kind == "bool":
            valid = type(value) is bool
        elif kind == "long":
            valid = type(value) is int
        elif kind == "float":
            valid = (
                type(value) in (int, float)
                and math.isfinite(float(value)))
        else:
            raise ValueError("synthetic target kind is unsupported")
        if not valid:
            raise TypeError(f"synthetic target {field} has invalid scalar type")

    @staticmethod
    def TensorFromSyntheticField(
        payload: Dict[str, Any],
        field: str,
        kind: str,
        shape: Tuple[int, ...],
        device: torch.device,
        floatDtype: torch.dtype,
    ) -> torch.Tensor:
        if field not in payload:
            raise ValueError(f"synthetic target {field} is missing")
        value = payload[field]
        DataPreprocessor.ValidateSyntheticValueType(value, kind, field)
        dtype = {
            "bool": torch.bool,
            "long": torch.long,
            "float": floatDtype,
        }[kind]
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"synthetic target {field} must have shape {shape}")
        if tensor.is_floating_point() and not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise ValueError(f"synthetic target {field} must be finite")
        return tensor

    @staticmethod
    def PadSyntheticTarget(
        tensor: torch.Tensor,
        capacity: int,
        default: float = 0.0,
        pair: bool = False,
    ) -> torch.Tensor:
        nodeCount = int(tensor.size(0))
        if nodeCount > capacity:
            raise ValueError("synthetic target exceeds observed slot capacity")
        if pair:
            if tensor.dim() < 2 or int(tensor.size(1)) != nodeCount:
                raise ValueError("synthetic pair target must be square")
            shape = (capacity, capacity) + tuple(tensor.shape[2:])
            result = torch.full(
                shape,
                default,
                device=tensor.device,
                dtype=tensor.dtype)
            result[:nodeCount, :nodeCount] = tensor
            return result
        shape = (capacity,) + tuple(tensor.shape[1:])
        result = torch.full(
            shape,
            default,
            device=tensor.device,
            dtype=tensor.dtype)
        result[:nodeCount] = tensor
        return result

    @staticmethod
    def ValidateSyntheticTargetSemantics(
        targets: Dict[str, torch.Tensor],
        contractView: RobotEmbodimentContractView,
        nodeCount: int,
    ) -> None:
        active = targets["node_valid"][:nodeCount]
        activeIndex = torch.nonzero(active, as_tuple=False).flatten()
        if activeIndex.numel() == 0:
            raise ValueError("synthetic supervision must contain an active entity")
        nodeIds = targets["node_id"][:nodeCount][active]
        if bool((nodeIds < 0).any().item()) or int(
            torch.unique(nodeIds).numel()
        ) != int(nodeIds.numel()):
            raise ValueError("active synthetic entity identifiers must be unique")
        levels = targets["node_level"][:nodeCount]
        if bool(((levels[active] < 0) | (levels[active] > 2)).any().item()):
            raise ValueError("synthetic hierarchy level is outside the vocabulary")
        parents = targets["parent_index"][:nodeCount]
        for index in activeIndex.tolist():
            parent = int(parents[index].item())
            if parent >= index or parent < -1:
                raise ValueError("synthetic hierarchy must be topologically ordered")
            if parent >= 0 and not bool(active[parent].item()):
                raise ValueError("synthetic hierarchy parent must be active")
            if int(levels[index].item()) == 0 and parent != -1:
                raise ValueError("synthetic root entity cannot have a parent")
            if int(levels[index].item()) > 0 and parent < 0:
                raise ValueError("synthetic part entity requires a parent")
        classChecks = (
            ("object_classes", ModuleDim.PstObjectClasses),
            ("part_classes", ModuleDim.PstPartClasses),
            ("symbol_type", ModuleDim.PstSymbolClasses),
            ("realm", ModuleDim.PstRealmClasses),
        )
        for field, count in classChecks:
            value = targets[field][:nodeCount][active]
            if bool(((value < 0) | (value >= count)).any().item()):
                raise ValueError(f"synthetic target {field} is outside its vocabulary")
        sceneClass = int(targets["scene_class"].item())
        if not 0 <= sceneClass < ModuleDim.PstSceneClasses:
            raise ValueError("synthetic scene class is outside its vocabulary")
        globalLabels = targets["global_labels"]
        if bool(((globalLabels < 0.0) | (globalLabels > 1.0)).any().item()):
            raise ValueError("synthetic global labels must be in [0, 1]")
        probabilityFields = (
            "visible_ratio",
            "occlusion_ratio",
            "physical_entity",
            "physical_interaction",
            "body_membership",
            "content_change",
            "display_surface",
            "verification_confidence",
            "is_moving",
            "contact",
        )
        for field in probabilityFields:
            value = targets[field][:nodeCount][active]
            if bool(((value < 0.0) | (value > 1.0)).any().item()):
                raise ValueError(f"synthetic target {field} must be in [0, 1]")
        multihotFields = (
            "motion_layer_multi_hot",
            "ontology_relation_multi_hot",
            "affordance",
            "external_relation",
        )
        for field in multihotFields:
            value = targets[field]
            if bool(((value < 0.0) | (value > 1.0)).any().item()):
                raise ValueError(f"synthetic target {field} must be in [0, 1]")
        validPoseFields = (
            ("spatial_frame", "pose_valid"),
            ("carrier_motion", "carrier_motion_valid"),
            ("articulation_motion", "articulation_motion_valid"),
            ("motion", "motion_valid"),
        )
        for field, validityField in validPoseFields:
            validity = targets[validityField][:nodeCount] & active
            quaternion = targets[field][:nodeCount, 3:7][validity]
            if quaternion.numel() > 0 and not torch.allclose(
                quaternion.norm(dim=-1),
                torch.ones_like(quaternion[:, 0]),
                rtol=1e-3,
                atol=1e-3,
            ):
                raise ValueError(f"synthetic target {field} has invalid quaternion")
        poseValid = targets["pose_valid"][:nodeCount] & active
        observerQuaternion = targets[
            "orientation_observer"][:nodeCount][poseValid]
        if observerQuaternion.numel() > 0 and not torch.allclose(
            observerQuaternion.norm(dim=-1),
            torch.ones_like(observerQuaternion[:, 0]),
            rtol=1e-3,
            atol=1e-3,
        ):
            raise ValueError("synthetic observer orientation has invalid quaternion")
        if not torch.allclose(
            targets["spatial_frame"][:nodeCount, :3][poseValid],
            targets["position_observer"][:nodeCount][poseValid],
            rtol=1e-4,
            atol=1e-4,
        ) or not torch.allclose(
            targets["spatial_frame"][:nodeCount, 3:7][poseValid],
            targets["orientation_observer"][:nodeCount][poseValid],
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError("synthetic observer pose representations conflict")
        geometryValid = targets["geometry_valid"][:nodeCount] & active
        if bool((targets["size_3d"][:nodeCount][geometryValid] <= 0.0).any().item()):
            raise ValueError("synthetic valid geometry size must be positive")
        bbox = targets["bbox_2d"][:nodeCount][active]
        if bool(((bbox < 0.0) | (bbox > 1.0)).any().item()) or bool(
            (bbox[:, :2] >= bbox[:, 2:4]).any().item()
        ):
            raise ValueError("synthetic bounding boxes must be normalized and ordered")
        layerValid = targets["motion_layer_valid"][:nodeCount]
        layerActive = targets[
            "motion_layer_multi_hot"][:nodeCount] > 0.5
        agencyValid = targets["agency_by_layer_valid"][:nodeCount]
        if bool((layerActive[active] != agencyValid[active]).any().item()) or bool(
            (agencyValid[active] & ~layerValid[active].unsqueeze(-1)).any().item()
        ):
            raise ValueError("synthetic motion layer and agency validity conflict")
        agency = targets["agency_by_layer"][:nodeCount][agencyValid]
        if bool(((agency < 0) | (agency >= ModuleDim.PstAgencyClasses)).any().item()):
            raise ValueError("synthetic agency is outside its vocabulary")
        bodyValid = targets["body_membership_valid"][:nodeCount] & active
        body = targets["body_membership"][:nodeCount] > 0.5
        selfPartValid = targets["self_part_valid"][:nodeCount] & active
        if bool((bodyValid & (body != selfPartValid)).any().item()):
            raise ValueError("synthetic self binding and body membership conflict")
        selfPart = targets["self_part_id"][:nodeCount][selfPartValid]
        if bool(((selfPart < 0) | (selfPart >= contractView.end_effector_count)).any().item()):
            raise ValueError("synthetic self binding is outside the contract")
        realmValid = targets["realm_valid"][:nodeCount] & active
        realm = targets["realm"][:nodeCount]
        selfRealm = realm.eq(int(Realm.SELF_BODY))
        if bool((bodyValid & body & (~realmValid | ~selfRealm)).any().item()):
            raise ValueError("synthetic body membership requires the self realm")
        virtual = realm.eq(int(Realm.VIRTUAL_CONTENT)) | realm.eq(
            int(Realm.VISUAL_EFFECT))
        if bool((realmValid & virtual & (poseValid | geometryValid)).any().item()):
            raise ValueError("synthetic virtual entities cannot own physical geometry")
        if int((realmValid & virtual).sum().item()) > ModuleDim.PstVirtualSlots:
            raise ValueError("synthetic virtual entity capacity is exceeded")
        selfPartParent = targets["surface_parent_index"][:nodeCount]
        selfPartParentValid = targets["surface_parent_valid"][:nodeCount] & active
        for index in torch.nonzero(
            selfPartParentValid,
            as_tuple=False,
        ).flatten().tolist():
            parent = int(selfPartParent[index].item())
            if (
                parent < 0
                or parent >= nodeCount
                or parent == index
                or not bool(active[parent].item())
                or not bool(targets["display_surface_valid"][parent].item())
                or float(targets["display_surface"][parent].item()) <= 0.5
            ):
                raise ValueError("synthetic surface parent is invalid")
        pairActive = active.unsqueeze(1) & active.unsqueeze(0)
        diagonal = torch.eye(nodeCount, device=active.device, dtype=torch.bool)
        for field in ("relation_valid", "ontology_relation_valid"):
            valid = targets[field][:nodeCount, :nodeCount]
            if bool((valid & (~pairActive | diagonal)).any().item()):
                raise ValueError(f"synthetic target {field} contains an invalid pair")
        movingWithIndex = ONTOLOGY_RELATION_NAMES.index("moving_with")
        movingWith = targets[
            "ontology_relation_multi_hot"][:nodeCount, :nodeCount, movingWithIndex] > 0.5
        movingWithValid = targets[
            "ontology_relation_valid"][:nodeCount, :nodeCount]
        if bool((movingWith & (~movingWith.t() | ~movingWithValid.t())).any().item()):
            raise ValueError("synthetic moving-with relation must be symmetric")
        contactActive = targets["contact_valid"][:nodeCount] & active & (
            targets["contact"][:nodeCount] > 0.5)
        if bool((targets["contact_force"][:nodeCount][contactActive] < 0.0).any().item()):
            raise ValueError("synthetic contact force must be nonnegative")

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
        contractView: RobotEmbodimentContractView,
        maxNodes: int = ModuleDim.PstObservedSlots,
    ) -> Dict[str, torch.Tensor]:
        if type(annotation) is not dict or set(annotation) != {
            "contract_binding",
            "targets",
        }:
            raise ValueError("synthetic supervision fields do not match schema")
        expectedBinding = DataPreprocessor.ExpectedSyntheticOntologyVocabularyContract(
            contractView,
            maxNodes=maxNodes)
        if annotation["contract_binding"] != expectedBinding:
            raise ValueError("synthetic supervision contract binding does not match")
        payload = annotation["targets"]
        if type(payload) is not dict:
            raise TypeError("synthetic targets must be a mapping")
        if (
            rgb.dim() != 3
            or int(rgb.size(0)) != 3
            or depth.dim() != 3
            or tuple(depth.shape) != (1, int(rgb.size(1)), int(rgb.size(2)))
            or depthValid.shape != depth.shape
            or normal.shape != rgb.shape
            or semanticSegmentation.shape != rgb.shape[-2:]
            or instanceSegmentation.shape != rgb.shape[-2:]
        ):
            raise ValueError("synthetic dense supervision shapes must match RGB-D")
        if (
            not rgb.is_floating_point()
            or not depth.is_floating_point()
            or not normal.is_floating_point()
            or depthValid.dtype != torch.bool
            or semanticSegmentation.dtype != torch.long
            or instanceSegmentation.dtype != torch.long
        ):
            raise TypeError("synthetic dense supervision dtypes are invalid")
        if not bool(torch.isfinite(normal).all().item()):
            raise ValueError("synthetic normal supervision must be finite")
        if bool(((semanticSegmentation < 0) | (
            semanticSegmentation >= ModuleDim.PstObjectClasses)).any().item()):
            raise ValueError("synthetic semantic segmentation is outside its vocabulary")
        if bool((instanceSegmentation < 0).any().item()):
            raise ValueError("synthetic instance segmentation must be nonnegative")
        rawNodeValid = payload.get("node_valid")
        if type(rawNodeValid) is not list or not 1 <= len(rawNodeValid) <= maxNodes:
            raise ValueError("synthetic node validity has invalid capacity")
        nodeCount = len(rawNodeValid)
        device = rgb.device
        dtype = rgb.dtype
        scalarSchema = {
            "scene_class": ("long", ()),
            "global_labels": ("float", (ModuleDim.PstGlobalLabels,)),
            "temporal_kind": ("long", ()),
            "temporal_kind_valid": ("bool", ()),
            "temporal_duration_ms": ("float", ()),
            "temporal_duration_valid": ("bool", ()),
        }
        nodeSchema = {
            "node_valid": ("bool", ()),
            "node_id": ("long", ()),
            "node_level": ("long", ()),
            "parent_index": ("long", ()),
            "object_classes": ("long", ()),
            "part_classes": ("long", ()),
            "track_id": ("long", ()),
            "instance_id": ("long", ()),
            "position_observer": ("float", (3,)),
            "orientation_observer": ("float", (4,)),
            "spatial_frame": ("float", (7,)),
            "pose_valid": ("bool", ()),
            "geometry_valid": ("bool", ()),
            "size_3d": ("float", (3,)),
            "bbox_2d": ("float", (4,)),
            "visible_ratio": ("float", ()),
            "occlusion_ratio": ("float", ()),
            "has_text": ("long", ()),
            "text_embed": ("float", (ModuleDim.PstTextDim,)),
            "symbol_type": ("long", ()),
            "physical_entity": ("float", ()),
            "physical_entity_valid": ("bool", ()),
            "physical_interaction": ("float", ()),
            "physical_interaction_valid": ("bool", ()),
            "realm": ("long", ()),
            "realm_valid": ("bool", ()),
            "motion_layer_multi_hot": (
                "float",
                (ModuleDim.PstMotionLayerClasses,)),
            "motion_layer_valid": ("bool", ()),
            "agency_by_layer": (
                "long",
                (ModuleDim.PstMotionLayerClasses,)),
            "agency_by_layer_valid": (
                "bool",
                (ModuleDim.PstMotionLayerClasses,)),
            "body_membership": ("float", ()),
            "body_membership_valid": ("bool", ()),
            "self_part_id": ("long", ()),
            "self_part_valid": ("bool", ()),
            "carrier_motion": ("float", (7,)),
            "carrier_motion_valid": ("bool", ()),
            "articulation_motion": ("float", (7,)),
            "articulation_motion_valid": ("bool", ()),
            "content_motion_uv": ("float", (2,)),
            "content_motion_uv_valid": ("bool", ()),
            "content_change": ("float", ()),
            "content_change_valid": ("bool", ()),
            "display_surface": ("float", ()),
            "display_surface_valid": ("bool", ()),
            "surface_parent_index": ("long", ()),
            "surface_parent_valid": ("bool", ()),
            "surface_uv": ("float", (2,)),
            "surface_uv_valid": ("bool", ()),
            "verification_confidence": ("float", ()),
            "verification_confidence_valid": ("bool", ()),
            "node_state": ("float", (ModuleDim.PstStateDim,)),
            "node_state_valid": ("bool", ()),
            "node_attributes": ("float", (ModuleDim.PstAttrDim,)),
            "node_attributes_valid": ("bool", ()),
            "affordance": ("float", (ModuleDim.PstAffordanceDim,)),
            "affordance_valid": ("bool", ()),
            "external_relation": (
                "float",
                (ModuleDim.PstRelationClasses,)),
            "external_relation_valid": ("bool", ()),
            "motion": ("float", (7,)),
            "motion_valid": ("bool", ()),
            "is_moving": ("float", ()),
            "contact": ("float", ()),
            "contact_valid": ("bool", ()),
            "contact_force": ("float", (2,)),
            "contact_point_observer": ("float", (3,)),
        }
        pairSchema = {
            "ontology_relation_multi_hot": (
                "float",
                (ModuleDim.PstOntologyRelationClasses,)),
            "ontology_relation_valid": ("bool", ()),
            "relation_type": ("long", ()),
            "relation_valid": ("bool", ()),
        }
        expectedFields = set(scalarSchema) | set(nodeSchema) | set(pairSchema)
        if set(payload) != expectedFields:
            missing = sorted(expectedFields - set(payload))
            extra = sorted(set(payload) - expectedFields)
            raise ValueError(
                f"synthetic target fields mismatch missing={missing} extra={extra}")
        targets: Dict[str, torch.Tensor] = {}
        for field, (kind, shape) in scalarSchema.items():
            targets[field] = DataPreprocessor.TensorFromSyntheticField(
                payload,
                field,
                kind,
                shape,
                device,
                dtype)
        for field, (kind, tailShape) in nodeSchema.items():
            value = DataPreprocessor.TensorFromSyntheticField(
                payload,
                field,
                kind,
                (nodeCount,) + tailShape,
                device,
                dtype)
            default = -1.0 if field in {
                "node_id",
                "parent_index",
                "surface_parent_index",
            } else 0.0
            targets[field] = DataPreprocessor.PadSyntheticTarget(
                value,
                maxNodes,
                default=default)
        for field, (kind, tailShape) in pairSchema.items():
            value = DataPreprocessor.TensorFromSyntheticField(
                payload,
                field,
                kind,
                (nodeCount, nodeCount) + tailShape,
                device,
                dtype)
            targets[field] = DataPreprocessor.PadSyntheticTarget(
                value,
                maxNodes,
                pair=True)
        for field in (
            "orientation_observer",
            "spatial_frame",
            "carrier_motion",
            "articulation_motion",
            "motion",
        ):
            targets[field][nodeCount:, -1] = 1.0
        instanceIds = targets.pop("instance_id")
        if bool((instanceIds[:nodeCount][targets[
            "node_valid"][:nodeCount]] < 0).any().item()):
            raise ValueError("active synthetic instance identifiers must be nonnegative")
        nodeMasks = torch.zeros(
            maxNodes,
            int(instanceSegmentation.size(0)),
            int(instanceSegmentation.size(1)),
            device=device,
            dtype=torch.bool)
        for index in torch.nonzero(
            targets["node_valid"],
            as_tuple=False,
        ).flatten().tolist():
            nodeMasks[index] = instanceSegmentation.eq(
                int(instanceIds[index].item()))
        targets["node_instance_masks"] = nodeMasks
        targets.update({
            "rgb": rgb,
            "depth": depth,
            "depth_valid": depthValid,
            "normal": normal,
            "normal_valid": normal.norm(dim=0, keepdim=True) > 0.5,
            "semantic_segmentation": semanticSegmentation,
            "instance_segmentation": instanceSegmentation,
        })
        temporalKind = int(targets["temporal_kind"].item())
        temporalKindValid = bool(targets["temporal_kind_valid"].item())
        temporalDuration = float(targets["temporal_duration_ms"].item())
        temporalDurationValid = bool(
            targets["temporal_duration_valid"].item())
        if temporalKindValid and not 0 <= temporalKind < ModuleDim.TemporalPrimitiveCount:
            raise ValueError("synthetic temporal kind is outside its vocabulary")
        if temporalDurationValid != (temporalDuration > 0.0):
            raise ValueError("synthetic temporal duration validity is inconsistent")
        DataPreprocessor.ValidateSyntheticTargetSemantics(
            targets,
            contractView,
            nodeCount)
        return targets

    @staticmethod
    def CollateSyntheticSupervision(
        batch: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        if type(batch) is not list or len(batch) < 1:
            raise ValueError("synthetic supervision batch must not be empty")
        fields = set(batch[0])
        if any(type(sample) is not dict or set(sample) != fields for sample in batch):
            raise ValueError("synthetic supervision batch fields must match")
        return {
            field: torch.stack([sample[field] for sample in batch], dim=0)
            for field in sorted(fields)
        }

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
    def ConvertSensoryInputs(
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
    def ConvertCppPerceptionFrame(
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
