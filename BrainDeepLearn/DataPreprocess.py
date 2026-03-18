from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

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
        if iio is None:
            raise RuntimeError("imageio.v3 cant use")

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
        self.texts = sorted((p / "OCRTexts").glob("*.txt"))
        self.boxes = sorted((p / "boxes").glob("*.npy"))

        assert len(self.imgs) == len(self.texts), "frames/OCRTexts The number of files is inconsistent."
        if len(self.imgs) == 0:
            raise RuntimeError(f"no OCR samples found under {p}")

        self.use_text_annotations = False
        for txt_path in self.texts:
            if DataPreprocessor.TextFileLooksLikeOCRAnnotations(txt_path):
                self.use_text_annotations = True
                break

        if not self.use_text_annotations:
            assert len(self.boxes) == len(self.imgs), "frames/boxes/OCRTexts The number of files is inconsistent."

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        if iio is None:
            raise RuntimeError("imageio.v3 cant use")

        img = iio.imread(self.imgs[idx])
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
    def __init__(self, root: str) -> None:
        p = Path(root)
        self.imgs = sorted((p / "frames").glob("*.png"))
        self.texts = sorted((p / "OCRTexts").glob("*.txt"))

        assert len(self.imgs) == len(self.texts), "frames/OCRTexts The number of files is inconsistent."
        if len(self.imgs) == 0:
            raise RuntimeError(f"no OCR recognition samples found under {p}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        if iio is None:
            raise RuntimeError("imageio.v3 cant use")

        img = iio.imread(self.imgs[idx])
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
    @staticmethod
    def SplitOCRCsvLine(line: str) -> List[str]:
        cleaned = str(line).strip().lstrip("\ufeff").strip()
        cleaned = cleaned.lstrip(",")
        if not cleaned:
            return []
        return next(csv.reader([cleaned], skipinitialspace=True))

    @staticmethod
    def LooksLikeOCRAnnotationLine(line: str) -> bool:
        parts = DataPreprocessor.SplitOCRCsvLine(line)
        if len(parts) < 6:
            return False

        for coord_count in (8, 4):
            if len(parts) < coord_count + 2:
                continue
            try:
                [float(parts[i]) for i in range(coord_count)]
                return parts[coord_count].strip() in ("0", "1")
            except Exception:
                continue
        return False

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
        if len(parts) < 6:
            raise ValueError(f"OCR annotation line has too few fields: {line!r}")

        coord_count = None
        for cand in (8, 4):
            if len(parts) < cand + 2:
                continue
            try:
                [float(parts[i]) for i in range(cand)]
                if parts[cand].strip() not in ("0", "1"):
                    continue
                coord_count = cand
                break
            except Exception:
                continue

        if coord_count is None:
            raise ValueError(f"OCR annotation line has invalid coordinates: {line!r}")

        coords = [float(parts[i]) for i in range(coord_count)]
        idx = coord_count

        ignore_flag = int(parts[idx].strip())
        idx += 1

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

        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(-1)
        if img_t.ndim != 3:
            raise ValueError(f"image must have 2 or 3 dims, but got shape {tuple(img_t.shape)}")

        if img_t.shape[-1] in (1, 3):
            img_t = img_t.permute(2, 0, 1)
        elif img_t.shape[0] not in (1, 3):
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
    def ResizeImage(
        imageTensor: torch.Tensor,
        size: Union[int, Tuple[int, int]],) -> Tuple[torch.Tensor, DataResizeMeta]:
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
            align_corners=False,).squeeze(0)

        meta = DataResizeMeta(
            src_h=int(src_h),
            src_w=int(src_w),
            dst_h=int(dst_h),
            dst_w=int(dst_w),
            scale_x=float(dst_w) / float(max(1, src_w)),
            scale_y=float(dst_h) / float(max(1, src_h)),)
        return resized, meta

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
    def CropAndResizeLineImages(
        imageTensor: torch.Tensor,
        boxes: Union[np.ndarray, torch.Tensor],
        *,
        targetH: int = 32,
        maxW: int = 256,) -> torch.Tensor:
        c, h_img, w_img = imageTensor.shape
        if c < 3:
            raise ValueError(f"imageTensor channel count must be at least 3, but got {c}")

        gray = (
            0.299 * imageTensor[0]
            + 0.587 * imageTensor[1]
            + 0.114 * imageTensor[2])

        line_tensors: List[torch.Tensor] = []
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
            patch_resized = F.interpolate(
                patch,
                size=(targetH, new_w),
                mode="bilinear",
                align_corners=False,)

            if new_w > maxW:
                patch_resized = patch_resized[:, :, :, :maxW]
                new_w = maxW

            pad = torch.zeros(1, 1, targetH, maxW, dtype=imageTensor.dtype)
            pad[:, :, :, :new_w] = patch_resized
            line_tensors.append(pad)

        if not line_tensors:
            return torch.empty(0, 1, targetH, maxW, dtype=imageTensor.dtype)

        return torch.cat(line_tensors, dim=0)

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
        dtype: torch.dtype = torch.float32,
        shrinkRatio: float = 0.15,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gt_shrink = torch.zeros(1, imageHeight, imageWidth, dtype=dtype)
        gt_thresh = torch.zeros(1, imageHeight, imageWidth, dtype=dtype)
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

            gt_thresh[:, y1:y2, x1:x2] = 1.0

            shrink_dx = max(1, int(round((x2 - x1) * shrinkRatio)))
            shrink_dy = max(1, int(round((y2 - y1) * shrinkRatio)))
            sx1 = min(max(0, x1 + shrink_dx), imageWidth - 1)
            sy1 = min(max(0, y1 + shrink_dy), imageHeight - 1)
            sx2 = max(sx1 + 1, min(imageWidth, x2 - shrink_dx))
            sy2 = max(sy1 + 1, min(imageHeight, y2 - shrink_dy))
            gt_shrink[:, sy1:sy2, sx1:sx2] = 1.0

        return gt_shrink, gt_thresh, gt_mask

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
        maxW: int = 256,
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

        gt_shrink, gt_thresh, gt_mask = DataPreprocessor.OCRBuildDbTargets(
            det_boxes,
            imageHeight=imageSize,
            imageWidth=imageSize,
            dtype=img_rgb.dtype,)

        if len(rec_boxes) == 0:
            recog_imgs = torch.empty(0, 1, targetH, maxW, dtype=img_rgb.dtype)
            targets = torch.empty(0, dtype=torch.long)
            target_lengths = torch.empty(0, dtype=torch.long)
        else:
            recog_imgs = DataPreprocessor.CropAndResizeLineImages(
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
            gt_shrink = gt_shrink.to(device)
            gt_thresh = gt_thresh.to(device)
            gt_mask = gt_mask.to(device)
            recog_imgs = recog_imgs.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

        return {
            "detect_img": img_rgb,
            "scaled_boxes": boxes_np,
            "det_boxes": det_boxes,
            "rec_boxes": rec_boxes,
            "gt_shrink": gt_shrink,
            "gt_thresh": gt_thresh,
            "gt_mask": gt_mask,
            "recog_imgs": recog_imgs,
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
        maxW: int = 256,
        device: Optional[torch.device] = None,) -> Dict[str, Any]:
        image_t = DataPreprocessor.ToImageTensor(image)
        _, h_img, w_img = image_t.shape

        recog_imgs = DataPreprocessor.CropAndResizeLineImages(
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
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

        return {
            "recog_imgs": recog_imgs,
            "targets": targets,
            "target_lengths": target_lengths,
            "norm_text": norm_text,
            "ignore": ignore,}
