from __future__ import annotations

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


def ResolveOCRTextsDir(root: Path, *, preferNamedDir: bool = False) -> Path:
    ocr_texts_dir = root / "OCRTexts"
    if preferNamedDir or ocr_texts_dir.exists():
        return ocr_texts_dir
    return root / "texts"


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
        self.boxes = sorted((p / "boxes").glob("*.npy"))
        self.texts = sorted(ResolveOCRTextsDir(p).glob("*.txt"))

        assert len(self.imgs) == len(self.boxes) == len(self.texts), "frames/boxes/texts The number of files is inconsistent."
        if len(self.imgs) == 0:
            raise RuntimeError(f"no OCR samples found under {p}")

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        if iio is None:
            raise RuntimeError("imageio.v3 cant use")

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
        return "".join(ch for ch in str(text).strip() if ch in char2Idx)

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

        det_boxes: List[np.ndarray] = []
        rec_boxes: List[np.ndarray] = []
        rec_texts: List[str] = []

        for box, raw_text in zip(boxes_np, texts_list):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1 = max(0, min(imageSize - 1, x1))
            y1 = max(0, min(imageSize - 1, y1))
            x2 = max(x1 + 1, min(imageSize, x2))
            y2 = max(y1 + 1, min(imageSize, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            det_boxes.append(np.array([x1, y1, x2, y2], dtype=np.int32))

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
            "resize_meta": resize_meta,}
