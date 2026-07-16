from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from postprocess.coco_utils import mmdet_masks_to_instances
from postprocess.sliding_window_infer import (
    _extract_patch_instances,
    _filter_small_instances,
    _flat_instances_to_global_instances,
    _iter_windows,
    _prepare_patch_image,
    _resolve_overlaps,
)


def _add_path(path: str | None) -> None:
    if not path:
        return
    abs_path = os.path.abspath(path)
    if abs_path not in sys.path:
        sys.path.insert(0, abs_path)


class Sam3SlidingModel:
    """SAM3 text-prompt predictor with the postprocess sliding-window contract."""

    def __init__(
        self,
        *,
        sam3_root: str,
        checkpoint_path: str,
        text_prompt: str = "Hexagon",
        device: str = "cuda:0",
        resolution: int = 1008,
    ) -> None:
        self.sam3_root = os.path.abspath(sam3_root)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.text_prompt = text_prompt
        self.device = device
        self.resolution = int(resolution)
        self.model = None
        self.processor = None
        self._torch = None

    def _load(self) -> None:
        if self.model is not None and self.processor is not None:
            return

        _add_path(self.sam3_root)
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        builder_device = "cuda" if str(self.device).startswith("cuda") else "cpu"

        if str(self.device).startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = build_sam3_image_model(
            checkpoint_path=self.checkpoint_path,
            load_from_HF=False,
            device=builder_device,
            eval_mode=True,
        )
        self.model.to(self.device)
        if (
            str(self.device).startswith("cuda")
            and os.getenv("BL_SAM3_MODEL_BF16", "0").strip().lower()
            in {"1", "true", "yes", "on"}
        ):
            self.model.to(dtype=torch.bfloat16)
        self.processor = Sam3Processor(
            self.model,
            resolution=self.resolution,
            device=self.device,
            confidence_threshold=0.5,
        )
        self._torch = torch

    def _predict_patch(
        self,
        patch_rgb: np.ndarray,
        *,
        score_thresh: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._load()
        assert self.processor is not None
        assert self._torch is not None

        self.processor.confidence_threshold = float(score_thresh)
        patch_img = Image.fromarray(patch_rgb.astype(np.uint8, copy=False)).convert("RGB")

        use_cuda_amp = (
            str(self.device).startswith("cuda")
            and os.getenv("BL_SAM3_USE_BF16", "1").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        autocast_context = (
            self._torch.autocast(
                "cuda",
                dtype=self._torch.bfloat16,
                enabled=True,
            )
            if use_cuda_amp
            else nullcontext()
        )
        with self._torch.inference_mode():
            with autocast_context:
                state = self.processor.set_image(patch_img)
                state = self.processor.set_text_prompt(
                    state=state,
                    prompt=self.text_prompt,
                )

        masks = state.get("masks")
        scores = state.get("scores")
        if masks is None or scores is None:
            h, w = patch_rgb.shape[:2]
            return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=np.float32)

        masks_np = masks.squeeze(1).detach().cpu().numpy().astype(bool)
        if masks_np.ndim == 2:
            masks_np = masks_np[np.newaxis, ...]
        scores_np = scores.detach().to(self._torch.float32).cpu().numpy()
        return masks_np, scores_np.astype(np.float32, copy=False)

    def _infer_full_image(
        self,
        img_path: str,
        *,
        score_thresh: float,
        target_label: int,
        min_pixel_count: int,
    ) -> tuple[list[dict], Image.Image, list[dict], list[dict]]:
        pil_img = Image.open(img_path).convert("RGB")
        masks_np, scores_np = self._predict_patch(
            np.asarray(pil_img),
            score_thresh=score_thresh,
        )
        labels_np = np.full((len(scores_np),), int(target_label), dtype=np.int64)
        instances = mmdet_masks_to_instances(
            masks_np,
            scores=scores_np,
            labels=labels_np,
            bboxes=None,
            score_thresh=score_thresh,
            target_label=target_label,
            min_pixel_count=min_pixel_count,
        )
        return instances, pil_img, [], []

    def infer_postprocess_instances(
        self,
        img_path: str,
        *,
        score_thresh: float = 0.5,
        target_label: int = 0,
        min_pixel_count: int = 10,
        device: str = "cuda:0",
        sliding_window: bool = False,
        patch_size: int = 1024,
        patch_overlap_ratio: float = 0.0,
        batch_size: int = 1,
    ) -> tuple[list[dict], Image.Image, list[dict], list[dict]]:
        if device:
            self.device = device
        if not sliding_window:
            return self._infer_full_image(
                img_path,
                score_thresh=score_thresh,
                target_label=target_label,
                min_pixel_count=min_pixel_count,
            )
        return infer_image_with_sam3_overlap_windows(
            self,
            img_path,
            score_thresh=score_thresh,
            target_label=target_label,
            min_pixel_count=min_pixel_count,
            patch_size=patch_size,
            patch_overlap_ratio=patch_overlap_ratio,
            batch_size=batch_size,
        )


def infer_image_with_sam3_overlap_windows(
    model: Sam3SlidingModel,
    img_path: str,
    *,
    score_thresh: float = 0.5,
    target_label: int = 0,
    min_pixel_count: int = 10,
    patch_size: int = 1024,
    patch_overlap_ratio: float = 0.0,
    batch_size: int = 1,
) -> tuple[list[dict], Image.Image, list[dict], list[dict]]:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if not (0.0 <= patch_overlap_ratio < 1.0):
        raise ValueError(
            f"patch_overlap_ratio must satisfy 0 <= ratio < 1, got {patch_overlap_ratio}"
        )

    pil_img = Image.open(img_path).convert("RGB")
    image_np = np.asarray(pil_img)
    image_height, image_width = image_np.shape[:2]

    flat_instances: list[dict] = []
    merge_records: list[dict] = []
    windows: list[dict] = []
    next_instance_id = 1

    verbose_windows = os.getenv("BL_SAM3_VERBOSE_WINDOWS", "1").strip().lower()
    show_progress = verbose_windows not in {"0", "false", "no", "off"}

    all_windows = list(
        _iter_windows(
            image_height,
            image_width,
            patch_size,
            patch_overlap_ratio,
        )
    )

    for win_idx, window in enumerate(all_windows, start=1):
        windows.append(
            {
                "idx": int(window["idx"]),
                "left": int(window["left"]),
                "top": int(window["top"]),
                "right": int(window["right"]),
                "bottom": int(window["bottom"]),
                "core_left": float(window["core_left"]),
                "core_top": float(window["core_top"]),
                "core_right": float(window["core_right"]),
                "core_bottom": float(window["core_bottom"]),
                "safe_left": float(window["safe_left"]),
                "safe_top": float(window["safe_top"]),
                "safe_right": float(window["safe_right"]),
                "safe_bottom": float(window["safe_bottom"]),
                "grid_x": int(window["grid_x"]),
                "grid_y": int(window["grid_y"]),
                "overlap": int(window["overlap"]),
                "requested_overlap": int(window["requested_overlap"]),
                "context_margin": int(window["context_margin"]),
                "edge_touch_margin": int(window["edge_touch_margin"]),
            }
        )

        if show_progress and (win_idx == 1 or win_idx == len(all_windows) or win_idx % 10 == 0):
            print(
                f"  SAM3 sliding window {win_idx}/{len(all_windows)} "
                f"(batch-size parameter={max(int(batch_size), 1)})"
            )

        patch_rgb = _prepare_patch_image(
            image_np,
            int(window["left"]),
            int(window["top"]),
            int(window["right"]),
            int(window["bottom"]),
            patch_size,
        )
        masks_np, scores_np = model._predict_patch(
            patch_rgb,
            score_thresh=score_thresh,
        )
        labels_np = np.full((len(scores_np),), int(target_label), dtype=np.int64)
        pred_instances = SimpleNamespace(
            masks=masks_np,
            scores=scores_np,
            labels=labels_np,
        )
        patch_instances, next_instance_id = _extract_patch_instances(
            pred_instances,
            window=window,
            image_height=image_height,
            image_width=image_width,
            score_thresh=score_thresh,
            target_label=target_label,
            min_pixel_count=min_pixel_count,
            next_instance_id=next_instance_id,
        )
        flat_instances.extend(patch_instances)

    flat_instances = _resolve_overlaps(
        flat_instances,
        image_height,
        image_width,
        merge_records,
    )
    flat_instances = _filter_small_instances(
        flat_instances,
        min_pixel_count,
        merge_records,
    )
    return (
        _flat_instances_to_global_instances(flat_instances, image_width),
        pil_img,
        windows,
        merge_records,
    )
