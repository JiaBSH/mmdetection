"""Lean single-image sliding-window sensitivity evaluation.

This module deliberately bypasses the geometry-analysis and visualization
pipeline. It performs only model inference, COCO bbox/segm evaluation, and
pixel-union Precision/Recall/F1/IoU computation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


WINDOW_SIZES = (192, 256, 320, 400, 512)
OVERLAP_RATIOS = (0.0, 0.10, 0.15, 0.20, 0.30)


def parameter_grid(
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
    overlap_ratios: tuple[float, ...] = OVERLAP_RATIOS,
) -> list[tuple[int, float]]:
    return [
        (size, overlap)
        for size in window_sizes
        for overlap in overlap_ratios
    ]


def build_prediction_union_mask(
    instances: list[dict[str, Any]],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.bool_)
    for instance in instances:
        coords = np.asarray(instance.get("coords", []))
        if coords.ndim != 2 or coords.shape[1] != 2:
            continue
        ys = coords[:, 0].astype(np.int64, copy=False)
        xs = coords[:, 1].astype(np.int64, copy=False)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        mask[ys[valid], xs[valid]] = True
    return mask


def result_filename(patch_size: int, overlap_ratio: float) -> str:
    overlap_text = f"{float(overlap_ratio):.2f}".replace(".", "p")
    return f"window_{int(patch_size):04d}_overlap_{overlap_text}.json"


def normalize_result(
    *,
    patch_size: int,
    overlap_ratio: float,
    image: str,
    image_id: int,
    model_name: str,
    checkpoint: str,
    coco_metrics: dict[str, float],
    pixel_metrics: dict[str, float],
    inference_seconds: float,
    instance_count: int,
    window_count: int,
) -> dict[str, Any]:
    return {
        "patch_size": int(patch_size),
        "overlap_ratio": float(overlap_ratio),
        "image": str(image),
        "image_id": int(image_id),
        "model_name": str(model_name),
        "checkpoint": str(checkpoint),
        "bbox_mAP": float(coco_metrics["bbox_mAP"]),
        "bbox_mAP_50": float(coco_metrics["bbox_mAP_50"]),
        "bbox_mAP_75": float(coco_metrics["bbox_mAP_75"]),
        "segm_mAP": float(coco_metrics["segm_mAP"]),
        "segm_mAP_50": float(coco_metrics["segm_mAP_50"]),
        "segm_mAP_75": float(coco_metrics["segm_mAP_75"]),
        "pixel_precision": float(pixel_metrics["Precision"]),
        "pixel_recall": float(pixel_metrics["Recall"]),
        "pixel_f1": float(pixel_metrics["F1-score"]),
        "pixel_iou": float(pixel_metrics["IoU"]),
        "inference_seconds": float(inference_seconds),
        "instance_count": int(instance_count),
        "window_count": int(window_count),
    }


def _load_image_record(ann_file: Path, image_path: Path) -> dict[str, Any]:
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    target_name = image_path.name
    for record in data.get("images", []):
        if Path(str(record.get("file_name", ""))).name == target_name:
            return record
    raise ValueError(f"Image {target_name!r} not found in {ann_file}")


def _build_gt_union_mask(
    ann_file: Path,
    image_name: str,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    from postprocess._pixel_metrics import build_pred_mask_from_polygons
    from postprocess.coco_utils import load_coco_gt_polygons

    polygons, ann_width, ann_height = load_coco_gt_polygons(
        str(ann_file),
        image_filename=image_name,
    )
    if (ann_width, ann_height) != (width, height):
        raise ValueError(
            "Annotation/image size mismatch: "
            f"annotation={(ann_width, ann_height)}, image={(width, height)}"
        )
    return build_pred_mask_from_polygons(polygons, (width, height))


def evaluate_configuration(
    *,
    ann_file: Path,
    image_path: Path,
    model_config: Path,
    checkpoint: Path,
    model_name: str,
    patch_size: int,
    overlap_ratio: float,
    batch_size: int,
    score_threshold: float,
    coco_max_dets: int,
    device: str,
) -> dict[str, Any]:
    from postprocess._coco_eval import (
        COCOResultCollector,
        evaluate_coco_from_predictions,
    )
    from postprocess._pixel_metrics import compute_pixel_metrics
    from postprocess.run_postprocess import _infer_one_image, _load_model

    image_record = _load_image_record(ann_file, image_path)
    image_id = int(image_record["id"])
    width = int(image_record["width"])
    height = int(image_record["height"])

    model = _load_model(str(model_config), str(checkpoint), device=device)
    started = time.perf_counter()
    instances, pil_image, windows, _merge_records = _infer_one_image(
        model,
        str(image_path),
        score_thresh=score_threshold,
        target_label=0,
        min_pixel_count=10,
        device=device,
        sliding_window=True,
        patch_size=patch_size,
        patch_overlap_ratio=overlap_ratio,
        batch_size=batch_size,
    )
    inference_seconds = time.perf_counter() - started

    if pil_image.size != (width, height):
        raise ValueError(
            f"Annotation/image size mismatch: annotation={(width, height)}, "
            f"image={pil_image.size}"
        )

    pred_mask = build_prediction_union_mask(
        instances,
        height=height,
        width=width,
    )
    gt_mask = _build_gt_union_mask(
        ann_file,
        image_path.name,
        width=width,
        height=height,
    )
    pixel_metrics = compute_pixel_metrics(gt_mask, pred_mask)

    collector = COCOResultCollector()
    collector.add_image_predictions(
        image_id=image_id,
        global_instances=instances,
        img_width=width,
        img_height=height,
    )
    coco_metrics = evaluate_coco_from_predictions(
        collector.to_coco_list(),
        str(ann_file),
        metrics=["bbox", "segm"],
        image_ids=[image_id],
        max_dets=coco_max_dets,
    )

    return normalize_result(
        patch_size=patch_size,
        overlap_ratio=overlap_ratio,
        image=image_path.name,
        image_id=image_id,
        model_name=model_name,
        checkpoint=str(checkpoint),
        coco_metrics=coco_metrics,
        pixel_metrics=pixel_metrics,
        inference_seconds=inference_seconds,
        instance_count=len(instances),
        window_count=len(windows),
    )


def _validated_grid_cell(patch_size: int, overlap_ratio: float) -> None:
    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
        )


def _existing_file(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {resolved}")
    return resolved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one 2.5x image at one sliding-window grid cell."
    )
    parser.add_argument("--ann-file", type=_existing_file, required=True)
    parser.add_argument("--image", type=_existing_file, required=True)
    parser.add_argument("--model-config", type=_existing_file, required=True)
    parser.add_argument("--checkpoint", type=_existing_file, required=True)
    parser.add_argument(
        "--model-name",
        default="detectors_htc-r50_custom_coco_instance",
    )
    parser.add_argument("--patch-size", type=int, required=True)
    parser.add_argument("--overlap-ratio", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--coco-max-dets", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    _validated_grid_cell(args.patch_size, args.overlap_ratio)
    if args.output_json.suffix.lower() != ".json":
        parser.error("--output-json must end in .json")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def _write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = evaluate_configuration(
        ann_file=args.ann_file,
        image_path=args.image,
        model_config=args.model_config,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
        patch_size=args.patch_size,
        overlap_ratio=args.overlap_ratio,
        batch_size=args.batch_size,
        score_threshold=args.score_threshold,
        coco_max_dets=args.coco_max_dets,
        device=args.device,
    )
    _write_json_atomic(result, args.output_json)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
