"""Aggregate compact sensitivity JSON files and draw requested heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from postprocess.window_sensitivity import (
    OVERLAP_RATIOS,
    WINDOW_SIZES,
    parameter_grid,
)


HEATMAP_METRICS = (
    ("segm_mAP", "COCO Segm"),
    ("bbox_mAP", "COCO Box"),
    ("pixel_precision", "Precision"),
    ("pixel_recall", "Recall"),
    ("pixel_f1", "F1-score"),
    ("pixel_iou", "IoU"),
)

SUMMARY_FIELDS = (
    "patch_size",
    "overlap_ratio",
    "segm_mAP",
    "segm_mAP_50",
    "segm_mAP_75",
    "bbox_mAP",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_iou",
    "inference_seconds",
    "instance_count",
    "window_count",
    "image",
    "image_id",
    "model_name",
    "checkpoint",
)


def load_grid_results(raw_dir: Path) -> list[dict[str, Any]]:
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)

    expected = set(parameter_grid())
    by_cell: dict[tuple[int, float], dict[str, Any]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        cell = (int(row["patch_size"]), float(row["overlap_ratio"]))
        if cell in by_cell:
            raise ValueError(f"duplicate grid cell {cell}: {path}")
        by_cell[cell] = row

    actual = set(by_cell)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        raise ValueError(f"missing grid cells: {missing}")
    if unexpected:
        raise ValueError(f"unexpected grid cells: {unexpected}")

    metric_keys = [key for key, _label in HEATMAP_METRICS]
    rows: list[dict[str, Any]] = []
    for cell in parameter_grid():
        row = by_cell[cell]
        for key in metric_keys:
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"non-finite {key} at grid cell {cell}")
        rows.append(row)
    return rows


def load_multimage_grid_results(
    raw_dir: Path,
    window_sizes: tuple[int, ...],
    overlap_ratios: tuple[float, ...],
    expected_images: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise NotADirectoryError(raw_dir)
    if len(expected_images) != 4 or len(set(expected_images)) != 4:
        raise ValueError("exactly four images are required")

    expected_cells = set(parameter_grid(window_sizes, overlap_ratios))
    by_key: dict[tuple[int, float, str], dict[str, Any]] = {}
    for path in sorted(raw_dir.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        cell = (int(row["patch_size"]), float(row["overlap_ratio"]))
        image = str(row["image"])
        if cell not in expected_cells or image not in expected_images:
            raise ValueError(f"unexpected result {cell + (image,)} in {path}")
        key = cell + (image,)
        if key in by_key:
            raise ValueError(f"duplicate image result {key}: {path}")
        for metric, _label in HEATMAP_METRICS:
            if not np.isfinite(float(row[metric])):
                raise ValueError(f"non-finite {metric} in {path}")
        by_key[key] = row

    per_image: list[dict[str, Any]] = []
    means: list[dict[str, Any]] = []
    for size, overlap in parameter_grid(window_sizes, overlap_ratios):
        image_rows = [
            by_key.get((size, overlap, image)) for image in expected_images
        ]
        if any(row is None for row in image_rows):
            present = sum(row is not None for row in image_rows)
            raise ValueError(
                f"four images required at {(size, overlap)}, found {present}"
            )
        complete_rows = [row for row in image_rows if row is not None]
        per_image.extend(complete_rows)
        mean_row: dict[str, Any] = {
            "patch_size": size,
            "overlap_ratio": overlap,
            "image_count": len(complete_rows),
            "reused_count": sum("reused_from" in row for row in complete_rows),
        }
        for metric, _label in HEATMAP_METRICS:
            values = np.asarray(
                [float(row[metric]) for row in complete_rows], dtype=np.float64
            )
            mean_row[metric] = float(values.mean())
            mean_row[f"{metric}_std"] = float(values.std(ddof=1))
        means.append(mean_row)
    return per_image, means


def _metric_matrix(
    rows: list[dict[str, Any]],
    metric: str,
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
    overlap_ratios: tuple[float, ...] = OVERLAP_RATIOS,
) -> np.ndarray:
    values = {
        (int(row["patch_size"]), float(row["overlap_ratio"])): float(row[metric])
        for row in rows
    }
    return np.asarray(
        [
            [values[(size, overlap)] for overlap in overlap_ratios]
            for size in window_sizes
        ],
        dtype=np.float64,
    )


def _draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    *,
    include_legend: bool,
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
    overlap_ratios: tuple[float, ...] = OVERLAP_RATIOS,
) -> Any:
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(overlap_ratios)))
    ax.set_xticklabels([f"{value:.2f}" for value in overlap_ratios])
    ax.set_yticks(range(len(window_sizes)))
    ax.set_yticklabels([str(value) for value in window_sizes])
    ax.set_xlabel("Sliding-window overlap ratio")
    ax.set_ylabel("Window size (px)")
    ax.set_title(title)

    span = float(np.max(matrix) - np.min(matrix))
    midpoint = float(np.min(matrix) + 0.55 * span)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            color = "white" if span > 0 and value >= midpoint else "black"
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    has_marker = False
    if 0.15 in overlap_ratios and 256 in window_sizes:
        ax.scatter(
            [overlap_ratios.index(0.15)],
            [window_sizes.index(256)],
            marker="*",
            s=280,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="DINOv2 (256, 0.15)",
        )
        has_marker = True
    if 0.15 in overlap_ratios and 400 in window_sizes:
        ax.scatter(
            [overlap_ratios.index(0.15)],
            [window_sizes.index(400)],
            marker="s",
            s=190,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="Manual (400, 0.15)",
        )
        has_marker = True
    if include_legend and has_marker:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return image


def write_summaries(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in SUMMARY_FIELDS})

    summary = {
        "window_sizes": list(WINDOW_SIZES),
        "overlap_ratios": list(OVERLAP_RATIOS),
        "dino_setting": {"patch_size": 256, "overlap_ratio": 0.15},
        "manual_setting": {"patch_size": 400, "overlap_ratio": 0.15},
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def render_heatmaps(
    rows: list[dict[str, Any]],
    output_dir: Path,
    window_sizes: tuple[int, ...] = WINDOW_SIZES,
    overlap_ratios: tuple[float, ...] = OVERLAP_RATIOS,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, title in HEATMAP_METRICS:
        matrix = _metric_matrix(rows, metric, window_sizes, overlap_ratios)
        figure, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        image = _draw_heatmap(
            axis,
            matrix,
            title,
            include_legend=True,
            window_sizes=window_sizes,
            overlap_ratios=overlap_ratios,
        )
        figure.colorbar(image, ax=axis, label=title)
        filename = f"window_sensitivity_{metric}"
        figure.savefig(output_dir / f"{filename}.png", dpi=300)
        figure.savefig(output_dir / f"{filename}.svg")
        plt.close(figure)

    combined, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10.5),
        constrained_layout=True,
    )
    for index, ((metric, title), axis) in enumerate(
        zip(HEATMAP_METRICS, axes.flat)
    ):
        image = _draw_heatmap(
            axis,
            _metric_matrix(rows, metric, window_sizes, overlap_ratios),
            title,
            include_legend=index == 0,
            window_sizes=window_sizes,
            overlap_ratios=overlap_ratios,
        )
        combined.colorbar(image, ax=axis, label=title, shrink=0.9)
    combined.savefig(output_dir / "window_sensitivity_combined.png", dpi=300)
    combined.savefig(output_dir / "window_sensitivity_combined.svg")
    plt.close(combined)


def find_metric_peaks(
    rows: list[dict[str, Any]],
    window_sizes: tuple[int, ...],
    overlap_ratios: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for metric, _label in HEATMAP_METRICS:
        matrix = _metric_matrix(rows, metric, window_sizes, overlap_ratios)
        maximum = float(np.max(matrix))
        locations = np.argwhere(np.isclose(matrix, maximum, rtol=0.0, atol=1e-12))
        row_index, column_index = (int(value) for value in locations[0])
        all_cells = [
            {
                "patch_size": int(window_sizes[int(r)]),
                "overlap_ratio": float(overlap_ratios[int(c)]),
            }
            for r, c in locations
        ]
        on_boundary = any(
            int(r) in (0, len(window_sizes) - 1)
            or int(c) in (0, len(overlap_ratios) - 1)
            for r, c in locations
        )
        report[metric] = {
            "value": maximum,
            "patch_size": int(window_sizes[row_index]),
            "overlap_ratio": float(overlap_ratios[column_index]),
            "on_boundary": bool(on_boundary),
            "max_cells": all_cells,
        }
    return report


def write_multimage_outputs(
    per_image_rows: list[dict[str, Any]],
    mean_rows: list[dict[str, Any]],
    output_dir: Path,
    window_sizes: tuple[int, ...],
    overlap_ratios: tuple[float, ...],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_image_fields = SUMMARY_FIELDS + ("reused_from",)
    with (output_dir / "summary_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_fields)
        writer.writeheader()
        for row in per_image_rows:
            writer.writerow({key: row.get(key, "") for key in per_image_fields})

    mean_fields = ["patch_size", "overlap_ratio", "image_count", "reused_count"]
    for metric, _label in HEATMAP_METRICS:
        mean_fields.extend((metric, f"{metric}_std"))
    with (output_dir / "summary_mean.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=mean_fields)
        writer.writeheader()
        writer.writerows(mean_rows)

    summary = {
        "window_sizes": list(window_sizes),
        "overlap_ratios": list(overlap_ratios),
        "image_records": len(per_image_rows),
        "grid_cells": len(mean_rows),
        "reused_records": sum("reused_from" in row for row in per_image_rows),
        "peaks": find_metric_peaks(mean_rows, window_sizes, overlap_ratios),
        "rows": mean_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render_heatmaps(mean_rows, output_dir, window_sizes, overlap_ratios)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a complete 2.5x sensitivity grid and plot heatmaps."
    )
    parser.add_argument("--mode", choices=("single", "multimage"), default="single")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-sizes", type=int, nargs="+")
    parser.add_argument("--overlap-ratios", type=float, nargs="+")
    parser.add_argument("--expected-images", nargs="+")
    args = parser.parse_args(argv)
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is used")
    if args.mode == "multimage" and not (
        args.window_sizes and args.overlap_ratios and args.expected_images
    ):
        parser.error(
            "multimage mode requires --window-sizes, --overlap-ratios, "
            "and --expected-images"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "multimage":
        window_sizes = tuple(args.window_sizes)
        overlap_ratios = tuple(args.overlap_ratios)
        expected_images = tuple(args.expected_images)
        per_image, means = load_multimage_grid_results(
            args.raw_dir,
            window_sizes,
            overlap_ratios,
            expected_images,
        )
        reused = sum("reused_from" in row for row in per_image)
        print(
            f"cells={len(means)} images_per_cell={len(expected_images)} "
            f"records={len(per_image)} reused={reused} finite_metrics=yes"
        )
        if args.validate_only:
            return 0
        output_dir = args.output_dir.expanduser().resolve()
        write_multimage_outputs(
            per_image,
            means,
            output_dir,
            window_sizes,
            overlap_ratios,
        )
        print(f"saved_grid_rows={len(means)} output_dir={output_dir}")
        return 0

    rows = load_grid_results(args.raw_dir)
    if args.validate_only:
        print(f"cells={len(rows)} records={len(rows)} finite_metrics=yes")
        return 0
    output_dir = args.output_dir.expanduser().resolve()
    write_summaries(rows, output_dir)
    render_heatmaps(rows, output_dir)
    print(f"saved_grid_rows={len(rows)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
