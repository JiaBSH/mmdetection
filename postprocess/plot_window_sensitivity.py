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


def _metric_matrix(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
    values = {
        (int(row["patch_size"]), float(row["overlap_ratio"])): float(row[metric])
        for row in rows
    }
    return np.asarray(
        [
            [values[(size, overlap)] for overlap in OVERLAP_RATIOS]
            for size in WINDOW_SIZES
        ],
        dtype=np.float64,
    )


def _draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    *,
    include_legend: bool,
) -> Any:
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(OVERLAP_RATIOS)))
    ax.set_xticklabels([f"{value:.2f}" for value in OVERLAP_RATIOS])
    ax.set_yticks(range(len(WINDOW_SIZES)))
    ax.set_yticklabels([str(value) for value in WINDOW_SIZES])
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

    dino_x = OVERLAP_RATIOS.index(0.15)
    dino_y = WINDOW_SIZES.index(256)
    manual_x = OVERLAP_RATIOS.index(0.15)
    manual_y = WINDOW_SIZES.index(400)
    ax.scatter(
        [dino_x],
        [dino_y],
        marker="*",
        s=280,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        label="DINOv2 (256, 0.15)",
    )
    ax.scatter(
        [manual_x],
        [manual_y],
        marker="s",
        s=190,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        label="Manual (400, 0.15)",
    )
    if include_legend:
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


def render_heatmaps(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, title in HEATMAP_METRICS:
        matrix = _metric_matrix(rows, metric)
        figure, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
        image = _draw_heatmap(axis, matrix, title, include_legend=True)
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
            _metric_matrix(rows, metric),
            title,
            include_legend=index == 0,
        )
        combined.colorbar(image, ax=axis, label=title, shrink=0.9)
    combined.savefig(output_dir / "window_sensitivity_combined.png", dpi=300)
    combined.savefig(output_dir / "window_sensitivity_combined.svg")
    plt.close(combined)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate a complete 2.5x sensitivity grid and plot heatmaps."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = load_grid_results(args.raw_dir)
    output_dir = args.output_dir.expanduser().resolve()
    write_summaries(rows, output_dir)
    render_heatmaps(rows, output_dir)
    print(f"saved_grid_rows={len(rows)} output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
