#!/usr/bin/env python3
"""Plot grouped comparison charts for multiple magnification scales."""

import argparse
import csv
import os
from typing import Dict, Iterable, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = ("iou", "precision", "recall", "f1")
METRIC_LABELS = ("IoU", "Precision", "Recall", "F1")
DEFAULT_SCALES = ("20x", "50x", "100x")
DEFAULT_RELATIVE_CSV = os.path.join("plain", "comparison_mean.csv")
EXCLUDED_MODELS = ("mask-rcnn_convnext-v2-b", "sparseinst_r50")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read comparison_mean.csv under multiple scale directories and "
            "generate one grouped bar chart per scale."
        )
    )
    parser.add_argument(
        "--root-dir",
        default="outputs/custom_all_main_es_test_set_1024",
        help="Root directory containing 20x/50x/100x subdirectories.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=list(DEFAULT_SCALES),
        help="Scale subdirectories to plot.",
    )
    parser.add_argument(
        "--csv-relative-path",
        default=DEFAULT_RELATIVE_CSV,
        help="Relative CSV path inside each scale directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save generated figures. Defaults to <root-dir>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    return parser.parse_args()


def simplify_model_name(model_name: str) -> str:
    suffixes = (
        "_custom_coco_instance",
        "_custom_coco",
    )
    for suffix in suffixes:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


def read_rows(csv_path: str) -> List[Dict[str, Union[float, str]]]:
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            parsed = {"model": row["model"]}
            for metric in METRICS:
                parsed[metric] = float(row[metric])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")
    return rows


def build_output_dir(root_dir: str, output_dir: Optional[str]) -> str:
    target_dir = output_dir or os.path.join(root_dir, "plots")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def plot_scale(
    scale: str,
    rows: Iterable[Dict[str, Union[float, str]]],
    output_dir: str,
    dpi: int,
) -> str:
    row_list = [
        row for row in rows
        if simplify_model_name(str(row["model"])) not in EXCLUDED_MODELS
    ]
    if not row_list:
        raise ValueError(
            f"No rows left to plot for {scale} after excluding: "
            f"{', '.join(EXCLUDED_MODELS)}"
        )
    model_names = [simplify_model_name(str(row["model"])) for row in row_list]
    x = np.arange(len(model_names), dtype=float)
    width = 0.18
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

    fig_width = max(12, len(model_names) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    for index, (metric, label, color) in enumerate(zip(METRICS, METRIC_LABELS, colors)):
        values = [float(row[metric]) for row in row_list]
        offset = (index - (len(METRICS) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, label=label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title(f"Model Comparison for {scale}", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    plt.tight_layout(rect=(0, 0, 0.86, 1))
    output_path = os.path.join(output_dir, f"comparison_metrics_{scale}.png")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    root_dir = os.path.abspath(args.root_dir)
    output_dir = build_output_dir(root_dir, args.output_dir)

    generated_files = []
    for scale in args.scales:
        csv_path = os.path.join(root_dir, scale, args.csv_relative_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Missing CSV for {scale}: {csv_path}")
        rows = read_rows(csv_path)
        generated_files.append(plot_scale(scale, rows, output_dir, args.dpi))

    print("Generated figures:")
    for file_path in generated_files:
        print(file_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())