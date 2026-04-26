#!/usr/bin/env python3
"""Plot training/validation metrics from MMDetection vis_data/scalars.json."""
import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_scalars_json(work_dir):
    """Find the most recently modified scalars.json under work_dir."""
    pattern = os.path.join(work_dir, "*", "vis_data", "scalars.json")
    candidates = glob.glob(pattern)
    if not candidates:
        # also try direct path
        direct = os.path.join(work_dir, "vis_data", "scalars.json")
        if os.path.isfile(direct):
            return direct
        raise FileNotFoundError(
            f"No scalars.json found under {work_dir}. "
            "Expected: <work_dir>/<timestamp>/vis_data/scalars.json"
        )
    return max(candidates, key=os.path.getmtime)


def parse_scalars(path):
    """Return (train_records, val_records) as lists of dicts."""
    train_records = []
    val_records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "epoch" in d:
                train_records.append(d)
            elif any(k.startswith("coco/") for k in d):
                val_records.append(d)
    return train_records, val_records


def aggregate_train_by_epoch(records):
    """Average each loss metric per epoch."""
    loss_keys = [
        k for k in records[0] if k.startswith("loss") or k == "lr"
    ] if records else []
    buckets = defaultdict(lambda: defaultdict(list))
    for r in records:
        ep = r["epoch"]
        for k in loss_keys:
            if k in r:
                buckets[ep][k].append(r[k])
    epochs = sorted(buckets)
    result = {k: [] for k in loss_keys}
    result["epoch"] = epochs
    for ep in epochs:
        for k in loss_keys:
            vals = buckets[ep][k]
            result[k].append(sum(vals) / len(vals) if vals else float("nan"))
    return result


def aggregate_val_by_epoch(records):
    """Collect validation metrics; use 'step' as epoch index."""
    if not records:
        return {}
    val_keys = [k for k in records[0] if k.startswith("coco/")]
    result = {k: [] for k in val_keys}
    result["epoch"] = []
    for r in sorted(records, key=lambda x: x.get("step", 0)):
        result["epoch"].append(r.get("step", len(result["epoch"]) + 1))
        for k in val_keys:
            result[k].append(r.get(k, float("nan")))
    return result


def plot_train_losses(train_agg, out_dir):
    epochs = train_agg["epoch"]
    loss_keys = [k for k in train_agg if k not in ("epoch", "lr")]
    if not loss_keys:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in loss_keys:
        ax.plot(epochs, train_agg[k], label=k)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses per Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "train_losses.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_lr(train_agg, out_dir):
    if "lr" not in train_agg:
        return
    epochs = train_agg["epoch"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_agg["lr"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "learning_rate.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_val_metrics(val_agg, out_dir):
    if not val_agg:
        return
    epochs = val_agg["epoch"]

    # Group: bbox metrics and segm metrics separately
    groups = {
        "bbox": [k for k in val_agg if "bbox_mAP" in k],
        "segm": [k for k in val_agg if "segm_mAP" in k],
    }

    for gname, keys in groups.items():
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in keys:
            vals = val_agg[k]
            # Filter out -1 (means "no instances in that size bucket")
            clean_vals = [v if v >= 0 else float("nan") for v in vals]
            label = k.replace("coco/", "")
            ax.plot(epochs, clean_vals, label=label, marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        title = "Bbox mAP" if gname == "bbox" else "Segmentation mAP"
        ax.set_title(f"Validation {title} per Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out_dir, f"val_{gname}_mAP.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


def plot_combined(train_agg, val_agg, out_dir):
    """One figure: total loss (left y) + segm/bbox mAP (right y)."""
    if not train_agg or not val_agg:
        return
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    if "loss" in train_agg:
        ax1.plot(
            train_agg["epoch"], train_agg["loss"],
            color=color_loss, label="loss (train)"
        )
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.set_ylabel("mAP", color="tab:red")
    for metric, color in [
        ("coco/segm_mAP", "tab:red"),
        ("coco/bbox_mAP", "tab:orange"),
    ]:
        if metric in val_agg:
            vals = [v if v >= 0 else float("nan") for v in val_agg[metric]]
            ax2.plot(
                val_agg["epoch"], vals,
                color=color, linestyle="--", marker="o", markersize=4,
                label=metric.replace("coco/", "")
            )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax1.set_title("Training Loss & Validation mAP")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "loss_and_mAP.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot MMDetection training metrics from scalars.json"
    )
    parser.add_argument("work_dir", help="Training work directory")
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for plots (default: <work_dir>/metric_plots)"
    )
    args = parser.parse_args()

    scalars_path = find_scalars_json(args.work_dir)
    print(f"Reading: {scalars_path}")

    out_dir = args.out_dir or os.path.join(args.work_dir, "metric_plots")
    os.makedirs(out_dir, exist_ok=True)

    train_records, val_records = parse_scalars(scalars_path)
    print(f"  Train records: {len(train_records)}, Val records: {len(val_records)}")

    train_agg = aggregate_train_by_epoch(train_records) if train_records else {}
    val_agg = aggregate_val_by_epoch(val_records) if val_records else {}

    plot_train_losses(train_agg, out_dir)
    plot_lr(train_agg, out_dir)
    plot_val_metrics(val_agg, out_dir)
    plot_combined(train_agg, val_agg, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
