"""多模型批量对比后处理评估脚本

对多个模型（Mask R-CNN / SOLOv2 等）在同一个测试集上做推理，
然后对每个模型的预测实例分别做后处理几何分析，最后汇总到一张对比CSV。

用法示例:

  python postprocess/compare_models.py \
      --ann-file dataset_root/dataset_1024_aug/annotations/instances_test.json \
      --img-dir  dataset_root/dataset_1024_aug/images/test \
      --out-dir  outputs/model_comparison \
      --models \
        "MaskRCNN:work_dirs/custom_maskrcnn_1cls_1024_aug/mask-rcnn_r50_fpn_1x_custom_coco_instance.py:work_dirs/custom_maskrcnn_1cls_1024_aug/epoch_50.pth" \
        "SOLOv2:work_dirs/custom_solov2_1cls_1024_aug/solov2_r50_fpn_1x_custom_coco_instance.py:work_dirs/custom_solov2_1cls_1024_aug/epoch_50.pth" \
      --score-thresh 0.5 \
      --enable-poly-metrics

  # 也可以通过 YAML 配置文件指定模型列表（见 --model-cfg）
  python postprocess/compare_models.py \
      --ann-file ... --img-dir ... --out-dir ... \
      --model-cfg postprocess/model_list.yaml

每个模型的输出目录: <out-dir>/<model_name>/
汇总对比CSV:        <out-dir>/comparison_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_TEMP_DIR = os.path.join(_REPO_ROOT, "temp")
if _TEMP_DIR not in sys.path:
    sys.path.insert(0, _TEMP_DIR)

from postprocess.run_postprocess import (  # noqa: E402
    _get_test_images,
    _load_model,
    process_one_image,
)


# ---------------------------------------------------------------------------
# 配置解析
# ---------------------------------------------------------------------------

def _parse_model_spec(spec: str) -> tuple[str, str, str]:
    """解析 "ModelName:config_path:checkpoint_path" 格式。"""
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"模型规格格式错误，应为 'Name:config:checkpoint'，实际: {spec!r}"
        )
    name, config, ckpt = [p.strip() for p in parts]
    return name, config, ckpt


def _load_model_list_yaml(yaml_path: str) -> list[tuple[str, str, str]]:
    """从YAML文件加载模型列表。

    YAML格式:
      models:
        - name: MaskRCNN
          config: work_dirs/.../config.py
          checkpoint: work_dirs/.../epoch_50.pth
        - name: SOLOv2
          ...
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ImportError("需要安装 PyYAML: pip install pyyaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    result = []
    for m in data.get("models", []):
        result.append((str(m["name"]), str(m["config"]), str(m["checkpoint"])))
    return result


# ---------------------------------------------------------------------------
# 单模型评估
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    config: str,
    checkpoint: str,
    img_list: list[tuple[str, str]],
    ann_file: str,
    out_dir: str,
    *,
    score_thresh: float = 0.5,
    min_pixel_count: int = 10,
    scale_ratio: float | None = None,
    scale_unit: str | None = None,
    enable_plots: bool = False,
    enable_gt: bool = False,
    enable_polygon_metrics: bool = False,
    device: str = "cuda:0",
    verbose: bool = True,
) -> list[dict]:
    """对一个模型评估所有测试图，返回每张图的指标rows。"""
    model_out_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"# 模型: {model_name}")
    print(f"# Config:     {config}")
    print(f"# Checkpoint: {checkpoint}")
    print(f"{'#'*60}")

    model = _load_model(config, checkpoint, device=device)

    rows: list[dict] = []
    for img_path, img_name in img_list:
        stem = os.path.splitext(img_name)[0]
        out_dir_i = os.path.join(model_out_dir, stem)
        try:
            row = process_one_image(
                model,
                img_path,
                ann_file,
                out_dir_i,
                score_thresh=score_thresh,
                target_label=0,
                min_pixel_count=min_pixel_count,
                scale_ratio=scale_ratio,
                scale_unit=scale_unit,
                enable_plots=True if enable_plots else None,
                enable_gt=True if enable_gt else None,
                enable_polygon_metrics=True if enable_polygon_metrics else None,
                device=device,
                verbose=verbose,
            )
        except Exception:
            traceback.print_exc()
            row = {
                "image": img_name,
                "iou": float("nan"), "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan"),
                "pred_count": float("nan"), "gt_count": float("nan"),
                "pred_coverage": float("nan"), "gt_coverage": float("nan"),
            }
        row["model"] = model_name
        rows.append(row)

    # 保存单模型汇总CSV
    model_csv = os.path.join(model_out_dir, "metrics_summary.csv")
    fieldnames = ["model", "image", "iou", "precision", "recall", "f1",
                  "pred_count", "gt_count", "pred_coverage", "gt_coverage"]
    with open(model_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if np.isfinite(r.get("iou", float("nan")))]
    if valid:
        mean_iou = float(np.mean([r["iou"] for r in valid]))
        mean_f1  = float(np.mean([r["f1"]  for r in valid]))
        print(f"\n📊 {model_name}: 平均IoU={mean_iou:.4f}  平均F1={mean_f1:.4f}  ({len(valid)}/{len(rows)}张)")
    else:
        print(f"\n⚠️ {model_name}: 无有效指标")

    return rows


# ---------------------------------------------------------------------------
# 汇总对比
# ---------------------------------------------------------------------------

def write_comparison_csv(all_rows: list[dict], out_path: str) -> None:
    """将所有模型、所有图的结果写入同一CSV，便于横向对比。"""
    fieldnames = ["model", "image", "iou", "precision", "recall", "f1",
                  "pred_count", "gt_count", "pred_coverage", "gt_coverage"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)


def write_mean_comparison_csv(all_rows: list[dict], out_path: str) -> None:
    """每个模型的各指标均值汇总。"""
    from collections import defaultdict

    model_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        model_rows[row["model"]].append(row)

    metrics = ["iou", "precision", "recall", "f1", "pred_count", "gt_count",
               "pred_coverage", "gt_coverage"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + metrics)
        writer.writeheader()
        for model_name, rows in sorted(model_rows.items()):
            mean_row = {"model": model_name}
            for m in metrics:
                vals = [r[m] for r in rows if np.isfinite(r.get(m, float("nan")))]
                mean_row[m] = float(np.mean(vals)) if vals else float("nan")
            writer.writerow(mean_row)
            print(f"  {model_name}: IoU={mean_row['iou']:.4f}  F1={mean_row['f1']:.4f}")


def plot_comparison(mean_csv: str, out_dir: str) -> None:
    """为对比均值CSV生成柱状图（可选，依赖matplotlib）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as csv_mod

        with open(mean_csv, "r", encoding="utf-8") as f:
            rows = list(csv_mod.DictReader(f))

        if not rows:
            return

        metrics = ["iou", "precision", "recall", "f1"]
        metric_labels = ["IoU", "Precision", "Recall", "F1"]
        model_names = [r["model"] for r in rows]
        x = np.arange(len(model_names))
        width = 0.18

        fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2), 5))
        for i, (m, label) in enumerate(zip(metrics, metric_labels)):
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[m]))
                except Exception:
                    vals.append(float("nan"))
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label)
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison (mean over test set)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "comparison_bar.png"), dpi=150)
        plt.close()
        print(f"  📊 对比柱状图: {os.path.join(out_dir, 'comparison_bar.png')}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="多模型实例分割后处理对比评估（COCO格式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ann-file", required=True,
                   help="COCO格式标注文件（instances_test.json等）")
    p.add_argument("--img-dir",  required=True,
                   help="测试图像目录")
    p.add_argument("--out-dir",  default="outputs/model_comparison",
                   help="输出根目录")
    p.add_argument("--models",   nargs="+", default=None,
                   help="模型规格列表，格式: Name:config_path:checkpoint_path")
    p.add_argument("--model-cfg", default=None,
                   help="YAML模型配置文件路径（与 --models 二选一）")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--min-pixels",   type=int,   default=10)
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--enable-plots",       action="store_true", default=False)
    p.add_argument("--enable-gt",          action="store_true", default=False)
    p.add_argument("--enable-poly-metrics", action="store_true", default=False)
    p.add_argument("--scale-ratio", type=float, default=None)
    p.add_argument("--scale-unit",  default=None)
    args = p.parse_args(argv)

    # 环境变量设置
    if args.enable_plots:
        os.environ["BL_GEOM_PLOTS"] = "1"
    if args.enable_gt:
        os.environ["BL_GEOM_GT"] = "1"
        os.environ["BL_GEOM_GT_MATCH"] = "1"
    if args.enable_poly_metrics:
        os.environ["BL_GEOM_POLY_METRICS"] = "1"

    # 解析模型列表
    model_specs: list[tuple[str, str, str]] = []
    if args.model_cfg:
        model_specs = _load_model_list_yaml(args.model_cfg)
    elif args.models:
        for spec in args.models:
            model_specs.append(_parse_model_spec(spec))
    else:
        # 默认：自动发现 work_dirs 下的模型
        model_specs = _autodiscover_models()
        if not model_specs:
            p.error("需要指定 --models 或 --model-cfg，或在 work_dirs/ 下有可识别的模型")

    # 加载图像列表
    img_list = _get_test_images(args.ann_file, args.img_dir)
    if not img_list:
        print(f"❌ 在 {args.img_dir} 中未找到图像，请检查路径。")
        return 1
    print(f"共 {len(img_list)} 张测试图像，{len(model_specs)} 个模型")

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows: list[dict] = []

    for model_name, config, checkpoint in model_specs:
        try:
            rows = evaluate_model(
                model_name,
                config,
                checkpoint,
                img_list,
                args.ann_file,
                args.out_dir,
                score_thresh=args.score_thresh,
                min_pixel_count=args.min_pixels,
                scale_ratio=args.scale_ratio,
                scale_unit=args.scale_unit,
                enable_plots=args.enable_plots,
                enable_gt=args.enable_gt,
                enable_polygon_metrics=args.enable_poly_metrics,
                device=args.device,
            )
            all_rows.extend(rows)
        except Exception:
            traceback.print_exc()
            print(f"❌ 模型 {model_name} 评估失败，跳过")

    if not all_rows:
        print("❌ 所有模型均失败，无输出。")
        return 1

    # 写入汇总CSV
    comparison_csv = os.path.join(args.out_dir, "comparison_summary.csv")
    write_comparison_csv(all_rows, comparison_csv)
    print(f"\n✅ 全量对比CSV: {comparison_csv}")

    mean_csv = os.path.join(args.out_dir, "comparison_mean.csv")
    print("\n📊 各模型均值:")
    write_mean_comparison_csv(all_rows, mean_csv)
    print(f"✅ 均值对比CSV: {mean_csv}")

    plot_comparison(mean_csv, args.out_dir)

    return 0


def _autodiscover_models() -> list[tuple[str, str, str]]:
    """自动扫描 work_dirs/ 下的已知模型目录。"""
    work_dirs = os.path.join(_REPO_ROOT, "work_dirs")
    if not os.path.isdir(work_dirs):
        return []
    result = []
    for entry in sorted(os.listdir(work_dirs)):
        d = os.path.join(work_dirs, entry)
        if not os.path.isdir(d):
            continue
        # 找config.py
        cfgs = [f for f in os.listdir(d) if f.endswith(".py")]
        # 找checkpoint
        ckpts = sorted(
            [f for f in os.listdir(d) if f.endswith(".pth")],
            key=lambda f: os.path.getmtime(os.path.join(d, f)),
            reverse=True,
        )
        if cfgs and ckpts:
            # 用目录名作为模型名
            name = entry
            config = os.path.join(d, cfgs[0])
            ckpt = os.path.join(d, ckpts[0])
            result.append((name, config, ckpt))
            print(f"  自动发现模型: {name}  ckpt={ckpts[0]}")
    return result


if __name__ == "__main__":
    sys.exit(main())
