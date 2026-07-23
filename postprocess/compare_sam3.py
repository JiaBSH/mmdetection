"""SAM3 sliding-window evaluation through the existing postprocess pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from postprocess._coco_eval import COCOResultCollector, evaluate_coco_from_predictions  # noqa: E402
from postprocess.compare_models import (  # noqa: E402
    _build_image_id_map,
    plot_comparison,
    write_comparison_csv,
    write_geometry_summaries,
    write_mean_comparison_csv,
)
from postprocess.run_postprocess import _get_test_images, process_one_image  # noqa: E402
from postprocess.sam3_sliding_window_infer import Sam3SlidingModel  # noqa: E402


def _set_bool_env(name: str, enabled: bool) -> None:
    os.environ[name] = "1" if enabled else "0"


def _configure_env(args: argparse.Namespace) -> None:
    _set_bool_env("BL_GEOM_PLOTS", args.enable_plots)
    _set_bool_env("BL_GEOM_GT", args.enable_gt)
    _set_bool_env("BL_GEOM_GT_MATCH", args.enable_gt_matching)
    _set_bool_env("BL_GEOM_POLY_METRICS", args.enable_poly_metrics)
    _set_bool_env("BL_GEOM_SAVE_IMAGES", args.enable_save_images)
    if args.geom_workers is not None and args.geom_workers > 0:
        os.environ["BL_GEOM_WORKERS"] = str(args.geom_workers)
    if args.scatter_metric is not None:
        os.environ["BL_GEOM_SCATTER_METRIC"] = str(args.scatter_metric)


def _write_model_metrics(rows: list[dict], model_out_dir: str) -> None:
    model_csv = os.path.join(model_out_dir, "metrics_summary.csv")
    fieldnames = [
        "model",
        "image",
        "iou",
        "precision",
        "recall",
        "f1",
        "pred_count",
        "gt_count",
        "pred_coverage",
        "gt_coverage",
    ]
    with open(model_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ SAM3 metrics CSV: {model_csv}")


def evaluate_sam3(
    args: argparse.Namespace,
    img_list: list[tuple[str, str]],
) -> tuple[list[dict], dict[str, float]]:
    model_out_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    print(f"\n{'#' * 60}")
    print(f"# 模型: {args.model_name}")
    print(f"# SAM3 root:   {args.sam3_root}")
    print(f"# Checkpoint:  {args.checkpoint}")
    print(f"# Text prompt: {args.prompt}")
    print(f"{'#' * 60}")

    model = Sam3SlidingModel(
        sam3_root=args.sam3_root,
        checkpoint_path=args.checkpoint,
        text_prompt=args.prompt,
        device=args.device,
        resolution=args.sam3_resolution,
    )

    image_id_map = _build_image_id_map(args.ann_file)
    evaluated_image_ids = sorted({
        image_id_map[img_name]
        for _, img_name in img_list
        if img_name in image_id_map
    })
    coco_collector = COCOResultCollector()
    rows: list[dict] = []

    for img_path, img_name in img_list:
        stem = os.path.splitext(img_name)[0]
        out_dir_i = os.path.join(model_out_dir, stem)
        img_id = image_id_map.get(img_name)
        try:
            row = process_one_image(
                model,
                img_path,
                args.ann_file,
                out_dir_i,
                score_thresh=args.score_thresh,
                target_label=0,
                min_pixel_count=args.min_pixels,
                scale_ratio=args.scale_ratio,
                scale_unit=args.scale_unit,
                enable_plots=args.enable_plots,
                enable_gt=args.enable_gt,
                enable_polygon_metrics=args.enable_poly_metrics,
                device=args.device,
                sliding_window=args.sliding_window,
                patch_size=args.patch_size,
                patch_overlap_ratio=args.patch_overlap_ratio,
                batch_size=args.batch_size,
                verbose=True,
                coco_collector=coco_collector,
                image_id=img_id,
            )
        except Exception:
            traceback.print_exc()
            row = {
                "image": img_name,
                "iou": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "pred_count": float("nan"),
                "gt_count": float("nan"),
                "pred_coverage": float("nan"),
                "gt_coverage": float("nan"),
            }
        row["model"] = args.model_name
        rows.append(row)

    _write_model_metrics(rows, model_out_dir)

    valid = [r for r in rows if np.isfinite(r.get("iou", float("nan")))]
    if valid:
        mean_iou = float(np.mean([r["iou"] for r in valid]))
        mean_f1 = float(np.mean([r["f1"] for r in valid]))
        print(f"\n📊 {args.model_name}: 平均IoU={mean_iou:.4f}  平均F1={mean_f1:.4f}  ({len(valid)}/{len(rows)}张)")
    else:
        print(f"\n⚠️ {args.model_name}: 无有效指标")

    coco_metrics: dict[str, float] = {}
    if evaluated_image_ids:
        try:
            print(
                f"\n📐 COCO 评估: {args.model_name} "
                f"({len(evaluated_image_ids)} images, "
                f"{len(coco_collector)} predictions)"
            )
            coco_metrics = evaluate_coco_from_predictions(
                coco_collector.to_coco_list(),
                args.ann_file,
                metrics=["bbox", "segm"],
                image_ids=evaluated_image_ids,
                max_dets=args.coco_max_dets,
            )
            coco_json_path = os.path.join(model_out_dir, "coco_metrics.json")
            with open(coco_json_path, "w", encoding="utf-8") as f:
                json.dump(coco_metrics, f, ensure_ascii=False, indent=2)
            print(f"  ✅ COCO metrics: {coco_json_path}")
            for key in sorted(coco_metrics):
                print(f"     {key}: {coco_metrics[key]:.4f}")
        except Exception:
            print(f"  ⚠️ COCO 评估失败: {traceback.format_exc()}")

    return rows, coco_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM3 postprocess comparison with the same output contract as compare_models.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-dir", default="outputs/sam3_comparison")
    parser.add_argument("--model-name", default="SAM3")
    parser.add_argument("--sam3-root", default="/data/home/scvi576/run/JiaBSH/nano_sam3")
    parser.add_argument(
        "--checkpoint",
        default="/data/home/scvi576/run/JiaBSH/nano_sam3/ms_cache/facebook/sam3/sam3.pt",
    )
    parser.add_argument("--prompt", default="Hexagon")
    parser.add_argument("--sam3-resolution", type=int, default=1008)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--min-pixels", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sliding-window", action="store_true", default=False)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--patch-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--coco-max-dets", type=int, default=10000)
    parser.add_argument("--enable-plots", action="store_true", default=False)
    parser.add_argument("--enable-gt", action="store_true", default=False)
    parser.add_argument("--enable-gt-matching", action="store_true", default=False)
    parser.add_argument("--enable-save-images", action="store_true", default=False)
    parser.add_argument("--enable-poly-metrics", action="store_true", default=False)
    parser.add_argument("--geom-workers", type=int, default=None)
    parser.add_argument("--scatter-metric", default=None, choices=["mae", "r2", "both"])
    parser.add_argument("--scale-ratio", type=float, default=None)
    parser.add_argument("--scale-unit", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _configure_env(args)

    img_list = _get_test_images(args.ann_file, args.img_dir)
    if not img_list:
        print(f"❌ 在 {args.img_dir} 中未找到图像，请检查路径。")
        return 1
    print(f"共 {len(img_list)} 张测试图像，1 个模型")

    os.makedirs(args.out_dir, exist_ok=True)
    rows, coco_metrics = evaluate_sam3(args, img_list)
    if not rows:
        print("❌ SAM3 评估失败，无输出。")
        return 1

    comparison_csv = os.path.join(args.out_dir, "comparison_summary.csv")
    write_comparison_csv(rows, comparison_csv)
    print(f"\n✅ 全量对比CSV: {comparison_csv}")

    mean_csv = os.path.join(args.out_dir, "comparison_mean.csv")
    print("\n📊 各模型均值:")
    write_mean_comparison_csv(
        rows,
        mean_csv,
        coco_metrics={args.model_name: coco_metrics} if coco_metrics else {},
    )
    print(f"✅ 均值对比CSV: {mean_csv}")

    if os.getenv("BL_COMPARE_SAVE_PLOTS", "1").strip().lower() not in {"0", "false", "no", "off"}:
        plot_comparison(mean_csv, args.out_dir)
    else:
        print("BL_COMPARE_SAVE_PLOTS=0: skip comparison_bar.png")

    try:
        write_geometry_summaries(args.out_dir)
    except Exception:
        traceback.print_exc()
        print(f"⚠️ 几何汇总生成失败: {args.out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
