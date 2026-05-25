from __future__ import annotations

import argparse
import csv
import os
import traceback

import numpy as np
from PIL import Image

from postprocess.analyze_main_dy2 import analyze_domain_geometry
from postprocess.coco_utils import load_isat_instances


def _set_bool_env(name: str, enabled: bool) -> None:
    os.environ[name] = "1" if enabled else "0"


def _configure_analysis_env(
    *,
    enable_plots: bool,
    enable_gt: bool,
    enable_polygon_metrics: bool,
    enable_save_images: bool,
    boundary_margin: int,
) -> None:
    _set_bool_env("BL_GEOM_PLOTS", enable_plots)
    _set_bool_env("BL_GEOM_GT", enable_gt)
    _set_bool_env("BL_GEOM_GT_MATCH", enable_gt)
    _set_bool_env("BL_GEOM_POLY_METRICS", enable_polygon_metrics)
    _set_bool_env("BL_GEOM_SAVE_IMAGES", enable_save_images)
    os.environ["BL_GEOM_BOUNDARY_MARGIN"] = str(int(boundary_margin))


def _build_overlay(pil_img: Image.Image, instances: list[dict]) -> Image.Image:
    import random

    width, height = pil_img.size
    base = pil_img.convert("RGBA")
    color_mask = np.zeros((height, width, 4), dtype=np.uint8)

    for inst in instances:
        coords = inst.get("coords")
        if coords is None or len(coords) == 0:
            continue
        inst_id = int(inst.get("id", 1))
        random.seed(inst_id)
        r, g, b = [random.randint(50, 255) for _ in range(3)]
        ys = coords[:, 0].astype(np.int64)
        xs = coords[:, 1].astype(np.int64)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        color_mask[ys[valid], xs[valid]] = [r, g, b, 150]

    overlay_img = Image.fromarray(color_mask, mode="RGBA")
    return Image.alpha_composite(base, overlay_img)


def _pick_metric(values, idx: int = 1) -> float:
    if values is None or len(values) == 0:
        return float("nan")
    if len(values) > idx:
        return float(values[idx])
    return float(values[0])


def _iter_images(image_dir: str) -> list[str]:
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = []
    for name in sorted(os.listdir(image_dir)):
        path = os.path.join(image_dir, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() in valid_exts:
            image_paths.append(path)
    return image_paths


def process_one_sample(
    image_path: str,
    gt_json_path: str,
    pred_json_path: str,
    out_dir: str,
    *,
    enable_plots: bool,
    enable_gt: bool,
    enable_polygon_metrics: bool,
    enable_save_images: bool,
    scale_ratio: float | None,
    scale_unit: str | None,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    image_name = os.path.basename(image_path)

    pred_instances, _, _ = load_isat_instances(
        pred_json_path,
        exclude_categories=["__background__"],
        rasterize=True,
    )

    pil_img = Image.open(image_path).convert("RGB")
    overlayed = _build_overlay(pil_img, pred_instances)

    ious, precisions, recalls, f1s, pred_count, gt_count, pred_cov, gt_cov = analyze_domain_geometry(
        image_path,
        pred_instances,
        overlayed,
        out_dir,
        gt_json_path=gt_json_path,
        scale_ratio=scale_ratio,
        scale_unit=scale_unit,
        enable_plots=enable_plots,
        enable_gt=enable_gt,
        enable_gt_matching=enable_gt,
        enable_save_images=enable_save_images,
        enable_polygon_metrics=enable_polygon_metrics,
    )

    return {
        "image": image_name,
        "gt_json": os.path.basename(gt_json_path),
        "pred_json": os.path.basename(pred_json_path),
        "iou": _pick_metric(ious),
        "precision": _pick_metric(precisions),
        "recall": _pick_metric(recalls),
        "f1": _pick_metric(f1s),
        "pred_count": int(pred_count) if pred_count is not None else float("nan"),
        "gt_count": int(gt_count) if gt_count is not None else float("nan"),
        "pred_coverage": float(pred_cov) if pred_cov is not None else float("nan"),
        "gt_coverage": float(gt_cov) if gt_cov is not None else float("nan"),
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Use existing postprocess pipeline to analyze SAM3 ISAT predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sam3-dir", default="work_dirs/sam3", help="SAM3结果根目录")
    parser.add_argument("--image-dir", default=None, help="原图目录，默认使用 <sam3-dir>/image")
    parser.add_argument("--label-dir", default=None, help="GT JSON目录，默认使用 <sam3-dir>/label")
    parser.add_argument("--pred-dir", default=None, help="预测 JSON目录，默认使用 <sam3-dir>/pred_isat")
    parser.add_argument("--out-dir", default=None, help="输出目录，默认使用 <sam3-dir>/analysis")
    parser.add_argument("--image", default=None, help="仅处理单张图，可传文件名或 stem，例如 50x-1 或 50x-1.png")
    parser.add_argument("--enable-plots", dest="enable_plots", action="store_true", help="启用 GT vs Pred 统计图")
    parser.add_argument("--disable-plots", dest="enable_plots", action="store_false", help="关闭 GT vs Pred 统计图")
    parser.add_argument("--disable-gt", action="store_true", default=False, help="跳过 GT 几何分析")
    parser.add_argument("--disable-poly-metrics", action="store_true", default=False, help="跳过 IoU/Precision/Recall/F1")
    parser.add_argument("--disable-save-images", action="store_true", default=False, help="跳过中间可视化图保存")
    parser.add_argument("--boundary-margin", type=int, default=0, help="边界剔除像素数，设为 0 可最大化保留参考图中的匹配对")
    parser.add_argument("--scale-ratio", type=float, default=None, help="像素到物理单位的比例尺")
    parser.add_argument("--scale-unit", default=None, help="物理单位名，例如 nm")
    parser.set_defaults(enable_plots=True)
    args = parser.parse_args(argv)

    image_dir = args.image_dir or os.path.join(args.sam3_dir, "image")
    label_dir = args.label_dir or os.path.join(args.sam3_dir, "label")
    pred_dir = args.pred_dir or os.path.join(args.sam3_dir, "pred_isat")
    out_dir = args.out_dir or os.path.join(args.sam3_dir, "analysis")

    for required_dir in (image_dir, label_dir, pred_dir):
        if not os.path.isdir(required_dir):
            parser.error(f"目录不存在: {required_dir}")

    enable_gt = not args.disable_gt
    enable_polygon_metrics = not args.disable_poly_metrics
    enable_save_images = not args.disable_save_images

    _configure_analysis_env(
        enable_plots=bool(args.enable_plots),
        enable_gt=enable_gt,
        enable_polygon_metrics=enable_polygon_metrics,
        enable_save_images=enable_save_images,
        boundary_margin=args.boundary_margin,
    )

    image_paths = _iter_images(image_dir)
    if args.image:
        target = os.path.splitext(os.path.basename(args.image))[0]
        image_paths = [
            path for path in image_paths
            if os.path.splitext(os.path.basename(path))[0] == target
        ]

    if not image_paths:
        print("❌ 没有找到待处理图像。")
        return 1

    os.makedirs(out_dir, exist_ok=True)
    rows: list[dict] = []

    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        stem = os.path.splitext(image_name)[0]
        gt_json_path = os.path.join(label_dir, f"{stem}.json")
        pred_json_path = os.path.join(pred_dir, f"{stem}.json")
        sample_out_dir = os.path.join(out_dir, stem)

        if not os.path.exists(gt_json_path) or not os.path.exists(pred_json_path):
            rows.append({
                "image": image_name,
                "gt_json": os.path.basename(gt_json_path),
                "pred_json": os.path.basename(pred_json_path),
                "iou": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "pred_count": float("nan"),
                "gt_count": float("nan"),
                "pred_coverage": float("nan"),
                "gt_coverage": float("nan"),
                "status": "missing_json",
            })
            print(f"⚠️ 缺少配套 JSON，跳过: {image_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Processing: {image_name}")
        try:
            row = process_one_sample(
                image_path,
                gt_json_path,
                pred_json_path,
                sample_out_dir,
                enable_plots=bool(args.enable_plots),
                enable_gt=enable_gt,
                enable_polygon_metrics=enable_polygon_metrics,
                enable_save_images=enable_save_images,
                scale_ratio=args.scale_ratio,
                scale_unit=args.scale_unit,
            )
            print(
                f"  ✅ {image_name}: pred={row['pred_count']}, "
                f"iou={row['iou']:.4f}, f1={row['f1']:.4f}"
            )
        except Exception:
            traceback.print_exc()
            os.makedirs(sample_out_dir, exist_ok=True)
            with open(os.path.join(sample_out_dir, "error.log"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            row = {
                "image": image_name,
                "gt_json": os.path.basename(gt_json_path),
                "pred_json": os.path.basename(pred_json_path),
                "iou": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "pred_count": float("nan"),
                "gt_count": float("nan"),
                "pred_coverage": float("nan"),
                "gt_coverage": float("nan"),
                "status": "error",
            }
        rows.append(row)

    summary_csv = os.path.join(out_dir, "metrics_summary.csv")
    fieldnames = [
        "image",
        "gt_json",
        "pred_json",
        "iou",
        "precision",
        "recall",
        "f1",
        "pred_count",
        "gt_count",
        "pred_coverage",
        "gt_coverage",
        "status",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    print(f"\n汇总已保存到: {summary_csv}")
    if ok_rows:
        mean_iou = np.nanmean([row["iou"] for row in ok_rows])
        mean_f1 = np.nanmean([row["f1"] for row in ok_rows])
        print(f"平均 IoU: {mean_iou:.4f}")
        print(f"平均 F1 : {mean_f1:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())