"""单模型单图像后处理评估脚本

用法示例（从命令行）:

  # 对单张测试图推理 + 后处理几何分析
  python postprocess/run_postprocess.py \
      --config  work_dirs/custom_maskrcnn_1cls_1024_aug/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
      --checkpoint work_dirs/custom_maskrcnn_1cls_1024_aug/epoch_50.pth \
      --img      dataset_root/dataset_1024_aug/images/test/xxx.png \
      --ann-file dataset_root/dataset_1024_aug/annotations/instances_test.json \
      --out-dir  outputs/maskrcnn_postprocess/xxx \
      --score-thresh 0.5

  # 对整个test集批量处理（遍历 ann-file 中所有图像）
  python postprocess/run_postprocess.py \
      --config  work_dirs/custom_maskrcnn_1cls_1024_aug/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
      --checkpoint work_dirs/custom_maskrcnn_1cls_1024_aug/epoch_50.pth \
      --ann-file dataset_root/dataset_1024_aug/annotations/instances_test.json \
      --img-dir  dataset_root/dataset_1024_aug/images/test \
      --out-dir  outputs/maskrcnn_postprocess \
      --score-thresh 0.5

流程
----
1. 用 MMDetection inference API 对图像推理，得到实例masks
2. 通过 coco_utils.mmdet_masks_to_instances() 转为 global_instances 格式
3. 通过 coco_utils.load_coco_gt_polygons() 读取COCO GT，并转换到当前后处理流程使用的格式
4. 调用 postprocess.analyze_main_dy2.analyze_domain_geometry() 做几何分析
5. 汇总每张图的指标到 metrics_summary.csv

几何分析开关（与 temp 管道一致，通过环境变量控制）:
  BL_GEOM_PLOTS=1          生成GT vs Pred直方图/R^2散点图
  BL_GEOM_GT=1             分析GT几何
  BL_GEOM_GT_MATCH=1       GT↔Pred匹配
  BL_GEOM_SAVE_IMAGES=1    保存中间可视化图像
  BL_GEOM_POLY_METRICS=1   IoU/Precision/Recall/F1评估
  BL_GEOM_BOUNDARY_MARGIN=5  边界剔除像素数
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback

import numpy as np
from PIL import Image

from postprocess.coco_utils import (  # noqa: E402
    load_coco_gt_polygons,
    mmdet_masks_to_instances,
)
from postprocess.analyze_main_dy2 import (  # noqa: E402
    analyze_domain_geometry as _analyze_domain_geometry,
)
from postprocess.sliding_window_infer import (  # noqa: E402
    infer_image_with_overlap_windows,
)
from postprocess._shared import _env_flag, _to_numpy, _build_overlay

# ---------------------------------------------------------------------------
# 推理单张图（MMDetection inference API）
# ---------------------------------------------------------------------------


def _pred_instances_to_global_instances(
    pred,
    *,
    score_thresh: float = 0.5,
    target_label: int = 0,
    min_pixel_count: int = 10,
) -> list[dict]:
    """将 MMDet ``pred_instances`` 转为当前后处理使用的 global_instances。"""
    masks = getattr(pred, "masks", None)
    scores = getattr(pred, "scores", None)
    labels = getattr(pred, "labels", None)
    bboxes = getattr(pred, "bboxes", None)

    if masks is None:
        return []

    if hasattr(masks, "to_ndarray"):
        masks_np = masks.to_ndarray().astype(bool)
    elif hasattr(masks, "numpy"):
        if hasattr(masks, "cpu"):
            masks = masks.cpu()
        masks_np = masks.numpy().astype(bool)
    else:
        masks_np = np.asarray(masks, dtype=bool)

    return mmdet_masks_to_instances(
        masks_np,
        scores=_to_numpy(scores),
        labels=_to_numpy(labels),
        bboxes=_to_numpy(bboxes),
        score_thresh=score_thresh,
        target_label=target_label,
        min_pixel_count=min_pixel_count,
    )

def _infer_one_image(
    model,
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
    """推理单张图，返回 (global_instances, PIL_image_RGB)。"""
    from mmdet.apis import inference_detector  # type: ignore
    import numpy as np
    pil_img = Image.open(img_path).convert("RGB")

    if not sliding_window:
        # Pass as numpy array to use LoadImageFromNDArray path consistently
        # (avoids pipeline type mismatch between file-path and array inference)
        img_np = np.asarray(pil_img)
        result = inference_detector(model, img_np)
        instances = _pred_instances_to_global_instances(
            result.pred_instances,
            score_thresh=score_thresh,
            target_label=target_label,
            min_pixel_count=min_pixel_count,
        )
        return instances, pil_img, [], []

    instances, pil_img, windows, merge_records = infer_image_with_overlap_windows(
        model,
        img_path,
        score_thresh=score_thresh,
        target_label=target_label,
        min_pixel_count=min_pixel_count,
        patch_size=patch_size,
        patch_overlap_ratio=patch_overlap_ratio,
        batch_size=batch_size,
    )
    return instances, pil_img, windows, merge_records


# ---------------------------------------------------------------------------
# 构建彩色实例overlay（与 temp 管道一致）
# ---------------------------------------------------------------------------

def _save_sliding_window_visualization(
    overlayed: Image.Image,
    out_dir: str,
    windows: list[dict],
    instances: list[dict],
    merge_records: list[dict],
) -> None:
    if not windows:
        return

    from PIL import ImageDraw, ImageFont

    width, height = overlayed.size
    windows_vis = overlayed.convert("RGBA")
    vis_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(vis_overlay)

    def draw_window_rect(
        window: dict,
        prefix: str,
        color: tuple[int, int, int, int],
        line_width: int,
    ) -> None:
        left_key = f"{prefix}_left"
        top_key = f"{prefix}_top"
        right_key = f"{prefix}_right"
        bottom_key = f"{prefix}_bottom"
        if not all(key in window for key in (left_key, top_key, right_key, bottom_key)):
            return
        draw.rectangle(
            [
                int(round(float(window[left_key]))),
                int(round(float(window[top_key]))),
                int(round(float(window[right_key]))),
                int(round(float(window[bottom_key]))),
            ],
            outline=color,
            width=line_width,
        )

    for window in windows:
        draw.rectangle(
            [
                int(window["left"]),
                int(window["top"]),
                int(window["right"]),
                int(window["bottom"]),
            ],
            outline=(30, 144, 255, 200),
            width=3,
        )
        draw_window_rect(window, "core", (255, 215, 0, 210), 2)
        draw_window_rect(window, "safe", (255, 0, 255, 220), 2)

    windows_vis = Image.alpha_composite(windows_vis, vis_overlay)

    record_mask = np.zeros((height, width, 4), dtype=np.uint8)
    for record in merge_records:
        removed_coords = record.get("removed_coords")
        if isinstance(removed_coords, np.ndarray) and removed_coords.ndim == 1:
            ys = (removed_coords // width).astype(np.int64)
            xs = (removed_coords % width).astype(np.int64)
            valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
            if np.any(valid):
                record_mask[ys[valid], xs[valid]] = (255, 0, 0, 150)

        overlap = record.get("overlap")
        if isinstance(overlap, np.ndarray) and overlap.ndim == 1 and overlap.size > 0:
            ys = (overlap // width).astype(np.int64)
            xs = (overlap % width).astype(np.int64)
            valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
            if np.any(valid):
                record_mask[ys[valid], xs[valid]] = (0, 255, 0, 180)

    windows_vis = Image.alpha_composite(
        windows_vis,
        Image.fromarray(record_mask, mode="RGBA"),
    )

    draw_labels = ImageDraw.Draw(windows_vis)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    for inst in instances:
        coords = inst.get("coords")
        if coords is None or len(coords) == 0:
            continue
        ys = coords[:, 0].astype(np.int64)
        xs = coords[:, 1].astype(np.int64)
        centroid_x = int(xs.mean())
        centroid_y = int(ys.mean())
        '''
        draw_labels.text(
            (centroid_x, centroid_y),
            f"#{inst['id']}",
            fill=(0, 255, 0, 255),
            font=font,
        )
        '''

    removed_ids_drawn: set[int | None] = set()
    for record in merge_records:
        removed_id = record.get("removed")
        if removed_id is None or removed_id in removed_ids_drawn:
            continue
        removed_ids_drawn.add(removed_id)
        removed_coords = record.get("removed_coords")
        if not isinstance(removed_coords, np.ndarray) or removed_coords.ndim != 1:
            continue
        if removed_coords.size == 0:
            continue
        ys = (removed_coords // width).astype(np.int64)
        xs = (removed_coords % width).astype(np.int64)
        centroid_x = int(xs.mean())
        centroid_y = int(ys.mean())
        '''
        draw_labels.text(
            (centroid_x, centroid_y),
            f"-{removed_id}",
            fill=(255, 0, 0, 255),
            font=font,
        )
        '''

    windows_vis.save(os.path.join(out_dir, "overlap_windows_visualization.png"))


# ---------------------------------------------------------------------------
# 核心：单张图后处理
# ---------------------------------------------------------------------------

def process_one_image(
    model,
    img_path: str,
    ann_file: str,
    out_dir: str,
    *,
    score_thresh: float = 0.5,
    target_label: int = 0,
    min_pixel_count: int = 10,
    scale_ratio: float | None = None,
    scale_unit: str | None = None,
    enable_plots: bool | None = None,
    enable_gt: bool | None = None,
    enable_polygon_metrics: bool | None = None,
    device: str = "cuda:0",
    sliding_window: bool = False,
    patch_size: int = 1024,
    patch_overlap_ratio: float = 0.0,
    batch_size: int = 1,
    verbose: bool = True,
    coco_collector=None,    # COCOResultCollector | None
    image_id: int | None = None,
) -> dict:
    """对单张图推理并做几何分析，返回指标dict。

    Parameters
    ----------
    coco_collector : COCOResultCollector | None
        若提供，推理后将预测结果以 COCO 格式收集到该对象中。
    image_id : int | None
        COCO annotation 中的 image_id（与 coco_collector 配合使用）。

    Returns
    -------
    dict with keys: image, iou, precision, recall, f1,
                    pred_count, gt_count, pred_coverage, gt_coverage
    """
    # 先创建输出目录，确保后续所有异常都能写 error.log
    os.makedirs(out_dir, exist_ok=True)
    img_name = os.path.basename(img_path)
    error_log = os.path.join(out_dir, "error.log")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")

    windows: list[dict] = []
    merge_records: list[dict] = []

    # 1) 推理
    try:
        instances, pil_img, windows, merge_records = _infer_one_image(
            model,
            img_path,
            score_thresh=score_thresh,
            target_label=target_label,
            min_pixel_count=min_pixel_count,
            device=device,
            sliding_window=sliding_window,
            patch_size=patch_size,
            patch_overlap_ratio=patch_overlap_ratio,
            batch_size=batch_size,
        )
    except Exception:
        tb = traceback.format_exc()
        print(f"  ❌ 推理失败: {img_name}\n{tb}")
        with open(error_log, "w", encoding="utf-8") as _f:
            _f.write(f"推理失败:\n{tb}")
        return {
            "image": img_name,
            "iou": float("nan"), "precision": float("nan"),
            "recall": float("nan"), "f1": float("nan"),
            "pred_count": float("nan"), "gt_count": float("nan"),
            "pred_coverage": float("nan"), "gt_coverage": float("nan"),
        }

    if verbose:
        print(f"  Predicted instances: {len(instances)}")

    # 1.5) COCO 预测收集
    if coco_collector is not None and image_id is not None:
        try:
            coco_collector.add_image_predictions(
                image_id=image_id,
                global_instances=instances,
                img_width=pil_img.width,
                img_height=pil_img.height,
            )
        except Exception:
            pass  # COCO 收集失败不影响主流程

    # 2) 构建overlay
    mask_alpha = int(os.getenv("BL_MASK_ALPHA", "160"))
    overlayed = _build_overlay(pil_img, instances, mask_alpha=mask_alpha)
    if sliding_window and _env_flag("BL_GEOM_SAVE_IMAGES", True):
        try:
            _save_sliding_window_visualization(
                overlayed,
                out_dir,
                windows,
                instances,
                merge_records,
            )
        except Exception:
            tb = traceback.format_exc()
            print(f"  ⚠️ 滑窗可视化保存失败: {img_name}\n{tb}")
            with open(error_log, "a", encoding="utf-8") as _f:
                _f.write(f"滑窗可视化保存失败:\n{tb}\n")

    # 3) 几何分析
    try:
        ious, precisions, recalls, f1s, pred_count, gt_count, pred_cov, gt_cov = \
            analyze_domain_geometry_coco(
                img_path,
                instances,
                overlayed,
                out_dir,
                gt_coco_ann_file=ann_file,
                gt_image_filename=img_name,
                scale_ratio=scale_ratio,
                scale_unit=scale_unit,
                enable_plots=enable_plots,
                enable_gt=enable_gt,
                enable_polygon_metrics=enable_polygon_metrics,
            )
    except Exception:
        tb = traceback.format_exc()
        print(f"  ❌ 几何分析失败: {img_name}\n{tb}")
        with open(error_log, "a", encoding="utf-8") as _f:
            _f.write(f"几何分析失败:\n{tb}")
        ious, precisions, recalls, f1s = [], [], [], []
        pred_count, gt_count = len(instances), 0
        pred_cov, gt_cov = 0.0, 0.0

    def _pick(arr, idx=1):
        if arr is None or len(arr) == 0:
            return float("nan")
        if len(arr) > idx:
            return float(arr[idx])
        return float(arr[0])

    return {
        "image": img_name,
        "iou":        _pick(ious),
        "precision":  _pick(precisions),
        "recall":     _pick(recalls),
        "f1":         _pick(f1s),
        "pred_count": int(pred_count) if pred_count is not None else float("nan"),
        "gt_count":   int(gt_count)   if gt_count   is not None else float("nan"),
        "pred_coverage": float(pred_cov) if pred_cov is not None else float("nan"),
        "gt_coverage":   float(gt_cov)   if gt_cov   is not None else float("nan"),
    }


# ---------------------------------------------------------------------------
# COCO适配wrapper：对 analyze_domain_geometry 做GT格式桥接
# ---------------------------------------------------------------------------

def analyze_domain_geometry_coco(
    orig_image: str,
    global_instances: list[dict],
    overlayed: Image.Image,
    save_dir: str,
    *,
    gt_coco_ann_file: str | None = None,
    gt_image_filename: str | None = None,
    scale_ratio: float | None = None,
    scale_unit: str | None = None,
    enable_plots: bool | None = None,
    enable_gt: bool | None = None,
    enable_polygon_metrics: bool | None = None,
):
    """analyze_domain_geometry 的COCO格式桥接包装器。

    由于本地 analyze_main_dy2.analyze_domain_geometry 内部用ISAT JSON读GT，
    这里先将COCO GT转为临时ISAT兼容JSON，再调用原函数。
    这样避免改变现有几何分析主逻辑，同时所有实现都保留在 postprocess/ 内。

    ISAT兼容格式（最小子集）：
      { "objects": [ { "segmentation": [[x,y],...], "category": "畴区" }, ... ] }
    """
    # 将COCO GT多边形转为临时ISAT JSON
    tmp_isat_path: str | None = None
    if gt_coco_ann_file is not None and os.path.exists(gt_coco_ann_file):
        try:
            gt_polygons, W_gt, H_gt = load_coco_gt_polygons(
                gt_coco_ann_file,
                image_filename=gt_image_filename,
            )
            # 构建最小ISAT格式: segmentation为 [[x,y],...] 即 (col,row)
            isat_objects = []
            for poly_rc in gt_polygons:
                # poly_rc shape (N,2): (row, col) → 转为 [[col, row], ...]
                seg = [[float(p[1]), float(p[0])] for p in poly_rc]
                isat_objects.append({
                    "category": "畴区",
                    "segmentation": seg,
                })
            tmp_isat = {"objects": isat_objects}
            tmp_isat_path = os.path.join(save_dir, "_tmp_gt_isat.json")
            with open(tmp_isat_path, "w", encoding="utf-8") as f:
                json.dump(tmp_isat, f, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ GT转换失败，跳过GT分析: {e}")
            tmp_isat_path = None

    result = _analyze_domain_geometry(
        orig_image,
        global_instances,
        overlayed,
        save_dir,
        gt_json_path=tmp_isat_path,
        scale_ratio=scale_ratio,
        scale_unit=scale_unit,
        enable_plots=enable_plots,
        enable_gt=enable_gt,
        enable_polygon_metrics=enable_polygon_metrics,
    )

    # 清理临时文件
    if tmp_isat_path is not None and os.path.exists(tmp_isat_path):
        try:
            os.remove(tmp_isat_path)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _load_model(config: str, checkpoint: str, device: str = "cuda:0"):
    """加载MMDetection模型。"""
    import torch
    # PyTorch 2.6+ defaults to weights_only=True, breaking mmdet checkpoints
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
    from mmdet.apis import init_detector  # type: ignore
    return init_detector(config, checkpoint, device=device)


def _get_test_images(ann_file: str, img_dir: str) -> list[tuple[str, str]]:
    """从COCO ann_file + img_dir 返回 (img_path, filename) 列表。"""
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)
    result = []
    for img_info in coco.get("images", []):
        fn = img_info.get("file_name", "")
        # file_name 可能含子目录前缀
        basename = os.path.basename(fn)
        full_path = os.path.join(img_dir, basename)
        if not os.path.exists(full_path):
            # 尝试完整路径
            full_path = os.path.join(img_dir, fn)
        if os.path.exists(full_path):
            result.append((full_path, basename))
        else:
            print(f"⚠️ Image not found: {full_path}")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="MMDetection实例分割后处理几何评估（COCO格式）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",      required=True,  help="MMDetection config文件路径")
    p.add_argument("--checkpoint",  required=True,  help="模型checkpoint路径")
    p.add_argument("--ann-file",    required=True,  help="COCO格式标注文件（instances_test.json等）")
    p.add_argument("--img",         default=None,   help="单张图像路径（与 --img-dir 二选一）")
    p.add_argument("--img-dir",     default=None,   help="测试图像目录（批量处理）")
    p.add_argument("--out-dir",     default="outputs/postprocess", help="输出根目录")
    p.add_argument("--score-thresh",type=float, default=0.5,  help="预测置信度阈值")
    p.add_argument("--min-pixels",  type=int,   default=10,   help="实例最小像素数")
    p.add_argument("--device",      default="cuda:0")
    p.add_argument("--sliding-window", action="store_true", default=False,
                   help="启用滑窗推理；不启用时默认整图推理")
    p.add_argument("--patch-size", type=int, default=1024,
                   help="滑窗边长（像素）")
    p.add_argument("--patch-overlap-ratio", type=float, default=0.0,
                   help="相邻滑窗的交叠比例；0 表示非交叠，>0 表示交叠滑窗")
    p.add_argument("--batch-size", type=int, default=1,
                   help="滑窗推理 batch size")
    p.add_argument("--enable-plots",    action="store_true", default=False,
                   help="生成GT vs Pred直方图/R^2散点图（较慢）")
    p.add_argument("--enable-gt",       action="store_true", default=False,
                   help="分析GT几何（需GT标注文件）")
    p.add_argument("--enable-poly-metrics", action="store_true", default=False,
                   help="计算IoU/Precision/Recall/F1像素级指标")
    p.add_argument("--scale-ratio", type=float, default=None,
                   help="比例尺换算系数（像素→物理单位，如 nm/px）")
    p.add_argument("--scale-unit",  default=None,
                   help="物理单位名称（如 nm）")
    args = p.parse_args(argv)

    # 环境变量同步：显式设置 0/1，确保底层 _env_flag 读到正确的值
    os.environ["BL_GEOM_PLOTS"] = "1" if args.enable_plots else "0"
    os.environ["BL_GEOM_GT"] = "1" if args.enable_gt else "0"
    os.environ["BL_GEOM_GT_MATCH"] = "1" if args.enable_gt else "0"
    os.environ["BL_GEOM_POLY_METRICS"] = "1" if args.enable_poly_metrics else "0"

    # 加载模型
    print(f"Loading model: {args.config}")
    print(f"Checkpoint:    {args.checkpoint}")
    model = _load_model(args.config, args.checkpoint, device=args.device)

    # 构建图像列表
    if args.img is not None:
        img_list = [(args.img, os.path.basename(args.img))]
    elif args.img_dir is not None:
        img_list = _get_test_images(args.ann_file, args.img_dir)
    else:
        p.error("需要指定 --img 或 --img-dir")

    if not img_list:
        print("❌ 没有找到图像，退出。")
        return 1

    print(f"共 {len(img_list)} 张图像待处理")

    all_rows: list[dict] = []
    for img_path, img_name in img_list:
        stem = os.path.splitext(img_name)[0]
        out_dir_i = os.path.join(args.out_dir, stem)
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
        all_rows.append(row)
        print(f"  ✅ {img_name}: pred={row['pred_count']}, "
              f"iou={row['iou']:.4f}, f1={row['f1']:.4f}")

    # 保存汇总CSV
    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    fieldnames = ["image", "iou", "precision", "recall", "f1",
                  "pred_count", "gt_count", "pred_coverage", "gt_coverage"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅ 汇总结果已保存: {summary_csv}")

    # 打印均值
    valid = [r for r in all_rows if not np.isnan(r.get("iou", float("nan")))]
    if valid:
        mean_iou = np.mean([r["iou"] for r in valid])
        mean_f1  = np.mean([r["f1"]  for r in valid])
        print(f"   平均 IoU={mean_iou:.4f}  平均 F1={mean_f1:.4f}  (共{len(valid)}张有效图)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
