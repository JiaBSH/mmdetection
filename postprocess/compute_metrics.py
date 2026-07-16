"""像素级 IoU / Precision / Recall / F1 计算（Facade）。

保持向后兼容的 API，内部委托给 _pixel_metrics 的纯函数实现。
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from ._pixel_metrics import (
    build_gt_mask_from_json,
    build_iou_overlay,
    build_pred_mask_from_polygons,
    compute_pixel_metrics,
    save_metrics_bar_chart,
    save_metrics_text,
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = str(value).strip().lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return default


def Compute_metrics(
    method,
    orig_image,
    json_path,
    all_polygons,
    image_size,
    save_dir="./output",
    mode="union",
    *,
    save_visualization: bool = True,
    save_metrics: bool = True,
    save_bar: bool = True,
    verbose: bool = True,
):
    """计算 IoU / Precision / Recall / F1 并保存结果与可视化。

    :param method: 用于保存文件的标识
    :param orig_image: 原始图像路径 (PNG/JPG)
    :param json_path: ISAT 风格标注 json 文件
    :param all_polygons: 预测多边形顶点列表 (list of ndarray, (m,2) in row,col)
    :param image_size: (H, W)
    :param save_dir: 保存目录
    :param mode: "instance" 或 "union"
    :return: dict {"IoU": ..., "Precision": ..., "Recall": ..., "F1-score": ...}
    """
    os.makedirs(save_dir, exist_ok=True)
    save_visualization = save_visualization and _env_flag(
        "BL_METRICS_SAVE_VISUALIZATION",
        True,
    )
    save_metrics = save_metrics and _env_flag("BL_METRICS_SAVE_TEXT", True)
    save_bar = save_bar and _env_flag("BL_METRICS_SAVE_BAR", True)

    # 1) GT 掩膜
    mask_gt = build_gt_mask_from_json(json_path, image_size, exclude_bg=True)

    # 2) 预测掩膜
    mask_pred = build_pred_mask_from_polygons(all_polygons, image_size)

    # 3) 计算指标
    metrics = compute_pixel_metrics(mask_gt, mask_pred)

    # 4) 可视化
    if save_visualization:
        original_img = Image.open(orig_image).convert("RGB")
        overlay = build_iou_overlay(original_img, mask_gt, mask_pred)
        vis_path = os.path.join(save_dir, f"iou_visualization_{method}_{mode}.png")
        overlay.save(vis_path)
        if verbose:
            print(f"📌 可视化已保存到 {vis_path}")

    # 5) 保存文本指标
    if save_metrics:
        metrics_path = os.path.join(save_dir, f"metrics_{method}_{mode}.txt")
        save_metrics_text(metrics, metrics_path)
        if verbose:
            print(f"📌 指标已保存到 {metrics_path}")

    # 6) 柱状图
    if save_bar:
        bar_path = os.path.join(save_dir, f"metrics_bar_{method}_{mode}.png")
        save_metrics_bar_chart(metrics, bar_path, title=f"Prediction vs GT ({mode})")
        if verbose:
            print(f"📊 柱状图已保存到 {bar_path}")

    if verbose:
        print(
            f"✅ [{mode}] IoU={metrics['IoU']:.4f}, "
            f"P={metrics['Precision']:.4f}, "
            f"R={metrics['Recall']:.4f}, "
            f"F1={metrics['F1-score']:.4f}"
        )

    return metrics
