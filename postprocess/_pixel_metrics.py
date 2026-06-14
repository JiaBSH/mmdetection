"""像素级指标：纯计算 + 掩膜构建 + 可视化输出。

将原先 compute_metrics.py 的 Compute_metrics 函数的四个职责拆分为独立函数：
  1. compute_pixel_metrics — 纯计算 TP/FP/FN → IoU/P/R/F1
  2. build_gt_mask_from_json — GT 掩膜构建（I/O）
  3. build_pred_mask_from_polygons — 预测掩膜构建（光栅化）
  4. build_iou_overlay — 可视化 overlay
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def compute_pixel_metrics(
    mask_gt: np.ndarray, mask_pred: np.ndarray
) -> dict[str, float]:
    """纯计算：从二值掩膜计算像素级 TP/FP/FN → IoU/P/R/F1。

    Parameters
    ----------
    mask_gt : ndarray, bool
    mask_pred : ndarray, bool

    Returns
    -------
    dict with keys: IoU, Precision, Recall, F1-score
    """
    TP = np.logical_and(mask_gt, mask_pred).sum()
    FP = np.logical_and(~mask_gt, mask_pred).sum()
    FN = np.logical_and(mask_gt, ~mask_pred).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return {"IoU": iou, "Precision": precision, "Recall": recall, "F1-score": f1}


def build_gt_mask_from_json(
    json_path: str, image_size: tuple[int, int], exclude_bg: bool = True
) -> np.ndarray:
    """从 ISAT 格式 JSON 加载 GT 掩膜。

    Parameters
    ----------
    json_path : ISAT 格式标注 JSON，含 objects[].segmentation: [[x,y],...]
    image_size : (W, H)
    exclude_bg : 是否排除 __background__ 类

    Returns
    -------
    ndarray, bool — 形状 (H, W)
    """
    W, H = image_size

    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    mask_gt = Image.new("L", (W, H), 0)
    draw_gt = ImageDraw.Draw(mask_gt)
    mask_bg = Image.new("L", (W, H), 0)
    draw_bg = ImageDraw.Draw(mask_bg)

    for obj in ann.get("objects", []):
        coords = obj.get("segmentation", [])
        if not coords:
            continue
        polygon = [(x, y) for x, y in coords]
        if obj.get("category", "").strip() == "__background__":
            draw_bg.polygon(polygon, outline=1, fill=1)
        else:
            draw_gt.polygon(polygon, outline=1, fill=1)

    mask_gt_np = np.array(mask_gt, dtype=bool)
    if exclude_bg:
        mask_bg_np = np.array(mask_bg, dtype=bool)
        mask_gt_np = np.logical_and(mask_gt_np, ~mask_bg_np)

    return mask_gt_np


def build_pred_mask_from_polygons(
    all_polygons: list, image_size: tuple[int, int]
) -> np.ndarray:
    """从预测多边形列表构建掩膜。

    Parameters
    ----------
    all_polygons : list of ndarray, 每个 (m, 2) in (row, col)
    image_size : (W, H)

    Returns
    -------
    ndarray, bool — 形状 (H, W)
    """
    W, H = image_size
    mask_pred = Image.new("L", (W, H), 0)
    draw_pred = ImageDraw.Draw(mask_pred)

    for poly in all_polygons:
        if poly is None or len(poly) < 3:
            continue
        polygon = [(p[1], p[0]) for p in poly]  # (row, col) → (col, row)
        draw_pred.polygon(polygon, outline=1, fill=1)

    return np.array(mask_pred, dtype=bool)


def build_iou_overlay(
    orig_image: Image.Image,
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
) -> Image.Image:
    """创建 IoU 可视化 overlay：蓝=GT, 红=Pred, 绿=重叠。"""
    H, W = mask_gt.shape
    base = orig_image.convert("RGBA")

    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[mask_gt] = [0, 0, 255]  # 蓝色 GT
    vis[mask_pred] = [255, 0, 0]  # 红色预测
    vis[np.logical_and(mask_gt, mask_pred)] = [0, 255, 0]  # 绿色重叠

    mask_img = Image.fromarray(vis).convert("RGBA")
    mask_img.putalpha(128)  # 半透明

    return Image.alpha_composite(base, mask_img)


def save_metrics_text(metrics: dict, path: str) -> None:
    """保存指标到文本文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"IoU: {metrics['IoU']:.4f}\n")
        f.write(f"Precision: {metrics['Precision']:.4f}\n")
        f.write(f"Recall: {metrics['Recall']:.4f}\n")
        f.write(f"F1-score: {metrics['F1-score']:.4f}\n")


def save_metrics_bar_chart(
    metrics: dict, path: str, title: str = "Prediction vs GT"
) -> None:
    """保存指标柱状图。"""
    names = ["IoU", "Precision", "Recall", "F1-score"]
    values = [metrics[n] for n in names]

    plt.figure(figsize=(6, 4))
    plt.bar(
        names, values,
        color=["orange", "red", "blue", "green"],
        edgecolor="black",
    )
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    for i, v in enumerate(values):
        plt.text(i, min(v + 0.03, 1.0), f"{v:.2f}", ha="center", fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
