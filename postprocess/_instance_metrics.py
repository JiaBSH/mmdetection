"""逐实例指标计算。

利用匹配阶段（match_by_overlap）已算出的像素集合交集，
零额外光栅化成本归一化为逐实例 IoU / Precision / Recall / F1。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def compute_per_instance_iou(
    gt_pixel_set: set, pred_pixel_set: set
) -> dict[str, float]:
    """从 GT 和 Pred 的像素集合计算逐实例指标。

    Parameters
    ----------
    gt_pixel_set : set of (y, x) tuples
    pred_pixel_set : set of (y, x) tuples

    Returns
    -------
    dict with keys: iou, precision, recall, f1
    """
    inter = len(gt_pixel_set & pred_pixel_set)
    gt_count = len(gt_pixel_set)
    pred_count = len(pred_pixel_set)
    union = gt_count + pred_count - inter

    iou = inter / union if union > 0 else 0.0
    precision = inter / pred_count if pred_count > 0 else 0.0
    recall = inter / gt_count if gt_count > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def compute_centroid_distance(
    gt_polygon: np.ndarray | None, pred_coords: np.ndarray | None
) -> float:
    """计算 GT 和 Pred 实例质心之间的欧氏距离（像素单位）。

    Parameters
    ----------
    gt_polygon : ndarray of shape (N, 2) in (row, col), or None
    pred_coords : ndarray of shape (M, 2) in (row, col), or None

    Returns
    -------
    float — 质心距离（像素），任一为空则返回 NaN。
    """
    if gt_polygon is None or len(gt_polygon) == 0:
        return float("nan")
    if pred_coords is None or len(pred_coords) == 0:
        return float("nan")

    gt_c = np.mean(np.asarray(gt_polygon), axis=0)
    pr_c = np.mean(np.asarray(pred_coords), axis=0)
    return float(np.linalg.norm(gt_c - pr_c))


def compute_instance_metrics_for_matches(
    gt_polygons_pts: list[np.ndarray],
    valid_instances: list[dict],
    match_row: list[int],
    match_col: list[int],
    image_size: tuple[int, int],
) -> list[dict[str, float]]:
    """对每对匹配计算逐实例 IoU/P/R/F1 + 质心距离。

    为每个匹配对重新光栅化 GT 多边形和 Pred 实例的像素集，
    然后调用 compute_per_instance_iou 得到逐实例指标。

    Parameters
    ----------
    gt_polygons_pts : GT 多边形顶点列表，每个 ndarray (N, 2) in (row, col)
    valid_instances : 预测实例列表，每个 dict 含 'coords' (M, 2) ndarray
    match_row : 匹配的 GT 索引列表
    match_col : 匹配的 Pred 索引列表
    image_size : (W, H)

    Returns
    -------
    list[dict] — 长度与匹配数相同，每个 dict 含 iou, precision, recall, f1, centroid_distance
    """
    W, H = image_size
    results: list[dict[str, float]] = []

    for r, c in zip(match_row, match_col):
        # 光栅化 GT 多边形
        gt_poly = gt_polygons_pts[r] if 0 <= r < len(gt_polygons_pts) else None
        if gt_poly is not None and len(gt_poly) >= 3:
            mask_gt = Image.new("L", (W, H), 0)
            poly_xy = [(int(p[1]), int(p[0])) for p in gt_poly]
            ImageDraw.Draw(mask_gt).polygon(poly_xy, outline=1, fill=1)
            gs_ys, gs_xs = np.where(np.array(mask_gt) > 0)
            gt_set = set(zip(gs_ys, gs_xs))
        else:
            gt_set = set()

        # 获取 Pred 实例像素集
        pred_inst = valid_instances[c] if 0 <= c < len(valid_instances) else None
        if pred_inst is not None:
            pred_coords = pred_inst.get("coords")
            if pred_coords is not None and len(pred_coords) > 0:
                pred_set = set(
                    (int(y), int(x)) for y, x in np.asarray(pred_coords)
                )
            else:
                pred_set = set()
                pred_coords = np.array([])
        else:
            pred_set = set()
            pred_coords = np.array([])

        metrics = compute_per_instance_iou(gt_set, pred_set)
        metrics["centroid_distance"] = compute_centroid_distance(
            gt_poly, pred_coords if isinstance(pred_coords, np.ndarray) else None
        )
        results.append(metrics)

    return results
