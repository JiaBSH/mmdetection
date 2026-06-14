"""GT↔Pred 实例匹配策略。

策略模式（Strategy Pattern）：可插拔的匹配算法，支持组合和扩展。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# 匹配辅助函数
# ---------------------------------------------------------------------------

def filter_matches_by_distance(
    row_ind: list[int],
    col_ind: list[int],
    pred_centroids: np.ndarray,
    gt_centroids: np.ndarray,
    max_dist: float,
) -> tuple[list[int], list[int]]:
    """按质心距离过滤匹配对。

    只保留质心欧氏距离 ≤ max_dist 的匹配对。

    Returns (filtered_row_ind, filtered_col_ind).
    """
    if pred_centroids is None or gt_centroids is None:
        return [], []
    if pred_centroids.size == 0 or gt_centroids.size == 0:
        return [], []
    if not np.isfinite(max_dist) or max_dist <= 0:
        return list(row_ind), list(col_ind)

    filtered_r: list[int] = []
    filtered_c: list[int] = []
    for r, c in zip(row_ind, col_ind):
        if 0 <= r < gt_centroids.shape[0] and 0 <= c < pred_centroids.shape[0]:
            d = np.linalg.norm(pred_centroids[c] - gt_centroids[r])
            if d <= max_dist:
                filtered_r.append(r)
                filtered_c.append(c)
    return filtered_r, filtered_c


def build_gt_raster_sets(
    gt_polygons_pts: list[np.ndarray], image_size: tuple[int, int]
) -> list[set]:
    """将 GT 多边形光栅化为像素集合。"""
    W, H = image_size
    gt_sets: list[set] = []
    for poly in gt_polygons_pts:
        mask_img = Image.new("L", (W, H), 0)
        if len(poly) < 3:
            gt_sets.append(set())
            continue
        poly_xy = [(int(p[1]), int(p[0])) for p in poly]
        ImageDraw.Draw(mask_img).polygon(poly_xy, outline=1, fill=1)
        ys, xs = np.where(np.array(mask_img) > 0)
        gt_sets.append(set(zip(ys, xs)))
    return gt_sets


# ---------------------------------------------------------------------------
# 匹配策略抽象
# ---------------------------------------------------------------------------

class MatchingStrategy(ABC):
    """GT-Pred 实例匹配策略抽象基类。"""

    @abstractmethod
    def match(
        self,
        global_instances: list[dict],
        gt_polygons_pts: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        """执行匹配。

        Parameters
        ----------
        global_instances : 预测实例列表（含 'coords' ndarray）
        gt_polygons_pts : GT 多边形顶点列表（每个 ndarray (N,2) in row,col）
        image_size : (W, H)

        Returns (row_ind, col_ind) — 匹配的 (GT index, Pred index) 列表。
        """
        ...


class OverlapHungarianStrategy(MatchingStrategy):
    """基于像素重叠的匈牙利匹配。

    构建 cost = -overlap 矩阵，用 scipy.optimize.linear_sum_assignment 求解。
    失败时回退到贪心匹配。
    """

    def match(
        self,
        global_instances: list[dict],
        gt_polygons_pts: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        W, H = image_size

        # 预测实例像素集
        pred_sets = [inst["coords"] for inst in global_instances]

        # GT 光栅化像素集
        gt_sets = build_gt_raster_sets(gt_polygons_pts, image_size)

        if len(pred_sets) == 0 or len(gt_sets) == 0:
            return [], []

        # 构建 cost 矩阵
        cost = np.zeros((len(gt_sets), len(pred_sets)), dtype=float)
        for i, gset in enumerate(gt_sets):
            for j, pset in enumerate(pred_sets):
                inter = len(gset & pset)
                cost[i, j] = -inter

        try:
            row_ind, col_ind = linear_sum_assignment(cost)
            return list(row_ind), list(col_ind)
        except Exception:
            # 贪心回退
            row_ind = []
            col_ind = []
            for i, gset in enumerate(gt_sets):
                best_j = -1
                best_inter = 0
                for j, pset in enumerate(pred_sets):
                    inter = len(gset & pset)
                    if inter > best_inter:
                        best_inter = inter
                        best_j = j
                if best_j >= 0:
                    row_ind.append(i)
                    col_ind.append(best_j)
            return row_ind, col_ind


class CentroidNNStrategy(MatchingStrategy):
    """按质心最近邻匹配。

    用作 OverlapHungarian 的降级回退。
    """

    def __init__(self, max_dist: float = 200.0):
        self.max_dist = float(max_dist)

    def match(
        self,
        global_instances: list[dict],
        gt_polygons_pts: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        from ._shared import compute_instance_centroids

        pred_centroids = compute_instance_centroids(global_instances)

        # GT 质心
        gt_centroids_list = []
        for poly in gt_polygons_pts:
            poly_arr = np.asarray(poly)
            if poly_arr.ndim == 2 and poly_arr.shape[0] > 0:
                gt_centroids_list.append(poly_arr.mean(axis=0))
            else:
                gt_centroids_list.append(np.array([0.0, 0.0]))
        gt_centroids = np.array(gt_centroids_list)

        if pred_centroids.shape[0] == 0 or gt_centroids.shape[0] == 0:
            return [], []

        if not np.isfinite(self.max_dist) or self.max_dist <= 0:
            return [], []

        # 构建候选
        candidates = []
        for gi in range(gt_centroids.shape[0]):
            d = np.linalg.norm(pred_centroids - gt_centroids[gi], axis=1)
            if d.size == 0:
                continue
            pj = int(np.argmin(d))
            dist = float(d[pj])
            if np.isfinite(dist) and dist <= self.max_dist:
                candidates.append((gi, pj, dist))

        if not candidates:
            return [], []

        # 一对一匹配：距离最近优先
        candidates.sort(key=lambda x: x[2])
        used_pred: set[int] = set()
        row_ind: list[int] = []
        col_ind: list[int] = []
        for gi, pj, _ in candidates:
            if pj not in used_pred:
                used_pred.add(pj)
                row_ind.append(gi)
                col_ind.append(pj)

        return row_ind, col_ind


# ---------------------------------------------------------------------------
# 向后兼容的独立函数（与 analyze_main_dy2 原有 API 一致）
# ---------------------------------------------------------------------------

_overlap_strategy = OverlapHungarianStrategy()
_centroid_strategy = CentroidNNStrategy(max_dist=200.0)


def match_by_overlap(
    global_instances: list[dict],
    gt_polygons_pts: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[list[int], list[int]]:
    """按像素重叠的匈牙利匹配（向后兼容）。"""
    return _overlap_strategy.match(global_instances, gt_polygons_pts, image_size)


def match_by_centroid_nn(
    pred_centroids: np.ndarray,
    gt_centroids: np.ndarray,
    max_dist: float,
) -> tuple[list[int], list[int]]:
    """按质心最近邻匹配（向后兼容）。

    注意：此函数接受预计算的质心数组，而非 raw instances。
    """
    s = CentroidNNStrategy(max_dist=max_dist)
    # 需要适配签名 — 此函数接受质心而非 instances
    # 直接实现以保持精确兼容
    pred_centroids = np.asarray(pred_centroids, dtype=float)
    gt_centroids = np.asarray(gt_centroids, dtype=float)
    if pred_centroids.ndim != 2 or gt_centroids.ndim != 2:
        return [], []
    if pred_centroids.shape[1] != 2 or gt_centroids.shape[1] != 2:
        return [], []
    if pred_centroids.shape[0] == 0 or gt_centroids.shape[0] == 0:
        return [], []

    max_dist = float(max_dist)
    if not np.isfinite(max_dist) or max_dist <= 0:
        return [], []

    candidates = []
    for gi in range(gt_centroids.shape[0]):
        d = np.linalg.norm(pred_centroids - gt_centroids[gi], axis=1)
        if d.size == 0:
            continue
        pj = int(np.argmin(d))
        dist = float(d[pj])
        if np.isfinite(dist) and dist <= max_dist:
            candidates.append((gi, pj, dist))

    if not candidates:
        return [], []

    candidates.sort(key=lambda x: x[2])
    used_pred: set[int] = set()
    row_ind: list[int] = []
    col_ind: list[int] = []
    for gi, pj, _ in candidates:
        if pj not in used_pred:
            used_pred.add(pj)
            row_ind.append(gi)
            col_ind.append(pj)

    return row_ind, col_ind


class CompositeStrategy(MatchingStrategy):
    """组合策略：primary 失败时自动回退到 fallback。

    默认组合：OverlapHungarian → CentroidNN
    """

    def __init__(
        self,
        primary: MatchingStrategy | None = None,
        fallback: MatchingStrategy | None = None,
        fallback_max_dist: float = 200.0,
    ):
        self.primary = primary or OverlapHungarianStrategy()
        self.fallback = fallback or CentroidNNStrategy(max_dist=fallback_max_dist)

    def match(
        self,
        global_instances: list[dict],
        gt_polygons_pts: list[np.ndarray],
        image_size: tuple[int, int],
    ) -> tuple[list[int], list[int]]:
        row_ind, col_ind = self.primary.match(
            global_instances, gt_polygons_pts, image_size
        )
        if len(row_ind) == 0 and len(gt_polygons_pts) > 0 and len(global_instances) > 0:
            row_ind, col_ind = self.fallback.match(
                global_instances, gt_polygons_pts, image_size
            )
        return row_ind, col_ind
