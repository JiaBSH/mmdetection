"""后处理公共工具函数。

收纳原先散落在多个模块中的重复代码：
  - 环境变量读取辅助（_env_flag / _env_int / _env_float / _env_str）
  - safe_convex_hull（退化检测）
  - _to_numpy（tensor → ndarray）
  - _build_overlay（实例掩膜叠加到原图）
  - _compute_physical_scaling（像素→物理单位缩放换算）
  - compute_instance_centroids（实例质心计算）
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull


# ---------------------------------------------------------------------------
# 环境变量辅助
# ---------------------------------------------------------------------------

def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return str(default)
    try:
        return str(v)
    except Exception:
        return str(default)


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def safe_convex_hull(pts: np.ndarray) -> ConvexHull | None:
    """计算凸包，处理退化和点数不足的情况。"""
    pts = np.asarray(pts)
    if pts.shape[0] < 3:
        return None
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()
    if x_range == 0 or y_range == 0:
        return None  # 退化为 1D
    return ConvexHull(pts)


def compute_instance_centroids(instances: list[dict]) -> np.ndarray:
    """计算每个实例的质心坐标 (row, col)，返回 (N, 2) ndarray。

    空实例或退化实例的质心为 [0.0, 0.0]，保持 index 对应关系不变。
    """
    centroids = []
    for inst in instances:
        pts = np.array(list(inst.get("coords", [])))
        if pts.size == 0:
            centroids.append(np.array([0.0, 0.0]))
        else:
            centroids.append(pts.mean(axis=0))
    return np.array(centroids)


# ---------------------------------------------------------------------------
# Tensor / 图像工具
# ---------------------------------------------------------------------------

def _to_numpy(data: Any) -> np.ndarray | None:
    """将 tensor 安全转为 numpy 数组。"""
    if data is None:
        return None
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.asarray(data)


def _build_overlay(
    pil_img: Image.Image,
    instances: list[dict],
    *,
    mask_alpha: int = 160,
    outline_width: int = 2,
) -> Image.Image:
    """将实例 masks 半透明叠加到原图上（带描边），返回 RGBA PIL Image。

    Parameters
    ----------
    mask_alpha : int
        mask 填充透明度 (0-255)。默认 100。
    outline_width : int
        描边宽度（像素）。0 表示不描边。
    """
    import colorsys
    import random
    from PIL import ImageDraw

    W, H = pil_img.size
    base = pil_img.convert("RGBA")

    color_mask = np.zeros((H, W, 4), dtype=np.uint8)
    colors: dict[int, tuple[int, int, int]] = {}

    for inst in instances:
        coords = inst.get("coords")
        if coords is None or len(coords) == 0:
            continue
        inst_id = int(inst.get("id", 1))
        random.seed(inst_id)
        # HSV 生成鲜艳颜色: 饱和/亮度 80-100%, 色相随机
        h = random.random()
        s = random.uniform(0.7, 1.0)
        v = random.uniform(0.8, 1.0)
        r, g, b = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)]
        colors[inst_id] = (r, g, b)
        ys = coords[:, 0].astype(np.int64)
        xs = coords[:, 1].astype(np.int64)
        vm = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        color_mask[ys[vm], xs[vm]] = [r, g, b, mask_alpha]

    overlay_img = Image.fromarray(color_mask, mode="RGBA")

    # 描边：用凸包轮廓，更高不透明度
    if outline_width > 0:
        draw = ImageDraw.Draw(overlay_img)
        outline_alpha = min(255, mask_alpha + 100)
        for inst in instances:
            coords = inst.get("coords")
            if coords is None or len(coords) < 3:
                continue
            inst_id = int(inst.get("id", 1))
            r, g, b = colors.get(inst_id, (255, 255, 255))
            # 提取边界：用凸包顶点作为轮廓
            try:
                hull = safe_convex_hull(coords)
                if hull is not None:
                    pts = np.asarray(coords)[hull.vertices]
                    polygon_xy = [(int(p[1]), int(p[0])) for p in pts]
                else:
                    polygon_xy = [(int(p[1]), int(p[0])) for p in coords]
            except Exception:
                polygon_xy = [(int(p[1]), int(p[0])) for p in coords]
            draw.polygon(polygon_xy, outline=(r, g, b, outline_alpha), width=outline_width)

    return Image.alpha_composite(base, overlay_img)


# ---------------------------------------------------------------------------
# 物理缩放换算
# ---------------------------------------------------------------------------

def _compute_physical_scaling(
    scale_ratio: float | None = None,
    scale_unit: str | None = None,
) -> tuple[float, str, float, str]:
    """从 scale_ratio / scale_unit 导出长度与面积的缩放系数及单位。

    Returns (length_scale, length_unit, area_scale, area_unit).
    """
    try:
        _sr = float(scale_ratio) if scale_ratio is not None else None
        if _sr is not None and _sr > 0:
            length_scale = _sr
            length_unit = (scale_unit or "").strip() or "unit"
        else:
            length_scale = 1.0
            length_unit = "px"
    except Exception:
        length_scale = 1.0
        length_unit = "px"
    area_scale = float(length_scale) * float(length_scale)
    area_unit = f"{length_unit}^2" if length_unit != "px" else "px^2"
    return length_scale, length_unit, area_scale, area_unit
