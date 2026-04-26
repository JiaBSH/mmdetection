from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import math


def _draw_arrow(draw, p1_xy, p2_xy, *, color="blue", width=3, head_len=None, head_angle_deg=30):
    """Draw an arrow from p1->p2 using PIL.ImageDraw.

    p1_xy/p2_xy are (x, y).
    """
    try:
        x1, y1 = float(p1_xy[0]), float(p1_xy[1])
        x2, y2 = float(p2_xy[0]), float(p2_xy[1])
    except Exception:
        return

    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if not math.isfinite(L) or L <= 1e-6:
        return

    draw.line([(int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))], fill=color, width=int(width))

    if head_len is None:
        head_len = max(10.0, float(width) * 3.0)
    head_len = min(head_len, L * 0.5)

    ux = dx / L
    uy = dy / L
    bx = -ux
    by = -uy

    ang = math.radians(float(head_angle_deg))
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)

    rx1 = bx * cos_a - by * sin_a
    ry1 = bx * sin_a + by * cos_a
    rx2 = bx * cos_a + by * sin_a
    ry2 = -bx * sin_a + by * cos_a

    p3 = (x2 + rx1 * head_len, y2 + ry1 * head_len)
    p4 = (x2 + rx2 * head_len, y2 + ry2 * head_len)

    draw.line([(int(round(x2)), int(round(y2))), (int(round(p3[0])), int(round(p3[1])))], fill=color, width=int(width))
    draw.line([(int(round(x2)), int(round(y2))), (int(round(p4[0])), int(round(p4[1])))], fill=color, width=int(width))


def diaglen_edgeangle_overlay(
    all_hexagons: Iterable,
    base_img: Image.Image,
    save_dir: str,
    *,
    save_name: str = "length_diag_edge_angle.png",
    outline_color: str = "orange",
    outline_width: int = 8,
    diag_color: str = "#fbb03b",
    diag_width: int = 8,
    edge_color: str = "#0000ff",
    edge_width: int = 8,
) -> "str | None":
    """Draw DiagonalLength and EdgeAngle overlays on ONE image.

    This mirrors the visual conventions of:
    - `lenght_diag_reg` (yellow diagonal line between vertex 0 and vertex n//2)
    - `angle_edge_reg`  (orange polygon outline + blue edge between vertex 0 and 1)

    It intentionally does not change the existing single-output images.

    Returns the saved image path, or None if nothing was saved.
    """

    if base_img is None:
        return None

    os.makedirs(save_dir, exist_ok=True)
    img = base_img.copy()
    draw = ImageDraw.Draw(img)

    for hull_pts in all_hexagons:
        if hull_pts is None:
            continue
        pts = np.asarray(hull_pts)
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 2:
            continue

        n = int(pts.shape[0])

        # polygon outline (same as angle_edge)
        try:
            polygon_xy = [(int(p[1]), int(p[0])) for p in pts]
        except Exception:
            pass

        # edge direction (vertex 0 -> 1) (same as angle_edge)
        try:
            p1 = pts[0]
            p2 = pts[1]
            _draw_arrow(
                draw,
                tuple(map(int, p1[::-1])),
                tuple(map(int, p2[::-1])),
                color=edge_color,
                width=int(edge_width),
            )
        except Exception:
            pass

        # diagonal length line (vertex 0 -> n//2) (same as lenght_diag)
        if n >= 3:
            try:
                p1 = pts[0]
                p2 = pts[n // 2]
                draw.line(
                    [tuple(map(int, p1[::-1])), tuple(map(int, p2[::-1]))],
                    fill=diag_color,
                    width=int(diag_width),
                )
            except Exception:
                pass

    out_path = os.path.join(save_dir, save_name)
    img.save(out_path)
    return out_path
