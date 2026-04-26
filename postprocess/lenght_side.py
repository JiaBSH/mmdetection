from PIL import ImageDraw
import numpy as np
import random
from .plot_mod import plot_hist
import os
def lenght_side_reg(all_hexagons, base_img, save_dir, scale_ratio=None, unit='px', *, save_images: bool = True, save_hist: bool = True):
    side_dists = []
    img1 = None
    draw1 = None
    if save_images:
        img1 = base_img.copy()
        draw1 = ImageDraw.Draw(img1)
    # optional scaling
    scale = 1.0
    try:
        if scale_ratio is not None and float(scale_ratio) > 0:
            scale = float(scale_ratio)
    except Exception:
        scale = 1.0
    unit = unit if unit else 'px'
    for hull_pts in all_hexagons:
        if hull_pts is None or len(hull_pts) < 3:
            side_dists.append(0.0)
            continue
        max_dist = 0
        best_pair = None
        n = len(hull_pts)
        for i in range(n):
            p1, p2 = hull_pts[i], hull_pts[(i+1) % n]
            edge_vec = p2 - p1
            norm = np.linalg.norm(edge_vec)
            if norm == 0:
                continue
            edge_vec = edge_vec / norm
            normal = np.array([-edge_vec[1], edge_vec[0]])
            proj = hull_pts @ normal
            dist = proj.max() - proj.min()
            if dist > max_dist:
                max_dist = dist
                best_pair = (proj.argmax(), proj.argmin(), normal)
        side_dists.append(max_dist)
        if best_pair is not None:
            i1, i2, _ = best_pair
            p1, p2 = hull_pts[i1], hull_pts[i2]
            if draw1 is not None:
                draw1.line([tuple(p1[::-1]), tuple(p2[::-1])], fill="red", width=3)
    if img1 is not None:
        img1.save(os.path.join(save_dir, "length_side.png"))
    scaled_side_dists = [d * scale for d in side_dists]
    if save_hist:
        plot_hist(scaled_side_dists, bins=20, color="red",
                  title="Opposite-Side Distance",
                  xlabel=f"Distance ({unit})", ylabel="Count",
                  save_path=os.path.join(save_dir, "hist_length_side.png"))
    return scaled_side_dists
