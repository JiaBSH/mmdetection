from PIL import ImageDraw
import numpy as np
import random
from .plot_mod import plot_hist
import os
def lenght_diag_reg(all_hexagons, base_img, save_dir, scale_ratio=None, unit='px', *, save_images: bool = True, save_hist: bool = True):
    diag_lens = []
    img2 = None
    draw2 = None
    if save_images:
        img2 = base_img.copy()
        draw2 = ImageDraw.Draw(img2)
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
            diag_lens.append(0.0)
            continue
        n=len(hull_pts)
        polygon_hull_xy = [(p[1], p[0]) for p in hull_pts]
        #draw2.polygon(polygon_hull_xy, outline="orange", width=3)
        p1, p2 = hull_pts[0], hull_pts[n//2]
        length = np.linalg.norm(p1 - p2)
        diag_lens.append(length * scale)
        if draw2 is not None:
            draw2.line([tuple(p1[::-1]), tuple(p2[::-1])], fill="#fbb03b", width=16)
    if img2 is not None:
        img2.save(os.path.join(save_dir, "length_diag.png"))
    if save_hist:
        plot_hist(diag_lens, bins=20, color="green",
                  title="Longest Diagonal Length",
                  xlabel=f"Length ({unit})", ylabel="Count",
                  save_path=os.path.join(save_dir, "hist_length_diag.png"))
    return diag_lens
