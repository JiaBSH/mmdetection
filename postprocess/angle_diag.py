from PIL import ImageDraw
import numpy as np
import random
from .util import angle_mod60
from .plot_mod import plot_hist
import os
def angle_diag_reg(all_hexagons, base_img, save_dir, *, save_images: bool = True, save_hist: bool = True):
    """计算并绘制所有六边形的对角线方向角分布（相对于随机参考凸包对角线）"""
    diag_angles = []
    img3 = None
    draw3 = None
    if save_images:
        img3 = base_img.copy()
        draw3 = ImageDraw.Draw(img3)
    for hull_pts in all_hexagons:
        if hull_pts is None or len(hull_pts) < 3:
            diag_angles.append(0.0)
            continue
        n=len(hull_pts)
        if draw3 is not None:
            polygon_hull_xy = [(p[1], p[0]) for p in hull_pts]
            draw3.polygon(polygon_hull_xy, outline="orange", width=3)
        #p1, p2 = hull_pts[n-1], hull_pts[n//2-1]
        p1, p2 = hull_pts[0], hull_pts[n//2]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        angle = angle_mod60(np.degrees(np.arctan2(dy, dx)))
        diag_angles.append(angle)
        if draw3 is not None:
            draw3.line([tuple(p1[::-1]), tuple(p2[::-1])], fill="blue", width=3)
    if img3 is not None:
        img3.save(os.path.join(save_dir, "angle_diag.png"))
    if save_hist:
        plot_hist(diag_angles, bins=30, color="blue",
                  title="diag_Angle (0~60°)",
                  xlabel="Angle (°)", ylabel="Count",
                  save_path=os.path.join(save_dir, "hist_angle_diag.png"),
                  xlim=(0,60))

    return diag_angles
