from PIL import ImageDraw
import numpy as np
import math
from .util import angle_mod60
from .plot_mod import plot_hist
import os


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
    # direction reversed from tip back towards tail
    bx = -ux
    by = -uy

    ang = math.radians(float(head_angle_deg))
    cos_a = math.cos(ang)
    sin_a = math.sin(ang)

    # rotate base vector by +/- ang
    rx1 = bx * cos_a - by * sin_a
    ry1 = bx * sin_a + by * cos_a
    rx2 = bx * cos_a + by * sin_a
    ry2 = -bx * sin_a + by * cos_a

    p3 = (x2 + rx1 * head_len, y2 + ry1 * head_len)
    p4 = (x2 + rx2 * head_len, y2 + ry2 * head_len)

    draw.line([(int(round(x2)), int(round(y2))), (int(round(p3[0])), int(round(p3[1])))], fill=color, width=int(width))
    draw.line([(int(round(x2)), int(round(y2))), (int(round(p4[0])), int(round(p4[1])))], fill=color, width=int(width))
def angle_edge_reg(all_hexagons,  base_img, save_dir, *, save_images: bool = True, save_hist: bool = True, draw_vertex_labels: bool = False):
    """计算并绘制所有六边形的边缘方向角分布（相对于随机参考凸包边缘）"""
    edge_angles = []
    XP1=[]
    YP1=[]
    XP2=[]
    YP2=[]
    img4 = None
    draw4 = None
    if save_images:
        img4 = base_img.copy()
        draw4 = ImageDraw.Draw(img4)

    for hull_pts in all_hexagons:
        if hull_pts is None or len(hull_pts) < 2:
            edge_angles.append(0.0)
            XP1.append(0)
            YP1.append(0)
            XP2.append(0)
            YP2.append(0)
            continue
        n = len(hull_pts)
        #p1, p2 = hull_pts[n-2], hull_pts[n-1]
        p1, p2 = hull_pts[0], hull_pts[1]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        XP1.append(p1[0])
        YP1.append(p1[1])
        XP2.append(p2[0])
        YP2.append(p2[1])
        if draw4 is not None and draw_vertex_labels:
            js=0
            for  px,py in hull_pts:
                js+=1
                draw4.ellipse((py-2, px-2, py+2, px+2), fill="red")
                draw4.text((py+5, px-5), f"{js}", fill="yellow")
        angle = angle_mod60(np.degrees(np.arctan2(dy, dx)))
        edge_angles.append(angle)
        #draw4.line([tuple(p1[::-1]), tuple(p2[::-1])], fill="purple", width=3)
        if draw4 is not None:
            _draw_arrow(draw4, tuple(p1[::-1]), tuple(p2[::-1]), color="#0000ff", width=16)
    
    if img4 is not None:
        img4.save(os.path.join(save_dir, "angle_edge.png"))
    if save_hist:
        plot_hist(edge_angles, bins=30, color="purple",
                  title="Edge Angle Δ (0~60°)",
                  xlabel="Angle (°)", ylabel="Count",
                  save_path=os.path.join(save_dir, "hist_angle_edge.png"),
                  xlim=(0,60))
    return edge_angles,XP1,YP1,XP2,YP2
