import numpy as np
from scipy.spatial import ConvexHull


def draw_dashed_line(draw, p0, p1, dash=(5, 5), fill="green", width=1):
    """Draw a dashed line between p0 and p1 using a PIL-like draw object.

    p0, p1: (x, y) sequences
    dash: (on_length, off_length) in pixels
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length == 0:
        return
    direction = vec / length
    on_len, off_len = dash
    pos = 0.0
    while pos < length:
        start = p0 + direction * pos
        end = p0 + direction * min(pos + on_len, length)
        draw.line([tuple(start.astype(int)), tuple(end.astype(int))], fill=fill, width=width)
        pos += on_len + off_len
def fit_polygon(draw_img_, coords, k=6, use_convex_hull=False):
    """
    旋转投影法多边形拟合：
    在 k 个方向上取投影最大点作为 k 边形顶点。
    最左边的点为第一个点，顺时针依次取点。
    
    :param coords: iterable of (x,y)，注意这里是图像行列坐标风格 (x=row, y=col)
    :param k: 多边形边数（5/6/7/8/9/10等）
    :param use_convex_hull: 已弃用，仅为兼容性保留
    :return: 顶点 ndarray，shape=(k,2)，以最左边的点开始，顺时针排列
    """
    # coords: iterable/array of (row=x, col=y)
    # draw_img_ may be None to disable expensive debug drawing.
    pts = np.asarray(coords)
    if pts.ndim != 2 or pts.shape[1] != 2:
        pts = np.array(list(coords))
    
    if len(pts) < 3:
        return None

    # 使用点集质心作为中心
    center = pts.mean(axis=0)

    polygon = []
    # 均匀分布的 k 个方向（从 0.5π 开始，逆时针增加）
    for theta in np.linspace(0.5*np.pi, 2.5*np.pi, k, endpoint=False):
        dir_vec = np.array([np.cos(theta), np.sin(theta)])
        projections = (pts - center) @ dir_vec
        max_pt = pts[np.argmax(projections)]
        polygon.append(max_pt)
        
        # 绘制投影方向和最大点（可视化，可选）
        if draw_img_ is not None:
            p0 = tuple(center[::-1])
            p1 = tuple((center[::-1] + dir_vec * 100))
            draw_dashed_line(draw_img_, p0, p1, dash=(8, 4), fill="green", width=3)
            draw_img_.ellipse((max_pt[1]-3, max_pt[0]-3, max_pt[1]+3, max_pt[0]+3), fill="blue")
    
    polygon = np.array(polygon)
    
    # 找到最左边的点（x 坐标最小）
    leftmost_idx = np.argmin(polygon[:, 1])  # 第二列（y/col）是"左右"维度
    
    # 按照最左边的点开始，重新排列顶点
    polygon = np.roll(polygon, -leftmost_idx, axis=0)
    
    # 判断当前顺序是否顺时针，如果不是则反转
    # 使用有向面积：如果为负则是顺时针，为正则是逆时针
    signed_area = 0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        signed_area += (x2 - x1) * (y2 + y1)
    
    if signed_area < 0:  # 逆时针，需要反转
        polygon = np.vstack([polygon[0], polygon[1:][::-1]])
    
    return polygon
