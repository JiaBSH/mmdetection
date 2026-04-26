from PIL import ImageDraw
def draw_polygons_overlay(overlayed, polygons, save_path, outline="orange", width=3, draw_hull_points=False):
    """
    把 polygons 叠加画在 overlayed 上保存
    :param overlayed: PIL.Image
    :param polygons: list of ndarray, 每个 (m,2) 顶点 (row,col)
    :param save_path: 保存路径
    :param outline: 线条颜色
    :param width: 线宽
    :param draw_hull_points: 是否把每个多边形顶点画成小圆点（调试用）
    """
    img = overlayed.copy()
    draw = ImageDraw.Draw(img)

    for poly in polygons:
        if poly is None or len(poly) < 3:
            continue
        polygon_xy = [(p[1], p[0]) for p in poly]  # (col,row)
        draw.polygon(polygon_xy, outline=outline, width=width)
        if draw_hull_points:
            for (x, y) in poly:  # (row,col)
                r = 4
                draw.ellipse((y - r, x - r, y + r, x + r), outline="lime", fill="lime")

    img.save(save_path)
