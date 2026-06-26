import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _triangle_area(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) * 0.5


def _simplify_closed_polygon_indices(points, k):
    alive = list(range(len(points)))
    while len(alive) > k:
        n = len(alive)
        areas = [
            _triangle_area(points[alive[(i - 1) % n]], points[idx], points[alive[(i + 1) % n]])
            for i, idx in enumerate(alive)
        ]
        del alive[int(np.argmin(areas))]
    return np.array(alive, dtype=int)


def _closed_slice(points, start, end):
    if start <= end:
        return points[start:end + 1]
    return np.vstack([points[start:], points[:end + 1]])


def _fit_line_tls(points, trim_quantile=0.85):
    source = np.asarray(points, dtype=float)
    fit_pts = source

    for _ in range(4):
        mean = fit_pts.mean(axis=0)
        if len(fit_pts) == 2:
            direction = fit_pts[1] - fit_pts[0]
        else:
            _, _, vh = np.linalg.svd(fit_pts - mean, full_matrices=False)
            direction = vh[0]

        direction /= np.linalg.norm(direction)
        normal = np.array([-direction[1], direction[0]])
        c = -normal @ mean

        if len(source) <= 3:
            break

        residual = np.abs(source @ normal + c)
        kept = source[residual <= np.quantile(residual, trim_quantile)]
        if len(kept) < 2 or len(kept) == len(fit_pts):
            break
        fit_pts = kept

    return normal, c


def _intersect_lines(line_a, line_b):
    normal_a, c_a = line_a
    normal_b, c_b = line_b
    mat = np.vstack([normal_a, normal_b])
    if abs(np.linalg.det(mat)) < 1e-8:
        return None
    return np.linalg.solve(mat, -np.array([c_a, c_b]))


def _order_leftmost_clockwise(polygon):
    polygon = np.asarray(polygon, dtype=float)
    leftmost_idx = np.lexsort((polygon[:, 0], polygon[:, 1]))[0]
    polygon = np.roll(polygon, -leftmost_idx, axis=0)

    rows, cols = polygon[:, 0], polygon[:, 1]
    signed_area = 0.5 * np.sum(cols * np.roll(rows, -1) - np.roll(cols, -1) * rows)
    if signed_area < 0:
        polygon = np.vstack([polygon[0], polygon[:0:-1]])

    return polygon


def fit_polygon(draw_img_, coords, k=6, use_convex_hull=True,
                refine_edges=True, return_int=False, trim_quantile=0.85):
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return None

    pts = np.unique(pts, axis=0)

    try:
        hull = pts[ConvexHull(pts).vertices]
    except QhullError:
        return None

    if len(hull) < k:
        return None

    corner_idx = _simplify_closed_polygon_indices(hull, k)
    rough_polygon = hull[corner_idx]

    if refine_edges:
        lines = []
        for i in range(k):
            side_pts = _closed_slice(hull, corner_idx[i], corner_idx[(i + 1) % k])
            lines.append(_fit_line_tls(side_pts, trim_quantile=trim_quantile))

        polygon = []
        for i in range(k):
            vertex = _intersect_lines(lines[i - 1], lines[i])
            polygon.append(rough_polygon[i] if vertex is None else vertex)
        polygon = np.asarray(polygon)
    else:
        polygon = rough_polygon

    polygon = _order_leftmost_clockwise(polygon)

    if return_int:
        polygon = np.rint(polygon).astype(int)

    if draw_img_ is not None:
        draw_poly = np.rint(polygon).astype(int)
        for i in range(len(draw_poly)):
            p0 = draw_poly[i]
            p1 = draw_poly[(i + 1) % len(draw_poly)]
            draw_img_.line([(p0[1], p0[0]), (p1[1], p1[0])], fill="red", width=3)
            draw_img_.ellipse((p0[1]-3, p0[0]-3, p0[1]+3, p0[0]+3), fill="blue")

    return polygon