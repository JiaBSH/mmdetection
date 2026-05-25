import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

from scipy.ndimage import binary_erosion


def _fallback_contour_points(canvas):
    mask = canvas > 0
    if not np.any(mask):
        return None

    boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
    cc, rr = np.where(boundary)
    if rr.size < 3:
        return None

    pts = np.stack([rr, cc], axis=1).astype(np.float32)
    center = pts.mean(axis=0, keepdims=True)
    rel = pts - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(angles)
    return pts[order].astype(np.int32)


def ex_c(global_instances):
    output=[]
    # NOTE: `instance['coords']` is expected to be an iterable of (row, col) points.
    # We keep the original behavior's convention by rasterizing into a *transposed* mask:
    #   mask[col, row] = 255
    # so the resulting contour points are (row, col).
    for instance in global_instances:
        coords = instance.get('coords', [])
        pts = np.asarray(coords, dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            # fallback to original behavior
            pts = np.array(list(coords), dtype=np.int32)
        if pts.size == 0 or pts.shape[0] < 3:
            continue

        rows = pts[:, 0]
        cols = pts[:, 1]
        rmin = int(rows.min())
        rmax = int(rows.max())
        cmin = int(cols.min())
        cmax = int(cols.max())

        pad = 3
        r0 = rmin - pad
        c0 = cmin - pad
        h = (cmax - cmin) + 1 + 2 * pad  # transposed: first dim is col-range
        w = (rmax - rmin) + 1 + 2 * pad  # second dim is row-range
        if h <= 2 or w <= 2:
            continue

        # Build transposed mask: (col, row)
        canvas = np.zeros((h, w), dtype=np.uint8)
        rr = rows - r0
        cc = cols - c0
        inb = (rr >= 0) & (rr < w) & (cc >= 0) & (cc < h)
        rr = rr[inb]
        cc = cc[inb]
        if rr.size == 0:
            continue
        canvas[cc, rr] = 255

        if cv2 is not None:
            contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            wlu = contour.reshape(-1, 2)
        else:
            wlu = _fallback_contour_points(canvas)
            if wlu is None:
                continue

        # Shift back to global (row, col)
        wlu[:, 0] += r0
        wlu[:, 1] += c0
        output.append(wlu)
    return output
