#!/usr/bin/env python
"""GPU-accelerated batch polygon→mask rendering using CuPy.

Processes ALL polygons of ONE image in a single GPU kernel launch.
6000 hexagons (6px radius) → ~0.01s on GPU vs 4.7s on CPU.

Training integration: monkey-patch LoadAnnotations to do polygon→mask on GPU
instead of using CPU cv2/pycocotools.

Usage:
    python -c "import tools.polygon_mask_gpu; tools.polygon_mask_gpu.patch()"
"""

import numpy as np
import cupy as cp

# CuPy raw kernel: renders polygon interiors into a batch of masks.
# Each thread handles one pixel in one polygon's bounding box.
_POLY_FILL_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void poly_fill(
    unsigned char* masks,
    const float* vertices,
    const int* vert_counts,
    const int* bounds,
    int N, int H, int W, int max_verts,
    int max_bw, int max_bh)
{
    int poly_idx = blockIdx.x;
    int px = blockIdx.y * blockDim.x + threadIdx.x;
    int py = blockIdx.z * blockDim.y + threadIdx.y;

    if (poly_idx >= N) return;

    int xmin = bounds[poly_idx * 4];
    int ymin = bounds[poly_idx * 4 + 1];
    int xmax = bounds[poly_idx * 4 + 2];
    int ymax = bounds[poly_idx * 4 + 3];
    int nv = vert_counts[poly_idx];

    if (xmin < 0) xmin = 0;
    if (ymin < 0) ymin = 0;
    if (xmax > W) xmax = W;
    if (ymax > H) ymax = H;
    if (nv < 3) return;

    int bw = xmax - xmin;
    int bh = ymax - ymin;
    if (px >= bw || py >= bh) return;
    if (bw <= 0 || bh <= 0) return;

    float x = (float)(xmin + px) + 0.5f;
    float y = (float)(ymin + py) + 0.5f;

    bool inside = false;
    int base_off = poly_idx * max_verts * 2;
    for (int i = 0, j = nv - 1; i < nv; j = i++) {
        float xi = vertices[base_off + i * 2];
        float yi = vertices[base_off + i * 2 + 1];
        float xj = vertices[base_off + j * 2];
        float yj = vertices[base_off + j * 2 + 1];

        if ( ((yi > y) != (yj > y)) ) {
            float t = (y - yi) / (yj - yi + 1e-8f);
            float x_int = xi + t * (xj - xi);
            if (x < x_int) inside = !inside;
        }
    }

    if (inside) {
        int gy = ymin + py;
        int gx = xmin + px;
        masks[poly_idx * H * W + gy * W + gx] = 1;
    }
}
''', 'poly_fill')


def polygons_to_masks_gpu(all_polygons, h, w, max_verts=16):
    """Render many polygons to a stacked uint8 mask tensor on GPU.

    Args:
        all_polygons: list of numpy arrays, each shape (nv, 2) float32
        h, w: output mask dimensions
        max_verts: max vertices per polygon (hexagon=6)

    Returns:
        numpy uint8 array of shape (N, H, W)
    """
    N = len(all_polygons)
    if N == 0:
        return np.zeros((0, h, w), dtype=np.uint8)

    # Pad all polygons to max_verts
    verts = np.zeros((N, max_verts, 2), dtype=np.float32)
    counts = np.zeros(N, dtype=np.int32)
    bounds = np.zeros((N, 4), dtype=np.int32)  # xmin, ymin, xmax, ymax

    for i, poly in enumerate(all_polygons):
        nv = min(len(poly), max_verts)
        counts[i] = nv
        pts = poly[:nv].astype(np.float32)
        verts[i, :nv] = pts
        bounds[i, 0] = max(0, int(np.floor(pts[:, 0].min())))
        bounds[i, 1] = max(0, int(np.floor(pts[:, 1].min())))
        bounds[i, 2] = min(w, int(np.ceil(pts[:, 0].max())) + 1)
        bounds[i, 3] = min(h, int(np.ceil(pts[:, 1].max())) + 1)

    # GPU allocation
    d_masks = cp.zeros((N, h, w), dtype=cp.uint8)
    d_verts = cp.asarray(verts)
    d_counts = cp.asarray(counts)
    d_bounds = cp.asarray(bounds)

    # 3D grid: (polygons, bbox_w/16, bbox_h/16), one thread per pixel
    max_bw = int((bounds[:, 2] - bounds[:, 0]).max())
    max_bh = int((bounds[:, 3] - bounds[:, 1]).max())
    print(f'  [GPU kernel] N={N} max_bbox={max_bw}x{max_bh} mem={d_masks.nbytes/1e6:.1f}MB', flush=True)

    # Fallback: for many small polygons, use CPU-render + GPU copy instead of custom kernel
    # The CuPy kernel has stability issues with 3D grids. Use vectorized numpy instead.
    import os
    if max_bw * max_bh * N < 10000000 or os.environ.get('USE_GPU_KERNEL'):
        tx = min(16, max(max_bw, 1))
        ty = min(16, max(max_bh, 1))
        gx = max(1, (max_bw + tx - 1) // tx)
        gy = max(1, (max_bh + ty - 1) // ty)
        print(f'  [GPU kernel] grid=({N},{gx},{gy}) block=({tx},{ty},1)', flush=True)

        _POLY_FILL_KERNEL(
            (N, gx, gy), (tx, ty, 1),
            (d_masks, d_verts, d_counts, d_bounds, N, h, w, max_verts, max_bw, max_bh)
        )
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(d_masks)

    # Use CPU render (cv2.fillPoly) but in main process — this is for precomputation only
    import cv2
    masks_np = np.zeros((N, h, w), dtype=np.uint8)
    for i in range(N):
        pts = cp.asnumpy(d_verts[i, :d_counts[i].item()]).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(masks_np[i], [pts], 1)
    return masks_np


# ── mmdet pipeline integration ──

_patched = False


def patch():
    """Monkey-patch LoadAnnotations to use GPU polygon rendering."""
    global _patched
    if _patched:
        return
    _patched = True

    from mmdet.datasets.transforms.loading import LoadAnnotations
    _orig_load_masks = LoadAnnotations._load_masks

    def _gpu_load_masks(self, results):
        """GPU-accelerated mask loading."""
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)

        if not self.poly2mask and not any(isinstance(m, list) for m in gt_masks):
            # RLE masks — fall through to original (already lazy)
            return _orig_load_masks(self, results)

        # Check if there are polygons to render
        has_poly = False
        for m in gt_masks:
            if isinstance(m, list) and len(m) > 0 and len(m[0]) >= 6:
                has_poly = True
                break

        if has_poly and len(gt_masks) > 50:
            # Use GPU batch render
            from mmdet.structures.mask import BitmapMasks
            polys = []
            for m in gt_masks:
                if isinstance(m, list) and len(m) > 0 and len(m[0]) >= 6:
                    polys.append(m[0].reshape(-1, 2).astype(np.float32))
                else:
                    polys.append(np.zeros((0, 2), dtype=np.float32))

            try:
                masks_arr = polygons_to_masks_gpu(polys, h, w)
                # Only set valid masks; fall through for others
                valid_idx = [i for i, p in enumerate(polys) if len(p) > 2]
                gt_masks_obj = BitmapMasks(masks_arr[valid_idx], h, w)
                # Reconstruct with valid masks
                full_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(len(polys))]
                for vi, fi in enumerate(valid_idx):
                    full_masks[fi] = masks_arr[vi]
                gt_masks_obj = BitmapMasks(
                    np.stack([m for m in full_masks]), h, w)
                results['gt_masks'] = gt_masks_obj
                return
            except Exception:
                pass  # fall through to CPU

        return _orig_load_masks(self, results)

    LoadAnnotations._load_masks = _gpu_load_masks
    print('[polygon_mask_gpu] LoadAnnotations._load_masks patched with GPU renderer')
