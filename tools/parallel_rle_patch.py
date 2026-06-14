"""Monkey-patch PolygonMasks.to_bitmap() to use parallel RLE decode.

pycocotools.mask.decode releases the GIL, so ThreadPoolExecutor gives
near-linear speedup for images with thousands of RLE masks.

Usage:
    python -c "import tools.parallel_rle_patch"  # before training
Or import at top of train.py.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor

_pool = None

def _get_pool(max_workers=8):
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=max_workers)
    return _pool


def _patched_to_bitmap(self):
    """Parallel version of PolygonMasks.to_bitmap()."""
    from mmdet.structures.mask.structures import PolygonMasks
    import pycocotools.mask as maskUtils

    masks = self.masks  # List[RLE dict]
    n = len(masks)

    if n == 0:
        return np.empty((0, self.height, self.width), dtype=np.uint8)

    if n < 50:
        # Small count: serial is faster (no thread overhead)
        return np.stack([maskUtils.decode(m) for m in masks])

    # Large count: decode in parallel
    pool = _get_pool()
    decoded = list(pool.map(maskUtils.decode, masks))

    return np.stack(decoded)


def _patched_pad(self, out_shape, pad_val=0.0):
    """Parallel version of PolygonMasks.pad to pad_gt_masks faster."""
    bitmaps = self.to_bitmap()
    padded_masks = []
    pad_h = max(0, out_shape[0] - bitmaps.shape[1])
    pad_w = max(0, out_shape[1] - bitmaps.shape[2])
    for bm in bitmaps:
        if pad_h > 0 or pad_w > 0:
            bm = np.pad(bm, ((0, pad_h), (0, pad_w)), mode='constant',
                       constant_values=pad_val)
        padded_masks.append(bm)
    return np.stack(padded_masks)


def apply_patch():
    """Apply the parallel RLE decode patch."""
    from mmdet.structures.mask.structures import PolygonMasks
    PolygonMasks.to_bitmap = _patched_to_bitmap
    PolygonMasks.pad = _patched_pad
    print('[parallel_rle_patch] PolygonMasks.to_bitmap patched (ThreadPool, 8 workers)')
    return True


# Auto-apply on import
apply_patch()
