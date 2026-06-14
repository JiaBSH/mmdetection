#!/usr/bin/env python
"""Precompute npz-compressed masks to eliminate RLE decode bottleneck.

pycocotools.mask.decode holds the GIL and takes 4.7s for 6000 masks.
np.savez_compressed achieves >100x compression on sparse masks and
np.load decompresses in C (no GIL) at ~200 MB/s.

Usage:
    python tools/precompute_masks_fast.py
"""

import json
import os
import sys
import time
import numpy as np
import cv2
from multiprocessing import Pool

COCO_ROOT = './data/syn_multimag/coco'
TARGET_SIZE = 1024
NUM_WORKERS = 8


def render_one_image(args):
    """Render all masks for one image → single npz file."""
    img_info, annotations, img_dir, output_dir = args

    out_path = os.path.join(output_dir, img_info['file_name'].replace('.png', '.npz'))
    if os.path.exists(out_path):
        return img_info['file_name'], 'skip', 0

    h_orig, w_orig = img_info['height'], img_info['width']
    scale = TARGET_SIZE / max(h_orig, w_orig)
    h_new = int(round(h_orig * scale))
    w_new = int(round(w_orig * scale))

    # Collect polygons
    polys = []
    for ann in annotations:
        seg = ann.get('segmentation', [])
        if not seg or not isinstance(seg, list) or len(seg) == 0:
            continue
        poly = seg[0] if isinstance(seg[0], list) else seg
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = pts[:, 0] * scale
        pts[:, 1] = pts[:, 1] * scale
        polys.append(pts.astype(np.int32))

    if not polys:
        return img_info['file_name'], 0, 0

    # Render all masks using cv2.fillPoly (fast C implementation)
    masks = np.zeros((len(polys), h_new, w_new), dtype=np.bool_)  # bool for max compression
    for i, pts in enumerate(polys):
        cv2.fillPoly(masks[i], [pts.reshape(-1, 1, 2)], 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # np.savez_compressed: zip-level compression. Sparse bool masks compress >1000x.
    np.savez_compressed(out_path, masks=masks)
    size_kb = os.path.getsize(out_path) / 1024
    return img_info['file_name'], len(polys), size_kb


def main():
    annotations_dir = os.path.join(COCO_ROOT, 'annotations')
    images_dir = os.path.join(COCO_ROOT, 'images')

    for split in ['train', 'val']:
        json_path = os.path.join(annotations_dir, f'instances_{split}.json')
        if not os.path.exists(json_path) or '20xonly' in json_path:
            continue

        print(f'[{split}] Loading {json_path}...')
        with open(json_path) as f:
            coco = json.load(f)

        anns_by_image = {}
        for ann in coco['annotations']:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        output_dir = os.path.join(COCO_ROOT, 'masks_npz', split)
        img_dir = os.path.join(images_dir, split)

        tasks = [(img, anns_by_image.get(img['id'], []), img_dir, output_dir)
                 for img in coco['images'] if img['id'] in anns_by_image]

        print(f'[{split}] Rendering {len(tasks)} images...')
        t0 = time.time()

        with Pool(NUM_WORKERS) as pool:
            results = list(pool.imap_unordered(render_one_image, tasks))

        elapsed = time.time() - t0
        n_done = sum(1 for r in results if r[1] != 'skip')
        n_masks = sum(r[1] for r in results if isinstance(r[1], int))
        total_kb = sum(r[2] for r in results if len(r) > 2 and isinstance(r[2], (int, float)))
        print(f'[{split}] {n_done} imgs, {n_masks} masks in {elapsed:.1f}s')
        print(f'[{split}] Total size: {total_kb/1024:.1f} MB ({total_kb/(n_done+1):.0f} KB/img)')

    print('\nDone!')
    print('Usage in training: np.load(path)[\"masks\"] → uint8 [N,H,W] array')


if __name__ == '__main__':
    main()
