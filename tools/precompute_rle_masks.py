#!/usr/bin/env python
"""Precompute RLE masks for all COCO annotations to accelerate data loading.

Polygon→mask conversion is the dominant bottleneck in data loading (4.6s/iter
with 6000+ polygons per 2.5x image).  This script converts all polygon
segmentations to COCO-compatible RLE format, which mmdet's PolygonMasks
decodes natively at ~10× the speed.

Usage:
    python tools/precompute_rle_masks.py [--coco-root data/syn_multimag/coco]
"""

import argparse
import json
import os
import time
import numpy as np
import cv2
from pycocotools import mask as mask_utils


def polygon_to_rle(polygon_flat, height, width):
    """Convert COCO flat polygon [x1,y1,x2,y2,...] to RLE dict."""
    pts = np.array(polygon_flat, dtype=np.float32).reshape(-1, 1, 2)
    pts_int = pts.astype(np.int32)
    bitmap = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(bitmap, [pts_int], 1)
    rle = mask_utils.encode(np.asfortranarray(bitmap))
    # Convert bytes counts to string for JSON serialization
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def rle_to_coco_seg(rle):
    """Convert RLE dict to COCO segmentation format."""
    return {
        'size': rle['size'],
        'counts': rle['counts'],
    }


def main():
    parser = argparse.ArgumentParser(description='Precompute RLE masks for COCO dataset')
    parser.add_argument('--coco-root', default='./data/syn_multimag/coco',
                        help='COCO dataset root directory')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--max-annotations', type=int, default=None,
                        help='Only process first N annotations (for testing)')
    args = parser.parse_args()

    splits = ['train', 'val', 'test']
    annotations_dir = os.path.join(args.coco_root, 'annotations')

    for split in splits:
        json_path = os.path.join(annotations_dir, f'instances_{split}.json')
        out_path = os.path.join(annotations_dir, f'instances_{split}_rle.json')

        if not os.path.exists(json_path):
            print(f'[Skip] {json_path} not found')
            continue

        # Skip 20x-only files
        if '20xonly' in json_path:
            continue

        print(f'[Loading] {json_path} ...')
        t0 = time.time()
        with open(json_path) as f:
            coco = json.load(f)
        print(f'  Loaded in {time.time() - t0:.1f}s')

        images_by_id = {img['id']: img for img in coco['images']}
        n_total = len(coco['annotations'])
        if args.max_annotations:
            coco['annotations'] = coco['annotations'][:args.max_annotations]
            n_total = len(coco['annotations'])

        print(f'  Converting {n_total} annotations to RLE ...')
        t1 = time.time()
        n_converted = 0
        n_skipped = 0

        for i, ann in enumerate(coco['annotations']):
            if i % 10000 == 0:
                elapsed = time.time() - t1
                rate = (i + 1) / max(elapsed, 0.1)
                eta = (n_total - i) / max(rate, 1)
                print(f'    {i}/{n_total} ({100*i/n_total:.1f}%) '
                      f'rate={rate:.0f}/s eta={eta:.1f}s')

            seg = ann.get('segmentation', [])

            # Skip if already RLE (dict with 'counts' key)
            if isinstance(seg, dict) and 'counts' in seg:
                n_skipped += 1
                continue

            # Skip empty
            if not seg or len(seg) == 0:
                n_skipped += 1
                continue

            # Get image dimensions
            img_id = ann['image_id']
            img_info = images_by_id.get(img_id)
            if img_info is None:
                n_skipped += 1
                continue

            h, w = img_info['height'], img_info['width']

            try:
                # seg is a list of polygons: [[x1,y1,x2,y2,...], [x1,y1,...], ...]
                # Usually there is only one polygon per annotation
                if isinstance(seg, list):
                    if len(seg) == 1 and isinstance(seg[0], list):
                        # Single polygon: [[x1,y1,...]]
                        rle = polygon_to_rle(seg[0], h, w)
                    elif len(seg) > 0 and isinstance(seg[0], (int, float)):
                        # Flat polygon already: [x1,y1,x2,y2,...]
                        rle = polygon_to_rle(seg, h, w)
                    elif len(seg) > 1 and isinstance(seg[0], list):
                        # Multiple polygons: merge
                        bitmap = np.zeros((h, w), dtype=np.uint8)
                        for poly in seg:
                            pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
                            cv2.fillPoly(bitmap, [pts.astype(np.int32)], 1)
                        rle = mask_utils.encode(np.asfortranarray(bitmap))
                        rle['counts'] = rle['counts'].decode('ascii')
                    else:
                        n_skipped += 1
                        continue

                    ann['segmentation'] = {
                        'size': rle['size'],
                        'counts': rle['counts'],
                    }
                    n_converted += 1
                else:
                    n_skipped += 1
            except Exception as e:
                print(f'    Warning: annotation {ann["id"]}: {e}')
                n_skipped += 1

        elapsed = time.time() - t1
        print(f'  Converted {n_converted}, skipped {n_skipped} in {elapsed:.1f}s '
              f'({n_converted/max(elapsed,0.1):.0f}/s)')

        # Save
        print(f'  Saving to {out_path} ...')
        t2 = time.time()
        with open(out_path, 'w') as f:
            json.dump(coco, f, ensure_ascii=False)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f'  Saved {size_mb:.1f} MB in {time.time() - t2:.1f}s')

    print('\n=== Done ===')
    print('Now update your config:')
    print('  ann_file="annotations/instances_train_rle.json"')


if __name__ == '__main__':
    main()
