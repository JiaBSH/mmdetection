#!/usr/bin/env python
"""Convert all polygon segmentations to RLE in COCO JSONs.

pycocotools.frPyObjects is implemented in C — much faster than cv2.fillPoly.
RLE compresses sparse masks ~100-1000× vs raw bitmaps.

mmdet's PolygonMasks handles RLE natively in the training pipeline.
No code changes needed — just point to the *_rle.json annotation files.

Usage:
    python tools/convert_polygons_to_rle.py
"""

import json
import os
import sys
import time
from multiprocessing import Pool
import numpy as np
from pycocotools import mask as mask_utils

COCO_ROOT = './data/syn_multimag/coco'
SPLITS = ['train', 'val', 'test']


def convert_image_annotations(args):
    """Convert all annotations for one image to RLE.  Runs in subprocess."""
    img_info, annotations = args
    h, w = img_info['height'], img_info['width']
    updated = []
    skipped = 0

    for ann in annotations:
        seg = ann.get('segmentation', [])
        if isinstance(seg, dict) and 'counts' in seg:
            # Already RLE
            updated.append(ann)
            continue
        if not seg or not isinstance(seg, list) or len(seg) == 0:
            updated.append(ann)
            skipped += 1
            continue

        try:
            # pycocotools expects polygons as list of polygon points
            # COCO format: [[x1,y1,x2,y2,...]] or [x1,y1,x2,y2,...]
            rle = mask_utils.frPyObjects(seg, h, w)
            if isinstance(rle, list):
                rle = rle[0]
            # Convert bytes counts to base64 string for JSON serialization
            if isinstance(rle['counts'], bytes):
                rle['counts'] = rle['counts'].decode('ascii')
            ann_new = dict(ann)
            ann_new['segmentation'] = rle
            updated.append(ann_new)
        except Exception:
            updated.append(ann)
            skipped += 1

    return updated, skipped


def main():
    for split in SPLITS:
        json_path = os.path.join(COCO_ROOT, 'annotations', f'instances_{split}.json')
        out_path = os.path.join(COCO_ROOT, 'annotations', f'instances_{split}_rle.json')

        if not os.path.exists(json_path):
            continue
        if '20xonly' in json_path:
            continue

        print(f'[{split}] Loading {json_path}...')
        t0 = time.time()
        with open(json_path) as f:
            coco = json.load(f)

        n_anns = len(coco['annotations'])
        n_imgs = len(coco['images'])
        print(f'[{split}] {n_imgs} images, {n_anns} annotations ({time.time()-t0:.1f}s)')

        # Group annotations by image
        anns_by_image = {}
        for ann in coco['annotations']:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        img_by_id = {img['id']: img for img in coco['images']}
        tasks = [(img_by_id[img_id], anns) for img_id, anns in anns_by_image.items()]

        print(f'[{split}] Converting {len(tasks)} images with 8 workers...')
        t1 = time.time()

        all_updated = []
        total_skipped = 0

        with Pool(8) as pool:
            for updated, skipped in pool.imap_unordered(convert_image_annotations, tasks):
                all_updated.extend(updated)
                total_skipped += skipped

        elapsed = time.time() - t1

        # Sort by original ID to maintain order
        all_updated.sort(key=lambda a: a['id'])

        coco['annotations'] = all_updated
        n_rle = sum(1 for a in all_updated if isinstance(a.get('segmentation'), dict))

        print(f'[{split}] {n_rle} RLE, {total_skipped} skipped in {elapsed:.1f}s')

        # Serialize efficiently
        print(f'[{split}] Writing {out_path}...')
        t2 = time.time()
        with open(out_path, 'w') as f:
            json.dump(coco, f, ensure_ascii=False)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f'[{split}] {size_mb:.1f} MB in {time.time()-t2:.1f}s')

    print('\nDone!')
    print('Training config: ann_file=annotations/instances_train_rle.json')


if __name__ == '__main__':
    main()
