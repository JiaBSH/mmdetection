#!/usr/bin/env python
"""Adaptive-window training data preprocessing.

For each image, determine the optimal window size from magnification,
extract overlapping patches, and resize them to a fixed training resolution.
This ensures consistent physical scale across all magnifications and keeps
instance count per patch manageable (<500).

The window size for each magnification comes from the DINOv2 scale pipeline:
  2.5x → 256, 5x → 512, 20x → 2048, 50x → 5120, 100x → 10240

Images where window >= original size: use the whole image (resized).
Images where window < original size: sliding-window crop into patches.

Output: COCO-format dataset with all patches at the target training resolution.

Usage:
    python tools/prepare_multimag_training.py --target-size 1024 --overlap 0.2
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from copy import deepcopy

# ── magnification → window size (from DINOv2 pipeline) ──
MAG_WINDOW = {
    '2.5x': 256,
    '5x': 512,
    '20x': 2048,
    '50x': 5120,
    '100x': 10240,
}

COCO_ROOT = './data/syn_multimag/coco_rotation'
OUTPUT_ROOT = './data/syn_multimag/adaptive_patches_rotation'
CATEGORY_ID = 1


def mag_from_filename(fname):
    """Extract magnification label from filename, e.g., '2p5x_00000.png' → '2.5x'"""
    parts = fname.split('_')
    mag = parts[0].replace('p', '.')
    return mag


def window_from_mag(mag_label):
    """Get recommended square window size for a magnification."""
    return MAG_WINDOW.get(mag_label, 2048)


def process_image(img_info, annotations, img_dir, splits_map, target_size, overlap, window_jitter=0.0):
    """Process one image: adaptive window → patches.

    For low-mag images (2.5x, 5x) with window < image: 1 random crop per image.
    For mid/high-mag (20x, 50x, 100x): whole image (window >= image size).

    If window_jitter > 0, the crop window size is randomly scaled by ±window_jitter
    to simulate scale estimation errors at inference time.
    """
    mag = mag_from_filename(img_info['file_name'])
    window = window_from_mag(mag)
    H, W = img_info['height'], img_info['width']

    # Apply window size jitter: randomly scale the window by ±window_jitter
    if window_jitter > 0:
        scale_factor = 1.0 + np.random.uniform(-window_jitter, window_jitter)
        window = int(round(window * scale_factor))
        window = max(16, min(window, max(W, H)))  # clamp to image bounds

    img_path = os.path.join(img_dir, img_info['file_name'])
    image = cv2.imread(img_path)
    if image is None:
        print(f'  WARNING: cannot read {img_path}')
        return []

    out_root = splits_map['output_root']
    results = []

    if window >= min(W, H):
        scale = target_size / max(W, H)
        new_w = int(round(W * scale))
        new_h = int(round(H * scale))
        img_resized = cv2.resize(image, (new_w, new_h))

        patch_anns = _transform_annotations(annotations, 0, 0, scale, new_w, new_h)
        if patch_anns:
            patch_name = img_info['file_name']
            out_path = os.path.join(out_root, 'images', splits_map['split'], patch_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, img_resized)
            results.append((patch_name, new_w, new_h, patch_anns))
    else:
        max_x = W - window
        max_y = H - window
        if max_x < 0 or max_y < 0:
            return results
        x = np.random.randint(0, max_x + 1)
        y = np.random.randint(0, max_y + 1)

        crop = image[y:y + window, x:x + window]
        crop_resized = cv2.resize(crop, (target_size, target_size))
        scale = target_size / window

        patch_anns = _transform_annotations(annotations, x, y, scale, target_size, target_size)
        if patch_anns:
            patch_name = img_info['file_name']
            out_path = os.path.join(out_root, 'images', splits_map['split'], patch_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv2.imwrite(out_path, crop_resized)
            results.append((patch_name, target_size, target_size, patch_anns))

    return results


def _transform_annotations(annotations, offset_x, offset_y, scale, out_w, out_h):
    """Transform polygon annotations to patch coordinates."""
    result = []
    for ann in annotations:
        seg = ann.get('segmentation', [])
        if not seg or len(seg) == 0:
            continue
        flat = seg[0] if isinstance(seg[0], list) else seg
        if len(flat) < 6:
            continue
        poly = np.array(flat, dtype=np.float32).reshape(-1, 2)

        poly[:, 0] = (poly[:, 0] - offset_x) * scale
        poly[:, 1] = (poly[:, 1] - offset_y) * scale
        poly[:, 0] = np.clip(poly[:, 0], 0, out_w)
        poly[:, 1] = np.clip(poly[:, 1], 0, out_h)

        x_min, y_min = poly[:, 0].min(), poly[:, 1].min()
        x_max, y_max = poly[:, 0].max(), poly[:, 1].max()
        new_w = x_max - x_min
        new_h = y_max - y_min
        if new_w < 2 or new_h < 2:
            continue

        area = float(cv2.contourArea(poly.reshape(-1, 1, 2).astype(np.float32)))
        if area < 1:
            continue

        new_ann = deepcopy(ann)
        new_ann['bbox'] = [float(x_min), float(y_min), float(new_w), float(new_h)]
        new_ann['area'] = area
        new_ann['segmentation'] = [poly.flatten().tolist()]
        result.append(new_ann)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-size', type=int, default=1024)
    parser.add_argument('--overlap', type=float, default=0.2)
    parser.add_argument('--window-jitter', type=float, default=0.0,
                        help='Window size jitter ratio, e.g. 0.25 = ±25%% random scale')
    parser.add_argument('--output-root', type=str, default=OUTPUT_ROOT,
                        help=f'Output directory (default: {OUTPUT_ROOT})')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    annotations_dir = os.path.join(COCO_ROOT, 'annotations')
    images_dir = os.path.join(COCO_ROOT, 'images')

    for split in args.splits:
        json_path = os.path.join(annotations_dir, f'instances_{split}.json')
        if not os.path.exists(json_path) or '20xonly' in json_path:
            continue

        print(f'[{split}] Loading {json_path}...')
        with open(json_path) as f:
            coco = json.load(f)

        anns_by_image = {}
        for ann in coco['annotations']:
            anns_by_image.setdefault(ann['image_id'], []).append(ann)

        img_dir = os.path.join(images_dir, split)
        splits_map = {'split': split, 'output_root': args.output_root}

        new_images = []
        new_annotations = []
        next_img_id = 1
        next_ann_id = 1

        total = len(coco['images'])
        for idx, img in enumerate(coco['images']):
            if img['id'] not in anns_by_image:
                continue

            if idx % 10 == 0:
                print(f'  [{split}] {idx}/{total}...')

            mag = mag_from_filename(img['file_name'])
            window = window_from_mag(mag)
            patches = process_image(img, anns_by_image[img['id']],
                                    img_dir, splits_map,
                                    args.target_size, args.overlap,
                                    args.window_jitter)

            for pname, pw, ph, panns in patches:
                img_entry = {
                    'id': next_img_id,
                    'file_name': pname,
                    'width': pw,
                    'height': ph,
                }
                new_images.append(img_entry)

                for ann in panns:
                    ann['id'] = next_ann_id
                    ann['image_id'] = next_img_id
                    new_annotations.append(ann)
                    next_ann_id += 1

                next_img_id += 1

        # Save COCO JSON
        out_coco = {
            'images': new_images,
            'annotations': new_annotations,
            'categories': coco['categories'],
        }

        out_dir = os.path.join(args.output_root, 'annotations')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'instances_{split}.json')

        with open(out_path, 'w') as f:
            json.dump(out_coco, f, ensure_ascii=False)

        print(f'[{split}] {len(new_images)} patches, {len(new_annotations)} annotations → {out_path}')

    print(f'\nOutput: {args.output_root}/')
    print(f'  annotations/instances_train.json')
    print(f'  annotations/instances_val.json')
    print(f'  images/train/ and images/val/')
    print(f'\nTraining config:')
    print(f'  train_dataloader.dataset.data_root={args.output_root}')
    print(f'  train_dataloader.dataset.ann_file=annotations/instances_train.json')
    print(f'  train_dataloader.dataset.data_prefix.img=images/train/')


if __name__ == '__main__':
    main()
