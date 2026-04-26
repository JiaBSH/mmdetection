"""Offline image resize for COCO-style instance segmentation datasets.

Resizes all images so that the longest side <= MAX_SIZE (keep aspect ratio),
updates bbox / segmentation / area in the annotation JSON accordingly,
and writes outputs to a new directory without touching the originals.

Usage:
    python tools/preprocess_resize.py \
        --data-root dataset_root \
        --splits train val test \
        --max-size 512 \
        --out-root dataset_root_512 \
        --num-workers 8
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scale_factor(w: int, h: int, max_size: int) -> float:
    return min(max_size / w, max_size / h, 1.0)


def _resize_image(args):
    src_path, dst_path, max_size = args
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path)
    orig_w, orig_h = img.size
    scale = _scale_factor(orig_w, orig_h, max_size)
    if scale < 1.0:
        new_w = round(orig_w * scale)
        new_h = round(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    img.save(dst_path)
    # Return both original PIL size and new size so caller can derive the
    # exact scale factor actually applied (independent of JSON metadata).
    return str(src_path), (orig_w, orig_h), img.size  # (orig, new)


def _scale_bbox(bbox, scale):
    x, y, bw, bh = bbox
    return [x * scale, y * scale, bw * scale, bh * scale]


def _scale_segmentation(seg, scale):
    """Scale polygon segmentation (list of [x,y,x,y,...] lists)."""
    scaled = []
    for poly in seg:
        scaled.append([c * scale for c in poly])
    return scaled


def process_split(split: str, data_root: Path, out_root: Path, max_size: int,
                  num_workers: int):
    ann_src = data_root / "annotations" / f"instances_{split}.json"
    img_src_dir = data_root / "images" / split

    ann_dst = out_root / "annotations" / f"instances_{split}.json"
    img_dst_dir = out_root / "images" / split

    ann_dst.parent.mkdir(parents=True, exist_ok=True)
    img_dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{split}] Loading annotations from {ann_src} ...")
    with open(ann_src) as f:
        coco = json.load(f)

    # Build lookup: src path string -> image_id
    path_to_id = {
        str(img_src_dir / img_info["file_name"]): img_info["id"]
        for img_info in coco["images"]
    }
    id_to_json_size = {img_info["id"]: (img_info["width"], img_info["height"])
                       for img_info in coco["images"]}

    resize_tasks = [
        (img_src_dir / img_info["file_name"],
         img_dst_dir / img_info["file_name"],
         max_size)
        for img_info in coco["images"]
    ]

    # Resize images in parallel
    print(f"[{split}] Resizing {len(resize_tasks)} images "
          f"(max_size={max_size}) with {num_workers} workers ...")
    # actual_sizes: id -> (new_w, new_h) as truly written to disk
    actual_sizes = {}
    # id_to_scale: derived from actual PIL orig size, NOT JSON metadata,
    # so annotation coordinates are scaled by the exact same factor.
    id_to_scale = {}
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futs = {exe.submit(_resize_image, t): t for t in resize_tasks}
        done = 0
        for fut in as_completed(futs):
            src_path, (orig_w, orig_h), (nw, nh) = fut.result()
            img_id = path_to_id.get(src_path)
            if img_id is not None:
                actual_sizes[img_id] = (nw, nh)
                # Use actual PIL dimensions to compute scale, not JSON metadata.
                # With keep_ratio, x-scale == y-scale; use x for safety.
                id_to_scale[img_id] = nw / orig_w if orig_w else 1.0
            done += 1
            if done % 20 == 0 or done == len(resize_tasks):
                print(f"  {done}/{len(resize_tasks)}")

    # Update annotation JSON
    new_images = []
    for img_info in coco["images"]:
        iid = img_info["id"]
        new_info = dict(img_info)
        if iid in actual_sizes:
            new_info["width"], new_info["height"] = actual_sizes[iid]
        new_images.append(new_info)

    new_anns = []
    for ann in coco.get("annotations", []):
        iid = ann["image_id"]
        scale = id_to_scale.get(iid, 1.0)
        new_ann = dict(ann)
        if scale < 1.0:
            new_ann["bbox"] = _scale_bbox(ann["bbox"], scale)
            new_ann["area"] = ann.get("area", 0) * scale * scale
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):  # polygon format
                new_ann["segmentation"] = _scale_segmentation(seg, scale)
            # RLE format does not need coordinate scaling (shape changes below)
        new_anns.append(new_ann)

    new_coco = dict(coco)
    new_coco["images"] = new_images
    new_coco["annotations"] = new_anns

    with open(ann_dst, "w") as f:
        json.dump(new_coco, f)

    print(f"[{split}] Done. Annotations -> {ann_dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="dataset_root/dataset",
                        help="Original dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--max-size", type=int, default=1024,
                        help="Longest side after resize")
    parser.add_argument("--out-root", default="dataset_root/dataset_1024",
                        help="Output dataset root")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    for split in args.splits:
        process_split(split, data_root, out_root, args.max_size,
                      args.num_workers)

    print(f"\nAll splits written to: {out_root}")
    print("Update your config's  data_root  to point to this directory.")


if __name__ == "__main__":
    main()
