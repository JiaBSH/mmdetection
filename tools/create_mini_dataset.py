#!/usr/bin/env python3
"""Create a mini dataset (few images per split) from dataset_1024 for quick validation."""
import json
import shutil
import random
from pathlib import Path

SRC = Path("dataset_root/dataset_1024")
DST = Path("dataset_root/dataset_mini")
SIZES = {"train": 4, "val": 2, "test": 2}
SEED = 42


def main():
    random.seed(SEED)
    for split, n in SIZES.items():
        ann_src = SRC / "annotations" / f"instances_{split}.json"
        with open(ann_src) as f:
            coco = json.load(f)

        images = coco["images"][:]
        random.shuffle(images)
        sel_images = images[:n]
        sel_ids = {img["id"] for img in sel_images}
        sel_anns = [a for a in coco["annotations"] if a["image_id"] in sel_ids]

        img_dst_dir = DST / "images" / split
        img_dst_dir.mkdir(parents=True, exist_ok=True)
        for img in sel_images:
            src_f = SRC / "images" / split / img["file_name"]
            dst_f = img_dst_dir / img["file_name"]
            if src_f.exists():
                shutil.copy2(src_f, dst_f)
            else:
                print(f"WARNING: {src_f} not found")

        ann_dst_dir = DST / "annotations"
        ann_dst_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "categories": coco["categories"],
            "images": sel_images,
            "annotations": sel_anns,
        }
        with open(ann_dst_dir / f"instances_{split}.json", "w") as f:
            json.dump(out, f, ensure_ascii=False)

        print(f"{split}: {len(sel_images)} images, {len(sel_anns)} annotations")

    print(f"\nMini dataset created at: {DST}")


if __name__ == "__main__":
    main()
