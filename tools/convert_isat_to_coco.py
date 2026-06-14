#!/usr/bin/env python
"""ISAT JSON → COCO JSON 批量转换。

将 data/syn_multimag/raw/{mag}/ 目录下的 ISAT 标注转换为
统一的 COCO 格式数据集，按索引切分 train/val/test。

用法: python tools/convert_isat_to_coco.py
"""

import json
import os
import shutil
import cv2
import numpy as np

RAW_ROOT = "./data/syn_multimag/raw"
COCO_ROOT = "./data/syn_multimag/coco"

MAG_LABELS = ["2.5x", "5x", "20x", "50x", "100x"]

TRAIN_PER_MAG = 50
VAL_PER_MAG = 20
TEST_PER_MAG = 30  # 50+20+30 = 100

CATEGORY_ID = 1
CATEGORY_NAME = "畴区"


def isat_polygon_to_coco(segmentation: list) -> list:
    """ISAT [[x,y],...] → COCO flattened [x1,y1,x2,y2,...]"""
    flat = []
    for pt in segmentation:
        flat.extend([round(pt[0], 2), round(pt[1], 2)])
    return flat


def convert_split(mag_label: str, split_name: str, start_idx: int, end_idx: int,
                  images_out: list, annotations_out: list,
                  next_image_id: int, next_ann_id: int,
                  img_dir: str):
    """转换一个倍率的某个 split 的图像。"""
    mag_dir = os.path.join(RAW_ROOT, mag_label)
    label_dir = os.path.join(mag_dir, "label")
    image_dir = os.path.join(mag_dir, "image")
    safe_mag = mag_label.replace(".", "p")

    count = 0
    for idx in range(start_idx, end_idx):
        basename = f"syn_{safe_mag}_{idx:05d}"
        json_path = os.path.join(label_dir, basename + ".json")
        img_path = os.path.join(image_dir, basename + ".png")

        if not os.path.exists(json_path) or not os.path.exists(img_path):
            continue

        with open(json_path) as f:
            isat = json.load(f)

        info = isat["info"]
        width, height = info["width"], info["height"]

        # 复制图像到 COCO 目录
        dst_name = f"{safe_mag}_{idx:05d}.png"
        dst_path = os.path.join(img_dir, dst_name)
        shutil.copy2(img_path, dst_path)

        image_entry = {
            "id": next_image_id,
            "file_name": dst_name,
            "width": width,
            "height": height,
        }
        images_out.append(image_entry)

        for obj in isat.get("objects", []):
            seg_isat = obj.get("segmentation", [])
            if len(seg_isat) < 3:
                continue

            seg_coco = isat_polygon_to_coco(seg_isat)
            bbox_isat = obj.get("bbox", [0, 0, 0, 0])

            # 从 polygon 计算 bbox [x, y, w, h] 和 area
            pts = np.array(seg_isat, dtype=np.float32).reshape(-1, 1, 2)
            x, y, w, h = cv2.boundingRect(pts)
            area = float(cv2.contourArea(pts))
            if area <= 0:
                continue

            annotation_entry = {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": CATEGORY_ID,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": area,
                "segmentation": [seg_coco],
                "iscrowd": 0,
            }
            annotations_out.append(annotation_entry)
            next_ann_id += 1

        next_image_id += 1
        count += 1

    return next_image_id, next_ann_id, count


def main():
    os.makedirs(os.path.join(COCO_ROOT, "annotations"), exist_ok=True)

    splits = {
        "train": (0, TRAIN_PER_MAG),
        "val": (TRAIN_PER_MAG, TRAIN_PER_MAG + VAL_PER_MAG),
        "test": (TRAIN_PER_MAG + VAL_PER_MAG,
                 TRAIN_PER_MAG + VAL_PER_MAG + TEST_PER_MAG),
    }

    for split_name, (start, end) in splits.items():
        img_dir = os.path.join(COCO_ROOT, "images", split_name)
        os.makedirs(img_dir, exist_ok=True)

        images: list = []
        annotations: list = []
        next_image_id = 1
        next_ann_id = 1

        for mag in MAG_LABELS:
            next_image_id, next_ann_id, cnt = convert_split(
                mag, split_name, start, end,
                images, annotations,
                next_image_id, next_ann_id,
                img_dir,
            )
            print(f"  {mag}/{split_name}: {cnt} images")

        coco = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": None}
            ],
        }

        json_path = os.path.join(COCO_ROOT, "annotations",
                                 f"instances_{split_name}.json")
        with open(json_path, "w") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"[Done] {split_name}: {len(images)} images, "
              f"{len(annotations)} annotations → {json_path}")


if __name__ == "__main__":
    main()
