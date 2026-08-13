#!/usr/bin/env python3
"""
Compute semantic segmentation metrics from instance segmentation predictions.

For each model in run_syn_rotation, this script:
  1. Reads per-image instance predictions (RLE masks) from test_vis/preds/
  2. Merges instance masks of the same class → semantic segmentation map
  3. Reads COCO ground-truth annotations, merges → GT semantic map
  4. Computes per-image and dataset-level precision, recall, IoU, F1-score

Output: a TSV summary table + per-model JSON files with detailed metrics.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import glob

import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image

# ---------------------------------------------------------------------------
#  RLE & mask helpers
# ---------------------------------------------------------------------------

def decode_rle(rle_dict: dict) -> np.ndarray:
    """Decode a COCO-compressed RLE dict (size + counts) to a binary mask
    of shape (H, W)."""
    return maskUtils.decode(rle_dict).astype(np.uint8).squeeze()


def ann_seg_to_mask(seg: dict | list, height: int, width: int) -> np.ndarray:
    """Convert a single annotation's segmentation (polygon list or RLE dict)
    to a binary mask of shape (H, W)."""
    if isinstance(seg, dict):
        # already RLE – decode directly
        return maskUtils.decode(seg).astype(np.uint8).squeeze()
    # polygon(s) → RLE → mask
    rle = maskUtils.frPyObjects(seg, height, width)
    return maskUtils.decode(rle).astype(np.uint8).squeeze()


def resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a (H,W) binary mask to (target_h, target_w) via nearest-neighbour."""
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    img = Image.fromarray(mask * 255)
    img = img.resize((target_w, target_h), Image.NEAREST)
    return (np.array(img) > 127).astype(np.uint8)


def merge_masks_to_semantic(masks: list[np.ndarray],
                            ref_shape: tuple = None) -> np.ndarray:
    """Merge a list of binary instance masks into one semantic (binary) mask.
    Pixels belonging to ANY instance become foreground (logical OR)."""
    if not masks:
        if ref_shape is None:
            return np.zeros((1, 1), dtype=np.uint8)
        return np.zeros(ref_shape, dtype=np.uint8)
    semantic = masks[0].copy()
    for m in masks[1:]:
        np.maximum(semantic, m, out=semantic)
    return semantic


# ---------------------------------------------------------------------------
#  Confusion & metrics
# ---------------------------------------------------------------------------

def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """Return (tp, fp, fn, tn) for binary masks."""
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))
    return tp, fp, fn, tn


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict:
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "iou": round(iou, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
#  Instance segmentation metrics (from pre-computed test JSON)
# ---------------------------------------------------------------------------

def get_instance_metrics(model_dir: str) -> dict | None:
    """Extract instance-seg metrics (bbox_mAP, segm_mAP) from the test JSON.

    Returns a dict like {'coco/bbox_mAP': 0.903, 'coco/segm_mAP': 0.949}
    or None if no test JSON is found.
    """
    test_dir = os.path.join(model_dir, "test")
    if not os.path.isdir(test_dir):
        return None
    candidates = glob.glob(os.path.join(test_dir, "*", "*.json"))
    # pick the first JSON that has coco/ keys (skip vis_data subdirs)
    for p in sorted(candidates):
        if "vis_data" in p:
            continue
        try:
            with open(p, "r") as f:
                data = json.load(f)
        except Exception:
            continue
        metrics = {k: v for k, v in data.items() if k.startswith("coco/")}
        if metrics:
            return metrics
    return None


# ---------------------------------------------------------------------------
#  Main evaluation for one model
# ---------------------------------------------------------------------------

def evaluate_model(model_dir: str, ann_file: str) -> dict:
    """Evaluate one model directory.

    Returns a dict with keys:
      - model_name
      - per_image: {filename: metrics_dict}
      - overall: metrics_dict aggregated across all images
    """
    preds_dir = os.path.join(model_dir, "test_vis", "preds")
    if not os.path.isdir(preds_dir):
        print(f"  [WARN] No test_vis/preds/ in {model_dir} – skipping")
        return None

    # --- load COCO GT -------------------------------------------------------
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # Build lookup: file_name → list of annotations
    # Also create: file_name → image info
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    file_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        info = img_id_to_info.get(ann["image_id"])
        if info is None:
            continue
        file_to_anns[info["file_name"]].append(ann)

    # Build: file_name → (height, width)
    file_to_size = {
        img["file_name"]: (img["height"], img["width"])
        for img in coco["images"]
    }

    # --- iterate over prediction files --------------------------------------
    per_image = {}
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    pred_files = sorted(os.listdir(preds_dir))
    for pf in pred_files:
        if not pf.endswith(".json"):
            continue

        # Map prediction filename to COCO image filename
        # pred: "100x_00016.json" → image: "100x_00016.png"
        img_file = pf.replace(".json", ".png")

        if img_file not in file_to_anns:
            print(f"  [WARN] {pf}: no GT for image {img_file} – skip")
            continue

        gt_h, gt_w = file_to_size[img_file]

        # —— GT: merge all instance masks → semantic mask at GT resolution ——
        gt_anns = file_to_anns[img_file]
        gt_masks = []
        for ann in gt_anns:
            gt_masks.append(ann_seg_to_mask(ann["segmentation"], gt_h, gt_w))
        gt_sem = merge_masks_to_semantic(gt_masks)

        # —— Pred: decode RLE masks, resize to GT resolution, merge ————
        with open(os.path.join(preds_dir, pf), "r") as f:
            pred_data = json.load(f)

        pred_masks_raw = []
        for rle_dict in pred_data.get("masks", []):
            mask = decode_rle(rle_dict)
            pred_masks_raw.append(mask)

        # Resize each pred mask to GT size
        pred_masks_resized = [
            resize_mask(m, gt_h, gt_w) for m in pred_masks_raw
        ]
        pred_sem = merge_masks_to_semantic(pred_masks_resized)

        # —— Compute metrics ——————————————————————————————————————
        tp, fp, fn, tn = confusion(pred_sem, gt_sem)
        m = metrics_from_confusion(tp, fp, fn, tn)
        per_image[img_file] = m

        for k in ("tp", "fp", "fn", "tn"):
            total[k] += m[k]

    if not per_image:
        print("  [WARN] No matching images found")
        return None

    overall = metrics_from_confusion(
        total["tp"], total["fp"], total["fn"], total["tn"]
    )

    model_name = os.path.basename(os.path.normpath(model_dir))
    return {
        "model_name": model_name,
        "per_image": per_image,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic-seg metrics from instance predictions"
    )
    parser.add_argument(
        "--work-dir",
        default="/data/home/scvi576/run/JiaBSH/mmdetection_para/work_dirs/run_syn_rotation",
        help="Root directory containing all model subdirs",
    )
    parser.add_argument(
        "--ann-file",
        default="/data/home/scvi576/run/JiaBSH/mmdetection_para/data/syn_multimag/adaptive_patches_rotation/annotations/instances_test.json",
        help="COCO instances_test.json with GT annotations",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Evaluate only this model subdirectory (default: all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for per-model JSON results",
    )
    args = parser.parse_args()

    # --- discover models ----------------------------------------------------
    if args.model:
        model_dirs = [os.path.join(args.work_dir, args.model)]
    else:
        model_dirs = sorted(
            [
                os.path.join(args.work_dir, d)
                for d in os.listdir(args.work_dir)
                if os.path.isdir(os.path.join(args.work_dir, d))
                and os.path.exists(os.path.join(args.work_dir, d, "test_vis", "preds"))
            ]
        )

    if not model_dirs:
        print("No model directories with test_vis/preds/ found.")
        sys.exit(1)

    print(f"Found {len(model_dirs)} model(s) to evaluate.\n")

    # --- evaluate -----------------------------------------------------------
    results = []
    for md in model_dirs:
        name = os.path.basename(md)
        print(f"Evaluating {name} ...")
        r = evaluate_model(md, args.ann_file)
        if r is not None:
            # merge instance-seg metrics from test JSON
            inst = get_instance_metrics(md)
            r["instance"] = inst
            results.append(r)
            o = r["overall"]
            print(f"  semantic: prec={o['precision']:.4f}  rec={o['recall']:.4f}  "
                  f"iou={o['iou']:.4f}  f1={o['f1']:.4f}")
            if inst:
                bbox = inst.get("coco/bbox_mAP")
                segm = inst.get("coco/segm_mAP")
                bbox_s = f"{bbox:.4f}" if bbox is not None else "-"
                segm_s = f"{segm:.4f}" if segm is not None else "-"
                print(f"  instance: bbox_mAP={bbox_s}  segm_mAP={segm_s}")
        else:
            print(f"  FAILED")

    if not results:
        print("\nNo results to report.")
        sys.exit(1)

    # --- save per-model JSON ------------------------------------------------
    out_dir = args.output or os.path.join(args.work_dir, "semantic_metrics")
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        out_path = os.path.join(out_dir, f"{r['model_name']}_semseg.json")
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"\nPer-model JSONs saved to {out_dir}/")

    # --- print TSV summary --------------------------------------------------
    tsv_path = os.path.join(out_dir, "summary.tsv")
    header = [
        "model",
        "bbox_mAP",
        "segm_mAP",
        "sem_precision",
        "sem_recall",
        "sem_iou",
        "sem_f1",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with open(tsv_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in results:
            o = r["overall"]
            inst = r.get("instance") or {}
            bbox_val = inst.get("coco/bbox_mAP") if inst else None
            segm_val = inst.get("coco/segm_mAP") if inst else None
            row = [
                r["model_name"],
                f"{bbox_val:.6f}" if bbox_val is not None else "N/A",
                f"{segm_val:.6f}" if segm_val is not None else "N/A",
                f"{o['precision']:.6f}",
                f"{o['recall']:.6f}",
                f"{o['iou']:.6f}",
                f"{o['f1']:.6f}",
                str(o["tp"]),
                str(o["fp"]),
                str(o["fn"]),
                str(o["tn"]),
            ]
            f.write("\t".join(row) + "\n")
    print(f"Summary TSV saved to {tsv_path}")

    # --- print table to stdout ----------------------------------------------
    print(f"\n{'Model':<51s} {'bbox':>7s} {'segm':>7s} "
          f"{'s-Prec':>7s} {'s-Rec':>7s} {'s-IoU':>7s} {'s-F1':>7s}")
    print("-" * 101)
    for r in results:
        o = r["overall"]
        inst = r.get("instance") or {}
        bbox = f"{inst['coco/bbox_mAP']:.4f}" if inst.get("coco/bbox_mAP") is not None else "-"
        segm = f"{inst['coco/segm_mAP']:.4f}" if inst.get("coco/segm_mAP") is not None else "-"
        print(
            f"{r['model_name']:<51s} "
            f"{bbox:>7s} {segm:>7s} "
            f"{o['precision']:7.4f} {o['recall']:7.4f} "
            f"{o['iou']:7.4f} {o['f1']:7.4f}"
        )


if __name__ == "__main__":
    main()
