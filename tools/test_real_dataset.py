"""Test best models on real microscope dataset with adaptive sliding window.

Usage:
    python tools/test_real_dataset.py \
        --config work_dirs/ablation_m2_multimag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
        --checkpoint work_dirs/ablation_m2_multimag/epoch_5.pth \
        --dataset-root dataset_root/mmdata_test \
        --scale-model data/syn_multimag/scale_pipeline_dinov2.joblib \
        --out-dir work_dirs/real_test_results/M2_adaptive

    # Also run baseline M1 (no SW):
    python tools/test_real_dataset.py \
        --config work_dirs/ablation_m1_single_mag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
        --checkpoint work_dirs/ablation_m1_single_mag/epoch_5.pth \
        --dataset-root dataset_root/mmdata_test \
        --no-adaptive \
        --out-dir work_dirs/real_test_results/M1_noSW
"""
import sys, os, json, csv, argparse, traceback
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # real images are 4908x3264, some SR are 19632x13056

# Add module paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'Microscope_Magnification_Identification', 'src'))

from postprocess.adaptive_scale import AdaptiveWindowPredictor
from postprocess.run_postprocess import (
    _load_model, _infer_one_image, _build_overlay,
    _save_sliding_window_visualization,
)
from postprocess.coco_utils import load_coco_gt_polygons
import cv2


# Real dataset magnification directories
REAL_MAG_DIRS = ['2_5x_unsup', '5x_unsup', '20x', '50x', '100x']

# Category filter: our model was trained on category 1 (畴区) only
# Real data has cats 1=畴区, 2=凸包 (and 100x has 3=scale_text, 4=scale_bar)
FILTER_CATEGORY_ID = 1


def filter_coco_annotations(coco_data: dict, cat_id: int = 1) -> dict:
    """Return a copy of COCO data with only annotations of the given category."""
    import copy
    filtered = copy.deepcopy(coco_data)
    filtered['annotations'] = [a for a in filtered['annotations']
                                if a.get('category_id') == cat_id]
    filtered['categories'] = [c for c in filtered.get('categories', [])
                               if c.get('id') == cat_id]
    return filtered


def create_filtered_coco(dataset_root: str, mag_dir: str, tmp_dir: str) -> str:
    """Create a filtered COCO JSON (cat 1 only) for a magnification directory.
    Returns path to the filtered JSON file.
    """
    coco_path = os.path.join(dataset_root, mag_dir, f'instances_test_{mag_dir}.json')
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    filtered = filter_coco_annotations(coco, FILTER_CATEGORY_ID)

    os.makedirs(tmp_dir, exist_ok=True)
    out_path = os.path.join(tmp_dir, f'instances_test_{mag_dir}_filtered.json')
    with open(out_path, 'w') as f:
        json.dump(filtered, f, ensure_ascii=False)

    n_orig = len(coco.get('annotations', []))
    n_filt = len(filtered['annotations'])
    print(f"  {mag_dir}: {n_orig} → {n_filt} annotations (kept cat_id={FILTER_CATEGORY_ID})")
    return out_path


def fast_pixel_metrics(instances: list, gt_coco_ann_file: str, img_name: str,
                       img_w: int, img_h: int, return_masks: bool = False):
    """Compute IoU/F1 at pixel level using cv2.fillPoly (fast, no geometric analysis).

    Returns dict with keys: iou, precision, recall, f1, pred_count, gt_count,
    pred_coverage, gt_coverage.
    If return_masks=True, also returns (metrics_dict, pred_mask, gt_mask).
    """
    # Rasterize predicted instances
    pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    poly_count = 0
    for inst in instances:
        coords = inst.get('coords')
        if coords is None or len(coords) == 0:
            continue
        # coords is (N, 2) with (row, col) format
        pts = coords.astype(np.int32)
        # cv2.fillPoly expects (col, row) i.e. (x, y)
        pts_xy = pts[:, ::-1].reshape(-1, 1, 2)
        cv2.fillPoly(pred_mask, [pts_xy], 1)
        poly_count += 1

    # Load GT polygons and rasterize
    gt_polygons, _, _ = load_coco_gt_polygons(gt_coco_ann_file, image_filename=img_name)

    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for poly_rc in gt_polygons:
        # poly_rc is (N, 2) with (row, col) format
        pts_xy = poly_rc[:, ::-1].reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(gt_mask, [pts_xy], 1)

    # Compute pixel-level metrics
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    pred_pixels = pred_mask.sum()
    gt_pixels = gt_mask.sum()

    tp = float(intersection)
    fp = float(pred_pixels - intersection)
    fn = float(gt_pixels - intersection)

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    pred_coverage = pred_pixels / (img_w * img_h) if img_w * img_h > 0 else 0.0
    gt_coverage = gt_pixels / (img_w * img_h) if img_w * img_h > 0 else 0.0

    result = {
        'iou': iou, 'precision': precision, 'recall': recall, 'f1': f1,
        'pred_count': poly_count, 'gt_count': len(gt_polygons),
        'pred_coverage': pred_coverage, 'gt_coverage': gt_coverage,
    }
    if return_masks:
        return result, pred_mask, gt_mask
    return result


def _save_visualizations(pil_img, instances, pred_mask, gt_mask, out_dir, img_name, metrics):
    """Save prediction overlay and IoU visualization images."""
    import random
    from PIL import ImageDraw

    W, H = pil_img.size
    img_np = np.array(pil_img)

    # 1) Prediction overlay: color each instance differently
    overlay = img_np.copy()
    for i, inst in enumerate(instances):
        coords = inst.get('coords')
        if coords is None or len(coords) == 0:
            continue
        random.seed(i)
        color = [random.randint(50, 255) for _ in range(3)]
        pts = coords.astype(np.int32)
        pts_xy = pts[:, ::-1].reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts_xy], isClosed=True, color=color, thickness=2)
    overlay_img = Image.fromarray(overlay)
    overlay_img.save(os.path.join(out_dir, 'pred_overlay.png'))

    # 2) IoU visualization: green=TP, red=FP, blue=FN
    tp = pred_mask & gt_mask
    fp = pred_mask & ~gt_mask
    fn = ~pred_mask & gt_mask

    iou_viz = img_np.copy()
    # Green overlay for TP
    iou_viz[tp > 0] = (iou_viz[tp > 0] * 0.5 + [0, 255, 0] * 0.5).astype(np.uint8)
    # Red overlay for FP
    iou_viz[fp > 0] = (iou_viz[fp > 0] * 0.5 + [255, 0, 0] * 0.5).astype(np.uint8)
    # Blue overlay for FN
    iou_viz[fn > 0] = (iou_viz[fn > 0] * 0.5 + [0, 0, 255] * 0.5).astype(np.uint8)

    viz_img = Image.fromarray(iou_viz)
    viz_img.save(os.path.join(out_dir, 'iou_visualization.png'))

    # 3) Save per-image metrics CSV
    import csv
    metrics_csv = os.path.join(out_dir, 'metrics.csv')
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def run_real_dataset_test(
    config: str,
    checkpoint: str,
    dataset_root: str,
    out_dir: str,
    *,
    scale_model_path: str | None = None,
    score_thresh: float = 0.5,
    min_pixels: int = 10,
    device: str = "cuda:0",
    overlap_ratio: float = 0.2,
    skip_sr: bool = True,
):
    """Test on all real dataset directories.

    Args:
        config: mmdet config path
        checkpoint: model checkpoint path
        dataset_root: path to mmdata_test/
        out_dir: output directory for results
        scale_model_path: if set, use DINOv2 adaptive windows
        skip_sr: skip super-resolution directories (sr2_5x_unsup, sr5x_unsup)
    """
    # Load model
    print(f"Loading model: {config}")
    model = _load_model(config, checkpoint, device=device)

    # Load DINOv2 pipeline if needed
    adaptive_predictor = None
    if scale_model_path:
        print(f"Loading DINOv2 scale pipeline: {scale_model_path}")
        adaptive_predictor = AdaptiveWindowPredictor(scale_model_path)

    # Create filtered COCO JSONs in tmp dir
    tmp_dir = os.path.join(out_dir, '.tmp_filtered_coco')
    os.makedirs(tmp_dir, exist_ok=True)

    mag_dirs = REAL_MAG_DIRS[:]
    if not skip_sr:
        mag_dirs += ['sr2_5x_unsup', 'sr5x_unsup']

    filtered_coco_map = {}
    print("\nFiltering COCO annotations (category 1 only):")
    for mag_dir in mag_dirs:
        filtered_coco_map[mag_dir] = create_filtered_coco(dataset_root, mag_dir, tmp_dir)

    # Process each magnification directory
    all_rows = []
    for mag_dir in mag_dirs:
        print(f"\n{'='*70}")
        print(f"Processing: {mag_dir}")
        print(f"{'='*70}")

        img_dir = os.path.join(dataset_root, mag_dir, 'image')
        ann_file = filtered_coco_map[mag_dir]

        # Get image list from COCO
        with open(ann_file, 'r') as f:
            coco = json.load(f)
        images = coco.get('images', [])
        print(f"  {len(images)} images, {len(coco['annotations'])} GT annotations")

        for img_info in images:
            img_name = img_info['file_name']
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, os.path.basename(img_name))
            if not os.path.exists(img_path):
                print(f"  ⚠️ Image not found: {img_path}")
                continue

            stem = os.path.splitext(os.path.basename(img_name))[0]
            img_out_dir = os.path.join(out_dir, mag_dir, stem)
            os.makedirs(img_out_dir, exist_ok=True)

            # Determine window strategy
            if adaptive_predictor is not None:
                s, mag_pred, frac = adaptive_predictor.predict(img_path)
                # Compute actual window from image dimensions
                pil_img = Image.open(img_path)
                w, h = pil_img.size
                window = adaptive_predictor.predict_window(img_path, w, h)
                if window == 0:
                    sliding_window = False
                    patch_size = 1024  # unused
                    window_note = f"noSW (frac={frac:.3f}×{min(w,h)}={int(frac*min(w,h))} >= {min(w,h)})"
                else:
                    sliding_window = True
                    patch_size = window
                    window_note = f"SW frac={frac:.3f} window={window} (img={w}×{h})"
                print(f"\n  [{mag_dir}/{img_name}] scale={s:.4f} → mag={mag_pred} → {window_note}")
            else:
                sliding_window = False
                patch_size = 1024
                window_note = "noSW (baseline)"
                print(f"\n  [{mag_dir}/{img_name}] {window_note}")

            # Inference
            try:
                instances, pil_img, windows, merge_records = _infer_one_image(
                    model,
                    img_path,
                    score_thresh=score_thresh,
                    target_label=0,  # model only predicts class 0 (single class)
                    min_pixel_count=min_pixels,
                    device=device,
                    sliding_window=sliding_window,
                    patch_size=patch_size,
                    patch_overlap_ratio=overlap_ratio,
                    batch_size=1,
                )
            except Exception:
                tb = traceback.format_exc()
                print(f"  ❌ Inference failed: {tb[:300]}")
                all_rows.append({
                    "magnification": mag_dir,
                    "image": img_name,
                    "window_strategy": window_note,
                    "iou": float("nan"), "f1": float("nan"),
                    "precision": float("nan"), "recall": float("nan"),
                    "pred_count": float("nan"), "gt_count": float("nan"),
                    "pred_coverage": float("nan"), "gt_coverage": float("nan"),
                })
                continue


            print(f"  Predicted instances: {len(instances)}")
            img_w, img_h = pil_img.size

            # Fast pixel-level evaluation
            try:
                metrics, pred_mask, gt_mask = fast_pixel_metrics(
                    instances, ann_file, img_name, img_w, img_h, return_masks=True)
            except Exception:
                tb = traceback.format_exc()
                print(f"  ❌ Evaluation failed: {tb[:300]}")
                metrics = {
                    'iou': float('nan'), 'f1': float('nan'),
                    'precision': float('nan'), 'recall': float('nan'),
                    'pred_count': len(instances), 'gt_count': 0,
                    'pred_coverage': float('nan'), 'gt_coverage': float('nan'),
                }
                pred_mask = None
                gt_mask = None

            # Save visualizations
            if pred_mask is not None and gt_mask is not None:
                try:
                    _save_visualizations(
                        pil_img, instances, pred_mask, gt_mask,
                        img_out_dir, img_name, metrics)
                except Exception:
                    pass

            row = {
                "magnification": mag_dir,
                "image": img_name,
                "window_strategy": window_note,
                **metrics,
            }
            all_rows.append(row)
            print(f"  ✅ IoU={row['iou']:.4f} F1={row['f1']:.4f} pred={row['pred_count']} gt={row['gt_count']}")

    # Save summary CSV
    summary_csv = os.path.join(out_dir, "metrics_summary.csv")
    fieldnames = ["magnification", "image", "window_strategy",
                  "iou", "precision", "recall", "f1",
                  "pred_count", "gt_count", "pred_coverage", "gt_coverage"]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n✅ Summary saved: {summary_csv}")

    # Per-magnification averages
    print(f"\n{'='*70}")
    print("Per-Magnification Averages")
    print(f"{'='*70}")
    print(f"{'Mag':<18} {'Images':>6} {'Avg IoU':>8} {'Avg F1':>8} {'Avg Pred':>8} {'Avg GT':>8}")
    print("-" * 60)
    for mag_dir in mag_dirs:
        mag_rows = [r for r in all_rows if r['magnification'] == mag_dir]
        valid = [r for r in mag_rows if not np.isnan(r.get('iou', float('nan')))]
        if valid:
            avg_iou = np.mean([r['iou'] for r in valid])
            avg_f1 = np.mean([r['f1'] for r in valid])
            avg_pred = np.mean([r['pred_count'] for r in valid])
            avg_gt = np.mean([r['gt_count'] for r in valid])
            print(f"{mag_dir:<18} {len(valid):>6} {avg_iou:8.4f} {avg_f1:8.4f} {avg_pred:8.1f} {avg_gt:8.1f}")

    # Overall
    valid = [r for r in all_rows if not np.isnan(r.get('iou', float('nan')))]
    if valid:
        avg_iou = np.mean([r['iou'] for r in valid])
        avg_f1 = np.mean([r['f1'] for r in valid])
        print(f"{'OVERALL':<18} {len(valid):>6} {avg_iou:8.4f} {avg_f1:8.4f}")

    return all_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test on real microscope dataset')
    parser.add_argument('--config', required=True, help='mmdet config path')
    parser.add_argument('--checkpoint', required=True, help='model checkpoint path')
    parser.add_argument('--dataset-root', required=True, help='path to mmdata_test/')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument('--scale-model', default=None, help='DINOv2 scale pipeline .joblib')
    parser.add_argument('--no-adaptive', action='store_true', default=False,
                        help='Disable adaptive window (use whole-image inference)')
    parser.add_argument('--score-thresh', type=float, default=0.5)
    parser.add_argument('--min-pixels', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overlap-ratio', type=float, default=0.2)
    parser.add_argument('--include-sr', action='store_true', default=False,
                        help='Include super-resolution directories')
    args = parser.parse_args()

    scale_path = None if args.no_adaptive else args.scale_model
    if not args.no_adaptive and scale_path is None:
        scale_path = 'data/syn_multimag/scale_pipeline_dinov2.joblib'

    run_real_dataset_test(
        config=args.config,
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        scale_model_path=scale_path,
        score_thresh=args.score_thresh,
        min_pixels=args.min_pixels,
        device=args.device,
        overlap_ratio=args.overlap_ratio,
        skip_sr=not args.include_sr,
    )
