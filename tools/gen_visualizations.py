"""Generate prediction overlay and IoU visualization images for real dataset results."""
import sys, os, csv, random
sys.path.insert(0, '.')
sys.path.insert(0, 'Microscope_Magnification_Identification/src')

import numpy as np
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from postprocess.run_postprocess import _load_model, _infer_one_image
from postprocess.adaptive_scale import AdaptiveWindowPredictor
from postprocess.coco_utils import load_coco_gt_polygons
from tools.test_real_dataset import fast_pixel_metrics, create_filtered_coco


def gen_visualizations(config, checkpoint, dataset_root, out_base, scale_model):
    model = _load_model(config, checkpoint, device='cuda:0')
    p = AdaptiveWindowPredictor(scale_model)

    tmp_dir = os.path.join(out_base, '.tmp_filtered_coco')
    os.makedirs(tmp_dir, exist_ok=True)

    test_cases = [
        ('2_5x_unsup', '2.5x-1.png'),
        ('5x_unsup', '5x-1.png'),
        ('20x', '20x-1.png'),
        ('50x', '50x-1.png'),
        ('100x', '100x-1.png'),
    ]

    for mag_dir, img_name in test_cases:
        print(f'\n=== {mag_dir}/{img_name} ===')

        filtered_coco = create_filtered_coco(dataset_root, mag_dir, tmp_dir)
        img_path = os.path.join(dataset_root, mag_dir, 'image', img_name)

        s, mag_pred, frac = p.predict(img_path)
        pil_img = Image.open(img_path)
        w, h = pil_img.size
        window = p.predict_window(img_path, w, h)

        sliding_window = window != 0
        patch_size = window if sliding_window else 1024
        print(f'  scale={s:.4f} mag={mag_pred} window={window} SW={sliding_window}')

        instances, pil_img, windows, merge_records = _infer_one_image(
            model, img_path, score_thresh=0.5, target_label=0, min_pixel_count=10,
            device='cuda:0', sliding_window=sliding_window, patch_size=patch_size,
            patch_overlap_ratio=0.2, batch_size=1)
        print(f'  Pred instances: {len(instances)}')

        metrics, pred_mask, gt_mask = fast_pixel_metrics(
            instances, filtered_coco, img_name, w, h, return_masks=True)
        print(f'  IoU={metrics["iou"]:.4f} F1={metrics["f1"]:.4f}')

        stem = os.path.splitext(img_name)[0]
        out_dir = os.path.join(out_base, mag_dir, stem)
        os.makedirs(out_dir, exist_ok=True)

        img_np = np.array(pil_img)

        # 1) Prediction overlay — colored polygon outlines
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
        Image.fromarray(overlay).save(os.path.join(out_dir, 'pred_overlay.png'))

        # 2) GT overlay — colored polygon outlines
        gt_overlay = img_np.copy()
        gt_polygons, _, _ = load_coco_gt_polygons(filtered_coco, image_filename=img_name)
        for i, poly_rc in enumerate(gt_polygons):
            random.seed(i)
            color = [random.randint(50, 255) for _ in range(3)]
            pts_xy = poly_rc[:, ::-1].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(gt_overlay, [pts_xy], isClosed=True, color=color, thickness=2)
        Image.fromarray(gt_overlay).save(os.path.join(out_dir, 'gt_overlay.png'))

        # 3) IoU visualization — green=TP, red=FP, blue=FN
        tp = pred_mask & gt_mask
        fp = pred_mask & ~gt_mask
        fn = ~pred_mask & gt_mask
        iou_viz = img_np.copy().astype(np.float32)
        iou_viz[tp > 0] = iou_viz[tp > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        iou_viz[fp > 0] = iou_viz[fp > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        iou_viz[fn > 0] = iou_viz[fn > 0] * 0.5 + np.array([0, 0, 255]) * 0.5
        Image.fromarray(iou_viz.astype(np.uint8)).save(os.path.join(out_dir, 'iou_visualization.png'))

        # 4) Binary mask overlay — pred vs GT side by side
        mask_viz = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
        mask_viz[:, :w, 1] = (pred_mask * 255).astype(np.uint8)   # Pred in green
        mask_viz[:, :w, 2] = (gt_mask * 255).astype(np.uint8)     # Pred border
        mask_viz[:, w + 10:, 2] = (gt_mask * 255).astype(np.uint8)  # GT in red
        # White border
        mask_viz[:, w:w + 10] = 255
        Image.fromarray(mask_viz).save(os.path.join(out_dir, 'mask_comparison.png'))

        # 5) Per-image metrics CSV
        with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)

    print(f'\nDone! Visualizations saved to {out_base}')


if __name__ == '__main__':
    gen_visualizations(
        config='work_dirs/run_20260606_231505/condinst_r50_fpn_custom_coco_instance/condinst_r50_fpn_custom_coco_instance.py',
        checkpoint='work_dirs/run_20260606_231505/condinst_r50_fpn_custom_coco_instance/epoch_76.pth',
        dataset_root='dataset_root/mmdata_test',
        out_base='work_dirs/real_test_results/condinst_scratch_overlap',
        scale_model='data/syn_multimag/scale_pipeline_dinov2.joblib',
    )
