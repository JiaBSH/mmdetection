import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import re
import os

# Register Arial font
font_path = "/tmp/Arial.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    matplotlib.rcParams['font.family'] = font_name
else:
    # Fallback to Liberation Sans (metrically identical to Arial)
    matplotlib.rcParams['font.family'] = 'Liberation Sans'

matplotlib.rcParams.update({
    'font.size': 120,
    'axes.titlesize': 144,
    'axes.labelsize': 132,
    'xtick.labelsize': 114,
    'ytick.labelsize': 114,
    'legend.fontsize': 100,
    'lines.linewidth': 10.0,
    'lines.markersize': 16,
})

base_dir = "work_dirs/run_isat"

models = [
    "boxinst_r50_fpn_custom_coco_instance",
    "cascade-mask-rcnn_r50_fpn_1x_custom_coco_instance",
    "condinst_r50_fpn_custom_coco_instance",
    "detectors_htc-r50_custom_coco_instance",
    "htc-without-semantic_r50_fpn_1x_custom_coco_instance",
    "mask-rcnn_r50_fpn_1x_custom_coco_instance",
    "mask2former_r50_custom_coco_instance",
    "mask-rcnn_r50_fpn_instaboost_custom_coco_instance",
    "ms-rcnn_r50-caffe_fpn_1x_custom_coco_instance",
    "point-rend_r50-caffe_fpn_custom_coco_instance",
    "queryinst_r50_fpn_1x_custom_coco_instance",
    "rtmdet-ins_tiny_custom_coco_instance",
    "solo_r50_fpn_1x_custom_coco_instance",
    "solov2_r50_fpn_1x_custom_coco_instance",
    "yolact_r50_custom_coco_instance",
]

display_names = [
    "BoxInst",
    "CM R-CNN",
    "CondInst",
    "DetectoRS",
    "HT Cascade",
    "Mask R-CNN",
    "Mask2Former",
    "InstaBoost",
    "MS R-CNN",
    "PointRend",
    "QueryInst",
    "RTMDet",
    "SOLO",
    "SOLOv2",
    "YOLACT",
]


def parse_run_log(log_path):
    """Extract train loss, val bbox_mAP, val segm_mAP per epoch."""
    train_loss = {}  # epoch -> loss
    val_bbox = {}    # epoch -> bbox_mAP
    val_segm = {}    # epoch -> segm_mAP

    with open(log_path, 'r') as f:
        for line in f:
            # Epoch(train) [N][...] ... loss: X.XXXX ...
            m = re.search(r'Epoch\(train\)\s+\[(\d+)\]\[[\d/]+\].*?\bloss:\s*([\d.]+)', line)
            if m:
                epoch = int(m.group(1))
                loss = float(m.group(2))
                train_loss[epoch] = loss

            # Epoch(val) [N][...] ... coco/bbox_mAP: X ... coco/segm_mAP: Y (most models)
            m_val = re.search(r'Epoch\(val\)\s+\[(\d+)\]\[[\d/]+\].*?coco/bbox_mAP:\s*([\d.]+).*?coco/segm_mAP:\s*([\d.]+)', line)
            if m_val:
                epoch = int(m_val.group(1))
                val_bbox[epoch] = float(m_val.group(2))
                val_segm[epoch] = float(m_val.group(3))
                continue

            # Epoch(val) [N][...] ... coco/segm_mAP: X (segment-only models like SOLO)
            m_segm = re.search(r'Epoch\(val\)\s+\[(\d+)\]\[[\d/]+\].*?coco/segm_mAP:\s*([\d.]+)', line)
            if m_segm:
                epoch = int(m_segm.group(1))
                val_segm[epoch] = float(m_segm.group(2))

    # Convert to sorted lists — use union of epochs from both sources
    all_val_epochs = sorted(set(val_bbox.keys()) | set(val_segm.keys()))
    val_bboxes = [val_bbox.get(e) for e in all_val_epochs]  # None if missing
    val_segms = [val_segm.get(e) for e in all_val_epochs]

    train_epochs = sorted(train_loss.keys())
    train_losses = [train_loss[e] for e in train_epochs]

    return train_epochs, train_losses, all_val_epochs, val_bboxes, val_segms


n_rows, n_cols = 3, 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(140, 70))

for idx, (model, name) in enumerate(zip(models, display_names)):
    log_path = os.path.join(base_dir, model, "run.log")
    row, col = divmod(idx, n_cols)
    ax = axes[row, col]

    train_epochs, train_losses, val_epochs, val_bboxes, val_segms = parse_run_log(log_path)

    # Left y-axis: loss
    ax.plot(train_epochs, train_losses, color='#1f77b4', label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss', color='#1f77b4')
    ax.tick_params(axis='y', labelcolor='#1f77b4')

    # Right y-axis: mAP
    ax2 = ax.twinx()
    ax2.plot(val_epochs, val_segms, color='#d62728', label='Segm mAP', marker='.')
    # Some models (SOLO, SOLOv2) have no bbox_mAP
    if any(v is not None for v in val_bboxes):
        ax2.plot(val_epochs, val_bboxes, color='#2ca02c', label='Bbox mAP', marker='.')
    ax2.set_ylabel('mAP', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')

    ax.set_title(name, fontweight='bold')

    # Collect legend handles from first subplot only (same for all)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines = lines1 + lines2
    all_labels = labels1 + labels2

    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(7)
    for spine in ax2.spines.values():
        spine.set_linewidth(7)

# Single shared legend at the bottom
fig.legend(all_lines, all_labels, loc='upper center', ncol=3,
           fontsize=108, frameon=True, bbox_to_anchor=(0.5, 0.06))

plt.subplots_adjust(left=0.10, right=0.90, top=0.88, bottom=0.18, wspace=0.65, hspace=0.60)
output_path = "metric_plots/merged_loss_and_mAP_3x5.png"
os.makedirs("metric_plots", exist_ok=True)
fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved to {output_path}")
