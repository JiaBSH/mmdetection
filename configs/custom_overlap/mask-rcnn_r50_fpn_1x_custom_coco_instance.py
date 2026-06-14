# Mask R-CNN R50-FPN for dense overlapping instance segmentation.
#
# Overlap-oriented changes vs configs/custom/:
#   - RPN: more proposals (nms_pre=4000, max_per_img=3000)
#   - RCNN: SoftNMS replaces hard NMS → keeps overlapping true positives
#   - RCNN: lower score_thr (0.001), higher max_per_img (300)
#   - RPN train: pos_iou_thr=0.6 (lower from 0.7) → more positive anchors
#     for overlapping instances
#   - RCNN train: pos_iou_thr=0.4 (lower from 0.5) → higher recall on
#     overlapping proposals
_base_ = [
    '../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/custom_coco_instance.py',
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]

# Load full COCO-pretrained Mask R-CNN (backbone + FPN + RPN + heads).
load_from = ('https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/'
             'mask_rcnn_r50_fpn_1x_coco/'
             'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth')

num_classes = 1

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)),
    # ── Training: lower IoU thresholds for overlapping instances ──
    train_cfg=dict(
        rpn=dict(
            assigner=dict(pos_iou_thr=0.6),  # 0.7 → 0.6: more pos anchors
            rpn_proposal=dict(
                nms_pre=4000,           # 2000 → 4000: more proposals before NMS
                max_per_img=3000,       # 1000 → 3000: more proposals after NMS
            ),
        ),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.4,        # 0.5 → 0.4: more pos samples for overlap
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
            ),
        ),
    ),
    # ── Inference: SoftNMS + more detections ──
    test_cfg=dict(
        rpn=dict(
            nms_pre=4000,               # 1000 → 4000
            max_per_img=3000,           # 1000 → 3000
        ),
        rcnn=dict(
            score_thr=0.001,            # 0.05 → 0.001: keep low-conf overlaps
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001),
            max_per_img=300,            # 100 → 300
        ),
    ),
)

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True, pin_memory=True)
val_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True)
test_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, pin_memory=True)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
