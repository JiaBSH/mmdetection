"""Mask R-CNN for multi-mag ablation M3: scale jitter training."""
_base_ = '../custom/mask-rcnn_r50_fpn_1x_custom_coco_instance.py'

# 尺度抖动: RandomResize with ±25% scale variation
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.75, 1.25),   # ±25% scale jitter
        keep_ratio=True,
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
