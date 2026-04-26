_base_ = './mask-rcnn_r50_fpn_1x_custom_coco_instance.py'

# InstaBoost augmentation applied on top of the custom Mask R-CNN baseline.
# Requires: pip install instaboost
# The rest of the config (data, num_classes, load_from) is inherited from the
# custom Mask R-CNN config above.

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
