# SOLO for dense overlapping instance segmentation.
#
# SOLO segments objects by location (grid cells) without bbox detection
# or NMS. Its Matrix NMS is already overlap-aware, but we lower the
# score threshold and increase max_per_img to retain more instances.
#
# Overlap changes vs configs/custom/:
#   - test_cfg: score_thr=0.01 (from 0.1), mask_thr=0.3 (from 0.5),
#     max_per_img=500 (from 100)
#   - nms kernel sigma=2.0 for softer suppression
_base_ = '../../configs/solo/solo_r50_fpn_1x_coco.py'

load_from = ('https://download.openmmlab.com/mmdetection/v2.0/solo/'
             'solo_r50_fpn_1x_coco/'
             'solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth')

num_classes = 1

model = dict(
    mask_head=dict(num_classes=num_classes),
    test_cfg=dict(
        nms=dict(type='matrix_nms', kernel='gaussian', sigma=2.0),
        score_thr=0.01,            # 0.1 → 0.01: keep low-conf overlaps
        mask_thr=0.3,              # 0.5 → 0.3: softer mask binarization
        max_per_img=500,           # 100 → 500
        update_thr=0.05,           # default
    ),
)

dataset_type = 'CocoDataset'
data_root = 'dataset_root/dataset_mini/'
metainfo = dict(classes=('畴区',))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

train_dataloader = dict(
    batch_size=2, num_workers=8, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_train.json',
                 data_prefix=dict(img='images/train/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=1),
                 pipeline=train_pipeline, backend_args=backend_args))

val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_val.json',
                 data_prefix=dict(img='images/val/'), test_mode=True,
                 pipeline=test_pipeline, backend_args=backend_args))

test_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_test.json',
                 data_prefix=dict(img='images/test/'), test_mode=True,
                 pipeline=test_pipeline, backend_args=backend_args))

val_evaluator = dict(type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric='segm', format_only=False, backend_args=backend_args)
test_evaluator = dict(type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric='segm', format_only=False, backend_args=backend_args)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
