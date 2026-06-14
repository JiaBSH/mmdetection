# RTMDet-Ins for dense overlapping instance segmentation.
#
# RTMDet is a single-stage detector with instance mask head. For overlapping
# instances, we lower score thresholds and increase max detections.
#
# Overlap changes vs configs/custom/:
#   - test_cfg: score_thr=0.001 (from 0.001), max_per_img=300 (from 100)
#   - nms with higher iou_threshold → keep more overlapping dets
_base_ = '../../configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py'

load_from = ('https://download.openmmlab.com/mmdetection/v3.0/rtmdet/'
             'rtmdet-ins_tiny_8xb32-300e_coco/'
             'rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth')

num_classes = 1

model = dict(
    bbox_head=dict(num_classes=num_classes),
    test_cfg=dict(
        nms=dict(type='nms', iou_threshold=0.7),  # 0.65 → 0.7: softer NMS
        score_thr=0.001,
        max_per_img=300,                           # 100 → 300
    ),
)

dataset_type = 'CocoDataset'
data_root = 'dataset_root/dataset_mini/'
metainfo = dict(classes=('畴区',))
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='Pad', size=(1024, 1024), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

train_dataloader = dict(
    batch_size=2, num_workers=8, persistent_workers=True, pin_memory=True,
    batch_sampler=None, sampler=dict(type='DefaultSampler', shuffle=True),
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
    metric=['bbox', 'segm'], format_only=False, backend_args=backend_args)
test_evaluator = dict(type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['bbox', 'segm'], format_only=False, backend_args=backend_args)

max_epochs = 50
base_lr = 0.004

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05,
         begin=max_epochs // 2, end=max_epochs, T_max=max_epochs // 2,
         by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

custom_hooks = [
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002,
         update_buffers=True, priority=49)
]

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
