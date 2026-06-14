# Cascade Mask R-CNN for dense overlapping instance segmentation.
#
# The 3-stage cascade naturally helps with overlapping instances by
# progressively refining bboxes with stricter IoU thresholds.
#
# Overlap changes vs configs/custom/:
#   - RPN: nms_pre=4000, max_per_img=3000
#   - RCNN: SoftNMS, score_thr=0.001, max_per_img=300
#   - Stage assigners: lowered pos_iou_thr (0.4/0.5/0.6 vs 0.5/0.6/0.7)
#   - RPN train: pos_iou_thr=0.6
_base_ = [
    '../../configs/_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/custom_coco_instance.py',
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py',
]

load_from = ('https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/'
             'cascade_mask_rcnn_r50_fpn_1x_coco/'
             'cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth')

num_classes = 1

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                    target_means=[0.,0.,0.,0.], target_stds=[0.1,0.1,0.2,0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                    target_means=[0.,0.,0.,0.], target_stds=[0.05,0.05,0.1,0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                    target_means=[0.,0.,0.,0.], target_stds=[0.033,0.033,0.067,0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ],
        mask_head=dict(num_classes=num_classes)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(pos_iou_thr=0.6),
        ),
        # NOTE: rpn_proposal is at train_cfg level, NOT inside rpn
        rpn_proposal=dict(nms_pre=4000, max_per_img=3000),
        # mmengine replaces list items by index (does NOT deep-merge),
        # so we must include the full dict for each stage.
        rcnn=[
            # Stage 1: pos_iou_thr 0.5→0.4
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4,
                    match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler', num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28, pos_weight=-1, debug=False),
            # Stage 2: pos_iou_thr 0.6→0.5
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5,
                    match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler', num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28, pos_weight=-1, debug=False),
            # Stage 3: pos_iou_thr 0.7→0.6
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6,
                    match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler', num=512, pos_fraction=0.25,
                    neg_pos_ub=-1, add_gt_as_proposals=True),
                mask_size=28, pos_weight=-1, debug=False),
        ],
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=4000, max_per_img=3000),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.001),
            max_per_img=300,
        ),
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
    batch_size=8, num_workers=8, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_train.json',
                 data_prefix=dict(img='images/train/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=1),
                 pipeline=train_pipeline, backend_args=backend_args))

val_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_val.json',
                 data_prefix=dict(img='images/val/'), test_mode=True,
                 pipeline=test_pipeline, backend_args=backend_args))

test_dataloader = dict(
    batch_size=8, num_workers=4, persistent_workers=True, drop_last=False,
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

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
