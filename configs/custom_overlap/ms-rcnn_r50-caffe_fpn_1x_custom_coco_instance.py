# Mask Scoring R-CNN for dense overlapping instance segmentation.
#
# MS R-CNN adds a mask IoU head that learns to predict mask quality.
# This is naturally beneficial for overlapping instances because it
# can down-weight masks that are confused by neighboring instances.
#
# Overlap changes vs configs/custom/:
#   - RPN: nms_pre=4000, max_per_img=3000
#   - RCNN: SoftNMS, score_thr=0.001, max_per_img=300
#   - Train assigner: lower pos_iou_thr for better recall
_base_ = '../../configs/ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py'

load_from = ('https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/'
             'ms_rcnn_r50_caffe_fpn_1x_coco/'
             'ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth')

num_classes = 1

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes),
        mask_iou_head=dict(num_classes=num_classes)),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(pos_iou_thr=0.6),
            rpn_proposal=dict(nms_pre=4000, max_per_img=3000),
        ),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4),
        ),
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
