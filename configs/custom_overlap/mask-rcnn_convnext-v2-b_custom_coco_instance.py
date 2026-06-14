# Mask R-CNN + ConvNeXt-V2-B for dense overlapping instance segmentation.
#
# ConvNeXt-V2 provides stronger backbone features (GRN, larger capacity)
# which help disambiguate overlapping instances through better representations.
#
# Overlap changes vs configs/custom/:
#   - RPN: nms_pre=4000, max_per_img=3000
#   - RCNN/test: SoftNMS (already had it in original, kept), score_thr=0.001,
#     max_per_img=300
_base_ = [
    '../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/custom_coco_instance.py',
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmpretrain.models'], allow_failed_imports=False)

checkpoint_file = ('https://download.openmmlab.com/mmclassification/v0/'
                   'convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_'
                   '20230104-8a798eaf.pth')

num_classes = 1
image_size = (1024, 1024)

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt', arch='base',
        out_indices=[0, 1, 2, 3], drop_path_rate=0.4,
        layer_scale_init_value=0., gap_before_final_norm=False,
        use_grn=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file,
                      prefix='backbone.')),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)),
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
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

train_dataloader = dict(
    batch_size=1, num_workers=8, persistent_workers=True, pin_memory=True,
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
    metric=['bbox', 'segm'], format_only=False, backend_args=backend_args)
test_evaluator = dict(type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['bbox', 'segm'], format_only=False, backend_args=backend_args)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
