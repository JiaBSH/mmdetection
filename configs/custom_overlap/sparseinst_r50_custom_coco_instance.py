# SparseInst for dense overlapping instance segmentation.
#
# SparseInst is a one-stage instance segmentation model using instance
# activation maps (IAM). Its SparseInstMatcher does bipartite matching,
# naturally avoiding the NMS problem for overlapping instances.
#
# Overlap changes vs configs/custom/:
#   - test_cfg: score_thr=0.001 (from 0.005), mask_thr_binary=0.3 (from 0.45)
#   - num_masks=300 (from 100) for more instance predictions
#   - SparseInstMatcher alpha=0.5 (from 0.8) for softer matching
_base_ = [
    '../../configs/_base_/schedules/schedule_1x.py',
    '../../configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.SparseInst.sparseinst'], allow_failed_imports=False)

num_classes = 1

model = dict(
    type='SparseInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True, pad_mask=True, pad_size_divisor=32),
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(1, 2, 3),
        frozen_stages=0, norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    encoder=dict(
        type='InstanceContextEncoder',
        in_channels=[512, 1024, 2048], out_channels=256),
    decoder=dict(
        type='BaseIAMDecoder',
        in_channels=256 + 2, num_classes=num_classes,
        ins_dim=256, ins_conv=4, mask_dim=256, mask_conv=4,
        kernel_dim=128, scale_factor=2.0, output_iam=False,
        num_masks=300),               # 100 → 300: more instance slots
    criterion=dict(
        type='SparseInstCriterion',
        num_classes=num_classes,
        assigner=dict(
            type='SparseInstMatcher',
            alpha=0.5,                # 0.8 → 0.5: softer matching for overlap
            beta=0.5),                # 0.2 → 0.5
        loss_cls=dict(type='FocalLoss', use_sigmoid=True,
            alpha=0.25, gamma=2.0, reduction='sum', loss_weight=2.0),
        loss_obj=dict(type='CrossEntropyLoss', use_sigmoid=True,
            reduction='mean', loss_weight=1.0),
        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True,
            reduction='mean', loss_weight=5.0),
        loss_dice=dict(type='DiceLoss', use_sigmoid=True,
            reduction='sum', eps=5e-5, loss_weight=2.0)),
    test_cfg=dict(
        score_thr=0.001,             # 0.005 → 0.001
        mask_thr_binary=0.3,         # 0.45 → 0.3: softer mask binarization
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
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))
]

train_dataloader = dict(
    batch_size=4, num_workers=8, persistent_workers=True, pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_train.json',
                 data_prefix=dict(img='images/train/'),
                 filter_cfg=dict(filter_empty_gt=True, min_size=1),
                 pipeline=train_pipeline, backend_args=backend_args))

val_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root, metainfo=metainfo,
                 ann_file='annotations/instances_val.json',
                 data_prefix=dict(img='images/val/'), test_mode=True,
                 pipeline=test_pipeline, backend_args=backend_args))

test_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True, drop_last=False,
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

optim_wrapper = dict(
    _delete_=True, type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.05))

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
