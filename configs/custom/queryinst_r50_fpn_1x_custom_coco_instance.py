_base_ = '../queryinst/queryinst_r50_fpn_1x_coco.py'

# Load full COCO-pretrained QueryInst R50.
# DIIHead/DynamicMaskHead num_classes layers (80→1) are skipped automatically.
load_from = ('https://download.openmmlab.com/mmdetection/v2.0/queryinst/'
             'queryinst_r50_fpn_1x_coco/'
             'queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth')

num_classes = 1
num_stages = 6
num_proposals = 100

# QueryInst uses lists of 6 DIIHead bbox_heads and 6 DynamicMaskHead
# mask_heads; lists must be fully re-specified.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=num_classes,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ],
        mask_head=[
            dict(
                type='DynamicMaskHead',
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=14,
                    with_proj=False,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                num_convs=4,
                num_classes=num_classes,
                roi_feat_size=14,
                in_channels=256,
                conv_kernel_size=3,
                conv_out_channels=256,
                class_agnostic=False,
                norm_cfg=dict(type='BN'),
                upsample_cfg=dict(type='deconv', scale_factor=2),
                loss_mask=dict(
                    type='DiceLoss',
                    loss_weight=8.0,
                    use_sigmoid=True,
                    activate=False,
                    eps=1e-5)) for _ in range(num_stages)
        ]))

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset_type = 'CocoDataset'
data_root = 'dataset_root/dataset_mini/'
metainfo = dict(classes=('畴区', ))
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
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
