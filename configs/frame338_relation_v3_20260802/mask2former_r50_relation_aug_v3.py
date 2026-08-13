_base_ = '../frame338_relation_v2_20260802/mask2former_r50_relation_aug_v2.py'

# Continue from the stronger v1 checkpoint; v3 is a low-dose relation fine-tune.
load_from = ('/data/home/scvi576/run/JiaBSH/mmdetection_para/frame338_g5090_20260801/'
             'work_dirs/mask2former_r50_custom_coco_instance/'
             'best_coco_segm_mAP_epoch_4.pth')

data_root = ('/data/home/scvi576/run/JiaBSH/mmdetection_para/'
             'frame338_relation_v3_20260802/data/frame338_relation_aug_v3/')

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))
val_evaluator = dict(ann_file=data_root + 'annotations/instances_val.json')
test_evaluator = dict(ann_file=data_root + 'annotations/instances_test.json')

optim_wrapper = dict(optimizer=dict(lr=0.00001))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.2, by_epoch=False, begin=0, end=80),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=20,
         milestones=[12, 16], gamma=0.1),
]
train_cfg = dict(max_epochs=20)
