_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/custom_coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Load full COCO-pretrained Mask R-CNN (backbone + FPN + RPN + heads).
# Layers whose shapes differ (bbox_head/mask_head num_classes=80→1) are
# automatically skipped and randomly re-initialised.
load_from = ('https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/'
             'mask_rcnn_r50_fpn_1x_coco/'
             'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth')

# Set this to your dataset class count.
num_classes = 1

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)))

# Use all 8 CPUs allocated by SLURM; persistent workers and pin_memory reduce
# the per-iteration data-loading overhead and improve GPU utilisation.
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    )
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    )
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    )
# Reduce end-of-epoch memory spikes and skip in-train validation.
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_optimizer=False))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
