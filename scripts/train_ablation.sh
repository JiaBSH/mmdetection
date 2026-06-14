#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate mmdetection_para

BASE=$(realpath .)
M1=$BASE/data/syn_multimag/m1_50xonly
M2=$BASE/data/syn_multimag/m2_allmag

CFG_OPTS="train_cfg.max_epochs=5 default_hooks.checkpoint.max_keep_ckpts=1"
M1_DO="train_dataloader.dataset.data_root=$M1 val_dataloader.dataset.data_root=$M1 test_dataloader.dataset.data_root=$M1 val_evaluator.ann_file=$M1/annotations/instances_val.json test_evaluator.ann_file=$M1/annotations/instances_test.json"
M2_DO="train_dataloader.dataset.data_root=$M2 val_dataloader.dataset.data_root=$M2 test_dataloader.dataset.data_root=$M2 val_evaluator.ann_file=$M2/annotations/instances_val.json test_evaluator.ann_file=$M2/annotations/instances_test.json"

echo "===== M1: Single-mag (50x only, 5 epochs) $(date) ====="
python tools/train.py configs/custom/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --cfg-options $CFG_OPTS $M1_DO \
    --work-dir work_dirs/ablation_m1_single_mag 2>&1 | tail -15

echo "===== M2: Multi-mag (all, 5 epochs) $(date) ====="
python tools/train.py configs/custom/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --cfg-options $CFG_OPTS $M2_DO \
    --work-dir work_dirs/ablation_m2_multimag 2>&1 | tail -15

echo "===== M3: Multi-mag + jitter (5 epochs) $(date) ====="
python tools/train.py configs/ablation/m3_multimag_jitter.py \
    --cfg-options $CFG_OPTS $M2_DO \
    --work-dir work_dirs/ablation_m3_multimag_jitter 2>&1 | tail -15

echo "=== Done $(date) ==="
for d in work_dirs/ablation_m*; do
    echo -n "$d: "; ls $d/epoch_5.pth 2>/dev/null || echo "missing"
done
