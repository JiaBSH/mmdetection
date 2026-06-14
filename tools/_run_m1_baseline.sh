#!/bin/bash
# Launch M1 baseline inference on real dataset (no sliding window)
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate mmdetection_para
cd /data/run01/scvi576/JiaBSH/mmdetection_para
python tools/test_real_dataset.py \
    --config work_dirs/ablation_m1_single_mag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m1_single_mag/epoch_5.pth \
    --dataset-root dataset_root/mmdata_test \
    --no-adaptive \
    --out-dir work_dirs/real_test_results/M1_noSW \
    2>&1
echo "EXIT_CODE=$?"
