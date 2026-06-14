#!/bin/bash
# Launch M2 adaptive inference on real dataset
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate mmdetection_para
cd /data/run01/scvi576/JiaBSH/mmdetection_para
python tools/test_real_dataset.py \
    --config work_dirs/ablation_m2_multimag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m2_multimag/epoch_5.pth \
    --dataset-root dataset_root/mmdata_test \
    --scale-model data/syn_multimag/scale_pipeline_dinov2.joblib \
    --out-dir work_dirs/real_test_results/M2_adaptive \
    --overlap-ratio 0.2 \
    2>&1
echo "EXIT_CODE=$?"
