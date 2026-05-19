#!/bin/bash
#SBATCH --job-name=postproc_t1024
#SBATCH -p qgpu_4090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

module purge
source /hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh
conda activate openmmlab2

cd /hpcfs/fhome/sunxc/JiaBSH/mmdetection

echo "===== test_set_1024 多模型测评 ====="

MODEL_ROOT="${MODEL_ROOT:-work_dirs/custom_all_main}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
PATCH_SIZE="${PATCH_SIZE:-512}"
PATCH_OVERLAP_RATIO="${PATCH_OVERLAP_RATIO:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
OUT_ROOT="${OUT_ROOT:-outputs/test_overlap_set_1024}"
SCORE_THRESH="${SCORE_THRESH:-0.5}"

run_compare() {
    local split_name="$1"
    local ann_file="$2"
    local img_dir="$3"
    local mode="$4"

    local out_dir="${OUT_ROOT}/${split_name}/${mode}"

    echo "===== ${split_name} | ${mode} ====="

    local compare_args=(
        --ann-file "${ann_file}"
        --img-dir "${img_dir}"
        --out-dir "${out_dir}"
        --model-cfg postprocess/test_list.yaml
        --model-root "${MODEL_ROOT}"
        --score-thresh "${SCORE_THRESH}"
        --enable-poly-metrics
        --enable-gt
        --device cuda:0
    )

    if [[ -n "${CHECKPOINT_EPOCH}" ]]; then
        compare_args+=(--checkpoint-epoch "${CHECKPOINT_EPOCH}")
    fi

    if [[ "${mode}" == "sliding" ]]; then
        compare_args+=(
            --sliding-window
            --patch-size "${PATCH_SIZE}"
            --patch-overlap-ratio "${PATCH_OVERLAP_RATIO}"
            --batch-size "${BATCH_SIZE}"
        )
    fi

    python postprocess/compare_models.py "${compare_args[@]}"
}

# 20x / 50x / 100x: 非滑窗预测

run_compare \
    "sr2_5x_unsup" \
    "dataset_root/test_set_1024/annotations/instances_sr2_5x_unsup.json" \
    "dataset_root/test_set_1024/images/sr2_5x_unsup" \
    "sliding"
    
    
'''
run_compare \
    "20x" \
    "dataset_root/test_set_1024/annotations/instances_20x.json" \
    "dataset_root/test_set_1024/images/20x" \
    "plain"

run_compare \
    "50x" \
    "dataset_root/test_set_1024/annotations/instances_50x.json" \
    "dataset_root/test_set_1024/images/50x" \
    "plain"

run_compare \
    "100x" \
    "dataset_root/test_set_1024/annotations/instances_100x.json" \
    "dataset_root/test_set_1024/images/100x" \
    "plain"
'''
# 2_5x / 5x: 同时跑非滑窗和滑窗预测
run_compare \
    "2_5x_unsup" \
    "dataset_root/test_set_1024/annotations/instances_2_5x_unsup.json" \
    "dataset_root/test_set_1024/images/2_5x_unsup" \
    "plain"



run_compare \
    "5x_unsup" \
    "dataset_root/test_set_1024/annotations/instances_5x_unsup.json" \
    "dataset_root/test_set_1024/images/5x_unsup" \
    "plain"

run_compare \
    "5x_unsup" \
    "dataset_root/test_set_1024/annotations/instances_5x_unsup.json" \
    "dataset_root/test_set_1024/images/5x_unsup" \
    "sliding"

echo "===== 完成 ====="
