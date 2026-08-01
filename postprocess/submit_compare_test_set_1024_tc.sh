#!/bin/bash
#SBATCH --job-name=post
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
module load cuda/13.0
set -euo pipefail

module purge 2>/dev/null || true

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/data/apps/miniforge/25.3.0-3}"
CONDA_ENV="${CONDA_ENV_NAME:-mmdetection_para}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/run01/scvi576/JiaBSH/mmdetection_para}"
TORCH_HOME="${TORCH_HOME:-/data/run01/scvi576/JiaBSH/.torch_cache}"
CUDA_HOME="${CUDA_HOME:-/data/apps/cuda/13.0}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_HOME

# ── libstdc++ workaround ────────────────────────────────────────────────────
# mmcv was compiled with GCC 15.x (requires CXXABI_1.3.15), but the system
# /lib/x86_64-linux-gnu/libstdc++.so.6 only provides up to CXXABI_1.3.13.
# Preload the newer libstdc++ from the conda environment (installed via
# conda install -n mmdet_cu128 libstdcxx-ng).
CONDA_ENV_LIB="${CONDA_PREFIX}/lib/libstdc++.so.6"
if [[ -f "$CONDA_ENV_LIB" ]]; then
    export LD_PRELOAD="$CONDA_ENV_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[setup] LD_PRELOAD = $LD_PRELOAD"
else
    echo "[setup] WARNING: newer libstdc++ not found at $CONDA_ENV_LIB"
fi
cd /data/home/scvi576/run/JiaBSH/mmdetection_para

echo "===== test_set_1024 多模型测评 ====="

# ── 基本路径 ──────────────────────────────────────────────────────────────────
MODEL_ROOT="${MODEL_ROOT:-work_dirs/run_isat_aug}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
OUT_ROOT="${OUT_ROOT:-outputs/run_isat_allll_SCORE_THRESH04}"

# ── 滑窗/推理参数 ────────────────────────────────────────────────────────────
PATCH_SIZE="${PATCH_SIZE:-650}"
PATCH_OVERLAP_RATIO="${PATCH_OVERLAP_RATIO:-0.15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SCORE_THRESH="${SCORE_THRESH:-0.4}"
MERGE_OVERLAP_RATIO="${MERGE_OVERLAP_RATIO:-0.25}"

# ── 功能开关（0=关闭, 1=开启）────────────────────────────────────────────────
ENABLE_GT="${ENABLE_GT:-1}"                  # GT 几何分析（取向/尺寸）
ENABLE_GT_MATCHING="${ENABLE_GT_MATCHING:-1}" # GT↔Pred 实例匹配
ENABLE_POLY_METRICS="${ENABLE_POLY_METRICS:-1}" # 多边形 IoU/F1 评估
ENABLE_PLOTS="${ENABLE_PLOTS:-1}"            # 直方图/散点图
ENABLE_SAVE_IMAGES="${ENABLE_SAVE_IMAGES:-1}" # 中间过程图（hull/hex 等）

# ── 几何分析调参 ─────────────────────────────────────────────────────────────
GEOM_WORKERS="${GEOM_WORKERS:-8}"            # 并行线程数
SCATTER_METRIC="${SCATTER_METRIC:-both}"      # 散点图标题指标: mae / r2 / both
export BL_MASK_ALPHA=70
export BL_SLIDING_MERGE_OVERLAP_RATIO="${MERGE_OVERLAP_RATIO}"
export BL_SLIDING_CONTEXT_MARGIN_RATIO="${BL_SLIDING_CONTEXT_MARGIN_RATIO:-0.20}"
export BL_ONLY_PRED_VS_GT_IOU="${BL_ONLY_PRED_VS_GT_IOU:-0}"
export BL_METRICS_SAVE_VISUALIZATION="${BL_METRICS_SAVE_VISUALIZATION:-1}"
export BL_METRICS_SAVE_BAR="${BL_METRICS_SAVE_BAR:-1}"
export BL_COMPARE_SAVE_PLOTS="${BL_COMPARE_SAVE_PLOTS:-1}"
export BL_PRED_GEOM_HISTS="${BL_PRED_GEOM_HISTS:-1}"
export BL_PRED_DOA_HISTS="${BL_PRED_DOA_HISTS:-1}"
# ── 物理尺度换算（可选）───────────────────────────────────────────────────────
SCALE_RATIO="${SCALE_RATIO:-}"               # μm/px 等比例
SCALE_UNIT="${SCALE_UNIT:-}"                 # 单位名称

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
        --model-cfg postprocess/model_list_sr25.yaml
        --model-root "${MODEL_ROOT}"
        --score-thresh "${SCORE_THRESH}"
        --device cuda:0
    )

    # ── 功能开关 ──
    [[ "${ENABLE_POLY_METRICS}" == "1" ]] && compare_args+=(--enable-poly-metrics)
    [[ "${ENABLE_GT}" == "1" ]]          && compare_args+=(--enable-gt)
    [[ "${ENABLE_GT_MATCHING}" == "1" ]] && compare_args+=(--enable-gt-matching)
    [[ "${ENABLE_PLOTS}" == "1" ]]       && compare_args+=(--enable-plots)
    [[ "${ENABLE_SAVE_IMAGES}" == "1" ]] && compare_args+=(--enable-save-images)

    # ── 几何分析参数 ──
    [[ -n "${GEOM_WORKERS}" ]]   && compare_args+=(--geom-workers "${GEOM_WORKERS}")
    [[ -n "${SCATTER_METRIC}" ]] && compare_args+=(--scatter-metric "${SCATTER_METRIC}")

    # ── 物理尺度 ──
    [[ -n "${SCALE_RATIO}" ]] && compare_args+=(--scale-ratio "${SCALE_RATIO}")
    [[ -n "${SCALE_UNIT}" ]]  && compare_args+=(--scale-unit "${SCALE_UNIT}")

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
run_compare \
    "sr2_5x_plain_isat025_boundary_all" \
    "dataset_root/mmdata_test/annotations/instances_sr2_5x_unsup.json" \
    "dataset_root/mmdata_test/sr2_5x_unsup/image" \
    "plain"

# ---- old tests (heredoc skip) ----
: <<'SKIP'
run_compare \
    "20x" \
    "dataset_root/mmdata_test_1024/annotations/instances_20x.json" \
    "dataset_root/mmdata_test_1024/images/20xtest" \
    "plain"

run_compare \
    "50x_test" \
    "data/syn_multimag/coco_rotation/test5_t1/instances_test.json" \
    "data/syn_multimag/coco_rotation/test50/images" \
    "plain"
run_compare \
    "5x_unsup" \
    "data/syn_multimag/coco_rotation/test5_t1/instances_test.json" \
    "data/syn_multimag/coco_rotation/test5_t1/images" \
    "sliding"





    # 20x / 50x / 100x: 非滑窗预测

    



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
SKIP
echo "===== 完成 ====="
