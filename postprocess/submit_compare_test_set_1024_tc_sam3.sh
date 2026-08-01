#!/bin/bash
#SBATCH --job-name=sam3post
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

set -euo pipefail

module purge 2>/dev/null || true
module load cuda/12.8 2>/dev/null || module load cuda/13.0 2>/dev/null || true

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/data/apps/miniforge/25.3.0-3}"
CONDA_ENV="${CONDA_ENV_NAME:-sam3}"
REPO_ROOT="${REPO_ROOT:-/data/home/scvi576/run/JiaBSH/mmdetection_para}"
SAM3_ROOT="${SAM3_ROOT:-/data/home/scvi576/run/JiaBSH/nano_sam3}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${SAM3_ROOT}/ms_cache/facebook/sam3/sam3.pt}"
SAM3_PROMPT="${SAM3_PROMPT:-Hexagon}"
SAM3_RESOLUTION="${SAM3_RESOLUTION:-1008}"
TORCH_HOME="${TORCH_HOME:-/data/run01/scvi576/JiaBSH/.torch_cache}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="${REPO_ROOT}:${SAM3_ROOT}:${PYTHONPATH:-}"
export TORCH_HOME
export BL_SAM3_USE_BF16="${BL_SAM3_USE_BF16:-1}"
export BL_SAM3_MODEL_BF16="${BL_SAM3_MODEL_BF16:-0}"

CONDA_ENV_LIB="${CONDA_PREFIX}/lib/libstdc++.so.6"
if [[ -f "$CONDA_ENV_LIB" ]]; then
    export LD_PRELOAD="$CONDA_ENV_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[setup] LD_PRELOAD = $LD_PRELOAD"
fi

cd "${REPO_ROOT}"

echo "===== test_set_1024 SAM3 多模型测评兼容输出 ====="

# ── 基本路径 ──────────────────────────────────────────────────────────────────
OUT_ROOT="${OUT_ROOT:-outputs/run_syn_sam3}"
SAM3_MODEL_NAME="${SAM3_MODEL_NAME:-SAM3}"

# ── 滑窗/推理参数：与 submit_compare_test_set_1024_tc.sh 保持一致 ─────────────
PATCH_SIZE="${PATCH_SIZE:-400}"
PATCH_OVERLAP_RATIO="${PATCH_OVERLAP_RATIO:-0.15}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SCORE_THRESH="${SCORE_THRESH:-0.5}"
MERGE_OVERLAP_RATIO="${MERGE_OVERLAP_RATIO:-0.3}"

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

    echo "===== ${split_name} | ${mode} | SAM3 ====="

    local compare_args=(
        --ann-file "${ann_file}"
        --img-dir "${img_dir}"
        --out-dir "${out_dir}"
        --model-name "${SAM3_MODEL_NAME}"
        --sam3-root "${SAM3_ROOT}"
        --checkpoint "${SAM3_CHECKPOINT}"
        --prompt "${SAM3_PROMPT}"
        --sam3-resolution "${SAM3_RESOLUTION}"
        --score-thresh "${SCORE_THRESH}"
        --device cuda:0
    )

    [[ "${ENABLE_POLY_METRICS}" == "1" ]] && compare_args+=(--enable-poly-metrics)
    [[ "${ENABLE_GT}" == "1" ]]          && compare_args+=(--enable-gt)
    [[ "${ENABLE_GT_MATCHING}" == "1" ]] && compare_args+=(--enable-gt-matching)
    [[ "${ENABLE_PLOTS}" == "1" ]]       && compare_args+=(--enable-plots)
    [[ "${ENABLE_SAVE_IMAGES}" == "1" ]] && compare_args+=(--enable-save-images)

    [[ -n "${GEOM_WORKERS}" ]]   && compare_args+=(--geom-workers "${GEOM_WORKERS}")
    [[ -n "${SCATTER_METRIC}" ]] && compare_args+=(--scatter-metric "${SCATTER_METRIC}")

    [[ -n "${SCALE_RATIO}" ]] && compare_args+=(--scale-ratio "${SCALE_RATIO}")
    [[ -n "${SCALE_UNIT}" ]]  && compare_args+=(--scale-unit "${SCALE_UNIT}")

    if [[ "${mode}" == "sliding" ]]; then
        compare_args+=(
            --sliding-window
            --patch-size "${PATCH_SIZE}"
            --patch-overlap-ratio "${PATCH_OVERLAP_RATIO}"
            --batch-size "${BATCH_SIZE}"
        )
    fi

    python postprocess/compare_sam3.py "${compare_args[@]}"
}

run_compare \
    "2_5x_old2sliding_syn_sboundary_all" \
    "data/syn_multimag/coco_rotation/test2_5_t1/instances_test.json" \
    "data/syn_multimag/coco_rotation/test2_5_t1/images" \
    "sliding"

echo "===== 完成 ====="
