#!/bin/bash
#SBATCH --job-name=window-grid-2p5x
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-24%4
#SBATCH --output=logs/window_sensitivity_%A_%a.out
#SBATCH --error=logs/window_sensitivity_%A_%a.err

set -euo pipefail

WINDOW_SIZES=(192 256 320 400 512)
OVERLAP_RATIOS=(0.00 0.10 0.15 0.20 0.30)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= 25 )); then
    echo "SLURM_ARRAY_TASK_ID must be in [0, 24], got ${TASK_ID}" >&2
    exit 2
fi

SIZE_INDEX=$((TASK_ID / 5))
OVERLAP_INDEX=$((TASK_ID % 5))
PATCH_SIZE="${WINDOW_SIZES[$SIZE_INDEX]}"
OVERLAP_RATIO="${OVERLAP_RATIOS[$OVERLAP_INDEX]}"
OVERLAP_TAG="${OVERLAP_RATIO/./p}"

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/scvi576/run/JiaBSH/mmdetection_para}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/dino_window_supplement/03_window_sensitivity_2p5x_single}"
RAW_DIR="${OUTPUT_ROOT}/raw"
OUTPUT_JSON="${RAW_DIR}/$(printf 'window_%04d_overlap_%s.json' "${PATCH_SIZE}" "${OVERLAP_TAG}")"

echo "task_id=${TASK_ID} patch_size=${PATCH_SIZE} overlap_ratio=${OVERLAP_RATIO} output_json=${OUTPUT_JSON}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    module load cuda/13.0
fi

CONDA_BASE="${CONDA_BASE:-/data/apps/miniforge/25.3.0-3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-mmdetection_para}"
CUDA_HOME="${CUDA_HOME:-/data/apps/cuda/13.0}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export BL_SLIDING_MERGE_OVERLAP_RATIO="${BL_SLIDING_MERGE_OVERLAP_RATIO:-0.3}"

mkdir -p "${RAW_DIR}"
cd "${PROJECT_ROOT}"

python -u postprocess/window_sensitivity.py \
    --ann-file data/syn_multimag/coco_rotation/test2_5_t1/instances_test.json \
    --image data/syn_multimag/coco_rotation/test2_5_t1/images/2p5x_00016.png \
    --model-config work_dirs/run_syn_rotation/detectors_htc-r50_custom_coco_instance/detectors_htc-r50_custom_coco_instance.py \
    --checkpoint work_dirs/run_syn_rotation/detectors_htc-r50_custom_coco_instance/epoch_17.pth \
    --model-name detectors_htc-r50_custom_coco_instance \
    --patch-size "${PATCH_SIZE}" \
    --overlap-ratio "${OVERLAP_RATIO}" \
    --batch-size 4 \
    --score-threshold 0.5 \
    --coco-max-dets 10000 \
    --device cuda:0 \
    --output-json "${OUTPUT_JSON}"
