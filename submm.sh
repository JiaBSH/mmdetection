#!/bin/bash
#SBATCH --job-name=isat_a_overlap
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# =============================================================================
# mmdetection batch training — thin SLURM wrapper
# =============================================================================
# Usage:
#   sbatch submm.sh
#   CONFIG_DIR=configs/ablation MAX_EPOCHS=20 sbatch submm.sh
#   USE_RAMDISK=0 DATA_ROOT=/path/to/data sbatch submm.sh
#
# All parameters below can be overridden via environment or by editing the
# defaults in this file.  submm.py reads these same env vars.
# =============================================================================

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/data/apps/miniforge/25.3.0-3}"
CONDA_ENV="${CONDA_ENV_NAME:-mmdetection_para}"
PROJECT_ROOT="${PROJECT_ROOT:-/data/run01/scvi576/JiaBSH/mmdetection_para}"
TORCH_HOME="${TORCH_HOME:-/data/run01/scvi576/JiaBSH/.torch_cache}"
CUDA_HOME="${CUDA_HOME:-/data/apps/cuda/12.8}"

module purge 2>/dev/null || true
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TORCH_HOME

# PyTorch 2.6+ defaults to weights_only=True in torch.load, which rejects
# checkpoints containing objects like mmengine HistoryBuffer.  Force the old
# behaviour so mmengine's load_checkpoint works during test/val.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

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

cd "$PROJECT_ROOT"
mkdir -p "$TORCH_HOME" logs

# ── Training parameters ──────────────────────────────────────────────────────
# Exposed so you can tweak them at the SBATCH/env level without editing submm.py.
export CONFIG_DIR="${CONFIG_DIR:-configs/custom_overlap}"
#export DATA_ROOT="${DATA_ROOT:-data/syn_multimag/adaptive_patches_jitt/}"
#export WORK_DIR_ROOT="${WORK_DIR_ROOT:-work_dirs/run_$(date +%Y%m%d_%H%M%S)}"
export WORK_DIR_ROOT="${WORK_DIR_ROOT:-work_dirs/run_isat_aug_overlap}"
export NUM_GPUS="${NUM_GPUS:-1}"
export TEST_MAX_EPOCHS="${TEST_MAX_EPOCHS:-100}"
export TEST_TRAIN_BATCH_SIZE="${TEST_TRAIN_BATCH_SIZE:-2}"
export TEST_VAL_BATCH_SIZE="${TEST_VAL_BATCH_SIZE:-2}"
export TEST_TEST_BATCH_SIZE="${TEST_TEST_BATCH_SIZE:-2}"

# ── Early stopping ───────────────────────────────────────────────────────────
export ENABLE_EARLY_STOPPING="${ENABLE_EARLY_STOPPING:-1}"
export EARLY_STOP_MONITOR="${EARLY_STOP_MONITOR:-coco/segm_mAP}"
export EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
export EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
export EARLY_STOP_RULE="${EARLY_STOP_RULE:-greater}"
export EARLY_STOP_THRESHOLD="${EARLY_STOP_THRESHOLD:-}"

# ── Ramdisk — extract dataset to /dev/shm for low-latency I/O ────────────────
USE_RAMDISK="${USE_RAMDISK:-1}"
RAMDISK_TAR="${RAMDISK_TAR:-dataset_root/mmdata_isat_1024_aug.tar}"
RAMDISK_CLEAN="${RAMDISK_CLEAN:-1}"

RAMDISK_DIR=""  # set after extraction

if [[ "$USE_RAMDISK" == "1" ]]; then
    echo "[ramdisk] Extracting $RAMDISK_TAR to /dev/shm ..."
    date
    tar -xf "$RAMDISK_TAR" -C /dev/shm

    # Auto-derive extracted directory name from tar filename (strip .tar suffix)
    RAMDISK_DIR="/dev/shm/$(basename "$RAMDISK_TAR" .tar)"
    export DATA_ROOT="${RAMDISK_DIR}/"
    echo "[ramdisk] DATA_ROOT = $DATA_ROOT"
    date
fi

# ── Cleanup trap — always remove /dev/shm data after job ─────────────────────
cleanup_ramdisk() {
    if [[ "${RAMDISK_CLEAN:-1}" == "1" && -n "${RAMDISK_DIR:-}" && -d "${RAMDISK_DIR:-}" ]]; then
        echo "[ramdisk] Cleaning up $RAMDISK_DIR ..."
        rm -rf "$RAMDISK_DIR"
        echo "[ramdisk] Done."
    fi
}
trap cleanup_ramdisk EXIT

# ── Run ──────────────────────────────────────────────────────────────────────
exec python submm.py
