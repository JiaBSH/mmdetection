#!/bin/bash
#SBATCH --job-name=mm_batch8
#SBATCH -p qgpu_4090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------
CONDA_PROFILE_SCRIPT="/hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh"
CONDA_ENV_NAME="openmmlab2"
PROJECT_ROOT="/hpcfs/fhome/sunxc/JiaBSH/mmdetection"
CONFIG_DIR="configs/custom_pretrain"
WORK_DIR_ROOT="work_dirs/custom_all_main_es_max50"
TRAIN_TEST_SCRIPT="tools/train_then_test_instance_seg.sh"
DATA_ROOT="${DATA_ROOT:-dataset_root/dataset_1024_aug/}"
NUM_GPUS=1
TEST_MAX_EPOCHS="${TEST_MAX_EPOCHS:-50}"
TEST_TRAIN_BATCH_SIZE="${TEST_TRAIN_BATCH_SIZE:-2}"
TEST_VAL_BATCH_SIZE="${TEST_VAL_BATCH_SIZE:-2}"
TEST_TEST_BATCH_SIZE="${TEST_TEST_BATCH_SIZE:-2}"
ENABLE_EARLY_STOPPING="${ENABLE_EARLY_STOPPING:-1}"
EARLY_STOP_MONITOR="${EARLY_STOP_MONITOR:-coco/segm_mAP}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
EARLY_STOP_RULE="${EARLY_STOP_RULE:-greater}"
EARLY_STOP_THRESHOLD="${EARLY_STOP_THRESHOLD:-}"
REQUIRED_PYTHON_MODULES=(
    "skimage:scikit-image"
    "mmpretrain:mmpretrain"
    "instaboostfast:instaboostfast"
)

# ❗不要 load python module
module purge

# ✅ 用你自己的 conda
source "$CONDA_PROFILE_SCRIPT"

#conda init
conda activate "$CONDA_ENV_NAME"

set -euo pipefail

# ---------------------------------------------------------------------------
# Collect model statistics after each run:
#   params (M), train peak memory (MiB), inference time/image (ms), FPS.
# ---------------------------------------------------------------------------
collect_model_stats() {
    local config_path="$1"
    local work_dir="$2"

    python - "$config_path" "$work_dir" <<'PY'
import json
import re
import sys
import copy
import pathlib

config_path = pathlib.Path(sys.argv[1])
work_dir    = pathlib.Path(sys.argv[2])

# ── 1. Parameter count (CPU, no weights needed) ──────────────────────────────
try:
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    from mmengine.config import Config
    from mmdet.registry import MODELS
    cfg = Config.fromfile(str(config_path))
    # deepcopy keeps ConfigDict (attribute access); then strip weight keys
    model_cfg = copy.deepcopy(cfg.model)
    def _strip_init_cfg(node):
        if isinstance(node, dict):
            node.pop('init_cfg', None)
            node.pop('load_from', None)
            node.pop('pretrained', None)
            for v in list(node.values()):
                _strip_init_cfg(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _strip_init_cfg(v)
    _strip_init_cfg(model_cfg)
    model = MODELS.build(model_cfg)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    del model
except Exception as e:
    params_m = float('nan')
    print(f'[warn] param count failed: {e}', file=sys.stderr)

# ── 2. Train peak memory (MiB) from scalars.json ────────────────────────────
train_mem = float('nan')
try:
    scalars_files = sorted(work_dir.glob('*/vis_data/scalars.json'))
    if scalars_files:
        mem_vals = []
        with open(scalars_files[-1]) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if 'memory' in d and ('epoch' in d or 'iter' in d):
                        mem_vals.append(d['memory'])
                except Exception:
                    pass
        if mem_vals:
            train_mem = max(mem_vals)
except Exception as e:
    print(f'[warn] train memory read failed: {e}', file=sys.stderr)

# ── 3. Inference time + FPS from run.log "Epoch(test)" line ─────────────────
test_time_ms = float('nan')
fps          = float('nan')
try:
    log_file = work_dir / 'run.log'
    time_pat = re.compile(r'\btime:\s*([\d.]+)')
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                if 'Epoch(test)' not in line and 'Iter(test)' not in line:
                    continue
                tm = time_pat.search(line)
                if tm:
                    t = float(tm.group(1))
                    test_time_ms = t * 1000
                    fps = 1.0 / t if t > 0 else float('nan')
except Exception as e:
    print(f'[warn] run.log time parse failed: {e}', file=sys.stderr)

# ── 4. Output TSV row ────────────────────────────────────────────────────────
def fmt(v):
    if isinstance(v, float) and v != v:
        return 'N/A'
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)

parts = [
    config_path.stem,
    fmt(params_m) + ' M',
    fmt(train_mem) + ' MiB',
    fmt(test_time_ms) + ' ms',
    fmt(fps) + ' fps',
]
print('\t'.join(parts))
PY
}

print_stats_table() {
    local stats_file="$1"
    echo ""
    echo "===== MODEL STATS ====="
    printf '%-45s | %-10s | %-12s | %-12s | %-9s\n' \
        "MODEL" "PARAMS" "TRAIN MEM" "INF TIME" "FPS"
    printf '%.0s-' {1..100}; echo
    while IFS=$'\t' read -r model params tmem inft fps; do
        printf '%-45s | %-10s | %-12s | %-12s | %-9s\n' \
            "$model" "$params" "$tmem" "$inft" "$fps"
    done < "$stats_file"
}

print_summary_table() {
    local summary_file="$1"

    echo "===== RUN SUMMARY ====="
        printf '%-45s | %-10s | %-12s | %-10s | %s\n' \
            "MODEL" "RUN" "WEIGHTS" "LOAD_OK" "REASON"
        printf '%-45s-+-%-10s-+-%-12s-+-%-10s-+-%s\n' \
        "---------------------------------------------" \
            "----------" "------------" "----------" \
        "------------------------------"

        while IFS=$'\t' read -r model run_status weights_source load_ok reason; do
            printf '%-45s | %-10s | %-12s | %-10s | %s\n' \
                "$model" "$run_status" "$weights_source" "$load_ok" "$reason"
    done < "$summary_file"
}

is_model_completed() {
    local work_dir="$1"

    [[ $(find "$work_dir" -maxdepth 1 \( -name 'best_*.pth' -o -name 'latest.pth' -o -name 'epoch_*.pth' -o -name 'iter_*.pth' \) | head -n 1) \
        && -d "$work_dir/test" && -d "$work_dir/metric_plots" ]]
}

extract_failure_reason() {
    local log_file="$1"

    python - "$log_file" <<'PY'
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
if not log_path.exists():
    print('no log captured')
    raise SystemExit(0)

lines = [line.strip() for line in log_path.read_text(errors='ignore').splitlines() if line.strip()]

priority_prefixes = (
    'RuntimeError:',
    'ImportError:',
    'ModuleNotFoundError:',
    'FileNotFoundError:',
    'AssertionError:',
    'ValueError:',
    'KeyError:',
)

for prefix in priority_prefixes:
    for line in reversed(lines):
        if line.startswith(prefix):
            print(line)
            raise SystemExit(0)

for line in reversed(lines):
    if 'No module named' in line or 'not installed' in line or 'FAILED' in line:
        print(line)
        raise SystemExit(0)

print(lines[-1] if lines else 'unknown failure')
PY
}

detect_weight_load_status() {
    local log_file="$1"
    local weights_source="$2"

    python - "$log_file" "$weights_source" <<'PY'
import pathlib
import sys

log_path = pathlib.Path(sys.argv[1])
weights_source = sys.argv[2]

if weights_source == 'none':
    print('n/a')
    raise SystemExit(0)

if not log_path.exists():
    print('no')
    raise SystemExit(0)

lines = log_path.read_text(errors='ignore').splitlines()
load_markers = ('Loads checkpoint by', 'Load checkpoint from')
failure_markers = (
    'FileNotFoundError:',
    'RuntimeError:',
    'URLError',
    'HTTPError',
    'No such file or directory',
)

for line in lines:
    if any(marker in line for marker in load_markers):
        print('yes')
        raise SystemExit(0)

for line in reversed(lines):
    if any(marker in line for marker in failure_markers):
        print('no')
        raise SystemExit(0)

print('unknown')
PY
}

get_weight_info() {
    local config_path="$1"

    python - "$config_path" <<'PY'
import sys

from mmengine.config import Config


def find_checkpoint(node):
    if isinstance(node, dict):
        init_cfg = node.get('init_cfg')
        if isinstance(init_cfg, dict) and 'checkpoint' in init_cfg:
            return init_cfg['checkpoint']
        for value in node.values():
            found = find_checkpoint(value)
            if found:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = find_checkpoint(value)
            if found:
                return found
    return None


cfg = Config.fromfile(sys.argv[1])
load_from = cfg.get('load_from')
if load_from:
    print(f'load_from\t{load_from}')
else:
    checkpoint = find_checkpoint(cfg.get('model', {}))
    if checkpoint:
        print(f'init_cfg\t{checkpoint}')
    else:
        print('none\t-')
PY
}

ensure_python_module() {
    local module_name="$1"
    local package_name="$2"

    if ! python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('${module_name}') else 1)"; then
        echo "Installing missing package: $package_name"
        python -m pip install "$package_name"
    fi
}

echo "===== DEBUG ====="
which python
python -V
pip list | grep -E "mmcv|mmdet|mmengine"
echo "================="

for module_spec in "${REQUIRED_PYTHON_MODULES[@]}"; do
    IFS=':' read -r module_name package_name <<< "$module_spec"
    ensure_python_module "$module_name" "$package_name"
done

cd "$PROJECT_ROOT"

COMMON_CFG_OPTIONS=(
    --cfg-options
    "data_root=${DATA_ROOT}"
    "train_dataloader.dataset.data_root=${DATA_ROOT}"
    "val_dataloader.dataset.data_root=${DATA_ROOT}"
    "test_dataloader.dataset.data_root=${DATA_ROOT}"
    "val_evaluator.ann_file=${DATA_ROOT}annotations/instances_val.json"
    "test_evaluator.ann_file=${DATA_ROOT}annotations/instances_test.json"
    "train_cfg.max_epochs=${TEST_MAX_EPOCHS}"
    "default_hooks.checkpoint.interval=1"
    "train_dataloader.batch_size=${TEST_TRAIN_BATCH_SIZE}"
    "val_dataloader.batch_size=${TEST_VAL_BATCH_SIZE}"
    "test_dataloader.batch_size=${TEST_TEST_BATCH_SIZE}"
)

EARLY_STOP_ARGS=()
if [[ "$ENABLE_EARLY_STOPPING" == "1" ]]; then
    EARLY_STOP_ARGS=(
        --early-stop-monitor "$EARLY_STOP_MONITOR"
        --early-stop-patience "$EARLY_STOP_PATIENCE"
        --early-stop-min-delta "$EARLY_STOP_MIN_DELTA"
        --early-stop-rule "$EARLY_STOP_RULE"
    )
    if [[ -n "$EARLY_STOP_THRESHOLD" ]]; then
        EARLY_STOP_ARGS+=(--early-stop-stopping-threshold "$EARLY_STOP_THRESHOLD")
    fi
fi

shopt -s nullglob
configs=("$CONFIG_DIR"/*.py)
shopt -u nullglob

if [[ ${#configs[@]} -eq 0 ]]; then
    echo "No config files found under $CONFIG_DIR"
    exit 1
fi

mkdir -p "$WORK_DIR_ROOT"
SUMMARY_FILE="$WORK_DIR_ROOT/run_summary.tsv"
STATS_FILE="$WORK_DIR_ROOT/model_stats.tsv"
: > "$SUMMARY_FILE"
: > "$STATS_FILE"

failed_configs=()

for config in "${configs[@]}"; do
    config_name=$(basename "$config" .py)
    work_dir="$WORK_DIR_ROOT/$config_name"
    run_status="SKIPPED"
    reason="completed"
    log_file="$work_dir/run.log"
    load_ok="unknown"

    mkdir -p "$work_dir"
    IFS=$'\t' read -r weights_source weights_detail < <(get_weight_info "$config")

    if is_model_completed "$work_dir"; then
        echo "===== SKIPPING COMPLETED: $config_name ====="
        load_ok=$(detect_weight_load_status "$log_file" "$weights_source")
        reason="already completed"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$config_name" "$run_status" "$weights_source" "$load_ok" "$reason" >> "$SUMMARY_FILE"
        if ! collect_model_stats "$config" "$work_dir" >> "$STATS_FILE" 2>> "$log_file"; then
            echo "[warn] Failed to collect model stats for $config_name" | tee -a "$log_file"
        fi
        continue
    fi


    : > "$log_file"

    echo "===== RUNNING: $config_name ====="
    if ! bash "$TRAIN_TEST_SCRIPT" \
        "$config" \
        "$NUM_GPUS" \
        "$work_dir" \
        "${COMMON_CFG_OPTIONS[@]}" \
        "${EARLY_STOP_ARGS[@]}" 2>&1 | tee -a "$log_file"; then
        echo "FAILED: $config_name"
        run_status="FAILED"
        failed_configs+=("$config_name")
    else
        run_status="OK"
    fi

    load_ok=$(detect_weight_load_status "$log_file" "$weights_source")
    if [[ "$run_status" == "OK" ]]; then
        if [[ "$load_ok" == "yes" || "$load_ok" == "n/a" ]]; then
            reason="completed"
        else
            reason="completed but weight load status is $load_ok"
        fi
    else
        reason=$(extract_failure_reason "$log_file")
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$config_name" "$run_status" "$weights_source" "$load_ok" "$reason" >> "$SUMMARY_FILE"

    if [[ "$run_status" == "OK" ]]; then
        if ! collect_model_stats "$config" "$work_dir" >> "$STATS_FILE" 2>> "$log_file"; then
            echo "[warn] Failed to collect model stats for $config_name" | tee -a "$log_file"
        fi
    fi
done

print_summary_table "$SUMMARY_FILE"
print_stats_table "$STATS_FILE"

if [[ ${#failed_configs[@]} -gt 0 ]]; then
    echo "===== FAILED CONFIGS ====="
    printf '%s\n' "${failed_configs[@]}"
    exit 1
fi

echo "All configs under $CONFIG_DIR completed successfully."
