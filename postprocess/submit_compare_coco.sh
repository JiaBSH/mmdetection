#!/bin/bash
#SBATCH --job-name=coco_t1024
#SBATCH -p qgpu_4090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00

module purge
source /hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh
conda activate openmmlab2

set -euo pipefail

cd /hpcfs/fhome/sunxc/JiaBSH/mmdetection

echo "===== test_set_1024 COCO mask AP ====="

MODEL_ROOT="${MODEL_ROOT:-work_dirs/custom_all_main}"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"
COCO_OUT_ROOT="${COCO_OUT_ROOT:-outputs/coco_eval}"
MODEL_CFG="${MODEL_CFG:-postprocess/model_list.yaml}"
VISUALIZE="${VISUALIZE:-0}"
VIS_DIR_NAME="${VIS_DIR_NAME:-visualizations}"
VIS_SCORE_THR="${VIS_SCORE_THR:-0.3}"
SUMMARY_DIR_NAME="${SUMMARY_DIR_NAME:-summary}"
PROFILE_PER_IMAGE="${PROFILE_PER_IMAGE:-1}"
PROFILE_DIR_NAME="${PROFILE_DIR_NAME:-profile}"
PROFILE_NUM_WARMUP="${PROFILE_NUM_WARMUP:-0}"
PROFILE_MAX_IMAGES="${PROFILE_MAX_IMAGES:-0}"

resolve_checkpoint() {
    local checkpoint_template="$1"
    local checkpoint_path

    if [[ "${checkpoint_template}" == *"{epoch}"* ]]; then
        if [[ -n "${CHECKPOINT_EPOCH}" ]]; then
            checkpoint_path="${checkpoint_template//\{epoch\}/${CHECKPOINT_EPOCH}}"
        else
            local checkpoint_dir
            checkpoint_dir="$(dirname "${checkpoint_template}")"

            if [[ -f "${checkpoint_dir}/last_checkpoint" ]]; then
                local last_checkpoint
                last_checkpoint="$(<"${checkpoint_dir}/last_checkpoint")"
                if [[ "${last_checkpoint}" = /* ]]; then
                    checkpoint_path="${last_checkpoint}"
                else
                    checkpoint_path="${checkpoint_dir}/${last_checkpoint}"
                fi
            else
                local latest_checkpoint
                latest_checkpoint="$(find "${checkpoint_dir}" -maxdepth 1 -name 'epoch_*.pth' | sort -V | tail -n 1 || true)"
                if [[ -z "${latest_checkpoint}" ]]; then
                    echo "无法解析 checkpoint: ${checkpoint_template}" >&2
                    exit 1
                fi
                checkpoint_path="${latest_checkpoint}"
            fi
        fi
    else
        checkpoint_path="${checkpoint_template}"
    fi

    printf '%s\n' "${checkpoint_path}"
}

generate_coco_summary() {
    local summary_dir="${COCO_OUT_ROOT}/${SUMMARY_DIR_NAME}"
    mkdir -p "${summary_dir}"

    python - "${COCO_OUT_ROOT}" "${summary_dir}" "${VIS_DIR_NAME}" "${PROFILE_DIR_NAME}" <<'PY'
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

out_root = Path(sys.argv[1])
summary_dir = Path(sys.argv[2])
vis_dir_name = sys.argv[3]
profile_dir_name = sys.argv[4]

metric_keys = [
    'coco/segm_mAP',
    'coco/segm_mAP_50',
    'coco/segm_mAP_75',
    'coco/segm_mAP_s',
    'coco/segm_mAP_m',
    'coco/segm_mAP_l',
    'time',
    'data_time',
]

rows = []
timestamp_pattern = re.compile(r'^\d{8}_\d{6}$')

for split_dir in sorted(out_root.iterdir()):
    if not split_dir.is_dir() or split_dir.name == summary_dir.name:
        continue
    plain_dir = split_dir / 'plain'
    if not plain_dir.is_dir():
        continue

    for model_dir in sorted(plain_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        timestamp_dirs = sorted(
            [p for p in model_dir.iterdir() if p.is_dir() and timestamp_pattern.match(p.name)],
            key=lambda p: p.name,
        )
        if not timestamp_dirs:
            continue
        latest_dir = timestamp_dirs[-1]
        metrics_json = latest_dir / f'{latest_dir.name}.json'
        if not metrics_json.is_file():
            json_candidates = sorted(
                [p for p in latest_dir.glob('*.json') if p.is_file()],
                key=lambda p: p.name,
            )
            if not json_candidates:
                continue
            metrics_json = json_candidates[-1]

        with open(metrics_json, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        vis_dir = latest_dir / vis_dir_name
        profile_dir = model_dir / profile_dir_name
        profile_summary_json = profile_dir / 'profile_summary.json'
        row = {
            'split': split_dir.name,
            'mode': 'plain',
            'model': model_dir.name,
            'run_dir': str(latest_dir),
            'metrics_json': str(metrics_json),
            'vis_dir': str(vis_dir) if vis_dir.exists() else '',
            'profile_dir': str(profile_dir) if profile_dir.exists() else '',
            'profile_summary_json': str(profile_summary_json) if profile_summary_json.exists() else '',
            'profile_csv': str(profile_dir / 'per_image_profile.csv') if (profile_dir / 'per_image_profile.csv').exists() else '',
            'profile_time_plot': str(profile_dir / 'per_image_time.png') if (profile_dir / 'per_image_time.png').exists() else '',
            'profile_memory_plot': str(profile_dir / 'per_image_memory.png') if (profile_dir / 'per_image_memory.png').exists() else '',
        }
        for key in metric_keys:
            row[key] = metrics.get(key, float('nan'))
        if profile_summary_json.exists():
            with open(profile_summary_json, 'r', encoding='utf-8') as f:
                profile_summary = json.load(f)
            for key in [
                'mean_time_ms',
                'median_time_ms',
                'max_time_ms',
                'mean_peak_allocated_mb',
                'median_peak_allocated_mb',
                'max_peak_allocated_mb',
                'mean_peak_reserved_mb',
                'median_peak_reserved_mb',
                'max_peak_reserved_mb',
                'num_images',
                'num_warmup',
            ]:
                row[key] = profile_summary.get(key, float('nan'))
        else:
            for key in [
                'mean_time_ms',
                'median_time_ms',
                'max_time_ms',
                'mean_peak_allocated_mb',
                'median_peak_allocated_mb',
                'max_peak_allocated_mb',
                'mean_peak_reserved_mb',
                'median_peak_reserved_mb',
                'max_peak_reserved_mb',
                'num_images',
                'num_warmup',
            ]:
                row[key] = float('nan')
        rows.append(row)

if not rows:
    print('⚠️ 未找到可汇总的 COCO 评估结果')
    raise SystemExit(0)

summary_csv = summary_dir / 'coco_summary.csv'
fieldnames = [
    'split',
    'mode',
    'model',
    'coco/segm_mAP',
    'coco/segm_mAP_50',
    'coco/segm_mAP_75',
    'coco/segm_mAP_s',
    'coco/segm_mAP_m',
    'coco/segm_mAP_l',
    'time',
    'data_time',
    'run_dir',
    'metrics_json',
    'vis_dir',
    'profile_dir',
    'profile_summary_json',
    'profile_csv',
    'profile_time_plot',
    'profile_memory_plot',
    'mean_time_ms',
    'median_time_ms',
    'max_time_ms',
    'mean_peak_allocated_mb',
    'median_peak_allocated_mb',
    'max_peak_allocated_mb',
    'mean_peak_reserved_mb',
    'median_peak_reserved_mb',
    'max_peak_reserved_mb',
    'num_images',
    'num_warmup',
]

with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

by_split = defaultdict(list)
for row in rows:
    by_split[row['split']].append(row)

for split, split_rows in sorted(by_split.items()):
    split_csv = summary_dir / f'{split}_summary.csv'
    with open(split_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def _to_float(v):
        try:
            return float(v)
        except Exception:
            return float('nan')

    def _plot_grouped(rows_for_plot, metric_names, title, out_path):
        labels = [row['model'] for row in rows_for_plot]
        if not labels:
            return
        x = list(range(len(labels)))
        width = 0.8 / max(1, len(metric_names))
        fig_w = max(8, len(labels) * 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, 8))
        for idx, metric in enumerate(metric_names):
            values = [_to_float(row.get(metric, float('nan'))) for row in rows_for_plot]
            offset = (idx - (len(metric_names) - 1) / 2.0) * width
            bars = ax.bar([v + offset for v in x], values, width, label=metric.replace('coco/', ''))
            for bar, value in zip(bars, values):
                if math.isfinite(value):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        value + 0.01,
                        f'{value:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                    )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        all_values = []
        for metric in metric_names:
            for row in rows_for_plot:
                value = _to_float(row.get(metric, float('nan')))
                if math.isfinite(value):
                    all_values.append(value)
        ymax = max(all_values) if all_values else 1.0
        ax.set_ylim(0, max(1.05, ymax * 1.15))
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    def _plot_single_metric(rows_for_plot, metric_name, title, out_path, ylabel):
        labels = [row['model'] for row in rows_for_plot]
        if not labels:
            return
        values = [_to_float(row.get(metric_name, float('nan'))) for row in rows_for_plot]
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            return

        x = list(range(len(labels)))
        fig_w = max(8, len(labels) * 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, 8))
        bars = ax.bar(x, values, width=0.6)
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + max(finite_values) * 0.02,
                    f'{value:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylim(0, max(finite_values) * 1.15)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

    rows_sorted = sorted(rows, key=lambda row: (row['split'], row['model']))
    _plot_grouped(
        rows_sorted,
        ['coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75'],
        'COCO Segm AP Summary',
        summary_dir / 'coco_summary_main.png',
    )

    for split, split_rows in sorted(by_split.items()):
        split_rows = sorted(split_rows, key=lambda row: row['model'])
        _plot_grouped(
            split_rows,
            ['coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75'],
            f'{split} COCO Segm AP',
            summary_dir / f'{split}_main.png',
        )
        _plot_grouped(
            split_rows,
            ['coco/segm_mAP_s', 'coco/segm_mAP_m', 'coco/segm_mAP_l'],
            f'{split} COCO Segm AP by Size',
            summary_dir / f'{split}_size.png',
        )

    _plot_single_metric(
        rows_sorted,
        'mean_time_ms',
        'Mean Per-image Inference Time',
        summary_dir / 'coco_summary_time.png',
        'Time (ms / image)',
    )
    _plot_single_metric(
        rows_sorted,
        'mean_peak_allocated_mb',
        'Mean Per-image Peak GPU Memory',
        summary_dir / 'coco_summary_memory.png',
        'Peak GPU Memory (MB)',
    )

    for split, split_rows in sorted(by_split.items()):
        split_rows = sorted(split_rows, key=lambda row: row['model'])
        _plot_single_metric(
            split_rows,
            'mean_time_ms',
            f'{split} Mean Per-image Inference Time',
            summary_dir / f'{split}_time.png',
            'Time (ms / image)',
        )
        _plot_single_metric(
            split_rows,
            'mean_peak_allocated_mb',
            f'{split} Mean Per-image Peak GPU Memory',
            summary_dir / f'{split}_memory.png',
            'Peak GPU Memory (MB)',
        )
except Exception as exc:
    print(f'⚠️ 绘图失败: {exc}')

print(f'✅ 汇总CSV: {summary_csv}')
for split in sorted(by_split):
    print(f'✅ 分组CSV: {summary_dir / (split + "_summary.csv")}')
print(f'✅ 汇总目录: {summary_dir}')
PY
}

run_coco_eval() {
    local split_name="$1"
    local ann_file="$2"
    local img_dir="$3"

    echo "===== ${split_name} | plain | COCO segm AP ====="

    mapfile -t model_rows < <(
        python - "${MODEL_CFG}" <<'PY'
import sys
import yaml
from pathlib import Path

yaml_path = Path(sys.argv[1])
with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

for model in data.get("models", []):
    print(f"{model['name']}\t{model['config']}\t{model['checkpoint']}")
PY
    )

    for row in "${model_rows[@]}"; do
        IFS=$'\t' read -r model_name rel_config rel_checkpoint <<< "${row}"

        local config_path="${MODEL_ROOT}/${rel_config}"
        local checkpoint_template="${MODEL_ROOT}/${rel_checkpoint}"
        local checkpoint_path
        checkpoint_path="$(resolve_checkpoint "${checkpoint_template}")"

        local work_dir="${COCO_OUT_ROOT}/${split_name}/plain/${model_name}"
        mkdir -p "${work_dir}"

        echo "--- ${model_name}"
        echo "config: ${config_path}"
        echo "checkpoint: ${checkpoint_path}"
        echo "work_dir: ${work_dir}"

        local test_args=(
            tools/test.py
            "${config_path}"
            "${checkpoint_path}"
            --work-dir "${work_dir}"
        )

        if [[ "${VISUALIZE}" == "1" ]]; then
            test_args+=(--show-dir "${VIS_DIR_NAME}")
        fi

        test_args+=(
            --cfg-options
                "data_root="
                "test_dataloader.dataset.data_root="
                "test_dataloader.dataset.ann_file=${ann_file}"
                "test_dataloader.dataset.data_prefix.img=${img_dir}/"
                "test_evaluator.ann_file=${ann_file}"
                "test_evaluator.metric=segm"
                "default_hooks.visualization.score_thr=${VIS_SCORE_THR}"
        )

        python "${test_args[@]}"

        if [[ "${PROFILE_PER_IMAGE}" == "1" ]]; then
            local profile_dir="${work_dir}/${PROFILE_DIR_NAME}"
            mkdir -p "${profile_dir}"
            python postprocess/profile_inference.py \
                "${config_path}" \
                "${checkpoint_path}" \
                --ann-file "${ann_file}" \
                --img-dir "${img_dir}" \
                --out-dir "${profile_dir}" \
                --device cuda:0 \
                --num-warmup "${PROFILE_NUM_WARMUP}" \
                --max-images "${PROFILE_MAX_IMAGES}" \
                --plot-title "${split_name} | ${model_name}" \
                --cfg-options \
                    "data_root=" \
                    "test_dataloader.dataset.data_root=" \
                    "test_dataloader.dataset.ann_file=${ann_file}" \
                    "test_dataloader.dataset.data_prefix.img=${img_dir}/"
        fi
    done
}

# 如需启用 20x / 50x / 100x，取消下面对应调用的注释即可


run_coco_eval \
    "50x_unsup" \
    "dataset_root/test_set_1024/annotations/instances_50x.json" \
    "dataset_root/test_set_1024/images/50x"

generate_coco_summary

echo "===== 完成 ====="