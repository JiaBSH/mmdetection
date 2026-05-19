from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.apis import init_detector
from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Profile per-image inference time and GPU memory.')
    parser.add_argument('config', help='Model config path')
    parser.add_argument('checkpoint', help='Checkpoint path')
    parser.add_argument('--ann-file', required=True, help='COCO annotation file')
    parser.add_argument('--img-dir', required=True, help='Image directory')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Inference device')
    parser.add_argument('--num-warmup', type=int, default=0,
                        help='Number of initial images to mark as warmup')
    parser.add_argument('--max-images', type=int, default=0,
                        help='Maximum images to profile; 0 means all')
    parser.add_argument('--plot-title', default=None,
                        help='Optional title prefix for saved plots')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Config overrides in key=value format')
    return parser.parse_args()


def _extract_img_path(data_batch: dict, outputs) -> str:
    data_samples = data_batch.get('data_samples')
    if isinstance(data_samples, (list, tuple)) and data_samples:
        sample = data_samples[0]
        img_path = getattr(sample, 'img_path', None)
        if img_path:
            return str(img_path)
    if isinstance(outputs, (list, tuple)) and outputs:
        sample = outputs[0]
        img_path = getattr(sample, 'img_path', None)
        if img_path:
            return str(img_path)
    return ''


def _save_plot(rows: list[dict], value_key: str, ylabel: str, title: str,
               out_path: Path) -> None:
    if not rows:
        return

    x = np.arange(len(rows))
    y = np.array([float(row[value_key]) for row in rows], dtype=float)
    fig_w = max(10, min(20, len(rows) * 0.35))
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    ax.plot(x, y, marker='o', linewidth=1.5, markersize=4)
    ax.set_xlabel('Image Index')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.35)

    if len(rows) <= 20:
        labels = [row['image'] for row in rows]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        tick_count = min(12, len(rows))
        tick_idx = np.linspace(0, len(rows) - 1, num=tick_count, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([rows[idx]['image'] for idx in tick_idx],
                           rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    dataloader_cfg = cfg.test_dataloader.copy()
    data_loader = Runner.build_dataloader(dataloader_cfg)

    device = torch.device(args.device)
    use_cuda = device.type == 'cuda' and torch.cuda.is_available()

    rows: list[dict] = []
    max_images = int(args.max_images)
    num_warmup = max(0, int(args.num_warmup))
    title_prefix = args.plot_title or Path(args.config).stem

    with torch.no_grad():
        for idx, data_batch in enumerate(data_loader):
            if max_images > 0 and idx >= max_images:
                break

            if use_cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            start_time = time.perf_counter()
            outputs = model.test_step(data_batch)
            if use_cuda:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            img_path = _extract_img_path(data_batch, outputs)
            img_name = os.path.basename(img_path) if img_path else f'image_{idx:05d}'

            if use_cuda:
                peak_allocated_mb = (
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                )
                peak_reserved_mb = (
                    torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
                )
            else:
                peak_allocated_mb = float('nan')
                peak_reserved_mb = float('nan')

            rows.append({
                'index': idx,
                'image': img_name,
                'img_path': img_path,
                'elapsed_ms': float(elapsed_ms),
                'peak_allocated_mb': float(peak_allocated_mb),
                'peak_reserved_mb': float(peak_reserved_mb),
                'is_warmup': idx < num_warmup,
            })

    csv_path = out_dir / 'per_image_profile.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'index', 'image', 'img_path', 'elapsed_ms',
                'peak_allocated_mb', 'peak_reserved_mb', 'is_warmup'
            ])
        writer.writeheader()
        writer.writerows(rows)

    effective_rows = [row for row in rows if not row['is_warmup']]
    if not effective_rows:
        effective_rows = rows

    def _stats(key: str) -> dict:
        values = np.asarray([float(row[key]) for row in effective_rows], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return {'mean': float('nan'), 'median': float('nan'), 'max': float('nan')}
        return {
            'mean': float(np.mean(finite_values)),
            'median': float(np.median(finite_values)),
            'max': float(np.max(finite_values)),
        }

    time_stats = _stats('elapsed_ms')
    alloc_stats = _stats('peak_allocated_mb')
    reserved_stats = _stats('peak_reserved_mb')

    summary = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'ann_file': args.ann_file,
        'img_dir': args.img_dir,
        'device': args.device,
        'num_images': len(rows),
        'num_warmup': num_warmup,
        'mean_time_ms': time_stats['mean'],
        'median_time_ms': time_stats['median'],
        'max_time_ms': time_stats['max'],
        'mean_peak_allocated_mb': alloc_stats['mean'],
        'median_peak_allocated_mb': alloc_stats['median'],
        'max_peak_allocated_mb': alloc_stats['max'],
        'mean_peak_reserved_mb': reserved_stats['mean'],
        'median_peak_reserved_mb': reserved_stats['median'],
        'max_peak_reserved_mb': reserved_stats['max'],
        'csv_path': str(csv_path),
    }

    summary_path = out_dir / 'profile_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _save_plot(
        rows,
        'elapsed_ms',
        'Time (ms / image)',
        f'{title_prefix} Per-image Inference Time',
        out_dir / 'per_image_time.png',
    )
    _save_plot(
        rows,
        'peak_allocated_mb',
        'Peak GPU Memory (MB)',
        f'{title_prefix} Per-image Peak GPU Memory',
        out_dir / 'per_image_memory.png',
    )

    print(f'✅ Saved per-image profile CSV: {csv_path}')
    print(f'✅ Saved profile summary JSON: {summary_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())