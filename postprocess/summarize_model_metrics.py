from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


GEOMETRY_METRICS = [
    'Width',
    'Height',
    'Area',
    'SideDistance',
    'DiagonalLength',
    'DiagonalAngle',
    'EdgeAngle',
]

AVERAGE_METRICS = ['iou', 'precision', 'recall', 'f1']
SUMMARY_FIELDNAMES = [
    'model',
    'iou_mean',
    'precision_mean',
    'recall_mean',
    'f1_mean',
    'count_mae',
    'coverage_mae',
    *[f'{metric}_mae' for metric in GEOMETRY_METRICS],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Summarize one row per model from matched_metrics_merged.csv and '
            'metrics_summary.csv.'
        )
    )
    parser.add_argument(
        'root',
        type=Path,
        help='Root directory that contains model folders, such as outputs/.../plain',
    )
    parser.add_argument(
        '--matched-name',
        default='matched_metrics_merged.csv',
        help='Merged matched metrics filename inside each model directory.',
    )
    parser.add_argument(
        '--summary-name',
        default='metrics_summary.csv',
        help='Per-model summary filename inside each model directory.',
    )
    parser.add_argument(
        '--output-name',
        default='model_metrics_mae_summary.csv',
        help='Output CSV filename written to the parent directory.',
    )
    return parser.parse_args()


def to_float(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if math.isnan(number):
        return None
    return number


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def compute_geometry_mae(matched_csv: Path) -> dict[str, float | None]:
    metric_diffs: dict[str, list[float]] = {metric: [] for metric in GEOMETRY_METRICS}

    with matched_csv.open('r', newline='', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for metric in GEOMETRY_METRICS:
                gt_value = to_float(row.get(f'gt_{metric}', ''))
                pred_value = to_float(row.get(f'pred_{metric}', ''))
                if gt_value is None or pred_value is None:
                    continue
                metric_diffs[metric].append(abs(pred_value - gt_value))

    return {f'{metric}_mae': mean(metric_diffs[metric]) for metric in GEOMETRY_METRICS}


def compute_summary_metrics(summary_csv: Path) -> dict[str, float | None]:
    averages: dict[str, list[float]] = {metric: [] for metric in AVERAGE_METRICS}
    count_abs_errors: list[float] = []
    coverage_abs_errors: list[float] = []

    with summary_csv.open('r', newline='', encoding='utf-8-sig') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for metric in AVERAGE_METRICS:
                value = to_float(row.get(metric, ''))
                if value is not None:
                    averages[metric].append(value)

            pred_count = to_float(row.get('pred_count', ''))
            gt_count = to_float(row.get('gt_count', ''))
            if pred_count is not None and gt_count is not None:
                count_abs_errors.append(abs(pred_count - gt_count))

            pred_coverage = to_float(row.get('pred_coverage', ''))
            gt_coverage = to_float(row.get('gt_coverage', ''))
            if pred_coverage is not None and gt_coverage is not None:
                coverage_abs_errors.append(abs(pred_coverage - gt_coverage))

    count_mae = mean(count_abs_errors)
    coverage_mae = mean(coverage_abs_errors)
    return {
        'iou_mean': mean(averages['iou']),
        'precision_mean': mean(averages['precision']),
        'recall_mean': mean(averages['recall']),
        'f1_mean': mean(averages['f1']),
        'count_mae': count_mae,
        'coverage_mae': coverage_mae,
    }


def format_value(value: float | None) -> str:
    if value is None:
        return ''
    return f'{value:.15g}'


def summarize_model_dir(
    model_dir: Path,
    matched_name: str,
    summary_name: str,
) -> dict[str, float | str | None] | None:
    summary_csv = model_dir / summary_name
    if not summary_csv.is_file():
        return None

    row: dict[str, float | str | None] = {'model': model_dir.name}
    row.update(compute_summary_metrics(summary_csv))

    matched_csv = model_dir / matched_name
    if matched_csv.is_file():
        row.update(compute_geometry_mae(matched_csv))
    else:
        for metric in GEOMETRY_METRICS:
            row[f'{metric}_mae'] = None

    return row


def collect_model_summary_rows(
    root_dir: Path,
    matched_name: str,
    summary_name: str,
) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []
    for model_dir in sorted(child for child in root_dir.iterdir() if child.is_dir()):
        row = summarize_model_dir(model_dir, matched_name, summary_name)
        if row is None:
            print(f'[skip] {model_dir.name}: missing {summary_name}')
            continue
        rows.append(row)
    return rows


def write_model_summary_csv(
    root_dir: Path,
    output_name: str,
    matched_name: str = 'matched_metrics_merged.csv',
    summary_name: str = 'metrics_summary.csv',
) -> tuple[Path, int]:
    rows = collect_model_summary_rows(root_dir, matched_name, summary_name)
    if not rows:
        raise ValueError(
            f'No model directories with {summary_name} were found under {root_dir}'
        )

    output_path = root_dir / output_name
    with output_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: format_value(row.get(key)) if key != 'model' else row.get(key, '')
                    for key in SUMMARY_FIELDNAMES
                }
            )

    return output_path, len(rows)


def main() -> int:
    args = parse_args()
    root_dir = args.root.expanduser().resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(f'Root directory does not exist: {root_dir}')

    try:
        output_path, row_count = write_model_summary_csv(
            root_dir,
            args.output_name,
            matched_name=args.matched_name,
            summary_name=args.summary_name,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    print(f'Wrote {row_count} model summaries to {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())