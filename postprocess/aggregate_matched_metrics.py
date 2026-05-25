from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Aggregate matched_metrics.csv from each image folder into one '
            'CSV per model directory.'
        )
    )
    parser.add_argument(
        'root',
        type=Path,
        help='Root directory that contains model folders, such as outputs/.../plain',
    )
    parser.add_argument(
        '--output-name',
        default='matched_metrics_merged.csv',
        help='Filename written into each model directory.',
    )
    return parser.parse_args()


def collect_model_rows(model_dir: Path, output_name: str) -> tuple[list[str] | None, list[list[str]], int]:
    header: list[str] | None = None
    rows: list[list[str]] = []
    source_count = 0

    for image_dir in sorted(child for child in model_dir.iterdir() if child.is_dir()):
        csv_path = image_dir / 'matched_metrics.csv'
        if not csv_path.is_file():
            continue
        if csv_path.name == output_name and csv_path.parent == model_dir:
            continue

        with csv_path.open('r', newline='', encoding='utf-8-sig') as csv_file:
            reader = csv.reader(csv_file)
            current_header = next(reader, None)
            if not current_header:
                continue

            if header is None:
                header = current_header
            elif current_header != header:
                raise ValueError(
                    f'Header mismatch in {csv_path}: expected {header}, got {current_header}'
                )

            source_count += 1
            for row in reader:
                if not row:
                    continue
                rows.append([image_dir.name, *row])

    return header, rows, source_count


def aggregate_model_dir(model_dir: Path, output_name: str) -> tuple[int, int] | None:
    header, rows, source_count = collect_model_rows(model_dir, output_name)
    if header is None or source_count == 0:
        return None

    output_path = model_dir / output_name
    with output_path.open('w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image_folder', *header])
        writer.writerows(rows)

    return source_count, len(rows)


def aggregate_root_dir(root_dir: Path, output_name: str) -> int:
    aggregated_models = 0
    for model_dir in sorted(child for child in root_dir.iterdir() if child.is_dir()):
        result = aggregate_model_dir(model_dir, output_name)
        if result is None:
            print(f'[skip] {model_dir.name}: no per-image matched_metrics.csv found')
            continue

        source_count, row_count = result
        aggregated_models += 1
        print(
            f'[ok] {model_dir.name}: merged {source_count} files and {row_count} rows '
            f'into {output_name}'
        )

    return aggregated_models


def main() -> int:
    args = parse_args()
    root_dir = args.root.expanduser().resolve()

    if not root_dir.is_dir():
        raise NotADirectoryError(f'Root directory does not exist: {root_dir}')

    aggregated_models = aggregate_root_dir(root_dir, args.output_name)

    if aggregated_models == 0:
        print(f'No model directories with matched_metrics.csv were found under {root_dir}')
        return 1

    print(f'Aggregated {aggregated_models} model directories under {root_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())