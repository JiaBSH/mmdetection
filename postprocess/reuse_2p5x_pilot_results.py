"""Reuse exact single-image pilot cells in the four-image coarse grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from postprocess.window_sensitivity import parameter_grid, result_filename


COARSE_WINDOW_SIZES = (96, 128, 160, 192, 256, 320, 400, 512, 640, 768)
COARSE_OVERLAP_RATIOS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)
PILOT_IMAGE = "2p5x_00016.png"
MODEL_NAME = "detectors_htc-r50_custom_coco_instance"
CHECKPOINT_NAME = "epoch_17.pth"


def reuse_pilot_results(source: Path, destination: Path) -> int:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    expected_cells = set(parameter_grid(COARSE_WINDOW_SIZES, COARSE_OVERLAP_RATIOS))
    copied_cells: set[tuple[int, float]] = set()

    for path in sorted(source.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        cell = (int(row["patch_size"]), float(row["overlap_ratio"]))
        if cell not in expected_cells or row.get("image") != PILOT_IMAGE:
            continue
        if row.get("model_name") != MODEL_NAME:
            raise ValueError(f"model mismatch in {path}")
        if Path(str(row.get("checkpoint", ""))).name != CHECKPOINT_NAME:
            raise ValueError(f"checkpoint mismatch in {path}")
        if cell in copied_cells:
            raise ValueError(f"duplicate pilot cell {cell}")

        cell_dir = destination / Path(result_filename(*cell)).stem
        cell_dir.mkdir(parents=True, exist_ok=True)
        row["reused_from"] = str(path)
        output_path = cell_dir / f"{Path(PILOT_IMAGE).stem}.json"
        output_path.write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        copied_cells.add(cell)

    if len(copied_cells) != 20:
        raise ValueError(f"expected 20 reusable cells, found {len(copied_cells)}")
    return len(copied_cells)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    reused = reuse_pilot_results(args.source, args.destination)
    print(f"reused={reused} destination={args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
