"""Reuse exact four-image coarse cells in the common fine grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from postprocess.window_sensitivity import parameter_grid, result_filename


FINE_WINDOW_SIZES = (96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 400, 416, 448)
FINE_OVERLAP_RATIOS = (
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
)
EXPECTED_IMAGES = tuple(f"2p5x_{number:05d}.png" for number in range(16, 20))
MODEL_NAME = "detectors_htc-r50_custom_coco_instance"
CHECKPOINT_NAME = "epoch_17.pth"


def reuse_coarse_results(source: Path, destination: Path) -> int:
    source = source.expanduser().resolve()
    destination = destination.expanduser().resolve()
    fine_cells = set(parameter_grid(FINE_WINDOW_SIZES, FINE_OVERLAP_RATIOS))
    copied: set[tuple[int, float, str]] = set()

    for path in sorted(source.rglob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        cell = (int(row["patch_size"]), float(row["overlap_ratio"]))
        image = str(row["image"])
        key = cell + (image,)
        if cell not in fine_cells or image not in EXPECTED_IMAGES:
            continue
        if row.get("model_name") != MODEL_NAME:
            raise ValueError(f"model mismatch in {path}")
        if Path(str(row.get("checkpoint", ""))).name != CHECKPOINT_NAME:
            raise ValueError(f"checkpoint mismatch in {path}")
        if key in copied:
            raise ValueError(f"duplicate coarse result {key}")

        cell_dir = destination / Path(result_filename(*cell)).stem
        cell_dir.mkdir(parents=True, exist_ok=True)
        if "reused_from" in row:
            row["original_reused_from"] = row["reused_from"]
        row["reused_from"] = str(path)
        (cell_dir / f"{Path(image).stem}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        copied.add(key)

    if len(copied) != 168:
        raise ValueError(f"expected 168 reusable records, found {len(copied)}")
    return len(copied)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    reused = reuse_coarse_results(args.source, args.destination)
    print(f"reused={reused} destination={args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
