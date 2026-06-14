#!/usr/bin/env python3
"""
Generate config variants with load_from disabled, for training when
pretrained COCO checkpoints are unavailable (e.g., domain expired).

These configs use torchvision://resnet50 backbone pretrained weights only,
randomly initializing detection heads. Suitable for single-class fine-tuning
with adequate data.

Usage:
    python tools/generate_no_pretrain_configs.py [--output-dir configs/custom_scratch]
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scratch-training configs")
    parser.add_argument("--input-dir", default="configs/custom_pretrain",
                        help="Directory containing configs with load_from URLs")
    parser.add_argument("--output-dir", default="configs/custom_scratch",
                        help="Output directory for scratch-training configs")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / args.input_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for config_path in sorted(input_dir.glob("*.py")):
        content = config_path.read_text()

        # Check if config has load_from
        if "load_from" not in content:
            print(f"  SKIP (no load_from): {config_path.name}")
            continue

        # Comment out load_from lines and add explanatory note
        lines = content.split("\n")
        new_lines = []
        in_load_from = False
        for line in lines:
            # Detect multi-line load_from assignment
            if re.match(r"^\s*load_from\s*=", line):
                in_load_from = True
                # Insert override before commenting out
                new_lines.append("# === load_from disabled (pretrained COCO checkpoint unavailable) ===")
                new_lines.append("load_from = None")
                new_lines.append(f"# Original: {line.lstrip()}")
                continue

            if in_load_from:
                if line.rstrip().endswith(")") or (not line.strip().startswith(("'", '"')) and ")" not in line):
                    # Check if this line closes the paren
                    if ")" in line and "'" not in line.split("#")[0]:
                        # Just a closing paren
                        new_lines.append(f"# {line}")
                        in_load_from = False
                        continue
                    elif line.rstrip().endswith(")") and ("'" in line or '"' in line):
                        # Last line of load_from with content
                        new_lines.append(f"# {line}")
                        in_load_from = False
                        continue
                    else:
                        in_load_from = False

                # Continuation of load_from string
                new_lines.append(f"# {line}")
                continue

            new_lines.append(line)

        output_path = output_dir / config_path.name
        output_path.write_text("\n".join(new_lines) + "\n")
        print(f"  OK: {config_path.name}")
        count += 1

    print(f"\nGenerated {count} scratch-training configs in {output_dir}")
    print(f"Use --config-dir {args.output_dir} when running submm.py")


if __name__ == "__main__":
    main()
