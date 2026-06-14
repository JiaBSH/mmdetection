"""Table formatting for run summary and model statistics.

Extracted from submm.sh print_stats_table() and print_summary_table().
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def print_stats_table(stats_file: str) -> None:
    """Read a tab-separated stats file and print a formatted MODEL STATS table.

    Input format (per line):  model \t params_m \t train_mem_mib \t test_time_ms \t fps
    """
    rows = _read_tsv(stats_file, ncols=5)
    if not rows:
        return

    print()
    print("===== MODEL STATS =====")
    header = f"{'MODEL':<45} | {'PARAMS':>10} | {'TRAIN MEM':>12} | {'INF TIME':>12} | {'FPS':>9}"
    print(header)
    print("-" * len(header))
    for cols in rows:
        model, params, tmem, inft, fps = cols
        print(f"{model:<45} | {params:>10} | {tmem:>12} | {inft:>12} | {fps:>9}")


def print_summary_table(summary_file: str) -> None:
    """Read a tab-separated summary file and print a formatted RUN SUMMARY table.

    Input format (per line):  model \t run_status \t weights_source \t load_ok \t reason
    """
    rows = _read_tsv(summary_file, ncols=5)
    if not rows:
        return

    print()
    print("===== RUN SUMMARY =====")
    header = f"{'MODEL':<45} | {'RUN':>10} | {'WEIGHTS':>12} | {'LOAD_OK':>10} | {'REASON'}"
    print(header)
    # Separator line matching old format
    sep = (
        f"{'-' * 45}-+-"
        f"{'-' * 10}-+-"
        f"{'-' * 12}-+-"
        f"{'-' * 10}-+-"
        f"{'-' * 30}"
    )
    print(sep)
    for cols in rows:
        model, run_status, weights_source, load_ok, reason = cols
        print(f"{model:<45} | {run_status:>10} | {weights_source:>12} | {load_ok:>10} | {reason}")


def print_all(summary_file: str, stats_file: str) -> None:
    """Print both tables — matches old submm.sh final output."""
    print_summary_table(summary_file)
    print_stats_table(stats_file)


# ── Helpers ────────────────────────────────────────────────────────────

def _read_tsv(path: str, ncols: int = 5) -> list[list[str]]:
    """Read a tab-separated file, returning rows with exactly ncols columns."""
    rows: list[list[str]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                cols = line.split("\t")
                # Pad or truncate to exactly ncols
                if len(cols) < ncols:
                    cols += [""] * (ncols - len(cols))
                rows.append(cols[:ncols])
    except FileNotFoundError:
        print(f"[warn] File not found: {path}", file=sys.stderr)
    return rows
