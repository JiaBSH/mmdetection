"""Parse training log files for failure reasons and weight load status.

Extracted from submm.sh extract_failure_reason() and detect_weight_load_status() heredocs.
"""

from __future__ import annotations

import pathlib


def extract_failure_reason(log_path: str) -> str:
    """Scan a training log file and return the most specific failure reason.

    Priority order (last matching line within each priority tier):
      1. RuntimeError / ImportError / ModuleNotFoundError / FileNotFoundError /
         AssertionError / ValueError / KeyError
      2. Lines containing 'No module named', 'not installed', 'FAILED'
      3. Last non-empty line in the file
      4. Fallback: 'unknown failure' or 'no log captured'

    Returns a single-line string suitable for the run_summary.tsv "reason" column.
    """
    log_path = pathlib.Path(log_path)
    if not log_path.exists():
        return "no log captured"

    lines = [
        line.strip()
        for line in log_path.read_text(errors="ignore").splitlines()
        if line.strip()
    ]
    if not lines:
        return "no log captured"

    # Priority 1: Python exception prefixes (last occurrence wins)
    priority_prefixes = (
        "RuntimeError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "FileNotFoundError:",
        "AssertionError:",
        "ValueError:",
        "KeyError:",
    )
    for prefix in priority_prefixes:
        for line in reversed(lines):
            if line.startswith(prefix):
                return line

    # Priority 2: Known error keywords
    keywords = ("No module named", "not installed", "FAILED")
    for line in reversed(lines):
        if any(kw in line for kw in keywords):
            return line

    # Priority 3: Last non-empty line
    if lines:
        return lines[-1]

    return "unknown failure"


def detect_weight_load_status(log_path: str, weights_source: str) -> str:
    """Detect whether checkpoint weights were successfully loaded.

    Args:
        log_path: Path to the training run.log file.
        weights_source: The source type ("load_from", "init_cfg", "none", etc.)

    Returns:
        "n/a"      when weights_source is "none"
        "yes"      when 'Loads checkpoint by' or 'Load checkpoint from' found
        "no"       when FileNotFoundError / RuntimeError / URLError / HTTPError /
                    'No such file or directory' found (checked in reverse)
        "unknown"  otherwise
    """
    if weights_source == "none":
        return "n/a"

    log_path = pathlib.Path(log_path)
    if not log_path.exists():
        return "no"

    lines = log_path.read_text(errors="ignore").splitlines()

    load_markers = ("Loads checkpoint by", "Load checkpoint from")
    for line in lines:
        if any(marker in line for marker in load_markers):
            return "yes"

    failure_markers = (
        "FileNotFoundError:",
        "RuntimeError:",
        "URLError",
        "HTTPError",
        "No such file or directory",
    )
    for line in reversed(lines):
        if any(marker in line for marker in failure_markers):
            return "no"

    return "unknown"
