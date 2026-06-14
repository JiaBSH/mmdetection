"""Check whether a model training run has already completed.

Extracted from submm.sh is_model_completed().
"""

from __future__ import annotations

import pathlib


def is_model_completed(work_dir: str) -> bool:
    """Return True if work_dir contains a checkpoint, test/ dir, and metric_plots/ dir.

    Checks for any .pth file matching: best_*.pth, latest.pth, epoch_*.pth, or iter_*.pth.
    """
    wd = pathlib.Path(work_dir)
    if not wd.is_dir():
        return False

    # Check for any checkpoint file
    checkpoint_patterns = ("best_*.pth", "latest.pth", "epoch_*.pth", "iter_*.pth")
    has_checkpoint = False
    for pat in checkpoint_patterns:
        if list(wd.glob(pat)):
            has_checkpoint = True
            break
    if not has_checkpoint:
        return False

    # Must have test/ and metric_plots/ subdirectories
    return (wd / "test").is_dir() and (wd / "metric_plots").is_dir()
