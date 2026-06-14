#!/usr/bin/env python3
"""mmdetection batch training — iterate over configs, train→test→vis, collect stats.

Usage:
    python submm.py
    DATA_ROOT=/data/path sbatch submm.sh        # SLURM wrapper forwards env vars
    python submm.py --config-dir configs/custom --max-epochs 10
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# =============================================================================
# Configuration — edit defaults here, override via environment variables
# =============================================================================

class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    PROJECT_ROOT   = Path(os.environ.get("PROJECT_ROOT", "/data/run01/scvi576/JiaBSH/mmdetection_para"))
    CONFIG_DIR     = Path(os.environ.get("CONFIG_DIR", "configs/custom_pretrain"))
    DATA_ROOT      = os.environ.get("DATA_ROOT", "dataset_root/mmdata_isat_1024/")
    WORK_DIR_ROOT  = Path(os.environ.get("WORK_DIR_ROOT",
                         f"work_dirs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    TRAIN_TEST_SCRIPT = "tools/train_then_test_instance_seg.sh"

    # ── Training ───────────────────────────────────────────────────────────
    NUM_GPUS          = int(os.environ.get("NUM_GPUS", "1"))
    MAX_EPOCHS        = int(os.environ.get("TEST_MAX_EPOCHS", "50"))
    TRAIN_BATCH_SIZE  = int(os.environ.get("TEST_TRAIN_BATCH_SIZE", "2"))
    VAL_BATCH_SIZE    = int(os.environ.get("TEST_VAL_BATCH_SIZE", "2"))
    TEST_BATCH_SIZE   = int(os.environ.get("TEST_TEST_BATCH_SIZE", "2"))

    # ── Early stopping ────────────────────────────────────────────────────
    ENABLE_EARLY_STOP = os.environ.get("ENABLE_EARLY_STOPPING", "1") == "1"
    ES_MONITOR        = os.environ.get("EARLY_STOP_MONITOR", "coco/segm_mAP")
    ES_PATIENCE       = os.environ.get("EARLY_STOP_PATIENCE", "5")
    ES_MIN_DELTA      = os.environ.get("EARLY_STOP_MIN_DELTA", "0.0")
    ES_RULE           = os.environ.get("EARLY_STOP_RULE", "greater")
    ES_THRESHOLD      = os.environ.get("EARLY_STOP_THRESHOLD", "")

    # ── Ramdisk (/dev/shm) ────────────────────────────────────────────────
    # Extract dataset into node-local memory filesystem to reduce I/O bottleneck.
    # submm.sh handles the actual tar -xf + cleanup; these flags are for reference.
    USE_RAMDISK   = os.environ.get("USE_RAMDISK", "1") == "1"
    RAMDISK_TAR   = os.environ.get("RAMDISK_TAR", "dataset_root/mmdata_isat_1024.tar")
    RAMDISK_CLEAN = os.environ.get("RAMDISK_CLEAN", "1") == "1"

    # ── Dependencies ──────────────────────────────────────────────────────
    PYTHON_DEPS = [
        ("skimage",      "scikit-image"),
        ("mmpretrain",   "mmpretrain"),
        ("instaboostfast","instaboostfast"),
    ]


# =============================================================================
# Orchestration
# =============================================================================

def main() -> None:
    cfg = Config()

    os.chdir(cfg.PROJECT_ROOT)
    sys.path.insert(0, str(cfg.PROJECT_ROOT))

    # Late imports — require PROJECT_ROOT on sys.path
    from mmdet_utils.completed import is_model_completed
    from mmdet_utils.config import get_weight_info
    from mmdet_utils.log_parser import detect_weight_load_status, extract_failure_reason
    from mmdet_utils.stats import collect_model_stats, format_stats_line
    from mmdet_utils.display import print_all

    # ── Env info ──────────────────────────────────────────────────────────
    print_env_info(cfg)

    # ── Install deps ──────────────────────────────────────────────────────
    ensure_deps(cfg.PYTHON_DEPS)

    # ── Collect configs ───────────────────────────────────────────────────
    config_dir = cfg.CONFIG_DIR if cfg.CONFIG_DIR.is_absolute() else cfg.PROJECT_ROOT / cfg.CONFIG_DIR
    config_paths = sorted(config_dir.glob("*.py"))
    if not config_paths:
        print(f"ERROR: No config files found under {config_dir}")
        sys.exit(1)
    print(f"Found {len(config_paths)} config(s) in {config_dir}")

    # ── Build cfg-options ─────────────────────────────────────────────────
    common_opts = build_cfg_options(cfg)
    early_stop_args = build_early_stop_args(cfg) if cfg.ENABLE_EARLY_STOP else []

    # ── Prepare output dirs ───────────────────────────────────────────────
    work_root = cfg.WORK_DIR_ROOT if cfg.WORK_DIR_ROOT.is_absolute() else cfg.PROJECT_ROOT / cfg.WORK_DIR_ROOT
    work_root.mkdir(parents=True, exist_ok=True)
    summary_file = work_root / "run_summary.tsv"
    stats_file = work_root / "model_stats.tsv"
    summary_file.write_text("")
    stats_file.write_text("")

    failed: list[str] = []

    # ── Training loop ─────────────────────────────────────────────────────
    for config_path in config_paths:
        name = config_path.stem
        work_dir = work_root / name
        log_file = work_dir / "run.log"
        run_status = "SKIPPED"
        reason = "completed"
        load_ok = "unknown"

        work_dir.mkdir(parents=True, exist_ok=True)

        weights_source, _ = get_weight_info(str(config_path))

        # Skip completed
        if is_model_completed(str(work_dir)):
            print(f"\n===== SKIPPING COMPLETED: {name} =====")
            load_ok = detect_weight_load_status(str(log_file), weights_source)
            append_tsv(summary_file, name, run_status, weights_source, load_ok, "already completed")
            append_stats(stats_file, config_path, work_dir)
            continue

        log_file.write_text("")

        print(f"\n===== RUNNING: {name} =====")
        print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run train→test→vis pipeline
        cmd = [
            "bash", cfg.TRAIN_TEST_SCRIPT,
            str(config_path), str(cfg.NUM_GPUS), str(work_dir),
            *common_opts, *early_stop_args,
        ]
        try:
            with open(log_file, "a") as log_f:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, cwd=cfg.PROJECT_ROOT)
                log_f.write(proc.stdout)
                sys.stdout.write(proc.stdout)
            if proc.returncode == 0:
                run_status = "OK"
            else:
                print(f"FAILED: {name}")
                run_status = "FAILED"
                failed.append(name)
        except Exception as e:
            print(f"FAILED: {name} — {e}")
            run_status = "FAILED"
            failed.append(name)

        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Outcome
        load_ok = detect_weight_load_status(str(log_file), weights_source)
        if run_status == "OK":
            reason = "completed" if load_ok in ("yes", "n/a") else f"completed but weight load status is {load_ok}"
        else:
            reason = extract_failure_reason(str(log_file))

        append_tsv(summary_file, name, run_status, weights_source, load_ok, reason)
        if run_status == "OK":
            append_stats(stats_file, config_path, work_dir)

    # ── Results ───────────────────────────────────────────────────────────
    print_all(str(summary_file), str(stats_file))

    if failed:
        print("\n===== FAILED CONFIGS =====")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)

    print(f"\nAll configs under {config_dir} completed successfully.")


# =============================================================================
# Helpers
# =============================================================================

def print_env_info(cfg: Config) -> None:
    """Print environment diagnostics."""
    def _py_info(pkg: str) -> str:
        try:
            m = __import__(pkg)
            return getattr(m, "__version__", "unknown")
        except Exception:
            return "unknown"

    print("===== ENVIRONMENT INFO =====")
    print(f"Job ID:     {os.environ.get('SLURM_JOB_ID', 'none')}")
    print(f"Node:       {os.uname().nodename}")
    print(f"Python:     {sys.version.split()[0]}")
    print(f"PyTorch:    {_py_info('torch')}")
    print(f"mmcv:       {_py_info('mmcv')}")
    print(f"mmdet:      {_py_info('mmdet')}")
    print(f"CUDA_HOME:  {os.environ.get('CUDA_HOME', 'unknown')}")
    print(f"DATA_ROOT:  {cfg.DATA_ROOT}")
    print("============================")


def ensure_deps(deps: list[tuple[str, str]]) -> None:
    """Install missing optional Python packages."""
    import importlib.util
    for module_name, package_name in deps:
        if importlib.util.find_spec(module_name) is None:
            print(f"[setup] Installing missing package: {package_name}")
            subprocess.run([sys.executable, "-m", "pip", "install", package_name, "--quiet"], check=False)


def build_cfg_options(cfg: Config) -> list[str]:
    """Build shared --cfg-options list for train.py / test.py."""
    dr = cfg.DATA_ROOT.rstrip("/")
    return [
        "--cfg-options",
        f"data_root={dr}",
        f"train_dataloader.dataset.data_root={dr}",
        f"val_dataloader.dataset.data_root={dr}",
        f"test_dataloader.dataset.data_root={dr}",
        f"val_evaluator.ann_file={dr}/annotations/instances_val.json",
        f"test_evaluator.ann_file={dr}/annotations/instances_test.json",
        f"train_cfg.max_epochs={cfg.MAX_EPOCHS}",
        "default_hooks.checkpoint.interval=1",
        f"train_dataloader.batch_size={cfg.TRAIN_BATCH_SIZE}",
        f"val_dataloader.batch_size={cfg.VAL_BATCH_SIZE}",
        f"test_dataloader.batch_size={cfg.TEST_BATCH_SIZE}",
    ]


def build_early_stop_args(cfg: Config) -> list[str]:
    """Build early-stopping CLI args."""
    args = [
        "--early-stop-monitor", cfg.ES_MONITOR,
        "--early-stop-patience", cfg.ES_PATIENCE,
        "--early-stop-min-delta", cfg.ES_MIN_DELTA,
        "--early-stop-rule", cfg.ES_RULE,
    ]
    if cfg.ES_THRESHOLD:
        args += ["--early-stop-stopping-threshold", cfg.ES_THRESHOLD]
    return args


def append_tsv(path: Path, *fields: str) -> None:
    with open(path, "a") as f:
        f.write("\t".join(fields) + "\n")


def append_stats(stats_path: Path, config_path: Path, work_dir: Path) -> None:
    try:
        from mmdet_utils.stats import collect_model_stats, format_stats_line
        line = format_stats_line(collect_model_stats(str(config_path), str(work_dir)))
        with open(stats_path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
