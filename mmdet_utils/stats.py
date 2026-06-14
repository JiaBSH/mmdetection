"""Model statistics collection: parameter count, training memory, inference time, FPS.

Extracted from submm.sh collect_model_stats() heredoc.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys


def collect_model_stats(config_path: str, work_dir: str) -> dict[str, object]:
    """Collect model statistics from a trained work directory.

    Args:
        config_path: Path to the mmdet config .py file.
        work_dir:   Path to the training work directory.

    Returns:
        dict with keys:
            name:          config stem (model name)
            params_m:      parameter count in millions, or None
            train_mem_mib: peak training GPU memory in MiB, or None
            test_time_ms:  inference time per image in ms, or None
            fps:           frames per second, or None
    """
    config_path = pathlib.Path(config_path)
    work_dir = pathlib.Path(work_dir)

    result: dict[str, object] = {
        "name": config_path.stem,
        "params_m": None,
        "train_mem_mib": None,
        "test_time_ms": None,
        "fps": None,
    }

    # ── 1. Parameter count ──────────────────────────────────────────────
    try:
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        from mmengine.config import Config
        from mmdet.registry import MODELS

        cfg = Config.fromfile(str(config_path))
        model_cfg = _deepcopy_safe(cfg.model)

        _strip_init_cfg(model_cfg)

        model = MODELS.build(model_cfg)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        del model
        result["params_m"] = params_m
    except Exception as e:
        print(f"[warn] param count failed: {e}", file=sys.stderr)

    # ── 2. Train peak memory (MiB) from scalars.json ───────────────────
    try:
        scalars_files = sorted(work_dir.glob("*/vis_data/scalars.json"))
        if scalars_files:
            mem_vals = []
            with open(scalars_files[-1]) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        if "memory" in d and ("epoch" in d or "iter" in d):
                            mem_vals.append(d["memory"])
                    except Exception:
                        pass
            if mem_vals:
                result["train_mem_mib"] = max(mem_vals)
    except Exception as e:
        print(f"[warn] train memory read failed: {e}", file=sys.stderr)

    # ── 3. Inference time + FPS from run.log ───────────────────────────
    try:
        log_file = work_dir / "run.log"
        time_pat = re.compile(r"\btime:\s*([\d.]+)")
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    if "Epoch(test)" not in line and "Iter(test)" not in line:
                        continue
                    tm = time_pat.search(line)
                    if tm:
                        t = float(tm.group(1))
                        result["test_time_ms"] = t * 1000
                        result["fps"] = 1.0 / t if t > 0 else None
    except Exception as e:
        print(f"[warn] run.log time parse failed: {e}", file=sys.stderr)

    return result


def format_stats_line(stats: dict[str, object]) -> str:
    """Format a stats dict as a tab-separated line (matching old submm.sh output)."""
    def fmt(v: object) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return "N/A"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    parts = [
        str(stats.get("name", "")),
        fmt(stats.get("params_m")) + " M",
        fmt(stats.get("train_mem_mib")) + " MiB",
        fmt(stats.get("test_time_ms")) + " ms",
        fmt(stats.get("fps")) + " fps",
    ]
    return "\t".join(parts)


# ── Helpers ────────────────────────────────────────────────────────────

def _deepcopy_safe(obj: object) -> object:
    """Best-effort deep copy using copy.deepcopy, fallback to json round-trip."""
    try:
        import copy
        return copy.deepcopy(obj)
    except Exception:
        pass

    # Fallback: use pickle-style reconstruction via Config
    try:
        from mmengine.config import Config
        return Config(obj)._cfg_dict
    except Exception:
        pass

    # Last resort: json round-trip (loses non-serializable types but ok here)
    return json.loads(json.dumps(obj, default=str))


def _strip_init_cfg(node: object) -> None:
    """Recursively remove 'init_cfg', 'load_from', 'pretrained' keys from a config dict."""
    if isinstance(node, dict):
        for key in ("init_cfg", "load_from", "pretrained"):
            node.pop(key, None)
        for v in node.values():
            _strip_init_cfg(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _strip_init_cfg(v)
