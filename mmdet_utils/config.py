"""Extract weight/checkpoint information from mmengine config files.

Extracted from submm.sh get_weight_info() heredoc.
"""

from __future__ import annotations

from typing import Optional

from mmengine.config import Config


def _find_checkpoint(node: object) -> Optional[str]:
    """Recursively search a config subtree for init_cfg.checkpoint."""
    if isinstance(node, dict):
        init_cfg = node.get("init_cfg")
        if isinstance(init_cfg, dict) and "checkpoint" in init_cfg:
            return init_cfg["checkpoint"]
        for value in node.values():
            found = _find_checkpoint(value)
            if found:
                return found
    elif isinstance(node, (list, tuple)):
        for value in node:
            found = _find_checkpoint(value)
            if found:
                return found
    return None


def get_weight_info(config_path: str) -> tuple[str, str]:
    """Return (weights_source, weights_detail) from a mmdet config file.

    weights_source is one of:
      - "load_from"  — top-level load_from key
      - "init_cfg"   — checkpoint found inside model.init_cfg
      - "none"       — no weights configured

    weights_detail is the path/URL string, or "-" when weights_source is "none".
    """
    cfg = Config.fromfile(config_path)

    load_from = cfg.get("load_from")
    if load_from:
        return ("load_from", str(load_from))

    checkpoint = _find_checkpoint(cfg.get("model", {}))
    if checkpoint:
        return ("init_cfg", str(checkpoint))

    return ("none", "-")
