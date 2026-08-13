from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from ._shared import _to_numpy


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _overlap_pixels(patch_size: int, patch_overlap_ratio: float) -> int:
    overlap = int(patch_size * patch_overlap_ratio)
    return max(0, min(overlap, patch_size - 1))


def _context_margin_pixels(patch_size: int) -> int:
    margin_px = _env_int("BL_SLIDING_CONTEXT_MARGIN_PX", None)
    if margin_px is None:
        margin_px = int(
            round(
                patch_size
                * _env_float("BL_SLIDING_CONTEXT_MARGIN_RATIO", 0.25)
            )
        )
    max_margin = max((patch_size - 1) // 2, 0)
    return max(0, min(int(margin_px), max_margin))


def _effective_overlap_pixels(
    patch_size: int,
    patch_overlap_ratio: float,
    context_margin: int,
) -> tuple[int, int]:
    requested = _overlap_pixels(patch_size, patch_overlap_ratio)
    # Context metadata must not override the user-requested window overlap.
    return requested, requested


def _edge_touch_margin_pixels(patch_size: int, context_margin: int) -> int:
    margin_px = _env_int("BL_SLIDING_EDGE_TOUCH_MARGIN_PX", None)
    if margin_px is None:
        margin_px = int(
            round(
                patch_size
                * _env_float("BL_SLIDING_EDGE_TOUCH_MARGIN_RATIO", 0.02)
            )
        )
    return max(0, min(int(margin_px), max(int(context_margin), 0)))


def _axis_starts(length: int, patch_size: int, overlap: int) -> list[int]:
    """Return endpoint-aligned starts without creating narrow edge crops."""
    if length <= patch_size:
        return [0]

    step = max(patch_size - overlap, 1)
    span = int(length) - int(patch_size)
    intervals = max(1, int(np.ceil(span / float(step))))
    starts = np.rint(np.linspace(0, span, intervals + 1)).astype(np.int64)

    deduped: list[int] = []
    for start in starts.tolist():
        start_i = int(max(0, min(span, start)))
        if not deduped or start_i != deduped[-1]:
            deduped.append(start_i)
    if deduped[-1] != span:
        deduped.append(span)
    return deduped


def _axis_core_bounds(
    starts: list[int],
    index: int,
    patch_size: int,
    image_length: int,
) -> tuple[float, float]:
    if len(starts) == 1:
        return 0.0, float(image_length)

    start = starts[index]
    if index == 0:
        core_start = 0.0
    else:
        prev_start = starts[index - 1]
        core_start = 0.5 * (float(prev_start) + float(start) + float(patch_size))

    if index == len(starts) - 1:
        core_end = float(image_length)
    else:
        next_start = starts[index + 1]
        core_end = 0.5 * (float(start) + float(next_start) + float(patch_size))

    return core_start, core_end


def _axis_safe_bounds(
    start: int,
    end: int,
    image_length: int,
    context_margin: int,
) -> tuple[float, float]:
    safe_start = float(start)
    safe_end = float(end)
    if start > 0:
        safe_start = min(safe_end, safe_start + float(context_margin))
    if end < image_length:
        safe_end = max(safe_start, safe_end - float(context_margin))
    return safe_start, safe_end


def _iter_windows(
    image_height: int,
    image_width: int,
    patch_size: int,
    patch_overlap_ratio: float,
):
    context_margin = _context_margin_pixels(patch_size)
    edge_touch_margin = _edge_touch_margin_pixels(patch_size, context_margin)
    requested_overlap, overlap = _effective_overlap_pixels(
        patch_size,
        patch_overlap_ratio,
        context_margin,
    )
    if overlap >= patch_size:
        raise ValueError(
            "patch_overlap_ratio produces overlap >= patch_size"
        )

    y_starts = _axis_starts(image_height, patch_size, overlap)
    x_starts = _axis_starts(image_width, patch_size, overlap)
    idx = 0
    for gy, top in enumerate(y_starts):
        core_top, core_bottom = _axis_core_bounds(
            y_starts, gy, patch_size, image_height
        )
        for gx, left in enumerate(x_starts):
            core_left, core_right = _axis_core_bounds(
                x_starts, gx, patch_size, image_width
            )
            bottom = min(top + patch_size, image_height)
            right = min(left + patch_size, image_width)
            safe_top, safe_bottom = _axis_safe_bounds(
                int(top),
                int(bottom),
                image_height,
                context_margin,
            )
            safe_left, safe_right = _axis_safe_bounds(
                int(left),
                int(right),
                image_width,
                context_margin,
            )
            yield {
                "idx": idx,
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "core_left": float(core_left),
                "core_top": float(core_top),
                "core_right": float(core_right),
                "core_bottom": float(core_bottom),
                "safe_left": float(safe_left),
                "safe_top": float(safe_top),
                "safe_right": float(safe_right),
                "safe_bottom": float(safe_bottom),
                "grid_x": int(gx),
                "grid_y": int(gy),
                "overlap": int(overlap),
                "requested_overlap": int(requested_overlap),
                "context_margin": int(context_margin),
                "edge_touch_margin": int(edge_touch_margin),
            }
            idx += 1


def _prepare_patch_image(
    image_np: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    patch_size: int,
) -> np.ndarray:
    patch = image_np[top:bottom, left:right, :]
    patch_height, patch_width = patch.shape[:2]
    if patch_height == patch_size and patch_width == patch_size:
        return patch

    padded = np.zeros((patch_size, patch_size, 3), dtype=image_np.dtype)
    padded[:patch_height, :patch_width, :] = patch
    return padded


def _mask_center(
    ys_global: np.ndarray,
    xs_global: np.ndarray,
) -> tuple[float, float]:
    mode = os.getenv("BL_SLIDING_ASSIGN_CENTER", "bbox")
    if mode == "centroid":
        return float(ys_global.mean()), float(xs_global.mean())

    cy = 0.5 * (float(ys_global.min()) + float(ys_global.max()))
    cx = 0.5 * (float(xs_global.min()) + float(xs_global.max()))
    return cy, cx


def _center_in_core(
    center_y: float,
    center_x: float,
    window: dict[str, Any],
) -> bool:
    return (
        float(window["core_top"]) <= center_y < float(window["core_bottom"])
        and float(window["core_left"]) <= center_x < float(window["core_right"])
    )


def _mask_touches_crop_edge(
    ys_global: np.ndarray,
    xs_global: np.ndarray,
    window: dict[str, Any],
    image_height: int,
    image_width: int,
) -> bool:
    margin = int(window.get("edge_touch_margin", 0))
    if margin <= 0 or ys_global.size == 0:
        return False

    left = int(window["left"])
    top = int(window["top"])
    right = int(window["right"])
    bottom = int(window["bottom"])

    min_y = int(ys_global.min())
    max_y = int(ys_global.max())
    min_x = int(xs_global.min())
    max_x = int(xs_global.max())
    return (
        (left > 0 and min_x < left + margin)
        or (right < image_width and max_x >= right - margin)
        or (top > 0 and min_y < top + margin)
        or (bottom < image_height and max_y >= bottom - margin)
    )


def _extract_patch_instances(
    pred_instances,
    *,
    window: dict[str, Any],
    image_height: int,
    image_width: int,
    score_thresh: float,
    target_label: int,
    min_pixel_count: int,
    next_instance_id: int,
) -> tuple[list[dict], int]:
    masks = getattr(pred_instances, "masks", None)
    if masks is None:
        return [], next_instance_id

    if hasattr(masks, "to_ndarray"):
        masks_np = masks.to_ndarray().astype(bool)
    else:
        masks_np = np.asarray(_to_numpy(masks), dtype=bool)
    if masks_np.ndim == 2:
        masks_np = masks_np[np.newaxis, ...]

    scores_np = _to_numpy(getattr(pred_instances, "scores", None))
    labels_np = _to_numpy(getattr(pred_instances, "labels", None))

    left = int(window["left"])
    top = int(window["top"])
    right = int(window["right"])
    bottom = int(window["bottom"])
    actual_height = max(0, min(bottom - top, masks_np.shape[-2]))
    actual_width = max(0, min(right - left, masks_np.shape[-1]))

    instances: list[dict] = []
    for index, mask in enumerate(masks_np):
        score = float(scores_np[index]) if scores_np is not None else 1.0
        if score < score_thresh:
            continue

        label = int(labels_np[index]) if labels_np is not None else target_label
        if label != target_label:
            continue

        ys, xs = np.where(mask)
        if ys.size == 0:
            continue

        inside_image_crop = (
            (ys >= 0)
            & (ys < actual_height)
            & (xs >= 0)
            & (xs < actual_width)
        )
        ys = ys[inside_image_crop]
        xs = xs[inside_image_crop]
        if ys.size < min_pixel_count:
            continue

        ys_global = ys.astype(np.int64, copy=False) + top
        xs_global = xs.astype(np.int64, copy=False) + left
        valid = (
            (ys_global >= 0)
            & (ys_global < image_height)
            & (xs_global >= 0)
            & (xs_global < image_width)
        )
        if int(valid.sum()) < min_pixel_count:
            continue

        ys_valid = ys_global[valid].astype(np.int64, copy=False)
        xs_valid = xs_global[valid].astype(np.int64, copy=False)
        center_y, center_x = _mask_center(ys_valid, xs_valid)
        if not _center_in_core(center_y, center_x, window):
            continue
        if _mask_touches_crop_edge(
            ys_valid,
            xs_valid,
            window,
            image_height,
            image_width,
        ):
            continue

        flat_coords = (
            ys_valid * int(image_width) + xs_valid
        ).astype(np.int32, copy=False)
        bbox = [
            int(xs_valid.min()),
            int(ys_valid.min()),
            int(xs_valid.max()),
            int(ys_valid.max()),
        ]
        instances.append(
            {
                "id": next_instance_id,
                "coords": flat_coords,
                "bbox": bbox,
                "score": score,
                "window_idx": int(window["idx"]),
            }
        )
        next_instance_id += 1

    return instances, next_instance_id


def _resolve_overlaps(
    instances: list[dict],
    image_height: int,
    image_width: int,
    merge_records: list[dict],
) -> list[dict]:
    """Merge true duplicate detections, then assign residual overlaps by score."""
    if not instances:
        return []

    size = int(image_height) * int(image_width)
    n = len(instances)
    id_map = np.zeros(size, dtype=np.int32)
    conflict_pixels: dict[tuple[int, int], set[int]] = {}

    order = sorted(range(n), key=lambda i: instances[i].get("score", 0.0))
    for idx in order:
        inst = instances[idx]
        flat = np.asarray(inst.get("coords"), dtype=np.int64)
        valid = (flat >= 0) & (flat < size)
        if not np.any(valid):
            continue

        flat_valid = flat[valid]
        existing = id_map[flat_valid]
        for existing_id in np.unique(existing[existing > 0]):
            existing_id = int(existing_id)
            if existing_id == int(inst["id"]):
                continue
            key = (min(int(inst["id"]), existing_id), max(int(inst["id"]), existing_id))
            overlap_pixels = flat_valid[existing == existing_id]
            conflict_pixels.setdefault(key, set()).update(overlap_pixels.tolist())
        id_map[flat_valid] = int(inst["id"])

    overlap_ratio_threshold = _env_float("BL_SLIDING_MERGE_OVERLAP_RATIO", 0.25)
    id_to_idx: dict[int, int] = {
        int(inst["id"]): i for i, inst in enumerate(instances)
    }
    instance_sizes = {
        int(inst["id"]): int(np.asarray(inst.get("coords"), dtype=np.int64).size)
        for inst in instances
    }

    parent = list(range(n))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for (id_a, id_b), pixels in conflict_pixels.items():
        if id_a not in id_to_idx or id_b not in id_to_idx:
            continue
        min_size = min(instance_sizes.get(id_a, 0), instance_sizes.get(id_b, 0))
        if min_size <= 0:
            continue
        if len(pixels) / float(min_size) >= overlap_ratio_threshold:
            union(id_to_idx[id_a], id_to_idx[id_b])

    groups: dict[int, list[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    merged: list[dict] = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            merged.append(instances[group_indices[0]])
            continue

        group_insts = [instances[i] for i in group_indices]
        ordered = sorted(group_insts, key=lambda x: x.get("score", 0.0))
        keep_inst = dict(ordered[-1])
        keep_id = int(keep_inst["id"])
        keep_coords = np.asarray(keep_inst.get("coords"), dtype=np.int32)

        for other in ordered[:-1]:
            other_id = int(other.get("id", 0))
            other_coords = np.asarray(other.get("coords"), dtype=np.int32)
            overlap = (
                np.intersect1d(other_coords, keep_coords)
                if other_coords.size and keep_coords.size
                else np.array([], dtype=np.int32)
            )
            removed_only = (
                np.setdiff1d(other_coords, keep_coords)
                if other_coords.size
                else np.array([], dtype=np.int32)
            )
            merge_records.append(
                {
                    "kept": keep_id,
                    "removed": other_id,
                    "overlap": overlap,
                    "removed_only": removed_only,
                    "removed_coords": other_coords.copy(),
                }
            )
            keep_coords = np.union1d(keep_coords, other_coords).astype(
                np.int32, copy=False
            )

        keep_inst["coords"] = keep_coords
        keep_inst["bbox"] = _bbox_from_flat(keep_coords, image_width)
        merged.append(keep_inst)

    return _assign_visible_pixels_by_score(merged, image_height, image_width)


def _bbox_from_flat(flat_coords: np.ndarray, image_width: int) -> list[int]:
    ys = (flat_coords // image_width).astype(np.int64, copy=False)
    xs = (flat_coords % image_width).astype(np.int64, copy=False)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _assign_visible_pixels_by_score(
    instances: list[dict],
    image_height: int,
    image_width: int,
) -> list[dict]:
    if not instances:
        return []

    size = int(image_height) * int(image_width)
    id_map = np.zeros(size, dtype=np.int32)
    for inst in sorted(instances, key=lambda x: x.get("score", 0.0)):
        flat = np.asarray(inst.get("coords"), dtype=np.int64)
        valid = (flat >= 0) & (flat < size)
        if np.any(valid):
            id_map[flat[valid]] = int(inst["id"])

    resolved: list[dict] = []
    for inst in instances:
        inst_id = int(inst.get("id", 0))
        flat = np.asarray(inst.get("coords"), dtype=np.int64)
        valid = (flat >= 0) & (flat < size)
        if inst_id == 0 or not np.any(valid):
            continue

        flat_valid = flat[valid]
        keep_mask = id_map[flat_valid] == inst_id
        if not np.any(keep_mask):
            continue

        kept_flat = flat_valid[keep_mask].astype(np.int32, copy=False)
        new_inst = dict(inst)
        new_inst["coords"] = kept_flat
        new_inst["bbox"] = _bbox_from_flat(kept_flat, image_width)
        resolved.append(new_inst)
    return resolved


def _filter_small_instances(
    instances: list[dict],
    min_pixel_count: int,
    merge_records: list[dict],
) -> list[dict]:
    filtered: list[dict] = []
    for inst in instances:
        coords = np.asarray(inst.get("coords"), dtype=np.int32)
        if coords.size >= min_pixel_count:
            filtered.append(inst)
            continue
        merge_records.append(
            {
                "kept": None,
                "removed": int(inst.get("id", 0)) or None,
                "overlap": np.array([], dtype=np.int32),
                "removed_only": coords.copy(),
                "removed_coords": coords.copy(),
            }
        )
    return filtered


def _flat_instances_to_global_instances(
    instances: list[dict],
    image_width: int,
) -> list[dict]:
    global_instances: list[dict] = []
    for inst in instances:
        flat_coords = np.asarray(inst.get("coords"), dtype=np.int64)
        if flat_coords.size == 0:
            continue
        ys = (flat_coords // image_width).astype(np.int32, copy=False)
        xs = (flat_coords % image_width).astype(np.int32, copy=False)
        coords = np.stack([ys, xs], axis=1)
        global_instances.append(
            {
                "id": int(inst["id"]),
                "coords": coords,
                "bbox": list(inst["bbox"]),
                "score": float(inst.get("score", 0.0)),
            }
        )
    return global_instances


def infer_image_with_overlap_windows(
    model,
    img_path: str,
    *,
    score_thresh: float = 0.5,
    target_label: int = 0,
    min_pixel_count: int = 10,
    patch_size: int = 1024,
    patch_overlap_ratio: float = 0.0,
    batch_size: int = 1,
) -> tuple[list[dict], Image.Image, list[dict], list[dict]]:
    from mmdet.apis import inference_detector  # type: ignore

    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if not (0.0 <= patch_overlap_ratio < 1.0):
        raise ValueError(
            f"patch_overlap_ratio must satisfy 0 <= ratio < 1, got {patch_overlap_ratio}"
        )

    pil_img = Image.open(img_path).convert("RGB")
    image_np = np.asarray(pil_img)
    image_height, image_width = image_np.shape[:2]

    flat_instances: list[dict] = []
    merge_records: list[dict] = []
    windows: list[dict] = []
    next_instance_id = 1
    pending_windows: list[dict[str, Any]] = []
    pending_patches: list[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal next_instance_id
        if not pending_windows:
            return

        batch_result = inference_detector(model, pending_patches)
        if not isinstance(batch_result, list):
            batch_result = [batch_result]

        for result, window in zip(batch_result, pending_windows):
            patch_instances, next_instance_id = _extract_patch_instances(
                result.pred_instances,
                window=window,
                image_height=image_height,
                image_width=image_width,
                score_thresh=score_thresh,
                target_label=target_label,
                min_pixel_count=min_pixel_count,
                next_instance_id=next_instance_id,
            )
            flat_instances.extend(patch_instances)

        pending_windows.clear()
        pending_patches.clear()

    for window in _iter_windows(
        image_height,
        image_width,
        patch_size,
        patch_overlap_ratio,
    ):
        windows.append(
            {
                "idx": int(window["idx"]),
                "left": int(window["left"]),
                "top": int(window["top"]),
                "right": int(window["right"]),
                "bottom": int(window["bottom"]),
                "core_left": float(window["core_left"]),
                "core_top": float(window["core_top"]),
                "core_right": float(window["core_right"]),
                "core_bottom": float(window["core_bottom"]),
                "safe_left": float(window["safe_left"]),
                "safe_top": float(window["safe_top"]),
                "safe_right": float(window["safe_right"]),
                "safe_bottom": float(window["safe_bottom"]),
                "grid_x": int(window["grid_x"]),
                "grid_y": int(window["grid_y"]),
                "overlap": int(window["overlap"]),
                "requested_overlap": int(window["requested_overlap"]),
                "context_margin": int(window["context_margin"]),
                "edge_touch_margin": int(window["edge_touch_margin"]),
            }
        )
        pending_windows.append(window)
        pending_patches.append(
            _prepare_patch_image(
                image_np,
                int(window["left"]),
                int(window["top"]),
                int(window["right"]),
                int(window["bottom"]),
                patch_size,
            )
        )
        if len(pending_windows) >= max(int(batch_size), 1):
            flush_batch()

    flush_batch()

    flat_instances = _resolve_overlaps(
        flat_instances,
        image_height,
        image_width,
        merge_records,
    )
    flat_instances = _filter_small_instances(
        flat_instances,
        min_pixel_count,
        merge_records,
    )
    return (
        _flat_instances_to_global_instances(flat_instances, image_width),
        pil_img,
        windows,
        merge_records,
    )
