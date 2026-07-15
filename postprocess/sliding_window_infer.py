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


def _overlap_pixels(patch_size: int, patch_overlap_ratio: float) -> int:
    overlap = int(patch_size * patch_overlap_ratio)
    return max(0, min(overlap, int(patch_size) - 1))


def _edge_suppression_margin(patch_size: int, patch_overlap_ratio: float) -> int:
    if os.getenv("BL_SLIDING_DISABLE_EDGE_SUPPRESSION", "0") == "1":
        return 0

    explicit_margin = os.getenv("BL_SLIDING_EDGE_SUPPRESS_MARGIN_PX")
    if explicit_margin is not None:
        try:
            margin = int(float(explicit_margin))
        except ValueError:
            margin = 0
    else:
        overlap = _overlap_pixels(patch_size, patch_overlap_ratio)
        margin_fraction = _env_float("BL_SLIDING_EDGE_MARGIN_FRACTION", 0.5)
        margin = int(overlap * margin_fraction)

    max_margin = max(0, int(patch_size) // 2 - 1)
    return max(0, min(margin, max_margin))


def _drop_edge_touching_instances_enabled() -> bool:
    return os.getenv("BL_SLIDING_DROP_EDGE_TOUCHING_INSTANCES", "1") != "0"


def _touches_internal_patch_edge(
    ys: np.ndarray,
    xs: np.ndarray,
    *,
    actual_height: int,
    actual_width: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
    image_height: int,
    image_width: int,
) -> bool:
    if ys.size == 0 or xs.size == 0:
        return False

    tol = int(_env_float("BL_SLIDING_EDGE_TOUCH_TOLERANCE_PX", 2.0))
    tol = max(0, tol)

    if top > 0 and int(ys.min()) <= tol:
        return True
    if left > 0 and int(xs.min()) <= tol:
        return True
    if bottom < image_height and int(ys.max()) >= max(0, actual_height - 1 - tol):
        return True
    if right < image_width and int(xs.max()) >= max(0, actual_width - 1 - tol):
        return True
    return False


def _iter_windows(
    image_height: int,
    image_width: int,
    patch_size: int,
    patch_overlap_ratio: float,
):
    overlap = int(patch_size * patch_overlap_ratio)
    if overlap >= patch_size:
        raise ValueError(
            "patch_overlap_ratio 对应的重叠像素不能大于等于 patch_size"
        )
    step = max(patch_size - overlap, 1)

    for top in range(0, image_height, step):
        for left in range(0, image_width, step):
            bottom = min(top + patch_size, image_height)
            right = min(left + patch_size, image_width)
            yield left, top, right, bottom


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


def _extract_patch_instances(
    pred_instances,
    *,
    left: int,
    top: int,
    right: int,
    bottom: int,
    image_height: int,
    image_width: int,
    score_thresh: float,
    target_label: int,
    min_pixel_count: int,
    next_instance_id: int,
    edge_margin: int,
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

    instances: list[dict] = []
    for index, mask in enumerate(masks_np):
        score = float(scores_np[index]) if scores_np is not None else 1.0
        if score < score_thresh:
            continue

        label = int(labels_np[index]) if labels_np is not None else target_label
        if label != target_label:
            continue

        mask_height, mask_width = mask.shape[:2]
        actual_height = max(0, min(int(bottom - top), int(mask_height)))
        actual_width = max(0, min(int(right - left), int(mask_width)))
        valid_top = 0
        valid_left = 0
        valid_bottom = actual_height
        valid_right = actual_width

        if edge_margin > 0:
            if top > 0:
                valid_top = min(edge_margin, valid_bottom)
            if left > 0:
                valid_left = min(edge_margin, valid_right)
            if bottom < image_height:
                valid_bottom = max(valid_top, valid_bottom - edge_margin)
            if right < image_width:
                valid_right = max(valid_left, valid_right - edge_margin)

        if valid_bottom <= valid_top or valid_right <= valid_left:
            continue

        ys, xs = np.where(mask)
        if (
            edge_margin > 0
            and _drop_edge_touching_instances_enabled()
            and _touches_internal_patch_edge(
                ys,
                xs,
                actual_height=actual_height,
                actual_width=actual_width,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                image_height=image_height,
                image_width=image_width,
            )
        ):
            continue

        in_valid_region = (
            (ys >= valid_top)
            & (ys < valid_bottom)
            & (xs >= valid_left)
            & (xs < valid_right)
        )
        ys = ys[in_valid_region]
        xs = xs[in_valid_region]
        if ys.size < min_pixel_count:
            continue

        ys_global = ys + top
        xs_global = xs + left
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
    """Merge overlapping instances when the overlap is substantial.

    Two *different* domains never overlap spatially — they are separated by
    domain walls.  Therefore two instances that share a large fraction of
    pixels must be the same domain predicted from different windows (or
    double-detected).  A tiny overlap (boundary fuzz between adjacent
    domains) is handled by winner-take-all instead of merging.
    """
    if not instances:
        return []

    size = int(image_height) * int(image_width)
    n = len(instances)

    # ── 1. build pixel→id map; record conflict pairs + overlap size ────
    id_map = np.zeros(size, dtype=np.int32)
    # conflict_pairs: (id_a, id_b) → set of flat pixel coords where they overlap
    conflict_pixels: dict[tuple[int, int], set] = {}

    order = sorted(range(n), key=lambda i: instances[i].get("score", 0.0))
    for idx in order:
        inst = instances[idx]
        coords = inst.get("coords")
        if coords is None:
            continue
        flat = np.asarray(coords, dtype=np.int64)
        valid = (flat >= 0) & (flat < size)
        if not np.any(valid):
            continue
        flat_valid = flat[valid]
        existing = id_map[flat_valid]
        conflict_ids = np.unique(existing[existing > 0])
        for eid in conflict_ids:
            eid = int(eid)
            if eid != inst["id"]:
                key = (min(inst["id"], eid), max(inst["id"], eid))
                # record pixels where this pair conflicts
                mask = np.isin(flat_valid, flat_valid[existing == eid])
                if key not in conflict_pixels:
                    conflict_pixels[key] = set()
                conflict_pixels[key].update(flat_valid[mask].tolist())
        id_map[flat_valid] = int(inst["id"])

    # ── 2. decide merge vs keep-separate based on overlap ratio ─────────
    # Only merge when overlap / min_size > threshold (same domain).
    overlap_ratio_threshold = float(
        os.getenv("BL_SLIDING_MERGE_OVERLAP_RATIO", "0.25")
    )
    id_to_idx: dict[int, int] = {inst["id"]: i for i, inst in enumerate(instances)}
    instance_sizes = {
        inst["id"]: len(np.asarray(inst.get("coords"), dtype=np.int64))
        for inst in instances
    }

    parent = list(range(n))

    def _find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for (id_a, id_b), pixels in conflict_pixels.items():
        if id_a not in id_to_idx or id_b not in id_to_idx:
            continue
        overlap_sz = len(pixels)
        min_sz = min(instance_sizes.get(id_a, 1), instance_sizes.get(id_b, 1))
        if min_sz <= 0:
            continue
        ratio = overlap_sz / float(min_sz)
        if ratio >= overlap_ratio_threshold:
            _union(id_to_idx[id_a], id_to_idx[id_b])

    # ── 3. group by root ───────────────────────────────────────────────
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = _find(i)
        groups.setdefault(root, []).append(i)

    # ── 4. materialise ─────────────────────────────────────────────────
    resolved: list[dict] = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            inst = instances[group_indices[0]]
            flat = np.asarray(inst.get("coords"), dtype=np.int64)
            valid = (flat >= 0) & (flat < size)
            if not np.any(valid):
                continue
            flat_valid = flat[valid]
            keep_mask = id_map[flat_valid] == inst["id"]
            if not np.any(keep_mask):
                continue
            kept_flat = flat_valid[keep_mask].astype(np.int32, copy=False)
            ys_kept = (kept_flat // image_width).astype(np.int64, copy=False)
            xs_kept = (kept_flat % image_width).astype(np.int64, copy=False)
            new_inst = dict(inst)
            new_inst["coords"] = kept_flat
            new_inst["bbox"] = [
                int(xs_kept.min()), int(ys_kept.min()),
                int(xs_kept.max()), int(ys_kept.max()),
            ]
            resolved.append(new_inst)
        else:
            group_insts = [instances[i] for i in group_indices]
            ordered = sorted(group_insts, key=lambda x: x.get("score", 0.0))
            keep_inst = dict(ordered[-1])
            keep_id = int(keep_inst["id"])
            keep_coords = np.asarray(keep_inst.get("coords"), dtype=np.int32)

            for other in ordered[:-1]:
                other_id = int(other.get("id", 0))
                other_coords = np.asarray(other.get("coords"), dtype=np.int32)
                ol = (
                    np.intersect1d(other_coords, keep_coords)
                    if other_coords.size > 0 and keep_coords.size > 0
                    else np.array([], dtype=np.int32)
                )
                ro = (
                    np.setdiff1d(other_coords, keep_coords)
                    if other_coords.size > 0
                    else np.array([], dtype=np.int32)
                )
                merge_records.append({
                    "kept": keep_id,
                    "removed": other_id,
                    "overlap": ol,
                    "removed_only": ro,
                    "removed_coords": other_coords.copy(),
                })
                keep_coords = np.union1d(keep_coords, other_coords).astype(
                    np.int32, copy=False
                )

            ys = (keep_coords // image_width).astype(np.int64, copy=False)
            xs = (keep_coords % image_width).astype(np.int64, copy=False)
            keep_inst["coords"] = keep_coords
            keep_inst["bbox"] = [
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            ]
            resolved.append(keep_inst)

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
        raise ValueError(f"patch_size 必须大于 0，实际为 {patch_size}")
    if not (0.0 <= patch_overlap_ratio < 1.0):
        raise ValueError(
            f"patch_overlap_ratio 必须满足 0 <= ratio < 1，实际为 {patch_overlap_ratio}"
        )

    pil_img = Image.open(img_path).convert("RGB")
    image_np = np.asarray(pil_img)
    image_height, image_width = image_np.shape[:2]
    edge_margin = _edge_suppression_margin(patch_size, patch_overlap_ratio)

    flat_instances: list[dict] = []
    merge_records: list[dict] = []
    windows: list[dict] = []
    next_instance_id = 1
    pending_windows: list[tuple[int, int, int, int]] = []
    pending_patches: list[np.ndarray] = []

    def flush_batch() -> None:
        nonlocal next_instance_id
        if not pending_windows:
            return

        batch_result = inference_detector(model, pending_patches)
        if not isinstance(batch_result, list):
            batch_result = [batch_result]

        for result, (left, top, right, bottom) in zip(batch_result, pending_windows):
            patch_instances, next_instance_id = _extract_patch_instances(
                result.pred_instances,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                image_height=image_height,
                image_width=image_width,
                score_thresh=score_thresh,
                target_label=target_label,
                min_pixel_count=min_pixel_count,
                next_instance_id=next_instance_id,
                edge_margin=edge_margin,
            )
            flat_instances.extend(patch_instances)

        pending_windows.clear()
        pending_patches.clear()

    for left, top, right, bottom in _iter_windows(
        image_height,
        image_width,
        patch_size,
        patch_overlap_ratio,
    ):
        windows.append(
            {
                "idx": len(windows),
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "edge_margin": int(edge_margin),
            }
        )
        pending_windows.append((left, top, right, bottom))
        pending_patches.append(
            _prepare_patch_image(
                image_np,
                left,
                top,
                right,
                bottom,
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
