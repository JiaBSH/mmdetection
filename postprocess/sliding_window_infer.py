from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def _to_numpy(data: Any) -> np.ndarray | None:
    if data is None:
        return None
    if hasattr(data, "cpu"):
        data = data.cpu()
    if hasattr(data, "numpy"):
        return data.numpy()
    return np.asarray(data)


def _bbox_overlap(box1: list[int], box2: list[int]) -> bool:
    return not (
        box1[2] < box2[0]
        or box1[0] > box2[2]
        or box1[3] < box2[1]
        or box1[1] > box2[3]
    )


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

    instances: list[dict] = []
    for index, mask in enumerate(masks_np):
        score = float(scores_np[index]) if scores_np is not None else 1.0
        if score < score_thresh:
            continue

        label = int(labels_np[index]) if labels_np is not None else target_label
        if label != target_label:
            continue

        ys, xs = np.where(mask)
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
    if not instances:
        return []

    sorted_instances = sorted(instances, key=lambda inst: inst.get("score", 0.0))
    size = int(image_height) * int(image_width)
    id_map_flat = np.zeros(size, dtype=np.int32)

    for inst in sorted_instances:
        coords = inst.get("coords")
        if coords is None:
            continue
        flat = np.asarray(coords, dtype=np.int64)
        valid = (flat >= 0) & (flat < size)
        if np.any(valid):
            id_map_flat[flat[valid]] = int(inst["id"])

    resolved: list[dict] = []
    for inst in instances:
        inst_id = int(inst.get("id", 0))
        if inst_id == 0:
            continue

        flat = np.asarray(inst.get("coords"), dtype=np.int64)
        valid = (flat >= 0) & (flat < size)
        if not np.any(valid):
            continue

        flat_valid = flat[valid]
        keep = id_map_flat[flat_valid] == inst_id
        if not np.any(keep):
            owner_ids = id_map_flat[flat_valid]
            owner_ids = owner_ids[owner_ids > 0]
            kept_id = int(owner_ids[0]) if owner_ids.size > 0 else None
            merge_records.append(
                {
                    "kept": kept_id,
                    "removed": inst_id,
                    "overlap": None,
                    "removed_only": flat_valid.astype(np.int32, copy=False),
                    "removed_coords": flat_valid.astype(np.int32, copy=False),
                }
            )
            continue

        kept_flat = flat_valid[keep].astype(np.int32, copy=False)
        removed_flat = flat_valid[~keep].astype(np.int32, copy=False)
        if removed_flat.size > 0:
            owner_ids = id_map_flat[removed_flat]
            owner_ids = owner_ids[owner_ids > 0]
            kept_id = int(owner_ids[0]) if owner_ids.size > 0 else None
            merge_records.append(
                {
                    "kept": kept_id,
                    "removed": inst_id,
                    "overlap": kept_flat.copy(),
                    "removed_only": removed_flat.copy(),
                    "removed_coords": flat_valid.astype(np.int32, copy=False),
                }
            )
        ys_kept = (kept_flat // image_width).astype(np.int64, copy=False)
        xs_kept = (kept_flat % image_width).astype(np.int64, copy=False)
        new_inst = dict(inst)
        new_inst["coords"] = kept_flat
        new_inst["bbox"] = [
            int(xs_kept.min()),
            int(ys_kept.min()),
            int(xs_kept.max()),
            int(ys_kept.max()),
        ]
        resolved.append(new_inst)

    return resolved


def _centroid_from_flat(coords_flat: np.ndarray, image_width: int) -> tuple[float, float]:
    ys = (coords_flat // image_width).astype(np.float64, copy=False)
    xs = (coords_flat % image_width).astype(np.float64, copy=False)
    return float(ys.mean()), float(xs.mean())


def _merge_close_fragments(
    instances: list[dict],
    image_width: int,
    merge_distance: int,
    merge_records: list[dict],
) -> list[dict]:
    if len(instances) <= 1:
        return instances

    instance_count = len(instances)
    centroids = np.zeros((instance_count, 2), dtype=np.float64)
    bboxes = np.zeros((instance_count, 4), dtype=np.int64)
    for index, inst in enumerate(instances):
        coords = np.asarray(inst.get("coords"), dtype=np.int32)
        if coords.size == 0:
            continue
        centroids[index] = _centroid_from_flat(coords, image_width)
        bboxes[index] = np.asarray(inst.get("bbox", [0, 0, 0, 0]), dtype=np.int64)

    cell_size = max(int(merge_distance), 1)
    grid: dict[tuple[int, int], list[int]] = {}
    for index in range(instance_count):
        grid_y = int(centroids[index, 0] // cell_size)
        grid_x = int(centroids[index, 1] // cell_size)
        grid.setdefault((grid_y, grid_x), []).append(index)

    parent = list(range(instance_count))
    rank = [0] * instance_count

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
            return
        if rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
            return
        parent[root_right] = root_left
        rank[root_left] += 1

    threshold_sq = float(merge_distance * merge_distance)
    for index in range(instance_count):
        grid_y = int(centroids[index, 0] // cell_size)
        grid_x = int(centroids[index, 1] // cell_size)
        for delta_y in (-1, 0, 1):
            for delta_x in (-1, 0, 1):
                for other in grid.get((grid_y + delta_y, grid_x + delta_x), []):
                    if other <= index:
                        continue
                    diff_y = centroids[index, 0] - centroids[other, 0]
                    diff_x = centroids[index, 1] - centroids[other, 1]
                    if diff_y * diff_y + diff_x * diff_x > threshold_sq:
                        continue

                    bbox_a = bboxes[index]
                    bbox_b = bboxes[other]
                    expanded_box = [
                        int(bbox_a[0] - merge_distance),
                        int(bbox_a[1] - merge_distance),
                        int(bbox_a[2] + merge_distance),
                        int(bbox_a[3] + merge_distance),
                    ]
                    if not _bbox_overlap(
                        expanded_box,
                        [
                            int(bbox_b[0]),
                            int(bbox_b[1]),
                            int(bbox_b[2]),
                            int(bbox_b[3]),
                        ],
                    ):
                        continue
                    union(index, other)

    components: dict[int, list[int]] = {}
    for index in range(instance_count):
        root = find(index)
        components.setdefault(root, []).append(index)

    if len(components) == instance_count:
        return instances

    merged_instances: list[dict] = []
    for member_indices in components.values():
        if len(member_indices) == 1:
            merged_instances.append(instances[member_indices[0]])
            continue

        ordered_members = sorted(
            member_indices,
            key=lambda idx: int(instances[idx].get("id", 0)),
        )
        keep_inst = dict(instances[ordered_members[0]])
        keep_coords = np.asarray(keep_inst.get("coords"), dtype=np.int32)
        for member_index in ordered_members[1:]:
            member_coords = np.asarray(
                instances[member_index].get("coords"),
                dtype=np.int32,
            )
            overlap = (
                np.intersect1d(member_coords, keep_coords)
                if member_coords.size > 0 and keep_coords.size > 0
                else np.array([], dtype=np.int32)
            )
            removed_only = (
                np.setdiff1d(member_coords, keep_coords)
                if member_coords.size > 0
                else np.array([], dtype=np.int32)
            )
            merge_records.append(
                {
                    "kept": int(keep_inst.get("id", 0)) or None,
                    "removed": int(instances[member_index].get("id", 0)) or None,
                    "overlap": overlap,
                    "removed_only": removed_only,
                    "removed_coords": member_coords.copy(),
                }
            )
            keep_coords = np.union1d(keep_coords, member_coords).astype(
                np.int32,
                copy=False,
            )

        ys = (keep_coords // image_width).astype(np.int64, copy=False)
        xs = (keep_coords % image_width).astype(np.int64, copy=False)
        keep_inst["coords"] = keep_coords
        keep_inst["bbox"] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ]
        merged_instances.append(keep_inst)

    return merged_instances


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
    merge_distance: int = 5,
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

        for result, (left, top, _, _) in zip(batch_result, pending_windows):
            patch_instances, next_instance_id = _extract_patch_instances(
                result.pred_instances,
                left=left,
                top=top,
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
    flat_instances = _merge_close_fragments(
        flat_instances,
        image_width,
        merge_distance=merge_distance,
        merge_records=merge_records,
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