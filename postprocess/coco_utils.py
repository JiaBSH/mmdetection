"""COCO格式工具函数

将COCO格式的GT标注和MMDetection推理mask输出转换为
analyze_main_dy2.py 所需的 global_instances 格式。

COCO格式约定:
  GT  : annotations[i].segmentation = [[x1,y1,x2,y2,...]]  (polygon, x=col,y=row)
  Pred: mmdet 推理结果的 pred_instances.masks (HxW bool/uint8 numpy array 或 RLE)

global_instances 格式约定（与 temp 管道一致）:
  每个实例为 dict:
    'id'    : int
    'coords': np.ndarray shape=(N,2), 每行 (row=y, col=x)
    'bbox'  : [xmin,ymin,xmax,ymax]  (pixel, int)
    'score' : float  (GT实例设为 1.0)
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# GT: COCO JSON → global_instances
# ---------------------------------------------------------------------------

def load_coco_gt_instances(
    ann_file: str,
    image_id: int | str | None = None,
    image_filename: str | None = None,
    *,
    category_ids: list[int] | None = None,
    exclude_crowd: bool = True,
) -> tuple[list[dict], int, int]:
    """从COCO JSON加载GT实例，转换为global_instances格式。

    优先用 image_id 定位图像；若为 None 则用 image_filename（不含路径/扩展名也可匹配）。

    Returns
    -------
    instances : list[dict]  global_instances 格式
    W         : int  图像宽度（像素）
    H         : int  图像高度（像素）
    """
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # ---- 找到目标图像 ----
    img_info: dict | None = None
    if image_id is not None:
        for img in coco.get("images", []):
            if str(img["id"]) == str(image_id):
                img_info = img
                break
    if img_info is None and image_filename is not None:
        stem = os.path.splitext(os.path.basename(image_filename))[0]
        for img in coco.get("images", []):
            fn = img.get("file_name", "")
            if fn == image_filename or os.path.basename(fn) == image_filename:
                img_info = img
                break
            if os.path.splitext(os.path.basename(fn))[0] == stem:
                img_info = img
                break
    if img_info is None:
        raise ValueError(
            f"Image not found in {ann_file}: image_id={image_id}, filename={image_filename}"
        )

    W: int = int(img_info["width"])
    H: int = int(img_info["height"])
    target_img_id = int(img_info["id"])

    # ---- 过滤 annotations ----
    anns = [
        a for a in coco.get("annotations", [])
        if int(a["image_id"]) == target_img_id
        and (not exclude_crowd or not a.get("iscrowd", 0))
        and (category_ids is None or int(a["category_id"]) in category_ids)
    ]

    instances: list[dict] = []
    for idx, ann in enumerate(anns, start=1):
        seg = ann.get("segmentation", [])
        if not seg:
            continue

        # COCO polygon: [[x1,y1,x2,y2,...]] 每个多边形为 flat list
        # RLE格式跳过（本项目主要使用polygon GT）
        if isinstance(seg, dict):
            # RLE — 用 pycocotools 解码（可选支持）
            try:
                from pycocotools import mask as cocomask  # type: ignore
                mask_np = cocomask.decode(seg).astype(bool)
                ys, xs = np.where(mask_np)
                if ys.size == 0:
                    continue
                coords = np.stack([ys, xs], axis=1).astype(np.int32)
            except ImportError:
                continue
        else:
            # polygon: 将所有polygon合并光栅化为mask再提取像素坐标
            # 这样与 temp 管道的 'coords' 语义一致（像素级）
            mask_img = Image.new("L", (W, H), 0)
            draw = ImageDraw.Draw(mask_img)
            for poly_flat in seg:
                if len(poly_flat) < 6:
                    continue
                xy = [(float(poly_flat[i]), float(poly_flat[i + 1]))
                      for i in range(0, len(poly_flat) - 1, 2)]
                if len(xy) < 3:
                    continue
                draw.polygon(xy, outline=1, fill=1)
            mask_np = np.array(mask_img, dtype=np.bool_)
            ys, xs = np.where(mask_np)
            if ys.size == 0:
                continue
            coords = np.stack([ys, xs], axis=1).astype(np.int32)

        bbox_coco = ann.get("bbox", None)  # [x,y,w,h]
        if bbox_coco and len(bbox_coco) == 4:
            bx, by, bw, bh = bbox_coco
            bbox = [int(bx), int(by), int(bx + bw), int(by + bh)]
        else:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        instances.append({
            "id": idx,
            "coords": coords,
            "bbox": bbox,
            "score": 1.0,
        })

    return instances, W, H


# ---------------------------------------------------------------------------
# GT: COCO polygon vertices（不光栅化，直接返回顶点ndarray列表）
# ---------------------------------------------------------------------------

def load_coco_gt_polygons(
    ann_file: str,
    image_id: int | str | None = None,
    image_filename: str | None = None,
    *,
    category_ids: list[int] | None = None,
    exclude_crowd: bool = True,
) -> tuple[list[np.ndarray], int, int]:
    """返回GT多边形顶点列表（每个元素为 (N,2) ndarray，(row=y, col=x)）。

    用于 analyze_main_dy2.py 中直接做六边形拟合的GT路径。
    """
    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_info = _find_image_info(coco, image_id, image_filename)
    W: int = int(img_info["width"])
    H: int = int(img_info["height"])
    target_img_id = int(img_info["id"])

    anns = [
        a for a in coco.get("annotations", [])
        if int(a["image_id"]) == target_img_id
        and (not exclude_crowd or not a.get("iscrowd", 0))
        and (category_ids is None or int(a["category_id"]) in category_ids)
    ]

    polygons: list[np.ndarray] = []
    for ann in anns:
        seg = ann.get("segmentation", [])
        if not seg or isinstance(seg, dict):
            continue
        for poly_flat in seg:
            if len(poly_flat) < 6:
                continue
            # COCO: [x1,y1,x2,y2,...] → (row=y, col=x)
            pts = np.array(poly_flat, dtype=np.float32).reshape(-1, 2)
            pts_rc = pts[:, ::-1].copy()  # (y, x)
            if len(pts_rc) >= 3:
                polygons.append(pts_rc)

    return polygons, W, H


# ---------------------------------------------------------------------------
# Pred: MMDetection推理mask → global_instances
# ---------------------------------------------------------------------------

def mmdet_masks_to_instances(
    masks: Any,
    scores: Any | None = None,
    labels: Any | None = None,
    bboxes: Any | None = None,
    *,
    score_thresh: float = 0.5,
    target_label: int = 0,
    image_shape: tuple[int, int] | None = None,
    iou_merge_thresh: float = 0.4,
    min_pixel_count: int = 10,
) -> list[dict]:
    """将MMDetection推理输出的masks转换为global_instances格式。

    Parameters
    ----------
    masks   : np.ndarray (N,H,W) bool/uint8 或 list of (H,W) arrays，或 RLE list
    scores  : np.ndarray (N,) float32，置信度
    labels  : np.ndarray (N,) int，类别（0-indexed）
    bboxes  : np.ndarray (N,4) [x1,y1,x2,y2]
    score_thresh : 仅保留 score >= score_thresh 的预测
    target_label : 仅保留该标签的预测（通常0代表唯一类别"畴区"）
    image_shape  : (H,W)，用于RLE解码
    iou_merge_thresh : 相邻patch重叠IoU阈值（兼容接口，此处不做合并）
    min_pixel_count  : 过滤过小实例

    Returns
    -------
    global_instances : list[dict]
    """
    scores_np = np.asarray(scores, dtype=np.float32) if scores is not None else None
    labels_np = np.asarray(labels, dtype=np.int64) if labels is not None else None

    # 处理RLE格式（pycocotools）
    if isinstance(masks, list) and len(masks) > 0 and isinstance(masks[0], dict):
        try:
            from pycocotools import mask as cocomask  # type: ignore
            decoded = []
            for rle in masks:
                decoded.append(cocomask.decode(rle).astype(bool))
            masks_arr = np.stack(decoded, axis=0)  # (N,H,W)
        except ImportError:
            raise ImportError(
                "RLE格式的masks需要安装 pycocotools: pip install pycocotools"
            )
    else:
        masks_arr = np.asarray(masks, dtype=bool) if not isinstance(masks, np.ndarray) else masks.astype(bool)

    if masks_arr.ndim == 2:
        masks_arr = masks_arr[np.newaxis]  # (1,H,W)

    N = masks_arr.shape[0]
    instances: list[dict] = []
    inst_id = 1

    for i in range(N):
        # 分数过滤
        if scores_np is not None and float(scores_np[i]) < score_thresh:
            continue
        # 类别过滤
        if labels_np is not None and int(labels_np[i]) != target_label:
            continue

        mask = masks_arr[i]  # (H,W) bool
        ys, xs = np.where(mask)
        if ys.size < min_pixel_count:
            continue

        coords = np.stack([ys, xs], axis=1).astype(np.int32)

        if bboxes is not None and i < len(bboxes):
            bb = bboxes[i]
            bbox = [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])]
        else:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

        instances.append({
            "id": inst_id,
            "coords": coords,
            "bbox": bbox,
            "score": float(scores_np[i]) if scores_np is not None else 1.0,
        })
        inst_id += 1

    return instances


# ---------------------------------------------------------------------------
# Pred: 从MMDetection保存的results.pkl加载
# ---------------------------------------------------------------------------

def load_mmdet_results_pkl(
    results_pkl: str,
    img_idx: int,
    *,
    score_thresh: float = 0.5,
    target_label: int = 0,
    min_pixel_count: int = 10,
) -> list[dict]:
    """从MMDetection test时保存的results.pkl加载单张图的预测实例。

    results.pkl 通常是 list[DetDataSample]，每个元素对应一张图。
    """
    import pickle

    with open(results_pkl, "rb") as f:
        results = pickle.load(f)

    if img_idx >= len(results):
        raise IndexError(f"img_idx={img_idx} out of range (len={len(results)})")

    data_sample = results[img_idx]

    # MMDet3.x DetDataSample
    pred = getattr(data_sample, "pred_instances", None)
    if pred is None:
        raise AttributeError(f"results[{img_idx}] has no pred_instances")

    masks = getattr(pred, "masks", None)
    scores = getattr(pred, "scores", None)
    labels = getattr(pred, "labels", None)
    bboxes = getattr(pred, "bboxes", None)

    if masks is None:
        raise ValueError(f"results[{img_idx}].pred_instances.masks is None")

    # 转为numpy
    if hasattr(masks, "numpy"):
        masks = masks.numpy()
    if scores is not None and hasattr(scores, "numpy"):
        scores = scores.numpy()
    if labels is not None and hasattr(labels, "numpy"):
        labels = labels.numpy()
    if bboxes is not None and hasattr(bboxes, "numpy"):
        bboxes = bboxes.numpy()

    return mmdet_masks_to_instances(
        masks,
        scores=scores,
        labels=labels,
        bboxes=bboxes,
        score_thresh=score_thresh,
        target_label=target_label,
        min_pixel_count=min_pixel_count,
    )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _find_image_info(coco: dict, image_id, image_filename) -> dict:
    if image_id is not None:
        for img in coco.get("images", []):
            if str(img["id"]) == str(image_id):
                return img
    if image_filename is not None:
        stem = os.path.splitext(os.path.basename(image_filename))[0]
        for img in coco.get("images", []):
            fn = img.get("file_name", "")
            if fn == image_filename or os.path.basename(fn) == image_filename:
                return img
            if os.path.splitext(os.path.basename(fn))[0] == stem:
                return img
    raise ValueError(
        f"Image not found: image_id={image_id}, filename={image_filename}"
    )
