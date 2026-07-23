"""COCO 格式结果收集与 pycocotools 评估。

一次推理 → 同时输出 bbox_mAP + segm_mAP + 像素级指标 + 逐实例指标。
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# COCO 格式结果收集器
# ---------------------------------------------------------------------------

class COCOResultCollector:
    """从推理结果中收集 COCO 格式预测（bbox + segm）。"""

    def __init__(self):
        self._predictions: list[dict[str, Any]] = []

    def add_image_predictions(
        self,
        image_id: int,
        global_instances: list[dict],
        img_width: int,
        img_height: int,
        category_id: int = 1,
    ) -> None:
        """从 global_instances 格式提取 COCO 预测。

        Parameters
        ----------
        image_id : int — COCO annotation 中的 image id
        global_instances : list[dict] — 含 coords (ndarray N×2, row=y, col=x),
                                        bbox [xmin, ymin, xmax, ymax],
                                        score
        img_width : int
        img_height : int
        category_id : int — 类别 ID，默认 1
        """
        for inst in global_instances:
            coords = inst.get("coords")
            score = float(inst.get("score", 1.0))
            bbox = inst.get("bbox")

            if coords is None or len(coords) == 0:
                continue

            # 从 pixel coords 构建二进制掩膜
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            ys = coords[:, 0].astype(np.int64)
            xs = coords[:, 1].astype(np.int64)
            valid = (
                (ys >= 0) & (ys < img_height) & (xs >= 0) & (xs < img_width)
            )
            mask[ys[valid], xs[valid]] = 1

            # RLE 编码
            try:
                from pycocotools import mask as mask_utils
                rle = mask_utils.encode(
                    np.asfortranarray(mask.astype(np.uint8))
                )
                # COCOeval expects counts as string
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")
                segmentation = rle
            except ImportError:
                # 无 pycocotools 时跳过 segm
                segmentation = None

            # bbox: COCO 格式 [x, y, width, height]
            if bbox is not None and len(bbox) == 4:
                coco_bbox = [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]) - float(bbox[0]),
                    float(bbox[3]) - float(bbox[1]),
                ]
            else:
                # 从 coords 重建 bbox
                if valid.sum() > 0:
                    coco_bbox = [
                        float(xs[valid].min()),
                        float(ys[valid].min()),
                        float(xs[valid].max()) - float(xs[valid].min()),
                        float(ys[valid].max()) - float(ys[valid].min()),
                    ]
                else:
                    continue

            pred = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": coco_bbox,
                "score": score,
            }
            if segmentation is not None:
                pred["segmentation"] = segmentation

            self._predictions.append(pred)

    def to_coco_list(self) -> list[dict[str, Any]]:
        """返回收集到的 COCO 预测列表。"""
        return list(self._predictions)

    def save(self, path: str) -> None:
        """保存为 JSON 文件。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._predictions, f, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self._predictions)


# ---------------------------------------------------------------------------
# COCO 评估
# ---------------------------------------------------------------------------

def _mean_coco_precision(
    coco_eval: Any,
    *,
    max_dets: int,
    iou_threshold: float | None = None,
    area_label: str = "all",
) -> float:
    """Read AP directly from COCOeval tensors without its maxDets=100 summary."""
    precision = coco_eval.eval["precision"]
    params = coco_eval.params

    area_indices = [
        index
        for index, label in enumerate(params.areaRngLbl)
        if label == area_label
    ]
    max_det_indices = [
        index
        for index, value in enumerate(params.maxDets)
        if int(value) == int(max_dets)
    ]
    if not area_indices or not max_det_indices:
        return -1.0

    if iou_threshold is not None:
        iou_indices = np.flatnonzero(
            np.isclose(params.iouThrs, iou_threshold)
        )
        precision = precision[iou_indices]

    precision = precision[:, :, :, area_indices, max_det_indices]
    valid = precision[precision > -1]
    return float(np.mean(valid)) if valid.size else -1.0


def evaluate_coco_from_predictions(
    coco_predictions: list[dict[str, Any]],
    ann_file: str,
    metrics: list[str] | None = None,
    image_ids: list[int] | None = None,
    max_dets: int = 10000,
) -> dict[str, float]:
    """用 pycocotools 对收集的预测做 COCO 评估。

    Parameters
    ----------
    coco_predictions : list[dict] — COCO 标准格式预测列表
    ann_file : str — COCO 标注 JSON 路径
    metrics : list[str] | None — 评估指标，默认 ["bbox", "segm"]
    image_ids : list[int] | None — 本次实际参与推理的 COCO image_id。
        未提供时兼容旧行为，仅评估预测结果中出现的图像。
    max_dets : int — COCOeval 每张图最多参与评估的实例数。
        密集颗粒图远超 COCO 默认的 100，因此默认使用 10000。

    Returns
    -------
    dict — 例如 {"bbox_mAP": 0.xxx, "bbox_mAP_50": 0.xxx, "segm_mAP": 0.xxx, ...}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if metrics is None:
        metrics = ["bbox", "segm"]
    if max_dets < 100:
        raise ValueError(f"max_dets must be >= 100, got {max_dets}")

    coco_gt = COCO(ann_file)
    valid_gt_ids = set(coco_gt.getImgIds())
    if image_ids is None:
        eval_img_ids = sorted({
            int(pred["image_id"])
            for pred in coco_predictions
            if "image_id" in pred and int(pred["image_id"]) in valid_gt_ids
        })
    else:
        eval_img_ids = sorted({
            int(image_id)
            for image_id in image_ids
            if int(image_id) in valid_gt_ids
        })
    if not eval_img_ids:
        return {}

    if coco_predictions:
        coco_dt = coco_gt.loadRes(coco_predictions)
    else:
        # pycocotools.loadRes([]) cannot build an empty result set. Construct
        # one explicitly so an evaluated image with zero detections scores 0.
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": [
                image
                for image in coco_gt.dataset.get("images", [])
                if int(image["id"]) in eval_img_ids
            ],
            "categories": list(coco_gt.dataset.get("categories", [])),
            "annotations": [],
        }
        coco_dt.createIndex()

    results: dict[str, float] = {}
    for metric in metrics:
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        coco_eval.params.imgIds = eval_img_ids
        coco_eval.params.maxDets = [1, 10, int(max_dets)]
        coco_eval.evaluate()
        coco_eval.accumulate()
        ap_values = {
            f"{metric}_mAP": _mean_coco_precision(
                coco_eval, max_dets=max_dets
            ),
            f"{metric}_mAP_50": _mean_coco_precision(
                coco_eval, max_dets=max_dets, iou_threshold=0.50
            ),
            f"{metric}_mAP_75": _mean_coco_precision(
                coco_eval, max_dets=max_dets, iou_threshold=0.75
            ),
            f"{metric}_mAP_s": _mean_coco_precision(
                coco_eval, max_dets=max_dets, area_label="small"
            ),
            f"{metric}_mAP_m": _mean_coco_precision(
                coco_eval, max_dets=max_dets, area_label="medium"
            ),
            f"{metric}_mAP_l": _mean_coco_precision(
                coco_eval, max_dets=max_dets, area_label="large"
            ),
        }
        results.update(ap_values)
        print(
            f" Average Precision ({metric}) "
            f"@[ IoU=0.50:0.95 | area=all | maxDets={max_dets} ] "
            f"= {ap_values[f'{metric}_mAP']:.3f}"
        )

    return results
