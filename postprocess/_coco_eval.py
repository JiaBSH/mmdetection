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

def evaluate_coco_from_predictions(
    coco_predictions: list[dict[str, Any]],
    ann_file: str,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """用 pycocotools 对收集的预测做 COCO 评估。

    Parameters
    ----------
    coco_predictions : list[dict] — COCO 标准格式预测列表
    ann_file : str — COCO 标注 JSON 路径
    metrics : list[str] | None — 评估指标，默认 ["bbox", "segm"]

    Returns
    -------
    dict — 例如 {"bbox_mAP": 0.xxx, "bbox_mAP_50": 0.xxx, "segm_mAP": 0.xxx, ...}
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if metrics is None:
        metrics = ["bbox", "segm"]

    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(coco_predictions)

    results: dict[str, float] = {}
    for metric in metrics:
        coco_eval = COCOeval(coco_gt, coco_dt, metric)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # stats: [AP, AP50, AP75, AP_s, AP_m, AP_l,
        #         AR1, AR10, AR100, AR_s, AR_m, AR_l]
        stats = coco_eval.stats
        results[f"{metric}_mAP"] = float(stats[0])
        results[f"{metric}_mAP_50"] = float(stats[1])
        results[f"{metric}_mAP_75"] = float(stats[2])
        results[f"{metric}_mAP_s"] = float(stats[3])
        results[f"{metric}_mAP_m"] = float(stats[4])
        results[f"{metric}_mAP_l"] = float(stats[5])

    return results
