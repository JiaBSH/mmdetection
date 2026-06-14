"""Chunked MaxIoUAssigner: batches GT boxes to avoid GPU OOM during IoU.

Standard MaxIoUAssigner computes IoU(gt_bboxes, priors) in one shot.
With 6000+ GT boxes × 170K+ anchors, the intermediate tensors consume
~35 GB GPU memory (formula: 9 × N × M × 4 bytes).  This assigner splits
GT boxes into chunks to keep peak memory bounded while staying on GPU.

Usage in config:
    assigner=dict(
        type='ChunkedMaxIoUAssigner',
        pos_iou_thr=0.7,
        neg_iou_thr=0.3,
        min_pos_iou=0.3,
        match_low_quality=True,
        iou_chunk_size=200,       # max GT boxes per IoU chunk (default 500)
    )
"""

from typing import Optional, Union

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .max_iou_assigner import MaxIoUAssigner, perm_repeat_bboxes


@TASK_UTILS.register_module()
class ChunkedMaxIoUAssigner(MaxIoUAssigner):
    """MaxIoUAssigner with chunked IoU to bound GPU peak memory.

    Additional Args:
        iou_chunk_size (int): Max number of GT boxes per IoU chunk.
            Default 500.  Lower = less peak memory, more iterations.
    """

    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: Union[float, tuple],
                 min_pos_iou: float = .0,
                 gt_max_assign_all: bool = True,
                 ignore_iof_thr: float = -1,
                 ignore_wrt_candidates: bool = True,
                 match_low_quality: bool = True,
                 gpu_assign_thr: float = -1,
                 iou_calculator: dict = dict(type='BboxOverlaps2D'),
                 perm_repeat_gt_cfg=None,
                 iou_chunk_size: int = 500):
        super().__init__(
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr,
            min_pos_iou=min_pos_iou,
            gt_max_assign_all=gt_max_assign_all,
            ignore_iof_thr=ignore_iof_thr,
            ignore_wrt_candidates=ignore_wrt_candidates,
            match_low_quality=match_low_quality,
            gpu_assign_thr=gpu_assign_thr,
            iou_calculator=iou_calculator,
            perm_repeat_gt_cfg=perm_repeat_gt_cfg,
        )
        self.iou_chunk_size = iou_chunk_size

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to priors with chunked IoU to bound GPU memory.

        When num_gt > iou_chunk_size, IoU is computed in chunks and
        max_overlaps / argmax_overlaps are tracked incrementally, *without*
        materializing the full (num_gt, num_priors) overlap matrix.
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        num_gts = gt_bboxes.shape[0]

        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        # — permute repeated gt (standard mmdet logic) —
        if self.perm_repeat_gt_cfg is not None and priors.numel() > 0:
            gt_bboxes_unique = perm_repeat_bboxes(gt_bboxes,
                                                  self.iou_calculator,
                                                  self.perm_repeat_gt_cfg)
        else:
            gt_bboxes_unique = gt_bboxes

        # — chunked IoU when too many GT boxes —
        if num_gts > self.iou_chunk_size and priors.numel() > 0:
            num_priors = priors.size(0)
            device = priors.device

            # Running max across chunks
            max_overlaps = priors.new_full((num_priors,), -1.0)
            argmax_overlaps = priors.new_zeros((num_priors,), dtype=torch.long)
            gt_max_overlaps = priors.new_zeros((num_gts,))
            gt_argmax_overlaps = priors.new_zeros((num_gts,), dtype=torch.long)

            for start in range(0, num_gts, self.iou_chunk_size):
                end = min(start + self.iou_chunk_size, num_gts)
                gt_chunk = gt_bboxes_unique[start:end]

                # IoU for this chunk: shape (chunk_size, num_priors)
                overlaps = self.iou_calculator(gt_chunk, priors)

                # Update per-prior max
                chunk_max, chunk_argmax = overlaps.max(dim=0)
                better = chunk_max > max_overlaps
                max_overlaps[better] = chunk_max[better]
                argmax_overlaps[better] = chunk_argmax[better] + start  # global index

                # Per-GT max (tracks the full row for each GT)
                row_max, row_argmax = overlaps.max(dim=1)
                gt_max_overlaps[start:end] = row_max
                gt_argmax_overlaps[start:end] = row_argmax

            # — assign using the aggregated statistics —
            assigned_gt_inds = overlaps.new_full((num_priors,), -1, dtype=torch.long)

            if num_gts == 0:
                assigned_gt_inds[:] = 0
                assigned_labels = overlaps.new_full((num_priors,), -1, dtype=torch.long)
                return AssignResult(
                    num_gts=num_gts,
                    gt_inds=assigned_gt_inds,
                    max_overlaps=max_overlaps,
                    labels=assigned_labels,
                )

            # Step 2: assign negatives (below neg_iou_thr)
            if isinstance(self.neg_iou_thr, float):
                assigned_gt_inds[(max_overlaps >= 0)
                                 & (max_overlaps < self.neg_iou_thr)] = 0
            elif isinstance(self.neg_iou_thr, tuple):
                assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                                 & (max_overlaps < self.neg_iou_thr[1])] = 0

            # Step 3: assign positives
            pos_inds = max_overlaps >= self.pos_iou_thr
            assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

            # Step 4: low-quality matching
            if self.match_low_quality:
                for i in range(num_gts):
                    if gt_max_overlaps[i] >= self.min_pos_iou:
                        if self.gt_max_assign_all:
                            # Need the full row for this GT — compute it on demand
                            gt_i = gt_bboxes_unique[i:i + 1]
                            row_overlaps = self.iou_calculator(gt_i, priors)[0]
                            max_iou_inds = row_overlaps == gt_max_overlaps[i]
                            assigned_gt_inds[max_iou_inds] = i + 1
                        else:
                            assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

            # Build labels
            assigned_labels = assigned_gt_inds.new_full((num_priors,), -1)
            pos_mask = assigned_gt_inds > 0
            if pos_mask.any():
                assigned_labels[pos_mask] = gt_labels[assigned_gt_inds[pos_mask] - 1]

            assign_result = AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels,
            )

            # — handle ignore bboxes —
            if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                    and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
                if self.ignore_wrt_candidates:
                    ignore_overlaps = self.iou_calculator(
                        priors, gt_bboxes_ignore, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
                else:
                    ignore_overlaps = self.iou_calculator(
                        gt_bboxes_ignore, priors, mode='iof')
                    ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
                assign_result.gt_inds[ignore_max_overlaps > self.ignore_iof_thr] = -1

            return assign_result

        # — fall through to standard assign for small num_gts —
        return super().assign(
            pred_instances, gt_instances, gt_instances_ignore, **kwargs)
