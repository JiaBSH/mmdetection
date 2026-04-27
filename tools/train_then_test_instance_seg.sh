#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <config> <gpu_num> <work_dir> [extra args for tools/train.py and tools/test.py]"
  echo "Example:"
  echo "  bash tools/train_then_test_instance_seg.sh \\
    configs/custom/mask-rcnn_r50_fpn_1x_custom_coco_instance.py 1 work_dirs/custom_maskrcnn \\
    --cfg-options data_root=/data/my_coco/ \\
    train_dataloader.dataset.ann_file=annotations/instances_train.json \\
    train_dataloader.dataset.data_prefix.img=images/train/ \\
    val_dataloader.dataset.ann_file=annotations/instances_val.json \\
    val_dataloader.dataset.data_prefix.img=images/val/ \\
    test_dataloader.dataset.ann_file=annotations/instances_val.json \\
    test_dataloader.dataset.data_prefix.img=images/val/ \\
    val_evaluator.ann_file=/data/my_coco/annotations/instances_val.json \\
    test_evaluator.ann_file=/data/my_coco/annotations/instances_val.json \\
    model.roi_head.bbox_head.num_classes=1 model.roi_head.mask_head.num_classes=1"
  echo "Note: extra args should usually start with --cfg-options"
  exit 1
fi

CONFIG="$1"
GPUS="$2"
WORK_DIR="$3"
shift 3

# Ignore accidental standalone '.' argument from shell completion/history.
RAW_EXTRA_ARGS=("$@")
EXTRA_ARGS=()
for arg in "${RAW_EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "." ]]; then
    continue
  fi
  EXTRA_ARGS+=("$arg")
done

mkdir -p "$WORK_DIR"

echo "================ TRAIN ================="
if [[ "$GPUS" -gt 1 ]]; then
  bash tools/dist_train.sh "$CONFIG" "$GPUS" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
else
  python tools/train.py "$CONFIG" --work-dir "$WORK_DIR" "${EXTRA_ARGS[@]}"
fi

echo "========== FIND BEST CHECKPOINT =========="
BEST_CKPT=""

if compgen -G "$WORK_DIR/best_coco_segm_mAP*.pth" > /dev/null; then
  BEST_CKPT=$(ls -t "$WORK_DIR"/best_coco_segm_mAP*.pth | head -n 1)
elif compgen -G "$WORK_DIR/best_coco_bbox_mAP*.pth" > /dev/null; then
  BEST_CKPT=$(ls -t "$WORK_DIR"/best_coco_bbox_mAP*.pth | head -n 1)
elif [[ -f "$WORK_DIR/latest.pth" ]]; then
  BEST_CKPT="$WORK_DIR/latest.pth"
elif compgen -G "$WORK_DIR/epoch_*.pth" > /dev/null; then
  BEST_CKPT=$(ls -t "$WORK_DIR"/epoch_*.pth | head -n 1)
elif compgen -G "$WORK_DIR/iter_*.pth" > /dev/null; then
  BEST_CKPT=$(ls -t "$WORK_DIR"/iter_*.pth | head -n 1)
else
  echo "No checkpoint found in $WORK_DIR"
  exit 2
fi

echo "Use checkpoint: $BEST_CKPT"

echo "================ TEST ================="
if [[ "$GPUS" -gt 1 ]]; then
  bash tools/dist_test.sh "$CONFIG" "$BEST_CKPT" "$GPUS" --work-dir "$WORK_DIR/test" "${EXTRA_ARGS[@]}"
else
  python tools/test.py "$CONFIG" "$BEST_CKPT" --work-dir "$WORK_DIR/test" "${EXTRA_ARGS[@]}"
fi

echo "============= VISUALIZE TEST ============="
# Auto-derive test image directory from --cfg-options if provided.
_DATA_ROOT=""
_TEST_IMG_PREFIX=""
for _arg in "${EXTRA_ARGS[@]}"; do
  case "$_arg" in
    data_root=*)                               _DATA_ROOT="${_arg#data_root=}" ;;
    test_dataloader.dataset.data_prefix.img=*) _TEST_IMG_PREFIX="${_arg#test_dataloader.dataset.data_prefix.img=}" ;;
  esac
done

if [[ -z "${VIS_INPUTS:-}" ]]; then
  # If not provided via --cfg-options, parse from the saved config in WORK_DIR.
  if [[ -z "$_DATA_ROOT" || -z "$_TEST_IMG_PREFIX" ]]; then
    _SAVED_CFG=$(find "$WORK_DIR" -maxdepth 1 -name "*.py" | head -n 1)
    if [[ -n "$_SAVED_CFG" ]]; then
      [[ -z "$_DATA_ROOT" ]] && \
        _DATA_ROOT=$(grep -Eo "data_root\s*=\s*'[^']+'" "$_SAVED_CFG" | head -n1 | grep -Eo "'[^']+'" | tr -d "'")
      [[ -z "$_TEST_IMG_PREFIX" ]] && \
        _TEST_IMG_PREFIX=$(python3 -c "
import ast, sys
src = open('$_SAVED_CFG').read()
# look for test_dataloader dict img prefix
import re
m = re.search(r\"test_dataloader\s*=.*?data_prefix\s*=\s*dict\(img='([^']+)'\)\", src, re.S)
if m: print(m.group(1))
else: print('images/test/')
" 2>/dev/null)
    fi
  fi
  if [[ -n "$_DATA_ROOT" && -n "$_TEST_IMG_PREFIX" ]]; then
    VIS_INPUTS="${_DATA_ROOT%/}/${_TEST_IMG_PREFIX#/}"
  else
    VIS_INPUTS="dataset_root/images/test"
  fi
fi
VIS_OUT_DIR=${VIS_OUT_DIR:-$WORK_DIR/test_vis}
VIS_SCORE_THR=${VIS_SCORE_THR:-0.3}
VIS_DEVICE=${VIS_DEVICE:-cuda:0}

if [[ -d "$VIS_INPUTS" ]]; then
  # Pass BEST_CKPT directly as the model argument so image_demo.py reads
  # the config stored inside the checkpoint (which includes all cfg-options
  # overrides such as num_classes applied during training).
  if python demo/image_demo.py "$VIS_INPUTS" "$BEST_CKPT" \
      --device "$VIS_DEVICE" \
      --pred-score-thr "$VIS_SCORE_THR" \
      --out-dir "$VIS_OUT_DIR" \
      --batch-size 4; then
    echo "Visualization results saved to: $VIS_OUT_DIR"
  else
    echo "Skip visualization: demo/image_demo.py failed for $CONFIG"
  fi
else
  echo "Skip visualization: test image dir not found: $VIS_INPUTS"
  echo "You can set VIS_INPUTS to your test image directory and rerun."
fi

echo "============= PLOT METRICS ============="
if python tools/plot_metrics.py "$WORK_DIR" --out-dir "$WORK_DIR/metric_plots"; then
  echo "Metric plots saved to: $WORK_DIR/metric_plots"
else
  echo "Skip metric plotting: tools/plot_metrics.py failed for $CONFIG"
fi

echo "Done. Check logs under: $WORK_DIR and $WORK_DIR/test"
