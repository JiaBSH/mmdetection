#!/bin/bash
# ==========================================================================
# 消融实验矩阵 (6组): M1/M2/M3 模型 × 固定/自适应窗口 × overlap
#
# 测试集: data/syn_multimag/coco/images/test/ (原始全图, 150张)
# 标注:   data/syn_multimag/coco/annotations/instances_test.json
# ==========================================================================
eval "$(conda shell.bash hook)"
conda activate mmdetection_para

BASE=$(realpath .)
TEST_IMG_DIR=$BASE/data/syn_multimag/coco/images/test
TEST_ANN=$BASE/data/syn_multimag/coco/annotations/instances_test.json
SCALE_MODEL=$BASE/data/syn_multimag/scale_pipeline_dinov2.joblib
POSTPROC=$BASE/postprocess/run_postprocess.py

# Model checkpoints
M1_CKPT=$BASE/work_dirs/ablation_m1_single_mag/epoch_5.pth
M2_CKPT=$BASE/work_dirs/ablation_m2_multimag/epoch_5.pth
M3_CKPT=$BASE/work_dirs/ablation_m3_multimag_jitter/epoch_5.pth

CONFIG=$BASE/configs/custom/mask-rcnn_r50_fpn_1x_custom_coco_instance.py

OUTROOT=$BASE/work_dirs/ablation_results
mkdir -p $OUTROOT

run_exp() {
    local exp_id=$1 model_ckpt=$2 patch_size=$3 overlap=$4
    local outdir=$OUTROOT/${exp_id}

    echo "===== $exp_id $(date) ====="
    echo "  model: $model_ckpt"
    echo "  patch_size: $patch_size  overlap: $overlap"
    echo "  outdir: $outdir"

    python $POSTPROC \
        --config $CONFIG \
        --checkpoint $model_ckpt \
        --ann-file $TEST_ANN \
        --img-dir $TEST_IMG_DIR \
        --out-dir $outdir \
        --sliding-window \
        --patch-size $patch_size \
        --patch-overlap-ratio $overlap \
        --enable-poly-metrics \
        --enable-gt \
        --score-thresh 0.5

    echo "===== $exp_id DONE $? ====="
}

# ------------------------------------------------------------------
# E1: M1 (50x only)  + Fixed-1024 + overlap=0
# ------------------------------------------------------------------
run_exp E1_M1_fixed1024 "$M1_CKPT" 1024 0.0

# ------------------------------------------------------------------
# E2: M2 (all mag, adaptive patches)  + Fixed-1024 + overlap=0
# ------------------------------------------------------------------
run_exp E2_M2_fixed1024 "$M2_CKPT" 1024 0.0

# ------------------------------------------------------------------
# E3: M3 (all mag + window jitter)  + Fixed-1024 + overlap=0
# ------------------------------------------------------------------
run_exp E3_M3_fixed1024 "$M3_CKPT" 1024 0.0

# ------------------------------------------------------------------
# E4: M3  + 最优固定窗口对比 (Fixed-512 + overlap=0.2)
# ------------------------------------------------------------------
run_exp E4_M3_fixed512 "$M3_CKPT" 512 0.2

# ------------------------------------------------------------------
# E5: M3  + 自适应窗口 (DINOv2) + overlap=0
#    每张图根据倍率预测动态选窗口: 2.5x→256, 5x→512, 20x→2048, 50x→2048, 100x→2048
# ------------------------------------------------------------------
echo "===== E5: M3 + adaptive window ====="
python -c "
import os, sys, subprocess, json
sys.path.insert(0, '$BASE')
from postprocess.adaptive_scale import AdaptiveWindowPredictor

predictor = AdaptiveWindowPredictor('$SCALE_MODEL')
ann_file = '$TEST_ANN'
img_dir = '$TEST_IMG_DIR'
config = '$CONFIG'
ckpt = '$M3_CKPT'
postproc = '$POSTPROC'
out_base = '$OUTROOT/E5_M3_adaptive_window'

with open(ann_file) as f:
    coco = json.load(f)

for img_info in coco['images']:
    fname = img_info['file_name']
    img_path = os.path.join(img_dir, fname)
    if not os.path.exists(img_path):
        continue

    s, mag, window, _ = predictor.predict(img_path)
    print(f'{fname}: scale={s:.2f} mag={mag:.1f}x window={window}')

    outdir = os.path.join(out_base, os.path.splitext(fname)[0])
    os.makedirs(outdir, exist_ok=True)

    subprocess.run([
        'python', postproc,
        '--config', config,
        '--checkpoint', ckpt,
        '--ann-file', ann_file,
        '--img', img_path,
        '--out-dir', outdir,
        '--sliding-window',
        '--patch-size', str(window),
        '--patch-overlap-ratio', '0.0',
        '--enable-poly-metrics',
        '--score-thresh', '0.5',
    ], check=False)
" 2>&1 | tail -20

# ------------------------------------------------------------------
# E6: M3  + 自适应窗口 (DINOv2) + 自适应overlap (OURS)
# ------------------------------------------------------------------
echo "===== E6: M3 + adaptive window + adaptive overlap (OURS) ====="
python -c "
import os, sys, subprocess, json
sys.path.insert(0, '$BASE')
from postprocess.adaptive_scale import AdaptiveWindowPredictor

predictor = AdaptiveWindowPredictor('$SCALE_MODEL')
ann_file = '$TEST_ANN'
img_dir = '$TEST_IMG_DIR'
config = '$CONFIG'
ckpt = '$M3_CKPT'
postproc = '$POSTPROC'
out_base = '$OUTROOT/E6_M3_adaptive_full'

with open(ann_file) as f:
    coco = json.load(f)

for img_info in coco['images']:
    fname = img_info['file_name']
    img_path = os.path.join(img_dir, fname)
    if not os.path.exists(img_path):
        continue

    s, mag, window, overlap = predictor.predict(img_path)
    print(f'{fname}: scale={s:.2f} mag={mag:.1f}x window={window} overlap={overlap:.2f}')

    outdir = os.path.join(out_base, os.path.splitext(fname)[0])
    os.makedirs(outdir, exist_ok=True)

    subprocess.run([
        'python', postproc,
        '--config', config,
        '--checkpoint', ckpt,
        '--ann-file', ann_file,
        '--img', img_path,
        '--out-dir', outdir,
        '--sliding-window',
        '--patch-size', str(window),
        '--patch-overlap-ratio', str(overlap),
        '--enable-poly-metrics',
        '--score-thresh', '0.5',
    ], check=False)
" 2>&1 | tail -20

echo ""
echo "===== All ablation experiments done $(date) ====="
for d in $OUTROOT/E*; do
    echo -n "$(basename $d): "
    ls $d/metrics_summary.csv 2>/dev/null && echo "OK" || echo "missing"
done
