#!/bin/bash
#SBATCH --job-name=postproc
#SBATCH -p qgpu_4090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

module purge
source /hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh
conda activate openmmlab2

cd /hpcfs/fhome/sunxc/JiaBSH/mmdetection

echo "===== 多模型后处理几何评估 ====="

MODEL_ROOT="work_dirs/custom_all_main_es"
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-}"

COMPARE_ARGS=(
    --ann-file dataset_root/dataset_1024_aug/annotations/instances_test.json
    --img-dir dataset_root/dataset_1024_aug/images/test
    --out-dir outputs/custom_all_main_es_comparison
    --model-cfg postprocess/model_list.yaml
    --model-root "${MODEL_ROOT}"
    --score-thresh 0.5
    --enable-poly-metrics
    --enable-gt
    --device cuda:0
)

if [[ -n "${CHECKPOINT_EPOCH}" ]]; then
    COMPARE_ARGS+=(--checkpoint-epoch "${CHECKPOINT_EPOCH}")
fi

# 使用 model_list.yaml 批量对比所有模型
# --enable-poly-metrics 开启 IoU/Precision/Recall/F1 像素级指标
# --enable-gt           分析GT几何分布（需GT标注文件）
# --enable-plots        生成GT vs Pred直方图/R^2散点图（耗时，可按需开启）
python postprocess/compare_models.py "${COMPARE_ARGS[@]}"

echo "===== 完成 ====="
