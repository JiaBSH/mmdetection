#!/bin/bash
#SBATCH --job-name=mm
#SBATCH -p qgpu_4090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# ❗不要 load python module
module purge

# ✅ 用你自己的 conda
source /hpcfs/fpublic/app/miniforge3/conda/etc/profile.d/conda.sh

#conda init
conda activate openmmlab2

echo "===== DEBUG ====="
which python
python -V
pip list | grep -E "mmcv|mmdet|mmengine"
echo "================="

cd /hpcfs/fhome/sunxc/JiaBSH/mmdetection

bash tools/train_then_test_instance_seg.sh \
    configs/custom/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    1 \
    work_dirs/custom_maskrcnn_1cls
