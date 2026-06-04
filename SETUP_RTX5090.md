# RTX 50 系显卡 (Blackwell) 上配置 mmdetection 环境教程

## 适用场景

- GPU: NVIDIA RTX 5090 / 5080 等 Blackwell 架构 (compute capability sm_120)
- 系统: Linux + SLURM 集群
- 目标: 在 50 系显卡上运行 OpenMMLab mmdetection 目标检测框架

## 环境概览

| 组件 | 最终版本 | 安装方式 |
|------|----------|----------|
| Python | 3.10 | conda |
| PyTorch | 2.12.0+cu130 | pip (官方 whl) |
| mmcv | 2.2.0 | conda-forge (预编译，无需编译 CUDA) |
| mmengine | 0.10.7 | pip |
| mmdet | 3.3.0 | pip -e . (源码可编辑) |
| CUDA Toolkit | 12.8 (系统) / 12.9 (mmcv 运行库) | — |

> **核心原则**: 不要自己编译 mmcv！conda-forge 有预编译包，省去数小时的 CUDA 编译和兼容性问题。

---

## 第一步：创建 conda 环境

```bash
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda create -n mmdetection_para python=3.10 -y
conda activate mmdetection_para
```

---

## 第二步：安装 mmcv（必须先于 PyTorch！）

> **关键顺序**: 先装 mmcv，再装 PyTorch。否则 conda 会把 pip 的 PyTorch 替换掉。

```bash
conda install -c conda-forge "mmcv=2.2.0=*cuda129*py310*" -y
```

这条命令会：
- 安装预编译的 mmcv 2.2.0（包含所有 CUDA 算子，开箱即用）
- 附带安装一个 conda 版本的 PyTorch（之后会被 pip 版覆盖）
- **无需 nvcc 编译**，几十秒完成

---

## 第三步：安装 PyTorch（覆盖 conda 版）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> **为什么这样做？** conda 装 mmcv 时自动装了一个 conda-forge 的 PyTorch，那个版本在计算节点上可能 `torch.cuda.is_available()` 返回 False。用 pip 官方包覆盖它。

验证 GPU:

```bash
# 注意：登录节点无 GPU，必须在计算节点测试！
srun --gpus=1 bash -c 'source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh && conda activate mmdetection_para && python -c "import torch; print(torch.cuda.get_device_name(0))"'
# 应输出: NVIDIA GeForce RTX 5090
```

---

## 第四步：安装其他依赖

```bash
pip install mmengine opencv-python matplotlib numpy pyyaml \
    tqdm six scipy shapely terminaltables pycocotools
```

---

## 第五步：安装 mmdet

mmdet 源码里有 mmcv < 2.2.0 的硬编码版本检查，需要先放宽。

编辑 `mmdet/__init__.py`，找到：

```python
mmcv_maximum_version = '2.2.0'
```

改为：

```python
mmcv_maximum_version = '2.3.0'
```

然后安装：

```bash
cd /path/to/mmdetection_para
pip install --no-build-isolation -e .
```

---

## 第六步：GPU 功能验证

在项目目录下创建 `test_gpu_quick.py`：

```python
#!/usr/bin/env python
"""Minimal GPU functional test for mmdetection on G5090."""
import torch, mmcv, mmengine, mmdet
from mmdet.utils import register_all_modules
from mmdet.registry import MODELS
from mmengine.config import Config

register_all_modules()

print("=== Environment ===")
print(f"mmdet: {mmdet.__version__}  mmcv: {mmcv.__version__}  mmengine: {mmengine.__version__}")
print(f"PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Build RTMDet-tiny
cfg = Config.fromfile("configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py")
cfg.model.bbox_head.num_classes = 1

print("\n=== Building Model ===")
model = MODELS.build(cfg.model)
model = model.cuda()
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model built: {params:.2f}M params")

# ---- Inference ----
print("\n=== Forward (Inference) ===")
x = torch.randn(2, 3, 512, 512, device="cuda")
model.eval()
with torch.no_grad():
    out = model(x, mode="tensor")
print(f"Input: {list(x.shape)}, OK")

# ---- Training ----
print("\n=== Forward+Backward (Training) ===")
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

model.train()
H, W = 512, 512
x = torch.randn(2, 3, H, W, device="cuda")
data_samples = []
for i in range(2):
    ds = DetDataSample()
    ds.set_metainfo({"img_shape": (H, W), "pad_shape": (H, W),
                     "scale_factor": (1.0, 1.0), "ori_shape": (H, W)})
    gt_instances = InstanceData()
    gt_instances.bboxes = torch.tensor([[50., 50., 200., 200.]], device="cuda")
    gt_instances.labels = torch.tensor([0], device="cuda")
    ds.gt_instances = gt_instances
    data_samples.append(ds)

loss_dict = model(x, data_samples, mode="loss")
# RTMDet returns per-level lists; flatten and sum
all_losses = []
for v in loss_dict.values():
    if isinstance(v, list):
        all_losses.extend(v)
    else:
        all_losses.append(v)
total_loss = sum(all_losses)
total_loss.backward()
print(f"Loss keys: {list(loss_dict.keys())}")
print(f"Total loss: {total_loss.item():.4f}, Backward: OK")

print(f"\nGPU max memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print(f"\n===== ALL TESTS PASSED =====")
print(f"G5090 environment is ready for training!")
```

在计算节点上运行：

```bash
srun --gpus=1 bash -c 'source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh && conda activate mmdetection_para && python test_gpu_quick.py'
```

期望输出：

```
=== Environment ===
mmdet: 3.3.0  mmcv: 2.2.0  mmengine: 0.10.7
PyTorch: 2.12.0+cu130  CUDA: True
GPU: NVIDIA GeForce RTX 5090

=== Building Model ===
Model built: 4.87M params

=== Forward (Inference) ===
Input: [2, 3, 512, 512], OK

=== Forward+Backward (Training) ===
Loss keys: ['loss_cls', 'loss_bbox']
Total loss: xxxx.xxxx, Backward: OK

GPU max memory: 0.41 GB

===== ALL TESTS PASSED =====
G5090 environment is ready for training!
```

---

## 常见问题

### 1. conda install mmcv 后 PyTorch GPU 不可用

**现象**:
```python
import torch; print(torch.cuda.is_available())  # False!
```

**原因**: conda 安装 mmcv 时替换了 pip 安装的 PyTorch 为 conda-forge 版本，该版本可能缺 GPU 支持。

**解决**: 在 mmcv 安装后重新用 pip 安装 PyTorch：
```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. pip install mmcv 源码编译时 Disk quota exceeded

**现象**:
```
fatal error: when writing output to .../sparse_pool_ops_cuda.cpp1.ii: Disk quota exceeded
```

**原因**: nvcc 编译 CUDA 文件的临时输出远超 `/data` 分区配额。

**解决**: 根本不需要编译——直接用 conda-forge 预编译包。如果实在要编译：
```bash
export TMPDIR=/tmp  # /tmp 通常有 >500GB 空间
```

### 3. mmcv 源码编译报 `ATen/CollapseDims.h: No such file or directory`

**现象**:
```
fatal error: ATen/CollapseDims.h: No such file or directory
```

**原因**: mmcv 2.1.0 不兼容 PyTorch 2.10+/2.12+。该头文件在新版 PyTorch 中已移除。

**解决**: 使用 mmcv 2.2.0 (conda-forge)，不要用 2.1.0。

### 4. `AssertionError: MMCV==2.2.0 is used but incompatible`

**原因**: mmdet 3.3.0 硬编码 `mmcv_maximum_version = '2.2.0'`。

**解决**: 修改 `mmdet/__init__.py`，将 `'2.2.0'` 改为 `'2.3.0'`。

### 5. 登录节点 vs 计算节点

登录节点没有 GPU (nvidia-smi 不可用，CUDA_VISIBLE_DEVICES 为空)。所有 GPU 相关测试必须通过 srun 在计算节点上运行：

```bash
# 交互式
srun --gpus=1 --pty bash

# 或直接运行命令
srun --gpus=1 bash -c 'conda activate mmdetection_para && python my_script.py'
```

### 6. `persistent_workers` 和 `num_workers=0` 冲突

**现象**:
```
ValueError: persistent_workers option needs num_workers > 0
```

**解决**: 如果设 `num_workers=0`（调试时），必须同时设 `persistent_workers=False`。

---

## 安装顺序总结

```
1. conda create -n <env> python=3.10
2. conda install -c conda-forge mmcv=2.2.0=*cuda129*py310*   ← 预编译！
3. pip install torch torchvision torchaudio --index-url ...   ← 覆盖 conda 版
4. pip install mmengine opencv-python matplotlib numpy ...
5. 修改 mmdet/__init__.py (mmcv_maximum_version = '2.3.0')
6. pip install --no-build-isolation -e .
7. srun --gpus=1 python test_gpu_quick.py                     ← 必须在 GPU 节点
```

---

**编写日期**: 2026-06-03  
**验证环境**: NVIDIA RTX 5090 (32GB) + CUDA 12.8 Driver 570.144  
**适用 GPU**: RTX 5090 / 5080 / 5070 等 Blackwell 架构 (sm_120)
