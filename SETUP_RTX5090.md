# RTX 50 系显卡 (Blackwell) 上配置 mmdetection 环境教程

## 适用场景

- GPU: NVIDIA RTX 5090 / 5080 等 Blackwell 架构 (compute capability sm_120)
- CUDA Driver: **12.8** 或 **13.0**（本教程兼容两者）
- 系统: Linux + SLURM 集群
- 目标: 在 50 系显卡上运行 OpenMMLab mmdetection 目标检测框架

> **注意**: RTX 5090 依赖 CUDA 12.8 编译的 PyTorch。部分节点只有 CUDA 12.8 driver（不支持 CUDA 13.0），本教程使用 cu128 PyTorch + cuda129 mmcv 的混合方案，同时兼容两种 driver。

## 环境概览

| 组件 | 最终版本 | 安装方式 |
|------|----------|----------|
| Python | 3.10 | conda |
| PyTorch | 2.12.0 (cu128) | pip (官方 whl) |
| mmcv | 2.2.0 (cuda129) | conda-forge (预编译，--no-deps) |
| mmengine | 0.10.7 | pip |
| mmdet | 3.3.0 | pip -e . (源码可编辑) |
| NCCL | 2.29.7 (cu12) | pip (替换 PyTorch 自带的 cu13 版本) |
| CUDA Runtime | cu12 12.9 + cu13 13.0 | pip (双 runtime 共存) |

> **核心原则**: 不要自己编译 mmcv！conda-forge 有预编译包，省去数小时的 CUDA 编译和兼容性问题。

---

## 第一步：创建 conda 环境

```bash
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda create -n mmdet_cu128 python=3.10 -y
conda activate mmdet_cu128
```

> 环境名建议用 `mmdet_cu128` 以便区分。

---

## 第二步：安装 PyTorch（必须先于 mmcv！）

> **关键顺序**: 先装 PyTorch，再装 mmcv。否则 conda solver 会因为版本冲突无法解析 mmcv 依赖。

```bash
pip install torch==2.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> 超算官方说明：RTX 5090 依赖 CUDA 12.8，PyTorch 必须使用 cu128 索引安装，且版本 ≥ 2.7.0。本教程使用当前最新 2.12.0。
>
> 安装后 `torch.__version__` 会显示 `2.12.0+cu130`（cu128 索引的包内部版本号如此），不影响使用。

---

## 第三步：替换 NCCL 和添加 CUDA 12 Runtime

PyTorch cu128 自带的 NCCL 是 `nvidia-nccl-cu13`（CUDA 13 编译），在纯 CUDA 12.8 节点上可能有问题。需要替换为 CUDA 12 编译的新版 NCCL。

同时，mmcv cuda129 需要 `libcudart.so.12`，而 PyTorch cu128 只带了 `libcudart.so.13`，需要额外安装。

```bash
# 3a. 替换 NCCL 为 cu12 版本（必须 2.29.7+，PyTorch 2.12.0 需要 ncclCommResume 符号）
pip install nvidia-nccl-cu12==2.29.7 --force-reinstall --no-deps

# 3b. 安装 CUDA 12.x runtime（mmcv cuda129 依赖 libcudart.so.12）
pip install nvidia-cuda-runtime-cu12
```

验证 NCCL:
```bash
# 在计算节点上
srun --gpus=1 bash -c 'source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh && conda activate mmdet_cu128 && export NCCL_DEBUG=INFO && python -c "import torch; print(torch.cuda.nccl.version())"'
# 应输出: (2, 29, 7)
```

---

## 第四步：安装 mmcv（预编译，跳过依赖）

conda-forge 的 mmcv cuda129 包依赖 `pytorch >=2.7.1,<2.8.0a0`，与已装的 2.12.0 冲突。下载本地包 + `--no-deps` 跳过依赖即可。

```bash
wget -q "https://conda.anaconda.org/conda-forge/linux-64/mmcv-2.2.0-cuda129py310ha3febd4_211.conda" -O /tmp/mmcv.conda
conda install /tmp/mmcv.conda --no-deps -y
```

> **`--no-deps` 安全吗？** 安全。mmcv 的 CUDA 算子（nms、roi_align 等）是预编译在 `.conda` 包里的 `.so` 文件，不依赖 conda 的 PyTorch。运行时只需要 PyTorch（已 pip 安装）和 CUDA runtime（已安装），无需额外依赖。

---

## 第五步：安装其他依赖

```bash
pip install mmengine opencv-python matplotlib numpy pyyaml \
    tqdm six scipy shapely terminaltables pycocotools addict yapf
```

---

## 第六步：安装 mmdet

mmdet 源码里有 mmcv < 2.2.0 的硬编码版本检查，需先确认已放宽（本项目已修改为 2.3.0）。

检查 `mmdet/__init__.py`：

```python
mmcv_maximum_version = '2.3.0'  # 应已改为 2.3.0
```

然后安装：

```bash
cd /path/to/mmdetection_para
pip install --no-build-isolation --no-deps -e .
```

---

## 第七步：GPU 功能验证

在项目目录下创建 `test_gpu_quick.py`：

```python
#!/usr/bin/env python
"""Minimal GPU functional test for mmdetection on RTX 5090."""
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
print(f"RTX 5090 environment is ready for training!")
```

在计算节点上运行：

```bash
srun --gpus=1 bash -c 'source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh && conda activate mmdet_cu128 && python test_gpu_quick.py'
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
Total loss: 2264.8955, Backward: OK

GPU max memory: 0.41 GB

===== ALL TESTS PASSED =====
RTX 5090 environment is ready for training!
```

---

## 常见问题

### 1. `ImportError: undefined symbol: ncclCommResume`

**现象**:
```
ImportError: libtorch_cuda.so: undefined symbol: ncclCommResume
```

**原因**: PyTorch 2.12.0 需要 NCCL ≥ 2.29.x 的 `ncclCommResume` 符号，但安装的 nvidia-nccl-cu12 版本过低（如 2.27.5）。

**解决**: 安装 2.29.7+ 版本：
```bash
pip install nvidia-nccl-cu12==2.29.7 --force-reinstall --no-deps
```

### 2. `libcudart.so.12: cannot open shared object file`

**现象**:
```
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

**原因**: PyTorch cu128 只提供 `libcudart.so.13`，但 mmcv cuda129 需要 `libcudart.so.12`。

**解决**:
```bash
pip install nvidia-cuda-runtime-cu12
```

### 3. conda install mmcv 时 python 版本冲突

**现象**:
```
Pins seem to be involved in the conflict. Currently pinned specs:
 - python=3.10
```

**原因**: conda-forge 的 mmcv 依赖 `pytorch >=2.7.1,<2.8.0a0`，而新版 pytorch 需要 python ≥ 3.14，与 python=3.10 冲突。

**解决**: 下载本地包 + `--no-deps` 安装（见第四步）。

### 4. conda install mmcv 后 PyTorch GPU 不可用

**现象**:
```python
import torch; print(torch.cuda.is_available())  # False!
```

**原因**: conda 安装 mmcv 时替换了 pip 安装的 PyTorch 为 conda-forge 版本。

**解决**: 本教程已通过调整安装顺序（先 pip PyTorch，再 --no-deps 装 mmcv）规避此问题。如果出现问题：
```bash
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 5. pip install mmcv 源码编译时 Disk quota exceeded

**现象**:
```
fatal error: when writing output to .../sparse_pool_ops_cuda.cpp1.ii: Disk quota exceeded
```

**原因**: nvcc 编译 CUDA 文件的临时输出远超 `/data` 分区配额。

**解决**: 根本不需要编译——直接用 conda-forge 预编译包。如果实在要编译：
```bash
export TMPDIR=/tmp  # /tmp 通常有 >500GB 空间
```

### 6. mmcv 源码编译报 `ATen/CollapseDims.h: No such file or directory`

**原因**: mmcv 2.1.0 不兼容 PyTorch 2.10+/2.12+。该头文件在新版 PyTorch 中已移除。

**解决**: 使用 mmcv 2.2.0 (conda-forge)，不要用 2.1.0。

### 7. `AssertionError: MMCV==2.2.0 is used but incompatible`

**原因**: mmdet 3.3.0 硬编码 `mmcv_maximum_version = '2.2.0'`。

**解决**: 修改 `mmdet/__init__.py`，将 `'2.2.0'` 改为 `'2.3.0'`（本项目已修改）。

### 8. 登录节点 vs 计算节点

登录节点没有 GPU (nvidia-smi 不可用，CUDA_VISIBLE_DEVICES 为空)。所有 GPU 相关测试必须通过 srun 在计算节点上运行：

```bash
# 交互式
srun --gpus=1 --pty bash

# 或直接运行命令
srun --gpus=1 bash -c 'conda activate mmdet_cu128 && python my_script.py'
```

### 9. `persistent_workers` 和 `num_workers=0` 冲突

**现象**:
```
ValueError: persistent_workers option needs num_workers > 0
```

**解决**: 如果设 `num_workers=0`（调试时），必须同时设 `persistent_workers=False`。

---

## 安装顺序总结

```
1. conda create -n mmdet_cu128 python=3.10
2. pip install torch torchvision torchaudio --index-url .../cu128    ← 先 PyTorch！
3. pip install nvidia-nccl-cu12==2.29.7 --force-reinstall --no-deps  ← 替换 NCCL
4. pip install nvidia-cuda-runtime-cu12                               ← libcudart.so.12
5. wget mmcv.conda + conda install --no-deps                          ← 预编译，不编译！
6. pip install mmengine opencv-python matplotlib numpy ...
7. 确认 mmdet/__init__.py (mmcv_maximum_version = '2.3.0')
8. pip install --no-build-isolation --no-deps -e .
9. srun --gpus=1 python test_gpu_quick.py                             ← GPU 节点验证
```

---

## 环境名

本教程使用 `mmdet_cu128` 以区分原有环境。已配置好的环境：

| 环境名 | PyTorch | mmcv | 适用节点 |
|--------|---------|------|---------|
| `mmdetection_para` | 旧版 | 旧版 | 视配置而定 |
| `mmdet_cu128` | 2.12.0 cu128 | 2.2.0 cuda129 | **CUDA 12.8 + 13.0** |

---

**编写日期**: 2026-06-03  
**更新日期**: 2026-06-09 — 适配 CUDA 12.8 节点，更新至 PyTorch 2.12.0  
**验证环境**: NVIDIA RTX 5090 (32GB) + CUDA Driver 580.82 (supports up to CUDA 13.0)  
**适用 GPU**: RTX 5090 / 5080 / 5070 等 Blackwell 架构 (sm_120)
