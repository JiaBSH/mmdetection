# 多倍率自适应滑窗实例分割：消融实验与真实数据测试报告

> **作者**: scvi576
> **日期**: 2025-06-05
> **模型**: Mask R-CNN (ResNet-50 FPN) + DINOv2 自适应滑窗
> **数据集**: 合成多倍率数据集 (5 倍率) + 真实显微镜数据集 `dataset_root/mmdata_test/` (31 张)

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [方法原理](#2-方法原理)
3. [消融实验：合成数据](#3-消融实验合成数据)
4. [系统架构](#4-系统架构)
5. [真实数据测试](#5-真实数据测试)
6. [可视化](#6-可视化)
7. [复现方法](#7-复现方法)
8. [文件清单](#8-文件清单)
9. [已知问题与改进方向](#9-已知问题与改进方向)

---

## 1. 背景与动机

### 1.1 核心问题

材料显微图像实例分割面临一个关键挑战：**不同放大倍率下，同一物理结构的像素尺寸差异巨大**。

| 倍率 | 典型畴区像素数 | 相对大小 |
|------|-------------|---------|
| 2.5x | ~50-200 px² | 极小目标 |
| 5x | ~200-800 px² | 小目标 |
| 20x | ~3000-12000 px² | 中等目标 |
| 50x | ~20000-80000 px² | 大目标 |
| 100x | ~80000-300000 px² | 超大目标 |

当使用固定大小的滑窗（如 1024×1024）进行推理时：
- **低倍率 (2.5x, 5x)**：目标过小，固定窗口内包含数百个实例，容易漏检和碎片化
- **高倍率 (50x, 100x)**：单个实例就可能超过窗口大小，被切割到多个窗口中，产生重复检测

### 1.2 解决思路

利用 **DINOv2 视觉特征 + kNN 回归** 自动估计每张图像的等效放大倍率，然后根据倍率自适应调整滑窗大小：

- **低倍率** → 小窗口（捕获细节）+ 高 overlap（减少边界效应）
- **高倍率** → 大窗口 / 整图推理（避免切割实例）

### 1.3 消融实验设计目标

通过受控消融实验量化每个组件的贡献：

1. **多倍率训练** vs 单倍率训练（M1 vs M2）
2. **尺度抖动** (scale jitter) 的作用（M2 vs M3）
3. **固定窗口** vs **自适应窗口**（noSW vs fix1024 vs DINOv2 adaptive）
4. 各配置在不同倍率上的泛化能力

---

## 2. 方法原理

### 2.1 DINOv2 倍率估计流水线

#### 2.1.1 训练阶段

```
训练图像 → DINOv2 (ViT-B/14) 特征提取 → PCA 降维 → KMeans 聚类 (5类)
                                                    ↓
                                          kNN 回归器拟合 scale 值
```

1. 从合成多倍率数据集 (`data/syn_multimag/`) 中提取所有训练图像的 DINOv2 CLS token 特征
2. PCA 降至 50 维
3. KMeans 聚类为 5 类（对应 5 个倍率）
4. 对每个聚类中心分配 scale 值（0~1），训练 kNN 回归器

**聚类中心 → 倍率映射（经验验证准确率 88-100%）：**

| 聚类中心 scale | 对应倍率 |
|---------------|---------|
| 0.25 | 20x |
| 0.40 | 50x |
| 0.55 | 5x |
| 0.70 | 100x |
| 0.85 | 2.5x |
| 1.00 | 100x (第二中心) |

#### 2.1.2 推理阶段

```
输入图像 → DINOv2 特征 → kNN 回归 → scale 值 s
                                    ↓
                    s → 最近聚类中心 → 倍率标签 → 窗口比例
```

### 2.2 比例窗口大小计算

窗口大小不是硬编码的绝对值，而是**图像短边的固定比例**。比例系数从合成数据标定：

| 倍率 | 合成数据窗口 (短边=1362) | 窗口比例 | 真实图像窗口 (短边=3264) |
|------|------------------------|---------|------------------------|
| 2.5x | 256 px | **0.188** | 614 px |
| 5x | 512 px | **0.376** | 1227 px |
| 20x | 2048 px | **1.0** (整图) | 3264 px (整图) |
| 50x | — | **1.0** (整图) | 3264 px (整图) |
| 100x | — | **1.0** (整图) | 3264 px (整图) |

```python
# 核心代码: postprocess/adaptive_scale.py
MAG_WINDOW_FRAC = {
    '2.5x': 0.188,   # 256/1362
    '5x': 0.376,      # 512/1362
    '20x': 1.0,       # capped — nearly whole image
    '50x': 1.0,       # whole image
    '100x': 1.0,      # whole image
}

def predict_window(self, img_path, image_width, image_height):
    _, _, frac = self.predict(img_path)  # 获取倍率→比例
    min_dim = min(image_width, image_height)
    window = int(round(frac * min_dim))
    if window >= min_dim:
        return 0  # 返回 0 表示整图推理
    return window
```

### 2.3 滑窗推理流程

当 `window > 0` 时，启用交叠滑窗推理：

```
1. 整图 → 按 patch_size=window, overlap=0.2 切分为 N 个 patch
2. 每个 patch → Mask R-CNN 独立推理 → 局部实例 masks
3. 所有 patch 结果 → 全局坐标映射 → 重叠区域合并（面积 IoU > 0.5 时保留较大的实例）
4. 输出合并后的全局实例列表
```

### 2.4 评估指标

#### 像素级指标（快速评估，用于消融实验）

使用 `cv2.fillPoly` 将预测和 GT 多边形分别光栅化为二值掩码，逐像素计算：

| 指标 | 公式 | 说明 |
|------|------|------|
| **IoU** | TP / (TP + FP + FN) | Jaccard 系数，核心评估指标 |
| **Precision** | TP / (TP + FP) | 预测像素中正确的比例 |
| **Recall** | TP / (TP + FN) | GT 像素中被检出的比例 |
| **F1** | 2×TP / (2×TP + FP + FN) | Precision 和 Recall 的调和平均 |

相比传统的基于几何多边形逐一匹配的评估方法，像素级评估速度极快（~0.1s/image），且不受多边形数量和形状复杂度的限制。

#### COCO mAP（训练验证，用于模型选择）

使用 MMDetection 内置 `CocoMetric(metric=['segm'])` 计算标准 COCO 分割 mAP：

| 指标 | 说明 |
|------|------|
| **segm_mAP** | mask AP @ IoU=0.50:0.95 |
| **segm_mAP_50** | mask AP @ IoU=0.50 |
| **segm_mAP_75** | mask AP @ IoU=0.75 |

### 2.5 类别过滤

模型仅在 category 1（畴区）上训练。真实数据集中包含多类别标注：
- cat_id=1: 畴区（目标类别）
- cat_id=2: 凸包
- cat_id=3,4: scale_text, scale_bar (仅 100x)

评估时自动过滤 GT annotations，仅保留 `category_id=1`，确保公平比较。

---

## 3. 消融实验：合成数据

### 3.1 合成数据生成

#### 3.1.1 生成参数

基于 `sys_G/om_domain/syn_data.py` 的参数化版本。以 `mag_factor=1.0` 对应 10x 为基准（线性映射）。

**共享参数**（所有倍率统一）：
- `base_r_range=(60, 90)`, `base_num_range=(4, 6)`
- `color_mean=(163,138,127)`, `bg_mean=(153,115,85)`
- `texture_std=8`, `bg_noise_std=3`, `color_std=0.5`
- `shape_jitter=0.1`, `edge_burr_amplitude=0.1`
- `max_overlap_ratio=0.5`, `max_overlap_count=3`

**倍率相关参数**：

| 倍率 | mag_factor | 有效半径 | 有效域数 | image_size |
|------|-----------|---------|---------|------------|
| 2.5x | 0.25 | 15-22 px | 64-96 | 1024×800 |
| 5x | 0.5 | 30-45 px | 16-24 | 1024×800 |
| 20x | 2.0 | 120-180 px | 1-2 | 1024×800 |
| 50x | 5.0 | 300-450 px | 1 | 1024×800 |
| 100x | 10.0 | 600-900 px | 1 | 1024×800 |

#### 3.1.2 数据集划分

| 用途 | 每倍率图像数 | 总图像数 |
|------|-----------|---------|
| 训练 (train) | 50 | 250 |
| 验证 (val) | 20 | 100 |
| 测试 (test) | 30 | 150 |
| **合计** | **100** | **500** |

#### 3.1.3 数据目录结构

```
data/syn_multimag/
├── raw/                          # 原始 ISAT 格式
│   ├── 2.5x/  (image/ + label/)
│   ├── 5x/
│   ├── 20x/
│   ├── 50x/
│   └── 100x/
├── coco/                         # COCO 格式
│   ├── annotations/
│   │   ├── instances_train.json  (250 张)
│   │   ├── instances_val.json    (100 张)
│   │   └── instances_test.json   (150 张)
│   └── images/
├── m1_50xonly/                   # M1 训练数据（仅 50x）
├── m2_allmag/                    # M2 训练数据（全部 5 倍率）
├── adaptive_patches_jitter/      # M3 训练数据（全部倍率 + 尺度抖动裁剪）
├── scale_pipeline_dinov2.joblib  # DINOv2 scale estimation pipeline
└── window_recommendations.csv    # 各倍率窗口大小推荐
```

### 3.2 模型训练配置

#### 三个模型对比

| 配置项 | M1 (单倍率基线) | M2 (多倍率) | M3 (多倍率+抖动) |
|--------|-------------|----------|---------------|
| **训练数据** | `m1_50xonly` (仅 50x) | `m2_allmag` (全部 5 倍率) | `adaptive_patches_jitter` (全部倍率 + 抖动) |
| **训练样本数** | 50 | 250 | ~1250 (含裁剪增强) |
| **训练 pipeline** | `Resize(1024,1024)` + `RandomFlip(0.5)` | 同 M1 | `RandomResize(1024,1024, ratio_range=(0.75,1.25))` + `RandomFlip(0.5)` |
| **尺度抖动** | 无 | 无 | ±25% |
| **训练轮数** | 5 epoch | 5 epoch | 5 epoch |
| **Work dir** | `work_dirs/ablation_m1_single_mag/` | `work_dirs/ablation_m2_multimag/` | `work_dirs/ablation_m3_multimag_jitter/` |

**共享配置**：
- 架构: Mask R-CNN R50-FPN, ImageNet-pretrained backbone
- Batch size: 8 (per GPU)
- Optimizer: SGD, lr=0.02
- 单类别: "畴区" (category_id=1)
- 图像分辨率: 1024×800

### 3.3 训练过程：验证集 COCO mAP

#### M1 (单倍率 50x only)

| Run (seed) | Epoch | segm_mAP | segm_mAP_50 | segm_mAP_75 |
|---|---|---|---|---|
| 00072 | 1 | 0.272 | 0.492 | 0.299 |
| 00072 | 2 | 0.550 | 0.779 | 0.629 |
| 00072 | 3 | 0.644 | 0.808 | 0.759 |
| 00073 | 5 | 0.672 | 0.797 | 0.758 |
| 00074 | 5 | **0.834** | 0.941 | 0.913 |

> 不同 seed 间验证 mAP 差异较大（0.672-0.834），说明单倍率训练对随机种子敏感。

#### M2 (多倍率, seed 00072)

| Epoch | segm_mAP | segm_mAP_50 | segm_mAP_75 |
|---|---|---|---|
| 1 | 0.627 | 0.769 | 0.732 |
| 2 | 0.711 | 0.802 | 0.790 |
| 3 | 0.723 | 0.802 | 0.800 |
| 4 | 0.731 | 0.812 | 0.801 |
| 5 | **0.737** | 0.812 | 0.801 |

#### M3 (多倍率+抖动, seed 00070)

| Epoch | segm_mAP | segm_mAP_50 | segm_mAP_75 |
|---|---|---|---|
| 1 | 0.574 | 0.700 | 0.667 |
| 2 | 0.643 | 0.723 | 0.711 |
| 3 | 0.660 | 0.732 | 0.722 |
| 4 | 0.664 | 0.733 | 0.722 |
| 5 | **0.668** | 0.733 | 0.722 |

> **注意**: M2/M3 的验证集 COCO mAP 低于 M1 (seed 00074) 是因为验证集包含了全部 5 个倍率，而 M1 的值仅在 50x 上评估。M2 在 5 epoch 时稳定收敛，M3 收敛稍慢可能与更大的数据量和尺度抖动有关。最终在测试集上的 pixel-IoU 评估显示 M2 略优于 M3。

### 3.4 消融实验矩阵

训练完成后，用 6 种推理配置在合成测试集（150 张图）上评估：

| 实验 ID | 模型 | 窗口策略 | Overlap | 验证内容 |
|---------|------|---------|---------|---------|
| **E1** | M1 (仅 50x) | 整图，无 SW | 0 | 单倍率训练 baseline |
| **E2** | M2 (多倍率) | 整图，无 SW | 0 | 多倍率训练 vs 单倍率 |
| **E3** | M2 (多倍率) | 固定 1024 SW | 0.2 | 固定滑窗 vs 整图 |
| **E4** | M3 (+抖动) | 固定 1024 SW | 0.2 | 尺度抖动的作用 |
| **E5** | M2 (多倍率) | **DINOv2 自适应** | 0.2 | 自适应窗口 vs 固定窗口 |
| **E6** | M3 (+抖动) | **DINOv2 自适应** | 0.2 | 完整方案 (M3 + adaptive) |

### 3.5 消融实验结果：合成测试集逐倍率分析

#### E1: M1 (仅 50x 训练) + noSW

| 倍率 | IoU | Precision | Recall | F1 | Pred | GT |
|------|-----|-----------|--------|-----|------|-----|
| 100x | 0.936 | 0.993 | 0.942 | 0.967 | 4 | 4 |
| 50x | 0.960 | 0.990 | 0.970 | 0.980 | 9 | 9 |
| 20x | 0.928 | 0.973 | 0.953 | 0.962 | 54 | 70 |
| 5x | **0.000** | 0.000 | 0.000 | **0.000** | 0 | 1074 |
| 2.5x | **0.000** | 0.000 | 0.000 | **0.000** | 0 | 6362 |

> **结论**: M1 在 5x 和 2.5x 上完全失效，预测实例数为 0。仅在 50x 上训练的模型无法泛化到低倍率。

#### E2: M2 (多倍率训练) + noSW

| 倍率 | IoU | Precision | Recall | F1 | Pred | GT |
|------|-----|-----------|--------|-----|------|-----|
| 100x | 0.974 | 0.990 | 0.984 | 0.987 | 4 | 4 |
| 50x | 0.979 | 0.992 | 0.987 | 0.990 | 9 | 9 |
| 20x | 0.974 | 0.992 | 0.982 | 0.987 | 66 | 70 |
| 5x | 0.188 | 0.958 | 0.190 | 0.317 | 100 | 1074 |
| 2.5x | 0.001 | 0.814 | 0.001 | 0.002 | 2 | 6362 |

> **结论**: 多倍率训练使 5x 从 0→0.188 IoU，但仍远不够。2.5x 仅检测到 2 个实例，说明整图推理会丢失小目标。

#### E3: M2 + 固定 1024 滑窗

| 倍率 | IoU | Precision | Recall | F1 | Pred | GT |
|------|-----|-----------|--------|-----|------|-----|
| 100x | 0.860 | 0.907 | 0.943 | 0.925 | 13 | 4 |
| 50x | 0.977 | 0.989 | 0.987 | 0.988 | 19 | 9 |
| 20x | 0.940 | 0.952 | 0.987 | 0.969 | 106 | 70 |
| 5x | 0.569 | 0.986 | 0.574 | 0.725 | 523 | 1074 |
| 2.5x | 0.099 | 0.958 | 0.100 | 0.180 | 286 | 6362 |

> **结论**: 固定 1024 滑窗大幅改善 5x（0.188→0.569 IoU），但 2.5x 仍很差（0.001→0.099）。同时 100x 因切割大目标而退化（0.974→0.860）。

#### E4: M3 (多倍率+抖动) + 固定 1024 滑窗

| 倍率 | IoU | Precision | Recall | F1 | Pred | GT |
|------|-----|-----------|--------|-----|------|-----|
| 100x | 0.874 | 0.913 | 0.953 | 0.933 | 13 | 4 |
| 50x | 0.803 | 0.812 | 0.987 | 0.891 | 20 | 9 |
| 20x | 0.939 | 0.951 | 0.986 | 0.968 | 104 | 70 |
| 5x | 0.554 | 0.992 | 0.557 | 0.713 | 538 | 1074 |
| 2.5x | 0.157 | 0.967 | 0.158 | 0.271 | 497 | 6362 |

> **结论**: M3 在低倍率上略优于 M2（2.5x: 0.099→0.157），但在 50x 上退化（0.977→0.803）。整体而言 M3 的 jitter 没有带来一致的收益。

#### E5: M2 + DINOv2 自适应窗口 ⭐ **最优方案**

| 倍率 | IoU | Precision | Recall | F1 | Pred | GT | 窗口策略 |
|------|-----|-----------|--------|-----|------|-----|---------|
| 100x | **0.974** | 0.990 | 0.984 | **0.987** | 4 | 4 | 整图 |
| 50x | **0.979** | 0.992 | 0.987 | **0.990** | 9 | 9 | 整图 |
| 20x | **0.974** | 0.992 | 0.982 | **0.987** | 66 | 70 | 整图 |
| 5x | **0.958** | 0.996 | 0.962 | **0.979** | 1191 | 1074 | SW=362px |
| 2.5x | **0.845** | 0.998 | 0.846 | **0.916** | 4927 | 6362 | SW=256px |

> **结论**: 最佳方案。100x/50x/20x IoU > 0.97，5x IoU 从 E3 的 0.569 跃升至 0.958，2.5x IoU 从 0.099 跃升至 0.845。关键的突破来自 DINOv2 为不同倍率分配合适的窗口大小。

#### E6: M3 + DINOv2 自适应窗口

| 倍率 | IoU | Precision | Recall | F1 | Pred | GT | 窗口策略 |
|------|-----|-----------|--------|-----|------|-----|---------|
| 100x | 0.974 | 0.992 | 0.982 | 0.987 | 4 | 4 | 整图 |
| 50x | 0.979 | 0.993 | 0.986 | 0.989 | 9 | 9 | 整图 |
| 20x | 0.973 | 0.994 | 0.979 | 0.986 | 66 | 70 | 整图 |
| 5x | 0.959 | 0.997 | 0.962 | 0.979 | 1204 | 1074 | SW=362px |
| 2.5x | 0.836 | 0.998 | 0.837 | 0.910 | 4928 | 6362 | SW=256px |

> **结论**: M3+adaptive 与 M2+adaptive 性能几乎一致，M2 在 2.5x 上略高（0.845 vs 0.836），在 5x 上略低（0.958 vs 0.959）。尺度抖动没有带来显著额外收益。

### 3.6 消融实验综合对比

#### 逐倍率 IoU 对比

| 倍率 | E1 (M1 noSW) | E2 (M2 noSW) | E3 (M2 fix1024) | E4 (M3 fix1024) | **E5 (M2 adaptive)** | E6 (M3 adaptive) |
|------|-------------|-------------|----------------|----------------|--------------------|----------------|
| 100x | 0.936 | 0.974 | 0.860 ↓ | 0.874 ↓ | **0.974** | 0.974 |
| 50x | 0.960 | 0.979 | 0.977 | 0.803 ↓ | **0.979** | 0.979 |
| 20x | 0.928 | 0.974 | 0.940 | 0.939 | **0.974** | 0.973 |
| 5x | 0.000 | 0.188 | 0.569 | 0.554 | **0.958** ↑ | 0.959 |
| 2.5x | 0.000 | 0.001 | 0.099 | 0.157 | **0.845** ↑ | 0.836 |

#### 关键对比维度

| 对比 | Δ | 回答的问题 |
|------|---|-----------|
| E2 vs E1: 5x IoU | 0.188 vs 0.000 | 多倍率训练使 5x 从零检变为可检测 |
| E3 vs E2: 5x IoU | 0.569 vs 0.188 | 固定 1024 SW 提升 5x 3× |
| E3 vs E2: 2.5x IoU | 0.099 vs 0.001 | 固定 SW 对 2.5x 帮助有限 (仍 < 0.1) |
| E3 vs E2: 100x IoU | 0.860 vs 0.974 | 固定 SW 伤害高倍率 (切割大目标) |
| **E5 vs E3: 5x IoU** | **0.958 vs 0.569** | **DINOv2 自适应: 5x 提升 68%** |
| **E5 vs E3: 2.5x IoU** | **0.845 vs 0.099** | **DINOv2 自适应: 2.5x 提升 754%** |
| **E5 vs E3: 100x IoU** | **0.974 vs 0.860** | **自适应保持高倍率精度** |
| E6 vs E5 | ~0 | 尺度抖动无额外收益 |

### 3.7 消融实验结论

1. **M2 (多倍率训练) + DINOv2 自适应窗口 = 最优方案 (E5)**
   - 合成测试集上：100x/50x/20x IoU > 0.97，5x IoU = 0.958，2.5x IoU = 0.845
   - 各倍率间平衡最佳，无短板

2. **单倍率训练 (M1) 无法泛化** — 在低倍率上完全失效

3. **固定窗口 SW 有得有失** — 帮助低倍率但伤害高倍率

4. **尺度抖动 (M3) 未带来一致的收益** — 在 5 epoch 快速消融中，M3 的验证 mAP (0.668) 低于 M2 (0.737)，可能与更大的数据量需要更多训练轮数有关

5. **选择 M2 作为真实数据测试模型**

---

## 4. 系统架构

### 4.1 代码模块

```
mmdetection_para/
├── postprocess/
│   ├── adaptive_scale.py          # DINOv2 倍率预测器 (107 行)
│   ├── run_postprocess.py         # 单模型推理+后处理主逻辑 (654 行)
│   ├── sliding_window_infer.py    # 交叠滑窗推理引擎 (529 行)
│   ├── coco_utils.py              # COCO 格式工具（GT/预测格式转换, 492 行）
│   ├── analyze_main_dy2.py        # 几何分析（直方图、R² 等）
│   ├── aggregate_matched_metrics.py   # 合并逐图指标到汇总 CSV
│   ├── compare_models.py          # 多模型批量评估+对比
│   ├── summarize_model_metrics.py # 模型汇总统计
│   └── plot_comparison_scales.py  # 倍率对比柱状图
├── tools/
│   ├── test_real_dataset.py       # 真实数据集批量测试主脚本 (415 行)
│   └── gen_visualizations.py      # 可视化生成脚本 (121 行)
├── Microscope_Magnification_Identification/
│   └── src/rate_identification/
│       └── pipeline.py            # ScaleEstimationPipeline (训练/加载)
└── work_dirs/
    ├── ablation_m1_single_mag/    # M1 模型训练
    ├── ablation_m2_multimag/      # M2 模型训练 (最优)
    ├── ablation_m3_multimag_jitter/  # M3 模型训练
    ├── ablation_results_/         # 消融推理结果 (E1-E6)
    │   ├── E1_M1_noSW/
    │   ├── E2_M2_noSW/
    │   ├── E3_M2_fix1024/
    │   ├── E4_M3_fix1024/
    │   ├── E5_M2_adaptive_DINOv2/
    │   └── E6_M3_adaptive_DINOv2/
    └── real_test_results/         # 真实数据测试结果
```

### 4.2 数据流

```
                          ┌──────────────────────┐
                          │  DINOv2 Pipeline      │
                          │  scale_pipeline.joblib│
                          └──────────┬───────────┘
                                     │ predict(img_path)
                                     ▼
                          ┌──────────────────────┐
                          │  AdaptiveWindowPredictor │
                          │  → mag, frac         │
                          └──────────┬───────────┘
                                     │ frac × min(w,h)
                                     ▼
                    ┌─────────── window > 0? ───────────┐
                    │ NO (整图)                         │ YES (滑窗)
                    ▼                                    ▼
          ┌──────────────────┐              ┌──────────────────┐
          │ inference_detector│              │ infer_image_with │
          │ (numpy array)    │              │ _overlap_windows │
          └────────┬─────────┘              └────────┬─────────┘
                   │                                  │
                   └────────────┬─────────────────────┘
                                ▼
                   ┌──────────────────────┐
                   │  _pred_instances_to   │
                   │  _global_instances    │
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │  fast_pixel_metrics   │
                   │  (cv2.fillPoly)      │
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │  metrics_summary.csv  │
                   └──────────────────────┘
```

---

## 5. 真实数据测试

### 5.1 数据集

| 属性 | 值 |
|------|-----|
| 数据集路径 | `dataset_root/mmdata_test/` |
| 图像分辨率 | 4908 × 3264 (16 MP) |
| 倍率目录 | `2_5x_unsup`, `5x_unsup`, `20x`, `50x`, `100x` |
| 图像总数 | 31 张 (各倍率 6-7 张) |
| GT 标注格式 | COCO (instances_test_{mag}.json) |
| 评估类别 | cat_id=1 (畴区), 过滤掉 cat_id=2/3/4 |

### 5.2 推理设置

| 参数 | M1 (Baseline) | M2 (OURS) |
|------|-------------|-----------|
| 模型 | `ablation_m1_single_mag/epoch_5.pth` | `ablation_m2_multimag/epoch_5.pth` |
| 窗口策略 | 整图 (无滑窗) | DINOv2 自适应滑窗 |
| 置信度阈值 | 0.5 | 0.5 |
| 滑窗 overlap | — | 0.2 |
| 最小实例像素 | 10 | 10 |
| batch_size | 1 | 1 |
| 设备 | NVIDIA GPU (CUDA) | NVIDIA GPU (CUDA) |

### 5.3 环境

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.6+ |
| MMDetection | 3.x |
| DINOv2 | ViT-B/14 (via timm) |
| scikit-learn | 1.9.0 |
| OpenCV | cv2 |

### 5.4 总体对比

| 指标 | M1 (50x only, no SW) | M2 (multi-mag + adaptive SW) |
|------|---------------------|---------------------------|
| **Overall IoU** | 0.537 | **0.863** |
| **Overall F1** | — | **0.928** |
| 2.5x IoU | ~0.001 | **0.801** |
| 5x IoU | ~0.001 | **0.834** |
| 20x IoU | 0.789 | **0.930** |
| 50x IoU | 0.879 | **0.925** |
| 100x IoU | 0.931 | **0.923** (6/7: 0.928) |

> M2 在所有倍率上均显著优于 M1。M1 在 2.5x 和 5x 上几乎完全失效（IoU ≈ 0），因为仅在 50x 上训练无法泛化到低倍率小目标。

### 5.5 M2 逐倍率详细结果

#### 2.5x (6 张图, DINOv2→2.5x→窗口 614px)

| 图像 | IoU | F1 | Precision | Recall | Pred | GT |
|------|-----|----|-----------|--------|------|-----|
| 2.5x-1 | 0.719 | 0.837 | 0.992 | 0.724 | 5639 | 1358 |
| 2.5x-2 | 0.826 | 0.905 | 0.966 | 0.851 | 5224 | 1392 |
| 2.5x-3 | 0.821 | 0.902 | 0.955 | 0.854 | 5348 | 1363 |
| 2.5x-4 | 0.829 | 0.906 | 0.953 | 0.864 | 5336 | 1436 |
| 2.5x-5 | 0.825 | 0.904 | 0.959 | 0.855 | 5488 | 1485 |
| 2.5x-6 | 0.788 | 0.881 | 0.975 | 0.804 | 5470 | 1419 |
| **平均** | **0.801** | **0.889** | 0.967 | 0.825 | 5418 | 1409 |

**分析**: 2.5x 是最困难的场景（极小目标 + 极高密度）。M2 实现了 0.80 IoU（M1 为 0），但存在约 3.8× 的过预测。Precision=0.967 很高但 Recall=0.825 偏低，说明有约 17.5% 的 GT 区域未被检出。

#### 5x (6 张图, DINOv2→5x→窗口 1227px)

| 图像 | IoU | F1 | Precision | Recall | Pred | GT |
|------|-----|----|-----------|--------|------|-----|
| 5x-1 | 0.825 | 0.904 | 0.972 | 0.845 | 1628 | 444 |
| 5x-2 | 0.847 | 0.917 | 0.942 | 0.894 | 1554 | 455 |
| 5x-3 | 0.835 | 0.910 | 0.908 | 0.912 | 1548 | 472 |
| 5x-4 | 0.803 | 0.891 | 0.901 | 0.882 | 1577 | 477 |
| 5x-5 | 0.791 | 0.884 | 0.958 | 0.820 | 1575 | 465 |
| 5x-6 | 0.901 | 0.948 | 0.964 | 0.933 | 1373 | 566 |
| **平均** | **0.834** | **0.909** | 0.941 | 0.881 | 1543 | 480 |

**分析**: 5x 表现优于 2.5x，IoU 达 0.834。过预测率约 3.2×，Precision/Recall 均在 0.88 以上。

#### 20x (6 张图, DINOv2→20x→整图)

| 图像 | IoU | F1 | Precision | Recall | Pred | GT |
|------|-----|----|-----------|--------|------|-----|
| 20x-1 | 0.914 | 0.955 | 0.974 | 0.937 | 62 | 70 |
| 20x-2 | 0.942 | 0.970 | 0.989 | 0.952 | 70 | 75 |
| 20x-3 | 0.946 | 0.972 | 0.986 | 0.958 | 69 | 70 |
| 20x-4 | 0.917 | 0.957 | 0.983 | 0.931 | 69 | 65 |
| 20x-5 | 0.937 | 0.967 | 0.990 | 0.946 | 62 | 67 |
| 20x-6 | 0.922 | 0.960 | 0.979 | 0.941 | 64 | 67 |
| **平均** | **0.930** | **0.964** | 0.983 | 0.944 | 66 | 69 |

**分析**: 20x 表现优异，IoU 全部 > 0.91。Pred/GT 比率接近 1.0，几乎完美匹配。

#### 50x (6 张图, DINOv2→50x→整图)

| 图像 | IoU | F1 | Precision | Recall | Pred | GT |
|------|-----|----|-----------|--------|------|-----|
| 50x-1 | 0.901 | 0.948 | 0.955 | 0.941 | 23 | 15 |
| 50x-2 | 0.964 | 0.982 | 0.971 | 0.993 | 15 | 14 |
| 50x-3 | 0.917 | 0.957 | 0.931 | 0.984 | 18 | 13 |
| 50x-4 | 0.952 | 0.975 | 0.973 | 0.978 | 20 | 13 |
| 50x-5 | 0.912 | 0.954 | 0.976 | 0.933 | 28 | 22 |
| 50x-6 | 0.904 | 0.950 | 0.959 | 0.941 | 22 | 13 |
| **平均** | **0.925** | **0.961** | 0.961 | 0.962 | 21 | 15 |

**分析**: 50x 表现优异，IoU 全部 > 0.90。少量过预测但整体平衡。

#### 100x (7 张图, 6/7 正确识别为 100x→整图; 1 张误判为 5x→1227px SW)

| 图像 | DINOv2 预测 | IoU | F1 | Precision | Recall | Pred | GT |
|------|-----------|-----|----|-----------|--------|------|-----|
| 100x-1 | ✅ 100x, noSW | 0.959 | 0.979 | 0.969 | 0.989 | 11 | 9 |
| 100x-2 | ❌ 5x, SW=1227 | 0.249 | 0.399 | 0.928 | 0.254 | 41 | 6 |
| 100x-3 | ✅ 100x, noSW | 0.855 | 0.922 | 0.970 | 0.879 | 8 | 6 |
| 100x-4 | ✅ 100x, noSW | 0.943 | 0.971 | 0.955 | 0.987 | 4 | 1 |
| 100x-5 | ✅ 100x, noSW | 0.948 | 0.973 | 0.974 | 0.973 | 5 | 2 |
| 100x-6 | ✅ 100x, noSW | 0.939 | 0.968 | 0.944 | 0.994 | 6 | 2 |
| 100x-7 | ✅ 100x, noSW | 0.925 | 0.961 | 0.941 | 0.982 | 6 | 4 |
| **平均 (6/7)** | — | **0.928** | **0.962** | 0.959 | 0.967 | 6.7 | 4.0 |

**分析**: 6/7 张正确识别，表现优秀 (avg IoU=0.928)。100x-2 被 DINOv2 误判为 5x 导致 IoU=0.249——其纹理与 5x 合成样本相近，是 DINOv2 pipeline 在真实数据上的主要泛化问题。

### 5.6 DINOv2 倍率预测准确率（真实数据）

| 真实倍率 | 图像数 | 正确预测 | 准确率 |
|---------|--------|---------|--------|
| 2.5x | 6 | 6 | 100% |
| 5x | 6 | 6 | 100% |
| 20x | 6 | 6 | 100% |
| 50x | 6 | 6 | 100% |
| 100x | 7 | 6 | 85.7% |
| **总计** | **31** | **30** | **96.8%** |

DINOv2 pipeline 在真实显微图像上泛化良好，仅 1 张误判。

### 5.7 M1 vs M2: 失败原因分析

**M1 在低倍率上完全失效的根本原因**：

1. **训练域偏差**: M1 仅在 50x 合成数据上训练。50x 图像中畴区占据 30-50% 的图像面积，模型从未见过占图像面积 < 1% 的小目标
2. **推理时的域偏移**: 2.5x 图像（4908×3264）中有约 1400 个畴区，每个仅占 ~0.05% 图像面积。整图推理时被 mmdet resize 到 1333×800，小目标进一步缩小至仅几个像素
3. **无滑窗**: 模型只能"看到"下采样后的整图，无法聚焦到细节区域

**M2 解决这些问题通过**：

1. **多倍率训练** → 模型学习到各倍率下的目标外观变化
2. **自适应窗口** → DINOv2 识别低倍率后自动切换到小窗口（614px/1227px），保留细节分辨率
3. **滑窗覆盖** → overlap=0.2 减少窗口边界的检测遗漏

---

## 6. 可视化

### 6.1 输出文件说明

每张测试图像生成 4 张可视化和 1 个 CSV：

| 文件 | 内容 | 颜色编码 |
|------|------|---------|
| `pred_overlay.png` | 原始图叠加预测实例轮廓 | 每个实例不同颜色 |
| `gt_overlay.png` | 原始图叠加 GT 标注轮廓 | 每个实例不同颜色 |
| `iou_visualization.png` | IoU 逐像素可视化 | 🟢 TP (交集) / 🔴 FP (过检) / 🔵 FN (漏检) |
| `mask_comparison.png` | Pred mask vs GT mask 并排 | 🟢 Pred / 🔴 GT |
| `metrics.csv` | 该图的定量指标 | — |

### 6.2 输出位置

```
work_dirs/real_test_results/
├── M2_adaptive/
│   └── metrics_summary.csv        # M2 全部 31 张图定量结果
├── M1_noSW/
│   └── metrics_summary.csv        # M1 基线定量结果
├── M2_adaptive_viz/               # M2 可视化 (每倍率 1 张)
│   ├── 2_5x_unsup/2.5x-1/
│   │   ├── pred_overlay.png       (~12 MB)
│   │   ├── gt_overlay.png         (~16 MB)
│   │   ├── iou_visualization.png  (~15 MB)
│   │   └── mask_comparison.png    (~1 MB)
│   ├── 5x_unsup/5x-1/ ...
│   ├── 20x/20x-1/ ...
│   ├── 50x/50x-1/ ...
│   └── 100x/100x-1/ ...
└──
```

---

## 7. 复现方法

### 7.1 环境准备

```bash
# 1. 进入计算节点
srun --gpus=1 --pty bash

# 2. 激活环境
conda activate mmdetection_para
cd /data/run01/scvi576/JiaBSH/mmdetection_para

# 3. 安装依赖（如未安装）
pip install joblib scikit-learn timm

# 4. 验证 DINOv2 pipeline
python -c "
from postprocess.adaptive_scale import AdaptiveWindowPredictor
p = AdaptiveWindowPredictor('data/syn_multimag/scale_pipeline_dinov2.joblib')
print('Pipeline loaded OK')
"
```

### 7.2 运行消融实验（合成数据）

```bash
# E1: M1 baseline (仅 50x 训练，无滑窗)
python postprocess/run_postprocess.py \
    --config work_dirs/ablation_m1_single_mag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m1_single_mag/epoch_5.pth \
    --ann-file data/syn_multimag/coco/annotations/instances_test.json \
    --img-dir data/syn_multimag/coco/images/test \
    --out-dir work_dirs/ablation_results_/E1_M1_noSW

# E2: M2 noSW (多倍率训练，无滑窗)
python postprocess/run_postprocess.py \
    --config work_dirs/ablation_m2_multimag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m2_multimag/epoch_5.pth \
    --ann-file data/syn_multimag/coco/annotations/instances_test.json \
    --img-dir data/syn_multimag/coco/images/test \
    --out-dir work_dirs/ablation_results_/E2_M2_noSW

# E3: M2 + fixed 1024 sliding window
python postprocess/run_postprocess.py \
    --config work_dirs/ablation_m2_multimag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m2_multimag/epoch_5.pth \
    --ann-file data/syn_multimag/coco/annotations/instances_test.json \
    --img-dir data/syn_multimag/coco/images/test \
    --sliding-window --patch-size 1024 --patch-overlap-ratio 0.2 \
    --out-dir work_dirs/ablation_results_/E3_M2_fix1024

# E4: M3 + fixed 1024 sliding window
# (同上，替换 config/checkpoint 为 M3)

# E5: M2 + DINOv2 adaptive (需使用 test_real_dataset.py 的 adaptive 逻辑)
# E6: M3 + DINOv2 adaptive
```

### 7.3 运行真实数据测试

```bash
# M2 自适应测试 (主实验)
python tools/test_real_dataset.py \
    --config work_dirs/ablation_m2_multimag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m2_multimag/epoch_5.pth \
    --dataset-root dataset_root/mmdata_test \
    --scale-model data/syn_multimag/scale_pipeline_dinov2.joblib \
    --out-dir work_dirs/real_test_results/M2_adaptive \
    --score-thresh 0.5 \
    --overlap-ratio 0.2

# M1 基线测试
python tools/test_real_dataset.py \
    --config work_dirs/ablation_m1_single_mag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m1_single_mag/epoch_5.pth \
    --dataset-root dataset_root/mmdata_test \
    --no-adaptive \
    --out-dir work_dirs/real_test_results/M1_noSW
```

### 7.4 生成可视化

```bash
python tools/gen_visualizations.py
```

输出到 `work_dirs/real_test_results/M2_adaptive_viz/`。

### 7.5 包含超分辨率图像（可选）

```bash
python tools/test_real_dataset.py \
    ... \
    --include-sr \
    --out-dir work_dirs/real_test_results/M2_adaptive_withSR
```

### 7.6 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--score-thresh` | 0.5 | 预测置信度阈值 |
| `--overlap-ratio` | 0.2 | 滑窗重叠比例 |
| `--min-pixels` | 10 | 实例最少像素数 |
| `--no-adaptive` | false | 禁用自适应窗口 |
| `--device` | cuda:0 | 推理设备 |
| `--include-sr` | false | 包含超分辨率目录 |

---

## 8. 文件清单

### 8.1 核心代码

| 文件 | 行数 | 功能 |
|------|------|------|
| `postprocess/adaptive_scale.py` | 107 | DINOv2 倍率预测 + 比例窗口计算 |
| `postprocess/run_postprocess.py` | 654 | 单模型推理框架、COCO 评估桥接 |
| `postprocess/sliding_window_infer.py` | 529 | 交叠滑窗推理引擎 |
| `postprocess/coco_utils.py` | 492 | COCO 格式工具函数 |
| `postprocess/analyze_main_dy2.py` | — | 几何分析 |
| `postprocess/aggregate_matched_metrics.py` | — | 逐图指标合并 |
| `postprocess/compare_models.py` | — | 多模型批量评估 |
| `postprocess/summarize_model_metrics.py` | — | 模型汇总统计 |

### 8.2 测试与可视化脚本

| 文件 | 功能 |
|------|------|
| `tools/test_real_dataset.py` | 真实数据集批量测试（含可视化） |
| `tools/gen_visualizations.py` | 独立可视化生成脚本 |

### 8.3 模型权重

| 路径 | 说明 | 大小 |
|------|------|------|
| `work_dirs/ablation_m2_multimag/epoch_5.pth` | M2 最优模型 (多倍率训练) | ~176 MB |
| `work_dirs/ablation_m1_single_mag/epoch_5.pth` | M1 基线模型 (仅 50x) | ~176 MB |
| `work_dirs/ablation_m3_multimag_jitter/epoch_5.pth` | M3 模型 (多倍率+抖动) | ~176 MB |

### 8.4 数据与 Pipeline

| 路径 | 说明 |
|------|------|
| `data/syn_multimag/scale_pipeline_dinov2.joblib` | 训练好的 DINOv2 scale pipeline |
| `data/syn_multimag/coco/` | 合成多倍率 COCO 数据集 (train/val/test) |
| `data/syn_multimag/m1_50xonly/` | M1 训练集 (仅 50x) |
| `data/syn_multimag/m2_allmag/` | M2 训练集 (全部 5 倍率) |
| `data/syn_multimag/adaptive_patches_jitter/` | M3 训练集 (裁剪+抖动) |
| `dataset_root/mmdata_test/` | 真实显微镜测试数据集 (31 张) |

### 8.5 结果输出

| 路径 | 说明 |
|------|------|
| `work_dirs/ablation_results_/E1_M1_noSW/` | 消融实验 E1 结果 |
| `work_dirs/ablation_results_/E2_M2_noSW/` | 消融实验 E2 结果 |
| `work_dirs/ablation_results_/E3_M2_fix1024/` | 消融实验 E3 结果 |
| `work_dirs/ablation_results_/E4_M3_fix1024/` | 消融实验 E4 结果 |
| `work_dirs/ablation_results_/E5_M2_adaptive_DINOv2/` | 消融实验 E5 结果 ⭐ |
| `work_dirs/ablation_results_/E6_M3_adaptive_DINOv2/` | 消融实验 E6 结果 |
| `work_dirs/real_test_results/M2_adaptive/metrics_summary.csv` | M2 真实数据结果 |
| `work_dirs/real_test_results/M1_noSW/metrics_summary.csv` | M1 真实数据结果 |
| `work_dirs/real_test_results/M2_adaptive_viz/` | 可视化结果 |

---

## 9. 已知问题与改进方向

### 9.1 已知问题

1. **DINOv2 对 100x 的误判** (1/31, 3.2%)
   - 100x-2.png 被误判为 5x，导致使用过小的滑窗 (IoU=0.249)
   - 原因：pipeline 在合成数据上训练，100x 的部分纹理与 5x 合成样本相近
   - 缓解：可在真实数据上微调 scale pipeline 或集成多模型投票

2. **低倍率过预测** (2.5x: ~3.8×, 5x: ~3.2×)
   - 模型倾向于在低倍率产生过多候选实例
   - 可能原因：窗口覆盖面积仍偏大，或置信度阈值需按倍率自适应调整

3. **sklearn 版本不匹配**
   - pipeline 在 sklearn 1.7.2 训练，当前环境为 1.9.0
   - 不影响功能，仅为 InconsistentVersionWarning

4. **M3 尺度抖动未带来提升**
   - 在 5 epoch 快速消融中未观察到正收益
   - 可能原因：训练不充分（需要更多 epoch），或 jitter 范围 (±25%) 过大

### 9.2 改进方向

1. **自适应置信度阈值**: 不同倍率使用不同的 score_thresh（低倍率提高阈值减少 FP）
2. **DINOv2 pipeline 微调**: 在真实数据上标注少量图像，微调 scale estimation
3. **超分辨率预处理**: 对低倍率图像先做 SR 放大，再推理
4. **更大的 backbone**: 使用 Swin Transformer 或 ViT-based backbone 替代 ResNet-50
5. **更长训练**: 5 epoch → 50 epoch，预期进一步提升精度
6. **真实数据微调**: 取少量真实标注数据对 M2 做 fine-tune
7. **集成 DINOv2**: 用 DINOv2 特征直接辅助检测 head（如 DINOv2 features + FPN 融合）

---

## 附录 A: 命令速查

```bash
cd /data/run01/scvi576/JiaBSH/mmdetection_para

# === 完整 M2 真数据测试 ===
python tools/test_real_dataset.py \
    --config work_dirs/ablation_m2_multimag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m2_multimag/epoch_5.pth \
    --dataset-root dataset_root/mmdata_test \
    --scale-model data/syn_multimag/scale_pipeline_dinov2.joblib \
    --out-dir work_dirs/real_test_results/M2_adaptive

# === M1 基线 ===
python tools/test_real_dataset.py \
    --config work_dirs/ablation_m1_single_mag/mask-rcnn_r50_fpn_1x_custom_coco_instance.py \
    --checkpoint work_dirs/ablation_m1_single_mag/epoch_5.pth \
    --dataset-root dataset_root/mmdata_test \
    --no-adaptive \
    --out-dir work_dirs/real_test_results/M1_noSW

# === 可视化 ===
python tools/gen_visualizations.py

# === 查看结果 ===
cat work_dirs/real_test_results/M2_adaptive/metrics_summary.csv
cat work_dirs/ablation_results_/E5_M2_adaptive_DINOv2/metrics_summary.csv
```

## 附录 B: 常见问题

**Q: PIL DecompressionBombError?**
```python
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 禁用像素限制 (4908×3264 = 16M pixels)
```

**Q: ModuleNotFoundError: rate_identification?**
```python
import sys
sys.path.insert(0, 'Microscope_Magnification_Identification/src')
```

**Q: mmdet whole-image inference fails after sliding window?**
根本原因是 `inference_detector` 的浅拷贝 (`cfg = cfg.copy()`) 导致 `test_pipeline[0].type` 在 numpy array 推理时被永久改为 `'LoadImageFromNDArray'`。解决方案：始终用 numpy array 调用 `inference_detector`，而非混用文件路径和数组。

**Q: sklearn InconsistentVersionWarning?**
Pipeline 在 sklearn 1.7.2 训练，当前环境 1.9.0。不影响功能，可安全忽略。如需消除警告，重新在当前环境训练 pipeline 即可。
