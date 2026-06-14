# 重叠实例分割优化配置

本目录包含针对**密集、重叠实例分割**场景调优的模型配置——即每张图实例
数量多、实例之间空间重叠显著（标注也有重叠）。

基于 `configs/custom/` 修改而来，具体改动见下文。

## 核心问题

标准 NMS（非极大值抑制）会杀死重叠的 true positive：当同一类别的两个
实例高度重叠时，hard NMS 会直接丢弃得分较低的那个，即使两者都是正确检
测。

## 按模型族分类的重叠优化改动

### 两阶段检测器（Mask R-CNN、MS R-CNN、PointRend、Cascade、HTC、DetectoRS、QueryInst、ConvNeXt-V2）

| 设置项 | 原始值 | 重叠优化值 | 说明 |
|---|---|---|---|
| RPN `nms_pre` | 1000–2000 | **4000** | NMS 前保留更多 proposals |
| RPN `max_per_img` | 1000–2000 | **3000** | NMS 后保留更多 proposals |
| RCNN NMS 类型 | `nms`（硬抑制） | **`soft_nms`** | SoftNMS 衰减得分而非直接归零 |
| RCNN `score_thr` | 0.05 | **0.001** | 保留低置信度的重叠检测 |
| RCNN `max_per_img` | 100 | **300** | 为更多重叠实例留出空间 |
| RPN 训练 `pos_iou_thr` | 0.7 | **0.6** | 为重叠目标生成更多正 anchor |
| RCNN 训练 `pos_iou_thr` | 0.5 | **0.4** | 提高重叠 proposal 的召回率 |
| Cascade 各级 assigner | 0.5/0.6/0.7 | **0.4/0.5/0.6** | 用更多样本进行逐级精修 |

### SOLO / SOLOv2

| 设置项 | 原始值 | 重叠优化值 | 说明 |
|---|---|---|---|
| `score_thr` | 0.1 | **0.01** | 保留低置信度的重叠 mask |
| `mask_thr` | 0.5 | **0.3** | 更宽松的 mask 二值化 |
| `max_per_img` | 100 | **500** | 更多输出实例 |
| Matrix NMS sigma | 2.0（默认） | **2.0** | 高斯核实现软抑制 |

### Mask2Former

| 设置项 | 原始值 | 重叠优化值 | 说明 |
|---|---|---|---|
| `max_per_image` | 100 | **300** | 更多输出实例 |

基于 Transformer，天然对重叠友好——每个 query 关注一个实例，不受空间
重叠影响。

### YOLACT

| 设置项 | 原始值 | 重叠优化值 | 说明 |
|---|---|---|---|
| NMS 类型 | Fast NMS | **标准 NMS** | 标准 NMS 抑制更温和 |
| `score_thr` | 0.05 | **0.01** | 保留低置信度检测 |
| `max_per_img` | 100 | **300** | 更多输出实例 |

### SparseInst

| 设置项 | 原始值 | 重叠优化值 | 说明 |
|---|---|---|---|
| `num_masks` | 100 | **300** | 更多实例预测槽位 |
| `score_thr` | 0.005 | **0.001** | 保留低置信度实例 |
| `mask_thr_binary` | 0.45 | **0.3** | 更宽松的 mask 二值化 |
| 匹配器 `alpha/beta` | 0.8/0.2 | **0.5/0.5** | 平衡分类与位置匹配 |

### RTMDet-Ins

| 设置项 | 原始值 | 重叠优化值 | 说明 |
|---|---|---|---|
| NMS `iou_threshold` | 0.65 | **0.7** | 阈值更高 → 抑制更少重叠框 |
| `max_per_img` | 100 | **300** | 更多输出实例 |

### BoxInst / CondInst

| 设置项 | 原始值 | 重叠优化值 | 说明 |
|---|---|---|---|
| `score_thr` | 0.05 | **0.001** | 保留低置信度检测 |
| `max_per_img` | 100 | **300** | 更多输出实例 |
| CondInst NMS iou | 0.6 | **0.7** | 阈值更高 → 抑制更少 |

## 注意事项

- `mask-rcnn_r50_fpn_instaboost` 变体继承重叠优化后的 Mask R-CNN 基线，
  并在此基础上叠加 InstaBoost 数据增强。
- 这些配置会**增加预测输出数量**——预期会有更多误检，但召回率更高。
  如果精确率过低，可适当调高 `score_thr`。
- 对于极端重叠场景，建议同时启用 test-time augmentation（TTA）。
