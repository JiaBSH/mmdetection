# Pretrained Checkpoints 下载指南

## 问题分析

### 根因

`download.openmmlab.com` 域名已过期（DNS CNAME → `expired.hichina.com`，阿里万网的过期域名 parking 页），导致所有 `load_from` URL 无法解析。

```
nslookup download.openmmlab.com
# → canonical name = expired.hichina.com
```

### 技术背景

根据 `docs/en/model_zoo.md` 第 5 行：

> **We only use aliyun to maintain the model zoo since MMDetection V2.0.**

即模型库文件托管在阿里云 OSS bucket 上，`download.openmmlab.com` 是该 bucket 绑定的**自定义域名**（custom domain）。OSS bucket 配置为私有访问模式，文件仅能通过自定义域名对外提供服务（标准 OSS 实践），无法通过 `open-mmlab.oss-cn-shanghai.aliyuncs.com` 直接访问。

验证结果：

| 地址 | 结果 | 说明 |
|------|------|------|
| `download.openmmlab.com` | DNS → expired.hichina.com | 域名过期 |
| `open-mmlab.oss-cn-beijing.aliyuncs.com/...` | `403 AccessDenied` → 重定向至 shanghai | bucket 在上海 |
| `open-mmlab.oss-cn-shanghai.aliyuncs.com/...` | `404 NoSuchKey` | 私有 bucket，需通过自定义域名访问 |
| `oss-cn-shanghai.aliyuncs.com/open-mmlab/...` | `403 SecondLevelDomainForbidden` | 禁止直接通过 OSS endpoint 访问 |

---

## 存储位置

所有 checkpoint 下载后放到以下缓存目录（mmengine 的 `torch.hub.load_url` 会优先读取本地缓存）：

```
/data/home/scvi576/.cache/torch/hub/checkpoints/
```

文件名必须与 URL 最后一段完全一致。

---

## 下载清单（13 个文件，1 个已完成）

| # | 模型配置 | 文件名 | 状态 |
|---|---------|--------|------|
| ✅ | boxinst | `boxinst_r50_fpn_ms-90k_coco_20221228_163052-6add751a.pth` | 已下载 |
| 1 | cascade-mask-rcnn | `cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth` | 缺失 |
| 2 | condinst | `condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth` | 缺失 |
| 3 | detectors_htc | `detectors_htc_r50_1x_coco-329b1453.pth` | 缺失 |
| 4 | htc-without-semantic | `htc_r50_fpn_1x_coco_20200317-7332cf16.pth` | 缺失 |
| 5 | mask2former | `mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth` | 缺失 |
| 6 | mask-rcnn / instaboost | `mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth` | 缺失 |
| 7 | ms-rcnn | `ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth` | 缺失 |
| 8 | point-rend | `point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth` | 缺失 |
| 9 | queryinst | `queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth` | 缺失 |
| 10 | rtmdet-ins_tiny | `rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth` | 缺失 |
| 11 | solo | `solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth` | 缺失 |
| 12 | solov2 | `solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth` | 缺失 |
| 13 | yolact | `yolact_r50_1x8_coco_20200908-f38d58df.pth` | 缺失 |

> **mask-rcnn_r50_fpn_instaboost** 继承自 mask-rcnn 配置，共用同一个 checkpoint，无需额外下载。

---

## 原始 URL 列表（域名已过期，仅供参考）

```
https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth
https://download.openmmlab.com/mmdetection/v3.0/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance/condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth
https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth
https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth
https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth
https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth
https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth
https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth
https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth
https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_1x_coco/solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth
https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth
https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth
```

---

## 替代下载方案

### 方案 1：Internet Archive（Wayback Machine）

```bash
CACHE_DIR="/data/home/scvi576/.cache/torch/hub/checkpoints"

URLS=(
    "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
    "https://download.openmmlab.com/mmdetection/v3.0/condinst/condinst_r50_fpn_ms-poly-90k_coco_instance/condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/detectors/detectors_htc_r50_1x_coco/detectors_htc_r50_1x_coco-329b1453.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/htc/htc_r50_fpn_1x_coco/htc_r50_fpn_1x_coco_20200317-7332cf16.pth"
    "https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x_coco/ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/point_rend/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth"
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/solo/solo_r50_fpn_1x_coco/solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_1x_coco/solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth"
    "https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth"
)

for url in "${URLS[@]}"; do
    fname=$(basename "$url")
    echo "Downloading: $fname"
    # 尝试最新的 Wayback Machine 快照
    wget -c "https://web.archive.org/web/2024/${url}" \
        -O "${CACHE_DIR}/${fname}" || echo "FAILED: $fname"
done
```

### 方案 2：OpenXLab 平台

OpenMMLab 官方新平台，部分模型可能已迁移：

- 平台地址：https://openxlab.org.cn
- 搜索模型名或 checkpoint 文件名
- 下载链接可能格式：`https://download.openxlab.org.cn/models/openmmlab/mmdetection/...`

### 方案 3：HuggingFace 社区

- 在 https://huggingface.co/models 搜索 `mmdetection` + 模型名
- 直接搜索 checkpoint 文件名，如 `cascade_mask_rcnn_r50_fpn_1x_coco_20200203`

### 方案 4：有代理/VPN 的机器

如果你有其他能访问阿里云 OSS（或 DNS 缓存还未更新）的机器，可以按以下路径操作：

```bash
# 在那台机器上执行
CACHE_DIR="/path/to/checkpoints"
for url in "${URLS[@]}"; do
    wget -c "$url" -P "$CACHE_DIR"
done

# 然后 scp 到集群
scp -r /path/to/checkpoints/*.pth \
    scvi576@m4gn1401:/data/home/scvi576/.cache/torch/hub/checkpoints/
```

---

## 下载后操作

### 步骤 1：清理残留文件

```bash
cd /data/home/scvi576/.cache/torch/hub/checkpoints/

# 删除所有不完整的 .partial 文件
rm -f *.partial

# 确认所有 checkpoint 都有内容（非 0 字节）
ls -lh *.pth
find . -name "*.pth" -size 0 -delete  # 如果有 0 字节文件，删掉
```

### 步骤 2：验证 checkpoint 可加载

```bash
cd /data/run01/scvi576/JiaBSH/mmdetection_para
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate mmdetection_para

# 验证每个 checkpoint
python -c "
import torch

files = [
    'cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth',
    'condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth',
    'detectors_htc_r50_1x_coco-329b1453.pth',
    'htc_r50_fpn_1x_coco_20200317-7332cf16.pth',
    'mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth',
    'mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth',
    'ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth',
    'point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth',
    'queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth',
    'rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth',
    'solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth',
    'solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth',
    'yolact_r50_1x8_coco_20200908-f38d58df.pth',
]

base = '/data/home/scvi576/.cache/torch/hub/checkpoints'
for f in files:
    path = f'{base}/{f}'
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt or 'meta' in ckpt:
            print(f'OK  : {f}')
        else:
            print(f'WARN: {f} - unexpected keys: {list(ckpt.keys())[:5]}')
    except Exception as e:
        print(f'FAIL: {f} - {e}')
"
```

> 注意：`weights_only=False` 是 PyTorch 2.6+ 验证时必须的，见下文。

### 步骤 3：重新运行训练

```bash
cd /data/run01/scvi576/JiaBSH/mmdetection_para
sbatch submm.sh
```

---

## 额外修复：PyTorch 2.6 `weights_only` 兼容性

PyTorch 2.6 把 `torch.load` 的 `weights_only` 默认从 `False` 改成了 `True`，旧格式 checkpoint 加载会报错：

```
Check the documentation of torch.load to learn more about types accepted by default with weights_only...
```

### 修复方式：在 `submm.sh` 中添加环境变量

```bash
# 在 submm.sh 的 export 区域加入
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0
```

### 或在 `submm.py` 开头 monkey-patch

```python
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
```

---

## 配置文件 → checkpoint 对照表

| 配置文件 | checkpoint 文件名 |
|----------|-----------------|
| `boxinst_r50_fpn_custom_coco_instance.py` | `boxinst_r50_fpn_ms-90k_coco_20221228_163052-6add751a.pth` |
| `cascade-mask-rcnn_r50_fpn_1x_custom_coco_instance.py` | `cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth` |
| `condinst_r50_fpn_custom_coco_instance.py` | `condinst_r50_fpn_ms-poly-90k_coco_instance_20221129_125223-4c186406.pth` |
| `detectors_htc-r50_custom_coco_instance.py` | `detectors_htc_r50_1x_coco-329b1453.pth` |
| `htc-without-semantic_r50_fpn_1x_custom_coco_instance.py` | `htc_r50_fpn_1x_coco_20200317-7332cf16.pth` |
| `mask2former_r50_custom_coco_instance.py` | `mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth` |
| `mask-rcnn_r50_fpn_1x_custom_coco_instance.py` | `mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth` |
| `mask-rcnn_r50_fpn_instaboost_custom_coco_instance.py` | 同上（继承） |
| `ms-rcnn_r50-caffe_fpn_1x_custom_coco_instance.py` | `ms_rcnn_r50_caffe_fpn_1x_coco_20200702_180848-61c9355e.pth` |
| `point-rend_r50-caffe_fpn_custom_coco_instance.py` | `point_rend_r50_caffe_fpn_mstrain_1x_coco-1bcb5fb4.pth` |
| `queryinst_r50_fpn_1x_custom_coco_instance.py` | `queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth` |
| `rtmdet-ins_tiny_custom_coco_instance.py` | `rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth` |
| `solo_r50_fpn_1x_custom_coco_instance.py` | `solo_r50_fpn_1x_coco_20210821_035055-2290a6b8.pth` |
| `solov2_r50_fpn_1x_custom_coco_instance.py` | `solov2_r50_fpn_1x_coco_20220512_125858-a357fa23.pth` |
| `yolact_r50_custom_coco_instance.py` | `yolact_r50_1x8_coco_20200908-f38d58df.pth` |
