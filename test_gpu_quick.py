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

# Forward pass (inference mode)
print("\n=== Forward (Inference) ===")
x = torch.randn(2, 3, 512, 512, device="cuda")
model.eval()
with torch.no_grad():
    out = model(x, mode="tensor")
shapes = [o.shape for lvl in out for o in lvl]
print(f"Input: {list(x.shape)}, Outputs: {len(shapes)} levels, shapes: {shapes}")

# Forward + backward (training mode) — needs dummy data_samples
print("\n=== Forward+Backward (Training) ===")
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

model.train()
H, W = 512, 512
x = torch.randn(2, 3, H, W, device="cuda")
data_samples = []
for i in range(2):
    ds = DetDataSample()
    ds.set_metainfo({
        "img_shape": (H, W),
        "pad_shape": (H, W),
        "scale_factor": (1.0, 1.0),
        "ori_shape": (H, W),
    })
    gt_instances = InstanceData()
    gt_instances.bboxes = torch.tensor([[50.0, 50.0, 200.0, 200.0]], dtype=torch.float32, device="cuda")
    gt_instances.labels = torch.tensor([0], dtype=torch.int64, device="cuda")
    ds.gt_instances = gt_instances
    data_samples.append(ds)

loss_dict = model(x, data_samples, mode="loss")
# RTMDet returns per-FPN-level lists; flatten and sum
all_losses = []
for k, v in loss_dict.items():
    if isinstance(v, list):
        all_losses.extend(v)
    elif isinstance(v, torch.Tensor):
        all_losses.append(v)
total_loss = sum(all_losses)
total_loss.backward()
print(f"Loss keys: {list(loss_dict.keys())}")
print(f"Total loss: {total_loss.item():.4f}, Backward: OK")

print(f"\nGPU max memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print(f"\n===== ALL TESTS PASSED =====")
print(f"G5090 environment is ready for training!")
