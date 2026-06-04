#!/usr/bin/env python
"""Quick end-to-end GPU training test for mmdetection on G5090.

Creates a tiny synthetic COCO dataset and runs a few training + inference steps.
"""
import json
import os
import shutil
import tempfile

import numpy as np
from PIL import Image

# ── Create synthetic COCO dataset ──────────────────────────────────────────
tmp_dir = tempfile.mkdtemp(prefix="mmdet_test_")
print(f"Test dir: {tmp_dir}")

img_dir = os.path.join(tmp_dir, "images", "train")
ann_dir = os.path.join(tmp_dir, "annotations")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

# Generate 20 tiny synthetic images
images = []
annotations = []
for i in range(20):
    fname = f"img_{i:04d}.jpg"
    # Small 256x256 random RGB image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(img).save(os.path.join(img_dir, fname))

    images.append({
        "id": i,
        "file_name": f"images/train/" + fname,
        "width": 256,
        "height": 256,
    })

    # Add one random bbox per image
    x, y, w, h = np.random.randint(0, 200, 4)
    annotations.append({
        "id": i,
        "image_id": i,
        "category_id": 0,
        "bbox": [int(x), int(y), max(int(w), 10), max(int(h), 10)],
        "area": max(int(w), 10) * max(int(h), 10),
        "iscrowd": 0,
        "segmentation": [],
    })

coco_json = {
    "images": images,
    "annotations": annotations,
    "categories": [{"id": 0, "name": "test_class"}],
}
with open(os.path.join(ann_dir, "instances_train.json"), "w") as f:
    json.dump(coco_json, f)

# Copy as val/test too (same data, just for test)
shutil.copy(
    os.path.join(ann_dir, "instances_train.json"),
    os.path.join(ann_dir, "instances_val.json"),
)
shutil.copy(
    os.path.join(ann_dir, "instances_train.json"),
    os.path.join(ann_dir, "instances_test.json"),
)
# Make val/test image dirs
for split in ["val", "test"]:
    sdir = os.path.join(tmp_dir, "images", split)
    os.makedirs(sdir, exist_ok=True)
    for i in range(20):
        shutil.copy(
            os.path.join(img_dir, f"img_{i:04d}.jpg"),
            os.path.join(sdir, f"img_{i:04d}.jpg"),
        )

print("Synthetic dataset created (20 images)")

# ── Import mmdet ────────────────────────────────────────────────────────────
import mmcv
import mmengine
import mmdet
import torch

print(f"Torch:      {torch.__version__}  CUDA: {torch.cuda.is_available()}")
print(f"GPU:        {torch.cuda.get_device_name(0)}")
print(f"mmcv:       {mmcv.__version__}")
print(f"mmdet:      {mmdet.__version__}")
print(f"mmengine:   {mmengine.__version__}")

# ── Model config via RTMDet tiny (fast to test) ─────────────────────────────
from mmengine.config import Config

# Use a very simple built-in config overridden for our synthetic data
cfg = Config.fromfile("configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py")

# Override for tiny test
cfg.data_root = tmp_dir + "/"
cfg.metainfo = dict(classes=("test_class",))
cfg.num_classes = 1

# Reduce everything for quick test
cfg.train_dataloader.batch_size = 2
cfg.train_dataloader.num_workers = 0
cfg.train_dataloader.persistent_workers = False
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.metainfo = cfg.metainfo
cfg.train_dataloader.dataset.ann_file = "annotations/instances_train.json"
cfg.train_dataloader.dataset.data_prefix = dict(img="images/train/")

cfg.val_dataloader.batch_size = 2
cfg.val_dataloader.num_workers = 0
cfg.val_dataloader.persistent_workers = False
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.metainfo = cfg.metainfo
cfg.val_dataloader.dataset.ann_file = "annotations/instances_val.json"
cfg.val_dataloader.dataset.data_prefix = dict(img="images/val/")

cfg.test_dataloader = cfg.val_dataloader

cfg.val_evaluator.ann_file = cfg.data_root + "annotations/instances_val.json"
cfg.val_evaluator.metric = ["bbox"]

cfg.model.bbox_head.num_classes = 1
cfg.model.bbox_head.loss_cls = dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0)

cfg.train_cfg.max_epochs = 1
cfg.train_cfg.val_interval = 1
cfg.default_hooks.checkpoint = dict(type="CheckpointHook", interval=999)
cfg.default_hooks.logger.interval = 1
cfg.log_level = "INFO"

work_dir = os.path.join(tmp_dir, "work_dir")
os.makedirs(work_dir, exist_ok=True)
cfg.work_dir = work_dir

# ── Build model ─────────────────────────────────────────────────────────────
from mmengine.runner import Runner

print("\n===== Building Runner =====")
runner = Runner.from_cfg(cfg)

print("\n===== Training (1 epoch) =====")
runner.train()

print("\n===== Running inference =====")
# Quick inference on a single image
img_tensor = torch.randn(1, 3, 256, 256).cuda()
runner.model.eval()
with torch.no_grad():
    result = runner.model.test_step(
        {"inputs": [img_tensor], "data_samples": [runner.model.data_preprocessor({"inputs": [img_tensor.cpu()]})["data_samples"][0]]}
    )

print(f"Inference result type: {type(result)}")
print(f"GPU memory used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# ── Cleanup ─────────────────────────────────────────────────────────────────
shutil.rmtree(tmp_dir)
print(f"\n===== ALL TESTS PASSED =====")
print(f"Environment is ready for training on {torch.cuda.get_device_name(0)}")
