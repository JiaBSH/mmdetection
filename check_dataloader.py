import mmcv
from mmengine.config import Config
from mmdet.registry import DATASETS, TRANSFORMS
from mmengine.registry import DATA_LOADERS
import os
import torch

def main():
    config_path = 'configs/custom/mask-rcnn_r50_fpn_instaboost_custom_coco_instance.py'
    cfg = Config.fromfile(config_path)
    
    # Force minimal settings
    cfg.train_dataloader.batch_size = 1
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False
    
    # Build dataloader
    dataloader = DATA_LOADERS.build(cfg.train_dataloader)
    
    # Fetch one batch
    data_iter = iter(dataloader)
    batch = next(data_iter)
    
    print("Successfully fetched one batch!")
    print(f"Batch keys: {batch.keys()}")

if __name__ == '__main__':
    main()
