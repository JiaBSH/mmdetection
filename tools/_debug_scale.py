import sys
sys.path.insert(0, '.')
import numpy as np
from rate_identification.pipeline import ScaleEstimationPipeline

p = ScaleEstimationPipeline.load('data/syn_multimag/scale_pipeline_dinov2.joblib')

# Check scale proxy database
sdb = p.scale_proxy_db
print(f'scale_proxy_db shape={sdb.shape}')
print(f'values: min={sdb.min():.4f} max={sdb.max():.4f}')
unique = np.unique(sdb)
print(f'unique values ({len(unique)}): {unique}')

# Check config
cfg = p.config
print(f'\ncluster_count={cfg.cluster_count}')
print(f'knn_neighbors={cfg.knn_neighbors}')
print(f'resize_long_edge={cfg.resize_long_edge}')
print(f'window: w_min={cfg.window.w_min} w_max={cfg.window.w_max} alpha={cfg.window.alpha} beta={cfg.window.beta}')

# Check what predict_scale returns for each mag
import os
test_dir = 'data/syn_multimag/coco/images/test'
for fname in ['2p5x_00070.png', '5x_00070.png', '20x_00070.png', '50x_00070.png', '100x_00070.png']:
    path = os.path.join(test_dir, fname)
    s = p.predict_scale(path)
    # Also check nearest neighbors
    from rate_identification.pipeline import load_rgb_image
    img = load_rgb_image(path)
    feat = p._predict_from_path(path)
    # Get kNN
    dist, idx = p.neighbors.kneighbors(p.reduced_embeddings, n_neighbors=3)
    # Actually this is wrong - let me just check the prediction
    print(f'\n{fname}: predict_scale={s:.4f}')
