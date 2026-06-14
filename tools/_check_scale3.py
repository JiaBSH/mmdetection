import sys
sys.path.insert(0, '.')
from rate_identification.pipeline import ScaleEstimationPipeline

p = ScaleEstimationPipeline.load('data/syn_multimag/scale_pipeline_dinov2.joblib')

print("=== PipelineConfig ===")
cfg = p.config
for k in dir(cfg):
    if not k.startswith('_'):
        v = getattr(cfg, k)
        if not callable(v):
            print(f"  {k}: {v}")

print("\n=== Predict on test images ===")
import os
test_dir = 'data/syn_multimag/coco/images/test'
for f in ['2p5x_00070.png', '5x_00070.png', '20x_00070.png', '50x_00070.png', '100x_00070.png']:
    path = os.path.join(test_dir, f)
    s = p.predict_scale(path)
    print(f"  {f}: predict_scale={s:.4f}")
