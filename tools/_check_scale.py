import os, sys, inspect
sys.path.insert(0, '.')
from rate_identification.pipeline import ScaleEstimationPipeline

p = ScaleEstimationPipeline.load('data/syn_multimag/scale_pipeline_dinov2.joblib')
print("predict_scale signature:", inspect.signature(p.predict_scale))

test_dir = 'data/syn_multimag/coco/images/test'
for f in ['2p5x_00070.png', '50x_00070.png', '100x_00070.png']:
    path = os.path.join(test_dir, f)
    result = p.predict_scale(path)
    print(f'{f}: type={type(result).__name__}, val={result}')
