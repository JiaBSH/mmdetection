import sys
sys.path.insert(0, '.')
from rate_identification.pipeline import ScaleEstimationPipeline

p = ScaleEstimationPipeline.load('data/syn_multimag/scale_pipeline_dinov2.joblib')

print("=== Pipeline config ===")
for k, v in p.__dict__.items():
    print(f"  {k}: {type(v).__name__}")

if hasattr(p, 'config'):
    print("\n=== Config ===")
    for k, v in p.config.items():
        print(f"  {k}: {v}")

if hasattr(p, 'scale_to_window'):
    print("\n  scale_to_window:", p.scale_to_window)

# Check what methods are available
methods = [m for m in dir(p) if not m.startswith('_') and callable(getattr(p, m))]
print("\n=== Methods ===")
for m in methods:
    print(f"  {m}")
