"""Map DINOv2 scale proxy values to magnification labels."""
import sys, os, json
sys.path.insert(0, '.')
import numpy as np
from collections import defaultdict
from rate_identification.pipeline import ScaleEstimationPipeline

p = ScaleEstimationPipeline.load('data/syn_multimag/scale_pipeline_dinov2.joblib')

# Load training images and their magnifications
coco_root = 'data/syn_multimag/coco'
with open(os.path.join(coco_root, 'annotations/instances_train.json')) as f:
    coco = json.load(f)

# Predict scale for each training image, record magnification from filename
scale_to_mags = defaultdict(list)
for img in coco['images']:
    fname = img['file_name']
    mag = fname.split('_')[0].replace('p', '.')
    img_path = os.path.join(coco_root, 'images/train', fname)
    if not os.path.exists(img_path):
        continue
    s = p.predict_scale(img_path)
    # Find nearest cluster center
    centers = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    nearest = min(centers, key=lambda c: abs(s - c))
    scale_to_mags[nearest].append(mag)

print("Cluster center → magnification mapping (majority vote):")
cluster_to_mag = {}
for center in sorted(scale_to_mags.keys()):
    mags = scale_to_mags[center]
    unique, counts = np.unique(mags, return_counts=True)
    majority = unique[np.argmax(counts)]
    count = np.max(counts)
    cluster_to_mag[center] = majority
    print(f"  s={center:.2f} → {majority} ({count}/{len(mags)} images)")

# Window mapping
MAG_WINDOW = {'2.5x': 256, '5x': 512, '20x': 2048, '50x': 5120, '100x': 10240}
print("\nFinal scale→window mapping:")
for center in sorted(cluster_to_mag.keys()):
    mag = cluster_to_mag[center]
    window = MAG_WINDOW[mag]
    print(f"  s≈{center:.2f} → {mag} → window={window}")
