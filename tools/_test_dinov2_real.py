"""Test DINOv2 scale pipeline on real microscope images."""
import sys, os
sys.path.insert(0, '/data/run01/scvi576/JiaBSH/mmdetection_para')
sys.path.insert(0, '/data/run01/scvi576/JiaBSH/mmdetection_para/Microscope_Magnification_Identification/src')

# Allow large real microscope images (4908x3264)
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from postprocess.adaptive_scale import AdaptiveWindowPredictor

p = AdaptiveWindowPredictor('data/syn_multimag/scale_pipeline_dinov2.joblib')
print('Pipeline loaded successfully')

test_base = 'dataset_root/mmdata_test'
for d in ['2_5x_unsup', '5x_unsup', '20x', '50x', '100x', 'sr2_5x_unsup', 'sr5x_unsup']:
    img_dir = os.path.join(test_base, d, 'image')
    if not os.path.isdir(img_dir):
        print(f'{d}: directory not found at {img_dir}')
        continue
    imgs = sorted(os.listdir(img_dir))
    for img_name in imgs:
        path = os.path.join(img_dir, img_name)
        s, mag, window = p.predict(path)
        # Also compute clamped window size
        from PIL import Image
        img = Image.open(path)
        w, h = img.size
        window_clamped = window if window < min(w, h) else 0
        print(f'{d}/{img_name}: WxH={w}x{h}, scale={s:.4f}, mag={mag}, window={window}, clamped_window={window_clamped}')
