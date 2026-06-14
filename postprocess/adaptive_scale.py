"""Adaptive window predictor using DINOv2 scale estimation pipeline.

Loads the trained scale_pipeline.joblib and predicts optimal sliding-window
parameters (patch_size, overlap) for each input image based on its estimated
magnification.

The DINOv2 pipeline predicts a continuous scale value s via kNN regression
on KMeans cluster centers.  Each cluster center maps to one magnification
(verified with ~92% accuracy on training data).

Usage:
    predictor = AdaptiveWindowPredictor('data/syn_multimag/scale_pipeline_dinov2.joblib')
    s, mag, window, overlap = predictor.predict(img_path)
"""

from __future__ import annotations

import os
import numpy as np

# ── Cluster center (from DINOv2 kNN) → magnification → window ──
# Verified 2026-06-05: majority-vote accuracy 88–100% per cluster
SCALE_TO_MAG = {
    0.25: '20x',
    0.40: '50x',
    0.55: '5x',
    0.70: '100x',
    0.85: '2.5x',
    1.00: '100x',
}
CLUSTER_CENTERS = np.array(sorted(SCALE_TO_MAG.keys()))

# ── Window sizes as fractions of image short side ──
# Calibrated on synthetic 2048×1362 images (short_side=1362):
#   2.5x: 256/1362 ≈ 0.188  → window ≈ 19% of short side
#   5x:   512/1362 ≈ 0.376  → window ≈ 38% of short side
#   20x: 2048/1362 ≈ 1.504 → capped at 1.0 (whole image)
# These fractions are applied to the actual image short side at prediction time.
MAG_WINDOW_FRAC = {
    '2.5x': 0.188,
    '5x': 0.376,
    '20x': 1.0,      # capped — nearly whole image
    '50x': 1.0,      # whole image
    '100x': 1.0,     # whole image
}


class AdaptiveWindowPredictor:
    """Predict sliding-window size from image content via DINOv2 pipeline."""

    def __init__(self, pipeline_path: str):
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Scale pipeline not found: {pipeline_path}")
        from rate_identification.pipeline import ScaleEstimationPipeline
        self.pipeline = ScaleEstimationPipeline.load(pipeline_path)

    def predict(self, img_path: str) -> tuple[float, str, float]:
        """Predict scale, magnification and window fraction.

        Returns:
            scale_s: continuous scale estimate from DINOv2 pipeline
            magnification: predicted magnification label (e.g. '20x')
            window_frac: fraction of image short side for square sliding window
        """
        s = self.pipeline.predict_scale(img_path)
        # Snap to nearest cluster center, then map to magnification
        idx = np.argmin(np.abs(CLUSTER_CENTERS - s))
        center = float(CLUSTER_CENTERS[idx])
        mag = SCALE_TO_MAG[center]
        frac = MAG_WINDOW_FRAC[mag]
        return float(s), mag, frac

    def predict_window(self, img_path: str, image_width: int, image_height: int) -> int:
        """Predict window size in pixels, scaled to image dimensions.

        Window is computed as a fraction of the image's short side.
        Returns 0 if window >= image short side (no sliding window needed).
        """
        _, _, frac = self.predict(img_path)
        min_dim = min(image_width, image_height)
        window = int(round(frac * min_dim))
        if window >= min_dim:
            return 0
        return window

    def predict_magnification(self, img_path: str) -> str:
        """Return predicted magnification label."""
        _, mag, _ = self.predict(img_path)
        return mag


def load_window_from_csv(csv_path: str) -> dict[float, dict]:
    """Load window recommendations from CSV (alternative to hardcoded map)."""
    import csv
    result = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mag = float(row['magnification'])
            result[mag] = {
                'window_width': int(row['window_width']),
                'window_height': int(row['window_height']),
                'square_window': int(row['square_window']),
                'square_window_aligned': int(row['square_window_aligned']),
            }
    return result
