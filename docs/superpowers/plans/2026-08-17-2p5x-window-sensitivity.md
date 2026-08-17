# 2.5× Single-Image Window Sensitivity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a 25-cell window-size/overlap sensitivity pilot on `test2_5_t1/images/2p5x_00016.png`, computing COCO segm, COCO bbox, pixel precision, recall, F1, and IoU without generating inference visualizations or unrelated geometry artifacts.

**Architecture:** Add a lean evaluator that calls the existing sliding-window inference function directly, builds exact prediction/GT union masks for pixel metrics, and calls the existing COCO collector/evaluator for bbox and mask AP. Each Slurm array task writes one compact JSON. A separate CPU-only plotter validates all 25 JSON files, writes one summary CSV/JSON, and creates only the requested six individual heatmaps and one combined figure.

**Tech Stack:** Python 3.10, NumPy, Pillow, pycocotools, Matplotlib, MMDetection, unittest, Bash/Slurm.

---

## File structure

- Create `postprocess/window_sensitivity.py`: lean single-configuration evaluator and CLI; no plotting imports.
- Create `postprocess/plot_window_sensitivity.py`: result validation, CSV/JSON aggregation, and heatmap rendering.
- Create `postprocess/run_2p5x_window_sensitivity_array.sh`: maps Slurm array indices to the 5×5 grid and runs one configuration per task.
- Create `tests/test_window_sensitivity.py`: pure unit tests that require neither GPU nor MMDetection model loading.
- Create `docs/superpowers/plans/2026-08-17-2p5x-window-sensitivity.md`: committed copy of this plan.

### Task 1: Pure grid and mask metric helpers

**Files:**
- Create: `postprocess/window_sensitivity.py`
- Create: `tests/test_window_sensitivity.py`

- [ ] **Step 1: Write failing unit tests**

Add tests for the exact 25-cell grid, exact union-mask construction from prediction coordinates, and metric field normalization:

```python
import unittest
import numpy as np

from postprocess.window_sensitivity import (
    OVERLAP_RATIOS,
    WINDOW_SIZES,
    build_prediction_union_mask,
    parameter_grid,
)


class WindowSensitivityTest(unittest.TestCase):
    def test_parameter_grid_has_25_unique_cells(self):
        grid = parameter_grid()
        self.assertEqual(WINDOW_SIZES, (192, 256, 320, 400, 512))
        self.assertEqual(OVERLAP_RATIOS, (0.0, 0.10, 0.15, 0.20, 0.30))
        self.assertEqual(len(grid), 25)
        self.assertEqual(len(set(grid)), 25)
        self.assertIn((256, 0.15), grid)
        self.assertIn((400, 0.15), grid)

    def test_prediction_union_mask_preserves_all_instance_pixels(self):
        instances = [
            {"coords": np.array([[0, 0], [0, 1], [8, 8]])},
            {"coords": np.array([[0, 1], [1, 1], [-1, 3], [3, 20]])},
        ]
        mask = build_prediction_union_mask(instances, height=3, width=3)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.shape, (3, 3))
        self.assertEqual(int(mask.sum()), 3)
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[0, 1])
        self.assertTrue(mask[1, 1])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify the red state**

Run:

```bash
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate mmdetection_para
python -m unittest tests.test_window_sensitivity -v
```

Expected: import failure because `postprocess.window_sensitivity` does not exist.

- [ ] **Step 3: Implement minimal pure helpers**

Create constants and helpers without importing MMDetection at module import time:

```python
WINDOW_SIZES = (192, 256, 320, 400, 512)
OVERLAP_RATIOS = (0.0, 0.10, 0.15, 0.20, 0.30)


def parameter_grid() -> list[tuple[int, float]]:
    return [(size, overlap) for size in WINDOW_SIZES for overlap in OVERLAP_RATIOS]


def build_prediction_union_mask(instances, *, height: int, width: int):
    mask = np.zeros((height, width), dtype=np.bool_)
    for instance in instances:
        coords = np.asarray(instance.get("coords", []))
        if coords.ndim != 2 or coords.shape[1] != 2:
            continue
        ys = coords[:, 0].astype(np.int64, copy=False)
        xs = coords[:, 1].astype(np.int64, copy=False)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        mask[ys[valid], xs[valid]] = True
    return mask
```

- [ ] **Step 4: Run the focused tests**

Run `python -m unittest tests.test_window_sensitivity -v`.

Expected: two tests pass.

- [ ] **Step 5: Commit the pure helpers**

```bash
git add postprocess/window_sensitivity.py tests/test_window_sensitivity.py
git commit -m "Add window sensitivity metric helpers"
```

### Task 2: Lean one-cell evaluator

**Files:**
- Modify: `postprocess/window_sensitivity.py`
- Modify: `tests/test_window_sensitivity.py`

- [ ] **Step 1: Add failing payload and filename tests**

Test that the output filename is stable (`window_0256_overlap_0p15.json`) and that the final payload contains every required metric key plus runtime/window count.

- [ ] **Step 2: Run the focused tests and verify failure**

Run `python -m unittest tests.test_window_sensitivity -v`.

Expected: failure because `result_filename` and `normalize_result` are not defined.

- [ ] **Step 3: Implement the evaluator**

Implement these operations in `evaluate_configuration`:

1. Read the matching image record from the COCO JSON and capture `image_id`, width, and height.
2. Load the model lazily with `postprocess.run_postprocess._load_model`.
3. Call `postprocess.run_postprocess._infer_one_image` with `sliding_window=True`, the requested window/overlap, and batch size 4.
4. Build the exact predicted union mask from instance coordinates.
5. Load GT polygons with `postprocess.coco_utils.load_coco_gt_polygons` and rasterize their union once with `postprocess._pixel_metrics.build_pred_mask_from_polygons`.
6. Compute pixel metrics with `compute_pixel_metrics`.
7. Add the same prediction instances to `COCOResultCollector` and run `evaluate_coco_from_predictions(..., metrics=["bbox", "segm"], image_ids=[image_id], max_dets=10000)`.
8. Return a compact dictionary containing parameters, image/model/checkpoint provenance, `bbox_mAP`, `segm_mAP`, AP50/AP75 auxiliaries, `pixel_precision`, `pixel_recall`, `pixel_f1`, `pixel_iou`, inference seconds, instance count, and window count.
9. Write JSON atomically via a sibling `.tmp` path followed by `Path.replace`.

The evaluator must not call `process_one_image`, `analyze_domain_geometry_coco`, `_build_overlay`, any plotting function, or any image-save function.

- [ ] **Step 4: Implement and test CLI validation**

Reject window sizes not in `WINDOW_SIZES`, overlaps not in `OVERLAP_RATIOS`, missing image/config/checkpoint/annotation paths, and an output suffix other than `.json` before model loading.

- [ ] **Step 5: Run all focused tests**

Run `python -m unittest tests.test_window_sensitivity -v`.

Expected: all tests pass without importing `mmdet` or allocating a GPU.

- [ ] **Step 6: Commit the evaluator**

```bash
git add postprocess/window_sensitivity.py tests/test_window_sensitivity.py
git commit -m "Add lean single-image sensitivity evaluator"
```

### Task 3: Strict result aggregation and requested heatmaps

**Files:**
- Create: `postprocess/plot_window_sensitivity.py`
- Modify: `tests/test_window_sensitivity.py`

- [ ] **Step 1: Add failing aggregation tests**

Use a temporary directory containing 25 small JSON files. Assert that the loader returns 25 rows in deterministic size/overlap order, rejects duplicate cells, rejects missing cells, and exposes exactly the six requested heatmap metric names.

- [ ] **Step 2: Run tests and verify failure**

Run `python -m unittest tests.test_window_sensitivity -v`.

Expected: import failure for `postprocess.plot_window_sensitivity`.

- [ ] **Step 3: Implement validation and summary writers**

Define:

```python
HEATMAP_METRICS = (
    ("segm_mAP", "COCO Segm"),
    ("bbox_mAP", "COCO Box"),
    ("pixel_precision", "Precision"),
    ("pixel_recall", "Recall"),
    ("pixel_f1", "F1-score"),
    ("pixel_iou", "IoU"),
)
```

Write `summary.csv` with one row per cell and `summary.json` with grid/provenance/metric data. Require all 25 cells and finite values for the six main metrics before plotting.

- [ ] **Step 4: Implement Matplotlib-only rendering**

Use `imshow(..., cmap="YlOrRd")`; do not add seaborn. Annotate all cells to three decimals. Mark `(256, 0.15)` with a star outline and `(400, 0.15)` with a square outline. Save six individual PNG/SVG pairs and one 2×3 combined PNG/SVG pair.

- [ ] **Step 5: Verify plotting outputs in a temporary directory**

Expected files: `summary.csv`, `summary.json`, 6 PNG, 6 SVG, `window_sensitivity_combined.png`, and `window_sensitivity_combined.svg`; no additional figures.

- [ ] **Step 6: Commit the plotter**

```bash
git add postprocess/plot_window_sensitivity.py tests/test_window_sensitivity.py
git commit -m "Plot window sensitivity heatmaps"
```

### Task 4: Slurm array launcher

**Files:**
- Create: `postprocess/run_2p5x_window_sensitivity_array.sh`

- [ ] **Step 1: Implement exact array mapping**

Use `#SBATCH --array=0-24%4`. Map `task_id // 5` to `(192,256,320,400,512)` and `task_id % 5` to `(0.00,0.10,0.15,0.20,0.30)`. Use the existing `mmdetection_para` Conda environment, CUDA 13.0, and libstdc++ preload.

- [ ] **Step 2: Fix all inputs and outputs**

Defaults:

```text
image=data/syn_multimag/coco_rotation/test2_5_t1/images/2p5x_00016.png
annotation=data/syn_multimag/coco_rotation/test2_5_t1/instances_test.json
model=work_dirs/run_syn_rotation/detectors_htc-r50_custom_coco_instance/detectors_htc-r50_custom_coco_instance.py
checkpoint=work_dirs/run_syn_rotation/detectors_htc-r50_custom_coco_instance/epoch_17.pth
output=outputs/dino_window_supplement/03_window_sensitivity_2p5x_single/raw
```

Export only `BL_SLIDING_MERGE_OVERLAP_RATIO=0.3`; call the lean evaluator directly. Do not enable any `BL_GEOM_*` visualization or analysis flags.

- [ ] **Step 3: Verify shell syntax and index mapping**

Run `bash -n postprocess/run_2p5x_window_sensitivity_array.sh` and a dry-run mode for indices 0, 7, 12, 24. Expected mappings: `(192,0.00)`, `(256,0.15)`, `(320,0.15)`, `(512,0.30)`.

- [ ] **Step 4: Commit the launcher**

```bash
git add postprocess/run_2p5x_window_sensitivity_array.sh
git commit -m "Add 2.5x sensitivity Slurm array"
```

### Task 5: Real run and verification

**Files:**
- Output only: `outputs/dino_window_supplement/03_window_sensitivity_2p5x_single`

- [ ] **Step 1: Run the full unit suite relevant to changed code**

```bash
python -m unittest tests.test_window_sensitivity tests.test_sliding_window_overlap -v
bash -n postprocess/run_2p5x_window_sensitivity_array.sh
git diff --check
```

- [ ] **Step 2: Submit the 25-cell array**

```bash
mkdir -p outputs/dino_window_supplement/03_window_sensitivity_2p5x_single/raw logs
sbatch postprocess/run_2p5x_window_sensitivity_array.sh
```

- [ ] **Step 3: Verify completion before plotting**

Require all 25 array tasks to have Slurm state `COMPLETED` and exit code `0:0`. Require exactly 25 nonempty JSON files and no image files below `raw/`.

- [ ] **Step 4: Generate only the requested result artifacts**

```bash
python postprocess/plot_window_sensitivity.py \
  --raw-dir outputs/dino_window_supplement/03_window_sensitivity_2p5x_single/raw \
  --output-dir outputs/dino_window_supplement/03_window_sensitivity_2p5x_single/figures
```

- [ ] **Step 5: Cross-check every heatmap cell**

Programmatically compare each CSV value to its source JSON with absolute tolerance `1e-12`; verify 25 rows, six finite main metrics per row, DINO/manual marker coordinates, and the exact expected final figure count of 14.

- [ ] **Step 6: Verify no unrelated artifacts**

Run `find .../raw -type f` and require only `.json`. Report total output size. Do not retain predictions, masks, overlays, window previews, geometry tables, or per-image charts.

- [ ] **Step 7: Commit final code and report the Slurm job ID**

```bash
git status --short
git log --oneline -4
```

Do not commit generated experiment outputs to Git.
