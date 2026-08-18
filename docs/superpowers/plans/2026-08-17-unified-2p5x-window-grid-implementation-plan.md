# Unified 2.5× Window Grid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a common coarse-to-fine sliding-window grid on four 2.5× images and produce six directly comparable metric heatmaps.

**Architecture:** Keep the existing one-image evaluator as the only inference path. A 70-cell coarse grid writes 280 compact JSON records; 20 exact pilot records are reused and only 260 are inferred. Thirteen Slurm tasks each process five or six strided cells, leaving room for the three existing account jobs under the 16-job limit. The plotter verifies four-image completeness, averages each metric, reports dispersion and checks boundaries before any fine-grid run.

**Tech Stack:** Python 3.10, NumPy, Matplotlib, MMDetection, pycocotools, Bash, Slurm, unittest.

---

### Task 1: Generalize grid specification without breaking the pilot

**Files:**
- Modify: `postprocess/window_sensitivity.py`
- Modify: `postprocess/plot_window_sensitivity.py`
- Modify: `tests/test_window_sensitivity.py`

- [ ] **Step 1: Write failing tests for explicit axes**

```python
COARSE_SIZES = (96, 128, 160, 192, 256, 320, 400, 512, 640, 768)
COARSE_OVERLAPS = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)

def test_parameter_grid_accepts_explicit_axes(self):
    grid = parameter_grid(COARSE_SIZES, COARSE_OVERLAPS)
    self.assertEqual(len(grid), 70)
    self.assertEqual(len(set(grid)), 70)
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `python -m unittest discover -s tests -p 'test_window_sensitivity.py' -v`

Expected: FAIL because `parameter_grid` does not accept axes.

- [ ] **Step 3: Add explicit-axis helpers while retaining pilot defaults**

```python
PILOT_WINDOW_SIZES = (192, 256, 320, 400, 512)
PILOT_OVERLAP_RATIOS = (0.0, 0.10, 0.15, 0.20, 0.30)

def parameter_grid(window_sizes=PILOT_WINDOW_SIZES,
                   overlap_ratios=PILOT_OVERLAP_RATIOS):
    return [(int(size), float(overlap))
            for size in window_sizes for overlap in overlap_ratios]
```

Pass requested axes into loader, matrix and renderer functions; do not mutate global constants.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest discover -s tests -p 'test_window_sensitivity.py' -v`

Expected: all tests PASS, including existing 5×5 pilot coverage.

- [ ] **Step 5: Commit**

```bash
git add postprocess/window_sensitivity.py postprocess/plot_window_sensitivity.py tests/test_window_sensitivity.py
git commit -m "Generalize window sensitivity grid axes"
```

### Task 2: Add pilot reuse and the four-image coarse-grid launcher

**Files:**
- Create: `postprocess/run_2p5x_window_sensitivity_coarse_array.sh`
- Create: `postprocess/reuse_2p5x_pilot_results.py`
- Modify: `tests/test_window_sensitivity.py`

- [ ] **Step 1: Write a failing mapping test**

```python
def test_coarse_launcher_maps_strided_cells_and_four_images(self):
    env = {**os.environ, "DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": "1"}
    out = subprocess.check_output(["bash", str(script)], env=env, text=True)
    self.assertIn("cell_id=1 patch_size=96 overlap_ratio=0.10", out)
    self.assertIn("cell_id=66 patch_size=768 overlap_ratio=0.20", out)
    for name in ("2p5x_00016.png", "2p5x_00017.png",
                 "2p5x_00018.png", "2p5x_00019.png"):
        self.assertIn(name, out)
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `python -m unittest discover -s tests -p 'test_window_sensitivity.py' -v`

Expected: FAIL because the launcher is absent.

- [ ] **Step 3: Implement validated reuse, 70-cell mapping and the four-image loop**

```bash
WINDOW_SIZES=(96 128 160 192 256 320 400 512 640 768)
OVERLAP_RATIOS=(0.05 0.10 0.15 0.20 0.30 0.45 0.60)
IMAGES=(2p5x_00016.png 2p5x_00017.png 2p5x_00018.png 2p5x_00019.png)
for ((CELL_ID=${SLURM_ARRAY_TASK_ID:-0}; CELL_ID<70; CELL_ID+=13)); do
    SIZE_INDEX=$((CELL_ID / 7))
    OVERLAP_INDEX=$((CELL_ID % 7))
done
```

The reuse helper must validate image name, size, overlap, model name and checkpoint before copying the 20 intersecting pilot JSON files into the new tree. For each missing image invoke `window_sensitivity.py` with unchanged settings and write `raw/window_0096_overlap_0p05/2p5x_00016.json`. Existing valid outputs are skipped. Use cluster-managed memory and at most four concurrent GPU tasks.

- [ ] **Step 4: Verify mapping and shell syntax**

Run: `bash -n postprocess/run_2p5x_window_sensitivity_coarse_array.sh && python -m unittest discover -s tests -p 'test_window_sensitivity.py' -v`

Expected: syntax check and all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add postprocess/run_2p5x_window_sensitivity_coarse_array.sh postprocess/reuse_2p5x_pilot_results.py tests/test_window_sensitivity.py
git commit -m "Add four-image coarse window grid launcher"
```

### Task 3: Aggregate four-image metrics and enforce common axes

**Files:**
- Modify: `postprocess/plot_window_sensitivity.py`
- Modify: `tests/test_window_sensitivity.py`

- [ ] **Step 1: Write failing completeness and boundary tests**

```python
def test_aggregate_requires_four_unique_images_per_cell(self):
    with self.assertRaisesRegex(ValueError, "four images"):
        load_multimage_grid_results(raw, sizes, overlaps, expected_images)

def test_boundary_report_is_metric_specific(self):
    report = find_metric_peaks(rows, sizes, overlaps)
    self.assertTrue(report["segm_mAP"]["on_boundary"])
    self.assertFalse(report["pixel_f1"]["on_boundary"])
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `python -m unittest discover -s tests -p 'test_window_sensitivity.py' -v`

Expected: FAIL because multi-image aggregation and boundary reporting are absent.

- [ ] **Step 3: Implement mean, sample deviation and peak checks**

```python
def summarize_cell(image_rows, metric):
    values = np.asarray([float(row[metric]) for row in image_rows])
    return float(values.mean()), float(values.std(ddof=1))

def is_boundary(row_index, column_index, height, width):
    return row_index in (0, height - 1) or column_index in (0, width - 1)
```

Write only `summary_per_image.csv`, `summary_mean.csv`, `summary.json`, six PNG/SVG heatmaps and one combined PNG/SVG. All heatmaps receive the same axes.

- [ ] **Step 4: Verify exact outputs**

Run: `python -W error -m unittest discover -s tests -p 'test_window_sensitivity.py' -v`

Expected: all tests PASS and the output-name assertion finds exactly 17 files.

- [ ] **Step 5: Commit**

```bash
git add postprocess/plot_window_sensitivity.py tests/test_window_sensitivity.py
git commit -m "Aggregate four-image window sensitivity results"
```

### Task 4: Execute and validate the common 10×7 coarse grid

**Files:**
- Generate: `outputs/dino_window_supplement/04_window_sensitivity_2p5x_coarse/`

- [ ] **Step 1: Reuse exact pilot cells and submit one scheduler-safe array**

Run:

```bash
python postprocess/reuse_2p5x_pilot_results.py --source .../03_window_sensitivity_2p5x_single/raw --destination .../04_window_sensitivity_2p5x_coarse/raw
sbatch --array=0-12%4 postprocess/run_2p5x_window_sensitivity_coarse_array.sh
```

Expected at submission: `reused=20`, 13 array tasks accepted, at most four running. Final expected state: all tasks `COMPLETED 0:0`.

- [ ] **Step 2: Validate raw records before plotting**

Run the aggregation CLI with `--validate-only`.

Expected: `cells=70 images_per_cell=4 records=280 reused=20 finite_metrics=yes`.

- [ ] **Step 3: Generate only approved summaries and heatmaps**

Run `plot_window_sensitivity.py --mode multimage` with the ten window sizes, seven overlap ratios and four expected image names from the design.

Expected: 70 mean rows, six common-axis heatmaps, one combined figure and a six-metric boundary report.

- [ ] **Step 4: Stop at the coarse-result checkpoint**

Do not start fine-grid inference. Report each metric's peak, boundary status and adjacent-cell values for review.

### Task 5: Define and run one common fine grid after approval

**Files:**
- Create after coarse review: `postprocess/run_2p5x_window_sensitivity_fine_array.sh`
- Generate: `outputs/dino_window_supplement/05_window_sensitivity_2p5x_fine/`

- [ ] **Step 1: Derive one shared fine range**

The coarse peaks span 128–400 px and overlap 0.20–0.60. Use one shared fine grid with margins: window sizes `96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 400, 416, 448` and overlap ratios `0.10, 0.15, ..., 0.70`. This is 13×13×4 = 676 records. Reuse the 168 exact coarse-grid records and infer only 508 missing records.

- [ ] **Step 2: Add and run launcher tests**

Assert the fine launcher maps its first and last cells, processes exactly four images and creates no visualization files.

- [ ] **Step 3: Run the fine grid in scheduler-safe waves**

Use the unchanged model/evaluator and maximum concurrency of four. Save only compact JSON records.

- [ ] **Step 4: Validate and plot with common fine axes**

Require four unique image records per cell and finite values for all six metrics. Generate the same 17 approved summary/figure files.

- [ ] **Step 5: Final verification and commit**

```bash
python -W error -m unittest discover -s tests -p 'test_window_sensitivity.py' -v
python -m unittest discover -s tests -p 'test_sliding_window_overlap.py' -v
python -m py_compile postprocess/window_sensitivity.py postprocess/plot_window_sensitivity.py
bash -n postprocess/run_2p5x_window_sensitivity_fine_array.sh
git diff --check
git add postprocess tests docs/superpowers
git commit -m "Complete coarse-to-fine 2.5x window sensitivity study"
```

Expected: all tests and syntax checks PASS; the working tree is clean after commit.
