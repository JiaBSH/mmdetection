import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from postprocess.window_sensitivity import (
    OVERLAP_RATIOS,
    WINDOW_SIZES,
    build_prediction_union_mask,
    normalize_result,
    parameter_grid,
    result_filename,
)
from postprocess.plot_window_sensitivity import (
    HEATMAP_METRICS,
    load_grid_results,
    render_heatmaps,
    write_summaries,
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

    def test_prediction_union_mask_preserves_all_valid_pixels(self):
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

    def test_result_filename_is_stable(self):
        self.assertEqual(
            result_filename(256, 0.15),
            "window_0256_overlap_0p15.json",
        )

    def test_normalize_result_exposes_required_metrics(self):
        result = normalize_result(
            patch_size=256,
            overlap_ratio=0.15,
            image="2p5x_00016.png",
            image_id=16,
            model_name="detectors_htc-r50_custom_coco_instance",
            checkpoint="epoch_17.pth",
            coco_metrics={
                "bbox_mAP": 0.4,
                "bbox_mAP_50": 0.5,
                "bbox_mAP_75": 0.3,
                "segm_mAP": 0.6,
                "segm_mAP_50": 0.7,
                "segm_mAP_75": 0.5,
            },
            pixel_metrics={
                "Precision": 0.8,
                "Recall": 0.9,
                "F1-score": 0.8470588235294118,
                "IoU": 0.735,
            },
            inference_seconds=12.5,
            instance_count=123,
            window_count=45,
        )
        required = {
            "bbox_mAP",
            "segm_mAP",
            "pixel_precision",
            "pixel_recall",
            "pixel_f1",
            "pixel_iou",
            "inference_seconds",
            "instance_count",
            "window_count",
        }
        self.assertTrue(required.issubset(result))
        self.assertEqual(result["patch_size"], 256)
        self.assertEqual(result["overlap_ratio"], 0.15)

    def _write_complete_grid(self, root: Path) -> None:
        for patch_size, overlap_ratio in parameter_grid():
            payload = {
                "patch_size": patch_size,
                "overlap_ratio": overlap_ratio,
                "image": "2p5x_00016.png",
                "image_id": 16,
                "model_name": "detectors_htc-r50_custom_coco_instance",
                "checkpoint": "epoch_17.pth",
                "bbox_mAP": 0.4,
                "bbox_mAP_50": 0.5,
                "bbox_mAP_75": 0.3,
                "segm_mAP": 0.6,
                "segm_mAP_50": 0.7,
                "segm_mAP_75": 0.5,
                "pixel_precision": 0.8,
                "pixel_recall": 0.9,
                "pixel_f1": 0.847,
                "pixel_iou": 0.735,
                "inference_seconds": 1.0,
                "instance_count": 10,
                "window_count": 20,
            }
            (root / result_filename(patch_size, overlap_ratio)).write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    def test_grid_loader_returns_deterministic_complete_rows(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write_complete_grid(root)
            rows = load_grid_results(root)
        self.assertEqual(len(rows), 25)
        self.assertEqual(
            [(row["patch_size"], row["overlap_ratio"]) for row in rows],
            parameter_grid(),
        )
        self.assertEqual(
            tuple(key for key, _label in HEATMAP_METRICS),
            (
                "segm_mAP",
                "bbox_mAP",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
                "pixel_iou",
            ),
        )

    def test_grid_loader_rejects_missing_cell(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self._write_complete_grid(root)
            (root / result_filename(192, 0.0)).unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                load_grid_results(root)

    def test_summary_and_heatmap_writers_create_only_requested_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            raw_dir = root / "raw"
            output_dir = root / "figures"
            raw_dir.mkdir()
            self._write_complete_grid(raw_dir)
            rows = load_grid_results(raw_dir)
            write_summaries(rows, output_dir)
            render_heatmaps(rows, output_dir)
            actual = {path.name for path in output_dir.iterdir() if path.is_file()}

        expected = {"summary.csv", "summary.json"}
        for metric, _label in HEATMAP_METRICS:
            expected.add(f"window_sensitivity_{metric}.png")
            expected.add(f"window_sensitivity_{metric}.svg")
        expected.add("window_sensitivity_combined.png")
        expected.add("window_sensitivity_combined.svg")
        self.assertEqual(actual, expected)

    def test_array_launcher_maps_selected_indices(self):
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "run_2p5x_window_sensitivity_array.sh"
        )
        expected = {
            0: "patch_size=192 overlap_ratio=0.00",
            7: "patch_size=256 overlap_ratio=0.15",
            12: "patch_size=320 overlap_ratio=0.15",
            24: "patch_size=512 overlap_ratio=0.30",
        }
        for task_id, text in expected.items():
            environment = os.environ.copy()
            environment.update(
                {
                    "DRY_RUN": "1",
                    "SLURM_ARRAY_TASK_ID": str(task_id),
                }
            )
            output = subprocess.check_output(
                ["bash", str(script)],
                env=environment,
                text=True,
            )
            self.assertIn(text, output)

    def test_array_launcher_uses_cluster_managed_memory(self):
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "run_2p5x_window_sensitivity_array.sh"
        )
        text = script.read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --mem=", text)


if __name__ == "__main__":
    unittest.main()
