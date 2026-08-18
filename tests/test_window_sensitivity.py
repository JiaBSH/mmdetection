import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import postprocess.plot_window_sensitivity as plot_sensitivity

from postprocess.window_sensitivity import (
    OVERLAP_RATIOS,
    WINDOW_SIZES,
    _validated_grid_cell,
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
    def test_expanded_grid_cells_are_valid(self):
        _validated_grid_cell(96, 0.60)
        _validated_grid_cell(768, 0.05)

    def test_parameter_grid_accepts_explicit_axes(self):
        sizes = (96, 128, 160, 192, 256, 320, 400, 512, 640, 768)
        overlaps = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)
        grid = parameter_grid(sizes, overlaps)
        self.assertEqual(len(grid), 70)
        self.assertEqual(len(set(grid)), 70)
        self.assertEqual(grid[0], (96, 0.05))
        self.assertEqual(grid[-1], (768, 0.60))

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

    def _write_multimage_grid(self, root, sizes, overlaps, images):
        for size, overlap in parameter_grid(sizes, overlaps):
            cell_dir = root / Path(result_filename(size, overlap)).stem
            cell_dir.mkdir(parents=True, exist_ok=True)
            for image_index, image in enumerate(images):
                value = size / 1000.0 + overlap + image_index / 100.0
                payload = {
                    "patch_size": size,
                    "overlap_ratio": overlap,
                    "image": image,
                    "image_id": image_index + 1,
                    "model_name": "detectors_htc-r50_custom_coco_instance",
                    "checkpoint": "epoch_17.pth",
                    "bbox_mAP": value,
                    "bbox_mAP_50": value,
                    "bbox_mAP_75": value,
                    "segm_mAP": value,
                    "segm_mAP_50": value,
                    "segm_mAP_75": value,
                    "pixel_precision": value,
                    "pixel_recall": value,
                    "pixel_f1": value,
                    "pixel_iou": value,
                    "inference_seconds": 1.0,
                    "instance_count": 10,
                    "window_count": 20,
                }
                (cell_dir / f"{Path(image).stem}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
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

    def test_coarse_launcher_maps_strided_cells_and_four_images(self):
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "run_2p5x_window_sensitivity_coarse_array.sh"
        )
        environment = os.environ.copy()
        environment.update({"DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": "1"})
        output = subprocess.check_output(
            ["bash", str(script)], env=environment, text=True
        )
        self.assertIn("cell_id=1 patch_size=96 overlap_ratio=0.10", output)
        self.assertIn("cell_id=66 patch_size=768 overlap_ratio=0.20", output)
        for image in (
            "2p5x_00016.png",
            "2p5x_00017.png",
            "2p5x_00018.png",
            "2p5x_00019.png",
        ):
            self.assertIn(image, output)

    def test_coarse_launcher_uses_complete_coco_image_directory(self):
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "run_2p5x_window_sensitivity_coarse_array.sh"
        )
        text = script.read_text(encoding="utf-8")
        self.assertIn(
            'data/syn_multimag/coco_rotation/images/test/${IMAGE_NAME}', text
        )
        self.assertNotIn(
            'data/syn_multimag/coco_rotation/test2_5_t1/images/${IMAGE_NAME}', text
        )

    def test_reuse_helper_copies_only_20_exact_pilot_records(self):
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "reuse_2p5x_pilot_results.py"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            self._write_complete_grid(source)
            output = subprocess.check_output(
                [
                    "python",
                    str(script),
                    "--source",
                    str(source),
                    "--destination",
                    str(destination),
                ],
                text=True,
            )
            copied = sorted(destination.rglob("*.json"))
            payloads = [json.loads(path.read_text()) for path in copied]
        self.assertIn("reused=20", output)
        self.assertEqual(len(copied), 20)
        self.assertTrue(all(row["image"] == "2p5x_00016.png" for row in payloads))
        self.assertTrue(all("reused_from" in row for row in payloads))

    def test_multimage_loader_requires_four_images_and_aggregates(self):
        sizes = (96, 128)
        overlaps = (0.05, 0.10)
        images = tuple(f"2p5x_{number:05d}.png" for number in range(16, 20))
        with tempfile.TemporaryDirectory() as temporary_dir:
            raw = Path(temporary_dir)
            self._write_multimage_grid(raw, sizes, overlaps, images)
            per_image, means = plot_sensitivity.load_multimage_grid_results(
                raw, sizes, overlaps, images
            )
            next(raw.rglob("2p5x_00019.json")).unlink()
            with self.assertRaisesRegex(ValueError, "four images"):
                plot_sensitivity.load_multimage_grid_results(
                    raw, sizes, overlaps, images
                )
        self.assertEqual(len(per_image), 16)
        self.assertEqual(len(means), 4)
        first = means[0]
        self.assertAlmostEqual(first["segm_mAP"], 0.161)
        self.assertGreater(first["segm_mAP_std"], 0.0)
        self.assertEqual(first["image_count"], 4)

    def test_metric_peak_report_flags_boundary_per_metric(self):
        sizes = (96, 128, 160)
        overlaps = (0.05, 0.10, 0.15)
        rows = []
        for row_index, size in enumerate(sizes):
            for column_index, overlap in enumerate(overlaps):
                row = {"patch_size": size, "overlap_ratio": overlap}
                for metric, _label in HEATMAP_METRICS:
                    row[metric] = 1.0 - abs(row_index - 1) - abs(column_index - 1)
                row["segm_mAP"] = float(row_index + column_index)
                rows.append(row)
        report = plot_sensitivity.find_metric_peaks(rows, sizes, overlaps)
        self.assertTrue(report["segm_mAP"]["on_boundary"])
        self.assertFalse(report["pixel_f1"]["on_boundary"])
        self.assertEqual(report["pixel_f1"]["patch_size"], 128)
        self.assertEqual(report["pixel_f1"]["overlap_ratio"], 0.10)

    def test_multimage_outputs_are_exactly_the_approved_17_files(self):
        sizes = (96, 128)
        overlaps = (0.05, 0.10)
        images = tuple(f"2p5x_{number:05d}.png" for number in range(16, 20))
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            raw = root / "raw"
            output = root / "figures"
            raw.mkdir()
            self._write_multimage_grid(raw, sizes, overlaps, images)
            per_image, means = plot_sensitivity.load_multimage_grid_results(
                raw, sizes, overlaps, images
            )
            plot_sensitivity.write_multimage_outputs(
                per_image, means, output, sizes, overlaps
            )
            actual = {path.name for path in output.iterdir() if path.is_file()}
        expected = {"summary_per_image.csv", "summary_mean.csv", "summary.json"}
        for metric, _label in HEATMAP_METRICS:
            expected.update(
                {f"window_sensitivity_{metric}.png", f"window_sensitivity_{metric}.svg"}
            )
        expected.update(
            {"window_sensitivity_combined.png", "window_sensitivity_combined.svg"}
        )
        self.assertEqual(actual, expected)

    def test_multimage_cli_validates_without_writing_outputs(self):
        sizes = (96, 128)
        overlaps = (0.05, 0.10)
        images = tuple(f"2p5x_{number:05d}.png" for number in range(16, 20))
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "plot_window_sensitivity.py"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            raw = Path(temporary_dir) / "raw"
            raw.mkdir()
            self._write_multimage_grid(raw, sizes, overlaps, images)
            output = subprocess.check_output(
                [
                    sys.executable,
                    str(script),
                    "--mode",
                    "multimage",
                    "--validate-only",
                    "--raw-dir",
                    str(raw),
                    "--window-sizes",
                    "96",
                    "128",
                    "--overlap-ratios",
                    "0.05",
                    "0.10",
                    "--expected-images",
                    *images,
                ],
                text=True,
            )
        self.assertIn("cells=4 images_per_cell=4 records=16", output)

    def test_fine_launcher_maps_common_13_by_13_grid(self):
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "run_2p5x_window_sensitivity_fine_array.sh"
        )
        environment = os.environ.copy()
        environment.update({"DRY_RUN": "1", "SLURM_ARRAY_TASK_ID": "12"})
        output = subprocess.check_output(
            ["bash", str(script)], env=environment, text=True
        )
        self.assertIn("cell_id=12 patch_size=96 overlap_ratio=0.70", output)
        self.assertIn("cell_id=168 patch_size=448 overlap_ratio=0.70", output)
        self.assertIn("2p5x_00019.png", output)
        text = script.read_text(encoding="utf-8")
        self.assertIn(
            'data/syn_multimag/coco_rotation/images/test/${IMAGE_NAME}', text
        )
        self.assertNotIn("#SBATCH --mem=", text)

    def test_fine_reuse_helper_copies_168_exact_coarse_records(self):
        coarse_sizes = (96, 128, 160, 192, 256, 320, 400, 512, 640, 768)
        coarse_overlaps = (0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)
        images = tuple(f"2p5x_{number:05d}.png" for number in range(16, 20))
        script = Path(__file__).resolve().parents[1] / "postprocess" / (
            "reuse_2p5x_coarse_results.py"
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "coarse"
            destination = root / "fine"
            source.mkdir()
            self._write_multimage_grid(
                source, coarse_sizes, coarse_overlaps, images
            )
            output = subprocess.check_output(
                [
                    sys.executable,
                    str(script),
                    "--source",
                    str(source),
                    "--destination",
                    str(destination),
                ],
                text=True,
            )
            copied = list(destination.rglob("*.json"))
            payloads = [json.loads(path.read_text()) for path in copied]
        self.assertIn("reused=168", output)
        self.assertEqual(len(copied), 168)
        self.assertTrue(all("reused_from" in row for row in payloads))


if __name__ == "__main__":
    unittest.main()
