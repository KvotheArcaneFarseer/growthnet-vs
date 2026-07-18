import csv
import importlib
import tempfile
import unittest
from pathlib import Path


longitudinal = importlib.import_module("scripts.generate_synthetic_longitudinal_dataset")


class LongitudinalDatasetAuditTests(unittest.TestCase):
    def _write_csv(self, path, fieldnames, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _read_csv(self, path):
        with path.open("r", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_helper_behavior_is_deterministic_and_conservative(self):
        self.assertEqual(longitudinal._safe_id(" Patient 01 / T1 "), "Patient_01_T1")
        self.assertEqual(longitudinal._safe_id("   "), "item")

        first = longitudinal._stable_seed(20260523, "P001", "T1")
        second = longitudinal._stable_seed(20260523, "P001", "T1")
        different_timepoint = longitudinal._stable_seed(20260523, "P001", "T2")
        self.assertEqual(first, second)
        self.assertNotEqual(first, different_timepoint)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**32)

        self.assertEqual(longitudinal._growth_mode("stable"), "stable")
        self.assertEqual(longitudinal._growth_mode("growing"), "steady")
        with self.assertRaisesRegex(ValueError, "Unsupported growth_label"):
            longitudinal._growth_mode("exponential")

    def test_missing_background_records_all_patient_timepoints_as_failures(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            timeline_csv = tmp / "timeline.csv"
            background_csv = tmp / "backgrounds.csv"
            out_dir = tmp / "out"

            self._write_csv(
                timeline_csv,
                longitudinal.TIMELINE_COLUMNS,
                [
                    {
                        "patient_id": "P_MISSING",
                        "background_mri_id": "BG_DOES_NOT_EXIST",
                        "T1_volume_mm3": "100",
                        "T2_volume_mm3": "110",
                        "T3_volume_mm3": "120",
                        "T4_volume_mm3": "130",
                        "growth_label": "stable",
                    }
                ],
            )
            self._write_csv(
                background_csv,
                longitudinal.BACKGROUND_COLUMNS,
                [
                    {
                        "background_mri_id": "BG_OTHER",
                        "mri_path": str(tmp / "missing_mri.nii.gz"),
                        "seg_path": str(tmp / "missing_seg.nii.gz"),
                    }
                ],
            )

            metadata_path, qc_path = longitudinal.generate_longitudinal_dataset(
                timeline_csv=timeline_csv,
                background_csv=background_csv,
                out_dir=out_dir,
            )

            metadata_rows = self._read_csv(metadata_path)
            qc_rows = self._read_csv(qc_path)
            self.assertEqual([row["timepoint"] for row in metadata_rows], ["T1", "T2", "T3", "T4"])
            self.assertEqual([row["target_volume_mm3"] for row in metadata_rows], ["100.0", "110.0", "120.0", "130.0"])
            self.assertTrue(all(row["image_path"] == "" for row in metadata_rows))
            self.assertTrue(all(row["mask_path"] == "" for row in metadata_rows))
            self.assertEqual(len(qc_rows), 4)
            self.assertTrue(all(row["qc_pass"] == "False" for row in qc_rows))
            self.assertTrue(all("BG_DOES_NOT_EXIST" in row["qc_failure_reason"] for row in qc_rows))

    def test_metadata_and_seed_routing_with_mocked_embedding(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            timeline_csv = tmp / "timeline.csv"
            background_csv = tmp / "backgrounds.csv"
            out_dir = tmp / "out"
            mri_path = tmp / "background_mri.nii.gz"
            seg_path = tmp / "background_seg.nii.gz"
            mri_path.write_text("mri")
            seg_path.write_text("seg")

            self._write_csv(
                timeline_csv,
                longitudinal.TIMELINE_COLUMNS,
                [
                    {
                        "patient_id": "P001",
                        "background_mri_id": "BG001",
                        "T1_volume_mm3": "100",
                        "T2_volume_mm3": "125",
                        "T3_volume_mm3": "150",
                        "T4_volume_mm3": "175",
                        "growth_label": "growing",
                    }
                ],
            )
            self._write_csv(
                background_csv,
                longitudinal.BACKGROUND_COLUMNS,
                [
                    {
                        "background_mri_id": "BG001",
                        "mri_path": str(mri_path),
                        "seg_path": str(seg_path),
                    }
                ],
            )

            calls = []
            original_embedding = longitudinal.run_embedding_case
            original_qc = longitudinal._qc_mask

            def fake_embedding(**kwargs):
                calls.append(kwargs)
                kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
                (kwargs["out_dir"] / "embedded_tumor_volume.nii.gz").write_text("image")
                (kwargs["out_dir"] / "embedded_tumor_mask.nii.gz").write_text("mask")

            def fake_qc(mask_path, target_volume_mm3, tolerance):
                return {
                    "synthetic_volume_mm3": float(target_volume_mm3),
                    "relative_volume_error": 0.0,
                    "connected_components": 1,
                    "qc_pass": True,
                    "qc_failure_reason": "",
                }

            longitudinal.run_embedding_case = fake_embedding
            longitudinal._qc_mask = fake_qc
            try:
                metadata_path, qc_path = longitudinal.generate_longitudinal_dataset(
                    timeline_csv=timeline_csv,
                    background_csv=background_csv,
                    out_dir=out_dir,
                    seed=12345,
                    volume_ravd_tolerance=0.03,
                    volume_max_iterations=4,
                    gen_size=32,
                )
            finally:
                longitudinal.run_embedding_case = original_embedding
                longitudinal._qc_mask = original_qc

            metadata_rows = self._read_csv(metadata_path)
            qc_rows = self._read_csv(qc_path)
            self.assertEqual(len(calls), 4)
            self.assertEqual([row["timepoint"] for row in metadata_rows], ["T1", "T2", "T3", "T4"])
            self.assertTrue(all(row["patient_id"] == "P001" for row in metadata_rows))
            self.assertTrue(all(row["background_mri_id"] == "BG001" for row in metadata_rows))
            self.assertTrue(all(row["growth_label"] == "growing" for row in metadata_rows))
            self.assertTrue(all(Path(row["image_path"]).is_file() for row in metadata_rows))
            self.assertTrue(all(Path(row["mask_path"]).is_file() for row in metadata_rows))
            self.assertTrue(all(row["qc_pass"] == "True" for row in qc_rows))

            expected_seeds = [
                longitudinal._stable_seed(12345, "P001", f"T{i}")
                for i in range(1, 5)
            ]
            self.assertEqual([call["seed"] for call in calls], expected_seeds)
            self.assertTrue(all(call["mri_path"] == mri_path.resolve() for call in calls))
            self.assertTrue(all(call["seg_path"] == seg_path.resolve() for call in calls))
            self.assertTrue(all(call["dates"] == [0, 1] for call in calls))
            self.assertTrue(all(call["growth"] == "steady" for call in calls))
            self.assertTrue(all(call["volume_target_timepoint"] == "first" for call in calls))
            self.assertTrue(all(call["volume_ravd_tolerance"] == 0.03 for call in calls))
            self.assertTrue(all(call["volume_max_iterations"] == 4 for call in calls))
            self.assertTrue(all(call["gen_size"] == 32 for call in calls))


if __name__ == "__main__":
    unittest.main()
