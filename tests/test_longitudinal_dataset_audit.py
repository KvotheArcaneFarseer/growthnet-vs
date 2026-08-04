import csv
import importlib
import json
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
                provenance_path = out_dir / "longitudinal_provenance.json"
                metadata_path, qc_path = longitudinal.generate_longitudinal_dataset(
                    timeline_csv=timeline_csv,
                    background_csv=background_csv,
                    out_dir=out_dir,
                    seed=12345,
                    volume_ravd_tolerance=0.03,
                    volume_max_iterations=4,
                    gen_size=32,
                    provenance_json=provenance_path,
                )
            finally:
                longitudinal.run_embedding_case = original_embedding
                longitudinal._qc_mask = original_qc

            metadata_rows = self._read_csv(metadata_path)
            qc_rows = self._read_csv(qc_path)
            longitudinal_qc_path = out_dir / "longitudinal_qc_summary.csv"
            longitudinal_qc_rows = self._read_csv(longitudinal_qc_path)
            expected_seeds = [
                longitudinal._stable_seed(12345, "P001", f"T{i}")
                for i in range(1, 5)
            ]
            self.assertEqual(len(calls), 4)
            self.assertEqual([row["timepoint"] for row in metadata_rows], ["T1", "T2", "T3", "T4"])
            self.assertTrue(all(row["patient_id"] == "P001" for row in metadata_rows))
            self.assertTrue(all(row["background_mri_id"] == "BG001" for row in metadata_rows))
            self.assertTrue(all(row["growth_label"] == "growing" for row in metadata_rows))
            self.assertTrue(all(row["embedding_growth_mode"] == "steady" for row in metadata_rows))
            self.assertEqual([float(row["visit_day"]) for row in metadata_rows], [0.0, 365.25, 730.5, 1095.75])
            self.assertTrue(all(row["target_volume_source"] == "timeline_csv" for row in metadata_rows))
            self.assertTrue(all(row["growth_law_name"] == "none" for row in metadata_rows))
            self.assertTrue(all(row["growth_law_scenario"] == "timeline_csv" for row in metadata_rows))
            self.assertEqual([row["variant_id"] for row in metadata_rows], ["V01", "V01", "V01", "V01"])
            self.assertEqual([int(row["visit_seed"]) for row in metadata_rows], expected_seeds)
            self.assertEqual(
                [int(row["variant_seed"]) for row in metadata_rows],
                [
                    longitudinal._stable_seed(12345, "P001", f"T{i}:V01")
                    for i in range(1, 5)
                ],
            )
            self.assertTrue(all(row["source_mri_path"] == str(mri_path.resolve()) for row in metadata_rows))
            self.assertTrue(all(row["source_seg_path"] == str(seg_path.resolve()) for row in metadata_rows))
            self.assertTrue(all(row["volume_ravd_tolerance"] == "0.03" for row in metadata_rows))
            self.assertTrue(all(row["volume_max_iterations"] == "4" for row in metadata_rows))
            self.assertTrue(all(row["gen_size"] == "32" for row in metadata_rows))
            self.assertTrue(all(Path(row["image_path"]).is_file() for row in metadata_rows))
            self.assertTrue(all(Path(row["mask_path"]).is_file() for row in metadata_rows))
            self.assertTrue(all(row["qc_pass"] == "True" for row in qc_rows))
            self.assertEqual(len(longitudinal_qc_rows), 1)
            self.assertEqual(longitudinal_qc_rows[0]["patient_id"], "P001")
            self.assertEqual(longitudinal_qc_rows[0]["timepoint_count"], "4")
            self.assertEqual(longitudinal_qc_rows[0]["variant_count"], "1")
            self.assertEqual(longitudinal_qc_rows[0]["qc_pass_count"], "4")

            self.assertEqual(
                [call["seed"] for call in calls],
                [
                    longitudinal._stable_seed(12345, "P001", f"T{i}:V01")
                    for i in range(1, 5)
                ],
            )
            self.assertTrue(all(call["mri_path"] == mri_path.resolve() for call in calls))
            self.assertTrue(all(call["seg_path"] == seg_path.resolve() for call in calls))
            self.assertTrue(all(call["dates"] == [0, 1] for call in calls))
            self.assertTrue(all(call["growth"] == "steady" for call in calls))
            self.assertTrue(all(call["volume_target_timepoint"] == "first" for call in calls))
            self.assertTrue(all(call["volume_ravd_tolerance"] == 0.03 for call in calls))
            self.assertTrue(all(call["volume_max_iterations"] == 4 for call in calls))
            self.assertTrue(all(call["gen_size"] == 32 for call in calls))
            provenance = json.loads(provenance_path.read_text())
            self.assertEqual(provenance["schema_version"], "synthetic_longitudinal_provenance_v1")
            self.assertEqual(provenance["generation_parameters"]["seed"], 12345)
            self.assertEqual(provenance["generation_parameters"]["clinical_growth_law"], "none")
            self.assertEqual(provenance["generation_parameters"]["visit_days"], [0.0, 365.25, 730.5, 1095.75])
            self.assertEqual(provenance["generation_parameters"]["variants_per_timepoint"], 1)
            self.assertTrue(provenance["longitudinal_qc_summary_csv_sha256"])
            self.assertEqual(provenance["longitudinal_qc_rows"][0]["patient_id"], "P001")
            self.assertEqual(provenance["patient_count"], 1)
            self.assertEqual(provenance["timepoint_count"], 4)
            self.assertEqual(provenance["qc_pass_count"], 4)

    def test_multiple_variants_share_timepoint_targets_but_use_distinct_seeds(self):
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
                    variants_per_timepoint=2,
                )
            finally:
                longitudinal.run_embedding_case = original_embedding
                longitudinal._qc_mask = original_qc

            metadata_rows = self._read_csv(metadata_path)
            qc_rows = self._read_csv(qc_path)
            self.assertEqual(len(metadata_rows), 8)
            self.assertEqual(len(calls), 8)
            self.assertEqual([row["variant_id"] for row in metadata_rows[:2]], ["V01", "V02"])
            self.assertEqual([row["timepoint"] for row in metadata_rows[:2]], ["T1", "T1"])
            self.assertEqual([row["target_volume_mm3"] for row in metadata_rows[:2]], ["100.0", "100.0"])
            self.assertNotEqual(metadata_rows[0]["variant_seed"], metadata_rows[1]["variant_seed"])
            self.assertNotEqual(calls[0]["seed"], calls[1]["seed"])
            self.assertIn("_V01_", Path(metadata_rows[0]["image_path"]).name)
            self.assertIn("_V02_", Path(metadata_rows[1]["image_path"]).name)
            self.assertEqual([row["variant_id"] for row in qc_rows[:2]], ["V01", "V02"])

    def test_longitudinal_qc_summary_reports_ordering_volume_and_variant_counts(self):
        metadata_rows = [
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "variant_id": "V01",
                "visit_day": "0",
                "background_mri_id": "BG001",
                "target_volume_mm3": "100",
                "growth_law_scenario": "moderate_growth",
            },
            {
                "patient_id": "P001",
                "timepoint": "T2",
                "variant_id": "V01",
                "visit_day": "365.25",
                "background_mri_id": "BG001",
                "target_volume_mm3": "150",
                "growth_law_scenario": "moderate_growth",
            },
            {
                "patient_id": "P001",
                "timepoint": "T2",
                "variant_id": "V02",
                "visit_day": "365.25",
                "background_mri_id": "BG001",
                "target_volume_mm3": "150",
                "growth_law_scenario": "moderate_growth",
            },
        ]
        qc_rows = [
            {
                "patient_id": "P001",
                "timepoint": "T1",
                "variant_id": "V01",
                "synthetic_volume_mm3": "101",
                "relative_volume_error": "0.01",
                "qc_pass": "True",
            },
            {
                "patient_id": "P001",
                "timepoint": "T2",
                "variant_id": "V01",
                "synthetic_volume_mm3": "149",
                "relative_volume_error": "-0.0067",
                "qc_pass": "True",
            },
            {
                "patient_id": "P001",
                "timepoint": "T2",
                "variant_id": "V02",
                "synthetic_volume_mm3": "145",
                "relative_volume_error": "-0.033",
                "qc_pass": "False",
            },
        ]

        summary = longitudinal._summarize_longitudinal_qc(metadata_rows, qc_rows, volume_ravd_tolerance=0.05)

        self.assertEqual(len(summary), 1)
        row = summary[0]
        self.assertEqual(row["patient_id"], "P001")
        self.assertEqual(row["timepoint_count"], 2)
        self.assertEqual(row["variant_count"], 2)
        self.assertTrue(row["visit_days_strictly_increasing"])
        self.assertTrue(row["background_consistent"])
        self.assertTrue(row["target_volume_monotone_non_decreasing"])
        self.assertEqual(row["qc_pass_count"], 2)
        self.assertEqual(row["qc_fail_count"], 1)
        self.assertAlmostEqual(row["max_abs_relative_volume_error"], 0.033)
        self.assertEqual(row["engineering_qc_gate"], "FAIL")
        self.assertEqual(row["engineering_qc_failure_reasons"], "qc_fail_count>0")
        self.assertEqual(row["target_volume_trend_status"], "PASS")
        self.assertEqual(row["actual_volume_trend_status"], "PASS")

    def test_longitudinal_qc_gate_reports_engineering_failures_without_science_claims(self):
        metadata_rows = [
            {
                "patient_id": "P_BAD",
                "timepoint": "T2",
                "variant_id": "V01",
                "visit_day": "365.25",
                "background_mri_id": "BG001",
                "target_volume_mm3": "120",
                "growth_law_scenario": "moderate_growth",
            },
            {
                "patient_id": "P_BAD",
                "timepoint": "T1",
                "variant_id": "V01",
                "visit_day": "0",
                "background_mri_id": "BG002",
                "target_volume_mm3": "100",
                "growth_law_scenario": "moderate_growth",
            },
        ]
        qc_rows = [
            {
                "patient_id": "P_BAD",
                "timepoint": "T1",
                "variant_id": "V01",
                "synthetic_volume_mm3": "115",
                "relative_volume_error": "0.15",
                "qc_pass": "False",
            },
            {
                "patient_id": "P_BAD",
                "timepoint": "T2",
                "variant_id": "V01",
                "synthetic_volume_mm3": "110",
                "relative_volume_error": "-0.0833",
                "qc_pass": "True",
            },
        ]

        row = longitudinal._summarize_longitudinal_qc(
            metadata_rows,
            qc_rows,
            volume_ravd_tolerance=0.05,
        )[0]

        self.assertEqual(row["engineering_qc_gate"], "FAIL")
        self.assertEqual(
            row["engineering_qc_failure_reasons"],
            "background_inconsistent;qc_fail_count>0;max_abs_relative_volume_error>0.05",
        )
        self.assertEqual(row["target_volume_trend_status"], "PASS")
        self.assertEqual(row["actual_volume_trend_status"], "WARNING_NONMONOTONE_ACTUAL")

    def test_longitudinal_qc_gate_allows_declared_regression_target_trends(self):
        metadata_rows = [
            {
                "patient_id": "P_REG",
                "timepoint": "T1",
                "variant_id": "V01",
                "visit_day": "0",
                "background_mri_id": "BG001",
                "target_volume_mm3": "200",
                "growth_law_scenario": "regression",
            },
            {
                "patient_id": "P_REG",
                "timepoint": "T2",
                "variant_id": "V01",
                "visit_day": "365.25",
                "background_mri_id": "BG001",
                "target_volume_mm3": "180",
                "growth_law_scenario": "regression",
            },
        ]
        qc_rows = [
            {
                "patient_id": "P_REG",
                "timepoint": "T1",
                "variant_id": "V01",
                "synthetic_volume_mm3": "199",
                "relative_volume_error": "-0.005",
                "qc_pass": "True",
            },
            {
                "patient_id": "P_REG",
                "timepoint": "T2",
                "variant_id": "V01",
                "synthetic_volume_mm3": "181",
                "relative_volume_error": "0.0056",
                "qc_pass": "True",
            },
        ]

        row = longitudinal._summarize_longitudinal_qc(
            metadata_rows,
            qc_rows,
            volume_ravd_tolerance=0.05,
        )[0]

        self.assertEqual(row["engineering_qc_gate"], "PASS")
        self.assertEqual(row["engineering_qc_failure_reasons"], "")
        self.assertEqual(row["target_volume_trend_status"], "ALLOWED_BY_SCENARIO")
        self.assertEqual(row["actual_volume_trend_status"], "WARNING_NONMONOTONE_ACTUAL")


if __name__ == "__main__":
    unittest.main()
