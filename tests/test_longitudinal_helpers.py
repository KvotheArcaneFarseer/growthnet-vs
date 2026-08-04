from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from scripts.generate_synthetic_longitudinal_dataset import (
    BACKGROUND_COLUMNS,
    CLINICAL_GROWTH_LAWS,
    TIMELINE_COLUMNS,
    TIMEPOINT_COLUMNS,
    _build_longitudinal_provenance_payload,
    _empirical_vs_annual_rate,
    _growth_mode,
    _load_backgrounds,
    _parse_visit_days,
    _qc_mask,
    _read_csv_rows,
    _target_volumes_for_patient,
    _safe_id,
    _stable_seed,
)


pytestmark = pytest.mark.fast


def test_longitudinal_constants_match_four_visit_schema():
    assert TIMEPOINT_COLUMNS == ("T1_volume_mm3", "T2_volume_mm3", "T3_volume_mm3", "T4_volume_mm3")
    for required in ("patient_id", "background_mri_id", "growth_label"):
        assert required in TIMELINE_COLUMNS
    assert BACKGROUND_COLUMNS == ("background_mri_id", "mri_path", "seg_path")


def test_safe_id_and_stable_seed_are_deterministic():
    assert _safe_id(" patient 1 / visit ") == "patient_1_visit"
    assert _safe_id(" ") == "item"
    assert _stable_seed(10, "patient-a", "T1") == _stable_seed(10, "patient-a", "T1")
    assert _stable_seed(10, "patient-a", "T1") != _stable_seed(10, "patient-a", "T2")


def test_read_csv_rows_validates_headers_and_nonempty_input(tmp_path: Path):
    timeline = tmp_path / "timeline.csv"
    timeline.write_text(
        "patient_id,background_mri_id,T1_volume_mm3,T2_volume_mm3,T3_volume_mm3,T4_volume_mm3,growth_label\n"
        "p1,bg1,10,20,30,40,growing\n"
    )

    rows = _read_csv_rows(timeline, TIMELINE_COLUMNS)

    assert rows[0]["patient_id"] == "p1"

    missing = tmp_path / "missing.csv"
    missing.write_text("patient_id\np1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _read_csv_rows(missing, TIMELINE_COLUMNS)

    empty = tmp_path / "empty.csv"
    empty.write_text(",".join(TIMELINE_COLUMNS) + "\n")
    with pytest.raises(ValueError, match="No rows found"):
        _read_csv_rows(empty, TIMELINE_COLUMNS)


def test_load_backgrounds_rejects_duplicate_ids(tmp_path: Path):
    csv_path = tmp_path / "backgrounds.csv"
    csv_path.write_text(
        "background_mri_id,mri_path,seg_path\n"
        "bg1,/tmp/mri-a.nii.gz,/tmp/seg-a.nii.gz\n"
        "bg1,/tmp/mri-b.nii.gz,/tmp/seg-b.nii.gz\n"
    )

    with pytest.raises(ValueError, match="Duplicate background_mri_id"):
        _load_backgrounds(csv_path)


def test_growth_mode_maps_only_supported_labels():
    assert _growth_mode(" stable ") == "stable"
    assert _growth_mode("Growing") == "steady"
    with pytest.raises(ValueError, match="Unsupported growth_label"):
        _growth_mode("rapid")


def test_parse_visit_days_requires_four_monotone_nonnegative_days():
    assert _parse_visit_days("0,180,365,730") == (0.0, 180.0, 365.0, 730.0)
    with pytest.raises(ValueError, match="four comma-separated"):
        _parse_visit_days("0,365")
    with pytest.raises(ValueError, match="non-negative"):
        _parse_visit_days("0,-1,365,730")
    with pytest.raises(ValueError, match="strictly increasing"):
        _parse_visit_days("0,365,365,730")


def test_empirical_vs_growth_law_is_deterministic_and_label_bounded():
    assert "empirical_vs_v1" in CLINICAL_GROWTH_LAWS
    assert "empirical_vs_v2" in CLINICAL_GROWTH_LAWS
    assert "scenario_mixture_v1" in CLINICAL_GROWTH_LAWS
    stable_a, stable_scenario_a = _empirical_vs_annual_rate("stable", np.random.default_rng(1))
    stable_b, stable_scenario_b = _empirical_vs_annual_rate("stable", np.random.default_rng(1))
    growing, growing_scenario = _empirical_vs_annual_rate("growing", np.random.default_rng(2))
    growing_v2, growing_v2_scenario = _empirical_vs_annual_rate(
        "growing",
        np.random.default_rng(2),
        clinical_growth_law="empirical_vs_v2",
    )
    growing_mixture, growing_mixture_scenario = _empirical_vs_annual_rate(
        "growing",
        np.random.default_rng(2),
        clinical_growth_law="scenario_mixture_v1",
    )

    assert stable_a == pytest.approx(stable_b)
    assert stable_scenario_a == stable_scenario_b == "stable"
    assert -0.20 <= stable_a <= 0.20
    assert growing >= 0.20
    assert growing_scenario in {"moderate_growth", "fast_growth"}
    assert 0.20 <= growing_v2 <= 1.20
    assert growing_v2_scenario == "moderate_growth"
    assert -0.20 <= growing_mixture <= 2.0
    assert growing_mixture_scenario in {"slow_growth", "moderate_growth", "fast_growth", "regression"}


def test_target_volumes_for_patient_can_use_empirical_growth_law():
    patient = {
        "patient_id": "P001",
        "background_mri_id": "BG001",
        "T1_volume_mm3": "100",
        "T2_volume_mm3": "999",
        "T3_volume_mm3": "999",
        "T4_volume_mm3": "999",
        "growth_label": "growing",
    }

    visits = _target_volumes_for_patient(
        patient=patient,
        clinical_growth_law="empirical_vs_v1",
        visit_days=(0.0, 365.25, 730.5, 1095.75),
        seed=123,
    )

    assert [visit["timepoint"] for visit in visits] == ["T1", "T2", "T3", "T4"]
    assert visits[0]["target_volume_mm3"] == pytest.approx(100.0)
    assert visits[1]["target_volume_mm3"] > visits[0]["target_volume_mm3"]
    assert visits[0]["target_volume_source"] == "clinical_growth_law"
    assert visits[0]["growth_law_name"] == "empirical_vs_v1"
    assert visits[0]["growth_law_annual_volume_change_fraction"] == pytest.approx(
        visits[1]["growth_law_annual_volume_change_fraction"]
    )


def test_empirical_vs_v2_is_less_explosive_than_v1_for_same_seeded_patient():
    patient = {
        "patient_id": "P_GROW",
        "background_mri_id": "BG001",
        "T1_volume_mm3": "100",
        "T2_volume_mm3": "999",
        "T3_volume_mm3": "999",
        "T4_volume_mm3": "999",
        "growth_label": "growing",
    }

    visits_v1 = _target_volumes_for_patient(
        patient=patient,
        clinical_growth_law="empirical_vs_v1",
        visit_days=(0.0, 365.25, 730.5, 1095.75),
        seed=20260523,
    )
    visits_v2 = _target_volumes_for_patient(
        patient=patient,
        clinical_growth_law="empirical_vs_v2",
        visit_days=(0.0, 365.25, 730.5, 1095.75),
        seed=20260523,
    )

    assert visits_v2[0]["target_volume_mm3"] == pytest.approx(100.0)
    assert visits_v2[-1]["target_volume_mm3"] < visits_v1[-1]["target_volume_mm3"]
    assert visits_v2[0]["growth_law_name"] == "empirical_vs_v2"
    assert visits_v2[1]["target_volume_mm3"] > visits_v2[0]["target_volume_mm3"]


def test_scenario_mixture_growth_law_records_named_growth_scenario():
    patient = {
        "patient_id": "P_SCENARIO",
        "background_mri_id": "BG001",
        "T1_volume_mm3": "100",
        "T2_volume_mm3": "999",
        "T3_volume_mm3": "999",
        "T4_volume_mm3": "999",
        "growth_label": "growing",
    }

    visits = _target_volumes_for_patient(
        patient=patient,
        clinical_growth_law="scenario_mixture_v1",
        visit_days=(0.0, 365.25, 730.5, 1095.75),
        seed=20260523,
    )

    assert {visit["growth_law_name"] for visit in visits} == {"scenario_mixture_v1"}
    assert visits[0]["growth_law_scenario"] in {
        "stable",
        "slow_growth",
        "moderate_growth",
        "fast_growth",
        "regression",
    }
    assert visits[0]["growth_law_scenario"] == visits[-1]["growth_law_scenario"]


def test_target_volumes_for_patient_default_preserves_timeline_csv_values():
    patient = {
        "patient_id": "P001",
        "background_mri_id": "BG001",
        "T1_volume_mm3": "100",
        "T2_volume_mm3": "125",
        "T3_volume_mm3": "150",
        "T4_volume_mm3": "175",
        "growth_label": "stable",
    }

    visits = _target_volumes_for_patient(
        patient=patient,
        clinical_growth_law="none",
        visit_days=(0.0, 180.0, 365.0, 730.0),
        seed=123,
    )

    assert [visit["target_volume_mm3"] for visit in visits] == [100.0, 125.0, 150.0, 175.0]
    assert all(visit["target_volume_source"] == "timeline_csv" for visit in visits)
    assert all(visit["growth_law_name"] == "none" for visit in visits)
    assert all(visit["growth_law_scenario"] == "timeline_csv" for visit in visits)


def test_build_longitudinal_provenance_payload_records_inputs_and_outputs(tmp_path: Path):
    timeline = tmp_path / "timeline.csv"
    timeline.write_text(
        "patient_id,background_mri_id,T1_volume_mm3,T2_volume_mm3,T3_volume_mm3,T4_volume_mm3,growth_label\n"
        "p1,bg1,10,20,30,40,growing\n"
    )
    backgrounds = tmp_path / "backgrounds.csv"
    backgrounds.write_text("background_mri_id,mri_path,seg_path\nbg1,/tmp/mri.nii.gz,/tmp/seg.nii.gz\n")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("patient_id,timepoint\np1,T1\n")
    qc = tmp_path / "qc_summary.csv"
    qc.write_text("patient_id,timepoint,qc_pass\np1,T1,True\n")
    longitudinal_qc = tmp_path / "longitudinal_qc_summary.csv"
    longitudinal_qc.write_text("patient_id,qc_pass_count,qc_fail_count\np1,1,0\n")

    payload = _build_longitudinal_provenance_payload(
        timeline_csv=timeline,
        background_csv=backgrounds,
        out_dir=tmp_path,
        metadata_path=metadata,
        qc_summary_path=qc,
        longitudinal_qc_summary_path=longitudinal_qc,
        generation_parameters={
            "seed": 7,
            "volume_ravd_tolerance": 0.05,
            "volume_max_iterations": 3,
            "gen_size": 32,
        },
        timeline_rows=[{"patient_id": "p1", "background_mri_id": "bg1"}],
        metadata_rows=[{"patient_id": "p1", "timepoint": "T1"}],
        qc_rows=[{"patient_id": "p1", "timepoint": "T1", "qc_pass": True}],
        longitudinal_qc_rows=[{"patient_id": "p1", "qc_pass_count": 1, "qc_fail_count": 0}],
    )

    assert payload["schema_version"] == "synthetic_longitudinal_provenance_v1"
    assert payload["timeline_csv_sha256"]
    assert payload["background_csv_sha256"]
    assert payload["metadata_csv_sha256"]
    assert payload["qc_summary_csv_sha256"]
    assert payload["longitudinal_qc_summary_csv_sha256"]
    assert payload["generation_parameters"]["seed"] == 7
    assert payload["patient_count"] == 1
    assert payload["timepoint_count"] == 1
    assert payload["longitudinal_qc_rows"][0]["patient_id"] == "p1"


@pytest.mark.integration
def test_qc_mask_reports_volume_and_component_failures(tmp_path: Path):
    mask = np.zeros((8, 8, 8), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1
    mask[6, 6, 6] = 1
    seg_path = tmp_path / "two_component_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(seg_path))

    qc = _qc_mask(mask_path=seg_path, target_volume_mm3=8.0, tolerance=0.05)

    assert qc["synthetic_volume_mm3"] == pytest.approx(9.0)
    assert qc["connected_components"] == 2
    assert qc["qc_pass"] is False
    assert "connected_components=2" in qc["qc_failure_reason"]
    assert "relative_volume_error>0.05" in qc["qc_failure_reason"]
