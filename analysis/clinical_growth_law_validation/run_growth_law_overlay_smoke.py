#!/usr/bin/env python3
"""Local overlay smoke for the empirical VS longitudinal growth law.

This script uses the repository's existing embedded MRI/mask pair as a local
engineering fixture. It validates target-volume propagation into the final
pasted mask and QC overlay, but it does not establish clinical realism.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embed_tumor import main as run_embedding_case  # noqa: E402
from scripts.generate_synthetic_longitudinal_dataset import (  # noqa: E402
    DEFAULT_VISIT_DAYS,
    _growth_mode,
    _stable_seed,
    _target_volumes_for_patient,
)


OUT_DIR = REPO_ROOT / "analysis" / "clinical_growth_law_validation"
OVERLAY_DIR = OUT_DIR / "overlays"
RESULT_CSV = OUT_DIR / "growth_law_overlay_smoke.csv"
REPORT_MD = OUT_DIR / "GROWTH_LAW_OUTPUT_VALIDATION.md"
LOCAL_SOURCE_MRI = REPO_ROOT / "embedding_outputs" / "embedded_tumor_volume.nii.gz"
LOCAL_SOURCE_SEG = REPO_ROOT / "embedding_outputs" / "embedded_tumor_mask.nii.gz"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "timepoint",
        "visit_day",
        "growth_label",
        "growth_law_name",
        "annual_volume_change_fraction",
        "target_volume_mm3",
        "actual_volume_mm3",
        "ravd",
        "volume_converged",
        "volume_iterations",
        "target_timepoint_voxels",
        "qc_overlay_path",
        "hard_failures",
        "warnings",
        "volume_status",
        "embedding_qc_status",
        "overall_status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    volume_passed = [row for row in rows if row["volume_status"] == "PASS"]
    embedding_failed = [row for row in rows if row["embedding_qc_status"] != "PASS"]
    max_ravd = max((float(row["ravd"]) for row in volume_passed), default=float("nan"))
    lines = [
        "# Growth Law Output Validation",
        "",
        "## Scope",
        "",
        "This is a local engineering smoke test for `clinical_growth_law=empirical_vs_v2`.",
        "It uses `embedding_outputs/embedded_tumor_volume.nii.gz` and",
        "`embedding_outputs/embedded_tumor_mask.nii.gz` as a reusable local MRI/mask",
        "fixture, then runs the real embedding path for four visits. This checks",
        "whether law-derived target volumes are propagated into the pasted mask and",
        "QC overlay. It is not a clinical validation dataset.",
        "",
        "## Result",
        "",
        f"- Visits attempted: {len(rows)}",
        f"- Visits passing volume tolerance: {len(volume_passed)}",
        f"- Visits with embedding QC hard failures: {len(embedding_failed)}",
        f"- Max RAVD among volume-passing visits: {max_ravd:.4f}" if volume_passed else "- Max RAVD among volume-passing visits: n/a",
        f"- Results CSV: `{RESULT_CSV.relative_to(REPO_ROOT)}`",
        f"- QC overlays: `{OVERLAY_DIR.relative_to(REPO_ROOT)}/`",
        "",
        "## Per-Visit Summary",
        "",
        "| Timepoint | Visit day | Target mm3 | Actual mm3 | RAVD | Converged | Volume | Embedding QC |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {timepoint} | {visit_day:.2f} | {target_volume_mm3:.2f} | "
            "{actual_volume_mm3:.2f} | {ravd:.4f} | {volume_converged} | "
            "{volume_status} | {embedding_qc_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The longitudinal law produces target volumes in mm3. The embedding engine then",
            "optimizes the generated tumor size for each visit and reports the actual",
            "pasted mask volume. Any nonzero RAVD here is an embedding/voxelization",
            "realization error, not a growth-law math error.",
            "",
            "The local source MRI/mask fixture contains a very small prior embedded mask.",
            "Its `placed_to_seg_ratio` warnings/failures therefore limit anatomical QC",
            "interpretation. They do not invalidate the target-to-output volume check.",
            "",
            "Scientific validation remains separate: this smoke does not prove that the",
            "shape trajectory, anatomy, or patient-level growth process is clinically",
            "realistic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if not LOCAL_SOURCE_MRI.is_file() or not LOCAL_SOURCE_SEG.is_file():
        raise FileNotFoundError(
            "Local embedded MRI/mask fixture is missing; expected "
            f"{LOCAL_SOURCE_MRI} and {LOCAL_SOURCE_SEG}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    patient = {
        "patient_id": "GROWTH_LAW_SMOKE",
        "background_mri_id": "LOCAL_EMBED_FIXTURE",
        "T1_volume_mm3": "100",
        "T2_volume_mm3": "0",
        "T3_volume_mm3": "0",
        "T4_volume_mm3": "0",
        "growth_label": "growing",
    }
    visits = _target_volumes_for_patient(
        patient=patient,
        clinical_growth_law="empirical_vs_v2",
        visit_days=DEFAULT_VISIT_DAYS,
        seed=20260523,
    )

    rows: list[dict[str, Any]] = []
    for visit in visits:
        timepoint = str(visit["timepoint"])
        target = float(visit["target_volume_mm3"])
        visit_seed = _stable_seed(20260523, patient["patient_id"], timepoint)
        with tempfile.TemporaryDirectory(prefix=f"growth_law_{timepoint}_") as tmp_name:
            tmp_out = Path(tmp_name)
            run_embedding_case(
                mri_path=LOCAL_SOURCE_MRI,
                seg_path=LOCAL_SOURCE_SEG,
                out_dir=tmp_out,
                gen_size=64,
                dates=[0, 1],
                growth=_growth_mode(patient["growth_label"]),
                seed=visit_seed,
                target_tumor_volume_mm3=target,
                volume_target_timepoint="first",
                volume_ravd_tolerance=0.15,
                volume_max_iterations=8,
            )
            metrics = json.loads((tmp_out / "embedding_metrics.json").read_text(encoding="utf-8"))
            overlay_path = OVERLAY_DIR / f"{patient['patient_id']}_{timepoint}_qc_embedding.png"
            shutil.copy2(tmp_out / "qc_embedding.png", overlay_path)

        hard_failures = metrics.get("hard_failures", [])
        warnings = metrics.get("warnings", [])
        volume_status = "PASS" if float(metrics["ravd"]) <= 0.15 and bool(metrics["volume_converged"]) else "FAIL"
        embedding_qc_status = "PASS" if not hard_failures else "FAIL"
        overall_status = "VOLUME_PASS_EMBEDDING_QC_FAIL" if volume_status == "PASS" and hard_failures else volume_status
        rows.append(
            {
                "patient_id": patient["patient_id"],
                "timepoint": timepoint,
                "visit_day": float(visit["visit_day"]),
                "growth_label": patient["growth_label"],
                "growth_law_name": visit["growth_law_name"],
                "annual_volume_change_fraction": float(visit["growth_law_annual_volume_change_fraction"]),
                "target_volume_mm3": target,
                "actual_volume_mm3": float(metrics["actual_volume_mm3"]),
                "ravd": float(metrics["ravd"]),
                "volume_converged": bool(metrics["volume_converged"]),
                "volume_iterations": int(metrics["volume_iterations"]),
                "target_timepoint_voxels": int(metrics["target_timepoint_voxels"]),
                "qc_overlay_path": str(overlay_path.relative_to(REPO_ROOT)),
                "hard_failures": json.dumps(hard_failures),
                "warnings": json.dumps(warnings),
                "volume_status": volume_status,
                "embedding_qc_status": embedding_qc_status,
                "overall_status": overall_status,
            }
        )

    _write_csv(RESULT_CSV, rows)
    _write_report(REPORT_MD, rows)
    print(f"Wrote {RESULT_CSV}")
    print(f"Wrote {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
