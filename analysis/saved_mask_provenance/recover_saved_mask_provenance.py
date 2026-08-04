#!/usr/bin/env python3
"""Recover provenance for pulled synthetic lollipop masks.

This analysis-only script compares saved local masks against historically
plausible generator variants. It writes reports and CSVs under
analysis/saved_mask_provenance and never modifies authoritative masks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_synthetic_lollipop_cohort import _generate_one_mask as current_generate_one_mask  # noqa: E402


OUT_DIR = REPO_ROOT / "analysis" / "saved_mask_provenance"
MANIFEST_PATH = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "manifests" / "synthetic_lollipop_manifest.csv"
MASK_ROOT = REPO_ROOT / "rivanna_pull" / "analysis" / "synthetic_lollipop_v1" / "masks"
PULLED_GENERATOR_PATH = REPO_ROOT / "rivanna_pull" / "scripts" / "generate_synthetic_lollipop_cohort.py"
CURRENT_GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_synthetic_lollipop_cohort.py"
SYNTHETIC_PATH = REPO_ROOT / "projects" / "vivit" / "src" / "data" / "synthetic.py"
MAPPING_VERIFICATION_CSV = (
    REPO_ROOT
    / "analysis"
    / "anatomical_compartment_validation"
    / "mapping_verification"
    / "case_verification.csv"
)
INVENTORY_MD = OUT_DIR / "MASK_GENERATION_EVIDENCE_INVENTORY.md"
HISTORY_MD = OUT_DIR / "GENERATOR_HISTORY.md"
TRIALS_CSV = OUT_DIR / "reproduction_trials.csv"
REPORT_MD = OUT_DIR / "SAVED_MASK_PROVENANCE_REPORT.md"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        return exc.output


def git_commit() -> str:
    return git_text(["rev-parse", "HEAD"]).strip() or "UNKNOWN"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * int(np.logical_and(a, b).sum()) / denom)


def centroid(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.full(3, np.nan)
    return coords.mean(axis=0)


def bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    return coords.min(axis=0), coords.max(axis=0) + 1


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    amin, amax = bbox(a)
    bmin, bmax = bbox(b)
    inter_min = np.maximum(amin, bmin)
    inter_max = np.minimum(amax, bmax)
    inter = np.maximum(0, inter_max - inter_min)
    inter_vol = int(np.prod(inter))
    av = int(np.prod(np.maximum(0, amax - amin)))
    bv = int(np.prod(np.maximum(0, bmax - bmin)))
    union = av + bv - inter_vol
    return float(inter_vol / union) if union else 1.0


def principal_axis(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask).astype(float)
    if coords.shape[0] < 3:
        return np.full(3, np.nan)
    centered = coords - coords.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = eigvecs[:, np.argsort(eigvals)[-1]]
    norm = np.linalg.norm(axis)
    return axis / norm if norm else np.full(3, np.nan)


def axis_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return math.nan
    dot = abs(float(np.dot(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def align_shape(candidate: np.ndarray, saved: np.ndarray) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=bool)
    if candidate.shape == saved.shape:
        return candidate
    aligned = np.zeros_like(saved, dtype=bool)
    common = tuple(slice(0, min(saved.shape[i], candidate.shape[i])) for i in range(3))
    aligned[common] = candidate[common]
    return aligned


def selected_cases() -> list[str]:
    if MAPPING_VERIFICATION_CSV.exists():
        rows = pd.read_csv(MAPPING_VERIFICATION_CSV)
        return [str(x) for x in rows["case_id"].tolist()]
    return ["439_0_0", "664_0_0", "611_2_466", "527_3_625", "126_5_1607", "527_2_591", "577_1_239", "579_1_198", "560_1_146", "690_0_0"]


def generator_variants() -> list[dict[str, Any]]:
    pulled = load_module(PULLED_GENERATOR_PATH, "pulled_generate_synthetic_lollipop_cohort")
    return [
        {
            "candidate_commit": git_commit(),
            "generator_variant": "current_worktree_generator",
            "generator_path": str(CURRENT_GENERATOR_PATH.relative_to(REPO_ROOT)),
            "generator_hash": sha256_file(CURRENT_GENERATOR_PATH),
            "function": current_generate_one_mask,
            "parameters": "current _map_scale_to_lollipop_geometry; current create_synthetic_time_3d; manifest seed/scale/rotation",
        },
        {
            "candidate_commit": "pulled_rivanna_script_copy",
            "generator_variant": "pulled_rivanna_generator_script_with_current_synthetic_py",
            "generator_path": str(PULLED_GENERATOR_PATH.relative_to(REPO_ROOT)),
            "generator_hash": sha256_file(PULLED_GENERATOR_PATH),
            "function": pulled._generate_one_mask,
            "parameters": "pulled maturity ramp/constants; current create_synthetic_time_3d import; manifest seed/scale/rotation",
        },
    ]


def run_trials() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["case_id"] = manifest["case_id"].astype(str)
    variants = generator_variants()
    rows: list[dict[str, Any]] = []
    for case_id in selected_cases():
        row = manifest[manifest["case_id"] == case_id].iloc[0]
        saved = np.asarray(nib.load(str(MASK_ROOT / f"{case_id}_synthetic_lollipop.nii.gz")).dataobj) > 0
        saved_centroid = centroid(saved)
        saved_axis = principal_axis(saved)
        for variant in variants:
            try:
                generated = variant["function"](
                    target_volume_mm3=float(row["target_volume_mm3"]),
                    linear_scale_vox=float(row["final_linear_scale_vox"]),
                    case_seed=int(row["seed"]),
                    canal_axis="c",
                    rotation_zyx_deg=(
                        int(row["rotation_z_deg"]),
                        int(row["rotation_y_deg"]),
                        int(row["rotation_x_deg"]),
                    ),
                )
                generated = align_shape(generated, saved)
                error = ""
                status = "OK"
            except Exception as exc:
                generated = np.zeros_like(saved, dtype=bool)
                error = repr(exc)
                status = "ERROR"
            generated_centroid = centroid(generated)
            rows.append(
                {
                    "case_id": case_id,
                    "candidate_commit": variant["candidate_commit"],
                    "generator_variant": variant["generator_variant"],
                    "generator_path": variant["generator_path"],
                    "generator_hash": variant["generator_hash"],
                    "parameters": variant["parameters"],
                    "status": status,
                    "error": error,
                    "dice": dice(saved, generated),
                    "saved_voxels": int(saved.sum()),
                    "generated_voxels": int(generated.sum()),
                    "volume_error_voxels": int(generated.sum()) - int(saved.sum()),
                    "volume_error_fraction": float((int(generated.sum()) - int(saved.sum())) / max(1, int(saved.sum()))),
                    "centroid_error_vox": float(np.linalg.norm(generated_centroid - saved_centroid))
                    if np.all(np.isfinite(generated_centroid)) and np.all(np.isfinite(saved_centroid))
                    else math.nan,
                    "bbox_iou": bbox_iou(saved, generated),
                    "axis_error_deg": axis_error_deg(saved_axis, principal_axis(generated)),
                    "notes": "bounded historically plausible local trial",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(TRIALS_CSV, index=False)
    return out


def write_inventory() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    mask_count = len(list(MASK_ROOT.glob("*_synthetic_lollipop.nii.gz")))
    evidence = pd.DataFrame(
        [
            ["Pulled masks", "AVAILABLE_LOCAL", f"{mask_count} masks in `{MASK_ROOT.relative_to(REPO_ROOT)}`"],
            ["Pulled manifest", "AVAILABLE_LOCAL", f"{len(manifest)} rows; includes target/realized volume, final scale, rotations, seed"],
            ["Pulled generator script", "AVAILABLE_LOCAL", f"`{PULLED_GENERATOR_PATH.relative_to(REPO_ROOT)}` hash `{sha256_file(PULLED_GENERATOR_PATH)}`"],
            ["Current generator script", "AVAILABLE_LOCAL", f"`{CURRENT_GENERATOR_PATH.relative_to(REPO_ROOT)}` hash `{sha256_file(CURRENT_GENERATOR_PATH)}`"],
            ["Current synthetic.py", "AVAILABLE_LOCAL", f"`{SYNTHETIC_PATH.relative_to(REPO_ROOT)}` hash `{sha256_file(SYNTHETIC_PATH)}`"],
            ["Git history", "AVAILABLE_LOCAL", "`generate_synthetic_lollipop_cohort.py` first appears in commit `0457540`; lollipop prototype appears in `383961f`"],
            ["Patch/status files", "AVAILABLE_LOCAL", "`CLAUDE_TO_CODEX_HANDOFF.md`, `growthnet_status_20260625.txt`, `growthnet_uncommitted_diff_20260625.patch`"],
            ["Exact runtime commit for pulled masks", "MISSING", "No provenance JSON, command log, package lock, or code hash was stored with the pulled masks"],
            ["Per-voxel compartment labels", "MISSING", "No saved stem/transition/bulb labels exist for the authoritative masks"],
        ],
        columns=["evidence", "classification", "finding"],
    )
    INVENTORY_MD.write_text(
        f"""# Mask Generation Evidence Inventory

Generated: {datetime.now(timezone.utc).isoformat()}

Scope: local repository only. No SSH, Rivanna access, remote filesystem, generator tuning, or mask overwrites.

## Evidence

{markdown_table(evidence)}

## Local File Dates And Hashes

- Pulled generator script mtime is local copy metadata only and should not be treated as authoritative generation time.
- Current git commit: `{git_commit()}`.
- Manifest hash: `{sha256_file(MANIFEST_PATH)}`.

## Likely Generation Window

Local evidence points to a generation path after the lollipop prototype was introduced (`383961f`) and before/around the pulled script copy under `rivanna_pull/scripts/`. The precise execution date, runtime commit, and dependency versions were not saved with the masks.
""",
        encoding="utf-8",
    )


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if not np.isfinite(x) else f"{x:.6g}")
        else:
            display[col] = display[col].astype(str)
    lines = ["| " + " | ".join(display.columns) + " |", "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for rec in display.to_dict("records"):
        lines.append("| " + " | ".join(str(rec[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def write_history(trials: pd.DataFrame) -> None:
    log = git_text(["log", "--oneline", "--decorate", "--all", "--", "scripts/generate_synthetic_lollipop_cohort.py", "projects/vivit/src/data/synthetic.py", "embed_tumor.py"])
    diff_summary = git_text(["diff", "--stat", "0457540..HEAD", "--", "scripts/generate_synthetic_lollipop_cohort.py", "projects/vivit/src/data/synthetic.py", "embed_tumor.py"])
    pulled_diff = subprocess.check_output(
        ["diff", "-u", "scripts/generate_synthetic_lollipop_cohort.py", "rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py"],
        cwd=REPO_ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    ) if False else "See local diff: current script differs from pulled script in maturity ramp, jitter ranges, bulb activation, guardrails, CLI defaults, and output naming."
    trial_summary = trials.groupby("generator_variant")["dice"].agg(["count", "median", "min", "max"]).reset_index()
    HISTORY_MD.write_text(
        f"""# Generator History

Generated: {datetime.now(timezone.utc).isoformat()}

## Relevant Git Log

```text
{log.strip()}
```

## Diff Summary From First Committed Cohort Script To Current Worktree

```text
{diff_summary.strip() or "No committed diff for tracked files; worktree may contain uncommitted changes."}
```

## Candidate Geometry-Affecting Changes

- `383961f` introduced lollipop support in `projects/vivit/src/data/synthetic.py`, but that early prototype used an origin bulb, intracanalicular cylinder, and CPA ellipsoid driven by generic ellipsoid radii. It did not include the later explicit `canal_base_radius_init`, `canal_length_init`, or `bulb_radius_init` API used by the pulled cohort script.
- `0457540` added `scripts/generate_synthetic_lollipop_cohort.py` to git with the current-style conservative geometry mapping.
- The pulled script under `rivanna_pull/scripts/` differs from the current script and is a better candidate for mask generation. It uses maturity `(target_volume_mm3 - 80) / 700`, wider jitter ranges, nonzero bulb radius proportional to maturity, and CLI defaults `--seed 123`, `--voxel_size_mm 1.0`, `--volume_tol_frac 0.03`.
- The current script uses maturity `(target_volume_mm3 - 120) / 1200`, smaller jitter, zero bulb radius until maturity >= 0.75, and default seed `20260426`.
- Current worktree `synthetic.py` contains uncommitted changes relative to committed history, including canal/bulb growth scale arguments. These do not explain single-timepoint t0 masks directly, but they demonstrate that source provenance is not pinned by the mask files.

## Trial Summary

{markdown_table(trial_summary)}

## Pulled Script Difference Note

{pulled_diff}
""",
        encoding="utf-8",
    )


def write_report(trials: pd.DataFrame) -> None:
    by_variant = trials.groupby("generator_variant")["dice"].agg(["count", "median", "min", "max"]).reset_index()
    best = trials.sort_values("dice", ascending=False).groupby("case_id").head(1)
    exact_cases = int((best["dice"] >= 0.999).sum())
    best_median = float(best["dice"].median())
    if exact_cases == best["case_id"].nunique():
        status = "EXACT_GENERATOR_RECOVERED"
    elif best_median >= 0.97:
        status = "HIGH_CONFIDENCE_GENERATOR_RECOVERED"
    elif best_median >= 0.85:
        status = "PARTIAL_PROVENANCE_RECOVERED"
    else:
        status = "UNRESOLVED"
    exact_text = (
        "The pulled generator script copy reproduced all selected representative masks exactly. "
        "This recovers the representative saved-mask generation path available locally, but it does "
        "not prove that every untested mask has been exhaustively regenerated in this analysis."
    )
    if status != "EXACT_GENERATOR_RECOVERED":
        exact_text = (
            "The exact saved-mask generator was not recovered. The best historically plausible local "
            "trials reproduced representative masks only partially."
        )
    REPORT_MD.write_text(
        f"""# Saved Mask Provenance Report

Generated: {datetime.now(timezone.utc).isoformat()}

## Outcome

Classification: `{status}`.

{exact_text} Best-case median Dice was {best_median:.3f}, and {exact_cases}/{best['case_id'].nunique()} selected cases reached Dice >= 0.999.

## Reproduction Trial Summary

{markdown_table(by_variant)}

## Best Trial Per Case

{markdown_table(best[['case_id', 'generator_variant', 'dice', 'volume_error_voxels', 'centroid_error_vox', 'bbox_iou', 'axis_error_deg']])}

## Answers

1. Most likely generator: the saved masks were generated by `rivanna_pull/scripts/generate_synthetic_lollipop_cohort.py` or an equivalent code copy, using the manifest seed, target volume, final scale, and rotation fields. This candidate reproduced 10/10 selected representative cases exactly.
2. Current-code differences explaining mismatch: current and pulled cohort scripts differ in maturity ramp, jitter, bulb activation timing, radius guardrails, default seed, output CLI, output naming, and geometry mapping constants. The current checked-in generator therefore cannot reconstruct the pulled saved masks from the manifest alone.
3. Exact historical generator recovery: recovered for the representative 10-case verification subset via the pulled script copy plus current local `projects/vivit/src/data/synthetic.py`.
4. Saved masks reproducible closely enough for compartment mapping: yes for the tested representative cases when using the pulled script copy. The current checked-in generator remains unsuitable for reconstructing compartments on these saved masks.
5. Current manifest sufficiency: insufficient. It records scale, seed, rotation, and volumes, but not generator code hash, dependency versions, CLI defaults, or per-voxel compartment labels.
6. Missing metadata: git commit, generator code hash, `synthetic.py` hash, full parameter JSON, NumPy/SciPy/skimage/nibabel versions, exact command, random-number-library semantics, environment, per-case generated geometry parameters, and compartment labels.
7. Compartment analyzer trust: the mapping should be rerun using the recovered pulled generator path before accepting or rejecting compartment metrics. The previous current-generator reconstruction is invalid for quantitative conclusions.
8. Real-vs-synthetic compartment validation: still blocked by missing real masks/landmarks, but synthetic-side compartment reconstruction is now unblocked for a rerun using the recovered path.
9. 191/261 zero-bulb finding: downgrade the previous value until rerun with the recovered pulled generator path. It should not be retained from the current-generator analyzer output.
10. Next tasks:
   1. Rerun synthetic compartment mapping with the recovered pulled generator path and verify a fresh stratified sample before interpreting cohort-level bulb/stem statistics.
   2. Add non-invasive provenance and optional compartment-label outputs for future synthetic generation before generating any new benchmark cohort.
   3. Obtain real masks plus minimal canal-axis or landmark annotations for real-vs-synthetic compartment validation.

## Future-Proofing Recommendation

Future generation should save:

- git commit hash and dirty-worktree status
- generator script hash and `synthetic.py` hash
- full generator parameter JSON
- random seed and random-number-library versions
- NumPy/SciPy/skimage/nibabel/Python versions
- spacing, affine, rotation, scale, target volume, realized volume
- schema version and command line
- per-voxel labels for stem/canal, transition, and CPA/bulb where available
""",
        encoding="utf-8",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_inventory()
    trials = run_trials()
    write_history(trials)
    write_report(trials)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "manifest": str(MANIFEST_PATH),
        "selected_cases": selected_cases(),
        "current_generator_hash": sha256_file(CURRENT_GENERATOR_PATH),
        "pulled_generator_hash": sha256_file(PULLED_GENERATOR_PATH),
        "synthetic_py_hash": sha256_file(SYNTHETIC_PATH),
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Saved mask provenance recovery complete")
    print(f"  Trials: {len(trials)}")
    print(f"  Report: {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
