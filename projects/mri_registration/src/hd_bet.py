"""
HD-BET Brain Extraction Script
-------------------------------
Reads the metadata CSV produced by 01_data_ingestion.ipynb (or the
enriched meta_df_entropy.csv from 02_quality_metrics.ipynb), stages
unprocessed images into hd_input/, runs HD-BET (batch with per-file
fallback), and writes todo/skipped manifests.

Aligned with VS_preprocessing.final.py:
  - Uses ALLOWED_STUDY_TYPES list filtering (not regex)
  - Reads 'image path' column from the organized folder structure
  - Outputs to hd_output/ where VS_preprocessing looks for _bet masks

Usage:
    python hd_bet_processing.py --meta_csv meta_df_entropy.csv
    python hd_bet_processing.py  # auto-finds meta_df_entropy.csv or most recent meta_df_*.csv
"""

import argparse
import gc
import glob
import os
import re
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pandas as pd


# ── Configuration ──
# Directory where 'raw_studies/', 'hd_input/', 'hd_output/' live.
# This is the notebook's cwd when it created the organized files.
# The CSV 'image path' column has relative paths like './raw_studies/t2_thin/...'
# which are resolved relative to this directory.
DATA_DIR = '/sfs/ceph/standard/gam_ai_group/600 qc1 scans/studies_20260130105647718474'
SEED = 42
THREADS = 16

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(THREADS)

HD_BET_PATH = '/home/mnn7bx/.conda/envs/nguyen_ants_env/bin/hd-bet'
DEVICE = "cuda"
DISABLE_TTA = True
SAVE_ONLY_MASK = True   # --save_bet_mask --no_bet_image (matching notebook)
KNOWN_OUTPUT_SUFFIXES = ['_bet', '_bet_mask', '_mask']
CLEAN_HD_INPUT_FIRST = True

# Study types to process — must match VS_preprocessing.final.py ALLOWED_STUDY_TYPES
ALLOWED_STUDY_TYPES = ['t2_thin', 't1+_thin', 't1+_thick_ax', 't1+_thick_cor']

# Chunked fallback: when a batch fails, retry in progressively smaller groups.
# The last element should always be 1 (per-file fallback).
BATCH_SIZES = [50, 10, 1]
# Seconds to wait between retry levels for RAM recovery
RETRY_COOLDOWN_SECS = 30


def find_meta_csv():
    """Find the metadata CSV: prefer meta_df_entropy.csv, then most recent meta_df_YYYY-MM-DD.csv."""
    # Prefer meta_df_entropy.csv (output of 02_quality_metrics, used by VS_preprocessing)
    if os.path.exists('meta_df_entropy.csv'):
        return 'meta_df_entropy.csv'
    # Fall back to most recent date-stamped meta_df
    candidates = sorted(glob.glob('meta_df_[0-9]*.csv'))
    if not candidates:
        raise FileNotFoundError(
            'No meta_df_entropy.csv or meta_df_*.csv found. '
            'Run 01_data_ingestion.ipynb (and optionally 02_quality_metrics.ipynb) first.'
        )
    return candidates[-1]


def strip_nii_ext(fn):
    """Return filename without .nii or .nii.gz extension."""
    if fn.endswith('.nii.gz'):
        return fn[:-7]
    if fn.endswith('.nii'):
        return fn[:-4]
    return fn


def expected_outputs(basename_noext, suffixes=None):
    """Build set of possible HD-BET output names."""
    suffixes = suffixes or KNOWN_OUTPUT_SUFFIXES
    return {f"{basename_noext}{s}.nii.gz" for s in suffixes}


def build_hd_bet_command(hd_bet_cmd, input_path, output_path):
    """Build HD-BET command line arguments."""
    args = [
        shlex.quote(hd_bet_cmd),
        "-i", shlex.quote(input_path),
        "-o", shlex.quote(output_path),
        "-device", shlex.quote(DEVICE),
    ]
    if DISABLE_TTA:
        args.append("--disable_tta")
    if SAVE_ONLY_MASK:
        args.extend(["--save_bet_mask", "--no_bet_image"])
    return " ".join(args)


def _stage_chunk(files, staging_dir):
    """Copy a list of files into a clean staging directory."""
    for p in Path(staging_dir).glob("*"):
        if p.is_file():
            p.unlink()
    for fp in files:
        if os.path.exists(fp):
            shutil.copy(fp, os.path.join(staging_dir, os.path.basename(fp)))


def _flush_ram():
    """Aggressively reclaim RAM and GPU memory."""
    gc.collect()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _parse_culprit(output_text):
    """Parse HD-BET output to find the file it was processing when it crashed.

    HD-BET prints 'Predicting <filename>:' before each file. The last such
    line before a crash identifies the problematic file.
    """
    matches = re.findall(r'Predicting\s+(\S+\.nii(?:\.gz)?)\s*:', output_text)
    if matches:
        return matches[-1]  # last file being predicted when crash happened
    return None


def _run_chunked(rows_todo, hd_bet_cmd, staging_dir, output_dir, batch_sizes):
    """
    Run HD-BET with progressive chunk sizes and automatic recovery.

    Processes files in chunks of batch_sizes[0]. When a chunk fails, the
    culprit file is excluded and the remaining files from that chunk are
    retried at progressively smaller batch sizes. Once the smaller retries
    finish, processing resumes at the original (largest) batch size for
    any remaining chunks — one bad file doesn't penalise the rest of the run.

    Returns a list of (file, error) tuples for files that failed.
    """
    all_files = [rt["CopiedTo"] for rt in rows_todo if os.path.exists(rt["CopiedTo"])]
    all_failures = []
    excluded = set()  # culprit files permanently excluded

    # ── Outer loop: always processes at the largest batch size ──
    remaining = list(all_files)
    pass_num = 0
    while remaining:
        pass_num += 1
        chunk_size = batch_sizes[0]
        chunks = [remaining[i:i + chunk_size]
                  for i in range(0, len(remaining), chunk_size)]

        print(f"\n[INFO] === Pass {pass_num}: {len(remaining)} files, "
              f"{len(chunks)} chunks of up to {chunk_size} ===")

        for ci, chunk in enumerate(chunks):
            _stage_chunk(chunk, staging_dir)
            cmd = build_hd_bet_command(hd_bet_cmd, staging_dir, output_dir)
            print(f"[INFO] Running HD-BET chunk {ci+1}/{len(chunks)} "
                  f"({len(chunk)} files, batch_size={chunk_size})")
            try:
                result = subprocess.run(cmd, shell=True, check=True,
                                        capture_output=True, text=True)
                print(result.stdout[-500:] if len(result.stdout) > 500
                      else result.stdout)
            except (subprocess.CalledProcessError, Exception) as e:
                # Drill down into smaller batches for this failed chunk
                failed, new_excluded = _retry_failed_chunk(
                    chunk, e, hd_bet_cmd, staging_dir, output_dir,
                    batch_sizes, 1)
                all_failures.extend(failed)
                excluded.update(new_excluded)
            gc.collect()

        # After processing all chunks, check what's still unfinished
        # (files that weren't culprits but may not have completed in
        # a crashed batch before the retry kicked in)
        existing_outputs = set(os.listdir(output_dir))
        still_pending = []
        for fp in all_files:
            if fp in excluded:
                continue
            base_noext = strip_nii_ext(os.path.basename(fp))
            if not (expected_outputs(base_noext) & existing_outputs):
                still_pending.append(fp)

        if still_pending and len(still_pending) < len(remaining):
            # Made progress — loop back to large batches
            remaining = still_pending
            print(f"[INFO] {len(remaining)} files still pending, "
                  f"looping back to batch_size={batch_sizes[0]}")
            _flush_ram()
            time.sleep(RETRY_COOLDOWN_SECS)
            _flush_ram()
        else:
            # No progress or nothing left — stop
            if still_pending:
                print(f"[WARN] {len(still_pending)} files made no progress, giving up")
                all_failures.extend(
                    (fp, "no_progress") for fp in still_pending
                    if not any(fp == f for f, _ in all_failures))
            break

    return all_failures


def _retry_failed_chunk(chunk, error, hd_bet_cmd, staging_dir, output_dir,
                         batch_sizes, level):
    """Handle a failed chunk: exclude culprit, retry at smaller batch sizes.

    Returns (failures, excluded) where failures is a list of (file, error)
    tuples and excluded is a set of culprit file paths to permanently skip."""
    failures = []
    excluded = set()

    # ── Identify the culprit file ──
    combined_output = ""
    if isinstance(error, subprocess.CalledProcessError):
        combined_output = (error.stdout or "") + (error.stderr or "")
    culprit_name = _parse_culprit(combined_output)

    culprit_path = None
    if culprit_name:
        for fp in chunk:
            if os.path.basename(fp) == culprit_name:
                culprit_path = fp
                break

    if culprit_path:
        print(f"[ERROR] Identified culprit: {culprit_name} — excluding from retries")
        failures.append((culprit_path, f"crashed_batch (batch_size={batch_sizes[level-1]})"))
        excluded.add(culprit_path)
    else:
        print(f"[WARN] Chunk failed (batch_size={batch_sizes[level-1]}), "
              f"could not identify culprit from output")

    # ── Collect unfinished files, minus culprit ──
    existing_outputs = set(os.listdir(output_dir))
    unfinished = []
    for fp in chunk:
        if fp == culprit_path:
            continue
        base_noext = strip_nii_ext(os.path.basename(fp))
        if not (expected_outputs(base_noext) & existing_outputs):
            unfinished.append(fp)

    if not unfinished:
        return failures, excluded

    # ── Aggressive RAM recovery + cooldown ──
    print(f"[INFO] Flushing RAM and waiting {RETRY_COOLDOWN_SECS}s before retry...")
    _flush_ram()
    time.sleep(RETRY_COOLDOWN_SECS)
    _flush_ram()

    # ── Drill down through remaining batch sizes ──
    for lvl in range(level, len(batch_sizes)):
        if not unfinished:
            break
        chunk_size = batch_sizes[lvl]
        sub_chunks = [unfinished[i:i + chunk_size]
                      for i in range(0, len(unfinished), chunk_size)]
        newly_unfinished = []

        next_label = (f"batch_size={batch_sizes[lvl+1]}"
                      if lvl + 1 < len(batch_sizes) else "exhausted")
        print(f"[INFO] Retrying {len(unfinished)} files at batch_size={chunk_size} "
              f"(next fallback: {next_label})")

        for sci, sub in enumerate(sub_chunks):
            if chunk_size == 1:
                fp = sub[0]
                base_noext = strip_nii_ext(os.path.basename(fp))
                out_fp = os.path.join(output_dir, f"{base_noext}.nii.gz")
                cmd = build_hd_bet_command(hd_bet_cmd, fp, out_fp)
                print(f"[INFO] Running HD-BET (single): {os.path.basename(fp)}")
                try:
                    subprocess.run(cmd, shell=True, check=True,
                                   capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Failed: {os.path.basename(fp)} (rc={e.returncode})")
                    failures.append((fp, e.returncode))
                except Exception as ex:
                    print(f"[ERROR] Failed: {os.path.basename(fp)} ({ex})")
                    failures.append((fp, str(ex)))
            else:
                _stage_chunk(sub, staging_dir)
                cmd = build_hd_bet_command(hd_bet_cmd, staging_dir, output_dir)
                print(f"[INFO] Retry sub-chunk {sci+1}/{len(sub_chunks)} "
                      f"({len(sub)} files, batch_size={chunk_size})")
                try:
                    subprocess.run(cmd, shell=True, check=True,
                                   capture_output=True, text=True)
                except (subprocess.CalledProcessError, Exception) as sub_e:
                    # Find culprit in this sub-chunk too
                    sub_output = ""
                    if isinstance(sub_e, subprocess.CalledProcessError):
                        sub_output = (sub_e.stdout or "") + (sub_e.stderr or "")
                    sub_culprit_name = _parse_culprit(sub_output)
                    sub_culprit_path = None
                    if sub_culprit_name:
                        for fp in sub:
                            if os.path.basename(fp) == sub_culprit_name:
                                sub_culprit_path = fp
                                break
                    if sub_culprit_path:
                        print(f"[ERROR] Identified culprit: {sub_culprit_name} — excluding")
                        failures.append((sub_culprit_path,
                                         f"crashed_batch (batch_size={chunk_size})"))
                        excluded.add(sub_culprit_path)

                    # Gather still-unfinished from this sub-chunk
                    existing_outputs = set(os.listdir(output_dir))
                    for fp in sub:
                        if fp == sub_culprit_path:
                            continue
                        base_noext = strip_nii_ext(os.path.basename(fp))
                        if not (expected_outputs(base_noext) & existing_outputs):
                            newly_unfinished.append(fp)

                    _flush_ram()
                    time.sleep(RETRY_COOLDOWN_SECS)
                    _flush_ram()
            gc.collect()

        unfinished = newly_unfinished

    # Anything still unfinished after all levels
    for fp in unfinished:
        failures.append((fp, "all_retries_exhausted"))

    return failures, excluded


def resolve_image_path(img_path, data_dir):
    """Resolve an image path from the CSV, which may be relative (./raw_studies/...)."""
    if not img_path or img_path == 'nan':
        return None
    # Already absolute and exists
    if os.path.isabs(img_path) and os.path.exists(img_path):
        return img_path
    # Relative path — resolve against data_dir
    resolved = os.path.normpath(os.path.join(data_dir, img_path))
    if os.path.exists(resolved):
        return resolved
    # Try stripping leading './' just in case
    if img_path.startswith('./'):
        resolved2 = os.path.normpath(os.path.join(data_dir, img_path[2:]))
        if os.path.exists(resolved2):
            return resolved2
    return None


def main(meta_csv=None, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
    raw_path = os.path.join(data_dir, 'hd_input')
    bet_path = os.path.join(data_dir, 'hd_output')

    print(f"[INFO] Data directory: {data_dir}")

    # Load metadata — search in cwd first, then data_dir
    if meta_csv is None:
        try:
            meta_csv = find_meta_csv()
        except FileNotFoundError:
            # Also search in data_dir
            for candidate in ['meta_df_entropy.csv'] + sorted(glob.glob(os.path.join(data_dir, 'meta_df_[0-9]*.csv'))):
                if candidate.startswith(data_dir):
                    if os.path.exists(candidate):
                        meta_csv = candidate
                        break
                elif os.path.exists(os.path.join(data_dir, candidate)):
                    meta_csv = os.path.join(data_dir, candidate)
                    break
            if meta_csv is None:
                raise FileNotFoundError(
                    f'No meta_df_entropy.csv or meta_df_*.csv found in cwd or {data_dir}. '
                    'Run the VS data timeline notebook first.'
                )
    print(f"[INFO] Loading metadata from: {meta_csv}")

    meta_df = pd.read_csv(meta_csv)
    initial_count = len(meta_df)

    # Drop rows without image paths
    meta_df = meta_df.dropna(subset=['image path'])
    print(f"[INFO] {initial_count} total rows, {len(meta_df)} with image paths")

    # Filter to allowed study types (matching VS_preprocessing.final.py)
    meta_df = meta_df[meta_df['Study Type'].isin(ALLOWED_STUDY_TYPES)]
    print(f"[INFO] {len(meta_df)} images after filtering to study types: {ALLOWED_STUDY_TYPES}")

    # Derive timeline columns (matching VS_preprocessing.final.py)
    if 'Patient_MRI_Days Tracker' in meta_df.columns:
        meta_df['_Patient_ID'] = meta_df['Patient_MRI_Days Tracker'].astype(str).str.split('_', n=1).str[0]
        meta_df['_Timepoint'] = meta_df['Patient_MRI_Days Tracker'].astype(str).str.split('_', n=1).str[1]
        n_patients = meta_df['_Patient_ID'].nunique()
        n_timepoints = meta_df['_Timepoint'].nunique()
        print(f"[INFO] {n_patients} patients, {n_timepoints} unique timepoints")
    elif 'Patient_ID' in meta_df.columns:
        meta_df['_Patient_ID'] = meta_df['Patient_ID'].astype(str)
        meta_df['_Timepoint'] = ''
        print(f"[INFO] {meta_df['_Patient_ID'].nunique()} patients")
    else:
        meta_df['_Patient_ID'] = ''
        meta_df['_Timepoint'] = ''

    # Resolve image paths against data_dir (CSV may have relative paths like ./raw_studies/...)
    meta_df['_Image_Path'] = meta_df['image path'].apply(
        lambda p: resolve_image_path(str(p), data_dir) if pd.notna(p) else None
    )
    meta_df['_Image'] = meta_df['image path'].astype(str).apply(os.path.basename)

    # Report how many paths resolved successfully
    resolved_count = meta_df['_Image_Path'].notna().sum()
    print(f"[INFO] {resolved_count}/{len(meta_df)} image paths resolved against data_dir")

    # Print study type distribution
    print(f"\n[INFO] Study type distribution:")
    for st, count in meta_df['Study Type'].value_counts().items():
        print(f"  {st}: {count}")

    # Prepare input/output dirs
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(bet_path, exist_ok=True)

    if CLEAN_HD_INPUT_FIRST:
        for p in Path(raw_path).glob("*"):
            if p.is_file():
                p.unlink()

    existing_files = set(os.listdir(bet_path)) if os.path.exists(bet_path) else set()

    # ── Decide: TODO vs SKIPPED ──
    rows_todo = []
    rows_skipped = []
    skip_counts = {'already_processed': 0, 'missing_file': 0, 'unsupported_extension': 0}

    print(f"\n=== Staging {len(meta_df)} files ===")
    print(f"Found {len(existing_files)} existing output files in {bet_path}")

    for idx, row in meta_df.iterrows():
        img_name = str(row["_Image"])
        img_path = row["_Image_Path"]  # None if unresolved
        patient_id = str(row.get("_Patient_ID", ""))
        timepoint = str(row.get("_Timepoint", ""))
        study_type = str(row.get("Study Type", ""))

        # NIfTI guard
        if not (img_name.endswith(".nii") or img_name.endswith(".nii.gz")):
            rs = {"Image": img_name, "Image_Path": img_path, "Patient_ID": patient_id,
                  "Timepoint": timepoint, "Study_Type": study_type, "Reason": "unsupported_extension"}
            rows_skipped.append(rs)
            skip_counts['unsupported_extension'] += 1
            continue

        # Already processed?
        base_noext = strip_nii_ext(img_name)
        matches = expected_outputs(base_noext) & existing_files
        if matches:
            rs = {"Image": img_name, "Image_Path": img_path, "Patient_ID": patient_id,
                  "Timepoint": timepoint, "Study_Type": study_type, "Reason": "already_processed",
                  "ExistingOutputsMatched": ";".join(sorted(matches))}
            rows_skipped.append(rs)
            skip_counts['already_processed'] += 1
            continue

        # Missing file? (_Image_Path is None if resolve_image_path failed)
        if not img_path:
            raw_csv_path = str(row.get("image path", ""))
            rs = {"Image": img_name, "Image_Path": raw_csv_path, "Patient_ID": patient_id,
                  "Timepoint": timepoint, "Study_Type": study_type, "Reason": "missing_file"}
            rows_skipped.append(rs)
            skip_counts['missing_file'] += 1
            print(f"[WARN] Missing: {img_name} (patient={patient_id}, tp={timepoint}) "
                  f"csv_path={raw_csv_path}")
            continue

        # Copy to hd_input
        dest = os.path.join(raw_path, img_name)
        shutil.copy(img_path, dest)
        rows_todo.append({"Image": img_name, "Image_Path": img_path, "CopiedTo": dest,
                          "Patient_ID": patient_id, "Timepoint": timepoint, "Study_Type": study_type})
        print(f"Staged {img_name} (patient={patient_id}, tp={timepoint}, {study_type})")

    todo_df = pd.DataFrame(rows_todo)
    skipped_df = pd.DataFrame(rows_skipped)

    # ── Summary ──
    print(f"\n=== Staging Summary ===")
    print(f"To process: {len(todo_df)}")
    print(f"Skipped:    {len(skipped_df)}")
    for reason, count in skip_counts.items():
        if count > 0:
            print(f"  - {reason}: {count}")

    if not todo_df.empty:
        print(f"\nTodo by study type:")
        for st, count in todo_df['Study_Type'].value_counts().items():
            print(f"  {st}: {count}")
        print(f"Todo patients: {todo_df['Patient_ID'].nunique()}")

    # Verify hd_input was actually filled
    staged_files = list(Path(raw_path).glob("*.nii*"))
    print(f"\n[INFO] hd_input contains {len(staged_files)} files")
    if len(staged_files) == 0 and not todo_df.empty:
        print("[ERROR] hd_input is empty despite having files to process!")
        print("[ERROR] Check that 'image path' column points to existing files.")

    # ── Write manifests ──
    ts = datetime.now().strftime("%Y%m")
    todo_df.to_csv(os.path.join(data_dir, f"hd_bet_todo_{ts}.csv"), index=False)
    skipped_df.to_csv(os.path.join(data_dir, f"hd_bet_skipped_{ts}.csv"), index=False)
    print(f"[INFO] Wrote manifests to {data_dir}: hd_bet_todo_{ts}.csv, hd_bet_skipped_{ts}.csv")

    # ── Run HD-BET ──
    if todo_df.empty:
        print("[INFO] Nothing to run. All images are either processed, missing, or unsupported.")
        return

    # Resolve hd-bet command
    if HD_BET_PATH and Path(HD_BET_PATH).exists():
        hd_bet_cmd = HD_BET_PATH
    else:
        hd_bet_cmd = shutil.which("hd-bet")
        if not hd_bet_cmd:
            raise FileNotFoundError(
                "hd-bet command not found. Set HD_BET_PATH or ensure it's on PATH."
            )

    all_failures = _run_chunked(rows_todo, hd_bet_cmd, raw_path, bet_path, BATCH_SIZES)

    try:
        shutil.rmtree(raw_path)
        print(f"[INFO] Removed temp folder: {raw_path}")
    except Exception as e:
        print(f"[WARN] Could not remove {raw_path}: {e}")
    gc.collect()

    if all_failures:
        fail_df = pd.DataFrame(all_failures, columns=["File", "Error"])
        fail_csv = os.path.join(data_dir, f"hd_bet_failures_{ts}.csv")
        fail_df.to_csv(fail_csv, index=False)
        print(f"[WARN] {len(all_failures)} total failures. Wrote: {fail_csv}")

    print(f"[SUMMARY] processed={len(todo_df)}, skipped={len(skipped_df)}, "
          f"failures={len(all_failures)}")


def _apply_config(config):
    """Override module-level config variables from a dict (loaded from YAML)."""
    g = globals()
    config_map = {
        'data_dir': 'DATA_DIR',
        'seed': 'SEED',
        'threads': 'THREADS',
        'hd_bet_path': 'HD_BET_PATH',
        'device': 'DEVICE',
        'disable_tta': 'DISABLE_TTA',
        'save_only_mask': 'SAVE_ONLY_MASK',
        'clean_hd_input_first': 'CLEAN_HD_INPUT_FIRST',
        'allowed_study_types': 'ALLOWED_STUDY_TYPES',
        'batch_sizes': 'BATCH_SIZES',
        'retry_cooldown_secs': 'RETRY_COOLDOWN_SECS',
    }
    for yaml_key, global_name in config_map.items():
        if yaml_key in config:
            g[global_name] = config[yaml_key]

    if 'threads' in config:
        os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config['threads'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HD-BET brain extraction")
    parser.add_argument("--meta_csv", type=str, default=None,
                        help="Path to metadata CSV (default: meta_df_entropy.csv or most recent meta_df_*.csv)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing raw_studies/, hd_input/, hd_output/ "
                             f"(default: {DATA_DIR})")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides other defaults)")
    args = parser.parse_args()

    if args.config:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from shared.config_loader import load_config
        config = load_config(args.config)
        _apply_config(config)

    main(meta_csv=args.meta_csv, data_dir=args.data_dir)
