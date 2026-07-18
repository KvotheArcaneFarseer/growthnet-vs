#!/usr/bin/env python3
"""
Extract representative frames from napari 3D tumor MP4 videos.

Outputs PNGs to lab_meeting_exports/3d_frames/.
These serve as the "3D screenshots" for the PowerPoint deck.

Usage:
    python3 scripts/export_3d_frames.py
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ANIM_DIR = REPO_ROOT / "animation_outputs"
OUT_DIR = REPO_ROOT / "lab_meeting_exports" / "3d_frames"

SOURCES = [
    ("napari_tumor_angle1.mp4", "3d_angle1"),
    ("napari_tumor_angle2.mp4", "3d_angle2"),
]

FRAME_INDEX = 3  # middle frame (videos have ~7 frames each)


def extract_frame(mp4_path: Path, out_path: Path, frame_idx: int = FRAME_INDEX) -> bool:
    if not mp4_path.exists():
        print(f"  SKIP (missing): {mp4_path.name}")
        return False
    if mp4_path.stat().st_size < 1024:
        print(f"  SKIP (empty): {mp4_path.name}")
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vframes", "1",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try with different frame selection syntax
        cmd2 = ["ffmpeg", "-y", "-i", str(mp4_path),
                "-vf", f"thumbnail=n=7", "-vframes", "1", str(out_path)]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)
        if result2.returncode != 0:
            print(f"  FAIL: {mp4_path.name}\n    {result.stderr[-200:]}")
            return False
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  OK: {out_path.name} ({out_path.stat().st_size // 1024}KB)")
        return True
    print(f"  FAIL: output empty for {mp4_path.name}")
    return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Extracting 3D frames to {OUT_DIR}/")
    results = {}
    for mp4_name, out_stem in SOURCES:
        mp4 = ANIM_DIR / mp4_name
        out = OUT_DIR / f"{out_stem}.png"
        results[out_stem] = extract_frame(mp4, out)

    successes = sum(results.values())
    print(f"\nDone: {successes}/{len(SOURCES)} frames extracted")
    if successes == 0:
        print("WARNING: No 3D frames extracted — deck will skip 3D slide")
        sys.exit(1)


if __name__ == "__main__":
    main()
