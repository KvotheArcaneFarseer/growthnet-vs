"""
Napari-based 3D animation of synthetic lollipop tumor growth.

Renders each timepoint in napari's 3D isosurface view with a fixed oblique
camera, captures screenshots, and saves as GIF and (optionally) MP4.

Usage
-----
From the GrowthNet repo root::

    python make_lollipop_napari.py

Dependencies
------------
    pip install "napari[all]>=0.4.17" Pillow imageio imageio-ffmpeg

Output
------
    animation_outputs/lollipop_napari.gif
    animation_outputs/lollipop_napari.mp4  (requires imageio-ffmpeg)

Notes
-----
- napari >= 0.4.17 is required for ``Viewer(show=False)`` (headless rendering).
- On macOS this script must be run from a normal terminal, not inside a
  Jupyter kernel, because Qt requires the main thread.
- If you see a blank/black frame, increase SETTLE_MS to give the GPU more
  time to finish the first render.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ── repo root on sys.path so project imports resolve ─────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.vivit.src.data.synthetic import create_synthetic_time_3d

# ── run parameters ────────────────────────────────────────────────────────────
DATES: List[int] = [0, 50, 100, 150, 200]

# ── output paths ──────────────────────────────────────────────────────────────
OUT_DIR = _REPO_ROOT / "animation_outputs"
GIF_PATH = OUT_DIR / "lollipop_napari.gif"
MP4_PATH = OUT_DIR / "lollipop_napari.mp4"

# ── camera / rendering ────────────────────────────────────────────────────────
# Oblique side-on view: lollipop grows along depth (c-axis / z).
# Angles are (elevation°, azimuth°, roll°) in napari's convention.
CAMERA_ANGLES: Tuple[float, float, float] = (25.0, -55.0, 0.0)
CAMERA_ZOOM: float = 5.0
VOLUME_CENTER: Tuple[float, float, float] = (48.0, 48.0, 48.0)

# Isosurface threshold for the 0/1 mask (any value strictly between 0 and 1)
ISO_THRESHOLD: float = 0.5

# How long to wait for the Qt render to finish before screenshotting (ms)
SETTLE_MS: int = 700

# GIF frame duration in ms
GIF_FRAME_MS: int = 900


# ── data generation ───────────────────────────────────────────────────────────
def generate_data() -> Tuple[List[np.ndarray], List[np.ndarray]]:
    print("Generating synthetic lollipop sequence...")
    images, labels = create_synthetic_time_3d(
        height=96,
        width=96,
        depth=96,
        dates=DATES,
        rotation_degrees=[0, 0, 0],
        geometry_mode="lollipop",
        canal_axis="c",
        rad_max=40,
        growth="steady",
    )
    for i, lab in enumerate(labels):
        print(f"  t{i}  day={DATES[i]:3d}  mask_voxels={int(np.sum(lab > 0))}")
    return images, labels


# ── napari rendering ──────────────────────────────────────────────────────────
def _apply_camera(viewer) -> None:  # type: ignore[no-untyped-def]
    """Pin the camera to the fixed oblique view."""
    viewer.camera.angles = CAMERA_ANGLES
    viewer.camera.zoom = CAMERA_ZOOM
    viewer.camera.center = VOLUME_CENTER


def capture_napari_frames(labels: List[np.ndarray]) -> List[np.ndarray]:
    """
    Open a napari 3D viewer, render each label timepoint, and
    return a list of RGBA uint8 screenshots (H, W, 4).

    The viewer must be shown (show=True) so the Qt widget receives an expose
    event and the OpenGL pipeline actually runs before screenshot() is called.
    With show=False the GL buffer stays black regardless of settle time.
    """
    try:
        import napari  # noqa: PLC0415
    except ImportError as exc:
        raise SystemExit(
            "napari is not installed. Run:  pip install 'napari[all]>=0.4.17'"
        ) from exc

    try:
        from qtpy.QtWidgets import QApplication  # type: ignore[import]
        _has_qtpy = True
    except ImportError:
        _has_qtpy = False

    print("Opening napari viewer...")
    viewer = napari.Viewer(show=True)
    viewer.dims.ndisplay = 3
    _apply_camera(viewer)

    screenshots: List[np.ndarray] = []

    try:
        for i, label in enumerate(labels):
            # Swap out the single image layer each iteration
            viewer.layers.clear()

            viewer.add_image(
                label.astype(np.float32),
                name=f"day {DATES[i]}",
                colormap="cyan",
                rendering="iso",
                iso_threshold=ISO_THRESHOLD,
                contrast_limits=[0.0, 1.0],
                opacity=1.0,
            )

            # Re-apply camera and 3D display mode after clear() (both can reset)
            viewer.dims.ndisplay = 3
            _apply_camera(viewer)

            # Give Qt time to process events and complete the render
            if _has_qtpy:
                QApplication.processEvents()
            time.sleep(SETTLE_MS / 1000.0)

            # Force a synchronous GL render: repaint() blocks until the
            # QOpenGLWidget has finished painting, so glReadPixels inside
            # screenshot() sees the current frame, not a stale one.
            viewer.window.qt_viewer.canvas.native.repaint()
            if _has_qtpy:
                QApplication.processEvents()

            shot = viewer.screenshot(canvas_only=True)
            screenshots.append(shot.copy())  # .copy() guards against buffer aliasing
            print(f"  Frame {i + 1}/{len(labels)} captured  (day {DATES[i]})")

    finally:
        viewer.close()

    return screenshots


# ── output helpers ─────────────────────────────────────────────────────────────
def save_gif(screenshots: List[np.ndarray], path: Path) -> None:
    try:
        from PIL import Image  # type: ignore[import]
    except ImportError as exc:
        raise SystemExit(
            "Pillow is not installed. Run:  pip install Pillow"
        ) from exc

    # Drop alpha channel — Pillow GIF writer works best with RGB
    frames = [Image.fromarray(s[..., :3]) for s in screenshots]
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=GIF_FRAME_MS,
        loop=0,
    )
    print(f"Saved GIF:  {path}")


def save_mp4(screenshots: List[np.ndarray], path: Path) -> None:
    try:
        import imageio  # type: ignore[import]
    except ImportError:
        print(
            "Skipping MP4: imageio not installed.\n"
            "  Install with:  pip install imageio imageio-ffmpeg"
        )
        return

    fps = 1000.0 / GIF_FRAME_MS
    rgb_frames = [s[..., :3] for s in screenshots]

    try:
        # macro_block_size=1 avoids the "width/height not divisible by 16" error
        imageio.mimsave(str(path), rgb_frames, fps=fps, macro_block_size=1)
        print(f"Saved MP4:  {path}")
    except Exception as exc:
        print(f"MP4 export failed ({type(exc).__name__}): {exc}")
        print("  Make sure imageio-ffmpeg is installed: pip install imageio-ffmpeg")


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    _, labels = generate_data()

    screenshots = capture_napari_frames(labels)

    print("Saving outputs...")
    save_gif(screenshots, GIF_PATH)
    save_mp4(screenshots, MP4_PATH)

    print(f"\nDone.\n  GIF → {GIF_PATH}\n  MP4 → {MP4_PATH}")


if __name__ == "__main__":
    main()
