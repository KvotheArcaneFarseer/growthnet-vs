from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from projects.vivit.src.data.synthetic import create_synthetic_time_3d


def crop_2d(img: np.ndarray, pad: int = 6, out_size: int = 64) -> np.ndarray:
    coords = np.argwhere(img > 0)
    if len(coords) == 0:
        return np.zeros((out_size, out_size), dtype=img.dtype)

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1

    y0 = max(mins[0] - pad, 0)
    y1 = min(maxs[0] + pad, img.shape[0])
    x0 = max(mins[1] - pad, 0)
    x1 = min(maxs[1] + pad, img.shape[1])

    cropped = img[y0:y1, x0:x1]

    h, w = cropped.shape
    canvas = np.zeros((out_size, out_size), dtype=img.dtype)

    y_start = max((out_size - h) // 2, 0)
    x_start = max((out_size - w) // 2, 0)

    h_use = min(h, out_size)
    w_use = min(w, out_size)

    canvas[y_start:y_start + h_use, x_start:x_start + w_use] = cropped[:h_use, :w_use]
    return canvas


def make_views(mask: np.ndarray) -> List[np.ndarray]:
    sx = mask.shape[0] // 2
    sy = mask.shape[1] // 2
    sz = mask.shape[2] // 2

    sagittal = crop_2d(mask[sx, :, :])
    coronal = crop_2d(mask[:, sy, :])
    axial = crop_2d(mask[:, :, sz])

    return [sagittal, coronal, axial]


def main() -> None:
    out_dir = Path("animation_outputs")
    out_dir.mkdir(exist_ok=True)

    images, labels = create_synthetic_time_3d(
        height=96,
        width=96,
        depth=96,
        dates=[0, 50, 100, 150, 200],
        rotation_degrees=[0, 0, 0],
        geometry_mode="lollipop",
        canal_axis="c",
        rad_max=40,
        growth="steady",
    )

    frames = [make_views(lab.astype(np.float32)) for lab in labels]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    titles = ["Sagittal", "Coronal", "Axial"]

    ims = []
    for ax, title, view in zip(axes, titles, frames[0]):
        im = ax.imshow(view, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        ims.append(im)

    suptitle = fig.suptitle("Synthetic Tumor Growth - t0", fontsize=14)

    def update(i: int):
        for im, view in zip(ims, frames[i]):
            im.set_data(view)
        suptitle.set_text(f"Synthetic Tumor Growth - t{i}")
        return ims + [suptitle]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=900,
        blit=False,
        repeat=True,
    )

    gif_path = out_dir / "lollipop_growth_3view.gif"
    anim.save(gif_path, writer=PillowWriter(fps=1.2))
    print(f"Saved GIF to: {gif_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()