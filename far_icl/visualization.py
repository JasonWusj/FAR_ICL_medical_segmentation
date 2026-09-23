"""Case-level interpretation panels from observed predictions and model outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_correction_explanation(path, data, initial, uncertainty, final, support, regional, grid):
    """Show actual masks beside the selected set's predicted repair/harm cover.

    The maps are model predictions for independent candidates. They are not
    observed query errors, and they do not measure final set performance.
    """
    if regional.ndim != 3 or regional.shape[1:] != (grid * grid, 2) or len(regional) == 0:
        raise ValueError("Expected selected repair/harm predictions [K, grid², 2]")
    maps = regional.detach().cpu().numpy().max(axis=0).reshape(grid, grid, 2)
    fig, axes = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)
    rgb = data["rgb"].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    values = (
        (rgb, "Query image", None),
        (initial.detach().cpu().squeeze().numpy(), "Coarse mask probability", "gray"),
        (uncertainty.detach().cpu().squeeze().numpy(), "Context uncertainty", "magma"),
        (final.detach().cpu().squeeze().numpy(), "Final mask probability", "gray"),
        (maps[:, :, 0], "Predicted repair coverage", "Greens"),
        (maps[:, :, 1], "Predicted harm exposure", "Reds"),
        (support["image"].detach().cpu().squeeze().numpy(), "First selected support", "gray"),
        (support["mask"].detach().cpu().squeeze().numpy(), "Its ground-truth mask", "gray"),
    )
    for ax, (image, title, cmap) in zip(axes.flat, values):
        if title == "Context uncertainty":
            vmax = max(float(np.max(image)), 1e-8)
        else:
            vmax = 1.0
        ax.imshow(
            image, cmap=cmap, vmin=0 if cmap else None, vmax=vmax if cmap else None, interpolation="nearest"
        )
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Repair/harm maps are predictions; inspect selected IDs in per_case.json", fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
