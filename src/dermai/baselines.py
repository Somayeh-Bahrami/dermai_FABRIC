"""Model-free control heatmaps shared by the localization and faithfulness evaluations."""
from __future__ import annotations

import numpy as np

SIZE = 224


def center_prior_heatmap(size: int = SIZE, tie_break: bool = False) -> np.ndarray:
    """Fixed centered heatmap in [0, 1], peaking at the image center.

    Radial distance has many ties. tie_break=True subtracts a tiny row-major ramp
    (far below the smallest distance gap) so the pixel ordering used by
    deletion/insertion is well defined. Localization uses tie_break=False, the
    exact heatmap behind the reference center-prior numbers.
    """
    rows, cols = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2
    distance_squared = (rows - center) ** 2 + (cols - center) ** 2
    heatmap = 1.0 - distance_squared / distance_squared.max()
    if not tie_break:
        return heatmap
    ramp = np.arange(size * size, dtype=np.float64).reshape(size, size) / (size * size)
    return heatmap - 1e-9 * ramp


def random_heatmap(size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random((size, size)).astype(np.float32)
