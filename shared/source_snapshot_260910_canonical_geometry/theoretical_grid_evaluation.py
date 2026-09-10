"""Experimental image sampling on FFT-defined theoretical hexagonal grids.

This module intentionally generates grid positions from image FFT phase/orientation
before evaluating pixels.  Peak detections may be used only for a separate
diagnostic (distance-field sensitivity), never to choose the grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from lattice_indexing import HexLattice, grid_coordinates
from registration import sample_contrast


def sample_grid_features(image, lattice: HexLattice, margin=30):
    """Sample intensity, local contrast, and a simple centre-vs-ring shape score."""
    height, width = image.shape
    indices, coordinates = grid_coordinates(lattice, width, height, margin=margin)
    sampled = sample_contrast(image, coordinates)
    arr = np.asarray(image, float) / 65535.0
    x, y = np.rint(coordinates).astype(int).T
    centre = np.full(len(x), np.nan)
    ring = np.full(len(x), np.nan)
    for row, (px, py) in enumerate(zip(x, y)):
        patch = arr[py-3:py+4, px-3:px+4]
        if patch.shape != (7, 7):
            continue
        centre[row] = patch[2:5, 2:5].mean()
        ring[row] = (patch.sum()-patch[2:5, 2:5].sum())/40
    out = pd.DataFrame({"m": indices[:, 0], "n": indices[:, 1],
                        "x": coordinates[:, 0], "y": coordinates[:, 1]})
    out = pd.concat((out, sampled.reset_index(drop=True)), axis=1)
    out["centre_intensity"] = centre
    out["ring_intensity"] = ring
    out["centre_minus_ring"] = centre-ring
    return out


def nearest_grid_distance(points, lattice: HexLattice):
    """Continuous Euclidean distance from points to the true nearest hex cell."""
    points = np.asarray(points, float)
    inv_basis = np.linalg.inv(lattice.basis)
    central = np.rint((points-lattice.origin) @ inv_basis.T).astype(int)
    offsets = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)])
    candidates = central[:, None, :] + offsets
    coords = lattice.origin + candidates @ lattice.basis.T
    return np.linalg.norm(points[:, None, :]-coords, axis=2).min(axis=1)


def spatial_mean_field(points, values, width, height, bins=64):
    """Mean value in a fixed image grid, returning NaN for empty bins."""
    points, values = np.asarray(points), np.asarray(values)
    xi = np.clip((points[:, 0]/width*bins).astype(int), 0, bins-1)
    yi = np.clip((points[:, 1]/height*bins).astype(int), 0, bins-1)
    total = np.zeros((bins, bins)); count = np.zeros((bins, bins))
    np.add.at(total, (yi, xi), values); np.add.at(count, (yi, xi), 1)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def field_correlation(first, second):
    mask = np.isfinite(first) & np.isfinite(second)
    return float(np.corrcoef(first[mask], second[mask])[0, 1]) if mask.sum() > 3 else np.nan
