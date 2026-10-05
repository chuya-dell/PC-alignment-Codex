"""Result-preserving faster replacements for two hot spots of the Phase 2 production path.

Both functions reproduce the numbers of the originals in ``shared/`` (the originals are
not modified). Equality is verified by ``field_verify_speedup.py``.

* ``fast_sample_grid_features``: replaces the Python loop of 85,793 ``ndarray.mean`` calls
  of ``shared.theoretical_grid_evaluation.sample_grid_features`` by array operations that
  add the 3x3 centre and the 7x7 patch elements in the same row-major order as the
  original (so the floating point sums are expected to be bit-identical).
* ``fast_estimate_phase_origin_fft``: the original sums ``centered * exp(-2j*pi*(gx*x + gy*y))``
  over all 4.2 million pixels with a full 2-D complex exponential per reciprocal vector.
  The exponential is separable (exp of the x part times exp of the y part), so one
  1-D exponential per axis suffices.  This changes the result only at floating point
  rounding level (not bit-identical); the tolerance is checked by the verifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.lattice_indexing import HexLattice, grid_coordinates, estimate_hex_orientation_fft, hex_basis
from shared.registration import sample_contrast


def fast_sample_grid_features(image, lattice: HexLattice, margin=30, invalid_mask=None):
    height, width = image.shape
    indices, coordinates = grid_coordinates(lattice, width, height, margin=margin)
    sampled = sample_contrast(image, coordinates, invalid_mask=invalid_mask)
    arr = np.asarray(image, float) / 65535.0
    x, y = np.rint(coordinates).astype(int).T
    n = len(x)
    centre = np.full(n, np.nan)
    ring = np.full(n, np.nan)
    # Same condition as ``patch.shape == (7, 7)`` in the original slicing code
    # (negative start indices wrap in numpy slicing, giving an empty slice -> skipped).
    full = (y - 3 >= 0) & (y + 4 <= height) & (x - 3 >= 0) & (x + 4 <= width)
    rows = np.flatnonzero(full)
    px, py = x[rows], y[rows]
    # Row-major sequential accumulation, identical in order to ndarray.sum/mean on the patch.
    total = np.zeros(len(rows))
    centre_sum = np.zeros(len(rows))
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            v = arr[py + dy, px + dx]
            total = total + v
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            v = arr[py + dy, px + dx]
            centre_sum = centre_sum + v
    centre[rows] = centre_sum / 9
    ring[rows] = (total - centre_sum) / 40
    out = pd.DataFrame({"m": indices[:, 0], "n": indices[:, 1],
                        "x": coordinates[:, 0], "y": coordinates[:, 1]})
    out = pd.concat((out, sampled.reset_index(drop=True)), axis=1)
    out["centre_intensity"] = centre
    out["ring_intensity"] = ring
    out["centre_minus_ring"] = centre - ring
    return out


def fast_estimate_phase_origin_fft(image, basis):
    image = np.asarray(image, float)
    h, w = image.shape
    reciprocal = np.linalg.inv(basis)
    centered = image - image.mean()
    xs = np.arange(w, dtype=float)
    ys = np.arange(h, dtype=float)
    phases = []
    for gx, gy in reciprocal:
        ex = np.exp(-2j * np.pi * (gx * xs))
        ey = np.exp(-2j * np.pi * (gy * ys))
        # sum_{y,x} centered[y,x] * ey[y] * ex[x]
        coefficient = np.sum((centered @ ex) * ey)
        if not np.isfinite(coefficient) or abs(coefficient) == 0:
            raise ValueError("Cannot estimate lattice phase from image.")
        phases.append(-np.angle(coefficient) / (2 * np.pi))
    return np.linalg.solve(reciprocal, np.mod(phases, 1.0))


def fast_lattice_from_fft(image, pitch_px=7.286):
    angle = estimate_hex_orientation_fft(image, pitch_px)
    basis = hex_basis(pitch_px, angle)
    return HexLattice(fast_estimate_phase_origin_fft(image, basis), basis, angle, pitch_px)
