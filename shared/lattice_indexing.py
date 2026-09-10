"""Common-frame hexagonal-lattice indexing without cross-image nearest matching."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter


@dataclass(frozen=True)
class HexLattice:
    """A hexagonal direct lattice in image coordinates (x right, y down)."""

    origin: np.ndarray
    basis: np.ndarray  # columns are direct-lattice vectors b1, b2
    angle_rad: float
    pitch_px: float


def _periodic_mean(angles, period, weights=None):
    phases = np.exp(2j*np.pi*np.asarray(angles)/period)
    mean = np.average(phases, weights=weights)
    return float(np.angle(mean) * period/(2*np.pi)) % period


def estimate_hex_orientation_fft(image, pitch_px=7.286, radial_tolerance=0.15):
    """Estimate direct-lattice angle modulo 60 degrees from reciprocal FFT peaks."""
    image = np.asarray(image, float)
    h, w = image.shape
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image-image.mean()))))
    yy, xx = np.indices(image.shape)
    cy, cx = h//2, w//2
    fx, fy = (xx-cx)/w, (yy-cy)/h
    radius = np.hypot(fx, fy)
    reciprocal_radius = 2/(math.sqrt(3)*pitch_px)
    candidate = ((maximum_filter(spectrum, size=9) == spectrum) &
                 (radius >= reciprocal_radius*(1-radial_tolerance)) &
                 (radius <= reciprocal_radius*(1+radial_tolerance)))
    weights = spectrum[candidate]
    if len(weights) < 6:
        raise ValueError("Insufficient FFT reciprocal-lattice peaks.")
    angles = np.arctan2(fy[candidate], fx[candidate])
    strongest = np.argsort(weights)[-min(12, len(weights)):]
    # Any first-shell reciprocal peak is alpha-30 degrees modulo 60.
    alpha = _periodic_mean(angles[strongest] + np.pi/6, np.pi/3, weights=weights[strongest])
    return alpha if alpha <= np.pi/6 else alpha-np.pi/3


def hex_basis(pitch_px, angle_rad):
    """Return direct-lattice vectors [a at alpha, a at alpha+60deg] as columns."""
    return pitch_px*np.array([
        [np.cos(angle_rad), np.cos(angle_rad+np.pi/3)],
        [np.sin(angle_rad), np.sin(angle_rad+np.pi/3)],
    ])


def estimate_phase_origin_fft(image, basis):
    """Estimate one lattice-point origin modulo the direct lattice from FFT phases."""
    image = np.asarray(image, float)
    h, w = image.shape
    yy, xx = np.indices(image.shape)
    reciprocal = np.linalg.inv(basis)  # rows g_i: g_i dot b_j = delta_ij
    centered = image-image.mean()
    phases = []
    for gx, gy in reciprocal:
        coefficient = np.sum(centered*np.exp(-2j*np.pi*(gx*xx+gy*yy)))
        if not np.isfinite(coefficient) or abs(coefficient) == 0:
            raise ValueError("Cannot estimate lattice phase from image.")
        phases.append(-np.angle(coefficient)/(2*np.pi))
    return np.linalg.solve(reciprocal, np.mod(phases, 1.0))


def lattice_from_fft(image, pitch_px=7.286):
    angle = estimate_hex_orientation_fft(image, pitch_px)
    basis = hex_basis(pitch_px, angle)
    return HexLattice(estimate_phase_origin_fft(image, basis), basis, angle, pitch_px)


def grid_coordinates(lattice, width, height, margin=0.0):
    """Enumerate all theoretical lattice coordinates inside the rectangular image."""
    corners = np.array([[margin, margin], [width-margin, margin],
                        [margin, height-margin], [width-margin, height-margin]])
    inv_basis = np.linalg.inv(lattice.basis)
    coefficients = (corners-lattice.origin) @ inv_basis.T
    lo = np.floor(coefficients.min(axis=0)).astype(int)-2
    hi = np.ceil(coefficients.max(axis=0)).astype(int)+2
    m, n = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1), indexing="ij")
    indices = np.column_stack((m.ravel(), n.ravel()))
    coords = lattice.origin + indices @ lattice.basis.T
    keep = ((coords[:, 0] >= margin) & (coords[:, 0] < width-margin) &
            (coords[:, 1] >= margin) & (coords[:, 1] < height-margin))
    return indices[keep], coords[keep]


def assign_lattice_indices(points, lattice, max_distance=1.5):
    """Assign each point to a nearest integer (m,n), retaining one point per cell."""
    points = np.asarray(points, float)
    inv_basis = np.linalg.inv(lattice.basis)
    continuous = (points-lattice.origin) @ inv_basis.T
    central = np.rint(continuous).astype(int)
    offsets = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)])
    candidates = central[:, None, :]+offsets[None, :, :]
    candidate_coords = lattice.origin + candidates @ lattice.basis.T
    candidate_distance = np.linalg.norm(points[:, None, :]-candidate_coords, axis=2)
    winner = np.argmin(candidate_distance, axis=1)
    indices = candidates[np.arange(len(points)), winner]
    fitted = candidate_coords[np.arange(len(points)), winner]
    distance = candidate_distance[np.arange(len(points)), winner]
    accepted = distance <= max_distance
    # Resolve multiple detections assigned to one theoretical cell by residual only.
    best = {}
    for row in np.flatnonzero(accepted)[np.argsort(distance[accepted])]:
        key = tuple(indices[row])
        if key not in best:
            best[key] = int(row)
    selected = np.zeros(len(points), dtype=bool)
    selected[list(best.values())] = True
    return {
        "indices": indices,
        "fitted_coordinates": fitted,
        "distance": distance,
        "within_distance": accepted,
        "selected": selected,
        "duplicate": accepted & ~selected,
        "outlier": ~accepted,
    }


def refine_origin_from_points(lattice, points, iterations=5, support_distance=3.0):
    """Robustly refine only the pre-grid phase; no post detections are consulted."""
    origin = lattice.origin.copy()
    for _ in range(iterations):
        trial = HexLattice(origin, lattice.basis, lattice.angle_rad, lattice.pitch_px)
        assigned = assign_lattice_indices(points, trial, max_distance=support_distance)
        support = assigned["within_distance"]
        if support.sum() < 20:
            raise ValueError("Too few pre points support lattice-phase refinement.")
        shift = np.median(np.asarray(points)[support]-assigned["fitted_coordinates"][support], axis=0)
        origin += shift
        if np.linalg.norm(shift) < 1e-5:
            break
    return HexLattice(origin, lattice.basis, lattice.angle_rad, lattice.pitch_px)


def optimize_origin_phase(lattice, points, max_distance=1.5, subdivisions=24):
    """Choose the pre-grid phase that maximizes in-cell detections over one unit cell."""
    points = np.asarray(points, float)
    best_origin, best_count = lattice.origin, -1
    for u in np.arange(subdivisions)/subdivisions:
        for v in np.arange(subdivisions)/subdivisions:
            origin = lattice.origin + np.array([u, v]) @ lattice.basis.T
            trial = HexLattice(origin, lattice.basis, lattice.angle_rad, lattice.pitch_px)
            count = int(assign_lattice_indices(points, trial, max_distance)["within_distance"].sum())
            if count > best_count:
                best_origin, best_count = origin, count
    seeded = HexLattice(best_origin, lattice.basis, lattice.angle_rad, lattice.pitch_px)
    return refine_origin_from_points(seeded, points, support_distance=max_distance), best_count


def rigid_transform_from_common_fft_grid(pre_image, post_image, pitch_px=7.286):
    """Return post->pre rigid map from FFT orientation/phase; resolve only phase residual."""
    pre = lattice_from_fft(pre_image, pitch_px)
    post = lattice_from_fft(post_image, pitch_px)
    theta = pre.angle_rad-post.angle_rad
    theta = (theta+np.pi/6) % (np.pi/3)-np.pi/6
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # FFT origins are modulo a lattice vector. The caller supplies a landmark translation,
    # then this phase residual provides only the sub-cell correction.
    return pre, post, rotation


def transform_points_rigid(points, rotation, translation):
    return np.asarray(points, float) @ rotation.T + np.asarray(translation, float)
