"""Canonical image-registration implementation.

The advanced registration history belongs to this module.  The returned
transform retains the complete quadratic model so that forward alignment and
inverse grid projection use the same geometry.
"""
import numpy as np
import cv2
import pandas as pd
from scipy.spatial import KDTree
import os

def load_image_unicode(path):
    """Unicode/Japanese path safe image read using numpy and cv2."""
    try:
        # IMREAD_ANYDEPTH preserves 16-bit TIFF depth
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    except Exception as e:
        print(f"Error loading image '{path}': {e}")
        return None

def detect_grooves(img):
    """
    Detect vertical and horizontal groove landmarks from the microscope image.
    Returns globally stable (groove_x, groove_y) coordinates.
    """
    h, w = img.shape
    if h < 1500 or w < 1900:
        raise ValueError('Groove registration requires image height >=1500 and width >=1900.')

    # Sub-pixel interpolation helper
    def get_subpixel_min(profile, min_idx):
        if 0 < min_idx < len(profile) - 1:
            y0, y1, y2 = profile[min_idx-1], profile[min_idx], profile[min_idx+1]
            denom = (y0 - 2*y1 + y2)
            if denom != 0:
                return min_idx + 0.5 * (y0 - y2) / denom
        return float(min_idx)

    # 1. Vertical Groove (X coordinate)
    # Average columns vertically in the central y-band [500, 1500] to avoid edge artifacts
    x_profile = np.mean(img[500:1500, 0:300], axis=0)
    # Remove shading gradient by subtracting a 51-pixel moving average
    x_detrend = x_profile - np.convolve(x_profile, np.ones(51)/51, mode='same')
    x_profile_smooth = np.convolve(x_detrend, np.ones(15)/15, mode='same')
    groove_x_idx = 30 + np.argmin(x_profile_smooth[30:270])
    groove_x = get_subpixel_min(x_profile_smooth, groove_x_idx)

    # 2. Horizontal Groove (Y coordinate)
    # Average rows horizontally in the right-side x-band [1000, 1900]
    y_profile = np.mean(img[0:300, 1000:1900], axis=1)
    # Remove shading gradient
    y_detrend = y_profile - np.convolve(y_profile, np.ones(51)/51, mode='same')
    y_profile_smooth = np.convolve(y_detrend, np.ones(15)/15, mode='same')
    groove_y_idx = 30 + np.argmin(y_profile_smooth[30:270])
    groove_y = get_subpixel_min(y_profile_smooth, groove_y_idx)

    return groove_x, groove_y

def calculate_coarse_shift(ref_img_path, tgt_img_path):
    """
    Detect grooves in both reference and target, and compute the coarse translation offset.
    Returns (dx, dy) to shift the target to align with reference.
    """
    ref_img = load_image_unicode(ref_img_path)
    tgt_img = load_image_unicode(tgt_img_path)

    if ref_img is None or tgt_img is None:
        raise ValueError("Failed to load reference or target image.")

    ref_x, ref_y = detect_grooves(ref_img)
    tgt_x, tgt_y = detect_grooves(tgt_img)

    dx = ref_x - tgt_x
    dy = ref_y - tgt_y

    print(f"Landmark detection:")
    print(f"  Reference grooves: X={ref_x:.2f}, Y={ref_y:.2f}")
    print(f"  Target grooves   : X={tgt_x:.2f}, Y={tgt_y:.2f}")
    print(f"  Coarse Shift (dx, dy): ({dx:.2f}, {dy:.2f})")

    return dx, dy

def find_grid_orientation(pts, pitch=7.286):
    """
    Finds dominant grid orientation angle (in radians) in [-30, 30] degrees
    by computing the mode of nearest-neighbor vector angles. Extremely robust.
    """
    tree = KDTree(pts)
    # Query 7 closest neighbors for each point
    distances, indices = tree.query(pts, k=7)

    angles_list = []

    for i in range(len(pts)):
        p = pts[i]
        for j in range(1, 7): # Skip index 0 (itself)
            d = distances[i, j]
            # Use the measured hexagonal pitch and a 15% neighbor tolerance.
            if 0.85 * pitch <= d <= 1.15 * pitch:
                q = pts[indices[i, j]]
                # Compute vector angle
                angle_rad = np.arctan2(q[1] - p[1], q[0] - p[0])
                angle_deg = np.degrees(angle_rad)
                # Fold into [-30, 30] degree range (hexagonal 60-degree symmetry)
                angle_fold = (angle_deg + 30.0) % 60.0 - 30.0
                angles_list.append(angle_fold)

    if len(angles_list) == 0:
        raise ValueError('No nearest-neighbor vectors near the specified lattice pitch.')

    # Build histogram of angles with very high resolution (0.02 degrees bins)
    bins = np.arange(-30.0, 30.0, 0.02)
    hist, bin_edges = np.histogram(angles_list, bins=bins)

    # Smooth histogram to find the true continuous peak (moving average of size 25)
    hist_smooth = np.convolve(hist, np.ones(25)/25, mode='same')

    best_idx = np.argmax(hist_smooth)
    best_angle_deg = 0.5 * (bin_edges[best_idx] + bin_edges[best_idx + 1])

    return np.radians(best_angle_deg)


def polynomial_features(points):
    x, y = np.asarray(points, dtype=float).T
    return np.column_stack([x, y, np.ones_like(x), x*x, y*y, x*y])


def fine_alignment_icp(ref_pts, tgt_pts_coarse, max_iter=100, tolerance=1e-6, return_diagnostics=False):
    """Dynamic 3.5/1.5 px quadratic ICP; retain every coefficient."""
    ref_pts, orig = np.asarray(ref_pts, float), np.asarray(tgt_pts_coarse, float)
    if min(len(ref_pts), len(orig)) < 20 or not (np.isfinite(ref_pts).all() and np.isfinite(orig).all()):
        raise ValueError("ICP requires at least 20 finite points per image.")
    origin, scale = orig.mean(axis=0), np.maximum(orig.std(axis=0), 1)
    design = polynomial_features((orig-origin)/scale)
    aligned, tree, prev_err = orig.copy(), KDTree(ref_pts), np.inf
    converged = False
    for i in range(max_iter):
        distances, indices = tree.query(aligned)
        valid = distances < (3.5 if i < 3 else 1.5)
        if valid.sum() < 20:
            raise ValueError(f"Only {valid.sum()} ICP pairs at iteration {i+1}.")
        coeff, _, rank, _ = np.linalg.lstsq(design[valid], ref_pts[indices[valid]], rcond=None)
        if rank != 6:
            raise ValueError("Rank-deficient quadratic fit.")
        aligned = design @ coeff
        err = np.mean(np.linalg.norm(aligned[valid]-ref_pts[indices[valid]], axis=1))
        if i >= 3 and abs(prev_err-err) < tolerance:
            converged = True
            break
        prev_err = err
    if not converged:
        raise ValueError("ICP did not converge.")
    transform = dict(coefficients=coeff, origin=origin, scale=scale,
                     rotation=np.eye(2), translation=np.zeros(2))
    if return_diagnostics:
        return transform, aligned, i+1, converged
    return transform, aligned


def transform_points(points, transform):
    coarse = np.asarray(points) @ transform["rotation"].T + transform["translation"]
    aligned = polynomial_features((coarse-transform["origin"])/transform["scale"]) @ transform["coefficients"]
    local = transform.get("local_refinement")
    if local is not None:
        aligned = _apply_local_shifts(aligned, local)
    return aligned


def _interpolate_local_shifts(points, local):
    """Bilinearly interpolate cell-centre shifts into a continuous field."""
    points = np.asarray(points, float)
    grid_n = local["grid_n"]
    col_f = points[:, 0] / local["cell_w"] - 0.5
    row_f = points[:, 1] / local["cell_h"] - 0.5
    col0 = np.floor(col_f).astype(int)
    row0 = np.floor(row_f).astype(int)
    tx, ty = col_f-col0, row_f-row0
    outside_left, outside_right = col0 < 0, col0 >= grid_n-1
    outside_top, outside_bottom = row0 < 0, row0 >= grid_n-1
    col0 = np.clip(col0, 0, grid_n-1)
    row0 = np.clip(row0, 0, grid_n-1)
    col1 = np.clip(col0+1, 0, grid_n-1)
    row1 = np.clip(row0+1, 0, grid_n-1)
    tx[outside_left | outside_right] = 0
    ty[outside_top | outside_bottom] = 0
    shifts = local["shifts"]
    return ((1-tx)[:, None]*(1-ty)[:, None]*shifts[row0, col0] +
            tx[:, None]*(1-ty)[:, None]*shifts[row0, col1] +
            (1-tx)[:, None]*ty[:, None]*shifts[row1, col0] +
            tx[:, None]*ty[:, None]*shifts[row1, col1])


def _apply_local_shifts(points, local):
    points = np.asarray(points, float)
    return points + _interpolate_local_shifts(points, local)


def _remove_local_shifts(points, local):
    """Invert the continuous local displacement field by fixed-point iteration."""
    target = np.asarray(points, float)
    unshifted = target.copy()
    for _ in range(30):
        candidate = target - _interpolate_local_shifts(unshifted, local)
        if np.max(np.linalg.norm(candidate-unshifted, axis=1), initial=0) < 1e-9:
            unshifted = candidate
            break
        unshifted = candidate
    if np.max(np.linalg.norm(_apply_local_shifts(unshifted, local)-target, axis=1), initial=0) > 1e-7:
        raise ValueError("Local-refinement inverse failed.")
    return unshifted


def inverse_transform_points(points, transform):
    """Invert the full quadratic, never silently substitute an identity."""
    target, coeff = np.asarray(points, float), transform["coefficients"]
    local = transform.get("local_refinement")
    if local is not None:
        target = _remove_local_shifts(target, local)
    q = np.linalg.solve(coeff[:2].T, (target-coeff[2]).T).T
    for _ in range(30):
        residual = polynomial_features(q) @ coeff-target
        if np.max(np.linalg.norm(residual, axis=1), initial=0) < 1e-7:
            break
        x, y = q.T
        jx = coeff[0] + 2*x[:, None]*coeff[3] + y[:, None]*coeff[5]
        jy = coeff[1] + 2*y[:, None]*coeff[4] + x[:, None]*coeff[5]
        q -= np.linalg.solve(np.stack([jx, jy], axis=2), residual[..., None])[..., 0]
    residual = np.linalg.norm(polynomial_features(q) @ coeff-target, axis=1)
    if not np.isfinite(q).all() or np.any(residual > 1e-4):
        raise ValueError("Quadratic inverse failed.")
    coarse = q*transform["scale"]+transform["origin"]
    return (coarse-transform["translation"]) @ transform["rotation"]


def fine_alignment_local_refinement(ref_pts, aligned_pts, img_w=2048, img_h=2044,
                                    grid_n=8, min_pairs=15, distance_threshold=2.0,
                                    return_diagnostics=False):
    """Fit 8x8 median residual shifts and apply their continuous interpolation."""
    ref_pts = np.asarray(ref_pts, float)
    aligned_pts = np.asarray(aligned_pts, float)
    refined = aligned_pts.copy()
    cell_w, cell_h = img_w / grid_n, img_h / grid_n
    shifts = np.zeros((grid_n, grid_n, 2), dtype=float)
    tree = KDTree(ref_pts)
    diagnostics = []
    for row in range(grid_n):
        for col in range(grid_n):
            in_cell = ((aligned_pts[:, 0] >= col * cell_w) &
                       (aligned_pts[:, 0] < (col + 1) * cell_w) &
                       (aligned_pts[:, 1] >= row * cell_h) &
                       (aligned_pts[:, 1] < (row + 1) * cell_h))
            cell_idx = np.flatnonzero(in_cell)
            support = 0
            applied = False
            if len(cell_idx):
                distances, indices = tree.query(aligned_pts[cell_idx])
                valid = distances < distance_threshold
                support = int(valid.sum())
                if support >= min_pairs:
                    shifts[row, col] = np.median(
                        ref_pts[indices[valid]] - aligned_pts[cell_idx][valid], axis=0
                    )
                    applied = True
            diagnostics.append({
                "row": row, "col": col, "point_count": int(len(cell_idx)),
                "support_count": support, "applied": applied,
                "dx": float(shifts[row, col, 0]), "dy": float(shifts[row, col, 1]),
            })
    local = {"grid_n": grid_n, "cell_w": cell_w, "cell_h": cell_h, "shifts": shifts}
    refined = _apply_local_shifts(aligned_pts, local)
    if return_diagnostics:
        return refined, local, diagnostics
    return refined, local


def align_and_match_dataframes(df_ref, df_tgt, ref_img_path, tgt_img_path,
                               return_diagnostics=False, pitch=7.286,
                               local_refinement=True, grid_n=8):
    for name, frame in (("reference", df_ref), ("target", df_tgt)):
        if not {"pillar_id", "x", "y"}.issubset(frame.columns):
            raise ValueError(f"{name}: missing pillar_id/x/y.")
        if frame["pillar_id"].duplicated().any() or len(frame) < 20:
            raise ValueError(f"{name}: need at least 20 unique pillar IDs.")
    ref, tgt = df_ref[["x", "y"]].to_numpy(float), df_tgt[["x", "y"]].to_numpy(float)
    if not np.isfinite(ref).all() or not np.isfinite(tgt).all():
        raise ValueError("Nonfinite coordinates.")
    ref_img, tgt_img = load_image_unicode(ref_img_path), load_image_unicode(tgt_img_path)
    if ref_img is None or tgt_img is None or ref_img.shape != tgt_img.shape:
        raise ValueError("Images must be readable and have equal dimensions.")
    ref_landmark, tgt_landmark = np.array(detect_grooves(ref_img)), np.array(detect_grooves(tgt_img))
    theta = find_grid_orientation(ref, pitch)-find_grid_orientation(tgt, pitch)
    theta = (theta+np.pi/6) % (np.pi/3)-np.pi/6
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    translation = ref_landmark-rotation @ tgt_landmark
    coarse = tgt @ rotation.T+translation
    transform, aligned, iterations, converged = fine_alignment_icp(ref, coarse, return_diagnostics=True)
    transform.update(rotation=rotation, translation=translation)
    local_diagnostics = []
    if local_refinement:
        aligned, local, local_diagnostics = fine_alignment_local_refinement(
            ref, aligned, img_w=ref_img.shape[1], img_h=ref_img.shape[0],
            grid_n=grid_n, return_diagnostics=True,
        )
        transform["local_refinement"] = local
        transform["local_diagnostics"] = local_diagnostics
    result = df_tgt.copy()
    result["x_original"], result["y_original"] = tgt.T
    result["x"], result["y"] = aligned.T
    distances, indices = KDTree(ref).query(aligned)
    result["matched_ref_id"] = df_ref["pillar_id"].to_numpy()[indices]
    result["alignment_distance"] = distances
    result.loc[distances > 1.5, "matched_ref_id"] = -1
    candidates = result[result["matched_ref_id"] != -1].sort_values("alignment_distance")
    result.loc[candidates.index[candidates["matched_ref_id"].duplicated()], "matched_ref_id"] = -1
    if not (result["matched_ref_id"] != -1).any():
        raise ValueError("No unique matches within 1.5 pixels.")
    if return_diagnostics:
        dx, dy = ref_landmark-tgt_landmark
        return result, transform, iterations, converged, dx, dy
    return result, transform


def sample_contrast(img, coords, invalid_mask=None):
    """Sample 3x3 contrast, optionally rejecting any mask-overlapping patch.

    ``invalid_mask`` is a boolean image in the same native pixel coordinate
    system as ``img``.  A coordinate is invalid when any pixel in its 3x3
    sampling footprint is masked.  This mirrors the existing image-boundary
    rule and deliberately does not impute a contrast value.
    """
    if img is None or img.ndim != 2 or not np.isfinite(img).all():
        raise ValueError("A finite grayscale image is required.")
    img = img.astype(np.float32)/65535.0
    background = cv2.GaussianBlur(img, (51, 51), 0)
    coords = np.asarray(coords, float)
    finite = np.isfinite(coords).all(axis=1)
    x, y = np.rint(np.where(finite[:, None], coords, 0)).astype(int).T
    h, w = img.shape
    valid = finite & (x >= 1) & (x < w-1) & (y >= 1) & (y < h-1)
    if invalid_mask is not None:
        invalid_mask = np.asarray(invalid_mask, dtype=bool)
        if invalid_mask.shape != img.shape:
            raise ValueError("invalid_mask must have the same shape as img.")
        dilated = cv2.dilate(invalid_mask.astype(np.uint8), np.ones((3, 3), np.uint8))
        inside = finite & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        overlaps = np.zeros(len(x), dtype=bool)
        overlaps[inside] = dilated[y[inside], x[inside]] != 0
        valid &= ~overlaps
    ints = np.full(len(x), np.nan)
    bg_ints = ints.copy()
    ints[valid], bg_ints[valid] = 0, 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ints[valid] += img[y[valid]+dy, x[valid]+dx]
            bg_ints[valid] += background[y[valid]+dy, x[valid]+dx]
    return pd.DataFrame(dict(ints=ints, bg_ints=bg_ints, contrast=ints-bg_ints,
                             valid_sampling=valid))


def create_grid_estimated_dataframe(df_ref, H_final, tgt_img_path):
    """Full inverse pre grid projection; retain invalid rows for explicit QC."""
    ref = df_ref[["x", "y"]].to_numpy(float)
    coords = inverse_transform_points(ref, H_final)
    result = sample_contrast(load_image_unicode(tgt_img_path), coords)
    result.insert(0, "pillar_id", df_ref["pillar_id"].to_numpy())
    result["matched_ref_id"] = df_ref["pillar_id"].to_numpy()
    result["x"], result["y"] = coords.T
    result["x_ref"], result["y_ref"] = ref.T
    result["mean_intensity"] = result["ints"]/9
    result["is_estimated"] = True
    return result
