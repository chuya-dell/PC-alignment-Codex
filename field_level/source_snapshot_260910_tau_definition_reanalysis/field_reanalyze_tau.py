"""Auditable reanalysis of p50 pre/post and repeat-pair noise.

The script intentionally uses one registration and one ROI/statistic definition for
both the handling pair and the repeat pairs.  It does not apply the invalidated
per-image ``median(diff(unique(image)))`` quantization normalization.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "work" / "input"
OUTPUT = ROOT / "data" / "results" / "tau_definition_reanalysis"
U1_SEED = np.array([-6.73781, -2.93370], dtype=float)
U2_SEED = np.array([5.67093, -4.26859], dtype=float)
ROI_RADIUS = 1.5
ROI_STEP = 0.25
EXPOSURES_MS = [2, 4, 6, 9, 13, 15]


def load_tiff(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError(str(path))
    return image.astype(np.float32)


def coarse_shift(pre: np.ndarray, post: np.ndarray) -> tuple[float, float, float]:
    height, width = pre.shape
    size = min(1000, height - 40, width - 40)
    cy, cx, half, margin = height // 2, width // 2, size // 2, 15
    template = pre[cy-half:cy+half, cx-half:cx+half]
    search = post[cy-half-margin:cy+half+margin, cx-half-margin:cx+half+margin]
    response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    return float(location[0] - margin), float(location[1] - margin), float(score)


def pyramid_ecc(pre: np.ndarray, post: np.ndarray, levels: tuple[float, ...],
                initial_full: np.ndarray, motion: int) -> tuple[float, np.ndarray, list[float]]:
    warp = initial_full.astype(np.float32).copy()
    previous = 1.0
    scores: list[float] = []
    for level in levels:
        warp[:, 2] *= level / previous
        if level == 1.0:
            template, target = pre, post
        else:
            template = cv2.resize(pre, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
            target = cv2.resize(post, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
        score, warp = cv2.findTransformECC(
            template, target, warp, motion,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7),
            None, 5,
        )
        scores.append(float(score))
        previous = level
    return scores[-1], warp, scores


def registration_candidates(pre: np.ndarray, post: np.ndarray, deep: bool):
    tx, ty, coarse_score = coarse_shift(pre, post)
    base = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    candidates = [("affine_coarse_025", (0.25, 0.5, 1.0),
                   base, cv2.MOTION_AFFINE)]
    if deep:
        candidates.append(("affine_coarse_0125", (0.125, 0.25, 0.5, 1.0),
                           base, cv2.MOTION_AFFINE))
        shifts = [U1_SEED, -U1_SEED, U2_SEED, -U2_SEED,
                  U1_SEED + U2_SEED, -(U1_SEED + U2_SEED)]
        for index, shift in enumerate(shifts):
            trial = base.copy()
            trial[:, 2] += shift.astype(np.float32)
            candidates.append((f"affine_pitch_shift_{index}",
                               (0.125, 0.25, 0.5, 1.0), trial, cv2.MOTION_AFFINE))
    records = []
    for name, levels, initial, motion in candidates:
        try:
            rho, warp, scores = pyramid_ecc(pre, post, levels, initial, motion)
            determinant = float(np.linalg.det(warp[:, :2]))
            records.append({"name": name, "rho": rho, "warp": warp,
                            "scores": scores, "determinant": determinant,
                            "coarse_score": coarse_score, "motion": motion})
        except cv2.error:
            records.append({"name": name, "rho": float("nan"), "warp": initial,
                            "scores": [], "determinant": float("nan"),
                            "coarse_score": coarse_score, "motion": motion})
    plausible = [r for r in records if np.isfinite(r["rho"]) and 0.90 <= r["determinant"] <= 1.10]
    if not plausible:
        name, levels, initial, motion = ("translation_fallback_025",
                                          (0.25, 0.5, 1.0), base,
                                          cv2.MOTION_TRANSLATION)
        try:
            rho, warp, scores = pyramid_ecc(pre, post, levels, initial, motion)
            determinant = float(np.linalg.det(warp[:, :2]))
            fallback = {"name": name, "rho": rho, "warp": warp,
                        "scores": scores, "determinant": determinant,
                        "coarse_score": coarse_score, "motion": motion}
        except cv2.error:
            fallback = {"name": name, "rho": float("nan"), "warp": initial,
                        "scores": [], "determinant": float("nan"),
                        "coarse_score": coarse_score, "motion": motion}
        records.append(fallback)
        plausible = [r for r in records if np.isfinite(r["rho"]) and
                     0.90 <= r["determinant"] <= 1.10]
    if not plausible:
        raise RuntimeError("No finite physically plausible registration candidate")
    best = max(plausible, key=lambda item: item["rho"])
    return best, records


def warp_post(post: np.ndarray, warp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(post, warp, (shape[1], shape[0]),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def detect_pillars(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled = np.nan_to_num(image, nan=float(np.nanmedian(image)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(filled, cv2.MORPH_TOPHAT, kernel)
    smooth = cv2.GaussianBlur(tophat, (3, 3), 0)
    dilated = cv2.dilate(smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    positive = smooth[smooth > 100]
    cutoff = max(1000.0, float(np.percentile(positive, 25)))
    peaks = (smooth == dilated) & (smooth >= cutoff)
    peaks[:40] = False
    peaks[-40:] = False
    peaks[:, :40] = False
    peaks[:, -40:] = False
    _, _, _, centers = cv2.connectedComponentsWithStats(peaks.astype(np.uint8))
    return centers[1:].astype(float), tophat


def fit_affine_lattice(points: np.ndarray, shape: tuple[int, int]):
    basis = np.column_stack([U1_SEED, U2_SEED])
    origin_seed = np.array([shape[1] / 2, shape[0] / 2], dtype=float)
    inverse = np.linalg.inv(basis)
    best_count, best_origin = -1, origin_seed
    for first in np.arange(16) / 16:
        for second in np.arange(16) / 16:
            origin = origin_seed + basis @ np.array([first, second])
            indices = np.rint((points - origin) @ inverse.T)
            fitted = origin + indices @ basis.T
            count = int(np.sum(np.linalg.norm(points - fitted, axis=1) <= 1.5))
            if count > best_count:
                best_count, best_origin = count, origin
    origin = best_origin
    matched = 0
    for _ in range(6):
        inverse = np.linalg.inv(basis)
        indices = np.rint((points - origin) @ inverse.T).astype(int)
        fitted = origin + indices @ basis.T
        keep = np.linalg.norm(points - fitted, axis=1) <= 1.8
        design = np.column_stack([np.ones(int(keep.sum())), indices[keep]])
        coefficient, _, _, _ = np.linalg.lstsq(design, points[keep], rcond=None)
        origin, basis = coefficient[0], coefficient[1:].T
        matched = int(keep.sum())
    return origin, basis, matched


def enumerate_grid(origin: np.ndarray, basis: np.ndarray, shape: tuple[int, int], margin=3.0):
    height, width = shape
    corners = np.array([[margin, margin], [width-margin, margin],
                        [margin, height-margin], [width-margin, height-margin]])
    coefficient = (corners - origin) @ np.linalg.inv(basis).T
    low = np.floor(coefficient.min(axis=0)).astype(int) - 2
    high = np.ceil(coefficient.max(axis=0)).astype(int) + 2
    first, second = np.meshgrid(np.arange(low[0], high[0] + 1),
                                np.arange(low[1], high[1] + 1), indexing="ij")
    indices = np.column_stack([first.ravel(), second.ravel()])
    centers = origin + indices @ basis.T
    keep = ((centers[:, 0] >= margin) & (centers[:, 0] < width-margin) &
            (centers[:, 1] >= margin) & (centers[:, 1] < height-margin))
    return indices[keep], centers[keep]


def prop_c_centers(grid: np.ndarray, pre_points: np.ndarray, post_points: np.ndarray):
    residuals = []
    available = []
    for points in (pre_points, post_points):
        distance, nearest = cKDTree(points).query(grid, k=1)
        residual = points[nearest] - grid
        valid = distance <= 3.6
        residuals.append(residual)
        available.append(valid)
    r0, r1 = residuals
    v0, v1 = available
    mean = np.zeros_like(grid)
    both = v0 & v1
    mean[both] = (r0[both] + r1[both]) / 2
    mean[v0 & ~v1] = r0[v0 & ~v1]
    mean[v1 & ~v0] = r1[v1 & ~v0]
    return grid + mean, int(both.sum()), int((~v0 & ~v1).sum())


def roi_mean(image: np.ndarray, centers: np.ndarray) -> np.ndarray:
    axis = np.arange(-ROI_RADIUS, ROI_RADIUS + ROI_STEP / 2, ROI_STEP)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    inside = xx * xx + yy * yy <= ROI_RADIUS * ROI_RADIUS + 1e-9
    ox, oy = xx[inside].astype(np.float32), yy[inside].astype(np.float32)
    values = np.empty(len(centers), dtype=np.float32)
    for start in range(0, len(centers), 5000):
        block = centers[start:start+5000]
        map_x = (block[:, 0, None] + ox).astype(np.float32)
        map_y = (block[:, 1, None] + oy).astype(np.float32)
        sampled = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        values[start:start+len(block)] = np.nanmean(sampled, axis=1)
    return values


def legacy_tail_metrics(pre: np.ndarray, aligned: np.ndarray, pre_points: np.ndarray):
    height, width = pre.shape
    ix = np.clip(np.rint(pre_points[:, 0]).astype(int), 0, width - 1)
    iy = np.clip(np.rint(pre_points[:, 1]).astype(int), 0, height - 1)
    pre_blur = cv2.blur(pre, (3, 3))
    post_blur = cv2.blur(np.nan_to_num(aligned, nan=0.0), (3, 3))
    first, second = pre_blur[iy, ix], post_blur[iy, ix]
    keep = (first >= np.percentile(first, 8)) & (second > 0)
    delta = (second[keep] - first[keep]) / first[keep] * 100
    centered_raw = delta - np.median(delta)
    dark = float(np.percentile(pre, 0.5))
    net_delta = (second[keep] - first[keep]) / np.maximum(100.0, first[keep] - dark) * 100
    centered_net = net_delta - np.median(net_delta)
    return float(np.percentile(centered_raw, 99.9)), float(np.percentile(centered_net, 99.9))


def analyze_pair(label: str, pre_path: Path, post_path: Path, deep=False):
    pre, post = load_tiff(pre_path), load_tiff(post_path)
    best, candidates = registration_candidates(pre, post, deep)
    aligned = warp_post(post, best["warp"], pre.shape)
    pre_points, pre_tophat = detect_pillars(pre)
    post_points, post_tophat = detect_pillars(aligned)
    origin, basis, lattice_matches = fit_affine_lattice(pre_points, pre.shape)
    _, grid = enumerate_grid(origin, basis, pre.shape)
    centers, prop_both, prop_none = prop_c_centers(grid, pre_points, post_points)
    first, second = roi_mean(pre_tophat, centers), roi_mean(post_tophat, centers)
    valid = np.isfinite(first) & np.isfinite(second) & (first > 0) & (second > 0)
    first, second = first[valid], second[valid]
    mean_signal = (first + second) / 2
    difference = second - first
    relative = difference / first * 100
    tau_pair = float(np.std(difference, ddof=0) / np.mean(mean_signal) * 100)
    tau_single_equivalent = tau_pair / math.sqrt(2)
    tau_relative_pair = float(np.std(relative, ddof=0))
    legacy_raw, legacy_net = legacy_tail_metrics(pre, aligned, pre_points)
    linear = best["warp"][:, :2]
    row = {
        "label": label, "pre": str(pre_path), "post": str(post_path),
        "n_pre_detected": len(pre_points), "n_post_detected": len(post_points),
        "n_grid": len(grid), "n_valid_roi": int(valid.sum()),
        "n_prop_c_both": prop_both, "n_prop_c_none": prop_none,
        "lattice_fit_matches": lattice_matches,
        "u1_x": basis[0, 0], "u1_y": basis[1, 0],
        "u2_x": basis[0, 1], "u2_y": basis[1, 1],
        "ecc_rho": best["rho"], "tx_px": best["warp"][0, 2],
        "ty_px": best["warp"][1, 2],
        "rotation_deg": math.degrees(math.atan2(linear[1, 0], linear[0, 0])),
        "area_scale_change_pct": (math.sqrt(abs(np.linalg.det(linear))) - 1) * 100,
        "mean_signal_adu": float(np.mean(mean_signal)),
        "sd_pair_difference_adu": float(np.std(difference, ddof=0)),
        "median_relative_change_pct": float(np.median(relative)),
        "mad_relative_change_pct": float(median_abs_deviation(relative)),
        "tau_pair_pct": tau_pair,
        "tau_single_equivalent_pct": tau_single_equivalent,
        "tau_relative_pair_pct": tau_relative_pair,
        "legacy_tau_raw_p999_pct": legacy_raw,
        "legacy_tau_net_p999_pct": legacy_net,
    }
    diagnostics = []
    for candidate in candidates:
        warp = candidate["warp"]
        diagnostics.append({
            "label": label, "name": candidate["name"], "rho": candidate["rho"],
            "determinant": candidate["determinant"], "coarse_score": candidate["coarse_score"],
            "tx_px": float(warp[0, 2]), "ty_px": float(warp[1, 2]),
            "a00": float(warp[0, 0]), "a01": float(warp[0, 1]),
            "a10": float(warp[1, 0]), "a11": float(warp[1, 1]),
            "level_scores": json.dumps(candidate["scores"]),
        })
    centered_relative = relative - np.median(relative)
    return row, diagnostics, centered_relative


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manifest():
    rows = []
    for path in sorted(INPUT.rglob("*.tif")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                     "sha256": digest})
    return rows


def main():
    OUTPUT.mkdir(exist_ok=True)
    control_rows, control_distributions, diagnostic_rows = [], {}, []
    for position in range(1, 9):
        row, diagnostic, distribution = analyze_pair(
            f"1-{position}", INPUT / "control" / f"1-{position}-0.tif",
            INPUT / "control" / f"1-{position}-1.tif", deep=position == 6)
        control_rows.append(row)
        control_distributions[row["label"]] = distribution
        diagnostic_rows.extend(diagnostic)

    noise_rows, noise_distributions = [], {}
    for index, exposure in enumerate(EXPOSURES_MS, start=1):
        row, diagnostic, distribution = analyze_pair(
            f"260901_{exposure}ms", INPUT / "gain_260901" / f"{index}-1.tif",
            INPUT / "gain_260901" / f"{index}-2.tif")
        row["exposure_ms"] = exposure
        noise_rows.append(row)
        noise_distributions[row["label"]] = distribution
        diagnostic_rows.extend(diagnostic)
    row, diagnostic, distribution = analyze_pair(
        "260904_6ms", INPUT / "repeat_260904" / "1.tif",
        INPUT / "repeat_260904" / "2.tif")
    row["exposure_ms"] = 6
    noise_rows.append(row)
    noise_distributions[row["label"]] = distribution
    diagnostic_rows.extend(diagnostic)

    noise_260901 = next(item["tau_pair_pct"] for item in noise_rows if item["label"] == "260901_6ms")
    noise_260904 = next(item["tau_pair_pct"] for item in noise_rows if item["label"] == "260904_6ms")
    operation_rows = []
    for item in control_rows:
        for source, noise in [("260901_6ms", noise_260901), ("260904_6ms", noise_260904)]:
            difference = item["tau_pair_pct"] ** 2 - noise ** 2
            operation_rows.append({
                "fov": item["label"], "device_noise_source": source,
                "tau_pair_pct": item["tau_pair_pct"], "device_noise_pair_pct": noise,
                "variance_difference_pct2": difference,
                "operation_caused_spread_pct": math.sqrt(difference) if difference >= 0 else float("nan"),
                "nonnegative_variance_difference": int(difference >= 0),
            })

    reference_distribution = noise_distributions["260904_6ms"]
    positive_threshold = float(np.percentile(reference_distribution, 99.9))
    negative_threshold = float(np.percentile(reference_distribution, 0.1))
    digital_rows = []
    for item in control_rows:
        distribution = control_distributions[item["label"]]
        positive_count = int(np.sum(distribution >= positive_threshold))
        negative_count = int(np.sum(distribution <= negative_threshold))
        digital_rows.append({
            "fov": item["label"], "threshold_source": "260904_6ms",
            "positive_threshold_centered_pct": positive_threshold,
            "negative_threshold_centered_pct": negative_threshold,
            "n_valid_roi": len(distribution),
            "positive_count": positive_count,
            "positive_rate_pct": positive_count / len(distribution) * 100,
            "negative_count": negative_count,
            "negative_rate_pct": negative_count / len(distribution) * 100,
            "reference_tail_rate_pct": 0.1,
        })

    background_k_6ms = 8.93
    detection_rows = []
    for source, noise in [("260901_6ms", noise_260901), ("260904_6ms", noise_260904)]:
        background_factor = math.sqrt(1 + background_k_6ms)
        roi_factor = 2.0
        frame_factor = math.sqrt(1000)
        detection_rows.append({
            "device_noise_source": source, "device_noise_pair_pct": noise,
            "background_k_6ms_model_estimate": background_k_6ms,
            "background_reduction_factor_model": background_factor,
            "roi_expansion_factor_theoretical": roi_factor,
            "frame_averaging_factor_theoretical": frame_factor,
            "detection_limit_3sigma_pct_theoretical": 3 * noise / (background_factor * roi_factor * frame_factor),
        })

    projected_operation_rows = []
    for item in operation_rows:
        if item["device_noise_source"] != "260904_6ms":
            continue
        operation_spread = item["operation_caused_spread_pct"]
        improvement = math.sqrt(1 + background_k_6ms) * 2.0 * math.sqrt(1000)
        projected_operation_rows.append({
            "fov": item["fov"],
            "tau_pair_pct_measured": item["tau_pair_pct"],
            "device_noise_pair_pct_measured": item["device_noise_pair_pct"],
            "operation_caused_spread_pct_measured": operation_spread,
            "background_k_6ms_model_estimate": background_k_6ms,
            "roi_expansion_factor_theoretical": 2.0,
            "frame_averaging_factor_theoretical": math.sqrt(1000),
            "projected_operation_spread_pct_theoretical": operation_spread / improvement,
            "projected_3sigma_limit_pct_theoretical": 3 * operation_spread / improvement,
        })

    write_csv(OUTPUT / "tau_pair_all_fovs.csv", control_rows)
    write_csv(OUTPUT / "device_noise_pairs.csv", noise_rows)
    write_csv(OUTPUT / "operation_caused_spread.csv", operation_rows)
    write_csv(OUTPUT / "digital_counts.csv", digital_rows)
    write_csv(OUTPUT / "registration_diagnostics.csv", diagnostic_rows)
    write_csv(OUTPUT / "detection_limit_scenarios.csv", detection_rows)
    write_csv(OUTPUT / "projected_detection_limit_by_fov.csv", projected_operation_rows)
    write_csv(OUTPUT / "input_manifest_sha256.csv", manifest())
    numeric_summary = {
        "control_fovs": len(control_rows), "noise_pairs": len(noise_rows),
        "fov16_candidates": sum(item["label"] == "1-6" for item in diagnostic_rows),
        "tau_pair_mean_pct": float(np.mean([item["tau_pair_pct"] for item in control_rows])),
        "tau_pair_median_pct": float(np.median([item["tau_pair_pct"] for item in control_rows])),
        "device_noise_260901_6ms_pct": noise_260901,
        "device_noise_260904_6ms_pct": noise_260904,
    }
    (OUTPUT / "numeric_summary.json").write_text(json.dumps(numeric_summary, indent=2), encoding="utf-8")
    print(json.dumps(numeric_summary))


if __name__ == "__main__":
    main()
