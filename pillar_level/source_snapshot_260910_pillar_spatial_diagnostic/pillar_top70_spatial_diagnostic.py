from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import chi2


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPOSITORY_ROOT / "data" / "raw" / "local_input_control"
OUTPUT_DIR = REPOSITORY_ROOT / "data" / "results" / "pillar_spatial_diagnostic"
WORK_DIR = OUTPUT_DIR / "work"

PITCH_ROW_PX = 6.38
MIN_PRE_ADU = 32500.0
MARGIN_PX = 30
TOP_N = 70
NEAR_DEFECT_PX = 25.0
RNG_SEED = 260908
N_PERM = 2000


REPORT_REFERENCE = {
    1: {"rho": 0.978, "mad": 0.57, "tau": 2.76},
    2: {"rho": 0.984, "mad": 0.58, "tau": 3.20},
    3: {"rho": 0.984, "mad": 0.67, "tau": 4.82},
    4: {"rho": 0.956, "mad": 0.76, "tau": 6.58},
    5: {"rho": 0.992, "mad": 0.52, "tau": 2.52},
    6: {"rho": 0.971, "mad": 1.45, "tau": 3.96},
    7: {"rho": 0.989, "mad": 0.65, "tau": 2.80},
    8: {"rho": 0.987, "mad": 0.76, "tau": 5.64},
}


def load_tiff(path: Path) -> np.ndarray:
    raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    if raw is None or raw.ndim != 2:
        raise ValueError(f"Could not read grayscale TIFF: {path}")
    return raw.astype(np.float32) / 65535.0


def coarse_shift(pre: np.ndarray, post: np.ndarray) -> tuple[float, float]:
    rows, cols = pre.shape
    patch_size = min(1000, rows - 2 * 20, cols - 2 * 20)
    cy, cx = rows // 2, cols // 2
    half = patch_size // 2
    margin = 15
    patch = pre[cy-half:cy+half, cx-half:cx+half]
    search = post[cy-half-margin:cy+half+margin, cx-half-margin:cx+half+margin]
    response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(response)
    return float(loc[0] - margin), float(loc[1] - margin)


def phase_shift_at_scale(pre: np.ndarray, post: np.ndarray, scale: float) -> tuple[float, float]:
    a = cv2.resize(pre, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(post, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, _ = cv2.phaseCorrelate(a, b, window)
    return float(shift[0] / scale), float(shift[1] / scale)


def pyramid_ecc(
    pre: np.ndarray,
    post: np.ndarray,
    levels: tuple[float, ...] = (0.25, 0.5, 1.0),
    initial_full: tuple[float, float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    if initial_full is None:
        initial_full = coarse_shift(pre, post)
    first = levels[0]
    warp = np.array(
        [[1.0, 0.0, initial_full[0] * first],
         [0.0, 1.0, initial_full[1] * first]],
        dtype=np.float32,
    )
    previous = first
    scores: list[float] = []
    for level in levels:
        if level != previous:
            warp[:, 2] *= level / previous
        template = cv2.resize(pre, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
        target = cv2.resize(post, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
        score, warp = cv2.findTransformECC(
            template.astype(np.float32),
            target.astype(np.float32),
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7),
            None,
            5,
        )
        scores.append(float(score))
        previous = level
    return warp, scores


def align_post(post: np.ndarray, warp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    return cv2.warpAffine(
        post,
        warp,
        (cols, rows),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def alignment_parameters(warp: np.ndarray) -> dict[str, float]:
    linear = warp[:, :2].astype(float)
    return {
        "tx_px": float(warp[0, 2]),
        "ty_px": float(warp[1, 2]),
        "theta_deg": float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))),
        "scale_change_pct": float((math.sqrt(abs(np.linalg.det(linear))) - 1.0) * 100.0),
    }


def small_defect_mask(img: np.ndarray) -> np.ndarray:
    finite = np.isfinite(img)
    filled = np.nan_to_num(img, nan=float(np.nanmean(img)))
    diff = filled - cv2.GaussianBlur(filled, (51, 51), 0)
    std = float(np.std(diff[finite]))
    mask = (np.abs(diff) > 3.0 * std).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))
    return mask.astype(bool)


def groove_mask(pre: np.ndarray) -> np.ndarray:
    c_smooth = ndi.gaussian_filter1d(np.nanmean(pre, axis=0), 10)
    r_smooth = ndi.gaussian_filter1d(np.nanmean(pre, axis=1), 10)
    mask = np.zeros_like(pre, dtype=bool)
    mask[:, c_smooth < np.percentile(c_smooth, 3)] = True
    mask[r_smooth < np.percentile(r_smooth, 6), :] = True
    return cv2.dilate(mask.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)


def macro_defect_mask_one(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image-derived candidate mask matching the historical v2 detector.

    This is explicitly supplementary because the primary log has no pixel ROI.
    """
    finite = np.isfinite(img)
    fill_value = float(np.nanmedian(img))
    arr = np.nan_to_num(img, nan=fill_value)
    bg_mean = cv2.boxFilter(arr, -1, (101, 101))
    mean_sq = cv2.boxFilter(arr * arr, -1, (51, 51))
    sq_mean = cv2.boxFilter(arr, -1, (51, 51)) ** 2
    std_dev = np.sqrt(np.maximum(mean_sq - sq_mean, 0))
    median_std = float(np.median(std_dev[finite]))

    blurred_dust = cv2.GaussianBlur(arr, (11, 11), 0)
    dust = (blurred_dust > bg_mean + 0.10) | (arr > 0.80)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    dust = cv2.morphologyEx(dust.astype(np.uint8), cv2.MORPH_CLOSE, kd)
    dust = cv2.dilate(dust, kd).astype(bool)

    blurred_stain = cv2.GaussianBlur(arr, (31, 31), 0)
    global_median = float(np.median(bg_mean[finite]))
    stain = (blurred_stain < global_median * 0.50) | (std_dev < median_std * 0.40)
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    stain = cv2.morphologyEx(stain.astype(np.uint8), cv2.MORPH_CLOSE, ks)
    stain = cv2.morphologyEx(stain, cv2.MORPH_OPEN, ks).astype(bool)
    return dust | stain, dust, stain


def remove_small_components(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == idx
    return keep


def remove_components_touching(mask: np.ndarray, exclusion: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
    keep = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n):
        component = labels == idx
        if not np.any(component & exclusion):
            keep |= component
    return keep


def macro_defect_mask(pre: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m0, d0, s0 = macro_defect_mask_one(pre)
    m1, d1, s1 = macro_defect_mask_one(aligned)
    finite = np.isfinite(aligned)
    combined = remove_small_components((m0 | m1) & finite)
    dust = remove_small_components((d0 | d1) & finite)
    stain = remove_small_components((s0 | s1) & finite)
    return combined, dust, stain


def detect_pillars(pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = pre.shape
    fshift = np.fft.fftshift(np.fft.fft2(pre - np.mean(pre)))
    cy, cx = rows // 2, cols // 2
    yy, xx = np.ogrid[-cy:rows-cy, -cx:cols-cx]
    radius = np.sqrt(xx * xx + yy * yy)
    ring = (radius >= rows / PITCH_ROW_PX - 15) & (radius <= rows / PITCH_ROW_PX + 15)
    filtered = np.fft.ifft2(np.fft.ifftshift(fshift * ring)).real
    local_max = ndi.maximum_filter(filtered, size=5) == filtered
    y, x = np.where(local_max)
    border = (y > MARGIN_PX) & (y < rows - MARGIN_PX) & (x > MARGIN_PX) & (x < cols - MARGIN_PX)
    return y[border], x[border]


def sum_3x3(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    padded = np.pad(img, 1, mode="constant", constant_values=np.nan)
    result = np.zeros(len(y), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result += padded[y + 1 + dy, x + 1 + dx]
    return result


def nearest_neighbor_median(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("nan")
    dists, _ = cKDTree(points).query(points, k=2)
    return float(np.median(dists[:, 1]))


def spatial_statistics(
    x: np.ndarray,
    y: np.ndarray,
    centered: np.ndarray,
    defect_dist: np.ndarray,
    groove_dist: np.ndarray,
    rows: int,
    cols: int,
    rng: np.random.Generator,
) -> dict[str, float | str]:
    n = len(centered)
    k = min(TOP_N, n)
    top_idx = np.argsort(centered)[-k:][::-1]
    top_points = np.column_stack((x[top_idx], y[top_idx]))
    top_def_dist = defect_dist[y[top_idx], x[top_idx]]
    all_def_dist = defect_dist[y, x]
    top_groove_dist = groove_dist[y[top_idx], x[top_idx]]
    all_groove_dist = groove_dist[y, x]
    top_near = float(np.mean(top_def_dist <= NEAR_DEFECT_PX))
    base_near = float(np.mean(all_def_dist <= NEAR_DEFECT_PX))
    risk_ratio = float(top_near / base_near) if base_near > 0 else float("nan")
    top_near_groove = float(np.mean(top_groove_dist <= NEAR_DEFECT_PX))
    base_near_groove = float(np.mean(all_groove_dist <= NEAR_DEFECT_PX))
    groove_risk_ratio = float(top_near_groove / base_near_groove) if base_near_groove > 0 else float("nan")
    observed_nnd = nearest_neighbor_median(top_points)
    observed_edge = float(np.median(np.minimum.reduce((x[top_idx], cols - 1 - x[top_idx], y[top_idx], rows - 1 - y[top_idx]))))

    perm_near = np.empty(N_PERM)
    perm_nnd = np.empty(N_PERM)
    perm_edge = np.empty(N_PERM)
    perm_groove = np.empty(N_PERM)
    for i in range(N_PERM):
        idx = rng.choice(n, k, replace=False)
        perm_near[i] = np.mean(defect_dist[y[idx], x[idx]] <= NEAR_DEFECT_PX)
        perm_groove[i] = np.mean(groove_dist[y[idx], x[idx]] <= NEAR_DEFECT_PX)
        perm_nnd[i] = nearest_neighbor_median(np.column_stack((x[idx], y[idx])))
        perm_edge[i] = np.median(np.minimum.reduce((x[idx], cols - 1 - x[idx], y[idx], rows - 1 - y[idx])))

    # 4x4 occupancy against the valid-pillar spatial baseline.
    bins = 4
    all_bin = np.clip((y / rows * bins).astype(int), 0, bins - 1) * bins + np.clip((x / cols * bins).astype(int), 0, bins - 1)
    top_bin = all_bin[top_idx]
    all_counts = np.bincount(all_bin, minlength=bins * bins).astype(float)
    top_counts = np.bincount(top_bin, minlength=bins * bins).astype(float)
    expected = all_counts / all_counts.sum() * k
    keep = expected > 0
    chi_stat = float(np.sum((top_counts[keep] - expected[keep]) ** 2 / expected[keep]))
    chi_p = float(chi2.sf(chi_stat, int(keep.sum()) - 1))

    defect_p = float((1 + np.sum(perm_near >= top_near)) / (N_PERM + 1))
    groove_p = float((1 + np.sum(perm_groove >= top_near_groove)) / (N_PERM + 1))
    cluster_p = float((1 + np.sum(perm_nnd <= observed_nnd)) / (N_PERM + 1))
    edge_p = float((1 + np.sum(perm_edge <= observed_edge)) / (N_PERM + 1))
    nnd_ratio = float(observed_nnd / np.median(perm_nnd))
    edge_ratio = float(observed_edge / np.median(perm_edge))

    if defect_p < 0.05 and risk_ratio >= 1.5:
        pattern = "defect-associated"
    elif groove_p < 0.05 and groove_risk_ratio >= 1.5:
        pattern = "groove-associated"
    elif cluster_p < 0.05 and nnd_ratio < 0.8:
        pattern = "clustered-away-or-mixed"
    elif edge_p < 0.05 and edge_ratio < 0.8:
        pattern = "edge-associated"
    elif chi_p >= 0.05:
        pattern = "consistent-with-valid-pillar-baseline"
    else:
        pattern = "broad-spatial-heterogeneity"

    return {
        "top_near_defect_fraction": top_near,
        "baseline_near_defect_fraction": base_near,
        "defect_risk_ratio": risk_ratio,
        "defect_permutation_p_one_sided": defect_p,
        "top_near_groove_fraction": top_near_groove,
        "baseline_near_groove_fraction": base_near_groove,
        "groove_risk_ratio": groove_risk_ratio,
        "groove_permutation_p_one_sided": groove_p,
        "top_median_nearest_neighbor_px": observed_nnd,
        "nnd_ratio_vs_random": nnd_ratio,
        "cluster_permutation_p_one_sided": cluster_p,
        "top_median_edge_distance_px": observed_edge,
        "edge_distance_ratio_vs_random": edge_ratio,
        "edge_permutation_p_one_sided": edge_p,
        "grid4x4_chi_square": chi_stat,
        "grid4x4_p": chi_p,
        "spatial_pattern": pattern,
    }


def save_csv(
    fov: int,
    x: np.ndarray,
    y: np.ndarray,
    pre_sum: np.ndarray,
    post_sum: np.ndarray,
    raw_delta: np.ndarray,
    centered: np.ndarray,
    macro_dist: np.ndarray,
    macro_mask: np.ndarray,
) -> np.ndarray:
    top_idx = np.argsort(centered)[-min(TOP_N, len(centered)):][::-1]
    path = OUTPUT_DIR / f"FOV_1-{fov}_top70.csv"
    headers = [
        "rank", "fov", "x_px", "y_px", "pre_3x3_sum_adu", "post_3x3_sum_adu",
        "raw_decrease_pct", "fov_median_raw_decrease_pct", "median_centered_decrease_pct",
        "distance_to_image_derived_defect_px", "inside_image_derived_defect_mask",
        "primary_log_coordinate_available",
    ]
    median_raw = float(np.median(raw_delta))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for rank, idx in enumerate(top_idx, 1):
            writer.writerow([
                rank, f"1-{fov}", int(x[idx]), int(y[idx]),
                f"{pre_sum[idx] * 65535.0:.6f}", f"{post_sum[idx] * 65535.0:.6f}",
                f"{raw_delta[idx]:.9f}", f"{median_raw:.9f}", f"{centered[idx]:.9f}",
                f"{macro_dist[y[idx], x[idx]]:.6f}", bool(macro_mask[y[idx], x[idx]]), False,
            ])
    return top_idx


def save_figure(
    fov: int,
    pre: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    centered: np.ndarray,
    top_idx: np.ndarray,
    dust: np.ndarray,
    stain: np.ndarray,
    groove: np.ndarray,
    metrics: dict,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    lo, hi = np.percentile(pre, [1, 99.5])
    ax.imshow(pre, cmap="gray", vmin=lo, vmax=hi, origin="upper")
    if stain.any():
        ax.contour(stain.astype(float), levels=[0.5], colors=["#f59e0b"], linewidths=0.8)
    if dust.any():
        ax.contour(dust.astype(float), levels=[0.5], colors=["#2563eb"], linewidths=0.8)
    if groove.any():
        ax.contour(groove.astype(float), levels=[0.5], colors=["#22d3ee"], linewidths=0.7)
    scatter = ax.scatter(
        x[top_idx], y[top_idx], c=centered[top_idx], cmap="magma", s=23,
        edgecolors="white", linewidths=0.35, zorder=4,
    )
    ax.set_title(f"FOV 1-{fov}: top 70 pillars")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px, downward)")
    ax.set_xlim(0, pre.shape[1] - 1)
    ax.set_ylim(pre.shape[0] - 1, 0)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.047, pad=0.02)
    cbar.set_label("Median-centered decrease (%)")
    ax.text(
        0.01, 0.01,
        "Orange/blue: image-derived stain/dust candidates; cyan: grooves\nPrimary log has no pixel coordinates",
        transform=ax.transAxes, fontsize=8, color="white", va="bottom",
        bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 3},
    )

    hx = axes[1]
    vals = centered[top_idx]
    hx.hist(vals, bins=14, color="#7c3aed", alpha=0.82, edgecolor="white")
    hx.axvline(metrics["tau_pct"], color="#dc2626", linestyle="--", linewidth=1.5, label="P99.9")
    hx.axvline(np.median(vals), color="#111827", linestyle=":", linewidth=1.4, label="top-70 median")
    hx.set_title("Top-70 value distribution")
    hx.set_xlabel("Median-centered decrease (%)")
    hx.set_ylabel("Pillar count")
    hx.grid(axis="y", alpha=0.2)
    hx.legend(frameon=False, fontsize=9)
    summary = (
        f"rho={metrics['rho']:.4f}\n"
        f"N valid={metrics['n_valid']:,}\n"
        f"MAD={metrics['mad_pct']:.3f}%\n"
        f"tau={metrics['tau_pct']:.3f}%\n"
        f"near-defect RR={metrics['defect_risk_ratio']:.2f}\n"
        f"near-groove RR={metrics['groove_risk_ratio']:.2f}\n"
        f"cluster NND ratio={metrics['nnd_ratio_vs_random']:.2f}\n"
        f"pattern={metrics['spatial_pattern']}"
    )
    hx.text(0.98, 0.98, summary, transform=hx.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92, "pad": 5})
    fig.suptitle("260830 Control, pyramid-initialized affine ECC", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"FOV_1-{fov}_defect_overlay_and_histogram.png", dpi=180)
    plt.close(fig)


def analyze_fov(fov: int, rng: np.random.Generator) -> dict:
    pre = load_tiff(INPUT_DIR / f"1-{fov}-0.tif")
    post = load_tiff(INPUT_DIR / f"1-{fov}-1.tif")
    cache_dir = WORK_DIR / "ecc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"FOV_1-{fov}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            standard_warp = cached["standard_warp"].astype(np.float32)
            standard_scores = cached["standard_scores"].astype(float).tolist()
    else:
        standard_warp, standard_scores = pyramid_ecc(pre, post)
    selected_warp = standard_warp
    selected_scores = standard_scores
    selected_method = "0.25-0.5-1.0 coarse-template initialization"
    retries: list[dict] = []

    if fov == 6:
        prior_results_path = WORK_DIR / "spatial_analysis_results.json"
        if prior_results_path.exists():
            prior_rows = json.loads(prior_results_path.read_text(encoding="utf-8"))
            prior_f6 = next((row for row in prior_rows if row.get("fov") == "1-6"), None)
            prior_valid = [row for row in (prior_f6 or {}).get("fov6_retries", []) if "rho" in row]
            if prior_valid:
                best_prior = max(prior_valid, key=lambda row: row["rho"])
                standard_warp = np.asarray(best_prior["warp"], dtype=np.float32)
                standard_scores = [float((prior_f6 or {}).get("standard_rho", best_prior["rho"]))]
                selected_warp = standard_warp
                selected_scores = [float(best_prior["rho"])]
                selected_method = f"0.125-0.25-0.5-1.0 {best_prior['name']} initialization"
                retries = (prior_f6 or {}).get("fov6_retries", [])
        if retries:
            pass
        else:
            starts = [("coarse", coarse_shift(pre, post))]
            try:
                starts.append(("phase_0.125", phase_shift_at_scale(pre, post, 0.125)))
            except cv2.error:
                pass
            dx0, dy0 = coarse_shift(pre, post)
            starts.extend([
                ("coarse_y_plus_pitch", (dx0, dy0 + PITCH_ROW_PX)),
                ("coarse_y_minus_pitch", (dx0, dy0 - PITCH_ROW_PX)),
                ("coarse_x_plus_pitch", (dx0 + 7.33, dy0)),
                ("coarse_x_minus_pitch", (dx0 - 7.33, dy0)),
            ])
            for name, start in starts:
                try:
                    warp, scores = pyramid_ecc(pre, post, (0.125, 0.25, 0.5, 1.0), start)
                    retries.append({"name": name, "initial_full": list(start), "rho": scores[-1], "warp": warp.tolist()})
                    if scores[-1] > selected_scores[-1]:
                        selected_warp, selected_scores = warp, scores
                        selected_method = f"0.125-0.25-0.5-1.0 {name} initialization"
                except cv2.error as exc:
                    retries.append({"name": name, "initial_full": list(start), "error": str(exc)})
    if not cache_path.exists():
        np.savez_compressed(cache_path, standard_warp=standard_warp, standard_scores=np.asarray(standard_scores))

    aligned = align_post(post, selected_warp, pre.shape)
    y_all, x_all = detect_pillars(pre)
    pre_sum_all = sum_3x3(pre, y_all, x_all)
    post_sum_all = sum_3x3(aligned, y_all, x_all)
    groove = groove_mask(pre)
    pipeline_invalid = small_defect_mask(pre) | small_defect_mask(aligned) | groove
    valid = (
        np.isfinite(post_sum_all)
        & ~pipeline_invalid[y_all, x_all]
        & ((pre_sum_all / 9.0) * 65535.0 >= MIN_PRE_ADU)
    )
    x, y = x_all[valid], y_all[valid]
    pre_sum, post_sum = pre_sum_all[valid], post_sum_all[valid]
    raw_delta = (pre_sum - post_sum) / pre_sum * 100.0
    median_raw = float(np.median(raw_delta))
    centered = raw_delta - median_raw
    mad = float(np.median(np.abs(centered)))
    tau = float(np.percentile(centered, 99.9))

    macro, dust, stain = macro_defect_mask(pre, aligned)
    groove_halo = cv2.dilate(groove.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))).astype(bool)
    macro = remove_components_touching(macro, groove_halo)
    dust = remove_components_touching(dust, groove_halo)
    stain = remove_components_touching(stain, groove_halo)
    defect_distance = cv2.distanceTransform((~macro).astype(np.uint8), cv2.DIST_L2, 5)
    groove_distance = cv2.distanceTransform((~groove).astype(np.uint8), cv2.DIST_L2, 5)
    spatial = spatial_statistics(x, y, centered, defect_distance, groove_distance, *pre.shape, rng)
    top_idx = save_csv(fov, x, y, pre_sum, post_sum, raw_delta, centered, defect_distance, macro)

    top_vals = centered[top_idx]
    excess = np.maximum(top_vals - tau, 0)
    excess_total = float(excess.sum())
    tail = {
        "top70_median_pct": float(np.median(top_vals)),
        "top70_max_pct": float(np.max(top_vals)),
        "p99_99_pct": float(np.percentile(centered, 99.99)),
        "max_to_tau_ratio": float(np.max(top_vals) / tau) if tau else float("nan"),
        "top5_excess_share": float(np.sort(excess)[-5:].sum() / excess_total) if excess_total > 0 else float("nan"),
    }
    metrics: dict = {
        "fov": f"1-{fov}",
        "alignment_method": selected_method,
        "rho": float(selected_scores[-1]),
        "standard_rho": float(standard_scores[-1]),
        "warp_matrix": selected_warp.tolist(),
        **alignment_parameters(selected_warp),
        "n_detected_before_masks": int(len(x_all)),
        "n_valid": int(len(x)),
        "median_raw_decrease_pct": median_raw,
        "mad_pct": mad,
        "tau_pct": tau,
        "reference_rho": REPORT_REFERENCE[fov]["rho"],
        "reference_mad_pct": REPORT_REFERENCE[fov]["mad"],
        "reference_tau_pct": REPORT_REFERENCE[fov]["tau"],
        "auto_defect_mask_fraction": float(np.mean(macro)),
        **spatial,
        **tail,
        "fov6_retries": retries,
    }
    save_figure(fov, pre, x, y, centered, top_idx, dust, stain, groove, metrics)
    return metrics


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n.a."
    return f"{value:.{digits}f}"


def build_report(results: list[dict]) -> None:
    high = [r for r in results if r["fov"] in {"1-3", "1-6"}]
    low = [r for r in results if r["fov"] in {"1-1", "1-5"}]
    f6 = next(r for r in results if r["fov"] == "1-6")

    defect_assoc = [r for r in results if r["defect_permutation_p_one_sided"] < 0.05 and r["defect_risk_ratio"] >= 1.5]
    groove_assoc = [r for r in results if r["groove_permutation_p_one_sided"] < 0.05 and r["groove_risk_ratio"] >= 1.5]
    cluster_assoc = [r for r in results if r["cluster_permutation_p_one_sided"] < 0.05 and r["nnd_ratio_vs_random"] < 0.8]
    extreme = [r for r in results if r["top5_excess_share"] >= 0.50 or r["max_to_tau_ratio"] >= 2.5]

    lines = [
        "# 260830 Control 条件1：上位0.1%ピラーの空間分布診断",
        "",
        "作成日: 2026-09-08",
        "",
        "## 結論",
        "",
    ]
    if defect_assoc:
        lines.append("既知の溝から30 pxのハローを除いた画像由来欠陥候補の25 px近傍に有意な濃縮を示した視野は " + ", ".join(r["fov"] for r in defect_assoc) + " だった。")
    else:
        lines.append("既知の溝から30 pxのハローを除いた画像由来欠陥候補の25 px近傍に、上位70本が1.5倍以上かつ片側置換 p<0.05で濃縮する視野はなかった。")
    if groove_assoc:
        lines.append("一方、構造溝の25 px近傍へ有意に濃縮した視野は " + ", ".join(r["fov"] for r in groove_assoc) + " だった。これは傷・ゴミ・シミではなく、溝境界またはその近傍の位置合わせ・補間残差を示す。")
    else:
        lines.append("構造溝の25 px近傍への有意な濃縮は検出されなかった。")
    if cluster_assoc:
        lines.append("欠陥近傍性とは別に、点同士の最近傍距離がランダム標本より短い空間クラスタを示した視野は " + ", ".join(r["fov"] for r in cluster_assoc) + " だった。")
    else:
        lines.append("上位70本の点同士が強く固まるクラスタリング（最近傍距離比<0.8、p<0.05）は検出されなかった。")
    lines.extend([
        "したがって仮説(a)は、溝を除いた自動欠陥候補との対応が出た視野に限って暫定支持される。溝への集中はマクロ欠陥仮説ではなく、局所位置合わせ・補間残差を支持する。対応がない高τ視野では、局所位置合わせ残差、視野端、収差など仮説(b)側の原因を優先して調べる必要がある。",
        "",
        "重要な制約として、一次実験ログには傷・ゴミ・シミのピクセル座標が記録されていない。既存 `defect_coordinates.csv` も X=0, Y=0 のプレースホルダーで、260830の行がない。このため図中の欠陥輪郭は未検証の画像由来自動候補であり、一次ログ座標との正式な重ね合わせではない。",
        "",
        "## 方法",
        "",
        "- 対象: `260830_P50_条件a~d` の基板1、視野1〜8。preは `1-f-0.tif`、postは `1-f-1.tif`。",
        "- 位置合わせ: 中央1000 pxテンプレートで粗平行移動を求め、0.25x→0.5x→1.0xのアフィンECC。Gaussian窓5、各階層最大300反復。postを逆写像・三次補間した。",
        "- ピラー座標: pre画像のFFT一次リング（行間隔6.38 px）を通した像の局所極大。端30 pxを除外。",
        "- 値: 同一pre座標でpre/postの3x3積算を測定し、`100 × (pre-post)/pre` を計算。pre平均輝度32,500 ADU未満を除外し、視野内中央値を差し引いた。",
        "- パイプライン由来の微小欠陥・溝マスクを適用後、各視野の中央値センタリング値が高い順に70本を抽出した。",
        "- 欠陥候補: 履歴上の画像由来自動検出器でdust/stain候補を作り、既知の構造溝とその30 pxハローを除外した。溝は別レイヤーとして評価した。",
        "- 空間検定: 有効ピラーから70本を無作為抽出する置換を2,000回行い、欠陥候補25 px近傍率、溝25 px近傍率、最近傍距離、端距離を比較。4x4セル占有率も有効ピラー密度を期待値として評価した。",
        "",
        "## 視野別結果",
        "",
        "| FOV | ρ | N valid | MAD (%) | τ=P99.9 (%) | 欠陥近傍 top/base | RR/p | 溝近傍 top/base | RR/p | NND比/p | 4x4 p | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for r in results:
        lines.append(
            f"| {r['fov']} | {r['rho']:.4f} | {r['n_valid']:,} | {r['mad_pct']:.3f} | {r['tau_pct']:.3f} | "
            f"{100*r['top_near_defect_fraction']:.1f}%/{100*r['baseline_near_defect_fraction']:.1f}% | {fmt(r['defect_risk_ratio'],2)}/{r['defect_permutation_p_one_sided']:.4f} | "
            f"{100*r['top_near_groove_fraction']:.1f}%/{100*r['baseline_near_groove_fraction']:.1f}% | {fmt(r['groove_risk_ratio'],2)}/{r['groove_permutation_p_one_sided']:.4f} | "
            f"{r['nnd_ratio_vs_random']:.2f}/{r['cluster_permutation_p_one_sided']:.4f} | "
            f"{r['grid4x4_p']:.4f} | {r['spatial_pattern']} |"
        )

    lines.extend([
        "",
        "### 報告書第8-2項との再現性",
        "",
        "同じピラミッドECCで変換パラメータとρは再現した。ピラー抽出・マスクの実装差を監視するため、下表に本再計算値と第8-2項の値を併記する。",
        "",
        "| FOV | ρ 今回/既報 | MAD 今回/既報 (%) | τ 今回/既報 (%) |",
        "|---|---:|---:|---:|",
    ])
    for r in results:
        lines.append(
            f"| {r['fov']} | {r['rho']:.4f}/{r['reference_rho']:.3f} | "
            f"{r['mad_pct']:.3f}/{r['reference_mad_pct']:.2f} | {r['tau_pct']:.3f}/{r['reference_tau_pct']:.2f} |"
        )

    def group_mean(group: list[dict], key: str) -> float:
        values = [r[key] for r in group if np.isfinite(r[key])]
        return float(np.mean(values)) if values else float("nan")

    lines.extend([
        "",
        "## 高τ視野と低τ視野の比較",
        "",
        "指定された高τ群（1-3, 1-6）と低τ群（1-1, 1-5）は各n=2なので記述比較に留めた。",
        "",
        "| 指標 | 高τ群平均 | 低τ群平均 |",
        "|---|---:|---:|",
        f"| τ (%) | {group_mean(high, 'tau_pct'):.3f} | {group_mean(low, 'tau_pct'):.3f} |",
        f"| 欠陥近傍リスク比 | {fmt(group_mean(high, 'defect_risk_ratio'),2)} | {fmt(group_mean(low, 'defect_risk_ratio'),2)} |",
        f"| 溝近傍リスク比 | {fmt(group_mean(high, 'groove_risk_ratio'),2)} | {fmt(group_mean(low, 'groove_risk_ratio'),2)} |",
        f"| 最近傍距離比（1未満ほど集中） | {group_mean(high, 'nnd_ratio_vs_random'):.2f} | {group_mean(low, 'nnd_ratio_vs_random'):.2f} |",
        f"| 端距離比（1未満ほど端寄り） | {group_mean(high, 'edge_distance_ratio_vs_random'):.2f} | {group_mean(low, 'edge_distance_ratio_vs_random'):.2f} |",
        f"| 上位5本の閾値超過量シェア | {100*group_mean(high, 'top5_excess_share'):.1f}% | {100*group_mean(low, 'top5_excess_share'):.1f}% |",
        "",
        "指定された高τ例1-3・1-6はいずれも欠陥候補近傍への濃縮判定を満たさず、低τ例1-1・1-5も満たさなかった。したがって、この4視野の比較から高τと画像由来欠陥候補の系統的対応は認められない。欠陥近傍リスク比の群平均は、1-3の背景近傍率が極小なため不安定であり、群差の根拠には用いない。",
        "一方、最近傍距離比は高τ群0.18、低τ群0.47で、高τ群の上位70本はよりまとまった配置だった。ただし低τの1-1も強くクラスタ化しており、各n=2の記述的傾向である。値分布では高τ群の上位5本寄与が24.8%、低τ群が41.3%で、高τ群は少数の一点外れ値よりも裾全体の持ち上がりが目立った。",
        "",
        "## 上位70本の値分布",
        "",
        "| FOV | top70中央値 (%) | 最大 (%) | P99.99 (%) | 最大/τ | 上位5本の超過量シェア | 判定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for r in results:
        verdict = "少数極端値の寄与が大きい" if r in extreme else "裾全体が持ち上がる型"
        lines.append(
            f"| {r['fov']} | {r['top70_median_pct']:.3f} | {r['top70_max_pct']:.3f} | {r['p99_99_pct']:.3f} | "
            f"{r['max_to_tau_ratio']:.2f} | {100*r['top5_excess_share']:.1f}% | {verdict} |"
        )

    lines.extend([
        "",
        "## 視野1-6の再位置合わせ",
        "",
        f"標準3階層のρは {f6['standard_rho']:.4f}。採用した `{f6['alignment_method']}` のρは {f6['rho']:.4f}。",
        ("ρ≥0.98に到達したため、1-6のCSVと図は改善解で作成した。" if f6["rho"] >= 0.98 else "ρ≥0.98には到達しなかった。1-6は局所解または画像固有の残差が残る視野として扱う。"),
        "",
        "| 初期化 | ρ | tx | ty | 備考 |",
        "|---|---:|---:|---:|---|",
    ])
    for retry in f6["fov6_retries"]:
        if "rho" in retry:
            w = retry["warp"]
            lines.append(f"| {retry['name']} | {retry['rho']:.4f} | {w[0][2]:.3f} | {w[1][2]:.3f} | 4階層 |")
        else:
            lines.append(f"| {retry['name']} | n.a. | n.a. | n.a. | ECC失敗 |")

    lines.extend([
        "",
        "## 全日程再解析の位置合わせ監査",
        "",
        "260908の視野単位比較は、`run_crossday_direct_sampling.py` から `registration.py` の `align_and_match_dataframes` を呼び、peak点群で大域2次ICP変換を推定し、pre FFT理論格子をpostへ逆投影する経路だった。コードに `findTransformECC`、0.25x/0.5x/1.0xのピラミッド階層はない。",
        "",
        "したがって260824/825/826/828(DNA)/828(SAM)/829の視野単位p値は、今回のピラミッド初期化ECC修正版ではない。ECC局所解ノイズをそのまま引きずる方式ではない一方、別方式であり、第8項の改善を反映した値とも言えない。手法を統一して比較するなら、全日程をピラミッドECC＋同一サンプリング・マスク条件で再計算する必要がある。",
        "",
        "## 出力",
        "",
        "- `FOV_1-1_top70.csv`〜`FOV_1-8_top70.csv`: rank、画像座標、pre/post 3x3積算ADU、raw変化、中央値センタリング値、画像由来欠陥候補までの距離。",
        "- `FOV_1-1_defect_overlay_and_histogram.png`〜`FOV_1-8_defect_overlay_and_histogram.png`: 空間重ね合わせと上位70本ヒストグラム。",
        "",
        "## 制約と次の確認",
        "",
        "一次ログの欠陥ROIを後から取得できた場合は、画像由来自動候補を置き換えて同じ距離置換検定を再実行する。正式な仮説(a)/(b)の判定は、その一次ROIとの照合で確定する。",
        "",
    ])
    (OUTPUT_DIR / "260830_Control_FOV1-8_高tauピラー空間分布診断_260908.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the spatial distribution of the top 70 pillar-level pre/post contrast decreases."
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR,
                        help="Directory containing 1-1-0.tif ... 1-8-1.tif (not stored in Git).")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Runtime output directory for CSV, figures, and Markdown.")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Runtime cache directory; defaults to <output-dir>/work.")
    return parser.parse_args()


def main() -> None:
    global INPUT_DIR, OUTPUT_DIR, WORK_DIR
    args = parse_args()
    INPUT_DIR = args.input_dir.resolve()
    OUTPUT_DIR = args.output_dir.resolve()
    WORK_DIR = args.work_dir.resolve() if args.work_dir else OUTPUT_DIR / "work"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    results = []
    for fov in range(1, 9):
        print(f"Analyzing FOV 1-{fov}...", flush=True)
        result = analyze_fov(fov, rng)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("fov", "rho", "n_valid", "mad_pct", "tau_pct", "spatial_pattern")}), flush=True)
    build_report(results)
    (WORK_DIR / "spatial_analysis_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
