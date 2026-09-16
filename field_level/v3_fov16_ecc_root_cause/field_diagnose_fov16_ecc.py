"""Phase-2 diagnosis of the 260826 Sample-1 Position-6 ECC local solution.

This diagnostic does not alter production registration.  It reuses the repository's
sub-pixel peak detector and FFT lattice functions, while recording an independently
versioned ECC multi-start experiment plus image-quality diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import analyzer
from shared import lattice_indexing as li
from shared.registration import load_image_unicode


def angular_delta(a: float, b: float) -> float:
    """Difference in a hex-grid angle, reduced modulo 60 degrees."""
    return float((a - b + np.pi/6) % (np.pi/3) - np.pi/6)


def image_for_ecc(image: np.ndarray) -> np.ndarray:
    """Use the native 16-bit intensity scale for ECC, without extra filtering.

    The established 2026-09-01 ECC figures were obtained from the image content,
    not a newly introduced high-pass representation.  Keeping that condition is
    essential before comparing rho across pyramid strategies.
    """
    return (np.asarray(image, np.float32)/65535.0).astype(np.float32)


def full_from_level(warp: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(warp, np.float32).copy()
    out[:, 2] /= scale
    return out


def level_from_full(warp: np.ndarray, scale: float) -> np.ndarray:
    out = np.asarray(warp, np.float32).copy()
    out[:, 2] *= scale
    return out


def affine_metrics(warp: np.ndarray) -> dict:
    linear = np.asarray(warp[:, :2], float)
    singular = np.linalg.svd(linear, compute_uv=False)
    scale = math.sqrt(abs(np.linalg.det(linear)))
    rotation = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    return {"tx_px": float(warp[0, 2]), "ty_px": float(warp[1, 2]),
            "theta_deg": rotation, "delta_scale_pct": float((scale-1)*100),
            "anisotropy_pct": float((singular[0]/singular[1]-1)*100)}


def run_ecc_path(template_full: np.ndarray, input_full: np.ndarray, initial_tx: float,
                 initial_ty: float, scales: list[float], iterations: int = 250,
                 eps: float = 1e-7) -> tuple[dict, np.ndarray]:
    """Run affine ECC progressively from the coarsest scale to full resolution."""
    warp_full = np.array([[1, 0, initial_tx], [0, 1, initial_ty]], np.float32)
    levels = []
    for scale in scales:
        size = (round(template_full.shape[1]*scale), round(template_full.shape[0]*scale))
        template = cv2.resize(template_full, size, interpolation=cv2.INTER_AREA)
        input_image = cv2.resize(input_full, size, interpolation=cv2.INTER_AREA)
        warp_level = level_from_full(warp_full, scale)
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, iterations, eps)
        rho, warp_level = cv2.findTransformECC(template, input_image, warp_level,
                                                cv2.MOTION_AFFINE, criteria, None, 5)
        warp_full = full_from_level(warp_level, scale)
        levels.append({"scale": scale, "rho": float(rho), **affine_metrics(warp_full)})
    return {"rho": levels[-1]["rho"], "levels": levels, **affine_metrics(warp_full)}, warp_full


def run_ecc_grid(pre: np.ndarray, post: np.ndarray, shifts: list[float], scales: list[float],
                 label: str) -> tuple[pd.DataFrame, dict, np.ndarray]:
    template, input_image = image_for_ecc(pre), image_for_ecc(post)
    rows, warps = [], {}
    for tx0 in shifts:
        for ty0 in shifts:
            row = {"mode": label, "initial_tx_px": tx0, "initial_ty_px": ty0}
            try:
                result, warp = run_ecc_path(template, input_image, tx0, ty0, scales)
                row.update({k: v for k, v in result.items() if k != "levels"})
                row["converged"] = True
                row["level_history_json"] = json.dumps(result["levels"])
                warps[(tx0, ty0)] = warp
            except cv2.error as exc:
                row.update({"converged": False, "rho": np.nan, "error": str(exc)})
            rows.append(row)
    frame = pd.DataFrame(rows)
    valid = frame[frame["converged"]].sort_values("rho", ascending=False)
    if valid.empty:
        raise RuntimeError(f"No ECC path converged for {label}.")
    best = valid.iloc[0].to_dict()
    return frame, best, warps[(best["initial_tx_px"], best["initial_ty_px"])]


def fft_orientation_quality(image: np.ndarray, pitch: float = 7.286, tiles: int = 5) -> dict:
    """Global orientation plus tile-bootstrap stability and reciprocal peak sharpness."""
    angle = li.estimate_hex_orientation_fft(image, pitch)
    h, w = image.shape
    estimates = []
    for row in range(tiles):
        for col in range(tiles):
            crop = image[round(row*h/tiles):round((row+1)*h/tiles),
                         round(col*w/tiles):round((col+1)*w/tiles)]
            try:
                estimates.append(li.estimate_hex_orientation_fft(crop, pitch))
            except ValueError:
                pass
    deviations = np.array([angular_delta(x, angle) for x in estimates])
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image-image.mean()))))
    yy, xx = np.indices(image.shape)
    fy = (yy-image.shape[0]//2)/image.shape[0]
    fx = (xx-image.shape[1]//2)/image.shape[1]
    radius = np.hypot(fx, fy)
    expected = 2/(np.sqrt(3)*pitch)
    annulus = (radius > expected*.85) & (radius < expected*1.15)
    direction = np.arctan2(fy, fx)
    peak_values = []
    for k in range(6):
        target = angle-np.pi/6+k*np.pi/3
        error = np.abs(np.angle(np.exp(1j*(direction-target))))
        wedge = annulus & (error < np.deg2rad(8))
        peak_values.append(float(spectrum[wedge].max()))
    annular_median, annular_std = np.median(spectrum[annulus]), np.std(spectrum[annulus])
    return {"angle_deg": math.degrees(angle), "tile_count": len(estimates),
            "tile_angle_sd_deg": math.degrees(np.std(deviations, ddof=1)) if len(deviations) > 1 else np.nan,
            "tile_angle_max_abs_deg": math.degrees(np.max(np.abs(deviations))) if len(deviations) else np.nan,
            "fft_peak_snr": float((np.median(peak_values)-annular_median)/max(annular_std, 1e-9)),
            "fft_peak_min_to_median": float(min(peak_values)/np.median(peak_values))}


def focus_metrics(image: np.ndarray) -> dict:
    x = np.asarray(image, np.float32)/65535.0
    # This OpenCV build does not implement float32→float64 Laplacian kernels;
    # float32 preserves the diagnostic exactly for this image range.
    laplacian_var = float(cv2.Laplacian(x, cv2.CV_32F).var())
    gx, gy = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return {"laplacian_variance": laplacian_var, "tenengrad_mean": float(np.mean(gx*gx+gy*gy))}


def detect_quality(path: str, image: np.ndarray, pitch: float, output: Path, label: str):
    """Use current detector and quantify unique cells, duplicates, and missing cells."""
    detections = analyzer.analyze_image(path, method="peak", min_dist=3, threshold=.2)
    points = detections[["x", "y"]].to_numpy(float)
    lattice = li.lattice_from_fft(image, pitch)
    grid_indices, grid_xy = li.grid_coordinates(lattice, image.shape[1], image.shape[0], margin=30)
    assigned = li.assign_lattice_indices(points, lattice, max_distance=1.5)
    accepted = assigned["within_distance"]
    selected = assigned["selected"]
    selected_cells = {tuple(x) for x in assigned["indices"][selected]}
    grid_cells = {tuple(x) for x in grid_indices}
    unique = len(selected_cells & grid_cells)
    accepted_count = int(accepted.sum())
    h, w = image.shape
    rows = []
    for row in range(8):
        for col in range(8):
            in_grid = ((grid_xy[:, 0] >= col*w/8) & (grid_xy[:, 0] < (col+1)*w/8) &
                       (grid_xy[:, 1] >= row*h/8) & (grid_xy[:, 1] < (row+1)*h/8))
            cells = {tuple(x) for x in grid_indices[in_grid]}
            found = len(cells & selected_cells)
            rows.append({"image": label, "tile_row": row, "tile_col": col,
                         "theoretical_cells": len(cells), "unique_detected_cells": found,
                         "detection_rate": found/max(len(cells), 1), "missing_rate": 1-found/max(len(cells), 1)})
    tile_frame = pd.DataFrame(rows)
    detections["lattice_distance_px"] = assigned["distance"]
    detections["within_lattice_distance"] = accepted
    detections["selected_unique_cell"] = selected
    detections.to_csv(output / f"{label}_detections.csv", index=False)
    tile_frame.to_csv(output / f"{label}_detection_tiles.csv", index=False)
    summary = {"image": label, "detected_peaks": int(len(detections)), "theoretical_cells": int(len(grid_cells)),
               "unique_detected_cells": unique, "detection_rate": unique/max(len(grid_cells), 1),
               "double_peak_count": int(assigned["duplicate"].sum()),
               "double_peak_rate": float(assigned["duplicate"].sum()/max(accepted_count, 1)),
               "missing_cell_rate": 1-unique/max(len(grid_cells), 1)}
    return detections, tile_frame, summary


def residual_quality(pre_detections: pd.DataFrame, post_detections: pd.DataFrame, warp: np.ndarray,
                     shape: tuple[int, int], output: Path, label: str) -> tuple[pd.DataFrame, dict]:
    """Map post detections into pre coordinates and expose spatial correspondence outliers."""
    pre_xy = pre_detections[["x", "y"]].to_numpy(float)
    post_xy = post_detections[["x", "y"]].to_numpy(np.float32)
    inverse = cv2.invertAffineTransform(warp)
    mapped = cv2.transform(post_xy[None, :, :], inverse)[0]
    distance, index = KDTree(pre_xy).query(mapped)
    h, w = shape
    table = pd.DataFrame({"post_x": post_xy[:, 0], "post_y": post_xy[:, 1],
                          "pre_match_x": pre_xy[index, 0], "pre_match_y": pre_xy[index, 1],
                          "mapped_pre_x": mapped[:, 0], "mapped_pre_y": mapped[:, 1],
                          "match_residual_px": distance, "matched": distance <= 1.5})
    table["tile_col"] = np.clip((table.post_x/(w/8)).astype(int), 0, 7)
    table["tile_row"] = np.clip((table.post_y/(h/8)).astype(int), 0, 7)
    table.to_csv(output / f"{label}_correspondence_residuals.csv", index=False)
    tiles = (table.groupby(["tile_row", "tile_col"], as_index=False)
             .agg(point_count=("matched", "size"), match_rate=("matched", "mean"),
                  median_residual_px=("match_residual_px", "median"),
                  p95_residual_px=("match_residual_px", lambda x: np.percentile(x, 95))))
    tiles.to_csv(output / f"{label}_correspondence_tiles.csv", index=False)
    return table, {"correspondence_match_rate": float(table.matched.mean()),
                   "correspondence_median_residual_px": float(table.match_residual_px.median()),
                   "correspondence_p95_residual_px": float(np.percentile(table.match_residual_px, 95))}


def save_heatmaps(quality_tiles: pd.DataFrame, residual_tiles: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for axis, label in zip(axes.flat[:3], ["fov11_pre", "fov15_pre", "fov16_pre"]):
        frame = quality_tiles[(quality_tiles.image == label)]
        image = frame.pivot(index="tile_row", columns="tile_col", values="missing_rate")
        shown = axis.imshow(image, vmin=0, vmax=.5, cmap="magma")
        axis.set(title=f"{label}: missing-cell rate", xlabel="tile col", ylabel="tile row")
        fig.colorbar(shown, ax=axis, fraction=.046)
    for axis, label in zip(axes.flat[3:6], ["fov11_post", "fov15_post", "fov16_post"]):
        frame = quality_tiles[(quality_tiles.image == label)]
        image = frame.pivot(index="tile_row", columns="tile_col", values="missing_rate")
        shown = axis.imshow(image, vmin=0, vmax=.5, cmap="magma")
        axis.set(title=f"{label}: missing-cell rate", xlabel="tile col", ylabel="tile row")
        fig.colorbar(shown, ax=axis, fraction=.046)
    for axis, value in zip(axes.flat[6:], ["match_rate", "median_residual_px", "p95_residual_px"]):
        image = residual_tiles.pivot(index="tile_row", columns="tile_col", values=value)
        shown = axis.imshow(image, cmap="viridis")
        axis.set(title=f"FOV 1-6: {value}", xlabel="tile col", ylabel="tile row")
        fig.colorbar(shown, ax=axis, fraction=.046)
    fig.savefig(output / "quality_and_residual_heatmaps.png", dpi=180)
    plt.close(fig)


def save_ecc_plot(grid: pd.DataFrame, output: Path):
    valid = grid[grid.converged].copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(valid.initial_tx_px, valid.initial_ty_px, c=valid.rho, s=120, cmap="viridis")
    for _, row in valid.iterrows():
        ax.annotate(f"{row.rho:.4f}", (row.initial_tx_px, row.initial_ty_px), ha="center", va="center", fontsize=7)
    fig.colorbar(scatter, ax=ax, label="ECC correlation ρ")
    ax.set(title="FOV 1-6: deep-pyramid ECC initial-translation grid", xlabel="initial tx (px)", ylabel="initial ty (px)")
    fig.savefig(output / "fov16_ecc_initial_translation_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/fov16_ecc_diagnostic_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    root = Path(config["root"])
    positions = [config["target_position"], *config["comparison_positions"]]
    images, paths = {}, {}
    for pos in positions:
        for phase in ("pre", "post"):
            path = root / f"{config['sample']}-{pos}-{0 if phase == 'pre' else 1}.tif"
            image = load_image_unicode(str(path))
            if image is None:
                raise FileNotFoundError(path)
            images[(pos, phase)], paths[(pos, phase)] = image, str(path)

    # FOV 1-6: 25 full-resolution initial translations through 0.125→1.0.
    ecc_grid, deep_best, deep_warp = run_ecc_grid(
        images[(config["target_position"], "pre")], images[(config["target_position"], "post")],
        config["initial_translation_grid_px"], config["deep_pyramid_scales"], "deep_0125_to_1")
    ecc_grid.to_csv(output / "fov16_ecc_deep_grid.csv", index=False)
    save_ecc_plot(ecc_grid, output)
    # Baseline three-level zero-start is retained for direct comparison, not replacement.
    standard_grid, standard_best, _ = run_ecc_grid(
        images[(config["target_position"], "pre")], images[(config["target_position"], "post")], [0],
        config["standard_pyramid_scales"], "standard_025_to_1_zero_start")
    standard_grid.to_csv(output / "fov16_ecc_standard_baseline.csv", index=False)
    comparison_ecc = {}
    for pos in config["comparison_positions"]:
        frame, best, warp = run_ecc_grid(images[(pos, "pre")], images[(pos, "post")], [0],
                                         config["deep_pyramid_scales"], f"fov1{pos}_deep_zero_start")
        frame.to_csv(output / f"fov1{pos}_ecc_deep_zero_start.csv", index=False)
        comparison_ecc[pos] = (best, warp)

    orientations, focus_rows, quality_rows, quality_tiles = [], [], [], []
    detection_frames = {}
    for pos in positions:
        for phase in ("pre", "post"):
            label = f"fov1{pos}_{phase}"
            image = images[(pos, phase)]
            orient = fft_orientation_quality(image, config["established_pitch_px"])
            orientations.append({"position": pos, "phase": phase, **orient})
            focus_rows.append({"position": pos, "phase": phase, **focus_metrics(image)})
            detections, tiles, quality = detect_quality(paths[(pos, phase)], image,
                                                        config["established_pitch_px"], output, label)
            detection_frames[(pos, phase)] = detections
            quality_rows.append({"position": pos, "phase": phase, **quality})
            quality_tiles.append(tiles)
    orientation_frame = pd.DataFrame(orientations)
    orientation_frame.to_csv(output / "fft_orientation_quality.csv", index=False)
    focus_frame = pd.DataFrame(focus_rows); focus_frame.to_csv(output / "focus_metrics.csv", index=False)
    quality_frame = pd.DataFrame(quality_rows); quality_frame.to_csv(output / "peak_detection_quality.csv", index=False)
    quality_tile_frame = pd.concat(quality_tiles, ignore_index=True); quality_tile_frame.to_csv(output / "peak_detection_quality_tiles.csv", index=False)
    # A physical angle difference is pre-minus-post; report it alongside ECC theta.
    orient_pivot = orientation_frame.pivot(index="position", columns="phase", values="angle_deg")
    angle_rows = []
    for pos in positions:
        angle_rows.append({"position": pos, "fft_pre_minus_post_deg": angular_delta(
            math.radians(orient_pivot.loc[pos, "pre"]), math.radians(orient_pivot.loc[pos, "post"])) * 180/np.pi})
    pd.DataFrame(angle_rows).to_csv(output / "fft_pre_post_angle_differences.csv", index=False)

    residual_frame, residual_summary = residual_quality(
        detection_frames[(config["target_position"], "pre")], detection_frames[(config["target_position"], "post")],
        deep_warp, images[(config["target_position"], "pre")].shape, output, "fov16")
    comparison_residuals = [{"position": config["target_position"], **residual_summary}]
    for pos, (_, warp) in comparison_ecc.items():
        _, summary = residual_quality(detection_frames[(pos, "pre")], detection_frames[(pos, "post")],
                                      warp, images[(pos, "pre")].shape, output, f"fov1{pos}")
        comparison_residuals.append({"position": pos, **summary})
    pd.DataFrame(comparison_residuals).to_csv(output / "correspondence_quality_comparison.csv", index=False)
    residual_tiles = pd.read_csv(output / "fov16_correspondence_tiles.csv")
    save_heatmaps(quality_tile_frame, residual_tiles, output)
    summary = {"fov": "260826_SAM_S1_P6", "deep_grid_converged": int(ecc_grid.converged.sum()),
               "deep_grid_total": int(len(ecc_grid)), "deep_best": deep_best,
               "standard_zero_start": standard_best, "comparison_ecc": {str(pos): best for pos, (best, _) in comparison_ecc.items()},
               "residual": residual_summary, "comparison_residuals": comparison_residuals,
               "known_reference": {"good_rho_floor": .978, "previous_fov16_rho": .971,
                                   "previous_fov16_delta_scale_pct": -.239,
                                   "previous_fov16_theta_deg": -.026}}
    (output / "phase2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x)),
        encoding="utf-8")


if __name__ == "__main__":
    main()
