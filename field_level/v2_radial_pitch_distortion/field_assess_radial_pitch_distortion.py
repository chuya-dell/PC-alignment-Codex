"""Measure whether p50 pillar pitch changes across a pre-image field of view.

This is deliberately diagnostic-only: it retains the established global pitch of
7.286 px and reuses ``shared.analyzer`` for the sub-pixel local-maximum
detections.  It does not change the production lattice/registration pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import KDTree
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import analyzer
from shared import lattice_indexing as li
from shared.registration import load_image_unicode

PITCH = 7.286
CENTRE = np.array([1023.5, 1021.5])  # geometric centre of 2048 x 2044 image
RADIAL_EDGES = np.array([0, 200, 400, 600, 800, 1000, np.inf])


def image_records(manifest: dict):
    """Yield manifest records for sample-1 pre images only."""
    for ds in manifest["screening_sets"]:
        for position in ds["positions"]:
            stem = f"{ds['sample']}-{position}"
            yield {
                **{k: ds[k] for k in ("id", "date", "modality", "sample", "root")},
                "position": position,
                "stem": stem,
                "path": str(Path(ds["root"]) / f"{stem}-0.tif"),
            }


def detect_pillars(path: str) -> pd.DataFrame:
    """Reuse current analyzer: top-hat + local maxima + 3x3 sub-pixel centroid."""
    return analyzer.analyze_image(path, method="peak", min_dist=3, threshold=0.2)


def nearest_neighbour_metrics(points: np.ndarray):
    """Return a conservative one-neighbour pitch measurement per detected pillar."""
    distances, indices = KDTree(points).query(points, k=2)
    nearest = distances[:, 1]
    neighbour_index = indices[:, 1]
    # Broad enough to preserve an optical trend, narrow enough to reject double-peaks
    # and missed-neighbour spacings.  This filter is only for the pitch diagnostic.
    valid = (nearest >= 5.5) & (nearest <= 9.0)
    return nearest, neighbour_index, valid


def screen_image(record: dict) -> dict:
    image = load_image_unicode(record["path"])
    if image is None:
        raise FileNotFoundError(record["path"])
    points = detect_pillars(record["path"])[["x", "y"]].to_numpy(float)
    nn, _, valid = nearest_neighbour_metrics(points)
    result = dict(record)
    result.update({
        "height_px": int(image.shape[0]), "width_px": int(image.shape[1]),
        "intensity_mean": float(np.mean(image)), "intensity_std": float(np.std(image)),
        "p99_minus_p01": float(np.percentile(image, 99)-np.percentile(image, 1)),
        "detections": int(len(points)), "valid_nn_count": int(valid.sum()),
        "valid_nn_fraction": float(valid.mean()) if len(valid) else np.nan,
        "valid_nn_mean_px": float(nn[valid].mean()) if valid.any() else np.nan,
        "valid_nn_std_px": float(nn[valid].std(ddof=1)) if valid.sum() > 1 else np.nan,
    })
    # A transparent screening score: dense, pitch-consistent detections are preferred.
    result["screen_score"] = float(valid.sum() * result["valid_nn_fraction"])
    return result


def local_fft_pitch(image: np.ndarray, tiles=5) -> pd.DataFrame:
    """Estimate first-shell reciprocal radius and pitch independently in each tile."""
    h, w = image.shape
    rows = []
    expected = 2 / (np.sqrt(3) * PITCH)
    # Existing global FFT logic supplies the direct-lattice direction.  The six
    # reciprocal directions are offset by 30 degrees; restricting the local
    # search to them rejects non-lattice streaks/rings in an individual tile.
    direct_angle = li.estimate_hex_orientation_fft(image, PITCH)
    reciprocal_angles = direct_angle - np.pi/6 + np.arange(6)*np.pi/3
    for row in range(tiles):
        for col in range(tiles):
            y0, y1 = round(row*h/tiles), round((row+1)*h/tiles)
            x0, x1 = round(col*w/tiles), round((col+1)*w/tiles)
            tile = image[y0:y1, x0:x1].astype(float)
            tile -= tile.mean()
            tile *= np.outer(np.hanning(tile.shape[0]), np.hanning(tile.shape[1]))
            # Fourfold zero padding refines reciprocal-peak positions.  Without it,
            # a 409-px tile quantises pitch too coarsely for a sub-percent test.
            padded_shape = (tile.shape[0]*4, tile.shape[1]*4)
            power = np.abs(np.fft.fftshift(np.fft.fft2(tile, s=padded_shape)))
            yy, xx = np.indices(padded_shape)
            fy = (yy-padded_shape[0]//2)/padded_shape[0]
            fx = (xx-padded_shape[1]//2)/padded_shape[1]
            radius = np.hypot(fx, fy)
            angle = np.arctan2(fy, fx)
            shell = []
            for target_angle in reciprocal_angles:
                angular_error = np.abs(np.angle(np.exp(1j*(angle-target_angle))))
                wedge = ((radius > expected*.80) & (radius < expected*1.20) &
                         (angular_error < np.deg2rad(8)))
                if not wedge.any():
                    continue
                # Peak position after fourfold padding gives <0.3% pitch
                # resolution.  Median over the six symmetry-related vectors
                # protects against one weak/contaminated direction.
                index = np.argmax(np.where(wedge, power, -np.inf))
                shell.append(float(radius.ravel()[index]))
            frequency = float(np.median(shell)) if len(shell) >= 4 else np.nan
            pitch = float(2/(np.sqrt(3)*frequency)) if np.isfinite(frequency) and frequency else np.nan
            xy = np.array([(x0+x1-1)/2, (y0+y1-1)/2])
            rows.append({"tile_row": row, "tile_col": col, "x0": x0, "x1": x1,
                         "y0": y0, "y1": y1, "x_center": xy[0], "y_center": xy[1],
                         "radius_px": float(np.linalg.norm(xy-CENTRE)),
                         "peak_count": int(len(shell)), "reciprocal_frequency": frequency,
                         "fft_pitch_px": pitch})
    return pd.DataFrame(rows)


def brown_fit(points: np.ndarray, image: np.ndarray) -> dict:
    """Fit Brown radial scale to detected points assigned to a fixed-a=7.286 lattice."""
    lattice = li.lattice_from_fft(image, PITCH)
    radius = np.linalg.norm(points-CENTRE, axis=1)
    central = points[radius < 350]
    if len(central) < 100:
        raise ValueError("Too few central detections to establish grid phase.")
    lattice = li.refine_origin_from_points(lattice, central, support_distance=2.5)
    all_indices = None
    support = np.ones(len(points), dtype=bool)
    params = np.zeros(5)  # tx, ty, theta, k1(px^-2), k2(px^-4)
    for _ in range(3):
        assigned = li.assign_lattice_indices(points, lattice, max_distance=3.0)
        if all_indices is None:
            all_indices = assigned["indices"]
        ideal = lattice.origin + all_indices @ lattice.basis.T
        def predicted(p):
            tx, ty, theta, k1, k2 = p
            vector = ideal-CENTRE
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            vector = vector @ rot.T
            r = np.linalg.norm(vector, axis=1)
            scale = 1 + k1*r*r + k2*r**4
            return CENTRE + vector*scale[:, None] + np.array([tx, ty])
        def residual(p):
            return (predicted(p)[support]-points[support]).ravel()
        baseline = np.linalg.norm(predicted(params)-points, axis=1)
        support = baseline < 2.0
        if support.sum() < 100:
            raise ValueError("Too few lattice-consistent detections for Brown fit.")
        # Bounds constrain the optimizer to weak, physically plausible image distortion.
        fit = least_squares(residual, params, bounds=([-5, -5, -.05, -2e-5, -2e-11],
                                                       [5, 5, .05, 2e-5, 2e-11]),
                            loss="soft_l1", f_scale=.25, max_nfev=300)
        params = fit.x
    pred0 = predicted(np.zeros(5))
    pred = predicted(params)
    before = np.linalg.norm(pred0[support]-points[support], axis=1)
    after = np.linalg.norm(pred[support]-points[support], axis=1)
    edge_scale = 1 + params[3]*1000**2 + params[4]*1000**4
    return {"fit_success": True, "fit_points": int(support.sum()), "tx_px": params[0],
            "ty_px": params[1], "theta_rad": params[2], "k1_px_minus2": params[3],
            "k2_px_minus4": params[4], "rmse_before_px": float(np.sqrt(np.mean(before**2))),
            "rmse_after_px": float(np.sqrt(np.mean(after**2))),
            "predicted_scale_at_1000px": float(edge_scale),
            "predicted_pitch_change_at_1000px_pct": float((edge_scale-1)*100)}


def analyse_image(record: dict, output: Path):
    image = load_image_unicode(record["path"])
    if image is None:
        raise FileNotFoundError(record["path"])
    detections = detect_pillars(record["path"])
    points = detections[["x", "y"]].to_numpy(float)
    nn, neighbour_index, valid = nearest_neighbour_metrics(points)
    radial = np.linalg.norm(points-CENTRE, axis=1)
    table = detections.copy()
    table["nearest_neighbour_px"] = nn
    table["nearest_neighbour_id"] = neighbour_index
    table["pitch_valid"] = valid
    table["radius_px"] = radial
    table["radial_bin"] = pd.cut(radial, RADIAL_EDGES, right=False, include_lowest=True,
                                  labels=["0-200", "200-400", "400-600", "600-800", "800-1000", "1000+"])
    accepted = table[table["pitch_valid"]].copy()
    bin_summary = (accepted.groupby("radial_bin", observed=False)
                    .agg(pillar_count=("nearest_neighbour_px", "size"),
                         radius_mean_px=("radius_px", "mean"),
                         pitch_mean_px=("nearest_neighbour_px", "mean"),
                         pitch_std_px=("nearest_neighbour_px", "std"))
                    .reset_index())
    bin_summary["pitch_sem_px"] = bin_summary["pitch_std_px"] / np.sqrt(bin_summary["pillar_count"])
    slope = linregress(accepted["radius_px"], accepted["nearest_neighbour_px"])
    fft = local_fft_pitch(image)
    centre_fft = float(fft.loc[(fft.tile_row == 2) & (fft.tile_col == 2), "fft_pitch_px"].iloc[0])
    corners = fft[(fft.tile_row.isin([0, 4])) & (fft.tile_col.isin([0, 4]))]
    corner_fft = float(corners["fft_pitch_px"].mean())
    nn_change = float(bin_summary.pitch_mean_px.iloc[-1]/bin_summary.pitch_mean_px.iloc[0]-1)
    fft_change = float(corner_fft/centre_fft-1)
    # A detector-specific NN bias is not evidence of optical distortion.  Require
    # both independent measurements to exceed their magnitude gates *and* agree
    # on barrel (+) versus pincushion (-) direction before fitting Brown.
    trend = (abs(nn_change) >= .001 and abs(fft_change) >= .0005 and
             np.sign(nn_change) == np.sign(fft_change))
    brown = brown_fit(points[valid], image) if trend else {"fit_success": False, "reason": "No joint radial-NN and corner-vs-centre FFT effect above predeclared magnitude gates."}
    ident = f"{record['id']}_S{record['sample']}_P{record['position']}"
    image_dir = output / ident
    image_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(image_dir / "detected_pillars_and_nearest_neighbours.csv", index=False)
    bin_summary.to_csv(image_dir / "radial_pitch_bins.csv", index=False)
    fft.to_csv(image_dir / "local_fft_tiles.csv", index=False)
    with (image_dir / "brown_fit.json").open("w", encoding="utf-8") as f:
        json.dump(brown, f, ensure_ascii=False, indent=2)
    fig, ax = plt.subplots(figsize=(8, 5))
    draw = accepted.sample(min(40000, len(accepted)), random_state=0)
    ax.scatter(draw.radius_px, draw.nearest_neighbour_px, s=1, alpha=.08, label="valid NN pitch")
    ax.errorbar(bin_summary.radius_mean_px, bin_summary.pitch_mean_px, yerr=bin_summary.pitch_sem_px,
                marker="o", color="crimson", label="radial-bin mean ± SEM")
    ax.axhline(PITCH, color="black", ls="--", lw=1, label="established global a = 7.286 px")
    ax.set(xlabel="radius from geometric image centre (px)", ylabel="nearest-neighbour pitch (px)",
           title=ident)
    ax.legend(markerscale=4); fig.tight_layout(); fig.savefig(image_dir / "radial_pitch.png", dpi=180); plt.close(fig)
    return {**record, "identifier": ident, "detections": int(len(points)), "valid_pitch_points": int(valid.sum()),
            "valid_pitch_fraction": float(valid.mean()), "pitch_slope_px_per_px": float(slope.slope),
            "pitch_slope_pvalue": float(slope.pvalue), "pitch_center_bin_px": float(bin_summary.pitch_mean_px.iloc[0]),
            "pitch_outer_bin_px": float(bin_summary.pitch_mean_px.iloc[-1]),
            "pitch_outer_minus_center_pct": float(nn_change*100),
            "fft_center_px": centre_fft, "fft_corner_mean_px": corner_fft,
            "fft_corner_minus_center_pct": float(fft_change*100), "joint_trend_gate": bool(trend), **brown}


def summarise_existing(record: dict, output: Path):
    """Rebuild the cross-FOV table from saved per-image audit artefacts only."""
    ident = f"{record['id']}_S{record['sample']}_P{record['position']}"
    image_dir = output / ident
    table = pd.read_csv(image_dir / "detected_pillars_and_nearest_neighbours.csv")
    bins = pd.read_csv(image_dir / "radial_pitch_bins.csv")
    fft = pd.read_csv(image_dir / "local_fft_tiles.csv")
    brown = json.loads((image_dir / "brown_fit.json").read_text(encoding="utf-8"))
    accepted = table[table["pitch_valid"].astype(str).str.lower() == "true"]
    slope = linregress(accepted["radius_px"], accepted["nearest_neighbour_px"])
    nn_change = float(bins["pitch_mean_px"].iloc[-1]/bins["pitch_mean_px"].iloc[0]-1)
    centre_fft = float(fft.loc[(fft.tile_row == 2) & (fft.tile_col == 2), "fft_pitch_px"].iloc[0])
    corner_fft = float(fft[(fft.tile_row.isin([0, 4])) & (fft.tile_col.isin([0, 4]))]["fft_pitch_px"].mean())
    fft_change = float(corner_fft/centre_fft-1)
    trend = (abs(nn_change) >= .001 and abs(fft_change) >= .0005 and
             np.sign(nn_change) == np.sign(fft_change))
    return {**record, "identifier": ident, "detections": int(len(table)),
            "valid_pitch_points": int(len(accepted)), "valid_pitch_fraction": float(len(accepted)/len(table)),
            "pitch_slope_px_per_px": float(slope.slope), "pitch_slope_pvalue": float(slope.pvalue),
            "pitch_center_bin_px": float(bins["pitch_mean_px"].iloc[0]),
            "pitch_outer_bin_px": float(bins["pitch_mean_px"].iloc[-1]),
            "pitch_outer_minus_center_pct": float(nn_change*100), "fft_center_px": centre_fft,
            "fft_corner_mean_px": corner_fft, "fft_corner_minus_center_pct": float(fft_change*100),
            "joint_trend_gate": bool(trend), **brown}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/raw/p50_radial_distortion_manifest.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--screen-only", action="store_true")
    ap.add_argument("--summarize-existing", action="store_true",
                    help="Rebuild cross_fov_summary.csv from saved per-image artefacts without loading TIFFs.")
    ap.add_argument("--identifiers", nargs="*", help="Selected IDs, e.g. 260824_SAM_S1_P2")
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = list(image_records(manifest))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.screen_only:
        rows = []
        for record in records:
            print("screening", record["path"], flush=True)
            rows.append(screen_image(record))
        pd.DataFrame(rows).sort_values("screen_score", ascending=False).to_csv(args.output / "screening_all_high_sample1_pre.csv", index=False)
        return
    wanted = set(args.identifiers or [])
    if wanted:
        records = [r for r in records if f"{r['id']}_S{r['sample']}_P{r['position']}" in wanted]
    if not records:
        raise ValueError("No selected images. Provide --identifiers.")
    if args.summarize_existing:
        pd.DataFrame([summarise_existing(record, args.output) for record in records]).to_csv(
            args.output / "cross_fov_summary.csv", index=False)
        return
    rows = []
    for record in records:
        print("analysing", record["path"], flush=True)
        rows.append(analyse_image(record, args.output))
    pd.DataFrame(rows).to_csv(args.output / "cross_fov_summary.csv", index=False)


if __name__ == "__main__":
    main()
