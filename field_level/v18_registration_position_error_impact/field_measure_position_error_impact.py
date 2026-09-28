"""Measure how known sampling-position offsets change the production statistic.

Diagnostic-only. Keeps the current registration, FFT grid, contrast sampler,
blank threshold rule, and exact per-FOV Mann-Whitney test unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import registration as reg
from shared.image_qc import bright_band_mask
from shared.lattice_indexing import lattice_from_fft
from shared.theoretical_grid_evaluation import sample_grid_features
from shared.concentration_series_stats import blank_threshold

MAGNITUDES = (0.05, 0.1, 0.2, 0.5, 1.0)


def fov_key(row):
    return f"{row.dataset}_{row.sample}_{row.position}"


def random_unit_vector(key):
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    angle = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    return np.array([np.cos(angle), np.sin(angle)])


def process_pair(row, diagnostics):
    key = fov_key(row)
    pre = reg.load_image_unicode(row.pre_path)
    post = reg.load_image_unicode(row.post_path)
    if pre is None or post is None or pre.shape != post.shape:
        raise RuntimeError(f"Image load/shape failure: {key}")
    pre_mask = bright_band_mask(pre)
    post_mask = bright_band_mask(post)
    lattice = lattice_from_fft(pre, 7.286)
    grid = sample_grid_features(pre, lattice, margin=30, invalid_mask=pre_mask)
    pre_xy = grid[["x", "y"]].to_numpy(dtype=np.float32)
    diag = json.loads((diagnostics / f"{key}.json").read_text(encoding="utf-8"))
    matrix = np.asarray(diag["matrix"], dtype=np.float32)
    base_xy = cv2.transform(pre_xy[None, :, :], matrix)[0]
    pre_delta_source = grid["contrast"].to_numpy(dtype=float)
    pre_valid = grid["valid_sampling"].to_numpy(dtype=bool) & np.isfinite(pre_delta_source)
    unit_random = random_unit_vector(key)
    records = []

    def sample(offset):
        sample_post = reg.sample_contrast(post, base_xy + offset.astype(np.float32),
                                          invalid_mask=post_mask)
        post_values = sample_post["contrast"].to_numpy(dtype=float)
        valid = pre_valid & sample_post["valid_sampling"].to_numpy(dtype=bool) & np.isfinite(post_values)
        return (post_values - pre_delta_source)[valid], valid, post_values

    baseline_delta, baseline_valid, baseline_post_values = sample(np.zeros(2))
    records.append({"dataset": row.dataset, "sample": row.sample, "position": row.position,
                    "concentration": row.concentration, "series": row.series,
                    "is_blank": bool(row.is_blank), "fov_key": key, "direction": "baseline",
                    "magnitude_px": 0.0, "n_valid": int(len(baseline_delta)),
                    "delta": baseline_delta,
                    "contrast_abs_error_median": 0.0, "contrast_abs_error_p95": 0.0})
    for magnitude in MAGNITUDES:
        for direction, unit in (("systematic_x", np.array([1.0, 0.0])),
                                ("random_per_fov", unit_random)):
            delta, valid, shifted_post_values = sample(unit * magnitude)
            common = valid & baseline_valid
            if common.any():
                abs_error = np.abs(shifted_post_values[common] - baseline_post_values[common])
                e50, e95 = float(np.median(abs_error)), float(np.quantile(abs_error, .95))
            else:
                e50 = e95 = float("nan")
            records.append({"dataset": row.dataset, "sample": row.sample, "position": row.position,
                            "concentration": row.concentration, "series": row.series,
                            "is_blank": bool(row.is_blank), "fov_key": key, "direction": direction,
                            "magnitude_px": magnitude, "n_valid": int(len(delta)), "delta": delta,
                            "contrast_abs_error_median": e50, "contrast_abs_error_p95": e95})
    return records


def summarize(records):
    fov_rows = []
    # Calculate the production blank-derived threshold independently for each
    # dataset and perturbation. Baseline uses its own unperturbed blank pool.
    all_groups = {}
    for r in records:
        all_groups.setdefault((r["dataset"], r["direction"], r["magnitude_px"]), []).append(r)
    group_metrics = {}
    for group_key, rows in all_groups.items():
        dataset, direction, magnitude = group_key
        blanks = [r["delta"] for r in rows if r["is_blank"] and len(r["delta"])]
        if not blanks:
            continue
        threshold = blank_threshold(np.concatenate(blanks))
        for r in rows:
            delta = r["delta"]
            rate = float(np.mean(delta < threshold)) if len(delta) else float("nan")
            fov_rows.append({k: r[k] for k in ("dataset", "sample", "position", "concentration", "series", "is_blank", "fov_key", "direction", "magnitude_px", "n_valid", "contrast_abs_error_median", "contrast_abs_error_p95")} | {
                "blank_threshold": threshold, "delta_mean": float(np.mean(delta)) if len(delta) else float("nan"),
                "exceeds_rate": rate})
        metrics = {r["fov_key"]: float(np.mean(r["delta"] < threshold)) if len(r["delta"]) else float("nan") for r in rows}
        ok = pd.DataFrame([{**r, "rate": metrics[r["fov_key"]]} for r in rows])
        high = ok[(ok.concentration == "1nM") & (~ok.is_blank) & (ok.series == "primary") & np.isfinite(ok.rate)]
        blank = ok[ok.is_blank & np.isfinite(ok.rate)]
        if len(high) and len(blank):
            test = mannwhitneyu(high.rate.to_numpy(), blank.rate.to_numpy(), alternative="two-sided", method="exact")
            group_metrics[group_key] = {"n_high": len(high), "n_blank": len(blank), "U": float(test.statistic),
                                        "p": float(test.pvalue), "threshold": threshold}
    fov_df = pd.DataFrame(fov_rows)
    # Contrasts with baseline and perturbed per-FOV indicators, by exact test.
    ds_rows = []
    for (dataset, direction, magnitude), metric in group_metrics.items():
        base = group_metrics.get((dataset, "baseline", 0.0))
        ds_rows.append({"dataset": dataset, "direction": direction, "magnitude_px": magnitude,
                        "n_high": metric["n_high"], "n_blank": metric["n_blank"],
                        "blank_threshold": metric["threshold"], "U_exact": metric["U"],
                        "p_two_sided_exact": metric["p"],
                        "baseline_p_two_sided_exact": base["p"] if base else np.nan,
                        "conclusion_changed_at_0_05": bool((metric["p"] <= .05) != (base["p"] <= .05)) if base else False})
    ds_df = pd.DataFrame(ds_rows)
    summary_rows = []
    for (direction, magnitude), part in fov_df.groupby(["direction", "magnitude_px"]):
        changed = ds_df[(ds_df.direction == direction) & (ds_df.magnitude_px == magnitude)]
        baseline_means = fov_df[(fov_df.direction == "baseline")].set_index("fov_key").delta_mean
        changed_mean = part.apply(lambda r: r.delta_mean - baseline_means.get(r.fov_key, np.nan), axis=1)
        summary_rows.append({"direction": direction, "magnitude_px": magnitude,
            "n_fovs": int(len(part)), "median_pillar_contrast_abs_error": float(part.contrast_abs_error_median.median()),
            "p95_fov_median_pillar_contrast_abs_error": float(part.contrast_abs_error_median.quantile(.95)),
            "median_pillar_contrast_abs_error_p95": float(part.contrast_abs_error_p95.median()),
            "median_abs_fov_delta_mean_change": float(changed_mean.abs().median()) if len(part) else np.nan,
            "p95_abs_fov_delta_mean_change": float(changed_mean.abs().quantile(.95)) if len(part) else np.nan,
            "datasets_p_le_0_05": int((changed.p_two_sided_exact <= .05).sum()),
            "datasets_conclusion_changed": int(changed.conclusion_changed_at_0_05.sum())})
    return fov_df, ds_df, pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--registration-output", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    baseline = pd.read_csv(args.baseline_summary)
    baseline = baseline[(baseline.n_high > 0) & (baseline.n_blank > 0)]
    datasets = set(baseline.dataset)
    # The manifest status column records a separate older comparison attempt;
    # v16 Phase-2 acceptance below is the sole authority for including pairs.
    manifest = manifest[manifest.dataset.isin(datasets)]
    accepted = pd.read_csv(args.registration_output / "summary_0" / "phase2_reanalysis_fov_summary.csv")
    accepted_keys = set(accepted.loc[accepted.status == "ok"].apply(
        lambda r: f"{r['dataset']}_{r['sample']}_{r['position']}", axis=1))
    manifest = manifest[manifest.apply(
        lambda r: f"{r['dataset']}_{r['sample']}_{r['position']}" in accepted_keys, axis=1)]
    print(f"Datasets ({len(datasets)}): {', '.join(sorted(datasets))}; accepted pairs={len(manifest)}", flush=True)
    records = []
    tasks = list(manifest.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_pair, row, args.registration_output / "diagnostics"): row for row in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                records.extend(future.result())
            except Exception as exc:
                print(f"FAILED {fov_key(row)}: {type(exc).__name__}: {exc}", flush=True)
                raise
            if i % 10 == 0 or i == len(futures):
                print(f"sampled {i}/{len(futures)} pairs", flush=True)
    fov_df, ds_df, summary_df = summarize(records)
    fov_df.to_csv(args.out_dir / "position_error_fov_metrics.csv", index=False)
    ds_df.to_csv(args.out_dir / "position_error_dataset_tests.csv", index=False)
    summary_df.to_csv(args.out_dir / "position_error_summary_by_magnitude.csv", index=False)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
