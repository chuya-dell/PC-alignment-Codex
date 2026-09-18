"""Phase 2 (QC-gated), self-contained production concentration-series reanalysis.

Runs the full pre-image-FFT-grid -> Phase 2 registration -> post-image contrast
sampling -> blank-threshold -> Mann-Whitney pipeline for the production pre/post
pairs spanning the six 2026-08-24..29 canonical schedules (409 attempted pairs
across all eight dataset x modality combinations in the source manifest; see
"Pair manifest input" below for why 409, not the "382" cited in the originating
request, is the right population), using
``shared.registration.register_image_pair_affine`` (Phase 2, QC-gated) instead
of ``align_and_match_dataframes`` (Phase 1, which run_canonical_theoretical_grid.py
uses in PC-alignment-anti and which has an independently documented landmark
misidentification failure mode; see docs/PHASE1_PRODUCTION_USAGE_VERIFICATION_20260916.md
and docs/PHASE2_TRANSFORM_QC_GATE_20260918.md).

Ported logic and its source (PC-alignment-anti, read-only reference; not modified):
  - Grid generation and direct contrast sampling flow: ``run_canonical_theoretical_grid.py``
    (module-level ``main()``) -- pre-image FFT lattice, ``sample_grid_features(..., margin=30)``,
    projecting pre-image grid points into the post image via the estimated transform, then
    sampling post contrast directly (no detection in the post image). Rewritten here to use
    the Phase 2 affine warp (``cv2.transform`` forward application; the warp maps pre-image
    points to post-image points -- verified empirically against groove landmarks, see
    docs/PHASE2_PRODUCTION_REANALYSIS_20260918.md) in place of
    ``registration.inverse_transform_points`` on a Phase 1 quadratic transform.
  - Blank threshold and FOV-level exact Mann-Whitney:
    ``investigation/260907_pipeline_rebuild/scripts/run_fov_level_direct_sampling.py``
    (``summarize()`` and the ``fov_mw`` call). Reimplemented in
    ``shared/concentration_series_stats.py``.

Any ``AffineTransformQCError`` from ``register_image_pair_affine`` marks that FOV
``registration_qc_rejected`` and excludes it from every downstream statistic.

Pair manifest input: use the *full* 409-row pre/post path list from the
2026-09-17 Phase 1/Phase 2 comparison run (``pair_transform_comparison_20260917.csv``,
a data artifact stored in Google Drive/Obsidian, not PC-alignment-anti source
code), not the 382-row ``..._with_flags_20260917.csv`` filtered file. The
"with_flags" file only kept rows where the *old* comparison's Phase 1 attempt
also succeeded (its ``status`` column requires both phases ok), so it silently
drops pairs Phase 2 alone can register -- exactly the pairs this rerun exists to
recover. Confirmed empirically: four pairs missing from the 382-row file
(e.g. 260827_p50_dna sample 12 positions 3/5) are present and Phase-2-registrable
in the 409-row file, and only with the 409-row file as input does this script's
Mann-Whitney summary reproduce the 2026-09-17 throwaway Phase-2-substitution
recomputation (``phase2_recompute_mannwhitney_summary_20260917.csv``) exactly
for all six canonical schedules -- see
docs/PHASE2_PRODUCTION_REANALYSIS_20260918.md, item 4 of the request this
script implements.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import registration as reg
from shared.lattice_indexing import lattice_from_fft
from shared.theoretical_grid_evaluation import sample_grid_features
from shared.concentration_series_stats import blank_threshold, fov_exceeds_rate_mann_whitney

PITCH_DEFAULT = 7.286
GRID_MARGIN = 30


def fov_key(row) -> str:
    return f"{row.dataset}_{row.sample}_{row.position}"


def process_pair(row, pitch: float) -> tuple[dict, np.ndarray | None]:
    """Register one pre/post pair with Phase 2 and sample the pre-image FFT grid.

    Returns (summary_dict, valid_delta_array_or_None). ``summary_dict['status']``
    is one of "ok", "registration_qc_rejected", or "error".
    """
    base = {
        "dataset": row.dataset, "sample": row.sample, "position": row.position,
        "concentration": row.concentration, "series": row.series, "is_blank": row.is_blank,
    }
    try:
        pre = reg.load_image_unicode(row.pre_path)
        post = reg.load_image_unicode(row.post_path)
        if pre is None or post is None or pre.shape != post.shape:
            return {**base, "status": "error", "error": "image load failed or shape mismatch"}, None
        try:
            warp, qc = reg.register_image_pair_affine(pre, post, return_qc=True)
        except reg.AffineTransformQCError as exc:
            qc = exc.diagnostics
            return {
                **base, "status": "registration_qc_rejected", "error": str(exc),
                "dx_center_px": qc["dx_center_px"], "dy_center_px": qc["dy_center_px"],
                "rotation_deg": qc["rotation_deg"], "scale": qc["scale"],
                "anisotropy": qc["anisotropy"], "mask_fraction": qc.get("mask_fraction"),
                "qc_reasons": "; ".join(qc["reasons"]),
            }, None
        lattice = lattice_from_fft(pre, pitch)
        grid = sample_grid_features(pre, lattice, margin=GRID_MARGIN)
        pre_xy = grid[["x", "y"]].to_numpy(dtype=np.float32)[None, :, :]
        post_xy = cv2.transform(pre_xy, warp)[0]
        post_sample = reg.sample_contrast(post, post_xy)
        valid = (grid["valid_sampling"].to_numpy() & post_sample["valid_sampling"].to_numpy()
                 & np.isfinite(grid["contrast"].to_numpy()) & np.isfinite(post_sample["contrast"].to_numpy()))
        delta = (post_sample["contrast"].to_numpy() - grid["contrast"].to_numpy())[valid]
        summary = {
            **base, "status": "ok", "error": "",
            "n_grid_points": int(len(grid)), "n_valid": int(valid.sum()),
            "delta_mean": float(delta.mean()) if delta.size else float("nan"),
            "dx_center_px": qc["dx_center_px"], "dy_center_px": qc["dy_center_px"],
            "rotation_deg": qc["rotation_deg"], "scale": qc["scale"],
            "anisotropy": qc["anisotropy"], "mask_fraction": qc.get("mask_fraction"),
        }
        return summary, delta
    except Exception as exc:  # Retain failures explicitly; never silently drop a FOV.
        return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}, None


def process_pair_cached(row, cache_dir: Path, pitch: float) -> tuple[dict, np.ndarray | None]:
    """Persist each completed FOV so an interrupted batch resumes safely.

    Caching pattern ported from PC-alignment-anti's
    investigation/260907_pipeline_rebuild/scripts/run_fov_level_direct_sampling.py
    (``sample_one_cached``).
    """
    result_path = cache_dir / f"{fov_key(row)}.npz"
    if result_path.exists():
        with np.load(result_path, allow_pickle=False) as saved:
            summary = json.loads(str(saved["summary_json"].item()))
            delta = saved["delta"] if "delta" in saved else None
            return summary, delta
    summary, delta = process_pair(row, pitch)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary_json": json.dumps(summary)}
    if delta is not None:
        payload["delta"] = delta
    np.savez_compressed(result_path, **payload)
    return summary, delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-manifest", type=Path, required=True,
                        help="pair_transform_comparison_20260917.csv (the full 409-row file, "
                             "not the 382-row _with_flags_ filtered one -- see module docstring) "
                             "or equivalent (needs dataset,sample,position,concentration,series,"
                             "is_blank,pre_path,post_path columns)")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pitch", type=float, default=PITCH_DEFAULT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.pair_manifest)
    needed = {"dataset", "sample", "position", "concentration", "series", "is_blank",
              "pre_path", "post_path"}
    missing = needed - set(manifest.columns)
    if missing:
        raise ValueError(f"pair manifest missing columns: {sorted(missing)}")

    summaries, deltas = [], {}
    for row in manifest.itertuples(index=False):
        try:
            summary, delta = process_pair_cached(row, args.cache_dir, args.pitch)
        except Exception:
            summary = {"dataset": row.dataset, "sample": row.sample, "position": row.position,
                       "concentration": row.concentration, "series": row.series,
                       "is_blank": row.is_blank, "status": "error", "error": traceback.format_exc()}
            delta = None
        summaries.append(summary)
        if delta is not None:
            deltas[fov_key(row)] = delta
        print(f"{summary['dataset']} sample{summary['sample']} pos{summary['position']}: "
              f"{summary['status']}", flush=True)

    fov_df = pd.DataFrame(summaries)
    fov_df.to_csv(args.out_dir / "phase2_reanalysis_fov_summary.csv", index=False)

    threshold_rows, mw_rows, exceeds_rows = [], [], []
    for dataset, group in fov_df.groupby("dataset"):
        ok = group[group.status == "ok"].copy()
        blank_rows = ok[ok.is_blank]
        blank_pooled = np.concatenate([deltas[fov_key(r)] for r in blank_rows.itertuples(index=False)
                                       if fov_key(r) in deltas]) if len(blank_rows) else np.array([])
        if blank_pooled.size == 0:
            threshold_rows.append({"dataset": dataset, "blank_threshold": np.nan,
                                   "n_blank_fov": len(blank_rows), "n_blank_grid_points": 0,
                                   "note": "no accepted blank FOV"})
            continue
        threshold = blank_threshold(blank_pooled)
        threshold_rows.append({"dataset": dataset, "blank_threshold": threshold,
                               "n_blank_fov": len(blank_rows), "n_blank_grid_points": int(blank_pooled.size)})

        exceeds_rate_by_key = {}
        for r in ok.itertuples(index=False):
            delta = deltas.get(fov_key(r))
            exceeds_rate = float(np.mean(delta < threshold)) if delta is not None and delta.size else np.nan
            exceeds_rate_by_key[fov_key(r)] = exceeds_rate
            exceeds_rows.append({"dataset": r.dataset, "sample": r.sample, "position": r.position,
                                 "concentration": r.concentration, "series": r.series,
                                 "is_blank": r.is_blank, "delta_mean": r.delta_mean,
                                 "exceeds_rate": exceeds_rate, "blank_threshold": threshold})

        high = ok[(ok.concentration == "1nM") & (~ok.is_blank) & (ok.series == "primary")]
        if len(high) == 0 or len(blank_rows) == 0:
            mw_rows.append({"dataset": dataset, "n_high": len(high), "n_blank": len(blank_rows),
                            "blank_threshold": threshold, "note": "insufficient high or blank FOVs"})
            continue
        high_exceeds = np.array([exceeds_rate_by_key[fov_key(r)] for r in high.itertuples(index=False)])
        blank_exceeds = np.array([exceeds_rate_by_key[fov_key(r)] for r in blank_rows.itertuples(index=False)])
        mw = fov_exceeds_rate_mann_whitney(high_exceeds, blank_exceeds)
        mw_rows.append({"dataset": dataset, "blank_threshold": threshold, **mw})

    pd.DataFrame(threshold_rows).to_csv(args.out_dir / "phase2_reanalysis_blank_thresholds.csv", index=False)
    pd.DataFrame(exceeds_rows).to_csv(args.out_dir / "phase2_reanalysis_fov_exceeds_rate.csv", index=False)
    pd.DataFrame(mw_rows).to_csv(args.out_dir / "phase2_reanalysis_mannwhitney_summary.csv", index=False)

    status_counts = fov_df.status.value_counts().to_dict()
    (args.out_dir / "phase2_reanalysis_run_summary.json").write_text(
        json.dumps({"total_pairs": len(manifest), "status_counts": status_counts,
                   "datasets": sorted(fov_df.dataset.unique().tolist())}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status_counts, indent=2))


if __name__ == "__main__":
    main()
