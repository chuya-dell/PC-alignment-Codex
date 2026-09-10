"""Apply canonical direct-grid sampling to a manifest of clean FOVs.

Each lattice is defined from the pre image only.  Peak detections are only
geometry supports for canonical registration.py; they never select sampled
lattice locations.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import analyzer, lattice_indexing as li, registration
from theoretical_grid_evaluation import sample_grid_features


def points(image: Path, cache: Path):
    if cache.exists(): return pd.read_csv(cache)
    frame = analyzer.analyze_image(str(image), method="peak", min_dist=3, threshold=.2).copy()
    if "pillar_id" not in frame: frame.insert(0, "pillar_id", np.arange(len(frame)))
    cache.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(cache, index=False)
    return frame


def sample_pair(root: Path, sample: int, position: int, cache: Path, pitch: float):
    stem = f"{sample}-{position}"; saved = cache / "sampled" / f"{stem}.npz"
    if saved.exists():
        with np.load(saved, allow_pickle=False) as z: return z["delta"], json.loads(str(z["qc"].item()))
    pre, post = root / f"{stem}-0.tif", root / f"{stem}-1.tif"
    p = points(pre, cache / f"{stem}-pre.csv"); q = points(post, cache / f"{stem}-post.csv")
    _, transform, iterations, converged, _, _ = registration.align_and_match_dataframes(
        p, q, str(pre), str(post), return_diagnostics=True, local_refinement=False)
    pre_img, post_img = registration.load_image_unicode(str(pre)), registration.load_image_unicode(str(post))
    grid = sample_grid_features(pre_img, li.lattice_from_fft(pre_img, pitch), margin=30)
    post_xy = registration.inverse_transform_points(grid[["x", "y"]].to_numpy(), transform)
    after = registration.sample_contrast(post_img, post_xy)
    valid = grid.valid_sampling.to_numpy() & after.valid_sampling.to_numpy()
    delta = (after.contrast.to_numpy() - grid.contrast.to_numpy())[valid]
    qc = {"valid_grid_points": int(valid.sum()), "grid_points": int(len(grid)),
          "pre_peak_points": int(len(p)), "post_peak_points": int(len(q)),
          "icp_iterations": int(iterations), "icp_converged": bool(converged)}
    saved.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(saved, delta=delta, qc=json.dumps(qc))
    return delta, qc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--pitch", type=float, default=7.286)
    a = ap.parse_args(); datasets = json.loads(a.manifest.read_text(encoding="utf-8")); a.output.mkdir(parents=True, exist_ok=True)
    all_results, qc_rows = [], []
    for ds in datasets:
        groups = {}
        for group in ("high", "blank"):
            values=[]
            for pos in ds[group]["positions"]:
                try:
                    delta, qc = sample_pair(Path(ds["root"]), ds[group]["sample"], pos, a.cache / ds["id"], a.pitch)
                    values.append(delta); qc_rows.append({"dataset":ds["id"],"group":group,"sample":ds[group]["sample"],"position":pos,"status":"ok","delta_mean":float(delta.mean()),**qc})
                except Exception as exc:
                    qc_rows.append({"dataset":ds["id"],"group":group,"sample":ds[group]["sample"],"position":pos,"status":"failed","error":str(exc)})
            groups[group] = np.concatenate(values) if values else np.array([])
        if not len(groups["high"]) or not len(groups["blank"]):
            all_results.append({"dataset":ds["id"],"status":"no_complete_comparison"}); continue
        threshold = float(groups["blank"].mean() - 3*groups["blank"].std())
        fov = pd.DataFrame(qc_rows)
        fov = fov[(fov["dataset"] == ds["id"]) & (fov["status"] == "ok")]
        hmeans = fov.query("group == 'high'").delta_mean.to_numpy(); bmeans=fov.query("group == 'blank'").delta_mean.to_numpy()
        fov_test = mannwhitneyu(hmeans,bmeans,alternative="two-sided",method="exact")
        point_test = mannwhitneyu(groups["high"],groups["blank"],alternative="two-sided",method="asymptotic")
        all_results.append({"dataset":ds["id"],"status":"ok","high_sample":ds["high"]["sample"],"blank_sample":ds["blank"]["sample"],
          "high_fov_n":int(len(hmeans)),"blank_fov_n":int(len(bmeans)),"high_points":int(len(groups["high"])),"blank_points":int(len(groups["blank"])),
          "threshold_blank_mean_minus_3sd":threshold,"high_exceedance_rate":float(np.mean(groups["high"]<threshold)),"blank_exceedance_rate":float(np.mean(groups["blank"]<threshold)),
          "high_p01":float(np.quantile(groups["high"],.01)),"blank_p01":float(np.quantile(groups["blank"],.01)),
          "fov_mannwhitney_U":float(fov_test.statistic),"fov_mannwhitney_p":float(fov_test.pvalue),"point_mannwhitney_p":float(point_test.pvalue)})
        print(ds["id"], all_results[-1])
    pd.DataFrame(qc_rows).to_csv(a.output/"crossday_fov_qc.csv",index=False)
    pd.DataFrame(all_results).to_csv(a.output/"crossday_sample1_blank_summary.csv",index=False)
    (a.output/"crossday_manifest_used.json").write_text(json.dumps(datasets,indent=2),encoding="utf-8")

if __name__ == "__main__": main()
