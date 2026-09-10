from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
CACHE = ROOT / "cache" / "260827_DNA_Sample12_vs_Blank11" / "sampled"

qc = pd.read_csv(RESULTS / "crossday_fov_qc.csv")
ok = qc[qc.status.eq("ok")].copy()

deltas = {}
for row in ok.itertuples():
    stem = f"{int(row.sample)}-{int(row.position)}"
    with np.load(CACHE / f"{stem}.npz", allow_pickle=False) as z:
        deltas[stem] = z["delta"]

blank_all = np.concatenate([
    deltas[f"11-{int(row.position)}"]
    for row in ok[ok.group.eq("blank")].itertuples()
])
threshold = float(blank_all.mean() - 3 * blank_all.std())

rows = []
for row in qc.itertuples():
    stem = f"{int(row.sample)}-{int(row.position)}"
    base = {
        "dataset": row.dataset,
        "group": row.group,
        "sample": int(row.sample),
        "position": int(row.position),
        "pre_file": f"{stem}-0.tif",
        "post_file": f"{stem}-1.tif",
        "status": row.status,
        "exclusion_reason": row.error if row.status == "failed" else "",
        "threshold_blank_mean_minus_3sd": threshold,
    }
    if row.status == "ok":
        delta = deltas[stem]
        base.update({
            "valid_grid_points": len(delta),
            "delta_mean": float(delta.mean()),
            "threshold_exceedance_count": int(np.sum(delta < threshold)),
            "threshold_exceedance_rate": float(np.mean(delta < threshold)),
            "icp_iterations": int(row.icp_iterations),
            "icp_converged": bool(row.icp_converged),
        })
    rows.append(base)

# Pre-specified defect exclusion: retain it explicitly in the file inventory.
rows.append({
    "dataset": "260827_DNA_Sample12_vs_Blank11",
    "group": "blank",
    "sample": 11,
    "position": 7,
    "pre_file": "11-7-0.tif",
    "post_file": "11-7-1.tif",
    "status": "excluded_pre_analysis",
    "exclusion_reason": "Primary log: position 7 stain (要確認); pre-specified defect exclusion",
    "threshold_blank_mean_minus_3sd": threshold,
})

out = pd.DataFrame(rows).sort_values(["group", "sample", "position"])
out.to_csv(RESULTS / "260827_DNA_Sample12vs11_fov_statistics.csv", index=False)

success = out[out.status.eq("ok")]
high = success[success.group.eq("high")]
blank = success[success.group.eq("blank")]
rate_test = mannwhitneyu(
    high.threshold_exceedance_rate.to_numpy(),
    blank.threshold_exceedance_rate.to_numpy(),
    alternative="two-sided",
    method="exact",
)
mean_test = mannwhitneyu(
    high.delta_mean.to_numpy(),
    blank.delta_mean.to_numpy(),
    alternative="two-sided",
    method="exact",
)
pipeline_summary = pd.read_csv(RESULTS / "crossday_sample1_blank_summary.csv").iloc[0]

summary = pd.DataFrame([{
    "dataset": "260827_DNA_Sample12_vs_Blank11",
    "high_sample": 12,
    "blank_sample": 11,
    "high_fov_n": len(high),
    "blank_fov_n": len(blank),
    "threshold_blank_mean_minus_3sd": threshold,
    "high_points": int(pipeline_summary.high_points),
    "blank_points": int(pipeline_summary.blank_points),
    "pooled_high_exceedance_rate": float(pipeline_summary.high_exceedance_rate),
    "pooled_blank_exceedance_rate": float(pipeline_summary.blank_exceedance_rate),
    "fov_rate_mannwhitney_U": float(rate_test.statistic),
    "fov_rate_mannwhitney_p": float(rate_test.pvalue),
    "canonical_fov_delta_mean_U": float(mean_test.statistic),
    "canonical_fov_delta_mean_p": float(mean_test.pvalue),
    "point_mannwhitney_p_descriptive_only": float(pipeline_summary.point_mannwhitney_p),
}])
summary.to_csv(RESULTS / "260827_DNA_Sample12vs11_statistics_summary.csv", index=False)
print(out.to_string(index=False))
print(summary.to_string(index=False))
