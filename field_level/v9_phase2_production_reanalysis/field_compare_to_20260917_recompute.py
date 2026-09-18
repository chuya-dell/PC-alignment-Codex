"""Compare this entry point's Mann-Whitney summary against the 2026-09-17 throwaway
Phase-2-substitution recomputation (phase2_recompute_mannwhitney_summary_20260917.csv,
a Google Drive/Obsidian data artifact, not code). Item 4 of the
2026-09-18 request this v9 entry point implements.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-summary", type=Path, required=True,
                        help="phase2_reanalysis_mannwhitney_summary.csv from field_run_phase2_production_reanalysis.py")
    parser.add_argument("--reference-summary", type=Path, required=True,
                        help="phase2_recompute_mannwhitney_summary_20260917.csv")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    new = pd.read_csv(args.new_summary)
    ref = pd.read_csv(args.reference_summary).rename(columns={
        "high_concentration": "ref_high_concentration", "n_high": "ref_n_high",
        "n_blank": "ref_n_blank", "p_two_sided_exact": "ref_p_two_sided_exact",
        "p_one_sided_greater_exact": "ref_p_one_sided_greater_exact",
        "blank_threshold": "ref_blank_threshold",
    })
    merged = ref.merge(new, on="dataset", how="left")
    merged["p_two_sided_diff"] = merged["p_two_sided_exact"] - merged["ref_p_two_sided_exact"]
    merged["n_high_matches"] = merged["n_high"] == merged["ref_n_high"]
    merged["n_blank_matches"] = merged["n_blank"] == merged["ref_n_blank"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    pd.set_option("display.width", 200)
    print(merged[["dataset", "ref_n_high", "n_high", "ref_n_blank", "n_blank",
                  "ref_p_two_sided_exact", "p_two_sided_exact", "p_two_sided_diff"]].to_string(index=False))


if __name__ == "__main__":
    main()
