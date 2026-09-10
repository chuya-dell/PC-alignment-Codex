"""Recompute the two FOV-level Mann--Whitney tests from the reference CSV.

This is intentionally limited to the small exported FOV table.  It does not
read experimental TIFFs, caches, or generated images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).with_name("fov_statistics_reference.csv"),
    )
    args = parser.parse_args()
    table = pd.read_csv(args.input)
    used = table.loc[table["status"].eq("ok")]
    high = used.loc[used["group"].eq("high")]
    blank = used.loc[used["group"].eq("blank")]

    rate = mannwhitneyu(
        high["threshold_exceedance_rate"],
        blank["threshold_exceedance_rate"],
        alternative="two-sided",
        method="exact",
    )
    delta_mean = mannwhitneyu(
        high["delta_mean"],
        blank["delta_mean"],
        alternative="two-sided",
        method="exact",
    )
    print(f"rate: n={len(high)} vs {len(blank)}, U={rate.statistic}, p={rate.pvalue}")
    print(
        "delta_mean: "
        f"n={len(high)} vs {len(blank)}, U={delta_mean.statistic}, p={delta_mean.pvalue}"
    )


if __name__ == "__main__":
    main()
