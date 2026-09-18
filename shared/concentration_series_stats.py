"""Blank-threshold and Mann-Whitney statistics for a high-vs-blank concentration series.

Rewritten (not copied) from logic read in PC-alignment-anti (the frozen, read-only
reference production repo; not modified by this module):

- ``blank_threshold``: the ``blank.mean() - 3 * blank.std()`` (population std) rule
  used by ``summarize()`` and the per-assay threshold in
  ``investigation/260907_pipeline_rebuild/scripts/run_fov_level_direct_sampling.py``.
- ``fov_exceeds_rate_mann_whitney``: the per-FOV exact two-sided Mann-Whitney U
  test (``fov_mw`` in the same script) over each FOV's *threshold-exceedance
  rate* (the fraction of its valid grid points with contrast delta below the
  dataset's blank threshold -- ``summarize()``'s ``threshold_exceedance_rate``),
  not the FOV's raw mean delta. Confirmed against the 2026-09-17 throwaway
  Phase-2-substitution recompute (``phase2_recompute_mannwhitney_summary_20260917.csv``):
  its reported p-values reproduce exactly when the per-FOV test is run on
  exceedance rate, and do not when run on the raw per-FOV mean delta. Extended
  here with an exact one-sided ``alternative="greater"`` test so the sign of
  any departure from the null is explicit rather than only its two-sided
  significance.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import mannwhitneyu


def blank_threshold(blank_pooled_delta) -> float:
    """Pooled-blank contrast-delta ``mean - 3 * population_std``.

    ``blank_pooled_delta`` is every valid grid-point ``post_contrast - pre_contrast``
    value from every accepted blank-sample FOV of one dataset, concatenated.
    """
    values = np.asarray(blank_pooled_delta, dtype=float)
    if values.size == 0:
        raise ValueError("blank_pooled_delta is empty.")
    return float(values.mean() - 3 * values.std())


def fov_exceeds_rate_mann_whitney(high_fov_exceeds_rate, blank_fov_exceeds_rate) -> dict:
    """Exact Mann-Whitney U test over per-FOV threshold-exceedance rates.

    Returns both the two-sided p-value and the one-sided ``high > blank``
    p-value: a large one-sided p-value here means the departure (if any) runs
    the other way (blank tends greater than high), which the two-sided p-value
    alone does not distinguish.
    """
    high = np.asarray(high_fov_exceeds_rate, dtype=float)
    blank = np.asarray(blank_fov_exceeds_rate, dtype=float)
    if high.size == 0 or blank.size == 0:
        raise ValueError("high_fov_exceeds_rate and blank_fov_exceeds_rate must both be non-empty.")
    two_sided = mannwhitneyu(high, blank, alternative="two-sided", method="exact")
    greater = mannwhitneyu(high, blank, alternative="greater", method="exact")
    return {
        "n_high": int(high.size), "n_blank": int(blank.size),
        "U_two_sided": float(two_sided.statistic),
        "p_two_sided_exact": float(two_sided.pvalue),
        "p_one_sided_greater_exact": float(greater.pvalue),
    }
