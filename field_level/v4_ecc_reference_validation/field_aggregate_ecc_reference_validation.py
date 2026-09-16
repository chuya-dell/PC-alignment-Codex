"""Aggregate bounded per-FOV v4 ECC validation runs without rerunning ECC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons = sorted(args.runs.glob("fov*/ecc_20260901_reference_comparison.csv"))
    histories = sorted(args.runs.glob("fov*/ecc_20260901_reference_level_history.csv"))
    if len(comparisons) != 7 or len(histories) != 7:
        raise RuntimeError(f"Expected 7 complete runs, got {len(comparisons)} comparisons and {len(histories)} histories")
    results = pd.concat([pd.read_csv(path) for path in comparisons], ignore_index=True).sort_values("position")
    history = pd.concat([pd.read_csv(path) for path in histories], ignore_index=True).sort_values(["position", "scale"])
    results.to_csv(args.output / "ecc_20260901_reference_comparison.csv", index=False)
    history.to_csv(args.output / "ecc_20260901_reference_level_history.csv", index=False)
    summary = {
        "method": "reused Phase-2 native-16-bit-to-0..1 affine ECC, zero start, 0.25->0.5->1.0; 100 iterations, EPS=1e-6 per level",
        "positions": results.position.astype(int).tolist(),
        "full_match_count": int(results.full_match.sum()),
        "eligible_for_position6_rescue": False,
        "reason": "The proposed minimum of six full matches was not met; no Position-6 rescue search was run.",
        "mean_absolute_differences": {
            key: float(results[f"difference_{key}"].abs().mean())
            for key in ("rho", "tx_px", "ty_px", "theta_deg", "delta_scale_pct")
        },
        "median_absolute_differences": {
            key: float(results[f"difference_{key}"].abs().median())
            for key in ("rho", "tx_px", "ty_px", "theta_deg", "delta_scale_pct")
        },
    }
    (args.output / "ecc_20260901_reference_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
