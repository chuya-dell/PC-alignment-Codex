"""Validate the versioned affine ECC path against the 2026-09-01 FOV ledger.

This reuses the Phase-2 image normalization, affine metric extraction, and
coarse-to-fine ECC implementation.  It deliberately does not alter production
registration and does not run a Position-6 rescue grid: that requires evidence
that the seven reference FOVs reproduce the historical transform convention.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
V3_PATH = ROOT / "field_level" / "v3_fov16_ecc_root_cause" / "field_diagnose_fov16_ecc.py"
spec = importlib.util.spec_from_file_location("phase2_ecc", V3_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load reused Phase-2 ECC module: {V3_PATH}")
phase2_ecc = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = phase2_ecc
spec.loader.exec_module(phase2_ecc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/raw/ecc_20260901_reference_validation_manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", default=None,
                        help="Optional subset of validation positions; useful for bounded per-FOV runs.")
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    root = Path(config["root"])
    rows, histories = [], []
    tolerances = config["proposed_validation_tolerances"]
    positions = args.positions if args.positions is not None else config["validation_positions"]
    unknown = set(positions) - set(config["validation_positions"])
    if unknown:
        raise ValueError(f"Positions are not validation references: {sorted(unknown)}")
    for position in positions:
        print(f"Position {position}: starting ECC", flush=True)
        pre_path = root / f"{config['sample']}-{position}-0.tif"
        post_path = root / f"{config['sample']}-{position}-1.tif"
        pre = phase2_ecc.load_image_unicode(str(pre_path))
        post = phase2_ecc.load_image_unicode(str(post_path))
        if pre is None or post is None:
            raise FileNotFoundError(f"Missing pre/post image for Position {position}: {pre_path}, {post_path}")
        result, _warp = phase2_ecc.run_ecc_path(
            phase2_ecc.image_for_ecc(pre), phase2_ecc.image_for_ecc(post), 0.0, 0.0,
            config["pyramid_scales"], iterations=100, eps=1e-6,
        )
        reference = config["reference_20260901"][str(position)]
        row = {"position": position, **{f"rebuilt_{k}": v for k, v in result.items() if k != "levels"},
               **{f"recorded_{k}": v for k, v in reference.items()}}
        for key in ("rho", "tx_px", "ty_px", "theta_deg", "delta_scale_pct"):
            row[f"difference_{key}"] = row[f"rebuilt_{key}"] - row[f"recorded_{key}"]
        row["rho_within_tolerance"] = abs(row["difference_rho"]) <= tolerances["rho_abs"]
        row["tx_within_tolerance"] = abs(row["difference_tx_px"]) <= tolerances["translation_axis_px"]
        row["ty_within_tolerance"] = abs(row["difference_ty_px"]) <= tolerances["translation_axis_px"]
        row["theta_within_tolerance"] = abs(row["difference_theta_deg"]) <= tolerances["theta_deg"]
        row["scale_within_tolerance"] = abs(row["difference_delta_scale_pct"]) <= tolerances["delta_scale_pct"]
        row["full_match"] = all(row[k] for k in (
            "rho_within_tolerance", "tx_within_tolerance", "ty_within_tolerance",
            "theta_within_tolerance", "scale_within_tolerance"))
        rows.append(row)
        for level in result["levels"]:
            histories.append({"position": position, **level})
        print(f"Position {position}: rho={result['rho']:.6f}", flush=True)

    results = pd.DataFrame(rows).sort_values("position")
    history = pd.DataFrame(histories).sort_values(["position", "scale"])
    results.to_csv(output / "ecc_20260901_reference_comparison.csv", index=False)
    history.to_csv(output / "ecc_20260901_reference_level_history.csv", index=False)
    summary = {
        "method": "reused Phase-2 native-16-bit-to-0..1 affine ECC, zero start, 0.25->0.5->1.0; 100 iterations, EPS=1e-6 per level",
        "positions": positions,
        "tolerances": tolerances,
        "full_match_count": int(results.full_match.sum()),
        "minimum_full_matches": int(tolerances["minimum_full_matches"]),
        "eligible_for_position6_rescue": bool(results.full_match.sum() >= tolerances["minimum_full_matches"]),
        "mean_absolute_differences": {
            key: float(results[f"difference_{key}"].abs().mean())
            for key in ("rho", "tx_px", "ty_px", "theta_deg", "delta_scale_pct")
        },
    }
    (output / "ecc_20260901_reference_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
