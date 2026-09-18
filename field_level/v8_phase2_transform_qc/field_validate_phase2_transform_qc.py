"""Validate the Phase-2 affine QC gate against a saved pair-comparison CSV.

The source comparison has scalar centre displacement, rotation and scale, not
the original matrices. This script reconstructs the equivalent similarity
matrix at a 2048x2048 image centre and evaluates the production QC function.
It is therefore a regression check of the gate, not a re-registration run.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.registration import assess_affine_transform_qc


IMAGE_SHAPE = (2048, 2048)


def matrix_from_metrics(dx: float, dy: float, rotation_deg: float, scale: float) -> np.ndarray:
    theta = math.radians(rotation_deg)
    linear = scale * np.array([[math.cos(theta), -math.sin(theta)],
                               [math.sin(theta), math.cos(theta)]])
    centre = np.array([(IMAGE_SHAPE[1] - 1) / 2, (IMAGE_SHAPE[0] - 1) / 2])
    translation = centre + np.array([dx, dy]) - linear @ centre
    return np.column_stack([linear, translation])


def legacy_degenerate(row: pd.Series) -> bool:
    return (abs(row.phase2_dx_center) > 150 or abs(row.phase2_dy_center) > 150
            or abs(row.phase2_scale - 1) > .10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = pd.read_csv(args.comparison_csv)
    source = source[source.phase2_status.eq("ok")].copy()
    numeric = ["phase2_dx_center", "phase2_dy_center", "phase2_rotation_deg", "phase2_scale"]
    source[numeric] = source[numeric].apply(pd.to_numeric, errors="coerce")
    source = source.dropna(subset=numeric).copy()
    rows = []
    for row in source.itertuples(index=False):
        matrix = matrix_from_metrics(row.phase2_dx_center, row.phase2_dy_center,
                                     row.phase2_rotation_deg, row.phase2_scale)
        qc = assess_affine_transform_qc(matrix, IMAGE_SHAPE)
        old = legacy_degenerate(row)
        rows.append({
            "dataset": row.dataset, "sample": row.sample, "position": row.position,
            "legacy_degenerate_150px_10pct": old,
            "qc_accepted": qc["accepted"], "qc_reasons": "; ".join(qc["reasons"]),
            "dx_center_px": qc["dx_center_px"], "dy_center_px": qc["dy_center_px"],
            "rotation_deg": qc["rotation_deg"], "scale": qc["scale"],
            "anisotropy": qc["anisotropy"],
        })
    result = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "phase2_transform_qc_validation_rows.csv", index=False)
    true_positive = int((result.legacy_degenerate_150px_10pct & ~result.qc_accepted).sum())
    false_negative = int((result.legacy_degenerate_150px_10pct & result.qc_accepted).sum())
    false_positive = int((~result.legacy_degenerate_150px_10pct & ~result.qc_accepted).sum())
    summary = pd.DataFrame([{
        "source_ok_rows": len(result), "legacy_degenerate_rows": int(result.legacy_degenerate_150px_10pct.sum()),
        "qc_rejected_rows": int((~result.qc_accepted).sum()), "true_positive": true_positive,
        "false_negative": false_negative, "false_positive": false_positive,
    }])
    summary.to_csv(args.out_dir / "phase2_transform_qc_validation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
