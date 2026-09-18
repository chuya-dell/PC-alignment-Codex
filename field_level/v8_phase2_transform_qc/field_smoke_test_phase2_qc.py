"""Exercise the real Phase-2 entry point and record safe batch-style outcomes.

The current production pipeline does not yet invoke Phase 2.  This tool is
therefore a minimal reference for its future batch caller: an affine QC
rejection is caught and recorded as ``registration_qc_rejected`` instead of
being allowed into contrast sampling or statistics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import registration as reg


def run_pair(row: pd.Series) -> dict:
    pre = reg.load_image_unicode(row.pre_path)
    post = reg.load_image_unicode(row.post_path)
    if pre is None or post is None:
        return {"id": row.id, "status": "input_error", "error": "image load failed"}
    try:
        _, qc = reg.register_image_pair_affine(pre, post, return_qc=True)
    except reg.AffineTransformQCError as exc:
        qc = exc.diagnostics
        return {
            "id": row.id, "status": "registration_qc_rejected",
            "error": str(exc), "dx_center_px": qc["dx_center_px"],
            "dy_center_px": qc["dy_center_px"], "rotation_deg": qc["rotation_deg"],
            "scale": qc["scale"], "anisotropy": qc["anisotropy"],
            "mask_fraction": qc.get("mask_fraction"), "qc_reasons": "; ".join(qc["reasons"]),
        }
    return {
        "id": row.id, "status": "ok", "error": "",
        "dx_center_px": qc["dx_center_px"], "dy_center_px": qc["dy_center_px"],
        "rotation_deg": qc["rotation_deg"], "scale": qc["scale"],
        "anisotropy": qc["anisotropy"], "mask_fraction": qc["mask_fraction"], "qc_reasons": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs_csv", type=Path, help="CSV with id,pre_path,post_path columns")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs_csv)
    needed = {"id", "pre_path", "post_path"}
    if not needed.issubset(pairs.columns):
        raise ValueError(f"pairs_csv must contain {sorted(needed)}")
    rows = [run_pair(row) for row in pairs.itertuples(index=False)]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
