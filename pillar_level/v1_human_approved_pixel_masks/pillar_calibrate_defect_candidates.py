"""Measure candidate sensitivity on a no-binding control without approving masks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import defect_masking as dm, registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, action="append", default=[])
    parser.add_argument("--min-area-px", type=int, default=212)
    parser.add_argument("--max-images", type=int,
                        help="Use a deterministic first-N control-image subset for a fast calibration run.")
    args = parser.parse_args()
    thresholds = args.threshold or [5.0, 6.0, 7.0]
    rows = []
    image_paths = sorted(args.control_dir.glob("*-*-*.tif"))
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    for path in image_paths:
        image = registration.load_image_unicode(str(path))
        base_settings = dm.CandidateSettings(min_area_px=args.min_area_px)
        try:
            zscore = dm.residual_zscore(image, base_settings)
        except Exception as exc:
            for threshold in thresholds:
                rows.append({"image": path.name, "residual_z": threshold,
                             "candidate_count": 0, "candidate_area_px": 0,
                             "status": "failed", "error": str(exc)})
            continue
        for threshold in thresholds:
            settings = dm.CandidateSettings(
                residual_z=threshold, min_area_px=args.min_area_px,
            )
            try:
                regions = dm.candidate_regions_from_zscore(zscore, settings)
                rows.append({"image": path.name, "residual_z": threshold,
                             "candidate_count": len(regions),
                             "candidate_area_px": sum(r["area_px"] for r in regions),
                             "status": "measured"})
            except Exception as exc:
                rows.append({"image": path.name, "residual_z": threshold,
                             "candidate_count": 0, "candidate_area_px": 0,
                             "status": "failed", "error": str(exc)})
    detail = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output / "control_candidate_sensitivity_detail.csv", index=False)
    summary = detail.groupby("residual_z", as_index=False).agg(
        image_count=("image", "count"),
        median_candidates=("candidate_count", "median"),
        max_candidates=("candidate_count", "max"),
        total_candidates=("candidate_count", "sum"),
        median_candidate_area_px=("candidate_area_px", "median"),
    )
    summary.to_csv(args.output / "control_candidate_sensitivity_summary.csv", index=False)
    print(len(detail))
    print(len(summary))


if __name__ == "__main__":
    main()
