"""Compare legacy whole-view exclusion with approved pixel masks.

This command reports counts only.  It does not recompute or replace any
previously fixed p-value and refuses to run while candidate review is pending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyzer
import defect_masking as dm
import lattice_indexing as li
import registration
from theoretical_grid_evaluation import sample_grid_features


def strict_bool(value, field, candidate):
    text = str(value).strip().lower()
    if text not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{candidate}: {field} must be TRUE or FALSE.")
    return text in {"true", "1", "yes"}


def validate_completed_review(review_csv: Path, datasets, primary_records: pd.DataFrame) -> pd.DataFrame:
    review = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    missing = dm.REQUIRED_APPROVAL_COLUMNS - set(review.columns)
    if missing:
        raise ValueError(f"Review table is missing columns: {sorted(missing)}")
    for _, row in review.iterrows():
        candidate = row["candidate_id"]
        if row["decision"].strip().lower() not in {"approved", "rejected"}:
            raise ValueError(f"{candidate}: review is incomplete.")
        if not strict_bool(row["pre_reviewed"], "pre_reviewed", candidate):
            raise ValueError(f"{candidate}: pre image has not been reviewed.")
        if not strict_bool(row["post_reviewed"], "post_reviewed", candidate):
            raise ValueError(f"{candidate}: post image has not been reviewed.")
    for ds in datasets:
        for group in ("high", "blank"):
            sample = ds[group]["sample"]
            for position in ds[group]["positions"]:
                pair_rows = review[
                    (review.date.astype(str) == str(ds["date"]))
                    & (review.sample_id.astype(str) == str(sample))
                    & (review.position_id.astype(str) == str(position))
                ]
                if not (pair_rows.source == "full_view_review").any():
                    raise ValueError(
                        f"{ds['id']} {sample}-{position}: complete-view review is missing."
                    )
                primary = primary_records[
                    (primary_records.date.astype(str) == str(ds["date"]))
                    & (primary_records.modality == ds["modality"])
                    & (primary_records.sample_id.astype(str) == str(sample))
                    & (primary_records.position_id.astype(str) == str(position))
                ]
                actual = int((pair_rows.source == "primary_record").sum())
                if actual < len(primary):
                    raise ValueError(
                        f"{ds['id']} {sample}-{position}: primary-record review row is missing."
                    )
    return review


def dense_points(path: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    frame = analyzer.analyze_image(str(path), method="peak", min_dist=3, threshold=.2).copy()
    if "pillar_id" not in frame:
        frame.insert(0, "pillar_id", np.arange(len(frame), dtype=int))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index=False)
    return frame


def sample_pair(ds, sample, position, review_csv, cache, pitch, whole_threshold):
    root = Path(ds["root"]); stem = f"{sample}-{position}"
    pre_path, post_path = root / f"{stem}-0.tif", root / f"{stem}-1.tif"
    pre_image = registration.load_image_unicode(str(pre_path))
    post_image = registration.load_image_unicode(str(post_path))
    if pre_image is None or post_image is None:
        raise ValueError("Image pair is unreadable.")
    p = dense_points(pre_path, cache / ds["id"] / stem / "pre_points.csv")
    q = dense_points(post_path, cache / ds["id"] / stem / "post_points.csv")
    _, transform, iterations, converged, _, _ = registration.align_and_match_dataframes(
        p, q, str(pre_path), str(post_path), return_diagnostics=True,
        local_refinement=False,
    )
    lattice = li.lattice_from_fft(pre_image, pitch)
    base = sample_grid_features(pre_image, lattice, margin=30)
    pre_xy = base[["x", "y"]].to_numpy(float)
    post_xy = registration.inverse_transform_points(pre_xy, transform)
    post_base = registration.sample_contrast(post_image, post_xy)
    unmasked_valid = base.valid_sampling.to_numpy() & post_base.valid_sampling.to_numpy()
    polygons = dm.load_approved_polygons(review_csv, ds["date"], sample, position)
    pre_mask, post_mask = dm.masks_for_pair(pre_image.shape, polygons, transform)
    masked_pre = registration.sample_contrast(pre_image, pre_xy, invalid_mask=pre_mask)
    masked_post = registration.sample_contrast(post_image, post_xy, invalid_mask=post_mask)
    masked_valid = masked_pre.valid_sampling.to_numpy() & masked_post.valid_sampling.to_numpy()
    fraction, _ = dm.excluded_grid_fraction(pre_xy, post_xy, pre_mask, post_mask)
    whole_view_excluded = fraction > whole_threshold
    if whole_view_excluded:
        masked_valid[:] = False
    return {
        "dataset": ds["id"], "date": ds["date"], "sample_id": sample,
        "position_id": position, "registration_status": "ok",
        "grid_points_after_border": int(len(base)),
        "valid_points_without_defect_mask": int(unmasked_valid.sum()),
        "approved_mask_count": int(len(polygons)),
        "masked_grid_fraction": fraction,
        "whole_view_excluded_over_30pct": bool(whole_view_excluded),
        "valid_points_with_pixel_mask": int(masked_valid.sum()),
        "icp_iterations": int(iterations), "icp_converged": bool(converged),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--primary-records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--pitch", type=float, default=7.286)
    parser.add_argument("--whole-view-threshold", type=float, default=.30)
    args = parser.parse_args()
    datasets = json.loads(args.manifest.read_text(encoding="utf-8"))
    primary_records = pd.read_csv(args.primary_records, dtype=str, keep_default_na=False)
    validate_completed_review(args.review, datasets, primary_records)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in datasets:
        for group in ("high", "blank"):
            sample = ds[group]["sample"]
            for position in ds[group]["positions"]:
                try:
                    row = sample_pair(ds, sample, position, args.review, args.cache,
                                      args.pitch, args.whole_view_threshold)
                except Exception as exc:
                    row = {"dataset": ds["id"], "date": ds["date"],
                           "sample_id": sample, "position_id": position,
                           "registration_status": "failed", "error": str(exc)}
                row["group"] = group
                # Historical membership is used only as the before-condition.
                # It is not evidence that an unrecorded view was defect-free.
                row["legacy_whole_view_included"] = position in ds["legacy_included"][group]
                rows.append(row)
    detail = pd.DataFrame(rows)
    detail.to_csv(args.output / "view_count_detail.csv", index=False)
    summaries = []
    for ds in datasets:
        for group in ("high", "blank"):
            part = detail[(detail.dataset == ds["id"]) & (detail.group == group)]
            successful = part.registration_status == "ok"
            legacy = successful & part.legacy_whole_view_included.astype(bool)
            pixel = successful & ~part.whole_view_excluded_over_30pct.fillna(True).astype(bool)
            summaries.append({
                "dataset": ds["id"], "date": ds["date"], "group": group,
                "sample_id": ds[group]["sample"],
                "legacy_valid_views": int(legacy.sum()),
                "pixel_mask_valid_views": int(pixel.sum()),
                "restored_views": int(pixel.sum() - legacy.sum()),
                "legacy_valid_pillars": int(part.loc[legacy, "valid_points_without_defect_mask"].sum()),
                "pixel_mask_valid_pillars": int(part.loc[pixel, "valid_points_with_pixel_mask"].sum()),
                "pillar_change": int(part.loc[pixel, "valid_points_with_pixel_mask"].sum()
                                     - part.loc[legacy, "valid_points_without_defect_mask"].sum()),
            })
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.output / "count_comparison.csv", index=False)
    dm.write_audit_log(args.review, args.output / "approved_mask_audit_log.csv")
    report = [
        "# ピクセル欠陥マスク適用前後の有効数比較", "",
        "このレポートは有効ピラー数・有効視野数の変化だけを示す。260907〜260908に確定した、視野全体除外方式に基づく各日程のp値を上書き・再確定しない。統計的な再検定は別タスクとする。", "",
        summary.to_markdown(index=False), "",
        "未承認の自動候補は欠陥マスクとして使用していない。監査ログに記録された、pre/post双方を目視確認済みの承認マスクだけを適用した。", "",
    ]
    (args.output / "count_comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    print(len(detail))
    print(len(summary))


if __name__ == "__main__":
    main()
