"""Generate human-review candidates; this command never writes approved masks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import analyzer, defect_masking as dm, registration


REVIEW_COLUMNS = [
    "candidate_id", "date", "modality", "sample_id", "position_id",
    "source", "primary_observation", "automatic_image", "area_px",
    "max_residual_z", "review_image", "mask_polygon_pre_json",
    "decision", "pre_reviewed", "post_reviewed", "pre_defect_seen",
    "post_defect_seen", "approved_at", "approved_by", "review_notes",
]


def dense_points(image_path: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)
    points = analyzer.analyze_image(str(image_path), method="peak", min_dist=3, threshold=.2).copy()
    if "pillar_id" not in points:
        points.insert(0, "pillar_id", np.arange(len(points), dtype=int))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(cache_path, index=False)
    return points


def pair_transform(pre: Path, post: Path, cache: Path) -> dict:
    p = dense_points(pre, cache / f"{pre.stem}_points.csv")
    q = dense_points(post, cache / f"{post.stem}_points.csv")
    _, transform, _, _, _, _ = registration.align_and_match_dataframes(
        p, q, str(pre), str(post), return_diagnostics=True, local_refinement=False,
    )
    return transform


def display_image(image: np.ndarray) -> np.ndarray:
    low, high = np.quantile(image, [.01, .99])
    scaled = np.clip((image.astype(float) - low) / max(high - low, 1), 0, 1)
    return cv2.cvtColor(np.rint(scaled * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def crop_bounds(polygon: np.ndarray, shape: tuple[int, int], padding: int = 80):
    x0, y0 = np.floor(polygon.min(axis=0)).astype(int) - padding
    x1, y1 = np.ceil(polygon.max(axis=0)).astype(int) + padding + 1
    return max(0, x0), max(0, y0), min(shape[1], x1), min(shape[0], y1)


def review_montage(
    pre_image: np.ndarray, post_image: np.ndarray, polygon_pre: np.ndarray | None,
    transform: dict, label: str,
) -> np.ndarray:
    pre_view, post_view = display_image(pre_image), display_image(post_image)
    if polygon_pre is not None:
        polygon_post = dm.transform_polygon(polygon_pre, transform, "pre_to_post")
        cv2.polylines(pre_view, [np.rint(polygon_pre).astype(np.int32)], True, (0, 0, 255), 3)
        cv2.polylines(post_view, [np.rint(polygon_post).astype(np.int32)], True, (0, 0, 255), 3)
        pre_box = crop_bounds(polygon_pre, pre_image.shape)
        post_box = crop_bounds(polygon_post, post_image.shape)
        x0, y0, x1, y1 = pre_box
        a = pre_view[y0:y1, x0:x1]
        x0, y0, x1, y1 = post_box
        b = post_view[y0:y1, x0:x1]
    else:
        target_width = 900
        scale = target_width / pre_view.shape[1]
        size = (target_width, int(round(pre_view.shape[0] * scale)))
        a = cv2.resize(pre_view, size, interpolation=cv2.INTER_AREA)
        b = cv2.resize(post_view, size, interpolation=cv2.INTER_AREA)
    height = max(a.shape[0], b.shape[0])
    def pad(im):
        return cv2.copyMakeBorder(im, 0, height - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    montage = np.hstack([pad(a), pad(b)])
    cv2.rectangle(montage, (0, 0), (montage.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(montage, f"PRE | {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    cv2.putText(montage, "POST", (a.shape[1] + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
    return montage


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(path)


def blank_review_fields(row: dict) -> dict:
    return {**row, "decision": "", "pre_reviewed": "", "post_reviewed": "",
            "pre_defect_seen": "", "post_defect_seen": "", "approved_at": "",
            "approved_by": "", "review_notes": ""}


def process_pair(ds, sample, position, records, output, cache, settings):
    stem = f"{sample}-{position}"
    pre = Path(ds["root"]) / f"{stem}-0.tif"
    post = Path(ds["root"]) / f"{stem}-1.tif"
    if not pre.exists() or not post.exists():
        return [], {"dataset": ds["id"], "sample_id": sample, "position_id": position,
                    "status": "missing_pair", "candidate_count": 0}
    pre_image = registration.load_image_unicode(str(pre))
    post_image = registration.load_image_unicode(str(post))
    transform = pair_transform(pre, post, cache / ds["id"] / stem)
    rows = []
    sequence = 0
    for image_name, image in (("pre", pre_image), ("post", post_image)):
        regions, _ = dm.residual_candidate_regions(image, settings)
        for region in regions:
            sequence += 1
            polygon = np.asarray(json.loads(region["polygon_native_json"]), float)
            polygon_pre = polygon if image_name == "pre" else dm.transform_polygon(
                polygon, transform, "post_to_pre")
            candidate_id = f"{ds['id']}-{stem}-A{sequence:03d}"
            relative = Path("images") / f"{candidate_id}.png"
            write_png(output / relative, review_montage(
                pre_image, post_image, polygon_pre, transform,
                f"automatic candidate ({image_name})",
            ))
            rows.append(blank_review_fields({
                "candidate_id": candidate_id, "date": ds["date"],
                "modality": ds["modality"], "sample_id": sample,
                "position_id": position, "source": "automatic",
                "primary_observation": "", "automatic_image": image_name,
                "area_px": region["area_px"], "max_residual_z": region["max_residual_z"],
                "review_image": relative.as_posix(),
                "mask_polygon_pre_json": json.dumps(polygon_pre.tolist(), separators=(",", ":")),
            }))
    matching = records[
        (records.date.astype(str) == str(ds["date"]))
        & (records.modality == ds["modality"])
        & (records.sample_id.astype(str) == str(sample))
        & (records.position_id.astype(str) == str(position))
    ]
    # Every image pair receives a coverage row.  This prevents absence of a
    # primary note or absence of an automatic blob from becoming "clean" by
    # default; the reviewer must inspect both complete images independently.
    candidate_id = f"{ds['id']}-{stem}-V001"
    relative = Path("images") / f"{candidate_id}.png"
    write_png(output / relative, review_montage(
        pre_image, post_image, None, transform, "complete-view coverage review",
    ))
    rows.append(blank_review_fields({
        "candidate_id": candidate_id, "date": ds["date"],
        "modality": ds["modality"], "sample_id": sample,
        "position_id": position, "source": "full_view_review",
        "primary_observation": "", "automatic_image": "",
        "area_px": "", "max_residual_z": "",
        "review_image": relative.as_posix(), "mask_polygon_pre_json": "",
    }))
    for primary_index, (_, record) in enumerate(matching.iterrows(), 1):
        candidate_id = f"{ds['id']}-{stem}-P{primary_index:03d}"
        relative = Path("images") / f"{candidate_id}.png"
        write_png(output / relative, review_montage(
            pre_image, post_image, None, transform, "primary-record view; locate manually",
        ))
        rows.append(blank_review_fields({
            "candidate_id": candidate_id, "date": ds["date"],
            "modality": ds["modality"], "sample_id": sample,
            "position_id": position, "source": "primary_record",
            "primary_observation": record.observation,
            "automatic_image": "", "area_px": "", "max_residual_z": "",
            "review_image": relative.as_posix(), "mask_polygon_pre_json": "",
        }))
    return rows, {"dataset": ds["id"], "sample_id": sample, "position_id": position,
                  "status": "review_required", "candidate_count": len(rows),
                  "automatic_count": sequence, "primary_record_count": len(matching),
                  "full_view_review_count": 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--primary-records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--dataset", action="append", help="Limit to an id; repeatable.")
    parser.add_argument("--residual-z", type=float, default=6.0)
    parser.add_argument("--min-area-px", type=int, default=212)
    args = parser.parse_args()
    datasets = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.dataset:
        datasets = [d for d in datasets if d["id"] in set(args.dataset)]
    records = pd.read_csv(args.primary_records, dtype=str, keep_default_na=False)
    settings = dm.CandidateSettings(residual_z=args.residual_z, min_area_px=args.min_area_px)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, status = [], []
    for ds in datasets:
        for group in ("high", "blank"):
            sample = ds[group]["sample"]
            for position in ds[group]["positions"]:
                try:
                    new_rows, state = process_pair(
                        ds, sample, position, records, args.output, args.cache, settings,
                    )
                    rows.extend(new_rows); status.append(state)
                except Exception as exc:
                    status.append({"dataset": ds["id"], "sample_id": sample,
                                   "position_id": position, "status": "failed",
                                   "candidate_count": 0, "error": str(exc)})
    pd.DataFrame(rows, columns=REVIEW_COLUMNS).to_csv(args.output / "candidate_review.csv", index=False)
    pd.DataFrame(status).to_csv(args.output / "generation_status.csv", index=False)
    metadata = {"coordinate_system": "native pre-image pixels; top-left origin; x-right; y-down",
                "pitch_px": settings.pitch_px, "settings": settings.__dict__,
                "automatic_output_is_candidate_only": True,
                "approval_required_before_analysis": True}
    (args.output / "review_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(len(rows))
    print(len(status))


if __name__ == "__main__":
    main()
