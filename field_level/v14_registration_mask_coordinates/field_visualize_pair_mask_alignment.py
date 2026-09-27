"""Render independent pre/post stain masks before and after coordinate mapping."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.image_qc import stain_artifact_mask
from shared.registration import register_image_pair_affine, load_image_unicode


CASES = [
    ("substrate3_position2_100x", "100-3-2"),
    ("substrate3_position2_50x", "50-3-2"),
    ("substrate5_position5_100x", "100-5-5"),
    ("substrate5_position5_50x", "50-5-5"),
    ("substrate5_position6_100x", "100-5-6"),
    ("substrate5_position6_50x", "50-5-6"),
    ("sample3_position3_100x", "100-3-3"),
    ("sample3_position3_50x", "50-3-3"),
]


def paint(image, pre_mask=None, post_mask=None, union=False):
    small = cv2.resize(image, (720, 720), interpolation=cv2.INTER_AREA)
    view = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    canvas = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    if union:
        a = cv2.resize(pre_mask.astype(np.uint8), (720, 720), interpolation=cv2.INTER_NEAREST) > 0
        b = cv2.resize(post_mask.astype(np.uint8), (720, 720), interpolation=cv2.INTER_NEAREST) > 0
        canvas[a & ~b] = (0, 0, 255)       # pre only, red
        canvas[b & ~a] = (255, 0, 0)       # mapped post only, blue
        canvas[a & b] = (255, 0, 255)      # overlap, magenta
    else:
        mask = pre_mask if pre_mask is not None else post_mask
        m = cv2.resize(mask.astype(np.uint8), (720, 720), interpolation=cv2.INTER_NEAREST) > 0
        canvas[m] = (0, 0, 255)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for case_id, stem in CASES:
        root = args.data_root / "260922-p50-dna"
        pre_path, post_path = root / f"{stem}-0.tif", root / f"{stem}-1.tif"
        pre, post = load_image_unicode(str(pre_path)), load_image_unicode(str(post_path))
        pre_mask, post_mask = stain_artifact_mask(pre), stain_artifact_mask(post)
        try:
            matrix, qc = register_image_pair_affine(
                pre, post, mask_stains=True, return_qc=True)
        except Exception as exc:
            records.append({"case_id": case_id, "status": "registration_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "pre_mask_fraction": float(pre_mask.mean()),
                            "post_mask_fraction": float(post_mask.mean())})
            continue
        h, w = pre.shape
        post_in_pre = cv2.warpAffine(
            post_mask.astype(np.uint8), matrix, (w, h),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
        common = pre_mask | post_in_pre
        naive_union = pre_mask | post_mask
        aligned_post = cv2.warpAffine(
            post, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101)
        panels = [
            paint(pre, pre_mask=pre_mask),
            paint(post, post_mask=post_mask),
            paint(pre, pre_mask=pre_mask, post_mask=post_in_pre, union=True),
            paint(aligned_post, pre_mask=common),
        ]
        header = np.full((34, 720 * 4, 3), 32, np.uint8)
        titles = ["pre native | red=pre mask", "post native | red=post mask",
                  "pre frame | red=pre, blue=mapped post, magenta=both",
                  "post warped to pre | red=common union"]
        for i, title in enumerate(titles):
            cv2.putText(header, title, (i * 720 + 8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .50, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.output / f"pair_mask_alignment_{case_id}.jpg"),
                    np.vstack([header, np.hstack(panels)]), [cv2.IMWRITE_JPEG_QUALITY, 90])
        pre_count, post_count = int(pre_mask.sum()), int(post_mask.sum())
        common_count = int(common.sum())
        records.append({
            "case_id": case_id, "status": "ok", "pre_mask_pixels": pre_count,
            "post_mask_pixels": post_count, "naive_same_coordinate_or_pixels": int(naive_union.sum()),
            "aligned_common_pre_pixels": common_count,
            "pre_mask_fraction": float(pre_mask.mean()), "post_mask_fraction": float(post_mask.mean()),
            "aligned_common_pre_fraction": float(common.mean()),
            "pre_post_raw_coordinate_iou": float(np.logical_and(pre_mask, post_mask).sum() /
                                                 max(1, naive_union.sum())),
            "naive_union_vs_aligned_union_iou": float(
                np.logical_and(naive_union, common).sum() /
                max(1, np.logical_or(naive_union, common).sum())),
            "post_raw_vs_mapped_iou": float(np.logical_and(post_mask, post_in_pre).sum() /
                                            max(1, np.logical_or(post_mask, post_in_pre).sum())),
            "matrix": np.asarray(matrix).tolist(),
            "center_translation_px": float(np.linalg.norm(
                np.array([(w - 1) / 2, (h - 1) / 2]) @ matrix[:, :2].T +
                matrix[:, 2] - np.array([(w - 1) / 2, (h - 1) / 2]))),
            "qc_mask_fraction": qc["mask_fraction"],
        })
    with (args.output / "pair_mask_alignment_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        keys = list(dict.fromkeys(k for row in records for k in row))
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
