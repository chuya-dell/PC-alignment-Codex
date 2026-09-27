"""Compare stain candidates with 260922 log-level defect records and inspect SAM outlier.

The experiment log records FOVs, not pixel annotations. This script therefore
reports candidate-mask coverage and overlays for human review; it does not claim
pixelwise precision/recall or use the candidates to alter scientific analysis.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.image_qc import stain_artifact_mask
from shared.lattice_indexing import grid_coordinates, lattice_from_fft
from shared.registration import load_image_unicode


RECORDS = [
    ("substrate3_position2_near_cross_100x", "100-3-2", "stain near cross scratch"),
    ("substrate3_position2_near_cross_50x", "50-3-2", "stain near cross scratch"),
    ("substrate5_position5_100x", "100-5-5", "stain"),
    ("substrate5_position5_50x", "50-5-5", "stain"),
    ("substrate5_position6_100x", "100-5-6", "stain"),
    ("substrate5_position6_50x", "50-5-6", "stain"),
    ("sample3_position3_100x", "100-3-3", "many dirt"),
    ("sample3_position3_50x", "50-3-3", "many dirt"),
]


def overlay_image(image: np.ndarray, mask: np.ndarray, points=None) -> np.ndarray:
    view = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    rgb = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    rgb[mask] = (0.45 * rgb[mask] + 0.55 * np.array([20, 30, 255])).astype(np.uint8)
    if points is not None:
        for x, y in points:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= iy < rgb.shape[0] and 0 <= ix < rgb.shape[1]:
                color = (0, 220, 255) if mask[iy, ix] else (80, 255, 80)
                cv2.circle(rgb, (ix, iy), 2, color, -1, cv2.LINE_AA)
    return rgb


def save_panel(path: Path, title: str, image: np.ndarray, mask: np.ndarray, points=None) -> None:
    rgb = overlay_image(image, mask, points)
    max_dim = 1800
    scale = min(1.0, max_dim / max(rgb.shape[:2]))
    if scale < 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    header = np.full((38, rgb.shape[1], 3), 30, np.uint8)
    cv2.putText(header, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([header, rgb]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []

    for label, stem, record in RECORDS:
        for phase in (0, 1):
            path = args.data_root / "260922-p50-dna" / f"{stem}-{phase}.tif"
            if not path.exists():
                continue
            image = load_image_unicode(str(path))
            mask = stain_artifact_mask(image)
            lattice_stats = {}
            if "100x" in label:
                try:
                    lattice = lattice_from_fft(image, pitch_px=7.286)
                    _, grid = grid_coordinates(lattice, image.shape[1], image.shape[0], margin=20)
                    xi = np.clip(np.rint(grid[:, 0]).astype(int), 0, image.shape[1] - 1)
                    yi = np.clip(np.rint(grid[:, 1]).astype(int), 0, image.shape[0] - 1)
                    near = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
                    lattice_stats = {
                        "fft_grid_centers": int(len(grid)),
                        "grid_centers_in_mask": int(mask[yi, xi].sum()),
                        "grid_center_mask_fraction": float(mask[yi, xi].mean()),
                        "grid_centers_within_2px_of_mask": int(near[yi, xi].sum()),
                        "grid_neighborhood_mask_fraction": float(near[yi, xi].mean()),
                    }
                except ValueError as exc:
                    lattice_stats = {"fft_grid_error": str(exc)}
            save_panel(args.output / f"recorded_{label}_{'pre' if phase == 0 else 'post'}.png",
                       f"{label} | {record} | {'pre' if phase == 0 else 'post'} | masked={mask.mean():.3%}",
                       image, mask)
            summaries.append({"record_id": label, "recorded_defect": record,
                              "phase": "pre" if phase == 0 else "post", "path": str(path),
                              "shape": f"{image.shape[1]}x{image.shape[0]}",
                              "mask_pixels": int(mask.sum()), "mask_fraction": float(mask.mean()),
                              "pixelwise_truth_available": False, **lattice_stats})

    sam_dir = args.data_root / "260829-p50-sam"
    for phase in (0, 1):
        path = sam_dir / f"1-6-{phase}.tif"
        image = load_image_unicode(str(path))
        mask = stain_artifact_mask(image)
        lattice = lattice_from_fft(image, pitch_px=7.286)
        _, points = grid_coordinates(lattice, image.shape[1], image.shape[0], margin=20)
        xi = np.clip(np.rint(points[:, 0]).astype(int), 0, image.shape[1] - 1)
        yi = np.clip(np.rint(points[:, 1]).astype(int), 0, image.shape[0] - 1)
        center_masked = mask[yi, xi]
        # Pillar-neighborhood coverage is descriptive: FFT phase gives lattice
        # locations, while a center hit alone does not prove a pillar was removed.
        kernel = np.ones((5, 5), np.uint8)
        near = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        near_masked = near[yi, xi]
        save_panel(args.output / f"sam_pos6_axis14_{'pre' if phase == 0 else 'post'}_mask_fft_grid.png",
                   f"260829 SAM pos6 axis14 source {phase} | red=mask; green=FFT centers; yellow=center in mask",
                   image, mask, points)
        summaries.append({"record_id": "260829_SAM_pos6_axis14", "recorded_defect": "not recorded",
                          "phase": "pre" if phase == 0 else "post", "path": str(path),
                          "shape": f"{image.shape[1]}x{image.shape[0]}",
                          "mask_pixels": int(mask.sum()), "mask_fraction": float(mask.mean()),
                          "fft_grid_centers": int(len(points)), "grid_centers_in_mask": int(center_masked.sum()),
                          "grid_center_mask_fraction": float(center_masked.mean()),
                          "grid_centers_within_2px_of_mask": int(near_masked.sum()),
                          "grid_neighborhood_mask_fraction": float(near_masked.mean()),
                          "fft_angle_deg": float(np.degrees(lattice.angle_rad))})

    with (args.output / "mask_validation_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in summaries for k in row)))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
