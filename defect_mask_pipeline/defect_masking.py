"""Human-gated pixel defect masks in native pre-image coordinates.

Automatic processing in this module creates *review candidates only*.  The
analysis-facing loader accepts a mask only after both images were reviewed and
the candidate was explicitly approved with reviewer and date metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


PITCH_PX = 7.286
REQUIRED_APPROVAL_COLUMNS = {
    "candidate_id", "date", "sample_id", "position_id", "source",
    "decision", "pre_reviewed", "post_reviewed", "pre_defect_seen",
    "post_defect_seen", "mask_polygon_pre_json", "approved_at",
    "approved_by",
}


@dataclass(frozen=True)
class CandidateSettings:
    pitch_px: float = PITCH_PX
    residual_z: float = 6.0
    min_area_px: int = 212
    min_span_px: float = 14.572
    reciprocal_bandwidth: float = 0.012
    low_frequency_radius: float = 0.025
    smoothing_sigma_px: float = 2.5
    close_radius_px: int = 5


def _robust_standardize(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, np.float32)
    med = float(np.median(arr))
    scale = float(np.median(np.abs(arr - med))) * 1.4826
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Image has no finite robust intensity scale.")
    return (arr - med) / scale


def periodic_reconstruction(image: np.ndarray, settings: CandidateSettings) -> np.ndarray:
    """Reconstruct shading plus the dominant reciprocal-lattice components."""
    arr = _robust_standardize(image)
    height, width = arr.shape
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    fy = np.fft.fftshift(np.fft.fftfreq(height))
    fx = np.fft.fftshift(np.fft.fftfreq(width))
    xx, yy = np.meshgrid(fx, fy)
    radius = np.hypot(xx, yy)
    lattice_frequency = 1.0 / settings.pitch_px
    annulus = (radius >= lattice_frequency * 0.72) & (radius <= lattice_frequency * 1.28)
    magnitude = np.log1p(np.abs(spectrum))
    local_max = magnitude == cv2.dilate(magnitude.astype(np.float32), np.ones((9, 9), np.uint8))
    peak_y, peak_x = np.where(annulus & local_max)
    if len(peak_x) == 0:
        raise ValueError("No reciprocal-lattice peaks were found.")
    order = np.argsort(magnitude[peak_y, peak_x])[::-1][:18]
    mask = radius <= settings.low_frequency_radius
    for row, col in zip(peak_y[order], peak_x[order]):
        px, py = fx[col], fy[row]
        mask |= np.hypot(xx - px, yy - py) <= settings.reciprocal_bandwidth
        mask |= np.hypot(xx + px, yy + py) <= settings.reciprocal_bandwidth
    reconstructed = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    return reconstructed.astype(np.float32)


def residual_zscore(
    image: np.ndarray, settings: CandidateSettings = CandidateSettings()
) -> np.ndarray:
    """Return the robust periodic-residual score used only for candidate review."""
    normalized = _robust_standardize(image)
    residual = normalized - periodic_reconstruction(image, settings)
    score = cv2.GaussianBlur(np.abs(residual), (0, 0), settings.smoothing_sigma_px)
    median = float(np.median(score))
    mad = float(np.median(np.abs(score - median))) * 1.4826
    if mad <= 0 or not np.isfinite(mad):
        raise ValueError("Residual image has no finite robust scale.")
    return (score - median) / mad


def candidate_regions_from_zscore(
    zscore: np.ndarray, settings: CandidateSettings = CandidateSettings()
) -> list[dict]:
    """Group a score image into multi-pixel connected review candidates."""
    binary = (zscore >= settings.residual_z).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = 2 * settings.close_radius_px + 1
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: list[dict] = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < settings.min_area_px or max(width, height) < settings.min_span_px:
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
        epsilon = max(1.0, 0.005 * cv2.arcLength(contour.astype(np.float32), True))
        contour = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True)[:, 0, :]
        regions.append({
            "bbox_x": int(x), "bbox_y": int(y), "bbox_w": int(width),
            "bbox_h": int(height), "area_px": int(area),
            "max_residual_z": float(np.max(zscore[labels == label])),
            "polygon_native_json": json.dumps(contour.astype(float).tolist(), separators=(",", ":")),
        })
    return regions


def residual_candidate_regions(
    image: np.ndarray, settings: CandidateSettings = CandidateSettings()
) -> tuple[list[dict], np.ndarray]:
    """Return connected residual candidates; never return an approved mask."""
    zscore = residual_zscore(image, settings)
    return candidate_regions_from_zscore(zscore, settings), zscore


def transform_polygon(polygon: np.ndarray, transform: dict, direction: str) -> np.ndarray:
    """Move a polygon through registration without changing its coordinate authority."""
    import registration

    points = np.asarray(polygon, float)
    if direction == "pre_to_post":
        return registration.inverse_transform_points(points, transform)
    if direction == "post_to_pre":
        return registration.transform_points(points, transform)
    raise ValueError("direction must be pre_to_post or post_to_pre.")


def rasterize_polygons(shape: tuple[int, int], polygons: Iterable[np.ndarray]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        points = np.rint(np.asarray(polygon, float)).astype(np.int32)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError("Every polygon must contain at least three [x,y] points.")
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _strict_bool(value, column: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{column} must be TRUE or FALSE.")


def load_approved_polygons(
    review_csv: str | Path, date: int | str, sample_id: int | str, position_id: int | str
) -> list[np.ndarray]:
    """Load only fully reviewed, explicitly approved masks for one image pair."""
    frame = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    missing = REQUIRED_APPROVAL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Approval table is missing columns: {sorted(missing)}")
    selected = frame[
        (frame["date"].astype(str) == str(date))
        & (frame["sample_id"].astype(str) == str(sample_id))
        & (frame["position_id"].astype(str) == str(position_id))
    ]
    polygons: list[np.ndarray] = []
    for _, row in selected.iterrows():
        decision = row["decision"].strip().lower()
        if decision not in {"", "approved", "rejected"}:
            raise ValueError(f"{row['candidate_id']}: decision must be approved or rejected.")
        if decision != "approved":
            continue
        if not (_strict_bool(row["pre_reviewed"], "pre_reviewed")
                and _strict_bool(row["post_reviewed"], "post_reviewed")):
            raise ValueError(f"{row['candidate_id']}: both images must be reviewed before approval.")
        pre_seen = _strict_bool(row["pre_defect_seen"], "pre_defect_seen")
        post_seen = _strict_bool(row["post_defect_seen"], "post_defect_seen")
        if not (pre_seen or post_seen):
            raise ValueError(f"{row['candidate_id']}: approved row has no defect in either image.")
        if not row["approved_at"].strip() or not row["approved_by"].strip():
            raise ValueError(f"{row['candidate_id']}: approval date and approver are required.")
        try:
            polygon = np.asarray(json.loads(row["mask_polygon_pre_json"]), float)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{row['candidate_id']}: invalid pre-image polygon JSON.") from exc
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError(f"{row['candidate_id']}: polygon needs at least three [x,y] points.")
        polygons.append(polygon)
    return polygons


def masks_for_pair(
    image_shape: tuple[int, int], polygons_pre: Iterable[np.ndarray], transform: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the same approved pre-coordinate masks in both images."""
    polygons_pre = list(polygons_pre)
    pre_mask = rasterize_polygons(image_shape, polygons_pre)
    post_polygons = [transform_polygon(p, transform, "pre_to_post") for p in polygons_pre]
    post_mask = rasterize_polygons(image_shape, post_polygons)
    return pre_mask, post_mask


def excluded_grid_fraction(
    pre_coordinates: np.ndarray,
    post_coordinates: np.ndarray,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Measure the union of masked 3x3 sampling footprints over the grid."""
    import registration

    dummy = np.zeros(pre_mask.shape, np.float32)
    pre_valid = registration.sample_contrast(dummy, pre_coordinates, invalid_mask=pre_mask).valid_sampling.to_numpy()
    post_valid = registration.sample_contrast(dummy, post_coordinates, invalid_mask=post_mask).valid_sampling.to_numpy()
    eligible = registration.sample_contrast(dummy, pre_coordinates).valid_sampling.to_numpy()
    eligible &= registration.sample_contrast(dummy, post_coordinates).valid_sampling.to_numpy()
    excluded = eligible & ~(pre_valid & post_valid)
    denominator = int(eligible.sum())
    return (float(excluded.sum() / denominator) if denominator else 1.0), excluded


def write_audit_log(review_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    """Validate and export the immutable analysis-facing approval evidence."""
    frame = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    missing = REQUIRED_APPROVAL_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Approval table is missing columns: {sorted(missing)}")
    approved = frame[frame["decision"].str.strip().str.lower() == "approved"].copy()
    for _, row in approved.iterrows():
        load_approved_polygons(review_csv, row["date"], row["sample_id"], row["position_id"])
    columns = [
        "candidate_id", "date", "sample_id", "position_id", "source",
        "pre_reviewed", "post_reviewed", "pre_defect_seen", "post_defect_seen",
        "mask_polygon_pre_json", "approved_at", "approved_by",
    ]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    approved[columns].to_csv(output_csv, index=False)
    return approved[columns]
