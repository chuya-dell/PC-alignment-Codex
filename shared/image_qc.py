"""Per-FOV image-quality checks that are not specific to pillar- or field-level analysis:
the bright write-field-boundary band mask (docs/POSITION6_IMAGE_FORENSICS_20260916.md) and
the saturation-based acquisition QC flag it motivated.

Both operate on a single raw 16-bit image and know nothing about pillar coordinates or
registration; they are meant to be reused from acquisition-time QC, pillar_level, and
field_level code alike.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d


def bright_band_mask(raw: np.ndarray, dn_threshold: int = 55000,
                      close_radius_px: int = 8, min_component_span_px: int = 200) -> np.ndarray:
    """Boolean mask, True where a pixel belongs to an elongated over-bright band.

    Thresholds the raw image at dn_threshold, closes small gaps so a line's pixels merge
    into one component, then keeps only components whose bounding box's longer side is at
    least min_component_span_px. This discriminates the write-field-boundary lines found at
    Position 6/7 (which span most of the frame) from isolated bright pillars or noise, and
    needs no per-position hand-tuned coordinates: it was verified against all of Positions
    1-8 on 260826-p50-sam and 260828-p50-SAM (see docs/POSITION6_IMAGE_FORENSICS_20260916.md
    and docs/MASKED_ALIGNMENT_FINAL_DECISION_20260916.md).
    """
    arr = np.asarray(raw)
    binary = (arr >= dn_threshold).astype(np.uint8)
    if not binary.any():
        return np.zeros(arr.shape, bool)
    k = 2 * close_radius_px + 1
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    mask = np.zeros(arr.shape, bool)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if max(w, h) >= min_component_span_px:
            mask |= (labels == label)
    if mask.any():
        dilate_k = 2 * close_radius_px + 1
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((dilate_k, dilate_k), np.uint8)) > 0
    return mask


def _cross_scratch_protection(raw: np.ndarray, *, half_width_px: int = 12) -> np.ndarray:
    """Protect long, thin cross-scratch tracks, including tracks with gaps.

    Profiles are evaluated in overlapping image segments. A candidate line must recur
    at the same coordinate in multiple segments, so one local stain cannot become a
    full-frame protected stripe. Dark and bright line profiles are both retained.
    """
    image = np.asarray(raw, dtype=np.float32)
    height, width = image.shape
    smooth = cv2.GaussianBlur(image, (0, 0), 3.0)
    protected = np.zeros((height, width), dtype=bool)

    def line_positions(axis: int) -> np.ndarray:
        # axis=0 finds vertical lines by scanning x in y-segments; axis=1 vice versa.
        scan_length = height if axis == 0 else width
        segment_length = max(64, int(round(scan_length / 12)))
        starts = list(range(0, scan_length, segment_length))
        support = np.zeros(width if axis == 0 else height, dtype=np.int16)
        for start in starts:
            stop = min(scan_length, start + segment_length)
            if stop - start < segment_length * .45:
                continue
            if axis == 0:
                profile = np.mean(smooth[start:stop], axis=0)
            else:
                profile = np.mean(smooth[:, start:stop], axis=1)
            trend = gaussian_filter1d(profile, max(12.0, len(profile) * .10), mode="nearest")
            residual = profile - trend
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med))) * 1.4826
            image_scale = float(np.percentile(image, 99.5) - np.percentile(image, .5))
            threshold = max(5.0 * mad, .003 * image_scale, 8.0)
            candidate = np.abs(residual - med) >= threshold
            # A scratch is a local line, not a broad illumination trough.
            candidate = cv2.morphologyEx(candidate.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                                         np.ones((1, 5), np.uint8))[0].astype(bool)
            support += candidate.astype(np.int16)
        required = max(5, int(np.ceil(len(starts) * .50)))
        candidates = np.flatnonzero(support >= required)
        if not len(candidates):
            return np.empty((0, 2), dtype=int)
        groups = np.split(candidates, np.flatnonzero(np.diff(candidates) > 12) + 1)
        # Only the strongest few repeated tracks qualify as fiducial scratches;
        # diffuse projection noise must not protect broad swaths of the image.
        groups = sorted(groups, key=lambda g: int(np.sum(support[g])), reverse=True)[:4]
        tracks = []
        for group in groups:
            center = int(group[np.argmax(support[group])])
            radius = min(100, max(half_width_px, int(np.ceil((group[-1]-group[0])/2))+8))
            tracks.append((center, radius))
        return np.asarray(tracks, dtype=int)

    for x, radius in line_positions(0):
        protected[:, max(0, x-radius):min(width, x+radius+1)] = True
    for y, radius in line_positions(1):
        protected[max(0, y-radius):min(height, y+radius+1), :] = True
    return protected


def stain_artifact_mask(raw: np.ndarray, *, min_component_area_px: int = 60,
                        threshold_fraction: float = .015,
                        max_component_extent_px: int = 240,
                        dilation_px: int = 3,
                        scratch_protection_half_width_px: int = 12) -> np.ndarray:
    """Return a conservative boolean exclusion mask for isolated dirt/stain blobs.

    This mask is separate from ``bright_band_mask``. It detects local bright or dark
    blemishes at a scale larger than the pillar pitch, rejects small lattice texture and
    elongated boundary/scratch components, and protects recurrent cross-scratch lines.
    It is opt-in: callers should OR it with ``bright_band_mask`` when excluding pixels
    from feature extraction or downstream image analysis.
    """
    image = np.asarray(raw)
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError("stain_artifact_mask expects a 2D image at least 32 pixels wide and high")
    if not np.isfinite(image).all():
        raise ValueError("stain_artifact_mask does not accept NaN or infinite pixels")
    values = image.astype(np.float32)
    fine = cv2.GaussianBlur(values, (0, 0), 3.0)
    broad = cv2.GaussianBlur(values, (0, 0), 18.0)
    residual = fine - broad
    scale = float(np.percentile(values, 99.5) - np.percentile(values, .5))
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center))) * 1.4826
    threshold = max(7.0 * mad, threshold_fraction * scale, 1.0)
    binary = (np.abs(residual - center) >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = np.zeros(image.shape, dtype=np.uint8)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_component_area_px or max(width, height) > max_component_extent_px:
            continue
        aspect = max(width, height) / max(1, min(width, height))
        if aspect > 8.0:
            continue
        candidates[labels == label] = 1
    if dilation_px > 0 and candidates.any():
        k = 2 * dilation_px + 1
        candidates = cv2.dilate(candidates, np.ones((k, k), dtype=np.uint8))
    protected = _cross_scratch_protection(values, half_width_px=scratch_protection_half_width_px)
    return (candidates > 0) & ~protected


def saturation_qc(raw: np.ndarray, bright_dn_threshold: int = 55000, bright_flag_count: int = 18000,
                   saturated_dn: int = 65000, saturated_flag_count: int = 1500) -> dict:
    """Acquisition-time QC counters and flag for the write-field-boundary defect.

    Thresholds verified against all 8 positions, pre and post, on both 260826-p50-sam and
    260828-p50-SAM Sample 1 (see data/results/v7_masked_alignment_benchmark/
    bright_band_mask_qc_all_positions.csv). Across those 32 images the highest count among
    Positions 1-5/8 (never Position 6/7) was 16252 px >=55000 DN and 1412 saturated
    (260828 Position 3 post, itself sitting near an ordinary write-field seam); the lowest
    count among Position 6/7 images was 20544 px >=55000 DN and 1445 saturated (260828
    Position 7 pre). bright_flag_count=18000 sits cleanly between 16252 and 20544 and alone
    separates every tested image correctly; saturated_flag_count=1500 is kept as a secondary
    signal (OR'd with the bright-pixel count) even though its own margin at that boundary is
    narrow. A FOV is flagged if either counter crosses its threshold.
    """
    arr = np.asarray(raw)
    bright_count = int(np.sum(arr >= bright_dn_threshold))
    saturated_count = int(np.sum(arr >= saturated_dn))
    flagged = bright_count >= bright_flag_count or saturated_count >= saturated_flag_count
    return {
        "bright_pixel_count": bright_count, "bright_dn_threshold": bright_dn_threshold,
        "bright_flag_count": bright_flag_count,
        "saturated_pixel_count": saturated_count, "saturated_dn_threshold": saturated_dn,
        "saturated_flag_count": saturated_flag_count,
        "flagged": bool(flagged),
    }
