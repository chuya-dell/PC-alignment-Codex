"""Verify whether v7's masked-benchmark recovery rate at Position 6 depends on excluding
mask-interior spikes from scoring (instructor's question: docs/MASKED_SCORING_VERIFICATION_20260916.md).

field_run_masked_benchmark.py's detect_and_score() drops truth spikes that fall inside the
write-field-boundary mask from the evaluated denominator (score_mask), rather than counting an
undetected mask-interior spike as a recovery failure. Spike placement itself
(field_run_masked_benchmark.sample_truth) does not avoid the mask -- spikes land inside it by
chance, same as anywhere else in the frame.

This script re-uses the exact same registration estimates (masked ORB/ECC, same seeds, same
ground truth per scenario) and re-scores each trial two ways:
  - "region_restricted": mask-interior truth points dropped from the denominator (matches the
    method_summary_*.csv numbers already reported).
  - "full_frame": every truth point is scored, including any inside the mask, against
    detections computed from the FULL (unmasked) difference image -- i.e. what recovery would
    have been if nothing were excluded from scoring, only from the registration input.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import field_run_masked_benchmark as v7
from shared import registration as reg
from shared.image_qc import bright_band_mask


def score_full_frame(pre01, mov01, estimated, truth, radius):
    aligned = cv2.warpAffine(mov01, estimated, (pre01.shape[1], pre01.shape[0]),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    d = ndimage.gaussian_filter(aligned - pre01, 1)
    peak = (d == ndimage.maximum_filter(d, 5)) & (d > .018)
    peak[:10] = False; peak[-10:] = False; peak[:, :10] = False; peak[:, -10:] = False
    y, x = np.where(peak); found = np.c_[x, y].astype(float)
    tree = KDTree(found) if len(found) else None
    matched = np.zeros(len(truth), bool)
    if tree: matched = tree.query(truth)[0] <= radius
    truth_tree = KDTree(truth)
    fp = int(np.sum(truth_tree.query(found)[0] > radius)) if len(found) else 0
    return int(matched.sum()), len(truth), len(found), fp


def main():
    manifest = ROOT / 'data/raw/semisynthetic_alignment_benchmark_manifest.json'
    c = json.loads(manifest.read_text(encoding='utf-8'))
    output = ROOT / 'data/results/v7_masked_alignment_benchmark'
    output.mkdir(parents=True, exist_ok=True)

    pos = 6
    rows = []
    mask_frac = None
    for scenario in v7.SCENARIOS:
        gt, src_fov = v7.gt_for(scenario, pos)
        raw = reg.load_image_unicode(str(Path(c['root']) / f"{c['sample']}-{pos}-0.tif"))
        pre01 = reg.image01_for_registration(raw)
        mask = bright_band_mask(raw)
        mask_frac = float(mask.mean())
        render_warp = cv2.invertAffineTransform(gt)
        base = v7.make_moving(pre01, render_warp)

        for method_name, fn in v7.METHODS.items():
            for masked in [False, True]:
                excl = mask if masked else None
                try:
                    estimated = fn(pre01, base, exclude_mask=excl)
                    ok = True
                except Exception:
                    estimated = np.eye(2, 3, dtype=np.float32)
                    ok = False
                for count in c['spike_counts']:
                    truth = v7.sample_truth(pre01.shape, count, np.random.default_rng(c['seed'] + pos * 1000 + count))
                    inside_mask = mask[np.clip(truth[:, 1].round().astype(int), 0, mask.shape[0] - 1),
                                        np.clip(truth[:, 0].round().astype(int), 0, mask.shape[1] - 1)]
                    mov = v7.inject(base, truth, gt, c['spike_amplitude_normalized'], c['spike_sigma_px'])

                    rec_r, n_r, det_r, fp_r = v7.detect_and_score(pre01, mov, estimated, truth,
                                                                    c['match_radius_px'], score_mask=mask)
                    rec_f, n_f, det_f, fp_f = score_full_frame(pre01, mov, estimated, truth, c['match_radius_px'])

                    rows.append({
                        'method': method_name, 'masked': masked, 'converged': ok, 'scenario': scenario,
                        'spike_count': count, 'spikes_inside_mask': int(inside_mask.sum()),
                        'region_restricted_evaluated_n': n_r, 'region_restricted_recovered': rec_r,
                        'region_restricted_recovery_rate': rec_r / max(n_r, 1),
                        'region_restricted_fp_rate': fp_r / max(det_r, 1),
                        'full_frame_evaluated_n': n_f, 'full_frame_recovered': rec_f,
                        'full_frame_recovery_rate': rec_f / max(n_f, 1),
                        'full_frame_fp_rate': fp_f / max(det_f, 1),
                    })
        print(f'scenario {scenario}: done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output / 'masked_scoring_verification_position6.csv', index=False)
    print(f'mask fraction of frame area: {mask_frac:.4f}')
    summary = df.groupby(['method', 'masked', 'scenario'], as_index=False).agg(
        spikes_inside_mask=('spikes_inside_mask', 'sum'),
        region_restricted_recovery_rate=('region_restricted_recovery_rate', 'mean'),
        full_frame_recovery_rate=('full_frame_recovery_rate', 'mean'),
        region_restricted_fp_rate=('region_restricted_fp_rate', 'mean'),
        full_frame_fp_rate=('full_frame_fp_rate', 'mean'),
    )
    summary.to_csv(output / 'masked_scoring_verification_summary.csv', index=False)
    pd.set_option('display.width', 220)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
