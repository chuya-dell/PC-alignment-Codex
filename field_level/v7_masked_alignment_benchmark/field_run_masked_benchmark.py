"""Re-run the semi-synthetic real-range and axis-sweep benchmarks
(field_level/v5_semisynthetic_alignment_benchmark/*_realrange.py and *_axis_sweep.py) with the
Position 6/7 write-field-boundary band (docs/POSITION6_IMAGE_FORENSICS_20260916.md) masked out,
using the promoted Phase 2 registration functions in shared.registration
(estimate_affine_ecc, estimate_affine_orb_ransac) and the mask from shared.image_qc.

For each of four scenarios (combined real-range translation-dominant transform, and isolated
ty/theta/scale sweeps -- same ground-truth tables as the v5 follow-ups), both methods are run
twice per position: once with exclude_mask=None (unmasked, matching the earlier v5 numbers for a
direct before/after comparison) and once with exclude_mask=bright_band_mask(pre_raw) (masked).
Dense Farneback is out of scope (already rejected; not part of this comparison).
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import KDTree

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import registration as reg
from shared.image_qc import bright_band_mask

# Real per-FOV (dx, dy, theta_deg, delta_scale_pct), from the 2026-09-01 record-reconstruction
# comparison (docs/ECC_20260901_REFERENCE_VALIDATION_20260916.md), same source used throughout
# the v5 real-range/axis-sweep follow-ups.
REAL_VALUES = {
    '1-1': dict(tx=-4.896, ty=-8.061, theta_deg=-0.19828, delta_scale_pct=0.45361),
    '1-2': dict(tx=-7.058, ty=3.464, theta_deg=-0.02826, delta_scale_pct=-0.47135),
    '1-3': dict(tx=4.493, ty=20.157, theta_deg=0.00097, delta_scale_pct=0.02058),
    '1-4': dict(tx=4.688, ty=4.426, theta_deg=0.00908, delta_scale_pct=-0.02779),
    '1-5': dict(tx=3.807, ty=9.651, theta_deg=0.01729, delta_scale_pct=-0.01115),
    '1-7': dict(tx=9.502, ty=13.604, theta_deg=0.00245, delta_scale_pct=0.01096),
    '1-8': dict(tx=-26.756, ty=1.188, theta_deg=0.00627, delta_scale_pct=-0.09860),
}
# scenario -> {benchmark position: source FOV}. 'realrange' uses the full (tx,ty,theta,scale)
# tuple (matches field_run_semisynthetic_alignment_benchmark_realrange.py); the axis scenarios
# zero out every field but the swept one (matches field_run_..._axis_sweep.py).
SCENARIOS = {
    'realrange': {1: '1-1', 2: '1-2', 5: '1-5', 6: '1-8', 8: '1-3'},
    'ty': {1: '1-1', 2: '1-2', 5: '1-5', 6: '1-3', 8: '1-8'},
    'theta_deg': {1: '1-2', 2: '1-3', 5: '1-5', 6: '1-1', 8: '1-8'},
    'delta_scale_pct': {1: '1-3', 2: '1-4', 5: '1-5', 6: '1-2', 8: '1-8'},
}
AXIS_KEYS = ['tx', 'ty', 'theta_deg', 'delta_scale_pct']
METHODS = {'ecc_affine_pyramid': reg.estimate_affine_ecc, 'orb_ransac_affine': reg.estimate_affine_orb_ransac}


def affine_metrics(m):
    a = m[:, :2]; s = math.sqrt(abs(np.linalg.det(a)))
    return dict(tx_px=float(m[0, 2]), ty_px=float(m[1, 2]),
                theta_deg=float(math.degrees(math.atan2(a[1, 0], a[0, 0]))),
                delta_scale_pct=float((s - 1) * 100))


def gt_for(scenario, pos):
    src = SCENARIOS[scenario][pos]
    full = REAL_VALUES[src]
    if scenario == 'realrange':
        e = dict(full)
    else:
        e = {k: (full[k] if k == scenario else 0.0) for k in AXIS_KEYS}
    theta = math.radians(e['theta_deg']); scale = 1 + e['delta_scale_pct'] / 100.0
    A = np.array([[scale * math.cos(theta), -scale * math.sin(theta)],
                  [scale * math.sin(theta), scale * math.cos(theta)]])
    t = np.array([e['tx'], e['ty']])
    return np.c_[A, t].astype(np.float32), src


def make_moving(pre01, render_warp):
    return cv2.warpAffine(pre01, render_warp, (pre01.shape[1], pre01.shape[0]),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def inject(moving, truth_template, gt, amp, sigma):
    out = moving.copy(); h, w = out.shape
    xy = cv2.transform(truth_template[None].astype(np.float32), gt)[0]
    r = max(3, int(math.ceil(4 * sigma))); yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = amp * np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    for x, y in xy:
        xi, yi = round(float(x)), round(float(y))
        x0, x1 = max(0, xi - r), min(w, xi + r + 1); y0, y1 = max(0, yi - r), min(h, yi + r + 1)
        out[y0:y1, x0:x1] += kernel[y0 - (yi - r):y1 - (yi - r), x0 - (xi - r):x1 - (xi - r)]
    return np.clip(out, 0, 1)


def sample_truth(shape, n, rng):
    h, w = shape; pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(60, w - 60), rng.uniform(60, h - 60)])
        if not pts or np.min(np.linalg.norm(np.asarray(pts) - p, axis=1)) > 16:
            pts.append(p)
    return np.asarray(pts, np.float32)


def detect_and_score(pre01, mov01, estimated, truth, radius, score_mask=None):
    """score_mask (True=exclude), if given, drops truth points and detections inside it so the
    boundary band's own bright pixels (which are not spikes) cannot masquerade as false
    positives regardless of which method aligned the image."""
    aligned = cv2.warpAffine(mov01, estimated, (pre01.shape[1], pre01.shape[0]),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    d = ndimage.gaussian_filter(aligned - pre01, 1)
    peak = (d == ndimage.maximum_filter(d, 5)) & (d > .018)
    peak[:10] = False; peak[-10:] = False; peak[:, :10] = False; peak[:, -10:] = False
    if score_mask is not None:
        peak &= ~score_mask
    y, x = np.where(peak); found = np.c_[x, y].astype(float)
    if score_mask is not None:
        keep = ~score_mask[np.clip(truth[:, 1].round().astype(int), 0, score_mask.shape[0] - 1),
                            np.clip(truth[:, 0].round().astype(int), 0, score_mask.shape[1] - 1)]
        eval_truth = truth[keep]
    else:
        eval_truth = truth
    tree = KDTree(found) if len(found) else None
    matched = np.zeros(len(eval_truth), bool)
    if tree and len(eval_truth): matched = tree.query(eval_truth)[0] <= radius
    truth_tree = KDTree(eval_truth) if len(eval_truth) else None
    fp = int(np.sum(truth_tree.query(found)[0] > radius)) if (len(found) and truth_tree) else len(found)
    return int(matched.sum()), len(eval_truth), len(found), fp


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', required=True, choices=list(SCENARIOS))
    p.add_argument('--manifest', type=Path, default=ROOT/'data/raw/semisynthetic_alignment_benchmark_manifest.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    c = json.loads(a.manifest.read_text(encoding='utf-8'))
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []; reg_rows = []
    for pos in sorted(SCENARIOS[a.scenario]):
        gt, src_fov = gt_for(a.scenario, pos)
        print(f'Position {pos} ({a.scenario}, src {src_fov}): loading/aligning', flush=True)
        raw = reg.load_image_unicode(str(Path(c['root'])/f"{c['sample']}-{pos}-0.tif"))
        pre01 = reg.image01_for_registration(raw)
        mask = bright_band_mask(raw)
        render_warp = cv2.invertAffineTransform(gt)
        base = make_moving(pre01, render_warp)
        for masked in [False, True]:
            excl = mask if masked else None
            estimates = {}
            for name, fn in METHODS.items():
                try:
                    estimates[name] = fn(pre01, base, exclude_mask=excl); ok = True; err = ''
                except Exception as e:
                    estimates[name] = np.eye(2, 3, dtype=np.float32); ok = False; err = str(e)
                gm, em = affine_metrics(gt), affine_metrics(estimates[name])
                reg_rows.append({'position': pos, 'method': name, 'scenario': a.scenario, 'masked': masked,
                                  'source_fov': src_fov, 'converged': ok, 'error': err,
                                  **{f'ground_truth_{k}': v for k, v in gm.items()},
                                  **{f'estimated_{k}': v for k, v in em.items()},
                                  **{f'error_{k}': em[k] - gm[k] for k in gm}})
            for count in c['spike_counts']:
                truth = sample_truth(pre01.shape, count, np.random.default_rng(c['seed'] + pos * 1000 + count))
                mov = inject(base, truth, gt, c['spike_amplitude_normalized'], c['spike_sigma_px'])
                for name, m in estimates.items():
                    recovered, n_truth, total, fp = detect_and_score(pre01, mov, m, truth, c['match_radius_px'],
                                                                       score_mask=mask)
                    rows.append({'position': pos, 'method': name, 'scenario': a.scenario, 'masked': masked,
                                 'spike_count': count, 'evaluated_truth_count': n_truth,
                                 'recovered_count': recovered, 'recovery_rate': recovered / max(n_truth, 1),
                                 'detected_count': total, 'false_positive_count': fp,
                                 'false_positive_rate': fp / max(total, 1)})
        print(f'Position {pos}: done', flush=True)
    trial = pd.DataFrame(rows); registration = pd.DataFrame(reg_rows)
    trial.to_csv(a.output/f'trial_results_{a.scenario}.csv', index=False)
    registration.to_csv(a.output/f'registration_errors_{a.scenario}.csv', index=False)
    summary = (trial.groupby(['method', 'masked'], as_index=False)
               .agg(mean_recovery_rate=('recovery_rate', 'mean'), mean_false_positive_rate=('false_positive_rate', 'mean')))
    f6 = (trial[trial.position == 6].groupby(['method', 'masked'], as_index=False)
          .agg(fov6_recovery_rate=('recovery_rate', 'mean'), fov6_false_positive_rate=('false_positive_rate', 'mean')))
    summary = summary.merge(f6, on=['method', 'masked'])
    summary.to_csv(a.output/f'method_summary_{a.scenario}.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
