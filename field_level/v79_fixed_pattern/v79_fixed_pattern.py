"""v79 step F: camera / illumination fixed pattern (hypothesis 5).
(1) dark frames (260904_p50_暗時): fixed-pattern amplitude in the analysis contrast units vs the delta noise SD.
(2) camera-coordinate stack of raw pre images per date: low-frequency illumination pattern; correlation of each field's positive-density map
    (S5P, outside scar mask) with the date-median illumination map and with the leave-one-out mean positive map of the other fields of that date
    (camera coordinates) versus a control with the other-fields' map rolled by a random offset.
usage: python v79_fixed_pattern.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, cv2
from scipy.stats import spearmanr
import v72_readout as R
import field_control_common as M60
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V78 = ROOT / 'data/results/v78_candidate_path'; V72 = ROOT / 'data/results/v72_real_field_readout'
OUT = ROOT / 'data/results/v79_fixed_pattern'; OUT.mkdir(parents=True, exist_ok=True)


def blockmap(xy, v, block=64, stat='mean'):
    gx = np.clip((xy[:, 0] // block).astype(int), 0, 31); gy = np.clip((xy[:, 1] // block).astype(int), 0, 31); ok = np.isfinite(v)
    n = np.bincount((gy * 32 + gx)[ok], minlength=1024); s = np.bincount((gy * 32 + gx)[ok], weights=v[ok], minlength=1024)
    return np.where(n > 20, s / np.maximum(n, 1), np.nan).reshape(32, 32)


def dark():
    d = M60.folder('260904_p50_暗時'); out = []
    ims = [M60.read(f) for f in sorted(d.glob('*.tif'))]
    for i, im in enumerate(ims):
        f = im / 65535; c = R.box_contrast(im)
        out.append(dict(frame=i + 1, mean_counts=float(im.mean()), pixel_sd_counts=float(im.std()), box_contrast_sd=float(c.std()),
                        lowfreq_sd_counts=float(cv2.GaussianBlur(im, (0, 0), 20).std())))
    if len(ims) >= 2:
        diff = ims[0] - ims[1]; avg = (ims[0] + ims[1]) / 2
        out.append(dict(frame='diff12', pixel_sd_counts=float(diff.std() / np.sqrt(2)), note='temporal noise per frame'))
        out.append(dict(frame='mean12', pixel_sd_counts=float(avg.std()), box_contrast_sd=float(R.box_contrast(avg).std()), note='fixed pattern + residual temporal noise/sqrt2'))
        fp_var = avg.var() - (diff.std() ** 2) / 4
        out.append(dict(frame='fixed_pattern_estimate', pixel_sd_counts=float(np.sqrt(max(fp_var, 0))), note='sqrt(var(mean) - var(diff)/4)'))
    pd.DataFrame(out).to_csv(OUT / 'dark_frames.csv', index=False); return pd.DataFrame(out)


def main():
    print(dark().round(4).to_string())
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    th = pd.read_csv(V72 / 'thresholds_v2.csv', dtype={'date': str})
    rows = []
    for date, g in led.groupby('date'):
        maps = {}; illum = []
        for r in g.itertuples():
            f = V78 / 'fields' / f'{r.fid}.npz'
            if not f.exists(): continue
            z = np.load(f); d = z['S5P'].astype(float); ok = np.isfinite(d) & (z['sd_periods'] > 2)
            med = np.median(d[ok]); mad = 1.4826 * np.median(abs(d[ok] - med)); pos = np.where(ok, (d > med + 4 * mad).astype(float), np.nan)
            bm = blockmap(z['ctr'].astype(float), pos)
            import os
            if os.environ.get('INNER') == '1': bm[:3, :] = np.nan; bm[-3:, :] = np.nan; bm[:, :3] = np.nan; bm[:, -3:] = np.nan
            maps[r.fid] = bm
            im = R.L.read(r.pre_path); illum.append(cv2.resize(cv2.GaussianBlur(im, (0, 0), 16), (32, 32), interpolation=cv2.INTER_AREA))
        if len(maps) < 4: continue
        im_med = np.median(np.array(illum), axis=0); stack = np.array(list(maps.values())); keys = list(maps)
        rng = np.random.default_rng(1)
        for i, k in enumerate(keys):
            others = np.nanmean(np.delete(stack, i, axis=0), axis=0); m = maps[k]
            g_ = np.isfinite(m) & np.isfinite(others)
            if g_.sum() < 100 or np.nanstd(m[g_]) == 0 or np.nanstd(others[g_]) == 0: continue
            rho = spearmanr(m[g_], others[g_]).statistic
            ctrl = []
            for _ in range(20):
                sh = np.roll(np.roll(others, rng.integers(5, 28), 0), rng.integers(5, 28), 1); gg = np.isfinite(m) & np.isfinite(sh)
                ctrl.append(spearmanr(m[gg], sh[gg]).statistic)
            gi = np.isfinite(m) & np.isfinite(im_med); rho_ill = spearmanr(m[gi], im_med[gi]).statistic if np.nanstd(m[gi]) > 0 else np.nan
            rows.append(dict(date=date, fid=k, rho_loo_camera=rho, rho_control_median=float(np.median(ctrl)), rho_control_p95=float(np.percentile(ctrl, 95)), rho_illum=rho_ill))
        print('date', date, len(keys), flush=True)
    t = pd.DataFrame(rows); import os
    t.to_csv(OUT / ('camera_pattern_correlation_inner.csv' if os.environ.get('INNER') == '1' else 'camera_pattern_correlation.csv'), index=False)
    print(t.groupby('date')[['rho_loo_camera', 'rho_control_median', 'rho_illum']].median().round(3))
    print('overall median rho(loo camera)', t.rho_loo_camera.median(), 'control median', t.rho_control_median.median(), 'fraction above control p95:', float((t.rho_loo_camera > t.rho_control_p95).mean()))
    print('illumination-map correlation median', t.rho_illum.median())


if __name__ == '__main__':
    main()
