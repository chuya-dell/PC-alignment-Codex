"""v83 part A-2: dark frames of 2026-10-08 (261008_ligjt: light path switched to eyepiece) vs 260904_p50_暗時 (full shading, 2 ms).
Bias (mean/median), read noise (SD of the difference of two frames taken ~1 s apart / sqrt2), fixed-pattern (SD of the 2-frame mean after removing temporal noise),
change between 15:18 and 15:55 (mean difference, SD of session-mean image difference vs the expectation from temporal noise), hot/dead pixels, row/column structure.
usage: python v83_dark.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, tifffile, pandas as pd
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu')
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'data/results/v83_new_shots_261008'; OUT.mkdir(parents=True, exist_ok=True)
rd = lambda p: tifffile.imread(p).astype(np.float64)
S = {'1018': [RAW/'261008_ligjt'/'1-1.tif', RAW/'261008_ligjt'/'1-2.tif'], '1018b': None}
sess = {'t1518': [RAW/'261008_ligjt'/'1-1.tif', RAW/'261008_ligjt'/'1-2.tif'],
        't1555': [RAW/'261008_ligjt'/'2-1.tif', RAW/'261008_ligjt'/'2-2.tif'],
        'd0904': sorted((RAW/'260904_p50_暗時').glob('*.tif'))}
rows = []; means = {}
for k, ps in sess.items():
    a, b = rd(ps[0]), rd(ps[1]); m = (a + b) / 2; means[k] = m
    d = a - b; rn = d.std() / np.sqrt(2); rn_rob = 1.4826 * np.median(abs(d - np.median(d))) / np.sqrt(2)
    # fixed pattern: var(frame) = var_fp + var_temporal  => var_fp = cov(a,b)
    cov = np.mean((a - a.mean()) * (b - b.mean())); fp = np.sqrt(max(cov, 0))
    rows.append(dict(session=k, dtype=str(tifffile.imread(ps[0]).dtype), shape=str(a.shape), mean=m.mean(), median=np.median(m), sd_frame=a.std(), read_noise_sd=rn, read_noise_robust=rn_rob,
                     fixed_pattern_sd=fp, frame_mean_a=a.mean(), frame_mean_b=b.mean(), p1=np.percentile(a, 1), p99=np.percentile(a, 99), p99_9=np.percentile(a, 99.9), max=a.max(), min=a.min(),
                     frac_zero=float((a == 0).mean()), row_mean_sd=np.std(m.mean(1)), col_mean_sd=np.std(m.mean(0)), n_hot_gt5sd_mean=int((m > np.median(m) + 5 * m.std()).sum())))
t = pd.DataFrame(rows); t.to_csv(OUT / 'dark_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); print(t.round(4).T.to_string())
# change between sessions
r1, r2 = means['t1518'], means['t1555']; diff = r2 - r1
rn = t.set_index('session').read_noise_sd
exp_sd = np.sqrt(rn['t1518'] ** 2 / 2 + rn['t1555'] ** 2 / 2)   # expected SD of difference of two 2-frame means from temporal noise alone
res = dict(mean_shift=float(diff.mean()), sd_diff=float(diff.std()), expected_sd_temporal_only=float(exp_sd), ratio=float(diff.std() / exp_sd),
           rowmean_shift_sd=float(np.std(diff.mean(1))), colmean_shift_sd=float(np.std(diff.mean(0))),
           corr_fp_1518_1555=float(np.corrcoef((r1 - r1.mean()).ravel(), (r2 - r2.mean()).ravel())[0, 1]))
# within-session pair difference (1 s apart) SD check: noise independence
a1, b1 = rd(sess['t1518'][0]), rd(sess['t1518'][1]); a2, b2 = rd(sess['t1555'][0]), rd(sess['t1555'][1])
res['frame_mean_drift_1518_to_1555'] = float((a2.mean() + b2.mean() - a1.mean() - b1.mean()) / 2)
# statistical error of the means: SE of frame mean ~ read noise / sqrt(N pix) (negligible); drift in units of read noise
res['mean_shift_in_read_noise_units'] = res['mean_shift'] / float(rn['t1518'])
# detail: does the difference show spatial structure at low frequency (block means)?
bm = diff[:2040, :2048].reshape(2040 // 60, 60, 2048 // 64, 64).mean((1, 3)); res['block60x64_mean_sd'] = float(bm.std()); res['block_expected_sd'] = float(exp_sd / np.sqrt(60 * 64))
# stdev vs mean exposure: 2 ms (0904) vs unknown (1008). Compare read noise only.
print(json.dumps(res, indent=1)); json.dump(res, open(OUT / 'dark_change.json', 'w'), indent=1)
# Analysis contrast units: the pipeline uses im/65535 -> noise in contrast units
print('read noise in /65535 units: 1518 %.3e  1555 %.3e  0904 %.3e' % tuple(rn[k] / 65535 for k in ('t1518', 't1555', 'd0904')))
