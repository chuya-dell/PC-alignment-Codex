"""v78 criterion 3: band power relative to a shuffled-label null (same number of positives, no spatial structure)."""
import sys, glob, os
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v78_candidate_path', 'field_level/v60_band_scar_controls', 'field_level/v72_real_field_readout'): sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd
import v78_calibrate as K, field_control_common as M
V70, V72, V78 = K.V70, K.V72, K.V78
rng = np.random.default_rng(20261008)
def ratio(xy, d, k=3, nsh=10):
    ok = np.isfinite(d); v = d[ok]; med = np.median(v); mad = 1.4826 * np.median(abs(v - med)); t = med + k * mad
    m = M.band_metrics(xy, d, t, False); pos = ok & (d > t); n = int(pos.sum())
    if n < 20: return np.nan, n, m['band_power']
    sh = []
    idx = np.flatnonzero(ok)
    for _ in range(nsh):
        dd = np.full_like(d, np.nan); lab = np.zeros(len(d), bool); lab[rng.choice(idx, n, replace=False)] = True
        dd[ok] = np.where(lab[ok], t + 1.0, t - 1.0); sh.append(M.band_metrics(xy, dd, t, False)['band_power'])
    return m['band_power'] / np.median(sh), n, m['band_power']
P = ROOT / 'data/results/v75_zero_truth_pairs/'
pr = []
for f in sorted(glob.glob(str(P / 'synth/*.npz'))) + [str(P / 'real/burst_1_2.npz')]:
    z = np.load(f); r, n, b = ratio(z['ctr_xy'].astype(float), z['S5'].astype(float)); pr.append(dict(pair=os.path.basename(f)[:-4], ratio=r, n=n, bp=b))
pr = pd.DataFrame(pr); pr.to_csv(V78 / 'band_ratio_pairs.csv', index=False); print(pr.round(2).describe().loc[['count', 'mean', '50%', 'max']])
led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}); rows = []
for r in led.itertuples():
    z72 = np.load(V72 / 'fields' / f'{r.fid}.npz'); z78 = np.load(V78 / 'fields' / f'{r.fid}.npz')
    d = np.where(z78['sd_periods'] > 2, z72['S5'].astype(float), np.nan); rr, n, b = ratio(z72['ctr_xy'].astype(float), d)
    f0 = K.load_field(r.fid); d0 = np.where(f0['S0'][1], f0['S0'][0], np.nan); r0, n0, b0 = ratio(z72['std_xy'].astype(float), d0)
    rows.append(dict(fid=r.fid, group=r.group, fixed29=r.fixed29, ratio_S5=rr, n_S5=n, ratio_S0=r0, n_S0=n0))
t = pd.DataFrame(rows); t.to_csv(V78 / 'band_ratio_fields.csv', index=False)
tol = float(pr.ratio.max()); print('tolerance (max ratio over pairs):', round(tol, 2), 'p95', round(pr.ratio.quantile(.95), 2))
for nm, sel in [('all632', t.fid.notna()), ('blank', t.group == 'blank'), ('fixed29', t.fixed29)]:
    g = t[sel]; print(nm, 'S5 ratio median', round(g.ratio_S5.median(), 2), 'within tol', round((g.ratio_S5 <= tol).mean(), 3), '| S0 (scar-excluded) ratio median', round(g.ratio_S0.median(), 2), 'within tol', round((g.ratio_S0 <= tol).mean(), 3))
