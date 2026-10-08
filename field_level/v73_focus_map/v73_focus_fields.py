"""v73 (code saved after independent critique): per-field focus proxies from the post/pre pillar amplitude ratio map (64 px blocks),
their spread (la_iqr), plane gradient, and relations to the delta noise (MAD), positive density and block-mean delta.  Needs v78_p_readout output.
usage: python v73_focus_fields.py [out.csv]"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v78_candidate_path'))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import v78_calibrate as K
V70, V72, V78 = K.V70, K.V72, K.V78
led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
def blockmap(xy, v, block=64, stat='median'):
    gx = np.clip((xy[:, 0] // block).astype(int), 0, 31); gy = np.clip((xy[:, 1] // block).astype(int), 0, 31); ok = np.isfinite(v); cell = (gy * 32 + gx)
    out = np.full(1024, np.nan); order = np.argsort(cell[ok]); cs = cell[ok][order]; vv = v[ok][order]; b = np.flatnonzero(np.diff(cs)) + 1
    for c_, seg in zip(np.split(cs, b), np.split(vv, b)):
        if len(seg) >= 20: out[c_[0]] = np.median(seg) if stat == 'median' else seg.mean()
    return out.reshape(32, 32)
rows = []
for r in led.itertuples():
    z78 = np.load(V78 / 'fields' / f'{r.fid}.npz'); z72 = np.load(V72 / 'fields' / f'{r.fid}.npz')
    xy = z78['ctr'].astype(float); la = z78['amp_ratio'].astype(float); wd = z78['wid_diff'].astype(float)
    d = z72['S5'].astype(float); ok = np.isfinite(d) & (z78['sd_periods'] > 2)
    if ok.sum() < 20000: continue
    med = np.median(d[ok]); mad = 1.4826 * np.median(abs(d[ok] - med)); pos = np.where(ok, (d > med + 4 * mad).astype(float), np.nan)
    la_map = blockmap(xy, la); wd_map = blockmap(xy, wd); pm = blockmap(xy, pos, stat='mean'); dm = blockmap(xy, np.where(ok, d, np.nan))
    g = np.isfinite(la_map) & np.isfinite(pm); yy, xx = np.mgrid[0:32, 0:32]; gg = np.isfinite(la_map)
    A = np.column_stack([np.ones(gg.sum()), xx[gg], yy[gg]]); coef = np.linalg.lstsq(A, la_map[gg], rcond=None)[0]
    absla = np.abs(la_map - np.nanmedian(la_map)); g2 = g & np.isfinite(dm)
    rows.append(dict(fid=r.fid, date=r.date, group=r.group, fixed29=r.fixed29, la_med=float(np.nanmedian(la_map)), la_iqr=float(np.nanpercentile(la_map, 75) - np.nanpercentile(la_map, 25)),
        la_grad_per1000=float(np.hypot(coef[1], coef[2]) / 64 * 1000), la_grad_dir=float((np.degrees(np.arctan2(coef[2], coef[1])) + 360) % 180), wd_med=float(np.nanmedian(wd_map)),
        mad=float(mad), count_k4=float(np.nansum(pos) / ok.sum() * 91000),
        rho_pos_vs_abs_la=float(spearmanr(pm[g], absla[g]).statistic) if g.sum() > 200 else np.nan,
        rho_dm_vs_la=float(spearmanr(dm[g2], la_map[g2]).statistic) if g2.sum() > 200 else np.nan))
t = pd.DataFrame(rows); out = sys.argv[1] if len(sys.argv) > 1 else str(V78 / 'focus_proxy_by_field_recheck.csv'); t.to_csv(out, index=False)
print('spearman(la_iqr, mad) all', round(spearmanr(t.la_iqr, t.mad).statistic, 3), ' median rho_pos', round(t.rho_pos_vs_abs_la.median(), 3), ' median rho_dm', round(t.rho_dm_vs_la.median(), 3))
