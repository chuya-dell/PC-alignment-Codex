"""v78 step H (part 1): threshold calibration rule and false-positive evaluation for candidate paths.
Rule (frozen): per field, centre = median and scale = 1.4826*MAD of the field's valid, scar-excluded delta; threshold = centre + k*scale.
k is the smallest value on a grid such that the 95th percentile (over calibration blank fields) of false positives per 91k pillars <= LIMIT.
Calibration is leave-one-date-out on the 69 development blank fields; the frozen k uses all blanks.
Readouts: S0 (current standard), S2 (L-map + ideal grid + bilinear), S5 (L-map + actual centres + aperture, post-valid), S5P (as S5 but post-fit not required).
usage: python v78_calibrate.py"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V72 = ROOT / 'data/results/v72_real_field_readout'; V78 = ROOT / 'data/results/v78_candidate_path'
OUT = V78
PER = 7.37
KGRID = np.round(np.arange(3.0, 15.01, 0.5), 2)
LIMIT_MAIN, LIMIT_OLD = 2.0, 10.0
REF = 91000.0
READS = ['S0', 'S2', 'S5', 'S5P']


def load_field(fid, excl_periods=2.0):
    z72 = np.load(V72 / 'fields' / f'{fid}.npz'); z78 = np.load(V78 / 'fields' / f'{fid}.npz')
    out = {}
    # std-grid readouts: exclusion = inside mask or within excl_periods outside
    ex_std = z72['in_mask_std'] | (z72['dist_std'] < excl_periods * PER)
    for r in ('S0', 'S2'):
        d = z72[r].astype(float); ok = np.isfinite(d)
        out[r] = (d, ok & ~ex_std, ok)
    ex_c = ~(z78['sd_periods'] > excl_periods)
    d5 = z72['S5'].astype(float); ok5 = np.isfinite(d5)
    out['S5'] = (d5, ok5 & ~ex_c, ok5)
    d5p = z78['S5P'].astype(float); okp = np.isfinite(d5p)
    out['S5P'] = (d5p, okp & ~ex_c, okp)
    return out


def fp_counts(d, valid_excl, ks=KGRID):
    v = d[valid_excl]
    if len(v) < 5000: return None
    med = np.median(v); mad = 1.4826 * np.median(abs(v - med))
    return {k: float((v > med + k * mad).sum() / len(v) * REF) for k in ks}, len(v), med, mad


def main():
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    bl = led[led.group == 'blank']
    rows = []
    for r in bl.itertuples():
        f = load_field(r.fid)
        for rd in READS:
            d, vex, vall = f[rd]; res = fp_counts(d, vex)
            if res is None: continue
            c, n, med, mad = res
            for k in KGRID: rows.append(dict(fid=r.fid, date=r.date, fixed29=r.fixed29, readout=rd, k=k, fp91k=c[k], n_valid=n, n_all=int(vall.sum()), mad=mad))
    t = pd.DataFrame(rows); t.to_csv(OUT / 'blank_fp_by_k.csv', index=False)
    # leave-one-date-out calibration
    def choose_k(df, limit):
        for k in KGRID:
            if np.percentile(df[df.k == k].fp91k, 95) <= limit: return float(k), True
        return float(KGRID[-1]), False
    cal = []
    for rd in READS:
        q = t[t.readout == rd]
        for lim, nm in ((LIMIT_MAIN, 'main2'), (LIMIT_OLD, 'old10')):
            kall, okk = choose_k(q, lim)
            cal.append(dict(readout=rd, limit=nm, scope='all_blanks', held_out='', k=kall, reached=okk))
            for date in sorted(q.date.unique()):
                kd, okd = choose_k(q[q.date != date], lim)
                te = q[(q.date == date) & (q.k == kd)].fp91k
                cal.append(dict(readout=rd, limit=nm, scope='leave_one_date_out', held_out=date, k=kd, reached=okd, test_fields=len(te), test_median=te.median(), test_mean=te.mean(), test_p95=np.percentile(te, 95), test_max=te.max(), test_frac_le2=float((te <= 2).mean()), test_frac_le10=float((te <= 10).mean())))
    c = pd.DataFrame(cal); c.to_csv(OUT / 'calibration_k.csv', index=False)
    pd.set_option('display.width', 220)
    print(c[c.scope == 'all_blanks'].to_string(index=False))
    lo = c[(c.scope == 'leave_one_date_out') & (c.limit == 'main2')]
    print(lo.groupby('readout').agg(k_med=('k', 'median'), k_max=('k', 'max'), reached=('reached', 'mean'), test_median=('test_median', 'median'), test_mean=('test_mean', 'mean'), test_p95=('test_p95', 'median'), test_max=('test_max', 'max'), frac_le2=('test_frac_le2', 'mean'), frac_le10=('test_frac_le10', 'mean')).round(2))
    # at the frozen k: distribution on development blanks (in-sample) including worst field and unmeasurable fraction
    out = []
    for rd in READS:
        kall = float(c[(c.readout == rd) & (c.scope == 'all_blanks') & (c.limit == 'main2')].k.iloc[0])
        q = t[(t.readout == rd) & (t.k == kall)]
        out.append(dict(readout=rd, k=kall, fields=len(q), median=q.fp91k.median(), mean=q.fp91k.mean(), p90=q.fp91k.quantile(.9), p95=q.fp91k.quantile(.95), worst=q.fp91k.max(), frac_le2=(q.fp91k <= 2).mean(), frac_le10=(q.fp91k <= 10).mean(),
                        measurable_fraction_median=float((q.n_valid / 88000).median()), unmeasurable_fraction_median=float(1 - (q.n_valid / 88000).median())))
    pd.DataFrame(out).to_csv(OUT / 'frozen_k_blank_summary.csv', index=False); print(pd.DataFrame(out).round(2).to_string(index=False))


if __name__ == '__main__':
    main()
