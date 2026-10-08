"""v78 step H: re-analysis of the DNA experiment days (9 dates in the development set) with the old path and the candidate path (reference values only,
development data: NOT a final judgment).  Field-level rates; per date: each analyte concentration vs the date's blank fields, one-sided Mann-Whitney (higher in the
concentration), Holm over all comparisons; plus Spearman(log10 concentration, rate) per date.  Difference direction pre-post, bright-band mask auto (inside registration),
stain mask off; the 540-test family of the old analysis is a different family (9 threshold multipliers) and is not mixed with this one."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v78_candidate_path'))
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import v78_calibrate as K
V70, V72, V78 = K.V70, K.V72, K.V78


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


def path_rates(led, readout, k, excl=2.0):
    rows = []
    for r in led.itertuples():
        f = K.load_field(r.fid, excl)
        d, vex, vall = f[readout]; v = d[vex]
        if len(v) < 5000: rows.append(dict(fid=r.fid, rate=np.nan, n_valid=len(v))); continue
        med = np.median(v); mad = 1.4826 * np.median(abs(v - med)); thr = med + k * mad
        rows.append(dict(fid=r.fid, rate=float((v > thr).mean()), count91k=float((v > thr).mean() * 91000), n_valid=len(v), k=k))
    return pd.DataFrame(rows)


def dose_tests(led, rates, tag):
    t = led.merge(rates[['fid', 'rate']], on='fid')
    res = []; sp = []
    for date, g in t.groupby('date'):
        bl = g[g.group == 'blank'].rate.dropna()
        an = g[g.group == 'analyte']
        for c, gg in an.groupby('conc_M'):
            x = gg.rate.dropna()
            if len(bl) < 3 or len(x) < 3: continue
            u = mannwhitneyu(x, bl, alternative='greater'); res.append(dict(path=tag, date=date, conc_M=c, fields=len(x), blank_fields=len(bl), median_conc=x.median(), median_blank=bl.median(), p=u.pvalue))
        h = g[g.group.isin(['blank', 'analyte'])].dropna(subset=['rate']); h = h.assign(lc=np.log10(h.conc_M.replace(0, np.nan)).fillna(-18))
        if len(h) > 6: s = spearmanr(h.lc, h.rate); sp.append(dict(path=tag, date=date, rho=s.statistic, p=s.pvalue, fields=len(h)))
    r = pd.DataFrame(res); r['p_holm'] = holm(r.p.values); r['n_comparisons'] = len(r)
    return r, pd.DataFrame(sp)


def main():
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    cal = pd.read_csv(V78 / 'calibration_k.csv')
    k5 = float(cal[(cal.readout == 'S5') & (cal.scope == 'all_blanks') & (cal.limit == 'main2')].k.iloc[0])
    old = pd.read_csv(V70 / 'current_standard_v70_by_field.csv', dtype={'date': str})[['fid', 'positive_rate']].rename(columns={'positive_rate': 'rate'})
    paths = {'old_standard_3SDpool': old, f'new_S5_k{k5}_scarexcl': path_rates(led, 'S5', k5), 'new_S5_k4_scarexcl': path_rates(led, 'S5', 4.0), 'new_S5_k6_scarexcl': path_rates(led, 'S5', 6.0)}
    allres, allsp, summ = [], [], []
    for tag, rt in paths.items():
        r, s = dose_tests(led, rt, tag); allres.append(r); allsp.append(s)
        t = led.merge(rt[['fid', 'rate']], on='fid'); b = t[t.group == 'blank'].rate
        summ.append(dict(path=tag, blank_median=b.median(), blank_mean=b.mean(), analyte_median=t[t.group == 'analyte'].rate.median(), comparisons=len(r), holm_lt005=int((r.p_holm < .05).sum()), raw_lt005=int((r.p < .05).sum()),
                         spearman_median_rho=s.rho.median(), spearman_dates_p_lt005=int((s.p < .05).sum())))
        rt.to_csv(V78 / f'dna_rates_{tag}.csv', index=False)
    pd.concat(allres).to_csv(V78 / 'dna_dose_tests.csv', index=False); pd.concat(allsp).to_csv(V78 / 'dna_spearman.csv', index=False)
    s = pd.DataFrame(summ); s.to_csv(V78 / 'dna_summary.csv', index=False); pd.set_option('display.width', 220); print(s.round(6).to_string(index=False))


if __name__ == '__main__':
    main()
