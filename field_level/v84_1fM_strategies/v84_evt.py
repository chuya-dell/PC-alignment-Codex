"""v84 part B (d)+B-2+B-4: extreme-value (GPD) threshold calibration, how many blank boards are needed, and numeric checks of the main risks.
Blank fields: 69 development fields, 9 boards (one per date).  z = (S5 - field median)/(1.4826 MAD), valid and scar-excluded (C1).
(d): pooled exceedances above u fitted by a GPD; k(mu) = threshold where the mean false positives per field (valid pillars) = mu (Poisson: mu=0.82 gives P(count<=2)=95%).
Boards: bootstrap over the 9 boards, and sub-sampling n_b boards (3..8) with leave-boards-out evaluation.
Risks: common-mode drift sensitivity, low-frequency drift (block medians), focus (la_iqr) vs null counts, dilution of group averaging, frame-average share.
usage: python v84_evt.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.stats import genpareto, poisson, spearmanr
R = ROOT / 'data/results/v84_1fM_strategies'
rng = np.random.default_rng(7)


def load():
    m = pd.read_csv(R / 'field_meta.csv', dtype={'date': str, 'board': str}); m = m[(m.group == 'blank') & m.ok].reset_index(drop=True)
    foc = pd.read_csv(ROOT / 'data/results/v78_candidate_path/focus_proxy_by_field.csv', dtype={'date': str}).set_index('fid')
    F = []
    for r in m.itertuples():
        z = np.load(R / 'cache' / f'{r.fid}.npz'); ex = z['excl']; a = z['z5'][ex]; ok = np.isfinite(a)
        F.append(dict(fid=r.fid, board=r.date, z=a[ok].astype(np.float64), ctr=z['ctr'][ex][ok], sigma=r.sigma, la_iqr=foc.loc[r.fid, 'la_iqr']))
    return m, F


def fit_gpd(zs, u):
    ex = zs[zs > u] - u
    if len(ex) < 30: return None
    c, loc, sc = genpareto.fit(ex, floc=0); return dict(xi=float(c), beta=float(sc), zeta=len(ex) / len(zs), nex=len(ex))


def tail_prob(g, u, k):
    if g['xi'] == 0: return g['zeta'] * np.exp(-(k - u) / g['beta'])
    base = 1 + g['xi'] * (k - u) / g['beta']
    return 0.0 if base <= 0 else g['zeta'] * base ** (-1 / g['xi'])


def k_evt(F, side, mu, u=5.0):
    zs = np.concatenate([f['z'] for f in F]) * side; g = fit_gpd(zs, u)
    if g is None: return np.nan, None
    n_mean = np.mean([len(f['z']) for f in F])
    f = lambda k: n_mean * tail_prob(g, u, k) - mu
    try: return float(optimize.brentq(f, u, 1000)), g
    except Exception: return np.nan, g


def k_emp(F, side, limit=2, grid=np.arange(3, 40.01, 0.5)):
    for k in grid:
        c = np.array([int((f['z'] * side > k).sum()) for f in F])
        if np.percentile(c, 95) <= limit: return float(k)
    return float(grid[-1])


def main():
    m, F = load(); boards = sorted(set(f['board'] for f in F)); out = {}
    # ---- (d) EVT vs empirical ----
    for side, nm in ((1, 'pos(dark)'), (-1, 'neg(bright)')):
        for mu in (0.5, 0.82):
            for u in (4.0, 5.0, 6.0):
                k, g = k_evt(F, side, mu, u); out[f'EVT_{nm}_mu{mu}_u{u}'] = dict(k=k, **(g or {}))
        out[f'EMP_{nm}_P95le2'] = dict(k=k_emp(F, side))
        # mean FP/field at the empirical k and the check of the EVT tail against the empirical tail counts
        zs = np.concatenate([f['z'] for f in F]) * side; g = fit_gpd(zs, 5.0); n_mean = np.mean([len(f['z']) for f in F])
        out[f'TAILCHECK_{nm}'] = {str(k): dict(emp_mean=float(np.mean([(f['z'] * side > k).sum() for f in F])), evt_mean=float(n_mean * tail_prob(g, 5.0, k))) for k in (6, 8, 10, 12, 15)}
    # ---- bootstrap over boards ----
    bs = {'emp_pos': [], 'emp_neg': [], 'evt_pos': [], 'evt_neg': []}
    for b_ in range(300):
        pick = rng.choice(boards, len(boards), replace=True); G = [f for bd in pick for f in F if f['board'] == bd]
        bs['emp_pos'].append(k_emp(G, 1)); bs['emp_neg'].append(k_emp(G, -1)); bs['evt_pos'].append(k_evt(G, 1, 0.82)[0]); bs['evt_neg'].append(k_evt(G, -1, 0.82)[0])
    for k, v in bs.items(): v = np.array(v, float); out[f'BOOT_{k}'] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)), lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)))
    # ---- n_b boards for calibration, evaluated on the other boards ----
    sub = []
    for nb in (3, 4, 5, 6, 7, 8):
        ks_e, ks_v, passes = [], [], []
        for rep in range(60):
            tr = list(rng.choice(boards, nb, replace=False)); te = [b for b in boards if b not in tr]
            Gtr = [f for f in F if f['board'] in tr]; Gte = [f for f in F if f['board'] in te]
            ke = k_emp(Gtr, 1); kv = k_evt(Gtr, 1, 0.82)[0]
            ks_e.append(ke); ks_v.append(kv)
            for kk, tag in ((ke, 'emp'), (kv, 'evt'), (ke + 1.0, 'emp+1'), (ke + 2.0, 'emp+2'), (kv + 1.0, 'evt+1')):
                c = np.array([int((f['z'] > kk).sum()) for f in Gte]); p95 = np.percentile(c, 95)
                passes.append(dict(nb=nb, rule=tag, ok_p95=p95 <= 2, frac_le2=(c <= 2).mean(), mean=c.mean()))
        pdf = pd.DataFrame(passes)
        sub.append(dict(nb=nb, k_emp_mean=np.mean(ks_e), k_emp_sd=np.std(ks_e), k_evt_mean=np.nanmean(ks_v), k_evt_sd=np.nanstd(ks_v),
                        **{f'pass_{r}': float(pdf[pdf.rule == r].ok_p95.mean()) for r in ('emp', 'evt', 'emp+1', 'emp+2', 'evt+1')}, **{f'fle2_{r}': float(pdf[pdf.rule == r].frac_le2.mean()) for r in ('emp', 'evt', 'emp+1', 'emp+2')}))
    sub = pd.DataFrame(sub); sub.to_csv(R / 'B4_boards_subsample.csv', index=False)
    # extrapolation: sd(k) ~ c / sqrt(nb) (fit on nb=3..8, pooled empirical + EVT)
    for col in ('k_emp_sd', 'k_evt_sd'):
        cfit = float(np.mean(sub[col] * np.sqrt(sub.nb))); out[f'EXTRAP_{col}'] = dict(c=cfit, **{f'nb_for_sd_{s}': float((cfit / s) ** 2) for s in (1.0, 0.5, 0.25)})
    # ---- board-level variation of the null: ICC of the field counts at k=4,8 (positive side) ----
    for k in (4, 8, 10):
        c = np.array([(f['z'] > k).sum() for f in F]); bd = np.array([f['board'] for f in F]); gm = c.mean()
        ssb = sum((bd == b).sum() * (c[bd == b].mean() - gm) ** 2 for b in boards); ssw = sum(((c[bd == b] - c[bd == b].mean()) ** 2).sum() for b in boards)
        msb = ssb / (len(boards) - 1); msw = ssw / (len(c) - len(boards)); n0 = len(c) / len(boards)
        out[f'ICC_k{k}'] = dict(icc=float(max(0, (msb - msw) / (msb + (n0 - 1) * msw))), board_mean_sd=float(np.std([c[bd == b].mean() for b in boards])), within_sd=float(np.sqrt(msw)))
    # ---- risk 1: common-mode drift (global gain/offset change between the pair) : FP counts when the centre moves by d sigma ----
    for d in (0.0, 0.1, 0.25, 0.5):
        out[f'DRIFT_shift{d}'] = {str(k): float(np.mean([(f['z'] + d > k).sum() for f in F])) for k in (3, 4, 6, 9.5)}
    out['DRIFT_median_over_sigma_all632'] = None
    mm = pd.read_csv(R / 'field_meta.csv'); mm = mm[mm.ok]
    # medians relative to sigma over all fields (median of the S5 delta itself)
    out['DRIFT_median_over_sigma_all632'] = dict(abs_median_p50=float((abs(mm['median']) / mm.sigma).median()), abs_median_p95=float((abs(mm['median']) / mm.sigma).quantile(.95)))
    # ---- risk 2: low-frequency drift: SD of 128-px block medians (in sigma units) ----
    sds = []
    for f in F:
        gx = np.clip((f['ctr'][:, 0] // 128).astype(int), 0, 15); gy = np.clip((f['ctr'][:, 1] // 128).astype(int), 0, 15); cell = gy * 16 + gx
        df = pd.DataFrame(dict(c=cell, z=f['z'])); bm = df.groupby('c').z.median(); sds.append(bm.std())
    out['LOWFREQ_block128_median_sd_sigma'] = dict(mean=float(np.mean(sds)), p50=float(np.median(sds)), p95=float(np.percentile(sds, 95)), gauss_expect=float(1.253 / np.sqrt(np.mean([len(f['z']) for f in F]) / 256)))
    # ---- risk 3: focus: null counts vs la_iqr and sigma ----
    for k in (4, 8):
        c = np.array([(f['z'] > k).sum() + (f['z'] < -k).sum() for f in F]); la = np.array([f['la_iqr'] for f in F]); sg = np.array([f['sigma'] for f in F])
        out[f'FOCUS_k{k}'] = dict(rho_la=float(spearmanr(la, c)[0]), p_la=float(spearmanr(la, c)[1]), rho_sigma=float(spearmanr(sg, c)[0]), mean_low_la=float(c[la <= 0.15].mean()) if (la <= 0.15).any() else None,
                                  mean_mid=float(c[(la > 0.15) & (la <= 0.3)].mean()), mean_high=float(c[la > 0.3].mean()) if (la > 0.3).any() else None, n_low=int((la <= 0.15).sum()), n_mid=int(((la > 0.15) & (la <= 0.3)).sum()), n_high=int((la > 0.3).sum()))
    # ---- risk 4: dilution (analytic): fraction of pillars that are signal and chance that a 7-pillar group has >=2 signals ----
    n = np.mean([len(f['z']) for f in F]); out['DILUTION'] = {str(M): dict(frac=M / n, p_group7_ge2=float(1 - (1 - M / n) ** 7 - 7 * (M / n) * (1 - M / n) ** 6), snr_group7_one_signal=float(1 / np.sqrt(7))) for M in (18, 46)}
    # ---- frame average shares ----
    for s in (0.505, 0.53):
        out[f'FRAMEAVG_s{s}'] = {str(N): float(np.sqrt(1 - s + s / N)) for N in (1, 2, 4, 8, 16, 1000)}
    json.dump(out, open(R / 'B_evt_risks.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k.startswith(('EVT', 'EMP', 'BOOT', 'EXTRAP', 'ICC', 'TAIL'))}, indent=0, default=float)[:6000]); print(sub.round(3).to_string())
    print(json.dumps({k: v for k, v in out.items() if k.startswith(('DRIFT', 'LOWFREQ', 'FOCUS', 'DILUTION', 'FRAMEAVG'))}, indent=0, default=float))


if __name__ == '__main__':
    main()
