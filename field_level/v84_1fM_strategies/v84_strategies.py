"""v84 part B-1: false positives and detection of 1 fM-like signals (M = 18 or 46 signal pillars per field) for strategies on the 69 development blank fields (C1 readouts, cached by v84_cache.py).
Signals are injected in the delta domain (z + s*A on M random valid pillars; A in units of the field robust SD; s=+1 darkening (post darker, positive delta), s=-1 brightening).
This ignores effects of the signal on the alignment and on the pillar fit (S5 invalid for strongly darkened pillars is handled by a validity probability v(A), an ASSUMPTION for A<3; G measured 61% at 3 sigma).
Strategies: per-pillar threshold (k from the frozen rule), (a) frame average N (model + real check in A: the tail does not shrink, the bulk shrinks by r_N), (b) per-field excess count (leave-one-board-out threshold = 95th percentile of the null counts).
usage: python v84_strategies.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
R = ROOT / 'data/results/v84_1fM_strategies'
S_TEMP = 0.505          # temporal (burst) share of the delta variance (measured: burst 0.0255^2 / remount 0.0352^2, A part)
VALID_DARK = {0: 1.0, 1: 0.9, 2: 0.75, 3: 0.61, 5: 0.29}   # P(post-fit success) for a darkened pillar; 3 sigma and above measured in G, below ASSUMED
rng = np.random.default_rng(20261008)


def rN(N): return float(np.sqrt(1 - S_TEMP + S_TEMP / N))


def load_blanks():
    m = pd.read_csv(R / 'field_meta.csv', dtype={'date': str, 'board': str}); m = m[(m.group == 'blank') & m.ok].reset_index(drop=True)
    F = []
    for r in m.itertuples():
        z = np.load(R / 'cache' / f'{r.fid}.npz'); ex = z['excl']
        a = z['z5'][ex]; a = a[np.isfinite(a)].astype(np.float64)
        b = z['z5p'][ex]; b = b[np.isfinite(b)].astype(np.float64)
        F.append(dict(fid=r.fid, board=r.date, z5=a, z5p=b))
    return m, F


def avg_transform(z, N):
    """bulk (|z|<=3) shrinks by r_N, tail (|z|>=5) unchanged (A: avg-of-2 real pairs, tail counts at fixed absolute threshold unchanged within 10%), linear blend between; returns z in units of the new bulk SD."""
    if N == 1: return z
    r = rN(N); w = np.clip((np.abs(z) - 3) / 2, 0, 1); x = z * (r + (1 - r) * w)
    return x / r


def vprob(A):
    ks = sorted(VALID_DARK); return float(np.interp(A, ks, [VALID_DARK[k] for k in ks]))


def null_counts(F, key, k0, side, N):
    c = []
    for f in F:
        z = avg_transform(f[key], N); c.append(int((z > k0).sum()) if side > 0 else int((z < -k0).sum()))
    return np.array(c)


def excess_power(F, key, k0, side, A, M, N, reps=20, thin=False):
    boards = np.array([f['board'] for f in F]); n0 = null_counts(F, key, k0, side, N); out = []
    fa = []
    for b in np.unique(boards):
        ho = boards == b; T = np.percentile(n0[~ho], 95)
        fa += list(n0[ho] > T)
        for i in np.where(ho)[0]:
            z = avg_transform(F[i][key], N); n = len(z)
            for _ in range(reps):
                m = M if not thin else rng.binomial(M, vprob(A))
                idx = rng.choice(n, m, replace=False); zs = z[idx]
                zn = (zs + side * A / (rN(N) if N > 1 else 1.0))
                cnt = n0[i] + (np.sum(zn > k0) - np.sum(zs > k0)) if side > 0 else n0[i] + (np.sum(zn < -k0) - np.sum(zs < -k0))
                out.append(cnt > T)
    return float(np.mean(out)), float(np.mean(fa))


def per_pillar(F, key, k, side, A, N):
    """fraction of injected pillars above k (per-pillar), and false positives per field at k (mean, 95th pct, max, fields<=2)."""
    n0 = null_counts(F, key, k, side, N); z = np.concatenate([avg_transform(f[key], N) for f in F])
    zs = z + side * A / (rN(N) if N > 1 else 1.0)
    det = float(np.mean(zs > k)) if side > 0 else float(np.mean(zs < -k))
    return det, float(n0.mean()), float(np.percentile(n0, 95)), int(n0.max()), float((n0 <= 2).mean())


def main():
    m, F = load_blanks(); boards = sorted(set(f['board'] for f in F)); print('blank fields', len(F), 'boards', len(boards))
    rows = []
    # ---- null counts of the strategies (per field, raw counts; valid pillars ~7.4e4) ----
    for key, lab in (('z5', 'S5(abn excl)'), ('z5p', 'S5P(abn counted)')):
        for N in (1, 4, 8):
            for k in (3, 4, 5, 6, 8, 10.5, 12, 15):
                for side, sl in ((1, 'pos(dark)'), (-1, 'neg(bright)')):
                    c = null_counts(F, key, k, side, N)
                    rows.append(dict(kind='null', readout=lab, N=N, k=k, side=sl, mean=c.mean(), sd=c.std(), p95=np.percentile(c, 95), max=c.max(), frac_le2=(c <= 2).mean(),
                                     board_mean_sd=float(np.std([c[[f['board'] == b for f in F]].mean() for b in boards]))))
    pd.DataFrame(rows).to_csv(R / 'B1_null_counts.csv', index=False)
    # ---- per-pillar detection at frozen-style thresholds, brightening and darkening ----
    pr = []
    kfp = {}
    for key, lab in (('z5', 'S5'), ('z5p', 'S5P')):
        for N in (1, 4, 8):
            for side in (1, -1):
                # smallest k (0.5 grid) with 95th percentile of the null count <= 2
                for k in np.arange(3, 30.01, 0.5):
                    c = null_counts(F, key, k, side, N)
                    if np.percentile(c, 95) <= 2: break
                kfp[(lab, N, side)] = float(k)
                for A in ((1, 2, 3, 4) if side > 0 else (2, 3, 5, 8, 12, 16, 20)):
                    d = per_pillar(F, key, k, side, A, N)
                    thin = vprob(A) if (side > 0 and lab == 'S5') else 1.0
                    pr.append(dict(readout=lab, N=N, side='dark' if side > 0 else 'bright', k_fp2=float(k), A=A, detect_rate=d[0] * thin, fp_mean=d[1], fp_p95=d[2], fp_max=d[3], frac_le2=d[4]))
    pd.DataFrame(pr).to_csv(R / 'B1_per_pillar.csv', index=False)
    # ---- (b) excess count power (leave-one-board-out), M in {18,46} ----
    ex = []
    for key, lab, thin in (('z5', 'S5', True), ('z5p', 'S5P', False)):
        for N in (1, 4, 8):
            for k0 in (3, 4, 5, 6, 8):
                for side in (1, -1):
                    for A in ((1, 2, 3) if side > 0 else (2, 3, 5, 8)):
                        for M in (18, 46):
                            if side < 0 and lab == 'S5P': continue
                            p, fa = excess_power(F, key, k0, side, A, M, N, reps=10, thin=thin and side > 0)
                            ex.append(dict(readout=lab, N=N, k0=k0, side='dark' if side > 0 else 'bright', A=A, M=M, power=p, false_alarm_cv=fa))
        print('done', lab, flush=True)
    e = pd.DataFrame(ex); e.to_csv(R / 'B1_excess_power.csv', index=False)
    pd.set_option('display.width', 250)
    print(json.dumps({str(k): v for k, v in kfp.items()}, indent=0))
    best = e.sort_values('power', ascending=False).groupby(['readout', 'N', 'side', 'A', 'M']).head(1)
    print(best[best.N == 1].sort_values(['readout', 'side', 'A', 'M']).round(3).to_string())


if __name__ == '__main__':
    main()
