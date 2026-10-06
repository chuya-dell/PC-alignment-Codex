"""v50: real-data held-out-pillar test of local correction (v24 residual vectors).

Pre-registered (before running): per FOV, hold out 20% of pillars (deterministic hash split,
or spatial 6x6 checkerboard blocks), select a candidate from the v49 family using ONLY the
training pillars (inner validation split), refit on all training pillars, then evaluate the
median |residual| on held-out pillars.  Compared: raw residual (current standard),
constant shift only (train median), selected local, and a permutation control (vectors
shuffled among training pillars; any gain there would be an overfit/selection artefact).
Residual reduction is a proxy, not accuracy gain: residuals contain detector error and
before/after-wash differences as well as registration error.
"""
import argparse, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'field_level/v49_semisynthetic_local_correction/field_run_semisynthetic_local_correction.py'
spec = importlib.util.spec_from_file_location('v49', P); v49 = importlib.util.module_from_spec(spec); sys.modules['v49'] = v49; spec.loader.exec_module(v49)
SEED = 20261006


def evaluate(xy, r, test_mask, key, rng):
    tr = ~test_mask; te = test_mask
    out = {}
    out['raw'] = np.linalg.norm(r[te], axis=1)
    c = np.median(r[tr], axis=0)
    out['const'] = np.linalg.norm(r[te] - c, axis=1)
    # Inner-split keys must differ from the outer split key modulo 10 (the hash split is mod 10);
    # the first version used a key difference of 400 (=0 mod 10) in random20, which made the inner
    # validation set EMPTY and silently selected the first candidate (block_affine_2x2).
    # outer key is 500+i (random20); here key=900+i -> use +3 / +5 offsets (difference 403 / 405).
    name, cv, fn = v49.choose_and_fit(xy[tr], r[tr], key + 3)
    assert np.isfinite(cv), 'inner validation empty or NaN'
    out['local'] = np.linalg.norm(r[te] - fn(xy[te]), axis=1)
    perm = rng.permutation(tr.sum())
    name_p, cv_p, fn_p = v49.choose_and_fit(xy[tr], r[tr][perm], key + 5)
    assert np.isfinite(cv_p), 'inner validation empty or NaN (perm)'
    out['perm'] = np.linalg.norm(r[te] - fn_p(xy[te]), axis=1)
    return out, name, float(cv)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--vectors', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--max-fovs', type=int, default=0); a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.vectors, compression='gzip', usecols=['fov_key', 'x_px', 'y_px', 'residual_x_px', 'residual_y_px'])
    rows = []
    for i, (fov, g) in enumerate(df.groupby('fov_key', sort=True)):
        if a.max_fovs and i >= a.max_fovs: break
        xy = g[['x_px', 'y_px']].to_numpy(float); r = g[['residual_x_px', 'residual_y_px']].to_numpy(float)
        if len(xy) < 60: continue
        rng = np.random.default_rng(SEED + i)
        for scheme in ('random20', 'block_checker'):
            if scheme == 'random20':
                test = ~v49.split(xy, 500 + i)
            else:
                bx = np.clip((xy[:, 0] / 2048 * 6).astype(int), 0, 5); by = np.clip((xy[:, 1] / 2044 * 6).astype(int), 0, 5)
                test = ((bx + by) % 5 == 0)  # ~20% of blocks, spatially separated
            if test.sum() < 15 or (~test).sum() < 40: continue
            res, name, cv = evaluate(xy, r, test, 900 + i, rng)
            rows.append(dict(fov_key=fov, scheme=scheme, n_train=int((~test).sum()), n_test=int(test.sum()), selected=name, inner_cv_score_px=cv,
                             raw_median=float(np.median(res['raw'])), const_median=float(np.median(res['const'])),
                             local_median=float(np.median(res['local'])), perm_median=float(np.median(res['perm']))))
        if (i + 1) % 20 == 0: print(f'{i+1} fovs', flush=True)
    t = pd.DataFrame(rows); t.to_csv(a.output / 'per_fov_holdout.csv', index=False)
    t['local_minus_const'] = t.local_median - t.const_median; t['local_minus_raw'] = t.local_median - t.raw_median; t['perm_minus_const'] = t.perm_median - t.const_median
    tol = 0.002
    summ = []
    for s, g in t.groupby('scheme'):
        d = g.local_minus_const
        summ.append(dict(scheme=s, n_fovs=len(g), median_raw=g.raw_median.median(), median_const=g.const_median.median(), median_local=g.local_median.median(),
                         median_perm=g.perm_median.median(), improved_vs_const=int((d <= -tol).sum()), worsened_vs_const=int((d >= tol).sum()),
                         unchanged_vs_const=int((d.abs() < tol).sum()), perm_improved_vs_const=int((g.perm_minus_const <= -tol).sum()),
                         median_reduction_vs_const=-d.median(), median_reduction_vs_raw=-g.local_minus_raw.median()))
    pd.DataFrame(summ).to_csv(a.output / 'summary_holdout.csv', index=False)
    print(pd.DataFrame(summ).round(4).to_string())


if __name__ == '__main__':
    main()
