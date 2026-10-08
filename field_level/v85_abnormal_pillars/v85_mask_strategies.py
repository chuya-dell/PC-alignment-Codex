"""v85: part B strategies re-evaluated with the camera-fixed hotspot mask (post-frame hotspots from NON-blank fields, R=20 px on the pre OR the post position).
Same code as v84_strategies (delta-domain injection, leave-one-board-out), only the pillars inside the mask are removed from the blank fields.   usage: python v85_mask_strategies.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'field_level/v84_1fM_strategies'))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import v84_strategies as V
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V85 = ROOT / 'data/results/v85_abnormal_pillars'; R = V.R
RM = 20.0


def load_masked():
    m = pd.read_csv(R / 'field_meta.csv', dtype={'date': str, 'board': str}); m = m[(m.group == 'blank') & m.ok].reset_index(drop=True)
    hs = pd.read_csv(V85 / 'E3_hotspots_post_from_nonblank.csv'); tree = cKDTree(hs[['x', 'y']].values); F = []
    for r in m.itertuples():
        z = np.load(R / 'cache' / f'{r.fid}.npz'); z2 = np.load(V70 / 'fields2' / f'{r.fid}.npz'); ex = z['excl']; ctr = z['ctr'].astype(float); post = ctr @ z2['Llin'].T + z2['tL']
        inh = (tree.query(ctr)[0] < RM) | (tree.query(post)[0] < RM); keep = ex & ~inh
        a = z['z5'][keep]; a = a[np.isfinite(a)].astype(np.float64); b = z['z5p'][keep]; b = b[np.isfinite(b)].astype(np.float64)
        F.append(dict(fid=r.fid, board=r.date, z5=a, z5p=b))
    return m, F


def main():
    m, F = load_masked(); print('blank fields', len(F), 'mean valid', np.mean([len(f['z5']) for f in F]))
    out = {}; pr = []
    for key, lab in (('z5', 'S5'), ('z5p', 'S5P')):
        for N in (1, 8):
            for side in (1, -1):
                for k in np.arange(3, 30.01, 0.5):
                    c = V.null_counts(F, key, k, side, N)
                    if np.percentile(c, 95) <= 2: break
                for A in ((2, 3, 4) if side > 0 else (3, 5, 6, 7, 8, 9, 10, 12, 16)):
                    d = V.per_pillar(F, key, k, side, A, N); thin = V.vprob(A) if (side > 0 and lab == 'S5') else 1.0
                    pr.append(dict(readout=lab, N=N, side='dark' if side > 0 else 'bright', k_fp2=float(k), A=A, detect_rate=d[0] * thin, fp_mean=d[1], fp_p95=d[2], fp_max=d[3], frac_le2=d[4]))
    pd.DataFrame(pr).to_csv(R / 'B1m_per_pillar.csv', index=False)
    ex = []
    for key, lab, thin in (('z5', 'S5', True), ('z5p', 'S5P', False)):
        for N in (1, 8):
            for k0 in (3, 4, 5, 6, 8):
                for side in (1, -1):
                    if side < 0 and lab == 'S5P': continue
                    for A in ((1, 2, 3) if side > 0 else (2, 3, 4, 5, 8)):
                        for M in (18, 46):
                            p, fa = V.excess_power(F, key, k0, side, A, M, N, reps=10, thin=thin and side > 0)
                            ex.append(dict(readout=lab, N=N, k0=k0, side='dark' if side > 0 else 'bright', A=A, M=M, power=p, false_alarm_cv=fa))
    e = pd.DataFrame(ex); e.to_csv(R / 'B1m_excess_power.csv', index=False)
    pd.set_option('display.width', 250)
    p = pd.DataFrame(pr); print(p[(p.side == 'bright') & (p.readout == 'S5')].round(3).to_string())
    print(p[(p.side == 'dark') & (p.readout.isin(['S5', 'S5P']))].round(3).head(12).to_string())
    q = e[(e.false_alarm_cv <= 0.10)].sort_values('power', ascending=False).groupby(['readout', 'N', 'side', 'A', 'M']).head(1).sort_values(['readout', 'side', 'A', 'M', 'N'])
    print(q[(q.side == 'bright')][['readout', 'N', 'A', 'M', 'k0', 'power', 'false_alarm_cv']].round(2).to_string())


if __name__ == '__main__':
    main()
