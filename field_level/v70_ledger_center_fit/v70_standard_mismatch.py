"""v70: measured reading-position error of the current standard (ideal grid + integer rounding) against
the fitted actual pillar centres, per field.  Uses fields/ (standard xy, matrix) and fields2/ (centres, L map)."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import v70_lib as L
import v70_centers2 as C
OUT = L.ROOT / 'data/results/v70_ledger_center_fit'


def one(fid):
    z1 = np.load(OUT / 'fields' / f'{fid}.npz'); z2 = np.load(OUT / 'fields2' / f'{fid}.npz')
    pre = dict(ctr=z2['pre_ctr'].astype(float), pred=z2['pre_pred'], amp=z2['pre_amp'], wid=z2['pre_wid'], aloc=z2['pre_aloc'])
    post = dict(ctr=z2['post_ctr'].astype(float), pred=z2['post_pred'], amp=z2['post_amp'], wid=z2['post_wid'], aloc=z2['post_aloc'])
    Sp = C.sub_flags(pre, 'main'); So = C.sub_flags(post, 'main')
    jL = z2['jL']; consL = z2['consL']
    xy = z1['std_xy'].astype(float); A = z1['matrix']
    d, n = cKDTree(pre['ctr']).query(xy)                    # nearest actual pre pillar centre
    ok = (~Sp[n]) & (~So[jL[n]]) & consL[n] & (d < 0.5 * L.PITCH)
    xyp = xy @ A[:, :2].T + A[:, 2]
    s_pre = np.rint(xy) - pre['ctr'][n]
    s_post = np.rint(xyp) - post['ctr'][jL[n]]
    mism = s_post - s_pre
    # A versus the lattice-implied physical map at the sample points
    xyL = xy @ z2['Llin'].T + z2['tL']
    a_vs_l = np.linalg.norm(xyp - xyL, axis=1)
    off_pre = np.linalg.norm(xy - pre['ctr'][n], axis=1)          # ideal-grid offset from actual centre
    rd = np.linalg.norm(np.rint(xy) - xy, axis=1)
    def q(v, p): return float(np.percentile(v[ok], p))
    mm = np.linalg.norm(mism, axis=1)
    return dict(fid=fid, n_std=len(xy), n_ok=int(ok.sum()), ideal_offset_med=q(off_pre, 50), ideal_offset_p95=q(off_pre, 95), ideal_offset_max=float(off_pre[ok].max()),
                mismatch_med=q(mm, 50), mismatch_p95=q(mm, 95), mismatch_frac_gt1=float((mm[ok] > 1).mean()), mismatch_frac_gt2=float((mm[ok] > 2).mean()),
                a_vs_l_med=q(a_vs_l, 50), a_vs_l_p95=q(a_vs_l, 95), a_vs_l_frac_gt1=float((a_vs_l[ok] > 1).mean()), a_vs_l_frac_gt3=float((a_vs_l[ok] > 3.6).mean()),
                mismatch_x_med=float(np.median(mism[ok, 0])), mismatch_y_med=float(np.median(mism[ok, 1])))


def main():
    led = pd.read_csv(OUT / 'ledger.csv', dtype={'date': str, 'board': str})
    rows = []
    for r in led.itertuples():
        try: rows.append(one(r.fid))
        except Exception as e: rows.append(dict(fid=r.fid, error=repr(e)))
    t = pd.DataFrame(rows).merge(led[['fid', 'date', 'group', 'fixed29']], on='fid')
    t.to_csv(OUT / 'standard_reading_mismatch_by_field.csv', index=False, encoding='utf-8-sig')
    print(t[['ideal_offset_med', 'ideal_offset_p95', 'ideal_offset_max', 'mismatch_med', 'mismatch_p95', 'mismatch_frac_gt1', 'mismatch_frac_gt2', 'a_vs_l_med', 'a_vs_l_p95', 'a_vs_l_frac_gt1', 'a_vs_l_frac_gt3']].describe().loc[['mean', '25%', '50%', '75%', 'max']].round(3).T)
    print(t.groupby('fixed29')[['mismatch_med', 'mismatch_p95', 'a_vs_l_med', 'a_vs_l_p95']].median().round(3))
    print(t.groupby('group')[['mismatch_med', 'a_vs_l_med']].median().round(3))
    print('errors:', t.get('error', pd.Series(dtype=object)).notna().sum())


if __name__ == '__main__':
    main()
