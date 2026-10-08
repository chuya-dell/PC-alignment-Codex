"""v87 D4: integer-period check (pixel-value correlation scan, v70_period_fix) applied to the zero-truth pairs of step B: the burst pair (260904_p50_repeat), the 6 small-step pairs (261006 stepping motor),
and the 2026-10-08 burst (-0/-1, 24) and re-mount (-0/-3, 24) pairs.  For each pair: shift chosen by analyse_pair, corr at (0,0) (the chosen mapping), best other, margin, and whether the best other shift is (0,0).
usage: python v87_d4_period_on_pairs.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls'): sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, cv2, tifffile
from scipy import ndimage
import v75_pair_lib as P
import v75_run_pairs as RP
import v70_centers2 as C
import v70_period_fix as PF
import field_control_common as M60
OUT = ROOT / 'data/results/v87_part_d'; RAWN = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_z')


def margin_of(a, b, res):
    ca = C.centers_smooth(a); cb = C.centers_smooth(b)
    a_hp = a - cv2.GaussianBlur(a, (0, 0), 8); b_hp = b - cv2.GaussianBlur(b, (0, 0), 8); pc = ca['ctr'].astype(float)
    va = ndimage.map_coordinates(a_hp, [pc[:, 1], pc[:, 0]], order=1, mode='nearest')
    r = PF.scan(va, pc, b_hp, res['Llin'], res['tL'], cb['model'][1:], 3)
    items = sorted(r.items(), key=lambda kv: -kv[1]); others = [v for k, v in r.items() if k != (0, 0)]
    return dict(corr_zero=r[(0, 0)], corr_best=items[0][1], best=str(items[0][0]), best_is_zero=items[0][0] == (0, 0), margin_zero_vs_other=r[(0, 0)] - max(others))


def one(job):
    cv2.setNumThreads(1); label, pa, pb, kind = job
    a = tifffile.imread(pa).astype(np.float32); b = tifffile.imread(pb).astype(np.float32)
    res = P.analyse_pair(a, b); m = margin_of(a, b, res)
    return dict(label=label, kind=kind, period_shift=str(tuple(int(x) for x in res['period_shift'])), corr_zero_in_pair=float(res['period_corr_zero']), corr_best_in_pair=float(res['period_corr_best']), **m, corner_disagree_px=float(res['corner_disagree_px']))


if __name__ == '__main__':
    jobs = [(l, pa, pb, k) for l, pa, pb, k in RP.real_pairs()]
    for b in (1, 2, 3):
        for p in range(1, 9):
            jobs.append((f'new_{b}-{p}_burst', RAWN / f'{b}-{p}-0.tif', RAWN / f'{b}-{p}-1.tif', 'new_burst')); jobs.append((f'new_{b}-{p}_remount', RAWN / f'{b}-{p}-0.tif', RAWN / f'{b}-{p}-3.tif', 'new_remount'))
    from multiprocessing import Pool
    with Pool(8) as pool: rows = pool.map(one, jobs, chunksize=1)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'D4_period_on_pairs.csv', index=False); pd.set_option('display.width', 250)
    print(t[t.kind.isin(['burst', 'step'])].round(3).to_string())
    print(t.groupby('kind').agg(n=('label', 'size'), best_is_zero=('best_is_zero', 'sum'), margin_min=('margin_zero_vs_other', 'min'), margin_med=('margin_zero_vs_other', 'median'), n_margin_lt_0_1=('margin_zero_vs_other', lambda x: int((x < 0.1).sum())), shift_nonzero=('period_shift', lambda x: int((x != '(0, 0)').sum()))).round(3))
