"""v86: is the substitution rate noise-limited?  On the 24 reshoot fields (2026-10-08, same view imaged twice without touching: -0 and -1): v70 estimator (centers_smooth + sub_flags 'main') on -0 alone,
-1 alone and the average of the two (N=2).  If the substituted share falls with averaging, part of the failures is noise; the part that stays is not.  Also the share by failing reason.
usage: python v86_noise_limited.py [--workers K]"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, tifffile, cv2
import v70_lib as L
import v70_centers2 as C
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_z'); OUT = ROOT / 'data/results/v86_correspondence_mf'


def stats(im):
    cf = C.centers_smooth(im); d = dict(ctr=cf['ctr'].astype(float), pred=cf['pred'], amp=cf['amp'], wid=cf['wid'], aloc=cf['aloc']); bad = C.sub_flags(d, 'main')
    c = C.CRIT2['main']; shift = np.linalg.norm(d['ctr'] - d['pred'], axis=1)
    return dict(sub=float(bad.mean()), fail_amp=float((d['amp'] < c['amp_frac'] * d['aloc']).mean()), fail_shift=float((shift > c['shift_max']).mean()), shift_med=float(np.median(shift)), n=len(bad),
                amp_iqr_rel=float(np.subtract(*np.percentile(d['amp'] / d['aloc'], [75, 25]))))


def one(job):
    b, p = job; cv2.setNumThreads(1)
    a = tifffile.imread(RAW / f'{b}-{p}-0.tif').astype(np.float32); c = tifffile.imread(RAW / f'{b}-{p}-1.tif').astype(np.float32)
    r = dict(board=b, pos=p)
    for nm, im in (('f0', a), ('f1', c), ('avg2', (a + c) / 2)):
        for k, v in stats(im).items(): r[f'{nm}_{k}'] = v
    return r


if __name__ == '__main__':
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 8
    from multiprocessing import Pool
    with Pool(workers) as p: rows = p.map(one, [(b, q) for b in (1, 2, 3) for q in range(1, 9)])
    t = pd.DataFrame(rows); t.to_csv(OUT / 'noise_limited.csv', index=False); pd.set_option('display.width', 250)
    t['f01_sub'] = (t.f0_sub + t.f1_sub) / 2
    print(t[['f0_sub', 'f1_sub', 'avg2_sub', 'f0_fail_amp', 'avg2_fail_amp', 'f0_fail_shift', 'avg2_fail_shift', 'f0_shift_med', 'avg2_shift_med']].describe().loc[['mean', '50%']].round(4).T)
    print('median single-frame sub %.4f -> avg2 sub %.4f ; relative drop %.3f' % (t.f01_sub.median(), t.avg2_sub.median(), 1 - t.avg2_sub.median() / t.f01_sub.median()))
