"""v83 part A-3 (focus): per-image focus metrics of the 72 reshoot images with the SAME functions as the Z-step calibration (v73_series.image_metrics),
and a rough mapping of the -0 vs -3 focus difference onto the Z-step ordinal axis (260901_p50_Zstep, file number as ordinal only).
usage: python v83_focus_est.py"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v73_focus_map', 'field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, tifffile, cv2
import v73_series as V
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_z'); OUT = ROOT / 'data/results/v83_new_shots_261008'


def one(a):
    cv2.setNumThreads(1); b, p, k = a
    im = tifffile.imread(RAW / f'{b}-{p}-{k}.tif').astype(np.float32); r = V.image_metrics(im); r.update(board=b, pos=p, kind=k); return r


if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [(b, p, k) for b in (1, 2, 3) for p in range(1, 9) for k in (0, 1, 3)]
    with Pool(8) as pool: rows = pool.map(one, jobs)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'A3_focus_metrics_72.csv', index=False)
    w = t.pivot_table(index=['board', 'pos'], columns='kind', values=['pillar_amp', 'lattice_peak', 'pillar_width', 'contrast_sd'])
    w.to_csv(OUT / 'A3_focus_metrics_wide.csv')
    pd.set_option('display.width', 250); print(w.round(4).to_string())
