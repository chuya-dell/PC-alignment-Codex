"""v83 part A-4 (2/3): align the 5 needle-scar images (all 10 pairs, a=earlier name) with the current standard route C1 (v75_pair_lib.analyse_pair) in two variants:
 'masked': scar pixels (cross-scar + needle gouge masks) are replaced by a smooth background BEFORE any alignment step (the scar is not an input of the alignment),
 'plain' : the images as they are (standard route without scar masking; for comparison).
Saves per pair/variant: A (feature affine), Llin, tL (lattice map actually used for reading), U, period shift, corr, S5 summary.  usage: python v83_needle_align.py"""
import sys, json, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, cv2, tifffile, itertools
import v75_pair_lib as P
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_傷'); OUT = ROOT / 'data/results/v83_new_shots_261008/needle'
NAMES = ['6', '6-1', '6-2', '6-3', '6-4']


def load_mask(n):
    with np.load(OUT / f'align_mask_{n}.npz') as z:
        shp = tuple(z['shape']); f = lambda k: np.unpackbits(z[k])[:np.prod(shp)].reshape(shp).astype(bool)
        return f('cross') | f('gouge')


def neutralise(im, m):
    w = (~cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.float32)
    num = cv2.GaussianBlur(im * w, (0, 0), 20); den = cv2.GaussianBlur(w, (0, 0), 20)
    bg = num / np.maximum(den, 1e-6)
    # keep the pillar texture's mean level; add nothing else
    out = im.copy(); mm = w == 0; out[mm] = bg[mm]
    return out


def one(job):
    i, j, var = job; tag = f'{NAMES[i]}__{NAMES[j]}__{var}'; dst = OUT / 'align' / f'{tag}.npz'
    if dst.exists(): return dict(tag=tag, ok=True)
    cv2.setNumThreads(1)
    try:
        a = tifffile.imread(RAW / f'{NAMES[i]}.tif').astype(np.float32); b = tifffile.imread(RAW / f'{NAMES[j]}.tif').astype(np.float32)
        if var == 'masked':
            a = neutralise(a, load_mask(NAMES[i])); b = neutralise(b, load_mask(NAMES[j]))
        res = P.analyse_pair(a, b)
        S = res['S5']; v = S[np.isfinite(S)]
        np.savez_compressed(dst, A=res['A'], Llin=res['Llin'], tL=res['tL'], U=res['U'], period_shift=res['period_shift'], period_corr_zero=res['period_corr_zero'], period_corr_best=res['period_corr_best'],
                            frac_corr=res['frac_corr_px'], corner=res['corner_disagree_px'], ok_frac=float(res['ok'].mean()), sd_S5=float(v.std()), mad_S5=float(1.4826 * np.median(abs(v - np.median(v)))))
        return dict(tag=tag, ok=True)
    except Exception as e:
        return dict(tag=tag, ok=False, error=repr(e), trace=traceback.format_exc()[-700:])


if __name__ == '__main__':
    (OUT / 'align').mkdir(exist_ok=True)
    jobs = [(i, j, v) for i, j in itertools.combinations(range(5), 2) for v in ('masked', 'plain')]
    from multiprocessing import Pool
    with Pool(8) as pool:
        for r in pool.imap_unordered(one, jobs): print(r, flush=True)
