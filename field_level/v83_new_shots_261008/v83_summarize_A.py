"""v83 part A-3 summary: false positives per field for -0/-1 (burst, nothing touched) and -0/-3 (re-mounted + re-focused), by board (n=3 independent), position, scar distance.
Rule = frozen candidate C1: per-field centre = median, scale = 1.4826*MAD of the valid, scar-excluded (outside 2 periods of the mask) S5 delta; threshold = centre + k*scale, k = 10.5 (frozen); k = 4, 6, 8 as references.
Counts are positive side (same convention as v78_calibrate) and two-sided; per field raw count and per 91k pillars.  Abnormal pillars = post-side fit failed (okP & ~okL) with |S5P - centre| > k*scale.
Nothing is tuned here.  usage: python v83_summarize_A.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'data/results/v83_new_shots_261008'
KS = [4, 6, 8, 10.5]; REF = 91000.0; PER = 7.37
CLS = [('in_mask', -1e9, 0), ('out_0-2', 0, 2), ('out_2-4', 2, 4), ('out_4-8', 4, 8), ('out_>8', 8, 1e9)]
rows = []; drows = []
for b in (1, 2, 3):
    for p in range(1, 9):
        for kind in (1, 3):
            z = np.load(OUT / 'pairs' / f'{b}-{p}_{kind}.npz'); S = z['S5'].astype(float); SP = z['S5P'].astype(float); sd = z['sd_periods'].astype(float)
            okL = np.isfinite(S); okP = np.isfinite(SP)
            exc = np.isfinite(sd) & (sd > 2.0) | (~z['has_mask'] & True)   # no mask -> nothing excluded (not happening: all 48 have masks)
            v = S[okL & exc]; med = np.median(v); sig = 1.4826 * np.median(abs(v - med)); n = len(v)
            r = dict(board=b, pos=p, kind=f'-0/-{kind}', n_valid=n, n_all_valid=int(okL.sum()), frac_valid=n / len(S), sigma=sig, median=med, sd=v.std(), mask_frac=float((sd <= 0).mean()))
            ab_base = okP & ~okL & exc
            r['n_abnormal_candidates'] = int(ab_base.sum())
            for k in KS:
                pos_ = int((v > med + k * sig).sum()); neg_ = int((v < med - k * sig).sum())
                r[f'pos_k{k}'] = pos_; r[f'two_k{k}'] = pos_ + neg_; r[f'pos91k_k{k}'] = pos_ / n * REF
                ab = ab_base & (np.abs(SP - med) > k * sig); r[f'abn_k{k}'] = int(ab.sum())
                r[f'abn_pos_k{k}'] = int((ab_base & (SP - med > k * sig)).sum())
            rows.append(r)
            # by scar distance (all valid pillars, not excluded), k=4 and 8 (more events)
            for name, lo, hi in CLS:
                sel = okL & (sd > lo) & (sd <= hi) if name != 'in_mask' else okL & (sd <= 0)
                m = sel.sum()
                drows.append(dict(board=b, pos=p, kind=f'-0/-{kind}', cls=name, n=int(m), **{f'pos_k{k}': int((S[sel] > med + k * sig).sum()) for k in (4, 6, 8)},
                                  **{f'abn_k{k}': int((okP & ~okL & (sd > lo) & (sd <= hi) & (np.abs(SP - med) > k * sig)).sum()) for k in (4, 8)},
                                  n_abn_cand=int((okP & ~okL & (sd > lo) & (sd <= hi)).sum())))
t = pd.DataFrame(rows); t.to_csv(OUT / 'A3_fp_by_field.csv', index=False)
d = pd.DataFrame(drows); d.to_csv(OUT / 'A3_fp_by_scar_class.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(t.groupby(['kind', 'board']).agg(sigma=('sigma', 'median'), frac_valid=('frac_valid', 'mean'), mask=('mask_frac', 'mean'),
      k105_mean=('pos91k_k10.5', 'mean'), k105_max=('pos91k_k10.5', 'max'), k8_mean=('pos91k_k8', 'mean'), k6_mean=('pos91k_k6', 'mean'), k4_mean=('pos91k_k4', 'mean'), abn8=('abn_k8', 'mean'), abn4=('abn_k4', 'mean'), abncand=('n_abnormal_candidates', 'mean')).round(3))
print(t.groupby('kind').agg(sigma=('sigma', 'median'), k105_mean=('pos91k_k10.5', 'mean'), le2_k105=('pos91k_k10.5', lambda x: (x <= 2).mean()), k8=('pos91k_k8', 'mean'), le2_k8=('pos91k_k8', lambda x: (x <= 2).mean()),
      k4=('pos91k_k4', 'mean'), abn8=('abn_k8', 'mean'), abn4=('abn_k4', 'mean')).round(3))
