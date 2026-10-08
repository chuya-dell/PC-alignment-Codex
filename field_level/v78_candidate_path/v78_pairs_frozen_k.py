"""v78 criterion 1 on the zero-truth pairs at the frozen k (own MAD per pair, NO scar exclusion = conservative).  Saved as a script after the critique."""
import sys, glob, os
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v78_candidate_path'))
import numpy as np, pandas as pd
import v78_calibrate as K
V78 = K.V78; P = ROOT / 'data/results/v75_zero_truth_pairs/'
cal = pd.read_csv(V78 / 'calibration_k.csv'); kk = {rd: float(cal[(cal.readout == rd) & (cal.scope == 'all_blanks') & (cal.limit == 'main2')].k.iloc[0]) for rd in ['S0', 'S2', 'S5']}
files = sorted(glob.glob(str(P / 'real/*.npz'))) + sorted(glob.glob(str(P / 'synth/*.npz'))); rows = []
for f in files:
    z = np.load(f); lab = os.path.basename(f)[:-4]
    for rd, k in kk.items():
        d = z[rd].astype(float); v = d[np.isfinite(d)]; med = np.median(v); mad = 1.4826 * np.median(abs(v - med))
        rows.append(dict(pair=lab, readout=rd, k=k, fp=float((v > med + k * mad).sum() / len(v) * 91000), fp_raw=int((v > med + k * mad).sum()), n=len(v)))
t = pd.DataFrame(rows); t.to_csv(V78 / 'zero_truth_pairs_at_frozen_k.csv', index=False); print('frozen k', kk)
for rd in kk:
    q = t[t.readout == rd]; print(rd, 'pairs', len(q), 'mean', round(q.fp.mean(), 2), 'p95', round(q.fp.quantile(.95), 2), 'max', round(q.fp.max(), 2), 'frac==0', round((q.fp == 0).mean(), 2), 'frac<=2', round((q.fp <= 2).mean(), 2))
