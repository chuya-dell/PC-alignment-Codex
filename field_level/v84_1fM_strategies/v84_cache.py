"""v84 (part B/E common): compact per-field cache of the development fields (632) under candidate C1: S5 and S5P deltas, scar distance, positions, abnormal-pillar flag.
For each field: centre = median, scale = 1.4826 MAD of the valid, scar-excluded (outside 2 periods of the v57 mask) S5 delta (the frozen C1 rule, v78_calibrate.load_field).
Saved: data/results/v84_1fM_strategies/cache/{fid}.npz with z5 (all pillars, NaN where invalid, (S5-med)/sigma), z5p, sd_periods, ctr, excl (kept pillars), abn (post-fit failed, P-readout valid)
and field_meta.csv.  usage: python v84_cache.py"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V72 = ROOT / 'data/results/v72_real_field_readout'; V78 = ROOT / 'data/results/v78_candidate_path'
OUT = ROOT / 'data/results/v84_1fM_strategies'; (OUT / 'cache').mkdir(parents=True, exist_ok=True)
led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}); rows = []
for r in led.itertuples():
    try:
        z72 = np.load(V72 / 'fields' / f'{r.fid}.npz'); z78 = np.load(V78 / 'fields' / f'{r.fid}.npz')
    except Exception as e:
        rows.append(dict(fid=r.fid, ok=False)); continue
    S5 = z72['S5'].astype(np.float64); S5P = z78['S5P'].astype(np.float64); sd = z78['sd_periods'].astype(np.float64)
    okL = np.isfinite(S5); okP = np.isfinite(S5P); excl = np.isfinite(sd) & (sd > 2.0)
    v = S5[okL & excl]
    if len(v) < 5000:
        rows.append(dict(fid=r.fid, ok=False, n_valid=len(v))); continue
    med = np.median(v); sig = 1.4826 * np.median(abs(v - med))
    z5 = np.where(okL, (S5 - med) / sig, np.nan).astype(np.float32); z5p = np.where(okP, (S5P - med) / sig, np.nan).astype(np.float32)
    abn = okP & ~okL
    np.savez_compressed(OUT / 'cache' / f'{r.fid}.npz', z5=z5, z5p=z5p, sd=sd.astype(np.float32), ctr=z72['ctr_xy'].astype(np.float32), excl=excl, abn=abn)
    rows.append(dict(fid=r.fid, ok=True, date=r.date, board=r.board, field=r.field, group=r.group, fixed29=r.fixed29, conc_M=r.conc_M, condition=r.condition, n_valid=int((okL & excl).sum()), n_nodes=len(S5),
                     median=med, sigma=sig, sd_all=float(v.std()), n_abn=int((abn & excl).sum())))
m = pd.DataFrame(rows); m.to_csv(OUT / 'field_meta.csv', index=False); print(m.ok.value_counts().to_dict(), m.groupby('group').size().to_dict()); print(m[m.ok].sigma.describe().round(4))
