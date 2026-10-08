"""v78: injection recovery at the frozen thresholds including the 15-40 sigma extension (reads v77_signal_injection_P2 and _P3_high).  Saved as a script after the critique."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
V78 = ROOT / 'data/results/v78_candidate_path/'
cal = pd.read_csv(V78 / 'calibration_k.csv')
def kfor(rd, lim): return float(cal[(cal.readout == rd) & (cal.scope == 'all_blanks') & (cal.limit == lim)].k.iloc[0])
t = pd.concat([pd.read_csv(ROOT / 'data/results/v77_signal_injection_P2/injection_rows.csv'), pd.read_csv(ROOT / 'data/results/v77_signal_injection_P3_high/injection_rows.csv')])
rows = []
for lim in ('main2', 'old10'):
    for rd in ['S0', 'S5', 'S5P']:
        k = kfor(rd, lim); q = t[t.readout == rd]
        for d in ('dim', 'bright'):
            for A in sorted(q.A.unique()):
                g = q[(q.A == A) & (q.dir == d)]; v = g.snr.to_numpy(); ok = g.valid.to_numpy()
                rows.append(dict(limit=lim, readout=rd, k=k, dir=d, A=A, valid=ok.mean(), recovery=np.mean(ok & ((v > k) if d == 'dim' else (np.abs(v) > k))), median_ratio=g.ratio.median()))
r = pd.DataFrame(rows); r.to_csv(V78 / 'injection_recovery_extended.csv', index=False); pd.set_option('display.width', 200)
for lim in ('main2', 'old10'):
    print('==', lim)
    for d in ('dim', 'bright'): print(d); print(r[(r.limit == lim) & (r.dir == d)].pivot_table(index=['readout', 'k'], columns='A', values='recovery').round(2).to_string())
    for rd in ['S0', 'S5', 'S5P']:
        for d in ('bright', 'dim'):
            q = r[(r.limit == lim) & (r.readout == rd) & (r.dir == d) & (r.recovery >= 0.5)]; print(lim, rd, d, 'A50 ~', q.A.min() if len(q) else '>40')
print(r[(r.readout == 'S5') & (r.limit == 'main2')][['dir', 'A', 'valid', 'recovery', 'median_ratio']].round(3).to_string(index=False))
