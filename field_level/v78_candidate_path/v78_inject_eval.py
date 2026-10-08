"""v78 step H criterion 4: injection recovery and photometric bias at the frozen (calibrated) thresholds, relative to the current standard."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
OUT = ROOT / 'data/results/v77_signal_injection_P2'; V78 = ROOT / 'data/results/v78_candidate_path'
t = pd.read_csv(OUT / 'injection_rows.csv'); cal = pd.read_csv(V78 / 'calibration_k.csv')
def kfor(rd, lim): return float(cal[(cal.readout == rd) & (cal.scope == 'all_blanks') & (cal.limit == lim)].k.iloc[0])
pd.set_option('display.width', 250)
rows = []
for lim in ('main2', 'old10'):
    for rd in ['S0', 'S2', 'S5', 'S5P']:
        k = kfor(rd, lim); q = t[t.readout == rd]
        for A in (3, 5, 10):
            for d in ('dim', 'bright'):
                g = q[(q.A == A) & (q.dir == d)]
                v = g.snr.to_numpy(); ok = g.valid.to_numpy()
                rows.append(dict(limit=lim, readout=rd, k=k, A=A, dir=d, n=len(g), valid=ok.mean(),
                                 rec_pos=float(np.mean(ok & (v > k))), rec_sym=float(np.mean(ok & (np.abs(v) > k))), rec_pos_among_valid=float(np.mean(v[ok] > k)) if ok.any() else np.nan))
r = pd.DataFrame(rows); r.to_csv(V78 / 'injection_recovery_at_frozen_k.csv', index=False)
for lim in ('main2', 'old10'):
    print('== limit', lim, '(recovery of injected pillars; positive-side detection for dim, symmetric for bright)')
    p = r[r.limit == lim]; piv = p.pivot_table(index=['readout', 'k'], columns=['dir', 'A'], values=['rec_pos', 'rec_sym', 'valid']).round(3)
    print(p[(p.dir == 'dim')].pivot_table(index=['readout', 'k'], columns='A', values='rec_pos').round(3).to_string()); print(p[(p.dir == 'bright')].pivot_table(index=['readout', 'k'], columns='A', values='rec_sym').round(3).to_string())
# photometric bias: raw and gain-calibrated (calibrate gain on half of the cases, evaluate on the other half)
cases = sorted(t.label.unique()); cal_cases = cases[::2]; test_cases = cases[1::2]
out = []
for rd in ['S0', 'S2', 'S5', 'S5P']:
    q = t[(t.readout == rd) & t.valid]
    gain = q[q.label.isin(cal_cases)].ratio.median(); tt = q[q.label.isin(test_cases)]
    adj = tt.ratio / gain
    by = tt.assign(adj=adj).groupby(['A', 'dir', 'zone', 'pattern']).adj.median()
    out.append(dict(readout=rd, raw_bias_median=q.ratio.median(), gain=gain, calibrated_median_test=adj.median(), calibrated_condition_range=f'{by.min():.2f}-{by.max():.2f}', frac_conditions_within_10pct=float(((by > .9) & (by < 1.1)).mean()),
                    iqr_of_ratio=float(q.ratio.quantile(.75) - q.ratio.quantile(.25))))
b = pd.DataFrame(out); b.to_csv(V78 / 'photometric_bias.csv', index=False); print(b.round(3).to_string(index=False))
