"""v77 summary: recovery (positive-only and symmetric) and photometric bias by readout, amplitude, direction, zone, pattern, test kind."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parents[2] / 'data/results/v77_signal_injection_P2'
t = pd.read_csv(OUT / 'injection_rows.csv'); fp = pd.read_csv(OUT / 'injection_fp.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
KS = [3, 4, 5, 6, 8]
print('sites', len(t) // 6, 'rows', len(t), 'kinds', t.kind.value_counts().to_dict())
valid = t[t.valid]
print('valid fraction by readout', t.groupby('readout').valid.mean().round(3).to_dict())
# recovery: dim (positive direction), by readout and amplitude; for k in KS
rows = []
for rd, g in t.groupby('readout'):
    for A in [3, 5, 10]:
        for d in ['dim', 'bright']:
            q = g[(g.A == A) & (g.dir == d)]
            r = dict(readout=rd, A=A, dir=d, n=len(q), valid=q.valid.mean(), bias_median=q.ratio.median(), bias_mean=q.ratio.mean())
            for k in KS:
                r[f'pos_k{k}'] = q[f'det_pos_k{k}'].mean(); r[f'sym_k{k}'] = q[f'det_sym_k{k}'].mean()
            rows.append(r)
s = pd.DataFrame(rows); s.to_csv(OUT / 'recovery_by_readout_amp_dir.csv', index=False)
print('== DIM (positive-direction) recovery (positive-only detection), by readout x amplitude (all zones, patterns, kinds)')
print(s[s.dir == 'dim'][['readout', 'A', 'valid', 'bias_median'] + [f'pos_k{k}' for k in KS]].round(3).to_string(index=False))
print('== BRIGHT (negative-direction): positive-only vs symmetric recovery')
print(s[s.dir == 'bright'][['readout', 'A', 'bias_median', 'pos_k4', 'sym_k4', 'sym_k6', 'sym_k8']].round(3).to_string(index=False))
z = t.groupby(['readout', 'zone', 'dir']).agg(bias=('ratio', 'median'), valid=('valid', 'mean'), rec_k4=('det_sym_k4', 'mean'), rec_k6=('det_sym_k6', 'mean')).reset_index()
print('== by zone (symmetric detection)'); print(z[z.readout.isin(['S0', 'S3', 'S5'])].round(3).to_string(index=False))
z = t.groupby(['readout', 'pattern']).agg(bias=('ratio', 'median'), rec_k4=('det_sym_k4', 'mean'), rec_k6=('det_sym_k6', 'mean')).reset_index(); print('== by pattern'); print(z[z.readout.isin(['S0', 'S3', 'S5'])].round(3).to_string(index=False))
z = t.groupby(['readout', 'kind']).agg(bias=('ratio', 'median'), valid=('valid', 'mean'), rec_k4=('det_sym_k4', 'mean'), rec_k6=('det_sym_k6', 'mean')).reset_index(); print('== by test kind'); print(z[z.readout.isin(['S0', 'S3', 'S5'])].round(3).to_string(index=False))
print('== false positives away from injected sites (count/91k, median over cases)'); print(fp.groupby(['readout', 'k']).fp_count91k.median().unstack().round(1))
