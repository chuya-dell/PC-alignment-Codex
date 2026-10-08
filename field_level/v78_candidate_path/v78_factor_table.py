"""v78: one-factor-at-a-time table.  Counts per field (91k pillars) above median + k*1.4826*MAD, outside the scar mask + 2 periods, valid pillars; blanks (69) and fixed29 (29).
Readouts: S0 standard, S1 lattice map, S2 +bilinear, S3 actual centres (integer), S5 actual centres + aperture.  Band power at the whole-field 3SD threshold from v72 set summary."""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v78_candidate_path'))
import numpy as np, pandas as pd
import v78_calibrate as K
V70, V72, V78 = K.V70, K.V72, K.V78
led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
sets = json.load(open(V72 / 'sets.json')); rows = []
for r in led.itertuples():
    if r.fid not in set(sets['all_blank']) | set(sets['fixed29']): continue
    z72 = np.load(V72 / 'fields' / f'{r.fid}.npz'); z78 = np.load(V78 / 'fields' / f'{r.fid}.npz')
    ex_std = z72['in_mask_std'] | (z72['dist_std'] < 2 * K.PER); ex_c = ~(z78['sd_periods'] > 2)
    for rd, ex in (('S0', ex_std), ('S1', ex_std), ('S2', ex_std), ('S3', ex_c), ('S5', ex_c)):
        d = z72[rd].astype(float); ok = np.isfinite(d) & ~ex; v = d[ok]
        if len(v) < 5000: continue
        med = np.median(v); mad = 1.4826 * np.median(abs(v - med))
        for k in (4, 6, 8): rows.append(dict(fid=r.fid, group=r.group, fixed29=r.fixed29, readout=rd, k=k, count=float((v > med + k * mad).sum() / len(v) * 91000), mad=float(mad), frac_gt_0p2=float((v > med + 0.2).mean() * 91000)))
t = pd.DataFrame(rows); t.to_csv(V78 / 'factor_table_counts.csv', index=False)
sm = pd.read_csv(V72 / 'set_summary_v2.csv')
pd.set_option('display.width', 200)
for name, sel in (('blank69', t.group == 'blank'), ('fixed29', t.fixed29)):
    print('==', name, '(median counts per field at k=4/6/8)'); print(t[sel].groupby(['readout', 'k'])['count'].median().unstack().round(1).to_string())
for name, sel in (('blank69', t.group == 'blank'), ('fixed29', t.fixed29)):
    q = t[sel & (t.k == 4)]; print('==', name, 'robust SD (MAD) median and count per field above an ABSOLUTE threshold median+0.2 (box units)'); print(q.groupby('readout')[['mad', 'frac_gt_0p2']].median().round(4).to_string())
print(sm[(sm.set == 'fixed29') & (sm.thr == 'G') & sm.readout.isin(['S0', 'S1', 'S2', 'S3', 'S5'])][['readout', 'band_power_median']].round(0).to_string(index=False))
