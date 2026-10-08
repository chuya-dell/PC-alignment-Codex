"""v74: scar-distance profile with substrate-level bootstrap 95% intervals and the exclusion width (smallest distance from which every farther class is inside the far-value upper bound).
(Code saved after independent critique; earlier numbers came from an inline script.)  Reads scar_distance_by_field.csv."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
O = ROOT / 'data/results/v74_scar_distance/'
t = pd.read_csv(O / 'scar_distance_by_field.csv', dtype={'date': str, 'board': str})
names = ['in_core(>2per)', 'in_depth1-2', 'in_depth0-1', 'out_0-1', 'out_1-2', 'out_2-4', 'out_4-8', 'out_>8']
rng = np.random.default_rng(20261008)
def boot(df, readout, nb=2000):
    g = df[df.readout == readout]; subs = g.substrate.unique()
    P = g.groupby(['substrate', 'cls'])[['n', 'pos']].sum().unstack('cls').reindex(subs)
    n = P['n'][names].to_numpy(float); pos = P['pos'][names].to_numpy(float); est = pos.sum(0) / n.sum(0); bs = []
    for _ in range(nb):
        idx = rng.integers(0, len(subs), len(subs)); bs.append(pos[idx].sum(0) / np.maximum(n[idx].sum(0), 1))
    bs = np.array(bs); return est, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0), len(subs)
rows = []
for name, sel in [('blank', t.group == 'blank'), ('all_nonfixed', ~t.fixed29), ('fixed29', t.fixed29)]:
    for rd in ['S0', 'S1', 'S3']:
        est, lo, hi, ns = boot(t[sel], rd); far_hi = hi[-1]; far = est[-1]; width = None
        for k in range(3, 8):
            if all(est[j] <= far_hi for j in range(k, 8)): width = names[k]; break
        for j, nm in enumerate(names): rows.append(dict(set=name, readout=rd, cls=nm, rate=est[j], lo=lo[j], hi=hi[j], ratio_to_far=est[j] / far, substrates=ns))
        rows.append(dict(set=name, readout=rd, cls='EXCLUSION_FIRST_CLASS_WITHIN_FAR_CI', rate=np.nan, lo=np.nan, hi=far_hi, ratio_to_far=np.nan, substrates=ns, note=str(width)))
r = pd.DataFrame(rows); r.to_csv(O / 'scar_distance_ci.csv', index=False)
pd.set_option('display.width', 200)
for name in ['blank', 'fixed29']:
    for rd in ['S1', 'S3']:
        q = r[(r.set == name) & (r.readout == rd)]; print('==', name, rd); print(q[['cls', 'rate', 'lo', 'hi', 'ratio_to_far', 'note']].round(5).to_string(index=False))
