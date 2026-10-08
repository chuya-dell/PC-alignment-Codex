"""v75: do the S0 band peaks of the zero-truth synthetic pairs sit at the predicted moire wave vectors (ideal 7.286 grid vs real lattice)?  Saved as a script after the critique.
Prediction: dK = K_real - K_ideal (reciprocal vectors, cycles/px), candidates h1*dK1+h2*dK2 (|h1|+|h2|<=3), period = 1/|k| in 100..900 px, direction = wave-vector angle + 90 deg, plus the
rounding beat (period 618 px, 94 deg).  Match: period within +-10 percent and direction within +-10 deg.  Chance level: uniform null (log-uniform period 128..1024, uniform direction), and the
number of DISTINCT observed peaks (the 18 pairs share one image and one moire, so they are not independent)."""
import sys, itertools
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls', 'field_level/v72_real_field_readout'): sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd
import field_control_common as M60, v70_lib as L
from shared.lattice_indexing import lattice_from_fft
O = ROOT / 'data/results/v75_zero_truth_pairs/'
p = M60.folder('260904_p50_repeat'); a = M60.read(p / '1.tif')
lat = lattice_from_fft(a, 7.286); Ki = np.linalg.inv(np.asarray(lat.basis))
model, quad, info = L.fit_lattice(a); Kr = np.linalg.inv(model[1:].T); best = None
for perm in itertools.permutations([0, 1]):
    for s0, s1 in itertools.product([1, -1], [1, -1]):
        Kt = np.array([s0 * Kr[perm[0]], s1 * Kr[perm[1]]]); d = np.linalg.norm(Kt - Ki)
        if best is None or d < best[0]: best = (d, Kt)
dK = best[1] - Ki; pr = []
for h1 in range(-3, 4):
    for h2 in range(-3, 4):
        if (h1, h2) == (0, 0) or abs(h1) + abs(h2) > 3: continue
        k = h1 * dK[0] + h2 * dK[1]; per = 1 / np.linalg.norm(k)
        if 100 <= per <= 900: pr.append((per, (np.degrees(np.arctan2(k[1], k[0])) + 90) % 180, f'{h1},{h2}'))
pr.append((618.3, 94.0, 'rounding_y4'))
def match(per, axis, tol_p=0.10, tol_a=10):
    for pp, aa, lab in pr:
        if abs(per - pp) / pp <= tol_p and abs((axis - aa + 90) % 180 - 90) <= tol_a: return lab
    return None
t = pd.read_csv(O / 'synth_pairs_metrics.csv'); t = t[t.thr == 'each']
print('predicted:', sorted({(round(x), round(y)) for x, y, z in pr}))
for rd in ['S0', 'S1', 'S2', 'S3', 'S5']:
    q = t[t.readout == rd]; hit = [match(r.period_px, r.axis_deg) for r in q.itertuples()]; strong = q.band_power > 100
    peaks = {(round(r.period_px / 10) * 10, r.axis_deg) for r in q.itertuples()}
    print(rd, 'pairs', len(q), 'band_power>100:', int(strong.sum()), 'matched among them:', int(sum(h is not None for h, s in zip(hit, strong) if s)), '| all matched', sum(h is not None for h in hit), '| distinct peaks', len(peaks))
rng = np.random.default_rng(1); ch = np.mean([match(np.exp(rng.uniform(np.log(128), np.log(1024))), rng.uniform(0, 180)) is not None for _ in range(20000)]); print('chance per observation (uniform null)', round(ch, 3))
q = t[t.readout == 'S0']; print(pd.DataFrame(dict(label=q.label, period=q.period_px.round(0), axis=q.axis_deg, bp=q.band_power.round(0), match=[match(r.period_px, r.axis_deg) for r in q.itertuples()])).to_string(index=False))
