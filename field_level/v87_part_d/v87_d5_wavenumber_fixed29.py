"""v87 D5: real-data check of the band wave-number prediction on the fixed-29 fields (S0 current standard).
Prediction per field (same rule as v75_wavenumber_match): dK = K_real - K_ideal (reciprocal vectors of the pre-image lattice fitted by v70 fit_lattice vs the standard sampling lattice lattice_from_fft(pre, 7.286)),
candidates h1*dK1 + h2*dK2 with |h1|+|h2| <= 3, period = 1/|k| in 100..900 px, band direction = wave-vector angle + 90 deg.  Observed: S0 band peak (period_px, axis_deg) from v72 metrics_by_field_v2
(thr G, readout S0) for fields with band_power above a cut.  Match = period within +-10 percent and direction within +-10 deg.  Chance: per-field uniform null (log-uniform period 128..1024, uniform direction)
evaluated against THAT field's own predicted set.   usage: python v87_d5_wavenumber_fixed29.py"""
import sys, itertools, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2
import v70_lib as L
from shared.lattice_indexing import lattice_from_fft
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V72 = ROOT / 'data/results/v72_real_field_readout'; OUT = ROOT / 'data/results/v87_part_d'


def predicted(pre):
    lat = lattice_from_fft(pre, 7.286); Ki = np.linalg.inv(np.asarray(lat.basis))
    model, quad, info = L.fit_lattice(pre); Kr = np.linalg.inv(model[1:].T); best = None
    cand = [s_ * v for s_ in (1, -1) for v in (Kr[0], Kr[1], Kr[0] + Kr[1], Kr[0] - Kr[1])]       # the six shortest reciprocal vectors of the hexagonal lattice (+- each)
    for u in cand:
        for v in cand:
            if abs(np.cross(u, v)) < 0.5 * abs(np.cross(Kr[0], Kr[1])): continue
            Kt = np.array([u, v]); d = np.linalg.norm(Kt - Ki)
            if best is None or d < best[0]: best = (d, Kt)
    dK = best[1] - Ki; pr = []
    for h1 in range(-3, 4):
        for h2 in range(-3, 4):
            if (h1, h2) == (0, 0) or abs(h1) + abs(h2) > 3: continue
            k = h1 * dK[0] + h2 * dK[1]; per = 1 / max(np.linalg.norm(k), 1e-9)
            if 100 <= per <= 900: pr.append((per, (np.degrees(np.arctan2(k[1], k[0])) + 90) % 180, f'{h1},{h2}'))
    pr.append((618.3, 94.0, 'rounding_y4'))
    return pr, float(np.linalg.norm(best[1] - Ki)), info['pitch_px']


def match(pr, per, axis, tp=0.10, ta=10):
    for pp, aa, lab in pr:
        if abs(per - pp) / pp <= tp and abs((axis - aa + 90) % 180 - 90) <= ta: return lab
    return None


def one(rec):
    cv2.setNumThreads(1); pre = L.read(rec['pre_path']); pr, dKn, pitch = predicted(pre); return rec['fid'], pr, dKn, pitch


if __name__ == '__main__':
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}); fx = led[led.fixed29]
    from multiprocessing import Pool
    with Pool(6) as p: res = p.map(one, fx.to_dict('records'))
    pred = {f: (pr, dk, pitch) for f, pr, dk, pitch in res}
    m = pd.read_csv(V72 / 'metrics_by_field_v2.csv', dtype={'date': str, 'board': str}); m = m[(m.readout == 'S0') & (m.thr == 'G') & m.fid.isin(fx.fid)]
    rng = np.random.default_rng(2); rows = []
    for r in m.itertuples():
        pr, dk, pitch = pred[r.fid]; mt = match(pr, r.period_px, r.axis_deg)
        ch = np.mean([match(pr, np.exp(rng.uniform(np.log(128), np.log(1024))), rng.uniform(0, 180)) is not None for _ in range(4000)])
        rows.append(dict(fid=r.fid, band_power=r.band_power, strength=r.strength, period_px=r.period_px, axis_deg=r.axis_deg, matched=mt, chance=ch, n_pred=len(pr), dK_norm=dk, pitch=pitch))
    t = pd.DataFrame(rows); t.to_csv(OUT / 'D5_wavenumber_fixed29.csv', index=False); pd.set_option('display.width', 250); print(t.round(3).to_string())
    for cut in (0, 100, 1000):
        q = t[t.band_power > cut]; print('band_power >', cut, 'n', len(q), 'matched', int(q.matched.notna().sum()), 'expected by chance %.1f' % q.chance.sum())
    json.dump({f: dict(n_pred=len(p[0]), dK_norm=p[1]) for f, p in pred.items()}, open(OUT / 'D5_predictions.json', 'w'))
