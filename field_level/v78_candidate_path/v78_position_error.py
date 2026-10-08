"""v78 step H criterion 6: position error of the registration under known deformations (synthetic shifts and rotations of the burst pair).
Error of the INCREMENT relative to the zero-shift synthetic pair: [map_pair(p) - map_base(p)] - [truth(p) - p].
Maps: A (feature-based affine of the current standard) and L (lattice-implied linear map with anchored translation).  Evaluated at all pre nodes."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, cv2
O = ROOT / 'data/results/v75_zero_truth_pairs/synth/'
c0 = np.array([1024., 1022.])
def maps(z):
    A = z['A']; L = z['Llin']; t = z['tL']
    return (lambda p: p @ A[:, :2].T + A[:, 2]), (lambda p: p @ L.T + t)
base = np.load(O / 'shift_0.00_0.00.npz'); bA, bL = maps(base)
xy = base['ctr_xy'].astype(float)
rows = []
labels = [f'shift_{dx:.2f}_{dy:.2f}' for dx in (0, .25, .5, .75) for dy in (0, .25, .5, .75)] + ['rot_0.1', 'rot_0.3']
for lab in labels:
    z = np.load(O / f'{lab}.npz'); fA, fL = maps(z)
    if lab.startswith('shift'):
        dx, dy = map(float, lab.split('_')[1:]); truth = xy + np.array([dx, dy])
    else:
        th = float(lab.split('_')[1]); M = cv2.getRotationMatrix2D((c0[0], c0[1]), th, 1.0); truth = xy @ M[:, :2].T + M[:, 2]
    for name, f, b in (('A_standard', fA, bA), ('L_lattice', fL, bL)):
        inc = f(xy) - b(xy); err = np.linalg.norm(inc - (truth - xy), axis=1)
        # sign check: also error if the truth increment had the opposite sign
        err_neg = np.linalg.norm(inc + (truth - xy), axis=1)
        rows.append(dict(pair=lab, map=name, median=np.median(err), p95=np.percentile(err, 95), max=err.max(), p95_if_opposite_sign=np.percentile(err_neg, 95)))
t = pd.DataFrame(rows); t.to_csv(ROOT / 'data/results/v78_candidate_path/position_error_synthetic.csv', index=False)
pd.set_option('display.width', 200)
print(t.groupby('map')[['median', 'p95', 'max']].agg(['median', 'max']).round(4))
print('overall p95 over all nodes of all pairs: ', {m: float(np.percentile(np.concatenate([np.linalg.norm((maps(np.load(O/f'{lab}.npz'))[0 if m=='A_standard' else 1](xy) - (bA if m=='A_standard' else bL)(xy)) - ((xy + np.array(list(map(float, lab.split('_')[1:])))) - xy if lab.startswith('shift') else (xy @ cv2.getRotationMatrix2D((c0[0], c0[1]), float(lab.split('_')[1]), 1.0)[:, :2].T + cv2.getRotationMatrix2D((c0[0], c0[1]), float(lab.split('_')[1]), 1.0)[:, 2]) - xy), axis=1) for lab in labels]), 95)) for m in ('A_standard', 'L_lattice')})
print(t[t.pair.str.startswith('rot')].round(3).to_string(index=False))
