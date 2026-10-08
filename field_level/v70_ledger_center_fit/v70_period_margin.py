"""v70: confidence of the integer-period decision (suggested by the Gemini second opinion): for the corrected mapping, scan integer shifts (-3..3)^2 around the
chosen mapping and report best (should be the zero shift after correction), second best, and the margin.  Fields with a small margin are flagged."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd, cv2
from scipy import ndimage
import v70_lib as L
import v70_period_fix as PF
V70 = PF.V70


def one(rec):
    fid = rec['fid']; z = np.load(V70 / 'fields2' / f'{fid}.npz')
    a = L.read(rec['pre_path']); b = L.read(rec['post_path'])
    ca = a - cv2.GaussianBlur(a, (0, 0), 8); cb = b - cv2.GaussianBlur(b, (0, 0), 8)
    pc = z['pre_ctr'].astype(float); va = ndimage.map_coordinates(ca, [pc[:, 1], pc[:, 0]], order=1, mode='nearest')
    res = PF.scan(va, pc, cb, z['Llin'], z['tL'], z['post_model'][1:], 3)       # tL here is already corrected: best should be (0,0)
    items = sorted(res.items(), key=lambda kv: -kv[1]); (b0, c0), (b1, c1) = items[0], items[1]
    nb = [v for k, v in res.items() if k != (0, 0)]
    return dict(fid=fid, best=str(b0), corr_best=c0, corr_second=c1, margin=c0 - c1, corr_zero=res[(0, 0)], best_is_zero=(b0 == (0, 0)), max_other=max(nb), margin_zero_vs_other=res[(0, 0)] - max(nb))


if __name__ == '__main__':
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str})
    from multiprocessing import Pool
    with Pool(6) as p: r = p.map(one, led.to_dict('records'), chunksize=2)
    t = pd.DataFrame(r).merge(led[['fid', 'fixed29', 'group']], on='fid'); t.to_csv(V70 / 'period_margin.csv', index=False)
    print('fields', len(t), 'best shift is zero:', int(t.best_is_zero.sum()), '| margin_zero_vs_other quantiles', t.margin_zero_vs_other.quantile([.01, .05, .25, .5]).round(3).to_dict())
    print('margin<0.05:', int((t.margin_zero_vs_other < 0.05).sum()), 'margin<0.1:', int((t.margin_zero_vs_other < 0.1).sum()), '; fixed29 min margin', round(t[t.fixed29].margin_zero_vs_other.min(), 3))
    print(t[t.margin_zero_vs_other < 0.1][['fid', 'corr_zero', 'max_other', 'margin_zero_vs_other', 'fixed29']].round(3).head(15).to_string(index=False))
