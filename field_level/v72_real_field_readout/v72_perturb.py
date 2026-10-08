"""v72: induce the phenomenon.  Normal blank fields, post sampling map = lattice-implied map L plus a deliberate linear error E
(scale / rotation / shear types), corner displacement m px.  Reading S1-type (ideal grid, integer rounding).  Thresholds fixed from the
unperturbed S1 date-blank pool (mean+3SD).  Compare rate / band power with the real standard (S0) versus measured |A-L|."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v60_band_scar_controls')); sys.path.insert(0, str(ROOT / 'field_level/v72_real_field_readout'))
import numpy as np, pandas as pd
import field_control_common as M
import v72_readout as R
V70 = ROOT / 'data/results/v70_ledger_center_fit'
OUT = ROOT / 'data/results/v72_real_field_readout'
TYPES = {'scale': np.array([[1., 0], [0, 1.]]), 'rotation': np.array([[0., -1], [1., 0]]), 'shear_x': np.array([[0., 1], [0, 0.]]), 'shear_y': np.array([[0., 0], [1, 0.]])}
MS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
c0 = np.array([1024., 1022.]); corners = np.array([[0, 0], [2048, 0], [0, 2044], [2048, 2044]], float) - c0


def main():
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid')
    sets = json.load(open(OUT / 'sets.json'))
    th = pd.read_csv(OUT / 'thresholds_v2.csv', dtype={'date': str}); th = th[(th.readout == 'S1') & (th.thr == 'G')].set_index('date').threshold
    rows = []
    for fid in sets['normal_blank20']:
        r = led.loc[fid]; z2 = np.load(V70 / 'fields2' / f'{fid}.npz'); z1 = np.load(V70 / 'fields' / f'{fid}.npz')
        ca = R.box_contrast(R.L.read(r.pre_path)); cb = R.box_contrast(R.L.read(r.post_path))
        xy = z1['std_xy'].astype(float); Llin, tL = z2['Llin'], z2['tL']; t = th[r.date]
        a = R.rnd(ca, xy)
        for name, E in TYPES.items():
            scale = 1.0 / np.linalg.norm(E @ corners.T, axis=0).max()          # E scaled so that the largest corner displacement is 1 px per unit m
            for m in MS:
                if m == 0 and name != 'scale': continue
                xyp = xy @ Llin.T + tL + (xy - c0) @ (E * scale * m).T
                d = (a - R.rnd(cb, xyp)).astype(float); ok = np.isfinite(d)
                met = M.band_metrics(xy, d, t, False)
                rows.append(dict(fid=fid, date=r.date, type=name, m_px=m, sd=float(np.nanstd(d)), rate=float((d[ok] > t).mean()), count91k=float((d[ok] > t).mean() * 91000),
                                 band_power=met['band_power'], strength=met['strength'], density_sd=met['density_sd'], axis_deg=met['axis_deg'], period_px=met['period_px']))
        print('done', fid, flush=True)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'perturbation_by_field.csv', index=False)
    s = t.groupby(['type', 'm_px']).agg(rate=('rate', 'median'), count91k=('count91k', 'median'), band_power=('band_power', 'median'), strength=('strength', 'median'), sd=('sd', 'median')).reset_index()
    s.to_csv(OUT / 'perturbation_summary.csv', index=False)
    pd.set_option('display.width', 200); print(s.round(5).to_string())


if __name__ == '__main__':
    main()
