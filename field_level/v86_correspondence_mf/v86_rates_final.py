"""v86 part C summary: correspondence rate (fraction of pillars read at actual centres on both sides, 'main' thresholds) for the fixed-29 + blank fields (98),
(a) as in v70 (all pillars), (b) outside the scar zone (sd > 2 periods of the v57 mask), (c) with a local-offset lattice prediction (shift measured against the local 96-px median offset instead of the global cubic
lattice) -- the only estimator change that raised the rate --, and the matched filter variants (measured in v86_mf_centers/v86_variants: no gain).  Plus the features of the fields that stay below the pass line.
usage: python v86_rates_final.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v70_ledger_center_fit'))
import numpy as np, pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
import v70_centers2 as C
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V84 = ROOT / 'data/results/v84_1fM_strategies'; V85 = ROOT / 'data/results/v85_abnormal_pillars'; OUT = ROOT / 'data/results/v86_correspondence_mf'
W, H = 2048, 2044; PER = 7.37; BLK = 96


def flags(d, local):
    c = C.CRIT2['main']; off = d['ctr'] - d['pred']; shift = np.linalg.norm(off, axis=1)
    if local:
        good = (shift < 2.2) & (d['amp'] > 0.5 * d['aloc']); nx = W // BLK; ny = H // BLK + 1
        gx = np.clip((d['pred'][:, 0] // BLK).astype(int), 0, nx - 1); gy = np.clip((d['pred'][:, 1] // BLK).astype(int), 0, ny - 1); cell = gy * nx + gx
        mx = np.full(nx * ny, np.nan); my = mx.copy(); order = np.argsort(cell[good]); cs = cell[good][order]; ox = off[good, 0][order]; oy = off[good, 1][order]; b = np.flatnonzero(np.diff(cs)) + 1
        for cc, sx, sy in zip(np.split(cs, b), np.split(ox, b), np.split(oy, b)):
            if len(sx) >= 20: mx[cc[0]] = np.median(sx); my[cc[0]] = np.median(sy)
        mx = np.where(np.isfinite(mx), mx, np.nanmedian(mx)).reshape(ny, nx); my = np.where(np.isfinite(my), my, np.nanmedian(my)).reshape(ny, nx)
        px = d['pred'][:, 0] / BLK - .5; py = d['pred'][:, 1] / BLK - .5
        shift = np.linalg.norm(off - np.column_stack([ndimage.map_coordinates(mx, [py, px], order=1, mode='nearest'), ndimage.map_coordinates(my, [py, px], order=1, mode='nearest')]), axis=1)
    bad = (shift > c['shift_max']) | (d['amp'] < c['amp_frac'] * d['aloc']) | (d['wid'] < 0.4)
    for a, b_ in cKDTree(d['ctr']).query_pairs(2.0): bad[a if d['amp'][a] < d['amp'][b_] else b_] = True
    return bad


def main():
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}); sel = led[(led.fixed29 | (led.group == 'blank')) & led.pre_exists & led.post_exists]
    foc = pd.read_csv(ROOT / 'data/results/v78_candidate_path/focus_proxy_by_field.csv', dtype={'date': str}).set_index('fid'); rows = []
    for r in sel.itertuples():
        z = np.load(V70 / 'fields2' / f'{r.fid}.npz'); c = np.load(V84 / 'cache' / f'{r.fid}.npz'); sd = c['sd'].astype(float)
        pre = dict(ctr=z['pre_ctr'].astype(float), pred=z['pre_pred'].astype(float), amp=z['pre_amp'], wid=z['pre_wid'], aloc=z['pre_aloc']); post = dict(ctr=z['post_ctr'].astype(float), pred=z['post_pred'].astype(float), amp=z['post_amp'], wid=z['post_wid'], aloc=z['post_aloc'])
        ex = z['pre_pred_lin'] @ z['Llin'].T + z['tL']; ov = (ex[:, 0] >= 8) & (ex[:, 0] < W - 8) & (ex[:, 1] >= 8) & (ex[:, 1] < H - 8); out_ = np.isfinite(sd) & (sd > 2)
        d = dict(fid=r.fid, date=r.date, group=r.group, fixed29=r.fixed29, pos=r.field, n_ov=int(ov.sum()), scar_zone_frac=float((ov & ~out_).sum() / ov.sum()), la_iqr=foc.loc[r.fid, 'la_iqr'] if r.fid in foc.index else np.nan, mad=foc.loc[r.fid, 'mad'] if r.fid in foc.index else np.nan)
        for nm, loc in (('base', False), ('local', True)):
            Sp = flags(pre, loc); So = flags(post, loc); ok = (~Sp) & (~So[z['jL']]) & z['consL'] & ov
            d[f'rate_all_{nm}'] = float(ok.sum() / ov.sum()); d[f'rate_out_{nm}'] = float(ok[out_ & ov].sum() / max((out_ & ov).sum(), 1))
            if nm == 'base':
                bad = ov & ~ok
                nxb = W // 64; gx = np.clip((pre['pred'][:, 0] // 64).astype(int), 0, nxb - 1); gy = np.clip((pre['pred'][:, 1] // 64).astype(int), 0, 31); cell = gy * nxb + gx
                tot = np.bincount(cell[ov & out_], minlength=nxb * 32); fb = np.bincount(cell[bad & out_], minlength=nxb * 32); blkbad = (tot > 40) & (fb / np.maximum(tot, 1) > 0.4)
                d['poor_region_area_frac'] = float(blkbad.sum() * 64 * 64 / (W * H)); d['fail_in_scar_zone_share'] = float((bad & ~out_).sum() / max(bad.sum(), 1))
                d['fail_in_poor_region_share'] = float((bad & out_ & blkbad[cell]).sum() / max(bad.sum(), 1)); d['fail_isolated_share'] = float((bad & out_ & ~blkbad[cell]).sum() / max(bad.sum(), 1))
                d['rate_out_isolated_only'] = float(1 - (bad & out_ & ~blkbad[cell]).sum() / max((out_ & ov).sum(), 1))
        rows.append(d)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'C_rates_fixed29_blank.csv', index=False); pd.set_option('display.width', 250)
    out = {}
    for lab, q in (('fixed29', t[t.fixed29]), ('blank', t[t.group == 'blank']), ('fixed29+blank', t)):
        out[lab] = {k: dict(median=float(q[k].median()), min=float(q[k].min()), n_lt95=int((q[k] < 0.95).sum()), n_lt85=int((q[k] < 0.85).sum()), n=len(q)) for k in ('rate_all_base', 'rate_all_local', 'rate_out_base', 'rate_out_local', 'rate_out_isolated_only')}
        out[lab]['scar_zone_frac_median'] = float(q.scar_zone_frac.median()); out[lab]['poor_region_area_frac_median'] = float(q.poor_region_area_frac.median())
    json.dump(out, open(OUT / 'C_rates_summary.json', 'w'), indent=1, ensure_ascii=False, default=float); print(json.dumps(out, indent=0, ensure_ascii=False, default=float))
    low = t[(t.rate_out_local < 0.95)].sort_values('rate_out_local')
    print(low[['fid', 'group', 'fixed29', 'pos', 'rate_all_base', 'rate_out_base', 'rate_out_local', 'scar_zone_frac', 'poor_region_area_frac', 'la_iqr', 'mad', 'fail_in_scar_zone_share', 'fail_in_poor_region_share', 'fail_isolated_share']].round(3).to_string())
    low.to_csv(OUT / 'C_fields_below_95_after.csv', index=False)


if __name__ == '__main__':
    main()
