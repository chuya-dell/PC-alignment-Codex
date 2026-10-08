"""v87 D2 (2): sub-pixel check of the integer period with the cross-scar lines.  The post image is warped into the pre frame with the chosen lattice map L (and with the 8 neighbouring integer-period alternatives),
band-passed (DoG 3-30 px), and the residual translation of windows centred on the pre cross-scar lines (vertical line: x residual; horizontal line: y residual) is measured by Lucas-Kanade (v83_needle_landmark.lk_translation).
The alternative with the smallest line residual is the landmark's choice.  usage: python v87_d2b_scar_lk.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v70_ledger_center_fit', 'field_level/v83_new_shots_261008', 'field_level/v57_false_positive_facts'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, tifffile, cv2
import v70_lib as L
import v83_needle_landmark as N
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V57 = Path(r'C:\Users\chuya\PC-alignment-localcorr\data\results\v57_false_positive_facts_20261006\stepD'); OUT = ROOT / 'data/results/v87_part_d'
H, W = 2044, 2048
FIDS = ['260924_1_3', '260825_4_8', '260828_10_5', '260825_5_4', '260825_4_5', '260924_1_4', '260829_8_8', '260923_7_1', '260924_1_2', '260825_7_2']   # last = control (margin large)


def windows(lines):
    ws = []
    for l in lines:
        if l['family'] == 'vertical':
            for k, y0 in enumerate(range(150, 1750, 400)):
                yc = y0 + 150; x = l['X0_full'] + l['S'] * (yc - (H - 2) / 2); ws.append(('v', (int(x - 80), y0, int(x + 80), y0 + 300)))
        else:
            for k, x0 in enumerate(range(250, 1800, 400)):
                xc = x0 + 150; y = l['X0_full'] + l['S'] * (xc - (W - 2) / 2); ws.append(('h', (x0, int(y - 80), x0 + 300, int(y + 80))))
    return ws


def one(fid):
    cv2.setNumThreads(1)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid'); rec = led.loc[fid]
    z = np.load(V70 / 'fields2' / f'{fid}.npz'); Llin, tL = z['Llin'], z['tL']; bas = z['post_model'][1:]
    j = json.load(open(V57 / 'results' / f'{fid}.json', encoding='utf8')); lines = json.loads(j.get('lines_json', '[]'))
    ws = [w for w in windows(lines) if w[1][0] >= 0 and w[1][1] >= 0 and w[1][2] <= W and w[1][3] <= H]
    pre = N.bandpass(L.read(rec.pre_path)); post = N.bandpass(L.read(rec.post_path)); rows = []
    for n1 in range(-1, 2):
        for n2 in range(-1, 2):
            M = np.hstack([Llin, (tL + n1 * bas[0] + n2 * bas[1])[:, None]]); wj = N.warp_into_i(post, M); res = []
            for kind, box in ws:
                r = N.lk_translation(pre, wj, box)
                if r and r['ncc'] > 0.5: res.append(r['dx'] if kind == 'v' else r['dy'])
            rows.append(dict(fid=fid, n1=n1, n2=n2, n_win=len(res), med_abs=float(np.median(np.abs(res))) if res else np.nan, med=float(np.median(res)) if res else np.nan))
    return rows


if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(8) as p: rows = sum(p.map(one, FIDS), [])
    t = pd.DataFrame(rows); t.to_csv(OUT / 'D2b_scar_lk.csv', index=False); pd.set_option('display.width', 250)
    for fid, g in t.groupby('fid', sort=False):
        g = g.sort_values('med_abs'); z0 = g[(g.n1 == 0) & (g.n2 == 0)].iloc[0]; print(fid, 'zero: |res| %.2f (n=%d)' % (z0.med_abs, z0.n_win), '| best', (int(g.iloc[0].n1), int(g.iloc[0].n2)), '%.2f' % g.iloc[0].med_abs, '| second %.2f' % g.iloc[1].med_abs)
