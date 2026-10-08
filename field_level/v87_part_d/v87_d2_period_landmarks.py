"""v87 D2: fields with a small integer-period margin (<0.05; 9 fields incl. 260825_4_8) re-checked with landmarks that do not depend on the pillar lattice:
 (1) non-periodic registration (v60 register: lattice-suppressed coarse correlation, sub-pixel) -> translation at the image centre,
 (2) cross-scar lines (v57 detector on pre and post): perpendicular line displacement (vertical line -> dx, horizontal line -> dy) at the image centre.
Compared with the translation implied by the chosen lattice map L at the centre (d_L = L c0 - c0) and with the integer-period alternatives d_L + n1*b1 + n2*b2 (post lattice vectors, n in -2..2).
If the landmark translation is closest to the chosen (0,0), the integer period is confirmed independently; if closest to another (n1,n2), the field is flagged.   usage: python v87_d2_period_landmarks.py"""
import sys, json
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls', 'field_level/v57_false_positive_facts'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, tifffile, cv2
import v70_lib as L
import field_control_common as M60
import field_v57_detect_scars as D57
V70 = ROOT / 'data/results/v70_ledger_center_fit'; OUT = ROOT / 'data/results/v87_part_d'; OUT.mkdir(parents=True, exist_ok=True)
H, W = 2044, 2048; c0 = np.array([W / 2, H / 2])
FIDS = ['260924_1_3', '260825_4_8', '260828_10_5', '260825_5_4', '260825_4_5', '260924_1_4', '260829_8_8', '260923_7_1', '260924_1_2']


def lines_of(path, tag):
    r = D57.process((tag, path, str(OUT / 'd2_scar'), False))
    return json.loads(r.get('lines_json', '[]')) if r.get('status') == 'ok' else []


def line_pos_at_centre(lines):
    pos = {}
    for l in lines:
        if l['family'] == 'vertical': pos['v'] = l['X0_full'] + l['S'] * (c0[1] - (H - 2) / 2)
        else: pos['h'] = l['X0_full'] + l['S'] * (c0[0] - (W - 2) / 2)
    return pos


def one(fid):
    cv2.setNumThreads(1)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid'); rec = led.loc[fid]
    z = np.load(V70 / 'fields2' / f'{fid}.npz'); Llin, tL = z['Llin'], z['tL']; bas = z['post_model'][1:]
    d_L = Llin @ c0 + tL - c0; A = z['A']; d_A = A[:, :2] @ c0 + A[:, 2] - c0
    pre = L.read(rec.pre_path); post = L.read(rec.post_path)
    sft, inf = M60.register(pre, post); d_np = np.array(sft, float)
    out = dict(fid=fid, d_L=d_L.tolist(), d_A=d_A.tolist(), d_np=d_np.tolist(), resid_L_np=float(np.linalg.norm(d_L - d_np)), resid_A_np=float(np.linalg.norm(d_A - d_np)))
    cands = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            d = d_L + n1 * bas[0] + n2 * bas[1]; cands.append((float(np.linalg.norm(d - d_np)), n1, n2))
    cands.sort(); out['np_closest_shift'] = (cands[0][1], cands[0][2]); out['np_closest_resid'] = cands[0][0]; out['np_second_resid'] = cands[1][0]; out['np_resid_at_zero'] = out['resid_L_np']
    # scar lines
    lp = line_pos_at_centre(lines_of(rec.pre_path, f'{fid}_pre')); lq = line_pos_at_centre(lines_of(rec.post_path, f'{fid}_post'))
    out['line_v_dx'] = (lq['v'] - lp['v']) if ('v' in lp and 'v' in lq) else None; out['line_h_dy'] = (lq['h'] - lp['h']) if ('h' in lp and 'h' in lq) else None
    # perpendicular-component residual for each integer alternative
    res = []
    for n1 in range(-2, 3):
        for n2 in range(-2, 3):
            d = d_L + n1 * bas[0] + n2 * bas[1]; e = []
            if out['line_v_dx'] is not None: e.append(d[0] - out['line_v_dx'])
            if out['line_h_dy'] is not None: e.append(d[1] - out['line_h_dy'])
            if e: res.append((float(np.sqrt(np.mean(np.square(e)))), n1, n2))
    if res: res.sort(); out['line_closest_shift'] = (res[0][1], res[0][2]); out['line_closest_resid'] = res[0][0]; out['line_resid_at_zero'] = [r[0] for r in res if r[1:] == (0, 0)][0]
    return out


if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(6) as p: rows = p.map(one, FIDS)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'D2_period_landmarks.csv', index=False); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
    print(t.drop(columns=['d_L', 'd_A', 'd_np']).round(3).to_string())
    json.dump(rows, open(OUT / 'D2_period_landmarks.json', 'w'), indent=1, default=float)
