"""v77 step G: signal injection before registration.  Known intensity changes are added to the post (washed) raw image at known pillar
centres; the whole pipeline (registration, centres, readouts S0..S5) is then run; recovery and photometric bias are measured.
Amplitude = A x robust SD of the field's delta (A = 3, 5, 10); direction dim / bright; zones centre / edge / scar-outside; isolated / cluster.
Test kinds: 'both' (real blank fields: real displacement + injection), 'brightness_only' (burst pair), 'both_synth' (burst + known 1/4-px shift + injection).
usage: python v77_inject.py [--fields N]"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'field_level/v75_zero_truth_pairs')); sys.path.insert(0, str(ROOT / 'field_level/v74_scar_distance'))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import v75_pair_lib as P
import v75_run_pairs as RP
import v74_scar_distance as E
import v72_readout as R
import field_control_common as M60
OUT = ROOT / ('data/results/v77_signal_injection_P3_high' if '--amps' in sys.argv else 'data/results/v77_signal_injection_P2')
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V72 = ROOT / 'data/results/v72_real_field_readout'
W, H = 2048, 2044
AMPS = [int(x) for x in sys.argv[sys.argv.index('--amps') + 1].split(',')] if '--amps' in sys.argv else [3, 5, 10]; DIRS = ['dim', 'bright']; ZONES = ['center', 'edge', 'scar_out']; PATS = ['isolated', 'cluster']
KS = [3, 4, 5, 6, 8]
RDS = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S3P', 'S4P', 'S5P']
SIG = 1.3
S0_BOX = sum(np.exp(-(dx * dx + dy * dy) / (2 * SIG ** 2)) for dx in (-1, 0, 1) for dy in (-1, 0, 1))      # 3x3 sum of the unit Gaussian centred on a pixel


def blob(img, x, y, amp):
    """Add amp*exp(-r^2/2s^2) at float position (x,y) (window +-6 px)."""
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    ys, xs = np.mgrid[y0 - 6:y0 + 8, x0 - 6:x0 + 8]
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    img[ys[ok], xs[ok]] += amp * np.exp(-((xs[ok] - x) ** 2 + (ys[ok] - y) ** 2) / (2 * SIG ** 2))


def pick_sites(pre_ctr, zone_ok, rng, n, taken, min_sep=60.0, cluster=False):
    """Return list of node-index arrays (isolated: n singles; cluster: n/7 groups of 7)."""
    cand = np.flatnonzero(zone_ok); rng.shuffle(cand); out = []
    tree_pts = pre_ctr
    need = n if not cluster else n // 7
    for c in cand:
        p = pre_ctr[c]
        if any(np.hypot(*(p - q)) < min_sep for q in taken): continue
        if cluster:
            d = np.linalg.norm(pre_ctr - p, axis=1); nb = np.argsort(d)[:7]
            if d[nb[-1]] > 1.3 * 7.37 * 1.2 or not zone_ok[nb].all(): continue
            out.append(nb); taken.append(p)
        else:
            out.append(np.array([c])); taken.append(p)
        if len(out) >= need: break
    return out


def run_case(pre, post, label, kind, pre_ctr, post_ctr_for_pre, ok_nodes, zone_masks, sigma_field, rng, shift_hint=None):
    """pre_ctr: (n,2) saved pre centres; post_ctr_for_pre: where each pre node sits in the post image (partner centre).  Returns per-site rows."""
    f = post.astype(np.float64) / 65535
    taken = []; sites = []
    for zone in ZONES:
        zm = zone_masks.get(zone)
        if zm is None: continue
        for pat in PATS:
            for d in DIRS:
                for A in AMPS:
                    groups = pick_sites(pre_ctr, zm & ok_nodes, rng, 24 if pat == 'isolated' else 28, taken, cluster=(pat == 'cluster'))
                    for g in groups: sites.append(dict(zone=zone, pattern=pat, dir=d, A=A, nodes=g))
    for s in sites:
        sign = -1.0 if s['dir'] == 'dim' else 1.0
        a = sign * s['A'] * sigma_field / S0_BOX
        for n in s['nodes']: blob(f, post_ctr_for_pre[n, 0], post_ctr_for_pre[n, 1], a)
    post_inj = np.clip(f * 65535, 0, 65535).astype(np.float32)
    t0 = time.time(); res = P.analyse_pair(pre, post_inj)
    # evaluation arrays
    rows = []
    ctr = res['ctr_xy']; tree_c = cKDTree(ctr); xy0 = res['xy0']; tree_s = cKDTree(xy0)
    inj_pts = np.concatenate([pre_ctr[s['nodes']] for s in sites]) if sites else np.zeros((0, 2))
    tree_inj = cKDTree(inj_pts)
    stats = {}
    for rd in RDS:
        d = res[rd].astype(float); ok = np.isfinite(d); med = float(np.median(d[ok])); mad = 1.4826 * float(np.median(abs(d[ok] - med)))
        stats[rd] = (med, mad)
    # false-positive counts away from injected pillars
    fp = {}
    for rd in RDS:
        d = res[rd].astype(float); xy = xy0 if rd in ('S0', 'S1', 'S2') else ctr; ok = np.isfinite(d)
        dist, _ = tree_inj.query(xy); far = ok & (dist > 12)
        med, mad = stats[rd]
        for k in KS: fp[(rd, k)] = float(((d[far] > med + k * mad).mean()) * 91000)
    for s in sites:
        for n in s['nodes']:
            p = pre_ctr[n]; target = s['A'] * sigma_field * (1.0 if s['dir'] == 'dim' else -1.0)
            dc, ic = tree_c.query(p); ds, is_ = tree_s.query(p)
            for rd in RDS:
                if rd in ('S0', 'S1', 'S2'): idx, dd = is_, ds
                else: idx, dd = ic, dc
                if dd > 3.7: val = np.nan
                else: val = float(res[rd][idx])
                med, mad = stats[rd]
                row = dict(label=label, kind=kind, readout=rd, zone=s['zone'], pattern=s['pattern'], dir=s['dir'], A=s['A'], target=target, measured=val,
                           ratio=(val - med) / target if np.isfinite(val) else np.nan, snr=(val - med) / mad if np.isfinite(val) else np.nan, mad=mad, valid=bool(np.isfinite(val)))
                for k in KS:
                    row[f'det_pos_k{k}'] = bool(np.isfinite(val) and val > med + k * mad)          # current convention: positive side only
                    row[f'det_sym_k{k}'] = bool(np.isfinite(val) and abs(val - med) > k * mad)      # either direction
                rows.append(row)
    fprow = [dict(label=label, kind=kind, readout=rd, k=k, fp_count91k=v) for (rd, k), v in fp.items()]
    return rows, fprow, dict(label=label, n_sites=len(sites), seconds=time.time() - t0, ok_frac=float(res['ok'].mean()), A_corner_disagree=res['corner_disagree_px'])


def zones_for(pre_ctr, mask, dist_p):
    x, y = pre_ctr[:, 0], pre_ctr[:, 1]
    edge_d = np.minimum.reduce([x, y, W - x, H - y])
    zm = {'center': (x > 400) & (x < 1650) & (y > 400) & (y < 1650), 'edge': (edge_d > 40) & (edge_d < 160)}
    if mask is not None:
        sd = E.signed_dist_periods(mask, pre_ctr)
        zm['center'] &= sd > 8; zm['edge'] &= sd > 8; zm['scar_out'] = (sd > 2) & (sd < 6)
    return zm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nf = int(sys.argv[sys.argv.index('--fields') + 1]) if '--fields' in sys.argv else 6
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}).set_index('fid')
    sets = json.load(open(V72 / 'sets.json'))
    nb = [f for f in sets['normal_blank20']]; seen = set(); fields = []
    for f in nb:
        if led.loc[f].date not in seen: fields.append(f); seen.add(led.loc[f].date)
        if len(fields) >= nf: break
    allrows, allfp, info = [], [], []
    rng = np.random.default_rng(20261008)
    for fid in fields:
        r = led.loc[fid]; z2 = np.load(V70 / 'fields2' / f'{fid}.npz'); z = np.load(V72 / 'fields' / f'{fid}.npz')
        pre = M60.read(r.pre_path); post = M60.read(r.post_path); mask = R.load_mask(fid)
        pre_ctr = z2['pre_ctr'].astype(float); post_ctr = z2['post_ctr'].astype(float)[z2['jL']]
        ok_nodes = z['ok'].astype(bool)
        S3 = z['S3']; v = S3[np.isfinite(S3)]; sigma = 1.4826 * float(np.median(abs(v - np.median(v))))
        rows, fp, inf = run_case(pre, post, fid, 'both_real', pre_ctr, post_ctr, ok_nodes, zones_for(pre_ctr, mask, None), sigma, rng)
        allrows += rows; allfp += fp; info.append(inf); print('done', fid, inf, flush=True)
        pd.DataFrame(allrows).to_csv(OUT / 'injection_rows.csv', index=False); pd.DataFrame(allfp).to_csv(OUT / 'injection_fp.csv', index=False); pd.DataFrame(info).to_csv(OUT / 'injection_info.csv', index=False)
    # burst field: brightness-only and both(synth shift)
    p = M60.folder('260904_p50_repeat'); a = M60.read(p / '1.tif'); b = M60.read(p / '2.tif')
    base = P.analyse_pair(a, b)
    pre_ctr = base['ctr_xy'].astype(float); ok_nodes = base['ok'].astype(bool)
    # partner positions via L map
    post_ctr_for_pre = pre_ctr @ base['Llin'].T + base['tL']
    v = base['S3'][np.isfinite(base['S3'])]; sigma = 1.4826 * float(np.median(abs(v - np.median(v))))
    for kind, bb in (('brightness_only', b), ('both_synth', None)):
        if bb is None:
            fine = RP.upsample4(b); bb = RP.integrate4(RP.synth_shift(fine, 0.25, 0.5)).astype(np.float32)
            base2 = P.analyse_pair(a, bb); post_ctr_for_pre = pre_ctr @ base2['Llin'].T + base2['tL']
        rows, fp, inf = run_case(a, bb, 'burst', kind, pre_ctr, post_ctr_for_pre, ok_nodes, zones_for(pre_ctr, None, None), sigma, rng)
        allrows += rows; allfp += fp; info.append(inf); print('done burst', kind, inf, flush=True)
        pd.DataFrame(allrows).to_csv(OUT / 'injection_rows.csv', index=False); pd.DataFrame(allfp).to_csv(OUT / 'injection_fp.csv', index=False); pd.DataFrame(info).to_csv(OUT / 'injection_info.csv', index=False)


if __name__ == '__main__':
    main()
