"""v78: P readouts (pre-centre based, post at L-mapped position + local offset) for all 632 development fields.
Saves S3P, S5P (+ S2 reference from v72), pre-validity, scar-distance (periods, signed), and local focus proxies (amp/width ratios) per pre node.
usage: python v78_p_readout.py [--workers K]"""
from __future__ import annotations
import sys, json, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v74_scar_distance', 'field_level/v70_ledger_center_fit'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, cv2
import v75_pair_lib as P
import v72_readout as R
import v74_scar_distance as E
import v70_centers2 as C
V70 = ROOT / 'data/results/v70_ledger_center_fit'; V72 = ROOT / 'data/results/v72_real_field_readout'
OUT = ROOT / 'data/results/v78_candidate_path'
W, H = 2048, 2044


def one(rec):
    fid = rec['fid']; dst = OUT / 'fields' / f'{fid}.npz'
    if dst.exists(): return dict(fid=fid, ok=True, cached=True)
    t0 = time.time()
    try:
        z2 = np.load(V70 / 'fields2' / f'{fid}.npz')
        pre = R.L.read(rec['pre_path']); post = R.L.read(rec['post_path'])
        boxa, boxb = R.box_contrast(pre), R.box_contrast(post); pa, pb = R.plain_contrast(pre), R.plain_contrast(post)
        pre_d = dict(ctr=z2['pre_ctr'].astype(float), pred=z2['pre_pred'], amp=z2['pre_amp'], wid=z2['pre_wid'], aloc=z2['pre_aloc'])
        post_d = dict(ctr=z2['post_ctr'].astype(float), pred=z2['post_pred'], amp=z2['post_amp'], wid=z2['post_wid'], aloc=z2['post_aloc'])
        Sp = C.sub_flags(pre_d, 'main'); So = C.sub_flags(post_d, 'main'); jL = z2['jL']; dL = z2['dL']
        Llin, tL = z2['Llin'], z2['tL']
        expL = z2['pre_pred_lin'] @ Llin.T + tL
        ov = (expL[:, 0] >= 8) & (expL[:, 0] < W - 8) & (expL[:, 1] >= 8) & (expL[:, 1] < H - 8)
        okL = (~Sp) & (~So[jL]) & z2['consL'] & ov
        okP = (~Sp) & ov & z2['consL']
        expC = pre_d['ctr'] @ Llin.T + tL
        posP = P.local_offset_positions(expC, post_d['ctr'][jL], okL)
        ac = pre_d['ctr']
        S3P = R.rnd(boxa, ac) - R.rnd(boxb, posP); S5P = R.aper(pa, ac) - R.aper(pb, posP)
        S3P[~okP] = np.nan; S5P[~okP] = np.nan
        mask = R.load_mask(fid)
        sd = E.signed_dist_periods(mask, ac) if mask is not None else np.full(len(ac), np.nan)
        # focus proxies: pillar-image amplitude/width ratio post/pre at the matched node (post fit may be unreliable where substituted -> NaN)
        amp_ratio = np.where(~So[jL], np.log(np.maximum(post_d['amp'][jL], 1e-9) / np.maximum(pre_d['amp'], 1e-9)), np.nan)
        wid_diff = np.where(~So[jL], post_d['wid'][jL] - pre_d['wid'], np.nan)
        np.savez_compressed(dst, ctr=ac.astype(np.float32), ids=z2['pre_ids'].astype(np.int32), S3P=S3P.astype(np.float32), S5P=S5P.astype(np.float32), okP=okP, sd_periods=sd.astype(np.float32),
                            amp_ratio=amp_ratio.astype(np.float32), wid_diff=wid_diff.astype(np.float32), pre_amp=pre_d['amp'].astype(np.float32), pre_wid=pre_d['wid'].astype(np.float32), has_mask=mask is not None)
        return dict(fid=fid, ok=True, seconds=time.time() - t0)
    except Exception as e:
        return dict(fid=fid, ok=False, error=repr(e), trace=traceback.format_exc()[-800:])


def main():
    (OUT / 'fields').mkdir(parents=True, exist_ok=True)
    led = pd.read_csv(V70 / 'ledger.csv', dtype={'date': str, 'board': str}); recs = led.to_dict('records')
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 4
    from multiprocessing import Pool
    res = []
    with Pool(workers) as p:
        for r in p.imap_unordered(one, recs, chunksize=1):
            res.append(r); print('DONE' if r['ok'] else 'FAIL', r['fid'], r.get('seconds', ''), r.get('error', ''), flush=True)
    pd.DataFrame(res).to_csv(OUT / 'run_status.csv', index=False); print('finished', sum(r['ok'] for r in res), '/', len(res))


if __name__ == '__main__':
    main()
