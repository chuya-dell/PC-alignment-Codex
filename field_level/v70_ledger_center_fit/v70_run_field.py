"""v70: per-field run. Current standard re-run + per-pillar centers + physical index matching.
usage: python v70_run_field.py [--pilot N] [--workers K]   (resumable: skips fields with finished npz)"""
from __future__ import annotations
import sys, json, time, hashlib, traceback
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
L_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, cv2
sys.path.insert(0, str(L_ROOT / 'field_level/v60_band_scar_controls'))
import v70_lib as L
import field_control_common as M60
OUT = L.ROOT / 'data/results/v70_ledger_center_fit'
FIELDS = OUT / 'fields'


def nonperiodic(im):
    a = cv2.GaussianBlur(im, (0, 0), 5) - cv2.GaussianBlur(im, (0, 0), 45)
    return ((a - a.mean()) / (a.std() + 1e-9)).astype(np.float32)


def one(rec):
    fid = rec['fid']; dst = FIELDS / f'{fid}.npz'; js = FIELDS / f'{fid}.json'
    if dst.exists() and js.exists():
        prev = json.loads(js.read_text(encoding='utf8'))
        if prev.get('ok') and 'U_deviation' in prev:
            return prev
    t0 = time.time(); row = dict(fid=fid, ok=False)
    try:
        pre = L.read(rec['pre_path']); post = L.read(rec['post_path'])
        h, w = pre.shape
        std = L.standard_run(pre, post)
        A = std['matrix']
        cpre = L.center_fit_for_image(pre); cpost = L.center_fit_for_image(post)
        j, dmatch, exp = L.match_pre_post(cpre, cpost, A)
        Bpre = cpre['model'][1:].T; Bpost = cpost['model'][1:].T
        Ucont = np.linalg.inv(Bpost) @ A[:, :2] @ Bpre
        U = np.rint(Ucont).astype(int); u_dev = float(np.abs(Ucont - U).max())
        di = cpost['ids'][j] - (cpre['ids'] @ U.T)
        good = dmatch < 1.8
        vals, cnt = np.unique(di[good], axis=0, return_counts=True)
        k = int(np.argmax(cnt)); mode = vals[k]
        idx_consistent = np.all(di == mode, axis=1) & good
        # independent non-periodic landmark check (v60 register: lattice-suppressed coarse correlation, SIFT, ECC)
        c0 = np.array([w / 2, h / 2]); d_aff = (A[:, :2] @ c0 + A[:, 2]) - c0
        try:
            sft, inf60 = M60.register(pre, post)
            np_coarse = np.array([inf60['coarse_dx'], inf60['coarse_dy']]); np_fine = np.array(sft)
            np_resid = float(np.linalg.norm(d_aff - np_coarse)); np_resid_fine = float(np.linalg.norm(d_aff - np_fine))
            np_feat = float(np.linalg.norm(d_aff - np.array([inf60['feature_dx'], inf60['feature_dy']]))) if np.isfinite(inf60['feature_dx']) else np.nan
        except Exception as e:
            np_resid = np.nan; np_resid_fine = np.nan; np_feat = np.nan; row['np_error'] = repr(e)
        # stored comparison
        corr = np.nan; rmse = np.nan
        try:
            with np.load(L.CACHE / f'{fid}.npz') as z:
                sd, sids = z['delta'], z['ids']
            if sd.shape == std['delta'].shape and np.array_equal(sids, std['ids']):
                corr = float(np.corrcoef(sd, std['delta'])[0, 1]); rmse = float(np.sqrt(np.mean((sd - std['delta']) ** 2)))
            else:
                a = pd.Series(std['delta'], index=pd.MultiIndex.from_arrays(std['ids'].T))
                b = pd.Series(sd, index=pd.MultiIndex.from_arrays(sids.T))
                m = a.index.intersection(b.index)
                if len(m) > 1000:
                    corr = float(np.corrcoef(a.loc[m], b.loc[m])[0, 1]); rmse = float(np.sqrt(np.mean((a.loc[m] - b.loc[m]) ** 2)))
                row['stored_overlap'] = int(len(m)); row['stored_n'] = int(len(sd)); row['rerun_n'] = int(len(std['delta']))
        except Exception as e:
            row['stored_cmp_error'] = str(e)
        np.savez_compressed(dst,
            std_delta=std['delta'].astype(np.float32), std_xy=std['xy'].astype(np.float32), std_ids=std['ids'].astype(np.int32),
            matrix=A, pre_ids=cpre['ids'].astype(np.int32), pre_pred=cpre['pred'].astype(np.float32), pre_ctr=cpre['ctr'].astype(np.float32),
            pre_amp=cpre['amp'].astype(np.float32), pre_wid=cpre['wid'].astype(np.float32),
            post_ids=cpost['ids'].astype(np.int32), post_pred=cpost['pred'].astype(np.float32), post_ctr=cpost['ctr'].astype(np.float32),
            post_amp=cpost['amp'].astype(np.float32), post_wid=cpost['wid'].astype(np.float32),
            match_j=j.astype(np.int32), match_d=dmatch.astype(np.float32), idx_consistent=idx_consistent, U=U,
            pre_model=cpre['model'], post_model=cpost['model'])
        row.update(ok=True, n_std=int(len(std['delta'])), matrix=A.tolist(), qc_ok=bool(std['qc'].get('physical_ok', True)) if isinstance(std['qc'], dict) else None,
                   pre_aref=cpre['aref'], post_aref=cpost['aref'], pre_info=cpre['info'], post_info=cpost['info'],
                   n_pre_nodes=int(len(cpre['ids'])), n_post_nodes=int(len(cpost['ids'])),
                   matched_fraction=float(good.mean()), index_mode=[int(mode[0]), int(mode[1])],
                   index_consistent_fraction=float(idx_consistent.mean()), index_mode_fraction_of_matched=float((di[good] == mode).all(axis=1).mean()),
                   np_coarse_resid_px=np_resid, np_fine_resid_px=np_resid_fine, np_feature_resid_px=np_feat, aff_center_shift=[float(d_aff[0]), float(d_aff[1])], U=U.tolist(), U_deviation=u_dev,
                   stored_corr=corr, stored_rmse=rmse, seconds=time.time() - t0)
    except Exception as e:
        row.update(ok=False, error=repr(e), trace=traceback.format_exc()[-1500:], seconds=time.time() - t0)
    js.write_text(json.dumps(row, ensure_ascii=False, default=str), encoding='utf8')
    print('DONE' if row['ok'] else 'FAIL', fid, f"{row['seconds']:.0f}s", row.get('index_consistent_fraction', ''), row.get('error', ''), flush=True)
    return row


def main():
    FIELDS.mkdir(parents=True, exist_ok=True)
    led = pd.read_csv(OUT / 'ledger.csv', dtype={'date': str, 'board': str})
    recs = led[led.pre_exists & led.post_exists].to_dict('records')
    workers = 3
    if '--workers' in sys.argv: workers = int(sys.argv[sys.argv.index('--workers') + 1])
    if '--pilot' in sys.argv:
        n = int(sys.argv[sys.argv.index('--pilot') + 1]); recs = recs[::max(1, len(recs) // n)][:n]
    if workers <= 1:
        res = [one(r) for r in recs]
    else:
        from multiprocessing import Pool
        with Pool(workers) as p:
            res = p.map(one, recs, chunksize=1)
    print('finished', sum(r['ok'] for r in res), '/', len(res))


if __name__ == '__main__':
    main()
