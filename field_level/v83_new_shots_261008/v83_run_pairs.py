"""v83 part A-3: run the current candidate path C1 (v75_pair_lib.analyse_pair: lattice map + integer period by pixel correlation + actual centres + area aperture S5,
and S5P = post fit not required) on the 2026-10-08 reshoot pairs (-0/-1 and -0/-3), plus the scar mask of the pre (-0) image (v57 detector, unchanged).
Nothing is tuned on these data.  Output: data/results/v83_new_shots_261008/pairs/{board}-{pos}_{kind}.npz and masks/.
usage: python v83_run_pairs.py [--workers K]"""
import sys, json, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
for p in ('field_level/v75_zero_truth_pairs', 'field_level/v72_real_field_readout', 'field_level/v70_ledger_center_fit', 'field_level/v60_band_scar_controls', 'field_level/v74_scar_distance', 'field_level/v57_false_positive_facts'):
    sys.path.insert(0, str(ROOT / p))
import numpy as np, pandas as pd, cv2, tifffile
import v75_pair_lib as P
import v74_scar_distance as E
import field_v57_detect_scars as D57
RAW = Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu\261008_p50_z')
OUT = ROOT / 'data/results/v83_new_shots_261008'; (OUT / 'pairs').mkdir(parents=True, exist_ok=True); (OUT / 'masks').mkdir(exist_ok=True)


def one(job):
    b, pos, kind = job; tag = f'{b}-{pos}_{kind}'; dst = OUT / 'pairs' / f'{tag}.npz'
    if dst.exists(): return dict(tag=tag, ok=True, cached=True)
    cv2.setNumThreads(1); t0 = time.time()
    try:
        pa = RAW / f'{b}-{pos}-0.tif'; pb = RAW / f'{b}-{pos}-{kind}.tif'
        a = tifffile.imread(pa).astype(np.float32); bb = tifffile.imread(pb).astype(np.float32)
        mp = OUT / 'masks' / f'{b}-{pos}_mask.npz'
        if not mp.exists():
            r = D57.process((f'{b}-{pos}', str(pa), str(OUT / 'masks'), True))
            json.dump(r, open(OUT / 'masks' / f'{b}-{pos}_scar.json', 'w', encoding='utf8'), ensure_ascii=False, default=str)
        mask = None
        if mp.exists():
            with np.load(mp) as z:
                shp = tuple(z['shape']); mask = np.unpackbits(z['mask'])[:np.prod(shp)].reshape(shp).astype(bool)
        res = P.analyse_pair(a, bb)
        ctr = res['ctr_xy'].astype(np.float32)
        sd = E.signed_dist_periods(mask, ctr) if mask is not None else np.full(len(ctr), np.nan)
        keep = {k: v for k, v in res.items() if isinstance(v, np.ndarray) and k not in ('pre_dict', 'S0', 'xy0', 'std_xy', 'S1', 'S2', 'S3', 'S4', 'S3A', 'S4A', 'S3P', 'S4P', 'okA')}
        np.savez_compressed(dst, sd_periods=sd.astype(np.float32), has_mask=mask is not None, **keep,
                            S0=res['S0'].astype(np.float32), xy0=res['xy0'].astype(np.float32))
        info = dict(tag=tag, ok=True, seconds=time.time() - t0, A=json.dumps(res['A'].tolist()), U=json.dumps(res['U'].tolist()), corner_disagree_px=res['corner_disagree_px'], frac_corr_px=res['frac_corr_px'],
                    period_shift=json.dumps([int(x) for x in res['period_shift']]), period_corr_zero=res['period_corr_zero'], period_corr_best=res['period_corr_best'],
                    ok_frac=float(res['ok'].mean()), sub_pre=res['sub_pre'], sub_post=res['sub_post'], pitch_pre=res['pitch_pre'], pitch_post=res['pitch_post'], has_mask=mask is not None)
        return info
    except Exception as e:
        return dict(tag=tag, ok=False, error=repr(e), trace=traceback.format_exc()[-900:])


if __name__ == '__main__':
    workers = int(sys.argv[sys.argv.index('--workers') + 1]) if '--workers' in sys.argv else 5
    jobs = [(b, p, k) for b in (1, 2, 3) for p in range(1, 9) for k in (1, 3)]
    if '--pilot' in sys.argv: jobs = [(1, 6, 1), (1, 6, 3)]
    from multiprocessing import Pool
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(one, jobs, chunksize=1):
            rows.append(r); print('DONE' if r['ok'] else 'FAIL', r['tag'], round(r.get('seconds', 0)), r.get('error', ''), flush=True)
    pd.DataFrame(rows).sort_values('tag').to_csv(OUT / 'pairs_run_status.csv', index=False)
