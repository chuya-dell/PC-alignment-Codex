"""Comparison-only recovery using the read-only v24 code snapshot.

Never search a transform against stored differences. Every field is estimated
once from native images by the original calls and then audited against cache.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse, hashlib, json, time, os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from field_round5_common import ROOT, OUT, LOCAL, discover, fields, load, verify as verify_prediction
from field_round5_methods import read_image, sample, binned, diagnostics, expected_diagnostics

CODE = next(p for p in LOCAL.iterdir() if p.name == 'code_v24' and p.is_dir())
sys.path.insert(0, str(CODE))
from shared import registration as reg
from shared.v2_registration_precision.refinement import register_refined
from shared.lattice_indexing import lattice_from_fft, grid_coordinates

DEST = OUT/'bf_v24_recovery'
DEST.mkdir(exist_ok=True)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def csv(rows, name):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def verify():
    return verify_prediction()

def pairs_for(dates):
    raw = next(p for p in LOCAL.iterdir() if p.name == 'raw_readonly')
    children = list(raw.iterdir())
    if not any(p.is_file() and 'copy_done' in p.name for p in children):
        raise RuntimeError('生画像コピー未完了')
    extra_done = any(p.is_file() and 'copy_done_2' in p.name for p in children)
    q = pd.read_csv(discover()/'tables'/'table_registration_field_qc.csv',dtype={'date':str,'board':str})
    pairs = {}
    for date in dates:
        if date != '260926' and not extra_done:
            continue
        roots = [p for p in children if p.is_dir() and p.name.startswith(date)]
        if len(roots) != 1:
            continue
        actual = {p.name:p for p in roots[0].iterdir() if p.is_file()}
        for _,r in q[q.date==date].iterrows():
            names = [str(r[c]).replace('\\','/').split('/')[-1] for c in ['path_pre','path_post']]
            if all(n in actual for n in names):
                pairs[f'{date}_{r.board}_{int(r.field)}'] = tuple(actual[n] for n in names)
    return pairs, extra_done, children

def recovery(dates, limit=0):
    verify()
    cv2.setNumThreads(1)
    pairs, extra_done, children = pairs_for(dates)
    f = fields()
    f = f[f['日程'].isin(dates)]
    if limit:
        f = f.iloc[:limit]
    quality = pd.read_csv(discover()/'tables'/'table_alignment_quality_features_and_sensitivity_by_field.csv',dtype={'date':str,'board':str}).set_index('field_key')
    run = dict(dates=dates, additional_copy_complete=extra_done, raw_children=[p.name for p in children],
               code_root=str(CODE), recovery_tolerance=1e-6,
               estimator='original register_image_pair_affine(mask_stains=False), register_refined(stage=subpixel)',
               no_transform_search=True, cv_threads=1, cv_version=cv2.__version__, np_version=np.__version__)
    (DEST/('run_'+'_'.join(dates)+('_pilot' if limit else '')+'.json')).write_text(json.dumps(run,ensure_ascii=False,indent=2),encoding='utf-8')
    for _,r in f.iterrows():
        done = DEST/(r.key+'.json')
        if done.exists():
            continue
        if r.key not in pairs:
            print('not_available',r.key,flush=True)
            continue
        start=time.monotonic()
        pp,qp=pairs[r.key]
        row=dict(key=r.key,日程=r['日程'],基板=r['基板'],視野番号=int(r['視野番号']),濃度=r['濃度'],
                 ブランク=bool(r['ブランク']),固定29視野=bool(r['外れ値視野3percent以上']),
                 外れ値割合=float(r['外れ値ピラー割合']),洗浄前パス=str(pp),洗浄後パス=str(qp),保存差パス=r['保存データ実在パス'],
                 洗浄前指紋=digest(pp),洗浄後指紋=digest(qp),保存差指紋=digest(Path(r['保存データ実在パス'])),回復=False)
        try:
            pre,post=read_image(pp),read_image(qp)
            # The original execution script sets no random seed in these calls.
            # Do not seed differently, vary masks, or retry to improve cache fit.
            coarse,qc=reg.register_image_pair_affine(pre,post,mask_stains=False,return_qc=True)
            matrix,ref=register_refined(pre,post,stage='subpixel',initial=coarse,mask_stains=False)
            lattice=lattice_from_fft(pre,7.286)
            ids,xy=grid_coordinates(lattice,pre.shape[1],pre.shape[0],margin=30)
            postxy=xy@matrix[:,:2].T+matrix[:,2]
            aa,bb=reg.sample_contrast(pre,xy),reg.sample_contrast(post,postxy)
            good=aa.valid_sampling.to_numpy()&bb.valid_sampling.to_numpy()
            actual=aa.contrast.to_numpy()[good]-bb.contrast.to_numpy()[good]
            actualxy,actualids=xy[good],ids[good]
            d,oldxy,oldids=load(r)
            sameids=bool(actualids.shape==oldids.shape and np.array_equal(actualids,oldids))
            samexy=bool(actualxy.shape==oldxy.shape and np.max(abs(actualxy-oldxy))<=1e-6)
            err=float(np.max(abs(actual-d))) if sameids and len(actual)==len(d) else np.nan
            errmean=float(np.mean(abs(actual-d))) if sameids and len(actual)==len(d) else np.nan
            # Audit the original cached pre coordinates separately, without fitting.
            nativepost=oldxy@matrix[:,:2].T+matrix[:,2]
            oncache=sample(pre,oldxy)-sample(post,nativepost)
            cacheerr=float(np.max(abs(oncache-d))) if np.isfinite(oncache).all() else np.nan
            dg=diagnostics(coarse)
            olddg=expected_diagnostics(quality.loc[r.key,'coarse_qc_json']) if r.key in quality.index else {}
            diagnosticerr=max(abs(dg[k]-olddg[k]) for k in dg) if len(olddg)==6 else np.nan
            row.update(保存ピラー数=len(d),再計算ピラー数=len(actual),ピラー識別一致=sameids,洗浄前座標一致=samexy,
                       洗浄前座標最大差=float(np.max(abs(actualxy-oldxy))) if actualxy.shape==oldxy.shape else np.nan,
                       差最大絶対差=err,差平均絶対差=errmean,保存座標での差最大絶対差=cacheerr,
                       差一致ピラー率=float(np.mean(abs(actual-d)<=1e-6)) if sameids else np.nan,
                       診断最大差=diagnosticerr,位置合わせ明部マスク割合=qc['mask_fraction'],
                       元位置合わせ明部マスク割合=float(__import__('re').search(r"'mask_fraction': ([\d.eE+\-]+)",str(quality.loc[r.key,'coarse_qc_json'])).group(1)) if r.key in quality.index else np.nan,
                       微小位置合わせ採用=bool(ref['subpixel']['accepted']),coarse_qc=qc,refinement=ref,
                       最終行列=matrix.tolist(),粗い行列=coarse.tolist(),**dg)
            row['回復']=bool(sameids and samexy and np.isfinite(err) and err<=1e-6)
            row['状態']='回復' if row['回復'] else '保存差と不一致'
            row['原因候補']='未確定。実行コミット・依存版・乱数状態の相違。マスク割合と洗浄前格子再現は別列で監査。'
            np.savez_compressed(DEST/(r.key+'.npz'),matrix=matrix,coarse=coarse,xy=oldxy,postxy=nativepost,
                                computed_delta=oncache,bilinear_delta=sample(pre,oldxy,True)-sample(post,nativepost,True),
                                recomputed_xy=actualxy,recomputed_ids=actualids,recomputed_delta=actual,
                                template_basis=lattice.basis,template_origin=lattice.origin)
        except Exception as e:
            row.update(状態='実行失敗',理由=repr(e))
        row['秒']=time.monotonic()-start
        if digest(pp)!=row['洗浄前指紋'] or digest(qp)!=row['洗浄後指紋']:
            raise RuntimeError('入力画像の指紋変化')
        done.write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=True,default=str),encoding='utf-8')
        print(r.key,row['状態'],row.get('差最大絶対差'),round(row['秒'],2),flush=True)
        collect()
    collect()

def collect():
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(DEST.glob('26*.json'))]
    # Keep detailed nested records in field checkpoints; flatten numerical audit.
    csv([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows],'round5_v24_recovery.csv')
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dates',nargs='+',default=['260926'])
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    recovery(args.dates,args.limit)
