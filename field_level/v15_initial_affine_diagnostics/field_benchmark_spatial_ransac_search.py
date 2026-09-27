"""Evaluate a spatial-support-selected, multi-candidate ORB/RANSAC affine fit."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from shared import registration as reg
from shared.image_qc import bright_band_mask
from field_level.v10_registration_precision.field_precision_benchmark import errors

CANDIDATES=[(.72,2.5),(.80,2.5),(.72,4.0)]

def candidates(pre,post,truth,mask):
    a=np.uint8(np.clip(reg.image01_for_registration(pre)*255,0,255));b=np.uint8(np.clip(reg.image01_for_registration(post)*255,0,255))
    valid=(~mask).astype(np.uint8)*255;det=cv2.ORB_create(nfeatures=12000,fastThreshold=3)
    ka,da=det.detectAndCompute(a,valid);kb,db=det.detectAndCompute(b,valid)
    if da is None or db is None:return []
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    rows=[]
    for ratio,threshold in CANDIDATES:
        good=[m for m,n in pairs if m.distance<ratio*n.distance]
        if len(good)<8:continue
        src=np.float32([ka[m.queryIdx].pt for m in good]);dst=np.float32([kb[m.trainIdx].pt for m in good])
        warp,inliers=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=threshold,maxIters=4000,confidence=.995)
        if warp is None:continue
        keep=inliers.ravel()!=0
        span=np.ptp(src[keep],axis=0) if keep.any() else np.zeros(2)
        # Reward independent support across both image axes and a broad inlier consensus.
        score=float(span[0]*span[1]*(keep.mean()))
        metric=errors(warp,truth,pre.shape)
        rows.append({'ratio':ratio,'threshold':threshold,'n_matches':len(good),'n_inliers':int(keep.sum()),
                     'inlier_fraction':float(keep.mean()),'span_x_px':float(span[0]),'span_y_px':float(span[1]),
                     'support_score':score,'warp':warp.tolist(),**metric})
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shards',type=int,default=1);p.add_argument('--shard-index',type=int,default=0);p.add_argument('--case-list',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    ledger=json.loads(Path('data/results/v11_registration_precision_20260926/baseline_inputs.json').read_text(encoding='utf-8'))
    if a.case_list:
        wanted=set(a.case_list.read_text(encoding='utf-8').splitlines())
        ledger=[x for x in ledger if x['case_id'] in wanted]
    ledger=[x for i,x in enumerate(ledger) if i%a.shards==a.shard_index]
    details=[];selected=[];cache={}
    for i,item in enumerate(ledger):
        case=item['case_id'];path=Path(item['path'])
        if not path.is_absolute():path=a.data_root/path
        key=str(path)
        if key not in cache:cache[key]=reg.load_image_unicode(str(path))
        pre=cache[key];truth=np.asarray(item['truth'],dtype=np.float64)
        cv2.setRNGSeed(20260926)
        post=cv2.warpAffine(pre,truth,(pre.shape[1],pre.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        cs=candidates(pre,post,truth,bright_band_mask(pre))
        for c in cs:
            details.append({'case_id':case,**{k:v for k,v in c.items() if k!='warp'},
                            **{f'm{i}{j}':c['warp'][i][j] for i in range(2) for j in range(3)}})
        if not cs:selected.append({'case_id':case,'status':'failed','error':'all RANSAC candidates failed'});continue
        chosen=max(cs,key=lambda x:x['support_score'])
        selected.append({'case_id':case,'status':'ok','selected_ratio':chosen['ratio'],'selected_threshold':chosen['threshold'],
                         **{f'm{i}{j}':chosen['warp'][i][j] for i in range(2) for j in range(3)},
                         'selected_score':chosen['support_score'],**{k:v for k,v in chosen.items() if k not in ('warp','ratio','threshold','support_score')}})
        if (i+1)%10==0:print(f'{i+1}/{len(ledger)}',flush=True)
    suffix=f'_shard_{a.shard_index}' if a.shards>1 else ''
    pd.DataFrame(details).to_csv(a.output/f'candidates{suffix}.csv',index=False)
    pd.DataFrame(selected).to_csv(a.output/f'spatial_search{suffix}.csv',index=False)
    print(f'completed {len(ledger)} cases; errors={sum(r.get("status")!="ok" for r in selected)}',flush=True)
if __name__=='__main__':main()
