"""Decompose large known-truth Phase-2 affine errors into ORB match/RANSAC stages."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import registration as reg
from shared.image_qc import bright_band_mask
from field_level.v10_registration_precision.field_precision_benchmark import errors

def stage(pre, post, truth, mask, ratio=.72, threshold=2.5):
    a=np.uint8(np.clip(reg.image01_for_registration(pre)*255,0,255))
    b=np.uint8(np.clip(reg.image01_for_registration(post)*255,0,255))
    valid=(~mask).astype(np.uint8)*255
    detector=cv2.ORB_create(nfeatures=12000,fastThreshold=3)
    # Match register_image_pair_affine's backwards-compatible mask reuse on both frames.
    ka,da=detector.detectAndCompute(a,valid); kb,db=detector.detectAndCompute(b,valid)
    if da is None or db is None: return {'n_pre':len(ka),'n_post':len(kb),'reason':'no_descriptors'}
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[m for m,n in pairs if m.distance<ratio*n.distance]
    if len(good)<8:return {'n_pre':len(ka),'n_post':len(kb),'n_ratio':len(good),'reason':'too_few_matches'}
    src=np.float32([ka[m.queryIdx].pt for m in good]); dst=np.float32([kb[m.trainIdx].pt for m in good])
    pred=cv2.transform(src[:,None,:],truth)[:,0,:]
    match_res=np.linalg.norm(dst-pred,axis=1)
    warp,inliers=cv2.estimateAffine2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=threshold,
                                      maxIters=4000,confidence=.995)
    if warp is None:return {'n_pre':len(ka),'n_post':len(kb),'n_ratio':len(good),'reason':'ransac_failed'}
    sel=inliers.ravel()!=0
    return {'n_pre':len(ka),'n_post':len(kb),'n_ratio':len(good),
      'match_truth_le_0_5':int((match_res<=.5).sum()),'match_truth_le_2':int((match_res<=2).sum()),
      'match_truth_median':float(np.median(match_res)), 'match_truth_p90':float(np.quantile(match_res,.9)),
      'inliers':int(sel.sum()),'inlier_fraction':float(sel.mean()),
      'inlier_truth_le_0_5':int((match_res[sel]<=.5).sum()),
      'inlier_truth_median':float(np.median(match_res[sel])) if sel.any() else None,
      'inlier_span_x':float(np.ptp(src[sel,0])) if sel.any() else 0,
      'inlier_span_y':float(np.ptp(src[sel,1])) if sel.any() else 0,
      'ransac_grid_rmse':errors(warp,truth,pre.shape)['grid_rmse_px'],
      'ransac_center_error':errors(warp,truth,pre.shape)['center_error_px'],
      'ransac_rotation_edge_error':errors(warp,truth,pre.shape)['rotation_edge_error_px'],
      'ransac_scale_edge_error':errors(warp,truth,pre.shape)['scale_edge_error_px']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1)
    old=Path('data/results/v11_registration_precision_20260926/baseline.csv')
    inputs=json.loads(Path('data/results/v11_registration_precision_20260926/baseline_inputs.json').read_text(encoding='utf-8'))
    chosen=pd.read_csv(old).query("status == 'ok' and grid_rmse_px >= 0.5").case_id.tolist()
    index={x['case_id']:x for x in inputs}; out=[]
    for case in chosen:
      item=index[case]; source=case.split('_1-')[0];pos=int(case.split('_1-')[1].split('_')[0])
      path=a.data_root/source/f'1-{pos}-0.tif';pre=reg.load_image_unicode(str(path))
      truth=np.asarray(item['truth'],dtype=np.float64)
      cv2.setRNGSeed(20260926)
      post=cv2.warpAffine(pre,truth,(pre.shape[1],pre.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
      mask=bright_band_mask(pre)
      base={'case_id':case,'truth_dx':float(truth[0,2]),'truth_dy':float(truth[1,2]),'mask_fraction':float(mask.mean())}
      for ratio,thr in [(.72,2.5),(.65,2.5),(.80,2.5),(.72,1.0),(.72,1.5),(.72,4.0)]:
        info=stage(pre,post,truth,mask,ratio,thr)
        out.append({**base,'ratio':ratio,'ransac_threshold':thr,**info})
      print(case,flush=True)
    df=pd.DataFrame(out);df.to_csv(a.output/'initial_affine_stage_diagnostics.csv',index=False)
    print(df.groupby(['ratio','ransac_threshold']).agg(n=('case_id','count'),median_rmse=('ransac_grid_rmse','median'),p95_rmse=('ransac_grid_rmse',lambda x:x.quantile(.95)),max_rmse=('ransac_grid_rmse','max'),n_over_1=('ransac_grid_rmse',lambda x:(x>=1).sum())).reset_index().to_string(index=False))
if __name__=='__main__':main()
