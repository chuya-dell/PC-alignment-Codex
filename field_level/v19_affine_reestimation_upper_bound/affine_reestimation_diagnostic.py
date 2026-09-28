"""Diagnostic-only affine refit from v17 independently detected centers."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared.registration import load_image_unicode
from field_level.v17_independent_pillar_center_residual.field_measure_independent_center_residuals import detect_centers

PITCH=7.286
MAX_REMAP=10
MAX_IRLS=30
FRAME_TOL=1e-4

def apply(p,m): return np.asarray(p) @ np.asarray(m)[:,:2].T + np.asarray(m)[:,2]

def associate(p,ps,q,qs,m):
    pi=np.flatnonzero(ps>=np.quantile(ps,.5)); qi=np.flatnonzero(qs>=np.quantile(qs,.5))
    pp=p[pi]; qq=q[qi]; pred=apply(pp,m)
    inside=(pred[:,0]>=12)&(pred[:,0]<2036)&(pred[:,1]>=12)&(pred[:,1]<2032)
    ids=np.flatnonzero(inside); d,j=cKDTree(qq).query(pred[inside])
    order=np.argsort(d,kind='stable'); _,first=np.unique(j[order],return_index=True)
    keep=np.zeros(len(d),bool); keep[order[first]]=True
    ids=ids[keep]; d=d[keep]; j=j[keep]
    same=d<=PITCH/2
    return pp[ids[same]],qq[j[same]],pi[ids[same]],qi[j[same]]

def fit_irls(x,y,seed):
    # Center and scale coordinates to condition the affine least-squares system.
    center=np.array([1023.5,1021.5]); scale=1024.
    z=(x-center)/scale; A=np.c_[z,np.ones(len(z))]
    target=y
    weights=np.ones(len(x)); coef=None
    for _ in range(MAX_IRLS):
        coef=np.linalg.lstsq(A*weights[:,None]**.5,target*weights[:,None]**.5,rcond=None)[0]
        r=np.linalg.norm(A@coef-target,axis=1)
        med=np.median(r); sig=max(.12,1.4826*np.median(np.abs(r-med)))
        u=r/(1.345*sig); nw=np.ones_like(u); mask=u>1; nw[mask]=1/u[mask]
        if np.max(np.abs(nw-weights))<1e-5: weights=nw; break
        weights=nw
    linear=coef[:2].T/scale
    trans=coef[2]-linear@center
    return np.c_[linear,trans]

def frame_delta(a,b):
    pts=np.array([[0.,0.],[2047.,0.],[0.,2043.],[2047.,2043.],[1023.5,1021.5]])
    return np.linalg.norm(apply(pts,a)-apply(pts,b),axis=1).max()

def refit(p,ps,q,qs,seed):
    m=np.array(seed,float); last=None; changed=0; compared=0
    for it in range(1,MAX_REMAP+1):
        x,y,pi,qi=associate(p,ps,q,qs,m)
        pairs=np.c_[pi,qi]
        if len(x)<12: return m,0,it,changed,False
        nm=fit_irls(x,y,m)
        if last is not None:
            old={int(a):int(b) for a,b in last}; new={int(a):int(b) for a,b in zip(pi,qi)}
            step_changed=sum(old.get(k)!=v for k,v in new.items() if k in old)
            changed+=step_changed; compared+=sum(k in old for k in new)
            frac=step_changed/max(1,sum(k in old for k in new))
        else: frac=0.
        delta=frame_delta(m,nm)
        stable=last is not None and np.array_equal(last,pairs)
        m=nm
        if stable and delta<FRAME_TOL: return m,len(x),it,changed,compared,changed/max(1,compared),True
        last=pairs
    return m,len(x),MAX_REMAP,changed,compared,changed/max(1,compared),False

def blocked_cv(p,ps,q,qs,seed,truth=None):
    x,y,_,_=associate(p,ps,q,qs,seed)
    if len(x)<32: return np.nan,np.nan,0
    bx=np.minimum(3,(x[:,0]/2048*4).astype(int)); by=np.minimum(3,(x[:,1]/2044*4).astype(int)); block=by*4+bx
    pred0=apply(x,seed); base=np.linalg.norm(y-pred0,axis=1)
    new=np.full(len(x),np.nan)
    for b in range(16):
        test=block==b; train=~test
        if test.sum()<1 or train.sum()<12: continue
        m=fit_irls(x[train],y[train],seed)
        new[test]=np.linalg.norm(y[test]-apply(x[test],m),axis=1)
    valid=np.isfinite(new)
    truth_rmse=np.nan
    if truth is not None and valid.any():
        # Evaluate held-out predicted coordinates against the known geometric truth.
        pred=apply(x[valid],fit_irls(x[~valid],y[~valid],seed)) if (~valid).sum()>=12 else apply(x[valid],seed)
        truth_rmse=float(np.sqrt(np.mean(np.sum((pred-apply(x[valid],truth))**2,axis=1))))
    return (float(np.sqrt(np.mean(base[valid]**2))) if valid.any() else np.nan,
            float(np.sqrt(np.mean(new[valid]**2))) if valid.any() else np.nan,int(valid.sum()))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    items=json.loads((ROOT/'data/results/v15_initial_affine_spatial_support_20260928/baseline_inputs.json').read_text(encoding='utf-8'))
    std=pd.read_csv(ROOT/'data/results/v15_initial_affine_spatial_support_20260928/subpixel_current_spatial_search.csv').set_index('case_id')
    groups={}
    for z in items: groups.setdefault(z['path'],[]).append(z)
    rows=[]
    for gi,(src,cases) in enumerate(groups.items(),1):
        rel=Path(src.replace('\\','/').split('/4.生データD_remo/')[-1]); path=Path(a.data_root)/rel
        image=load_image_unicode(str(path))
        if image is None: raise FileNotFoundError(path)
        pre,ps,_=detect_centers(image); h,w=image.shape
        for case in cases:
            truth=np.asarray(case['truth'],float); moving=cv2.warpAffine(image,truth.astype(np.float32),(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            post,qs,_=detect_centers(moving)
            initrow=std.loc[case['case_id']]
            # Matrix columns in benchmark table are stored as six scalar entries.
            seed=np.array([[initrow.m00,initrow.m01,initrow.m02],[initrow.m10,initrow.m11,initrow.m12]])
            m,n,it,chg,ncmp,chgf,conv=refit(pre,ps,post,qs,seed)
            basecv,newcv,ncv=blocked_cv(pre,ps,post,qs,seed,truth)
            grid=np.array([[x,y] for x in np.linspace(0,w-1,17) for y in np.linspace(0,h-1,17)])
            e0=np.linalg.norm(apply(grid,seed)-apply(grid,truth),axis=1); e1=np.linalg.norm(apply(grid,m)-apply(grid,truth),axis=1)
            rows.append(dict(case_id=case['case_id'],n_pairs=n,iterations=it,changed_pair_count=chg,changed_pair_comparisons=ncmp,
                changed_pair_fraction=chgf,converged=conv,
                current_truth_rmse=float(np.sqrt(np.mean(e0**2))),refit_truth_rmse=float(np.sqrt(np.mean(e1**2))),
                current_truth_p95=float(np.quantile(e0,.95)),refit_truth_p95=float(np.quantile(e1,.95)),
                heldout_current_rmse=basecv,heldout_refit_rmse=newcv,n_heldout=ncv))
        print(f'synthetic source groups {gi}/{len(groups)}',flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'semisynthetic_452_refit.csv',index=False)
    summ={c:{'median':float(df[c].median()),'p95':float(df[c].quantile(.95)),'max':float(df[c].max())} for c in ['current_truth_rmse','refit_truth_rmse','heldout_current_rmse','heldout_refit_rmse']}
    (out/'semisynthetic_summary.json').write_text(json.dumps({'n_conditions':len(df),'summary':summ},indent=2),encoding='utf-8')
    print(json.dumps(summ,indent=2),flush=True)
if __name__=='__main__': main()
