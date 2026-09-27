"""Staged, paired known-truth experiment extending the v5/v7 real-range benchmark."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import registration as reg
from field_level.v7_masked_alignment_benchmark.field_run_masked_benchmark import gt_for

def cases():
    # Independently vary every axis, include both signs and coupled corners of the
    # historical range. Rotation is about the image centre, not the top-left.
    specs = [(0,0,0,1), (-26.756,0,0,1), (9.502,0,0,1),
             (0,-8.061,0,1), (0,20.157,0,1), (0,0,-.19828,1),
             (0,0,.19828,1), (0,0,0,.9952865), (0,0,0,1.0045361),
             (-26.756,20.157,-.19828,.9952865),
             (9.502,-8.061,.19828,1.0045361), (.37,-.43,.071,1.0013),
             (-47.,33.,-.855,.97), (47.,-33.,.855,1.027),
             (-106.,20.,-1.907,.939), (50.,-47.,1.907,1.027)]
    return specs

def truth_matrix(shape, spec):
    dx,dy,degrees,scale = spec
    angle=np.deg2rad(degrees)
    a=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    center=(np.array(shape[::-1])-1)/2
    return np.c_[a,center+np.array([dx,dy])-a@center]

def errors(estimate, truth, shape):
    h,w=shape; center=np.array([(w-1)/2,(h-1)/2,1.])
    d=(estimate-truth)@center
    # Polar rotation avoids interpreting shear as rotation.
    def decomp(m):
        u,s,v=np.linalg.svd(m[:,:2]); r=u@v
        return np.arctan2(r[1,0],r[0,0]),np.sqrt(np.linalg.det(m[:,:2])),s
    a,s,_=decomp(estimate); b,t,_=decomp(truth)
    theta=(a-b+np.pi)%(2*np.pi)-np.pi
    xy=np.array([[x,y,1] for y in np.linspace(32,h-33,9) for x in np.linspace(32,w-33,9)])
    e=xy@(estimate-truth).T
    radius=np.hypot(w-1,h-1)/2
    return dict(center_dx_px=d[0],center_dy_px=d[1],center_error_px=np.linalg.norm(d),
                rotation_error_deg=np.rad2deg(theta),rotation_edge_error_px=abs(theta)*radius,
                scale_error=s-t,scale_edge_error_px=abs(s-t)*radius,
                grid_rmse_px=np.sqrt(np.mean(np.sum(e*e,axis=1))),
                grid_max_error_px=np.max(np.linalg.norm(e,axis=1)))

def run(stage, pre, post, initial=None):
    if stage=='baseline':
        return reg.register_image_pair_affine(pre,post),{}
    from shared.v2_registration_precision.refinement import register_refined
    return register_refined(pre,post,stage=stage,initial=initial)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['baseline','subpixel','lattice','iterative'],required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    folders=['260824_p50_SHC6OH/Raw_Images_生データのみ','260825_p50_dna',
             '260826-p50-sam','260827_pp50_dna','260828-p50-SAM','260829-p50-sam',
             '260828-p50-dna','260829_p50_DNA']
    inputs=[(f,pos) for f in folders for pos in [1,6,7]]
    inputs += [('260826-p50-sam',p) for p in [2,5,8]]
    inputs=[item for i,item in enumerate(inputs) if i%a.shards==a.shard_index]
    rows=[]; ledger=[]
    output=a.output/(a.stage+'.csv')
    anchor_path=a.output.parent/'baseline.csv'
    if not anchor_path.is_file(): anchor_path=anchor_path.parent.parent/'baseline.csv'
    anchor_rows=(pd.read_csv(anchor_path).set_index('case_id') if a.stage!='baseline' else None)
    done=pd.read_csv(output).to_dict('records') if output.exists() else []
    completed={r['case_id'] for r in done}; rows.extend(done)
    for folder,pos in inputs:
        path=a.data_root/folder/f'1-{pos}-0.tif'
        if not path.is_file(): raise FileNotFoundError(path)
        raw=reg.load_image_unicode(str(path))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        specs=[(f'axis_{i:02d}',truth_matrix(raw.shape,s)) for i,s in enumerate(cases())]
        if folder=='260826-p50-sam' and pos in [1,2,5,6,8]:
            specs += [(f'legacy_{s}',gt_for(s,pos)[0]) for s in ['realrange','ty','theta_deg','delta_scale_pct']]
        for name,truth in specs:
            case=f'{folder.split("/")[0]}_1-{pos}_{name}'
            ledger.append(dict(case_id=case,path=str(path),sha256=digest,truth=truth.tolist()))
            if case in completed: continue
            if a.limit and len(rows)>=a.limit: break
            cv2.setRNGSeed(20260926)
            # Independent forward renderer; do not use estimator's inverse convention.
            moving=cv2.warpAffine(raw,truth,(raw.shape[1],raw.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            row=dict(case_id=case,stage=a.stage,source=folder,position=pos,scenario=name)
            start=time.perf_counter()
            try:
                initial=None
                if anchor_rows is not None:
                    anchor=anchor_rows.loc[case]
                    initial=np.array([[anchor[f'm{i}{j}'] for j in range(3)] for i in range(2)],np.float32)
                estimate,diag=run(a.stage,raw,moving,initial)
                row.update(status='ok',**errors(estimate,truth,raw.shape),diagnostics=json.dumps(diag))
                row.update({f'm{i}{j}':estimate[i,j] for i in range(2) for j in range(3)})
            except Exception as exc:
                row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            row['seconds']=time.perf_counter()-start
            rows.append(row); pd.DataFrame(rows).to_csv(output,index=False)
            print(case,row['status'],round(row.get('grid_rmse_px',float('nan')),5),flush=True)
        if a.limit and len(rows)>=a.limit: break
    (a.output/(a.stage+'_inputs.json')).write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
    df=pd.DataFrame(rows); good=df[df.status=='ok']
    summary=dict(stage=a.stage,n=len(df),success=len(good),failed=int((df.status!='ok').sum()))
    for col in ['center_error_px','rotation_edge_error_px','scale_edge_error_px','grid_rmse_px']:
        if col in good:
            summary.update({col+'_'+stat:float(fn(good[col])) for stat,fn in [('mean',np.mean),('median',np.median),('p95',lambda x:np.quantile(x,.95)),('max',np.max)]})
    (a.output/(a.stage+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(summary,flush=True)

if __name__=='__main__': main()
