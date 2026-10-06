"""v49: pre-registered coordinate-level local-deformation semi-synthetic benchmark.

This retains the v11 452-condition affine ledger and v17 independent-centre
detector.  It does not alter production registration or inspect local truth
while choosing a correction model.
"""
from __future__ import annotations
import argparse, importlib.util, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
V17=ROOT/'field_level/v17_independent_pillar_center_residual/field_measure_independent_center_residuals.py'
spec=importlib.util.spec_from_file_location('v17centres',V17); v17=importlib.util.module_from_spec(spec); sys.modules[spec.name]=v17; spec.loader.exec_module(v17)
SEED=20261006; SHAPE=(2044,2048); TOL=0.002

def local_path(old, root):
    tail=str(old).replace('\\','/').split('/4.生データD_remo/',1)[-1]
    return Path(root)/tail

def affine(points,m): return points@np.asarray(m)[:,:2].T+np.asarray(m)[:,2]

def field(xy, amplitude, scale_frac, kind, key):
    """Two-component deterministic smooth truth field, RMS-normalised on xy."""
    if amplitude==0: return np.zeros_like(xy)
    rng=np.random.default_rng(SEED+key)
    z=(xy-np.array([1023.5,1021.5]))/np.sqrt(2048*2044)
    wave=1/scale_frac
    if kind=='vertical': a=z[:,1]*wave; b=z[:,0]*wave
    elif kind=='diagonal': a=(z[:,0]+z[:,1])*wave/np.sqrt(2); b=(z[:,0]-z[:,1])*wave/np.sqrt(2)
    else: a=(.73*z[:,0]+.68*z[:,1])*wave; b=(-.55*z[:,0]+.84*z[:,1])*wave
    phase=rng.uniform(0,2*np.pi,4)
    fx=np.sin(2*np.pi*a+phase[0])+.35*np.sin(2*np.pi*b*1.7+phase[1])
    fy=np.cos(2*np.pi*b+phase[2])+.35*np.cos(2*np.pi*a*1.3+phase[3])
    out=np.c_[fx,fy]; rms=np.sqrt(np.mean(np.sum(out*out,axis=1)))
    return out*(amplitude/rms)

def baseline_error(xy,key):
    # A spatially constant affine-estimation residual calibrated to median 0.004 px.
    rng=np.random.default_rng(SEED*7+key); direction=rng.normal(size=2); direction/=np.linalg.norm(direction)
    return np.tile(direction*.004,(len(xy),1))

def design(xy, degree):
    x=(xy[:,0]-1023.5)/1024; y=(xy[:,1]-1021.5)/1022
    cols=[np.ones(len(x)),x,y]
    if degree>=2: cols += [x*x,x*y,y*y]
    if degree>=3: cols += [x*x*x,x*x*y,x*y*y,y*y*y]
    return np.asarray(cols).T

def fit_poly(x,r,degree,ridge):
    d=design(x,degree); eye=np.eye(d.shape[1]); eye[0,0]=0
    c=np.linalg.solve(d.T@d+ridge*eye,d.T@r)
    return lambda q: design(q,degree)@c

def fit_block(x,r,n):
    global_fn=fit_poly(x,r,1,1e-4); fns={}
    bx=np.clip((x[:,0]/2048*n).astype(int),0,n-1); by=np.clip((x[:,1]/2044*n).astype(int),0,n-1)
    for iy in range(n):
      for ix in range(n):
        take=(bx==ix)&(by==iy)
        if take.sum()>=12: fns[iy,ix]=fit_poly(x[take],r[take],1,1e-3)
    def predict(q):
      out=global_fn(q); qx=np.clip((q[:,0]/2048*n).astype(int),0,n-1); qy=np.clip((q[:,1]/2044*n).astype(int),0,n-1)
      for k,fn in fns.items():
        take=(qx==k[1])&(qy==k[0]); out[take]=fn(q[take])
      return out
    return predict

def fit_tps(x,r,smooth):
    # Fixed capped support makes the large pre-registered grid tractable.
    if len(x)>180: x,r=x[:180],r[:180]
    model=RBFInterpolator(x,r,kernel='thin_plate_spline',smoothing=smooth,neighbors=80)
    return lambda q:model(q)

def candidate_functions(x,r):
    out=[]
    for n in (2,4,6): out.append((f'block_affine_{n}x{n}',lambda n=n:fit_block(x,r,n)))
    for s in (.01,.1,1.): out.append((f'tps_smoothing_{s:g}',lambda s=s:fit_tps(x,r,s)))
    for d in (2,3):
      for lam in (1e-4,1e-2,1.): out.append((f'poly{d}_ridge_{lam:g}',lambda d=d,lam=lam:fit_poly(x,r,d,lam)))
    return out

def split(xy,key):
    q=(np.rint(xy[:,0]).astype(np.int64)*73856093+np.rint(xy[:,1]).astype(np.int64)*19349663+key)%10
    return q<8

def choose_and_fit(xy,observed,key):
    tr=split(xy,key); # deterministic cap, independent of truth
    train=np.flatnonzero(tr); valid=np.flatnonzero(~tr)
    rng=np.random.default_rng(SEED+key*11); rng.shuffle(train); train=train[:600]
    best=None
    for name,maker in candidate_functions(xy[train],observed[train]):
      try:
        fn=maker(); score=float(np.median(np.linalg.norm(fn(xy[valid])-observed[valid],axis=1)))
        if best is None or score<best[0]: best=(score,name)
      except Exception: pass
    if best is None: raise RuntimeError('no candidate fitted')
    # refit selected candidate only; model identity is frozen by validation
    maker=dict(candidate_functions(xy[train],observed[train]))[best[1]]
    return best[1],best[0],maker()

def load_centres(ledger, root, cache_dir):
    paths=sorted({str(local_path(x['path'],root)) for x in ledger})
    out={}; cache_dir.mkdir(parents=True,exist_ok=True)
    for i,p in enumerate(paths,1):
      cache=cache_dir/(hashlib.sha256(p.encode('utf8')).hexdigest()+'.npy')
      if cache.exists():
        out[p]=np.load(cache); print(f'centres {i}/{len(paths)}: cached {len(out[p])}',flush=True); continue
      image=v17.load_image_unicode(p)
      if image is None: raise FileNotFoundError(p)
      xy,score,_=v17.detect_centers(image)
      # Confidence centre selection and deterministic cap are fixed before fitting.
      xy=xy[score>=np.quantile(score,.5)]
      if len(xy)>900: xy=xy[np.linspace(0,len(xy)-1,900,dtype=int)]
      np.save(cache,xy)
      out[p]=xy
      print(f'centres {i}/{len(paths)}: {len(xy)}',flush=True)
    return out

def errrows(case_id,condition,xy,truth,base,selected,cv,name,amp,scale,kind,noise):
    # Truth-space residual: the standard affine fit does not model the local field,
    # so its remaining error is truth+base; a local correction removes `selected`.
    # (First probe omitted `truth` here; superseded, see superseded_probe_truth_omitted_bug/.)
    methods={'standard_affine':truth+base,'selected_local':truth+base-selected}
    rows=[]
    for method,resid in methods.items():
      e=np.linalg.norm(resid,axis=1)
      rows.append(dict(case_id=case_id,condition=condition,method=method,selected_candidate=name if method!='standard_affine' else '',cv_median_px=cv if method!='standard_affine' else np.nan,amplitude_rms_px=amp,scale_fraction=scale,field_kind=kind,noise_sd_px=noise,n_pillars=len(xy),median_error_px=float(np.median(e)),p95_error_px=float(np.quantile(e,.95))))
    return rows

def make_plots(rows, examples, figdir):
    df=pd.DataFrame(rows); agg=df.groupby(['condition','method'],as_index=False).agg(median_error_px=('median_error_px','median'),p95_error_px=('p95_error_px','median'))
    agg.to_csv(figdir.parent/'tables'/'condition_error_comparison.csv',index=False)
    plt.figure(figsize=(10,5));
    for method,part in agg.groupby('method'): plt.plot(range(len(part)),part.median_error_px,label=method,marker='.',lw=.8)
    plt.yscale('log');plt.ylabel('median truth error (px)');plt.xlabel('condition (grouped order)');plt.legend();plt.tight_layout();plt.savefig(figdir/'condition_error_comparison.png',dpi=160);plt.close()
    for label,(xy,true,base,corrected) in examples.items():
      take=np.linspace(0,len(xy)-1,min(250,len(xy)),dtype=int); fig,axs=plt.subplots(1,3,figsize=(13,4),sharex=True,sharey=True)
      for ax,val,title in zip(axs,[true,base,corrected],['known local truth','standard remainder','selected-candidate remainder']):
        ax.quiver(xy[take,0],xy[take,1],val[take,0],val[take,1],angles='xy',scale_units='xy',scale=.01,width=.003);ax.set_title(title);ax.invert_yaxis();ax.set_aspect('equal')
      fig.tight_layout();fig.savefig(figdir/f'{label}_vector_fields.png',dpi=160);plt.close(fig)

def main():
  ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--stage',choices=['baseline','probe','all'],default='all');a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True);(a.output/'tables').mkdir(exist_ok=True);(a.output/'figures').mkdir(exist_ok=True)
  ledger=json.loads((ROOT/'data/results/v11_registration_precision_20260926/baseline_inputs.json').read_text(encoding='utf8'))
  centres=load_centres(ledger,a.data_root,a.output/'centre_cache')
  base_rows=[]
  # Stage A: all original affine cases, no local field; truth-space error is calibrated standard residual.
  for i,item in enumerate(ledger):
    xy=centres[str(local_path(item['path'],a.data_root))]; b=baseline_error(xy,i); e=np.linalg.norm(b,axis=1)
    base_rows.append(dict(case_id=item['case_id'],method='standard_affine',n_pillars=len(xy),median_error_px=float(np.median(e)),p95_error_px=float(np.quantile(e,.95))))
  pd.DataFrame(base_rows).to_csv(a.output/'tables'/'baseline_452_no_local_deformation.csv',index=False)
  if a.stage=='baseline': return
  # One deterministic complex affine ledger condition per unique source: a balanced probe over all sources.
  chosen=[]
  for p in sorted({x['path'] for x in ledger}):
    options=[x for x in ledger if x['path']==p]; chosen.append(next((x for x in options if x['case_id'].endswith('axis_09')),options[0]))
  checkpoint=a.output/'tables'/'per_fov_truth_error.csv'
  rows=pd.read_csv(checkpoint).to_dict('records') if checkpoint.exists() else []
  done={r['case_id'] for r in rows}; examples={}; c=0
  for ci,item in enumerate(chosen):
    if item['case_id'] in done:
      print(f'probe source {ci+1}/{len(chosen)}: checkpointed',flush=True); continue
    xy=centres[str(local_path(item['path'],a.data_root))]; b=baseline_error(xy,1000+ci)
    for amp in (0.,.1,.3,.415,.6):
      scales=(.125,.25,.5) if amp else (.25,)
      kinds=('vertical','diagonal','irregular') if amp else ('none',)
      for scale in scales:
       for kind in kinds:
        for noise in (0.,.46):
          c+=1; true=field(xy,amp,scale,kind,c+ci*10000); rng=np.random.default_rng(SEED+c*101); observed=true+b+rng.normal(0,noise,(len(xy),2))
          name,cv,fn=choose_and_fit(xy,observed,c+ci*10000); correction=fn(xy); condition=f'a{amp:g}_s{scale:g}_{kind}_n{noise:g}'
          rows += errrows(item['case_id'],condition,xy,true,b,correction,cv,name,amp,scale,kind,noise)
          if ci==0 and amp in (.415,.6) and scale==.25 and kind=='diagonal' and noise in (0.,.46): examples[f'{condition}']=(xy,true,true+b,true+b-correction)
    pd.DataFrame(rows).to_csv(checkpoint,index=False)
    print(f'probe source {ci+1}/{len(chosen)} complete',flush=True)
  pd.DataFrame(rows).to_csv(checkpoint,index=False)
  d=pd.DataFrame(rows); wide=d.pivot(index=['case_id','condition','amplitude_rms_px','scale_fraction','field_kind','noise_sd_px'],columns='method',values='median_error_px').reset_index(); wide['difference_selected_minus_standard_px']=wide.selected_local-wide.standard_affine;wide['outcome']=np.where(wide.difference_selected_minus_standard_px<=-TOL,'improved',np.where(wide.difference_selected_minus_standard_px>=TOL,'worsened','unchanged'));wide.to_csv(a.output/'tables'/'paired_outcomes.csv',index=False)
  strata=(wide.groupby(['amplitude_rms_px','scale_fraction'],dropna=False).agg(n=('outcome','size'),improved=('outcome',lambda x:(x=='improved').sum()),worsened=('outcome',lambda x:(x=='worsened').sum()),unchanged=('outcome',lambda x:(x=='unchanged').sum()),median_difference_px=('difference_selected_minus_standard_px','median')).reset_index());strata.to_csv(a.output/'tables'/'amplitude_scale_summary.csv',index=False)
  make_plots(rows,examples,a.output/'figures')
  (a.output/'tables'/'run_metadata.json').write_text(json.dumps(dict(seed=SEED,stage='probe',n_baseline_cases=len(base_rows),n_probe_sources=len(chosen),n_probe_conditions=c,tolerance_px=TOL),indent=2),encoding='utf8')
if __name__=='__main__': main()
