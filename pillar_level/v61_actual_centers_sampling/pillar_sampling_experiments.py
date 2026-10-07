"""Steps 1/2. Refit actual centers per image, retain fixed29 and date-matched blanks."""
from __future__ import annotations
import sys,os,json,time,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'field_level/v60_band_scar_controls'))
import field_control_common as M
import numpy as np,pandas as pd,cv2
from scipy import ndimage,optimize
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(limits):return nullcontext()
from multiprocessing import Pool
OUT=ROOT/'data/results/v61_actual_centers_sampling'; OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'data/results/v58_band_scar_causal/checkout'
STD=Path(r'W:\GoogleDrive\chuya2816\5.解析結果_chu\20260928_digital_judgment_current_alignment')
CACHE=STD/'tables/cached_field_differences'
sys.path.insert(0,str(SOURCE))
from shared import registration as reg
from shared.v2_registration_precision.refinement import register_refined
METHODS=['rerun_round','ideal_bilinear','centers_round','centers_bilinear']+[f'offset_{i}_{j}' for i in range(4) for j in range(4)]
METHODS+=['rerun_round_center_valid','ideal_bilinear_center_valid']
SAMPLES='samples_fourier';AUDITS='audits_fourier'
def inventory():
    inv=pd.read_csv(SOURCE/'data/results/v57_false_positive_facts_20261006/step0/fields_inventory.csv',dtype={'date':str,'board':str})
    fixed=pd.read_csv(SOURCE/'data/results/v57_false_positive_facts_20261006/stepC/C1_outlier29_fields.csv',dtype={'date':str,'board':str})
    dates=set(fixed.date); wanted=set(fixed.fid)
    t=inv[inv.fid.isin(wanted)|((inv.group=='blank')&inv.date.isin(dates))].copy()
    t['fixed29']=t.fid.isin(wanted)
    for r in t.itertuples():
        for p in [r.pre_path,r.post_path,CACHE/(r.fid+'.npz')]:
            if not Path(p).is_file(): raise FileNotFoundError(str(p))
    t.to_csv(OUT/'selected_fields.csv',index=False); return t
def lattice_centers(im,guess=7.343):
    h,w=im.shape; c=im-cv2.GaussianBlur(im,(51,51),0)
    win=np.outer(np.hanning(h),np.hanning(w)); f=abs(np.fft.fftshift(np.fft.fft2(c*win)))
    fy=np.fft.fftshift(np.fft.fftfreq(h)); fx=np.fft.fftshift(np.fft.fftfreq(w)); rad=np.hypot(fy[:,None],fx[None,:])
    cand=(ndimage.maximum_filter(f,9)==f)&(rad>2/(np.sqrt(3)*guess)*.94)&(rad<2/(np.sqrt(3)*guess)*1.06)
    iy,ix=np.where(cand); order=np.argsort(f[iy,ix])[::-1]; ks=[]
    par=lambda a,b,c: .5*(a-c)/(a-2*b+c) if a-2*b+c!=0 else 0
    for n in order:
        y,x=iy[n],ix[n]; k=np.array([fx[x]+par(f[y,x-1],f[y,x],f[y,x+1])/w,fy[y]+par(f[y-1,x],f[y,x],f[y+1,x])/h])
        if k[1]<0 or any(abs(k@q/np.linalg.norm(k)/np.linalg.norm(q))>.8 for q in ks): continue
        ks.append(k)
        if len(ks)==2: break
    # Continuous Fourier refinement removes sub-bin drift across the 2048 px field.
    # A two-pixel stride stays below the Nyquist limit for the first lattice shell.
    yy=np.arange(0,h,2);xx=np.arange(0,w,2);cw=(c*win)[::2,::2]
    def coef(k):return np.exp(-2j*np.pi*k[1]*yy)@(cw@np.exp(-2j*np.pi*k[0]*xx))
    with threadpool_limits(limits=1):
        ks=[optimize.minimize(lambda v:-abs(coef(v))/1e8,k,method='Nelder-Mead',options={'xatol':1e-9,'fatol':1e-6,'maxiter':100}).x for k in ks]
        phase=[np.angle(coef(k)) for k in ks]
    K=np.array(ks); B=np.linalg.inv(K)
    origin=np.linalg.solve(K,-np.array(phase)/(2*np.pi))
    # All detected peaks form a fit support; they are NOT candidate defect masks or positive labels.
    cc=cv2.GaussianBlur(c,(0,0),.6); maxima=(ndimage.maximum_filter(cc,5)==cc)&(cc>np.percentile(cc,60))
    y,x=np.where(maxima); inside=(x>30)&(x<w-30)&(y>30)&(y<h-30); x=x[inside];y=y[inside]
    dx=.5*(cc[y,x-1]-cc[y,x+1])/(cc[y,x-1]-2*cc[y,x]+cc[y,x+1]+1e-12)
    dy=.5*(cc[y-1,x]-cc[y+1,x])/(cc[y-1,x]-2*cc[y,x]+cc[y+1,x]+1e-12)
    points=np.column_stack([x+np.clip(dx,-.5,.5),y+np.clip(dy,-.5,.5)])
    ids=np.rint((points-origin)@K.T).astype(int); design=np.column_stack([np.ones(len(ids)),ids])
    model=np.vstack([origin,B.T]); keep=np.linalg.norm(design@model-points,axis=1)<1.7
    for _ in range(5):
        model=np.linalg.lstsq(design[keep],points[keep],rcond=None)[0]
        err=np.linalg.norm(design@model-points,axis=1)
        keep=err<max(.4,min(1.2,3*np.median(err[keep])))
    normids=ids/300.; X=np.column_stack([design,normids[:,0]**2,normids[:,0]*normids[:,1],normids[:,1]**2])
    quad=np.linalg.lstsq(X[keep],points[keep],rcond=None)[0]
    errq=np.linalg.norm(X@quad-points,axis=1)
    useq=np.median(errq[keep])<.8*np.median(err[keep]) and np.percentile(np.linalg.norm(X[keep]@quad-design[keep]@model,axis=1),95)<1.0
    return model,(quad if useq else None),dict(pitch_px=float(np.mean(np.linalg.norm(model[1:],axis=1))),
        fitted_peaks=int(keep.sum()),residual_median_px=float(np.median((errq if useq else err)[keep])),
        residual95_px=float(np.percentile((errq if useq else err)[keep],95)),quadratic=bool(useq))
def evaluate(ids,model,quad):
    ids=np.asarray(ids); X=np.column_stack([np.ones(len(ids)),ids])
    if quad is None:return X@model
    norm=ids/300.; return np.column_stack([X,norm[:,0]**2,norm[:,0]*norm[:,1],norm[:,1]**2])@quad
def nearest(points,model,quad):
    ids=np.rint((points-model[0])@np.linalg.inv(model[1:])).astype(int)
    # Search nine cells to handle the nonorthogonal hexagonal basis.
    offsets=np.array([(i,j) for i in [-1,0,1] for j in [-1,0,1]])
    cand=ids[:,None,:]+offsets; pos=evaluate(cand.reshape(-1,2),model,quad).reshape(-1,9,2)
    choice=np.argmin(np.linalg.norm(pos-points[:,None,:],axis=2),axis=1)
    return pos[np.arange(len(points)),choice],cand[np.arange(len(points)),choice]
def one(rec):
    fid=rec['fid']; dst=OUT/SAMPLES/f'{fid}.npz'; audit=OUT/AUDITS/f'{fid}.json'
    if dst.exists() and audit.exists(): return json.loads(audit.read_text(encoding='utf8'))
    a=M.read(rec['pre_path']); b=M.read(rec['post_path']); z=np.load(CACHE/(fid+'.npz'))
    xy=z['xy']; original=z['delta']; ids=z['ids']
    coarse,qc=reg.register_image_pair_affine(a,b,mask_stains=False,return_qc=True)
    matrix,ref=register_refined(a,b,stage='subpixel',initial=coarse,mask_stains=False)
    postxy=xy@matrix[:,:2].T+matrix[:,2]
    ca,cb=M.contrast(a),M.contrast(b)
    ma,qa,infoa=lattice_centers(a); mb,qb,infob=lattice_centers(b)
    centers,centerids=nearest(xy,ma,qa)
    expected=centers@matrix[:,:2].T+matrix[:,2]; postcenters,_=nearest(expected,mb,qb)
    matcherr=np.linalg.norm(postcenters-expected,axis=1); common=matcherr<7.343/4
    _,first=np.unique(centerids,axis=0,return_index=True); unique=np.zeros(len(xy),bool);unique[first]=True;common &=unique
    data={}
    data['rerun_round']=M.sample(ca,xy)-M.sample(cb,postxy)
    data['ideal_bilinear']=M.sample(ca,xy,True)-M.sample(cb,postxy,True)
    data['centers_round']=M.sample(ca,centers)-M.sample(cb,postcenters)
    data['centers_bilinear']=M.sample(ca,centers,True)-M.sample(cb,postcenters,True)
    data['rerun_round_center_valid']=data['rerun_round'].copy();data['rerun_round_center_valid'][~common]=np.nan
    data['ideal_bilinear_center_valid']=data['ideal_bilinear'].copy();data['ideal_bilinear_center_valid'][~common]=np.nan
    # Both native-frame coordinates receive the same camera offset; transform fixed.
    for i in range(4):
        for j in range(4):
            off=np.array([i*.25,j*.25]); data[f'offset_{i}_{j}']=M.sample(ca,xy+off)-M.sample(cb,postxy+off)
    for k in data:
        if k.startswith('centers'):data[k][~common]=np.nan
    finite=np.isfinite(data['rerun_round'])&np.isfinite(original)
    r={**rec,'matrix':matrix.tolist(),'pre_lattice':infoa,'post_lattice':infob,
       'center_match_fraction':float(common.mean()),'center_match_med_px':float(np.median(matcherr)),
       'stored_vs_rerun_rmse':float(np.sqrt(np.mean((original[finite]-data['rerun_round'][finite])**2))),
       'stored_vs_rerun_corr':float(np.corrcoef(original[finite],data['rerun_round'][finite])[0,1]),
       'pre_sha256':M.fingerprint(rec['pre_path']),'post_sha256':M.fingerprint(rec['post_path'])}
    np.savez_compressed(dst,xy=xy,centerxy=centers,ids=ids,stored=original,**data)
    audit.write_text(json.dumps(r,ensure_ascii=False,indent=2,default=str),encoding='utf8')
    print('DONE',fid,'match',r['center_match_fraction'],'pitch',infoa['pitch_px'],'baseline correlation',r['stored_vs_rerun_corr'],flush=True)
    return r
def summarize():
    f=pd.read_csv(OUT/'selected_fields.csv',dtype={'date':str,'board':str}); rows=[]; th=[]; maps={}
    for date,g in f.groupby('date'):
        pools={m:[] for m in METHODS+['stored']}
        for r in g[g.group=='blank'].itertuples():
            with np.load(OUT/SAMPLES/f'{r.fid}.npz') as z:
                for m in pools:pools[m].append(z[m][np.isfinite(z[m])])
        thresholds={}
        for m,values in pools.items():
            d=np.concatenate(values); thresholds[m]=float(d.mean()+3*d.std(ddof=0))
            th.append(dict(date=date,method=m,blank_fields=len(values),n=len(d),mean=float(d.mean()),sd=float(d.std()),threshold=thresholds[m]))
        for r in g.itertuples():
            with np.load(OUT/SAMPLES/f'{r.fid}.npz') as z:
                baseline,dm=M.band_metrics(z['xy'],z['rerun_round'],thresholds['rerun_round'],True)
                for m in METHODS+['stored']:
                    xy=z['centerxy'] if m.startswith('centers') else z['xy']
                    met,dmap=M.band_metrics(xy,z[m],thresholds[m],True)
                    pair=np.isfinite(dm)&np.isfinite(dmap)
                    corr=float(np.corrcoef(dm[pair],dmap[pair])[0,1]) if np.std(dmap[pair])>0 else np.nan
                    xx,yy=np.meshgrid(np.arange(64)*32,np.arange(64)*32); wave=np.exp(-2j*np.pi*(baseline['peak_fx']*xx+baseline['peak_fy']*yy))*np.outer(np.hanning(64),np.hanning(64))
                    phase=float(np.angle(np.sum(np.nan_to_num(dmap-np.nanmean(dmap))*wave)))
                    rows.append(dict(fid=r.fid,date=date,board=r.board,field=r.field,group=r.group,fixed29=bool(r.fixed29),method=m,threshold=thresholds[m],baseline_map_corr=corr,phase_at_baseline_frequency=phase,**met))
                    if r.fixed29 and m in ['stored','rerun_round','ideal_bilinear','centers_round','centers_bilinear','offset_1_1','offset_2_2','offset_3_3']:maps[r.fid+'__'+m]=dmap
    pd.DataFrame(rows).to_csv(OUT/'sampling_metrics.csv',index=False);pd.DataFrame(th).to_csv(OUT/'thresholds_by_date_method.csv',index=False)
    np.savez_compressed(OUT/'density_maps.npz',**maps)
    plot(pd.DataFrame(rows),maps)
def plot(t,maps):
    plt=M.plt
    g=t[t.fixed29]; base=g[g.method=='rerun_round'].set_index('fid')
    fig,ax=plt.subplots(1,3,figsize=(14,4))
    for m in ['ideal_bilinear','centers_round','centers_bilinear']:
        q=g[g.method==m].set_index('fid').loc[base.index]
        ax[0].scatter(base.rate*100,q.rate*100,label=m,s=18)
        ax[1].scatter(base.strength,q.strength,label=m,s=18)
        ax[2].scatter(base.density_sd,q.density_sd,label=m,s=18)
    for a in ax:a.legend(fontsize=7)
    ax[0].set(xlabel='Rerun rounded positive rate (%)',ylabel='Changed reading rate (%)')
    ax[1].set(xlabel='Rounded directional power fraction',ylabel='Changed directional power fraction')
    ax[2].set(xlabel='Rounded density-map SD',ylabel='Changed density-map SD')
    fig.tight_layout();fig.savefig(OUT/'step1_comparison.png',dpi=160);plt.close(fig)
    off=g[g.method.str.startswith('offset')].copy(); off['x']=off.method.str.split('_').str[1].astype(int);off['y']=off.method.str.split('_').str[2].astype(int)
    fig,ax=plt.subplots(1,3,figsize=(13,4))
    for a,col in zip(ax,['rate','strength','baseline_map_corr']):
        im=a.imshow(off.groupby(['y','x'])[col].median().unstack().to_numpy(),origin='lower',extent=[-.125,.875,-.125,.875]);a.set(title='fixed29 median '+col,xlabel='x offset (px)',ylabel='y offset (px)');fig.colorbar(im,ax=a)
    fig.tight_layout();fig.savefig(OUT/'step2_offsets.png',dpi=160);plt.close(fig)
    for fid in g.fid.unique():
        keys=[fid+'__'+m for m in ['stored','rerun_round','ideal_bilinear','centers_bilinear','offset_1_1','offset_2_2','offset_3_3']]
        fig,ax=plt.subplots(1,len(keys),figsize=(19,3)); vmax=max(np.nanpercentile(maps[k],99) for k in keys)
        for a,k in zip(ax,keys):a.imshow(maps[k],origin='upper',vmin=0,vmax=vmax,cmap='magma');a.set_title(k.split('__')[1],fontsize=8);a.axis('off')
        fig.suptitle(fid+'; common color scale');fig.tight_layout();fig.savefig(OUT/'atlas'/f'{fid}.png',dpi=130);plt.close(fig)
def main():
    for d in [SAMPLES,AUDITS,'atlas']:(OUT/d).mkdir(exist_ok=True)
    f=inventory();print('SELECTED',len(f),'fixed',f.fixed29.sum(),'blanks',(f.group=='blank').sum(),flush=True)
    if '--summary' not in sys.argv:
        records=f.to_dict('records')
        if '--pilot' in sys.argv: records=records[:1]
        with Pool(3) as p: result=p.map(one,records,chunksize=1)
        pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in result]).to_csv(OUT/'fit_audit.csv',index=False)
        if '--pilot' in sys.argv:return
    summarize()
if __name__=='__main__':main()
