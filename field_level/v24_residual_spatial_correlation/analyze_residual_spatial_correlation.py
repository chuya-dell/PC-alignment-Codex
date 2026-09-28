"""Diagnose spatial correlation in independent-center registration residuals."""
from __future__ import annotations
import argparse, csv, gzip, importlib.util, json, math, os, sys
import zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'field_level/v17_independent_pillar_center_residual'))
import field_measure_independent_center_residuals as v17
from shared.registration import load_image_unicode

PITCH = 7.286
HALF_PITCH = PITCH / 2
Q = .50
MAX_RANGE_PITCH = 10.0
MAX_ANCHORS = 768
RNG_SEED = 20260928
FOCUS = ['260825_p50_dna_1_7','260829_p50_sam_3_4','260829_p50_sam_7_8','260826_p50_sam_11_7']

def shells(max_pitch=MAX_RANGE_PITCH):
    vals=set()
    n=int(max_pitch)+2
    for i in range(-n,n+1):
        for j in range(-n,n+1):
            d2=i*i+j*j+i*j
            if d2 and math.sqrt(d2)<=max_pitch+.51: vals.add(round(math.sqrt(d2),8))
    return np.asarray(sorted(vals),float)

SHELLS=shells()
EDGES=np.r_[0.,(SHELLS[:-1]+SHELLS[1:])/2,(SHELLS[-1]+MAX_RANGE_PITCH+.7)/2]

def robust_vectors(r):
    r=np.asarray(r,float)
    center=np.median(r,axis=0)
    centered=r-center
    norms=np.linalg.norm(centered,axis=1)
    med=float(np.median(norms))
    mad=float(np.median(np.abs(norms-med)))
    cutoff=max(med+3*1.4826*mad, 1e-8)
    clipped=np.minimum(1.,cutoff/np.maximum(norms,1e-12))
    return centered*clipped[:,None],center,cutoff,int((norms>cutoff).sum())

def pair_bins(xy, r, seed):
    """Sample spatial pairs from uniformly selected lower-index anchors."""
    n=len(xy)
    if n<100: return pd.DataFrame(columns=['shell_pitch','n_pairs','covariance_dot_px2','semivariogram_px2'])
    rng=np.random.default_rng(seed)
    anchors=np.sort(rng.choice(n,size=min(MAX_ANCHORS,n),replace=False))
    tree=cKDTree(xy)
    pairs=[]
    for i in anchors:
        js=tree.query_ball_point(xy[i],r=(MAX_RANGE_PITCH+.51)*PITCH)
        js=[j for j in js if j>i]
        if js: pairs.extend((i,j) for j in js)
    if not pairs: return pd.DataFrame(columns=['shell_pitch','n_pairs','covariance_dot_px2','semivariogram_px2'])
    ij=np.asarray(pairs,dtype=np.int32)
    delta=xy[ij[:,0]]-xy[ij[:,1]]
    dist=np.linalg.norm(delta,axis=1)/PITCH
    bi=np.digitize(dist,EDGES)-1
    dot=np.einsum('ij,ij->i',r[ij[:,0]],r[ij[:,1]])
    gamma=.5*np.einsum('ij,ij->i',r[ij[:,0]]-r[ij[:,1]],r[ij[:,0]]-r[ij[:,1]])
    rows=[]
    for b in np.unique(bi):
        m=bi==b
        if b<0 or b>=len(SHELLS) or not m.any(): continue
        rows.append({'shell_pitch':SHELLS[b],'mean_pair_distance_pitch':float(dist[m].mean()),
                     'n_pairs':int(m.sum()),'covariance_dot_px2':float(dot[m].mean()),
                     'semivariogram_px2':float(gamma[m].mean())})
    return pd.DataFrame(rows)

def fit_variogram(bins,sill_reference):
    b=bins[bins.n_pairs>=40].copy()
    if len(b)<5: return dict(nugget_px2=np.nan,sill_px2=np.nan,structured_variance_px2=np.nan,
        structured_rms_px=np.nan,correlated_fraction=np.nan,range_pitch=np.nan,correlation_distance_pitch=np.nan,
        range_censored=False,fit_rmse_px2=np.nan)
    h=b.mean_pair_distance_pitch.to_numpy(); y=b.semivariogram_px2.to_numpy(); w=np.sqrt(b.n_pairs.to_numpy()/b.n_pairs.max())
    sill0=max(float(sill_reference),1e-8)
    def residual(p):
        nug,ran=p;struct=max(sill0-nug,0.)
        return (nug+struct*(1-np.exp(-h/ran))-y)*w
    try:
        res=least_squares(residual,[max(.2*sill0,1e-8),3.0],
            bounds=([0,.15],[sill0,200]),max_nfev=300,loss='soft_l1',f_scale=max(sill0*.05,1e-6))
        nug,ran=map(float,res.x);sill=sill0;struct=max(sill-nug,0.)
        return dict(nugget_px2=nug,sill_px2=sill,structured_variance_px2=struct,
            structured_rms_px=float(np.sqrt(struct)),correlated_fraction=float(struct/sill) if sill>1e-12 else 0.,
            range_pitch=ran,correlation_distance_pitch=3*ran,range_censored=bool(ran>=MAX_RANGE_PITCH),
            fit_rmse_px2=float(np.sqrt(np.mean(res.fun**2))))
    except Exception:
        return dict(nugget_px2=np.nan,sill_px2=np.nan,structured_variance_px2=np.nan,structured_rms_px=np.nan,
            correlated_fraction=np.nan,range_pitch=np.nan,correlation_distance_pitch=np.nan,range_censored=False,fit_rmse_px2=np.nan)

def summarize_vectors(fov,kind,xy,raw,seed):
    r,center,cutoff,nclip=robust_vectors(raw)
    bins=pair_bins(xy,r,seed)
    robust_var=float(np.mean(np.sum(r*r,axis=1)))
    fit=fit_variogram(bins,robust_var)
    radial=np.linalg.norm(raw,axis=1)
    norm=fit.get('sill_px2',np.nan)
    rec={'fov_key':fov,'kind':kind,'n_centers':len(raw),'raw_median_residual_px':float(np.median(radial)),
         'raw_rms_vector_px':float(np.sqrt(np.mean(radial**2))),'robust_center_x_px':float(center[0]),
         'robust_center_y_px':float(center[1]),'winsor_cutoff_px':cutoff,'n_winsorized':nclip,
         'robust_total_variance_px2':robust_var,**fit,
         'pair_sample_count':int(bins.n_pairs.sum()) if len(bins) else 0}
    bins.insert(0,'kind',kind);bins.insert(0,'fov_key',fov)
    return rec,bins,r

def matrix_for(key,registration_dir):
    d=json.loads((registration_dir/'diagnostics'/f'{key}.json').read_text(encoding='utf-8'))
    return np.asarray(d['matrix'],float)

def data_path(s,root):
    norm=str(s).replace('\\','/')
    markers=('4.生データ D/','4.生データD_remo/')
    tail=None
    for marker in markers:
        if marker in norm:
            tail=norm.split(marker,1)[1];break
    if tail is None: tail='/'.join(norm.split('/')[-3:])
    return str(root/Path(tail))

def image_pair_tasks(root,registration_dir):
    manifest=pd.read_csv(ROOT/'data/results/v10_phase2_bright_band_mask_20260926/masked_50862c8/source_manifest_attached.csv')
    summary=pd.read_csv(registration_dir/'summary_0/phase2_reanalysis_fov_summary.csv')
    accepted=summary[summary.status=='ok']
    ok={f"{r.dataset}_{int(r['sample'])}_{int(r.position)}" for _,r in accepted.iterrows()}
    tasks=[]
    for _,r in manifest.iterrows():
        key=f"{r.dataset}_{int(r['sample'])}_{int(r.position)}"
        if key in ok: tasks.append(dict(key=key,pre=data_path(r.pre_path,root),post=data_path(r.post_path,root)))
    return tasks

def extract_pair(task,matrix):
    pre=load_image_unicode(task['pre']);post=load_image_unicode(task['post'])
    if pre is None or post is None: raise FileNotFoundError(task['key'])
    pp,ps,_=v17.detect_centers(pre);qq,qs,_=v17.detect_centers(post)
    rec,v=v17.residual_record(task['key'],'current_spatial_subpixel',matrix,pp,qq,ps,qs,Q,pre.shape,True)
    p,pred,t,resid,distance,inlier=v
    return rec,p[inlier],resid[inlier],(pre,post,pp,qq,ps,qs,matrix)

def load_semisynthetic(root):
    cases=json.loads((ROOT/'data/results/v11_registration_precision_20260926/baseline_inputs.json').read_text(encoding='utf-8'))
    groups={}
    for c in cases:
        src=data_path(c['path'],root);groups.setdefault(src,[]).append(c)
    return groups

def run_semisynthetic(out,root,workers):
    groups=load_semisynthetic(root);rows=[];bins_all=[]
    def source_job(item):
        path,cs=item;image=load_image_unicode(path)
        if image is None: raise FileNotFoundError(path)
        pre,ps,_=v17.detect_centers(image);outrows=[];outbins=[];outvectors=[]
        for case in cs:
            m=np.asarray(case['truth'],float)
            if np.max(np.abs(m-np.c_[np.eye(2),np.zeros(2)]))<1e-8: moving=image;post,qs=pre,ps
            else:
                moving=cv2.warpAffine(image,m.astype(np.float32),(image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                post,qs,_=v17.detect_centers(moving)
            _,vv=v17.residual_record(case['case_id'],'known_truth',m,pre,post,ps,qs,Q,image.shape,True)
            p,pred,t,r,d,inside=vv;xy=p[inside];r=r[inside]
            rec,bb,_=summarize_vectors(case['case_id'],'semisynthetic',xy,r,zlib.crc32(case['case_id'].encode()))
            rec.update(source=str(Path(path).relative_to(root)),truth_scale=float(np.sqrt(np.linalg.det(m[:,:2]))),
                       truth_rotation_deg=float(np.degrees(np.arctan2(m[1,0],m[0,0]))))
            outrows.append(rec);outbins.append(bb)
            outvectors.append(pd.DataFrame({'fov_key':case['case_id'],'x_px':xy[:,0],'y_px':xy[:,1],
                'residual_x_px':r[:,0],'residual_y_px':r[:,1],
                'residual_magnitude_px':np.linalg.norm(r,axis=1),'condition':'known_truth'}))
        return outrows,outbins,outvectors
    with gzip.open(out/'semisynthetic_center_residual_vectors.csv.gz','wt',encoding='utf-8',newline='') as gz:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs=[pool.submit(source_job,x) for x in groups.items()]
            for i,f in enumerate(as_completed(fs),1):
                rr,bb,vv=f.result();rows.extend(rr);bins_all.extend(bb)
                for frame in vv:
                    frame.to_csv(gz,index=False,header=(gz.tell()==0))
                if i%3==0 or i==len(fs): print(f'semi-synthetic source images {i}/{len(fs)}',flush=True)
    pd.DataFrame(rows).to_csv(out/'semisynthetic_by_condition.csv',index=False)
    pd.concat(bins_all,ignore_index=True).to_csv(out/'semisynthetic_variogram_by_condition.csv',index=False)
    return rows

def select_sensitivity_cases(groups):
    chosen=[]
    for path,cases in groups.items():
        # Choose the case nearest the group's median non-identity transform magnitude.
        mags=[]
        for c in cases:
            m=np.asarray(c['truth'],float);mags.append(np.linalg.norm(m[:,:2]-np.eye(2))+np.linalg.norm(m[:,2]))
        target=np.median(mags);idx=int(np.argmin(np.abs(np.asarray(mags)-target)))
        chosen.append((path,cases[idx]))
    return chosen

def deform_image(image,amplitude,wavelength):
    h,w=image.shape;x,y=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32))
    disp=amplitude*np.sin(2*np.pi*x/wavelength)
    return cv2.remap(image,x-disp,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101),disp

def run_sensitivity(out,root,workers):
    groups=load_semisynthetic(root);chosen=select_sensitivity_cases(groups);rows=[];bins_all=[]
    settings=[(a,f) for a in (.3,.6,1.) for f in (1/8,1/4,1/2)]
    for k,(path,case) in enumerate(chosen,1):
        image=load_image_unicode(path);m=np.asarray(case['truth'],float)
        pre,ps,_=v17.detect_centers(image)
        affine=cv2.warpAffine(image,m.astype(np.float32),(image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        for amp,frac in settings:
            deformed,_=deform_image(affine,amp,frac*image.shape[1]);post,qs,_=v17.detect_centers(deformed)
            _,vv=v17.residual_record(case['case_id'],f'sensitivity_{amp}_{frac}',m,pre,post,ps,qs,Q,image.shape,True)
            p,pred,t,r,d,inside=vv;xy=p[inside];r=r[inside]
            rec,bb,_=summarize_vectors(case['case_id'],'sensitivity',xy,r,RNG_SEED+k*100+int(amp*100)+int(frac*1000))
            rec.update(source=str(Path(path).relative_to(root)),amplitude_px=amp,wavelength_fraction_width=frac,
                       wavelength_px=frac*image.shape[1],truth_case=case['case_id'])
            rows.append(rec);bb['amplitude_px']=amp;bb['wavelength_fraction_width']=frac;bins_all.append(bb)
        if k%5==0 or k==len(chosen):print(f'sensitivity source images {k}/{len(chosen)}',flush=True)
    pd.DataFrame(rows).to_csv(out/'sensitivity_by_condition.csv',index=False)
    pd.concat(bins_all,ignore_index=True).to_csv(out/'sensitivity_variogram_by_condition.csv',index=False)
    return rows

def run_real(out,root,registration_dir,workers):
    tasks=image_pair_tasks(root,registration_dir);rows=[];allbins=[];focus_arrays={};top_arrays={}
    def job(t):
        m=matrix_for(t['key'],registration_dir)
        rec,xy,r,images=extract_pair(t,m)
        summary,bins,rr=summarize_vectors(t['key'],'real',xy,r,zlib.crc32(t['key'].encode()))
        summary.update(dataset=rec.get('dataset','_'.join(t['key'].split('_')[:-2])),
                       same_cell_median_px=rec['same_cell_median_px'],same_cell_p95_px=rec['same_cell_p95_px'],
                       cell_mismatch_fraction=rec['cell_mismatch_fraction'],n_same_cell_matches=rec['n_same_cell_matches'])
        frame=pd.DataFrame({'fov_key':t['key'],'x_px':xy[:,0],'y_px':xy[:,1],
            'residual_x_px':r[:,0],'residual_y_px':r[:,1],
            'residual_magnitude_px':np.linalg.norm(r,axis=1),'condition':'current_spatial_subpixel'})
        return summary,bins,(t['key'],xy,rr),frame
    with gzip.open(out/'real_center_residual_vectors.csv.gz','wt',encoding='utf-8',newline='') as gz:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fs=[pool.submit(job,t) for t in tasks]
            for i,f in enumerate(as_completed(fs),1):
                rr,bb,vec,frame=f.result();rows.append(rr);allbins.append(bb)
                frame.to_csv(gz,index=False,header=(gz.tell()==0))
                key,xy,rv=vec
                if key in FOCUS: focus_arrays[key]=(xy,rv)
                top_arrays[key]=(rr['structured_rms_px'],xy,rv)
                if len(top_arrays)>15:
                    worst=sorted(top_arrays,key=lambda k:top_arrays[k][0])[:1]
                    for drop in worst:
                        if drop not in FOCUS: top_arrays.pop(drop,None)
                if i%10==0 or i==len(fs):print(f'real FOVs {i}/{len(fs)}',flush=True)
    pd.DataFrame(rows).sort_values('fov_key').to_csv(out/'real_by_fov.csv',index=False)
    pd.concat(allbins,ignore_index=True).to_csv(out/'real_variogram_by_fov.csv',index=False)
    chosen={k:(v[1],v[2]) for k,v in top_arrays.items() if k not in FOCUS}
    np.savez_compressed(out/'selected_plot_vectors.npz',**{f'{k}_xy':v[0] for k,v in {**focus_arrays,**chosen}.items()},
                        **{f'{k}_r':v[1] for k,v in {**focus_arrays,**chosen}.items()})
    return rows,{**focus_arrays,**chosen}

def summarize_v21(out):
    p=ROOT/'data/results/v21_ambiguous_correspondence_diagnosis_20260928/ambiguous_residual_vectors.csv.gz'
    chunks=[]
    for d in pd.read_csv(p,chunksize=100000):
        d=d[d.group.isin(['high','control'])]
        d['tile_x']=np.minimum((d.x_px/2048*5).astype(int),4);d['tile_y']=np.minimum((d.y_px/2044*5).astype(int),4)
        chunks.append(d)
    d=pd.concat(chunks,ignore_index=True)
    rows=[]
    for (fov,group,tx,ty),g in d.groupby(['fov_key','group','tile_x','tile_y']):
        v=g[['residual_x_px','residual_y_px']].to_numpy();mags=np.linalg.norm(v,axis=1);n=len(v)
        if n<10: continue
        resultant=float(np.linalg.norm(v.mean(axis=0))/(mags.mean()+1e-12))
        # For isotropic random directions, a Rayleigh approximation gives E[R] ~= sqrt(pi)/(2 sqrt(n)).
        expected=float(np.sqrt(np.pi)/(2*np.sqrt(n)))
        rows.append(dict(fov_key=fov,group=group,tile_x=tx,tile_y=ty,n_vectors=n,resultant_length_ratio=resultant,
                         random_direction_expected=expected,resultant_excess=resultant-expected))
    d=pd.DataFrame(rows);d.to_csv(out/'v21_ambiguous_tile_coherence.csv',index=False)
    summary=d.groupby('group').agg(n_tiles=('n_vectors','size'),median_n=('n_vectors','median'),
        median_resultant=('resultant_length_ratio','median'),median_random_expected=('random_direction_expected','median'),
        fraction_above_random_95=('resultant_length_ratio',lambda s:float((s>np.sqrt(-np.log(.05)/np.maximum(d.loc[s.index,'n_vectors'],1))).mean()))).reset_index()
    summary.to_csv(out/'v21_ambiguous_tile_coherence_summary.csv',index=False)
    return d

def plots(out,real_rows,focus_arrays,semi_rows,sensitivity_rows,v21tiles):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Meiryo'
    od=out/'visualizations';od.mkdir(exist_ok=True)
    # Global semivariogram curves, averaged by field/condition so large fields do not dominate.
    for name in ['semisynthetic','real']:
        f=out/('semisynthetic_variogram_by_condition.csv' if name=='semisynthetic' else 'real_variogram_by_fov.csv')
        d=pd.read_csv(f);fig,ax=plt.subplots(figsize=(8,5))
        z=d.groupby('shell_pitch').semivariogram_px2.agg(['median',lambda x:np.quantile(x,.25),lambda x:np.quantile(x,.75)]).reset_index()
        ax.plot(z.shell_pitch,z['median'],marker='o');ax.fill_between(z.shell_pitch,z.iloc[:,2],z.iloc[:,3],alpha=.2)
        ax.set(xlabel='距離（格子ピッチ）',ylabel='半変動関数（px²）',title=f'{name}残差の空間半変動関数');fig.tight_layout();fig.savefig(od/f'{name}_variogram.png',dpi=150);plt.close(fig)
    s=pd.DataFrame(sensitivity_rows);fig,ax=plt.subplots(figsize=(8,5))
    for a,g in s.groupby('amplitude_px'):
        x=g.groupby('wavelength_fraction_width').structured_rms_px.median()
        ax.plot(x.index.astype(str),x.values,marker='o',label=f'{a:g} px')
    ax.set(xlabel='波長（視野幅に対する比）',ylabel='推定相関成分（px）',title='既知の滑らかな局所変位に対する感度');ax.legend();fig.tight_layout();fig.savefig(od/'sensitivity_curve.png',dpi=150);plt.close(fig)
    real=pd.DataFrame(real_rows).sort_values('structured_rms_px',ascending=False)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(real.structured_rms_px,real.raw_median_residual_px,s=18,alpha=.65)
    ax.set(xlabel='空間相関成分（px）',ylabel='視野内残差中央値（px）',title='実データ401視野');fig.tight_layout();fig.savefig(od/'real_correlated_vs_median.png',dpi=150);plt.close(fig)
    map_keys=list(dict.fromkeys(list(real.head(10).fov_key)+[k for k in FOCUS if k in focus_arrays]))
    z=np.load(out/'selected_plot_vectors.npz')
    for key in map_keys:
        # Recover coordinates/vectors from the raw files retained in results.
        xy=z[f'{key}_xy'];rv=z[f'{key}_r']
        fig,ax=plt.subplots(figsize=(7,6));ax.quiver(xy[:,0],xy[:,1],rv[:,0]*10,rv[:,1]*10,angles='xy',scale_units='xy',scale=1,color='tab:blue',width=.002)
        ax.set_xlim(0,2048);ax.set_ylim(2044,0);ax.set_title(f'{key}: 残差矢印は10倍表示');ax.set_aspect('equal');fig.tight_layout();fig.savefig(od/f'{key}_residual_vectors.png',dpi=150);plt.close(fig)
    # Feature correlations use v21's precomputed per-field features.
    feat=pd.read_csv(ROOT/'data/results/v21_ambiguous_correspondence_diagnosis_20260928/fov_features_401.csv')
    joined=pd.DataFrame(real_rows).merge(feat,on='fov_key',how='inner',suffixes=('','_v21'))
    num=[c for c in feat.columns if c!='fov_key' and pd.api.types.is_numeric_dtype(feat[c])]
    cor=[]
    for c in num:
        ok=joined[[c,'structured_rms_px']].dropna()
        if len(ok)>10:
            rho,p=spearmanr(ok[c],ok.structured_rms_px);cor.append(dict(feature=c,n=len(ok),spearman_rho=rho,p_value=p))
    corrdf=pd.DataFrame(cor).sort_values('spearman_rho',key=lambda s:s.abs(),ascending=False)
    corrdf.to_csv(out/'quality_feature_rank_correlations.csv',index=False)
    show=[c for c in corrdf.feature.head(6) if c in joined.columns]
    if show:
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for ax,c in zip(axs.flat,show):
            ax.scatter(joined[c],joined.structured_rms_px,s=12,alpha=.55)
            rho=float(corrdf.loc[corrdf.feature==c,'spearman_rho'].iloc[0])
            ax.set_title(f'{c}\n順位相関={rho:.2f}',fontsize=8);ax.set_ylabel('空間相関成分（px）');ax.tick_params(labelsize=7)
        for ax in axs.flat[len(show):]: ax.set_visible(False)
        fig.tight_layout();fig.savefig(od/'quality_feature_scatterplots.png',dpi=150);plt.close(fig)
    return joined

def write_secondary_tables(out):
    real=pd.read_csv(out/'real_by_fov.csv');semi=pd.read_csv(out/'semisynthetic_by_condition.csv')
    threshold=float(semi.structured_rms_px.quantile(.95))
    real['above_semisynthetic_p95']=real.structured_rms_px>threshold
    real.groupby('dataset').agg(n_fovs=('fov_key','size'),structured_rms_median_px=('structured_rms_px','median'),
        structured_rms_q25_px=('structured_rms_px',lambda x:x.quantile(.25)),
        structured_rms_q75_px=('structured_rms_px',lambda x:x.quantile(.75)),
        structured_rms_p95_px=('structured_rms_px',lambda x:x.quantile(.95)),
        same_cell_median_px=('same_cell_median_px','median'),
        n_above_semisynthetic_p95=('above_semisynthetic_p95','sum')).reset_index().to_csv(out/'real_by_dataset_summary.csv',index=False)
    summaries=[]
    for kind,summary_file,variogram_file in [('real','real_by_fov.csv','real_variogram_by_fov.csv'),
        ('semisynthetic','semisynthetic_by_condition.csv','semisynthetic_variogram_by_condition.csv')]:
        a=pd.read_csv(out/summary_file);b=pd.read_csv(out/variogram_file);parts=[]
        for key,g in b.groupby('fov_key'):
            ix=(g.shell_pitch-1.).abs().idxmin();row=g.loc[ix];den=float(a.loc[a.fov_key==key,'robust_total_variance_px2'].iloc[0])
            parts.append(dict(fov_key=key,kind=kind,nearest_shell_pitch=float(row.shell_pitch),
                nearest_covariance_dot_px2=float(row.covariance_dot_px2),robust_total_variance_px2=den,
                nearest_neighbor_correlation=float(row.covariance_dot_px2/den) if den>0 else np.nan))
        v=pd.DataFrame(parts);v.to_csv(out/f'{kind}_nearest_neighbor_correlation_by_fov.csv',index=False)
        x=v.nearest_neighbor_correlation.dropna()
        summaries.append(dict(kind=kind,n_fovs=len(x),median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75),
            p95=x.quantile(.95),fraction_positive=float((x>0).mean()),
            fraction_above_real_null95=np.nan))
    pd.DataFrame(summaries).to_csv(out/'nearest_neighbor_correlation_summary.csv',index=False)
    # Use the semi-synthetic distribution as the empirical detector-and-interpolation reference.
    ss=pd.read_csv(out/'semisynthetic_nearest_neighbor_correlation_by_fov.csv').nearest_neighbor_correlation
    rn=pd.read_csv(out/'real_nearest_neighbor_correlation_by_fov.csv');cut=float(ss.quantile(.95))
    rn['above_semisynthetic_p95']=rn.nearest_neighbor_correlation>cut
    rn.to_csv(out/'real_nearest_neighbor_correlation_by_fov.csv',index=False)
    sm=pd.read_csv(out/'nearest_neighbor_correlation_summary.csv')
    sm.loc[sm.kind=='real','fraction_above_real_null95']=float(rn.above_semisynthetic_p95.mean())
    sm.to_csv(out/'nearest_neighbor_correlation_summary.csv',index=False)
    for kind,variogram_file in [('real','real_variogram_by_fov.csv'),('semisynthetic','semisynthetic_variogram_by_condition.csv')]:
        lag=pd.read_csv(out/variogram_file)
        lag.groupby('shell_pitch').agg(n_fovs=('fov_key','nunique'),
            median_covariance_dot_px2=('covariance_dot_px2','median'),
            q25_covariance_dot_px2=('covariance_dot_px2',lambda x:x.quantile(.25)),
            q75_covariance_dot_px2=('covariance_dot_px2',lambda x:x.quantile(.75)),
            median_semivariogram_px2=('semivariogram_px2','median'),
            q25_semivariogram_px2=('semivariogram_px2',lambda x:x.quantile(.25)),
            q75_semivariogram_px2=('semivariogram_px2',lambda x:x.quantile(.75)),
            median_pairs=('n_pairs','median')).reset_index().to_csv(out/f'{kind}_variogram_summary_by_shell.csv',index=False)
    meta=dict(version='v24',real_fovs=int(len(real)),semisynthetic_conditions=int(len(semi)),
        sensitivity_conditions=int(len(pd.read_csv(out/'sensitivity_by_condition.csv'))),
        independent_detector='v17 detect_centers and residual_record unchanged',confidence_quantile=Q,
        one_to_one_nearest_neighbor=True,half_pitch_cutoff_px=HALF_PITCH,pitch_px=PITCH,
        robust_winsor_rule='center residual vectors by component-wise median; cap centered-vector norm at median norm plus three scaled median absolute deviations',
        pair_sampling=f'uniform random anchors, maximum {MAX_ANCHORS} per FOV, deterministic CRC32 seed, lattice distance shells through about {MAX_RANGE_PITCH} pitches',
        variogram_model='exponential; sill fixed to robust empirical total variance to avoid unconstrained extrapolation',
        sensitivity_amplitudes_px=[.3,.6,1.],sensitivity_wavelength_fraction_of_width=[.125,.25,.5],
        sensitivity_direction='horizontal sinusoidal displacement',semi_p95_structured_rms_px=threshold,
        semi_p95_nearest_neighbor_correlation=float(ss.quantile(.95)))
    (out/'run_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['semisynthetic','sensitivity','real','v21','all'],default='all')
    ap.add_argument('--data-root',type=Path,default=Path(r'W:\GoogleDrive\fdtdremote\4.生データD_remo'))
    ap.add_argument('--registration-dir',type=Path,default=ROOT/'data/results/v16_real_spatial_subpixel_20260928')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'visualizations').mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(a.output/'.mplconfig');Path(os.environ['MPLCONFIGDIR']).mkdir(exist_ok=True)
    semi=sens=real=focus=v21tiles=None
    if a.phase in ('semisynthetic','all'): semi=run_semisynthetic(a.output,a.data_root,a.workers)
    if a.phase in ('sensitivity','all'): sens=run_sensitivity(a.output,a.data_root,a.workers)
    if a.phase in ('real','all'): real,focus=run_real(a.output,a.data_root,a.registration_dir,a.workers)
    if a.phase in ('v21','all'): v21tiles=summarize_v21(a.output)
    if a.phase=='all':
        plots(a.output,real,focus,semi,sens,v21tiles)
        # Field-level robust summaries and bootstrap confidence intervals.
        df=pd.DataFrame(real);metrics=['raw_median_residual_px','raw_rms_vector_px','nugget_px2','sill_px2','structured_variance_px2','structured_rms_px','correlated_fraction','range_pitch','correlation_distance_pitch']
        stats=[]
        for c in metrics:
            x=df[c].dropna();stats.append(dict(metric=c,n=len(x),median=x.median(),q25=x.quantile(.25),q75=x.quantile(.75),p95=x.quantile(.95),maximum=x.max()))
        pd.DataFrame(stats).to_csv(a.output/'real_metric_distribution.csv',index=False)
        baseline=pd.DataFrame(semi);threshold=baseline.structured_rms_px.quantile(.95)
        sensdf=pd.DataFrame(sens);base_map=baseline.set_index('fov_key').structured_rms_px
        sensdf['baseline_structured_rms_px']=sensdf.truth_case.map(base_map)
        sensdf['paired_increase_px']=sensdf.structured_rms_px-sensdf.baseline_structured_rms_px
        srng=np.random.default_rng(RNG_SEED);sens_summ=[]
        for (amp,wave),g in sensdf.groupby(['amplitude_px','wavelength_fraction_width']):
            delta=g.paired_increase_px.to_numpy();boots=np.asarray([np.median(srng.choice(delta,len(delta),replace=True)) for _ in range(2000)])
            sens_summ.append(dict(amplitude_px=amp,wavelength_fraction_width=wave,wavelength_px=float(g.wavelength_px.iloc[0]),
                n_source_cases=len(g),median_detected_component_px=float(g.structured_rms_px.median()),
                median_paired_increase_px=float(np.median(delta)),paired_increase_ci95_low=float(np.quantile(boots,.025)),
                paired_increase_ci95_high=float(np.quantile(boots,.975)),fraction_above_semisynthetic_p95=float((g.structured_rms_px>threshold).mean()),
                detectable=bool((g.structured_rms_px>threshold).mean()>=.8 and np.quantile(boots,.025)>0)))
        pd.DataFrame(sens_summ).to_csv(a.output/'sensitivity_detection_summary.csv',index=False)
        med_obs=float(df.same_cell_median_px.median())
        df['random_equivalent_median_px']=df.same_cell_median_px*np.sqrt(df.nugget_px2.clip(lower=0)/df.sill_px2.clip(lower=1e-12))
        df['structured_equivalent_median_px']=df.same_cell_median_px*np.sqrt(df.structured_variance_px2.clip(lower=0)/df.sill_px2.clip(lower=1e-12))
        pd.DataFrame([dict(observed_same_cell_median_px=med_obs,
            random_component_equivalent_median_px=float(df.random_equivalent_median_px.median()),
            structured_component_equivalent_median_px=float(df.structured_equivalent_median_px.median()),
            semi_structured_rms_p95_px=threshold,real_fovs_above_semi_p95=int((df.structured_rms_px>threshold).sum()),
            real_fraction_above_semi_p95=float((df.structured_rms_px>threshold).mean()),
            range_censored_fovs=int(df.range_censored.sum()))]).to_csv(a.output/'real_vs_semisynthetic_threshold.csv',index=False)
        write_secondary_tables(a.output)
        (a.output/'selected_plot_vectors.npz').unlink(missing_ok=True)

if __name__=='__main__':main()
