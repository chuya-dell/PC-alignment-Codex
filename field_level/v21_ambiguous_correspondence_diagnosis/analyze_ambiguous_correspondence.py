"""V21 read-only diagnosis of high half-pitch ambiguous center correspondences."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared.registration import load_image_unicode, image01_for_registration
from shared.lattice_indexing import hex_basis
from field_level.v17_independent_pillar_center_residual.field_measure_independent_center_residuals import (
    detect_centers, residual_record, independently_seeded_lattice, transform,
)

PITCH=7.286
HALF=PITCH/2
SEED=20260928
FOLDER_MAP={
    '260824_p50_SHC6OH':'260824_p50_SHC6OH','260825_p50_dna':'260825_p50_dna',
    '260826_p50_sam':'260826-p50-sam','260827_p50_dna':'260827_pp50_dna',
    '260827_pp50_dna':'260827_pp50_dna','260828_p50_dna':'260828-p50-dna',
    '260828_p50_sam':'260828-p50-SAM','260828_p50_SAM':'260828-p50-SAM',
    '260829_p50_dna':'260829_p50_DNA','260829_p50_DNA':'260829_p50_DNA',
    '260829_p50_sam':'260829-p50-sam',
}

def image_path(root,dataset,sample,position,phase):
    base=Path(root)/FOLDER_MAP[dataset]
    if dataset=='260824_p50_SHC6OH': base/='Raw_Images_生データのみ'
    return base/f'{sample}-{position}-{phase}.tif'

def read_matrix(root,key):
    p=Path(root)/'diagnostics'/f'{key}.json'
    if not p.is_file(): raise FileNotFoundError(p)
    return np.asarray(json.loads(p.read_text(encoding='utf-8'))['matrix'],float)

def wrapped(x): return x-np.round(x)

def image_features(im):
    a=np.asarray(im)
    n=a.astype(np.float32)/65535.
    return {'saturated_pixels':int((a>=55000).sum()),'mean':float(a.mean()),
            'contrast_std':float(a.std()),'sharpness_laplacian_variance':float(cv2.Laplacian(n,cv2.CV_32F).var())}

def cell_summary(vectors,basis):
    coeff=np.linalg.solve(basis,np.asarray(vectors).T).T
    frac=wrapped(coeff)
    integer=np.rint(coeff).astype(int)
    if len(coeff):
        keys=[tuple(q) for q in integer]
        counts=pd.Series(keys).value_counts()
        dom=tuple(map(int,counts.index[0])); dom_frac=float(counts.iloc[0]/len(keys))
        resultants=np.abs(np.mean(np.exp(2j*np.pi*coeff),axis=0))
    else:
        dom=(0,0);dom_frac=float('nan');resultants=np.array([np.nan,np.nan])
    return coeff,frac,{'dominant_integer_cell_shift_u':dom[0],'dominant_integer_cell_shift_v':dom[1],
        'dominant_integer_cell_shift_fraction':dom_frac,'fractional_offset_circular_concentration_u':float(resultants[0]),
        'fractional_offset_circular_concentration_v':float(resultants[1]),'median_fractional_u':float(np.median(frac[:,0])) if len(frac) else np.nan,
        'median_fractional_v':float(np.median(frac[:,1])) if len(frac) else np.nan}

def plot_group_distributions(out,vectors,labels):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    colors={'high':'#bc4b51','control':'#376996'}
    for group in ['high','control']:
        g=vectors[vectors.group==group]
        if g.empty: continue
        c=g[['lattice_u','lattice_v']].to_numpy(); f=g[['fractional_u','fractional_v']].to_numpy()
        axs[0,0].scatter(c[:,0],c[:,1],s=7,alpha=.35,color=colors[group],label=f'{group} (n={len(g)})')
        axs[0,1].scatter(f[:,0],f[:,1],s=7,alpha=.35,color=colors[group],label=group)
        axs[1,0].hist(f[:,0],bins=np.linspace(-.5,.5,41),histtype='step',density=True,color=colors[group],label=group)
        axs[1,1].hist(f[:,1],bins=np.linspace(-.5,.5,41),histtype='step',density=True,color=colors[group],label=group)
    axs[0,0].set(xlabel='Lattice coordinate u (cells)',ylabel='Lattice coordinate v (cells)',title='Raw residual vector in pre-image lattice basis')
    axs[0,1].set(xlim=(-.5,.5),ylim=(-.5,.5),xlabel='Wrapped u (fraction of cell)',ylabel='Wrapped v (fraction of cell)',title='Fractional lattice phase')
    axs[1,0].set(xlabel='Wrapped u (fraction of cell)',ylabel='Density',title='Fractional u distribution')
    axs[1,1].set(xlabel='Wrapped v (fraction of cell)',ylabel='Density',title='Fractional v distribution')
    for ax in axs.ravel(): ax.grid(alpha=.2); ax.legend(loc='best')
    fig.savefig(out/'ambiguous_residual_lattice_distribution.png',dpi=170);plt.close(fig)

def plot_spatial_map(out,points):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins=20; fig,axs=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
    for ax,grp,title in zip(axs,['high','control'],['High ambiguity (>25%)','Low-ambiguity controls']):
        g=points[points.group==grp]; den=np.zeros((bins,bins));num=np.zeros_like(den)
        ix=np.clip((g.x_norm*bins).astype(int),0,bins-1);iy=np.clip((g.y_norm*bins).astype(int),0,bins-1)
        np.add.at(den,(iy,ix),1);np.add.at(num,(iy,ix),g.ambiguous.astype(int))
        frac=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
        im=ax.imshow(frac,origin='upper',extent=(0,1,1,0),cmap='magma',vmin=0,vmax=max(.1,float(np.nanquantile(frac,.95)) if np.isfinite(frac).any() else .1))
        ax.set(title=title,xlabel='Normalized field x',ylabel='Normalized field y');fig.colorbar(im,ax=ax,label='Ambiguous fraction')
    fig.savefig(out/'ambiguous_spatial_distribution.png',dpi=170);plt.close(fig)

def plot_feature_correlations(out,frame,corr,features):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    nrows=int(np.ceil(len(features)/4));fig,axs=plt.subplots(nrows,4,figsize=(16,3.3*nrows),constrained_layout=True)
    for ax,name in zip(axs.ravel(),features):
        r=corr[corr.feature==name].iloc[0]
        g=frame[['cell_mismatch_fraction',name]].dropna()
        ax.scatter(g[name],g.cell_mismatch_fraction,s=9,alpha=.45,color='#426b8a')
        ax.set_title(f"{name}\nρ={r.spearman_rank_correlation:.2f}, n={len(g)}",fontsize=8)
        ax.set_xlabel(name,fontsize=8);ax.set_ylabel('Ambiguous fraction',fontsize=8);ax.grid(alpha=.2)
    for ax in axs.ravel()[len(features):]: ax.set_axis_off()
    fig.savefig(out/'ambiguity_feature_scatterplots.png',dpi=160);plt.close(fig)

def draw_fov_visual(outdir,key,display,matrix,pre,post,vec_record):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p,pred,t,vectors,distance,inlier=vec_record
    h,w=pre.shape; aa=image01_for_registration(pre);bb=cv2.warpAffine(image01_for_registration(post),matrix.astype(np.float32),(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    lo,hi=np.quantile(np.r_[aa.ravel()[::100],bb.ravel()[::100]],[.01,.99]);rgb=np.zeros((h,w,3),np.float32)
    rgb[:,:,0]=np.clip((aa-lo)/(hi-lo),0,1);rgb[:,:,1]=np.clip((bb-lo)/(hi-lo),0,1)
    amb=~inlier; inv=np.linalg.inv(matrix[:,:2]); back=vectors[amb]@inv.T
    fig,axs=plt.subplots(1,2,figsize=(13,6),constrained_layout=True)
    axs[0].imshow(rgb);axs[0].set_title('Aligned image overlay (red=pre, green=post)')
    mark_idx=np.flatnonzero(amb)
    if len(mark_idx)>1800:
        rng=np.random.default_rng(sum(ord(c) for c in key));mark_idx=np.sort(rng.choice(mark_idx,1800,replace=False))
    axs[0].scatter(p[mark_idx,0],p[mark_idx,1],s=6,c='#ffdd00',alpha=.65,label=f'> half-pitch ({int(amb.sum())}; marker sample {len(mark_idx)})');axs[0].legend(loc='lower right')
    axs[1].imshow(rgb);axs[1].set_title('Ambiguous locations and residual vectors (x20)')
    if amb.any():
        stride=max(1,int(amb.sum()/900)); q=np.flatnonzero(amb)[::stride]
        axs[1].quiver(p[q,0],p[q,1],back[np.arange(len(back))[::stride],0]*20,back[np.arange(len(back))[::stride],1]*20,
            color='#00ffff',angles='xy',scale_units='xy',scale=1,width=.002)
        axs[1].scatter(p[mark_idx,0],p[mark_idx,1],s=3,c='#ffdd00',alpha=.3)
    for ax in axs: ax.set(xlim=(0,w),ylim=(h,0),xlabel='x (pixel)',ylabel='y (pixel)')
    fig.suptitle(f'{display} | {key}');fig.savefig(outdir/f'{key}_overlay_ambiguous.png',dpi=145);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--registration-output',default='data/results/v16_real_spatial_subpixel_20260928');ap.add_argument('--output',required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    out=Path(a.output);vis=out/'visualizations';vis.mkdir(parents=True,exist_ok=True)
    tab=pd.read_csv(ROOT/'data/results/v17_independent_center_residual_20260928/real_independent_center_residuals_by_fov.csv')
    tab=tab[(tab.stage=='current_spatial_subpixel')&(tab.confidence_quantile==.5)].copy()
    tab=tab.sort_values('fov_key').reset_index(drop=True)
    highs=tab[tab.cell_mismatch_fraction>.25].copy()
    q25=tab.cell_mismatch_fraction.quantile(.25);low=tab[tab.cell_mismatch_fraction<=q25].copy()
    controls=low.sample(n=47,random_state=SEED).copy()
    focus=['260825_p50_dna_1_7','260829_p50_sam_3_4','260829_p50_sam_7_8','260826_p50_sam_11_7']
    selected=set(highs.fov_key)|set(controls.fov_key)|set(focus)
    local=pd.read_csv(ROOT/'data/results/v17_independent_center_residual_20260928/local_phase_vs_center_residuals.csv')
    if 'fov_key' in local: tab=tab.merge(local[['fov_key','local_phase_median_px']],on='fov_key',how='left')
    else: tab['local_phase_median_px']=np.nan
    def task(row):
        cv2.setNumThreads(1); key=row.fov_key; dataset=row.dataset
        m=re.match(r'^(.*)_(\d+)_(\d+)$',key); sample,pos=int(m.group(2)),int(m.group(3))
        ppath=image_path(a.data_root,dataset,sample,pos,0);qpath=image_path(a.data_root,dataset,sample,pos,1)
        pre=load_image_unicode(str(ppath));post=load_image_unicode(str(qpath))
        if pre is None or post is None: raise FileNotFoundError(f'{key}: {ppath} or {qpath}')
        fp=image_features(pre);fq=image_features(post)
        lp=independently_seeded_lattice(pre.astype(np.float32)/65535.);lq=independently_seeded_lattice(post.astype(np.float32)/65535.)
        basis=hex_basis(lp.pitch_px,lp.angle_rad); phase=wrapped(np.linalg.solve(basis,lq.origin-lp.origin)); phase_px=float(np.linalg.norm(basis@phase))
        angle=float((lq.angle_rad-lp.angle_rad+np.pi/6)%(np.pi/3)-np.pi/6)
        feat={'fov_key':key,'dataset':dataset,'sample':sample,'position':pos,'saturation_pre':fp['saturated_pixels'],'saturation_post':fq['saturated_pixels'],
          'saturation_total':fp['saturated_pixels']+fq['saturated_pixels'],'saturation_abs_difference':abs(fp['saturated_pixels']-fq['saturated_pixels']),
          'sharpness_pre':fp['sharpness_laplacian_variance'],'sharpness_post':fq['sharpness_laplacian_variance'],'sharpness_abs_difference':abs(fp['sharpness_laplacian_variance']-fq['sharpness_laplacian_variance']),
          'mean_pre':fp['mean'],'mean_post':fq['mean'],'mean_abs_difference':abs(fp['mean']-fq['mean']),'mean_signed_difference':fq['mean']-fp['mean'],
          'contrast_pre':fp['contrast_std'],'contrast_post':fq['contrast_std'],'contrast_abs_difference':abs(fp['contrast_std']-fq['contrast_std']),'contrast_signed_difference':fq['contrast_std']-fp['contrast_std'],
          'fft_pitch_pre':lp.pitch_px,'fft_pitch_post':lq.pitch_px,'fft_pitch_abs_difference':abs(lp.pitch_px-lq.pitch_px),
          'fft_angle_abs_difference_deg':abs(np.degrees(angle)),'fft_phase_difference_px':phase_px,'fft_phase_u':phase[0],'fft_phase_v':phase[1]}
        residual=None;detstats={};mat=None
        if key in selected:
            pp,ps,pi=detect_centers(pre);qq,qs,qi=detect_centers(post)
            mat=read_matrix(a.registration_output,key)
            rec,residual=residual_record(key,'current_spatial_subpixel',mat,pp,qq,ps,qs,.5,pre.shape,True)
            ambiguous=~residual[5]; coeff,frac,cs=cell_summary(residual[3][ambiguous],basis)
            detstats={'n_detected_pre':len(pp),'n_detected_post':len(qq),'pre_peak_prominence_median':float(np.median(ps)),
                'post_peak_prominence_median':float(np.median(qs)),'detector_match_count':int(rec['n_compared']),
                'detector_ambiguous_fraction':float(ambiguous.mean()) if len(ambiguous) else np.nan,**cs}
        return feat,detstats,residual,pre,post,ppath,qpath,mat
    feats=[];dets=[];vectors=[];spatial=[];visuals=[]
    grphigh=set(highs.fov_key);grpcontrol=set(controls.fov_key)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        fs={pool.submit(task,row):row.fov_key for _,row in tab.iterrows()}
        for i,f in enumerate(as_completed(fs),1):
            key=fs[f];feat,det,res,pre,post,ppath,qpath,mat=f.result();feat['cell_mismatch_fraction']=float(tab.loc[tab.fov_key==key,'cell_mismatch_fraction'].iloc[0]);feat['same_cell_median_px']=float(tab.loc[tab.fov_key==key,'same_cell_median_px'].iloc[0]);feat['local_phase_median_px']=float(tab.loc[tab.fov_key==key,'local_phase_median_px'].iloc[0]) if pd.notna(tab.loc[tab.fov_key==key,'local_phase_median_px'].iloc[0]) else np.nan
            feats.append(feat)
            if key in selected:
                group='high' if key in grphigh else ('control' if key in grpcontrol else 'focus');det.update({'fov_key':key,'group':group,'cell_mismatch_fraction':feat['cell_mismatch_fraction']});dets.append(det)
                p,pred,t,vec,dist,inlier=res;amb=~inlier
                # Basis is reconstructed from the independent pre-image Fourier estimates returned above.
                lattice=independently_seeded_lattice(pre.astype(np.float32)/65535.);basis=hex_basis(lattice.pitch_px,lattice.angle_rad)
                coeff,frac,_=cell_summary(vec[amb],basis)
                for j in np.flatnonzero(amb):
                    c=np.linalg.solve(basis,vec[j]);ff=wrapped(c)
                    vectors.append({'fov_key':key,'group':group,'x_px':float(p[j,0]),'y_px':float(p[j,1]),'residual_x_px':float(vec[j,0]),'residual_y_px':float(vec[j,1]),'residual_magnitude_px':float(dist[j]),'lattice_u':float(c[0]),'lattice_v':float(c[1]),'fractional_u':float(ff[0]),'fractional_v':float(ff[1])})
                for j in range(len(p)):
                    spatial.append({'fov_key':key,'group':group,'x_norm':float(p[j,0]/pre.shape[1]),'y_norm':float(p[j,1]/pre.shape[0]),'ambiguous':bool(amb[j])})
                visuals.append((key,group,pre,post,ppath,mat,res))
            if i%20==0 or i==len(tab): print(f'feature FOVs {i}/{len(tab)}',flush=True)
    fdf=pd.DataFrame(feats).merge(tab[['fov_key','cell_mismatch_fraction','same_cell_median_px','same_cell_p95_px','n_same_cell_matches','n_compared','fft_pre_pitch_px','fft_post_pitch_px','fft_pre_angle_deg','fft_post_angle_deg','local_phase_median_px']],on='fov_key',suffixes=('','_v17'))
    fdf.to_csv(out/'fov_features_401.csv',index=False)
    pd.DataFrame(dets).to_csv(out/'selected_fov_detection_features.csv',index=False)
    vdf=pd.DataFrame(vectors);vdf.to_csv(out/'ambiguous_residual_vectors.csv.gz',index=False,compression='gzip')
    fdet=pd.DataFrame(dets)
    vsummary=vdf.groupby(['fov_key','group']).agg(n_ambiguous=('residual_magnitude_px','size'),median_residual_px=('residual_magnitude_px','median'),p95_residual_px=('residual_magnitude_px',lambda z:z.quantile(.95)),median_lattice_u=('lattice_u','median'),median_lattice_v=('lattice_v','median'),fractional_u_sd=('fractional_u','std'),fractional_v_sd=('fractional_v','std')).reset_index()
    vsummary=vsummary.merge(fdet[['fov_key','cell_mismatch_fraction','dominant_integer_cell_shift_fraction','fractional_offset_circular_concentration_u','fractional_offset_circular_concentration_v']],on='fov_key',how='left')
    vsummary.to_csv(out/'per_fov_ambiguous_vector_summary.csv',index=False)
    spatial_fits=[]
    for (key,group_name),part in vdf.groupby(['fov_key','group']):
        xx=2*part.x_px.to_numpy()/2047-1;yy=2*part.y_px.to_numpy()/2043-1;design=np.c_[np.ones(len(part)),xx,yy]
        resid=part[['residual_x_px','residual_y_px']].to_numpy();coef=np.linalg.lstsq(design,resid,rcond=None)[0];fitted=design@coef
        raw_rms=float(np.sqrt(np.mean(np.sum(resid*resid,axis=1))));fit_rms=float(np.sqrt(np.mean(np.sum(fitted*fitted,axis=1))))
        remaining=float(np.sqrt(np.mean(np.sum((resid-fitted)**2,axis=1))))
        spatial_fits.append({'fov_key':key,'group':group_name,'n_ambiguous':len(part),'ambiguous_vector_rms_px':raw_rms,'affine_component_rms_px':fit_rms,'affine_remaining_rms_px':remaining,'affine_explained_fraction':float(np.sum(fitted*fitted)/(np.sum(resid*resid)+1e-12))})
    pd.DataFrame(spatial_fits).to_csv(out/'per_fov_ambiguous_spatial_fit.csv',index=False)
    sdf=pd.DataFrame(spatial)
    sdf['x_bin']=np.minimum(19,(sdf.x_norm*20).astype(int));sdf['y_bin']=np.minimum(19,(sdf.y_norm*20).astype(int))
    bins=sdf.groupby(['group','x_bin','y_bin']).agg(n_correspondences=('ambiguous','size'),n_ambiguous=('ambiguous','sum')).reset_index();bins['ambiguous_fraction']=bins.n_ambiguous/bins.n_correspondences
    bins.to_csv(out/'ambiguous_spatial_bin_rates.csv',index=False)
    fdf['group']=np.where(fdf.fov_key.isin(grphigh),'high',np.where(fdf.fov_key.isin(grpcontrol),'control','other'))
    fdf.to_csv(out/'fov_features_401.csv',index=False)
    continuous=['saturation_pre','saturation_post','saturation_total','saturation_abs_difference','sharpness_pre','sharpness_post','sharpness_abs_difference','mean_abs_difference','mean_signed_difference','contrast_abs_difference','contrast_signed_difference','fft_pitch_abs_difference','fft_angle_abs_difference_deg','fft_phase_difference_px','local_phase_median_px','same_cell_median_px','n_same_cell_matches']
    cr=[]
    for name in continuous:
        g=fdf[['cell_mismatch_fraction',name]].dropna();r,pv=spearmanr(g.cell_mismatch_fraction,g[name]) if len(g)>2 else (np.nan,np.nan)
        cr.append({'feature':name,'n':len(g),'spearman_rank_correlation':float(r),'p_value_descriptive':float(pv)})
    cdf=pd.DataFrame(cr);cdf.to_csv(out/'ambiguity_feature_rank_correlations.csv',index=False)
    plot_feature_correlations(out,fdf,cdf,continuous)
    group=fdf[fdf.group!='other'].groupby(['group','dataset']).agg(n=('fov_key','size'),median_ambiguity=('cell_mismatch_fraction','median'),mean_ambiguity=('cell_mismatch_fraction','mean')).reset_index();group.to_csv(out/'ambiguity_by_group_dataset.csv',index=False)
    fdf[fdf.group!='other'].groupby(['group','dataset','sample']).agg(n=('fov_key','size'),median_ambiguity=('cell_mismatch_fraction','median'),mean_ambiguity=('cell_mismatch_fraction','mean')).reset_index().to_csv(out/'ambiguity_by_group_sample.csv',index=False)
    fdf[fdf.group!='other'].groupby(['group','dataset','position']).agg(n=('fov_key','size'),median_ambiguity=('cell_mismatch_fraction','median'),mean_ambiguity=('cell_mismatch_fraction','mean')).reset_index().to_csv(out/'ambiguity_by_group_position.csv',index=False)
    fdf.assign(high_ambiguity=fdf.cell_mismatch_fraction>.25).groupby('dataset').agg(n=('fov_key','size'),high_count=('high_ambiguity','sum'),high_fraction=('high_ambiguity','mean'),median_ambiguity=('cell_mismatch_fraction','median')).reset_index().to_csv(out/'all_fov_dataset_summary.csv',index=False)
    gtab=fdf[fdf.group!='other'].groupby('group').agg(n=('fov_key','size'),median_ambiguity=('cell_mismatch_fraction','median'),median_phase=('local_phase_median_px','median'),median_pitch_diff=('fft_pitch_abs_difference','median')).reset_index();gtab.to_csv(out/'high_vs_control_summary.csv',index=False)
    vdf['group']=vdf.group.astype(str);plot_group_distributions(out,vdf,['high','control']);plot_spatial_map(out,sdf)
    fdet['mean_fractional_concentration']=fdet[['fractional_offset_circular_concentration_u','fractional_offset_circular_concentration_v']].mean(axis=1)
    signature=[]
    for name,part in fdet.groupby('group'):
        signature.append({'group':name,'n_fovs':len(part),'median_ambiguous_fraction':part.detector_ambiguous_fraction.median(),'median_dominant_integer_shift_fraction':part.dominant_integer_cell_shift_fraction.median(),'n_fov_dominant_integer_shift_fraction_ge_0_5':int((part.dominant_integer_cell_shift_fraction>=.5).sum()),'median_circular_concentration_u':part.fractional_offset_circular_concentration_u.median(),'median_circular_concentration_v':part.fractional_offset_circular_concentration_v.median(),'n_fov_any_axis_circular_concentration_ge_0_5':int(((part.fractional_offset_circular_concentration_u>=.5)|(part.fractional_offset_circular_concentration_v>=.5)).sum()),'median_pre_peak_prominence':part.pre_peak_prominence_median.median(),'median_post_peak_prominence':part.post_peak_prominence_median.median()})
    pd.DataFrame(signature).to_csv(out/'lattice_signature_group_summary.csv',index=False)
    # Requested top ten among high-rate fields; also include separately named focus cases if not already present.
    top=highs.sort_values('cell_mismatch_fraction',ascending=False).head(10).fov_key.tolist()
    (out/'top10_and_focus_fov_keys.txt').write_text('\n'.join(['TOP10']+top+['FOCUS']+focus)+'\n',encoding='utf-8')
    vlookup={z[0]:z for z in visuals}
    for key in dict.fromkeys(top+focus):
        if key not in vlookup: continue
        _,grp,pre,post,ppath,mat,res=vlookup[key]
        row=tab.loc[tab.fov_key==key].iloc[0]
        display=f"{row.dataset} sample{key.rsplit('_',2)[1]} pos{key.rsplit('_',2)[2]} | ambiguity {row.cell_mismatch_fraction:.1%}"
        draw_fov_visual(vis,key,display,mat,pre,post,res)
    selected_summary=tab[tab.fov_key.isin(selected)].copy();selected_summary['group']=np.where(selected_summary.fov_key.isin(grphigh),'high',np.where(selected_summary.fov_key.isin(grpcontrol),'control','focus'));selected_summary.to_csv(out/'selected_fov_v17_rows.csv',index=False)
    metadata={'version':21,'n_real_qc_fields':int(len(tab)),'n_high_gt_25pct':int(len(highs)),'high_threshold':.25,'low_quartile_cutoff':float(q25),'control_n':int(len(controls)),'control_seed':SEED,'n_selected_centers_processed':len(selected),'n_ambiguous_vectors':int(len(vdf)),'lattice_pitch_px':PITCH,'half_pitch_threshold_px':HALF,'v17_grid_confidence_numeric_available':False,'grid_confidence_note':'v17 stores FFT pitch/angle and center prominence thresholds, not a per-image FFT grid-confidence scalar.','diagnostic_only':True}
    (out/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
