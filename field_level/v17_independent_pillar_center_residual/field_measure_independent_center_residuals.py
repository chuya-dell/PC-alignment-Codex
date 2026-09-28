"""Measure independent pre/post pillar-center residuals and detector noise.

Diagnostic-only: no registration estimates or production defaults are changed.
Each image gets its own FFT orientation/phase/pitch and local top-hat peak centers.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT))
from shared.registration import load_image_unicode
from shared.lattice_indexing import (HexLattice, estimate_hex_orientation_fft,
    estimate_phase_origin_fft, hex_basis, grid_coordinates)

PITCH=7.286
HALF_PITCH=PITCH/2
QUANTILES=(.25,.50,.75)
TARGETS={
    '260829_p50_sam_3_4':'260829 SAM sample3 pos4',
    '260829_p50_sam_7_8':'260829 SAM sample7 pos8',
    '260826_p50_sam_11_7':'260826 SAM sample11 pos7',
    '260825_p50_dna_1_7':'260825 DNA sample1 pos7',
}

def independently_seeded_lattice(image):
    """Estimate this image's FFT angle, pitch and phase without the paired image."""
    image=np.asarray(image,dtype=np.float32)
    h,w=image.shape
    spectrum=np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image-image.mean()))))
    yy,xx=np.indices(image.shape)
    radius=np.hypot((xx-w//2)/w,(yy-h//2)/h)
    expected=2/(np.sqrt(3)*PITCH)
    maxima=cv2.dilate(spectrum.astype(np.float32),np.ones((9,9),np.uint8))
    candidate=(spectrum>=maxima-1e-6)&(radius>=expected*.82)&(radius<=expected*1.18)
    flat=np.flatnonzero(candidate)
    if len(flat)<6: raise ValueError('Too few FFT lattice peaks')
    strongest=flat[np.argsort(spectrum.ravel()[flat])[-12:]]
    pitch=float(2/(np.sqrt(3)*np.median(radius.ravel()[strongest])))
    angle=estimate_hex_orientation_fft(image,PITCH)
    basis=hex_basis(pitch,angle)
    return HexLattice(estimate_phase_origin_fft(image,basis),basis,angle,pitch)

def detect_centers(image):
    """FFT grid seeds + independent local top-hat maximum + 2-D parabola.

    A separate grid is estimated for every image.  At each seed, a 7x7 local
    window finds the strongest top-hat peak; separable three-point parabolas
    provide subpixel coordinates.  Peak prominence is retained for confidence
    filtering and sensitivity analysis.
    """
    raw=np.asarray(image)
    norm=raw.astype(np.float32)/65535.0
    h,w=norm.shape
    lattice=independently_seeded_lattice(norm)
    _,seeds=grid_coordinates(lattice,w,h,margin=12)
    top=cv2.morphologyEx(norm,cv2.MORPH_TOPHAT,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    smooth=cv2.GaussianBlur(top,(5,5),0)
    ix=np.rint(seeds[:,0]).astype(int);iy=np.rint(seeds[:,1]).astype(int)
    windows=np.lib.stride_tricks.sliding_window_view(smooth,(7,7))
    patches=windows[iy-3,ix-3]
    flat=patches.reshape(len(seeds),-1)
    arg=np.argmax(flat,axis=1)
    py=iy-3+arg//7;px=ix-3+arg%7
    peak=smooth[py,px]
    def parabola(minus,mid,plus):
        denom=minus-2*mid+plus
        valid=denom < -1e-12
        delta=np.zeros_like(mid,dtype=float)
        delta[valid]=.5*(minus[valid]-plus[valid])/denom[valid]
        return np.clip(delta,-.5,.5)
    dx=parabola(smooth[py,px-1],peak,smooth[py,px+1])
    dy=parabola(smooth[py-1,px],peak,smooth[py+1,px])
    points=np.c_[px+dx,py+dy]
    prominence=peak-np.median(flat,axis=1)
    return points,prominence,{'fft_pitch_px':lattice.pitch_px,'fft_angle_deg':float(np.degrees(lattice.angle_rad)),
                             'n_grid_seeds':len(seeds)}

def transform(points,matrix):
    return np.asarray(points)@np.asarray(matrix)[:,:2].T+np.asarray(matrix)[:,2]

def model_fit(xy,residual,degree):
    h,w=2044,2048
    x=(xy[:,0]-(w-1)/2)/(w/2); y=(xy[:,1]-(h-1)/2)/(h/2)
    if degree==1: design=np.c_[np.ones(len(x)),x,y]
    else: design=np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y]
    coeff=np.linalg.lstsq(design,residual,rcond=None)[0]
    fitted=design@coeff
    rms=lambda a:float(np.sqrt(np.mean(np.sum(a*a,axis=1))))
    raw=rms(residual); explained=rms(fitted); remain=rms(residual-fitted)
    return {'fit_component_rms_px':explained,'remaining_rms_px':remain,
            'explained_energy_fraction':float(explained**2/(raw**2+1e-15))}

def residual_record(fov,stage,matrix,pre_points,post_points,pre_scores,post_scores,q,shape,
                    retain_vectors=False):
    pcut=np.quantile(pre_scores,q);qcut=np.quantile(post_scores,q)
    p=pre_points[pre_scores>=pcut];t=post_points[post_scores>=qcut]
    predicted=transform(p,matrix)
    h,w=shape
    inside=(predicted[:,0]>=12)&(predicted[:,0]<w-12)&(predicted[:,1]>=12)&(predicted[:,1]<h-12)
    p=p[inside];predicted=predicted[inside]
    distance,index=cKDTree(t).query(predicted,k=1)
    # A nearest-neighbour query can assign one post center to several pre
    # centers. Keep only the closest prediction for each post center so the
    # reported residuals represent one-to-one pillar pairs.
    order=np.argsort(distance,kind='stable')
    _,first=np.unique(index[order],return_index=True)
    keep=np.zeros(len(index),dtype=bool)
    keep[order[first]]=True
    n_duplicate_post_matches=int((~keep).sum())
    p=p[keep];predicted=predicted[keep];distance=distance[keep];index=index[keep]
    vectors=t[index]-predicted
    inlier=distance<=HALF_PITCH
    rec={'fov_key':fov,'stage':stage,'confidence_quantile':q,'n_pre_centers':len(pre_points),
         'n_post_centers':len(post_points),'n_pre_selected':len(p),'n_post_selected':len(t),
         'n_compared':len(distance),'n_cell_mismatch_gt_half_pitch':int((~inlier).sum()),
         'cell_mismatch_fraction':float((~inlier).mean()) if len(inlier) else np.nan,
         'n_same_cell_matches':int(inlier.sum()),
         'same_cell_median_px':float(np.median(distance[inlier])) if inlier.any() else np.nan,
         'same_cell_p95_px':float(np.quantile(distance[inlier],.95)) if inlier.any() else np.nan,
         'same_cell_max_px':float(np.max(distance[inlier])) if inlier.any() else np.nan,
         'nearest_all_median_px':float(np.median(distance)) if len(distance) else np.nan,
         'nearest_all_p95_px':float(np.quantile(distance,.95)) if len(distance) else np.nan,
         'nearest_all_max_px':float(np.max(distance)) if len(distance) else np.nan,
         'n_duplicate_post_matches':n_duplicate_post_matches}
    if inlier.sum()>=100:
        rec.update({'affine_'+k:v for k,v in model_fit(p[inlier],vectors[inlier],1).items()})
        rec.update({'quadratic_'+k:v for k,v in model_fit(p[inlier],vectors[inlier],2).items()})
    if retain_vectors:
        return rec,(p,predicted,t[index],vectors,distance,inlier)
    return rec,None

def run_synthetic_group(item):
    path=item['path']; image=load_image_unicode(path)
    if image is None: raise FileNotFoundError(path)
    cv2.setNumThreads(1)
    pre,ps,pi=detect_centers(image)
    rows=[]; shape=image.shape
    for case in item['cases']:
        matrix=np.asarray(case['truth'],float)
        if np.max(np.abs(matrix-np.c_[np.eye(2),np.zeros(2)]))<1e-8:
            moving=image;post,qs,qi=pre,ps,pi
        else:
            moving=cv2.warpAffine(image,matrix.astype(np.float32),(shape[1],shape[0]),
                                  flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            post,qs,qi=detect_centers(moving)
        for q in QUANTILES:
            r,_=residual_record(case['case_id'],'known_truth',matrix,pre,post,ps,qs,q,shape)
            r.update({'source':item['source'],'truth_scale':float(np.sqrt(np.linalg.det(matrix[:,:2]))),
                      'pre_fft_pitch_px':pi['fft_pitch_px'],'post_fft_pitch_px':qi['fft_pitch_px']})
            rows.append(r)
    return rows

def semisynthetic_noise_floor(output,data_root,workers):
    inputs=json.loads((ROOT/'data/results/v11_registration_precision_20260926/baseline_inputs.json').read_text(encoding='utf-8'))
    groups={}
    for case in inputs:
        source=case['path']
        groups.setdefault(source,[]).append(case)
    tasks=[{'path':path,'source':str(Path(path).relative_to(Path(data_root))),'cases':cases}
           for path,cases in groups.items()]
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run_synthetic_group,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            rows.extend(f.result());print(f'semisynthetic source groups {i}/{len(tasks)}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(output/'semisynthetic_detection_floor_by_case.csv',index=False)
    return summarize_semisynthetic_floor(output,data_root)

def summarize_semisynthetic_floor(output,data_root):
    df=pd.read_csv(output/'semisynthetic_detection_floor_by_case.csv')
    inputs=json.loads((ROOT/'data/results/v11_registration_precision_20260926/baseline_inputs.json').read_text(encoding='utf-8'))
    source_by_case={r['case_id']:str(Path(r['path']).relative_to(Path(data_root))) for r in inputs}
    df['source']=df.fov_key.map(source_by_case)
    df.to_csv(output/'semisynthetic_detection_floor_by_case.csv',index=False)
    summaries=[]
    rng=np.random.default_rng(20260928)
    for q,part in df.groupby('confidence_quantile'):
        main=part[part['n_same_cell_matches']>0]
        source_stats=main.groupby('source').agg(median_of_case_medians=('same_cell_median_px','median'),
            median_of_case_p95=('same_cell_p95_px','median'),cell_mismatch_rate=('cell_mismatch_fraction','mean'))
        med=source_stats.median_of_case_medians.to_numpy();p95=source_stats.median_of_case_p95.to_numpy()
        boots=np.array([np.median(rng.choice(med,len(med),replace=True)) for _ in range(2000)])
        pboots=np.array([np.median(rng.choice(p95,len(p95),replace=True)) for _ in range(2000)])
        summaries.append({'confidence_quantile':q,'n_conditions':int(len(main)),
            'n_independent_source_images':int(len(source_stats)),
            'median_case_median_px':float(main.same_cell_median_px.median()),
            'median_case_p95_px':float(main.same_cell_p95_px.median()),
            'p95_case_p95_px':float(main.same_cell_p95_px.quantile(.95)),
            'median_case_max_px':float(main.same_cell_max_px.median()),
            'median_cell_mismatch_fraction':float(main.cell_mismatch_fraction.median()),
            'source_cluster_bootstrap_median_ci95_px':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
            'source_cluster_bootstrap_p95_ci95_px':[float(np.quantile(pboots,.025)),float(np.quantile(pboots,.975))]} )
    pd.DataFrame(summaries).to_csv(output/'semisynthetic_detection_floor_summary.csv',index=False)
    return pd.DataFrame(summaries)

def path_from_manifest(s,data_root):
    tail=s.replace('\\','/').split('/4.生データ D/')[-1]
    return str(Path(data_root)/tail)

def matrix_records(fov,output):
    d=json.loads((output/'diagnostics'/f'{fov}.json').read_text(encoding='utf-8'))
    out={'current_spatial_subpixel':np.asarray(d['matrix'],float),
         'v15_coarse_no_subpixel':np.asarray(d['baseline_matrix'],float)}
    old=ROOT/'data/results/v11_registration_precision_20260926/real_baseline/diagnostics'/f'{fov}.json'
    if old.is_file(): out['v11_coarse_baseline']=np.asarray(json.loads(old.read_text(encoding='utf-8'))['matrix'],float)
    return out

def overlay_plot(path,outpath,title,pre,post,matrix,record_vectors):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shared.registration import image01_for_registration
    p,pred,t,vectors,distance,inlier=record_vectors
    h,w=pre.shape
    aligned=cv2.warpAffine(image01_for_registration(post),matrix.astype(np.float32),(w,h),
                           flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    a=image01_for_registration(pre)
    lo,hi=np.quantile(np.r_[a.ravel()[::100],aligned.ravel()[::100]],[.01,.99])
    rgb=np.zeros((h,w,3),np.float32)
    rgb[:,:,0]=np.clip((a-lo)/(hi-lo),0,1);rgb[:,:,1]=np.clip((aligned-lo)/(hi-lo),0,1)
    fig,axs=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    axs[0].imshow(rgb);axs[0].set_title('Aligned overlay (red=pre, green=post)')
    bins=44; bx=np.clip((p[:,0]/w*bins).astype(int),0,bins-1);by=np.clip((p[:,1]/h*bins).astype(int),0,bins-1)
    count=np.zeros((bins,bins));sx=np.zeros_like(count);sy=np.zeros_like(count);sm=np.zeros_like(count)
    good=inlier
    np.add.at(count,(by[good],bx[good]),1);np.add.at(sx,(by[good],bx[good]),vectors[good,0]);np.add.at(sy,(by[good],bx[good]),vectors[good,1]);np.add.at(sm,(by[good],bx[good]),distance[good])
    mask=count>0; xx=(np.arange(bins)+.5)*w/bins; yy=(np.arange(bins)+.5)*h/bins; X,Y=np.meshgrid(xx,yy)
    mag=np.divide(sm,count,out=np.full_like(sm,np.nan),where=mask)
    im=axs[1].imshow(mag,extent=(0,w,h,0),cmap='magma',vmin=0,vmax=max(.6,float(np.nanquantile(mag,.95))))
    ux=np.divide(sx,count,out=np.zeros_like(sx),where=mask);uy=np.divide(sy,count,out=np.zeros_like(sy),where=mask)
    axs[1].quiver(X[mask],Y[mask],ux[mask]*25,uy[mask]*25,color='cyan',angles='xy',scale_units='xy',scale=1,width=.002)
    axs[1].set_title('Independent-center residuals (arrows x25; color=mean px)');fig.colorbar(im,ax=axs[1],label='Residual magnitude (px)')
    for ax in axs: ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_xlabel('x (px)');ax.set_ylabel('y (px)')
    fig.suptitle(title);fig.savefig(outpath,dpi=150);plt.close(fig)

def run_real(output,data_root,workers,registration_output):
    manifest=pd.read_csv(ROOT/'data/results/v10_phase2_bright_band_mask_20260926/masked_50862c8/source_manifest_attached.csv')
    status=pd.read_csv(registration_output/'summary_0/phase2_reanalysis_fov_summary.csv')
    accepted=status[status.status=='ok']
    accepted_keys=set(f"{r.dataset}_{int(r['sample'])}_{int(r.position)}" for _,r in accepted.iterrows())
    tasks=[]
    for _,r in manifest.iterrows():
        fov=f"{r.dataset}_{int(r['sample'])}_{int(r.position)}"
        if fov not in accepted_keys: continue
        tasks.append({'fov':fov,'pre':path_from_manifest(r.pre_path,data_root),'post':path_from_manifest(r.post_path,data_root)})
    def job(t):
        cv2.setNumThreads(1)
        pre=load_image_unicode(t['pre']);post=load_image_unicode(t['post'])
        if pre is None or post is None: raise FileNotFoundError(t['fov'])
        p,ps,pi=detect_centers(pre);q,qs,qi=detect_centers(post)
        ms=matrix_records(t['fov'],registration_output)
        rows=[];plotdata=None
        for stage,m in ms.items():
            for conf in QUANTILES:
                retain=(t['fov'] in TARGETS and stage=='current_spatial_subpixel' and conf==.5)
                rec,v=residual_record(t['fov'],stage,m,p,q,ps,qs,conf,pre.shape,retain)
                rec.update({'dataset':t['fov'].rsplit('_',2)[0],'fft_pre_pitch_px':pi['fft_pitch_px'],
                            'fft_post_pitch_px':qi['fft_pitch_px'],'fft_pre_angle_deg':pi['fft_angle_deg'],
                            'fft_post_angle_deg':qi['fft_angle_deg']})
                rows.append(rec)
                if retain: plotdata=(pre,post,m,v)
        return rows,plotdata,t['fov']
    rows=[];plots=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(job,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            rr,plot,fov=f.result();rows.extend(rr)
            if plot: plots.append((fov,*plot))
            if i%10==0 or i==len(futures):print(f'real FOVs {i}/{len(futures)}',flush=True)
    df=pd.DataFrame(rows);df.to_csv(output/'real_independent_center_residuals_by_fov.csv',index=False)
    for fov,pre,post,m,v in plots:
        overlay_plot(None,output/'visualizations'/f'{fov}_independent_residual_field.png',
                     TARGETS[fov],pre,post,m,v)
    return df

def plot_selected_targets(output,data_root,registration_output):
    manifest=pd.read_csv(ROOT/'data/results/v10_phase2_bright_band_mask_20260926/masked_50862c8/source_manifest_attached.csv')
    for fov,title in TARGETS.items():
        keyparts=fov.rsplit('_',2);dataset=keyparts[0];sample=int(keyparts[1]);position=int(keyparts[2])
        found=manifest[(manifest.dataset==dataset)&(manifest['sample']==sample)&(manifest.position==position)]
        if found.empty: print('target missing from manifest',fov);continue
        row=found.iloc[0]
        pre=load_image_unicode(path_from_manifest(row.pre_path,data_root));post=load_image_unicode(path_from_manifest(row.post_path,data_root))
        p,ps,_=detect_centers(pre);q,qs,_=detect_centers(post)
        m=matrix_records(fov,registration_output)['current_spatial_subpixel']
        _,v=residual_record(fov,'current_spatial_subpixel',m,p,q,ps,qs,.5,pre.shape,True)
        overlay_plot(None,output/'visualizations'/f'{fov}_independent_residual_field.png',title,pre,post,m,v)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--phase',choices=['synthetic','real','all'],default='all')
    ap.add_argument('--data-root',type=Path,default=Path(r'W:\4.生データD_remo'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--registration-output',type=Path,default=ROOT/'data/results/v16_real_spatial_subpixel_20260928')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--plots-only',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'visualizations').mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(a.output/'.mplconfig');Path(os.environ['MPLCONFIGDIR']).mkdir(exist_ok=True)
    if a.plots_only:
        plot_selected_targets(a.output,a.data_root,a.registration_output);return
    if a.phase in ('synthetic','all'):
        s=semisynthetic_noise_floor(a.output,a.data_root,a.workers);print('synthetic summary\n',s.to_string(index=False),flush=True)
    if a.phase in ('real','all'):
        df=run_real(a.output,a.data_root,a.workers,a.registration_output)
        print('real rows',len(df),'accepted FOVs',df.fov_key.nunique(),flush=True)

if __name__=='__main__':main()
