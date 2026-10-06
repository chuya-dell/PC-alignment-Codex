"""v56: quantify the spatial structure in the existing v24 residual vectors.

This is a diagnostic-only consumer of v24 output.  It neither estimates a new
registration transform nor changes any production default.
"""
from __future__ import annotations
import argparse, hashlib, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

PITCH = 7.286
W, H = 2048, 2044

def half_maps(g, nbin=6, min_n=15):
    x, y = g.x_px.to_numpy(), g.y_px.to_numpy()
    # Deterministic checkerboard split on image bins, independent of residual.
    bx=np.minimum((x/W*nbin).astype(int),nbin-1); by=np.minimum((y/H*nbin).astype(int),nbin-1)
    split=(bx+by)%2
    out=[]
    for iy in range(nbin):
      for ix in range(nbin):
        m=(bx==ix)&(by==iy)
        for s in (0,1):
          # alternate lattice-like subsets within every spatial block
          q=m & (((np.floor(x/PITCH)+np.floor(y/(PITCH*np.sqrt(3)/2))).astype(int)&1)==s)
          if q.sum()>=min_n:
            out.append(dict(bin_x=ix,bin_y=iy,split=s,n=int(q.sum()),mean_dx=float(g.residual_x_px.to_numpy()[q].mean()),mean_dy=float(g.residual_y_px.to_numpy()[q].mean()),mean_magnitude=float(g.residual_magnitude_px.to_numpy()[q].mean())))
    return pd.DataFrame(out)

def analyze_field(key,g,rng):
    xy=g[['x_px','y_px']].to_numpy(float); v=g[['residual_x_px','residual_y_px']].to_numpy(float)
    # retain v24's raw vectors: do not remove spatially structured signal.
    tree=cKDTree(xy); dd, ii=tree.query(xy,k=2); j=ii[:,1]
    dot=(v*v[j]).sum(1); denom=np.linalg.norm(v,axis=1)*np.linalg.norm(v[j],axis=1)
    cosine=np.divide(dot,denom,out=np.full(len(dot),np.nan),where=denom>1e-12)
    # permutation comparison preserves vectors and positions independently.
    null=[]
    for _ in range(80):
      p=rng.permutation(len(v)); null.append(float(np.nanmean((v*v[p]).sum(1)/(np.linalg.norm(v,axis=1)*np.linalg.norm(v[p],axis=1)+1e-12))))
    hm=half_maps(g); pivot=hm.pivot_table(index=['bin_x','bin_y'],columns='split',values=['mean_dx','mean_dy'])
    common=pivot.dropna()
    if len(common)>=6:
      a=common[[('mean_dx',0),('mean_dy',0)]].to_numpy(); b=common[[('mean_dx',1),('mean_dy',1)]].to_numpy()
      split_cos=float(np.mean((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)+1e-12)))
      split_corr=float(np.corrcoef(a.ravel(),b.ravel())[0,1])
    else: split_cos=split_corr=np.nan
    return dict(fov_key=key,n_vectors=len(g),median_magnitude_px=float(g.residual_magnitude_px.median()),
      nearest_neighbor_distance_pitch=float(np.median(dd[:,1])/PITCH),nearest_neighbor_cosine=float(np.nanmean(cosine)),
      nearest_neighbor_dot_px2=float(np.mean(dot)),neighbor_cosine_null_median=float(np.median(null)),
      neighbor_cosine_null_p95=float(np.quantile(null,.95)),neighbor_cosine_excess=float(np.nanmean(cosine)-np.median(null)),
      neighbor_cosine_above_null95=bool(np.nanmean(cosine)>np.quantile(null,.95)),
      split_half_n_blocks=len(common),split_half_vector_cosine=split_cos,split_half_component_correlation=split_corr),hm

def plot(out, summary, maps):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    ax[0].hist(summary.nearest_neighbor_cosine,bins=35,color='#2878b5'); ax[0].axvline(0,color='k',lw=1);ax[0].set(xlabel='nearest-neighbour residual-vector cosine',ylabel='FOV count')
    ax[1].scatter(summary.nearest_neighbor_cosine,summary.split_half_vector_cosine,s=12,alpha=.65);ax[1].axhline(0,color='k',lw=1);ax[1].axvline(0,color='k',lw=1);ax[1].set(xlabel='nearest-neighbour cosine',ylabel='split-half block-vector cosine')
    fig.tight_layout();fig.savefig(out/'figures'/'coherence_distributions.png',dpi=180);plt.close(fig)
    # maps from the three strongest split-half coherent fields; explicitly values, no image input.
    pick=summary.sort_values('split_half_vector_cosine',ascending=False).head(3).fov_key
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key in zip(axs,pick):
      d=maps[key].groupby(['bin_x','bin_y'])[['mean_dx','mean_dy']].mean().reset_index();x=(d.bin_x+.5)*W/6;y=(d.bin_y+.5)*H/6
      ax.quiver(x,y,d.mean_dx,d.mean_dy,angles='xy',scale_units='xy',scale=0.008,width=.006);ax.set(xlim=(0,W),ylim=(H,0),aspect='equal',title=key)
    fig.suptitle('6×6 block-mean residual vectors (scale shared)',y=1.02);fig.tight_layout();fig.savefig(out/'figures'/'example_block_vector_maps.png',dpi=180,bbox_inches='tight');plt.close(fig)

def main():
 p=argparse.ArgumentParser();p.add_argument('--vectors',type=Path,required=True);p.add_argument('--v24-summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output; (out/'tables').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
 sha=hashlib.sha256(a.vectors.read_bytes()).hexdigest()
 rows=[]; maps={}; rng=np.random.default_rng(20261006)
 for key,g in pd.read_csv(a.vectors,compression='gzip').groupby('fov_key',sort=True):
   rec,hm=analyze_field(key,g,rng);rows.append(rec);hm.insert(0,'fov_key',key);maps[key]=hm
 summary=pd.DataFrame(rows); summary.to_csv(out/'tables'/'fov_spatial_structure.csv',index=False)
 pd.concat(maps.values(),ignore_index=True).to_csv(out/'tables'/'block_mean_vectors_6x6.csv',index=False)
 v24=pd.read_csv(a.v24_summary); joined=summary.merge(v24[['fov_key','structured_rms_px','correlation_distance_pitch','range_censored']],on='fov_key',how='left')
 joined.to_csv(out/'tables'/'fov_spatial_structure_with_v24.csv',index=False)
 med=float(summary.median_magnitude_px.median()); v17=0.6015415177359438
 dist=pd.DataFrame([dict(metric='v48_vector_magnitude_median_across_401_fovs_px',value=med),dict(metric='v17_reported_same_cell_median_across_401_fovs_px',value=v17),dict(metric='absolute_difference_px',value=abs(med-v17)),dict(metric='fraction_neighbor_cosine_above_permutation_null95',value=float(summary.neighbor_cosine_above_null95.mean())),dict(metric='median_neighbor_cosine',value=float(summary.nearest_neighbor_cosine.median())),dict(metric='median_split_half_vector_cosine',value=float(summary.split_half_vector_cosine.median())),dict(metric='median_v24_structured_rms_px',value=float(joined.structured_rms_px.median())),dict(metric='v24_range_censored_fovs',value=int(joined.range_censored.sum())),dict(metric='n_fovs',value=len(summary)),dict(metric='v24_vectors_sha256',value=sha)])
 dist.to_csv(out/'tables'/'summary_metrics.csv',index=False)
 plot(out,summary,maps)
if __name__=='__main__': main()
