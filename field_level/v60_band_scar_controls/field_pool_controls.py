"""Series-pooled blank thresholds and unchanged round1 classifier; pair-specific sensitivity retained."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from field_controls import M,OUT,classify
import numpy as np,pandas as pd

def main(stage):
    dest=OUT/stage;old=pd.read_csv(dest/'metrics.csv');rows=[]
    files=sorted(dest.glob('*.npz'));maps=[]
    for method,col in [('round','delta'),('bilinear','bilinear')]:
        pool=[]
        for p in files:
            with np.load(p) as z:d=z[col];pool.append(d[np.isfinite(d)])
        v=np.concatenate(pool);th=float(v.mean()+3*v.std())
        for p in files:
            with np.load(p) as z:xy,d,mat=z['xy'],z[col],z['matrix']
            met,dm=M.band_metrics(xy,d,th,True);c=classify(dm,p.stem)
            rows.append(dict(key=p.stem,method=method,threshold=th,blank_pool_pairs=len(files),blank_pool_n=len(v),dx=mat[0,2],dy=mat[1,2],frac_dx=mat[0,2]-np.rint(mat[0,2]),frac_dy=mat[1,2]-np.rint(mat[1,2]),**met,**c))
            np.save(dest/f'{p.stem}_{method}_pooled_map.npy',dm)
    t=pd.DataFrame(rows);t.to_csv(dest/'metrics_series_pooled.csv',index=False)
    keys=t[t.method=='round'].key.tolist();fig,ax=M.plt.subplots(len(keys),2,figsize=(9,2.5*len(keys)),squeeze=False)
    vmax=max(np.nanpercentile(np.load(dest/f'{k}_round_pooled_map.npy'),99) for k in keys)
    for axes,key in zip(ax,keys):
        for a,m in zip(axes,['round','bilinear']):
            r=t[(t.key==key)&(t.method==m)].iloc[0];a.imshow(np.load(dest/f'{key}_{m}_pooled_map.npy'),vmin=0,vmax=vmax,cmap='magma');a.set_title(f'{key} {m}: {100*r.rate:.3f}% S={r.strength:.3f} {r["分類"]}',fontname='Yu Gothic',fontsize=8);a.axis('off')
    fig.tight_layout();fig.savefig(dest/'atlas_series_pooled.png',dpi=110);M.plt.close(fig)
    fig,ax=M.plt.subplots(1,3,figsize=(13,4))
    for m,g in t.groupby('method'):
        if stage=='motor':x=g.frac_dx;label='dx minus nearest integer (px)'
        else:x=g.key.str.extract(r'_(\d+)_')[0].astype(float) if stage=='focus' else np.arange(len(g));label='Pre focus dial / pair'
        for a,col in zip(ax,['rate','strength','density_sd']):a.plot(x,g[col],'o-',label=m);a.set(xlabel=label,ylabel=col)
    for a in ax:a.legend()
    fig.tight_layout();fig.savefig(dest/'summary_series_pooled.png',dpi=150);M.plt.close(fig)
    print(t[['key','method','rate','strength','分類']].to_string(index=False),flush=True)

if __name__=='__main__':main(sys.argv[1])
