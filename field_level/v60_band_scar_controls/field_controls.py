"""Liquid-free controls, native-frame photometry and provisional measured shifts."""
from pathlib import Path
import sys,json,os
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).parent))
import field_control_common as M
import numpy as np,pandas as pd,cv2
from scipy import ndimage
sys.path.insert(0,str(ROOT/'data/results/v58_band_scar_causal/checkout'))
from shared.lattice_indexing import lattice_from_fft,grid_coordinates
from shared import registration as reg
from shared.v2_registration_precision.refinement import register_refined
OUT=M.OUT

def measure(a,b,shift=None):
    if shift is None:
        # These controls have no large stage translation: local translation initialized at zero.
        aa=(a-a.mean())/(a.std()+1e-9);bb=(b-b.mean())/(b.std()+1e-9)
        mat=np.float32([[1,0,0],[0,1,0]])
        try:
            rho,mat=cv2.findTransformECC(aa,bb,mat,cv2.MOTION_TRANSLATION,(3,120,1e-7),None,3)
            info={'rho':rho,'registration':'translation from zero'}
        except cv2.error as e:
            # Severe defocus can have no detectable lattice; keep zero as explicit sensitivity.
            info={'rho':None,'registration':'zero fallback; provisional','error':str(e)}
    else:
        mat=np.float32([[1,0,shift[0]],[0,1,shift[1]]]);info={'registration':'v58 measured shift; provisional'}
    lat=lattice_from_fft(a,7.286);ids,xy=grid_coordinates(lat,a.shape[1],a.shape[0],margin=30)
    post=xy@mat[:,:2].T+mat[:,2]
    # Use the exact stored-production photometry, not the approximate v58 contrast helper.
    ca=reg.sample_contrast(a,xy);cb=reg.sample_contrast(b,post)
    valid=ca.valid_sampling.to_numpy()&cb.valid_sampling.to_numpy()
    d=ca.contrast.to_numpy()-cb.contrast.to_numpy();d[~valid]=np.nan
    return xy,d,mat,info

def classify(dm,key):
    import importlib.util
    p=ROOT/'data/results/v58_band_scar_causal/source_snapshots/field_round1_analysis.py'
    spec=importlib.util.spec_from_file_location('round1',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    stats,*_=mod.analyze_map(dm,np.random.default_rng(20261007),1000)
    return stats

def controls(stage):
    pairs=[]
    if stage=='repeat':
        p=M.folder('260904_p50_repeat');pairs=[('repeat_1_2',p/'1.tif',p/'2.tif',None)]
    elif stage=='focus':
        p=M.folder('260901_p50_Zstep');files=sorted([f for f in p.iterdir() if f.suffix=='.tif'],key=lambda f:float(f.stem))
        pairs=[(f'focus_{a.stem}_{b.stem}',a,b,None) for a,b in zip(files[:-1],files[1:])]
    else:
        t=pd.read_csv(ROOT/'data/results/v58_motor_calibration/marker_pairs_checkpoint.csv').iloc[:6]
        p=M.folder('261006-p50-ステッピングモーター_test')
        pairs=[(f'motor_{r.pair}',p/r.pre,p/r.post,(r.dx_px,r.dy_px)) for r in t.itertuples()]
    dest=OUT/stage;dest.mkdir(exist_ok=True);rows=[];inputs=[]
    for key,pa,pb,shift in pairs:
        cache=dest/f'{key}.npz';audit=dest/f'{key}.json'
        if not cache.exists():
            a,b=M.read(pa),M.read(pb);xy,d,mat,info=measure(a,b,shift)
            # bilinear native contrast as a sensitivity comparison on the same fixed transform.
            ca=M.contrast(a);cb=M.contrast(b);post=xy@mat[:,:2].T+mat[:,2]
            interp=M.sample(ca,xy,True)-M.sample(cb,post,True)
            np.savez_compressed(cache,xy=xy,delta=d,bilinear=interp,matrix=mat)
            audit.write_text(json.dumps(info,indent=2),encoding='utf8')
        with np.load(cache) as z:
            xy,d,mat=z['xy'],z['delta'],z['matrix'];bi=z['bilinear']
        # Liquid-free series are their own blank controls. Same mean+3 population SD rule.
        # Per-pair thresholds are diagnostic (not an independent false-positive validation).
        for method,v in [('round',d),('bilinear',bi)]:
            good=v[np.isfinite(v)];threshold=float(good.mean()+3*good.std())
            met,dm=M.band_metrics(xy,v,threshold,True);c=classify(dm,key)
            row=dict(key=key,method=method,threshold=threshold,mean=float(good.mean()),sd=float(good.std()),dx=float(mat[0,2]),dy=float(mat[1,2]),frac_dx=float(mat[0,2]-np.rint(mat[0,2])),frac_dy=float(mat[1,2]-np.rint(mat[1,2])),**met,**c)
            rows.append(row);np.save(dest/f'{key}_{method}_map.npy',dm)
        inputs.extend([dict(path=str(p),sha256=M.fingerprint(p)) for p in [pa,pb]])
        print(stage,key,rows[-2]['rate'],rows[-2]['分類'],flush=True)
    t=pd.DataFrame(rows);t.to_csv(dest/'metrics.csv',index=False);pd.DataFrame(inputs).drop_duplicates().to_csv(dest/'input_sha256.csv',index=False)
    n=len(pairs);fig,axs=M.plt.subplots(n,3,figsize=(12,2.5*n),squeeze=False)
    vmax=max(np.nanpercentile(np.load(dest/f'{k}_round_map.npy'),99) for k,*_ in pairs)
    for ax,(key,pa,pb,_) in zip(axs,pairs):
        r=t[(t.key==key)&(t.method=='round')].iloc[0]
        for a,m in zip(ax[:2],['round','bilinear']):
            a.imshow(np.load(dest/f'{key}_{m}_map.npy'),vmin=0,vmax=vmax,cmap='magma');a.set_title(key+' '+m);a.axis('off')
        aa=M.read(pa);bb=M.read(pb);ax[2].imshow(cv2.resize(aa-bb,(512,511)),cmap='coolwarm');ax[2].set_title(f"unaligned raw; rate={r.rate*100:.3f}%, S={r.strength:.3f}");ax[2].axis('off')
    fig.tight_layout();fig.savefig(dest/'atlas.png',dpi=110);M.plt.close(fig)

if __name__=='__main__':controls(sys.argv[1])
