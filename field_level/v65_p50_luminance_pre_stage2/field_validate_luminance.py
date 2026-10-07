"""Additional B: frozen candidate-mask diagnostics, never Stage 2 adoption."""
from pathlib import Path
import argparse, csv, hashlib, json, sys, platform, os
import numpy as np
import cv2
import skimage
from scipy.stats import trim_mean

ROOT = Path(__file__).resolve().parents[2]
import field_v20_snapshot as old
OUT = ROOT / 'data/results/v65_p50_luminance_pre_stage2'
RAW = Path('W:/GoogleDrive/chuya2816/5.生データD_chu/260922-p50-dna')
VIEWS = [('1',5),('1',6),('3',2),('3',3),('4',1),('4',2),('5',3),('5',5),('5',6)]
PARAM = ROOT / 'data/raw/v65_p50_v20_parameter_snapshot.json'
P = old.load_params(PARAM)
SIZES = [0.1262, 0.12694]
cv2.setNumThreads(1)

def write(name, rows):
    old.write_csv(OUT / name, rows)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(a,b,keep):
    if keep.sum() < 100:
        raise ValueError('Insufficient support')
    out = {}
    for name, fun in [('median',np.median),('trim10',lambda x: trim_mean(x,0.1))]:
        x,y = float(fun(a[keep])),float(fun(b[keep]))
        out.update({name+'_pre':x,name+'_post':y,name+'_difference':y-x,name+'_ratio':y/x})
    return out

def valid_support(shape,warp):
    h,w=shape
    # Linear interpolation needs all four source neighbours. Constant border is invalid.
    support=cv2.warpAffine(np.ones(shape,np.float32),warp,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    return support >= 1-1e-6

def cache_path(board,pos,size):
    return OUT / 'cache' / f'{board}_{pos}_{size}.npz'

def get_pair(board,pos,size):
    file=cache_path(board,pos,size)
    if not file.exists():
        old.save_pair_products=lambda *args: None
        row,pr=old.process_pair(RAW,board,pos,size,P,OUT)
        np.savez_compressed(file,pre=pr['pre'],post=pr['post_in_pre'],mask=pr['pre_union'],
                            warp=pr['warp'],marker=pr['common']['marker'],accepted=row['registration_accepted'])
        (file.with_suffix('.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
    return np.load(file)

def pairs():
    rows=[]; inventory=[]; hashes={}
    for board,pos in VIEWS+[('8',i) for i in range(1,9)]:
        for phase in [0,1]:
            file=RAW/f'50-{board}-{pos}-{phase}.tif'
            inventory.append(dict(path=str(file),bytes=file.stat().st_size,sha256=digest(file)))
            hashes[str(file)]=inventory[-1]['sha256']
        for size in SIZES:
            z=get_pair(board,pos,size)
            a,b,m=z['pre'],z['post'],z['mask']; v=valid_support(a.shape,z['warp'])
            for masked in [False,True]:
                keep=v & ~m if masked else v
                rows.append(dict(board=board,position=pos,pixel_size_um=size,masked=masked,
                                 valid_pixels=int(v.sum()),used_pixels=int(keep.sum()),
                                 candidate_mask_percent=float(m.mean()*100),accepted=bool(z['accepted']),
                                 **stats(a,b,keep)))
            write('luminance.csv',rows)
            print('pair',board,pos,size,'accepted',bool(z['accepted']),flush=True)
    write('input_inventory.csv',inventory)
    (OUT/'input_hashes_before.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    comparisons=[]
    for size in SIZES:
        for metric in ['median_difference','median_ratio','trim10_difference','trim10_ratio']:
            baseline=[r[metric] for r in rows if r['board']=='8' and not r['masked'] and r['pixel_size_um']==size]
            sd=float(np.std(baseline,ddof=1)); med=float(np.median(baseline)); mad=float(np.median(np.abs(np.array(baseline)-med))*1.4826)
            for board,pos in VIEWS+[('8',i) for i in range(1,9)]:
                pair=[r for r in rows if r['board']==board and r['position']==pos and r['pixel_size_um']==size]
                effect=pair[1][metric]-pair[0][metric]
                comparisons.append(dict(board=board,position=pos,pixel_size_um=size,metric=metric,
                    mask_effect=effect,blank_sd=sd,blank_scaled_mad=mad,blank_min=min(baseline),blank_max=max(baseline),
                    absolute_effect_over_blank_sd=abs(effect)/sd if sd else None,
                    absolute_effect_over_blank_scaled_mad=abs(effect)/mad if mad else None))
    write('mask_effect_vs_blank.csv',comparisons)

def spots():
    # Paired diagnostic in the existing aligned coordinate frame. Registration and
    # marker/boundary masks stay frozen; only stain and pair-change are recalculated.
    # This isolates spot rejection; it is not an end-to-end registration test.
    rows=[]; biases=[]; false=[]
    for board,pos in [('4',1),('4',2),('5',3)]:
      for size in SIZES:
        z=get_pair(board,pos,size); pre=z['pre']; post=z['post']; v=valid_support(pre.shape,z['warp'])
        # Retain marker/boundary protection from nominal pair, reconstruct categories.
        native=old.native_masks(pre,pos,size,P)
        protect=native['marker']|native['write_boundary']|z['marker']
        base_stain=old.blob_outlier_mask(post,size,P,protect)
        base_change=old.differential_stain(pre,post,size,P,protect,v)
        fixed=z['mask']
        rad=old.px(P['final_dilation_um'],size,0)
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1))
        base=fixed| (cv2.dilate((base_stain|base_change).astype(np.uint8),kernel)>0)
        baseline=stats(pre,post,v & ~base)
        for radius in [0.4,0.8,1.6,3.2]:
          seed=261008+int(board)*1000+pos*100+int(radius*10)
          rng=np.random.default_rng(seed)
          r=radius/size; margin=int(np.ceil(r))+4
          safe=cv2.erode((v & ~base).astype(np.uint8),np.ones((2*margin+1,2*margin+1),np.uint8))>0
          safe[:margin,:]=False; safe[-margin:,:]=False; safe[:,:margin]=False; safe[:,-margin:]=False
          ys,xs=np.where(safe); centers=[]
          for attempt in range(10000):
            idx=int(rng.integers(len(xs))); x,y=int(xs[idx]),int(ys[idx])
            if all((x-a)**2+(y-b)**2>(2*margin+1)**2 for a,b in centers): centers.append((x,y))
            if len(centers)==5: break
          if len(centers)!=5: raise RuntimeError('Spot placement failed')
          truth=np.zeros(pre.shape,bool); truths=[]
          for x,y in centers:
            yy,xx=np.ogrid[-margin:margin+1,-margin:margin+1]
            disk=(xx*xx+yy*yy)<=r*r
            sl=(slice(y-margin,y+margin+1),slice(x-margin,x+margin+1))
            truth[sl]|=disk; truths.append((x,y,sl,disk))
          expanded=cv2.dilate(truth.astype(np.uint8),np.ones((7,7),np.uint8))>0
          for contrast in [0.5,0.8,0.9,1.1,1.2,1.5]:
            injected=post.astype(np.float32).copy()
            for x,y,sl,disk in truths: injected[sl][disk]*=contrast
            clipped=int(np.count_nonzero(injected>np.iinfo(post.dtype).max))
            injected=np.clip(injected,0,np.iinfo(post.dtype).max).astype(post.dtype)
            stain=old.blob_outlier_mask(injected,size,P,protect)
            change=old.differential_stain(pre,injected,size,P,protect,v)
            mask=fixed|(cv2.dilate((stain|change).astype(np.uint8),kernel)>0)
            increment=mask & ~base
            fp=float((increment & ~expanded & v).sum()/pre.size)
            false.append(dict(board=board,position=pos,pixel_size_um=size,radius_um=radius,contrast_ratio=contrast,seed=seed,fp_all_pixels_fraction=fp))
            for repeat,(x,y,sl,disk) in enumerate(truths):
                coverage=float((mask[sl]&disk).sum()/disk.sum())
                incremental=float((increment[sl]&disk).sum()/disk.sum())
                rows.append(dict(board=board,position=pos,pixel_size_um=size,radius_um=radius,
                    contrast_ratio=contrast,seed=seed,repeat=repeat,x=x,y=y,detected=incremental>0,
                    mask_coverage_fraction=coverage,incremental_coverage_fraction=incremental,clipped_pixels=clipped))
            after=stats(pre,injected,v & ~mask); unmasked=stats(pre,injected,v)
            before_unmasked=stats(pre,post,v)
            for metric in ['median_difference','median_ratio','trim10_difference','trim10_ratio']:
                biases.append(dict(board=board,position=pos,pixel_size_um=size,radius_um=radius,
                    contrast_ratio=contrast,metric=metric,seed=seed,bias_after_mask=after[metric]-baseline[metric],
                    bias_without_mask=unmasked[metric]-before_unmasked[metric],fp_all_pixels_fraction=fp))
            write('spot_detection.csv',rows); write('spot_bias.csv',biases); write('spot_false_positive.csv',false)
          print('spots',board,pos,size,radius,flush=True)
    summary=[]
    for size in SIZES:
      for radius in [0.4,0.8,1.6,3.2]:
       for contrast in [0.5,0.8,0.9,1.1,1.2,1.5]:
        group=[r for r in rows if r['pixel_size_um']==size and r['radius_um']==radius and r['contrast_ratio']==contrast]
        summary.append(dict(pixel_size_um=size,radius_um=radius,contrast_ratio=contrast,spots=len(group),
            detection_rate=float(np.mean([r['detected'] for r in group])),
            mean_coverage=float(np.mean([r['mask_coverage_fraction'] for r in group]))))
    write('spot_summary.csv',summary)

def edges():
    rows=[]
    for board,pos in VIEWS:
      pre,post,*_=old.load_pair(RAW,board,pos)
      for phase,image in enumerate([pre,post]):
       for size in SIZES:
        masks={}
        masks['default']=old.line_ridge_mask(image,size,P,'dark')|old.profile_line_mask(image,size,P,'dark')
        for mode in ['reflect','edge']:
            pad=256
            padded=np.pad(image,pad,mode=mode)
            masks[mode+'_pad256']=(old.line_ridge_mask(padded,size,P,'dark')|old.profile_line_mask(padded,size,P,'dark'))[pad:-pad,pad:-pad]
        edge=np.zeros(image.shape,bool); edge[:50]=True;edge[-50:]=True;edge[:,:50]=True;edge[:,-50:]=True
        for mode,m in masks.items():
            rows.append(dict(board=board,position=pos,phase=phase,pixel_size_um=size,boundary=mode,
                edge_percent=float(m[edge].mean()*100),inside_percent=float(m[~edge].mean()*100),
                changed_edge_percent=float((m[edge]!=masks['default'][edge]).mean()*100),
                changed_inside_percent=float((m[~edge]!=masks['default'][~edge]).mean()*100)))
            if (board,pos)==('3',2) and size==SIZES[0]:
                overlay=old.make_overlay(image,{'marker':m},f'3-2 phase {phase} {mode} candidate')
                cv2.imencode('.png',overlay)[1].tofile(OUT/'images'/f'edge_3_2_{phase}_{mode}.png')
        write('edge_marker_rates.csv',rows)
        print('edges',board,pos,phase,size,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pairs','spots','edges','audit']);args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'cache').mkdir(exist_ok=True);(OUT/'images').mkdir(exist_ok=True)
    if args.action=='audit':
        before=json.loads((OUT/'input_hashes_before.json').read_text(encoding='utf-8'))
        result={p:digest(Path(p))==h for p,h in before.items()}
        (OUT/'input_unchanged.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        assert all(result.values())
    else:
        (OUT/'environment.json').write_text(json.dumps(dict(executable=sys.executable,python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,scikit_image=skimage.__version__,input=str(RAW),stage2=False,automatic_mask_approval=False),ensure_ascii=False,indent=2),encoding='utf-8')
        globals()[args.action]()

if __name__=='__main__': main()
