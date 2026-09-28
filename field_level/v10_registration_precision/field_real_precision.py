"""Re-use v9 production statistics with an opt-in registration adapter.

This runner never imports Anti. Its source manifest is the exact 409-row artifact
already used by the preceding mask-sampling reanalysis. Frozen Anti nonconvergence
cases are neither repaired nor used to alter exclusions here.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared import registration as reg
from shared.image_qc import bright_band_mask
from field_level.v9_phase2_production_reanalysis import field_run_phase2_production_reanalysis as production

def residuals(pre,post,matrices):
    """Common support, independent high-pass photometry and local phase shifts.

These are image-consistency surrogates, not errors against physical ground truth.
No metric from this function is used to accept a refinement.
"""
    a=reg.image01_for_registration(pre); b=reg.image01_for_registration(post)
    a=a-cv2.GaussianBlur(a,(0,0),3)
    b=b-cv2.GaussianBlur(b,(0,0),3)
    h,w=a.shape
    ma=cv2.dilate(bright_band_mask(pre).astype(np.uint8),np.ones((17,17),np.uint8))>0
    mb=cv2.dilate(bright_band_mask(post).astype(np.uint8),np.ones((17,17),np.uint8))
    common=~ma; common[:40]=False;common[-40:]=False;common[:,:40]=False;common[:,-40:]=False
    aligned=[]
    for m in matrices:
        aligned.append(cv2.warpAffine(b,m,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP))
        mask=cv2.warpAffine(mb,m,(w,h),flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,borderValue=1)
        common &= mask==0
    result=[]; tile_rows=[]
    window=cv2.createHanningWindow((64,64),cv2.CV_32F)
    # Fixed spatial tiles; independent of fitted feature and lattice sites.
    for y in np.linspace(96,h-97,7).astype(int):
        for x in np.linspace(96,w-97,7).astype(int):
            sl=np.s_[y-32:y+32,x-32:x+32]
            if not common[sl].all(): continue
            rows=[]
            for image in aligned:
                shift,response=cv2.phaseCorrelate(a[sl].copy(),image[sl].copy(),window)
                rows.append((float(np.hypot(*shift)),float(response)))
            # Identical accepted tiles for every compared transform.
            if all(r[1]>.15 and r[0]<16 for r in rows): tile_rows.append(rows)
    sparse=common[3::8,5::8]
    x=a[3::8,5::8][sparse].astype(float); x-=x.mean()
    for i,image in enumerate(aligned):
        y=image[3::8,5::8][sparse].astype(float); y-=y.mean()
        correlation=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))) if len(x) and np.linalg.norm(y)>0 else float('nan')
        phase=[r[i][0] for r in tile_rows]
        result.append(dict(highpass_correlation=correlation,photometric_residual=1-correlation,
                           phase_median_px=float(np.median(phase)) if phase else float('nan'),
                           n_common_phase_tiles=len(phase),n_common_pixels=len(x)))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['baseline','subpixel','lattice','iterative','spatial_subpixel'],required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'data/results/v10_phase2_bright_band_mask_20260926/masked_50862c8/source_manifest_attached.csv')
    p.add_argument('--shards',type=int,default=1); p.add_argument('--shard-index',type=int,default=0)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); cv2.setNumThreads(1)
    manifest=pd.read_csv(a.manifest)
    for col in ['pre_path','post_path']:
        manifest[col]=[str(a.data_root/s.replace('\\','/').split('/4.生データ D/')[-1]) for s in manifest[col]]
        missing=[s for s in manifest[col] if not Path(s).is_file()]
        if missing: raise FileNotFoundError(missing[:3])
    manifest=manifest.iloc[a.shard_index::a.shards]
    local=a.output/f'manifest_{a.shard_index}.csv'; manifest.to_csv(local,index=False)
    cache=a.output/'cache'; diagnostic_dir=a.output/'diagnostics'; diagnostic_dir.mkdir(exist_ok=True)
    current={}
    original_process=production.process_pair
    def process(row,pitch):
        current['key']=production.fov_key(row)
        cv2.setRNGSeed(20260926)
        return original_process(row,pitch)
    def register(pre,post,return_qc=False,**kwargs):
        if a.stage=='baseline':
            coarse=reg.register_image_pair_affine(pre,post);final,info=coarse,{}
        elif a.stage=='spatial_subpixel':
            # Re-estimate coarse registration through the v15 spatial-support path,
            # then apply the same guarded subpixel stage used by the earlier cohort.
            coarse=reg.register_image_pair_affine(pre,post)
            from shared.v2_registration_precision.refinement import register_refined
            final,info=register_refined(pre,post,stage='subpixel',initial=coarse)
        else:
            prior={'subpixel':'real_baseline','lattice':'real_cascade_subpixel',
                   'iterative':'real_cascade_lattice'}[a.stage]
            anchor= a.output.parent/prior/'diagnostics'/(current['key']+'.json')
            coarse=(np.asarray(json.loads(anchor.read_text(encoding='utf-8'))['matrix'],dtype=np.float32)
                    if anchor.exists() else reg.register_image_pair_affine(pre,post))
            from shared.v2_registration_precision.refinement import register_refined
            final,info=register_refined(pre,post,stage=a.stage,initial=coarse)
        measures=residuals(pre,post,[coarse,final])
        payload=dict(fov_key=current['key'],stage=a.stage,baseline_matrix=coarse.tolist(),matrix=final.tolist(),
                     baseline_residual=measures[0],refined_residual=measures[1],refinement=info)
        (diagnostic_dir/(current['key']+'.json')).write_text(json.dumps(payload,indent=2),encoding='utf-8')
        qc=reg.assess_affine_transform_qc(final,pre.shape)
        qc['mask_fraction']=float(bright_band_mask(pre).mean())
        return (final,qc) if return_qc else final
    production.reg=SimpleNamespace(**{k:getattr(reg,k) for k in dir(reg) if not k.startswith('__')})
    production.reg.register_image_pair_affine=register
    production.process_pair=process
    sys.argv=[sys.argv[0],'--pair-manifest',str(local),'--cache-dir',str(cache),
              '--out-dir',str(a.output/f'summary_{a.shard_index}')]
    production.main()

if __name__=='__main__': main()
