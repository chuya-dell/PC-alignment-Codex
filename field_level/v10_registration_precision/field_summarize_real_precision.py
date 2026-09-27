"""Summarize common-support real-image residuals without labeling them geometric truth."""
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=[]
    for path in sorted(a.input.glob('diagnostics/*.json')):
        d=json.loads(path.read_text(encoding='utf-8')); b=d['baseline_residual']; f=d['refined_residual']
        row={'fov_key':d['fov_key'],'stage':d['stage'],'baseline_matrix':json.dumps(d['baseline_matrix']),
             'matrix':json.dumps(d['matrix']),
             'subpixel_accepted':d.get('refinement',{}).get('subpixel',{}).get('accepted',False),
             'lattice_accepted':d.get('refinement',{}).get('lattice',{}).get('accepted',False)}
        for k in ['photometric_residual','highpass_correlation','phase_median_px','n_common_phase_tiles','n_common_pixels']:
            row[f'baseline_{k}']=b[k];row[f'refined_{k}']=f[k]
            if k in ('photometric_residual','phase_median_px'):
                row[f'delta_{k}']=f[k]-b[k]
        rows.append(row)
    df=pd.DataFrame(rows)
    base=pd.DataFrame([json.loads(x.read_text(encoding='utf-8')) for x in a.baseline.glob('diagnostics/*.json')])
    if set(df.fov_key)!=set(base.fov_key):raise RuntimeError('Real-data registration accepted cohort differs from baseline')
    a.output.parent.mkdir(parents=True,exist_ok=True);df.to_csv(a.output,index=False)
    for metric in ['photometric_residual','phase_median_px']:
        d=df[f'delta_{metric}'];print(metric,{'n':len(d),'lower':int((d< -1e-8).sum()),
            'higher':int((d>1e-8).sum()),'unchanged':int((abs(d)<=1e-8).sum()),
            'median_delta':float(d.median()),'mean_delta':float(d.mean())})
    print('refinement accepted',{'subpixel':int(df.subpixel_accepted.sum()),
                                'lattice':int(df.lattice_accepted.sum())})

if __name__=='__main__':main()
