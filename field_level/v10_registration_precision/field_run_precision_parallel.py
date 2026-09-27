"""Independent process shards, each with its own random state and checkpoint file."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True)
    p.add_argument('--data-root',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    existing=a.output/(a.stage+'.csv')
    prerequisite={'subpixel':'baseline.csv','lattice':'subpixel.csv','iterative':'lattice.csv'}
    if a.stage in prerequisite and not (a.output/prerequisite[a.stage]).is_file():
        raise FileNotFoundError(f'Run the preceding stage first: {a.output/prerequisite[a.stage]}')
    prior=pd.read_csv(existing) if existing.exists() else pd.DataFrame()
    jobs=[]
    for index in range(a.workers):
        out=a.output/f'shard_{index}'; out.mkdir(exist_ok=True)
        # Existing rows are used only as skip keys; they are removed when merging.
        dest=out/(a.stage+'.csv')
        if not dest.exists() and len(prior): prior.to_csv(dest,index=False)
        log=(out/(a.stage+'.log')).open('w',encoding='utf-8')
        env=os.environ.copy(); env['OPENBLAS_NUM_THREADS']='1'; env['OMP_NUM_THREADS']='1'
        cmd=[sys.executable,str(Path(__file__).with_name('field_precision_benchmark.py')),
             '--stage',a.stage,'--data-root',a.data_root,'--output',str(out),
             '--shards',str(a.workers),'--shard-index',str(index)]
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        jobs.append((proc,log,out))
    for proc,log,out in jobs:
        status=proc.wait(); log.close()
        if status: raise RuntimeError(f'Worker failed: {out}; inspect stage log')
    frames=[]; ledger=[]
    for _,_,out in jobs:
        local=json.loads((out/(a.stage+'_inputs.json')).read_text(encoding='utf-8'))
        keys={r['case_id'] for r in local}; ledger.extend(local)
        frame=pd.read_csv(out/(a.stage+'.csv')); frames.append(frame[frame.case_id.isin(keys)])
    df=pd.concat(frames,ignore_index=True).sort_values('case_id')
    if df.case_id.duplicated().any() or len(df)!=len(ledger): raise RuntimeError('Incomplete/duplicate cases')
    df.to_csv(existing,index=False)
    (a.output/(a.stage+'_inputs.json')).write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
    good=df[df.status=='ok']; summary=dict(stage=a.stage,n=len(df),success=len(good),failed=int((df.status!='ok').sum()))
    for col in ['center_error_px','rotation_edge_error_px','scale_edge_error_px','grid_rmse_px']:
        summary.update({col+'_'+stat:float(fn(good[col])) for stat,fn in [('mean',np.mean),('median',np.median),('p95',lambda x:np.quantile(x,.95)),('max',np.max)]})
    (a.output/(a.stage+'_summary.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
