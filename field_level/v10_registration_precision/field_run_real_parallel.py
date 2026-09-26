"""Run disjoint production pairs, then reproduce v9 statistics from shared caches."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',required=True)
    p.add_argument('--data-root',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    base=[sys.executable,str(Path(__file__).with_name('field_real_precision.py')),
          '--stage',a.stage,'--data-root',a.data_root,'--output',str(a.output)]
    jobs=[]; env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    for index in range(a.workers):
        log=(a.output/f'worker_{index}.log').open('w',encoding='utf-8')
        proc=subprocess.Popen(base+['--shards',str(a.workers),'--shard-index',str(index)],
                              stdout=log,stderr=subprocess.STDOUT,env=env,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        jobs.append((proc,log))
    statuses=[]
    for proc,log in jobs: statuses.append(proc.wait());log.close()
    if any(statuses): raise RuntimeError(f'Production worker failures: {statuses}')
    # With every cache complete, this performs only the common full-cohort aggregation.
    with (a.output/'aggregate.log').open('w',encoding='utf-8') as log:
        subprocess.run(base,stdout=log,stderr=subprocess.STDOUT,env=env,check=True,
                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print((a.output/'summary_0/phase2_reanalysis_run_summary.json').read_text(),flush=True)

if __name__=='__main__':main()
