"""Aggregate individual v5 semi-synthetic runs and compute spike-count linearity."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
 p=argparse.ArgumentParser(); p.add_argument('--runs',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 trial_files=sorted(a.runs.glob('fov*/semisynthetic_trial_results.csv')); reg_files=sorted(a.runs.glob('fov*/registration_known_truth_errors.csv'))
 if len(trial_files)!=5 or len(reg_files)!=5: raise RuntimeError(f'Expected 5 runs; found {len(trial_files)} trial and {len(reg_files)} registration files')
 trial=pd.concat([pd.read_csv(x) for x in trial_files],ignore_index=True); reg=pd.concat([pd.read_csv(x) for x in reg_files],ignore_index=True)
 linear=[]
 for (method,pos),g in trial.groupby(['method','position']):
  x=g.spike_count.to_numpy(float); y=g.recovered_count.to_numpy(float); slope,intercept=np.polyfit(x,y,1); pred=slope*x+intercept; r2=1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)),1e-12); linear.append(dict(method=method,position=pos,recovery_slope=slope,recovery_intercept=intercept,recovery_r2=r2))
 linear=pd.DataFrame(linear); summary=trial.groupby('method',as_index=False).agg(mean_recovery_rate=('recovery_rate','mean'),mean_false_positive_rate=('false_positive_rate','mean')).merge(linear.groupby('method',as_index=False).agg(mean_recovery_slope=('recovery_slope','mean'),mean_recovery_r2=('recovery_r2','mean')),on='method'); f6=trial[trial.position==6].groupby('method',as_index=False).agg(fov6_recovery_rate=('recovery_rate','mean'),fov6_false_positive_rate=('false_positive_rate','mean')).merge(linear[linear.position==6][['method','recovery_slope','recovery_r2']],on='method').rename(columns={'recovery_slope':'fov6_recovery_slope','recovery_r2':'fov6_recovery_r2'}); summary=summary.merge(f6,on='method')
 trial.to_csv(a.output/'semisynthetic_trial_results.csv',index=False); reg.to_csv(a.output/'registration_known_truth_errors.csv',index=False); linear.to_csv(a.output/'spike_recovery_linearity.csv',index=False); summary.to_csv(a.output/'method_summary.csv',index=False); print(summary.to_string(index=False))
if __name__=='__main__': main()
