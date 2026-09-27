"""Compare full-cohort real-data statistics and QC status against the baseline."""
import argparse
import json
from pathlib import Path
import pandas as pd

def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if pd.isna(value):return None
    if hasattr(value,'item'):return value.item()
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args()
    cur=a.input/'summary_0';base=a.baseline/'summary_0'
    f=pd.read_csv(cur/'phase2_reanalysis_fov_summary.csv')
    b=pd.read_csv(base/'phase2_reanalysis_fov_summary.csv')
    keys=['dataset','sample','position']
    if f.duplicated(keys).any() or b.duplicated(keys).any():raise RuntimeError('Duplicate FOV key')
    merged=b[keys+['status']].merge(f[keys+['status']],on=keys,suffixes=('_baseline','_stage'),how='outer',indicator=True)
    exact=bool((merged._merge=='both').all() and (merged.status_baseline==merged.status_stage).all())
    p0=pd.read_csv(base/'phase2_reanalysis_mannwhitney_summary.csv')
    p1=pd.read_csv(cur/'phase2_reanalysis_mannwhitney_summary.csv')
    columns=['dataset','n_high','n_blank','p_two_sided_exact','p_one_sided_greater_exact']
    joined=p0[columns].merge(p1[columns],on='dataset',suffixes=('_baseline','_stage'),how='outer')
    joined.to_csv(a.input/'statistical_comparison.csv',index=False)
    rows=[]
    for r in joined.itertuples(index=False):
        cols=joined.columns.tolist();row=dict(zip(cols,r))
        equal=True
        for metric in ['p_two_sided_exact','p_one_sided_greater_exact']:
            x=row[f'{metric}_baseline'];y=row[f'{metric}_stage']
            equal &= (pd.isna(x) and pd.isna(y)) or (pd.notna(x) and pd.notna(y) and abs(x-y)<1e-12)
        row['p_values_unchanged']=bool(equal);rows.append(row)
    report={'stage':a.input.name,'pairs':int(len(f)),'status_counts':{str(k):int(v) for k,v in f.status.value_counts().items()},
            'same_fov_keys_and_statuses_as_baseline':exact,
            'all_two_sided_and_directional_p_values_unchanged':bool(all(x['p_values_unchanged'] for x in rows)),
            'datasets':rows}
    (a.input/'analysis_comparison.json').write_text(json.dumps(clean(report),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in report if k!='datasets'},indent=2))
    print(joined[['dataset','n_high_stage','n_blank_stage','p_two_sided_exact_stage']].to_string(index=False))

if __name__=='__main__':main()
