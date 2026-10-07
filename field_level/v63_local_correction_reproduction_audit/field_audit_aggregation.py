"""Separate historical-cache replay, raw gate, and Holm-family diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from field_audit_reproduction import BASE, V54, OUT, DATES, load_runner, digest

def check(new, reference, keys, metrics, label):
    new, reference = new.copy(), reference.copy()
    for frame in (new, reference):
        frame['date'] = frame.date.astype(str)
        if 'board' in keys:
            frame['board'] = frame.board.astype(str)
    joined = new.merge(reference,on=keys,how='outer',suffixes=('_new','_reference'),indicator=True,validate='one_to_one')
    rows = []
    for _, r in joined.iterrows():
        for metric in metrics:
            a,b = r[metric+'_new'],r[metric+'_reference']
            same = r['_merge']=='both' and np.isclose(a,b,rtol=1e-10,atol=1e-12,equal_nan=True)
            if metric in ('positive','pillar_total','blank_n','field_n'):
                same = r['_merge']=='both' and a==b
            rows.append({**{k:r[k] for k in keys},'metric':metric,'new':a,'reference':b,'match':bool(same)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'tables'/f'table_{label}.csv',index=False,encoding='utf-8-sig')
    return dict(comparisons=len(result), matched=int(result.match.sum()), mismatched=int((~result.match).sum()))

def replay(cache, destination):
    run=load_runner()
    run.OUT=destination
    (destination/'tables').mkdir(parents=True,exist_ok=True)
    pairs,_,_=run.build_inventory()
    data=[]
    for p in pairs:
        src=cache/f"{p['date']}_{p['board']}_{p['fov']}.npz"
        if src.exists():
            with np.load(src,allow_pickle=False) as z:
                data.append((p,z['delta'].copy(),None,None))
    # Standard numerical block unchanged, supplied with explicitly named caches.
    import inspect,textwrap
    source=inspect.getsource(run.main)
    code=source[source.index('    # assign same-date blank thresholds'):source.index('    # Plots:')]
    scope=dict(vars(run),data=data)
    exec(compile(textwrap.dedent(code),'standard_aggregation','exec'),scope)
    return scope

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'tables').mkdir(exist_ok=True)
    reference=pd.read_csv(BASE/'tables/table_threshold_sweep_concentration_summary.csv')
    reference['date']=reference.date.astype(str)
    if len(reference)!=540 or reference.date.nunique()!=9:
        raise RuntimeError('Historical Holm family changed')
    replay_result=replay(BASE/'tables/cached_field_differences',OUT/'accepted_cache_replay')
    fresh_result=replay(V54/'standard/tables/cached_field_differences',OUT/'v54_cache_replay')
    results={}
    for label,scope in [('accepted',replay_result),('v54',fresh_result)]:
        # Replace only the four dates' p values in the frozen 540-row family.
        dose=scope['dose_df'].copy()
        merged=reference.merge(dose[['date','n','dose_molar','p_one_sided']],on=['date','n','dose_molar'],how='left',suffixes=('_ref','_new'),validate='one_to_one')
        p=merged.p_one_sided_new.fillna(merged.p_one_sided_ref).to_numpy()
        merged['p_holm_540']=multipletests(p,method='holm')[1]
        lookup=merged[['date','n','dose_molar','p_holm_540']]
        dose=dose.rename(columns={'p_holm_global':'p_holm_234'}).merge(lookup,on=['date','n','dose_molar'],validate='one_to_one')
        dose['p_holm_global']=dose.p_holm_540
        dose.to_csv(OUT/'tables'/f'table_{label}_holm_540_and_234.csv',index=False,encoding='utf-8-sig')
        ref4=reference[reference.date.isin(DATES)]
        checks={}
        checks['dose']=check(dose,ref4,['date','n','dose_molar'],['positive','pillar_total','field_n','p_one_sided','p_holm_global'],label+'_dose')
        for key,filename,keys,metrics in [
            ('ff','table_threshold_sweep_excess_rate_by_field.csv',['date','board','field','n'],['pillar_total','positive','excess_rate']),
            ('thresholds','table_threshold_sweep_thresholds_by_date.csv',['date','n'],['blank_mean','blank_sd','threshold','blank_n'])]:
            old=pd.read_csv(BASE/'tables'/filename);old['date']=old.date.astype(str)
            checks[key]=check(scope[key],old[old.date.isin(DATES)],keys,metrics,label+'_'+key)
        rows=[]
        for (date,n),g in scope['ff'].groupby(['date','n']):
            g=g[pd.to_numeric(g.condition,errors='coerce')>0]
            r=spearmanr(g.condition.astype(float),g.excess_rate)
            rows.append(dict(date=date,n=n,spearman_field_rho=r.statistic,spearman_two_sided_p=r.pvalue))
        old=pd.read_csv(BASE/'tables/table_spearman_field_level_by_date_threshold.csv');old['date']=old.date.astype(str)
        checks['spearman']=check(pd.DataFrame(rows),old[old.date.isin(DATES)],['date','n'],['spearman_field_rho','spearman_two_sided_p'],label+'_spearman')
        results[label]=checks
    configuration=dict(reference_holm_dates=sorted(reference.date.unique()),reference_holm_test_count=540,
                       diagnostic_holm_dates=list(DATES),diagnostic_holm_test_count=234,
                       primary_multiplier=3, multipliers=[1,1.5,2,2.5,3,3.5,4,5,6],
                       bright_band_mask='auto (unchanged)',mask_stains=False,pitch_px=7.286,blank_sd_ddof=0,
                       delta='pre minus post',integer_match='exact',float_rtol=1e-10,float_atol=1e-12,
                       gate='raw image regeneration must match historical arrays and all numerical tables; accepted-cache replay is not raw gate',
                       reference_sha256=digest(BASE/'tables/table_threshold_sweep_concentration_summary.csv'))
    (OUT/'reproduction_basis.json').write_text(json.dumps(configuration,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'aggregation_audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    main()
