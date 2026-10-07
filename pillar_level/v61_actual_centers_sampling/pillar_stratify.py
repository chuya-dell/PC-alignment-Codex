"""Compare like-for-like field subsets and support; no low-match efficacy claim."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'field_level/v60_band_scar_controls'))
import field_control_common as M
import pandas as pd,numpy as np
OUT=ROOT/'data/results/v61_actual_centers_sampling'

def main():
    t=pd.read_csv(OUT/'sampling_metrics.csv',dtype={'date':str});a=pd.read_csv(OUT/'fit_audit.csv');t=t.merge(a[['fid','center_match_fraction']],on='fid')
    t['match_stratum']=np.where(t.center_match_fraction>=.8,'ge80','lt80');t.to_csv(OUT/'sampling_metrics_stratified.csv',index=False)
    rows=[]
    for fixed,name in [(True,'fixed29'),(False,'all90')]:
        sel=t[t.fixed29] if fixed else t
        for stratum,g in sel.groupby('match_stratum'):
            for method,q in g.groupby('method'):
                rows.append(dict(selection=name,stratum=stratum,method=method,fields=q.fid.nunique(),rate_median=q.rate.median(),strength_median=q.strength.median(),density_sd_median=q.density_sd.median(),power_median=q.band_power.median(),map_corr_median=q.baseline_map_corr.median(),axis_median=q.axis_deg.median()))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/'stratum_summary.csv',index=False)
    comparisons=[]
    for (stratum,fixed),g in t.groupby(['match_stratum','fixed29']):
        base=g[g.method=='rerun_round_center_valid'].set_index('fid')
        for method in ['ideal_bilinear_center_valid','centers_round','centers_bilinear']:
            q=g[g.method==method].set_index('fid').loc[base.index]
            for metric in ['rate','strength','density_sd','band_power']:
                delta=q[metric]-base[metric];comparisons.append(dict(stratum=stratum,fixed29=bool(fixed),method=method,metric=metric,fields=len(delta),before_median=base[metric].median(),after_median=q[metric].median(),paired_change_median=delta.median(),reduced_fields=int((delta<0).sum())))
    pd.DataFrame(comparisons).to_csv(OUT/'common_support_paired_comparison.csv',index=False)
    off=t[t.method.str.startswith('offset')].copy()
    base=t[t.method=='rerun_round'].set_index('fid')
    off['phase_change_rad']=[np.angle(np.exp(1j*(r.phase_at_baseline_frequency-base.loc[r.fid,'phase_at_baseline_frequency']))) for r in off.itertuples()]
    off['axis_change_deg']=[abs((r.axis_deg-base.loc[r.fid,'axis_deg']+90)%180-90) for r in off.itertuples()]
    off['position_change_normal_px']=[r.phase_change_rad/(2*np.pi)*base.loc[r.fid,'period_px'] for r in off.itertuples()]
    off.to_csv(OUT/'offset_relationships.csv',index=False)
    off[off.fixed29].groupby(['match_stratum','method'])[['rate','strength','baseline_map_corr','phase_change_rad','position_change_normal_px','axis_change_deg']].median().to_csv(OUT/'offset_stratum_summary.csv')
    fig,axs=M.plt.subplots(2,3,figsize=(14,8))
    for axes,(stratum,g) in zip(axs,t.groupby('match_stratum',sort=True)):
        base=g[g.method=='rerun_round_center_valid'].set_index('fid')
        for m in ['ideal_bilinear_center_valid','centers_round','centers_bilinear']:
            q=g[g.method==m].set_index('fid').loc[base.index]
            for ax,col in zip(axes,['rate','strength','density_sd']):ax.scatter(base[col],q[col],s=16,label=m)
        for ax,col in zip(axes,['rate','strength','density_sd']):ax.set(xlabel='Rounded common-support '+col,ylabel='Changed '+col,title=f'{stratum}, fields={len(base)}, fixed29={g[g.fixed29].fid.nunique()}');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/'step1_stratified.png',dpi=150);M.plt.close(fig)
    print(a.groupby(a.center_match_fraction>=.8).size());print(summary[(summary.selection=='fixed29')&summary.method.isin(['rerun_round_center_valid','centers_bilinear','ideal_bilinear_center_valid'])].to_string(index=False))

if __name__=='__main__':main()
