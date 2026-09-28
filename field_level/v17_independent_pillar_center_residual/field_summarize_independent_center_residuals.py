"""Summarize the v17 independent-center residual and detection-floor outputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/v17_independent_center_residual_20260928'

def main():
    d=pd.read_csv(OUT/'real_independent_center_residuals_by_fov.csv')
    phases=[]
    for p in (ROOT/'data/results/v16_real_spatial_subpixel_20260928/diagnostics').glob('*.json'):
        x=json.loads(p.read_text(encoding='utf-8'))
        phases.append({'fov_key':x['fov_key'],'local_phase_median_px':x['refined_residual']['phase_median_px'],
                       'n_local_phase_tiles':x['refined_residual']['n_common_phase_tiles']})
    phases=pd.DataFrame(phases)
    rows=[]
    metrics=['n_same_cell_matches','cell_mismatch_fraction','same_cell_median_px','same_cell_p95_px',
             'same_cell_max_px','nearest_all_median_px','nearest_all_p95_px','nearest_all_max_px',
             'affine_fit_component_rms_px','affine_remaining_rms_px','affine_explained_energy_fraction',
             'quadratic_fit_component_rms_px','quadratic_remaining_rms_px','quadratic_explained_energy_fraction']
    for (stage,q),part in d.groupby(['stage','confidence_quantile']):
        row={'stage':stage,'confidence_quantile':q,'n_fovs':int(part.fov_key.nunique())}
        for metric in metrics:
            x=part[metric].dropna()
            if not len(x): continue
            for label,value in [('median',x.median()),('p95_across_fovs',x.quantile(.95)),('maximum_across_fovs',x.max())]:
                row[f'{metric}_{label}']=float(value)
        row['pooled_compared_centers']=int(part.n_compared.sum())
        row['pooled_half_pitch_ambiguous']=int(part.n_cell_mismatch_gt_half_pitch.sum())
        row['pooled_half_pitch_ambiguous_fraction']=row['pooled_half_pitch_ambiguous']/max(1,row['pooled_compared_centers'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'real_residual_stage_summary.csv',index=False)

    q50=d[d.confidence_quantile==.5]
    current=q50[q50.stage=='current_spatial_subpixel']
    comparison=[]
    for stage in ['v11_coarse_baseline','v15_coarse_no_subpixel']:
        old=q50[q50.stage==stage]
        x=current.merge(old,on='fov_key',suffixes=('_current','_reference'),validate='one_to_one')
        row={'comparison':'current_spatial_subpixel_vs_'+stage,'n_paired':int(len(x))}
        for metric in ['same_cell_median_px','same_cell_p95_px','cell_mismatch_fraction']:
            delta=x[f'{metric}_current']-x[f'{metric}_reference']
            row[metric+'_current_lower']=int((delta < -1e-8).sum())
            row[metric+'_current_higher']=int((delta > 1e-8).sum())
            row[metric+'_unchanged']=int((delta.abs() <= 1e-8).sum())
            row[metric+'_median_delta']=float(delta.median())
            row[metric+'_mean_delta']=float(delta.mean())
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(OUT/'real_residual_paired_stage_comparison.csv',index=False)

    ranked=current.merge(phases,on='fov_key',how='left',validate='one_to_one')
    ranked.sort_values(['same_cell_p95_px','cell_mismatch_fraction'],ascending=False).to_csv(
        OUT/'real_residual_ranked_fov.csv',index=False)
    linked=ranked[['fov_key','local_phase_median_px','n_local_phase_tiles','cell_mismatch_fraction',
                   'same_cell_median_px','same_cell_p95_px','affine_fit_component_rms_px',
                   'affine_remaining_rms_px','affine_explained_energy_fraction',
                   'quadratic_fit_component_rms_px','quadratic_remaining_rms_px',
                   'quadratic_explained_energy_fraction']]
    linked.to_csv(OUT/'local_phase_vs_center_residuals.csv',index=False)
    rho=spearmanr(ranked.local_phase_median_px,ranked.same_cell_median_px)
    rho_p95=spearmanr(ranked.local_phase_median_px,ranked.same_cell_p95_px)
    targets=['260829_p50_sam_3_4','260829_p50_sam_7_8','260826_p50_sam_11_7',
             '260825_p50_dna_1_7','260828_p50_dna_8_6']
    linked[linked.fov_key.isin(targets)].to_csv(OUT/'focus_case_residual_summary.csv',index=False)

    synth=pd.read_csv(OUT/'semisynthetic_detection_floor_summary.csv')
    real=ranked
    summary={'n_real_qc_accepted_fovs':int(real.fov_key.nunique()),
       'center_pair_compared':int(real.n_compared.sum()),
       'center_pair_half_pitch_ambiguous':int(real.n_cell_mismatch_gt_half_pitch.sum()),
       'center_pair_half_pitch_ambiguous_fraction':float(real.n_cell_mismatch_gt_half_pitch.sum()/real.n_compared.sum()),
       'fovs_half_pitch_ambiguous_fraction_gt_25pct':int((real.cell_mismatch_fraction>.25).sum()),
       'fovs_half_pitch_ambiguous_fraction_gt_50pct':int((real.cell_mismatch_fraction>.50).sum()),
       'local_phase_px':{'median':float(ranked.local_phase_median_px.median()),
                         'p95':float(ranked.local_phase_median_px.quantile(.95)),
                         'max':float(ranked.local_phase_median_px.max()),
                         'spearman_vs_center_median_rho':float(rho.statistic),
                         'spearman_vs_center_median_p':float(rho.pvalue),
                         'spearman_vs_center_p95_rho':float(rho_p95.statistic),
                         'spearman_vs_center_p95_p':float(rho_p95.pvalue)},
       'current_spatial_subpixel_q50':{metric:{'median_fov_stat':float(real[metric].median()),
           'p95_across_fov_stat':float(real[metric].quantile(.95)),
           'maximum_across_fovs':float(real[metric].max())} for metric in metrics if metric in real},
       'semisynthetic_floor':synth.to_dict('records'),
       'stage_comparisons':comparison,
       'focus_cases':linked[linked.fov_key.isin(targets)].to_dict('records'),
       'high_local_phase_top10':ranked.sort_values('local_phase_median_px',ascending=False)[
           ['fov_key','local_phase_median_px','same_cell_median_px','same_cell_p95_px','cell_mismatch_fraction']
           ].head(10).to_dict('records')}
    (OUT/'independent_center_residual_summary.json').write_text(
        json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pair_count':summary['center_pair_compared'],
        'half_pitch_ambiguous_fraction':summary['center_pair_half_pitch_ambiguous_fraction'],
        'current':summary['current_spatial_subpixel_q50'],
        'local_phase':summary['local_phase_px'],'comparisons':comparison,
        'focus_cases':summary['focus_cases']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
