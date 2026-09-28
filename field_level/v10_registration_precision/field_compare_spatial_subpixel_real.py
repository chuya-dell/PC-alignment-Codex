"""Compare the v15 spatial-support + subpixel run with the saved real baseline."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / 'data/results/v16_real_spatial_subpixel_20260928'
OLD = ROOT / 'data/results/v11_registration_precision_20260926/real_baseline'

old = pd.read_csv(OLD / 'real_residuals.csv')
new_rows = []
for path in sorted((NEW / 'diagnostics').glob('*.json')):
    d = json.loads(path.read_text(encoding='utf-8'))
    b, f = d['baseline_residual'], d['refined_residual']
    new_rows.append({
        'fov_key': d['fov_key'],
        'new_coarse_photometric_residual': b['photometric_residual'],
        'new_final_photometric_residual': f['photometric_residual'],
        'new_coarse_phase_median_px': b['phase_median_px'],
        'new_final_phase_median_px': f['phase_median_px'],
        'new_coarse_phase_tiles': b['n_common_phase_tiles'],
        'new_final_phase_tiles': f['n_common_phase_tiles'],
    })
new = pd.DataFrame(new_rows)
old = old[old['stage'] == 'baseline'][[
    'fov_key', 'baseline_photometric_residual', 'baseline_phase_median_px',
    'baseline_n_common_phase_tiles'
]]
merged = old.merge(new, on='fov_key', how='inner', validate='one_to_one')
merged['delta_photometric_residual'] = (
    merged['new_final_photometric_residual'] - merged['baseline_photometric_residual'])
merged['delta_phase_median_px'] = (
    merged['new_final_phase_median_px'] - merged['baseline_phase_median_px'])
merged['delta_coarse_photometric_residual'] = (
    merged['new_coarse_photometric_residual'] - merged['baseline_photometric_residual'])
merged['delta_coarse_phase_median_px'] = (
    merged['new_coarse_phase_median_px'] - merged['baseline_phase_median_px'])
merged.to_csv(NEW / 'real_residual_comparison.csv', index=False)

def summarize(column):
    d = merged[column].dropna()
    return {
        'n': int(len(d)), 'lower': int((d < -1e-8).sum()),
        'higher': int((d > 1e-8).sum()), 'unchanged': int((d.abs() <= 1e-8).sum()),
        'median_delta': float(d.median()), 'mean_delta': float(d.mean()),
    }

status_old = pd.read_csv(OLD / 'summary_0/phase2_reanalysis_fov_summary.csv')
status_new = pd.read_csv(NEW / 'summary_0/phase2_reanalysis_fov_summary.csv')
keys = ['dataset', 'sample', 'position']
status = status_old[keys + ['status']].merge(
    status_new[keys + ['status']], on=keys, how='outer', suffixes=('_old', '_new'),
    indicator=True, validate='one_to_one')
status.to_csv(NEW / 'qc_status_comparison.csv', index=False)

summary = {
    'pairs_attempted': int(len(status_new)),
    'old_status_counts': {str(k): int(v) for k, v in status_old.status.value_counts().items()},
    'new_status_counts': {str(k): int(v) for k, v in status_new.status.value_counts().items()},
    'same_status_pairs': int(((status['_merge'] == 'both') &
                              (status.status_old == status.status_new)).sum()),
    'old_only_accepted': status.loc[(status.status_old == 'ok') &
                                    (status.status_new != 'ok'), keys].to_dict('records'),
    'new_only_accepted': status.loc[(status.status_new == 'ok') &
                                    (status.status_old != 'ok'), keys].to_dict('records'),
    'residual_common_pairs': int(len(merged)),
    'photometric_residual_delta_old_to_new_final': summarize('delta_photometric_residual'),
    'phase_median_delta_old_to_new_final_px': summarize('delta_phase_median_px'),
    'photometric_residual_delta_old_to_new_coarse': summarize('delta_coarse_photometric_residual'),
    'phase_median_delta_old_to_new_coarse_px': summarize('delta_coarse_phase_median_px'),
}
(NEW / 'real_evaluation_summary.json').write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
