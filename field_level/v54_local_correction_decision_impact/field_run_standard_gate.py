"""Run the unmodified standard functions, restricted to the four requested dates.

The derived source differs only by the authorized import and machine paths.
No accepted difference cache is used. The original main's numerical summary
block is executed verbatim, without its expensive per-pillar text export or plots.
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import sys
sys.dont_write_bytecode = True
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/v54_local_correction_decision_impact_20261006'
BASE = Path('W:/GoogleDrive/chuya2816/5.解析結果_chu/20260928_digital_judgment_current_alignment/tables')
DATES = ('260825', '260827', '260828', '260829')


def compute_date(pairs):
    """Independent date worker; retain original compute and its field order."""
    spec = importlib.util.spec_from_file_location('standard_worker', Path(__file__).with_name('field_standard_derived.py'))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.cv2.setNumThreads(1)
    return run.compute(pairs)


def compare(new, old, keys, columns, name):
    for table in (new, old):
        table['date'] = table.date.astype(str)
        if 'board' in keys:
            table['board'] = table.board.astype(str)
    old = old[old.date.isin(DATES)]
    merged = new.merge(old, on=keys, how='outer', suffixes=('_new', '_accepted'), indicator=True, validate='one_to_one')
    records = []
    for _, row in merged.iterrows():
        for column in columns:
            a, b = row.get(column+'_new', np.nan), row.get(column+'_accepted', np.nan)
            same = row['_merge'] == 'both' and ((pd.isna(a) and pd.isna(b)) or np.isclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True))
            if column in ('positive', 'pillar_total', 'field_n', 'blank_n'):
                same = row['_merge'] == 'both' and a == b
            records.append({**{key:row[key] for key in keys}, 'metric':column, 'recomputed':a, 'accepted':b, 'difference':a-b, 'match':bool(same), 'row_presence':row['_merge']})
    result = pd.DataFrame(records)
    result.to_csv(OUT/'tables'/f'table_gate_{name}.csv', index=False, encoding='utf-8-sig')
    return {'table':name, 'comparisons':len(result), 'matched':int(result.match.sum()), 'mismatched':int((~result.match).sum())}


def main():
    source = Path(__file__).with_name('field_standard_derived.py')
    spec = importlib.util.spec_from_file_location('standard_v54', source)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.DATES = {k:v for k,v in run.DATES.items() if k in DATES}
    run.cv2.setNumThreads(1)
    pairs, inventory, exceptions = run.build_inventory()
    print('FRESH RAW RUN:', len(pairs), 'fields', flush=True)
    pd.DataFrame(inventory).to_csv(OUT/'tables/table_input_inventory.csv', index=False, encoding='utf-8-sig')
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(compute_date, [[p for p in pairs if p['date']==date] for date in DATES]))
    rows = [row for result in results for row in result[0]]
    data = [row for result in results for row in result[1]]
    pd.DataFrame([{k:v for k,v in row.items() if k not in ('pre','post')} for row in rows]).to_csv(run.OUT/'tables/table_registration_field_qc.csv', index=False, encoding='utf-8-sig')
    # Extract the original numerical block unchanged; all names resolve in its module.
    code = source.read_text(encoding='utf-8')
    begin = code.index('    # assign same-date blank thresholds')
    end = code.index('    # Plots:')
    import textwrap
    scope = dict(vars(run), data=data)
    exec(compile(textwrap.dedent(code[begin:end]), str(source), 'exec'), scope)
    stats = []
    ff = scope['ff']
    for (date, n), group in ff.groupby(['date','n']):
        group = group[pd.to_numeric(group.condition, errors='coerce') > 0]
        result = spearmanr(group.condition.astype(float), group.excess_rate)
        stats.append(dict(date=date,n=n,spearman_field_rho=result.statistic,spearman_two_sided_p=result.pvalue))
    sp = pd.DataFrame(stats)
    sp.to_csv(OUT/'tables/table_recomputed_spearman.csv',index=False,encoding='utf-8-sig')
    summaries = []
    summaries.append(compare(ff, pd.read_csv(BASE/'table_threshold_sweep_excess_rate_by_field.csv'), ['date','board','field','n'], ['pillar_total','positive','excess_rate'], 'fields'))
    summaries.append(compare(scope['dose_df'], pd.read_csv(BASE/'table_threshold_sweep_concentration_summary.csv'), ['date','n','dose_molar'], ['field_n','pillar_total','positive','p_one_sided','p_holm_global'], 'concentrations'))
    summaries.append(compare(sp, pd.read_csv(BASE/'table_spearman_field_level_by_date_threshold.csv'), ['date','n'], ['spearman_field_rho','spearman_two_sided_p'], 'spearman'))
    summaries.append(compare(scope['thresholds'], pd.read_csv(BASE/'table_threshold_sweep_thresholds_by_date.csv'), ['date','n'], ['blank_mean','blank_sd','threshold','blank_n'], 'thresholds'))
    pd.DataFrame(summaries).to_csv(OUT/'tables/table_standard_reproduction_gate.csv',index=False,encoding='utf-8-sig')
    status = {'gate_passed': all(x['mismatched']==0 for x in summaries), 'raw_pairs':len(pairs), 'successful_fields':len(data), 'summaries':summaries, 'scope':list(DATES), 'holm_family_recomputed':len(scope['dose_df']), 'holm_family_accepted':len(pd.read_csv(BASE/'table_threshold_sweep_concentration_summary.csv'))}
    (OUT/'gate_result.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2),flush=True)
    return 0 if status['gate_passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
