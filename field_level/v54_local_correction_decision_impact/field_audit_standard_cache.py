"""Audit fresh raw-image results against accepted caches, without reusing them."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/v54_local_correction_decision_impact_20261006'
BASE = Path('W:/GoogleDrive/chuya2816/5.解析結果_chu/20260928_digital_judgment_current_alignment/tables/cached_field_differences')


def main():
    records = []
    for path in sorted((OUT/'standard/tables/cached_field_differences').glob('*.npz')):
        row = {'field_key':path.stem, 'accepted_cache_exists':(BASE/path.name).exists()}
        if row['accepted_cache_exists']:
            with np.load(path,allow_pickle=False) as fresh, np.load(BASE/path.name,allow_pickle=False) as old:
                row.update(fresh_pillars=len(fresh['delta']), accepted_pillars=len(old['delta']))
                for key in ('delta','xy','ids'):
                    same_shape = fresh[key].shape==old[key].shape
                    row[key+'_identical'] = bool(same_shape and np.array_equal(fresh[key],old[key]))
                    row[key+'_max_abs_difference'] = float(np.max(np.abs(fresh[key]-old[key]))) if same_shape else np.nan
        records.append(row)
    table = pd.DataFrame(records)
    table.to_csv(OUT/'tables/table_fresh_vs_accepted_cache.csv',index=False,encoding='utf-8-sig')
    summary = {'fresh_caches':len(table), 'identical_delta_fields':int(table.delta_identical.sum()), 'identical_xy_fields':int(table.xy_identical.sum()), 'identical_id_fields':int(table.ids_identical.sum()), 'max_abs_delta_difference':float(table.delta_max_abs_difference.max())}
    (OUT/'cache_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
