"""v70 step A-1: ledger of the existing 632 fields (all 'development' data)."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v70_lib as L
OUT = L.ROOT / 'data/results/v70_ledger_center_fit'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    inv = pd.read_csv(L.V57 / 'step0/fields_inventory.csv', dtype={'date': str, 'board': str})
    fixed = pd.read_csv(L.V57 / 'stepC/C1_outlier29_fields.csv', dtype={'date': str, 'board': str})
    qc = pd.read_csv(L.STD / 'tables/table_registration_field_qc.csv', encoding='utf-8-sig', dtype={'board': str, 'date': str})
    qc = qc[qc.qc_accepted.astype(str) == 'True']
    led = inv[['fid', 'date', 'board', 'field', 'condition', 'group', 'pre_path', 'post_path', 'pre_name', 'n']].copy()
    led = led.rename(columns={'n': 'n_pillars_stored'})
    led['modality'] = 'DNA'
    led['substrate_id'] = led.date + '_' + led.board.str.lstrip('0').replace('', '0')
    led['substrate_id_raw'] = led.date + '_' + led.board
    led['fixed29'] = led.fid.isin(set(fixed.fid))
    led['split'] = 'development'            # all existing 632 fields (request section 3-A)
    led['pre_exists'] = led.pre_path.map(os.path.isfile)
    led['post_exists'] = led.post_path.map(os.path.isfile)
    led['stored_cache_exists'] = led.fid.map(lambda f: (L.CACHE / f'{f}.npz').is_file())
    led['source_status'] = np.where(led.pre_exists & led.post_exists, 'raw_images_present',
                                    np.where(led.stored_cache_exists, 'stored_values_only', 'correspondence_unknown'))
    led['pre_size'] = led.pre_path.map(lambda p: os.path.getsize(p) if os.path.isfile(p) else -1)
    led['post_size'] = led.post_path.map(lambda p: os.path.getsize(p) if os.path.isfile(p) else -1)
    led['raw_root_folder'] = led.pre_path.map(lambda p: Path(p).parent.name)
    led['conc_M'] = pd.to_numeric(led.condition, errors='coerce')
    led = led.sort_values(['date', 'board', 'field']).reset_index(drop=True)
    led.to_csv(OUT / 'ledger.csv', index=False, encoding='utf-8-sig')
    # audit: board tokens that collapse when leading zeros are stripped (possible duplicate substrate names)
    g = led.groupby(['date', 'substrate_id']).board.nunique()
    dup = g[g > 1]
    summ = led.groupby('date').agg(fields=('fid', 'size'), substrates=('substrate_id', 'nunique'), fixed29=('fixed29', 'sum'),
                                   blank_fields=('group', lambda s: int((s == 'blank').sum()))).reset_index()
    summ.to_csv(OUT / 'ledger_by_date.csv', index=False, encoding='utf-8-sig')
    print(led.source_status.value_counts().to_dict(), 'fixed29', int(led.fixed29.sum()), 'fields', len(led))
    print('raw folder names:', sorted(led.raw_root_folder.unique()))
    print('board-token collisions (same substrate after lstrip(0)):', dup.to_dict())
    print(summ.to_string())


if __name__ == '__main__':
    main()
