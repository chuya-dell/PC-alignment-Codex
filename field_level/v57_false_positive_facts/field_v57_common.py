"""v57 common paths and loaders (read-only on stored v25 judgment outputs and raw images)."""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

STD = Path(r'W:\GoogleDrive\chuya2816\5.解析結果_chu\20260928_digital_judgment_current_alignment')
CACHE = STD / 'tables' / 'cached_field_differences'
RAWS = [Path(r'W:\GoogleDrive\chuya2816\5.生データD_chu'), Path(r'W:\GoogleDrive\remotefdtd\4.生データD_remo')]
REPO = Path(r'C:\Users\chuya\PC-alignment-localcorr')
OUT = REPO / 'data' / 'results' / 'v57_false_positive_facts_20261006'
SEED = 20261006
NS_MAIN = 3.0

DATE_DIR = {'260825': '260825_p50_dna', '260827': '260827_pp50_dna', '260828': '260828-p50-dna',
            '260829': '260829_p50_DNA', '260922': '260922-p50-dna', '260923': '260923-p50-dna',
            '260924': '260924-p50-dna', '260926': '260926-p50-dna', '260927': '260927-p50-dna'}

def cond_group(cond: str) -> str:
    """blank / analyte / mismatch / other (string forms as stored in the QC table)."""
    c = str(cond)
    if c in ('0', '0.0'): return 'blank'
    if c == 'mismatch': return 'mismatch'
    if c == 'blank_reference': return 'blank_reference'
    try:
        v = float(c)
        return 'analyte' if v > 0 else 'other'
    except ValueError:
        return 'other'

def cond_value(cond: str):
    try: return float(cond)
    except ValueError: return np.nan

def resolve_raw(date: str, name: str) -> Path | None:
    for r in RAWS:
        p = r / DATE_DIR[date] / name
        if p.exists(): return p
    return None

def field_list() -> pd.DataFrame:
    """632 accepted fields with condition, from the stored QC table."""
    q = pd.read_csv(STD / 'tables' / 'table_registration_field_qc.csv', encoding='utf-8-sig', dtype={'board': str, 'date': str})
    q = q[q.qc_accepted.astype(str) == 'True'].copy()
    q['field'] = q.field.astype(int)
    q['condition'] = q.condition.astype(str)
    q['group'] = q.condition.map(cond_group)
    q['conc_M'] = q.condition.map(cond_value)
    q['fid'] = q.date + '_' + q.board + '_' + q.field.astype(str)
    return q.reset_index(drop=True)

def load_cache(fid: str):
    with np.load(CACHE / f'{fid}.npz', allow_pickle=False) as z:
        return z['delta'], z['xy'], z['ids']
