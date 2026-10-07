"""Read-only forensic audit of v54 versus the original reference inputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import statsmodels

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import registration as reg
from shared.lattice_indexing import lattice_from_fft, grid_coordinates
from shared.v2_registration_precision.refinement import register_refined

OUT = ROOT / 'data/results/v63_local_correction_reproduction_audit'
BASE = Path('W:/GoogleDrive/chuya2816/5.解析結果_chu/20260928_digital_judgment_current_alignment')
V54 = Path('W:/GoogleDrive/chuya2816/5.解析結果_chu/v54_local_correction_decision_impact_20261006')
CHU = Path('W:/GoogleDrive/chuya2816/5.生データD_chu')
REMO_PARENT = Path('W:/GoogleDrive/remotefdtd')
DATES = ('260825', '260827', '260828', '260829')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_runner():
    source = BASE / 'scripts_and_config/field_run_digital_judgment.py'
    # Only import/path repair; numerical functions are unchanged.
    text = source.read_text(encoding='utf-8-sig')
    text = text.replace('from scipy.stats import mannwhitneyu, spearmanr, norm',
                        'from scipy.stats import mannwhitneyu, spearmanr, norm\nfrom statsmodels.stats.multitest import multipletests')
    text = text.replace("ROOT=Path(r'C:\\Users\\labuser\\Documents\\Codex\\2026-09-26\\gikt\\outputs\\PC-alignment-Codex')", f'ROOT=Path({str(ROOT)!r})')
    dest = OUT / 'standard_source_path_repaired.py'
    dest.write_text(text, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('standard_v63', dest)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.DATES = {k: v for k, v in mod.DATES.items() if k in DATES}
    mod.REMOTE = CHU
    return mod

def compare_arrays(new, old):
    result = {}
    for key in ('delta', 'xy', 'ids'):
        a, b = new[key], old[key]
        same_shape = a.shape == b.shape
        result[key + '_identical'] = same_shape and np.array_equal(a, b)
        result[key + '_max_abs'] = float(np.max(np.abs(a-b))) if same_shape else None
    return result

def compute_pair(pre_path, post_path):
    pre, post = reg.load_image_unicode(str(pre_path)), reg.load_image_unicode(str(post_path))
    coarse, qc = reg.register_image_pair_affine(pre, post, mask_stains=False, return_qc=True)
    matrix, refinement = register_refined(pre, post, stage='subpixel', initial=coarse, mask_stains=False)
    lattice = lattice_from_fft(pre, 7.286)
    ids, xy = grid_coordinates(lattice, pre.shape[1], pre.shape[0], margin=30)
    postxy = xy @ matrix[:, :2].T + matrix[:, 2]
    a, b = reg.sample_contrast(pre, xy), reg.sample_contrast(post, postxy)
    valid = a.valid_sampling.to_numpy() & b.valid_sampling.to_numpy()
    arrays = dict(delta=a.contrast.to_numpy()[valid]-b.contrast.to_numpy()[valid], xy=xy[valid], ids=ids[valid])
    return arrays, dict(coarse=coarse.tolist(), matrix=matrix.tolist(), qc=qc, refinement=refinement)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun', type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'tables').mkdir(exist_ok=True)
    (OUT/'reference_snapshots').mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    run = load_runner()
    pairs, inventory, _ = run.build_inventory()
    remo = next(p for p in REMO_PARENT.iterdir() if p.is_dir() and p.name == '4.生データD_remo')
    actual_dirs = {p.name: p for p in remo.iterdir() if p.is_dir()}
    records = []
    for pair in pairs:
        date, board, field = pair['date'], pair['board'], pair['fov']
        for phase in ('pre', 'post'):
            chu = pair[phase]
            folder = actual_dirs[run.DATES[date][0]]
            # Locate by listing the verified directory, never fabricate a path.
            matches = [p for p in folder.iterdir() if p.name == chu.name]
            if len(matches) != 1:
                raise RuntimeError(f'Missing/ambiguous reference image: {folder}, {chu.name}')
            remote = matches[0]
            hc, hr = digest(chu), digest(remote)
            records.append(dict(date=date, board=str(board), field=field, phase=phase,
                                chu_path=str(chu), reference_path=str(remote),
                                chu_sha256=hc, reference_sha256=hr, bytes_equal=hc == hr))
    images = pd.DataFrame(records)
    images.to_csv(OUT/'tables/table_input_hash_comparison.csv', index=False, encoding='utf-8-sig')
    audit = pd.read_csv(V54/'tables/table_fresh_vs_accepted_cache.csv')
    hashes = images.groupby(['date','board','field']).bytes_equal.all()
    audit['images_identical'] = [bool(hashes.loc[(s.split('_')[0], s.split('_')[1], int(s.split('_')[2]))]) for s in audit.field_key]
    audit.to_csv(OUT/'tables/table_input_vs_cache_match.csv', index=False, encoding='utf-8-sig')
    sources = []
    for rel in ['scripts_and_config/field_run_digital_judgment.py', 'scripts_and_config/field_recompute_from_cached_differences.py',
                'scripts_and_config/threshold_sweep_config.json', 'tables/table_registration_field_qc.csv',
                'tables/table_difference_cache_manifest_by_field.csv', 'tables/table_threshold_sweep_thresholds_by_date.csv',
                'tables/table_threshold_sweep_concentration_summary.csv', 'tables/table_threshold_sweep_excess_rate_by_field.csv',
                'tables/table_spearman_field_level_by_date_threshold.csv']:
        src = BASE/rel
        dst = OUT/'reference_snapshots'/src.name
        shutil.copyfile(src, dst)
        sources.append(dict(source=str(src), snapshot=str(dst), sha256=digest(src), newness_status='unknown', relationship='reference for numerical audit'))
    for rel in ['shared/registration.py','shared/lattice_indexing.py','shared/image_qc.py','shared/v2_registration_precision/refinement.py']:
        sources.append(dict(source=str(ROOT/rel), sha256=digest(ROOT/rel), v54_dependency_sha256=digest(Path('C:/Users/chuya/PC-alignment-localcorr')/rel)))
    summary = dict(input_pairs=len(pairs), images=len(images), identical_images=int(images.bytes_equal.sum()),
                   different_images=int((~images.bytes_equal).sum()),
                   input_cache_crosstab=audit.groupby(['images_identical','delta_identical','xy_identical']).size().reset_index(name='fields').to_dict('records'),
                   environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                    numpy=np.__version__, scipy=scipy.__version__, cv2=cv2.__version__, pandas=pd.__version__, statsmodels=statsmodels.__version__, threads=cv2.getNumThreads()),
                   sources=sources)
    (OUT/'input_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sources','environment')},ensure_ascii=False), flush=True)
    reruns = []
    # Include every difference category plus an unchanged field as a control.
    selected = []
    for _, group in audit.groupby(['images_identical','delta_identical','xy_identical']):
        selected.extend(group.field_key.head(args.rerun).tolist())
    by_key = {f"{p['date']}_{p['board']}_{p['fov']}": p for p in pairs}
    image_map = images.set_index(['date','board','field','phase'])
    for key in selected:
        p = by_key[key]
        index = (p['date'],str(p['board']),p['fov'])
        with np.load(BASE/'tables/cached_field_differences'/f'{key}.npz', allow_pickle=False) as z:
            accepted = {k:z[k].copy() for k in z.files}
        with np.load(V54/'standard/tables/cached_field_differences'/f'{key}.npz', allow_pickle=False) as z:
            fresh = {k:z[k].copy() for k in z.files}
        for input_name, col in [('chu','chu_path'),('reference','reference_path')]:
            before = [digest(image_map.loc[(*index,phase),col]) for phase in ('pre','post')]
            arrays, info = compute_pair(*[image_map.loc[(*index,phase),col] for phase in ('pre','post')])
            after = [digest(image_map.loc[(*index,phase),col]) for phase in ('pre','post')]
            row = dict(field_key=key, input=input_name, source_unchanged=before==after, **compare_arrays(arrays,accepted),
                       v54_delta_identical=np.array_equal(arrays['delta'],fresh['delta']), **info)
            reruns.append(row)
            (OUT/'rerun_probe.json').write_text(json.dumps(reruns,ensure_ascii=False,indent=2,default=str), encoding='utf-8')
            pd.DataFrame([{k:v for k,v in r.items() if k not in ('coarse','matrix','qc','refinement')} for r in reruns]).to_csv(OUT/'tables/table_rerun_probe.csv',index=False,encoding='utf-8-sig')
            print(key, input_name, {k:v for k,v in row.items() if k.endswith('identical')},flush=True)

if __name__ == '__main__':
    main()
