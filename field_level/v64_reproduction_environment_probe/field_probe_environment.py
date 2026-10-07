"""Read-only thread/optimization probe; never changes the reproduction gate."""
from pathlib import Path
import hashlib
import json
import sys
import platform
import time
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
import cv2
import numpy as np
import pandas as pd
import scipy
from shared import registration as reg
from shared.lattice_indexing import lattice_from_fft, grid_coordinates
from shared.v2_registration_precision.refinement import register_refined

OUT = ROOT / 'data/results/v64_reproduction_environment_probe'
BASE = Path('W:/GoogleDrive/chuya2816/5.解析結果_chu/20260928_digital_judgment_current_alignment')
V54 = Path('W:/GoogleDrive/chuya2816/5.解析結果_chu/v54_local_correction_decision_impact_20261006')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('audit63', ROOT/'field_level/v63_local_correction_reproduction_audit/field_audit_reproduction.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    run = audit.load_runner()
    pairs, _, _ = run.build_inventory()
    selected = {f"{p['date']}_{p['board']}_{p['fov']}": p for p in pairs}
    default_threads = cv2.getNumThreads()
    rows = []
    for key in ('260825_0_1', '260825_0_2', '260828_1_1'):
        p = selected[key]
        before = [sha(p[k]) for k in ('pre', 'post')]
        pre, post = [reg.load_image_unicode(str(p[k])) for k in ('pre', 'post')]
        with np.load(BASE/'tables/cached_field_differences'/f'{key}.npz') as z:
            accepted = {k:z[k].copy() for k in z.files}
        with np.load(V54/'standard/tables/cached_field_differences'/f'{key}.npz') as z:
            fresh = {k:z[k].copy() for k in z.files}
        for threads, optimized in ((1, True), (default_threads, True), (1, False)):
            cv2.setNumThreads(threads)
            cv2.setUseOptimized(optimized)
            started = time.monotonic()
            coarse, qc = reg.register_image_pair_affine(pre, post, mask_stains=False, return_qc=True)
            matrix, refinement = register_refined(pre, post, stage='subpixel', initial=coarse, mask_stains=False)
            lattice = lattice_from_fft(pre, 7.286)
            ids, xy = grid_coordinates(lattice, pre.shape[1], pre.shape[0], margin=30)
            postxy = xy @ matrix[:, :2].T + matrix[:, 2]
            a, b = reg.sample_contrast(pre, xy), reg.sample_contrast(post, postxy)
            valid = a.valid_sampling.to_numpy() & b.valid_sampling.to_numpy()
            arrays = dict(delta=a.contrast.to_numpy()[valid]-b.contrast.to_numpy()[valid], xy=xy[valid], ids=ids[valid])
            row = dict(field_key=key, threads=cv2.getNumThreads(), optimized=cv2.useOptimized(), seconds=time.monotonic()-started,
                       **audit.compare_arrays(arrays, accepted),
                       v54_delta_identical=np.array_equal(arrays['delta'], fresh['delta']),
                       coarse=coarse.tolist(), matrix=matrix.tolist(), refinement=refinement, qc=qc)
            rows.append(row)
            (OUT/'thread_probe.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            pd.DataFrame([{k:v for k,v in r.items() if k not in ('coarse','matrix','refinement','qc')} for r in rows]).to_csv(OUT/'table_thread_probe.csv',index=False,encoding='utf-8-sig')
            print(key, threads, optimized, row['delta_identical'], row['xy_identical'], row['delta_max_abs'], flush=True)
        assert before == [sha(p[k]) for k in ('pre', 'post')], 'Input changed during read-only probe'
    deps = ('shared/registration.py','shared/lattice_indexing.py','shared/image_qc.py','shared/v2_registration_precision/refinement.py')
    meta = dict(python=sys.version, platform=platform.platform(), executable=sys.executable, numpy=np.__version__, scipy=scipy.__version__, cv2=cv2.__version__,
                historical_environment_record='docs/environment-requirements-20260926.txt (preparation record; cache-generation environment is not proven)',
                dependencies=[dict(path=r, current_sha256=sha(ROOT/r), v54_sha256=sha(Path('C:/Users/chuya/PC-alignment-localcorr')/r)) for r in deps],
                gate_passed=False, scope='diagnostic only; three fields do not establish all-field reproduction')
    (OUT/'environment.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
