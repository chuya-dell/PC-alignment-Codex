"""Parallel, result-preserving runner for the v9 Phase 2 production reanalysis.

The v9 entry point (``field_level/v9_phase2_production_reanalysis``) is not modified.  This
runner (1) computes the per-field cache files (``<dataset>_<sample>_<position>.npz``, the
format ``process_pair_cached`` of v9 reads and writes) for all pairs of the manifest in a
process pool, optionally using the faster kernels of ``field_fast_kernels.py``, and then
(2) calls v9's own ``main()`` unchanged, which reads those cache files and computes the
thresholds / Mann-Whitney tables exactly as before.

Usage (same arguments as v9, plus --workers / --kernels):
    python field_run_phase2_production_parallel.py --pair-manifest M.csv --cache-dir C --out-dir O
        [--pitch 7.286] [--workers 8] [--kernels original|fast] [--cv2-threads 1]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V9_PATH = ROOT / "field_level" / "v9_phase2_production_reanalysis" / "field_run_phase2_production_reanalysis.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_v9():
    spec = importlib.util.spec_from_file_location("v9_production", V9_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(module, kernels: str, cv2_threads: int | None):
    import cv2
    if cv2_threads is not None:
        cv2.setNumThreads(cv2_threads)
    if kernels == "fast":
        import field_fast_kernels as fk
        module.sample_grid_features = fk.fast_sample_grid_features
        module.lattice_from_fft = fk.fast_lattice_from_fft
    elif kernels != "original":
        raise ValueError("kernels must be 'original' or 'fast'")


_STATE = {}


def _init_worker(kernels, cv2_threads):
    module = load_v9()
    configure(module, kernels, cv2_threads)
    _STATE["module"] = module


def _work(args):
    row_dict, cache_dir, pitch = args
    module = _STATE["module"]
    t = time.perf_counter()
    summary, _ = module.process_pair_cached(SimpleNamespace(**row_dict), Path(cache_dir), pitch)
    return summary["status"], time.perf_counter() - t


def precompute_cache(manifest, cache_dir, pitch, workers, kernels, cv2_threads):
    rows = [(r._asdict(), str(cache_dir), pitch) for r in manifest.itertuples(index=False)]
    t0 = time.perf_counter()
    if workers <= 1:
        _init_worker(kernels, cv2_threads)
        results = [_work(r) for r in rows]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(kernels, cv2_threads)) as pool:
            results = list(pool.map(_work, rows, chunksize=1))
    wall = time.perf_counter() - t0
    print(f"precompute: {len(rows)} pairs, workers={workers}, kernels={kernels}, "
          f"cv2_threads={cv2_threads}, wall={wall:.1f}s, per-pair mean={sum(r[1] for r in results)/len(results):.2f}s",
          flush=True)
    return wall


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pitch", type=float, default=7.286)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kernels", choices=["original", "fast"], default="fast")
    parser.add_argument("--cv2-threads", type=int, default=1)
    args = parser.parse_args()
    manifest = pd.read_csv(args.pair_manifest)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    precompute_cache(manifest, args.cache_dir, args.pitch, args.workers, args.kernels, args.cv2_threads)
    # v9's own main(): reads the cache, computes tables.  Its argument parser reads sys.argv.
    module = load_v9()
    sys.argv = [str(V9_PATH), "--pair-manifest", str(args.pair_manifest),
                "--cache-dir", str(args.cache_dir), "--out-dir", str(args.out_dir),
                "--pitch", str(args.pitch)]
    module.main()


if __name__ == "__main__":
    main()
