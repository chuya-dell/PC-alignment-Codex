"""Verify that the faster/parallel variants reproduce the v9 outputs, and time them.

Baseline = v9 ``process_pair_cached`` run sequentially with the original kernels and the
default OpenCV thread count.  Each variant is compared with the baseline cache files:
status strings, integer counts, every float summary value, and the full per-field ``delta``
array (max absolute difference and bitwise equality are reported).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import field_run_phase2_production_parallel as runner  # noqa: E402


def build_manifest(raw_dir: Path, dataset: str, limit: int) -> pd.DataFrame:
    pairs = {}
    for f in raw_dir.glob("*.tif"):
        m = re.fullmatch(r"(\d+)-(\d+)-([01])\.tif", f.name)
        if m:
            pairs.setdefault((int(m[1]), int(m[2])), {})[int(m[3])] = f
    rows = []
    for (s, p), d in sorted(pairs.items()):
        if 0 in d and 1 in d:
            rows.append(dict(dataset=dataset, sample=s, position=p, concentration="n/a", series="speed_test",
                             is_blank=False, pre_path=str(d[0]), post_path=str(d[1])))
    return pd.DataFrame(rows[:limit])


def load_cache(cache_dir: Path, manifest: pd.DataFrame):
    out = {}
    for r in manifest.itertuples(index=False):
        key = f"{r.dataset}_{r.sample}_{r.position}"
        with np.load(cache_dir / f"{key}.npz", allow_pickle=False) as z:
            out[key] = (json.loads(str(z["summary_json"].item())), z["delta"].copy() if "delta" in z else None)
    return out


def compare(base, other):
    n = len(base)
    summary_equal = 0
    float_max = 0.0
    float_bitwise_all = True
    delta_equal = 0
    delta_max = 0.0
    delta_present = 0
    for key, (bs, bd) in base.items():
        os_, od = other[key]
        same = True
        for k, v in bs.items():
            w = os_.get(k)
            if isinstance(v, float) and isinstance(w, float):
                if not (v == w or (np.isnan(v) and np.isnan(w))):
                    same = False
                    float_bitwise_all = False
                    float_max = max(float_max, abs(v - w))
            elif v != w:
                same = False
        summary_equal += same
        if bd is not None:
            delta_present += 1
            if od is not None and bd.shape == od.shape:
                if np.array_equal(bd, od):
                    delta_equal += 1
                else:
                    delta_max = max(delta_max, float(np.nanmax(np.abs(bd - od))))
            else:
                delta_max = float("inf")
    return dict(n_fields=n, summaries_identical=summary_equal, summary_float_max_abs_diff=float_max,
                delta_arrays_present=delta_present, delta_arrays_bitwise_equal=delta_equal,
                delta_max_abs_diff_where_different=delta_max)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--dataset", default="speed_test")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--pitch", type=float, default=7.286)
    args = ap.parse_args()
    manifest = build_manifest(args.raw_dir, args.dataset, args.limit)
    print(f"{len(manifest)} pairs", flush=True)
    variants = [
        ("baseline_sequential_original", dict(workers=1, kernels="original", cv2_threads=None)),
        ("sequential_fast_kernels", dict(workers=1, kernels="fast", cv2_threads=None)),
        ("parallel_original", dict(workers=args.workers, kernels="original", cv2_threads=1)),
        ("parallel_fast", dict(workers=args.workers, kernels="fast", cv2_threads=1)),
    ]
    report = {"n_pairs": len(manifest), "workers": args.workers, "variants": {}}
    base = None
    for name, opt in variants:
        cache = args.work_dir / name
        if cache.exists():
            shutil.rmtree(cache)
        wall = runner.precompute_cache(manifest, cache, args.pitch, **opt)
        loaded = load_cache(cache, manifest)
        entry = {"wall_seconds": round(wall, 2), "seconds_per_pair_wall": round(wall / len(manifest), 3), **opt}
        if base is None:
            base = loaded
            entry["role"] = "baseline"
        else:
            entry.update(compare(base, loaded))
        report["variants"][name] = entry
        print(name, json.dumps(entry, ensure_ascii=False), flush=True)
    (args.work_dir / "speedup_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
