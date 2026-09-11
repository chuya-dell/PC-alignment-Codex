from __future__ import annotations

import argparse
import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from scipy.stats import spearmanr

# Prevent OpenCV's internal worker pool from intermittently terminating an ECC
# call on this Windows host.  This does not alter the optimizer or its inputs.
cv2.setNumThreads(1)


PITCH_ROW_PX = 6.38
MARGIN_PX = 30
MIN_PRE_MEAN_ADU = 32500.0
ECC_LEVELS = (0.25, 0.5, 1.0)
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)


@dataclass(frozen=True)
class SeriesSpec:
    assay: str
    root: Path
    concentrations_m: dict[int, float]


def load_tiff(path: Path) -> np.ndarray:
    raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_ANYDEPTH)
    if raw is None or raw.ndim != 2:
        raise ValueError(f"Unreadable grayscale TIFF: {path}")
    return raw.astype(np.float32) / 65535.0


def coarse_shift(pre: np.ndarray, post: np.ndarray) -> tuple[float, float]:
    rows, cols = pre.shape
    patch_size = min(1000, rows - 40, cols - 40)
    cy, cx = rows // 2, cols // 2
    half = patch_size // 2
    margin = 15
    patch = pre[cy - half : cy + half, cx - half : cx + half]
    search = post[
        cy - half - margin : cy + half + margin,
        cx - half - margin : cx + half + margin,
    ]
    response = cv2.matchTemplate(search, patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(response)
    return float(loc[0] - margin), float(loc[1] - margin)


def pyramid_ecc(
    pre: np.ndarray,
    post: np.ndarray,
    levels: tuple[float, ...] = ECC_LEVELS,
    initial_full: tuple[float, float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Affine ECC with coarse-template initialization and a true image pyramid.

    The returned matrix follows OpenCV findTransformECC convention and therefore
    must be applied to post with WARP_INVERSE_MAP.
    """
    if initial_full is None:
        initial_full = coarse_shift(pre, post)
    first = levels[0]
    warp = np.array(
        [[1.0, 0.0, initial_full[0] * first], [0.0, 1.0, initial_full[1] * first]],
        dtype=np.float32,
    )
    previous = first
    scores: list[float] = []
    for level in levels:
        if level != previous:
            warp[:, 2] *= level / previous
        template = cv2.resize(pre, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
        target = cv2.resize(post, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
        score, warp = cv2.findTransformECC(
            template,
            target,
            warp,
            cv2.MOTION_AFFINE,
            ECC_CRITERIA,
            None,
            5,
        )
        scores.append(float(score))
        previous = level
    return warp, scores


def align_post(post: np.ndarray, warp: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    return cv2.warpAffine(
        post,
        warp,
        (cols, rows),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )


def detect_pillars(pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = pre.shape
    centered = pre - float(np.mean(pre))
    fshift = np.fft.fftshift(np.fft.fft2(centered))
    cy, cx = rows // 2, cols // 2
    yy, xx = np.ogrid[-cy : rows - cy, -cx : cols - cx]
    radius = np.sqrt(xx * xx + yy * yy)
    ring_center = rows / PITCH_ROW_PX
    ring = (radius >= ring_center - 15) & (radius <= ring_center + 15)
    filtered = np.fft.ifft2(np.fft.ifftshift(fshift * ring)).real
    local_max = ndi.maximum_filter(filtered, size=5) == filtered
    y, x = np.where(local_max)
    keep = (
        (y > MARGIN_PX)
        & (y < rows - MARGIN_PX)
        & (x > MARGIN_PX)
        & (x < cols - MARGIN_PX)
    )
    return y[keep], x[keep]


def sum_3x3(img: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    padded = np.pad(img, 1, mode="constant", constant_values=np.nan)
    result = np.zeros(len(y), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result += padded[y + 1 + dy, x + 1 + dx]
    return result


def analyze_fov(task: tuple[str, str, int, int, str]) -> dict:
    assay, root_text, sample, fov, cache_text = task
    root = Path(root_text)
    pre_path = root / f"{sample}-{fov}-0.tif"
    post_path = root / f"{sample}-{fov}-1.tif"
    cache_path = Path(cache_text) / f"{assay}_S{sample}_F{fov}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            return {
                "assay": assay,
                "sample": sample,
                "fov": fov,
                "delta_pct": cached["delta_pct"],
                "ecc_scores": cached["ecc_scores"].tolist(),
                "warp": cached["warp"].tolist(),
                "correlation": float(cached["correlation"]),
                "post_timestamp": str(cached["post_timestamp"].item()),
                "n_detected": int(cached["n_detected"]),
                "n_valid": int(cached["n_valid"]),
                "cache_used": True,
            }

    pre = load_tiff(pre_path)
    post = load_tiff(post_path)
    if pre.shape != post.shape:
        raise ValueError(f"Shape mismatch: {pre_path} / {post_path}")
    warp, scores = pyramid_ecc(pre, post)
    aligned = align_post(post, warp, pre.shape)
    y, x = detect_pillars(pre)
    pre_sum = sum_3x3(pre, y, x)
    post_sum = sum_3x3(aligned, y, x)
    valid = (
        np.isfinite(pre_sum)
        & np.isfinite(post_sum)
        & (pre_sum > 0)
        & ((pre_sum / 9.0) * 65535.0 >= MIN_PRE_MEAN_ADU)
    )
    pre_valid = pre_sum[valid]
    post_valid = post_sum[valid]
    delta_pct = (pre_valid - post_valid) / pre_valid * 100.0
    correlation = float(np.corrcoef(pre_valid, post_valid)[0, 1])
    post_timestamp = datetime.fromtimestamp(post_path.stat().st_mtime).astimezone().isoformat()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        delta_pct=delta_pct.astype(np.float32),
        ecc_scores=np.asarray(scores),
        warp=warp,
        correlation=correlation,
        post_timestamp=np.asarray(post_timestamp),
        n_detected=len(y),
        n_valid=len(delta_pct),
    )
    return {
        "assay": assay,
        "sample": sample,
        "fov": fov,
        "delta_pct": delta_pct,
        "ecc_scores": scores,
        "warp": warp.tolist(),
        "correlation": correlation,
        "post_timestamp": post_timestamp,
        "n_detected": len(y),
        "n_valid": len(delta_pct),
        "cache_used": False,
    }


def load_defect_exclusions(path: Path) -> dict[str, set[tuple[int, int]]]:
    exclusions: dict[str, set[tuple[int, int]]] = {"SAM": set(), "DNA": set()}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["date"] != "260828" or row["modality"] not in exclusions:
                continue
            sample = int(row["sample_id"])
            if row["record_scope"] == "sample" or row["position_id"] == "ALL":
                exclusions[row["modality"]].update((sample, fov) for fov in range(1, 9))
            else:
                fov = int(row["position_id"])
                if 1 <= fov <= 8:
                    exclusions[row["modality"]].add((sample, fov))
    return exclusions


def concentration_label(value: float) -> str:
    if value == 0:
        return "Blank"
    labels = {1e-9: "1 nM", 1e-10: "100 pM", 1e-11: "10 pM", 1e-12: "1 pM", 1e-14: "10 fM", 1e-15: "1 fM"}
    return labels.get(value, f"{value:.3g} M")


def sample_metrics(
    spec: SeriesSpec,
    fov_rows: list[dict],
    threshold_pct: float,
) -> list[dict]:
    metrics: list[dict] = []
    for sample, concentration in spec.concentrations_m.items():
        rows = [r for r in fov_rows if r["sample"] == sample]
        pooled = np.concatenate([r["delta_pct"] for r in rows])
        q25, median, q75, q99, q999, q9999 = np.quantile(
            pooled, [0.25, 0.5, 0.75, 0.99, 0.999, 0.9999]
        )
        timestamps = sorted(datetime.fromisoformat(r["post_timestamp"]) for r in rows)
        mid_time = timestamps[len(timestamps) // 2]
        metrics.append(
            {
                "assay": spec.assay,
                "sample": sample,
                "concentration_m": concentration,
                "concentration_label": concentration_label(concentration),
                "n_clean_fovs": len(rows),
                "n_pillars": len(pooled),
                "median_pct": median,
                "q25_pct": q25,
                "q75_pct": q75,
                "iqr_pct": q75 - q25,
                "q99_pct": q99,
                "q99_9_pct": q999,
                "q99_99_pct": q9999,
                "nc_mean_plus_3sd_threshold_pct": threshold_pct,
                "threshold_exceedance_fraction": float(np.mean(pooled > threshold_pct)),
                "median_post_timestamp": mid_time.isoformat(),
                "pooled": pooled,
            }
        )
    for rank, row in enumerate(sorted(metrics, key=lambda r: r["median_post_timestamp"]), 1):
        row["processing_order_rank"] = rank
    return metrics


def plot_overlay(metrics: list[dict], output: Path, order: str) -> None:
    all_values = np.concatenate([r["pooled"] for r in metrics])
    lo, hi = np.quantile(all_values, [0.0001, 0.9999])
    pad = 0.03 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, 180)
    if order == "concentration":
        rows = sorted(metrics, key=lambda r: r["concentration_m"], reverse=True)
        title_tail = "concentration order"
        cmap = plt.get_cmap("turbo")
    else:
        rows = sorted(metrics, key=lambda r: r["processing_order_rank"])
        title_tail = "measured post-image order"
        cmap = plt.get_cmap("viridis")

    fig, (ax, sx) = plt.subplots(1, 2, figsize=(14.2, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = [cmap(i / max(1, len(rows) - 1)) for i in range(len(rows))]
    for color, row in zip(colors, rows):
        label = f"S{row['sample']} {row['concentration_label']}"
        if order == "processing":
            label = f"#{row['processing_order_rank']} S{row['sample']} {row['concentration_label']}"
        ax.hist(
            row["pooled"], bins=edges, density=True, histtype="step",
            linewidth=1.25, alpha=0.9, color=color, label=label,
        )
        sorted_values = np.sort(row["pooled"])
        survival = (len(sorted_values) - np.arange(len(sorted_values))) / len(sorted_values)
        keep = (sorted_values >= lo) & (sorted_values <= hi) & (survival >= 1e-4)
        sx.plot(sorted_values[keep], survival[keep], color=color, linewidth=1.3, label=label)

    threshold = metrics[0]["nc_mean_plus_3sd_threshold_pct"]
    for axis in (ax, sx):
        axis.axvline(threshold, color="#111827", linestyle="--", linewidth=1.2, label="N.C. mean + 3 SD")
        axis.set_xlim(edges[0], edges[-1])
        axis.grid(alpha=0.2)
        axis.set_xlabel("Delta I / I (%) = 100 x (pre - post) / pre")
    ax.set_ylabel("Probability density")
    ax.set_title("Pooled clean-pillar histogram")
    sx.set_ylabel("Upper-tail survival fraction")
    sx.set_yscale("log")
    sx.set_ylim(1e-4, 1)
    sx.set_title("Upper tail (log survival)")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="center right", bbox_to_anchor=(0.995, 0.5), frameon=False, fontsize=8.5)
    fig.suptitle(f"260828 p50 {metrics[0]['assay']}: Delta I / I distributions ({title_tail})", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.80, 0.95))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def spearman_rows(metrics: list[dict]) -> list[dict]:
    metric_names = ["median_pct", "iqr_pct", "q99_pct", "q99_9_pct", "q99_99_pct", "threshold_exceedance_fraction"]
    result: list[dict] = []
    nonblank = [r for r in metrics if r["concentration_m"] > 0]
    all_conc = sorted(metrics, key=lambda r: r["concentration_m"])
    for name in metric_names:
        for predictor, rows, x in (
            ("log10_concentration_nonblank", nonblank, [math.log10(r["concentration_m"]) for r in nonblank]),
            ("concentration_rank_including_blank", all_conc, list(range(len(all_conc)))),
            ("processing_order", metrics, [r["processing_order_rank"] for r in metrics]),
        ):
            rho, p = spearmanr(x, [r[name] for r in rows])
            result.append(
                {
                    "assay": metrics[0]["assay"],
                    "predictor": predictor,
                    "metric": name,
                    "n_samples": len(rows),
                    "spearman_rho": float(rho),
                    "two_sided_p_descriptive": float(p),
                }
            )
    return result


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def report_text(metrics: list[dict], trends: list[dict], fov_rows: list[dict]) -> str:
    lines = [
        "# 260828 p50 ピラー単位 Delta I / I 分布の濃度・処理順比較",
        "",
        "## 方法",
        "",
        "- 位置合わせは中央1000 pxテンプレートによる粗平行移動を初期値とした0.25x→0.5x→1.0xアフィンECC。postは`WARP_INVERSE_MAP`付き三次補間でpre座標へ写像した。",
        "- pre画像FFT一次リング（行間隔6.38 px）の局所極大をピラー座標とし、端30 pxとpre平均輝度32,500 ADU未満を除外した。",
        "- 一次欠陥記録にview/sample記録がある視野を丸ごと除外した。ピクセル座標がないため、自動欠陥候補を正式マスクとしては使用していない。",
        "- 各サンプルの残存視野に含まれるピラーを非集約でプールした。N.C.閾値は各assayのBlank Sample8をプールした平均+3×母標準偏差（ddof=0）。",
        "- Spearman相関のnはサンプル数。非Blankのlog10濃度、Blankを最低順位に含む濃度順位、実測post TIFF時刻による処理順位を併記した。p値は記述値であり、独立反復の有意性を表さない。",
        "",
        "## サンプル別結果",
        "",
        "| Assay | Sample | 濃度 | clean FOV | pillars | median % | IQR % | P99 % | P99.9 % | P99.99 % | >N.C.+3SD | 処理順 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['assay']} | {row['sample']} | {row['concentration_label']} | {row['n_clean_fovs']} | {row['n_pillars']:,} | "
            f"{row['median_pct']:.4f} | {row['iqr_pct']:.4f} | {row['q99_pct']:.4f} | {row['q99_9_pct']:.4f} | "
            f"{row['q99_99_pct']:.4f} | {100*row['threshold_exceedance_fraction']:.4f}% | {row['processing_order_rank']} |"
        )

    lines.extend(["", "## Spearman順位相関", "", "| Assay | predictor | metric | n | rho | p（記述） |", "|---|---|---|---:|---:|---:|"])
    for row in trends:
        lines.append(
            f"| {row['assay']} | {row['predictor']} | {row['metric']} | {row['n_samples']} | "
            f"{row['spearman_rho']:.3f} | {row['two_sided_p_descriptive']:.4f} |"
        )

    lines.extend(["", "## 位置合わせ・入力QC", ""])
    for assay in ("SAM", "DNA"):
        rows = [r for r in fov_rows if r["assay"] == assay]
        correlations = np.asarray([r["correlation"] for r in rows])
        scores = np.asarray([r["ecc_scores"][-1] for r in rows])
        lines.append(
            f"- {assay}: clean FOV {len(rows)}、ECC最終スコア範囲 {scores.min():.4f}–{scores.max():.4f}、"
            f"ピラーpre/post相関範囲 {correlations.min():.4f}–{correlations.max():.4f}。"
        )
    lines.extend(
        [
            "",
            "## 解釈上の制約",
            "",
            "- 同じサンプル内の多数のピラーは独立な生物学的反復ではない。裾指標はサンプル記述量であり、ピラー数をnとして有意性を主張しない。",
            "- 濃度と処理順は同一実験内で分離設計されていない。両者の相関が似る場合、濃度効果と時間ドリフトを識別できない。",
            "- TIFFのLastWriteTimeを実測時刻の代理にした。取得ソフトが保存時刻を書き換えた可能性は残る。",
            "- 欠陥記録のない視野を無条件にcleanとはみなせない。本解析は一次記録に明示された欠陥視野を除外したスナップショットで、完全な目視レビュー台帳が得られた場合は再計算が必要。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-root", type=Path, required=True)
    parser.add_argument("--dna-root", type=Path, required=True)
    parser.add_argument("--defect-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--assay", choices=("SAM", "DNA"), help="Process one assay only; useful for resumable cache creation.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        SeriesSpec("SAM", args.sam_root, {1: 1e-9, 2: 1e-10, 3: 1e-11, 4: 1e-12, 5: 1e-13, 6: 1e-14, 7: 1e-15, 8: 0.0}),
        SeriesSpec("DNA", args.dna_root, {1: 1e-9, 2: 1e-10, 9: 1e-11, 4: 1e-12, 10: 1e-14, 7: 1e-15, 8: 0.0}),
    ]
    if args.assay:
        specs = [spec for spec in specs if spec.assay == args.assay]
    exclusions = load_defect_exclusions(args.defect_records)
    tasks: list[tuple[str, str, int, int, str]] = []
    for spec in specs:
        for sample in spec.concentrations_m:
            for fov in range(1, 9):
                if (sample, fov) in exclusions[spec.assay]:
                    continue
                pre = spec.root / f"{sample}-{fov}-0.tif"
                post = spec.root / f"{sample}-{fov}-1.tif"
                if not pre.is_file() or not post.is_file():
                    raise FileNotFoundError(f"Missing required pair: {pre} / {post}")
                tasks.append((spec.assay, str(spec.root), sample, fov, str(args.cache_dir)))

    fov_rows: list[dict] = []
    if args.workers == 1:
        # Sequential mode is intentionally retained for forensic reruns of a
        # troublesome FOV; it also avoids OpenCV process-pool instability.
        for task in tasks:
            row = analyze_fov(task)
            fov_rows.append(row)
            print(f"{task[0]} S{task[2]} F{task[3]}: n={row['n_valid']} rho={row['correlation']:.4f}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(analyze_fov, task): task for task in tasks}
            for future in as_completed(futures):
                task = futures[future]
                row = future.result()
                fov_rows.append(row)
                print(f"{task[0]} S{task[2]} F{task[3]}: n={row['n_valid']} rho={row['correlation']:.4f}", flush=True)

    all_metrics: list[dict] = []
    all_trends: list[dict] = []
    for spec in specs:
        rows = [r for r in fov_rows if r["assay"] == spec.assay]
        blank = np.concatenate([r["delta_pct"] for r in rows if r["sample"] == 8])
        threshold = float(np.mean(blank) + 3.0 * np.std(blank, ddof=0))
        metrics = sample_metrics(spec, rows, threshold)
        all_metrics.extend(metrics)
        all_trends.extend(spearman_rows(metrics))
        plot_overlay(metrics, args.output_dir / f"260828_p50_{spec.assay.lower()}_histogram_concentration.png", "concentration")
        plot_overlay(metrics, args.output_dir / f"260828_p50_{spec.assay.lower()}_histogram_processing_order.png", "processing")

    metric_fields = [
        "assay", "sample", "concentration_m", "concentration_label", "n_clean_fovs", "n_pillars",
        "median_pct", "q25_pct", "q75_pct", "iqr_pct", "q99_pct", "q99_9_pct", "q99_99_pct",
        "nc_mean_plus_3sd_threshold_pct", "threshold_exceedance_fraction", "median_post_timestamp", "processing_order_rank",
    ]
    write_csv(args.output_dir / "260828_p50_pillar_distribution_metrics.csv", all_metrics, metric_fields)
    trend_fields = ["assay", "predictor", "metric", "n_samples", "spearman_rho", "two_sided_p_descriptive"]
    write_csv(args.output_dir / "260828_p50_spearman_trends.csv", all_trends, trend_fields)
    qc_fields = ["assay", "sample", "fov", "n_detected", "n_valid", "correlation", "post_timestamp", "ecc_scores", "warp", "cache_used"]
    write_csv(args.output_dir / "260828_p50_fov_qc.csv", fov_rows, qc_fields)
    (args.output_dir / "260828_p50_delta_i_over_i_report.md").write_text(
        report_text(all_metrics, all_trends, fov_rows), encoding="utf-8"
    )
    manifest = {
        "method": "coarse-template initialized 0.25x/0.5x/1.0 affine ECC with WARP_INVERSE_MAP",
        "delta_definition": "100 * (pre_3x3_sum - aligned_post_3x3_sum) / pre_3x3_sum",
        "defect_policy": "exclude whole FOVs listed for 260828; no unapproved image-derived pixel masks",
        "tasks": len(tasks),
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
