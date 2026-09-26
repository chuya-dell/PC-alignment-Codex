"""Independent, provisional seven-pillar aggregation of one existing field table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ASSUMPTION = "PROVISIONAL / owner unconfirmed: one centre + six nearest hex neighbours; overlap allowed."
RESULTS = Path(__file__).resolve().parents[2] / "data" / "results"


def aggregate_groups(frame, metrics, *, pitch, tolerance=0.15, x="x", y="y",
                     id_column=None, valid_column="valid_sampling"):
    """Return new group rows; never mutate the single-field input.

    Coordinates and pitch must share units. Only the first shell, within
    pitch*(1 +/- tolerance), is accepted. Edge/hole/ambiguous centres are
    EXCLUDED, never padded with second-shell pillars. Six neighbours must also
    occupy six approximately 60-degree sectors (gaps 40..80 degrees).
    Population standard deviation (ddof=0) describes the seven members.
    Nonfinite or invalid samples propagate NaN, rather than silently averaging
    fewer than seven values. Counts remain available for auditing.
    """
    metrics = list(metrics)
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Provide distinct metric columns.")
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("pitch must be finite and positive.")
    if not np.isfinite(tolerance) or not 0 <= tolerance < 0.3:
        raise ValueError("tolerance must be in [0, 0.3).")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not supported.")
    coords = frame[[x, y]].to_numpy(dtype=float, copy=True)
    if not np.isfinite(coords).all() or len(np.unique(coords, axis=0)) != len(coords):
        raise ValueError("Coordinates must be finite and unique within one field.")
    ids = list(range(len(frame))) if id_column is None else frame[id_column].tolist()
    if pd.Series(ids, dtype=object).isna().any() or len(set(map(str, ids))) != len(ids):
        raise ValueError("Pillar identifiers must be non-null and unique.")
    values = frame[metrics].to_numpy(dtype=float, copy=True)
    if valid_column is not None and valid_column in frame:
        # Preserve the existing sampler's validity decision; do not infer approval.
        valid = frame[valid_column].astype(str).str.lower().map({"true": True, "false": False})
        if valid.isna().any():
            raise ValueError("Validity flags must be explicit True/False values.")
        values[~valid.to_numpy(dtype=bool)] = np.nan
    columns = ["centre_id", "centre_row", "centre_x", "centre_y", "member_ids",
               "member_rows", "member_count", "definition_status", "pitch", "tolerance", "std_ddof"]
    for metric in metrics:
        columns.extend([f"{metric}_count", f"{metric}_mean", f"{metric}_std"])
    tree = cKDTree(coords)
    rows = []
    for centre, point in enumerate(coords):
        candidates = tree.query_ball_point(point, pitch * (1 + tolerance) + pitch * 1e-10)
        neighbours = [i for i in candidates if i != centre]
        # Any extra near point is ambiguous, even if it lies below the shell.
        if len(neighbours) != 6:
            continue
        delta = coords[neighbours] - point
        if np.any(np.linalg.norm(delta, axis=1) < pitch * (1 - tolerance) - pitch * 1e-10):
            continue
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        order = np.argsort(angles)
        gaps = np.diff(np.r_[angles[order], angles[order][0] + 2 * np.pi])
        if np.any(np.abs(gaps - np.pi / 3) > np.pi / 9 + 1e-10):
            continue
        members = [centre] + [neighbours[i] for i in order]
        row = dict(centre_id=ids[centre], centre_row=centre,
                   centre_x=point[0], centre_y=point[1],
                   member_ids=json.dumps([str(ids[i]) for i in members]),
                   member_rows=json.dumps(members), member_count=7,
                   definition_status="provisional_owner_unconfirmed",
                   pitch=pitch, tolerance=tolerance, std_ddof=0)
        for j, metric in enumerate(metrics):
            samples = values[members, j]
            count = int(np.isfinite(samples).sum())
            row[f"{metric}_count"] = count
            row[f"{metric}_mean"] = float(np.mean(samples)) if count == 7 else np.nan
            row[f"{metric}_std"] = float(np.std(samples, ddof=0)) if count == 7 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def run(input_path, output_path, metrics, **options):
    """Read an existing single-field CSV and exclusively create a separate file."""
    source, destination = Path(input_path).resolve(), Path(output_path).resolve()
    if source == destination:
        raise ValueError("Group output must differ from the individual input.")
    if not destination.is_relative_to(RESULTS.resolve()):
        raise ValueError("Group output must be under this clone's data/results/.")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite any existing output.")
    result = aggregate_groups(pd.read_csv(source), metrics, **options)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        result.to_csv(handle, index=False)
    print(ASSUMPTION)
    print(f"Complete groups: {len(result)}; output: {destination}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__ + " " + ASSUMPTION)
    parser.add_argument("input", type=Path, help="Existing individual result CSV for ONE field")
    parser.add_argument("output", type=Path, help="New CSV under data/results/")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--pitch", type=float, required=True, help="Known first-shell distance in coordinate units")
    parser.add_argument("--tolerance", type=float, default=0.15)
    parser.add_argument("--x", default="x")
    parser.add_argument("--y", default="y")
    parser.add_argument("--id-column")
    parser.add_argument("--valid-column", default="valid_sampling")
    args = vars(parser.parse_args())
    source, output, metrics = args.pop("input"), args.pop("output"), args.pop("metrics")
    run(source, output, metrics, **args)


if __name__ == "__main__":
    main()
