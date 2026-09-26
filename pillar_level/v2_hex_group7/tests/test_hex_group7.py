"""R-04: PROVISIONAL, owner unconfirmed: centre + six nearest neighbours.

Groups may overlap. This test suite does not constitute owner confirmation.
Run with pytest, or unittest discovery when pytest is unavailable.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
MODULE = ROOT / "pillar_level/v2_hex_group7/pillar_hex_group7.py"
spec = importlib.util.spec_from_file_location("pillar_hex_group7", MODULE)
group7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(group7)

from shared.lattice_indexing import HexLattice, grid_coordinates, hex_basis
from shared.theoretical_grid_evaluation import sample_grid_features


def field(radius=2, angle=0.0):
    indices = [(m, n) for m in range(-radius, radius + 1)
               for n in range(-radius, radius + 1) if abs(m + n) <= radius]
    coords = np.asarray(indices) @ hex_basis(10, angle).T + [50, 50]
    return pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1],
                         "pillar_id": [f"{m}:{n}" for m, n in indices],
                         "intensity": np.arange(len(indices), dtype=float) + 1})


class HexGroup7Tests(unittest.TestCase):
    def test_r01_interior_boundary_overlap_rotation(self):
        for angle in (0, 0.37, 1.1):
            with self.subTest(angle=angle):
                frame = field(angle=angle)
                out = group7.aggregate_groups(frame, ["intensity"], pitch=10, id_column="pillar_id")
                self.assertEqual(len(out), 7)  # radius-two hex: 19 sites, 7 interior centres
                self.assertTrue((out.member_count == 7).all())
                for members in out.member_ids:
                    self.assertEqual(len(set(json.loads(members))), 7)
                self.assertEqual(out.member_ids.map(lambda s: "0:0" in json.loads(s)).sum(), 7)

    def test_r02_hand_calculated_field(self):
        frame = field(radius=1)
        frame["contrast"] = 2 * frame.intensity - 3
        out = group7.aggregate_groups(frame, ["intensity", "contrast"], pitch=10)
        self.assertEqual(len(out), 1)
        # One synthetic field, all seven members: 1..7, sum=28, mean=4.
        # Squared deviations: 9+4+1+0+1+4+9=28, population std=sqrt(28/7)=2.
        # Contrast = 2*intensity-3: mean=5, population std=4.
        self.assertEqual(out.intensity_mean.iloc[0], 4)
        self.assertEqual(out.intensity_std.iloc[0], 2)
        self.assertEqual(out.contrast_mean.iloc[0], 5)
        self.assertEqual(out.contrast_std.iloc[0], 4)

    def test_hole_never_replaced_by_second_shell(self):
        frame = field()
        frame = frame[frame.pillar_id != "1:0"]
        out = group7.aggregate_groups(frame, ["intensity"], pitch=10, id_column="pillar_id")
        self.assertNotIn("0:0", out.centre_id.tolist())

    def test_empty_and_tiny_fields_have_stable_schema(self):
        for size in (0, 1, 6):
            out = group7.aggregate_groups(field().iloc[:size], ["intensity"], pitch=10)
            self.assertTrue(out.empty)
            self.assertIn("intensity_std", out.columns)

    def test_missing_and_invalid_metrics_do_not_shrink_group(self):
        for invalid in (np.nan, np.inf):
            frame = field(1)
            frame.loc[0, "intensity"] = invalid
            row = group7.aggregate_groups(frame, ["intensity"], pitch=10).iloc[0]
            self.assertEqual(row.intensity_count, 6)
            self.assertTrue(np.isnan(row.intensity_mean))
            self.assertTrue(np.isnan(row.intensity_std))
        frame = field(1)
        frame["valid_sampling"] = True
        frame.loc[0, "valid_sampling"] = False
        row = group7.aggregate_groups(frame, ["intensity"], pitch=10).iloc[0]
        self.assertEqual(row.intensity_count, 6)
        self.assertTrue(np.isnan(row.intensity_mean))

    def test_bad_inputs_rejected(self):
        for pitch in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                group7.aggregate_groups(field(), ["intensity"], pitch=pitch)
        for tolerance in (-1, 0.3, np.inf):
            with self.assertRaises(ValueError):
                group7.aggregate_groups(field(), ["intensity"], pitch=10, tolerance=tolerance)
        for frame in (pd.concat([field(), field().iloc[:1]]), field().assign(x=np.nan),
                      field().assign(pillar_id="duplicate"), field().assign(valid_sampling="unknown")):
            with self.assertRaises(ValueError):
                group7.aggregate_groups(frame, ["intensity"], pitch=10, id_column="pillar_id")

    def test_ambiguous_and_nonhex_neighbours_excluded(self):
        frame = field(1)
        extra = pd.DataFrame({"x": [51], "y": [50], "intensity": [1]})
        out = group7.aggregate_groups(pd.concat([frame, extra]), ["intensity"], pitch=10)
        self.assertTrue(out.empty)
        angles = np.deg2rad([0, 10, 20, 30, 40, 50])
        coords = np.vstack(([0, 0], np.column_stack((10*np.cos(angles), 10*np.sin(angles)))))
        frame = pd.DataFrame(coords, columns=["x", "y"]).assign(intensity=1)
        self.assertTrue(group7.aggregate_groups(frame, ["intensity"], pitch=10).empty)

    def test_shuffled_rows_preserve_identified_groups(self):
        frame = field()
        first = group7.aggregate_groups(frame, ["intensity"], pitch=10, id_column="pillar_id")
        second = group7.aggregate_groups(frame.sample(frac=1, random_state=4), ["intensity"],
                                         pitch=10, id_column="pillar_id")
        for col in ("member_ids", "intensity_mean", "intensity_std"):
            pd.testing.assert_series_equal(first.set_index("centre_id")[col].sort_index(),
                                           second.set_index("centre_id")[col].sort_index(), check_exact=True)

    def test_r03_existing_analysis_exact_before_after_and_output_protection(self):
        # Execute the unchanged real lattice and individual feature sampler twice
        # on identical synthetic image input, with aggregation in between.
        yy, xx = np.indices((128, 128))
        image = (1000 + (xx * 17 + yy * 13) % 5000).astype(np.uint16)
        lattice = HexLattice(np.array([60., 60.]), hex_basis(10, .2), .2, 10)
        indices_before, coords_before = grid_coordinates(lattice, 128, 128, margin=10)
        before = sample_grid_features(image, lattice, margin=10)
        snapshot = before.copy(deep=True)
        source_files = list((ROOT / "shared").glob("*.py"))
        code_before = {p: p.read_bytes() for p in source_files}
        results = ROOT / "data/results"
        results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temporary:
            folder = Path(temporary)
            source, destination = folder / "individual.csv", folder / "groups.csv"
            before.to_csv(source, index=False)
            original_bytes = source.read_bytes()
            out = group7.run(source, destination, ["ints", "contrast"], pitch=10)
            self.assertGreater(len(out), 0)
            self.assertEqual(source.read_bytes(), original_bytes)
            saved_groups = destination.read_bytes()
            with self.assertRaises(FileExistsError):
                group7.run(source, destination, ["ints"], pitch=10)
            with self.assertRaises(ValueError):
                group7.run(source, source, ["ints"], pitch=10)
            with self.assertRaises(ValueError):
                group7.run(source, ROOT / "forbidden.csv", ["ints"], pitch=10)
            self.assertEqual(destination.read_bytes(), saved_groups)
            group7.aggregate_groups(before, ["ints", "contrast"], pitch=10)
            pd.testing.assert_frame_equal(before, snapshot, check_exact=True)
            after = sample_grid_features(image, lattice, margin=10)
            pd.testing.assert_frame_equal(before, after, check_exact=True)
            rerun = folder / "individual_rerun.csv"
            after.to_csv(rerun, index=False)
            self.assertEqual(rerun.read_bytes(), original_bytes)
        indices_after, coords_after = grid_coordinates(lattice, 128, 128, margin=10)
        np.testing.assert_array_equal(indices_before, indices_after)
        np.testing.assert_array_equal(coords_before, coords_after)
        self.assertEqual(code_before, {p: p.read_bytes() for p in source_files})

    def test_command_line(self):
        results = ROOT / "data/results"
        results.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=results) as temporary:
            source, output = Path(temporary) / "individual.csv", Path(temporary) / "groups.csv"
            field(1).to_csv(source, index=False)
            completed = subprocess.run([sys.executable, str(MODULE), str(source), str(output),
                                        "--pitch", "10", "--metrics", "intensity"],
                                       capture_output=True, text=True, check=True)
            self.assertIn("owner unconfirmed", completed.stdout)
            self.assertEqual(len(pd.read_csv(output)), 1)
