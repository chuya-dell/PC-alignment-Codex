import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import defect_masking as dm
import registration


def approval_row(**updates):
    row = {
        "candidate_id": "260826_SAM-9-1-A001", "date": "260826",
        "sample_id": "9", "position_id": "1", "source": "automatic",
        "decision": "approved", "pre_reviewed": "TRUE",
        "post_reviewed": "TRUE", "pre_defect_seen": "TRUE",
        "post_defect_seen": "FALSE",
        "mask_polygon_pre_json": json.dumps([[2, 2], [4, 2], [4, 4], [2, 4]]),
        "approved_at": "2026-09-09", "approved_by": "本人",
    }
    row.update(updates)
    return row


class DefectMaskingTests(unittest.TestCase):
    def test_sampling_rejects_any_3x3_mask_overlap(self):
        image = np.zeros((12, 12), np.float32)
        mask = np.zeros_like(image, dtype=bool)
        mask[5, 5] = True
        result = registration.sample_contrast(image, [[4, 4], [3, 3]], invalid_mask=mask)
        self.assertEqual(result.valid_sampling.tolist(), [False, True])


    def test_unapproved_candidates_are_never_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            pd.DataFrame([approval_row(decision="")]).to_csv(path, index=False)
            self.assertEqual(dm.load_approved_polygons(path, 260826, 9, 1), [])


    def test_approval_requires_both_images_reviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            pd.DataFrame([approval_row(post_reviewed="FALSE")]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "both images"):
                dm.load_approved_polygons(path, 260826, 9, 1)


    def test_approval_requires_auditable_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            pd.DataFrame([approval_row(approved_by="")]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "approver"):
                dm.load_approved_polygons(path, 260826, 9, 1)


    def test_rasterized_polygon_uses_xy_coordinates(self):
        mask = dm.rasterize_polygons((10, 12), [np.array([[2, 3], [5, 3], [5, 6], [2, 6]])])
        self.assertTrue(mask[4, 3])
        self.assertFalse(mask[3, 8])


    def test_large_connected_residual_is_candidate_but_single_pixel_is_not(self):
        height = width = 256
        yy, xx = np.indices((height, width))
        frequency = 1 / dm.PITCH_PX
        image = (20000 + 2500 * np.cos(2*np.pi*frequency*xx)
                 + 2500 * np.cos(2*np.pi*frequency*(.5*xx + .8660254*yy)))
        image[90:130, 110:160] += 18000
        image[30, 30] += 30000
        regions, _ = dm.residual_candidate_regions(image.astype(np.float32))
        self.assertTrue(any(r["area_px"] >= dm.CandidateSettings().min_area_px for r in regions))
        self.assertTrue(all(not (r["bbox_x"] <= 30 < r["bbox_x"] + r["bbox_w"]
                                 and r["bbox_y"] <= 30 < r["bbox_y"] + r["bbox_h"])
                            for r in regions))


if __name__ == "__main__":
    unittest.main()
