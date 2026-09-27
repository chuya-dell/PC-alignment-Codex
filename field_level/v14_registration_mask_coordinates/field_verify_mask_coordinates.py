"""Verify symmetric stain exclusions for pre-only and post-only contamination."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared import registration as reg


class PairMaskCoordinates(unittest.TestCase):
    def test_post_only_and_pre_only_stains_are_unioned_in_common_frame(self):
        shape = (64, 64)
        pre_stain = np.zeros(shape, bool)
        post_stain = np.zeros(shape, bool)
        pre_stain[25, 30] = True
        # pre -> post translates +10 px: this post-only stain corresponds to x=10 in pre.
        post_stain[25, 20] = True
        matrix = np.array([[1, 0, 10], [0, 1, 0]], np.float32)

        common_pre, common_post = reg.align_pair_exclusion_masks(pre_stain, post_stain, matrix)

        self.assertTrue(common_pre[25, 10], "post-only contamination was not mapped to pre")
        self.assertTrue(common_pre[25, 30], "pre-only contamination was not retained")
        self.assertFalse(common_pre[25, 20], "post-native coordinate leaked into common pre frame")
        self.assertTrue(common_post[25, 20], "post-only contamination was not retained in post")
        self.assertTrue(common_post[25, 40], "pre-only contamination was not mapped to post")
        self.assertFalse(common_post[25, 30], "pre-native coordinate leaked into common post frame")

    def test_registration_refits_with_distinct_frame_masks(self):
        shape = (64, 64)
        pre = np.zeros(shape, np.float32)
        post = np.ones(shape, np.float32)
        pre_stain = np.zeros(shape, bool)
        post_stain = np.zeros(shape, bool)
        pre_stain[25, 30] = True
        post_stain[25, 20] = True
        matrix = np.array([[1, 0, 10], [0, 1, 0]], np.float32)
        seen = []

        def detector(raw):
            return pre_stain.copy() if raw is pre else post_stain.copy()

        def estimate(_pre, _post, *, exclude_mask=None, exclude_mask_post=None, **_kwargs):
            seen.append((exclude_mask, exclude_mask_post))
            return matrix.copy()

        with patch("shared.image_qc.stain_artifact_mask", side_effect=detector), \
             patch.object(reg, "estimate_affine_orb_ransac", side_effect=estimate):
            reg.register_image_pair_affine(pre, post, exclude_mask=None, qc=False, mask_stains=True)

        self.assertEqual(len(seen), 2)
        self.assertIsNone(seen[0][0], "coarse initialization should not use stain masks")
        self.assertIsNone(seen[0][1])
        expected_pre, expected_post = reg.align_pair_exclusion_masks(pre_stain, post_stain, matrix)
        np.testing.assert_array_equal(seen[1][0], expected_pre)
        np.testing.assert_array_equal(seen[1][1], expected_post)


if __name__ == "__main__":
    unittest.main()
