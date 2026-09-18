"""Unit tests for the Phase-2 affine physical-plausibility gate."""
import math
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from shared.registration import (
    AffineTransformQCError,
    assess_affine_transform_qc,
    register_image_pair_affine,
)


SHAPE = (2048, 2048)


def similarity(dx=0.0, dy=0.0, rotation_deg=0.0, scale=1.0):
    """Build a matrix whose displacement at the image centre is (dx, dy)."""
    theta = math.radians(rotation_deg)
    linear = scale * np.array([[math.cos(theta), -math.sin(theta)],
                               [math.sin(theta), math.cos(theta)]])
    centre = np.array([(SHAPE[1] - 1) / 2, (SHAPE[0] - 1) / 2])
    translation = centre + np.array([dx, dy]) - linear @ centre
    return np.column_stack([linear, translation]).astype(np.float32)


class Phase2TransformQCTest(unittest.TestCase):
    def test_realistic_transform_is_accepted(self):
        result = assess_affine_transform_qc(similarity(-27, 20, -.2, 1.004), SHAPE)
        self.assertTrue(result["accepted"])

    def test_excess_translation_is_rejected(self):
        result = assess_affine_transform_qc(similarity(dx=126), SHAPE)
        self.assertFalse(result["accepted"])
        self.assertIn("dx_center", " ".join(result["reasons"]))

    def test_scale_rotation_and_reflection_are_rejected(self):
        for matrix in (similarity(scale=.90), similarity(rotation_deg=3.1),
                       np.array([[-1., 0., 0.], [0., 1., 0.]])):
            self.assertFalse(assess_affine_transform_qc(matrix, SHAPE)["accepted"])

    def test_nonfinite_matrix_is_rejected(self):
        self.assertFalse(assess_affine_transform_qc(np.full((2, 3), np.nan), SHAPE)["accepted"])

    def test_register_raises_before_returning_a_bad_transform(self):
        raw = np.zeros(SHAPE, np.uint16)
        with patch("shared.registration.estimate_affine_orb_ransac", return_value=similarity(dx=200)):
            with self.assertRaises(AffineTransformQCError):
                register_image_pair_affine(raw, raw, exclude_mask=None)

    def test_register_returns_diagnostics_for_an_accepted_transform(self):
        raw = np.zeros(SHAPE, np.uint16)
        expected = similarity(dx=10, dy=-5)
        with patch("shared.registration.estimate_affine_orb_ransac", return_value=expected):
            warp, qc = register_image_pair_affine(raw, raw, exclude_mask=None, return_qc=True)
        np.testing.assert_allclose(warp, expected)
        self.assertTrue(qc["accepted"])
        self.assertEqual(qc["mask_fraction"], 0.0)

    def test_large_mask_warns_but_valid_transform_remains_usable(self):
        raw = np.zeros(SHAPE, np.uint16)
        with patch("shared.registration.estimate_affine_orb_ransac", return_value=similarity()):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, qc = register_image_pair_affine(
                    raw, raw, exclude_mask=np.ones(SHAPE, bool), return_qc=True,
                )
        self.assertTrue(qc["accepted"])
        self.assertEqual(qc["mask_fraction"], 1.0)
        self.assertTrue(any("feature starvation" in str(item.message) for item in caught))


if __name__ == "__main__":
    unittest.main()
