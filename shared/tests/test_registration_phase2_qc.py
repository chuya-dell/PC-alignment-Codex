"""Unit tests for the Phase-2 affine physical-plausibility gate."""
import math
import unittest
import warnings
from unittest.mock import patch
from types import SimpleNamespace

import cv2
import numpy as np

from shared.registration import (
    AffineTransformQCError,
    assess_affine_transform_qc,
    estimate_affine_orb_ransac,
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

    def test_spatial_search_tries_an_alternative_for_row_concentrated_inliers(self):
        raw = np.zeros(SHAPE, np.float32)
        points = [cv2.KeyPoint(float(x), 100.0, 1.0) for x in np.linspace(20, 1900, 18)]
        points += [cv2.KeyPoint(1900.0, 1900.0, 1.0), cv2.KeyPoint(1950.0, 1950.0, 1.0)]
        descriptors = np.arange(len(points), dtype=np.uint8).reshape(-1, 1)

        class Detector:
            def detectAndCompute(self, image, mask):
                return points, descriptors

        class Matcher:
            def knnMatch(self, desc_a, desc_b, k):
                return [(SimpleNamespace(queryIdx=i, trainIdx=i, distance=0.0),
                         SimpleNamespace(distance=100.0)) for i in range(len(points))]

        narrow = np.zeros((len(points), 1), np.uint8); narrow[:18] = 1
        broad = np.ones((len(points), 1), np.uint8)
        sparse = np.zeros((len(points), 1), np.uint8); sparse[:8] = 1
        expected = np.array([[1., 0., 12.], [0., 1., -7.]])
        with patch("shared.registration.cv2.ORB_create", return_value=Detector()), \
             patch("shared.registration.cv2.BFMatcher", return_value=Matcher()), \
             patch("shared.registration.cv2.estimateAffine2D", side_effect=[
                 (similarity(dx=1), narrow), (expected, broad), (similarity(dx=2), sparse),
             ]) as fit:
            result = estimate_affine_orb_ransac(raw, raw, exclude_mask=None)

        np.testing.assert_allclose(result, expected)
        self.assertEqual(fit.call_count, 3)

    def test_spatial_search_keeps_well_supported_primary_transform(self):
        raw = np.zeros(SHAPE, np.float32)
        points = [cv2.KeyPoint(float(x), float(y), 1.0)
                  for y in np.linspace(20, 1950, 5) for x in np.linspace(20, 1950, 5)]
        descriptors = np.arange(len(points), dtype=np.uint8).reshape(-1, 1)

        class Detector:
            def detectAndCompute(self, image, mask):
                return points, descriptors

        class Matcher:
            def knnMatch(self, desc_a, desc_b, k):
                return [(SimpleNamespace(queryIdx=i, trainIdx=i, distance=0.0),
                         SimpleNamespace(distance=100.0)) for i in range(len(points))]

        expected = similarity(dx=3, dy=-2)
        with patch("shared.registration.cv2.ORB_create", return_value=Detector()), \
             patch("shared.registration.cv2.BFMatcher", return_value=Matcher()), \
             patch("shared.registration.cv2.estimateAffine2D", return_value=(expected, np.ones((len(points), 1), np.uint8))) as fit:
            result = estimate_affine_orb_ransac(raw, raw, exclude_mask=None)

        np.testing.assert_allclose(result, expected)
        self.assertEqual(fit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
