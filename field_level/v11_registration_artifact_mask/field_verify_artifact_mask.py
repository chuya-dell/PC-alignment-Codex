"""Tests for conservative stain masking around registration landmarks.

Requirement summary (write this before implementation): the returned mask marks
excluded pixels, and it must never cover the three cross-scratch reference marks
(one vertical and two horizontal), including thin, faint, and interrupted marks.
The tests explicitly check every mark pixel, not only the apparent centerline.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from shared import registration as reg
from shared.image_qc import bright_band_mask, stain_artifact_mask


def image_with_grooves(*, faint=False, interrupted=False):
    """Synthetic image with known scratch marks and isolated stain-like blobs."""
    height, width = 768, 896
    rng = np.random.default_rng(381)
    image = 34000.0 + rng.normal(0, 45, (height, width)).astype(np.float32)
    vertical_x = 241
    horizontal_ys = (173, 521)
    groove_depth = 520 if faint else 1600
    image[:, vertical_x - 1:vertical_x + 2] -= groove_depth
    for y in horizontal_ys:
        image[y - 1:y + 2, :] -= groove_depth
    if interrupted:
        # Keep long but visibly broken marks, like a faint/partially obscured cross.
        for x in range(80, width, 175):
            image[horizontal_ys[0] - 1:horizontal_ys[0] + 2, x:x + 45] += groove_depth
        for y in range(65, height, 155):
            image[y:y + 38, vertical_x - 1:vertical_x + 2] += groove_depth
    yy, xx = np.ogrid[:height, :width]
    image[(xx - 630) ** 2 + (yy - 325) ** 2 <= 27**2] -= 2800
    image[(xx - 720) ** 2 + (yy - 620) ** 2 <= 18**2] += 2400
    return image, vertical_x, horizontal_ys


class StainArtifactMask(unittest.TestCase):
    def assert_grooves_protected(self, image, x, ys):
        mask = stain_artifact_mask(image)
        self.assertTrue(mask[250:500, x - 1:x + 2].size)
        self.assertFalse(mask[:, x - 1:x + 2].any(), "vertical scratch was masked")
        for y in ys:
            self.assertFalse(mask[y - 1:y + 2, :].any(), f"horizontal scratch y={y} was masked")
        self.assertTrue(mask[325 - 12:325 + 13, 630 - 12:630 + 13].any(),
                        "dark stain was not detected")
        self.assertTrue(mask[620 - 10:620 + 11, 720 - 10:720 + 11].any(),
                        "bright stain was not detected")

    def test_cross_scratches_are_preserved_and_stains_are_excluded(self):
        image, x, ys = image_with_grooves()
        self.assert_grooves_protected(image, x, ys)

    def test_faint_and_interrupted_cross_scratches_are_preserved(self):
        image, x, ys = image_with_grooves(faint=True, interrupted=True)
        self.assert_grooves_protected(image, x, ys)

    def test_registration_opt_in_unions_stain_mask_without_changing_default(self):
        image, _, _ = image_with_grooves()
        captured = []

        def estimate(_pre, _post, *, exclude_mask=None, **_kwargs):
            captured.append(np.asarray(exclude_mask, dtype=bool).copy())
            return np.eye(2, 3, dtype=np.float32)

        with patch.object(reg, "estimate_affine_orb_ransac", side_effect=estimate):
            reg.register_image_pair_affine(image, image, qc=False)
            reg.register_image_pair_affine(image, image, qc=False, mask_stains=True)
        np.testing.assert_array_equal(captured[0], bright_band_mask(image))
        expected = bright_band_mask(image) | stain_artifact_mask(image)
        np.testing.assert_array_equal(captured[1], expected)
        self.assertFalse(captured[1][:, 240:243].any(), "registration mask covered the vertical mark")
        self.assertFalse(captured[1][170:176, :].any(), "registration mask covered a horizontal mark")
        self.assertTrue(captured[1][315:336, 620:641].any(), "opt-in mask missed the dark stain")


if __name__ == "__main__":
    unittest.main()
