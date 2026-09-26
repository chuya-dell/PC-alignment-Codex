"""Independent direction/metric checks; no real benchmark thresholds tuned here."""
import unittest
import cv2
import numpy as np
from field_precision_benchmark import errors, truth_matrix

class Metrics(unittest.TestCase):
    def test_forward_renderer(self):
        a=np.zeros((80,100),np.float32); a[30,40]=1
        m=truth_matrix(a.shape,(7,-3,0,1))
        b=cv2.warpAffine(a,m,(100,80))
        self.assertEqual(np.unravel_index(b.argmax(),b.shape),(27,47))
        aligned=cv2.warpAffine(b,m,(100,80),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        np.testing.assert_array_equal(a,aligned)
    def test_center_separates_rotation(self):
        shape=(2044,2048); identity=np.eye(2,3)
        m=truth_matrix(shape,(.3,-.4,.2,1.005))
        d=errors(m,identity,shape)
        self.assertAlmostEqual(d['center_error_px'],.5)
        self.assertAlmostEqual(d['rotation_error_deg'],.2)
        self.assertAlmostEqual(d['scale_error'],.005)
    def test_identical(self):
        m=truth_matrix((2044,2048),(-27,20,-.2,.99))
        for value in errors(m,m,(2044,2048)).values(): self.assertAlmostEqual(value,0)

if __name__=='__main__': unittest.main()
