"""Geometry and failure-path tests on deterministic nonperiodic texture."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from shared.v2_registration_precision.refinement import subpixel_refine,permissible
from field_precision_benchmark import errors,truth_matrix

class Subpixel(unittest.TestCase):
    def test_recovers_injected_subpixel_error(self):
        cv2.setRNGSeed(17); cv2.setNumThreads(1)
        rng=np.random.default_rng(17)
        pre=cv2.GaussianBlur(rng.uniform(5000,45000,(512,512)).astype(np.float32),(0,0),1)
        truth=truth_matrix(pre.shape,(3.37,-2.43,.12,1.001))
        post=cv2.warpAffine(pre,truth,(512,512),borderMode=cv2.BORDER_REFLECT_101)
        coarse=truth.copy(); coarse[:,2]+=[.35,-.25]
        final,info=subpixel_refine(pre,post,coarse)
        self.assertTrue(info['accepted'],info)
        self.assertLess(errors(final,truth,pre.shape)['grid_rmse_px'],.05)
    def test_blank_retains_coarse(self):
        image=np.full((256,256),20000,np.float32); m=np.eye(2,3)
        out,info=subpixel_refine(image,image,m)
        np.testing.assert_array_equal(out,m); self.assertFalse(info['accepted'])
    def test_no_lattice_cell_jump(self):
        m=np.eye(2,3); n=m.copy(); n[0,2]=7.286
        self.assertFalse(permissible(n,m,(512,512)))

if __name__=='__main__': unittest.main()
