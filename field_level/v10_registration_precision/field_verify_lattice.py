"""Known-truth hexagonal intensity pattern checks the fixed-pitch lattice step."""
import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np
from shared.registration import image01_for_registration
from shared.v2_registration_precision.refinement import lattice_refine,iterative_lattice_refine,transform
from field_precision_benchmark import errors,truth_matrix

class Lattice(unittest.TestCase):
    def test_hex_grid_refines_small_motion(self):
        h=w=512; y,x=np.mgrid[:h,:w]; pitch=7.286
        phase1=x/pitch-y/(np.sqrt(3)*pitch);phase2=2*y/(np.sqrt(3)*pitch)
        image=(28000+7000*(np.cos(2*np.pi*phase1)+np.cos(2*np.pi*phase2)+np.cos(2*np.pi*(phase1+phase2))))
        image=image.astype(np.float32)
        truth=truth_matrix(image.shape,(.34,-.28,.035,1.0004))
        post=cv2.warpAffine(image,truth,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        refined,info=lattice_refine(image,post,np.eye(2,3,dtype=np.float32),iterations=1)
        self.assertTrue(info['accepted'],info)
        before=errors(np.eye(2,3),truth,image.shape)['grid_rmse_px']
        after=errors(refined,truth,image.shape)['grid_rmse_px']
        self.assertLess(after,before*.2)
        self.assertLess(info['validation_loss_after'],info['validation_loss_before'])

    def test_iteration_is_monotone_and_stays_in_cell(self):
        h=w=512;y,x=np.mgrid[:h,:w];pitch=7.286
        p=x/pitch-y/(np.sqrt(3)*pitch);q=2*y/(np.sqrt(3)*pitch)
        image=(28000+7000*(np.cos(2*np.pi*p)+np.cos(2*np.pi*q)+np.cos(2*np.pi*(p+q)))).astype(np.float32)
        truth=truth_matrix(image.shape,(.34,-.28,.035,1.0004))
        post=cv2.warpAffine(image,truth,(w,h),borderMode=cv2.BORDER_REFLECT_101)
        refined,info=iterative_lattice_refine(image,post,np.eye(2,3,dtype=np.float32))
        self.assertTrue(np.isfinite(refined).all())
        self.assertLess(info['total_change_px'],pitch/4)
        losses=[item['validation_loss'] for item in info['history']]
        self.assertTrue(all(y<=x+1e-8 for x,y in zip(losses,losses[1:])))
        self.assertGreater(info['iterations'],0)

if __name__=='__main__':unittest.main()
