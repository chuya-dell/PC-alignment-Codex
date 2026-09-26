"""Subpixel local correspondences after the production coarse affine estimate.

Every matrix maps pre coordinates to post coordinates. Original images are used
at every step; no repeatedly resampled images enter fitting. The original coarse
QC gate must pass before local refinement is attempted.
"""
from __future__ import annotations
import cv2
import numpy as np
from shared import registration as reg
from shared.image_qc import bright_band_mask

PITCH=7.286

def transform(points, matrix):
    return np.asarray(points)@matrix[:,:2].T+matrix[:,2]

def valid_points(points, mask, margin=12):
    xy=np.rint(points).astype(int); h,w=mask.shape
    valid=(xy[:,0]>=margin)&(xy[:,0]<w-margin)&(xy[:,1]>=margin)&(xy[:,1]<h-margin)
    index=np.flatnonzero(valid)
    valid[index] &= ~mask[xy[index,1],xy[index,0]]
    return valid

def support_mask(raw, radius=12):
    return cv2.dilate(bright_band_mask(raw).astype(np.uint8),
                      np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)

def probes(shape):
    h,w=shape
    return np.array([[x,y] for y in np.linspace(20,h-21,5) for x in np.linspace(20,w-21,5)])

def permissible(candidate, anchor, shape):
    # Stay within the same lattice cell everywhere, including at the image edge.
    return (reg.assess_affine_transform_qc(candidate,shape)['accepted'] and
            np.max(np.linalg.norm(transform(probes(shape),candidate)-transform(probes(shape),anchor),axis=1))<PITCH/4)

def subpixel_refine(pre_raw,post_raw,coarse):
    pre=reg.image01_for_registration(pre_raw); post=reg.image01_for_registration(post_raw)
    a=np.uint8(np.clip(pre*255,0,255)); b=np.uint8(np.clip(post*255,0,255))
    ma=support_mask(pre_raw); mb=support_mask(post_raw)
    points=cv2.goodFeaturesToTrack(a,maxCorners=2500,qualityLevel=.015,minDistance=12,
                                  mask=(~ma).astype(np.uint8)*255,blockSize=5)
    info={'stage':'subpixel','accepted':False}
    if points is None or len(points)<30:
        return coarse,{**info,'reason':'insufficient_features'}
    seeds=points.copy()
    # Full-precision native-intensity corner refinement before correspondence.
    points=cv2.cornerSubPix(pre,points,(3,3),(-1,-1),
                           (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,30,1e-4))
    keep=(np.linalg.norm(points[:,0]-seeds[:,0],axis=1)<1.5)&valid_points(points[:,0],ma)
    points=np.ascontiguousarray(points[keep],dtype=np.float32)
    if len(points)<30: return coarse,{**info,'reason':'insufficient_refined_features'}
    prediction=transform(points[:,0],coarse).astype(np.float32)[:,None,:]
    criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-4)
    tracked,status,_=cv2.calcOpticalFlowPyrLK(a,b,points,prediction.copy(),winSize=(15,15),
                     maxLevel=0,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    back,back_status,_=cv2.calcOpticalFlowPyrLK(b,a,tracked,points.copy(),winSize=(15,15),
                     maxLevel=0,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    valid=(status.ravel()!=0)&(back_status.ravel()!=0)&valid_points(tracked[:,0],mb)
    valid &= np.linalg.norm(back[:,0]-points[:,0],axis=1)<.3
    valid &= np.linalg.norm(tracked[:,0]-prediction[:,0],axis=1)<PITCH/4
    src=points[valid,0].astype(float); dst=tracked[valid,0].astype(float)
    info.update(n_features=len(points),n_tracks=len(src))
    if len(src)<30: return coarse,{**info,'reason':'insufficient_bidirectional_tracks'}
    # Deterministic held-out correspondences, never used in the affine fit.
    test=np.arange(len(src))%5==0
    candidate,inliers=cv2.estimateAffine2D(src[~test],dst[~test],method=cv2.RANSAC,
                            ransacReprojThreshold=.4,maxIters=2000,confidence=.999,refineIters=10)
    if candidate is None: return coarse,{**info,'reason':'fit_failed'}
    selected=src[~test][inliers.ravel()!=0]
    if len(selected)<20 or np.any(np.ptp(selected,axis=0)<np.array(pre.shape[::-1])*.4):
        return coarse,{**info,'reason':'insufficient_spatial_support'}
    old=np.median(np.linalg.norm(transform(src[test],coarse)-dst[test],axis=1))
    new=np.median(np.linalg.norm(transform(src[test],candidate)-dst[test],axis=1))
    info.update(heldout_before_px=float(old),heldout_after_px=float(new),n_inliers=len(selected))
    accept=permissible(candidate,coarse,pre.shape) and new<old-1e-5
    info.update(accepted=bool(accept),reason='improved' if accept else 'guard_or_no_heldout_improvement')
    return (candidate.astype(np.float32) if accept else coarse),info

def register_refined(pre_raw,post_raw,*,stage='subpixel'):
    if stage!='subpixel': raise ValueError(f'Unavailable stage: {stage}')
    coarse=reg.register_image_pair_affine(pre_raw,post_raw)
    refined,info=subpixel_refine(pre_raw,post_raw,coarse)
    qc=reg.assess_affine_transform_qc(refined,pre_raw.shape)
    if not qc['accepted']: raise reg.AffineTransformQCError(qc)
    return refined,{'subpixel':info}
