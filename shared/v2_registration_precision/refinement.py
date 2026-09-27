"""Subpixel local correspondences after the production coarse affine estimate.

Every matrix maps pre coordinates to post coordinates. Original images are used
at every step; no repeatedly resampled images enter fitting. The original coarse
QC gate must pass before local refinement is attempted.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from shared import registration as reg
from shared.image_qc import bright_band_mask
from shared.lattice_indexing import grid_coordinates,lattice_from_fft

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

def _sample(image, xy):
    return map_coordinates(image,[xy[:,1],xy[:,0]],order=1,mode='nearest',prefilter=False)

def _photometric_data(pre,post,matrix,centers,offsets,train):
    """Patch-centered, contrast-normalized measurements and affine Jacobian."""
    h,w=pre.shape; radius=np.hypot(w/2,h/2)
    pre_xy=centers[:,None,:]+offsets[None,:,:]
    post_xy=transform(pre_xy.reshape(-1,2),matrix).reshape(pre_xy.shape)
    pre_values=_sample(pre,pre_xy.reshape(-1,2)).reshape(len(centers),-1)
    post_values=_sample(post,post_xy.reshape(-1,2)).reshape(len(centers),-1)
    gy,gx=np.gradient(post)
    dx=_sample(gx,post_xy.reshape(-1,2)).reshape(post_xy.shape[:2])
    dy=_sample(gy,post_xy.reshape(-1,2)).reshape(post_xy.shape[:2])
    def normalize(values,derivatives=None):
        centered=values-values.mean(axis=1,keepdims=True)
        variance=np.mean(centered**2,axis=1,keepdims=True)
        denom=np.sqrt(variance+1e-6)
        norm=centered/denom
        if derivatives is None:return norm
        dcenter=derivatives-derivatives.mean(axis=1,keepdims=True)
        projection=np.sum(centered[:, :, None]*dcenter,axis=1,keepdims=True)
        jac=dcenter/denom[:,:,None]-centered[:,:,None]*projection/(len(offsets)*denom[:,:,None]**3)
        return norm,jac
    pre_norm=normalize(pre_values)
    post_norm,jxy=normalize(post_values,np.stack([dx,dy],axis=2))
    xc=post_xy[:,:,0]-((w-1)/2);yc=post_xy[:,:,1]-((h-1)/2)
    # Parameters are x shift, y shift, rotation at the image edge, scale at the edge.
    jac=np.empty((len(centers),len(offsets),4),float)
    jac[:,:,0]=jxy[:,:,0];jac[:,:,1]=jxy[:,:,1]
    jac[:,:,2]=jxy[:,:,0]*(-yc/radius)+jxy[:,:,1]*(xc/radius)
    jac[:,:,3]=jxy[:,:,0]*(xc/radius)+jxy[:,:,1]*(yc/radius)
    return post_norm-pre_norm,jac,post_xy

def lattice_refine(pre_raw,post_raw,initial,*,iterations=1):
    """Use the fixed-pitch pre-image hex grid as photometric support for a rigid update."""
    shape=pre_raw.shape
    pre=reg.image01_for_registration(pre_raw);post=reg.image01_for_registration(post_raw)
    # Suppress sub-pixel sensor noise equally while retaining the pillar contrast.
    pre=cv2.GaussianBlur(pre,(0,0),.55);post=cv2.GaussianBlur(post,(0,0),.55)
    invalid_pre=support_mask(pre_raw,8);invalid_post=support_mask(post_raw,8)
    clean=pre.copy(); clean[invalid_pre]=cv2.GaussianBlur(pre,(0,0),8)[invalid_pre]
    try:
        lattice=lattice_from_fft(clean,PITCH)
        _,coords=grid_coordinates(lattice,shape[1],shape[0],margin=24)
    except Exception as exc:
        return initial,{'stage':'lattice','accepted':False,'reason':f'fft_grid_failed: {exc}'}
    rng=np.random.default_rng(20260926)
    if len(coords)>4000: coords=coords[rng.choice(len(coords),4000,replace=False)]
    offsets=np.vstack([np.zeros((1,2)),np.c_[1.4*np.cos(np.arange(6)*np.pi/3),1.4*np.sin(np.arange(6)*np.pi/3)],
                              np.c_[2.8*np.cos(np.arange(6)*np.pi/3),2.8*np.sin(np.arange(6)*np.pi/3)]])
    xy0=coords[:,None,:]+offsets[None,:,:]
    q0=transform(xy0.reshape(-1,2),initial).reshape(xy0.shape)
    def is_clear(points,mask):
        p=np.rint(points).astype(int);h,w=mask.shape
        ok=(p[:,:,0]>=8)&(p[:,:,0]<w-8)&(p[:,:,1]>=8)&(p[:,:,1]<h-8)
        indices=np.where(ok)
        ok[indices] &= ~mask[p[:,:,1][indices],p[:,:,0][indices]]
        return ok.all(axis=1)
    usable=is_clear(xy0,invalid_pre)&is_clear(q0,invalid_post)
    centers=coords[usable]
    if len(centers)<400:
        return initial,{'stage':'lattice','accepted':False,'reason':'insufficient_unmasked_grid','n_grid':len(centers)}
    order=rng.permutation(len(centers));validation=np.zeros(len(centers),bool);validation[order[::5]]=True
    training=~validation
    identity=np.arange(len(centers))
    residual,jac,_=_photometric_data(pre,post,initial,centers,offsets,training)
    def loss(mat,sel):
        r,_,_=_photometric_data(pre,post,mat,centers[sel],offsets,training)
        ar=np.abs(r);return float(np.mean(np.where(ar<=.5,.5*ar*ar,.5*(ar-.25))))
    before_train=loss(initial,training);before_valid=loss(initial,validation)
    current=initial.astype(float).copy();current_train=before_train;current_valid=before_valid
    accepted=0;history=[]
    for _ in range(max(1,iterations)):
        residual,jac,_=_photometric_data(pre,post,current,centers,offsets,training)
        r=residual.reshape(-1);J=jac.reshape(-1,4)
        weights=np.minimum(1.,.5/np.maximum(np.abs(r),1e-12))
        delta=np.linalg.lstsq(J*np.sqrt(weights[:,None]),-r*np.sqrt(weights),rcond=None)[0]
        if not np.isfinite(delta).all(): break
        stepnorm=float(np.linalg.norm(delta));
        if stepnorm>.45: delta*=.45/stepnorm
        old=current.copy(); best=None
        for factor in (1.,.5,.25,.125,.0625):
            dx,dy,edge_theta,edge_scale=delta*factor
            h,w=shape;radius=np.hypot(w/2,h/2);c=np.array([(w-1)/2,(h-1)/2])
            angle=edge_theta/radius;s=1+edge_scale/radius
            rot=s*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            correction=np.c_[rot,c+np.array([dx,dy])-rot@c]
            candidate=(np.vstack([correction,[0,0,1]])@np.vstack([current,[0,0,1]]))[:2]
            if not permissible(candidate,initial,shape):continue
            train_loss=loss(candidate,training);valid_loss=loss(candidate,validation)
            if train_loss<current_train-1e-10 and valid_loss<=current_valid+1e-8:
                best=(candidate,train_loss,valid_loss,float(factor));break
        if best is None:break
        current,current_train,current_valid,factor=best;accepted+=1
        history.append({'step_px':stepnorm*factor,'train_loss':current_train,'validation_loss':current_valid})
        if stepnorm*factor<1e-4:break
    change=float(np.max(np.linalg.norm(transform(probes(shape),current)-transform(probes(shape),initial),axis=1)))
    did_accept=accepted>0
    return (current.astype(np.float32) if did_accept else initial),{
        'stage':'lattice','accepted':did_accept,'reason':'validation_improved' if did_accept else 'no_validated_step',
        'n_grid':len(centers),'n_training':int(training.sum()),'n_validation':int(validation.sum()),
        'iterations':accepted,'max_change_px':change,'train_loss_before':before_train,
        'train_loss_after':current_train,'validation_loss_before':before_valid,
        'validation_loss_after':current_valid,'history':history}

def iterative_lattice_refine(pre_raw,post_raw,initial,*,max_iterations=10,tolerance_px=1e-4):
    """Repeat one validated lattice update from its saved predecessor to convergence."""
    current=np.asarray(initial,dtype=np.float32).copy();anchor=current.copy();history=[];reason='no_validated_step'
    for iteration in range(max_iterations):
        updated,info=lattice_refine(pre_raw,post_raw,current,iterations=1)
        if not info.get('accepted'):
            reason='no_validated_step';break
        delta=float(np.max(np.linalg.norm(transform(probes(pre_raw.shape),updated)-
                                           transform(probes(pre_raw.shape),current),axis=1)))
        total=float(np.max(np.linalg.norm(transform(probes(pre_raw.shape),updated)-
                                           transform(probes(pre_raw.shape),anchor),axis=1)))
        if total>=PITCH/4:
            reason='cumulative_cell_guard';break
        history.append({'iteration':iteration+1,'change_px':delta,
                        'validation_loss':info['validation_loss_after']})
        current=updated
        if delta<tolerance_px:
            reason='converged';break
    else:
        reason='max_iterations'
    return current,{'stage':'iterative','accepted':bool(history),
                    'reason':reason,
                    'iterations':len(history),'max_change_px':max((x['change_px'] for x in history),default=0.),
                    'total_change_px':float(np.max(np.linalg.norm(transform(probes(pre_raw.shape),current)-
                                                                  transform(probes(pre_raw.shape),anchor),axis=1))),
                    'history':history}

def register_refined(pre_raw,post_raw,*,stage='subpixel',initial=None):
    coarse=(reg.register_image_pair_affine(pre_raw,post_raw) if initial is None
            else np.asarray(initial,dtype=np.float32).copy())
    if coarse.shape!=(2,3) or not reg.assess_affine_transform_qc(coarse,pre_raw.shape)['accepted']:
        raise ValueError('The supplied initial transform failed physical affine QC.')
    if stage=='subpixel': refined,info=subpixel_refine(pre_raw,post_raw,coarse); diagnostics={'subpixel':info}
    elif stage=='lattice':
        refined,info=lattice_refine(pre_raw,post_raw,coarse,iterations=1);diagnostics={'lattice':info}
    elif stage=='iterative':
        refined,info=iterative_lattice_refine(pre_raw,post_raw,coarse,max_iterations=10);diagnostics={'iterative':info}
    else: raise ValueError(f'Unavailable stage: {stage}')
    qc=reg.assess_affine_transform_qc(refined,pre_raw.shape)
    if not qc['accepted']: raise reg.AffineTransformQCError(qc)
    return refined,diagnostics
