"""Read-only motor calibration, task-specific measurements reused by v59/v60."""
from __future__ import annotations
import os, sys, json, hashlib, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/v60_band_scar_controls'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT/'mplconfig')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import cv2, tifffile
from scipy import ndimage, optimize, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cv2.setNumThreads(2)
RAW_PARENT = Path(r'W:\GoogleDrive\chuya2816')
RAW = next(p for p in RAW_PARENT.iterdir() if p.is_dir() and p.name=='5.生データD_chu')
def folder(name):
    hits=[p for p in RAW.iterdir() if p.is_dir() and p.name==name]
    if len(hits)!=1: raise FileNotFoundError(name)
    return hits[0]
def read(path): return tifffile.imread(path).astype(np.float32)
def fingerprint(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def contrast(im):
    f=im.astype(np.float32)/65535
    return cv2.boxFilter(f-cv2.GaussianBlur(f,(51,51),0),-1,(3,3),normalize=False)
def sample(c,xy,interp=False):
    xy=np.asarray(xy); h,w=c.shape
    valid=(xy[:,0]>=1)&(xy[:,0]<w-2)&(xy[:,1]>=1)&(xy[:,1]<h-2)
    d=np.full(len(xy),np.nan)
    if interp: d[valid]=ndimage.map_coordinates(c,xy[valid,::-1].T,order=1,prefilter=False)
    else:
        q=np.rint(xy[valid]).astype(int); d[valid]=c[q[:,1],q[:,0]]
    return d
def nonperiodic(im):
    # Suppress the lattice BEFORE coarse registration; defects and broad structure remain.
    a=cv2.GaussianBlur(im,(0,0),5)-cv2.GaussianBlur(im,(0,0),45)
    a=(a-a.mean())/(a.std()+1e-9)
    return a.astype(np.float32)
def register(a,b):
    h,w=a.shape; win=cv2.createHanningWindow((w,h),cv2.CV_32F)
    na,nb=nonperiodic(a),nonperiodic(b)
    phase_shift,response=cv2.phaseCorrelate(na.copy(),nb.copy(),win)
    # Ordinary broad-structure correlation retains defect amplitudes; phase-only
    # correlation can lock onto camera-fixed structure or a long cross arm.
    sa=cv2.resize(na[80:-80,80:-80],None,fx=.25,fy=.25)
    sb=cv2.resize(nb[80:-80,80:-80],None,fx=.25,fy=.25)
    corr=signal.fftconvolve(sb,sa[::-1,::-1],mode='full')
    yy,xx=np.unravel_index(np.argmax(corr),corr.shape)
    coarse=((xx-sa.shape[1]+1)*4.,(yy-sa.shape[0]+1)*4.)
    # Independent defect-feature coarse check (lattice suppressed, no command displacement used).
    u8=lambda z: np.clip((z+3)*255/6,0,255).astype(np.uint8)
    sift=cv2.SIFT_create(nfeatures=2500,contrastThreshold=.025)
    ka,da=sift.detectAndCompute(u8(na),None); kb,db=sift.detectAndCompute(u8(nb),None)
    feature_shift=np.array([np.nan,np.nan]); nmatch=0; nin=0
    if da is not None and db is not None:
        good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance < .7*n.distance]
        nmatch=len(good)
        if good:
            pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
            dif=pb-pa
            affine,inliers=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=10000,confidence=.999)
            ok=np.zeros(len(good),bool) if inliers is None else inliers.ravel().astype(bool)
            nin=int(ok.sum())
            if nin>=4: feature_shift=np.median(dif[ok],axis=0)
    init=np.array(coarse,dtype=np.float32)
    if nin>=10 and np.linalg.norm(feature_shift-init)>8: init=feature_shift.astype(np.float32)
    lowwarp=np.array([[1,0,init[0]],[0,1,init[1]]],np.float32)
    try:
        _,lowwarp=cv2.findTransformECC(na,nb,lowwarp,cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-6),None,5)
        init=lowwarp[:,2]
    except cv2.error: pass
    warp=np.array([[1,0,init[0]],[0,1,init[1]]],np.float32)
    aa=cv2.GaussianBlur(a,(0,0),.7); bb=cv2.GaussianBlur(b,(0,0),.7)
    aa=(aa-aa.mean())/aa.std(); bb=(bb-bb.mean())/bb.std()
    mask=np.zeros((h,w),np.uint8)
    x0=max(35,int(-init[0])+35); x1=min(w-35,int(w-init[0])-35)
    y0=max(35,int(-init[1])+35); y1=min(h-35,int(h-init[1])-35)
    mask[y0:y1,x0:x1]=255
    rho,warp=cv2.findTransformECC(aa,bb,warp,cv2.MOTION_TRANSLATION,
        (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-7),mask,3)
    shift=warp[:,2].astype(float)
    return shift,dict(coarse_dx=coarse[0],coarse_dy=coarse[1],coarse_response=response,
        feature_dx=feature_shift[0],feature_dy=feature_shift[1],feature_matches=nmatch,
        phase_dx=phase_shift[0],phase_dy=phase_shift[1],feature_inliers=nin,ecc=rho,fine_minus_coarse_px=float(np.linalg.norm(shift-init)),
        defect_vs_fine_px=float(np.linalg.norm(shift-feature_shift)))
def lattice_fit(im,guess=7.3):
    # Refine reciprocal peaks by windowed continuous Fourier coefficients.
    h,w=im.shape
    c=im-cv2.GaussianBlur(im,(51,51),0)
    wy,wx=np.hanning(h),np.hanning(w)
    F=np.abs(np.fft.fftshift(np.fft.fft2(c*wy[:,None]*wx[None,:])))
    fy=np.fft.fftshift(np.fft.fftfreq(h)); fx=np.fft.fftshift(np.fft.fftfreq(w))
    rad=np.hypot(fy[:,None],fx[None,:]); target=2/(np.sqrt(3)*guess)
    candidates=(ndimage.maximum_filter(F,9)==F)&(rad>target*.9)&(rad<target*1.1)
    yy,xx=np.where(candidates); order=np.argsort(F[yy,xx])[::-1]
    ks=[]; vals=[]
    y=np.arange(h); x=np.arange(w); cw=(c*wy[:,None]*wx[None,:]).astype(np.float64)
    for i in order:
        k0=np.array([fx[xx[i]],fy[yy[i]]])
        if k0[1]<0 or (abs(k0[1])<1e-12 and k0[0]<0): continue
        if any(abs(np.dot(k0,k)/np.linalg.norm(k0)/np.linalg.norm(k))>.9 for k in ks): continue
        def coef(k): return np.exp(-2j*np.pi*k[1]*y) @ (cw @ np.exp(-2j*np.pi*k[0]*x))
        res=optimize.minimize(lambda q:-abs(coef(q))/1e8,k0,method='Nelder-Mead',
            options={'xatol':1e-9,'fatol':1e-6,'maxiter':110})
        ks.append(res.x); vals.append(coef(res.x))
        if len(ks)==3: break
    if len(ks)<2: raise ValueError('No reciprocal basis')
    K=np.array(ks[:2]); B=np.linalg.inv(K)
    origin=np.linalg.solve(K,-np.angle(vals[:2])/(2*np.pi))
    pitch=float(np.median(2/(np.sqrt(3)*np.linalg.norm(ks,axis=1))))
    return B,origin,dict(pitch_px=pitch,k_json=json.dumps(np.array(ks).tolist()),basis_json=json.dumps(B.tolist()),origin_json=json.dumps(origin.tolist()))
def band_metrics(xy,d,threshold,return_map=False):
    ok=np.isfinite(d)&np.isfinite(xy).all(axis=1); xy=xy[ok]; d=d[ok]
    cell=np.clip((xy[:,1]//32).astype(int),0,63)*64+np.clip((xy[:,0]//32).astype(int),0,63)
    n=np.bincount(cell,minlength=4096); k=np.bincount(cell,weights=d>threshold,minlength=4096)
    m=np.divide(k,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)
    fy=np.fft.fftfreq(64,d=32)[:,None]; fx=np.fft.rfftfreq(64,d=32)[None,:]
    r=np.hypot(fy,fx); axis=(np.degrees(np.arctan2(np.broadcast_to(fy,r.shape),np.broadcast_to(fx,r.shape)))+90)%180
    band=(r>=1/1024)&(r<=1/128); bins=np.floor(((axis+5)%180)/10).astype(int)
    c=np.where(np.isfinite(m),m-np.nanmean(m),0); window=np.outer(np.hanning(64),np.hanning(64))
    z=np.fft.rfft2(c*window); mult=np.full(r.shape,2.); mult[:,0]=1; mult[:,-1]=1
    power=abs(z)**2*mult; total=power[band].sum()
    p=np.array([power[band&(bins==j)].sum() for j in range(18)])
    j=int(p.argmax()); theta=j*10.
    dist=abs((axis-theta+90)%180-90); sel=band&(dist<=15)
    peak=np.unravel_index(np.argmax(np.where(sel,power,-1)),power.shape)
    result=dict(n=len(d),positive=int((d>threshold).sum()),rate=float(np.mean(d>threshold)),
        strength=float(p[j]/total) if total>0 else 0,concentration=float(power[sel].sum()/total) if total>0 else 0,
        axis_deg=theta,period_px=float(1/r[peak]),phase_rad=float(np.angle(z[peak])),
        band_power=float(total),density_sd=float(np.nanstd(m)),peak_fx=float(np.broadcast_to(fx,r.shape)[peak]),peak_fy=float(np.broadcast_to(fy,r.shape)[peak]))
    return (result,m) if return_map else result
def main():
    paths=sorted([p for p in folder('261006-p50-ステッピングモーター_test').iterdir() if p.suffix.lower()=='.tif'],key=lambda p:p.stat().st_mtime)
    inv=[dict(path=str(p),position_mm=float(p.stem),mtime=p.stat().st_mtime,sha256=fingerprint(p)) for p in paths]
    pd.DataFrame(inv).to_csv(OUT/'input_inventory.csv',index=False)
    a=read(paths[0]); B,o,lat=lattice_fit(a,7.3)
    # Also inspect 50x annulus: strongest lattice family must support magnification decision.
    _,_,lat50=lattice_fit(a,3.6)
    lat['alternative_50x_pitch']=lat50['pitch_px']
    lat['magnification']='100x' if 6.8<lat['pitch_px']<7.8 else 'undetermined'
    rows=[]; images=[a]
    for i,p in enumerate(paths[1:],1):
        b=read(p); shift,q=register(a,b)
        rows.append(dict(pair=i,pre=paths[i-1].name,post=p.name,command_um=(float(p.stem)-float(paths[i-1].stem))*1000,dx_px=shift[0],dy_px=shift[1],distance_px=float(np.linalg.norm(shift)),**q))
        print(rows[-1],flush=True); a=b
    t=pd.DataFrame(rows); large=t[t.command_um.abs()>=10]
    A=large.command_um.to_numpy()[:,None]; shifts=large[['dx_px','dy_px']].to_numpy()
    v=np.linalg.lstsq(A,shifts,rcond=None)[0][0]; nm=1000/np.linalg.norm(v)
    rev=t.iloc[6]; measured_signed=np.dot([rev.dx_px,rev.dy_px],v)/np.dot(v,v)
    lat.update(nm_per_px_stage=nm,nm_per_px_lattice=460/lat['pitch_px'],stage_axis_deg=float(np.degrees(np.arctan2(v[1],v[0]))),
        reversal_command_um=float(rev.command_um),reversal_measured_um=float(measured_signed),backlash_um=float(abs(rev.command_um)-abs(measured_signed)),
        calibration_vx_px_per_um=float(v[0]),calibration_vy_px_per_um=float(v[1]))
    t['measured_um_signed']=t[['dx_px','dy_px']].to_numpy()@v/np.dot(v,v)
    t['command_error_um']=t.measured_um_signed-t.command_um
    t['fractional_dx']=t.dx_px-np.rint(t.dx_px); t['fractional_dy']=t.dy_px-np.rint(t.dy_px)
    t.to_csv(OUT/'pair_shifts.csv',index=False); (OUT/'calibration.json').write_text(json.dumps(lat,ensure_ascii=False,indent=2),encoding='utf8')
    fig,ax=plt.subplots(1,3,figsize=(13,4)); ax[0].plot(t.command_um,t.measured_um_signed,'o'); ax[0].plot([-22,2],[-22,2],'k--'); ax[0].set(xlabel='Command (um)',ylabel='Measured (um)')
    ax[1].plot(t.pair,t.command_error_um,'o-'); ax[1].set(xlabel='Pair in acquisition order',ylabel='Measured - command (um)')
    ax[2].plot(t.dx_px,t.dy_px,'o'); ax[2].set(xlabel='dx (px)',ylabel='dy (px)'); fig.tight_layout(); fig.savefig(OUT/'motor_calibration.png',dpi=160); plt.close(fig)
    print(json.dumps(lat,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
