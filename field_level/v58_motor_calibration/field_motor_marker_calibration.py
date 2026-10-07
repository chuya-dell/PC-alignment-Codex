"""Verified sample marker branch: cross arms move, camera-fixed dust does not."""
import sys,json
from pathlib import Path
import field_motor_calibration as M
import numpy as np,pandas as pd,cv2
from scipy.stats import theilslopes
sys.dont_write_bytecode=True
sys.path.insert(0,str(M.ROOT/'data/results/v58_band_scar_causal/checkout/field_level/v57_false_positive_facts'))
import field_v57_detect_scars as D
OUT=M.OUT
def marker(im):
    E,_=D.energy(im); med=float(np.median(E));sig=1.4826*np.median(abs(E-med))
    lv=D.detect_family(E,sig,med,'vertical'); lh=D.detect_family(E.T.copy(),sig,med,'horizontal')
    v=max([r for r in lv if r.get('status')=='accepted'],key=lambda r:r['z']); h=max([r for r in lh if r.get('status')=='accepted'],key=lambda r:r['z'])
    # Detector line coordinates are centered on downsampled image dimensions.
    xv=float(v['X0'])*2; yv=(im.shape[0]/2-1); sh=float(h['S']);sv=float(v['S']);yh=float(h['X0'])*2;xh=im.shape[1]/2-1
    intersect=np.linalg.solve([[1,-sv],[-sh,1]],[xv-sv*yv,yh-sh*xh])
    return intersect,dict(v=v,h=h)
def main():
    paths=sorted(M.folder('261006-p50-ステッピングモーター_test').glob('*.tif'),key=lambda p:p.stat().st_mtime)
    info=[]
    for p in paths:
        q,details=marker(M.read(p));info.append(dict(path=str(p),name=p.name,position_mm=float(p.stem),x=q[0],y=q[1],lines=details));print('MARKER',p.name,q,flush=True)
    (OUT/'marker_landmarks.json').write_text(json.dumps(info,ensure_ascii=False,indent=2,default=float),encoding='utf8')
    rows=[]
    for i in range(1,len(paths)):
        a,b=M.read(paths[i-1]),M.read(paths[i]); init=np.array([info[i]['x']-info[i-1]['x'],info[i]['y']-info[i-1]['y']],np.float32)
        warp=np.array([[1,0,init[0]],[0,1,init[1]]],np.float32)
        aa=cv2.GaussianBlur(a,(0,0),.7);bb=cv2.GaussianBlur(b,(0,0),.7);aa=(aa-aa.mean())/aa.std();bb=(bb-bb.mean())/bb.std()
        # Mask for fitting only: central sample-side landmark strips. Never a positive mask.
        yy,xx=np.indices(a.shape);v=info[i-1]['lines']['v'];h=info[i-1]['lines']['h']
        vx=float(v['X0'])*2+float(v['S'])*(yy-(a.shape[0]/2-1));hy=float(h['X0'])*2+float(h['S'])*(xx-(a.shape[1]/2-1))
        strip=(abs(xx-vx)<75)|(abs(yy-hy)<75)
        valid=(xx+init[0]>40)&(xx+init[0]<a.shape[1]-40)&(yy+init[1]>40)&(yy+init[1]<a.shape[0]-40)&(xx>40)&(xx<a.shape[1]-40)&(yy>40)&(yy<a.shape[0]-40)
        # ECC inputMask is in input-image coordinates; transform this support to post.
        mask=cv2.warpAffine((strip&valid).astype(np.uint8)*255,warp,(a.shape[1],a.shape[0]))
        rho,warp=cv2.findTransformECC(aa,bb,warp,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-7),mask,3)
        shift=warp[:,2];err=float(np.linalg.norm(shift-init))
        if err>7.3/2:
            pd.DataFrame(rows).to_csv(OUT/'marker_pairs_checkpoint.csv',index=False)
            (OUT/'marker_failure.json').write_text(json.dumps(dict(pair=i,marker_shift=init.tolist(),fine_shift=shift.tolist(),fine_minus_marker_px=err,status='rejected')),encoding='utf8')
            raise RuntimeError(f'Fine branch slipped: pair {i}, {err}')
        rows.append(dict(pair=i,pre=paths[i-1].name,post=paths[i].name,command_um=(info[i]['position_mm']-info[i-1]['position_mm'])*1000,
            dx_px=float(shift[0]),dy_px=float(shift[1]),distance_px=float(np.linalg.norm(shift)),marker_dx=float(init[0]),marker_dy=float(init[1]),fine_minus_marker_px=err,ecc=float(rho)))
        print('PAIR',rows[-1],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'marker_pairs_checkpoint.csv',index=False)
    t=pd.DataFrame(rows);lat=json.loads((OUT/'calibration_initial_failed.json').read_text(encoding='utf-8-sig'))
    # Reversal axis is established by the six initial small steps, independently of commands for later images.
    small=t.iloc[:6];v=np.linalg.lstsq(small.command_um.to_numpy()[:,None],small[['dx_px','dy_px']].to_numpy(),rcond=None)[0][0]
    lat['small_step_axis_deg']=float(np.degrees(np.arctan2(v[1],v[0])));lat['small_step_nm_per_px']=1000/np.linalg.norm(v)
    large=t.iloc[7:]; pix=np.linalg.norm(large[['dx_px','dy_px']],axis=1);slope=np.dot(abs(large.command_um),pix)/np.dot(large.command_um,large.command_um)
    lat['nm_per_px_stage_all_large']=float(1000/slope)
    # Final consecutive monotone large steps form an explicit sensitivity, not a replacement silently chosen for agreement.
    last=t.iloc[9:];pixlast=np.linalg.norm(last[['dx_px','dy_px']],axis=1);slope_last=np.dot(abs(last.command_um),pixlast)/np.dot(last.command_um,last.command_um)
    lat['nm_per_px_stage_last_two']=float(1000/slope_last)
    lat['large_pair_nm_per_px']=(1000*abs(large.command_um.to_numpy())/pix).tolist()
    lat['nm_per_px_stage']=None;lat['stage_calibration_status']='single_scale_not_validated; individual large steps disagree'
    # Project reversal onto initial stage axis, using lattice calibration for physical magnitude.
    unit=v/np.linalg.norm(v);reverse=t.iloc[6]; signed_um=float(np.dot(reverse[['dx_px','dy_px']].to_numpy(dtype=float),unit)*lat['nm_per_px_lattice']/1000)
    lat['reversal_measured_um_lattice_scale']=signed_um;lat['backlash_um_apparent']=float(abs(reverse.command_um)-abs(signed_um));lat['backlash_status']='apparent_only; large transverse motion and nonmonotone commands'
    lat['reversal_transverse_um']=float(abs(np.cross(unit,reverse[['dx_px','dy_px']].to_numpy(dtype=float)))*lat['nm_per_px_lattice']/1000)
    for k in ['backlash_um','reversal_measured_um','calibration_vx_px_per_um','calibration_vy_px_per_um','stage_axis_deg']:lat.pop(k,None)
    t['fractional_dx']=t.dx_px-np.rint(t.dx_px);t['fractional_dy']=t.dy_px-np.rint(t.dy_px)
    t['measured_um_signed_lattice_scale']=t[['dx_px','dy_px']].to_numpy()@unit*lat['nm_per_px_lattice']/1000
    t['command_error_um_lattice_scale']=t.measured_um_signed_lattice_scale-t.command_um
    t.to_csv(OUT/'pair_shifts.csv',index=False);(OUT/'calibration.json').write_text(json.dumps(lat,ensure_ascii=False,indent=2),encoding='utf8')
    fig,ax=M.plt.subplots(1,3,figsize=(14,4));ax[0].plot(t.command_um,t.measured_um_signed_lattice_scale,'o');ax[0].plot([-22,2],[-22,2],'k--');ax[0].set(xlabel='Command (um)',ylabel='Measured using lattice scale (um)')
    ax[1].plot(t.pair,t.dx_px,'o-',label='dx');ax[1].plot(t.pair,t.dy_px,'o-',label='dy');ax[1].legend();ax[1].set(xlabel='Acquisition pair',ylabel='Measured shift (px)')
    ax[2].bar(large.pair,lat['large_pair_nm_per_px']);ax[2].axhline(63.1,c='r',label='63.1');ax[2].axhline(65,c='g',label='65');ax[2].legend();ax[2].set(xlabel='Large command pair',ylabel='Apparent nm/px')
    fig.tight_layout();fig.savefig(OUT/'motor_calibration.png',dpi=160);M.plt.close(fig);print(json.dumps(lat,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
