import field_motor_calibration as M
import numpy as np,cv2
from scipy import signal
paths=sorted(M.folder('261006-p50-ステッピングモーター_test').glob('*.tif'),key=lambda p:p.stat().st_mtime)
fig,ax=M.plt.subplots(2,4,figsize=(16,8))
for k,i in enumerate([8,9,10,11]):
    a=M.read(paths[i]);small=cv2.resize(a,(512,511),interpolation=cv2.INTER_AREA)
    ax[0,k].imshow(small,cmap='gray',vmin=np.percentile(a,1),vmax=np.percentile(a,99));ax[0,k].set_title(paths[i].name)
    n=M.nonperiodic(a);ax[1,k].imshow(cv2.resize(n,(512,511)),cmap='gray',vmin=-2,vmax=2)
fig.tight_layout();fig.savefig(M.OUT/'diagnostic_images.png',dpi=120)
for i in [6,7,8,9]:
    a=M.nonperiodic(M.read(paths[i-1]));b=M.nonperiodic(M.read(paths[i]));a=a[80:-80,80:-80];b=b[80:-80,80:-80]
    a=cv2.resize(a,None,fx=.25,fy=.25);b=cv2.resize(b,None,fx=.25,fy=.25)
    corr=signal.fftconvolve(b,a[::-1,::-1],mode='full');mask=np.ones_like(corr,bool)
    peaks=[]
    for j in range(10):
        y,x=np.unravel_index(np.argmax(np.where(mask,corr,-np.inf)),corr.shape);peaks.append(((x-(a.shape[1]-1))*4,(y-(a.shape[0]-1))*4,float(corr[y,x])))
        mask[max(0,y-4):y+5,max(0,x-4):x+5]=False
    print(i,peaks,flush=True)
