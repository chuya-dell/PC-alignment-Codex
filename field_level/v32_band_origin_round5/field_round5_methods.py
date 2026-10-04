import sys
sys.dont_write_bytecode=True
import numpy as np
import cv2
import re
from scipy import fft
from field_round5_common import *
yy,xx=np.indices((64,64))
def angular_difference(a, b):
    return np.abs((np.asarray(a)-b+90) % 180-90)

def folds(valid):
    labels=((xx//8)+(yy//8))%8;out=[]
    for k in range(8):
        test=(labels==k)&valid
        embargo=cv2.dilate((labels==k).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        out.append((np.flatnonzero(valid&~embargo),np.flatnonzero(test)))
    return out

def cv_score(response,X,valid):
    response=response.ravel().astype(float);X=X.reshape(4096,-1).astype(float)
    sse=0.;base=0.;pred=np.full(4096,np.nan)
    for tr,te in folds(valid):
        if len(te)==0:continue
        if len(tr)<20:return np.nan,pred.reshape(64,64)
        xm=X[tr].mean(axis=0);xs=X[tr].std(axis=0);xs[xs<1e-12]=1
        a=(X[tr]-xm)/xs;b=(X[te]-xm)/xs;ym=response[tr].mean()
        coef=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@(response[tr]-ym))
        pr=ym+b@coef;pred[te]=pr
        sse+=float(np.sum((response[te]-pr)**2));base+=float(np.sum((response[te]-ym)**2))
    return (1-sse/base if base>0 else np.nan),pred.reshape(64,64)

def shifted(a,dy,dx):
    out=np.full_like(a,np.nan)
    out[max(0,dy):min(64,64+dy),max(0,dx):min(64,64+dx)]=a[max(0,-dy):min(64,64-dy),max(0,-dx):min(64,64-dx)]
    return out

def controls(a):return [shifted(a,0,d) for d in (-16,-8,8,16)]+[shifted(a,d,0) for d in (-16,-8,8,16)]+[np.rot90(a)]
def read_image(p):
    a = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None or a.ndim != 2 or a.shape != (2044, 2048):
        raise RuntimeError('画像の形式・寸法不一致 '+str(p))
    return a

def sample(raw, xy, bilinear=False):
    # Preserve float32 normalization/background and nine separate additions.
    a = raw.astype(np.float32)/65535.
    b = cv2.GaussianBlur(a, (51, 51), 0)
    h, w = a.shape
    valid = np.isfinite(xy).all(axis=1)
    if bilinear:
        x, y = np.floor(xy).astype(int).T
        valid &= (x >= 1) & (x < w-2) & (y >= 1) & (y < h-2)
        tx, ty = (xy-np.floor(xy)).T
    else:
        x, y = np.rint(xy).astype(int).T
        valid &= (x >= 1) & (x < w-1) & (y >= 1) & (y < h-1)
    intensity = np.zeros(len(xy))
    background = np.zeros(len(xy))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not bilinear:
                intensity[valid] += a[y[valid]+dy, x[valid]+dx]
                background[valid] += b[y[valid]+dy, x[valid]+dx]
            else:
                for ox, oy, weight in ((0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)):
                    intensity[valid] += a[y[valid]+dy+oy, x[valid]+dx+ox]*weight[valid]
                    background[valid] += b[y[valid]+dy+oy, x[valid]+dx+ox]*weight[valid]
    value = intensity-background
    value[~valid] = np.nan
    return value

def diagnostics(matrix):
    linear = matrix[:, :2].astype(float)
    center = np.array([(2048-1)/2, (2044-1)/2])
    delta = linear@center+matrix[:, 2]-center
    singular = np.linalg.svd(linear, compute_uv=False)
    # Exact definitions used by the current shared estimator.
    return dict(dx_center_px=float(delta[0]), dy_center_px=float(delta[1]), rotation_deg=float(np.degrees(np.arctan2(linear[1,0], linear[0,0]))),
                scale=float(np.sqrt(abs(np.linalg.det(linear)))), anisotropy=float(singular[0]/singular[-1]), determinant=float(np.linalg.det(linear)))

def expected_diagnostics(s):
    return {k:float(v) for k,v in re.findall(r"'(dx_center_px|dy_center_px|rotation_deg|scale|anisotropy|determinant)': ([\d.eE+\-]+)", str(s))}

def binned(xy, values):
    cell = (xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
    good = np.isfinite(values)
    n = np.bincount(cell[good],minlength=4096)
    s = np.bincount(cell[good],weights=values[good],minlength=4096)
    return np.divide(s,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)
def kernel_field(support,delta,query):
    out=[]
    for start in range(0,len(query),512):
        dist=((query[start:start+512,None,:]-support[None,:,:])/256)**2
        logw=-.5*dist.sum(axis=2)
        logw-=logw.max(axis=1,keepdims=True)
        w=np.exp(logw)
        out.append(w@delta/w.sum(axis=1,keepdims=True))
    return np.concatenate(out)

def tracking(pre,post,matrix):
    a=np.uint8(np.clip(pre.astype(np.float32)/65535*255,0,255))
    b=np.uint8(np.clip(post.astype(np.float32)/65535*255,0,255))
    points=cv2.goodFeaturesToTrack(a,maxCorners=2500,qualityLevel=.015,minDistance=12,blockSize=5)
    if points is None:
        return {},dict(採用候補=False,理由='特徴点なし')
    points=points.astype(np.float32)
    pred=(points[:,0]@matrix[:,:2].T+matrix[:,2]).astype(np.float32)[:,None,:]
    criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-4)
    data=[]
    for win in (21,31):
        t,s,_=cv2.calcOpticalFlowPyrLK(a,b,points,pred.copy(),winSize=(win,win),maxLevel=0,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        back,sb,_=cv2.calcOpticalFlowPyrLK(b,a,t,points.copy(),winSize=(win,win),maxLevel=0,criteria=criteria,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
        fb=np.linalg.norm(back[:,0]-points[:,0],axis=1)
        good=(s.ravel()!=0)&(sb.ravel()!=0)&(fb<=.2)&(np.linalg.norm(t[:,0]-pred[:,0],axis=1)<=2)
        good&=(points[:,0,0]>=20)&(points[:,0,0]<2028)&(points[:,0,1]>=20)&(points[:,0,1]<2024)
        good&=(t[:,0,0]>=20)&(t[:,0,0]<2028)&(t[:,0,1]>=20)&(t[:,0,1]<2024)
        data.append((t[:,0],good,fb))
    good=data[0][1]&data[1][1]&(np.linalg.norm(data[0][0]-data[1][0],axis=1)<=.2)
    p=points[good,0].astype(float)
    d=(data[0][0]-pred[:,0])[good].astype(float)
    occupied=len(np.unique((p[:,1]//256).astype(int)*8+(p[:,0]//256).astype(int)))
    info=dict(元特徴点数=len(points),支持点数=len(p),支持区画数=occupied,採用候補=False)
    arrays=dict(support=p,displacement=d,fb_error=data[0][2][good],window_error=np.linalg.norm(data[0][0]-data[1][0],axis=1)[good])
    if len(p)<100 or occupied<32:
        return arrays,{**info,'理由':'支持不足'}
    groups=np.arange(len(p))%2
    before=[];after=[]
    yy,xx=np.indices((64,64));q=np.column_stack([xx.ravel()*32+16,yy.ravel()*32+16])
    maps=[]
    for g in (0,1):
        train=groups==g;test=~train
        held=kernel_field(p[train],d[train],p[test])
        before.append(float(np.median(np.linalg.norm(d[test],axis=1))))
        after.append(float(np.median(np.linalg.norm(d[test]-held,axis=1))))
        maps.append(kernel_field(p[train],d[train],q))
    disagreement=float(np.median(np.linalg.norm(maps[0]-maps[1],axis=1)))
    accept=all(y<x for x,y in zip(before,after)) and disagreement<=.2
    info.update(補正前誤差中央値=before,補正後誤差中央値=after,二群地図差中央値=disagreement,
                採用候補=bool(accept),理由='独立支持条件を通過。非周期目印検証が別途必要' if accept else '独立支持条件不成立')
    arrays['residual_grid']=kernel_field(p,d,q).reshape(64,64,2)
    arrays['support_groups']=groups
    return arrays,info

def sign_test(values,groups=None):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if len(values)==0:return np.nan
    rng=np.random.default_rng(20261004)
    observed=values.mean();hits=0
    if groups is not None:
        _,g=np.unique(np.asarray(groups,str),return_inverse=True);n=int(g.max())+1
    else:g=np.arange(len(values));n=len(values)
    for _ in range(100):
        signs=rng.choice(np.array([-1,1],np.int8),size=(100,n))
        stats=(signs[:,g]@values)/len(values)
        hits+=int((stats>=observed-1e-15).sum())
    return (1+hits)/10001
