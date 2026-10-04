from field_round7_common import *
from scipy.special import expit, logit, xlogy
from scipy.stats import rankdata
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


def rankcorr(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3:return np.nan
    ra=rankdata(a[ok]);rb=rankdata(b[ok])
    if np.std(ra)<1e-12 or np.std(rb)<1e-12:return np.nan
    return float(np.corrcoef(ra,rb)[0,1])
def deviance(k,n,p):
    p=np.clip(p,EPS,1-EPS);mu=n*p
    return float(2*np.sum(xlogy(k,np.divide(k,mu,out=np.ones_like(mu),where=mu>0))+xlogy(n-k,np.divide(n-k,n-mu,out=np.ones_like(mu),where=n-mu>0))))

def fit_binomial(A,k,n,offset):
    # Newton iterations for strictly convex binomial negative log likelihood
    # plus coefficient-square penalty of strength one; monotone line search.
    coef=np.zeros(A.shape[1]);converged=False
    def objective(c):
        eta=offset+A@c
        return float(np.sum(n*np.logaddexp(0,eta)-k*eta)+.5*c@c)
    value=objective(coef)
    for it in range(60):
        eta=offset+A@coef;p=expit(eta)
        grad=A.T@(n*p-k)+coef
        H=A.T@((n*p*(1-p))[:,None]*A)+np.eye(A.shape[1])
        step=np.linalg.solve(H,grad)
        if np.max(abs(step))<1e-7:converged=True;break
        factor=1.
        for _ in range(30):
            proposal=coef-factor*step;next_value=objective(proposal)
            if next_value<=value+1e-9:break
            factor*=.5
        coef=proposal
        if abs(value-next_value)<1e-9:converged=True;break
        value=next_value
    return coef,converged,it+1

def cv_count(k,n,q,X,valid,return_maps=False):
    k=k.ravel().astype(float);n=n.ravel().astype(float);q=q.ravel().astype(float)
    X=X.reshape(4096,-1).astype(float);v=valid.ravel()
    pred=np.full(4096,np.nan);base=np.full(4096,np.nan);flat=np.full(4096,np.nan)
    maxit=0;fails=0;clipped=0;total=0
    for tr0,te0 in SPLITS:
        tr=tr0[v[tr0]];te=te0[v[te0]]
        if not len(te):continue
        if len(tr)<20:continue
        xm=X[tr].mean(axis=0);xs=X[tr].std(axis=0);xs[xs<1e-12]=1.
        A=(X[tr]-xm)/xs;B=(X[te]-xm)/xs
        avg=k[tr].sum()/n[tr].sum();mu=(k[tr]-n[tr]*q[tr]).sum()/n[tr].sum()
        p0tr=np.clip(q[tr]+mu,EPS,1-EPS);p0te=np.clip(q[te]+mu,EPS,1-EPS)
        clipped+=int(((q[te]+mu<=EPS)|(q[te]+mu>=1-EPS)).sum());total+=len(te)
        coef,conv,it=fit_binomial(A,k[tr],n[tr],logit(p0tr))
        maxit=max(maxit,it);fails+=not conv
        pred[te]=np.clip(expit(logit(p0te)+B@coef),EPS,1-EPS)
        base[te]=p0te;flat[te]=np.clip(avg,EPS,1-EPS)
    ok=np.isfinite(pred)&(n>0)
    if not ok.any():return {c:np.nan for c in ('逸脱度改善','共通除去逸脱度改善','順位相関','共通除去順位相関','決定係数')}
    d=deviance(k[ok],n[ok],pred[ok]);d0=deviance(k[ok],n[ok],flat[ok]);dq=deviance(k[ok],n[ok],base[ok])
    observed=k[ok]/n[ok]
    sse=np.sum((observed-pred[ok])**2);den=np.sum((observed-base[ok])**2)
    result=dict(逸脱度改善=1-d/d0 if d0>0 else np.nan,共通除去逸脱度改善=1-d/dq if dq>0 else np.nan,順位相関=rankcorr(pred[ok],observed),共通除去順位相関=rankcorr(pred[ok]-q[ok],observed-q[ok]),決定係数=1-sse/den if den>0 else np.nan,予測逸脱度=d,平坦基準逸脱度=d0,共通基準逸脱度=dq,保留升目数=int(ok.sum()),確率制限升目数=clipped,最大反復数=maxit,未収束分割数=fails)
    if return_maps:result['予測地図']=pred.reshape(64,64);result['基準地図']=base.reshape(64,64)
    return result

def score(k,n,q,X):
    X=X[:,:,None] if X.ndim==2 else X
    valid=(n>0)&np.isfinite(q)&np.isfinite(X).all(axis=2)
    full=cv_count(k,n,q,X,valid);ctl=controls(X);shared=valid.copy()
    for c in ctl:shared &= np.isfinite(c).all(axis=2)
    real=cv_count(k,n,q,X,shared)
    values=[cv_count(k,n,q,c,shared) for c in ctl]
    result={('全面'+c):v for c,v in full.items()}
    result.update({('共通領域'+c):v for c,v in real.items()})
    for c in ('逸脱度改善','共通除去逸脱度改善','順位相関','共通除去順位相関','決定係数'):
        arr=[v.get(c,np.nan) for v in values];median=float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan
        result['対照中央値'+c]=median;result['対照差'+c]=real.get(c,np.nan)-median
        for j,v in enumerate(arr):result[f'対照{j+1}'+c]=v
    result['対照未収束分割数']=sum(v.get('未収束分割数',0) for v in values)
    return result

