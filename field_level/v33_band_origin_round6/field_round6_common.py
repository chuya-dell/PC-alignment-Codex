"""Read-only discovery and count evaluation; no imports of previous versions."""
import sys, os
sys.dont_write_bytecode=True
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import hashlib, json, time, subprocess
import numpy as np
import pandas as pd
import cv2
from scipy.special import expit, logit, xlogy
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/v33_band_origin_round6'
LOCAL=ROOT/'data/inputs_local'
OLD=ROOT/'data/results/v28_band_origin_round1'
R2=ROOT/'data/results/v29_band_origin_round2'
R3=ROOT/'data/results/v30_band_origin_round3'
R4=ROOT/'data/results/v31_band_origin_round4'
R5=ROOT/'data/results/v32_band_origin_round5'
RESUME=R5/'resume_20261004'
METHODS=['平均標準偏差','中央値絶対偏差']
SEED=20261004
PRED_HASH='21c82af86a92cc3d76c173b53e0c5ab055352a84b17806f0859cfd92500b0180'
EPS=1e-6
cv2.setNumThreads(1)
yy,xx=np.indices((64,64))
LABELS=((xx//8)+(yy//8))%8
SPLITS=[(np.flatnonzero(~cv2.dilate((LABELS==k).astype('uint8'),np.ones((5,5),'uint8')).astype(bool)),np.flatnonzero(LABELS==k)) for k in range(8)]

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def dump(a,p):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def csv(a,name):pd.DataFrame(a).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def verify():assert digest(OUT/'261004_周6_予測と検定規定.md')==PRED_HASH
def rng(s):return np.random.default_rng(SEED+int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little'))

# One durable record per actually read file, independent of checkpoint resume.
def used(p):
    p=Path(p).resolve()
    assert LOCAL.resolve() in p.parents or any(d.resolve() in p.parents for d in (OLD,R2,R3,R4,R5)),str(p)
    dest=OUT/'input_fingerprints';dest.mkdir(exist_ok=True)
    ident=hashlib.sha256(str(p).encode()).hexdigest()
    rec=dest/(ident+'.json')
    if not rec.exists():dump(dict(実在パス=str(p),バイト数=p.stat().st_size,開始内容指紋=digest(p)),rec)
    return p
def table(p):return pd.read_csv(used(p),dtype={'日程':str,'基板':str,'date':str,'board':str})
def arrays(p):
    with np.load(used(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def record(p):return json.loads(used(p).read_text(encoding='utf-8-sig'))
def discover():
    children={p.name:p for p in LOCAL.iterdir()}
    src=[p for n,p in children.items() if p.is_dir() and n.startswith('2026') and 'digital_judgment' in n]
    assert len(src)==1
    subs={p.name:p for p in src[0].iterdir()};tabs={p.name:p for p in subs['tables'].iterdir()}
    caches={p.stem:p for p in tabs['cached_field_differences'].iterdir() if p.suffix=='.npz'}
    raw={p.name:p for p in children['raw_readonly'].iterdir()}
    assert any(p.is_file() and 'copy_done' in n for n,p in raw.items())
    return src[0],tabs,caches,raw
def fields():return table(OLD/'round1_fields_both_definitions.csv').query('定義 == @METHODS[0]').reset_index(drop=True)
def meta(r):return {k:r[k] for k in ('key','日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野')}
def common_data():
    a=arrays(OLD/'density_maps.npz');b=arrays(OLD/'common_profile_residuals.npz')
    n=a['counts'].astype(float);dm=a['density'].astype(float)
    k=np.rint(np.nan_to_num(dm)*n[:,None]).astype(int)
    assert np.nanmax(abs(np.nan_to_num(dm)*n[:,None]-k))<1e-4
    assert np.all(k<=n[:,None]);assert a['keys'].tolist()==b['keys'].tolist()
    return a['keys'].tolist(),k,n,b['common'].astype(float),b['residual'].astype(float),dm
def shifted(a,dy,dx):
    out=np.full_like(a,np.nan,dtype=float)
    out[max(0,dy):min(64,64+dy),max(0,dx):min(64,64+dx)]=a[max(0,-dy):min(64,64-dy),max(0,-dx):min(64,64-dx)]
    return out
def controls(a):return [shifted(a,0,d) for d in (-16,-8,8,16)]+[shifted(a,d,0) for d in (-16,-8,8,16)]+[np.rot90(a)]

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

def signflip(v,groups=None):
    v=np.asarray(v,float);ok=np.isfinite(v);v=v[ok]
    if not len(v):return np.nan,np.nan,0
    if groups is None:sums=v
    else:
        codes=pd.factorize(np.asarray(groups)[ok])[0];sums=np.bincount(codes,weights=v)
    ob=v.mean();random=rng('signflip');ge=0
    for start in range(0,10000,500):
        signs=random.choice([-1,1],size=(min(500,10000-start),len(sums)))
        vals=signs@sums/len(v);ge+=int((vals>=ob-1e-14).sum())
    return float(ob),(1+ge)/10001,len(v)

def read_image(p):
    p=used(p);a=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    assert a is not None and a.ndim==2 and a.shape==(2044,2048),str(p)
    return a
def init():
    verify();src,tabs,caches,raw=discover();f=fields()
    for p in (LOCAL/'lab_notes').iterdir():
        if p.is_file():used(p).read_text(encoding='utf-8-sig')
    for d in (OLD,R2,R3,R4,R5):
        for p in d.glob('*.md'):used(p).read_text(encoding='utf-8-sig')
    used(RESUME/'261004_周5_報告.md').read_text(encoding='utf-8-sig')
    paths=[]
    for folder in (ROOT/'field_level').iterdir():
        if folder.name in [d.name for d in (OLD,R2,R3,R4,R5)]:
            for p in folder.glob('*.py'):
                paths.append(dict(実在パス=str(p.resolve()),開始内容指紋=digest(p)))
    csv(paths,'prior_code_fingerprints.csv')
    dump(dict(予測内容指紋=PRED_HASH,入力起点=str(LOCAL),入力直下=[p.name for p in LOCAL.iterdir()],解析入力=str(src),解析直下=[p.name for p in src.iterdir()],生画像直下=list(raw),コピー完了印=[n for n,p in raw.items() if p.is_file() and 'copy_done' in n],視野数=len(f),固定29数=int(f['固定29視野'].sum()),版確認='開始時にv33はfield_level・data/results・入力解析直下・全ブランチ・タグに未使用',作業ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),開始コミット=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),開始状態='',環境=dict(Python=sys.version,数値配列=np.__version__,画像処理=cv2.__version__)),OUT/'preflight.json')
    dump(dict(パス=str((OUT/'261004_周6_予測と検定規定.md').resolve()),内容指紋=PRED_HASH),OUT/'prediction_sha256.json')
    print('preflight',len(f),flush=True)
def integrity():
    verify();rows=[json.loads(p.read_text(encoding='utf-8')) for p in (OUT/'input_fingerprints').glob('*.json')]
    rows+=pd.read_csv(OUT/'prior_code_fingerprints.csv').to_dict('records')
    for r in rows:r['終了内容指紋']=digest(r['実在パス']);r['不変']=r['開始内容指紋']==r['終了内容指紋']
    csv(rows,'input_integrity_final.csv');assert all(r['不変'] for r in rows)
    print('integrity',len(rows),flush=True)
if __name__=='__main__':{'init':init,'integrity':integrity}[sys.argv[1]]()
