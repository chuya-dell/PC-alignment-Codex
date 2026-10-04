"""Round 7 read-only discovery, band-limited camera model, and field metrics."""
import sys, os
sys.dont_write_bytecode = True
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib, json, time, subprocess, itertools
import numpy as np
import pandas as pd
import cv2
from scipy import fft
from scipy.signal import resample
from scipy.ndimage import map_coordinates
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'data/results/v35_band_origin_round8'
os.environ['MPLCONFIGDIR']=str(OUT/'plot_settings')
LOCAL = ROOT/'data/inputs_local'
PRED_HASH = '15015c89750339800fe55a8ce4c460559daee95aaa9248a358436565d98c3877'
METHODS = ['平均標準偏差', '中央値絶対偏差']
CORE = ['260926_7_5']
EPS = 1e-6
cv2.setNumThreads(1)
yy, xx = np.indices((64,64))
LABELS = ((xx//8)+(yy//8)) % 8
SPLITS = [(np.flatnonzero(~cv2.dilate((LABELS==k).astype('uint8'),np.ones((5,5),'uint8')).astype(bool)),np.flatnonzero(LABELS==k)) for k in range(8)]

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def dump(a,p):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=True,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
    os.replace(temp,p)

def verify():
    p=next(p for p in OUT.iterdir() if p.name.endswith('予測と検定規定.md'))
    assert digest(p)==PRED_HASH

def rng(s):
    return np.random.default_rng(20261004+int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little'))

def used(p):
    p=Path(p).resolve()
    assert LOCAL in p.parents or ROOT/'data/results' in p.parents or ROOT/'field_level' in p.parents
    assert OUT not in p.parents
    dest=OUT/'input_fingerprints';dest.mkdir(exist_ok=True)
    rec=dest/(hashlib.sha256(str(p).encode()).hexdigest()+'.json')
    if not rec.exists(): dump(dict(実在パス=str(p),開始内容指紋=digest(p),バイト数=p.stat().st_size),rec)
    return p

def table(p):
    return pd.read_csv(used(p),dtype={'日程':str,'基板':str,'date':str,'board':str})

def arrays(p):
    with np.load(used(p),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def record(p):return json.loads(used(p).read_text(encoding='utf-8-sig'))

def setup():
    children={p.name:p for p in LOCAL.iterdir()}
    src=next(p for n,p in children.items() if n.startswith('2026') and 'digital_judgment' in n)
    subs={p.name:p for p in src.iterdir()}
    tabs={p.name:p for p in subs['tables'].iterdir()}
    caches={p.stem:p for p in tabs['cached_field_differences'].iterdir() if p.suffix=='.npz'}
    raw={p.name:p for p in children['raw_readonly'].iterdir()}
    assert any('copy_done' in n and p.is_file() for n,p in raw.items())
    native={str(p.resolve()):p for d in raw.values() if d.is_dir() for p in d.iterdir() if p.is_file()}
    results={p.name:p for p in (ROOT/'data/results').iterdir() if p.is_dir()}
    r1=results['v28_band_origin_round1'];r5=results['v32_band_origin_round5']
    r1files={p.name:p for p in r1.iterdir()};r5sub={p.name:p for p in r5.iterdir()}
    resume={p.name:p for p in r5sub['resume_20261004'].iterdir()}
    f=table(r1files['round1_fields_both_definitions.csv']).query('定義 == @METHODS[0]').reset_index(drop=True)
    recs=table(resume['round5_recovery_combined.csv'])
    assert int(recs['回復'].sum())==266
    # Explicit precedence of the original per-field recovery checkpoints.
    recovery={}
    for d in (r1,r5,r5sub['resume_20261004']):
        dirs={p.name:p for p in d.iterdir() if p.is_dir()}
        if 'bf_v24_recovery' in dirs:
            for p in dirs['bf_v24_recovery'].iterdir():
                if p.suffix=='.json' and p.stem.startswith('26'):recovery[p.stem]=p
    recovered=sorted(recs.loc[recs['回復'],'key'])
    donor={k:recovered[(i+133)%266] for i,k in enumerate(recovered)}
    normals=[]
    rf=f[f.key.isin(recovered)]
    for date,g in rf[~rf['ブランク'] & (rf['外れ値割合']<.03)].groupby('日程'):
        g=g.copy();g['距離']=(g['外れ値割合']-g['外れ値割合'].median()).abs()
        normals+=g.sort_values(['距離','key']).head(3).key.tolist()
    selected=set(CORE)|set(rf.loc[rf['固定29視野']|rf['ブランク'],'key'])|set(normals)
    f['回復成功']=f.key.isin(recovered);f['必須対象']=f.key.isin(selected);f['通常選択']=f.key.isin(normals)
    f['指定3視野']=f.key.isin(CORE);f['供給元']=f.key.map(donor)
    return dict(children=children,src=src,tabs=tabs,caches=caches,raw=raw,native=native,results=results,r1files=r1files,resume=resume,f=f,recs=recs,recovery=recovery,donor=donor)

def original_code(state):
    code=state['children']['code_v24'];sub={p.name:p for p in code.iterdir()}
    assert 'shared' in sub
    # Enumerate and hash every Python dependency; imports cannot create bytecode.
    for p in sub['shared'].rglob('*.py'):used(p)
    sys.path.insert(0,str(code))
    from shared import registration
    from shared.v2_registration_precision.refinement import register_refined
    from shared.lattice_indexing import lattice_from_fft,grid_coordinates
    assert Path(registration.__file__).resolve().is_relative_to(code)
    return registration,register_refined,lattice_from_fft,grid_coordinates

def read_image(p):
    p=used(p);a=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    assert a is not None and a.ndim==2 and a.shape==(2044,2048)
    return a

def transform(xy,m):return xy@m[:,:2].T+m[:,2]

def phase_donor(m,d,shape):
    center=(np.array(shape[::-1])-1)/2
    motion=transform(center[None],m)[0]-center
    donor_motion=transform(center[None],d)[0]-center
    a=d[:,:2].copy()
    t=center+np.rint(motion)+(donor_motion-np.rint(donor_motion))-a@center
    return np.column_stack([a,t]).astype(np.float32)

def synthesize(pre,m,factor=4,identity_numerical=False):
    start=time.monotonic()
    ident=np.array([[1,0,0],[0,1,0]],float)
    if np.array_equal(m,ident) and not identity_numerical:
        return pre.copy(),dict(恒等厳密経路=True,制限画素割合=0.,補間倍率=factor,秒=time.monotonic()-start)
    padding=32
    a=np.pad(pre.astype(np.float32),padding,mode='reflect');h,w=a.shape
    fy=fft.fftfreq(h)[:,None];fx=fft.rfftfreq(w)[None,:]
    inv=np.linalg.inv(m[:,:2].astype(float))
    ratio=(np.sinc(fx*inv[0,0]+fy*inv[1,0])*np.sinc(fx*inv[0,1]+fy*inv[1,1])/(np.sinc(fx)*np.sinc(fy))).astype(np.float32)
    spectrum=fft.rfft2(a,workers=1);spectrum*=ratio
    filtered=fft.irfft2(spectrum,s=a.shape,workers=1)
    del a,spectrum,ratio
    # scipy.signal.resample handles splitting the even-length Nyquist bins.
    enlarged=resample(resample(filtered,w*factor,axis=1),h*factor,axis=0).astype(np.float32)
    del filtered
    ph,pw=pre.shape;out=np.empty(pre.shape,np.float32)
    for y in range(0,ph,64):
        sy,sx=np.indices((min(64,ph-y),pw),dtype=float);sy+=y
        coords=np.column_stack([sx.ravel(),sy.ravel()]);source=(coords-m[:,2])@inv.T
        sampled=map_coordinates(enlarged,[(source[:,1]+padding)*factor,(source[:,0]+padding)*factor],order=1,mode='reflect',prefilter=False)
        out[y:y+len(sy)]=sampled.reshape(sy.shape)
    clipped=float(((out<0)|(out>65535)).mean())
    raw=np.rint(np.clip(out,0,65535)).astype(np.uint16)
    del enlarged
    return raw,dict(恒等厳密経路=False,制限画素割合=clipped,補間倍率=factor,秒=time.monotonic()-start)

def sampler(raw,xy,bilinear=False):
    a=raw.astype(np.float32)/65535.;b=cv2.GaussianBlur(a,(51,51),0)
    h,w=a.shape;valid=np.isfinite(xy).all(axis=1)
    rounded=np.floor(xy) if bilinear else np.rint(xy)
    x,y=rounded.astype(int).T
    valid&=(x>=1)&(y>=1)&(x<w-(2 if bilinear else 1))&(y<h-(2 if bilinear else 1))
    intensity=np.zeros(len(xy));background=np.zeros(len(xy))
    tx,ty=(xy-rounded).T
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            terms=[(0,0,np.ones(len(xy)))] if not bilinear else [(0,0,(1-tx)*(1-ty)),(1,0,tx*(1-ty)),(0,1,(1-tx)*ty),(1,1,tx*ty)]
            for ox,oy,weight in terms:
                # Exact float32 nine-addition path when nearest-neighbour.
                if bilinear:
                    intensity[valid]+=a[y[valid]+dy+oy,x[valid]+dx+ox]*weight[valid]
                    background[valid]+=b[y[valid]+dy+oy,x[valid]+dx+ox]*weight[valid]
                else:
                    intensity[valid]+=a[y[valid]+dy,x[valid]+dx]
                    background[valid]+=b[y[valid]+dy,x[valid]+dx]
    value=intensity-background;value[~valid]=np.nan
    return value

def binned(xy,values):
    cells=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
    ok=np.isfinite(values);n=np.bincount(cells[ok],minlength=4096)
    s=np.bincount(cells[ok],weights=values[ok],minlength=4096)
    return np.divide(s,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64),n.reshape(64,64)

def shifted(a,dy,dx):
    out=np.full_like(a,np.nan,dtype=float)
    out[max(0,dy):min(64,64+dy),max(0,dx):min(64,64+dx)]=a[max(0,-dy):min(64,64-dy),max(0,-dx):min(64,64-dx)]
    return out

def controls(a):return [shifted(a,0,d) for d in (-16,-8,8,16)]+[shifted(a,d,0) for d in (-16,-8,8,16)]+[np.rot90(a)]

def corr(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3 or np.std(a[ok])<1e-12 or np.std(b[ok])<1e-12:return np.nan
    return float(np.corrcoef(a[ok],b[ok])[0,1])

def overlap(a,b):
    ok=np.isfinite(a)&np.isfinite(b);den=(a[ok]+b[ok]).sum()
    return float(2*np.minimum(a[ok],b[ok]).sum()/den) if den>0 else np.nan

def paired_map(a,b,function=corr):
    ctl=controls(b);valid=np.isfinite(a)&np.isfinite(b)
    for c in ctl:valid&=np.isfinite(c)
    av=np.where(valid,a,np.nan);bv=np.where(valid,b,np.nan)
    main=function(av,bv);vals=[function(av,np.where(valid,c,np.nan)) for c in ctl]
    median=float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
    return dict(全面=function(a,b),共通領域=main,対照中央値=median,対照差=main-median,共通升目数=int(valid.sum()),対照値=vals)

def variance_metrics(a,b):
    ok=np.isfinite(a)&np.isfinite(b)
    if ok.sum()<3:return {}
    va=np.var(a[ok]);vb=np.var(b[ok]);c=corr(a,b)
    return dict(実測分散=float(va),模擬分散=float(vb),分散比=float(vb/va) if va>0 else np.nan,無調整分散再現=float(1-np.var(a[ok]-b[ok])/va) if va>0 else np.nan,相関二乗=c*c)

def spectrum(a):
    valid=np.isfinite(a);y,x=np.indices(a.shape)
    if valid.sum()<100 or np.nanstd(a)<1e-12:return None
    A=np.column_stack([np.ones(valid.sum()),x[valid],y[valid]])
    co=np.linalg.lstsq(A,a[valid],rcond=None)[0]
    detrend=np.where(valid,a-(co[0]+co[1]*x+co[2]*y),0.)
    ft=fft.fft2(detrend*np.hanning(64)[:,None]*np.hanning(64)[None,:],s=(512,512))
    f=fft.fftfreq(512,d=32);fy,fx=np.meshgrid(f,f,indexing='ij');r=np.hypot(fx,fy)
    search=(r>=1/1024)&(r<=1/64)&((fy>0)|((fy==0)&(fx>0)))
    p=abs(ft)**2;masked=np.where(search,p,-np.inf);iy,ix=np.unravel_index(np.argmax(masked),p.shape)
    vectors=np.column_stack([fx[search],fy[search]]);weights=p[search]
    tensor=(vectors.T*weights)@vectors/weights.sum();eig,vec=np.linalg.eigh(tensor)
    normal=float(np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))%180)
    return dict(周期=float(1/r[iy,ix]),帯方向=float((np.degrees(np.arctan2(fy[iy,ix],fx[iy,ix]))+90)%180),第一軸帯方向=(normal+90)%180,軸異方性=float((eig[-1]-eig[0])/(eig.sum())),ピーク位置=(int(iy),int(ix)),変換=ft)

def angle(a,b):return float(abs((a-b+90)%180-90))

def spectral_pair(a,b):
    sa,sb=spectrum(a),spectrum(b)
    if sa is None or sb is None:return dict(三成分一致=False)
    iy,ix=sa['ピーク位置'];phase=float(np.degrees(np.angle(sb['変換'][iy,ix]*np.conj(sa['変換'][iy,ix]))))
    direction=angle(sa['帯方向'],sb['帯方向']);ratio=sb['周期']/sa['周期']
    return dict(実測周期=sa['周期'],模擬周期=sb['周期'],実測帯方向=sa['帯方向'],模擬帯方向=sb['帯方向'],方向差=direction,第一軸方向差=angle(sa['第一軸帯方向'],sb['第一軸帯方向']),周期比=ratio,位相差=phase,実測軸異方性=sa['軸異方性'],模擬軸異方性=sb['軸異方性'],三成分一致=bool(direction<=15 and .8<=ratio<=1.25 and abs(phase)<=45))

def sharpness(raw,xy):
    index=np.linspace(0,len(xy)-1,min(len(xy),5000)).astype(int)
    centers=np.rint(xy[index]).astype(int);h,w=raw.shape
    valid=(centers[:,0]>=3)&(centers[:,0]<w-3)&(centers[:,1]>=3)&(centers[:,1]<h-3)
    c=centers[valid];a=raw.astype(np.float32)/65535.;a-=cv2.GaussianBlur(a,(51,51),0)
    offsets=np.array([(dx,dy) for dy in (-1,0,1) for dx in (-1,0,1)])
    vals=np.stack([a[c[:,1]+dy,c[:,0]+dx] for dx,dy in offsets],axis=1)
    peaks=c+offsets[np.argmax(vals,axis=1)];v=a[peaks[:,1],peaks[:,0]]
    neigh=np.mean([a[peaks[:,1]+dy,peaks[:,0]+dx] for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))],axis=0)
    drop=(v[v>1e-6]-neigh[v>1e-6])/v[v>1e-6]
    return dict(ピラー数=len(drop),低下率中央値=float(np.median(drop)),低下率第1四分位=float(np.quantile(drop,.25)),低下率第3四分位=float(np.quantile(drop,.75)))

def signflip(values,groups=None):
    values=np.asarray(values,float);ok=np.isfinite(values);v=values[ok]
    if not len(v):return np.nan,np.nan,0
    sums=v if groups is None else np.bincount(pd.factorize(np.asarray(groups)[ok])[0],weights=v)
    ob=v.mean();hits=0;n=len(sums)
    if n<=18:
        for start in range(0,2**n,4096):
            bits=((np.arange(start,min(start+4096,2**n))[:,None]>>np.arange(n))&1)
            vals=(bits*2-1)@sums/len(v);hits+=int((vals>=ob-1e-14).sum())
        p=hits/(2**n)
    else:
        random=rng('signflip')
        for start in range(0,10000,500):
            vals=random.choice([-1,1],size=(500,n))@sums/len(v);hits+=int((vals>=ob-1e-14).sum())
        p=(hits+1)/10001
    return float(ob),float(p),len(v)

def integrity():
    verify();rows=[]
    for p in (OUT/'input_fingerprints').iterdir():
        r=json.loads(p.read_text(encoding='utf-8'));r['終了内容指紋']=digest(r['実在パス']);r['不変']=r['開始内容指紋']==r['終了内容指紋'];rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'input_integrity_final.csv',index=False,encoding='utf-8-sig')
    assert all(r['不変'] for r in rows)
    print('integrity',len(rows),flush=True)

old_setup = setup
def setup():
    s=old_setup();r5=s['results']['v32_band_origin_round5']
    subs={p.name:p for p in r5.iterdir()};resume={p.name:p for p in subs['resume_20261004'].iterdir()}
    gates=table(resume['round5_recovery_and_F_gates.csv'])
    eligible=set(gates.loc[gates['回復']&gates['F支持条件']&gates['F非周期目印確認'],'key'])
    assert len(eligible)==179
    feature={}
    for base in [s['results']['v28_band_origin_round1'],r5,subs['resume_20261004']]:
        children={p.name:p for p in base.iterdir()}
        for name in ['bf_models','F_features']:
            if name in children:
                for p in children[name].iterdir():
                    if p.suffix=='.npz':feature[p.stem]=p
    assert eligible<=set(feature)
    marker=table(resume['round5_F_marker_audit.csv']).set_index('key')
    original_marker=table(subs['round5_F_marker_audit.csv']).set_index('key')
    first_marker=table(s['r1files']['round1_bf_marker_phase_audit.csv']).set_index('key')
    marker=pd.concat([first_marker,original_marker,marker]).loc[lambda x:~x.index.duplicated(keep='last')]
    f=s['f'];normals=[]
    f['参考ブランク']=f['濃度'].eq('blank_reference')
    rf=f[f.key.isin(eligible)]
    for date,g in rf[~rf['ブランク']&~rf['参考ブランク']&(rf['外れ値割合']<.03)].groupby('日程'):
        g=g.copy();g['距離']=(g['外れ値割合']-g['外れ値割合'].median()).abs()
        normals+=g.sort_values(['距離','key']).head(3).key.tolist()
    required=set(rf.loc[rf['固定29視野']|rf['ブランク'],'key'])|set(normals)|set(CORE)
    f['適格']=f.key.isin(eligible);f['必須対象']=f.key.isin(required);f['通常選択']=f.key.isin(normals)
    f['指定3視野']=f.key.isin(CORE) # retained column spelling; now the one eligible requested field
    keys=sorted(eligible);donor={k:keys[(i+89)%179] for i,k in enumerate(keys)}
    f['供給元']=f.key.map(donor)
    s.update(f=f,eligible=eligible,feature=feature,marker=marker,donor=donor)
    return s
