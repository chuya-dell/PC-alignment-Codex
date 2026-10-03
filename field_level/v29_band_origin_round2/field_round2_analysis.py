"""Read-only, preregistered round-two comparison; no production imports."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os, json, hashlib, argparse, re, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import cv2
import tifffile

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT/'data'/'inputs_local'
OLD = ROOT/'data'/'results'/'v28_band_origin_round1'
OUT = ROOT/'data'/'results'/'v29_band_origin_round2'
METHODS = ['平均標準偏差', '中央値絶対偏差']
SEED = 20261004
cv2.setNumThreads(1)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()

def verify():
    a = pd.read_csv(OUT/'prediction_sha256.csv').iloc[0]
    assert digest(Path(a.Path)).upper() == a.Hash
    if (OUT/'completion_addendum_sha256.csv').exists():
        b=pd.read_csv(OUT/'completion_addendum_sha256.csv').iloc[0]
        assert digest(Path(b.Path)).upper()==b.Hash

def file_records():
    p=OUT/'round2_field_files_times_completed.csv'
    return table(p if p.exists() else OUT/'round2_field_files_times.csv')

def csv(a, name):
    pd.DataFrame(a).to_csv(OUT/name, index=False, encoding='utf-8-sig')

def table(p):
    return pd.read_csv(p, dtype={'日程':str, '基板':str, 'date':str, 'board':str})

def discover():
    children = list(LOCAL.iterdir())
    source = [p for p in children if p.is_dir() and p.name.startswith('2026') and 'digital_judgment' in p.name]
    assert len(source) == 1
    actual = {p.name:p for p in source[0].iterdir()}
    tables = {p.name:p for p in actual['tables'].iterdir()}
    raw = next(p for p in children if p.name == 'raw_readonly')
    roots = {p.name:p for p in raw.iterdir()}
    return children, source[0], tables, raw, roots

def fields(): return table(OLD/'round1_fields_both_definitions.csv')

def init():
    verify()
    children, source, tables, raw, roots = discover()
    f = table(OLD/'all_fields_existing_definition.csv')
    f['key'] = f['日程']+'_'+f['基板']+'_'+f['視野番号'].astype(str)
    qc = table(tables['table_registration_field_qc.csv'])
    times = table(next(p for p in children if p.name == 'raw_file_times_original.csv'))
    rec = table(OLD/'round1_bf_v24_recovery.csv').set_index('key')
    rawfiles = {}
    for name, folder in roots.items():
        if folder.is_dir() and name.startswith(('260926', '260830')):
            rawfiles[name] = {p.name:p for p in folder.iterdir() if p.is_file()}
    rows=[]; used=set()
    for _, r in f.iterrows():
        q = qc[(qc.date == r['日程']) & (qc.board == r['基板']) & (qc.field == r['視野番号'])]
        assert len(q) == 1
        q=q.iloc[0]
        before=str(q.path_pre).replace('\\','/').split('/')[-1]
        after=str(q.path_post).replace('\\','/').split('/')[-1]
        # Folder matching uses enumerated table values, never inferred spelling.
        for stage, filename in [('洗浄前',before), ('洗浄後',after)]:
            matches=times[(times.file == filename) & times.folder.str.startswith(r['日程'])]
            if len(matches)>1:
                oldfolder=str(q.root).replace('\\','/').split('/')[-1]
                matches=matches[matches.folder == oldfolder]
            row=dict(key=r.key, 日程=r['日程'], 基板=r['基板'], 視野番号=int(r['視野番号']), 濃度=r['濃度'], ブランク=r['ブランク'], 固定29視野=r['外れ値視野3percent以上'], 時点=stage, ファイル=filename,
                     時刻一意=len(matches)==1, 元更新時刻=None, 元作成時刻=None, 生画像パス=None, 回復=False)
            if len(matches)==1:
                t=matches.iloc[0];row.update(フォルダ=t.folder,元更新時刻=t.modified_local,元作成時刻=t.created_local,元バイト数=int(t.bytes))
            if r['日程']=='260926':
                folder=next(n for n in rawfiles if n.startswith('260926'))
                p=rawfiles[folder].get(filename)
                if p is None: raise RuntimeError('260926入力不足 '+filename)
                row['生画像パス']=str(p);row['回復']=bool(rec.loc[r.key,'回復']);used.add(p)
                if len(matches)==1: assert p.stat().st_size == int(matches.iloc[0].bytes)
            rows.append(row)
    csv(rows,'round2_field_files_times.csv')
    for n,fs in rawfiles.items():
        if n.startswith('260830'):used.update(p for p in fs.values() if p.suffix.lower() in ('.tif','.tiff'))
    oldnames=['all_fields_existing_definition.csv','round1_fields_both_definitions.csv','round1_spatial_with_metadata.csv', 'density_maps.npz','common_profile_residuals.npz','round1_bf_v24_recovery.csv','round1_bf_explained_fraction.csv']
    used.update(OLD/n for n in oldnames)
    used.add(tables['table_registration_field_qc.csv'])
    used.add(next(p for p in children if p.name=='raw_file_times_original.csv'))
    used.update(p for p in (ROOT/'field_level'/'v28_band_origin_round1').iterdir() if p.is_file())
    used.update(p for p in OLD.iterdir() if p.is_file() and ('予測' in p.name or 'sha256' in p.name or '実行整理' in p.name or 'NOTES' in p.name))
    # Actual cache names and recovered arrays are enumerated before use.
    caches={p.name:p for p in tables['cached_field_differences'].iterdir() if p.is_file()}
    recovery={p.name:p for p in (OLD/'bf_v24_recovery').iterdir() if p.is_file()}
    for _,r in f[f['日程']=='260926'].iterrows():
        used.add(caches[r.key+'.npz'])
        if bool(rec.loc[r.key,'回復']):used.add(recovery[r.key+'.npz'])
    notes=next(p for p in children if p.name=='lab_notes')
    used.update(p for p in notes.iterdir() if p.is_file())
    csv([dict(実在パス=str(p),バイト数=p.stat().st_size,開始内容指紋=digest(p)) for p in sorted(used)], 'round2_input_integrity.csv')
    info=dict(入力起点=str(LOCAL),解析フォルダ=str(source),生画像フォルダ=[str(roots[n]) for n in rawfiles],
              完了印=[n for n,p in roots.items() if p.is_file() and 'copy_done' in n],追加日程生画像実施=False,
              理由='開始時copy_done_2なし。別日程は元時刻表のみ利用。',視野数=len(f),入力ファイル数=len(used),予測指紋=pd.read_csv(OUT/'prediction_sha256.csv').iloc[0].Hash,
              数値環境=dict(Python=sys.version,数値配列=np.__version__,表処理=pd.__version__,画像処理=cv2.__version__,画像タグ読取=tifffile.__version__))
    (OUT/'round2_preflight.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    print('preflight',len(f),len(used),flush=True)

yy,xx=np.indices((64,64))
x=(xx+0.5)/32-1; y=(yy+0.5)/32-1
P=np.stack([x,y,x*x,y*y,x*y],axis=2)

def folds(valid):
    # Copied numerically from v28, without importing or executing that version.
    labels=((xx//8)+(yy//8))%8
    out=[]
    for k in range(8):
        test=(labels==k)&valid
        embargo=cv2.dilate((labels==k).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        out.append((np.flatnonzero(valid&~embargo),np.flatnonzero(test)))
    return out

def cv_score(response, X, valid):
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

def plane(a):
    d=np.column_stack([np.ones(4096),P.reshape(4096,5)]);v=np.isfinite(a.ravel())
    return (d@np.linalg.lstsq(d[v],a.ravel()[v],rcond=None)[0]).reshape(64,64)

def energy(a):
    v=np.isfinite(a); a=np.where(v,a-np.nanmean(a),0)
    pow=np.abs(np.fft.rfft2(a*np.outer(np.hanning(64),np.hanning(64))))**2
    freq=np.hypot(np.fft.fftfreq(64,d=32)[:,None],np.fft.rfftfreq(64,d=32)[None,:])
    mult=np.full_like(pow,2);mult[:,0]=1;mult[:,-1]=1
    return float((pow*mult)[(freq>=1/1024)&(freq<=1/128)].sum())

def score_models(key, response, models, meta):
    records=[];arrays={}
    # Same validity mask across G/I/joint for a meaningful signed partition.
    valid=np.isfinite(response)
    for X in models.values():valid &= np.isfinite(X).all(axis=2)
    common=valid.copy()
    for X in models.values():
        for c in controls(X):common &= np.isfinite(c).all(axis=2)
    for name,X in models.items():
        full,pr=cv_score(response,X,valid);real,prc=cv_score(response,X,common)
        scores=[cv_score(response,c,common)[0] for c in controls(X)]
        row=dict(key=key,モデル=name,全面決定係数=full,対照共通領域決定係数=real,対照決定係数中央値=float(np.nanmedian(scores)),対照との差=real-float(np.nanmedian(scores)),全面升目数=int(valid.sum()),対照共通升目数=int(common.sum()),**meta)
        for j,s in enumerate(scores):row[f'対照{j+1}決定係数']=s
        if name=='二次面':
            e=energy(response);row['周期帯エネルギー残存率']=energy(response-pr)/e if e>0 else np.nan
        records.append(row);arrays[name+'_prediction']=pr;arrays[name+'_common_prediction']=prc
    if '共同' in models:
        d={r['モデル']:r for r in records}
        for domain in ('全面決定係数','対照共通領域決定係数'):
            g=d['G'][domain];i=d['I'][domain];both=d['共同'][domain]
            for r in records:
                r[domain+'_G固有']=both-i;r[domain+'_I固有']=both-g;r[domain+'_共通']=g+i-both
    return records,arrays

def geometry(limit=0):
    verify();f=fields();g=f[f['定義']==METHODS[0]].reset_index(drop=True)
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual'];keys=list(z['keys'])
    assert keys==g.key.tolist()
    spatial=table(OLD/'round1_spatial_with_metadata.csv');spatial=spatial[spatial['縁除外画素']==0].set_index(['key','定義'])
    dest=OUT/'geometry';dest.mkdir(exist_ok=True)
    for j,r in g.iterrows():
        if limit and j>=limit:break
        done=dest/(r.key+'.json')
        if done.exists():continue
        rows=[];arrays={}
        for mi,method in enumerate(METHODS):
            meta={k:r[k] for k in ['日程','基板','濃度','ブランク','固定29視野']};meta.update(定義=method,分類=spatial.loc[(r.key,method),'分類'])
            rr,aa=score_models(r.key,res[j,mi],{'二次面':P},meta)
            rows+=rr;arrays.update({method+'_'+k:v for k,v in aa.items()})
        done.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        np.savez_compressed(dest/(r.key+'.npz'),**arrays)
        if j%50==0:print('geometry',j,r.key,flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'round2_geometry_explained_fraction.csv')

def block_mean(a):
    padded=np.full((2048,2048),np.nan,np.float32);padded[:2044]=a
    return np.nanmean(padded.reshape(64,32,64,32),axis=(1,3))

def read_image(p):
    a=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    assert a.shape==(2044,2048) and a.ndim==2
    return a.astype(np.float32)/65535

def image_maps(a):
    mu=block_mean(a);sd=np.sqrt(np.maximum(block_mean(a*a)-mu*mu,0))
    dx=np.zeros_like(a);dy=np.zeros_like(a)
    dx[:,:-1]=np.diff(a,axis=1);dy[:-1]=np.diff(a,axis=0)
    sharp=np.sqrt(block_mean(dx*dx+dy*dy))
    return np.stack([mu,sd/np.maximum(mu,1e-9),sharp/np.maximum(mu,1e-9)],axis=2)

def global_features(a):
    mu=float(a.mean());sd=float(a.std());sharp=float(np.sqrt((np.diff(a,axis=0)**2).mean()+(np.diff(a,axis=1)**2).mean()))
    py,px=np.indices(a.shape);edge=np.minimum.reduce([px,2047-px,py,2043-py])<200
    center=(np.abs(px-1024)<512)&(np.abs(py-1022)<512)
    return dict(平均輝度=mu,コントラスト=sd/max(mu,1e-9),鮮明さ=sharp/max(mu,1e-9),周辺中央比=float(a[edge].mean()/max(float(a[center].mean()),1e-9)),飽和画素割合=float((a>=1).mean()))

def peak_map(a,xy):
    bg=cv2.GaussianBlur(a,(51,51),0);p=np.rint(xy).astype(int);v=(p[:,0]>=1)&(p[:,0]<2047)&(p[:,1]>=1)&(p[:,1]<2043)
    pp=p[v];signal=np.zeros(len(pp));background=np.zeros(len(pp))
    for oy in (-1,0,1):
        for ox in (-1,0,1):
            signal+=a[pp[:,1]+oy,pp[:,0]+ox];background+=bg[pp[:,1]+oy,pp[:,0]+ox]
    ratio=signal/np.maximum(background,1e-9)-1
    cell=(xy[v,1]//32).astype(int)*64+(xy[v,0]//32).astype(int)
    n=np.bincount(cell,minlength=4096);s=np.bincount(cell,weights=ratio,minlength=4096)
    return np.divide(s,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)

def metadata(p):
    with tifffile.TiffFile(p) as tf:
        tags=[]; dt=[];exposure=[]
        for tag in tf.pages[0].tags.values():
            val=tag.value
            if tag.name in ('DateTime','ImageDescription','Software','Artist','Make','Model','ExposureTime','ExifTag') or 'Date' in tag.name or 'Time' in tag.name:
                item=dict(名前=tag.name,番号=int(tag.code),値=str(val));tags.append(item)
                if tag.name=='DateTime':dt.append(str(val))
                if 'Exposure' in tag.name:exposure.append(str(val))
        return dict(画像パス=str(p),画像内日時=dt,露光記録=exposure,関連タグ=tags)

def raw(limit=0):
    verify();a=file_records();pairs=a[a['生画像パス'].notna()].groupby('key',sort=True)
    dest=OUT/'raw_features';dest.mkdir(exist_ok=True)
    children,source,tables,root,roots=discover()
    caches={p.name:p for p in tables['cached_field_differences'].iterdir()}
    for j,(key,g) in enumerate(pairs):
        if limit and j>=limit:break
        done=dest/(key+'.json')
        if done.exists():continue
        start=time.monotonic();row=g.iloc[0]
        paths={r['時点']:Path(r['生画像パス']) for _,r in g.iterrows()}
        pre=read_image(paths['洗浄前']);post=read_image(paths['洗浄後'])
        pm=image_maps(pre);qm=image_maps(post)
        with np.load(caches[key+'.npz']) as z:xy=z['xy']
        peak=peak_map(pre,xy)
        gf0=global_features(pre);gf1=global_features(post)
        info=dict(key=key,日程=row['日程'],基板=row['基板'],濃度=row['濃度'],ブランク=bool(row['ブランク']),固定29視野=bool(row['固定29視野']),回復=bool(row['回復']),
                  洗浄前パス=str(paths['洗浄前']),洗浄後パス=str(paths['洗浄後']),洗浄前=gf0,洗浄後=gf1,
                  画像内記録=[metadata(paths[s]) for s in ('洗浄前','洗浄後')])
        arrays=dict(pre_maps=pm,post_native_maps=qm,pre_peak=peak)
        if bool(row['回復']):
            recovery={p.name:p for p in (OLD/'bf_v24_recovery').iterdir()}
            with np.load(recovery[key+'.npz']) as z:matrix=z['matrix']
            # Native 32-pixel statistic maps are interpolated to pre-image cell centers.
            coords=np.column_stack([(xx.ravel()+.5)*32,(yy.ravel()+.5)*32])@matrix[:,:2].T+matrix[:,2]
            mx=(coords[:,0]/32-.5).reshape(64,64).astype(np.float32);my=(coords[:,1]/32-.5).reshape(64,64).astype(np.float32)
            q=np.stack([cv2.remap(qm[:,:,k],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan')) for k in range(3)],axis=2)
            G=np.stack([plane(pm[:,:,k]) for k in range(3)]+[plane(peak)]+[plane(pm[:,:,k]-q[:,:,k]) for k in range(3)],axis=2)
            changes=np.array([np.log(gf0[k]/gf1[k]) for k in ('平均輝度','コントラスト','鮮明さ')])
            I=pm*changes[None,None,:]
            arrays.update(G=G,I=I,post_maps=q,global_log_changes=changes)
        info['秒']=time.monotonic()-start
        np.savez_compressed(dest/(key+'.npz'),**arrays)
        done.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print('raw',j,key,info['回復'],round(info['秒'],2),flush=True)
    # Control images: no before/after assertions.
    control=next(p for n,p in roots.items() if p.is_dir() and n.startswith('260830'))
    for p in sorted(control.iterdir()):
        if p.suffix.lower() not in ('.tif','.tiff'):continue
        done=dest/('control_'+p.stem+'.json')
        if done.exists():continue
        im=read_image(p);m=image_maps(im)
        info=dict(key='260830_'+p.stem,画像パス=str(p),対応='不明。単独像のみ',全体=global_features(im),画像内記録=[metadata(p)])
        np.savez_compressed(dest/('control_'+p.stem+'.npz'),maps=m,planes=np.stack([plane(m[:,:,k]) for k in range(3)],axis=2))
        done.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    export_raw()

def export_raw():
    dest=OUT/'raw_features';rows=[];meta=[]
    for p in sorted(dest.glob('*.json')):
        a=json.loads(p.read_text(encoding='utf-8'));meta+=a['画像内記録']
        if p.name.startswith('control_'):continue
        row={k:a[k] for k in ['key','日程','基板','濃度','ブランク','固定29視野','回復','洗浄前パス','洗浄後パス']}
        for stage in ('洗浄前','洗浄後'):
            for k,v in a[stage].items():row[stage+k]=v
        for k in ('平均輝度','コントラスト','鮮明さ'):
            row[k+'前後対数比']=np.log(a['洗浄前'][k]/a['洗浄後'][k])
        rows.append(row)
    csv(rows,'round2_global_image_features.csv')
    (OUT/'round2_image_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')

def optical(limit=0):
    verify();f=fields();g=f[f['定義']==METHODS[0]].reset_index(drop=True);index={k:i for i,k in enumerate(g.key)}
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual']
    spatial=table(OLD/'round1_spatial_with_metadata.csv');spatial=spatial[spatial['縁除外画素']==0].set_index(['key','定義'])
    dest=OUT/'optical_evaluation';dest.mkdir(exist_ok=True)
    selected=table(OUT/'round2_global_image_features.csv');selected=selected[selected['回復']]
    if limit:selected=selected.iloc[:limit]
    for _,r in selected.iterrows():
        key=r.key;done=dest/(key+'.json')
        if done.exists():continue
        with np.load(OUT/'raw_features'/(key+'.npz')) as z:G=z['G'];I=z['I']
        records=[];arrays={}
        for mi,method in enumerate(METHODS):
            meta={k:r[k] for k in ['日程','基板','濃度','ブランク','固定29視野']};meta.update(定義=method,分類=spatial.loc[(key,method),'分類'])
            rr,aa=score_models(key,res[index[key],mi],{'G':G,'I':I,'共同':np.concatenate([G,I],axis=2)},meta)
            records+=rr;arrays.update({method+'_'+k:v for k,v in aa.items()})
        done.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');np.savez_compressed(dest/(key+'.npz'),**arrays)
        print('optical',key,flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'round2_optical_explained_fraction.csv')

def time_features():
    a=file_records();rows=[]
    for key,g in a.groupby('key',sort=False):
        base=g.iloc[0];row={k:base[k] for k in ['key','日程','基板','視野番号','濃度','ブランク','固定29視野']}
        for stage in ('洗浄前','洗浄後'):
            r=g[g['時点']==stage].iloc[0];row[stage+'元更新時刻']=r['元更新時刻'] if r['時刻一意'] else None
        rows.append(row)
    a=pd.DataFrame(rows)
    for s in ('洗浄前','洗浄後'):
        dt=pd.to_datetime(a[s+'元更新時刻']);a[s+'秒']=dt.astype('int64')/1e9;a.loc[dt.isna(),s+'秒']=np.nan
        a[s+'順位']=a.groupby(['日程','基板'])[s+'秒'].rank(method='average')
        n=a.groupby(['日程','基板'])[s+'秒'].transform('count')
        a[s+'順位']=(a[s+'順位']-1)/np.maximum(n-1,1)
    a['撮影間隔時間']=(a['洗浄後秒']-a['洗浄前秒'])/3600
    a=a.merge(table(OUT/'round2_global_image_features.csv'),on=['key','日程','基板','濃度','ブランク','固定29視野'],how='left')
    csv(a,'round2_field_time_features.csv');return a

def design(a):
    group=a['日程']+'_'+a['基板']
    return np.column_stack([np.ones(len(a)),pd.get_dummies(group).to_numpy(float),pd.get_dummies(a['視野番号'].astype(str)).to_numpy(float)])

def residual(a,C):return a-C@np.linalg.lstsq(C,a,rcond=None)[0]

def maxcorr(a,b):
    den=np.sqrt((a*a).sum(axis=0)[:,None]*(b*b).sum(axis=0)[None,:])
    r=np.divide(a.T@b,den,out=np.zeros_like(den),where=den>1e-24)
    return float(np.abs(r).max()),r

def association(a, acol,bcol,label,method):
    s=a.dropna(subset=acol+bcol).reset_index(drop=True)
    if len(s)<10:return dict(検定=label,定義=method,視野数=len(s),補正前有意確率=np.nan,状態='不足')
    C=design(s);Q=np.linalg.svd(C,full_matrices=False)[0][:,:np.linalg.matrix_rank(C)]
    A=residual(s[acol].to_numpy(float),C);B=residual(s[bcol].to_numpy(float),C)
    observed,r=maxcorr(A,B);rng=np.random.default_rng(SEED);hits=0
    groups=list(s.groupby(['日程','基板']).indices.values())
    # Permute whole field rows, with the same permutation for all outcomes.
    for batch in range(100):
        idx=np.tile(np.arange(len(s)),(100,1))
        for block in groups:
            for row in idx:row[block]=rng.permutation(block)
        perm=B[idx];perm-=np.einsum('nk,bkm->bnm',Q,np.einsum('kn,bnm->bkm',Q.T,perm))
        num=np.einsum('nc,bnd->bcd',A,perm)
        den=np.sqrt((A*A).sum(axis=0)[None,:,None]*(perm*perm).sum(axis=1)[:,None,:])
        rr=np.divide(num,den,out=np.zeros_like(num),where=den>1e-24)
        hits+=int((np.max(np.abs(rr),axis=(1,2))>=observed-1e-15).sum())
    return dict(検定=label,定義=method,視野数=len(s),日程数=s['日程'].nunique(),基板数=len(groups),統計量=observed,補正前有意確率=(1+hits)/10001,ボンフェローニ補正後=min(1,20*(1+hits)/10001),状態='実施',相関配列=json.dumps(r.tolist()),変数左=';'.join(acol),変数右=';'.join(bcol))

def sign_test(v,groups=None):
    v=np.asarray(v,float);ok=np.isfinite(v);v=v[ok]
    if not len(v):return np.nan
    if groups is None:g=np.arange(len(v));n=len(v)
    else:_,g=np.unique(np.asarray(groups)[ok],return_inverse=True);n=g.max()+1
    rng=np.random.default_rng(SEED);hits=0;observed=v.mean()
    for _ in range(100):
        signs=rng.choice(np.array([-1,1],np.int8),size=(100,n));stats=signs[:,g]@v/len(v)
        hits+=int((stats>=observed-1e-15).sum())
    return (hits+1)/10001

def field_cv(a,extra):
    y=a['外れ値割合'].to_numpy(float);C=design(a);X=a[extra].to_numpy(float) if extra else np.empty((len(a),0));pred=[]
    for j in range(len(a)):
        tr=np.arange(len(a))!=j
        xm=X[tr].mean(axis=0);xs=X[tr].std(axis=0);xs[xs<1e-12]=1
        train=np.column_stack([C[tr],(X[tr]-xm)/xs]);test=np.r_[C[j],(X[j]-xm)/xs]
        pen=np.r_[np.full(C.shape[1],1e-9),np.ones(len(extra))]
        coef=np.linalg.solve(train.T@train+np.diag(pen),train.T@y[tr]);pred.append(test@coef)
    return np.array(pred)

def tests():
    verify();geom=table(OUT/'round2_geometry_explained_fraction.csv');opt=table(OUT/'round2_optical_explained_fraction.csv');rows=[]
    for method in METHODS:
        for model,df,metric in [('二次面',geom,'全面決定係数'),('G',opt,'対照との差'),('I',opt,'対照との差')]:
            for outliers in (False,True):
                s=df[(df['定義']==method)&(df['モデル']==model)]
                if outliers:s=s[s['固定29視野']]
                values=s[metric].to_numpy();p=sign_test(values);cluster=sign_test(values,(s['日程']+'_'+s['基板']).to_numpy())
                rows.append(dict(検定=model+'空間_'+('固定29利用可能' if outliers else '全利用可能'),定義=method,視野数=len(s),統計量=float(np.nanmean(values)),補正前有意確率=p,ボンフェローニ補正後=min(1,p*20),日程基板反転有意確率=cluster,状態='実施'))
        s=geom[(geom['定義']==method)&geom['固定29視野']&(geom['分類']=='帯状候補')]
        v=s['周期帯エネルギー残存率'].to_numpy()-.8;p=sign_test(v)
        rows.append(dict(検定='二次面後の周期エネルギー80%以上',定義=method,視野数=len(s),統計量=float(np.nanmean(v)) if len(v) else np.nan,補正前有意確率=p,ボンフェローニ補正後=min(1,p*20) if np.isfinite(p) else np.nan,日程基板反転有意確率=sign_test(v,(s['日程']+'_'+s['基板']).to_numpy()),状態='実施' if len(s) else '帯状候補なし'))
    timea=time_features();f=fields();a=timea.merge(f[['key','定義','外れ値割合']],on='key')
    tcols=['洗浄前順位','洗浄後順位','撮影間隔時間'];changes=[k+'前後対数比' for k in ('平均輝度','コントラスト','鮮明さ')]
    for method in METHODS:
        s=a[a['定義']==method]
        for left,right,label in [(tcols,['外れ値割合'],'時刻と外れ値割合'),(changes,['外れ値割合'],'画像条件変化と外れ値割合'),(tcols,changes,'時刻と画像条件変化')]:
            rows.append(association(s,left,right,label,method));print('test',label,method,flush=True)
    assert len(rows)==20;csv(rows,'round2_confirmatory_tests.csv')
    cvrows=[];predrows=[]
    for method in METHODS:
        s=a[a['定義']==method];gcols=['洗浄前コントラスト','洗浄前鮮明さ','洗浄前周辺中央比']
        for domain,cols in [('時刻利用可能',tcols),('画像利用可能',gcols+changes+tcols)]:
            ss=s.dropna(subset=cols).reset_index(drop=True);base=field_cv(ss,[]);yy=ss['外れ値割合'].to_numpy();bsse=((yy-base)**2).sum();total=((yy-yy.mean())**2).sum()
            models={'I時刻':tcols} if domain=='時刻利用可能' else {'G':gcols,'I':changes+tcols,'共同':gcols+changes+tcols}
            result=[]
            for name,features in models.items():
                pred=field_cv(ss,features);sse=((yy-pred)**2).sum()
                row=dict(定義=method,対象=domain,モデル=name,視野数=len(ss),基準追加決定係数=1-sse/bsse,全分散決定係数=1-sse/total,基準全分散決定係数=1-bsse/total)
                result.append(row)
                for j,r in ss.iterrows():predrows.append(dict(key=r.key,定義=method,対象=domain,モデル=name,実測=yy[j],予測=pred[j],基準予測=base[j]))
            if '共同' in models:
                d={r['モデル']:r['基準追加決定係数'] for r in result}
                for r in result:r.update(G固有=d['共同']-d['I'],I固有=d['共同']-d['G'],共通=d['G']+d['I']-d['共同'])
            cvrows+=result
    csv(cvrows,'round2_field_rate_cv_summary.csv');csv(predrows,'round2_field_rate_predictions.csv')

def validate():
    verify();valid=np.ones((64,64),bool)
    gradient=.05*x+.02*y+.04*x*x
    band=.05*np.sin(2*np.pi*(xx+.5)*32/448)
    s,pr=cv_score(gradient,P,valid);sb,pb=cv_score(band,P,valid)
    split=folds(valid)
    assert s>.99 and sb<.1 and energy(band-pb)/energy(band)>.8
    for tr,te in split:
        assert not set(tr)&set(te)
        assert not np.any(cv2.dilate(np.isin(np.arange(4096),te).reshape(64,64).astype(np.uint8),np.ones((5,5),np.uint8)).ravel()[tr])
    moved=shifted(P,0,16);assert np.isnan(moved[:,:16]).all()
    # Independent implementation against the copied score, avoiding legacy execution.
    errors=[]
    for k,(tr,te) in enumerate(split):
        X=P.reshape(4096,5);m=X[tr].mean(0);sd=X[tr].std(0);a=(X[tr]-m)/sd;b=(X[te]-m)/sd;ym=gradient.ravel()[tr].mean()
        aug=np.vstack([a,np.eye(5)]);target=np.r_[gradient.ravel()[tr]-ym,np.zeros(5)]
        coef=np.linalg.lstsq(aug,target,rcond=None)[0];errors.append(float(np.max(np.abs(ym+b@coef-pr.ravel()[te]))))
    assert max(errors)<1e-10
    out=dict(二次勾配決定係数=s,周期帯決定係数=sb,周期帯エネルギー残存率=energy(band-pb)/energy(band),独立解最大差=max(errors),分割と学習余白=True,非循環移動=True)
    (OUT/'round2_numerical_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','validate','geometry','raw','optical','tests']);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    if args.stage in ('geometry','raw','optical'):globals()[args.stage](args.limit)
    else:globals()[args.stage]()
