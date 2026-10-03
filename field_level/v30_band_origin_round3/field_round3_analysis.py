"""Preregistered descriptive validation; never changes pillar selection or production."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage, stats

cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT/'data'/'inputs_local'
OLD = ROOT/'data'/'results'/'v28_band_origin_round1'
PREVIOUS = ROOT/'data'/'results'/'v29_band_origin_round2'
OUT = ROOT/'data'/'results'/'v30_band_origin_round3'
METHODS = ['平均標準偏差','中央値絶対偏差']
RADII = [16,32,64,128,256]
EDGES = np.array([0,16,32,64,128,256,512,np.inf])
SEED = 20261004
yy,xx = np.indices((64,64))
CELLXY = np.stack([(xx+.5)*32,(yy+.5)*32],axis=2)

def table(p):
    return pd.read_csv(p,dtype={'日程':str,'基板':str,'date':str,'board':str})

def csv(rows,name):
    pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')

def dump(a,p):
    p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def verify():
    r=pd.read_csv(OUT/'prediction_sha256.csv').iloc[0]
    assert digest(Path(r.Path)).upper()==r.Hash

def discover():
    children={p.name:p for p in LOCAL.iterdir()}
    source=[p for n,p in children.items() if p.is_dir() and n.startswith('2026') and 'digital_judgment' in n]
    assert len(source)==1
    dirs={p.name:p for p in source[0].iterdir()}
    tables={p.name:p for p in dirs['tables'].iterdir()}
    caches={p.stem:p for p in tables['cached_field_differences'].iterdir() if p.suffix=='.npz'}
    raw={p.name:p for p in children['raw_readonly'].iterdir()}
    return children,source[0],tables,caches,raw

def fields():return table(OLD/'round1_fields_both_definitions.csv')

def spatial():
    a=table(OLD/'round1_spatial_with_metadata.csv')
    return a[a['縁除外画素']==0].copy()

def init():
    verify();children,source,tables,caches,raw=discover()
    markers=[p for n,p in raw.items() if p.is_file() and 'copy_done' in n]
    assert markers,'コピー完了印なし'
    qc=table(tables['table_registration_field_qc.csv'])
    times=table(children['raw_file_times_original.csv'])
    f=fields();base=f[f['定義']==METHODS[0]]
    existing={n:{p.name:p for p in d.iterdir() if p.is_file()} for n,d in raw.items() if d.is_dir()}
    rec=table(OLD/'round1_bf_v24_recovery.csv').set_index('key')
    recoveries={p.stem:p for p in (OLD/'bf_v24_recovery').iterdir() if p.suffix=='.npz'}
    used=set(markers)|set(caches.values());rows=[]
    for _,r in base.iterrows():
        assert r.key in caches
        q=qc[(qc.date==r['日程'])&(qc.board==r['基板'])&(qc.field==r['視野番号'])]
        assert len(q)==1
        q=q.iloc[0];meta={k:r[k] for k in ['key','日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野']}
        row=dict(**meta,生画像組利用可能=False,保存差再現群=False)
        for label,col in [('洗浄前','path_pre'),('洗浄後','path_post')]:
            filename=str(q[col]).replace('\\','/').split('/')[-1]
            found=[fs[filename] for n,fs in existing.items() if n.startswith(r['日程']) and filename in fs]
            assert len(found)<=1,(r.key,label,found)
            row[label+'パス']=str(found[0]) if found else None
            match=times[(times.file==filename)&times.folder.str.startswith(r['日程'])]
            row[label+'元時刻一意']=len(match)==1
            row[label+'元更新時刻']=str(match.iloc[0].modified_local) if len(match)==1 else None
            if found:
                used.add(found[0])
                if len(match)==1:assert found[0].stat().st_size==match.iloc[0].bytes
        row['生画像組利用可能']=bool(row['洗浄前パス'] and row['洗浄後パス'])
        if r.key in rec.index and bool(rec.loc[r.key,'回復']):
            row['保存差再現群']=True;row['対応行列パス']=str(recoveries[r.key]);used.add(recoveries[r.key])
        rows.append(row)
    controls=[]
    for n,fs in existing.items():
        if n.startswith('260830'):
            for filename,p in fs.items():
                if p.suffix.lower() in ('.tif','.tiff'):
                    controls.append(dict(key='260830_'+p.stem,画像パス=str(p)));used.add(p)
    csv(rows,'round3_field_manifest.csv');csv(controls,'round3_control_manifest.csv')
    for d in (OLD,PREVIOUS):
        used.update(p for p in d.iterdir() if p.is_file())
        used.update(p for p in (ROOT/'field_level'/d.name).iterdir() if p.is_file())
    used.update([tables['table_registration_field_qc.csv'],children['raw_file_times_original.csv']])
    notes=list(children['lab_notes'].iterdir())
    documents=list(source.rglob('*.md'))+notes
    search=[]
    for p in documents:
        if not p.is_file():continue
        used.add(p)
        for number,line in enumerate(p.read_text(encoding='utf-8-sig').splitlines(),1):
            if any(w in line for w in ['水流','風向','ブロワ','乾燥','視野番号','物理的位置','位置6','260916']):
                search.append(dict(実在パス=str(p),行番号=number,記載=line))
    csv(search,'round3_record_search.csv')
    csv([dict(実在パス=str(p),バイト数=p.stat().st_size,開始内容指紋=digest(p)) for p in sorted(used)],'round3_input_integrity.csv')
    dump(dict(入力起点=str(LOCAL),解析コピー=str(source),生画像フォルダ=[str(p) for p in raw.values() if p.is_dir()],完了印=[str(p) for p in markers],保存差視野数=len(base),生画像組数=sum(r['生画像組利用可能'] for r in rows),保存差再現群数=sum(r['保存差再現群'] for r in rows),コントロール単独像数=len(controls),指紋記録数=len(used),実行環境=dict(Python=sys.version,数値配列=np.__version__,表処理=pd.__version__,画像処理=cv2.__version__)),OUT/'round3_preflight.json')
    print('preflight',len(base),sum(r['生画像組利用可能'] for r in rows),len(used),flush=True)

# The following validation geometry is copied numerically from v28/v29;
# neither prior version nor production code is imported or executed.
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

def score_models(response,models,meta):
    valid=np.isfinite(response)
    for X in models.values():valid &= np.isfinite(X).all(axis=2)
    common=valid.copy()
    for X in models.values():
        for c in controls(X):common &= np.isfinite(c).all(axis=2)
    rows=[];arrays={}
    for name,X in models.items():
        full,pr=cv_score(response,X,valid);real,pc=cv_score(response,X,common)
        scores=[cv_score(response,c,common)[0] for c in controls(X)]
        med=float(np.nanmedian(scores)) if np.isfinite(scores).any() else np.nan
        row=dict(**meta,モデル=name,全面決定係数=full,対照共通領域決定係数=real,対照決定係数中央値=med,対照との差=real-med,全面升目数=int(valid.sum()),対照共通升目数=int(common.sum()))
        row.update({f'対照{j+1}決定係数':s for j,s in enumerate(scores)})
        rows.append(row);arrays[name+'_prediction']=pr;arrays[name+'_common_prediction']=pc
    if all(n in models for n in ('H','C','共同')):
        d={r['モデル']:r for r in rows}
        for domain in ('全面決定係数','対照共通領域決定係数'):
            h=d['H'][domain];c=d['C'][domain];joint=d['共同'][domain]
            for r in rows:r.update({domain+'_H固有':joint-c,domain+'_C固有':joint-h,domain+'_共通':h+c-joint})
    return rows,arrays

def read_image(p):
    a=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    assert a.ndim==2 and a.shape==(2044,2048)
    return a.astype(np.float32)/65535

def sample(a,xy,border=np.nan):
    shape=xy.shape[:-1]
    x=xy[...,0].reshape(-1,1).astype(np.float32)
    y=xy[...,1].reshape(-1,1).astype(np.float32)
    # OpenCV remap dimensions are limited to 32767, so sample vectors in chunks.
    values=[]
    for j in range(0,len(x),16000):
        values.append(cv2.remap(a.astype(np.float32),x[j:j+16000],y[j:j+16000],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float(border)).ravel())
    return np.concatenate(values).reshape(shape)

def component_distance(mask):
    if not mask.any():return np.full(mask.shape,8192,np.float32)
    return (ndimage.distance_transform_edt(~mask)*4).astype(np.float32)

def candidates(image):
    smooth=cv2.GaussianBlur(image,(0,0),3)[::4,::4]
    residual=smooth-cv2.GaussianBlur(smooth,(0,0),16)
    mid=float(np.median(residual));scale=float(1.4826*np.median(np.abs(residual-mid)))
    z=(residual-mid)/max(scale,1e-9)
    allmask=np.zeros(z.shape,np.uint8);objects=[]
    for sign in (1,-1):
        n,labels,box,centers=cv2.connectedComponentsWithStats((sign*z>=6).astype(np.uint8),connectivity=8)
        for k in range(1,n):
            if box[k,cv2.CC_STAT_AREA]<16:continue
            cy,cx=np.where(labels==k);coords=np.column_stack([cx,cy])*4
            cov=np.cov(coords.T);vals=np.maximum(np.linalg.eigvalsh(cov),1e-9)
            length=4*np.sqrt(vals[-1]);ratio=np.sqrt(vals[-1]/vals[0])
            kind='線状' if length>=128 and ratio>=4 else ('明点' if sign==1 else '暗部')
            index=len(objects)+1;allmask[cy,cx]=1 if sign==1 else 2
            objects.append(dict(候補番号=index,種別=kind,明暗='明' if sign==1 else '暗',面積画素=int(len(cx)*16),中心横画素=float(centers[k,0]*4),中心縦画素=float(centers[k,1]*4),長軸画素=float(length),長短軸比=float(ratio)))
    distance=component_distance(allmask>0)
    profile=[]
    for low,high in zip(EDGES[:-1],EDGES[1:]):
        v=(distance>=low)&(distance<high)
        profile.append(float(np.median(np.abs(z[v]))) if v.any() and objects else np.nan)
    radius=np.nan
    for j in range(1,len(profile)-1):
        if profile[j]<1.5 and profile[j+1]<1.5:radius=float(EDGES[j]);break
    # Spectrum is descriptive and uses only this image's low-frequency structure.
    block=cv2.resize(smooth,(64,64),interpolation=cv2.INTER_AREA)
    py,px=np.indices(block.shape);design=np.stack([np.ones_like(px),px,py,px*px,py*py,px*py],axis=-1).reshape(-1,6)
    block=block-(design@np.linalg.lstsq(design,block.ravel(),rcond=None)[0]).reshape(64,64)
    power=np.abs(np.fft.fft2(block*np.outer(np.hanning(64),np.hanning(64))))**2
    fy,fx=np.meshgrid(np.fft.fftfreq(64,d=32),np.fft.fftfreq(64,d=32),indexing='ij')
    freq=np.hypot(fx,fy);v=(freq>=1/1024)&(freq<=1/128)
    angle=(np.rad2deg(np.arctan2(fy,fx))+90)%180
    bins=np.bincount((angle[v]//10).astype(int),weights=power[v],minlength=18)
    w=power[v];vector=np.sum(w*np.exp(2j*np.deg2rad(angle[v])))/max(w.sum(),1e-20)
    info=dict(候補数=len(objects),明候補数=sum(o['明暗']=='明' for o in objects),暗候補数=sum(o['明暗']=='暗' for o in objects),線状候補数=sum(o['種別']=='線状' for o in objects),候補画素割合=float((allmask>0).mean()),残差換算絶対偏差=scale,影響半径代理画素=radius,影響半径状態='推定' if np.isfinite(radius) else ('候補なし' if not objects else '測定範囲内で未確定'),単独像帯軸度=float(np.argmax(bins)*10+5),単独像方向集中度=float(abs(vector)),単独像周期帯パワー=float(w.sum()),距離殻絶対残差中央値=profile)
    return dict(info=info,objects=objects,mask=allmask,z=z,distance=distance)

def registration(pre,post):
    a=cv2.resize(cv2.GaussianBlur(pre,(0,0),3),(512,511),interpolation=cv2.INTER_AREA)
    b=cv2.resize(cv2.GaussianBlur(post,(0,0),3),(512,511),interpolation=cv2.INTER_AREA)
    shift,response=cv2.phaseCorrelate(a,b)
    matrix=np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32)
    info=dict(位相相関応答=float(response),独立対応=True)
    try:
        corr,matrix=cv2.findTransformECC(a,b,matrix,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-5),None,5)
        full=matrix.astype(float);full[:,2]*=4
        det=float(np.linalg.det(full[:,:2]));move=float(np.linalg.norm((full[:,:2]-np.eye(2))@np.array([1024,1022])+full[:,2]))
        accepted=bool(corr>=.90 and move<64 and .97<=det<=1.03 and abs(full[0,1])<=.03 and abs(full[1,0])<=.03)
        info.update(対応相関=float(corr),行列式=det,中心移動画素=move,対応採用=accepted,対応理由='規定を満たす' if accepted else '規定を満たさない')
        return full,info
    except cv2.error as error:
        info.update(対応採用=False,対応理由='相関最大化失敗',詳細=str(error).splitlines()[0]);return None,info

def distance_features(d):return np.stack([np.exp(-d/r) for r in RADII],axis=-1)

def density_shells(key,cache,cand,pre_cand,matrix,metadata,f):
    with np.load(cache) as z:xy=z['xy'];delta=z['delta']
    transformed=xy@matrix[:,:2].T+matrix[:,2]
    coord=transformed/4
    postdistance=sample(cand['distance'],coord)
    predistance=sample(pre_cand['distance'],xy/4)
    rows=[]
    # Candidate types are descriptive alternatives, never selection masks.
    ds={'後全候補':postdistance,'前全候補':predistance}
    for value,label in [(1,'後明候補'),(2,'後暗候補')]:
        if np.any(cand['mask']==value):ds[label]=sample(component_distance(cand['mask']==value),coord)
    inverse=cv2.invertAffineTransform(matrix)
    postgrid=np.stack(np.meshgrid(np.arange(512)*4,np.arange(511)*4),axis=-1)
    pre_at_post=sample(pre_cand['distance'],(postgrid@inverse[:,:2].T+inverse[:,2])/4)
    for label,mask in [('後既存候補',(cand['mask']>0)&(pre_at_post<=32)),('後新規移動候補',(cand['mask']>0)&(pre_at_post>32))]:
        if mask.any():ds[label]=sample(component_distance(mask),coord)
    for _,r in f[f.key==key].iterrows():
        pos=delta>r['閾値']
        for label,d in ds.items():
            for k,(low,high) in enumerate(zip(EDGES[:-1],EDGES[1:])):
                v=np.isfinite(d)&(d>=low)&(d<high)
                n=int(v.sum());positive=int(pos[v].sum())
                rows.append(dict(**metadata,定義=r['定義'],候補区分=label,殻番号=k,距離下限画素=float(low),距離上限画素=float(high),ピラー数=n,外れ値数=positive,外れ値密度=positive/n if n else np.nan))
    return rows

def raw(limit=0):
    verify();_,_,_,caches,_=discover();a=table(OUT/'round3_field_manifest.csv');f=fields()
    selected=a[a['生画像組利用可能']].copy()
    if limit:selected=pd.concat([selected.head(1),selected[selected['保存差再現群']].head(1)]).drop_duplicates('key')
    dest=OUT/'raw_candidates';dest.mkdir(exist_ok=True)
    for j,(_,r) in enumerate(selected.iterrows()):
        done=dest/(r.key+'.json')
        if done.exists():continue
        started=time.monotonic();pre=read_image(Path(r['洗浄前パス']));post=read_image(Path(r['洗浄後パス']))
        pc=candidates(pre);qc=candidates(post)
        metadata={k:r[k] for k in ['key','日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野','保存差再現群']}
        if r['保存差再現群']:
            with np.load(Path(r['対応行列パス'])) as z:matrix=z['matrix']
            reg=dict(対応採用=True,独立対応=False,対応理由='周1保存差再現行列')
        else:matrix,reg=registration(pre,post)
        info=dict(**metadata,洗浄前パス=r['洗浄前パス'],洗浄後パス=r['洗浄後パス'],洗浄前=pc['info'],洗浄後=qc['info'],対応=reg)
        objects=[dict(**metadata,時点=stage,**o) for stage,c in [('洗浄前',pc),('洗浄後',qc)] for o in c['objects']]
        arrays=dict(pre_mask=pc['mask'],post_mask=qc['mask'],pre_z=pc['z'],post_z=qc['z'],pre_distance=pc['distance'],post_native_distance=qc['distance'])
        shellrows=[]
        if reg['対応採用']:
            qxy=CELLXY@matrix[:,:2].T+matrix[:,2]
            pre_dist=sample(pc['distance'],CELLXY/4);post_dist=sample(qc['distance'],qxy/4)
            arrays.update(matrix=matrix,pre_cell_distance=pre_dist,post_cell_distance=post_dist,H=distance_features(post_dist),前像=distance_features(pre_dist))
            shellrows=density_shells(r.key,caches[r.key],qc,pc,matrix,metadata,f)
            inverse=cv2.invertAffineTransform(matrix)
            positions=np.array([[o['中心横画素'],o['中心縦画素']] for o in qc['objects']])
            if len(positions):
                distances=sample(pc['distance'],(positions@inverse[:,:2].T+inverse[:,2])/4)
                labels=['既存候補' if d<=32 else '新規または移動候補' if np.isfinite(d) else '対応域外' for d in distances]
            else:labels=[]
            info['後候補対応分類']=labels;info['後既存候補数']=labels.count('既存候補');info['後新規移動候補数']=labels.count('新規または移動候補')
        info['候補一覧']=objects;info['距離集計']=shellrows;info['秒']=time.monotonic()-started
        np.savez_compressed(dest/(r.key+'.npz'),**arrays);dump(info,done)
        print('raw',j,r.key,pc['info']['候補数'],qc['info']['候補数'],reg['対応採用'],round(info['秒'],2),flush=True)
    c=table(OUT/'round3_control_manifest.csv')
    if limit:c=c.head(1)
    for _,r in c.iterrows():
        done=dest/(r.key+'.json')
        if done.exists():continue
        q=candidates(read_image(Path(r['画像パス'])))
        dump(dict(key=r.key,画像パス=r['画像パス'],対応='単独像のみ、前後差なし',単独像=q['info'],候補一覧=q['objects']),done)
        np.savez_compressed(dest/(r.key+'.npz'),mask=q['mask'],z=q['z'],distance=q['distance'])
    export_raw()

def export_raw():
    rows=[];objects=[];shells=[];controls=[]
    for p in sorted((OUT/'raw_candidates').glob('*.json')):
        a=json.loads(p.read_text(encoding='utf-8'));objects += [dict(key=a['key'],**o) if 'key' not in o else o for o in a['候補一覧']]
        if a['key'].startswith('260830'):
            controls.append(dict(key=a['key'],画像パス=a['画像パス'],**a['単独像']));continue
        shells+=a['距離集計']
        row={k:a[k] for k in ['key','日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野','保存差再現群','洗浄前パス','洗浄後パス']}
        row.update(a['対応'])
        for stage in ('洗浄前','洗浄後'):
            row.update({stage+k:v for k,v in a[stage].items()})
        for k in ('後既存候補数','後新規移動候補数'):row[k]=a.get(k,np.nan)
        rows.append(row)
    csv(rows,'round3_image_features.csv');csv(objects,'round3_candidate_objects.csv');csv(shells,'round3_distance_density.csv');csv(controls,'round3_control_features.csv')

def orientation_vectors(a):return np.exp(2j*np.deg2rad(a['第一帯軸度'].to_numpy()))

def orientation_values(a):
    vec=orientation_vectors(a);board=a['日程']+'_'+a['基板'];day=a['日程']
    bg=pd.Series(vec).groupby(board.to_numpy()).transform('sum').to_numpy()-vec
    bn=board.groupby(board).transform('size').to_numpy()-1
    dg=pd.Series(vec).groupby(day.to_numpy()).transform('sum').to_numpy()-vec
    dn=day.groupby(day).transform('size').to_numpy()-1
    valid=(bn>=2)&(dn>=2)&(np.abs(bg)>1e-12)&(np.abs(dg)>1e-12)
    bscore=np.real(vec*np.conj(bg))/np.maximum(np.abs(bg),1e-12)
    dscore=np.real(vec*np.conj(dg))/np.maximum(np.abs(dg),1e-12)
    return bscore-dscore,valid

def directional():
    verify();a=spatial();f=fields();rows=[];groups=[]
    for method in METHODS:
        s=a[a['定義']==method].reset_index(drop=True)
        for subset,q in [('周期候補',s[s['周期候補']]),('全第一軸感度',s),('固定29周期候補',s[s['周期候補']&s['固定29視野']])]:
            for cols,label in [(['日程','基板'],'基板'),(['日程'],'日程')]:
                for ids,g in q.groupby(cols):
                    ids=ids if isinstance(ids,tuple) else (ids,)
                    z=orientation_vectors(g).mean()
                    groups.append(dict(定義=method,対象=subset,単位=label,日程=ids[0],基板=ids[1] if len(ids)>1 else None,視野数=len(g),帯平均軸度=float(np.angle(z)*90/np.pi%180),向き集中度=float(abs(z))))
        predictors=s[s['周期候補']]
        for _,r in s.iterrows():
            q=predictors[(predictors['日程']==r['日程'])&(predictors['基板']==r['基板'])&(predictors.key!=r.key)]
            z=orientation_vectors(q).mean() if len(q) else complex(np.nan,np.nan)
            eligible=len(q)>=2 and abs(z)>=.2
            meta={k:r[k] for k in ['key','日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野','分類','周期候補','第一帯軸度','カメラ軸最小角度差','六方格子最小角度差','十字線最小角度差']}
            rows.append(dict(**meta,定義=method,他周期視野数=len(q),予測帯軸度=float(np.angle(z)*90/np.pi%180),他視野向き集中度=float(abs(z)),C利用可能=bool(eligible)))
    csv(rows,'round3_orientation_predictions.csv');csv(groups,'round3_orientation_groups.csv')
    print('directional',len(rows),sum(r['C利用可能'] for r in rows),flush=True)

def c_features(angle):
    theta=np.deg2rad(angle+90);u=(CELLXY[:,:,0]-1024)*np.cos(theta)+(CELLXY[:,:,1]-1024)*np.sin(theta)
    return np.stack([u/1024,(u/1024)**2]+[func(2*np.pi*u/p) for p in (256,512,1024) for func in (np.sin,np.cos)],axis=-1)

def evaluate(limit=0):
    verify();f=fields();g=f[f['定義']==METHODS[0]];direction=table(OUT/'round3_orientation_predictions.csv').set_index(['key','定義'])
    rawinfo=table(OUT/'round3_image_features.csv').set_index('key')
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual'];keys=list(z['keys'])
    assert keys==g.key.tolist()
    dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
    if limit:g=g[g.key.isin(rawinfo.index)].head(2)
    index={k:i for i,k in enumerate(keys)}
    for j,(_,r) in enumerate(g.iterrows()):
        done=dest/(r.key+'.json')
        if done.exists():continue
        rows=[];arrays={};distance_rows=[]
        for mi,method in enumerate(METHODS):
            d=direction.loc[(r.key,method)];models={};hasraw=r.key in rawinfo.index and rawinfo.loc[r.key,'対応採用']
            metadata={k:r[k] for k in ['key','日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野']}
            metadata.update(定義=method,他視野向き集中度=d['他視野向き集中度'],C他周期視野数=int(d['他周期視野数']),保存差再現群=bool(rawinfo.loc[r.key,'保存差再現群']) if r.key in rawinfo.index else False,生画像対応利用可能=bool(hasraw))
            if hasraw:
                with np.load(OUT/'raw_candidates'/(r.key+'.npz')) as z:models.update(H=z['H'],前像=z['前像'])
            if d['C利用可能']:models['C']=c_features(d['予測帯軸度'])
            if 'H' in models and 'C' in models:
                models['共同']=np.concatenate([models['H'],models['C']],axis=-1)
                models['前像C共同']=np.concatenate([models['前像'],models['C']],axis=-1)
            response=res[index[r.key],mi]
            # All directly compared models share the same finite cells and controls.
            if models:
                rr,aa=score_models(response,models,metadata);rows+=rr;arrays.update({method+'_'+k:v for k,v in aa.items()})
            if hasraw:
                for ri,radius in enumerate(RADII):
                    X=models['H'][:,:,ri:ri+1];valid=np.isfinite(response)&np.isfinite(X).all(axis=2)
                    score,_=cv_score(response,X,valid)
                    distance_rows.append(dict(**metadata,距離関数半径画素=radius,全面決定係数=score))
        dump(dict(説明率=rows,距離関数別説明率=distance_rows),done)
        if arrays:np.savez_compressed(dest/(r.key+'.npz'),**arrays)
        if j%40==0:print('evaluate',j,r.key,flush=True)
    allrows=[];distance=[]
    for p in sorted(dest.glob('*.json')):
        a=json.loads(p.read_text(encoding='utf-8'));allrows+=a['説明率'];distance+=a['距離関数別説明率']
    csv(allrows,'round3_explained_fraction.csv');csv(distance,'round3_radius_explained_fraction.csv')

def sign_test(values,groups=None):
    values=np.asarray(values,float);finite=np.isfinite(values);v=values[finite]
    if not len(v):return np.nan,np.nan,0
    if groups is None:codes=np.arange(len(v))
    else:codes=pd.factorize(np.asarray(groups)[finite])[0]
    sums=np.bincount(codes,weights=v);rng=np.random.default_rng(SEED)
    measured=float(v.mean());greater=0
    for _ in range(20):
        null=rng.choice([-1,1],size=(500,len(sums)))@sums/len(v)
        greater+=int(np.sum(null>=measured-1e-14))
    return measured,(greater+1)/10001,len(v)

def orientation_test(a,label,method):
    a=a.reset_index(drop=True);values,valid=orientation_values(a)
    n=int(valid.sum());measured=float(values[valid].mean()) if n else np.nan
    if not n:return dict(検定=label,定義=method,視野数=0,統計量=np.nan,補正前有意確率=np.nan,状態='必要な基板内周期視野が不足')
    vec=orientation_vectors(a);boards=pd.factorize(a['日程']+'_'+a['基板'])[0]
    daycodes=pd.factorize(a['日程'])[0];dayidx=[np.flatnonzero(daycodes==k) for k in np.unique(daycodes)]
    bcounts=np.bincount(boards);dcounts=np.bincount(daycodes)
    daytotal=np.bincount(daycodes,weights=vec.real)+1j*np.bincount(daycodes,weights=vec.imag)
    rng=np.random.default_rng(SEED);greater=0;nullcounts=[]
    for _ in range(10000):
        perm=vec.copy()
        for idx in dayidx:perm[idx]=rng.permutation(perm[idx])
        total=np.bincount(boards,weights=perm.real)+1j*np.bincount(boards,weights=perm.imag)
        bv=total[boards]-perm;dv=daytotal[daycodes]-perm
        v=(bcounts[boards]>=3)&(dcounts[daycodes]>=3)&(abs(bv)>1e-12)&(abs(dv)>1e-12)
        if v.any():
            null=float(np.mean(np.real(perm[v]*np.conj(bv[v]))/abs(bv[v])-np.real(perm[v]*np.conj(dv[v]))/abs(dv[v])))
            greater+=null>=measured-1e-14
        else:greater+=True
        nullcounts.append(int(v.sum()))
    contributions=a.loc[valid,['key','日程','基板','第一帯軸度']].copy();contributions['基板方向一致引く日程方向一致']=values[valid]
    contributions.to_csv(OUT/('round3_direction_contributions_'+method+'_'+label+'.csv'),index=False,encoding='utf-8-sig')
    return dict(検定=label,定義=method,視野数=n,統計量=measured,補正前有意確率=(greater+1)/10001,並替有効視野数最小=min(nullcounts),並替有効視野数最大=max(nullcounts),状態='実施')

def test_row(a,column,label,method):
    group=a['日程']+'_'+a['基板']
    effect,p,n=sign_test(a[column]);_,gp,_=sign_test(a[column],group)
    return dict(検定=label,定義=method,視野数=n,統計量=effect,補正前有意確率=p,日程基板一括反転有意確率=gp,状態='実施' if n else '利用可能視野なし')

def tests():
    verify();s=spatial();e=table(OUT/'round3_explained_fraction.csv');density=table(OUT/'round3_distance_density.csv');rows=[]
    for method in METHODS:
        d=s[(s['定義']==method)&s['周期候補']]
        rows.append(orientation_test(d,'C基板内方向一致_全周期候補',method))
        rows.append(orientation_test(d[d['固定29視野']],'C基板内方向一致_固定29周期候補',method))
        for model,subset in [('H','全利用可能'),('H','固定29利用可能'),('C','全利用可能'),('C','固定29利用可能'),('共同','両モデル利用可能')]:
            a=e[(e['定義']==method)&(e['モデル']==model)]
            if subset=='固定29利用可能':a=a[a['固定29視野']]
            rows.append(test_row(a,'対照との差',model+'空間対照差_'+subset,method))
        q=density[(density['定義']==method)&(density['候補区分']=='後全候補')]
        near=q[q['殻番号'].isin([0,1,2])].groupby('key')[['ピラー数','外れ値数']].sum()
        far=q[q['殻番号']==5].set_index('key')[['ピラー数','外れ値数']]
        near['近傍密度']=near['外れ値数']/near['ピラー数'];far['遠方密度']=far['外れ値数']/far['ピラー数']
        metadata=q.drop_duplicates('key')[['key','日程','基板','ブランク','固定29視野']].set_index('key')
        dd=metadata.join(near[['近傍密度']]).join(far[['遠方密度']]);dd['近傍遠方差']=dd['近傍密度']-dd['遠方密度']
        rows.append(test_row(dd,'近傍遠方差','H距離集中_全利用可能',method))
        rows.append(test_row(dd[dd['ブランク']],'近傍遠方差','H距離集中_ブランク',method))
        dd.reset_index().to_csv(OUT/('round3_distance_field_effects_'+method+'.csv'),index=False,encoding='utf-8-sig')
        a=e[(e['定義']==method)&(e['モデル']=='C')].dropna(subset=['他視野向き集中度','全面決定係数']).reset_index(drop=True)
        if len(a)>=4 and a['他視野向き集中度'].nunique()>1:
            x=stats.rankdata(a['他視野向き集中度']);y=stats.rankdata(a['全面決定係数']);measured=float(np.corrcoef(x,y)[0,1])
            rng=np.random.default_rng(SEED);idx=[np.array(g.index) for _,g in a.groupby('日程')];greater=0
            for _ in range(10000):
                yp=y.copy()
                for ids in idx:yp[ids]=rng.permutation(yp[ids])
                greater+=np.corrcoef(x,yp)[0,1]>=measured-1e-14
            p=(greater+1)/10001
        else:measured=p=np.nan
        rows.append(dict(検定='C集中度と説明率',定義=method,視野数=len(a),統計量=measured,補正前有意確率=p,状態='実施' if np.isfinite(p) else '視野数または変動不足'))
    assert len(rows)==20
    for r in rows:r['ボンフェローニ補正後']=min(1,20*r['補正前有意確率']) if np.isfinite(r['補正前有意確率']) else np.nan
    csv(rows,'round3_tests.csv')
    # Descriptive sensitivity: all primary axes, no new confirmatory evidence.
    sensitivity=[]
    for method in METHODS:
        a=s[s['定義']==method]
        sensitivity.append(orientation_test(a,'C全第一軸_記述的感度',method))
    csv(sensitivity,'round3_direction_sensitivity.csv')
    print('tests',len(rows),flush=True)

def summarize():
    verify();e=table(OUT/'round3_explained_fraction.csv');rows=[]
    for (method,model),g in e.groupby(['定義','モデル']):
        subsets={'全利用可能':g,'固定29利用可能':g[g['固定29視野']],'保存差再現群':g[g['保存差再現群']],'追加対応群':g[g['生画像対応利用可能']&~g['保存差再現群']]}
        if model=='C':subsets['生画像対応不要群']=g[~g['生画像対応利用可能']]
        for label,a in subsets.items():
            for metric in ['全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差','全面決定係数_H固有','全面決定係数_C固有','全面決定係数_共通']:
                if metric not in a:continue
                v=a[metric].dropna()
                rows.append(dict(定義=method,モデル=model,対象=label,指標=metric,視野数=len(v),中央値=float(v.median()),第1四分位=float(v.quantile(.25)),第3四分位=float(v.quantile(.75)),負値視野数=int((v<0).sum())))
    csv(rows,'round3_explanation_summary.csv')
    joint=e[e['モデル']=='共同'].copy();csv(joint,'round3_shared_unique_fractions.csv')
    manifest=table(OUT/'round3_field_manifest.csv');image=table(OUT/'round3_image_features.csv');cov=[]
    for day,g in manifest.groupby('日程'):
        im=image[image['日程']==day]
        cov.append(dict(日程=day,保存差視野数=len(g),生画像組数=int(g['生画像組利用可能'].sum()),候補処理組数=len(im),対応利用可能数=int(im['対応採用'].sum()),保存差再現群数=int(im['保存差再現群'].sum()),追加対応失敗数=int((~im['対応採用']).sum())))
    csv(cov,'round3_date_coverage.csv')
    # Compare all hypotheses on the common spatial response, preserving missing values.
    previous=table(PREVIOUS/'round2_explanation_summary.csv');comparison=[]
    decisions={'周1_B':('支持されない','中'),'周1_F':('判定不能','低'),'二次面':('滑らかな勾配に部分整合、帯の主因は支持されない','中'),'G':('光学原因全体は判定不能','中'),'I':('今回の撮影順序予測では支持されない','中')}
    for _,r in previous[previous['指標']=='全面決定係数'].iterrows():
        if r['モデル'] not in decisions:continue
        model={'周1_B':'B','周1_F':'F','二次面':'G（方向を限定しない二次面）'}.get(r['モデル'],r['モデル'])
        comparison.append(dict(仮説=model,定義=r['定義'],対象=r['対象'],対象視野数=int(r['視野数']),説明率中央値=r['中央値'],第1四分位=r['第1四分位'],第3四分位=r['第3四分位'],判定=decisions[r['モデル']][0],確度=decisions[r['モデル']][1],独立性='G・I画像モデルは差と画像共有、記述のみ' if model in ('G','I') else '周1・周2の規定を参照'))
    for _,r in pd.DataFrame(rows).query("指標 == '全面決定係数' and 対象 in ['全利用可能','固定29利用可能']").iterrows():
        if r['モデル'] not in ('H','C','共同','前像'):continue
        comparison.append(dict(仮説={'共同':'H・C共同','前像':'H（前像対照）'}.get(r['モデル'],r['モデル']),定義=r['定義'],対象=r['対象'],対象視野数=r['視野数'],説明率中央値=r['中央値'],第1四分位=r['第1四分位'],第3四分位=r['第3四分位'],判定='判定不能（因果）',確度='低',独立性='Cは対象視野を除く他視野方向。画像共有・共通原因の限界あり' if r['モデル']=='C' else '差の片側と画像共有、記述のみ'))
    for model in ('A','D','E','J'):
        for method in METHODS:
            comparison.append(dict(仮説=model,定義=method,対象='未測定',対象視野数=0,説明率中央値=np.nan,第1四分位=np.nan,第3四分位=np.nan,判定='位置固定は帯の原因として支持されない（既存記録）' if model=='A' else '未検証',確度='中' if model=='A' else '未評価',独立性='共通尺度の説明率は未測定'))
    csv(comparison,'round3_all_hypotheses_comparison.csv')

def validate():
    verify();a=table(OUT/'round3_input_integrity.csv');a['終了内容指紋']=[digest(Path(p)) for p in a['実在パス']];a['不変']=a['開始内容指紋']==a['終了内容指紋'];assert a['不変'].all();csv(a,'round3_input_integrity_final.csv')
    f=fields();manifest=table(OUT/'round3_field_manifest.csv');im=table(OUT/'round3_image_features.csv');e=table(OUT/'round3_explained_fraction.csv');tests=table(OUT/'round3_tests.csv');joint=e[e['モデル']=='共同'];direction=table(OUT/'round3_orientation_predictions.csv')
    assert len(f)==1264 and len(manifest)==632 and len(im)==286 and len(tests)==20 and len(direction)==1264
    assert im.key.nunique()==286 and len(table(OUT/'round3_control_features.csv'))==6
    assert len(list((OUT/'evaluation').glob('*.json')))==632
    assert np.isfinite(e['全面決定係数']).all()
    assert (e['対照共通升目数']<=e['全面升目数']).all()
    errors=[];relative_errors=[]
    for domain in ('全面決定係数','対照共通領域決定係数'):
        v=joint[domain+'_H固有']+joint[domain+'_C固有']+joint[domain+'_共通']-joint[domain]
        if len(v):
            errors.append(float(np.nanmax(np.abs(v))))
            magnitude=joint[domain].abs()+joint[domain+'_H固有'].abs()+joint[domain+'_C固有'].abs()+joint[domain+'_共通'].abs()
            relative_errors.append(float(np.nanmax(np.abs(v)/np.maximum(1,magnitude))))
    # Signed partitions can have magnitude >100000; use scaled floating precision.
    # This changes only the verification tolerance, never any model or saved score.
    assert not relative_errors or max(relative_errors)<1e-12
    leakage=[]
    for tr,te in folds(np.ones((64,64),bool)):
        assert not set(tr)&set(te)
        ty,tx=np.unravel_index(te,(64,64));train=np.zeros((64,64),np.uint8);train.ravel()[tr]=1
        for dy in range(-2,3):
            for dx in range(-2,3):
                inside=(ty+dy>=0)&(ty+dy<64)&(tx+dx>=0)&(tx+dx<64)
                leakage.append(int(train[ty[inside]+dy,tx[inside]+dx].sum()))
    assert sum(leakage)==0
    # Known affine translation confirms before-to-after coordinate direction.
    synthetic=np.zeros((256,256),np.float32);rng=np.random.default_rng(SEED)
    synthetic[40:210,40:210]=rng.random((170,170));synthetic=cv2.GaussianBlur(synthetic,(0,0),2)
    moved=cv2.warpAffine(synthetic,np.array([[1,0,7],[0,1,-5]],np.float32),(256,256))
    shift,_=cv2.phaseCorrelate(synthetic,moved);assert np.linalg.norm(np.array(shift)-[7,-5])<.2
    # Verify density response really matches raw cached delta and locked thresholds.
    _,_,_,caches,_=discover();withcheck=[]
    with np.load(OLD/'density_maps.npz') as z:maps=z['density'];mapkeys=list(z['keys'])
    for key in ['260926_3_3','260828_1_1']:
        if key not in caches:continue
        with np.load(caches[key]) as z:xy=z['xy'];delta=z['delta']
        ix=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int);count=np.bincount(ix,minlength=4096)
        for mi,method in enumerate(METHODS):
            threshold=f[(f.key==key)&(f['定義']==method)]['閾値'].iloc[0]
            positive=np.bincount(ix,weights=(delta>threshold),minlength=4096)
            d=np.divide(positive,count,out=np.full(4096,np.nan),where=count>0).reshape(64,64)
            error=float(np.nanmax(np.abs(d-maps[mapkeys.index(key),mi])));assert error<1e-7
            withcheck.append(dict(key=key,定義=method,密度最大絶対差=error))
    verification=dict(入力指紋照合数=len(a),入力旧版不変=bool(a['不変'].all()),画像処理組数=len(im),単独コントロール数=6,二定義説明率行数=len(e),二定義検定数=len(tests),分解恒等最大差=max(errors) if errors else None,分解恒等量級相対最大差=max(relative_errors) if relative_errors else None,検証の許容誤差変更='絶対1e-12は極端な負値で丸め誤差を過剰判定した。計算値は変更せず量級に対する1e-12で確認。',学習保留および64画素余白重複数=sum(leakage),既知移動推定誤差画素=float(np.linalg.norm(np.array(shift)-[7,-5])),保存密度再確認=withcheck,自動候補の採用除外マスク利用=False)
    dump(verification,OUT/'round3_verification.json');print(verification,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','raw','directional','evaluate','tests','summarize','validate']);p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    if args.stage in ('raw','evaluate'):globals()[args.stage](args.limit)
    else:globals()[args.stage]()
