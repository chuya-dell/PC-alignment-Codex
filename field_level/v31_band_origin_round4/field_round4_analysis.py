"""Round-four validation only. No production execution or source input writes."""
from __future__ import annotations
import sys, os
sys.dont_write_bytecode = True
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, hashlib, json, time, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from field_round4_methods import cv_score, controls, analyze_map, angular_difference

ROOT=Path(__file__).resolve().parents[2]
LOCAL=ROOT/'data/inputs_local'
OLD=ROOT/'data/results/v28_band_origin_round1'
PREV=ROOT/'data/results/v30_band_origin_round3'
OUT=ROOT/'data/results/v31_band_origin_round4'
METHODS=['平均標準偏差','中央値絶対偏差']
CONDITIONS=[(0,400),(40,400),(80,640),(40,256)]
STRENGTHS=[.01,.03,.10]
SEED=20261004
cv2.setNumThreads(1)

def table(p): return pd.read_csv(p,dtype={'日程':str,'基板':str,'date':str,'board':str})
def csv(rows,name): pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
def dump(a,p): p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verify():
    r=pd.read_csv(OUT/'prediction_sha256.csv').iloc[0]
    assert digest(Path(r.Path)).upper()==r.Hash
def rng_for(s): return np.random.default_rng(SEED+int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little'))
def base_fields(): return table(OLD/'round1_fields_both_definitions.csv').query('定義 == @METHODS[0]').reset_index(drop=True)
def meta(r): return {k:r[k] for k in ['key','日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野']}

def discover():
    children={p.name:p for p in LOCAL.iterdir()}
    source=[p for n,p in children.items() if p.is_dir() and n.startswith('2026') and 'digital_judgment' in n]
    assert len(source)==1
    folders={p.name:p for p in source[0].iterdir()}
    tables={p.name:p for p in folders['tables'].iterdir()}
    caches={p.stem:p for p in tables['cached_field_differences'].iterdir() if p.suffix=='.npz'}
    raw={p.name:p for p in children['raw_readonly'].iterdir()}
    return children,source[0],tables,caches,raw

def init():
    verify();children,source,tables,caches,raw=discover()
    markers=[p for n,p in raw.items() if p.is_file() and 'copy_done' in n]
    assert markers,'生画像コピー完了印なし'
    used=set(caches.values())|set(markers)|set(children['lab_notes'].iterdir())
    f=base_fields();qc=table(tables['table_registration_field_qc.csv']);used.add(tables['table_registration_field_qc.csv'])
    files={n:{p.name:p for p in d.iterdir() if p.is_file()} for n,d in raw.items() if d.is_dir()}
    rows=[]
    for _,r in f.iterrows():
        assert r.key in caches
        q=qc[(qc.date==r['日程'])&(qc.board==r['基板'])&(qc.field==r['視野番号'])]
        assert len(q)==1
        row=meta(r);row['保存差パス']=str(caches[r.key])
        for label,col in [('洗浄前','path_pre'),('洗浄後','path_post')]:
            filename=str(q.iloc[0][col]).replace('\\','/').split('/')[-1]
            found=[fs[filename] for n,fs in files.items() if n.startswith(r['日程']) and filename in fs]
            assert len(found)<=1
            row[label+'パス']=str(found[0]) if found else None
            if found:used.add(found[0])
        row['生画像組利用可能']=bool(row['洗浄前パス'] and row['洗浄後パス']);rows.append(row)
    ctrl=[dict(key='260830_'+p.stem,画像パス=str(p)) for n,fs in files.items() if n.startswith('260830') for p in fs.values() if p.suffix.lower() in ('.tif','.tiff')]
    used.update(Path(r['画像パス']) for r in ctrl)
    for folder in ['v28_band_origin_round1','v29_band_origin_round2','v30_band_origin_round3']:
        for root in [ROOT/'field_level'/folder,ROOT/'data/results'/folder]:
            used.update(p for p in root.rglob('*') if p.is_file())
    csv(rows,'round4_field_manifest.csv');csv(ctrl,'round4_control_manifest.csv')
    csv([dict(実在パス=str(p.resolve()),バイト数=p.stat().st_size,開始内容指紋=digest(p)) for p in sorted(used)],'round4_input_integrity.csv')
    dump(dict(入力起点=str(LOCAL),解析コピー=str(source),生画像フォルダ=[str(p) for p in raw.values() if p.is_dir()],完了印=[str(p) for p in markers],視野数=len(f),固定29数=int(f['固定29視野'].sum()),生画像組数=sum(r['生画像組利用可能'] for r in rows),単独対照像数=len(ctrl),指紋記録数=len(used),作業ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),開始コミット=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),版番号確認='開始時点で解析・結果フォルダ、ブランチ、タグにv31なし',環境=dict(Python=sys.version,数値配列=np.__version__,表処理=pd.__version__,画像処理=cv2.__version__)),OUT/'round4_preflight.json')
    print('preflight complete',len(used),flush=True)

def load(r,caches):
    p=caches[r.key]
    assert LOCAL.resolve() in p.resolve().parents
    with np.load(p) as z:return z['delta'],z['xy'],z['ids']

def density(values,cell,select=None):
    if select is None:select=np.ones(len(cell),bool)
    n=np.bincount(cell[select],minlength=4096)
    num=np.bincount(cell[select],weights=values[select],minlength=4096)
    return np.divide(num,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64),n.reshape(64,64)

def residuals(a):
    valid=np.isfinite(a);n=valid.sum(axis=0);s=np.nansum(a,axis=0)
    common=np.divide(s-np.nan_to_num(a),n-valid,out=np.full_like(a,np.nan),where=n-valid>0)
    return a-common

def splits():
    verify();_,_,_,caches,_=discover();f=base_fields()
    maps=np.full((len(f),2,2,2,64,64),np.nan,np.float32);count=np.zeros((len(f),2,2,64,64),np.int32)
    thresholds=table(OLD/'round1_thresholds.csv').set_index('日程')
    with np.load(OLD/'density_maps.npz') as z:old=z['density'];keys=z['keys'];oldcounts=z['counts']
    assert list(keys)==f.key.tolist()
    maximum=0.
    for i,r in f.iterrows():
        d,xy,ids=load(r,caches);cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
        partitions=[ids[:,0]%2,(ids[:,0]+ids[:,1])%2]
        for mi,method in enumerate(METHODS):
            threshold=thresholds.loc[r['日程'],['平均標準偏差','中央値絶対偏差閾値'][mi]]
            pos=d>threshold;dm,n=density(pos,cell)
            maximum=max(maximum,float(np.nanmax(abs(dm-old[i,mi]))))
            assert np.array_equal(n,oldcounts[i])
            for si,part in enumerate(partitions):
                for k in range(2):maps[i,mi,si,k],count[i,si,k]=density(pos,cell,part==k)
                assert np.array_equal(count[i,si].sum(axis=0),n)
        if i%100==0:print('split maps',i,r.key,flush=True)
    assert maximum<1e-7
    np.savez_compressed(OUT/'split_maps.npz',density=maps,counts=count,keys=keys,residual=residuals(maps))
    dump(dict(周1地図最大絶対差=maximum,全視野数=len(f),分割合計一致=True),OUT/'split_verification.json')

def score(response,X):
    X=X[:,:,None] if X.ndim==2 else X
    valid=np.isfinite(response)&np.isfinite(X).all(axis=2)
    shifted=controls(X);common=valid.copy()
    for a in shifted:common &= np.isfinite(a).all(axis=2)
    full,pr=cv_score(response,X,valid)
    real,_=cv_score(response,X,common)
    values=[cv_score(response,a,common)[0] for a in shifted]
    median=float(np.nanmedian(values))
    return dict(全面決定係数=full,対照共通領域決定係数=real,対照決定係数中央値=median,対照との差=real-median,全面升目数=int(valid.sum()),対照共通升目数=int(common.sum()),**{f'対照{j+1}決定係数':v for j,v in enumerate(values)})

def noise(limit):
    verify();f=base_fields();dest=OUT/'noise_checkpoints';dest.mkdir(exist_ok=True)
    with np.load(OUT/'split_maps.npz') as z:res=z['residual']
    for i,r in f.iloc[:limit or len(f)].iterrows():
        p=dest/(r.key+'.json')
        if p.exists():continue
        rows=[]
        for mi,method in enumerate(METHODS):
            for si,label in enumerate(['格子行偶奇','市松']):
                a,b=res[i,mi,si];valid=np.isfinite(a)&np.isfinite(b)
                C=float(np.cov(a[valid],b[valid],ddof=0)[0,1]);S=float((np.var(a[valid])+np.var(b[valid]))/2)
                estimates=[]
                for k,(response,X) in enumerate([(b,a),(a,b)]):
                    values=score(response,X);estimates.append(values['全面決定係数'])
                    rows.append(dict(**meta(r),定義=method,分割=label,予測向き='組1から組2' if k==0 else '組2から組1',升目共分散=C,升目分散平均=S,全数信号割合近似=2*C/(S+C) if S+C>0 else np.nan,**values))
                for row in rows[-2:]:row['双方向平均決定係数']=float(np.mean(estimates))
        dump(rows,p)
        if i%60==0:print('noise',i,r.key,flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'round4_noise_ceiling.csv')

def wave(xy,angle,period):
    normal=np.deg2rad(angle-90)
    u=xy[:,0]*np.cos(normal)+xy[:,1]*np.sin(normal)
    return (1+np.cos(2*np.pi*u/period))/2

def inject_before(d,g,threshold,target):
    pos=d>threshold;n=len(d);goal=min(n,int(pos.sum())+int(round(target*n)))
    high=max(1.,float(np.max(threshold-d)))
    for _ in range(50):
        if (d+high*g>threshold).sum()>=goal:break
        high*=2
    low=0.
    for _ in range(42):
        middle=(low+high)/2
        if (d+middle*g>threshold).sum()>=goal:high=middle
        else:low=middle
    result=d+high*g>threshold
    return result,high,float(result.mean()-pos.mean())

def inject_after(d,g,threshold,target,random):
    pos=d>threshold;n=len(d);available=~pos;goal=min(target,float(available.mean()))
    high=1.
    for _ in range(50):
        if np.minimum(1,high*g[available]).sum()/n>=goal:break
        high*=2
    low=0.
    for _ in range(42):
        middle=(low+high)/2
        if np.minimum(1,middle*g[available]).sum()/n>=goal:high=middle
        else:low=middle
    result=pos|(available&(random.random(n)<np.minimum(1,high*g)))
    return result,high,float(result.mean()-pos.mean())

def injection(limit):
    verify();_,_,_,caches,_=discover();f=base_fields();dest=OUT/'injection_checkpoints';dest.mkdir(exist_ok=True)
    thresholds=table(OLD/'round1_thresholds.csv').set_index('日程')
    with np.load(OLD/'common_profile_residuals.npz') as z:common=z['common'];orig=z['residual'];keys=z['keys']
    assert list(keys)==f.key.tolist()
    for i,r in f.iloc[:limit or len(f)].iterrows():
        p=dest/(r.key+'.json')
        if p.exists():continue
        started=time.monotonic();d,xy,ids=load(r,caches);cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
        rows=[];demo={}
        for mi,method in enumerate(METHODS):
            threshold=thresholds.loc[r['日程'],['平均標準偏差','中央値絶対偏差閾値'][mi]]
            for angle,period in CONDITIONS:
                g=wave(xy,angle,period);X,n=density(g,cell)
                base=score(orig[i,mi],X)
                rows.append(dict(**meta(r),定義=method,注入法='注入なし',帯軸度=angle,周期画素=period,目標追加割合=0.,実現追加割合=0.,輝度振幅または確率係数=0.,**base))
                for strength in STRENGTHS:
                    for kind in ['判定前','判定後']:
                        positive,A,increase=inject_before(d,g,threshold,strength) if kind=='判定前' else inject_after(d,g,threshold,strength,rng_for(f'{r.key}/{method}/{angle}/{period}/{strength}'))
                        dm,_=density(positive,cell);response=dm-common[i,mi]
                        values=score(response,X)
                        rows.append(dict(**meta(r),定義=method,注入法=kind,帯軸度=angle,周期画素=period,目標追加割合=strength,実現追加割合=increase,輝度振幅または確率係数=A,目標差=increase-strength,飽和目標不能=bool(float((d<=threshold).mean())<strength),実現増分の下振れ0_2ポイント超え=bool(increase<strength-.002),注入なし全面決定係数=base['全面決定係数'],**values))
                        if r['固定29視野'] and angle==40 and period==400:demo[f'{method}_{kind}_{strength}']=dm
                if r['固定29視野'] and angle==40 and period==400:demo[f'{method}_既知形']=X;demo[f'{method}_注入なし']=orig[i,mi]+common[i,mi]
        dump(rows,p)
        if demo:
            target=OUT/'injection_examples';target.mkdir(exist_ok=True);np.savez_compressed(target/(r.key+'.npz'),**demo)
        if i%20==0:print('injection',i,r.key,round(time.monotonic()-started,1),'seconds',flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'round4_injection.csv')

def block_image(p):
    a=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    assert a.ndim==2 and a.shape==(2044,2048),(str(p),a.shape)
    a=a.astype(float)/65535
    result=np.empty((64,64))
    result[:63]=a[:2016].reshape(63,32,64,32).mean(axis=(1,3))
    result[63]=a[2016:].reshape(28,64,32).mean(axis=(0,2))
    return result

def images(limit):
    verify();manifest=table(OUT/'round4_field_manifest.csv');control=table(OUT/'round4_control_manifest.csv')
    dest=OUT/'single_images';dest.mkdir(exist_ok=True);jobs=[]
    for _,r in manifest[manifest['生画像組利用可能']].iterrows():
        for stage in ['洗浄前','洗浄後']:jobs.append((r.key+'_'+stage,Path(r[stage+'パス']),dict(**meta(r),時点=stage)))
    for _,r in control.iterrows():jobs.append((r.key,Path(r['画像パス']),dict(key=r.key,日程='260830',基板='未確認',ブランク=False,固定29視野=False,時点='前後未確認')))
    for j,(key,p,metadata) in enumerate(jobs[:limit or len(jobs)]):
        done=dest/(key+'.json')
        if done.exists():continue
        assert LOCAL.resolve() in p.resolve().parents
        a=block_image(p);info,ac,overlap,power=analyze_map(a,rng_for(key))
        np.savez_compressed(dest/(key+'.npz'),brightness=a,autocorrelation=ac,overlap=overlap,spectrum=power)
        dump(dict(**metadata,画像パス=str(p),**info),done)
        if j%30==0:print('single image',j,key,flush=True)
    csv([json.loads(p.read_text(encoding='utf-8')) for p in sorted(dest.glob('*.json'))],'round4_single_image_spectrum.csv')

def pilot():
    verify();f=base_fields();_,_,_,caches,_=discover();r=f[f['固定29視野']].iloc[0];d,xy,ids=load(r,caches)
    cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int);threshold=table(OLD/'round1_thresholds.csv').set_index('日程').loc[r['日程'],'平均標準偏差']
    g=wave(xy,40,400);pos,A,inc=inject_before(d,g,threshold,.03);assert abs(inc-.03)<2/len(d)
    a,n=density(g,cell);geometry=analyze_map(a,rng_for('pilot'))[0];assert angular_difference(geometry['第一帯軸度'],40)<=10
    y,x=np.indices((64,64));predictor=np.cos(2*np.pi*x/12);v=np.ones((64,64),bool)
    clean,_=cv_score(2+3*predictor,predictor[:,:,None],v);assert clean>.99
    for k in ['density','residual']:
        with np.load(OUT/'split_maps.npz') as z:assert z[k].shape==(632,2,2,2,64,64)
    noise(2);injection(2);images(2)
    dump(dict(判定前注入視野=r.key,目標=.03,実現増分=inc,輝度振幅=A,既知帯第一軸度=geometry['第一帯軸度'],既知帯周期画素=geometry['第一スペクトル周期画素'],無雑音既知形決定係数=clean,判定='小規模確認成功'),OUT/'round4_pilot_verification.json')
    print('pilot passed',flush=True)

def integrity():
    verify();a=table(OUT/'round4_input_integrity.csv')
    a['終了内容指紋']=[digest(Path(p)) for p in a['実在パス']]
    a['不変']=a['開始内容指紋']==a['終了内容指紋']
    csv(a,'round4_input_integrity_final.csv');assert a['不変'].all()
    print('integrity unchanged',len(a),flush=True)

def repair_flags():
    """Correct metadata only; retain all scientific estimates and an audit trail."""
    verify();f=table(OLD/'round1_fields_both_definitions.csv').set_index(['key','定義']);records=[]
    for p in sorted((OUT/'injection_checkpoints').glob('*.json')):
        rows=json.loads(p.read_text(encoding='utf-8'));before=digest(p)
        changes=[]
        for row in rows:
            if row['注入法']=='注入なし':continue
            available=1-f.loc[(row['key'],row['定義']),'外れ値割合']
            corrected=bool(available<row['目標追加割合'])
            original=row['飽和目標不能']
            row['実現増分の下振れ0_2ポイント超え']=bool(row['実現追加割合']<row['目標追加割合']-.002)
            if original!=corrected:
                changes.append(dict(key=row['key'],定義=row['定義'],注入法=row['注入法'],帯軸度=row['帯軸度'],周期画素=row['周期画素'],目標追加割合=row['目標追加割合'],実現追加割合=row['実現追加割合'],訂正前飽和目標不能=original,訂正後飽和目標不能=corrected,理由='確率注入の標本変動による下振れを飽和と誤記。非陽性の総割合で飽和可否を判定。決定係数・振幅・実現増分は変更なし。'))
            row['飽和目標不能']=corrected
        dump(rows,p);after=digest(p)
        for change in changes:change.update(訂正前指紋=before,訂正後指紋=after);records.append(change)
    if records:csv(records,'round4_injection_flag_correction.csv')
    csv([r for p in sorted((OUT/'injection_checkpoints').glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'round4_injection.csv')
    print('metadata corrections',len(records),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['init','splits','pilot','noise','injection','images','integrity','repair_flags']);parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    if args.stage in ['noise','injection','images']:globals()[args.stage](args.limit)
    else:globals()[args.stage]()
