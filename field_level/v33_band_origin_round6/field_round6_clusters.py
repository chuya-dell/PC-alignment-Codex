"""Fixed six-neighbor components, field-unit tests and paired descriptive images."""
from field_round6_common import *
import argparse
from scipy.spatial import cKDTree
OFF=((1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1))

def large_components(ids,positive):
    pts={tuple(map(int,ids[i])):int(i) for i in np.flatnonzero(positive)};components=[];sizes=[]
    while pts:
        seed,i=pts.popitem();stack=[seed];indices=[i]
        while stack:
            x,y=stack.pop()
            for dx,dy in OFF:
                p=(x+dx,y+dy)
                if p in pts:indices.append(pts.pop(p));stack.append(p)
        sizes.append(len(indices))
        if len(indices)>=492:components.append(np.asarray(indices,int))
    return sorted(components,key=len,reverse=True),sizes

def raw_index():
    _,tabs,_,raw=discover();qc=table(tabs['table_registration_field_qc.csv'])
    directories={name:{p.name:p for p in d.iterdir() if p.is_file()} for name,d in raw.items() if d.is_dir()}
    result={}
    for _,r in qc.iterrows():
        key=f'{r.date}_{r.board}_{int(r.field)}';row={}
        for stage,col in (('洗浄前','path_pre'),('洗浄後','path_post')):
            name=str(r[col]).replace('\\','/').split('/')[-1]
            found=[fs[name] for folder,fs in directories.items() if folder.startswith(str(r.date)) and name in fs]
            assert len(found)<=1
            row[stage]=found[0] if found else None
        result[key]=row
    return result
def recovery_index():
    result={}
    for folder in (OLD/'bf_v24_recovery',R5/'bf_v24_recovery',RESUME/'bf_v24_recovery'):
        for p in folder.iterdir():
            if p.suffix=='.npz':result[p.stem]=p
    return result
def recover(key,cache,rec):
    if key not in rec:return None,dict(回復=False,理由='保存行列なし')
    a=arrays(rec[key]);d=cache['delta'];xy=cache['xy'];ids=cache['ids']
    sameids=np.array_equal(a['recomputed_ids'],ids)
    samexy=a['recomputed_xy'].shape==xy.shape and np.max(abs(a['recomputed_xy']-xy))<=1e-6
    err=float(np.max(abs(a['recomputed_delta']-d))) if a['recomputed_delta'].shape==d.shape else np.inf
    accepted=bool(sameids and samexy and np.isfinite(err) and err<=1e-6)
    return (a if accepted else None),dict(回復=accepted,理由='数値条件一致' if accepted else '保存差再現条件不合格',再計算識別一致=bool(sameids),再計算座標一致=bool(samexy),再計算差最大絶対差=err,回復保存パス=str(rec[key]))

def geometry(limit=0):
    verify();f=fields();_,_,caches,_=discover();dest=OUT/'cluster_checkpoints';dest.mkdir(exist_ok=True);started=time.monotonic()
    old=table(R5/'round5_cluster_fields.csv').set_index(['key','定義'])
    for j,r in f.iloc[:limit or len(f)].iterrows():
        p=dest/(r.key+'.json')
        if p.exists():
            prior=json.loads(p.read_text(encoding='utf-8'))
            if all('最大連結外れ値本数' in v for v in prior['fields']):continue
            # Correct the naming of the old maximum, which included components
            # smaller than 492; preserve it in a separate explicit column.
            for v in prior['fields']:
                v['最大連結外れ値本数']=v['最大塊本数']
                v['最大塊本数']=v['最大塊本数'] if v['塊あり'] else 0
            dump(prior,p);continue
        a=arrays(caches[r.key]);xy=a['xy'];ids=a['ids'];d=a['delta'];rows=[];objects=[];labels=[];maps=[]
        for method in METHODS:
            threshold=table(OLD/'round1_thresholds.csv').set_index('日程').loc[r['日程'],'平均標準偏差' if method==METHODS[0] else '中央値絶対偏差閾値']
            comp,sizes=large_components(ids,d>threshold);lab=np.zeros(len(ids),np.int16)
            cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int);den=np.bincount(cell,minlength=4096)
            for ci,members in enumerate(comp):
                lab[members]=ci+1;coords=xy[members];cent=coords.mean(axis=0)
                eigen,vec=np.linalg.eigh(np.cov(coords.T,ddof=0));long=vec[:,-1];ratio=np.sqrt(eigen[-1]/max(eigen[0],1e-12));angle=float(np.degrees(np.arctan2(long[1],long[0]))%180)
                edge=np.minimum.reduce([coords[:,0],2048-coords[:,0],coords[:,1],2044-coords[:,1]])
                objects.append(dict(**meta(r),定義=method,塊番号=ci+1,ピラー数=len(members),重心横=cent[0],重心縦=cent[1],縁距離最小=edge.min(),縁距離中央値=np.median(edge),中心距離=np.linalg.norm(cent-[1024,1022]),長軸標準偏差=np.sqrt(eigen[-1]),短軸標準偏差=np.sqrt(eigen[0]),長短軸比=ratio,長軸方向度=angle,方向利用可能=ratio>=1.2))
            row=dict(**meta(r),定義=method,塊あり=bool(comp),塊数=len(comp),最大塊本数=len(comp[0]) if comp else 0,最大連結外れ値本数=max(sizes,default=0),大塊総本数=int((lab>0).sum()),全外れ値本数=int((d>threshold).sum()))
            assert row['最大連結外れ値本数']==old.loc[(r.key,method),'最大陽性塊']
            assert row['塊数']==old.loc[(r.key,method),'大塊数']
            assert row['大塊総本数']==old.loc[(r.key,method),'大塊陽性ピラー数']
            rows.append(row);labels.append(lab)
            num=np.bincount(cell,weights=lab>0,minlength=4096)
            maps.append(np.divide(num,den,out=np.full(4096,np.nan),where=den>0).reshape(64,64))
        dump(dict(fields=rows,objects=objects),p)
        if any(r['塊あり'] for r in rows):np.savez_compressed(dest/(r.key+'.npz'),labels=np.stack(labels),maps=np.stack(maps))
        if j%100==0:print('clusters',j,r.key,round(time.monotonic()-started,1),flush=True)
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(dest.glob('*.json'))]
    csv([r for a in records for r in a['fields']],'cluster_fields.csv');csv([r for a in records for r in a['objects']],'cluster_objects.csv')

def permute_groups(g,cols,reps=10000):
    observed=[];null=np.zeros((reps,len(cols)));random=rng('cluster_groups')
    for _,h in g.groupby('日程'):
        v=h[cols].to_numpy(float);blank=h['ブランク'].to_numpy(bool);nb=int(blank.sum());nn=len(h)
        assert nb and nb<nn
        observed.append(v[~blank].mean(axis=0)-v[blank].mean(axis=0))
        for start in range(0,reps,500):
            sz=min(500,reps-start);pick=np.argsort(random.random((sz,nn)),axis=1)[:,:nb]
            b=v[pick].mean(axis=1);m=(v.sum(axis=0)-b*nb)/(nn-nb);null[start:start+sz]+=m-b
    return np.mean(observed,axis=0),null/len(observed)

def orientation_values(a,angles=None):
    vec=np.exp(2j*np.deg2rad(a['長軸方向度'].to_numpy() if angles is None else angles))
    boards=pd.factorize(a['日程']+'_'+a['基板'])[0];dates=pd.factorize(a['日程'])[0]
    bn=np.bincount(boards);dn=np.bincount(dates)
    bv=np.bincount(boards,weights=vec.real)+1j*np.bincount(boards,weights=vec.imag)
    dv=np.bincount(dates,weights=vec.real)+1j*np.bincount(dates,weights=vec.imag)
    b=bv[boards]-vec;d=dv[dates]-vec
    valid=(bn[boards]>=3)&(dn[dates]>=3)&(abs(b)>1e-12)&(abs(d)>1e-12)
    val=np.real(vec*np.conj(b))/np.maximum(abs(b),1e-12)-np.real(vec*np.conj(d))/np.maximum(abs(d),1e-12)
    return val,valid

def orientation_test(a,method):
    a=a.reset_index(drop=True);v,ok=orientation_values(a)
    if not ok.any():return dict(定義=method,検定='塊長軸基板内一致',視野数=0,補正前有意確率=np.nan,ボンフェローニ補正後=np.nan)
    ob=float(v[ok].mean());angles=a['長軸方向度'].to_numpy();random=rng('orientation');groups=[np.flatnonzero((a['日程']==d).to_numpy()) for d in a['日程'].unique()];ge=0
    for _ in range(10000):
        perm=angles.copy()
        for ix in groups:perm[ix]=random.permutation(angles[ix])
        z,valid=orientation_values(a,perm);ge+=float(z[valid].mean())>=ob-1e-14 if valid.any() else True
    p=(1+ge)/10001
    out=a.loc[ok].copy();out['基板一致引く日程一致']=v[ok];csv(out,'orientation_fields_'+method+'.csv')
    return dict(定義=method,検定='塊長軸基板内一致',視野数=int(ok.sum()),平均差=ob,補正前有意確率=p,ボンフェローニ補正後=min(1,p*10))

def summarize():
    a=pd.read_csv(OUT/'cluster_fields.csv',dtype={'日程':str,'基板':str});o=pd.read_csv(OUT/'cluster_objects.csv',dtype={'日程':str,'基板':str})
    ref=set(table(RESUME/'round5_reference_blank_fields.csv').key);a['分子条件区分']=np.where(a.key.isin(ref),'参考ブランク',np.where(a['ブランク'],'主ブランク','分子あり'))
    rows=[];tests=[];sizes=[]
    for (method,date,label),g in a.groupby(['定義','日程','分子条件区分']):
        objs=o[(o['定義']==method)&o.key.isin(g.key)]
        row=dict(定義=method,日程=date,分子条件区分=label,視野数=len(g),塊を持つ視野数=int(g['塊あり'].sum()),塊を持つ視野割合=float(g['塊あり'].mean()),塊総数=int(g['塊数'].sum()),塊本数中央値=objs['ピラー数'].median(),塊本数第1四分位=objs['ピラー数'].quantile(.25),塊本数第3四分位=objs['ピラー数'].quantile(.75),塊本数最大=objs['ピラー数'].max())
        for col in ('塊数','最大塊本数','大塊総本数'):
            row[col+'平均']=g[col].mean();row[col+'中央値']=g[col].median();row[col+'第1四分位']=g[col].quantile(.25);row[col+'第3四分位']=g[col].quantile(.75)
        rows.append(row)
    for method in METHODS:
        g=a[(a['定義']==method)&~a.key.isin(ref)];cols=['塊あり','塊数','最大塊本数','大塊総本数'];ob,null=permute_groups(g,cols)
        for j,col in enumerate(cols):
            p=(1+int((abs(null[:,j])>=abs(ob[j])-1e-14).sum()))/10001
            tests.append(dict(定義=method,検定='群差_'+col,視野数=len(g),分子あり視野数=int((~g['ブランク']).sum()),主ブランク視野数=int(g['ブランク'].sum()),平均差=ob[j],補正前有意確率=p,ボンフェローニ補正後=min(1,p*10)))
        largest=o[(o['定義']==method)&(o['塊番号']==1)&o['方向利用可能']];tests.append(orientation_test(largest,method))
    csv(rows,'cluster_by_date.csv');csv(tests,'cluster_tests.csv');csv(a[a.key.isin(ref)],'cluster_reference_blank.csv')
    # Position number and within-image locations: no pair-unit inference.
    position=[]
    for (method,pos),g in a.groupby(['定義','位置番号']):
        objs=o[(o['定義']==method)&o.key.isin(g.key)]
        position.append(dict(定義=method,位置番号=pos,視野数=len(g),塊視野割合=g['塊あり'].mean(),塊数平均=g['塊数'].mean(),塊縁距離中央値=objs['縁距離中央値'].median(),塊中心距離中央値=objs['中心距離'].median()))
    csv(position,'cluster_positions.csv')
    pairs=[]
    for (method,date,board),g in o[o['塊番号']==1].groupby(['定義','日程','基板']):
        rec=g.to_dict('records');mi=METHODS.index(method)
        for i,r in enumerate(rec):
            for s in rec[i+1:]:
                ar=np.load(OUT/'cluster_checkpoints'/(r['key']+'.npz'))['maps'][mi]
                br=np.load(OUT/'cluster_checkpoints'/(s['key']+'.npz'))['maps'][mi]
                valid=np.isfinite(ar)&np.isfinite(br);aa=ar[valid];bb=br[valid]
                union=(aa>0)|(bb>0);inter=(aa>0)&(bb>0)
                pairs.append(dict(定義=method,日程=date,基板=board,視野1=r['key'],視野2=s['key'],重心距離=np.hypot(r['重心横']-s['重心横'],r['重心縦']-s['重心縦']),長軸角度差=abs((r['長軸方向度']-s['長軸方向度']+90)%180-90),長短軸比差=abs(r['長短軸比']-s['長短軸比']),塊密度相関=float(np.corrcoef(aa,bb)[0,1]) if np.std(aa)>0 and np.std(bb)>0 else np.nan,塊升目重なり=float(inter.sum()/union.sum()) if union.any() else np.nan))
    csv(pairs,'cluster_within_board_pairs_descriptive.csv')
    print('cluster summary',len(a),len(o),flush=True)

def point_sample(a,xy):
    result=[]
    for start in range(0,len(xy),30000):
        part=xy[start:start+30000].astype('float32');result.append(cv2.remap(a.astype('float32'),part[:,0,None],part[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=float('nan')).ravel())
    return np.concatenate(result)
def image_values(raw,xy):
    a=raw.astype('float32');mean=cv2.blur(a,(3,3));background=cv2.GaussianBlur(a,(0,0),16);contrast=mean-background;lap=cv2.blur(cv2.Laplacian(a,cv2.CV_32F)**2,(7,7))
    return {name:point_sample(im,xy) for name,im in [('生輝度',mean),('平滑化背景',background),('背景差',contrast),('二階微分二乗',lap)]}
def control_indices(xy,members,all_large,key):
    distance=cKDTree(xy[all_large]).query(xy,k=1)[0]
    edge=np.minimum.reduce([xy[:,0],2048-xy[:,0],xy[:,1],2044-xy[:,1]])
    radius=np.linalg.norm(xy-[1024,1022],axis=1)
    strata=(edge//128).astype(int)*32+(radius//128).astype(int)
    available=(distance>=128)&~all_large;random=rng(key);chosen=[]
    for s in np.unique(strata[members]):
        number=int((strata[members]==s).sum());candidates=np.flatnonzero(available&(strata==s))
        if len(candidates):chosen.extend(random.choice(candidates,min(number,len(candidates)),replace=False))
    return np.asarray(chosen,int)

def describe_images(limit=0):
    verify();manifest=raw_index();rec=recovery_index();_,_,caches,_=discover();f=fields().set_index('key',drop=False);objects=pd.read_csv(OUT/'cluster_objects.csv')
    dest=OUT/'cluster_image_checkpoints';dest.mkdir(exist_ok=True);keys=objects.key.unique();started=time.monotonic()
    for j,key in enumerate(keys[:limit or len(keys)]):
        p=dest/(key+'.json')
        if p.exists():continue
        cache=arrays(caches[key]);a,state=recover(key,cache,rec);rows=[]
        if a is None:dump(dict(key=key,**state,metrics=[]),p);continue
        paths=manifest[key]
        if not all(paths.values()):dump(dict(key=key,回復=False,理由='生画像組不足',metrics=[]),p);continue
        xy=cache['xy'];pre=read_image(paths['洗浄前']);post=read_image(paths['洗浄後']);matrix=a['matrix'];postxy=xy@matrix[:,:2].T+matrix[:,2]
        values=[image_values(pre,xy),image_values(post,postxy)];labels=np.load(OUT/'cluster_checkpoints'/(key+'.npz'))['labels']
        for mi,method in enumerate(METHODS):
            for ci in np.unique(labels[mi]):
                if not ci:continue
                inside=np.flatnonzero(labels[mi]==ci);outside=control_indices(xy,inside,labels[mi]>0,f'{key}/{method}/{ci}')
                for metric in values[0]:
                    good=np.isfinite(values[0][metric])&np.isfinite(values[1][metric]);inn=inside[good[inside]];out=outside[good[outside]]
                    row=dict(**meta(f.loc[key]),定義=method,塊番号=int(ci),局所指標=metric,内部点数=len(inn),対照点数=len(out),対照不足=len(out)<50,洗浄前パス=str(paths['洗浄前']),洗浄後パス=str(paths['洗浄後']))
                    for t,name in enumerate(('前','後')):
                        vi=values[t][metric][inn];vo=values[t][metric][out]
                        row[name+'内部平均']=float(vi.mean()) if len(vi) else np.nan;row[name+'内部分散']=float(vi.var()) if len(vi) else np.nan
                        row[name+'対照平均']=float(vo.mean()) if len(vo)>=50 else np.nan;row[name+'対照分散']=float(vo.var()) if len(vo)>=50 else np.nan
                        row[name+'標準化平均差']=(row[name+'内部平均']-row[name+'対照平均'])/np.sqrt(row[name+'対照分散']) if row[name+'対照分散']>0 else np.nan
                    for select,name in ((inn,'内部'),(out,'対照')):
                        delta=values[0][metric][select]-values[1][metric][select]
                        row['前引く後'+name+'平均']=float(delta.mean()) if len(select)>=50 else np.nan;row['前引く後'+name+'分散']=float(delta.var()) if len(select)>=50 else np.nan
                    rows.append(row)
        dump(dict(key=key,**state,metrics=rows),p)
        if j%20==0:print('cluster images',j,key,round(time.monotonic()-started,1),flush=True)
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(dest.glob('*.json'))]
    csv([r for a in records for r in a['metrics']],'cluster_image_metrics.csv');csv([{k:v for k,v in a.items() if k!='metrics'} for a in records],'cluster_image_coverage.csv')
    if any(a['metrics'] for a in records):
        g=pd.read_csv(OUT/'cluster_image_metrics.csv');bg=g[g['局所指標']=='平滑化背景'].copy()
        valid=bg['前標準化平均差'].notna()&bg['後標準化平均差'].notna();pre=abs(bg['前標準化平均差'])>=.5;post=abs(bg['後標準化平均差'])>=.5
        bg['局所背景記述']=np.where(~valid,'比較不能',np.where(pre&post,'前後とも局所差あり',np.where(pre,'前だけ局所差あり',np.where(post,'後だけ局所差あり','前後とも目安未満'))))
        csv(bg,'cluster_background_presence_descriptive.csv')
    print('cluster image coverage',len(records),sum(r['回復'] for r in records),flush=True)

def pilot():
    grid=np.array([(x,y) for x in range(24) for y in range(24)])
    c,s=large_components(grid,np.ones(len(grid),bool));assert len(c)==1 and len(c[0])==576
    c,s=large_components(grid,np.arange(len(grid))<491);assert not c
    geometry(3)
    dump({'六近傍確認':True,'492本基準確認':True,'周5保存数値一致':True},OUT/'cluster_pilot.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','geometry','summarize','images']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.stage=='geometry':geometry(a.limit)
    elif a.stage=='images':describe_images(a.limit)
    else:globals()[a.stage]()
