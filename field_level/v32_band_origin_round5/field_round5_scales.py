"""Signed cross-spectrum partition and fixed six-neighbor component description."""
from field_round5_common import *
from field_round5_methods import angular_difference
import argparse
from scipy import fft
BIN_NAMES=['64以下','64超128以下','128超256以下','256超512以下','512超1024以下','1024超']
OFF=((1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1))
yy,xx=np.indices((64,64))
radius=np.hypot(fft.fftfreq(64,d=32)[:,None],fft.rfftfreq(64,d=32)[None,:])
wavelength=np.divide(1,radius,out=np.full_like(radius,np.inf),where=radius>0)
axis=(np.degrees(np.arctan2(np.broadcast_to(fft.fftfreq(64,d=32)[:,None],radius.shape),np.broadcast_to(fft.rfftfreq(64,d=32)[None,:],radius.shape)))+90)%180
mult=np.full(radius.shape,2.);mult[:,0]=1;mult[:,-1]=1
window=np.outer(np.hanning(64),np.hanning(64))
edges=[0,64,128,256,512,1024,np.inf]
masks=[(wavelength>edges[j])&(wavelength<=edges[j+1])&(radius>0) for j in range(6)]
def large_members(ids,positive):
    pts={tuple(map(int,ids[i])):int(i) for i in np.flatnonzero(positive)};large=np.zeros(len(ids),bool);sizes=[]
    while pts:
        seed,i=pts.popitem();stack=[seed];indices=[i]
        while stack:
            x,y=stack.pop()
            for dx,dy in OFF:
                z=(x+dx,y+dy)
                if z in pts:indices.append(pts.pop(z));stack.append(z)
        sizes.append(len(indices))
        if len(indices)>=492:large[indices]=True
    return large,sizes
def density(values,cell,select):
    n=np.bincount(cell[select],minlength=4096);v=np.bincount(cell[select],weights=values[select],minlength=4096)
    return np.divide(v,n,out=np.full(4096,np.nan),where=n>0).reshape(64,64)
def residuals(a):
    valid=np.isfinite(a);n=valid.sum(axis=0);s=np.nansum(a,axis=0)
    common=np.divide(s-np.nan_to_num(a),n-valid,out=np.full_like(a,np.nan),where=n-valid>0)
    return a-common
def clusters(limit=0):
    verify();f=fields();thr=table(OLD/'round1_thresholds.csv').set_index('日程');dest=OUT/'cluster_checkpoints';dest.mkdir(exist_ok=True)
    caches={p.stem:p for p in (discover()/'tables/cached_field_differences').iterdir() if p.suffix=='.npz'}
    for i,r in f.iloc[:limit or len(f)].iterrows():
        p=dest/(r.key+'.json')
        if p.exists():continue
        with np.load(caches[r.key]) as z:d=z['delta'];xy=z['xy'];ids=z['ids']
        cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
        maps=np.empty((2,2,2,64,64),np.float32);rows=[]
        for mi,method in enumerate(METHODS):
            threshold=thr.loc[r['日程'],['平均標準偏差','中央値絶対偏差閾値'][mi]]
            large,sizes=large_members(ids,d>threshold)
            rows.append(dict(key=r.key,定義=method,日程=r['日程'],基板=r['基板'],ブランク=bool(r['ブランク']),固定29視野=bool(r['外れ値視野3percent以上']),最大陽性塊=max(sizes,default=0),大塊数=sum(n>=492 for n in sizes),大塊陽性ピラー数=int(large.sum()),全陽性ピラー数=int((d>threshold).sum())))
            for si,part in enumerate([ids[:,0]%2,(ids[:,0]+ids[:,1])%2]):
                for k in range(2):maps[mi,si,k]=density(large,cell,part==k)
        np.savez_compressed(dest/(r.key+'.npz'),density=maps);dump(rows,p)
        if i%100==0:print('cluster maps',i,r.key,flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'round5_cluster_fields.csv')
    if not limit:
        maps=np.stack([np.load(dest/(r.key+'.npz'))['density'] for _,r in f.iterrows()])
        np.savez_compressed(OUT/'cluster_split_maps.npz',density=maps,residual=residuals(maps),keys=f.key.to_numpy(dtype=str))
def spectra(a,b,win):
    valid=np.isfinite(a)&np.isfinite(b)
    A=fft.rfft2(np.where(valid,a-np.mean(a[valid]),0)*win)
    B=fft.rfft2(np.where(valid,b-np.mean(b[valid]),0)*win)
    cross=(A*np.conj(B)).real*mult;auto=(abs(A)**2+abs(B)**2)/2*mult
    cross[0,0]=0;auto[0,0]=0
    return cross,auto,A,B
def peak_masks(cl,A,B):
    union=np.zeros(radius.shape,bool);bands=np.zeros(radius.shape,bool);axes=0
    if cl['分類'] not in ['帯状候補','うろこ状候補']:return union,bands,axes
    for prefix in ['第一','第二']:
        if prefix=='第二' and cl['分類']!='うろこ状候補':continue
        if not cl[prefix+'周期一致']:continue
        period=float(cl[prefix+'スペクトル周期画素']);ang=float(cl[prefix+'帯軸度'])
        region=(angular_difference(axis,ang)<=15)&(abs(radius-1/period)<=1/2048)&(radius>0)
        if not region.any():continue
        ka=np.argmax(np.where(region,abs(A)**2,-1));kb=np.argmax(np.where(region,abs(B)**2,-1))
        if abs(radius.flat[ka]-radius.flat[kb])>1/2048+1e-15:continue
        if angular_difference(axis.flat[ka],axis.flat[kb])>30:continue
        axes+=1;union|=region
        for m in masks:
            if m.flat[ka] or m.flat[kb]:bands|=m
    return union,bands,axes
def divide(a,b):return float(a/b) if b!=0 and np.isfinite(b) else np.nan
def scales(limit=0):
    verify();f=fields();cl=table(OLD/'round1_spatial_classification.csv').query('縁除外画素 == 0').set_index(['key','定義'])
    with np.load(PREV/'split_maps.npz') as z:res=z['residual'].copy();keys=z['keys'].tolist()
    with np.load(OUT/'cluster_split_maps.npz') as z:cluster=z['residual'].copy();assert z['keys'].tolist()==keys
    assert keys==f.key.tolist();dest=OUT/'scale_checkpoints';dest.mkdir(exist_ok=True)
    for i,r in f.iloc[:limit or len(f)].iterrows():
        p=dest/(r.key+'.json')
        if p.exists():continue
        rows=[];summaries=[]
        meta=dict(key=r.key,日程=r['日程'],基板=r['基板'],ブランク=bool(r['ブランク']),固定29視野=bool(r['外れ値視野3percent以上']))
        for mi,method in enumerate(METHODS):
            for si,split in enumerate(['格子行偶奇','市松']):
                a,b=res[i,mi,si];la,lb=cluster[i,mi,si]
                for label,win in [('ハニング窓',window),('窓なし',1.)]:
                    C,P,A,B=spectra(a,b,win);R,_,_,_=spectra(a-la,b-lb,win);L,_,_,_=spectra(la,lb,win)
                    den=C.sum();peak,peakbin,naxes=peak_masks(cl.loc[(r.key,method)],A,B)
                    summary=dict(**meta,定義=method,分割=split,窓=label,全共通成分=float(den),全自動パワー平均=float(P.sum()),全共通対自動比=divide(den,P.sum()),共通周期軸数=naxes,周期近傍割合=divide(C[peak].sum(),den),周期支持区間割合=divide(C[peakbin].sum(),den),波長512超割合=divide(C[(wavelength>512)&(radius>0)].sum(),den),波長1024超割合=divide(C[(wavelength>1024)&(radius>0)].sum(),den),塊関連寄与割合=divide(den-R.sum(),den),塊単独共通割合=divide(L.sum(),den),塊残部交差寄与割合=divide(den-R.sum()-L.sum(),den))
                    summaries.append(summary)
                    check=0
                    for j,m in enumerate(masks):
                        c=float(C[m].sum());power=float(P[m].sum());check+=c
                        rows.append(dict(**meta,定義=method,分割=split,窓=label,波長区間=BIN_NAMES[j],周波数点数=int(m.sum()),共通成分=c,自動パワー平均=power,全共通成分=float(den),共通成分割合=divide(c,den),区間共通対自動比=divide(c,power),周期ピーク支持区間=bool((peakbin&m).any()),塊関連共通成分=float((C[m]-R[m]).sum())))
                    assert np.isclose(check,den,rtol=1e-10,atol=1e-8)
        dump(dict(bins=rows,summary=summaries),p)
        if i%100==0:print('scale partition',i,r.key,flush=True)
    records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(dest.glob('*.json'))]
    csv([r for v in records for r in v['bins']],'round5_scale_by_field.csv');csv([r for v in records for r in v['summary']],'round5_structure_by_field.csv')
def summary():
    a=table(OUT/'round5_scale_by_field.csv');s=table(OUT/'round5_structure_by_field.csv');rows=[];struct=[]
    def groups(g):return [('全視野',g),('固定29視野',g[g['固定29視野']]),('ブランク',g[g['ブランク']]),('分子あり',g[~g['ブランク']])]
    for group,g in a.groupby(['定義','分割','窓','波長区間']):
        for target,h in groups(g):
            v=h['共通成分割合'];rows.append(dict(定義=group[0],分割=group[1],窓=group[2],波長区間=group[3],対象=target,視野数=len(h),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),共通成分合計比=divide(h['共通成分'].sum(),h['全共通成分'].sum()),区間共通対自動比合計=divide(h['共通成分'].sum(),h['自動パワー平均'].sum()),負の区間共通成分数=int((h['共通成分']<0).sum())))
    for group,g in s.groupby(['定義','分割','窓']):
        for target,h in groups(g):
            for col in ['全共通対自動比','周期近傍割合','周期支持区間割合','波長512超割合','波長1024超割合','塊関連寄与割合','塊単独共通割合','塊残部交差寄与割合']:
                v=h[col];struct.append(dict(定義=group[0],分割=group[1],窓=group[2],対象=target,指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),共通成分加重合計比=divide((v*h['全共通成分']).sum(),h['全共通成分'].sum()),共通周期支持視野数=int((h['共通周期軸数']>0).sum()),負値数=int((v<0).sum()),一超数=int((v>1).sum())))
    csv(rows,'round5_scale_summary.csv');csv(struct,'round5_structure_summary.csv')
def pilot():
    verify();ids=np.array([(x,y) for x in range(24) for y in range(24)])
    large,sizes=large_members(ids,np.ones(len(ids),bool));assert sizes==[576] and large.all()
    large2,sizes2=large_members(ids,np.arange(len(ids))<491);assert not large2.any() and max(sizes2)<=491
    a=np.cos(2*np.pi*xx/8);C,P,_,_=spectra(a,a,window)
    assert np.allclose(C,P) and np.isclose(sum(C[m].sum() for m in masks),C.sum())
    assert np.argmax([C[m].sum() for m in masks])==2
    clusters(2)
    dump(dict(既知256画素周期区間=BIN_NAMES[2],自己共通成分一致=True,分解和一致=True,六近傍576成分=True,成分491以下非採用=True),OUT/'round5_scale_pilot.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','clusters','scales','summary']);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    if args.stage in ['clusters','scales']:globals()[args.stage](args.limit)
    else:globals()[args.stage]()
