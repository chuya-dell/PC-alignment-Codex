"""Predicted sampling maps, independent tracking, and spatial prediction tests."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse, json, time
import numpy as np
import pandas as pd
import cv2
from field_round1_analysis import OUT, ROOT, METHODS, fields, load, plotting
from field_round1_bf_recovery import DEST, collect, verify, digest, csv
from field_round1_mechanisms import read_image, sample, binned

MD=OUT/'bf_models'
MD.mkdir(exist_ok=True)
B_NAMES=['pre_x','pre_y','post_x','post_y','pre_x_squared','pre_y_squared','post_x_squared','post_y_squared',
         'post_minus_pre_x','post_minus_pre_y','difference_x_squared','difference_y_squared','template_delta']
F_NAMES=['residual_x','residual_y','residual_x_squared','residual_y_squared','slope_delta']

def checkpoint(key):
    return json.loads((DEST/(key+'.json')).read_text(encoding='utf-8'))

def template(pre,xy,postxy,matrix,basis,origin):
    inv=np.linalg.inv(matrix[:,:2].astype(float))
    reciprocal=np.linalg.inv(basis)
    vectors=np.array([reciprocal[0],reciprocal[1],reciprocal[0]+reciprocal[1]])
    flat=np.arange(0,pre.size,17)
    p=np.column_stack([flat%pre.shape[1],flat//pre.shape[1]])
    phases=(p-origin)@vectors.T*2*np.pi
    design=np.column_stack([np.ones(len(p)),np.cos(phases),np.sin(phases)])
    coef=np.linalg.lstsq(design,(pre.astype(np.float32)/65535).ravel()[flat],rcond=None)[0]
    kernel=cv2.getGaussianKernel(51,0).ravel()
    offsets=np.arange(-25,26)
    def transfer(v):
        return (np.cos(2*np.pi*v[:,0,None]*offsets)@kernel)*(np.cos(2*np.pi*v[:,1,None]*offsets)@kernel)
    pre_gain=1-transfer(vectors)
    post_gain=1-transfer(vectors@inv)
    def sampled(coords,post):
        val=np.zeros(len(coords))
        q=np.rint(coords)
        for oy in (-1,0,1):
            for ox in (-1,0,1):
                points=q+np.array([ox,oy])
                if post:
                    points=(points-matrix[:,2])@inv.T
                ph=(points-origin)@vectors.T*2*np.pi
                gain=post_gain if post else pre_gain
                val+=((np.cos(ph)*coef[1:4]+np.sin(ph)*coef[4:7])*gain).sum(axis=1)
        return val
    return sampled(xy,False)-sampled(postxy,True),coef,vectors

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

def features(limit=0):
    verify();cv2.setNumThreads(1)
    rows=collect();chosen=[r for r in rows if r['回復']]
    if limit:chosen=chosen[:limit]
    ff=fields().set_index('key',drop=False)
    for row in chosen:
        key=row['key'];done=MD/(key+'.json')
        if done.exists():continue
        start=time.monotonic()
        pre,post=read_image(__import__('pathlib').Path(row['洗浄前パス'])),read_image(__import__('pathlib').Path(row['洗浄後パス']))
        with np.load(DEST/(key+'.npz')) as z:
            matrix=z['matrix'];xy=z['xy'];postxy=z['postxy'];basis=z['template_basis'];origin=z['template_origin']
        e0=np.rint(xy)-xy;e1=np.rint(postxy)-postxy;ed=e1-e0
        td,coef,vectors=template(pre,xy,postxy,matrix,basis,origin)
        bp=np.column_stack([e0,e1,e0**2,e1**2,ed,ed**2,td])
        bm=np.stack([binned(xy,bp[:,i]) for i in range(bp.shape[1])],axis=2)
        ta,info=tracking(pre,post,matrix)
        arrays=dict(B=bm,B_names=np.array(B_NAMES),template_coefficients=coef,reciprocal_vectors=vectors,
                    phase_pre=np.stack([binned(xy,e0[:,i]) for i in range(2)],axis=2),phase_post=np.stack([binned(xy,e1[:,i]) for i in range(2)],axis=2),**ta)
        if info['採用候補']:
            yy,xx=np.indices((64,64));cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
            residual=ta['residual_grid'].reshape(-1,2)[cell]
            native=residual@np.linalg.inv(matrix[:,:2]).T
            gx=(sample(pre,xy+np.array([.1,0]),True)-sample(pre,xy-np.array([.1,0]),True))/.2
            gy=(sample(pre,xy+np.array([0,.1]),True)-sample(pre,xy-np.array([0,.1]),True))/.2
            slope=gx*native[:,0]+gy*native[:,1]
            fp=np.column_stack([residual,residual**2,slope])
            arrays['F']=np.stack([binned(xy,fp[:,i]) for i in range(fp.shape[1])],axis=2)
            arrays['F_names']=np.array(F_NAMES)
        info.update(key=key,日程=row['日程'],基板=row['基板'],固定29視野=row['固定29視野'],非周期目印検証=False,秒=time.monotonic()-start)
        np.savez_compressed(MD/(key+'.npz'),**arrays)
        done.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        print('features',key,info['支持点数'],info['採用候補'],round(info['秒'],2),flush=True)
    csv([json.loads(p.read_text(encoding='utf-8')) for p in sorted(MD.glob('26*.json'))],'round1_bf_tracking.csv')

def shifted(a,dy,dx):
    out=np.full_like(a,np.nan)
    ys=slice(max(0,dy),min(64,64+dy));xs=slice(max(0,dx),min(64,64+dx))
    yt=slice(max(0,-dy),min(64,64-dy));xt=slice(max(0,-dx),min(64,64-dx))
    out[ys,xs]=a[yt,xt]
    return out

def controls(a):
    return [shifted(a,0,d) for d in (-16,-8,8,16)]+[shifted(a,d,0) for d in (-16,-8,8,16)]+[np.rot90(a)]

def folds(valid):
    yy,xx=np.indices((64,64));labels=((xx//8)+(yy//8))%8
    out=[]
    for k in range(8):
        test=(labels==k)&valid
        # Embargo is around evaluation blocks, including cells without a response.
        embargo=cv2.dilate((labels==k).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        train=valid&~embargo
        out.append((np.flatnonzero(train),np.flatnonzero(test)))
    return out,labels

def cv_score(y,X,valid):
    split,labels=folds(valid)
    y=y.ravel();X=X.reshape(4096,-1)
    sse=0.;base=0.;pred=np.full(4096,np.nan)
    for tr,te in split:
        if len(te)==0:continue
        if len(tr)<20:return np.nan,pred.reshape(64,64)
        xm=X[tr].mean(axis=0);xs=X[tr].std(axis=0);xs[xs<1e-12]=1
        a=(X[tr]-xm)/xs;b=(X[te]-xm)/xs
        ym=y[tr].mean()
        coef=np.linalg.solve(a.T@a+np.eye(a.shape[1]),a.T@(y[tr]-ym))
        pr=ym+b@coef
        pred[te]=pr;sse+=float(np.sum((y[te]-pr)**2));base+=float(np.sum((y[te]-ym)**2))
    return (1-sse/base if base>0 else np.nan),pred.reshape(64,64)

def evaluate(limit=0):
    verify();rows=collect();f=fields();index={k:i for i,k in enumerate(f.key)}
    z=np.load(OUT/'common_profile_residuals.npz');res=z['residual'];z.close()
    marker_file=OUT/'bf_marker_review.json'
    markers=json.loads(marker_file.read_text(encoding='utf-8')) if marker_file.exists() else {}
    chosen=[r for r in rows if r['回復'] and (MD/(r['key']+'.json')).exists()]
    if limit:chosen=chosen[:limit]
    ev=OUT/'bf_model_evaluation';ev.mkdir(exist_ok=True)
    for row in chosen:
        key=row['key'];done=ev/(key+'.json')
        if done.exists():continue
        info=json.loads((MD/(key+'.json')).read_text(encoding='utf-8'))
        with np.load(MD/(key+'.npz')) as a:
            B=a['B'];F=a['F'] if 'F' in a.files else None
        allmodels={'B':B}
        if F is not None:
            allmodels.update(F=F,BF=np.concatenate([B,F],axis=2),BF_interaction=np.concatenate([B,F,(B[:,:,-1]*F[:,:,-1])[:,:,None]],axis=2))
        results=[];arrays={}
        for mi,method in enumerate(METHODS):
            y=res[index[key],mi]
            for name,X in allmodels.items():
                finite=np.isfinite(y)&np.isfinite(X).all(axis=2)
                full,pr=cv_score(y,X,finite)
                ctl=controls(X)
                common=finite.copy()
                for c in ctl:common&=np.isfinite(c).all(axis=2)
                real,prc=cv_score(y,X,common)
                scores=[cv_score(y,c,common)[0] for c in ctl]
                result=dict(key=key,日程=row['日程'],基板=row['基板'],濃度=row['濃度'],ブランク=row['ブランク'],固定29視野=row['固定29視野'],
                            定義=method,モデル=name,全面決定係数=full,対照共通領域決定係数=real,対照決定係数中央値=float(np.nanmedian(scores)),
                            対照との差=real-float(np.nanmedian(scores)),全面升目数=int(finite.sum()),対照共通升目数=int(common.sum()),
                            非周期目印検証=bool(markers.get(key,{}).get('approved',False)),F支持条件=bool(info['採用候補']),
                            採用区分='確認的利用可能部分' if name=='B' or markers.get(key,{}).get('approved',False) else '候補残差のみの補助')
                for j,s in enumerate(scores):result[f'対照{j+1}決定係数']=s
                results.append(result)
                arrays[method+'_'+name+'_prediction']=pr;arrays[method+'_'+name+'_common_prediction']=prc
                arrays[method+'_'+name+'_common_valid']=common
            if F is not None:
                sub={r['モデル']:r for r in results if r['定義']==method}
                for domain in ('全面決定係数','対照共通領域決定係数'):
                    b=sub['B'][domain];ff=sub['F'][domain];bf=sub['BF'][domain]
                    for r in results:
                        if r['定義']==method:
                            r[domain+'_B固有']=bf-ff;r[domain+'_F固有']=bf-b;r[domain+'_共通']=b+ff-bf
                            r[domain+'_相互作用改善']=sub['BF_interaction'][domain]-bf
        done.write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
        np.savez_compressed(ev/(key+'.npz'),**arrays,fold_labels=folds(np.ones((64,64),bool))[1])
        print('evaluation',key,len(results),flush=True)
    result=[r for p in sorted(ev.glob('26*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]
    csv(result,'round1_bf_explained_fraction.csv')

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

def tests():
    verify()
    a=pd.read_csv(OUT/'round1_bf_explained_fraction.csv',dtype={'日程':str,'基板':str})
    rows=[];summary=[]
    for method in METHODS:
        for model in ('B','F'):
            for label,outliers in [('全視野の利用可能部分',False),('固定29視野の利用可能部分',True)]:
                s=a[(a['定義']==method)&(a['モデル']==model)]
                if model=='F':s=s[s['非周期目印検証']]
                if outliers:s=s[s['固定29視野']]
                s=s[np.isfinite(s['対照との差'])]
                values=s['対照との差'].to_numpy()
                p=sign_test(values)
                substrate=sign_test(values,(s['日程']+'_'+s['基板']).to_numpy()) if len(s) else np.nan
                day=sign_test(values,s['日程'].to_numpy()) if len(s) else np.nan
                rows.append(dict(仮説=model,定義=method,対象=label,視野数=len(s),元母集団視野数=29 if outliers else 632,
                                 日程数=s['日程'].nunique(),日程基板数=len(set(s['日程']+'_'+s['基板'])),
                                 差平均=float(np.mean(values)) if len(values) else np.nan,差中央値=float(np.median(values)) if len(values) else np.nan,
                                 補正前有意確率=p,ボンフェローニ補正後=min(1,p*8) if np.isfinite(p) else np.nan,
                                 日程基板符号有意確率=substrate,日程符号有意確率=day,
                                 状態='利用可能部分で実施。全母集団への一般化は制限' if len(s) else '未検定：必要な独立検証変数なし'))
    csv(rows,'round1_bf_confirmatory_tests.csv')
    for group,g in a.groupby(['定義','モデル','採用区分']):
        subsets=[('全利用可能',g),('固定29利用可能',g[g['固定29視野']]),('ブランク',g[g['ブランク']])]
        subsets += [('日程'+d,h) for d,h in g.groupby('日程')]
        for label,s in subsets:
            for metric in ['全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差','全面決定係数_B固有','全面決定係数_F固有','全面決定係数_共通','全面決定係数_相互作用改善']:
                chosen=s
                if group[1]=='B' and metric.startswith('全面決定係数_'):
                    # Joint decomposition inherits F's independent validation gate.
                    chosen=s[s['非周期目印検証'] & s['F支持条件']]
                v=chosen[metric].dropna() if metric in chosen else pd.Series(dtype=float)
                summary.append(dict(定義=group[0],モデル=group[1],採用区分=group[2],対象=label,指標=metric,視野数=len(v),
                                    中央値=v.median() if len(v) else np.nan,第1四分位=v.quantile(.25) if len(v) else np.nan,第3四分位=v.quantile(.75) if len(v) else np.nan))
    csv(summary,'round1_bf_explanation_summary.csv')

def marker_figures():
    # Only the seven already visually confirmed markers from the previous run.
    marker=pd.read_csv(OUT/'round1_visually_confirmed_marker_geometry.csv')
    plt=plotting();plt.rcParams['font.family']='Yu Gothic'
    dest=OUT/'bf_marker_review';dest.mkdir(exist_ok=True)
    for _,r in marker.iterrows():
        key=r.key
        if not (DEST/(key+'.npz')).exists():continue
        row=checkpoint(key)
        pre,post=read_image(__import__('pathlib').Path(row['洗浄前パス'])),read_image(__import__('pathlib').Path(row['洗浄後パス']))
        with np.load(DEST/(key+'.npz')) as z:matrix=z['matrix']
        aligned=cv2.warpAffine(post,matrix,(2048,2044),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        # Show complete contextual lines and a crop around their intersection.
        mat=np.array([[1,-r['垂直線横傾き']],[-r['水平線傾き'],1]])
        center=np.linalg.solve(mat,[r['垂直線横切片画素'],r['水平線切片画素']])
        x,y=np.clip(center,[128,128],[1920,1916]).astype(int)
        fig,axes=plt.subplots(2,3,figsize=(14,9))
        for j,(label,im) in enumerate([('洗浄前',pre),('洗浄後を最終変換で重ねた像',aligned)]):
            low,high=np.percentile(im,[1,99.5]);axes[j,0].imshow(im,cmap='gray',vmin=low,vmax=high)
            axes[j,0].set_title(label)
            axes[j,1].imshow(im[y-128:y+128,x-128:x+128],cmap='gray',vmin=low,vmax=high)
            axes[j,1].set_title('非周期の十字線の交点周辺')
        aa=pre.astype(float);bb=aligned.astype(float)
        scale=np.percentile(pre,99.5)
        rgb=np.stack([np.clip(aa/scale,0,1),np.clip(bb/scale,0,1),np.clip(bb/scale,0,1)],axis=2)
        axes[0,2].imshow(rgb);axes[0,2].set_title('赤：洗浄前、青緑：洗浄後')
        axes[1,2].imshow(rgb[y-128:y+128,x-128:x+128]);axes[1,2].set_title('重ね合わせ拡大')
        fig.suptitle(key+'；回復='+str(row['回復']));fig.tight_layout();fig.savefig(dest/(key+'.png'),dpi=130);plt.close(fig)

def comparison():
    verify();f=fields();rows=collect();recovered=[r for r in rows if r['回復']]
    thresholds=[];results=[]
    for date in sorted({r['日程'] for r in recovered}):
        group=[r for r in recovered if r['日程']==date]
        blank=[r for r in group if r['ブランク']]
        total=int(((f['日程']==date)&f['ブランク']).sum())
        if not blank:continue
        for sampling in ('整数丸め','双一次補間'):
            data={}
            for row in group:
                with np.load(DEST/(row['key']+'.npz')) as z:
                    data[row['key']]=z['computed_delta' if sampling=='整数丸め' else 'bilinear_delta'].copy()
            pooled=np.concatenate([data[r['key']][np.isfinite(data[r['key']])] for r in blank])
            med=np.median(pooled);mad=np.median(abs(pooled-med))
            ts=[pooled.mean()+3*pooled.std(),med+3*1.4826*mad]
            for method,th in zip(METHODS,ts):
                thresholds.append(dict(日程=date,標本化=sampling,定義=method,閾値=float(th),回復ブランク視野数=len(blank),元ブランク視野数=total,完全校正=len(blank)==total))
                for row in group:
                    d=data[row['key']];good=np.isfinite(d)
                    with np.load(DEST/(row['key']+'.npz')) as z:xy=z['xy']
                    density=binned(xy,np.where(good,(d>th).astype(float),np.nan))
                    # Windowed directional power is descriptive, not a new classification.
                    from field_round1_analysis import geometry
                    geo=geometry((64,64))
                    signal=np.nan_to_num(density-np.nanmean(density))*np.outer(np.hanning(64),np.hanning(64))
                    power=abs(np.fft.rfft2(signal))**2
                    # geometry returns a mapping in the existing implementation.
                    band=(np.hypot(np.fft.fftfreq(64,d=32)[:,None],np.fft.rfftfreq(64,d=32)[None,:])>=1/1024)&(np.hypot(np.fft.fftfreq(64,d=32)[:,None],np.fft.rfftfreq(64,d=32)[None,:])<=1/128)
                    results.append(dict(key=row['key'],日程=date,基板=row['基板'],固定29視野=row['固定29視野'],ブランク=row['ブランク'],標本化=sampling,定義=method,閾値=float(th),
                                        外れ値割合=float(np.mean(d[good]>th)),有効ピラー数=int(good.sum()),密度分散=float(np.nanvar(density)),
                                        周期域窓付きパワー=float(power[band].sum()),完全校正=len(blank)==total))
    csv(thresholds,'round1_bf_sampling_thresholds.csv');csv(results,'round1_bf_sampling_comparison.csv')

def plots(limit=0):
    plt=plotting();plt.rcParams['font.family']='Yu Gothic'
    ff=fields();idx={k:i for i,k in enumerate(ff.key)}
    with np.load(OUT/'density_maps.npz') as z:density=z['density'].copy()
    records=collect();selected=[r for r in records if r['回復'] and r['固定29視野'] and (MD/(r['key']+'.npz')).exists()]
    if limit:selected=selected[:limit]
    dest=OUT/'bf_overlay';dest.mkdir(exist_ok=True)
    correlations=[]
    for r in selected:
        key=r['key']
        with np.load(MD/(key+'.npz')) as z:
            B=z['B'];F=z['F'] if 'F' in z.files else None
        for mi,method in enumerate(METHODS):
            observed=density[idx[key],mi]
            pred=[('洗浄前の丸め横成分',B[:,:,0]),('洗浄後の丸め横成分',B[:,:,2]),('丸め二乗差の平均',B[:,:,6]+B[:,:,7]-B[:,:,4]-B[:,:,5]),('周期像の予測差',B[:,:,-1])]
            if F is not None:pred.append(('局所残差の斜面予測差（候補）',F[:,:,-1]))
            fig,axes=plt.subplots(2,3,figsize=(13,8));axes=axes.flat
            axes[0].imshow(observed,cmap='magma');axes[0].set_title('外れ値密度')
            for i,(label,m) in enumerate(pred,1):
                valid=np.isfinite(observed)&np.isfinite(m)
                corr=float(np.corrcoef(observed[valid],m[valid])[0,1]) if np.std(m[valid])>0 and np.std(observed[valid])>0 else np.nan
                correlations.append(dict(key=key,定義=method,説明変数=label,相関係数=corr,升目数=int(valid.sum()),区分='記述的補助・升目を独立標本とする検定なし'))
                axes[i].imshow(m,cmap='coolwarm');axes[i].contour(observed,levels=[np.nanpercentile(observed,90)],colors='black',linewidths=.5)
                axes[i].set_title(label+'；相関='+f'{corr:.3f}')
            fig.suptitle(key+' '+method);fig.tight_layout();fig.savefig(dest/(key+'_'+method+'.png'),dpi=130);plt.close(fig)
    csv(correlations,'round1_bf_phase_correlations.csv')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['features','evaluate','tests','markers','comparison','plots']);ap.add_argument('--limit',type=int,default=0);args=ap.parse_args()
    if args.stage=='features':features(args.limit)
    elif args.stage=='evaluate':evaluate(args.limit)
    elif args.stage=='tests':tests()
    elif args.stage=='markers':marker_figures()
    elif args.stage=='comparison':comparison()
    else:plots(args.limit)
