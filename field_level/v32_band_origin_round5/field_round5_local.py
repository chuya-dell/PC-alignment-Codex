"""Unchanged local tracking gates; visual marker audits and field-level tests."""
from field_round5_common import *
from field_round5_methods import tracking,sample,binned,read_image,cv_score,controls,sign_test
from field_round5_molecules import score
import cv2, argparse, time
cv2.setNumThreads(1)
DEST=OUT/'bf_v24_recovery';MD=OUT/'F_features'
def records():return [json.loads(p.read_text(encoding='utf-8')) for p in sorted(DEST.glob('26*.json'))]
def features(limit=0):
    verify();MD.mkdir(exist_ok=True);chosen=[r for r in records() if r['回復']]
    for r in chosen[:limit or len(chosen)]:
        key=r['key'];done=MD/(key+'.json')
        if done.exists():continue
        start=time.monotonic();pre=read_image(Path(r['洗浄前パス']));post=read_image(Path(r['洗浄後パス']))
        with np.load(DEST/(key+'.npz')) as z:matrix=z['matrix'];xy=z['xy']
        arrays,info=tracking(pre,post,matrix)
        if info['採用候補']:
            cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
            residual=arrays['residual_grid'].reshape(-1,2)[cell];native=residual@np.linalg.inv(matrix[:,:2]).T
            gx=(sample(pre,xy+np.array([.1,0]),True)-sample(pre,xy-np.array([.1,0]),True))/.2
            gy=(sample(pre,xy+np.array([0,.1]),True)-sample(pre,xy-np.array([0,.1]),True))/.2
            slope=gx*native[:,0]+gy*native[:,1];fp=np.column_stack([residual,residual**2,slope])
            arrays['F']=np.stack([binned(xy,fp[:,i]) for i in range(5)],axis=2)
        info.update(key=key,日程=r['日程'],基板=r['基板'],ブランク=r['ブランク'],固定29視野=r['固定29視野'],非周期目印検証=False,秒=time.monotonic()-start)
        np.savez_compressed(MD/(key+'.npz'),**arrays);dump(info,done)
        print('F tracking',key,info['採用候補'],round(info['秒'],1),flush=True)
    csv([json.loads(p.read_text(encoding='utf-8')) for p in sorted(MD.glob('*.json'))],'round5_F_tracking.csv')
def plotting():
    os.environ['MPLCONFIGDIR']=str(OUT/'plot_settings')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']='Yu Gothic';return plt
def sheets(unreviewed=False):
    verify();plt=plotting();dest=OUT/'marker_sheets';dest.mkdir(exist_ok=True)
    available={p.stem for p in MD.glob('*.json') if json.loads(p.read_text(encoding='utf-8'))['採用候補']}
    chosen=[r for r in records() if r['key'] in available]
    if unreviewed:
        manual=json.loads((OUT/'manual_marker_locations.json').read_text(encoding='utf-8'))
        rendered_file=OUT/'round5_additional_marker_sheet_index.csv'
        previous=table(rendered_file).to_dict('records') if rendered_file.exists() else []
        rendered={r['key'] for r in previous}
        chosen=[r for r in chosen if r['key'] not in manual and r['key'] not in rendered]
    rows=[]
    for start in range(0,len(chosen),6):
        group=chosen[start:start+6];p=dest/(f'追加全景_{group[0]["key"]}.png' if unreviewed else f'候補全景_{start//6:02}.png')
        fig,axes=plt.subplots(len(group),2,figsize=(11,5*len(group)),squeeze=False)
        for j,r in enumerate(group):
            pre=read_image(Path(r['洗浄前パス']));post=read_image(Path(r['洗浄後パス']))
            with np.load(DEST/(r['key']+'.npz')) as z:m=z['matrix']
            aligned=cv2.warpAffine(post,m,(2048,2044),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            for k,im in enumerate([pre,aligned]):
                lo,hi=np.percentile(im,[1,99.5]);axes[j,k].imshow(im,cmap='gray',vmin=lo,vmax=hi)
                axes[j,k].set_title(r['key']+(' 洗浄前' if k==0 else ' 洗浄後重ね'))
            rows.append(dict(key=r['key'],全景画像=str(p)))
        fig.tight_layout();fig.savefig(p,dpi=90);plt.close(fig)
    csv(previous+rows,'round5_additional_marker_sheet_index.csv') if unreviewed else csv(rows,'round5_marker_sheet_index.csv')
def marker_audit():
    verify();manual=json.loads((OUT/'manual_marker_locations.json').read_text(encoding='utf-8'));plt=plotting();dest=OUT/'marker_crops';dest.mkdir(exist_ok=True)
    rec={r['key']:r for r in records()};rows=[]
    prior_file=OUT/'round5_F_marker_audit.csv'
    prior={r['key']:r for r in table(prior_file).to_dict('records')} if prior_file.exists() else {}
    for key,location in manual.items():
        cx,cy=np.clip(location['center'],[128,128],[1920,1916]).astype(int)
        if key in prior and prior[key]['目印中心横']==cx and prior[key]['目印中心縦']==cy and (dest/(key+'.png')).exists():
            row=prior[key];row['拡大目視確認']=bool(location.get('crop_inspected',False))
            row['F位相検証採用']=bool(row['拡大目視確認'] and row['差距離画素']<7.286/2 and np.isfinite(row['十字線相関最大値']))
            rows.append(row);continue
        r=rec[key];pre=read_image(Path(r['洗浄前パス']));post=read_image(Path(r['洗浄後パス']))
        with np.load(DEST/(key+'.npz')) as z:m=z['matrix']
        aligned=cv2.warpAffine(post,m,(2048,2044),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        x,y=np.clip(location['center'],[128,128],[1920,1916]).astype(int)
        a=cv2.GaussianBlur(pre.astype(np.float32),(0,0),4)[y-110:y+110,x-110:x+110]
        b=cv2.GaussianBlur(aligned.astype(np.float32),(0,0),4)[y-122:y+122,x-122:x+122]
        match=cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED);iy,ix=np.unravel_index(np.argmax(match),match.shape)
        delta=np.array([ix-12,iy-12]);distance=float(np.linalg.norm(delta))
        approved=bool(location.get('crop_inspected',False) and distance<7.286/2 and np.isfinite(match.max()))
        rows.append(dict(key=key,日程=r['日程'],全景目視確認=True,拡大目視確認=bool(location.get('crop_inspected',False)),非周期目印=location['description'],目印中心横=int(x),目印中心縦=int(y),横差画素=int(delta[0]),縦差画素=int(delta[1]),差距離画素=distance,十字線相関最大値=float(match.max()),F位相検証採用=approved))
        scale=np.percentile(pre,99.5);rgb=np.stack([np.clip(pre/scale,0,1),np.clip(aligned/scale,0,1),np.clip(aligned/scale,0,1)],axis=2)
        fig,axes=plt.subplots(1,3,figsize=(10,3.5))
        for ax,im,label in zip(axes,[pre,aligned,rgb],['洗浄前','洗浄後重ね','赤前・青緑後']):
            crop=im[y-128:y+128,x-128:x+128];ax.imshow(crop,cmap='gray');ax.set_title(label);ax.axis('off')
        fig.suptitle(key+' 目印距離='+str(distance));fig.tight_layout();fig.savefig(dest/(key+'.png'),dpi=100);plt.close(fig)
    csv(rows,'round5_F_marker_audit.csv')
def crop_sheets():
    verify();plt=plotting();dest=OUT/'marker_crop_sheets';dest.mkdir(exist_ok=True)
    manual=json.loads((OUT/'manual_marker_locations.json').read_text(encoding='utf-8'))
    keys=[k for k,r in manual.items() if not r.get('crop_inspected',False) and (OUT/'marker_crops'/(k+'.png')).exists()]
    for start in range(0,len(keys),5):
        group=keys[start:start+5];p=dest/('拡大確認_'+group[0]+'.png')
        fig,axes=plt.subplots(len(group),1,figsize=(10,3.5*len(group)),squeeze=False)
        for ax,key in zip(axes[:,0],group):ax.imshow(plt.imread(OUT/'marker_crops'/(key+'.png')));ax.axis('off')
        fig.tight_layout(pad=0);fig.savefig(p,dpi=100);plt.close(fig)
        print(str(p),flush=True)

def evaluate(limit=0):
    verify();f=fields();index={k:i for i,k in enumerate(f.key)}
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual'].copy()
    markers=table(OUT/'round5_F_marker_audit.csv').set_index('key') if (OUT/'round5_F_marker_audit.csv').exists() else pd.DataFrame()
    dest=OUT/'F_evaluation';dest.mkdir(exist_ok=True);chosen=[r for r in records() if (MD/(r['key']+'.json')).exists()]
    for r in chosen[:limit or len(chosen)]:
        info=json.loads((MD/(r['key']+'.json')).read_text(encoding='utf-8'))
        if not info['採用候補']:continue
        key=r['key'];p=dest/(key+'.json')
        if p.exists():continue
        with np.load(MD/(key+'.npz')) as z:X=z['F']
        rows=[]
        for mi,method in enumerate(METHODS):
            y=res[index[key],mi];valid=np.isfinite(y)&np.isfinite(X).all(axis=2)
            full,_=cv_score(y,X,valid);ctl=controls(X);common=valid.copy()
            for c in ctl:common &= np.isfinite(c).all(axis=2)
            real,_=cv_score(y,X,common);values=[cv_score(y,c,common)[0] for c in ctl];median=float(np.nanmedian(values))
            rows.append(dict(key=key,日程=r['日程'],基板=r['基板'],ブランク=bool(r['ブランク']),固定29視野=bool(r['固定29視野']),定義=method,モデル='F',全面決定係数=full,対照共通領域決定係数=real,対照決定係数中央値=median,対照との差=real-median,非周期目印検証=False,新規回復=True,**{f'対照{j+1}決定係数':v for j,v in enumerate(values)}))
        dump(rows,p)
    new=pd.DataFrame([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))])
    if len(new):new['非周期目印検証']=[bool(markers.loc[k,'F位相検証採用']) if k in markers.index else False for k in new.key]
    old=table(OLD/'round1_bf_explained_fraction.csv').query("モデル == 'F'").copy();old['新規回復']=False
    combined=pd.concat([old,new],ignore_index=True);assert not combined.duplicated(['key','定義']).any()
    csv(combined,'round5_F_explained_fraction.csv')
def tests():
    verify();a=table(OUT/'round5_F_explained_fraction.csv');rows=[]
    for method in METHODS:
        for label,outlier in [('全利用可能',False),('固定29利用可能',True)]:
            g=a[(a['定義']==method)&a['非周期目印検証']]
            if outlier:g=g[g['固定29視野']]
            v=g['対照との差'].to_numpy(float);p=sign_test(v)
            rows.append(dict(定義=method,対象=label,視野数=len(g),日程数=g['日程'].nunique(),新規回復視野数=int(g['新規回復'].sum()),差平均=float(np.mean(v)) if len(v) else np.nan,補正前有意確率=p,ボンフェローニ補正後=min(1,p*8) if np.isfinite(p) else np.nan,日程基板共通符号有意確率=sign_test(v,(g['日程']+'_'+g['基板']).to_numpy()) if len(v) else np.nan,日程共通符号有意確率=sign_test(v,g['日程'].to_numpy()) if len(v) else np.nan))
    csv(rows,'round5_F_tests.csv')
def pilot():
    verify();rng=np.random.default_rng(41);X=rng.normal(size=(64,64,2));y=3*X[:,:,0]-2*X[:,:,1]
    good,_=cv_score(y,X,np.ones((64,64),bool));bad,_=cv_score(y,rng.permutation(X.reshape(4096,2)).reshape(X.shape),np.ones((64,64),bool))
    assert good>.999 and bad<.05;features(2);evaluate(2)
    dump(dict(既知応答決定係数=good,並替説明変数決定係数=bad,回復閾値=1e-6,追跡条件変更なし=True),OUT/'round5_F_pilot.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pilot','features','sheets','marker_audit','crop_sheets','evaluate','tests']);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--unreviewed',action='store_true');a=ap.parse_args()
    if a.stage=='sheets':sheets(a.unreviewed)
    elif a.stage in ['features','evaluate']:globals()[a.stage](a.limit)
    else:globals()[a.stage]()
