"""Null image construction and native-code registration, checkpoint per field."""
from field_round7_common import *
import argparse, traceback

def init():
    verify();s=setup();f=s['f']
    f.to_csv(OUT/'field_selection.csv',index=False,encoding='utf-8-sig')
    for name in ('lab_notes',):
        for p in s['children'][name].iterdir():
            if p.suffix=='.md':used(p).read_text(encoding='utf-8-sig')
    for name,d in s['results'].items():
        if name.startswith(('v28_','v29_','v30_','v31_','v32_','v33_')):
            for p in d.glob('*.md'):used(p).read_text(encoding='utf-8-sig')
            if name.startswith('v32_'):
                for p in d.iterdir():
                    if p.name=='resume_20261004':
                        for q in p.glob('*.md'):used(q).read_text(encoding='utf-8-sig')
    prior=[]
    for d in (ROOT/'field_level').iterdir():
        if d.name.startswith(('v28_','v29_','v30_','v31_','v32_','v33_')):
            for p in d.iterdir():
                if p.is_file():used(p);prior.append(str(p))
    original_code(s)
    groups=f.groupby('日程')[['回復成功','必須対象','固定29視野','通常選択']].sum()
    groups.to_csv(OUT/'selection_by_date.csv',encoding='utf-8-sig')
    dump(dict(予測内容指紋=PRED_HASH,版確認='開始時の版フォルダ、結果フォルダ、入力解析直下、全ブランチ・タグにv34なし',入力直下=[p.name for p in LOCAL.iterdir()],解析直下=[p.name for p in s['src'].iterdir()],生画像直下=list(s['raw']),コピー完了印=[n for n,p in s['raw'].items() if 'copy_done' in n and p.is_file()],回復視野数=int(f['回復成功'].sum()),必須視野数=int(f['必須対象'].sum()),必須固定29数=int((f['必須対象']&f['固定29視野']).sum()),必須ブランク数=int((f['必須対象']&f['ブランク']).sum()),通常数=int(f['通常選択'].sum()),ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),開始コミット=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),環境=dict(Python=sys.version,数値配列=np.__version__,画像処理=cv2.__version__)),OUT/'preflight.json')
    print(groups.to_string());print('required',f['必須対象'].sum(),flush=True)

def estimate(pre,post,reg,refine,known,xy):
    start=time.monotonic()
    coarse,qc=reg.register_image_pair_affine(pre,post,mask_stains=False,return_qc=True)
    m,ref=refine(pre,post,stage='subpixel',initial=coarse,mask_stains=False)
    delta=np.linalg.norm(transform(xy,m)-transform(xy,known),axis=1)
    info=dict(最終行列=m.tolist(),粗い行列=coarse.tolist(),診断=qc,微小調整=ref,既知変換誤差中央値=float(np.median(delta)),既知変換誤差最大=float(np.max(delta)),秒=time.monotonic()-start)
    return m,info

def build_field(key,s,reg,refine,latticefun,gridfun):
    f=s['f'].set_index('key');r=f.loc[key];rec=record(s['recovery'][key]);assert rec['回復']
    pp=s['native'][rec['洗浄前パス']];qp=s['native'][rec['洗浄後パス']]
    pre=read_image(pp);cache=arrays(s['caches'][key]);xy=cache['xy'];actual=cache['delta']
    m=np.asarray(rec['最終行列'],np.float32)
    assert digest(pp)==rec['洗浄前指紋'];assert digest(qp)==rec['洗浄後指紋']
    assert digest(s['caches'][key])==rec['保存差指紋']
    lattice=latticefun(pre,7.286);ids,generated=gridfun(lattice,2048,2044,margin=30)
    gen={tuple(i):p for i,p in zip(ids,generated)}
    matched=np.array([gen[tuple(i)] for i in cache['ids']]);err=float(np.max(abs(matched-xy)))
    assert err<=1e-6
    # Audit the independently implemented comparison sampler against native code.
    va=reg.sample_contrast(pre,xy)['contrast'].to_numpy();before=sampler(pre,xy)
    assert np.array_equal(va,before,equal_nan=True)
    drec=record(s['recovery'][s['donor'][key]]);assert drec['回復']
    m1=phase_donor(m,np.array(drec['最終行列'],np.float32),pre.shape)
    metadata={k:r[k] for k in ('日程','基板','視野番号','位置番号','濃度','ブランク','固定29視野','必須対象','通常選択','指定3視野')}
    result=dict(key=key,**metadata,回復記録パス=str(s['recovery'][key]),洗浄前パス=str(pp),洗浄後パス=str(qp),保存差パス=str(s['caches'][key]),供給元=s['donor'][key],格子生成最大差=err,元変換=m.tolist(),別位相変換=m1.tolist(),条件={},限界={})
    data=dict(xy=xy,ids=cache['ids'],actual_delta=actual,pre_contrast=before)
    primary_raw=None;primary_matrix=None
    # No real post image is available to these construction/estimation calls.
    for variant,mgen in [('主帰無',m),('N0',np.array([[1,0,0],[0,1,0]],np.float32)),('N1',m1),('N2',m)]:
        start=time.monotonic()
        if variant=='N2':
            raw=np.rint(np.clip(primary_raw.astype(float)+rng(key+'noise').normal(0,65.535,pre.shape),0,65535)).astype(np.uint16)
            generation=dict(ノイズ標準偏差=65.535)
        else:raw,generation=synthesize(pre,mgen)
        if variant=='主帰無':primary_raw=raw
        info=dict(生成=generation)
        oracle_post=transform(xy,mgen);oracle=sampler(raw,oracle_post)
        data[variant+'_known_delta']=before-oracle
        try:
            mest,est=estimate(pre,raw,reg,refine,mgen,xy)
            after=sampler(raw,transform(xy,mest))
            data[variant+'_delta']=before-after
            info.update(位置合わせ=est,状態='成功')
            if variant=='主帰無':primary_matrix=mest
        except Exception as exc:
            info.update(状態='位置合わせ失敗',理由=repr(exc));data[variant+'_delta']=np.full(len(xy),np.nan)
        result['条件'][variant]=info
        result['条件'][variant]['全体秒']=time.monotonic()-start
        print(key,variant,info['状態'],round(time.monotonic()-start,2),flush=True)
    data['既知変換_delta']=data['主帰無_known_delta']
    if primary_matrix is not None:
        data['N3_delta']=sampler(pre,xy,True)-sampler(primary_raw,transform(xy,primary_matrix),True)
    else:data['N3_delta']=np.full(len(xy),np.nan)
    # Only now read the real post image, for audit and fidelity assessment.
    post=read_image(qp);postxy=transform(xy,m);observed_post=sampler(post,postxy)
    audit=before-observed_post
    assert np.max(abs(audit-actual))<=1e-6
    model_post=sampler(primary_raw,postxy);ok=np.isfinite(model_post)&np.isfinite(observed_post)
    difference=observed_post[ok]-model_post[ok]
    result['限界']=dict(元差再現最大差=float(np.max(abs(audit-actual))),後像ピラー相関=corr(observed_post,model_post),後像差平均=float(np.mean(difference)),後像差標準偏差=float(np.std(difference)),後像差分位点=np.quantile(difference,[.025,.25,.5,.75,.975]).tolist(),前像鋭さ=sharpness(pre,xy),実測後像鋭さ=sharpness(post,postxy),模擬後像鋭さ=sharpness(primary_raw,postxy))
    data['actual_bilinear_delta']=sampler(pre,xy,True)-sampler(post,postxy,True)
    data['real_post_contrast']=observed_post;data['synthetic_post_contrast']=model_post
    if key in CORE:
        raw8,gen8=synthesize(pre,m,factor=8)
        data['8倍_known_delta']=before-sampler(raw8,postxy)
        result['限界']['8倍補間']=dict(生成=gen8,模擬画素平均絶対差=float(np.mean(abs(raw8.astype(float)-primary_raw))),模擬画素最大絶対差=float(np.max(abs(raw8.astype(float)-primary_raw))),差地図相関=corr(binned(xy,data['8倍_known_delta'])[0],binned(xy,data['既知変換_delta'])[0]),差平均絶対差=float(np.nanmean(abs(data['8倍_known_delta']-data['既知変換_delta']))))
    for variant in ['主帰無','既知変換','N0','N1','N2','N3']:
        result['条件'].setdefault(variant,dict(状態='成功'))
        result['条件'][variant]['差分散評価']=variance_metrics(actual,data[variant+'_delta'])
        result['条件'][variant]['差地図分散評価']=variance_metrics(binned(xy,actual)[0],binned(xy,data[variant+'_delta'])[0])
    dest=OUT/'experiment_checkpoints';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/(key+'.npz'),**data)
    dump(result,dest/(key+'.json'))
    # Durable progress after each complete field, with all construction variants.
    return result

def collect():
    dest=OUT/'experiment_checkpoints';rows=[]
    if not dest.exists():return
    for p in sorted(dest.glob('*.json')):
        r=json.loads(p.read_text(encoding='utf-8'))
        for variant,info in r['条件'].items():
            row={k:v for k,v in r.items() if not isinstance(v,(dict,list))}
            row.update(条件=variant,状態=info.get('状態'),位置誤差中央値=info.get('位置合わせ',{}).get('既知変換誤差中央値',np.nan),位置誤差最大=info.get('位置合わせ',{}).get('既知変換誤差最大',np.nan),**info.get('差地図分散評価',{}))
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'experiment_fields.csv',index=False,encoding='utf-8-sig')

def pilot():
    verify();s=setup();reg,refine,lat,grid=original_code(s)
    rec=record(s['recovery'][CORE[0]]);pre=read_image(s['native'][rec['洗浄前パス']])
    ident=np.array([[1,0,0],[0,1,0]],np.float32)
    numerical,info=synthesize(pre,ident,identity_numerical=True)
    maxerr=int(np.max(abs(pre.astype(int)-numerical.astype(int))))
    assert maxerr<=1
    record0=dict(恒等数値再構成最大画素差=maxerr,恒等数値再構成異なる画素割合=float(np.mean(pre!=numerical)),生成=info)
    dump(record0,OUT/'pilot_identity.json')
    dest=OUT/'experiment_checkpoints';dest.mkdir(exist_ok=True)
    if not (dest/(CORE[0]+'.json')).exists():build_field(CORE[0],s,reg,refine,lat,grid)
    print('pilot passed',flush=True)

WORKER_STATE=None
WORKER_CODE=None

def initialize_worker():
    global WORKER_STATE,WORKER_CODE
    verify();WORKER_STATE=setup();WORKER_CODE=original_code(WORKER_STATE)

def worker_field(key):
    try:
        build_field(key,WORKER_STATE,*WORKER_CODE)
        return key,True
    except Exception as exc:
        dump(dict(key=key,理由=repr(exc),詳細=traceback.format_exc()),OUT/'execution_errors'/(key+'.json'))
        return key,False

def run(all_fields=False,limit=0,extras=False,workers=1):
    verify();assert (OUT/'pilot_identity.json').exists()
    s=setup();reg,refine,lat,grid=original_code(s);f=s['f']
    target=f[f['回復成功'] if all_fields else f['必須対象']]
    if extras:target=target[~target['必須対象']]
    order=([k for k in CORE if k in set(target.key)]+[k for k in sorted(target.key) if k not in CORE])
    order=[k for k in order if not (OUT/'experiment_checkpoints'/(k+'.json')).exists()]
    if limit:order=order[:limit]
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
            futures=[pool.submit(worker_field,k) for k in order]
            for future in as_completed(futures):
                key,success=future.result();print('saved',key,success,flush=True)
        collect();return
    for key in order:
        dest=OUT/'experiment_checkpoints'
        if (dest/(key+'.json')).exists():continue
        try:build_field(key,s,reg,refine,lat,grid)
        except Exception as exc:
            dump(dict(key=key,理由=repr(exc),詳細=traceback.format_exc()),OUT/'execution_errors'/(key+'.json'))
            print('FAILED',key,repr(exc),flush=True)
    collect()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['init','pilot','run','all','integrity']);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--extras',action='store_true');ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    if args.action in ('init','pilot','integrity'):globals()[args.action]()
    else:run(args.action=='all',args.limit,args.extras,args.workers)
