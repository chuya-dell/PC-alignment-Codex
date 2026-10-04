"""Inject fixed, independently validated displacement fields into null images."""
from field_round8_common import *
import argparse,traceback

VARIANTS=['主帰無','N0','N1','N2','N3半分','N3二倍','N4']

def rotate_field(u):
    # np.rot90 maps image coordinates (x,y) to (y,63-x).
    # Row vectors must therefore map (dx,dy) to (dy,-dx).
    return np.rot90(u).copy()@np.array([[0.,-1.],[1.,0.]])

def init():
    verify();s=setup();f=s['f'];f.to_csv(OUT/'field_selection.csv',index=False,encoding='utf-8-sig')
    for name in ['lab_notes']:
        for p in s['children'][name].iterdir():
            if p.suffix=='.md':used(p).read_text(encoding='utf-8-sig')
    for name,d in s['results'].items():
        if name.startswith(tuple(f'v{n}_' for n in range(28,35))):
            for p in d.glob('*.md'):used(p).read_text(encoding='utf-8-sig')
    for p in (s['results']['v32_band_origin_round5']/'resume_20261004').glob('*.md'):used(p)
    for d in (ROOT/'field_level').iterdir():
        if d.name.startswith(tuple(f'v{n}_' for n in range(28,35))):
            for p in d.iterdir():
                if p.is_file():used(p)
    original_code(s)
    f.groupby('日程')[['適格','必須対象','固定29視野','通常選択']].sum().to_csv(OUT/'selection_by_date.csv',encoding='utf-8-sig')
    dump(dict(予測内容指紋=PRED_HASH,適格数=int(f['適格'].sum()),必須数=int(f['必須対象'].sum()),入力直下=[p.name for p in LOCAL.iterdir()],解析直下=[p.name for p in s['src'].iterdir()],生画像直下=list(s['raw']),コピー完了印=[n for n in s['raw'] if 'copy_done' in n],版確認='作業・結果・入力解析直下・全ブランチ・タグでv35未使用',ブランチ=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),開始コミット=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),履歴取得='git pull --ff-only は .git/FETCH_HEAD 書込拒否で未実施'),OUT/'preflight.json')
    print('eligible',f['適格'].sum(),'required',f['必須対象'].sum(),'normal',f['通常選択'].sum(),flush=True)

def sample_field(u,points):
    coords=[(points[:,1]-16)/32,(points[:,0]-16)/32]
    return np.column_stack([map_coordinates(u[:,:,j],coords,order=1,mode='nearest',prefilter=False) for j in range(2)])

def inverse_points(points,m,u):
    inv=np.linalg.inv(m[:,:2].astype(float));base=(points-m[:,2])@inv.T;x=base.copy()
    for _ in range(6):x=base-sample_field(u,x)@inv.T
    err=float(np.max(np.linalg.norm(transform(x,m)+sample_field(u,x)-points,axis=1)))
    return x,err

def prepare(pre,m,factor=4,deaperture=False):
    padding=32;a=np.pad(pre.astype(np.float32),padding,mode='reflect');h,w=a.shape
    fy=fft.fftfreq(h)[:,None];fx=fft.rfftfreq(w)[None,:];inv=np.linalg.inv(m[:,:2].astype(float))
    ratio=1/(np.sinc(fx)*np.sinc(fy))
    if not deaperture:ratio*=np.sinc(fx*inv[0,0]+fy*inv[1,0])*np.sinc(fx*inv[0,1]+fy*inv[1,1])
    spec=fft.rfft2(a,workers=1);spec*=ratio.astype(np.float32);filtered=fft.irfft2(spec,s=a.shape,workers=1)
    del a,spec,ratio
    enlarged=resample(resample(filtered,w*factor,axis=1),h*factor,axis=0).astype(np.float32)
    return enlarged

def generate(pre,m,u,enlarged,factor=4):
    h,w=pre.shape;out=np.empty(pre.shape,np.float32);maxerr=0.
    for y in range(0,h,64):
        sy,sx=np.indices((min(64,h-y),w),dtype=float);sy+=y;p=np.column_stack([sx.ravel(),sy.ravel()])
        source,err=inverse_points(p,m,u);maxerr=max(maxerr,err)
        z=map_coordinates(enlarged,[(source[:,1]+32)*factor,(source[:,0]+32)*factor],order=1,mode='reflect',prefilter=False)
        out[y:y+len(sy)]=z.reshape(sy.shape)
    assert maxerr<=1e-4,('inverse not converged',maxerr)
    clipped=float(((out<0)|(out>65535)).mean());raw=np.rint(np.clip(out,0,65535)).astype(np.uint16)
    return raw,dict(逆写像最大残差画素=maxerr,制限画素割合=clipped,補間倍率=factor)

def estimate(pre,post,reg,refine,known,xy):
    coarse,qc=reg.register_image_pair_affine(pre,post,mask_stains=False,return_qc=True)
    m,ref=refine(pre,post,stage='subpixel',initial=coarse,mask_stains=False)
    delta=np.linalg.norm(transform(xy,m)-transform(xy,known),axis=1)
    return m,dict(最終行列=m.tolist(),粗い行列=coarse.tolist(),診断=qc,微小調整=ref,全体変換差中央値画素=float(np.median(delta)),全体変換差最大画素=float(np.max(delta)))

def amplitude(u):
    v=u.reshape(-1,2);center=v.mean(axis=0);mag=np.linalg.norm(v,axis=1);c=v-center
    dux=np.gradient(u,32,axis=1);duy=np.gradient(u,32,axis=0)
    return dict(中心化二乗平均平方根画素=float(np.sqrt(np.mean(np.sum(c*c,axis=1)))),変位二乗平均平方根画素=float(np.sqrt(np.mean(np.sum(v*v,axis=1)))),変位中央値画素=float(np.median(mag)),変位第1四分位画素=float(np.quantile(mag,.25)),変位第3四分位画素=float(np.quantile(mag,.75)),局所勾配最大=float(np.max(np.sqrt(np.sum(dux*dux+duy*duy,axis=2)))))

def aperture_check(pre,m,u,raw):
    """Fixed 512 interior pixels; direct 5x5 Gauss area integral, no fitting."""
    points=rng('aperture_fixed').integers([300,300],[1748,1744],size=(512,2)).astype(float)
    expanded=prepare(pre,m,deaperture=True);node,weight=np.polynomial.legendre.leggauss(5);node/=2;weight/=2
    direct=np.zeros(len(points))
    for i in range(5):
        for j in range(5):
            source,err=inverse_points(points+np.array([node[i],node[j]]),m,u)
            direct+=weight[i]*weight[j]*map_coordinates(expanded,[(source[:,1]+32)*4,(source[:,0]+32)*4],order=1,mode='reflect',prefilter=False)
    target=raw[points[:,1].astype(int),points[:,0].astype(int)].astype(float)
    return dict(標本画素数=len(points),画素平均絶対差=float(np.mean(abs(direct-target))),画素二乗平均平方根差=float(np.sqrt(np.mean((direct-target)**2))),全幅相対二乗平均平方根差=float(np.sqrt(np.mean((direct-target)**2))/65535),画素最大絶対差=float(np.max(abs(direct-target))),注意='局所開口形変化の省略と数値補間・量子化の差を含む')

def build_field(key,s,reg,refine,lat,grid):
    start=time.monotonic();r=s['f'].set_index('key').loc[key];assert r['適格']
    rec=record(s['recovery'][key]);pp=s['native'][rec['洗浄前パス']];qp=s['native'][rec['洗浄後パス']]
    assert digest(pp)==rec['洗浄前指紋'] and digest(qp)==rec['洗浄後指紋']
    pre=read_image(pp);cache=arrays(s['caches'][key]);xy=cache['xy'];before=sampler(pre,xy);m=np.asarray(rec['最終行列'],np.float32)
    ids,gen=grid(lat(pre,7.286),2048,2044,margin=30);lookup={tuple(i):p for i,p in zip(ids,gen)}
    err=float(np.max(abs(np.array([lookup[tuple(i)] for i in cache['ids']])-xy)));assert err<=1e-6
    assert np.array_equal(before,reg.sample_contrast(pre,xy)['contrast'].to_numpy(),equal_nan=True)
    fa=arrays(s['feature'][key]);u=fa['residual_grid'].astype(float);fi=record(s['feature'][key].with_suffix('.json'));assert fi['採用候補']
    donor=s['donor'][key];other=arrays(s['feature'][donor])['residual_grid'].astype(float)
    rot=rotate_field(u)
    fields={'主帰無':u,'N0':np.zeros_like(u),'N1':other,'N2':rot,'N3半分':.5*u,'N3二倍':2*u,'N4':np.roll(u,(8,16),axis=(0,1))}
    meta={k:r[k] for k in ['日程','基板','視野番号','位置番号','濃度','ブランク','参考ブランク','固定29視野','必須対象','通常選択','指定3視野']}
    result=dict(key=key,**meta,場実在パス=str(s['feature'][key]),供給元=donor,供給元場パス=str(s['feature'][donor]),洗浄前パス=str(pp),洗浄後パス=str(qp),保存差パス=str(s['caches'][key]),回復記録パス=str(s['recovery'][key]),元変換=m.tolist(),格子生成最大差=err,場振幅=amplitude(u),追跡検証=fi,目印検証=s['marker'].loc[key].to_dict(),往復誤差中央値画素=float(np.median(fa['fb_error'])),窓差中央値画素=float(np.median(fa['window_error'])),条件={})
    data=dict(xy=xy,actual_delta=cache['delta'],ids=cache['ids'],pre_contrast=before,local_field=u)
    expanded=prepare(pre,m);primary_raw=None
    olddir={p.name:p for p in s['results']['v34_band_origin_round7'].iterdir()}['experiment_checkpoints'];oldfiles={p.name:p for p in olddir.iterdir()}
    old=arrays(oldfiles[key+'.npz']);oldrecord=record(oldfiles[key+'.json'])
    for variant,field in fields.items():
        t=time.monotonic();raw,info=generate(pre,m,field,expanded);info['場振幅']=amplitude(field)
        if variant=='N0':
            # Known-position equality verifies our cached affine aperture preparation.
            delta=before-sampler(raw,transform(xy,m));err0=float(np.nanmax(abs(delta-old['既知変換_delta'])));assert err0==0,err0
            data['N0_delta']=old['主帰無_delta'];info.update(状態='成功',周7既知変換差最大差=err0,周7再推定再利用=True,位置合わせ=oldrecord['条件']['主帰無']['位置合わせ'])
        else:
            try:
                mest,est=estimate(pre,raw,reg,refine,m,xy);data[variant+'_delta']=before-sampler(raw,transform(xy,mest));info.update(状態='成功',位置合わせ=est)
            except Exception as exc:
                data[variant+'_delta']=np.full(len(xy),np.nan);info.update(状態='位置合わせ失敗',理由=repr(exc))
        if variant=='主帰無':
            primary_raw=raw.copy();data['局所追随_delta']=before-sampler(raw,transform(xy,m)+sample_field(u,xy))
            if key in CORE:result['開口感度']=aperture_check(pre,m,u,raw)
        info['秒']=time.monotonic()-t;result['条件'][variant]=info
        if variant=='N2':info['回転整合確認済み']=True
        print(key,variant,info['状態'],round(info['秒'],1),flush=True)
    del expanded
    if key in CORE:
        exp8=prepare(pre,m,factor=8);raw8,info8=generate(pre,m,u,exp8,factor=8);del exp8
        # Keep the 4x estimated matrix fixed for interpolation sensitivity.
        mat=np.asarray(result['条件']['主帰無']['位置合わせ']['最終行列']);d8=before-sampler(raw8,transform(xy,mat));data['8倍_delta']=d8
        result['8倍感度']=dict(画素平均絶対差=float(np.mean(abs(raw8.astype(float)-primary_raw))),地図相関=corr(binned(xy,d8)[0],binned(xy,data['主帰無_delta'])[0]),差平均絶対差=float(np.nanmean(abs(d8-data['主帰無_delta']))))
    post=read_image(qp);audit=before-sampler(post,transform(xy,m));err=float(np.nanmax(abs(audit-cache['delta'])));assert err==0
    result['実測差再現最大差']=err;result['秒']=time.monotonic()-start
    dest=OUT/'experiment_checkpoints';dest.mkdir(exist_ok=True);np.savez_compressed(dest/(key+'.npz'),**data);dump(result,dest/(key+'.json'))
    return result

def pilot():
    verify();s=setup();code=original_code(s)
    # Known non-affine coordinates verify forward/inverse sign and field axes.
    u=np.zeros((64,64,2));y,x=np.indices((64,64));u[:,:,0]=.3*np.sin(x/8);u[:,:,1]=.2*np.cos(y/8)
    m=np.array([[1.0002,.0001,1.2],[-.0002,.9999,-.3]])
    points=np.array([[512.,512.],[1024.,700.],[1600.,1400.]])
    unit=np.zeros((64,64,2));unit[:,:,0]=1
    assert np.array_equal(rotate_field(unit),np.broadcast_to([0.,-1.],unit.shape))
    back,err=inverse_points(transform(points,m)+sample_field(u,points),m,u);assert np.max(abs(points-back))<1e-7
    # True continuous analytic cosine: center-vs-full pixel integration with slow field.
    p=rng('analytic').uniform([300,300],[1748,1744],size=(1000,2));f=np.array([.13,.11]);node,weight=np.polynomial.legendre.leggauss(5);node/=2;weight/=2
    source,_=inverse_points(p,m,u);inv=np.linalg.inv(m[:,:2]);center=np.cos(2*np.pi*(source@f))*np.prod(np.sinc(f@inv))
    direct=np.zeros(len(p))
    for i in range(5):
        for j in range(5):
            q,_=inverse_points(p+np.array([node[i],node[j]]),m,u);direct+=weight[i]*weight[j]*np.cos(2*np.pi*(q@f))
    analytic=float(np.sqrt(np.mean((center-direct)**2)));assert analytic<.001
    dump(dict(逆写像座標最大差=float(np.max(abs(points-back))),逆写像残差画素=err,人工余弦開口相対二乗平均平方根差=analytic),OUT/'pilot_geometry.json')
    if not (OUT/'experiment_checkpoints'/(CORE[0]+'.json')).exists():build_field(CORE[0],s,*code)
    print('pilot passed',flush=True)

STATE=None;CODE=None;COMMONS=None
def initialize_worker():
    global STATE,CODE,COMMONS
    verify();STATE=setup();CODE=original_code(STATE)
    from field_round8_evaluate import load_common
    COMMONS=load_common(STATE)
def worker(key):
    try:
        build_field(key,STATE,*CODE)
        from field_round8_evaluate import evaluate_key
        evaluate_key(key,STATE,COMMONS)
        return key,True
    except Exception as exc:
        dump(dict(key=key,理由=repr(exc),詳細=traceback.format_exc()),OUT/'execution_errors'/(key+'.json'));return key,False
def run(all_fields=False,workers=1,extras=False):
    verify();assert (OUT/'pilot_geometry.json').exists();s=setup();f=s['f'];target=f[f['適格'] if all_fields else f['必須対象']]
    if extras:target=target[~target['必須対象']]
    order=[k for k in sorted(target.key) if not (OUT/'experiment_checkpoints'/(k+'.json')).exists()]
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor,as_completed
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker) as pool:
            futures=[pool.submit(worker,k) for k in order]
            for fut in as_completed(futures):print('saved',*fut.result(),flush=True)
    else:
        initialize_worker()
        for key in order:print('saved',*worker(key),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['init','pilot','run','all','integrity']);a.add_argument('--workers',type=int,default=1);a.add_argument('--extras',action='store_true');args=a.parse_args()
    if args.action in ['init','pilot','integrity']:globals()[args.action]()
    else:run(args.action=='all',args.workers,args.extras)
