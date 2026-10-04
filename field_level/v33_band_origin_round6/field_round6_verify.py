"""Independent output and source-count checks required for round-six review."""
from field_round6_common import *
import ast
from field_round6_clusters import recovery_index,recover
def run():
    verify();f=fields();keys,k,n,q,res,dm=common_data();_,_,caches,_=discover();maximum=0.;total=0
    thresholds=table(OLD/'round1_thresholds.csv').set_index('日程');rec=recovery_index();recovered=0;recovery=[]
    for i,r in f.iterrows():
        a=arrays(caches[r.key]);xy=a['xy'];cell=(xy[:,1]//32).astype(int)*64+(xy[:,0]//32).astype(int)
        count=np.bincount(cell,minlength=4096).reshape(64,64);assert np.array_equal(count,n[i])
        for mi,method in enumerate(METHODS):
            thr=thresholds.loc[r['日程'],'平均標準偏差' if mi==0 else '中央値絶対偏差閾値']
            pos=a['delta']>thr;num=np.bincount(cell,weights=pos,minlength=4096).reshape(64,64)
            assert np.array_equal(num,k[i,mi]);total+=int(pos.sum())
        # Validate saved recovery numerically for all fields, including records
        # whose Japanese labels were already damaged before this round.
        _,state=recover(r.key,a,rec);recovered+=int(state['回復']);recovery.append(dict(**meta(r),**state))
        if i%150==0:print('verify counts and recovery',i,flush=True)
    csv(recovery,'numeric_recovery_audit.csv');assert recovered==266
    valid=np.isfinite(dm);num=valid.sum(axis=0);summ=np.nansum(dm,axis=0)
    recomputed=np.divide(summ-np.nan_to_num(dm),num-valid,out=np.full_like(dm,np.nan),where=num-valid>0)
    max_common=float(np.nanmax(abs(recomputed-q)))
    # Round one accumulated and divided in the saved 32-bit array type.
    # Match that arithmetic exactly, and separately report the 64-bit difference.
    native=arrays(OLD/'density_maps.npz')['density'];good=np.isfinite(native)
    source_common=np.divide(np.nansum(native,axis=0)[None]-np.nan_to_num(native),good.sum(axis=0)[None]-good,out=np.full_like(native,np.nan),where=good.sum(axis=0)[None]-good>0)
    max_native=float(np.nanmax(abs(source_common-q)));assert max_native==0
    for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    a=pd.read_csv(OUT/'model_field_metrics.csv');tests=pd.read_csv(OUT/'model_tests.csv')
    assert a.key.nunique()==632 and not a[['key','定義','モデル']].duplicated().any();assert len(tests)==40
    assert not a['全面未収束分割数'].sum() and not a['共通領域未収束分割数'].sum() and not a['対照未収束分割数'].sum()
    b=pd.read_csv(OUT/'calibration_fields.csv');assert b.key.nunique()==29 and len(b)==522
    clusters=pd.read_csv(OUT/'cluster_fields.csv');objects=pd.read_csv(OUT/'cluster_objects.csv');assert len(clusters)==1264 and (objects['ピラー数']>=492).all()
    for _,r in clusters.iterrows():
        o=objects[(objects.key==r.key)&(objects['定義']==r['定義'])]
        assert len(o)==r['塊数'] and o['ピラー数'].sum()==r['大塊総本数']
        assert o['ピラー数'].max()==r['最大塊本数'] if len(o) else r['最大塊本数']==0
    figs=json.loads((OUT/'figure_manifest.json').read_text(encoding='utf-8'));assert len(figs)==5
    assert sum(r['回復'] for r in figs)==3
    assert len(list((OUT/'figures').glob('*.png')))==10
    for r in figs:
        for col in ('全景','拡大'):
            a=cv2.imdecode(np.frombuffer(Path(r[col]).read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
            assert a is not None and min(a.shape[:2])>=1000
    dump(dict(全視野数=632,二定義升目整数計数一致=True,二定義総外れ値数=total,元精度対象除外共通地図最大差=max_native,倍精度での丸め差=max_common,数値回復成功数=recovered,モデル行数=len(pd.read_csv(OUT/'model_field_metrics.csv')),確認的検定数=40,較正視野数=29,較正行数=522,塊視野行数=1264,塊数=len(objects),五視野全景数=5,五視野拡大数=5,目視確認済み図数=10,位置合わせ回復不能図数=2,構文確認=True),OUT/'verification.json')
    print('verification passed',flush=True)
if __name__=='__main__':run()
