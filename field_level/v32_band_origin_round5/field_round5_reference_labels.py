"""Prespecified supplemental label audit; all original numerical records retained."""
import sys
sys.dont_write_bytecode=True
from field_round5_resume import *
from field_round5_molecules import permutations, score, METRICS

def verify():
    common.verify()
    p=json.loads((OUT/'reference_label_prediction_sha256.json').read_text(encoding='utf-8'))
    assert digest(Path(p['path']))==p['sha256']

def manifest():
    a=common.table(BASE/'round5_molecule_field_metrics.csv')
    ref=a['濃度'].eq('blank_reference')
    assert a[ref].key.nunique()==7 and a[ref]['日程'].eq('260827').all() and a[ref]['基板'].eq('8').all()
    csv(a[ref],'round5_reference_blank_fields.csv')
    return a[~ref].copy()

def tests():
    verify();a=manifest();rows=[];dates=[]
    for method,g in a.groupby('定義'):
        obs,null=permutations(g,METRICS)
        for j,col in enumerate(METRICS):
            p=(1+int((null[:,j]>=obs[j]-1e-15).sum()))/10001
            critical=float(np.quantile(null[:,j],1-.05/8))
            rows.append(dict(定義=method,指標=col,分子あり視野数=int((~g['ブランク']).sum()),ブランク視野数=int(g['ブランク'].sum()),日程数=g['日程'].nunique(),同重み日程平均差=obs[j],補正前有意確率=p,ボンフェローニ補正後=min(1,p*8),補正後帰無臨界差=critical,加算効果80パーセント検出近似=critical-float(np.quantile(null[:,j],.2))))
        for date,h in g.groupby('日程'):
            for col in METRICS+['帯状候補','うろこ状候補']:
                b=h.loc[h['ブランク'],col];m=h.loc[~h['ブランク'],col]
                dates.append(dict(定義=method,日程=date,指標=col,ブランク視野数=len(b),分子あり視野数=len(m),ブランク平均=b.mean(),分子あり平均=m.mean(),分子あり引くブランク=m.mean()-b.mean(),ブランク中央値=b.median(),分子あり中央値=m.median()))
    csv(rows,'round5_J_reference_excluded_tests.csv');csv(dates,'round5_J_reference_excluded_by_date.csv')
    print('reference-excluded label tests saved',flush=True)

def explain(limit=0):
    verify();a=manifest();f=fields();included=set(a.key);index={k:i for i,k in enumerate(f.key)}
    with np.load(OLD/'common_profile_residuals.npz') as z:res=z['residual'].copy();assert z['keys'].tolist()==f.key.tolist()
    with np.load(OLD/'density_maps.npz') as z:dm=z['density'].copy()
    dest=OUT/'J_reference_checkpoints';dest.mkdir(exist_ok=True)
    todo=f[(f['日程']=='260827')&(~f['ブランク'])&f.key.isin(included)]
    assert len(todo)==40
    for _,r in todo.iloc[:limit or len(todo)].iterrows():
        p=dest/(r.key+'.json')
        if p.exists():continue
        same=(f['日程']==r['日程'])&(f['基板']==r['基板'])
        select=(~same)&(f['日程']==r['日程'])&(f['ブランク']==r['ブランク'])&f.key.isin(included)
        assert select.any()
        rows=[]
        for mi,method in enumerate(METHODS):
            predictor=np.nanmean(dm[select,mi],axis=0)-np.nanmean(dm[~same,mi],axis=0)
            rows.append(dict(key=r.key,日程=r['日程'],基板=r['基板'],ブランク=bool(r['ブランク']),固定29視野=bool(r['外れ値視野3percent以上']),定義=method,同日同群学習不能=False,学習視野数=int(select.sum()),モデル='参考ブランク除外後の分子群他基板平均地図',**score(res[index[r.key],mi],predictor)))
        dump(rows,p);print('supplemental J map',r.key,flush=True)
    original=common.table(BASE/'round5_J_explained_fraction.csv')
    changed=pd.DataFrame([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))])
    combined=pd.concat([original[original.key.isin(included)&~original.key.isin(changed.key)],changed],ignore_index=True)
    assert not combined.duplicated(['key','定義']).any()
    csv(combined,'round5_J_reference_excluded_explained_fraction.csv')
    summary=[]
    for method,g in combined.groupby('定義'):
        for target,h in [('全利用可能',g),('固定29利用可能',g[g['固定29視野']]),('ブランク',g[g['ブランク']]),('分子あり',g[~g['ブランク']])]:
            for col in ['全面決定係数','対照共通領域決定係数','対照決定係数中央値','対照との差']:
                v=h[col];summary.append(dict(定義=method,対象=target,指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75)))
    csv(summary,'round5_J_reference_explanation_summary.csv')

def scales():
    verify();ref=set(common.table(BASE/'round5_molecule_field_metrics.csv').query("濃度 == 'blank_reference'").key)
    from field_round5_scales import divide
    a=common.table(BASE/'round5_scale_by_field.csv');s=common.table(BASE/'round5_structure_by_field.csv');rows=[];struct=[]
    for group,g in a.groupby(['定義','分割','窓','波長区間']):
        h=g[(~g['ブランク'])&~g.key.isin(ref)];v=h['共通成分割合']
        rows.append(dict(定義=group[0],分割=group[1],窓=group[2],波長区間=group[3],対象='分子あり',視野数=len(h),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),共通成分合計比=divide(h['共通成分'].sum(),h['全共通成分'].sum()),区間共通対自動比合計=divide(h['共通成分'].sum(),h['自動パワー平均'].sum()),負の区間共通成分数=int((h['共通成分']<0).sum())))
    for group,g in s.groupby(['定義','分割','窓']):
        h=g[(~g['ブランク'])&~g.key.isin(ref)]
        for col in ['全共通対自動比','周期近傍割合','周期支持区間割合','波長512超割合','波長1024超割合','塊関連寄与割合','塊単独共通割合','塊残部交差寄与割合']:
            v=h[col];struct.append(dict(定義=group[0],分割=group[1],窓=group[2],対象='分子あり',指標=col,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),共通成分加重合計比=divide((v*h['全共通成分']).sum(),h['全共通成分'].sum()),共通周期支持視野数=int((h['共通周期軸数']>0).sum()),負値数=int((v<0).sum()),一超数=int((v>1).sum())))
    for original,new,name in [('round5_scale_summary.csv',rows,'round5_scale_summary.csv'),('round5_structure_summary.csv',struct,'round5_structure_summary.csv')]:
        prior=common.table(BASE/original).copy();prior.loc[prior['対象']=='分子あり','対象']='主ブランク以外（参考ブランク混入）'
        csv(pd.concat([prior,pd.DataFrame(new)],ignore_index=True),name)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['tests','explain','scales']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    explain(a.limit) if a.stage=='explain' else globals()[a.stage]()
