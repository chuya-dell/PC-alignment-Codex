"""Compare saved incorrect/corrected controls without regenerating any image."""
from field_round8_resume import configure, RESUME, PRIOR
c=configure()
from field_round8_common import *
from field_round8_evaluate import GROUPS

def audit():
    verify()
    signed=pd.read_csv(RESUME/'signed_field_metrics.csv',dtype={'日程':str,'基板':str})
    tests=pd.read_csv(RESUME/'confirmatory_tests.csv')
    rows=[];before=[]
    for path in sorted((RESUME/'rotation_control_before_correction').glob('*.npz')):
        key=path.stem
        with np.load(RESUME/'experiment_checkpoints'/(key+'.npz')) as z:
            xy=z['xy'];actual=z['actual_delta'];main=z['主帰無_delta'];corrected=z['N2_delta']
        with np.load(path) as z:legacy=z['N2_delta']
        primary=signed[signed.key.eq(key)&signed['条件'].eq('主帰無')].iloc[0]
        meta={name:primary[name] for name in ['日程','基板','ブランク','固定29視野','必須対象','通常選択','指定3視野']}
        realmap,_=binned(xy,actual);mainmap,_=binned(xy,main)
        for label,delta in [('修正前',legacy),('修正後',corrected)]:
            ok=np.isfinite(actual)&np.isfinite(delta)
            real,_=binned(xy,np.where(ok,actual,np.nan));sim,_=binned(xy,np.where(ok,delta,np.nan))
            shared=np.isfinite(realmap)&np.isfinite(mainmap)&np.isfinite(sim)
            difference=corr(np.where(shared,realmap,np.nan),np.where(shared,mainmap,np.nan))-corr(np.where(shared,realmap,np.nan),np.where(shared,sim,np.nan))
            pair=paired_map(real,sim)
            row=dict(key=key,**meta,対照版=label,相関=pair['全面'],主対N2共通相関差=difference,
                     **variance_metrics(real,sim),**spectral_pair(real,sim))
            rows.append(row)
            if label=='修正前':before.append(row)
            else:
                saved=signed[signed.key.eq(key)&signed['条件'].eq('N2')].iloc[0]
                assert abs(saved['相関']-row['相関'])<1e-12
                assert abs(primary['主対N2共通相関差']-difference)<1e-12
    legacy=pd.DataFrame(before);metrics=pd.DataFrame(rows)
    metrics.to_csv(RESUME/'rotation_before_after_field_metrics.csv',index=False,encoding='utf-8-sig')
    comparison=[];beforetests=tests.copy()
    for group,select in GROUPS.items():
        fields=legacy[select(legacy)]
        mean,p,n=signflip(fields['主対N2共通相関差'])
        _,pb,_=signflip(fields['主対N2共通相関差'],fields['日程']+'_'+fields['基板'])
        index=tests.index[tests['集合'].eq(group)&tests['指標'].eq('主対N2共通相関差')][0]
        assert n==tests.loc[index,'視野数']
        updates={'平均差':mean,'補正前有意確率':p,'補正後有意確率':min(1,50*p),
                 '基板一括補正前有意確率':pb,'基板一括補正後有意確率':min(1,50*pb)}
        for column,value in updates.items():beforetests.loc[index,column]=value
        comparison.append(dict(集合=group,視野数=n,修正前平均差=mean,修正後平均差=tests.loc[index,'平均差'],
                               修正前補正前有意確率=p,修正後補正前有意確率=tests.loc[index,'補正前有意確率'],
                               修正前補正後有意確率=min(1,50*p),修正後補正後有意確率=tests.loc[index,'補正後有意確率'],
                               有意判定変更=(min(1,50*p)<.05)!=(tests.loc[index,'補正後有意確率']<.05)))
    pd.DataFrame(comparison).to_csv(RESUME/'rotation_before_after_tests.csv',index=False,encoding='utf-8-sig')
    beforetests.to_csv(RESUME/'confirmatory_tests_rotation_before.csv',index=False,encoding='utf-8-sig')
    def correlation_gate(tests):
        return any(tests[tests['集合'].eq(g)&tests['定義'].eq('符号付き差')]['補正後有意確率'].lt(.05).all() for g in ['必須全体','固定29該当'])
    decision=json.loads((RESUME/'decision.json').read_text(encoding='utf-8'))
    old_signed=pd.read_csv(PRIOR/'signed_field_metrics.csv',dtype={'日程':str,'基板':str})
    current_signed=signed.set_index(['key','条件'])
    other=old_signed[~old_signed['条件'].eq('N2')].set_index(['key','条件'])
    columns=[name for name in other.select_dtypes(include=[np.number]).columns if name!='主対N2共通相関差']
    old_numeric=other[columns].to_numpy(float);new_numeric=current_signed.loc[other.index,columns].to_numpy(float)
    unchanged=np.allclose(old_numeric,new_numeric,rtol=1e-12,atol=1e-12,equal_nan=True)
    assert unchanged
    status=dict(比較視野数=len(legacy),全適格視野数=179,再開追加9視野='修正済み実装だけを実行。修正前対照は存在せず、比較対象に含めない',
                五確認集合の視野欠落なし=True,修正前相関四比較支持=bool(correlation_gate(beforetests)),
                修正後相関四比較支持=bool(correlation_gate(tests)),
                修正前主因支持=bool(correlation_gate(beforetests) and decision['固定29主因再現条件数']>2),
                修正後主因支持=decision['主因支持'],
                有意判定変更数=int(sum(r['有意判定変更'] for r in comparison)),
                旧途中評価との他条件数値照合行数=len(other),旧途中評価との他条件数値不変=bool(unchanged),
                修正時他条件全配列不変の独立照合='未確認：修正前保存はN2差と変換・記録指紋のみで、他条件全配列の修正前内容指紋がない。再開以降の旧成果全ファイル不変は別途照合',
                再計算理由='修正前の保存差から90度対照の評価と関連5検定を再構成。模擬像・追跡・他条件の再生成なし。確認的50枠は固定。')
    dump(status,RESUME/'rotation_correction_verification.json')
    print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':audit()
