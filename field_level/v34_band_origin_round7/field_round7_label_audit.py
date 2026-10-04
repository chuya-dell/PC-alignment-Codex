"""Keep original tests; describe the independently verified label exclusion."""
from field_round7_common import *

def main():
    verify();s=setup();p=next(p for p in OUT.iterdir() if p.name.endswith('ラベル監査と補助集計規定.md'))
    pointer=json.loads((OUT/'label_audit_prediction_sha256.json').read_text(encoding='utf-8-sig'))
    assert digest(p)==pointer['sha256']
    reference=table(s['resume']['round5_reference_blank_fields.csv']);refs=set(reference.key)
    f=s['f'].copy();f['参考ブランク']=f.key.isin(refs)
    f['通常分子付き選択']=f['通常選択']&~f['参考ブランク']
    f['参考除外必須対象']=f['必須対象']&~f['参考ブランク']
    f.to_csv(OUT/'field_selection_label_audited.csv',index=False,encoding='utf-8-sig')
    f.loc[f['参考ブランク'],['key','日程','基板','濃度','回復成功','必須対象','通常選択','ブランク','固定29視野','参考ブランク']].to_csv(OUT/'reference_blank_label_audit.csv',index=False,encoding='utf-8-sig')
    assert set(f.loc[f['参考ブランク']&f['必須対象'],'key'])=={'260827_8_6'}
    assert not (f['参考ブランク']&f['固定29視野']).any()
    signed=pd.read_csv(OUT/'signed_field_metrics.csv');density=pd.read_csv(OUT/'outlier_field_metrics.csv')
    signed=signed[signed['条件']=='主帰無'];density=density[density['条件']=='主帰無']
    rows=[]
    for group,col in [('参考除外必須（補助）','必須対象'),('参考除外通常（補助）','通常選択'),('参考除外回復全体（補助）',None)]:
        for frame,cols in [(signed,['相関','相関対照差','主対N1共通相関差','分散比','無調整分散再現']), (density,['実測外れ値割合','模擬外れ値割合','ジャカード係数','密度重なり対照差','全面決定係数','対照中央値決定係数','対照差決定係数','全面共通除去逸脱度改善','対照中央値共通除去逸脱度改善','対照差共通除去逸脱度改善'])]:
            valid=~frame.key.isin(refs)
            if col:valid&=frame[col]
            sub=frame[valid];parts=[('符号付き差',sub)] if frame is signed else list(sub.groupby('定義'))
            for method,g in parts:
                for c in cols:
                    v=g[c].dropna();rows.append(dict(集合=group,定義=method,指標=c,視野数=len(v),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),最小=v.min(),最大=v.max(),負値数=int((v<0).sum())))
    pd.DataFrame(rows).to_csv(OUT/'reference_excluded_metric_summary.csv',index=False,encoding='utf-8-sig')
    dump(dict(元必須数=int(f['必須対象'].sum()),元通常選択数=int(f['通常選択'].sum()),必須混入=['260827_8_6'],参考除外必須数=int((f['必須対象']&~f['参考ブランク']).sum()),参考除外通常数=int((f['通常選択']&~f['参考ブランク']).sum()),参考除外全回復数=int((f['回復成功']&~f['参考ブランク']).sum()),確認的検定追加=False,補助予測指紋=pointer['sha256']),OUT/'label_audit.json')
    print('label audit',int((f['必須対象']&~f['参考ブランク']).sum()),flush=True)

if __name__=='__main__':main()
