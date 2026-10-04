"""Exactly paired saved artificial bands and split-sample calibration."""
from field_round6_common import *
import argparse
def calibration(limit=0):
    verify();f=fields();keys,k,n,q,res,dm=common_data();index={key:i for i,key in enumerate(keys)}
    examples={p.stem:p for p in (R4/'injection_examples').iterdir() if p.suffix=='.npz'}
    assert set(f[f['固定29視野']].key)==set(examples)
    old=table(R4/'round4_injection.csv');old=old[(old['帯軸度']==40)&(old['周期画素']==400)].set_index(['key','定義','注入法','目標追加割合'])
    split=arrays(R4/'split_maps.npz');assert split['keys'].tolist()==keys
    dest=OUT/'calibration_checkpoints';dest.mkdir(exist_ok=True);started=time.monotonic()
    for j,(_,r) in enumerate(f[f['固定29視野']].iloc[:limit or 29].iterrows()):
        p=dest/(r.key+'.json')
        if p.exists():continue
        ex=arrays(examples[r.key]);i=index[r.key];rows=[]
        for mi,method in enumerate(METHODS):
            X=ex[method+'_既知形'];base=score(k[i,mi],n[i],q[i,mi],X)
            v=old.loc[(r.key,method,'注入なし',0.)]
            rows.append(dict(**meta(r),定義=method,用途='人工帯',注入法='注入なし',強さ=0.,旧全面決定係数=v['全面決定係数'],**base))
            for kind in ('判定前','判定後'):
                for strength in (.01,.03,.1):
                    density=ex[f'{method}_{kind}_{strength}'].astype(float)
                    count=np.rint(np.nan_to_num(density)*n[i]);assert np.nanmax(abs(np.nan_to_num(density)*n[i]-count))<1e-4
                    vals=score(count,n[i],q[i,mi],X);v=old.loc[(r.key,method,kind,strength)]
                    vals['新逸脱度引く旧決定係数']=vals['全面逸脱度改善']-float(v['全面決定係数'])
                    vals['新共通除去逸脱度引く旧決定係数']=vals['全面共通除去逸脱度改善']-float(v['全面決定係数'])
                    rows.append(dict(**meta(r),定義=method,用途='人工帯',注入法=kind,強さ=strength,実現追加割合=v['実現追加割合'],旧全面決定係数=v['全面決定係数'],**vals))
            # Use the existing opposite parity's residual to predict held parity.
            # Every target half's count stays binomial and its own common is restored.
            for target in (0,1):
                den=split['counts'][i,0,target].astype(float);dens=split['density'][i,mi,0,target].astype(float)
                count=np.rint(np.nan_to_num(dens)*den)
                common=dens-split['residual'][i,mi,0,target]
                predictor=split['residual'][i,mi,0,1-target]
                vals=score(count,den,common,predictor)
                rows.append(dict(**meta(r),定義=method,用途='二分割再現性',注入法='格子行偶奇',強さ=0.,予測方向=target,**vals))
        dump(rows,p)
        if j%3==0:print('calibration',j,r.key,round(time.monotonic()-started,1),flush=True)
    csv([r for p in sorted(dest.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))],'calibration_fields.csv')

def summary():
    a=pd.read_csv(OUT/'calibration_fields.csv');rows=[]
    cols=['全面逸脱度改善','全面共通除去逸脱度改善','全面順位相関','全面共通除去順位相関','対照差逸脱度改善','対照差共通除去順位相関','旧全面決定係数','新逸脱度引く旧決定係数','新共通除去逸脱度引く旧決定係数','実現追加割合']
    for group,g in a.groupby(['用途','定義','注入法','強さ']):
        # Split directions are averaged within each field before describing the population.
        h=g.groupby('key')[cols].mean()
        for c in cols:
            v=h[c];rows.append(dict(用途=group[0],定義=group[1],注入法=group[2],強さ=group[3],指標=c,視野数=len(h),有効視野数=int(v.notna().sum()),中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75)))
    csv(rows,'calibration_summary.csv');print('calibration summary',len(a),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','summary']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.stage=='run':calibration(a.limit)
    else:summary()
