"""Resume round five without changing any pre-existing artifact or criterion."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json, argparse, inspect, time
import numpy as np
import pandas as pd
import field_round5_common as common
from field_round5_common import ROOT, LOCAL, OLD, PREV, METHODS, fields, discover, digest

BASE = common.OUT
OUT = BASE/'resume_20261004'
OUT.mkdir(exist_ok=True)
DATES = ['260825','260827','260828','260829','260922','260923','260924','260927']

def dump(obj, path):
    path = Path(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True, default=str), encoding='utf-8')

def csv(rows, name):
    pd.DataFrame(rows).to_csv(OUT/name, index=False, encoding='utf-8-sig')

def table(path):
    path = Path(path)
    if path.parent == BASE and (OUT/path.name).exists():
        path = OUT/path.name
    return common.table(path)

def recovery_records():
    paths = list((BASE/'bf_v24_recovery').glob('26*.json')) + list((OUT/'bf_v24_recovery').glob('26*.json'))
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(paths)]
    assert len({r['key'] for r in rows}) == len(rows)
    return rows

def collect():
    rows = recovery_records()
    csv([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows], 'round5_v24_recovery.csv')
    return rows

def inventory():
    common.verify()
    import field_round5_recovery as rec
    pairs, _, children = rec.pairs_for(DATES+['260926'])
    f = fields()
    old = common.table(OLD/'round1_bf_v24_recovery.csv')
    already = set(old.key) | {r['key'] for r in recovery_records()}
    qc = common.table(discover()/'tables/table_registration_field_qc.csv').set_index(['date','board','field'])
    rows = []
    for r in f.to_dict('records'):
        q = qc.loc[(r['日程'],r['基板'],r['視野番号'])]
        names = [str(q[c]).replace('\\','/').split('/')[-1] for c in ['path_pre','path_post']]
        roots = [p for p in children if p.is_dir() and p.name.startswith(r['日程'])]
        files = {p.name:p for p in roots[0].iterdir() if p.is_file()} if len(roots)==1 else {}
        rows.append(dict(key=r['key'],日程=r['日程'],基板=r['基板'],ブランク=r['ブランク'],固定29視野=r['外れ値視野3percent以上'],
                         実在日程フォルダ=str(roots[0]) if len(roots)==1 else '',必要前像名=names[0],必要後像名=names[1],
                         前像存在=names[0] in files,後像存在=names[1] in files,前後組存在=r['key'] in pairs,
                         前像実在パス=str(files.get(names[0],'')),後像実在パス=str(files.get(names[1],'')),回復試行済み=r['key'] in already))
    a = pd.DataFrame(rows)
    csv(a,'round5_raw_pair_inventory.csv')
    csv(a.groupby('日程').agg(保存視野数=('key','size'),前後組存在=('前後組存在','sum'),回復試行済み=('回復試行済み','sum'),固定29視野数=('固定29視野','sum')).reset_index(),'round5_raw_pair_summary.csv')
    dump(dict(生画像直下=[p.name for p in children],第三コピー完了=any('copy_done_3' in p.name for p in children if p.is_file()),
              日程フォルダ全ファイル数={p.name:sum(1 for q in p.rglob('*') if q.is_file()) for p in children if p.is_dir()}), OUT/'round5_raw_final_inventory.json')
    missing = a[a['前後組存在'] & ~a['回復試行済み']]
    print('remaining native pairs', missing.groupby('日程').size().to_dict(), flush=True)
    return missing

def baseline():
    common.verify()
    target=OUT/'resume_existing_artifacts_sha256.csv'
    assert not target.exists(), 'Existing resume baseline must not be replaced'
    paths=[]
    for n in range(28,33):
        for parent in ['field_level','data/results']:
            folders=[p for p in (ROOT/parent).iterdir() if p.is_dir() and p.name.startswith(f'v{n}_band_origin_round')]
            for folder in folders:
                paths.extend(p for p in folder.rglob('*') if p.is_file() and OUT not in p.parents and p != Path(__file__))
    csv([dict(実在パス=str(p.resolve()),バイト数=p.stat().st_size,再開開始内容指紋=digest(p)) for p in sorted(paths)],target.name)
    missing=inventory()
    dump(dict(再開時既存追加回復試行数=len(recovery_records()),再開時追加回復成功数=sum(r['回復'] for r in recovery_records()),
              既存検証表試行数=json.loads((BASE/'round5_verification.json').read_text(encoding='utf-8'))['新規回復試行数'],
              再開時未試行前後組数=len(missing),再開時未試行日程別=missing.groupby('日程').size().to_dict(),
              作業1='632視野・二定義・八検定・説明率保存済み。再計算しない。',作業3='632視野・二定義・二分割・二窓・六波長区間保存済み。再計算しない。',
              作業2='回復のチェックポイントは集計報告より先まで進んでいる。追加日程未試行と追跡・目印・集計更新が未完了。',
              旧報告='保全。コピー条件の記述矛盾と古い集計を含むため、新しい同名報告を再開結果サブフォルダへ作成する。'),OUT/'resume_start_status.json')
    print('baseline saved',len(paths),flush=True)

def recovery(limit=0):
    common.verify()
    import field_round5_recovery as rec
    completed={r['key'] for r in recovery_records()}
    f=fields(); f=f[f['日程'].isin(DATES)&~f.key.isin(completed)]
    rec.fields=lambda: f
    rec.DEST=OUT/'bf_v24_recovery';rec.DEST.mkdir(exist_ok=True)
    rec.csv=csv;rec.collect=collect
    # Unique run metadata; every existing field is skipped before estimation.
    rec.recovery(DATES,limit)

def features(limit=0):
    common.verify()
    import field_round5_local as local
    orig_md=BASE/'F_features';new_md=OUT/'F_features'
    local.MD=new_md;local.OUT=OUT;local.csv=csv;local.dump=dump
    done={p.stem for p in orig_md.glob('*.json')} | {p.stem for p in new_md.glob('*.json')}
    todo=[r for r in recovery_records() if r['回復'] and r['key'] not in done]
    if limit:todo=todo[:limit]
    for r in todo:
        local.DEST=BASE/'bf_v24_recovery' if (BASE/'bf_v24_recovery'/(r['key']+'.npz')).exists() else OUT/'bf_v24_recovery'
        local.records=lambda r=r:[r]
        local.features()
    rows=[json.loads(p.read_text(encoding='utf-8')) for folder in [orig_md,new_md] for p in sorted(folder.glob('*.json'))]
    assert len({r['key'] for r in rows})==len(rows)
    csv(rows,'round5_F_tracking.csv')

def evaluate(limit=0):
    common.verify()
    import field_round5_local as local
    # All original per-field evaluations are reused; only missing keys are run.
    local.OUT=OUT;local.csv=csv;local.dump=dump;local.table=table
    done={p.stem for folder in [BASE/'F_evaluation',OUT/'F_evaluation'] for p in folder.glob('*.json')}
    todo=[r for r in recovery_records() if r['回復'] and r['key'] not in done]
    if limit:todo=todo[:limit]
    for r in todo:
        local.MD=BASE/'F_features' if (BASE/'F_features'/(r['key']+'.json')).exists() else OUT/'F_features'
        if not (local.MD/(r['key']+'.json')).exists():continue
        local.records=lambda r=r:[r]
        local.evaluate()
    markers=table(BASE/'round5_F_marker_audit.csv').set_index('key')
    rows=[r for folder in [BASE/'F_evaluation',OUT/'F_evaluation'] for p in sorted(folder.glob('*.json')) for r in json.loads(p.read_text(encoding='utf-8'))]
    new=pd.DataFrame(rows)
    if len(new):new['非周期目印検証']=[bool(markers.loc[k,'F位相検証採用']) if k in markers.index else False for k in new.key]
    old=common.table(OLD/'round1_bf_explained_fraction.csv').query("モデル == 'F'").copy();old['新規回復']=False
    combined=pd.concat([old,new],ignore_index=True);assert not combined.duplicated(['key','定義']).any()
    csv(combined,'round5_F_explained_fraction.csv')

def summaries():
    common.verify()
    import field_round5_local as local
    import field_round5_report as report
    local.OUT=BASE;local.table=table;local.csv=csv;local.tests()
    report.OUT=BASE;report.table=table;report.csv=csv
    report.summaries();report.comparison()
    es=common.table(OUT/'round5_explanation_summary.csv')
    es.loc[es['対象']=='分子あり','対象']='主ブランク以外（参考ブランク混入の可能性）'
    csv(es,'round5_explanation_summary.csv')
    gates=common.table(OUT/'round5_recovery_and_F_gates.csv')
    f=fields().set_index('key')
    gates['分子条件区分']=['参考ブランク' if f.loc[k,'濃度']=='blank_reference' else '主ブランク' if f.loc[k,'ブランク'] else '分子あり' for k in gates.key]
    csv(gates.groupby(['日程','分子条件区分','固定29視野']).agg(試行視野数=('回復','size'),回復視野数=('回復','sum'),支持条件視野数=('F支持条件','sum'),目印確認視野数=('F非周期目印確認','sum')).reset_index(),'round5_recovery_molecule_label_coverage.csv')
    supplemental=OUT/'round5_J_reference_excluded_explained_fraction.csv'
    if supplemental.exists():
        ah=common.table(OUT/'round5_all_hypotheses_comparison.csv')
        ah.loc[ah['仮説']=='J','仮説']='J（参考ブランク混入の旧群）'
        data=common.table(supplemental)
        noise=common.table(PREV/'round4_noise_ceiling.csv').query("分割 == '格子行偶奇'").groupby(['key','定義'],as_index=False)['双方向平均決定係数'].first()
        inj=common.table(PREV/'round4_injection.csv');extra=[]
        for method,g in data.groupby('定義'):
            for target,h in [('全利用可能',g),('固定29利用可能',g[g['固定29視野']])]:
                v=h['全面決定係数'];keys=set(h.key)
                n=noise[(noise['定義']==method)&noise.key.isin(keys)]['双方向平均決定係数']
                q=inj[(inj['定義']==method)&inj.key.isin(keys)&(inj['目標追加割合']==.03)]
                extra.append(dict(仮説='J（参考ブランク除外の補助）',定義=method,対象=target,対象視野数=len(h),説明率中央値=v.median(),第1四分位=v.quantile(.25),第3四分位=v.quantile(.75),判定='原因全体は判定不能。ラベル監査後の補助',確度='低',同一集合半数天井中央値=n.median(),同一集合3ポイント人工帯判定前中央値=q[q['注入法']=='判定前']['全面決定係数'].median(),同一集合3ポイント人工帯判定後中央値=q[q['注入法']=='判定後']['全面決定係数'].median()))
        csv(pd.concat([ah,pd.DataFrame(extra)],ignore_index=True),'round5_all_hypotheses_comparison.csv')

def integrity_start():
    target=OUT/'resume_inputs_sha256.csv'
    assert not target.exists(), 'Existing input baseline must not be replaced'
    rows=[]
    for i,p in enumerate(sorted(p for p in LOCAL.rglob('*') if p.is_file())):
        rows.append(dict(実在パス=str(p.resolve()),バイト数=p.stat().st_size,再開開始内容指紋=digest(p)))
        if i%2000==0:print('input hash',i,flush=True)
    csv(rows,target.name)

def sheets():
    """Render only new/unreviewed candidates; automatic locations are never approved."""
    import cv2
    from field_round5_methods import read_image
    import field_round5_local as local
    local.OUT=OUT
    plt=local.plotting()
    manual=json.loads((BASE/'manual_marker_locations.json').read_text(encoding='utf-8'))
    if (OUT/'manual_marker_locations.json').exists():manual.update(json.loads((OUT/'manual_marker_locations.json').read_text(encoding='utf-8')))
    available={p.stem for folder in [BASE/'F_features',OUT/'F_features'] for p in folder.glob('*.json') if json.loads(p.read_text(encoding='utf-8'))['採用候補']}
    dest=OUT/'marker_sheets';dest.mkdir(exist_ok=True)
    candidates=[r for r in recovery_records() if r['key'] in available and r['key'] not in manual and not (dest/(r['key']+'.png')).exists()]
    for r in candidates:
        key=r['key'];pre=read_image(Path(r['洗浄前パス']));post=read_image(Path(r['洗浄後パス']))
        folder=BASE/'bf_v24_recovery' if (BASE/'bf_v24_recovery'/(key+'.npz')).exists() else OUT/'bf_v24_recovery'
        with np.load(folder/(key+'.npz')) as z:m=z['matrix']
        aligned=cv2.warpAffine(post,m,(2048,2044),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        fig,axes=plt.subplots(2,3,figsize=(12,8))
        for row,im in enumerate([pre,aligned]):
            lo,hi=np.percentile(im,[1,99.5])
            axes[row,0].imshow(im,cmap='gray',vmin=lo,vmax=hi);axes[row,0].set_title(key+(' 洗浄前全景' if row==0 else ' 洗浄後重ね全景'))
            # Edges in native coordinates, with full width retained for inspection.
            for col,(sl,label) in enumerate([(slice(0,320),'上端'),(slice(1724,2044),'下端')],1):
                axes[row,col].imshow(im[sl,:],cmap='gray',vmin=lo,vmax=hi,extent=[0,2048,sl.stop,sl.start]);axes[row,col].set_title(label)
        fig.tight_layout();fig.savefig(dest/(key+'.png'),dpi=130);plt.close(fig)
        print('marker sheet',key,flush=True)

def marker_audit():
    common.verify()
    import cv2
    from field_round5_methods import read_image
    import field_round5_local as local
    local.OUT=OUT;plt=local.plotting()
    dest=OUT/'marker_crops';dest.mkdir(exist_ok=True)
    manual=json.loads((OUT/'manual_marker_locations.json').read_text(encoding='utf-8')) if (OUT/'manual_marker_locations.json').exists() else {}
    rec={r['key']:r for r in recovery_records()}
    rows=common.table(BASE/'round5_F_marker_audit.csv').to_dict('records')
    existing={r['key'] for r in rows}
    for key,location in manual.items():
        assert key not in existing, 'Original marker assessments must not be replaced'
        r=rec[key];pre=read_image(Path(r['洗浄前パス']));post=read_image(Path(r['洗浄後パス']))
        folder=BASE/'bf_v24_recovery' if (BASE/'bf_v24_recovery'/(key+'.npz')).exists() else OUT/'bf_v24_recovery'
        with np.load(folder/(key+'.npz')) as z:m=z['matrix']
        aligned=cv2.warpAffine(post,m,(2048,2044),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
        x,y=np.clip(location['center'],[128,128],[1920,1916]).astype(int)
        a=cv2.GaussianBlur(pre.astype(np.float32),(0,0),4)[y-110:y+110,x-110:x+110]
        b=cv2.GaussianBlur(aligned.astype(np.float32),(0,0),4)[y-122:y+122,x-122:x+122]
        match=cv2.matchTemplate(b,a,cv2.TM_CCOEFF_NORMED);iy,ix=np.unravel_index(np.argmax(match),match.shape)
        delta=np.array([ix-12,iy-12]);distance=float(np.linalg.norm(delta))
        approved=bool(location.get('crop_inspected',False) and distance<7.286/2 and np.isfinite(match.max()))
        rows.append(dict(key=key,日程=r['日程'],全景目視確認=True,拡大目視確認=bool(location.get('crop_inspected',False)),非周期目印=location['description'],
                         目印中心横=int(x),目印中心縦=int(y),横差画素=int(delta[0]),縦差画素=int(delta[1]),差距離画素=distance,十字線相関最大値=float(match.max()),F位相検証採用=approved))
        path=dest/(key+'.png')
        if path.exists():continue
        scale=np.percentile(pre,99.5);rgb=np.stack([np.clip(pre/scale,0,1),np.clip(aligned/scale,0,1),np.clip(aligned/scale,0,1)],axis=2)
        fig,axes=plt.subplots(1,3,figsize=(10,3.5))
        for ax,im,label in zip(axes,[pre,aligned,rgb],['洗浄前','洗浄後重ね','赤前・青緑後']):
            ax.imshow(im[y-128:y+128,x-128:x+128],cmap='gray');ax.set_title(label);ax.axis('off')
        fig.suptitle(key+' 目印距離='+str(distance));fig.tight_layout();fig.savefig(path,dpi=100);plt.close(fig)
    csv(rows,'round5_F_marker_audit.csv')

def contacts():
    from PIL import Image
    for folder,label,ncol in [('marker_sheets','full',4),('marker_crops','crop',1)]:
        dest=OUT/'review_contacts';dest.mkdir(exist_ok=True)
        index=dest/(label+'_index.json')
        records=json.loads(index.read_text(encoding='utf-8')) if index.exists() else []
        known={k for r in records for k in r['keys']}
        todo=[p for p in sorted((OUT/folder).glob('*.png')) if p.stem not in known]
        batch=4 if label=='full' else 5
        for start in range(0,len(todo),batch):
            group=todo[start:start+batch]
            ims=[Image.open(p).convert('RGB') for p in group]
            if label=='full':ims=[im.crop((0,0,520,1040)) for im in ims]
            width=max(im.width for im in ims);height=max(im.height for im in ims)
            canvas=Image.new('RGB',(width*(len(ims) if label=='full' else 1),height*(1 if label=='full' else len(ims))),'white')
            for j,im in enumerate(ims):canvas.paste(im,(j*width,0) if label=='full' else (0,j*height))
            path=dest/(label+'_'+group[0].stem+'.png');assert not path.exists();canvas.save(path)
            records.append(dict(path=str(path),keys=[p.stem for p in group]))
            print('contact',path.name,flush=True)
        dump(records,index)

def final_audit():
    common.verify()
    inventory()
    gates=table(OUT/'round5_recovery_and_F_gates.csv')
    pair=table(OUT/'round5_raw_pair_inventory.csv')
    tracks=table(OUT/'round5_F_tracking.csv').set_index('key')
    markers=table(OUT/'round5_F_marker_audit.csv').set_index('key')
    rows=[]
    for r in pair.to_dict('records'):
        k=r['key'];g=gates[gates.key==k]
        if not r['前後組存在']:reason='前後画像組が不足'
        elif not len(g):reason='実在組だが回復未試行'
        elif not bool(g['回復'].iloc[0]):reason='回復試行済み・事前条件不合格'
        elif not bool(g['F支持条件'].iloc[0]):reason='局所追跡・独立支持条件不合格'
        elif not bool(g['F非周期目印確認'].iloc[0]):
            reason='目印目視済み・半周期基準不合格' if k in markers.index and bool(markers.loc[k,'拡大目視確認']) else '独立支持候補・非周期目印未確認'
        else:continue
        rows.append(dict(key=k,日程=r['日程'],理由=reason,ブランク=r['ブランク'],固定29視野=r['固定29視野'],必要前像名=r['必要前像名'],必要後像名=r['必要後像名']))
    csv(rows,'round5_unfinished_fields.csv')
    for file,n in [('round5_molecule_field_metrics.csv',1264),('round5_J_explained_fraction.csv',1264),('round5_structure_by_field.csv',5056),('round5_scale_by_field.csv',30336)]:
        assert len(common.table(BASE/file))==n
    assert not gates.key.duplicated().any()
    good=gates[gates['回復']]
    assert good['ピラー識別一致'].all() and good['洗浄前座標一致'].all() and (good['差最大絶対差']<=1e-6).all()
    fs=table(OUT/'round5_F_explained_fraction.csv')
    assert not fs.duplicated(['key','定義']).any()
    allowed=set(gates[gates['F支持条件']&gates['F非周期目印確認']].key)
    assert set(fs[fs['非周期目印検証']].key)<=allowed
    assert not len(pair[pair['前後組存在']&~pair.key.isin(gates.key)])
    assert len(table(OUT/'round5_J_reference_excluded_explained_fraction.csv'))==1250
    assert len(list((OUT/'J_reference_checkpoints').glob('*.json')))==40
    refs=set(table(OUT/'round5_reference_blank_fields.csv').key)
    assert not set(table(OUT/'round5_J_reference_excluded_explained_fraction.csv').key)&refs
    dump(dict(回復試行数=len(gates),回復成功数=len(good),新規追加試行数=len(list((OUT/'bf_v24_recovery').glob('26*.json'))),
              F確認的視野数=fs[fs['非周期目印検証']].key.nunique(),F候補視野数=fs.key.nunique(),全実在組試行済み=True,
              回復事前基準違反数=0,作業1と3の元計算再計算なし=True,Jラベル補助の再計算視野数=40,
              元のJ説明率再利用視野数=585,追加日程の目印未確認候補数=int(sum(r['日程']!='260926' and r['理由']=='独立支持候補・非周期目印未確認' for r in rows))),OUT/'round5_verification.json')

def integrity_final():
    rows=[]
    for file in ['resume_existing_artifacts_sha256.csv','resume_inputs_sha256.csv']:
        a=common.table(OUT/file)
        for i,r in enumerate(a.to_dict('records')):
            end=digest(Path(r['実在パス']));rows.append(dict(**r,終了内容指紋=end,不変=r['再開開始内容指紋']==end,監査集合=file))
            if i%3000==0:print('final hash',file,i,flush=True)
    csv(rows,'resume_integrity_final.csv')
    a=pd.DataFrame(rows);changed=a[~a['不変']]
    dump(dict(監査ファイル数=len(a),既存成果物と入力の内容変更数=len(changed),予測指紋一致=True,
              追加のみで既存ファイルは保全=not len(changed)),OUT/'resume_integrity_verification.json')
    assert not len(changed),changed['実在パス'].tolist()

def report():
    common.verify()
    from field_round5_report import md
    status=json.loads((OUT/'resume_start_status.json').read_text(encoding='utf-8'))
    lines=['# 2026年10月4日・周5再開報告\n']
    def add(s):lines.append(s)
    add('## 再開時点の完了・未完了（事実、確度：高）\n\n'
        '作業1：632視野・二定義の四指標、八層別検定、群予測地図の説明率、日程別表は保存済み。元の計算は再計算していない。ただし再開時のラベル監査で参考ブランク7視野の混入を確認し、両群から除く補助解析を別規定・別指紋で追加した。\n\n'
        '作業3：632視野・二定義・二分割・二窓・六波長区間の記述、492本以上の陽性塊、集計と図は保存済み。再計算していない。\n\n'
        f"作業2：別日程の回復は視野別記録で{status['再開時既存追加回復試行数']}試行・{status['再開時追加回復成功数']}成功まで保存済みだった。一方、検証表と報告は216試行・93成功、確認的検定は17視野の集計で止まっていた。現在のコピーで未試行の前後組は178視野（260825：74、260827：54、260927：50）。不足試行と追跡・目印確認・集計更新を再開対象とした。\n\n"
        '既存の `261004_周5_報告.md` も途中成果物として保全した。その報告は開始時の第三完了印を「あり」としつつ「ない」と記述する矛盾と古い集計を含む。新しい指定名の報告を `resume_20261004/261004_周5_報告.md` に置く。元報告と予測・規定・指紋を変更しない。')
    rec=pd.DataFrame(recovery_records())
    old=common.table(OLD/'round1_bf_v24_recovery.csv')
    combined=pd.concat([old,rec],ignore_index=True)
    pairs=table(OUT/'round5_raw_pair_inventory.csv')
    pending=pairs[pairs['前後組存在']&~pairs.key.isin(combined.key)]
    missing=pairs[~pairs['前後組存在']]
    add('## 現在の完了状態\n\n'+f'回復試行は260926の旧70視野を含め{len(combined)}視野、成功{int(combined["回復"].sum())}視野。実在前後組で未試行は{len(pending)}視野。前後組不足は{len(missing)}視野。'
        '\n\n'+md(combined.groupby('日程').agg(試行視野数=('key','size'),回復成功数=('回復','sum')).reset_index()))
    if not (OUT/'round5_all_hypotheses_comparison.csv').exists():
        add('**計算途中の保全報告。** 位置合わせ回復、追加追跡と非周期目印確認、集計更新、保全の終了照合、最終報告が未完了。作業1と作業3は再計算せず保存済み結果を使用する。')
        (OUT/'261004_周5_報告.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
        return
    raw=json.loads((OUT/'round5_raw_final_inventory.json').read_text(encoding='utf-8'))
    add('## 入力と運用（事実、確度：高）\n\n'
        '`AGENTS.md`、`docs/REPOSITORY_CONVENTIONS.md`、`docs/LOCAL_ENVIRONMENT_20260926.md`を確認。最新の依頼に従い入力は `data/inputs_local/` のみ。起点と各上位フォルダを一覧して実在名を用いた。凍結リポジトリを参照・実行していない。標準経路・既定値・ゴミやシミのマスクを変更していない。\n\n'
        '生画像直下の実在名：\n\n'+'\n'.join('- `'+n+'`' for n in raw['生画像直下'])+
        '\n\n第三完了印 `_copy_done_3.txt` を確認。260827の実在フォルダは `260827_pp50_dna`。完了印だけで各組の存在を推定せず、保存台帳の元ファイル名を前後両方で照合した。')
    add(md(table(OUT/'round5_raw_pair_summary.csv'))+
        '\n\n不足する正確な前後像名と実在パスは `round5_raw_pair_inventory.csv`。コピー済み日程に欠ける組を、別名・派生画像で代用していない。未試行・欠測は `round5_unfinished_fields.csv` に理由を保存する。')
    source=discover()
    source_files=[source/'tables/cached_field_differences',source/'tables/table_registration_field_qc.csv',source/'tables/table_alignment_quality_features_and_sensitivity_by_field.csv',
                  source/'20260929_外れ値解析/20260929_報告書.md',LOCAL/'code_v24',LOCAL/'lab_notes',OLD/'round1_fields_both_definitions.csv',
                  OLD/'round1_spatial_classification.csv',OLD/'common_profile_residuals.npz',PREV/'split_maps.npz',PREV/'round4_noise_ceiling.csv',PREV/'round4_injection.csv']
    assert all(p.exists() for p in source_files)
    add('主な実在絶対パス：\n\n'+'\n'.join('- `'+str(p.resolve())+'`' for p in source_files)+
        '\n\nラボノートは元の階層が平坦化された八ファイルを参照。塊基準は解析結果の9月29日報告書の「ブランク69視野の95百分位491.8本を切り上げ492本」と六近傍を保持。欠陥の未記載を欠陥なしと判断していない。全入力のパスと再開時指紋は `resume_inputs_sha256.csv`、旧成果物は `resume_existing_artifacts_sha256.csv`。終了照合は `resume_integrity_final.csv`。')
    pred=common.table(BASE/'prediction_sha256.csv').iloc[0]
    add('## 先に保存された予測と規定\n\n'
        '原予測は `../261004_周5_予測と検定規定.md`、256ビット暗号学的内容指紋は `'+pred.Hash+'`。再開の各段階で一致を照合。内容と基準を変更していない。\n\n'
        '仮説J（実際の不均一な分子吸着）：同日程の分子ありで、外れ値割合、帯状・うろこ状候補割合、二分割再現性、共通平均除去後の分散が高い。濃度単調増加は予測しない。前像だけの構造は吸着と整合し得るが分子同定ではなく、後像にも残れば差は弱まり得る。ブランク・260830の構造は他原因を残す。260830は単独像で前後対応不明のため差検定に入れない。\n\n'
        '仮説F（局所位置ずれ）：独立支持・目印条件を満たす残差と前像斜面から作る地図が、外れ値残差地図を空間対照より改善する。ブランクにも生じ得る。単独像では前後の位置差を測れない。\n\n'
        '四指標×二定義の仮説Jの八検定と、仮説Fの周1と同じ補正倍率8を別群とした。波長分解と群地図の説明率は記述であり、検定数を増やしていない。')
    jt=common.table(BASE/'round5_J_tests.csv')
    add('## 作業1：分子あり対ブランク\n\n'
        '元規定の九日程632視野の比較は主ブランク69・主ブランク以外563。後者に260827の基板8の参考ブランク7視野が入るため、全てを分子ありと呼べない。保存表の「分子あり」はこの旧群の列名であり、数値を保全して読み替える。ミスマッチ配列は分子あり。260922・260923の基板01はミスマッチ、260926の01はブランク。二定義は日程別の周1固定閾値を保持。差は洗浄前から洗浄後を引く。\n\n'
        '日程ごとの群平均差（分子ありからブランクを引く）を九日程等重みでまとめ、日程内で視野ラベルを10,000回並べ替えた片側検定。帯状・うろこ状は周1の全面分類の合計指標。再現性は周4の格子行偶奇二分割の双方向平均決定係数。残差分散は対象を除く共通平均除去後の32ピクセル密度地図。割合差0.001は0.1百分率ポイント。\n\n'+md(jt))
    jd=common.table(BASE/'round5_J_by_date.csv')
    for method in METHODS:
        d=jd[jd['定義']==method]
        add('### '+method+'：日程別の記述\n\n'+md(d.pivot(index='日程',columns='指標',values='分子あり引くブランク').reset_index()))
    refhash=json.loads((OUT/'reference_label_prediction_sha256.json').read_text(encoding='utf-8'))
    add('### 参考ブランクを除く補助解析\n\n'
        '参考ブランクを主ブランクへ移さず、両群から除いた。主ブランク69・分子あり556の625視野。日程別の閾値、元の残差、分類、再現性はそのまま。再開時に混入を発見したため、内容指紋を保存した追加規定 `261004_参考ブランク混入の補助解析規定.md` に従う補助解析とし、元の事前検定と区別する。追加指紋：`'+refhash['sha256']+'`。\n\n'+md(table(OUT/'round5_J_reference_excluded_tests.csv')))
    add('群予測の学習からも参考ブランクを除き、共通構造の基準平均は元の632視野を保持。予測地図が変わる260827の分子あり40視野だけ再計算し、残る585視野は保存値を再利用した。元の563群と補助556群を選んで結論を変えていない。補助の日程別両群値と説明率は `round5_J_reference_excluded_by_date.csv` と `round5_J_reference_excluded_explained_fraction.csv`。\n\n'+md(table(OUT/'round5_J_reference_explanation_summary.csv'),percent=['中央値','第1四分位','第3四分位']))
    es=table(OUT/'round5_explanation_summary.csv')
    add('群地図の説明率：対象基板全視野を学習から除き、同日程・同ラベルの他基板平均を使用。同群が存在しない場合は他日程同ラベルへ切り替え、視野別に記録。空間8分割、8×8升目ブロック、評価周囲2升目除外、正則化強さ1、学習平均基準。±256・±512ピクセルの横縦非循環移動と90度回転の九対照を、同じ有効領域で比較。負値を保持。\n\n'+md(es[es['仮説']=='J'],percent=['中央値','第1四分位','第3四分位']))
    ref_tests=table(OUT/'round5_J_reference_excluded_tests.csv')
    add('**判定：吸着という原因全体は判定不能、確度：低。** 元の八検定は全て補正後1。参考ブランクを除く補助解析で補正後0.05未満の検定数は'+str(int((ref_tests['ボンフェローニ補正後']<.05).sum()))+'。数値照合の確度は高。濃度依存性は検定していない。群地図で説明する変動を分子起源の因果割合と呼べない。元の個々の値と両群値は元保存表を保全し、ラベル混入を弱点として明示した。')
    gates=table(OUT/'round5_recovery_and_F_gates.csv')
    fs=table(OUT/'round5_F_explained_fraction.csv')
    add('## 作業2：回復と仮説Fの拡張\n\n'
        '元の `code_v24` の呼出し、マスク無効、微小位置合わせ、前像格子生成、整数丸め標本化を保持し、各未試行組を一度だけ実行。保存値へ合わせる探索・再試行をしていない。識別番号完全一致、前座標最大差0.000001以内、全ピラー差最大差0.000001以内を要求。基準を緩めず、失敗と不一致を保存した。\n\n'+md(gates.groupby('日程').agg(試行視野数=('回復','size'),回復視野数=('回復','sum'),固定29試行=('固定29視野','sum'),独立支持条件視野数=('F支持条件','sum'),目印確認視野数=('F非周期目印確認','sum')).reset_index()))
    add('ブランク・外れ値視野による偏り：\n\n'+md(table(OUT/'round5_recovery_coverage.csv')))
    add('主ブランクと参考ブランクを区別した回復の偏り：\n\n'+md(table(OUT/'round5_recovery_molecule_label_coverage.csv')))
    add('追跡は21・31ピクセルの二窓、前後逆追跡誤差0.2ピクセル以下、窓間差0.2以下、変換から2以下、100点以上、8×8空間区画の半分以上。交互支持二群から帯域幅256ピクセルの残差場を作り、他群への誤差改善と地図差中央値0.2以下を要求。非周期目印は前後両像を目視し、4ピクセル平滑化後の対応距離が半周期3.643未満という周1基準を保持。目印をマスクや除外へ使わない。確認できない候補は補助とする。\n\n'
        '既存の目視記録111件・既存採用監査は変更していない。追加目視はこの再開結果フォルダの `manual_marker_locations.json` と `round5_F_marker_audit.csv`、`marker_sheets/` と `marker_crops/` に記録する。既存確認の妥当性を再認定する解析はしていない。')
    add('対象を増やしたため集計・仮説Fの検定だけを更新した。既存の視野別説明率は再計算していない。視野単位10,000回符号並べ替えで実地図と九対照中央値の差を検定。基板・日程共通符号の値は感度比較。\n\n'+md(table(OUT/'round5_F_tests.csv')))
    add('仮説Fの説明率：\n\n'+md(es[es['仮説']=='F'],percent=['中央値','第1四分位','第3四分位']))
    ft=table(OUT/'round5_F_tests.csv')
    f_all=ft[ft['対象']=='全利用可能']
    add('確認的集合の全利用可能視野で、二定義のうち補正後0.05未満となる検定数は'+str(int((f_all['ボンフェローニ補正後']<.05).sum()))+'。支持が得られる場合も「今回の局所残差モデルが空間対照を改善する」という範囲に限定する。')
    add('**判定：局所位置ずれを原因として因果帰属することは判定不能、確度：低。** 確認的集合で空間対照を上回るかは上表の補正後有意確率による。数値計算の確度は高。低い説明率だけで原因を否定しない。目印一点は視野全域の位相を保証しない。回復・支持条件・目印確認により対象が選別され、固定29全体へ一般化しない。丸め後差が一致しても丸め前座標が一意とはいえない。')
    add('事実：確認的179視野の全面説明率中央値は、平均標準偏差0.880%（四分位−0.599〜3.313%）、中央値絶対偏差2.209%（−0.281〜5.903%）。固定29内の利用可能4視野では8.123%（6.629〜9.656%）・11.217%（7.985〜15.847%）。一方、対照との差の視野平均は全179で−4.865・−5.596百分率ポイントとなり、補正後有意確率はいずれも1。対照との差の中央値が正でも、事前の平均差検定を中央値検定へ変更していない。固定4視野は補正前0.1276・補正後1。確度：高。\n\n'
        '採用の偏り：固定29では回復11、独立支持4、目印確認4（全29の13.8%）。他の603視野では回復255、独立支持214、目印確認175（29.0%）。外れ値視野を十分代表していない。')
    add('## 作業3：波長別の再現構造（記述、数値の確度：高）\n\n'
        '保存済み結果を再利用。各半分の他視野共通平均を除いた残差を平均除去し、二次元ハニング窓で高速フーリエ変換。二地図の複素スペクトル積の実部を符号付き共通成分とし、直流を除外、実数変換の多重度を補正して六区間へ分けた。負値を零へ切り詰めない。格子行偶奇を主、市松と窓なしを感度比較とする。\n\n'
        '![波長別共通成分](../figures/波長別共通成分.png)')
    bs=table(BASE/'round5_scale_summary.csv');bs=bs[(bs['分割']=='格子行偶奇')&(bs['窓']=='ハニング窓')]
    add(md(bs[['定義','対象','波長区間','視野数','中央値','第1四分位','第3四分位','共通成分合計比']],percent=['中央値','第1四分位','第3四分位','共通成分合計比']))
    ss=table(BASE/'round5_structure_summary.csv');ss=ss[(ss['分割']=='格子行偶奇')&(ss['窓']=='ハニング窓')]
    add('### 周期成分と大きな塊\n\n'
        '周1の候補分類と自己相関・スペクトル周期一致を必要とし、両半分でピークが残る軸のみ。ピーク近傍は軸±15度、逆周期±基本周波数一格子幅。整数の調波一致だけを証拠にしない。周期を含む広い区間は純粋な周期の割合ではなく上限的範囲。狭い近傍にも窓漏れ・塊が混入し得る。\n\n'
        '大塊は六方格子六近傍で連結した陽性492本以上を固定。塊関連寄与は全体の二分割共通成分から塊除去後を引き、全体で割る。塊と残部の交差寄与を含み、塊単独パワーと異なる。定義間で同じ492本を使うが中央値定義での独立較正は未実施。既知五視野の最大塊1414・203・408・497・1477本を保存結果で照合。\n\n'
        '![周期と大塊](../figures/周期と大塊の比較.png)\n\n'+md(ss[['定義','対象','指標','視野数','中央値','第1四分位','第3四分位','共通成分加重合計比','共通周期支持視野数','負値数','一超数']],percent=['中央値','第1四分位','第3四分位','共通成分加重合計比']))
    add('周期近傍・低周波・塊は互いに重なるため、足して100%にできない。視野等重みの中央値と共通成分合計で重み付けした値を区別する。32ピクセル升目で軸方向64未満は分解不能。64以下区間には斜めの約45〜64ピクセルが入り、7.286ピクセルの格子周期を測っていない。窓ありは縁を弱め、窓なしは境界漏れを増やす。128〜512の成分は周期・非周期を共に含む。全個別値・感度比較は元の `round5_scale_by_field.csv`、`round5_structure_by_field.csv` と対応集計表を参照。')
    add('元図の「分子あり」は参考ブランクを含む563視野の旧群。再開の表ではその群を「主ブランク以外（参考ブランク混入）」と明記し、正確な分子条件556視野の集計を追加した。波長変換は再計算せず、視野別保存値だけを再集計。')
    add('事実：固定29の周期の狭い近傍の共通成分割合は中央値17.93%・21.44%、共通成分合計での比20.14%・22.25%（順に平均標準偏差・中央値絶対偏差）。周期を含む広い区間は中央値38.15%・45.07%。512ピクセル超は中央値13.88%・13.06%。大塊関連寄与は中央値0.277%・47.90%に対し、合計比34.65%・51.47%。確度：高。\n\n'
        '解釈：周期の狭い成分だけが再現構造の過半を占めるとは示されない。平均標準偏差定義の大塊は一部視野へ偏り、中央値だけでは重要性を見落とす。中央値絶対偏差定義では多くの視野で大塊関連寄与が大きい。ただし区間と塊は重なり、純粋な原因分解ではない。確度：中。')
    add('## 較正と分析の弱点\n\n'
        '周4の固定29視野の二分割天井は平均標準偏差78.45%、中央値絶対偏差82.54%。正解の形を知る人工帯でも3百分率ポイント追加の説明率中央値は0.58〜2.59%、10ポイントで7.25〜19.92%。数%で仮説を否定できない。この較正は空間モデルの尺度で、群間検定の検出力と同一ではない。\n\n'
        '仮説Jの検出感度参考は、補正後の帰無臨界差から20百分位を引いた等分散加算モデル。80%検出を近似する参考であり、実験的な検出力を保証しない。上表に二定義・四指標全てを併記。\n\n'
        '基板番号は洗浄順と1対1、濃度と交絡。日程内視野単位の交換可能性は基板内依存により保証されない。少数ブランクと閾値のブランク再利用、二分割が同じ撮影を共有する雑音、共通平均、回復と目印による選別が残る。説明率は除外可能な陽性本数の割合ではない。\n\n'
        '別の有力な説明として丸めと局所残差の非線形相互作用、前後の光学条件、乾燥・残渣・傷を残す。蒸着・成形は単独像帰属の方法限界で判定不能。分子なし260830の単独像は周4記述のみで、前後差を作らない。欠測を零としない。')
    unfinished=table(OUT/'round5_unfinished_fields.csv')
    add('## 未完了・確認不能\n\n'+md(unfinished.groupby(['日程','理由']).size().reset_index(name='視野数'))+
        '\n\n不足画像・回復不一致・支持条件不合格・目印未確認を区別し、各視野の理由を保存した。回復条件不合格を未実行と混同しない。260830の前後対応、直接の分子同定、独立反復撮影による天井、基板番号・洗浄順・濃度の因果分離は確認不能。')
    add('今回追加した日程の回復試行・追跡・説明率・候補全景と拡大の確認は完了。追加日程に目印未確認の採用候補は0。半周期基準を満たさない11視野は、確認したうえで不採用。非周期目印未確認28視野は全て周1の260926の補助集合であり、周1の既存採否と視野別数値をそのまま保持した。周1集合の再目視・採否変更は今回行っていない。従って、全632視野の確認的な局所残差が得られたわけではない。\n\n'
        '小規模で回復1組・追加追跡1視野・補助群予測1視野の動作を確認してから広げた。再開中の読み取り用短文命令で引用符と文字コードの不一致が二回あり、解析値を書き出す前に失敗した。新しい実行スクリプトへ移して解消した。空の縁升目の平均に関する警告は元処理同様の有効領域で除外し、欠測を埋めていない。最終の形状・件数・回復条件・識別重複・補助ラベル除外の検証は `round5_verification.json`。')
    integrity=OUT/'resume_integrity_verification.json'
    add('## 保全確認と作業状態\n\n'+(integrity.read_text(encoding='utf-8') if integrity.exists() else '終了指紋照合は未完了。')+
        '\n\n作業ブランチ：`'+__import__('subprocess').check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()+'`。コミット・タグ・プッシュ・取得による履歴更新は実施していない。再開前にv32がこの周の途中成果だけに使われていることを版フォルダ・入力結果・ブランチ・タグで確認。書き込みはv32の追加コードと結果配下のみ。\n\n```text\n'+__import__('subprocess').check_output(['git','status','--short'],cwd=ROOT,text=True).strip()+'\n```\n\n生成物は追跡除外されるため、未追跡一覧だけで結果の有無を判断しない。')
    add('## 次に検証すべき順序と理由\n\n'
        '1. 仮説B（画素と六方格子のうなり）・F（局所位置ずれ）の非線形閾値応答と相互作用。回復できた別日程を利用し、次周に予測を先に固定する。小さい線形説明率だけで否定できないため。\n'
        '2. 仮説G（ピント・照明）・I（撮影順・条件）。撮影条件記録や反復撮影で独立な説明変数を測り、画像共有による自己説明を減らす。\n'
        '3. 仮説C（洗浄・乾燥）・H（ゴミ・シミ・傷）。大塊寄与と定義感度を踏まえ、洗浄方向と目視承認された傷位置から予測する。\n'
        '4. 仮説D（蒸着）・E（成形）。単独像の第一ピーク帰属は難しく、形状や厚みの独立測定が必要。\n'
        '5. 仮説J（実吸着）。分子条件と洗浄順を無作為配置などで分離した独立実験で検証する。仮説A（位置関係）は縁構造の対照として維持する。')
    ah=table(OUT/'round5_all_hypotheses_comparison.csv')
    add('## ここまでの全仮説の説明率（天井・人工帯つき）\n\n'
        '原因記号は前節の説明と対応。複数行は予測モデルや対象が異なる。周4値を保持し、周5のJとFのみ追加・更新。数値は視野中央値・四分位で、天井と人工帯は同一利用可能集合。未算出は零ではない。\n\n'+md(ah[['仮説','定義','対象','対象視野数','説明率中央値','第1四分位','第3四分位','同一集合半数天井中央値','同一集合3ポイント人工帯判定前中央値','同一集合3ポイント人工帯判定後中央値']],percent=['説明率中央値','第1四分位','第3四分位','同一集合半数天井中央値','同一集合3ポイント人工帯判定前中央値','同一集合3ポイント人工帯判定後中央値']))
    add('全視野個別値・対照は各保存表、判定・独立性は `round5_all_hypotheses_comparison.csv`。ここで周5を終了し、次周を開始しない。')
    (OUT/'261004_周5_報告.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    print('resume report written',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['baseline','inventory','recovery','features','evaluate','summaries','integrity_start','report','sheets','marker_audit','contacts','final_audit','integrity_final']);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    if a.stage in ['recovery','features','evaluate']:globals()[a.stage](a.limit)
    else:globals()[a.stage]()
