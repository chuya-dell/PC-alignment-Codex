"""Enumerate actual native-image pairs; audit conditional additional copies."""
from field_round5_common import *
import argparse

def pairs():
    verify()
    raw=next(p for p in LOCAL.iterdir() if p.name=='raw_readonly')
    children=list(raw.iterdir())
    qc=table(discover()/'tables/table_registration_field_qc.csv')
    index={f'{r.date}_{r.board}_{int(r.field)}':r for r in qc.itertuples()}
    rows=[]
    for r in fields().to_dict('records'):
        roots=[p for p in children if p.is_dir() and p.name.startswith(r['日程'])]
        root=roots[0] if len(roots)==1 else None
        native={p.name:p for p in root.iterdir() if p.is_file()} if root else {}
        q=index[r['key']]
        names=[str(getattr(q,c)).replace('\\','/').split('/')[-1] for c in ['path_pre','path_post']]
        rows.append(dict(key=r['key'],日程=r['日程'],基板=r['基板'],固定29視野=bool(r['外れ値視野3percent以上']),ブランク=bool(r['ブランク']),実在日程フォルダ=str(root) if root else '',必要前像名=names[0],必要後像名=names[1],前像存在=names[0] in native,後像存在=names[1] in native,前後組存在=all(n in native for n in names),前像実在パス=str(native.get(names[0],'')),後像実在パス=str(native.get(names[1],''))))
    a=pd.DataFrame(rows);csv(a,'round5_raw_pair_inventory.csv')
    s=a.groupby('日程').agg(保存視野数=('key','size'),前像存在=('前像存在','sum'),後像存在=('後像存在','sum'),前後組存在=('前後組存在','sum'),固定29視野数=('固定29視野','sum')).reset_index()
    csv(s,'round5_raw_pair_summary.csv')
    dump(dict(生画像直下=[p.name for p in children],第三コピー完了=any('copy_done_3' in p.name for p in children if p.is_file()),日程フォルダ全ファイル数={p.name:sum(1 for q in p.rglob('*') if q.is_file()) for p in children if p.is_dir()}),OUT/'round5_raw_final_inventory.json')
    print(s.to_json(force_ascii=True),flush=True)

def integrity_extra():
    verify()
    a=table(OUT/'round5_copy3_input_integrity.csv')
    a['終了内容指紋']=[digest(p) for p in a['実在パス']]
    a['不変']=a['開始内容指紋']==a['終了内容指紋']
    csv(a,'round5_copy3_input_integrity_final.csv');assert a['不変'].all()
    known=set(a['実在パス'])|set(table(OUT/'round5_input_integrity.csv')['実在パス'])
    new=[p for p in LOCAL.rglob('*') if p.is_file() and str(p.resolve()) not in known]
    csv([dict(実在パス=str(p.resolve()),バイト数=p.stat().st_size,終了時初確認内容指紋=digest(p),比較状態='開始時未存在または未取得、前後不変は未検証') for p in new],'round5_late_copy_inventory.csv')
    print('additional copies unchanged',len(a),'later arrivals',len(new),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pairs','integrity_extra']);a=ap.parse_args();globals()[a.stage]()
