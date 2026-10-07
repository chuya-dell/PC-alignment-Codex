"""Summarize saved diagnostics and write Japanese audit notes."""
from pathlib import Path
import csv,json,sys,hashlib,datetime,os,shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/results/v65_p50_luminance_pre_stage2'
VAULT=Path('W:/GoogleDrive/chuya2816/Obsidian Vault/ラボノート')
THEME=VAULT/'05_解析/261008_【未読】50倍視野輝度解析_追加B再開'
PREFIX='---\ndate: 2026-10-08\n確認: 未確認\n状態: 未読\n---\n\n【未確認】本人未読。候補マスクによる方法診断。Claude Code検収待ち。\n\n'

def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(rows,fields):
    return '| '+' | '.join(fields)+' |\n| '+' | '.join(['---']*len(fields))+' |\n'+''.join('| '+' | '.join(str(r[f]) for f in fields)+' |\n' for r in rows)

def provenance():
    registry=ROOT/'data/raw/artifact_provenance_registry.csv'
    existing=registry.read_text(encoding='utf-8-sig')
    header=next(csv.reader([existing.splitlines()[0]]))
    rows=[]
    for ident,path,dest,kind in [
       ('v65-v20-code-261008',ROOT/'field_level/v47_p50_field_luminance_stage1_followup/field_v20_source_snapshot.py','field_level/v65_p50_luminance_pre_stage2/field_v20_snapshot.py','script'),
       ('v65-v20-params-261008',ROOT/'data/raw/v47_p50_v20_parameter_snapshot.json','data/raw/v65_p50_v20_parameter_snapshot.json','parameters')]:
        row=dict(artifact_id=ident,source_thread_id='v47_source_snapshot',source_title='v20 reconstruction reused for Additional B',
          source_updated_at_jst='unknown; acquired 2026-10-08',source_path=str(path),analysis_level='field_level',
          artifact_kind=kind,newness_status='unknown',supersedes_artifact_id='',import_state='imported_snapshot',
          repository_destination=dest,summary='Independent snapshot; byte identical to v47 snapshot; SHA256 '+hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
    with (OUT/'provenance_rows.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=header);w.writeheader();w.writerows(rows)
    with registry.open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=header)
        for r in rows:
            if r['artifact_id'] not in existing: w.writerow(r)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);THEME.mkdir(parents=True,exist_ok=True)
    provenance()
    (THEME/'【未読】261008_実行条件と検証計画.md').write_text(PREFIX+'''目的：50倍の視野全体でも、マーカー・シミ・ゴミをマスクした有効画素だけから、前後（pre/post）の輝度変化を出せるようにする。
この作業が目的に効く点：旧9視野とブランク8視野でマスクの輝度影響を測り、追加Bの項目2〜5と段階2の可否を記録する。

問いの読み方（依頼文の原文）：
「マスクが、本来の目的（視野全体の輝度変化）の値を、ブランクの視野間のばらつきより十分小さい範囲でしか動かさないか。」
「マスクあり・なしの差の大きさを、基板8の8視野の間のばらつき(ブランクの視野間の変動)と比べる。」

画素寸法0.1262は推定、0.12694は感度条件（+0.5864%）。中央値・両側10%トリム平均の前後代表値から差・比を求める。画素ごとの差・比の代表値とは異なる定義。位置合わせ後の同じ有効領域を両条件で使用。十分小さいの数値閾値は指定されていないので、標準偏差・1.4826×中央絶対偏差で規格化し記述する。

調整6視野：1-5、1-6、3-2、3-3、5-5、5-6。旧確認3視野：4-1、4-2、5-3。旧確認3視野は選別に使われており独立検証ではない。ブランク8視野は凍結パラメータで追加評価し、今後の調整には戻さない。旧コードが全72組を候補選別に走査しているため、同日の残り55視野も完全に未閲覧とは扱わない。今後の変更は、別日・別基板の未使用視野を事前に割り当て、パラメータ凍結後に検証する。

人工斑点：4-1・4-2・5-3、半径0.4/0.8/1.6/3.2、背景画素値に対する倍率0.5/0.8/0.9/1.1/1.2/1.5、各5個。円内の値を倍率で変更し前後の後像に埋め込む。位置合わせ・マーカー保護を固定し、シミと前後差候補を再検出する診断であり、位置合わせを含む全面再検証ではない。検出は新規マスクが円内に1画素以上あること。被覆率と併記する。誤検出は埋込前の非マスク画素で、埋込後にマスクされ、真円を3画素膨張した範囲外かつ有効領域内にある画素数／画像全画素数。旧0.0354%は余裕なし・真範囲外非保護画素数が分母、正規化画像上のガウス斑点なので同じ定義ではない。

画像端：50画素の帯と内側。反射256画素／端値256画素の余白追加を比較。余白追加は輪郭フィルタだけでなくプロファイルの分割・閾値・線追跡にも作用するため、差を境界の副作用だけには帰属できない。

旧段階1の実物未発見はClaude Codeの探索報告を引き継ぐ。v47の再現オーバーレイは新しい再構成であり旧実物との一致を意味しない。「導入を完了できなかった」の旧説明は原因の根拠がなかったとの既存記録を維持する。
''',encoding='utf-8')
    now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log=VAULT/'10_引き継ぎ/周回記録'/f'{now}_Codex周回記録.md'
    text='''# 追加B再開・v65
開始区切り：規則・依頼文・Claude Code最終報告・v47を読了。v64までのフォルダとGitタグを列挙しv65を確保。旧9＋ブランク8の実在を列挙確認。Fなし、C/H/W/X/Yあり。入力は実在するW上の260922-p50-dna。
元クローンpullはFETCH_HEAD Permission denied。data/results/v65_p50_luminance_pre_stage2/checkoutに独立クローンを作成。pushはユーザー追記に従いClaude Code担当。
最後に終えた区切り：実行条件と検証計画、出所行を保存。
再開地点：pairs/edgesの診断継続、その後spots、入力指紋照合、最終報告、commit/tag。
'''
    log.write_text(text,encoding='utf-8');assert log.read_text(encoding='utf-8')==text
    (OUT/'round_log_path.txt').write_text(str(log),encoding='utf-8')
    print(str(log))

def sync_checkout():
    checkout=OUT/'checkout'
    dest=checkout/'field_level/v65_p50_luminance_pre_stage2'
    dest.mkdir(exist_ok=True,parents=True)
    for file in (ROOT/'field_level/v65_p50_luminance_pre_stage2').iterdir():
        if file.is_file():shutil.copy2(file,dest/file.name)
    shutil.copy2(ROOT/'data/raw/v65_p50_v20_parameter_snapshot.json',checkout/'data/raw/v65_p50_v20_parameter_snapshot.json')
    registry=checkout/'data/raw/artifact_provenance_registry.csv'
    text=registry.read_text(encoding='utf-8-sig')
    with (OUT/'provenance_rows.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    with registry.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        for row in rows:
            if row['artifact_id'] not in text:writer.writerow(row)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='sync':sync_checkout()
    else:prepare()
