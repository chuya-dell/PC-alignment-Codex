#!/usr/bin/env python3
"""周回の最後の更新(03_やること・02_進捗)を機械で確かめる。

使い方:
  python round_guard.py start [--theme テーマ名]... [--record 周回記録のパス]
                                 周回の開始時に印(作業用ファイル)を作り、開始時刻・テーマ名・周回記録を書く
  python round_guard.py end     周回の終わりに印を消す
  python round_guard.py check   Claude Code の Stop フックから呼ぶ。印があるときだけ検査する

印がなければ何もしない(周回でない普通の作業を邪魔しない)。
検査に通らないときは、標準出力のJSON(decision=block)で、Claude Code の終了を止める（理由は標準エラーにも出す）。
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

DRIVE_ROOT = Path(os.environ.get("ROUND_GUARD_DRIVE_ROOT", r"W:\GoogleDrive\chuya2816"))
MARKER = Path(os.environ.get("ROUND_GUARD_MARKER", str(Path.home() / ".claude" / "round-guard" / "active-round.json")))
MAX_AGE_HOURS = 8      # 最終の安全上限(6時間)より長い印は、残骸として無視する
MAX_BLOCKS = 5         # 同じ印で止める回数の上限(無限に終われなくなるのを防ぐ)。超えたら、周回記録に書いてから通す
WAIVER_RE = re.compile(r"02_進捗.*(変更なし|変わらなかった|更新しない)")


def nfc(name):
    return unicodedata.normalize("NFC", name)


def find_child(parent, wanted):
    """親フォルダを列挙して、実在する名前だけを返す(綴りの決め打ちをしない)。"""
    try:
        for entry in os.listdir(parent):
            if nfc(entry) == nfc(wanted):
                return Path(parent) / entry
    except OSError:
        pass
    return None


def find_lab_note():
    """DRIVE_ROOT 配下を2階層まで列挙して、「Obsidian Vault」→「ラボノート」を探す。"""
    candidates = [DRIVE_ROOT]
    try:
        candidates += [DRIVE_ROOT / e for e in os.listdir(DRIVE_ROOT) if (DRIVE_ROOT / e).is_dir()]
    except OSError:
        return None
    for base in candidates:
        vault = find_child(base, "Obsidian Vault")
        if vault:
            lab = find_child(vault, "ラボノート")
            if lab:
                return lab
    return None


def newest_after(folder, start_ts, suffix=".md"):
    """folder 配下(再帰)で、start_ts より後に更新されたファイルを新しい順に返す。"""
    hits = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if not f.endswith(suffix):
                continue
            p = Path(root) / f
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m > start_ts:
                hits.append((m, p))
    return [p for _m, p in sorted(hits, reverse=True)]


def read_marker():
    try:
        return json.loads(MARKER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cmd_start(argv):
    ap = argparse.ArgumentParser(prog="round_guard.py start")
    ap.add_argument("--theme", action="append", default=[], help="この周回のテーマ名（複数可）。更新したファイルにリンクかテーマ名が要る")
    ap.add_argument("--record", default="", help="この周回の記録ファイルのパス（通すときの書き込み先）")
    args = ap.parse_args(argv)
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    data = {"start_ts": now, "start": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
            "blocks": 0, "themes": args.theme, "record": args.record}
    MARKER.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"周回の印を作りました: {MARKER}（開始 {data['start']}、テーマ {args.theme or '未指定'}）")
    return 0


def cmd_end():
    try:
        MARKER.unlink()
        print(f"周回の印を消しました: {MARKER}")
    except FileNotFoundError:
        print("周回の印はありません（何もしません）")
    return 0


def report_names(lab, start_ts):
    """開始後に更新された最終報告書（05_解析/…/00_*最終報告書*.md）のファイル名（拡張子なし）。"""
    names = []
    analysis = find_child(lab, "05_解析")
    for p in (newest_after(analysis, start_ts) if analysis else []):
        if p.name.startswith("00_") and "最終報告書" in p.name:
            names.append(p.stem)
    return names


def mentions(paths, keys):
    """paths のどれかの本文に keys のどれかが含まれるか。"""
    for p in paths:
        try:
            text = nfc(p.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError:
            continue
        if any(nfc(k) in text for k in keys):
            return True
    return False


def find_record_dir(lab):
    h = find_child(lab, "10_引き継ぎ")
    return find_child(h, "周回記録") if h else None


def run_checks(start_ts, marker):
    """足りないものの一覧と、ラボノートのパスを返す。足りないものが空なら合格。"""
    lab = find_lab_note()
    if lab is None:
        return [f"ラボノートが見つかりません（{DRIVE_ROOT} 配下を列挙しても「Obsidian Vault/ラボノート」がない）。ドライブの同期を確かめてください"], None
    keys = list(marker.get("themes") or []) + report_names(lab, start_ts)
    key_hint = "（含めるもの：この周回の最終報告書のファイル名か、テーマ名 " + " / ".join(keys) + "）" if keys else ""
    missing = []
    todo = find_child(lab, "03_やること")
    todo_new = newest_after(todo, start_ts) if todo else []
    if not todo_new:
        missing.append(f"{lab}/03_やること が、周回の開始時刻より後に更新されていません。終わった項目のチェック（【周回・完了(日付)】と最終報告書へのリンク）と、次にやることを書いてください（日付つきの追記票を新規作成してもよい）" + key_hint)
    elif keys and not mentions(todo_new, keys):
        missing.append(f"03_やること の更新されたファイルに、この周回の最終報告書のファイル名もテーマ名も入っていません。結果へのリンクを書いてください" + key_hint)
    progress = find_child(lab, "02_進捗")
    progress_new = newest_after(progress, start_ts) if progress else []
    rec_dir = find_record_dir(lab)
    waived = False
    for p in (newest_after(rec_dir, start_ts) if rec_dir else []):
        try:
            if any(WAIVER_RE.search(line) for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines()):
                waived = True
                break
        except OSError:
            pass
    if not progress_new and not waived:
        missing.append(f"{lab}/02_進捗/02_進捗.md が、周回の開始時刻より後に更新されていません。方針が変わったなら「現行の方針」に1行足す。変わらなかったなら、今回の周回記録（{rec_dir or '10_引き継ぎ/周回記録/'}）に「02_進捗は変更なし」と書いてください")
    elif progress_new and not waived and keys and not mentions(progress_new, keys):
        missing.append("02_進捗 の更新されたファイルに、この周回の最終報告書のファイル名もテーマ名も入っていません。方針の1行にリンクを添えるか、周回記録に「02_進捗は変更なし」と書いてください" + key_hint)
    return missing, lab


def write_pass_through_record(marker, lab, missing):
    """5回止めても直らず通すときに、周回記録へ書く。書き込み先のパスを返す。"""
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = ["", "## 周回の終わりの検査（round_guard）：未更新のまま終了",
             f"- 時刻: {stamp}",
             f"- 03_やること・02_進捗が未更新のまま終了した（{MAX_BLOCKS}回止めたが直らず、通した）"]
    lines += [f"- 足りなかった項目: {m}" for m in missing]
    lines.append("- 3行報告の(3)本人待ち・確認待ちに、同じことを書くこと")
    text = "\n".join(lines) + "\n"
    target = Path(marker["record"]) if marker.get("record") else None
    if target is None or not target.parent.exists():
        rec_dir = find_record_dir(lab) if lab else None
        cands = newest_after(rec_dir, float(marker.get("start_ts", 0))) if rec_dir else []
        if cands:
            target = cands[0]
        elif rec_dir:
            target = rec_dir / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_round_guard記録.md")
        else:
            return None
    with open(target, "a", encoding="utf-8") as f:
        f.write(text)
    return target


def cmd_check():
    try:
        sys.stdin.read()  # フックの入力(JSON)。使わないが読み捨てる
    except Exception:
        pass
    marker = read_marker()
    if not marker:
        return 0  # 印がない＝周回でない。邪魔しない
    start_ts = float(marker.get("start_ts", 0))
    if time.time() - start_ts > MAX_AGE_HOURS * 3600:
        return 0  # 古い印は残骸
    missing, lab = run_checks(start_ts, marker)
    if not missing:
        return 0
    marker["blocks"] = int(marker.get("blocks", 0)) + 1
    try:
        MARKER.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    if marker["blocks"] > MAX_BLOCKS:
        target = None
        try:
            target = write_pass_through_record(marker, lab, missing)
        except OSError as e:
            print(f"[round_guard] 周回記録に書けませんでした: {e}", file=sys.stderr)
        print(f"[round_guard] {MAX_BLOCKS}回止めても更新されないため通します。周回記録（{target or '書けず'}）に「未更新のまま終了」と書きました。"
              "3行報告の(3)にも同じことを書いてください。未更新: " + " / ".join(missing), file=sys.stderr)
        return 0
    lines = ["[round_guard] 周回の最後の更新が終わっていないため、終了できません"
             f"（開始 {marker.get('start')}、{marker['blocks']}/{MAX_BLOCKS}回目）。足りないもの:"]
    lines += [" - " + m for m in missing]
    if marker["blocks"] == MAX_BLOCKS:
        lines.append("※次に終了しようとすると、未更新のまま通し、周回記録に「未更新のまま終了」と書きます。"
                     "その場合は、3行報告の(3)に同じことを書いてください。")
    lines.append("足りないものを更新したら、もう一度終了してください。周回でない作業なら、印（python round_guard.py end）を消してください。")
    reason = "\n".join(lines)
    # 終了コードはシェル(PowerShell)経由で2にならず1になる（実測）。標準出力のJSONで止める（日本語は\u形式で出す）
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=True))
    print(reason, file=sys.stderr)
    return 0


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "start":
        return cmd_start(sys.argv[2:])
    fn = {"end": cmd_end, "check": cmd_check}.get(cmd)
    if not fn:
        print(__doc__)
        return 1
    return fn()


if __name__ == "__main__":
    sys.exit(main())
