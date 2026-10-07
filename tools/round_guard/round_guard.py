#!/usr/bin/env python3
"""周回の最後の更新(03_やること・02_進捗)を機械で確かめる。

使い方:
  python round_guard.py start   周回の開始時に印(作業用ファイル)を作り、開始時刻を書く
  python round_guard.py end     周回の終わりに印を消す
  python round_guard.py check   Claude Code の Stop フックから呼ぶ。印があるときだけ検査する

印がなければ何もしない(周回でない普通の作業を邪魔しない)。
検査に通らないときは、終了コード2と標準エラー出力で、Claude Code の終了を止める。
"""
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
MAX_BLOCKS = 5         # 同じ印で止める回数の上限(無限に終われなくなるのを防ぐ)
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


def cmd_start():
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    data = {"start_ts": now, "start": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"), "blocks": 0}
    MARKER.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"周回の印を作りました: {MARKER}（開始 {data['start']}）")
    return 0


def cmd_end():
    try:
        MARKER.unlink()
        print(f"周回の印を消しました: {MARKER}")
    except FileNotFoundError:
        print("周回の印はありません（何もしません）")
    return 0


def run_checks(start_ts):
    """足りないものの一覧を返す。空なら合格。"""
    lab = find_lab_note()
    if lab is None:
        return [f"ラボノートが見つかりません（{DRIVE_ROOT} 配下を列挙しても「Obsidian Vault/ラボノート」がない）。ドライブの同期を確かめてください"]
    missing = []
    todo = find_child(lab, "03_やること")
    if todo is None or not newest_after(todo, start_ts):
        missing.append("ラボノート/03_やること が、周回の開始時刻より後に更新されていません。終わった項目のチェック（【周回・完了(日付)】と最終報告書へのリンク）と、次にやることを書いてください（日付つきの追記票を新規作成してもよい）")
    progress = find_child(lab, "02_進捗")
    progress_ok = progress is not None and bool(newest_after(progress, start_ts))
    if not progress_ok:
        waived = False
        rec = None
        hikitsugi = find_child(lab, "10_引き継ぎ")
        rec = find_child(hikitsugi, "周回記録") if hikitsugi else None
        for p in (newest_after(rec, start_ts) if rec else []):
            try:
                if any(WAIVER_RE.search(line) for line in p.read_text(encoding="utf-8-sig", errors="replace").splitlines()):
                    waived = True
                    break
            except OSError:
                pass
        if not waived:
            missing.append("ラボノート/02_進捗/02_進捗.md が、周回の開始時刻より後に更新されていません。方針が変わったなら「現行の方針」に1行足す。変わらなかったなら、今回の周回記録（10_引き継ぎ/周回記録/）に「02_進捗：方針の変更なし」と書いてください")
    return missing


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
    missing = run_checks(start_ts)
    if not missing:
        return 0
    marker["blocks"] = int(marker.get("blocks", 0)) + 1
    try:
        MARKER.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    if marker["blocks"] > MAX_BLOCKS:
        print(f"[round_guard] {MAX_BLOCKS}回止めても更新されないため、今回は通します。未更新: " + " / ".join(missing), file=sys.stderr)
        return 0
    print("[round_guard] 周回の最後の更新が終わっていないため、終了できません"
          f"（開始 {marker.get('start')}、{marker['blocks']}/{MAX_BLOCKS}回目）。足りないもの:", file=sys.stderr)
    for m in missing:
        print(" - " + m, file=sys.stderr)
    print("更新したら、もう一度終了してください。周回でない作業なら、印（python round_guard.py end）を消してください。", file=sys.stderr)
    return 2


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"start": cmd_start, "end": cmd_end, "check": cmd_check}.get(cmd)
    if not fn:
        print(__doc__)
        return 1
    return fn()


if __name__ == "__main__":
    sys.exit(main())
