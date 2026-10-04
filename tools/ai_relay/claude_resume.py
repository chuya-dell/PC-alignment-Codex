#!/usr/bin/env python3
"""Claude Code 5時間制限 自動再開 watchdog（逆方向: Codex/Task Scheduler -> Claude）。

仕組み:
  1. ~/.claude/projects/*/*.jsonl（セッショントランスクリプト）の「最後の会話エントリ」を見る。
  2. それが isApiErrorMessage の assistant で、利用制限(rate_limit)なら「再開対象」。
     ログイン切れ・その他APIエラー・正常終了は再開しない。
  3. 本文の「resets 3pm (Asia/Tokyo)」等から再開時刻を取得（取れなければ発生時刻+5h+余裕）。
  4. 時刻到来後 `claude -p --resume <session_id>` を同じcwdで実行して続きを実行。
  5. 暴走防止: 1回の制限イベントにつき試行回数上限、バックオフ、ロックファイル、
     「制限後にトランスクリプトが更新された(=人が手動再開した)」場合はスキップ。

使い方:
  python claude_resume.py check   # 状態判定だけ（副作用なし）
  python claude_resume.py run     # 1回判定し、時刻が来ていれば再開（Task Schedulerから5分毎）
  python claude_resume.py status  # state/ログ末尾を表示（Codexから状態確認用）
"""
import argparse, json, os, re, subprocess, sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOME = Path(os.environ.get("CLAUDE_HOME", Path.home() / ".claude"))
BASE = Path(os.environ.get("CLAUDE_RESUME_DIR", HOME / "claude-resume"))
STATE = BASE / "state.json"
LOG = BASE / "resume.log"
LOCK = BASE / "run.lock"

MAX_ATTEMPTS = 3          # 1イベントあたりの「制限以外の失敗」上限
MAX_LIMIT_HITS = 4        # 再開直後にまた制限に当たった回数の上限
MARGIN_MIN = 2            # 再開時刻に足す余裕
FALLBACK_HOURS = 5        # 時刻が取れない場合の待ち
BACKOFF_MIN = [10, 30, 60]
RECENT_HOURS = 12         # これより古い制限イベントは無視
CAPTURE = BASE / "captured_limit_events.jsonl"   # 実際の制限エントリ原文（正規表現検証用）
# 最小権限: 無人実行は読み取り系ツールのみ許可。permission-mode は default 固定（承認が要る操作は headless では拒否される）。
# 広げたい場合のみ環境変数 CLAUDE_RESUME_ALLOWED_TOOLS="Read,Glob,Grep,Edit" 等で明示指定する。
DEFAULT_ALLOWED_TOOLS = "Read,Glob,Grep"
RESUME_PROMPT = ("[auto-resume watchdog] 利用制限が解除されたので、直前の作業を中断した所から続けてください。"
                 "無人実行のため読み取り系ツールしか使えません。許可されず実行できなかった操作は、実行せず「承認待ちの操作」として一覧で報告してください。"
                 "すでに完了した作業は繰り返さず、未完了のタスクだけ進めてください。")

LIMIT_RE = re.compile(r"(hit your (?:usage )?limit|usage limit|limit reached|5-hour limit|rate.?limit)", re.I)
RESET_RE = re.compile(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?:\s*\(([^)]+)\))?", re.I)
EPOCH_RE = re.compile(r"\|(\d{10})\b")


def log(msg):
    BASE.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"events": {}}


def save_state(st):
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def entry_text(d):
    c = (d.get("message") or {}).get("content")
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") for b in (c or []) if isinstance(b, dict) and b.get("type") == "text")


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_reset(text, event_time):
    """利用制限メッセージから再開時刻(aware datetime)を返す。取れなければ None。"""
    m = EPOCH_RE.search(text)
    if m:
        return datetime.fromtimestamp(int(m.group(1)), timezone.utc)
    m = RESET_RE.search(text)
    if not m:
        return None
    hh, mm, ap, tzname = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower(), m.group(4)
    if ap == "pm" and hh < 12:
        hh += 12
    if ap == "am" and hh == 12:
        hh = 0
    if hh > 23 or mm > 59:
        return None
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tzname.strip()) if tzname else event_time.astimezone().tzinfo
    except Exception:
        tz = event_time.astimezone().tzinfo
    local = event_time.astimezone(tz)
    cand = local.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if cand <= local:
        cand += timedelta(days=1)
    return cand


def last_conversation_entry(path):
    """末尾から遡って最後の user/assistant エントリ(dict)を返す。"""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 262144))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for l in reversed(lines):
        try:
            d = json.loads(l)
        except Exception:
            continue
        if d.get("type") in ("user", "assistant") and not d.get("isSidechain"):
            return d
    return None


def classify(d):
    """('limit'|'other_error'|'normal', info)"""
    if d.get("type") == "assistant" and d.get("isApiErrorMessage"):
        text = entry_text(d)
        if d.get("error") == "rate_limit" or LIMIT_RE.search(text):
            return "limit", text
        return "other_error", f"{d.get('error')}: {text}"
    return "normal", ""


def scan(projects_dir, now):
    """再開候補のリストを返す: dict(event_id, session_id, cwd, text, event_time, resume_at, path)"""
    out, skipped = [], []
    for p in Path(projects_dir).glob("*/*.jsonl"):
        if now.timestamp() - p.stat().st_mtime > RECENT_HOURS * 3600:
            continue
        d = last_conversation_entry(p)
        if not d:
            continue
        kind, info = classify(d)
        if kind == "other_error":
            skipped.append((p.stem, "other_error", info))
            continue
        if kind != "limit":
            continue
        et = parse_ts(d["timestamp"])
        if (now - et).total_seconds() > RECENT_HOURS * 3600:
            continue
        ra = parse_reset(info, et)
        src = "parsed"
        if ra is None:
            ra, src = et + timedelta(hours=FALLBACK_HOURS), "fallback"
        out.append(dict(event_id=f"{p.stem}:{d.get('uuid')}", session_id=d.get("sessionId") or p.stem,
                        cwd=d.get("cwd"), text=info, event_time=et.isoformat(),
                        resume_at=(ra + timedelta(minutes=MARGIN_MIN)).isoformat(),
                        resume_src=src, path=str(p), raw=d))
    return out, skipped


def acquire_lock():
    BASE.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        age = time.time() - LOCK.stat().st_mtime
        if age < 3 * 3600:
            return False
        log("stale lock removed")
    LOCK.write_text(str(os.getpid()))
    return True


def run_claude(cmd, ev, timeout):
    args = cmd + ["-p", RESUME_PROMPT, "--resume", ev["session_id"], "--output-format", "json",
                  "--permission-mode", "default",
                  "--allowedTools", os.environ.get("CLAUDE_RESUME_ALLOWED_TOOLS", DEFAULT_ALLOWED_TOOLS)]
    cwd = ev["cwd"] if ev["cwd"] and Path(ev["cwd"]).is_dir() else None
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    return r


def judge_result(r):
    """再開実行の結果を ok / limit / error に分類"""
    out = (r.stdout or "") + (r.stderr or "")
    try:
        j = json.loads(r.stdout)
        text = str(j.get("result", ""))
        if j.get("is_error") or r.returncode != 0:
            return ("limit" if LIMIT_RE.search(text + out) else "error"), text[:300]
        return "ok", text[:300]
    except Exception:
        if LIMIT_RE.search(out):
            return "limit", out[:300]
        return ("ok" if r.returncode == 0 else "error"), out[:300]


def cmd_check(a):
    now = datetime.now(timezone.utc)
    evs, skipped = scan(a.projects_dir, now)
    for s in skipped:
        print(f"SKIP(非制限エラー) session={s[0]} {s[1]}: {s[2][:80]}")
    for e in evs:
        print(f"LIMIT session={e['session_id']} resume_at={e['resume_at']} ({e['resume_src']}) cwd={e['cwd']}")
    if not evs:
        print("再開対象なし")
    return evs


def cmd_run(a):
    if not acquire_lock():
        log("another run in progress; exit")
        return 0
    try:
        now = datetime.now(timezone.utc) if not a.now else parse_ts(a.now)
        st = load_state()
        evs, skipped = scan(a.projects_dir, now)
        seen = st.setdefault("seen_other", [])
        for s in skipped:
            if s[0] + s[2] not in seen:
                seen.append(s[0] + s[2]); del seen[:-50]
                log(f"skip non-limit error session={s[0]} {s[2][:100]}")
        # 状態を区別: 未完了イベントの会話がその後進んでいれば「手動再開」と記録（watchdog再開とは別）
        live = {e["event_id"] for e in evs}
        for eid, r0 in st["events"].items():
            if not r0.get("done") and eid not in live:
                r0["done"], r0["result"] = True, "manual_resume_detected"
                log(f"MANUAL RESUME detected (not by watchdog) {eid}")
        for ev in evs:
            new = ev["event_id"] not in st["events"]
            rec = st["events"].setdefault(ev["event_id"], {"attempts": 0, "limit_hits": 0, "done": False,
                                                           "next_try": ev["resume_at"]})
            rec.update(session_id=ev["session_id"], resume_at=ev["resume_at"], resume_src=ev["resume_src"])
            if new:
                with open(CAPTURE, "a", encoding="utf-8") as f:   # 実ログ原文を保存（正規表現検証用）
                    f.write(json.dumps(ev["raw"], ensure_ascii=False) + "\n")
                if ev["resume_src"] == "fallback":
                    log(f"WARN reset time NOT parsed -> fallback event+{FALLBACK_HOURS}h used. "
                        f"session={ev['session_id']} raw_text={ev['text']!r} (原文は captured_limit_events.jsonl)")
                else:
                    log(f"limit detected session={ev['session_id']} reset parsed -> resume_at={ev['resume_at']} raw_text={ev['text']!r}")
            if rec["done"]:
                continue
            if rec["attempts"] >= MAX_ATTEMPTS or rec["limit_hits"] >= MAX_LIMIT_HITS:
                rec["done"], rec["result"] = True, "gave_up"
                log(f"GIVE UP {ev['event_id']} attempts={rec['attempts']} limit_hits={rec['limit_hits']}")
                continue
            if now < parse_ts(rec["next_try"]):
                log(f"waiting {ev['session_id']} until {rec['next_try']}")
                continue
            if a.dry_run:
                log(f"DRY-RUN would resume {ev['session_id']} cwd={ev['cwd']}")
                continue
            log(f"RESUME {ev['session_id']} cwd={ev['cwd']} (attempt {rec['attempts']+1})")
            try:
                r = run_claude(a.claude_cmd, ev, a.timeout)
                res, detail = judge_result(r)
            except Exception as e:
                res, detail = "error", repr(e)
            if res == "ok":
                rec["done"], rec["result"] = True, "resumed_by_watchdog"
                log(f"OK {ev['session_id']}: {detail[:120]!r}")
            elif res == "limit":
                rec["limit_hits"] += 1
                ra2 = parse_reset(detail, now) or (now + timedelta(hours=1))
                rec["next_try"] = max(ra2 + timedelta(minutes=MARGIN_MIN), now + timedelta(minutes=10)).isoformat()
                log(f"STILL LIMITED {ev['session_id']} next_try={rec['next_try']} ({rec['limit_hits']}/{MAX_LIMIT_HITS})")
            else:
                rec["attempts"] += 1
                bo = BACKOFF_MIN[min(rec["attempts"] - 1, len(BACKOFF_MIN) - 1)]
                rec["next_try"] = (now + timedelta(minutes=bo)).isoformat()
                log(f"FAILED {ev['session_id']} attempt={rec['attempts']}/{MAX_ATTEMPTS} retry in {bo}m: {detail[:200]!r}")
        # 古いイベントの掃除
        st["events"] = dict(list(st["events"].items())[-50:])
        st["last_run"] = now.isoformat()
        save_state(st)
    finally:
        LOCK.unlink(missing_ok=True)
    return 0


def cmd_status(a):
    st = load_state()
    print(json.dumps(st, ensure_ascii=False, indent=2))
    if LOG.exists():
        print("--- log tail ---")
        print("\n".join(LOG.read_text(encoding="utf-8").splitlines()[-15:]))


def main():
    for st_ in (sys.stdout, sys.stderr):
        try: st_.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "run", "status"])
    ap.add_argument("--projects-dir", default=str(HOME / "projects"))
    ap.add_argument("--claude-cmd", nargs="+", default=[__import__("shutil").which("claude") or "claude"], help="claude実行コマンド（テスト用に差替え可）")
    ap.add_argument("--timeout", type=int, default=3 * 3600)
    ap.add_argument("--now", help="テスト用: 現在時刻を固定(ISO8601)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.mode == "check":
        cmd_check(a)
    elif a.mode == "status":
        cmd_status(a)
    else:
        return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
