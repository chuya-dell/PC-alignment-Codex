import json, os, subprocess, sys, shutil, tempfile
from pathlib import Path
HERE = Path(__file__).parent
SCRIPT = HERE.parent / "claude_resume.py"
T = Path(tempfile.mkdtemp(prefix="cr_test_"))
PROJ = T / "projects" / "proj"; PROJ.mkdir(parents=True)
env = dict(os.environ, CLAUDE_RESUME_DIR=str(T / "state"), STUB_LOG=str(T / "stub.log"), PYTHONUTF8="1")

def entry(kind, sid, ts, text="hi", error=None):
    d = {"type": kind, "uuid": f"u-{ts}", "sessionId": sid, "cwd": str(T), "timestamp": ts,
         "message": {"role": kind, "content": [{"type": "text", "text": text}]}}
    if error: d["isApiErrorMessage"], d["error"] = True, error
    return json.dumps(d, ensure_ascii=False)

def write(sid, entries):
    (PROJ / f"{sid}.jsonl").write_text("\n".join(entries) + "\n", encoding="utf-8")

def run(mode, now, stub="ok", extra=()):
    e = dict(env, STUB_MODE=stub)
    r = subprocess.run([sys.executable, str(SCRIPT), mode, "--projects-dir", str(T / "projects"), "--now", now,
        "--claude-cmd", sys.executable, str(HERE / "stub_claude.py"), *extra], env=e, capture_output=True, text=True, encoding="utf-8")
    return r.stdout + r.stderr

def stub_calls():
    p = T / "stub.log"; return p.read_text(encoding="utf-8").splitlines() if p.exists() else []

def reset():
    shutil.rmtree(T / "state", ignore_errors=True); (T / "stub.log").unlink(missing_ok=True)
    for f in PROJ.glob("*.jsonl"): f.unlink()

fails = []
def check(name, cond, out=""):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: fails.append(name); print(out)

LIM = "You've hit your limit · resets 3pm (Asia/Tokyo)"   # 06:00Z = 15:00JST
# 1 制限: 再開時刻前は待機、実行しない
reset(); write("s1", [entry("user","s1","2026-10-04T03:00:00Z"), entry("assistant","s1","2026-10-04T05:50:00Z",LIM,"rate_limit")])
o = run("run","2026-10-04T05:55:00Z"); check("1 limit: 時刻前は待機", "waiting" in o and not stub_calls(), o)
# 2 時刻到来 -> --resume で再開
o = run("run","2026-10-04T06:03:00Z"); c = stub_calls()
check("2 limit: 時刻後に --resume s1 で再開", len(c)==1 and '"--resume", "s1"' in c[0] and "OK s1" in o, o+str(c))
# 3 同イベントを二重再開しない
o = run("run","2026-10-04T06:30:00Z"); check("3 再開済みイベントは再実行しない", len(stub_calls())==1, o)
# 4 ログイン切れは対象外
reset(); write("s2", [entry("user","s2","2026-10-04T03:00:00Z"), entry("assistant","s2","2026-10-04T05:50:00Z","Login expired · Please run /login","authentication_failed")])
o = run("run","2026-10-04T09:00:00Z"); check("4 認証エラーは再開しない", not stub_calls() and "skip non-limit" in o, o)
# 5 正常終了は対象外
reset(); write("s3", [entry("user","s3","2026-10-04T03:00:00Z"), entry("assistant","s3","2026-10-04T05:50:00Z","完了しました")])
o = run("run","2026-10-04T09:00:00Z"); check("5 正常終了は再開しない", not stub_calls(), o)
# 6 制限後に人が手動再開（以降の発話あり）→ 対象外
reset(); write("s4", [entry("assistant","s4","2026-10-04T05:50:00Z",LIM,"rate_limit"), entry("user","s4","2026-10-04T06:10:00Z","続けて"), entry("assistant","s4","2026-10-04T06:11:00Z","了解")])
o = run("run","2026-10-04T06:30:00Z"); check("6 手動再開済みはスキップ", not stub_calls(), o)
# 7 非制限エラーが続く -> 3回で諦める（無限再起動防止）
reset(); write("s5", [entry("assistant","s5","2026-10-04T05:50:00Z",LIM,"rate_limit")])
t = ["2026-10-04T06:03:00Z","2026-10-04T06:20:00Z","2026-10-04T07:00:00Z","2026-10-04T09:00:00Z","2026-10-04T12:00:00Z"]
for x in t: o = run("run", x, "error")
check("7 失敗は3回で打切り(以降は呼ばない)", len(stub_calls())==3 and "GIVE UP" in open(T/"state"/"resume.log",encoding="utf-8").read(), str(stub_calls()))
# 8 再開直後にまた制限 -> 新しい時刻まで待ち、上限で打切り
reset(); write("s6", [entry("assistant","s6","2026-10-04T05:50:00Z",LIM,"rate_limit")])
run("run","2026-10-04T06:03:00Z","limit"); n1=len(stub_calls())
o = run("run","2026-10-04T06:30:00Z","limit"); n2=len(stub_calls())  # 次は 23:00JST=14:00Z 以降
check("8 再制限後は新リセット時刻まで待機", n1==1 and n2==1 and "STILL LIMITED" in open(T/"state"/"resume.log",encoding="utf-8").read(), o)
# 9 時刻が取れない文言 -> 発生+5h にフォールバック
reset(); write("s7", [entry("assistant","s7","2026-10-04T00:00:00Z","Claude usage limit reached.","rate_limit")])
o = run("run","2026-10-04T04:00:00Z"); a=len(stub_calls()); o2 = run("run","2026-10-04T05:03:00Z")
check("9 時刻不明は+5hフォールバック", a==0 and len(stub_calls())==1, o+o2)
# 10 ロック中は走らない
reset(); write("s8", [entry("assistant","s8","2026-10-04T05:50:00Z",LIM,"rate_limit")])
(T/"state").mkdir(); (T/"state"/"run.lock").write_text("1")
o = run("run","2026-10-04T06:30:00Z"); check("10 ロック中は多重起動しない", not stub_calls() and "in progress" in o, o)
# 11 権限: 最小権限(default + 読み取りのみ)で呼ぶ
reset(); write("s9", [entry("assistant","s9","2026-10-04T05:50:00Z",LIM,"rate_limit")])
run("run","2026-10-04T06:03:00Z"); c = stub_calls()[0]
check("11 permission-mode default + 読取系のみ、acceptEditsなし", '"--permission-mode", "default"' in c and '"Read,Glob,Grep"' in c and "acceptEdits" not in c and "auto-resume watchdog" in c, c)
# 12 フォールバック時は明確にWARNログ + 原文キャプチャ
reset(); write("s10", [entry("assistant","s10","2026-10-04T00:00:00Z","Claude usage limit reached.","rate_limit")])
run("run","2026-10-04T01:00:00Z"); lg=open(T/"state"/"resume.log",encoding="utf-8").read()
check("12 fallback WARNログと原文保存", "WARN reset time NOT parsed" in lg and (T/"state"/"captured_limit_events.jsonl").exists(), lg)
# 13 手動再開は watchdog再開と区別して記録
reset(); write("s11", [entry("assistant","s11","2026-10-04T05:50:00Z",LIM,"rate_limit")])
run("run","2026-10-04T05:55:00Z")
write("s11", [entry("assistant","s11","2026-10-04T05:50:00Z",LIM,"rate_limit"), entry("user","s11","2026-10-04T05:58:00Z","手動で続行")])
run("run","2026-10-04T06:10:00Z"); stt=json.load(open(T/"state"/"state.json",encoding="utf-8"))
check("13 手動再開=manual_resume_detected, watchdog未実行", [r["result"] for r in stt["events"].values()]==["manual_resume_detected"] and not stub_calls(), str(stt))
# 14 watchdog再開は resumed_by_watchdog
reset(); write("s12", [entry("assistant","s12","2026-10-04T05:50:00Z",LIM,"rate_limit")])
run("run","2026-10-04T06:03:00Z"); stt=json.load(open(T/"state"/"state.json",encoding="utf-8"))
check("14 watchdog再開=resumed_by_watchdog", [r["result"] for r in stt["events"].values()]==["resumed_by_watchdog"], str(stt))
print("\nFAILED:", fails if fails else "none"); shutil.rmtree(T, ignore_errors=True); sys.exit(1 if fails else 0)
