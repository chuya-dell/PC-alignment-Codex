#!/usr/bin/env python3
"""Codex usage-limit watchdog: resume a Codex session that stopped on its usage limit.

Run from Task Scheduler every few minutes (independent of Claude and Codex, so it still
works when either one is stopped). Mirror of claude_resume.py for the other direction.

A session is resumable when its LAST event is a `task_complete` carrying
error.codex_error_info == "usage_limit_exceeded" (observed on real Codex 0.160.0 logs).
Anything after it (a new turn, i.e. a human resumed) cancels the resume.
Unattended resume is read-only (sandbox_mode="read-only").

  python codex_resume.py check   # list resumable sessions, no side effects
  python codex_resume.py run     # resume those whose reset time has passed
  python codex_resume.py status  # state + log tail
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HOME = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
BASE = Path(os.environ.get("CODEX_RESUME_DIR", HOME / "codex-resume"))
STATE, LOG, LOCK = BASE / "state.json", BASE / "resume.log", BASE / "run.lock"
CAPTURE = BASE / "captured_limit_events.jsonl"

MAX_ATTEMPTS, MAX_LIMIT_HITS = 3, 4
MARGIN_MIN, FALLBACK_HOURS, RECENT_HOURS = 2, 5, 12
BACKOFF_MIN = [10, 30, 60]
LIMIT_CODE = "usage_limit_exceeded"
RESET_RE = re.compile(r"try again at\s*(\d{1,2}):(\d{2})\s*([AP]M)", re.I)
RESUME_PROMPT = ("[auto-resume watchdog] The usage limit has been lifted. Continue the interrupted work "
                 "from where it stopped. This is unattended and read-only: do not repeat finished work, and "
                 "list any operation you could not perform as 'awaiting approval'.")


def log(msg):
    BASE.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line)


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def parse_reset(text, event_time):
    """'try again at 2:28 PM' (local clock) -> next occurrence after event_time, else None."""
    match = RESET_RE.search(text)
    if not match:
        return None
    hour, minute, ampm = int(match.group(1)), int(match.group(2)), match.group(3).upper()
    if not (1 <= hour <= 12 and minute <= 59):
        return None
    hour = hour % 12 + (12 if ampm == "PM" else 0)
    local = event_time.astimezone()
    cand = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return cand if cand > local else cand + timedelta(days=1)


def read_events(path):
    """(session_meta payload or None, list of event dicts) from the head and the tail of a rollout."""
    try:
        with open(path, "rb") as handle:
            head = handle.readline().decode("utf-8", "replace")
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 262144))
            tail = handle.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None, []
    try:
        first = json.loads(head)
        meta = first.get("payload") if first.get("type") == "session_meta" else None
    except ValueError:
        meta = None
    events = []
    for line in tail:
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return meta, events


def find_limit(events):
    """The trailing task_complete limit event, or None if anything real happened after it."""
    for index in range(len(events) - 1, -1, -1):
        payload = events[index].get("payload") or {}
        if events[index].get("type") != "event_msg" or payload.get("type") == "token_count":
            continue
        error = payload.get("error") or {}
        if payload.get("type") == "task_complete" and error.get("codex_error_info") == LIMIT_CODE:
            return events[index], error.get("message", "")
        return None
    return None


def scan(sessions_dir, now):
    found = []
    for path in Path(sessions_dir).rglob("rollout-*.jsonl"):
        if now.timestamp() - path.stat().st_mtime > RECENT_HOURS * 3600:
            continue
        meta, events = read_events(path)
        hit = find_limit(events)
        if not meta or not hit:
            continue
        event, text = hit
        event_time = parse_ts(event["timestamp"])
        if (now - event_time).total_seconds() > RECENT_HOURS * 3600:
            continue
        reset = parse_reset(text, event_time)
        source = "parsed"
        if reset is None:
            reset, source = event_time + timedelta(hours=FALLBACK_HOURS), "fallback"
        found.append(dict(event_id=f"{meta['id']}:{event.get('ordinal')}", session_id=meta["id"],
                          cwd=meta.get("cwd"), text=text, event_time=event_time.isoformat(),
                          resume_at=(reset + timedelta(minutes=MARGIN_MIN)).isoformat(),
                          resume_src=source, raw=event))
    return found


def load_state():
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"events": {}}


def save_state(state):
    BASE.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def acquire_lock():
    BASE.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        if time.time() - LOCK.stat().st_mtime < 3 * 3600:
            return False
        log("stale lock removed")
    LOCK.write_text(str(os.getpid()))
    return True


def run_codex(cmd, ev, timeout):
    args = cmd + ["exec", "resume", "--skip-git-repo-check", "-c", 'sandbox_mode="read-only"',
                  ev["session_id"], "-"]
    cwd = ev["cwd"] if ev["cwd"] and Path(ev["cwd"]).is_dir() else None
    return subprocess.run(args, cwd=cwd, input=RESUME_PROMPT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


def judge(done):
    out = (done.stdout or "") + (done.stderr or "")
    if LIMIT_CODE in out or "hit your usage limit" in out:
        return "limit", out[-300:]
    return ("ok" if done.returncode == 0 else "error"), out[-300:]


def cmd_check(args):
    events = scan(args.sessions_dir, datetime.now(timezone.utc))
    for ev in events:
        print(f"LIMIT session={ev['session_id']} resume_at={ev['resume_at']} ({ev['resume_src']}) cwd={ev['cwd']}")
    if not events:
        print("nothing to resume")


def cmd_run(args):
    if not acquire_lock():
        log("another run in progress; exit")
        return 0
    try:
        now = parse_ts(args.now) if args.now else datetime.now(timezone.utc)
        state = load_state()
        events = scan(args.sessions_dir, now)
        live = {ev["event_id"] for ev in events}
        for eid, rec in state["events"].items():
            if not rec.get("done") and eid not in live:
                rec["done"], rec["result"] = True, "manual_resume_detected"
                log(f"MANUAL RESUME detected {eid}")
        for ev in events:
            new = ev["event_id"] not in state["events"]
            rec = state["events"].setdefault(ev["event_id"], dict(attempts=0, limit_hits=0, done=False,
                                                                   next_try=ev["resume_at"]))
            if new:
                with open(CAPTURE, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(ev["raw"], ensure_ascii=False) + "\n")
                log(f"limit detected session={ev['session_id']} resume_at={ev['resume_at']} ({ev['resume_src']})")
            if rec["done"]:
                continue
            if rec["attempts"] >= MAX_ATTEMPTS or rec["limit_hits"] >= MAX_LIMIT_HITS:
                rec["done"], rec["result"] = True, "gave_up"
                log(f"GIVE UP {ev['event_id']}")
                continue
            if now < parse_ts(rec["next_try"]):
                continue
            if args.dry_run:
                log(f"DRY-RUN would resume {ev['session_id']} cwd={ev['cwd']}")
                continue
            log(f"RESUME {ev['session_id']} (attempt {rec['attempts'] + 1})")
            try:
                result, detail = judge(run_codex(args.codex_cmd, ev, args.timeout))
            except Exception as error:
                result, detail = "error", repr(error)
            if result == "ok":
                rec["done"], rec["result"] = True, "resumed_by_watchdog"
                log(f"OK {ev['session_id']}")
            elif result == "limit":
                rec["limit_hits"] += 1
                reset = parse_reset(detail, now) or now + timedelta(hours=1)
                rec["next_try"] = max(reset + timedelta(minutes=MARGIN_MIN), now + timedelta(minutes=10)).isoformat()
                log(f"STILL LIMITED {ev['session_id']} next_try={rec['next_try']}")
            else:
                rec["attempts"] += 1
                backoff = BACKOFF_MIN[min(rec["attempts"] - 1, len(BACKOFF_MIN) - 1)]
                rec["next_try"] = (now + timedelta(minutes=backoff)).isoformat()
                log(f"FAILED {ev['session_id']} retry in {backoff}m: {detail[:200]!r}")
        state["events"] = dict(list(state["events"].items())[-50:])
        state["last_run"] = now.isoformat()
        save_state(state)
    finally:
        LOCK.unlink(missing_ok=True)
    return 0


def cmd_status(_args):
    print(json.dumps(load_state(), ensure_ascii=False, indent=2))
    if LOG.exists():
        print("--- log tail ---")
        print("\n".join(LOG.read_text(encoding="utf-8").splitlines()[-15:]))


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "run", "status"])
    parser.add_argument("--sessions-dir", default=str(HOME / "sessions"))
    parser.add_argument("--codex-cmd", nargs="+", default=[shutil.which("codex") or "codex"])
    parser.add_argument("--timeout", type=int, default=3 * 3600)
    parser.add_argument("--now", help="test: fix the current time (ISO8601)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return {"check": cmd_check, "status": cmd_status}.get(args.mode, cmd_run)(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
