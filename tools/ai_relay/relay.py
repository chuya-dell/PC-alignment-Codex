"""Claude <-> Codex relay: hand over to the other agent when one hits a usage limit,
and let the two review each other's answers. Read-only unless --write is given."""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIMIT_RE = re.compile(r"usage limit|rate[_ ]limit|limit reached|hit your .*limit", re.I)
RESET_RE = re.compile(r"(?:try again at|resets?(?: at)?)\s*(\d{1,2}:\d{2}\s*[AP]M)", re.I)
OTHER = {"claude": "codex", "codex": "claude", "gemini": "claude"}
ORDER = ["claude", "codex", "gemini"]


class AgentError(RuntimeError):
    """The agent could not answer (missing, crashed, limited); hand over to another."""


class Limited(AgentError):
    def __init__(self, agent, reset):
        super().__init__(f"{agent} hit its usage limit (reset: {reset})")
        self.agent, self.reset = agent, reset


def parse_reset(text, now=None):
    """Next occurrence of the 'H:MM AM/PM' time quoted in a limit message, else None."""
    match = RESET_RE.search(text)
    if not match:
        return None
    now = now or datetime.now()
    at = datetime.strptime(match.group(1).upper().replace(" ", ""), "%I:%M%p")
    reset = now.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    return reset if reset > now else reset + timedelta(days=1)


def check_limit(agent, text):
    if LIMIT_RE.search(text):
        reset = parse_reset(text)
        raise Limited(agent, reset.strftime("%Y-%m-%d %H:%M") if reset else "unknown")


def _run(cmd, prompt):
    exe = shutil.which(cmd[0])
    if not exe:
        raise AgentError(f"{cmd[0]} not found on PATH")
    done = subprocess.run([exe] + cmd[1:], input=prompt or "", capture_output=True,
                          text=True, encoding="utf-8", cwd=ROOT)
    return done.returncode, (done.stdout or "") + (done.stderr or ""), done.stdout or ""


def call_codex(prompt, write):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "last.txt"
        sandbox = "workspace-write" if write else "read-only"
        code, both, _ = _run(["codex", "exec", "-C", str(ROOT), "-s", sandbox,
                              "-o", str(out), "-"], prompt)
        answer = out.read_text(encoding="utf-8").strip() if out.exists() else ""
        if code != 0 or not answer:
            check_limit("codex", both)
            raise AgentError(f"codex failed (exit {code}): {both[-500:]}")
        return answer


def call_claude(prompt, write):
    tools = "Read,Glob,Grep" + (",Edit,Write" if write else "")
    code, both, stdout = _run(["claude", "-p", "--permission-mode", "default",
                               "--allowedTools", tools], prompt)
    if code != 0 or not stdout.strip():
        check_limit("claude", both)
        raise AgentError(f"claude failed (exit {code}): {both[-500:]}")
    if len(stdout) < 300:
        check_limit("claude", stdout)
    return stdout.strip()


def call_gemini(prompt, write):
    if write:
        raise AgentError("gemini (agy) is read-only here; --write is not supported")
    code, both, stdout = _run(["agy", "-p", prompt], None)
    if code != 0 or not stdout.strip():
        check_limit("gemini", both)
        raise AgentError(f"gemini failed (exit {code}): {both[-500:]}")
    if len(stdout) < 300:
        check_limit("gemini", stdout)
    return stdout.strip()


CALL = {"claude": call_claude, "codex": call_codex, "gemini": call_gemini}


def ask(first, prompt, write, notes):
    """Ask `first`; if it is limited or fails, hand over to the next agent. Returns (agent, answer)."""
    failed = []
    for agent in dict.fromkeys([first, OTHER[first], *ORDER]):
        try:
            answer = CALL[agent](prompt, write)
            if failed:
                notes.append(f"{', '.join(failed)} unavailable; {agent} answered instead")
            return agent, answer
        except AgentError as error:
            failed.append(agent)
            notes.append(str(error))
    raise SystemExit("no agent could answer: " + "; ".join(notes))


def log(board, title, body):
    board.parent.mkdir(parents=True, exist_ok=True)
    with board.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {datetime.now():%H:%M} {title}\n\n{body}\n")


def discuss(first, question, rounds, write, board):
    notes = []
    who, answer = ask(first, question, write, notes)
    log(board, f"{who}: answer", answer)
    for number in range(1, rounds + 1):
        other = OTHER[who]
        review_prompt = (
            f"Question:\n{question}\n\n{who}'s answer:\n{answer}\n\n"
            "Review it. For each point say AGREE, DISAGREE (with reason) or UNVERIFIED. "
            "If there is no problem, say so plainly.")
        reviewer, review = ask(other, review_prompt, False, notes)
        if reviewer == who:
            notes.append("reviewer was limited: this review is NOT independent")
        log(board, f"{reviewer}: review {number}", review)
        if "DISAGREE" not in review.upper():
            break
        reply_prompt = (f"Question:\n{question}\n\nYour answer:\n{answer}\n\n"
                        f"Review:\n{review}\n\nRespond: revise or defend, point by point.")
        who, answer = ask(who, reply_prompt, False, notes)
        log(board, f"{who}: reply {number}", answer)
    if notes:
        log(board, "notes", "\n".join(f"- {note}" for note in notes))
    print(f"board: {board}")
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["ask", "discuss"])
    parser.add_argument("prompt")
    parser.add_argument("--first", choices=ORDER, default="codex")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--write", action="store_true", help="allow file edits (default: read-only)")
    parser.add_argument("--board", type=Path,
                        default=ROOT / f"data/results/ai_relay/board_{datetime.now():%Y%m%d}.md")
    args = parser.parse_args()
    if args.mode == "ask":
        notes = []
        agent, answer = ask(args.first, args.prompt, args.write, notes)
        log(args.board, f"{agent}: ask", f"{args.prompt}\n\n{answer}")
        print(f"[{agent}] {answer}")
        for note in notes:
            print("NOTE:", note, file=sys.stderr)
    else:
        discuss(args.first, args.prompt, min(args.rounds, 3), args.write, args.board)


if __name__ == "__main__":
    main()
