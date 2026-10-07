r"""Yoga 用の会話保存フック。元：W:\GoogleDrive\chuya2816\Claude\セッション\save_session.py（共有のため変更しない）。
保存先だけ W: 側に変えてある。ThinkPad など他の機械の設定には使わない。
"""
import json
import sys
import os
import glob
from datetime import datetime

def extract_messages(jsonl_path):
    messages = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") == "user":
                content = obj.get("message", {}).get("content", "")
                if isinstance(content, str) and content.strip():
                    messages.append(("user", content.strip()))
                elif isinstance(content, list):
                    text = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                    ).strip()
                    if text:
                        messages.append(("user", text))

            elif obj.get("type") == "assistant":
                content = obj.get("message", {}).get("content", [])
                if isinstance(content, list):
                    text = " ".join(
                        c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                    ).strip()
                    if text:
                        messages.append(("assistant", text))

    return messages


def find_jsonl(session_id):
    base = os.path.expanduser(r"~\.claude\projects")
    pattern = os.path.join(base, "**", f"{session_id}.jsonl")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def main():
    try:
        data = json.loads(sys.stdin.read())
        session_id = data.get("session_id", "")
    except Exception:
        session_id = ""

    if not session_id:
        sys.exit(0)

    jsonl_path = find_jsonl(session_id)
    if not jsonl_path:
        sys.exit(0)

    messages = extract_messages(jsonl_path)
    if not messages:
        sys.exit(0)

    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    # Yoga 用：保存先は Google ドライブの同期フォルダ。無ければ何もせず正常に終わる（親フォルダは作らない）
    output_dir = os.environ.get("SAVE_SESSION_YOGA_DIR", r"W:\GoogleDrive\chuya2816\AI会話ログ")  # 環境変数は試験用
    if not os.path.isdir(os.path.dirname(output_dir)):
        sys.exit(0)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{date_str}_{session_id[:8]}.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Claude会話ログ\n\n")
        f.write(f"- 日時: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- セッションID: {session_id}\n\n---\n\n")
        for role, text in messages:
            label = "## You" if role == "user" else "## Claude"
            f.write(f"{label}\n\n{text}\n\n---\n\n")


if __name__ == "__main__":
    main()
