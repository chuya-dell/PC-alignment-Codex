import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import codex_resume as cr

MSG = ("You’ve hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro) "
       "or try again at 2:28 PM.")
META = {"timestamp": "2026-10-04T04:09:07.824Z", "ordinal": 0, "type": "session_meta",
        "payload": {"id": "sid-1", "cwd": "."}}
LIMIT = {"timestamp": "2026-10-04T04:09:10.734Z", "ordinal": 11, "type": "event_msg",
         "payload": {"type": "task_complete", "last_agent_message": None,
                     "error": {"message": MSG, "codex_error_info": cr.LIMIT_CODE}}}
TOKENS = {"timestamp": "2026-10-04T04:09:11Z", "ordinal": 12, "type": "event_msg",
          "payload": {"type": "token_count"}}
NEXT_TURN = {"timestamp": "2026-10-04T05:30:00Z", "ordinal": 13, "type": "event_msg",
             "payload": {"type": "task_started"}}
NOW = datetime(2026, 10, 4, 5, 0, tzinfo=timezone.utc)


def write_session(root, events):
    path = Path(root) / "2026" / "10" / "04" / "rollout-x.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(e) for e in [META, *events]), encoding="utf-8")
    return path


class CodexResumeTest(unittest.TestCase):
    def test_parse_reset_uses_next_occurrence(self):
        event_time = cr.parse_ts("2026-10-04T04:09:10.734Z")
        reset = cr.parse_reset(MSG, event_time)
        self.assertEqual((reset.hour, reset.minute), (14, 28))
        self.assertGreater(reset, event_time.astimezone())
        self.assertIsNone(cr.parse_reset("no time", event_time))
        self.assertEqual(cr.parse_reset("try again at 12:05 AM", event_time).hour, 0)

    def test_trailing_limit_is_found_even_after_token_count(self):
        self.assertIsNotNone(cr.find_limit([LIMIT, TOKENS]))

    def test_new_turn_after_limit_cancels(self):
        self.assertIsNone(cr.find_limit([LIMIT, TOKENS, NEXT_TURN]))

    def test_normal_completion_is_not_a_limit(self):
        ok = {"type": "event_msg", "payload": {"type": "task_complete", "error": None}}
        self.assertIsNone(cr.find_limit([ok]))

    def test_scan_finds_session(self):
        with tempfile.TemporaryDirectory() as root:
            path = write_session(root, [LIMIT, TOKENS])
            now = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
            found = cr.scan(root, now.replace(tzinfo=timezone.utc))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["session_id"], "sid-1")

    def test_judge(self):
        limited = mock.Mock(returncode=1, stdout="", stderr=MSG + cr.LIMIT_CODE)
        self.assertEqual(cr.judge(limited)[0], "limit")
        self.assertEqual(cr.judge(mock.Mock(returncode=0, stdout="done", stderr=""))[0], "ok")
        self.assertEqual(cr.judge(mock.Mock(returncode=2, stdout="", stderr="boom"))[0], "error")


if __name__ == "__main__":
    unittest.main()
