import unittest
from datetime import datetime
from unittest import mock

import relay

NOW = datetime(2026, 10, 4, 14, 21)
MSG = "ERROR: You’ve hit your usage limit. Upgrade to Pro (...) or try again at 2:28 PM."


class RelayTest(unittest.TestCase):
    def test_parse_reset_today_and_tomorrow(self):
        self.assertEqual(relay.parse_reset(MSG, NOW), datetime(2026, 10, 4, 14, 28))
        self.assertEqual(relay.parse_reset(MSG, datetime(2026, 10, 4, 15, 0)),
                         datetime(2026, 10, 5, 14, 28))
        self.assertIsNone(relay.parse_reset("no time here", NOW))

    def test_check_limit(self):
        with self.assertRaises(relay.Limited):
            relay.check_limit("codex", MSG)
        relay.check_limit("codex", "all good")

    def test_handover_to_other_agent(self):
        def limited(prompt, write):
            raise relay.Limited("codex", "2026-10-04 14:28")
        notes = []
        with mock.patch.dict(relay.CALL, {"codex": limited, "claude": lambda p, w: "ok"}):
            self.assertEqual(relay.ask("codex", "q", False, notes), ("claude", "ok"))
        self.assertIn("answered instead", notes[-1])

    def test_both_limited_stops(self):
        def limited(prompt, write):
            raise relay.Limited("x", "unknown")
        with mock.patch.dict(relay.CALL, {"codex": limited, "claude": limited, "gemini": limited}):
            with self.assertRaises(SystemExit):
                relay.ask("codex", "q", False, [])

    def test_crash_falls_back_to_next_agent(self):
        def broken(prompt, write):
            raise relay.AgentError("codex not found on PATH")
        with mock.patch.dict(relay.CALL, {"codex": broken, "claude": lambda p, w: "ok"}):
            self.assertEqual(relay.ask("codex", "q", False, []), ("claude", "ok"))

    def test_falls_through_to_gemini(self):
        def limited(prompt, write):
            raise relay.Limited("x", "unknown")
        calls = {"codex": limited, "claude": limited, "gemini": lambda p, w: "g"}
        with mock.patch.dict(relay.CALL, calls):
            self.assertEqual(relay.ask("codex", "q", False, [])[0], "gemini")

    def test_long_answer_mentioning_limit_is_kept(self):
        done = mock.Mock(returncode=0, stdout="x" * 400 + " rate limit", stderr="")
        with mock.patch("relay.shutil.which", return_value="claude"), \
             mock.patch("relay.subprocess.run", return_value=done):
            self.assertIn("rate limit", relay.call_claude("q", False))


if __name__ == "__main__":
    unittest.main()
