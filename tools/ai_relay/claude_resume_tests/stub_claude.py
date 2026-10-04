"""テスト用の偽claude。STUB_MODE: ok | limit | error"""
import json, os, sys
mode = os.environ.get("STUB_MODE", "ok")
open(os.environ["STUB_LOG"], "a", encoding="utf-8").write(json.dumps(sys.argv[1:], ensure_ascii=False) + " cwd=" + os.getcwd() + "\n")
if mode == "ok":
    print(json.dumps({"is_error": False, "result": "続きを実行しました"}))
elif mode == "limit":
    print(json.dumps({"is_error": True, "result": "You've hit your limit · resets 11pm (Asia/Tokyo)"})); sys.exit(1)
else:
    print("boom", file=sys.stderr); sys.exit(2)
