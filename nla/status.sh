#!/bin/zsh
# usage: ./status.sh [grep-pattern]   — tail rank-0 lines of every running qwen38-nla container
pat=${1:-'\[r0\]|ddp_run|Error|Traceback'}
for cid in $(modal container list --json 2>/dev/null | python3 -c "import json,sys;[print(x['container_id']) for x in json.load(sys.stdin) if x['app_name']=='qwen38-nla']"); do
  echo "== $cid"
  python3 - "$cid" "$pat" <<'PY'
import re, subprocess, sys
cid, pat = sys.argv[1], sys.argv[2]
try: out = subprocess.run(["modal", "container", "logs", cid], capture_output=True, text=True, timeout=25).stdout
except subprocess.TimeoutExpired as e: out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
print("\n".join([l for l in out.splitlines() if re.search(pat, l)][-8:]))
PY
done
