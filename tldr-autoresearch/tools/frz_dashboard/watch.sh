#!/bin/bash
# Event stream for the frozen-v5 (insights vs none) dashboard: on every run/attempt/group event or error in
# logs/frz/*.log, rebuild dashboard_frz/index.html and print one line per event (the session republishes it).
#   tools/frz_dashboard/watch.sh      (RLTLDR_ROOT selects the project root; arms from data/frz/arms.json)
CODE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
ROOT=${RLTLDR_ROOT:-$CODE}
cd "$ROOT" || exit 1
export RLTLDR_ROOT=$ROOT H2H_CONFIG=$ROOT/data/frz/arms.json
tail -n 0 -F logs/frz/driver-x1.log logs/frz/driver-x2.log logs/frz/runner-x1.log logs/frz/runner-x2.log logs/frz/gateway-x1.log logs/frz/gateway-x2.log 2>/dev/null |
  grep --line-buffered -E "attempt .* (done|VOID)|run for .* finished|group .* closed|Traceback|ERROR| exited rc=" |
  while IFS= read -r line; do
    out=$(/usr/bin/python3 "$CODE/tools/frz_dashboard/build.py" 2>&1 | tail -1)
    printf '%s %s || %s\n' "$(date +%H:%M)" "$(printf '%s' "$line" | sed -E 's/^.*(INFO|ERROR|WARNING) [a-z_]+: //' | cut -c1-170)" "$out"
  done
