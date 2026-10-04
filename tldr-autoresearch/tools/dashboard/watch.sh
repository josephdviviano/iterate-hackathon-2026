#!/bin/bash
# Event stream for the dashboard: on every run event that changes what the dashboard shows (experiment
# started/finished/voided, a training or confirmation run finished, group closed, policy published/activated)
# or an error, rebuild dashboard/index.html and
# print one line "HH:MM <event> || <build result>". The session watching this stream republishes the page.
#   tools/dashboard/watch.sh          (RLTLDR_ROOT selects the project root, as for build.py)
CODE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
ROOT=${RLTLDR_ROOT:-$CODE}
cd "$ROOT" || exit 1
tail -n 0 -F logs/driver.log logs/trainer.log logs/gateway.log logs/runner.log 2>/dev/null |
  grep --line-buffered -E "attempt .* (done|VOID)|attempt g[0-9]+-a[0-9]+(-r[0-9]+)?: policy=|run for .* finished|group .* closed|published|activated policy|Traceback|ERROR" |
  while IFS= read -r line; do
    out=$(/usr/bin/python3 "$CODE/tools/dashboard/build.py" 2>&1 | tail -1)
    printf '%s %s || %s\n' "$(date +%H:%M)" "$(printf '%s' "$line" | sed -E 's/^.*(INFO|ERROR|WARNING) [a-z]+: //' | cut -c1-160)" "$out"
  done
