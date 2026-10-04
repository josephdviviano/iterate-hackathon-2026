#!/bin/bash
# Event stream for the h2h dashboard: on every event that changes what the dashboard shows or needs attention
# (a runner finished a run; the supervisor nudged, restarted or compacted an agent, or backed off; an error or
# Traceback in any component; a component exited and ctl_h2h.sh restarts it) rebuild dashboard_h2h/index.html
# and print one line per event: "HH:MM <component>: <event> || <build result>". Events that arrive together
# (within 2 s of each other, at most 10 s / 20 events) share one rebuild. The session watching this stream
# republishes the page.
#   tools/h2h_dashboard/watch.sh          (RLTLDR_ROOT selects the data root, as for build.py)
# Matching is done in bash: mawk (this machine's awk) block-buffers pipe input and would sit on events.
CODE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
ROOT=${RLTLDR_ROOT:-$CODE}
LOGS=$ROOT/logs/h2h
ARMS=$(PYTHONPATH=$CODE /usr/bin/python3 -c 'from rltldr.h2h_config import load_h2h_config as l
print(" ".join(a.name for a in l().arms))') || exit 1
files="$LOGS/gateway.log"
for a in $ARMS; do files="$files $LOGS/runner-$a.log $LOGS/agent-$a.log"; done
mkdir -p "$LOGS"

ANY='Traceback|ERROR|CRITICAL| exited rc='
RUNNER='[Rr]un .*(finished|done|complete)'                       # h2h_runner: "run finished rc=0 in 352 s: ..."
AGENT='[Nn]udge|[Rr]estart|[Cc]ompaction #|[Bb]ack-?off|[Ww]atchdog' # h2h_supervisor + ctl "restarting ..."
HEADER='^==> (.*) <==$'                                           # tail -F names the file it switches to

flush() {
  local out l msg
  out=$(/usr/bin/python3 "$CODE/tools/h2h_dashboard/build.py" 2>&1 | tail -1)
  for l in "${batch[@]}"; do
    # "runner-v5: 2026-10-04 01:02:03,456 INFO h2h_runner: run finished ..." -> "runner-v5: run finished ..."
    msg=$(printf '%s' "$l" | sed -E 's/^([A-Za-z0-9_-]+): [0-9-]+[ T][0-9:,.]+ (INFO|WARNING) [A-Za-z0-9_.]+: /\1: /; s/^([A-Za-z0-9_-]+): [0-9-]+[ T][0-9:,.]+ (ERROR|CRITICAL) [A-Za-z0-9_.]+: /\1: \2 /' | cut -c1-180)
    printf '%s %s || %s\n' "$(date +%H:%M)" "$msg" "$out"
  done
  batch=()
}

batch=(); comp=""; deadline=0
tail -n 0 -F $files 2>/dev/null | while true; do
  if [ ${#batch[@]} -eq 0 ]; then IFS= read -r line; rc=$?; else IFS= read -r -t 2 line; rc=$?; fi
  if [ $rc -eq 0 ]; then
    if [[ $line =~ $HEADER ]]; then comp=$(basename "${BASH_REMATCH[1]}" .log); continue; fi
    [[ $line == "=== "*" starting "* ]] && continue
    if [[ $line =~ $ANY ]] || { [[ $comp == runner-* ]] && [[ $line =~ $RUNNER ]]; } ||
       { [[ $comp == agent-* ]] && [[ $line =~ $AGENT ]]; }; then
      [ ${#batch[@]} -eq 0 ] && deadline=$((SECONDS + 10))
      batch+=("$comp: $line")
    fi
  elif [ $rc -le 128 ]; then                     # EOF: tail is gone
    [ ${#batch[@]} -gt 0 ] && flush
    exit 0
  fi
  if [ ${#batch[@]} -gt 0 ] && { [ $rc -gt 128 ] || [ $SECONDS -ge $deadline ] || [ ${#batch[@]} -ge 20 ]; }; then
    flush
  fi
done
