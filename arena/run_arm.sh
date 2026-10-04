#!/usr/bin/env bash
# Supervise one arm: an autonomous Claude Code session in the arm's workspace, resumed whenever it
# ends, until a STOP file appears. The prompt is identical for every arm and says nothing about the
# task or the framework: both come from the workspace's program.md and task.md.
#
#   run_arm.sh <arena dir> <arm>          stop: touch <arena dir>/STOP (or STOP.<arm>)
set -u
ARENA=$1 ARM=$2
DIR=$ARENA/$ARM LOG=$ARENA/logs/$ARM SID_FILE=$ARENA/logs/$ARM.session
AGENT=(--model claude-opus-5-5 --effort high --dangerously-skip-permissions --output-format stream-json --verbose)
FIRST="Hi! Have a look at program.md and let's kick off a new experiment run. The human has already \
confirmed the setup, so do not wait for confirmation: use the run tag \`run1\`. Do the setup, then \
start the experiment loop and never stop."
CONT="Continue the experiment loop as program.md says. Check the current state first. Never stop."
stopped() { [ -e "$ARENA/STOP" ] || [ -e "$ARENA/STOP.$ARM" ]; }
mkdir -p "$ARENA/logs"; cd "$DIR" || exit 1
if [ ! -s "$SID_FILE" ]; then
  python3 -c 'import uuid; print(uuid.uuid4())' > "$SID_FILE"
  echo "$(date -Is) start $(cat "$SID_FILE")" >> "$LOG.events"
  claude -p "$FIRST" --session-id "$(cat "$SID_FILE")" "${AGENT[@]}" >> "$LOG.jsonl" 2>> "$LOG.err"
  echo "$(date -Is) ended ($?)" >> "$LOG.events"
fi
while ! stopped; do
  t0=$(date +%s)
  echo "$(date -Is) resume" >> "$LOG.events"
  claude -p "$CONT" --resume "$(cat "$SID_FILE")" "${AGENT[@]}" >> "$LOG.jsonl" 2>> "$LOG.err"
  echo "$(date -Is) ended ($?)" >> "$LOG.events"
  if [ $(( $(date +%s) - t0 )) -lt 120 ]; then sleep 300; else sleep 15; fi
done
echo "$(date -Is) stopped" >> "$LOG.events"
