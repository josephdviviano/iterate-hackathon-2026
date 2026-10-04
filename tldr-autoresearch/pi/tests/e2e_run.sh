#!/usr/bin/env bash
# End-to-end pi run against a live OpenAI-compatible server, in a fresh throwaway git repo.
#
#   e2e_run.sh <template_dir> <work_dir> <agent_dir> <provider> <model> <thinking> <prompt_file>
#
#   e.g. pi/tests/e2e_run.sh pi/tests/toy_repo /tmp/pi-e2e/run1 "$PWD/pi/agent" vllm qwen3.8-27b-fp8 medium pi/tests/prompts/task.txt
#
# Copies <template_dir> into <work_dir>/repo, makes it a git repo on branch autoresearch/test,
# runs `pi --mode json` headlessly (stdin = /dev/null; `pi` from PATH, override with PI_BIN) and writes:
#   <work_dir>/events.jsonl   pi JSON event stream
#   <work_dir>/sessions/      pi session JSONL
#   <work_dir>/stderr.log     pi stderr
#   <work_dir>/time.txt       wall time and exit code
set -euo pipefail
TEMPLATE=$(realpath "$1") WORK=$(realpath -m "$2") AGENT_DIR=$(realpath "$3") PROVIDER=$4 MODEL=$5 THINKING=$6
PROMPT_FILE=$(realpath "$7")
export PATH=$HOME/.local/bin:$PATH
# the guard protects the project root (default: two levels above this script) from the agent's tools
export RLTLDR_ROOT="${RLTLDR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PI_CODING_AGENT_DIR=$AGENT_DIR PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0
export AR_GUARD_LOG=$WORK/guard_blocks.jsonl
rm -rf "$WORK" && mkdir -p "$WORK/sessions"
cp -r "$TEMPLATE" "$WORK/repo"
cd "$WORK/repo"
git init -q -b master && git config user.email agent@localhost && git config user.name agent
git add -A && git commit -qm baseline && git checkout -qb autoresearch/test
start=$(date +%s%N)
set +e
"${PI_BIN:-pi}" --mode json --no-approve --no-context-files --provider "$PROVIDER" --model "$MODEL" --thinking "$THINKING" \
   --session-dir "$WORK/sessions" -- "$(cat "$PROMPT_FILE")" < /dev/null > "$WORK/events.jsonl" 2> "$WORK/stderr.log"
rc=$?
set -e
end=$(date +%s%N)
python3 -c "import sys; print(f\"wall_s={(int(sys.argv[2])-int(sys.argv[1]))/1e9:.1f} exit={sys.argv[3]}\")" "$start" "$end" "$rc" | tee "$WORK/time.txt"
