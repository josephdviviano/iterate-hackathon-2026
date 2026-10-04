#!/bin/bash
# Tests for ctl_h2h.sh and tools/h2h_dashboard/watch.sh with FAKE components in a temp root: test ports, a fake
# vLLM, fake adapters/repos, component commands replaced via H2H_CMD_*. Never touches production paths, the
# real vLLM or any GPU. Everything started here is stopped (and checked to be gone) before exit.
#   bash tests/test_ctl_h2h.sh
set -u
CODE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
CTL=$CODE/ctl_h2h.sh
T=$(mktemp -d /tmp/ctl_h2h_test.XXXXXX)
export RLTLDR_ROOT=$T H2H_STOP_TIMEOUT=6
unset H2H_CONFIG
PASS=0; FAIL=0; FAKE_VLLM=""; WATCH=""
ok()   { PASS=$((PASS + 1)); echo "ok   $*"; }
bad()  { FAIL=$((FAIL + 1)); echo "FAIL $*"; }
check() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }

cleanup() {
  [ -n "$WATCH" ] && kill -- -"$WATCH" 2>/dev/null
  "$CTL" stop >/dev/null 2>&1
  [ -n "$FAKE_VLLM" ] && kill "$FAKE_VLLM" 2>/dev/null
  pkill -KILL -f "^/usr/bin/python3 $T/fake.py" 2>/dev/null
  pkill -KILL -f "^sleep 98765" 2>/dev/null
  [ "$FAIL" -eq 0 ] && [ "$PASS" -gt 0 ] && rm -rf "$T"      # keep the temp root for inspection on failure
}
trap cleanup EXIT

# free test ports
read -r P_VLLM P_GW P_RB P_RV <<<"$(/usr/bin/python3 -c '
import socket
ps = []
for _ in range(4):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); ps.append(s)
print(*[s.getsockname()[1] for s in ps])')"

cat > "$T/fake.py" <<'EOF'
"""Fake h2h component: answers /health, /control/state, /v1/models on PORT (0 = no server); logs like the real
ones; --ignore-term ignores SIGTERM (exercises the force-kill path), --child spawns a grandchild."""
import http.server, json, signal, subprocess, sys, threading, time
role, port = sys.argv[1], int(sys.argv[2])
if "--child" in sys.argv:
    subprocess.Popen(["sleep", "98765"])
if "--ignore-term" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
else:
    signal.signal(signal.SIGTERM, lambda *a: (print("INFO shutting down", flush=True), sys.exit(0)))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"role": role, "busy": False, "job": None, "arms": {"base": {"calls": 3, "in_flight": 0},
                           "v5": {"calls": 4, "in_flight": 1}}, "data": [{"id": "qwen"}, {"id": "h2h-v5"}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass
if port:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"2026-10-04 01:00:00,000 INFO {role}: up on {port}", flush=True)
while True:
    time.sleep(1)
EOF
cat > "$T/h2h_config.json" <<EOF
{"gateway_port": $P_GW, "vllm_url": "http://127.0.0.1:$P_VLLM",
 "arms": [{"name": "base", "served_model": "h2h-base0", "adapter_dir": "$T/data/h2h/adapters/base0",
           "gpu_uuid": "GPU-00000000-0000-0000-0000-0000000000b0", "gpu_minor": 2, "runner_port": $P_RB},
          {"name": "v5", "served_model": "h2h-v5", "adapter_dir": "$T/data/h2h/adapters/v5",
           "gpu_uuid": "GPU-00000000-0000-0000-0000-0000000000f5", "gpu_minor": 3, "runner_port": $P_RV}]}
EOF
F="/usr/bin/python3 $T/fake.py"
export H2H_CMD_gateway="$F gateway $P_GW" H2H_CMD_runner_base="$F runner-base $P_RB" \
       H2H_CMD_runner_v5="$F runner-v5 $P_RV" H2H_CMD_agent_base="$F agent-base 0 --child" \
       H2H_CMD_agent_v5="$F agent-v5 0 --ignore-term --child"
running() { "$CTL" status 2>/dev/null | grep -cE "^  [a-z0-9-]+ +RUNNING"; }

echo "temp root $T (ports vllm $P_VLLM gateway $P_GW runners $P_RB/$P_RV)"
# 1. refusals: unknown component, vLLM down, adapters missing, agents before setup / before their services
out=$("$CTL" start bogus 2>&1); check "unknown component rejected" grep -q "unknown component" <<<"$out"
out=$("$CTL" start 2>&1); rc=$?
check "start refuses while vLLM is down" [ $rc -ne 0 ]; check "  says why" grep -q "vLLM does not answer" <<<"$out"
check "  nothing started" [ "$(running)" -eq 0 ]
/usr/bin/python3 "$T/fake.py" vllm "$P_VLLM" > /dev/null 2>&1 & FAKE_VLLM=$!
sleep 1
out=$("$CTL" start 2>&1); check "start refuses without adapters" grep -q "make_adapters.py" <<<"$out"
for a in base0 v5; do mkdir -p "$T/data/h2h/adapters/$a"; echo '{}' > "$T/data/h2h/adapters/$a/adapter_config.json"
  : > "$T/data/h2h/adapters/$a/adapter_model.safetensors"; done
out=$("$CTL" start agent-base 2>&1); check "agent refused before setup" grep -q "setup.py" <<<"$out"
for a in base v5; do mkdir -p "$T/data/h2h/$a/repo/.git" "$T/data/h2h/$a/session/agent"; done
out=$("$CTL" start agent-base 2>&1); check "agent refused while gateway is down" grep -q "gateway not answering" <<<"$out"
check "  nothing started" [ "$(running)" -eq 0 ]
mkdir -p "$T/run"; sleep 1000 & OLD=$!; echo $OLD > "$T/run/runner.child.pid"
out=$("$CTL" start 2>&1); check "refuses while the old RLTL;DR runner runs" grep -q "old RLTL;DR components" <<<"$out"
kill $OLD; rm -f "$T/run/runner.child.pid"
sed 's/"GPU-[0-9a-f-]*"/""/' "$T/h2h_config.json" > "$T/nogpu.json"
out=$(H2H_CONFIG=$T/nogpu.json "$CTL" start runner-base 2>&1)
check "runner refused while its GPU is not configured" grep -q "gpu_uuid/gpu_minor of arm base not set" <<<"$out"
check "  nothing started" [ "$(running)" -eq 0 ]

# 2. full start: dependency order, readiness waits, agents last
out=$("$CTL" start 2>&1); rc=$?
check "start all succeeds" [ $rc -eq 0 ]
check "  gateway + runners ready" [ "$(grep -c " ready$" <<<"$out")" -eq 3 ]
order=$(grep -o "^started [a-z0-9-]*" <<<"$out" | cut -d' ' -f2 | tr '\n' ' ')
check "  start order gateway runners agents ($order)" [ "$order" = "gateway runner-base runner-v5 agent-base agent-v5 " ]
check "  5 components RUNNING" [ "$(running)" -eq 5 ]
check "  pid files under run/h2h" [ -f "$T/run/h2h/gateway.pid" ] && [ -f "$T/run/h2h/agent-v5.child.pid" ]
check "  logs under logs/h2h" grep -q "up on" "$T/logs/h2h/runner-v5.log"
out=$("$CTL" start gateway 2>&1); check "second start is a no-op" grep -q "already running" <<<"$out"
st=$("$CTL" status 2>&1)
check "status: vLLM up" grep -q "up · models: qwen, h2h-v5" <<<"$st"
check "status: gateway state shown" grep -qE "^  v5:|in_flight=1" <<<"$st"
check "status: per-arm progress" grep -q "\[v5\] GPU 3" <<<"$st"
check "status: GPU lines" grep -q "arm base  GPU 2" <<<"$st"

# 3. watch.sh: events from the component logs -> one line each, dashboard rebuilt
setsid "$CODE/tools/h2h_dashboard/watch.sh" > "$T/watch.out" 2>&1 & WATCH=$!
sleep 2
echo "2026-10-04 01:02:03,456 INFO h2h_runner: run finished rc=0 in 352 s: val_bpb: 1.071 | peak_vram_mb: 25277.5" >> "$T/logs/h2h/runner-base.log"
echo "2026-10-04 01:02:04,000 INFO h2h_gateway: [v5] call abc ok fin=stop prompt=1 completion=2" >> "$T/logs/h2h/gateway.log"
echo "2026-10-04 01:02:05,000 INFO h2h_supervisor: prompt (nudge) #3 sent [nudge #2]" >> "$T/logs/h2h/agent-v5.log"
echo "2026-10-04 01:02:05,100 INFO h2h_supervisor: compaction started (threshold)" >> "$T/logs/h2h/agent-base.log"
echo "2026-10-04 01:02:06,000 INFO h2h_supervisor: compaction #2 done (threshold): 120000 -> 30000 tokens" >> "$T/logs/h2h/agent-base.log"
echo "2026-10-04 01:02:07,000 ERROR h2h_gateway: [v5] stream for call x stalled for 600s; aborting" >> "$T/logs/h2h/gateway.log"
for _ in $(seq 30); do [ "$(wc -l < "$T/watch.out")" -ge 4 ] && break; sleep 1; done
cat "$T/watch.out" | sed 's/^/     watch: /'
check "watch: run finished event" grep -q "runner-base: run finished rc=0" "$T/watch.out"
check "watch: nudge event" grep -q "agent-v5: prompt (nudge)" "$T/watch.out"
check "watch: routine gateway call ignored" bash -c "! grep -q 'call abc' '$T/watch.out'"
check "watch: dashboard rebuilt" grep -q "|| built $T/dashboard_h2h/index.html" "$T/watch.out"
check "watch: one compaction event (not the start)" [ "$(grep -c "agent-base: compaction" "$T/watch.out")" -eq 1 ]
check "watch: gateway error event" grep -q "gateway: ERROR \[v5\] stream for call x stalled" "$T/watch.out"
check "watch: burst shares one rebuild time" [ "$(cut -d' ' -f1 "$T/watch.out" | sort -u | wc -l)" -le 2 ]

# 4. crash -> supervised restart with back-off; the restart shows up in watch.sh too
kill -KILL "$(cat "$T/run/h2h/runner-base.child.pid")"
sleep 2
check "crashed component shows RESTARTING" grep -q "runner-base  RESTARTING" <<<"$("$CTL" status 2>&1)"
for _ in $(seq 25); do [ "$(running)" -eq 5 ] && break; sleep 1; done
check "  restarted after the back-off" [ "$(running)" -eq 5 ]
check "  restart counted" grep -qE "runner-base +RUNNING .*restarts 1" <<<"$("$CTL" status 2>&1)"
check "  log says why" grep -q "restarting runner-base in 15s" "$T/logs/h2h/runner-base.log"
sleep 3
check "watch: component exit event" grep -q "runner-base: === .* exited rc=137" "$T/watch.out"

# 5. restart one component
p1=$(cat "$T/run/h2h/gateway.child.pid")
"$CTL" restart gateway > /dev/null 2>&1
p2=$(cat "$T/run/h2h/gateway.child.pid" 2>/dev/null)
check "restart gateway: new process" [ -n "$p2" ] && [ "$p1" != "$p2" ] && kill -0 "$p2"

# 6. logs
out=$(timeout 3 "$CTL" logs runner-v5 2>&1); check "logs tails the component log" grep -q "up on $P_RV" <<<"$out"

# 7. stop: agents first and together, the TERM-ignoring agent force-killed with its whole tree
t0=$(date +%s); out=$("$CTL" stop 2>&1); dt=$(( $(date +%s) - t0 ))
echo "$out" | sed 's/^/     stop: /'
order=$(grep -o "^stopped [a-z0-9-]*" <<<"$out" | cut -d' ' -f2 | tr '\n' ' ')
check "stop order agents runners gateway ($order)" [ "$order" = "agent-base agent-v5 runner-base runner-v5 gateway " ]
check "  TERM-ignoring agent force-killed" grep -q "agent-v5 did not exit within 6s" <<<"$out"
check "  agents stopped in parallel (${dt}s)" [ $dt -lt 20 ]
check "  nothing running" [ "$(running)" -eq 0 ]
sleep 1
check "  no fake component left" bash -c "! pgrep -f '^/usr/bin/python3 $T/fake.py (gateway|runner|agent)' >/dev/null"
check "  grandchildren killed (agent-v5 tree)" bash -c "[ \$(pgrep -fc '^sleep 98765') -le 1 ]"
check "  pid files removed" bash -c "! ls $T/run/h2h/*.pid >/dev/null 2>&1"
# agent-base handled TERM itself: its grandchild 'sleep 98765' is orphaned by design of the fake (the real
# supervisor kills its sandbox tree); clean it up here
pkill -KILL -f "^sleep 98765" 2>/dev/null

kill -- -"$WATCH" 2>/dev/null; WATCH=""
kill "$FAKE_VLLM" 2>/dev/null; FAKE_VLLM=""
sleep 1
check "watch.sh stopped" bash -c "! pgrep -f '^/bin/bash $CODE/tools/h2h_dashboard/watch.sh' >/dev/null"
echo; echo "$PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
