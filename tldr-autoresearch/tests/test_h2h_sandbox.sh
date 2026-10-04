#!/usr/bin/env bash
# Tests for tools/h2h_sandbox.sh, the per-arm agent sandbox of the head-to-head (h2h) runs.
#   tests/test_h2h_sandbox.sh
# Needs passwordless sudo. Uses no GPU (the sandbox masks them all) and never contacts vLLM.
#
# A throw-away stand-in root (RLTLDR_ROOT override) is built under $H2H_TEST_PARENT (default $ROOT/scratch, i.e.
# inside the hidden home tree when the project lives under $HOME, like production): data/h2h/{base,v5}/{repo,session}
# (repos cloned from canon.git h2h/start, base with `uv sync --frozen --offline` .venv) and
# run/h2h/{base,v5}/{gateway,runner}.sock served by tiny fake HTTP servers that answer with their own label. Hiding is
# checked against the REAL contents of the invoking user's $HOME (the tree the sandbox covers). A checker (written
# into each arm's session dir) runs inside the sandbox for both arms and both layouts (neutral, host) and prints
# PASS/FAIL lines; the rest is checked from outside. Also: a stand-in root under /tmp (which the sandbox's private
# /tmp shadows), killing a sandbox from outside, leftovers, and, when tools/h2h/setup.py has created them, read-only
# checks against the real arm dirs.
# Env: H2H_TEST_PARENT, H2H_TEST_KEEP=1 (keep the stand-in roots), H2H_TEST_REAL=0 (skip the real-arm checks).
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
SBX=$ROOT/tools/h2h_sandbox.sh
H=$HOME                                   # the sandbox covers the invoking user's home
RUID=$(id -u) RGID=$(id -g)               # what the sandbox drops to
REL=""; [[ $ROOT == "$H"/* ]] && REL=${ROOT#"$H"/}     # project root relative to $H ("" = outside the home)
PARENT=${H2H_TEST_PARENT:-$ROOT/scratch}
NPASS=0 NFAIL=0 FAILED=()
pass() { NPASS=$((NPASS + 1)); echo "PASS $*"; }
fail() { NFAIL=$((NFAIL + 1)); FAILED+=("$*"); echo "FAIL $*"; }
check() { local name=$1; shift; if "$@"; then pass "$name"; else fail "$name"; fi; }

TROOT=$(mktemp -d "$PARENT/h2h_sbx_test.XXXXXX")
TMPROOT=$(mktemp -d "${TMPDIR:-/tmp}/h2h_sbx_tmproot.XXXXXX")     # must lie under /tmp (see below)
[[ $TMPROOT == /tmp/* ]] || { echo "TMPDIR must be under /tmp"; exit 1; }
SERVER_PIDS=()
cleanup() {
  local p
  for p in "${SERVER_PIDS[@]}"; do kill "$p" 2>/dev/null; done
  if [ "${H2H_TEST_KEEP:-0}" = 1 ]; then echo "kept $TROOT $TMPROOT"; else rm -rf "$TROOT" "$TMPROOT"; fi
}
trap cleanup EXIT

sbx() { RLTLDR_ROOT=$TROOT timeout 180 "$SBX" "$@"; }

# ---------------------------------------------------------------------------------------------------- fixtures
cat > "$TROOT/fake_uds.py" <<'PY'
"""Tiny HTTP servers on Unix sockets: `fake_uds.py LABEL=PATH ...`; every answer names its server."""
import json, os, socketserver, sys, threading
from http.server import BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def address_string(self): return "uds"
    def reply(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        self.reply({"who": self.server.label, "method": "GET", "path": self.path})
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/run":
            d = json.loads(body or b"{}")
            self.reply({"rc": 0, "output": f"fake-runner {self.server.label} desc={d.get('desc')!r} "
                                           f"train_py_bytes={len(d.get('train_py', ''))}"})
        else:
            self.reply({"who": self.server.label, "method": "POST", "path": self.path, "len": len(body)})

class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

for spec in sys.argv[1:]:
    label, path = spec.split("=", 1)
    if os.path.exists(path):
        os.unlink(path)
    s = Server(path, Handler)
    s.label = label
    os.chmod(path, 0o666)
    threading.Thread(target=s.serve_forever, daemon=True).start()
print("ready", flush=True)
threading.Event().wait()
PY

cat > "$TROOT/checker.py" <<'PY'
"""Runs INSIDE the sandbox: checker.py ARM OTHER LAYOUT TROOT TOOLS MODE(full|real) HOME UID GID REL.
REL = the project root relative to HOME ("" if it lies outside). Prints PASS/FAIL lines."""
import errno, json, os, re, socket, stat, subprocess, sys, time, urllib.request

arm, other, layout, troot, tools, mode, H, ruid, rgid, root_rel = sys.argv[1:11]
ruid, rgid = int(ruid), int(rgid)
if layout == "neutral":
    repo, sess, gw_dir, rn_dir = f"{H}/autoresearch", f"{H}/.session", f"{H}/.sock/gateway", f"{H}/.sock/runner"
else:
    repo, sess = f"{troot}/data/h2h/{arm}/repo", f"{troot}/data/h2h/{arm}/session"
    gw_dir = rn_dir = f"{troot}/run/h2h/{arm}"
tool_files = [f"{tools}/run_client.py", f"{tools}/uds_forward.py"]     # the only files of tools/ inside
exposed = [f"{H}/.local/bin", f"{H}/.local/lib", f"{H}/.local/opt", f"{H}/.local/share/uv",
           f"{H}/.cache/autoresearch", f"{H}/.cache/uv", *tool_files, gw_dir, rn_dir, repo, sess]


def out(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f": {detail}"), flush=True)


def run(*cmd, **kw):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, repr(e)


def try_create(d):
    fn = os.path.join(d, ".h2h_sbx_probe")
    try:
        os.close(os.open(fn, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600))
        os.unlink(fn)
        return "ok"
    except OSError as e:
        return errno.errorcode.get(e.errno, str(e.errno))


# --- identity / privileges
st = dict(l.split(":", 1) for l in open("/proc/self/status").read().splitlines() if ":" in l)
out(os.getuid() == ruid and os.geteuid() == ruid and os.getgid() == rgid,
    f"uid {ruid} / gid {rgid} (the invoking user)", f"{os.getuid()}")
out(st["NoNewPrivs"].strip() == "1", "no_new_privs set", st["NoNewPrivs"])
caps = {k: int(st[k].strip(), 16) for k in ("CapInh", "CapPrm", "CapEff", "CapAmb")}
out(not any(caps.values()), "no capabilities", str(caps))
try:
    os.setuid(0)
    out(False, "setuid(0) refused")
except PermissionError:
    out(True, "setuid(0) refused")
rc, o = run("sudo", "-n", "true")
out(rc != 0, "sudo refused", o[:200])
rc, o = run("su", "-c", "true", stdin=subprocess.DEVNULL)
out(rc != 0, "su refused", o[:200])
rc, o = run("unshare", "-Ur", "id", "-u")
out(rc != 0 or o.strip() != "0", "no root via a user namespace", o[:200])
rc, o = run("unshare", "-Urm", "sh", "-c", f"umount -l {H} 2>&1; mount --bind / /mnt 2>&1; ls -a {H} /mnt{H} 2>&1")
out(".claude" not in o and "models" not in o, "user-namespace umount/bind cannot uncover the home", o[:300])

# --- hidden paths (real home contents) and a whitelist walk of everything visible under it
HOME_HIDDEN = [".claude", ".claude.json", ".pi", "models", "envs", "work", "logs", ".config", ".jupyter", ".bashrc",
          ".npm", ".nv", ".triton", ".cache/huggingface", ".cache/vllm", ".cache/claude", ".cache/flashinfer",
          ".local/share/claude", ".local/share/jupyter", ".local/share/pki", ".local/state"]
ROOT_HIDDEN = [  # relative to the project root (checked when it lies under the home)
          "data", "data/ledger.jsonl", "data/adapters", "data/h2h/adapters",
          f"data/h2h/{arm}/ledger.jsonl", f"data/h2h/{arm}/runner_ws", f"data/h2h/{other}",
          "canon.git", "logs", "README.md", "RESULTS.md", "results", "examples", "docs", "docs/AR_RUN.md",
          "docs/DESIGN.md", "docs/RETROSPECTIVE.md", "docs/SETUP.md", "config.json", "config.example.json",
          "h2h_config.json", "h2h_config.example.json", "ctl.sh", "serve.sh", "autoresearch", "pi_sessions", "runner_ws",
          "canon_ws", "policy", "pi", "rltldr", "run", "run/gateway.sock",
          f"run/h2h/{other}", "scratch", "dashboard", "tests", "agent_venv",
          "tools/h2h", "tools/h2h/program.md", "tools/h2h_dashboard",
          "tools/dashboard", "tools/export_results.py", "tools/report.py",
          "tools/h2h_sandbox.sh", "tools/sandbox.sh", "tools/sandbox_lib.sh",
          "tools/ar_run.py", "tools/run.sh", "tools/__pycache__"]
HIDDEN = HOME_HIDDEN + ([os.path.join(root_rel, p) for p in ROOT_HIDDEN] if root_rel else [])
under = [e for e in exposed if e.startswith(H + "/")]
visible = []
for rel in HIDDEN:
    p = os.path.normpath(f"{H}/{rel}")
    if any(e == p or e.startswith(p + "/") for e in under):
        continue          # an ancestor of an exposed path: only its skeleton exists (checked by the walk below)
    if os.path.lexists(p):
        visible.append(rel)
out(not visible, "real home contents hidden", str(visible))
allowed = {H}
for e in under:
    while e.startswith(H):
        allowed.add(e)
        e = os.path.dirname(e)
dev, bad = os.stat(H).st_dev, []
for d, dirs, files in os.walk(H):
    for n in dirs + files:
        if os.path.join(d, n) not in allowed:
            bad.append(os.path.join(d, n))
    dirs[:] = [n for n in dirs if os.lstat(os.path.join(d, n)).st_dev == dev]      # stop at exposed mounts
out(not bad, f"only the listed paths are visible under {H}", str(bad[:20]))
need = [f"{H}/.local/bin/pi", f"{H}/.local/bin/node", f"{H}/.local/bin/uv", f"{H}/.cache/autoresearch/tokenizer",
        *tool_files, f"{repo}/train.py", f"{repo}/.git"]
out(all(os.path.exists(p) for p in need), "needed paths present", str([p for p in need if not os.path.exists(p)]))
listed = sorted(os.listdir(tools)) if os.path.isdir(tools) else None
out(listed == ["run_client.py", "uds_forward.py"], "tools/: only run_client.py and uds_forward.py", str(listed))
# nothing readable in tools/ names the experiment or an arm (the dashboards and the spec do)
MARK = re.compile(rb"v5|rltl|h2h|head.?to.?head|base0|policy|lora|adapter|qwen", re.I)
hits = []
for d, _, files in os.walk(tools):
    for n in files:
        with open(os.path.join(d, n), "rb") as f:
            hits += [f"{n}: {m.group().decode(errors='replace')}" for m in MARK.finditer(f.read())]
out(not hits, "no file in tools/ mentions the experiment or the arms", str(hits[:10]))
if layout == "neutral":
    out(not os.path.lexists(f"{troot}/data/h2h/{arm}") and not os.path.lexists(f"{troot}/run/h2h/{arm}"),
        "neutral: host paths of the arm invisible")
else:
    out(os.path.isdir(repo) and not os.path.lexists(f"{troot}/data/h2h/{other}")
        and not os.path.lexists(f"{troot}/run/h2h/{other}"), "host: own arm visible, other arm invisible")
out(not os.path.lexists(f"{troot}/data/h2h/{other}") and not os.path.lexists(f"{troot}/run/h2h/{other}"),
    "other arm's repo and sockets invisible")
out(os.listdir("/tmp") == [] and os.listdir("/dev/shm") == [], "private empty /tmp and /dev/shm",
    str(os.listdir("/tmp")[:5] + os.listdir("/dev/shm")[:5]))

# --- write access
if mode == "full":
    rw = {d: try_create(d) for d in (repo, sess, "/tmp", "/var/tmp", "/dev/shm")}
    out(all(v == "ok" for v in rw.values()), "own repo, session dir, private tmp dirs writable", str(rw))
    rc1, o1 = run("git", "status", "--porcelain", cwd=repo)
    rc2, o2 = run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m",
                  "h2h sandbox test", cwd=repo)
    out(rc1 == 0 and rc2 == 0, "git status/commit in the repo", (o1 + o2)[:300])
    with open(f"{repo}/created_inside.txt", "w") as f:
        f.write(f"{arm}\n")
ro = [H, f"{H}/.local/bin", f"{H}/.local/opt", f"{H}/.cache/autoresearch", tools, gw_dir, rn_dir]
if os.path.isdir(f"{repo}/.venv"):
    ro.append(f"{repo}/.venv")
res = {d: try_create(d) for d in ro}
out(all(v == "EROFS" for v in res.values()), "home skeleton, tools, sockets, data, .venv read-only", str(res))
def try_append(p):
    try:
        open(p, "ab").close()
        return "ok"
    except OSError as e:
        return errno.errorcode.get(e.errno, str(e.errno))
res = {p: try_append(p) for p in tool_files}
out(all(v == "EROFS" for v in res.values()), "tools files read-only", str(res))
res = {d: try_create(d) for d in ("/", "/usr", "/usr/bin", "/etc", "/opt")}
out(all(v in ("EROFS", "EACCES") for v in res.values()), "rest of the filesystem read-only", str(res))

# --- GPUs
devs = [f"/dev/{n}" for n in os.listdir("/dev") if n.startswith("nvidia")]
notnull = [d for d in devs if stat.S_ISCHR(os.stat(d).st_mode) and (os.major(os.stat(d).st_rdev),
                                                                    os.minor(os.stat(d).st_rdev)) != (1, 3)]
out(len(devs) >= 4 and not notnull, "every /dev/nvidia* is /dev/null", f"{len(devs)} devs, real: {notnull}")
rc, o = run("nvidia-smi", "-L")
out(rc != 0 and "GPU-" not in o, "nvidia-smi sees no GPU", o[:200])

# --- network
def tcp(host, port):
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect((host, port))
        return "connected"
    except OSError as e:
        return errno.errorcode.get(e.errno, str(e)) if e.errno else type(e).__name__
    finally:
        s.close()
res = {f"{h}:{p}": tcp(h, p) for h, p in (("1.1.1.1", 443), ("8.8.8.8", 53), ("127.0.0.1", 8000),
                                            ("127.0.0.1", 8110), ("127.0.0.1", 8210), ("127.0.0.1", 8220))}
out(all(v != "connected" for v in res.values()), "no network besides the forwards (vLLM, trusted ports, internet)",
    str(res))
try:
    socket.getaddrinfo("example.com", 443)
    out(False, "no DNS")
except OSError:
    out(True, "no DNS")
ifs = [l.split(":")[0].strip() for l in open("/proc/net/dev").read().splitlines()[2:]]
out(ifs == ["lo"], "only the loopback interface", str(ifs))
listen = sorted(int(l.split()[1].split(":")[1], 16) for l in open("/proc/net/tcp").read().splitlines()[1:]
                if l.split()[3] == "0A")
out(listen == [8100, 8200], "only 8100/8200 listen", str(listen))

# --- forwards
if mode == "full":
    def get(url, data=None):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    try:
        g = get("http://127.0.0.1:8100/v1/models")
        r = get("http://127.0.0.1:8200/run", json.dumps({"desc": "d", "train_py": "x" * 5}).encode())
        ok = g["who"] == f"gateway-{arm}" and r["output"].startswith(f"fake-runner runner-{arm} ")
        out(ok, "8100 -> own gateway, 8200 -> own runner", f"{g} {r}")
    except Exception as e:
        out(False, "8100 -> own gateway, 8200 -> own runner", repr(e))
    rc, o = run("./run.sh", "sandbox test run", cwd=repo)
    out(rc == 0 and f"fake-runner runner-{arm} desc='sandbox test run'" in o, "./run.sh reaches own runner", o[:300])

# --- processes, environment, handles into the hidden tree
procs = {}
for p in os.listdir("/proc"):
    if p.isdigit():
        try:
            procs[int(p)] = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            pass
flat = b" ".join(b" ".join(v) for v in procs.values())
out(1 in procs and procs[1][:2] == [b"/bin/bash", b"-c"] and b"h2h-init" in procs[1] and len(procs) < 15
    and b"vllm" not in flat and b"sudo" not in flat and b"unshare" not in flat,
    "own PID namespace (bash init, no outside processes)", str({k: v[:3] for k, v in procs.items()}))
out(all(os.stat(f"/proc/{p}").st_uid == ruid for p in procs if os.path.exists(f"/proc/{p}")),
    "no root process inside")
extra = set(os.environ) - {"HOME", "PATH", "LANG", "USER", "LOGNAME", "PWD", "SHLVL", "_", "OLDPWD"}
out(not extra and os.environ.get("HOME") == H, "clean environment", str(sorted(extra)))
out(os.getcwd() == repo, "cwd re-resolved to the repo", os.getcwd())
FORBID = {"canon.git", ".claude", ".claude.json", "RETROSPECTIVE.md", "pi_sessions", "runner_ws", "envs", other}
leaks = []
for p in procs:
    links = [f"/proc/{p}/cwd", f"/proc/{p}/root"]
    try:
        links += [f"/proc/{p}/fd/{n}" for n in os.listdir(f"/proc/{p}/fd")]
    except OSError:
        pass
    for l in links:
        try:
            target = os.readlink(l)
        except OSError:
            continue
        if not target.startswith("/") or target.endswith(" (deleted)") or not os.path.isdir(l):
            if target.startswith("/") and not target.startswith(("/dev/", "/proc/")) and not os.path.exists(target):
                leaks.append(f"{l} -> {target} (not visible by path)")
            continue
        for k in range(30):
            try:
                names = set(os.listdir(l + "/.." * k))
            except OSError:
                break
            if names & FORBID:
                leaks.append(f"{l}{'/..' * k}: {sorted(names & FORBID)}")
                break
out(not leaks, "no cwd/root/fd of any process reaches the hidden tree", str(leaks[:5]))

# --- orphans are reaped by PID 1
subprocess.run(["/bin/sh", "-c", "sleep 0.3 & exit 0"])
time.sleep(1.5)
zombies = []
for p in os.listdir("/proc"):
    if p.isdigit():
        try:
            if open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[0] == "Z":
                zombies.append(p)
        except OSError:
            pass
out(not zombies, "orphans reaped (no zombies)", str(zombies))
PY

echo "== setup: stand-in root $TROOT"
for arm in base v5; do
  A=$TROOT/data/h2h/$arm
  mkdir -p "$A/session/agent" "$TROOT/run/h2h/$arm"
  git clone -q --single-branch --branch h2h/start --no-tags "$ROOT/canon.git" "$A/repo" || { echo "clone failed"; exit 1; }
  cp "$TROOT/checker.py" "$A/session/h2h_sbx_checker.py"
done
(cd "$TROOT/data/h2h/base/repo" && "${UV_BIN:-uv}" sync --frozen --offline -q) || echo "WARN: uv sync failed"
/usr/bin/python3 -I "$TROOT/fake_uds.py" \
  "gateway-base=$TROOT/run/h2h/base/gateway.sock" "runner-base=$TROOT/run/h2h/base/runner.sock" \
  "gateway-v5=$TROOT/run/h2h/v5/gateway.sock" "runner-v5=$TROOT/run/h2h/v5/runner.sock" > "$TROOT/fake.out" 2>&1 &
SERVER_PIDS+=($!)
for i in $(seq 50); do grep -q ready "$TROOT/fake.out" 2>/dev/null && break; sleep 0.1; done
[ -n "$REL" ] || echo "WARN: $ROOT is not under $H: the project-root hiding checks are skipped"
for p in ${REL:+$REL/data $REL/canon.git} .claude .claude.json .pi models envs; do
  [ -e "$H/$p" ] || echo "WARN: $H/$p does not exist on the host, its hiding check is vacuous"
done

# ------------------------------------------------------------------------- inside checks: 2 layouts x 2 arms
for layout in neutral host; do
  for arm in base v5; do
    other=$([ "$arm" = base ] && echo v5 || echo base)
    A=$TROOT/data/h2h/$arm
    # host paths in the command: the neutral layout rewrites them
    o=$(cd "$A/repo" && H2H_SANDBOX_LAYOUT=$layout sbx "$arm" /usr/bin/python3 -I "$A/session/h2h_sbx_checker.py" \
        "$arm" "$other" "$layout" "$TROOT" "$ROOT/tools" full "$H" "$RUID" "$RGID" "$REL" </dev/null 2>&1)
    rc=$?
    n=$(grep -c '^PASS ' <<< "$o")
    [ "$rc" -eq 0 ] && [ "$n" -ge 30 ] || fail "[$layout/$arm] checker ran (rc=$rc, $n PASS lines): ${o:0:500}"
    while IFS= read -r line; do
      case $line in PASS\ *) pass "[$layout/$arm] ${line#PASS }" ;; FAIL\ *) fail "[$layout/$arm] ${line#FAIL }" ;; esac
    done <<< "$o"
    check "[$layout/$arm] file written inside is in the host repo" grep -qx "$arm" "$A/repo/created_inside.txt"
    rm -f "$A/repo/created_inside.txt"
  done
  B=$TROOT/data/h2h/base
  o=$(cd "$B/repo" && H2H_SANDBOX_LAYOUT=$layout sbx base /usr/bin/env -i PATH=$H/.local/bin:/usr/bin:/bin HOME=$H \
      PI_CODING_AGENT_DIR=$B/session/agent PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 pi --version \
      </dev/null 2>&1)
  check "[$layout] pi --version (supervisor env) -> '$o'" grep -qE '^[0-9]+\.[0-9]+' <<< "$o"
  o=$(cd "$B/repo" && H2H_SANDBOX_LAYOUT=$layout sbx base pi --version </dev/null 2>&1)
  check "[$layout] pi --version (bare) -> '$o'" grep -qE '^[0-9]+\.[0-9]+' <<< "$o"
  o=$(cd "$B/repo" && H2H_SANDBOX_LAYOUT=$layout sbx base "$B/repo/.venv/bin/python" -c \
      "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())" \
      </dev/null 2>&1)
  check "[$layout] .venv python imports torch, no CUDA device -> '${o##*$'\n'}'" grep -qE '^torch .* False 0$' <<< "$o"
done

# ---------------------------------------------------------------------------------------- outside checks
B=$TROOT/data/h2h/base
sbx base /bin/sh -c 'exit 7' </dev/null; rc=$?
check "exit status passes through (7) -> $rc" test "$rc" = 7
printf 'a\r\nb\n\0c\377' > "$TROOT/bin.in"
sbx base /bin/cat < "$TROOT/bin.in" > "$TROOT/bin.out" 2>/dev/null
check "stdin/stdout pass through byte-exact" cmp -s "$TROOT/bin.in" "$TROOT/bin.out"
o=$(cd "$B/repo" && sbx base /usr/bin/printf '%s\n' "$B/repo/train.py" "X=$B/session/agent" "--session-dir=$B/session/s" \
    "$B/repo-other" "/x$B/repo" "$B/repo" </dev/null 2>&1)
exp=$(printf '%s\n' "$H/autoresearch/train.py" "X=$H/.session/agent" "--session-dir=$H/.session/s" "$B/repo-other" \
      "/x$B/repo" "$H/autoresearch")
check "neutral: host repo/session paths in args rewritten (whole components only)" test "$o" = "$exp"
o=$(cd "$B/repo/.git" && sbx base /bin/pwd </dev/null 2>&1)
check "neutral: cwd inside the repo mapped -> $o" test "$o" = "$H/autoresearch/.git"
o=$(cd "$B/session" && sbx base /bin/pwd </dev/null 2>&1)
check "neutral: cwd in the session dir mapped -> $o" test "$o" = "$H/.session"
o=$(cd / && sbx base /bin/pwd </dev/null 2>&1)
check "cwd outside the arm kept -> $o" test "$o" = "/"
o=$(cd "$B/repo" && H2H_SANDBOX_LAYOUT=host sbx base /usr/bin/printf '%s' "$B/repo" </dev/null 2>&1)
check "host: args not rewritten" test "$o" = "$B/repo"
o=$(sbx nosuch /bin/true 2>&1); rc=$?
check "unknown arm refused (rc=$rc: $o)" test "$rc" = 2
o=$(H2H_SANDBOX_LAYOUT=bogus sbx base /bin/true 2>&1); rc=$?
check "bad layout refused (rc=$rc)" test "$rc" = 2
mv "$TROOT/data/h2h/v5/repo" "$TROOT/data/h2h/v5/repo.away"
o=$(sbx v5 /bin/true 2>&1); rc=$?
check "missing repo refused (rc=$rc: $o)" test "$rc" = 2
mv "$TROOT/data/h2h/v5/repo.away" "$TROOT/data/h2h/v5/repo"
mkdir -p "$TROOT/stub/tools" "$TROOT/stub/rltldr"     # a copy of the script next to a tools/ without run_client.py
cp "$SBX" "$ROOT/tools/sandbox_lib.sh" "$ROOT/tools/uds_forward.py" "$TROOT/stub/tools/"
cp "$ROOT/rltldr/h2h_config.py" "$TROOT/stub/rltldr/"
o=$(RLTLDR_ROOT=$TROOT timeout 60 "$TROOT/stub/tools/h2h_sandbox.sh" base /bin/true 2>&1); rc=$?
check "missing tools/run_client.py refused (rc=$rc: $o)" test "$rc" = 2 -a -z "${o##*missing*run_client.py*}"
# a gateway restart (new socket file in the same dir) is picked up by a running sandbox
(cd "$B/repo" && sbx base /usr/bin/python3 -I -c '
import json, time, urllib.request
get = lambda: json.loads(urllib.request.urlopen("http://127.0.0.1:8100/v1/models", timeout=5).read())["who"]
a = get(); time.sleep(4); print(a, get())' </dev/null > "$TROOT/restart.out" 2>&1) &
BG=$!
sleep 2
/usr/bin/python3 -I "$TROOT/fake_uds.py" "gateway-base-2=$TROOT/run/h2h/base/gateway.sock" > "$TROOT/fake2.out" 2>&1 &
SERVER_PIDS+=($!)
wait "$BG"
check "socket replaced by a restarted server is reached by a running sandbox -> $(tail -1 "$TROOT/restart.out")" \
  test "$(tail -1 "$TROOT/restart.out")" = "gateway-base gateway-base-2"
check "nothing written to the real home by the runs" test ! -e "$H/autoresearch" -a ! -e "$H/.session" -a ! -e "$H/.sock"

# socket path overrides + a stand-in root under /tmp (shadowed by the sandbox's private /tmp)
SOCKD=$TMPROOT/sock                                     # AF_UNIX paths are limited to 108 bytes
[ ${#SOCKD} -le 90 ] || SOCKD=$TROOT/ovr
mkdir -p "$TMPROOT/data/h2h/base/repo" "$TMPROOT/data/h2h/base/session" "$SOCKD"
echo hello > "$TMPROOT/data/h2h/base/repo/train.py"
/usr/bin/python3 -I "$TROOT/fake_uds.py" "gateway-ovr=$SOCKD/gw.sock" "runner-ovr=$SOCKD/rn.sock" \
  > "$TMPROOT/fake.out" 2>&1 &
SERVER_PIDS+=($!)
for i in $(seq 50); do grep -q ready "$TMPROOT/fake.out" 2>/dev/null && break; sleep 0.1; done
probe='import json, os, urllib.request
g = json.loads(urllib.request.urlopen("http://127.0.0.1:8100/x", timeout=5).read())["who"]
r = json.loads(urllib.request.urlopen("http://127.0.0.1:8200/x", timeout=5).read())["who"]
open("w.txt", "w").write("x")
print(g, r, open("train.py").read().strip(), sorted(os.listdir("/tmp")))'
for layout in neutral host; do
  o=$(cd "$TMPROOT/data/h2h/base/repo" && H2H_SANDBOX_LAYOUT=$layout RLTLDR_ROOT=$TMPROOT \
      H2H_GATEWAY_SOCK=$SOCKD/gw.sock H2H_RUNNER_SOCK=$SOCKD/rn.sock \
      timeout 60 "$SBX" base /usr/bin/python3 -I -c "$probe" </dev/null 2>&1)
  first=${TMPROOT#/tmp/}; first=${first%%/*}     # the only thing of /tmp the host layout shows: the root's path
  if [ "$layout" = neutral ]; then exp="gateway-ovr runner-ovr hello []"; else exp="gateway-ovr runner-ovr hello ['$first']"; fi
  check "[$layout] root under /tmp + socket overrides: forwards, repo rw, nothing else of /tmp -> $o" \
    test "$o" = "$exp" -a -f "$TMPROOT/data/h2h/base/repo/w.txt"
  rm -f "$TMPROOT/data/h2h/base/repo/w.txt"
done

# killing the sandbox from outside (as the supervisor does) takes everything inside down: SIGKILL to sudo (the
# process the caller started: h2h_sandbox.sh execs it) or to unshare
inns() {   # number of processes in a PID namespace
  sudo -n bash -c 'n=0; for p in /proc/[0-9]*; do [ "$(readlink "$p/ns/pid" 2>/dev/null)" = "$1" ] && n=$((n + 1)); done; echo $n' _ "$1"
}
for victim in sudo unshare; do
  (cd "$B/repo" && sbx base /bin/sh -c 'sleep 300 & sleep 300' </dev/null >/dev/null 2>&1) &
  BG=$!
  sleep 2
  U=$(pgrep -f "^unshare --mount --pid --net .*$TROOT/data/h2h/base/repo" | head -1)
  S=$(ps -o ppid= -p "${U:-0}" | tr -d ' ')
  INIT=$(pgrep -P "${U:-0}" | head -1)
  NS=$(sudo -n readlink "/proc/${INIT:-0}/ns/pid" 2>/dev/null)
  before=$(inns "$NS")
  if [ "$victim" = sudo ]; then K=$S; else K=$U; fi
  [ -n "$K" ] && sudo -n kill -KILL "$K"
  sleep 1
  after=$(inns "$NS")
  check "kill -9 of $victim ($K) kills the whole sandbox: $NS $before -> $after processes, unshare gone" \
    test -n "$NS" -a "$before" -ge 3 -a "$after" = 0 -a ! -e "/proc/$U"
  wait "$BG" 2>/dev/null
done
check "no process of the test sandboxes left" test -z "$(pgrep -f "$TROOT/data|$TMPROOT/data")"

# the real arm dirs, when setup.py has created them: read-only checks with the production config
if [ "${H2H_TEST_REAL:-1}" = 1 ]; then
  for arm in base v5; do
    other=$([ "$arm" = base ] && echo v5 || echo base)
    if [ ! -d "$ROOT/data/h2h/$arm/repo/.git" ]; then echo "SKIP real arm $arm: $ROOT/data/h2h/$arm/repo not set up yet"; continue; fi
    for layout in neutral host; do
      # checker source on a pipe (an inherited fd to a host file would itself be a handle into the hidden tree)
      o=$(cd "$ROOT/data/h2h/$arm/repo" && cat "$TROOT/checker.py" | env -u RLTLDR_ROOT -u H2H_CONFIG \
          H2H_SANDBOX_LAYOUT=$layout timeout 120 "$SBX" "$arm" /usr/bin/python3 -I - "$arm" "$other" "$layout" \
          "$ROOT" "$ROOT/tools" real "$H" "$RUID" "$RGID" "$REL" 2>&1)
      while IFS= read -r line; do
        case $line in PASS\ *) pass "[real $layout/$arm] ${line#PASS }" ;; FAIL\ *) fail "[real $layout/$arm] ${line#FAIL }" ;; esac
      done <<< "$o"
      grep -q '^PASS ' <<< "$o" || fail "[real $layout/$arm] checker ran: ${o:0:400}"
    done
  done
fi

echo
echo "$NPASS passed, $NFAIL failed"
for f in "${FAILED[@]}"; do echo "  FAILED: $f"; done
[ "$NFAIL" -eq 0 ]
