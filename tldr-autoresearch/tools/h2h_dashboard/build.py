#!/usr/bin/env python3
"""Build the head-to-head dashboard (base vs RLTL;DR v5): collect each arm's data into one JSON document and
inline it into tools/h2h_dashboard/template.html -> dashboard_h2h/index.html. Stdlib only (/usr/bin/python3);
read-only on the run (the one file it writes besides the page is a parse cache next to the page).

Per arm (paths from rltldr/h2h_config.py; the data root follows RLTLDR_ROOT, or --data for the h2h dir):
  * ledger.jsonl       trusted, one line per run (tools/ar_run.py): the only source of val_bpb and validity.
  * repo/results.tsv   UNTRUSTED (the agent writes it): used only for the agent's keep/discard decisions. Rows
                       are mapped onto ledger runs by git commit (ledger agent_head starts with the row's
                       commit), else by train.py content (the commit's train.py read from loose git objects
                       with plain file reads), else by the exact 6-decimal val_bpb the agent copied from the run
                       output. git is never run in an agent repo (its config is untrusted), and every read there
                       is a bounded read of a regular file (no symlinks, FIFOs or devices), so the agent cannot
                       hang or blow up the build.
  * status.json        supervisor counters (nudges, restarts, compactions, state).
  * calls.jsonl        gateway call records. They carry token ids and grow to GBs, so they are aggregated
                       incrementally: byte offset + aggregates are cached in <out dir>/.calls_cache.json.
  * events.jsonl       first and last event only (head/tail reads).
  * runner_state.json  the run in flight, if the runner reports one.

    /usr/bin/python3 tools/h2h_dashboard/build.py [--out FILE] [--data H2H_DIR]
    /usr/bin/python3 tools/h2h_dashboard/build.py --status       # plain-text summary for ctl_h2h.sh status
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import zlib

CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CODE)
from rltldr import h2h_config  # noqa: E402

try:                                    # the harness's own definition of an invalid measurement
    sys.path.insert(0, os.path.join(CODE, "tools"))
    from ar_run import TRUSTED_FATAL    # noqa: E402
    INVALID_FLAGS = set(TRUSTED_FATAL) | {"prepare_py_modified"}
except Exception:                       # pragma: no cover - keep the dashboard alive if ar_run is mid-edit
    INVALID_FLAGS = {"no_trusted_eval", "multiple_evals", "val_leak", "non_causal", "eval_unverifiable",
                     "over_time_budget", "prepare_py_modified"}

ARM_TEXT = {   # display text; anything else falls back to the served adapter name
    "base": ("base", "Qwen3.8-27B-FP8 + all-zero LoRA (exactly the base model)"),
    "v5": ("v5", "Qwen3.8-27B-FP8 + RLTL;DR v5 LoRA (unmerged)"),
}
DESCRIPTION = __doc__.split("\n")[0]
OUT_DIR = "dashboard_h2h"   # default output: <RLTLDR_ROOT>/<OUT_DIR>/index.html
TEMPLATE_SUBS = ()          # (old, new) text replacements applied to template.html (tools/frz_dashboard reuses it)
LIVE_S = 30 * 60            # an arm with no activity for this long is shown as stopped
TSV_MAX = 4 << 20           # results.tsv larger than this is ignored (never legitimate)
OBJ_MAX = 8 << 20           # loose git object size caps (compressed / inflated)
OBJ_INFLATED_MAX = 32 << 20
HEX = re.compile(r"[0-9a-f]{4,40}")
MIN_DECODE_TOKENS = 32      # calls shorter than this give noisy decode rates
RECENT_CALLS = 30


# ---------------------------------------------------------------------------------------------------------
# small readers
# ---------------------------------------------------------------------------------------------------------
def ts_of(x):
    """Epoch seconds from an epoch number (s or ms) or an ISO-8601 string; None if unparseable."""
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) / 1000.0 if x > 1e12 else float(x)
    if isinstance(x, str):
        try:
            d = dt.datetime.fromisoformat(x.strip())
        except ValueError:
            try:
                return ts_of(float(x))
            except ValueError:
                return None
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.timestamp()
    return None


def num(x):
    ok = isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    return float(x) if ok else None


def rj(path, default=None):
    try:
        with open(path) as f:
            v = json.load(f)
        return v if isinstance(v, dict) else default
    except (OSError, ValueError):
        return default


def jl(path):
    out = []
    try:
        with open(path, "rb") as f:
            for line in f:
                try:
                    v = json.loads(line)
                except ValueError:
                    continue            # a torn last line while the writer appends
                if isinstance(v, dict):
                    out.append(v)
    except OSError:
        pass
    return out


def edge_lines(path, n=65536):
    """(first complete JSON object, last complete JSON object) of a JSONL file, reading only its ends."""
    first = last = None
    try:
        with open(path, "rb") as f:
            head = f.read(n)
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - n))
            tail = f.read(n)
    except OSError:
        return None, None
    for line in head.split(b"\n")[:-1] or []:
        try:
            first = json.loads(line)
            break
        except ValueError:
            continue
    for line in reversed(tail.split(b"\n")):
        try:
            last = json.loads(line)
            break
        except ValueError:
            continue
    return (first if isinstance(first, dict) else None), (last if isinstance(last, dict) else None)


def read_untrusted(path, limit):
    """Bounded read of a file the agent controls: a regular file (no symlink, FIFO or device) of at most
    `limit` bytes, else None."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit:
            return None
        parts, n = [], 0
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            n += len(b)
            if n > limit:
                return None
            parts.append(b)
        return b"".join(parts)
    except OSError:
        return None
    finally:
        os.close(fd)


def real_dir(path):
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------------------
# results.tsv -> ledger runs
# ---------------------------------------------------------------------------------------------------------
def git_object(repo, sha):
    """(type, body) of a LOOSE object in repo/.git whose id starts with `sha`, read as plain files; None when
    it is packed, ambiguous, missing or anything looks off."""
    objs = os.path.join(repo, ".git", "objects")
    if not (len(sha) >= 4 and HEX.fullmatch(sha) and real_dir(os.path.join(repo, ".git")) and real_dir(objs)):
        return None
    sub = os.path.join(objs, sha[:2])
    if not real_dir(sub):
        return None
    try:
        names = []
        with os.scandir(sub) as it:
            for i, e in enumerate(it):
                if i > 20000:
                    return None
                if e.name.startswith(sha[2:]):
                    names.append(e.name)
    except OSError:
        return None
    if len(names) != 1:
        return None
    raw = read_untrusted(os.path.join(sub, names[0]), OBJ_MAX)
    if raw is None:
        return None
    try:
        d = zlib.decompressobj()
        data = d.decompress(raw, OBJ_INFLATED_MAX)
        if d.unconsumed_tail:
            return None
    except zlib.error:
        return None
    hdr, _, body = data.partition(b"\x00")
    kind, _, _ = hdr.partition(b" ")
    return kind.decode("ascii", "replace"), body


def commit_train_sha(repo, commit, memo):
    """sha256 of train.py in `commit` (loose objects only), or None."""
    if commit in memo:
        return memo[commit]
    out = None
    c = git_object(repo, commit)
    if c and c[0] == "commit":
        m = re.match(rb"tree ([0-9a-f]{40})\n", c[1])
        t = git_object(repo, m.group(1).decode()) if m else None
        if t and t[0] == "tree":
            body, i = t[1], 0
            while i < len(body):
                sp, nul = body.find(b" ", i), body.find(b"\x00", i)
                if sp < 0 or nul < 0 or nul + 21 > len(body):
                    break
                mode, name, oid = body[i:sp], body[sp + 1:nul], body[nul + 1:nul + 21].hex()
                i = nul + 21
                if name == b"train.py" and mode.startswith(b"100"):
                    b = git_object(repo, oid)
                    if b and b[0] == "blob":
                        out = hashlib.sha256(b[1]).hexdigest()
                    break
    memo[commit] = out
    return out


def read_tsv(repo):
    """The agent's results.tsv rows: commit, val, mem_gb, status, desc (header/garbage lines skipped)."""
    raw = read_untrusted(os.path.join(repo, "results.tsv"), TSV_MAX)
    if raw is None:
        return []
    rows = []
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        cells = line.split("\t") if "\t" in line else line.split(",", 4)   # tolerate a CSV slip
        cells = [c.strip() for c in cells]
        if len(cells) < 4 or cells[0].lower() == "commit":
            continue
        try:
            val = num(float(cells[1]))
        except ValueError:
            val = None
        rows.append({"commit": cells[0].lower()[:40], "val": val if val and val > 0 else None,
                     "status": cells[3].lower()[:20], "desc": "\t".join(cells[4:])[:300]})
    return rows


def same_val(a, b):
    return a is not None and b is not None and abs(a - b) < 1.5e-6     # results.tsv carries 6 decimals


def match_rows(rows, runs, repo):
    """Attach each results.tsv row to one ledger run (sets run dec/claimed/match); returns the frontier: the
    agent's keep rows in logging order, as trusted values."""
    memo, used = {}, set()
    cand_runs = [r for r in runs if r["kind"] != "refused"]
    frontier, n_unmatched = [], 0
    for row in rows:
        c, how, cands = row["commit"], None, []
        if HEX.fullmatch(c):
            cands = [r for r in cand_runs if r["_head"].startswith(c)]
            how = "head"
            if not cands:
                sha = commit_train_sha(repo, c, memo)
                cands = [r for r in cand_runs if sha and r["_train_sha"] == sha]
                how = "train_sha"
        if not cands and row["val"] is not None:
            cands = [r for r in cand_runs if same_val(r["val"], row["val"])]
            how = "value"
        if not cands:
            n_unmatched += 1
            row["run"] = None
            continue
        crash_row = row["status"] == "crash" or row["val"] is None
        fits = [r for r in cands if (r["val"] is None if crash_row else same_val(r["val"], row["val"]))] or cands
        free = [r for r in fits if r["i"] not in used]
        run = free[0] if free else fits[-1]
        used.add(run["i"])
        run.update(dec=row["status"], claimed=row["val"], match=how, agent_desc=row["desc"])
        row["run"] = run
        if row["status"] == "keep" and run["val"] is not None:
            frontier.append({"i": run["i"], "te": run["te"], "val": run["val"]})
    return frontier, n_unmatched


# ---------------------------------------------------------------------------------------------------------
# ledger, calls, supervisor state
# ---------------------------------------------------------------------------------------------------------
def kind_of(e):
    st = e.get("status")
    if st == "refused":
        return "refused"
    if st != "ok":
        return "crash"
    if num(e.get("val_bpb")) is None or INVALID_FLAGS.intersection(e.get("flags") or []):
        return "invalid"
    return "valid"


def err_summary(tail):
    """One line for a failed run: the exception line of the error tail if there is one, else its first line."""
    lines = [x.strip() for x in (tail or "").splitlines() if x.strip()]
    exc = [x for x in lines if re.match(r"^[\w.]*(Error|Exception|Interrupt)\b", x) or x.startswith("killed:")]
    return (exc[-1] if exc else lines[0] if lines else "")[:200]


def load_runs(ledger_path):
    runs, i = [], 0
    for e in jl(ledger_path):
        kind = kind_of(e)
        if kind != "refused":
            i += 1
        m = e.get("metrics") if isinstance(e.get("metrics"), dict) else {}
        ls = e.get("last_step") if isinstance(e.get("last_step"), dict) else {}
        ts = ts_of(e.get("ts"))
        te = ts_of(e.get("ts_end")) or ((ts + num(e.get("wall_s"))) if ts and num(e.get("wall_s")) else ts)
        err = (e.get("error_tail") or "") if kind in ("crash", "refused") else ""
        head = str(e.get("agent_head") or "").lower()
        runs.append({
            "i": i if kind != "refused" else None, "id": e.get("run_id"), "ts": ts, "te": te,
            "desc": str(e.get("desc") or "")[:300], "status": e.get("status"), "kind": kind,
            "val": num(e.get("val_bpb")) if kind == "valid" else None,
            "val_raw": num(e.get("val_bpb")), "flags": [str(f) for f in (e.get("flags") or [])][:12],
            "steps": num(m.get("num_steps")), "vram_gb": (num(m.get("peak_vram_mb")) or 0) / 1024 or None,
            "t_train": num(m.get("t_train_trusted")) or num(m.get("t_train")) or num(m.get("training_seconds")),
            "mfu": num(m.get("mfu_percent")), "tok_s": num(ls.get("tok_per_sec")),
            "params_m": num(m.get("num_params_M")),
            "head": head[:7], "err": err_summary(err),
            "dec": None, "claimed": None, "match": None, "agent_desc": None,
            "_head": head if HEX.fullmatch(head) else "", "_train_sha": e.get("train_sha256"),
            "_flags": e.get("flags") or [],
        })
    return runs


def calls_agg(path, cache):
    """Aggregate gateway call records incrementally (resumes from the cached byte offset)."""
    zero = {"n": 0, "n_err": 0, "out": 0, "inp": 0, "dec_tok": 0, "dec_s": 0.0, "ttft_sum": 0.0, "ttft_n": 0,
            "first": None, "last": None, "recent": []}
    try:
        st = os.stat(path)
    except OSError:
        return dict(zero)
    c = cache.get(path)
    if not c or c.get("ino") != st.st_ino or c.get("dev") != st.st_dev or c.get("off", 0) > st.st_size \
            or "dec_tok" not in c.get("agg", {}):
        c = {"ino": st.st_ino, "dev": st.st_dev, "off": 0, "agg": dict(zero, recent=[])}
    a = c["agg"]
    with open(path, "rb") as f:
        f.seek(c["off"])
        buf = b""
        while True:
            chunk = f.read(8 << 20)
            if not chunk:
                break
            buf += chunk
            lines = buf.split(b"\n")
            buf = lines.pop()               # incomplete last line: picked up next time
            for line in lines:
                c["off"] += len(line) + 1
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    add_call(a, r)
    cache[path] = c
    return a


def add_call(a, r):
    a["n"] += 1
    st = str(r.get("status", "ok"))
    if not (st.startswith("ok") or st == "200"):
        a["n_err"] += 1
    u = r.get("usage") if isinstance(r.get("usage"), dict) else {}
    nc = num(r.get("n_completion"))
    nc = nc if nc is not None else num(u.get("completion_tokens"))
    if nc is None and isinstance(r.get("completion_ids"), list):
        nc = len(r["completion_ids"])
    npr = num(r.get("n_prompt"))
    npr = npr if npr is not None else num(u.get("prompt_tokens"))
    if npr is None and isinstance(r.get("prompt_ids"), list):
        npr = len(r["prompt_ids"])
    a["out"] += int(nc or 0)
    a["inp"] += int(npr or 0)
    d = num(r.get("decode_tok_s"))
    if d and d > 0 and (nc or 0) >= MIN_DECODE_TOKENS:      # token-weighted: total tokens / total decode time
        a["dec_tok"] += int(nc)
        a["dec_s"] += nc / d
        a["recent"] = (a["recent"] + [[int(nc), nc / d]])[-RECENT_CALLS:]
    tt = num(r.get("ttft_s"))
    if tt is not None and tt >= 0:
        a["ttft_sum"] += tt
        a["ttft_n"] += 1
    t0 = ts_of(r.get("ts_start")) or ts_of(r.get("t0"))
    t1 = ts_of(r.get("ts_end")) or ts_of(r.get("t1")) or t0
    if t0 and (a["first"] is None or t0 < a["first"]):
        a["first"] = t0
    if t1 and (a["last"] is None or t1 > a["last"]):
        a["last"] = t1


def last_event_type(status, ev):
    """The supervisor's own last event type, else the last record of events.jsonl."""
    ev = ev or {}
    t = status.get("last_event_type") or (ev.get("event") if ev.get("type") == "supervisor" else ev.get("type"))
    return str(t)[:40] if t else None


def inflight_of(runner_state):
    """The run in flight, if runner_state.json names one (tolerant of the exact field names)."""
    if not isinstance(runner_state, dict):
        return None
    for k in ("job", "inflight", "in_flight", "current", "running", "run"):
        j = runner_state.get(k)
        if isinstance(j, dict) and (j.get("desc") is not None or j.get("run_id") or j.get("t0")):
            since = next((ts_of(j.get(x)) for x in ("t0", "since", "started", "started_at", "ts", "t_start")
                          if ts_of(j.get(x))), None)
            return {"desc": str(j.get("desc") or "")[:300], "since": since}
    return None


# ---------------------------------------------------------------------------------------------------------
# document
# ---------------------------------------------------------------------------------------------------------
def arm_doc(arm, now, cache):
    runs = load_runs(arm.ledger)
    rows = read_tsv(arm.repo)
    frontier, n_unmatched = match_rows(rows, runs, arm.repo)
    status = rj(arm.status, {}) or {}
    ev_first, ev_last = edge_lines(arm.events)
    calls = calls_agg(arm.calls, cache) if cache is not None else None
    inflight = inflight_of(rj(os.path.join(arm.dir, "runner_state.json")))

    starts = [ts_of((ev_first or {}).get("ts")), ts_of(status.get("started_at")),
              runs[0]["ts"] if runs else None, (calls or {}).get("first")]
    t0 = min((t for t in starts if t), default=None)
    ends = [ts_of((ev_last or {}).get("ts")), ts_of(status.get("last_event_ts")),
            max((r["te"] or 0 for r in runs), default=None) or None, (calls or {}).get("last"),
            inflight["since"] if inflight else None]
    last = max((t for t in ends if t), default=None)
    live = bool(last and now - last < LIVE_S)

    done = [r for r in runs if r["kind"] != "refused"]
    valid = [r for r in done if r["kind"] == "valid"]
    calib = next((r for r in valid if "empty_diff" in r["_flags"]), None) or (valid[0] if valid and
                                                                               valid[0]["i"] == 1 else None)
    keep_rows = [w for w in rows if w["status"] == "keep"]
    last_keep = keep_rows[-1] if keep_rows else None
    best_kept = frontier[-1] if frontier else None
    best_valid = min(valid, key=lambda r: r["val"]) if valid else None
    hours = ((now if live else (last or now)) - t0) / 3600 if t0 else None

    summary = {
        "n_runs": len(done), "n_valid": len(valid), "n_crash": sum(r["kind"] == "crash" for r in done),
        "n_invalid": sum(r["kind"] == "invalid" for r in done),
        "n_refused": sum(r["kind"] == "refused" for r in runs),
        "hours": hours, "runs_per_hour": len(done) / hours if hours and hours >= 0.5 else None,
        "best_kept": best_kept and {**best_kept, "desc": next(r["desc"] for r in runs if r["i"] == best_kept["i"])},
        "last_keep_unmatched": bool(last_keep and last_keep.get("run") is None),
        "last_keep_invalid": (last_keep["run"]["i"] if last_keep and last_keep.get("run")
                              and last_keep["run"]["val"] is None else None),
        "last_keep_claimed": last_keep["val"] if last_keep else None,
        "best_valid": best_valid and {"i": best_valid["i"], "val": best_valid["val"], "desc": best_valid["desc"]},
        "baseline": calib and {"i": calib["i"], "val": calib["val"]},
        "n_tsv": len(rows), "n_keep": len(keep_rows), "n_tsv_unmatched": n_unmatched,
        "n_logged_runs": sum(1 for r in done if r["dec"]),
        "nudges": num(status.get("n_nudges")), "restarts": num(status.get("n_restarts")),
        "compactions": num(status.get("n_compactions")), "prompts": num(status.get("n_prompts")),
        "runsh_calls": num(status.get("n_runsh_calls")),
        "state": str(status.get("state") or "")[:60] or None,
        "context_tokens": num(status.get("context_tokens")), "context_percent": num(status.get("context_percent")),
        "last_event": last, "last_event_type": last_event_type(status, ev_last),
        "live": live,
    }
    if calls is not None:
        summary.update(n_calls=calls["n"], n_call_err=calls["n_err"], tokens_out=calls["out"],
                       tokens_in=calls["inp"],
                       decode_tok_s=calls["dec_tok"] / calls["dec_s"] if calls["dec_s"] else None,
                       decode_recent=(sum(n for n, _ in calls["recent"]) / sum(s for _, s in calls["recent"])
                                      if calls["recent"] else None),
                       ttft_s=calls["ttft_sum"] / calls["ttft_n"] if calls["ttft_n"] else None)
    label, model = ARM_TEXT.get(arm.name, (arm.name, arm.served_model))
    return {
        "name": arm.name, "label": label, "model": model, "served_model": arm.served_model,
        "gpu": arm.gpu_minor, "t0": t0, "last": last, "summary": summary, "frontier": frontier,
        "inflight": inflight,
        "calib": calib and {k: calib[k] for k in ("i", "val", "steps", "tok_s", "mfu", "t_train", "vram_gb", "te",
                                                   "params_m")},
        "runs": [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs],
    }


def build(cfg, cache=None):
    now = time.time()
    arms = [arm_doc(a, now, cache) for a in cfg.arms]
    t0 = min((a["t0"] for a in arms if a["t0"]), default=None)
    last = max((a["last"] for a in arms if a["last"]), default=None)
    return {"generated_at": now, "t0": t0, "last": last, "live": any(a["summary"]["live"] for a in arms),
            "arms": arms}


def status_text(doc):
    now = doc["generated_at"]
    f6 = lambda x: "-" if x is None else f"{x:.6f}"                       # noqa: E731
    ago = lambda t: "-" if not t else (f"{(now - t) / 60:.0f} min ago" if now - t < 5400  # noqa: E731
                                       else f"{(now - t) / 3600:.1f} h ago")
    out = []
    for a in doc["arms"]:
        s = a["summary"]
        out.append(f"[{a['name']}] GPU {a['gpu']} · {a['served_model']} · "
                   f"{'active' if s['live'] else 'idle'} (last activity {ago(a['last'])})")
        rate = f" · {s['runs_per_hour']:.1f}/h" if s["runs_per_hour"] else ""
        out.append(f"  runs {s['n_runs']}{rate} · valid {s['n_valid']} · crash {s['n_crash']} · "
                   f"invalid {s['n_invalid']}" + (f" · refused {s['n_refused']}" if s["n_refused"] else ""))
        done = [r for r in a["runs"] if r["kind"] != "refused"]
        if done:
            r = done[-1]
            res = f6(r["val"]) if r["kind"] == "valid" else (r["kind"] if r["kind"] != "crash" else r["status"])
            out.append(f"  last run #{r['i']} ({ago(r['te'])}): {res} · agent: {r['dec'] or '-'} · {r['desc'][:90]}")
        bk, bv = s["best_kept"], s["best_valid"]
        out.append(f"  best kept {f6(bk and bk['val'])}" + (f" (#{bk['i']})" if bk else "") +
                   (" [latest keep row unmatched]" if s["last_keep_unmatched"] else "") +
                   f" · best single run {f6(bv and bv['val'])}" + (f" (#{bv['i']})" if bv else "") +
                   f" · baseline {f6(s['baseline'] and s['baseline']['val'])}")
        out.append(f"  results.tsv {s['n_tsv']} rows ({s['n_keep']} keep, {s['n_tsv_unmatched']} unmatched) · "
                   f"nudges {fmt_int(s['nudges'])} · restarts {fmt_int(s['restarts'])} · "
                   f"compactions {fmt_int(s['compactions'])} · state {s['state'] or '-'}")
        if a["inflight"]:
            out.append(f"  in flight: {a['inflight']['desc'][:90]} (since {ago(a['inflight']['since'])})")
    return "\n".join(out)


def fmt_int(x):
    return "-" if x is None else str(int(x))


def main():
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument("--data", default=None, help="h2h data dir (default: <RLTLDR_ROOT>/data/h2h)")
    ap.add_argument("--out", default=None, help=f"default: <RLTLDR_ROOT>/{OUT_DIR}/index.html")
    ap.add_argument("--status", action="store_true", help="print a plain-text per-arm summary and exit")
    a = ap.parse_args()
    if a.data:
        h2h_config.H2H = os.path.abspath(a.data)      # Arm paths are derived from this module global
    cfg = h2h_config.load_h2h_config()
    if a.status:
        print(status_text(build(cfg, cache=None)))
        return
    out = os.path.abspath(a.out or os.path.join(cfg.root, OUT_DIR, "index.html"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cache_path = os.path.join(os.path.dirname(out), ".calls_cache.json")
    cache = rj(cache_path, {}) or {}
    doc = build(cfg, cache)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")) as f:
        tpl = f.read()
    for old, new in TEMPLATE_SUBS:
        assert old in tpl, f"template text {old!r} missing"
        tpl = tpl.replace(old, new)
    # every "<" escaped: agent-written strings can never close the <script> element or open a comment
    payload = json.dumps(doc, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    html = tpl.replace("/*__DATA__*/{}", payload)
    assert html != tpl, "data placeholder missing from template"
    for path, text in ((out, html), (cache_path, json.dumps(cache))):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    parts = []
    for arm in doc["arms"]:
        s = arm["summary"]
        bk = s["best_kept"]
        parts.append(f"{arm['name']}: {s['n_runs']} runs, best kept {bk['val']:.6f}" if bk else
                     f"{arm['name']}: {s['n_runs']} runs, no keep yet")
    print(f"built {out}: " + "; ".join(parts) + f" ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
