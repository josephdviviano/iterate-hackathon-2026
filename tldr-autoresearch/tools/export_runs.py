#!/usr/bin/env python3
"""Export the raw experiment records of our runs into runs/ (stdlib only, reads $RLTLDR_ROOT/data read-only).

  RLTLDR_ROOT=<live root> python3 tools/export_runs.py [--out runs]

What is exported per run (see runs/README.md):
  * the trusted ledger (one line per training run), results tables and driver/trainer metrics,
  * per experiment: the task prompt, the rollout record (verdict, insight, insights in context) and the pi
    agent's session transcript,
  * per training run: the agent's train.py, its diff against the parent, the training log (carriage-return
    progress collapsed) and the signed evaluation record.
Left out: LLM call records with token ids/logprobs (tens of MB per run, used only for training), pi event
streams (duplicate the transcripts), adapters/checkpoints, repos, venvs, caches, and the per-run HMAC keys.
Every text file is rewritten so it holds no absolute paths: the data root becomes $RLTLDR_ROOT, the home
directory ~, and GPU UUIDs become GPU<n>-UUID.
"""
import argparse
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
ROOT = os.environ.get("RLTLDR_ROOT") or CODE
DATA = os.path.join(ROOT, "data")
HOME = os.path.expanduser("~")
GPU_RE = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def gpu_names():
    """Map this machine's GPU UUIDs to GPU<index>-UUID (nvidia-smi order), if nvidia-smi is available."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return {u.strip(): f"GPU{i.strip()}-UUID" for i, u in (l.split(",") for l in out.strip().splitlines() if "," in l)}


GPUS = gpu_names()


def clean(text: str) -> str:
    text = text.replace(os.path.abspath(ROOT), "$RLTLDR_ROOT").replace(HOME, "~")
    text = re.sub(r"/tmp/claude-[^\s\"']*", "$TMP", text)
    return GPU_RE.sub(lambda m: GPUS.get(m.group(0), "GPU-UUID"), text)


def collapse_cr(text: str) -> str:
    """Training logs redraw a progress line with \\r: keep only the last state of each line."""
    return "\n".join(line.rsplit("\r", 1)[-1] for line in text.split("\n"))


def copy_text(src, dst, log=False):
    if not os.path.isfile(src) or os.path.islink(src):
        return 0
    with open(src, encoding="utf-8", errors="replace") as f:
        text = f.read()
    if log:
        text = collapse_cr(text)
    text = clean(text)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)
    return 1


def export_runs_dir(src_runs, dst_runs):
    n = 0
    if not os.path.isdir(src_runs):
        return 0
    for rid in sorted(os.listdir(src_runs)):
        s, d = os.path.join(src_runs, rid), os.path.join(dst_runs, rid)
        for name in ("train.py", "diff.patch"):
            copy_text(os.path.join(s, name), os.path.join(d, name))
        copy_text(os.path.join(s, "run.log"), os.path.join(d, "run.log"), log=True)
        copy_text(os.path.join(s, "trusted", "record.jsonl"), os.path.join(d, "trusted_record.jsonl"))
        n += 1
    return n


def export_sessions(src_dir, dst_dir):
    n = 0
    if os.path.isdir(src_dir):
        for name in sorted(os.listdir(src_dir)):
            if name.endswith(".jsonl"):
                n += copy_text(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    return n


def export_attempts(src_att, dst_att):
    n = 0
    if not os.path.isdir(src_att):
        return 0
    for aid in sorted(os.listdir(src_att)):
        s, d = os.path.join(src_att, aid), os.path.join(dst_att, aid)
        if not os.path.isfile(os.path.join(s, "rollout.json")):
            continue                        # an attempt interrupted when the run was stopped
        for name in ("task.md", "rollout.json"):
            copy_text(os.path.join(s, name), os.path.join(d, name))
        export_sessions(os.path.join(s, "session"), os.path.join(d, "session"))
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(CODE, "runs"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    for sub in ("rl_oct3", "h2h", "frz"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)

    # R1: online RLTL;DR run
    rl = os.path.join(out, "rl_oct3")
    for name in ("ledger.jsonl", "calib_ledger.jsonl", "results.tsv", "metrics_driver.jsonl", "metrics_trainer.jsonl",
                 "insight_blocklist.json"):
        copy_text(os.path.join(DATA, name), os.path.join(rl, name))
    na = export_attempts(os.path.join(DATA, "attempts"), os.path.join(rl, "attempts"))
    nr = export_runs_dir(os.path.join(DATA, "runs"), os.path.join(rl, "runs"))
    print(f"rl_oct3: {na} experiments, {nr} training runs")

    # R2: head-to-head, continuous sessions
    for arm in ("base", "v5"):
        s, d = os.path.join(DATA, "h2h", arm), os.path.join(out, "h2h", arm)
        for name in ("ledger.jsonl", "status.json"):
            copy_text(os.path.join(s, name), os.path.join(d, name))
        copy_text(os.path.join(s, "repo", "results.tsv"), os.path.join(d, "agent_results.tsv"))
        ns = export_sessions(os.path.join(s, "session", "session"), os.path.join(d, "session"))
        nr = export_runs_dir(os.path.join(s, "runs"), os.path.join(d, "runs"))
        print(f"h2h/{arm}: {nr} training runs, {ns} session transcript(s)")

    # R3: frozen v5 in the training harness, insights on (x1) / off (x2)
    for arm in ("x1", "x2"):
        s, d = os.path.join(DATA, "h2h", arm, "rl"), os.path.join(out, "frz", arm)
        for name in ("ledger.jsonl", "calib_ledger.jsonl", "results.tsv", "metrics_driver.jsonl"):
            copy_text(os.path.join(s, name), os.path.join(d, name))
        na = export_attempts(os.path.join(s, "attempts"), os.path.join(d, "attempts"))
        nr = export_runs_dir(os.path.join(s, "runs"), os.path.join(d, "runs"))
        print(f"frz/{arm}: {na} experiments, {nr} training runs")


if __name__ == "__main__":
    main()
