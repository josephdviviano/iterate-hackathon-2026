#!/usr/bin/env python3
"""
ar_parse.py - parser for autoresearch run logs (train.py stdout+stderr).

parse_run_log(text) -> dict:
  status:     "ok" | "crash" | "fail_loss" | "oom" | "no_kernel"
              ("timeout" / "refused" are decided by ar_run.py, not from the log)
  metrics:    the final summary block {val_bpb, training_seconds, total_seconds, peak_vram_mb,
              mfu_percent, total_tokens_M, num_steps, num_params_M, depth, ...}; {} unless status == "ok"
  last_step:  the last "step NNNNN (...)" progress record, or None
  error_tail: sanitized excerpt (<= 4 KB) of the end of the log for non-ok statuses, else None

train.py prints its progress line with '\r' and end="" - a 5-minute log is ~64 KB of '\r'-separated
progress on a handful of physical lines. Everything here first splits on '\r', and the error tail
drops the progress records entirely so they never reach the agent's context.

CLI (debugging):  python3 ar_parse.py run.log [...]
"""
import json
import math
import re
import sys

SUMMARY_TYPES = {
    "val_bpb": float, "training_seconds": float, "total_seconds": float,
    "peak_vram_mb": float, "mfu_percent": float, "total_tokens_M": float,
    "num_steps": int, "num_params_M": float, "depth": int,
}
REQUIRED = ("val_bpb", "training_seconds", "num_steps", "peak_vram_mb")
KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s+(-?(?:[0-9.]+(?:[eE][-+]?\d+)?|nan|inf))\s*$")
STEP_RE = re.compile(
    r"step (\d+) \(([\d.]+)%\) \| loss: (\S+) \| lrm: ([\d.\-]+) \| dt: (\d+)ms \| "
    r"tok/sec: ([\d,]+) \| mfu: ([\d.]+)% \| epoch: (\d+) \| remaining: (\d+)s")
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")   # all control chars except \t and \n

MAX_TAIL_LINES = 40
MAX_TAIL_BYTES = 4096
MAX_LINE_CHARS = 400


def _normalize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return CTRL_RE.sub("", ANSI_RE.sub("", text))


def _float_or_none(s):
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def format_last_step(ls):
    if not ls:
        return None
    loss = "nan" if ls["loss"] is None else f"{ls['loss']:.4f}"
    return (f"last progress: step {ls['step']} ({ls['pct']:.1f}% of budget) loss {loss} "
            f"dt {ls['dt_ms']}ms tok/sec {ls['tok_per_sec']:,}")


LIB_FRAME_RE = re.compile(r'^\s*File ".*/(site-packages|lib/python3\.\d+)/')


def _collapse_library_frames(lines):
    """Replace runs of traceback frames inside site-packages / the stdlib (a torch.compile OOM has
    ~40 of them) by one marker line, so the agent sees its own train.py frames and the exception."""
    out, i, skipped = [], 0, 0
    while i < len(lines):
        if LIB_FRAME_RE.match(lines[i]):
            skipped += 1
            i += 1
            # the frame's source line (indented deeper than "File"), if present
            if i < len(lines) and lines[i].startswith("    ") and not lines[i].lstrip().startswith("File "):
                i += 1
            continue
        if skipped:
            out.append(f"  [... {skipped} library frame(s) omitted ...]")
            skipped = 0
        out.append(lines[i])
        i += 1
    if skipped:
        out.append(f"  [... {skipped} library frame(s) omitted ...]")
    return out


def sanitize_tail(lines, max_lines=MAX_TAIL_LINES, max_bytes=MAX_TAIL_BYTES):
    """Last `max_lines` non-progress lines, each <= MAX_LINE_CHARS, total <= max_bytes UTF-8 bytes,
    with library traceback frames collapsed. Keeps the agent's context small and limits the
    indirect prompt-injection surface (karpathy/autoresearch#64)."""
    keep = []
    for l in _collapse_library_frames(lines):
        s = l.rstrip()
        m = STEP_RE.search(s)
        if m:
            # progress is printed with end="", so whatever is printed next (a "Traceback ..."
            # header, "FAIL") is glued to the last progress record: keep only that remainder.
            s = s[m.end():].strip()
        if not s.strip() or s.lstrip().startswith("step "):
            continue
        if len(s) > MAX_LINE_CHARS:
            s = s[:MAX_LINE_CHARS] + " [...]"
        keep.append(s)
    tail = "\n".join(keep[-max_lines:])
    b = tail.encode("utf-8")
    if len(b) > max_bytes:
        tail = b[-max_bytes:].decode("utf-8", "ignore")
        tail = tail.split("\n", 1)[-1] if "\n" in tail else tail   # start on a line boundary
    return tail


def parse_run_log(text):
    lines = _normalize(text).split("\n")
    out = {"status": None, "metrics": {}, "last_step": None, "error_tail": None}

    for l in lines:
        m = STEP_RE.search(l)
        if m:
            out["last_step"] = {"step": int(m[1]), "pct": float(m[2]), "loss": _float_or_none(m[3]),
                                "dt_ms": int(m[5]), "tok_per_sec": int(m[6].replace(",", "")),
                                "epoch": int(m[8]), "remaining_s": int(m[9])}

    # Summary block = "key: value" lines after the LAST line that is exactly '---'.
    sep = max((i for i, l in enumerate(lines) if l.strip() == "---"), default=None)
    if sep is not None:
        for l in lines[sep + 1:]:
            m = KV_RE.match(l.strip())
            if not m:
                continue
            try:
                v = float(m[2])
                out["metrics"][m[1]] = SUMMARY_TYPES[m[1]](v) if m[1] in SUMMARY_TYPES and math.isfinite(v) else v
            except (ValueError, OverflowError):
                pass

    met = out["metrics"]
    if all(k in met for k in REQUIRED) and math.isfinite(met["val_bpb"]) and met["val_bpb"] > 0:
        out["status"] = "ok"
        return out

    joined = "\n".join(lines)
    if re.search(r"(^|\s)FAIL\s*$", joined, re.M):
        out["status"] = "fail_loss"          # train.py fast-fail: NaN loss or loss > 100
    elif "OutOfMemoryError" in joined or "CUDA out of memory" in joined or "out of memory" in joined.lower():
        out["status"] = "oom"
    elif "no kernel image is available" in joined:
        out["status"] = "no_kernel"          # e.g. a Hopper-only kernel on sm_120
    else:
        out["status"] = "crash"
    out["metrics"] = {}
    out["error_tail"] = sanitize_tail(lines)
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        with open(p, errors="replace") as f:
            r = parse_run_log(f.read())
        print(p, json.dumps({k: v for k, v in r.items() if k != "error_tail"}))
        if r["error_tail"]:
            print("  error_tail (last 300 chars):", r["error_tail"][-300:].replace("\n", " | "))
