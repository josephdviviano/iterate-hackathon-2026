"""Opt-in: a shared library of mechanisms extracted from the admitted programs.

Not part of the base committee. Two questions it lets us test:
1. Prior over mechanisms: does the number of mechanisms a member implements
   track its held-out accuracy better than source length did (R14)?
2. Sample complexity: on a new level with little data, does synthesis with
   the library available admit programs more often, faster, or with better
   held-out accuracy than synthesis without it? Conditions `lib_*` vs
   `nolib_*`, same synthesizer, same prompt otherwise.
Mechanism-level uncertainty falls out of the mapping: the share of members
that implement each mechanism.
"""

from __future__ import annotations

import json
import time

from .evaluate import load_runs
from .experiment import condition_dir, require_objects
from .loader import build_buffer, temporal_split
from .synth import CHECK_SCRIPT
from .synth_devin import API, _headers, _upload

SCHEMA = {
    "type": "object",
    "properties": {
        "library": {"type": "string", "description": "Full source of library.py: named mechanism functions, standard library only"},
        "mechanisms": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "description": {"type": "string"},
            "object_types": {"type": "array", "items": {"type": "string"}},
            "actions": {"type": "array", "items": {"type": "string"}}}, "required": ["name", "description"]}},
        "program_mechanisms": {"type": "object", "description": "program index as string -> list of mechanism names it implements",
                               "additionalProperties": {"type": "array", "items": {"type": "string"}}},
        "refactor_check": {"type": "string", "description": "For the programs you refactored through the library: checker first lines and line counts before/after"},
        "notes": {"type": "string"},
    },
    "required": ["library", "mechanisms", "program_mechanisms"],
}

TASK = """# Task: extract the shared mechanisms of several programs that model the same game

You get {k} Python programs. Each defines `transition_function(state, action)` for one
level of an unknown grid game, and each reproduces every observed transition in
`buffer.json` exactly (`python3 check.py` with the program copied to `program.py`).
They were written independently, so they describe the same mechanics in different
ways and may disagree about mechanics the data does not settle.

Do this:

1. Read all programs. Identify the distinct MECHANISMS they implement: a mechanism is
   one rule of the game, such as "a piece moves one cell in the action's direction
   unless a wall blocks it", "the mirror axis reflects every piece", "a counter
   decrements per move". Name each mechanism and describe it in one sentence.
2. Write `library.py`: one pure function per mechanism, standard library only, with a
   docstring, so that a program for this game or a later level could be written as
   glue over these functions. Where two programs implement the same mechanism
   differently, implement the version the observed transitions support, and note the
   alternative in the docstring.
3. Map each program index to the list of mechanism names it implements.
4. Refactor at least two of the programs to call `library.py` (same directory, `from
   library import ...`), copy each to `program.py` and run `python3 check.py` to
   confirm ALL PASS. Report the checker's first line and the line counts before and
   after for each refactored program.
5. Return the structured output. Do not ask questions; decide and proceed.

Files: program_0.py ... program_{kmax}.py, buffer.json, check.py.
"""


def extract(game: str, level: int, train_frac: float, condition: str, test_level: int | None = None,
            timeout_s: float = 2400, poll_s: float = 20) -> dict:
    import httpx

    from .synth_api import load_env_file

    load_env_file()
    transitions = build_buffer(game)
    train, test = temporal_split(transitions, level, train_frac, test_level)
    cond = condition_dir(game, level, train_frac, condition, test_level)
    require_objects(cond, "the mechanism library")
    runs = [(n, s, m) for n, m, s, _ in load_runs(cond, train, test) if m["consistent"]]
    client = httpx.Client(timeout=60)
    urls = []
    for i, (_, src, _) in enumerate(runs):
        urls.append((f"program_{i}.py", _upload(client, f"program_{i}.py", src)))
    rows = [{"step": t.step, "action": t.action, "before": t.before_objs, "after": t.after_objs} for t in train]
    urls.append(("buffer.json", _upload(client, "buffer.json", json.dumps(rows))))
    urls.append(("check.py", _upload(client, "check.py", CHECK_SCRIPT)))
    if any(u is None for _, u in urls):
        raise RuntimeError("attachment upload failed; check DEVIN_API_KEY")
    prompt = TASK.format(k=len(runs), kmax=len(runs) - 1) + "\n# Workspace\n\nDownload these files into one directory:\n" + \
        "\n".join(f"- `{n}`: {u}" for n, u in urls) + "\n"
    body = {"prompt": prompt, "structured_output_schema": SCHEMA, "max_acu_limit": 6, "unlisted": True,
            "tags": ["committee", "library"], "title": f"library {game} L{level}"}
    r = client.post(f"{API}/sessions", headers=_headers(), json=body)
    if r.status_code != 200:
        raise RuntimeError(f"Devin create session {r.status_code}: {r.text[:300]}")
    sid = r.json()["session_id"]
    t0 = time.time()
    out, status, nudged, last_nudge = {}, None, 0, 0.0
    while time.time() - t0 < timeout_s:
        time.sleep(poll_s)
        g = client.get(f"{API}/session/{sid}", headers=_headers())
        if g.status_code != 200:
            continue
        s = g.json()
        status = s.get("status_enum") or s.get("status")
        out = s.get("structured_output") or {}
        if status in ("finished", "expired") or (status == "blocked" and out.get("library")):
            break
        reported = any("ALL PASS" in (m.get("message") or "") for m in (s.get("messages") or []) if m.get("type") == "devin_message")
        if not out.get("library") and (status == "blocked" or reported) and nudged < 4 and time.time() - last_nudge > 150:
            client.post(f"{API}/session/{sid}/message", headers=_headers(),
                        json={"message": "Do not wait for me. Return the structured output now: library source, mechanisms, "
                                         "the program-to-mechanism map and the refactor check."})
            nudged, last_nudge = nudged + 1, time.time()
    lib_dir = cond / "library"
    lib_dir.mkdir(exist_ok=True)
    (lib_dir / "library.py").write_text(out.get("library") or "")
    meta = {"session_id": sid, "status": status, "wall_s": round(time.time() - t0, 1), "nudges": nudged,
            "members": [n for n, _, _ in runs], "mechanisms": out.get("mechanisms"),
            "program_mechanisms": out.get("program_mechanisms"), "refactor_check": out.get("refactor_check"),
            "notes": out.get("notes")}
    (lib_dir / "mechanisms.json").write_text(json.dumps(meta, indent=1))
    return meta


def mechanism_report(game: str, level: int, train_frac: float, condition: str, test_level: int | None = None) -> dict:
    """Mechanism consensus across members, and mechanism count against held-out accuracy."""
    cond = condition_dir(game, level, train_frac, condition, test_level)
    meta = json.loads((cond / "library" / "mechanisms.json").read_text())
    train, test = temporal_split(build_buffer(game), level, train_frac, test_level)
    runs = [(n, m) for n, m, _, _ in load_runs(cond, train, test) if m["consistent"]]
    pm = meta.get("program_mechanisms") or {}
    names = [m["name"] for m in (meta.get("mechanisms") or [])]
    k = len(runs)
    consensus = {name: round(sum(1 for i in range(k) if name in (pm.get(str(i)) or [])) / max(1, k), 3) for name in names}
    counts = [len(pm.get(str(i)) or []) for i in range(k)]
    accs = [m["test_accuracy"] for _, m in runs]

    def spearman(x, y):
        def rk(v):
            order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for t in range(i, j + 1):
                    r[order[t]] = (i + j) / 2
                i = j + 1
            return r
        rx, ry = rk(x), rk(y); n = len(x); mx, my = sum(rx) / n, sum(ry) / n
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return round(num / den, 3) if den else None

    report = {"n_mechanisms": len(names), "consensus": consensus,
              "contested": [n for n, c in consensus.items() if 0 < c < 1],
              "mechanism_count_per_member": counts, "held_out_accuracy_per_member": accs,
              "spearman_count_vs_accuracy": spearman(counts, accs) if len(set(counts)) > 1 else None}
    (cond / "library" / "mechanism_report.json").write_text(json.dumps(report, indent=1))
    return report


def library_seed(game: str, level: int, train_frac: float, condition: str, test_level: int | None = None) -> str:
    cond = condition_dir(game, level, train_frac, condition, test_level)
    lib = (cond / "library" / "library.py").read_text()
    mech = json.loads((cond / "library" / "mechanisms.json").read_text()).get("mechanisms") or []
    lines = ["A library of mechanisms extracted from programs that modelled an earlier level of this game. "
             "Use the functions that apply, adapt or ignore the rest, and keep program.py self-contained "
             "(copy what you use; no imports of this file).", "", "Mechanisms:"]
    lines += [f"- {m['name']}: {m.get('description', '')}" for m in mech]
    lines += ["", "```python", lib.strip(), "```"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract a shared mechanism library from a committee (opt-in).")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--condition", default="committee_devin")
    parser.add_argument("--test-level", type=int, default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.report_only:
        meta = extract(args.game, args.level, args.train_frac, args.condition, args.test_level)
        print({k: v for k, v in meta.items() if k not in ("mechanisms", "program_mechanisms")})
    print(json.dumps(mechanism_report(args.game, args.level, args.train_frac, args.condition, args.test_level), indent=1))


if __name__ == "__main__":
    main()
