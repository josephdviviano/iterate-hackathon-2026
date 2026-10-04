"""LLM-synthesized hypothesis programs for ONC-AGI worlds: the ARC synthesizer on cohorts.

A program is a function ``design(feats) -> Design`` written by a model from the
world card, summary statistics of the revealed data and a seed hypothesis. It
enters the committee as one more template: admission by cross-validated log
loss, likelihood weights, the vote and the leak filter are unchanged, so a
program chooses what to fit and cannot tabulate predictions. Programs are
cached per world, so an evaluation replays them without model calls.

    uv run python -m onc.synth --store artifacts/onc/benchmark/full-access --first 120 --k 8 --model qwen --workers 16
    uv run python -m onc.evaluate --store artifacts/onc/benchmark/full-access --mode full --first 120 --members synth --k 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from onc_agi.core.schema import WorldCard
from scipy import stats

from onc.hypotheses import MAX_DRIVERS, Design, Features, Template, _subset, features_from

MODELS = {"qwen": "OPENAI_BASE_URL", "gptoss": "OPENAI_BASE_URL_GPTOSS", "devin": "DEVIN_API_KEY"}
SERVED_NAME = "llm"
CACHE = Path("artifacts/onc/synth")
ROUNDS = 3
CHECK_SECONDS = 10.0
FORBIDDEN = ("import os", "import sys", "subprocess", "open(", "exec(", "eval(", "__import__", "pathlib", "socket", "shutil", "importlib", "getattr(")
ALLOWED_MODULES = ("numpy", "scipy", "sklearn", "math", "itertools", "collections", "statistics")

# The seed is the only thing that differs between members; the order is the order of admission at small K.
SEEDS = {
    "analyst": "Find the features that cause the outcome. Use whatever analysis the data supports.",
    "direct": "The outcome has one or two direct causes among the measured features. Screen each feature and keep the strongest few.",
    "sparse": "Several features matter jointly; a sparse joint model separates causes from their correlates.",
    "confounder": "A clinical or cohort variable drives both the outcome and many measurements. Adjust for it before judging any feature.",
    "upstream": "Correlated features share a common cause that sits upstream and explains its cluster. Prefer the parent, not the children.",
    "interaction": "Two features act together: a product term carries signal that neither carries alone.",
    "block": "A co-regulated module of features acts through a shared score; list the module's members and model the loading.",
    "skeptic": "Suspect that no mechanism was planted. Only an effect that survives adjustment and holds across strata earns a driver; otherwise claim none.",
}

SYSTEM = """You write one Python hypothesis program for a biomarker discovery world.
The program must define `def design(feats) -> Design`. `feats` has:
  feats.ids: tuple of feature ids (baseline columns only; post-outcome columns are already removed)
  feats.types: tuple of data types, one per id
  feats.x: numpy array, rows = patients, columns = feats.ids, standardised, missing values imputed
  feats.y: numpy int array of binary outcomes
  feats.stratum: tuple of stratum labels, one per patient
  feats.n: number of patients; feats.column(fid): the column index of a feature id
Return Design(drivers=(...), columns=(...), products=(...), loadings={...}):
  drivers: the feature ids you claim cause the outcome, most likely first, at most MAX_DRIVERS; () claims no mechanism
  columns: feature ids whose standardised values enter a logistic regression
  products: pairs (a, b) of feature ids whose product enters the regression
  loadings: {id: weight} for one weighted score that enters the regression
The regression is fitted outside your program with cross-validation, and your program is called on each
training fold, so choose features from the data you are given. Correlates of a cause earn nothing;
near-duplicates count once. About one world in five has no mechanism, and an empty driver list is right there.
Available names: np (numpy), stats (scipy.stats), Design, MAX_DRIVERS; `import numpy`, `scipy` and `sklearn` are allowed.
No file, network or process access. Keep it under 80 lines. Reply with one ```python block and nothing else."""

_CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)


def load_env_file() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    for p in (root / ".env.committee", root / ".env"):
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    return dict(os.environ)


# ---------------------------------------------------------------------- prompt


def summary(card: WorldCard, feats: Features, top: int = 40) -> str:
    """Per-feature statistics on the revealed baseline data, and the strongly correlated pairs."""
    y = feats.y.astype(bool)
    strata = {s: feats.stratum.count(s) for s in sorted(set(feats.stratum))}
    lines = [
        f"World {card.world_id}: {feats.n} patients, outcome rate {y.mean():.2f}, strata {strata}.",
        f"{len(feats.ids)} baseline features (post-outcome features were removed). Statistics per feature, strongest first:",
        "feature  type  mean_y1-mean_y0  welch_t  p  auc",
    ]
    rows = []
    for j, fid in enumerate(feats.ids):
        col = feats.x[:, j]
        a, b = col[y], col[~y]
        if len(a) < 3 or len(b) < 3:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        auc = stats.mannwhitneyu(a, b).statistic / (len(a) * len(b))
        rows.append((abs(float(t)) if np.isfinite(t) else 0.0, f"{fid}  {feats.types[j]}  {a.mean() - b.mean():+.2f}  {float(t):+.2f}  {float(p):.2g}  {float(auc):.2f}"))
    rows.sort(key=lambda r: -r[0])
    lines += [r[1] for r in rows[:top]]
    if len(rows) > top:
        lines.append(f"... {len(rows) - top} weaker features omitted; every id in feats.ids is usable.")
    if len(feats.ids) >= 2 and feats.n >= 3:
        corr = np.nan_to_num(np.corrcoef(feats.x, rowvar=False))
        pairs = [(abs(corr[i, j]), feats.ids[i], feats.ids[j], corr[i, j]) for i in range(len(feats.ids)) for j in range(i + 1, len(feats.ids)) if abs(corr[i, j]) >= 0.6]
        pairs.sort(key=lambda p: -p[0])
        if pairs:
            lines.append("Correlated pairs (|r| >= 0.6): " + "; ".join(f"{a}~{b} r={r:+.2f}" for _, a, b, r in pairs[:20]))
    return "\n".join(lines)


def task_text(card: WorldCard, feats: Features, seed: str) -> str:
    return summary(card, feats) + f"\n\nSeed hypothesis: {SEEDS[seed]}\n\nWrite `design` now."


def extract_code(text: str) -> str | None:
    blocks = _CODE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    return text.strip() + "\n" if "def design" in text else None


# ---------------------------------------------------------------------- compile and check


def _importer(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of {name} is not allowed")
    return __import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "len", "list",
        "map", "max", "min", "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip", "print",
        "Exception", "ValueError", "KeyError", "IndexError", "ZeroDivisionError", "TypeError", "frozenset", "divmod", "pow",
    )
}
_SAFE_BUILTINS["__import__"] = _importer


def compile_program(source: str) -> Template:
    """Execute the source in a restricted namespace and return its ``design`` function."""
    bad = [tok for tok in FORBIDDEN if tok in source]
    if bad:
        raise ValueError(f"forbidden token(s) {bad}")
    ns: dict = {"np": np, "stats": stats, "Design": Design, "MAX_DRIVERS": MAX_DRIVERS, "__builtins__": _SAFE_BUILTINS}
    exec(compile(source, "<program>", "exec"), ns)
    fn = ns.get("design")
    if not callable(fn):
        raise ValueError("the program defines no `design` function")
    return fn


def _validate(d, feats: Features) -> str | None:
    if not isinstance(d, Design):
        return f"design returned {type(d).__name__}, not Design"
    ids = set(feats.ids)
    if any(not isinstance(f, str) or f not in ids for f in d.drivers):
        return f"drivers contain ids outside feats.ids: {[f for f in d.drivers if f not in ids][:5]}"
    if len(d.drivers) > MAX_DRIVERS:
        return f"{len(d.drivers)} drivers; at most {MAX_DRIVERS}"
    if any(f not in ids for f in d.columns):
        return f"columns outside feats.ids: {[f for f in d.columns if f not in ids][:5]}"
    if any(len(p) != 2 or p[0] not in ids or p[1] not in ids for p in d.products):
        return "products must be pairs of ids in feats.ids"
    if any(f not in ids for f in d.loadings) or any(not np.isfinite(float(w)) for w in d.loadings.values()):
        return "loadings must map ids in feats.ids to finite weights"
    if len(d.columns) + len(d.products) + bool(d.loadings) > 12:
        return "more than 12 design terms; keep the model small"
    return None


def check(fn: Template, feats: Features) -> str | None:
    """None when the program is admissible; otherwise the checker's report for the repair round."""
    rows = np.arange(feats.n)
    for label, sub in (("all patients", feats), ("a 60 percent training fold", _subset(feats, rows[: max(MAX_DRIVERS + 1, int(0.6 * feats.n))]))):
        t = time.time()
        try:
            d = fn(sub)
        except Exception:
            return f"design raised on {label}:\n" + "".join(traceback.format_exception_only(*traceback.sys.exc_info()[:2])).strip()
        if time.time() - t > CHECK_SECONDS:
            return f"design took {time.time() - t:.0f}s on {label}; it must run in under {CHECK_SECONDS:.0f}s"
        bad = _validate(d, sub)
        if bad:
            return f"on {label}: {bad}"
    return None


def guarded(fn: Template) -> Template:
    """A program that fails on a fold claims no mechanism there, so the fit never aborts."""

    def template(feats: Features) -> Design:
        try:
            d = fn(feats)
            return d if _validate(d, feats) is None else Design(drivers=())
        except Exception:
            return Design(drivers=())

    return template


# ---------------------------------------------------------------------- synthesis


def synthesize(client, card: WorldCard, feats: Features, seed: str, *, rounds: int = ROUNDS, temperature: float = 0.7, max_tokens: int = 3000) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_text(card, feats, seed)}]
    meta: dict = {"rounds": 0, "prompt_tokens": 0, "completion_tokens": 0, "round_log": []}
    source, ok, t0 = None, False, time.time()
    for r in range(rounds):
        try:
            resp = client.chat.completions.create(model=SERVED_NAME, messages=messages, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            meta["round_log"].append({"round": r, "error": repr(e)[:300]})
            break
        text = resp.choices[0].message.content or ""
        if resp.usage:
            meta["prompt_tokens"] += resp.usage.prompt_tokens or 0
            meta["completion_tokens"] += resp.usage.completion_tokens or 0
        meta["rounds"] = r + 1
        code = extract_code(text)
        if code is None:
            messages += [{"role": "assistant", "content": text}, {"role": "user", "content": "Reply with the full program in one ```python block."}]
            meta["round_log"].append({"round": r, "no_code": True})
            continue
        source = code
        try:
            report = check(compile_program(code), feats)
        except Exception as e:
            report = f"{type(e).__name__}: {e}"
        meta["round_log"].append({"round": r, "ok": report is None, "report": (report or "")[:300], "bytes": len(code)})
        if report is None:
            ok = True
            break
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": f"Checker report:\n\n{report[:3000]}\n\nRepair the program and reply with the full source."}]
    meta["wall_s"] = round(time.time() - t0, 1)
    return {"world_id": card.world_id, "seed": seed, "ok": ok, "source": source if ok else None, "last_source": source, "meta": meta}


# ---------------------------------------------------------------------- Devin sessions

DEVIN_API = "https://api.devin.ai/v1"
DEVIN_SCHEMA = {
    "type": "object",
    "properties": {"program": {"type": "string", "description": "Full source of the hypothesis program: `def design(feats) -> Design`"}},
    "required": ["program"],
}


def _devin_headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['DEVIN_API_KEY']}"}


def _devin_upload(client, name: str, text: str) -> str | None:
    try:
        r = client.post(f"{DEVIN_API}/attachments", headers=_devin_headers(), files={"file": (name, text.encode())}, timeout=120)
        r.raise_for_status()
        body = r.json()
        return body if isinstance(body, str) else body.get("url")
    except Exception:
        return None


def _devin_wait(client, sid: str, deadline: float, poll_s: float, nudge: str) -> tuple[str | None, dict]:
    """Poll a session until it returns a program or finishes; a blocked session is told to return its best program."""
    status, out, nudged, last = None, {}, 0, 0.0
    while time.time() < deadline:
        time.sleep(poll_s)
        try:
            g = client.get(f"{DEVIN_API}/session/{sid}", headers=_devin_headers())
        except Exception:
            continue
        if g.status_code != 200:
            continue
        body = g.json()
        status, out = body.get("status_enum") or body.get("status"), body.get("structured_output") or {}
        if out.get("program") or status in ("finished", "expired", "suspended"):
            break
        if status == "blocked" and nudged < 3 and time.time() - last > 120:
            try:
                client.post(f"{DEVIN_API}/session/{sid}/message", headers=_devin_headers(), json={"message": nudge})
            except Exception:
                pass
            nudged, last = nudged + 1, time.time()
    return status, out


def synthesize_devin(card: WorldCard, feats: Features, seed: str, *, timeout_s: float = 1200, poll_s: float = 20, max_acu: int = 2) -> dict:
    """One Devin session per program. The revealed data is an attachment; the program is structured output."""
    import httpx

    client = httpx.Client(timeout=60)
    header = ["outcome", *feats.ids]
    rows = [",".join([str(int(y)), *(f"{v:.4f}" for v in x)]) for x, y in zip(feats.x, feats.y)]
    url = _devin_upload(client, "data.csv", "\n".join([",".join(header), *rows]))
    prompt = (
        SYSTEM.replace("Reply with one ```python block and nothing else.", "")
        + "\n\n" + task_text(card, feats, seed).replace("Write `design` now.", "")
        + ("\nThe revealed data is attached as data.csv (column outcome, then one standardised column per feature id); "
           "test your program on it before returning it. " if url else "")
        + "Return the full program source through the structured output. Do not ask questions; decide and return."
    )
    body = {"prompt": prompt, "structured_output_schema": DEVIN_SCHEMA, "max_acu_limit": max_acu, "unlisted": True,
            "tags": ["onc-synth"], "title": f"onc {card.world_id} {seed}"}
    if url:
        body["attachments"] = [url]
    t0 = time.time()
    meta: dict = {"backend": "devin", "rounds": 0, "prompt_tokens": 0, "completion_tokens": 0, "round_log": [], "attachment": url is not None}
    try:
        r = client.post(f"{DEVIN_API}/sessions", headers=_devin_headers(), json=body)
        if r.status_code != 200:
            raise RuntimeError(f"create session {r.status_code}: {r.text[:300]}")
        session = r.json()
    except Exception as e:
        meta.update({"error": repr(e)[:300], "wall_s": round(time.time() - t0, 1)})
        return {"world_id": card.world_id, "seed": seed, "ok": False, "source": None, "last_source": None, "meta": meta}
    sid = session.get("session_id")
    meta["session_id"], meta["session_url"] = sid, session.get("url")
    nudge = "Do not wait for me. Return the structured output now with your best complete program."
    source, ok = None, False
    for r_ in range(2):  # the session's first program, then one repair on the checker's report
        status, out = _devin_wait(client, sid, t0 + timeout_s, poll_s, nudge)
        meta["rounds"], meta["status"] = r_ + 1, status
        code = out.get("program")
        if not code:
            meta["round_log"].append({"round": r_, "no_code": True, "status": status})
            break
        code = extract_code(code) or code
        source = code
        try:
            report = check(compile_program(code), feats)
        except Exception as e:
            report = f"{type(e).__name__}: {e}"
        meta["round_log"].append({"round": r_, "ok": report is None, "report": (report or "")[:300], "bytes": len(code)})
        if report is None:
            ok = True
            break
        if status in ("finished", "expired", "suspended") or r_ == 1:
            break
        try:
            client.post(f"{DEVIN_API}/session/{sid}/message", headers=_devin_headers(),
                        json={"message": f"Checker report:\n\n{report[:3000]}\n\nRepair the program and return the full source through the structured output again."})
        except Exception:
            break
    meta["wall_s"] = round(time.time() - t0, 1)
    return {"world_id": card.world_id, "seed": seed, "ok": ok, "source": source if ok else None, "last_source": source, "meta": meta}


def cache_path(cache: Path, model: str, world_id: str, seed: str) -> Path:
    return cache / model / world_id / f"{seed}.json"


def load_programs(world_id: str, model: str, k: int, cache: Path = CACHE, seeds: tuple[str, ...] | None = None) -> dict[str, Template]:
    """The admissible cached programs of a world, compiled, named ``synth:<seed>``; ``seeds`` overrides the first K."""
    out: dict[str, Template] = {}
    for seed in (seeds if seeds is not None else list(SEEDS)[:k]):
        p = cache_path(cache, model, world_id, seed)
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        if rec.get("ok") and rec.get("source"):
            try:
                out[f"synth:{seed}"] = guarded(compile_program(rec["source"]))
            except Exception:
                continue
    return out


def client_for(model: str):
    from openai import OpenAI

    env = load_env_file()
    if model == "devin":
        return None  # sessions, not a chat client
    return OpenAI(base_url=env[MODELS[model]].rstrip("/"), api_key=env["VLLM_API_KEY"], timeout=240, max_retries=2)


def main(argv: list[str] | None = None) -> int:
    from onc.evaluate import open_store, world_ids

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default="toy")
    parser.add_argument("--mode", default="full", choices=["full", "seq"])
    parser.add_argument("--first", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=len(SEEDS), help="seeds per world, in SEEDS order")
    parser.add_argument("--model", default="qwen", choices=sorted(MODELS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--cache", default=str(CACHE))
    args = parser.parse_args(argv)

    store, cache = open_store(args.store), Path(args.cache)
    ids = world_ids(store, args.mode, args.limit, args.first)
    client = client_for(args.model)
    worlds = {}
    for wid in ids:
        data = store.world(wid)
        worlds[wid] = (data.card, features_from(data.card, data.card.feature_ids(), data.x, data.y, data.stratum))
    todo = [(wid, s) for wid in ids for s in list(SEEDS)[:args.k] if not cache_path(cache, args.model, wid, s).exists()]
    print(f"{len(ids)} worlds, {len(todo)} programs to synthesize ({len(ids) * args.k - len(todo)} cached)", flush=True)
    t0, done, ok = time.time(), 0, 0

    def run(job):
        wid, seed = job
        card, feats = worlds[wid]
        rec = synthesize_devin(card, feats, seed) if args.model == "devin" else synthesize(client, card, feats, seed, rounds=args.rounds)
        p = cache_path(cache, args.model, wid, seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1))
        return rec

    with ThreadPoolExecutor(args.workers) as pool:
        for rec in pool.map(run, todo):
            done += 1
            ok += int(rec["ok"])
            if done % 20 == 0 or done == len(todo):
                print(f"{done}/{len(todo)} programs, {ok} admissible, {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------- the agent

from onc.agent import CommitteeAgent, Policy  # noqa: E402  (after the synthesis helpers it depends on)
from onc.committee import build_committee  # noqa: E402
from onc.hypotheses import TEMPLATES, null  # noqa: E402


class SynthCommitteeAgent(CommitteeAgent):
    """The committee over a world's cached synthesized programs; ``both`` adds the eight templates."""

    def __init__(self, policy: Policy = Policy(), *, model: str = "qwen", k: int = len(SEEDS), members: str = "synth", cache: Path = CACHE, seeds: tuple[str, ...] | None = None) -> None:
        super().__init__(policy, name=f"{members}_{'+'.join(seeds) if seeds else f'k{k}'}_{model}")
        self.model, self.k, self.members, self.cache, self.seeds = model, k, members, Path(cache), seeds
        self._card: WorldCard | None = None

    def choose_action(self, card, view):
        self._card = card
        return super().choose_action(card, view)

    def _fit(self, feats: Features):
        assert self._card is not None
        programs = load_programs(self._card.world_id, self.model, self.k, self.cache, self.seeds)
        templates: dict[str, Template] = {"null": null, **programs}
        if self.members == "both":
            templates.update(TEMPLATES)
        if not programs:
            log = self.logs.get(self._card.world_id)
            if log is not None and isinstance(getattr(log, "notes", None), list):
                log.notes.append("no admissible program")
        return build_committee(feats, weighting=self.policy.weighting, beta=self.policy.beta, templates=templates, seed=self.policy.seed)
