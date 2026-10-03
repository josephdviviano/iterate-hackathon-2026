"""An LLM tool-calling agent for ONC-AGI: the baseline agent of the open track.

    uv run python -m onc.llm_agent --model qwen --store toy --workers 4
    uv run python -m onc.llm_agent --model gptoss --store artifacts/onc/dev --limit 40

One model call per turn. The model sees the world card, acts through the
arena's tools (recruit, assay, submit) and two analysis tools (a statistics
summary and a Python cell over the revealed data), and states P(signal) and
P(driver) with its submission. Tool calls are one JSON object per reply, so
the loop needs no provider-side tool parser. The model is served by vLLM on
Modal through the OpenAI chat API (credentials in ``.env.committee``).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from onc_agi.core.schema import Action, Assay, Mode, Recruit, Submit, Timing, WorldCard
from onc_agi.services import scoring
from onc_agi.services.engine import Episode, EpisodeView
from onc_agi.services.kit import Agent, run_episode
from openai import OpenAI
from scipy import stats

from onc.agent import EpisodeLog, Policy
from onc.evaluate import open_store, summarise, table, world_ids

MODELS = {"qwen": "OPENAI_BASE_URL", "gptoss": "OPENAI_BASE_URL_GPTOSS"}
MAX_TURNS = 14
MAX_FORMAT_ERRORS = 3
PYTHON_SECONDS = 60
OUTPUT_CHARS = 3000

SYSTEM = """You are a biomarker discovery analyst playing one world of the ONC-AGI benchmark.
A world is a cohort of patients with a binary outcome and measured features. A hidden mechanism was planted, or none.
Your job: return the features that DRIVE the outcome, most likely first, or an empty list if nothing can be found.

Scoring facts you must respect:
- Only causes earn credit; correlates of a cause do not. Near-duplicate features count once.
- Features with timing post_outcome were measured after the outcome. Listing one anywhere scores the world 0. Never list them.
- About one world in five has no mechanism. Abstaining there (empty list) earns credit; abstaining on a world with a mechanism costs credit. Padding the list with guesses earns nothing.
- Sequential mode: you pay to recruit patients (reveals their outcome) and to assay features on recruited patients. Credit is scaled down if you spend more than a well-designed study would, so recruit in batches and stop when the answer is clear.

Tools, one call per reply, as the LAST line of your reply, a single JSON object:
  {"tool": "recruit", "args": {"count": 40, "stratum": "all"}}      sequential only; reveals outcomes
  {"tool": "assay", "args": {"feature_ids": ["f1", "f2"]}}          sequential only; measures features on every recruited patient
  {"tool": "summary", "args": {}}                                    per-feature statistics on the revealed data
  {"tool": "python", "args": {"code": "print(data.describe())"}}     runs Python in a fresh process; the revealed cohort is preloaded as `data` (a pandas DataFrame, column `outcome` plus one column per measured feature, named by feature id) and is also the file ./data.csv; numpy, pandas, scipy and sklearn are available; print what you need
  {"tool": "submit", "args": {"ranking": ["f1"], "p_signal": 0.9, "p_driver": {"f1": 0.8}}}   ends the world
p_signal is your probability that the world has any mechanism. p_driver gives, for each listed feature, your probability that it is a true driver. Be honest: these are scored by the Brier score.
Exactly one tool call per reply; a reply with several calls runs only the first. Think briefly before the JSON. Keep replies short.
In python, `data` is already loaded with the revealed cohort; never construct or simulate data."""


def task_card(card: WorldCard) -> str:
    lines = [f"World {card.world_id}, mode {card.mode.value}, {card.n_pool} patients in the pool."]
    if card.mode is Mode.SEQUENTIAL:
        lines.append(f"Budget {card.budget:.0f} USD; recruit {card.prices.recruit_per_patient:.0f} USD per patient; strata {list(card.strata)} with sizes {card.stratum_sizes}.")
        lines.append("Nothing is revealed yet. Recruit, then assay, then analyse, then submit.")
    else:
        lines.append("Every patient and feature is revealed. Analyse, then submit.")
    lines.append("Features (id, type, timing, assay price per patient):")
    lines += [f"  {f.feature_id}  {f.data_type}  {f.timing.value}  {f.assay_price:g}" for f in card.features]
    return "\n".join(lines)


def summary(card: WorldCard, view: EpisodeView) -> str:
    """Per-feature statistics on the revealed data, and the strongly correlated pairs."""
    if not view.rows:
        return "No patients revealed yet."
    y = view.outcome.astype(bool)
    timing = {f.feature_id: f.timing.value for f in card.features}
    dtype = {f.feature_id: f.data_type for f in card.features}
    lines = [f"{len(view.rows)} patients revealed, outcome rate {y.mean():.2f}.", "feature  type  timing  n  mean_y1-mean_y0  welch_t  p  auc"]
    stats_rows = []
    for j, fid in enumerate(view.feature_ids):
        col = view.x[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 6 or y[ok].all() or not y[ok].any():
            continue
        a, b = col[ok & y], col[ok & ~y]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        ranks = stats.rankdata(col[ok])
        auc = (ranks[y[ok]].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
        stats_rows.append((fid, dtype[fid], timing[fid], int(ok.sum()), float(a.mean() - b.mean()), float(t), float(p), float(auc)))
    stats_rows.sort(key=lambda r: r[6])
    lines += [f"{r[0]}  {r[1]}  {r[2]}  {r[3]}  {r[4]:+.2f}  {r[5]:+.2f}  {r[6]:.2e}  {r[7]:.2f}" for r in stats_rows]
    measured = [j for j, m in enumerate(view.measured) if m]
    if len(measured) >= 2 and len(view.rows) >= 10:
        x = view.x[:, measured]
        x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
        corr = np.corrcoef(x, rowvar=False)
        pairs = [(abs(corr[a, b]), view.feature_ids[measured[a]], view.feature_ids[measured[b]]) for a in range(len(measured)) for b in range(a + 1, len(measured)) if abs(corr[a, b]) >= 0.5]
        if pairs:
            lines.append("Pairs with |r| >= 0.5: " + ", ".join(f"{f}~{g} ({r:.2f})" for r, f, g in sorted(pairs, reverse=True)[:12]))
    return "\n".join(lines)


def run_python(code: str, card: WorldCard, view: EpisodeView, workdir: Path) -> str:
    """Run the model's code in a fresh interpreter over the revealed data; return its output."""
    import pandas as pd

    frame = pd.DataFrame(view.x, columns=list(view.feature_ids))
    frame = frame[[f for j, f in enumerate(view.feature_ids) if view.measured[j]]]
    frame.insert(0, "outcome", view.outcome)
    # One directory per world; the cohort is both the preloaded `data` and ./data.csv, so either habit works.
    cwd = workdir / card.world_id
    cwd.mkdir(exist_ok=True)
    frame.to_csv(cwd / "data.csv", index=False)
    prelude = "import numpy as np, pandas as pd, scipy, sklearn\nfrom scipy import stats\ndata = pd.read_csv('data.csv')\n"
    try:
        proc = subprocess.run([sys.executable, "-c", prelude + code], capture_output=True, text=True, timeout=PYTHON_SECONDS, cwd=cwd)
        err = proc.stderr.strip().splitlines()[-3:] if proc.returncode else []
        out = (proc.stdout + ("\nerror: " + " | ".join(err) if err else "")).strip()
    except subprocess.TimeoutExpired:
        out = f"timeout after {PYTHON_SECONDS} s"
    return out[:OUTPUT_CHARS] or "(no output)"


def parse_call(reply: str) -> tuple[str, dict] | None:
    """The first JSON object in the reply that has a 'tool' key; one call per turn, the rest is ignored."""
    for m in re.finditer(r'\{\s*"tool"', reply):
        depth = 0
        for k in range(m.start(), len(reply)):
            if reply[k] == "{":
                depth += 1
            elif reply[k] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(reply[m.start() : k + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict) and isinstance(obj.get("tool"), str):
                        return obj["tool"], obj.get("args") or {}
                    break
    return None


def read_env(path: str = ".env.committee") -> dict[str, str]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"')
    return out


class LLMAgent(Agent):
    def __init__(self, client: OpenAI, model: str, key: str, logs: dict[str, EpisodeLog] | None = None, *, temperature: float = 0.0, max_turns: int = MAX_TURNS) -> None:
        super().__init__()
        self.name = f"llm_{key}"
        self.client, self.model, self.temperature, self.max_turns = client, model, temperature, max_turns
        self.logs = {} if logs is None else logs
        self.workdir = Path(tempfile.mkdtemp(prefix="onc-llm-"))
        self._messages: list[dict] = []
        self._world: str | None = None
        self._turns = 0
        self._format_errors = 0
        self._pending: str | None = None

    @classmethod
    def from_env(cls, key: str, logs: dict[str, EpisodeLog] | None = None, **kw) -> LLMAgent:
        env = read_env()
        base = env[MODELS[key]].rstrip("/")
        client = OpenAI(base_url=base, api_key=env["VLLM_API_KEY"], timeout=240, max_retries=2)
        model = client.models.list().data[0].id
        return cls(client, model, key, logs, **kw)

    # ------------------------------------------------------------------ the loop

    def _ask(self) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=self._messages, temperature=self.temperature, max_tokens=1500)
        text = resp.choices[0].message.content or ""
        log = self.logs[self._world]
        log.notes["tokens"] = log.notes.get("tokens", 0) + (resp.usage.total_tokens if resp.usage else 0)
        return text

    def _start(self, card: WorldCard) -> None:
        self._world = card.world_id
        self.logs[card.world_id] = EpisodeLog(card.world_id, card.mode.value, notes={"turns": 0, "tokens": 0, "python_calls": 0, "errors": []})
        self._messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task_card(card)}]
        self._turns, self._format_errors, self._pending = 0, 0, None

    def _tool_result(self, text: str) -> None:
        self._messages.append({"role": "user", "content": f"Tool result:\n{text}"})

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        if self._world != card.world_id or view.step == 0 and self._turns == 0:
            self._start(card)
        log = self.logs[card.world_id]
        if self._pending is not None:  # the engine applied the last action; report what it revealed
            self._tool_result(f"{self._pending}: {len(view.rows)} patients revealed, {sum(view.measured)} features measured, spent {view.spent:.0f} of {view.budget:.0f} USD.")
            self._pending = None
        while True:
            if self._turns >= self.max_turns:
                log.notes["errors"].append("turn limit")
                return self._submit(card, view, [], 0.5, {})
            self._turns += 1
            log.notes["turns"] = self._turns
            try:
                reply = self._ask()
            except Exception as exc:  # the model server failed: abstain and record it
                log.notes["errors"].append(f"model: {type(exc).__name__}: {str(exc)[:120]}")
                return self._submit(card, view, [], 0.5, {})
            self._messages.append({"role": "assistant", "content": reply})
            call = parse_call(reply)
            if call is None:
                self._format_errors += 1
                if self._format_errors > MAX_FORMAT_ERRORS:
                    log.notes["errors"].append("format")
                    return self._submit(card, view, [], 0.5, {})
                self._tool_result('Your reply had no tool call. End your reply with one JSON object such as {"tool": "summary", "args": {}}.')
                continue
            tool, args = call
            if tool == "summary":
                self._tool_result(summary(card, view))
            elif tool == "python":
                log.notes["python_calls"] += 1
                self._tool_result(run_python(str(args.get("code", "")), card, view, self.workdir))
            elif tool == "recruit" and card.mode is Mode.SEQUENTIAL:
                sizes = card.stratum_sizes or {s: card.n_pool for s in card.strata}
                stratum = args.get("stratum") if args.get("stratum") in card.strata else card.strata[0]
                left = sizes[stratum] - view.stratum.count(stratum)
                if left <= 0:
                    self._tool_result(f"stratum {stratum} is exhausted; strata left: {[s for s in card.strata if sizes[s] - view.stratum.count(s) > 0]}")
                    continue
                count = max(1, min(int(args.get("count", 40)), left))
                if view.spent + count * card.prices.recruit_per_patient > card.budget:
                    self._tool_result("over budget: recruit fewer patients or submit")
                    continue
                self._pending = f"recruit {count} from {stratum}"
                return Recruit(request_id=self.request_id(), count=count, stratum=stratum)
            elif tool == "assay" and card.mode is Mode.SEQUENTIAL:
                known = set(card.feature_ids())
                fids = tuple(dict.fromkeys(f for f in args.get("feature_ids", []) if f in known)) or tuple(f for f in card.feature_ids() if card.features[card.feature_ids().index(f)].timing is Timing.BASELINE)
                if not view.rows:
                    self._tool_result("no patients recruited yet; recruit first")
                    continue
                price = sum(card.features[card.feature_ids().index(f)].assay_price for f in fids) * len(view.rows)
                if view.spent + price > card.budget:
                    self._tool_result("over budget: assay fewer features or submit")
                    continue
                self._pending = f"assay {len(fids)} features"
                return Assay(request_id=self.request_id(), feature_ids=fids)
            elif tool == "submit":
                known = set(card.feature_ids())
                unknown = [f for f in args.get("ranking", []) if f not in known]
                if unknown and self._turns < self.max_turns:
                    self._tool_result(f"unknown feature ids {unknown[:5]}: use the ids from the world card, then submit again")
                    continue
                ranking = [f for f in dict.fromkeys(args.get("ranking", [])) if f in known]
                p_driver = {f: float(v) for f, v in (args.get("p_driver") or {}).items() if f in known}
                try:
                    p_signal = float(args.get("p_signal", 1.0 if ranking else 0.0))
                except (TypeError, ValueError):
                    p_signal = 1.0 if ranking else 0.0
                return self._submit(card, view, ranking, p_signal, p_driver)
            else:
                self._tool_result(f"tool {tool!r} is not available in {card.mode.value} mode")

    def _submit(self, card: WorldCard, view: EpisodeView, ranking: list[str], p_signal: float, p_driver: dict[str, float]) -> Submit:
        log = self.logs[card.world_id]
        log.ranking = tuple(ranking)
        log.p_signal = min(max(p_signal, 0.0), 1.0)
        log.p_driver = {f: min(max(p_driver.get(f, 1.0), 0.0), 1.0) for f in ranking}
        log.spent, log.n_rows = view.spent, len(view.rows)
        self._world = None
        return Submit(request_id=self.request_id(), ranking=tuple(ranking))


def run(store, ids: list[str], key: str, *, workers: int, policy: Policy = Policy(), bootstrap: int = 200) -> dict:
    """Play the worlds in parallel, one agent per world, and summarise like onc.evaluate."""
    logs: dict[str, EpisodeLog] = {}
    start = time.time()

    def play(world_id: str):
        agent = LLMAgent.from_env(key, logs)
        world = store.world(world_id)
        result = run_episode(agent, Episode(world))
        score = scoring.score_world(result.ranking, store.answer_key(world_id), spent=result.spent, sequential=world.card.mode is Mode.SEQUENTIAL)
        return result, score

    with ThreadPoolExecutor(workers) as pool:
        played = list(pool.map(play, ids))
    import dataclasses

    from onc_agi.services import alignment as alignment_service

    results, scores, regrets = [], [], []
    for world_id, (result, score) in zip(ids, played):
        results.append(dataclasses.replace(result, score=score))
        scores.append(score)
        regrets.append(alignment_service.world_alignment(store.world(world_id), store.answer_key(world_id), result.final_view, score))
    card = scoring.aggregate(scores, scorecard_id=f"llm_{key}-{len(ids)}", agent=f"llm_{key}", tier=store.world(ids[0]).card.tier, alignment=alignment_service.summarise(regrets), bootstrap_draws=bootstrap)
    row = summarise(store, f"llm_{key}", f"llm:{key}", policy, card, results, logs, time.time() - start)
    row["turns"] = float(np.mean([w["notes"].get("turns", 0) for w in row["worlds"]]))
    row["tokens"] = int(sum(w["notes"].get("tokens", 0) for w in row["worlds"]))
    row["errors"] = int(sum(1 for w in row["worlds"] if w["notes"].get("errors")))
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen", choices=sorted(MODELS))
    parser.add_argument("--store", default="toy")
    parser.add_argument("--mode", default="both", choices=["full", "seq", "both"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="artifacts/onc/llm")
    args = parser.parse_args(argv)
    store = open_store(args.store)
    rows = []
    for mode in (["full", "seq"] if args.mode == "both" else [args.mode]):
        ids = world_ids(store, mode, args.limit)
        row = run(store, ids, args.model, workers=args.workers)
        row["mode"], row["label"] = mode, f"{mode} / llm {args.model}"
        rows.append(row)
        print(f"{row['label']:30s} DS {row['discovery_score'] or float('nan'):.3f} find {row['find'] or float('nan'):.2f} restraint {row['restraint'] or float('nan'):+.2f} leak {row['leak_rate']:.2f} cost {row['mean_data_cost']:.0f} turns {row['turns']:.1f} errors {row['errors']} {row['seconds']:.0f}s", flush=True)
    stem = Path(args.out).with_name(Path(args.out).name + f"_{args.model}_{Path(args.store).name}")
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps({"store": args.store, "model": args.model, "conditions": rows}, indent=1, default=float))
    stem.with_suffix(".md").write_text(table(rows) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
