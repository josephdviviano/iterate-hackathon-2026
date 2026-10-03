"""Live play on the ARC-AGI-3 engine: the committee as a query-by-committee explorer.

The game runs locally through the arc-agi package. Its source is downloaded with
ARC_API_KEY into cache/arc_games (gitignored) and is never read by this code. At
each step every member predicts the outcome of every available action from the
observed object state (the game's released extractor, as in the loader). The agent
takes the action the members disagree on most, observes the result and scores every
member. This is the simulated exploration of explore.py on a fresh trajectory made
of the agent's own actions instead of the recorded ones.
"""

from __future__ import annotations

import json
import math
import os
import random
import select
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .cegis import counterexample_text, probe_split_dir, split_after_probes, stored_round
from .committee import Member, auroc
from .env import call_expr, engine_source, extract_objects, is_frame, returns_frame, takes_frame
from .evaluate import members_from
from .experiment import ARTIFACTS, add_backend_args, backend_cfg, condition_dir, run_split
from .loader import CLICK, ROOT, Transition, build_buffer, final_engine_source, load_bundle, run_extractor, temporal_split
from .seeds import make_seeds
from .synth_api import load_env_file
from .verify import canonical, run_program

GAMES_DIR = ROOT / "cache" / "arc_games"


def open_game(game: str):
    import arc_agi
    from arc_agi import OperationMode

    load_env_file()
    arcade = arc_agi.Arcade(arc_api_key=os.environ.get("ARC_API_KEY", ""), operation_mode=OperationMode.NORMAL,
                            environments_dir=str(GAMES_DIR))
    env = arcade.make(game)
    if env is None:
        raise SystemExit(f"could not load {game}: set ARC_API_KEY and check the network")
    return env


def _key(pred: list[dict] | None) -> str:
    return json.dumps(canonical(pred)) if pred is not None else "<error>"


def _entropy(keys: list[str]) -> float:
    counts = Counter(keys).values()
    if len(counts) < 2:
        return 0.0
    n = len(keys)
    return -sum(c / n * math.log(c / n) for c in counts) / math.log(n)


def predict(source: str, before: list[dict], actions: list[int], grid: list[list[int]] | None = None,
            mode: str = "objects") -> list[list | None]:
    """One member's prediction for each candidate action, each from the same observed state."""
    rows = [Transition(i, 0, a, None, False, grid or [], [], before, []) for i, a in enumerate(actions)]
    verdict = run_program(source, [], rows, timeout_s=60, mode=mode)
    return verdict.test_preds if len(verdict.test_preds) == len(actions) else [None] * len(actions)


def members_for(cond: Path) -> list[Member]:
    return members_from(cond, need_preds=False)


def start(env, level: int):
    """Reset, then move the local engine to the requested level (its public level selector)."""
    frame = env.reset()
    if level > 1:
        game = env._game
        game.set_level(level - 1)
        return frame, game.camera.render(game.current_level.get_sprites()).tolist()
    return frame, frame.frame[-1].tolist()


_STEP_RUNNER = r'''
import sys, json, importlib.util, copy, traceback
sys.setrecursionlimit(10000)
spec = importlib.util.spec_from_file_location("program", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception:
    sys.exit(0)
for line in sys.stdin:
    r = json.loads(line)
    try:
        pred = json.loads(json.dumps(CALL))
        print(json.dumps({"pred": pred}), flush=True)
    except Exception:
        print(json.dumps({"error": traceback.format_exc()[-400:]}), flush=True)
'''


class MemberProcess:
    """One long-lived process per member, so a program keeps hidden state along the trajectory as it
    does under verify.run_program. Every call gets the observed before state (teacher forcing)."""

    def __init__(self, source: str, timeout_s: float = 30, mode: str = "objects"):
        self.dir = Path(tempfile.mkdtemp(prefix="live_member_"))
        (self.dir / "program.py").write_text(source)
        (self.dir / "run.py").write_text(_STEP_RUNNER.replace("CALL", call_expr(mode)))
        self.mode = mode
        self.timeout_s = timeout_s
        self.proc = subprocess.Popen([sys.executable, "-I", "run.py", "program.py"], cwd=self.dir,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def step(self, before: list[dict], action, frame: list[list[int]] | None = None) -> list | None:
        if self.proc.poll() is not None:
            return None
        row = {"before": before, "action": action}
        if takes_frame(self.mode):
            row["frame"] = frame
        try:
            self.proc.stdin.write(json.dumps(row) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            return None
        ready, _, _ = select.select([self.proc.stdout], [], [], self.timeout_s)
        if not ready:
            self.proc.kill()
            return None
        line = self.proc.stdout.readline()
        pred = json.loads(line).get("pred") if line else None
        if returns_frame(self.mode) and not is_frame(pred, frame):
            return None
        return pred

    def close(self) -> None:
        self.proc.kill()
        shutil.rmtree(self.dir, ignore_errors=True)


def play(game: str, level: int, members: list[Member], steps: int, seed: int = 0, verbose: bool = True,
         brief: bool = False) -> dict:
    """Candidate actions are scored by stateless what-if predictions; the chosen action is scored by
    each member's persistent process, which sees the trajectory in order."""
    from arcengine import GameAction, GameState

    env = open_game(game)
    engine_src = final_engine_source(load_bundle(game))
    mode = members[0].mode
    frames = returns_frame(mode)
    frame, grid = start(env, level)
    state = run_extractor(engine_src, [grid])[0]
    rng = random.Random(seed)
    alive = set(range(len(members)))
    seen = {_key(grid if frames else state)}
    tried: set[tuple[str, int]] = set()
    levels_done = frame.levels_completed
    log: list[dict] = []
    t0 = time.time()
    procs = [MemberProcess(m.source, mode=mode) for m in members]
    with ThreadPoolExecutor(max_workers=len(members)) as pool:
        for t in range(steps):
            actions = [a for a in frame.available_actions if a not in (0, CLICK)]
            if not actions:
                break
            preds = list(pool.map(lambda m: predict(m.source, state, actions, grid, mode), members))
            dis = [_entropy([_key(p[j]) for p in preds]) for j in range(len(actions))]
            here = _key(grid if frames else state)
            untried = [j for j, a in enumerate(actions) if (here, a) not in tried] or list(range(len(actions)))
            top = max(dis[j] for j in untried)
            choices = [j for j in untried if dis[j] == top]
            if top == 0:
                # unanimous everywhere: prefer an action whose predicted state is new, then one
                # that changes the state at all
                novel = [j for j in choices if _key(preds[0][j]) not in seen]
                moving = [j for j in choices if _key(preds[0][j]) != here]
                choices = novel or moving or choices
            j = rng.choice(choices)
            action = actions[j]
            tried.add((here, action))
            scored = [p.step(state, action, grid) for p in procs]
            frame = env.step(GameAction[f"ACTION{action}"])
            new_grid = frame.frame[-1].tolist()
            new_state = run_extractor(engine_src, [new_grid])[0]
            advanced = frame.levels_completed > levels_done or frame.full_reset
            levels_done = frame.levels_completed
            truth = _key(new_grid if frames else new_state)
            seen.add(truth)
            keys = [_key(p) for p in scored]
            vote = Counter(keys).most_common(1)[0][0]
            correct = [k == truth for k in keys]
            if not advanced:
                alive = {k for k in alive if correct[k]}
            counts = Counter(keys)
            row = {"step": t, "action": action, "disagreement": round(dis[j], 3), "n_distinct": len(set(keys)),
                   "shares": sorted((c / len(members) for c in counts.values()), reverse=True),
                   "truth_share": counts.get(truth, 0) / len(members),
                   "vote_correct": vote == truth, "members_correct": sum(correct), "alive": len(alive),
                   "level_advance": advanced, "levels_completed": frame.levels_completed, "state": str(frame.state)}
            log.append(row)
            if verbose and not (brief and dis[j] == 0 and row["vote_correct"] and t % 10 and not advanced):
                print(f"  step {t:3d}  action {action}  disagreement {dis[j]:.2f} ({len(set(keys))} outcomes)  "
                      f"vote {'right' if row['vote_correct'] else 'WRONG'}  members right {sum(correct)}/{len(members)}"
                      f"  never wrong {len(alive)}" + ("  level advanced" if advanced else ""), flush=True)
            state, grid = new_state, new_grid
            if frame.state in (GameState.WIN, GameState.GAME_OVER):
                break
    for p in procs:
        p.close()
    scored = [r for r in log if not r["level_advance"]]
    errors = [not r["vote_correct"] for r in scored]
    unanimous = [r for r in scored if r["n_distinct"] == 1]
    split = [r for r in scored if r["n_distinct"] > 1]

    def err(rows):
        return round(sum(not r["vote_correct"] for r in rows) / len(rows), 3) if rows else None

    return {"game": game, "level": level, "mode": mode, "members": len(members), "steps": len(log), "scored": len(scored),
            "vote_accuracy": round(1 - err(scored), 3) if scored else None,
            "auroc_disagreement_vs_error": auroc([r["disagreement"] for r in scored], errors) if scored else None,
            "unanimous_n": len(unanimous), "unanimous_error": err(unanimous),
            "split_n": len(split), "split_error": err(split),
            "first_all_wrong": next((r["step"] for r in scored if r["members_correct"] == 0), None),
            "first_version_space_empty": next((r["step"] for r in log if r["alive"] == 0), None),
            "levels_completed": levels_done, "wall_s": round(time.time() - t0, 1), "log": log}


def trajectory(game: str, level: int, actions: list[int]) -> list[Transition]:
    """Replay recorded live actions on a fresh engine and return the observed transitions.
    Steps are move indices; level advances are flagged so the loader's split rules apply."""
    env = open_game(game)
    engine_src = final_engine_source(load_bundle(game))
    frame, grid = start(env, level)
    before = run_extractor(engine_src, [grid])[0]
    levels_done = frame.levels_completed
    out = []
    for i, a in enumerate(actions):
        from arcengine import GameAction
        frame = env.step(GameAction[f"ACTION{a}"])
        after_grid = frame.frame[-1].tolist()
        after = run_extractor(engine_src, [after_grid])[0]
        advanced = frame.levels_completed > levels_done or frame.full_reset
        levels_done = frame.levels_completed
        out.append(Transition(i, level, a, None, advanced, grid, after_grid, before, after))
        before, grid = after, after_grid
    return out


def live_round(game: str, level: int, train_frac: float, probe: int, log_path: Path, through: int,
               condition: str, runs: int, cfg: dict, parallel: int, dry_run: bool) -> None:
    """Resynthesize on the live trajectory: the recorded train set of the stored round plus the live
    transitions up to and including the refuting move, which is stated as the counterexample."""
    train, test = temporal_split(build_buffer(game), level, train_frac)
    probes, cond = stored_round(game, level, train_frac, "committee_devin", "cegis_devin", probe)
    train_r, _ = split_after_probes(train, test, probes)
    members = members_for(cond)
    mode = members[0].mode
    engine_src = engine_source(game, mode)
    log = json.loads(log_path.read_text())["log"]
    live = [t for t in trajectory(game, level, [r["action"] for r in log[:through + 1]]) if not t.level_advance]
    refuting = next(r["step"] for r in log if r["members_correct"] == 0)
    t = live[refuting]
    preds = [predict(m.source, t.before_objs, [t.action_id], t.before_grid, mode)[0] for m in members]
    objs = extract_objects(engine_src, preds) if engine_src else [None] * len(preds)
    witnesses = [Member(m.name, m.source, [p], m.length, mode, [o] if engine_src else None)
                 for m, p, o in zip(members, preds, objs)]
    cfg = {**cfg, "mode": mode}
    text = counterexample_text(witnesses, [t], [0]).replace("The last transition in transitions.md",
                                                            f"Transition {len(train_r) + refuting} in transitions.md")
    train_all = train_r + live
    seeds = [f"{s}\n\n{text}" for s in make_seeds(train_all, runs)]
    base = ARTIFACTS / game / f"L{level}_f{int(round(train_frac * 100))}_probe{probe}_live{through}" / condition
    print(f"train {len(train_r)} recorded + {len(live)} live (refuting move {refuting}); artifacts {base}\n\n{seeds[0]}\n")
    if dry_run:
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    (base.parent / "split.json").write_text(json.dumps(
        {"source": str(cond.relative_to(ARTIFACTS)), "live_log": str(log_path.resolve().relative_to(ARTIFACTS)),
         "through": through, "refuting_move": refuting, "actions": [r["action"] for r in log[:through + 1]]}, indent=1))
    run_split(train_all, [], base, seeds, cfg, f"{game} L{level} {condition}", 0, parallel, engine_src)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Play a game live with a stored committee as the explorer.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, default=3, help="level to start at; the committee's training level")
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--condition", default=None, help="committee_devin for round 1, cegis_devin with --probe")
    parser.add_argument("--probe", type=int, default=0, help="use the round that observed this many probes")
    parser.add_argument("--members-dir", default=None, help="any condition directory under artifacts/")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resynth-from", default=None, help="a live log: resynthesize on its trajectory")
    parser.add_argument("--through", type=int, default=70, help="last live move to include in train")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--brief", action="store_true", help="print only every 10th move and the moves with disagreement or error")
    add_backend_args(parser)
    args = parser.parse_args(argv)
    if args.resynth_from:
        live_round(args.game, args.level, args.train_frac, args.probe, Path(args.resynth_from), args.through,
                   args.condition or "live_devin", args.runs, backend_cfg(args), args.parallel, args.dry_run)
        return
    args.condition = args.condition or ("cegis_devin" if args.probe else "committee_devin")
    if args.members_dir:
        cond, tag = ARTIFACTS / args.members_dir, args.members_dir.replace("/", "_")
    else:
        cond = (probe_split_dir(args.game, args.level, args.train_frac, args.probe, args.condition) if args.probe
                else condition_dir(args.game, args.level, args.train_frac, args.condition))
        tag = f"L{args.level}_{args.condition}_probe{args.probe}"
    members = members_for(cond)
    print(f"{args.game} level {args.level}, live, {len(members)} members from {cond.relative_to(ARTIFACTS)}")
    out = play(args.game, args.level, members, args.steps, args.seed, brief=args.brief)
    print(f"  steps {out['steps']} (scored {out['scored']})  vote accuracy {out['vote_accuracy']}  "
          f"AUROC {out['auroc_disagreement_vs_error']}  unanimous {out['unanimous_n']} (error {out['unanimous_error']})"
          f"  split {out['split_n']} (error {out['split_error']})  all members wrong first at step "
          f"{out['first_all_wrong']}  levels completed {out['levels_completed']}  wall {out['wall_s']}s")
    path = ARTIFACTS / args.game / "live" / f"{tag}_seed{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
