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
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .committee import Member, auroc, description_length
from .evaluate import load_runs
from .experiment import ARTIFACTS, condition_dir
from .loader import CLICK, ROOT, Transition, final_engine_source, load_bundle, run_extractor
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


def predict(source: str, before: list[dict], actions: list[int]) -> list[list[dict] | None]:
    """One member's prediction for each candidate action, each from the same observed state."""
    rows = [Transition(i, 0, a, None, False, [], [], before, []) for i, a in enumerate(actions)]
    verdict = run_program(source, [], rows, timeout_s=60)
    return verdict.test_preds if len(verdict.test_preds) == len(actions) else [None] * len(actions)


def members_for(cond: Path) -> list[Member]:
    return [Member(name, src, [], description_length(src)) for name, m, src, _ in load_runs(cond) if m["consistent"]]


def start(env, level: int):
    """Reset, then move the local engine to the requested level (its public level selector)."""
    frame = env.reset()
    if level > 1:
        game = env._game
        game.set_level(level - 1)
        return frame, game.camera.render(game.current_level.get_sprites()).tolist()
    return frame, frame.frame[-1].tolist()


def play(game: str, level: int, members: list[Member], steps: int, seed: int = 0, verbose: bool = True) -> dict:
    from arcengine import GameAction, GameState

    env = open_game(game)
    engine_src = final_engine_source(load_bundle(game))
    frame, grid = start(env, level)
    state = run_extractor(engine_src, [grid])[0]
    rng = random.Random(seed)
    alive = set(range(len(members)))
    seen = {_key(state)}
    tried: set[tuple[str, int]] = set()
    levels_done = frame.levels_completed
    log: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(members)) as pool:
        for t in range(steps):
            actions = [a for a in frame.available_actions if a not in (0, CLICK)]
            if not actions:
                break
            preds = list(pool.map(lambda m: predict(m.source, state, actions), members))
            dis = [_entropy([_key(p[j]) for p in preds]) for j in range(len(actions))]
            here = _key(state)
            untried = [j for j, a in enumerate(actions) if (here, a) not in tried] or list(range(len(actions)))
            top = max(dis[j] for j in untried)
            choices = [j for j in untried if dis[j] == top]
            if top == 0:
                # unanimous everywhere: prefer an action whose predicted state is new, then one
                # that changes the state at all
                novel = [j for j in choices if _key(preds[0][j]) not in seen]
                moving = [j for j in choices if _key(preds[0][j]) != _key(state)]
                choices = novel or moving or choices
            j = rng.choice(choices)
            action = actions[j]
            tried.add((here, action))
            frame = env.step(GameAction[f"ACTION{action}"])
            new_state = run_extractor(engine_src, [frame.frame[-1].tolist()])[0]
            advanced = frame.levels_completed > levels_done or frame.full_reset
            levels_done = frame.levels_completed
            truth = _key(new_state)
            seen.add(truth)
            keys = [_key(p[j]) for p in preds]
            vote = Counter(keys).most_common(1)[0][0]
            correct = [k == truth for k in keys]
            if not advanced:
                alive = {k for k in alive if correct[k]}
            row = {"step": t, "action": action, "disagreement": round(dis[j], 3), "n_distinct": len(set(keys)),
                   "vote_correct": vote == truth, "members_correct": sum(correct), "alive": len(alive),
                   "level_advance": advanced, "levels_completed": frame.levels_completed, "state": str(frame.state)}
            log.append(row)
            if verbose:
                print(f"  step {t:3d}  action {action}  disagreement {dis[j]:.2f} ({len(set(keys))} outcomes)  "
                      f"vote {'right' if row['vote_correct'] else 'WRONG'}  members right {sum(correct)}/{len(members)}"
                      f"  never wrong {len(alive)}" + ("  level advanced" if advanced else ""), flush=True)
            state = new_state
            if frame.state in (GameState.WIN, GameState.GAME_OVER):
                break
    scored = [r for r in log if not r["level_advance"]]
    errors = [not r["vote_correct"] for r in scored]
    unanimous = [r for r in scored if r["n_distinct"] == 1]
    split = [r for r in scored if r["n_distinct"] > 1]

    def err(rows):
        return round(sum(not r["vote_correct"] for r in rows) / len(rows), 3) if rows else None

    return {"game": game, "level": level, "members": len(members), "steps": len(log), "scored": len(scored),
            "vote_accuracy": round(1 - err(scored), 3) if scored else None,
            "auroc_disagreement_vs_error": auroc([r["disagreement"] for r in scored], errors) if scored else None,
            "unanimous_n": len(unanimous), "unanimous_error": err(unanimous),
            "split_n": len(split), "split_error": err(split),
            "first_all_wrong": next((r["step"] for r in scored if r["members_correct"] == 0), None),
            "first_version_space_empty": next((r["step"] for r in log if r["alive"] == 0), None),
            "levels_completed": levels_done, "wall_s": round(time.time() - t0, 1), "log": log}


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .cegis import probe_split_dir

    parser = argparse.ArgumentParser(description="Play a game live with a stored committee as the explorer.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, default=3, help="level to start at; the committee's training level")
    parser.add_argument("--train-frac", type=float, default=0.4)
    parser.add_argument("--condition", default=None, help="committee_devin for round 1, cegis_devin with --probe")
    parser.add_argument("--probe", type=int, default=0, help="use the round that observed this many probes")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.condition = args.condition or ("cegis_devin" if args.probe else "committee_devin")
    cond = (probe_split_dir(args.game, args.level, args.train_frac, args.probe, args.condition) if args.probe
            else condition_dir(args.game, args.level, args.train_frac, args.condition))
    members = members_for(cond)
    print(f"{args.game} level {args.level}, live, {len(members)} members from {cond.relative_to(ARTIFACTS)}")
    out = play(args.game, args.level, members, args.steps, args.seed)
    print(f"  steps {out['steps']} (scored {out['scored']})  vote accuracy {out['vote_accuracy']}  "
          f"AUROC {out['auroc_disagreement_vs_error']}  unanimous {out['unanimous_n']} (error {out['unanimous_error']})"
          f"  split {out['split_n']} (error {out['split_error']})  all members wrong first at step "
          f"{out['first_all_wrong']}  levels completed {out['levels_completed']}  wall {out['wall_s']}s")
    path = ARTIFACTS / args.game / "live" / f"L{args.level}_{args.condition}_probe{args.probe}_seed{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
