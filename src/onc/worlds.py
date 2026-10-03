"""Seeded generator of ONC-AGI dev worlds.

A dev world has the shape of a shipped toy world: 240 patients, a 35 percent
outcome rate and 23 columns (10 expression, 3 copy number, 2 protein,
2 clinical, 2 derived, 1 baseline lab, 3 post-outcome lab). One role plants
one mechanism. The answer key is correct by construction; no oracle certifies
it. Scores on dev worlds are dev results, not benchmark results.

Build a store::

    uv run python -m onc.worlds --out artifacts/onc/dev --n-per-role 8 --seed 0
"""

from __future__ import annotations

import argparse
import json
import shutil
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from onc_agi.core.schema import (
    AnswerKey,
    CreditRule,
    FeatureMeta,
    GroupLabel,
    Mode,
    NameVisibility,
    PriceList,
    Tier,
    Timing,
    TrueGroup,
    TruthPart,
    WorldCard,
)
from onc_agi.core.world import WorldData
from onc_agi.infra.bundles import FileWorldStore, write_world
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import brentq
from scipy.special import expit
from scipy.spatial.distance import squareform

BUILDER = "dev-1"
DEFAULT_OUT = Path("artifacts/onc/dev")
RECRUIT_PRICE = 40.0
ASSAY_PRICE = {
    "expression": 2.0,
    "copy_number": 3.0,
    "protein": 8.0,
    "clinical": 0.5,
    "derived": 1.0,
    "lab": 5.0,
}
CLUSTER_R = 0.8
MIN_STRATUM = 4
NULL_SHARE = 0.2
PERMUTATIONS = 200
MODE_TAG = {Mode.FULL_ACCESS: "full", Mode.SEQUENTIAL: "seq"}
FREE_EXPRESSION = ("ex_1", "ex_2", "ex_3")
POST = ("po_1", "po_2", "po_3")

Array = NDArray[np.float64]


@dataclass
class Pool:
    """Columns under slot names. Fake names replace the slot names at write time."""

    rng: np.random.Generator
    n: int
    values: dict[str, Array] = field(default_factory=dict)
    kind: dict[str, str] = field(default_factory=dict)

    def put(self, slot: str, values: Array, kind: str) -> None:
        self.values[slot] = np.asarray(values, dtype=float)
        self.kind[slot] = kind

    def z(self, slot: str) -> Array:
        v = self.values[slot]
        return (v - v.mean()) / v.std()

    def noise(self) -> Array:
        return self.rng.standard_normal(self.n)

    def pick(self, slots: Sequence[str], count: int = 1) -> list[str]:
        return [str(s) for s in self.rng.choice(slots, size=count, replace=False)]


def backbone(pool: Pool) -> None:
    """The shared column layout. Correlation strengths are drawn per world."""
    rng = pool.rng
    twin = pool.noise()
    a = rng.uniform(0.2, 0.35)
    pool.put("dup_a", twin + a * pool.noise(), "expression")
    pool.put("dup_b", twin + a * pool.noise(), "expression")
    core = pool.noise()
    b = rng.uniform(0.9, 1.2)
    for slot in ("blk_1", "blk_2", "blk_3"):
        pool.put(slot, core + b * pool.noise(), "expression")
    for slot in ("cn_src", "cn_1", "cn_2"):
        pool.put(slot, pool.noise(), "copy_number")
    g = rng.uniform(0.6, 0.75)
    pool.put("cn_tgt", g * pool.z("cn_src") + np.sqrt(1 - g * g) * pool.noise(), "expression")
    for slot in (*FREE_EXPRESSION, "ex_4"):
        pool.put(slot, pool.noise(), "expression")
    pool.put("pr_1", pool.noise(), "protein")
    h = rng.uniform(0.5, 0.65)
    pool.put("pr_2", h * pool.z("ex_4") + np.sqrt(1 - h * h) * pool.noise(), "protein")
    for slot, low, high in (("cl_1", 0.35, 0.55), ("cl_2", 0.4, 0.6)):
        share = rng.uniform(low, high)
        pool.put(slot, (rng.random(pool.n) < share).astype(float), "clinical")
    pool.put("dv_1", pool.noise(), "derived")
    pool.put("dv_2", pool.noise(), "derived")
    pool.put("lb_1", pool.noise(), "lab")


def post_outcome(pool: Pool, y: NDArray[np.int64], leak: bool) -> None:
    for slot in POST:
        pool.put(slot, pool.noise(), "lab")
    if leak:
        pool.values[POST[0]] = y + 0.5 * pool.noise()


def outcome(signal: Array, prevalence: float, rng: np.random.Generator) -> NDArray[np.int64]:
    """Binary outcome with logit = alpha + signal, alpha set to the target prevalence."""
    alpha = brentq(lambda a: float(expit(a + signal).mean()) - prevalence, -40.0, 40.0)
    return (rng.random(len(signal)) < expit(alpha + signal)).astype(np.int64)


# --------------------------------------------------------------------------- roles


@dataclass(frozen=True)
class Planted:
    signal: Array
    groups: tuple[TrueGroup, ...] = ()
    leak: bool = False


def single(
    slot: str,
    role: str,
    *,
    equivalent: Sequence[str] = (),
    exact: bool = True,
    label: GroupLabel = GroupLabel.RECOVERABLE,
    weight: float = 1.0,
    group_id: str | None = None,
) -> TrueGroup:
    part = TruthPart(
        true_feature=slot, equivalence_set=(slot, *equivalent), exact_recoverable=exact, weight=weight
    )
    return TrueGroup(group_id=group_id or f"g-{role}", role=role, label=label, parts=(part,))


def plant_null(pool: Pool, effect: float) -> Planted:
    return Planted(np.zeros(pool.n))


def plant_driver(pool: Pool, effect: float) -> Planted:
    (f,) = pool.pick(FREE_EXPRESSION)
    return Planted(1.6 * effect * pool.z(f), (single(f, "generating"),))


def plant_stand_in(pool: Pool, effect: float) -> Planted:
    group = single("dup_a", "stand_in", equivalent=("dup_b",), exact=False)
    return Planted(1.6 * effect * pool.z("dup_a"), (group,))


def plant_wrong_type(pool: Pool, effect: float) -> Planted:
    return Planted(1.6 * effect * pool.z("cn_src"), (single("cn_src", "wrong_data_type"),))


def plant_confounder(pool: Pool, effect: float) -> Planted:
    s = pool.values["cl_1"]
    for slot in pool.pick(FREE_EXPRESSION, 2):
        pool.values[slot] = pool.values[slot] + 1.2 * (s - s.mean())
    return Planted(2.2 * effect * pool.z("cl_1"), (single("cl_1", "observed_confounder"),))


def plant_leak(pool: Pool, effect: float) -> Planted:
    return Planted(1.6 * effect * pool.z("blk_1"), (single("blk_1", "generating"),), leak=True)


def plant_interaction(pool: Pool, effect: float) -> Planted:
    a, b = pool.pick(FREE_EXPRESSION, 2)
    parts = tuple(TruthPart(true_feature=f, equivalence_set=(f,), exact_recoverable=True) for f in (a, b))
    group = TrueGroup(
        group_id="g-interaction",
        role="interaction",
        label=GroupLabel.RECOVERABLE,
        credit_rule=CreditRule.JOINT,
        parts=parts,
    )
    return Planted(1.8 * effect * pool.z(a) * pool.z(b), (group,))


def plant_module(pool: Pool, effect: float) -> Planted:
    weights = {"pr_1": 1.2 * effect, "cn_1": 0.9 * effect, "blk_2": 0.7 * effect}
    signal = sum(w * pool.z(f) for f, w in weights.items())
    parts = tuple(
        TruthPart(true_feature=f, equivalence_set=(f,), exact_recoverable=True, weight=w)
        for f, w in weights.items()
    )
    group = TrueGroup(
        group_id="g-module",
        role="module",
        label=GroupLabel.RECOVERABLE,
        credit_rule=CreditRule.WEIGHTED_COVERAGE,
        parts=parts,
    )
    return Planted(np.asarray(signal, dtype=float), (group,))


def plant_neutral(pool: Pool, effect: float) -> Planted:
    (f,) = pool.pick(FREE_EXPRESSION)
    small = single(
        "dv_1",
        "generating",
        exact=False,
        label=GroupLabel.NEUTRAL,
        weight=0.05 * effect,
        group_id="g-neutral",
    )
    signal = 1.6 * effect * pool.z(f) + 0.05 * effect * pool.z("dv_1")
    return Planted(signal, (single(f, "generating"), small))


def plant_hidden_cause(pool: Pool, effect: float) -> Planted:
    latent = pool.noise()
    (strong,) = pool.pick(FREE_EXPRESSION)
    s, w = pool.rng.uniform(0.8, 0.9), pool.rng.uniform(0.5, 0.65)
    pool.values[strong] = s * latent + np.sqrt(1 - s * s) * pool.noise()
    pool.values["pr_1"] = w * latent + np.sqrt(1 - w * w) * pool.noise()
    group = single(strong, "hidden_cause", equivalent=("pr_1",), exact=False)
    return Planted(1.6 * effect * latent, (group,))


def plant_mediator(pool: Pool, effect: float) -> Planted:
    group = single("cn_tgt", "mediator", equivalent=("cn_src",))
    return Planted(1.6 * effect * pool.z("cn_tgt"), (group,))


@dataclass(frozen=True)
class Role:
    name: str
    tier: int
    n_ref: float  # reference cohort as a share of the pool
    plant: Callable[[Pool, float], Planted]
    two_strata: bool = False


ROLES = (
    Role("null", 0, 2 / 3, plant_null),
    Role("driver", 0, 1 / 2, plant_driver),
    Role("stand_in", 1, 2 / 3, plant_stand_in),
    Role("wrong_data_type", 1, 2 / 3, plant_wrong_type),
    Role("observed_confounder", 0, 2 / 3, plant_confounder, two_strata=True),
    Role("leak", 0, 1 / 2, plant_leak),
    Role("interaction", 2, 1.0, plant_interaction),
    Role("module", 2, 5 / 6, plant_module),
    Role("neutral", 1, 2 / 3, plant_neutral),
    Role("hidden_cause", 1, 2 / 3, plant_hidden_cause),
    Role("mediator", 1, 2 / 3, plant_mediator),
)
SIGNAL_ROLES = tuple(r for r in ROLES if r.name != "null")


# --------------------------------------------------------------------------- answer-key parts


def clusters(x: Array, ids: Sequence[str]) -> dict[str, int]:
    """Complete-linkage clusters at |r| >= CLUSTER_R."""
    r = np.nan_to_num(np.abs(np.corrcoef(x, rowvar=False)))
    np.fill_diagonal(r, 1.0)
    dist = squareform(np.clip(1.0 - r, 0.0, None), checks=False)
    labels = fcluster(linkage(dist, method="complete"), t=1.0 - CLUSTER_R, criterion="distance")
    return {f: int(c) for f, c in zip(ids, labels, strict=True)}


def strata(x: Array, types: Sequence[str], ids: Sequence[str]) -> dict[str, str]:
    """Data type x correlation band, coarsened until each stratum has MIN_STRATUM members."""
    r = np.nan_to_num(np.abs(np.corrcoef(x, rowvar=False)))
    np.fill_diagonal(r, 0.0)
    band = np.digitize(r.max(axis=0), [0.5, CLUSTER_R])
    fine = [f"{t}|r{b}" for t, b in zip(types, band, strict=True)]
    labels = ["mixed"] * len(ids)
    for data_type in set(types):
        idx = [j for j, t in enumerate(types) if t == data_type]
        counts = Counter(fine[j] for j in idx)
        if all(counts[fine[j]] >= MIN_STRATUM for j in idx):
            level = fine
        elif len(idx) >= MIN_STRATUM:
            level = list(types)
        else:
            continue
        for j in idx:
            labels[j] = level[j]
    return dict(zip(ids, labels, strict=True))


def detection_threshold(x: Array, prevalence: float, rng: np.random.Generator) -> float:
    """95th percentile of the largest |z| over features, under permuted outcomes."""
    n = x.shape[0]
    sd = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    z = (x - x.mean(axis=0)) / sd
    base = np.zeros(n)
    base[: max(1, round(prevalence * n))] = 1.0
    perms = rng.permuted(np.tile(base, (PERMUTATIONS, 1)), axis=1)
    yc = (perms - perms.mean(axis=1, keepdims=True)) / perms.std(axis=1, keepdims=True)
    maxima = np.abs(yc @ z).max(axis=1) / np.sqrt(n)
    return float(np.quantile(maxima, 0.95))


def fake_names(count: int, rng: np.random.Generator) -> list[str]:
    letters = list("abcdefghijklmnopqrstuvwxyz")
    out: list[str] = []
    while len(out) < count:
        name = "".join(rng.choice(letters, size=3)) + str(int(rng.integers(10, 100)))
        if name not in out:
            out.append(name)
    return out


def renamed(group: TrueGroup, public: dict[str, str]) -> TrueGroup:
    parts = tuple(
        p.model_copy(
            update={
                "true_feature": public[p.true_feature],
                "equivalence_set": tuple(public[f] for f in p.equivalence_set),
            }
        )
        for p in group.parts
    )
    return group.model_copy(update={"parts": parts})


# --------------------------------------------------------------------------- world assembly


def world_seed(seed: int, role: str, index: int, mode: Mode) -> int:
    entropy = [seed, zlib.crc32(role.encode()), index, list(Mode).index(mode)]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def build_world(
    role: Role,
    index: int,
    mode: Mode,
    *,
    seed: int,
    n: int,
    effect: float,
    prevalence: float,
) -> tuple[WorldData, AnswerKey, int]:
    """One fresh draw for (seed, role, index, mode). Returns the world, its key and its seed."""
    world_id = f"dev-{role.name}-{index:02d}-{MODE_TAG[mode]}"
    draw = world_seed(seed, role.name, index, mode)
    rng = np.random.default_rng(draw)
    pool = Pool(rng, n)
    backbone(pool)
    planted = role.plant(pool, effect)
    y = outcome(planted.signal, prevalence, rng)
    post_outcome(pool, y, planted.leak)

    slots = list(pool.values)
    order = [slots[i] for i in rng.permutation(len(slots))]
    public = dict(zip(order, fake_names(len(order), rng), strict=True))
    x = np.column_stack([pool.values[s] for s in order])
    types = [pool.kind[s] for s in order]
    ids = [public[s] for s in order]
    features = tuple(
        FeatureMeta(
            feature_id=public[s],
            data_type=pool.kind[s],
            timing=Timing.POST_OUTCOME if s in POST else Timing.BASELINE,
            assay_price=ASSAY_PRICE[pool.kind[s]],
        )
        for s in order
    )
    per_patient = RECRUIT_PRICE + sum(f.assay_price for f in features)

    # Null worlds cycle through the tiers so every tier has null worlds; odd ones have two strata.
    is_null = role.name == "null"
    tier = index % 3 if is_null else role.tier
    if role.two_strata or (is_null and index % 2 == 1):
        names = ("s-a", "s-b")
        stratum = tuple("s-a" if v > 0.5 else "s-b" for v in pool.values["cl_1"])
    else:
        names = ("all",)
        stratum = tuple("all" for _ in range(n))
    queues = {
        s: tuple(int(i) for i in rng.permutation([i for i, v in enumerate(stratum) if v == s]))
        for s in names
    }
    patients = tuple(f"p{i:04d}" for i in rng.permutation(n))
    card = WorldCard(
        world_id=world_id,
        tier=Tier.PUBLIC_TRAIN,
        mode=mode,
        n_pool=n,
        features=features,
        strata=names,
        stratum_sizes={s: len(q) for s, q in queues.items()},
        prices=PriceList(recruit_per_patient=RECRUIT_PRICE),
        budget=n * per_patient,
        name_visibility=NameVisibility.FAKE,
    )
    key = AnswerKey(
        world_id=world_id,
        difficulty_tier=tier,
        groups=tuple(renamed(g, public) for g in planted.groups),
        reject_set=tuple(public[s] for s in POST),
        clusters=clusters(x, ids),
        strata=strata(x, types, ids),
        reference_cost=round(role.n_ref * n) * per_patient,
        detection_threshold=detection_threshold(x, prevalence, rng),
        oracle_version=f"{BUILDER}-by-construction",
    )
    world = WorldData(card=card, patient_ids=patients, x=x, y=y, stratum=stratum, queues=queues)
    return world, key, draw


def build_dev_store(
    root: Path,
    *,
    n_per_role: int,
    seed: int,
    modes: Sequence[str] = ("full_access", "sequential"),
    n_patients: int = 240,
    effect: float = 1.0,
    prevalence: float = 0.35,
) -> list[dict]:
    """Write a store of dev worlds under ``root`` and return the manifest records.

    Each signal role gets ``n_per_role`` worlds per mode. Null worlds are
    NULL_SHARE of the store. An existing store under ``root`` is replaced.
    """
    if effect <= 0:
        raise ValueError("effect must be positive; the answer key stores the planted weights")
    root = Path(root)
    shutil.rmtree(root / Tier.PUBLIC_TRAIN.value, ignore_errors=True)
    n_null = max(1, round(NULL_SHARE / (1 - NULL_SHARE) * len(SIGNAL_ROLES) * n_per_role))
    records: list[dict] = []
    for role in ROLES:
        for index in range(n_null if role.name == "null" else n_per_role):
            for mode in (Mode(m) for m in modes):
                world, key, draw = build_world(
                    role, index, mode, seed=seed, n=n_patients, effect=effect, prevalence=prevalence
                )
                write_world(root, world, key)
                records.append(
                    {
                        "world_id": world.card.world_id,
                        "role": role.name,
                        "mode": mode.value,
                        "difficulty_tier": key.difficulty_tier,
                        "depth": key.depth,
                        "seed": draw,
                    }
                )
    manifest = {
        "builder": BUILDER,
        "kind": "dev worlds, not benchmark results; answer keys by construction",
        "seed": seed,
        "n_per_role": n_per_role,
        "n_patients": n_patients,
        "effect": effect,
        "prevalence": prevalence,
        "worlds": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return records


def split_ids(
    manifest: dict | Sequence[dict], *, held_out_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Deterministic (train, held_out) world ids, stratified by role and mode."""
    records = manifest["worlds"] if isinstance(manifest, dict) else manifest
    groups: dict[tuple[str, str], list[str]] = {}
    for r in records:
        groups.setdefault((r["role"], r["mode"]), []).append(r["world_id"])
    rng = np.random.default_rng(seed)
    train: list[str] = []
    held: list[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        order = [ids[i] for i in rng.permutation(len(ids))]
        k = round(held_out_fraction * len(ids))
        held.extend(order[:k])
        train.extend(order[k:])
    return sorted(train), sorted(held)


# --------------------------------------------------------------------------- engine checks


def oracle_ranking(key: AnswerKey) -> list[str]:
    return [p.true_feature for g in key.recoverable for p in g.parts]


def check_store(root: Path, agent: str = "univariate_bh") -> dict:
    """Load each world through the engine, score the oracle and the empty list, and run ``agent``."""
    from onc_agi.adapters.agents import make_agent
    from onc_agi.services import scoring
    from onc_agi.services.kit import evaluate

    root = Path(root)
    store = FileWorldStore(root)
    manifest = json.loads((root / "manifest.json").read_text())
    role_of = {r["world_id"]: r["role"] for r in manifest["worlds"]}
    ids = store.world_ids(Tier.PUBLIC_TRAIN)
    oracle_find: list[float] = []
    restrained: list[bool] = []
    for world_id in ids:
        store.world(world_id)
        key = store.answer_key(world_id)
        if key.is_null:
            restrained.append(scoring.score_world([], key).restrained)
        else:
            oracle_find.append(scoring.score_world(oracle_ranking(key), key).find)
    full = [w for w in ids if w.endswith("-full")] or list(ids)
    card, _ = evaluate(make_agent(agent), store, Tier.PUBLIC_TRAIN, world_ids=full)
    per_role: dict[str, list[float]] = {}
    for w in card.worlds:
        value = float(w.restrained) if w.is_null else w.find
        per_role.setdefault(role_of[w.world_id], []).append(value)
    return {
        "n_worlds": len(ids),
        "oracle_find_min": min(oracle_find, default=1.0),
        "null_restrained": all(restrained),
        "agent": agent,
        "n_scored": len(full),
        "discovery_score": card.discovery_score,
        "find": card.find,
        "restraint": card.restraint,
        "per_role": {r: float(np.mean(v)) for r, v in sorted(per_role.items())},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a store of seeded ONC-AGI dev worlds.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="store directory")
    parser.add_argument("--n-per-role", type=int, default=8, help="worlds per signal role and mode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-patients", type=int, default=240)
    parser.add_argument("--effect", type=float, default=1.0, help="scale on every planted effect")
    parser.add_argument("--prevalence", type=float, default=0.35)
    parser.add_argument("--modes", nargs="+", default=[m.value for m in Mode], choices=[m.value for m in Mode])
    parser.add_argument("--check", action="store_true", help="load each world through the engine and score a baseline")
    args = parser.parse_args(argv)
    records = build_dev_store(
        args.out,
        n_per_role=args.n_per_role,
        seed=args.seed,
        modes=args.modes,
        n_patients=args.n_patients,
        effect=args.effect,
        prevalence=args.prevalence,
    )
    counts = Counter(r["role"] for r in records)
    for role in ROLES:
        print(f"{role.name:<20} {counts[role.name]:>4}")
    print(f"wrote {len(records)} dev worlds to {args.out}")
    if args.check:
        print(json.dumps(check_store(args.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
