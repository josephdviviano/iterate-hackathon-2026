"""Effect matrix over (type, action, context) rows and the ontology error from counts.

Each object transition is filed into a row keyed by the object's type, the
action, and its local context: the sorted types of the visible objects whose
bounding boxes touch or overlap its own. The effect is the sorted list of
fields that changed, or born, gone, no_change. Row counts with a symmetric
Dirichlet prior give a posterior over effects. The ontology error of a row is
its normalised posterior entropy; rows with no counts are undefined, which is
the gap the program committee fills.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .loader import Transition

SKIP_FIELDS = frozenset({"name"})


@dataclass(frozen=True)
class RowKey:
    type: str
    action: str
    context: str

    def __str__(self) -> str:
        return f"{self.type} | {self.action} | [{self.context}]"


@dataclass
class Row:
    key: RowKey
    effects: Counter = field(default_factory=Counter)

    @property
    def n(self) -> int:
        return sum(self.effects.values())

    def posterior(self, alphabet: list[str], alpha0: float) -> dict[str, float]:
        denom = alpha0 * len(alphabet) + self.n
        return {e: (alpha0 + self.effects.get(e, 0)) / denom for e in alphabet}

    def eta(self, alphabet: list[str], alpha0: float) -> float | None:
        """Normalised entropy of the effect posterior. None when the row has no counts."""
        if self.n == 0 or len(alphabet) < 2:
            return None
        q = self.posterior(alphabet, alpha0)
        h = -sum(p * math.log(p) for p in q.values() if p > 0)
        return h / math.log(len(alphabet))


def object_type(obj: dict) -> str:
    t = obj.get("type")
    if t:
        return str(t)
    tags = obj.get("tags") or []
    return str(tags[0]) if tags else "Unknown"


def _hashable(v):
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


def effect_signature(before: dict | None, after: dict | None) -> str:
    if before is None:
        return "born"
    if after is None:
        return "gone"
    changed = sorted(
        k for k in set(before) | set(after)
        if k not in SKIP_FIELDS and _hashable(before.get(k)) != _hashable(after.get(k))
    )
    return ",".join(changed) if changed else "no_change"


def pair_objects(before: list[dict], after: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """Match objects by name, preferring the same position, then by name alone."""
    after_by_name: dict[str, list[dict]] = defaultdict(list)
    for o in after:
        after_by_name[str(o.get("name"))].append(o)
    used: set[int] = set()
    pairs: list[tuple[dict | None, dict | None]] = []
    unmatched: list[int] = []
    for bo in before:
        candidates = after_by_name.get(str(bo.get("name")), [])
        match = next((ao for ao in candidates if id(ao) not in used
                      and ao.get("x") == bo.get("x") and ao.get("y") == bo.get("y")), None)
        if match is None:
            pairs.append((bo, None))
            unmatched.append(len(pairs) - 1)
        else:
            used.add(id(match))
            pairs.append((bo, match))
    for i in unmatched:
        bo = pairs[i][0]
        assert bo is not None
        for ao in after_by_name.get(str(bo.get("name")), []):
            if id(ao) not in used:
                used.add(id(ao))
                pairs[i] = (bo, ao)
                break
    pairs.extend((None, ao) for ao in after if id(ao) not in used)
    return pairs


def _bbox(o: dict) -> tuple[int, int, int, int]:
    x, y = int(o.get("x", 0)), int(o.get("y", 0))
    return x, y, x + int(o.get("w", 1)) - 1, y + int(o.get("h", 1)) - 1


def context_signature(state: list[dict], obj: dict, click: tuple[int, int] | None = None,
                      margin: int = 1) -> str:
    """Sorted types of touching visible neighbours, plus 'hit' when a click lands on the object."""
    x0, y0, x1, y1 = _bbox(obj)
    parts = []
    for o in state:
        if o is obj or not o.get("visible", True):
            continue
        ox0, oy0, ox1, oy1 = _bbox(o)
        if ox0 <= x1 + margin and ox1 >= x0 - margin and oy0 <= y1 + margin and oy1 >= y0 - margin:
            parts.append(object_type(o))
    if click is not None and x0 <= click[0] <= x1 and y0 <= click[1] <= y1:
        parts.append("hit")
    return ",".join(sorted(parts))


@dataclass
class EffectMatrix:
    rows: dict[RowKey, Row]
    alphabet: list[str]
    alpha0: float = 0.5

    @classmethod
    def from_transitions(cls, transitions: list[Transition], alpha0: float = 0.5) -> "EffectMatrix":
        rows: dict[RowKey, Row] = {}
        effects: set[str] = set()
        for t in transitions:
            if t.action_id == 0 or t.level_advance:
                continue
            for bo, ao in pair_objects(t.before_objs, t.after_objs):
                ref = bo if bo is not None else ao
                assert ref is not None
                key = RowKey(object_type(ref), t.action_key,
                             context_signature(t.before_objs, bo, t.click) if bo is not None else "")
                sig = effect_signature(bo, ao)
                effects.add(sig)
                rows.setdefault(key, Row(key)).effects[sig] += 1
        return cls(rows=rows, alphabet=sorted(effects), alpha0=alpha0)

    def eta(self, key: RowKey) -> float | None:
        row = self.rows.get(key)
        return row.eta(self.alphabet, self.alpha0) if row else None

    def mixed_rows(self) -> list[Row]:
        """Rows with more than one observed effect, worst first."""
        mixed = [r for r in self.rows.values() if len(r.effects) > 1]
        return sorted(mixed, key=lambda r: -(r.eta(self.alphabet, self.alpha0) or 0.0))

    def report(self, limit: int = 30) -> str:
        lines = [f"{len(self.rows)} rows, effect alphabet {self.alphabet}", ""]
        ranked = sorted(self.rows.values(), key=lambda r: -(r.eta(self.alphabet, self.alpha0) or 0.0))
        for r in ranked[:limit]:
            eta = r.eta(self.alphabet, self.alpha0)
            lines.append(f"eta={eta:.2f} n={r.n:3d}  {r.key}  {dict(r.effects)}")
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .loader import build_buffer

    parser = argparse.ArgumentParser(description="Print the effect matrix of a game's buffer.")
    parser.add_argument("game")
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args(argv)
    transitions = build_buffer(args.game)
    if args.level is not None:
        transitions = [t for t in transitions if t.level == args.level]
    print(EffectMatrix.from_transitions(transitions).report(args.limit))


if __name__ == "__main__":
    main()
