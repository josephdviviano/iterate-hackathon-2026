"""Prioritize child runs (trajectories) by their expected certified gain, and track the success rate.

A proposal is (parent, lane). Its score is a Thompson sample of the lane's
success rate (Beta posterior over this climb's own history) times the mean
gain of past successes in that lane, times the parent's room, 1 minus the
parent's self-reported probability that its value is already tight. A run is
a success when its certified value exceeds its parent's by more than
SUCCESS_GAIN. Every prediction is recorded with its outcome so the success
rate and the prioritizer's Brier score are measured, not asserted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

SUCCESS_GAIN = 1e-5
PRIOR_GAIN = 1e-4  # expected gain of a success in a lane with no history yet


@dataclass
class Record:
    round: int
    child: str
    lane: str
    parent: str
    parent_value: float
    p_success: float
    predicted_gain: float
    gain: float | None = None
    success: bool | None = None
    policy: str = "prioritized"  # or "random": the interleaved control arm


@dataclass
class Prioritizer:
    history: list[Record] = field(default_factory=list)

    def lane_stats(self, lane: str) -> dict:
        done = [r for r in self.history if r.lane == lane and r.success is not None]
        wins = [r for r in done if r.success]
        return {"attempts": len(done), "successes": len(wins),
                "rate": (len(wins) + 1) / (len(done) + 2),
                "mean_gain": (sum(r.gain for r in wins) / len(wins)) if wins else PRIOR_GAIN}

    def propose(self, parents: list, lanes: list[str], k: int, rng: random.Random, p_tight_of) -> list[dict]:
        """Score every (parent, lane) and return the top k. p_tight_of(parent) -> stated p or None."""
        scored = []
        for p in parents:
            pt = p_tight_of(p)
            room = 1.0 - (pt if pt is not None else 0.5)
            for lane in lanes:
                st = self.lane_stats(lane)
                a, b = 1 + st["successes"], 1 + st["attempts"] - st["successes"]
                sample = rng.betavariate(a, b)
                scored.append({"parent": p, "lane": lane, "p_success": st["rate"], "sample": sample,
                               "predicted_gain": st["mean_gain"], "score": sample * st["mean_gain"] * room,
                               "room": room})
        scored.sort(key=lambda d: -d["score"])
        # a lane with no outcome yet is tried once before any exploitation, with the best parent
        untried = [lane for lane in lanes if self.lane_stats(lane)["attempts"] == 0]
        for d in scored:
            d["forced"] = d["lane"] in untried and d["parent"] is parents[0]
        scored.sort(key=lambda d: (not d["forced"], -d["score"]))
        chosen, seen = [], set()
        for d in scored:
            key = (d["parent"].name, d["lane"])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(d)
            if len(chosen) == k:
                break
        return chosen

    def record(self, rec: Record) -> None:
        self.history.append(rec)

    def resolve(self, child: str, value: float | None) -> Record | None:
        rec = next((r for r in self.history if r.child == child), None)
        if rec is None:
            return None
        rec.gain = (value - rec.parent_value) if value is not None else None
        rec.success = rec.gain is not None and rec.gain > SUCCESS_GAIN
        return rec

    def propose_random(self, parents: list, lanes: list[str], k: int, rng: random.Random, p_tight_of) -> list[dict]:
        """Control arm: the same proposal set, chosen uniformly at random without replacement."""
        pool = [{"parent": p, "lane": lane, "p_success": self.lane_stats(lane)["rate"],
                 "predicted_gain": self.lane_stats(lane)["mean_gain"],
                 "room": 1.0 - ((p_tight_of(p) or 0.5))} for p in parents for lane in lanes]
        rng.shuffle(pool)
        return pool[:k]

    def arm_report(self) -> dict:
        out = {}
        for arm in ("prioritized", "random"):
            done = [r for r in self.history if r.success is not None and r.policy == arm]
            out[arm] = {"children": len(done), "successes": sum(r.success for r in done),
                        "mean_gain": (sum(r.gain for r in done) / len(done)) if done else None,
                        "by_lane": {l: sum(1 for r in done if r.lane == l) for l in sorted({r.lane for r in done})}}
        return out

    def report(self) -> dict:
        done = [r for r in self.history if r.success is not None]
        by_lane = {lane: self.lane_stats(lane) for lane in sorted({r.lane for r in self.history})}
        brier = (sum((r.p_success - r.success) ** 2 for r in done) / len(done)) if done else None
        buckets: dict = {}
        for r in done:
            b = f"{int(r.p_success * 4) / 4:.2f}"
            d = buckets.setdefault(b, {"n": 0, "successes": 0})
            d["n"] += 1
            d["successes"] += int(r.success)
        return {"runs": len(done), "successes": sum(r.success for r in done),
                "success_rate": (sum(r.success for r in done) / len(done)) if done else None,
                "by_lane": by_lane, "brier": brier, "by_predicted_bucket": buckets,
                "arms": self.arm_report(),
                "records": [r.__dict__ for r in self.history]}

    @staticmethod
    def from_json(d: dict | None) -> "Prioritizer":
        pr = Prioritizer()
        for r in (d or {}).get("records", []):
            pr.history.append(Record(**r))
        return pr
