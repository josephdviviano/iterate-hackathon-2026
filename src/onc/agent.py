"""The committee agent for ONC-AGI, in full-access and sequential mode.

Sequential mode recruits an initial batch, assays every baseline feature on
each recruited patient (post-outcome features are never bought), and then
decides between one more recruit batch and submission by the expected drop in
committee disagreement per unit price. The committee from before a batch is
scored on that batch after it is revealed, which gives a true held-out record
of p(y | x) for calibration and conformal coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from onc_agi.core.schema import Action, Assay, Mode, Recruit, Submit, Timing, WorldCard
from onc_agi.services.engine import EpisodeView
from onc_agi.services.kit import Agent

from onc.committee import TAU, Committee, build_committee
from onc.hypotheses import TEMPLATES, Features, features_from


@dataclass(frozen=True)
class Policy:
    """The decision parameters the trained policy may change. The committee is not one of them."""

    tau: float = TAU
    n_initial: int = 60
    batch: int = 40
    stop_threshold: float = 0.02  # expected disagreement drop per 1000 USD
    spend_cap: float = 0.5  # share of the budget taken as the reference-cost estimate
    acquisition: str = "disagreement"  # disagreement | random | all
    weighting: str = "likelihood"
    beta: float = 1.0
    sharpness: float = 1.0  # exponent on the odds of every reported probability; 1 reports the committee as it is
    templates: tuple[str, ...] = ()  # template names for an ablation; empty means every template
    seed: int = 0

    def report(self, p: float) -> float:
        """The probability the policy reports for a committee probability ``p``."""
        p = min(max(p, 1e-6), 1 - 1e-6)
        odds = (p / (1 - p)) ** self.sharpness
        return float(odds / (1 + odds))


@dataclass
class EpisodeLog:
    world_id: str
    mode: str
    ranking: tuple[str, ...] = ()
    p_signal: float = 0.0
    p_driver: dict[str, float] = field(default_factory=dict)
    disagreement: list[float] = field(default_factory=list)
    held_out: list[tuple[float, int]] = field(default_factory=list)
    resynthesis_steps: list[int] = field(default_factory=list)
    spent: float = 0.0
    n_rows: int = 0
    members: list[dict] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    expected_drop: list[float] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


def fit_committee(feats: Features, policy: Policy) -> Committee:
    templates = {name: TEMPLATES[name] for name in policy.templates} if policy.templates else None
    return build_committee(feats, weighting=policy.weighting, beta=policy.beta, templates=templates, seed=policy.seed)


class CommitteeAgent(Agent):
    def __init__(self, policy: Policy = Policy(), name: str = "committee") -> None:
        super().__init__()
        self.name = name
        self.policy = policy
        self.logs: dict[str, EpisodeLog] = {}
        self._committee: Committee | None = None
        self._rows: set[int] = set()
        self._rng = np.random.default_rng(policy.seed)
        self._random_stop: int = 0
        self._initial_pending: list[str] = []

    # ------------------------------------------------------------------ helpers

    def _features(self, card: WorldCard, view: EpisodeView) -> Features:
        return features_from(card, view.feature_ids, view.x, view.outcome, view.stratum)

    def _fit(self, feats: Features) -> Committee:
        return fit_committee(feats, self.policy)

    def _baseline_ids(self, card: WorldCard) -> tuple[str, ...]:
        return tuple(f.feature_id for f in card.features if f.timing is Timing.BASELINE)

    def _submit(self, card: WorldCard, committee: Committee, log: EpisodeLog, spent: float) -> Submit:
        post = {f.feature_id for f in card.features if f.timing is Timing.POST_OUTCOME}
        ranking = tuple(f for f in committee.ranking(self.policy.tau) if f not in post)
        log.ranking, log.p_signal = ranking, self.policy.report(committee.p_signal)
        log.p_driver = {f: self.policy.report(v) for f, v in committee.p_driver().items()}
        log.members, log.dropped, log.spent = committee.summary()["members"], committee.dropped, spent
        return Submit(request_id=self.request_id(), ranking=ranking)

    def _per_patient_price(self, card: WorldCard) -> float:
        return card.prices.recruit_per_patient + sum(
            f.assay_price for f in card.features if f.timing is Timing.BASELINE
        )

    def _expected_drop(self, feats: Features, committee: Committee, batch: int, draws: int = 2) -> float:
        """Expected drop in disagreement from ``batch`` more rows.

        The rows are simulated by resampling the revealed rows, so the estimate
        carries the sharpening of the weights with n, not new information.
        """
        n = feats.n
        after = []
        for _ in range(draws):
            extra = self._rng.choice(n, size=batch, replace=True)
            rows = np.concatenate([np.arange(n), extra])
            aug = Features(feats.ids, feats.types, feats.x[rows], feats.y[rows], tuple(feats.stratum[i] for i in rows))
            after.append(self._fit(aug).disagreement)
        return max(0.0, committee.disagreement - float(np.mean(after)))

    def _stratum(self, card: WorldCard, view: EpisodeView, committee: Committee | None, feats: Features) -> str | None:
        """The stratum with patients left whose members' predictions disagree most."""
        sizes = card.stratum_sizes or {s: card.n_pool for s in card.strata}
        left = {s: sizes[s] - view.stratum.count(s) for s in card.strata}
        open_strata = [s for s in card.strata if left[s] > 0]
        if not open_strata:
            return None
        if committee is None or len(open_strata) == 1 or feats.n == 0:
            return max(open_strata, key=lambda s: left[s])
        preds = committee.member_predictions(feats)
        spread = preds.std(axis=0)
        by_stratum = {s: float(np.mean([v for v, st in zip(spread, feats.stratum) if st == s] or [0.0])) for s in open_strata}
        return max(open_strata, key=lambda s: (by_stratum[s], left[s]))

    # ------------------------------------------------------------------ actions

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        if view.mode is Mode.FULL_ACCESS:
            log = self.logs[card.world_id] = EpisodeLog(card.world_id, "full_access", n_rows=len(view.rows))
            committee = self._fit(self._features(card, view))
            log.disagreement.append(committee.disagreement)
            return self._submit(card, committee, log, view.spent)

        if not view.rows:
            self.logs[card.world_id] = EpisodeLog(card.world_id, "sequential")
            self._committee, self._rows = None, set()
            self._random_stop = int(self._rng.integers(1, 5))
            self._initial_pending = list(card.strata)
        log = self.logs[card.world_id]
        if self._initial_pending:
            return self._recruit(card, view, self._initial_pending.pop(0), self._initial_count(card))
        baseline = self._baseline_ids(card)
        cols = [view.feature_ids.index(f) for f in baseline]
        if np.isnan(view.x[:, cols]).all(axis=1).any():
            return Assay(request_id=self.request_id(), feature_ids=baseline)

        feats = self._features(card, view)
        new_rows = [i for i, r in enumerate(view.rows) if r not in self._rows]
        if self._committee is not None and new_rows:
            preds = self._committee.predict(
                Features(feats.ids, feats.types, feats.x[new_rows], feats.y[new_rows], tuple(feats.stratum[i] for i in new_rows))
            )
            log.held_out.extend((self.policy.report(float(p)), int(y)) for p, y in zip(preds, feats.y[new_rows]))
        self._rows = set(view.rows)
        committee = self._fit(feats)
        self._committee = committee
        log.disagreement.append(committee.disagreement)
        log.n_rows = feats.n

        stratum = self._stratum(card, view, committee, feats)
        if all(h.is_null for h in committee.members) and stratum is not None and self._affordable(card, view):
            # No hypothesis survives: the only repair is more data. The step earns no shaping reward.
            log.resynthesis_steps.append(len(log.disagreement) - 1)
            return self._recruit(card, view, stratum, self.policy.batch)
        if stratum is None or self._should_stop(card, view, feats, committee, log):
            return self._submit(card, committee, log, view.spent)
        return self._recruit(card, view, stratum, self.policy.batch)

    def _initial_count(self, card: WorldCard) -> int:
        return max(1, self.policy.n_initial // len(card.strata))

    def _recruit(self, card: WorldCard, view: EpisodeView, stratum: str | None, count: int) -> Recruit:
        sizes = card.stratum_sizes or {s: card.n_pool for s in card.strata}
        if stratum is None:
            stratum = card.strata[0]
        left = sizes[stratum] - view.stratum.count(stratum)
        return Recruit(request_id=self.request_id(), count=max(1, min(count, left)), stratum=stratum)

    def _affordable(self, card: WorldCard, view: EpisodeView) -> bool:
        price = self.policy.batch * self._per_patient_price(card)
        return view.spent + price <= min(card.budget, self.policy.spend_cap * card.budget) + 1e-9

    def _should_stop(self, card: WorldCard, view: EpisodeView, feats: Features, committee: Committee, log: EpisodeLog) -> bool:
        price = self.policy.batch * self._per_patient_price(card)
        if not self._affordable(card, view):
            return True
        if self.policy.acquisition == "all":
            return False
        if self.policy.acquisition == "random":
            return len(log.disagreement) >= self._random_stop
        drop = self._expected_drop(feats, committee, self.policy.batch)
        log.expected_drop.append(drop)
        return drop / (price / 1000.0) < self.policy.stop_threshold


class StagedCommitteeAgent(Agent):
    """The staged design of the reference template: recruit evenly across strata in fixed batches,
    assay every feature, and submit when the same non-empty list appears twice or the pool is spent."""

    def __init__(self, policy: Policy = Policy(), logs: dict[str, EpisodeLog] | None = None, batch: int = 60) -> None:
        super().__init__()
        self.name = "committee_staged"
        self.analyst = committee_analyst(policy, logs)
        self.batch = batch
        self._previous: tuple[str, ...] | None = None

    def choose_action(self, card: WorldCard, view: EpisodeView) -> Action:
        from onc_agi.services.kit import analysis_input

        if view.mode is Mode.FULL_ACCESS:
            return Submit(request_id=self.request_id(), ranking=tuple(self.analyst(analysis_input(card, view))))
        if not view.rows:
            self._previous = None
        elif not all(view.measured) or np.isnan(view.x).all(axis=1).any():
            return Assay(request_id=self.request_id(), feature_ids=card.feature_ids())
        else:
            current = tuple(self.analyst(analysis_input(card, view)))
            if len(view.rows) >= card.n_pool or (current and current == self._previous):
                return Submit(request_id=self.request_id(), ranking=current)
            self._previous = current
        sizes = card.stratum_sizes or {s: card.n_pool for s in card.strata}
        left = {s: sizes[s] - view.stratum.count(s) for s in card.strata}
        stratum = max(left, key=lambda s: left[s])
        return Recruit(request_id=self.request_id(), count=min(self.batch // len(card.strata), left[stratum]), stratum=stratum)


def committee_analyst(policy: Policy = Policy(), logs: dict[str, EpisodeLog] | None = None):
    """An ``analyst(AnalysisInput) -> ranking`` for the kit's PipelineAgent and GroupSequentialAgent."""

    def analyst(data) -> list[str]:
        feats = features_from(data.card, data.feature_ids, data.x, data.y, data.stratum)
        committee = fit_committee(feats, policy)
        post = {f.feature_id for f in data.card.features if f.timing is Timing.POST_OUTCOME}
        ranking = tuple(f for f in committee.ranking(policy.tau) if f not in post)
        if logs is not None:
            log = logs.setdefault(data.card.world_id, EpisodeLog(data.card.world_id, data.card.mode.value))
            log.ranking, log.p_signal, log.n_rows = ranking, policy.report(committee.p_signal), feats.n
            log.p_driver = {f: policy.report(v) for f, v in committee.p_driver().items()}
            log.members, log.dropped = committee.summary()["members"], committee.dropped
            log.disagreement.append(committee.disagreement)
        return list(ranking)

    return analyst
