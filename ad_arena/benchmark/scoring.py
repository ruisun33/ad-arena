"""Scoring and ranking — aggregate results into a leaderboard."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ad_arena.benchmark.adaptation import AdaptationMetrics
from ad_arena.core.models import EpisodeResult


# Baseline bidder class names → model_type "baseline"
_BASELINE_NAMES = {"SimpleBidder", "BudgetPacingBidder", "KeywordValueBidder"}


@dataclass
class RunResult:
    """Extended episode result with benchmark metadata."""

    episode_result: EpisodeResult
    adaptation_metrics: AdaptationMetrics
    model_name: str
    model_type: str  # "llm" or "baseline"
    scenario_name: str
    scenario_hash: str
    root_seed: int
    wall_clock_seconds: float
    llm_api_calls: int
    software_version: str
    strategy_optimization_score: float = 0.0  # profit from final 7 days (days 24–30)


@dataclass
class LeaderboardEntry:
    """One row in the leaderboard."""

    rank: int
    model_name: str
    model_type: str  # "llm" or "baseline"
    aggregate_score: float  # mean of per-scenario scores for the selected suite
    per_scenario_scores: dict[str, float] = field(default_factory=dict)
    campaign_efficiency_score: float = 0.0  # aggregate across scenarios
    strategy_optimization_score: float = 0.0  # aggregate across scenarios
    roas: float = 0.0
    cpa: float = 0.0
    ctr: float = 0.0
    cvr: float = 0.0
    total_conversions: int = 0
    learning_rate: float = 0.0
    learning_days_ratio: float = 0.0
    llm_api_calls: int = 0


def classify_model_type(model_name: str) -> str:
    """Return ``"baseline"`` for known baseline bidders, ``"llm"`` otherwise."""
    # Check if any baseline class name appears in the model name.
    # SimpleBidder names look like "SimpleBidder($1.50)", BudgetPacer, KeywordValue, etc.
    for base in _BASELINE_NAMES:
        if base in model_name:
            return "baseline"
    # Also match the display names used by the baseline instances
    if model_name in {"BudgetPacer", "KeywordValue"}:
        return "baseline"
    return "llm"


def compute_strategy_optimization_score(daily_log: list) -> float:
    """Sum daily profit for the final 7 days (days 24–30, 1-indexed).

    The DailyFeedback.day field is 0-indexed, so days 24–30 (1-indexed)
    correspond to day values >= 23 (0-indexed).
    Returns 0.0 if the episode has fewer than 24 days.
    """
    if len(daily_log) < 24:
        return 0.0
    return sum(fb.profit for fb in daily_log if fb.day >= 23)


def compute_leaderboard(
    results: list[RunResult],
    suite: str = "campaign_efficiency",
) -> list[LeaderboardEntry]:
    """Compute a ranked leaderboard from run results for a given evaluation suite.

    ``suite`` is one of ``"campaign_efficiency"`` or ``"strategy_optimization"``.

    Ranking logic:
    1. Group results by model name.
    2. For each model, take the best result per scenario (by the selected suite's score).
    3. Compute aggregate score = mean of per-scenario scores for the selected suite.
    4. Rank by the selected suite's aggregate score descending.
    5. Average secondary metrics (ROAS, CPA, CTR, CVR, etc.) across scenarios.
    6. Each entry carries both campaign_efficiency_score and strategy_optimization_score aggregates.
    """
    if not results:
        return []

    # ── 1. Group by model name ────────────────────────────────
    by_model: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        by_model[r.model_name].append(r)

    # ── 2. Best result per scenario for each model ────────────
    entries: list[LeaderboardEntry] = []

    for model_name, model_results in by_model.items():
        # Group this model's results by scenario
        by_scenario: dict[str, list[RunResult]] = defaultdict(list)
        for r in model_results:
            by_scenario[r.scenario_name].append(r)

        # Pick best per scenario (by selected suite's score)
        best_per_scenario: dict[str, RunResult] = {}
        for scenario_name, scenario_results in by_scenario.items():
            if suite == "strategy_optimization":
                best_per_scenario[scenario_name] = max(
                    scenario_results,
                    key=lambda r: r.strategy_optimization_score,
                )
            else:
                best_per_scenario[scenario_name] = max(
                    scenario_results,
                    key=lambda r: r.episode_result.total_profit,
                )

        bests = list(best_per_scenario.values())
        n = len(bests)

        # ── 3. Per-scenario scores and aggregate ──────────────
        # Compute per-scenario scores for the selected suite
        if suite == "strategy_optimization":
            per_scenario_scores = {
                sn: rr.strategy_optimization_score
                for sn, rr in best_per_scenario.items()
            }
        else:
            per_scenario_scores = {
                sn: rr.episode_result.total_profit
                for sn, rr in best_per_scenario.items()
            }
        aggregate_score = sum(per_scenario_scores.values()) / n

        # Both suite aggregates (always computed)
        ce_scores = [rr.episode_result.total_profit for rr in bests]
        so_scores = [rr.strategy_optimization_score for rr in bests]
        campaign_efficiency_agg = sum(ce_scores) / n
        strategy_optimization_agg = sum(so_scores) / n

        # ── 5. Average secondary metrics ──────────────────────
        avg_roas = sum(rr.episode_result.roas for rr in bests) / n
        avg_ctr = sum(rr.episode_result.ctr for rr in bests) / n
        avg_cvr = sum(rr.episode_result.cvr for rr in bests) / n
        total_conv = sum(rr.episode_result.total_conversions for rr in bests)

        # CPA: average across scenarios (inf values kept as-is)
        cpa_values = [rr.episode_result.cpa for rr in bests]
        finite_cpas = [c for c in cpa_values if c != float("inf")]
        avg_cpa = sum(finite_cpas) / len(finite_cpas) if finite_cpas else float("inf")

        avg_lr = sum(rr.adaptation_metrics.learning_rate for rr in bests) / n

        # learning_days_ratio = learning_days / total classifiable days
        total_learning = sum(rr.adaptation_metrics.learning_days for rr in bests)
        total_classifiable = sum(
            rr.adaptation_metrics.learning_days + rr.adaptation_metrics.optimizing_days
            for rr in bests
        )
        ld_ratio = total_learning / total_classifiable if total_classifiable > 0 else 0.0

        total_api = sum(rr.llm_api_calls for rr in bests)

        # Determine model type from the first result (consistent within a model)
        model_type = model_results[0].model_type

        entries.append(
            LeaderboardEntry(
                rank=0,  # assigned below
                model_name=model_name,
                model_type=model_type,
                aggregate_score=aggregate_score,
                per_scenario_scores=per_scenario_scores,
                campaign_efficiency_score=campaign_efficiency_agg,
                strategy_optimization_score=strategy_optimization_agg,
                roas=avg_roas,
                cpa=avg_cpa,
                ctr=avg_ctr,
                cvr=avg_cvr,
                total_conversions=total_conv,
                learning_rate=avg_lr,
                learning_days_ratio=ld_ratio,
                llm_api_calls=total_api,
            )
        )

    # ── 4. Rank by aggregate score descending ─────────────────
    entries.sort(key=lambda e: e.aggregate_score, reverse=True)
    for i, entry in enumerate(entries, start=1):
        entry.rank = i

    return entries
