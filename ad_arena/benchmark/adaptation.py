"""Adaptation metrics — measures LLM learning behavior from daily logs."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from ad_arena.core.models import DailyFeedback, DailyStrategy


@dataclass
class AdaptationMetrics:
    """Metrics derived from daily logs measuring LLM learning behavior."""

    learning_rate: float
    strategy_volatility: list[float]
    learning_days: int
    optimizing_days: int
    profit_trajectory: list[float]
    convergence_day: int | None
    keyword_convergence_day: int | None


def _active_keywords(strategy: DailyStrategy) -> set[str]:
    """Return the set of keywords with a positive bid."""
    return {k for k, v in strategy.keyword_bids.items() if v > 0}


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    """1 - |A ∩ B| / |A ∪ B|.  Returns 0.0 for two empty sets."""
    union = a | b
    if not union:
        return 0.0
    return 1.0 - len(a & b) / len(union)


def _normalized_bid_change(
    bids_curr: dict[str, float],
    bids_prev: dict[str, float],
) -> float:
    """Mean of |bid_d[k] - bid_{d-1}[k]| / max(bid_{d-1}[k], 0.01) for shared keywords."""
    shared = set(bids_curr) & set(bids_prev)
    if not shared:
        return 0.0
    total = 0.0
    for k in shared:
        prev = max(bids_prev[k], 0.01)
        total += abs(bids_curr[k] - bids_prev[k]) / prev
    return total / len(shared)


def _compute_volatility(
    strategies: list[DailyStrategy],
) -> list[float]:
    """Compute per-day strategy volatility (day 1 onward).

    volatility(d) = jaccard_distance(keywords_d, keywords_{d-1})
                  + normalized_bid_change(bids_d, bids_{d-1})
    """
    volatility: list[float] = []
    for i in range(1, len(strategies)):
        kw_prev = _active_keywords(strategies[i - 1])
        kw_curr = _active_keywords(strategies[i])
        jd = _jaccard_distance(kw_curr, kw_prev)
        nbc = _normalized_bid_change(
            strategies[i].keyword_bids,
            strategies[i - 1].keyword_bids,
        )
        volatility.append(jd + nbc)
    return volatility


def _compute_learning_rate(profit_trajectory: list[float]) -> float:
    """(mean(profit[20:30]) - mean(profit[0:10])) / abs(mean(profit[0:10])).

    Returns 0.0 when fewer than 10 early or late days exist, or early mean is zero.
    """
    early = profit_trajectory[:10]
    late = profit_trajectory[20:30]
    if len(early) < 10 or len(late) < 10:
        return 0.0
    early_mean = mean(early)
    if early_mean == 0:
        return 0.0
    late_mean = mean(late)
    return (late_mean - early_mean) / abs(early_mean)


def _compute_convergence_day(profit_trajectory: list[float]) -> int | None:
    """First day d where all subsequent profits are within 10% of episode mean.

    Returns None if no such day exists.
    """
    if not profit_trajectory:
        return None
    episode_mean = mean(profit_trajectory)
    if episode_mean == 0:
        # When mean is 0, "within 10%" means all remaining profits must be exactly 0.
        for d in range(len(profit_trajectory)):
            if all(p == 0 for p in profit_trajectory[d:]):
                return d
        return None

    threshold = abs(episode_mean) * 0.1
    for d in range(len(profit_trajectory)):
        if all(abs(p - episode_mean) <= threshold for p in profit_trajectory[d:]):
            return d
    return None


def _compute_keyword_convergence_day(
    strategies: list[DailyStrategy],
) -> int | None:
    """First day d where the set of bid-on keywords doesn't change for the rest."""
    if not strategies:
        return None
    for d in range(len(strategies)):
        kw_d = _active_keywords(strategies[d])
        if all(_active_keywords(strategies[j]) == kw_d for j in range(d + 1, len(strategies))):
            return d
    return None


def compute_adaptation_metrics(
    daily_log: list[DailyFeedback],
    daily_strategies: list[DailyStrategy],
    volatility_threshold: float = 0.3,
) -> AdaptationMetrics:
    """Compute adaptation metrics from an episode's daily logs and strategies."""
    profit_trajectory = [fb.profit for fb in daily_log]

    volatility = _compute_volatility(daily_strategies)

    learning_days = sum(1 for v in volatility if v > volatility_threshold)
    optimizing_days = sum(1 for v in volatility if v <= volatility_threshold)

    return AdaptationMetrics(
        learning_rate=_compute_learning_rate(profit_trajectory),
        strategy_volatility=volatility,
        learning_days=learning_days,
        optimizing_days=optimizing_days,
        profit_trajectory=profit_trajectory,
        convergence_day=_compute_convergence_day(profit_trajectory),
        keyword_convergence_day=_compute_keyword_convergence_day(daily_strategies),
    )
