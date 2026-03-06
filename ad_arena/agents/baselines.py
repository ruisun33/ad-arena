"""
Baseline bidders — reference implementations and benchmarks.
"""

from __future__ import annotations

from ad_arena.agents.bidder import Bidder, EpisodeConfig
from ad_arena.core.models import DailyFeedback, DailyStrategy


class SimpleBidder(Bidder):
    """Bids a fixed amount on every keyword. No headlines. The simplest strategy."""

    def __init__(self, base_bid: float = 1.50):
        self.name = f"SimpleBidder(${base_bid:.2f})"
        self._base_bid = base_bid
        self._keywords: list[str] = []

    def on_episode_start(self, config: EpisodeConfig) -> None:
        self._keywords = config.keywords

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        return DailyStrategy(
            keyword_bids={kw: self._base_bid for kw in self._keywords},
        )


class BudgetPacingBidder(Bidder):
    """Spreads budget evenly across days."""

    def __init__(self, base_bid: float = 2.00):
        self.name = "BudgetPacer"
        self._base_bid = base_bid
        self._keywords: list[str] = []
        self._budget = 0.0
        self._days = 30

    def on_episode_start(self, config: EpisodeConfig) -> None:
        self._keywords = config.keywords
        self._budget = config.budget
        self._days = config.duration_days

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        if feedback is not None:
            remaining = feedback.budget_remaining
            days_left = self._days - feedback.day - 1
        else:
            remaining = self._budget
            days_left = self._days

        daily_cap = remaining / max(days_left, 1)

        return DailyStrategy(
            keyword_bids={kw: self._base_bid for kw in self._keywords},
            daily_budget=daily_cap,
        )


class KeywordValueBidder(Bidder):
    """
    Adjusts bids per keyword based on observed ROAS.
    Uses EMA to smooth noisy daily signals.
    """

    def __init__(self, base_bid: float = 2.50, target_roas: float = 1.5):
        self.name = "KeywordValue"
        self._base_bid = base_bid
        self._target_roas = target_roas
        self._keyword_bids: dict[str, float] = {}
        self._keyword_roas_ema: dict[str, float] = {}
        self._alpha = 0.3
        self._budget = 0.0
        self._days = 30

    def on_episode_start(self, config: EpisodeConfig) -> None:
        self._keyword_bids = {kw: self._base_bid for kw in config.keywords}
        self._keyword_roas_ema = {}
        self._budget = config.budget
        self._days = config.duration_days

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        if feedback is not None:
            # Update bids based on yesterday's performance
            for kw, stats in feedback.keyword_stats.items():
                if stats.spend > 0 and stats.clicks >= 1:
                    roas = stats.revenue / stats.spend
                    if kw in self._keyword_roas_ema:
                        self._keyword_roas_ema[kw] = (
                            self._alpha * roas + (1 - self._alpha) * self._keyword_roas_ema[kw]
                        )
                    else:
                        self._keyword_roas_ema[kw] = roas

                    ema = self._keyword_roas_ema[kw]
                    current = self._keyword_bids.get(kw, self._base_bid)

                    if ema > self._target_roas * 1.5:
                        self._keyword_bids[kw] = current * 1.20
                    elif ema > self._target_roas:
                        self._keyword_bids[kw] = current * 1.05
                    elif ema < 0.8:
                        self._keyword_bids[kw] = current * 0.70
                    elif ema < self._target_roas:
                        self._keyword_bids[kw] = current * 0.90

                    self._keyword_bids[kw] = max(0.10, min(self._keyword_bids[kw], 15.0))

            remaining = feedback.budget_remaining
            days_left = self._days - feedback.day - 1
        else:
            remaining = self._budget
            days_left = self._days

        daily_cap = remaining / max(days_left, 1)

        return DailyStrategy(
            keyword_bids=dict(self._keyword_bids),
            daily_budget=daily_cap,
        )
