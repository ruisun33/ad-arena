"""
Bidder — the interface participants implement.

The agent sets strategy ONCE PER DAY (not per auction). This is:
- Realistic (real advertisers set rules, not per-query bids)
- LLM-tractable (1 LLM call per day, not 2,500)
- Still strategically rich (30 decisions per episode)

Minimal contract:
    class MyBidder(Bidder):
        def strategy(self, feedback):
            return DailyStrategy(
                keyword_bids={"running shoes": 2.50},
                keyword_headlines={"running shoes": "Best Running Shoes"},
                daily_budget=350.0,
            )
"""

from __future__ import annotations
from abc import ABC, abstractmethod

from ad_arena.core.models import DailyFeedback, DailyStrategy


class Bidder(ABC):
    """Base class for all bidder submissions."""

    name: str = "unnamed"

    @abstractmethod
    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        """
        Called once per day. Return your bidding strategy for today.

        Args:
            feedback: Yesterday's performance report. None on day 0.

        Returns:
            DailyStrategy with keyword bids, headlines, and daily budget.
        """
        ...

    def on_episode_start(self, config: 'EpisodeConfig') -> None:
        """Called once before the episode begins with market info."""
        pass


class EpisodeConfig:
    """Read-only market info given to bidders at episode start."""

    def __init__(
        self,
        keywords: list[str],
        budget: float,
        duration_days: int,
        num_competitors: int,
    ):
        self.keywords = list(keywords)
        self.budget = budget
        self.duration_days = duration_days
        self.num_competitors = num_competitors

    def __repr__(self) -> str:
        return (
            f"EpisodeConfig(keywords={len(self.keywords)}, "
            f"budget=${self.budget:,.0f}, days={self.duration_days}, "
            f"competitors={self.num_competitors})"
        )
