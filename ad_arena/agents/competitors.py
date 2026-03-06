"""
Built-in competitor bots that populate the auction.
These create the competitive landscape the agent competes against.
"""

from __future__ import annotations

import numpy as np

from ad_arena.agents.bidder import Bidder
from ad_arena.core.models import DailyFeedback, DailyStrategy, Keyword


class CompetitorBot(Bidder):
    """Base for rule-based competitors."""

    def __init__(self, name: str, keywords: dict[str, Keyword], rng: np.random.RandomState):
        self.name = name
        self._keywords = keywords
        self._rng = rng
        self._daily_budget = float("inf")


class AggressiveBot(CompetitorBot):
    """Bids high on everything with decent ad copy."""

    def __init__(self, keywords: dict[str, Keyword], rng: np.random.RandomState):
        super().__init__("AggressiveCo", keywords, rng)
        self._daily_budget = 500.0

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        bids = {}
        headlines = {}
        for text, kw in self._keywords.items():
            bids[text] = kw.base_cpc * self._rng.uniform(1.2, 1.8)
            headlines[text] = f"Shop {text.title()} - Free Shipping Today"
        return DailyStrategy(
            keyword_bids=bids,
            keyword_headlines=headlines,
            daily_budget=self._daily_budget,
        )


class ConservativeBot(CompetitorBot):
    """Bids low, skips informational keywords."""

    def __init__(self, keywords: dict[str, Keyword], rng: np.random.RandomState):
        super().__init__("ValueShop", keywords, rng)
        self._daily_budget = 200.0

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        bids = {}
        headlines = {}
        for text, kw in self._keywords.items():
            if kw.intent == "informational":
                continue
            bids[text] = kw.base_cpc * self._rng.uniform(0.4, 0.7)
            headlines[text] = f"Best Deals on {text.title()} - Save 30%"
        return DailyStrategy(
            keyword_bids=bids,
            keyword_headlines=headlines,
            daily_budget=self._daily_budget,
        )


class SmartBot(CompetitorBot):
    """Bids based on expected value per click."""

    def __init__(self, keywords: dict[str, Keyword], rng: np.random.RandomState):
        super().__init__("SmartBidder", keywords, rng)
        self._daily_budget = 350.0

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        bids = {}
        headlines = {}
        for text, kw in self._keywords.items():
            ev_per_click = kw.base_cvr * kw.avg_order_value
            bids[text] = ev_per_click * self._rng.uniform(0.3, 0.6)
            headlines[text] = f"Top Rated {text.title()} - Expert Reviews"
        return DailyStrategy(
            keyword_bids=bids,
            keyword_headlines=headlines,
            daily_budget=self._daily_budget,
        )


def create_default_competitors(
    keywords: dict[str, Keyword],
    seed: int,
) -> list[CompetitorBot]:
    return [
        AggressiveBot(keywords, np.random.RandomState(seed + 1001)),
        ConservativeBot(keywords, np.random.RandomState(seed + 1002)),
        SmartBot(keywords, np.random.RandomState(seed + 1003)),
    ]
