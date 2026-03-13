"""
Example submission: selective keyword targeting with relevant ad copy.

Run with:  arena-run --bidder examples/my_bidder.py
"""

from ad_arena.agents.bidder import Bidder, EpisodeConfig
from ad_arena.core.models import DailyFeedback, DailyStrategy

SKIP = {"how to choose running shoes", "running shoes vs walking shoes",
        "cheap running shoes", "running shoes"}

HEADLINES = {
    "buy running shoes online": "Buy Running Shoes Online - Free 2-Day Shipping",
    "best running shoes": "Best Running Shoes 2025 - Expert Tested & Reviewed",
    "running shoes sale": "Running Shoes Sale - Up to 40% Off Top Brands",
    "running shoes for flat feet": "Running Shoes for Flat Feet - Podiatrist Recommended",
    "waterproof trail running shoes": "Waterproof Trail Running Shoes - Built for Any Terrain",
    "nike pegasus review": "Nike Pegasus Review - Honest Ratings & Best Price",
}


class SelectiveBidder(Bidder):
    name = "SelectiveBidder"

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

        bids = {}
        headlines = {}
        for kw in self._keywords:
            if kw in SKIP:
                continue
            bids[kw] = 4.00
            headlines[kw] = HEADLINES.get(kw, f"Shop {kw.title()} Today")

        return DailyStrategy(
            keyword_bids=bids,
            keyword_headlines=headlines,
            daily_budget=remaining / max(days_left, 1),
        )
