"""
Simulation engine — models the search ads platform pipeline.

The pipeline for each query:

    Query + User Context
         │
    ┌────▼─────────────────────────────────────────┐
    │  1. RETRIEVAL                                 │
    │  Find all candidate ads that match this query │
    │  (keyword match + budget eligibility)         │
    └────┬─────────────────────────────────────────┘
         │  candidates[]
    ┌────▼─────────────────────────────────────────┐
    │  2. RANKING                                   │
    │  ad_rank = bid × pCTR                         │
    │  Sort by ad_rank descending                   │
    └────┬─────────────────────────────────────────┘
         │  ranked candidates[]
    ┌────▼─────────────────────────────────────────┐
    │  3. PRICING (generalized second-price)        │
    │  price = next_ad_rank / winner_pCTR + ε       │
    └────┬─────────────────────────────────────────┘
         │  winner + price
    ┌────▼─────────────────────────────────────────┐
    │  4. USER RESPONSE (ground truth)              │
    │  Simulate actual click and conversion         │
    └──────────────────────────────────────────────┘

Bidders set strategy once per day (keyword bids + headlines + daily budget).
The engine applies that strategy to every auction that day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ad_arena.core.models import (
    AuctionOutcome,
    DailyStrategy,
    KeywordDayStats,
    Keyword,
    UserProfile,
)
from ad_arena.core.user import UserSimulator


# ── Per-bidder state ──────────────────────────────────────────

@dataclass
class BidderState:
    """Mutable per-bidder state tracked by the engine."""
    name: str
    budget_remaining: float
    daily_spend: float = 0.0
    # Current day's strategy (set before each day)
    strategy: DailyStrategy = field(default_factory=DailyStrategy)
    # Historical performance (used by pCTR model)
    keyword_impressions: dict[str, int] = field(default_factory=dict)
    keyword_clicks: dict[str, int] = field(default_factory=dict)
    # per-keyword yesterday stats
    yesterday: dict[str, KeywordDayStats] = field(default_factory=dict)


# ── Candidate (internal, produced by retrieval) ───────────────

@dataclass
class AdCandidate:
    """An ad eligible for the auction. Produced by retrieval stage."""
    bidder_name: str
    keyword: Keyword
    bid: float
    ad_headline: str = ""
    predicted_ctr: float = 0.0
    ad_rank: float = 0.0


# ── Stage 1: Retrieval ───────────────────────────────────────

def _get_daypart_modifier(daypart_modifiers: dict[str, float], hour: int) -> float:
    """Return the daypart modifier for the given hour, or 1.0 if none matches.

    Keys are "HH-HH" range strings parsed as start <= hour < end.
    """
    for range_str, modifier in daypart_modifiers.items():
        try:
            start_s, end_s = range_str.split("-")
            start, end = int(start_s), int(end_s)
            if start <= hour < end:
                return modifier
        except (ValueError, AttributeError):
            continue
    return 1.0


class Retriever:
    """
    Find candidate ads for a query by looking up each bidder's
    pre-set daily strategy.
    """

    def retrieve(
        self,
        keyword: Keyword,
        bidder_states: dict[str, BidderState],
        user_segment: str = "",
        hour: int = 0,
    ) -> list[AdCandidate]:
        """Collect candidates from all eligible bidders for this keyword.

        Applies audience and daypart modifiers multiplicatively to the base bid.
        Negative modifiers are clamped to 0.0; absent modifiers default to 1.0.
        """
        candidates: list[AdCandidate] = []

        for name, st in bidder_states.items():
            # Budget eligibility
            if st.budget_remaining <= 0:
                continue
            if st.daily_spend >= st.strategy.daily_budget:
                continue

            # Look up bid from daily strategy
            bid = st.strategy.keyword_bids.get(keyword.text, 0.0)
            if bid <= 0:
                continue

            # Apply audience modifier
            aud_mod = st.strategy.audience_modifiers.get(user_segment, 1.0)
            aud_mod = max(aud_mod, 0.0)

            # Apply daypart modifier
            day_mod = _get_daypart_modifier(st.strategy.daypart_modifiers, hour)
            day_mod = max(day_mod, 0.0)

            effective_bid = bid * aud_mod * day_mod
            if effective_bid <= 0:
                continue

            headline = st.strategy.keyword_headlines.get(keyword.text, "")

            candidates.append(AdCandidate(
                bidder_name=name,
                keyword=keyword,
                bid=effective_bid,
                ad_headline=headline[:90] if headline else "",
            ))

        return candidates



# ── Stage 2: Ranking (pCTR model + ad_rank) ──────────────────

class PredictedCTRModel:
    """
    The platform's pCTR model — predicts click probability for an ad.

    ad_rank = bid × pCTR maximizes expected revenue per impression
    in a CPC auction. This IS the quality score.
    """

    def __init__(self, rng: np.random.RandomState):
        self._rng = rng

    def predict(self, keyword: Keyword, bidder_state: BidderState) -> float:
        """Predict CTR for this bidder-keyword pair. Returns P(click)."""
        intent_prior = {
            "transactional": 0.22,
            "commercial": 0.15,
            "informational": 0.06,
        }
        prior = intent_prior.get(keyword.intent, 0.12)

        imps = bidder_state.keyword_impressions.get(keyword.text, 0)
        clicks = bidder_state.keyword_clicks.get(keyword.text, 0)

        if imps >= 20:
            actual_ctr = clicks / imps
            data_weight = min(imps / 200, 0.8)
            estimate = data_weight * actual_ctr + (1 - data_weight) * prior
        else:
            estimate = prior

        noise = self._rng.normal(0, 0.02)
        estimate += noise
        return float(np.clip(estimate, 0.01, 0.50))


class Ranker:
    """Score and sort candidates: ad_rank = bid × pCTR."""

    def __init__(self, pctr_model: PredictedCTRModel, reserve_price: float):
        self._pctr_model = pctr_model
        self._reserve_price = reserve_price

    def rank(
        self,
        candidates: list[AdCandidate],
        bidder_states: dict[str, BidderState],
    ) -> list[AdCandidate]:
        for c in candidates:
            c.predicted_ctr = self._pctr_model.predict(
                c.keyword, bidder_states[c.bidder_name],
            )
            c.ad_rank = c.bid * c.predicted_ctr

        eligible = [c for c in candidates if c.ad_rank >= self._reserve_price]
        eligible.sort(key=lambda c: c.ad_rank, reverse=True)
        return eligible


# ── Stage 3: Pricing ─────────────────────────────────────────

class Pricer:
    """Generalized second-price: price = next_ad_rank / winner_pCTR + ε."""

    def __init__(self, reserve_price: float):
        self._reserve_price = reserve_price

    def price(self, ranked: list[AdCandidate]) -> float | None:
        if not ranked:
            return None
        winner = ranked[0]
        if len(ranked) > 1:
            runner_up = ranked[1]
            price = (runner_up.ad_rank / winner.predicted_ctr) + 0.01
        else:
            price = (self._reserve_price / winner.predicted_ctr) + 0.01
        return min(price, winner.bid)


# ── Main Engine ───────────────────────────────────────────────

class AuctionEngine:
    """Orchestrates: retrieval → ranking → pricing → user response."""

    def __init__(
        self,
        keywords: list[Keyword],
        reserve_price: float = 0.10,
        seed: int = 42,
    ):
        self.keywords = keywords
        self.reserve_price = reserve_price
        self.rng = np.random.RandomState(seed)

        self._retriever = Retriever()
        self._pctr_model = PredictedCTRModel(np.random.RandomState(seed + 100))
        self._ranker = Ranker(self._pctr_model, reserve_price)
        self._pricer = Pricer(reserve_price)
        self._user_sim = UserSimulator(seed=seed + 200)

    def simulate_day(
        self,
        day: int,
        days_remaining: int,
        bidder_states: dict[str, BidderState],
    ) -> list[AuctionOutcome]:
        """Simulate all auctions for one day."""
        self._user_sim.reset_daily()
        outcomes: list[AuctionOutcome] = []

        for kw in self.keywords:
            volume = self._sample_volume(kw)
            for _ in range(volume):
                hour = self._sample_hour()
                user = self._user_sim.sample_user(kw)
                outcome = self._run_query(kw, user, hour, day, bidder_states)
                outcomes.append(outcome)

        return outcomes

    def _sample_volume(self, kw: Keyword) -> int:
        noise = self.rng.lognormal(mean=0, sigma=0.15)
        return max(1, int(kw.daily_volume * noise))

    def _sample_hour(self) -> int:
        if self.rng.random() < 0.5:
            h = self.rng.normal(10.0, 2.5)
        else:
            h = self.rng.normal(20.0, 2.5)
        return int(h) % 24

    def _run_query(
        self,
        kw: Keyword,
        user: UserProfile,
        hour: int,
        day: int,
        states: dict[str, BidderState],
    ) -> AuctionOutcome:
        no_winner = AuctionOutcome(
            keyword=kw.text, winner_id="__none__", price=0,
            clicked=False, converted=False, revenue=0,
            hour=hour, day=day, predicted_ctr=0,
        )

        # Stage 1: Retrieval
        candidates = self._retriever.retrieve(kw, states, user_segment=user.segment, hour=hour)
        if not candidates:
            return no_winner

        # Stage 2: Ranking
        ranked = self._ranker.rank(candidates, states)
        if not ranked:
            return no_winner

        # Stage 3: Pricing
        price = self._pricer.price(ranked)
        if price is None:
            return no_winner

        winner = ranked[0]
        st = states[winner.bidder_name]
        price = min(price, st.budget_remaining)

        # Stage 3.5: Ad Variant Selection
        variant_index: int | None = None
        ad_headline = winner.ad_headline
        variants = st.strategy.keyword_variants.get(kw.text, [])
        if variants:
            variant_index = int(self.rng.randint(0, len(variants)))
            ad_headline = variants[variant_index]

        # Track impression for fatigue
        self._user_sim.record_impression(winner.bidder_name)

        # Stage 4: User Response (uses selected headline for CTR)
        clicked = self._user_sim.simulate_click(
            kw, user, ad_headline, bidder_name=winner.bidder_name,
        )
        converted = False
        revenue = 0.0

        if clicked:
            st.budget_remaining -= price
            st.daily_spend += price
            st.keyword_impressions[kw.text] = st.keyword_impressions.get(kw.text, 0) + 1
            st.keyword_clicks[kw.text] = st.keyword_clicks.get(kw.text, 0) + 1
            converted = self._user_sim.simulate_conversion(kw, user)
            if converted:
                revenue = kw.avg_order_value
        else:
            st.keyword_impressions[kw.text] = st.keyword_impressions.get(kw.text, 0) + 1

        return AuctionOutcome(
            keyword=kw.text,
            winner_id=winner.bidder_name,
            price=price if clicked else 0,
            clicked=clicked,
            converted=converted,
            revenue=revenue,
            hour=hour,
            day=day,
            predicted_ctr=winner.predicted_ctr,
            variant_index=variant_index,
            ad_headline=ad_headline,
        )
