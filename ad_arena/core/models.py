"""Core value objects for the simulation."""

from __future__ import annotations
from dataclasses import dataclass, field


# ── Market ────────────────────────────────────────────────────

@dataclass(frozen=True)
class Keyword:
    """A keyword in the market (immutable config)."""
    text: str
    daily_volume: int          # avg searches / day
    base_cpc: float            # market-average CPC
    base_cvr: float            # base conversion rate given click
    avg_order_value: float     # revenue per conversion
    intent: str = "commercial" # transactional | commercial | informational


@dataclass(frozen=True)
class UserProfile:
    """A user performing a search. Sampled per query."""
    segment: str = "casual"        # young_professional | budget_shopper | enthusiast | casual
    device: str = "desktop"        # mobile | desktop
    purchase_intent: float = 0.5   # 0.0 (browsing) to 1.0 (ready to buy)






# ── Auction result (internal) ─────────────────────────────────

@dataclass
class AuctionOutcome:
    """Result of a single auction."""
    keyword: str
    winner_id: str             # bidder name or "__none__"
    price: float               # CPC charged (second-price)
    clicked: bool
    converted: bool
    revenue: float             # 0 unless converted
    hour: int
    day: int
    predicted_ctr: float = 0.0 # platform's pCTR estimate for the winner
    variant_index: int | None = None  # index into keyword_variants list
    ad_headline: str = ""             # the headline shown


# ── Daily strategy (what the agent outputs each day) ──────────

@dataclass
class DailyStrategy:
    """What the agent decides each day. Applied to all auctions that day."""
    keyword_bids: dict[str, float] = field(default_factory=dict)      # keyword → CPC bid ($0 or absent = skip)
    keyword_headlines: dict[str, str] = field(default_factory=dict)   # keyword → ad headline
    daily_budget: float = float("inf")                                 # total daily spend cap
    reasoning: str = ""                                                # optional, for analysis/logging
    audience_modifiers: dict[str, float] = field(default_factory=dict)    # segment → bid multiplier
    daypart_modifiers: dict[str, float] = field(default_factory=dict)     # "HH-HH" → bid multiplier
    keyword_variants: dict[str, list[str]] = field(default_factory=dict)  # keyword → [headline variants]


# ── Feedback sent to bidders after each day ───────────────────

@dataclass(frozen=True)
class VariantDayStats:
    """Per-variant performance for a keyword on a single day."""
    variant_index: int = 0
    headline: str = ""
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    revenue: float = 0.0


@dataclass(frozen=True)
class DailyFeedback:
    """End-of-day performance summary sent to the bidder."""
    day: int
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float
    profit: float
    budget_remaining: float
    # per-keyword breakdown
    keyword_stats: dict[str, KeywordDayStats] = field(default_factory=dict)
    # per-keyword variant breakdown
    variant_stats: dict[str, list[VariantDayStats]] = field(default_factory=dict)




@dataclass(frozen=True)
class KeywordDayStats:
    impressions: int = 0
    clicks: int = 0
    conversions: int = 0
    spend: float = 0.0
    revenue: float = 0.0


# ── Episode results ───────────────────────────────────────────

@dataclass
class EpisodeResult:
    """Final scorecard for one episode."""
    bidder_name: str
    total_spend: float = 0.0
    total_revenue: float = 0.0
    total_impressions: int = 0
    total_clicks: int = 0
    total_conversions: int = 0
    total_profit: float = 0.0
    days: int = 0
    budget: float = 0.0
    daily_log: list[DailyFeedback] = field(default_factory=list)
    daily_strategies: list[DailyStrategy] = field(default_factory=list)

    @property
    def roas(self) -> float:
        return self.total_revenue / self.total_spend if self.total_spend > 0 else 0.0

    @property
    def cpa(self) -> float:
        return self.total_spend / self.total_conversions if self.total_conversions > 0 else float("inf")

    @property
    def ctr(self) -> float:
        return self.total_clicks / self.total_impressions if self.total_impressions > 0 else 0.0

    @property
    def cvr(self) -> float:
        return self.total_conversions / self.total_clicks if self.total_clicks > 0 else 0.0

    def score(self) -> float:
        """Single leaderboard score: profit (revenue - spend)."""
        return self.total_profit

    def summary(self) -> str:
        lines = [
            f"=== {self.bidder_name} — {self.days}-day episode ===",
            f"Budget: ${self.budget:,.0f}  |  Spend: ${self.total_spend:,.2f}",
            f"Impressions: {self.total_impressions:,}  |  Clicks: {self.total_clicks:,}  |  Conversions: {self.total_conversions:,}",
            f"Revenue: ${self.total_revenue:,.2f}  |  Profit: ${self.total_profit:,.2f}",
            f"ROAS: {self.roas:.2f}x  |  CPA: ${self.cpa:.2f}  |  CTR: {self.ctr:.1%}  |  CVR: {self.cvr:.1%}",
            f"Score: {self.score():,.2f}",
        ]
        return "\n".join(lines)
