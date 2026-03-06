"""
Episode runner — wires bidders to the engine and runs a full episode.

    from ad_arena import run_episode, SimpleBidder
    result = run_episode(SimpleBidder(base_bid=2.0))
    print(result.summary())
"""

from __future__ import annotations

from ad_arena.agents.bidder import Bidder, EpisodeConfig
from ad_arena.agents.competitors import create_default_competitors
from ad_arena.core.engine import AuctionEngine, BidderState
from ad_arena.core.models import (
    DailyFeedback,
    DailyStrategy,
    EpisodeResult,
    KeywordDayStats,
    VariantDayStats,
)
from ad_arena.core.scenario import Scenario, default_scenario


def run_episode(
    bidder: Bidder,
    scenario: Scenario | None = None,
    seed: int | None = None,
) -> EpisodeResult:
    """Run one full episode and return the bidder's results."""
    if scenario is None:
        scenario = default_scenario(seed=seed or 42)
    if seed is not None:
        scenario.seed = seed

    kw_map = {kw.text: kw for kw in scenario.keywords}
    competitors = create_default_competitors(kw_map, scenario.seed)

    engine = AuctionEngine(
        keywords=scenario.keywords,
        reserve_price=scenario.reserve_price,
        seed=scenario.seed,
    )

    # Notify bidder of episode start
    ep_config = EpisodeConfig(
        keywords=scenario.keyword_texts,
        budget=scenario.agent_budget,
        duration_days=scenario.duration_days,
        num_competitors=len(competitors),
    )
    bidder.on_episode_start(ep_config)

    # Initialize states
    agent_state = BidderState(name=bidder.name, budget_remaining=scenario.agent_budget)
    states: dict[str, BidderState] = {bidder.name: agent_state}
    for bot in competitors:
        states[bot.name] = BidderState(name=bot.name, budget_remaining=100_000.0)

    result = EpisodeResult(
        bidder_name=bidder.name,
        budget=scenario.agent_budget,
        days=scenario.duration_days,
    )

    feedback: DailyFeedback | None = None

    for day in range(scenario.duration_days):
        days_remaining = scenario.duration_days - day

        # Reset daily spend
        for st in states.values():
            st.daily_spend = 0.0

        if agent_state.budget_remaining <= 0:
            break

        # Get daily strategy from the agent
        agent_strategy = bidder.strategy(feedback)
        agent_state.strategy = agent_strategy
        result.daily_strategies.append(agent_strategy)

        # Get daily strategies from competitors
        for bot in competitors:
            bot_strategy = bot.strategy(None)  # bots ignore feedback
            states[bot.name].strategy = bot_strategy

        # Simulate the day
        outcomes = engine.simulate_day(day, days_remaining, states)

        # Aggregate agent's results
        day_imps = 0
        day_clicks = 0
        day_convs = 0
        day_spend = 0.0
        day_revenue = 0.0
        kw_stats: dict[str, _KwAccum] = {}
        # Per-variant accumulators keyed by (keyword, variant_index, headline)
        variant_accum: dict[tuple[str, int, str], _VariantAccum] = {}

        for o in outcomes:
            if o.winner_id != bidder.name:
                continue
            day_imps += 1
            kw_a = kw_stats.setdefault(o.keyword, _KwAccum())
            kw_a.impressions += 1

            # Track per-variant stats when variant_index is present
            if o.variant_index is not None:
                vkey = (o.keyword, o.variant_index, o.ad_headline)
                va = variant_accum.setdefault(vkey, _VariantAccum())
                va.impressions += 1

            if o.clicked:
                day_clicks += 1
                day_spend += o.price
                kw_a.clicks += 1
                kw_a.spend += o.price
                if o.variant_index is not None:
                    va = variant_accum[(o.keyword, o.variant_index, o.ad_headline)]
                    va.clicks += 1
                    va.spend += o.price
                if o.converted:
                    day_convs += 1
                    day_revenue += o.revenue
                    kw_a.conversions += 1
                    kw_a.revenue += o.revenue
                    if o.variant_index is not None:
                        va = variant_accum[(o.keyword, o.variant_index, o.ad_headline)]
                        va.conversions += 1
                        va.revenue += o.revenue

        day_profit = day_revenue - day_spend

        kw_day_stats = {
            k: KeywordDayStats(
                impressions=v.impressions, clicks=v.clicks,
                conversions=v.conversions, spend=v.spend, revenue=v.revenue,
            )
            for k, v in kw_stats.items()
        }

        # Build variant_stats grouped by keyword
        variant_stats: dict[str, list[VariantDayStats]] = {}
        for (kw, vidx, headline), va in variant_accum.items():
            variant_stats.setdefault(kw, []).append(
                VariantDayStats(
                    variant_index=vidx,
                    headline=headline,
                    impressions=va.impressions,
                    clicks=va.clicks,
                    conversions=va.conversions,
                    spend=va.spend,
                    revenue=va.revenue,
                )
            )
        # Sort each keyword's variants by variant_index for deterministic order
        for kw in variant_stats:
            variant_stats[kw].sort(key=lambda vs: vs.variant_index)

        agent_state.yesterday = kw_day_stats

        feedback = DailyFeedback(
            day=day,
            impressions=day_imps,
            clicks=day_clicks,
            conversions=day_convs,
            spend=day_spend,
            revenue=day_revenue,
            profit=day_profit,
            budget_remaining=agent_state.budget_remaining,
            keyword_stats=kw_day_stats,
            variant_stats=variant_stats,
        )

        result.total_impressions += day_imps
        result.total_clicks += day_clicks
        result.total_conversions += day_convs
        result.total_spend += day_spend
        result.total_revenue += day_revenue
        result.total_profit += day_profit
        result.daily_log.append(feedback)

    return result


class _KwAccum:
    __slots__ = ("impressions", "clicks", "conversions", "spend", "revenue")
    def __init__(self):
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.spend = 0.0
        self.revenue = 0.0


class _VariantAccum:
    """Accumulator for per-(keyword, variant_index, headline) stats."""
    __slots__ = ("impressions", "clicks", "conversions", "spend", "revenue")
    def __init__(self):
        self.impressions = 0
        self.clicks = 0
        self.conversions = 0
        self.spend = 0.0
        self.revenue = 0.0
