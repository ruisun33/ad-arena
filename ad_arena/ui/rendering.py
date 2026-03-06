"""
Text rendering for observations — makes DailyFeedback readable by LLMs.
"""

from __future__ import annotations

from ad_arena.agents.bidder import EpisodeConfig
from ad_arena.core.models import DailyFeedback



def render_episode_start(config: EpisodeConfig) -> str:
    """Render episode config as text for the LLM's first turn."""
    lines = [
        f"You are managing a search ads campaign for a running shoes store.",
        f"",
        f"Budget: ${config.budget:,.0f} over {config.duration_days} days",
        f"Competitors: {config.num_competitors} other advertisers",
        f"",
        f"Available keywords to bid on:",
    ]
    for kw in config.keywords:
        lines.append(f'  - "{kw}"')

    lines.append("")
    lines.append("Each day you set: keyword bids, ad headlines, and a daily budget cap.")
    lines.append("You only pay when a user clicks your ad (CPC model).")
    lines.append("Higher bids win more auctions but cost more.")
    lines.append("Relevant ad headlines get more clicks (and cheaper prices over time).")
    lines.append("")
    lines.append("Advanced strategies available:")
    lines.append("  - Audience bid modifiers: adjust bids per user segment "
                 "(young_professional, budget_shopper, enthusiast, casual)")
    lines.append("  - Daypart modifiers: adjust bids by hour range (e.g. bid higher during peak hours)")
    lines.append("  - Ad variants: test multiple headlines per keyword to find the best performer")
    lines.append("")
    lines.append("Your goal: maximize total profit (revenue from conversions - ad spend).")
    return "\n".join(lines)




def render_feedback(feedback: DailyFeedback, total_budget: float, total_days: int) -> str:
    """Render daily feedback as natural language for LLM consumption."""
    lines = []

    day_num = feedback.day + 1  # 1-indexed for humans
    days_left = total_days - day_num
    pct_remaining = (feedback.budget_remaining / total_budget) * 100 if total_budget > 0 else 0

    lines.append(f"=== DAY {day_num} PERFORMANCE REPORT ===")
    lines.append("")
    lines.append(
        f"Budget: ${feedback.budget_remaining:,.0f} remaining "
        f"of ${total_budget:,.0f} ({pct_remaining:.0f}%) | "
        f"{days_left} days left"
    )
    lines.append("")

    # Summary
    lines.append("Yesterday's Summary:")
    lines.append(
        f"  Spend: ${feedback.spend:,.2f} | "
        f"Impressions: {feedback.impressions:,} | "
        f"Clicks: {feedback.clicks:,} | "
        f"Conversions: {feedback.conversions}"
    )

    cpc = feedback.spend / feedback.clicks if feedback.clicks > 0 else 0
    ctr = feedback.clicks / feedback.impressions if feedback.impressions > 0 else 0
    cvr = feedback.conversions / feedback.clicks if feedback.clicks > 0 else 0
    cpa = feedback.spend / feedback.conversions if feedback.conversions > 0 else float("inf")

    lines.append(
        f"  CPC: ${cpc:.2f} | CTR: {ctr:.1%} | "
        f"Conv Rate: {cvr:.1%} | CPA: ${cpa:.2f}"
    )
    lines.append(
        f"  Revenue: ${feedback.revenue:,.2f} | "
        f"Profit: ${feedback.profit:,.2f} | "
        f"ROAS: {feedback.revenue / feedback.spend:.2f}x"
        if feedback.spend > 0 else
        f"  Revenue: ${feedback.revenue:,.2f} | Profit: ${feedback.profit:,.2f}"
    )
    lines.append("")

    # Per-keyword breakdown
    if feedback.keyword_stats:
        lines.append("Keyword Breakdown:")
        for kw, stats in sorted(feedback.keyword_stats.items()):
            kw_roas = f"{stats.revenue / stats.spend:.2f}x" if stats.spend > 0 else "n/a"
            flag = ""
            if stats.spend > 0 and stats.revenue < stats.spend:
                flag = "  ← losing money"
            elif stats.spend > 0 and stats.revenue > stats.spend * 2:
                flag = "  ← profitable"
            lines.append(
                f'  "{kw}": ${stats.spend:.2f} spend, '
                f"{stats.clicks} clicks, {stats.conversions} conv, "
                f"ROAS {kw_roas}{flag}"
            )
        lines.append("")

    # Per-variant performance breakdown
    if feedback.variant_stats:
        lines.append("Ad Variant Performance:")
        for kw, variants in sorted(feedback.variant_stats.items()):
            if not variants:
                continue
            lines.append(f'  "{kw}":')
            for v in variants:
                v_roas = f"{v.revenue / v.spend:.2f}x" if v.spend > 0 else "n/a"
                v_ctr = f"{v.clicks / v.impressions:.1%}" if v.impressions > 0 else "n/a"
                lines.append(
                    f"    Variant {v.variant_index} \"{v.headline}\": "
                    f"{v.impressions} imp, {v.clicks} clicks, "
                    f"{v.conversions} conv, ROAS {v_roas}, CTR {v_ctr}"
                )
        lines.append("")

    return "\n".join(lines)




def render_strategy_prompt() -> str:
    """The instruction for what JSON to output."""
    return """Respond with a JSON strategy for today:

```json
{
  "keyword_bids": {
    "keyword text": bid_amount_in_dollars,
    ...
  },
  "keyword_headlines": {
    "keyword text": "Your ad headline (max 90 chars)",
    ...
  },
  "daily_budget": 350.00,
  "reasoning": "Brief explanation of your strategy",

  "audience_modifiers": {
    "young_professional": 1.2,
    "budget_shopper": 0.8,
    "enthusiast": 1.5,
    "casual": 1.0
  },
  "daypart_modifiers": {
    "9-12": 1.3,
    "18-21": 1.2,
    "0-6": 0.5
  },
  "keyword_variants": {
    "keyword text": ["Headline A", "Headline B", "Headline C"],
    ...
  }
}
```

Required fields: keyword_bids, keyword_headlines, daily_budget.
Optional fields:
  - audience_modifiers: bid multipliers per user segment (young_professional, budget_shopper, enthusiast, casual). Default 1.0.
  - daypart_modifiers: bid multipliers per hour range ("HH-HH" format, 24h clock). Default 1.0.
  - keyword_variants: up to 3 headline variants per keyword for A/B testing. Performance reported in daily feedback.

Set bid to 0 or omit a keyword to skip it.
Write headlines that include the keyword words — this improves click-through rate.
Budget your daily spend to last the remaining days."""

