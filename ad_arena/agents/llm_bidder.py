"""
LLM Bidder — adapter that lets any LLM play the ad auction game.

Renders observations as natural language, sends to LLM, parses JSON response.

Usage:
    from ad_arena.llm_bidder import LLMBidder

    # With any callable that takes messages and returns text
    def call_llm(messages: list[dict]) -> str:
        # Your LLM API call here
        ...

    bidder = LLMBidder(name="GPT-4o", llm_fn=call_llm)
    result = run_episode(bidder)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from ad_arena.agents.bidder import Bidder, EpisodeConfig
from ad_arena.core.models import DailyFeedback, DailyStrategy
from ad_arena.ui.rendering import (
    render_episode_start,
    render_feedback,
    render_strategy_prompt,
)

logger = logging.getLogger(__name__)


# Type for the LLM callable: takes list of messages, returns string
LLMCallable = Callable[[list[dict[str, str]]], str]


SYSTEM_PROMPT = """You are an expert advertising campaign manager competing in a simulated search ads auction.

Your goal is to maximize total profit (revenue from conversions minus ad spend) over the episode.

Key mechanics:
- You bid on keywords. Higher bids win more auctions but cost more.
- You only pay when someone clicks (CPC model). Price is set by second-price auction.
- Writing ad headlines that include the keyword words increases your click-through rate.
- Different user segments respond to different messaging (e.g., "sale" appeals to budget shoppers).
- The platform's quality score rewards ads with high CTR — good ads get cheaper clicks over time.
- Some keywords are profitable, some are traps. Use the performance data to figure out which.
- Keywords with zero impressions may need HIGHER bids to clear the auction reserve or beat competitors.

Strategy tips:
- Start by bidding on most keywords to gather data
- If a keyword gets zero impressions, try INCREASING the bid significantly — you may be below the auction floor
- After a few days, cut keywords with poor ROAS (below 1.0x after 3+ days of data)
- Increase bids on profitable keywords (ROAS > 2.0x)
- Write keyword-specific headlines that include the exact keyword text
- Pace your budget to last the full episode
- High-intent transactional keywords (e.g., "buy X", specific product queries) often have the best conversion rates — prioritize discovering these
- Don't over-concentrate on a single keyword — diversify across profitable keywords to maximize total volume
- Track cumulative ROAS per keyword, not just daily — single-day data is noisy"""


class LLMBidder(Bidder):
    """
    Adapter that wraps any LLM as a Bidder.

    Args:
        name: Display name for the leaderboard.
        llm_fn: Callable that takes a list of chat messages
                 [{"role": "system"|"user"|"assistant", "content": str}]
                 and returns the assistant's response as a string.
        verbose: If True, print LLM inputs/outputs for debugging.
    """

    def __init__(
        self,
        name: str = "LLM",
        llm_fn: LLMCallable | None = None,
        verbose: bool = False,
        log_dir: str | None = "logs",
    ):
        self.name = name
        self._llm_fn = llm_fn
        self._verbose = verbose
        self._log_dir = Path(log_dir) if log_dir else None
        self._config: EpisodeConfig | None = None
        self._messages: list[dict[str, str]] = []
        self._interaction_log: list[dict] = []
        self._total_budget = 0.0
        self._total_days = 0
        # Cumulative per-keyword stats
        self._cumulative_stats: dict[str, dict] = {}
        # Strategy history for context compression
        self._strategy_history: list[dict] = []

    def on_episode_start(self, config: EpisodeConfig) -> None:
        self._config = config
        self._total_budget = config.budget
        self._total_days = config.duration_days
        # Reset per-episode state
        self._cumulative_stats = {}
        self._strategy_history = []
        self._interaction_log = []
        self._log_timestamp = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S")

        # Initialize conversation with system prompt + market info
        market_info = render_episode_start(config)
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": market_info + "\n\n" + render_strategy_prompt()},
        ]

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        if self._llm_fn is None:
            return self._default_strategy()

        day = feedback.day + 1 if feedback else 0

        if feedback is not None:
            # Update cumulative keyword stats
            self._update_cumulative_stats(feedback)

            # Build compact message: feedback + cumulative stats + strategy history
            report = render_feedback(
                feedback, self._total_budget, self._total_days,
                cumulative_stats=self._cumulative_stats,
            )
            history_summary = self._render_strategy_history()
            user_msg = report
            if history_summary:
                user_msg += "\n" + history_summary
            user_msg += "\n\n" + render_strategy_prompt()

            # Use sliding window: system + market intro + latest turn only
            # This prevents context from growing unboundedly
            self._messages = [
                self._messages[0],  # system prompt
                self._messages[1],  # episode start + first strategy prompt
            ]
            if len(self._strategy_history) > 0:
                # Add the latest feedback as a new user message
                self._messages.append({"role": "user", "content": user_msg})

        if self._verbose:
            print(f"\n--- LLM INPUT (day {day}) ---")
            print(self._messages[-1]["content"][:500])

        # Call LLM
        try:
            response = self._llm_fn(self._messages)
        except Exception as e:
            if self._verbose:
                print(f"LLM call failed: {e}")
            print(f"[{self.name}] Day {day}: LLM call failed, using fallback strategy")
            self._log_interaction(day, self._messages[-1]["content"], None, error=str(e))
            return self._default_strategy()

        if self._verbose:
            print(f"\n--- LLM OUTPUT ---")
            print(response[:500])

        self._log_interaction(day, self._messages[-1]["content"], response)
        logger.info("[%s] Day %d: LLM responded (%d chars)", self.name, day, len(response))

        # Parse response and record in strategy history
        strategy = self._parse_response(response)
        self._strategy_history.append({
            "day": day,
            "profit": feedback.profit if feedback else 0,
            "spend": feedback.spend if feedback else 0,
            "reasoning": strategy.reasoning[:150] if strategy.reasoning else "",
            "top_bids": dict(sorted(
                strategy.keyword_bids.items(),
                key=lambda x: x[1], reverse=True,
            )[:5]),
        })

        return strategy

    def _parse_response(self, text: str) -> DailyStrategy:
        """Extract DailyStrategy from LLM text output."""
        # Try to find JSON block
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
        else:
            # Try raw JSON
            raw = text

        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find any JSON object in the text
            obj_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if obj_match:
                try:
                    d = json.loads(obj_match.group(0))
                except json.JSONDecodeError:
                    return self._default_strategy()
            else:
                return self._default_strategy()

        return DailyStrategy(
            keyword_bids={str(k): float(v) for k, v in d.get("keyword_bids", {}).items()},
            keyword_headlines={str(k): str(v) for k, v in d.get("keyword_headlines", {}).items()},
            daily_budget=float(d.get("daily_budget", float("inf"))),
            reasoning=str(d.get("reasoning", "")),
            audience_modifiers={str(k): float(v) for k, v in d.get("audience_modifiers", {}).items()},
            daypart_modifiers={str(k): float(v) for k, v in d.get("daypart_modifiers", {}).items()},
            keyword_variants={str(k): [str(h) for h in v] for k, v in d.get("keyword_variants", {}).items()},
        )

    def _default_strategy(self) -> DailyStrategy:
        """Fallback strategy if LLM fails."""
        if self._config is None:
            return DailyStrategy()
        return DailyStrategy(
            keyword_bids={kw: 2.00 for kw in self._config.keywords},
            daily_budget=self._total_budget / max(self._total_days, 1),
        )

    def _update_cumulative_stats(self, feedback: DailyFeedback) -> None:
        """Accumulate per-keyword stats across days."""
        for kw, stats in feedback.keyword_stats.items():
            if kw not in self._cumulative_stats:
                self._cumulative_stats[kw] = {
                    "impressions": 0, "clicks": 0, "conversions": 0,
                    "spend": 0.0, "revenue": 0.0,
                }
            cs = self._cumulative_stats[kw]
            cs["impressions"] += stats.impressions
            cs["clicks"] += stats.clicks
            cs["conversions"] += stats.conversions
            cs["spend"] += stats.spend
            cs["revenue"] += stats.revenue

    def _render_strategy_history(self) -> str:
        """Render a compact summary of past strategies for context."""
        if not self._strategy_history:
            return ""
        lines = ["Strategy History (recent days):"]
        # Show last 7 days to keep it compact
        recent = self._strategy_history[-7:]
        for entry in recent:
            top_kws = ", ".join(
                f'"{k}"=${v:.2f}' for k, v in list(entry["top_bids"].items())[:3]
            )
            lines.append(
                f"  Day {entry['day']}: profit=${entry['profit']:,.2f}, "
                f"spend=${entry['spend']:,.2f} | top bids: {top_kws}"
            )
            if entry["reasoning"]:
                lines.append(f"    → {entry['reasoning']}")
        return "\n".join(lines)

    def _log_interaction(self, day: int, prompt: str, response: str | None, error: str | None = None) -> None:
        """Append one turn to the interaction log and flush to disk."""
        entry = {"day": day, "prompt": prompt, "response": response, "error": error}
        self._interaction_log.append(entry)
        self._save_log()

    def _save_log(self) -> None:
        """Write the full interaction log to logs/<name>_<timestamp>.json."""
        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)
        ts = getattr(self, "_log_timestamp", "unknown")
        path = self._log_dir / f"{self.name}_{ts}.json"
        path.write_text(json.dumps(self._interaction_log, indent=2), encoding="utf-8")
