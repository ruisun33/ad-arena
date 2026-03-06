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
import re
from typing import Callable

from ad_arena.agents.bidder import Bidder, EpisodeConfig
from ad_arena.core.models import DailyFeedback, DailyStrategy
from ad_arena.ui.rendering import (
    render_episode_start,
    render_feedback,
    render_strategy_prompt,
)


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

Strategy tips:
- Start by bidding on most keywords to gather data
- After a few days, cut keywords with poor ROAS
- Increase bids on profitable keywords
- Write keyword-specific headlines
- Pace your budget to last the full episode"""


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
    ):
        self.name = name
        self._llm_fn = llm_fn
        self._verbose = verbose
        self._config: EpisodeConfig | None = None
        self._messages: list[dict[str, str]] = []
        self._total_budget = 0.0
        self._total_days = 0

    def on_episode_start(self, config: EpisodeConfig) -> None:
        self._config = config
        self._total_budget = config.budget
        self._total_days = config.duration_days

        # Initialize conversation with system prompt + market info
        market_info = render_episode_start(config)
        self._messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": market_info + "\n\n" + render_strategy_prompt()},
        ]

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        if self._llm_fn is None:
            # Fallback: no LLM configured, return default strategy
            return self._default_strategy()

        if feedback is not None:
            # Add yesterday's report to conversation
            report = render_feedback(feedback, self._total_budget, self._total_days)
            user_msg = report + "\n\n" + render_strategy_prompt()
            self._messages.append({"role": "user", "content": user_msg})

        if self._verbose:
            print(f"\n--- LLM INPUT (day {feedback.day + 1 if feedback else 0}) ---")
            print(self._messages[-1]["content"][:500])

        # Call LLM
        try:
            response = self._llm_fn(self._messages)
        except Exception as e:
            if self._verbose:
                print(f"LLM call failed: {e}")
            return self._default_strategy()

        if self._verbose:
            print(f"\n--- LLM OUTPUT ---")
            print(response[:500])

        # Add response to conversation history
        self._messages.append({"role": "assistant", "content": response})

        # Parse response into DailyStrategy
        return self._parse_response(response)

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
