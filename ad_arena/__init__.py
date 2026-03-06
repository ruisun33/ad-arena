"""
Ad Arena — v1.0  (Search Ads, single HQP slot)

A simulated search-ads auction where AI agents set daily bidding strategies
and compete for the top ad slot. Designed as a benchmark for LLM-based
campaign optimization.

Quick start:
    from ad_arena import run_episode, SimpleBidder
    result = run_episode(SimpleBidder(base_bid=2.0))
    print(result.summary())

LLM usage:
    from ad_arena import run_episode
    from ad_arena.agents.llm_bidder import LLMBidder

    bidder = LLMBidder(name="GPT-4o", llm_fn=my_llm_callable)
    result = run_episode(bidder)
"""

__version__ = "1.0.0"

from ad_arena.agents.bidder import Bidder
from ad_arena.runner import run_episode
from ad_arena.agents.baselines import SimpleBidder

__all__ = ["Bidder", "run_episode", "SimpleBidder"]
