"""Benchmark harness — orchestrates LLM × scenario matrix runs."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ad_arena import __version__
from ad_arena.benchmark.adaptation import compute_adaptation_metrics
from ad_arena.agents.baselines import (
    BudgetPacingBidder,
    KeywordValueBidder,
    SimpleBidder,
)
from ad_arena.agents.bidder import Bidder
from ad_arena.benchmark.results_store import ResultsStore
from ad_arena.runner import run_episode
from ad_arena.core.scenario import Scenario, discover_scenarios
from ad_arena.benchmark.scoring import RunResult, classify_model_type, compute_strategy_optimization_score
from ad_arena.benchmark.seed_manager import SeedManager

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""

    model_name: str
    model_type: str  # "llm" or "baseline"
    bidder_factory: Callable[[], Bidder]
    scenario: Scenario


class BenchmarkHarness:
    """Orchestrates benchmark runs across LLMs × scenarios."""

    def __init__(
        self,
        llm_configs: list[dict] | None = None,
        scenario_names: list[str] | None = None,
        root_seed: int = 42,
    ) -> None:
        self._root_seed = root_seed
        self._seed_manager = SeedManager(root_seed)
        self._results_store = ResultsStore()

        # Discover scenarios
        all_scenarios = discover_scenarios(Path("scenarios"))
        if scenario_names is not None:
            self._scenarios = {
                name: sc
                for name, sc in all_scenarios.items()
                if name in scenario_names
            }
        else:
            self._scenarios = all_scenarios

        if not self._scenarios:
            raise FileNotFoundError(
                "No valid scenarios found. Add YAML files to scenarios/ directory."
            )

        # Build run configs
        self._configs: list[RunConfig] = []
        self._build_baseline_configs()
        self._build_llm_configs(llm_configs or [])

    def _build_baseline_configs(self) -> None:
        """Add baseline bidder configs for every scenario."""
        baselines: list[tuple[str, Callable[[], Bidder]]] = [
            ("SimpleBidder($1.50)", lambda: SimpleBidder(base_bid=1.50)),
            ("BudgetPacer", lambda: BudgetPacingBidder(base_bid=2.00)),
            ("KeywordValue", lambda: KeywordValueBidder(base_bid=2.50)),
        ]
        for scenario in self._scenarios.values():
            for name, factory in baselines:
                self._configs.append(
                    RunConfig(
                        model_name=name,
                        model_type="baseline",
                        bidder_factory=factory,
                        scenario=scenario,
                    )
                )

    def _build_llm_configs(self, llm_configs: list[dict]) -> None:
        """Add LLM bidder configs for every scenario."""
        from ad_arena.agents.llm_bidder import LLMBidder

        for cfg in llm_configs:
            name = cfg["name"]
            llm_fn = cfg["llm_fn"]
            for scenario in self._scenarios.values():
                # Capture loop variables properly
                def _make_factory(n: str, fn: Callable) -> Callable[[], Bidder]:
                    return lambda: LLMBidder(name=n, llm_fn=fn)

                self._configs.append(
                    RunConfig(
                        model_name=name,
                        model_type="llm",
                        bidder_factory=_make_factory(name, llm_fn),
                        scenario=scenario,
                    )
                )

    def run_all(self) -> list[RunResult]:
        """Execute all (model, scenario) pairs and return results."""
        results: list[RunResult] = []

        for i, config in enumerate(self._configs, 1):
            logger.info(
                "Run %d/%d: %s on %s",
                i,
                len(self._configs),
                config.model_name,
                config.scenario.name,
            )
            try:
                result = self._run_single(config)
                results.append(result)
                # Persist result
                try:
                    self._results_store.save(result)
                except OSError:
                    logger.error(
                        "Failed to persist result for %s on %s",
                        config.model_name,
                        config.scenario.name,
                    )
            except Exception:
                logger.exception(
                    "Run failed for %s on %s",
                    config.model_name,
                    config.scenario.name,
                )

        # Update leaderboard with all results
        if results:
            self._results_store.update_leaderboard(results)

        return results

    def _run_single(self, config: RunConfig) -> RunResult:
        """Execute a single run with timing and error handling."""
        scenario = config.scenario
        model_name = config.model_name

        # Get deterministic seed for this (scenario, model) pair
        engine_seed = self._seed_manager.engine_seed(scenario.name, model_name)

        # Create the bidder
        bidder = config.bidder_factory()

        # Wrap LLM bidders to count API calls
        api_call_counter = _APICallCounter()
        if config.model_type == "llm":
            bidder = _wrap_llm_bidder_for_counting(bidder, api_call_counter)

        # Time the run
        start = time.monotonic()
        episode_result = run_episode(bidder, scenario=scenario, seed=engine_seed)
        wall_clock = time.monotonic() - start

        # Compute adaptation metrics
        adaptation = compute_adaptation_metrics(
            episode_result.daily_log,
            episode_result.daily_strategies,
        )

        # Compute strategy optimization score (final 7 days profit)
        so_score = compute_strategy_optimization_score(episode_result.daily_log)

        return RunResult(
            episode_result=episode_result,
            adaptation_metrics=adaptation,
            model_name=model_name,
            model_type=config.model_type,
            scenario_name=scenario.name,
            scenario_hash=scenario.config_hash(),
            root_seed=self._root_seed,
            wall_clock_seconds=wall_clock,
            llm_api_calls=api_call_counter.count,
            software_version=__version__,
            strategy_optimization_score=so_score,
        )


# ── LLM API call counting ────────────────────────────────────


class _APICallCounter:
    """Simple counter for LLM API invocations."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    def increment(self) -> None:
        self.count += 1


def _wrap_llm_bidder_for_counting(
    bidder: Bidder,
    counter: _APICallCounter,
) -> Bidder:
    """Wrap an LLMBidder's llm_fn to count API calls and log failures.

    Returns the same bidder instance with its _llm_fn replaced by a
    counting wrapper. The LLMBidder's own error handling (fallback to
    _default_strategy) remains in place.
    """
    from ad_arena.agents.llm_bidder import LLMBidder

    if not isinstance(bidder, LLMBidder):
        return bidder

    original_fn = bidder._llm_fn
    if original_fn is None:
        return bidder

    def counting_wrapper(messages: list[dict[str, str]]) -> str:
        counter.increment()
        try:
            return original_fn(messages)
        except Exception as exc:
            logger.warning("LLM API call failed: %s", exc)
            raise  # Let LLMBidder handle the fallback

    bidder._llm_fn = counting_wrapper  # type: ignore[attr-defined]
    return bidder
