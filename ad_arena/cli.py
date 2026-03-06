"""
CLI entry point for running evaluations.

Usage:
    # Run all baselines
    aag-run

    # Run a custom bidder
    aag-run --bidder path/to/my_bidder.py

    # Custom seed
    aag-run --seed 123
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from ad_arena.agents.baselines import SimpleBidder, BudgetPacingBidder, KeywordValueBidder
from ad_arena.agents.bidder import Bidder
from ad_arena.runner import run_episode
from ad_arena.core.scenario import default_scenario


def load_bidder_from_file(path: str) -> Bidder:
    """Dynamically load a Bidder subclass from a Python file."""
    p = Path(path)
    if not p.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("user_bidder", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the first Bidder subclass that isn't Bidder itself
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, Bidder)
            and attr is not Bidder
        ):
            return attr()

    print(f"Error: no Bidder subclass found in {path}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Ad Arena — run bidder evaluation")
    parser.add_argument("--bidder", type=str, help="Path to a .py file containing a Bidder subclass")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    scenario = default_scenario(seed=args.seed)

    if args.bidder:
        bidders = [load_bidder_from_file(args.bidder)]
    else:
        bidders = [
            SimpleBidder(base_bid=1.00),
            SimpleBidder(base_bid=2.00),
            SimpleBidder(base_bid=3.00),
            BudgetPacingBidder(base_bid=2.00),
            KeywordValueBidder(base_bid=2.00),
        ]

    print(f"Scenario: {scenario.name}  |  {len(scenario.keywords)} keywords  |  "
          f"{scenario.duration_days} days  |  seed={scenario.seed}")
    print(f"Budget: ${scenario.agent_budget:,.0f}  |  Competitors: 3")
    print("=" * 70)

    results = []
    for bidder in bidders:
        result = run_episode(bidder, scenario=scenario)
        results.append(result)
        print()
        print(result.summary())

    if len(results) > 1:
        print()
        print("=" * 70)
        print("LEADERBOARD")
        print("-" * 70)
        ranked = sorted(results, key=lambda r: r.score(), reverse=True)
        for i, r in enumerate(ranked, 1):
            print(f"  #{i}  {r.bidder_name:30s}  profit=${r.total_profit:>10,.2f}  "
                  f"ROAS={r.roas:.2f}x  conv={r.total_conversions}")


def benchmark_main():
    """aag-benchmark [--models m1,m2] [--scenarios s1,s2] [--seed 42]"""
    parser = argparse.ArgumentParser(
        description="Ad Arena — run benchmark across scenarios",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to run (default: baselines only)",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario names (default: all discovered)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Root seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    scenario_names = (
        [s.strip() for s in args.scenarios.split(",")]
        if args.scenarios
        else None
    )

    # For v1, --models is accepted but LLM configs require programmatic setup.
    # The CLI runs baselines only unless LLM configs are provided externally.
    from ad_arena.benchmark.harness import BenchmarkHarness

    harness = BenchmarkHarness(
        llm_configs=None,
        scenario_names=scenario_names,
        root_seed=args.seed,
    )

    print(f"Running benchmark (seed={args.seed})...")
    results = harness.run_all()

    if not results:
        print("No results produced.")
        return

    # Print summary for both suites
    from ad_arena.benchmark.scoring import compute_leaderboard

    for suite in ("campaign_efficiency", "strategy_optimization"):
        entries = compute_leaderboard(results, suite=suite)
        _print_leaderboard_table(entries, suite=suite)


def leaderboard_main():
    """aag-leaderboard — generate static HTML from results."""
    from ad_arena.ui.web_publisher import WebPublisher

    publisher = WebPublisher()
    publisher.generate()
    print("Leaderboard generated at docs/leaderboard/index.html")


def results_main():
    """aag-results — print summary table to terminal."""
    from ad_arena.benchmark.results_store import ResultsStore
    from ad_arena.benchmark.scoring import compute_leaderboard

    store = ResultsStore()
    results = store.load_all()

    if not results:
        print("No results found in results/ directory.")
        return

    for suite in ("campaign_efficiency", "strategy_optimization"):
        entries = compute_leaderboard(results, suite=suite)
        _print_leaderboard_table(entries, suite=suite)


def _print_leaderboard_table(entries, suite: str = "campaign_efficiency"):
    """Print a formatted leaderboard table to stdout."""
    if not entries:
        print("No entries to display.")
        return

    suite_label = "Campaign Efficiency" if suite == "campaign_efficiency" else "Strategy Optimization"
    print()
    print(f"=== {suite_label} ===")

    header = (
        f"{'Rank':>4}  {'Model':<30}  {'Type':<8}  {'Profit':>10}  "
        f"{'ROAS':>6}  {'CPA':>8}  {'CTR':>6}  {'CVR':>6}  "
        f"{'Conv':>5}  {'LR':>6}  {'LD%':>5}  {'API':>5}"
    )
    print()
    print(header)
    print("-" * len(header))

    for e in entries:
        cpa_str = f"${e.cpa:>7.2f}" if e.cpa != float("inf") else "     inf"
        print(
            f"{e.rank:>4}  {e.model_name:<30}  {e.model_type:<8}  "
            f"${e.aggregate_score:>9,.2f}  "
            f"{e.roas:>5.2f}x  {cpa_str}  "
            f"{e.ctr:>5.1%}  {e.cvr:>5.1%}  "
            f"{e.total_conversions:>5}  "
            f"{e.learning_rate:>6.2f}  "
            f"{e.learning_days_ratio:>4.0%}  "
            f"{e.llm_api_calls:>5}"
        )

    print()


if __name__ == "__main__":
    main()
