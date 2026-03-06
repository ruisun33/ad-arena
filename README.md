# Ad Arena

A simulated search-ads auction environment for benchmarking LLM bidding strategies.

Ad Arena runs LLMs through standardized advertising scenarios — keyword bidding, budget pacing, audience targeting, ad copy testing — and measures how well each model learns to optimize a campaign from data alone.

## Installation

```bash
pip install -e .
```

For development (includes pytest and Hypothesis):

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

Run the built-in baselines against the default scenario:

```bash
aag-run
```

This executes `SimpleBidder`, `BudgetPacingBidder`, and `KeywordValueBidder` through a 30-day simulated ad auction and prints a results summary.

## Custom Bidder

Create a `Bidder` subclass and point the CLI at it:

```python
# my_bidder.py
from ad_arena.agents.bidder import Bidder, EpisodeConfig
from ad_arena.core.models import DailyFeedback, DailyStrategy

class MyBidder(Bidder):
    name = "MyBidder"

    def on_episode_start(self, config: EpisodeConfig) -> None:
        self._keywords = config.keywords

    def strategy(self, feedback: DailyFeedback | None) -> DailyStrategy:
        bids = {kw: 3.00 for kw in self._keywords}
        return DailyStrategy(keyword_bids=bids)
```

```bash
aag-run --bidder my_bidder.py
```

See [`examples/my_bidder.py`](examples/my_bidder.py) for a more complete example with selective keyword targeting and ad copy.

## LLM Benchmark

Run all baselines through all scenarios:

```bash
aag-benchmark
```

Filter by scenario or set a seed:

```bash
aag-benchmark --scenarios SearchAds-v1 --seed 42
```

### Configuring LLM Providers

LLM providers are configured programmatically via the `BenchmarkHarness`:

```python
from ad_arena.benchmark.harness import BenchmarkHarness

harness = BenchmarkHarness(
    llm_configs=[
        {"name": "gpt-4o", "llm_fn": my_openai_callable},
        {"name": "claude-sonnet", "llm_fn": my_anthropic_callable},
    ],
    root_seed=42,
)
results = harness.run_all()
```

Each `llm_fn` takes a list of chat messages and returns a string. The harness handles fallback strategies and API call counting.

## Evaluation Suites

Each benchmark run produces two scores from the same episode:

- **Campaign Efficiency** — total profit across all 30 days. Rewards models that balance exploration cost against optimization gain.
- **Strategy Optimization** — profit from only the final 7 days (days 24–30). Rewards models that converge on the best strategy regardless of early exploration spend.

## Leaderboard

Generate the static HTML leaderboard:

```bash
aag-leaderboard
```

View results in the terminal:

```bash
aag-results
```

## Project Structure

```
ad_arena/
├── core/              # Simulation engine + data models
│   ├── models.py      # DailyStrategy, DailyFeedback, EpisodeResult, etc.
│   ├── engine.py      # GSP auction pipeline (retrieval → ranking → pricing)
│   ├── user.py        # User behavior simulation (clicks, conversions)
│   └── scenario.py    # Scenario loading from YAML
├── agents/            # Bidder interface + implementations
│   ├── bidder.py      # Bidder ABC + EpisodeConfig
│   ├── baselines.py   # SimpleBidder, BudgetPacer, KeywordValue
│   ├── competitors.py # Bot advertisers (AggressiveCo, ValueShop, SmartBidder)
│   └── llm_bidder.py  # LLM adapter (prompt rendering + JSON parsing)
├── benchmark/         # Harness, scoring, adaptation, results
│   ├── harness.py     # BenchmarkHarness (LLM × scenario matrix)
│   ├── scoring.py     # Dual-suite scoring and leaderboard ranking
│   ├── adaptation.py  # Learning rate, strategy volatility, convergence
│   ├── results_store.py # JSON persistence
│   └── seed_manager.py  # Deterministic seed derivation
├── ui/                # Rendering + web publisher
│   ├── rendering.py   # Text rendering for LLM observations
│   └── web_publisher.py # Static HTML leaderboard generator
├── runner.py          # Episode orchestrator
└── cli.py             # CLI entry points (aag-run, aag-benchmark, etc.)

scenarios/             # YAML scenario definitions
examples/              # Example custom bidders
```

## License

Apache-2.0
