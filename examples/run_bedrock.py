"""
Run Ad Arena benchmark with Claude models via Amazon Bedrock.

Prerequisites:
    pip install boto3
    # Configure AWS credentials (aws configure, env vars, or IAM role)

Usage:
    python examples/run_bedrock.py

    # Single model, single episode:
    python examples/run_bedrock.py --model claude-sonnet-4.6 --seed 42

    # Full benchmark across all scenarios:
    python examples/run_bedrock.py --benchmark --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from ad_arena import run_episode
from ad_arena.agents.llm_bidder import LLMBidder
from ad_arena.benchmark.scoring import compute_strategy_optimization_score
from ad_arena.benchmark.adaptation import compute_adaptation_metrics


# ── Bedrock model IDs ─────────────────────────────────────────

MODELS = {
    "claude-opus-4.5": "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "claude-opus-4.6": "us.anthropic.claude-opus-4-6-v1",
    "claude-sonnet-4.6": "us.anthropic.claude-sonnet-4-6",
    "nova-premier": "us.amazon.nova-premier-v1:0",
    "llama3-3-70b": "us.meta.llama3-3-70b-instruct-v1:0",
    "deepseek-r1": "us.deepseek.r1-v1:0",
    "mistral-large": "mistral.mistral-large-2407-v1:0",
}


# Per-model max_tokens overrides (reasoning models need more space)
MODEL_MAX_TOKENS = {
    "deepseek-r1": 8000,
}


# ── Bedrock LLM callable factory ──────────────────────────────

def make_bedrock_callable(
    model_id: str,
    region: str = "us-west-2",
    max_tokens: int = 2000,
    temperature: float = 0.7,
):
    """Create an llm_fn callable for LLMBidder using Bedrock Converse API."""
    client = boto3.client("bedrock-runtime", region_name=region)

    def call_bedrock(messages: list[dict[str, str]]) -> str:
        # LLMBidder sends system prompt as the first message with role="system"
        system_blocks = []
        converse_messages = []

        for m in messages:
            if m["role"] == "system":
                system_blocks.append({"text": m["content"]})
            else:
                converse_messages.append({
                    "role": m["role"],
                    "content": [{"text": m["content"]}],
                })

        response = client.converse(
            modelId=model_id,
            system=system_blocks,
            messages=converse_messages,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        )

        # Extract text from response — handle reasoning models (e.g. DeepSeek R1)
        # that return reasoningContent instead of text blocks
        content_blocks = response["output"]["message"]["content"]
        for block in content_blocks:
            if "text" in block:
                return block["text"]
        # Fallback: extract from reasoningContent
        for block in content_blocks:
            rc = block.get("reasoningContent", {})
            rt = rc.get("reasoningText", {})
            if "text" in rt:
                return rt["text"]
        return ""

    print(f"  [make_bedrock_callable] Created callable for model_id={model_id} region={region}")
    return call_bedrock


# ── Single episode run ────────────────────────────────────────

def run_single(model_key: str, seed: int = 42, verbose: bool = False):
    """Run one episode with a Bedrock model and print results."""
    model_id = MODELS[model_key]
    max_tokens = MODEL_MAX_TOKENS.get(model_key, 2000)
    print(f"Running {model_key} ({model_id}) ...")

    llm_fn = make_bedrock_callable(model_id, max_tokens=max_tokens)
    bidder = LLMBidder(name=model_key, llm_fn=llm_fn, verbose=verbose)
    result = run_episode(bidder, seed=seed)

    print()
    print(result.summary())

    so_score = compute_strategy_optimization_score(result.daily_log)
    metrics = compute_adaptation_metrics(result.daily_log, result.daily_strategies)

    print(f"Strategy Optimization (final 7d): ${so_score:,.2f}")
    print(f"Learning Rate: {metrics.learning_rate:.2f}")
    print(f"Learning Days: {metrics.learning_days} | Optimizing Days: {metrics.optimizing_days}")
    print(f"Convergence Day: {metrics.convergence_day}")

    return result


# ── Full benchmark run ────────────────────────────────────────

def run_benchmark(model_keys: list[str] | None = None, seed: int = 42,
                  scenario: str | None = None, skip_baselines: bool = False,
                  num_runs: int = 1):
    """Run the full benchmark harness with Bedrock models."""
    from ad_arena.benchmark.harness import BenchmarkHarness

    if model_keys is None:
        model_keys = list(MODELS.keys())

    llm_configs = []
    for key in model_keys:
        model_id = MODELS[key]
        max_tokens = MODEL_MAX_TOKENS.get(key, 2000)
        print(f"  Model: {key} -> {model_id} (max_tokens={max_tokens})")
        llm_configs.append({
            "name": key,
            "llm_fn": make_bedrock_callable(model_id, max_tokens=max_tokens),
        })

    scenario_names = [scenario] if scenario else None

    print(f"Running benchmark with {len(llm_configs)} models (seed={seed}, runs={num_runs}) ...")
    harness = BenchmarkHarness(
        llm_configs=llm_configs,
        scenario_names=scenario_names,
        root_seed=seed,
        skip_baselines=skip_baselines,
        num_runs=num_runs,
    )

    results = harness.run_all()

    # Print leaderboard
    from ad_arena.benchmark.scoring import compute_leaderboard
    for suite in ("campaign_efficiency", "strategy_optimization"):
        entries = compute_leaderboard(results, suite=suite)
        print(f"\n{'=' * 70}")
        print(f"LEADERBOARD — {suite.replace('_', ' ').title()}")
        print(f"{'-' * 70}")
        for e in entries:
            print(f"  #{e.rank}  {e.model_name:<25}  "
                  f"score=${e.aggregate_score:>10,.2f}  "
                  f"type={e.model_type}")

    print(f"\nResults saved to results/. Run 'arena-leaderboard' to generate HTML.")


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ad Arena — Bedrock LLM benchmark")
    parser.add_argument(
        "--model", type=str, default="claude-sonnet-4.6",
        choices=list(MODELS.keys()),
        help="Model to run (default: claude-sonnet-4.6)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of runs per model per scenario (default: 1)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run full benchmark across all scenarios")
    parser.add_argument("--scenario", type=str, default=None,
                        help="Run only this scenario (e.g. SearchAds-v1)")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip baseline bidders, run LLM only")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            model_keys=[args.model],
            seed=args.seed,
            scenario=args.scenario,
            skip_baselines=args.skip_baselines,
            num_runs=args.runs,
        )
    else:
        run_single(args.model, seed=args.seed, verbose=args.verbose)


if __name__ == "__main__":
    main()
