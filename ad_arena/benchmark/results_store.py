"""Results persistence — JSON storage for benchmark results and leaderboard."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ad_arena.benchmark.adaptation import AdaptationMetrics
from ad_arena.core.models import (
    DailyFeedback,
    DailyStrategy,
    EpisodeResult,
    KeywordDayStats,
    VariantDayStats,
)
from ad_arena.benchmark.scoring import RunResult

logger = logging.getLogger(__name__)


# ── Serialization helpers ─────────────────────────────────────


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dicts.

    Handles frozen dataclasses with dict fields (where dataclasses.asdict
    works but we want explicit control over the recursion).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    return obj


# ── Deserialization helpers ───────────────────────────────────


def _variant_day_stats_from_dict(d: dict) -> VariantDayStats:
    return VariantDayStats(
        variant_index=d.get("variant_index", 0),
        headline=d.get("headline", ""),
        impressions=d.get("impressions", 0),
        clicks=d.get("clicks", 0),
        conversions=d.get("conversions", 0),
        spend=d.get("spend", 0.0),
        revenue=d.get("revenue", 0.0),
    )


def _keyword_day_stats_from_dict(d: dict) -> KeywordDayStats:
    return KeywordDayStats(
        impressions=d.get("impressions", 0),
        clicks=d.get("clicks", 0),
        conversions=d.get("conversions", 0),
        spend=d.get("spend", 0.0),
        revenue=d.get("revenue", 0.0),
    )


def _daily_feedback_from_dict(d: dict) -> DailyFeedback:
    keyword_stats = {
        k: _keyword_day_stats_from_dict(v)
        for k, v in d.get("keyword_stats", {}).items()
    }
    variant_stats = {
        k: [_variant_day_stats_from_dict(vs) for vs in v]
        for k, v in d.get("variant_stats", {}).items()
    }
    return DailyFeedback(
        day=d["day"],
        impressions=d["impressions"],
        clicks=d["clicks"],
        conversions=d["conversions"],
        spend=d["spend"],
        revenue=d["revenue"],
        profit=d["profit"],
        budget_remaining=d["budget_remaining"],
        keyword_stats=keyword_stats,
        variant_stats=variant_stats,
    )


def _daily_strategy_from_dict(d: dict) -> DailyStrategy:
    return DailyStrategy(
        keyword_bids=d.get("keyword_bids", {}),
        keyword_headlines=d.get("keyword_headlines", {}),
        daily_budget=d.get("daily_budget", float("inf")),
        reasoning=d.get("reasoning", ""),
        audience_modifiers=d.get("audience_modifiers", {}),
        daypart_modifiers=d.get("daypart_modifiers", {}),
        keyword_variants={
            k: list(v) for k, v in d.get("keyword_variants", {}).items()
        },
    )


def _episode_result_from_dict(d: dict) -> EpisodeResult:
    daily_log = [_daily_feedback_from_dict(fb) for fb in d.get("daily_log", [])]
    daily_strategies = [
        _daily_strategy_from_dict(ds) for ds in d.get("daily_strategies", [])
    ]
    return EpisodeResult(
        bidder_name=d["bidder_name"],
        total_spend=d.get("total_spend", 0.0),
        total_revenue=d.get("total_revenue", 0.0),
        total_impressions=d.get("total_impressions", 0),
        total_clicks=d.get("total_clicks", 0),
        total_conversions=d.get("total_conversions", 0),
        total_profit=d.get("total_profit", 0.0),
        days=d.get("days", 0),
        budget=d.get("budget", 0.0),
        daily_log=daily_log,
        daily_strategies=daily_strategies,
    )


def _adaptation_metrics_from_dict(d: dict) -> AdaptationMetrics:
    return AdaptationMetrics(
        learning_rate=d.get("learning_rate", 0.0),
        strategy_volatility=d.get("strategy_volatility", []),
        learning_days=d.get("learning_days", 0),
        optimizing_days=d.get("optimizing_days", 0),
        profit_trajectory=d.get("profit_trajectory", []),
        convergence_day=d.get("convergence_day"),
        keyword_convergence_day=d.get("keyword_convergence_day"),
    )


def _run_result_from_dict(d: dict) -> RunResult:
    return RunResult(
        episode_result=_episode_result_from_dict(d["episode_result"]),
        adaptation_metrics=_adaptation_metrics_from_dict(d["adaptation_metrics"]),
        model_name=d["model_name"],
        model_type=d["model_type"],
        scenario_name=d["scenario_name"],
        scenario_hash=d["scenario_hash"],
        root_seed=d["root_seed"],
        wall_clock_seconds=d["wall_clock_seconds"],
        llm_api_calls=d["llm_api_calls"],
        software_version=d["software_version"],
        strategy_optimization_score=d.get("strategy_optimization_score", 0.0),
    )


# ── ResultsStore ──────────────────────────────────────────────


class ResultsStore:
    """Persists and loads benchmark results as JSON files."""

    def __init__(self, base_dir: Path = Path("results")) -> None:
        self.base_dir = Path(base_dir)

    def save(self, result: RunResult) -> Path:
        """Save a RunResult to results/{scenario}/{model}/{timestamp}.json.

        Returns the path to the written file.
        Handles disk write failures gracefully (logs and continues).
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        directory = self.base_dir / result.scenario_name / result.model_name
        filepath = directory / f"{timestamp}.json"

        data = _to_dict(result)

        try:
            directory.mkdir(parents=True, exist_ok=True)
            filepath.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Failed to save result to %s: %s", filepath, exc)
            raise

        return filepath

    def load_all(self) -> list[RunResult]:
        """Load all stored results. Skips corrupt files with a warning."""
        results: list[RunResult] = []
        if not self.base_dir.exists():
            return results

        for json_path in sorted(self.base_dir.glob("**/*.json")):
            # Skip the leaderboard summary file
            if json_path.name == "leaderboard.json":
                continue
            try:
                raw = json_path.read_text(encoding="utf-8")
                data = json.loads(raw)
                results.append(_run_result_from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping corrupt result file %s: %s", json_path, exc)
                continue

        return results

    def update_leaderboard(self, results: list[RunResult]) -> None:
        """Write results/leaderboard.json with top result per (model, scenario) pair.

        Each entry includes both campaign_efficiency_score and strategy_optimization_score
        for dual-suite leaderboard generation.
        """
        # Group by (model_name, scenario_name), pick highest total_profit (Campaign_Efficiency)
        # Strip run suffixes like " (run 1)" so multiple runs aggregate under one model
        best: dict[tuple[str, str], RunResult] = {}
        for r in results:
            base_name = re.sub(r"\s*\(run \d+\)$", "", r.model_name)
            key = (base_name, r.scenario_name)
            if key not in best or r.episode_result.total_profit > best[key].episode_result.total_profit:
                best[key] = r

        entries = []
        for (model_name, scenario_name), r in sorted(best.items()):
            entries.append({
                "model_name": model_name,
                "scenario_name": scenario_name,
                "model_type": r.model_type,
                "total_profit": r.episode_result.total_profit,
                "campaign_efficiency_score": r.episode_result.total_profit,
                "strategy_optimization_score": r.strategy_optimization_score,
                "total_spend": r.episode_result.total_spend,
                "total_revenue": r.episode_result.total_revenue,
                "total_conversions": r.episode_result.total_conversions,
                "total_impressions": r.episode_result.total_impressions,
                "total_clicks": r.episode_result.total_clicks,
                "days": r.episode_result.days,
                "scenario_hash": r.scenario_hash,
                "root_seed": r.root_seed,
                "wall_clock_seconds": r.wall_clock_seconds,
                "llm_api_calls": r.llm_api_calls,
                "learning_rate": r.adaptation_metrics.learning_rate,
                "learning_days": r.adaptation_metrics.learning_days,
                "optimizing_days": r.adaptation_metrics.optimizing_days,
            })

        filepath = self.base_dir / "leaderboard.json"
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            filepath.write_text(
                json.dumps(entries, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            logger.error("Failed to write leaderboard to %s: %s", filepath, exc)

    def validate_hash(
        self,
        result_path: Path,
        current_scenarios: dict[str, "Scenario"],
    ) -> bool:
        """Compare stored scenario_hash with current config_hash.

        Returns True if hashes match, False on mismatch or if the scenario
        is not found in current_scenarios.
        """
        try:
            raw = Path(result_path).read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Cannot read result file %s: %s", result_path, exc)
            return False

        scenario_name = data.get("scenario_name", "")
        stored_hash = data.get("scenario_hash", "")

        scenario = current_scenarios.get(scenario_name)
        if scenario is None:
            logger.warning(
                "Scenario %r not found in current scenarios for %s",
                scenario_name,
                result_path,
            )
            return False

        current_hash = scenario.config_hash()
        if stored_hash != current_hash:
            logger.warning(
                "Scenario hash mismatch for %s: stored=%s, current=%s",
                result_path,
                stored_hash,
                current_hash,
            )
            return False

        return True
