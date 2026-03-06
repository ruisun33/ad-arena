"""
v1.0 scenario: single search page, one HQP ad slot, second-price auction.

The market simulates a "running shoes" e-commerce vertical with 10 keywords
spanning different intent levels and competition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ad_arena.core.models import Keyword


@dataclass
class Scenario:
    """Full specification of a simulation run."""
    name: str = "SearchAds-v1"
    keywords: list[Keyword] = field(default_factory=list)
    duration_days: int = 30
    agent_budget: float = 10_000.0
    num_ad_slots: int = 1          # v1: single HQP slot
    reserve_price: float = 0.10    # minimum CPC
    seed: int = 42
    competitor_config: list[dict] = field(default_factory=list)

    @property
    def keyword_texts(self) -> list[str]:
        return [kw.text for kw in self.keywords]

    @classmethod
    def from_yaml(cls, path: Path) -> "Scenario":
        """Load a scenario from a YAML file.

        Raises ValueError for invalid YAML or missing required fields
        (name, keywords, duration_days).
        """
        path = Path(path)
        try:
            raw = path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"Cannot read {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML in {path}: expected a mapping at top level")

        # Validate required fields
        required = ("name", "keywords", "duration_days")
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(
                f"Missing required field(s) in {path}: {', '.join(missing)}"
            )

        # Build Keyword objects from dicts
        kw_dicts = data["keywords"]
        if not isinstance(kw_dicts, list) or len(kw_dicts) == 0:
            raise ValueError(f"'keywords' must be a non-empty list in {path}")

        keywords = [
            Keyword(
                text=kw["text"],
                daily_volume=int(kw["daily_volume"]),
                base_cpc=float(kw["base_cpc"]),
                base_cvr=float(kw["base_cvr"]),
                avg_order_value=float(kw["avg_order_value"]),
                intent=kw.get("intent", "commercial"),
            )
            for kw in kw_dicts
        ]

        # Parse competitor config
        competitors = data.get("competitors", [])
        if not isinstance(competitors, list):
            competitors = []

        return cls(
            name=data["name"],
            keywords=keywords,
            duration_days=int(data["duration_days"]),
            agent_budget=float(data.get("agent_budget", 10_000.0)),
            num_ad_slots=int(data.get("num_ad_slots", 1)),
            reserve_price=float(data.get("reserve_price", 0.10)),
            seed=int(data.get("seed", 42)),
            competitor_config=competitors,
        )

    def config_hash(self) -> str:
        """SHA-256 hex digest of the scenario configuration.

        Excludes seed (runtime parameter) for deterministic config identity.
        """
        config = {
            "name": self.name,
            "keywords": [
                {
                    "text": kw.text,
                    "daily_volume": kw.daily_volume,
                    "base_cpc": kw.base_cpc,
                    "base_cvr": kw.base_cvr,
                    "avg_order_value": kw.avg_order_value,
                    "intent": kw.intent,
                }
                for kw in self.keywords
            ],
            "duration_days": self.duration_days,
            "agent_budget": self.agent_budget,
            "num_ad_slots": self.num_ad_slots,
            "reserve_price": self.reserve_price,
            "competitor_config": self.competitor_config,
        }
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def discover_scenarios(directory: Path) -> dict[str, Scenario]:
    """Glob *.yaml files in *directory* and return a dict of name → Scenario.

    Skips files that fail to parse (logs nothing — caller can handle).
    """
    directory = Path(directory)
    scenarios: dict[str, Scenario] = {}
    for yaml_path in sorted(directory.glob("*.yaml")):
        try:
            scenario = Scenario.from_yaml(yaml_path)
            scenarios[scenario.name] = scenario
        except ValueError:
            continue  # skip invalid files
    return scenarios


def default_scenario(seed: int = 42) -> Scenario:
    """The standard v1 benchmark scenario."""
    return Scenario(
        name="SearchAds-v1",
        duration_days=30,
        agent_budget=10_000.0,
        num_ad_slots=1,
        reserve_price=0.10,
        seed=seed,
        keywords=[
            # === PROFITABLE: high intent, good unit economics ===
            # These are the gems — smart bidders should find and bid on these
            Keyword("buy running shoes online", daily_volume=120, base_cpc=3.50,
                    base_cvr=0.10, avg_order_value=130.0, intent="transactional"),
            Keyword("waterproof trail running shoes", daily_volume=35, base_cpc=1.80,
                    base_cvr=0.12, avg_order_value=170.0, intent="transactional"),
            Keyword("running shoes for flat feet", daily_volume=50, base_cpc=1.50,
                    base_cvr=0.09, avg_order_value=145.0, intent="commercial"),

            # === MARGINAL: can be profitable with the right bid ===
            Keyword("best running shoes", daily_volume=200, base_cpc=3.00,
                    base_cvr=0.05, avg_order_value=130.0, intent="commercial"),
            Keyword("running shoes sale", daily_volume=300, base_cpc=2.00,
                    base_cvr=0.06, avg_order_value=85.0, intent="transactional"),
            Keyword("nike pegasus review", daily_volume=100, base_cpc=1.20,
                    base_cvr=0.05, avg_order_value=130.0, intent="commercial"),

            # === TRAPS: high volume but poor economics ===
            # Naive "bid on everything" strategies lose money here
            Keyword("running shoes", daily_volume=600, base_cpc=2.80,
                    base_cvr=0.02, avg_order_value=110.0, intent="commercial"),
            Keyword("cheap running shoes", daily_volume=500, base_cpc=1.20,
                    base_cvr=0.02, avg_order_value=45.0, intent="transactional"),

            # === MONEY PITS: informational, very low conversion ===
            Keyword("how to choose running shoes", daily_volume=300, base_cpc=0.70,
                    base_cvr=0.005, avg_order_value=110.0, intent="informational"),
            Keyword("running shoes vs walking shoes", daily_volume=200, base_cpc=0.60,
                    base_cvr=0.004, avg_order_value=100.0, intent="informational"),
        ],
    )
