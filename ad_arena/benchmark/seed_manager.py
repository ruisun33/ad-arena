"""Deterministic seed derivation for reproducible benchmark runs."""

import hashlib


class SeedManager:
    """Derives deterministic sub-seeds from a single root seed.

    Uses SHA-256 hashing to produce unique, reproducible seeds for each
    (scenario, model, component) combination. Same inputs always yield
    the same seed; different inputs yield different seeds.
    """

    def __init__(self, root_seed: int):
        self.root_seed = root_seed

    def _derive(self, scenario_name: str, model_name: str, component: str) -> int:
        """Derive a 32-bit seed from root_seed, scenario, model, and component."""
        key = f"{self.root_seed}:{scenario_name}:{model_name}:{component}"
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest[:8], 16)

    def engine_seed(self, scenario_name: str, model_name: str) -> int:
        """Deterministic seed for the auction engine."""
        return self._derive(scenario_name, model_name, "engine")

    def user_seed(self, scenario_name: str, model_name: str) -> int:
        """Deterministic seed for the user simulator."""
        return self._derive(scenario_name, model_name, "user")

    def competitor_seed(self, scenario_name: str, model_name: str, competitor_idx: int) -> int:
        """Deterministic seed for a specific competitor bot."""
        return self._derive(scenario_name, model_name, f"competitor_{competitor_idx}")
