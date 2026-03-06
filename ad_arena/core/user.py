"""
User simulation — sampling profiles and modeling click/conversion behavior.

This is the ground-truth user model. It determines:
- Who is searching (UserProfile sampling)
- Whether they click an ad (based on intent, ad relevance, segment affinity, device)
- Whether they convert after clicking (based on keyword CVR, user intent, segment-AOV fit)

The platform's PredictedCTRModel (in engine.py) tries to ESTIMATE this
behavior, but imperfectly. The gap between prediction and reality is
what makes the quality score dynamics interesting.

Behavioral complexity:
- Different user segments respond to different ad messaging
- Mobile vs desktop users behave differently
- Ad fatigue: repeated exposure to the same advertiser reduces CTR
- Segment-AOV affinity: budget shoppers convert more on cheap items,
  enthusiasts convert more on premium items
"""

from __future__ import annotations

import numpy as np

from ad_arena.core.models import Keyword, UserProfile


# ── Segment-headline affinity ─────────────────────────────────
# Words that resonate with each user segment.
# If the ad headline contains words that match the user's segment,
# they're more likely to click.

SEGMENT_TRIGGER_WORDS: dict[str, set[str]] = {
    "budget_shopper": {
        "sale", "deal", "cheap", "save", "discount", "affordable",
        "clearance", "off", "value", "budget", "free", "lowest", "%",
    },
    "enthusiast": {
        "best", "top", "premium", "expert", "rated", "review",
        "performance", "advanced", "pro", "tested", "quality",
    },
    "young_professional": {
        "new", "trending", "popular", "fast", "modern", "stylish",
        "sleek", "smart", "innovative", "2025", "latest",
    },
    "casual": {
        "easy", "simple", "comfortable", "everyday", "great",
        "good", "nice", "shop", "find", "browse",
    },
}


class UserSimulator:
    """
    Samples user profiles and simulates their responses to ads.

    Behavioral model:
    - Profile sampling with segment-dependent characteristics
    - Click = f(intent, keyword relevance, segment affinity, device, fatigue)
    - Conversion = f(keyword CVR, user intent, segment-AOV fit)
    """

    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        # Ad fatigue tracking: {bidder_name: impression_count}
        # Resets daily (managed by engine)
        self._daily_impressions: dict[str, int] = {}

    def reset_daily(self) -> None:
        """Called at start of each day to reset fatigue counters."""
        self._daily_impressions = {}

    def record_impression(self, bidder_name: str) -> None:
        """Track that a user saw this bidder's ad (for fatigue)."""
        self._daily_impressions[bidder_name] = (
            self._daily_impressions.get(bidder_name, 0) + 1
        )

    # ── Profile sampling ──────────────────────────────────────

    def sample_user(self, keyword: Keyword) -> UserProfile:
        """
        Sample a user profile for a search query.

        Segments:
        - young_professional: moderate intent, responds to "trending/new"
        - budget_shopper: price-sensitive, responds to "sale/deal/cheap"
        - enthusiast: high intent, responds to "best/premium/expert"
        - casual: low intent, responds to "easy/comfortable"
        """
        segment = self._rng.choice(
            ["young_professional", "budget_shopper", "enthusiast", "casual"],
            p=[0.25, 0.30, 0.20, 0.25],
        )
        device = self._rng.choice(["mobile", "desktop"], p=[0.65, 0.35])

        # Purchase intent depends on keyword intent + user segment
        intent_base = {
            "transactional": 0.70,
            "commercial": 0.50,
            "informational": 0.15,
        }.get(keyword.intent, 0.40)

        segment_boost = {
            "enthusiast": 0.15,
            "young_professional": 0.05,
            "budget_shopper": -0.05,
            "casual": -0.10,
        }.get(segment, 0.0)

        purchase_intent = float(np.clip(
            self._rng.normal(intent_base + segment_boost, 0.15),
            0.0, 1.0,
        ))

        return UserProfile(
            segment=segment,
            device=device,
            purchase_intent=purchase_intent,
        )

    # ── Click behavior ────────────────────────────────────────

    def simulate_click(
        self,
        keyword: Keyword,
        user: UserProfile,
        ad_headline: str,
        bidder_name: str = "",
    ) -> bool:
        """
        Decide whether the user clicks the ad.

        CTR = base × intent × keyword_relevance × segment_affinity
              × device × fatigue × noise

        Components:
        - base: keyword intent type (transactional > commercial > info)
        - intent: user's purchase intent (0.7-1.3x)
        - keyword_relevance: headline-keyword word overlap (0.6-1.4x)
        - segment_affinity: headline-segment trigger word match (0.8-1.3x)
        - device: mobile penalty (0.85x)
        - fatigue: repeated exposure to same bidder today (decays)
        - noise: random variation (0.85-1.15x)
        """
        intent_ctr = {
            "transactional": 0.25,
            "commercial": 0.18,
            "informational": 0.08,
        }
        base = intent_ctr.get(keyword.intent, 0.15)

        # User purchase intent
        intent_mult = 0.7 + 0.6 * user.purchase_intent           # 0.7 - 1.3

        # Keyword-ad relevance (does the headline match the query?)
        kw_relevance = self._keyword_relevance(ad_headline, keyword.text)
        kw_mult = 0.6 + 0.8 * kw_relevance                      # 0.6 - 1.4

        # Segment-ad affinity (does the headline speak to this user type?)
        seg_affinity = self._segment_affinity(ad_headline, user.segment)
        seg_mult = 0.8 + 0.5 * seg_affinity                     # 0.8 - 1.3

        # Device
        device_mult = 0.85 if user.device == "mobile" else 1.0

        # Ad fatigue: CTR decays with repeated exposure to same bidder
        fatigue_mult = self._fatigue_factor(bidder_name)

        # Noise
        noise = self._rng.uniform(0.85, 1.15)

        ctr = base * intent_mult * kw_mult * seg_mult * device_mult * fatigue_mult * noise
        ctr = min(ctr, 0.50)
        return self._rng.random() < ctr

    # ── Conversion behavior ───────────────────────────────────

    def simulate_conversion(
        self,
        keyword: Keyword,
        user: UserProfile,
    ) -> bool:
        """
        Decide whether the user converts after clicking.

        CVR = base × intent × segment_aov_fit × noise

        Components:
        - base: keyword's base conversion rate
        - intent: user purchase intent (0.5-2.0x)
        - segment_aov_fit: how well the product price matches the segment
          (budget shoppers convert more on cheap items, enthusiasts on premium)
        - noise: random variation
        """
        base = keyword.base_cvr

        # User intent
        intent_mult = 0.5 + 1.5 * user.purchase_intent           # 0.5 - 2.0

        # Segment-AOV fit: does the product price match what this user wants?
        aov_fit = self._segment_aov_fit(user.segment, keyword.avg_order_value)

        # Noise
        noise = self._rng.uniform(0.7, 1.3)

        cvr = base * intent_mult * aov_fit * noise
        cvr = min(cvr, 0.40)
        return self._rng.random() < cvr

    # ── Relevance scoring ─────────────────────────────────────

    def _keyword_relevance(self, ad_headline: str, keyword_text: str) -> float:
        """
        Keyword-ad relevance: does the headline match the search query?
        Range 0-1.

        No headline → 0.3 baseline (generic ad).
        Keyword words in headline → up to 1.0.
        """
        if not ad_headline:
            return 0.3

        kw_words = set(keyword_text.lower().split())
        ad_words = set(ad_headline.lower().split())

        if not kw_words:
            return 0.3

        overlap = len(kw_words & ad_words) / len(kw_words)
        return min(overlap + 0.3, 1.0)

    def _segment_affinity(self, ad_headline: str, segment: str) -> float:
        """
        Segment-ad affinity: does the headline use words that resonate
        with this user segment? Range 0-1.

        A "budget_shopper" seeing "Save 30%" has high affinity.
        An "enthusiast" seeing "Expert Reviewed" has high affinity.
        Mismatched messaging (e.g., "cheap" shown to enthusiast) → low affinity.
        """
        if not ad_headline:
            return 0.3  # no headline = neutral

        trigger_words = SEGMENT_TRIGGER_WORDS.get(segment, set())
        if not trigger_words:
            return 0.3

        ad_words = set(ad_headline.lower().split())
        matches = len(ad_words & trigger_words)

        # 0 matches → 0.2, 1 match → 0.5, 2+ matches → 0.8-1.0
        if matches == 0:
            return 0.2
        elif matches == 1:
            return 0.5
        else:
            return min(0.5 + matches * 0.2, 1.0)

    def _fatigue_factor(self, bidder_name: str) -> float:
        """
        Ad fatigue: users tune out ads they've seen many times today.

        First ~100 impressions: no fatigue (1.0x)
        100-500 impressions: gradual decay to 0.85x
        500+ impressions: floors at 0.85x

        This penalizes bidders who dominate every single auction —
        their CTR naturally degrades through the day.
        """
        if not bidder_name:
            return 1.0

        imps = self._daily_impressions.get(bidder_name, 0)
        if imps < 100:
            return 1.0
        elif imps < 500:
            # Linear decay from 1.0 to 0.85
            return 1.0 - 0.15 * ((imps - 100) / 400)
        else:
            return 0.85

    def _segment_aov_fit(self, segment: str, avg_order_value: float) -> float:
        """
        How well does the product price match what this segment wants?

        Budget shoppers convert better on cheap items (AOV < $80).
        Enthusiasts convert better on premium items (AOV > $130).
        Others are neutral.

        Range: 0.7 (mismatch) to 1.3 (perfect fit).
        """
        if segment == "budget_shopper":
            if avg_order_value < 60:
                return 1.3   # cheap item, perfect for budget shopper
            elif avg_order_value < 100:
                return 1.1
            elif avg_order_value < 140:
                return 0.9
            else:
                return 0.7   # expensive item, budget shopper hesitates

        elif segment == "enthusiast":
            if avg_order_value > 150:
                return 1.3   # premium item, enthusiast loves it
            elif avg_order_value > 120:
                return 1.1
            elif avg_order_value > 80:
                return 0.9
            else:
                return 0.75  # too cheap, enthusiast suspects low quality

        elif segment == "young_professional":
            # Mid-range preference
            if 80 <= avg_order_value <= 150:
                return 1.15
            else:
                return 0.9

        else:  # casual
            return 1.0  # price-indifferent
