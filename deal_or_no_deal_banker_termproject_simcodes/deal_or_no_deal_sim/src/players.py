"""
players.py - Player acceptance behavior models
"""

import numpy as np
from enum import Enum


class PlayerType(Enum):
    RISK_AVERSE = "Risk-Averse"
    RISK_NEUTRAL = "Risk-Neutral"
    RISK_SEEKING = "Risk-Seeking"


class LogisticAcceptancePlayer:
    """
    Player who accepts offers via a logistic (sigmoid) probability function.

    P(accept) = 1 / (1 + exp(-k * (ratio - threshold)))

    where:
      ratio     = offer / expected_value_remaining
      k         = steepness of the sigmoid (sensitivity)
      threshold = the ratio at which acceptance probability = 0.5

    Player types and thresholds:
      Risk-Averse:   threshold=0.70  (accepts at 70% of EV)
      Risk-Neutral:  threshold=0.90  (accepts near EV)
      Risk-Seeking:  threshold=1.10  (only accepts above EV)
    """

    THRESHOLDS = {
        PlayerType.RISK_AVERSE:  0.70,
        PlayerType.RISK_NEUTRAL: 0.90,
        PlayerType.RISK_SEEKING: 1.10,
    }

    def __init__(self, player_type: PlayerType, k: float = 8.0, rng=None):
        self.player_type = player_type
        self.k = k
        self.threshold = self.THRESHOLDS[player_type]
        self.rng = rng or np.random.default_rng()

    @property
    def name(self):
        return self.player_type.value

    def acceptance_probability(self, offer: float, expected_value: float,
                                round_number: int = 1, remaining_cases: int = 26) -> float:
        """
        Compute probability of accepting the offer.

        ratio > threshold → high acceptance probability
        ratio < threshold → low acceptance probability
        """
        if expected_value <= 0:
            # Edge case: EV is zero or negative; accept if offer is non-negative
            return 1.0 if offer >= 0 else 0.0

        ratio = offer / expected_value
        prob = 1.0 / (1.0 + np.exp(-self.k * (ratio - self.threshold)))
        return float(np.clip(prob, 0.0, 1.0))

    def accept_offer(self, offer: float, expected_value: float,
                     round_number: int = 1, remaining_cases: int = 26) -> bool:
        """Stochastic accept/reject decision."""
        p = self.acceptance_probability(offer, expected_value, round_number, remaining_cases)
        return bool(self.rng.random() < p)
