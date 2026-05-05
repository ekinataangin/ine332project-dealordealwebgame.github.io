"""
policies.py - Banker offer policies
"""

import numpy as np


class BaselinePolicy:
    """
    Policy 1: Baseline Expected Value Policy
    Offer = 0.8 * mean(remaining prizes)

    Simple fraction-of-EV strategy. The 0.8 multiplier gives the banker
    a built-in margin while still making reasonable offers.
    """
    name = "Baseline (0.8×EV)"

    def make_offer(self, ev, std, round_number, progress, remaining_prizes):
        return 0.8 * ev


class RiskAdjustedPolicy:
    """
    Policy 2: Risk-Adjusted Policy
    Offer = alpha * mean - beta * std

    When remaining prizes are highly volatile (large std), the banker
    reduces the offer to protect against accidentally overpaying. In low-
    variance endgame situations the penalty term shrinks naturally.

    Default parameters:
      alpha = 0.85  — slightly more generous than baseline on EV
      beta  = 0.05  — small penalty per unit of spread

    Justification: beta=0.05 means for every $10,000 of std the offer
    is reduced by $500, a modest but meaningful risk hedge.
    """
    name = "Risk-Adjusted (α·EV − β·σ)"

    def __init__(self, alpha=0.85, beta=0.05):
        self.alpha = alpha
        self.beta = beta

    def make_offer(self, ev, std, round_number, progress, remaining_prizes):
        offer = self.alpha * ev - self.beta * std
        return max(0.0, offer)


class DynamicPolicy:
    """
    Policy 3: Dynamic Round-Based Policy
    multiplier ramps from early_mult to late_mult as the game progresses.

    Early rounds: low offers because the player has little information and
    many cases remain — the banker exploits uncertainty.
    Later rounds: offers approach EV because the player has more info,
    fewer cases, and is under more psychological pressure to deal.

    multiplier = early_mult + progress * (late_mult - early_mult)
    Offer = clamp(multiplier, 0.65, 0.95) * mean(remaining prizes)
    """
    name = "Dynamic Round-Based"

    def __init__(self, early_mult=0.65, late_mult=0.95):
        self.early_mult = early_mult
        self.late_mult = late_mult

    def make_offer(self, ev, std, round_number, progress, remaining_prizes):
        multiplier = self.early_mult + progress * (self.late_mult - self.early_mult)
        multiplier = np.clip(multiplier, 0.65, 0.95)
        return multiplier * ev


class PlayerAdaptivePolicy:
    """
    Policy 4 (Optional): Player-Adaptive Policy
    Adjusts the offer multiplier based on inferred player risk attitude,
    approximated by how often the player has rejected offers so far
    (high rejection rate → likely risk-seeking → need higher offer).

    Since we cannot know the player type in advance, we use a proxy:
      if the player has rejected many previous offers relative to the round,
      we classify them as risk-seeking and increase the offer.

    multiplier_base = 0.80
    If rejections/offers_made_so_far > 0.8 → add 0.08 (risk-seeking adjustment)
    If rejections/offers_made_so_far < 0.3 → subtract 0.05 (risk-averse adjustment)

    State management:
      - reset() is called by the game engine at the start of each new game.
      - record_rejection() is called by the game engine each time the player
        rejects an offer, incrementing only the rejection counter.
      - make_offer() increments the total offer counter and computes the
        rejection rate as: rejections / (offers_made - 1), i.e., how many
        of the *previous* offers were rejected.
    """
    name = "Player-Adaptive"

    def __init__(self, base_mult=0.80, seeking_bonus=0.08, averse_discount=0.05):
        self.base_mult = base_mult
        self.seeking_bonus = seeking_bonus
        self.averse_discount = averse_discount
        self._rejection_count = 0
        self._offer_count = 0

    def reset(self):
        self._rejection_count = 0
        self._offer_count = 0

    def record_rejection(self):
        """Called by the game engine when the player rejects an offer."""
        self._rejection_count += 1

    def make_offer(self, ev, std, round_number, progress, remaining_prizes):
        self._offer_count += 1
        mult = self.base_mult
        if self._offer_count > 1:
            rejection_rate = self._rejection_count / (self._offer_count - 1)
            if rejection_rate > 0.8:
                mult += self.seeking_bonus
            elif rejection_rate < 0.3:
                mult -= self.averse_discount
        return max(0.0, mult * ev)
