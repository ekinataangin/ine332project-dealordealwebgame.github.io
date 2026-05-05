"""
game.py - Core Deal or No Deal game logic
"""

import numpy as np

# Standard U.S. 26-case prize distribution
PRIZE_VALUES = [
    0.01, 1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 750,
    1_000, 5_000, 10_000, 25_000, 50_000, 75_000, 100_000,
    200_000, 300_000, 400_000, 500_000, 750_000, 1_000_000
]

# Realistic case-opening schedule (U.S. version)
OPENING_SCHEDULE = [6, 5, 4, 3, 2, 1, 1, 1, 1]


class DealOrNoDealGame:
    """
    Simulates a single game of Deal or No Deal from the banker's perspective.
    """

    def __init__(self, prize_values=None, opening_schedule=None, rng=None):
        self.prize_values = prize_values or PRIZE_VALUES
        self.opening_schedule = opening_schedule or OPENING_SCHEDULE
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        """Initialize / re-initialize game state."""
        prizes = list(self.prize_values)
        self.rng.shuffle(prizes)  # randomly assign prizes to cases

        self.cases = {i: prizes[i] for i in range(len(prizes))}  # case_id -> prize
        self.player_case = self.rng.integers(0, len(prizes))      # player's chosen case
        self.opened_cases = set()
        self.current_round = 0
        self.game_over = False
        self.accepted_offer = None
        self.offer_history = []
        self.decision_history = []

    def get_remaining_prizes(self):
        """Return list of prize values in all still-closed cases (including player's)."""
        closed = [cid for cid in self.cases if cid not in self.opened_cases]
        return [self.cases[cid] for cid in closed]

    def get_unopened_non_player_cases(self):
        """Return case ids that are closed AND not the player's case."""
        return [cid for cid in self.cases
                if cid not in self.opened_cases and cid != self.player_case]

    def open_cases(self, n):
        """Open n random cases from the non-player closed cases."""
        available = self.get_unopened_non_player_cases()
        n = min(n, len(available))
        to_open = self.rng.choice(available, size=n, replace=False)
        for cid in to_open:
            self.opened_cases.add(cid)
        return [self.cases[cid] for cid in to_open]

    def expected_value_remaining(self):
        """Expected value (mean) of remaining prizes."""
        remaining = self.get_remaining_prizes()
        if not remaining:
            return 0.0
        return float(np.mean(remaining))

    def std_remaining(self):
        """Standard deviation of remaining prizes."""
        remaining = self.get_remaining_prizes()
        if len(remaining) < 2:
            return 0.0
        return float(np.std(remaining, ddof=0))

    def play_game(self, policy, player):
        """
        Run a full game.

        Returns a dict with game result metrics.
        """
        self.reset()
        player_case_value = self.cases[self.player_case]

        # Reset stateful policies (e.g. PlayerAdaptivePolicy) at the start of each game
        if hasattr(policy, "reset"):
            policy.reset()

        for round_idx, cases_to_open in enumerate(self.opening_schedule):
            # Check if only player's case remains
            non_player_remaining = self.get_unopened_non_player_cases()
            if not non_player_remaining:
                break

            # Open cases for this round
            self.open_cases(cases_to_open)
            self.current_round = round_idx + 1

            remaining = self.get_remaining_prizes()
            if not remaining:
                break

            ev = self.expected_value_remaining()
            std = self.std_remaining()
            n_remaining = len(remaining)
            total_cases = len(self.cases)
            progress = 1.0 - (n_remaining / total_cases)

            # Banker makes offer
            offer = policy.make_offer(
                ev=ev,
                std=std,
                round_number=self.current_round,
                progress=progress,
                remaining_prizes=remaining
            )
            offer = max(0.0, offer)
            self.offer_history.append(offer)

            # Player decides
            accept = player.accept_offer(
                offer=offer,
                expected_value=ev,
                round_number=self.current_round,
                remaining_cases=n_remaining
            )
            self.decision_history.append(accept)

            if accept:
                self.accepted_offer = offer
                self.game_over = True
                break
            else:
                # Inform stateful policies (e.g. PlayerAdaptivePolicy) of the rejection
                if hasattr(policy, "record_rejection"):
                    policy.record_rejection()

        # If player never accepted, payout = player's own case value
        if self.accepted_offer is None:
            payout = player_case_value
        else:
            payout = self.accepted_offer

        banker_profit = player_case_value - payout
        accepted = self.accepted_offer is not None
        round_of_acceptance = self.current_round if accepted else None

        return {
            "player_case_value": player_case_value,
            "accepted": accepted,
            "accepted_offer": self.accepted_offer,
            "payout": payout,
            "banker_profit": banker_profit,
            "round_of_acceptance": round_of_acceptance,
            "n_offers": len(self.offer_history),
            "offers": self.offer_history.copy(),
        }
