"""
simulation.py - Monte Carlo simulation engine
"""

import numpy as np
import pandas as pd
from src.game import DealOrNoDealGame, PRIZE_VALUES, OPENING_SCHEDULE
from src.players import LogisticAcceptancePlayer, PlayerType


def run_single_game(policy, player, prize_values=None, opening_schedule=None, rng=None):
    """Run one independent game and return result dict."""
    game = DealOrNoDealGame(
        prize_values=prize_values or PRIZE_VALUES,
        opening_schedule=opening_schedule or OPENING_SCHEDULE,
        rng=rng or np.random.default_rng()
    )
    return game.play_game(policy, player)


def run_simulations(policy, player_type: PlayerType, n_games: int,
                    prize_values=None, opening_schedule=None, seed=42):
    """
    Run n_games independent simulations for a single (policy, player_type) pair.
    Returns a list of result dicts.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_games):
        player = LogisticAcceptancePlayer(player_type, rng=rng)
        result = run_single_game(policy, player, prize_values, opening_schedule, rng)
        result["policy"] = policy.name
        result["player_type"] = player_type.value
        results.append(result)
    return results


def run_policy_player_grid(policies, player_types, n_games=10_000,
                            prize_values=None, opening_schedule=None, base_seed=42):
    """
    Run simulations for every (policy, player_type) combination.
    Returns a flat DataFrame of all results.
    """
    all_results = []
    for i, policy in enumerate(policies):
        for j, pt in enumerate(player_types):
            seed = base_seed + i * 100 + j
            results = run_simulations(policy, pt, n_games, prize_values, opening_schedule, seed)
            all_results.extend(results)
    df = pd.DataFrame(all_results)
    # Drop the offers list column for the raw CSV (keep scalar columns)
    return df.drop(columns=["offers"], errors="ignore")
