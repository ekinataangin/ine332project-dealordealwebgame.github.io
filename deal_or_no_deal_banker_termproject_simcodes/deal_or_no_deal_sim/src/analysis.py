"""
analysis.py - Statistical analysis and convergence testing
"""

import numpy as np
import pandas as pd
from src.game import PRIZE_VALUES, OPENING_SCHEDULE
from src.players import LogisticAcceptancePlayer, PlayerType
from src.simulation import run_simulations


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for banker profit grouped by policy and player_type.
    """
    rows = []
    for (policy, player_type), grp in df.groupby(["policy", "player_type"]):
        profits = grp["banker_profit"].values
        accepted = grp["accepted"].values
        offers = grp["accepted_offer"].dropna().values
        rounds = grp["round_of_acceptance"].dropna().values

        rows.append({
            "Policy": policy,
            "Player Type": player_type,
            "Mean Profit ($)": np.mean(profits),
            "Std Dev ($)": np.std(profits, ddof=1),
            "Variance ($²)": np.var(profits, ddof=1),
            "Median Profit ($)": np.median(profits),
            "5th Pct ($)": np.percentile(profits, 5),
            "95th Pct ($)": np.percentile(profits, 95),
            "Acceptance Rate": np.mean(accepted),
            "Avg Accepted Offer ($)": np.mean(offers) if len(offers) > 0 else np.nan,
            "Avg Round of Acceptance": np.mean(rounds) if len(rounds) > 0 else np.nan,
            "P(Profit < 0)": np.mean(profits < 0),
        })
    return pd.DataFrame(rows)


def compute_confidence_interval(values, confidence=0.95):
    """
    Compute mean and 95% CI: mean ± z * (std / sqrt(n))
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    z = 1.96  # 95% CI
    se = std / np.sqrt(n)
    return mean, mean - z * se, mean + z * se


def convergence_analysis(policy, player_type: PlayerType,
                          sizes=(100, 500, 1000, 2500, 5000, 10000, 20000),
                          prize_values=None, opening_schedule=None, seed=42):
    """
    Run increasing numbers of simulations to check convergence of mean profit.
    Returns a DataFrame with columns: n, mean, ci_lower, ci_upper, std, se
    """
    prize_values = prize_values or PRIZE_VALUES
    opening_schedule = opening_schedule or OPENING_SCHEDULE

    # Run max size once, then subsample
    max_n = max(sizes)
    results = run_simulations(policy, player_type, max_n, prize_values, opening_schedule, seed)
    profits = np.array([r["banker_profit"] for r in results])

    rows = []
    for n in sizes:
        sample = profits[:n]
        mean, lo, hi = compute_confidence_interval(sample)
        rows.append({
            "n": n,
            "mean_profit": mean,
            "ci_lower": lo,
            "ci_upper": hi,
            "std": np.std(sample, ddof=1),
            "se": np.std(sample, ddof=1) / np.sqrt(n),
        })
    return pd.DataFrame(rows)
