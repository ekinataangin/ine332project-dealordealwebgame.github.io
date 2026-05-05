"""
main.py - Entry point for Deal or No Deal Banker Simulation Project

Run:  python main.py
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.game import PRIZE_VALUES, OPENING_SCHEDULE
from src.policies import (BaselinePolicy, RiskAdjustedPolicy,
                           DynamicPolicy, PlayerAdaptivePolicy)
from src.players import PlayerType
from src.simulation import run_policy_player_grid
from src.analysis import compute_summary_statistics, convergence_analysis
from src.visualization import (plot_expected_profit_comparison,
                                plot_profit_distribution,
                                plot_convergence,
                                plot_acceptance_rates)

# ─── Configuration ────────────────────────────────────────────────────────────
RANDOM_SEED   = 42
N_FINAL_GAMES = 10_000
OUTPUT_DIR    = "outputs"
FIGURES_DIR   = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CONV_SIZES = [100, 500, 1000, 2500, 5000, 10_000, 20_000]

# ─── Policies & Players ───────────────────────────────────────────────────────
POLICIES = [
    BaselinePolicy(),
    RiskAdjustedPolicy(alpha=0.85, beta=0.05),
    DynamicPolicy(early_mult=0.65, late_mult=0.95),
    PlayerAdaptivePolicy(),
]

PLAYER_TYPES = [
    PlayerType.RISK_AVERSE,
    PlayerType.RISK_NEUTRAL,
    PlayerType.RISK_SEEKING,
]


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ─── 1. Convergence Analysis ──────────────────────────────────────────────────
print_header("STEP 1 — Convergence Analysis")
print("  Running convergence analysis: Baseline Policy × Risk-Neutral Player …")

baseline = POLICIES[0]
conv_df = convergence_analysis(
    policy=baseline,
    player_type=PlayerType.RISK_NEUTRAL,
    sizes=CONV_SIZES,
    prize_values=PRIZE_VALUES,
    opening_schedule=OPENING_SCHEDULE,
    seed=RANDOM_SEED,
)

conv_path = os.path.join(OUTPUT_DIR, "convergence_results.csv")
conv_df.to_csv(conv_path, index=False)
print(f"\n  Convergence table (Baseline × Risk-Neutral):")
print(conv_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# Determine final N based on convergence (stabilization within 1% of final estimate)
final_mean = conv_df["mean_profit"].iloc[-1]
stable_n = N_FINAL_GAMES
for _, row in conv_df.iterrows():
    if abs(row["mean_profit"] - final_mean) / (abs(final_mean) + 1) < 0.01:
        stable_n = int(row["n"])
        break

print(f"\n  → Estimates stabilise at n ≈ {stable_n:,}. Using {N_FINAL_GAMES:,} runs for final analysis.")

plot_convergence(
    conv_df,
    save_path=os.path.join(FIGURES_DIR, "convergence_plot.png"),
    label="Baseline Policy × Risk-Neutral Player"
)

# ─── 2. Full Simulations ──────────────────────────────────────────────────────
print_header("STEP 2 — Full Monte Carlo Simulations")
print(f"  Policies : {len(POLICIES)}")
print(f"  Players  : {len(PLAYER_TYPES)}")
print(f"  Games    : {N_FINAL_GAMES:,} per combination  ({len(POLICIES)*len(PLAYER_TYPES)*N_FINAL_GAMES:,} total)")
print("  Running …")

raw_df = run_policy_player_grid(
    policies=POLICIES,
    player_types=PLAYER_TYPES,
    n_games=N_FINAL_GAMES,
    prize_values=PRIZE_VALUES,
    opening_schedule=OPENING_SCHEDULE,
    base_seed=RANDOM_SEED,
)

raw_path = os.path.join(OUTPUT_DIR, "raw_simulation_results.csv")
raw_df.to_csv(raw_path, index=False)
print(f"  Saved raw results → {raw_path}  ({len(raw_df):,} rows)")

# ─── 3. Summary Statistics ────────────────────────────────────────────────────
print_header("STEP 3 — Summary Statistics")
summary_df = compute_summary_statistics(raw_df)
summary_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
summary_df.to_csv(summary_path, index=False)

pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 150)
print(summary_df.to_string(index=False))

# ─── 4. Visualizations ───────────────────────────────────────────────────────
print_header("STEP 4 — Generating Visualizations")
plot_expected_profit_comparison(
    summary_df,
    save_path=os.path.join(FIGURES_DIR, "expected_profit_comparison.png")
)
plot_profit_distribution(
    raw_df,
    save_path=os.path.join(FIGURES_DIR, "profit_distribution_boxplot.png")
)
plot_acceptance_rates(
    summary_df,
    save_path=os.path.join(FIGURES_DIR, "acceptance_rate_comparison.png")
)

# ─── 5. Final Terminal Summary ────────────────────────────────────────────────
print_header("FINAL SUMMARY — Top Policies by Mean Banker Profit")
best = (summary_df
        .groupby("Policy")["Mean Profit ($)"]
        .mean()
        .reset_index()
        .sort_values("Mean Profit ($)", ascending=False))
print(best.to_string(index=False, float_format=lambda x: f"${x:,.2f}"))

print("\n✓ All outputs saved to:", OUTPUT_DIR)
print("✓ Figures saved to    :", FIGURES_DIR)
print()
