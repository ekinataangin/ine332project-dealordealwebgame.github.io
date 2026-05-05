"""
visualization.py - All plot generation
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PALETTE = {
    "Baseline (0.8×EV)":           "#2196F3",
    "Risk-Adjusted (α·EV − β·σ)":  "#FF5722",
    "Dynamic Round-Based":          "#4CAF50",
    "Player-Adaptive":              "#9C27B0",
}

PLAYER_MARKERS = {
    "Risk-Averse":   "o",
    "Risk-Neutral":  "s",
    "Risk-Seeking":  "^",
}


def _fmt(x):
    """Format dollar amount."""
    if abs(x) >= 1000:
        return f"${x:,.0f}"
    return f"${x:.2f}"


def plot_expected_profit_comparison(summary_df: pd.DataFrame, save_path: str):
    """Grouped bar chart: mean banker profit by policy and player type."""
    policies = summary_df["Policy"].unique()
    player_types = summary_df["Player Type"].unique()
    n_pol = len(policies)
    n_pt = len(player_types)

    x = np.arange(n_pol)
    width = 0.22
    offsets = np.linspace(-(n_pt - 1) / 2 * width, (n_pt - 1) / 2 * width, n_pt)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0F1923")
    ax.set_facecolor("#0F1923")

    pt_colors = ["#64B5F6", "#81C784", "#FFB74D"]

    for j, pt in enumerate(player_types):
        subset = summary_df[summary_df["Player Type"] == pt]
        means = [subset[subset["Policy"] == p]["Mean Profit ($)"].values[0]
                 if len(subset[subset["Policy"] == p]) > 0 else 0 for p in policies]
        bars = ax.bar(x + offsets[j], means, width, label=pt,
                      color=pt_colors[j], alpha=0.88, edgecolor="#FFFFFF22")
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(means) * 0.01 if max(means) != 0 else 500),
                    _fmt(val), ha="center", va="bottom", fontsize=8.5,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(policies, color="white", fontsize=10, wrap=True)
    ax.set_ylabel("Mean Banker Profit ($)", color="white", fontsize=11)
    ax.set_title("Mean Banker Profit by Policy and Player Type", color="white",
                 fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#FFFFFF33")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", color="#FFFFFF22", linewidth=0.5)
    ax.axhline(0, color="#FFFFFF66", linewidth=0.8, linestyle="--")

    legend = ax.legend(facecolor="#1A2530", edgecolor="#FFFFFF33",
                        labelcolor="white", fontsize=10, title="Player Type",
                        title_fontsize=10)
    legend.get_title().set_color("white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def plot_profit_distribution(raw_df: pd.DataFrame, save_path: str):
    """Violin + boxplot of profit distributions by policy."""
    policies = raw_df["policy"].unique()
    n_pol = len(policies)

    fig, axes = plt.subplots(1, n_pol, figsize=(5 * n_pol, 7), sharey=False)
    if n_pol == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0F1923")

    pt_colors = {"Risk-Averse": "#64B5F6", "Risk-Neutral": "#81C784", "Risk-Seeking": "#FFB74D"}

    for ax, policy in zip(axes, policies):
        ax.set_facecolor("#0F1923")
        grp = raw_df[raw_df["policy"] == policy]
        player_types = grp["player_type"].unique()
        data = [grp[grp["player_type"] == pt]["banker_profit"].values for pt in player_types]
        colors = [pt_colors.get(pt, "#AAAAAA") for pt in player_types]

        parts = ax.violinplot(data, positions=range(len(player_types)),
                               showmedians=False, showextrema=False)
        for pc, col in zip(parts["bodies"], colors):
            pc.set_facecolor(col)
            pc.set_alpha(0.45)
            pc.set_edgecolor(col)

        bp = ax.boxplot(data, positions=range(len(player_types)),
                         patch_artist=True, widths=0.18,
                         medianprops=dict(color="white", linewidth=2),
                         whiskerprops=dict(color="#FFFFFF99"),
                         capprops=dict(color="#FFFFFF99"),
                         flierprops=dict(marker=".", color="#FFFFFF44", markersize=2))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        ax.set_xticks(range(len(player_types)))
        ax.set_xticklabels(player_types, color="white", fontsize=9)
        ax.set_title(policy, color="white", fontsize=10, fontweight="bold", pad=8)
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#FFFFFF33")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.grid(axis="y", color="#FFFFFF22", linewidth=0.5)
        ax.axhline(0, color="#FF5252AA", linewidth=1, linestyle="--")
        ax.set_ylabel("Banker Profit ($)" if ax == axes[0] else "", color="white")

    fig.suptitle("Banker Profit Distribution by Policy and Player Type",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def plot_convergence(conv_df: pd.DataFrame, save_path: str, label: str = ""):
    """
    Convergence plot: mean profit ± 95% CI as a function of simulation count.
    conv_df must have columns: n, mean_profit, ci_lower, ci_upper
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0F1923")
    ax.set_facecolor("#0F1923")

    ax.plot(conv_df["n"], conv_df["mean_profit"], color="#64B5F6",
            linewidth=2.5, marker="o", markersize=6, label="Mean Profit")
    ax.fill_between(conv_df["n"], conv_df["ci_lower"], conv_df["ci_upper"],
                    alpha=0.25, color="#64B5F6", label="95% CI")

    final_mean = conv_df["mean_profit"].iloc[-1]
    ax.axhline(final_mean, color="#FFFFFF55", linewidth=1, linestyle="--",
               label=f"Final estimate: {_fmt(final_mean)}")

    ax.set_xscale("log")
    ax.set_xlabel("Number of Simulations (log scale)", color="white", fontsize=11)
    ax.set_ylabel("Mean Banker Profit ($)", color="white", fontsize=11)
    title = f"Convergence of Mean Banker Profit Estimate"
    if label:
        title += f"\n({label})"
    ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#FFFFFF33")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(color="#FFFFFF22", linewidth=0.5)
    legend = ax.legend(facecolor="#1A2530", edgecolor="#FFFFFF33", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


def plot_acceptance_rates(summary_df: pd.DataFrame, save_path: str):
    """Grouped bar chart comparing acceptance rates."""
    policies = summary_df["Policy"].unique()
    player_types = summary_df["Player Type"].unique()
    n_pt = len(player_types)
    x = np.arange(len(policies))
    width = 0.22
    offsets = np.linspace(-(n_pt - 1) / 2 * width, (n_pt - 1) / 2 * width, n_pt)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0F1923")
    ax.set_facecolor("#0F1923")

    pt_colors = ["#64B5F6", "#81C784", "#FFB74D"]

    for j, pt in enumerate(player_types):
        subset = summary_df[summary_df["Player Type"] == pt]
        rates = [subset[subset["Policy"] == p]["Acceptance Rate"].values[0]
                 if len(subset[subset["Policy"] == p]) > 0 else 0 for p in policies]
        bars = ax.bar(x + offsets[j], [r * 100 for r in rates], width,
                      label=pt, color=pt_colors[j], alpha=0.88, edgecolor="#FFFFFF22")
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val * 100:.1f}%", ha="center", va="bottom",
                    fontsize=8.5, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(policies, color="white", fontsize=10)
    ax.set_ylabel("Acceptance Rate (%)", color="white", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Offer Acceptance Rate by Policy and Player Type",
                 color="white", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#FFFFFF33")
    ax.grid(axis="y", color="#FFFFFF22", linewidth=0.5)
    legend = ax.legend(facecolor="#1A2530", edgecolor="#FFFFFF33",
                        labelcolor="white", fontsize=10, title="Player Type",
                        title_fontsize=10)
    legend.get_title().set_color("white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")
