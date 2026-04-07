"""
visualizations.py — All plots for the Prisoner's Dilemma tournament.

Plots generated:
  01_strategy_overall.png      — bar chart: avg total payoff ± std per strategy
  02_round_by_round.png        — line chart: avg payoff per round per strategy
  03_payoff_distribution.png   — violin plot: total payoff distribution per strategy
  04_strategy_vs_strategy.png  — heatmap: avg payoff/round when X faces Y
  05_cooperation_rate.png      — line chart: cooperation rate per round per strategy
  06_player_strip.png          — strip plot: individual player payoffs per strategy
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Canonical ordering & colour palette ──────────────────────────────────────

STRATEGY_ORDER = [
    "Always Cooperate",
    "Always Defect",
    "Random(p=0.2)",
    "Random(p=0.4)",
    "Random(p=0.6)",
    "Random(p=0.8)",
    "Tit-for-Tat",
]

STRATEGY_COLORS = {
    "Always Cooperate": "#27ae60",
    "Always Defect":    "#e74c3c",
    "Random(p=0.2)":    "#f1c40f",
    "Random(p=0.4)":    "#e67e22",
    "Random(p=0.6)":    "#d35400",
    "Random(p=0.8)":    "#922b21",
    "Tit-for-Tat":      "#2980b9",
}

NUM_ROUNDS = 5


def _present_strategies(df: pd.DataFrame, col: str = "strategy") -> list:
    """Return STRATEGY_ORDER filtered to those present in df[col]."""
    present = set(df[col].unique())
    return [s for s in STRATEGY_ORDER if s in present]


def _colors_for(strategies: list) -> list:
    return [STRATEGY_COLORS.get(s, "#999999") for s in strategies]


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ── Plot 1: Strategy Overall Performance ─────────────────────────────────────


def plot_strategy_overall(
    strategy_overall: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Horizontal bar chart of avg total payoff ± std per strategy."""
    order = _present_strategies(strategy_overall)
    df = strategy_overall.set_index("strategy").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = _colors_for(order)

    bars = ax.barh(df["strategy"], df["avg_total_payoff"], color=colors, edgecolor="white", height=0.6)
    ax.errorbar(
        df["avg_total_payoff"],
        range(len(df)),
        xerr=df["std_total_payoff"],
        fmt="none",
        ecolor="gray",
        capsize=5,
        linewidth=1.4,
    )

    # Value labels
    for bar, val, std in zip(bars, df["avg_total_payoff"], df["std_total_payoff"]):
        ax.text(
            bar.get_width() + std + df["avg_total_payoff"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.1f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Average Total Payoff (across all opponents)", fontsize=12)
    ax.set_title(
        "Strategy Overall Performance\nAverage total payoff ± 1 std across players of that strategy",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlim(0, df["avg_total_payoff"].max() * 1.18)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Plot 2: Round-by-Round Payoff ─────────────────────────────────────────────


def plot_round_by_round(
    strategy_round: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Line chart: average payoff per round within a match, one line per strategy."""
    order = _present_strategies(strategy_round)

    fig, ax = plt.subplots(figsize=(10, 6))

    for strat in order:
        sub = strategy_round[strategy_round["strategy"] == strat].sort_values("round")
        if sub.empty:
            continue
        color = STRATEGY_COLORS.get(strat, "#999")
        ax.plot(sub["round"], sub["avg_payoff"], marker="o", color=color, label=strat, linewidth=2.2, markersize=7)
        ax.fill_between(
            sub["round"],
            sub["avg_payoff"] - sub["std_payoff"],
            sub["avg_payoff"] + sub["std_payoff"],
            alpha=0.12,
            color=color,
        )

    ax.set_xlabel("Round number (within match)", fontsize=12)
    ax.set_ylabel("Average payoff", fontsize=12)
    ax.set_title(
        "Round-by-Round Average Payoff by Strategy\nShaded band = ±1 std",
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(range(1, NUM_ROUNDS + 1))
    ax.set_xlim(0.8, NUM_ROUNDS + 0.2)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Plot 3: Payoff Distribution (Violin) ─────────────────────────────────────


def plot_payoff_distribution(
    player_overall: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Violin + box plot of each player's total payoff, faceted by strategy."""
    order = _present_strategies(player_overall)
    df = player_overall.copy()
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    df = df.sort_values("strategy")

    fig, ax = plt.subplots(figsize=(13, 6))

    sns.violinplot(
        data=df,
        x="strategy",
        y="total_payoff",
        hue="strategy",
        order=order,
        palette=STRATEGY_COLORS,
        inner="box",
        cut=0,
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Total payoff (sum across all matches)", fontsize=12)
    ax.set_title(
        "Distribution of Total Payoff by Strategy\nViolin = density · Box = IQR · Line = median",
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Plot 4: Strategy vs Strategy Heatmap ─────────────────────────────────────


def plot_strategy_heatmap(
    vs_matrix: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Heatmap: rows = focal strategy, cols = opponent strategy.
    Cell value = avg payoff per round for the focal strategy.
    """
    row_order = [s for s in STRATEGY_ORDER if s in vs_matrix.index]
    col_order = [s for s in STRATEGY_ORDER if s in vs_matrix.columns]
    vs = vs_matrix.reindex(index=row_order, columns=col_order)

    fig, ax = plt.subplots(figsize=(11, 8))

    sns.heatmap(
        vs,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.6,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Avg payoff per round (focal strategy)", "shrink": 0.85},
        vmin=0,
        vmax=5,
    )

    ax.set_title(
        "Strategy vs Strategy — Avg Payoff Per Round\nRow = focal strategy · Column = opponent",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Opponent strategy", fontsize=12)
    ax.set_ylabel("Focal strategy", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Plot 5: Cooperation Rate by Round ─────────────────────────────────────────


def plot_cooperation_rate(
    coop_rate_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Line chart: cooperation rate per round per strategy."""
    order = _present_strategies(coop_rate_df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for strat in order:
        sub = coop_rate_df[coop_rate_df["strategy"] == strat].sort_values("round")
        if sub.empty:
            continue
        color = STRATEGY_COLORS.get(strat, "#999")
        ax.plot(
            sub["round"],
            sub["cooperation_rate"],
            marker="o",
            color=color,
            label=strat,
            linewidth=2.2,
            markersize=7,
        )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.45, linewidth=1.2, label="50% reference")
    ax.set_xlabel("Round number (within match)", fontsize=12)
    ax.set_ylabel("Cooperation rate", fontsize=12)
    ax.set_title(
        "Cooperation Rate by Round and Strategy",
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(range(1, NUM_ROUNDS + 1))
    ax.set_xlim(0.8, NUM_ROUNDS + 0.2)
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(loc="center right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Plot 6: Individual Player Strip ──────────────────────────────────────────


def plot_player_strip(
    player_overall: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """Strip plot of individual player total payoffs, with mean overlaid."""
    order = _present_strategies(player_overall)
    df = player_overall.copy()
    df["strategy"] = pd.Categorical(df["strategy"], categories=order, ordered=True)
    df = df.sort_values("strategy")

    fig, ax = plt.subplots(figsize=(13, 6))

    sns.stripplot(
        data=df,
        x="strategy",
        y="total_payoff",
        hue="strategy",
        order=order,
        palette=STRATEGY_COLORS,
        jitter=0.28,
        alpha=0.35,
        size=3.5,
        legend=False,
        ax=ax,
    )

    # Mean bar per strategy
    means = df.groupby("strategy", observed=True)["total_payoff"].mean().reindex(order)
    for i, mean_val in enumerate(means):
        if not np.isnan(mean_val):
            ax.hlines(mean_val, i - 0.38, i + 0.38, colors="black", linewidth=2.5, zorder=5)

    ax.set_xlabel("Strategy", fontsize=12)
    ax.set_ylabel("Total payoff", fontsize=12)
    ax.set_title(
        "Individual Player Payoffs by Strategy\nEach dot = one player · Black bar = mean",
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Orchestrator ──────────────────────────────────────────────────────────────


def plot_all(
    strategy_overall: pd.DataFrame,
    strategy_round: pd.DataFrame,
    player_overall: pd.DataFrame,
    vs_matrix: pd.DataFrame,
    coop_rate: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> None:
    setup_style()

    def sp(name: str) -> Optional[Path]:
        if output_dir is None:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / name

    plot_strategy_overall(strategy_overall,    sp("01_strategy_overall.png"))
    plot_round_by_round(strategy_round,        sp("02_round_by_round.png"))
    plot_payoff_distribution(player_overall,   sp("03_payoff_distribution.png"))
    plot_strategy_heatmap(vs_matrix,           sp("04_strategy_vs_strategy.png"))
    plot_cooperation_rate(coop_rate,           sp("05_cooperation_rate.png"))
    plot_player_strip(player_overall,          sp("06_player_strip.png"))
