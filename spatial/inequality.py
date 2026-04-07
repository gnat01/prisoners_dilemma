"""
inequality.py — Lorenz curves and Gini coefficients for the spatial PD.

Mirrors the three plots from tournament/inequality.py but operates on the
spatial simulation history dict (per-generation strategy + score grids).

Because players only face their neighbours (not everyone), within-strategy
variance is genuine — neighbourhood luck matters. Watch Gini collapse to
zero as TfT fixates and all payoffs equalise.

Plots produced (saved alongside existing spatial plots):
  07_spatial_lorenz_by_strategy.png   — per-strategy Lorenz, final generation
  08_spatial_lorenz_population.png    — whole-population Lorenz, final generation
  09_spatial_gini_bar.png             — Gini bar chart, final generation
  10_spatial_gini_over_generations.png — Gini trajectory across all generations
"""

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tournament.inequality import lorenz, gini
from tournament.visualizations import STRATEGY_COLORS, STRATEGY_ORDER

PLOTS_DIR = Path("prisoners_dilemma/spatial/plots")


# ── History → per-generation DataFrame ───────────────────────────────────────

def history_to_df(history: dict, generation: int) -> pd.DataFrame:
    """
    Returns a flat DataFrame for one generation with columns:
      player_id, strategy, total_payoff, avg_payoff
    """
    strat_grid  = history["strategy_grids"][generation]
    score_grid  = history["score_grids"][generation]
    avg_grid    = history["avg_score_grids"][generation]
    side        = strat_grid.shape[0]

    rows = []
    for r in range(side):
        for c in range(side):
            rows.append({
                "player_id":    r * side + c,
                "strategy":     strat_grid[r, c],
                "total_payoff": score_grid[r, c],
                "avg_payoff":   avg_grid[r, c],
            })
    return pd.DataFrame(rows)


def compute_gini_table(df: pd.DataFrame) -> pd.DataFrame:
    """Gini per strategy present in df, plus whole population."""
    rows = []
    present = [s for s in STRATEGY_ORDER if s in df["strategy"].values]
    for strat in present:
        vals = df[df["strategy"] == strat]["total_payoff"].values
        if len(vals) < 2:
            continue
        rows.append({
            "strategy":    strat,
            "gini":        gini(vals),
            "n_players":   len(vals),
            "mean_payoff": vals.mean(),
            "std_payoff":  vals.std(),
        })
    all_vals = df["total_payoff"].values
    rows.append({
        "strategy":    "Whole Population",
        "gini":        gini(all_vals),
        "n_players":   len(all_vals),
        "mean_payoff": all_vals.mean(),
        "std_payoff":  all_vals.std(),
    })
    return pd.DataFrame(rows).sort_values("gini").reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _equality_line(ax):
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2,
            alpha=0.7, label="Perfect equality", zorder=1)


def plot_strategy_lorenz(df: pd.DataFrame, generation: int,
                         save_path: Optional[Path] = None) -> None:
    """Per-strategy Lorenz curves for a single generation."""
    present = [s for s in STRATEGY_ORDER if s in df["strategy"].values]

    fig, ax = plt.subplots(figsize=(9, 7))
    _equality_line(ax)

    for strat in present:
        vals = df[df["strategy"] == strat]["total_payoff"].values
        if len(vals) < 2:
            continue
        x, y = lorenz(vals)
        g = gini(vals)
        color = STRATEGY_COLORS.get(strat, "#999")
        ax.plot(x, y, color=color, linewidth=2.2, label=f"{strat}  (Gini={g:.4f})")
        ax.fill_between(x, x, y, alpha=0.07, color=color)

    ax.set_xlabel("Cumulative share of players (ranked by payoff)", fontsize=12)
    ax.set_ylabel("Cumulative share of total payoff", fontsize=12)
    ax.set_title(
        f"Spatial PD — Lorenz Curves by Strategy\nGeneration {generation}  ·  "
        "Deviation from diagonal = inequality within strategy",
        fontweight="bold", pad=14,
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_population_lorenz(df: pd.DataFrame, generation: int,
                           save_path: Optional[Path] = None) -> None:
    """Whole-population Lorenz curve for a single generation."""
    vals = df["total_payoff"].values
    x, y = lorenz(vals)
    g = gini(vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    _equality_line(ax)
    ax.plot(x, y, color="#8e44ad", linewidth=2.5, label=f"Population  (Gini={g:.4f})")
    ax.fill_between(x, x, y, alpha=0.15, color="#8e44ad", label="Inequality area")

    ax.set_xlabel("Cumulative share of players (ranked by payoff)", fontsize=12)
    ax.set_ylabel("Cumulative share of total payoff", fontsize=12)
    ax.set_title(
        f"Spatial PD — Whole-Population Lorenz Curve\n"
        f"Generation {generation}  ·  Gini = {g:.4f}",
        fontweight="bold", pad=14,
    )
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_gini_bar(gini_table: pd.DataFrame, generation: int,
                  save_path: Optional[Path] = None) -> None:
    """Horizontal Gini bar chart for a single generation."""
    df = gini_table.copy()
    colors = [
        STRATEGY_COLORS.get(s, "#8e44ad") for s in df["strategy"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["strategy"], df["gini"], color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, df["gini"]):
        ax.text(bar.get_width() + df["gini"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Gini coefficient  (0 = perfect equality, 1 = maximal inequality)", fontsize=11)
    ax.set_title(
        f"Spatial PD — Payoff Inequality by Strategy\n"
        f"Generation {generation}  ·  Higher Gini = neighbourhood luck matters more",
        fontweight="bold", pad=14,
    )
    ax.set_xlim(0, df["gini"].max() * 1.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_gini_over_generations(history: dict,
                               save_path: Optional[Path] = None) -> None:
    """
    Line chart: Gini coefficient per strategy + whole population across all generations.
    The collapse toward zero as TfT fixates is the key visual.
    """
    G = len(history["strategy_grids"])
    records = []

    for gen in range(G):
        df = history_to_df(history, gen)
        present = [s for s in STRATEGY_ORDER if s in df["strategy"].values]
        for strat in present:
            vals = df[df["strategy"] == strat]["total_payoff"].values
            if len(vals) < 2:
                continue
            records.append({"generation": gen + 1, "strategy": strat, "gini": gini(vals)})
        all_vals = df["total_payoff"].values
        records.append({"generation": gen + 1, "strategy": "Whole Population", "gini": gini(all_vals)})

    gini_df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Per-strategy lines
    for strat in STRATEGY_ORDER:
        sub = gini_df[gini_df["strategy"] == strat]
        if sub.empty:
            continue
        ax.plot(sub["generation"], sub["gini"],
                color=STRATEGY_COLORS.get(strat, "#999"),
                linewidth=2.0, marker="o", markersize=4, label=strat)

    # Whole-population line — dashed purple, prominent
    pop = gini_df[gini_df["strategy"] == "Whole Population"]
    ax.plot(pop["generation"], pop["gini"],
            color="#8e44ad", linewidth=2.5, linestyle="--",
            marker="s", markersize=5, label="Whole Population", zorder=5)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Gini coefficient", fontsize=12)
    ax.set_title(
        "Spatial PD — Payoff Inequality Across Generations\n"
        "Gini collapses toward 0 as TfT fixates and all payoffs equalise",
        fontweight="bold", pad=14,
    )
    ax.set_xlim(1, G)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(1, G + 1))
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Summary print ─────────────────────────────────────────────────────────────

def print_summary(gini_table: pd.DataFrame, generation: int) -> None:
    print(f"\n── Spatial Gini Coefficients — Generation {generation} ──────────────")
    print(f"{'Strategy':<22} {'Gini':>7}  {'Mean payoff':>12}  {'Std':>8}  {'N':>5}")
    print("─" * 62)
    for _, row in gini_table.iterrows():
        print(f"{row['strategy']:<22} {row['gini']:>7.4f}  "
              f"{row['mean_payoff']:>12.1f}  {row['std_payoff']:>8.2f}  {int(row['n_players']):>5}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_inequality_analysis(history: dict, output_dir: Path = PLOTS_DIR) -> None:
    """
    Full inequality analysis on a spatial simulation history dict.
    Generates all four plots. Called from spatial/main.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    G = len(history["strategy_grids"])

    # Use generation 1 for the three static plots — most interesting because
    # strategies are still mixed and neighbourhood luck creates real variance.
    # (Final generation is all-TfT with zero Gini — boring visually.)
    target_gen = 0
    df_gen1 = history_to_df(history, target_gen)
    gini_table = compute_gini_table(df_gen1)
    print_summary(gini_table, generation=target_gen + 1)

    print("\nGenerating inequality plots...")
    plot_strategy_lorenz(
        df_gen1, generation=target_gen + 1,
        save_path=output_dir / "07_spatial_lorenz_by_strategy.png",
    )
    plot_population_lorenz(
        df_gen1, generation=target_gen + 1,
        save_path=output_dir / "08_spatial_lorenz_population.png",
    )
    plot_gini_bar(
        gini_table, generation=target_gen + 1,
        save_path=output_dir / "09_spatial_gini_bar.png",
    )
    plot_gini_over_generations(
        history,
        save_path=output_dir / "10_spatial_gini_over_generations.png",
    )
