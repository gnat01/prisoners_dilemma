"""
inequality.py — Lorenz curves and Gini coefficients for the PD tournament.

Can be run standalone (reads from results CSVs) or imported and called
with a player_overall DataFrame directly.

Usage:
    python prisoners_dilemma/tournament/inequality.py
    python prisoners_dilemma/tournament/inequality.py --results_dir prisoners_dilemma/tournament/results
    python prisoners_dilemma/tournament/inequality.py --spatial_dir prisoners_dilemma/spatial  --generations 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tournament.visualizations import STRATEGY_COLORS, STRATEGY_ORDER

PLOTS_DIR = Path("prisoners_dilemma/tournament/plots")

# ── Core maths ────────────────────────────────────────────────────────────────


def lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (x, y) for the Lorenz curve of `values`.
    x: cumulative population share (0 → 1)
    y: cumulative payoff share (0 → 1)
    Both arrays start at (0, 0).
    """
    sorted_vals = np.sort(values)
    cumulative = np.cumsum(sorted_vals)
    n = len(sorted_vals)
    x = np.linspace(0, 1, n + 1)
    y = np.concatenate([[0], cumulative / cumulative[-1]])
    return x, y


def gini(values: np.ndarray) -> float:
    """
    Gini coefficient via the standard area formula.
    0 = perfect equality, 1 = maximal inequality.
    """
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * sorted_vals) / (n * sorted_vals.sum())) - (n + 1) / n


# ── Analysis ──────────────────────────────────────────────────────────────────


def compute_gini_table(player_overall: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with Gini coefficient per strategy + whole population."""
    rows = []
    for strat in STRATEGY_ORDER:
        subset = player_overall[player_overall["strategy"] == strat]["total_payoff"].values
        if len(subset) < 2:
            continue
        rows.append({"strategy": strat, "gini": gini(subset), "n_players": len(subset),
                     "mean_payoff": subset.mean(), "std_payoff": subset.std()})

    # Whole population
    all_vals = player_overall["total_payoff"].values
    rows.append({"strategy": "Whole Population", "gini": gini(all_vals),
                 "n_players": len(all_vals), "mean_payoff": all_vals.mean(),
                 "std_payoff": all_vals.std()})

    return pd.DataFrame(rows).sort_values("gini").reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────


def _equality_line(ax):
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2,
            alpha=0.7, label="Perfect equality", zorder=1)


def plot_strategy_lorenz(player_overall: pd.DataFrame, save_path: Path = None) -> None:
    """One Lorenz curve per strategy, all on the same axes."""
    fig, ax = plt.subplots(figsize=(9, 7))
    _equality_line(ax)

    for strat in STRATEGY_ORDER:
        subset = player_overall[player_overall["strategy"] == strat]["total_payoff"].values
        if len(subset) < 2:
            continue
        x, y = lorenz(subset)
        g = gini(subset)
        color = STRATEGY_COLORS.get(strat, "#999")
        ax.plot(x, y, color=color, linewidth=2.2, label=f"{strat}  (Gini={g:.4f})")
        ax.fill_between(x, x, y, alpha=0.07, color=color)

    ax.set_xlabel("Cumulative share of players (ranked by payoff)", fontsize=12)
    ax.set_ylabel("Cumulative share of total payoff", fontsize=12)
    ax.set_title("Lorenz Curves by Strategy\nDeviation from diagonal = inequality within strategy",
                 fontweight="bold", pad=14)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_population_lorenz(player_overall: pd.DataFrame, save_path: Path = None) -> None:
    """Single Lorenz curve for the whole population."""
    all_vals = player_overall["total_payoff"].values
    x, y = lorenz(all_vals)
    g = gini(all_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    _equality_line(ax)
    ax.plot(x, y, color="#8e44ad", linewidth=2.5, label=f"Population  (Gini={g:.4f})")
    ax.fill_between(x, x, y, alpha=0.15, color="#8e44ad", label="Inequality area")

    ax.set_xlabel("Cumulative share of players (ranked by payoff)", fontsize=12)
    ax.set_ylabel("Cumulative share of total payoff", fontsize=12)
    ax.set_title(f"Whole-Population Lorenz Curve\nGini = {g:.4f}",
                 fontweight="bold", pad=14)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_gini_bar(gini_table: pd.DataFrame, save_path: Path = None) -> None:
    """Horizontal bar chart of Gini coefficients, ranked low to high."""
    df = gini_table.copy()

    colors = [
        STRATEGY_COLORS.get(s, "#8e44ad") if s != "Whole Population" else "#8e44ad"
        for s in df["strategy"]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["strategy"], df["gini"], color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, df["gini"]):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    ax.set_xlabel("Gini coefficient  (0 = perfect equality, 1 = maximal inequality)", fontsize=11)
    ax.set_title("Payoff Inequality by Strategy\nLower Gini = more equal outcomes within that strategy",
                 fontweight="bold", pad=14)
    ax.set_xlim(0, df["gini"].max() * 1.2)
    ax.axvline(0, color="gray", linewidth=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_spatial_gini_over_generations(spatial_dir: Path, generations: int,
                                       save_path: Path = None) -> None:
    """
    Loads per-generation score grids from the spatial simulation history
    and plots Gini coefficient across generations.
    Requires the spatial simulation to have been run and history saved as npy files,
    OR recomputes from the spatial plots dir if numpy arrays are present.

    Falls back gracefully if spatial data is not available.
    """
    # Look for saved numpy history
    score_path = spatial_dir / "plots" / "score_grids.npy"
    if not score_path.exists():
        print(f"  Spatial score grids not found at {score_path} — skipping spatial Gini plot.")
        print("  Run the spatial simulation with --save-history to generate this plot.")
        return

    score_grids = np.load(str(score_path))  # shape: (G, side, side)
    G = min(generations, score_grids.shape[0])
    ginis = [gini(score_grids[g].flatten()) for g in range(G)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, G + 1), ginis, marker="o", color="#2980b9", linewidth=2.2, markersize=6)
    ax.fill_between(range(1, G + 1), ginis, alpha=0.12, color="#2980b9")
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Gini coefficient", fontsize=12)
    ax.set_title("Payoff Inequality Across Generations (Spatial Model)\n"
                 "Gini → 0 as TfT fixates and all payoffs equalise",
                 fontweight="bold", pad=14)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(1, G + 1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Print summary ─────────────────────────────────────────────────────────────


def print_summary(gini_table: pd.DataFrame) -> None:
    print("\n── Gini Coefficients ────────────────────────────────────────")
    print(f"{'Strategy':<22} {'Gini':>7}  {'Mean payoff':>12}  {'Std':>8}  {'N':>5}")
    print("─" * 62)
    for _, row in gini_table.iterrows():
        print(f"{row['strategy']:<22} {row['gini']:>7.4f}  "
              f"{row['mean_payoff']:>12.1f}  {row['std_payoff']:>8.2f}  {int(row['n_players']):>5}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main(results_dir: Path, spatial_dir: Path = None, generations: int = 20) -> None:
    player_overall_path = results_dir / "player_overall.csv"
    if not player_overall_path.exists():
        print(f"ERROR: {player_overall_path} not found.")
        print("Run the tournament first:  python -m prisoners_dilemma.tournament.main")
        sys.exit(1)

    print(f"Loading results from {results_dir}/")
    player_overall = pd.read_csv(player_overall_path)

    gini_table = compute_gini_table(player_overall)
    print_summary(gini_table)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")
    plot_strategy_lorenz(player_overall,
                         save_path=PLOTS_DIR / "07_lorenz_by_strategy.png")
    plot_population_lorenz(player_overall,
                           save_path=PLOTS_DIR / "08_lorenz_population.png")
    plot_gini_bar(gini_table,
                  save_path=PLOTS_DIR / "09_gini_coefficients.png")

    if spatial_dir:
        plot_spatial_gini_over_generations(
            spatial_dir, generations,
            save_path=PLOTS_DIR / "10_spatial_gini_over_generations.png",
        )

    print(f"\nPlots saved → {PLOTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lorenz curves & Gini coefficients for the PD tournament")
    parser.add_argument("--results_dir",  type=Path, default=Path("prisoners_dilemma/tournament/results"),
                        help="Directory containing player_overall.csv")
    parser.add_argument("--spatial_dir",  type=Path, default=None,
                        help="Spatial simulation directory (optional, for cross-generation Gini)")
    parser.add_argument("--generations",  type=int,  default=20,
                        help="Number of spatial generations to plot (if spatial_dir provided)")
    args = parser.parse_args()

    main(results_dir=args.results_dir, spatial_dir=args.spatial_dir, generations=args.generations)
