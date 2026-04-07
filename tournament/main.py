"""
main.py — Entry point for the Prisoner's Dilemma population tournament.

Usage:
    python -m prisoners_dilemma.main
    python -m prisoners_dilemma.main --n 200 --seed 7
"""

import argparse
import time
from pathlib import Path

import numpy as np

from prisoners_dilemma.analysis import (
    cooperation_rate_df,
    matches_to_df,
    player_match_df,
    player_overall_df,
    strategy_overall_df,
    strategy_round_df,
    strategy_vs_strategy_df,
)
from prisoners_dilemma.simulation import create_population, run_tournament
from prisoners_dilemma.visualizations import plot_all

RESULTS_DIR = Path("prisoners_dilemma/results")
PLOTS_DIR = Path("prisoners_dilemma/plots")


def print_banner(n: int, num_rounds: int) -> None:
    total_matches = n * (n - 1) // 2
    print(f"\n{'='*62}")
    print(f"  Prisoner's Dilemma  |  Population Tournament")
    print(f"  N = {n} players  |  {num_rounds} rounds / match  |  {total_matches:,} total matches")
    print(f"{'='*62}\n")


def print_strategy_summary(strategy_overall) -> None:
    print("\n── Strategy Overall Performance ─────────────────────────────")
    cols = ["strategy", "n_players", "avg_total_payoff", "std_total_payoff", "avg_per_round"]
    print(strategy_overall[cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))


def print_vs_matrix(vs_df) -> None:
    print("\n── Avg Payoff Per Round (row strategy vs column opponent) ───")
    print(vs_df.round(2).to_string())


def save_results(match_df, pm_df, po_df, sr_df, so_df, vs_df) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    match_df.to_csv(RESULTS_DIR / "match_results.csv",          index=False)
    pm_df.to_csv(   RESULTS_DIR / "player_match_results.csv",   index=False)
    po_df.to_csv(   RESULTS_DIR / "player_overall.csv",         index=False)
    sr_df.to_csv(   RESULTS_DIR / "strategy_round.csv",         index=False)
    so_df.to_csv(   RESULTS_DIR / "strategy_overall.csv",       index=False)
    vs_df.to_csv(   RESULTS_DIR / "strategy_vs_strategy.csv")
    print(f"\nResults saved → {RESULTS_DIR}/")


def main(n: int = 1000, num_rounds: int = 5, seed: int = 42) -> None:
    np.random.seed(seed)
    print_banner(n, num_rounds)

    # ── Population ────────────────────────────────────────────────
    print("Creating population...")
    players = create_population(n)
    strategy_counts = {}
    for p in players:
        strategy_counts[p.strategy_name] = strategy_counts.get(p.strategy_name, 0) + 1
    for strat, count in strategy_counts.items():
        print(f"  {count:4d}  {strat}")

    # ── Tournament ────────────────────────────────────────────────
    print(f"\nRunning {n*(n-1)//2:,} matches...")
    t0 = time.perf_counter()
    records = run_tournament(players, num_rounds=num_rounds, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"Tournament finished in {elapsed:.1f}s")

    # ── Analysis ──────────────────────────────────────────────────
    print("\nBuilding analysis DataFrames...")
    match_df = matches_to_df(records)
    pm_df    = player_match_df(match_df)
    po_df    = player_overall_df(pm_df)
    sr_df    = strategy_round_df(pm_df)
    so_df    = strategy_overall_df(po_df)
    vs_df    = strategy_vs_strategy_df(pm_df)
    cr_df    = cooperation_rate_df(pm_df)

    print_strategy_summary(so_df)
    print_vs_matrix(vs_df)
    save_results(match_df, pm_df, po_df, sr_df, so_df, vs_df)

    # ── Visualisations ────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_all(so_df, sr_df, po_df, vs_df, cr_df, output_dir=PLOTS_DIR)
    print(f"Plots saved → {PLOTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prisoner's Dilemma population tournament")
    parser.add_argument("--n",          type=int, default=1000, help="Population size (must be divisible by 10)")
    parser.add_argument("--rounds",     type=int, default=5,    help="Rounds per match")
    parser.add_argument("--seed",       type=int, default=42,   help="Random seed")
    args = parser.parse_args()

    main(n=args.n, num_rounds=args.rounds, seed=args.seed)
