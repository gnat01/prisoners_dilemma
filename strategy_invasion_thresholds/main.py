"""
main.py — Entry point for the spatial invasion threshold analysis.

For every (resident, invader) strategy pair, sweeps the invader starting
fraction from 0.01 → 0.95, runs the spatial PD across multiple seeds,
and identifies the minimum fraction at which the invader wins.

Usage:
    python prisoners_dilemma/strategy_invasion_thresholds/main.py
    python prisoners_dilemma/strategy_invasion_thresholds/main.py --side 20 --generations 25 --n-seeds 5

CLI flags — see --help for full list.
"""

import argparse
import sys
import time
from pathlib import Path

import os
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_invasion_thresholds.invasion import (
    DEFAULT_FRACTIONS,
    STRATEGY_NAMES,
    aggregate_results,
    compute_threshold_matrix,
    run_full_sweep,
    threshold_matrix_wide,
)
from strategy_invasion_thresholds.visualizations import plot_all, setup_style

DEFAULT_OUTPUT_DIR = Path("prisoners_dilemma/strategy_invasion_thresholds/results")
DEFAULT_PLOTS_DIR  = Path("prisoners_dilemma/strategy_invasion_thresholds/plots")


def print_banner(side, generations, n_seeds, fractions, neighbourhood, n_workers):
    n_pairs = len(STRATEGY_NAMES) * (len(STRATEGY_NAMES) - 1)
    total   = n_pairs * len(fractions) * n_seeds
    print(f"\n{'='*64}")
    print(f"  Spatial Invasion Threshold Analysis")
    print(f"  Grid: {side}×{side}  |  {neighbourhood.replace('_',' ').title()} neighbourhood")
    print(f"  {generations} max generations  |  {n_seeds} seeds/point  |  {n_workers} workers")
    print(f"  {len(fractions)} fraction points  |  {n_pairs} strategy pairs  |  {total:,} total trials")
    print(f"{'='*64}\n")


def print_threshold_summary(wide: pd.DataFrame):
    short = {
        "Always Cooperate": "AC",   "Always Defect":  "AD",
        "Random(p=0.2)":   "R0.2", "Random(p=0.4)": "R0.4",
        "Random(p=0.6)":   "R0.6", "Random(p=0.8)": "R0.8",
        "Tit-for-Tat":     "TfT",
    }
    print("\n── Invasion Threshold Matrix (f*) ───────────────────────────────")
    print("   Rows = resident  ·  Cols = invader  ·  NaN = never invades\n")
    renamed = wide.copy()
    renamed.index   = [short.get(s, s) for s in wide.index]
    renamed.columns = [short.get(s, s) for s in wide.columns]
    print(renamed.round(2).to_string(na_rep="  X "))


def main(
    side: int = 20,
    generations: int = 25,
    n_seeds: int = 5,
    fractions: list = DEFAULT_FRACTIONS,
    neighbourhood: str = "moore",
    rounds: int = 5,
    base_seed: int = 42,
    n_workers: int = 4,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
) -> None:
    print_banner(side, generations, n_seeds, fractions, neighbourhood, n_workers)

    t0 = time.perf_counter()
    raw = run_full_sweep(
        fractions=fractions,
        side=side,
        neighbourhood=neighbourhood,
        num_rounds=rounds,
        max_generations=generations,
        n_seeds=n_seeds,
        base_seed=base_seed,
        n_workers=n_workers,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nSweep complete in {elapsed:.1f}s")

    print("Aggregating results...")
    agg  = aggregate_results(raw)
    thresh_df = compute_threshold_matrix(agg)
    wide = threshold_matrix_wide(thresh_df)

    print_threshold_summary(wide)

    # Save CSVs
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.drop(columns=["fraction_history"]).to_csv(output_dir / "raw_trials.csv",    index=False)
    agg.to_csv(output_dir / "aggregated.csv",  index=False)
    thresh_df.to_csv(output_dir / "thresholds.csv", index=False)
    wide.to_csv(output_dir / "threshold_matrix.csv")
    print(f"\nResults saved → {output_dir}/")

    print("\nGenerating plots...")
    plot_all(raw, agg, wide, output_dir=plots_dir)
    print(f"Plots saved → {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spatial Prisoner's Dilemma — Invasion Threshold Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--side",          type=int,   default=20,
                        help="Grid side length (players = side²)")
    parser.add_argument("--generations",   type=int,   default=25,
                        help="Max generations per trial")
    parser.add_argument("--n-seeds",       type=int,   default=5,
                        help="Random seeds per (pair, fraction) point")
    parser.add_argument("--neighbourhood", type=str,   default="moore",
                        choices=["moore", "von_neumann"],
                        help="Neighbourhood type")
    parser.add_argument("--rounds",        type=int,   default=5,
                        help="PD rounds per match within a generation")
    parser.add_argument("--seed",          type=int,   default=42,
                        help="Base random seed")
    parser.add_argument("--n-workers",     type=int,   default=4,
                        help="Parallel workers (ProcessPoolExecutor)")
    parser.add_argument("--output-dir",    type=Path,  default=DEFAULT_OUTPUT_DIR,
                        help="Directory for CSV results")
    parser.add_argument("--plots-dir",     type=Path,  default=DEFAULT_PLOTS_DIR,
                        help="Directory for plots")
    args = parser.parse_args()

    main(
        side=args.side,
        generations=args.generations,
        n_seeds=args.n_seeds,
        neighbourhood=args.neighbourhood,
        rounds=args.rounds,
        base_seed=args.seed,
        n_workers=args.n_workers,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
    )
