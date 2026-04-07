"""
main_fast_payoff_sweep.py — Fast invasion-threshold runner with configurable
Prisoner's Dilemma payoffs.

This keeps the same accelerated integer-grid engine as main_fast.py, but lets
the user supply PD payoffs via:
  --R  reward for mutual cooperation
  --S  sucker's payoff
  --T  temptation to defect
  --P  punishment for mutual defection

Validation is strict PD-only:
  T > R > P > S
  2R > T + S
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial.grid import STRATEGY_NAMES
from strategy_invasion_thresholds.invasion import (
    DEFAULT_FRACTIONS,
    aggregate_results,
    compute_threshold_matrix,
    threshold_matrix_wide,
)

DEFAULT_OUTPUT_DIR = Path("prisoners_dilemma/strategy_invasion_thresholds/results_fast_payoff_sweep")
DEFAULT_PLOTS_DIR = Path("prisoners_dilemma/strategy_invasion_thresholds/plots_fast_payoff_sweep")

AC = STRATEGY_NAMES.index("Always Cooperate")
AD = STRATEGY_NAMES.index("Always Defect")
R02 = STRATEGY_NAMES.index("Random(p=0.2)")
R04 = STRATEGY_NAMES.index("Random(p=0.4)")
R06 = STRATEGY_NAMES.index("Random(p=0.6)")
R08 = STRATEGY_NAMES.index("Random(p=0.8)")
TFT = STRATEGY_NAMES.index("Tit-for-Tat")

RANDOM_P = {
    R02: 0.2,
    R04: 0.4,
    R06: 0.6,
    R08: 0.8,
}

NEIGHBOUR_OFFSETS = {
    "von_neumann": [(-1, 0), (1, 0), (0, -1), (0, 1)],
    "moore": [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ],
}


def validate_pd_payoffs(R: float, S: float, T: float, P: float) -> None:
    if not (T > R > P > S):
        raise ValueError(
            f"Invalid PD payoffs: require T > R > P > S, got T={T}, R={R}, P={P}, S={S}"
        )
    if not (2 * R > T + S):
        raise ValueError(
            f"Invalid PD payoffs: require 2R > T + S, got 2R={2 * R} and T+S={T + S}"
        )


def print_banner(
    side: int,
    generations: int,
    n_seeds: int,
    fractions: list[float],
    neighbourhood: str,
    n_workers: int,
    payoffs: dict[str, float],
) -> None:
    n_pairs = len(STRATEGY_NAMES) * (len(STRATEGY_NAMES) - 1)
    total = n_pairs * len(fractions) * n_seeds
    print(f"\n{'=' * 76}")
    print("  Spatial Invasion Threshold Analysis — FAST PAYOFF SWEEP")
    print(f"  Grid: {side}x{side}  |  {neighbourhood.replace('_', ' ').title()} neighbourhood")
    print(f"  {generations} max generations  |  {n_seeds} seeds/point  |  {n_workers} workers")
    print(f"  Payoffs: T={payoffs['T']}  R={payoffs['R']}  P={payoffs['P']}  S={payoffs['S']}")
    print(f"  {len(fractions)} fraction points  |  {n_pairs} strategy pairs  |  {total:,} total trials")
    print(f"{'=' * 76}\n")


def print_threshold_summary(wide: pd.DataFrame) -> None:
    short = {
        "Always Cooperate": "AC",
        "Always Defect": "AD",
        "Random(p=0.2)": "R0.2",
        "Random(p=0.4)": "R0.4",
        "Random(p=0.6)": "R0.6",
        "Random(p=0.8)": "R0.8",
        "Tit-for-Tat": "TfT",
    }
    print("\n-- Invasion Threshold Matrix (f*) --------------------------------")
    print("   Rows = resident  |  Cols = invader  |  NaN = never invades\n")
    renamed = wide.copy()
    renamed.index = [short.get(s, s) for s in wide.index]
    renamed.columns = [short.get(s, s) for s in wide.columns]
    print(renamed.round(2).to_string(na_rep="  X "))


def make_seed(base_seed: int, pair_index: int, fraction_index: int, seed_index: int) -> int:
    return int(base_seed + pair_index * 1_000_003 + fraction_index * 10_007 + seed_index * 101)


def make_numeric_grid(
    side: int,
    resident_id: int,
    invader_id: int,
    invader_fraction: float,
    seed: int,
) -> np.ndarray:
    n = side * side
    n_invaders = max(1, round(n * invader_fraction))
    grid = np.full(n, resident_id, dtype=np.int8)
    rng = np.random.default_rng(seed)
    invader_positions = rng.choice(n, size=n_invaders, replace=False)
    grid[invader_positions] = invader_id
    return grid.reshape((side, side))


def _categorical_payoff_sample(
    count: int,
    num_rounds: int,
    probabilities: list[float],
    payoff_values: tuple[float, float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    draws = rng.multinomial(num_rounds, probabilities, size=count)
    payoff = (
        draws[:, 0] * payoff_values[0]
        + draws[:, 1] * payoff_values[1]
        + draws[:, 2] * payoff_values[2]
        + draws[:, 3] * payoff_values[3]
    )
    return payoff.astype(np.float32, copy=False)


def payoff_samples_for_pair(
    strategy_a: int,
    strategy_b: int,
    count: int,
    num_rounds: int,
    rng: np.random.Generator,
    payoffs: dict[str, float],
) -> np.ndarray | float:
    if count == 0:
        return 0.0

    R = payoffs["R"]
    S = payoffs["S"]
    T = payoffs["T"]
    P = payoffs["P"]

    if strategy_a == AC and strategy_b == AC:
        return float(R * num_rounds)
    if strategy_a == AC and strategy_b == AD:
        return float(S * num_rounds)
    if strategy_a == AD and strategy_b == AC:
        return float(T * num_rounds)
    if strategy_a == AD and strategy_b == AD:
        return float(P * num_rounds)
    if strategy_a == TFT and strategy_b == TFT:
        return float(R * num_rounds)
    if strategy_a == TFT and strategy_b == AC:
        return float(R * num_rounds)
    if strategy_a == AC and strategy_b == TFT:
        return float(R * num_rounds)
    if strategy_a == TFT and strategy_b == AD:
        return float(S + (num_rounds - 1) * P)
    if strategy_a == AD and strategy_b == TFT:
        return float(T + (num_rounds - 1) * P)

    p_a = RANDOM_P.get(strategy_a)
    p_b = RANDOM_P.get(strategy_b)

    if p_a is not None and strategy_b == AC:
        cooperations = rng.binomial(num_rounds, p_a, size=count)
        payoff = R * cooperations + T * (num_rounds - cooperations)
        return payoff.astype(np.float32, copy=False)

    if strategy_a == AC and p_b is not None:
        cooperations = rng.binomial(num_rounds, p_b, size=count)
        payoff = R * cooperations + S * (num_rounds - cooperations)
        return payoff.astype(np.float32, copy=False)

    if p_a is not None and strategy_b == AD:
        cooperations = rng.binomial(num_rounds, p_a, size=count)
        payoff = S * cooperations + P * (num_rounds - cooperations)
        return payoff.astype(np.float32, copy=False)

    if strategy_a == AD and p_b is not None:
        cooperations = rng.binomial(num_rounds, p_b, size=count)
        payoff = T * cooperations + P * (num_rounds - cooperations)
        return payoff.astype(np.float32, copy=False)

    if p_a is not None and p_b is not None:
        p_cc = p_a * p_b
        p_cd = p_a * (1.0 - p_b)
        p_dc = (1.0 - p_a) * p_b
        p_dd = (1.0 - p_a) * (1.0 - p_b)
        return _categorical_payoff_sample(
            count=count,
            num_rounds=num_rounds,
            probabilities=[p_cc, p_cd, p_dc, p_dd],
            payoff_values=(R, S, T, P),
            rng=rng,
        )

    if p_a is not None and strategy_b == TFT:
        random_actions = rng.random((count, num_rounds)) < p_a
        tft_actions = np.empty_like(random_actions)
        tft_actions[:, 0] = True
        if num_rounds > 1:
            tft_actions[:, 1:] = random_actions[:, :-1]
        payoff = (
            R * np.sum(random_actions & tft_actions, axis=1)
            + S * np.sum(random_actions & (~tft_actions), axis=1)
            + T * np.sum((~random_actions) & tft_actions, axis=1)
            + P * np.sum((~random_actions) & (~tft_actions), axis=1)
        )
        return payoff.astype(np.float32, copy=False)

    if strategy_a == TFT and p_b is not None:
        random_actions = rng.random((count, num_rounds)) < p_b
        tft_actions = np.empty_like(random_actions)
        tft_actions[:, 0] = True
        if num_rounds > 1:
            tft_actions[:, 1:] = random_actions[:, :-1]
        payoff = (
            R * np.sum(tft_actions & random_actions, axis=1)
            + S * np.sum(tft_actions & (~random_actions), axis=1)
            + T * np.sum((~tft_actions) & random_actions, axis=1)
            + P * np.sum((~tft_actions) & (~random_actions), axis=1)
        )
        return payoff.astype(np.float32, copy=False)

    raise ValueError(f"Unsupported strategy pair: {strategy_a}, {strategy_b}")


def run_generation_fast(
    grid: np.ndarray,
    resident_id: int,
    invader_id: int,
    neighbourhood: str,
    num_rounds: int,
    rng: np.random.Generator,
    payoffs: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = NEIGHBOUR_OFFSETS[neighbourhood]
    scores = np.zeros(grid.shape, dtype=np.float32)

    for dr, dc in offsets:
        neighbour_grid = np.roll(grid, shift=(dr, dc), axis=(0, 1))
        rr = (grid == resident_id) & (neighbour_grid == resident_id)
        ri = (grid == resident_id) & (neighbour_grid == invader_id)
        ir = (grid == invader_id) & (neighbour_grid == resident_id)
        ii = (grid == invader_id) & (neighbour_grid == invader_id)

        rr_count = int(rr.sum())
        if rr_count:
            scores[rr] += payoff_samples_for_pair(resident_id, resident_id, rr_count, num_rounds, rng, payoffs)

        ri_count = int(ri.sum())
        if ri_count:
            scores[ri] += payoff_samples_for_pair(resident_id, invader_id, ri_count, num_rounds, rng, payoffs)

        ir_count = int(ir.sum())
        if ir_count:
            scores[ir] += payoff_samples_for_pair(invader_id, resident_id, ir_count, num_rounds, rng, payoffs)

        ii_count = int(ii.sum())
        if ii_count:
            scores[ii] += payoff_samples_for_pair(invader_id, invader_id, ii_count, num_rounds, rng, payoffs)

    best_scores = scores.copy()
    best_grid = grid.copy()

    for dr, dc in offsets:
        candidate_scores = np.roll(scores, shift=(dr, dc), axis=(0, 1))
        better = candidate_scores > best_scores
        if np.any(better):
            best_scores[better] = candidate_scores[better]
            best_grid[better] = np.roll(grid, shift=(dr, dc), axis=(0, 1))[better]

    return scores, best_grid


def run_invasion_trial_fast(
    resident_name: str,
    invader_name: str,
    invader_fraction: float,
    side: int,
    neighbourhood: str,
    num_rounds: int,
    max_generations: int,
    seed: int,
    payoffs: dict[str, float],
) -> dict:
    resident_id = STRATEGY_NAMES.index(resident_name)
    invader_id = STRATEGY_NAMES.index(invader_name)
    total_cells = side * side
    rng = np.random.default_rng(seed)
    grid = make_numeric_grid(side, resident_id, invader_id, invader_fraction, seed)

    fraction_history = [float(np.count_nonzero(grid == invader_id) / total_cells)]

    for _ in range(max_generations):
        _, grid = run_generation_fast(
            grid=grid,
            resident_id=resident_id,
            invader_id=invader_id,
            neighbourhood=neighbourhood,
            num_rounds=num_rounds,
            rng=rng,
            payoffs=payoffs,
        )
        frac = float(np.count_nonzero(grid == invader_id) / total_cells)
        fraction_history.append(frac)
        if frac == 0.0 or frac == 1.0:
            break

    final = fraction_history[-1]
    return {
        "resident": resident_name,
        "invader": invader_name,
        "initial_fraction": invader_fraction,
        "final_fraction": final,
        "generations_run": len(fraction_history) - 1,
        "fixated": final in (0.0, 1.0),
        "invader_won": final > 0.5,
        "seed": seed,
        "fraction_history": fraction_history,
        "R": payoffs["R"],
        "S": payoffs["S"],
        "T": payoffs["T"],
        "P": payoffs["P"],
    }


def _worker(args: tuple) -> dict:
    return run_invasion_trial_fast(*args)


def _build_tasks(
    fractions: list[float],
    side: int,
    neighbourhood: str,
    num_rounds: int,
    max_generations: int,
    n_seeds: int,
    base_seed: int,
    payoffs: dict[str, float],
) -> list[tuple]:
    tasks: list[tuple] = []
    pair_index = 0
    for resident_name in STRATEGY_NAMES:
        for invader_name in STRATEGY_NAMES:
            if resident_name == invader_name:
                continue
            for fraction_index, fraction in enumerate(fractions):
                for seed_index in range(n_seeds):
                    seed = make_seed(base_seed, pair_index, fraction_index, seed_index)
                    tasks.append(
                        (
                            resident_name,
                            invader_name,
                            fraction,
                            side,
                            neighbourhood,
                            num_rounds,
                            max_generations,
                            seed,
                            payoffs,
                        )
                    )
            pair_index += 1
    return tasks


def run_full_sweep_fast(
    fractions: list[float],
    side: int,
    neighbourhood: str,
    num_rounds: int,
    max_generations: int,
    n_seeds: int,
    base_seed: int,
    n_workers: int,
    chunksize: int,
    verbose: bool,
    payoffs: dict[str, float],
) -> pd.DataFrame:
    tasks = _build_tasks(
        fractions=fractions,
        side=side,
        neighbourhood=neighbourhood,
        num_rounds=num_rounds,
        max_generations=max_generations,
        n_seeds=n_seeds,
        base_seed=base_seed,
        payoffs=payoffs,
    )

    if n_workers <= 1:
        iterator: Iterable[dict] = map(_worker, tasks)
        if verbose:
            iterator = tqdm(iterator, total=len(tasks), desc="Fast payoff-sweep trials", unit="trial")
        return pd.DataFrame.from_records(list(iterator))

    records: list[dict] = []
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            iterator = executor.map(_worker, tasks, chunksize=chunksize)
            if verbose:
                iterator = tqdm(iterator, total=len(tasks), desc="Fast payoff-sweep trials", unit="trial")
            records.extend(iterator)
    except (PermissionError, OSError) as exc:
        print(f"Process pool unavailable ({exc}); falling back to threads.")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            iterator = executor.map(_worker, tasks)
            if verbose:
                iterator = tqdm(iterator, total=len(tasks), desc="Fast payoff-sweep trials", unit="trial")
            records.extend(iterator)

    return pd.DataFrame.from_records(records)


def suggest_chunksize(total_tasks: int, n_workers: int) -> int:
    return max(1, math.ceil(total_tasks / max(1, n_workers * 8)))


def write_metadata(
    output_dir: Path,
    *,
    side: int,
    generations: int,
    n_seeds: int,
    neighbourhood: str,
    rounds: int,
    base_seed: int,
    n_workers: int,
    chunksize: int,
    fractions: list[float],
    payoffs: dict[str, float],
) -> None:
    metadata = {
        "side": side,
        "generations": generations,
        "n_seeds": n_seeds,
        "neighbourhood": neighbourhood,
        "rounds": rounds,
        "seed": base_seed,
        "n_workers": n_workers,
        "chunksize": chunksize,
        "fractions": fractions,
        "payoffs": payoffs,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main(
    side: int = 20,
    generations: int = 25,
    n_seeds: int = 5,
    fractions: list[float] = DEFAULT_FRACTIONS,
    neighbourhood: str = "moore",
    rounds: int = 5,
    base_seed: int = 42,
    n_workers: int | None = None,
    chunksize: int | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    plots_dir: Path = DEFAULT_PLOTS_DIR,
    skip_plots: bool = False,
    R: float = 3.0,
    S: float = 0.0,
    T: float = 5.0,
    P: float = 1.0,
) -> None:
    validate_pd_payoffs(R=R, S=S, T=T, P=P)
    payoffs = {"R": float(R), "S": float(S), "T": float(T), "P": float(P)}

    n_workers = n_workers or (os.cpu_count() or 1)
    total_tasks = len(STRATEGY_NAMES) * (len(STRATEGY_NAMES) - 1) * len(fractions) * n_seeds
    chunksize = chunksize or suggest_chunksize(total_tasks, n_workers)

    print_banner(side, generations, n_seeds, fractions, neighbourhood, n_workers, payoffs)
    print(f"Chunksize: {chunksize}")

    t0 = time.perf_counter()
    raw = run_full_sweep_fast(
        fractions=fractions,
        side=side,
        neighbourhood=neighbourhood,
        num_rounds=rounds,
        max_generations=generations,
        n_seeds=n_seeds,
        base_seed=base_seed,
        n_workers=n_workers,
        chunksize=chunksize,
        verbose=True,
        payoffs=payoffs,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nFast payoff sweep complete in {elapsed:.1f}s")

    print("Aggregating results...")
    agg = aggregate_results(raw)
    for key, value in payoffs.items():
        agg[key] = value

    thresh_df = compute_threshold_matrix(agg)
    for key, value in payoffs.items():
        thresh_df[key] = value

    wide = threshold_matrix_wide(thresh_df)
    print_threshold_summary(wide)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw.drop(columns=["fraction_history"]).to_csv(output_dir / "raw_trials.csv", index=False)
    agg.to_csv(output_dir / "aggregated.csv", index=False)
    thresh_df.to_csv(output_dir / "thresholds.csv", index=False)
    wide.to_csv(output_dir / "threshold_matrix.csv")
    write_metadata(
        output_dir,
        side=side,
        generations=generations,
        n_seeds=n_seeds,
        neighbourhood=neighbourhood,
        rounds=rounds,
        base_seed=base_seed,
        n_workers=n_workers,
        chunksize=chunksize,
        fractions=fractions,
        payoffs=payoffs,
    )
    print(f"\nResults saved -> {output_dir}/")

    if not skip_plots:
        from strategy_invasion_thresholds.visualizations import plot_all

        print("\nGenerating plots...")
        plot_all(raw, agg, wide, output_dir=plots_dir)
        print(f"Plots saved -> {plots_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast spatial Prisoner's Dilemma invasion-threshold analysis with configurable PD payoffs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--side", type=int, default=20, help="Grid side length (players = side^2)")
    parser.add_argument("--generations", type=int, default=25, help="Max generations per trial")
    parser.add_argument("--n-seeds", type=int, default=5, help="Random seeds per (pair, fraction) point")
    parser.add_argument(
        "--neighbourhood",
        type=str,
        default="moore",
        choices=["moore", "von_neumann"],
        help="Neighbourhood type",
    )
    parser.add_argument("--rounds", type=int, default=5, help="PD rounds per match within a generation")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel worker processes")
    parser.add_argument("--chunksize", type=int, default=None, help="Tasks per process batch")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV results")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="Directory for plots")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation to maximize throughput")
    parser.add_argument("--R", type=float, default=3.0, help="Reward for mutual cooperation")
    parser.add_argument("--S", type=float, default=0.0, help="Sucker's payoff")
    parser.add_argument("--T", type=float, default=5.0, help="Temptation to defect")
    parser.add_argument("--P", type=float, default=1.0, help="Punishment for mutual defection")
    args = parser.parse_args()

    main(
        side=args.side,
        generations=args.generations,
        n_seeds=args.n_seeds,
        neighbourhood=args.neighbourhood,
        rounds=args.rounds,
        base_seed=args.seed,
        n_workers=args.n_workers,
        chunksize=args.chunksize,
        output_dir=args.output_dir,
        plots_dir=args.plots_dir,
        skip_plots=args.skip_plots,
        R=args.R,
        S=args.S,
        T=args.T,
        P=args.P,
    )
