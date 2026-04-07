"""
invasion.py — Spatial invasion threshold analysis for the Prisoner's Dilemma.

For every (resident, invader) strategy pair:
  - Sweep invader starting fractions from 0.01 → 0.95
  - Run the spatial PD to fixation (or max_generations)
  - Repeat across n_seeds to capture spatial randomness
  - Record whether the invader won, lost, or stalled

The invasion threshold f* is the smallest starting fraction at which the
invader wins in the majority of seeds. Below f* the resident is stable;
above f* the invader takes over.

Reuses run_generation and strategy catalogue from spatial/grid.py.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial.grid import (
    STRATEGY_NAMES,
    NEIGHBOURHOOD_FN,
    run_generation,
    strategy_by_name,
)
from tournament.simulation import Player

# ── Default sweep parameters ──────────────────────────────────────────────────

DEFAULT_FRACTIONS = [
    0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
    0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95,
]

# ── Grid initialisation ───────────────────────────────────────────────────────


def make_invasion_grid(
    side: int,
    resident_name: str,
    invader_name: str,
    invader_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Creates a side×side grid populated with two strategies.
    Invader cells are placed uniformly at random.
    """
    rng = np.random.default_rng(seed)
    n = side * side
    n_invaders = max(1, round(n * invader_fraction))
    n_residents = n - n_invaders

    labels = [resident_name] * n_residents + [invader_name] * n_invaders
    rng.shuffle(labels)

    grid = np.empty((side, side), dtype=object)
    for pid, (r, c) in enumerate(
        (r, c) for r in range(side) for c in range(side)
    ):
        grid[r, c] = Player(pid, strategy_by_name(labels[pid]))

    return grid


# ── Single trial ──────────────────────────────────────────────────────────────


def run_invasion_trial(
    resident_name: str,
    invader_name: str,
    invader_fraction: float,
    side: int,
    neighbourhood: str,
    num_rounds: int,
    max_generations: int,
    seed: int,
) -> dict:
    """
    Runs one invasion trial to fixation or max_generations.

    Returns a dict with:
      resident, invader, initial_fraction, final_fraction,
      generations_run, fixated, invader_won, seed,
      fraction_history (list of invader fraction per generation)
    """
    np.random.seed(seed)
    grid = make_invasion_grid(side, resident_name, invader_name, invader_fraction, seed)

    fraction_history = [round((np.vectorize(lambda p: p.strategy_name)(grid) == invader_name).mean(), 6)]

    for gen in range(max_generations):
        _, _, grid = run_generation(grid, neighbourhood, num_rounds)
        strat_grid = np.vectorize(lambda p: p.strategy_name)(grid)
        frac = float((strat_grid == invader_name).mean())
        fraction_history.append(frac)

        if frac == 0.0 or frac == 1.0:
            break

    final = fraction_history[-1]
    return {
        "resident":         resident_name,
        "invader":          invader_name,
        "initial_fraction": invader_fraction,
        "final_fraction":   final,
        "generations_run":  len(fraction_history) - 1,
        "fixated":          final in (0.0, 1.0),
        "invader_won":      final > 0.5,
        "seed":             seed,
        "fraction_history": fraction_history,
    }


# ── Worker function (module-level for pickling) ───────────────────────────────

def _worker(args: tuple) -> dict:
    return run_invasion_trial(*args)


# ── Full sweep ────────────────────────────────────────────────────────────────


def run_full_sweep(
    fractions: list = DEFAULT_FRACTIONS,
    side: int = 20,
    neighbourhood: str = "moore",
    num_rounds: int = 5,
    max_generations: int = 25,
    n_seeds: int = 5,
    base_seed: int = 42,
    n_workers: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Runs the full (resident × invader × fraction × seed) sweep in parallel.
    Returns a flat DataFrame of all trial results.
    """
    pairs = [
        (res, inv)
        for res in STRATEGY_NAMES
        for inv in STRATEGY_NAMES
        if res != inv
    ]

    tasks = []
    for res, inv in pairs:
        for f in fractions:
            for s in range(n_seeds):
                seed = base_seed + s * 1000 + hash((res, inv, f)) % 1000
                tasks.append((res, inv, f, side, neighbourhood, num_rounds, max_generations, seed))

    records = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        it = tqdm(as_completed(futures), total=len(tasks), desc="Invasion trials", unit="trial") if verbose else as_completed(futures)
        for future in it:
            records.append(future.result())

    return pd.DataFrame(records)


# ── Aggregation ───────────────────────────────────────────────────────────────


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates trial results by (resident, invader, initial_fraction).

    Columns:
      mean_final_fraction, std_final_fraction,
      invasion_rate (fraction of seeds where invader won),
      mean_generations, n_seeds
    """
    agg = (
        df.groupby(["resident", "invader", "initial_fraction"])
        .agg(
            mean_final_fraction=("final_fraction", "mean"),
            std_final_fraction=("final_fraction", "std"),
            invasion_rate=("invader_won", "mean"),
            mean_generations=("generations_run", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    return agg


def compute_threshold_matrix(agg: pd.DataFrame) -> pd.DataFrame:
    """
    For each (resident, invader) pair, finds the invasion threshold f*:
    the smallest starting fraction at which invasion_rate >= 0.5.

    Uses linear interpolation between the bracketing fraction points.
    Returns NaN if the invader never wins in majority of seeds.
    Returns 0.0 if the invader wins even at the smallest tested fraction.
    """
    rows = []
    for (res, inv), grp in agg.groupby(["resident", "invader"]):
        grp = grp.sort_values("initial_fraction")
        rates = grp["invasion_rate"].values
        fracs = grp["initial_fraction"].values

        if rates.max() < 0.5:
            threshold = float("nan")  # never invades
        elif rates.min() >= 0.5:
            threshold = fracs[0]      # invades even at smallest fraction
        else:
            # Linear interpolation between last below-0.5 and first above-0.5
            idx = np.searchsorted(rates, 0.5)
            f_lo, r_lo = fracs[idx - 1], rates[idx - 1]
            f_hi, r_hi = fracs[idx],     rates[idx]
            threshold = f_lo + (0.5 - r_lo) * (f_hi - f_lo) / (r_hi - r_lo)

        rows.append({"resident": res, "invader": inv, "threshold": threshold})

    return pd.DataFrame(rows)


def threshold_matrix_wide(threshold_df: pd.DataFrame) -> pd.DataFrame:
    """Pivots threshold DataFrame to a (resident × invader) matrix."""
    return (
        threshold_df
        .pivot(index="resident", columns="invader", values="threshold")
        .reindex(index=STRATEGY_NAMES, columns=STRATEGY_NAMES)
    )
