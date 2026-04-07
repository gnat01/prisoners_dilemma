"""
grid.py — Spatial Prisoner's Dilemma on a 2-D grid.

Each player occupies one cell of a √N × √N toroidal grid.
Each generation:
  1. Every player plays num_rounds rounds against each of their neighbours.
  2. Each player adopts the strategy of their highest-scoring neighbour
     (including themselves) — imitate-the-best update rule.

Reuses strategies and play_match from tournament/simulation.py.
"""

import sys
from pathlib import Path

import numpy as np

# Make tournament importable regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).parent.parent))

from tournament.simulation import (
    AlwaysCooperate,
    AlwaysDefect,
    RandomCooperate,
    TitForTat,
    play_match,
    Player,
    PAYOFFS,
    C, D,
)

# ── Neighbourhood helpers ─────────────────────────────────────────────────────

def von_neumann_neighbours(row: int, col: int, rows: int, cols: int):
    """4-connectivity, toroidal wrap."""
    return [
        ((row - 1) % rows, col),
        ((row + 1) % rows, col),
        (row, (col - 1) % cols),
        (row, (col + 1) % cols),
    ]


def moore_neighbours(row: int, col: int, rows: int, cols: int):
    """8-connectivity, toroidal wrap."""
    return [
        ((row + dr) % rows, (col + dc) % cols)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]


NEIGHBOURHOOD_FN = {
    "von_neumann": von_neumann_neighbours,
    "moore": moore_neighbours,
}

# ── Strategy catalogue (same set as tournament) ────────────────────────────────

def all_strategies():
    return [
        AlwaysCooperate(),
        AlwaysDefect(),
        RandomCooperate(0.2),
        RandomCooperate(0.4),
        RandomCooperate(0.6),
        RandomCooperate(0.8),
        TitForTat(),
    ]

STRATEGY_NAMES = [s.name for s in all_strategies()]


def strategy_by_name(name: str):
    for s in all_strategies():
        if s.name == name:
            return s
    raise ValueError(f"Unknown strategy: {name}")


# ── Grid initialisation ───────────────────────────────────────────────────────

def make_grid(side: int, seed: int = 42) -> np.ndarray:
    """
    Returns a (side, side) object array of Player instances.
    Strategies are assigned uniformly at random from the full catalogue.
    """
    rng = np.random.default_rng(seed)
    strategies = all_strategies()
    grid = np.empty((side, side), dtype=object)
    pid = 0
    for r in range(side):
        for c in range(side):
            strat = strategies[rng.integers(len(strategies))]
            # Give each player a fresh instance so state doesn't leak
            grid[r, c] = Player(pid, strategy_by_name(strat.name))
            pid += 1
    return grid


# ── Single generation ─────────────────────────────────────────────────────────

def run_generation(
    grid: np.ndarray,
    neighbourhood: str = "moore",
    num_rounds: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Plays one full generation on the grid.

    Returns:
      scores      — (side, side) float array: total payoff per cell this generation
      avg_scores  — (side, side) float array: avg payoff per round per cell
      new_grid    — (side, side) updated Player array after imitate-best
    """
    side = grid.shape[0]
    neighbour_fn = NEIGHBOURHOOD_FN[neighbourhood]
    scores = np.zeros((side, side), dtype=float)

    # ── Phase 1: play all matches ─────────────────────────────────────────────
    for r in range(side):
        for c in range(side):
            player = grid[r, c]
            for nr, nc in neighbour_fn(r, c, side, side):
                neighbour = grid[nr, nc]
                result = play_match(player, neighbour, num_rounds)
                # Accumulate from player_a perspective (player is always a)
                scores[r, c] += result["total_payoff_a"]

    avg_scores = scores / (len(neighbour_fn(0, 0, side, side)) * num_rounds)

    # ── Phase 2: imitate-best update ─────────────────────────────────────────
    new_grid = np.empty((side, side), dtype=object)
    pid_offset = side * side * 1000  # ensure fresh unique ids

    for r in range(side):
        for c in range(side):
            # Candidates: self + all neighbours
            candidates = [(r, c)] + neighbour_fn(r, c, side, side)
            best_rc = max(candidates, key=lambda pos: scores[pos[0], pos[1]])
            best_strategy_name = grid[best_rc[0], best_rc[1]].strategy_name
            new_pid = grid[r, c].player_id + pid_offset
            new_grid[r, c] = Player(new_pid, strategy_by_name(best_strategy_name))

    return scores, avg_scores, new_grid


# ── Full simulation ───────────────────────────────────────────────────────────

def run_simulation(
    side: int,
    generations: int,
    neighbourhood: str = "moore",
    num_rounds: int = 5,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Runs the spatial PD for `generations` generations.

    Returns a dict with per-generation snapshots:
      strategy_grids  — list of (side, side) str arrays (strategy name per cell)
      score_grids     — list of (side, side) float arrays (total payoff)
      avg_score_grids — list of (side, side) float arrays (avg payoff per round)
    """
    np.random.seed(seed)
    grid = make_grid(side, seed=seed)

    history = {
        "strategy_grids": [],
        "score_grids": [],
        "avg_score_grids": [],
    }

    for gen in range(generations):
        if verbose:
            print(f"  Generation {gen + 1}/{generations}...", end="\r", flush=True)

        scores, avg_scores, new_grid = run_generation(grid, neighbourhood, num_rounds)

        # Record strategy snapshot before update
        strat_grid = np.vectorize(lambda p: p.strategy_name)(grid)
        history["strategy_grids"].append(strat_grid)
        history["score_grids"].append(scores.copy())
        history["avg_score_grids"].append(avg_scores.copy())

        grid = new_grid

    if verbose:
        print()  # newline after \r

    return history
