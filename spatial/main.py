"""
main.py — Entry point for the Spatial Prisoner's Dilemma.

Usage:
    python -m prisoners_dilemma.spatial.main
    python prisoners_dilemma/spatial/main.py --side 32 --generations 20 --neighbourhood moore

CLI flags:
    --side          Side length of the grid (total players = side²). Default: 32.
    --generations   Number of evolutionary generations (G). Default: 20.
    --neighbourhood von_neumann (4 neighbours) or moore (8 neighbours). Default: moore.
    --rounds        Rounds per match within a generation. Default: 5.
    --fps           Frames per second for animations. Default: 4.
    --seed          Random seed. Default: 42.
    --output-dir    Directory for all plots and animations. Default: prisoners_dilemma/spatial/plots.
    --no-anim       Skip animation generation (useful for quick runs).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial.grid import run_simulation, STRATEGY_NAMES
from spatial.visualizations import animate_all, plot_final_static, setup_style
from spatial.inequality import run_inequality_analysis

DEFAULT_OUTPUT_DIR = Path("prisoners_dilemma/spatial/plots")


def print_banner(side: int, generations: int, neighbourhood: str, rounds: int) -> None:
    print(f"\n{'='*62}")
    print(f"  Spatial Prisoner's Dilemma")
    print(f"  Grid: {side}×{side} = {side**2} players  |  {neighbourhood.replace('_', ' ').title()} neighbourhood")
    print(f"  {generations} generations  |  {rounds} rounds/match")
    print(f"{'='*62}\n")


def print_strategy_summary(history: dict) -> None:
    print("\n── Strategy composition over generations ────────────────────")
    header = f"{'Gen':>4}  " + "  ".join(f"{s[:12]:>12}" for s in STRATEGY_NAMES)
    print(header)

    for i, sg in enumerate(history["strategy_grids"]):
        total = sg.size
        counts = {s: (sg == s).sum() for s in STRATEGY_NAMES}
        row = f"{i+1:>4}  " + "  ".join(f"{counts[s]:>12}" for s in STRATEGY_NAMES)
        print(row)

    print()
    final = history["strategy_grids"][-1]
    print("── Final generation payoffs ─────────────────────────────────")
    scores = history["score_grids"][-1]
    avg = history["avg_score_grids"][-1]
    print(f"  Total payoff  — min: {scores.min():.1f}  max: {scores.max():.1f}  mean: {scores.mean():.2f}")
    print(f"  Avg per round — min: {avg.min():.2f}  max: {avg.max():.2f}  mean: {avg.mean():.2f}")


def main(
    side: int = 32,
    generations: int = 20,
    neighbourhood: str = "moore",
    rounds: int = 5,
    fps: int = 4,
    seed: int = 42,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    no_anim: bool = False,
) -> None:
    print_banner(side, generations, neighbourhood, rounds)

    print(f"Running {generations} generations on a {side}×{side} grid...")
    t0 = time.perf_counter()
    history = run_simulation(
        side=side,
        generations=generations,
        neighbourhood=neighbourhood,
        num_rounds=rounds,
        seed=seed,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"Simulation complete in {elapsed:.1f}s")

    print_strategy_summary(history)

    setup_style()

    print("\nSaving static plots (final generation)...")
    plot_final_static(history, output_dir=output_dir)

    if not no_anim:
        print("\nGenerating animations (this may take a moment)...")
        animate_all(history, output_dir=output_dir, fps=fps)

    print("\nRunning inequality analysis...")
    run_inequality_analysis(history, output_dir=output_dir)

    print(f"\nAll output saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatial Prisoner's Dilemma")
    parser.add_argument("--side",           type=int,   default=32,            help="Grid side length (players = side²)")
    parser.add_argument("--generations",    type=int,   default=20,            help="Number of evolutionary generations (G)")
    parser.add_argument("--neighbourhood",  type=str,   default="moore",       choices=["moore", "von_neumann"], help="Neighbourhood type")
    parser.add_argument("--rounds",         type=int,   default=5,             help="Rounds per match")
    parser.add_argument("--fps",            type=int,   default=4,             help="Animation frames per second")
    parser.add_argument("--seed",           type=int,   default=42,            help="Random seed")
    parser.add_argument("--output-dir",     type=Path,  default=DEFAULT_OUTPUT_DIR, help="Directory for all plots and animations")
    parser.add_argument("--no-anim",        action="store_true",               help="Skip animation generation")
    args = parser.parse_args()

    main(
        side=args.side,
        generations=args.generations,
        neighbourhood=args.neighbourhood,
        rounds=args.rounds,
        fps=args.fps,
        seed=args.seed,
        output_dir=args.output_dir,
        no_anim=args.no_anim,
    )
