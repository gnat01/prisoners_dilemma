"""
visualizations.py — Spatial PD plots and animations.

Static plots (final generation):
  01_strategy_map_final.png     — strategy per cell
  02_avg_payoff_final.png       — avg payoff per round per cell
  03_total_payoff_final.png     — total payoff per cell

Animations (across all generations):
  anim_strategy_map.gif         — strategy clusters evolving
  anim_avg_payoff.gif           — avg payoff heatmap evolving
  anim_total_payoff.gif         — total payoff heatmap evolving
"""

from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from grid import STRATEGY_NAMES

# ── Colour palette (matches tournament) ───────────────────────────────────────

STRATEGY_COLORS = {
    "Always Cooperate": "#27ae60",
    "Always Defect":    "#e74c3c",
    "Random(p=0.2)":   "#f1c40f",
    "Random(p=0.4)":   "#e67e22",
    "Random(p=0.6)":   "#d35400",
    "Random(p=0.8)":   "#922b21",
    "Tit-for-Tat":     "#2980b9",
}

_STRAT_INDEX = {name: i for i, name in enumerate(STRATEGY_NAMES)}
_CMAP_STRAT = mpl.colors.ListedColormap([STRATEGY_COLORS[s] for s in STRATEGY_NAMES])
_NORM_STRAT = mpl.colors.BoundaryNorm(range(len(STRATEGY_NAMES) + 1), len(STRATEGY_NAMES))


def _strat_grid_to_int(strat_grid: np.ndarray) -> np.ndarray:
    return np.vectorize(_STRAT_INDEX.get)(strat_grid)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


# ── Static plots ──────────────────────────────────────────────────────────────

def plot_strategy_map(strat_grid: np.ndarray, generation: int, save_path: Optional[Path] = None) -> None:
    present = [s for s in STRATEGY_NAMES if s in strat_grid]
    n = len(present)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.82])
    ax.axis("off")

    int_grid = _strat_grid_to_int(strat_grid)
    ax.imshow(int_grid, cmap=_CMAP_STRAT, norm=_NORM_STRAT, interpolation="nearest")
    ax.set_title(f"Strategy Map — Generation {generation}", fontweight="bold", pad=12)

    slot_w = 1.0 / n
    for i, s in enumerate(present):
        x = i * slot_w
        fig.add_axes([x + slot_w * 0.05, 0.01, slot_w * 0.18, 0.055]).set(
            facecolor=STRATEGY_COLORS[s], xticks=[], yticks=[]
        )
        fig.text(
            x + slot_w * 0.27, 0.037,
            s, va="center", ha="left",
            fontsize=7.5, color="0.15",
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_payoff_map(
    payoff_grid: np.ndarray,
    title: str,
    generation: int,
    save_path: Optional[Path] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(payoff_grid, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.85, label="Payoff")
    ax.set_title(f"{title} — Generation {generation}", fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_final_static(history: dict, output_dir: Path) -> None:
    """Saves static plots of the final generation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = len(history["strategy_grids"])

    plot_strategy_map(
        history["strategy_grids"][-1], gen,
        save_path=output_dir / "01_strategy_map_final.png",
    )
    plot_payoff_map(
        history["avg_score_grids"][-1],
        "Avg Payoff Per Round", gen,
        save_path=output_dir / "02_avg_payoff_final.png",
    )
    plot_payoff_map(
        history["score_grids"][-1],
        "Total Payoff", gen,
        save_path=output_dir / "03_total_payoff_final.png",
    )


# ── Animations ────────────────────────────────────────────────────────────────

def _save_gif(anim: FuncAnimation, path: Path, fps: int) -> None:
    writer = PillowWriter(fps=fps)
    anim.save(str(path), writer=writer)
    print(f"  Saved: {path}")


def animate_strategy_map(
    history: dict,
    save_path: Optional[Path] = None,
    fps: int = 4,
) -> FuncAnimation:
    grids = history["strategy_grids"]
    G = len(grids)

    # All strategies ever present across all generations
    all_present = [s for s in STRATEGY_NAMES if any(s in np.unique(g) for g in grids)]

    # Two-row layout: grid on top, legend strip on bottom
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.05, 0.12, 0.90, 0.82])   # grid axes
    ax.axis("off")

    int_grid_0 = _strat_grid_to_int(grids[0])
    im = ax.imshow(int_grid_0, cmap=_CMAP_STRAT, norm=_NORM_STRAT, interpolation="nearest")
    title = ax.set_title("Strategy Map — Generation 1", fontweight="bold", pad=12)

    # Horizontal legend below grid — one coloured rectangle + label per strategy
    n = len(all_present)
    slot_w = 1.0 / n
    for i, s in enumerate(all_present):
        x = i * slot_w
        # Colour swatch
        fig.add_axes([x + slot_w * 0.05, 0.01, slot_w * 0.18, 0.055]).set(
            facecolor=STRATEGY_COLORS[s], xticks=[], yticks=[]
        )
        # Label
        fig.text(
            x + slot_w * 0.27, 0.037,
            s, va="center", ha="left",
            fontsize=7.5, color="0.15",
        )

    def update(frame):
        im.set_data(_strat_grid_to_int(grids[frame]))
        title.set_text(f"Strategy Map — Generation {frame + 1}/{G}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=G, interval=1000 // fps, blit=True)

    if save_path:
        _save_gif(anim, save_path, fps)

    return anim


def animate_payoff_map(
    history: dict,
    key: str,
    label: str,
    save_path: Optional[Path] = None,
    fps: int = 4,
) -> FuncAnimation:
    grids = history[key]
    G = len(grids)

    # Fix colour scale across all generations for fair comparison
    vmin = min(g.min() for g in grids)
    vmax = max(g.max() for g in grids)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axis("off")

    im = ax.imshow(grids[0], cmap="RdYlGn", interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, label="Payoff")
    title = ax.set_title(f"{label} — Generation 1/{G}", fontweight="bold", pad=12)

    plt.tight_layout()

    def update(frame):
        im.set_data(grids[frame])
        title.set_text(f"{label} — Generation {frame + 1}/{G}")
        return [im, title]

    anim = FuncAnimation(fig, update, frames=G, interval=1000 // fps, blit=True)

    if save_path:
        _save_gif(anim, save_path, fps)

    return anim


def animate_all(
    history: dict,
    output_dir: Path,
    fps: int = 4,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Animating strategy map...")
    animate_strategy_map(history, save_path=output_dir / "anim_strategy_map.gif", fps=fps)

    print("Animating avg payoff grid...")
    animate_payoff_map(
        history, key="avg_score_grids", label="Avg Payoff Per Round",
        save_path=output_dir / "anim_avg_payoff.gif", fps=fps,
    )

    print("Animating total payoff grid...")
    animate_payoff_map(
        history, key="score_grids", label="Total Payoff",
        save_path=output_dir / "anim_total_payoff.gif", fps=fps,
    )
