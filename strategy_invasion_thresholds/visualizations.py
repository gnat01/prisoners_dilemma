"""
visualizations.py — Plots for the invasion threshold analysis.

Plots produced:
  01_threshold_matrix.png       — 7×7 heatmap of invasion thresholds f*
  02_invasion_rate_curves.png   — invasion rate vs starting fraction, per resident
  03_final_fraction_curves.png  — mean final invader fraction vs starting fraction, key pairs
  04_seed_variance.png          — individual seed outcomes for the most contested pair
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial.grid import STRATEGY_NAMES
from tournament.visualizations import STRATEGY_COLORS

SHORT = {
    "Always Cooperate": "AC",
    "Always Defect":    "AD",
    "Random(p=0.2)":   "R0.2",
    "Random(p=0.4)":   "R0.4",
    "Random(p=0.6)":   "R0.6",
    "Random(p=0.8)":   "R0.8",
    "Tit-for-Tat":     "TfT",
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight"})


# ── Plot 1: Threshold matrix heatmap ─────────────────────────────────────────

def plot_threshold_matrix(
    wide: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    7×7 heatmap. Rows = resident, cols = invader.
    Cell = invasion threshold f* (fraction invader needs to take over).
    Gray = invader never wins majority of seeds.
    """
    mat = wide.reindex(index=STRATEGY_NAMES, columns=STRATEGY_NAMES)
    short_labels = [SHORT[s] for s in STRATEGY_NAMES]

    # Build annotation strings
    annot = mat.copy().astype(object)
    for r in mat.index:
        for c in mat.columns:
            v = mat.loc[r, c]
            if r == c:
                annot.loc[r, c] = "—"
            elif np.isnan(v):
                annot.loc[r, c] = "X"
            else:
                annot.loc[r, c] = f"{v:.2f}"

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a masked array so NaN cells render as gray
    data = mat.values.astype(float)
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#cccccc")

    im = ax.imshow(
        np.ma.masked_invalid(data),
        cmap=cmap, vmin=0, vmax=1, aspect="auto",
    )
    plt.colorbar(im, ax=ax, shrink=0.82,
                 label="Invasion threshold f*  (lower = easier to invade)")

    # Annotate cells
    for i in range(len(STRATEGY_NAMES)):
        for j in range(len(STRATEGY_NAMES)):
            txt = annot.iloc[i, j]
            color = "white" if txt not in ("—", "X") and float(data[i, j]) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(STRATEGY_NAMES)))
    ax.set_yticks(range(len(STRATEGY_NAMES)))
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Invader strategy", fontsize=12)
    ax.set_ylabel("Resident strategy", fontsize=12)
    ax.set_title(
        "Spatial Invasion Threshold Matrix\n"
        "f* = minimum invader fraction needed to win majority of trials\n"
        "X = never invades  ·  — = self  ·  green = easy invasion  ·  red = hard",
        fontweight="bold", pad=14,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Plot 2: Invasion rate curves (per resident) ───────────────────────────────

def plot_invasion_rate_curves(
    agg: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    7 subplots — one per resident strategy.
    Each subplot: x = starting fraction, y = invasion rate, one line per invader.
    The 0.5 horizontal line marks the threshold.
    """
    residents = STRATEGY_NAMES
    n = len(residents)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharey=True)
    axes = axes.flatten()

    for ax, resident in zip(axes, residents):
        sub = agg[agg["resident"] == resident]
        for invader in STRATEGY_NAMES:
            if invader == resident:
                continue
            pair = sub[sub["invader"] == invader].sort_values("initial_fraction")
            if pair.empty:
                continue
            color = STRATEGY_COLORS.get(invader, "#999")
            ax.plot(pair["initial_fraction"], pair["invasion_rate"],
                    marker="o", markersize=3, linewidth=1.8, color=color,
                    label=SHORT[invader])
            ax.fill_between(
                pair["initial_fraction"],
                np.clip(pair["invasion_rate"] - pair["invasion_rate"].std(), 0, 1),
                np.clip(pair["invasion_rate"] + pair["invasion_rate"].std(), 0, 1),
                alpha=0.08, color=color,
            )

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(f"Resident: {SHORT[resident]}", fontweight="bold", fontsize=10)
        ax.set_xlabel("Invader starting fraction")
        ax.set_ylabel("Invasion rate")
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=7, ncol=2, loc="upper left")

    # Hide unused axes
    for ax in axes[len(residents):]:
        ax.set_visible(False)

    fig.suptitle(
        "Invasion Rate vs Starting Fraction by Resident Strategy\n"
        "Dashed line = 50% threshold  ·  Curve crossing dashed = invasion threshold f*",
        fontweight="bold", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── Plot 3: Final fraction curves for key pairs ───────────────────────────────

def plot_final_fraction_curves(
    agg: pd.DataFrame,
    raw: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    Mean final invader fraction vs starting fraction for the most interesting pairs.
    ±1 std error band. Diagonal (y=x) reference: above = invader grows, below = shrinks.
    """
    # Pick the most interesting pairs: those with a threshold between 0.05 and 0.95
    from invasion import compute_threshold_matrix, threshold_matrix_wide
    thresh_df = compute_threshold_matrix(agg)
    interesting = thresh_df[
        thresh_df["threshold"].between(0.05, 0.90, inclusive="both")
    ].sort_values("threshold")

    # Cap at 8 pairs for readability
    pairs = list(zip(interesting["resident"], interesting["invader"]))[:8]

    if not pairs:
        # Fallback: just pick a few hard-coded interesting ones
        pairs = [
            ("Always Defect", "Tit-for-Tat"),
            ("Tit-for-Tat", "Always Defect"),
            ("Always Cooperate", "Tit-for-Tat"),
            ("Always Defect", "Always Cooperate"),
        ]
        pairs = [(r, i) for r, i in pairs if not agg[(agg["resident"]==r)&(agg["invader"]==i)].empty]

    ncols = min(4, len(pairs))
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (res, inv) in zip(axes, pairs):
        sub = agg[(agg["resident"] == res) & (agg["invader"] == inv)].sort_values("initial_fraction")
        color = STRATEGY_COLORS.get(inv, "#555")

        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="y = x")
        ax.plot(sub["initial_fraction"], sub["mean_final_fraction"],
                color=color, linewidth=2.2, marker="o", markersize=4,
                label=f"{SHORT[inv]} invades")
        ax.fill_between(
            sub["initial_fraction"],
            (sub["mean_final_fraction"] - sub["std_final_fraction"]).clip(0, 1),
            (sub["mean_final_fraction"] + sub["std_final_fraction"]).clip(0, 1),
            alpha=0.15, color=color,
        )
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_title(f"{SHORT[res]} ← {SHORT[inv]}", fontweight="bold")
        ax.set_xlabel("Starting fraction")
        ax.set_ylabel("Final fraction")
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.legend(fontsize=8)

    for ax in axes[len(pairs):]:
        ax.set_visible(False)

    fig.suptitle(
        "Final Invader Fraction vs Starting Fraction — Key Pairs\n"
        "Above y=x diagonal: invader grows  ·  Below: invader shrinks  ·  Band = ±1 std",
        fontweight="bold", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ── Plot 4: Seed variance for the most contested pair ─────────────────────────

def plot_seed_variance(
    raw: pd.DataFrame,
    agg: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> None:
    """
    For the pair with the most contested threshold (highest variance in outcomes
    around the threshold fraction), scatter all individual seed outcomes.
    Shows how much spatial randomness matters.
    """
    from invasion import compute_threshold_matrix
    thresh_df = compute_threshold_matrix(agg)

    # Pick pair where threshold is closest to 0.30 — most dynamically interesting
    thresh_valid = thresh_df.dropna()
    if thresh_valid.empty:
        return

    best = thresh_valid.iloc[(thresh_valid["threshold"] - 0.30).abs().argsort().iloc[0]]
    res, inv = best["resident"], best["invader"]

    sub_raw = raw[(raw["resident"] == res) & (raw["invader"] == inv)].copy()
    sub_agg = agg[(agg["resident"] == res) & (agg["invader"] == inv)].sort_values("initial_fraction")

    color_inv = STRATEGY_COLORS.get(inv, "#555")
    color_res = STRATEGY_COLORS.get(res, "#999")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Individual seed dots
    jitter = np.random.default_rng(0).uniform(-0.005, 0.005, len(sub_raw))
    ax.scatter(
        sub_raw["initial_fraction"] + jitter,
        sub_raw["final_fraction"],
        c=sub_raw["invader_won"].map({True: color_inv, False: color_res}),
        alpha=0.5, s=30, zorder=3, label="_nolegend_",
    )

    # Mean line
    ax.plot(sub_agg["initial_fraction"], sub_agg["mean_final_fraction"],
            color="black", linewidth=2.2, marker="o", markersize=5, label="Mean final fraction", zorder=5)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="y = x")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    # Threshold marker
    t = best["threshold"]
    if not np.isnan(t):
        ax.axvline(t, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Threshold f* ≈ {t:.2f}")

    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color=color_inv, label=f"Invader wins ({SHORT[inv]})"),
        mpatches.Patch(color=color_res, label=f"Resident wins ({SHORT[res]})"),
        plt.Line2D([0], [0], color="black", linewidth=2, label="Mean final fraction"),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label=f"Threshold f* ≈ {t:.2f}"),
        plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="y = x"),
    ], fontsize=9, loc="upper left")

    ax.set_xlabel("Invader starting fraction", fontsize=12)
    ax.set_ylabel("Final invader fraction", fontsize=12)
    ax.set_title(
        f"Seed Variance: {SHORT[inv]} invading {SHORT[res]}\n"
        "Each dot = one seed  ·  Color = who won  ·  Spread shows spatial randomness",
        fontweight="bold", pad=14,
    )
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ── Orchestrator ──────────────────────────────────────────────────────────────

def plot_all(
    raw: pd.DataFrame,
    agg: pd.DataFrame,
    wide: pd.DataFrame,
    output_dir: Path,
) -> None:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Threshold matrix...")
    plot_threshold_matrix(wide,          save_path=output_dir / "01_threshold_matrix.png")
    print("  Invasion rate curves...")
    plot_invasion_rate_curves(agg,       save_path=output_dir / "02_invasion_rate_curves.png")
    print("  Final fraction curves...")
    plot_final_fraction_curves(agg, raw, save_path=output_dir / "03_final_fraction_curves.png")
    print("  Seed variance...")
    plot_seed_variance(raw, agg,         save_path=output_dir / "04_seed_variance.png")
