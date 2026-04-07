"""
analysis.py — Transform raw match records into analysis DataFrames.

DataFrame hierarchy:
  match_df         — one row per match
  player_match_df  — one row per (player, match) — two rows per match
  player_overall   — one row per player: cumulative stats
  strategy_round   — one row per (strategy, round): avg payoff across all players & matches
  strategy_overall — one row per strategy: summary stats
  vs_matrix        — (strategy × opponent_strategy) avg payoff per round
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

NUM_ROUNDS = 5

# ── Raw → match DataFrame ─────────────────────────────────────────────────────


def matches_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per match."""
    return pd.DataFrame(records)


# ── Match → player-match DataFrame ───────────────────────────────────────────


def player_match_df(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots each match into two player-perspective rows.
    Columns: player_id, strategy, opponent_id, opponent_strategy,
             r1_action, r1_payoff, ..., r5_action, r5_payoff,
             match_total_payoff
    """
    shared_cols = ["player_id", "strategy", "opponent_id", "opponent_strategy", "match_total_payoff"]

    # ── Player A perspective ──────────────────────────────────────────────────
    df_a = match_df[["player_a_id", "strategy_a", "player_b_id", "strategy_b", "total_payoff_a"]].copy()
    df_a.columns = shared_cols

    for r in range(1, NUM_ROUNDS + 1):
        df_a[f"r{r}_action"] = match_df[f"r{r}_action_a"].values
        df_a[f"r{r}_payoff"] = match_df[f"r{r}_payoff_a"].values

    # ── Player B perspective ──────────────────────────────────────────────────
    df_b = match_df[["player_b_id", "strategy_b", "player_a_id", "strategy_a", "total_payoff_b"]].copy()
    df_b.columns = shared_cols

    for r in range(1, NUM_ROUNDS + 1):
        df_b[f"r{r}_action"] = match_df[f"r{r}_action_b"].values
        df_b[f"r{r}_payoff"] = match_df[f"r{r}_payoff_b"].values

    return pd.concat([df_a, df_b], ignore_index=True)


# ── Player-match → player overall ────────────────────────────────────────────


def player_overall_df(pm_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per player.
    Columns: player_id, strategy, n_matches, total_payoff,
             avg_payoff_per_match, avg_payoff_per_round
    """
    agg = (
        pm_df.groupby(["player_id", "strategy"])
        .agg(
            n_matches=("match_total_payoff", "count"),
            total_payoff=("match_total_payoff", "sum"),
            avg_payoff_per_match=("match_total_payoff", "mean"),
        )
        .reset_index()
    )
    agg["avg_payoff_per_round"] = agg["avg_payoff_per_match"] / NUM_ROUNDS
    return agg


# ── Round-level strategy stats ────────────────────────────────────────────────


def strategy_round_df(pm_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (strategy, round).
    Columns: strategy, round, avg_payoff, std_payoff, n
    """
    rows = []
    for strat, grp in pm_df.groupby("strategy"):
        for r in range(1, NUM_ROUNDS + 1):
            payoffs = grp[f"r{r}_payoff"]
            rows.append(
                {
                    "strategy": strat,
                    "round": r,
                    "avg_payoff": payoffs.mean(),
                    "std_payoff": payoffs.std(),
                    "n": len(payoffs),
                }
            )
    return pd.DataFrame(rows)


# ── Strategy overall stats ────────────────────────────────────────────────────


def strategy_overall_df(player_overall: pd.DataFrame) -> pd.DataFrame:
    """
    One row per strategy.
    Columns: strategy, n_players, avg_total_payoff, std_total_payoff,
             avg_per_match, avg_per_round
    """
    return (
        player_overall.groupby("strategy")
        .agg(
            n_players=("player_id", "count"),
            avg_total_payoff=("total_payoff", "mean"),
            std_total_payoff=("total_payoff", "std"),
            avg_per_match=("avg_payoff_per_match", "mean"),
            avg_per_round=("avg_payoff_per_round", "mean"),
        )
        .reset_index()
        .sort_values("avg_total_payoff", ascending=False)
        .reset_index(drop=True)
    )


# ── Strategy vs strategy matrix ───────────────────────────────────────────────


def strategy_vs_strategy_df(pm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: rows = focal strategy, cols = opponent strategy.
    Values = average payoff per round for the focal strategy.
    """
    payoff_cols = [f"r{r}_payoff" for r in range(1, NUM_ROUNDS + 1)]
    pm = pm_df.copy()
    pm["avg_round_payoff"] = pm[payoff_cols].mean(axis=1)

    return (
        pm.groupby(["strategy", "opponent_strategy"])["avg_round_payoff"]
        .mean()
        .unstack(fill_value=np.nan)
    )


# ── Cooperation rate per round ────────────────────────────────────────────────


def cooperation_rate_df(pm_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (strategy, round).
    Columns: strategy, round, cooperation_rate
    """
    rows = []
    for strat, grp in pm_df.groupby("strategy"):
        for r in range(1, NUM_ROUNDS + 1):
            rate = (grp[f"r{r}_action"] == "C").mean()
            rows.append({"strategy": strat, "round": r, "cooperation_rate": rate})
    return pd.DataFrame(rows)
