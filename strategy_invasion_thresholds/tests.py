"""
tests.py — Test suite for the invasion threshold analysis.

Run with:
    pytest prisoners_dilemma/strategy_invasion_thresholds/tests.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_invasion_thresholds.invasion import (
    make_invasion_grid,
    run_invasion_trial,
    aggregate_results,
    compute_threshold_matrix,
    threshold_matrix_wide,
    DEFAULT_FRACTIONS,
)
from spatial.grid import STRATEGY_NAMES
from tournament.simulation import AlwaysCooperate, AlwaysDefect, TitForTat

# ── make_invasion_grid ────────────────────────────────────────────────────────

class TestMakeInvasionGrid:
    def test_shape(self):
        g = make_invasion_grid(8, "Always Defect", "Tit-for-Tat", 0.25, seed=0)
        assert g.shape == (8, 8)

    def test_only_two_strategies(self):
        g = make_invasion_grid(8, "Always Defect", "Tit-for-Tat", 0.25, seed=0)
        names = {g[r, c].strategy_name for r in range(8) for c in range(8)}
        assert names == {"Always Defect", "Tit-for-Tat"}

    def test_invader_fraction_approx(self):
        side = 20
        for f in [0.10, 0.25, 0.50, 0.75]:
            g = make_invasion_grid(side, "Always Defect", "Tit-for-Tat", f, seed=42)
            actual = sum(1 for r in range(side) for c in range(side)
                         if g[r, c].strategy_name == "Tit-for-Tat") / (side * side)
            assert abs(actual - f) < 0.05, f"fraction={f}: expected ~{f}, got {actual:.3f}"

    def test_at_least_one_invader(self):
        # Even tiny fractions should place at least 1 invader
        g = make_invasion_grid(4, "Always Defect", "Tit-for-Tat", 0.01, seed=0)
        count = sum(1 for r in range(4) for c in range(4)
                    if g[r, c].strategy_name == "Tit-for-Tat")
        assert count >= 1

    def test_reproducible_with_seed(self):
        g1 = make_invasion_grid(8, "Always Defect", "Tit-for-Tat", 0.3, seed=7)
        g2 = make_invasion_grid(8, "Always Defect", "Tit-for-Tat", 0.3, seed=7)
        for r in range(8):
            for c in range(8):
                assert g1[r, c].strategy_name == g2[r, c].strategy_name

    def test_different_seeds_can_differ(self):
        g1 = make_invasion_grid(8, "Always Defect", "Tit-for-Tat", 0.3, seed=1)
        g2 = make_invasion_grid(8, "Always Defect", "Tit-for-Tat", 0.3, seed=99)
        names1 = [g1[r, c].strategy_name for r in range(8) for c in range(8)]
        names2 = [g2[r, c].strategy_name for r in range(8) for c in range(8)]
        assert names1 != names2

    def test_unique_player_ids(self):
        g = make_invasion_grid(6, "Always Defect", "Tit-for-Tat", 0.3, seed=0)
        ids = [g[r, c].player_id for r in range(6) for c in range(6)]
        assert len(ids) == len(set(ids))

# ── run_invasion_trial ────────────────────────────────────────────────────────

class TestRunInvasionTrial:
    def _trial(self, resident, invader, fraction, side=6, gens=5, seed=42):
        return run_invasion_trial(resident, invader, fraction, side, "moore", 5, gens, seed)

    def test_result_keys(self):
        r = self._trial("Always Defect", "Tit-for-Tat", 0.3)
        for key in ["resident", "invader", "initial_fraction", "final_fraction",
                    "generations_run", "fixated", "invader_won", "seed", "fraction_history"]:
            assert key in r

    def test_fraction_history_starts_at_initial(self):
        r = self._trial("Always Defect", "Tit-for-Tat", 0.5)
        assert abs(r["fraction_history"][0] - 0.5) < 0.1  # approximate due to rounding

    def test_final_fraction_matches_history(self):
        r = self._trial("Always Defect", "Tit-for-Tat", 0.3)
        assert r["final_fraction"] == r["fraction_history"][-1]

    def test_final_fraction_bounded(self):
        r = self._trial("Always Defect", "Tit-for-Tat", 0.3)
        assert 0.0 <= r["final_fraction"] <= 1.0

    def test_invader_won_consistent_with_final_fraction(self):
        r = self._trial("Always Defect", "Tit-for-Tat", 0.3)
        assert r["invader_won"] == (r["final_fraction"] > 0.5)

    def test_generations_run_leq_max(self):
        r = self._trial("Always Defect", "Tit-for-Tat", 0.3, gens=5)
        assert r["generations_run"] <= 5

    def test_fixation_stops_early(self):
        # AD vs AC: AD should fixate quickly
        r = run_invasion_trial("Always Cooperate", "Always Defect", 0.5,
                               side=6, neighbourhood="moore", num_rounds=5,
                               max_generations=20, seed=42)
        assert r["fixated"] or r["generations_run"] <= 20

    def test_all_defect_vs_cooperate_invader_wins_at_high_fraction(self):
        # AD starting at 95% against AC should always win
        r = run_invasion_trial("Always Cooperate", "Always Defect", 0.95,
                               side=8, neighbourhood="moore", num_rounds=5,
                               max_generations=15, seed=42)
        assert r["invader_won"]

    def test_strategy_names_recorded(self):
        r = self._trial("Always Defect", "Tit-for-Tat", 0.3)
        assert r["resident"] == "Always Defect"
        assert r["invader"] == "Tit-for-Tat"

# ── aggregate_results ─────────────────────────────────────────────────────────

class TestAggregateResults:
    def _make_raw(self):
        rows = []
        for f in [0.1, 0.3, 0.5]:
            for s in range(3):
                rows.append({
                    "resident": "Always Defect", "invader": "Tit-for-Tat",
                    "initial_fraction": f, "final_fraction": 1.0 if f > 0.3 else 0.0,
                    "invader_won": f > 0.3, "generations_run": 10, "seed": s,
                })
        return pd.DataFrame(rows)

    def test_output_columns(self):
        raw = self._make_raw()
        agg = aggregate_results(raw)
        for col in ["resident", "invader", "initial_fraction", "mean_final_fraction",
                    "invasion_rate", "mean_generations", "n_seeds"]:
            assert col in agg.columns

    def test_n_rows(self):
        raw = self._make_raw()
        agg = aggregate_results(raw)
        assert len(agg) == 3  # 3 fraction points

    def test_invasion_rate_values(self):
        raw = self._make_raw()
        agg = aggregate_results(raw).sort_values("initial_fraction")
        rates = agg["invasion_rate"].tolist()
        assert rates[0] == 0.0   # f=0.1 → invader never won
        assert rates[2] == 1.0   # f=0.5 → invader always won

# ── compute_threshold_matrix ──────────────────────────────────────────────────

class TestComputeThresholdMatrix:
    def _make_agg(self, rates_by_fraction: dict, res="A", inv="B"):
        rows = []
        for f, rate in rates_by_fraction.items():
            rows.append({"resident": res, "invader": inv,
                         "initial_fraction": f, "invasion_rate": rate,
                         "mean_final_fraction": rate, "std_final_fraction": 0.1,
                         "mean_generations": 10, "n_seeds": 5})
        return pd.DataFrame(rows)

    def test_never_invades_is_nan(self):
        agg = self._make_agg({0.1: 0.0, 0.5: 0.2, 0.9: 0.4})
        t = compute_threshold_matrix(agg)
        assert np.isnan(t.loc[0, "threshold"])

    def test_always_invades_is_min_fraction(self):
        agg = self._make_agg({0.01: 0.8, 0.1: 1.0, 0.5: 1.0})
        t = compute_threshold_matrix(agg)
        assert t.loc[0, "threshold"] == pytest.approx(0.01)

    def test_threshold_between_points(self):
        # rate goes 0.0 at f=0.2, 1.0 at f=0.4 → threshold ≈ 0.3
        agg = self._make_agg({0.2: 0.0, 0.4: 1.0})
        t = compute_threshold_matrix(agg)
        assert 0.2 <= t.loc[0, "threshold"] <= 0.4

    def test_threshold_matrix_wide_shape(self):
        rows = []
        for res in STRATEGY_NAMES:
            for inv in STRATEGY_NAMES:
                if res != inv:
                    rows.append({"resident": res, "invader": inv, "threshold": 0.3})
        df = pd.DataFrame(rows)
        wide = threshold_matrix_wide(df)
        assert wide.shape == (len(STRATEGY_NAMES), len(STRATEGY_NAMES))

    def test_threshold_matrix_wide_values(self):
        rows = [{"resident": "Always Defect", "invader": "Tit-for-Tat", "threshold": 0.42}]
        df = pd.DataFrame(rows)
        wide = threshold_matrix_wide(df)
        assert wide.loc["Always Defect", "Tit-for-Tat"] == pytest.approx(0.42)

# ── DEFAULT_FRACTIONS ─────────────────────────────────────────────────────────

class TestDefaultFractions:
    def test_starts_low(self):
        assert DEFAULT_FRACTIONS[0] <= 0.05

    def test_ends_high(self):
        assert DEFAULT_FRACTIONS[-1] >= 0.90

    def test_all_between_0_and_1(self):
        assert all(0 < f < 1 for f in DEFAULT_FRACTIONS)

    def test_strictly_increasing(self):
        assert all(a < b for a, b in zip(DEFAULT_FRACTIONS, DEFAULT_FRACTIONS[1:]))
