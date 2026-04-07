"""
tests.py — Test suite for the spatial Prisoner's Dilemma.

Run with:
    pytest prisoners_dilemma/spatial/tests.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial.grid import (
    von_neumann_neighbours,
    moore_neighbours,
    make_grid,
    run_generation,
    run_simulation,
    strategy_by_name,
    STRATEGY_NAMES,
)
from tournament.simulation import AlwaysCooperate, AlwaysDefect, TitForTat, Player

# ── Neighbourhood tests ───────────────────────────────────────────────────────

class TestVonNeumann:
    def test_interior_cell_has_4_neighbours(self):
        assert len(von_neumann_neighbours(2, 2, 5, 5)) == 4

    def test_no_self(self):
        for pos in von_neumann_neighbours(2, 2, 5, 5):
            assert pos != (2, 2)

    def test_wraps_top_edge(self):
        neighbours = von_neumann_neighbours(0, 0, 5, 5)
        assert (4, 0) in neighbours  # wraps to bottom

    def test_wraps_left_edge(self):
        neighbours = von_neumann_neighbours(0, 0, 5, 5)
        assert (0, 4) in neighbours  # wraps to right

    def test_wraps_bottom_edge(self):
        neighbours = von_neumann_neighbours(4, 4, 5, 5)
        assert (0, 4) in neighbours

    def test_wraps_right_edge(self):
        neighbours = von_neumann_neighbours(4, 4, 5, 5)
        assert (4, 0) in neighbours

    def test_neighbours_are_in_bounds(self):
        for r, c in von_neumann_neighbours(0, 0, 5, 5):
            assert 0 <= r < 5 and 0 <= c < 5

    def test_1x1_grid_wraps_to_self(self):
        # A 1×1 grid: all four neighbours resolve to (0,0)
        neighbours = von_neumann_neighbours(0, 0, 1, 1)
        assert all(pos == (0, 0) for pos in neighbours)


class TestMoore:
    def test_interior_cell_has_8_neighbours(self):
        assert len(moore_neighbours(2, 2, 5, 5)) == 8

    def test_no_self(self):
        for pos in moore_neighbours(2, 2, 5, 5):
            assert pos != (2, 2)

    def test_wraps_corner(self):
        neighbours = moore_neighbours(0, 0, 5, 5)
        assert (4, 4) in neighbours  # diagonal wrap

    def test_neighbours_are_in_bounds(self):
        for r, c in moore_neighbours(0, 0, 5, 5):
            assert 0 <= r < 5 and 0 <= c < 5

    def test_moore_is_superset_of_von_neumann(self):
        m = set(moore_neighbours(2, 2, 5, 5))
        vn = set(von_neumann_neighbours(2, 2, 5, 5))
        assert vn.issubset(m)

# ── Grid initialisation ───────────────────────────────────────────────────────

class TestMakeGrid:
    def test_shape(self):
        grid = make_grid(10)
        assert grid.shape == (10, 10)

    def test_all_cells_filled(self):
        grid = make_grid(5)
        for r in range(5):
            for c in range(5):
                assert grid[r, c] is not None

    def test_all_players_have_valid_strategy(self):
        grid = make_grid(5)
        for r in range(5):
            for c in range(5):
                assert grid[r, c].strategy_name in STRATEGY_NAMES

    def test_unique_player_ids(self):
        grid = make_grid(5)
        ids = [grid[r, c].player_id for r in range(5) for c in range(5)]
        assert len(ids) == len(set(ids))

    def test_reproducible_with_seed(self):
        g1 = make_grid(5, seed=0)
        g2 = make_grid(5, seed=0)
        for r in range(5):
            for c in range(5):
                assert g1[r, c].strategy_name == g2[r, c].strategy_name

    def test_different_seeds_differ(self):
        g1 = make_grid(8, seed=0)
        g2 = make_grid(8, seed=99)
        names1 = [g1[r, c].strategy_name for r in range(8) for c in range(8)]
        names2 = [g2[r, c].strategy_name for r in range(8) for c in range(8)]
        assert names1 != names2

# ── Single generation ─────────────────────────────────────────────────────────

class TestRunGeneration:
    def setup_method(self):
        self.side = 4
        self.grid = make_grid(self.side, seed=42)

    def test_scores_shape(self):
        scores, avg_scores, new_grid = run_generation(self.grid, "moore")
        assert scores.shape == (self.side, self.side)
        assert avg_scores.shape == (self.side, self.side)

    def test_new_grid_shape(self):
        _, _, new_grid = run_generation(self.grid, "moore")
        assert new_grid.shape == (self.side, self.side)

    def test_scores_non_negative(self):
        scores, _, _ = run_generation(self.grid, "moore")
        assert (scores >= 0).all()

    def test_avg_scores_leq_max_payoff(self):
        _, avg_scores, _ = run_generation(self.grid, "moore")
        assert (avg_scores <= 5.0).all()

    def test_new_grid_strategies_are_valid(self):
        _, _, new_grid = run_generation(self.grid, "moore")
        for r in range(self.side):
            for c in range(self.side):
                assert new_grid[r, c].strategy_name in STRATEGY_NAMES

    def test_von_neumann_vs_moore_different_scores(self):
        scores_m, _, _ = run_generation(self.grid, "moore")
        scores_vn, _, _ = run_generation(make_grid(self.side, seed=42), "von_neumann")
        # Moore has more neighbours so total scores should be higher
        assert scores_m.sum() > scores_vn.sum()

    def test_all_cooperate_grid_uniform_score(self):
        """If everyone cooperates, all cells should score identically (Moore)."""
        side = 4
        grid = np.empty((side, side), dtype=object)
        for i in range(side):
            for j in range(side):
                grid[i, j] = Player(i * side + j, AlwaysCooperate())
        scores, avg_scores, _ = run_generation(grid, "moore", num_rounds=5)
        # 8 neighbours × 5 rounds × 3 payoff = 120 per cell
        assert np.allclose(scores, 120.0)
        assert np.allclose(avg_scores, 3.0)

    def test_all_defect_grid_uniform_score(self):
        """If everyone defects, all cells should score identically (Moore)."""
        side = 4
        grid = np.empty((side, side), dtype=object)
        for i in range(side):
            for j in range(side):
                grid[i, j] = Player(i * side + j, AlwaysDefect())
        scores, avg_scores, _ = run_generation(grid, "moore", num_rounds=5)
        # 8 neighbours × 5 rounds × 1 payoff = 40 per cell
        assert np.allclose(scores, 40.0)
        assert np.allclose(avg_scores, 1.0)

    def test_imitate_best_adopts_highest_scorer(self):
        """
        2×2 grid: three AC players and one AD.
        AD should score highest and its neighbours should adopt AD.
        """
        grid = np.empty((2, 2), dtype=object)
        grid[0, 0] = Player(0, AlwaysDefect())
        grid[0, 1] = Player(1, AlwaysCooperate())
        grid[1, 0] = Player(2, AlwaysCooperate())
        grid[1, 1] = Player(3, AlwaysCooperate())
        _, _, new_grid = run_generation(grid, "moore", num_rounds=5)
        # AD is neighbour of every cell on a 2×2 toroidal grid — all should copy it
        for r in range(2):
            for c in range(2):
                assert new_grid[r, c].strategy_name == "Always Defect"

# ── Full simulation ───────────────────────────────────────────────────────────

class TestRunSimulation:
    def test_history_keys(self):
        h = run_simulation(side=4, generations=3, verbose=False)
        assert "strategy_grids" in h
        assert "score_grids" in h
        assert "avg_score_grids" in h

    def test_history_length(self):
        G = 5
        h = run_simulation(side=4, generations=G, verbose=False)
        assert len(h["strategy_grids"]) == G
        assert len(h["score_grids"]) == G
        assert len(h["avg_score_grids"]) == G

    def test_strategy_grids_dtype(self):
        h = run_simulation(side=4, generations=2, verbose=False)
        for sg in h["strategy_grids"]:
            assert sg.dtype.kind in ("U", "O")  # string or object

    def test_strategy_names_valid_throughout(self):
        h = run_simulation(side=4, generations=3, verbose=False)
        for sg in h["strategy_grids"]:
            for name in sg.flatten():
                assert name in STRATEGY_NAMES

    def test_scores_non_negative_throughout(self):
        h = run_simulation(side=4, generations=3, verbose=False)
        for sg in h["score_grids"]:
            assert (sg >= 0).all()

    def test_reproducible(self):
        h1 = run_simulation(side=4, generations=3, seed=7, verbose=False)
        h2 = run_simulation(side=4, generations=3, seed=7, verbose=False)
        for g1, g2 in zip(h1["strategy_grids"], h2["strategy_grids"]):
            assert (g1 == g2).all()

    def test_von_neumann_neighbourhood(self):
        h = run_simulation(side=4, generations=2, neighbourhood="von_neumann", verbose=False)
        assert len(h["strategy_grids"]) == 2

# ── Strategy catalogue ────────────────────────────────────────────────────────

class TestStrategyCatalogue:
    @pytest.mark.parametrize("name", STRATEGY_NAMES)
    def test_strategy_by_name_roundtrip(self, name):
        s = strategy_by_name(name)
        assert s.name == name

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            strategy_by_name("FakeStrategy")
