"""
tests.py — Test suite for the Prisoner's Dilemma simulation.

Run with:
    pytest prisoners_dilemma/tests.py -v
    pytest prisoners_dilemma/tests.py -v --tb=short   (for briefer tracebacks)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running from the game_theory root directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from prisoners_dilemma.simulation import (
    C, D, PAYOFFS,
    AlwaysCooperate, AlwaysDefect, RandomCooperate, TitForTat,
    Player, play_match, create_population, run_tournament,
)
from prisoners_dilemma.analysis import (
    matches_to_df,
    player_match_df,
    player_overall_df,
    strategy_round_df,
    strategy_overall_df,
    strategy_vs_strategy_df,
    cooperation_rate_df,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_player(pid: int, strategy) -> Player:
    return Player(pid, strategy)

def match_ac_ad():
    """AlwaysCooperate vs AlwaysDefect — deterministic reference match."""
    return play_match(make_player(0, AlwaysCooperate()), make_player(1, AlwaysDefect()))

# ── Payoff matrix ─────────────────────────────────────────────────────────────

class TestPayoffMatrix:
    def test_mutual_cooperation(self):
        assert PAYOFFS[(C, C)] == (3, 3)

    def test_mutual_defection(self):
        assert PAYOFFS[(D, D)] == (1, 1)

    def test_cooperate_defect(self):
        pa, pb = PAYOFFS[(C, D)]
        assert pa == 0 and pb == 5

    def test_defect_cooperate(self):
        pa, pb = PAYOFFS[(D, C)]
        assert pa == 5 and pb == 0

    def test_four_entries(self):
        assert len(PAYOFFS) == 4

    def test_symmetry_cooperate(self):
        assert PAYOFFS[(C, C)][0] == PAYOFFS[(C, C)][1]

    def test_symmetry_defect(self):
        assert PAYOFFS[(D, D)][0] == PAYOFFS[(D, D)][1]

# ── Strategies ────────────────────────────────────────────────────────────────

class TestAlwaysCooperate:
    def setup_method(self):
        self.s = AlwaysCooperate()

    def test_cooperates_empty_history(self):
        assert self.s.choose([], []) == C

    def test_cooperates_with_history(self):
        for _ in range(30):
            assert self.s.choose([D, D, D], [C, D, C]) == C

    def test_name(self):
        assert self.s.name == "Always Cooperate"


class TestAlwaysDefect:
    def setup_method(self):
        self.s = AlwaysDefect()

    def test_defects_empty_history(self):
        assert self.s.choose([], []) == D

    def test_defects_with_history(self):
        for _ in range(30):
            assert self.s.choose([C, C, C], [C, D, C]) == D

    def test_name(self):
        assert self.s.name == "Always Defect"


class TestRandomCooperate:
    def test_p_zero_never_cooperates(self):
        s = RandomCooperate(0.0)
        assert all(s.choose([], []) == D for _ in range(100))

    def test_p_one_always_cooperates(self):
        s = RandomCooperate(1.0)
        assert all(s.choose([], []) == C for _ in range(100))

    @pytest.mark.parametrize("p", [0.2, 0.4, 0.6, 0.8])
    def test_cooperation_rate_close_to_p(self, p):
        np.random.seed(42)
        s = RandomCooperate(p)
        n = 20_000
        rate = sum(1 for _ in range(n) if s.choose([], []) == C) / n
        assert abs(rate - p) < 0.02, f"p={p}: expected ~{p}, got {rate:.4f}"

    def test_memoryless_no_crash(self):
        s = RandomCooperate(0.5)
        s.choose([C, D], [D, C])
        s.choose([], [])
        s.choose([D] * 10, [C] * 10)

    @pytest.mark.parametrize("p", [0.2, 0.4, 0.6, 0.8])
    def test_name_format(self, p):
        assert RandomCooperate(p).name == f"Random(p={p})"

    def test_independent_of_history(self):
        """Same p value should not be influenced by history (statistical check)."""
        np.random.seed(0)
        s = RandomCooperate(0.5)
        n = 5000
        # Rate after long defection history vs empty history should be ~same
        rate_empty = sum(1 for _ in range(n) if s.choose([], []) == C) / n
        rate_full = sum(1 for _ in range(n) if s.choose([D]*20, [D]*20) == C) / n
        assert abs(rate_empty - rate_full) < 0.05


class TestTitForTat:
    def setup_method(self):
        self.s = TitForTat()

    def test_cooperates_first_move(self):
        assert self.s.choose([], []) == C

    def test_mirrors_cooperation(self):
        assert self.s.choose([C], [C]) == C

    def test_mirrors_defection(self):
        assert self.s.choose([C], [D]) == D

    def test_mirrors_last_move_not_first(self):
        # Sequence C,D,C — last was C → cooperate
        assert self.s.choose([C, D, C], [C, D, C]) == C
        # Sequence C,C,D — last was D → defect
        assert self.s.choose([C, C, D], [C, C, D]) == D

    def test_retaliation(self):
        # Opponent defects → TfT retaliates next move
        assert self.s.choose([C], [D]) == D

    def test_forgiveness(self):
        # Opponent defected then cooperated → TfT forgives
        assert self.s.choose([C, D], [D, C]) == C

    def test_name(self):
        assert self.s.name == "Tit-for-Tat"

# ── Match ─────────────────────────────────────────────────────────────────────

class TestMatch:
    def test_has_five_rounds_of_payoffs(self):
        result = match_ac_ad()
        for r in range(1, 6):
            assert f"r{r}_payoff_a" in result
            assert f"r{r}_payoff_b" in result

    def test_has_five_rounds_of_actions(self):
        result = match_ac_ad()
        for r in range(1, 6):
            assert f"r{r}_action_a" in result
            assert f"r{r}_action_b" in result

    def test_mutual_cooperation_payoffs(self):
        result = play_match(make_player(0, AlwaysCooperate()), make_player(1, AlwaysCooperate()))
        for r in range(1, 6):
            assert result[f"r{r}_payoff_a"] == 3
            assert result[f"r{r}_payoff_b"] == 3
        assert result["total_payoff_a"] == 15
        assert result["total_payoff_b"] == 15

    def test_mutual_defection_payoffs(self):
        result = play_match(make_player(0, AlwaysDefect()), make_player(1, AlwaysDefect()))
        for r in range(1, 6):
            assert result[f"r{r}_payoff_a"] == 1
            assert result[f"r{r}_payoff_b"] == 1
        assert result["total_payoff_a"] == 5
        assert result["total_payoff_b"] == 5

    def test_defect_vs_cooperate(self):
        result = match_ac_ad()   # A=cooperate, B=defect
        for r in range(1, 6):
            assert result[f"r{r}_payoff_a"] == 0
            assert result[f"r{r}_payoff_b"] == 5
        assert result["total_payoff_a"] == 0
        assert result["total_payoff_b"] == 25

    def test_tft_vs_always_cooperate_all_cooperate(self):
        result = play_match(make_player(0, TitForTat()), make_player(1, AlwaysCooperate()))
        for r in range(1, 6):
            assert result[f"r{r}_action_a"] == C
            assert result[f"r{r}_payoff_a"] == 3

    def test_tft_vs_always_defect_sequence(self):
        """TfT cooperates round 1, then defects rounds 2-5 (retaliating)."""
        result = play_match(make_player(0, TitForTat()), make_player(1, AlwaysDefect()))
        assert result["r1_action_a"] == C   # cooperates first
        assert result["r1_payoff_a"] == 0   # gets suckered
        for r in range(2, 6):
            assert result[f"r{r}_action_a"] == D   # retaliates
            assert result[f"r{r}_payoff_a"] == 1   # mutual defection

    def test_player_ids_in_result(self):
        result = play_match(make_player(7, AlwaysCooperate()), make_player(42, TitForTat()))
        assert result["player_a_id"] == 7
        assert result["player_b_id"] == 42

    def test_strategy_names_in_result(self):
        result = play_match(make_player(0, AlwaysCooperate()), make_player(1, TitForTat()))
        assert result["strategy_a"] == "Always Cooperate"
        assert result["strategy_b"] == "Tit-for-Tat"

    def test_totals_sum_from_rounds(self):
        result = play_match(make_player(0, RandomCooperate(0.5)), make_player(1, RandomCooperate(0.5)))
        assert result["total_payoff_a"] == sum(result[f"r{r}_payoff_a"] for r in range(1, 6))
        assert result["total_payoff_b"] == sum(result[f"r{r}_payoff_b"] for r in range(1, 6))

    def test_all_payoffs_valid_values(self):
        valid = {0, 1, 3, 5}
        for _ in range(50):
            result = play_match(make_player(0, RandomCooperate(0.5)), make_player(1, RandomCooperate(0.5)))
            for r in range(1, 6):
                assert result[f"r{r}_payoff_a"] in valid
                assert result[f"r{r}_payoff_b"] in valid

    def test_actions_are_c_or_d(self):
        valid_actions = {C, D}
        result = play_match(make_player(0, RandomCooperate(0.5)), make_player(1, RandomCooperate(0.5)))
        for r in range(1, 6):
            assert result[f"r{r}_action_a"] in valid_actions
            assert result[f"r{r}_action_b"] in valid_actions

    def test_custom_num_rounds(self):
        result = play_match(make_player(0, AlwaysCooperate()), make_player(1, AlwaysCooperate()), num_rounds=3)
        assert "r3_payoff_a" in result
        assert "r4_payoff_a" not in result
        assert result["total_payoff_a"] == 9   # 3 rounds × 3

# ── Population ────────────────────────────────────────────────────────────────

class TestPopulation:
    @pytest.fixture(autouse=True)
    def pop(self):
        self.players = create_population(1000)

    def test_total_count(self):
        assert len(self.players) == 1000

    def test_always_cooperate_count(self):
        assert sum(1 for p in self.players if isinstance(p.strategy, AlwaysCooperate)) == 100

    def test_always_defect_count(self):
        assert sum(1 for p in self.players if isinstance(p.strategy, AlwaysDefect)) == 100

    def test_random_total_count(self):
        assert sum(1 for p in self.players if isinstance(p.strategy, RandomCooperate)) == 400

    @pytest.mark.parametrize("p", [0.2, 0.4, 0.6, 0.8])
    def test_each_random_p_count(self, p):
        count = sum(
            1 for pl in self.players
            if isinstance(pl.strategy, RandomCooperate) and pl.strategy.p == p
        )
        assert count == 100, f"Expected 100 players with p={p}, got {count}"

    def test_tft_count(self):
        assert sum(1 for p in self.players if isinstance(p.strategy, TitForTat)) == 400

    def test_unique_ids(self):
        ids = [p.player_id for p in self.players]
        assert len(ids) == len(set(ids))

    def test_ids_are_sequential(self):
        ids = sorted(p.player_id for p in self.players)
        assert ids == list(range(1000))

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            create_population(999)

    def test_small_population(self):
        players = create_population(10)
        assert len(players) == 10


# ── Tournament ────────────────────────────────────────────────────────────────

class TestTournament:
    @pytest.fixture(autouse=True)
    def small_pop(self):
        """Use n=10 for speed in tournament tests."""
        self.players = create_population(10)
        self.records = run_tournament(self.players, verbose=False)

    def test_match_count(self):
        n = len(self.players)
        assert len(self.records) == n * (n - 1) // 2

    def test_all_pairs_distinct(self):
        pairs = set()
        for r in self.records:
            pair = tuple(sorted([r["player_a_id"], r["player_b_id"]]))
            assert pair not in pairs, f"Duplicate match for pair {pair}"
            pairs.add(pair)

    def test_no_self_play(self):
        for r in self.records:
            assert r["player_a_id"] != r["player_b_id"]

    def test_all_players_participate(self):
        seen = set()
        for r in self.records:
            seen.add(r["player_a_id"])
            seen.add(r["player_b_id"])
        expected = {p.player_id for p in self.players}
        assert seen == expected

# ── Analysis DataFrames ───────────────────────────────────────────────────────

class TestAnalysis:
    @pytest.fixture(autouse=True)
    def setup(self):
        players = create_population(10)
        records = run_tournament(players, verbose=False)
        self.match_df = matches_to_df(records)
        self.pm_df = player_match_df(self.match_df)
        self.po_df = player_overall_df(self.pm_df)
        self.sr_df = strategy_round_df(self.pm_df)
        self.so_df = strategy_overall_df(self.po_df)
        self.vs_df = strategy_vs_strategy_df(self.pm_df)
        self.cr_df = cooperation_rate_df(self.pm_df)

    def test_match_df_row_count(self):
        n = 10
        assert len(self.match_df) == n * (n - 1) // 2

    def test_player_match_df_is_double(self):
        assert len(self.pm_df) == 2 * len(self.match_df)

    def test_player_overall_has_all_players(self):
        assert len(self.po_df) == 10

    def test_strategy_round_five_rounds_per_strategy(self):
        for strat, grp in self.sr_df.groupby("strategy"):
            assert len(grp) == 5, f"{strat}: expected 5 rounds, got {len(grp)}"

    def test_cooperation_rate_df_five_rounds_per_strategy(self):
        for strat, grp in self.cr_df.groupby("strategy"):
            assert len(grp) == 5

    def test_cooperation_rates_bounded(self):
        assert (self.cr_df["cooperation_rate"] >= 0).all()
        assert (self.cr_df["cooperation_rate"] <= 1).all()

    def test_always_cooperate_coop_rate_is_one(self):
        ac_rows = self.cr_df[self.cr_df["strategy"] == "Always Cooperate"]
        assert (ac_rows["cooperation_rate"] == 1.0).all()

    def test_always_defect_coop_rate_is_zero(self):
        ad_rows = self.cr_df[self.cr_df["strategy"] == "Always Defect"]
        assert (ad_rows["cooperation_rate"] == 0.0).all()

    def test_player_total_payoff_non_negative(self):
        assert (self.po_df["total_payoff"] >= 0).all()

    def test_avg_per_round_equals_avg_per_match_over_5(self):
        np.testing.assert_allclose(
            self.po_df["avg_payoff_per_round"].values,
            (self.po_df["avg_payoff_per_match"] / 5).values,
            rtol=1e-6,
        )

    def test_vs_matrix_diagonal_makes_sense(self):
        # AlwaysDefect vs AlwaysDefect should have payoff 1.0 per round.
        # Requires ≥2 AD players — only guaranteed with n≥20; skip if cell is NaN.
        vs = self.vs_df
        if "Always Defect" in vs.index and "Always Defect" in vs.columns:
            val = vs.loc["Always Defect", "Always Defect"]
            if not np.isnan(val):
                assert val == pytest.approx(1.0)

    def test_vs_matrix_ac_vs_ad(self):
        # AlwaysCooperate facing AlwaysDefect should get 0.0 per round
        vs = self.vs_df
        if "Always Cooperate" in vs.index and "Always Defect" in vs.columns:
            assert vs.loc["Always Cooperate", "Always Defect"] == pytest.approx(0.0)

    def test_vs_matrix_ad_vs_ac(self):
        # AlwaysDefect facing AlwaysCooperate should get 5.0 per round
        vs = self.vs_df
        if "Always Defect" in vs.index and "Always Cooperate" in vs.columns:
            assert vs.loc["Always Defect", "Always Cooperate"] == pytest.approx(5.0)

    def test_strategy_overall_sorted_descending(self):
        payoffs = self.so_df["avg_total_payoff"].tolist()
        assert payoffs == sorted(payoffs, reverse=True)

# ── Integration: known payoff checks ─────────────────────────────────────────

class TestKnownPayoffs:
    """End-to-end checks using deterministic strategies."""

    def test_all_cooperate_population(self):
        """A homogeneous AC population: every player gets 3 per round."""
        players = [make_player(i, AlwaysCooperate()) for i in range(4)]
        records = run_tournament(players, verbose=False)
        for r in records:
            assert r["total_payoff_a"] == 15
            assert r["total_payoff_b"] == 15

    def test_all_defect_population(self):
        """A homogeneous AD population: every player gets 1 per round."""
        players = [make_player(i, AlwaysDefect()) for i in range(4)]
        records = run_tournament(players, verbose=False)
        for r in records:
            assert r["total_payoff_a"] == 5
            assert r["total_payoff_b"] == 5

    def test_tft_vs_tft_all_cooperate(self):
        """Two TfT players cooperate in every round (both start with C)."""
        result = play_match(make_player(0, TitForTat()), make_player(1, TitForTat()))
        for r in range(1, 6):
            assert result[f"r{r}_action_a"] == C
            assert result[f"r{r}_action_b"] == C
            assert result[f"r{r}_payoff_a"] == 3
            assert result[f"r{r}_payoff_b"] == 3

    def test_tft_total_vs_ad(self):
        """TfT vs AD: round 1 payoff=0, rounds 2-5 payoff=1 each → total 4."""
        result = play_match(make_player(0, TitForTat()), make_player(1, AlwaysDefect()))
        assert result["total_payoff_a"] == 4   # 0 + 1*4
        assert result["total_payoff_b"] == 9   # 5 + 1*4
