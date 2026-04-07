"""
simulation.py — Core Prisoner's Dilemma simulation logic.

Population (N=1000):
  - N/10  Always Cooperate
  - N/10  Always Defect
  - N/10  each for Random(p) with p in [0.2, 0.4, 0.6, 0.8]
  - 4N/10 Tit-for-Tat (cooperates on round 1, then mirrors)

Tournament: every pair plays num_rounds rounds (default 5).
"""

import itertools
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# ── Actions & Payoff Matrix ───────────────────────────────────────────────────

C, D = "C", "D"

PAYOFFS: Dict[Tuple[str, str], Tuple[int, int]] = {
    (C, C): (3, 3),
    (C, D): (0, 5),
    (D, C): (5, 0),
    (D, D): (1, 1),
}

# ── Strategies ────────────────────────────────────────────────────────────────


class Strategy:
    name: str = ""

    def choose(self, my_history: List[str], opp_history: List[str]) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name


class AlwaysCooperate(Strategy):
    name = "Always Cooperate"

    def choose(self, my_history, opp_history):
        return C


class AlwaysDefect(Strategy):
    name = "Always Defect"

    def choose(self, my_history, opp_history):
        return D


class RandomCooperate(Strategy):
    def __init__(self, p: float):
        self.p = p
        self.name = f"Random(p={p})"

    def choose(self, my_history, opp_history):
        return C if np.random.random() < self.p else D


class TitForTat(Strategy):
    name = "Tit-for-Tat"

    def choose(self, my_history, opp_history):
        if not opp_history:
            return C
        return opp_history[-1]


# ── Player ────────────────────────────────────────────────────────────────────


class Player:
    def __init__(self, player_id: int, strategy: Strategy):
        self.player_id = player_id
        self.strategy = strategy

    @property
    def strategy_name(self) -> str:
        return self.strategy.name

    def __repr__(self) -> str:
        return f"Player({self.player_id}, {self.strategy_name})"


# ── Match ─────────────────────────────────────────────────────────────────────


def play_match(
    player_a: Player,
    player_b: Player,
    num_rounds: int = 5,
) -> Dict[str, Any]:
    """
    Plays a match between two players.

    Returns a flat dict with:
      - player_a_id, strategy_a, player_b_id, strategy_b
      - r{i}_action_a, r{i}_payoff_a, r{i}_action_b, r{i}_payoff_b  (i = 1..num_rounds)
      - total_payoff_a, total_payoff_b
    """
    hist_a: List[str] = []
    hist_b: List[str] = []
    round_data: Dict[str, Any] = {}

    for r in range(1, num_rounds + 1):
        action_a = player_a.strategy.choose(hist_a, hist_b)
        action_b = player_b.strategy.choose(hist_b, hist_a)
        payoff_a, payoff_b = PAYOFFS[(action_a, action_b)]

        hist_a.append(action_a)
        hist_b.append(action_b)

        round_data[f"r{r}_action_a"] = action_a
        round_data[f"r{r}_payoff_a"] = payoff_a
        round_data[f"r{r}_action_b"] = action_b
        round_data[f"r{r}_payoff_b"] = payoff_b

    total_a = sum(round_data[f"r{r}_payoff_a"] for r in range(1, num_rounds + 1))
    total_b = sum(round_data[f"r{r}_payoff_b"] for r in range(1, num_rounds + 1))

    return {
        "player_a_id": player_a.player_id,
        "strategy_a": player_a.strategy_name,
        "player_b_id": player_b.player_id,
        "strategy_b": player_b.strategy_name,
        "total_payoff_a": total_a,
        "total_payoff_b": total_b,
        **round_data,
    }


# ── Population ────────────────────────────────────────────────────────────────


def create_population(n: int = 1000) -> List[Player]:
    """
    Creates population according to the agreed split:
      N/10  Always Cooperate
      N/10  Always Defect
      N/10  each for Random(p) with p in [0.2, 0.4, 0.6, 0.8]
      4N/10 Tit-for-Tat
    """
    if n % 10 != 0:
        raise ValueError(f"N must be divisible by 10, got {n}")

    tenth = n // 10
    players: List[Player] = []
    pid = 0

    for _ in range(tenth):
        players.append(Player(pid, AlwaysCooperate()))
        pid += 1

    for _ in range(tenth):
        players.append(Player(pid, AlwaysDefect()))
        pid += 1

    for p in [0.2, 0.4, 0.6, 0.8]:
        for _ in range(tenth):
            players.append(Player(pid, RandomCooperate(p)))
            pid += 1

    for _ in range(4 * tenth):
        players.append(Player(pid, TitForTat()))
        pid += 1

    return players


# ── Tournament ────────────────────────────────────────────────────────────────


def run_tournament(
    players: List[Player],
    num_rounds: int = 5,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    All-pairs round-robin tournament.
    Every pair plays exactly num_rounds rounds.
    Returns list of match record dicts (one per pair).
    """
    pairs = list(itertools.combinations(players, 2))
    records: List[Dict[str, Any]] = []

    it = tqdm(pairs, desc="Running tournament", unit="match") if verbose else pairs
    for player_a, player_b in it:
        records.append(play_match(player_a, player_b, num_rounds))

    return records
