# Spatial Prisoner's Dilemma — Notes & Observations

## Why Spatial Structure?

In the standard round-robin tournament (our `tournament/` module), every player meets every other player. This means a single defector immediately reaps rewards from the entire population of cooperators. The global interaction structure makes exploitation trivially easy and cooperation hard to sustain.

The spatial version breaks this assumption. Players only interact with their immediate neighbours. This one change transforms the game completely — because now **where you are matters as much as what strategy you play**.

The key insight: cooperation can survive if cooperators cluster together. A cooperator surrounded by other cooperators does well. A defector surrounded by cooperators does extremely well — but only briefly, because it poisons its own neighbourhood.

---

## Setup

- **Grid**: √N × √N toroidal grid (edges wrap around, no boundary effects)
- **Neighbourhood**: Moore (8 neighbours) or von Neumann (4 neighbours)
- **Each generation**:
  1. Every player plays 5 rounds of PD against each of their neighbours
  2. Each player then adopts the strategy of their highest-scoring neighbour (or keeps their own if they won) — the **imitate-the-best** update rule
- **Strategies**: same catalogue as the tournament — Always Cooperate, Always Defect, Random(p=0.2/0.4/0.6/0.8), Tit-for-Tat
- **Initial assignment**: uniform random across all strategies

---

## What We Observed (32×32, Moore, 20 generations, seed=42)

### Generation 1 — Random Chaos
The population starts as a roughly even mix of all 7 strategies. No clusters, no structure. Every strategy is present in similar numbers (~130–170 cells each).

### Generation 2 — Defectors Surge
Always Defect jumps from 134 to 538 cells (53% of the population). This is the classic first-move advantage of defection: any defector surrounded by cooperators or random players immediately scores the maximum payoff (5 per round) and its neighbours copy it. Always Cooperate collapses to just 2 cells. Random(p=0.8) and Random(p=0.6) shrink sharply — high cooperators get exploited.

This is the moment the naive interpretation of the Prisoner's Dilemma seems to win: defect, always.

### Generations 3–7 — The Plateau and the Turn
Always Defect peaks around generation 3 at 594 cells, then begins a slow decline. Why? Because defectors, once they have eliminated their cooperator neighbours, are left playing against other defectors. And mutual defection pays only 1 per round — far less than the 3 per round that two TfT players earn together.

Meanwhile, TfT clusters — even small ones — are quietly outperforming their AD neighbours at the cluster boundaries. A TfT cell surrounded mostly by other TfT cells earns close to 3 per round from each of them. It only loses on the rounds it faces an AD neighbour. Its AD neighbour, by contrast, earns 5 in the first round against TfT but then faces retaliation for rounds 2–5, averaging far less.

Random cooperators (especially p=0.2 and p=0.4) act as a buffer zone — they're not great cooperators, but they're not pure defectors either. They slow the spread of AD in some directions while TfT consolidates elsewhere.

### Generations 8–17 — TfT's Steady March
TfT grows monotonically from 435 cells at generation 8 to 1,022 at generation 17. The mechanism is spatial: TfT clusters expand outward cell by cell at their boundaries with AD territory. Each boundary cell compares its score to its neighbours'. The TfT cell on the edge earns less than interior TfT cells (some AD neighbours drag it down) but still more than the AD cell just outside the cluster, because that AD cell is surrounded by other ADs earning 1/round.

This is the critical asymmetry: **a TfT cluster has a protected interior that earns 3/round, while an AD cluster's interior earns only 1/round**. The boundary dynamics consistently favour TfT expansion.

Always Defect drops from 240 at generation 8 to 0 at generation 17. It cannot hold territory once it loses access to cooperators to exploit.

### Generation 18 — Full Fixation
TfT achieves complete fixation: 1,024/1,024 cells. Every player cooperates. Every cell earns exactly 3 per round, 120 total. The population has converged to the socially optimal outcome — not through altruism, but through spatial selection pressure.

---

## Why TfT Is Stable Against Invaders

Once TfT dominates the grid, can anything invade?

### Against Always Defect
A lone AD mutant in a sea of TfT earns: 5 (round 1 against each TfT neighbour) + 1×4 (rounds 2–5, TfT retaliates) = 9 per match. Its TfT neighbours earn 0 (round 1) + 1×4 = 4 per match against it, but 3×5 = 15 against their other TfT neighbours. An interior TfT cell earns 8×15 = 120. The TfT cell adjacent to the AD mutant earns roughly 7×15 + 4 = 109. The AD mutant earns 8×9 = 72 — far less than its neighbours. It does not spread. It dies out within one generation.

### Against Always Cooperate
AC would earn even more against TfT than TfT earns against itself (AC never retaliates, so no lost rounds). But AC is immediately exploited by any residual defector or low-p random player. AC cannot survive in any mixed environment. It is not a stable invader.

### Against Random(p)
A Random(p) player in a TfT sea earns based on p. High p values (0.8) cooperate often enough to do reasonably well but still earn less than a TfT player, because they occasionally defect unprovoked and trigger retaliation. Low p values (0.2) defect too often and face the same fate as AD. None can invade a stable TfT population.

**TfT is evolutionarily stable in the spatial setting** because it simultaneously:
1. Cooperates with cooperators (high joint payoff)
2. Punishes defectors immediately (limits exploitation)
3. Forgives after one round (re-establishes cooperation once the defector cooperates)

---

## The Beauty of It

The global tournament (`tournament/`) told us that Always Defect wins outright. Add spatial structure and the result inverts completely. This is not a parameter tweak — it is a fundamental change in what "competition" means.

The spatial model captures something true about real biological and social systems: interactions are local. You don't play against everyone. You play against your neighbours, and your neighbours' strategies determine whether your strategy survives. Cooperation doesn't need altruism or a central planner. It just needs geography.

The clusters of TfT that form in the early generations are not planned. They arise from random initial conditions. But once formed, they are self-reinforcing — cooperators protect each other by being surrounded by cooperators — and expansionary, because the boundary dynamics consistently favour them over defector clusters.

This is Robert Axelrod's core finding from his 1984 tournaments, extended into space: **niceness, provocability, and forgiveness — the three properties of TfT — are sufficient for cooperation to emerge and stabilise, even from a chaotic start.**

---

## Neighbourhood Matters

Moore (8 neighbours) vs von Neumann (4 neighbours) changes the dynamics:

- **Moore**: more neighbours means more total payoff per generation, and faster propagation of strategies. Clusters form and dissolve more quickly. TfT still wins but the timescale is compressed.
- **Von Neumann**: slower dynamics, smaller boundary effects. Clusters are more stable but also slower to expand. Invasion by defectors is easier to resist because each defector has fewer cooperator targets.

Try `--neighbourhood von_neumann` to see TfT take longer to dominate but follow the same eventual trajectory.

---

## What To Try Next

- `--seed N` with different seeds to see how initial random placement affects the invasion dynamics
- `--side 64` for a 64×64 grid (4,096 players) — richer cluster structures, more dramatic visuals
- `--generations 10` with `--neighbourhood von_neumann` — slower burn, more visible boundary dynamics
- Add noise (flip actions with probability ε) to see if TfT remains stable when communication is imperfect — this is where Generous TfT becomes relevant (see `extensions.md`)
