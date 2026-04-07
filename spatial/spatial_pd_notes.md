# Spatial Prisoner's Dilemma — Notes & Observations

## Why Spatial Structure?

In the standard round-robin tournament (our `tournament/` module), every player meets every other player. This means a single defector immediately reaps rewards from the entire population of cooperators. The global interaction structure makes exploitation trivially easy and cooperation hard to sustain.

The spatial version breaks this assumption. Players only interact with their immediate neighbours. This one change transforms the game completely — because now **where you are matters as much as what strategy you play**.

The key insight: cooperation can survive if cooperators cluster together. A cooperator surrounded by other cooperators does well. A defector surrounded by cooperators does extremely well — but only briefly, because it poisons its own neighbourhood.

---

## Setup

- **Grid**: √N × √N toroidal grid (edges wrap around, no boundary effects)
- **Neighbourhood**: Moore (8 neighbours) or von Neumann (4 neighbours) — `--neighbourhood` flag
- **Generations**: controlled via `--generations` flag
- **Each generation**:
  1. Every player plays `--rounds` rounds of PD against each of their neighbours
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

## Inequality Analysis (Lorenz / Gini)

`spatial/inequality.py` runs automatically as part of `main.py` and produces four plots.

### What the numbers show (32×32, Moore, generation 1, seed=42)

| Strategy | Gini | Mean payoff | Std |
|---|---|---|---|
| Tit-for-Tat | 0.0806 | 88.3 | 12.6 |
| Always Defect | 0.0974 | 117.4 | 20.2 |
| Random(p=0.2) | 0.1015 | 107.5 | 19.5 |
| Random(p=0.4) | 0.1021 | 98.6 | 18.0 |
| Random(p=0.6) | 0.1137 | 87.1 | 17.7 |
| Random(p=0.8) | 0.1193 | 82.6 | 17.5 |
| Always Cooperate | 0.1272 | 70.4 | 15.9 |
| **Whole Population** | **0.1379** | **92.2** | **22.7** |

Compare this to the tournament, where within-strategy Gini was 0.002–0.004. Here it is 0.08–0.13 — **neighbourhood luck genuinely matters**. Where you start on the grid shapes your fate in a way that doesn't exist in the round-robin.

### Why the spatial Gini is so much higher

In the tournament, every player faces exactly the same 999 opponents. Luck barely enters. In the spatial model, a TfT player who happens to start surrounded by other TfT players earns close to the maximum from turn 1. A TfT player who starts surrounded by AD players earns almost nothing in early generations. Same strategy, radically different outcomes depending on initial placement.

### Always Cooperate: highest inequality, lowest mean

AC has both the worst average payoff (70.4) and the highest within-strategy Gini (0.127). The two facts are linked: AC players in a good neighbourhood (surrounded by TfT or other AC) do reasonably well; AC players in a bad neighbourhood (surrounded by AD) are completely wiped out. The spread is enormous. This is what pure exploitability looks like spatially.

### TfT: lowest inequality despite being in a mixed world

TfT's Gini of 0.081 is the lowest of any strategy. Even in a chaotic generation-1 population, TfT clusters provide enough mutual protection that outcomes are relatively even across TfT players. The cluster structure buffers individual TfT players from the worst neighbourhood effects.

### The collapse

By generation 18, TfT has fixed at 100% of the population and every cell earns exactly 120. The Gini for every strategy and for the whole population drops to exactly 0. The `10_spatial_gini_over_generations.png` plot shows this collapse — inequality dissolves as cooperation takes over. It is arguably the most elegant single visualisation this project produces.

---

## Outputs

All plots saved to `spatial/plots/`:

| File | Description |
|---|---|
| `01_strategy_map_final.png` | Strategy per cell, final generation |
| `02_avg_payoff_final.png` | Avg payoff per round per cell, final generation |
| `03_total_payoff_final.png` | Total payoff per cell, final generation |
| `anim_strategy_map.gif` | Strategy cluster evolution across all generations |
| `anim_avg_payoff.gif` | Avg payoff heatmap animated |
| `anim_total_payoff.gif` | Total payoff heatmap animated |
| `07_spatial_lorenz_by_strategy.png` | Lorenz curves per strategy, generation 1 |
| `08_spatial_lorenz_population.png` | Whole-population Lorenz curve, generation 1 |
| `09_spatial_gini_bar.png` | Gini bar chart per strategy, generation 1 |
| `10_spatial_gini_over_generations.png` | Gini trajectory across all generations |

---

## How To Run

All commands run from the `game_theory/` root directory.

### Basic run (all defaults)
```bash
python prisoners_dilemma/spatial/main.py
```
32×32 grid · Moore neighbourhood · 20 generations · 5 rounds/match · seed 42 · animations on.

### Skip animations (much faster for quick experiments)
```bash
python prisoners_dilemma/spatial/main.py --no-anim
```

### Full run with all flags explicit
```bash
python prisoners_dilemma/spatial/main.py \
  --side 32 \
  --generations 20 \
  --neighbourhood moore \
  --rounds 5 \
  --fps 4 \
  --seed 42 \
  --output-dir prisoners_dilemma/spatial/plots
```

### All flags reference

| Flag | Default | Description |
|---|---|---|
| `--side` | 32 | Grid side length. Total players = side². |
| `--generations` | 20 | Number of evolutionary generations (G). |
| `--neighbourhood` | moore | `moore` (8 neighbours) or `von_neumann` (4 neighbours). |
| `--rounds` | 5 | Rounds of PD played per match within a generation. |
| `--fps` | 4 | Animation frames per second for the GIFs. |
| `--seed` | 42 | Random seed for reproducibility. |
| `--output-dir` | `prisoners_dilemma/spatial/plots` | Directory for all plots and animations. Created if it doesn't exist. |
| `--no-anim` | off | Pass this flag to skip GIF generation. |

### Interesting configurations to try

```bash
# Larger grid — richer cluster structures, more dramatic visuals
python prisoners_dilemma/spatial/main.py --side 64 --generations 30 --no-anim

# Von Neumann neighbourhood — slower dynamics, more visible boundary effects
python prisoners_dilemma/spatial/main.py --neighbourhood von_neumann --generations 30

# Different seed — watch how initial placement changes the invasion story
python prisoners_dilemma/spatial/main.py --seed 7

# Slow animation — easier to follow the cluster dynamics frame by frame
python prisoners_dilemma/spatial/main.py --fps 2
```

---

## What To Try Next

- `--seed N` with different seeds to see how initial random placement affects the invasion dynamics
- `--side 64` for a 64×64 grid (4,096 players) — richer cluster structures, more dramatic visuals
- `--generations 10` with `--neighbourhood von_neumann` — slower burn, more visible boundary dynamics
- Add noise (flip actions with probability ε) to see if TfT remains stable when communication is imperfect — this is where Generous TfT becomes relevant (see `extensions.md`)
- Mutation: with probability μ a player randomly switches strategy instead of imitating best — prevents fixation and allows re-invasion
