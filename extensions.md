# Extensions to the Population Prisoner's Dilemma

Status legend: ✅ Implemented · 🔲 Not yet built

---

## 1. Richer Strategies 🔲

### Conditional Random (bolted on next)
- Random(p) where p shifts based on what the opponent did last round
- e.g. cooperate with p=0.8 if opponent cooperated, p=0.2 if they defected
- Bridges the gap between pure Random and TfT

### Grudger (Grim Trigger)
- Cooperates until the opponent defects once, then defects forever
- Harsher than TfT — tests whether unforgiving retaliation pays off

### Pavlov (Win-Stay, Lose-Shift)
- Repeats last action if it got a good payoff (3 or 5), switches otherwise
- Known to outperform TfT in noisy environments

### Generous TfT
- Like TfT but occasionally forgives a defection with probability q
- Breaks mutual defection spirals that TfT can get locked into

### Suspicious TfT
- Defects on round 1 instead of cooperating, then mirrors
- Tests how first-move aggression propagates

### n-TfT
- Only retaliates after n consecutive defections (more forgiving)

### Lookahead / Exploiter
- Cooperates for k rounds to probe, then defects if opponent is AC

---

## 2. Noisy Environment 🔲
- With probability ε, an intended action is flipped (C→D or D→C)
- Models miscommunication or trembling-hand errors
- TfT is brittle under noise (one mistake triggers a defection spiral); Generous TfT and Pavlov are more robust
- Track how error rates affect the strategy ranking

---

## 3. Evolutionary Dynamics 🔲

### Replicator dynamics
- After each tournament generation, strategies that performed above average grow in population share; below-average strategies shrink
- Proportional update: n_i(t+1) = n_i(t) × (π_i / π̄)
- Run for G generations and watch which strategies take over or go extinct

### Imitation of best
- Each player adopts the strategy of the highest-scoring player in their neighbourhood with some probability
- Already partially done in the Rubinstein game in this repo

### Mutation
- With probability μ, a player randomly switches to any strategy
- Prevents fixation and allows re-invasion — tests evolutionary stability

### Invasion analysis
- Start with a near-homogeneous population (e.g. 99% TfT, 1% AD)
- Ask: can the minority invade and grow, or does it die out?
- Repeat for every (resident, invader) pair to build an invasion matrix

---

## 4. Spatial / Network Structure ✅

### Grid (Spatial Prisoner's Dilemma) ✅
**Implemented in `spatial/`**

- √N × √N toroidal grid (32×32 = 1,024 players by default)
- Moore (8 neighbours) or von Neumann (4 neighbours) — CLI flag `--neighbourhood`
- Each generation: every player plays `--rounds` rounds against each neighbour, then adopts the strategy of their highest-scoring neighbour (imitate-the-best)
- G generations controlled via `--generations` CLI flag

**Outputs** (`spatial/plots/`):
- `01_strategy_map_final.png` — strategy per cell, final generation
- `02_avg_payoff_final.png` — avg payoff per round per cell, final generation
- `03_total_payoff_final.png` — total payoff per cell, final generation
- `anim_strategy_map.gif` — strategy clusters evolving across all generations
- `anim_avg_payoff.gif` — avg payoff heatmap animated
- `anim_total_payoff.gif` — total payoff heatmap animated

**Key finding**: Always Defect surges to 53% in generation 2, then collapses. TfT achieves full fixation by generation 18 (seed=42, Moore, 32×32). See `spatial/spatial_pd_notes.md` for the full narrative.

**Run**:
```bash
python prisoners_dilemma/spatial/main.py --side 32 --generations 20 --neighbourhood moore --seed 42
```

### Other network structures 🔲
- Small-world graphs (Watts-Strogatz)
- Scale-free networks (Barabási-Albert)
- Compare cooperation rates across network topologies

---

## 5. Asymmetric Payoff Matrices 🔲
- Allow the payoff matrix to vary by player pair or by round
- Model situations where players have different costs of cooperation
- Introduce resource constraints: a player with low "budget" defects more

---

## 6. Memory Depth > 1 🔲
- Strategies that condition on the last m rounds, not just the last 1
- e.g. only retaliate if opponent defected in both of the last 2 rounds
- Combinatorial explosion of strategy space — worth sampling randomly (as in Axelrod's tournaments)

---

## 7. Reputation & Signalling 🔲
- Players accumulate a public reputation score based on past behaviour
- Others condition their strategy on the opponent's reputation before the match starts
- Models indirect reciprocity ("I help you because others are watching")

---

## 8. Repeated Tournament (Axelrod-style) 🔲
- Run multiple generations of the round-robin, letting strategy counts evolve
- After each generation, drop the bottom k% of strategies and replicate the top k%
- Compare final steady-state composition across different starting mixes

---

## 9. Analytical Overlays

### Lorenz Curve / Gini Coefficient ✅
**Implemented in `tournament/inequality.py` and `spatial/inequality.py`**

Lorenz curves and Gini coefficients measuring payoff inequality within and across strategies.

**Tournament findings** (`tournament/plots/`):
- Within-strategy Gini is very low (0.002–0.004) — round-robin means everyone faces the same opponents, so luck barely matters
- Whole-population Gini = 0.026 — an order of magnitude higher than within-strategy, confirming strategy choice is the dominant driver of inequality
- TfT has the lowest within-strategy Gini (0.0022); Random(p=0.4) the highest (0.0042)
- Plots: `07_lorenz_by_strategy.png`, `08_lorenz_population.png`, `09_gini_coefficients.png`

**Spatial findings** (`spatial/plots/`):
- Within-strategy Gini at generation 1 is much higher (0.08–0.13) — neighbourhood luck genuinely matters; where you start on the grid shapes your fate
- Whole-population Gini = 0.138 at generation 1, more than 5× the tournament figure
- Always Cooperate has the highest Gini (0.127) — AC players at the edge of a defector cluster are devastated; AC players inside a TfT cluster are fine
- TfT has the lowest Gini (0.081) even in the spatial setting — clusters provide mutual protection
- By generation 18 all Ginis collapse to exactly 0 as TfT fixates and every cell earns 120
- Plots: `07_spatial_lorenz_by_strategy.png`, `08_spatial_lorenz_population.png`, `09_spatial_gini_bar.png`, `10_spatial_gini_over_generations.png`

**Run**:
```bash
# Tournament
python prisoners_dilemma/tournament/inequality.py

# Spatial (run as part of main)
python prisoners_dilemma/spatial/main.py --side 32 --generations 20
```

### Nash Equilibrium Check 🔲
- Verify that mutual defection is the unique NE in the one-shot game
- Show that cooperation can be sustained in the repeated game via trigger strategies
- Contrast with empirical simulation results — backward induction predicts defection in finite games but TfT sustains cooperation anyway

### Price Equation Decomposition 🔲
- Partition payoff changes between generations into within-group and between-group selection terms
- Cov(w, z) / w̄ = between-group: strategies with above-average payoff grow
- E(wΔz) / w̄ = within-group: transmission bias (zero without mutation)
- Rigorously quantifies how much of TfT's rise is due to outperforming vs internal change

### Phase Diagrams on a Simplex 🔲
- For a 3-strategy subset (e.g. AC / AD / TfT), represent population mix as a point on an equilateral triangle
- Plot replicator-dynamic trajectories as a vector field — arrows show direction of evolution from any starting mix
- Fixed points reveal attractors (TfT corner), repellers (AC corner), and saddle points
- Requires implementing replicator dynamics first (see §3)

---

## 10. Richer Output & Interactivity 🔲
- Animated generation-by-generation evolution of strategy shares (bar chart race)
- Interactive HTML plots (Plotly/Altair) so you can hover over individual players
- Per-player "career" timeline: payoff across each of the 999 matches in order
- Head-to-head win/draw/loss matrix (not just average payoff)
