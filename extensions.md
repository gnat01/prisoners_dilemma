# Extensions to the Population Prisoner's Dilemma

## 1. Richer Strategies

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

## 2. Noisy Environment
- With probability ε, an intended action is flipped (C→D or D→C)
- Models miscommunication or trembling-hand errors
- TfT is brittle under noise (one mistake triggers a defection spiral); Generous TfT and Pavlov are more robust
- Track how error rates affect the strategy ranking

---

## 3. Evolutionary Dynamics (the big one)

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

## 4. Spatial / Network Structure
- Players occupy nodes on a graph (lattice, small-world, scale-free)
- Each player only plays against their k neighbours, not the full population
- Cooperation is known to survive better on lattices — test this
- Compare outcomes on random graphs vs clustered networks

---

## 5. Asymmetric Payoff Matrices
- Allow the payoff matrix to vary by player pair or by round
- Model situations where players have different costs of cooperation
- Introduce resource constraints: a player with low "budget" defects more

---

## 6. Memory Depth > 1
- Strategies that condition on the last m rounds, not just the last 1
- e.g. only retaliate if opponent defected in both of the last 2 rounds
- Combinatorial explosion of strategy space — worth sampling randomly (as in Axelrod's tournaments)

---

## 7. Reputation & Signalling
- Players accumulate a public reputation score based on past behaviour
- Others condition their strategy on the opponent's reputation before the match starts
- Models indirect reciprocity ("I help you because others are watching")

---

## 8. Repeated Tournament (Axelrod-style)
- Run multiple generations of the round-robin, letting strategy counts evolve
- After each generation, drop the bottom k% of strategies and replicate the top k%
- Compare final steady-state composition across different starting mixes

---

## 9. Analytical Overlays
- Nash equilibrium check: verify that mutual defection is the unique NE in the one-shot game, but cooperation can be sustained in repeated play
- Price equation decomposition: partition payoff changes into within-group and between-group selection
- Lorenz curve / Gini coefficient of payoff inequality across players and strategies
- Phase diagrams: for evolutionary runs, plot strategy shares over time on a simplex

---

## 10. Richer Output & Interactivity
- Animated generation-by-generation evolution of strategy shares (bar chart race)
- Interactive HTML plots (Plotly/Altair) so you can hover over individual players
- Per-player "career" timeline: payoff across each of the 999 matches in order
- Head-to-head win/draw/loss matrix (not just average payoff)
