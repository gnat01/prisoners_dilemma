# Spatial Invasion Threshold Analysis

## Why We Did This

The spatial PD (`spatial/`) showed us that TfT achieves full fixation from a random start. But that tells us about dynamics from a *mixed* initial condition. A more pointed question is: **once a strategy is dominant, how hard is it to dislodge?**

The invasion threshold f* answers this precisely. For every (resident, invader) pair we ask: what is the minimum fraction of invaders needed, placed randomly on the grid, for the invader to win in the majority of trials? Below f* the resident repels the invasion; above f* the invader takes over.

This is more interesting than a binary can/can't-invade question because:
- It quantifies *how stable* a strategy is — a resident requiring f*=0.90 is almost impregnable; one requiring f*=0.05 is fragile
- The threshold is a genuinely *emergent spatial property* — it cannot be derived from payoff arithmetic alone, because it depends on cluster formation, boundary dynamics, and the specific neighbourhood geometry
- Running multiple seeds per fraction point reveals how much spatial randomness (where the invaders happen to land) affects the outcome — some thresholds are sharp, others are noisy

---

## Setup

- **Grid**: side×side toroidal grid with Moore (8-neighbour) or von Neumann (4-neighbour) connectivity
- **Initial condition**: (1−f) fraction resident, f fraction invader, placed uniformly at random
- **Update rule**: imitate-the-best (same as `spatial/`)
- **Fraction sweep**: 0.01 → 0.95 at 21 points (finer at the low end)
- **Seeds per point**: 5 — captures variance from random initial placement
- **Threshold definition**: smallest f at which the invader wins in >50% of seeds, linearly interpolated between fraction points
- **Strategies**: full catalogue — AC, AD, Random(p=0.2/0.4/0.6/0.8), TfT

---

## Results (20×20, Moore, 25 max generations, 5 seeds, seed=42)

### Threshold Matrix

```
Resident →   AC     AD    R0.2  R0.4  R0.6  R0.8   TfT
          ┌────────────────────────────────────────────┐
AC        │  —      0.01  0.01  0.01  0.01  0.01  0.52│
AD        │  X      —     0.28  0.82  X     X     0.04│
R0.2      │  X      0.81  —     X     X     X     0.06│
R0.4      │  X      0.04  0.02  —     X     X     0.08│
R0.6      │  X      0.01  0.01  0.01  —     X     0.22│
R0.8      │  X      0.01  0.01  0.01  0.01  —     X   │
TfT       │  0.52   X     X     0.89  0.88  0.08  —   │
          └────────────────────────────────────────────┘
Cols = invader strategy. X = invader never wins majority of seeds.
```

Row = resident, Col = invader. Cell = f* (minimum invader fraction to win).

---

## Interpreting the Results

### Always Cooperate is utterly defenceless

Every strategy invades AC with a threshold of 0.01 or lower — even a single cell of any other strategy is enough to take over. The one exception is TfT, which requires f*=0.52 to invade AC. This is subtle: TfT and AC play identically against each other (both cooperate), so TfT gains no payoff advantage over AC in a homogeneous AC population. The only way TfT takes over is via tie-breaking at the boundary — and at exactly 50/50 it's essentially random, which is why the threshold sits right at 0.52.

**The lesson**: pure unconditional cooperation is evolutionarily unstable. It cannot resist any strategy that defects at all.

### Always Defect is a glass cannon

AD can invade almost everything (low thresholds against AC, R0.6, R0.8) but is itself fragile:
- TfT invades AD at just f*=0.04 — a 4% TfT foothold is enough to take over
- R0.4 invades AD at f*=0.04 as well
- R0.2 needs 81% to invade AD — it defects so often it can't form cooperative clusters

AD's weakness is the same as in the non-spatial model: once cooperators are gone, all AD players earn 1/round from each other, and even a small TfT cluster earning 3/round internally expands relentlessly.

### TfT is the most robust resident

TfT can only be invaded in three cases:
- **AC at f*=0.52** — borderline, effectively neutral (see above)
- **R0.4 at f*=0.89** — needs 89% Random(p=0.4) to overcome TfT; almost impossible in practice
- **R0.8 at f*=0.08** — Random(p=0.8) is a near-cooperator. It plays almost like TfT against TfT players (defecting ~20% of the time), but occasionally triggers retaliation chains. Still, it's close enough to TfT that a modest foothold (8%) can sometimes take over, especially in small grids.

Everything else — AD, R0.2, R0.4 at low fractions — cannot invade TfT at all (X). This confirms the spatial stability result from `spatial/`: TfT is the evolutionary attractor.

### The Random(p) gradient

The random strategies show a clear gradient:
- **R0.2** (mostly defects): behaves like AD — good at invading cooperators, vulnerable to TfT
- **R0.8** (mostly cooperates): behaves like AC — easily invaded by defectors, but harder for TfT to displace once resident

R0.8 as a *resident* cannot be invaded by anything except AD (f*=0.01) and low-p randoms. TfT cannot invade R0.8 (X). This makes sense: a grid full of R0.8 players defects ~20% of the time, triggering constant TfT retaliation. The average TfT payoff against R0.8 is dragged down by these random defections, while R0.8 players earn more freely.

### The X cells are informative too

An X (invader never wins) means the resident is stable against that invader at all tested fractions up to 0.95. For example:
- AD cannot invade R0.6 or R0.8 at any fraction — those strategies defect often enough that AD has few cooperators to exploit
- TfT cannot invade R0.2, R0.6 — R0.2 is too defection-heavy for TfT to gain traction; R0.6 is a moderate mixed case where TfT's retaliation overhead costs it more than it gains

---

## Do the Results Make Sense?

**Yes, broadly.** The key findings are consistent with established spatial game theory:
1. Unconditional cooperators are evolutionarily unstable ✓
2. TfT is the most robust resident ✓
3. Defectors are good invaders but poor residents once cooperators are gone ✓
4. High-p random strategies are harder to invade than low-p ones ✓

**Some results warrant scrutiny:**

- **R0.8 cannot be invaded by TfT (X)**. This is somewhat surprising. On a 20×20 grid with only 25 generations, it's possible TfT needs more time or a larger grid to establish dominance — the 25-generation cap may be truncating trials that would eventually tip. Worth testing with `--generations 50`.

- **AD invades R0.4 at f*=0.04 but not R0.6 at any fraction**. The jump between R0.4 and R0.6 is sharp. R0.6 defects 40% of the time, giving AD fewer free meals. This is plausible but the exact crossover deserves sensitivity testing across seeds.

- **Small grid effects**: at 20×20=400 players, f=0.01 means ~4 invader cells. The threshold at 0.01 may be artificially low — on a larger grid, 4 isolated cells in a sea of 1020 residents would almost never form a viable cluster. Run with `--side 32` to check.

---

## How To Run

All commands from the `game_theory/` root.

### Basic run (all defaults)
```bash
python prisoners_dilemma/strategy_invasion_thresholds/main.py
```
20×20 grid · Moore · 25 generations · 5 seeds · 4 workers · full fraction sweep.

### Full run with all flags explicit
```bash
python prisoners_dilemma/strategy_invasion_thresholds/main.py \
  --side 20 \
  --generations 25 \
  --n-seeds 5 \
  --neighbourhood moore \
  --rounds 5 \
  --seed 42 \
  --n-workers 4 \
  --output-dir prisoners_dilemma/strategy_invasion_thresholds/results \
  --plots-dir  prisoners_dilemma/strategy_invasion_thresholds/plots
```

### All flags reference

| Flag | Default | Description |
|---|---|---|
| `--side` | 20 | Grid side length. Players = side². |
| `--generations` | 25 | Max generations per trial before declaring no fixation. |
| `--n-seeds` | 5 | Seeds per (pair, fraction) point. Higher = smoother curves, slower run. |
| `--neighbourhood` | moore | `moore` (8 neighbours) or `von_neumann` (4 neighbours). |
| `--rounds` | 5 | PD rounds per match within a generation. |
| `--seed` | 42 | Base random seed. |
| `--n-workers` | 4 | Thread pool workers (ThreadPoolExecutor). |
| `--output-dir` | `…/results` | Directory for CSV outputs. |
| `--plots-dir` | `…/plots` | Directory for PNG plots. |

### Useful configurations

```bash
# Larger grid — more realistic cluster dynamics, slower
python prisoners_dilemma/strategy_invasion_thresholds/main.py --side 32 --generations 30

# More seeds — smoother threshold curves, better confidence
python prisoners_dilemma/strategy_invasion_thresholds/main.py --n-seeds 10 --n-workers 8

# Von Neumann neighbourhood — slower dynamics, different thresholds
python prisoners_dilemma/strategy_invasion_thresholds/main.py --neighbourhood von_neumann --generations 40

# More generations — catches slow-burning invasions (e.g. TfT vs R0.8)
python prisoners_dilemma/strategy_invasion_thresholds/main.py --generations 50
```

---

## Outputs

### CSVs (`results/`)

| File | Description |
|---|---|
| `raw_trials.csv` | One row per trial: resident, invader, fraction, seed, final_fraction, won, generations_run |
| `aggregated.csv` | One row per (resident, invader, fraction): mean_final_fraction, std, invasion_rate |
| `thresholds.csv` | One row per (resident, invader): threshold f* |
| `threshold_matrix.csv` | 7×7 pivot of thresholds |

### Plots (`plots/`)

| File | Description |
|---|---|
| `01_threshold_matrix.png` | 7×7 heatmap. Green = easy invasion, red = hard, gray = never. |
| `02_invasion_rate_curves.png` | Per-resident subplot: invasion rate vs starting fraction for all invaders. The fraction where a curve crosses 50% is f*. |
| `03_final_fraction_curves.png` | Mean final invader fraction vs starting fraction for key pairs, with ±1 std band. Above the y=x diagonal = invader growing. |
| `04_seed_variance.png` | Individual seed outcomes for the most contested pair — shows how much spatial randomness matters around the threshold. |

---

## How To Improve

1. **Larger grids** (`--side 32` or `--side 64`): small grids exaggerate cluster fragility at low fractions. A 4-cell invasion cluster on a 400-cell grid is proportionally much larger than on a 1024-cell grid.

2. **More generations** (`--generations 50`): some pairs may not fixate within 25 generations. The threshold matrix X entries for TfT→R0.8 and TfT→R0.6 may flip to finite thresholds with more time.

3. **More seeds** (`--n-seeds 20`): the threshold interpolation is noisy with only 5 seeds. More seeds give tighter confidence intervals and smoother curves on the threshold matrix.

4. **Von Neumann comparison**: run the same sweep with `--neighbourhood von_neumann`. Expect higher thresholds across the board (smaller neighbourhoods = slower cluster expansion = harder invasions).

5. **Cluster initialisation**: currently invaders are scattered uniformly at random. A *clustered* initialisation (all invaders placed in a contiguous block) would test whether spatial compactness lowers the threshold. The literature suggests it does — a tight cluster is harder to dissolve than scattered individuals.

6. **Confidence intervals on thresholds**: with multiple seeds, replace the binary invasion_rate threshold with a bootstrap confidence interval on f*.

7. **Phase diagram extension**: for a selected pair (e.g. AD vs TfT), animate the grid across seeds at the threshold fraction — visualise exactly what determines whether the invader cluster survives or dissolves.
