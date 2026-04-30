# Challenge 3 — NSGA-II + bounded-terrain refinement + CMA-ES

## Reward design
Co-evolving 8 leg lengths + MLP (27→8→8, 280 w, no bias) on 5° hill
(bump=0.1, σ=3). Two objectives mirror the TA eval:

- **Obj 1** = `reward_forward + healthy_reward`, `reward_forward =
  x_velocity − 0.3·|y_velocity|` — penalises lateral drift.
- **Obj 2** = `−ctrl_cost` — actuator efficiency.

**Out-of-bounds termination**: hfield only spans |y|≤20 m; ants drifting
past |y|>19 m get `info["healthy_reward"]=−50` and episode terminates. This
kills diagonal runners that would game +x reward by leaving the hill's side.

## Training pipeline
1. NSGA-II baseline (pop=150, DE cross=0.9, mut=0.5, 200 gens, pure x-vel).
2. Curriculum (1°→3°→5°, 90 gens, 750-step horizon), warm-start from (1).
3. **Bounded-terrain refinement** (50 gens NSGA-II, y-penalty + OOB term).
4. **CMA-ES fine-tune** (30 gens obj1, 1000-step) of best straight seed.

## Pareto trade-off (256 ep × 1000 step)
Obj1 1006–1224, obj2 -19.3–-0.06.
Spec obj1 (`x_best.npy`, CMA champion) walks **~15 m uphill, 98–99 %
heading**. Spec obj2 freezes (~0 ctrl). Generalist sits at the Pareto knee.

## Morphology — leg lengths via `(g+1)/4+0.1` (m)

| Param | Spec. obj1 | Spec. obj2 | Generalist |
|-------|-----------|-----------|-----------|
| FL_upper | 0.581 | 0.527 | 0.600 |
| FL_lower | 0.558 | 0.426 | 0.586 |
| FR_upper | 0.587 | 0.406 | 0.594 |
| FR_lower | 0.313 | 0.466 | 0.327 |
| BL_upper | 0.529 | 0.243 | 0.600 |
| BL_lower | 0.596 | 0.539 | 0.586 |
| BR_upper | 0.540 | 0.457 | 0.356 |
| BR_lower | 0.594 | 0.362 | 0.584 |

Spec obj1 → **long legs** (stride); spec obj2 → shorter (low torque).

## Contents
`x_best.npy` (CMA champion), `Challenge3.py`, `ant_hill.py`, `mlp.py`,
`nsga.py`, `pareto_front.pdf`, `morphology_comparison.pdf`, 3 videos.
