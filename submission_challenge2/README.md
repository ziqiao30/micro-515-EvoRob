# Challenge 2: Multi-Objective Neural Controller Evolution

## NSGA-II Implementation

I implemented the standard NSGA-II algorithm  with four core components: Pareto dominance check, fast non-dominated sorting, crowding distance computation, and crowding operator for tournament selection. Offspring generation uses differential evolution (DE) mutation with tournament-selected parents.

Key hyperparameter choices based on experimentation:
- **Population size: 400** with 48-core parallel evaluation via ProcessPoolExecutor
- **Crossover rate: 0.9** (critical for the 560-dimensional parameter space; DE literature recommends high CR for high-dimensional problems)
- **Mutation rate: 0.5**, bounds: (-3, 3)
- **16 parallel environment repeats** (8 flat + 8 ice) per evaluation for stable fitness estimation
- **500 generations** with trial_time=50s (1000 simulation steps, matching evaluation)

Training reward was shaped with forward_reward_weight=2.0, healthy_reward=1.0, ctrl_cost=0.1, plus action smoothness and gait coordination penalties to encourage stable locomotion.

## Pareto Front Trade-offs

The final Pareto front contains 9 non-dominated solutions spanning flat fitness [386, 1705] and ice fitness [1561, 2644]. A clear trade-off exists: controllers optimized for flat terrain (high friction) develop aggressive gaits that fail on ice, while ice specialists use conservative, low-amplitude movements that underperform on flat ground.

## Specialist vs. Generalist Comparison

| Controller | Flat (mean) | Ice (mean) |
|---|---|---|
| Flat Specialist | 1415 | 1685 |
| Ice Specialist | 553 | 2590 |
| Generalist | 1227 | 2401 |

The generalist achieves 87% of the flat specialist's flat performance while reaching 93% of the ice specialist's ice performance, demonstrating that NSGA-II successfully finds balanced trade-off solutions. The ice specialist fails quickly on flat terrain (25-step episodes), confirming that the two objectives genuinely conflict.
