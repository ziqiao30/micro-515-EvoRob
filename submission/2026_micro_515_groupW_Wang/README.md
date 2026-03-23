# Challenge 1 - MICRO-515 EvoRob
**Team:** groupW | **Name:** Ziqiao Wang

---

## Algorithm: CMA-ES

We used **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) from the `pycma` library as the evolutionary algorithm. CMA-ES adapts the covariance matrix of a multivariate Gaussian distribution to efficiently search the parameter space, making it well-suited for continuous black-box optimization.

**Hyperparameters:**
- Population size: 400
- Initial step size (sigma): 1.0
- Evaluated across 3 random seeds, best result selected

To accelerate training, fitness evaluations were parallelized across 48 CPU cores using `ProcessPoolExecutor`, with each worker maintaining a persistent MuJoCo environment instance.

---

## Controller: Oscillatory Controller

We selected an **oscillatory (sinusoidal) controller** over a neural network controller due to its strong inductive bias for periodic locomotion on flat terrain. Each of the 8 joints is driven by an independent parameterized sine wave:

```
action_i = amplitude_i × sin(2π × frequency_i × t + phase_i) + offset_i
```

This gives **32 parameters** total (4 per joint). The `offset` term was added to allow asymmetric gaits, enabling more efficient forward locomotion. The controller does not use observations, relying entirely on time-based periodic signals.

---

## Environment Design

The training environment is based on `AntFlatEnvironment`, matching the standard `Ant-v5` reward:

- **Forward reward:** `1.0 × v_x`
- **Survival reward:** `1.0` per step
- **Control cost:** `-0.5 × Σ(action²)`

The termination condition resets the episode when torso height falls outside `[0.26, 1.0]`. Training episodes run for 50 seconds (1000 steps) to match the evaluation protocol.
