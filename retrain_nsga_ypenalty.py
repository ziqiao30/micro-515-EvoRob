"""Warm-start NSGA-II retrain with the y-penalty reward.

Loads the current pop (multi/44), re-evaluates each individual under the
NEW reward (|y_velocity| penalty already baked into ant_hill.py), and runs
NSGA-II for N generations at target terrain (5°, bump=0.1, 750 steps).

The y-penalty in reward_forward directly punishes sideways drift, so argmax
of obj1 in the new f.npy is a STRAIGHT walker.
"""
import argparse
import os
import time
from os.path import join

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import Challenge3 as C3
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_project_root


ROOT_DIR = get_project_root()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-dir", default=join(ROOT_DIR, "results", "AntHill-v0", "multi", "44"))
    p.add_argument("--output-dir", default=join(ROOT_DIR, "results", "AntHill-v0", "nsga_ypenalty"))
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--n-repeats", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=750)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--mutation-prob", type=float, default=0.5)
    p.add_argument("--crossover-prob", type=float, default=0.9)
    p.add_argument("--bounds", type=float, nargs=2, default=(-1.0, 1.0))
    args = p.parse_args()

    os.environ["EVOROB_SLOPE_DEG"] = "5.0"
    os.environ["EVOROB_BUMP_SCALE"] = "0.1"
    os.environ["EVOROB_SIGMA"] = "3.0"

    # Load seed pop
    seed_pop = np.load(join(args.seed_dir, "x.npy"))
    pop_size = len(seed_pop)
    n_params = seed_pop.shape[1]
    print(f"Seed pop: {seed_pop.shape}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build NSGA-II
    ea = NSGAII(
        population_size=pop_size,
        n_opt_params=n_params,
        n_parents=pop_size,
        num_generations=args.gens,
        bounds=tuple(args.bounds),
        mutation_prob=args.mutation_prob,
        crossover_prob=args.crossover_prob,
        output_dir=args.output_dir,
    )
    ea.directory_name = args.output_dir

    # Re-eval seed pop under NEW reward (y-penalty)
    print(f"Re-evaluating seed pop under y-penalty reward ({args.n_steps} steps, {args.n_repeats} reps)...")
    t0 = time.time()
    seed_fit = C3.evaluate_population_parallel(
        seed_pop,
        n_repeats=args.n_repeats,
        n_steps=args.n_steps,
        max_workers=args.max_workers,
    )
    print(f"Seed re-eval complete in {time.time()-t0:.1f}s")
    print(f"  obj1: min={seed_fit[:,0].min():.1f}, mean={seed_fit[:,0].mean():.1f}, max={seed_fit[:,0].max():.1f}")
    print(f"  obj2: min={seed_fit[:,1].min():.2f}, mean={seed_fit[:,1].mean():.2f}, max={seed_fit[:,1].max():.2f}")

    # Seed NSGA-II with tell() at gen=0 — populates current_population, f_best_so_far
    ea.tell(seed_pop, seed_fit, save_checkpoint=False)

    # Main loop
    for gen in range(args.gens):
        t0 = time.time()
        pop = ea.ask()
        fit = C3.evaluate_population_parallel(
            pop,
            n_repeats=args.n_repeats,
            n_steps=args.n_steps,
            max_workers=args.max_workers,
        )
        ea.tell(pop, fit, save_checkpoint=True)
        print(f"  gen {gen+1}/{args.gens}  best obj1: {ea.f_best_so_far[0]:.1f}  "
              f"mean obj1: {fit[:,0].mean():.1f}  "
              f"pop-max obj1: {fit[:,0].max():.1f}  ({time.time()-t0:.1f}s)")

    # Copy to multi/ so downstream tools pick up the new population without changes
    import subprocess
    multi_dir = join(ROOT_DIR, "results", "AntHill-v0", "multi")
    if os.path.exists(multi_dir):
        subprocess.run(["rm", "-rf", multi_dir], check=True)
    subprocess.run(["cp", "-r", args.output_dir, multi_dir], check=True)
    print(f"\nCopied final checkpoint to {multi_dir}")
    print(f"Final best obj1 = {ea.f_best_so_far[0]:.1f}")


if __name__ == "__main__":
    main()
