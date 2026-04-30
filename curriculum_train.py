"""Curriculum-learning NSGA-II for Challenge 3.

Evolves body + MLP controller through a sequence of terrains, starting easy
(gentle slope, no bumps) and ramping up to the challenge target (slope_deg=5°,
bump_scale=0.1, sigma=3.0). Each phase warm-starts from the previous phase's
final population, with the seed fitness RE-evaluated on the new terrain before
NSGA elitism kicks in — otherwise stale cross-terrain fitness would corrupt
the non-dominated sort.

Environment variables EVOROB_SLOPE_DEG / EVOROB_BUMP_SCALE / EVOROB_SIGMA
pick up the per-phase terrain settings; workers forked from each phase's
ProcessPoolExecutor inherit them at fork time.
"""
import argparse
import os
import subprocess
import sys
import time
from os.path import join

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

# Default import of Challenge3 establishes module globals from env vars. For
# curriculum we manipulate env vars BEFORE each phase's worker pool is spawned.
from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_project_root

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"


# Each phase defines terrain params and NSGA settings. Warm-start from
# previous phase's final pop; FINAL phase must match TA target (5° / 0.1 / 3.0)
# so the saved checkpoint is representative of what the TA evaluates.
PHASES = [
    {"name": "easy",   "slope": 1.0, "bump": 0.03, "sigma": 3.0, "n_steps": 500, "gens": 20},
    {"name": "medium", "slope": 3.0, "bump": 0.07, "sigma": 3.0, "n_steps": 500, "gens": 25},
    {"name": "target", "slope": 5.0, "bump": 0.10, "sigma": 3.0, "n_steps": 750, "gens": 45},
]


def run_phase(phase, seed_pop, seed_fit_old, phase_output_dir, shared_cfg):
    """Run one NSGA-II phase. Returns (final_pop, final_fit).

    seed_pop/seed_fit_old may be None → fresh random init.
    """
    name = phase["name"]
    print(f"\n{'=' * 70}\nPhase '{name}':  slope={phase['slope']}°  bump={phase['bump']}"
          f"  sigma={phase['sigma']}  n_steps={phase['n_steps']}  gens={phase['gens']}"
          f"\n{'=' * 70}")

    # Set env vars so the (yet-to-be-spawned) workers and our own AntWorld
    # pick up the right terrain.
    os.environ["EVOROB_SLOPE_DEG"] = str(phase["slope"])
    os.environ["EVOROB_BUMP_SCALE"] = str(phase["bump"])
    os.environ["EVOROB_SIGMA"] = str(phase["sigma"])

    # Re-import Challenge3 in a fresh subprocess PER PHASE, so that the module
    # globals (TERRAIN_*) are re-read. Simpler alternative: import once here
    # and monkey-patch the globals — that works because each phase creates a
    # brand-new ProcessPoolExecutor, so workers fork with up-to-date env vars.
    import importlib
    import Challenge3 as C3
    importlib.reload(C3)
    AntWorld = C3.AntWorld
    evaluate_population_parallel = C3.evaluate_population_parallel

    # Build NSGA-II
    world = AntWorld()
    n_params = world.n_params
    ea = NSGAII(
        population_size=shared_cfg["pop"],
        n_opt_params=n_params,
        n_parents=shared_cfg["pop"],
        num_generations=phase["gens"],
        bounds=shared_cfg["bounds"],
        mutation_prob=shared_cfg["mutation_prob"],
        crossover_prob=shared_cfg["crossover_prob"],
        output_dir=phase_output_dir,
    )
    ea.directory_name = phase_output_dir

    # Warm-start: re-evaluate seed pop on the new terrain, then call tell()
    # once with current_gen=0 to seed elitism properly.
    if seed_pop is not None:
        print(f"  Warm-starting from {len(seed_pop)} seed individuals — re-evaluating on new terrain...")
        t0 = time.time()
        seed_fit_new = evaluate_population_parallel(
            seed_pop,
            n_repeats=shared_cfg["n_repeats"],
            n_steps=phase["n_steps"],
            max_workers=shared_cfg["max_workers"],
        )
        print(f"  Re-eval complete in {time.time()-t0:.1f}s. "
              f"Seed obj1: min={seed_fit_new[:,0].min():.1f} mean={seed_fit_new[:,0].mean():.1f} "
              f"max={seed_fit_new[:,0].max():.1f}")
        # seed tell() at gen 0 so f_best_so_far / x_best_so_far populate and
        # current_population becomes the elitism selection.
        ea.tell(seed_pop, seed_fit_new, save_checkpoint=False)

    # Main loop
    for gen in range(phase["gens"]):
        t0 = time.time()
        pop = ea.ask()
        fit = evaluate_population_parallel(
            pop,
            n_repeats=shared_cfg["n_repeats"],
            n_steps=phase["n_steps"],
            max_workers=shared_cfg["max_workers"],
        )
        ea.tell(pop, fit, save_checkpoint=True)
        print(f"  [{name}] gen {gen+1}/{phase['gens']}  best obj1 so far: "
              f"{ea.f_best_so_far[0]:.1f}  mean obj1: {fit[:,0].mean():.1f}  "
              f"({time.time()-t0:.1f}s)")

    return ea.current_population, ea.fitness


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pop", type=int, default=150)
    p.add_argument("--n-repeats", type=int, default=4)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--mutation-prob", type=float, default=0.5)
    p.add_argument("--crossover-prob", type=float, default=0.9)
    p.add_argument("--bounds", type=float, nargs=2, default=(-1.0, 1.0))
    p.add_argument("--output-root",
                   default=join(ROOT_DIR, "results", ENV_NAME, "curriculum"))
    p.add_argument("--final-dir",
                   default=join(ROOT_DIR, "results", ENV_NAME, "multi"),
                   help="Final phase checkpoint will be mirrored here so "
                        "the TA's evaluate_checkpoint / analyze_challenge3 "
                        "pick it up without changes.")
    p.add_argument("--warm-start",
                   default=None,
                   help="Path to checkpoint folder (e.g. multi_nsga200_baseline/199) "
                        "whose x.npy seeds Phase 1. Skips random init.")
    args = p.parse_args()

    shared_cfg = {
        "pop": args.pop,
        "n_repeats": args.n_repeats,
        "max_workers": args.max_workers,
        "mutation_prob": args.mutation_prob,
        "crossover_prob": args.crossover_prob,
        "bounds": tuple(args.bounds),
    }

    os.makedirs(args.output_root, exist_ok=True)

    seed_pop = None
    seed_fit = None
    if args.warm_start is not None:
        wp = args.warm_start
        if not os.path.isabs(wp):
            wp = join(ROOT_DIR, wp)
        seed_pop = np.load(join(wp, "x.npy"))
        # seed_fit is intentionally left None — Phase 1 will re-eval seed_pop
        # on its own (easy) terrain, which is exactly what we want.
        print(f"Warm-start enabled: seeding Phase 1 with {len(seed_pop)} genotypes from {wp}")
    t0 = time.time()
    for phase in PHASES:
        phase_dir = join(args.output_root, phase["name"])
        os.makedirs(phase_dir, exist_ok=True)
        seed_pop, seed_fit = run_phase(phase, seed_pop, seed_fit, phase_dir, shared_cfg)

    # Copy the FINAL phase checkpoint to the "multi" dir so downstream tools
    # (analyze_challenge3.py, package_submission.py) work unchanged.
    final_phase_dir = join(args.output_root, PHASES[-1]["name"])
    print(f"\nCopying final phase checkpoint to {args.final_dir} ...")
    if os.path.exists(args.final_dir):
        subprocess.run(["rm", "-rf", args.final_dir], check=True)
    subprocess.run(["cp", "-r", final_phase_dir, args.final_dir], check=True)

    elapsed = (time.time() - t0) / 60
    print(f"\nCurriculum training complete in {elapsed:.1f} min.")
    print(f"Final checkpoint: {args.final_dir}")


if __name__ == "__main__":
    main()
