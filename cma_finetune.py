"""CMA-ES fine-tuning of a NSGA-II seed genotype.

Starts from a specified checkpoint individual (default: idx 61), runs CMA-ES
to maximize mean obj1 (reward_forward + healthy_reward) at 1000 steps — i.e.
the distance-proxy metric that matches the TA's evaluation. Saves a new
checkpoint folder alongside the NSGA-II one.
"""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import time
from concurrent.futures import ProcessPoolExecutor
from os.path import join

import cma
import numpy as np

from Challenge3 import AntWorld
from evorob.utils.filesys import get_project_root


ROOT_DIR = get_project_root()

_EVAL_WORLD = None


def _worker_init():
    global _EVAL_WORLD
    os.environ.setdefault("MUJOCO_GL", "egl")
    _EVAL_WORLD = AntWorld()


def _eval_one(args):
    """Evaluate a genotype — return MEAN obj1 over n_repeats episodes."""
    genotype, n_repeats, n_steps = args
    global _EVAL_WORLD
    if _EVAL_WORLD is None:
        _worker_init()
    _, mo = _EVAL_WORLD.evaluate_individual(
        genotype, n_repeats=n_repeats, n_steps=n_steps,
    )
    # Return negative obj1 (CMA-ES minimises)
    return -float(mo[0])


def evaluate_population(pop, n_repeats, n_steps, max_workers):
    args_list = [(g, n_repeats, n_steps) for g in pop]
    out = np.empty(len(pop), dtype=float)
    with ProcessPoolExecutor(max_workers=max_workers,
                             initializer=_worker_init) as pool:
        for i, v in enumerate(pool.map(_eval_one, args_list)):
            out[i] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint",
                   default=join(ROOT_DIR, "results", "AntHill-v0", "multi", "199"))
    p.add_argument("--seed-idx", type=int, default=61)
    p.add_argument("--output-dir",
                   default=join(ROOT_DIR, "results", "AntHill-v0", "cma_finetune"))
    p.add_argument("--sigma0", type=float, default=0.15)
    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--generations", type=int, default=80)
    p.add_argument("--n-repeats", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--bounds", type=float, nargs=2, default=(-1, 1))
    args = p.parse_args()

    # Load seed
    x_all = np.load(join(args.seed_checkpoint, "x.npy"))
    f_all = np.load(join(args.seed_checkpoint, "f.npy"))
    x0 = x_all[args.seed_idx].copy()
    n_params = len(x0)
    print(f"Seed idx {args.seed_idx}, training f = {f_all[args.seed_idx]}")
    print(f"n_params = {n_params}, sigma0 = {args.sigma0}")

    os.makedirs(args.output_dir, exist_ok=True)

    # CMA-ES setup
    es = cma.CMAEvolutionStrategy(
        x0.tolist(), args.sigma0,
        {
            "popsize": args.popsize,
            "bounds": [float(args.bounds[0]), float(args.bounds[1])],
            "verbose": 1,
            "seed": 1,
        },
    )

    best_x, best_f = x0.copy(), -np.inf
    history = []
    t0 = time.time()
    for gen in range(args.generations):
        pop = np.asarray(es.ask())
        neg_obj1 = evaluate_population(pop, args.n_repeats, args.n_steps, args.max_workers)
        es.tell(pop.tolist(), neg_obj1.tolist())
        obj1 = -neg_obj1
        g_best_idx = int(np.argmax(obj1))
        g_best = obj1[g_best_idx]
        if g_best > best_f:
            best_f = float(g_best)
            best_x = pop[g_best_idx].copy()
        mean_fit = float(np.mean(obj1))
        history.append({"gen": gen, "best": float(g_best), "mean": mean_fit,
                         "best_so_far": best_f})
        dt = time.time() - t0
        print(f"Gen {gen:3d}  best={g_best:8.1f}  mean={mean_fit:8.1f}  "
              f"best_so_far={best_f:8.1f}  sigma={es.sigma:.3f}  "
              f"elapsed={dt/60:.1f}min")

        # Save checkpoint every gen
        gen_dir = join(args.output_dir, str(gen))
        os.makedirs(gen_dir, exist_ok=True)
        np.save(join(gen_dir, "x.npy"), pop)
        np.save(join(gen_dir, "f.npy"), obj1.reshape(-1, 1))  # 1-obj
        np.save(join(gen_dir, "x_best.npy"), best_x)
        np.save(join(gen_dir, "f_best.npy"), np.array([best_f]))

    # Final save
    np.save(join(args.output_dir, "x_best.npy"), best_x)
    np.save(join(args.output_dir, "f_best.npy"), np.array([best_f]))
    np.save(join(args.output_dir, "history.npy"), np.array(
        [(h["gen"], h["best"], h["mean"], h["best_so_far"]) for h in history]
    ))
    print(f"\nFinal best obj1 = {best_f:.1f}, seed obj1 was {f_all[args.seed_idx, 0]:.1f}")


if __name__ == "__main__":
    main()
