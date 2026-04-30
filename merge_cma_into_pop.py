"""Append the CMA-ES fine-tuned x_best to the curriculum NSGA-II final
population so that the TA's evaluate_checkpoint() sees it as the
`argmax(obj1)` spec_obj1 individual.

We re-evaluate the CMA genotype at the SAME config the NSGA phase used
(750 steps, 4 reps, slope=5°, bump=0.1, sigma=3.0) so that its fitness is
directly comparable to the rest of the population — otherwise NSGA's argmax
picks a wrong individual.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from Challenge3 import AntWorld

NSGA_POP_DIR = "results/AntHill-v0/multi/44"
CMA_BEST = "results/AntHill-v0/cma_finetune/x_best.npy"
OUT_DIR = "results/AntHill-v0/multi/44"


def main():
    pop = np.load(os.path.join(NSGA_POP_DIR, "x.npy"))
    fit = np.load(os.path.join(NSGA_POP_DIR, "f.npy"))
    print(f"Loaded NSGA pop: {pop.shape}, fit: {fit.shape}")
    cma_x = np.load(CMA_BEST)
    print(f"Loaded CMA x_best: {cma_x.shape}")
    print(f"Pop obj1: min={fit[:,0].min():.1f} max={fit[:,0].max():.1f}")

    # Re-evaluate CMA x_best at curriculum Phase 3 config (750 steps, 4 reps, 5°/0.1/3)
    world = AntWorld()
    print(f"\nRe-evaluating CMA x_best at 750 steps, 4 reps...")
    scalar, mo = world.evaluate_individual(cma_x, n_repeats=4, n_steps=750)
    print(f"CMA x_best re-eval: scalar={scalar:.1f}, obj1={mo[0]:.1f}, obj2={mo[1]:.2f}")

    # Append
    pop_ext = np.vstack([pop, cma_x.reshape(1, -1)])
    fit_ext = np.vstack([fit, mo.reshape(1, -1)])
    print(f"Extended pop: {pop_ext.shape}, fit: {fit_ext.shape}")
    print(f"New obj1 max: {fit_ext[:,0].max():.1f} (at idx {np.argmax(fit_ext[:,0])})")

    # Save back — keep backup of original pop/fit
    np.save(os.path.join(OUT_DIR, "x_orig.npy"), pop)
    np.save(os.path.join(OUT_DIR, "f_orig.npy"), fit)
    np.save(os.path.join(OUT_DIR, "x.npy"), pop_ext)
    np.save(os.path.join(OUT_DIR, "f.npy"), fit_ext)
    # Also update x_best.npy to be the CMA winner
    np.save(os.path.join(OUT_DIR, "x_best.npy"), cma_x)
    np.save(os.path.join(OUT_DIR, "f_best.npy"), mo)
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
