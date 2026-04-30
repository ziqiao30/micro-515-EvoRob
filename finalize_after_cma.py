"""Finalize submission after CMA-ES v2 with fixed reward.

Steps:
1. Re-evaluate ALL individuals on fixed-reward, 750-step config → new f.npy.
2. Append CMA-v2 x_best to population (re-eval at same config).
3. Verify straightness of new champion.
4. Save updated x.npy/f.npy to multi/44.
5. Caller (package_submission.py) generates videos + README + zip afterwards.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import gymnasium as gym

from Challenge3 import AntWorld, evaluate_population_parallel

POP_DIR = "results/AntHill-v0/multi/44"
CMA_DIR = "results/AntHill-v0/cma_finetune_v2"


def run_single_and_report(world, x_ind, label):
    world.update_robot_xml(x_ind)
    env = gym.make("AntHill-v0", robot_path=world.world_file, max_episode_steps=1000)
    world.controller.reset_controller(1)
    obs, _ = env.reset(seed=0)
    x0, y0 = None, None
    for _ in range(1000):
        a = world.controller.get_action(obs)
        if a.ndim > 1:
            a = a.squeeze(0)
        obs, _, term, trunc, info = env.step(a)
        if x0 is None:
            x0, y0 = info["x_position"], info["y_position"]
        x1, y1 = info["x_position"], info["y_position"]
        if term or trunc:
            break
    env.close()
    dx, dy = x1 - x0, y1 - y0
    straight = abs(dx) / max(np.sqrt(dx * dx + dy * dy), 0.1)
    print(f"{label:>28s}:  dx={dx:+6.1f} m  dy={dy:+6.1f} m  straightness={100*straight:.0f}%")
    return dx, dy


def main():
    pop_old = np.load(os.path.join(POP_DIR, "x_orig.npy"))  # 150 curriculum inds (original, before prior CMA append)
    print(f"Loaded original curriculum pop: {pop_old.shape}")

    cma_v1 = np.load(os.path.join(POP_DIR, "x_best.npy"))   # old cma super-ant (idx 150 that was appended)
    cma_v2 = np.load(os.path.join(CMA_DIR, "x_best.npy"))
    print(f"Loaded CMA v1 x_best: {cma_v1.shape}")
    print(f"Loaded CMA v2 x_best: {cma_v2.shape}")

    # Build new pop: 150 curriculum + CMA-v1 + CMA-v2 = 152 individuals
    pop = np.vstack([pop_old, cma_v1.reshape(1, -1), cma_v2.reshape(1, -1)])
    print(f"New pop shape: {pop.shape}")

    # Evaluate everyone with fixed reward (ant_hill.py is already reverted)
    # Use same config as curriculum Phase 3: 750 steps, 4 reps
    print("\nRe-evaluating entire pop with FIXED reward (750 steps, 4 reps)...")
    fit = evaluate_population_parallel(pop, n_repeats=4, n_steps=750, max_workers=16)
    print(f"New obj1: min={fit[:,0].min():.1f}, mean={fit[:,0].mean():.1f}, max={fit[:,0].max():.1f}")

    cma_v1_idx = 150
    cma_v2_idx = 151
    print(f"\nCMA v1 (old idx 150): obj1 = {fit[cma_v1_idx,0]:.1f}")
    print(f"CMA v2 (new idx 151): obj1 = {fit[cma_v2_idx,0]:.1f}")

    # Top 5 by new obj1
    top5 = np.argsort(-fit[:, 0])[:5]
    print("\nTop 5 by FIXED obj1:")
    for idx in top5:
        print(f"  idx {idx:>3d}  obj1={fit[idx,0]:>7.1f}  obj2={fit[idx,1]:>7.2f}")

    # Check straightness of top 2
    world = AntWorld()
    print("\nStraightness check (single episode):")
    for idx in top5[:3]:
        run_single_and_report(world, pop[idx], f"idx {idx} (obj1={fit[idx,0]:.0f})")

    # Save
    np.save(os.path.join(POP_DIR, "x.npy"), pop)
    np.save(os.path.join(POP_DIR, "f.npy"), fit)
    np.save(os.path.join(POP_DIR, "x_best.npy"), pop[top5[0]])
    np.save(os.path.join(POP_DIR, "f_best.npy"), fit[top5[0]])
    print(f"\nSaved updated pop/fit to {POP_DIR}")
    print(f"x_best.npy → idx {top5[0]}, obj1 = {fit[top5[0], 0]:.1f}")


if __name__ == "__main__":
    main()
