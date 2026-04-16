#!/usr/bin/env python3
"""Overnight parameter sweep for Challenge 2 NSGA-II optimization.

Run all configs sequentially, ~2h each, ~8h total for 4 configs.
Usage: python run_overnight.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import get_last_checkpoint_dir
from evorob.world.ant_multi_world import AntMultiWorld
from evorob.world.ant_world import AntFlatWorld
from evorob.world.envs.ant_flat import AntFlatEnvironment
from evorob.world.robot.controllers.mlp import NeuralNetworkController

# ---------------------------------------------------------------------------
# Configs to sweep
# ---------------------------------------------------------------------------

CONFIGS = [
    {
        # EMA smoothing + action rate penalty + high crossover (biggest expected gains)
        "name": "smooth_high_cx",
        "desc": "EMA smoothing in controller + action rate penalty + CR=0.9 (literature best for 584 dims)",
        "reward": {"forward_reward_weight": 2.0, "healthy_reward_weight": 1.0, "ctrl_cost_weight": 0.1},
        "nsga": {
            "num_generations": 400,
            "population_size": 400,
            "n_parents": 400,
            "n_repeats": 16,
            "mutation_prob": 0.5,
            "crossover_prob": 0.9,
            "bounds": (-3, 3),
        },
    },
    {
        # Higher speed reward with smoothing
        "name": "smooth_fast",
        "desc": "Higher forward reward (3.0) + moderate ctrl cost + CR=0.9",
        "reward": {"forward_reward_weight": 3.0, "healthy_reward_weight": 1.0, "ctrl_cost_weight": 0.1},
        "nsga": {
            "num_generations": 400,
            "population_size": 400,
            "n_parents": 400,
            "n_repeats": 16,
            "mutation_prob": 0.5,
            "crossover_prob": 0.9,
            "bounds": (-3, 3),
        },
    },
    {
        # Very aggressive speed, low penalty
        "name": "aggressive_speed",
        "desc": "forward=5.0, low ctrl=0.05, CR=0.9 -- max speed push",
        "reward": {"forward_reward_weight": 5.0, "healthy_reward_weight": 0.5, "ctrl_cost_weight": 0.05},
        "nsga": {
            "num_generations": 400,
            "population_size": 400,
            "n_parents": 400,
            "n_repeats": 16,
            "mutation_prob": 0.5,
            "crossover_prob": 0.9,
            "bounds": (-3, 3),
        },
    },
    {
        # Higher mutation for diversity + moderate reward
        "name": "diverse_explorer",
        "desc": "mutation=0.7 + CR=0.9 for better Pareto diversity",
        "reward": {"forward_reward_weight": 3.0, "healthy_reward_weight": 1.0, "ctrl_cost_weight": 0.1},
        "nsga": {
            "num_generations": 400,
            "population_size": 400,
            "n_parents": 400,
            "n_repeats": 16,
            "mutation_prob": 0.7,
            "crossover_prob": 0.9,
            "bounds": (-3, 3),
        },
    },
]

# ---------------------------------------------------------------------------
# Worker pool
# ---------------------------------------------------------------------------

_worker_world = None


def _init_worker(n_repeats, reward_kwargs):
    global _worker_world
    _worker_world = AntMultiWorld(
        controller_cls=NeuralNetworkController,
        n_repeats=n_repeats,
        **reward_kwargs,
    )


def _eval_worker(individual):
    return _worker_world.evaluate_individual(individual, trial_time=50)


# ---------------------------------------------------------------------------
# Evaluation (same as Challenge2.py evaluate_checkpoint but with reward kwargs)
# ---------------------------------------------------------------------------

def evaluate_best(checkpoint_dir, output_dir, reward_kwargs):
    """Evaluate x_best on both terrains with 256 episodes."""
    n_episodes = 256
    max_episode_steps = 1000
    seed = 0

    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    x_best_path = os.path.join(last_gen, "x_best.npy") if last_gen else ""
    if not os.path.isfile(x_best_path):
        x_best_path = os.path.join(checkpoint_dir, "x_best.npy")
    if not os.path.isfile(x_best_path):
        print(f"  ERROR: x_best.npy not found in {checkpoint_dir}")
        return None

    genotype = np.load(x_best_path)
    controller = NeuralNetworkController(input_size=27, output_size=8, hidden_size=16)

    terrains = {"flat": "ant_flat_terrain.xml", "ice": "ant_ice_terrain.xml"}
    results = {}

    for terrain_name, robot_path in terrains.items():
        env = AntFlatEnvironment(robot_path=robot_path, **reward_kwargs)
        controller.geno2pheno(genotype)
        rng = np.random.default_rng(seed)
        episode_rewards = []

        for _ in range(n_episodes):
            ep_seed = int(rng.integers(0, 2**31))
            obs, _ = env.reset(seed=ep_seed)
            controller.reset_controller(batch_size=1)
            total_reward = 0.0
            for _ in range(max_episode_steps):
                action = controller.get_action(obs)
                if action.ndim > 1:
                    action = action.squeeze(0)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            episode_rewards.append(total_reward)

        env.close()
        results[terrain_name] = {
            "mean": float(np.mean(episode_rewards)),
            "std": float(np.std(episode_rewards)),
            "best": float(np.max(episode_rewards)),
            "worst": float(np.min(episode_rewards)),
            "median": float(np.median(episode_rewards)),
        }

    return results


# ---------------------------------------------------------------------------
# Main training loop for a single config
# ---------------------------------------------------------------------------

def run_single_config(config, seed=42):
    """Run NSGA-II with a single config. Returns results dict."""
    name = config["name"]
    reward_kwargs = config["reward"]
    nsga_params = config["nsga"]

    np.random.seed(seed)

    n_repeats = nsga_params["n_repeats"]
    pop_size = nsga_params["population_size"]
    n_gens = nsga_params["num_generations"]

    # Setup checkpoint dir
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(f"results/{dt_str}_{name}_ckpts")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(ckpt_dir / "config.txt", "w") as f:
        f.write(f"Name: {name}\n")
        f.write(f"Description: {config['desc']}\n")
        f.write(f"Seed: {seed}\n\n")
        f.write("Reward:\n")
        for k, v in reward_kwargs.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nNSGA-II:\n")
        for k, v in nsga_params.items():
            f.write(f"  {k}: {v}\n")

    # Determine n_params
    tmp_world = AntMultiWorld(
        controller_cls=NeuralNetworkController,
        n_repeats=2,
        **reward_kwargs,
    )
    n_params = tmp_world.n_params
    tmp_world.close()

    nsga = NSGAII(
        population_size=pop_size,
        n_opt_params=n_params,
        n_parents=nsga_params["n_parents"],
        bounds=nsga_params["bounds"],
        mutation_prob=nsga_params["mutation_prob"],
        crossover_prob=nsga_params["crossover_prob"],
        output_dir=str(ckpt_dir),
    )

    print(f"\n{'='*70}")
    print(f"  CONFIG: {name}")
    print(f"  reward: fwd={reward_kwargs['forward_reward_weight']}, "
          f"healthy={reward_kwargs['healthy_reward_weight']}, "
          f"ctrl={reward_kwargs['ctrl_cost_weight']}")
    print(f"  pop={pop_size}, gens={n_gens}, repeats={n_repeats}, "
          f"mut={nsga_params['mutation_prob']}, cx={nsga_params['crossover_prob']}")
    print(f"{'='*70}\n")

    n_workers = min(48, pop_size)
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(n_repeats, reward_kwargs),
    ) as executor:
        for gen in range(n_gens):
            population = nsga.ask()
            results = list(executor.map(_eval_worker, population))
            multi_fitness = np.array(results)

            save_ckpt = (gen % 20 == 0) or (gen == n_gens - 1)
            nsga.tell(population, multi_fitness, save_checkpoint=save_ckpt)

            if gen % 10 == 0 or gen == n_gens - 1:
                sums = multi_fitness.sum(axis=1)
                print(
                    f"[{name}] Gen {gen:3d}/{n_gens}: "
                    f"flat={multi_fitness[:,0].mean():7.0f} "
                    f"ice={multi_fitness[:,1].mean():7.0f} "
                    f"best_sum={sums.max():7.0f}"
                )

    # Evaluate best
    print(f"\n[{name}] Evaluating x_best (256 episodes per terrain)...")
    eval_results = evaluate_best(str(ckpt_dir), str(ckpt_dir), reward_kwargs)

    if eval_results:
        flat = eval_results["flat"]
        ice = eval_results["ice"]
        summary = (
            f"  FLAT: mean={flat['mean']:.0f} ± {flat['std']:.0f} "
            f"(best={flat['best']:.0f}, worst={flat['worst']:.0f})\n"
            f"  ICE:  mean={ice['mean']:.0f} ± {ice['std']:.0f} "
            f"(best={ice['best']:.0f}, worst={ice['worst']:.0f})"
        )
        print(summary)
        with open(ckpt_dir / "eval_summary.txt", "w") as f:
            f.write(summary + "\n")
    else:
        summary = "  EVALUATION FAILED"
        print(summary)

    return {"name": name, "ckpt_dir": str(ckpt_dir), "eval": eval_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_time = time.time()
    all_results = []

    print(f"Starting overnight sweep: {len(CONFIGS)} configs")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estimated completion: ~{len(CONFIGS) * 2} hours\n")

    for i, config in enumerate(CONFIGS):
        t0 = time.time()
        print(f"\n{'#'*70}")
        print(f"# Config {i+1}/{len(CONFIGS)}: {config['name']}")
        print(f"{'#'*70}")

        result = run_single_config(config)
        all_results.append(result)

        elapsed = (time.time() - t0) / 3600
        total_elapsed = (time.time() - start_time) / 3600
        print(f"\n[{config['name']}] Done in {elapsed:.1f}h (total: {total_elapsed:.1f}h)")

    # Final comparison
    print(f"\n\n{'='*70}")
    print("OVERNIGHT SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {(time.time() - start_time) / 3600:.1f} hours\n")

    print(f"{'Config':<25s} {'FLAT mean':>10s} {'ICE mean':>10s} {'SUM':>10s}")
    print("-" * 60)
    for r in all_results:
        if r["eval"]:
            flat_m = r["eval"]["flat"]["mean"]
            ice_m = r["eval"]["ice"]["mean"]
            print(f"{r['name']:<25s} {flat_m:10.0f} {ice_m:10.0f} {flat_m+ice_m:10.0f}")
        else:
            print(f"{r['name']:<25s} {'FAILED':>10s} {'FAILED':>10s}")

    # Save summary
    with open("results/overnight_summary.txt", "w") as f:
        f.write(f"Sweep completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {(time.time() - start_time) / 3600:.1f} hours\n\n")
        f.write(f"{'Config':<25s} {'FLAT mean':>10s} {'ICE mean':>10s} {'SUM':>10s}\n")
        f.write("-" * 60 + "\n")
        for r in all_results:
            if r["eval"]:
                flat_m = r["eval"]["flat"]["mean"]
                ice_m = r["eval"]["ice"]["mean"]
                f.write(f"{r['name']:<25s} {flat_m:10.0f} {ice_m:10.0f} {flat_m+ice_m:10.0f}\n")
            else:
                f.write(f"{r['name']:<25s} {'FAILED':>10s} {'FAILED':>10s}\n")
        f.write("\nCheckpoint dirs:\n")
        for r in all_results:
            f.write(f"  {r['name']}: {r['ckpt_dir']}\n")

    print(f"\nSummary saved to results/overnight_summary.txt")
