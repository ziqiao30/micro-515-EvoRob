"""Post-training analysis for Challenge 3.

Loads the final NSGA-II checkpoint, plots the Pareto front, builds a
morphology-comparison plot of the specialists + generalist, and copies the
chosen x_best.npy into the submission folder.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import shutil
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

from evorob.utils.filesys import get_last_checkpoint_dir, get_project_root

ROOT_DIR = get_project_root()
BODY_PARAM_NAMES = [
    "FL_upper", "FL_lower",
    "FR_upper", "FR_lower",
    "BL_upper", "BL_lower",
    "BR_upper", "BR_lower",
]


def _try_load(dirs, filename):
    for d in dirs:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return np.load(p, allow_pickle=True)
    return None


def decode_body_params(genotype, n_weights):
    """Invert the geno2pheno body mapping: (genotype[i] + 1) / 4 + 0.1."""
    body_genes = genotype[n_weights:]
    return (body_genes + 1) / 4 + 0.1


def load_checkpoint(checkpoint_dir):
    last_gen = get_last_checkpoint_dir(checkpoint_dir)
    candidates = [last_gen] if last_gen else []
    candidates.append(checkpoint_dir)
    population = _try_load(candidates, "x.npy")
    fitness    = _try_load(candidates, "f.npy")
    x_best     = _try_load(candidates, "x_best.npy")
    full_f     = _try_load([checkpoint_dir], "full_f.npy")
    full_x     = _try_load([checkpoint_dir], "full_x.npy")
    assert population is not None and fitness is not None, (
        f"Missing x.npy / f.npy in {checkpoint_dir}"
    )
    return {
        "population": np.asarray(population),
        "fitness":    np.asarray(fitness),
        "x_best":     np.asarray(x_best) if x_best is not None else None,
        "full_f":     np.asarray(full_f) if full_f is not None else None,
        "full_x":     np.asarray(full_x) if full_x is not None else None,
        "last_gen":   last_gen,
        "checkpoint_dir": checkpoint_dir,
    }


def identify_individuals(population, fitness):
    """Return (spec1_idx, spec2_idx, generalist_idx) from the final pop.

    Generalist = Pareto-knee point, normalised OVER THE PARETO FRONT ONLY.
    Normalising over the whole pop lets slow/fallen individuals skew the
    scale, which pushes the "knee" toward a specialist extreme (we saw
    that with the old code: generalist was basically spec_obj2 with slight
    forward motion). Front-only gives a truly middle-of-front ant.
    """
    spec1_idx = int(np.argmax(fitness[:, 0]))
    spec2_idx = int(np.argmax(fitness[:, 1]))

    f = fitness.astype(float).copy()
    n = len(f)
    nd = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if np.all(f[j] >= f[i]) and np.any(f[j] > f[i]):
                nd[i] = False; break
    front = np.where(nd)[0]
    fm, fM = f[front].min(axis=0), f[front].max(axis=0)
    span = np.where(fM - fm > 1e-9, fM - fm, 1.0)
    f_norm_front = (f[front] - fm) / span
    dists = np.linalg.norm(1.0 - f_norm_front, axis=1)
    knee_local = int(np.argmin(dists))
    gen_idx = int(front[knee_local])
    if gen_idx in (spec1_idx, spec2_idx):
        order = np.argsort(dists)
        for cand_local in order:
            cand = int(front[cand_local])
            if cand not in (spec1_idx, spec2_idx):
                gen_idx = cand
                break
    return spec1_idx, spec2_idx, gen_idx


def plot_pareto_front(fitness, spec1, spec2, gen, out_path):
    """Scatter of all individuals with Pareto-front highlighted."""
    # Identify non-dominated (front-0) points for emphasis
    n = len(fitness)
    is_front0 = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (np.all(fitness[j] >= fitness[i])
                    and np.any(fitness[j] > fitness[i])):
                is_front0[i] = False
                break

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        fitness[~is_front0, 0], fitness[~is_front0, 1],
        c="lightgray", s=22, label="dominated", alpha=0.7,
    )
    front_sorted = np.argsort(fitness[is_front0, 0])
    ax.plot(
        fitness[is_front0, 0][front_sorted],
        fitness[is_front0, 1][front_sorted],
        c="steelblue", lw=1.2, marker="o", ms=5, label="Pareto front",
    )
    ax.scatter(
        fitness[spec1, 0], fitness[spec1, 1],
        c="tab:red", s=130, marker="*", label="Specialist obj1 (forward+healthy)",
        edgecolor="black", linewidth=0.6, zorder=6,
    )
    ax.scatter(
        fitness[spec2, 0], fitness[spec2, 1],
        c="tab:green", s=130, marker="*", label="Specialist obj2 (efficiency)",
        edgecolor="black", linewidth=0.6, zorder=6,
    )
    ax.scatter(
        fitness[gen, 0], fitness[gen, 1],
        c="tab:purple", s=130, marker="D", label="Generalist (best sum)",
        edgecolor="black", linewidth=0.6, zorder=6,
    )
    ax.set_xlabel("Objective 1: reward_forward + healthy_reward")
    ax.set_ylabel("Objective 2: -ctrl_cost")
    ax.set_title("Challenge 3 — Final NSGA-II Pareto Front (hilly terrain)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Pareto plot saved: {out_path}")


def plot_morphology_comparison(population, n_weights, spec1, spec2, gen, out_path):
    """Grouped bar plot of leg lengths for the three chosen individuals."""
    labels = ["Spec. obj1\n(forward)", "Spec. obj2\n(efficiency)", "Generalist"]
    colors = ["tab:red", "tab:green", "tab:purple"]
    body_vals = np.array([
        decode_body_params(population[spec1], n_weights),
        decode_body_params(population[spec2], n_weights),
        decode_body_params(population[gen], n_weights),
    ])  # (3, 8)

    x = np.arange(len(BODY_PARAM_NAMES))
    width = 0.26

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i in range(3):
        ax.bar(x + (i - 1) * width, body_vals[i], width,
               label=labels[i], color=colors[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(BODY_PARAM_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Leg segment length (m)")
    ax.set_title("Challenge 3 — Morphology comparison (specialists vs. generalist)")
    ax.axhline(0.1, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax.axhline(0.35, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Morphology plot saved: {out_path}")

    return body_vals


def plot_training_curve(full_f, out_path):
    """Best-so-far of each objective across generations.

    If a NSGA-200 baseline full_f exists (from our first training run), we
    concatenate it with the refinement run so the plot tells the whole
    pipeline story: 200 gens of baseline learning + curriculum/y-penalty
    refinement. Otherwise fall back to the refinement-only curve.
    """
    # Try to load the baseline curve
    baseline_path = os.path.join(ROOT_DIR, "results", "AntHill-v0",
                                  "multi_nsga200_baseline", "full_f.npy")
    baseline_f = None
    if os.path.exists(baseline_path):
        try:
            baseline_f = np.load(baseline_path)
            print(f"  Loaded baseline training history: {baseline_f.shape}")
        except Exception as e:
            print(f"  Baseline training history skipped ({e})")

    if full_f is None and baseline_f is None:
        print("  Skipping training curve (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    def _mean_best(ff):
        return np.mean(ff[..., 0], axis=1), np.max(ff[..., 0], axis=1), \
               np.mean(ff[..., 1], axis=1), np.max(ff[..., 1], axis=1)

    gen_offset = 0
    if baseline_f is not None:
        m1, b1, m2, b2 = _mean_best(baseline_f)
        xs = np.arange(len(m1))
        axes[0].plot(xs, b1, label="baseline — best", c="tab:red")
        axes[0].plot(xs, m1, label="baseline — mean", c="tab:orange", alpha=0.6)
        axes[1].plot(xs, b2, label="baseline — best", c="tab:green")
        axes[1].plot(xs, m2, label="baseline — mean", c="tab:olive", alpha=0.6)
        gen_offset = len(m1)
        for ax in axes:
            ax.axvline(gen_offset, color="gray", ls="--", alpha=0.5, lw=0.8)
            ax.text(gen_offset + 1, ax.get_ylim()[0] + 0.05,
                    "y-penalty\nrefinement →", fontsize=7, color="gray")

    if full_f is not None and len(full_f) > 0:
        m1, b1, m2, b2 = _mean_best(full_f)
        xs = gen_offset + np.arange(len(m1))
        axes[0].plot(xs, b1, label="refine — best", c="tab:purple")
        axes[0].plot(xs, m1, label="refine — mean", c="tab:pink", alpha=0.6)
        axes[1].plot(xs, b2, label="refine — best", c="tab:cyan")
        axes[1].plot(xs, m2, label="refine — mean", c="tab:blue", alpha=0.6)

    axes[0].set_title("Obj1: reward_forward + healthy_reward")
    axes[0].set_xlabel("Generation"); axes[0].set_ylabel("fitness")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7, loc="lower right")

    axes[1].set_title("Obj2: -ctrl_cost")
    axes[1].set_xlabel("Generation"); axes[1].set_ylabel("fitness")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7, loc="lower right")

    fig.suptitle("Challenge 3 — NSGA-II training history (baseline + y-penalty refinement)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Training curve saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=join(ROOT_DIR, "results", "AntHill-v0", "multi"))
    parser.add_argument("--output-dir",     default=join(ROOT_DIR, "submission_challenge3"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data = load_checkpoint(args.checkpoint_dir)
    pop, fit = data["population"], data["fitness"]
    print(f"Loaded: pop={pop.shape}, fit={fit.shape}, last_gen={data['last_gen']}")

    spec1, spec2, gen = identify_individuals(pop, fit)
    print(f"  spec1 idx={spec1}, f={fit[spec1]}")
    print(f"  spec2 idx={spec2}, f={fit[spec2]}")
    print(f"  gener idx={gen},  f={fit[gen]}")

    plot_pareto_front(
        fit, spec1, spec2, gen,
        out_path=join(args.output_dir, "pareto_front.pdf"),
    )

    # Use the controller params from Challenge3 (AntWorld default)
    from Challenge3 import AntWorld
    world = AntWorld()
    n_weights = world.n_weights
    body_vals = plot_morphology_comparison(
        pop, n_weights, spec1, spec2, gen,
        out_path=join(args.output_dir, "morphology_comparison.pdf"),
    )

    plot_training_curve(
        data["full_f"],
        out_path=join(args.output_dir, "training_curve.pdf"),
    )

    # Save the generalist as x_best.npy in the submission folder (overrides
    # any checkpoint file — the generalist is what we submit as the best).
    x_best_out = join(args.output_dir, "x_best.npy")
    np.save(x_best_out, pop[gen])
    print(f"  Saved submission x_best.npy (generalist): {x_best_out}")

    # Also save both specialists for completeness
    np.save(join(args.output_dir, "x_best_spec_forward.npy"), pop[spec1])
    np.save(join(args.output_dir, "x_best_spec_efficiency.npy"), pop[spec2])

    # Include the final population + fitness so the TA's evaluate_checkpoint()
    # can automatically find specialists/generalist if they re-run it.
    np.save(join(args.output_dir, "x.npy"), pop)
    np.save(join(args.output_dir, "f.npy"), fit)
    print(f"  Saved final population (x.npy {pop.shape}) and fitness (f.npy {fit.shape})")

    # Copy code files
    for src in [
        "Challenge3.py",
        "evorob/world/envs/ant_hill.py",
        "evorob/world/robot/controllers/mlp.py",
        "evorob/algorithms/nsga.py",
    ]:
        src_abs = join(ROOT_DIR, src)
        dst     = join(args.output_dir, os.path.basename(src))
        shutil.copy2(src_abs, dst)
        print(f"  Copied {src} -> {dst}")

    # Save body-param table as text
    table_path = join(args.output_dir, "morphology_table.txt")
    with open(table_path, "w") as f:
        f.write(f"{'Param':<12s} {'Spec obj1':>12s} {'Spec obj2':>12s} {'Generalist':>12s}\n")
        f.write("-" * 52 + "\n")
        for i, name in enumerate(BODY_PARAM_NAMES):
            f.write(f"{name:<12s} {body_vals[0, i]:12.3f} {body_vals[1, i]:12.3f} {body_vals[2, i]:12.3f}\n")
    print(f"  Morphology table saved: {table_path}")


if __name__ == "__main__":
    main()
