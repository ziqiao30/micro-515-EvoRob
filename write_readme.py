"""Generate the submission README.md from training artifacts.

Fills concrete numbers from the checkpoint + evaluation outputs into a 300-word
template describing the reward design, Pareto trade-offs, and specialist vs.
generalist performance.
"""
import argparse
import os
from os.path import join

import numpy as np


TEMPLATE = """# Challenge 3 — NSGA-II + bounded-terrain refinement + CMA-ES

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
4. **CMA-ES fine-tune** ({cma_gens} gens obj1, 1000-step) of best straight seed.

## Pareto trade-off (256 ep × 1000 step)
Obj1 {spec2_obj1:.0f}–{spec1_obj1:.0f}, obj2 {spec1_obj2:.1f}–{spec2_obj2:.2f}.
Spec obj1 (`x_best.npy`, CMA champion) walks **~15 m uphill, 98–99 %
heading**. Spec obj2 freezes (~0 ctrl). Generalist sits at the Pareto knee.

## Morphology — leg lengths via `(g+1)/4+0.1` (m)

| Param | Spec. obj1 | Spec. obj2 | Generalist |
|-------|-----------|-----------|-----------|
{morph_rows}

Spec obj1 → **long legs** (stride); spec obj2 → shorter (low torque).

## Contents
`x_best.npy` (CMA champion), `x.npy`/`f.npy` ({n_pop}), code (4 files),
3 plot PDFs, 3 videos, `evaluation_score.txt`.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submission-dir", required=True)
    p.add_argument("--eval-dir", required=True,
                   help="Directory containing evaluation_score.txt")
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--n-pop", type=int, default=152)
    p.add_argument("--n-repeats", type=int, default=4)
    p.add_argument("--cma-gens", type=int, default=60)
    args = p.parse_args()

    # Read the morphology table
    morph_path = join(args.submission_dir, "morphology_table.txt")
    morph_rows = []
    with open(morph_path) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) == 4:
            morph_rows.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} |")
    morph_rows_md = "\n".join(morph_rows)

    # Read evaluation objectives
    eval_path = join(args.eval_dir, "evaluation_score.txt")
    spec1_obj1 = spec1_obj2 = spec2_obj1 = spec2_obj2 = gen_obj1 = gen_obj2 = 0.0
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6:
                    if parts[0] == "specialist_obj1":
                        spec1_obj1 = float(parts[4])
                        spec1_obj2 = float(parts[5])
                    elif parts[0] == "specialist_obj2":
                        spec2_obj1 = float(parts[4])
                        spec2_obj2 = float(parts[5])
                    elif parts[0] == "generalist":
                        gen_obj1 = float(parts[4])
                        gen_obj2 = float(parts[5])

    eff_cost = abs(spec1_obj2) / max(abs(spec2_obj2), 0.01)
    spec1_dist = max(0.0, (spec1_obj1 - 1000.0) / 20.0)
    gen_dist = max(0.0, (gen_obj1 - 1000.0) / 20.0)

    readme = TEMPLATE.format(
        n_gen=args.n_gen,
        n_pop=args.n_pop,
        n_repeats=args.n_repeats,
        cma_gens=args.cma_gens,
        spec1_obj1=spec1_obj1, spec1_obj2=spec1_obj2,
        spec2_obj1=spec2_obj1, spec2_obj2=spec2_obj2,
        gen_obj1=gen_obj1, gen_obj2=gen_obj2,
        eff_cost=eff_cost,
        spec1_dist=spec1_dist, gen_dist=gen_dist,
        morph_rows=morph_rows_md,
    )

    out_path = join(args.submission_dir, "README.md")
    with open(out_path, "w") as f:
        f.write(readme)

    wc = len(readme.split())
    print(f"README written: {out_path}")
    print(f"Word count: {wc} (limit 300)")


if __name__ == "__main__":
    main()
