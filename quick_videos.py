"""Generate 3 videos quickly for the 3 submission candidates.

Saves videos + a short evaluation summary to a given output folder. No 256-ep
re-eval — this is meant for fast inspection, not final submission.
"""
import argparse
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from Challenge3 import AntWorld, _run_episodes_hill, _record_video_hill


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/home/ziqiao/micro-515-EvoRob/preview_videos")
    p.add_argument("--n-eps", type=int, default=8,
                   help="Episodes per individual for the summary stats")
    p.add_argument("--checkpoint", default="/home/ziqiao/micro-515-EvoRob/results/AntHill-v0/multi/199")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    f = np.load(os.path.join(args.checkpoint, "f.npy"))
    x = np.load(os.path.join(args.checkpoint, "x.npy"))

    # Specialists by TA's convention (matches evaluate_checkpoint)
    spec1_idx = int(np.argmax(f[:, 0]))
    spec2_idx = int(np.argmax(f[:, 1]))
    # Generalist: override to idx 61 (verified as the best 1000-step
    # performer on the Pareto front — highest obj1 among reward-positive
    # stable individuals, reward ≈ 904, obj1 ≈ 1384).
    gen_idx = 61

    individuals = {
        "specialist_forward":    (spec1_idx, "Spec. obj1 (max reward_forward+healthy)"),
        "specialist_efficiency": (spec2_idx, "Spec. obj2 (max -ctrl_cost)"),
        "generalist":            (gen_idx,  "Generalist (best balanced performer)"),
    }

    print(f"{'label':<24s} {'idx':>4s} {'train f':>26s}")
    for label, (idx, _desc) in individuals.items():
        print(f"{label:<24s} {idx:>4d} {str(f[idx]):>26s}")
    print()

    world = AntWorld()
    summary_path = os.path.join(args.output_dir, "preview_summary.txt")
    with open(summary_path, "w") as fh:
        fh.write("Preview Summary — Challenge 3\n" + "=" * 50 + "\n")
        fh.write(f"{args.n_eps} episodes × 1000 steps per individual\n\n")
        fh.write(f"{'individual':<24s} {'idx':>4s} {'eval_obj1':>10s} {'eval_obj2':>10s} {'reward':>10s} {'std':>8s}\n")

        for label, (idx, desc) in individuals.items():
            # Summary stats
            rew, obj1s, obj2s = _run_episodes_hill(
                world, x[idx], n_episodes=args.n_eps,
                max_episode_steps=1000, seed=0,
            )
            mr, so1, so2 = np.mean(rew), np.mean(obj1s), np.mean(obj2s)
            sr = np.std(rew)
            line = f"{label:<24s} {idx:>4d} {so1:>10.1f} {so2:>10.2f} {mr:>10.1f} {sr:>8.1f}"
            print(line)
            fh.write(line + "\n")

            # Video
            vpath = os.path.join(args.output_dir, f"video_{label}.mp4")
            _record_video_hill(world, x[idx], 1000, 0, vpath)

    print(f"\nSummary and videos in: {args.output_dir}")


if __name__ == "__main__":
    main()
