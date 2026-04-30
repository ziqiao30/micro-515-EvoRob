"""End-to-end post-training packaging for Challenge 3.

Runs: analyze_challenge3 → Challenge3.evaluate_checkpoint → write_readme →
zips the submission folder following the course naming convention.
"""
import argparse
import os
import shutil
import subprocess
import zipfile
from os.path import join

from evorob.utils.filesys import get_project_root


ROOT_DIR = get_project_root()


def run(cmd, env=None):
    print(f"\n$ {' '.join(cmd)}")
    r = subprocess.run(cmd, env=env or os.environ.copy())
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {' '.join(cmd)}")


def zip_dir(src_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for name in files:
                abs_path = os.path.join(root, name)
                rel_path = os.path.relpath(abs_path, os.path.dirname(src_dir))
                zf.write(abs_path, rel_path)
    print(f"Zip written: {zip_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default=join(ROOT_DIR, "results", "AntHill-v0", "multi"))
    p.add_argument("--n-episodes", type=int, default=256,
                   help="Episodes per individual in TA-style eval")
    p.add_argument("--n-gen", type=int, default=200)
    p.add_argument("--n-pop", type=int, default=150)
    p.add_argument("--n-repeats", type=int, default=4)
    p.add_argument("--cma-gens", type=int, default=60)
    p.add_argument("--sciper", default="322816")
    p.add_argument("--team",   default="groupW")
    p.add_argument("--name",   default="Wang")
    args = p.parse_args()

    folder_name = f"2026_micro_515_{args.sciper}_{args.team}_{args.name}_challenge3"
    submission_dir = join(ROOT_DIR, folder_name)
    eval_dir = join(submission_dir, "evaluation_output")

    # Wipe old output to start fresh
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Analyze: Pareto/morphology/training plots + code copy + x_best.npy
    run([
        "python", "analyze_challenge3.py",
        "--checkpoint-dir", args.checkpoint_dir,
        "--output-dir", submission_dir,
    ])

    # 2. TA-style evaluation (videos + score file + per-ind stats)
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    run([
        "python", "-c",
        f"from Challenge3 import evaluate_checkpoint;"
        f" evaluate_checkpoint("
        f"checkpoint_dir={args.checkpoint_dir!r},"
        f" output_dir={eval_dir!r},"
        f" n_episodes={args.n_episodes})",
    ], env=env)

    # Copy videos to submission root
    for name in ("evaluation_specialist_forward.mp4",
                 "evaluation_specialist_efficiency.mp4",
                 "evaluation_generalist.mp4"):
        src = join(eval_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, join(submission_dir, name))
    # Copy the score file too
    sp = join(eval_dir, "evaluation_score.txt")
    if os.path.exists(sp):
        shutil.copy2(sp, join(submission_dir, "evaluation_score.txt"))

    # 3. Generate README.md
    run([
        "python", "write_readme.py",
        "--submission-dir", submission_dir,
        "--eval-dir", eval_dir,
        "--n-gen", str(args.n_gen),
        "--n-pop", str(args.n_pop),
        "--n-repeats", str(args.n_repeats),
    ])

    # 4. Clean the eval_dir out (we already copied what we need)
    shutil.rmtree(eval_dir, ignore_errors=True)

    # 5. List final contents
    print("\nFinal submission contents:")
    for f in sorted(os.listdir(submission_dir)):
        size = os.path.getsize(join(submission_dir, f))
        print(f"  {f:<40s}  {size/1024:.1f} KB")

    # 6. Zip it
    zip_path = join(ROOT_DIR, f"{folder_name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    zip_dir(submission_dir, zip_path)


if __name__ == "__main__":
    main()
