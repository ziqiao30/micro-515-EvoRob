import os
os.environ.setdefault("MUJOCO_GL", "egl")

import xml.etree.ElementTree as xml
from concurrent.futures import ProcessPoolExecutor, as_completed
from os.path import join
from tempfile import TemporaryDirectory
from PIL import Image
import scipy.ndimage

import gymnasium as gym
import imageio
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange

from evorob.algorithms.nsga import NSGAII
from evorob.utils.filesys import (
    get_distinct_filename,
    get_last_checkpoint_dir,
    get_project_root,
)
from evorob.world.base import World
from evorob.world.robot.controllers.mlp import NeuralNetworkController
from evorob.world.robot.controllers.so2 import SO2Controller
from evorob.world.robot.controllers.mlp_hebbian import HebbianController
from evorob.world.robot.morphology.ant_custom_robot import AntRobot

"""
    Morphology and Controller optimisation: Ant Hill
"""

ROOT_DIR = get_project_root()
ENV_NAME = "AntHill-v0"

# Terrain parameters used everywhere (training + TA evaluation).
# Default to the challenge brief requirement (slope_deg=5.0, bump_scale=0.1,
# sigma=3.0). Can be overridden via env vars for curriculum training — when
# TA runs Challenge3.py without those env vars set, defaults apply.
TERRAIN_SLOPE_DEG = float(os.environ.get("EVOROB_SLOPE_DEG", "5.0"))
TERRAIN_BUMP_SCALE = float(os.environ.get("EVOROB_BUMP_SCALE", "0.1"))
TERRAIN_SIGMA = float(os.environ.get("EVOROB_SIGMA", "3.0"))


class AntWorld(World):

    def __init__(self,):
        action_space = 8  # https://gymnasium.farama.org/environments/mujoco/ant/#action-space
        state_space = 27  # https://gymnasium.farama.org/environments/mujoco/ant/#observation-space

        # The final submission uses an MLP (Challenge 3 brief: "Run NSGA-II on
        # the same environment with MLP"). AntWorld's default controller is the
        # one the TA's evaluate_checkpoint() reads from, so it must match the
        # controller that produced x_best.npy.
        self.controller = NeuralNetworkController(
            input_size=state_space,
            output_size=action_space,
            hidden_size=action_space,
        )

        self.n_weights = self.controller.n_params
        self.n_body_params = 8

        self.n_params = self.n_weights + self.n_body_params
        self.temp_dir = TemporaryDirectory()
        self.world_file = join(self.temp_dir.name, "AntHillEnv.xml")
        self.create_terrain_file("terrain.png")
        self.base_xml_path = join(ROOT_DIR, "evorob", "world", "robot", "assets", "hill_world.xml")

        self.joint_limits = [[-30, 30], [30, 70],
                        [-30, 30], [-70, -30],
                        [-30, 30], [-70, -30],
                        [-30, 30], [30, 70], ]
        self.joint_axis = [[0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      [0, 0, 1], [-1, 1, 0],
                      [0, 0, 1], [1, 1, 0],
                      ]

    def update_robot_xml(self, genotype: np.ndarray):
        points, connectivity_mat = self.geno2pheno(genotype)
        robot = AntRobot(points, connectivity_mat, self.joint_limits, self.joint_axis, verbose=False)
        robot.xml = robot.define_robot()
        robot.write_xml(self.temp_dir.name)

        #% Defining the Robot environment in MuJoCo
        world = xml.parse(self.base_xml_path)
        robot_env = world.getroot()

        robot_env.append(xml.Element("include", attrib={"file": "AntRobot.xml"}))
        world_xml = xml.tostring(robot_env, encoding="unicode")
        with open(self.world_file, "w") as f:
            f.write(world_xml)

    def create_env(self, render_mode: str = "rgb_array", n_envs: int = 1, max_episode_steps: int = 1000, reset_noise_scale=0.1, **kwargs):
        envs = AsyncVectorEnv(
            [
                lambda i_env=i_env: gym.make(
                    ENV_NAME,
                    robot_path=self.world_file,
                    reset_noise_scale=reset_noise_scale,
                    max_episode_steps=max_episode_steps,
                    render_mode=render_mode,
                )
                for i_env in range(n_envs)
            ]
        )
        return envs

    def geno2pheno(self, genotype):
        control_weights = genotype[:self.n_weights]*0.1
        body_params = (genotype[self.n_weights:]+1)/4+0.1
        assert len(body_params) == self.n_body_params
        assert len(control_weights) == self.n_weights
        assert not np.any(body_params <= 0)

        self.controller.geno2pheno(control_weights)

        front_left_leg, front_left_ankle, front_right_leg, front_right_ankle, back_left_leg, back_left_ankle, back_right_leg, back_right_ankle, = body_params

        # Define the 3D coordinates of the relative tree structure
        front_left_hip_xyz = np.array([0.2, 0.2, 0])
        front_left_knee_xyz = np.array([np.sqrt(0.5 * front_left_leg ** 2), np.sqrt(0.5 * front_left_leg ** 2), 0]) + front_left_hip_xyz
        front_left_toe_xyz = np.array([np.sqrt(0.5 * front_left_ankle ** 2), np.sqrt(0.5 * front_left_ankle ** 2), 0]) + front_left_knee_xyz

        front_right_hip_xyz = np.array([-0.2, 0.2, 0])
        front_right_knee_xyz = np.array([-np.sqrt(0.5 * front_right_leg ** 2), np.sqrt(0.5 * front_right_leg ** 2), 0]) + front_right_hip_xyz
        front_right_toe_xyz = np.array([-np.sqrt(0.5 * front_right_ankle ** 2), np.sqrt(0.5 * front_right_ankle ** 2), 0]) + front_right_knee_xyz

        back_left_hip_xyz = np.array([-0.2, -0.2, 0])
        back_left_knee_xyz = np.array([-np.sqrt(0.5 * back_left_leg ** 2), -np.sqrt(0.5 * back_left_leg ** 2), 0]) + back_left_hip_xyz
        back_left_toe_xyz = np.array([-np.sqrt(0.5 * back_left_ankle ** 2), -np.sqrt(0.5 * back_left_ankle ** 2), 0]) + back_left_knee_xyz

        back_right_hip_xyz = np.array([0.2, -0.2, 0])
        back_right_knee_xyz = np.array([np.sqrt(0.5 * back_right_leg ** 2), -np.sqrt(0.5 * back_right_leg ** 2), 0]) + back_right_hip_xyz
        back_right_toe_xyz = np.array([np.sqrt(0.5 * back_right_ankle ** 2), -np.sqrt(0.5 * back_right_ankle ** 2), 0]) + back_right_knee_xyz

        points = np.vstack([front_left_hip_xyz,
                            front_left_knee_xyz,
                            front_left_toe_xyz,
                            front_right_hip_xyz,
                            front_right_knee_xyz,
                            front_right_toe_xyz,
                            back_left_hip_xyz,
                            back_left_knee_xyz,
                            back_left_toe_xyz,
                            back_right_hip_xyz,
                            back_right_knee_xyz,
                            back_right_toe_xyz,
                            ])

        # define the type of connections [FIXED ARCHITECTURE]
        connectivity_mat = np.array(
            [[150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 150, np.inf, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ]
        )
        return points, connectivity_mat


    def create_terrain_file(self, filename="terrain.png", width=400, depth=400):
        # 1. Create the Slope (Gradient along X)
        # 0.0 at the back, 1.0 at the front
        slope_deg = TERRAIN_SLOPE_DEG
        bump_scale = TERRAIN_BUMP_SCALE
        sigma = TERRAIN_SIGMA

        # 1. Create Linear Slope (Gradient along X)
        rise = np.tan(np.deg2rad(slope_deg))
        slope_factor = rise
        x = np.linspace(0, 1, depth)
        y = np.linspace(0, 1, width)
        X, Y = np.meshgrid(x, y)

        # Use tan to get actual height ratio, but clip to 1.0 to stay within hfield Z-bounds
        # (Assuming max height in XML is defined as the Z-scale)
        slope_map = X * slope_factor

        # 2. Add Bumps (Noise)
        rng = np.random.default_rng(42)
        noise = rng.uniform(0, 1, (width, depth))
        gentle_bump = np.tanh(X*10)
        noise = scipy.ndimage.gaussian_filter(noise, sigma=sigma)
        noise = (noise - noise.min()) / (noise.max() - noise.min())*gentle_bump
        noise_map = noise * bump_scale

        terrain = slope_map + noise_map
        terrain = np.clip(terrain, 0, 1)
        terrain[-1, -1] = 1
        terrain_normalized = (terrain * 255).astype(np.uint8)

        img = Image.fromarray(terrain_normalized, mode='L')
        save_path = os.path.join(self.temp_dir.name, filename)
        img.save(save_path)



    def evaluate_individual(self, genotype, n_repeats=8, n_steps=500):
        self.update_robot_xml(genotype)
        envs = self.create_env(n_envs=n_repeats, max_episode_steps=n_steps)
        self.controller.reset_controller(batch_size=n_repeats)

        rewards_full = np.zeros((n_steps, n_repeats))
        multi_obj_rewards_full = np.zeros((n_steps, n_repeats, 2))

        observations, info = envs.reset()
        done_mask = np.zeros(n_repeats, dtype=bool)
        for step in range(n_steps):
            actions = np.where(done_mask[:, None], 0, self.controller.get_action(observations))
            observations, rewards, dones, truncated, infos = envs.step(actions)

            # Store rewards for active environments only
            rewards_full[step, ~done_mask] = rewards[~done_mask]

            # Multi-objective rewards matching the TA's evaluate_checkpoint:
            #   obj1 = reward_forward + healthy_reward   (progress + survival)
            #   obj2 = -ctrl_cost                        (actuator efficiency)
            # AsyncVectorEnv auto-resets on termination, so on the reset step
            # info carries `final_info` instead of the usual reward keys — fall
            # back to zero in that case (the episode is already ending).
            zero = np.zeros(n_repeats, dtype=float)
            rf = infos.get("reward_forward", zero)
            hr = infos.get("healthy_reward", zero)
            cc = infos.get("ctrl_cost", zero)
            obj1 = rf + hr
            obj2 = -cc
            multi_obj_reward = np.stack([obj1, obj2], axis=1)
            multi_obj_rewards_full[step, ~done_mask] = multi_obj_reward[~done_mask]

            # Update the done mask based on the "done" and "truncated" flags
            done_mask = done_mask | dones | truncated

            # Optionally, break if all environments have terminated
            if np.all(done_mask):
                break
        final_rewards = np.sum(rewards_full, axis=0)
        final_multi_obj_rewards = np.sum(multi_obj_rewards_full, axis=0)
        envs.close()
        return np.mean(final_rewards), np.mean(final_multi_obj_rewards, axis=0)


# ---------------------------------------------------------------------------
# Parallel evaluation helpers
# ---------------------------------------------------------------------------

_EVAL_WORLD = None  # per-process cached world

def _worker_init():
    """Create one AntWorld per worker process (MuJoCo temp files etc.)."""
    global _EVAL_WORLD
    os.environ.setdefault("MUJOCO_GL", "egl")
    _EVAL_WORLD = AntWorld()


def _evaluate_one(args):
    """Worker entry-point: evaluate a single genotype.

    Arguments are packed into a tuple so this is picklable.
    """
    genotype, n_repeats, n_steps = args
    global _EVAL_WORLD
    if _EVAL_WORLD is None:
        _worker_init()
    _, mo_reward = _EVAL_WORLD.evaluate_individual(
        genotype, n_repeats=n_repeats, n_steps=n_steps
    )
    return mo_reward


def evaluate_population_parallel(population, n_repeats, n_steps, max_workers):
    """Evaluate a whole population in parallel across processes.

    Each worker owns its own AntWorld (so its own MuJoCo XML temp dir).
    """
    args_list = [(g, n_repeats, n_steps) for g in population]
    fitnesses = np.empty((len(population), 2), dtype=float)
    with ProcessPoolExecutor(
        max_workers=max_workers, initializer=_worker_init
    ) as pool:
        for idx, fit in enumerate(pool.map(_evaluate_one, args_list)):
            fitnesses[idx] = fit
    return fitnesses


# ---------------------------------------------------------------------------
# Evaluation helpers (Challenge 3)
# ---------------------------------------------------------------------------

def _run_episodes_hill(world, genotype, n_episodes, max_episode_steps, seed):
    """Run n_episodes on the hill terrain, returning per-episode stats.

    Returns:
        episode_rewards: total reward per episode
        episode_obj1:    cumulative (reward_forward + healthy_reward) per episode
        episode_obj2:    cumulative (-ctrl_cost) per episode
    """
    world.update_robot_xml(genotype)
    env = gym.make(
        ENV_NAME,
        robot_path=world.world_file,
        max_episode_steps=max_episode_steps,
    )

    rng = np.random.default_rng(seed)
    episode_rewards, episode_obj1, episode_obj2 = [], [], []

    for _ in range(n_episodes):
        ep_seed = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=ep_seed)
        world.controller.reset_controller(batch_size=1)

        total_reward = total_obj1 = total_obj2 = 0.0
        for _ in range(max_episode_steps):
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_obj1 += float(info.get("reward_forward", 0.0)) + float(info.get("healthy_reward", 0.0))
            total_obj2 += -float(info.get("ctrl_cost", 0.0))
            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        episode_obj1.append(total_obj1)
        episode_obj2.append(total_obj2)

    env.close()
    return episode_rewards, episode_obj1, episode_obj2


def _record_video_hill(world, genotype, max_steps, seed, out_path):
    """Record one episode to out_path. Returns episode reward or None on failure."""
    try:
        world.update_robot_xml(genotype)
        env = gym.make(
            ENV_NAME,
            robot_path=world.world_file,
            max_episode_steps=max_steps,
            render_mode="rgb_array",
        )
        world.controller.reset_controller(batch_size=1)
        obs, _ = env.reset(seed=seed)
        frames, video_reward = [], 0.0

        for _ in range(max_steps):
            frames.append(env.render())
            action = world.controller.get_action(obs)
            if action.ndim > 1:
                action = action.squeeze(0)
            obs, reward, terminated, truncated, _ = env.step(action)
            video_reward += reward
            if terminated or truncated:
                break

        env.close()
        imageio.mimwrite(out_path, frames, fps=20)
        print(f"  Video saved: {out_path}")
        return video_reward
    except Exception as e:
        print(f"  Warning: video skipped ({e})")
        return None


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "values": list(values),
        "mean": float(np.mean(a)),
        "std":  float(np.std(a)),
        "best": float(np.max(a)),
        "worst": float(np.min(a)),
        "median": float(np.median(a)),
    }


def evaluate_checkpoint(
    checkpoint_dir: str,
    output_dir: str = "evaluation_output",
    n_episodes: int = 256,          # set to 256 for submission; lower for testing
):
    """Evaluate a Challenge-3 NSGA-II checkpoint on the hilly terrain.

    Identifies two specialists (best per objective) and a generalist (best
    combined score) from the last checkpoint's population, evaluates each for
    n_episodes, records three videos, and writes a score file.

    The controller type and genotype size are taken from the AntWorld default
    (whatever the student configured), so no hardcoded assumptions are made.
    Objectives: [reward_forward + healthy_reward,  -ctrl_cost].

    Args:
        checkpoint_dir: Path to your NSGA-II checkpoint folder.
        output_dir:     Where to save the score file and videos.
        n_episodes:     Episodes per individual (256 for submission).
    """
    max_episode_steps: int = 1000  # DO NOT CHANGE!
    seed: int = 0                  # DO NOT CHANGE!

    # --- Locate checkpoint files ---
    last_gen = get_last_checkpoint_dir(checkpoint_dir)

    def _try_load(filename):
        for d in ([last_gen] if last_gen else []) + [checkpoint_dir]:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                return np.load(p, allow_pickle=True)
        return None

    x_best = _try_load("x_best.npy")
    if x_best is None:
        print(f"ERROR: Could not find x_best.npy in '{checkpoint_dir}'.")
        return None

    population = _try_load("x.npy")
    fitness    = _try_load("f.npy")
    print(f"Loaded x_best  (shape: {x_best.shape})")

    # --- Identify specialist and generalist genotypes ---
    if (population is not None and fitness is not None
            and fitness.ndim == 2 and fitness.shape[1] >= 2):
        spec1_idx = int(np.argmax(fitness[:, 0]))           # best forward+healthy
        spec2_idx = int(np.argmax(fitness[:, 1]))           # best efficiency

        # Generalist = true Pareto-knee: closest to the normalised utopia
        # corner (1, 1) after min/max scaling each objective OVER THE PARETO
        # FRONT ONLY. Normalising over the whole population lets fallen/slow
        # individuals skew the scale, which in practice pushes the knee
        # toward one of the specialist extremes. Front-only normalisation
        # gives a genuinely middle-of-front ant.
        f = fitness.astype(float).copy()
        n = len(f)
        # Find Pareto (non-dominated) front
        nd = np.ones(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(f[j] >= f[i]) and np.any(f[j] > f[i]):
                    nd[i] = False; break
        front = np.where(nd)[0]
        # Normalise over the front
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
                    gen_idx = cand; break

        spec1_g, spec2_g, gen_g = population[spec1_idx], population[spec2_idx], population[gen_idx]
        print(f"Specialist obj1 (forward): idx={spec1_idx}  f={fitness[spec1_idx]}")
        print(f"Specialist obj2 (effic.) : idx={spec2_idx}  f={fitness[spec2_idx]}")
        print(f"Generalist (Pareto knee) : idx={gen_idx}    f={fitness[gen_idx]}")
    else:
        print("Warning: population/fitness not found — using x_best for all three roles.")
        spec1_g = spec2_g = gen_g = x_best

    # --- AntWorld uses whatever controller the student configured ---
    world = AntWorld()
    controller_name = type(world.controller).__name__
    print(f"Controller: {controller_name}  |  params={world.controller.n_params}  |  genotype size={world.n_params}\n")

    individuals = {
        "specialist_obj1": spec1_g,
        "specialist_obj2": spec2_g,
        "generalist":      gen_g,
    }

    # --- Run episodes ---
    results = {}
    for label, genotype in individuals.items():
        print(f"Evaluating {label} ({n_episodes} episodes)...")
        rewards, obj1_vals, obj2_vals = _run_episodes_hill(
            world, genotype, n_episodes, max_episode_steps, seed
        )
        results[label] = {
            "reward": _stats(rewards),
            "obj1":   _stats(obj1_vals),
            "obj2":   _stats(obj2_vals),
        }
        r = results[label]
        print(f"  reward: {r['reward']['mean']:.2f} +/- {r['reward']['std']:.2f}  "
              f"obj1: {r['obj1']['mean']:.2f}  obj2: {r['obj2']['mean']:.2f}")

    # --- Record videos ---
    os.makedirs(output_dir, exist_ok=True)
    video_names = {
        "specialist_obj1": "specialist_forward",
        "specialist_obj2": "specialist_efficiency",
        "generalist":      "generalist",
    }
    for label, genotype in individuals.items():
        vpath = os.path.join(output_dir, f"evaluation_{video_names[label]}.mp4")
        _record_video_hill(world, genotype, max_episode_steps, seed, vpath)

    # --- Score file ---
    score_path = os.path.join(output_dir, "evaluation_score.txt")
    with open(score_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MICRO-515 Challenge 3 - Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Controller      : {controller_name} ({world.controller.n_params} params)\n")
        f.write(f"Genotype size   : {world.n_params}  (weights={world.controller.n_params}, body={world.n_body_params})\n")
        f.write(f"Checkpoint      : {checkpoint_dir}\n")
        f.write(f"Episodes/indiv. : {n_episodes}\n")
        f.write(f"Objectives      : [reward_forward+healthy_reward, -ctrl_cost]\n\n")

        f.write("=" * 72 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Individual':<22s} {'Rew.Mean':>9s} {'Rew.Std':>8s} {'Rew.Best':>9s} "
                f"{'Obj1.Mean':>10s} {'Obj2.Mean':>10s}\n")
        f.write("-" * 72 + "\n")
        for label in individuals:
            r = results[label]
            f.write(f"{label:<22s} {r['reward']['mean']:9.2f} {r['reward']['std']:8.2f} "
                    f"{r['reward']['best']:9.2f} {r['obj1']['mean']:10.2f} {r['obj2']['mean']:10.2f}\n")
        f.write("\n")

        for label in individuals:
            r = results[label]
            f.write("-" * 50 + "\n")
            f.write(f"{label.upper()} — Per-episode rewards\n")
            f.write("-" * 50 + "\n")
            for i, rew in enumerate(r["reward"]["values"]):
                f.write(f"  Episode {i + 1:3d}: {rew:10.2f}\n")
            f.write("\n")

    print(f"\nScore saved to: {score_path}")
    print("=" * 60)
    for label in individuals:
        r = results[label]
        print(f"  {label:<22s}: reward={r['reward']['mean']:.2f} +/-{r['reward']['std']:.2f}"
              f"  obj1={r['obj1']['mean']:.2f}  obj2={r['obj2']['mean']:.2f}")
    print("=" * 60)
    return results


def run_EA_multi(ea_multi, world, n_repeats=4, n_steps=500, max_workers=16):
    """Run NSGA-II with a parallel-process evaluator over the population."""
    for _ in trange(ea_multi.n_gen):
        pop = ea_multi.ask()
        fitnesses_gen = evaluate_population_parallel(
            pop, n_repeats=n_repeats, n_steps=n_steps, max_workers=max_workers
        )
        ea_multi.tell(pop, fitnesses_gen, save_checkpoint=True)


def main():
    #%% Optimise multi-objective with NSGA-II on MLP controller
    world = AntWorld()
    n_parameters = world.n_params
    print(f"Number of parameters: {n_parameters}")
    print(f"Number of weights:    {world.n_weights}")
    print(f"Number of body params:{world.n_body_params}")

    # Hyperparameters
    population_size = 150
    num_generations = 200
    num_parents     = population_size
    mutation_prob   = 0.5
    crossover_prob  = 0.9
    bounds          = (-1, 1)
    n_repeats       = 4      # episodes per individual (averaged)
    n_steps         = 500    # steps per episode during training
    max_workers     = 16     # parallel evaluation processes

    results_dir = join(ROOT_DIR, "results", ENV_NAME, "multi")
    ea_multi_obj = NSGAII(
        population_size=population_size,
        n_opt_params=n_parameters,
        n_parents=num_parents,
        num_generations=num_generations,
        bounds=bounds,
        mutation_prob=mutation_prob,
        crossover_prob=crossover_prob,
        output_dir=results_dir,
    )
    ea_multi_obj.directory_name = results_dir

    run_EA_multi(
        ea_multi_obj, world,
        n_repeats=n_repeats, n_steps=n_steps, max_workers=max_workers,
    )

    #%% visualise the generalist
    checkpoint = get_last_checkpoint_dir(results_dir)
    best_individual = np.load(join(checkpoint, "x_best.npy"), allow_pickle=True)
    world.update_robot_xml(best_individual)
    env = world.create_env(max_episode_steps=1000)
    video_name = get_distinct_filename(join(results_dir, "best.mp4"))
    print(f"Finished NSGAII run, generating video [{video_name}]...")
    world.generate_best_individual_video(env, video_name=video_name, n_steps=1000)


if __name__ == "__main__":
    main()
