"""
train/curriculum.py — 3-stage curriculum training with MaskablePPO.

    Stage 1: vs random           (reward profile = minimal: A + F)
    Stage 2: vs stage-1 model    (reward profile = full)
    Stage 3: self-play ELO pool  (reward profile = full)

All stage fns accept overrides for total_timesteps / n_envs so the same code
path is used for smoke tests and full runs.
"""
import os
from env import tb_compat  # noqa: F401  (must precede any sb3 import)
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env.orbit_env import make_masked_env
from train.selfplay import SelfPlayPool

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

PPO_BASE = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.995,            # 500-turn games -> high discount
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=1,
    # Disable TB automatically when the image can't provide a working writer.
    tensorboard_log="./tb_logs/" if tb_compat.TENSORBOARD_OK else None,
)

N_ENVS = 8


def build_vec(opponent, reward_profile, n_envs, use_subproc=True):
    """Vectorised env. Falls back to DummyVecEnv for n_envs == 1 or when
    subprocesses are undesirable (debugging / Windows quirks)."""
    factories = [make_masked_env(opponent, reward_profile) for _ in range(n_envs)]
    if use_subproc and n_envs > 1:
        return SubprocVecEnv(factories)
    return DummyVecEnv(factories)


def train_stage1(total_timesteps=5_000_000, n_envs=N_ENVS, use_subproc=True,
                 save_path=None, ppo_overrides=None):
    """vs random — bootstrap from scratch. Reward = A + F (minimal)."""
    vec = build_vec("random", "minimal", n_envs, use_subproc)
    cfg = dict(PPO_BASE)
    if ppo_overrides:
        cfg.update(ppo_overrides)
    model = MaskablePPO("MlpPolicy", vec, **cfg)
    model.learn(total_timesteps, progress_bar=False)
    path = save_path or os.path.join(MODELS_DIR, "stage1_random")
    model.save(path)
    vec.close()
    return model, path


def train_stage2(stage1_path=None, total_timesteps=15_000_000, n_envs=N_ENVS,
                 use_subproc=True, save_path=None):
    """vs stage-1 model — learn orbital intercept. Reward = full."""
    stage1_path = stage1_path or os.path.join(MODELS_DIR, "stage1_random.zip")
    vec = build_vec(stage1_path, "full", n_envs, use_subproc)
    model = MaskablePPO.load(stage1_path.replace(".zip", ""), env=vec)
    model.learning_rate = 1e-4      # fine-tune
    model.ent_coef = 0.01
    model.learn(total_timesteps, reset_num_timesteps=False, progress_bar=False)
    path = save_path or os.path.join(MODELS_DIR, "stage2_rulebased")
    model.save(path)
    vec.close()
    return model, path


def train_stage3(stage2_path=None, rounds=3, steps_per_round=3_000_000,
                 n_envs=N_ENVS, use_subproc=True, save_path=None):
    """Self-play with an ELO pool. Reward = full."""
    stage2_path = stage2_path or os.path.join(MODELS_DIR, "stage2_rulebased.zip")
    pool = SelfPlayPool(seed=stage2_path)

    model = MaskablePPO.load(stage2_path.replace(".zip", ""))
    model.learning_rate = 5e-5
    model.ent_coef = 0.005

    last_round_path = None
    for round_i in range(rounds):
        opp = pool.sample()
        vec = build_vec(opp, "full", n_envs, use_subproc)
        model.set_env(vec)
        model.learn(steps_per_round, reset_num_timesteps=False, progress_bar=False)
        last_round_path = pool.save(model, round_i)
        vec.close()
        print(f"[stage3] round {round_i}: opp={opp} pool={pool.paths}")

    path = save_path or os.path.join(MODELS_DIR, "stage3_selfplay")
    model.save(path)
    # Also save a round-tagged alias matching task.md naming (e.g. _r3).
    model.save(os.path.join(MODELS_DIR, f"stage3_selfplay_r{rounds}"))
    return model, path, last_round_path
