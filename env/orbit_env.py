"""
env/orbit_env.py — Gymnasium wrapper around kaggle_environments `orbit_wars`,
plus helpers to turn a trained model into a kaggle opponent agent.

MaskablePPO needs an `action_masks()` method on the env; we expose it and wrap
with sb3-contrib's ActionMasker.
"""
import os
from env import tb_compat  # noqa: F401  (must precede any sb3 import)
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kaggle_environments import make
from sb3_contrib.common.wrappers import ActionMasker

from env.core import (
    encode_obs, decode_action, get_action_masks,
    OBS_DIM, ACTION_NVEC,
)
from env.reward import compute_reward

# Builtin string agents understood directly by kaggle_environments.
_BUILTIN = {"random", "starter"}


# ── Model → kaggle opponent agent ─────────────────────────────────────
def make_model_agent(model_path, device="cpu", deterministic=True):
    """Return a kaggle agent callable backed by a saved MaskablePPO model.

    The model is loaded lazily on first call (so the callable stays cheap to
    pickle for SubprocVecEnv / cloudpickle)."""
    from sb3_contrib import MaskablePPO

    state = {"model": None}

    def _agent(obs):
        if state["model"] is None:
            state["model"] = MaskablePPO.load(model_path, device=device)
        planets = obs.get("planets", []) if hasattr(obs, "get") else obs.planets
        player = obs.get("player", 0) if hasattr(obs, "get") else obs.player
        ang_vel = obs.get("angular_velocity", 0.03) if hasattr(obs, "get") else obs.angular_velocity
        vec = encode_obs(obs)
        masks = get_action_masks(planets, player)
        action, _ = state["model"].predict(vec, action_masks=masks, deterministic=deterministic)
        return decode_action(action, planets, player, ang_vel)

    return _agent


def resolve_opponent(opponent):
    """Normalise the `opponent` argument into something env.train accepts.

    Accepts: builtin string ("random"/"starter"), a path to a .zip model, or an
    already-callable agent."""
    if callable(opponent):
        return opponent
    if isinstance(opponent, str):
        if opponent in _BUILTIN:
            return opponent
        if opponent.endswith(".zip") and os.path.exists(opponent):
            return make_model_agent(opponent)
    # Fall back to the string as-is (lets kaggle resolve .py files etc.)
    return opponent


# ── Gym env ───────────────────────────────────────────────────────────
class OrbitWarsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent="starter", reward_profile="full", num_players=4):
        super().__init__()
        self.opponent = opponent          # str/callable, or list of (num_players-1)
        self.reward_profile = reward_profile
        self.num_players = num_players     # leaderboard runs 4-player games

        self.observation_space = spaces.Box(-2.0, 2.0, (OBS_DIM,), np.float32)
        self.action_space = spaces.MultiDiscrete(ACTION_NVEC)

        self._env = None
        self._trainer = None
        self._prev = None
        self._step = 0

    def _opponents(self):
        """Resolve the (num_players - 1) opponent agents."""
        n = self.num_players - 1
        opp = self.opponent
        specs = opp if isinstance(opp, (list, tuple)) else [opp] * n
        specs = (list(specs) + [specs[-1]] * n)[:n]   # pad/truncate to n
        return [resolve_opponent(s) for s in specs]

    # ── Gym API ──
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env = make("orbit_wars", debug=False)
        self._trainer = self._env.train([None] + self._opponents())
        raw = self._trainer.reset()
        self._prev = raw
        self._step = 0
        return encode_obs(raw), {}

    def _prev_field(self, key, default):
        p = self._prev
        if p is None:
            return default
        return p.get(key, default) if hasattr(p, "get") else getattr(p, key, default)

    def step(self, action):
        raw_planets = self._prev_field("planets", [])
        ang_vel = self._prev_field("angular_velocity", 0.03)
        player = self._prev_field("player", 0)

        kaggle_act = decode_action(action, raw_planets, player, ang_vel)
        raw, krew, done, info = self._trainer.step(kaggle_act)
        self._step += 1

        obs = encode_obs(raw) if raw is not None else np.zeros(OBS_DIM, np.float32)
        rew = compute_reward(self._prev, raw, player, done, krew, profile=self.reward_profile)
        self._prev = raw if raw is not None else self._prev
        return obs, rew, bool(done), False, info

    # ── Action masking ──
    def action_masks(self) -> np.ndarray:
        if self._prev is None:
            return np.ones(sum(ACTION_NVEC), dtype=bool)
        raw_planets = self._prev_field("planets", [])
        player = self._prev_field("player", 0)
        return get_action_masks(raw_planets, player)


def make_masked_env(opponent="starter", reward_profile="full", num_players=4):
    """Factory returning an ActionMasker-wrapped OrbitWarsEnv."""
    def _init():
        env = OrbitWarsEnv(opponent=opponent, reward_profile=reward_profile,
                           num_players=num_players)
        return ActionMasker(env, lambda e: e.action_masks())
    return _init
