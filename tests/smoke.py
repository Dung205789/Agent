"""
tests/smoke.py — end-to-end regression check (fast).

Run from repo root:  python -m tests.smoke

Covers: obs/mask shapes, masked-env stepping, a tiny real training run,
loading the saved model as a kaggle opponent, and running the inlined main.py
inside the real kaggle env with zero agent exceptions.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from env.core import encode_obs, decode_action, get_action_masks, OBS_DIM, ACTION_NVEC  # noqa: E402
from env.orbit_env import make_masked_env  # noqa: E402


def _sample_masked_action(mask):
    """Pick a valid MultiDiscrete action from a flat mask using ACTION_NVEC."""
    out, o = [], 0
    for n in ACTION_NVEC:
        choices = np.where(mask[o:o + n])[0]
        out.append(int(choices[-1]) if len(choices) else 0)
        o += n
    return np.array(out)


def test_shapes_and_step():
    env = make_masked_env("starter", "full")()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    m = env.action_masks()
    assert m.shape == (sum(ACTION_NVEC),), m.shape
    done, steps = False, 0
    while not done and steps < 50:
        obs, r, done, trunc, info = env.step(_sample_masked_action(env.action_masks()))
        steps += 1
    print(f"[ok] shapes+step: {steps} steps (action dim={len(ACTION_NVEC)})")


def test_multilaunch_decode():
    """A turn with several owned planets must emit several launches."""
    from kaggle_environments import make
    tr = make("orbit_wars", debug=False).train([None, "starter", "starter", "starter"])
    raw = tr.reset()
    for _ in range(40):
        planets = raw.get("planets", []); player = raw.get("player", 0)
        act = _sample_masked_action(get_action_masks(planets, player))
        raw, r, done, info = tr.step(decode_action(act, planets, player, raw.get("angular_velocity", 0.03)))
        if done:
            raw = tr.reset()
    planets, player = raw.get("planets", []), raw.get("player", 0)
    # force every owned slot to target planet 0 (slot 1 targets planet 1 to avoid self)
    act = []
    for i in range(len(ACTION_NVEC) // 2):
        act.append(0 if i != 0 else 1)   # target
        act.append(1)                    # fraction 50%
    moves = decode_action(np.array(act), planets, player, raw.get("angular_velocity", 0.03))
    nmine = sum(1 for p in planets if p[1] == player and p[5] >= 5)
    print(f"[ok] multilaunch: {len(moves)} moves for {nmine} eligible planets")
    assert len(moves) == nmine, (len(moves), nmine)


def test_train_and_eval():
    from train.curriculum import train_stage1
    from eval.arena import run_match
    _, path = train_stage1(
        total_timesteps=1024, n_envs=2, use_subproc=False,
        save_path="models/stage1_starter",
        ppo_overrides={"tensorboard_log": None, "n_steps": 256, "batch_size": 64, "verbose": 0},
    )
    wr = run_match(path + ".zip", "starter", n=2, verbose=True)
    print(f"[ok] train+eval (tiny, noisy wr={wr:.0%})")


def test_main_agent():
    from kaggle_environments import make
    os.environ["OW_MODEL_PATH"] = os.path.abspath("models/stage1_starter.zip")
    main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    exc = 0
    for seed in range(2):
        result = make("orbit_wars", configuration={"seed": seed}).run([main_path, "starter", "starter", "starter"])
        for a in result[-1]:
            if a.status not in ("ACTIVE", "DONE", "INACTIVE"):
                exc += 1
    assert exc == 0, f"{exc} agent exceptions"
    print("[ok] main.py runs in real 4p env, 0 exceptions")


if __name__ == "__main__":
    test_shapes_and_step()
    test_multilaunch_decode()
    test_train_and_eval()
    test_main_agent()
    print("\nALL SMOKE TESTS PASSED")
