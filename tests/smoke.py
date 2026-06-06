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


def test_shapes_and_step():
    env = make_masked_env("random", "minimal")()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    m = env.action_masks()
    assert m.shape == (sum(ACTION_NVEC),) and m[:10].any()
    rng = np.random.default_rng(0)
    done, steps = False, 0
    while not done and steps < 50:
        mask = env.action_masks()
        src = int(rng.choice(np.where(mask[:10])[0]))
        obs, r, done, trunc, info = env.step(np.array([src, int(rng.integers(0, 40)), int(rng.integers(0, 4))]))
        steps += 1
    print(f"[ok] shapes+step: {steps} steps")


def test_decode_validity():
    from kaggle_environments import make
    tr = make("orbit_wars", debug=False).train([None, "random"])
    raw = tr.reset()
    planets, player = raw.get("planets", []), raw.get("player", 0)
    av = raw.get("angular_velocity", 0.03)
    msk = get_action_masks(planets, player)
    encode_obs(raw)
    act = decode_action([int(np.where(msk[:10])[0][0]), 3, 1], planets, player, av)
    if act:
        fid, _, sh = act[0]
        assert any(p[0] == fid and p[1] == player for p in planets) and sh > 0
    print("[ok] decode validity")


def test_train_and_eval():
    from train.curriculum import train_stage1
    from eval.arena import run_match
    _, path = train_stage1(
        total_timesteps=1024, n_envs=2, use_subproc=False,
        save_path="models/stage1_random",
        ppo_overrides={"tensorboard_log": None, "n_steps": 256, "batch_size": 64, "verbose": 0},
    )
    wr = run_match(path + ".zip", "random", n=2, verbose=True)
    print(f"[ok] train+eval (tiny, noisy wr={wr:.0%})")


def test_main_agent():
    from kaggle_environments import make
    os.environ["OW_MODEL_PATH"] = os.path.abspath("models/stage1_random.zip")
    main_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")
    exc = 0
    for seed in range(2):
        result = make("orbit_wars", configuration={"seed": seed}).run([main_path, "random"])
        for a in result[-1]:
            if a.status not in ("ACTIVE", "DONE", "INACTIVE"):
                exc += 1
    assert exc == 0, f"{exc} agent exceptions"
    print("[ok] main.py runs in real env, 0 exceptions")


if __name__ == "__main__":
    test_shapes_and_step()
    test_decode_validity()
    test_train_and_eval()
    test_main_agent()
    print("\nALL SMOKE TESTS PASSED")
