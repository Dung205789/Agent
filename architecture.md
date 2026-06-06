# architecture.md — Orbit Wars: PPO System Architecture

## Sản phẩm cuối cùng

```
submission_notebook.ipynb
  ← chạy trên Kaggle (GPU T4, miễn phí)
  ← train PPO locally, upload weights lên Kaggle Dataset
  ← notebook tự viết main.py + đóng gói + submit
```

---

## 1. Cấu trúc project

```
orbit-wars/
│
├── submission_notebook.ipynb   ← DELIVERABLE DUY NHẤT
│
├── env/
│   ├── __init__.py
│   ├── core.py          # encode_obs, decode_action, get_action_masks, physics
│   ├── orbit_env.py     # Gymnasium wrapper (MaskableEnv)
│   └── reward.py        # compute_reward — tách riêng để ablate
│
├── train/
│   ├── curriculum.py    # 3-stage curriculum manager
│   ├── selfplay.py      # ELO-based opponent pool
│   └── train.py         # entry point: python train/train.py --stage [1|2|3]
│
├── eval/
│   ├── arena.py         # round-robin tournament local
│   └── tracker.py       # ghi ELO, win rate theo submission
│
├── models/
│   ├── stage1_random.zip
│   ├── stage2_rulebased.zip
│   └── stage3_selfplay.zip     ← model upload lên Kaggle Dataset
│
└── results.log
```

---

## 2. Gymnasium Wrapper (MaskableEnv)

```python
# env/orbit_env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from sb3_contrib.common.wrappers import ActionMasker
from env.core import encode_obs, decode_action, get_action_masks
from env.reward import compute_reward

OBS_DIM = 12 * 40 + 7 * 20 + 5  # 625

class OrbitWarsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent="random"):
        super().__init__()
        self.opponent = opponent

        self.observation_space = spaces.Box(-2.0, 2.0, (OBS_DIM,), np.float32)
        # MultiDiscrete([10 src, 40 tgt, 4 frac])
        self.action_space = spaces.MultiDiscrete([10, 40, 4])

        self._env = self._trainer = self._prev = None
        self._step = 0

    # ── Gym API ──────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env     = make("orbit_wars", debug=False)
        self._trainer = self._env.train([None, self.opponent])
        raw           = self._trainer.reset()
        self._prev    = raw
        self._step    = 0
        return encode_obs(raw), {}

    def step(self, action):
        raw_planets = self._prev.get("planets", []) if isinstance(self._prev, dict) else self._prev.planets
        ang_vel     = self._prev.get("angular_velocity", 0.03)
        player      = self._prev.get("player", 0)

        kaggle_act  = decode_action(action, raw_planets, player, ang_vel)
        raw, krew, done, info = self._trainer.step(kaggle_act)
        self._step += 1

        obs  = encode_obs(raw) if raw is not None else np.zeros(OBS_DIM, np.float32)
        rew  = compute_reward(self._prev, raw, player, done, krew)
        self._prev = raw
        return obs, rew, bool(done), False, info

    # ── Action masking (required by MaskablePPO) ─────────
    def action_masks(self) -> np.ndarray:
        if self._prev is None:
            return np.ones(10 + 40 + 4, dtype=bool)
        raw_planets = self._prev.get("planets", []) if isinstance(self._prev, dict) else self._prev.planets
        player      = self._prev.get("player", 0)
        return get_action_masks(raw_planets, player)


def make_masked_env(opponent="random"):
    """Bọc ActionMasker để MaskablePPO tự động gọi action_masks()."""
    env = OrbitWarsEnv(opponent=opponent)
    return ActionMasker(env, lambda e: e.action_masks())
```

---

## 3. Reward Function

```python
# env/reward.py
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

WEIGHTS = {
    "production_tick" : 0.002,   # A: mỗi ship sản xuất được mỗi turn
    "planet_ratio"    : 0.010,   # B: tỉ lệ planet đang nắm giữ
    "capture_bonus"   : 0.100,   # C: capture 1 planet (× production value)
    "lose_penalty"    : 0.150,   # D: mất 1 planet (× production value)
    "ship_ratio"      : 0.020,   # E: tỉ lệ ships tổng — align với win condition
    "terminal"        : 10.0,    # F: win/loss cuối game (amplified)
}

def compute_reward(prev_raw, curr_raw, player, done, kaggle_reward) -> float:
    if curr_raw is None:
        return kaggle_reward * WEIGHTS["terminal"] if done else 0.0

    def pl(raw): return [Planet(*p) for p in raw.get("planets", [])]
    prev_pl = pl(prev_raw) if prev_raw else []
    curr_pl = pl(curr_raw)

    reward = 0.0

    # A — production tick
    reward += sum(p.production for p in curr_pl if p.owner == player) * WEIGHTS["production_tick"]

    # B — planet ratio
    reward += (sum(1 for p in curr_pl if p.owner == player) / max(len(curr_pl), 1)) * WEIGHTS["planet_ratio"]

    # C & D — capture / loss
    prev_mine = {p.id: p for p in prev_pl if p.owner == player}
    curr_mine = {p.id: p for p in curr_pl if p.owner == player}
    for pid in set(curr_mine) - set(prev_mine):   # gained
        reward += curr_mine[pid].production * WEIGHTS["capture_bonus"]
    for pid in set(prev_mine) - set(curr_mine):   # lost
        reward -= prev_mine[pid].production * WEIGHTS["lose_penalty"]

    # E — ship ratio
    my_ships  = sum(p.ships for p in curr_pl if p.owner == player)
    all_ships = sum(p.ships for p in curr_pl) + 1e-6
    reward += (my_ships / all_ships) * WEIGHTS["ship_ratio"]

    # F — terminal
    if done:
        reward += kaggle_reward * WEIGHTS["terminal"]

    return float(reward)
```

---

## 4. Curriculum Training (3 giai đoạn)

```
Stage 1: vs random (5M steps)
  → Agent học: farm ships, không bắn qua sun, expand nhanh

Stage 2: vs rule-based (15M steps)
  → Agent học: intercept orbit, counter-attack, value production planets

Stage 3: self-play ELO pool (15M steps, 5 rounds × 3M)
  → Agent học: bluffing, multi-front, late-game consolidation
```

```python
# train/curriculum.py
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from env.orbit_env import make_masked_env

PPO_BASE = dict(
    learning_rate  = 3e-4,
    n_steps        = 2048,
    batch_size     = 512,
    n_epochs       = 10,
    gamma          = 0.995,   # game 500 turns — cao
    gae_lambda     = 0.95,
    clip_range     = 0.2,
    ent_coef       = 0.02,
    vf_coef        = 0.5,
    max_grad_norm  = 0.5,
    policy_kwargs  = dict(net_arch=[256, 256, 128]),
    verbose        = 1,
    tensorboard_log= "./tb_logs/",
)

N_ENVS = 8  # tune theo CPU: 4 nếu laptop, 8+ nếu workstation

def train_stage1():
    """vs random — bootstrap từ scratch."""
    vec = SubprocVecEnv([lambda: make_masked_env("random")] * N_ENVS)
    model = MaskablePPO("MlpPolicy", vec, **PPO_BASE)
    model.learn(5_000_000)
    model.save("models/stage1_random")
    return model

def train_stage2(model=None):
    """vs rule-based heuristic — học intercept orbit."""
    vec = SubprocVecEnv([lambda: make_masked_env("models/stage1_random.zip")] * N_ENVS)
    if model is None:
        model = MaskablePPO.load("models/stage1_random", env=vec)
    model.set_env(vec)
    model.learning_rate = 1e-4   # fine-tune
    model.ent_coef      = 0.01
    model.learn(15_000_000, reset_num_timesteps=False)
    model.save("models/stage2_rulebased")
    return model

def train_stage3():
    """Self-play với ELO pool."""
    from train.selfplay import SelfPlayPool
    pool = SelfPlayPool(seed="models/stage2_rulebased.zip")

    model = MaskablePPO.load("models/stage2_rulebased")
    model.learning_rate = 5e-5
    model.ent_coef      = 0.005

    for round_i in range(5):    # 5 × 3M = 15M steps
        opp = pool.sample()
        vec = SubprocVecEnv([lambda: make_masked_env(opp)] * N_ENVS)
        model.set_env(vec)
        model.learn(3_000_000, reset_num_timesteps=False)
        pool.save(model, round_i)
        print(f"Round {round_i}: pool={pool.paths}")

    model.save("models/stage3_selfplay")
    return model
```

---

## 5. Self-Play ELO Pool

```python
# train/selfplay.py
import os, random

class SelfPlayPool:
    """
    Giữ tối đa 6 snapshots gần nhất.
    Sampling weighted: 60% newest, 40% uniform từ pool.
    """
    MAX_SIZE = 6

    def __init__(self, seed=None, dir="snapshots/"):
        os.makedirs(dir, exist_ok=True)
        self.dir   = dir
        self.paths = [seed] if seed else []

    def save(self, model, round_i):
        path = f"{self.dir}/round_{round_i:03d}.zip"
        model.save(path)
        self.paths.append(path)
        if len(self.paths) > self.MAX_SIZE:
            self.paths.pop(0)

    def sample(self) -> str:
        if not self.paths:
            return "random"
        if random.random() < 0.6:
            return self.paths[-1]        # newest
        return random.choice(self.paths) # uniform
```

---

## 6. Evaluation Pipeline

```python
# eval/arena.py
from kaggle_environments import make

def run_match(agent_a, agent_b, n=30, seed_start=0) -> float:
    """Trả về win_rate của agent_a vs agent_b."""
    wins = 0
    for i in range(n):
        env    = make("orbit_wars", configuration={"seed": seed_start + i})
        result = env.run([agent_a, agent_b])
        r_a, r_b = result[-1][0].reward, result[-1][1].reward
        if r_a > r_b: wins += 1
    rate = wins / n
    print(f"{agent_a} vs {agent_b}: {wins}/{n} ({rate:.1%})")
    return rate

# Chạy trước mỗi submission
THRESHOLDS = {
    "S2": {"vs_random": 0.70},
    "S3": {"vs_stage1": 0.60, "vs_random": 0.85},
    "S4": {"vs_stage2": 0.55},
    "S5": {"vs_stage3_prev": 0.52},
}
```

---

## 7. Submission Notebook (sản phẩm cuối)

```
submission_notebook.ipynb
```

Cấu trúc notebook theo section:

```python
# ── Section 1: Setup ──────────────────────────────────
!pip install -q "kaggle-environments>=1.28.0" sb3-contrib

# ── Section 2: Upload weights (chạy 1 lần sau khi train local) ──
# Upload models/stage3_selfplay.zip lên Kaggle Dataset: "orbit-wars-weights"
# Sau đó attach Dataset vào notebook

# ── Section 3: Verify weights load ────────────────────
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("/kaggle/input/orbit-wars-weights/stage3_selfplay.zip", device="cpu")
print("Model loaded:", model.policy)

# ── Section 4: Viết main.py ───────────────────────────
%%writefile main.py
# [full agent code — copy từ agents.md § 1]
# Không có heuristic, không có fallback.
# Agent raise exception → episode fail → log để debug

import os, math, numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from sb3_contrib import MaskablePPO

# [paste encode_obs, decode_action, get_action_masks, physics helpers]
# [paste agent() function]

# ── Section 5: Test local trước khi submit ────────────
from kaggle_environments import make

results = {"win": 0, "lose": 0, "draw": 0}
for seed in range(20):
    env    = make("orbit_wars", configuration={"seed": seed})
    result = env.run(["main.py", "random"])
    r0, r1 = result[-1][0].reward, result[-1][1].reward
    if r0 > r1: results["win"]  += 1
    elif r0 < r1: results["lose"] += 1
    else: results["draw"] += 1

win_rate = results["win"] / 20
print(f"Win rate vs random: {win_rate:.1%}  {results}")
assert win_rate >= 0.75, f"Không đủ tiêu chuẩn submit: {win_rate:.1%}"

# ── Section 6: Package ────────────────────────────────
import tarfile, shutil, os

os.makedirs("submission/", exist_ok=True)
shutil.copy("main.py", "submission/main.py")

with tarfile.open("submission.tar.gz", "w:gz") as tar:
    tar.add("submission/main.py", arcname="main.py")

print("Kiểm tra nội dung:")
with tarfile.open("submission.tar.gz") as tar:
    tar.list()   # phải có main.py ở root

# ── Section 7: Submit ─────────────────────────────────
SUBMISSION_MESSAGE = "S3 PPO + orbital intercept v1"

!kaggle competitions submit orbit-wars \
    -f submission.tar.gz \
    -m "{SUBMISSION_MESSAGE}"

# ── Section 8: Monitor ────────────────────────────────
!kaggle competitions submissions orbit-wars
```

---

## 8. Luồng làm việc đầy đủ

```
[Local machine]
  python train/train.py --stage 1   →  models/stage1_random.zip
  python train/train.py --stage 2   →  models/stage2_rulebased.zip
  python train/train.py --stage 3   →  models/stage3_selfplay.zip
  python eval/arena.py              →  confirm win rate > threshold
        ↓ upload
[Kaggle Dataset: orbit-wars-weights]
  stage3_selfplay.zip
        ↓ attach vào notebook
[Kaggle Notebook: submission_notebook.ipynb]
  Section 4: %%writefile main.py
  Section 5: test 20 games vs random (win rate check)
  Section 6: tar -czf submission.tar.gz main.py
  Section 7: kaggle competitions submit ...
        ↓
[Kaggle Leaderboard]
  Episode chạy vs các bot khác → ELO update
```

---

## 9. Dependency versions (pin chặt)

```txt
# requirements.txt
kaggle-environments>=1.28.0
stable-baselines3==2.3.2
sb3-contrib==2.3.2
gymnasium==0.29.1
numpy==1.26.4
torch==2.2.2
```

> **Quan trọng**: Kaggle notebook chạy Python 3.10. Pin version để tránh breaking changes.
