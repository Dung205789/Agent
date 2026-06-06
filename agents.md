# agents.md — Orbit Wars: Pure PPO Agent

## Sản phẩm cuối cùng

```
1 Kaggle Notebook (submission_notebook.ipynb)
  ├── Section 1: Training pipeline (train local → upload weights as Dataset)
  ├── Section 2: %%writefile main.py  ← agent function load model từ Dataset
  ├── Section 3: Packaging + submit via Kaggle API
  └── Output: submission.tar.gz được submit lên leaderboard
```

Notebook chạy trên Kaggle (GPU T4) hoặc local, lưu weights lên Kaggle Dataset,
rồi tạo `main.py` + `submission.tar.gz` để submit.

---

## 1. Agent function signature

```python
# main.py — file duy nhất Kaggle load khi chạy episode
import os, math, numpy as np
from sb3_contrib import MaskablePPO

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        # Kaggle load model từ attached Dataset
        path = "/kaggle/input/orbit-wars-weights/best_model.zip"
        _MODEL = MaskablePPO.load(path, device="cpu")
    return _MODEL

def agent(obs, config=None):
    """
    Pure PPO agent — không có heuristic fallback.
    obs   : raw observation dict/SimpleNamespace từ Kaggle env
    return: list of [from_planet_id, angle_rad, num_ships]
    """
    from env.core import encode_obs, decode_action, get_action_masks

    planets  = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    player   = obs.get("player", 0)   if isinstance(obs, dict) else obs.player
    ang_vel  = obs.get("angular_velocity", 0.03)

    obs_vec  = encode_obs(obs)
    masks    = get_action_masks(planets, player)
    action, _= _get_model().predict(obs_vec, action_masks=masks, deterministic=True)

    return decode_action(action, planets, player, ang_vel)
```

---

## 2. Tại sao MaskablePPO thay vì PPO thường

| | PPO thường | MaskablePPO |
|---|---|---|
| Action không hợp lệ | Agent học từ penalty | Không bao giờ xảy ra |
| Sample efficiency | Lãng phí exploration budget | Tập trung vào action có ý nghĩa |
| Convergence speed | Chậm hơn | 2–3× nhanh hơn |
| Implementation | `stable-baselines3` | `sb3-contrib` |

```bash
pip install sb3-contrib  # bao gồm stable-baselines3
```

---

## 3. Observation space (625 dims)

### Planet features — 12 × 40 = 480 dims

```python
# env/core.py
import math
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
MAX_SPEED = 6.0
N_PLANETS, N_FLEETS = 40, 20

def fleet_speed(n: int) -> float:
    if n <= 1: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5

def orbital_radius(p) -> float:
    return math.hypot(p.x - SUN_X, p.y - SUN_Y)

def is_orbiting(p) -> bool:
    return (orbital_radius(p) + p.radius) < 50.0

def predict_pos(p, ang_vel: float, t: int):
    """Vị trí planet sau t turns. Static planet: không đổi."""
    if not is_orbiting(p):
        return p.x, p.y
    r = orbital_radius(p)
    a = math.atan2(p.y - SUN_Y, p.x - SUN_X) + ang_vel * t
    return SUN_X + r * math.cos(a), SUN_Y + r * math.sin(a)

def intercept_angle(src, tgt, ships: int, ang_vel: float):
    """Iterative intercept: tìm angle để fleet bắt kịp planet đang quay."""
    tx, ty = tgt.x, tgt.y
    for _ in range(8):
        dist = math.hypot(tx - src.x, ty - src.y)
        t    = max(1, math.ceil(dist / fleet_speed(ships)))
        tx, ty = predict_pos(tgt, ang_vel, t)
    return math.atan2(ty - src.y, tx - src.x), tx, ty

def hits_sun(x1, y1, x2, y2) -> bool:
    dx, dy = x2 - x1, y2 - y1
    den = dx*dx + dy*dy
    if den < 1e-9:
        return math.hypot(x1 - SUN_X, y1 - SUN_Y) < SUN_R
    t = max(0.0, min(1.0, ((SUN_X-x1)*dx + (SUN_Y-y1)*dy) / den))
    return math.hypot(x1 + t*dx - SUN_X, y1 + t*dy - SUN_Y) < SUN_R

def _encode_planet(p, player, ang_vel, comet_ids, ref_ships=50) -> list:
    dist  = math.hypot(p.x - SUN_X, p.y - SUN_Y)
    t_est = max(1, int(dist / fleet_speed(ref_ships)))
    px, py = predict_pos(p, ang_vel, t_est)
    r     = orbital_radius(p)
    return [
        (px - SUN_X) / 50.0,          # predicted rel_x
        (py - SUN_Y) / 50.0,          # predicted rel_y
        math.sin(math.atan2(p.y - SUN_Y, p.x - SUN_X)),  # current sin_angle
        math.cos(math.atan2(p.y - SUN_Y, p.x - SUN_X)),  # current cos_angle
        min(1.0, r / 50.0),            # orbit_radius_norm
        min(1.0, p.ships / 500.0),     # ships_norm
        p.production / 5.0,            # production_norm
        1.0 if p.owner == player else 0.0,
        1.0 if p.owner not in (-1, player) else 0.0,
        1.0 if p.owner == -1 else 0.0,
        1.0 if is_orbiting(p) else 0.0,
        1.0 if p.id in comet_ids else 0.0,
    ]

def encode_obs(raw_obs) -> np.ndarray:
    planets   = [Planet(*p) for p in raw_obs.get("planets", [])]
    fleets    = [Fleet(*f)  for f in raw_obs.get("fleets", [])]
    player    = raw_obs.get("player", 0)
    ang_vel   = raw_obs.get("angular_velocity", 0.03)
    step      = raw_obs.get("step", 0)
    comet_ids = set(raw_obs.get("comet_planet_ids", []))

    my_pl     = [p for p in planets if p.owner == player]
    ref_ships = max((p.ships for p in my_pl), default=50)

    # Planet features
    pl_feats = []
    for i in range(N_PLANETS):
        if i < len(planets):
            pl_feats.extend(_encode_planet(planets[i], player, ang_vel, comet_ids, ref_ships))
        else:
            pl_feats.extend([0.0] * 12)

    # Fleet features (top 20 by ships)
    fl_feats = []
    for f in (sorted(fleets, key=lambda x: -x.ships) + [None]*N_FLEETS)[:N_FLEETS]:
        if f:
            fl_feats.extend([
                (f.x - SUN_X) / 50.0,
                (f.y - SUN_Y) / 50.0,
                math.sin(f.angle), math.cos(f.angle),
                min(1.0, f.ships / 500.0),
                1.0 if f.owner == player else 0.0,
                1.0 if f.owner != player else 0.0,
            ])
        else:
            fl_feats.extend([0.0] * 7)

    # Global state
    total_pl = max(len(planets), 1)
    my_ships  = sum(p.ships for p in planets if p.owner == player)
    all_ships = sum(p.ships for p in planets) + 1e-6
    my_prod   = sum(p.production for p in planets if p.owner == player)
    all_prod  = sum(p.production for p in planets) + 1e-6
    global_feats = [
        step / 500.0,
        sum(1 for p in planets if p.owner == player) / total_pl,
        my_ships / all_ships,
        my_prod / all_prod,
        raw_obs.get("remainingOverageTime", 60.0) / 60.0,
    ]

    return np.array(pl_feats + fl_feats + global_feats, dtype=np.float32)
```

---

## 4. Action space + Masking

### Design: MultiDiscrete([10, 40, 4])

| Dimension | Ý nghĩa | Masking |
|-----------|---------|---------|
| `src_idx` (0–9) | Index trong danh sách my_planets | Mask slot nếu ships < 5 |
| `tgt_idx` (0–39) | Index trong tất cả planets | Mask chính planet src |
| `frac_idx` (0–3) | 25% / 50% / 75% / 95% ships | Mask nếu result < 1 ship |

```python
FRACTIONS = [0.25, 0.50, 0.75, 0.95]

def get_action_masks(raw_planets, player: int) -> np.ndarray:
    """
    Trả về bool mask shape (10 + 40 + 4,) cho MaskablePPO.
    True = action hợp lệ.
    """
    planets  = [Planet(*p) for p in raw_planets]
    my_pl    = [p for p in planets if p.owner == player]

    # src mask: 10 slots
    src_mask = np.zeros(10, dtype=bool)
    for i, p in enumerate(my_pl[:10]):
        src_mask[i] = p.ships >= 5

    # tgt mask: tất cả planets đều có thể là target
    tgt_mask = np.ones(40, dtype=bool)

    # frac mask: luôn valid (decode sẽ clamp)
    frac_mask = np.ones(4, dtype=bool)

    return np.concatenate([src_mask, tgt_mask, frac_mask])

def decode_action(action, raw_planets, player: int, ang_vel: float) -> list:
    """Chuyển action indices → list of [from_planet_id, angle, ships]."""
    planets = [Planet(*p) for p in raw_planets]
    my_pl   = [p for p in planets if p.owner == player]
    if not my_pl:
        return []

    src_idx, tgt_idx, frac_idx = int(action[0]), int(action[1]), int(action[2])

    src = my_pl[src_idx % len(my_pl)]
    tgt = planets[tgt_idx % len(planets)]
    if tgt.id == src.id:
        return []

    ships = max(1, int(src.ships * FRACTIONS[frac_idx % 4]))
    ships = min(ships, src.ships)

    angle, tx, ty = intercept_angle(src, tgt, ships, ang_vel)

    if hits_sun(src.x, src.y, tx, ty):
        # Không skip — thử angle lệch ±15° để vòng qua sun
        for delta in [math.pi/12, -math.pi/12, math.pi/6, -math.pi/6]:
            a2 = angle + delta
            ex = src.x + 80 * math.cos(a2)
            ey = src.y + 80 * math.sin(a2)
            if not hits_sun(src.x, src.y, ex, ey):
                angle = a2
                break

    return [[src.id, angle, ships]]
```

---

## 5. Test local

```python
# Kiểm tra agent không crash, action hợp lệ
from kaggle_environments import make

env    = make("orbit_wars", configuration={"seed": 42}, debug=True)
result = env.run(["main.py", "random"])
final  = result[-1]
print(f"Player 0: {final[0].reward} | Player 1: {final[1].reward}")

# Xem replay (trong Kaggle notebook)
env.render(mode="ipython", width=800, height=600)
```
