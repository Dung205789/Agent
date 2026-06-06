# task.md — Orbit Wars: 5-Submission Roadmap (Pure PPO)

## Sản phẩm cuối cùng

> **`submission_notebook.ipynb`** — Kaggle Notebook chứa agent function,
> tự động package và submit lên leaderboard.
> Model weights được train local, upload lên Kaggle Dataset, notebook load vào.

---

## Nguyên tắc của dân chuyên

1. **Mỗi submission = 1 hypothesis** — biết trước tại sao nộp, biết đo gì sau khi nộp
2. **Local arena trước, leaderboard sau** — không bao giờ "submit rồi xem"
3. **Win rate quan trọng hơn ELO** — ELO nhiễu, win rate vs fixed opponent ổn định hơn
4. **Không thay đổi 2 thứ cùng lúc** — ablate từng component riêng

---

## Submission 1 — PPO Stage 1 (vs Random)

**Hypothesis**: Agent có học được gì không khi opponent là random?

**Target ELO**: 1000–1100

**Train**: 5M steps, 4 envs, opponent = `"random"`

### Checklist

**Setup (1 lần duy nhất)**
- [ ] `pip install "kaggle-environments>=1.28.0" sb3-contrib`
- [ ] `kaggle competitions download orbit-wars` — đọc starter kit
- [ ] Chạy game tay: `env.run(["random", "random"])` → in obs ra, confirm field names
- [ ] Confirm action format thực tế: `[from_planet_id, angle_rad, num_ships]`
- [ ] Viết `env/core.py` — copy từ `agents.md § 3`
- [ ] Viết `env/orbit_env.py` — copy từ `architecture.md § 2`
- [ ] Viết `env/reward.py` — bật **chỉ component A + F** (production + terminal)

**Train**
- [ ] Chạy `train_stage1()` — 5M steps
- [ ] Mở TensorBoard: `tensorboard --logdir tb_logs/`
- [ ] Confirm `ep_rew_mean` tăng qua 2M steps đầu (nếu phẳng: reward function sai)
- [ ] Win rate vs random (local, 20 games): **phải > 70%**

**Submission notebook — S1**
```python
# Cell 1: Load model
model = MaskablePPO.load("/kaggle/input/orbit-wars-weights/stage1_random.zip")

# Cell 2: %%writefile main.py  (full agent code từ agents.md)

# Cell 3: Test 20 games vs random
# assert win_rate >= 0.70

# Cell 4: Package + submit
!tar -czf submission.tar.gz main.py
!kaggle competitions submit orbit-wars -f submission.tar.gz -m "S1 PPO stage1 vs random 5M"
```

**Sau khi nộp S1**
- [ ] Ghi lại ELO ban đầu vào `results.log`
- [ ] Download 2 replay: 1 game thắng, 1 game thua
- [ ] Xem replay: agent có pattern gì? Có bắn vào sun không?

```bash
kaggle competitions episodes <SUBMISSION_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays/
kaggle competitions logs <EPISODE_ID> 0
```

---

## Submission 2 — PPO Stage 2 (vs Rule-Based + Reward Upgrade)

**Hypothesis**: Orbital intercept prediction + đầy đủ reward có tăng ELO không?

**Target ELO**: 1150–1300 (+100 so với S1)

**Train**: Fine-tune từ S1, 15M steps, opponent = stage1 model

### Checklist

**Reward upgrade**
- [ ] Bật đầy đủ A + B + C + D + E + F trong `reward.py`
- [ ] Chạy 5 game in reward per component — xác nhận C (capture) > 30% tổng reward
- [ ] Nếu C quá thấp: tăng `capture_bonus` weight

**Observation upgrade**
- [ ] Confirm `predict_pos()` cho orbiting planets hoạt động đúng
- [ ] Test: print predicted pos vs actual pos sau 20 turns cho 1 fast-orbit planet

**Train Stage 2**
```python
model = MaskablePPO.load("models/stage1_random", env=vec)
model.learning_rate = 1e-4     # nhỏ hơn khi fine-tune
model.ent_coef      = 0.01
model.learn(15_000_000, reset_num_timesteps=False)
model.save("models/stage2_rulebased")
```
- [ ] Monitor: win rate vs stage1 tăng qua từng 3M steps checkpoint
- [ ] Nếu win rate plateau < 55% sau 8M steps → tăng ent_coef lên 0.03 để explore thêm

**Ablation** (làm trước submit)
- [ ] So sánh: model với predicted pos vs current pos (chỉ đổi 1 feature)
- [ ] Nếu predicted pos không giúp: kiểm tra `angular_velocity` field tên đúng chưa

**Local arena**
- [ ] `run_match("models/stage2.zip", "models/stage1.zip", n=30)` → **win rate > 60%**
- [ ] `run_match("models/stage2.zip", "random", n=30)` → **win rate > 85%**

**Submission notebook — S2**
```python
# Thay model path → stage2_rulebased.zip
# assert win_rate >= 0.80 (nâng threshold)
!kaggle competitions submit orbit-wars -f submission.tar.gz -m "S2 PPO stage2 orbital+reward 15M"
```

---

## Submission 3 — Self-Play Round 1 (3 rounds × 3M)

**Hypothesis**: Self-play có dạy agent counter-strategy không?

**Target ELO**: 1300–1450 (+150 so với S2)

**Train**: Fine-tune từ S2, 9M steps self-play (3 rounds)

### Checklist

**Setup self-play**
- [ ] `train/selfplay.py` — SelfPlayPool với seed = stage2 model
- [ ] Verify: pool sampling trả về valid model path, không crash

**Train Stage 3 (3 rounds)**
```python
pool = SelfPlayPool(seed="models/stage2_rulebased.zip")
model = MaskablePPO.load("models/stage2_rulebased")
model.learning_rate = 5e-5
model.ent_coef      = 0.005

for round_i in range(3):      # tổng 9M steps
    opp = pool.sample()
    vec = SubprocVecEnv([lambda: make_masked_env(opp)] * N_ENVS)
    model.set_env(vec)
    model.learn(3_000_000, reset_num_timesteps=False)
    pool.save(model, round_i)
```

**Kiểm tra sau mỗi round**
- [ ] Round 1: `run_match(current, stage2, n=20)` → win rate > 52%
- [ ] Round 2: win rate > 54%
- [ ] Round 3: win rate > 56%
- [ ] Nếu round nào win rate < 50% → self-play đang collapse, tăng pool size hoặc dùng lại stage2

**Degenerate strategy check**
- [ ] Xem replay: agent có chỉ camp/farm mà không attack không?
- [ ] Nếu camp: tăng `capture_bonus` weight, giảm `production_tick`

**Submission notebook — S3**
```python
# Model: stage3_selfplay_r3.zip
# Test: 30 games (nâng từ 20 lên 30 để giảm variance)
# assert win_rate >= 0.80 vs random, >= 0.58 vs stage2
!kaggle competitions submit orbit-wars -f submission.tar.gz -m "S3 self-play 3 rounds 9M"
```

---

## Submission 4 — Self-Play Round 2 (5 rounds × 3M) + Feature Upgrade

**Hypothesis**: Thêm enemy threat feature + thêm self-play rounds có cải thiện tiếp không?

**Target ELO**: 1450–1550 (+100 so với S3)

**Train**: Fine-tune từ S3, thêm 5 rounds × 3M = 15M steps

### Checklist

**Feature upgrade** (thêm vào observation)
- [ ] Thêm `enemy_incoming_ships_per_planet`: tổng ships của enemy fleet đang bay đến mỗi planet
  ```python
  # Trong encode_obs: thêm vào planet feature
  incoming_enemy = sum(
      f.ships for f in fleets
      if f.owner != player
      and abs(math.atan2(p.y - f.y, p.x - f.x) - f.angle) < 0.3  # hướng về planet này
  )
  planet_feat.append(min(1.0, incoming_enemy / 200.0))
  ```
- [ ] OBS_DIM tăng từ 625 → 665 (thêm 1 feature × 40 planets)
- [ ] Phải train lại từ đầu nếu thay đổi obs shape → fine-tune không được, train stage2 lại

**Train thêm 5 rounds self-play**
- [ ] Tiếp tục từ pool của S3 (không reset)
- [ ] Round 4–8: monitor win rate vs S3 model sau mỗi round

**Nếu obs upgrade quá tốn time**
- [ ] Bỏ feature upgrade, chỉ train thêm self-play rounds
- [ ] Ưu tiên ổn định hơn là thêm feature

**Local arena (bắt buộc)**
- [ ] `run_match(S4, S3, n=50)` → **win rate > 55%** (50 games để giảm variance)
- [ ] `run_match(S4, stage2, n=50)` → win rate > 65%

**Submission notebook — S4**
```python
# 50 game local test trước submit (không phải 20)
assert win_rate_vs_random >= 0.85
assert win_rate_vs_S3     >= 0.55
!kaggle competitions submit orbit-wars -f submission.tar.gz -m "S4 self-play 8 rounds + threat feature"
```

---

## Submission 5 — Best Model Selection + Final Hardening

**Hypothesis**: Model nào trong S1–S4 cho ELO cao và ổn định nhất?

**Target ELO**: Giữ hoặc vượt S4

**Không train thêm** — focus vào chọn và ổn định

### Checklist

**Tournament nội bộ**
```python
# eval/arena.py
models = {
    "S1": "models/stage1_random.zip",
    "S2": "models/stage2_rulebased.zip",
    "S3": "models/stage3_selfplay_r3.zip",
    "S4": "models/stage3_selfplay_r8.zip",
}

# Round-robin: mỗi cặp 30 games
for a in models:
    for b in models:
        if a >= b: continue
        run_match(models[a], models[b], n=30)
```
- [ ] Chọn model có **win rate cao nhất và variance thấp nhất**
- [ ] Ưu tiên model ổn định hơn model peak nếu chênh nhau < 5% win rate

**Hardening submission notebook**
- [ ] Thêm validation trong notebook: nếu model load fail → raise rõ lỗi (không silent)
- [ ] Test với `remainingOverageTime = 0.5` (budget cạn kiệt) — phải trả về action hợp lệ hoặc `[]`
- [ ] Test với map 4-player (nếu leaderboard có 4p format)
- [ ] Chạy 50 games local, **đếm exception = 0**

**Packaging final**
```bash
# Đảm bảo submission.tar.gz chỉ chứa main.py ở root
tar -czf final.tar.gz main.py
tar -tf final.tar.gz    # verify: chỉ có ./main.py

# KHÔNG include:
# - env/ (phải inline tất cả vào main.py)
# - model weights (phải load từ /kaggle/input/...)
# - requirements.txt (Kaggle env tự quản lý)
```

> **Quan trọng**: Toàn bộ code (`encode_obs`, `decode_action`, physics helpers)
> phải được **inline trực tiếp vào `main.py`** — không import từ file khác.
> Kaggle chỉ nhìn thấy file trong `submission.tar.gz`, không nhìn thấy `env/`.

**Submission notebook — S5**
```python
# Thêm timing test
import time
for seed in range(10):
    env = make("orbit_wars", configuration={"seed": seed})
    env.run(["main.py", "random"])

# Kiểm tra log không có timeout
!kaggle competitions logs <EPISODE_ID> 0 | grep -i "timeout\|error\|exception"

!kaggle competitions submit orbit-wars -f final.tar.gz -m "S5 FINAL best model hardened"
```

---

## results.log — cập nhật sau mỗi submission

```
| Sub | Model           | Steps | Opponent        | Local WR | Kaggle ELO | Notes                    |
|-----|-----------------|-------|-----------------|----------|------------|--------------------------|
| S1  | stage1_random   |  5M   | random          | ???%     | ???        | First PPO                |
| S2  | stage2_rulebased| 15M   | stage1          | ???%     | ???        | Orbital + full reward    |
| S3  | selfplay_r3     |  9M   | self-pool       | ???%     | ???        | Self-play 3 rounds       |
| S4  | selfplay_r8     | 15M   | self-pool       | ???%     | ???        | Self-play 8 rounds       |
| S5  | best of above   |  -    | -               | ???%     | ???        | Final hardened           |
```

---

## Main.py template (inline — dùng cho submission notebook)

```python
# %%writefile main.py
"""
Orbit Wars — Pure PPO Agent
Model: MaskablePPO (sb3-contrib)
Weights: /kaggle/input/orbit-wars-weights/best_model.zip
"""
import os, math
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet
from sb3_contrib import MaskablePPO

# ── Constants ─────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
MAX_SPEED = 6.0
N_PLANETS, N_FLEETS = 40, 20
FRACTIONS = [0.25, 0.50, 0.75, 0.95]

# ── Physics ───────────────────────────────────────────────────
def fleet_speed(n):
    if n <= 1: return 1.0
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5

def orbital_radius(p): return math.hypot(p.x - SUN_X, p.y - SUN_Y)
def is_orbiting(p):    return (orbital_radius(p) + p.radius) < 50.0

def predict_pos(p, av, t):
    if not is_orbiting(p): return p.x, p.y
    r = orbital_radius(p)
    a = math.atan2(p.y - SUN_Y, p.x - SUN_X) + av * t
    return SUN_X + r * math.cos(a), SUN_Y + r * math.sin(a)

def intercept_angle(src, tgt, ships, av):
    tx, ty = tgt.x, tgt.y
    for _ in range(8):
        d = math.hypot(tx - src.x, ty - src.y)
        t = max(1, math.ceil(d / fleet_speed(ships)))
        tx, ty = predict_pos(tgt, av, t)
    return math.atan2(ty - src.y, tx - src.x), tx, ty

def hits_sun(x1, y1, x2, y2):
    dx, dy = x2-x1, y2-y1
    den = dx*dx + dy*dy
    if den < 1e-9: return math.hypot(x1-SUN_X, y1-SUN_Y) < SUN_R
    t = max(0.0, min(1.0, ((SUN_X-x1)*dx + (SUN_Y-y1)*dy)/den))
    return math.hypot(x1+t*dx-SUN_X, y1+t*dy-SUN_Y) < SUN_R

# ── Observation ───────────────────────────────────────────────
def encode_obs(raw):
    planets   = [Planet(*p) for p in raw.get("planets", [])]
    fleets    = [Fleet(*f)  for f in raw.get("fleets",  [])]
    player    = raw.get("player", 0)
    av        = raw.get("angular_velocity", 0.03)
    step      = raw.get("step", 0)
    cids      = set(raw.get("comet_planet_ids", []))
    my_pl     = [p for p in planets if p.owner == player]
    ref       = max((p.ships for p in my_pl), default=50)

    def enc_p(p):
        d = math.hypot(p.x-SUN_X, p.y-SUN_Y)
        t = max(1, int(d / fleet_speed(ref)))
        px, py = predict_pos(p, av, t)
        r = orbital_radius(p)
        return [
            (px-SUN_X)/50, (py-SUN_Y)/50,
            math.sin(math.atan2(p.y-SUN_Y, p.x-SUN_X)),
            math.cos(math.atan2(p.y-SUN_Y, p.x-SUN_X)),
            min(1, r/50), min(1, p.ships/500), p.production/5,
            float(p.owner==player), float(p.owner not in (-1,player)),
            float(p.owner==-1), float(is_orbiting(p)), float(p.id in cids),
        ]

    pf = [v for i in range(N_PLANETS) for v in (enc_p(planets[i]) if i < len(planets) else [0]*12)]
    ff = []
    for f in (sorted(fleets, key=lambda x:-x.ships)+[None]*N_FLEETS)[:N_FLEETS]:
        ff += [(f.x-SUN_X)/50,(f.y-SUN_Y)/50,math.sin(f.angle),math.cos(f.angle),
               min(1,f.ships/500),float(f.owner==player),float(f.owner!=player)] if f else [0]*7
    tp = max(len(planets),1)
    gf = [step/500, sum(1 for p in planets if p.owner==player)/tp,
          sum(p.ships for p in planets if p.owner==player)/(sum(p.ships for p in planets)+1e-6),
          sum(p.production for p in planets if p.owner==player)/(sum(p.production for p in planets)+1e-6),
          raw.get("remainingOverageTime",60)/60]
    return np.array(pf+ff+gf, dtype=np.float32)

# ── Action Masking ────────────────────────────────────────────
def get_masks(raw_planets, player):
    planets = [Planet(*p) for p in raw_planets]
    my_pl   = [p for p in planets if p.owner == player]
    src_m   = np.zeros(10, dtype=bool)
    for i, p in enumerate(my_pl[:10]):
        src_m[i] = p.ships >= 5
    return np.concatenate([src_m, np.ones(40, bool), np.ones(4, bool)])

# ── Action Decode ─────────────────────────────────────────────
def decode_action(action, raw_planets, player, av):
    planets = [Planet(*p) for p in raw_planets]
    my_pl   = [p for p in planets if p.owner == player]
    if not my_pl: return []
    src = my_pl[int(action[0]) % len(my_pl)]
    tgt = planets[int(action[1]) % len(planets)]
    if tgt.id == src.id: return []
    ships = max(1, min(int(src.ships * FRACTIONS[int(action[2])%4]), src.ships))
    angle, tx, ty = intercept_angle(src, tgt, ships, av)
    if hits_sun(src.x, src.y, tx, ty):
        for d in [math.pi/12, -math.pi/12, math.pi/6, -math.pi/6]:
            a2 = angle + d
            ex, ey = src.x+80*math.cos(a2), src.y+80*math.sin(a2)
            if not hits_sun(src.x, src.y, ex, ey): angle = a2; break
    return [[src.id, angle, ships]]

# ── Model Loading ─────────────────────────────────────────────
_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        path = "/kaggle/input/orbit-wars-weights/best_model.zip"
        _MODEL = MaskablePPO.load(path, device="cpu")
    return _MODEL

# ── Agent Entry Point ─────────────────────────────────────────
def agent(obs, config=None):
    raw = obs if isinstance(obs, dict) else vars(obs)
    planets = raw.get("planets", [])
    player  = raw.get("player", 0)
    av      = raw.get("angular_velocity", 0.03)

    obs_vec = encode_obs(raw)
    masks   = get_masks(planets, player)
    action, _ = _get_model().predict(obs_vec, action_masks=masks, deterministic=True)
    return decode_action(action, planets, player, av)
```
