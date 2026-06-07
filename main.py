"""
Orbit Wars — Pure PPO Agent (self-contained submission file).

Everything (physics, observation encoding, action masking/decoding, model
loading, agent entry point) is inlined here. Kaggle only sees this file inside
submission.tar.gz — no imports from env/.

Model weights are loaded from an attached Kaggle Dataset. Set OW_MODEL_PATH to
override the path (used for local testing).
"""
import os
import sys
import types
import math
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# Make the sb3 import robust on any image: stable-baselines3 unconditionally
# does `from torch.utils.tensorboard import SummaryWriter`, which on some images
# drags in a broken tensorflow/protobuf pair and raises a non-ImportError. Probe
# it once and stub it if broken so the agent still loads. No-op on clean images.
try:
    import torch.utils.tensorboard  # noqa: F401
except Exception:
    _tb = types.ModuleType("torch.utils.tensorboard")
    class SummaryWriter:  # noqa: N801
        def __init__(self, *a, **k):
            raise RuntimeError("tensorboard unavailable (stub)")
    _tb.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = _tb

from sb3_contrib import MaskablePPO

# ── Constants ─────────────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0
N_PLANETS, N_FLEETS = 40, 20
FRACTIONS = [0.25, 0.50, 0.75, 0.95]
MIN_LAUNCH_SHIPS = 5

# ── Physics ───────────────────────────────────────────────────────────
def fleet_speed(n):
    if n <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)

def orbital_radius(p):
    return math.hypot(p.x - SUN_X, p.y - SUN_Y)

def is_orbiting(p):
    return (orbital_radius(p) + p.radius) < ROTATION_RADIUS_LIMIT

def predict_pos(p, av, t):
    if not is_orbiting(p):
        return p.x, p.y
    r = orbital_radius(p)
    a = math.atan2(p.y - SUN_Y, p.x - SUN_X) + av * t
    return SUN_X + r * math.cos(a), SUN_Y + r * math.sin(a)

def intercept_angle(src, tgt, ships, av):
    tx, ty = tgt.x, tgt.y
    spd = fleet_speed(ships)
    for _ in range(8):
        d = math.hypot(tx - src.x, ty - src.y)
        t = max(1, math.ceil(d / spd))
        tx, ty = predict_pos(tgt, av, t)
    return math.atan2(ty - src.y, tx - src.x), tx, ty

def hits_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    den = dx * dx + dy * dy
    if den < 1e-9:
        return math.hypot(x1 - SUN_X, y1 - SUN_Y) < SUN_R
    t = max(0.0, min(1.0, ((SUN_X - x1) * dx + (SUN_Y - y1) * dy) / den))
    return math.hypot(x1 + t * dx - SUN_X, y1 + t * dy - SUN_Y) < SUN_R

# ── Obs helpers ───────────────────────────────────────────────────────
def _get(raw, key, default):
    if isinstance(raw, dict):
        return raw.get(key, default)
    if hasattr(raw, "get"):
        try:
            return raw.get(key, default)
        except TypeError:
            pass
    return getattr(raw, key, default)

# ── Observation ───────────────────────────────────────────────────────
def encode_obs(raw):
    planets = [Planet(*p) for p in _get(raw, "planets", [])]
    fleets = [Fleet(*f) for f in _get(raw, "fleets", [])]
    player = _get(raw, "player", 0)
    av = _get(raw, "angular_velocity", 0.03)
    step = _get(raw, "step", 0)
    cids = set(_get(raw, "comet_planet_ids", []))
    my_pl = [p for p in planets if p.owner == player]
    ref = max((p.ships for p in my_pl), default=50)

    def enc_p(p):
        d = math.hypot(p.x - SUN_X, p.y - SUN_Y)
        t = max(1, int(d / fleet_speed(ref)))
        px, py = predict_pos(p, av, t)
        ca = math.atan2(p.y - SUN_Y, p.x - SUN_X)
        r = orbital_radius(p)
        return [
            (px - SUN_X) / 50.0, (py - SUN_Y) / 50.0,
            math.sin(ca), math.cos(ca),
            min(1.0, r / 50.0), min(1.0, p.ships / 500.0), p.production / 5.0,
            float(p.owner == player), float(p.owner not in (-1, player)),
            float(p.owner == -1), float(is_orbiting(p)), float(p.id in cids),
        ]

    pf = []
    for i in range(N_PLANETS):
        pf.extend(enc_p(planets[i]) if i < len(planets) else [0.0] * 12)

    ff = []
    top = sorted(fleets, key=lambda x: -x.ships)[:N_FLEETS]
    for i in range(N_FLEETS):
        if i < len(top):
            f = top[i]
            ff.extend([(f.x - SUN_X) / 50.0, (f.y - SUN_Y) / 50.0,
                       math.sin(f.angle), math.cos(f.angle),
                       min(1.0, f.ships / 500.0),
                       float(f.owner == player), float(f.owner != player)])
        else:
            ff.extend([0.0] * 7)

    tp = max(len(planets), 1)
    gf = [
        step / 500.0,
        sum(1 for p in planets if p.owner == player) / tp,
        sum(p.ships for p in planets if p.owner == player) / (sum(p.ships for p in planets) + 1e-6),
        sum(p.production for p in planets if p.owner == player) / (sum(p.production for p in planets) + 1e-6),
        _get(raw, "remainingOverageTime", 60.0) / 60.0,
    ]
    return np.clip(np.array(pf + ff + gf, dtype=np.float32), -2.0, 2.0)

# ── Action Masking ────────────────────────────────────────────────────
def get_masks(raw_planets, player):
    planets = [Planet(*p) for p in raw_planets]
    my_pl = [p for p in planets if p.owner == player]
    src_m = np.zeros(10, dtype=bool)
    for i, p in enumerate(my_pl[:10]):
        src_m[i] = p.ships >= MIN_LAUNCH_SHIPS
    if not src_m.any():
        if my_pl:
            best = max(range(min(len(my_pl), 10)), key=lambda i: my_pl[i].ships)
            src_m[best] = True
        else:
            src_m[0] = True
    return np.concatenate([src_m, np.ones(40, bool), np.ones(4, bool)])

# ── Action Decode ─────────────────────────────────────────────────────
def decode_action(action, raw_planets, player, av):
    planets = [Planet(*p) for p in raw_planets]
    my_pl = [p for p in planets if p.owner == player]
    if not my_pl or not planets:
        return []
    src = my_pl[int(action[0]) % len(my_pl)]
    tgt = planets[int(action[1]) % len(planets)]
    if tgt.id == src.id:
        return []
    ships = max(1, min(int(src.ships * FRACTIONS[int(action[2]) % 4]), src.ships))
    if ships <= 0:
        return []
    angle, tx, ty = intercept_angle(src, tgt, ships, av)
    if hits_sun(src.x, src.y, tx, ty):
        for d in (math.pi / 12, -math.pi / 12, math.pi / 6, -math.pi / 6):
            a2 = angle + d
            ex, ey = src.x + 80 * math.cos(a2), src.y + 80 * math.sin(a2)
            if not hits_sun(src.x, src.y, ex, ey):
                angle = a2
                break
    return [[src.id, float(angle), int(ships)]]

# ── Model Loading ─────────────────────────────────────────────────────
_MODEL = None
# NOTE: kaggle execs a file agent without __file__ defined, so never reference
# __file__ at module level here.
_CANDIDATE_PATHS = [
    os.environ.get("OW_MODEL_PATH", ""),
    "/kaggle_simulations/agent/best_model.zip",   # tar.gz bundle extract dir
    "best_model.zip",                              # cwd fallback
    "/kaggle/input/orbit-wars-weights/best_model.zip",
    "/kaggle/input/orbit-wars-weights/stage3_selfplay.zip",
    "/kaggle/input/orbit-wars-weights/stage1_random.zip",
]

def _get_model():
    global _MODEL
    if _MODEL is None:
        for path in _CANDIDATE_PATHS:
            if path and os.path.exists(path):
                _MODEL = MaskablePPO.load(path, device="cpu")
                return _MODEL
        raise FileNotFoundError(
            "No model weights found. Tried: "
            + ", ".join(p for p in _CANDIDATE_PATHS if p)
        )
    return _MODEL

# ── Agent Entry Point ─────────────────────────────────────────────────
def agent(obs, config=None):
    planets = _get(obs, "planets", [])
    player = _get(obs, "player", 0)
    av = _get(obs, "angular_velocity", 0.03)
    vec = encode_obs(obs)
    masks = get_masks(planets, player)
    action, _ = _get_model().predict(vec, action_masks=masks, deterministic=True)
    return decode_action(action, planets, player, av)
