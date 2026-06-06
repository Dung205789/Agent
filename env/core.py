"""
env/core.py — physics, observation encoding, action decoding, masking.

Single source of truth shared by the Gymnasium wrapper (training) and the
submission `main.py` (inference). Everything here is verified against the real
`kaggle_environments.envs.orbit_wars` interpreter:

    Planet fields : (id, owner, x, y, radius, ships, production)
    Fleet  fields : (id, owner, x, y, angle, from_planet_id, ships)
    Sun           : center (50, 50), radius 10
    Orbit rule    : a planet rotates iff orbital_radius + radius < 50
    Fleet speed   : 1 + (S-1)*(log(ships)/log(1000))**1.5, capped at shipSpeed(=6)
    angular_velocity: per-episode, uniform in [0.025, 0.05]  (read from obs)
    Action        : list of [from_planet_id, angle_rad, num_ships]
"""
import math
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# ── Constants (mirror the interpreter) ────────────────────────────────
SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0                 # configuration.shipSpeed default
N_PLANETS, N_FLEETS = 40, 20
FRACTIONS = [0.25, 0.50, 0.75, 0.95]

OBS_DIM = 12 * N_PLANETS + 7 * N_FLEETS + 5      # 480 + 140 + 5 = 625
ACTION_NVEC = [10, 40, 4]                         # MultiDiscrete([src, tgt, frac])
MIN_LAUNCH_SHIPS = 5                              # mask src planets below this


# ── Physics ───────────────────────────────────────────────────────────
def fleet_speed(n: int) -> float:
    if n <= 1:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(n) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)


def orbital_radius(p) -> float:
    return math.hypot(p.x - SUN_X, p.y - SUN_Y)


def is_orbiting(p) -> bool:
    return (orbital_radius(p) + p.radius) < ROTATION_RADIUS_LIMIT


def predict_pos(p, ang_vel: float, t: int):
    """Position of planet p after t turns. Static planets do not move."""
    if not is_orbiting(p):
        return p.x, p.y
    r = orbital_radius(p)
    a = math.atan2(p.y - SUN_Y, p.x - SUN_X) + ang_vel * t
    return SUN_X + r * math.cos(a), SUN_Y + r * math.sin(a)


def intercept_angle(src, tgt, ships: int, ang_vel: float):
    """Iteratively solve the firing angle to intercept a (possibly) rotating
    target. Returns (angle, predicted_tx, predicted_ty)."""
    tx, ty = tgt.x, tgt.y
    spd = fleet_speed(ships)
    for _ in range(8):
        dist = math.hypot(tx - src.x, ty - src.y)
        t = max(1, math.ceil(dist / spd))
        tx, ty = predict_pos(tgt, ang_vel, t)
    return math.atan2(ty - src.y, tx - src.x), tx, ty


def hits_sun(x1, y1, x2, y2) -> bool:
    """True if segment (x1,y1)->(x2,y2) passes within SUN_R of the sun."""
    dx, dy = x2 - x1, y2 - y1
    den = dx * dx + dy * dy
    if den < 1e-9:
        return math.hypot(x1 - SUN_X, y1 - SUN_Y) < SUN_R
    t = max(0.0, min(1.0, ((SUN_X - x1) * dx + (SUN_Y - y1) * dy) / den))
    return math.hypot(x1 + t * dx - SUN_X, y1 + t * dy - SUN_Y) < SUN_R


# ── Observation encoding ──────────────────────────────────────────────
def _raw_get(raw, key, default):
    """raw may be a dict, kaggle Struct, or SimpleNamespace."""
    if isinstance(raw, dict):
        return raw.get(key, default)
    if hasattr(raw, "get"):
        try:
            return raw.get(key, default)
        except TypeError:
            pass
    return getattr(raw, key, default)


def _encode_planet(p, player, ang_vel, comet_ids, ref_ships) -> list:
    dist = math.hypot(p.x - SUN_X, p.y - SUN_Y)
    t_est = max(1, int(dist / fleet_speed(ref_ships)))
    px, py = predict_pos(p, ang_vel, t_est)
    cur_ang = math.atan2(p.y - SUN_Y, p.x - SUN_X)
    r = orbital_radius(p)
    return [
        (px - SUN_X) / 50.0,                       # predicted rel_x
        (py - SUN_Y) / 50.0,                       # predicted rel_y
        math.sin(cur_ang),                         # current sin_angle
        math.cos(cur_ang),                         # current cos_angle
        min(1.0, r / 50.0),                        # orbit radius (norm)
        min(1.0, p.ships / 500.0),                 # ships (norm)
        p.production / 5.0,                        # production (norm)
        1.0 if p.owner == player else 0.0,         # mine
        1.0 if p.owner not in (-1, player) else 0.0,  # enemy
        1.0 if p.owner == -1 else 0.0,             # neutral
        1.0 if is_orbiting(p) else 0.0,            # orbiting flag
        1.0 if p.id in comet_ids else 0.0,         # comet flag
    ]


def encode_obs(raw_obs) -> np.ndarray:
    planets = [Planet(*p) for p in _raw_get(raw_obs, "planets", [])]
    fleets = [Fleet(*f) for f in _raw_get(raw_obs, "fleets", [])]
    player = _raw_get(raw_obs, "player", 0)
    ang_vel = _raw_get(raw_obs, "angular_velocity", 0.03)
    step = _raw_get(raw_obs, "step", 0)
    comet_ids = set(_raw_get(raw_obs, "comet_planet_ids", []))

    my_pl = [p for p in planets if p.owner == player]
    ref_ships = max((p.ships for p in my_pl), default=50)

    # Planet features — fixed N_PLANETS slots
    pl_feats = []
    for i in range(N_PLANETS):
        if i < len(planets):
            pl_feats.extend(_encode_planet(planets[i], player, ang_vel, comet_ids, ref_ships))
        else:
            pl_feats.extend([0.0] * 12)

    # Fleet features — top N_FLEETS by ship count
    fl_feats = []
    top_fleets = sorted(fleets, key=lambda x: -x.ships)[:N_FLEETS]
    for i in range(N_FLEETS):
        if i < len(top_fleets):
            f = top_fleets[i]
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

    # Global features
    total_pl = max(len(planets), 1)
    my_ships = sum(p.ships for p in planets if p.owner == player)
    all_ships = sum(p.ships for p in planets) + 1e-6
    my_prod = sum(p.production for p in planets if p.owner == player)
    all_prod = sum(p.production for p in planets) + 1e-6
    global_feats = [
        step / 500.0,
        sum(1 for p in planets if p.owner == player) / total_pl,
        my_ships / all_ships,
        my_prod / all_prod,
        _raw_get(raw_obs, "remainingOverageTime", 60.0) / 60.0,
    ]

    vec = np.array(pl_feats + fl_feats + global_feats, dtype=np.float32)
    # Defensive clip — Box space is [-2, 2]
    return np.clip(vec, -2.0, 2.0)


# ── Action masking ────────────────────────────────────────────────────
def get_action_masks(raw_planets, player: int) -> np.ndarray:
    """Bool mask of shape (10 + 40 + 4,). True == valid choice.

    Guarantees at least one valid src so MaskablePPO never sees an empty
    discrete dimension (which would raise)."""
    planets = [Planet(*p) for p in raw_planets]
    my_pl = [p for p in planets if p.owner == player]

    src_mask = np.zeros(10, dtype=bool)
    for i, p in enumerate(my_pl[:10]):
        src_mask[i] = p.ships >= MIN_LAUNCH_SHIPS

    if not src_mask.any():
        # Fallback: pick the planet with the most ships (or slot 0 if none).
        if my_pl:
            best = max(range(min(len(my_pl), 10)), key=lambda i: my_pl[i].ships)
            src_mask[best] = True
        else:
            src_mask[0] = True

    tgt_mask = np.ones(40, dtype=bool)
    frac_mask = np.ones(4, dtype=bool)
    return np.concatenate([src_mask, tgt_mask, frac_mask])


# ── Action decoding ───────────────────────────────────────────────────
def decode_action(action, raw_planets, player: int, ang_vel: float) -> list:
    """Map MultiDiscrete indices -> kaggle action [[from_id, angle, ships]]."""
    planets = [Planet(*p) for p in raw_planets]
    my_pl = [p for p in planets if p.owner == player]
    if not my_pl or not planets:
        return []

    src_idx, tgt_idx, frac_idx = int(action[0]), int(action[1]), int(action[2])
    src = my_pl[src_idx % len(my_pl)]
    tgt = planets[tgt_idx % len(planets)]
    if tgt.id == src.id:
        return []

    ships = max(1, int(src.ships * FRACTIONS[frac_idx % 4]))
    ships = min(ships, src.ships)
    if ships <= 0:
        return []

    angle, tx, ty = intercept_angle(src, tgt, ships, ang_vel)

    # If the straight intercept line clips the sun, nudge the angle to arc past it.
    if hits_sun(src.x, src.y, tx, ty):
        for delta in (math.pi / 12, -math.pi / 12, math.pi / 6, -math.pi / 6):
            a2 = angle + delta
            ex = src.x + 80 * math.cos(a2)
            ey = src.y + 80 * math.sin(a2)
            if not hits_sun(src.x, src.y, ex, ey):
                angle = a2
                break

    return [[src.id, float(angle), int(ships)]]
