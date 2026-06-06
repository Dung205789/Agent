"""
env/reward.py — shaped reward, with a stage switch for ablation.

Component map (from architecture.md §3):
    A production_tick  — reward owning productive planets each turn
    B planet_ratio     — fraction of planets held
    C capture_bonus    — capturing a planet (scaled by its production)
    D lose_penalty     — losing a planet (scaled by its production)
    E ship_ratio       — fraction of all ships you own (aligns with win cond.)
    F terminal         — amplified +/- at game end

Stage 1 submission spec (task.md) enables ONLY A + F.
Stage 2+ enables the full A..F set.

The interpreter's terminal `kaggle_reward` is +1 (win / top score) or -1.
"""
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

WEIGHTS = {
    "production_tick": 0.002,   # A
    "planet_ratio":    0.010,   # B
    "capture_bonus":   0.100,   # C
    "lose_penalty":    0.150,   # D
    "ship_ratio":      0.020,   # E
    "terminal":        10.0,    # F
}

# Which components are active. "full" = A..F, "minimal" = A + F only.
_PROFILES = {
    "minimal": {"production_tick", "terminal"},
    "full": set(WEIGHTS.keys()),
}


def _planets(raw):
    if raw is None:
        return []
    pls = raw.get("planets", []) if hasattr(raw, "get") else getattr(raw, "planets", [])
    return [Planet(*p) for p in pls]


def compute_reward(prev_raw, curr_raw, player, done, kaggle_reward, profile="full") -> float:
    active = _PROFILES.get(profile, _PROFILES["full"])

    if curr_raw is None:
        # Episode ended without a follow-up observation.
        if done and "terminal" in active:
            return float(kaggle_reward) * WEIGHTS["terminal"]
        return 0.0

    prev_pl = _planets(prev_raw)
    curr_pl = _planets(curr_raw)

    reward = 0.0

    # A — production tick
    if "production_tick" in active:
        reward += sum(p.production for p in curr_pl if p.owner == player) * WEIGHTS["production_tick"]

    # B — planet ratio
    if "planet_ratio" in active:
        reward += (sum(1 for p in curr_pl if p.owner == player) / max(len(curr_pl), 1)) * WEIGHTS["planet_ratio"]

    # C & D — capture / loss (scaled by production value)
    if "capture_bonus" in active or "lose_penalty" in active:
        prev_mine = {p.id: p for p in prev_pl if p.owner == player}
        curr_mine = {p.id: p for p in curr_pl if p.owner == player}
        if "capture_bonus" in active:
            for pid in set(curr_mine) - set(prev_mine):
                reward += curr_mine[pid].production * WEIGHTS["capture_bonus"]
        if "lose_penalty" in active:
            for pid in set(prev_mine) - set(curr_mine):
                reward -= prev_mine[pid].production * WEIGHTS["lose_penalty"]

    # E — ship ratio
    if "ship_ratio" in active:
        my_ships = sum(p.ships for p in curr_pl if p.owner == player)
        all_ships = sum(p.ships for p in curr_pl) + 1e-6
        reward += (my_ships / all_ships) * WEIGHTS["ship_ratio"]

    # F — terminal
    if done and "terminal" in active:
        reward += float(kaggle_reward) * WEIGHTS["terminal"]

    return float(reward)
