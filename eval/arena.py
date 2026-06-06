"""
eval/arena.py — local round-robin tournament + head-to-head win rates.

run_match resolves each agent through env.orbit_env.resolve_opponent, so you
can pass builtin strings ("random"/"starter"), saved .zip model paths, or
callables interchangeably.
"""
import itertools
from kaggle_environments import make

from env.orbit_env import resolve_opponent


def _label(agent):
    return agent if isinstance(agent, str) else getattr(agent, "__name__", "agent")


def run_match(agent_a, agent_b, n=30, seed_start=0, verbose=True):
    """Win rate of agent_a vs agent_b over n seeded games (draws excluded from wins)."""
    a = resolve_opponent(agent_a)
    b = resolve_opponent(agent_b)
    wins = losses = draws = 0
    for i in range(n):
        env = make("orbit_wars", configuration={"seed": seed_start + i})
        result = env.run([a, b])
        r_a, r_b = result[-1][0].reward, result[-1][1].reward
        if r_a > r_b:
            wins += 1
        elif r_a < r_b:
            losses += 1
        else:
            draws += 1
    rate = wins / n if n else 0.0
    if verbose:
        print(f"{_label(agent_a)} vs {_label(agent_b)}: "
              f"{wins}W/{losses}L/{draws}D over {n} ({rate:.1%})")
    return rate


def round_robin(models: dict, n=30):
    """models: {name: path-or-agent}. Each unordered pair plays n games."""
    table = {}
    for a, b in itertools.combinations(models, 2):
        rate = run_match(models[a], models[b], n=n)
        table[(a, b)] = rate
    return table


if __name__ == "__main__":
    # Quick sanity: random vs random ~ 50%.
    run_match("random", "random", n=10)
