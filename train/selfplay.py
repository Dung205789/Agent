"""
train/selfplay.py — ELO-ish self-play opponent pool.

Keeps the most recent snapshots and samples an opponent for each round:
60% newest, 40% uniform across the pool. Paths point at saved .zip models
(consumed by env.orbit_env.make_model_agent via resolve_opponent).
"""
import os
import random


class SelfPlayPool:
    MAX_SIZE = 6

    def __init__(self, seed=None, dir="snapshots/"):
        os.makedirs(dir, exist_ok=True)
        self.dir = dir
        self.paths = [seed] if seed else []

    def save(self, model, round_i):
        path = os.path.join(self.dir, f"round_{round_i:03d}.zip")
        model.save(path)
        self.paths.append(path)
        if len(self.paths) > self.MAX_SIZE:
            self.paths.pop(0)
        return path

    def sample(self) -> str:
        if not self.paths:
            return "random"
        if random.random() < 0.6:
            return self.paths[-1]            # newest
        return random.choice(self.paths)     # uniform from pool
