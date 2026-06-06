# Orbit Wars — Pure PPO Agent

MaskablePPO (sb3-contrib) agent for the Kaggle `orbit_wars` environment.
Deliverable: `submission_notebook.ipynb` (trains locally / on Kaggle, uploads
weights as a Dataset, writes an inlined `main.py`, packages + submits).

## Layout
```
env/      core.py (physics/encode/decode/masks), reward.py, orbit_env.py
train/    curriculum.py (3 stages), selfplay.py (ELO pool), train.py (CLI)
eval/     arena.py (round-robin / win-rate), tracker.py (ELO + results.log)
models/   saved .zip weights
submission_notebook.ipynb
```

## Verified against the real environment
- Planet `(id, owner, x, y, radius, ships, production)`; Fleet `(id, owner, x, y, angle, from_planet_id, ships)`
- Sun at (50,50) r=10; planet orbits iff `orbital_radius + radius < 50`
- `angular_velocity` is per-episode random in `[0.025, 0.05]` — always read from obs
- Action: `[[from_planet_id, angle_rad, num_ships], ...]`
- Config: `episodeSteps=500`, `actTimeout=1s`, `agentTimeout=2s`, `shipSpeed=6`, `cometSpeed=4`
- Terminal `kaggle_reward` is +1 (top score) / -1

> Note: the opponent for self-play / arena is a saved `.zip`; we wrap it into a
> kaggle callable via `make_model_agent` (kaggle's `env.train` only accepts
> builtin strings or callables, not a `.zip` path directly).

## Quickstart
```bash
pip install -r requirements.txt          # or newer locally
# fast end-to-end regression (train tiny + run main.py in real env):
python -m tests.smoke
# train:
python -m train.train --stage 1 --steps 20000 --envs 2 --no-subproc   # smoke
python -m train.train --stage 1                                        # full 5M
python -m train.train --stage 2
python -m train.train --stage 3 --rounds 3
# evaluate:
python -m eval.arena
```

## Deliverable
`submission_notebook.ipynb` is generated from the tested repo sources by
`python _build_notebook.py` (keeps notebook code in sync with `env/`, `train/`,
`main.py`). It writes the package, trains, inlines `main.py`, gates on win-rate,
packages `submission.tar.gz`, and submits.

## tensorboard auto-compat
`env/tb_compat.py` probes `from torch.utils.tensorboard import SummaryWriter`
once at import time. On broken images (SB3 drags in a tensorflow/protobuf pair
that raises), it registers a stub so SB3 imports cleanly and `PPO_BASE`
auto-sets `tensorboard_log=None`. On a healthy image (Kaggle) it is a no-op and
TB logging stays on. It is imported before any sb3 import in `orbit_env.py` and
`curriculum.py`. `main.py` stays standalone and does not depend on it.

## Observation / action shapes
`OBS_DIM = 12*40 + 7*20 + 5 = 625`, `action = MultiDiscrete([10, 40, 4])`
(src planet slot, target planet slot, ships fraction `[.25,.5,.75,.95]`).
