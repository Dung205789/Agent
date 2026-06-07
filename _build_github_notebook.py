"""
Generate `train_on_kaggle.ipynb` — a self-contained Kaggle notebook that clones
the GitHub repo, installs deps, and trains (4-player vs starter, multi-launch),
checkpointing best_model.zip to /kaggle/working. Upload this notebook to Kaggle
and run it (CPU recommended). Download best_model.zip from the output afterward.
"""
import json, io

REPO = "https://github.com/Dung205789/Agent.git"

_CID = [0]
def _id():
    _CID[0] += 1
    return f"c{_CID[0]:02d}"
def md(t):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t):
    return {"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}

cells = []
cells.append(md(
f"""# Orbit Wars — Train on Kaggle (pulls code from GitHub)

1. **Settings → Accelerator:** pick **CPU** (recommended — this workload is env-bound,
   4 CPU cores beat a 2-core GPU session). **Internet: ON**.
2. Run all cells. Training checkpoints `best_model.zip` to the output every 100k
   steps, so a timed-out session still yields a model.
3. After it stops, download **`/kaggle/working/best_model.zip`** from the
   notebook **Output** tab and send it back.

Repo: {REPO}"""))

cells.append(md("## 1. Clone repo + install deps"))
cells.append(code(
f"""!rm -rf /kaggle/working/Agent
!git clone --depth 1 {REPO} /kaggle/working/Agent
!pip install -q "kaggle-environments>=1.28.0" "sb3-contrib==2.8.0" tqdm rich

import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir("/kaggle/working/Agent")
sys.path.insert(0, "/kaggle/working/Agent")
print("files:", os.listdir("."))"""))

cells.append(md(
"""## 2. Train — 4-player vs `starter`, multi-launch, full reward

`TRAIN_STEPS` is set high on purpose; the checkpoint callback keeps the latest
model so you can stop anytime. Raise/lower to taste."""))
cells.append(code(
"""import time, multiprocessing as mp
from env import tb_compat            # noqa: makes sb3 import robust
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from train.curriculum import build_vec, PPO_BASE

class SaveCkpt(BaseCallback):
    def __init__(self, freq, path):
        super().__init__(); self.freq = freq; self.path = path; self._last = 0
    def _on_step(self):
        if self.num_timesteps - self._last >= self.freq:
            self._last = self.num_timesteps
            self.model.save(self.path)          # overwrite latest checkpoint
            print(f"  [ckpt] saved at {self.num_timesteps} steps", flush=True)
        return True

N_ENVS = min(4, max(2, mp.cpu_count()))         # parallel envs = CPU cores
TRAIN_STEPS = 5_000_000                          # checkpointed; stop anytime

vec = build_vec("starter", "full", N_ENVS, use_subproc=True)   # 4-player vs starter
cfg = dict(PPO_BASE); cfg["tensorboard_log"] = None
model = MaskablePPO("MlpPolicy", vec, **cfg)
ckpt = SaveCkpt(100_000, "/kaggle/working/best_model")

t0 = time.time()
try:
    model.learn(TRAIN_STEPS, progress_bar=True, callback=ckpt)
finally:
    model.save("/kaggle/working/best_model")     # always save final/partial
print(f"elapsed {(time.time()-t0)/60:.1f} min  (n_envs={N_ENVS}, cpu={mp.cpu_count()})")"""))

cells.append(md("## 3. Quick eval — top-1 rate vs 3× starter (4-player)"))
cells.append(code(
"""from env.orbit_env import resolve_opponent
from kaggle_environments import make

def quick_winrate(model_path, n=12):
    a = resolve_opponent(model_path); w = 0
    for i in range(n):
        r = make("orbit_wars", configuration={"seed": i}).run([a, "starter", "starter", "starter"])
        rewards = [s["reward"] for s in r[-1]]
        if rewards[0] == max(rewards): w += 1
    return w / n

print("top-1 rate vs 3x starter (4p):", f"{quick_winrate('/kaggle/working/best_model.zip'):.0%}")"""))

cells.append(md("## 4. Confirm output (download best_model.zip from the Output tab)"))
cells.append(code(
'import os\nz = "/kaggle/working/best_model.zip"\n'
'print("best_model.zip:", os.path.exists(z), round(os.path.getsize(z)/1e6, 2), "MB")'))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}

with io.open("train_on_kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote train_on_kaggle.ipynb (", len(cells), "cells )")
