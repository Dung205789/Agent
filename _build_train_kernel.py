"""
Generate the Kaggle training kernel (notebook + metadata) from repo sources.

Output: kaggle_train/train_kernel.ipynb + kaggle_train/kernel-metadata.json
Push with:  kaggle kernels push -p kaggle_train
The kernel trains on Kaggle cloud and saves /kaggle/working/best_model.zip
(downloadable as the kernel output, then promoted to a Dataset).
"""
import json
import io
import os

KAGGLE_USER = "dungun"
KERNEL_SLUG = "orbit-wars-train"


def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read().rstrip("\n")


_CID = [0]
def _id():
    _CID[0] += 1
    return f"c{_CID[0]:02d}"

def md(t):
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": t.splitlines(keepends=True)}

def code(t):
    return {"cell_type": "code", "id": _id(), "metadata": {}, "execution_count": None,
            "outputs": [], "source": t.splitlines(keepends=True)}


core = read("env/core.py")
reward = read("env/reward.py")
tb_compat = read("env/tb_compat.py")
orbit_env = read("env/orbit_env.py")
selfplay = read("train/selfplay.py")
curriculum = read("train/curriculum.py")

cells = []
cells.append(md(
"""# Orbit Wars — Training Kernel (Kaggle cloud)

Trains MaskablePPO and saves `/kaggle/working/best_model.zip` as the kernel
output. Promote that output to the `orbit-wars-weights` Dataset, then run the
submission notebook.

> The env simulation (Python) is the throughput bottleneck, so we parallelise
> with `SubprocVecEnv` across CPU cores. GPU is enabled but helps the small MLP
> only marginally."""))

cells.append(code('!pip install -q "kaggle-environments>=1.28.0" "sb3-contrib==2.8.0" tqdm rich\n'
                  "import warnings; warnings.filterwarnings('ignore')"))

cells.append(md("## Write package"))
cells.append(code('import os\nos.makedirs("env", exist_ok=True)\nopen("env/__init__.py","w").close()'))
cells.append(code("%%writefile env/core.py\n" + core))
cells.append(code("%%writefile env/reward.py\n" + reward))
cells.append(code("%%writefile env/tb_compat.py\n" + tb_compat))
cells.append(code("%%writefile env/orbit_env.py\n" + orbit_env))
cells.append(code('os.makedirs("train", exist_ok=True)\nopen("train/__init__.py","w").close()'))
cells.append(code("%%writefile train/selfplay.py\n" + selfplay))
cells.append(code("%%writefile train/curriculum.py\n" + curriculum))

cells.append(md(
"""## Train Stage 1 — vs `starter` bot, 4-player, full reward

Multi-launch action space (commands every owned planet/turn). A **checkpoint is
saved every 100k steps** to `/kaggle/working/best_model.zip`, so whatever the
session reaches is kept even if it hits the time limit. `TRAIN_STEPS` is set high
on purpose — let it run as long as the session allows."""))
cells.append(code(
"""import time, multiprocessing as mp
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from train.curriculum import build_vec, PPO_BASE

class SaveCkpt(BaseCallback):
    def __init__(self, freq, path):
        super().__init__(); self.freq = freq; self.path = path; self._last = 0
    def _on_step(self):
        if self.num_timesteps - self._last >= self.freq:
            self._last = self.num_timesteps
            self.model.save(self.path)   # overwrite latest checkpoint
        return True

N_ENVS = min(4, max(2, mp.cpu_count()))    # CPU cores -> parallel envs
TRAIN_STEPS = 5_000_000                     # checkpointed; keeps whatever completes

vec = build_vec("starter", "full", N_ENVS, use_subproc=True)   # 4-player vs starter
cfg = dict(PPO_BASE); cfg["tensorboard_log"] = None
model = MaskablePPO("MlpPolicy", vec, **cfg)
ckpt = SaveCkpt(100_000, "/kaggle/working/best_model")

t0 = time.time()
try:
    model.learn(TRAIN_STEPS, progress_bar=True, callback=ckpt)
finally:
    model.save("/kaggle/working/best_model")    # always save final/partial
print(f"elapsed {(time.time()-t0)/60:.1f} min  (n_envs={N_ENVS}, cpu={mp.cpu_count()})")"""))

cells.append(md("## Quick eval vs starter (4-player)"))
cells.append(code(
"""from env.orbit_env import resolve_opponent
from kaggle_environments import make

def quick_winrate(model_path, n=12):
    a = resolve_opponent(model_path)
    w = 0
    for i in range(n):
        r = make("orbit_wars", configuration={"seed": i}).run([a, "starter", "starter", "starter"])
        rewards = [s["reward"] for s in r[-1]]
        if rewards[0] == max(rewards): w += 1
    return w / n

wr = quick_winrate("/kaggle/working/best_model.zip", n=12)
print(f"top-1 rate vs 3x starter (4p): {wr:.0%}")"""))

cells.append(md("## Output check"))
cells.append(code('import os\nprint([f for f in os.listdir("/kaggle/working") if f.endswith(".zip")])'))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs("kaggle_train", exist_ok=True)
with io.open("kaggle_train/train_kernel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

meta = {
    "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
    "title": KERNEL_SLUG,
    "code_file": "train_kernel.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    # CPU on purpose: the 4-player env sim (Python) is the bottleneck, so 4 CPU
    # cores via SubprocVecEnv beat a GPU session's 2 cores. The tiny MLP gains
    # ~nothing from GPU.
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
with io.open("kaggle_train/kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("wrote kaggle_train/train_kernel.ipynb (", len(cells), "cells ) + kernel-metadata.json")
