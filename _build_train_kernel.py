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

KAGGLE_USER = "dzungngo179"
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
"""## Train Stage 1 (vs random)

Adjust `TRAIN_STEPS` to fit the session budget. ~3M is a solid first run;
throughput depends on CPU cores. Output is saved as `best_model.zip`."""))
cells.append(code(
"""import time, multiprocessing as mp
from train.curriculum import train_stage1

VERIFY = False                             # quick end-to-end check; False = real run
# Cap env count: Kaggle reports the host's core count, oversubscribing slows
# SubprocVecEnv badly. 4 is a safe sweet spot.
N_ENVS = min(4, max(2, mp.cpu_count()))
TRAIN_STEPS = 4_000 if VERIFY else 1_000_000

t0 = time.time()
model, path = train_stage1(
    total_timesteps=TRAIN_STEPS,
    n_envs=(2 if VERIFY else N_ENVS),
    use_subproc=(not VERIFY),              # DummyVecEnv for the fast verify
    save_path="/kaggle/working/best_model",
)
model.save("/kaggle/working/stage1_random")   # stage-named copy
print(f"done in {(time.time()-t0)/60:.1f} min -> {path}.zip "
      f"(n_envs={2 if VERIFY else N_ENVS}, steps={TRAIN_STEPS}, cpu={mp.cpu_count()})")"""))

cells.append(md("## Quick eval vs random"))
cells.append(code(
"""from eval_arena import run_match  # written below
"""))
# inline a tiny arena to avoid needing the eval package
cells[-1] = code(
"""from env.orbit_env import resolve_opponent
from kaggle_environments import make

def quick_winrate(model_path, n=10):
    a = resolve_opponent(model_path); b = "random"
    w = 0
    for i in range(n):
        env = make("orbit_wars", configuration={"seed": i})
        r = env.run([a, b])
        if r[-1][0].reward > r[-1][1].reward: w += 1
    return w / n

wr = quick_winrate("/kaggle/working/best_model.zip", n=(5 if VERIFY else 20))
print(f"win rate vs random: {wr:.0%}")""")

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
    # GPU is disabled on purpose: this PPO workload is env-simulation bound
    # (small MLP), so a GPU adds queue latency without speeding training. CPU
    # kernels start fast and give cores for SubprocVecEnv. Flip to True only if
    # you later switch to a CNN/large policy.
    "enable_gpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
}
with io.open("kaggle_train/kernel-metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print("wrote kaggle_train/train_kernel.ipynb (", len(cells), "cells ) + kernel-metadata.json")
