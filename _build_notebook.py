"""
Generate submission_notebook.ipynb from the actual repo files so notebook code
never drifts from the tested source.
"""
import json
import io

def read(p):
    with io.open(p, encoding="utf-8") as f:
        return f.read().rstrip("\n")

_CID = [0]
def _next_id():
    _CID[0] += 1
    return f"cell-{_CID[0]:02d}"

def md(text):
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {},
            "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "id": _next_id(), "metadata": {},
            "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}

core = read("env/core.py")
reward = read("env/reward.py")
orbit_env = read("env/orbit_env.py")
tb_compat = read("env/tb_compat.py")
selfplay = read("train/selfplay.py")
curriculum = read("train/curriculum.py")
main_py = read("main.py")

cells = []

cells.append(md(
"""# Orbit Wars — Pure PPO Submission Notebook

**Deliverable.** Trains a MaskablePPO agent, writes a self-contained `main.py`,
packages `submission.tar.gz`, and submits to the `orbit_wars` competition.

Workflow:
1. Install deps
2. (optional) Train locally / on Kaggle GPU → `models/best_model.zip`
3. Upload weights as Kaggle Dataset `orbit-wars-weights`, attach it
4. `%%writefile main.py` (inlined agent — no `env/` imports)
5. Local test vs random (win-rate gate)
6. Package + submit + monitor

> Verified against the real env: Planet `(id,owner,x,y,radius,ships,production)`,
> Fleet `(id,owner,x,y,angle,from_planet_id,ships)`, sun (50,50) r=10,
> `angular_velocity` per-episode in [0.025,0.05], action `[from_id, angle, ships]`,
> config `episodeSteps=500, actTimeout=1s, shipSpeed=6`.
"""))

# ── Section 1: setup ──
cells.append(md("## 1. Setup"))
cells.append(code(
'!pip install -q "kaggle-environments>=1.28.0" "stable-baselines3==2.3.2" "sb3-contrib==2.3.2"\n'
"import warnings; warnings.filterwarnings('ignore')"))

# ── Section 2: write env package for training ──
cells.append(md(
"""## 2. Write the training package (`env/`)

These are the exact, tested modules. Skip this whole section if you only want to
package pre-trained weights for submission."""))
cells.append(code('import os\nos.makedirs("env", exist_ok=True)\nopen("env/__init__.py","w").close()'))
cells.append(code("%%writefile env/core.py\n" + core))
cells.append(code("%%writefile env/reward.py\n" + reward))
cells.append(code("%%writefile env/tb_compat.py\n" + tb_compat))
cells.append(code("%%writefile env/orbit_env.py\n" + orbit_env))
cells.append(code('os.makedirs("train", exist_ok=True)\nopen("train/__init__.py","w").close()'))
cells.append(code("%%writefile train/selfplay.py\n" + selfplay))
cells.append(code("%%writefile train/curriculum.py\n" + curriculum))

# ── Section 3: train ──
cells.append(md(
"""## 3. Train

`NB_STEPS` is small by default so the notebook finishes quickly. For real
submissions follow `task.md`: Stage 1 = 5M, Stage 2 = 15M, Stage 3 = self-play.
Train heavy runs locally (CPU MLP) and just upload the weights — the MLP policy
barely benefits from the T4. Set `NB_STEPS` higher only if training in-notebook.

> `tensorboard_log=None` here avoids a tensorboard import on some images; pass
> a path if you want TB curves."""))
cells.append(code(
"""from train.curriculum import train_stage1

NB_STEPS = 200_000   # bump to 5_000_000 for a real Stage-1 run
model, path = train_stage1(
    total_timesteps=NB_STEPS,
    n_envs=4,
    use_subproc=False,            # notebooks: DummyVecEnv is most robust
    save_path="models/best_model",
    ppo_overrides={"tensorboard_log": None},
)
print("saved:", path + ".zip")"""))

# ── Section 4: upload weights ──
cells.append(md(
"""## 4. Upload weights as a Kaggle Dataset (run once)

Create / version a dataset named **`orbit-wars-weights`** containing
`best_model.zip`, then **attach it** to this notebook. The agent loads from
`/kaggle/input/orbit-wars-weights/best_model.zip`.

```bash
# locally, after training:
mkdir -p ow_weights && cp models/best_model.zip ow_weights/
kaggle datasets init -p ow_weights
# edit ow_weights/dataset-metadata.json -> title/id "orbit-wars-weights"
kaggle datasets create -p ow_weights      # first time
kaggle datasets version -p ow_weights -m "new weights"   # updates
```"""))

# ── Section 5: verify load ──
cells.append(md("## 5. Verify the attached weights load"))
cells.append(code(
"""import os
from sb3_contrib import MaskablePPO

CANDIDATES = [
    "/kaggle/input/orbit-wars-weights/best_model.zip",
    "models/best_model.zip",   # fallback if trained in this session
]
WEIGHTS = next((p for p in CANDIDATES if os.path.exists(p)), None)
assert WEIGHTS, f"No weights found in {CANDIDATES}"
_m = MaskablePPO.load(WEIGHTS, device="cpu")
print("Loaded weights from:", WEIGHTS)
print(_m.policy)"""))

# ── Section 6: write main.py ──
cells.append(md(
"""## 6. Write `main.py` (self-contained agent)

Set `OW_MODEL_PATH` so the agent finds the weights both on Kaggle and during the
local test below. The file itself tries the standard `/kaggle/input/...` paths
first, then `OW_MODEL_PATH`."""))
cells.append(code('import os\nos.environ["OW_MODEL_PATH"] = os.path.abspath(WEIGHTS)'))
cells.append(code("%%writefile main.py\n" + main_py))

# ── Section 7: local test ──
cells.append(md("## 7. Local test vs random (win-rate gate)"))
cells.append(code(
"""from kaggle_environments import make

results = {"win": 0, "lose": 0, "draw": 0}
exceptions = 0
N_GAMES = 20
for seed in range(N_GAMES):
    env = make("orbit_wars", configuration={"seed": seed})
    result = env.run(["main.py", "random"])
    last = result[-1]
    for a in last:
        if a.status not in ("ACTIVE", "DONE", "INACTIVE"):
            exceptions += 1
    r0, r1 = last[0].reward, last[1].reward
    if r0 > r1: results["win"] += 1
    elif r0 < r1: results["lose"] += 1
    else: results["draw"] += 1

win_rate = results["win"] / N_GAMES
print(f"Win rate vs random: {win_rate:.1%}  {results}  exceptions={exceptions}")
assert exceptions == 0, "agent raised during episodes"
# For a real Stage-1 model expect >= 0.70 (raise per stage; S2+ -> 0.80).
# assert win_rate >= 0.70, f"below submit threshold: {win_rate:.1%}\""""))

# ── Section 8: package ──
cells.append(md("## 8. Package `submission.tar.gz` (must contain only `main.py` at root)"))
cells.append(code(
"""import tarfile
with tarfile.open("submission.tar.gz", "w:gz") as tar:
    tar.add("main.py", arcname="main.py")
print("contents:")
with tarfile.open("submission.tar.gz") as tar:
    for m in tar.getmembers():
        print(" ", m.name)"""))

# ── Section 9: submit ──
cells.append(md("## 9. Submit"))
cells.append(code(
'SUBMISSION_MESSAGE = "S1 PPO stage1 vs random"\n'
'!kaggle competitions submit orbit-wars -f submission.tar.gz -m "{SUBMISSION_MESSAGE}"'))

# ── Section 10: monitor ──
cells.append(md("## 10. Monitor"))
cells.append(code("!kaggle competitions submissions orbit-wars"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with io.open("submission_notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("wrote submission_notebook.ipynb with", len(cells), "cells")
