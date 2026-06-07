"""
Build a dependency-light single-file agent `main_np.py`:
- extracts the MLP policy weights from the trained MaskablePPO model
- runs the deterministic forward pass in pure numpy (no torch, no sb3)
- verifies the numpy action selection matches SB3 exactly on real game states
- embeds the weights (base64 npz) and writes main_np.py

main_np.py imports only: os, math, numpy, base64, io, and Planet/Fleet from
kaggle_environments (always present in the competition runtime).
"""
import warnings; warnings.filterwarnings("ignore")
from env import tb_compat  # noqa: F401
import base64, io
import numpy as np
from sb3_contrib import MaskablePPO
from kaggle_environments import make
from env.core import encode_obs, get_action_masks

m = MaskablePPO.load("results/stage1/best_model.zip", device="cpu")
sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in m.policy.state_dict().items()}

# policy MLP + action head
W0, b0 = sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]
W1, b1 = sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]
W2, b2 = sd["mlp_extractor.policy_net.4.weight"], sd["mlp_extractor.policy_net.4.bias"]
Wa, ba = sd["action_net.weight"], sd["action_net.bias"]

NVEC = [10, 40, 4]

def np_forward(obs, mask):
    x = np.tanh(obs @ W0.T + b0)
    x = np.tanh(x @ W1.T + b1)
    x = np.tanh(x @ W2.T + b2)
    logits = x @ Wa.T + ba
    logits = np.where(mask, logits, -1e30)
    out, o = [], 0
    for n in NVEC:
        out.append(int(np.argmax(logits[o:o+n])))
        o += n
    return out

# ── verify numpy == sb3 deterministic action, on real game states ──
mism = 0; total = 0
for seed in range(6):
    tr = make("orbit_wars", configuration={"seed": seed}).train([None, "random"])
    obs = tr.reset(); done = False; steps = 0
    while not done and steps < 120:
        planets = obs.get("planets", []); player = obs.get("player", 0)
        vec = encode_obs(obs); msk = get_action_masks(planets, player)
        a_sb3, _ = m.predict(vec, action_masks=msk, deterministic=True)
        a_np = np_forward(vec, msk)
        total += 1
        if list(map(int, a_sb3)) != a_np:
            mism += 1
        # step env with sb3 action to follow on-policy trajectory
        from env.core import decode_action
        obs, r, done, info = tr.step(decode_action(a_sb3, planets, player, obs.get("angular_velocity", 0.03)))
        steps += 1
print(f"verify: {total-mism}/{total} actions match sb3 (mismatch={mism})")
assert mism == 0, "numpy forward does not match sb3!"

# ── pack weights to base64 npz ──
buf = io.BytesIO()
np.savez_compressed(buf, W0=W0, b0=b0, W1=W1, b1=b1, W2=W2, b2=b2, Wa=Wa, ba=ba)
b64 = base64.b64encode(buf.getvalue()).decode()
print("weights blob:", round(len(b64)/1e6, 2), "MB (base64)")

# ── assemble main_np.py from main.py's stable sections ──
src = open("main.py", encoding="utf-8").read()
mid = src[src.index("# ── Constants"):src.index("# ── Model Loading")]

header = '''"""
Orbit Wars — Pure PPO Agent (single file, numpy-only inference).

Self-contained: the trained MLP policy weights are embedded as base64 and the
deterministic forward pass runs in pure numpy. No torch / sb3-contrib needed at
runtime — only numpy + kaggle_environments (always present).
"""
import os
import io
import math
import base64
import numpy as np
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

'''

footer = '''# ── Embedded policy weights (base64 npz) ─────────────────────────────
_NVEC = [10, 40, 4]
_W = None
_WEIGHTS_B64 = (
%s)

def _weights():
    global _W
    if _W is None:
        _W = dict(np.load(io.BytesIO(base64.b64decode(_WEIGHTS_B64))))
    return _W

def _forward(obs_vec, mask):
    w = _weights()
    x = np.tanh(obs_vec @ w["W0"].T + w["b0"])
    x = np.tanh(x @ w["W1"].T + w["b1"])
    x = np.tanh(x @ w["W2"].T + w["b2"])
    logits = x @ w["Wa"].T + w["ba"]
    logits = np.where(mask, logits, -1e30)
    out, o = [], 0
    for n in _NVEC:
        out.append(int(np.argmax(logits[o:o + n])))
        o += n
    return out

# ── Agent Entry Point ─────────────────────────────────────────────────
def agent(obs, config=None):
    planets = _get(obs, "planets", [])
    player = _get(obs, "player", 0)
    av = _get(obs, "angular_velocity", 0.03)
    vec = encode_obs(obs)
    masks = get_masks(planets, player)
    action = _forward(vec, masks)
    return decode_action(action, planets, player, av)
'''

# chunk b64
CH = 120
blob = "".join('    "%s"\n' % b64[i:i+CH] for i in range(0, len(b64), CH))
out = header + mid + (footer % blob)
with open("main_np.py", "w", encoding="utf-8") as f:
    f.write(out)
import os as _os
print("wrote main_np.py", round(_os.path.getsize("main_np.py")/1e6, 2), "MB")
