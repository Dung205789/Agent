"""
env/tb_compat.py — make importing stable-baselines3 robust.

SB3 unconditionally does `from torch.utils.tensorboard import SummaryWriter`.
On some local images that drags in a broken tensorflow/protobuf pair and the
import raises a non-ImportError (e.g. protobuf VersionError), which SB3 does not
catch. This module probes that import once; if it fails, it registers a
lightweight stub so SB3 imports cleanly. On a healthy image (Kaggle) it is a
no-op.

Import this BEFORE any sb3_contrib / stable_baselines3 import. Use
`TENSORBOARD_OK` to decide whether to pass a real `tensorboard_log` path.
"""
import sys
import types

TENSORBOARD_OK = True

try:  # probe the real writer
    import torch.utils.tensorboard  # noqa: F401
except Exception:  # broken tensorboard/tensorflow/protobuf on this image
    TENSORBOARD_OK = False
    _stub = types.ModuleType("torch.utils.tensorboard")

    class SummaryWriter:  # noqa: N801 - mimic the real name
        def __init__(self, *a, **k):
            raise RuntimeError(
                "tensorboard is unavailable on this image (tb_compat stub); "
                "pass tensorboard_log=None"
            )

    _stub.SummaryWriter = SummaryWriter
    sys.modules["torch.utils.tensorboard"] = _stub
