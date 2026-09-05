"""A small gym-style wrapper around the Unity ML-Agents v0.4 Banana environment.

Why this exists: the raw v0.4 API is verbose (``env.step(a)[brain_name]``,
``info.vector_observations[0]``) and leaks Unity concepts into agent code. It
also has two sharp edges this wrapper files down:

1. **Leaked processes.** If an exception escapes before ``env.close()``, the
   gRPC server thread is non-daemon, so Python hangs forever holding a live
   ``Banana.exe``. This wrapper is a context manager and closes on ``__del__``.
2. **Port collisions.** ``worker_id`` maps directly onto TCP port 5005+id with
   no collision handling, so a leaked process from an earlier run makes the
   next one fail with a misleading "worker number is still in use". We probe
   for a free port instead.
"""
from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import numpy as np

# The vendored ml-agents client lives in vendor/ so it can be pip-installed as a
# top-level package; when running from a source checkout without installing,
# put it on the path here.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = _REPO_ROOT / "vendor"
if _VENDOR.is_dir() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from unityagents import UnityEnvironment  # noqa: E402

BASE_PORT = 5005

_CANDIDATE_EXES = [
    "Banana_Windows_x86_64/Banana.exe",
    "Banana_Windows_x86/Banana.exe",
    "Banana_Linux/Banana.x86_64",
    "Banana_Linux_NoVis/Banana.x86_64",
    "Banana.app",
]


def find_executable(explicit: str | None = None) -> str:
    """Locate the Unity build, preferring an explicit path or $BANANA_ENV_PATH."""
    for cand in filter(None, [explicit, os.environ.get("BANANA_ENV_PATH")]):
        if Path(cand).exists():
            return str(cand)
        raise FileNotFoundError(f"Unity environment not found at: {cand}")
    for rel in _CANDIDATE_EXES:
        p = _REPO_ROOT / rel
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Could not find the Banana Unity environment. Download it (see README), "
        "unzip it into the repo root, or set BANANA_ENV_PATH."
    )


def _port_is_free(port: int) -> bool:
    """True if nothing is listening on ``port``.

    Deliberately probes by *connecting*, not by binding. Two Windows-specific
    traps make the obvious bind-test silently useless:

    * ``SO_REUSEADDR`` on Windows permits binding a port that is already bound
      (the opposite of its BSD/Linux meaning), so the bind always "succeeds".
    * gRPC binds ``[::]`` (IPv6 dual-stack) while a naive probe binds IPv4
      ``0.0.0.0``; on Windows those need not collide, so the probe passes and
      the gRPC server then dies with WSAEADDRINUSE.

    A successful connect is unambiguous proof that something is listening.
    """
    for family, addr in ((socket.AF_INET, ("127.0.0.1", port)),
                         (socket.AF_INET6, ("::1", port))):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.3)
                if s.connect_ex(addr) == 0:
                    return False
        except OSError:
            continue
    return True


def pick_worker_id(preferred: int = 0, span: int = 64) -> int:
    """First worker_id whose port is actually free, starting at ``preferred``."""
    for offset in range(span):
        wid = preferred + offset
        if _port_is_free(BASE_PORT + wid):
            return wid
    raise RuntimeError(f"No free port in {BASE_PORT + preferred}..{BASE_PORT + preferred + span}")


class BananaEnv:
    """Single-agent gym-style view of the Banana Collector environment."""

    def __init__(
        self,
        exe_path: str | None = None,
        worker_id: int | None = None,
        no_graphics: bool = True,
        seed: int = 0,
        train_mode: bool = True,
    ):
        self.exe_path = find_executable(exe_path)
        self.worker_id = pick_worker_id(0 if worker_id is None else worker_id)
        self.train_mode = train_mode
        self._closed = False

        self._env = UnityEnvironment(
            file_name=self.exe_path,
            worker_id=self.worker_id,
            no_graphics=no_graphics,
            seed=seed,
        )
        self.brain_name = self._env.brain_names[0]
        brain = self._env.brains[self.brain_name]
        self.action_size = int(brain.vector_action_space_size)
        info = self._env.reset(train_mode=train_mode)[self.brain_name]
        self.state_size = int(len(info.vector_observations[0]))

    # -- gym-style API ---------------------------------------------------
    def reset(self, train_mode: bool | None = None) -> np.ndarray:
        mode = self.train_mode if train_mode is None else train_mode
        info = self._env.reset(train_mode=mode)[self.brain_name]
        return np.asarray(info.vector_observations[0], dtype=np.float32)

    def step(self, action: int):
        info = self._env.step(int(action))[self.brain_name]
        state = np.asarray(info.vector_observations[0], dtype=np.float32)
        return state, float(info.rewards[0]), bool(info.local_done[0]), {}

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._env.close()
            except Exception:
                pass

    # -- lifecycle safety ------------------------------------------------
    def __enter__(self) -> "BananaEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
