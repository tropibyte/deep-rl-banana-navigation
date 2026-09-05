"""YAML experiment configs -> AgentConfig / TrainConfig."""
from __future__ import annotations

from pathlib import Path

import yaml

from .agent import AgentConfig
from .train import TrainConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load(name_or_path: str) -> tuple[str, dict, dict]:
    """Load a config by bare name (``dqn``) or explicit path."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIG_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        available = sorted(f.stem for f in CONFIG_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No config '{name_or_path}'. Available: {available}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return raw.get("name", p.stem), raw.get("agent", {}) or {}, raw.get("train", {}) or {}


def build(name_or_path: str, state_size: int, action_size: int,
          overrides: dict | None = None) -> tuple[str, AgentConfig, TrainConfig]:
    name, a, t = load(name_or_path)
    a = dict(a)
    if "hidden" in a:
        a["hidden"] = tuple(a["hidden"])
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        (t if k in TrainConfig.__dataclass_fields__ else a)[k] = v
    return name, AgentConfig(state_size=state_size, action_size=action_size, **a), TrainConfig(**t)


def list_configs() -> list[str]:
    return sorted(f.stem for f in CONFIG_DIR.glob("*.yaml"))
