"""Training loop, solve detection, and run-artifact logging."""
from __future__ import annotations

import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent
from .env import BananaEnv

SOLVE_SCORE = 13.0
SOLVE_WINDOW = 100


@dataclass
class TrainConfig:
    n_episodes: int = 2000
    max_t: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    stop_on_solve: bool = False   # keep training past the threshold for a fuller curve
    solve_score: float = SOLVE_SCORE
    solve_window: int = SOLVE_WINDOW
    print_every: int = 50


@dataclass
class TrainResult:
    variant: str
    seed: int
    scores: list = field(default_factory=list)
    moving_avg: list = field(default_factory=list)
    solved_episode: int | None = None   # episodes needed, Udacity convention
    best_avg: float = float("-inf")
    wall_seconds: float = 0.0
    agent_config: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant, "seed": self.seed,
            "solved_episode": self.solved_episode, "best_avg": self.best_avg,
            "wall_seconds": self.wall_seconds, "scores": self.scores,
            "moving_avg": self.moving_avg,
            "agent_config": self.agent_config, "train_config": self.train_config,
        }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_torch_threads() -> int:
    """Pin this process to one BLAS/OMP thread (override with BANANA_TORCH_THREADS).

    Critical when running the ablation in parallel. Torch otherwise sizes its
    thread pool to the core count, so N training processes spawn N*cores
    threads, which then compete with N busy-spinning Unity ``-batchmode``
    processes. On an 8-core box that oversubscription starves the environments
    badly enough that runs make no measurable progress at all.

    These networks have ~13k parameters and a batch of 64: a single thread is
    faster than a pool even when running alone, so nothing is given up here.
    """
    n = int(os.environ.get("BANANA_TORCH_THREADS", "1"))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass  # already initialised; harmless
    return n


def train(
    env: BananaEnv,
    agent_cfg: AgentConfig,
    train_cfg: TrainConfig,
    variant: str = "dqn",
    seed: int = 0,
    checkpoint_path: str | Path | None = None,
    csv_path: str | Path | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Run one training job to completion and return its full history."""
    limit_torch_threads()
    set_global_seed(seed)
    agent = DQNAgent(agent_cfg, device="cpu", seed=seed)

    result = TrainResult(variant=variant, seed=seed,
                         agent_config=agent_cfg.to_dict(),
                         train_config=vars(train_cfg).copy())

    window: deque = deque(maxlen=train_cfg.solve_window)
    eps = train_cfg.eps_start
    # NoisyNet supplies its own exploration; epsilon-greedy on top would be
    # double-counting and measurably hurts it.
    if agent_cfg.noisy:
        eps = 0.0

    csv_file = csv_writer = None
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "score", "moving_avg_100", "epsilon", "loss", "elapsed_s"])

    t0 = time.time()
    try:
        for ep in range(1, train_cfg.n_episodes + 1):
            state = env.reset(train_mode=True)
            agent.end_episode()
            score = 0.0
            for _ in range(train_cfg.max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            window.append(score)
            avg = float(np.mean(window))
            result.scores.append(score)
            result.moving_avg.append(avg)
            result.best_avg = max(result.best_avg, avg)
            if not agent_cfg.noisy:
                eps = max(train_cfg.eps_end, train_cfg.eps_decay * eps)

            if csv_writer:
                csv_writer.writerow([ep, score, round(avg, 3), round(eps, 4),
                                     round(agent.last_loss, 5), round(time.time() - t0, 1)])
                # Flush every episode. Default buffering holds ~8KB, so an
                # unattended multi-hour study would show a 0-byte CSV for its
                # whole duration and a stalled run would be indistinguishable
                # from a healthy one. One flush per ~2s episode costs nothing.
                csv_file.flush()

            first_solve = (result.solved_episode is None
                           and len(window) == train_cfg.solve_window
                           and avg >= train_cfg.solve_score)
            if first_solve:
                # Udacity convention: the environment is "solved in N episodes"
                # where N is the first episode index whose trailing-100 mean
                # crosses the threshold, minus the 100-episode window.
                result.solved_episode = ep - train_cfg.solve_window
                if verbose:
                    print(f"\n[{variant} seed={seed}] SOLVED in {result.solved_episode} "
                          f"episodes! 100-ep average: {avg:.2f}")
                if checkpoint_path:
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    agent.save(checkpoint_path)
                if train_cfg.stop_on_solve:
                    break

            if verbose and ep % train_cfg.print_every == 0:
                print(f"[{variant} seed={seed}] ep {ep:4d}  avg100 {avg:6.2f}  "
                      f"eps {eps:.3f}  {time.time() - t0:6.0f}s", flush=True)
    finally:
        if csv_file:
            csv_file.close()

    result.wall_seconds = time.time() - t0
    # Always persist final weights: for a run that never solves, this is still
    # the artifact you want to inspect.
    if checkpoint_path and result.solved_episode is None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        agent.save(checkpoint_path)
    return result


def evaluate(env: BananaEnv, agent: DQNAgent, episodes: int = 100,
             train_mode: bool = True, verbose: bool = True) -> dict:
    """Greedy evaluation of a trained agent (no exploration at all)."""
    scores = []
    for ep in range(1, episodes + 1):
        state = env.reset(train_mode=train_mode)
        score = 0.0
        done = False
        while not done:
            action = agent.act(state, greedy=True)
            state, reward, done, _ = env.step(action)
            score += reward
        scores.append(score)
        if verbose and ep % 20 == 0:
            print(f"  eval ep {ep:3d}  mean so far {np.mean(scores):.2f}", flush=True)
    arr = np.array(scores)
    return {"episodes": episodes, "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()),
            "median": float(np.median(arr)), "scores": scores}


def save_result(result: TrainResult, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
