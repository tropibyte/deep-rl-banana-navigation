"""Replay buffers: uniform, prioritized (sum-tree), and an n-step accumulator.

All buffers use pre-allocated NumPy ring buffers rather than a deque of
namedtuples -- with ~100k transitions the per-object overhead of the latter
dominates the actual training cost.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class Batch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor       # importance-sampling weights (all 1.0 if uniform)
    indices: np.ndarray         # tree indices, for priority updates


class _RingStorage:
    """Pre-allocated transition storage shared by both buffer types."""

    def __init__(self, capacity: int, state_size: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def _store(self, s, a, r, s2, d) -> int:
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = float(d)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def _gather(self, idx: np.ndarray, weights: np.ndarray, device) -> Batch:
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)
        return Batch(
            states=t(self.states[idx], torch.float32),
            actions=t(self.actions[idx], torch.int64).unsqueeze(1),
            rewards=t(self.rewards[idx], torch.float32).unsqueeze(1),
            next_states=t(self.next_states[idx], torch.float32),
            dones=t(self.dones[idx], torch.float32).unsqueeze(1),
            weights=t(weights, torch.float32).unsqueeze(1),
            indices=idx,
        )

    def __len__(self) -> int:
        return self.size


class UniformReplayBuffer(_RingStorage):
    """Classic DQN experience replay: sample uniformly at random."""

    def add(self, s, a, r, s2, d) -> None:
        self._store(s, a, r, s2, d)

    def sample(self, batch_size: int, device) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._gather(idx, np.ones(batch_size, dtype=np.float32), device)

    def update_priorities(self, indices, td_errors) -> None:
        """No-op: uniform sampling has no priorities. Keeps the agent code uniform."""

    @property
    def beta(self) -> float:
        return 1.0


class SumTree:
    """Fixed-size binary tree where each parent holds the sum of its children.

    Gives O(log n) prefix-sum search, which is what makes proportional
    prioritized sampling affordable at every learning step.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float) -> None:
        idx = data_idx + self.capacity - 1
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def find(self, prefix: float) -> int:
        """Return the data index whose cumulative range contains ``prefix``."""
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):          # leaf reached
                break
            if prefix <= self.tree[left]:
                idx = left
            else:
                prefix -= self.tree[left]
                idx = left + 1
        return idx - (self.capacity - 1)

    def max_leaf(self) -> float:
        return float(self.tree[self.capacity - 1:].max())


class PrioritizedReplayBuffer(_RingStorage):
    """Proportional prioritized replay (Schaul et al., 2016).

    P(i) ~ p_i^alpha, corrected by importance-sampling weights
    w_i = (N * P(i))^-beta, with beta annealed to 1.0 over training.
    """

    def __init__(self, capacity: int, state_size: int, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100_000,
                 epsilon: float = 1e-3):
        super().__init__(capacity, state_size)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        self._max_priority = 1.0

    @property
    def beta(self) -> float:
        f = min(1.0, self.frame / max(1, self.beta_frames))
        return self.beta_start + f * (1.0 - self.beta_start)

    def add(self, s, a, r, s2, d) -> None:
        i = self._store(s, a, r, s2, d)
        # New transitions enter at max priority so each is replayed at least once.
        self.tree.update(i, self._max_priority ** self.alpha)

    def sample(self, batch_size: int, device) -> Batch:
        self.frame += 1
        total = self.tree.total()
        # Stratified sampling: one draw per equal-probability segment reduces
        # variance versus batch_size independent draws over the whole range.
        segment = total / batch_size
        bounds = (np.arange(batch_size) + np.random.rand(batch_size)) * segment
        idx = np.array([self.tree.find(b) for b in bounds], dtype=np.int64)
        idx = np.clip(idx, 0, self.size - 1)

        leaves = self.tree.tree[idx + self.capacity - 1]
        probs = leaves / max(total, 1e-12)
        weights = (self.size * np.maximum(probs, 1e-12)) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)
        return self._gather(idx, weights, device)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        prios = np.abs(td_errors).astype(np.float64) + self.epsilon
        self._max_priority = max(self._max_priority, float(prios.max()))
        for i, p in zip(indices, prios):
            self.tree.update(int(i), p ** self.alpha)


class NStepAccumulator:
    """Converts 1-step transitions into n-step transitions on the fly.

    Emits (s_t, a_t, sum_{k<n} gamma^k r_{t+k}, s_{t+n}, done) which propagates
    reward backwards n times faster at the cost of a slightly off-policy target.
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buf: list[tuple] = []

    def append(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))
        if len(self.buf) < self.n:
            return None
        return self._emit()

    def _emit(self):
        R = 0.0
        for k, (_, _, r, _, _) in enumerate(self.buf):
            R += (self.gamma ** k) * r
        s, a = self.buf[0][0], self.buf[0][1]
        _, _, _, s2, d = self.buf[-1]
        self.buf.pop(0)
        return (s, a, R, s2, d)

    def flush(self):
        """Drain the tail of an episode, shortening the horizon as we go."""
        out = []
        while self.buf:
            out.append(self._emit())
        return out

    def reset(self) -> None:
        self.buf.clear()
