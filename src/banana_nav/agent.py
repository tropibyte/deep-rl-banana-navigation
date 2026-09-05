"""Value-based agent with independently switchable Rainbow components.

Every extension is a boolean/int on ``AgentConfig`` so the ablation can toggle
exactly one thing at a time against a fixed baseline:

    double       Double DQN            -- decouple action selection from evaluation
    dueling      Dueling architecture  -- separate V(s) and A(s,a) streams
    prioritized  Prioritized replay    -- sample high-TD-error transitions more often
    n_step       Multi-step returns    -- propagate reward n times faster
    noisy        NoisyNet exploration  -- learned exploration, replaces epsilon-greedy
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import QNetwork
from .replay import UniformReplayBuffer, PrioritizedReplayBuffer, NStepAccumulator


@dataclass
class AgentConfig:
    state_size: int = 37
    action_size: int = 4
    hidden: tuple = (128, 64)

    # -- Rainbow component switches --
    double: bool = False
    dueling: bool = False
    prioritized: bool = False
    noisy: bool = False
    n_step: int = 1

    # -- core DQN hyperparameters --
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 5e-4
    tau: float = 1e-3           # soft target update rate
    update_every: int = 4       # env steps between learning updates
    learn_starts: int = 1_000   # warm-up transitions before learning
    grad_clip: float = 10.0
    huber: bool = True

    # -- prioritized replay --
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000

    def to_dict(self) -> dict:
        return asdict(self)


class DQNAgent:
    def __init__(self, cfg: AgentConfig, device: torch.device | str = "cpu", seed: int = 0):
        self.cfg = cfg
        self.device = torch.device(device)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        net = lambda: QNetwork(cfg.state_size, cfg.action_size, cfg.hidden,
                               dueling=cfg.dueling, noisy=cfg.noisy).to(self.device)
        self.qnet_local = net()
        self.qnet_target = net()
        self.qnet_target.load_state_dict(self.qnet_local.state_dict())
        for p in self.qnet_target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=cfg.lr)

        if cfg.prioritized:
            self.memory = PrioritizedReplayBuffer(
                cfg.buffer_size, cfg.state_size, alpha=cfg.per_alpha,
                beta_start=cfg.per_beta_start, beta_frames=cfg.per_beta_frames)
        else:
            self.memory = UniformReplayBuffer(cfg.buffer_size, cfg.state_size)

        self.nstep = NStepAccumulator(cfg.n_step, cfg.gamma) if cfg.n_step > 1 else None
        self.gamma_n = cfg.gamma ** cfg.n_step
        self.t_step = 0
        self.last_loss = float("nan")

    # -- acting ----------------------------------------------------------
    def act(self, state: np.ndarray, eps: float = 0.0, greedy: bool = False) -> int:
        st = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)

        if self.cfg.noisy and not greedy:
            # Exploration comes from the weight noise itself, so act greedily
            # w.r.t. a *noisy* forward pass rather than sampling a random action.
            self.qnet_local.train()
            self.qnet_local.reset_noise()
            with torch.no_grad():
                q = self.qnet_local(st)
            return int(q.argmax(dim=1).item())

        self.qnet_local.eval()
        with torch.no_grad():
            q = self.qnet_local(st)
        self.qnet_local.train()

        if not greedy and random.random() < eps:
            return int(random.randrange(self.cfg.action_size))
        return int(q.argmax(dim=1).item())

    # -- experience ------------------------------------------------------
    def step(self, s, a, r, s2, done) -> None:
        if self.nstep is None:
            self.memory.add(s, a, r, s2, done)
        else:
            tr = self.nstep.append(s, a, r, s2, done)
            if tr is not None:
                self.memory.add(*tr)
            if done:
                for tr in self.nstep.flush():
                    self.memory.add(*tr)

        self.t_step += 1
        if (self.t_step % self.cfg.update_every == 0
                and len(self.memory) >= max(self.cfg.batch_size, self.cfg.learn_starts)):
            self._learn()

    def end_episode(self) -> None:
        if self.nstep is not None:
            self.nstep.reset()

    # -- learning --------------------------------------------------------
    def _learn(self) -> None:
        batch = self.memory.sample(self.cfg.batch_size, self.device)

        if self.cfg.noisy:
            self.qnet_local.reset_noise()
            self.qnet_target.reset_noise()

        with torch.no_grad():
            if self.cfg.double:
                # Select with the online net, evaluate with the target net --
                # this is what removes the max-operator's optimistic bias.
                next_actions = self.qnet_local(batch.next_states).argmax(dim=1, keepdim=True)
                q_next = self.qnet_target(batch.next_states).gather(1, next_actions)
            else:
                q_next = self.qnet_target(batch.next_states).max(dim=1, keepdim=True)[0]
            q_target = batch.rewards + self.gamma_n * q_next * (1.0 - batch.dones)

        q_expected = self.qnet_local(batch.states).gather(1, batch.actions)

        if self.cfg.huber:
            elementwise = F.smooth_l1_loss(q_expected, q_target, reduction="none")
        else:
            elementwise = (q_expected - q_target).pow(2)
        # IS weights are all 1.0 for uniform replay, so this one line covers both.
        loss = (batch.weights * elementwise).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.qnet_local.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        self.last_loss = float(loss.detach())

        td = (q_target - q_expected).detach().abs().squeeze(1).cpu().numpy()
        self.memory.update_priorities(batch.indices, td)

        self._soft_update()

    def _soft_update(self) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for tp, lp in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
                tp.mul_(1.0 - tau).add_(tau * lp)

    # -- persistence -----------------------------------------------------
    def save(self, path) -> None:
        torch.save({"config": self.cfg.to_dict(),
                    "state_dict": self.qnet_local.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu") -> "DQNAgent":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = dict(ckpt["config"])
        cfg_dict["hidden"] = tuple(cfg_dict["hidden"])
        agent = cls(AgentConfig(**cfg_dict), device=device)
        agent.qnet_local.load_state_dict(ckpt["state_dict"])
        agent.qnet_target.load_state_dict(ckpt["state_dict"])
        return agent
