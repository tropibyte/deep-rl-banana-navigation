"""Unit tests for the agent's components.

Deliberately free of any Unity dependency, so they run in CI without the
environment binary. Everything here is checked against a property that must
hold analytically, not against a recorded value.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from banana_nav.agent import AgentConfig, DQNAgent
from banana_nav.config import build, list_configs
from banana_nav.networks import NoisyLinear, QNetwork
from banana_nav.replay import (
    NStepAccumulator, PrioritizedReplayBuffer, SumTree, UniformReplayBuffer,
)

STATE, ACTIONS = 37, 4


# ------------------------------------------------------------- networks ----
@pytest.mark.parametrize("dueling", [False, True])
@pytest.mark.parametrize("noisy", [False, True])
def test_network_shapes(dueling, noisy):
    net = QNetwork(STATE, ACTIONS, dueling=dueling, noisy=noisy)
    out = net(torch.randn(8, STATE))
    assert out.shape == (8, ACTIONS)
    assert torch.isfinite(out).all()


def test_dueling_head_is_mean_centred():
    """Q = V + (A - mean A) means the advantage stream cannot shift Q's mean."""
    net = QNetwork(STATE, ACTIONS, dueling=True)
    x = net.body(torch.randn(16, STATE))
    adv = net.adv(x)
    centred = adv - adv.mean(dim=1, keepdim=True)
    assert torch.allclose(centred.mean(dim=1), torch.zeros(16), atol=1e-6)


def test_noisy_layer_is_stochastic_in_train_and_exact_in_eval():
    layer = NoisyLinear(16, 8)
    x = torch.randn(4, 16)

    layer.train()
    a = layer(x)
    layer.reset_noise()
    b = layer(x)
    assert not torch.allclose(a, b), "resampling noise must change the output"

    layer.eval()
    c, d = layer(x), layer(x)
    assert torch.allclose(c, d), "eval must be deterministic (mean weights only)"


def test_noisy_network_reset_noise_reaches_every_layer():
    net = QNetwork(STATE, ACTIONS, dueling=True, noisy=True)
    before = [m.weight_epsilon.clone() for m in net.modules() if isinstance(m, NoisyLinear)]
    assert before, "noisy=True must create NoisyLinear layers"
    net.reset_noise()
    after = [m.weight_epsilon for m in net.modules() if isinstance(m, NoisyLinear)]
    assert all(not torch.allclose(x, y) for x, y in zip(before, after))


# -------------------------------------------------------------- sumtree ----
def test_sumtree_total_and_sampling_distribution():
    priorities = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    tree = SumTree(len(priorities))
    for i, p in enumerate(priorities):
        tree.update(i, p)
    assert tree.total() == pytest.approx(sum(priorities))

    # Sweeping the prefix range must land in each leaf proportional to priority.
    counts = np.zeros(len(priorities))
    for s in np.linspace(0, tree.total() - 1e-9, 40_000):
        counts[tree.find(s)] += 1
    empirical = counts / counts.sum()
    expected = np.array(priorities) / sum(priorities)
    assert np.abs(empirical - expected).max() < 0.01


def test_sumtree_update_changes_probability():
    tree = SumTree(4)
    for i in range(4):
        tree.update(i, 1.0)
    tree.update(2, 100.0)
    assert tree.total() == pytest.approx(103.0)
    hits = sum(tree.find(s) == 2 for s in np.linspace(0, 102.999, 5_000))
    assert hits / 5_000 > 0.9


# --------------------------------------------------------------- n-step ----
def test_nstep_return_is_correctly_discounted():
    gamma, n = 0.5, 3
    acc = NStepAccumulator(n, gamma)
    out = None
    for i in range(n):
        out = acc.append(np.zeros(1, np.float32), i, float(i + 1),
                         np.zeros(1, np.float32), False)
    # rewards 1, 2, 3 -> 1 + 0.5*2 + 0.25*3 = 2.75
    assert out is not None
    assert out[2] == pytest.approx(2.75)


def test_nstep_emits_nothing_before_horizon_and_flushes_tail():
    acc = NStepAccumulator(3, 0.99)
    assert acc.append(np.zeros(1, np.float32), 0, 1.0, np.zeros(1, np.float32), False) is None
    assert acc.append(np.zeros(1, np.float32), 1, 1.0, np.zeros(1, np.float32), False) is None
    assert acc.append(np.zeros(1, np.float32), 2, 1.0, np.zeros(1, np.float32), True) is not None
    assert len(acc.flush()) == 2  # the two shortened tail transitions


# --------------------------------------------------------------- replay ----
def _fill(buf, n=300):
    for i in range(n):
        buf.add(np.random.randn(STATE).astype(np.float32), i % ACTIONS,
                float(i % 3 - 1), np.random.randn(STATE).astype(np.float32), i % 50 == 0)
    return buf


def test_uniform_buffer_shapes_and_unit_weights():
    buf = _fill(UniformReplayBuffer(1000, STATE))
    b = buf.sample(64, torch.device("cpu"))
    assert b.states.shape == (64, STATE)
    assert b.actions.shape == (64, 1) and b.actions.dtype == torch.int64
    assert torch.allclose(b.weights, torch.ones_like(b.weights))


def test_ring_buffer_wraps_and_caps_length():
    buf = _fill(UniformReplayBuffer(100, STATE), n=250)
    assert len(buf) == 100


def test_per_weights_normalised_and_beta_anneals():
    buf = _fill(PrioritizedReplayBuffer(1000, STATE, beta_frames=100))
    start_beta = buf.beta
    b = buf.sample(32, torch.device("cpu"))
    assert float(b.weights.max()) == pytest.approx(1.0, abs=1e-5)
    assert (b.weights > 0).all()
    buf.update_priorities(b.indices, np.random.rand(32) * 5)
    for _ in range(200):
        buf.sample(32, torch.device("cpu"))
    assert buf.beta > start_beta
    assert buf.beta == pytest.approx(1.0)


def test_per_prioritises_high_td_error_transitions():
    buf = _fill(PrioritizedReplayBuffer(200, STATE), n=200)
    idx = np.arange(200)
    td = np.full(200, 0.01)
    td[7] = 500.0                      # one very surprising transition
    buf.update_priorities(idx, td)
    drawn = np.concatenate([buf.sample(64, torch.device("cpu")).indices for _ in range(20)])
    assert (drawn == 7).mean() > 0.5


# ---------------------------------------------------------------- agent ----
@pytest.mark.parametrize("kwargs", [
    {},
    {"double": True},
    {"dueling": True},
    {"prioritized": True},
    {"noisy": True},
    {"n_step": 3},
    {"double": True, "dueling": True, "prioritized": True, "noisy": True, "n_step": 3},
], ids=["dqn", "double", "dueling", "per", "noisy", "nstep", "rainbow"])
def test_agent_trains_one_step_for_every_variant(kwargs):
    cfg = AgentConfig(STATE, ACTIONS, learn_starts=32, batch_size=16, **kwargs)
    agent = DQNAgent(cfg, seed=0)
    state = np.random.randn(STATE).astype(np.float32)
    for i in range(200):
        action = agent.act(state, eps=0.1)
        assert 0 <= action < ACTIONS
        nxt = np.random.randn(STATE).astype(np.float32)
        agent.step(state, action, 1.0, nxt, i % 40 == 0)
        state = nxt
    assert np.isfinite(agent.last_loss), "a learning step should have run"


def test_target_network_tracks_but_lags_online_network():
    cfg = AgentConfig(STATE, ACTIONS, tau=0.5)
    agent = DQNAgent(cfg, seed=0)
    for p in agent.qnet_local.parameters():
        p.data.fill_(1.0)
    for p in agent.qnet_target.parameters():
        p.data.fill_(0.0)
    agent._soft_update()
    got = next(agent.qnet_target.parameters()).data.flatten()[0]
    assert got == pytest.approx(0.5)


def test_checkpoint_roundtrip_preserves_policy(tmp_path):
    cfg = AgentConfig(STATE, ACTIONS, dueling=True, noisy=True, n_step=3)
    agent = DQNAgent(cfg, seed=1)
    path = tmp_path / "ckpt.pth"
    agent.save(path)

    restored = DQNAgent.load(path)
    assert restored.cfg.dueling and restored.cfg.noisy and restored.cfg.n_step == 3
    states = np.random.randn(25, STATE).astype(np.float32)
    a = [agent.act(s, greedy=True) for s in states]
    b = [restored.act(s, greedy=True) for s in states]
    assert a == b, "a reloaded agent must reproduce the greedy policy exactly"


def test_noisy_agent_explores_without_epsilon():
    """NoisyNet must produce varied actions at eps=0, where plain DQN cannot."""
    plain = DQNAgent(AgentConfig(STATE, ACTIONS, noisy=False), seed=0)
    noisy = DQNAgent(AgentConfig(STATE, ACTIONS, noisy=True), seed=0)
    state = np.random.randn(STATE).astype(np.float32)
    assert len({plain.act(state, eps=0.0) for _ in range(40)}) == 1
    assert len({noisy.act(state, eps=0.0) for _ in range(40)}) > 1


# --------------------------------------------------------------- config ----
def test_every_shipped_config_builds():
    expected = {"dqn", "double", "dueling", "per", "nstep", "noisy", "rainbow"}
    assert expected.issubset(set(list_configs()))
    for name in list_configs():
        variant, acfg, tcfg = build(name, STATE, ACTIONS)
        assert acfg.state_size == STATE and acfg.action_size == ACTIONS
        assert tcfg.n_episodes > 0
        DQNAgent(acfg, seed=0)


def test_config_overrides_route_to_the_right_dataclass():
    _, acfg, tcfg = build("dqn", STATE, ACTIONS,
                          overrides={"n_episodes": 42, "lr": 0.1})
    assert tcfg.n_episodes == 42     # TrainConfig field
    assert acfg.lr == pytest.approx(0.1)   # AgentConfig field


def test_rainbow_config_enables_every_component():
    _, acfg, _ = build("rainbow", STATE, ACTIONS)
    assert acfg.double and acfg.dueling and acfg.prioritized and acfg.noisy
    assert acfg.n_step > 1
