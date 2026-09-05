# Report — Navigation (Banana Collector)

## 1. Problem

An agent navigates a square arena collecting bananas. Yellow bananas pay `+1`,
blue bananas pay `-1`. The state is a 37-dimensional continuous vector (agent
velocity plus ray-based perception of nearby objects along the forward
direction); four discrete actions move forward, move backward, turn left and
turn right. Episodes last 300 steps.

**The environment is solved when the average score over 100 consecutive episodes
reaches +13.** The project benchmark is to reach that inside 1800 episodes.

---

## 2. Learning algorithm

The foundation is **Deep Q-Learning** (Mnih et al., 2015). A neural network
`Q(s, a; θ)` approximates the action-value function, trained to minimise the
temporal-difference error against a bootstrapped target:

```
L(θ) = E[ ( r + γ · max_a' Q(s', a'; θ⁻) − Q(s, a; θ) )² ]
```

Two mechanisms make this stable, and both are used here:

* **Experience replay** — transitions are stored in a buffer and sampled in
  random minibatches, breaking the temporal correlation that would otherwise
  make consecutive updates highly dependent.
* **A target network** `θ⁻` — a slowly-tracking copy of the online weights used
  to compute targets, so the regression target does not move with every update.
  This implementation uses a **soft update**, blending `τ = 0.001` of the online
  weights into the target after every learning step, rather than periodic hard
  copies.

On top of that baseline, five extensions from the Rainbow paper (Hessel et al.,
2018) are implemented as **independent switches**, so each can be measured alone
and in combination. All that varies between the variants below is
`AgentConfig`; the code path is identical.

### 2.1 Double DQN (van Hasselt et al., 2016)

The `max` in the standard target both selects and evaluates the next action
using the same network, which systematically overestimates action values
whenever the estimates are noisy. Double DQN separates the two roles — the
**online** network picks the action, the **target** network scores it:

```
target = r + γ · Q(s', argmax_a' Q(s', a'; θ) ; θ⁻)
```

### 2.2 Dueling networks (Wang et al., 2016)

The head splits into a state-value stream `V(s)` and an advantage stream
`A(s, a)`, recombined as:

```
Q(s, a) = V(s) + ( A(s, a) − mean_a' A(s, a') )
```

Subtracting the mean advantage makes the decomposition identifiable. The value
of being in a good position is learned once per state rather than separately for
all four actions — useful here, where many states are ones in which the exact
action barely matters.

### 2.3 Prioritized experience replay (Schaul et al., 2016)

Sampling uniformly wastes capacity on transitions the network already predicts
well. Instead transitions are sampled proportional to their last TD error:

```
P(i) = p_i^α / Σ_k p_k^α          p_i = |δ_i| + ε
```

Because that biases the update distribution, gradients are corrected with
importance-sampling weights `w_i = (N · P(i))^(−β)`, normalised by their
maximum, with `β` annealed 0.4 → 1.0 so the correction becomes exact as training
converges.

Sampling is implemented over a **sum tree**, giving O(log n) proportional draws
rather than O(n) per sample, and uses **stratified sampling** — one draw from
each of `batch_size` equal-probability segments — which lowers variance relative
to independent draws. New transitions enter at maximum priority so every
transition is replayed at least once.

### 2.4 Multi-step (n-step) returns

Instead of bootstrapping after one transition, the target accumulates `n` real
rewards first:

```
target = Σ_{k=0}^{n−1} γ^k · r_{t+k}  +  γ^n · max_a Q(s_{t+n}, a; θ⁻)
```

This propagates reward information backwards `n` times faster, at the cost of a
slightly off-policy target. `n = 3` here.

### 2.5 NoisyNet exploration (Fortunato et al., 2018)

Epsilon-greedy explores uniformly at random forever, at a rate set by a schedule
that ignores what the agent has learned. NoisyNet instead adds learnable
Gaussian noise to the head's weights:

```
y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)
```

with **factorised** noise (`ε_w = f(ε_out) f(ε_in)ᵀ`, `f(x) = sgn(x)√|x|`),
which needs `p + q` normal samples per layer instead of `p × q`. The network can
learn to shrink `σ` where it is already confident and keep exploring elsewhere.
**When NoisyNet is enabled, epsilon is fixed at 0** — layering epsilon-greedy on
top would double-count exploration.

---

## 3. Network architecture

A small MLP; the 37-dimensional state needs nothing convolutional.

```
Input (37)
  └─ Linear(37 → 128) + ReLU          shared body
       ├─ [standard head]  Linear(128 → 64) + ReLU → Linear(64 → 4)
       └─ [dueling head]   value:     Linear(128 → 64) + ReLU → Linear(64 → 1)
                           advantage: Linear(128 → 64) + ReLU → Linear(64 → 4)
                           Q = V + (A − mean(A))
```

With `noisy: true`, every `Linear` **in the head** becomes a `NoisyLinear`; the
shared body stays deterministic. Parameter counts:

| Variant | Parameters |
|---|---|
| Standard | 13,380 |
| Dueling | 21,701 |
| Noisy | 21,896 |
| Dueling + Noisy | 38,538 |

---

## 4. Hyperparameters

Identical across every variant, so ablation differences are attributable to the
components rather than to tuning.

| Parameter | Value | Note |
|---|---|---|
| Replay buffer size | 100,000 | |
| Batch size | 64 | |
| Discount `γ` | 0.99 | |
| Learning rate | 5e-4 | Adam |
| Soft update `τ` | 1e-3 | applied every learning step |
| Learn every | 4 env steps | |
| Warm-up before learning | 1,000 transitions | |
| Gradient clipping | 10.0 | max grad norm |
| Loss | Huber (smooth L1) | less sensitive to outlier TD errors than MSE |
| `ε` start / end / decay | 1.0 / 0.01 / 0.995 | multiplicative per episode; **forced to 0 when NoisyNet is on** |
| PER `α` | 0.6 | prioritization exponent |
| PER `β` | 0.4 → 1.0 | annealed over 100k samples |
| PER `ε` | 1e-3 | priority floor |
| n-step | 3 | in `nstep` and `rainbow` only |
| Hidden layers | (128, 64) | |
| Episodes per run | 900 | |
| Seeds per variant | 5 | 0–4 |

---

## 5. Results

<!--HEADLINE-RESULT-->

### 5.1 Learning curve

![Learning curve](assets/learning_curve.png)

### 5.2 Component ablation

Each variant was trained for 900 episodes across 5 seeds (0–4). Reporting a
single run would be misleading — seed variance on this task is large enough to
reverse the apparent ranking of two components — so results are medians with
interquartile bands.

![Ablation curves](assets/ablation_curves.png)

![Episodes to solve](assets/episodes_to_solve.png)

<!--ABLATION-TABLE-->

<!--ABLATION-DISCUSSION-->

### 5.3 Greedy evaluation

<!--EVAL-RESULT-->

---

## 6. Ideas for future work

**Distributional RL (C51 / QR-DQN).** The one Rainbow component not implemented
here. Instead of regressing the *expected* return, learn a distribution over
returns. In Hessel et al.'s own ablation this was among the largest single
contributors, and it is the obvious next addition.

**Proper hyperparameter search.** Every variant deliberately shares one
hyperparameter set so the ablation isolates architecture from tuning. That is
the right call for a *comparison*, but it certainly understates the best
achievable result — prioritized replay in particular is known to prefer a lower
learning rate, since it already amplifies high-error transitions.

**Tune the ε schedule against episode budget.** `0.995` decay reaches ε=0.01
around episode 900, which means a meaningful fraction of a 500-episode solve is
still spent exploring. A faster decay would likely improve every epsilon-greedy
variant; that it was left alone is precisely why the NoisyNet comparison is
interesting.

**Learning from pixels.** The 84×84×3 first-person build turns this into a
representation-learning problem needing a CNN and frame stacking. Realistically
requires a GPU — a rough extrapolation from the CPU throughput measured here
puts it in the multi-day range on this hardware.

**Better exploration than either mechanism here.** The reward signal is dense,
so epsilon-greedy is adequate. Under sparse reward, count-based or curiosity-driven
exploration would matter far more than any of the value-function refinements
measured above.

**Sharper statistics.** Five seeds supports medians and interquartile ranges but
not strong significance claims about closely-ranked components. Rliable-style
bootstrapped confidence intervals over more seeds would let the near-ties be
called properly rather than merely displayed.

---

## 7. A note on the environment port

Getting the 2018 Unity build talking to a modern Python stack was a substantial
part of this project, and the four bugs involved — one of which silently
serialises parallel training with no error message at all — are documented
separately in **[docs/PORTING.md](docs/PORTING.md)**.

That work is what made the 5-seed ablation feasible: it took parallel training
from one working lane to six, which is the difference between a ~30-hour serial
study and an overnight one.
