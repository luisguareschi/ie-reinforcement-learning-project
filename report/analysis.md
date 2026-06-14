# Taxi-v4 Analysis Report

Made By: Luis Guareschi

Environment: [Gymnasium Taxi-v4](https://gymnasium.farama.org/environments/toy_text/taxi/) — 500 discrete states, 6 actions, rewards of **-1/step**, **+20** delivery, **-10** illegal pickup/dropoff.

## Baseline comparison

| Metric | Q-learning | DQN |
|--------|------------|-----|
| Final avg reward (last 100 ep) | 6.63 | 7.02 |
| Final success rate (last 100 ep) | 0.98 | 1.00 |
| Eval mean reward | 7.34 | 3.32 |
| Eval success rate | 1.00 | 0.98 |
| Avg steps (successful ep) | 14.3 | 14.8 |
| Training time (s) | 4.9 | 25.5 |
| Total episodes | 15,000 | 4,000 |

**Figures:** `results/q_learning/baseline/learning_curve.png`, `results/dqn/baseline/learning_curve.png`

### Interpretation

Both algorithms solve Taxi-v4. Tabular Q-learning reaches strong eval performance (7.34 mean reward) with very little compute (~5 s for 15k episodes). DQN matches training performance (7.02 rolling reward, 100% success in last 100 episodes) but eval reward is lower (3.32), suggesting some overfitting or instability in the greedy policy despite good training metrics.

Q-learning is the natural fit: the state space is small and fully observable, so storing one value per `(state, action)` avoids function approximation error entirely.


## Hyperparameter sweep

Full results: `results/sweep_summary.csv`

### Q-learning

| Parameter | Best observed | Worst observed | Impact |
|-----------|---------------|----------------|--------|
| α (0.1 / 0.5 / 0.8) | 7.45 (α=0.1) | 7.00 (α=0.8) | Moderate — lower α slightly more stable |
| γ (0.9 / 0.99) | 7.20 (γ=0.99) | **-81.97 (γ=0.9)** | **Critical** — low γ prevents long-horizon credit assignment |
| ε-decay (1500 / 2500 / 4000) | 7.37 (1500) | 1.16 (4000) | High — slow decay leaves too much exploration |

**Key finding:** γ=0.99 is essential. With γ=0.9 the agent barely learns the multi-step pickup→dropoff sequence within 3,000 episodes. Fast ε-decay (1,500 ep) works best; decay over 4,000 ep keeps ε high too long and hurts late-training reward.

### DQN

| Parameter | Best observed (train) | Notes |
|-----------|----------------------|-------|
| lr (1e-4 / 5e-4 / 1e-3) | 7.53 (1e-3 train) | Higher lr learns faster but eval can degrade (-0.8) |
| weight_decay (0 / 1e-4 / 1e-3) | 7.46 eval (wd=0 or 1e-3) | Regularization helps eval generalization |
| hidden_size (64 / 128 / 256) | 7.46 eval (128) | Larger nets slower; 64 is enough for Taxi |

**Key finding:** DQN is more sensitive to hyperparameters and shows a train/eval gap. Weight decay=0 or 0.001 gave the best eval rewards (7.46). Learning rate 5e-4 with hidden size 64 is a good default for speed and stability.

## Q-learning vs DQN — trade-offs

| | Q-learning | DQN |
|---|------------|-----|
| **State representation** | Exact tabular | Embedding + shallow MLP |
| **Sample efficiency** | High — converges in 3k ep | Moderate — needs replay + 4k ep |
| **Compute** | Very fast (numpy) | ~5× slower (PyTorch) |
| **Stability** | Robust once γ and ε are set | Requires masked targets + tuning |
| **Scalability** | Only small discrete spaces | Extends to large/continuous obs |

For Taxi-v4, **tabular Q-learning is simpler, faster, and more reliable**. DQN demonstrates that function approximation can work on the same problem but needs careful handling of action masking and regularization — skills that transfer to larger environments (e.g. Pong) where tabular methods are infeasible.