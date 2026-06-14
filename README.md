# IE Reinforcement Learning Project — Taxi

Solve the [Gymnasium Taxi-v4](https://gymnasium.farama.org/environments/toy_text/taxi/) environment using tabular Q-learning and a shallow regularized DQN.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Train

**Tabular Q-learning (baseline):**

```bash
python scripts/train_q_learning.py \
  --episodes 8000 \
  --alpha 0.5 \
  --gamma 0.99 \
  --output-dir results/q_learning/baseline
```

**DQN (baseline):**

```bash
python scripts/train_dqn.py \
  --episodes 4000 \
  --train-freq 4 \
  --batch-size 128 \
  --hidden-size 64 \
  --decay-episodes 3000 \
  --output-dir results/dqn/baseline
```

Defaults above converge in ~30s on CPU. DQN uses masked Double DQN with action-masked bootstrap targets (required for Taxi-v4).

**Hyperparameter sweep** (runs a small grid for both algorithms):

```bash
python scripts/sweep_hyperparams.py
```

Use `--no-progress` on any script to disable `tqdm` bars.

### Results

Each run writes to its `--output-dir`:

- `config.json` — hyperparameters
- `metrics.csv` — per-episode metrics
- `learning_curve.png` — reward over training
- `q_table.npy` or `agent.pt` — saved agent

Sweep output: `results/sweep_summary.csv`

---

## Visual demo

Run a saved agent with the Gymnasium renderer (interactive arrow-key picker):

```bash
python scripts/demo_taxi.py
```

Use ↑/↓ to navigate, Enter to select. Models are grouped by algorithm with ★ recommendations.

Pick a model by path to skip the menu:

```bash
python scripts/demo_taxi.py --run dqn/baseline
python scripts/demo_taxi.py --run 1 --episodes 5 --delay 0.3
python scripts/demo_taxi.py --render-mode ansi   # terminal-only
```

---

## Analysis

Open the notebook for plots and interpretation:

```bash
jupyter notebook notebooks/taxi_analysis.ipynb
```

Written report: [report/analysis.md](report/analysis.md)