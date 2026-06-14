#!/usr/bin/env python3
"""Train DQN on Taxi-v4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqn import DQNAgent, DQNConfig, evaluate_dqn, train_dqn
from src.metrics import save_run_artifacts, summarize_run
from src.taxi_env import make_taxi_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN on Taxi-v4")
    p.add_argument("--episodes", type=int, default=4000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-capacity", type=int, default=10000)
    p.add_argument("--target-update-freq", type=int, default=500)
    p.add_argument("--train-freq", type=int, default=4)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--decay-episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "dqn" / "baseline")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = DQNConfig(
        lr=args.lr,
        gamma=args.gamma,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        train_freq=args.train_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        decay_episodes=args.decay_episodes,
    )

    env = make_taxi_env(seed=args.seed)
    agent = DQNAgent(config, seed=args.seed)

    metrics, elapsed = train_dqn(
        env,
        agent,
        args.episodes,
        show_progress=not args.no_progress,
    )

    eval_metrics = evaluate_dqn(env, agent, n_episodes=50, seed=args.seed)
    summary = summarize_run(save_run_artifacts(
        args.output_dir,
        {**config.__dict__, "episodes": args.episodes, "seed": args.seed, "algo": "dqn"},
        metrics,
        "DQN",
    ), elapsed)

    agent.save(args.output_dir / "agent.pt")

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "eval": eval_metrics}, f, indent=2)

    print(f"Saved run to {args.output_dir}")
    print(f"Final avg reward (last 100): {summary['final_avg_reward']:.2f}")
    print(f"Eval mean reward: {eval_metrics['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
