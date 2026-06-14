#!/usr/bin/env python3
"""Train tabular Q-learning on Taxi-v4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import save_run_artifacts, summarize_run
from src.q_learning import QLearningAgent, QLearningConfig, evaluate_q_learning, train_q_learning
from src.taxi_env import make_taxi_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Q-learning on Taxi-v4")
    p.add_argument("--episodes", type=int, default=8000)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--decay-episodes", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "q_learning" / "baseline")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = QLearningConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        decay_episodes=args.decay_episodes,
    )

    env = make_taxi_env(seed=args.seed)
    agent = QLearningAgent(config, seed=args.seed)

    metrics, elapsed = train_q_learning(
        env,
        agent,
        args.episodes,
        show_progress=not args.no_progress,
    )

    eval_metrics = evaluate_q_learning(env, agent, n_episodes=50, seed=args.seed)
    summary = summarize_run(save_run_artifacts(
        args.output_dir,
        {**config.__dict__, "episodes": args.episodes, "seed": args.seed, "algo": "q_learning"},
        metrics,
        "Q-learning",
    ), elapsed)

    agent.save(str(args.output_dir / "q_table.npy"))

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({**summary, "eval": eval_metrics}, f, indent=2)

    print(f"Saved run to {args.output_dir}")
    print(f"Final avg reward (last 100): {summary['final_avg_reward']:.2f}")
    print(f"Eval mean reward: {eval_metrics['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
