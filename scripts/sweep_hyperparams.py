#!/usr/bin/env python3
"""Hyperparameter sweep for Q-learning and DQN on Taxi-v4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# Shared fast DQN settings that converge reliably on Taxi-v4 (~25s / 4000 ep on CPU)
DQN_COMMON_ARGS = ["--train-freq", "4", "--batch-size", "128", "--decay-episodes", "3000"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep hyperparameters on Taxi-v4")
    p.add_argument("--output-root", type=Path, default=ROOT / "results")
    p.add_argument("--q-episodes", type=int, default=3000)
    p.add_argument("--dqn-episodes", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def build_sweep() -> list[dict]:
    runs: list[dict] = []

    for alpha in [0.1, 0.5, 0.8]:
        runs.append({
            "algo": "q_learning",
            "run_id": f"ql_alpha_{alpha}",
            "args": ["--alpha", str(alpha), "--gamma", "0.99", "--decay-episodes", "2500"],
        })

    for gamma in [0.9, 0.99]:
        runs.append({
            "algo": "q_learning",
            "run_id": f"ql_gamma_{gamma}",
            "args": ["--alpha", "0.5", "--gamma", str(gamma), "--decay-episodes", "2500"],
        })

    for decay in [1500, 2500, 4000]:
        runs.append({
            "algo": "q_learning",
            "run_id": f"ql_decay_{decay}",
            "args": ["--alpha", "0.5", "--gamma", "0.99", "--decay-episodes", str(decay)],
        })

    for lr in [1e-4, 5e-4, 1e-3]:
        runs.append({
            "algo": "dqn",
            "run_id": f"dqn_lr_{lr}",
            "args": ["--lr", str(lr), "--weight-decay", "1e-4", "--hidden-size", "64"],
        })

    for wd in [0.0, 1e-4, 1e-3]:
        runs.append({
            "algo": "dqn",
            "run_id": f"dqn_wd_{wd}",
            "args": ["--lr", "5e-4", "--weight-decay", str(wd), "--hidden-size", "64"],
        })

    for hidden in [64, 128, 256]:
        runs.append({
            "algo": "dqn",
            "run_id": f"dqn_hidden_{hidden}",
            "args": ["--lr", "5e-4", "--weight-decay", "1e-4", "--hidden-size", str(hidden)],
        })

    return runs


def run_job(
    run: dict,
    output_root: Path,
    q_episodes: int,
    dqn_episodes: int,
    seed: int,
    no_progress: bool,
) -> dict:
    algo = run["algo"]
    out_dir = output_root / algo / "sweep" / run["run_id"]
    script = ROOT / "scripts" / ("train_q_learning.py" if algo == "q_learning" else "train_dqn.py")

    cmd = [
        sys.executable,
        str(script),
        "--output-dir",
        str(out_dir),
        "--seed",
        str(seed),
    ]
    if algo == "q_learning":
        cmd.extend(["--episodes", str(q_episodes)])
    else:
        cmd.extend(["--episodes", str(dqn_episodes)])
        cmd.extend(DQN_COMMON_ARGS)
    cmd.extend(run["args"])
    if no_progress:
        cmd.append("--no-progress")

    subprocess.run(cmd, check=True, cwd=ROOT)

    with open(out_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)

    with open(out_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    row = {
        "algo": algo,
        "run_id": run["run_id"],
        "output_dir": str(out_dir),
        **{k: v for k, v in config.items() if k not in ("algo",)},
        **{k: v for k, v in summary.items() if k != "eval"},
        "eval_mean_reward": summary.get("eval", {}).get("mean_reward"),
        "eval_success_rate": summary.get("eval", {}).get("success_rate"),
    }
    return row


def main() -> None:
    args = parse_args()
    runs = build_sweep()
    rows: list[dict] = []

    iterator = tqdm(runs, desc="Hyperparameter sweep", unit="run")
    for run in iterator:
        label = f"{run['algo']} {run['run_id']}"
        iterator.set_description(f"Sweep: {label}")
        row = run_job(
            run,
            args.output_root,
            args.q_episodes,
            args.dqn_episodes,
            args.seed,
            args.no_progress,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_path = args.output_root / "sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"Saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
