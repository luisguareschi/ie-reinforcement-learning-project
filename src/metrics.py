"""Training metrics, logging, and artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EpisodeMetrics:
    episode: int
    reward: float
    steps: int
    success: bool
    epsilon: float
    loss: float | None = None


def metrics_to_dataframe(records: list[EpisodeMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def rolling_mean(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    w = min(window, len(values))
    return float(np.mean(values[-w:]))


def plot_learning_curve(
    df: pd.DataFrame,
    output_path: Path,
    window: int = 100,
    title: str = "Learning Curve",
) -> None:
    rewards = df["reward"].to_numpy()
    episodes = df["episode"].to_numpy()
    rolling = pd.Series(rewards).rolling(window=window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.25, label="Episode reward")
    ax.plot(episodes, rolling, linewidth=2, label=f"Rolling mean ({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def save_run_artifacts(
    output_dir: Path,
    config: dict,
    metrics: list[EpisodeMetrics],
    algo_name: str,
    rolling_window: int = 100,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = metrics_to_dataframe(metrics)
    df.to_csv(output_dir / "metrics.csv", index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    plot_learning_curve(
        df,
        output_dir / "learning_curve.png",
        window=rolling_window,
        title=f"{algo_name} — Taxi-v4",
    )
    return df


def summarize_run(df: pd.DataFrame, training_time_sec: float, window: int = 100) -> dict:
    last = df.tail(window)
    successes = df[df["success"]]
    return {
        "final_avg_reward": float(last["reward"].mean()),
        "final_success_rate": float(last["success"].mean()),
        "avg_steps_success": float(successes["steps"].mean()) if len(successes) else float("nan"),
        "total_episodes": int(len(df)),
        "training_time_sec": training_time_sec,
    }
