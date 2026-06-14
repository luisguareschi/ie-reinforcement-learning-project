"""Tabular Q-learning for Taxi-v4."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src.metrics import EpisodeMetrics, rolling_mean
from src.taxi_env import get_action_mask


@dataclass
class QLearningConfig:
    alpha: float = 0.5
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_episodes: int = 8000
    n_states: int = 500
    n_actions: int = 6


class QLearningAgent:
    def __init__(self, config: QLearningConfig, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((config.n_states, config.n_actions), dtype=np.float64)
        self.epsilon = config.epsilon_start

    def epsilon_for_episode(self, episode: int) -> float:
        cfg = self.config
        if episode >= cfg.decay_episodes:
            return cfg.epsilon_end
        frac = episode / cfg.decay_episodes
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def select_action(self, state: int, action_mask: list[int], explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(action_mask))
        q_vals = self.Q[state, action_mask]
        best_idx = int(np.argmax(q_vals))
        return action_mask[best_idx]

    def update(self, state: int, action: int, reward: float, next_state: int, terminated: bool) -> None:
        cfg = self.config
        next_max = 0.0 if terminated else float(np.max(self.Q[next_state]))
        td_target = reward + cfg.gamma * next_max
        self.Q[state, action] += cfg.alpha * (td_target - self.Q[state, action])

    def save(self, path: str) -> None:
        np.save(path, self.Q)


def train_q_learning(
    env,
    agent: QLearningAgent,
    n_episodes: int,
    *,
    show_progress: bool = True,
    success_reward_threshold: float = 0.0,
    rolling_window: int = 100,
) -> tuple[list[EpisodeMetrics], float]:
    records: list[EpisodeMetrics] = []
    recent_rewards: list[float] = []
    recent_success: list[bool] = []
    start = time.perf_counter()

    iterator = range(n_episodes)
    if show_progress:
        iterator = tqdm(iterator, desc="Q-learning", unit="ep")

    for episode in iterator:
        agent.epsilon = agent.epsilon_for_episode(episode)
        state, info = env.reset()
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            mask = get_action_mask(info)
            action = agent.select_action(state, mask, explore=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            steps += 1

        success = terminated and total_reward >= success_reward_threshold
        records.append(
            EpisodeMetrics(
                episode=episode + 1,
                reward=total_reward,
                steps=steps,
                success=success,
                epsilon=agent.epsilon,
            )
        )
        recent_rewards.append(total_reward)
        recent_success.append(success)

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                reward=rolling_mean(recent_rewards, rolling_window),
                eps=f"{agent.epsilon:.3f}",
                success=f"{rolling_mean([float(s) for s in recent_success], rolling_window):.2f}",
            )

    elapsed = time.perf_counter() - start
    return records, elapsed


def evaluate_q_learning(
    env,
    agent: QLearningAgent,
    n_episodes: int = 100,
    seed: int | None = 42,
) -> dict:
    rewards: list[float] = []
    steps_list: list[int] = []
    successes: list[bool] = []

    for ep in range(n_episodes):
        state, info = env.reset(seed=(seed + ep) if seed is not None else None)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            mask = get_action_mask(info)
            action = agent.select_action(state, mask, explore=False)
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)
        steps_list.append(steps)
        successes.append(terminated and total_reward > 0)

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_list)),
        "success_rate": float(np.mean(successes)),
    }
