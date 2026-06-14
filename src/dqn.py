"""Shallow DQN with regularization for Taxi-v4."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.metrics import EpisodeMetrics, rolling_mean
from src.replay_buffer import ReplayBuffer, Transition
from src.taxi_env import action_mask_array, get_action_mask


@dataclass
class DQNConfig:
    n_states: int = 500
    n_actions: int = 6
    hidden_size: int = 128
    dropout: float = 0.0
    gamma: float = 0.99
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    buffer_capacity: int = 10_000
    min_buffer_size: int = 256
    target_update_freq: int = 500
    train_freq: int = 1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    decay_episodes: int = 4000


class QNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(n_states, hidden_size)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(state_ids)
        return self.head(x)


class DQNAgent:
    def __init__(self, config: DQNConfig, seed: int | None = None, device: str | None = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed or 0)

        self.policy_net = QNetwork(
            config.n_states, config.n_actions, config.hidden_size, config.dropout
        ).to(self.device)
        self.target_net = QNetwork(
            config.n_states, config.n_actions, config.hidden_size, config.dropout
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.buffer = ReplayBuffer(config.buffer_capacity)
        self.epsilon = config.epsilon_start
        self.env_steps = 0
        self.train_steps = 0
        self.last_loss: float | None = None

    def _state_tensor(self, states: np.ndarray) -> torch.Tensor:
        return torch.tensor(states, dtype=torch.long, device=self.device)

    def q_values(self, state: int) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            x = self._state_tensor(np.array([state]))
            return self.policy_net(x).cpu().numpy()[0]

    def epsilon_for_episode(self, episode: int) -> float:
        cfg = self.config
        if episode >= cfg.decay_episodes:
            return cfg.epsilon_end
        frac = episode / cfg.decay_episodes
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    def select_action(self, state: int, action_mask: list[int], explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(action_mask))
        q = self.q_values(state)
        masked_q = [(a, q[a]) for a in action_mask]
        return max(masked_q, key=lambda x: x[1])[0]

    def store(self, transition: Transition) -> None:
        self.buffer.push(transition)

    def train_step(self) -> float | None:
        cfg = self.config
        if not self.buffer.is_ready(cfg.batch_size):
            return None

        self.policy_net.train()
        batch = self.buffer.sample(cfg.batch_size, self.rng)
        states = np.array([t.state for t in batch])
        actions = np.array([t.action for t in batch])
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch])
        terminated = np.array([t.terminated for t in batch], dtype=np.float32)
        next_masks = np.stack([t.next_action_mask for t in batch])

        state_t = self._state_tensor(states)
        next_state_t = self._state_tensor(next_states)
        action_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_t = torch.tensor(rewards, device=self.device)
        done_t = torch.tensor(terminated, device=self.device)
        mask_t = torch.tensor(next_masks, dtype=torch.bool, device=self.device)

        q_sa = self.policy_net(state_t).gather(1, action_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_policy = self.policy_net(next_state_t)
            next_q_policy = next_q_policy.masked_fill(~mask_t, float("-inf"))
            best_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_state_t).gather(1, best_actions).squeeze(1)
            target = reward_t + cfg.gamma * next_q_target * (1.0 - done_t)
            target = torch.clamp(target, -200.0, 25.0)

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.last_loss = float(loss.item())
        return self.last_loss

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )


def train_dqn(
    env,
    agent: DQNAgent,
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
        iterator = tqdm(iterator, desc="DQN", unit="ep")

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
            agent.store(
                Transition(
                    state,
                    action,
                    reward,
                    next_state,
                    terminated,
                    action_mask_array(info),
                )
            )

            agent.env_steps += 1
            if (
                len(agent.buffer) >= agent.config.min_buffer_size
                and agent.env_steps % agent.config.train_freq == 0
            ):
                agent.train_step()

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
                loss=agent.last_loss,
            )
        )
        recent_rewards.append(total_reward)
        recent_success.append(success)

        if show_progress and hasattr(iterator, "set_postfix"):
            postfix = {
                "reward": f"{rolling_mean(recent_rewards, rolling_window):.1f}",
                "eps": f"{agent.epsilon:.3f}",
                "buf": len(agent.buffer),
            }
            if agent.last_loss is not None:
                postfix["loss"] = f"{agent.last_loss:.3f}"
            iterator.set_postfix(postfix)

    elapsed = time.perf_counter() - start
    return records, elapsed


def evaluate_dqn(
    env,
    agent: DQNAgent,
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
