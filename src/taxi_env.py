"""Taxi-v4 environment helpers."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordEpisodeStatistics


def make_taxi_env(seed: int | None = None, record_stats: bool = False) -> gym.Env:
    """Create Taxi-v4 with optional seeding and episode statistics wrapper."""
    env = gym.make("Taxi-v4")
    if record_stats:
        env = RecordEpisodeStatistics(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_action_mask(info: dict) -> list[int]:
    """Return indices of valid actions from env info."""
    mask = info.get("action_mask")
    if mask is None:
        return list(range(6))
    return [i for i, valid in enumerate(mask) if valid]


def action_mask_array(info: dict, n_actions: int = 6) -> np.ndarray:
    """Boolean mask of shape (n_actions,) for valid actions."""
    mask = info.get("action_mask")
    if mask is None:
        return np.ones(n_actions, dtype=bool)
    return np.asarray(mask, dtype=bool)
