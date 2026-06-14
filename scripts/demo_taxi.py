#!/usr/bin/env python3
"""Visual demo: run a saved Q-learning or DQN agent on Taxi-v4."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import questionary
import torch
from questionary import Choice, Separator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dqn import DQNAgent, DQNConfig
from src.q_learning import QLearningAgent, QLearningConfig
from src.taxi_env import get_action_mask

ACTION_NAMES = ["south", "north", "east", "west", "pickup", "dropoff"]

MENU_STYLE = questionary.Style([
    ("qmark", "fg:cyan bold"),
    ("question", "bold"),
    ("answer", "fg:cyan bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("separator", "fg:ansibrightblack"),
])


@dataclass
class ModelRun:
    rel_path: str
    algo: str
    run_dir: Path
    eval_reward: float | None = None
    eval_success: float | None = None
    tags: list[str] = field(default_factory=list)


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_runs(results_dir: Path) -> list[ModelRun]:
    runs: list[ModelRun] = []
    if not results_dir.exists():
        return runs

    for path in sorted(results_dir.rglob("*")):
        if not path.is_dir():
            continue
        has_q = (path / "q_table.npy").exists()
        has_dqn = (path / "agent.pt").exists()
        if not (has_q or has_dqn):
            continue

        algo = "q_learning" if has_q else "dqn"
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                algo = json.load(f).get("algo", algo)

        summary = load_summary(path)
        eval_block = summary.get("eval", {})
        runs.append(
            ModelRun(
                rel_path=str(path.relative_to(results_dir)),
                algo=algo,
                run_dir=path,
                eval_reward=eval_block.get("mean_reward"),
                eval_success=eval_block.get("success_rate"),
            )
        )

    _apply_recommendation_tags(runs)
    return runs


def _apply_recommendation_tags(runs: list[ModelRun]) -> None:
    ql = [r for r in runs if r.algo == "q_learning"]
    dqn = [r for r in runs if r.algo == "dqn"]

    for run in ql:
        if run.rel_path == "q_learning/baseline":
            run.tags.append("recommended")
            run.tags.append("start-here")
        elif run.eval_reward is not None and run.eval_reward >= 7.0:
            run.tags.append("recommended")
        elif run.eval_reward is not None and run.eval_reward < 0:
            run.tags.append("poor")

    best_dqn_eval = max(
        (r.eval_reward for r in dqn if r.eval_reward is not None),
        default=float("-inf"),
    )
    for run in dqn:
        if run.eval_reward is not None and run.eval_reward >= best_dqn_eval - 0.01:
            run.tags.append("recommended")
        if run.rel_path == "dqn/baseline":
            run.tags.append("training-demo")
        if run.eval_reward is not None and run.eval_reward < 0:
            run.tags.append("poor")


def _format_eval(run: ModelRun) -> str:
    if run.eval_reward is None:
        return "eval ?"
    success = ""
    if run.eval_success is not None:
        success = f", {run.eval_success:.0%} success"
    return f"eval {run.eval_reward:+.2f}{success}"


def _format_choice_label(run: ModelRun) -> str:
    name = run.rel_path.split("/", 1)[-1] if "/" in run.rel_path else run.rel_path
    prefix = "  "
    hints: list[str] = []

    if "start-here" in run.tags:
        prefix = "★ "
        hints.append("best starting point")
    elif "recommended" in run.tags:
        prefix = "★ "
    elif "poor" in run.tags:
        prefix = "✗ "
        hints.append("poor eval — demo only")

    if "training-demo" in run.tags and "recommended" not in run.tags:
        hints.append("good training curve, weaker eval")

    hint = f"  ({', '.join(hints)})" if hints else ""
    return f"{prefix}{name:<22} {_format_eval(run)}{hint}"


def _sort_runs_group(runs: list[ModelRun]) -> list[ModelRun]:
    def sort_key(r: ModelRun) -> tuple:
        if "start-here" in r.tags:
            tier = 0
        elif "recommended" in r.tags:
            tier = 1
        elif "poor" in r.tags:
            tier = 3
        else:
            tier = 2
        reward = r.eval_reward if r.eval_reward is not None else -999
        return (tier, -reward, r.rel_path)

    return sorted(runs, key=sort_key)


def _ordered_runs(runs: list[ModelRun]) -> list[ModelRun]:
    """Flat list matching menu order (for --run index)."""
    ql = _sort_runs_group([r for r in runs if r.algo == "q_learning"])
    dqn = _sort_runs_group([r for r in runs if r.algo == "dqn"])
    return ql + dqn


def print_menu_instructions() -> None:
    print()
    print("  Taxi-v4 Demo — model picker")
    print("  " + "─" * 44)
    print("  ↑/↓  navigate   Enter  select   Ctrl+C  quit")
    print()
    print("  What to pick:")
    print("  • Q-learning ★  — best for Taxi (fast, stable, strong eval)")
    print("  • q_learning/baseline ★  — recommended first demo")
    print("  • DQN ★ runs with eval ≥ 7.4  — best deep-learning comparison")
    print("  • dqn/baseline  — trains well but weaker eval (~3.3)")
    print("  • ✗ entries  — failed hyperparams (educational only)")
    print()


def prompt_for_run(runs: list[ModelRun]) -> ModelRun:
    if not runs:
        print(f"No trained models found under {ROOT / 'results'}")
        print("Train one first, e.g.: python scripts/train_q_learning.py")
        sys.exit(1)

    print_menu_instructions()

    ql_runs = _sort_runs_group([r for r in runs if r.algo == "q_learning"])
    dqn_runs = _sort_runs_group([r for r in runs if r.algo == "dqn"])

    choices: list = [
        Separator("── Q-learning  ·  recommended for Taxi ──"),
        *[
            Choice(_format_choice_label(r), value=r)
            for r in ql_runs
        ],
        Separator("── DQN  ·  function approximation ──"),
        *[
            Choice(_format_choice_label(r), value=r, disabled=False)
            for r in dqn_runs
        ],
    ]

    selected = questionary.select(
        "Select a trained model:",
        choices=choices,
        style=MENU_STYLE,
        use_indicator=True,
        use_shortcuts=False,
    ).ask()

    if selected is None:
        print("Cancelled.")
        sys.exit(0)

    return selected


def load_agent(run: ModelRun):
    if run.algo == "q_learning" or (run.run_dir / "q_table.npy").exists():
        with open(run.run_dir / "config.json", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        fields = QLearningConfig.__dataclass_fields__
        config = QLearningConfig(**{k: v for k, v in cfg_dict.items() if k in fields})
        agent = QLearningAgent(config)
        agent.Q = np.load(run.run_dir / "q_table.npy")
        agent.epsilon = 0.0
        return agent, "Q-learning"

    ckpt = torch.load(run.run_dir / "agent.pt", map_location="cpu", weights_only=False)
    config = DQNConfig(**ckpt["config"])
    agent = DQNAgent(config, device="cpu")
    agent.policy_net.load_state_dict(ckpt["policy_net"])
    agent.epsilon = 0.0
    return agent, "DQN"


def run_episode(
    env,
    agent,
    delay: float,
    verbose: bool,
    render_mode: str,
    seed: int | None = None,
) -> tuple[float, int, bool]:
    state, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    if verbose and render_mode == "ansi":
        frame = env.render()
        if frame:
            print(frame)

    while not (terminated or truncated):
        mask = get_action_mask(info)
        action = agent.select_action(state, mask, explore=False)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if verbose:
            print(f"  step {steps}: {ACTION_NAMES[action]} -> reward {reward:+g}")
        if delay > 0:
            time.sleep(delay)

    success = terminated and total_reward > 0
    return total_reward, steps, success


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visual Taxi-v4 demo with a saved agent")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--run", type=str, default=None, help="Run path (skip menu)")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--delay", type=float, default=0.4, help="Seconds between steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--render-mode",
        choices=["human", "ansi"],
        default="human",
        help="human = pygame window, ansi = terminal",
    )
    p.add_argument("--quiet", action="store_true", help="Hide step-by-step action log")
    return p.parse_args()


def resolve_run(runs: list[ModelRun], run_arg: str | None) -> ModelRun:
    if run_arg is None:
        return prompt_for_run(runs)

    if run_arg.isdigit():
        ordered = _ordered_runs(runs)
        idx = int(run_arg) - 1
        if 0 <= idx < len(ordered):
            return ordered[idx]
        sys.exit(f"No model with index {run_arg}")

    run_path = Path(run_arg)
    if not run_path.is_absolute():
        run_path = (ROOT / "results" / run_arg).resolve()

    for run in runs:
        if run.run_dir.resolve() == run_path or run.rel_path == run_arg:
            return run
    sys.exit(f"Model not found: {run_arg}")


def main() -> None:
    args = parse_args()
    runs = discover_runs(args.results_dir)
    run = resolve_run(runs, args.run)
    agent, label = load_agent(run)

    print(f"\nLoaded {label} from {run.rel_path}")
    if run.tags:
        print(f"Tags: {', '.join(run.tags)}")
    print(f"Running {args.episodes} episode(s) — close the render window to stop.\n")

    env = gym.make("Taxi-v4", render_mode=args.render_mode)

    try:
        for ep in range(args.episodes):
            print(f"--- Episode {ep + 1}/{args.episodes} ---")
            reward, steps, success = run_episode(
                env,
                agent,
                delay=args.delay,
                verbose=not args.quiet,
                render_mode=args.render_mode,
                seed=args.seed + ep,
            )
            status = "SUCCESS" if success else "FAILED"
            print(f"Episode {ep + 1}: {status} | reward={reward:.0f} | steps={steps}\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
