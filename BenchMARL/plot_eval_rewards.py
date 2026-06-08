#!/usr/bin/env python3
"""Графики наград каждого агента на eval rollout (CAMAR + BenchMARL).

Пример:
    CHECKPOINT = "outputs/.../checkpoints/checkpoint_1000000.pt"
    python plot_eval_rewards.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Colab: корень репозитория (где лежат BenchMARL/ и CAMAR/) ---
ROOT = Path("/content/Nauchka")

CHECKPOINT = "outputs/.../checkpoints/checkpoint_1000000.pt"
OUTPUT_PNG = "eval_rewards.png"
OUTPUT_SVG = "trajectory.svg"  # анимация эпизода (тот же rollout)
EVAL_SEED = 0  # фиксированный seed для одного эпизода
DEVICE = "cpu"  # "cuda" если есть GPU
SHOW_INLINE = True  # показать PNG в Colab

# ---------------------------------------------------------------------------

def _resolve_root() -> Path:
    try:
        candidate = Path(__file__).resolve().parent.parent
        if (candidate / "BenchMARL").is_dir():
            return candidate
    except NameError:
        pass
    root = Path(ROOT)
    if (root / "BenchMARL").is_dir():
        return root
    for candidate in (
        Path.cwd(),
        Path.cwd() / "Nauchka",
        Path("/content/Nauchka"),
        Path("/content/MyRepo/Nauchka"),
        Path("/content/MyRepo"),
    ):
        if (candidate / "BenchMARL").is_dir():
            return candidate
    raise FileNotFoundError(f"Не найден BenchMARL. Задай ROOT в начале скрипта. cwd={Path.cwd()}")


ROOT = _resolve_root()
sys.path.insert(0, str(ROOT / "BenchMARL"))
sys.path.insert(0, str(ROOT / "CAMAR" / "src"))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.experiment.experiment import Experiment
from camar.render import SVGVisualizer
from visualize_checkpoint import get_state_from_envs, unwrap_camar

COMPONENTS = (
    "goal_progress",
    "goal_bonus",
    "team_bonus",
    "goal_retreat_penalty",
    "collision_penalty",
    "total",
)
AGENT_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def collect_reward_trace(env, policy, max_steps: int, seed: int | None, device: str):
    camar = unwrap_camar(env)
    raw_env = camar._env
    batched = len(camar.batch_size) > 0

    if seed is not None:
        try:
            env.set_seed(seed)
        except NotImplementedError:
            pass

    states_after_step: list = []
    state_seq: list = []
    actions_per_step: list = []

    def callback(rollout_env, td):
        st = get_state_from_envs(unwrap_camar(rollout_env)._state, 0, batched=batched)
        states_after_step.append(st)
        state_seq.append(st)
        action = td.get(("agents", "action"))
        if action is not None:
            act = action[0].detach().cpu().numpy()
            if act.ndim == 3:
                act = act[0]
            actions_per_step.append(act)

    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = env.rollout(
            max_steps=max_steps,
            policy=policy,
            callback=callback,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )

    if not states_after_step:
        raise RuntimeError("Rollout не дал ни одного шага.")
    if len(actions_per_step) != len(states_after_step):
        raise RuntimeError(
            f"Несовпадение длины траектории: actions={len(actions_per_step)}, "
            f"states={len(states_after_step)}"
        )

    if seed is not None:
        try:
            env.set_seed(seed)
        except NotImplementedError:
            pass
    env.reset()
    reset_state = get_state_from_envs(camar._state, 0, batched=batched)

    # Пересчёт компонент reward по сохранённой траектории
    traces = {name: [] for name in COMPONENTS}
    prev_state = reset_state
    for t, new_state in enumerate(states_after_step):
        action = actions_per_step[t]
        comps = raw_env.get_reward_components(prev_state, action, new_state)
        for name in COMPONENTS:
            traces[name].append(np.asarray(comps[name], dtype=np.float32))
        prev_state = new_state

    for name in COMPONENTS:
        traces[name] = np.stack(traces[name], axis=0)  # (T, n_agents)

    return traces, rollout, state_seq, raw_env


def plot_reward_traces(traces: dict[str, np.ndarray], out_path: Path, title: str = ""):
    """Один subplot на компонент reward, на каждом — 8 линий (агенты)."""
    t_steps, n_agents = traces["total"].shape
    x = np.arange(t_steps)

    fig, axes = plt.subplots(
        len(COMPONENTS),
        1,
        figsize=(12, 2.4 * len(COMPONENTS)),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(title or "Eval rewards by component", fontsize=14)

    for comp_idx, name in enumerate(COMPONENTS):
        ax = axes[comp_idx, 0]
        for agent_idx in range(n_agents):
            ax.plot(
                x,
                traces[name][:, agent_idx],
                color=AGENT_COLORS[agent_idx % len(AGENT_COLORS)],
                linewidth=1.2,
                alpha=0.9,
                label=f"agent {agent_idx}",
            )
        ax.axhline(0.0, color="gray", linewidth=0.6)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
        if comp_idx == 0:
            ax.legend(ncol=4, fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eval_reward_plot(
    checkpoint: str | Path,
    *,
    root: str | Path | None = None,
    seed: int | None = None,
    output_png: str | Path = "eval_rewards.png",
    output_svg: str | Path | None = "trajectory.svg",
    device: str = "cpu",
    show_inline: bool = False,
) -> dict[str, np.ndarray]:
    """Один eval-эпизод (deterministic) и график наград по агентам."""
    repo_root = Path(root) if root is not None else _resolve_root()
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = repo_root / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt}")

    experiment = Experiment.reload_from_file(
        str(ckpt),
        experiment_patch={
            "evaluation_episodes": 1,
            "sampling_device": device,
            "train_device": device,
        },
    )

    episode_seed = seed if seed is not None else experiment.seed
    traces, rollout, state_seq, raw_env = collect_reward_trace(
        experiment.test_env,
        experiment.policy,
        max_steps=experiment.max_steps,
        seed=episode_seed,
        device=device,
    )

    out_dir = ckpt.parent.parent
    out = Path(output_png)
    if not out.is_absolute():
        out = out_dir / out

    plot_reward_traces(
        traces,
        out,
        title=f"Eval rewards, seed={episode_seed} ({traces['total'].shape[0]} steps)",
    )
    print(f"PNG: {out}")

    if output_svg:
        out_svg = Path(output_svg)
        if not out_svg.is_absolute():
            out_svg = out_dir / out_svg
        SVGVisualizer(raw_env, state_seq).save_svg(str(out_svg))
        print(f"SVG: {out_svg}")

    if hasattr(rollout, "get"):
        on_goal = rollout.get(("next", "agents", "on_goal"))
        if on_goal is not None:
            last = on_goal[-1, 0, :, 0].float()
            print(f"on_goal (последний шаг): {last.tolist()}")
            print(f"mean success rate: {last.mean():.3f}")

    print("Сумма total за эпизод по агентам:", traces["total"].sum(axis=0).tolist())

    if show_inline:
        try:
            from IPython.display import Image, display

            display(Image(filename=str(out)))
        except ImportError:
            pass

    return traces


def main():
    run_eval_reward_plot(
        CHECKPOINT,
        root=ROOT,
        seed=EVAL_SEED,
        output_png=OUTPUT_PNG,
        output_svg=OUTPUT_SVG,
        device=DEVICE,
        show_inline=SHOW_INLINE,
    )


if __name__ == "__main__":
    main()
