#!/usr/bin/env python3
"""Визуализация траекторий обученной BenchMARL-политики на CAMAR.

Подставь только CHECKPOINT — путь к .pt в папке checkpoints эксперимента.
Рядом должен лежать config.pkl (на уровень выше checkpoints/).

Пример:
    CHECKPOINT = "outputs/2026-06-07/12-00-00/mappo_.../checkpoints/checkpoint_1000000.pt"
    python visualize_checkpoint.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
from pathlib import Path

# --- Colab: корень репозитория (где лежат BenchMARL/ и CAMAR/) ---
ROOT = Path("/content/Nauchka")  # подставь свой путь

# --- подставь путь к чекпоинту ---
CHECKPOINT = "outputs/.../checkpoints/checkpoint_1000000.pt"

# --- опции ---
OUTPUT_SVG = "trajectory.svg"   # анимированный SVG (рекомендуется)
OUTPUT_MP4 = None                 # например "trajectory.mp4" (нужен ffmpeg)
EVAL_SEED = None                  # None = seed из эксперимента
DEVICE = "cpu"                    # "cuda" если есть GPU

# ---------------------------------------------------------------------------

def _resolve_root() -> Path:
    # Локальный запуск: .../Nauchka/BenchMARL/visualize_checkpoint.py
    try:
        candidate = Path(__file__).resolve().parent.parent
        if (candidate / "BenchMARL").is_dir():
            return candidate
    except NameError:
        pass  # Colab / Jupyter — нет __file__
    root = Path(ROOT)
    if (root / "BenchMARL").is_dir():
        return root
    cwd = Path.cwd()
    if (cwd / "BenchMARL").is_dir():
        return cwd
    if (cwd / "Nauchka" / "BenchMARL").is_dir():
        return cwd / "Nauchka"
    raise FileNotFoundError(
        f"Не найден BenchMARL. Задай ROOT = Path('...') в начале скрипта. cwd={cwd}"
    )


ROOT = _resolve_root()
sys.path.insert(0, str(ROOT / "BenchMARL"))
sys.path.insert(0, str(ROOT / "CAMAR" / "src"))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.experiment.experiment import Experiment
from camar.integrations.torchrl import CamarWrapper
from camar.render import MPLVisualizer, SVGVisualizer


def get_state_from_envs(state, env_id: int = 0, batched: bool = True):
    """Один env из батчевого JAX-state (рекурсивно по вложенным struct)."""

    def _slice(obj):
        if dataclasses.is_dataclass(obj):
            return type(obj)(**{f.name: _slice(getattr(obj, f.name)) for f in dataclasses.fields(obj)})
        # лист: jax/numpy array с batch-осью
        if batched and hasattr(obj, "shape") and len(getattr(obj, "shape", ())) > 0:
            try:
                return obj[env_id]
            except (TypeError, IndexError, KeyError):
                pass
        return obj

    return _slice(state)


def unwrap_camar(env) -> CamarWrapper:
    """Достать CamarWrapper из цепочки TransformedEnv."""
    while env is not None:
        if isinstance(env, CamarWrapper):
            return env
        env = getattr(env, "base_env", None)
    raise TypeError("CamarWrapper не найден в цепочке env")


def main():
    checkpoint = Path(CHECKPOINT)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint}")

    experiment = Experiment.reload_from_file(
        str(checkpoint),
        experiment_patch={
            "evaluation_episodes": 1,
            "sampling_device": DEVICE,
            "train_device": DEVICE,
        },
    )

    env = experiment.test_env
    camar = unwrap_camar(env)
    camar.state_seq = []

    batched = len(camar.batch_size) > 0

    def rendering_callback(rollout_env, _td):
        camar.state_seq.append(get_state_from_envs(camar._state, 0, batched=batched))

    if EVAL_SEED is not None:
        try:
            env.set_seed(EVAL_SEED)
        except NotImplementedError:
            pass

    print(f"Rollout: max_steps={experiment.max_steps}, frames={len(camar.state_seq)} (до)")
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = env.rollout(
            max_steps=experiment.max_steps,
            policy=experiment.policy,
            callback=rendering_callback,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )

    print(f"Собрано кадров: {len(camar.state_seq)}")
    if hasattr(rollout, "get"):
        on_goal = rollout.get(("next", "agents", "on_goal"))
        if on_goal is not None:
            last = on_goal[-1, 0, :, 0].float()
            print(f"on_goal на последнем шаге: {last.tolist()}")
            print(f"mean success rate: {last.mean():.3f}")

    raw_env = camar._env

    if OUTPUT_SVG:
        out_svg = Path(OUTPUT_SVG)
        if not out_svg.is_absolute():
            out_svg = checkpoint.parent.parent / out_svg
        SVGVisualizer(raw_env, camar.state_seq).save_svg(str(out_svg))
        print(f"SVG: {out_svg}")

    if OUTPUT_MP4:
        out_mp4 = Path(OUTPUT_MP4)
        if not out_mp4.is_absolute():
            out_mp4 = checkpoint.parent.parent / out_mp4
        MPLVisualizer(raw_env, camar.state_seq).animate(
            save_fname=str(out_mp4), view=False
        )
        print(f"MP4: {out_mp4}")


if __name__ == "__main__":
    main()
