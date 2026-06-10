#!/usr/bin/env python3
"""Batch eval: несколько чекпоинтов × несколько seeds → PNG rewards + SVG trajectory.

Пример (Colab / локально):
    python plot_eval_rewards_batch.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# --- настройки ---
ROOT = Path("/home/jupyter/project/MyRepo/")

CHECKPOINTS = [
    "/home/jupyter/project/MyRepo/BenchMARL/outputs/.../checkpoints/checkpoint_4000000.pt",
    # ... ещё 3 пути
]

EVAL_SEEDS = [0, 1, 2]  # 3 seed на каждый чекпоинт
OUTPUT_DIR = "eval_rollouts"  # общая папка для всех артефактов
DEVICE = "cuda"
DOWNLOAD_IN_COLAB = True  # files.download() в Colab

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
    raise FileNotFoundError(f"Не найден BenchMARL. Задай ROOT. cwd={Path.cwd()}")


ROOT = _resolve_root()
sys.path.insert(0, str(ROOT / "BenchMARL"))
sys.path.insert(0, str(ROOT / "CAMAR" / "src"))

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.experiment.experiment import Experiment
from camar.render import SVGVisualizer
from plot_eval_rewards import COMPONENTS, collect_reward_trace, plot_reward_traces
from visualize_checkpoint import unwrap_camar


def checkpoint_label(ckpt: Path) -> str:
    """Короткое имя прогона: experiment_folder + checkpoint_N."""
    exp_name = ckpt.parent.parent.name
    safe = re.sub(r"[^\w\-.]+", "_", exp_name)
    return f"{safe}_{ckpt.stem}"


def output_paths(out_dir: Path, label: str, seed: int) -> tuple[Path, Path, Path]:
    base = f"{label}_seed{seed}"
    return (
        out_dir / f"{base}_rewards.png",
        out_dir / f"{base}_trajectory.svg",
        out_dir / f"{base}_rewards.npz",
    )


def run_one_seed(
    experiment: Experiment,
    seed: int,
    out_dir: Path,
    label: str,
) -> dict[str, np.ndarray]:
    png_path, svg_path, npz_path = output_paths(out_dir, label, seed)

    traces, rollout, state_seq, raw_env = collect_reward_trace(
        experiment.test_env,
        experiment.policy,
        max_steps=experiment.max_steps,
        seed=seed,
        device=str(experiment.config.sampling_device),
    )

    plot_reward_traces(
        traces,
        png_path,
        title=f"{label} | seed={seed} | {traces['total'].shape[0]} steps",
    )
    SVGVisualizer(raw_env, state_seq).save_svg(str(svg_path))
    np.savez_compressed(npz_path, **{k: traces[k] for k in COMPONENTS})

    if hasattr(rollout, "get"):
        on_goal = rollout.get(("next", "agents", "on_goal"))
        if on_goal is not None:
            sr = on_goal[-1, 0, :, 0].float().mean().item()
            print(f"  SR={sr:.3f}")

    print(f"  PNG:  {png_path}")
    print(f"  SVG:  {svg_path}")
    print(f"  NPZ:  {npz_path}")
    return traces


def run_batch(
    checkpoints: list[str | Path],
    seeds: list[int],
    *,
    root: Path | None = None,
    output_dir: str | Path = "eval_rollouts",
    device: str = "cpu",
    download_in_colab: bool = False,
) -> list[Path]:
    repo_root = root or _resolve_root()
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / "BenchMARL" / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    print(f"Output: {out_dir}")
    print(f"Checkpoints: {len(checkpoints)}, seeds: {seeds}\n")

    for ckpt_i, checkpoint in enumerate(checkpoints, 1):
        ckpt = Path(checkpoint)
        if not ckpt.is_absolute():
            ckpt = repo_root / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"Чекпоинт не найден: {ckpt}")

        label = checkpoint_label(ckpt)
        print(f"[{ckpt_i}/{len(checkpoints)}] {label}")

        experiment = Experiment.reload_from_file(
            str(ckpt),
            experiment_patch={
                "evaluation_episodes": 1,
                "sampling_device": device,
                "train_device": device,
            },
        )

        for seed in seeds:
            print(f"  seed={seed}")
            run_one_seed(experiment, seed, out_dir, label)
            png_path, svg_path, npz_path = output_paths(out_dir, label, seed)
            saved.extend([png_path, svg_path, npz_path])
        print()

    if download_in_colab:
        try:
            from google.colab import files

            for path in saved:
                if path.exists():
                    files.download(str(path))
        except ImportError:
            print("google.colab недоступен — пропуск download")

    print(f"Готово: {len(saved)} файлов в {out_dir}")
    return saved


def main():
    run_batch(
        CHECKPOINTS,
        EVAL_SEEDS,
        root=ROOT,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        download_in_colab=DOWNLOAD_IN_COLAB,
    )


if __name__ == "__main__":
    main()
