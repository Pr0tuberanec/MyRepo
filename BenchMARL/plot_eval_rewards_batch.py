#!/usr/bin/env python3
"""Batch eval: несколько чекпоинтов × несколько seeds → PNG rewards + SVG trajectory.

Запуск (DataSphere / venv):
    PYTHONNOUSERSITE=1 /path/to/venvs/benchmarl/bin/python plot_eval_rewards_batch.py

Пример (Colab):
    python plot_eval_rewards_batch.py
"""

from __future__ import annotations

import os
import pickle
import re
import sys
from pathlib import Path

# --- настройки ---
ROOT = Path("/home/jupyter/project/Nikipumba/MyRepo/")

CHECKPOINTS = [
    "/home/jupyter/project/Nikipumba/MyRepo/BenchMARL/outputs/2026-06-15/22-35-22/mappo_labmaze_grid_lstm__53bcef01_26_06_15-22_35_22/checkpoints/checkpoint_20000000.pt",
    "/home/jupyter/project/Nikipumba/MyRepo/BenchMARL/outputs/2026-06-15/22-35-22/mappo_labmaze_grid_lstm__53bcef01_26_06_15-22_35_22/checkpoints/checkpoint_10000000.pt",
]

EVAL_SEEDS = [0, 1, 2]  # 3 seed на каждый чекпоинт
OUTPUT_DIR = "eval_rollouts"  # общая папка для всех артефактов
DEVICE = "cuda"
DOWNLOAD_IN_COLAB = False  # files.download() в Colab

# Jupyter: вставьте весь файл в одну ячейку и оставьте RUN_NOW = True
RUN_NOW = False

# Пакеты (labmaze, jax, torch) лежат здесь; kernel Jupyter может быть другим python
VENV_DIR = Path("/home/jupyter/project/Nikipumba/venvs/benchmarl")

# ---------------------------------------------------------------------------


def _use_venv_packages(venv_dir: Path) -> None:
    """Подключить site-packages venv, когда ячейка идёт через /usr/local/bin/python3."""
    os.environ["PYTHONNOUSERSITE"] = "1"

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = venv_dir / "lib" / py_ver / "site-packages"
    if not site_packages.is_dir():
        raise FileNotFoundError(f"venv site-packages не найден: {site_packages}")

    site = str(site_packages)
    if site not in sys.path:
        sys.path.insert(0, site)

    # Jupyter kernel мог уже импортировать jax/torch из /usr/local — сбросить кэш
    stale = [name for name in sys.modules if name in ("jax", "jaxlib") or name.startswith(("jax.", "jaxlib."))]
    for name in stale:
        del sys.modules[name]

    cudnn_lib = site_packages / "nvidia" / "cudnn" / "lib"
    if cudnn_lib.is_dir():
        path = str(cudnn_lib)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        if path not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{path}:{existing}" if existing else path


def _configure_runtime_env() -> None:
    """JAX: не резервировать всю GPU память заранее."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _resolve_root() -> Path:
    try:
        script_dir = Path(__file__).resolve().parent
        # .../MyRepo/BenchMARL/plot_eval_rewards_batch.py
        if (script_dir / "benchmarl").is_dir():
            return script_dir.parent
        # .../MyRepo/BenchMARL/ (fallback)
        candidate = script_dir.parent
        if (candidate / "BenchMARL").is_dir():
            return candidate
    except NameError:
        pass

    root = Path(ROOT)
    if (root / "BenchMARL").is_dir():
        return root

    for candidate in (
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd() / "Nauchka",
        Path("/home/jupyter/project/Nikipumba/MyRepo"),
        Path("/content/Nauchka"),
        Path("/content/MyRepo/Nauchka"),
        Path("/content/MyRepo"),
    ):
        if (candidate / "BenchMARL").is_dir():
            return candidate

    raise FileNotFoundError(f"Не найден BenchMARL. Задай ROOT. cwd={Path.cwd()}")


def _load_task_config_from_checkpoint(ckpt: Path) -> dict:
    config_file = ckpt.parent.parent / "config.pkl"
    if not config_file.exists():
        raise FileNotFoundError(f"config.pkl не найден: {config_file}")
    with open(config_file, "rb") as f:
        pickle.load(f)  # task object
        task_config = pickle.load(f)
    return task_config


def _ensure_task_dependencies(ckpt: Path) -> None:
    """Проверить optional-зависимости до Experiment.reload_from_file."""
    task_config = _load_task_config_from_checkpoint(ckpt)
    map_generator = str(task_config.get("map_generator", ""))

    if map_generator == "labmaze_grid" or "labmaze" in map_generator:
        try:
            import labmaze  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                f"labmaze не найден. Python: {sys.executable}\n"
                f"Установите в venv: {VENV_DIR / 'bin' / 'pip'} install --no-deps labmaze==1.0.6\n"
                f"И проверьте VENV_DIR = {VENV_DIR}"
            ) from exc


_use_venv_packages(VENV_DIR)
_configure_runtime_env()


def _check_jax() -> None:
    import jax
    import jax.numpy as jnp

    print(f"jax {jax.__version__} ({jax.__file__})")
    try:
        jnp.array([1], device=jax.devices("cpu")[0])
    except TypeError as exc:
        pip = VENV_DIR / "bin" / "pip"
        raise RuntimeError(
            f"jax=={jax.__version__} не поддерживает device= (CAMAR нужен >= 0.4.31).\n"
            f"Переустановите в venv:\n"
            f"  {pip} install --force-reinstall jax==0.6.2 jaxlib==0.6.2 "
            f"jax-cuda12-plugin==0.6.2 jax-cuda12-pjrt==0.6.2"
        ) from exc


_check_jax()
ROOT = _resolve_root()
sys.path.insert(0, str(ROOT / "BenchMARL"))
sys.path.insert(0, str(ROOT / "CAMAR" / "src"))

import numpy as np

from benchmarl.experiment.experiment import Experiment
from camar.render import SVGVisualizer
from plot_eval_rewards import COMPONENTS, collect_reward_trace, plot_reward_traces


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
    print(f"Python: {sys.executable}")
    print(f"VENV:   {VENV_DIR}")
    print(f"ROOT:   {repo_root}")
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

        _ensure_task_dependencies(ckpt)

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


if __name__ == "__main__" or RUN_NOW:
    main()
