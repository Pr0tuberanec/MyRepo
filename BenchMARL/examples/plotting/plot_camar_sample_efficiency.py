#!/usr/bin/env python3
"""Build CAMAR sample-efficiency plots (marl-eval style: IQM + 95% CI).

Usage examples:
  python examples/plotting/plot_camar_sample_efficiency.py \
    --input "/path/to/BenchMARL/outputs" \
    --out-dir "/path/to/plots"

  python examples/plotting/plot_camar_sample_efficiency.py \
    --input "/path/run1.json" "/path/run2.json" \
    --out-dir "./plots"
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable

import colorcet as cc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from marl_eval.utils.data_processing_utils import get_and_aggregate_data_single_task

from plotting_import import get_plotting

Plotting = get_plotting()

DEFAULT_TASKS = ("random_grid", "labmaze_grid")
TASK_LINESTYLES = {
    "random_grid": "-",
    "labmaze_grid": "--",
}
ALGO_COLORS = {
    "mappo_lstm": sns.color_palette(cc.glasbey_category10)[0],
    "mappo_hyperlstm": sns.color_palette(cc.glasbey_category10)[1],
}


def _seed_keys(d: dict) -> list[str]:
    return sorted(k for k in d.keys() if k.startswith("seed_"))


def _merge_jsons_as_independent_runs(json_files: list[str], env: str) -> dict:
    """Merge json files while preserving every run, even with different seed names.

    marl-eval needs consistent run keys across tasks/algorithms.
    We reindex all discovered runs as seed_0..seed_K per task/algorithm.
    """
    merged: dict = {env: {}}
    counters: dict[tuple[str, str], int] = {}

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
        env_block = data.get(env, {})
        for task_name, task_payload in env_block.items():
            merged[env].setdefault(task_name, {})
            for algo_name, algo_payload in task_payload.items():
                merged[env][task_name].setdefault(algo_name, {})
                key = (task_name, algo_name)
                counters.setdefault(key, 0)
                for seed_name in _seed_keys(algo_payload):
                    run_payload = deepcopy(algo_payload[seed_name])
                    new_seed = f"seed_{counters[key]}"
                    merged[env][task_name][algo_name][new_seed] = run_payload
                    counters[key] += 1

    if not merged[env]:
        raise ValueError(f"No data found for env='{env}' in provided inputs.")
    total = sum(counters.values())
    print(f"Merged runs as independent seeds: {total}")
    return merged


def _harmonize_seed_keys(raw: dict, env: str) -> dict:
    """Within each task, use the same seed_0..seed_K keys for every algorithm.

    marl-eval assumes all algorithms on a task share identical run keys.
  """
    out = deepcopy(raw)
    env_block = out.get(env, {})
    for task_name, task_payload in env_block.items():
        if not task_payload:
            continue

        counts = [len(_seed_keys(algo_payload)) for algo_payload in task_payload.values()]
        if not counts or min(counts) == 0:
            continue

        min_runs = min(counts)
        max_runs = max(counts)
        if max_runs != min_runs:
            print(
                f"Harmonize {task_name}: use {min_runs} run(s) per algorithm "
                f"(had {min_runs}..{max_runs})"
            )

        for algo_name, algo_payload in list(task_payload.items()):
            seeds = sorted(_seed_keys(algo_payload), key=lambda s: int(s.split("_")[1]))
            kept = seeds[:min_runs]
            reindexed = {f"seed_{i}": deepcopy(algo_payload[seed]) for i, seed in enumerate(kept)}
            task_payload[algo_name] = reindexed

    return out


def _collect_json_files(inputs: Iterable[str]) -> list[str]:
    files: list[str] = []
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if p.is_file() and p.suffix == ".json":
            files.append(str(p))
        elif p.is_dir():
            for json_path in p.rglob("*.json"):
                if "wandb" in json_path.parts:
                    continue
                files.append(str(json_path))
        else:
            raise FileNotFoundError(f"Input not found: {p}")
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("No JSON files found in provided --input paths.")
    return files


def _filter_tasks(raw: dict, env: str, include_substrings: tuple[str, ...]) -> dict:
    out = deepcopy(raw)
    env_block = out.get(env, {})
    filtered = {}
    for task_name, task_payload in env_block.items():
        if any(s in task_name for s in include_substrings):
            filtered[task_name] = task_payload
    out[env] = filtered
    return out


def _discover_tasks(raw: dict, env: str, tasks: tuple[str, ...]) -> list[str]:
    env_block = raw.get(env, {})
    if tasks:
        return [task for task in tasks if task in env_block and env_block[task]]
    return sorted(env_block.keys())


def _legend_label(algo: str, task: str) -> str:
    task_label = task.replace("_", " ")
    algo_label = algo.replace("_", " ").upper()
    return f"{algo_label} · {task_label}"


def _plot_tasks_combined(
    raw: dict,
    env: str,
    out_png: Path,
    title: str,
    metric_name: str = "return",
    metrics_to_normalize: list[str] | None = None,
    tasks: tuple[str, ...] = DEFAULT_TASKS,
) -> None:
    """One figure: each (task, algorithm) pair is a separate curve with 95% CI."""
    if metrics_to_normalize is None:
        metrics_to_normalize = [metric_name]

    processed = Plotting.process_data(raw, metrics_to_normalize=metrics_to_normalize)
    available_tasks = _discover_tasks(raw, env, tasks)
    if not available_tasks:
        raise ValueError(f"No tasks from {tasks} found in env='{env}'.")

    if metric_name in metrics_to_normalize:
        ylabel = "Normalized " + " ".join(metric_name.split("_"))
    else:
        ylabel = " ".join(metric_name.split("_")).capitalize()

    fig, ax = plt.subplots(figsize=(15, 8))
    plotted = 0

    for task in available_tasks:
        task_block = raw.get(env, {}).get(task, {})
        if not task_block:
            continue

        task_mean_ci = get_and_aggregate_data_single_task(
            processed_data=processed,
            environment_name=env,
            metric_name=metric_name,
            task_name=task,
            metrics_to_normalize=metrics_to_normalize,
        )
        extra = task_mean_ci.pop("extra")
        eval_interval = extra["evaluation_interval"]
        if isinstance(eval_interval, dict):
            eval_interval = eval_interval[env]

        for algo in sorted(task_block.keys()):
            algo_key = next(
                (k for k in task_mean_ci if k.lower() == algo.lower()),
                None,
            )
            if algo_key is None:
                print(f"Skip missing curve: {task}/{algo}")
                continue

            series = task_mean_ci[algo_key]
            x = np.arange(len(series["mean"])) * eval_interval
            color = ALGO_COLORS.get(algo.lower(), sns.color_palette(cc.glasbey_category10)[plotted % 10])
            linestyle = TASK_LINESTYLES.get(task, "-")
            label = _legend_label(algo, task)

            ax.plot(
                x,
                series["mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=label,
            )
            lower = np.array(series["mean"]) - np.array(series["ci"])
            upper = np.array(series["mean"]) + np.array(series["ci"])
            ax.fill_between(x, lower, upper, color=color, alpha=0.2)
            plotted += 1

    if plotted == 0:
        raise ValueError("No (task, algorithm) curves could be plotted from the input data.")

    ax.set_xlabel("Timesteps", fontsize="xx-large")
    ax.set_ylabel(ylabel, fontsize="xx-large")
    ax.tick_params(axis="both", which="major", labelsize="xx-large")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize="large", loc="best")
    fig.suptitle(title, fontsize="xx-large", y=1.02)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png} ({plotted} curves: {', '.join(available_tasks)})")


def _aggregate_table(
    raw: dict,
    env: str,
    out_prefix: Path,
    title: str,
    metric_name: str = "return",
    metrics_to_normalize: list[str] | None = None,
) -> None:
    if metrics_to_normalize is None:
        metrics_to_normalize = [metric_name]
    processed = Plotting.process_data(raw, metrics_to_normalize=metrics_to_normalize)
    env_comparison_matrix, _ = Plotting.create_matrices(
        processed,
        env_name=env,
        metrics_to_normalize=metrics_to_normalize,
    )
    fig, scores, cis = Plotting.aggregate_scores(
        env_comparison_matrix,
        metric_name=metric_name,
        metrics_to_normalize=metrics_to_normalize,
        tabular_results_file_path=str(out_prefix),
        save_tabular_as_latex=False,
    )
    plt.close(fig)
    print(f"\n{title}")
    for algo, metric_vals in scores.items():
        for score_name, value in metric_vals.items():
            ci_val = cis.get(algo, {}).get(score_name)
            if isinstance(value, (int, float)) and isinstance(ci_val, (tuple, list)) and len(ci_val) == 2:
                lo, hi = ci_val
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    print(f"  {algo} | {score_name}: {value:.4f} [{lo:.4f}, {hi:.4f}]")
                    continue
            if ci_val is not None:
                print(f"  {algo} | {score_name}: {value} {ci_val}")
            else:
                print(f"  {algo} | {score_name}: {value}")
    print(f"Saved table files with prefix: {out_prefix}")


def _count_runs(raw: dict, env: str) -> tuple[int, int]:
    """Return (num_task_algo_pairs, total_runs_over_pairs)."""
    env_block = raw.get(env, {})
    pairs = 0
    total_runs = 0
    for task_payload in env_block.values():
        for algo_payload in task_payload.values():
            pairs += 1
            total_runs += len(_seed_keys(algo_payload))
    return pairs, total_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input JSON files and/or directories with BenchMARL JSON outputs.",
    )
    parser.add_argument("--env", default="camar", help="Environment key in marl-eval dict.")
    parser.add_argument("--out-dir", required=True, help="Directory to save output figures.")
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Use when each configuration has one run (different seed names).",
    )
    parser.add_argument(
        "--metric",
        default="return",
        help="Metric key in marl-eval JSON (e.g. return, success_rate).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        help="Tasks to plot on one figure (default: random_grid labmaze_grid).",
    )
    args = parser.parse_args()

    metrics_to_normalize = [args.metric]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = _collect_json_files(args.input)
    print(f"Found {len(json_files)} json files.")
    raw = _merge_jsons_as_independent_runs(json_files, args.env)
    raw = _harmonize_seed_keys(raw, args.env)
    pairs, total_runs = _count_runs(raw, args.env)
    avg_runs = total_runs / pairs if pairs else 0.0
    print(f"Task/algorithm pairs: {pairs}, total runs: {total_runs}, avg runs per pair: {avg_runs:.2f}")
    if args.single_run:
        print("Mode: single-run per configuration (R=1).")

    _plot_tasks_combined(
        raw=raw,
        env=args.env,
        out_png=out_dir / "sample_efficiency_camar_all.png",
        title=(
            "CAMAR sample efficiency (random_grid + labmaze_grid)"
            if len(args.tasks) > 1
            else f"CAMAR sample efficiency ({args.tasks[0]})"
        ),
        metric_name=args.metric,
        metrics_to_normalize=metrics_to_normalize,
        tasks=tuple(args.tasks),
    )

    raw_random = _filter_tasks(raw, args.env, ("random_grid",))
    if raw_random.get(args.env):
        _aggregate_table(
            raw=raw_random,
            env=args.env,
            out_prefix=out_dir / "aggregate_scores_random_grid",
            title="Aggregate scores for random_grid",
            metric_name=args.metric,
            metrics_to_normalize=metrics_to_normalize,
        )
    else:
        print("Skip random_grid table: no matching tasks found.")

    raw_labmaze = _filter_tasks(raw, args.env, ("labmaze_grid",))
    if raw_labmaze.get(args.env):
        _aggregate_table(
            raw=raw_labmaze,
            env=args.env,
            out_prefix=out_dir / "aggregate_scores_labmaze_grid",
            title="Aggregate scores for labmaze_grid",
            metric_name=args.metric,
            metrics_to_normalize=metrics_to_normalize,
        )
    else:
        print("Skip labmaze_grid table: no matching tasks found.")


if __name__ == "__main__":
    main()
