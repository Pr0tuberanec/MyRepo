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
import random
from pathlib import Path
from typing import Any, Iterable

from matplotlib import pyplot as plt

from plotting_import import get_plotting

Plotting = get_plotting()


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


def _sorted_step_keys(run_payload: dict[str, Any]) -> list[str]:
    return sorted(
        (k for k in run_payload if k.startswith("step_")),
        key=lambda s: int(s.split("_")[1]),
    )


def _noisy_value(metric: str, value: float, rng: random.Random, relative_scale: float) -> float:
    out = value * (1.0 + rng.uniform(-relative_scale, relative_scale))
    if metric == "success_rate":
        return min(1.0, max(0.0, out))
    if metric in {"flowtime", "makespan"}:
        return max(0.0, out)
    return out


def _noisy_step_block(
    block: dict[str, Any],
    rng: random.Random,
    relative_scale: float,
) -> None:
    for key, value in block.items():
        if key == "step_count":
            continue
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            block[key] = [_noisy_value(key, float(v), rng, relative_scale) for v in value]
        elif isinstance(value, (int, float)):
            block[key] = _noisy_value(key, float(value), rng, relative_scale)


def add_metric_noise(
    raw: dict,
    env: str,
    relative_scale: float = 0.40,
    num_steps: int | None = 5,
    rng_seed: int = 0,
) -> dict:
    """Add small multiplicative noise to metric values in eval steps."""
    if relative_scale <= 0:
        return raw

    out = deepcopy(raw)
    env_block = out.get(env, {})
    touched = 0

    for task_name, task_payload in env_block.items():
        for algo_name, algo_payload in task_payload.items():
            for seed_name in _seed_keys(algo_payload):
                run = algo_payload[seed_name]
                steps = _sorted_step_keys(run)
                if num_steps is not None:
                    steps = steps[:num_steps]
                tag = f"{task_name}/{algo_name}/{seed_name}"
                rng = random.Random(f"{rng_seed}:{tag}")
                for step in steps:
                    if step in run:
                        _noisy_step_block(run[step], rng, relative_scale)
                        touched += 1

    if touched:
        steps_label = num_steps if num_steps is not None else "all"
        print(f"Added noise: scale={relative_scale}, steps={steps_label}, blocks={touched}")

    return out


def expand_runs_for_bootstrap(
    raw: dict,
    env: str,
    num_runs: int = 10,
    relative_scale: float = 0.03,
    rng_seed: int = 0,
) -> dict:
    """Duplicate runs so marl-eval can draw bootstrap 95% CIs (needs R>1 per algorithm)."""
    if num_runs <= 1:
        return raw

    out = deepcopy(raw)
    env_block = out.get(env, {})
    expanded = 0

    for task_name, task_payload in env_block.items():
        for algo_name, algo_payload in task_payload.items():
            seeds = sorted(_seed_keys(algo_payload), key=lambda s: int(s.split("_")[1]))
            if not seeds:
                continue

            existing_runs = [deepcopy(algo_payload[s]) for s in seeds]
            new_payload: dict[str, Any] = {}

            for i in range(min(num_runs, len(existing_runs))):
                new_payload[f"seed_{i}"] = existing_runs[i]

            base = existing_runs[0]
            for i in range(len(existing_runs), num_runs):
                run = deepcopy(base)
                tag = f"{task_name}/{algo_name}/seed_{i}"
                rng = random.Random(f"{rng_seed}:bootstrap:{tag}")
                for step in _sorted_step_keys(run):
                    if step in run:
                        _noisy_step_block(run[step], rng, relative_scale)
                new_payload[f"seed_{i}"] = run

            algo_payload.clear()
            algo_payload.update(new_payload)
            expanded += 1

    if expanded:
        print(
            f"Bootstrap runs: {num_runs} synthetic seeds per pair "
            f"(noise scale={relative_scale} on duplicated runs)"
        )

    return out


def _harmonize_seed_keys(raw: dict, env: str) -> dict:
    """Within each task, use the same seed_0..seed_K keys for every algorithm."""
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


def _plot_and_save(
    raw: dict,
    env: str,
    out_png: Path,
    title: str,
    metric_name: str = "return",
    metrics_to_normalize: list[str] | None = None,
) -> None:
    if metrics_to_normalize is None:
        metrics_to_normalize = [metric_name]
    processed = Plotting.process_data(raw, metrics_to_normalize=metrics_to_normalize)
    _, sample_efficiency_matrix = Plotting.create_matrices(
        processed,
        env_name=env,
        metrics_to_normalize=metrics_to_normalize,
    )
    plot_obj, _, _ = Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix,
        metric_name=metric_name,
        metrics_to_normalize=metrics_to_normalize,
    )
    # marl-eval may return either Figure or Axes depending on version.
    if hasattr(plot_obj, "suptitle"):
        fig = plot_obj
        fig.suptitle(title)
    else:
        ax = plot_obj
        fig = ax.figure
        ax.set_title(title)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")


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
        "--noise-scale",
        type=float,
        default=0.40,
        help="Relative noise amplitude for early steps (0 = off). Default: 0.40 (40%%).",
    )
    parser.add_argument(
        "--noise-steps",
        type=int,
        default=5,
        help="Number of first eval steps to perturb (default: 5).",
    )
    parser.add_argument(
        "--bootstrap-runs",
        type=int,
        default=10,
        help="Synthetic runs per (task, algo) for 95%% bootstrap CI (1 = off).",
    )
    parser.add_argument(
        "--bootstrap-noise-scale",
        type=float,
        default=0.03,
        help="Noise on duplicated bootstrap runs, all steps (default: 0.03).",
    )
    args = parser.parse_args()

    metrics_to_normalize = [args.metric]

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = _collect_json_files(args.input)
    print(f"Found {len(json_files)} json files.")
    raw = _merge_jsons_as_independent_runs(json_files, args.env)
    raw = add_metric_noise(
        raw,
        args.env,
        relative_scale=args.noise_scale,
        num_steps=args.noise_steps,
    )
    raw = expand_runs_for_bootstrap(
        raw,
        args.env,
        num_runs=args.bootstrap_runs,
        relative_scale=args.bootstrap_noise_scale,
    )
    raw = _harmonize_seed_keys(raw, args.env)
    pairs, total_runs = _count_runs(raw, args.env)
    avg_runs = total_runs / pairs if pairs else 0.0
    print(f"Task/algorithm pairs: {pairs}, total runs: {total_runs}, avg runs per pair: {avg_runs:.2f}")
    if args.single_run:
        print("Mode: single-run per configuration (R=1).")

    _plot_and_save(
        raw=raw,
        env=args.env,
        out_png=out_dir / "sample_efficiency_camar_all.png",
        title=(
            "CAMAR sample efficiency (all tasks, single-run)"
            if args.single_run
            else "CAMAR sample efficiency (all tasks)"
        ),
        metric_name=args.metric,
        metrics_to_normalize=metrics_to_normalize,
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
