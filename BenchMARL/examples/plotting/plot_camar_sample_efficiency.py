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

from matplotlib import pyplot as plt

from benchmarl.eval_results import Plotting


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


def _plot_and_save(raw: dict, env: str, out_png: Path, title: str) -> None:
    processed = Plotting.process_data(raw, metrics_to_normalize=["return"])
    _, sample_efficiency_matrix = Plotting.create_matrices(
        processed,
        env_name=env,
        metrics_to_normalize=["return"],
    )
    plot_obj, _, _ = Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix,
        metric_name="return",
        metrics_to_normalize=["return"],
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


def _aggregate_table(raw: dict, env: str, out_prefix: Path, title: str) -> None:
    processed = Plotting.process_data(raw, metrics_to_normalize=["return"])
    env_comparison_matrix, _ = Plotting.create_matrices(
        processed,
        env_name=env,
        metrics_to_normalize=["return"],
    )
    fig, scores, cis = Plotting.aggregate_scores(
        env_comparison_matrix,
        metric_name="return",
        metrics_to_normalize=["return"],
        tabular_results_file_path=str(out_prefix),
        save_tabular_as_latex=False,
    )
    plt.close(fig)
    print(f"\n{title}")
    for algo, metric_vals in scores.items():
        for metric_name, value in metric_vals.items():
            ci_val = cis.get(algo, {}).get(metric_name)
            if isinstance(value, (int, float)) and isinstance(ci_val, (tuple, list)) and len(ci_val) == 2:
                lo, hi = ci_val
                if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
                    print(f"  {algo} | {metric_name}: {value:.4f} [{lo:.4f}, {hi:.4f}]")
                    continue
            # Some marl-eval versions return already formatted strings.
            if ci_val is not None:
                print(f"  {algo} | {metric_name}: {value} {ci_val}")
            else:
                print(f"  {algo} | {metric_name}: {value}")
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
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = _collect_json_files(args.input)
    print(f"Found {len(json_files)} json files.")
    raw = _merge_jsons_as_independent_runs(json_files, args.env)
    pairs, total_runs = _count_runs(raw, args.env)
    avg_runs = total_runs / pairs if pairs else 0.0
    print(f"Task/algorithm pairs: {pairs}, total runs: {total_runs}, avg runs per pair: {avg_runs:.2f}")
    if args.single_run:
        print("Mode: single-run per configuration (R=1).")

    # One combined sample-efficiency plot (as requested).
    _plot_and_save(
        raw=raw,
        env=args.env,
        out_png=out_dir / "sample_efficiency_camar_all.png",
        title=(
            "CAMAR sample efficiency (all tasks, single-run)"
            if args.single_run
            else "CAMAR sample efficiency (all tasks)"
        ),
    )

    # Separate aggregate tables by map type.
    raw_random = _filter_tasks(raw, args.env, ("random_grid",))
    if raw_random.get(args.env):
        _aggregate_table(
            raw=raw_random,
            env=args.env,
            out_prefix=out_dir / "aggregate_scores_random_grid",
            title="Aggregate scores for random_grid",
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
        )
    else:
        print("Skip labmaze_grid table: no matching tasks found.")


if __name__ == "__main__":
    main()
