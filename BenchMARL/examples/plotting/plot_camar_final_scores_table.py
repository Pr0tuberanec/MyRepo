#!/usr/bin/env python3
"""CAMAR-style final scores table: IQM ± 95% CI over best-checkpoint metrics (marl-eval).

Reads marl-eval JSON files produced by BenchMARL (``create_json: true`` + CAMAR callback).
Uses ``absolute_metrics`` — by default recomputed as metrics at the eval step with the
highest mean *success_rate* (best checkpoint).

Usage:
  python examples/plotting/plot_camar_final_scores_table.py \\
    --input /path/to/run1.json /path/to/run2.json \\
    --out-dir ./tables

  python examples/plotting/plot_camar_final_scores_table.py \\
    --input /path/to/outputs \\
    --env camar \\
    --out-dir ./tables

After re-evaluating best checkpoints manually:
  python benchmarl/evaluate.py /path/to/checkpoint_X.pt
  # then pass the updated *.json files to this script
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics as rly_metrics

from plotting_import import get_load_and_merge_json_dicts, get_plotting

Plotting = get_plotting()
load_and_merge_json_dicts = get_load_and_merge_json_dicts()

CAMAR_METRICS: list[tuple[str, str, str]] = [
    ("success_rate", "mean_success_rate", "Success Rate"),
    ("flowtime", "mean_flowtime", "Flowtime"),
    ("makespan", "mean_makespan", "Makespan"),
    ("coordination", "mean_coordination", "Coordination"),
]

METRICS_TO_NORMALIZE: list[str] = []
DEFAULT_TASKS = ("random_grid", "labmaze_grid")
BOOTSTRAP_REPS = 50_000


def _seed_keys(d: dict) -> list[str]:
    return sorted(k for k in d.keys() if k.startswith("seed_"))


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


def _infer_algo_from_filepath(path: str) -> str | None:
    """Map experiment folder/json name to algorithm key (hyperlstm before lstm)."""
    name = Path(path).name.lower()
    if "hyperlstm" in name:
        return "mappo_hyperlstm"
    if "lstm" in name:
        return "mappo_lstm"
    return None


def _merge_jsons_as_independent_runs(json_files: list[str], env: str) -> dict:
    merged: dict = {env: {}}
    counters: dict[tuple[str, str], int] = {}

    for file in json_files:
        with open(file, "r") as f:
            import json

            data = json.load(f)
        env_block = data.get(env, {})
        file_algo = _infer_algo_from_filepath(file)
        for task_name, task_payload in env_block.items():
            merged[env].setdefault(task_name, {})
            for algo_name, algo_payload in task_payload.items():
                algo_name = file_algo or algo_name
                merged[env][task_name].setdefault(algo_name, {})
                key = (task_name, algo_name)
                counters.setdefault(key, 0)
                for seed_name in _seed_keys(algo_payload):
                    run_payload = deepcopy(algo_payload[seed_name])
                    merged[env][task_name][algo_name][f"seed_{counters[key]}"] = run_payload
                    counters[key] += 1

    if not merged[env]:
        raise ValueError(f"No data found for env='{env}' in provided inputs.")
    print(f"Merged runs: {sum(counters.values())} from {len(json_files)} file(s)")
    for key, n in sorted(counters.items()):
        print(f"  {key[0]} / {key[1]}: {n} run(s)")
    return merged


def _harmonize_seed_keys(raw: dict, env: str) -> dict:
    out = deepcopy(raw)
    env_block = out.get(env, {})
    for task_name, task_payload in env_block.items():
        if not task_payload:
            continue
        counts = [len(_seed_keys(a)) for a in task_payload.values()]
        if not counts or min(counts) == 0:
            continue
        min_runs = min(counts)
        if max(counts) != min_runs:
            print(f"Harmonize {task_name}: keep {min_runs} run(s) per algorithm")
        for algo_name, algo_payload in list(task_payload.items()):
            seeds = sorted(_seed_keys(algo_payload), key=lambda s: int(s.split("_")[1]))
            task_payload[algo_name] = {
                f"seed_{i}": deepcopy(algo_payload[s]) for i, s in enumerate(seeds[:min_runs])
            }
    return out


def _step_metric_mean(step_block: dict[str, Any], metric: str) -> float:
    if metric not in step_block:
        return float("-inf")
    value = step_block[metric]
    if isinstance(value, list):
        return float(np.mean(value))
    return float(value)


def _set_absolute_metrics_from_best_return(run: dict[str, Any], selection_metric: str) -> None:
    """Pick eval step with best mean return; store CAMAR metrics in absolute_metrics."""
    step_keys = [k for k in run if k.startswith("step_")]
    if not step_keys:
        return

    best_step = max(step_keys, key=lambda s: _step_metric_mean(run[s], selection_metric))
    best = run[best_step]
    absolute: dict[str, list[float]] = {}
    for metric, _, _ in CAMAR_METRICS:
        if metric in best:
            absolute[metric] = [_step_metric_mean(best, metric)]
    run["absolute_metrics"] = absolute


def _write_task_table(
    task: str,
    task_raw: dict,
    env: str,
    out_dir: Path,
    legend_map: dict[str, str] | None,
    decimals: int,
) -> None:
    task_proc = Plotting.process_data(task_raw, metrics_to_normalize=METRICS_TO_NORMALIZE)
    task_cm, _ = Plotting.create_matrices(
        task_proc, env_name=env, metrics_to_normalize=METRICS_TO_NORMALIZE
    )
    task_df, task_scores = build_camar_table(
        task_cm, legend_map=legend_map, decimals=decimals
    )
    csv_path = out_dir / f"camar_final_scores_{task}.csv"
    task_df.to_csv(csv_path)
    print(f"\n=== {task} (IQM ± 95% CI) ===\n{task_df}\nSaved: {csv_path}")

    latex_path = out_dir / f"camar_final_scores_{task}.tex"
    with open(latex_path, "w") as f:
        f.write(task_df.to_latex())
    print(f"Saved: {latex_path}")

    png_path = out_dir / f"camar_final_scores_{task}_iqm.png"
    _plot_iqm_bars(task_scores, png_path, title=f"CAMAR final scores — {task} (IQM)")
    print(f"Saved: {png_path}")


def apply_best_checkpoint_metrics(
    raw: dict, env: str, selection_metric: str = "success_rate", use_stored_absolute: bool = False
) -> dict:
    out = deepcopy(raw)
    for task_payload in out.get(env, {}).values():
        for algo_payload in task_payload.values():
            for seed_name in _seed_keys(algo_payload):
                if use_stored_absolute and "absolute_metrics" in algo_payload[seed_name]:
                    continue
                _set_absolute_metrics_from_best_return(algo_payload[seed_name], selection_metric)
    return out


def _iqm_with_ci(
    data_dictionary: dict[str, np.ndarray], reps: int = BOOTSTRAP_REPS
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    upper = {algo.upper(): arr for algo, arr in data_dictionary.items()}
    agg_func = lambda x: np.array([rly_metrics.aggregate_iqm(x)])  # noqa: E731
    scores, cis = rly.get_interval_estimates(upper, agg_func, reps=reps)
    point = {algo: float(scores[algo][0]) for algo in scores}
    interval = {algo: (float(cis[algo][0, 0]), float(cis[algo][1, 0])) for algo in cis}
    return point, interval


def _format_iqm_ci(value: float, ci: tuple[float, float], decimals: int = 3) -> str:
    return f"{value:.{decimals}f} [{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def build_camar_table(
    environment_comparison_matrix: dict[str, dict[str, np.ndarray]],
    metrics: list[tuple[str, str, str]] | None = None,
    legend_map: dict[str, str] | None = None,
    decimals: int = 3,
) -> pd.DataFrame:
    metrics = metrics or CAMAR_METRICS
    algorithms = sorted(
        {
            algo
            for _, matrix_key, _ in metrics
            if matrix_key in environment_comparison_matrix
            for algo in environment_comparison_matrix[matrix_key]
        }
    )

    rows: dict[str, dict[str, str]] = {}
    raw_scores: dict[str, dict[str, float]] = {}

    for algo in algorithms:
        label = legend_map.get(algo, legend_map.get(algo.lower(), algo)) if legend_map else algo
        rows[label] = {}
        raw_scores[label] = {}

    for _metric, matrix_key, col_name in metrics:
        if matrix_key not in environment_comparison_matrix:
            continue
        data = environment_comparison_matrix[matrix_key]
        algos_present = [a for a in algorithms if a in data]
        if not algos_present:
            continue
        subset = {a: data[a] for a in algos_present}
        iqm, ci = _iqm_with_ci(subset)
        for algo in algos_present:
            label = legend_map.get(algo, legend_map.get(algo.lower(), algo)) if legend_map else algo
            rows[label][col_name] = _format_iqm_ci(iqm[algo.upper()], ci[algo.upper()], decimals)
            raw_scores[label][col_name] = iqm[algo.upper()]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Algorithm"
    return df, raw_scores


def _plot_iqm_bars(
    raw_scores: dict[str, dict[str, float]],
    out_path: Path,
    title: str,
) -> None:
    if not raw_scores:
        return
    metrics = list(next(iter(raw_scores.values())).keys())
    algos = list(raw_scores.keys())
    x = np.arange(len(metrics))
    width = 0.8 / max(len(algos), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(metrics) * 2), 5))
    for i, algo in enumerate(algos):
        vals = [raw_scores[algo].get(m, np.nan) for m in metrics]
        ax.bar(x + i * width, vals, width, label=algo)

    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("IQM")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CAMAR final scores table via marl-eval.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="marl-eval JSON file(s) or directory(ies) with experiment outputs.",
    )
    parser.add_argument("--env", default="camar", help="Environment key in marl-eval JSON.")
    parser.add_argument("--out-dir", default="./camar_tables", help="Output folder for CSV/LaTeX/PNG.")
    parser.add_argument(
        "--use-stored-absolute",
        action="store_true",
        help="Use absolute_metrics from JSON as-is (default: recompute from best-return step).",
    )
    parser.add_argument(
        "--selection-metric",
        default="success_rate",
        help="Metric used to pick best checkpoint per run (default: success_rate).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(DEFAULT_TASKS),
        help="Tasks to tabulate (default: random_grid labmaze_grid).",
    )
    parser.add_argument(
        "--legend",
        nargs="*",
        default=[],
        metavar="ALGO=LABEL",
        help="Legend labels, e.g. mappo_lstm=MAPPO-LSTM",
    )
    parser.add_argument("--decimals", type=int, default=3)
    args = parser.parse_args()

    legend_map = dict(item.split("=", 1) for item in args.legend if "=" in item)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = _collect_json_files(args.input)
    raw = _merge_jsons_as_independent_runs(json_files, env=args.env)
    raw = _harmonize_seed_keys(raw, env=args.env)
    raw = apply_best_checkpoint_metrics(
        raw,
        env=args.env,
        selection_metric=args.selection_metric,
        use_stored_absolute=args.use_stored_absolute,
    )

    available = set(raw.get(args.env, {}).keys())
    tasks = [t for t in args.tasks if t in available]
    missing = [t for t in args.tasks if t not in available]
    for task in missing:
        print(f"Warning: task '{task}' not found in data, skipping.")

    if not tasks:
        raise ValueError(f"No requested tasks found. Available: {sorted(available)}")

    for task in tasks:
        task_raw = {args.env: {task: raw[args.env][task]}}
        _write_task_table(
            task,
            task_raw,
            env=args.env,
            out_dir=out_dir,
            legend_map=legend_map or None,
            decimals=args.decimals,
        )


if __name__ == "__main__":
    main()
