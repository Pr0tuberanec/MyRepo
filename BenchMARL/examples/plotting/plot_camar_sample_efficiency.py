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
from pathlib import Path
from typing import Iterable

from matplotlib import pyplot as plt

from benchmarl.eval_results import Plotting, load_and_merge_json_dicts


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
    fig, _, _ = Plotting.environemnt_sample_efficiency_curves(
        sample_effeciency_matrix=sample_efficiency_matrix,
        metric_name="return",
        metrics_to_normalize=["return"],
    )
    fig.suptitle(title)
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
            lo, hi = cis[algo][metric_name]
            print(f"  {algo} | {metric_name}: {value:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"Saved table files with prefix: {out_prefix}")


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
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = _collect_json_files(args.input)
    print(f"Found {len(json_files)} json files.")
    raw = load_and_merge_json_dicts(json_files)

    # One combined sample-efficiency plot (as requested).
    _plot_and_save(
        raw=raw,
        env=args.env,
        out_png=out_dir / "sample_efficiency_camar_all.png",
        title="CAMAR sample efficiency (all tasks)",
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
