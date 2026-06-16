#!/usr/bin/env python3
"""Evaluate best checkpoints, then build CAMAR final-scores table (marl-eval).

Scans experiment folders, picks the checkpoint whose logged eval *return* was
highest success_rate (from existing marl-eval JSON), runs ``benchmarl/evaluate.py``, and
calls ``plot_camar_final_scores_table.py``.

Usage:
  PYTHONPATH=BenchMARL python examples/plotting/evaluate_best_checkpoints.py \\
    --experiments /path/to/outputs \\
    --out-dir ./camar_eval_tables

  # Table only (JSON already on disk):
  python examples/plotting/evaluate_best_checkpoints.py \\
    --experiments /path/to/outputs --skip-eval --out-dir ./tables
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_BENCHMARL_ROOT = Path(__file__).resolve().parents[2]


def _find_experiment_dirs(paths: list[str]) -> list[Path]:
    dirs: list[Path] = []
    for item in paths:
        p = Path(item).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        if list(p.glob("checkpoints/*.pt")):
            dirs.append(p)
            continue
        for ckpt_dir in sorted(p.rglob("checkpoints")):
            if ckpt_dir.is_dir() and list(ckpt_dir.glob("*.pt")):
                dirs.append(ckpt_dir.parent)
    return sorted(set(dirs))


def _checkpoint_step(path: Path) -> int:
    m = re.search(r"checkpoint_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _best_checkpoint_by_metric(exp_dir: Path, metric: str = "return") -> Path | None:
    json_files = [p for p in exp_dir.glob("*.json")]
    best_step = -1
    best_value = float("-inf")

    if json_files:
        with open(json_files[0]) as f:
            data = json.load(f)
        for env_block in data.values():
            for task_block in env_block.values():
                for algo_block in task_block.values():
                    for run in algo_block.values():
                        for key, step_data in run.items():
                            if not key.startswith("step_"):
                                continue
                            if metric not in step_data:
                                continue
                            vals = step_data[metric]
                            mean_val = (
                                sum(vals) / len(vals)
                                if isinstance(vals, list)
                                else float(vals)
                            )
                            step_num = int(key.split("_")[1])
                            if mean_val > best_value:
                                best_value = mean_val
                                best_step = step_num

    if best_step >= 0:
        matches = list(exp_dir.glob(f"checkpoints/checkpoint_{best_step}.pt"))
        if matches:
            return matches[0]

    ckpts = sorted(exp_dir.glob("checkpoints/*.pt"), key=_checkpoint_step)
    return ckpts[-1] if ckpts else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval best checkpoints → CAMAR table.")
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--out-dir", default="./camar_eval_tables")
    parser.add_argument("--env", default="camar")
    parser.add_argument("--selection-metric", default="success_rate")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["random_grid", "labmaze_grid"],
    )
    parser.add_argument("--legend", nargs="*", default=[], metavar="ALGO=LABEL")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluate_py = _BENCHMARL_ROOT / "benchmarl" / "evaluate.py"
    table_script = Path(__file__).with_name("plot_camar_final_scores_table.py")

    exp_dirs = _find_experiment_dirs(args.experiments)
    if not exp_dirs:
        raise FileNotFoundError("No experiment folders with checkpoints found.")

    json_files: list[str] = []
    for exp_dir in exp_dirs:
        if args.skip_eval:
            json_files.extend(str(p) for p in exp_dir.glob("*.json"))
            continue

        ckpt = _best_checkpoint_by_metric(exp_dir, metric=args.selection_metric)
        if ckpt is None:
            print(f"Skip (no checkpoint): {exp_dir}")
            continue
        print(f"Evaluating best checkpoint: {ckpt}")
        subprocess.run(
            [sys.executable, str(evaluate_py), str(ckpt)],
            cwd=str(_BENCHMARL_ROOT),
            check=True,
        )
        found = list(exp_dir.glob("*.json"))
        if not found:
            raise FileNotFoundError(f"No marl-eval JSON after eval in {exp_dir}")
        json_files.append(str(found[0]))

    if not json_files:
        raise RuntimeError("No JSON files to aggregate.")

    cmd = [
        sys.executable,
        str(table_script),
        "--input",
        *json_files,
        "--env",
        args.env,
        "--out-dir",
        str(out_dir),
        "--selection-metric",
        args.selection_metric,
    ]
    if args.tasks:
        cmd.extend(["--tasks", *args.tasks])
    cmd.extend(args.legend)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
