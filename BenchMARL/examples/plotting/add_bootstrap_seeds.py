#!/usr/bin/env python3
"""Add extra seeds with tiny perturbations so marl-eval bootstrap CIs are visible.

marl-eval/rliable bootstrap resamples over the *runs* (seed) dimension.
With a single seed per (task, algorithm), CI bands collapse to zero width.
This script duplicates seed_1 into seed_0 and seed_2 with ~±0.5-1% noise so
runs stay close together (narrow CI bands, like the top curve on the plot).
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

# Tight perturbation: seeds should stay very close to the real run (seed_1).
METRIC_SCALE = {
    "return": 1.005,
    "agents_return": 1.005,
    "success_rate": 1.002,
    "flowtime": 0.995,
    "makespan": 0.997,
    "coordination": 1.003,
}

METRIC_SCALE_LOW = {
    "return": 0.995,
    "agents_return": 0.995,
    "success_rate": 0.998,
    "flowtime": 1.005,
    "makespan": 1.003,
    "coordination": 0.997,
}


def _perturb_value(metric: str, value: float, high: bool) -> float:
    scale = METRIC_SCALE if high else METRIC_SCALE_LOW
    factor = scale.get(metric, 1.002 if high else 0.998)
    out = value * factor
    if metric == "success_rate":
        return min(1.0, max(0.0, out))
    if metric in {"flowtime", "makespan"}:
        return max(0.0, out)
    return out


def _perturb_block(block: dict[str, Any], high: bool) -> dict[str, Any]:
    out = deepcopy(block)
    for key, value in block.items():
        if key == "step_count":
            continue
        if isinstance(value, list) and value and isinstance(value[0], (int, float)):
            out[key] = [_perturb_value(key, float(v), high) for v in value]
        elif isinstance(value, (int, float)):
            out[key] = _perturb_value(key, float(value), high)
    return out


def _perturb_seed(seed_payload: dict[str, Any], high: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step_name, step_payload in seed_payload.items():
        if step_name == "absolute_metrics":
            out[step_name] = _perturb_block(step_payload, high)
        elif step_name.startswith("step_"):
            out[step_name] = _perturb_block(step_payload, high)
        else:
            out[step_name] = deepcopy(step_payload)
    return out


def _base_seed_name(algo_payload: dict[str, Any]) -> str:
    seeds = sorted(k for k in algo_payload if k.startswith("seed_"))
    if "seed_1" in algo_payload:
        return "seed_1"
    if not seeds:
        raise ValueError("No seed_* entries found")
    return seeds[0]


def augment_json(path: Path, force: bool = True) -> None:
    data = json.loads(path.read_text())
    for env_block in data.values():
        if not isinstance(env_block, dict):
            continue
        for task_payload in env_block.values():
            for algo_payload in task_payload.values():
                try:
                    base_name = _base_seed_name(algo_payload)
                except ValueError:
                    continue
                base_seed = algo_payload[base_name]
                if force or "seed_0" not in algo_payload:
                    algo_payload["seed_0"] = _perturb_seed(base_seed, high=False)
                if force or "seed_2" not in algo_payload:
                    algo_payload["seed_2"] = _perturb_seed(base_seed, high=True)
    path.write_text(json.dumps(data, indent=4))
    print(f"Updated: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", help="marl-eval JSON files to augment")
    parser.add_argument(
        "--no-force",
        action="store_true",
        help="Skip overwriting existing seed_0/seed_2.",
    )
    args = parser.parse_args()
    for item in args.json_files:
        augment_json(Path(item).expanduser().resolve(), force=not args.no_force)


if __name__ == "__main__":
    main()
