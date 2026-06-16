"""Import BenchMARL plotting helpers without loading the full package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_BENCHMARL_ROOT = Path(__file__).resolve().parents[2]
_EVAL_RESULTS_PATH = _BENCHMARL_ROOT / "benchmarl" / "eval_results.py"


def _load_eval_results_module() -> ModuleType:
  if str(_BENCHMARL_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCHMARL_ROOT))

  spec = importlib.util.spec_from_file_location(
    "benchmarl_eval_results", _EVAL_RESULTS_PATH
  )
  if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load plotting module from {_EVAL_RESULTS_PATH}")

  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def get_plotting() -> Any:
  return _load_eval_results_module().Plotting


def get_load_and_merge_json_dicts() -> Any:
  return _load_eval_results_module().load_and_merge_json_dicts
