#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Tests for goal_dist in CAMAR global state (critic only, not actor).

import sys
from pathlib import Path

import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "CAMAR" / "src"))

try:
    import torchrl  # noqa: F401
except ImportError:
    raise SystemExit("torchrl not installed, skip test_camar_goal_state")

from torchrl.envs import TransformedEnv

from benchmarl.environments.camar.common import CamarClass
from benchmarl.environments.camar.transforms import CamarGlobalStateTransform, GOAL_STATE_DIM


def _make_task(include_goal_dist: bool = True) -> CamarClass:
    cfg = {
        "map_generator": "random_grid",
        "dynamic": "HolonomicDynamic",
        "lifelong": False,
        "window": 0.3,
        "max_steps": 10,
        "frameskip": 2,
        "max_obs": 3,
        "include_goal_dist_in_state": include_goal_dist,
        "map_kwargs": {
            "num_rows": 20,
            "num_cols": 20,
            "obstacle_density": 0.15,
            "num_agents": 8,
            "grain_factor": 3,
        },
        "dynamic_kwargs": {
            "accel": 5.0,
            "max_speed": 6.0,
            "damping": 0.25,
            "mass": 1.0,
            "dt": 0.01,
        },
    }
    return CamarClass("random_grid", cfg, device="cpu")


def _make_env(task: CamarClass, n_envs: int = 4):
    base = task.get_env_fun(n_envs, True, 42, "cpu")()
    transform = CamarGlobalStateTransform(
        include_goal_dist=task.config.get("include_goal_dist_in_state", True),
    )
    return TransformedEnv(base, transform), base


def test_actor_spec_excludes_goal_state():
    task = _make_task()
    env, _ = _make_env(task)
    obs_spec = task.observation_spec(env)
    assert "goal_state" not in obs_spec["agents"].keys()
    assert obs_spec["agents", "observation"].shape[-1] == 8


def test_global_state_shape(include_goal_dist: bool = True):
    task = _make_task(include_goal_dist=include_goal_dist)
    env, base = _make_env(task)
    td = env.reset()

    n_agents = 8
    obs_dim = base.observation_spec["agents", "observation"].shape[-1]
    extra = n_agents * GOAL_STATE_DIM if include_goal_dist else 0
    expected = n_agents * obs_dim + extra

    assert td["state"].shape[-1] == expected
    assert task.state_spec(env)["state"].shape[-1] == expected


def test_goal_state_matches_jax_env():
    task = _make_task()
    env, base = _make_env(task)
    td = env.reset()

    scale = base._env.window
    pos = base._state.physical_state.agent_pos
    goal_pos = base._state.goal_pos
    goal_dist = jnp.linalg.norm(pos - goal_pos, axis=-1)
    expected = jnp.stack(
        [goal_dist / scale, base._state.min_goal_dist / scale],
        axis=-1,
    )
    diff = (td["agents", "goal_state"].cpu().numpy() - jnp.asarray(expected)).max()
    assert diff < 1e-5


if __name__ == "__main__":
    test_actor_spec_excludes_goal_state()
    print("test_actor_spec_excludes_goal_state OK")
    for flag in (True, False):
        test_global_state_shape(flag)
        print(f"test_global_state_shape({flag}) OK")
    test_goal_state_matches_jax_env()
    print("test_goal_state_matches_jax_env OK")
    print("ALL PASSED")
