#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Build global state for GIRE / centralized critic from all agents' observations.

import torch
from tensordict import TensorDictBase
from torchrl.envs import Transform

GOAL_STATE_KEY = "goal_state"
GOAL_STATE_DIM = 2


class CamarGlobalStateTransform(Transform):
    """Flatten agent observations (and optional goal distances) into root-level ``state``."""

    def __init__(
        self,
        group: str = "agents",
        include_goal_dist: bool = True,
    ):
        super().__init__()
        self.group = group
        self.include_goal_dist = include_goal_dist

    def _build_state(self, tensordict: TensorDictBase) -> torch.Tensor:
        obs = tensordict.get((self.group, "observation"))
        parts = [obs.reshape(*obs.shape[:-2], -1)]
        if self.include_goal_dist and (self.group, GOAL_STATE_KEY) in tensordict.keys(True):
            goal_state = tensordict.get((self.group, GOAL_STATE_KEY))
            parts.append(goal_state.reshape(*goal_state.shape[:-2], -1))
        return torch.cat(parts, dim=-1)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if (self.group, "observation") in tensordict.keys(True):
            tensordict.set("state", self._build_state(tensordict), inplace=False)
        elif "observation" in tensordict.keys(True):
            obs = tensordict.get("observation")
            tensordict.set("state", obs.reshape(*obs.shape[:-2], -1), inplace=False)
        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def transform_observation_spec(self, observation_spec):
        obs_shape = observation_spec["agents", "observation"].shape
        n_agents, obs_dim = obs_shape[-2], obs_shape[-1]
        extra = 0
        if self.include_goal_dist and ("agents", GOAL_STATE_KEY) in observation_spec.keys(True):
            extra = n_agents * GOAL_STATE_DIM
        state_shape = (*obs_shape[:-2], n_agents * obs_dim + extra)
        from torchrl.data import Unbounded

        observation_spec["state"] = Unbounded(
            shape=state_shape,
            device=observation_spec.device,
        )
        return observation_spec
