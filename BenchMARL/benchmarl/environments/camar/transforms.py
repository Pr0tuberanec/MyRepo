#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Build global state for GIRE / centralized critic from all agents' observations.

from tensordict import TensorDictBase
from torchrl.envs import Transform


class CamarGlobalStateTransform(Transform):
    """Flatten (agents, observation) into root-level ``state`` for CTDE / GIRE."""

    def __init__(self, group: str = "agents"):
        super().__init__()
        self.group = group

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if (self.group, "observation") in tensordict.keys(True):
            obs = tensordict.get((self.group, "observation"))
            phys = tensordict.get((self.group, "phys_state"))
            import torch
            state = torch.cat([obs, phys], dim=-1).reshape(*obs.shape[:-2], -1)
            tensordict.set("state", state, inplace=False)
        elif "observation" in tensordict.keys(True):
            obs = tensordict.get("observation")
            state = obs.reshape(*obs.shape[:-2], -1)
            tensordict.set("state", state, inplace=False)
        return tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def transform_observation_spec(self, observation_spec):
        obs_shape = observation_spec["agents", "observation"].shape
        n_agents, obs_dim = obs_shape[-2], obs_shape[-1]
        state_shape = (*obs_shape[:-2], n_agents * (obs_dim + 5))
        from torchrl.data import Unbounded
        observation_spec["state"] = Unbounded(
            shape=state_shape,
            device=observation_spec.device,
        )
        return observation_spec
