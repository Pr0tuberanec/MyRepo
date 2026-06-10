import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs.common import _EnvWrapper
from torchrl.envs.libs.jax_utils import (
    _extract_spec,
    _ndarray_to_tensor,
    _object_to_tensordict,
    _tensor_to_ndarray,
    _tree_flatten,
)
from torchrl.envs.utils import MarlGroupMapType, _classproperty, check_marl_grouping
from camar import camar_v0


class CamarWrapper(_EnvWrapper):
    _jax = None

    @_classproperty
    def jax(cls):
        if cls._jax is not None:
            return cls._jax

        import jax

        cls._jax = jax
        return jax

    def __init__(
        self,
        seed,
        device,
        batch_size,
        env=None,
        collision_penalty_start: float = 0.0,
        collision_penalty_end: float = -0.1,
        collision_penalty_curriculum_frames=None,
        collision_penalty_curriculum: bool = True,
        **kwargs,
    ):
        if env is not None:
            kwargs["env"] = env

        if isinstance(batch_size, int):
            batch_size = [batch_size]

        kwargs["device"] = device
        kwargs["batch_size"] = batch_size
        kwargs["seed"] = seed

        self.collision_penalty_start = collision_penalty_start
        self.collision_penalty_end = collision_penalty_end
        self.collision_penalty_curriculum_frames = (
            1_000_000
            if collision_penalty_curriculum_frames is None
            else collision_penalty_curriculum_frames
        )
        self.collision_penalty_curriculum = collision_penalty_curriculum
        self._curriculum_frames = 0

        super().__init__(**kwargs)

        self.set_seed(seed)

    def _check_kwargs(self, kwargs: dict):
        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")

    def _build_env(
        self,
        env,
        **kwargs,
    ):
        return env

    def _make_state_spec(self, env):
        jax = self.jax

        key = jax.random.PRNGKey(0)
        state = env.reset(key)
        state_dict = _object_to_tensordict(state, self.device, batch_size=())
        state_spec = _extract_spec(state_dict).expand(self.batch_size)
        return state_spec

    def _make_specs(self, env) -> None:  # noqa: F821
        agent_names = [f"agent_{agent_idx}" for agent_idx in range(env.num_agents)]
        self.group_map = MarlGroupMapType.ALL_IN_ONE_GROUP.get_group_map(agent_names)

        # just to be confident
        check_marl_grouping(self.group_map, agent_names)
        assert len(self.group_map.keys()) == 1
        assert "agents" in self.group_map.keys()

        action = Bounded(
            low=-1,
            high=1,
            shape=(*self.batch_size, env.num_agents, env.action_size),
            device=self.device,
        )
        agents_action = Composite(
            action=action,
            shape=(*self.batch_size, env.num_agents),
            device=self.device,
        )
        self.action_spec = Composite(
            agents=agents_action,
            shape=self.batch_size,
            device=self.device,
        )

        reward = Bounded(
            low=-0.5,
            high=1,
            shape=(*self.batch_size, env.num_agents, 1),
            device=self.device,
        )
        agents_reward = Composite(
            reward=reward,
            shape=(*self.batch_size, env.num_agents),
            device=self.device,
        )
        self.reward_spec = Composite(
            agents=agents_reward,
            shape=self.batch_size,
            device=self.device,
        )

        observation = Unbounded(
            shape=(*self.batch_size, env.num_agents, env.observation_size),
            device=self.device,
        )
        agents_observation = Composite(
            observation=observation,
            shape=(*self.batch_size, env.num_agents),
            device=self.device,
        )
        info_spec = Composite(
            on_goal_rate=Unbounded(shape=(*self.batch_size,), device=self.device),
            on_goal_count=Unbounded(shape=(*self.batch_size,), device=self.device),
            collision_count_step=Unbounded(
                shape=(*self.batch_size,), device=self.device
            ),
            reward_goal_progress_mean=Unbounded(
                shape=(*self.batch_size,), device=self.device
            ),
            reward_goal_bonus_count=Unbounded(
                shape=(*self.batch_size,), device=self.device
            ),
            reward_team_bonus_count=Unbounded(
                shape=(*self.batch_size,), device=self.device
            ),
            reward_collision_penalty_mean=Unbounded(
                shape=(*self.batch_size,), device=self.device
            ),
            reward_total_mean=Unbounded(shape=(*self.batch_size,), device=self.device),
            shape=self.batch_size,
            device=self.device,
        )
        # agents_observation["phys_state"] = Unbounded(
        #     shape=(*self.batch_size, env.num_agents, 5),
        #     device=self.device,
        # )
        agents_observation["goal_state"] = Unbounded(
            shape=(*self.batch_size, env.num_agents, 2),
            device=self.device,
        )
        self.observation_spec = Composite(
            agents=agents_observation,
            info=info_spec,
            shape=self.batch_size,
            device=self.device,
        )

        self.done_spec = Categorical(
            n=2,
            shape=(*self.batch_size, 1),
            dtype=torch.bool,
            device=self.device,
        )

    # def _make_state_example(self):
    #     jax = self.jax

    #     key = jax.random.PRNGKey(0)
    #     keys = jax.random.split(key, self.batch_size.numel())
    #     state, obs, done = self._jit_vmap_env_reset(jax.numpy.stack(keys))
    #     # state = _tree_reshape(state, self.batch_size)
    #     return state

    def _collision_penalty_factor(self) -> float:
        if not self.collision_penalty_curriculum:
            return self.collision_penalty_end
        if (
            self.collision_penalty_curriculum_frames is None
            or self.collision_penalty_curriculum_frames <= 0
        ):
            return self.collision_penalty_end
        progress = min(
            1.0,
            self._curriculum_frames / self.collision_penalty_curriculum_frames,
        )
        return self.collision_penalty_start + progress * (
            self.collision_penalty_end - self.collision_penalty_start
        )

    def _init_env(self):
        jax = self.jax
        self._key = None
        self._jit_vmap_env_reset = jax.jit(jax.vmap(self._env.reset))
        self._jit_vmap_env_step = jax.jit(
            jax.vmap(self._env.step, in_axes=(0, 0, 0, None))
        )
        self._jit_vmap_env_get_obs = jax.jit(jax.vmap(self._env.get_obs))

        self._jit_partial_reset = jax.jit(self._partial_reset)
        self._state = None
        # self._state_example = self._make_state_example()

    def _set_seed(self, seed: int):
        jax = self.jax
        if seed is None:
            raise Exception("CAMAR requires an integer seed.")
        self._key = jax.random.PRNGKey(seed)

    # def _phys_state_tensor(self, state) -> torch.Tensor:
    #     """(batch, N, 5): per-agent [pos(2), vel(2), agent_rad(1)]."""
    #     jax = self.jax
    #     N = self._env.num_agents
    #     pos = state.physical_state.agent_pos   # (batch, N, 2)
    #     vel = state.physical_state.agent_vel   # (batch, N, 2)
    #     if self._env.homogeneous_agents:
    #         rad = jax.numpy.full((*self.batch_size, N, 1), self._env.map_generator.agent_rad)
    #     else:
    #         rad = state.sizes.agent_rad[..., jax.numpy.newaxis]  # (batch, N, 1)
    #     return _ndarray_to_tensor(jax.numpy.concatenate([pos, vel, rad], axis=-1))

    def _goal_state_tensor(self, state) -> torch.Tensor:
        """(batch, N, 2): per-agent [goal_dist, min_goal_dist] / window."""
        jax = self.jax
        pos = state.physical_state.agent_pos
        goal_dist = jax.numpy.linalg.norm(pos - state.goal_pos, axis=-1)
        scale = jax.numpy.float32(self._env.window)
        goal_state = jax.numpy.stack(
            [goal_dist / scale, state.min_goal_dist / scale],
            axis=-1,
        )
        return _ndarray_to_tensor(goal_state)

    def _partial_reset(self, keys, state, envs_to_reset):
        obs_r, state_r = self._jit_vmap_env_reset(keys)

        obs_old = self._jit_vmap_env_get_obs(state)

        state_old = state

        state = self.jax.tree.map(
            lambda x, y: self.jax.numpy.where(
                self.jax.numpy.expand_dims(envs_to_reset, range(1, x.ndim)), x, y
            ),
            state_r,
            state_old,
        )

        obs = self.jax.numpy.where(envs_to_reset[:, None, None], obs_r, obs_old)

        return obs, state

    def _reset(self, tensordict: TensorDictBase = None, **kwargs) -> TensorDictBase:
        jax = self.jax

        # generate random keys
        self._key, *keys = jax.random.split(self._key, 1 + self.numel())
        keys = jax.numpy.stack(keys)

        if tensordict is not None and "_reset" in tensordict.keys():
            _reset = tensordict.get("_reset")
            envs_to_reset = _reset.squeeze(-1)

            if envs_to_reset.all():
                # reset all
                obs, self._state = self._jit_vmap_env_reset(keys)
            else:
                envs_to_reset = _tensor_to_ndarray(envs_to_reset)

                obs, self._state = self._jit_partial_reset(keys, self._state, envs_to_reset)
        else:
            # call env reset with jit and vmap
            obs, self._state = self._jit_vmap_env_reset(keys)

        tensordict_agents = TensorDict(
            source={
                "observation": _ndarray_to_tensor(obs),
                "goal_state": self._goal_state_tensor(self._state),
                # "phys_state": self._phys_state_tensor(self._state),
            },
            batch_size=(*self.batch_size, self._env.num_agents),
            device=self.device,
        )

        done = self._state.on_goal.all(axis=-1)
        done = _ndarray_to_tensor(done)
        zero_info = TensorDict(
            source={
                "on_goal_rate": torch.zeros(*self.batch_size, device=self.device),
                "on_goal_count": torch.zeros(*self.batch_size, device=self.device),
                "collision_count_step": torch.zeros(
                    *self.batch_size, device=self.device
                ),
                "reward_goal_progress_mean": torch.zeros(
                    *self.batch_size, device=self.device
                ),
                "reward_goal_bonus_count": torch.zeros(
                    *self.batch_size, device=self.device
                ),
                "reward_team_bonus_count": torch.zeros(
                    *self.batch_size, device=self.device
                ),
                "reward_collision_penalty_mean": torch.zeros(
                    *self.batch_size, device=self.device
                ),
                "reward_total_mean": torch.zeros(*self.batch_size, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

        tensordict_out = TensorDict(
            source={
                "agents": tensordict_agents,
                "done": done,
                "terminated": done.clone(),
                "info": zero_info,
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out

    def _step(self, tensordict: TensorDictBase):
        jax = self.jax

        # convert tensors to ndarrays

        # state = _tensordict_to_object(tensordict.get("state"), self._state_example)
        action = _tensor_to_ndarray(tensordict.get(("agents", "action")))

        # flatten batch size
        # state = _tree_flatten(state, self.batch_size)
        action = _tree_flatten(action, self.batch_size)

        # call env step with jit and vmap
        self._key, *keys_s = jax.random.split(self._key, 1 + self.numel())

        collision_penalty_factor = self._collision_penalty_factor()
        obs, self._state, reward, done, info = self._jit_vmap_env_step(
            jax.numpy.stack(keys_s),
            self._state,
            action,
            collision_penalty_factor,
        )
        if self.collision_penalty_curriculum:
            self._curriculum_frames += self.numel()

        tensordict_agents = TensorDict(
            source={
                "observation": _ndarray_to_tensor(obs),
                "goal_state": self._goal_state_tensor(self._state),
                "reward": _ndarray_to_tensor(reward),
                "on_goal": _ndarray_to_tensor(self._state.on_goal).view(*self.batch_size, -1, 1),
                # "phys_state": self._phys_state_tensor(self._state),
            },
            batch_size=(*self.batch_size, self._env.num_agents),
            device=self.device,
        )

        time_to_reach_goal = _ndarray_to_tensor(self._state.time_to_reach_goal)
        coordination = 1 - _ndarray_to_tensor(self._state.num_collisions / self._state.step[:, None]).mean(
            -1
        )  # mean for agents

        flowtime = time_to_reach_goal.sum(dim=-1)
        makespan, _ = time_to_reach_goal.max(dim=-1)

        done = _ndarray_to_tensor(done)

        tensordict_out = TensorDict(
            source={
                "agents": tensordict_agents,
                "done": done,
                "terminated": done.clone(),
                "flowtime": flowtime,
                "makespan": makespan,
                "coordination": coordination,
                "info": TensorDict(
                    {
                        **{
                            key: _ndarray_to_tensor(value)
                            for key, value in info.items()
                        },
                        "collision_penalty_factor": torch.full(
                            self.batch_size,
                            collision_penalty_factor,
                            device=self.device,
                        ),
                    },
                    batch_size=self.batch_size,
                    device=self.device,
                ),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        return tensordict_out


class CamarEnv(CamarWrapper):
    def __init__(
        self,
        num_envs,
        seed,
        device,
        map_generator,
        dynamic,
        lifelong,
        window,
        max_steps,
        frameskip,
        max_obs,
        pos_shaping_factor,
        contact_force,
        contact_margin,
        map_kwargs,
        dynamic_kwargs,
        goal_rad_eval_factor=2.5,
    ):
        batch_size = [num_envs]

        super().__init__(
            batch_size=batch_size,
            num_envs=num_envs,
            seed=seed,
            device=device,
            map_generator=map_generator,
            dynamic=dynamic,
            lifelong=lifelong,
            window=window,
            max_steps=max_steps,
            frameskip=frameskip,
            max_obs=max_obs,
            pos_shaping_factor=pos_shaping_factor,
            goal_rad_eval_factor=goal_rad_eval_factor,
            contact_force=contact_force,
            contact_margin=contact_margin,
            map_kwargs=map_kwargs,
            dynamic_kwargs=dynamic_kwargs,
        )

    def _check_kwargs(self, kwargs: dict):
        if "map_generator" not in kwargs:
            raise TypeError("Cannot find the environment key 'map_generator' in kwargs.")
        if "num_envs" not in kwargs:
            raise TypeError("Cannot find the environment key 'num_envs' in kwargs.")

    def _build_env(
        self,
        num_envs,
        seed,
        map_generator,
        dynamic,
        lifelong,
        window,
        max_steps,
        frameskip,
        max_obs,
        pos_shaping_factor,
        contact_force,
        contact_margin,
        map_kwargs,
        dynamic_kwargs,
        goal_rad_eval_factor=2.5,
    ):
        self.map_generator = map_generator

        env = camar_v0(
            map_generator=map_generator,
            dynamic=dynamic,
            lifelong=lifelong,
            window=window,
            max_steps=max_steps,
            frameskip=frameskip,
            max_obs=max_obs,
            pos_shaping_factor=pos_shaping_factor,
            goal_rad_eval_factor=goal_rad_eval_factor,
            contact_force=contact_force,
            contact_margin=contact_margin,
            map_kwargs=map_kwargs,
            dynamic_kwargs=dynamic_kwargs,
        )

        return super()._build_env(env)
