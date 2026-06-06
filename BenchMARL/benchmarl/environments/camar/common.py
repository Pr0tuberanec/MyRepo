from typing import Callable, Dict, List, Optional
from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING
from tensordict import TensorDictBase
from torchrl.data import Composite, Unbounded, UnboundedContinuous
from torchrl.envs import EnvBase, Transform, TransformedEnv
from camar.integrations.torchrl import CamarWrapper
from camar import camar_v0

class CamarTask(Task):
    RANDOM_GRID = None
    RANDOM_GRID_2 = None

    @staticmethod
    def associated_class():
        return CamarClass

class CamarClass(TaskClass):
    def __init__(self, name, config, device=None, **kwargs):
        super().__init__(name=name, config=config, **kwargs)
        self.batch_size = (config.get("num_envs", 10),)
        self.device = device

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        self.batch_size = (num_envs,)
        self.device = device

        def make_env():
            base_env = camar_v0(
                map_generator=self.config.get("map_generator", "random_grid"),
                dynamic=self.config.get("dynamic", "HolonomicDynamic"),
                lifelong=self.config.get("lifelong", False),
                window=self.config.get("window", 0.3),
                max_steps=self.config.get("max_steps", 1000),
                frameskip=self.config.get("frameskip", 2),
                max_obs=self.config.get("max_obs", 3),
                pos_shaping_factor=self.config.get("pos_shaping_factor", 1.0),
                contact_force=self.config.get("contact_force", 500),
                contact_margin=self.config.get("contact_margin", 0.001),
                map_kwargs=self.config.get("map_kwargs", None),
                dynamic_kwargs=self.config.get("dynamic_kwargs", None),
            )

            return CamarWrapper(
                seed=seed,
                device=device,
                batch_size=num_envs,
                env=base_env,
            )

        return make_env

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        return self.config.get("max_steps", 50)

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        return {"agents": [f"agent_{i}" for i in range(env.num_agents)]}

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.full_observation_spec_unbatched.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.full_action_spec_unbatched

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        obs_shape = env.observation_spec["agents", "observation"].shape
        n_agents, obs_dim = obs_shape[-2], obs_shape[-1]
        return Composite(
            {"state": Unbounded(shape=(n_agents * (obs_dim + 5),), device=env.device)},
            device=env.device,
        )

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        """TorchRL transforms applied via ``TransformedEnv`` (wrapper-like: same env API, richer ``TensorDict``).

        Minimal set for GIRE + ``GireActorModel`` ``in_keys``:
        - ``CamarGlobalStateTransform``: builds root ``state`` from all agents' observations (teacher / CTDE).
        - ``InitTracker``: adds ``is_init`` so GRU resets hidden state at episode start (if use_gire).
        - ``TensorDictPrimer``: adds ``h_0`` so recurrent actor has a hidden state on the first step (if use_gire).
        """
        from benchmarl.environments.camar.transforms import CamarGlobalStateTransform
        transforms = [CamarGlobalStateTransform(group="agents")]

        # Only add h_0 and is_init if we explicitly enable GIRE in the task config
        if self.config.get("use_gire", False):
            from torchrl.envs import InitTracker, TensorDictPrimer
            from torchrl.data import Unbounded
            
            obs_shape = env.observation_spec["agents", "observation"].shape
            n_agents = obs_shape[-2]
            
            # Read hidden size from config, defaulting to 128 (BenchMARL GRU default)
            rnn_hidden_dim = self.config.get("rnn_hidden_dim", 128)
            
            primer = TensorDictPrimer(
                {
                    "h_0": Unbounded(
                        shape=(n_agents, rnn_hidden_dim),
                        device=env.device,
                    )
                },
                reset_key="_reset",
                expand_specs=True,
            )
            transforms.extend([InitTracker(init_key="is_init"), primer])

        return transforms

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "camar"

    def log_info(self, batch: TensorDictBase):
        return {}
