#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
from dataclasses import dataclass
from typing import Type, Optional

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.objectives import LossModule
from benchmarl.algorithms.mappo import Mappo, MappoConfig
from benchmarl.models.common import ModelConfig

from benchmarl.algorithms.gire_networks import (
    DecCoachNet_TwoStage,
    PolicyAppModule_TwoStage,
)

class GireMappo(Mappo):
    """GIRE + MAPPO algorithm."""

    def __init__(self, z_dims: int, training_stage: int, var_floor: float, teacher_checkpoint: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.z_dims = z_dims
        self.training_stage = training_stage
        self.var_floor = var_floor
        self.teacher_checkpoint = teacher_checkpoint

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        """Build policy as TensorDictSequential: GIP block -> GireActorModel -> MAPPO head.

        Each block is a TensorDictModule (reads/writes named keys in the rollout TensorDict).
        Goal: teacher or student produces z, actor consumes obs+z and writes logits, then
        NormalParamExtractor + ProbabilisticActor (from parent Mappo) sample actions.
        Branch on training_stage selects z_dist (Stage 1) vs z_dot (Stage 2).

        TODO: finish sequential wiring and verify state key exists for CAMAR.
        """
        n_agents = len(self.group_map[group])
        obs_dim = self.observation_spec[group, "observation"].shape[-1]
        
        # State might not be available during building if the transform hasn't run yet
        # We can calculate its size based on the CAMAR transform
        state_dim = n_agents * obs_dim

        # 1. Teacher (Stage 1): generates z_dist ~ N(mu, sigma)
        teacher_net = DecCoachNet_TwoStage(
            obs_input_dims=obs_dim, state_dims=state_dim, z_dims=self.z_dims, var_floor=self.var_floor
        )
        teacher_module = TensorDictModule(
            teacher_net,
            in_keys=["state", (group, "observation")],
            out_keys=[(group, "z_dist")],
        )
        if not hasattr(self, "teacher_modules"): self.teacher_modules = {}
        self.teacher_modules[group] = teacher_module
        
        # 2. Student (Stage 2): predicts z_dot (mean)
        student_net = PolicyAppModule_TwoStage(obs_input_dims=obs_dim, z_dims=self.z_dims)
        student_module = TensorDictModule(
            student_net,
            in_keys=[(group, "observation")],
            out_keys=[(group, "z_dot")],
        )
        if not hasattr(self, "student_modules"): self.student_modules = {}
        self.student_modules[group] = student_module

        # 3. Actor (GIRE): takes obs + z + rnn_state -> logits + next_rnn_state
        from benchmarl.algorithms.gire_networks import GireActorModel
        out_features = self.action_spec[group, "action"].shape[-1]
        if continuous:
            out_features *= 2

        actor_net = GireActorModel(
            obs_input_dims=obs_dim,
            z_dims=self.z_dims,
            rnn_hidden_dim=model_config.rnn_hidden_dim if hasattr(model_config, "rnn_hidden_dim") else 128,
            out_features=out_features,
        )
        
        z_in_key = (group, "z_dist_rsample") if self.training_stage == 1 else (group, "z_dot")
        actor_module = TensorDictModule(
            actor_net,
            in_keys=[(group, "observation"), z_in_key, "is_init", "h_0"],
            out_keys=[(group, "logits"), ("next", "h_0")],
        )

        # 4. Action Distribution (MAPPO logic)
        from torchrl.modules import NormalParamExtractor, ProbabilisticActor
        from torchrl.modules.distributions import IndependentNormal, TanhNormal

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            dist_class = IndependentNormal if not self.use_tanh_normal else TanhNormal
            dist_kwargs = {"low": self.action_spec[(group, "action")].space.low, "high": self.action_spec[(group, "action")].space.high} if self.use_tanh_normal else {}
            
            if self.training_stage == 1:
                rsample_module = TensorDictModule(
                    lambda dist: dist.rsample(),
                    in_keys=[(group, "z_dist")],
                    out_keys=[(group, "z_dist_rsample")],
                )
                seq_module = TensorDictSequential(teacher_module, rsample_module, actor_module, extractor_module)
            else:
                seq_module = TensorDictSequential(student_module, actor_module, extractor_module)
                if hasattr(self, "teacher_checkpoint") and self.teacher_checkpoint:
                    import torch
                    print(f"Loading Teacher and Actor weights from {self.teacher_checkpoint}")
                    ckpt = torch.load(self.teacher_checkpoint, map_location=self.device)
                    loss_sd = ckpt.get(f"loss_{group}", {})
                    teacher_state, actor_state = {}, {}
                    for k, v in loss_sd.items():
                        if "actor_network_params.module.0.module.0.module." in k:
                            teacher_state[k.replace("actor_network_params.module.0.module.0.module.", "")] = v
                        elif "actor_network_params.module.0.module.2.module." in k:
                            actor_state[k.replace("actor_network_params.module.0.module.2.module.", "")] = v
                    teacher_net.load_state_dict(teacher_state, strict=False)
                    actor_net.load_state_dict(actor_state, strict=False)

            prob_module = ProbabilisticActor(
                module=seq_module,
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=dist_class,
                distribution_kwargs=dist_kwargs,
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )
            policy = prob_module
        else:
            print("Discrete action, ALARM!!!")
            in_keys = {"logits": (group, "logits"), "mask": (group, "action_mask")} if self.action_mask_spec is not None else [(group, "logits")]
            dist_class = MaskedCategorical if self.action_mask_spec is not None else Categorical
            prob_module = ProbabilisticActor(
                module=TensorDictModule(lambda x: x, in_keys=[], out_keys=[]), # Dummy module
                spec=self.action_spec[group, "action"],
                in_keys=in_keys,
                out_keys=[(group, "action")],
                distribution_class=dist_class,
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )

        # Assemble Sequence is handled inside the continuous branch above
        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        if self.training_stage == 1:
            return super().process_batch(group, batch)
        # Stage 2: Distillation doesn't require GAE/value estimation
        return batch

    def process_loss_vals(self, group: str, loss_vals: TensorDictBase) -> TensorDictBase:
        if self.training_stage == 1:
            return super().process_loss_vals(group, loss_vals)
        return loss_vals

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> tuple[LossModule, bool]:
        if self.training_stage == 1:
            return super()._get_loss(group, policy_for_loss, continuous)
        
        # Stage 2: MSE Loss for Student
        from torchrl.objectives import LossModule
        import torch
        
        teacher_module = self.teacher_modules[group]
        student_module = self.student_modules[group]

        class StudentDistillationLoss(LossModule):
            def __init__(self, teacher, student, group_name):
                super().__init__()
                self.teacher = teacher
                self.student = student
                self.group_name = group_name
                
            def forward(self, tensordict):
                # 1. Get teacher prediction without tracking gradients
                with torch.no_grad():
                    self.teacher(tensordict)
                
                # 2. Get student prediction (tracks gradients)
                self.student(tensordict)
                
                # 3. Compute MSE Loss
                z_dist = tensordict.get((self.group_name, "z_dist"))
                #############################################################################
                
                # Retrieve the mean/loc from the distribution
                # Since torchrl / tensordict handles distributions inside TensorDicts differently
                # depending on the version, we provide fallbacks.
                if hasattr(z_dist, "loc") and not callable(z_dist.loc):
                    teacher_mu = z_dist.loc
                elif hasattr(z_dist, "mean") and not callable(z_dist.mean):
                    teacher_mu = z_dist.mean
                elif hasattr(z_dist, "get") and z_dist.get("loc") is not None:
                    teacher_mu = z_dist.get("loc")
                elif type(z_dist).__name__ == "NonTensorData" and hasattr(z_dist, "data") and hasattr(z_dist.data, "loc"):
                    teacher_mu = z_dist.data.loc
                else:
                    # In some recent versions, Normal distributions inside tensordicts
                    # are wrapped such that we can't easily access .loc, so we fallback
                    # to a dummy tensor of the same shape if this breaks during tracing.
                    student_mu = tensordict.get((self.group_name, "z_dot"))
                    teacher_mu = torch.zeros_like(student_mu)
                    print(f"WARNING: Unknown type for z_dist: {type(z_dist)}, falling back to zeros. EVERYTHING FAILED!")
                #############################################################################
                student_mu = tensordict.get((self.group_name, "z_dot"))
                
                loss = ((teacher_mu - student_mu) ** 2).sum(dim=-1).mean()
                
                from tensordict import TensorDict
                return TensorDict({"loss_distillation": loss}, [])

            def load_state_dict(self, state_dict, strict=False):
                """
                Called if BenchMARL tries to reload the full experiment state.
                I am not entirely sure about the necessity and correctness of this method 
                when just using teacher_checkpoint, but keeping it as a fallback.
                """
                print("DEBUG: StudentDistillationLoss.load_state_dict() was called АЛАРМ!")
                teacher_state = {}
                for k, v in state_dict.items():
                    if "actor_network_params.module.0.module.0.module." in k:
                        new_k = k.replace("actor_network_params.module.0.module.0.module.", "teacher.module.")
                        teacher_state[new_k] = v
                    elif "actor_network_params.module.module.0." in k:
                        # Fallback just in case the wrap depth changed
                        new_k = k.replace("actor_network_params.module.module.0.", "teacher.module.")
                        teacher_state[new_k] = v
                super().load_state_dict(teacher_state, strict=False)

        return StudentDistillationLoss(teacher_module, student_module, group), False

    def _get_parameters(self, group: str, loss: LossModule) -> dict[str, list]:
        if self.training_stage == 1:
            return super()._get_parameters(group, loss)
        
        # Stage 2: Optimize only student
        return {
            "loss_distillation": list(loss.student.parameters()),
        }

@dataclass
class GireMappoConfig(MappoConfig):
    """Config for GIRE + MAPPO (PTDE two-stage training)."""

    z_dims: int = 128
    training_stage: int = 1
    var_floor: float = 0.002
    teacher_checkpoint: Optional[str] = None

    @staticmethod
    def associated_class() -> Type:
        return GireMappo
