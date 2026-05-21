#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDictBase
from tensordict.utils import expand_as_right, unravel_key_list
from torchrl.data.tensor_specs import Composite, Unbounded

from benchmarl.models.common import Model, ModelConfig

GIRE_DEBUG_BUILD = "2026-05-21-rollout_step-v2"
print(f"[GIRE-DEBUG] loaded {__file__} build={GIRE_DEBUG_BUILD}", flush=True)

_gire_forward_calls = 0

try:
    from torchrl.envs.utils import ExplorationType, exploration_type
except ImportError:
    ExplorationType = None  # type: ignore
    exploration_type = None  # type: ignore


class DecCoachNet(nn.Module):
    """GIP teacher: Gaussian z from flattened (B*, A, E) obs features and (B*, S) state."""

    def __init__(
        self,
        obs_input_dims: int,
        state_dims: int,
        z_dims: int,
        var_floor: float,
    ):
        super().__init__()
        self.state_dims = state_dims
        self.z_dims = z_dims
        self.var_floor = var_floor

        self.w1 = nn.Linear(obs_input_dims, state_dims * z_dims)
        self.b1 = nn.Linear(obs_input_dims, z_dims)
        self.mu = nn.Sequential(
            nn.Linear(z_dims, z_dims),
            nn.LeakyReLU(),
            nn.Linear(z_dims, z_dims),
        )
        self.sigma = nn.Sequential(
            nn.Linear(z_dims, z_dims),
            nn.LeakyReLU(),
            nn.Linear(z_dims, z_dims),
        )

    def forward(self, obs_flat: torch.Tensor, state_flat: torch.Tensor) -> D.Normal:
        """obs_flat: (N, E), state_flat: (N, S). (b*seq*a, e), (b*seq*a, s)"""
        n = obs_flat.shape[0]
        s = state_flat.unsqueeze(1)
        w1 = self.w1(obs_flat).view(n, self.state_dims, self.z_dims)
        b1 = self.b1(obs_flat).view(n, 1, self.z_dims)
        z_hidden = F.elu(torch.matmul(s, w1) + b1).squeeze(1)

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.var_floor)
        return D.Normal(mu, sigma**0.5)


class PolicyAppModule(nn.Module):
    """Student (stage_two): deterministic z_mu from local features only."""

    def __init__(
        self,
        obs_input_dims: int,
        rnn_hidden_dim: int,
        z_dims: int,
    ):
        super().__init__()
        self.poli_app1 = nn.Linear(obs_input_dims, rnn_hidden_dim)
        self.poli_app2 = nn.Sequential(
            nn.Linear(rnn_hidden_dim, z_dims),
            nn.LeakyReLU(),
            nn.Linear(z_dims, z_dims),
        )

    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:
        z_dot_hidden = F.relu(self.poli_app1(obs_flat), inplace=True)
        return self.poli_app2(z_dot_hidden)


class GireAgentHead(nn.Module):
    """GIREAgent head: concat(h, z) -> logits."""

    def __init__(self, rnn_hidden_dim: int, z_dims: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim + z_dims, n_actions)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        *prefix, a, eh = h.shape
        h_flat = h.reshape(-1, eh)
        z_flat = z.reshape(-1, z.shape[-1])
        h_oi = F.relu(self.fc1(h_flat), inplace=True)
        out = self.fc2(torch.cat([h_oi, z_flat], dim=-1))
        return out.view(*prefix, a, -1)


class GIRE(Model):
    """PTDE GIRE: stage 1 = train teacher (MAPPO); rollout/eval student via ExplorationType."""

    def __init__(
        self,
        rnn_hidden_dim: int,
        z_dims: int,
        state_dims: int,
        two_hyper_layers: bool,
        high_hyper_hidden_dims: int,
        var_floor: float,
        obs_last_action: bool,
        obs_agent_id: bool,
        distill_coef: float,
        gire_stage: int = 1,
        **kwargs,
    ):
        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.n_actions = int(self.output_leaf_spec.shape[-1])
        self.rnn_hidden_dim = rnn_hidden_dim
        self.z_dims = z_dims
        self.state_dims = state_dims
        self.var_floor = var_floor
        self.obs_last_action = obs_last_action
        self.obs_agent_id = obs_agent_id
        self.distill_coef = distill_coef
        self.gire_stage = int(gire_stage)

        self.hidden_state_name = (self.agent_group, f"_hidden_gire_{self.model_index}")
        self.rnn_keys = unravel_key_list(["is_init", self.hidden_state_name])
        self.in_keys = list(self.input_spec.keys(True, True)) + self.rnn_keys

        self._global_state_key: Optional[Tuple] = None
        for k in self.input_spec.keys(True, True):
            if isinstance(k, tuple) and len(k) > 0 and k[0] == self.agent_group:
                continue
            self._global_state_key = k
            break

        self._observation_key = (self.agent_group, "observation")
        self._prev_action_key = (self.agent_group, "prev_action")

        obs_dim = int(self.input_spec[self._observation_key].shape[-1])
        keys_in = dict(self.input_spec.items(True, True))
        prev_dim = (
            int(self.input_spec[self._prev_action_key].shape[-1])
            if self.obs_last_action and self._prev_action_key in keys_in
            else 0
        )
        agent_id_dim = self.n_agents if self.obs_agent_id else 0
        self.prev_action_dim = prev_dim
        self.obs_input_dims = obs_dim + prev_dim + agent_id_dim

        self.encoder_fc = nn.Linear(self.obs_input_dims, rnn_hidden_dim)
        self.gru_cell = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)

        self.coach_net = DecCoachNet(
            obs_input_dims=self.obs_input_dims,
            state_dims=state_dims,
            z_dims=z_dims,
            var_floor=var_floor,
        )
        self.policy_app = PolicyAppModule(
            obs_input_dims=self.obs_input_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            z_dims=z_dims,
        )
        self.agent_head = GireAgentHead(rnn_hidden_dim, z_dims, self.n_actions)

        if self.gire_stage == 1:
            for p in self.policy_app.parameters():
                p.requires_grad = False
        elif self.gire_stage == 2:
            for p in self.coach_net.parameters():
                p.requires_grad = False
            self.coach_net.eval()

    def _use_student_z(self, tensordict: TensorDictBase, env_step: bool) -> bool:
        """Stage 1: teacher for MAPPO + collection; student only under DETERMINISTIC (eval)."""
        override = tensordict.get((self.agent_group, "gire_use_student"), None)
        if override is not None:
            return bool(override.reshape(-1)[0].item())

        if self.gire_stage == 2:
            return True

        if not env_step:
            return False
        if exploration_type is None or ExplorationType is None:
            return False
        return exploration_type() == ExplorationType.DETERMINISTIC

    def _gather_obs_features(self, tensordict: TensorDictBase) -> torch.Tensor:
        obs = tensordict.get(self._observation_key)
        parts = [obs]
        if self.obs_last_action:
            pa = tensordict.get(self._prev_action_key)
            if pa is not None:
                parts.append(pa.to(obs.dtype))
            else:
                *lead, a, _eo = obs.shape
                parts.append(
                    torch.zeros(
                        *lead,
                        a,
                        self.prev_action_dim,
                        device=obs.device,
                        dtype=obs.dtype,
                    )
                )
        if self.obs_agent_id:
            *lead, a, _ = obs.shape
            eye = torch.eye(self.n_agents, device=obs.device, dtype=obs.dtype)
            aid = eye.view(*([1] * (obs.ndim - 2)), self.n_agents, self.n_agents).expand(
                *lead, a, self.n_agents
            )
            parts.append(aid)
        return torch.cat(parts, dim=-1)

    def _state_bt(
        self, tensordict: TensorDictBase, b: int, t: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Returns (B, T, S) global state for coach; zeros if missing."""
        if self._global_state_key is None:
            return torch.zeros(b, t, self.state_dims, device=device, dtype=dtype)
        s = tensordict.get(self._global_state_key)
        if s is None:
            return torch.zeros(b, t, self.state_dims, device=device, dtype=dtype)
        if s.ndim == 2:
            s = s.unsqueeze(1).expand(b, t, -1)
        elif s.shape[1] == 1 and t > 1:
            s = s.expand(b, t, -1)
        elif s.shape[1] != t:
            s = s[:, :t]
        return s[..., : self.state_dims]

    def _run_encoder_over_time(
        self,
        x: torch.Tensor,
        is_init: torch.Tensor,
        h_0: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, a, _e = x.shape
        if h_0 is None:
            h = torch.zeros(b, a, self.rnn_hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h = h_0
            if h.ndim == 4:
                h = h.squeeze(-2)
            if h.shape[-2] != a and h.shape[-2] == 1:
                h = h.expand(b, a, -1).contiguous()

        if is_init.shape == (b, t, 1):
            is_exp = is_init.unsqueeze(-2).expand(b, t, a, 1)
        elif is_init.shape == (b, t, a, 1):
            is_exp = is_init
        elif is_init.shape == (b, 1, 1):
            is_exp = is_init.expand(b, t, 1).unsqueeze(-2).expand(b, t, a, 1)
        else:
            raise ValueError(
                f"GIRE: unexpected is_init shape {tuple(is_init.shape)} for (B,T)=({b},{t}), A={a}"
            )

        is_exp = is_exp.bool()

        outs = []
        for ti in range(t):
            init_t = is_exp[:, ti, :, :]
            h = torch.where(expand_as_right(init_t, h), torch.zeros_like(h), h)
            xt = F.relu(self.encoder_fc(x[:, ti]), inplace=True)
            h = self.gru_cell(xt.reshape(b * a, -1), h.reshape(b * a, -1)).view(b, a, -1)
            outs.append(h)
        h_out = torch.stack(outs, dim=1)
        return h_out, h

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        global _gire_forward_calls
        if _gire_forward_calls < 3:
            print(
                f"[GIRE-DEBUG] _forward #{_gire_forward_calls} build={GIRE_DEBUG_BUILD}",
                flush=True,
            )
            _gire_forward_calls += 1

        obs_feats = self._gather_obs_features(tensordict)
        is_init = tensordict.get("is_init")
        h_0 = tensordict.get(self.hidden_state_name, None)

        missing_batch = False
        if obs_feats.ndim == 2:
            missing_batch = True
            obs_feats = obs_feats.unsqueeze(0)
            if h_0 is not None:
                h_0 = h_0.unsqueeze(0)
            is_init = is_init.unsqueeze(0)

        if obs_feats.ndim == 3:
            obs_feats = obs_feats.unsqueeze(1)
            if h_0 is not None and h_0.ndim == 3:
                h_0 = h_0.unsqueeze(1)
            if is_init.ndim == 2:
                is_init = is_init.unsqueeze(1)

        b, seq, a, _e = obs_feats.shape
        # PPO replay may still carry hidden keys; unroll over T from is_init, not stored h.
        rollout_step = seq == 1
        if not rollout_step:
            h_0 = None

        h_out, h_n = self._run_encoder_over_time(obs_feats, is_init, h_0)

        state_bt = self._state_bt(tensordict, b, seq, obs_feats.device, obs_feats.dtype)

        flat_obs = obs_feats.reshape(b * seq * a, _e)
        flat_state = state_bt.unsqueeze(2).expand(b, seq, a, state_bt.shape[-1]).reshape(
            b * seq * a, -1
        )

        use_student = self._use_student_z(tensordict, rollout_step)

        if rollout_step:
            h_last = h_out[:, -1]
            if use_student:
                z_for_logits = self.policy_app(flat_obs).view(b, seq, a, self.z_dims)
                z_for_logits = (
                    z_for_logits[:, -1]
                    if z_for_logits.shape[1] > 1
                    else z_for_logits.squeeze(1)
                )
            else:
                z_teacher_dist = self.coach_net(flat_obs, flat_state)
                z_for_logits = z_teacher_dist.rsample().view(b, seq, a, self.z_dims)[:, -1]
            logits = self.agent_head(h_last, z_for_logits)
            logits = logits.squeeze(1)
        else:
            if use_student:
                z_for_logits = self.policy_app(flat_obs).view(b, seq, a, self.z_dims)
            else:
                z_teacher_dist = self.coach_net(flat_obs, flat_state)
                z_for_logits = z_teacher_dist.rsample().view(b, seq, a, self.z_dims)
            logits = self.agent_head(h_out, z_for_logits)

        if missing_batch:
            logits = logits.squeeze(0)
            h_n = h_n.squeeze(0)

        tensordict.set(self.out_key, logits)
        if rollout_step:
            tensordict.set(("next", *self.hidden_state_name), h_n.unsqueeze(-2))
        return tensordict

    def _perform_checks(self):
        super()._perform_checks()
        if not self.input_has_agent_dim:
            raise ValueError("GIRE expects input_has_agent_dim=True")


@dataclass
class GIREConfig(ModelConfig):
    rnn_hidden_dim: int = MISSING
    z_dims: int = MISSING
    state_dims: int = MISSING
    two_hyper_layers: bool = False
    high_hyper_hidden_dims: int = 64
    var_floor: float = 0.01
    obs_last_action: bool = False
    obs_agent_id: bool = True
    distill_coef: float = 0.0
    gire_stage: int = 1

    @staticmethod
    def associated_class():
        return GIRE

    @property
    def is_rnn(self) -> bool:
        return True

    def get_model_state_spec(self, model_index: int = 0) -> Composite:
        return Composite(
            {
                f"_hidden_gire_{model_index}": Unbounded(
                    shape=(1, self.rnn_hidden_dim),
                )
            }
        )


def train_policy_app_stage2(
    policy_app: PolicyAppModule,
    coach_net: DecCoachNet,
    obs_features: torch.Tensor,
    state_features: torch.Tensor,
    *,
    n_steps: int = 500_000,
    batch_size: int = 256,
    lr: float = 1e-4,
    grad_norm_clip: float = 10.0,
    device: Optional[torch.device] = None,
) -> PolicyAppModule:
    """Stage 2 (ptde-open main2.py): frozen coach, MSE on policy_app vs teacher.loc."""
    device = device or obs_features.device
    policy_app = policy_app.to(device)
    coach_net = coach_net.to(device)
    coach_net.eval()
    for p in coach_net.parameters():
        p.requires_grad = False

    policy_app.train()
    optimizer = torch.optim.Adam(policy_app.parameters(), lr=lr, weight_decay=0.0)
    n = obs_features.shape[0]
    obs_features = obs_features.to(device)
    state_features = state_features.to(device)

    for _ in range(n_steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        obs_b = obs_features[idx]
        state_b = state_features[idx]
        with torch.no_grad():
            z_teacher = coach_net(obs_b, state_b).loc
        z_student = policy_app(obs_b)
        loss = ((z_teacher - z_student) ** 2).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_app.parameters(), grad_norm_clip)
        optimizer.step()
    return policy_app
