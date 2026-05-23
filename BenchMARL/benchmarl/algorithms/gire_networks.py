#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  GIRE / PTDE network modules

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class DecCoachNet_TwoStage(nn.Module):
    """GIP teacher (Stage 1): personalized global knowledge z ~ N(mu, sigma).

    Input contract (must match tensors from the env / GireMappo pipeline):
        state: (batch, state_dims) — global state (CTDE; one vector per env step).
        obs:   (batch, n_agents, obs_input_dims) — local obs per agent (same layout as
               BenchMARL ``(group, "observation")``).

    Output:
        Normal distribution with loc/scale shape (batch, n_agents, z_dims).

    TODO: verify obs_input_dims and state_dims against ``observation_spec`` and
    ``state_spec`` when wiring into GireMappo (CAMAR currently has state_spec=None).
    """

    def __init__(
        self,
        obs_input_dims: int,
        state_dims: int,
        z_dims: int,
        high_hyper_hidden_dims: int = 64,
        var_floor: float = 0.002,
    ):
        super().__init__()
        self.state_dims = state_dims
        self.z_dims = z_dims
        self.var_floor = var_floor

        self.w1 = nn.Sequential(
            nn.Linear(obs_input_dims, high_hyper_hidden_dims),
            nn.ReLU(),
            nn.Linear(high_hyper_hidden_dims, state_dims * z_dims),
        )
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

    def forward(self, state: torch.Tensor, obs: torch.Tensor) -> D.Normal:
        """See class docstring for input contract.

        TODO: verify at runtime that state/obs shapes match init dims and that
        broadcasting (state vs per-agent w1/b1) matches ptde-open DecCoachNet_TwoStage.
        """
        w1 = self.w1(obs).view(*obs.shape[:-1], self.state_dims, self.z_dims)
        b1 = self.b1(obs).view(*obs.shape[:-1], 1, self.z_dims)

        # state is global (batch, state_dim), expand to match agents (batch, n_agents, 1, state_dim)
        state_reshaped = state.unsqueeze(-2).unsqueeze(-2)
        state_reshaped = state_reshaped.expand(*obs.shape[:-1], 1, self.state_dims)
            
        z_hidden = F.elu(torch.matmul(state_reshaped, w1) + b1).squeeze(-2)

        mu = self.mu(z_hidden)
        sigma = self.sigma(z_hidden)
        sigma = torch.clamp(torch.exp(sigma), min=self.var_floor)

        dist = D.Normal(mu.clone(), sigma.clone() ** 0.5)
        return dist


class PolicyAppModule_TwoStage(nn.Module):
    """GIP student (Stage 2): approximates teacher z using only local obs (returns mu').

    Input contract:
        obs: (batch, n_agents, obs_input_dims) — same layout as teacher obs input.

    Output:
        (batch, n_agents, z_dims) — student mean z'; distillation uses MSE vs teacher mu.

    TODO: verify obs_input_dims against ``observation_spec`` when wiring into GireMappo.
    TODO: verify forward output shape and alignment with teacher mu at Stage 2 integration.
    """

    def __init__(self, obs_input_dims: int, z_dims: int):
        super().__init__()
        self.poli_app1 = nn.Linear(obs_input_dims, obs_input_dims)
        self.poli_app2 = nn.Sequential(
            nn.Linear(obs_input_dims, z_dims),
            nn.LeakyReLU(),
            nn.Linear(z_dims, z_dims),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """See class docstring for input contract.

        TODO: verify obs shape and output (batch, n_agents, z_dims) at runtime in Stage 2.
        """
        h = F.relu(self.poli_app1(obs), inplace=True)
        z_dot_mu = self.poli_app2(h)
        return z_dot_mu


class GireActorModel(nn.Module):
    """Custom Actor for GIRE: processes 'obs' via GRU -> 'h', then concats 'h' with 'z'.

    Input contract:
        obs: (batch, n_agents, obs_input_dims)
        z: (batch, n_agents, z_dims) — either from Teacher (Stage 1) or Student (Stage 2)
        is_init: (batch, n_agents, 1) — episode reset flag
        h_0: (batch, n_agents, rnn_hidden_dim) — optional initial hidden state

    Output:
        logits (action features): (batch, n_agents, out_features)
        h_n: (batch, n_agents, rnn_hidden_dim) — next hidden state

    TODO: verify __init__ dims (obs_input_dims, z_dims, out_features for continuous
    actions = 2 * action_dim per BenchMARL MAPPO) against env/action_spec.
    TODO: verify forward shapes and is_init/h_0 reset against BenchMARL GRU + collection.
    """

    def __init__(
        self,
        obs_input_dims: int,
        z_dims: int,
        rnn_hidden_dim: int,
        out_features: int,
    ):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc1 = nn.Linear(obs_input_dims, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, rnn_hidden_dim)
        self.fc_out = nn.Linear(rnn_hidden_dim + z_dims, out_features)

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        is_init: torch.Tensor,
        h_0: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """See class docstring for input contract.

        TODO: verify flattening (batch/time/agents), is_init broadcast, h_0 reset, and
        output reshape match GireMappo / TorchRL rollout layout.
        """
        original_shape = obs.shape[:-1]
        batch_size = torch.prod(torch.tensor(original_shape)).item()

        if h_0 is None:
            h_0 = torch.zeros(*original_shape, self.rnn_hidden_dim, device=obs.device)

        # Expand is_init to match h_0 dimensions (batch, ..., 1)
        is_init_exp = is_init
        # We need is_init_exp to have length len(original_shape) + 1
        while is_init_exp.dim() <= len(original_shape):
            is_init_exp = is_init_exp.unsqueeze(-1)
        is_init_exp = is_init_exp.expand(*original_shape, 1)

        h_0 = torch.where(is_init_exp.bool(), torch.zeros_like(h_0), h_0)

        obs_flat = obs.view(batch_size, -1)
        z_flat = z.view(batch_size, -1)
        h_0_flat = h_0.view(batch_size, -1)

        x = F.relu(self.fc1(obs_flat), inplace=True)
        h_n = self.rnn(x, h_0_flat)

        h_oi = F.relu(self.fc2(h_n), inplace=True)
        q = self.fc_out(torch.cat([h_oi, z_flat], dim=-1))

        return q.view(*original_shape, -1), h_n.view(*original_shape, self.rnn_hidden_dim)
