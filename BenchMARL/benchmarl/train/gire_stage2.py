#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Offline stage-2 training for GIRE (ptde-open main2.py).
#
#  Example:
#    python -m benchmarl.train.gire_stage2 \
#      --data path/to/map.npy --dims path/to/map_dims.csv \
#      --coach path/to/coach_net.th --out path/to/policy_app.th

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from benchmarl.models.gire import DecCoachNet, PolicyAppModule, train_policy_app_stage2


def load_ptde_npy(data_path: str, dims_path: str):
    dims = np.loadtxt(dims_path, dtype=np.int32, delimiter=",")
    raw = np.load(data_path)
    data = raw.reshape(-1, raw.shape[-1])
    o_dim, s_dim, a_dim, n_agents = int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])
    obs_input_dims = o_dim + a_dim + n_agents
    obs = data[:, :obs_input_dims].astype(np.float32)
    state = data[:, obs_input_dims : obs_input_dims + s_dim].astype(np.float32)
    return obs, state, obs_input_dims, s_dim, n_agents


def main():
    p = argparse.ArgumentParser(description="GIRE stage-2: train PolicyApp against frozen coach")
    p.add_argument("--data", required=True, help="Offline .npy (ptde format)")
    p.add_argument("--dims", required=True, help="dims.csv: obs, state, action, n_agents")
    p.add_argument("--coach", required=True, help="coach_net state_dict from stage 1")
    p.add_argument("--out", required=True, help="Output policy_app .th path")
    p.add_argument("--z-dims", type=int, default=8)
    p.add_argument("--rnn-hidden-dim", type=int, default=64)
    p.add_argument("--var-floor", type=float, default=0.01)
    p.add_argument("--n-steps", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    obs, state, obs_input_dims, state_dims, _ = load_ptde_npy(args.data, args.dims)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    coach = DecCoachNet(
        obs_input_dims=obs_input_dims,
        state_dims=state_dims,
        z_dims=args.z_dims,
        var_floor=args.var_floor,
    )
    coach.load_state_dict(torch.load(args.coach, map_location=device))
    policy_app = PolicyAppModule(obs_input_dims, args.rnn_hidden_dim, args.z_dims)

    obs_t = torch.from_numpy(obs)
    state_t = torch.from_numpy(state)
    train_policy_app_stage2(
        policy_app,
        coach,
        obs_t,
        state_t,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy_app.state_dict(), args.out)
    print(f"Saved policy_app to {args.out}")


if __name__ == "__main__":
    main()
