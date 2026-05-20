#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Any, Dict, List

from tensordict import TensorDictBase
import torch


class Callback:
    """
    A Callback that can be added to experiments.
    To create your callback, you can inherit from this class
    and reimplement just the functions you need.

    Attributes:
        experiment (Experiment): the experiment associated to the callback
    """

    def __init__(self):
        self.experiment = None

    def on_setup(self):
        """A callback called at experiment setup."""
        pass

    def on_load_state_dict(self, state_dict: Dict[str, Any]):
        """A callback called at state_dict load."""
        pass

    def on_batch_collected(self, batch: TensorDictBase):
        """
        A callback called at the end of every collection step.

        Args:
            batch (TensorDictBase): batch of collected data

        """
        pass

    def on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        """
        A callback called for every training step.

        Args:
           batch (TensorDictBase): tensordict with the training batch
           group (str): group name

        Returns:
            TensorDictBase: a new tensordict containing the loss values

        """
        pass

    def on_train_end(self, training_td: TensorDictBase, group: str):
        """
        A callback called at the end of training.

        Args:
            training_td (TensorDictBase): tensordict containing the loss values
            group (str): group name

        """
        pass

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """
        A callback called at the end of every training step.

        Args:
            rollouts (list of TensorDictBase): tensordict containing the loss values

        """

        max_length_rollout_0 = 0
        for i in range(len(rollouts)):
            r = rollouts[i]
            next_done = r.get(("next", "done")).squeeze(-1)

            # First done index for this traj
            done_index = next_done.nonzero(as_tuple=True)[0]
            if done_index.numel() > 0:
                done_index = done_index[0]
                r = r[: done_index + 1]
            if i == 0:
                max_length_rollout_0 = max(r.batch_size[0], max_length_rollout_0)
            rollouts[i] = r

        mean_flowtime = []
        mean_makespan = []
        mean_coordination = []
        mean_success_rate = []
        for rollout in rollouts:

            flowtime = rollout.get(("next", "flowtime"))[-1]
            makespan = rollout.get(("next", "makespan"))[-1]
            coordination = rollout.get(("next", "coordination"))[-1]
            on_goal = rollout.get(("next", "agents", "on_goal"))[-1, :, :].squeeze(-1).to(torch.float32)

            mean_flowtime.append(flowtime)
            mean_makespan.append(makespan)
            mean_coordination.append(coordination)
            mean_success_rate.append(on_goal)

        mean_flowtime = torch.stack(mean_flowtime).nanmean()
        mean_makespan = torch.stack(mean_makespan).nanmean()
        mean_coordination = torch.stack(mean_coordination).nanmean()
        mean_success_rate = torch.stack(mean_success_rate).mean()

        self.experiment.logger.log(
            {
                "eval/mean_flowtime": mean_flowtime,
                "eval/mean_makespan": mean_makespan,
                "eval/mean_coordination": mean_coordination,
                "eval/mean_success_rate": mean_success_rate,
            },
            step=self.experiment.n_iters_performed,
        )

    def on_state_dict(self, state_dict: Dict[str, Any]):
        """A callback called at state_dict save."""
        pass


class CallbackNotifier:
    def __init__(self, experiment, callbacks: List[Callback]):
        self.callbacks = callbacks
        for callback in self.callbacks:
            callback.experiment = experiment

    def _on_setup(self):
        for callback in self.callbacks:
            callback.on_setup()

    def _on_load_state_dict(self, state_dict: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_load_state_dict(state_dict)

    def _on_batch_collected(self, batch: TensorDictBase):
        for callback in self.callbacks:
            callback.on_batch_collected(batch)

    def _on_train_step(self, batch: TensorDictBase, group: str) -> TensorDictBase:
        train_td = None
        for callback in self.callbacks:
            td = callback.on_train_step(batch, group)
            if td is not None:
                if train_td is None:
                    train_td = td
                else:
                    train_td.update(td)
        return train_td

    def _on_train_end(self, training_td: TensorDictBase, group: str):
        for callback in self.callbacks:
            callback.on_train_end(training_td, group)

    def _on_evaluation_end(self, rollouts: List[TensorDictBase]):
        for callback in self.callbacks:
            callback.on_evaluation_end(rollouts)

    def _on_state_dict(self, state_dict: Dict[str, Any]):
        for callback in self.callbacks:
            callback.on_state_dict(state_dict)
