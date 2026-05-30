# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-registry wiring and the controller's reference-KL guard.

These cover the integration the loss-level tests cannot: which loss each entry
point builds, and that RLTrainer.Config rejects a reference-KL GRPO config.
"""

import dataclasses

import pytest

from torchtitan.experiments.rl import config_registry as cr
from torchtitan.experiments.rl.loss import DAPOLoss, GRPOLoss

CONFIG_FNS = [
    cr.rl_grpo_qwen3_0_6b,
    cr.rl_grpo_qwen3_1_7b,
    cr.rl_grpo_qwen3_14b,
    cr.rl_grpo_qwen3_0_6b_batch_invariant,
]


@pytest.mark.parametrize("config_fn", CONFIG_FNS, ids=lambda f: f.__name__)
def test_registry_configs_use_dapo_loss(config_fn):
    """rl_grpo_* entry points intentionally build DAPOLoss (DAPO is the default)."""
    config = config_fn()
    assert isinstance(config.trainer.loss, DAPOLoss.Config)
    assert isinstance(config.trainer.loss.build(), DAPOLoss)


def test_controller_rejects_grpo_beta_without_reference():
    """RLTrainer.Config rejects GRPOLoss beta>0 until reference logprobs exist.

    dataclasses.replace re-runs RLTrainer.Config.__post_init__, which holds the
    guard.
    """
    config = cr.rl_grpo_qwen3_0_6b()
    with pytest.raises(NotImplementedError):
        dataclasses.replace(
            config,
            trainer=dataclasses.replace(config.trainer, loss=GRPOLoss.Config(beta=0.1)),
        )


def test_controller_allows_grpo_beta_zero():
    """GRPOLoss with beta=0 (no KL) passes the controller guard."""
    config = cr.rl_grpo_qwen3_0_6b()
    rebuilt = dataclasses.replace(
        config,
        trainer=dataclasses.replace(config.trainer, loss=GRPOLoss.Config(beta=0.0)),
    )
    assert isinstance(rebuilt.trainer.loss, GRPOLoss.Config)
