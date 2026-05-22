# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""RL loss implementations.

Losses in this package return `LossOutput`: a scalar `loss` tensor plus
`sum_metrics` and `max_metrics` dicts keyed by reduce-op. The trainer is
responsible for the SUM/MAX all-reduces across the loss mesh.

Example::

    from torchtitan.experiments.rl.loss import DAPOLoss

    loss_fn = DAPOLoss(DAPOLoss.Config(clip_low=0.2, clip_high=0.28))
    out = loss_fn(
        policy_logprobs=policy_logprobs,                  # [B, L]
        ref_logprobs=ref_logprobs,                        # [B, L]
        loss_mask=loss_mask,                              # [B, L] in {0, 1}
        advantages=advantages,                            # [B, L]
        num_global_valid_tokens=num_global_valid_tokens,  # scalar tensor
    )
    out.loss.backward()
    # out.sum_metrics["loss/mean"], out.max_metrics["loss/ratio/max_abs"], ...
"""

from torchtitan.experiments.rl.loss.dapo import DAPOLoss
from torchtitan.experiments.rl.loss.types import LossOutput

__all__ = ["DAPOLoss", "LossOutput"]
