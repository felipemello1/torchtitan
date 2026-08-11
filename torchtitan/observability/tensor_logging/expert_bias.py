# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Post-update expert-bias telemetry at the optimizer pre-hook boundary."""

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle

from torchtitan.models.common.moe import MoE


@dataclass(frozen=True, slots=True)
class ExpertBiasSnapshot:
    values: torch.Tensor
    local_error: Exception | None


class ExpertBiasRecorder:
    """Samples persistent expert bias after its source-owned update hook."""

    def __init__(
        self,
        *,
        model: nn.Module,
        layer_ids: tuple[int, ...],
        device: torch.device,
    ) -> None:
        modules = []
        num_experts: int | None = None
        for layer_id in layer_ids:
            fqn = f"layers.{layer_id}.moe"
            module = model.get_submodule(fqn)
            if type(module) is not MoE:
                raise ValueError(
                    f"expert-bias logging requires an ordinary MoE at {fqn!r}"
                )
            if module.load_balance_coeff is None or module.expert_bias_E is None:
                raise ValueError(
                    "expert-bias logging requires auxiliary-loss-free balancing at "
                    f"{fqn!r}"
                )
            bias = module.expert_bias_E
            if bias.dtype is not torch.float32 or bias.ndim != 1:
                raise ValueError(
                    f"expert-bias logging expected a float32 vector at {fqn!r}"
                )
            if isinstance(bias, DTensor) and any(
                not placement.is_replicate() for placement in bias.placements
            ):
                raise ValueError(
                    f"expert-bias logging requires replicated storage at {fqn!r}"
                )
            if num_experts is None:
                num_experts = bias.numel()
            elif bias.numel() != num_experts:
                raise ValueError(
                    "expert-bias logging requires selected layers to have the same "
                    "number of experts"
                )
            modules.append(module)

        assert num_experts is not None
        self._layer_ids = layer_ids
        self._modules = tuple(modules)
        self._values = torch.zeros(
            (len(layer_ids), num_experts), dtype=torch.float32, device=device
        )
        self._record_next_step = False
        self._local_error: Exception | None = None
        self._hook_handle: RemovableHandle | None = None

    def bind_optimizer(self, optimizer: Optimizer) -> None:
        """Register after the source expert-bias pre-hook has been installed."""
        if self._hook_handle is not None:
            raise RuntimeError("expert-bias recorder is already bound")
        self._hook_handle = optimizer.register_step_pre_hook(
            partial(self._record_post_update_bias)
        )

    def begin_step(self, *, should_log: bool) -> None:
        """Arm one point sample for the next optimizer step."""
        self._values.zero_()
        self._local_error = None
        self._record_next_step = should_log

    def collect(self) -> ExpertBiasSnapshot:
        """Consume the point sample after the optimizer step."""
        if self._record_next_step and self._local_error is None:
            self._local_error = RuntimeError(
                "expert-bias logging did not observe the optimizer pre-hook"
            )
        self._record_next_step = False
        snapshot = ExpertBiasSnapshot(
            values=self._values.clone(),
            local_error=self._local_error,
        )
        self._values.zero_()
        self._local_error = None
        return snapshot

    def derive_metrics(
        self,
        snapshot: ExpertBiasSnapshot,
        *,
        window_steps: int,
    ) -> dict[str, int | float]:
        """Derive stable per-expert keys from the selected writer's replica."""
        values = snapshot.values.cpu()
        metrics: dict[str, int | float] = {}
        for row_index, layer_id in enumerate(self._layer_ids):
            prefix = f"tensor_metrics/layers.{layer_id}"
            for expert_id, value in enumerate(values[row_index]):
                metrics[
                    f"{prefix}.experts.{expert_id}.router_expert_bias_post_update"
                ] = float(value)
            metrics[
                f"{prefix}.moe.router_expert_bias_post_update.observation_count"
            ] = 1
            metrics[
                f"{prefix}.moe.router_expert_bias_post_update.window_steps"
            ] = window_steps
        return metrics

    def close(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def _record_post_update_bias(
        self,
        _optimizer: Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        if not self._record_next_step:
            return
        self._record_next_step = False
        for row_index, module in enumerate(self._modules):
            try:
                bias = module.expert_bias_E
                if bias is None:
                    raise ValueError("expert bias is absent")
                local_bias = bias.to_local() if isinstance(bias, DTensor) else bias
                if local_bias.shape != self._values[row_index].shape:
                    raise ValueError(
                        f"expected shape {tuple(self._values[row_index].shape)}, "
                        f"got {tuple(local_bias.shape)}"
                    )
                self._values[row_index].copy_(local_bias.detach())
            except Exception as error:
                if self._local_error is None:
                    self._local_error = ValueError(
                        f"invalid expert-bias sample for layer "
                        f"{self._layer_ids[row_index]}: {error}"
                    )
