# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block-major BF16 expert weights for Dist-MoE inference."""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)

from . import _dist_moe as dist_moe
from .routed_experts import DistMoeRoutedExperts


__all__ = ["BlockMajorDistMoeRoutedExperts"]

# `dist_moe.prepare_bf16_block_major_weight` stores K in 64-wide tiles, so K must
# be a multiple of 64: D for W13, F for W2.
_BLOCK_MAJOR_K = 64


@dataclass(frozen=True, slots=True)
class _DistMoeBf16Operands:
    """Block-major BF16 storage of one expert weight, owned by one FSDP unshard lifetime."""

    block_major_data: torch.Tensor

    def prepared(self) -> dist_moe.PreparedBf16Weight:
        """Return the annex facade consumed by one Dist-MoE invocation."""
        return dist_moe.PreparedBf16Weight._create(
            block_major_data=self.block_major_data
        )


class _DistMoeBf16ShardedTensor(_ShardedFSDPTensor):
    """Persistent W13 or W2 parameter whose FSDP unshard writes a block-major copy.

        gathered (row-major)       block_major_data
        W13 [E, 2, F, D]      ->   [E, D / 64, 2F, 64]
        W2  [E, D, F]         ->   [E, F / 64, D, 64]

    Each unshard that gathers weights (the initial load, each weight sync)
    rewrites the same ``block_major_data``, so CUDA graphs keep one address.
    """

    def _build_operands(
        self,
        logical_tensor: torch.Tensor,
        out: _DistMoeBf16Operands | None = None,
    ) -> _DistMoeBf16Operands:
        # W13 [E, 2, F, D] -> [E, 2F, D]; W2 [E, D, F] is unchanged.
        weight_ENK = logical_tensor.flatten(1, -2)
        if out is None:
            prepared = dist_moe.prepare_bf16_block_major_weight(weight_ENK)
            return _DistMoeBf16Operands(prepared.block_major_data)
        dist_moe.prepare_bf16_block_major_weight(weight_ENK, out=out.prepared())
        return out


class BlockMajorDistMoeRoutedExperts(DistMoeRoutedExperts):
    """BF16 Dist-MoE routed experts whose unsharded weights are block-major, for inference.

    Parameters, state dicts and checkpoints keep the row-major W13/W2 of
    ``DistMoeRoutedExperts``. FSDP's unshard writes the block-major copy (see
    ``_DistMoeBf16ShardedTensor``), which Dist-MoE's BF16 inference GEMMs read
    faster. A training forward raises.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DistMoeRoutedExperts.Config):
        """Configure block-major BF16 experts (``DistMoeTransform.bf16_block_major_inference``)."""

        def __post_init__(self) -> None:
            """Validate the common expert policy and the block-major K tile."""
            DistMoeRoutedExperts.Config.__post_init__(self)
            hidden_dim = self.w13.in_features
            intermediate_dim = self.w2.in_features
            if (
                hidden_dim % _BLOCK_MAJOR_K != 0
                or intermediate_dim % _BLOCK_MAJOR_K != 0
            ):
                raise ValueError(
                    "Block-major Dist-MoE weights need hidden_dim (D) and "
                    f"intermediate_dim (F) to be multiples of {_BLOCK_MAJOR_K}; "
                    f"got D={hidden_dim}, F={intermediate_dim}"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self._wrap_weights()

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        # spmd_types returns a plain tensor when EP/TP shards a weight. Restore
        # the FSDP extension wrappers before fully_shard() consumes them.
        super()._parallelize(parallelism_context)
        self._wrap_weights()

    def _wrap_weights(self) -> None:
        """Wrap W13 and W2 in the FSDP tensor that writes the block-major copy."""
        for proj in (self.w13, self.w2):
            if isinstance(proj.weight, _DistMoeBf16ShardedTensor):
                continue
            weight = proj.weight
            wrapped_weight = torch.nn.Parameter(
                _DistMoeBf16ShardedTensor(weight.data),
                requires_grad=weight.requires_grad,
            )
            spmd.assert_type_like(wrapped_weight, weight)
            proj.weight = wrapped_weight

    def _weight_operands(
        self,
    ) -> tuple[dist_moe.PreparedBf16Weight, dist_moe.PreparedBf16Weight]:
        """Return the block-major W13 and W2 that this FSDP unshard wrote."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        # Raise without FSDP: copying every expert weight on each call would cost
        # far more than the block-major GEMMs save. A DeepSeek-V3 layer at EP 4 on
        # GB300: the copy takes 1.78 ms, the gain is 42.6 us at 64 tokens per GPU.
        if not (
            isinstance(w13_E2FD, _UnshardedFSDPTensor)
            and isinstance(w2_EDF, _UnshardedFSDPTensor)
        ):
            raise RuntimeError(
                "Block-major Dist-MoE weights are built by FSDP's unshard; unshard "
                "the experts' FSDP group before the forward, or set "
                "bf16_block_major_inference=False. Got W13 "
                f"{type(w13_E2FD).__name__} and W2 {type(w2_EDF).__name__}"
            )
        return w13_E2FD.operands.prepared(), w2_EDF.operands.prepared()
