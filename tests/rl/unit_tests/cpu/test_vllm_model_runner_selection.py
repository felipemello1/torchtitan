# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest

from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.model.gdn_backend import TorchTitanGDNAttentionMetadataBuilder
from torchtitan.rl.model.vllm_worker import use_v2_model_runner
from vllm.v1.attention.backend import AttentionCGSupport


@pytest.mark.parametrize(
    ("parallelism", "batch_invariant", "expected"),
    [
        (InferenceParallelismConfig(), False, True),
        (InferenceParallelismConfig(), True, False),
        (InferenceParallelismConfig(tensor_parallel_degree=2), False, False),
        (InferenceParallelismConfig(data_parallel_degree=2), False, False),
        (
            InferenceParallelismConfig(
                tensor_parallel_degree=2, enable_sequence_parallel=True
            ),
            False,
            False,
        ),
    ],
)
def test_use_v2_model_runner(parallelism, batch_invariant, expected):
    assert use_v2_model_runner(parallelism, batch_invariant=batch_invariant) is expected


@pytest.mark.parametrize(
    ("use_v2", "expected"),
    [
        (False, AttentionCGSupport.ALWAYS),
        (True, AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE),
    ],
)
def test_gdn_full_graphs_are_decode_only_on_v2_runner(use_v2, expected):
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, use_ubatching=False, enable_dbo=False
        ),
        use_v2_model_runner=use_v2,
    )
    support = TorchTitanGDNAttentionMetadataBuilder.get_cudagraph_support(
        vllm_config, kv_cache_spec=None
    )
    assert support == expected
