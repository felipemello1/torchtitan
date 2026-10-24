# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Streamed microbatches give the optimizer the same gradients as one forward_backward call.

The step has a power-of-two loss-token count N, so dividing by N is exact in any float format, and
both orders (divide each loss, or divide the summed gradients) give bitwise equal gradients.

    pytest tests/rl/unit_tests/gpu/test_stream_microbatches.py
"""

import dataclasses
import os
from functools import partial

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.elastic.utils.distributed import get_free_port

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.utils import get_local_tensor
from torchtitan.models.common.moe.moe import collect_routing_mismatch_metrics
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.rollout.types import RolloutTurnID
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.losses.dppo import DPPOLoss
from torchtitan.rl.trainer.losses.grpo import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.rl.trainer.training_sample_builder import (
    TrainingSample,
    TrainingSampleGroup,
)
from torchtitan_recipes.rl.alphabet_sort import (
    _qwen3_5_rl_model_configs,
    rl_grpo_qwen3_0_6b_varlen_no_compile,
)
from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_NUM_TOKENS_PER_MICROBATCH = 64

# world size, trainer parallelism
_LAYOUTS = {
    "one_gpu": (1, ParallelismConfig()),
    "fsdp": (2, ParallelismConfig(data_parallel_shard_degree=2)),
    "hsdp": (2, ParallelismConfig(data_parallel_replicate_degree=2)),
    "tp": (
        2,
        ParallelismConfig(tensor_parallel_degree=2, enable_sequence_parallel=True),
    ),
    "pp": (
        2,
        ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=2,
        ),
    ),
}


# Qwen3.5 debug MoE (GatedDeltaNet + attention, 4 MoE layers, top-2 of 8 experts) with router
# replay and DPPOLoss's symmetric trust region, as the 35B-A3B run trains (one GPU, no EP).
_QWEN3_5_MOE_DPPO_ROUTER_REPLAY = "qwen3_5_moe_dppo_router_replay"
_QWEN3_5_MOE_NUM_LAYERS, _QWEN3_5_MOE_NUM_EXPERTS, _QWEN3_5_MOE_TOP_K = 4, 8, 2
_LAYOUTS[_QWEN3_5_MOE_DPPO_ROUTER_REPLAY] = (1, ParallelismConfig())


def _trainer_config(parallelism: ParallelismConfig, layout: str) -> Trainer.Config:
    if layout == _QWEN3_5_MOE_DPPO_ROUTER_REPLAY:
        trainer_model, _ = _qwen3_5_rl_model_configs(
            "debugmodel_moe",
            seq_len=_NUM_TOKENS_PER_MICROBATCH,
            trainer_parallelism=parallelism,
            generator_parallelism=InferenceParallelismConfig(),
        )
        loss_fn = DPPOLoss.Config(
            symmetric=True, global_vocab_size=trainer_model.vocab_size
        )
    else:
        trainer_model, _ = build_rl_model_configs(
            build_model_config,
            "debugmodel",
            trainer_parallelism=parallelism,
            generator_parallelism=InferenceParallelismConfig(),
            seq_len=_NUM_TOKENS_PER_MICROBATCH,
            attn_backend="varlen",
        )
        loss_fn = GRPOLoss.Config(global_vocab_size=trainer_model.vocab_size)
    if parallelism.pipeline_parallel_degree > 1:
        # Tied embeddings cannot be split across pipeline stages. Untied, the embedding needs its
        # own init: the tied output layer initialized it.
        trainer_model.enable_weight_tying = False
        trainer_model.tok_embeddings.param_init = {
            "weight": partial(torch.nn.init.normal_, std=0.02)
        }
    config = rl_grpo_qwen3_0_6b_varlen_no_compile(seq_len=_NUM_TOKENS_PER_MICROBATCH)
    return dataclasses.replace(
        config.trainer,
        model=trainer_model,
        parallelism=parallelism,
        training=dataclasses.replace(
            config.trainer.training,
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
        ),
        loss=ChunkedLossWrapper.Config(num_chunks=2, loss_fn=loss_fn),
        checkpointer=None,
    )


def _training_sample_group(
    rng: np.random.Generator, *, router_replay: bool = False
) -> TrainingSampleGroup:
    """24 samples of 8 to 60 tokens with 512 loss tokens in total. With `router_replay`, each
    input token carries top-k expert ids per MoE layer, as the generator returns them."""
    samples = []
    for rollout_id, num_tokens in enumerate(rng.integers(8, 61, size=24)):
        routed_expert_ids = None
        if router_replay:
            routed_expert_ids = np.stack(
                [
                    rng.permutation(_QWEN3_5_MOE_NUM_EXPERTS)[:_QWEN3_5_MOE_TOP_K]
                    for _ in range((num_tokens - 1) * _QWEN3_5_MOE_NUM_LAYERS)
                ]
            ).astype(np.uint8)
            routed_expert_ids = routed_expert_ids.reshape(
                num_tokens - 1, _QWEN3_5_MOE_NUM_LAYERS, _QWEN3_5_MOE_TOP_K
            )
        samples.append(
            TrainingSample(
                min_policy_version=0,
                max_policy_version=0,
                rollout_id=RolloutTurnID(group_id=0, rollout_id=rollout_id, turn_id=0),
                token_ids=rng.integers(0, 2048, size=num_tokens),
                loss_mask=np.arange(num_tokens) >= num_tokens // 3,
                logprobs=-rng.uniform(5.0, 9.0, size=num_tokens).astype(np.float32),
                advantage=np.full(num_tokens, rng.choice([-1.0, 1.0]), np.float32),
                routed_expert_ids=routed_expert_ids,
            )
        )
    # Drop loss tokens until the step has 512.
    num_extra_loss_tokens = (
        sum(int(np.count_nonzero(sample.loss_mask[1:])) for sample in samples) - 512
    )
    for sample in samples:
        while num_extra_loss_tokens > 0 and np.count_nonzero(sample.loss_mask[1:]) > 1:
            sample.loss_mask[np.flatnonzero(sample.loss_mask)[0]] = False
            num_extra_loss_tokens -= 1
    return TrainingSampleGroup(group_id=0, training_samples=samples, metrics=[])


def _run_streamed_vs_one_call(
    rank: int, world_size: int, port: int, layout: str, output_dir: str
) -> None:
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    parallelism = _LAYOUTS[layout][1]
    router_replay = layout == _QWEN3_5_MOE_DPPO_ROUTER_REPLAY
    trainer = Trainer(
        _trainer_config(parallelism, layout),
        max_num_documents=None,
        output_dir=output_dir,
    )
    engine = trainer.engine
    batch, _ = (
        Batcher.Config()
        .build(
            num_tokens_per_microbatch_per_dp_rank=_NUM_TOKENS_PER_MICROBATCH,
            max_context_length=_NUM_TOKENS_PER_MICROBATCH,
            num_prompts_per_train_step=1,
            dp_degree=trainer.dp_size,
            num_pp_microbatches=parallelism.num_pp_microbatches,
            pad_id=0,
            temperature=1.0,
            replay_routed_experts=router_replay,
        )
        .add_training_samples(
            training_sample_group=_training_sample_group(
                np.random.default_rng(0), router_replay=router_replay
            )
        )
    )
    assert batch is not None
    assert batch.global_loss_token_counts.tolist() == [512]
    num_groups = len(batch.microbatch_groups)
    assert num_groups >= 2
    column = [
        [row[trainer.dp_rank] for row in group] for group in batch.microbatch_groups
    ]
    first, second = column[: num_groups // 2], column[num_groups // 2 :]
    num_first_loss_tokens = sum(
        microbatch.loss_token_counts
        for group in batch.microbatch_groups[: num_groups // 2]
        for row in group
        for microbatch in row
    )

    # Record the gradients the optimizer would clip, without updating the weights.
    optimizer_gradients = []
    engine.optim.step = lambda loss, *, current_step: optimizer_gradients.append(
        [get_local_tensor(p.grad).clone() for p in engine.optim.parameters]
    )

    # 1. Today: one call; each loss is divided by the step's N loss tokens.
    engine.forward_backward(
        microbatch_groups=column,
        global_loss_token_counts=batch.global_loss_token_counts,
        global_routing_token_counts=batch.global_routing_token_counts,
    )
    engine.optim_step()
    # 2. Streamed: two calls of summed losses; the optimizer step divides by N.
    engine.forward_backward(
        microbatch_groups=first,
        global_loss_token_counts=None,
        global_routing_token_counts=None,
        ends_step=False,
    )
    engine.forward_backward(
        microbatch_groups=second,
        global_loss_token_counts=None,
        global_routing_token_counts=None,
    )
    engine.optim_step(global_loss_token_counts=batch.global_loss_token_counts)
    # 3. Each call divided by its own count: the weighting streaming must not fall back to.
    engine.forward_backward(
        microbatch_groups=first,
        global_loss_token_counts=num_first_loss_tokens,
        global_routing_token_counts=batch.global_routing_token_counts,
        ends_step=False,
    )
    engine.forward_backward(
        microbatch_groups=second,
        global_loss_token_counts=batch.global_loss_token_counts - num_first_loss_tokens,
        global_routing_token_counts=batch.global_routing_token_counts,
    )
    engine.optim_step()

    if router_replay:
        # Every MoE layer routed the replayed expert ids, in all three runs.
        mismatch = collect_routing_mismatch_metrics(
            engine.model_parts, engine.parallelism_context
        )
        assert "moe_routing/set_mismatch_frac" in mismatch
    one_call, streamed, divided_per_call = optimizer_gradients
    for expected, actual in zip(one_call, streamed, strict=True):
        assert torch.equal(actual, expected)
    # Run 3 is ~90% off here (relative norm of the difference).
    difference = sum(
        (expected - actual).norm() ** 2
        for expected, actual in zip(one_call, divided_per_call, strict=True)
    )
    assert difference.sqrt() > 0.1 * sum(grad.norm() ** 2 for grad in one_call).sqrt()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(layout, marks=[pytest.mark.multi_gpu] if world_size > 1 else [])
        for layout, (world_size, _) in _LAYOUTS.items()
    ],
)
def test_streamed_microbatches_match_one_forward_backward(
    layout: str, tmp_path
) -> None:
    world_size = _LAYOUTS[layout][0]
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    mp.spawn(
        _run_streamed_vs_one_call,
        args=(world_size, get_free_port(), layout, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
