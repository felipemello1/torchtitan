# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configs for the sampled-softmax experiment.

Local (1 GPU): Qwen3-0.6B with an untied lm_head (like the 30B) on local c4 shards, for accuracy A/B runs.
8xH100: Qwen3-30B-A3B, dp4 tp2 ep2, DeepEP, the baseline run's shape.

Each shape has a baseline (stock ChunkedLossWrapper) and sampled variants. Scalars
such as ``--loss.correction importance`` or ``--loss.fused_full_softmax`` can be
overridden on the command line.
"""

import dataclasses
from functools import partial

import torch.nn as nn

from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.experiments.sampled_softmax.loss import SampledSoftmaxChunkedLoss
from torchtitan.experiments.sampled_softmax.trainer import SampledSoftmaxTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS, TextProcessor
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import qwen3_0_6b, qwen3_30b_a3b
from torchtitan.trainer import Trainer

# The tokenizer baseline run 2uo7gfn9 used, and the c4 shards cached on the devgpu
# (train shards 0-2 for training, shard 4 held out for validation).
H100_TOKENIZER_PATH = (
    "/home/felipemello/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
LOCAL_C4 = (
    "/home/felipemello/.cache/huggingface/hub/datasets--allenai--c4/snapshots/"
    "1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en"
)


def _local_c4(split_files: list[str]) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={
                "data_files": [f"{LOCAL_C4}/{f}" for f in split_files]
            },
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )


def _as_experiment(config: Trainer.Config) -> SampledSoftmaxTrainer.Config:
    fields = {
        f.name: getattr(config, f.name) for f in dataclasses.fields(Trainer.Config)
    }
    return SampledSoftmaxTrainer.Config(**fields)


def _sampled_loss(config: Trainer.Config, **kwargs) -> SampledSoftmaxChunkedLoss.Config:
    assert isinstance(config.loss, ChunkedLossWrapper.Config)
    return SampledSoftmaxChunkedLoss.Config(
        num_chunks=config.loss.num_chunks,
        loss_fn=config.loss.loss_fn,
        total_steps=config.training.steps,
        **kwargs,
    )


# ---- Local: Qwen3-0.6B, 1 GPU ------------------------------------------------


def qwen3_0_6b_local_baseline(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    config = qwen3_0_6b(seq_len=seq_len)
    config.hf_assets_path = H100_TOKENIZER_PATH
    # Untie the lm_head so its gradient is as sparse as the 30B's under sampling.
    config.model.enable_weight_tying = False
    config.model.tok_embeddings.param_init = {
        "weight": partial(nn.init.normal_, std=0.02)
    }
    config.dataloader = GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=_local_c4([f"c4-train.0000{i}-of-01024.json.gz" for i in range(3)])
        ),
        seed=42,
    )
    config.optimizer = default_adamw(lr=1e-3)
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=50)
    config.training.steps = 1000
    config.training.num_tokens_per_microbatch_per_dp_rank = 8 * 4096
    # Matches the 30B baseline, which runs without CUDA graphs.
    config.training.disable_cuda_graphs = True
    config.validator = Validator.Config(
        freq=100,
        steps=24,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=_local_c4(["c4-train.00004-of-01024.json.gz"])
            ),
            repeat=True,
            shuffle=False,
        ),
    )
    return _as_experiment(config)


def qwen3_0_6b_local_sampled(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    """Twice the per-TP-rank budgets of the 30B run: the same global budget on one GPU."""
    config = qwen3_0_6b_local_baseline(seq_len)
    config.loss = _sampled_loss(
        config, schedule=[(0.57, 16384), (0.81, 24576), (0.93, 49152)]
    )
    return config


def qwen3_0_6b_local_short_baseline(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    """100 steps with the 30B run's optimizer and LR schedule (7-step full-softmax tail when sampled)."""
    config = qwen3_0_6b_local_baseline(seq_len)
    reference = qwen3_30b_a3b_deepep_baseline(seq_len)
    config.optimizer = reference.optimizer
    config.lr_scheduler = reference.lr_scheduler
    config.training.steps = reference.training.steps
    config.validator.freq = 25
    return config


def qwen3_0_6b_local_short_sampled(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    config = qwen3_0_6b_local_short_baseline(seq_len)
    config.loss = _sampled_loss(
        config, schedule=[(0.57, 16384), (0.81, 24576), (0.93, 49152)]
    )
    return config


def qwen3_0_6b_local_fused_full(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    """Kernel-only arm: full softmax every step through the fused CE (no sampling)."""
    config = qwen3_0_6b_local_baseline(seq_len)
    config.loss = _sampled_loss(config, schedule=[])
    return config


# ---- 8xH100 devgpu: Qwen3-30B-A3B, dp4 tp2 ep2, DeepEP ------------------------------


def qwen3_30b_a3b_deepep_baseline(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    """Wandb run 2uo7gfn9 (qwen3_30b_a3b_c4_ep2_tp2_dp4_deepep, 32768 tokens per DP rank)."""
    config = qwen3_30b_a3b(seq_len=seq_len)
    config.model = model_registry("30B-A3B", seq_len=seq_len, moe_comm_backend="deepep")
    config.hf_assets_path = H100_TOKENIZER_PATH
    config.lr_scheduler = LRSchedulersContainer.Config(warmup_steps=10)
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    config.parallelism.expert_parallel_degree = 2
    config.training.steps = 100
    config.training.num_tokens_per_microbatch_per_dp_rank = 32768
    config.training.disable_cuda_graphs = True
    return _as_experiment(config)


def qwen3_30b_a3b_deepep_sampled(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    config = qwen3_30b_a3b_deepep_baseline(seq_len)
    config.loss = _sampled_loss(config)
    return config


def qwen3_30b_a3b_deepep_fused_full(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    """Kernel-only arm: full softmax every step through the fused CE (no sampling)."""
    config = qwen3_30b_a3b_deepep_baseline(seq_len)
    config.loss = _sampled_loss(config, schedule=[])
    return config


def qwen3_30b_a3b_deepep_baseline_val(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    return _with_c4_validation(qwen3_30b_a3b_deepep_baseline(seq_len))


def qwen3_30b_a3b_deepep_sampled_val(
    seq_len: int | None = 4096,
) -> SampledSoftmaxTrainer.Config:
    return _with_c4_validation(qwen3_30b_a3b_deepep_sampled(seq_len))


def _with_c4_validation(
    config: SampledSoftmaxTrainer.Config,
) -> SampledSoftmaxTrainer.Config:
    """Full-vocab CE on c4 validation every 25 steps (8 steps of 32768 tokens per DP rank)."""
    config.validator = Validator.Config(
        freq=25,
        steps=8,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_validation"]),
            repeat=True,
            shuffle=False,
        ),
    )
    return config
