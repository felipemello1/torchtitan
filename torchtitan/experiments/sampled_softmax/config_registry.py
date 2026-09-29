# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configs for the sampled-softmax experiment.

Local (1 GPU): Qwen3-0.6B on local c4 shards, for accuracy A/B runs.
8xH100 devgpu: Qwen3-30B-A3B, dp4 tp2 ep2, DeepEP, the baseline run's shape.

Each shape has a baseline (stock ChunkedLossWrapper) and sampled variants. Scalars
such as ``--loss.correction importance`` or ``--loss.fused_full_softmax`` can be
overridden on the command line.
"""

import dataclasses

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.validate import Validator
from torchtitan.experiments.sampled_softmax.loss import SampledSoftmaxChunkedLoss
from torchtitan.experiments.sampled_softmax.trainer import SampledSoftmaxTrainer
from torchtitan.components.data.dataset import SingleDatasetConfig
from torchtitan.components.data.loader import GrainDataLoader
from torchtitan.components.data.packing import ConcatThenSplitPackingConfig
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.hf_datasets.text_datasets import DATASETS, TextProcessor
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3.config_registry import qwen3_0_6b, qwen3_30b_a3b
from torchtitan.trainer import Trainer

# Local runs (GB300 login pod) and the 8xH100 devgpu keep the Qwen3 tokenizer in different places;
# the H100 path is the one baseline run 2uo7gfn9 used.
LOCAL_TOKENIZER_PATH = "/home/felipemello/fp32grads/qwen3_tokenizer"
H100_TOKENIZER_PATH = "/home/felipemello/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
LOCAL_C4 = "/home/felipemello/data/c4_en"


def _local_c4(split_files: list[str]) -> SingleDatasetConfig:
    return SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={"data_files": [f"{LOCAL_C4}/{f}" for f in split_files]},
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    )


def _as_experiment(config: Trainer.Config) -> SampledSoftmaxTrainer.Config:
    fields = {f.name: getattr(config, f.name) for f in dataclasses.fields(Trainer.Config)}
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


def qwen3_0_6b_local_baseline(seq_len: int | None = 4096) -> SampledSoftmaxTrainer.Config:
    config = qwen3_0_6b(seq_len=seq_len)
    config.hf_assets_path = LOCAL_TOKENIZER_PATH
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
                dataset=_local_c4(["c4-validation.00000-of-00008.json.gz"])
            ),
            repeat=True,
            shuffle=False,
        ),
    )
    return _as_experiment(config)


def qwen3_0_6b_local_sampled(seq_len: int | None = 4096) -> SampledSoftmaxTrainer.Config:
    config = qwen3_0_6b_local_baseline(seq_len)
    config.loss = _sampled_loss(config)
    return config


# ---- 8xH100 devgpu: Qwen3-30B-A3B, dp4 tp2 ep2, DeepEP ------------------------------


def qwen3_30b_a3b_deepep_baseline(seq_len: int | None = 4096) -> SampledSoftmaxTrainer.Config:
    """The shape of wandb run 2uo7gfn9 (qwen3_30b_a3b_c4_ep2_tp2_dp4_deepep) plus validation."""
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
    config.validator = Validator.Config(
        freq=25,
        steps=8,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_validation"]),
            repeat=True,
            shuffle=False,
        ),
    )
    return _as_experiment(config)


def qwen3_30b_a3b_deepep_sampled(seq_len: int | None = 4096) -> SampledSoftmaxTrainer.Config:
    config = qwen3_30b_a3b_deepep_baseline(seq_len)
    config.loss = _sampled_loss(config)
    return config


def qwen3_30b_a3b_deepep_fused_full(seq_len: int | None = 4096) -> SampledSoftmaxTrainer.Config:
    """Kernel-only arm: full softmax every step through the fused CE (no sampling)."""
    config = qwen3_30b_a3b_deepep_baseline(seq_len)
    config.loss = _sampled_loss(config, schedule=[], fused_full_softmax=True)
    return config
