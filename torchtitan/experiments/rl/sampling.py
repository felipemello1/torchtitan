# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sampling configuration shared by the vLLM actor and downstream RL code."""


TRAINING_VLLM_LOGPROBS_MODE = "processed_logprobs"
"""vLLM logprob mode used for behavior-policy probabilities in training.

Returns the logprob distribution after vLLM's sampling-temperature transform,
so the trainer can recover the same distribution by dividing its own logits
by the same temperature before ``log_softmax`` (see
:func:`torchtitan.experiments.rl.actors.utils.compute_logprobs`).
"""
