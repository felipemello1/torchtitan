# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.observability import structured_logger as sl


@sl.log_trace_span("compute_logprobs")
def compute_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs from logits.

    Returns logprobs for positions 1..N (the predicted tokens).
    Output shape is ``[batch, seq_len - 1]``.
    """
    from torch.distributed.tensor import DTensor

    # Config-based TP returns logits as a Replicate DTensor. Downstream RL
    # code (gather with plain-tensor indices, slicing per-sample) expects a
    # plain tensor - materialize once here.
    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        logits = logits.to_local()
    shift_logits = logits[:, :-1, :].float()
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    return logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)


def direct_rdma_weight_sync_enabled() -> bool:
    value = os.environ.get("TORCHTITAN_RL_DIRECT_RDMA", "1").lower()
    if value in {"0", "false", "no", "off"}:
        return False
    if value not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "TORCHTITAN_RL_DIRECT_RDMA must be one of 0/1, false/true, no/yes, or off/on"
        )

    from monarch.rdma import is_rdma_available

    return is_rdma_available()


@sl.log_trace_span("extract_response_logprobs")
def extract_response_logprobs(
    packed_logprobs: torch.Tensor,
    seq_lens: list[int],
    prompt_lens: list[int],
    response_lens: list[int],
) -> list[torch.Tensor]:
    """Extract per-sample response logprobs from packed logprobs."""
    seq_start = 0
    result = []
    for i in range(len(seq_lens)):
        # Logprobs are shifted: position j holds logprob of token j+1,
        # so response start (seq_start + prompt_len) maps to index
        # (seq_start + prompt_len - 1) in the logprobs tensor.
        s = seq_start + prompt_lens[i] - 1
        e = s + response_lens[i]
        result.append(packed_logprobs[0, s:e])
        seq_start += seq_lens[i]
    return result


@dataclass(frozen=True, slots=True)
class PackedPolicyLossInputs:
    """Policy-loss side tensors aligned to ``token_ids[:, 1:]``.

    ``policy_logprobs`` is intentionally absent: chunked CE supplies it one
    sequence chunk at a time to the reducer.
    """

    generator_logprobs: torch.Tensor
    advantages: torch.Tensor
    loss_mask: torch.Tensor
    loss_weights: torch.Tensor


def build_packed_policy_loss_inputs(
    *,
    num_shift_tokens: int,
    seq_lens: list[int],
    prompt_lens: list[int],
    response_lens: list[int],
    generator_logprobs: list[list[float]],
    advantages: torch.Tensor,
    num_global_valid_tokens: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> PackedPolicyLossInputs:
    """Build packed side tensors aligned to next-token prediction positions."""
    num_samples = len(response_lens)
    for name, values in (
        ("seq_lens", seq_lens),
        ("prompt_lens", prompt_lens),
        ("generator_logprobs", generator_logprobs),
    ):
        if len(values) != num_samples:
            raise ValueError(
                f"{name} must have {num_samples} entries, got {len(values)}"
            )
    if advantages.numel() != num_samples:
        raise ValueError(
            f"advantages must have {num_samples} elements, got {advantages.numel()}"
        )
    if sum(seq_lens) - 1 != num_shift_tokens:
        raise ValueError(
            f"num_shift_tokens must be sum(seq_lens) - 1, got {num_shift_tokens} "
            f"for seq_lens={seq_lens}"
        )

    shape = (1, num_shift_tokens)
    generator = torch.zeros(shape, device=device, dtype=dtype)
    expanded_advantages = torch.zeros(shape, device=device, dtype=dtype)
    loss_mask = torch.zeros(shape, device=device, dtype=torch.float32)

    advantages = advantages.to(device=device, dtype=dtype).view(num_samples)
    seq_start = 0
    for idx, (seq_len, prompt_len, response_len) in enumerate(
        zip(seq_lens, prompt_lens, response_lens, strict=True)
    ):
        if prompt_len < 1:
            raise ValueError(f"prompt_lens[{idx}] must be >= 1, got {prompt_len}")
        if response_len < 0:
            raise ValueError(f"response_lens[{idx}] must be >= 0, got {response_len}")
        if prompt_len + response_len > seq_len:
            raise ValueError(
                f"prompt_lens[{idx}] + response_lens[{idx}] exceeds seq_lens[{idx}]"
            )
        if len(generator_logprobs[idx]) != response_len:
            raise ValueError(
                f"generator_logprobs[{idx}] has {len(generator_logprobs[idx])} "
                f"tokens but response_lens[{idx}] is {response_len}"
            )

        start = seq_start + prompt_len - 1
        end = start + response_len
        if end > num_shift_tokens:
            raise ValueError(
                f"Response {idx} maps to shifted slice [{start}, {end}), beyond "
                f"num_shift_tokens={num_shift_tokens}"
            )
        if response_len > 0:
            generator[:, start:end] = torch.as_tensor(
                generator_logprobs[idx],
                device=device,
                dtype=dtype,
            )
            expanded_advantages[:, start:end] = advantages[idx]
            loss_mask[:, start:end] = 1.0

        seq_start += seq_len

    loss_weights = loss_mask / num_global_valid_tokens.to(
        device=device,
        dtype=torch.float32,
    ).clamp(min=1.0)
    return PackedPolicyLossInputs(
        generator_logprobs=generator,
        advantages=expanded_advantages,
        loss_mask=loss_mask,
        loss_weights=loss_weights,
    )


@dataclass(frozen=True, slots=True)
class PartialLogprobDrift:
    """Per-rank generator-vs-trainer logprob drift awaiting reduction across the loss-mesh.

    Args:
        logprob_diff_mean: Scalar tensor; To be sum-reduced.
        logprob_diff_max: Scalar tensor; To be max-reduced.
        ratio_tokens_different: Scalar tensor; To be sum-reduced.
    """

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor


@torch.no_grad()
@sl.log_trace_span("verify_logprob_identity")
def verify_logprob_identity(
    generator_token_logprobs: list[list[float]],
    trainer_token_logprobs: list[torch.Tensor],
    *,
    num_global_valid_tokens: torch.Tensor,
    device: torch.device,
) -> PartialLogprobDrift:
    """Compute per-rank drift between generator and trainer logprobs.

    Args:
        generator_token_logprobs (list[list[float]]): generator-side per-token logprobs, shaped
            `[num_episodes_local][response_len_i]`.
        trainer_token_logprobs (list[torch.Tensor]): Trainer-side per-token logprobs, one
            GPU tensor per episode, each of shape `[response_len_i]`.
        num_global_valid_tokens (torch.Tensor): Scalar tensor holding global token count
             across DP ranks. Used to normalize the output metrics.
        device: Device to use for tensor allocation, so metrics are ready for
            reduction across loss_mesh.

    Returns:
        PartialLogprobDrift.
    """
    # Each tensor has a different number of tokens, so we flatten them.
    generator_flat = torch.as_tensor(
        [v for sample in generator_token_logprobs for v in sample],
        dtype=torch.float32,
        device=device,
    )
    trainer_flat = torch.cat(trainer_token_logprobs).to(
        device=device, dtype=torch.float32
    )

    if generator_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return PartialLogprobDrift(zero, zero, zero)

    # 1e-6 threshold ignores bf16-quantization-level diffs
    diff = trainer_flat - generator_flat
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / num_global_valid_tokens,
        logprob_diff_max=diff.abs().max(),
        ratio_tokens_different=(diff.abs() > 1e-6).sum() / num_global_valid_tokens,
    )
