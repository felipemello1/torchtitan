# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import GenerateFn
from torchtitan.rl.types import Completion

if TYPE_CHECKING:
    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.rl.generator import SamplingConfig


class ThinkingBudget(Configurable):
    """Caps thinking per turn: a reply still thinking after `max_thinking_tokens` gets a forced end of
    thinking and is asked for its answer, instead of being cut at `max_tokens`.

    The forced text was not sampled by the policy, so the returned `Completion.loss_mask` marks it
    False and the loss never trains on it. The whole turn, answer included, stays within
    `SamplingConfig.max_tokens`.

    Example:

        # RolloutWorker.Config(..., thinking_budget=ThinkingBudget.Config(max_thinking_tokens=1024,
        #                                                                  answer_prefix="\\boxed{"))
        call 1: 1024 thinking tokens, finish "length", thinking still open    -> force the close
        call 2: prompt + thinking + "...directly now.\\n</think>\\n\\n\\boxed{"  -> "e4}" finish "stop"
        Completion: 1024 + 28 + 3 tokens; loss_mask False on the 28 forced tokens
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Not applied on the Verifiers path, which does not run `RolloutWorker`."""

        max_thinking_tokens: int
        """Tokens a turn may think before its end of thinking is forced; must leave room for the
        forced text and the answer within `SamplingConfig.max_tokens`."""

        close_text: str = (
            "\n\nConsidering the limited time by the user, I have to give the solution based on "
            "the thinking directly now.\n</think>\n\n"
        )
        """Appended when thinking runs out: Qwen's thinking-budget sentence, then the end-of-thinking tag."""

        answer_prefix: str = ""
        """Appended after `close_text` so the answer starts in the expected format, e.g. "\\boxed{".
        Without it, Qwen3.5-4B kept reasoning in its answer (chess: 89% of answers cut at 256 tokens)."""

        think_start_token: str = "<think>"
        """The tokenizer's single start-of-thinking token."""

        think_end_token: str = "</think>"
        """The tokenizer's single end-of-thinking token. A reply is still thinking when the last of
        the two tokens in its prompt + reply is `think_start_token`."""

    def __init__(self, config: Config, *, tokenizer: HuggingFaceTokenizer) -> None:
        self._max_thinking_tokens = config.max_thinking_tokens
        self._think_start_id = tokenizer.token_to_id(config.think_start_token)
        self._think_end_id = tokenizer.token_to_id(config.think_end_token)
        if self._think_start_id is None or self._think_end_id is None:
            raise ValueError(
                f"{config.think_start_token!r} and {config.think_end_token!r} must each be one token"
            )
        self._forced_ids = tokenizer.encode(
            config.close_text + config.answer_prefix, add_bos=False, add_eos=False
        )

    def wrap(self, generate_fn: GenerateFn) -> GenerateFn:
        """Return a `GenerateFn` that applies the budget around `generate_fn`."""

        async def generate(
            prompt_token_ids: list[int],
            *,
            request_id: str,
            group_id: int,
            routing_session_id: str | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> Completion | None:
            max_tokens = sampling_config.max_tokens
            if self._max_thinking_tokens + len(self._forced_ids) >= max_tokens:
                raise ValueError(
                    f"max_thinking_tokens ({self._max_thinking_tokens}) + {len(self._forced_ids)} forced "
                    f"tokens must be below SamplingConfig.max_tokens ({max_tokens}), which caps the turn"
                )
            first = await generate_fn(
                prompt_token_ids,
                request_id=request_id,
                group_id=group_id,
                routing_session_id=routing_session_id,
                sampling_config=replace(
                    sampling_config, max_tokens=self._max_thinking_tokens
                ),
            )
            if first is None or first.finish_reason != "length":
                if first is not None:
                    first.metrics.append(
                        m.Metric("thinking_budget/forced_close_rate", m.Mean(0.0))
                    )
                return first

            # Cut while thinking: force the close. Cut while answering: let the answer continue.
            if self._is_thinking(prompt_token_ids + first.token_ids):
                forced_ids = self._forced_ids
            else:
                forced_ids = []
            second = await generate_fn(
                prompt_token_ids + first.token_ids + forced_ids,
                request_id=f"{request_id}/answer",
                group_id=group_id,
                routing_session_id=routing_session_id,
                sampling_config=replace(
                    sampling_config,
                    max_tokens=max_tokens - len(first.token_ids) - len(forced_ids),
                ),
            )
            if second is None:
                return None
            return Completion(
                min_policy_version=min(
                    first.min_policy_version, second.min_policy_version
                ),
                max_policy_version=max(
                    first.max_policy_version, second.max_policy_version
                ),
                request_id=request_id,
                token_ids=first.token_ids + forced_ids + second.token_ids,
                # NaN: the forced tokens have no sampling logprob. The batcher and the loss also
                # drop non-finite logprobs, so a path that loses the mask still never trains them.
                token_logprobs=(
                    first.token_logprobs
                    + [math.nan] * len(forced_ids)
                    + second.token_logprobs
                ),
                loss_mask=(
                    [True] * len(first.token_ids)
                    + [False] * len(forced_ids)
                    + [True] * len(second.token_ids)
                ),
                routed_expert_ids=_join_routed_expert_ids(first, second),
                topk_token_ids=_join_topk_rows(
                    first.topk_token_ids, len(forced_ids), second.topk_token_ids
                ),
                topk_logprobs=_join_topk_rows(
                    first.topk_logprobs, len(forced_ids), second.topk_logprobs
                ),
                finish_reason=second.finish_reason,
                metrics=[
                    *first.metrics,
                    *second.metrics,
                    m.Metric(
                        "thinking_budget/forced_close_rate",
                        m.Mean(float(bool(forced_ids))),
                    ),
                ],
            )

        return generate

    def _is_thinking(self, token_ids: list[int]) -> bool:
        """Whether the last thinking delimiter in `token_ids` opens thinking.

        Example:

            [..., <think>, ...]               -> True   (the generation prompt opened thinking)
            [..., <think>, ..., </think>, ...] -> False  (thinking off, or the model closed it)
            [...]  (no delimiter)             -> False  (a model that never thinks)
        """
        for token_id in reversed(token_ids):
            if token_id == self._think_end_id:
                return False
            if token_id == self._think_start_id:
                return True
        return False


def _join_routed_expert_ids(
    first: Completion, second: Completion
) -> torch.Tensor | None:
    """Rows of the merged turn: the first call's, then the second call's from where the first
    stops (its last token, the forced tokens and its own completion ran forward only there).

    Example:
        # prompt [P0, P1], first [A0, A1], forced [F0], second [B0]
        # first rows: P0 P1 A0 (A1 never ran forward); second rows: P0 P1 A0 A1 F0
        # -> P0 P1 A0 from first, A1 F0 from second
    """
    if first.routed_expert_ids is None or second.routed_expert_ids is None:
        return None
    num_first_rows = len(first.routed_expert_ids)
    return torch.cat(
        [first.routed_expert_ids, second.routed_expert_ids[num_first_rows:]]
    )


def _join_topk_rows(
    first: torch.Tensor | None, num_forced: int, second: torch.Tensor | None
) -> torch.Tensor | None:
    """The first call's top-k rows, zero rows for the forced tokens (the loss skips them), then
    the second call's."""
    if first is None or second is None:
        return None
    return torch.cat([first, first.new_zeros(num_forced, *first.shape[1:]), second])
