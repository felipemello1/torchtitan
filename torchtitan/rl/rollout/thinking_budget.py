# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.types import Completion, GenerateFn

if TYPE_CHECKING:
    # Type-only: importing the generator module here would pull in vLLM at import time.
    from torchtitan.rl.generator.generator import SamplingConfig


class ThinkingBudget(Configurable):
    """Caps thinking per turn: a reply still thinking after `max_thinking_tokens` gets a forced end of
    thinking and is asked for its answer, instead of being cut at `max_tokens`.

    The forced text is loss-masked, and the whole turn stays within `SamplingConfig.max_tokens`.

    Example:

        # RolloutWorker.Config(..., thinking_budget=ThinkingBudget.Config(max_thinking_tokens=1024,
        #                                                                  answer_prefix="\\boxed{"))
        call 1: 1024 thinking tokens, finish "length", thinking still open    -> force the close
        call 2: prompt + thinking + "...directly now.\\n</think>\\n\\n\\boxed{"  -> "e4}" finish "stop"
        Completion: 1024 + 28 + 3 tokens; loss_mask False on the 28 forced tokens

        call 1: 500 thinking tokens + <|im_end|>, finish "stop", still open   -> drop <|im_end|>, force the close
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Set on `RolloutWorker.Config`."""

        max_thinking_tokens: int
        """Tokens a turn may think before its end of thinking is forced; must leave room for the
        forced text and the answer within `SamplingConfig.max_tokens`."""

        force_close_on_stop: bool = True
        """A reply that samples its stop token before `think_end_token`: True drops the stop token and
        forces the close; False returns it unchanged, so the env reads its thinking text as the answer."""

        close_text: str = (
            "\n\nConsidering the limited time by the user, I have to give the solution based on "
            "the thinking directly now.\n</think>\n\n"
        )
        """Appended when thinking runs out: Qwen's thinking-budget sentence, then the end-of-thinking tag."""

        answer_prefix: str = ""
        """Appended after `close_text` so a forced answer starts in the expected format, e.g. "\\boxed{".
        Without it, a forced answer tends to keep reasoning until `max_tokens` cuts it."""

        answer_end_text: str | None = None
        """After a forced close (or the model's own, with `answer_after_close`), end the turn at the first
        token containing this text, e.g. "}" after `\\boxed{e4}`. Without it, a forced answer can keep
        talking until `max_tokens` cuts the turn."""

        answer_after_close: bool = False
        """Also stop call 1 at the model's own `think_end_token`, append "\\n\\n" + `answer_prefix`
        (loss-masked), and end the answer at `answer_end_text`. Needs `answer_end_text`."""

        end_of_turn_token: str = "<|im_end|>"
        """The tokenizer's single end-of-turn token, appended untrained when `answer_end_text` stops
        the answer."""

        think_start_token: str = "<think>"
        """The tokenizer's single start-of-thinking token."""

        think_end_token: str = "</think>"
        """The tokenizer's single end-of-thinking token."""

    def __init__(self, config: Config, *, tokenizer: HuggingFaceTokenizer) -> None:
        self._max_thinking_tokens = config.max_thinking_tokens
        self._force_close_on_stop = config.force_close_on_stop
        self._think_start_id = tokenizer.token_to_id(config.think_start_token)
        self._think_end_id = tokenizer.token_to_id(config.think_end_token)
        if self._think_start_id is None or self._think_end_id is None:
            raise ValueError(
                f"{config.think_start_token!r} and {config.think_end_token!r} must each be one token"
            )
        self._forced_ids = tokenizer.encode(
            config.close_text + config.answer_prefix, add_bos=False, add_eos=False
        )
        # Every token whose text contains `answer_end_text` ("}", "}.", "}\n", ...): vLLM stops on ids,
        # not text, because the generator runs without detokenizing.
        self._answer_end_ids = set()
        if config.answer_end_text is not None:
            self._answer_end_ids = {
                token_id
                for token_id in range(tokenizer.get_vocab_size())
                if config.answer_end_text in tokenizer.decode([token_id])
            }
            self._end_of_turn_id = tokenizer.token_to_id(config.end_of_turn_token)
            if self._end_of_turn_id is None:
                raise ValueError(f"{config.end_of_turn_token!r} must be one token")
        if config.answer_after_close and config.answer_end_text is None:
            raise ValueError(
                "answer_after_close needs answer_end_text to end the answer"
            )
        self._answer_after_close = config.answer_after_close
        # After the model's own close: the line break a forced close also ends with, then the prefix.
        self._answer_prefix_ids = tokenizer.encode(
            "\n\n" + config.answer_prefix, add_bos=False, add_eos=False
        )

    def wrap(
        self, generate_fn: GenerateFn, *, metrics_prefix: str = "thinking_budget"
    ) -> GenerateFn:
        """Return a `GenerateFn` that applies the budget around `generate_fn`. Its metrics are
        `{metrics_prefix}/forced_close_rate` and `{metrics_prefix}/stopped_while_thinking_rate`."""

        async def generate(
            prompt_token_ids: list[int],
            *,
            request_id: str,
            group_id: int,
            routing_session_id: str | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> Completion | None:
            max_thinking_tokens = self._max_thinking_tokens
            max_tokens = sampling_config.max_tokens
            num_end_of_turn = int(bool(self._answer_end_ids))
            if (
                max_thinking_tokens + len(self._forced_ids) + num_end_of_turn
                >= max_tokens
            ):
                raise ValueError(
                    f"max_thinking_tokens ({max_thinking_tokens}) + {len(self._forced_ids)} forced "
                    f"tokens + {num_end_of_turn} end-of-turn token must be below "
                    f"SamplingConfig.max_tokens ({max_tokens}), which caps the turn"
                )
            first = await generate_fn(
                prompt_token_ids,
                request_id=request_id,
                group_id=group_id,
                routing_session_id=routing_session_id,
                sampling_config=replace(
                    sampling_config,
                    max_tokens=max_thinking_tokens,
                    # stop at the model's own close, to append the answer prefix
                    stop_token_ids=(
                        (sampling_config.stop_token_ids or []) + [self._think_end_id]
                        if self._answer_after_close
                        and self._is_thinking(prompt_token_ids)
                        else sampling_config.stop_token_ids
                    ),
                ),
            )
            if first is None:
                return None
            closed_by_model = (
                self._answer_after_close
                and first.finish_reason == "stop"
                and first.token_ids[-1:] == [self._think_end_id]
                and self._is_thinking(prompt_token_ids)
            )

            # ======== Done after call 1, unless the reply must be continued ========
            # A reply that samples its stop token before </think> has not answered.
            stopped_while_thinking = (
                first.finish_reason == "stop"
                and self._is_thinking(prompt_token_ids + first.token_ids)
            )
            close_stopped_reply = stopped_while_thinking and self._force_close_on_stop
            # "length" below the budget means the context cut it: a second call would be rejected.
            cut_at_budget = (
                first.finish_reason == "length"
                and len(first.token_ids) >= max_thinking_tokens
            )
            if not (close_stopped_reply or cut_at_budget or closed_by_model):
                first.metrics += [
                    m.Metric(f"{metrics_prefix}/forced_close_rate", m.Mean(0.0)),
                    m.Metric(
                        f"{metrics_prefix}/stopped_while_thinking_rate",
                        m.Mean(float(stopped_while_thinking)),
                    ),
                ]
                return first
            if close_stopped_reply:
                # Drop the stop token so the forced close follows the thinking.
                # TODO: the dropped stop token is never trained against; watch stopped_while_thinking_rate.
                first = replace(
                    first,
                    token_ids=first.token_ids[:-1],
                    token_logprobs=first.token_logprobs[:-1],
                )

            # ======== Call 2: the answer ========
            # Still thinking (cut or stopped): force the close. Closed by the model: append the
            # answer prefix. Cut while answering: continue.
            force_closed = self._is_thinking(prompt_token_ids + first.token_ids)
            if force_closed:
                appended_ids = self._forced_ids
            elif closed_by_model:
                appended_ids = self._answer_prefix_ids
            else:
                appended_ids = []
            answer_end_ids = sorted(self._answer_end_ids) if appended_ids else []
            second = await generate_fn(
                prompt_token_ids + first.token_ids + appended_ids,
                request_id=f"{request_id}/answer",
                group_id=group_id,
                routing_session_id=routing_session_id,
                sampling_config=replace(
                    sampling_config,
                    # one token of room for the end of turn appended below
                    max_tokens=max_tokens
                    - len(first.token_ids)
                    - len(appended_ids)
                    - bool(answer_end_ids),
                    stop_token_ids=(sampling_config.stop_token_ids or [])
                    + answer_end_ids,
                ),
            )
            if second is None:
                return None
            end_ids = []
            if (
                answer_end_ids
                and second.token_ids[-1:]
                and second.token_ids[-1] in self._answer_end_ids
            ):
                end_ids = [self._end_of_turn_id]
            return Completion(
                min_policy_version=min(
                    first.min_policy_version, second.min_policy_version
                ),
                max_policy_version=max(
                    first.max_policy_version, second.max_policy_version
                ),
                request_id=request_id,
                token_ids=first.token_ids + appended_ids + second.token_ids + end_ids,
                # NaN: the forced tokens have no sampling logprob. The batcher and the loss also
                # drop non-finite logprobs, so a path that loses the mask still never trains them.
                token_logprobs=(
                    first.token_logprobs
                    + [math.nan] * len(appended_ids)
                    + second.token_logprobs
                    + [math.nan] * len(end_ids)
                ),
                loss_mask=(
                    [True] * len(first.token_ids)
                    + [False] * len(appended_ids)
                    + [True] * len(second.token_ids)
                    + [False] * len(end_ids)
                ),
                finish_reason=second.finish_reason,
                metrics=[
                    *first.metrics,
                    *second.metrics,
                    m.Metric(
                        f"{metrics_prefix}/forced_close_rate",
                        m.Mean(float(force_closed)),
                    ),
                    m.Metric(
                        f"{metrics_prefix}/stopped_while_thinking_rate",
                        m.Mean(float(stopped_while_thinking)),
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
