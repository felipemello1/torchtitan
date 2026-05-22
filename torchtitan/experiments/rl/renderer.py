# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Renderer wiring for the RL controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from renderers import create_renderer, Renderer

from torchtitan.config import Configurable


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the renderer used for message <-> token conversion.

    Example::

        renderer = RendererConfig(name="qwen3").build(model_path="./Qwen3-0.6B")
        prompt_ids = list(renderer.render_ids(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        ))
        stop_token_ids = renderer.get_stop_token_ids()
    """

    name: str = "auto"
    """Renderer name passed to `renderers.create_renderer`."""

    tool_parser: str | None = None
    """Tool-call parser used by `DefaultRenderer`."""

    reasoning_parser: str | None = None
    """Reasoning parser used by `DefaultRenderer`."""

    enable_thinking: bool | None = None
    """Qwen3 thinking-mode override. `None` keeps the renderer default."""

    preserve_all_thinking: bool = False
    """Forward historical assistant reasoning back into future prompts."""

    preserve_thinking_between_tool_calls: bool = False
    """Keep assistant reasoning during active tool-call loops."""

    def build(
        self,
        *,
        model_path: str | None = None,
        tokenizer: Any | None = None,
    ) -> Renderer:
        """Construct the renderer from either a model path or tokenizer."""
        if (tokenizer is None) == (model_path is None):
            raise ValueError(
                "RendererConfig.build requires exactly one of "
                "`model_path` or `tokenizer`"
            )

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        preserve_kwargs = {
            "preserve_all_thinking": self.preserve_all_thinking,
            "preserve_thinking_between_tool_calls": (
                self.preserve_thinking_between_tool_calls
            ),
        }

        if self.enable_thinking is not None:
            if self.name != "qwen3":
                raise ValueError(
                    "RendererConfig.enable_thinking is only supported when "
                    "`name='qwen3'`; use the renderer's native config for "
                    f"{self.name!r}"
                )
            from renderers.qwen3 import Qwen3Renderer

            return Qwen3Renderer(
                tokenizer,
                enable_thinking=self.enable_thinking,
                **preserve_kwargs,
            )

        return create_renderer(
            tokenizer,
            renderer=self.name,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            **preserve_kwargs,
        )
