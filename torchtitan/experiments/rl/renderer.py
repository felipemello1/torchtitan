# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Renderer wiring for the RL controller.

The renderer (from the ``renderers`` package) converts chat messages to
token IDs before generation, and parses the generated token IDs back
into a structured ``ParsedResponse`` (text + tool calls + reasoning).
It owns model-specific chat-template logic so the rollout code can stay
template-agnostic.

This module exposes a small ``RendererConfig`` so the controller can be
declarative about which renderer to use. The renderer itself is built
on the controller process from the local HF tokenizer; no Monarch actor
involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from renderers import create_renderer, Renderer

from torchtitan.config import Configurable


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the renderer used for message <-> token conversion.

    The renderer is built from the model's HF tokenizer (loaded from
    ``hf_assets_path`` on the controller side).

    Example::

        renderer = RendererConfig(name="qwen3").build(
            model_path="./assets/Qwen3-0.6B"
        )
        token_ids = list(renderer.render_ids(
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
        ))
        # token_ids -> [151644, 872, 198, 6023, 151645, 198, 151644, 77091, 198]
    """

    name: str = "auto"
    """Renderer name passed to ``renderers.create_renderer``.

    ``"auto"`` lets ``create_renderer`` pick the renderer that matches the
    tokenizer's chat template. Pass a concrete name (e.g. ``"qwen3"``,
    ``"deepseek_v3"``, ``"gpt_oss"``) to override.
    """

    tool_parser: str | None = None
    """Tool-call parser used by ``DefaultRenderer``. Ignored by model-specific renderers."""

    reasoning_parser: str | None = None
    """Reasoning parser used by ``DefaultRenderer``. Ignored by model-specific renderers."""

    def build(
        self,
        *,
        model_path: str | None = None,
        tokenizer: Any | None = None,
    ) -> Renderer:
        """Construct the renderer.

        Exactly one of ``model_path`` and ``tokenizer`` must be provided.
        ``model_path`` is a local HF assets directory (the same one used
        for the trainer / generator weights); the tokenizer is loaded
        with ``trust_remote_code=True`` because some chat templates
        depend on tokenizer-side Python.
        """
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

        return create_renderer(
            tokenizer,
            renderer=self.name,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
        )
