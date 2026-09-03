# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from renderers import config_from_name, create_renderer, Renderer
from renderers.configs import BaseRendererConfig

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import Configurable

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class RendererConfig(Configurable.Config):
    """Selects the `renderers` renderer that converts chat messages <-> token ids.

    Args:
        name: Renderer name in `renderers`, e.g. `"qwen3"`, `"gpt-oss"`, `"llama-3"`, or
            TorchTitan's `"muse_glimmer"`. Renderers and their options:
            https://github.com/PrimeIntellect-ai/renderers/blob/renderers-v0.1.11/docs/renderer-config.md
        options: That renderer's options, passed through unchanged. An option the
            renderer does not have raises at build time.

    Example:

        RendererConfig(name="qwen3", options={"enable_thinking": False})
        RendererConfig(name="gpt-oss", options={"reasoning_effort": "low"})
        RendererConfig(name="deepseek-v3", options={"enable_thinking": False}).build(tokenizer=tokenizer)
        # -> ValidationError: DeepSeekV3RendererConfig has no `enable_thinking`
    """

    name: str
    options: dict[str, Any] = field(default_factory=dict)

    def build(self, *, tokenizer: HuggingFaceTokenizer) -> Renderer:
        renderer_tokenizer: RendererTokenizer = RendererTokenizer(tokenizer)
        renderer_config: BaseRendererConfig = self.to_renderers_config()
        return create_renderer(tokenizer=renderer_tokenizer, config=renderer_config)

    def to_renderers_config(self) -> BaseRendererConfig:
        """Return the selected renderer's validated config.

        A process that builds the renderer with its own tokenizer takes this object
        instead of the `RendererConfig`.
        """
        if self.name in ("auto", "default"):
            raise ValueError(
                f"renderer {self.name!r} needs Hugging Face `apply_chat_template`; "
                "name a model-specific renderer."
            )
        if self.name == "muse_glimmer":
            # TODO: delete this registration when Muse Glimmer is upstreamed.
            # Rollout workers do not import recipe modules, so register it here.
            from torchtitan.experiments.rl.models.muse_glimmer import (
                renderer as muse_glimmer_renderer,
            )

            muse_glimmer_renderer.register()

        config_cls = type(config_from_name(self.name))
        # pydantic `extra="forbid"`: an option the renderer lacks raises here.
        renderer_config: BaseRendererConfig = config_cls(**self.options)
        logger.info(f"Using renderer {renderer_config!r}")
        return renderer_config


class RendererTokenizer:
    """TorchTitan's tokenizer with the interface `renderers` expects (`renderers.OffsetTokenizer`).

    Example:

        tokenizer = RendererTokenizer(HuggingFaceTokenizer(tokenizer_path="./Qwen3-0.6B"))
        tokenizer.encode("hi")                          # [6023]; never adds BOS/EOS
        tokenizer("hi", return_offsets_mapping=True)    # {"input_ids": [6023], "offset_mapping": [(0, 2)]}
        tokenizer.convert_tokens_to_ids("<|im_end|>")   # 151645
    """

    def __init__(self, tokenizer: HuggingFaceTokenizer):
        # The `tokenizers.Tokenizer` inside; it has the offsets and token -> id lookup.
        self._tokenizer_backend = tokenizer.tokenizer
        self.name_or_path = tokenizer.tokenizer_path
        self.bos_token = tokenizer.bos_token
        self.eos_token = tokenizer.eos_token
        self.bos_token_id = tokenizer.bos_id
        self.eos_token_id = tokenizer.eos_id
        # `tokenizers` returns None for unknown tokens; it has no unk id.
        self.unk_token_id = None

    def encode(
        self, text: str, add_special_tokens: bool = False, **kwargs
    ) -> list[int]:
        return self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        ).ids

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        return self._tokenizer_backend.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )

    def convert_tokens_to_ids(
        self, tokens: str | list[str]
    ) -> int | None | list[int | None]:
        if isinstance(tokens, str):
            return self._tokenizer_backend.token_to_id(tokens)
        return [self._tokenizer_backend.token_to_id(token) for token in tokens]

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **kwargs,
    ) -> dict:
        encoding = self._tokenizer_backend.encode(
            text, add_special_tokens=add_special_tokens
        )
        output = {"input_ids": encoding.ids}
        if return_offsets_mapping:
            output["offset_mapping"] = encoding.offsets
        return output
