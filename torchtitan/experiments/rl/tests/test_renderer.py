# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ``RendererConfig`` against the pinned Qwen3-0.6B tokenizer.

The renderer ships with the ``renderers`` package; this test just confirms
the small ``RendererConfig`` wrapper builds it correctly and the round-trip
(messages -> tokens -> ParsedResponse) works on a real chat template.
"""

from __future__ import annotations

import pathlib

import pytest

from torchtitan.experiments.rl.renderer import RendererConfig


# Anchor the assets path to this file so the test passes regardless of
# pytest's working directory (worktrees don't carry the example_checkpoint
# tree, but the main repo does — and editable installs make the same
# import resolve back to it).
QWEN3_0_6B_ASSETS = (
    pathlib.Path(__file__).resolve().parents[1] / "example_checkpoint" / "Qwen3-0.6B"
)


def _have_assets() -> bool:
    return (QWEN3_0_6B_ASSETS / "tokenizer.json").exists()


needs_assets = pytest.mark.skipif(
    not _have_assets(),
    reason=f"Pinned tokenizer not at {QWEN3_0_6B_ASSETS}",
)


@needs_assets
def test_renderer_renders_chat_prefix():
    """``render_ids`` produces a non-empty token sequence containing the chat tags."""
    renderer = RendererConfig(name="auto").build(model_path=str(QWEN3_0_6B_ASSETS))
    token_ids = list(
        renderer.render_ids(
            [{"role": "user", "content": "hi"}], add_generation_prompt=True
        )
    )
    # Qwen3 wraps each turn in <|im_start|>role\n...<|im_end|> and finishes the
    # prompt with <|im_start|>assistant\n. The exact token IDs depend on the
    # tokenizer, but the start/end markers must round-trip via the stop tokens.
    assert token_ids, "renderer produced no tokens"
    assert token_ids.count(renderer.get_stop_token_ids()[0]) >= 1


@needs_assets
def test_renderer_parse_response_extracts_content():
    """Plain assistant text round-trips through ``parse_response``."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(QWEN3_0_6B_ASSETS), trust_remote_code=True
    )
    renderer = RendererConfig(name="auto").build(tokenizer=tokenizer)
    response_ids = tokenizer.encode("4", add_special_tokens=False)
    parsed = renderer.parse_response(response_ids)
    assert "4" in parsed.content


@needs_assets
def test_renderer_qwen3_can_disable_thinking():
    """Qwen3 no-thinking mode emits the empty think block expected by Qwen."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(QWEN3_0_6B_ASSETS), trust_remote_code=True
    )
    renderer = RendererConfig(name="qwen3", enable_thinking=False).build(
        tokenizer=tokenizer
    )
    token_ids = list(
        renderer.render_ids(
            [{"role": "user", "content": "answer briefly"}],
            add_generation_prompt=True,
        )
    )
    rendered = tokenizer.decode(token_ids)
    assert "<think>\n\n</think>" in rendered


def test_renderer_qwen3_thinking_override_uses_native_renderer(monkeypatch):
    """The Qwen3 thinking knob is wired without requiring tokenizer assets."""
    import renderers.qwen3 as qwen3

    class FakeQwen3Renderer:
        __slots__ = (
            "enable_thinking",
            "preserve_all_thinking",
            "preserve_thinking_between_tool_calls",
            "tokenizer",
        )

        def __init__(
            self,
            tokenizer,
            *,
            enable_thinking: bool,
            preserve_all_thinking: bool,
            preserve_thinking_between_tool_calls: bool,
        ) -> None:
            self.tokenizer = tokenizer
            self.enable_thinking = enable_thinking
            self.preserve_all_thinking = preserve_all_thinking
            self.preserve_thinking_between_tool_calls = (
                preserve_thinking_between_tool_calls
            )

    monkeypatch.setattr(qwen3, "Qwen3Renderer", FakeQwen3Renderer)

    tokenizer = object()
    renderer = RendererConfig(
        name="qwen3",
        enable_thinking=False,
        preserve_all_thinking=True,
        preserve_thinking_between_tool_calls=True,
    ).build(tokenizer=tokenizer)

    assert isinstance(renderer, FakeQwen3Renderer)
    assert renderer.tokenizer is tokenizer
    assert renderer.enable_thinking is False
    assert renderer.preserve_all_thinking is True
    assert renderer.preserve_thinking_between_tool_calls is True


@needs_assets
def test_renderer_config_requires_exactly_one_of_path_or_tokenizer():
    cfg = RendererConfig(name="auto")
    with pytest.raises(ValueError, match="exactly one of"):
        cfg.build()
    with pytest.raises(ValueError, match="exactly one of"):
        cfg.build(model_path="x", tokenizer=object())


def test_renderer_thinking_override_requires_qwen3_renderer():
    with pytest.raises(ValueError, match="enable_thinking"):
        RendererConfig(name="auto", enable_thinking=False).build(tokenizer=object())
